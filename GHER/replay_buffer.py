import threading
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions):
        """
            buffer_shapes (dict of ints): the shape fo r all buffers that are used in the replay buffer
                buffer_shapes = {'o': (51, 10), 'u': (50, 4), 'g': (50, 3), 'info_is_success': (50, 1), 'ag': (51, 3)}
        """
        self.buffer_shapes = buffer_shapes            # {'o': (51, 10), 'u': (50, 4), 'g': (50, 3), 'info_is_success': (50, 1), 'ag': (51, 3)}
        self.size = size_in_transitions // T          # Divide by T to get the maximum number of cycles stored in the buffer 1E6/50=20000
        self.T = T                                    # Number of samples per cycle
        self.sample_transitions = sample_transitions  # a function sampled from the experience pool 

        # Each key corresponds to an empty Numpy matrix, the keys are 'o', 'ag', 'g', 'u'. The shape of val is [capacity in cycles* (T/T+1 * each dimension) )]
        # Specifically: o (20000, 51, 10)  u (20000, 50, 4)  g (20000, 50, 3)  info_is_success (20000, 50, 1) ag (20000, 51, 3)
        self.buffers = {key: np.empty([self.size, *shape]) for key, shape in buffer_shapes.items()}

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0

        # Thread lock. Multiple workers need to operate on the same experience pool, so you need to lock when accessing the sample
        self.lock = threading.Lock()

    @property
    def full(self):
        """
            Return to the experience pool is full
        """
        with self.lock:
            return self.current_size == self.size

    def sample(self, batch_size):
        """
            Sampling. This function calls the sampling function defined in her.py and recalculates the reward based on the virtual goal.
             This function is called by the sample_batch function in ddpg.py.
            Returns a dict {key: array(batch_size x shapes[key])}
        """

        # buffers Part of the data that has been populated from the experience pool
        buffers = {}
        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        # O_2, ag_2 refers to next_obs and next_ag. In the cycle, when T=0, o_2 and ag_2 should correspond to the value of T=1.
        # Therefore, after interception, o_2 and ag_2 are o, ag corresponds to the value of the next state on the sequence number
        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        # Recalculate rewards based on the method of generating virtual goals defined in her
        # buffers Store all data in the current experience pool as input. batch_size
        transitions = self.sample_transitions(buffers, batch_size)  # self.batch_size=256

        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key

        return transitions

    def store_episode(self, episode_batch, verbose=False):
        """
            episode_batch: array(batch_size x (T or T+1) x dim_key)
            Episode_batch is a dictionary, keyh and val.shape are
                         o (2, 51, 10), u (2, 50, 4), g (2, 50, 3), ag (2, 51, 3), info_is_success (2, 50, 1)
        """

        # Extract the first dimension length of each key, after execution [2, 2, 2, 2, 2]
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]

        assert np.all(np.array(batch_sizes) == batch_sizes[0])   # np.all is used to compare the corresponding position elements of two arrays.
        batch_size = batch_sizes[0]    # Rollout represents the number of workers

        with self.lock:
            idxs = self._get_storage_idx(batch_size)   # Idxs is a list with a length equal to batch_size

            # load inputs into buffers
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]  # Store the corresponding element at the corresponding key

            self.n_transitions_stored += batch_size * self.T  # Record the number of stored elements (by transition)

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    def _get_storage_idx(self, inc=None):
        """
            Use the principle of random replacement after the experience pool is full, instead of using the principle of removing the oldest sample
             Inc represents the number of elements to be stored.
             Returns a List indicating where these elements should exist in the experience pool (serial number)
             When passing the parameter batch_size=inc, it means the number of workers used in the rollout. When storing, the size of the array to be saved is inc*self.T*ndim
        """
        inc = inc or 1   # If inc=None, it becomes 1; otherwise it does not change
        assert inc <= self.size, "Batch committed to replay is too large!"   # Self.size represents the experience pool capacity
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:                         # If the experience pool is not full after storage
            idx = np.arange(self.current_size, self.current_size+inc)  # List after arange, representing the serial number of the area to be saved
        elif self.current_size < self.size:                            # If the experience pool is full after storage, it will be full first, then randomly selected
            overflow = inc - (self.size - self.current_size)           # Number of samples overflowed
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)

        # update replay size
        self.current_size = min(self.size, self.current_size+inc)

        if inc == 1:
            idx = idx[0]
        return idx

