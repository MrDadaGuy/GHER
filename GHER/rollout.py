from collections import deque

import numpy as np
import pickle
from mujoco_py import MujocoException
from GHER.util import convert_episode_to_batch_major, store_args
import GHER.experiment.config as config
import gym, time

class RolloutWorker:

    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called  
            policy (object): DDPG object
            dims (dict of ints): Dimensions. Example:{'o': 10, 'u': 4, 'g': 3, 'info_is_success': 1}
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used. Generally set to 2
            
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration Control whether to explore

            use_target_net (boolean): Controls whether the self.main network or the self.target network is used when performing actions
            compute_Q (boolean): Control whether to calculate the Q value when calculating the action of the output
            noise_eps (float): scale of the additive Gaussian noise  The Gaussian noise parameter added on the basis of the action is set to 0.2.
            random_eps (float): probability of selecting a completely random action Exploratory factor, set to 0.3
            history_len (int): length of history for statistics smoothing The historical length used for smoothing, set to 100
            render (boolean): whether or not to render the rollouts  Whether to display
        """

        self.envs = [make_env() for _ in range(rollout_batch_size)]
        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        # For recording
        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        # g: goal   o: observation   ag: achieved goals
        self.n_episodes = 0
        self.g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)           # goals
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)   # observations
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        self.reset_all_rollouts()
        self.clear_history()

    def reset_rollout(self, i):
        """
            Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
            i represents the serial number of the parallel worker. This is required at the beginning of the cycle.
        """
        obs = self.envs[i].reset()
        self.initial_o[i] = obs['observation']
        self.initial_ag[i] = obs['achieved_goal']
        self.g[i] = obs['desired_goal']

    def reset_all_rollouts(self):
        """Resets all `rollout_batch_size` rollout workers.
            Repeatedly call the reset_rollout function to reset all workers
        """
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)

    def generate_rollouts(self):
        """
            Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
            policy acting on it accordingly.
            rollout_batch_size = 2, indicating that there are two parallel workers to sample
        """
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)   # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []
        # self.T=50, the loop produces a sample of a cycle
        for t in range(self.T):
            # Self.policy executes the action that can get the action u and the Q value corresponding to the action
            policy_output = self.policy.get_actions(                      # This function is defined in the DDPG class.
                o, ag, self.g,
                compute_Q=self.compute_Q,
                # If self.exploit is False, noise will be used
                noise_eps=self.noise_eps if not self.exploit else 0.,     # Whether to add
                random_eps=self.random_eps if not self.exploit else 0.,   # whether epsilon-greedy
                use_target_net=self.use_target_net)   # use_target_net Generally False

            # Extract actions and Q values. When rollout_batch_size=1, u and Q store 2 worker action and value functions
            if self.compute_Q:
                u, Q = policy_output
                Qs.append(Q)
            else:
                u = policy_output
            if u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)

            # According to the selected action u, execute the action to get the new state o and ag
            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)
            # Perform actions on each worker to get the next obs and ag
            for i in range(self.rollout_batch_size):
                try:
                    # We fully ignore the reward here because it will have to be re-computed for HER.
                    curr_o_new, _, _, info = self.envs[i].step(u[i])
                    if 'is_success' in info:
                        success[i] = info['is_success']
                    o_new[i] = curr_o_new['observation']       # extract o
                    ag_new[i] = curr_o_new['achieved_goal']    # extract ag
                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[key]
                    if self.render:
                        self.envs[i].render()

                except MujocoException as e:
                    return self.generate_rollouts()

            # warning
            if np.isnan(o_new).any():
                self.logger.warning('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            # Record obs and ag
            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(u.copy())             # Sequence of actions
            goals.append(self.g.copy())       # g constant

            # Update, proceed to the next action selection
            o[...] = o_new
            ag[...] = ag_new
        

        # Both obs and ag are adding 1 dimension. So after the execution ends, the first dimension of obs and ag is self.T+1=51
        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        self.initial_o[:] = o                # Ready to proceed next time

        # obs Scale is: (51, 2, 10)   acts for (50, 2, 4)   goals for (50, 2, 3)   achieved_goals for (51, 2, 3)
        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals)
        for key, value in zip(self.info_keys, info_values):
            # print("key =", key, "value =", value)
            episode['info_{}'.format(key)] = value

        # Stats only keeps the last element of successes, that is, only the success of the end of the cycle is recorded.
        # successes.shape=(50,2)(worker the number is 2),  successful.shape=(2,)
        successful = np.array(successes)[-1, :]                
        assert successful.shape == (self.rollout_batch_size,)  # (2,)
        
        # Success rate. The success rate of reaching the goal in T=50 steps. Average between workers
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)  # Record success rate
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))     # Record the mean of Q
        self.n_episodes += self.rollout_batch_size

        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)         # Dump the self.policy parameter locally

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)
