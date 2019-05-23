from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from GHER import logger
from GHER.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch)
from GHER.normalizer import Normalizer
from GHER.replay_buffer import ReplayBuffer
from GHER.common.mpi_adam import MpiAdam


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}

class DDPG(object):
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, layers, network_class, polyak, batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
                 sample_transitions, gamma, reuse=False, **kwargs):
        """Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).

        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            buffer_size (int): number of transitions that are stored in the replay buffer
            hidden (int): number of units in the hidden layers
            layers (int): number of hidden layers
            network_class (str): the network class that should be used (e.g. 'GHER.ActorCritic')
            polyak (float): coefficient for Polyak-averaging of the target network
            batch_size (int): batch size for training
            Q_lr (float): learning rate for the Q (critic) network
            pi_lr (float): learning rate for the pi (actor) network
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            max_u (float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
            action_l2 (float): coefficient for L2 penalty on the actions
            clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            scope (str): the scope used for the TensorFlow graph
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per DDPG agent
            subtract_goals (function): function that subtracts goals from each other
            relative_goals (boolean): whether or not relative goals should be fed into the network
            clip_pos_returns (boolean): whether or not positive returns should be clipped
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            sample_transitions (function) function that samples from the replay buffer
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused
        """
        
        # # print("\n\n\n\n1--", input_dims, "\n2--", buffer_size, "\n3--", hidden, 
        #         "\n4--", layers, "\n5--", network_class, "\n6--", polyak, "\n7--", batch_size,
        #          "\n8--", Q_lr, "\n9--", pi_lr, "\n10--", norm_eps, "\n11--", norm_clip, 
        #          "\n12--", max_u, "\n13--", action_l2, "\n14--", clip_obs, "\n15--", scope, "\n16--", T,
        #          "\n17--", rollout_batch_size, "\n18--", subtract_goals, "\n19--", relative_goals, 
        #          "\n20--", clip_pos_returns, "\n21--", clip_return,
        #          "\n22--", sample_transitions, "\n23--", gamma)

        """
         Example of parameter values ​​in the FetchReach-v1 run:
            Input_dims (dict of ints): {'o': 10, 'u': 4, 'g': 3, 'info_is_success': 1} (o, u, g are both input to the network)
            Buffer_size (int): 1E6 (total number of experience pool samples)
            Hidden (int): 256 (number of hidden layer neurons)
            Layers (int): 3 (three-layer neural network)
            Network_class (str): GHER.ActorCritic'
            Polyak (float): 0.95 (smooth parameter updated by target-Network)
            Batch_size (int): 256 (bulk size)
            Q_lr (float): 0.001 (learning rate)
            Pi_lr (float): 0.001 (learning rate)
            Norm_eps (float): 0.01 (to avoid data overflow)
            Norm_clip (float): 5 (norm_clip)
            Max_u (float): 1.0 (the range of the action is [-1.0, 1.0])
            Action_l2 (float): 1.0 (loss coefficient of the actor network)
            Clip_obs (float): 200 (obs is limited to (-200, +200))
            Scope (str): "ddpg" (scope named field used by tensorflow)
            T (int): 50 (the number of cycles of interaction)
            Rollout_batch_size (int): 2 (number of parallel rollouts per DDPG agent)
            Subtract_goals (function): A function that preprocesses the goal, with inputs a and b, and output a-b
            Relative_goals (boolean): False (true if the need for function subtract_goals processing for the goal)
            Clip_pos_returns (boolean): True (Do you need to eliminate the positive return)
            Clip_return (float): 50 (limit the range of return to [-clip_return, clip_return])
            Sample_transitions (function): The function returned by her. The parameters are defined by config.py
            Gamma (float): 0.98 (the discount factor used when Q network update)

            Where sample_transition comes from the definition of HER and is a key part
        """

        if self.clip_return is None:
            self.clip_return = np.inf

        # The creation of the network structure and calculation graph is done by the actor_critic.py file
        self.create_actor_critic = import_function(self.network_class)

        # Extract dimension
        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']   # 10
        self.dimg = self.input_dims['g']   # 4
        self.dimu = self.input_dims['u']   # 3
        # print("+++", input_shapes)    #  {'o': (10,), 'u': (4,), 'g': (3,), 'info_is_success': (1,)}

        # https://www.tensorflow.org/performance/performance_models
         # StagingArea provides simpler functionality and can be executed in parallel with other phases in the CPU and GPU.
         # Split the input pipeline into 3 separate parallel operations, and this is scalable to take advantage of large multi-core environments

         # Define the required storage variable. Suppose self.dimo=10, self.dimg=5, self.dimu=5
         # Then state_shapes={'o':(None, 10), 'g':(None, 5), 'u':(None:5)}
         # Add the variable used by the target network at the same time state_shapes={'o_2':(None, 10), 'g_2': (None, 5)}
         # Prepare staging area for feeding data to the model.

        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'g']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None,)       # Reward for scalar 
        self.stage_shapes = stage_shapes
        # After executing self.stage_shapes =
         # OrderedDict([('g', (None, 3)), ('o', (None, 10)), ('u', (None, 4)), ('o_2', (None, 10) ), ('g_2', (None, 3)), ('r', (None,))])
         # Including g, o, u, target used in o_2, g_2 and reward r
         # Create network.
         # Create tf variables based on state_shape, including g, o, u, o_2, g_2, r
        # self.buffer_ph_tf = [<tf.Tensor 'ddpg/Placeholder:0' shape=(?, 3) dtype=float32>, 
        #                     <tf.Tensor 'ddpg/Placeholder_1:0' shape=(?, 10) dtype=float32>, 
        #                     <tf.Tensor 'ddpg/Placeholder_2:0' shape=(?, 4) dtype=float32>, 
        #                     <tf.Tensor 'ddpg/Placeholder_3:0' shape=(?, 10) dtype=float32>, 
        #                     <tf.Tensor 'ddpg/Placeholder_4:0' shape=(?, 3) dtype=float32>, 
        #                     <tf.Tensor 'ddpg/Placeholder_5:0' shape=(?,) dtype=float32>]
        with tf.variable_scope(self.scope):
            # Create a StagingArea variable
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            # Create a Tensorflow variable placeholder
            self.buffer_ph_tf = [tf.placeholder(tf.float32, shape=shape) 
                for shape in self.stage_shapes.values()]
            # Correspond to the tensorflow variable and the StagingArea variable
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)
            #
            self._create_network(reuse=reuse)

        # Experience pool related operations
        # When T = 50, after execution, buffer_shapes=
        #         {'o': (51, 10), 'u': (50, 4), 'g': (50, 3), 'info_is_success': (50, 1), 'ag': (51, 3)}
        # Note that a, g, u all record all the samples experienced in a cycle, so it is 50 dimensions, but o and ag need 1 more? ? ? ?
        buffer_shapes = {key: (self.T if key != 'o' else self.T+1, *input_shapes[key])
                         for key, val in input_shapes.items()}      # 
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)     #
        buffer_shapes['ag'] = (self.T+1, self.dimg)                 #
        # print("+++", buffer_shapes)

        # buffer_size Is the length counted by sample
        # self.buffer_size=1E6  self.rollout_batch_size=2 buffer_size=1E6
        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions)

    def _random_action(self, n):
        """
            从 [-self.max_u, +self.max_u] Random sampling n actions
        """
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    def _preprocess_og(self, o, ag, g):
        """
            obs, goal, achieve_goal Pretreatment
            In case self.relative_goal=True， then goal = goal - achieved_goal
        """
        if self.relative_goals:
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)    # Increase 1 dimension
            ag = ag.reshape(-1, self.dimg)  # Increase 1 dimension
            g = self.subtract_goals(g, ag)  # g = g - ag
            g = g.reshape(*g_shape)         
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_Q=False):
        """
            Select the action according to the self.main network, then add Gaussian noise, clip, epsilon-greedy operation, and output the processed action
        """
        # If self.relative_goal=True, then the goal is preprocessed. Otherwise only clip
        o, g = self._preprocess_og(o, ag, g)
        # After calling the function self._create_network of this class, the self.main network and the self.target network are created, both of which are ActorCritic objects.
        policy = self.target if use_target_net else self.main   # Select an action based on self.main
        # actor Network output action tensor
        vals = [policy.pi_tf]

        # print("+++")
        # print(vals.shape)

        # Enter the vals of the actor output into the critic network again, and get the output as Q_pi_tf
        if compute_Q:
            vals += [policy.Q_pi_tf]
        # The construction of feed_dict, including obs, goal and action, as input to Actor and Critic
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }
        # Execute the current policy network, output ret. ret[0] for action, ret[1] for Q value
        ret = self.sess.run(vals, feed_dict=feed)
        
        # action postprocessing
        # Add Gaussian noise to Action. np.random.randn refers to sampling from a Gaussian distribution, the noise obeys Gaussian distribution
        u = ret[0]
        noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        u += noise
        u = np.clip(u, -self.max_u, self.max_u)   # After adding noise clip
        
        # Perform epsilon-greedy operation, epsilon for random_eps
        # Np.random.binomial refers to the binomial distribution, the output is 0 or 1, and the probability of output is 1 is random_eps
        # If the binomial distribution outputs 0, then u+=0 is equivalent to no operation; if the output is 1, then u = u + (random_action - u) = random_action
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u
        # 
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def store_episode(self, episode_batch, update_stats=True, verbose=False):
        """
            Episode_batch: array of batch_size x (T or T+1) x dim_key
                        'o' is of size T+1, others are of size T
             Call the store_episode function in replay_buffer to store samples for one sample period
             O_stats and g_stats update and store the mean and standard deviation of obs and goal, respectively, and update them regularly

        """

        # Episode_batch stores a sample of the cycle generated by generate_rollout in rollout.py
        # episode_batch is a dictionary, the keys include o, g, u, ag, info, and the values of the values are respectively
         # o (2, 51, 10), u (2, 50, 4), g (2, 50, 3), ag (2, 51, 3), info_is_success (2, 50, 1)
         # where the first dimension is the number of workers, and the second dimension is determined by the length of the cycle.

        self.buffer.store_episode(episode_batch, verbose=verbose)

        # Update the mean and standard deviation of o_stats and g_stats
        if update_stats:
            # add transitions to normalizer
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]    # Extract next_obs and next_state
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)   # Convert period to total number of samples

            # Call the sampling function in sample_transitions
            # Episode_batch is a dictionary with key and element shapes respectively o (2, 51, 10) u (2, 50, 4) g (2, 50, 3) ag (2, 51, 3) info_is_success (2, 50, 1)
            #                                          o_2 (2, 50, 10)  ag_2 (2, 50, 3)
            # Num_normalizing_transitions=100, there are 2 workers, each worker contains 50 samples of 1 cycle
            transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)

            # The sampled samples are preprocessed and used to update the calculations o_stats and g_stats, defined in the Normalizer, for storing mean and std
            o, o_2, g, ag = transitions['o'], transitions['o_2'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
            # No need to preprocess the o_2 and g_2 since this is only used for stats

            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])

            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()

    def get_current_buffer_size(self):
        """
            Returns the number of samples in the current experience pool
        """
        return self.buffer.get_current_size()

    def _sync_optimizers(self):
        """
            Q_adam and pi_adam are operators for updating actor networks and critic networks.
        """
        self.Q_adam.sync()
        self.pi_adam.sync()

    def _grads(self):
        """
            Return loss function and gradient
            Q_loss_tf, main.Q_pi_tf, Q_grad_tf, pi_grad_tf are all defined in the _create_network function
        """

        # Avoid feed_dict here for performance!
        critic_loss, actor_loss, Q_grad, pi_grad = self.sess.run([
            self.Q_loss_tf,
            self.main.Q_pi_tf,
            self.Q_grad_tf,
            self.pi_grad_tf,
        ])
        return critic_loss, actor_loss, Q_grad, pi_grad

    def _update(self, Q_grad, pi_grad):
        """
            Update main Actor and Critic network
             The updated op is defined in _create_network
        """
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)

    def sample_batch(self):
        """
            Sampling is called by calling the sample function in replay_buffer.py , which is derived from the definition in her.py
             The returned sample consists of batch, which is used to build the feed_dict in the self.stage_batch function.
             Feed_dict will be used as input to select actions and update network parameters

             Calls to sample a batch, then preprocesses o and g. The key of the sample includes o, o_2, ag, ag_2, g
        """
        # Call sample and return transition to dictionary, key and val.shape
        # o (256, 10) u (256, 4) g (256, 3) info_is_success (256, 1) ag (256, 3) o_2 (256, 10) ag_2 (256, 3) r (256,)
        # print("In DDPG: ", self.batch_size)
        transitions = self.buffer.sample(self.batch_size)
        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        ag, ag_2 = transitions['ag'], transitions['ag_2']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)

        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]
        return transitions_batch

    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))
        # tensorboard visualization
        self.tfboard_sample_batch = batch
        self.tfboard_sample_tf = self.buffer_ph_tf
  

    def train(self, stage=True):
        """
            Calculate the gradient and then update
             Self.stage_batch was executed before the parameter update was executed in the train to build the feed_dict used for training. This function is called.
                     The self.sample_batch function, which in turn calls self.buffer.sample, which calls config_her in config.py, which configures the parameters of her.py functions.
             The operators in train are defined in self._create_network .

        """
        if stage:
            self.stage_batch()         # Returns a feed_dict constructed using the sampling method of her.py to calculate the gradient

        critic_loss, actor_loss, Q_grad, pi_grad = self._grads()
        self._update(Q_grad, pi_grad)
        return critic_loss, actor_loss

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        """
            Update the target network, update_target_net_op is defined in the function _create_network
        """
        self.sess.run(self.update_target_net_op)

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _create_network(self, reuse=False):
        """
            Define the calculation flow graph required to calculate Actor and Critic losses
        """
        logger.info("Creating a DDPG agent with action space %d x %s..." % (self.dimu, self.max_u))

        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()
        # running averages
        # Define Normalizer objects for the rules obs and goal respectively
        with tf.variable_scope('o_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

         # mini-batch sampling.
         # Used to store the data structure of a batch sample, which is OrderedDict. After execution, batch_tf is as follows:
         # OrderedDict([('g', <tf.Tensor 'ddpg/ddpg/StagingArea_get:0' shape=(?, 3) dtype=float32>),
         # ('o', <tf.Tensor 'ddpg/ddpg/StagingArea_get:1' shape=(?, 10) dtype=float32>),
         # ('u', <tf.Tensor 'ddpg/ddpg/StagingArea_get:2' shape=(?, 4) dtype=float32>),
         # ('o_2', <tf.Tensor 'ddpg/ddpg/StagingArea_get:3' shape=(?, 10) dtype=float32>),
         # ('g_2', <tf.Tensor 'ddpg/ddpg/StagingArea_get:4' shape=(?, 3) dtype=float32>),
         # ('r', <tf.Tensor 'ddpg/Reshape:0' shape=(?, 1) dtype=float32>)])
         # Defined batch_tf variable will be used as input to the neural network

        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i])
                                for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])

        # Create main network according to ActorCritic.py
        # When creating an ActorCritic network, you don't need to explicitly pass arguments. Use self.__dict__ to assign the corresponding parameters of the DDPG class directly to the corresponding parameters of ActorCritic.
        # print(self.main.__dict__)
        # {'inputs_tf': OrderedDict([('g', <tf.Tensor 'ddpg/ddpg/StagingArea_get:0' shape=(?, 3) dtype=float32>), ('o', <tf.Tensor ' Ddpg/ddpg/StagingArea_get:1' shape=(?, 10) dtype=float32>), ('u', <tf.Tensor 'ddpg/ddpg/StagingArea_get:2' shape=(?, 4) dtype=float32> ), ('o_2', <tf.Tensor 'ddpg/ddpg/StagingArea_get:3' shape=(?, 10) dtype=float32>), ('g_2', <tf.Tensor 'ddpg/ddpg/StagingArea_get:4 ' shape=(?, 3) dtype=float32>), ('r', <tf.Tensor 'ddpg/Reshape:0' shape=(?, 1) dtype=float32>)]),
        # 'net_type': 'main', 'reuse': False, 'buffer_size': 1000000, 'hidden': 256, 'layers': 3, 'network_class': 'GHER.actor_critic:ActorCritic',
        # 'polyak': 0.95, 'batch_size': 256, 'Q_lr': 0.001, 'pi_lr': 0.001, 'norm_eps': 0.01, 'norm_clip': 5, 'max_u': 1.0,
        # 'action_l2': 1.0, 'clip_obs': 200.0, 'scope': 'ddpg', 'relative_goals': False, 'input_dims': {'o': 10, 'u': 4, 'g': 3, 'info_is_success': 1},
        # 'T': 50, 'clip_pos_returns': True, 'clip_return': 49.996, 'rollout_batch_size': 2, 'subtract_goals': <function simple_goal_subtract at 0x7fcf72caa510>, 'sample_transitions': <function make_sample_her_transitions.<locals>._sample_her_transitions at 0x7fcf6e2ce048>,
        # 'gamma': 0.98, 'info': {'env_name': 'FetchReach-v1'}, 'use_mpi': True, 'create_actor_critic': <class 'GHER.actor_critic.ActorCritic'>,
        # 'dimo': 10, 'dimg': 3, 'dimu': 4, 'stage_shapes': OrderedDict([('g', (None, 3)), ('o', (None, 10)), ('u', (None, 4)), ('o_2', (None, 10)), ('g_2', (None, 3)), ('r', (None,))]), ' Staging_tf': <tensorflow.python.ops.data_flow_ops.StagingArea object at 0x7fcf6e2dddd8>,
        # 'buffer_ph_tf': [<tf.Tensor 'ddpg/Placeholder:0' shape=(?, 3) dtype=float32>, <tf.Tensor 'ddpg/Placeholder_1:0' shape=(?, 10) dtype=float32 >, <tf.Tensor 'ddpg/Placeholder_2:0' shape=(?, 4) dtype=float32>, <tf.Tensor 'ddpg/Placeholder_3:0' shape=(?, 10) dtype=float32>, <tf .Tensor 'ddpg/Placeholder_4:0' shape=(?, 3) dtype=float32>, <tf.Tensor 'ddpg/Placeholder_5:0' shape=(?,) dtype=float32>],
        # 'stage_op': <tf.Operation 'ddpg/ddpg/StagingArea_put' type=Stage>, 'sess': <tensorflow.python.client.session.InteractiveSession object at 0x7fcf6e2dde10>, 'o_stats': <GHER.normalizer.Normalizer Object at 0x7fcf6e2ee940>, 'g_stats': <GHER.normalizer.Normalizer object at 0x7fcf6e2ee898>,
        # 'o_tf': <tf.Tensor 'ddpg/ddpg/StagingArea_get:1' shape=(?, 10) dtype=float32>, 'g_tf': <tf.Tensor 'ddpg/ddpg/StagingArea_get:0' shape=( ?, 3) dtype=float32>, 'u_tf': <tf.Tensor 'ddpg/ddpg/StagingArea_get:2' shape=(?, 4) dtype=float32>, 'pi_tf': <tf.Tensor 'ddpg/main /pi/mul:0' shape=(?, 4) dtype=float32>, 'Q_pi_tf': <tf.Tensor 'ddpg/main/Q/_3/BiasAdd:0' shape=(?, 1) dtype=float32 >, '_input_Q': <tf.Tensor 'ddpg/main/Q/concat_1:0' shape=(?, 17) dtype=float32>, 'Q_tf': <tf.Tensor 'ddpg/main/Q/_3_1/ BiasAdd: 0' shape=(?, 1) dtype=float32>}

        with tf.variable_scope('main') as vs:
            if reuse:
                vs.reuse_variables()
            self.main = self.create_actor_critic(batch_tf, net_type='main', **self.__dict__)
            vs.reuse_variables()

        # O_2, g_2 is used to create target network
        with tf.variable_scope('target') as vs:
            if reuse:
                vs.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            target_batch_tf['g'] = batch_tf['g_2']   # Since the target network is used to calculate the target-Q value, o and g need to use the value of the next state.
            self.target = self.create_actor_critic(
                target_batch_tf, net_type='target', **self.__dict__)
            vs.reuse_variables()
        assert len(self._vars("main")) == len(self._vars("target"))

        # loss functions
        # To calculate Critic's target-Q value, you need to use the Actor's target network and Critic's target network.
        # target_Q_pi_tf uses the next state o_2 and g_2
        target_Q_pi_tf = self.target.Q_pi_tf
        clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
        target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_pi_tf, *clip_range) 
        # The loss function of Critic is the square of the difference between target_tf and Q_tf. Note that the gradient is not passed through target_tf.
        self.Q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf))

        # The loss function of the Actor is the opposite of the Q value obtained by the actor's output in the main network.
        self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)
        # Add regulars to Actors
        self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
        
        # Calculating the gradient 
        Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'))
        assert len(self._vars('main/Q')) == len(Q_grads_tf)
        assert len(self._vars('main/pi')) == len(pi_grads_tf)
        self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))     # Gradient and variable name correspond
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/Q'))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))

        # optimizers
        self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)

        # polyak averaging
        self.main_vars = self._vars('main/Q') + self._vars('main/pi')        # Put together the parameters of the Actor and Critic network
        self.target_vars = self._vars('target/Q') + self._vars('target/pi')
        self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')
        self.init_target_net_op = list(            # Target Initialization operation, the main network parameter is directly assigned to the target
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(          # In the target update operation, the main network and the target network need to be weighted according to the parameter polyak
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))

        # # Tensorboard visualization
        # tf.summary.scalar("Q_target-Q-mean", tf.reduce_mean(target_tf))
        # tf.summary.histogram("Q_target-Q", target_tf)
        # tf.summary.scalar("Q_Td-error-mean", tf.reduce_mean(target_tf - self.main.Q_tf))
        # tf.summary.histogram("Q_Td-error", target_tf - self.main.Q_tf)
        # tf.summary.scalar("Q_reward-mean", tf.reduce_mean(batch_tf['r']))
        # tf.summary.histogram("Q_reward", batch_tf['r'])
        # tf.summary.scalar("Q_loss_tf", self.Q_loss_tf)
        # tf.summary.scalar("pi_loss_tf", self.pi_loss_tf)
        # self.merged = tf.summary.merge_all()

        # initialize all variables
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()
        self._init_target_net()

    def logs(self, prefix=''):
        logs = []
        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def tfboard_func(self, summary_writer, step):
        """
            Tensorboard visualization
        """
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.tfboard_sample_tf, self.tfboard_sample_batch)))
        summary = self.sess.run(self.merged) 
        summary_writer.add_summary(summary, global_step=step)

        print("S"+str(step), end=",")

    def __getstate__(self):
        """
            Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'env', 'sample_transitions',
                             'stage_shapes', 'create_actor_critic']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None

        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars('') if 'buffer' not in x.name]
        assert(len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)

    # -----------------------------------------
    def updata_loss_all(self, verbose=False):
        assert self.buffer.current_size > 0
        idxes = np.arange(self.buffer.current_size)
        print("--------------------------------------")
        print("Updata All loss start...")
        self.buffer.update_rnnLoss(idxes, verbose=verbose)
        print("Updata All loss end ...")


