import tensorflow as tf
from GHER.util import store_args, nn

class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            # Input_tf represents the input tensor, including obs, goal, action
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)

            # dimo, g, u Representing obs, goal, action Dimension
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions

            # action Range of need to be regulated
            max_u (float): the maximum magnitude of actions; action outputs will be scaled accordingly

            # Both o_stats and g_stats are objects of Normalizer, which are used to state the state. O_stats and g_stats are saved.
                    Mean and std corresponding to obs or goal, and updated. The Normalizer function is also provided.
            o_stats (GHER.Normalizer): normalizer for observations
            g_stats (GHER.Normalizer): normalizer for goals

            # Network structure control
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers

        """
        # Extract the tensor corresponding to obs, goal, action
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        # Normalize the tensor of obs, goal
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)

        # Actor network
        # Obs and goal are connected to form a new representation of the state state. As an input to the Actor, the output action
        input_pi = tf.concat(axis=1, values=[o, g])  # for actor
        # input_pi Input for the network
        # max_u Profiling the range of motion of the final output
        # The last layer activation function is tanh, and the internal activation function is relu
        # The number of neurons in each hidden layer is self.hidden, and the number of network layers is self.layers
        # The number of neurons in the last layer is self.dimu (action dimension)
        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(nn(
                input_pi, [self.hidden] * self.layers + [self.dimu]))
        
        # Critic Network
        # Q(s,a,g) so the input to the network is o, g and action u
        # Network structure: The number of neurons in each layer of the hidden layer is equal, which is self.layers, and the output has only one node.

        with tf.variable_scope('Q'):
 
            # for policy training
            # When training an actor, you need to use the action output by the Actor as an input to Critic
            # The goal of the Actor is to maximize the output of Critic, so the loss is the opposite of the Critic output, which is -self.Q_pi_tf
            input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            
            # for critic training
            # When training Critic, you need to enter the action that the agent actually performs. self.u_tf
            # Really executed actions may add noise to the Actor output, and gradients are not passed to the Actor
            input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)
