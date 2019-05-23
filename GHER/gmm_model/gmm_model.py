# -*- coding: utf-8 -*-
__author__ = "ChenjiaBai"

import time
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import os
from GHER.gmm_model.CONFIG import rnn_train_Config, rnn_eval_Config, rnn_sample_Config, rnn_meanStd_Config, rnn_test_Config
from GHER.gmm_model.util.lossfunc_ag_dim3 import lossfunc 
from GHER.gmm_model.util.save_config import save_config


class GMMModel(object):
    """
        The GMM RNN model.
    """
    def __init__(self, rnn_sess, config_str):
        # Basic setting initialization  --------------
        self.sess = rnn_sess
        config = None
        # config class
        if config_str == 'train':
            config = rnn_train_Config()
        elif config_str == 'eval':
            config = rnn_eval_Config()
        elif config_str == 'sample':
            config = rnn_sample_Config()
        elif config_str == "meanStd":
            config = rnn_meanStd_Config()
        elif config_str == "test":
            config = rnn_test_Config()
        else:
            print("GMMModel Config_str error ..")
            return
        self.config = config
        self.is_training = is_training = config.is_training

        self._cell = None
        
        with tf.variable_scope("RNN_input"):
            # Input, the value of the sentence after embedding. input_data.shape=(batch_size, seq_len), after embedding (batch_size, seq_len, size)
            self._input_data = tf.placeholder(tf.float32, [config.batch_size, config.seq_len, config.input_size], name="X_train")
            self._targets = tf.placeholder(tf.float32, [config.batch_size, config.seq_len, config.output_size], name="Y_train")

            # Dropout is only added between input and lstm. Add dropout during training, the rest is not added
            inputs = self._input_data
            if is_training and config.keep_prob < 1.:
                inputs = tf.nn.dropout(self._input_data, config.keep_prob)

        with tf.variable_scope("RNN_Multicell"):
            # output output and state. | output.shape = (batch_size*seq_len, hidden_size)  
            output, state = self._build_rnn_graph_lstm(inputs)
            self._final_state = state
            assert output.shape == (config.batch_size * config.seq_len, config.hidden_size) 

        # Calculate the number of parameters included in the output layer based on the number of components included in the GMM
        NOUT = config.NOUT

        with tf.variable_scope("RNN_output"):
            dense_output1 = tf.layers.dense(inputs=output, units=config.dense1_num, 
                                activation=tf.nn.relu, name="dense_1")
            dense_output2 = tf.layers.dense(inputs=dense_output1, units=NOUT, 
                                activation=None, name="dense_2")
            logits = tf.reshape(dense_output2, [config.batch_size, config.seq_len, NOUT])
            self._logits = logits

        with tf.name_scope("GMM"):
            # extract the GMM paramaters
            def get_mixture_conf(z):
                assert z.shape == (config.batch_size, config.seq_len, NOUT)
                flat_z = tf.reshape(z, [-1, NOUT])                        # reshape
                z_pi, z_mu1, z_mu2, z_mu3, z_sigma1, z_sigma2, z_sigma3 = tf.split(
                    axis=1, num_or_size_splits=7, value=flat_z)
                # softmax normalize
                max_pi = tf.reduce_max(z_pi, axis=1, keepdims=True)
                z_pi = tf.subtract(z_pi, max_pi)
                z_pi = tf.exp(z_pi)
                normalize_pi = tf.reciprocal(tf.reduce_sum(z_pi, 1, keepdims=True))
                z_pi = tf.multiply(z_pi, normalize_pi)

                # sigma > 0
                z_sigma1 = tf.exp(z_sigma1)
                z_sigma2 = tf.exp(z_sigma2)
                z_sigma3 = tf.exp(z_sigma3)

                return [z_pi, z_mu1, z_mu2, z_mu3, z_sigma1, z_sigma2, z_sigma3]

            self.pi, self.mu1, self.mu2, self.mu3, self.sigma1, self.sigma2, self.sigma3 = get_mixture_conf(self._logits)
            assert self.pi.shape == self.mu1.shape == self.sigma1.shape == (
                                            config.batch_size*config.seq_len, config.num_mixture)
                
        with tf.name_scope("RNN_loss"):
            """
                compute the loss function
                Compare the distribution of the output pi, mu, sigma with the true value of self._targets to calculate the loss
            """
            # 1. Preprocesses the real value self._targets. Each output contains 9 elements, but only agx, agy, agz losses
            flat_targets = tf.reshape(self._targets, [-1, config.output_size])     # (batch_size*seq_len,10)
            [agx, agy, agz, gx, gy, gz, step, success_rate, done, dis] = tf.split(
                            axis=1, num_or_size_splits=config.output_size, value=flat_targets)
            assert agx.shape == gx.shape == step.shape == success_rate.shape == (config.batch_size * config.seq_len, 1)

            # 2. Calculated loss
            gmm, loss = lossfunc(self.pi, self.mu1, self.mu2, self.mu3, 
                                self.sigma1, self.sigma2, self.sigma3, agx, agy, agz, config)
            
            self.loss = loss
            assert loss is not None and gmm is not None
            tf.summary.scalar('Loss', self.loss)
            self.gmm = gmm
            self.ag = tf.concat([agx, agy, agz], axis=1)
            assert self.ag.shape == (config.batch_size*config.seq_len, 3)                    

        with tf.variable_scope("Train_op"):
            self.lr = tf.Variable(0.0, trainable=False)    # learning rate decay
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), config.grad_clip)
            optimizer = tf.train.AdamOptimizer(self.lr)
            self._train_op = optimizer.apply_gradients(zip(grads, tvars))
            self.grads = grads

        self.merged = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(os.path.join(config.save_path, "sess"), self.sess.graph)
        self.saver = tf.train.Saver()
        self.config_str = config_str
        save_config(config_str, self.config)
        

    def _build_rnn_graph_lstm(self, inputs):
        """
            # inputs.shape = (batch_size, seq_len, size)
            # Input inputs, perform forward propagation, output output and hidden layer
            # Construct a single-layer RNN function
        """
        config = self.config
        assert config.rnn_type in ["lstm", "rnn", "gru"]
        def make_cell():
            cell = None
            if config.rnn_type == "lstm":
                cell = tf.contrib.rnn.LSTMBlockCell(num_units=config.hidden_size, use_peephole=False)
            elif config.rnn_type == "rnn":
                cell = tf.contrib.rnn.BasicRNNCell(num_units=config.hidden_size, forget_bias=1.0, state_is_tuple=True)
            elif config.rnn_type == "gru":
                cell = tf.contrib.rnn.GRUBlockCell(num_units=config.hidden_size)
            else:
                pass   
            if config.is_training and config.keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
 
            return cell

        # stack LSTM
        cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(config.num_layers)],state_is_tuple=True)
        if config.is_training and config.keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)

        # init hidden state
        self._initial_state = cell.zero_state(config.batch_size, tf.float32)

        # dynamic_rnn 
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=self._initial_state, time_major=False)
        
        # outputs.shape = (batch_size, seq_len, hidden_size)  
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])  # Link axis=1, time_steps
        return output, state


    def save_model(self, savename):
        self.saver.save(self.sess, os.path.join(self.config.save_path, "checkpoint", savename))

    def reload_model(self):
        ckpt_file = os.path.join(self.config.save_path, "checkpoint", "model-last.ckpt")
        if os.path.exists(ckpt_file+".index"):
            pass
        else:
            print("Need Load Model from Parent dir.")
            ckpt_file = os.path.join("../gmm_model", ckpt_file)
        print(os.path.abspath(ckpt_file))
        assert os.path.exists(ckpt_file+".index")
        self.saver.restore(self.sess, ckpt_file)


    def train_epoch(self, gmmInput, epoch, train_steps):
        """
            Epoch represents the number of times the big loop has been trained for counting
             Train_steps represents the number of iterations for looping
             Samples are generated by gmmInput for each iteration
        """
        # Return value and train_op
        fetches = {
            "loss": self.loss,
            "train_op": self._train_op,
            "merged": self.merged,          # tensorboard summary
        }

        # Start training
        costs = 0.0
        vals = None
        for step in range(train_steps): 
            # feed_dict
            input_data, targets = gmmInput.create_batch(mode='train')   # sampling，shape=(batch_size, 50, 10)  
            assert input_data.shape == (self.config.batch_size, self.config.seq_len, self.config.input_size)
            # upset
            permu = np.random.permutation(self.config.batch_size)      
            input_data = input_data[permu, :, :]
            targets = targets[permu, :, :]
            # Train Note that init_state is an all zero vector in each training
            feed_dict = {self._input_data: input_data, self._targets: targets}
            vals = self.sess.run(fetches, feed_dict)
            
            # Output
            loss = vals["loss"]
            summary = vals["merged"]    
            self.summary_writer.add_summary(summary, global_step=epoch*train_steps+step)
            costs += loss

        # print("\nloss_weight=\n", vals['loss_weight'][:20], "\n")
        # print("loss_no_weight=\n", vals['loss_no_weight'][:20], "\n")            

        costs_mean = costs / train_steps
        print("RNN train: epoch=", epoch, ": \nTrain Loss =", costs_mean)
        self.costs_mean = costs_mean    # Save the latest loss
        
        return costs_mean
        

    def eval(self, eval_data, eval_target):
        """
            evalueate: Loss of eval_data and eval_target
        """
        fetches = {"loss": self.loss,}
        feed_dict = {self._input_data: eval_data, self._targets: eval_target}
        vals = self.sess.run(fetches, feed_dict)

        loss = vals["loss"]    # (batchsize, 1)
        return loss

    # ---------------------------------------------------------------------------
    def sample_gmm(self, init_input, num_steps=None, inc_level=None, level=None):
        """
            Sampling directly from the GMM probability model. init_input contains ag, g, step, success_rate, success_start
             Init_input.shape=(batch, wn+1, 10) represents the initial input
            increase_level = True
        """
        # Setting
        batch = self.config.batch_size
        increase_level = self.config.increase_level if inc_level == None else inc_level
        steps = self.config.num_steps if num_steps == None else num_steps
        init_inputs = init_input.copy()
        assert init_inputs.shape == (batch, self.config.warm_num+1, 10)

        # Select components in the GMM according to different settings, and output the Gaussian mean as the sampled value. 
        def samplefunc(pi, mu1, mu2, mu3, method=self.config.method):
            assert self.config.method in ['max', 'softmax', 'random']
            assert mu1.shape == mu2.shape == mu3.shape == (batch, self.config.num_mixture)
            choose_pi = None
            if method == "max":
                max_pi = np.argmax(pi, axis=1)
                choose_pi = max_pi        
            elif method == "softmax":
                softmax_pi = []
                for i in range(pi.shape[0]):
                    pi_choose = np.random.choice(np.arange(0, int(self.config.num_mixture)), p=pi[i])
                    softmax_pi.append(pi_choose)
                softmax_pi = np.array(softmax_pi)
                choose_pi = softmax_pi
            elif method == "random":
                random_pi = np.random.choice(self.config.num_mixture, batch)
                choose_pi = random_pi
            else:
                pass
            assert choose_pi.shape == (batch,)

            # Extract the corresponding component mu value
            mu1_sample = np.choose(choose_pi, mu1.T)
            mu2_sample = np.choose(choose_pi, mu2.T)
            mu3_sample = np.choose(choose_pi, mu3.T)
            assert mu1_sample.shape == mu2_sample.shape == mu3_sample.shape == (batch,)
            next_sample = np.stack([mu1_sample, mu2_sample, mu3_sample], axis=1)  # (batch, 3)
            return next_sample
        
        assert init_inputs.shape == (batch, self.config.warm_num+1, self.config.input_size)
        
        # Warm start
        state = self.sess.run(self._initial_state)
        for i in range(self.config.warm_num):
            warm_input = init_inputs[:, i:i+1, :]
            assert warm_input.shape == (batch, 1, 10)
            feed_dict = {self._input_data: warm_input, self._initial_state: state}
            # run
            state = self.sess.run(self._final_state, feed_dict)

        if increase_level:
            init_inputs[:, :, -2] = 1.
            if self.config.envname == "FetchReach":
                init_inputs[:, :, -3] = 0.6 + (1.0 - 0.6) * np.random.random((init_inputs.shape[0], 1))    
            elif self.config.envname in ["FetchPush", "FetchPickAndPlace"]:
                init_inputs[:, :, -3] = 0.4 + (1.0 - 0.4) * np.random.random((init_inputs.shape[0], 1))
            elif self.config.envname == "FetchSlide":
                init_inputs[:, :, -3] = 0.02 + (0.2 - 0.02) * np.random.random((init_inputs.shape[0], 1))
            else:
                print("gmm_model.py increase level ERROR!")
                import sys
                sys.exit()

        inputs = init_inputs[:, -1:, :]
        outputs = []                         # Save the simulation sequence   
        assert inputs.shape == (batch, 1, self.config.input_size)
        
        for i in range(steps-self.config.warm_num-1):
            # feed
            feed_dict = {self._input_data: inputs, self._initial_state: state} 
            
            # sample, state continuity
            final_state, pi, mu1, mu2, mu3 = self.sess.run([self._final_state, self.pi, self.mu1, 
                self.mu2, self.mu3], feed_dict)  
            next_sample = samplefunc(pi, mu1, mu2, mu3)
            assert pi.shape == mu1.shape == mu2.shape == mu3.shape == (batch, self.config.num_mixture)
            assert next_sample.shape == (batch, 3)
            outputs.append(next_sample)

            # next input
            next_agx, next_agy, next_agz = np.split(next_sample, indices_or_sections=3, axis=1)  # New. 
            assert next_agx.shape == (batch, 1)
            agx, agy, agz, gx, gy, gz, step, success_rate, success_done, success_dis = np.split(
                    inputs, indices_or_sections=self.config.input_size, axis=2)            
            assert agx.shape == gx.shape == success_rate.shape == (batch, 1, 1)

            # Ag is replaced, g, success_rate, success_first remains unchanged, step+0.02.
            inputs = np.stack([next_agx, next_agy, next_agz,
                               gx[:,:,0], gy[:,:,0], gz[:,:,0], 
                                (step + (1./self.config.T))[:,:,0],   # step += 0.02 
                                success_rate[:,:,0],                  # success_rate constant 
                                success_done[:,:,0],                  # done
                                success_dis[:,:,0]], axis=2)          # success_first constant
            assert inputs.shape == (batch, 1, self.config.input_size)
            
            # state continue
            state = final_state
       
        seq_all = np.concatenate([init_inputs[:, :, :3], np.stack(outputs, axis=1)], axis=1)
        assert seq_all.shape == (batch, steps, 3)
        return seq_all