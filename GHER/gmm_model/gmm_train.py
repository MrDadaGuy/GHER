#-*- coding: utf-8 -*-
__author__ = "ChenjiaBai"

import collections
import os
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt        
import itertools
from GHER.gmm_model.CONFIG import rnn_train_Config, rnn_eval_Config, rnn_sample_Config
from GHER.gmm_model.gmm_model import GMMModel


class GMMInput:
    def __init__(self, ename=None, shuffer=True):
        """
            o.shape= (50000, 51, 25)
            u.shape= (50000, 50, 4)
            g.shape= (50000, 50, 3)
            ag.shape= (50000, 51, 3)
            info_is_success.shape= (50000, 50, 1)
            o_2.shape= (50000, 50, 25)
            ag_2.shape= (50000, 50, 3)
        """
        self.config = config = rnn_train_Config()
        self.envname = envname = config.envname
        self.batch_size_in_episode = batch_size_in_episode = config.batch_size_in_episode
        self.seq_len = seq_len = config.seq_len   
        self.seq_window = seq_window = config.seq_window
        self.T = T = config.T
        self.batch_size = batch_size = config.batch_size 
        self._next_idx = 0
        self.shuffer = shuffer

        self._keys = ['g', 'ag', 'info_is_success']
        f = os.path.join("Buffer", envname, "train.pkl")
        f = f if os.path.exists(f) else "../gmm_model/"+f
        assert os.path.exists(f)
        self.data = pickle.load(open(f, "rb"))

        for k, v in self.data.items():
            print("key:", k, ",  val:", v.shape)
        
        print("load data from ", envname)
        self.storage = self.data['g'].shape[0]   
        self._shuffle_data()
        
        # 分开训练集和测试集. 分别占 4/5 和 1/5
        self.train_data, self.valid_data = {}, {}
        for key in self._keys:
            self.train_data[key] = self.data[key][:int(self.storage*(4/5.)), :, :]
            self.valid_data[key] = self.data[key][int(self.storage*(4/5.)):, :, :]
        self.train_storage, self.valid_storage = self.train_data['g'].shape[0], self.valid_data['g'].shape[0]
        print("Number of training set samples: {}, number of validation set samples: {}".format(self.train_storage, self.valid_storage))

    def _shuffle_data(self):
        """
            The order of the data in self.data is upset
        """
        idx = np.random.permutation(self.storage)
        for key in self._keys:
            if self.shuffer:
                self.data[key] = self.data[key][idx, :, :]
            else:
                pass

    def _encode_sample(self, idxes, tv_data):
        """
            从经验池中将idxes取出. 随后提取指定的key，连接指定key，组成矩阵
        """
        # print("---", type(idxes), idxes.shape, idxes[0])
        batches = []
        for idx in idxes:
            # ag
            idx_ag = tv_data['ag'][idx]
            assert idx_ag.shape == (self.T+1, self.config.ag_len)
            # step Indicates where each element belongs in the cycle
            idx_step = (np.arange(0, self.T+1) / float(self.T)).reshape(int(self.T)+1, 1) 

            # The following is the amount that does not change in the sequence.
            # goal
            idx_g = tv_data['g'][idx][-1]    # (3,)

            # From the end, the number of successful time steps in the entire cycle
            info_succ = tv_data['info_is_success'][idx, :, 0][::-1]
            info_succ_dic = [(k, list(v)) for k, v in itertools.groupby(info_succ)]
            idx_success_rate = 0. if info_succ_dic[0][0] == 0 else len(info_succ_dic[0][1])/float(self.T)
            
            # Whether the end of the cycle is successful
            idx_done = float(tv_data['info_is_success'][idx, -1, 0]) 

            # Average position of the entire cycle from the target / distance from the starting point to the target
            mean_ag = np.mean(tv_data['ag'][idx, -5:, :], axis=0)        # Average position of the last 5 steps (3,)
            start_ag = tv_data['ag'][idx, 0, :]                          # 起始位置 (3,)
            idx_dis = np.linalg.norm(mean_ag-idx_g) / np.linalg.norm(start_ag-idx_g)  # Scalar, equivalent to the ratio, the smaller the better
 
            # Follow a sequence ag(x,y,z), g(x,y,z), step, success_rate, success_first Organize
            train_idx = np.hstack([idx_ag,                                        # ag     (51, 3)
                                   np.tile(idx_g, (self.T+1, 1)),                 # g      (51, 3)
                                   idx_step,                                      # step   (51, 1)
                                   np.tile(idx_success_rate, (self.T+1, 1)),      # success_rate  (51, 1)
                                   np.tile(idx_done, (self.T+1, 1)),              # idx_done  (51, 1)
                                   np.tile(idx_dis, (self.T+1, 1))                # idx_dis  (51, 1)
                                ])                   
            assert train_idx.shape == (self.T+1, self.config.input_size)
            batches.append(train_idx)

        # Integrate features into numpy
        batches_np = np.stack(batches)
        # assert batches_np.shape == (self.batch_size_in_episode, 51, 10)   
        self.batches_np = batches_np
        X = batches_np[:, :self.T, :]   
        Y = batches_np[:, 1:, :]         # dislocation
        # assert X.shape == Y.shape == (self.batch_size_in_episode, 50, 10)

        X_batch = []
        Y_batch = []
        for i in range(0, self.T-self.seq_len+1, self.seq_window):   # 0,2,4,6,...,38,40
            x = X[:, i: i+self.seq_len]
            y = Y[:, i: i+self.seq_len]  
            # assert x.shape == y.shape == (self.batch_size_in_episode, self.seq_len, 10)  # shape=(50, 10, 9)
            X_batch.append(x)
            Y_batch.append(y)
        
        # vstack
        X_train = np.vstack(X_batch)     
        Y_train = np.vstack(Y_batch)    
        return X_train, Y_train, batches_np


    def create_batch(self, mode="train"):
        """
            Mode control samples from self.train_data or self.valid_data
             Train data for sequential sampling, valid random sampling
        """
        assert mode in ["train", "valid"]

        if mode == 'train':
            batch = self.batch_size_in_episode
            if self._next_idx + batch > self.train_storage:
                self._next_idx = 0
            idxes = np.arange(self._next_idx, self._next_idx + batch)
            self._next_idx = (self._next_idx + batch) % self.train_storage
            X, Y, _ = self._encode_sample(idxes, self.train_data)

        elif mode == 'valid':
            idxes = np.random.randint(0, self.valid_storage, 1000)
            X, Y, _ = self._encode_sample(idxes, self.valid_data)

        return X, Y


def TRAIN(epoch_num):
    # config
    train_config = rnn_train_Config()
    eval_config = rnn_eval_Config()
    sample_config = rnn_sample_Config()
    
    # build model
    with tf.name_scope("GMM_Model"):
        # session
        config_proto = tf.ConfigProto()
        config_proto.gpu_options.per_process_gpu_memory_fraction = 0.7
        rnn_sess = tf.Session(config=config_proto)
        
        # train, eval, sample Three models reuse weights under different settings.
        with tf.name_scope("Train"):
            with tf.variable_scope("Model") as scope:  # training
                rnn_m = GMMModel(rnn_sess, config_str="train")

        with tf.name_scope("Eval"):
            with tf.variable_scope("Model", reuse=True) as scope:
                rnn_eval = GMMModel(rnn_sess, config_str="eval")

        with tf.name_scope("Sample"):
            with tf.variable_scope("Model", reuse=True) as scope:
                rnn_sample = GMMModel(rnn_sess, config_str="sample")
        
    with tf.name_scope("GMM_Input"):
        gmmInput = GMMInput()
    
    rnn_sess.run(tf.global_variables_initializer())

    # Learning rate
    init_lr = train_config.init_lr                    # Initial learning rate
    decay = train_config.lr_decay                     # Learning rate attenuation

    # cycle
    min_loss = float("inf")
    train_losses, eval_losses = [], []
    for epoch in range(epoch_num):
        # Learning rate update
        rnn_sess.run(tf.assign(rnn_m.lr, init_lr * (decay ** epoch)))        
        
        # Extract a batch of data and train train_steps times
        train_cost = rnn_m.train_epoch(gmmInput, epoch, train_steps=8)   # Train_steps modification
        train_losses.append(train_cost)

        # Eval
        X, Y = gmmInput.create_batch(mode="valid")
        eval_loss = rnn_eval.eval(X, Y)
        print("EVAL loss:", eval_loss, "\n------------") 
        eval_losses.append(eval_loss)

        # Sample model saved for generating samples
        if epoch % 20 == 0:
            savename = "model-"+str(epoch)+"-({:.4})".format(eval_loss)+".ckpt"
            rnn_m.save_model(savename)
            rnn_sample.save_model(savename)
    rnn_m.save_model("model-last.ckpt")
    
    print("average eval loss in all cycles: \t", np.mean(eval_losses))
    print("average eval loss in last 50 cycles: \t", np.mean(eval_losses[-50:]))
    

if __name__ == '__main__':
    TRAIN(epoch_num=2000)
    

