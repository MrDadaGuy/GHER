# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from collections import Counter
import itertools

CALL_NUM = 0

def make_sample_her_transitions(replay_strategy, replay_k, reward_fun, 
                               gmmSample, gmmmeanStd, gmmSample_config, gmmmeanStd_config, 
                               gmmTrain, gmmTrain_config, gmmEval, gmmEval_config):
    def _encode_sample(tv_data, idxes):
        """
            Extract the idxes from the experience pool. Then extract the specified key and connect the specified key to form the matrix.
        """
        for key in tv_data.keys():
            if gmmTrain_config.envname.startswith("Hand") and gmmTrain_config.envname != "HandReach":
                tv_data[key] = tv_data[key][:, ::2, :]

        batches = []
        for idx in idxes:
            # ag
            idx_ag = tv_data['ag'][idx]
            assert idx_ag.shape == (gmmTrain_config.T+1, gmmTrain_config.ag_len)
            # step Indicates where each element belongs in the cycle
            idx_step = (np.arange(0, gmmTrain_config.T+1) / float(gmmTrain_config.T)).reshape(gmmTrain_config.T+1, 1) 
            idx_g = tv_data['g'][idx][-1]    # original goal

            # From the end, the number of successful time steps in the entire cycle
            info_succ = tv_data['info_is_success'][idx, :, 0][::-1]
            info_succ_dic = [(k, list(v)) for k, v in itertools.groupby(info_succ)]
            idx_success_rate = 0. if info_succ_dic[0][0] == 0 else len(info_succ_dic[0][1]) / float(gmmTrain_config.T)
            idx_done = float(tv_data['info_is_success'][idx, -1, 0]) 

            mean_ag = np.mean(tv_data['ag'][idx, -5:, :], axis=0)       
            start_ag = tv_data['ag'][idx, 0, :]                         
            idx_dis = np.linalg.norm(mean_ag-idx_g) / np.linalg.norm(start_ag-idx_g)
 
            train_idx = np.hstack([idx_ag,                                               # ag     (51, 3)
                                   np.tile(idx_g, (gmmTrain_config.T+1, 1)),             # g      (51, 3)
                                   idx_step,                                             # step   (51, 1)
                                   np.tile(idx_success_rate, (gmmTrain_config.T+1, 1)),  # success_rate  (51, 1)
                                   np.tile(idx_done, (gmmTrain_config.T+1, 1)),          # idx_done  (51, 1)
                                   np.tile(idx_dis, (gmmTrain_config.T+1, 1))            # idx_dis  (51, 1)
                                ])                   
            assert train_idx.shape == (gmmTrain_config.T+1, gmmTrain_config.input_size)
            batches.append(train_idx)

        batches_np = np.stack(batches)       # (r4, 51, 10)
        return batches_np


    def _sample_her_transitions_gmm(episode_batch, batch_size_in_transitions):
        global CALL_NUM
        if batch_size_in_transitions == 256:
            CALL_NUM += 1  
            epoch = int(CALL_NUM / 2000)
        
        # Extract batch_size_in_transitions samples from episode_batch
        T = episode_batch['u'].shape[1]                        # Period length T=50
        rollout_batch_size = episode_batch['u'].shape[0]       # The number of cycles stored in the experience pool
        batch_size = batch_size_in_transitions                 # Batch_size = 256 or 100. by transition
    
        # :transitons为 o (256, 10)  u (256, 4)  g (256, 3)  info_is_success (256, 1) ag (256, 3) o_2 (256, 10) ag_2 (256, 3)
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)  # Select cycle number
        t_samples = np.random.randint(T, size=batch_size)   # Select the time step number in the cycle
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}

        # GHER. ---------------------------------------------------------------------- 
        # For three different kinds of samples, select different types of achieved_goal to replace the original goal.
        
        # 1. Determine which setting of the GMM class to use based on the number of batch_size_in_transitions 
        # print("****", batch_size_in_transitions)
        assert batch_size_in_transitions in [100, 256]
        gmmModel, gmmConfig = None, None
        if batch_size_in_transitions == 256:             # for training
            gmmModel = gmmSample
            gmmConfig = gmmSample_config
        if batch_size_in_transitions == 100:             # for Normalize
            gmmModel = gmmmeanStd
            gmmConfig = gmmmeanStd_config
        assert gmmModel is not None, gmmConfig is not None

        # 2. Sample from 3 kinds of goals
        r1, r2, r3 = gmmConfig.ratio1, gmmConfig.ratio2, gmmConfig.ratio3
        r1_n = int(r1 * batch_size_in_transitions)
        r2_n = int(r2 * batch_size_in_transitions)
        r3_n = gmmConfig.batch_size 
        assert r1_n + r2_n + r3_n == batch_size_in_transitions

        all_sample = np.random.permutation(batch_size_in_transitions)                 
        idx1,idx2,idx3 = np.split(all_sample, indices_or_sections=[r1_n, r1_n+r2_n])  
        idxs1, idxs2, idxs3 = sorted(idx1), sorted(idx2), sorted(idx3)
        assert len(idxs1) == r1_n and len(idxs2) == r2_n and len(idxs3) == r3_n
        episode_idxs1 = episode_idxs[idxs1] 
        episode_idxs2 = episode_idxs[idxs2] 
        episode_idxs3 = episode_idxs[idxs3]

        # --------------------------------------------------------------------
        # respectively achieved the achieved_goal replacement for the four samples
        # 1. No action on the first sample. The original goal is still retained.
        # 2. The second sample, the s_t target is set to the original method in HER 'future'
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples) 
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[idxs2]          # Here +2 means no immediate reward
        future_ag = episode_batch['ag'][episode_idxs2, future_t]
        transitions['g'][idxs2] = future_ag    # Replace the original goal of the sampled sample with the probability of future_p with ag
        
        # 4. sample from RNN model 
        batch_np = _encode_sample(episode_batch, episode_idxs3)   # Construct a test sample of the corresponding period (r3, 51, 9)
        assert batch_np.shape == (r3_n, gmmConfig.T+1, gmmConfig.input_size)
        start_point = batch_np[np.arange(0, r3_n), (t_samples[idxs3]/2).astype(np.int), :]

        # warm_up step
        warm_num = gmmConfig.warm_num
        start_point_list = []
        for i in np.arange(1, warm_num+1)[::-1]:
            t_samples_temp = (t_samples[idxs3]/2).astype(np.int) - i
            t_samples_temp[t_samples_temp < 0] = 0       # Boundary processing
            start_point_temp = batch_np[np.arange(0, r3_n), t_samples_temp, :]
            assert start_point_temp.shape == (r3_n, gmmConfig.input_size)
            start_point_list.append(start_point_temp)
        start_point_list.append(start_point)

        start_point_warmall = np.stack(start_point_list, axis=1)
        assert start_point_warmall.shape == (r3_n, warm_num+1, gmmConfig.input_size)

        steps_low = warm_num+2
        steps_high = None
        epoch = int(CALL_NUM / 2000)
        num_steps = None

        # sequence length
        if gmmConfig.envname in ["FetchSlide"]:
            if epoch <= 500:
                steps_high = int(0.1 * epoch + 30)
            else:
                steps_high = 80
            num_steps = np.random.randint(steps_low, steps_high)
        
        elif gmmConfig.envname in ["FetchReach", "FetchPush"]: 
            if epoch <= 20:
                steps_high = int(1 * epoch + 40)   # (epoch, steps_high)=(0,20) (100,30) (200,40), (300, 50)...
            else:
                steps_high = 60
            num_steps = np.random.randint(steps_low, steps_high)

        elif gmmConfig.envname in ["FetchPickAndPlace"]: 
            if epoch <= 200:
                steps_high = int(0.1 * epoch + 40)   # (epoch, steps_high)=(0,20) (100,30) (200,40), (300, 50)...
            else:
                steps_high = 60
            num_steps = np.random.randint(steps_low, steps_high)

        else:
            import sys
            sys.exit()

        # sample
        gmm_sample = gmmModel.sample_gmm(start_point_warmall, num_steps=num_steps, inc_level=True)   # sampling
        assert gmm_sample.shape == (r3_n, num_steps, gmmConfig.ag_len)
        gmm_goal = gmm_sample[:, -1, :]      # Use the last time step result of the sample generation sample as the goal
        transitions['g'][idxs3] = gmm_goal

        # Reconstruct info dictionary for reward computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):             
                info[key.replace('info_', '')] = value   # The info key is "is_success"

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        # transiton.shape = o (256, 10) u (256, 4) g (256, 3) info_is_success (256, 1) ag (256, 3) o_2 (256, 10) ag_2 (256, 3) r (256,)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}
        
        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions

    return _sample_her_transitions_gmm
