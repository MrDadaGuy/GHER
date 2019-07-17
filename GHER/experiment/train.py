#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, '/home/modsim/GHER/')

import click
import numpy as np
import json
from mpi4py import MPI
import tensorflow as tf
from GHER import logger
from GHER.common import set_global_seeds
from GHER.common.mpi_moments import mpi_moments
import GHER.experiment.config as config
from GHER.rollout import RolloutWorker
from GHER.util import mpi_fork


def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]


def train(policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_policies, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    # Path to save network parameters
    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')

    logger.info("Training...")
    best_success_rate = -1

    for epoch in range(n_epochs):
        print("Epoch=", epoch)
        # train
        rollout_worker.clear_history()

        for i in range(n_cycles):         # n_cycles=50
            episode = rollout_worker.generate_rollouts()  # Generate 1 cycle sample
            # transfer DDPG的 store_episode Function, further call replay_buffer middle store function
            policy.store_episode(episode, verbose=True)
            for j in range(n_batches):    # n_batches = 40
                policy.train()            # Defined in DDPG.train In, make an update
            # Update target-Q
            policy.update_target_net()

        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            epo_eval = evaluator.generate_rollouts()

        print("-----------------------------")
        # record logs
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()

        # File read and write when saving a policy tensorboard middle tf.summary Conflicting, not running at the same time
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and success_rate > best_success_rate and save_policies:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]

    

def launch(env_name, logdir, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return, override_params={}, save_policies=True):
    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        whoami = mpi_fork(num_cpu)
        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()
    # print("rank = ", rank)   # rank = 0

    # Configure logging
    if rank == 0:
        if logdir or logger.get_dir() is None:
            logger.configure(dir=logdir)
    else:
        logger.configure()
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)  # Create a directory for logging

    # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS   # Dictionary, default parameters are defined in config.py in
    params['env_name'] = env_name    # add parameters env_name
    params['replay_strategy'] = replay_strategy  # add parameters replay_strategy="future"
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    # Can be specified in the input of this function override_params to replace params specific parameters
    params.update(**override_params)  # makes it possible to override any parameter
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)                # Write all current parameter settings to the file
    # This function is config.py In, added ddpg_params Key, rename the original key to "_" + key name
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    if num_cpu == 1:
        logger.warn()
        logger.warn('*** Warning ***')
        logger.warn(
            'You are running HER with just a single MPI worker. This will work, but the ' +
            'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
            'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
            'are looking to reproduce those results, be aware of this. Please also refer to ' + 
            'https://github.com/openai/baselines/issues/314 for further details.')
        logger.warn('****************')
        logger.warn()

    # Return dimension after execution dims = {'o': 10, 'u': 4, 'g': 3, 'info_is_success': 1}
    dims = config.configure_dims(params)

    # Return after execution DDPG Instantiated object of class
    policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return)

    # The following parameters are used to control the selection of actions in training and testing.
    rollout_params = {
        'exploit': False,
        'use_target_net': False,  # Control action selection is used main Network or target The internet
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }
    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],  # Generally False
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }
    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    # RolloutWorker Defined in rollout.py in
    # rollout_worker The parameters are set to:
    # { 'rollout_batch_size': 2, 'exploit': False, 'use_target_net': False, 
    #   'compute_Q': False, 'noise_eps': 0.2, 'random_eps': 0.3, 'history_len': 100, 
    #   'render': False, 'make_env': function, 'policy': DDPG Class object, 'dims': {'o': 10, 'u': 4, 'g': 3, 'info_is_success': 1}, 
    #   'logger': class, 'use_demo_states': True, 'T': 50, 'gamma': 0.98, 'envs': [<TimeLimit<FetchReachEnv<FetchReach-v1>>>, 
    # 'info_keys': ['is_success'], 'success_history': deque([], maxlen=100), 'Q_history': deque([], maxlen=100), 'n_episodes': 0, 
    # 'g': array([[1.4879797 , 0.6269019 , 0.46735048], [1.3925381 , 0.8017641 , 0.49162573]], dtype=float32), 
    # 'initial_o': array([[ 1.3418437e+00,  7.4910051e-01,  5.3471720e-01,  1.8902746e-04, 7.7719116e-05,  3.4374943e-06, -1.2610036e-08, -9.0467189e-08, 4.5538709e-06, -2.1328783e-06],[ 1.3418437e+00,  7.4910051e-01,  5.3471720e-01,  1.8902746e-04, 7.7719116e-05,  3.4374943e-06, -1.2610036e-08, -9.0467189e-08, 4.5538709e-06, -2.1328783e-06]], dtype=float32), 
    # 'initial_ag': array([[1.3418437, 0.7491005, 0.5347172],

    rollout_worker = RolloutWorker(params['make_env'], policy, dims, logger, **rollout_params)
    rollout_worker.seed(rank_seed)

    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(rank_seed)
    
    # training
    train(
        logdir=logdir, policy=policy,                                  # Policy is an object of the DDPG class
        rollout_worker=rollout_worker, evaluator=evaluator,            # Rollout_worker and evaluator are defined in rollout.py 
        n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],  # N_epochs is the total number of training cycles, n_test_rollouts=10
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],    # n_cycles=10, n_batches=40
        policy_save_interval=policy_save_interval, save_policies=save_policies)  # policy_save_interval=5, save_polices=True


@click.command()
@click.option('--env_name', type=str, default='Ros-Unity-Sim-v1', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--logdir', type=str, default="result/GHer-result/FetchPush/result/", help='the path to where logs and policy pickles should go. If not specified, creates a folder in /tmp/')
@click.option('--n_epochs', type=int, default=500, help='the number of training epochs to run')
@click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (using MPI)')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=25, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['G-HER11-future', 'future', 'none']), 
    default='G-HER11-future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
def main(**kwargs):
    launch(**kwargs)


if __name__ == '__main__':
    main()
