import argparse
import time
import difflib
import os
from collections import OrderedDict
from pprint import pprint
import warnings
import importlib
from settings import CONFIGS
import tensorflow as tf
warnings.filterwarnings("ignore")
import gym
try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None
import numpy as np
import torch
import yaml

import wandb

try:
    import highway_env
except ImportError:
    highway_env = None
from mpi4py import MPI

from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import  make_atari_env_with_log_monitor
from stable_baselines.common.vec_env import VecFrameStack, VecNormalize, DummyVecEnv, VecEnv
from stable_baselines.ddpg import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.bench import Monitor
from stable_baselines.ppo2.ppo2 import constfn
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.gail import generate_expert_traj_mujoco
from stable_baselines.td3.lfd_envs import AbsorbingWrapper 

from utils import make_env_with_log_monitor, ALGOS, linear_schedule, get_wrapper_class
from utils.noise import LinearNormalActionNoise

best_mean_reward, n_steps = -np.inf, 0

def is_mujoco(env):
    for name in ['Hopper', 'Half', 'Walker', 'Reacher', 'Ant', 'Humanoid', 'InvertedDoublePendulum', 'Pusher', 'Swimmer', 'Thrower', 'Striker']:
        if name in env:
            return True
    return False

def check_if_atari(env_id):
    is_atari = False
    no_skip = False
    for game in ['Gravitar', 'MontezumaRevenge', 'Pitfall', 'Qbert', 'Pong']:
        if game in env_id:
            is_atari = True
            if 'NoFrameskip' in env_id:
                no_skip = True
            return is_atari, no_skip

def a2c_callback(log_dir, max_score, mode, R_random=None, R_expert=None):
    def callback(_locals, _globals):
        """
        Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
        :param _locals: (dict)
        :param _globals: (dict)
        """
        global n_steps, best_mean_reward
        # Print stats every 500 calls
        if (n_steps + 1) % 500 == 0:
            # Evaluate policy training performance from Monitor logs
            x, y = ts2xy(load_results(log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = float(np.mean(y[-100:]))
                current_step = int(x[-1])
                print(current_step, 'timesteps')
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                    best_mean_reward, mean_reward
                ))

                # ---- Stable-Baselines logger + W&B: mean reward ----
                self_ = _locals.get('self', None)

                # Log to SB/TF logger so it shows in TensorBoard
                if self_ is not None and hasattr(self_, "logger"):
                    self_.logger.record("eval/mean_reward_100ep", mean_reward)

                # Also log explicitly to W&B (in addition to TB sync)
                try:
                    if wandb.run is not None:
                        wandb.log(
                            {"eval/mean_reward_100ep": mean_reward},
                            step=current_step,
                        )
                except Exception:
                    # Don't let W&B errors kill training
                    pass

                # ---- Normalized score logging (if baselines are available) ----
                if (R_random is not None) and (R_expert is not None):
                    denom = (R_expert - R_random)
                    if denom != 0 and np.isfinite(denom):
                        norm_score = (mean_reward - R_random) / denom
                        if np.isfinite(norm_score):
                            print("[Norm] mean_reward = {:.2f}, normalized_score = {:.3f}".format(
                                mean_reward, norm_score
                            ))

                            # TB
                            if self_ is not None and hasattr(self_, "logger"):
                                self_.logger.record("eval/normalized_score", norm_score)

                            # W&B
                            try:
                                if wandb.run is not None:
                                    wandb.log({"eval/normalized_score": norm_score}, step=current_step)
                            except Exception:
                                pass
                        else:
                            print("[Norm] Skipping normalized_score (non-finite):", norm_score)
                    else:
                        print("[Norm] Skipping normalized_score (bad denom): R_expert-R_random =", denom)
                else:
                    if R_expert is None:
                        print("[Norm] Not logging normalized_score: R_expert is None (expert returns not loaded).")

                # ---- New best model: save with timestep + keep best_model.pkl ----
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward

                    if self_ is not None:
                        # Keep old behavior for eval scripts:
                        plain_path = os.path.join(log_dir, 'best_model.pkl')
                        # Also save a non-overwriting checkpoint with timesteps in name:
                        step_path = os.path.join(log_dir, 'best_model_{}steps.pkl'.format(current_step))

                        print("Saving new best model:")
                        print(" -", plain_path)
                        print(" -", step_path)

                        self_.save(plain_path)
                        self_.save(step_path)

                # Optional early stopping based on a target score
                if 'train' in mode and mean_reward > max_score:
                    print("Stop training.")
                    return False
        n_steps += 1
        return True
    return callback

def load_expert_hyperparams(args):
    with open('../hyperparams/{}.yml'.format(args.algo), 'r') as f:
        hyperparams_dict = yaml.safe_load(f)
        if env_id in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[env_id]
        elif is_atari:
            hyperparams = hyperparams_dict['atari']
        else:
            raise ValueError("Hyperparameters not found for {}-{}".format(args.algo, env_id))

    # Sort hyperparams that will be saved
    saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

    algo_ = args.algo

    if args.verbose > 0:
        pprint(saved_hyperparams)
    return hyperparams, saved_hyperparams


def initiate_hyperparams(args, hyperparams):
    n_envs = hyperparams.get('n_envs', 1)

    if args.verbose > 0:
        print("Using {} environments".format(n_envs))

    # Create learning rate schedules for ppo2 and sac
    if args.algo in ["ppo2", "sac", "td3", "dac", "sail", "dualsail", "bcq", "sdice", "dice"]:
        for key in ['learning_rate', 'cliprange', 'cliprange_vf']:
            if key not in hyperparams:
                continue
            if isinstance(hyperparams[key], str):
                schedule, initial_value = hyperparams[key].split('_')
                initial_value = float(initial_value)
                hyperparams[key] = linear_schedule(initial_value)
            elif isinstance(hyperparams[key], (float, int)):
                # Negative value: ignore (ex: for clipping)
                if hyperparams[key] < 0:
                    continue
                hyperparams[key] = constfn(float(hyperparams[key]))
            else:
                raise ValueError('Invalid value for {}: {}'.format(key, hyperparams[key]))
    return hyperparams

def create_env_2(args, hyperparams, n_envs, env_wrapper=None, normalize=False, sparse_wrapper=None, normalize_kwargs={}):
    #
    if is_atari:
        if args.verbose > 0:
            print("Using Atari wrapper")
        #env = make_atari_env(env_id, num_env=n_envs, seed=args.seed)
        env = make_atari_env_with_log_monitor(env_id, log_dir=args.log_dir, num_env=n_envs, seed=args.seed)
        # Frame-stacking with 4 frames
        env = VecFrameStack(env, n_stack=4)
    elif args.algo in ['dqn', 'ddpg', 'dqnrrs']:
        env = gym.make(env_id)
        env.seed(args.seed)
        env = Monitor(env, args.log_dir, allow_early_resets=True)
        ## add sparse environment wrapper after the monitor wrapper
        #env = CartPoleWrapper(env)
        if env_wrapper is not None:
            print("Using Predefined Env Wrapper")
            env = env_wrapper(env)
        env = DummyVecEnv([lambda: env])
    else: # ppo2, trpo
        if env_wrapper is not None:
            print("Using Predefined Env Wrapper")
        if n_envs == 1:
            env = make_env_with_log_monitor(
                env_id,
                0,
                args.seed,
                log_dir=args.log_dir,
                wrapper_class=[env_wrapper,AbsorbingWrapper] if args.algo in ['dac'] else [env_wrapper])
            env = DummyVecEnv([env])
        else:
            # On most env, SubprocVecEnv does not help and is quite memory hungry
            env = DummyVecEnv([
                make_env_with_log_monitor(
                    env_id,
                    i,
                    args.seed,
                    log_dir=args.log_dir,
                    wrapper_class=[env_wrapper,AbsorbingWrapper] if args.algo in ['dac'] else [env_wrapper],
                    sparse_wrapper=sparse_wrapper) for i in range(n_envs)])
        if normalize:
            if args.verbose > 0:
                if len(normalize_kwargs) > 0:
                    print("Normalization activated: {}".format(normalize_kwargs))
                else:
                    print("Normalizing input and reward")
            env = VecNormalize(env, **normalize_kwargs)
    # Optional Frame-stacking
    if hyperparams.get('frame_stack', False):
        n_stack = hyperparams['frame_stack']
        env = VecFrameStack(env, n_stack)
        print("Stacking {} frames".format(n_stack))
        del hyperparams['frame_stack']
    return env, hyperparams


def create_test_env(args, hyperparams):
    env_wrapper = get_wrapper_class(hyperparams)
    n_envs = hyperparams.get('n_envs', 1)
    print("Creating Testing environment for {} ....".format(args.env))

    if env_wrapper is not None:
        print("Using Predefined Env Wrapper")
    logdir = os.path.join(args.log_dir, 'test')
    if n_envs == 1:
        env = make_env_with_log_monitor(
            env_id,
            0,
            args.seed,
            log_dir=logdir, 
            wrapper_class=[env_wrapper,AbsorbingWrapper] if args.algo in ['dac'] else [env_wrapper])
        env = env()
    return env

def create_env(args, hyperparams, n_timesteps):
    normalize = False
    normalize_kwargs = {}
    if 'normalize' in hyperparams.keys():
        normalize = hyperparams['normalize']
        if isinstance(normalize, str):
            normalize_kwargs = eval(normalize)
            normalize = True
        del hyperparams['normalize']

    if 'policy_kwargs' in hyperparams.keys():
        hyperparams['policy_kwargs'] = eval(hyperparams['policy_kwargs'])

    # Delete keys so the dict can be passed to the model constructor
    if 'n_envs' in hyperparams.keys():
        del hyperparams['n_envs']
    del hyperparams['n_timesteps']
    # obtain a class object from a wrapper name string in hyperparams
    # and delete the entry
    env_wrapper = get_wrapper_class(hyperparams)


    n_envs = hyperparams.get('n_envs', 1)
    print("Creating Sparse environment for {} ....".format(args.env))
    sparse_wrapper=None

    env, hyperparams = create_env_2(
        args,
        hyperparams,
        n_envs,
        env_wrapper=env_wrapper,
        normalize=normalize,
        normalize_kwargs=normalize_kwargs)
    # Stop env processes to free memory
    if args.optimize_hyperparameters and n_envs > 1:
        env.close()

    # Parse noise string for DDPG and SAC
    if args.algo in ['ddpg', 'sac', 'td3', "sail", "dualsail", "bcq", "sdice", "dice", "dac"] and hyperparams.get('noise_type') is not None:
        noise_type = hyperparams['noise_type'].strip()
        noise_std = hyperparams['noise_std']
        n_actions = env.action_space.shape[0]
        if 'adaptive-param' in noise_type:
            assert args.algo == 'ddpg', 'Parameter is not supported by SAC'
            hyperparams['param_noise'] = AdaptiveParamNoiseSpec(initial_stddev=noise_std,
                                                                desired_action_stddev=noise_std)
        elif 'normal' in noise_type:
            if 'lin' in noise_type:
                hyperparams['action_noise'] = LinearNormalActionNoise(mean=np.zeros(n_actions),
                                                                      sigma=noise_std * np.ones(n_actions),
                                                                      final_sigma=hyperparams.get('noise_std_final', 0.0) * np.ones(n_actions),
                                                                      max_steps=n_timesteps)
            else:
                hyperparams['action_noise'] = NormalActionNoise(mean=np.zeros(n_actions),
                                                                sigma=noise_std * np.ones(n_actions))
                # hyperparams["batch_action_noise"] = BatchNormalActionNoise(mean=np.zeros(n_actions),
                #                                                 sigma=noise_std*np.ones(n_actions),
                #                                                 hyperparams['kernel_dim'])
        elif 'ornstein-uhlenbeck' in noise_type:
            hyperparams['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                                       sigma=noise_std * np.ones(n_actions))
        else:
            raise RuntimeError('Unknown noise type "{}"'.format(noise_type))
        print("Applying {} noise with std {}".format(noise_type, noise_std))
        del hyperparams['noise_type']
        #del hyperparams['noise_std']
        if 'noise_std_final' in hyperparams:
            del hyperparams['noise_std_final']
    return env, hyperparams, normalize


def eval_mujoco_model(args, model, env, step=50000):
    ###########################
    # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
    deterministic = args.algo in ['dqn', 'ddpg', 'sac', 'her', 'td3', "sail", "dualsail", "bcq", "dice", "sdice", "dac", 'dqnrrs']
    episode_reward = 0.0
    episode_rewards = []
    ep_len = 0
    # For HER, monitor success rate
    obs = env.reset()
    successes = []


    for i in range(step):
        action, _ = model.predict(obs, deterministic=deterministic)
        action_prob = model.action_probability(obs)
        #if i % 1000 == 0:
        #    print(action_prob)
        # Clip Action to avoid out of bound errors
        if isinstance(env.action_space, gym.spaces.Box):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, done, infos = env.step(action)
        if not args.no_render:
            env.render('human')

        ep_len += 1

        n_envs = hyperparams.get('n_envs', 1)
        if n_envs == 1:
            if done and not is_atari and args.verbose > 0:
                x, y = ts2xy(load_results(args.log_dir), 'timesteps')
                if len(x) > 0:
                    mean_reward = np.mean(y[-100:])
                    episode_reward = y[-1]
                    print(x[-1], 'timesteps')
                    print("Length: {}, Mean reward: {:.2f} - Last episode reward: {:.2f}".format(ep_len, mean_reward, episode_reward))
                episode_rewards.append(episode_reward)
                ep_len = 0


    if args.verbose > 0 and len(episode_rewards) > 0:
        print("Mean reward: {:.2f}".format(np.mean(episode_rewards)))

    # Workaround for https://github.com/openai/gym/issues/893
    if not args.no_render:
        if args.n_envs == 1 and 'Bullet' not in env_id and not is_atari and isinstance(env, VecEnv):
            # DummyVecEnv
            # Unwrap env
            while isinstance(env, VecNormalize) or isinstance(env, VecFrameStack):
                env = env.venv
            env.envs[0].env.close()
        else:
            env.close()

def eval_model(args, model, env, step=50000):
    #exp_folder = args.trained_agent.split('.pkl')[0]
    #if normalize:
    #    print("Loading saved running average")
    #    env.load_running_average(exp_folder)

    ###########################
    # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
    deterministic = args.algo in ['dqn', 'ddpg', 'sac', 'her', 'dac', 'td3', 'sail', 'dualsail', 'bcq', "dice", 'sdice', 'dqnrrs']
    episode_reward = 0.0
    episode_rewards = []
    ep_len = 0
    # For HER, monitor success rate
    obs = env.reset()
    successes = []
    for i in range(step):
        action, _ = model.predict(obs, deterministic=deterministic)
        action_prob = model.action_probability(obs)
        if i % 1000 == 0:
            print(action_prob)
        # Random Agent
        # action = [env.action_space.sample()]
        # Clip Action to avoid out of bound errors
        if isinstance(env.action_space, gym.spaces.Box):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, done, infos = env.step(action)
        if not args.no_render:
            env.render('human')

        episode_reward += reward[0]
        ep_len += 1

        n_envs = hyperparams.get('n_envs', 1)
        if n_envs == 1:
            # For atari the return reward is not the atari score
            # so we have to get it from the infos dict
            if is_atari and infos is not None and args.verbose >= 1:
                episode_infos = infos[0].get('episode')
                if episode_infos is not None:
                    print("Atari Episode Score: {:.2f}".format(episode_infos['r']))
                    print("Atari Episode Length", episode_infos['l'])

            if done and not is_atari and args.verbose > 0:
                # NOTE: for env using VecNormalize, the mean reward
                # is a normalized reward when `--norm_reward` flag is passed
                print("Episode Reward: {:.2f}".format(episode_reward))
                print("Episode Length", ep_len)
                episode_infos = infos[0].get('episode')
                if episode_infos is not None:
                    print("Episode Reward: {}".format(episode_infos))
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                ep_len = 0

            # Reset also when the goal is achieved when using HER
            if done or infos[0].get('is_success', False):
                if args.algo == 'her' and args.verbose > 1:
                    print("Success?", infos[0].get('is_success', False))
                # Alternatively, you can add a check to wait for the end of the episode
                # if done:
                obs = env.reset()
                if args.algo == 'her':
                    successes.append(infos[0].get('is_success', False))
                    episode_reward, ep_len = 0.0, 0

    if args.verbose > 0 and len(successes) > 0:
        print("Success rate: {:.2f}%".format(100 * np.mean(successes)))

    if args.verbose > 0 and len(episode_rewards) > 0:
        print("Mean reward: {:.2f}".format(np.mean(episode_rewards)))

    # Workaround for https://github.com/openai/gym/issues/893
    if not args.no_render:
        if args.n_envs == 1 and 'Bullet' not in env_id and not is_atari and isinstance(env, VecEnv):
            # DummyVecEnv
            # Unwrap env
            while isinstance(env, VecNormalize) or isinstance(env, VecFrameStack):
                env = env.venv
            env.envs[0].env.close()
        else:
            env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-pref-critic', action='store_true', help='Use TD3Pref with BPref reward')

    # --- Additive BPref -> discriminator reward (critic unchanged)
    parser.add_argument(
    "--add-pref-to-disc", action="store_true",
    help="Add BPref reward to the discriminator reward; TD3 critic unchanged."
)
    parser.add_argument(
    "--pref-weight", type=float, default=1.0,
    help="Weight on normalized BPref reward when adding to discriminator reward."
)
    parser.add_argument(
    "--pref-norm", type=str, default="zscore",
    choices=["zscore", "iqr-match", "none"],
    help="Normalization for BPref reward before weighting."
)
    parser.add_argument(
    "--pref-clip", type=float, default=5.0,
    help="Clip normalized BPref contribution to [-pref-clip, pref-clip]."
)


    parser.add_argument('--lambda-ref', type=float, default=0.0)
    parser.add_argument('--pref-rm', type=str, default=None)
    parser.add_argument('--pref-expect-obs-dim', type=int, default=17)

        # --- Preference-reweighted teacher buffer (PR-SAIL) ---
    parser.add_argument(
        '--pref-reweight-teacher',
        action='store_true',
        help='Use offline pref reward model to reweight teacher trajectories for discriminator expert batches.'
    )
    parser.add_argument(
        '--pref-beta',
        type=float,
        default=5.0,
        help='Temperature for Boltzmann weighting over trajectory J_phi (softmax over trajectories).'
    )
    parser.add_argument(
        '--pref-promote-quantile',
        type=float,
        default=0.5,
        help='Quantile of current teacher J_phi used as threshold for promoting student trajectories.'
    )
    parser.add_argument(
        '--pref-max-teacher-trajs',
        type=int,
        default=500,
        help='Maximum number of trajectories in the teacher buffer (low-score ones are pruned).'
    )
    # --- NEW: Preference ranking regularizer on discriminator (full episodes) ---
    parser.add_argument(
        '--pref-rank-disc',
        action='store_true',
        help='Enable preference-ranking regularizer on discriminator using full episodes.'
    )
    parser.add_argument(
        '--pref-rank-weight',
        type=float,
        default=0.0,
        help='Weight for preference-ranking loss on discriminator (0 disables).'
    )
    parser.add_argument(
        '--pref-rank-batch-size',
        type=int,
        default=32,
        help='Number of preference pairs per discriminator update (Bp).'
    )

    # --- NEW: soft ranking target from reward-model scores ---
    parser.add_argument('--pref-soft-rank-disc', action='store_true',
        help='Use soft BT target from reward-model scores J (sigma((Jpos-Jneg)/T)) to supervise disc ranking.')
    parser.add_argument('--pref-soft-rank-weight', type=float, default=0.0,
        help='Weight for soft ranking loss (added to discriminator total loss).')
    parser.add_argument('--pref-soft-rank-temp', type=float, default=1.0,
        help='Temperature T for soft preference target: sigma((Jpos-Jneg)/T).')
    

    # ============================================================
    # EXPERIMENT: preference-pair filtering for DISC ranking loss
    # (default OFF; does not affect other runs unless enabled)
    # ============================================================
    parser.add_argument(
        '--pref-pair-filter',
        type=str,
        default='none',
        choices=['none', 'disc_sim_rm_diff', 'disc_opp_rm_diff', 'disc_disagree_rm'],
        help=(
            "Filter preference pairs used for DISC ranking loss. "
            "'none' = random pairs. "
            "'disc_sim_rm_diff' = |ΔD| small but |ΔJ| large. "
            "'disc_opp_rm_diff' = |ΔD| large and |ΔJ| large. "
            "'disc_disagree_rm' = disc ordering disagrees with RM ordering (and both have clear prefs)."
        )
    )
    parser.add_argument(
        '--pref-pair-filter-disc-delta-max',
        type=float,
        default=0.5,
        help="For disc_sim_rm_diff: maximum allowed |ΔJ_disc| for a pair to be accepted."
    )
    parser.add_argument(
        '--pref-pair-filter-rm-delta-min',
        type=float,
        default=2.0,
        help="For disc_sim_rm_diff: minimum required |ΔJ_rm| for a pair to be accepted."
    )
    parser.add_argument(
        '--pref-pair-filter-tries',
        type=int,
        default=2000,
        help="How many random candidate pairs to try to fill a batch before falling back."
    )
    parser.add_argument(
        '--pref-pair-filter-refresh-freq',
        type=int,
        default=5000,
        help="How often (in env steps) to recompute cached episode J_disc values for filtering."
    )
    parser.add_argument(
        '--pref-pair-filter-fallback',
        action='store_true',
        help="If enabled, allow fallback to default random pair sampling when filtering can't fill a batch."
    )

    parser.add_argument(
        '--pref-pair-filter-disc-delta-min',
        type=float,
        default=0.15,
        help="For disc_opp_rm_diff/disc_disagree_rm: minimum required |ΔJ_disc|."
    )


    parser.add_argument('--env', type=str, default="PongNoFrameskip-v4", help='environment ID')
    parser.add_argument('-tb', '--tensorboard-log', help='Tensorboard log dir', default='tb_logs', type=str)
    parser.add_argument('-i', '--trained-agent', help='Path to a pretrained agent to continue training',
default='', type=str)
    parser.add_argument('--algo', help='RL Algorithm', default='dqnrrs',
type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=-1,
type=int)
    parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=1000,
                        type=int)
    parser.add_argument('-f', '--log-folder', help='Log folder', type=str, default='logs')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=1)
    parser.add_argument('--n-trials', help='Number of trials for optimizing hyperparameters', type=int, default=10)
    parser.add_argument('--log-dir', help='Log directory', type=str, default='/tmp/logs') # required=True,
    parser.add_argument('-optimize', '--optimize-hyperparameters', action='store_true', default=False,
                        help='Run hyperparameters search')
    parser.add_argument('--n-jobs', help='Number of parallel jobs when optimizing hyperparameters', type=int, default=6)
    parser.add_argument('--n-episodes', help='Number of expert episodes', type=int, default=1)
    parser.add_argument('--no-render', help='If render', default=True)
    parser.add_argument('--sampler', help='Sampler to use when optimizing hyperparameters', type=str,
                        default='skopt', choices=['random', 'tpe', 'skopt'])
    parser.add_argument('--pruner', help='Pruner to use when optimizing hyperparameters', type=str,
                        default='none', choices=['halving', 'median', 'none'])
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                        type=int)
    parser.add_argument('--gym-packages', type=str, nargs='+', default=[], help='Additional external Gym environemnt package modules to import (e.g. gym_minigrid)')
    parser.add_argument('--eval-model-path', type=str, default='', help="model path to be evaluated (when task == 'eval')")
    parser.add_argument(
        '--task',
        type=str,
        default='train',
        help='Figure it out yourself.╭(╯^╰)╮')
    
    parser.add_argument('--disc-dualhead', action='store_true',
                    help='Use dual-head discriminator (main + aux head).')
    parser.add_argument('--disc-aux-weight', type=float, default=0.0,
                    help='Weight for aux head loss. 0 disables aux loss even if dual-head is enabled.')
    parser.add_argument('--disc-use-expert-weights', action='store_true',
                    help='Use expert weights in discriminator expert loss (weighted discriminator). Default off.')

    args = parser.parse_args()

    ##__PREF_INIT__
    rm = None
    if getattr(args, 'use_pref_critic', False):
        from stable_baselines.common.pref_rm_eval import PrefRewardModel
        from stable_baselines.td3.td3_pref import TD3Pref
        rm = PrefRewardModel(args.pref_rm, expect_obs_dim=args.pref_expect_obs_dim)
        from stable_baselines.td3.td3 import TD3 as _TD3
        TD3 = TD3Pref  # swap class only for this run
        print('[SAIL] Using TD3Pref with', args.pref_rm, 'lambda_ref=', args.lambda_ref)
    else:
        print('[SAIL] Using vanilla TD3 (no pref critic)')
    if args.add_pref_to_disc:
        print('[SAIL] Additive BPref->Discriminator enabled (critic unchanged)')

    # build log_dir
    # $prefix/$task/$algo/$env/rank$seed"
    log_dir = os.path.join(args.log_dir, args.task, args.algo, args.env, "rank{}".format(args.seed))
    os.system("mkdir -p {}".format(log_dir))
    args.orig_log_dir = args.log_dir
    args.log_dir = log_dir

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    registered_envs = set(gym.envs.registry.env_specs.keys())

    # If the environment is not found, suggest the closest match
    if args.env not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(args.env, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError('{} not found in gym registry, you maybe meant {}?'.format(env_id, closest_match))

    set_global_seeds(args.seed)

    if args.trained_agent != "":
        #args.trained_agent = os.path.join(args.trained_agent, args.algo, "{}.pkl".format(args.env))
        print(args.trained_agent)
        assert args.trained_agent.endswith('.pkl') and os.path.isfile(args.trained_agent), \
            "The trained_agent must be a valid path to a .pkl file"

    rank = 0
    if MPI.COMM_WORLD.Get_size() > 1:
        print("Using MPI for multiprocessing with {} workers".format(MPI.COMM_WORLD.Get_size()))
        rank = MPI.COMM_WORLD.Get_rank()
        print("Worker rank: {}".format(rank))

        args.seed += rank
        if rank != 0:
            args.verbose = 0
            args.tensorboard_log = ''

    # ------------- W&B + TensorBoard sync (rank 0 only) -------------
    wandb_run = None
    if rank == 0:
        # Decide where TensorBoard logs go
        # If user gives a relative tb dir (e.g., "tb_logs"), join with log_dir
        if os.path.isabs(args.tensorboard_log):
            tb_root = args.tensorboard_log
        else:
            tb_root = os.path.join(args.log_dir, args.tensorboard_log)

        os.environ.setdefault("WANDB_SILENT", "true")

        run_name = f"{args.task}_{args.algo}_{args.env}_seed{args.seed}"
        wandb_run = wandb.init(
            project="SAIL_variants",    # <- rename if you want
            # name=run_name,
            config=vars(args),
            reinit=True
        )

        # This makes ALL TensorBoard scalars (reward, loss, normalized_score, etc.)
        # automatically appear in W&B as the run's history.
        wandb.tensorboard.patch(root_logdir=tb_root)
    # ---------------------------------------------------------------

    # tensorboard_log = os.path.join(args.log_dir, 'tb')

    is_atari = False
    if 'NoFrameskip' in env_id:
        is_atari = True

    print("=" * 10, env_id, "=" * 10)

    #############################################
    # Load hyperparameters & create environment
    #############################################
    hyperparams, saved_hyperparams = load_expert_hyperparams(args)

    # Build a full TensorBoard root dir from log_dir + CLI tb subdir (or use absolute path)
    if os.path.isabs(args.tensorboard_log):
        tb_root = args.tensorboard_log
    else:
        tb_root = os.path.join(args.log_dir, args.tensorboard_log)

    # Tell Stable-Baselines to log TensorBoard summaries here
    hyperparams['tensorboard_log'] = tb_root

    # replace str in hyperparams to be real entities
    hyperparams = initiate_hyperparams(args, hyperparams)
    # Should we overwrite the number of timesteps?
    if args.n_timesteps > 0:
        if args.verbose:
            print("Overwriting n_timesteps with n={}".format(args.n_timesteps))
        n_timesteps = args.n_timesteps
    else:
        n_timesteps = int(hyperparams['n_timesteps'] * 1.0 )
    print(n_timesteps)

    config = CONFIGS[args.env]
    config['n_episodes'] = args.n_episodes
    config['sparse'] = False

    config['disc_dualhead'] = bool(getattr(args, 'disc_dualhead', False))
    config['disc_aux_weight'] = float(getattr(args, 'disc_aux_weight', 0.0))
    config['disc_use_expert_weights'] = bool(getattr(args, 'disc_use_expert_weights', False))

    # >>> Additive BPref->Disc settings (only used if --add-pref-to-disc is set)
    config['add_pref_to_disc'] = bool(args.add_pref_to_disc)

    # >>> New: preference-reweighted teacher buffer flag
    config['pref_reweight_teacher'] = bool(getattr(args, 'pref_reweight_teacher', False))

    # >>> NEW: preference ranking regularizer for discriminator
    config['pref_rank_disc'] = bool(getattr(args, 'pref_rank_disc', False))
    config['pref_rank_weight'] = float(getattr(args, 'pref_rank_weight', 0.0))
    config['pref_rank_batch_size'] = int(getattr(args, 'pref_rank_batch_size', 32))

    # >>> EXPERIMENT: preference-pair filtering for DISC ranking loss
    config['pref_pair_filter'] = str(getattr(args, 'pref_pair_filter', 'none'))
    config['pref_pair_filter_disc_delta_max'] = float(getattr(args, 'pref_pair_filter_disc_delta_max', 0.5))
    config['pref_pair_filter_rm_delta_min'] = float(getattr(args, 'pref_pair_filter_rm_delta_min', 2.0))
    config['pref_pair_filter_tries'] = int(getattr(args, 'pref_pair_filter_tries', 2000))
    config['pref_pair_filter_refresh_freq'] = int(getattr(args, 'pref_pair_filter_refresh_freq', 5000))
    config['pref_pair_filter_fallback'] = bool(getattr(args, 'pref_pair_filter_fallback', False))
    config['pref_pair_filter_disc_delta_min'] = float(getattr(args, 'pref_pair_filter_disc_delta_min', 0.30))

    config['pref_soft_rank_disc'] = bool(args.pref_soft_rank_disc)
    config['pref_soft_rank_weight'] = float(args.pref_soft_rank_weight)
    config['pref_soft_rank_temp'] = float(args.pref_soft_rank_temp)

    # Common pref-RM config fields (used by add-pref-to-disc AND pref-reweight-teacher)
    # We only expect SAIL code to actually *use* them when the corresponding flags are True.
    # Common pref-RM config fields (used by add-pref-to-disc AND pref-reweight-teacher AND rank/filter)
    need_pref_rm = (
        config.get('add_pref_to_disc', False)
        or config.get('pref_reweight_teacher', False)
        or config.get('pref_rank_disc', False)
        or config.get('pref_soft_rank_disc', False)
        or (str(getattr(args, 'pref_pair_filter', 'none')) != 'none')
    )

    if need_pref_rm:
        if args.pref_rm is None:
            raise ValueError(
                "A preference RM is required (either because a pref feature is enabled or "
                "--pref-pair-filter != none), but --pref-rm was not provided."
            )

        config['pref_rm_path'] = args.pref_rm
        config['pref_expect_obs_dim'] = args.pref_expect_obs_dim
        config['pref_norm'] = args.pref_norm          # 'zscore' | 'iqr-match' | 'none'
        config['pref_clip'] = args.pref_clip          # e.g., 5.0

    # Only additive BPref->Disc uses these:
    if config['add_pref_to_disc']:
        config['pref_weight'] = args.pref_weight

    # Only pref-reweighted-teacher uses these:
    if config['pref_reweight_teacher']:
        config['pref_beta'] = args.pref_beta
        config['pref_promote_quantile'] = args.pref_promote_quantile
        config['pref_max_teacher_trajs'] = args.pref_max_teacher_trajs

    # Push SAIL config into W&B config (rank 0 only)
    if rank == 0 and wandb_run is not None:
        wandb_run.config.update(config, allow_val_change=True)

    #args.log_dir = args.log_dir.replace('eval-bc', 'eval-bc-episode-{}'.format(config['n_episodes']))  
    os.makedirs(args.log_dir, exist_ok=True)
    env, hyperparams, normalize = create_env(args, hyperparams, n_timesteps)

    ######################################
    # build a new model or a trained agent
    ######################################

    policy = hyperparams['policy']
    del hyperparams['policy']


    hyperparams['using_gail'] = 'gail' in args.task or 'airl' in args.task
    if args.algo in ["sail", "dac", "sdice"]:
        hyperparams['using_return'] = 'return' in args.task
        hyperparams['bc_expert'] = 'expert' in args.task
        hyperparams['lfd'] = 'lfd' in args.task
        hyperparams['adaptive'] = 'adaptive' in args.task
        hyperparams['max_episode_n'] = args.n_episodes
        hyperparams['n_jobs'] = args.n_jobs
        hyperparams['max_episode_n'] = args.n_episodes
        if 'explore' in args.task:
            if 'uniform' in args.task:
                hyperparams['explore_mode'] = "uniform"
            else:
                hyperparams['explore_mode'] = "uniform"
        else:
            hyperparams['explore_mode'] = None
    if args.algo in ["sail"]:
        hyperparams['sparse_reward'] = 'sparse' in args.task
        hyperparams['norm_sparse_reward'] = 'sparse' in args.task #and 'norm' in args.task

    if args.algo in ["sac"]:
        hyperparams['adaptive'] = 'adaptive' in args.task
        hyperparams['lfd'] = 'lfd' in args.task
        hyperparams['n_jobs'] = args.n_jobs
        hyperparams['shaping_mode'] = args.task


    config['gail'] = hyperparams['using_gail']
    config['shaping_mode'] = args.task

    ##################
    # get path to expert demonstration data
    ##################
    data_save_dir = os.path.join("../../teacher_dataset", "expert_data_no_img_{}_scores_{}_episodes_{}".format(args.env.split('-')[0], config['optimal_score'], config['n_episodes']))
    data_save_dir = '{}.npz'.format(data_save_dir)
    print("Demo Data save in : {}".format(data_save_dir))

        # === Normalization baselines ===
    # Hardcode R_random for now (from your random-policy eval; adjust later if you want)
    R_RANDOM = -270.0

    # Load R_expert from the expert NPZ you are already using
    R_expert = None
    try:
        exp_data = np.load(data_save_dir)
        if "episode_returns" in exp_data:
            R_expert = float(exp_data["episode_returns"].mean())
            print("[Norm] Loaded expert returns from {} -> R_expert = {:.2f}".format(
                data_save_dir, R_expert
            ))
        else:
            print("[Norm] WARNING: 'episode_returns' not found in {}; normalized logging will be disabled.".format(
                data_save_dir
            ))
    except FileNotFoundError:
        print("[Norm] WARNING: expert file {} not found; normalized logging will be disabled.".format(
            data_save_dir
        ))

    # ---- Change 2: W&B debug / summary fields (rank 0 only) ----
    if rank == 0 and wandb_run is not None:
        wandb_run.summary["R_random"] = float(R_RANDOM)
        wandb_run.summary["R_expert_loaded"] = (R_expert is not None)
        if R_expert is not None:
            wandb_run.summary["R_expert"] = float(R_expert)
            wandb_run.summary["R_expert_minus_R_random"] = float(R_expert - R_RANDOM)

    #####################################
    # build model
    #####################################

    # Train an agent from scratch
    policy = 'MlpPolicy'

    policy_kwargs = dict(act_fun=tf.nn.tanh, layers=[64, 64, 64])
    kwargs = {}
    if args.log_interval > -1:
        kwargs = {'log_interval': args.log_interval}
    if 'test' in args.task:
        test_env = create_test_env(args, hyperparams)
    else:
        test_env = None
    if 'env_wrapper' in hyperparams.keys():
        del hyperparams['env_wrapper']

    # model = ALGOS[args.algo](
    #     policy,
    #     env=env,
    #     verbose=args.verbose,
    #     expert_data_path=data_save_dir,
    #     _init_setup_model=False,
    #     **hyperparams)
    model_kwargs = dict(
    env=env,
    verbose=args.verbose,
    expert_data_path=data_save_dir,
    **hyperparams
    )

    if args.algo in ["sail", "dualsail"]:   # add other SAIL-derived ones you have
        model_kwargs["_init_setup_model"] = False

    model = ALGOS[args.algo](policy, **model_kwargs)
    print("Model buit successfully.")
    ##############################################################
    # begin learning, saving best model through each call back
    ##############################################################
    if 'eval' in args.task:
        model_path = args.task.replace('eval-', '')
        args.eval_model_path=os.path.join(args.orig_log_dir, model_path, args.algo, args.env,
                                          "rank{}".format(args.seed),
                                          "best_model.pkl")  # this model is trained from scratch
        print('*'*40 + args.eval_model_path)

        model_file = args.eval_model_path #os.path.join(args.eval_model_path, 'best_model.pkl')
        assert os.path.isfile(model_file), \
            " The eval_model_path must be a valid path to a pkl or zip file, but no file found at {} ".format(model_file)
        print(model_file)
        model = ALGOS[args.algo].load(
            model_file, env=env,
            verbose=args.verbose)
        if is_mujoco(args.env):
            eval_mujoco_model(args, model, env)
            #eval_model(args, model, env)
        else:
            eval_model(args, model, env)

    elif 'pretrain' == args.task:
        if not os.path.isfile(data_save_dir):
            # if expert data is not ready, generate data first
            model_file = os.path.join("{}/train/{}".format(args.orig_log_dir, args.algo),args.env, "rank{}".format(args.seed), "best_model.pkl")
            pretrain_model = ALGOS[args.algo].load(
                model_file, env=env,
                verbose=args.verbose)

            is_atari = 'NoFrameskip' in args.env
            #generate_expert_traj_pro(
            generate_expert_traj_mujoco(
                pretrain_model,
                env=env,
                save_path=data_save_dir.replace('.npz',''),
                n_episodes=config['n_episodes'] ,
                optimal_score=config['optimal_score']
                )
        print("Demo Data save in : {}".format(data_save_dir))
        env.close()
    elif 'bc' == args.task:
        model.pretrain_with_buffer(
            config,
            args.log_dir,
            n_epochs=100 * config['n_episodes'],
            learning_rate=1e-3,
            adam_epsilon=1e-8,
            val_interval=None)
    else:
        print(config)
        time.sleep(3)
        config['log_dir'] = args.log_dir
        # cb_func = a2c_callback(args.log_dir, config['max_score'], args.task)
        # Pass normalization baselines into the callback
        cb_func = a2c_callback(
            args.log_dir,
            config['max_score'],
            args.task,
            R_random=R_RANDOM,
            R_expert=R_expert,
        )
        model.learn(
            n_timesteps,
            callback=cb_func,
            config=config,
            test_env=test_env,
            **kwargs)
        with open(os.path.join(args.log_dir, 'config.yml'), 'w') as f:
            yaml.dump(saved_hyperparams, f)

        # ------------- Close W&B run (rank 0 only) -------------
    if rank == 0 and wandb_run is not None:
        # Save some nice run-level info
        try:
            wandb_run.summary["R_random"] = R_RANDOM
            if R_expert is not None:
                wandb_run.summary["R_expert"] = R_expert
            wandb_run.summary["best_mean_reward"] = best_mean_reward
        except NameError:
            # In case some of these are not defined in certain tasks
            pass

        wandb_run.finish()
    # -------------------------------------------------------










