import os, argparse, time, json, numpy as np
import gym

# SB TF1
from stable_baselines import TD3
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise

def is_gym_v26_step(ret):
    # (obs, reward, terminated, truncated, info)
    return isinstance(ret, tuple) and len(ret) == 5 and isinstance(ret[2], (bool, np.bool_, np.bool8))

def is_gym_v26_reset(ret):
    # (obs, info)
    return isinstance(ret, tuple) and len(ret) == 2

def rollouts_to_npz(env_id, model, n_trajs, max_ep_len=1000, deterministic=True):
    env = gym.make(env_id)
    obs_list, next_obs_list, act_list, rew_list, done_list = [], [], [], [], []
    ep_returns, ep_lengths, ep_ids = [], [], []

    for ep in range(n_trajs):
        ret = env.reset()
        obs = ret[0] if is_gym_v26_reset(ret) else ret
        done, ep_ret, ep_len = False, 0.0, 0
        while not done and ep_len < max_ep_len:
            action, _ = model.predict(obs, deterministic=deterministic)
            step_ret = env.step(action)
            if is_gym_v26_step(step_ret):
                next_obs, reward, terminated, truncated, info = step_ret
                done_flag = bool(terminated or truncated)
            else:
                next_obs, reward, done_flag, info = step_ret
            obs_list.append(np.asarray(obs, dtype=np.float32))
            next_obs_list.append(np.asarray(next_obs, dtype=np.float32))
            act_list.append(np.asarray(action, dtype=np.float32))
            rew_list.append(np.float32(reward))
            done_list.append(bool(done_flag))
            ep_ids.append(ep)
            ep_ret += float(reward)
            ep_len += 1
            obs = next_obs
            if done_flag: break

        ep_returns.append(ep_ret)
        ep_lengths.append(ep_len)

    env.close()

    # Pack arrays in the "new" SB-TF1 dataset style
    observations      = np.asarray(obs_list, dtype=np.float32)
    next_observations = np.asarray(next_obs_list, dtype=np.float32)
    actions           = np.asarray(act_list, dtype=np.float32)
    rewards           = np.asarray(rew_list, dtype=np.float32)
    dones             = np.asarray(done_list, dtype=bool)
    episode_returns   = np.asarray(ep_returns, dtype=np.float32)
    episode_lengths   = np.asarray(ep_lengths, dtype=np.int32)
    episode_ids       = np.asarray(ep_ids, dtype=np.int32)

    return dict(
        observations=observations,
        next_observations=next_observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        episode_ids=episode_ids,
        coarse_episode_returns=episode_returns.astype(np.int32)  # convenience, matches some of your older files
    )

def train_teacher(env_id, total_episodes, seed, policy_layers=(400,300), noise_sigma=0.1):
    # Convert episodes -> steps with HalfCheetah's usual 1000 step horizon
    total_timesteps = int(total_episodes) * 1000

    env = gym.make(env_id)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_sigma*np.ones(n_actions))

    policy_kwargs = dict(layers=list(policy_layers))
    model = TD3(
        MlpPolicy, env,
        policy_kwargs=policy_kwargs,
        buffer_size=1_000_000,
        batch_size=256,
        learning_rate=1e-3,
        action_noise=action_noise,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=0,
        seed=seed
    )

    model.learn(total_timesteps=total_timesteps)
    env.close()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+", required=True,
                    help="One or more teacher tags (e.g. 50 150 300 500). Used for naming and also as training-episode budget by default.")
    ap.add_argument("--n_trajs", type=int, default=4, help="How many trajectories to record per teacher.")
    ap.add_argument("--save_dir", type=str, required=True, help="Where to store .npz files.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env_id", type=str, default="HalfCheetah-v2")
    ap.add_argument("--train_episodes", type=int, default=None,
                    help="If set, overrides training budget for ALL teachers; otherwise each tag uses its own value.")
    ap.add_argument("--noise_sigma", type=float, default=0.1)
    ap.add_argument("--policy_layers", type=int, nargs="+", default=[400,300])
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    meta = dict(env_id=args.env_id, seed=args.seed,
                n_trajs=args.n_trajs, episodes_tags=args.episodes,
                train_episodes=args.train_episodes,
                noise_sigma=args.noise_sigma, policy_layers=args.policy_layers,
                created=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(args.save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    for tag in args.episodes:
        train_eps = args.train_episodes if args.train_episodes is not None else tag
        print(f"\n== Teacher tag {tag} :: training for {train_eps} episodes, seed {args.seed} ==")

        model = train_teacher(args.env_id, total_episodes=train_eps,
                              seed=args.seed, policy_layers=tuple(args.policy_layers),
                              noise_sigma=args.noise_sigma)

        data = rollouts_to_npz(args.env_id, model, n_trajs=args.n_trajs, max_ep_len=1000, deterministic=True)

        base = f"expert_data_no_img_HalfCheetah_scores_{tag}_episodes_{args.n_trajs}_e{args.seed:03d}.npz"
        out_path = os.path.join(args.save_dir, base)
        np.savez_compressed(out_path, **data)

        R = data["episode_returns"].astype(float)
        print(f"[WROTE] {out_path}")
        print(f"  steps={data['observations'].shape[0]}  episodes={R.shape[0]}  "
              f"returns mean/min/max = {R.mean():.2f}/{R.min():.2f}/{R.max():.2f}")

if __name__ == "__main__":
    main()
