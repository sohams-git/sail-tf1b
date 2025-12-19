import numpy as np
import gym
import os

from utils import ALGOS, make_env_with_log_monitor

# Directory to store eval monitor logs (can be anything)
EVAL_LOG_DIR = os.path.join(os.path.dirname(__file__), "eval_logs")
os.makedirs(EVAL_LOG_DIR, exist_ok=True)

print("[DEBUG] eval_normalized.py top-level executing")

# We will gradually fill these in next steps
def load_expert_return(npz_path: str) -> float:
    """
    Load the expert demonstration NPZ and return the average episode return.
    Assumes the file has an 'episode_returns' array.
    """
    data = np.load(npz_path)
    if "episode_returns" not in data:
        raise KeyError(f"'episode_returns' not found in {npz_path}")
    episode_returns = data["episode_returns"]
    R_expert = float(episode_returns.mean())
    print(f"[Expert] Loaded {len(episode_returns)} episodes from {npz_path}")
    print(f"[Expert] Average return: {R_expert:.2f}")
    return R_expert

def eval_random_policy(env_id: str, n_episodes: int = 50):
    """
    Evaluate a random policy on the given environment.
    Returns (mean_return, std_return) over n_episodes.
    """
    env = gym.make(env_id)
    returns = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            ep_return += reward

        returns.append(ep_return)

    env.close()
    returns = np.array(returns, dtype=np.float32)
    mean_ret = float(returns.mean())
    std_ret = float(returns.std())

    print(f"[Random] Evaluated {n_episodes} episodes on {env_id}")
    print(f"[Random] Mean return: {mean_ret:.2f}, Std: {std_ret:.2f}")
    return mean_ret, std_ret

# def eval_trained_policy(env_id: str, model_path: str, algo: str, n_episodes: int = 10):
#     """
#     Evaluate a trained SAIL policy on the given environment.
#     Uses the same ALGOS[algo] mapping and env wrapper as the training code.
#     Returns (mean_return, std_return) over n_episodes.

#     If the env observation is 17-dim but the model expects 18-dim,
#     we append a simple time feature t / max_steps to match the training setup.
#     """
#     # --- 1) Build env using the same helper as in training ---
#     env_or_fn = make_env_with_log_monitor(env_id, rank=0, log_dir=EVAL_LOG_DIR)

#     if callable(env_or_fn):
#         env = env_or_fn()   # call the factory to get the real env
#     else:
#         env = env_or_fn

#     print("[DEBUG] eval env observation space:", env.observation_space)

#     # --- 2) Load model using ALGOS[algo] ---
#     if algo not in ALGOS:
#         raise ValueError(f"Unknown algo '{algo}'. Available: {list(ALGOS.keys())}")

#     ModelClass = ALGOS[algo]
#     model = ModelClass.load(model_path)

#     if hasattr(model, "observation_space"):
#         print("[DEBUG] model observation space:", model.observation_space)

#     returns = []
#     is_vec_env = hasattr(env, "num_envs")

#     # Get max episode length for time feature
#     max_steps = getattr(getattr(env, "spec", None), "max_episode_steps", 1000)

#     for ep in range(n_episodes):
#         obs = env.reset()
#         ep_return = 0.0
#         done = False
#         t = 0  # time step counter

#         while not done:
#             # --- build model input obs_in with possible time-feature ---
#             if is_vec_env:
#                 # obs shape likely (n_env, obs_dim)
#                 obs_in = obs
#                 if isinstance(obs, np.ndarray) and obs.ndim == 2 and obs.shape[1] == 17:
#                     time_feat = float(t / max_steps)
#                     time_col = np.full((obs.shape[0], 1), time_feat, dtype=np.float32)
#                     obs_in = np.concatenate([obs, time_col], axis=1)
#             else:
#                 # single env
#                 obs_in = obs
#                 if isinstance(obs, np.ndarray) and obs.ndim == 1 and obs.shape[0] == 17:
#                     time_feat = float(t / max_steps)
#                     obs_in = np.concatenate([obs, np.array([time_feat], dtype=np.float32)], axis=0)

#             # --- model predicts on obs_in (18-dim if needed) ---
#             action, _ = model.predict(obs_in, deterministic=True)
#             obs, reward, done, info = env.step(action)

#             if is_vec_env:
#                 # reward/done may be arrays; take first env
#                 if np.isscalar(reward):
#                     ep_return += reward
#                 else:
#                     ep_return += reward[0]

#                 if isinstance(done, (np.ndarray, list)):
#                     done_flag = done[0]
#                 else:
#                     done_flag = done

#                 done = done_flag
#             else:
#                 ep_return += reward

#             t += 1

#         returns.append(ep_return)

#     env.close()
#     returns = np.array(returns, dtype=np.float32)
#     mean_ret = float(returns.mean())
#     std_ret = float(returns.std())

#     print(f"[Policy-{algo}] Evaluated {n_episodes} episodes on {env_id}")
#     print(f"[Policy-{algo}] Mean return: {mean_ret:.2f}, Std: {std_ret:.2f}")
#     return mean_ret, std_ret



def eval_trained_policy(env_id: str, model_path: str, algo: str, n_episodes: int = 10):
    """
    Evaluate a trained SAIL policy on a plain HalfCheetah-v2 env.
    We add a simple time feature if obs is 17D to match the model's 18D expectation.
    Returns (mean_return, std_return) over n_episodes.
    """
    env = gym.make(env_id)

    if algo not in ALGOS:
        raise ValueError(f"Unknown algo '{algo}'. Available: list(ALGOS.keys())")

    ModelClass = ALGOS[algo]
    model = ModelClass.load(model_path)

    print("[DEBUG] eval env observation space:", env.observation_space)
    if hasattr(model, "observation_space"):
        print("[DEBUG] model observation space:", model.observation_space)

    returns = []
    max_steps = getattr(getattr(env, "spec", None), "max_episode_steps", 1000)

    for ep in range(n_episodes):
        obs = env.reset()
        ep_return = 0.0
        done = False
        t = 0

        while not done:
            obs_in = obs
            # If env gives 17D obs but model expects 18D, append time feature
            if isinstance(obs, np.ndarray) and obs.ndim == 1 and obs.shape[0] == 17:
                time_feat = float(t / max_steps)
                obs_in = np.concatenate([obs, np.array([time_feat], dtype=np.float32)], axis=0)

            action, _ = model.predict(obs_in, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_return += reward
            t += 1

        returns.append(ep_return)

    env.close()
    returns = np.array(returns, dtype=np.float32)
    mean_ret = float(returns.mean())
    std_ret = float(returns.std())

    print(f"[Policy-{algo}] Evaluated {n_episodes} episodes on {env_id}")
    print(f"[Policy-{algo}] Mean return: {mean_ret:.2f}, Std: {std_ret:.2f}")
    return mean_ret, std_ret

def normalized_score(R_policy: float, R_random: float, R_expert: float) -> float:
    """
    PAIL-style normalization:
      0  -> random policy
      1  -> expert demonstrations
      >1 -> better than expert
    """
    return (R_policy - R_random) / (R_expert - R_random)

def evaluate_normalized(
    env_id: str,
    model_path: str,
    algo: str,
    expert_npz_path: str,
    n_eval_episodes: int = 10,
    n_random_episodes: int = 50,
):
    """
    Full evaluation pipeline:
      1) Load expert dataset to get R_expert
      2) Evaluate random policy to get R_random
      3) Evaluate trained SAIL policy (ALGOS[algo]) to get R_sail
      4) Compute normalized scores
    """
    print("=== Evaluating normalized score ===")
    print(f"Env:         {env_id}")
    print(f"Algo:        {algo}")
    print(f"Model path:  {model_path}")
    print(f"Expert npz:  {expert_npz_path}")
    print("-----------------------------------")

    # 1) Expert: from your NPZ
    R_expert = load_expert_return(expert_npz_path)

    # 2) Random policy on the same env
    R_random, R_random_std = eval_random_policy(env_id, n_episodes=n_random_episodes)

    # 3) Trained SAIL policy (via ALGOS[algo])
    R_sail, R_sail_std = eval_trained_policy(
        env_id=env_id,
        model_path=model_path,
        algo=algo,
        n_episodes=n_eval_episodes,
    )

    # 4) Normalized scores (PAIL-style)
    norm_random = normalized_score(R_random, R_random, R_expert)
    norm_expert = normalized_score(R_expert, R_random, R_expert)
    norm_sail   = normalized_score(R_sail,   R_random, R_expert)

    print("\n=== Results ===")
    print(f"Random: mean={R_random:.2f} ± {R_random_std:.2f}, normalized={norm_random:.3f}")
    print(f"Expert: mean={R_expert:.2f}, normalized={norm_expert:.3f} (should be ~1)")
    print(f"SAIL:   mean={R_sail:.2f} ± {R_sail_std:.2f}, normalized={norm_sail:.3f}")
    print("===============")

    return {
        "R_expert": R_expert,
        "R_random_mean": R_random,
        "R_random_std": R_random_std,
        "R_sail_mean": R_sail,
        "R_sail_std": R_sail_std,
        "norm_random": norm_random,
        "norm_expert": norm_expert,
        "norm_sail": norm_sail,
    }


if __name__ == "__main__":
    env_id = "HalfCheetah-v2"
    algo = "sail"   # because your run used the 'sail' entry in ALGOS

    expert_npz_path = "../../teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_8.npz"

    model_path = "/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/prefrew5_sbatch5000_decW_hc500e8_1000000_s0/gail-lfd-adaptive-dynamic/sail/HalfCheetah-v2/rank0/best_model.pkl"
                  
    results = evaluate_normalized(
        env_id=env_id,
        model_path=model_path,
        algo=algo,
        expert_npz_path=expert_npz_path,
        n_eval_episodes=100,
        n_random_episodes=100,
    )

    print("\nFinal normalized SAIL score:", results["norm_sail"])