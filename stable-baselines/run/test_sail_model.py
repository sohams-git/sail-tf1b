import os
import gym
import numpy as np

from utils import ALGOS

# ---- CONFIGURE THESE ----
ENV_ID = "HalfCheetah-v2"
ALGO = "sail"
MODEL_PATH = "/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/prefrew5_sbatch5000_hc500e8_1000000_s0/gail-lfd-adaptive-dynamic/sail/HalfCheetah-v2/rank0/best_model.pkl"
# --------------------------


def run_eval(n_episodes: int = 5):
    # Basic sanity checks
    print("=== TEST SAIL MODEL ===")
    print(f"Env:        {ENV_ID}")
    print(f"Algo:       {ALGO}")
    print(f"Model path: {MODEL_PATH}")
    print("========================")

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    env = gym.make(ENV_ID)

    if ALGO not in ALGOS:
        raise ValueError(f"Unknown algo '{ALGO}'. Available: {list(ALGOS.keys())}")

    ModelClass = ALGOS[ALGO]
    print("[DEBUG] ModelClass from ALGOS[ALGO]:", ModelClass)

    # Load model
    model = ModelClass.load(MODEL_PATH)
    print("[DEBUG] Loaded model type:", type(model))

    print("[DEBUG] Env observation space:", env.observation_space)
    if hasattr(model, "observation_space"):
        print("[DEBUG] Model observation space:", model.observation_space)

    max_steps = getattr(getattr(env, "spec", None), "max_episode_steps", 1000)
    returns = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        t = 0
        ep_ret = 0.0

        while not done:
            obs_in = obs

            # If env gives 17D obs but model expects 18D, append time feature
            if isinstance(obs, np.ndarray) and obs.ndim == 1:
                if hasattr(model, "observation_space"):
                    model_obs_dim = model.observation_space.shape[0]
                    if obs.shape[0] == model_obs_dim - 1:
                        time_feat = float(t / max_steps)
                        obs_in = np.concatenate([obs, np.array([time_feat], dtype=np.float32)], axis=0)

            action, _ = model.predict(obs_in, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_ret += reward
            t += 1

        print(f"[EP {ep}] return = {ep_ret:.2f}")
        returns.append(ep_ret)

    env.close()
    returns = np.array(returns, dtype=np.float32)
    mean_ret = float(returns.mean())
    std_ret = float(returns.std())
    print("\n=== SUMMARY ===")
    print(f"Mean return over {n_episodes} episodes: {mean_ret:.2f}")
    print(f"Std return: {std_ret:.2f}")
    print("===============")


if __name__ == "__main__":
    run_eval(n_episodes=10)
