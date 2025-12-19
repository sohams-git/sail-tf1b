import numpy as np

def normalized_score(R_policy: float, R_random: float, R_expert: float) -> float:
    """
    0 -> random policy
    1 -> expert demos
    >1 -> better than expert
    """
    return (R_policy - R_random) / (R_expert - R_random)


def eval_policy_mean_return(model, env, n_episodes: int = 10):
    """
    Evaluate a stable-baselines policy on a single env (no VecEnv),
    returns (mean_return, std_return).
    """
    returns = []

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            # deterministic eval
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_ret += reward

        returns.append(ep_ret)

    returns = np.array(returns, dtype=np.float32)
    return float(returns.mean()), float(returns.std())
