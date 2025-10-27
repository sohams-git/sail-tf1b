#!/usr/bin/env python
# Minimal HalfCheetah rollout: use learned reward for training signal, log true env return.
import argparse, numpy as np, gym
from pebble_runner.eval_pref_metrics_py37 import PebbleRewardNP

def adapt_obs(obs: np.ndarray, rm, env) -> np.ndarray:
    """Map env obs to reward-model obs. If rm needs 18 and env gives 17, prepend x-position."""
    o = np.asarray(obs, np.float32).ravel()
    target = rm.obs_dim
    if o.shape[0] == target:
        return o
    # Common HalfCheetah case: env=17D (qpos[1:]+qvel) vs model=18D (qpos[0:]+qvel)
    if o.shape[0] + 1 == target and hasattr(getattr(env, "unwrapped", env), "sim"):
        try:
            xpos = float(env.unwrapped.sim.data.qpos[0])  # missing absolute x-position
            return np.concatenate(([xpos], o), axis=0).astype(np.float32)
        except Exception:
            pass
    # Fallback: pad/truncate using rm.mu (keeps normalized 0 for padded dims)
    out = np.empty((target,), dtype=np.float32)
    k = min(o.shape[0], target)
    out[:k] = o[:k]
    if target > k:
        out[k:] = rm.mu[:target][k:]
    else:
        out = out[:target]
    return out

class PebbleRewardWrapper(gym.Wrapper):
    def __init__(self, env, rm):
        super().__init__(env)
        self.rm = rm
        self._last_obs = None
        self._warned = False
    def reset(self, **kw):
        obs = self.env.reset(**kw)
        if isinstance(obs, tuple) and len(obs) == 2:  # gym>=0.26 style
            obs, _info = obs
        ao = adapt_obs(obs, self.rm, self.env)
        if ao.shape[0] != self.rm.obs_dim and not self._warned:
            print(f"[warn] env obs_dim={obs.shape[-1]} -> adapted={ao.shape[-1]} != rm.obs_dim={self.rm.obs_dim}")
            self._warned = True
        self._last_obs = ao
        return obs
    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:  # gym>=0.26
            obs, true_r, term, trunc, info = out
            done = term or trunc
        else:              # gym 0.21
            obs, true_r, done, info = out
            term = done; trunc = False
        o = self._last_obs.reshape(1, -1)
        a = np.asarray(action, np.float32).reshape(1, -1)
        pref_r = float(self.rm.r_step(o, a)[0, 0])
        info = dict(info); info['true_reward'] = float(true_r)
        self._last_obs = adapt_obs(obs, self.rm, self.env)
        return (obs, pref_r, done, info) if len(out) == 4 else (obs, pref_r, term, trunc, info)

def run(args):
    rm = PebbleRewardNP(args.model)
    env = gym.make(args.env)
    if env.action_space.shape[0] != rm.act_dim:
        raise SystemExit(f"act_dim mismatch: env={env.action_space.shape[0]} rm={rm.act_dim}")
    env_pref = PebbleRewardWrapper(env, rm)

    preds, trues = [], []
    for ep in range(args.episodes):
        obs = env_pref.reset()
        pr_sum = 0.0; tr_sum = 0.0; steps = 0; done = False
        while not done and steps < args.max_steps:
            act = env_pref.action_space.sample()
            out = env_pref.step(act)
            if len(out) == 5:
                obs, pref_r, term, trunc, info = out
                done = term or trunc
            else:
                obs, pref_r, done, info = out
            pr_sum += pref_r
            tr_sum += info.get('true_reward', 0.0)
            steps += 1
        preds.append(pr_sum); trues.append(tr_sum)
        print(f"[{args.env}] ep={ep} steps={steps}  pred_return={pr_sum:.3f}  true_return={tr_sum:.3f}")
    P = np.asarray(preds, np.float32); T = np.asarray(trues, np.float32)
    if len(P) >= 3:
        pear = float(np.corrcoef(P, T)[0,1])
        rP = np.argsort(np.argsort(P)); rT = np.argsort(np.argsort(T))
        spear = float(np.corrcoef(rP, rT)[0,1])
        k, b = np.polyfit(P, T, 1)
        print(f"[summary] episodes={len(P)}  Pearson={pear:.3f}  Spearman={spear:.3f}  fit: T≈{k:.3f}*P+{b:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--env", default="HalfCheetah-v3")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--max-steps", type=int, default=1000)
    args = ap.parse_args()
    run(args)
