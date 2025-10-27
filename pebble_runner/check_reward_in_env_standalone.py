#!/usr/bin/env python
import argparse, numpy as np, gym

class PebbleRewardNP:
    """Minimal NP forwarder for the learned preference reward."""
    def __init__(self, npz_path):
        z = np.load(npz_path, allow_pickle=True)
        self.W1=z['W1']; self.b1=z['b1']; self.W2=z['W2']; self.b2=z['b2']
        self.W3=z['W3']; self.b3=z['b3']; self.mu=z['mu']; self.sigma=z['sigma']
        self.obs_dim=int(z['obs_dim']); self.act_dim=int(z['act_dim'])
    def _fwd(self, X):
        Xn = (X - self.mu) / np.maximum(self.sigma, 1e-6)
        h1 = np.maximum(0, Xn.dot(self.W1.T) + self.b1)
        h2 = np.maximum(0, h1.dot(self.W2.T) + self.b2)
        y  = np.tanh(h2.dot(self.W3.T) + self.b3)
        return y.astype(np.float32)
    def r_step(self, obs, act):
        X = np.concatenate([obs, act], axis=-1).astype(np.float32)
        return self._fwd(X)

def adapt_obs(obs: np.ndarray, rm, env) -> np.ndarray:
    """If env gives 17-D and rm expects 18-D for HalfCheetah, prepend qpos[0]."""
    o = np.asarray(obs, np.float32).ravel()
    if o.shape[0] == rm.obs_dim:
        return o
    if o.shape[0] + 1 == rm.obs_dim and hasattr(getattr(env, "unwrapped", env), "sim"):
        try:
            xpos = float(env.unwrapped.sim.data.qpos[0])
            return np.concatenate(([xpos], o), axis=0).astype(np.float32)
        except Exception:
            pass
    # fallback: pad/truncate with model mean (keeps normalized=0)
    out = np.empty((rm.obs_dim,), dtype=np.float32)
    k = min(o.shape[0], rm.obs_dim)
    out[:k] = o[:k]
    if rm.obs_dim > k: out[k:] = rm.mu[:rm.obs_dim][k:]
    return out

class PrefRewardWrapper(gym.Wrapper):
    def __init__(self, env, rm):
        super().__init__(env)
        self.rm = rm
        self._last_obs = None
        self._warned = False
    def reset(self, **kw):
        out = self.env.reset(**kw)
        obs = out[0] if isinstance(out, tuple) else out
        ao = adapt_obs(obs, self.rm, self.env)
        if ao.shape[0] != self.rm.obs_dim and not self._warned:
            print(f"[warn] env obs={obs.shape[-1]} -> adapted={ao.shape[-1]} != rm.obs_dim={self.rm.obs_dim}")
            self._warned = True
        self._last_obs = ao
        return obs if not isinstance(out, tuple) else (obs, out[1])
    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, true_r, term, trunc, info = out; done = term or trunc
        else:
            obs, true_r, done, info = out; term = done; trunc = False
        o = self._last_obs.reshape(1, -1)
        a = np.asarray(action, np.float32).reshape(1, -1)
        pref_r = float(self.rm.r_step(o, a)[0, 0])
        info = dict(info); info['true_reward'] = float(true_r)
        self._last_obs = adapt_obs(obs, self.rm, self.env)
        return (obs, pref_r, done, info) if len(out) == 4 else (obs, pref_r, term, trunc, info)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--env", default="HalfCheetah-v3")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--max-steps", type=int, default=1000)
    args = ap.parse_args()

    rm = PebbleRewardNP(args.model)
    env = gym.make(args.env)
    if env.action_space.shape[0] != rm.act_dim:
        raise SystemExit(f"act_dim mismatch: env={env.action_space.shape[0]} rm={rm.act_dim}")
    env = PrefRewardWrapper(env, rm)

    preds, trues = [], []
    for ep in range(args.episodes):
        out = env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        pr_sum = 0.0; tr_sum = 0.0; steps = 0; done = False
        while not done and steps < args.max_steps:
            act = env.action_space.sample()
            out = env.step(act)
            if len(out) == 5:
                obs, pref_r, term, trunc, info = out; done = term or trunc
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
    main()
