import os, glob
import numpy as np, torch

class _NPZPrefRewardModel:
    def __init__(self, path, device="cpu", expect_obs_dim=18, norm="zscore"):
        ckpt = np.load(path, allow_pickle=True)
        self.expect_obs_dim = expect_obs_dim
        self.norm = norm
        self.device = torch.device(device)
        self.layers = []
        i = 0
        while f"W{i}" in ckpt or f"W{i}_0" in ckpt:
            W = ckpt.get(f"W{i}", ckpt.get(f"W{i}_0"))
            b = ckpt.get(f"b{i}", ckpt.get(f"b{i}_0"))
            self.layers.append((torch.tensor(W, dtype=torch.float32),
                                torch.tensor(b, dtype=torch.float32)))
            i += 1
        self.mu    = torch.tensor(ckpt.get("mu", 0.0), dtype=torch.float32)
        self.sigma = torch.tensor(ckpt.get("sigma", 1.0), dtype=torch.float32)
        self.layers = [(W.to(self.device), b.to(self.device)) for W,b in self.layers]
        self.mu, self.sigma = self.mu.to(self.device), self.sigma.to(self.device)

    def _fixdim(self, x):
        D = self.expect_obs_dim
        if x.shape[1] == D - 1:
            t = torch.linspace(0.0, 1.0, steps=x.shape[0], device=x.device).unsqueeze(1)
            x = torch.cat([x, t], dim=1)
        elif x.shape[1] != D:
            if x.shape[1] < D:
                x = torch.cat([x, torch.zeros(x.shape[0], D - x.shape[1], device=x.device)], dim=1)
            else:
                x = x[:, :D]
        return x

    @torch.no_grad()
    def reward(self, obs_np, action_np=None, next_obs_np=None):
        x = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        x = self._fixdim(x)
        for W, b in self.layers[:-1]:
            x = torch.tanh(x @ W + b)
        W, b = self.layers[-1]
        r = (x @ W + b).squeeze(-1)
        return r.cpu().numpy()

class _BPrefTorchScriptReward:
    """
    Supports TorchScript .pt models with signatures:
      f(obs) or f(obs, act) or f(obs, act, next_obs).
    If a directory is passed, loads all 'reward_model*.pt' and averages outputs.
    """
    def __init__(self, path_or_dir, device="cpu", expect_obs_dim=17):
        self.device = torch.device(device)
        if os.path.isdir(path_or_dir):
            files = sorted(glob.glob(os.path.join(path_or_dir, "reward_model*.pt")))
            if not files:
                raise FileNotFoundError(f"No reward_model*.pt in {path_or_dir}")
        else:
            if not os.path.exists(path_or_dir):
                raise FileNotFoundError(path_or_dir)
            files = [path_or_dir]
        self.mods = []
        for f in files:
            m = torch.jit.load(f, map_location=self.device)
            m.eval()
            self.mods.append(m)
        self.expect_obs_dim = expect_obs_dim

    def _fixdim(self, x):
        D = self.expect_obs_dim
        if x.shape[1] == D - 1:
            t = torch.linspace(0.0, 1.0, steps=x.shape[0], device=x.device).unsqueeze(1)
            x = torch.cat([x, t], dim=1)
        elif x.shape[1] != D:
            if x.shape[1] < D:
                x = torch.cat([x, torch.zeros(x.shape[0], D - x.shape[1], device=x.device)], dim=1)
            else:
                x = x[:, :D]
        return x

    @torch.no_grad()
    def reward(self, obs_np, action_np=None, next_obs_np=None):
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        obs = self._fixdim(obs)
        act = torch.tensor(action_np, dtype=torch.float32, device=self.device) if action_np is not None else None
        nxt = torch.tensor(next_obs_np, dtype=torch.float32, device=self.device) if next_obs_np is not None else None

        outs = []
        for m in self.mods:
            try:
                r = m(obs)
            except Exception:
                if act is None:
                    raise
                try:
                    r = m(obs, act)
                except Exception:
                    if nxt is None:
                        raise
                    r = m(obs, act, nxt)
            outs.append(r.reshape(-1))
        r = torch.stack(outs, dim=0).mean(0)
        return r.cpu().numpy()

class PrefRewardModel:
    """Factory wrapper. Use .reward(obs, act=None, next_obs=None) -> np.ndarray"""
    def __init__(self, path, device="cpu", expect_obs_dim=17, norm="none"):
        if (os.path.isdir(path) or path.endswith(".pt")):
            self.impl = _BPrefTorchScriptReward(path, device=device, expect_obs_dim=expect_obs_dim)
        elif path.endswith(".npz"):
            self.impl = _NPZPrefRewardModel(path, device=device, expect_obs_dim=expect_obs_dim, norm=norm)
        else:
            raise ValueError(f"Unrecognized reward model path: {path}")

    def reward(self, obs_np, action_np=None, next_obs_np=None):
        return self.impl.reward(obs_np, action_np, next_obs_np)
