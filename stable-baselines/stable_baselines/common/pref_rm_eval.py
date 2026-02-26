import os, glob
import numpy as np, torch
import torch.nn as nn

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

class _BPrefStateDictReward:
    """
    Supports plain PyTorch state_dict checkpoints saved as OrderedDict
    with keys like: 0.weight, 0.bias, 2.weight, ...
    Assumes input is concat([obs, act]) and output is scalar reward.
    """
    def __init__(self, path, device="cpu", expect_obs_dim=17):
        self.device = torch.device(device)
        sd = torch.load(path, map_location="cpu")
        if not isinstance(sd, (dict,)):
            raise TypeError(f"Expected state_dict dict/OrderedDict, got {type(sd)}")

        self.sd = sd
        self.expect_obs_dim = expect_obs_dim  # can be -1 (auto) or fixed
        self.net = None
        self.rm_in_dim = int(sd["0.weight"].shape[1])  # total input dim (obs+act)

    def _build_net(self):
        sd = self.sd
        w0 = sd["0.weight"].shape
        w2 = sd["2.weight"].shape
        w4 = sd["4.weight"].shape
        w6 = sd["6.weight"].shape

        h1 = w0[0]; h2 = w2[0]; h3 = w4[0]; out = w6[0]
        assert out == 1, f"Expected scalar reward output, got out_dim={out}"

        net = nn.Sequential(
            nn.Linear(self.rm_in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, 1),
        )
        net.load_state_dict(sd, strict=True)
        net.to(self.device)
        net.eval()
        self.net = net

    @torch.no_grad()
    def reward(self, obs_np, action_np=None, next_obs_np=None):
        if action_np is None:
            raise ValueError("StateDict RM expects action_np (input is [obs, act]).")

        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        act = torch.tensor(action_np, dtype=torch.float32, device=self.device)

        act_dim = int(act.shape[1])
        expected_obs_dim = self.rm_in_dim - act_dim
        if expected_obs_dim <= 0:
            raise ValueError(f"Bad expected_obs_dim={expected_obs_dim} from rm_in_dim={self.rm_in_dim}, act_dim={act_dim}")

        # If user hardcoded expect_obs_dim, respect it (for HC backwards compat)
        # Otherwise auto-slice to what RM actually needs
        if self.expect_obs_dim is None or int(self.expect_obs_dim) <= 0:
            use_obs_dim = expected_obs_dim
        else:
            use_obs_dim = int(self.expect_obs_dim)

        if obs.shape[1] != use_obs_dim:
            obs = obs[:, :use_obs_dim]

        x = torch.cat([obs, act], dim=1)

        # RM expects rm_in_dim exactly; ensure x matches
        if x.shape[1] != self.rm_in_dim:
            # last-resort trim/pad (shouldn’t happen if slicing is right)
            if x.shape[1] > self.rm_in_dim:
                x = x[:, :self.rm_in_dim]
            else:
                pad = torch.zeros((x.shape[0], self.rm_in_dim - x.shape[1]), device=self.device)
                x = torch.cat([x, pad], dim=1)

        if self.net is None:
            self._build_net()

        r = self.net(x).squeeze(-1)
        return r.cpu().numpy()

class PrefRewardModel:
    """Factory wrapper. Use .reward(obs, act=None, next_obs=None) -> np.ndarray"""
    def __init__(self, path, device="cpu", expect_obs_dim=17, norm="none"):
        if os.path.isdir(path):
            # directory => TorchScript ensemble directory (unchanged behavior)
            self.impl = _BPrefTorchScriptReward(path, device=device, expect_obs_dim=expect_obs_dim)

        elif path.endswith(".pt"):
            # could be TorchScript OR state_dict
            try:
                self.impl = _BPrefTorchScriptReward(path, device=device, expect_obs_dim=expect_obs_dim)
            except Exception as e:
                # fallback to state_dict (Hopper case)
                self.impl = _BPrefStateDictReward(path, device=device, expect_obs_dim=expect_obs_dim)

        elif path.endswith(".npz"):
            self.impl = _NPZPrefRewardModel(path, device=device, expect_obs_dim=expect_obs_dim, norm=norm)

        else:
            raise ValueError(f"Unrecognized reward model path: {path}")


    def reward(self, obs_np, action_np=None, next_obs_np=None):
        return self.impl.reward(obs_np, action_np, next_obs_np)
