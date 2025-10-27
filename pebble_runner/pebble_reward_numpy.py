import numpy as np

class PebbleRewardNP:
    def __init__(self, npz_path):
        z = np.load(npz_path, allow_pickle=True)
        self.W1=z['W1']; self.b1=z['b1']
        self.W2=z['W2']; self.b2=z['b2']
        self.W3=z['W3']; self.b3=z['b3']
        self.mu=z['mu']; self.sigma=z['sigma']
        self.obs_dim=int(z['obs_dim']); self.act_dim=int(z['act_dim'])
    def _relu(self,x): return np.maximum(0.0, x)
    def _tanh(self,x): return np.tanh(x)
    def r_step(self, obs_np, act_np):
        x = np.concatenate([obs_np.astype(np.float32), act_np.astype(np.float32)], axis=-1)
        x = (x - self.mu) / np.maximum(self.sigma, 1e-6)
        h1 = self._relu(x @ self.W1.T + self.b1)
        h2 = self._relu(h1 @ self.W2.T + self.b2)
        r  = self._tanh(h2 @ self.W3.T + self.b3)
        return r.astype(np.float32).reshape(-1,1)
