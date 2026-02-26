#!/usr/bin/env python3
import argparse
import numpy as np

def find_key(d, candidates):
    for k in candidates:
        if k in d:
            return k
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rm", required=True, help="Path to reward_model_*.pt")
    ap.add_argument("--npz", required=True, help="Teacher dataset .npz")
    ap.add_argument("--expect-obs-dim", type=int, default=-1,
                    help="If <=0, infer from dataset obs shape")
    ap.add_argument("--max-episodes", type=int, default=20)
    ap.add_argument("--print-episodes", type=int, default=5)
    args = ap.parse_args()

    # ---- load RM wrapper used in your SAIL code ----
    from stable_baselines.common.pref_rm_eval import PrefRewardModel

    data = np.load(args.npz, allow_pickle=True)

    # Common keys used by these teacher npz files
    obs_key = find_key(data, ["obs", "observations"])
    act_key = find_key(data, ["acs", "actions"])
    done_key = find_key(data, ["dones", "terminals", "episode_starts"])  # dones is best

    if obs_key is None or act_key is None:
        print("NPZ keys:", list(data.keys()))
        raise KeyError("Could not find obs/actions keys in NPZ. See keys printed above.")

    obs = data[obs_key]
    acs = data[act_key]

    # Done flags
    dones = None
    if done_key is not None and done_key in data:
        dones = data[done_key]
    elif "dones" in data:
        dones = data["dones"]

    # Optional: true episode returns if present
    true_ep_returns = None
    if "episode_returns" in data:
        true_ep_returns = data["episode_returns"]
    elif "ep_returns" in data:
        true_ep_returns = data["ep_returns"]
    elif "returns" in data:
        true_ep_returns = data["returns"]

    # infer obs dim
    if args.expect_obs_dim <= 0:
        if obs.ndim == 2:
            args.expect_obs_dim = obs.shape[1]
        else:
            raise ValueError(f"Unexpected obs shape {obs.shape}; can’t infer obs dim safely.")
        print(f"[Auto] expect_obs_dim inferred from dataset: {args.expect_obs_dim}")

    rm = PrefRewardModel(args.rm, expect_obs_dim=args.expect_obs_dim)
    print(f"[RM] loaded: {args.rm}")
    print(f"[NPZ] obs_key={obs_key} act_key={act_key} done_key={done_key}")
    print(f"[NPZ] obs shape={obs.shape} acs shape={acs.shape}")

    if dones is None:
        raise ValueError(
            "Could not find dones/terminals in NPZ. "
            "Without dones we can’t segment into episodes."
        )

    # ---- segment into episodes ----
    done_idx = np.where(dones.astype(np.float32).reshape(-1) == 1.0)[0]
    if done_idx.size == 0:
        raise ValueError("No done flags found (= no episode boundaries).")

    n_eps_total = len(done_idx)
    n_eps = min(args.max_episodes, n_eps_total)
    print(f"[EP] total episodes in NPZ: {n_eps_total}, evaluating: {n_eps}")

    J_list = []
    L_list = []
    start = 0

    for ep_i, last in enumerate(done_idx[:n_eps]):
        # include transitions start..last inclusive
        obs_ep = obs[start:last+1]
        acs_ep = acs[start:last+1]

        # reward model output per step
        r = rm.reward(obs_ep, acs_ep)
        r = np.asarray(r).reshape(-1)

        J = float(r.sum())
        L = int(len(r))
        J_list.append(J)
        L_list.append(L)

        if ep_i < args.print_episodes:
            tr = None
            if true_ep_returns is not None and ep_i < len(true_ep_returns):
                tr = float(true_ep_returns[ep_i])
            print(f"[ep {ep_i:03d}] len={L:4d}  J_phi(sum)={J:10.3f}"
                  + (f"  true_return={tr:10.3f}" if tr is not None else ""))

        start = last + 1

    J_arr = np.asarray(J_list, dtype=np.float64)
    L_arr = np.asarray(L_list, dtype=np.int32)

    print("\n=== Summary over evaluated episodes ===")
    print(f"J_phi: mean={J_arr.mean():.3f} std={J_arr.std():.3f} min={J_arr.min():.3f} max={J_arr.max():.3f}")
    print(f"len : mean={L_arr.mean():.1f} std={L_arr.std():.1f} min={L_arr.min()} max={L_arr.max()}")

    if true_ep_returns is not None and len(true_ep_returns) >= len(J_arr):
        tr = np.asarray(true_ep_returns[:len(J_arr)], dtype=np.float64)
        # correlation is a very nice quick sanity check
        corr = np.corrcoef(J_arr, tr)[0, 1]
        print(f"corr(J_phi, true_return) = {corr:.4f}")

if __name__ == "__main__":
    main()