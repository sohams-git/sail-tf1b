#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torch.nn as nn


def find_key(npz, candidates):
    for k in candidates:
        if k in npz:
            return k
    return None


def build_mlp_from_state_dict(sd, in_dim, activation="relu"):
    """
    Build an MLP matching a Sequential of Linear layers stored as:
      0.weight, 0.bias, 2.weight, 2.bias, 4.weight, 4.bias, 6.weight, 6.bias
    That corresponds to:
      Linear -> Act -> Linear -> Act -> Linear -> Act -> Linear
    We infer hidden sizes from weight shapes.
    """
    # weight shapes: [out_dim, in_dim]
    w0 = sd["0.weight"].shape
    w2 = sd["2.weight"].shape
    w4 = sd["4.weight"].shape
    w6 = sd["6.weight"].shape

    h1 = int(w0[0])
    h2 = int(w2[0])
    h3 = int(w4[0])
    out = int(w6[0])

    assert int(w0[1]) == int(in_dim), f"Mismatch: first layer expects {w0[1]} but in_dim={in_dim}"
    assert out == 1, f"Expected scalar reward output, got out_dim={out}"

    if activation.lower() == "tanh":
        Act = nn.Tanh
    elif activation.lower() == "relu":
        Act = nn.ReLU
    else:
        raise ValueError(f"Unknown activation: {activation} (use relu|tanh)")

    net = nn.Sequential(
        nn.Linear(in_dim, h1),
        Act(),
        nn.Linear(h1, h2),
        Act(),
        nn.Linear(h2, h3),
        Act(),
        nn.Linear(h3, 1),
    )
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rm", required=True, help="path to .pt state_dict (OrderedDict)")
    ap.add_argument("--npz", required=True, help="teacher dataset npz")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-episodes", type=int, default=50)
    ap.add_argument("--print-episodes", type=int, default=10)
    ap.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh"],
                    help="Activation used when training the reward model MLP (most common: relu).")

    # Optional override if you *know* dims
    ap.add_argument("--force-obs-dim", type=int, default=-1,
                    help="If >0, force obs to this dim via obs[:, :force_obs_dim].")
    ap.add_argument("--force-act-dim", type=int, default=-1,
                    help="If >0, force act_dim to this dim via actions[:, :force_act_dim].")

    args = ap.parse_args()
    device = torch.device(args.device)

    # ---------------- Load teacher data ----------------
    data = np.load(args.npz, allow_pickle=True)

    obs_key = find_key(data, ["obs", "observations"])
    act_key = find_key(data, ["acs", "actions"])
    done_key = find_key(data, ["dones", "terminals"])
    start_key = find_key(data, ["episode_starts"])

    if obs_key is None or act_key is None:
        print("NPZ keys:", list(data.keys()))
        raise KeyError("Need obs/observations and actions/acs in NPZ.")

    obs = data[obs_key].astype(np.float32)
    acs = data[act_key].astype(np.float32)

    # Optionally force action dim first (rare, but keep it)
    if args.force_act_dim > 0 and acs.shape[1] != args.force_act_dim:
        print(f"[force] actions_dim {acs.shape[1]} -> {args.force_act_dim} via slice")
        acs = acs[:, :args.force_act_dim]

    act_dim = int(acs.shape[1])

    # Build dones:
    if done_key is not None:
        dones = data[done_key].astype(np.float32).reshape(-1)
    elif start_key is not None:
        episode_starts = data[start_key].astype(np.int32).reshape(-1)
        dones = np.zeros_like(episode_starts, dtype=np.float32)
        if len(dones) > 1:
            dones[:-1] = (episode_starts[1:] == 1).astype(np.float32)
        dones[-1] = 1.0
    else:
        print("NPZ keys:", list(data.keys()))
        raise KeyError("Need dones/terminals OR episode_starts in NPZ.")

    # ---------------- Load RM state_dict ----------------
    sd = torch.load(args.rm, map_location="cpu")
    if not isinstance(sd, dict):
        raise TypeError(f"Expected a state_dict dict/OrderedDict, got {type(sd)}")

    # Infer RM expected input dimension from first layer
    if "0.weight" not in sd:
        raise KeyError("State dict missing '0.weight' (unexpected RM format).")

    rm_in_dim = int(sd["0.weight"].shape[1])         # obs+act
    expected_obs_dim = rm_in_dim - act_dim

    if expected_obs_dim <= 0:
        raise ValueError(f"RM expects in_dim={rm_in_dim}, act_dim={act_dim} => expected_obs_dim={expected_obs_dim} (bad).")

    # Optional manual force for obs dim
    if args.force_obs_dim > 0:
        if obs.shape[1] != args.force_obs_dim:
            print(f"[force] obs_dim {obs.shape[1]} -> {args.force_obs_dim} via slice")
        obs = obs[:, :args.force_obs_dim]
    else:
        # Auto-slice to match RM
        if obs.shape[1] != expected_obs_dim:
            print(f"[AutoSlice] NPZ obs_dim={obs.shape[1]} -> RM expects {expected_obs_dim}. Slicing obs.")
            obs = obs[:, :expected_obs_dim]

    obs_dim = int(obs.shape[1])
    in_dim = obs_dim + act_dim

    if in_dim != rm_in_dim:
        raise ValueError(f"After slicing: obs_dim={obs_dim} act_dim={act_dim} => in_dim={in_dim}, "
                         f"but RM first layer expects {rm_in_dim}.")

    # ---------------- Build RM network ----------------
    net = build_mlp_from_state_dict(sd, in_dim=in_dim, activation=args.activation).to(device)
    net.eval()

    # ---------------- Episode boundaries ----------------
    done_idx = np.where(dones == 1.0)[0]
    n_eps_total = len(done_idx)
    n_eps = min(args.max_episodes, n_eps_total)

    print(f"[info] eps_total={n_eps_total} eval={n_eps} obs_dim={obs_dim} act_dim={act_dim} rm_in_dim={rm_in_dim}")

    # ---------------- Score episodes ----------------
    start = 0
    J_list = []

    for ep_i, last in enumerate(done_idx[:n_eps]):
        obs_ep = obs[start:last + 1]
        acs_ep = acs[start:last + 1]

        x = np.concatenate([obs_ep, acs_ep], axis=1)  # [T, obs+act]

        with torch.no_grad():
            r = net(torch.from_numpy(x).to(device)).cpu().numpy().reshape(-1)

        J = float(r.sum())
        J_list.append(J)

        if ep_i < args.print_episodes:
            print(
                f"[ep {ep_i:03d}] T={len(r):4d}  "
                f"J_phi={J:10.3f}  r(min/mean/max)=({r.min():.3f}/{r.mean():.3f}/{r.max():.3f})"
            )

        start = last + 1

    J = np.asarray(J_list, dtype=np.float64)
    if len(J) > 0:
        print(f"\nJ_phi summary: mean={J.mean():.3f} std={J.std():.3f} min={J.min():.3f} max={J.max():.3f}")
    else:
        print("\nNo episodes evaluated (check dones/episode_starts).")


if __name__ == "__main__":
    main()