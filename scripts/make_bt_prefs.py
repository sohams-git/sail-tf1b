#!/usr/bin/env python3
# Trajectories-only BT preference dataset from teacher .npz files.
# Compares full episodes by total return; higher return is preferred.

import argparse, re, json, random
from pathlib import Path
import numpy as np
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser("Make BT preference dataset from teacher .npz (episodes only)")
    p.add_argument("--sources", nargs="+", required=True,
                   help="Absolute or relative paths to teacher .npz files.")
    p.add_argument("--out_jsonl", required=True, help="Output JSONL with human-readable pairs")
    p.add_argument("--out_npz",   required=True, help="Output NPZ with compact arrays")
    p.add_argument("--delta", type=float, default=100.0,
                   help="Drop (or tie-label) pairs with |Gi-Gj| < delta.")
    p.add_argument("--keep_near_ties", action="store_true",
                   help="Keep |Gi-Gj| < delta as y=(0.5,0.5) instead of dropping.")
    p.add_argument("--pairing", choices=["cross_tier_only","within_tier_only","all"],
                   default="cross_tier_only",
                   help="How to allow pairs (default cross_tier_only).")
    p.add_argument("--max_pairs", type=int, default=0,
                   help="0 = keep all candidate pairs; >0 = downsample balanced by gap.")
    p.add_argument("--balance_bins", type=int, default=8,
                   help="Bins for gap-balancing when max_pairs>0.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def extract_tier_from_name(path):
    # e.g., ...scores_300_episodes_4_....
    m = re.search(r"scores_(\d+)_episodes", Path(path).name)
    return int(m.group(1)) if m else None

def load_episodes(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    if "rewards" not in d or "episode_starts" not in d:
        raise KeyError(f"{npz_path} must contain 'rewards' and 'episode_starts'")
    rewards = d["rewards"].reshape(-1)
    starts  = d["episode_starts"].astype(bool).reshape(-1)

    T = len(rewards)
    # episode start indices (ensure 0 included)
    idx = np.nonzero(starts)[0].tolist()
    if not idx or idx[0] != 0:
        idx = [0] + idx

    # episode boundaries
    bounds = []
    for k in range(len(idx)):
        s = idx[k]
        e = idx[k+1] if k+1 < len(idx) else T
        bounds.append((s, e))

    tier = extract_tier_from_name(npz_path)
    episodes = []
    for ep_id, (s, e) in enumerate(bounds):
        Gi = float(rewards[s:e].sum())
        episodes.append({
            "file": str(npz_path),
            "tier": tier,
            "ep_id": int(ep_id),
            "start": int(s),
            "end":   int(e),
            "length": int(e - s),
            "return": Gi
        })
    return episodes

def pair_ok(a, b, mode):
    if mode == "all":
        return True
    if mode == "cross_tier_only":
        # Prefer semantic check on 'tier'; fall back to file inequality if tier missing.
        if a["tier"] is not None and b["tier"] is not None:
            return a["tier"] != b["tier"]
        return a["file"] != b["file"]
    if mode == "within_tier_only":
        if a["tier"] is not None and b["tier"] is not None:
            return a["tier"] == b["tier"]
        return a["file"] == b["file"]
    return True

def label_from_gap(Gi, Gj, delta, keep_ties):
    gap = Gi - Gj
    agap = abs(gap)
    if agap < delta:
        if keep_ties:
            return [0.5, 0.5], agap
        else:
            return None, agap
    return ([1.0, 0.0], agap) if gap > 0 else ([0.0, 1.0], agap)

def linspace_edges(values, nbins):
    vmin, vmax = float(np.min(values)), float(np.max(values))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return np.linspace(vmin, vmax, nbins + 1)

def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    # 1) Load all episodes from all sources
    items = []
    for src in sorted(args.sources):
        items.extend(load_episodes(src))

    N = len(items)
    if N < 2:
        raise SystemExit("Not enough episodes to form pairs.")

    # 2) Build ALL unordered candidate pairs (i < j), episode-level only
    cand = []
    for i in range(N):
        ai = items[i]
        for j in range(i + 1, N):
            aj = items[j]
            if not pair_ok(ai, aj, args.pairing):
                continue
            y, agap = label_from_gap(ai["return"], aj["return"], args.delta, args.keep_near_ties)
            if y is None:
                continue
            cand.append((i, j, y, agap))

    if not cand:
        raise SystemExit("No candidate pairs after filtering. Consider lowering --delta or changing --pairing.")

    # 3) If capped, balance by gap magnitude bins; else keep all
    if args.max_pairs > 0 and len(cand) > args.max_pairs:
        gaps = np.array([c[3] for c in cand], dtype=float)
        edges = linspace_edges(gaps, args.balance_bins)
        bin_ids = np.digitize(gaps, edges) - 1
        bin_ids = np.clip(bin_ids, 0, args.balance_bins - 1)

        per_bin = max(1, args.max_pairs // args.balance_bins)
        chosen = []
        for b in range(args.balance_bins):
            idxs = np.nonzero(bin_ids == b)[0]
            if len(idxs) == 0:
                continue
            take = min(per_bin, len(idxs))
            sel  = np.random.choice(idxs, size=take, replace=False)
            chosen.extend([cand[k] for k in sel])

        # top up if under-filled
        if len(chosen) < args.max_pairs:
            remaining = list(set(range(len(cand))) - set([cand.index(c) for c in chosen]))
            extra = min(args.max_pairs - len(chosen), len(remaining))
            if extra > 0:
                sel = np.random.choice(remaining, size=extra, replace=False)
                chosen.extend([cand[k] for k in sel])
    else:
        chosen = cand

    # 4) Save outputs
    out_jsonl = Path(args.out_jsonl)
    out_npz   = Path(args.out_npz)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    i_idx = np.array([c[0] for c in chosen], dtype=np.int64)
    j_idx = np.array([c[1] for c in chosen], dtype=np.int64)
    y     = np.array([c[2] for c in chosen], dtype=np.float32)
    gaps  = np.array([c[3] for c in chosen], dtype=np.float32)

    # Compact NPZ: indices + labels + items (episodes)
    np.savez_compressed(str(out_npz),
                        i_idx=i_idx, j_idx=j_idx, y=y, gap=gaps,
                        items=np.array(items, dtype=object))

    # Human-readable JSONL
    with open(out_jsonl, "w") as f:
        for ii, jj, yy, gg in zip(i_idx, j_idx, y, gaps):
            f.write(json.dumps({
                "i": items[int(ii)],
                "j": items[int(jj)],
                "y": [float(yy[0]), float(yy[1])],
                "gap": float(gg)
            }) + "\n")

    # Manifest
    tiers_present = sorted(set([it["tier"] for it in items if it["tier"] is not None]))
    manifest = {
        "n_items": int(N),
        "n_pairs": int(len(i_idx)),
        "delta": float(args.delta),
        "keep_near_ties": bool(args.keep_near_ties),
        "pairing": args.pairing,
        "max_pairs_requested": int(args.max_pairs),
        "balance_bins": int(args.balance_bins),
        "sources": sorted(set([it["file"] for it in items])),
        "tiers_present": tiers_present,
        "return_stats": {
            "min": float(min(it["return"] for it in items)),
            "max": float(max(it["return"] for it in items)),
            "mean": float(np.mean([it["return"] for it in items]))
        },
        "gap_stats": {
            "min": float(gaps.min()) if len(gaps) else None,
            "max": float(gaps.max()) if len(gaps) else None,
            "mean": float(gaps.mean()) if len(gaps) else None
        }
    }
    mf = out_jsonl.with_suffix(".manifest.json")
    with open(mf, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== Done (episodes only) ===")
    print("Episodes:", N)
    print("Pairs   :", len(i_idx))
    print("JSONL   :", str(out_jsonl))
    print("NPZ     :", str(out_npz))
    print("Manifest:", str(mf))
    if len(i_idx) > 0:
        eg = {
            "i": items[int(i_idx[0])],
            "j": items[int(j_idx[0])],
            "y": y[0].tolist(),
            "gap": float(gaps[0])
        }
        print("\nExample pair:\n" + json.dumps(eg, indent=2))

if __name__ == "__main__":
    main()
