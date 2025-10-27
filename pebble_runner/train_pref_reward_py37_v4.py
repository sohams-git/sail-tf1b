#!/usr/bin/env python
# Py37-compatible. Handles JSONL with i/j entries, ep_id, start/end, and y as [p_i, p_j].
# Adds: early stopping (--patience) and periodic checkpoints (--save-every).
import os, json, argparse, numpy as np, torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
rng = np.random.RandomState(0)

# ---------- NPZ helpers ----------
def load_legacy_npz(path):
    f = np.load(path, allow_pickle=True)
    return (f["obs"].astype(np.float32),
            f["actions"].astype(np.float32),
            f["rewards"].astype(np.float32),
            f["episode_starts"].astype(bool))

# ---------- record parsing ----------
META_KEYS = {"tiers_present","return_stats","gap_stats","stats","meta"}

def normalize_path(p, root):
    if isinstance(p, str) and p.endswith(".npz"):
        if os.path.isabs(p): return p
        return os.path.normpath(os.path.join(root, p)) if root else p
    return p

def label_from_y(y_field):
    # Accept int/bool/str, or 2-vector [p_i, p_j]
    if isinstance(y_field, (int, np.integer)): return int(y_field)
    if isinstance(y_field, float): return int(round(y_field))
    if isinstance(y_field, str):
        s = y_field.strip().lower()
        if s in ("1","left","l","first","a","i"): return 1
        if s in ("0","right","r","second","b","j"): return 0
    if isinstance(y_field, (list, tuple)) and len(y_field)==2:
        return 1 if float(y_field[0]) >= float(y_field[1]) else 0
    raise ValueError("Unrecognized y field: {}".format(y_field))

def slice_by_start_end(obs, acts, start, end):
    start = int(start); end = int(end)
    if start < 0 or end > obs.shape[0] or end <= start:
        raise IndexError("Bad start/end: {}..{} for length {}".format(start, end, obs.shape[0]))
    return obs[start:end], acts[start:end]

def parse_pair_line(rec, obs_dim, act_dim, npz_cache, root):
    # Skip metadata-only lines
    if any(k in rec for k in META_KEYS):
        return None

    # Expect i and j dicts
    if not (isinstance(rec.get("i"), dict) and isinstance(rec.get("j"), dict)):
        return None

    i, j = rec["i"], rec["j"]
    # file paths (abs or relative to root)
    f1 = normalize_path(i.get("file"), root)
    f2 = normalize_path(j.get("file"), root)
    if not (isinstance(f1, str) and isinstance(f2, str)):
        return None
    if not (os.path.isfile(f1) and os.path.isfile(f2)):
        return None

    # load arrays
    if f1 not in npz_cache: npz_cache[f1] = load_legacy_npz(f1)
    if f2 not in npz_cache: npz_cache[f2] = load_legacy_npz(f2)
    obs1, act1, _, _ = npz_cache[f1]
    obs2, act2, _, _ = npz_cache[f2]

    # prefer start/end slicing if present; otherwise reconstruct from ep_id*length
    if all(k in i for k in ("start","end")) and all(k in j for k in ("start","end")):
        o1, a1 = slice_by_start_end(obs1, act1, i["start"], i["end"])
        o2, a2 = slice_by_start_end(obs2, act2, j["start"], j["end"])
    else:
        if ("ep_id" in i and "length" in i):
            s1 = int(i["ep_id"]) * int(i["length"]); e1 = s1 + int(i["length"])
            o1, a1 = slice_by_start_end(obs1, act1, s1, e1)
        else:
            return None
        if ("ep_id" in j and "length" in j):
            s2 = int(j["ep_id"]) * int(j["length"]); e2 = s2 + int(j["length"])
            o2, a2 = slice_by_start_end(obs2, act2, s2, e2)
        else:
            return None

    # dim check
    if o1.shape[-1]!=obs_dim or a1.shape[-1]!=act_dim or o2.shape[-1]!=obs_dim or a2.shape[-1]!=act_dim:
        return None

    # label
    if "y" in rec:
        y = label_from_y(rec["y"])
    else:
        if ("return" in i) and ("return" in j):
            y = 1 if float(i["return"]) >= float(j["return"]) else 0
        else:
            return None

    return o1.astype(np.float32), a1.astype(np.float32), o2.astype(np.float32), a2.astype(np.float32), int(y)

def load_pairs(jsonl_path, obs_dim, act_dim, root=""):
    O1,A1,O2,A2,Y = [],[],[],[],[]
    npz_cache = {}
    n_lines = 0; n_good = 0; n_meta = 0; n_bad = 0
    with open(jsonl_path) as fh:
        for line in fh:
            n_lines += 1
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
            except Exception:
                n_bad += 1; continue
            if any(k in rec for k in META_KEYS):
                n_meta += 1; continue
            parsed = parse_pair_line(rec, obs_dim, act_dim, npz_cache, root)
            if parsed is None:
                n_bad += 1; continue
            o1,a1,o2,a2,y = parsed
            O1.append(o1); A1.append(a1); O2.append(o2); A2.append(a2); Y.append(y); n_good += 1
    if n_good == 0:
        raise RuntimeError("No usable pairs found in {} ({} lines; {} meta; {} bad).".format(
            jsonl_path, n_lines, n_meta, n_bad))
    print("[info] loaded {} pairs ({} lines; {} meta; {} bad)".format(n_good, n_lines, n_meta, n_bad))
    return O1,A1,O2,A2,np.asarray(Y, np.int64)

# ---------- model ----------
class ZScore(nn.Module):
    def __init__(self,d,eps=1e-6): super().__init__(); self.register_buffer("mu",torch.zeros(d)); self.register_buffer("sigma",torch.ones(d)); self.eps=eps
    @torch.no_grad()
    def fit(self,X): self.mu.copy_(X.mean(0)); self.sigma.copy_(X.std(0).clamp_min(self.eps)); return self
    def forward(self,X): return (X-self.mu)/self.sigma

class RewardMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hid=256):
        super().__init__()
        d=obs_dim+act_dim
        self.norm=ZScore(d)
        self.net=nn.Sequential(
            nn.Linear(d,hid),nn.ReLU(),
            nn.Linear(hid,hid),nn.ReLU(),
            nn.Linear(hid,1),
            nn.Tanh()  # PEBBLE-style bounded head
        )
    def forward(self,s,a): return self.net(self.norm(torch.cat([s,a],-1)))

def sum_seq(model, O_list, A_list, device):
    N=len(O_list); T=[o.shape[0] for o in O_list]
    s=torch.from_numpy(np.concatenate(O_list,0)).float().to(device)
    a=torch.from_numpy(np.concatenate(A_list,0)).float().to(device)
    r=model(s,a)
    out=[]; i=0
    for t in T: out.append(r[i:i+t].sum(0,keepdim=True)); i+=t
    return torch.cat(out,0)

def stratified_split(labels, val_frac=0.1):
    y=np.asarray(labels); idx=np.arange(len(y))
    pos=idx[y==1]; neg=idx[y==0]
    rng.shuffle(pos); rng.shuffle(neg)
    vp=max(1,int(round(len(pos)*val_frac))); vn=max(1,int(round(len(neg)*val_frac)))
    va=np.concatenate([pos[:vp],neg[:vn]]); tr=np.concatenate([pos[vp:],neg[vn:]])
    rng.shuffle(tr); rng.shuffle(va); return tr, va

# ---------- export helper ----------
def export_npz(model, path, obs_dim, act_dim):
    sd = model.state_dict()
    W1,b1 = sd['net.0.weight'].cpu().numpy(), sd['net.0.bias'].cpu().numpy()
    W2,b2 = sd['net.2.weight'].cpu().numpy(), sd['net.2.bias'].cpu().numpy()
    W3,b3 = sd['net.4.weight'].cpu().numpy(), sd['net.4.bias'].cpu().numpy()
    mu,sigma = sd['norm.mu'].cpu().numpy(), sd['norm.sigma'].cpu().numpy()
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    np.savez(path, W1=W1,b1=b1,W2=W2,b2=b2,W3=W3,b3=b3, mu=mu, sigma=sigma,
             obs_dim=obs_dim, act_dim=act_dim)
    print("[checkpoint] saved:", path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--root", default="", help="Resolve relative NPZ paths if present")
    ap.add_argument("--obs-dim", type=int, default=18)
    ap.add_argument("--act-dim", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-items", type=int, default=0, help="Optional cap for quick tests")
    # new:
    ap.add_argument("--patience", type=int, default=200, help="epochs-between-checkpoints without val improvement")
    ap.add_argument("--save-every", type=int, default=0, help="save additional checkpoints every K epochs (0=off)")
    ap.add_argument("--save-npz", required=True)
    args=ap.parse_args()

    device=torch.device("cpu")
    O1,A1,O2,A2,Y = load_pairs(args.pairs, args.obs_dim, args.act_dim, root=args.root)
    if args.max_items and args.max_items < len(Y):
        O1,O2 = O1[:args.max_items], O2[:args.max_items]
        A1,A2 = A1[:args.max_items], A2[:args.max_items]
        Y = Y[:args.max_items]

    flat=np.concatenate([np.concatenate(O1,0), np.concatenate(A1,0)],1)
    model=RewardMLP(args.obs_dim,args.act_dim).to(device); model.norm.fit(torch.from_numpy(flat).float())

    tr,va = stratified_split(Y, val_frac=args.val_frac)
    bce=nn.BCEWithLogitsLoss()
    opt=optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    def batches(ix,bs):
        for i in range(0,len(ix),bs): yield ix[i:i+bs]

    best = float("inf")
    bad  = 0
    saved_any = False

    for ep in range(1, args.epochs+1):
        model.train(); losses=[]
        for ib in batches(tr, args.batch_size):
            r1=sum_seq(model,[O1[i] for i in ib],[A1[i] for i in ib],device)
            r2=sum_seq(model,[O2[i] for i in ib],[A2[i] for i in ib],device)
            y=torch.from_numpy(Y[ib]).float().to(device).unsqueeze(1)
            loss=bce(r1-r2, y); opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())

        if ep % 100 == 0:
            model.eval()
            with torch.no_grad():
                r1=sum_seq(model,[O1[i] for i in va],[A1[i] for i in va],device)
                r2=sum_seq(model,[O2[i] for i in va],[A2[i] for i in va],device)
                y=torch.from_numpy(Y[va]).float().to(device).unsqueeze(1)
                va_loss=bce(r1-r2,y).item(); acc=((r1-r2>0).float()==y).float().mean().item()
            print("[{}] tr_loss={:.4f} va_loss={:.4f} va_acc={:.3f}".format(
                ep, float(np.mean(losses)), va_loss, acc))

            # --- early stop & checkpoint ---
            if va_loss < best - 1e-4:
                best = va_loss; bad = 0
                export_npz(model, args.save_npz, args.obs_dim, args.act_dim)  # save best
                saved_any = True
            else:
                bad += 1

            if args.save_every and (ep % args.save_every == 0):
                export_npz(model, args.save_npz.replace(".npz", "_ep{}.npz".format(ep)),
                           args.obs_dim, args.act_dim)

            if args.patience and bad >= args.patience:
                print("[early-stop] no val improvement for {} evals".format(args.patience))
                break

    # Final save if nothing saved during training
    if not saved_any:
        export_npz(model, args.save_npz, args.obs_dim, args.act_dim)

if __name__=="__main__": main()
