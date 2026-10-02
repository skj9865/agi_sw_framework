"""Sensitivity-guided quantile allocation (knapsack / Lagrangian) for expert-prune thresholds.

Reads the calibration npz from alloc_calibrate.py and solves

    min_rho  sum_{l,e} S_{l,e}(rho_{l,e})   s.t.   sum_{l,e} c_l * rho_{l,e} >= B

where S_{l,e}(rho) = cumulative KL damage of pruning expert (l,e) on its rho-fraction
lowest-entropy calibration inputs, and c_l = per-expert energy (DRAM weight load +
compute) of block l. rho is discretized on a quantile grid; the Lagrangian sweep
(binary search on the energy price lambda, per-expert argmin of S - lambda*c*rho)
solves the multiple-choice knapsack; sweeping lambda traces the whole Pareto frontier.

The uniform percentile ladder is the special case rho_{l,e} = p for all (l,e);
budgets are matched to it via B(p) = sum c_l * p.

Outputs per-(block,expert) thresholds (this expert's calibration-entropy quantile at
its allocated rho) as a comma string consumable by prune_eval.py --exit-threshold-per-expert.

  python alloc_solve.py --calib alloc_calib.npz --arch maxformer --match-uniform 50,80,90
"""
import argparse
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--calib", default="alloc_calib.npz")
p.add_argument("--arch", default="maxformer", choices=["maxformer", "sdt", "sdt4_256", "sdt4_384", "sdt8_768"])
p.add_argument("--match-uniform", default="50,80,90",
               help="uniform ladder percentiles whose energy budgets to match")
p.add_argument("--grid", type=int, default=100, help="quantile grid resolution")
p.add_argument("--metric", default="kl", choices=["kl", "flip"], help="damage measure for S curves")
p.add_argument("--cost", default="model", choices=["model", "uniform"],
               help="per-expert cost: 'model' = energy model (DRAM+compute), 'uniform' = 1 per expert "
                    "(budget == average active experts)")
p.add_argument("--out", default="alloc_thresholds.txt")
args = p.parse_args()

# ---- per-expert energy cost c_l (same calibrated energy model as the paper tables) ----
c_dram = 0.112189 / (2 * 512 * 1024)                    # per weight-load unit
c_comp = 0.043923 / ((64 / 4) * 2 * 512 * 1024 * 4)     # per MAC unit
if args.arch == "maxformer":     # hierarchical H/4,H/8,H/16 sizing
    dims, N, T = [96, 192, 384, 384], [64, 16, 4, 4], 4
elif args.arch == "sdt4_256":    # sdt cifar100: 4 blocks, dim 256, ratio-1 experts
    dims, N, T = [256] * 4, [64] * 4, 4
elif args.arch == "sdt4_384":    # sdt cifar100: 4 blocks, dim 384, ratio-1 experts
    dims, N, T = [384] * 4, [64] * 4, 4
elif args.arch == "sdt8_768":    # sdt imagenet: 8 blocks, dim 768, ratio-1 experts (224/16=14 -> 196 tokens)
    dims, N, T = [768] * 8, [196] * 8, 4
else:                            # sdt cifar100: 2 blocks, dim 512, ratio-1 experts
    dims, N, T = [512, 512], [64, 64], 4
COST = [c_dram * 2 * d * d + c_comp * (n / 4) * 2 * d * d * T for d, n in zip(dims, N)]
if args.cost == "uniform":
    COST = [1.0] * len(dims)   # every expert equal -> budget is in active-expert units

d = np.load(args.calib)
L = 1 + max(int(k.split("_l")[1].split("_e")[0]) for k in d.files if k.startswith("ent_"))
E = 1 + max(int(k.split("_e")[1]) for k in d.files if k.startswith("ent_"))
Ncal = d["ent_l0_e0"].size
K = args.grid
rhos = np.arange(K + 1) / K                                  # 0, 1/K, ..., 1

# ---- S curves + threshold lookup per (l,e) ----
S = np.zeros((L, E, K + 1))          # cumulative damage at prune-rate rho
TH = np.zeros((L, E, K + 1))         # entropy threshold realizing that rho
for l in range(L):
    for e in range(E):
        ent = d[f"ent_l{l}_e{e}"]
        dmg = d[f"kl_l{l}_e{e}"] if args.metric == "kl" else d[f"flip_l{l}_e{e}"].astype(np.float64)
        order = np.argsort(ent, kind="stable")
        csum = np.concatenate([[0.0], np.cumsum(dmg[order])])
        idx = np.floor(rhos * Ncal).astype(int)
        S[l, e] = csum[idx] / Ncal                          # expected damage per image
        se = np.sort(ent, kind="stable")
        # threshold below the (idx)th lowest entropy -> prunes exactly those inputs
        th = np.empty(K + 1)
        th[0] = -1.0                                        # rho=0: prune nothing
        for k in range(1, K + 1):
            lo = se[idx[k] - 1]
            hi = se[idx[k]] if idx[k] < Ncal else lo + 1e-6
            th[k] = 0.5 * (lo + hi) if hi > lo else lo + 1e-9
        TH[l, e] = th

cost = np.array([[COST[l]] * E for l in range(L)])          # (L,E)

def solve(budget):
    """Lagrangian MCKP: binary-search lambda; each expert picks argmin_k S - lam*c*rho."""
    def alloc(lam):
        k = np.argmin(S - lam * cost[..., None] * rhos[None, None, :], axis=2)   # (L,E)
        sav = (cost * rhos[k]).sum()
        return k, sav
    lo, hi = 0.0, 1.0
    while alloc(hi)[1] < budget and hi < 1e9:
        hi *= 4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if alloc(mid)[1] < budget:
            lo = mid
        else:
            hi = mid
    k, sav = alloc(hi)
    return k, sav

total_c = cost.sum()
print(f"{args.arch}: L={L} E={E} Ncal={Ncal} grid={K}  per-block cost {[f'{c:.4f}' for c in COST]}")
lines = []
for pu in [float(x) for x in args.match_uniform.split(",")]:
    B = total_c * pu / 100.0
    k, sav = solve(B)
    dmg_a = S[np.arange(L)[:, None], np.arange(E)[None, :], k].sum()
    ku = int(round(K * pu / 100.0))
    dmg_u = S[:, :, ku].sum()
    thr = ",".join(f"{TH[l, e, k[l, e]]:.4f}" for l in range(L) for e in range(E))
    rho_tab = "; ".join("b%d:[%s]" % (l, " ".join(f"{rhos[k[l,e]]:.2f}" for e in range(E))) for l in range(L))
    print(f"\n== budget matched to uniform p{pu:g} (saving {100*B/total_c:.1f}% of expert energy) ==")
    print(f"  allocated rho: {rho_tab}")
    print(f"  predicted damage/img: allocated={dmg_a:.5f}  uniform={dmg_u:.5f}  "
          f"(reduction {100*(1-dmg_a/max(dmg_u,1e-12)):.1f}%)")
    print(f"  thresholds: {thr}")
    lines.append(f"p{pu:g}\tB={B:.5f}\trho={rho_tab}\tthr={thr}")
with open(args.out, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nwrote {args.out}")
