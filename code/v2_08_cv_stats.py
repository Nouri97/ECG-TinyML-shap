"""
v2_08_cv_stats.py - leave-one-fold-out cross-validation on the 10 official PTB-XL folds + statistical tests.

For every test fold k = 1..10 the models are trained on the other nine folds (patient-stratified folds, so no
patient appears in both). Hyperparameters: trees = manuscript Sec. 6.5 values; MLP/KNN/LogReg/SVM = the values
selected in v2_02 (Table 4). Caveat: those four were selected on folds 1-9, so their CV scores on folds 1-9 are
slightly optimistic; fold 10 is untouched.

A) 11 models on the 13 morphology features (variant F13):
   Friedman test on per-fold AUC, mean ranks, Nemenyi critical difference (CD), and pairwise Wilcoxon
   signed-rank tests of every model vs the best-ranked one with Holm correction.
   -> Table5 (all fold scores), Table6a (ranks, tests), Fig7 (critical-difference diagram).
B) Feature-set decision (LightGBM, XGBoost, MLP; fold scores averaged over the three models):
   F13 = morphology | F11 = F13 without T_asymmetry and J_to_Tpeak_ms | F15 = F13 + RR_median_ms, RR_cv.
   Pre-specified rule (alpha = 0.05, paired Wilcoxon on fold-averaged AUC): adopt F15 only if it is significantly
   better than F13; otherwise adopt F11 if it is not significantly worse than F13 (mean AUC loss < 0.003);
   otherwise keep F13.  -> Table6c.
With 10 folds the tests have low power: "not significantly different" is NOT evidence of equivalence.
Resumable: each (variant, model) is stored as its own chunk and reused only if the code hash matches.
"""
import ast
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range, wilcoxon

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
import v2_common as C  # noqa: E402
from v2_common import CONFIG  # noqa: E402
import v2_02_benchmark as B  # noqa: E402

SEED = 42
MAIN_VARIANT = "F15"   # "F13" = feature-set decision run; "F15" = final run (all 11 models on the adopted set)
OUT_KEYS = {"F13": ("table5", "table6a", "table6c", "fig7"), "F15": ("table5_f15", "table6a_f15", None, "fig7_f15")}
ALPHA = 0.05
VARIANT_MODELS = ["LightGBM", "XGBoost", "MLP (2-layer)"]
T_UNCERTAIN = ("T_asymmetry", "J_to_Tpeak_ms")
SLUG = {n: s["slug"] for n, s in B.MODELS.items()}


def variants():
    m = C.morphology_features()
    return {"F13": m, "F11": [f for f in m if f not in T_UNCERTAIN], "F15": m + C.FEATURE_GROUPS["rhythm"]}


def tuned_params(files):
    t4 = pd.read_csv(C.out_path("tables", files["table4"]))
    return {r["model"]: ast.literal_eval(r["hyperparameters"]) for _, r in t4.iterrows()
            if str(r["tuned_on"]).startswith("fold 9")}


def run_chunk(name, variant, feats, params, ok, log):
    est = B.MODELS[name]["make"](SEED, **params)
    cid = hashlib.md5((repr(est) + "|" + ",".join(feats)).encode()).hexdigest()[:8]   # config hash in the name
    path = C.out_path("cv", f"cv_{variant}_{SLUG[name]}_{cid}_v2.csv")
    if os.path.exists(path):
        log.info("%-18s %s: reused (identical config %s)", name, variant, cid)
        return pd.read_csv(path)
    t0, rows = time.time(), []
    for k in range(1, 11):
        tr, te = ok[ok["strat_fold"] != k], ok[ok["strat_fold"] == k]
        m = B.MODELS[name]["make"](SEED, **params).fit(tr[feats], tr["label"])
        met = C.metrics(te["label"].to_numpy(), B.proba(m, te[feats]))
        rows.append({"model": name, "variant": variant, "fold": k, "n_test": len(te),
                     **{x: met[x] for x in ("auc", "f1", "mcc", "bal_acc", "pr_auc")}})
    df = pd.DataFrame(rows)
    C.write_csv(df, path, extra={"features": feats, "params": str(params), "config_hash": cid})
    log.info("%-18s %s: done in %.0f s (mean AUC %.4f)", name, variant, time.time() - t0, df["auc"].mean())
    return df


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (len(p) - i) * p[idx])
        adj[idx] = min(1.0, run)
    return adj


def nemenyi_cd(k, n, alpha=ALPHA):
    q = studentized_range.ppf(1 - alpha, k, 1e6) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n))), float(q)


def cd_diagram(ranks, cd):
    """Demsar-style critical-difference diagram (rank 1 = best, on the left)."""
    import matplotlib.pyplot as plt
    k = len(ranks)
    names, vals = list(ranks.index), ranks.to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(k + 0.6, 0.4)          # best rank on the left
    ax.set_ylim(-3.4, 2.4)
    ax.axis("off")
    ax.hlines(0, 1, k, color="k", lw=1.2)
    for r in range(1, k + 1):
        ax.vlines(r, 0, 0.12, color="k", lw=1)
        ax.text(r, 0.28, str(r), ha="center", fontsize=9)
    half = (k + 1) // 2
    for i, (nm, r) in enumerate(zip(names, vals)):
        if i < half:      # best half: labels above, on the left side
            y = 0.8 + 0.32 * (half - 1 - i)
            ax.plot([r, r, 0.5], [0.12, y, y], color="gray", lw=0.8)
            ax.text(0.45, y, f"{nm} ({r:.2f})", ha="right", va="center", fontsize=8)
        else:             # worse half: labels below, on the right side
            y = -0.9 - 0.32 * (i - half)
            ax.plot([r, r, k + 0.5], [0, y, y], color="gray", lw=0.8)
            ax.text(k + 0.55, y, f"{nm} ({r:.2f})", ha="left", va="center", fontsize=8)
    ax.hlines(1.9, 1, 1 + cd, color="r", lw=3)
    ax.text(1 + cd / 2, 2.05, f"CD = {cd:.2f}", color="r", ha="center", fontsize=9)
    bars = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and not any(b0 <= i and b1 >= j for b0, b1 in bars):
            bars.append((i, j))
    for n_, (i, j) in enumerate(bars):
        ax.hlines(-0.3 - 0.16 * n_, vals[i] - 0.05, vals[j] + 0.05, color="k", lw=4, alpha=0.75)
    ax.set_title("Critical-difference diagram on AUC (10 folds): models joined by a bar are not significantly different (Nemenyi)",
                 fontsize=9)
    fig.tight_layout()
    return fig


def main():
    C.init_run(globals().get("__file__"), SEED)
    log = C.get_logger("cv_stats")
    files = CONFIG["files"]
    k5, k6a, k6c, k7 = OUT_KEYS[MAIN_VARIANT]
    outs = [C.out_path("tables", files[k5]), C.out_path("tables", files[k6a]),
            C.out_path("tables", files[k6c]) if k6c else None, C.out_path("figures", files[k7])]
    C.preflight([o for o in outs if o])
    df = pd.read_csv(C.out_path("features", files["ptbxl_features"]))
    ok = df[df["fail_reason"].fillna("") == ""].reset_index(drop=True)
    V = variants()
    for v, cols in V.items():
        if ok[cols].isna().any().any():
            raise ValueError(f"NaN in variant {v}")
    tuned = tuned_params(files)
    log.info("CV on %d records, folds %s", len(ok), sorted(ok["strat_fold"].unique()))

    parts = []
    for name in B.MODELS:
        parts.append(run_chunk(name, MAIN_VARIANT, V[MAIN_VARIANT], tuned.get(name, {}), ok, log))
    if MAIN_VARIANT == "F13":
        for variant in ("F11", "F15"):
            for name in VARIANT_MODELS:
                parts.append(run_chunk(name, variant, V[variant], tuned.get(name, {}), ok, log))
    scores = pd.concat(parts, ignore_index=True)

    # ---- A) 11 models on the main variant -------------------------------------------------------
    s13 = scores[scores["variant"] == MAIN_VARIANT]
    auc = s13.pivot(index="fold", columns="model", values="auc")
    k, n = auc.shape[1], auc.shape[0]
    stat, p_fr = friedmanchisquare(*[auc[c].to_numpy() for c in auc.columns])
    ranks_f = np.vstack([rankdata(-row) for row in auc.to_numpy()])
    mean_rank = pd.Series(ranks_f.mean(axis=0), index=auc.columns).sort_values()
    cd, q = nemenyi_cd(k, n)
    best = mean_rank.index[0]
    log.info("Friedman chi2=%.2f, p=%.2g | Nemenyi q=%.3f, CD=%.2f | best: %s", stat, p_fr, q, cd, best)
    others = [m for m in mean_rank.index if m != best]
    pw = [wilcoxon(auc[best], auc[m]).pvalue for m in others]
    p_h = holm(pw)
    tab = []
    for m in mean_rank.index:
        d = s13[s13["model"] == m]
        row = {"model": m, "mean_rank_auc": mean_rank[m], "mean_auc": d["auc"].mean(), "sd_auc": d["auc"].std(),
               "mean_f1": d["f1"].mean(), "sd_f1": d["f1"].std(), "mean_mcc": d["mcc"].mean(),
               "mean_bal_acc": d["bal_acc"].mean(), "rank_diff_vs_best": mean_rank[m] - mean_rank[best],
               "nemenyi_significant_vs_best": bool(mean_rank[m] - mean_rank[best] > cd) if m != best else False,
               "wilcoxon_p_vs_best": np.nan, "wilcoxon_p_holm": np.nan, "wilcoxon_significant_holm": False}
        if m != best:
            i = others.index(m)
            row.update(wilcoxon_p_vs_best=pw[i], wilcoxon_p_holm=p_h[i], wilcoxon_significant_holm=bool(p_h[i] < ALPHA))
        tab.append(row)
    t6a = pd.DataFrame(tab)
    pd.set_option("display.width", 250)
    log.info("\n%s", t6a.round(4).to_string(index=False))

    t6c, choice = None, None
    if MAIN_VARIANT == "F13":
        # ---- B) feature sets ---------------------------------------------------------------------------
        fs = scores[scores["model"].isin(VARIANT_MODELS)].groupby(["variant", "fold"])[["auc", "f1", "mcc"]].mean().reset_index()
        base = fs[fs["variant"] == "F13"].set_index("fold")
        rows6c = []
        for v in ("F11", "F15"):
            cur = fs[fs["variant"] == v].set_index("fold")
            d_auc, d_f1 = (cur["auc"] - base["auc"]), (cur["f1"] - base["f1"])
            rows6c.append({"variant": v, "vs": "F13", "n_features": len(V[v]), "mean_auc": cur["auc"].mean(),
                           "mean_auc_F13": base["auc"].mean(), "mean_auc_diff": d_auc.mean(), "mean_f1_diff": d_f1.mean(),
                           "folds_better_auc": int((d_auc > 0).sum()), "wilcoxon_p_auc": wilcoxon(d_auc).pvalue,
                           "wilcoxon_p_f1": wilcoxon(d_f1).pvalue})
        t6c = pd.DataFrame(rows6c)
        r15, r11 = t6c[t6c["variant"] == "F15"].iloc[0], t6c[t6c["variant"] == "F11"].iloc[0]
        if r15["mean_auc_diff"] > 0 and r15["wilcoxon_p_auc"] < ALPHA:
            choice = "F15 (rhythm adds a significant AUC gain)"
        elif r11["mean_auc_diff"] > -0.003 and not (r11["wilcoxon_p_auc"] < ALPHA and r11["mean_auc_diff"] < 0):
            choice = "F11 (dropping the two T-timing features is not significantly worse and costs < 0.003 AUC)"
        else:
            choice = "F13 (dropping the two T-timing features costs AUC)"
        t6c["pre_specified_rule_choice"] = choice
        log.info("\n%s\nRULE OUTCOME: %s", t6c.round(4).to_string(index=False), choice)

    C.write_csv(scores, outs[0])
    C.write_csv(t6a, outs[1], extra={"friedman_chi2": float(stat), "friedman_p": float(p_fr), "nemenyi_cd": cd,
                                       "nemenyi_q": q, "best": best, "n_folds": n})
    if t6c is not None:
        C.write_csv(t6c, outs[2], extra={"models_averaged": VARIANT_MODELS, "rule": choice})
    fig = cd_diagram(mean_rank, cd)
    C.write_png(fig, outs[3])
    C.show_and_close(fig)


if __name__ == "__main__":
    main()
