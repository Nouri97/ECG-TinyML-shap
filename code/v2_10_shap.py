"""
v2_10_shap.py - SHAP (TreeExplainer) on the FLOAT tree models, with the checks the reviewers asked for.

Explained models: LightGBM (primary), XGBoost and Gradient Boosting (agreement check). Explained data:
PTB-XL fold 10 (held out) and CPSC-2018 (external). SHAP values are in log-odds of "Abnormal".
SHAP is computed on the float models only; stability under INT8 is assessed through permutation
importance in v2_09 (Table 7b), not through SHAP on a quantised model.

Outputs
  Table8_shap_importance_v2.csv   : per feature mean |SHAP| (PTB-XL fold 10 and CPSC-2018 for LightGBM; fold 10 for
                                    XGBoost and Gradient Boosting), ranks, observed direction of effect (quintile
                                    means of SHAP vs feature value) and a generic electrophysiological reading.
  Table8b_shap_agreement_v2.csv   : Spearman agreement of the importance rankings across models and datasets.
  Fig4 (global bar), Fig5 (beeswarm), Fig6 (three waterfalls: confident abnormal, confident normal, borderline error).

IMPORTANT on interpretation: CLINICAL_READING below are generic textbook readings of what each measurement
captures. They are NOT evidence of agreement with AHA/ESC criteria. The authors must check every line against
references before using it in the paper; features such as skewness/kurtosis/zero-cross rate have no direct
clinical reading. SHAP explains the model, not the physiology.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
import v2_common as C  # noqa: E402
from v2_common import CONFIG  # noqa: E402

SEED = 42
PRIMARY = "LightGBM"
SLUGS = {"LightGBM": "lightgbm", "XGBoost": "xgboost", "Gradient Boosting": "gradient_boosting"}
CLINICAL_READING = {
    "ST_level_mV": "ST-segment deviation relative to baseline (elevation or depression) - repolarisation/ischaemia-related",
    "ST_slope_mV_per_s": "slope of the early ST segment (e.g. downsloping) - repolarisation-related",
    "T_amplitude_mV": "T-wave amplitude; flattened or inverted T waves accompany repolarisation abnormalities",
    "T_asymmetry": "shape asymmetry of the T wave (proxy; weak direct clinical reading)",
    "J_to_Tpeak_ms": "ST-T timing proxy (not a validated interval)",
    "QRS_duration_ms": "QRS width proxy; widening accompanies conduction delay/bundle-branch block",
    "R_amplitude_mV": "R-wave voltage (relevant to hypertrophy and low-voltage states)",
    "Q_depth_mV": "Q-wave depth (pathological Q waves in prior infarction)",
    "S_depth_mV": "S-wave depth (voltage criteria for hypertrophy)",
    "RMS_energy": "overall beat energy/voltage",
    "Skewness": "beat-shape statistic; no direct clinical reading",
    "Kurtosis": "beat-shape statistic; no direct clinical reading",
    "ZeroCrossRate": "beat-shape statistic; no direct clinical reading",
    "RR_median_ms": "heart rate (tachycardia/bradycardia)",
    "RR_cv": "RR irregularity (ectopy, atrial fibrillation)",
}


def margin_fn(name, m):
    if name == "LightGBM":
        return lambda X: m.predict(X, raw_score=True)
    if name == "XGBoost":
        return lambda X: m.predict(X, output_margin=True)
    return lambda X: m.decision_function(X)


def shap_values(name, model, X, feats, log):
    import shap
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(X)
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    ev = np.asarray(ex.expected_value).ravel()[-1]
    err = np.abs(ev + sv.sum(axis=1) - np.asarray(margin_fn(name, model)(X), float)).max()
    log.info("%s: SHAP additivity check, max |base + sum(SHAP) - margin| = %.2e", name, err)
    if err > 1e-3:
        log.warning("%s: additivity error above 1e-3; check shap/library versions before trusting the values", name)
    return sv, float(ev)


def direction(x, s):
    """Observed direction of effect from quintile means of SHAP across the feature's range."""
    q = pd.qcut(pd.Series(x).rank(method="first"), 5, labels=False)
    m = np.array([s[q == k].mean() for k in range(5)])
    rho = spearmanr(x, s).statistic
    if m[2] < min(m[0], m[4]) - 0.05:
        txt = "U-shaped (both extremes raise the abnormal score)"
    elif m[2] > max(m[0], m[4]) + 0.05:
        txt = "inverted-U (mid-range raises the abnormal score)"
    elif rho > 0.2:
        txt = "higher value -> higher abnormal score"
    elif rho < -0.2:
        txt = "lower value -> higher abnormal score"
    else:
        txt = "weak / no monotonic trend"
    return rho, m[0], m[4], txt


def main():
    C.init_run(globals().get("__file__"), SEED)
    log = C.get_logger("shap")
    files = CONFIG["files"]
    outs = [C.out_path("tables", files[k]) for k in ("table8", "table8b")] + \
           [C.out_path("figures", files[k]) for k in ("fig4", "fig5", "fig6_tp", "fig6_tn", "fig6_border")]
    C.preflight(outs)
    import matplotlib.pyplot as plt
    import shap

    models = {n: joblib.load(C.out_path("models", f"{s}_v2.pkl")) for n, s in SLUGS.items()}
    feats = list(models[PRIMARY].feature_names_in_)
    d = pd.read_csv(C.out_path("features", files["ptbxl_features"]))
    te = d[(d["fail_reason"].fillna("") == "") & (d["strat_fold"] == 10)].reset_index(drop=True)
    X, y = te[feats].to_numpy(float), te["label"].to_numpy()
    Xdf = pd.DataFrame(X, columns=feats)
    cp = C.out_path("features", files["cpsc_features"])
    ext = None
    if os.path.exists(cp):
        e = pd.read_csv(cp)
        e = e[e["fail_reason"].fillna("") == ""].reset_index(drop=True)
        ext = pd.DataFrame(e[feats].to_numpy(float), columns=feats)

    sv, ev, imp = {}, {}, {}
    for n, m in models.items():
        sv[n], ev[n] = shap_values(n, m, Xdf, feats, log)
        imp[(n, "PTB-XL_fold10")] = np.abs(sv[n]).mean(axis=0)
    sv_ext = None
    if ext is not None:
        sv_ext, _ = shap_values(PRIMARY, models[PRIMARY], ext, feats, log)
        imp[(PRIMARY, "CPSC2018")] = np.abs(sv_ext).mean(axis=0)

    rows = []
    prim = imp[(PRIMARY, "PTB-XL_fold10")]
    for j, f in enumerate(feats):
        rho, q1, q5, txt = direction(X[:, j], sv[PRIMARY][:, j])
        row = {"feature": f, "mean_abs_shap_LightGBM_PTBXL": prim[j],
               "share_pct_LightGBM_PTBXL": prim[j] / prim.sum() * 100,
               "rank_LightGBM_PTBXL": int((prim > prim[j]).sum() + 1),
               "mean_abs_shap_XGBoost_PTBXL": imp[("XGBoost", "PTB-XL_fold10")][j],
               "mean_abs_shap_GB_PTBXL": imp[("Gradient Boosting", "PTB-XL_fold10")][j],
               "spearman_value_vs_shap": rho, "mean_shap_lowest_quintile": q1, "mean_shap_highest_quintile": q5,
               "observed_direction": txt, "generic_reading_TO_VERIFY": CLINICAL_READING.get(f, "")}
        if sv_ext is not None:
            ie = imp[(PRIMARY, "CPSC2018")]
            row.update(mean_abs_shap_LightGBM_CPSC=ie[j], rank_LightGBM_CPSC=int((ie > ie[j]).sum() + 1))
        rows.append(row)
    t8 = pd.DataFrame(rows).sort_values("rank_LightGBM_PTBXL")

    keys = list(imp)
    agr = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            agr.append({"a": f"{a[0]} / {a[1]}", "b": f"{b[0]} / {b[1]}", "spearman_rho_mean_abs_shap": spearmanr(imp[a], imp[b]).statistic,
                        "top5_overlap": len(set(np.argsort(-imp[a])[:5]) & set(np.argsort(-imp[b])[:5])) / 5})
    t8b = pd.DataFrame(agr)
    pd.set_option("display.width", 250)
    log.info("\n%s", t8[["feature", "share_pct_LightGBM_PTBXL", "rank_LightGBM_PTBXL", "observed_direction"]].round(2).to_string(index=False))
    log.info("\n%s", t8b.round(3).to_string(index=False))

    # figures (compute first, write afterwards)
    order = np.argsort(prim)
    fig4, ax = plt.subplots(figsize=(8, 6))
    yy = np.arange(len(feats))
    ax.barh(yy - 0.2, prim[order], 0.4, label="PTB-XL fold 10", color="tab:blue")
    if sv_ext is not None:
        ax.barh(yy + 0.2, imp[(PRIMARY, "CPSC2018")][order], 0.4, label="CPSC-2018", color="tab:red")
    ax.set_yticks(yy, [feats[i] for i in order])
    ax.set_xlabel("mean |SHAP| (log-odds of Abnormal), LightGBM")
    ax.legend()
    fig4.tight_layout()

    plt.figure(figsize=(9, 6.5))
    shap.summary_plot(sv[PRIMARY], X, feature_names=feats, show=False)
    fig5 = plt.gcf()
    fig5.tight_layout()

    p = models[PRIMARY].predict_proba(Xdf)[:, 1]
    cases = {"fig6_tp": int(np.where(y == 1)[0][np.argmax(p[y == 1])]), "fig6_tn": int(np.where(y == 0)[0][np.argmin(p[y == 0])])}
    wrong = np.where((p >= 0.5) != (y == 1))[0]
    pool = wrong if len(wrong) else np.arange(len(p))          # no error at all (tiny test sets): most uncertain case
    cases["fig6_border"] = int(pool[np.argmin(np.abs(p[pool] - 0.5))])
    figs6 = {}
    for k, i in cases.items():
        plt.figure(figsize=(8, 5.5))
        shap.plots.waterfall(shap.Explanation(values=sv[PRIMARY][i], base_values=ev[PRIMARY], data=X[i], feature_names=feats),
                             max_display=10, show=False)
        figs6[k] = plt.gcf()
        figs6[k].suptitle(f"ecg_id {int(te.loc[i, 'ecg_id'])} | true {'Abnormal' if y[i] else 'Normal'} | p(Abnormal) = {p[i]:.2f}", fontsize=9)

    C.write_csv(t8, outs[0], extra={"primary_model": PRIMARY, "note": "generic readings are to be verified; SHAP explains the model"})
    C.write_csv(t8b, outs[1])
    C.write_png(fig4, outs[2])
    C.write_png(fig5, outs[3])
    for k, o in zip(("fig6_tp", "fig6_tn", "fig6_border"), outs[4:]):
        C.write_png(figs6[k], o)
    for f in [fig4, fig5, *figs6.values()]:
        C.show_and_close(f)


if __name__ == "__main__":
    main()
