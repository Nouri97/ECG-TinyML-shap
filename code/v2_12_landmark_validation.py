"""
v2_12_landmark_validation.py - automatic validation of the v2 landmarks (no clinician).

Four independent pieces of evidence, none of which is ground truth. Report them as such.

A1  R-peak detection vs WFDB XQRS (a published detector). XQRS detects QRS complexes ONLY: it validates beat
    positions and heart rate, not QRS duration, J point or T peak.
A2  Delineation vs NeuroKit2 (methods dwt, cwt, peak) on the same N_RECORDS records as v2_04. A pre-specified gate
    decides whether a comparison method can be used for a landmark: >= 80% coverage AND a physiologically plausible
    median (QRS duration 60-140 ms, J offset 20-100 ms after R, T peak 150-350 ms after R). Methods that fail the gate
    are listed with their median/IQR and NO agreement statistics are computed (a reference that is implausible cannot
    validate anything). Optional reference: a CSV you prepare from PTB-XL+ or another source (CONFIG['reference_csv'],
    columns ecg_id, qrs_duration_ms); its file layout was not verified here, and a global 12-lead duration is expected to be
    longer than a lead-II proxy, so only rank agreement (Spearman) and bias are interpretable.
B   Parameter stability: the same records are re-extracted with perturbed parameters; Spearman agreement with the
    baseline extraction and median absolute change per feature.
C   Known-groups validity: does each measurement differ between diagnoses and normal records in the clinically expected
    direction? Pre-specified criterion: direction-adjusted AUC >= 0.60. This is construct validity, not accuracy, and
    absolute clinical thresholds (e.g. QRS > 120 ms) are NOT used: a single-lead slope-based width is expected to be
    shorter than a clinical 12-lead duration.
Outputs: Table9a (A1, A2, reference), Table9b (C), Table9c (B), Fig10.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
import v2_common as C  # noqa: E402
from v2_common import CONFIG  # noqa: E402
import v2_01_extract_features as X  # noqa: E402

SEED = 42
N_RECORDS = 500
N_JOBS = -1
N_BOOT = 200
NK_METHODS = ("dwt", "cwt", "peak")
TOL_MS = {"qrs_duration": 20.0, "j_point": 20.0, "t_peak": 40.0}
PLAUSIBLE = {"qrs_duration": (60.0, 140.0), "j_point": (20.0, 100.0), "t_peak": (150.0, 350.0)}
MIN_COVERAGE = 0.8
RPEAK_TOL_SAMPLES = 50          # 100 ms at 500 Hz
AUC_PASS = 0.60
OURS = {"qrs_duration": "QRS_duration_ms", "j_point": "qc_J_ms", "t_peak": "qc_Tpeak_ms"}
PERTURBATIONS = {
    "slope_frac=0.05": {"slope_frac": 0.05}, "slope_frac=0.15": {"slope_frac": 0.15},
    "env_win_ms=12": {"env_win_ms": 12}, "env_win_ms=30": {"env_win_ms": 30},
    "baseline_win=(-110,-80)": {"baseline_win_ms": (-110, -80)}, "baseline_win=(-70,-40)": {"baseline_win_ms": (-70, -40)},
    "st_win=(50,70)": {"st_win_ms": (50, 70)}, "st_win=(70,90)": {"st_win_ms": (70, 90)},
    "t_win_start=100": {"t_win_ms": (100, 400)}, "det_thr_frac=0.2": {"det_thr_frac": 0.2},
    "det_thr_frac=0.4": {"det_thr_frac": 0.4},
}
STAB_FEATURES = ["QRS_duration_ms", "ST_level_mV", "T_amplitude_mV", "T_asymmetry", "J_to_Tpeak_ms", "Q_depth_mV", "S_depth_mV", "R_amplitude_mV"]
# (dataset, positive diagnosis, normal-only label, feature, expected direction, transform)
KNOWN_GROUPS = [
    ("CPSC2018", "LBBB", "Normal", "QRS_duration_ms", +1, None), ("CPSC2018", "RBBB", "Normal", "QRS_duration_ms", +1, None),
    ("CPSC2018", "STD", "Normal", "ST_level_mV", -1, None), ("CPSC2018", "STE", "Normal", "ST_level_mV", +1, None),
    ("CPSC2018", "STD", "Normal", "ST_level_mV", +1, "abs"), ("CPSC2018", "STE", "Normal", "ST_level_mV", +1, "abs"),
    ("CPSC2018", "AF", "Normal", "RR_cv", +1, None), ("CPSC2018", "PVC", "Normal", "RR_cv", +1, None),
    ("CPSC2018", "PAC", "Normal", "RR_cv", +1, None),
    ("PTB-XL", "CD", "NORM", "QRS_duration_ms", +1, None), ("PTB-XL", "STTC", "NORM", "T_amplitude_mV", -1, None),
    ("PTB-XL", "STTC", "NORM", "ST_level_mV", +1, "abs"), ("PTB-XL", "MI", "NORM", "Q_depth_mV", +1, None),
]


# ------------------------------------------------------------------ comparison methods (workers: no X inside)
def nk_all(sig, fs=500):
    import neurokit2 as nk
    out = {}
    try:
        clean = nk.ecg_clean(sig, sampling_rate=fs, method="neurokit")
        _, info = nk.ecg_peaks(clean, sampling_rate=fs, method="neurokit")
        r = np.asarray(info["ECG_R_Peaks"], dtype=float)
        if len(r) < 3:
            return out
        for m in NK_METHODS:
            try:
                _, w = nk.ecg_delineate(clean, info["ECG_R_Peaks"], sampling_rate=fs, method=m)
            except Exception:
                continue

            def arr(key):
                a = np.asarray(w.get(key, []), dtype=float)
                return a[: len(r)] if len(a) >= len(r) else np.r_[a, np.full(len(r) - len(a), np.nan)]

            def med(x):
                x = x[np.isfinite(x)]
                return float(np.median(x)) * 1000.0 / fs if len(x) >= 3 else np.nan

            on, off, tp = arr("ECG_R_Onsets"), arr("ECG_R_Offsets"), arr("ECG_T_Peaks")
            out[m] = {"qrs_duration": med(off - on), "j_point": med(off - r), "t_peak": med(tp - r)}
    except Exception:
        pass
    return out


def xqrs_peaks(sig, fs=500):
    import wfdb.processing as wp
    try:
        return np.asarray(wp.xqrs_detect(sig=sig, fs=fs, verbose=False), dtype=int)
    except Exception:
        return np.array([], dtype=int)


def match(a, b, tol):
    """Number of elements of `a` that have an element of `b` within tol samples."""
    if len(a) == 0 or len(b) == 0:
        return 0
    b = np.sort(b)
    i = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    d = np.minimum(np.abs(a - b[i - 1]), np.abs(a - b[i]))
    return int((d <= tol).sum())


def agree(a, b, tol):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 10:
        return {"n": int(len(a))}
    d = a - b
    sd = d.std(ddof=1)
    return {"n": int(len(a)), "bias_ms": d.mean(), "sd_ms": sd, "loa_low_ms": d.mean() - 1.96 * sd, "loa_high_ms": d.mean() + 1.96 * sd,
            "spearman_rho": spearmanr(a, b).statistic, "median_abs_diff_ms": float(np.median(np.abs(d))),
            "tol_ms": tol, "pct_within_tol": float(np.mean(np.abs(d) <= tol) * 100)}


def auc_adj(x, y, direction, boot, rng):
    a = roc_auc_score(y, direction * x)
    bs = []
    for _ in range(boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            bs.append(roc_auc_score(y[i], direction * x[i]))
    return (a, float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))) if bs else (a, np.nan, np.nan)


def main():
    C.init_run(globals().get("__file__"), SEED)
    log = C.get_logger("landmark_validation")
    files = CONFIG["files"]
    outs = [C.out_path("tables", files[k]) for k in ("table9a", "table9b", "table9c")] + [C.out_path("figures", files["fig10"])]
    C.preflight(outs)
    import matplotlib.pyplot as plt
    import wfdb

    feat = pd.read_csv(C.out_path("features", files["ptbxl_features"]))
    ok = feat[feat["fail_reason"].fillna("") == ""]
    db = pd.read_csv(os.path.join(CONFIG["ptbxl_dir"], "ptbxl_database.csv"), index_col="ecg_id")
    pick = ok.sample(n=min(N_RECORDS, len(ok)), random_state=SEED).reset_index(drop=True)      # same records as v2_04
    sigs = []
    for eid in pick["ecg_id"]:
        r = wfdb.rdrecord(os.path.join(CONFIG["ptbxl_dir"], db.loc[int(eid), "filename_hr"]))
        sigs.append(r.p_signal[:, r.sig_name.index("II")][: 10 * X.FS].astype(float))
    log.info("Loaded %d signals", len(sigs))

    rows9a = []
    # ---- A1: R peaks vs XQRS --------------------------------------------------------------------
    edge = X.FS // 2
    rr_o, rr_x, tp_ref, n_ref, tp_our, n_our = [], [], 0, 0, 0, 0
    for s in sigs:
        ours = X.detect_r_peaks(X.preprocess(s, X.FS), X.FS)
        ref = xqrs_peaks(s)
        ours, ref = ours[(ours > edge) & (ours < len(s) - edge)], ref[(ref > edge) & (ref < len(s) - edge)]
        tp_ref += match(ref, ours, RPEAK_TOL_SAMPLES)
        tp_our += match(ours, ref, RPEAK_TOL_SAMPLES)
        n_ref, n_our = n_ref + len(ref), n_our + len(ours)
        if len(ours) > 2 and len(ref) > 2:
            rr_o.append(np.median(np.diff(ours)) * 1000.0 / X.FS)
            rr_x.append(np.median(np.diff(ref)) * 1000.0 / X.FS)
    rows9a.append({"component": "R_peak_vs_WFDB_XQRS", "landmark": "R peak", "subset": "all", "status": "reference = XQRS (detector only)",
                   "n": len(sigs), "sensitivity_vs_XQRS": tp_ref / max(n_ref, 1), "ppv_vs_XQRS": tp_our / max(n_our, 1),
                   "tol_ms": RPEAK_TOL_SAMPLES * 1000 / X.FS, **{f"RR_{k}": v for k, v in agree(rr_o, rr_x, 20.0).items()}})
    log.info("R peaks vs XQRS: sensitivity %.4f, PPV %.4f (tolerance 100 ms)", tp_ref / max(n_ref, 1), tp_our / max(n_our, 1))

    # ---- A2: delineation vs NeuroKit2 ------------------------------------------------------------
    nk_res = Parallel(n_jobs=N_JOBS)(delayed(nk_all)(s) for s in sigs)
    inv, lab = pick["polarity_inverted"].astype(bool).to_numpy(), pick["label"].to_numpy()
    subsets = {"all": np.ones(len(pick), bool), "polarity_normal": ~inv, "polarity_inverted": inv,
               "label_normal": lab == 0, "label_abnormal": lab == 1}
    qualified = {}
    for m in NK_METHODS:
        for lm, col in OURS.items():
            ref = np.array([r.get(m, {}).get(lm, np.nan) for r in nk_res], float)
            fin = ref[np.isfinite(ref)]
            cov, med = len(fin) / len(ref), (float(np.median(fin)) if len(fin) else np.nan)
            lo, hi = PLAUSIBLE[lm]
            gate = cov >= MIN_COVERAGE and lo <= med <= hi
            base = {"component": f"delineation_vs_NeuroKit2_{m}", "landmark": lm, "coverage": cov, "ref_median_ms": med,
                    "ref_p05_ms": float(np.percentile(fin, 5)) if len(fin) else np.nan,
                    "ref_p95_ms": float(np.percentile(fin, 95)) if len(fin) else np.nan}
            if not gate:
                rows9a.append({**base, "subset": "all", "status": "reference implausible or low coverage: no agreement computed"})
                continue
            qualified[(m, lm)] = ref
            for sn, mask in subsets.items():
                rows9a.append({**base, "subset": sn, "status": "qualified", **agree(pick.loc[mask, col], ref[mask], TOL_MS[lm])})
            log.info("%s %s qualified (median %.0f ms)", m, lm, med)
    ref_csv = CONFIG.get("reference_csv", "")
    if ref_csv and os.path.exists(ref_csv):
        rc = pd.read_csv(ref_csv).set_index("ecg_id")
        v = pick["ecg_id"].map(rc["qrs_duration_ms"]).to_numpy(float)
        rows9a.append({"component": "reference_csv", "landmark": "qrs_duration", "subset": "all",
                       "status": "user-supplied reference (global duration expected longer than lead-II proxy)",
                       **agree(pick["QRS_duration_ms"], v, 20.0)})
    else:
        rows9a.append({"component": "reference_csv", "landmark": "qrs_duration", "subset": "all", "status": "not provided"})
    if not qualified:
        log.warning("No NeuroKit2 method passed the plausibility gate: delineation is NOT validated against any automatic reference.")

    # ---- B: parameter stability ---------------------------------------------------------------------
    def extract_all(overrides):
        saved = {k: X.P[k] for k in overrides}
        X.P.update(overrides)
        try:
            return pd.DataFrame([X.extract_from_signal(s, X.FS)[0] for s in sigs])
        finally:
            X.P.update(saved)

    base_df = extract_all({})
    rows9c = []
    for name, ov in PERTURBATIONS.items():
        d = extract_all(ov)
        failed = float((d["fail_reason"].fillna("") != "").mean())
        for f in STAB_FEATURES:
            m = (base_df["fail_reason"].fillna("") == "") & (d["fail_reason"].fillna("") == "")
            a, b = base_df.loc[m, f].to_numpy(float), d.loc[m, f].to_numpy(float)
            rows9c.append({"perturbation": name, "feature": f, "n": int(m.sum()), "pct_failed": failed * 100,
                           "spearman_vs_baseline": spearmanr(a, b).statistic if len(a) > 10 else np.nan,
                           "median_abs_change": float(np.median(np.abs(a - b))) if len(a) else np.nan,
                           "median_baseline": float(np.median(a)) if len(a) else np.nan})
    t9c = pd.DataFrame(rows9c)

    # ---- C: known groups ---------------------------------------------------------------------------------
    cpsc_path = C.out_path("features", files["cpsc_features"])
    data = {"PTB-XL": (ok, "scp_superclasses")}
    if os.path.exists(cpsc_path):
        cp = pd.read_csv(cpsc_path)
        data["CPSC2018"] = (cp[cp["fail_reason"].fillna("") == ""], "dx_classes")
    rng = np.random.default_rng(SEED)
    rows9b = []
    for ds, dx, neg, f, dirn, tr in KNOWN_GROUPS:
        if ds not in data:
            continue
        d, col = data[ds]
        cls = d[col].fillna("").astype(str)
        pos = cls.str.split("|").apply(lambda L: dx in L).to_numpy()
        negm = (cls == neg).to_numpy()
        sel = pos | negm
        if pos.sum() < 10 or negm.sum() < 10:      # too few records for a meaningful AUC (tiny test sets only)
            continue
        x = d[f].to_numpy(float)
        x = np.abs(x) if tr == "abs" else x
        a, lo, hi = auc_adj(x[sel], pos[sel].astype(int), dirn, N_BOOT, rng)
        rows9b.append({"dataset": ds, "diagnosis": dx, "vs": neg, "feature": ("|" + f + "|") if tr == "abs" else f,
                       "expected_direction": "higher in diagnosis" if dirn > 0 else "lower in diagnosis", "n_pos": int(pos.sum()),
                       "n_neg": int(negm.sum()), "median_pos": float(np.median(x[pos])), "median_neg": float(np.median(x[negm])),
                       "auc_expected_direction": a, "ci_lo": lo, "ci_hi": hi, "meets_criterion_auc_ge_0_60": bool(a >= AUC_PASS)})
    t9b = pd.DataFrame(rows9b)
    t9a = pd.DataFrame(rows9a)
    pd.set_option("display.width", 250)
    log.info("\n%s", t9b.round(3).to_string(index=False))
    log.info("\n%s", t9c[t9c["feature"].isin(["QRS_duration_ms", "ST_level_mV", "T_amplitude_mV"])].round(3).to_string(index=False))

    # figure (computed before any write)
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    a0 = ax[0, 0]
    if len(rr_o) > 10:
        m_, d_ = (np.array(rr_o) + np.array(rr_x)) / 2, np.array(rr_o) - np.array(rr_x)
        a0.scatter(m_, d_, s=8, alpha=0.4)
        a0.axhline(d_.mean(), c="k")
        a0.axhline(d_.mean() + 1.96 * d_.std(), c="k", ls="--")
        a0.axhline(d_.mean() - 1.96 * d_.std(), c="k", ls="--")
    a0.set(title="Median RR: v2 vs WFDB XQRS", xlabel="mean (ms)", ylabel="v2 - XQRS (ms)")
    a1 = ax[0, 1]
    if qualified:
        (m, lm), ref = next(iter(qualified.items()))
        mine = pick[OURS[lm]].to_numpy(float)
        f_ = np.isfinite(ref)
        a1.scatter((mine[f_] + ref[f_]) / 2, mine[f_] - ref[f_], s=8, alpha=0.4)
        a1.set(title=f"{lm}: v2 vs NeuroKit2 {m}", xlabel="mean (ms)", ylabel="v2 - reference (ms)")
    else:
        a1.text(0.5, 0.5, "No NeuroKit2 method passed the\nplausibility gate: no delineation\nreference available", ha="center", va="center")
        a1.set_axis_off()
    a2 = ax[1, 0]
    yy = np.arange(len(t9b))
    if len(t9b):
        a2.errorbar(t9b["auc_expected_direction"], yy, xerr=[t9b["auc_expected_direction"] - t9b["ci_lo"], t9b["ci_hi"] - t9b["auc_expected_direction"]],
                    fmt="none", ecolor="gray")
        a2.scatter(t9b["auc_expected_direction"], yy, c=np.where(t9b["meets_criterion_auc_ge_0_60"], "tab:green", "tab:red"), zorder=3)
        a2.set_yticks(yy, [f"{r.dataset}: {r.diagnosis} - {r.feature}" for r in t9b.itertuples()], fontsize=7)
        a2.axvline(0.5, c="k", lw=0.8)
        a2.axvline(AUC_PASS, c="gray", ls=":")
    a2.set(title="Known-groups validity (AUC in the expected direction)", xlabel="AUC")
    a3 = ax[1, 1]
    sub = t9c[t9c["feature"].isin(["QRS_duration_ms", "ST_level_mV", "T_amplitude_mV", "J_to_Tpeak_ms"])]
    for f in sub["feature"].unique():
        s_ = sub[sub["feature"] == f]
        a3.plot(range(len(s_)), s_["spearman_vs_baseline"], "o-", label=f)
    a3.set_xticks(range(len(PERTURBATIONS)), list(PERTURBATIONS), rotation=60, ha="right", fontsize=7)
    a3.set(title="Parameter stability (Spearman vs baseline)", ylabel="rho")
    a3.legend(fontsize=7)
    fig.tight_layout()

    C.write_csv(t9a, outs[0], extra={"n_records": len(sigs), "qualified_comparisons": [list(k) for k in qualified]})
    C.write_csv(t9b, outs[1])
    C.write_csv(t9c, outs[2])
    C.write_png(fig, outs[3])
    C.show_and_close(fig)


if __name__ == "__main__":
    main()
