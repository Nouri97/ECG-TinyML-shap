"""
v2_common.py - single source of truth for the v2 pipeline.

Holds CONFIG (the ONLY place paths are defined), write-safety, run metadata, logging, and the
metric / E2-diagnostic functions shared by v2_01, v2_02 and v2_03.

Rules enforced here
  1. Every written file lives under CONFIG["output_dir"] and has "_v2" in its file name.
  2. Nothing is ever written under read_only / frozen / checkpoints.
  3. An existing file is never overwritten: FileExistsError("Refusing to overwrite: ...").
     Files are created with exclusive mode ('x'), so a race cannot overwrite either.
  4. Every output gets a companion "<name>.meta_v2.json": created_at, git commit, script hash,
     seed, library versions and a CONFIG snapshot.
"""
import datetime as dt
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from importlib import metadata

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

# --------------------------------------------------------------------------- #
# CONFIG - edit paths here only
# --------------------------------------------------------------------------- #
CONFIG = {
    "root":       "/content/drive/MyDrive/ECG_Research",
    "output_dir": "/content/drive/MyDrive/ECG_Research/results_v2",
    "read_only":  "/content/drive/MyDrive/ECG_Research/results",
    "frozen":     "/content/drive/MyDrive/ECG_Research/freeze_2026-10-08",
    # additional protected folder (never written)
    "checkpoints": "/content/drive/MyDrive/ECG_Research/checkpoints",
    # read-only inputs
    "reference_csv": "",  # OPTIONAL: your own CSV (ecg_id, qrs_duration_ms) built from an external reference, e.g. PTB-XL+; "" = skip
    "cpsc_dir": "/content/cpsc2018",  # EDIT: folder holding the CPSC-2018 .hea/.mat records (any sub-folders)
    "ptbxl_dir": "/content/ptb-xl",  # EDIT: folder with ptbxl_database.csv, scp_statements.csv, records500/
    "kaggle_normal_csv":   "/content/drive/MyDrive/ECG_Research/PTB-XL/ptbdb_normal.csv",
    "kaggle_abnormal_csv": "/content/drive/MyDrive/ECG_Research/PTB-XL/ptbdb_abnormal.csv",
    # file names inside the frozen folder
    "frozen_files": {
        "processed": "processed_ptbxl.csv",
        "model": "best_model_Gradient_Boosting.pkl",
        "table2": "Table2_external_validation.csv",
        "kaggle_features": "kaggle_features_df.joblib",
    },
    # expected properties of the OLD label table, used only to identify the old label rule
    "label_expectation": {"n_total": 21799, "fold10_n": 2198, "fold10_prevalence": 0.5851, "tol": 0.004},
    # output file names (all carry _v2)
    "files": {
        "ptbxl_features": "ptbxl_v2_features.csv",
        "cpsc_features": "cpsc2018_v2_features.csv",
        "cpsc_summary": "cpsc2018_v2_extraction_summary.json",
        "kaggle_features": "kaggle_v2_features.csv",
        "feature_definitions": "feature_definitions_v2.json",
        "table1": "Table1_model_comparison_v2.csv",
        "table2": "Table2_external_validation_v2.csv",
        "table2b": "Table2b_per_diagnosis_v2.csv",
        "fig8": "Fig8_internal_vs_external_v2.png",
        "table3": "Table3_ablation_v2.csv",
        "table10a": "Table10a_polarity_fold_scores_v2.csv",
        "table10b": "Table10b_polarity_summary_v2.csv",
        "table9a": "Table9a_landmark_agreement_v2.csv",
        "table9b": "Table9b_known_groups_v2.csv",
        "table9c": "Table9c_parameter_stability_v2.csv",
        "fig10": "Fig10_landmark_validation_v2.png",
        "scoring_page": "Scoring_sheet_page{n}_v2.png",
        "scoring_template": "Scoring_template_v2.csv",
        "scoring_key": "Scoring_key_v2.csv",
        "table8": "Table8_shap_importance_v2.csv",
        "table8b": "Table8b_shap_agreement_v2.csv",
        "fig4": "Fig4_shap_global_v2.png",
        "fig5": "Fig5_shap_beeswarm_v2.png",
        "fig6_tp": "Fig6_shap_waterfall_abnormal_tp_v2.png",
        "fig6_tn": "Fig6_shap_waterfall_normal_tn_v2.png",
        "fig6_border": "Fig6_shap_waterfall_borderline_error_v2.png",
        "table7": "Table7_quantization_accuracy_v2.csv",
        "table7a": "Table7a_footprint_v2.csv",
        "table7b": "Table7b_importance_stability_v2.csv",
        "fig9": "Fig9_int8_footprint_accuracy_v2.png",
        "table5": "Table5_cv_fold_scores_v2.csv",
        "table6a": "Table6a_cv_ranks_and_tests_v2.csv",
        "table6c": "Table6c_feature_sets_v2.csv",
        "fig7": "Fig7_critical_difference_v2.png",
        "table5_f15": "Table5_cv_fold_scores_F15_v2.csv",
        "table6a_f15": "Table6a_cv_ranks_and_tests_F15_v2.csv",
        "fig7_f15": "Fig7_critical_difference_F15_v2.png",
        "table4": "Table4_hyperparameters_v2.csv",
        "fig1": "Fig1_model_comparison_v2.png",
        "fig2": "Fig2_tinyml_tradeoff_v2.png",
        "fig3": "Fig3_confusion_matrices_v2.png",
        "fig_landmarks": "Fig_landmark_check_v2.png",
        "e2_gb_internal": "gb_internal_v2.csv",
        "e2_gb_external": "gb_external_v2.csv",
        "e2_metrics": "metrics_v2.csv",
        "e2_png": "diagnostics_v2.png",
        "e2_comparison": "E2_comparison_old_vs_new_v2.csv",
        "nk_records": "NK_crosscheck_records_v2.csv",
        "nk_summary": "NK_crosscheck_summary_v2.csv",
        "fig_nk": "Fig_nk_agreement_v2.png",
        "fig_nk_gallery": "Fig_nk_disagreement_gallery_v2.png",
    },
}

_SUBDIRS = {"features": "features", "chunks": "features/_chunks", "models": "models",
            "tables": "tables", "cv": "tables/_cv_chunks", "figures": "figures", "e2": "figures/e2_diagnostics",
            "logs": "logs"}

FEATURE_GROUPS = {
    "statistics": ["RMS_energy", "Kurtosis", "Skewness", "ZeroCrossRate"],
    "qrs": ["R_amplitude_mV", "Q_depth_mV", "S_depth_mV", "QRS_duration_ms"],
    "st_t": ["ST_level_mV", "ST_slope_mV_per_s", "T_amplitude_mV", "T_asymmetry", "J_to_Tpeak_ms"],
    "rhythm": ["RR_median_ms", "RR_cv"],
}
MORPHOLOGY_GROUPS = ("statistics", "qrs", "st_t")


def morphology_features():
    return [f for g in MORPHOLOGY_GROUPS for f in FEATURE_GROUPS[g]]


# --------------------------------------------------------------------------- #
# Path safety
# --------------------------------------------------------------------------- #
def _inside(path, base):
    p, b = os.path.realpath(path), os.path.realpath(base)
    return p == b or p.startswith(b + os.sep)


def _validate_config():
    for k in ("read_only", "frozen", "checkpoints"):
        if _inside(CONFIG["output_dir"], CONFIG[k]):
            raise PermissionError(f"output_dir must not be inside protected folder '{k}'")


def out_path(kind, filename):
    return os.path.join(CONFIG["output_dir"], _SUBDIRS[kind], filename)


def meta_path(path):
    return os.path.splitext(path)[0] + ".meta_v2.json"


def guard_target(path):
    """Raise unless `path` is a new file under output_dir with a _v2 name. Returns the real path."""
    _validate_config()
    if not _inside(path, CONFIG["output_dir"]):
        raise PermissionError(f"Write outside output_dir refused: {path}")
    for k in ("read_only", "frozen", "checkpoints"):
        if _inside(path, CONFIG[k]):
            raise PermissionError(f"Write into protected folder '{k}' refused: {path}")
    if "_v2" not in os.path.basename(path):
        raise ValueError(f"File name must contain '_v2': {path}")
    target_path = os.path.realpath(path)
    if os.path.exists(target_path):
        raise FileExistsError(f"Refusing to overwrite: {target_path}")
    return target_path


def preflight(paths):
    """Check every target (and its meta file) BEFORE any long computation starts."""
    clashes = []
    for p in paths:
        guard_target(p) if not os.path.exists(p) else clashes.append(p)
        if os.path.exists(meta_path(p)):
            clashes.append(meta_path(p))
    if clashes:
        raise FileExistsError("Refusing to overwrite (delete manually if a rerun is intended):\n  "
                              + "\n  ".join(sorted(set(clashes))))


# --------------------------------------------------------------------------- #
# Run metadata, writers, logger
# --------------------------------------------------------------------------- #
RUN = {"script": "interactive", "script_sha256": None, "common_sha256": None,
       "git_commit": "unavailable", "seed": None, "started_at": None}


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _git_commit(start_dir):
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=start_dir, capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "unavailable"
    except Exception:
        return "unavailable"


def init_run(script_file=None, seed=None):
    RUN["started_at"] = _now()
    RUN["seed"] = seed
    RUN["common_sha256"] = _sha256(__file__)
    start_dir = os.getcwd()
    if script_file and os.path.exists(script_file):
        RUN["script"] = os.path.basename(script_file)
        RUN["script_sha256"] = _sha256(script_file)
        start_dir = os.path.dirname(os.path.abspath(script_file))
    RUN["git_commit"] = _git_commit(start_dir)


def _versions():
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost", "wfdb",
                "joblib", "matplotlib"):
        try:
            out[pkg] = metadata.version(pkg)
        except Exception:
            out[pkg] = "not installed"
    return out


def _write_meta(path, extra=None):
    meta = {"file": os.path.basename(path), "created_at": _now(), "git_commit": RUN["git_commit"],
            "script": RUN["script"], "script_sha256": RUN["script_sha256"],
            "v2_common_sha256": RUN["common_sha256"], "seed": RUN["seed"],
            "libraries": _versions(), "config": CONFIG, "extra": extra or {}}
    with open(meta_path(path), "x", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


def _prepare(path):
    p = guard_target(path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def write_csv(df, path, extra=None):
    p = _prepare(path)
    with open(p, "x", newline="", encoding="utf-8") as f:
        df.to_csv(f, index=False)
    _write_meta(p, {"rows": int(len(df)), "columns": list(df.columns), **(extra or {})})
    return p


def write_json(obj, path, extra=None):
    p = _prepare(path)
    with open(p, "x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    _write_meta(p, extra)
    return p


def write_pickle(obj, path, extra=None):
    p = _prepare(path)
    with open(p, "xb") as f:
        joblib.dump(obj, f)
    _write_meta(p, extra)
    return p


def write_png(fig, path, extra=None):
    p = _prepare(path)
    with open(p, "xb") as f:
        fig.savefig(f, format="png", dpi=200, bbox_inches="tight")
    _write_meta(p, extra)
    return p


def show_and_close(fig):
    import matplotlib.pyplot as plt
    plt.show()
    plt.close(fig)


def get_logger(name):
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_file = out_path("logs", f"pipeline_run_v2_{name}_{stamp}.log")
    p = _prepare(log_file)
    logger = logging.getLogger(f"v2.{name}.{stamp}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(p, mode="x", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    for h in (fh, sh):
        h.setFormatter(fmt)
        logger.addHandler(h)
    _write_meta(p, {"type": "log"})
    return logger


# --------------------------------------------------------------------------- #
# Metrics and E2 diagnostics (same logic as external_diagnostics.py)
# --------------------------------------------------------------------------- #
def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)) - 1, 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def metrics(y, p, t=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    yh = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(prevalence=float(y.mean()), auc=roc_auc_score(y, p),
                pr_auc=average_precision_score(y, p), accuracy=accuracy_score(y, yh),
                precision=precision_score(y, yh, zero_division=0),
                recall=recall_score(y, yh, zero_division=0),
                specificity=tn / (tn + fp) if (tn + fp) else np.nan,
                f1=f1_score(y, yh, zero_division=0), mcc=matthews_corrcoef(y, yh),
                bal_acc=balanced_accuracy_score(y, yh), brier=brier_score_loss(y, p),
                ece=ece(y, p))


def boot_ci(y, p, n=500, seed=0):
    """95% bootstrap CI for AUC and F1 (rows are resampled; see caveat on pseudo-replication)."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    auc, f1 = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        auc.append(roc_auc_score(y[i], p[i]))
        f1.append(f1_score(y[i], p[i] >= 0.5, zero_division=0))
    q = lambda a: tuple(np.quantile(a, [0.025, 0.975]))
    return {"auc": q(auc), "f1": q(f1)}


def bootstrap_ci(y, p, n=1000, seed=0):
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        rows.append(metrics(y[i], p[i]))
    return pd.DataFrame(rows).quantile([0.025, 0.975])


def resample_to_prevalence(y, p, target, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    cur = len(pos) / len(y)
    if cur > target:
        pos = rng.choice(pos, int(target / (1 - target) * len(neg)), replace=False)
    else:
        neg = rng.choice(neg, int((1 - target) / target * len(pos)), replace=False)
    idx = np.concatenate([pos, neg])
    return y[idx], p[idx]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def platt_split(y, p, frac=0.10, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    idx = rng.permutation(len(y))
    k = int(frac * len(y))
    cal, ev = idx[:k], idx[k:]
    lr = LogisticRegression().fit(logit(p[cal]), y[cal])
    return y[ev], lr.predict_proba(logit(p[ev]))[:, 1], y[ev], p[ev]


def summarize(y, p, target_prev=None, seed=0):
    """One-column summary used for the old-vs-new comparison table."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    m = metrics(y, p)
    out = {"n": len(y), **m}
    out["always_abnormal_f1"] = 2 * y.mean() / (1 + y.mean())
    out["f1_minus_always_abnormal"] = m["f1"] - out["always_abnormal_f1"]
    out["brier_skill_score"] = 1 - m["brier"] / (y.mean() * (1 - y.mean()))
    ts = np.linspace(0.01, 0.99, 99)
    f1s = np.array([f1_score(y, p >= t, zero_division=0) for t in ts])
    k = int(f1s.argmax())
    out.update(tau_star=float(ts[k]), f1_at_tau_star=float(f1s[k]),
               abs_tau_star_minus_0_5=float(abs(ts[k] - 0.5)))
    if target_prev is not None:
        yr, pr = resample_to_prevalence(y, p, target_prev, seed)
        mr = metrics(yr, pr)
        out.update(f1_prev_matched=mr["f1"], mcc_prev_matched=mr["mcc"], auc_prev_matched=mr["auc"])
        try:  # Platt needs both classes in the 10% calibration split (always true for real-size sets)
            y2, pcal, _, praw = platt_split(y, p, seed=seed)
            mp, mraw = metrics(y2, pcal), metrics(y2, praw)
            out.update(platt_f1=mp["f1"], platt_mcc=mp["mcc"], platt_brier=mp["brier"],
                       platt_auc=mp["auc"], raw_brier_same_subset=mraw["brier"])
        except ValueError:
            out.update(platt_f1=np.nan, platt_mcc=np.nan, platt_brier=np.nan, platt_auc=np.nan,
                       raw_brier_same_subset=np.nan)
    return out


def e2_long_table(y_int, p_int, y_ext, p_ext, n_boot=1000, seed=0):
    """Same row structure as metrics_<model>.csv from external_diagnostics.py."""
    rows = []
    for name, y, p in (("internal", y_int, p_int), ("external", y_ext, p_ext)):
        m = metrics(y, p)
        ci = bootstrap_ci(y, p, n=n_boot, seed=seed)
        rows.append({"setting": name, **m, **{f"{c}_lo": ci.loc[0.025, c] for c in ci},
                     **{f"{c}_hi": ci.loc[0.975, c] for c in ci}})
    ye = np.asarray(y_ext).astype(int)
    rows.append({"setting": "external_always_abnormal", "prevalence": ye.mean(),
                 "f1": f1_score(ye, np.ones_like(ye)), "mcc": 0.0, "bal_acc": 0.5, "auc": 0.5,
                 "pr_auc": ye.mean()})
    yr, pr = resample_to_prevalence(ye, p_ext, float(np.mean(y_int)), seed)
    rows.append({"setting": "external_prevalence_matched", **metrics(yr, pr)})
    ts = np.linspace(0.01, 0.99, 99)
    f1s = np.array([f1_score(ye, np.asarray(p_ext) >= t, zero_division=0) for t in ts])
    t_star = float(ts[f1s.argmax()])
    rows.append({"setting": "external_tau_star", "tau_star": t_star, "f1": float(f1s.max()),
                 "abs_tau_star_minus_0.5": abs(t_star - 0.5)})
    try:
        y2, pcal, _, praw = platt_split(ye, p_ext, seed=seed)
        rows.append({"setting": "external_platt_10pct_cal", **metrics(y2, pcal)})
        rows.append({"setting": "external_raw_same_eval_subset", **metrics(y2, praw)})
    except ValueError:  # calibration split with a single class (tiny sets only)
        pass
    return pd.DataFrame(rows), t_star


def plot_e2(y_int, p_int, y_ext, p_ext, t_star, title):
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    ts = np.linspace(0.01, 0.99, 99)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for k, y, p, c in (("internal", y_int, p_int, "tab:blue"), ("external", y_ext, p_ext, "tab:red")):
        y, p = np.asarray(y), np.asarray(p)
        pr_, rc_, _ = precision_recall_curve(y, p)
        ax[0].plot(rc_, pr_, c=c, label=f"{k} (prev={y.mean():.2f})")
        ax[0].axhline(y.mean(), c=c, ls=":", lw=1)
        ax[1].plot(ts, [f1_score(y, p >= t, zero_division=0) for t in ts], c=c, label=k)
        fr, mp = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax[2].plot(mp, fr, "o-", c=c, label=k)
    ax[1].axvline(0.5, c="k", ls="--", lw=1)
    ax[1].axvline(t_star, c="tab:red", ls=":", lw=1)
    ax[2].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    ax[1].set(xlabel="Threshold tau", ylabel="F1", title="F1 vs threshold")
    ax[2].set(xlabel="Mean predicted probability", ylabel="Observed fraction", title="Reliability")
    for a in ax:
        a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


_validate_config()
