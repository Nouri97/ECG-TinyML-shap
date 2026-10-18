"""
v2_03_compare_e2.py - E2 diagnostics, old (frozen) vs new (results_v2), side by side.

OLD  : recomputed from freeze_2026-10-08 (frozen model + frozen features; Kaggle labels rebuilt from the
       row counts of ptbdb_normal / ptbdb_abnormal; Q_depth NaN imputed with the training-fold median)
       plus the GB row of the frozen Table 2 "as reported in the manuscript".
NEW  : models/gradient_boosting_v2.pkl on PTB-XL fold 10 and on every external v2 feature file present.
Writes: tables/E2_comparison_old_vs_new_v2.csv and figures/e2_diagnostics/{gb_internal,gb_external,
metrics,diagnostics}_v2.* for the first available new external set.
Refuses to run until at least one new external feature file exists (REQUIRE_EXTERNAL), so a partial
comparison can never block the final one.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
import v2_common as C  # noqa: E402
from v2_common import CONFIG  # noqa: E402

SEED = 42
N_BOOT = 1000
REQUIRE_EXTERNAL = True
OLD_FEATURES = ["R_amplitude", "Q_depth", "QRS_width_ms", "ST_level", "T_amplitude", "T_symmetry",
                "beat_duration_ms", "PR_proxy_ms", "RMS_energy", "Kurtosis", "Skewness", "ZeroCrossRate"]
EXTERNAL_KEYS = (("CPSC2018", "cpsc_features"), ("Kaggle", "kaggle_features"))


def proba_any(model, X):
    names = getattr(model, "feature_names_in_", None)
    return model.predict_proba(X[list(names)] if names is not None else X.values)[:, 1]


def count_labels(path):
    s = pd.read_csv(path, header=None, usecols=[187])[187]
    return int(pd.to_numeric(s, errors="coerce").notna().sum())


def old_side():
    fz, ff = CONFIG["frozen"], CONFIG["frozen_files"]
    df = pd.read_csv(os.path.join(fz, ff["processed"]))
    test, train = df[df["fold"] == 10], df[df["fold"] != 10]
    med = train[OLD_FEATURES].median()
    Xi, yi = test[OLD_FEATURES].astype(float), test["label"].astype(int).to_numpy()
    Xe = joblib.load(os.path.join(fz, ff["kaggle_features"]))[OLD_FEATURES].astype(float).fillna(med)
    n0, n1 = count_labels(CONFIG["kaggle_normal_csv"]), count_labels(CONFIG["kaggle_abnormal_csv"])
    if n0 + n1 != len(Xe):
        raise ValueError(f"Kaggle label reconstruction invalid: {n0}+{n1} != {len(Xe)}")
    ye = np.r_[np.zeros(n0, dtype=int), np.ones(n1, dtype=int)]
    model = joblib.load(os.path.join(fz, ff["model"]))
    t2 = pd.read_csv(os.path.join(fz, ff["table2"]))
    row = t2[t2["Model"].astype(str).str.contains("Gradient", case=False)].iloc[0]
    reported = {"auc": row["Ext_AUC"], "f1": row["Ext_F1"], "recall": row["Ext_Recall"], "accuracy": row["Ext_Acc"]}
    return yi, proba_any(model, Xi), ye, proba_any(model, Xe), reported


def new_side(log):
    model = joblib.load(C.out_path("models", "gradient_boosting_v2.pkl"))
    feats = list(model.feature_names_in_)
    df = pd.read_csv(C.out_path("features", CONFIG["files"]["ptbxl_features"]))
    ok = df[df["fail_reason"].fillna("") == ""]
    te = ok[ok["strat_fold"] == 10]
    yi, pi = te["label"].to_numpy(), model.predict_proba(te[feats])[:, 1]
    ext = {}
    for dname, key in EXTERNAL_KEYS:
        p = C.out_path("features", CONFIG["files"][key])
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        d = d[d["fail_reason"].fillna("") == ""]
        miss = [f for f in feats if f not in d.columns]
        if miss:
            log.warning("%s skipped: missing features %s", dname, miss)
            continue
        if d[feats].isna().any().any():
            raise ValueError(f"NaN in model features of {dname}")
        ext[dname] = (d["label"].to_numpy(), model.predict_proba(d[feats])[:, 1])
    return yi, pi, ext


def main():
    C.init_run(globals().get("__file__"), SEED)
    log = C.get_logger("compare_e2")
    files = CONFIG["files"]
    targets = [C.out_path("tables", files["e2_comparison"])] + \
              [C.out_path("e2", files[k]) for k in ("e2_gb_internal", "e2_gb_external", "e2_metrics", "e2_png")]
    C.preflight(targets)

    yi_n, pi_n, ext_new = new_side(log)
    if REQUIRE_EXTERNAL and not ext_new:
        raise RuntimeError("No new external feature file yet (cpsc2018_v2 / kaggle_v2). Nothing written.")
    yi_o, pi_o, ye_o, pe_o, reported = old_side()

    cols = {"old_internal_fold10": C.summarize(yi_o, pi_o),
            "old_kaggle_external_frozen_features": C.summarize(ye_o, pe_o, target_prev=float(yi_o.mean()), seed=SEED),
            "old_kaggle_Table2_as_reported": reported,
            "new_internal_fold10": C.summarize(yi_n, pi_n)}
    for dname, (y, p) in ext_new.items():
        cols[f"new_external_{dname}"] = C.summarize(y, p, target_prev=float(yi_n.mean()), seed=SEED)
    comp = pd.DataFrame(cols).reset_index(names="metric")
    pd.set_option("display.width", 220)
    log.info("\n%s", comp.round(4).to_string(index=False))

    dname, (ye_n, pe_n) = next(iter(ext_new.items()))
    long_tbl, t_star = C.e2_long_table(yi_n, pi_n, ye_n, pe_n, n_boot=N_BOOT, seed=SEED)   # compute first ...
    fig = C.plot_e2(yi_n, pi_n, ye_n, pe_n, t_star, f"E2 diagnostics v2 - Gradient Boosting - {dname}")

    C.write_csv(comp, C.out_path("tables", files["e2_comparison"]),                         # ... then write
                extra={"old_source": CONFIG["frozen"], "new_source": CONFIG["output_dir"],
                       "external_sets_new": list(ext_new)})
    C.write_csv(pd.DataFrame({"y_true": yi_n, "y_score": pi_n}), C.out_path("e2", files["e2_gb_internal"]))
    C.write_csv(pd.DataFrame({"y_true": ye_n, "y_score": pe_n}), C.out_path("e2", files["e2_gb_external"]),
                extra={"external_set": dname})
    C.write_csv(long_tbl, C.out_path("e2", files["e2_metrics"]), extra={"external_set": dname})
    C.write_png(fig, C.out_path("e2", files["e2_png"]))
    C.show_and_close(fig)
    log.info("Done. Bootstrap CIs resample rows; if one patient contributes several rows they are optimistic.")


if __name__ == "__main__":
    main()
