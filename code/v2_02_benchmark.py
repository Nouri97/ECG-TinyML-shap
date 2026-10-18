"""
v2_02_benchmark.py - 11-model benchmark on the v2 PTB-XL features (+ ablation, + external sets if present).

Protocol (same as the submitted paper): train on PTB-XL folds 1-9, test on fold 10.
  * Tree models use the hyperparameters of manuscript Section 6.5 (fixed).
  * MLP / KNN / Logistic Regression / Linear SVM: small documented grid, selected by AUC on fold 9
    (trained on folds 1-8), then refitted on folds 1-9. Their original settings were lost.
  * Non-tree models: StandardScaler fitted on the training data inside the pipeline.
  * LightGBM: subsample_freq=1 is set so that subsample=0.8 is actually applied.
  * Linear SVM: sigmoid-calibrated (CalibratedClassifierCV, cv=3) to obtain probabilities.
TinyML columns: size = serialised float model (joblib bytes). INT8 quantisation is NOT done here
(needs the quantisation method fixed first; see the next step). Latency = median wall-clock of 200
single-record predict_proba calls, 1 thread, including library overhead.

Outputs: models/*_v2.pkl, tables/Table1,3,4, figures/Fig1-3. External validation (Table 2) = v2_07_external_eval.py.
Final feature set F15 (INCLUDE_RHYTHM = True); EXCLUDE_FEATURES can drop individual features.
"""
import io
import itertools
import os
import sys
import time
import warnings

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
import v2_common as C  # noqa: E402
from v2_common import CONFIG  # noqa: E402

SEED = 42
EXCLUDE_FEATURES = []      # e.g. ["T_asymmetry", "J_to_Tpeak_ms"]
INCLUDE_RHYTHM = True      # F15 = 13 morphology + RR_median_ms + RR_cv (adopted by the pre-specified CV rule, Table 6c)
ABLATION_SEEDS = 1
N_BOOT = 500
N_LATENCY = 200

MODELS = {
    "Gradient Boosting": dict(slug="gradient_boosting", grid=None, make=lambda s: GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, random_state=s)),
    "LightGBM": dict(slug="lightgbm", grid=None, make=lambda s: lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31, subsample=0.8, subsample_freq=1,
        random_state=s, n_jobs=1, verbose=-1)),
    "XGBoost": dict(slug="xgboost", grid=None, make=lambda s: xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, random_state=s, n_jobs=1,
        eval_metric="logloss", tree_method="hist")),
    "MLP (2-layer)": dict(slug="mlp", grid={"hidden": [(32, 16), (64, 32)], "alpha": [1e-4, 1e-2]},
                          make=lambda s, hidden, alpha: make_pipeline(StandardScaler(), MLPClassifier(
                              hidden_layer_sizes=hidden, alpha=alpha, early_stopping=True, max_iter=300,
                              random_state=s))),
    "Random Forest": dict(slug="random_forest", grid=None, make=lambda s: RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=s, n_jobs=1)),
    "Extra Trees": dict(slug="extra_trees", grid=None, make=lambda s: ExtraTreesClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=s, n_jobs=1)),
    "KNN": dict(slug="knn", grid={"k": [5, 15, 31]},
                make=lambda s, k: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))),
    "Decision Tree": dict(slug="decision_tree", grid=None, make=lambda s: DecisionTreeClassifier(random_state=s)),
    "Logistic Regression": dict(slug="logistic_regression", grid={"C": [0.01, 0.1, 1, 10]},
                                make=lambda s, C: make_pipeline(StandardScaler(), LogisticRegression(
                                    C=C, max_iter=2000))),
    "Linear SVM": dict(slug="linear_svm", grid={"C": [0.01, 0.1, 1]},
                       make=lambda s, C: make_pipeline(StandardScaler(), CalibratedClassifierCV(
                           LinearSVC(C=C, dual=False, max_iter=5000), method="sigmoid", cv=3))),
    "Naive Bayes": dict(slug="naive_bayes", grid=None, make=lambda s: GaussianNB()),
}


def proba(model, X):
    return model.predict_proba(X)[:, 1]


def tune(spec, Xa, ya, Xb, yb, seed, log, name):
    if spec["grid"] is None:
        return {}, []
    keys = list(spec["grid"])
    trials = []
    for vals in itertools.product(*[spec["grid"][k] for k in keys]):
        params = dict(zip(keys, vals))
        m = spec["make"](seed, **params).fit(Xa, ya)
        trials.append({**params, "val_auc_fold9": roc_auc_score(yb, proba(m, Xb))})
    best = max(trials, key=lambda t: t["val_auc_fold9"])
    log.info("%s: best %s (val AUC %.4f)", name, {k: best[k] for k in keys}, best["val_auc_fold9"])
    return {k: best[k] for k in keys}, trials


def model_size_kb(model):
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return len(buf.getvalue()) / 1024.0


def latency_ms(model, X, n=N_LATENCY, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = X.to_numpy()[rng.choice(len(X), size=min(n, len(X)), replace=False)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in rows[:10]:
            model.predict_proba(r.reshape(1, -1))
        times = []
        for r in rows:
            t0 = time.perf_counter()
            model.predict_proba(r.reshape(1, -1))
            times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def load_ok(path, feats):
    df = pd.read_csv(path)
    df["fail_reason"] = df["fail_reason"].fillna("")
    ok = df[df["fail_reason"] == ""].copy()
    missing = [f for f in feats if f not in ok.columns]
    if missing:
        raise KeyError(f"{os.path.basename(path)} lacks features {missing}")
    if ok[feats].isna().any().any():
        raise ValueError(f"NaN in model features of {os.path.basename(path)}")
    return df, ok


def main():
    C.init_run(globals().get("__file__"), SEED)
    log = C.get_logger("benchmark")
    files = CONFIG["files"]
    feats = [f for f in C.morphology_features() if f not in EXCLUDE_FEATURES]
    if INCLUDE_RHYTHM:
        feats += C.FEATURE_GROUPS["rhythm"]

    model_paths = {n: C.out_path("models", f"{s['slug']}_v2.pkl") for n, s in MODELS.items()}
    targets = list(model_paths.values()) + [C.out_path("tables", files[k]) for k in ("table1", "table3", "table4")] \
        + [C.out_path("figures", files[k]) for k in ("fig1", "fig2", "fig3")]
    C.preflight(targets)

    df, ok = load_ok(C.out_path("features", files["ptbxl_features"]), feats)
    log.info("PTB-XL v2: %d records, %d usable; features (%d): %s", len(df), len(ok), len(feats), feats)
    tr9, tr8 = ok[ok["strat_fold"] <= 9], ok[ok["strat_fold"] <= 8]
    va9, te = ok[ok["strat_fold"] == 9], ok[ok["strat_fold"] == 10]
    log.info("train(1-9)=%d, val(9)=%d, test(10)=%d, test prevalence=%.4f", len(tr9), len(va9), len(te),
             te["label"].mean())
    yte = te["label"].to_numpy()

    rows, hp_rows, fitted, preds = [], [], {}, {}
    for name, spec in MODELS.items():
        t0 = time.time()
        params, trials = tune(spec, tr8[feats], tr8["label"], va9[feats], va9["label"], SEED, log, name)
        model = spec["make"](SEED, **params).fit(tr9[feats], tr9["label"])
        p = proba(model, te[feats])
        m = C.metrics(yte, p)
        ci = C.boot_ci(yte, p, n=N_BOOT, seed=SEED)
        rows.append({"model": name, **m, "auc_ci_lo": ci["auc"][0], "auc_ci_hi": ci["auc"][1],
                     "f1_ci_lo": ci["f1"][0], "f1_ci_hi": ci["f1"][1],
                     "size_kb_float_pickle": model_size_kb(model), "latency_ms_single": latency_ms(model, te[feats]),
                     "n_train": len(tr9), "n_test": len(te), "n_features": len(feats)})
        hp = model.get_params(deep=False) if not params else params
        hp_rows.append({"model": name, "tuned_on": "fold 9 (AUC)" if params else "fixed (manuscript Sec. 6.5 / default)",
                        "hyperparameters": str(params) if params else str({k: v for k, v in
                                                                         model.get_params(deep=False).items()
                                                                         if k != "estimator"}),
                        "n_trials": len(trials)})
        fitted[name], preds[name] = model, p
        C.write_pickle(model, model_paths[name], extra={"features": feats, "hyperparameters": params,
                                                        "train_folds": "1-9", "test_fold": 10})
        log.info("%-20s AUC %.4f F1 %.4f MCC %.4f (%.0f s)", name, m["auc"], m["f1"], m["mcc"], time.time() - t0)

    t1 = pd.DataFrame(rows)
    C.write_csv(t1, C.out_path("tables", files["table1"]), extra={"features": feats})
    C.write_csv(pd.DataFrame(hp_rows), C.out_path("tables", files["table4"]))

    # ---- ablation (Gradient Boosting) ---------------------------------------------------------
    G = C.FEATURE_GROUPS
    keep = lambda L: [f for f in L if f in feats]
    full = list(feats)
    cfgs = {"all_features": full, "statistics_only": keep(G["statistics"]), "qrs_only": keep(G["qrs"]),
            "st_t_only": keep(G["st_t"]), "rhythm_only": keep(G["rhythm"]),
            "minus_statistics": [f for f in full if f not in G["statistics"]],
            "minus_qrs": [f for f in full if f not in G["qrs"]],
            "minus_st_t": [f for f in full if f not in G["st_t"]],
            "minus_rhythm": [f for f in full if f not in G["rhythm"]],
            "minus_uncertain_t_features": [f for f in full if f not in ("T_asymmetry", "J_to_Tpeak_ms")],
            "minus_qrs_duration": [f for f in full if f != "QRS_duration_ms"]}
    ab = []
    for cname, cols in cfgs.items():
        if not cols:
            continue
        r = []
        for s in range(ABLATION_SEEDS):
            m = MODELS["Gradient Boosting"]["make"](SEED + s).fit(tr9[cols], tr9["label"])
            r.append(C.metrics(yte, proba(m, te[cols])))
        ab.append({"config": cname, "n_features": len(cols), "features": "|".join(cols),
                   "f1": np.mean([x["f1"] for x in r]), "auc": np.mean([x["auc"] for x in r]),
                   "mcc": np.mean([x["mcc"] for x in r]), "seeds": ABLATION_SEEDS})
        log.info("ablation %-24s F1 %.4f AUC %.4f", cname, ab[-1]["f1"], ab[-1]["auc"])
    t3 = pd.DataFrame(ab)
    ref = t3.loc[t3["config"] == "all_features", "f1"].iloc[0]
    t3["delta_f1_vs_all"] = t3["f1"] - ref
    if (t3["config"] == "statistics_only").any():
        s_f1 = t3.loc[t3["config"] == "statistics_only", "f1"].iloc[0]
        t3["f1_uplift_vs_statistics_only_pct"] = (t3["f1"] - s_f1) / s_f1 * 100
    C.write_csv(t3, C.out_path("tables", files["table3"]))

    log.info("External validation is a separate step: v2_07_external_eval.py (loads these pickles).")

    # ---- figures ------------------------------------------------------------------------------
    order = t1.sort_values("auc")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for a, key, lo, hi in ((ax[0], "auc", "auc_ci_lo", "auc_ci_hi"), (ax[1], "f1", "f1_ci_lo", "f1_ci_hi")):
        a.barh(order["model"], order[key], xerr=[order[key] - order[lo], order[hi] - order[key]], color="tab:blue")
        a.set_xlabel(key.upper() + " (fold 10, 95% bootstrap CI)")
        a.set_xlim(0.3, 1.0)
    fig.suptitle("Model comparison, PTB-XL fold 10 (v2 features)")
    fig.tight_layout()
    C.write_png(fig, C.out_path("figures", files["fig1"]))
    C.show_and_close(fig)

    fig, a = plt.subplots(figsize=(8, 5.5))
    a.scatter(t1["size_kb_float_pickle"], t1["f1"], s=60 + 400 * t1["latency_ms_single"] / t1["latency_ms_single"].max(),
              alpha=0.6)
    for _, r in t1.iterrows():
        a.annotate(r["model"], (r["size_kb_float_pickle"], r["f1"]), fontsize=8, xytext=(4, 4),
                   textcoords="offset points")
    a.set_xscale("log")
    a.set_xlabel("Serialised float model size (KB, log); marker size = single-record latency")
    a.set_ylabel("F1 (fold 10)")
    a.set_title("Accuracy vs footprint (INT8 sizes to be added after quantisation step)")
    fig.tight_layout()
    C.write_png(fig, C.out_path("figures", files["fig2"]))
    C.show_and_close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for a, (name, p) in zip(axes.ravel(), preds.items()):
        cm = confusion_matrix(yte, (p >= 0.5).astype(int), labels=[0, 1])
        cmn = cm / cm.sum(axis=1, keepdims=True)
        a.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.2f})", ha="center", va="center", fontsize=8)
        a.set_title(name, fontsize=9)
        a.set_xticks([0, 1], ["Normal", "Abnormal"], fontsize=7)
        a.set_yticks([0, 1], ["Normal", "Abnormal"], fontsize=7)
    for a in axes.ravel()[len(preds):]:
        a.axis("off")
    fig.suptitle("Confusion matrices at tau = 0.5 (row-normalised)")
    fig.tight_layout()
    C.write_png(fig, C.out_path("figures", files["fig3"]))
    C.show_and_close(fig)
    log.info("Benchmark finished. SHAP figures (Fig4-6) and INT8 sizes are separate steps.")


if __name__ == "__main__":
    main()
