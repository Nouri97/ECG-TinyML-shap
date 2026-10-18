"""
v2_07_external_eval.py - zero-shot evaluation of the 11 saved v2 models on external sets (no retraining).

External sets: cpsc2018_v2_features.csv (and kaggle_v2_features.csv if it exists and has all model features).
Outputs
  tables/Table2_external_validation_v2.csv : per model and dataset, pooled metrics at tau = 0.5, prevalence,
      always-abnormal F1, F1 / MCC / AUC after resampling to the PTB-XL test prevalence, tau*, Platt (10% cal)
      and the deltas versus PTB-XL fold 10 (AUC, MCC, balanced accuracy).
  tables/Table2b_per_diagnosis_v2.csv       : recall at tau = 0.5 and AUC (diagnosis vs Normal) per diagnosis,
      for PTB-XL fold 10 (diagnostic superclasses) and CPSC-2018 (its 8 abnormal classes). Multi-label records
      count in every class they carry.
  figures/Fig8_internal_vs_external_v2.png : AUC, MCC and balanced accuracy, internal vs external.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
import v2_common as C  # noqa: E402
from v2_common import CONFIG  # noqa: E402

SEED = 42
N_BOOT = 500
SLUGS = {"gradient_boosting": "Gradient Boosting", "lightgbm": "LightGBM", "xgboost": "XGBoost",
         "mlp": "MLP (2-layer)", "random_forest": "Random Forest", "extra_trees": "Extra Trees", "knn": "KNN",
         "decision_tree": "Decision Tree", "logistic_regression": "Logistic Regression",
         "linear_svm": "Linear SVM", "naive_bayes": "Naive Bayes"}
EXTERNAL = (("CPSC2018", "cpsc_features"), ("Kaggle", "kaggle_features"))


def predict(model, df):
    return model.predict_proba(df[list(model.feature_names_in_)])[:, 1]


def load_ok(path):
    d = pd.read_csv(path)
    return d[d["fail_reason"].fillna("") == ""].reset_index(drop=True)


def per_dx(df, p, tag, model, class_col, normal_only, min_n=20):
    """Recall and AUC (diagnosis vs normal-only records) per diagnosis."""
    cls = df[class_col].fillna("").astype(str)
    neg = (cls == normal_only).to_numpy()
    rows = []
    for dx in sorted({c for v in cls for c in v.split("|") if c and c != normal_only}):
        pos = cls.str.split("|").apply(lambda L: dx in L).to_numpy()
        auc = np.nan
        if pos.sum() >= min_n and neg.sum() >= min_n:
            auc = roc_auc_score(np.r_[np.ones(pos.sum()), np.zeros(neg.sum())], np.r_[p[pos], p[neg]])
        rows.append({"dataset": tag, "model": model, "diagnosis": dx, "n": int(pos.sum()),
                     "recall_tau_0_5": float((p[pos] >= 0.5).mean()) if pos.sum() else np.nan,
                     "auc_vs_normal": auc})
    rows.append({"dataset": tag, "model": model, "diagnosis": f"(normal-only: {normal_only})", "n": int(neg.sum()),
                 "recall_tau_0_5": np.nan, "auc_vs_normal": np.nan,
                 "specificity_tau_0_5": float((p[neg] < 0.5).mean()) if neg.sum() else np.nan})
    return rows


def main():
    C.init_run(globals().get("__file__"), SEED)
    log = C.get_logger("external_eval")
    files = CONFIG["files"]
    t2_path = C.out_path("tables", files["table2"])
    t2b_path = C.out_path("tables", files["table2b"])
    fig_path = C.out_path("figures", files["fig8"])
    C.preflight([t2_path, t2b_path, fig_path])

    models = {name: joblib.load(C.out_path("models", f"{slug}_v2.pkl")) for slug, name in SLUGS.items()}
    internal = load_ok(C.out_path("features", files["ptbxl_features"]))
    te = internal[internal["strat_fold"] == 10].reset_index(drop=True)
    yi = te["label"].to_numpy()
    prev_int = float(yi.mean())
    ext = {}
    for dname, key in EXTERNAL:
        p = C.out_path("features", files[key])
        if os.path.exists(p):
            d = load_ok(p)
            miss = [f for m in models.values() for f in m.feature_names_in_ if f not in d.columns]
            if miss:
                log.warning("%s skipped: missing features %s", dname, sorted(set(miss)))
                continue
            ext[dname] = d
    if not ext:
        raise RuntimeError("No usable external v2 feature file. Nothing written.")
    log.info("Internal test n=%d (prevalence %.4f); external: %s", len(te), prev_int,
             {k: (len(v), round(float(v["label"].mean()), 4)) for k, v in ext.items()})

    rows, dx_rows, preds_int = [], [], {}
    for name, m in models.items():
        pi = predict(m, te)
        preds_int[name] = pi
        s_int = C.summarize(yi, pi)
        dx_rows += per_dx(te, pi, "PTB-XL_fold10", name, "scp_superclasses", "NORM")
        for dname, d in ext.items():
            ye, pe = d["label"].to_numpy().astype(int), predict(m, d)
            s = C.summarize(ye, pe, target_prev=prev_int, seed=SEED)
            ci = C.boot_ci(ye, pe, n=N_BOOT, seed=SEED)
            rows.append({"dataset": dname, "model": name, **s, "auc_ci_lo": ci["auc"][0], "auc_ci_hi": ci["auc"][1],
                         "internal_auc": s_int["auc"], "internal_mcc": s_int["mcc"], "internal_bal_acc": s_int["bal_acc"],
                         "delta_auc_vs_internal": s["auc"] - s_int["auc"], "delta_mcc_vs_internal": s["mcc"] - s_int["mcc"],
                         "delta_bal_acc_vs_internal": s["bal_acc"] - s_int["bal_acc"]})
            if "dx_classes" in d.columns:
                dx_rows += per_dx(d, pe, dname, name, "dx_classes", "Normal")
    t2 = pd.DataFrame(rows)
    C.write_csv(t2, t2_path, extra={"internal_test_prevalence": prev_int, "external_sets": list(ext)})
    C.write_csv(pd.DataFrame(dx_rows), t2b_path)
    cols = ["dataset", "model", "prevalence", "auc", "delta_auc_vs_internal", "mcc", "bal_acc", "recall", "specificity",
            "f1", "always_abnormal_f1", "f1_prev_matched"]
    pd.set_option("display.width", 250)
    log.info("\n%s", t2[cols].round(3).to_string(index=False))

    import matplotlib.pyplot as plt
    for dname in ext:
        sub = t2[t2["dataset"] == dname].sort_values("internal_auc", ascending=False)
        fig, ax = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        for a, (ext_col, int_col, ttl) in zip(ax, (("auc", "internal_auc", "AUC"), ("mcc", "internal_mcc", "MCC"),
                                                    ("bal_acc", "internal_bal_acc", "Balanced accuracy"))):
            y = np.arange(len(sub))
            a.barh(y - 0.2, sub[int_col], 0.4, label="PTB-XL fold 10", color="tab:blue")
            a.barh(y + 0.2, sub[ext_col], 0.4, label=dname, color="tab:red")
            a.set_yticks(y, sub["model"])
            a.invert_yaxis()
            a.set_title(ttl)
        ax[0].legend(loc="lower left", fontsize=8)
        fig.suptitle(f"Zero-shot generalisation: PTB-XL test fold vs {dname} (prevalence {sub['prevalence'].iloc[0]:.2f})")
        fig.tight_layout()
        C.write_png(fig, fig_path if dname == "CPSC2018" else fig_path.replace("Fig8_", f"Fig8_{dname}_"))
        C.show_and_close(fig)


if __name__ == "__main__":
    main()
