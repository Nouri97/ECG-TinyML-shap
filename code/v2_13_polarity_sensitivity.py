"""
v2_13_polarity_sensitivity.py - does the lead-polarity flag carry information the 15-feature model does not see?

The pipeline inverts a record when its dominant QRS deflection is negative (3.96% of normal, 25.25% of abnormal PTB-XL
records), so ST/T/QRS measures are relative to the dominant QRS direction and the flag itself is not a feature.
This is a SENSITIVITY ANALYSIS, not a change of the final model:
  F15 = 13 morphology + 2 rhythm features (final set)      F16 = F15 + polarity_inverted (0/1)
1) 10-fold cross-validation for LightGBM, XGBoost, MLP (hyperparameters from Table 4): per-fold AUC/F1/MCC, paired
   Wilcoxon on fold scores (averaged over the three models, as in the feature-set rule of v2_08, and per model).
2) Zero-shot evaluation on CPSC-2018 for models trained on PTB-XL folds 1-9 (AUC, MCC, prevalence-matched metrics).
3) Out-of-fold performance of the F15 models inside inverted and non-inverted records (is the model equally good
   there?), and the share of inverted records by class in PTB-XL and CPSC-2018 (does the polarity-abnormality link
   replicate in an independent dataset?).
Decision rule written in advance: F15 stays final unless F16 is significantly better in cross-validation AND not worse
on CPSC-2018 AND the authors judge the flag clinically defensible (a net-negative QRS in lead II can reflect axis
deviation, but also electrode misplacement or lead reversal). Outputs: Table10a (fold scores), Table10b (summary).
"""
import ast
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
import v2_common as C  # noqa: E402
from v2_common import CONFIG  # noqa: E402
import v2_02_benchmark as B  # noqa: E402

SEED = 42
ALPHA = 0.05
MODELS3 = ["LightGBM", "XGBoost", "MLP (2-layer)"]


def tuned_params(files):
    t4 = pd.read_csv(C.out_path("tables", files["table4"]))
    return {r["model"]: ast.literal_eval(r["hyperparameters"]) for _, r in t4.iterrows()
            if str(r["tuned_on"]).startswith("fold 9")}


def safe_wilcoxon(d):
    try:
        return float(wilcoxon(d).pvalue)
    except ValueError:
        return np.nan


def main():
    C.init_run(globals().get("__file__"), SEED)
    log = C.get_logger("polarity_sensitivity")
    files = CONFIG["files"]
    outs = [C.out_path("tables", files["table10a"]), C.out_path("tables", files["table10b"])]
    C.preflight(outs)

    d = pd.read_csv(C.out_path("features", files["ptbxl_features"]))
    ok = d[d["fail_reason"].fillna("") == ""].reset_index(drop=True)
    ok["polarity_inverted"] = ok["polarity_inverted"].astype(bool).astype(int)
    cp = pd.read_csv(C.out_path("features", files["cpsc_features"]))
    cp = cp[cp["fail_reason"].fillna("") == ""].reset_index(drop=True)
    cp["polarity_inverted"] = cp["polarity_inverted"].astype(bool).astype(int)
    tuned = tuned_params(files)
    F15 = C.morphology_features() + C.FEATURE_GROUPS["rhythm"]
    V = {"F15": F15, "F16": F15 + ["polarity_inverted"]}
    y = ok["label"].to_numpy()
    summary = []

    # polarity share by class
    for ds, df in (("PTB-XL", ok), ("CPSC2018", cp)):
        for lab, nm in ((0, "Normal"), (1, "Abnormal")):
            m = df["label"] == lab
            summary.append({"analysis": "polarity_share_by_class", "group": f"{ds} {nm}", "n": int(m.sum()),
                            "share_inverted": float(df.loc[m, "polarity_inverted"].mean())})
    log.info("Polarity share: %s", [(r["group"], round(r["share_inverted"], 4)) for r in summary])

    # 1) cross-validation
    fold_rows, oof = [], {}
    for name in MODELS3:
        for v, feats in V.items():
            p_oof = np.zeros(len(ok))
            for k in range(1, 11):
                te_m = (ok["strat_fold"] == k).to_numpy()
                tr, te = ok[~te_m], ok[te_m]
                m = B.MODELS[name]["make"](SEED, **tuned.get(name, {})).fit(tr[feats], tr["label"])
                p = B.proba(m, te[feats])
                p_oof[te_m] = p
                met = C.metrics(te["label"].to_numpy(), p)
                fold_rows.append({"model": name, "variant": v, "fold": k, "auc": met["auc"], "f1": met["f1"], "mcc": met["mcc"],
                                  "bal_acc": met["bal_acc"]})
            oof[(name, v)] = p_oof
            log.info("%-14s %s CV mean AUC %.4f", name, v, np.mean([r["auc"] for r in fold_rows if r["model"] == name and r["variant"] == v]))
    fs = pd.DataFrame(fold_rows)
    g = fs.groupby(["variant", "fold"])[["auc", "f1", "mcc"]].mean()
    a, b = g.loc["F15"], g.loc["F16"]
    summary.append({"analysis": "cv_paired_F16_minus_F15", "group": "mean of 3 models", "n": 10,
                    "mean_auc_diff": float((b["auc"] - a["auc"]).mean()), "mean_f1_diff": float((b["f1"] - a["f1"]).mean()),
                    "mean_mcc_diff": float((b["mcc"] - a["mcc"]).mean()), "folds_better_auc": int((b["auc"] > a["auc"]).sum()),
                    "wilcoxon_p_auc": safe_wilcoxon(b["auc"] - a["auc"])})
    for name in MODELS3:
        x = fs[(fs["model"] == name)].pivot(index="fold", columns="variant", values="auc")
        summary.append({"analysis": "cv_paired_F16_minus_F15", "group": name, "n": 10, "mean_auc_diff": float((x["F16"] - x["F15"]).mean()),
                        "folds_better_auc": int((x["F16"] > x["F15"]).sum()), "wilcoxon_p_auc": safe_wilcoxon(x["F16"] - x["F15"])})

    # 3) performance inside inverted / non-inverted records (F15, out-of-fold)
    inv = ok["polarity_inverted"].to_numpy() == 1
    for name in ("LightGBM", "MLP (2-layer)"):
        p = oof[(name, "F15")]
        for sn, mask in (("all", np.ones(len(ok), bool)), ("inverted", inv), ("not_inverted", ~inv)):
            yy, pp = y[mask], p[mask]
            summary.append({"analysis": "F15_oof_by_polarity", "model": name, "group": sn, "n": int(mask.sum()), "prevalence": float(yy.mean()),
                            "auc": roc_auc_score(yy, pp) if len(np.unique(yy)) == 2 and mask.sum() >= 10 else np.nan,
                            "sensitivity": float((pp[yy == 1] >= 0.5).mean()) if (yy == 1).any() else np.nan,
                            "specificity": float((pp[yy == 0] < 0.5).mean()) if (yy == 0).any() else np.nan})

    # 2) external CPSC-2018, models trained on folds 1-9
    tr9 = ok[ok["strat_fold"] <= 9]
    prev_te = float(ok.loc[ok["strat_fold"] == 10, "label"].mean())
    ye = cp["label"].to_numpy().astype(int)
    for name in MODELS3:
        for v, feats in V.items():
            m = B.MODELS[name]["make"](SEED, **tuned.get(name, {})).fit(tr9[feats], tr9["label"])
            s = C.summarize(ye, B.proba(m, cp[feats]), target_prev=prev_te, seed=SEED)
            summary.append({"analysis": "external_CPSC2018", "model": name, "group": v, "n": len(ye), "auc": s["auc"], "mcc": s["mcc"],
                            "bal_acc": s["bal_acc"], "f1_prev_matched": s.get("f1_prev_matched"), "mcc_prev_matched": s.get("mcc_prev_matched"),
                            "auc_prev_matched": s.get("auc_prev_matched")})
    t10b = pd.DataFrame(summary)
    pd.set_option("display.width", 250)
    log.info("\n%s", t10b.round(4).to_string(index=False))
    ext = t10b[t10b["analysis"] == "external_CPSC2018"].pivot(index="model", columns="group", values="auc")
    cvr = t10b[(t10b["analysis"] == "cv_paired_F16_minus_F15") & (t10b["group"] == "mean of 3 models")].iloc[0]
    sig = bool(cvr["mean_auc_diff"] > 0 and cvr["wilcoxon_p_auc"] < ALPHA)
    ext_ok = bool((ext["F16"] - ext["F15"]).mean() >= 0)
    log.info("PRE-SPECIFIED RULE: CV gain significant = %s; CPSC AUC not worse = %s -> %s", sig, ext_ok,
             "F16 is a candidate only if clinically defensible; otherwise keep F15" if sig and ext_ok else "keep F15")
    C.write_csv(fs, outs[0])
    C.write_csv(t10b, outs[1], extra={"cv_gain_significant": sig, "external_not_worse": ext_ok})


if __name__ == "__main__":
    main()
