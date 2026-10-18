"""
v2_09_quantize.py - post-training INT8 quantisation, before/after accuracy, footprint, importance stability.

Models: MLP (2-layer), LightGBM, XGBoost, Gradient Boosting (the statistically tied top group) from models/*_v2.pkl.
No retraining. Quantisation ranges come from PTB-XL folds 1-9; evaluation on PTB-XL fold 10 and CPSC-2018.

MLP  : the StandardScaler stays in float (2 x n_features values); standardised inputs -> int8 (per-tensor scale,
       99.9th-percentile clipping); weights -> int8 per output channel; biases -> int32; hidden activations ->
       uint8 (post-ReLU, 99.9th-percentile scale); integer matmul with int32 accumulators; the output logit is
       dequantised and passed through a float sigmoid (a 256-entry lookup table would replace it on a device).
Trees: tree structure is extracted from each library and VERIFIED (float traversal + base must reproduce the
       library's raw margin, otherwise the model is skipped). Inputs -> uint8 on a per-feature grid between the
       0.1th and 99.9th percentiles; thresholds -> the same grid (uint8); leaf values -> int8 with one global
       scale; accumulation in int32; base score and leaf scale in float.
Footprint (defined layouts, not binaries - measure on the target board before claiming device numbers):
       float32 layout : internal node 9 B (feature u8, threshold f32, left u16, right u16), leaf 4 B (f32)
       int8 layout    : internal node 6 B (feature u8, threshold u8, left u16, right u16), leaf 1 B (i8)
       + constants (feature grid, scales, base; MLP: weights, biases, scales, scaler).
Operations per inference: MLP multiply-accumulates; trees mean number of node comparisons.
Importance stability (answers R1-5 without recomputing SHAP on a quantised model): permutation importance
(AUC drop, 5 repeats) of the float and the INT8 predictor on fold 10; Spearman rho and top-3 overlap.
SHAP itself was computed on the float models only; this is stated, not hidden.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
import v2_common as C  # noqa: E402
from v2_common import CONFIG  # noqa: E402

SEED = 42
N_REPEATS = 5
LO_Q, HI_Q = 0.1, 99.9
MODELS = {"MLP (2-layer)": "mlp", "LightGBM": "lightgbm", "XGBoost": "xgboost", "Gradient Boosting": "gradient_boosting"}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------- tree extraction
def _arr(feat, thr, left, right, val):
    return {"feat": np.asarray(feat, int), "thr": np.asarray(thr, float), "left": np.asarray(left, int),
            "right": np.asarray(right, int), "val": np.asarray(val, float)}


def trees_gb(m):
    out = []
    for st in m.estimators_[:, 0]:
        t = st.tree_
        feat = np.where(t.children_left < 0, -1, t.feature)
        out.append(_arr(feat, t.threshold, t.children_left, t.children_right, t.value[:, 0, 0] * m.learning_rate))
    return out, False, (lambda X: m.decision_function(X))


def trees_lgbm(m, feats):
    df = m.booster_.trees_to_dataframe()
    idx = {f: i for i, f in enumerate(feats)}
    out = []
    for _, g in df.groupby("tree_index", sort=True):
        ids = {nid: i for i, nid in enumerate(g["node_index"])}
        feat, thr, left, right, val = [], [], [], [], []
        for _, r in g.iterrows():
            leaf = pd.isna(r["left_child"])
            feat.append(-1 if leaf else idx[r["split_feature"]] if r["split_feature"] in idx else int(str(r["split_feature"]).split("_")[-1]))
            thr.append(0.0 if leaf else float(r["threshold"]))
            left.append(-1 if leaf else ids[r["left_child"]])
            right.append(-1 if leaf else ids[r["right_child"]])
            val.append(float(r["value"]) if leaf else 0.0)
        out.append(_arr(feat, thr, left, right, val))
    return out, False, (lambda X: m.predict(X, raw_score=True))


def trees_xgb(m, feats):
    """Full-precision split values from the JSON dump (the DataFrame dump rounds to 6 digits)."""
    import json
    idx = {f: i for i, f in enumerate(feats)}
    out = []
    for js in m.get_booster().get_dump(dump_format="json"):
        nodes = {}

        def walk(n):
            nodes[n["nodeid"]] = n
            for c in n.get("children", []):
                walk(c)

        walk(json.loads(js))
        size = max(nodes) + 1
        feat, thr, left, right, val = [-1] * size, [0.0] * size, [-1] * size, [-1] * size, [0.0] * size
        for nid, n in nodes.items():
            if "leaf" in n:
                val[nid] = float(n["leaf"])
            else:
                fn = n["split"]
                feat[nid] = idx[fn] if fn in idx else int(str(fn).lstrip("f"))
                thr[nid], left[nid], right[nid] = float(np.float32(n["split_condition"])), n["yes"], n["no"]   # float32 exactly as XGBoost compares
        out.append(_arr(feat, thr, left, right, val))
    return out, True, (lambda X: m.predict(X, output_margin=True))


def traverse(tree, X, strict):
    """Vectorised traversal. Returns (leaf values, number of comparisons per sample)."""
    n = len(X)
    node = np.zeros(n, int)
    steps = np.zeros(n, int)
    rows = np.arange(n)
    while True:
        f = tree["feat"][node]
        leaf = f < 0
        if leaf.all():
            break
        xv = X[rows, np.where(leaf, 0, f)]
        thr = tree["thr"][node]
        go_left = (xv < thr) if strict else (xv <= thr)
        nxt = np.where(go_left, tree["left"][node], tree["right"][node])
        steps += (~leaf)
        node = np.where(leaf, node, nxt)
    return tree["val"][node], steps


def ensemble_margin(trees, X, strict):
    s, steps = np.zeros(len(X)), np.zeros(len(X))
    for t in trees:
        v, st = traverse(t, X, strict)
        s += v
        steps += st
    return s, steps


class TreeModel:
    def __init__(self, name, model, feats, Xtr):
        self.name, self.model, self.feats = name, model, feats
        if name == "Gradient Boosting":
            self.trees, self.strict, self.margin = trees_gb(model)
        elif name == "LightGBM":
            self.trees, self.strict, self.margin = trees_lgbm(model, feats)
        else:
            self.trees, self.strict, self.margin = trees_xgb(model, feats)
        self.cast32 = name == "XGBoost"      # XGBoost casts inputs to float32 before comparing
        Xdf = pd.DataFrame(Xtr[:200], columns=feats)
        s, _ = ensemble_margin(self.trees, self._c(Xtr[:200]), self.strict)
        ref = np.asarray(self.margin(Xdf), float)
        self.base = float(np.median(ref - s))
        err = np.abs(ref - (s + self.base)).max()
        if err > 1e-3:
            raise ValueError(f"float traversal does not reproduce the library margin (max err {err:.2e})")
        # quantisation parameters
        self.lo = np.percentile(Xtr, LO_Q, axis=0)
        self.hi = np.percentile(Xtr, HI_Q, axis=0)
        self.step = np.maximum((self.hi - self.lo) / 255.0, 1e-12)
        vmax = max(np.abs(t["val"]).max() for t in self.trees)
        self.leaf_scale = vmax / 127.0 if vmax > 0 else 1.0
        self.qtrees = []
        for t in self.trees:
            leaf = t["feat"] < 0
            thr_q = np.clip(np.round((t["thr"] - self.lo[np.maximum(t["feat"], 0)]) / self.step[np.maximum(t["feat"], 0)]), -1, 255)
            self.qtrees.append({**t, "thr": np.where(leaf, 0, thr_q),
                                "val": np.where(leaf, np.clip(np.round(t["val"] / self.leaf_scale), -127, 127), 0)})
        n_leaf = sum(int((t["feat"] < 0).sum()) for t in self.trees)
        n_int = sum(int((t["feat"] >= 0).sum()) for t in self.trees)
        self.n_int, self.n_leaf = n_int, n_leaf
        self.max_nodes_per_tree = max(len(t["feat"]) for t in self.trees)

    def _c(self, X):
        return X.astype(np.float32).astype(float) if self.cast32 else X

    def predict_float(self, X):
        return sigmoid(self.margin(pd.DataFrame(X, columns=self.feats)))

    def predict_int8(self, X):
        Xq = np.clip(np.round((X - self.lo) / self.step), 0, 255)
        s, steps = ensemble_margin(self.qtrees, Xq, self.strict)
        self.mean_comparisons = float(steps.mean())
        return sigmoid(self.base + self.leaf_scale * s)

    def footprint(self):
        f = len(self.feats)
        return {"float_bytes": 9 * self.n_int + 4 * self.n_leaf + 8, "int8_bytes": 6 * self.n_int + 1 * self.n_leaf + 8 * f + 8,
                "n_params": self.n_int + self.n_leaf, "ram_bytes_est": f + 4,
                "ops": self.mean_comparisons if hasattr(self, "mean_comparisons") else np.nan,
                "ops_unit": "node comparisons", "index_fits_u16": bool(self.max_nodes_per_tree < 65536)}


# --------------------------------------------------------------------------- MLP
class MLPModel:
    def __init__(self, pipe, feats, Xtr):
        self.pipe, self.feats = pipe, feats
        self.scaler, self.mlp = pipe.steps[0][1], pipe.steps[-1][1]
        self.W, self.b = [w.copy() for w in self.mlp.coefs_], [b.copy() for b in self.mlp.intercepts_]
        z = self._z(Xtr)
        ref = self.pipe.predict_proba(pd.DataFrame(Xtr[:200], columns=feats))[:, 1]
        if np.abs(ref - sigmoid(self._float_logit(Xtr[:200]))).max() > 1e-6:
            raise ValueError("float MLP re-implementation does not reproduce predict_proba")
        self.s_in = np.percentile(np.abs(z), HI_Q) / 127.0
        h, self.s_h = z, []
        for l in range(len(self.W) - 1):
            h = np.maximum(h @ self.W[l] + self.b[l], 0)
            self.s_h.append(max(np.percentile(h, HI_Q), 1e-9) / 255.0)
        self.sw = [np.maximum(np.abs(w).max(axis=0), 1e-12) / 127.0 for w in self.W]
        self.Wq = [np.clip(np.round(w / s), -127, 127).astype(np.int32) for w, s in zip(self.W, self.sw)]
        s_x = [self.s_in] + self.s_h
        self.bq = [np.round(b / (sx * s)).astype(np.int64) for b, sx, s in zip(self.b, s_x, self.sw)]
        self.s_x = s_x

    def _z(self, X):
        return (X - self.scaler.mean_) / self.scaler.scale_

    def _float_logit(self, X):
        h = self._z(X)
        for l in range(len(self.W) - 1):
            h = np.maximum(h @ self.W[l] + self.b[l], 0)
        return (h @ self.W[-1] + self.b[-1]).ravel()

    def predict_float(self, X):
        return self.pipe.predict_proba(pd.DataFrame(X, columns=self.feats))[:, 1]

    def predict_int8(self, X):
        xq = np.clip(np.round(self._z(X) / self.s_in), -127, 127).astype(np.int64)
        for l in range(len(self.Wq)):
            acc = xq @ self.Wq[l] + self.bq[l]                       # int accumulators
            real = acc * (self.s_x[l] * self.sw[l])
            if l < len(self.Wq) - 1:
                xq = np.clip(np.round(np.maximum(real, 0) / self.s_h[l]), 0, 255).astype(np.int64)
            else:
                return sigmoid(real.ravel())

    def footprint(self):
        nw = sum(w.size for w in self.W)
        nb = sum(b.size for b in self.b)
        nsc = sum(w.shape[1] for w in self.W)
        f = len(self.feats)
        return {"float_bytes": 4 * (nw + nb) + 8 * f, "int8_bytes": nw + 4 * nb + 4 * nsc + 4 * len(self.s_h) + 4 + 8 * f,
                "n_params": nw + nb, "ram_bytes_est": f + sum(w.shape[1] for w in self.W[:-1]) + 4 * max(w.shape[1] for w in self.W),
                "ops": float(nw), "ops_unit": "multiply-accumulates", "index_fits_u16": True}


def metric_row(y, pf, pq):
    mf, mq = C.metrics(y, pf), C.metrics(y, pq)
    row = {"n": len(y), "prevalence": float(np.mean(y))}
    for k in ("auc", "f1", "mcc", "bal_acc", "brier"):
        row[f"{k}_float"], row[f"{k}_int8"], row[f"d_{k}"] = mf[k], mq[k], mq[k] - mf[k]
    row.update(mean_abs_dprob=float(np.abs(pf - pq).mean()), max_abs_dprob=float(np.abs(pf - pq).max()),
               pct_label_flips=float(((pf >= 0.5) != (pq >= 0.5)).mean() * 100))
    return row


def perm_importance(predict, X, y, seed):
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, predict(X))
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        d = []
        for _ in range(N_REPEATS):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            d.append(base - roc_auc_score(y, predict(Xp)))
        imp[j] = np.mean(d)
    return imp


def main():
    C.init_run(globals().get("__file__"), SEED)
    log = C.get_logger("quantize")
    files = CONFIG["files"]
    outs = [C.out_path("tables", files[k]) for k in ("table7", "table7a", "table7b")] + [C.out_path("figures", files["fig9"])]
    C.preflight(outs)

    internal = pd.read_csv(C.out_path("features", files["ptbxl_features"]))
    ok = internal[internal["fail_reason"].fillna("") == ""]
    tr, te = ok[ok["strat_fold"] <= 9], ok[ok["strat_fold"] == 10]
    ext = {}
    p = C.out_path("features", files["cpsc_features"])
    if os.path.exists(p):
        d = pd.read_csv(p)
        ext["CPSC2018"] = d[d["fail_reason"].fillna("") == ""]

    acc_rows, fp_rows, imp_rows, imp_tab = [], [], [], {}
    for name, slug in MODELS.items():
        model = joblib.load(C.out_path("models", f"{slug}_v2.pkl"))
        feats = list(model.feature_names_in_)
        Xtr, Xte, yte = tr[feats].to_numpy(float), te[feats].to_numpy(float), te["label"].to_numpy()
        try:
            q = MLPModel(model, feats, Xtr) if name == "MLP (2-layer)" else TreeModel(name, model, feats, Xtr)
        except ValueError as e:
            log.warning("%s skipped: %s", name, e)
            continue
        sets = {"PTB-XL_fold10": (Xte, yte)}
        for dn, d in ext.items():
            sets[dn] = (d[feats].to_numpy(float), d["label"].to_numpy().astype(int))
        for dn, (X, y) in sets.items():
            pf, pq = q.predict_float(X), q.predict_int8(X)
            acc_rows.append({"model": name, "dataset": dn, **metric_row(y, pf, pq)})
            log.info("%-18s %-14s AUC %.4f -> %.4f | F1 %.4f -> %.4f | label flips %.2f%%", name, dn,
                     acc_rows[-1]["auc_float"], acc_rows[-1]["auc_int8"], acc_rows[-1]["f1_float"], acc_rows[-1]["f1_int8"],
                     acc_rows[-1]["pct_label_flips"])
        fp = q.footprint()
        fp_rows.append({"model": name, "float_layout_kb": fp["float_bytes"] / 1024, "int8_layout_kb": fp["int8_bytes"] / 1024,
                        "compression_x": fp["float_bytes"] / fp["int8_bytes"], "n_params_or_nodes": fp["n_params"],
                        "ops_per_inference": fp["ops"], "ops_unit": fp["ops_unit"], "ram_bytes_est": fp["ram_bytes_est"],
                        "node_index_fits_u16": fp["index_fits_u16"]})
        i_f = perm_importance(q.predict_float, Xte, yte, SEED)
        i_q = perm_importance(q.predict_int8, Xte, yte, SEED)
        rho = spearmanr(i_f, i_q).statistic
        top = lambda v: set(np.argsort(-v)[:3])
        imp_rows.append({"model": name, "spearman_rho_importance": rho, "top3_overlap": len(top(i_f) & top(i_q)) / 3,
                         "top3_float": "|".join(feats[i] for i in np.argsort(-i_f)[:3]),
                         "top3_int8": "|".join(feats[i] for i in np.argsort(-i_q)[:3]),
                         **{f"imp_float_{f}": v for f, v in zip(feats, i_f)}, **{f"imp_int8_{f}": v for f, v in zip(feats, i_q)}})
        log.info("%-18s importance stability: rho %.3f, top-3 overlap %.2f", name, rho, imp_rows[-1]["top3_overlap"])

    if not acc_rows:
        raise RuntimeError("No model could be quantised; nothing written.")
    t7, t7a, t7b = pd.DataFrame(acc_rows), pd.DataFrame(fp_rows), pd.DataFrame(imp_rows)
    pd.set_option("display.width", 250)
    log.info("\n%s", t7a.round(2).to_string(index=False))
    C.write_csv(t7, outs[0], extra={"layouts": "float: node 9 B / leaf 4 B; int8: node 6 B / leaf 1 B"})
    C.write_csv(t7a, outs[1])
    C.write_csv(t7b, outs[2])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.5))
    base = t7[t7["dataset"] == "PTB-XL_fold10"].set_index("model")
    for _, r in t7a.iterrows():
        a = base.loc[r["model"]]
        ax.annotate("", xy=(r["int8_layout_kb"], a["auc_int8"]), xytext=(r["float_layout_kb"], a["auc_float"]),
                    arrowprops=dict(arrowstyle="->", color="gray"))
        ax.scatter(r["float_layout_kb"], a["auc_float"], c="tab:blue", s=50)
        ax.scatter(r["int8_layout_kb"], a["auc_int8"], c="tab:red", s=50)
        ax.annotate(r["model"], (r["int8_layout_kb"], a["auc_int8"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Model size in the defined layouts (KB, log): blue = float32, red = INT8")
    ax.set_ylabel("AUC on PTB-XL fold 10")
    ax.set_title("Footprint vs accuracy before/after INT8 (layouts defined in v2_09)")
    fig.tight_layout()
    C.write_png(fig, outs[3])
    C.show_and_close(fig)


if __name__ == "__main__":
    main()
