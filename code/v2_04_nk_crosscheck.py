"""
v2_04_nk_crosscheck.py - independent check of the v2 landmarks against NeuroKit2 (DWT delineation).

For N_RECORDS random usable PTB-XL records (lead II, 500 Hz) NeuroKit2 detects R peaks and delineates
QRS onset/offset and T peak per beat (method="dwt"); the per-record median over beats is compared with
the v2 values already stored in ptbxl_v2_features.csv:
    QRS duration  : QRS_duration_ms           vs  median(R_Offsets - R_Onsets)
    J point       : qc_J_ms                   vs  median(R_Offsets - R)
    T peak time   : qc_Tpeak_ms               vs  median(T_Peaks - R)
Reported: n, bias (v2 - NK), SD, 95% limits of agreement, Pearson r, median |diff| and % within
tolerance, for all records and by polarity flag and class. NeuroKit2 is a reference method, not ground
truth: disagreement means "check", agreement means the simple rules behave like a published delineator.
Requires:  pip install neurokit2
"""
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
import v2_common as C  # noqa: E402
from v2_common import CONFIG  # noqa: E402

SEED = 42
FS = 500
N_RECORDS = 500
N_JOBS = -1
TOLERANCE_MS = {"qrs_duration": 20.0, "j_point": 20.0, "t_peak": 40.0}


def nk_landmarks(rel_path, ptbxl_dir):
    """NeuroKit2 per-record medians (ms). Returns dict of NaN on any failure."""
    import neurokit2 as nk
    import wfdb
    out = {"nk_qrs_duration": np.nan, "nk_j_point": np.nan, "nk_t_peak": np.nan, "nk_n_beats": 0}
    try:
        rec = wfdb.rdrecord(os.path.join(ptbxl_dir, rel_path))
        sig = rec.p_signal[:, rec.sig_name.index("II")][: 10 * FS]
        clean = nk.ecg_clean(sig, sampling_rate=FS, method="neurokit")
        _, info = nk.ecg_peaks(clean, sampling_rate=FS, method="neurokit")
        r = np.asarray(info["ECG_R_Peaks"], dtype=float)
        if len(r) < 3:
            return out
        _, w = nk.ecg_delineate(clean, info["ECG_R_Peaks"], sampling_rate=FS, method="dwt")

        def arr(key):
            a = np.asarray(w.get(key, []), dtype=float)
            return a[: len(r)] if len(a) >= len(r) else np.r_[a, np.full(len(r) - len(a), np.nan)]

        on, off, tp = arr("ECG_R_Onsets"), arr("ECG_R_Offsets"), arr("ECG_T_Peaks")

        def med(x):
            x = x[np.isfinite(x)]
            return float(np.median(x)) * 1000.0 / FS if len(x) >= 3 else np.nan

        out.update(nk_qrs_duration=med(off - on), nk_j_point=med(off - r), nk_t_peak=med(tp - r),
                   nk_n_beats=int(len(r)))
    except Exception:
        pass
    return out


def agreement(a, b, tol):
    d = (a - b)
    m = np.isfinite(d)
    a, b, d = a[m], b[m], d[m]
    if len(d) < 5:
        return {"n": int(len(d))}
    sd = d.std(ddof=1)
    return {"n": int(len(d)), "bias_ms": d.mean(), "sd_ms": sd, "loa_low_ms": d.mean() - 1.96 * sd,
            "loa_high_ms": d.mean() + 1.96 * sd, "pearson_r": float(np.corrcoef(a, b)[0, 1]),
            "median_abs_diff_ms": float(np.median(np.abs(d))), "tol_ms": tol, "pct_within_tol": float(np.mean(np.abs(d) <= tol) * 100)}


def main():
    C.init_run(globals().get("__file__"), SEED)
    log = C.get_logger("nk_crosscheck")
    files = CONFIG["files"]
    rec_path = C.out_path("tables", files["nk_records"])
    sum_path = C.out_path("tables", files["nk_summary"])
    fig_path = C.out_path("figures", files["fig_nk"])
    C.preflight([rec_path, sum_path, fig_path])

    df = pd.read_csv(C.out_path("features", files["ptbxl_features"]))
    df = df[df["fail_reason"].fillna("") == ""]
    ptbxl = pd.read_csv(os.path.join(CONFIG["ptbxl_dir"], "ptbxl_database.csv"), index_col="ecg_id")
    pick = df.sample(n=min(N_RECORDS, len(df)), random_state=SEED).copy()
    pick["filename_hr"] = pick["ecg_id"].map(ptbxl["filename_hr"])
    log.info("NeuroKit2 delineation on %d records (this takes a few minutes)", len(pick))
    res = Parallel(n_jobs=N_JOBS)(delayed(nk_landmarks)(p, CONFIG["ptbxl_dir"]) for p in pick["filename_hr"])
    rec = pd.concat([pick[["ecg_id", "label", "polarity_inverted", "QRS_duration_ms", "qc_J_ms", "qc_Tpeak_ms"]]
                     .reset_index(drop=True), pd.DataFrame(res)], axis=1)
    rec["polarity_inverted"] = rec["polarity_inverted"].astype(bool)
    log.info("NeuroKit2 produced all three landmarks for %d / %d records",
             int(rec[["nk_qrs_duration", "nk_j_point", "nk_t_peak"]].notna().all(axis=1).sum()), len(rec))

    pairs = {"qrs_duration": ("QRS_duration_ms", "nk_qrs_duration"), "j_point": ("qc_J_ms", "nk_j_point"),
             "t_peak": ("qc_Tpeak_ms", "nk_t_peak")}
    subsets = {"all": rec.index == rec.index, "polarity_normal": ~rec["polarity_inverted"],
               "polarity_inverted": rec["polarity_inverted"], "label_normal": rec["label"] == 0,
               "label_abnormal": rec["label"] == 1}
    rows = []
    for sname, mask in subsets.items():
        for lname, (mine, theirs) in pairs.items():
            sub = rec[mask]
            rows.append({"subset": sname, "landmark": lname,
                         **agreement(sub[mine].to_numpy(float), sub[theirs].to_numpy(float), TOLERANCE_MS[lname])})
    summ = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    log.info("\n%s", summ.round(2).to_string(index=False))
    C.write_csv(rec, rec_path)
    C.write_csv(summ, sum_path, extra={"n_records": len(rec), "reference": "NeuroKit2 ecg_delineate dwt"})

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for a, (lname, (mine, theirs)) in zip(ax, pairs.items()):
        m = rec[[mine, theirs]].notna().all(axis=1)
        x = (rec.loc[m, mine] + rec.loc[m, theirs]) / 2
        d = rec.loc[m, mine] - rec.loc[m, theirs]
        a.scatter(x, d, s=8, alpha=0.4, c=np.where(rec.loc[m, "polarity_inverted"], "tab:red", "tab:blue"))
        a.axhline(d.mean(), c="k")
        a.axhline(d.mean() + 1.96 * d.std(), c="k", ls="--")
        a.axhline(d.mean() - 1.96 * d.std(), c="k", ls="--")
        a.set(xlabel=f"mean of v2 and NeuroKit2 (ms)", ylabel="v2 - NeuroKit2 (ms)",
              title=f"{lname} (n={int(m.sum())}, bias {d.mean():.1f} ms); red = polarity inverted")
    fig.tight_layout()
    C.write_png(fig, fig_path)
    C.show_and_close(fig)


if __name__ == "__main__":
    main()
