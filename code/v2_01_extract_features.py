"""
v2_01_extract_features.py - documented Lead-II feature extraction for PTB-XL (500 Hz, native).

Pipeline (every parameter is in P below and is written to feature_definitions_v2.json):
  1. Lead II, 500 Hz, mV, first 10 s. Zero-phase Butterworth band-pass 0.5-40 Hz.
  2. R peaks: 5-15 Hz band-pass -> derivative -> square -> 150 ms moving average -> peak picking
     (min distance 200 ms); threshold = 0.30 x median of the largest 25% candidate heights;
     each peak refined to the max |amplitude| of the filtered ECG within +/-60 ms.
     Polarity: if the median R amplitude is negative the signal is inverted (flag recorded).
  3. Median beat: window [-300, +500] ms around each R; beats not fully inside the record are
     skipped; need >= 3 complete beats. NO fallback: records that fail are kept in the output
     with fail_reason and NaN features, and counted by class in feature_definitions_v2.json.
  4. Landmarks on the median beat (baseline = median of [-90,-60] ms before R, "PR-segment"):
       QRS onset/offset(J): contiguous region around R where a 20 ms max-filtered |slope| exceeds
                            10% of the peak |slope| within R+/-60 ms, searched within R+/-120 ms.
       Q = min in [-80 ms, R]; S = min in [R, +100 ms].
       ST level = mean of [J+60, J+80] ms; ST slope = linear fit on [J+20, J+80] ms.
       T peak = max |amplitude| in [J+80, min(J+400, R+max(RR_median-250, 0.6*RR_median))] ms (the upper bound keeps the next
       beat's P wave/QRS out of the search at fast heart rates); T asymmetry = (right-left)/(right+left) of the
       half-amplitude widths around the T peak (bounded in [-1, 1]).
     P-wave landmarks and PR interval are NOT estimated (no validated detector yet).
  5. Statistics of the baseline-corrected median beat (mV): RMS, kurtosis, skewness, zero-cross rate.
  6. Rhythm (optional group): median RR and RR coefficient of variation from detected R peaks.

Resumable: results are written per chunk (never overwritten; reused only if the code hash matches).
Run in one Colab cell:  %run /content/drive/MyDrive/ECG_Research/code_v2/v2_01_extract_features.py
"""
import ast
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.stats import kurtosis, skew

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
import v2_common as C  # noqa: E402
from v2_common import CONFIG  # noqa: E402

SEED = 42
FS = 500
N_JOBS = -1
CHUNK = 500
MAX_SECONDS = 10

P = dict(bp_low_hz=0.5, bp_high_hz=40.0, bp_order=4,
         det_band_hz=(5.0, 15.0), det_ma_ms=150, det_min_dist_ms=200, det_thr_frac=0.30,
         det_top_frac=0.25, refine_ms=60, pre_ms=300, post_ms=500, min_beats=3,
         baseline_win_ms=(-90, -60), qrs_search_ms=(-120, 120), qrs_slope_ref_ms=60,
         slope_frac=0.10, slope_smooth_ms=8, env_win_ms=20, q_win_ms=80, s_win_ms=100,
         st_win_ms=(60, 80), st_slope_win_ms=(20, 80), t_win_ms=(80, 400), t_min_amp_mV=0.02,
         t_end_rr_margin_ms=250, t_end_rr_fraction=0.6, t_min_window_ms=60,
         rr_valid_ms=(300, 2000))

DEFINITIONS = {
    "RMS_energy": "sqrt(mean(x^2)) of the baseline-corrected median beat (mV)",
    "Kurtosis": "excess kurtosis of the median beat",
    "Skewness": "skewness of the median beat",
    "ZeroCrossRate": "fraction of sign changes of the mean-removed median beat",
    "R_amplitude_mV": "R peak minus baseline (mV), no amplitude normalisation",
    "Q_depth_mV": "max(0, baseline - min amplitude in [-80 ms, R]) (mV); 0 if no Q wave",
    "S_depth_mV": "max(0, baseline - min amplitude in [R, +100 ms]) (mV)",
    "QRS_duration_ms": "QRS offset (J) minus onset from the slope-envelope rule (ms)",
    "ST_level_mV": "mean amplitude in [J+60, J+80] ms relative to baseline (mV)",
    "ST_slope_mV_per_s": "linear-fit slope in [J+20, J+80] ms (mV/s)",
    "T_amplitude_mV": "signed amplitude of the largest deflection in [J+80 ms, min(J+400 ms, R+max(RR_median-250 ms, 0.6*RR_median))] (mV)",
    "T_asymmetry": "(right - left)/(right + left) half-amplitude widths around the T peak, in [-1, 1]",
    "J_to_Tpeak_ms": "time from J point to T peak (ms)",
    "RR_median_ms": "median RR interval of valid detected beats (ms)",
    "RR_cv": "std/mean of valid RR intervals",
}
DROPPED = {
    "R_amplitude (v1)": "constant (=1): beats were max-normalised; replaced by R_amplitude_mV in mV",
    "beat_duration_ms (v1)": "constant (=1870): fixed 187 samples x 10 ms, no information",
    "PR_proxy_ms (v1)": "std 333 ms with 187 unique values: behaves like a landmark index, not a PR interval; "
                        "no validated P-wave detector, so no PR feature in v2",
    "T_symmetry (v1)": "unbounded ratio (up to 4e9, 1,449 records > 1e6); replaced by bounded T_asymmetry",
    "Q_depth / QRS_width_ms / ST_level / T_amplitude (v1)": "redefined in mV on non-normalised beats",
}


def ms(v, fs=FS):
    return int(round(v * fs / 1000.0))


# --------------------------------------------------------------------------- #
# Signal processing
# --------------------------------------------------------------------------- #
def preprocess(sig, fs):
    sos = butter(P["bp_order"], [P["bp_low_hz"], P["bp_high_hz"]], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)


def detect_r_peaks(x, fs):
    sos = butter(2, list(P["det_band_hz"]), btype="band", fs=fs, output="sos")
    y = np.gradient(sosfiltfilt(sos, x)) ** 2
    y = uniform_filter1d(y, size=max(1, ms(P["det_ma_ms"], fs)), mode="nearest")
    cand, _ = find_peaks(y, distance=ms(P["det_min_dist_ms"], fs))
    if len(cand) == 0:
        return np.array([], dtype=int)
    h = y[cand]
    k = max(1, int(np.ceil(P["det_top_frac"] * len(h))))
    thr = P["det_thr_frac"] * np.median(np.sort(h)[::-1][:k])
    cand = cand[h >= thr]
    half = ms(P["refine_ms"], fs)
    refined = []
    for c in cand:
        lo, hi = max(0, c - half), min(len(x), c + half + 1)
        refined.append(lo + int(np.argmax(np.abs(x[lo:hi]))))
    keep = []
    for i in np.unique(refined):
        if keep and i - keep[-1] < ms(P["det_min_dist_ms"], fs):
            if abs(x[i]) > abs(x[keep[-1]]):
                keep[-1] = int(i)
        else:
            keep.append(int(i))
    return np.array(keep, dtype=int)


def median_beat(x, peaks, fs):
    n_pre, n_post = ms(P["pre_ms"], fs), ms(P["post_ms"], fs)
    beats = [x[r - n_pre:r + n_post + 1] for r in peaks if r - n_pre >= 0 and r + n_post < len(x)]
    if len(beats) < P["min_beats"]:
        return None, len(beats)
    return np.median(np.vstack(beats), axis=0), len(beats)


def delineate(b, fs, rr_ms):
    n_pre = ms(P["pre_ms"], fs)
    w = ms(20, fs)
    i_r = n_pre - w + int(np.argmax(b[n_pre - w:n_pre + w + 1]))
    lo, hi = i_r + ms(P["baseline_win_ms"][0], fs), i_r + ms(P["baseline_win_ms"][1], fs)
    x = b - float(np.median(b[lo:hi + 1]))
    n = len(x)

    d = np.gradient(x) * fs
    ad = np.abs(uniform_filter1d(d, size=max(1, ms(P["slope_smooth_ms"], fs)), mode="nearest"))
    env = maximum_filter1d(ad, size=max(1, ms(P["env_win_ms"], fs)), mode="nearest")
    a, z = i_r + ms(P["qrs_search_ms"][0], fs), i_r + ms(P["qrs_search_ms"][1], fs)
    ref = ms(P["qrs_slope_ref_ms"], fs)
    thr = P["slope_frac"] * ad[i_r - ref:i_r + ref + 1].max()
    above = env > thr
    on = i_r
    while on - 1 >= a and above[on - 1]:
        on -= 1
    off = i_r
    while off + 1 <= z and above[off + 1]:
        off += 1
    shrink = ms(P["env_win_ms"], fs) // 2
    on, off = min(i_r, on + shrink), max(i_r, off - shrink)

    q_seg = x[i_r - ms(P["q_win_ms"], fs):i_r + 1]
    s_seg = x[i_r:i_r + ms(P["s_win_ms"], fs) + 1]
    j = off
    sa, sb = j + ms(P["st_win_ms"][0], fs), min(j + ms(P["st_win_ms"][1], fs), n - 1)
    ka, kb = j + ms(P["st_slope_win_ms"][0], fs), min(j + ms(P["st_slope_win_ms"][1], fs), n - 1)
    # T search stops before the next beat's P wave / QRS: end <= R + max(RR - 250 ms, 0.6 * RR).
    # The 0.6*RR floor keeps a usable window at fast heart rates (RR ~ 400 ms) where RR - 250 ms
    # would be too short for the T wave; at RR >= ~625 ms the RR - 250 ms bound applies.
    ta = j + ms(P["t_win_ms"][0], fs)
    t_bound = max(rr_ms - P["t_end_rr_margin_ms"], P["t_end_rr_fraction"] * rr_ms)
    tb = min(j + ms(P["t_win_ms"][1], fs), i_r + ms(t_bound, fs), n - 1)
    if sa > sb or kb - ka < 3:
        raise ValueError("windows_outside_beat")
    if tb - ta < ms(P["t_min_window_ms"], fs):
        raise ValueError("t_window_too_short")

    st_level = float(np.mean(x[sa:sb + 1]))
    st_slope = float(np.polyfit(np.arange(ka, kb + 1) / fs, x[ka:kb + 1], 1)[0])
    seg = x[ta:tb + 1]
    ipk = ta + int(np.argmax(np.abs(seg)))
    t_amp = float(x[ipk])
    if abs(t_amp) < P["t_min_amp_mV"]:
        asym = 0.0
    else:
        half_amp = 0.5 * abs(t_amp)
        li = ipk
        while li - 1 >= ta and abs(x[li - 1]) >= half_amp:
            li -= 1
        ri = ipk
        while ri + 1 <= tb and abs(x[ri + 1]) >= half_amp:
            ri += 1
        left, right = ipk - li, ri - ipk
        asym = float((right - left) / (right + left)) if (right + left) > 0 else 0.0

    zc = x - x.mean()
    feats = {
        "RMS_energy": float(np.sqrt(np.mean(x ** 2))), "Kurtosis": float(kurtosis(x)),
        "Skewness": float(skew(x)),
        "ZeroCrossRate": float(np.mean(np.signbit(zc[1:]) != np.signbit(zc[:-1]))),
        "R_amplitude_mV": float(x[i_r]), "Q_depth_mV": max(0.0, -float(q_seg.min())),
        "S_depth_mV": max(0.0, -float(s_seg.min())),
        "QRS_duration_ms": (off - on) * 1000.0 / fs,
        "ST_level_mV": st_level, "ST_slope_mV_per_s": st_slope, "T_amplitude_mV": t_amp,
        "T_asymmetry": asym, "J_to_Tpeak_ms": (ipk - j) * 1000.0 / fs,
        "qc_qrs_onset_ms": (on - i_r) * 1000.0 / fs, "qc_J_ms": (j - i_r) * 1000.0 / fs,
        "qc_Tpeak_ms": (ipk - i_r) * 1000.0 / fs, "qc_t_at_edge": bool(ipk in (ta, tb)),
        "qc_t_window_end_ms": (tb - i_r) * 1000.0 / fs,
    }
    dbg = {"x": x, "i_r": i_r, "on": on, "j": j, "ipk": ipk, "sa": sa, "sb": sb, "ta": ta, "tb": tb}
    return feats, dbg


def extract_from_signal(sig, fs=FS, debug=False):
    out = {"fail_reason": "", "n_peaks": 0, "n_beats": 0, "polarity_inverted": False}
    sig = np.asarray(sig, dtype=float)[: int(MAX_SECONDS * fs)]
    if len(sig) < 3 * fs or not np.all(np.isfinite(sig)) or np.std(sig) < 1e-6:
        out["fail_reason"] = "flat_short_or_nonfinite"
        return out, None
    x = preprocess(sig, fs)
    peaks = detect_r_peaks(x, fs)
    out["n_peaks"] = int(len(peaks))
    if len(peaks) < P["min_beats"]:
        out["fail_reason"] = "too_few_r_peaks"
        return out, None
    if np.median(x[peaks]) < 0:
        x, out["polarity_inverted"] = -x, True
    mb, nb = median_beat(x, peaks, fs)
    out["n_beats"] = int(nb)
    if mb is None:
        out["fail_reason"] = "too_few_complete_beats"
        return out, None
    rr = np.diff(peaks) * 1000.0 / fs
    rr = rr[(rr >= P["rr_valid_ms"][0]) & (rr <= P["rr_valid_ms"][1])]
    if len(rr) < 2:
        out["fail_reason"] = "too_few_valid_rr"
        return out, None
    try:
        feats, dbg = delineate(mb, fs, float(np.median(rr)))
    except Exception as e:  # recorded, never silently replaced by a fallback
        out["fail_reason"] = f"delineation_error:{type(e).__name__}:{e}"[:80]
        return out, None
    feats["RR_median_ms"] = float(np.median(rr))
    feats["RR_cv"] = float(np.std(rr) / np.mean(rr))
    if not all(np.isfinite(v) for k, v in feats.items() if not k.startswith("qc_t_at")):
        out["fail_reason"] = "nonfinite_feature"
        return out, None
    out.update(feats)
    return out, (dbg if debug else None)


def process_one(ecg_id, rel_path, ptbxl_dir):
    """Runs inside joblib workers: only plain arguments, no CONFIG access."""
    import wfdb
    try:
        rec = wfdb.rdrecord(os.path.join(ptbxl_dir, rel_path))
        if int(rec.fs) != FS or "II" not in rec.sig_name:
            return {"ecg_id": ecg_id, "fail_reason": f"unexpected_fs_or_leads:{rec.fs}"}
        out, _ = extract_from_signal(rec.p_signal[:, rec.sig_name.index("II")], FS)
        return {"ecg_id": ecg_id, **out}
    except Exception as e:
        return {"ecg_id": ecg_id, "fail_reason": f"read_error:{type(e).__name__}"}


# --------------------------------------------------------------------------- #
# Labels (old rule is identified, not assumed)
# --------------------------------------------------------------------------- #
def _apply_rule(supers, rule):
    norm_only = supers.apply(lambda s: s == ["NORM"])
    has_norm = supers.apply(lambda s: "NORM" in s)
    empty = supers.apply(len) == 0
    if rule == "A":
        return (~norm_only).astype(int), pd.Series(True, index=supers.index)
    if rule == "B":
        return (~has_norm).astype(int), pd.Series(True, index=supers.index)
    if rule == "C":
        return (~norm_only).astype(int), ~empty
    return (~has_norm).astype(int), ~empty


RULE_TEXT = {"A": "Normal iff diagnostic superclasses == [NORM]; all other records Abnormal (incl. no diagnostic class)",
             "B": "Normal iff NORM in superclasses; all other records Abnormal (incl. no diagnostic class)",
             "C": "Normal iff superclasses == [NORM]; records without any diagnostic class dropped",
             "D": "Normal iff NORM in superclasses; records without any diagnostic class dropped"}


def build_label_table(log):
    d = CONFIG["ptbxl_dir"]
    db = pd.read_csv(os.path.join(d, "ptbxl_database.csv"), index_col="ecg_id")
    scp = pd.read_csv(os.path.join(d, "scp_statements.csv"), index_col=0)
    diag = scp.loc[scp["diagnostic"] == 1, "diagnostic_class"].dropna()
    supers = db["scp_codes"].apply(ast.literal_eval).apply(
        lambda c: sorted({diag[k] for k in c if k in diag.index}))
    exp = CONFIG["label_expectation"]
    report = []
    for rule in "ABCD":
        label, keep = _apply_rule(supers, rule)
        f10 = keep & (db["strat_fold"] == 10)
        report.append({"rule": rule, "n_total": int(keep.sum()), "fold10_n": int(f10.sum()),
                       "fold10_prevalence": float(label[f10].mean()), "description": RULE_TEXT[rule]})
    rep = pd.DataFrame(report)
    log.info("Label-rule candidates vs expectation %s:\n%s", exp, rep.drop(columns="description").to_string(index=False))
    if exp is None:
        chosen = "A"
    else:
        ok = rep[(rep["fold10_prevalence"] - exp["fold10_prevalence"]).abs() <= exp["tol"]]
        if ok.empty:
            raise ValueError("No label rule reproduces the old fold-10 prevalence; inspect the table above "
                             "and set CONFIG['label_expectation'] / the rule explicitly.")
        chosen = ok.loc[(ok["n_total"] - exp["n_total"]).abs().idxmin(), "rule"]
    label, keep = _apply_rule(supers, chosen)
    log.info("Label rule used: %s - %s", chosen, RULE_TEXT[chosen])
    t = db.loc[keep, ["patient_id", "strat_fold", "age", "sex", "filename_hr"]].copy()
    t["label"] = label[keep]
    t["scp_superclasses"] = supers[keep].apply(lambda s: "|".join(s))
    return t.reset_index(), rep, chosen


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _chunk_hash(ids):
    return hashlib.md5(",".join(map(str, ids)).encode()).hexdigest()


def _reuse_ok(path, ids_hash):
    import json
    with open(C.meta_path(path), encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get("script_sha256") == C.RUN["script_sha256"] and \
        meta.get("extra", {}).get("ids_hash") == ids_hash


def landmark_figure(table, ok_ids, log, n=12):
    import matplotlib.pyplot as plt
    import wfdb
    rng = np.random.default_rng(SEED)
    pick = rng.choice(ok_ids, size=min(n, len(ok_ids)), replace=False)
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    for ax, eid in zip(axes.ravel(), pick):
        rel = table.loc[table["ecg_id"] == eid, "filename_hr"].iloc[0]
        rec = wfdb.rdrecord(os.path.join(CONFIG["ptbxl_dir"], rel))
        out, dbg = extract_from_signal(rec.p_signal[:, rec.sig_name.index("II")], FS, debug=True)
        x, i_r = dbg["x"], dbg["i_r"]
        t = (np.arange(len(x)) - i_r) * 1000.0 / FS
        ax.plot(t, x, c="k", lw=1)
        for key, col, lab in (("on", "tab:green", "QRS on"), ("j", "tab:red", "J"), ("ipk", "tab:blue", "T peak")):
            ax.axvline(t[dbg[key]], c=col, ls="--", lw=1, label=lab)
        ax.axvspan(t[dbg["sa"]], t[dbg["sb"]], color="tab:orange", alpha=0.25, label="ST window")
        lab = int(table.loc[table["ecg_id"] == eid, "label"].iloc[0])
        ax.set_title(f"ecg_id {eid} | {'Abnormal' if lab else 'Normal'} | QRS {out['QRS_duration_ms']:.0f} ms, "
                     f"ST {out['ST_level_mV']:.2f} mV", fontsize=9)
        ax.set_xlabel("ms from R")
        ax.set_ylabel("mV")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Landmark check on 12 random records (inspect visually before trusting any feature)")
    fig.tight_layout()
    return fig


def main():
    C.init_run(globals().get("__file__"), SEED)
    log = C.get_logger("extract_features")
    files = CONFIG["files"]
    final_path = C.out_path("features", files["ptbxl_features"])
    defs_path = C.out_path("logs", files["feature_definitions"])
    fig_path = C.out_path("figures", files["fig_landmarks"])
    C.preflight([final_path, defs_path, fig_path])

    table, rule_report, rule = build_label_table(log)
    log.info("Records to process: %d (labels: %s)", len(table), table["label"].value_counts().to_dict())

    ids = table["ecg_id"].tolist()
    paths = table["filename_hr"].tolist()
    parts = []
    t0 = time.time()
    for ci, s in enumerate(range(0, len(ids), CHUNK)):
        cid, cpath = ids[s:s + CHUNK], paths[s:s + CHUNK]
        cfile = C.out_path("chunks", f"ptbxl_v2_chunk_{ci:04d}.csv")
        h = _chunk_hash(cid)
        if os.path.exists(cfile):
            if not _reuse_ok(cfile, h):
                raise RuntimeError(f"Chunk {cfile} was written by different code or a different record set. "
                                   "Delete the chunk folder manually to start over.")
            parts.append(pd.read_csv(cfile))
            log.info("chunk %d: reused (code hash matches)", ci)
            continue
        res = Parallel(n_jobs=N_JOBS)(delayed(process_one)(i, p, CONFIG["ptbxl_dir"]) for i, p in zip(cid, cpath))
        df = pd.DataFrame(res)
        C.write_csv(df, cfile, extra={"chunk": ci, "ids_hash": h})
        parts.append(df)
        log.info("chunk %d done (%d records, %.0f s elapsed)", ci, len(df), time.time() - t0)

    feats = pd.concat(parts, ignore_index=True)
    out = table.merge(feats, on="ecg_id", how="left")
    out["fail_reason"] = out["fail_reason"].fillna("")
    ok = out["fail_reason"] == ""
    by_reason = out.loc[~ok, "fail_reason"].str.split(":").str[0].value_counts().to_dict()
    by_label = pd.crosstab(out["label"], ok.map({True: "ok", False: "failed"})).to_dict()
    log.info("Extraction OK: %d / %d. Failures by reason: %s. By label: %s", ok.sum(), len(out), by_reason, by_label)

    columns = (["ecg_id", "patient_id", "strat_fold", "label", "age", "sex", "scp_superclasses", "fail_reason",
                "n_peaks", "n_beats", "polarity_inverted"] + C.morphology_features()
               + C.FEATURE_GROUPS["rhythm"] + ["qc_qrs_onset_ms", "qc_J_ms", "qc_Tpeak_ms", "qc_t_at_edge", "qc_t_window_end_ms"])
    summary = {"n_records": int(len(out)), "n_ok": int(ok.sum()), "failures_by_reason": by_reason,
               "outcome_by_label": by_label, "polarity_inverted_n": int(out["polarity_inverted"].fillna(False).sum())}
    C.write_csv(out[columns], final_path, extra={"label_rule": rule, "extraction_summary": summary})
    defs = {"features": {g: {f: DEFINITIONS[f] for f in fs} for g, fs in C.FEATURE_GROUPS.items()},
            "model_feature_groups": list(C.MORPHOLOGY_GROUPS),
            "dropped_features": DROPPED, "sampling_rate_hz": FS, "lead": "II", "units": "mV",
            "normalization": "none (absolute mV, baseline-corrected on the median beat only)",
            "max_seconds_per_record": MAX_SECONDS, "parameters": P, "label_rule": {"chosen": rule,
            "candidates": rule_report.to_dict(orient="records")}, "extraction_summary": summary,
            "no_fallback": "records whose extraction fails are retained with fail_reason and excluded downstream",
            "created_at": C._now()}
    C.write_json(defs, defs_path)
    log.info("Wrote %s and %s", final_path, defs_path)

    try:
        fig = landmark_figure(out, out.loc[ok, "ecg_id"].to_numpy(), log)
        C.write_png(fig, fig_path)
        C.show_and_close(fig)
    except Exception as e:
        log.warning("Landmark figure failed (%s: %s); features are already saved.", type(e).__name__, e)


if __name__ == "__main__":
    main()