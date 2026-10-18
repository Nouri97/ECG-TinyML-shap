"""
v2_06_extract_cpsc2018.py - external set: CPSC-2018 (PhysioNet/CinC Challenge 2020 format), Lead II, 500 Hz.

The extraction code is IMPORTED from v2_01_extract_features.py: same filters, same R-peak rule, same median
beat, same landmark rules, same parameters. Nothing is tuned on CPSC-2018.

Data: the CPSC-2018 training records as distributed in the 2020 Challenge (.hea + .mat, 12 leads, 500 Hz,
6-60 s). Download them to CONFIG["cpsc_dir"] (any sub-folders). Verify the exact PhysioNet path/version.
Signal (mV) = (val - adc_zero) / gain, from the header line of lead II. The first 10 s are used, as in PTB-XL.

Labels (binary): Normal iff the only mapped diagnosis is Normal sinus rhythm (SNOMED 426783006);
Abnormal iff at least one of AF, I-AVB, LBBB, RBBB, PAC, PVC, STD, STE is present. Records with none of
these nine classes are excluded and counted ('unmapped_dx'). The class names are kept in 'dx_classes' so
recall per diagnosis can be reported. Note: CPSC-2018 contains no MI or hypertrophy labels and is rhythm-
and conduction-heavy, so this is a different "abnormal" than PTB-XL's.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.io import loadmat

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
import v2_common as C  # noqa: E402
from v2_common import CONFIG  # noqa: E402
import v2_01_extract_features as X  # noqa: E402

SEED = 42
N_JOBS = -1
DX = {"426783006": "Normal", "164889003": "AF", "270492004": "I-AVB", "164909002": "LBBB",
      "59118001": "RBBB", "713427006": "RBBB", "284470004": "PAC", "63593006": "PAC",
      "164884008": "PVC", "427172004": "PVC", "429622005": "STD", "164931005": "STE"}


def read_header(base):
    lines = open(base + ".hea", encoding="utf-8", errors="ignore").read().splitlines()
    head = lines[0].split()
    n_leads, fs = int(head[1]), float(head[2])
    names, gains, zeros = [], [], []
    for ln in lines[1:1 + n_leads]:
        t = ln.split()
        gains.append(float(re.match(r"[-+]?\d*\.?\d+", t[2]).group()))
        zeros.append(float(t[4]))
        names.append(t[-1])
    dx = [c.strip() for ln in lines if "Dx:" in ln for c in ln.split(":", 1)[1].split(",") if c.strip()]
    age = next((ln.split(":", 1)[1].strip() for ln in lines if ln.startswith("#Age")), "")
    sex = next((ln.split(":", 1)[1].strip() for ln in lines if ln.startswith("#Sex")), "")
    return fs, names, gains, zeros, dx, age, sex


def process_cpsc(base):
    rec = {"ecg_id": os.path.basename(base)}
    try:
        fs, names, gains, zeros, dx, age, sex = read_header(base)
        classes = sorted({DX[c] for c in dx if c in DX})
        rec.update(dx_codes="|".join(dx), dx_classes="|".join(classes), age=age, sex=sex)
        if not classes:
            rec.update(label=np.nan, fail_reason="unmapped_dx")
            return rec
        rec["label"] = 0 if classes == ["Normal"] else 1
        if int(fs) != X.FS or "II" not in names:
            rec["fail_reason"] = f"unexpected_fs_or_leads:{fs}"
            return rec
        k = names.index("II")
        val = loadmat(base + ".mat")["val"].astype(float)
        sig = (val[k] - zeros[k]) / gains[k]
        out, _ = X.extract_from_signal(sig, X.FS)
        rec.update(out)
    except Exception as e:
        rec["fail_reason"] = f"read_error:{type(e).__name__}"
    return rec


def main():
    C.init_run(globals().get("__file__"), SEED)
    log = C.get_logger("extract_cpsc2018")
    files = CONFIG["files"]
    feat_path = C.out_path("features", files["cpsc_features"])
    sum_path = C.out_path("logs", files["cpsc_summary"])
    C.preflight([feat_path, sum_path])

    bases = sorted(os.path.splitext(p)[0] for p in glob.glob(os.path.join(CONFIG["cpsc_dir"], "**", "*.hea"),
                                                           recursive=True))
    if not bases:
        raise FileNotFoundError(f"No .hea files under {CONFIG['cpsc_dir']}")
    log.info("CPSC-2018 records found: %d", len(bases))
    df = pd.DataFrame(Parallel(n_jobs=N_JOBS)(delayed(process_cpsc)(b) for b in bases))
    df["fail_reason"] = df["fail_reason"].fillna("")
    unmapped = df["fail_reason"] == "unmapped_dx"
    used = df[~unmapped]
    ok = used["fail_reason"] == ""
    by_reason = used.loc[~ok, "fail_reason"].str.split(":").str[0].value_counts().to_dict()
    by_label = pd.crosstab(used["label"], ok.map({True: "ok", False: "failed"})).to_dict()
    per_class = {c: int(used["dx_classes"].str.contains(c, regex=False).sum())
                 for c in sorted(set(DX.values()))}
    log.info("Unmapped (excluded): %d | used: %d | extraction OK: %d | failures: %s | by label: %s",
             int(unmapped.sum()), len(used), int(ok.sum()), by_reason, by_label)
    log.info("Records per diagnosis (multi-label): %s", per_class)

    columns = (["ecg_id", "label", "dx_classes", "dx_codes", "age", "sex", "fail_reason", "n_peaks", "n_beats",
                "polarity_inverted"] + C.morphology_features() + C.FEATURE_GROUPS["rhythm"]
               + ["qc_qrs_onset_ms", "qc_J_ms", "qc_Tpeak_ms", "qc_t_at_edge", "qc_t_window_end_ms"])
    summary = {"n_found": len(df), "n_unmapped_excluded": int(unmapped.sum()), "n_used": int(len(used)),
               "n_ok": int(ok.sum()), "failures_by_reason": by_reason, "outcome_by_label": by_label,
               "records_per_diagnosis_multilabel": per_class,
               "label_rule": "Normal iff only Normal sinus rhythm (426783006); Abnormal iff any of AF, I-AVB, LBBB, RBBB, PAC, PVC, STD, STE",
               "dx_code_map": DX, "extraction_parameters": X.P, "sampling_rate_hz": X.FS,
               "note": "identical code and parameters as PTB-XL (imported from v2_01); nothing tuned on CPSC-2018",
               "created_at": C._now()}
    C.write_csv(used[columns], feat_path, extra={"extraction_summary": summary})
    C.write_json(summary, sum_path)
    log.info("Wrote %s", feat_path)


if __name__ == "__main__":
    main()