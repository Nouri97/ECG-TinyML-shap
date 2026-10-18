"""
v2_05_nk_gallery.py - look at the records where v2 and NeuroKit2 disagree (visual adjudication).

Reads NK_crosscheck_records_v2.csv, picks 6 random records with a large QRS-duration disagreement
(|diff| > QRS_DIFF_MS) and 6 with a large T-peak disagreement (|diff| > T_DIFF_MS), re-extracts their
median beat and plots v2 landmarks (solid: QRS onset, J, T peak) against NeuroKit2's (dotted; NK onset =
NK J - NK QRS duration). Use it to decide, record by record, which method follows the waveform; ideally
have a clinician co-author score the 12 panels.
"""
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
import v2_common as C  # noqa: E402
from v2_common import CONFIG  # noqa: E402
import v2_01_extract_features as X  # noqa: E402

SEED = 42
QRS_DIFF_MS = 100.0
T_DIFF_MS = 100.0


def main():
    C.init_run(globals().get("__file__"), SEED)
    log = C.get_logger("nk_gallery")
    files = CONFIG["files"]
    fig_path = C.out_path("figures", files["fig_nk_gallery"])
    C.preflight([fig_path])
    import matplotlib.pyplot as plt
    import wfdb

    rec = pd.read_csv(C.out_path("tables", files["nk_records"]))
    db = pd.read_csv(os.path.join(CONFIG["ptbxl_dir"], "ptbxl_database.csv"), index_col="ecg_id")
    q = rec[(rec["QRS_duration_ms"] - rec["nk_qrs_duration"]).abs() > QRS_DIFF_MS]
    t = rec[(rec["qc_Tpeak_ms"] - rec["nk_t_peak"]).abs() > T_DIFF_MS]
    log.info("Large disagreements: QRS %d, T peak %d of %d records", len(q), len(t), len(rec))
    pick = [("QRS", r) for _, r in q.sample(n=min(6, len(q)), random_state=SEED).iterrows()] + \
           [("Tpeak", r) for _, r in t.sample(n=min(6, len(t)), random_state=SEED).iterrows()]

    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    for ax, (kind, r) in zip(axes.ravel(), pick):
        sig = wfdb.rdrecord(os.path.join(CONFIG["ptbxl_dir"], db.loc[int(r["ecg_id"]), "filename_hr"]))
        sig = sig.p_signal[:, sig.sig_name.index("II")]
        out, dbg = X.extract_from_signal(sig, X.FS, debug=True)
        x, i_r = dbg["x"], dbg["i_r"]
        tt = (np.arange(len(x)) - i_r) * 1000.0 / X.FS
        ax.plot(tt, x, c="k", lw=1)
        mine = {"QRS on": r["qc_J_ms"] - r["QRS_duration_ms"], "J": r["qc_J_ms"], "T peak": r["qc_Tpeak_ms"]}
        theirs = {"QRS on": r["nk_j_point"] - r["nk_qrs_duration"], "J": r["nk_j_point"], "T peak": r["nk_t_peak"]}
        for (name, col) in (("QRS on", "tab:green"), ("J", "tab:red"), ("T peak", "tab:blue")):
            ax.axvline(mine[name], c=col, ls="-", lw=1.2, label=f"v2 {name}")
            if np.isfinite(theirs[name]):
                ax.axvline(theirs[name], c=col, ls=":", lw=1.5, label=f"NK {name}")
        ax.set_title(f"[{kind} disagreement] ecg_id {int(r['ecg_id'])} | {'Abnormal' if r['label'] else 'Normal'} | "
                     f"v2 QRS {r['QRS_duration_ms']:.0f} vs NK {r['nk_qrs_duration']:.0f} ms", fontsize=8)
        ax.set_xlabel("ms from R")
        ax.set_ylabel("mV")
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle("Solid = v2 landmarks, dotted = NeuroKit2 (DWT). Which one follows the waveform?")
    fig.tight_layout()
    C.write_png(fig, fig_path)
    C.show_and_close(fig)


if __name__ == "__main__":
    main()
