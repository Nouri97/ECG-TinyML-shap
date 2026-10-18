"""
v2_11_scoring_sheet.py - blind scoring sheet for a clinician: are the v2 landmarks right?

72 PTB-XL records are drawn: 48 at random (24 Normal, 24 Abnormal by the PTB-XL label) plus 24 extra records whose
signal polarity was inverted by the pipeline (12 per class, or fewer if unavailable), because only about 16% of records are
inverted (4% of normal, 25% of abnormal) and a purely random sample would contain too few to estimate their error rate. Each panel shows the
median beat of lead II with the v2 landmarks (QRS onset green, J point red, T peak blue, ST window shaded). The panels are
numbered 1-72 in shuffled order (48 random + 24 extra inverted-polarity records with the default settings) and do NOT show the label, the record id or any model output (blind scoring).
The scorer fills Scoring_template_v2.csv:
    qrs_onset, j_point, t_peak : 2 = correct, 1 = acceptable (plausible, within ~20 ms), 0 = wrong, NA = cannot judge
    t_amplitude_usable         : Y / N  (is the signed T amplitude at the marked peak a fair measure of the T wave?)
    noise                      : low / medium / high
Scoring_key_v2.csv maps items to record ids and labels; keep it away from the scorer until scoring is finished.
Report error rates with Wilson 95% intervals, per class (24 records per class give wide intervals: say so).
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
N_PER_CLASS = 24
N_INVERTED_PER_CLASS = 12
PER_PAGE = 12
RUBRIC = "Score every landmark on the template: 2 = correct, 1 = acceptable, 0 = wrong, NA = cannot judge"


def main():
    C.init_run(globals().get("__file__"), SEED)
    log = C.get_logger("scoring_sheet")
    files = CONFIG["files"]
    df = pd.read_csv(C.out_path("features", files["ptbxl_features"]))
    ok = df[df["fail_reason"].fillna("") == ""]
    rnd = pd.concat([ok[ok["label"] == k].sample(n=N_PER_CLASS, random_state=SEED) for k in (0, 1)])
    rest = ok[~ok["ecg_id"].isin(rnd["ecg_id"]) & ok["polarity_inverted"].astype(bool)]
    inv = pd.concat([rest[rest["label"] == k].sample(n=min(N_INVERTED_PER_CLASS, int((rest["label"] == k).sum())), random_state=SEED)
                     for k in (0, 1)])
    rnd, inv = rnd.assign(group="random"), inv.assign(group="inverted_extra")
    n_items = len(rnd) + len(inv)
    if n_items % PER_PAGE:
        raise ValueError(f"{n_items} items do not fill whole pages of {PER_PAGE}; adjust N_PER_CLASS / N_INVERTED_PER_CLASS")
    pages = [C.out_path("figures", files["scoring_page"].format(n=i + 1)) for i in range(n_items // PER_PAGE)]
    tpl, key = C.out_path("tables", files["scoring_template"]), C.out_path("tables", files["scoring_key"])
    C.preflight(pages + [tpl, key])
    import matplotlib.pyplot as plt
    import wfdb

    pick = pd.concat([rnd, inv]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)          # shuffled, blind order
    db = pd.read_csv(os.path.join(CONFIG["ptbxl_dir"], "ptbxl_database.csv"), index_col="ecg_id")

    figs, rows_t, rows_k = [], [], []
    for p in range(len(pages)):
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        figs.append(fig)
        for q, ax in enumerate(axes.ravel()):
            i = p * PER_PAGE + q
            r = pick.iloc[i]
            rec = wfdb.rdrecord(os.path.join(CONFIG["ptbxl_dir"], db.loc[int(r["ecg_id"]), "filename_hr"]))
            _, dbg = X.extract_from_signal(rec.p_signal[:, rec.sig_name.index("II")], X.FS, debug=True)
            x, i_r = dbg["x"], dbg["i_r"]
            t = (np.arange(len(x)) - i_r) * 1000.0 / X.FS
            ax.plot(t, x, c="k", lw=1)
            for k, col, lab in (("on", "tab:green", "QRS onset"), ("j", "tab:red", "J"), ("ipk", "tab:blue", "T peak")):
                ax.axvline(t[dbg[k]], c=col, ls="--", lw=1.2, label=lab)
            ax.axvspan(t[dbg["sa"]], t[dbg["sb"]], color="tab:orange", alpha=0.25, label="ST window")
            ax.set_title(f"Item {i + 1}", fontsize=10)
            ax.set_xlabel("ms from R")
            ax.set_ylabel("mV")
            rows_t.append({"item": i + 1, "page": p + 1, "qrs_onset": "", "j_point": "", "t_peak": "",
                           "t_amplitude_usable": "", "noise": "", "comment": "", "scorer": ""})
            rows_k.append({"item": i + 1, "ecg_id": int(r["ecg_id"]), "label": int(r["label"]),
                           "scp_superclasses": r["scp_superclasses"], "strat_fold": int(r["strat_fold"]),
                           "polarity_inverted": bool(r["polarity_inverted"]), "RR_median_ms": r["RR_median_ms"],
                           "sample_group": r["group"]})
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"Landmark scoring sheet, page {p + 1}/{len(pages)} - {RUBRIC}", fontsize=10)
        fig.tight_layout()

    C.write_csv(pd.DataFrame(rows_t), tpl)
    C.write_csv(pd.DataFrame(rows_k), key, extra={"blind": "keep away from the scorer until scoring is finished"})
    for fig, pth in zip(figs, pages):
        C.write_png(fig, pth)
        C.show_and_close(fig)
    log.info("Wrote %d pages, the template and the key. Scorer sees pages + template only.", len(pages))


if __name__ == "__main__":
    main()
