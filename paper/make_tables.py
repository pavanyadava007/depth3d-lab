"""Generate the LaTeX tables and the number macros of the paper from results/*.json.

Reads the json files exactly the way scripts/make_results.py does (its helpers are replicated here so
that this script does not depend on the cwd or on importing a script). Writes

    paper/tables/*.tex     booktabs tables, one per track, pulled into main.tex with \\input
    paper/numbers.tex      \\newcommand macros for every number that the prose cites

Optional files (MoGe-2 rows, seed-1/2 files for 3DGS and next-frame) are used when present and
reported as "not measured" when absent. Every timing is labelled NVIDIA L4.

Run: .venv/bin/python paper/make_tables.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TAB = ROOT / "paper" / "tables"
NUM = ROOT / "paper" / "numbers.tex"
GPU = "NVIDIA L4"

# ------------------------------------------------------------------ helpers (same semantics as make_results.py)


def load(name: str) -> dict | None:
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def f(x, nd=3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    if x != x:  # nan
        return "nan"
    return f"{x:.{nd}f}"


def ci(c: dict | None, nd=3) -> str:
    if not c or c.get("n", 0) == 0:
        return "n/a"
    return f"{c['mean']:.{nd}f} [{c['lo']:.{nd}f}, {c['hi']:.{nd}f}]"


def tex(s: str) -> str:
    """Escape the few characters that appear in labels."""
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def tabular(header: list[str], rows: list[list[str]], align: str, small: bool = True) -> str:
    out = []
    if small:
        out.append(r"\footnotesize\renewcommand{\arraystretch}{0.92}")
    out.append(r"\begin{tabular}{" + align + "}")
    out.append(r"\toprule")
    out.append(" & ".join(header) + r" \\")
    out.append(r"\midrule")
    for r in rows:
        if r == ["MIDRULE"]:
            out.append(r"\midrule")
        else:
            out.append(" & ".join(r) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    return "\n".join(out) + "\n"


def write(name: str, text: str) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / name).write_text(text)


MACROS: dict[str, str] = {}


def M(name: str, value) -> None:
    """Register a macro. Names must be letters only (LaTeX)."""
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise ValueError(f"bad macro name {name}")
    MACROS[name] = str(value)


VARIANT_LABEL = {
    "depth_pro": "Depth Pro",
    "depth_pro_gtfocal": "Depth Pro (true f)",
    "dav2_metric_outdoor_large": "DA-V2 outdoor L",
    "moge2": "MoGe-2",
    "moge2_gtfocal": "MoGe-2 (true f)",
}
PROSE_NAME = {"depth_pro": "Depth Pro", "dav2_metric_outdoor_large": "Depth Anything V2", "moge2": "MoGe-2"}
VARIANT_MACRO = {
    "depth_pro": "DP",
    "depth_pro_gtfocal": "GTF",
    "dav2_metric_outdoor_large": "DAV",
    "moge2": "MOGE",
    "moge2_gtfocal": "MOGEGTF",
}
CAM_SHORT = {
    "CAM_FRONT": "F",
    "CAM_FRONT_LEFT": "FL",
    "CAM_FRONT_RIGHT": "FR",
    "CAM_BACK": "B",
    "CAM_BACK_LEFT": "BL",
    "CAM_BACK_RIGHT": "BR",
}
CAM_MACRO = {
    "CAM_FRONT": "Front",
    "CAM_FRONT_LEFT": "FrontLeft",
    "CAM_FRONT_RIGHT": "FrontRight",
    "CAM_BACK": "Back",
    "CAM_BACK_LEFT": "BackLeft",
    "CAM_BACK_RIGHT": "BackRight",
}
BIN_MACRO = {"1-20m": "Near", "20-40m": "Mid", "40-80m": "Far"}


# ------------------------------------------------------------------ data / GT


def gt_stats() -> None:
    gt = load("lidar_gt_stats.json")
    if not gt:
        M("GTimages", "not measured")
        return
    M("GTimages", f"{gt['n_images']:,}")
    M("GTkeyframes", f"{gt['n_keyframes']:,}")
    M("GTwallSeconds", f(gt["wall_time_s"], 0))
    pc = gt["per_camera"]
    ppi = [v["points_per_image"] for v in pc.values()]
    M("GTptsMin", f(min(ppi), 0))
    M("GTptsMax", f(max(ppi), 0))
    med = [v["median_depth_m"] for v in pc.values()]
    M("GTmedianZmin", f(min(med), 1))
    M("GTmedianZmax", f(max(med), 1))
    near = [v["frac_1_20m"] for v in pc.values()]
    M("GTfracNearMin", f(100 * min(near), 0))
    M("GTfracNearMax", f(100 * max(near), 0))
    rows = [
        [tex(c), f(v["points_per_image"], 0), f(v["median_depth_m"], 1), f(v["frac_1_20m"], 2),
         f(v["frac_20_40m"], 2), f(v["frac_40_80m"], 2)]
        for c, v in pc.items()
    ]  # fmt: skip
    write("lidar_gt.tex", tabular(["camera", "pts/img", "median z [m]", "frac 1-20 m", "frac 20-40 m", "frac 40-80 m"],
                                  rows, "lrrrrr"))  # fmt: skip


def scene_dynamics() -> None:
    dyn = load("scene_dynamics.json")
    if not dyn:
        M("GSscene", "not measured")
        return
    sel = dyn["selected_scene"]
    d = dyn["scenes"][sel]
    M("GSscene", sel)
    M("GSsceneMoving", f"{d['n_moving_annotations_vel']:,}")
    M("GSsceneAnnotations", f"{d['n_annotations']:,}")
    M("GSsceneFracMoving", f(100 * d["frac_moving_vel"], 0))
    M("GSsceneKeyframes", str(d["n_keyframes"]))
    M("DynVelThreshold", f(dyn["velocity_threshold_mps"], 1))
    rows = [[s, str(v["n_keyframes"]), f"{v['n_annotations']:,}", f"{v['n_moving_annotations_vel']:,}",
             f(v["frac_moving_vel"], 2)] for s, v in dyn["scenes"].items()]  # fmt: skip
    rows.sort(key=lambda r: int(r[3].replace(",", "")))
    write("scene_dynamics.tex", tabular(["scene", "keyframes", "annotations", "moving (velocity)", "frac moving"],
                                        rows, "lrrrr"))  # fmt: skip


# ------------------------------------------------------------------ Track A


def track_a() -> None:
    files = sorted(RES.glob("mono_depth_*.json"))
    results = [json.loads(p.read_text()) for p in files]
    order = ["depth_pro", "depth_pro_gtfocal", "dav2_metric_outdoor_large", "moge2", "moge2_gtfocal"]
    results.sort(key=lambda r: order.index(r["variant"]) if r["variant"] in order else 99)
    have = {r["variant"] for r in results}
    moge = "moge2" in have
    M("MoGeStatus", "measured" if moge else "not measured")
    M("MoGeSentence", "MoGe-2 was run with the same protocol, with its own focal estimate and with the true focal length." if moge else
       "A MoGe-2 wrapper exists in the code, but the model was not run inside the GPU time budget, so its row reads not measured.")
    by = {r["variant"]: r for r in results}
    if moge:
        m = by["moge2"]
        raw, sc = m["raw"]["abs_rel"], m["scaled"]["abs_rel"]
        share = 100 * (raw - sc) / raw
        scl = [x["median"] for x in m["scale_factor_by_camera"].values()]
        own = [r for r in results if not r["variant"].endswith("_gtfocal")]
        rank_raw = " $<$ ".join(PROSE_NAME.get(r["variant"], tex(r["variant"])) for r in sorted(own, key=lambda r: r["raw"]["abs_rel"]))
        rank_sc = " $<$ ".join(PROSE_NAME.get(r["variant"], tex(r["variant"])) for r in sorted(own, key=lambda r: r["scaled"]["abs_rel"]))
        txt = (f"MoGe-2 has raw AbsRel {f(raw)} and {f(sc)} after median scaling, so {f(share, 0)}\\% of its raw error is a "
               f"per-image scale error; its median scale factors per camera lie between {f(min(scl), 2)} and {f(max(scl), 2)}.")
        ab = f" MoGe-2 has raw AbsRel {f(raw)} and {f(sc)} after scaling"
        disc = ""
        if "moge2_gtfocal" in by:
            g = by["moge2_gtfocal"]
            graw = g["raw"]["abs_rel"]
            gscl = [x["median"] for x in g["scale_factor_by_camera"].values()]
            direction = "lower" if graw < raw else "higher"
            txt += (f" With the true focal length its raw AbsRel is {f(graw)}, {direction} than with its own focal estimate, and "
                    f"the scale factors lie between {f(min(gscl), 2)} and {f(max(gscl), 2)}.")
            ab += f", and with the true focal its raw AbsRel is {f(graw)} ({direction} than with its own estimate)"
            fr = [x["median_ratio_pred_over_gt"] for x in m.get("focal_by_camera", {}).values()]
            if fr:
                txt += f" Its focal estimate is {f(min(fr), 2)} to {f(max(fr), 2)} times the true focal."
            disc = ((f"MoGe-2 does not show this effect: the true focal lowers its raw AbsRel from {f(raw)} to {f(graw)}."
                     if graw < raw else
                     f"MoGe-2 shows the same effect: the true focal raises its raw AbsRel from {f(raw)} to {f(graw)}.")
                    )
        txt += (f" Ordered by raw AbsRel the own-focal variants are {rank_raw}; ordered by median-scaled AbsRel they are {rank_sc}."
                f" MoGe-2 takes {f(m['timing_ms_per_image']['median'], 0)} ms per image on the {GPU} (median).")
        M("MoGeResults", txt)
        M("MoGeAbstract", ab + ".")
        M("MoGeDiscussion", disc)
        shares = {r["variant"]: (r["raw"]["abs_rel"] - r["scaled"]["abs_rel"]) / r["raw"]["abs_rel"] for r in own}
        smallest = min(shares, key=shares.get) == "moge2"
        M("MoGeScaleShape", f"MoGe-2 has {'the smallest scale share of the three' if smallest else 'a scale share of'} ({f(share, 0)}\\%) "
                            f"and a median-scaled AbsRel of {f(sc)}.")
    else:
        M("MoGeResults", "MoGe-2: not measured.")
        M("MoGeAbstract", "")
        M("MoGeDiscussion", "")
        M("MoGeScaleShape", "")
    M("MoGeRowNote", "" if moge else r"MoGe-2 & \multicolumn{9}{l}{not measured (wrapper exists, model not run; see Section 6)} \\")
    if not results:
        M("TrackAStatus", "not measured")
        write("track_a_main.tex", "Track A: not measured (no results/mono\\_depth\\_*.json).\n")
        return
    M("TrackAStatus", "measured")

    # main table: raw and median-scaled rows per variant
    hdr = ["model", "alignment", "AbsRel", "SqRel", "RMSE [m]", r"RMSE$_{\log}$", r"$\delta_1$", r"$\delta_2$",
           r"$\delta_3$", "ms/img"]  # fmt: skip
    rows = []
    for r in results:
        v = r["variant"]
        mac = VARIANT_MACRO.get(v, re.sub("[^A-Za-z]", "", v).upper())
        for key, label in (("raw", "raw"), ("scaled", "median-scaled")):
            m = r[key]
            rows.append([VARIANT_LABEL.get(v, tex(v)) if key == "raw" else "", label, f(m["abs_rel"]), f(m["sq_rel"]),
                         f(m["rmse"]), f(m["rmse_log"]), f(m["d1"]), f(m["d2"]), f(m["d3"]),
                         f(r["timing_ms_per_image"]["median"], 0) if key == "raw" else ""])  # fmt: skip
            suf = "Raw" if key == "raw" else "Scaled"
            M(f"{mac}absrel{suf}", f(m["abs_rel"]))
            M(f"{mac}sqrel{suf}", f(m["sq_rel"]))
            M(f"{mac}rmse{suf}", f(m["rmse"], 2))
            M(f"{mac}rmselog{suf}", f(m["rmse_log"]))
            M(f"{mac}dOne{suf}", f(m["d1"]))
            M(f"{mac}dThree{suf}", f(m["d3"]))
        raw, sc = r["raw"]["abs_rel"], r["scaled"]["abs_rel"]
        M(f"{mac}scaleShare", f(100 * (raw - sc) / raw, 0))
        M(f"{mac}msPerImage", f(r["timing_ms_per_image"]["median"], 0))
        M(f"{mac}wallMinutes", f(r["wall_time_s"] / 60, 1))
        M(f"{mac}images", f"{r['n_images']:,}")
        far = r["raw"].get("frac_pred_above_max")
        M(f"{mac}fracAboveMax", f(100 * far, 2) if far is not None else "n/a")
        M(f"{mac}absrelRawCI", ci(r["per_image_abs_rel_raw_ci"]))
        M(f"{mac}absrelScaledCI", ci(r["per_image_abs_rel_scaled_ci"]))
        g = r["per_group_raw"]
        worst = max(g, key=lambda k: g[k]["abs_rel"])
        best = min(g, key=lambda k: g[k]["abs_rel"])
        M(f"{mac}worstGroup", worst)
        M(f"{mac}bestGroup", best)
        for grp in ("front", "side", "back"):
            G = grp.capitalize()
            M(f"{mac}absrel{G}Raw", f(r["per_group_raw"][grp]["abs_rel"]))
            M(f"{mac}absrel{G}Scaled", f(r["per_group_scaled"][grp]["abs_rel"]))
            M(f"{mac}dOne{G}Raw", f(r["per_group_raw"][grp]["d1"]))
            M(f"{mac}dOne{G}Scaled", f(r["per_group_scaled"][grp]["d1"]))
        for cam, s in r["scale_factor_by_camera"].items():
            C = CAM_MACRO[cam]
            M(f"{mac}scale{C}", f(s["median"], 2))
            M(f"{mac}scale{C}pTen", f(s["p10"], 2))
            M(f"{mac}scale{C}pNinety", f(s["p90"], 2))
            M(f"{mac}absrelCam{C}Raw", f(r["per_camera_raw"][cam]["abs_rel"]))
            M(f"{mac}absrelCam{C}Scaled", f(r["per_camera_scaled"][cam]["abs_rel"]))
        for cam, foc in r.get("focal_by_camera", {}).items():
            C = CAM_MACRO[cam]
            M(f"{mac}focalRatio{C}", f(foc["median_ratio_pred_over_gt"], 2))
            M(f"{mac}focalTrue{C}", f(foc["gt_focal_px"], 0))
            M(f"{mac}focalPred{C}", f(foc["pred_focal_px_median"], 0))
        if r.get("focal_by_camera"):
            ratios = [x["median_ratio_pred_over_gt"] for x in r["focal_by_camera"].values()]
            M(f"{mac}focalRatioMin", f(min(ratios), 2))
            M(f"{mac}focalRatioMax", f(max(ratios), 2))
        scales = [x["median"] for x in r["scale_factor_by_camera"].values()]
        nb = [x["median"] for c, x in r["scale_factor_by_camera"].items() if c != "CAM_BACK"]
        M(f"{mac}scaleNonBackMin", f(min(nb), 2))
        M(f"{mac}scaleNonBackMax", f(max(nb), 2))
        M(f"{mac}scaleMin", f(min(scales), 2))
        M(f"{mac}scaleMax", f(max(scales), 2))
        for b, ra in r["per_bin_raw"].items():
            Bm = BIN_MACRO[b]
            scb = r["per_bin_scaled"][b]
            M(f"{mac}absrel{Bm}Raw", f(ra["abs_rel"]))
            M(f"{mac}absrel{Bm}Scaled", f(scb["abs_rel"]))
            M(f"{mac}rmse{Bm}Raw", f(ra["rmse"], 2))
            M(f"{mac}rmse{Bm}Scaled", f(scb["rmse"], 2))
            M(f"{mac}points{Bm}", f"{ra['n_points']:,}")
    if not moge:
        rows.append(["MoGe-2", r"\multicolumn{9}{l}{not measured: the wrapper exists, the model was not run (see Section~\ref{sec:limits})}"])
    write("track_a_main.tex", tabular(hdr, rows, "llrrrrrrrr"))

    # per-image bootstrap CI table
    rows = [[VARIANT_LABEL.get(r["variant"], tex(r["variant"])), ci(r["per_image_abs_rel_raw_ci"]),
             ci(r["per_image_abs_rel_scaled_ci"])] for r in results]  # fmt: skip
    write("track_a_ci.tex", tabular(["model", "raw", "median-scaled"], rows, "lrr"))

    # per camera: AbsRel raw / scaled and scale factor
    cams = list(results[0]["per_camera_raw"].keys())
    hdr = ["model"] + [CAM_SHORT[c] for c in cams]
    rows = []
    for r in results:
        rows.append([VARIANT_LABEL.get(r["variant"], tex(r["variant"]))]
                    + [f"{f(r['per_camera_raw'][c]['abs_rel'])} / {f(r['per_camera_scaled'][c]['abs_rel'])}" for c in cams])  # fmt: skip
    write("track_a_per_camera.tex", tabular(hdr, rows, "l" + "r" * len(cams)))

    hdr = ["model", "quantity"] + [CAM_SHORT[c] for c in cams]
    rows = []
    foc0 = next((r["focal_by_camera"] for r in results if r.get("focal_by_camera")), None)
    if foc0:
        rows.append(["nuScenes calibration", "true focal [px]"] + [f(foc0[c]["gt_focal_px"], 0) for c in cams])
        rows.append(["MIDRULE"])
    for r in results:
        sc = r["scale_factor_by_camera"]
        rows.append([VARIANT_LABEL.get(r["variant"], tex(r["variant"])), "median scale GT/pred"] + [f(sc[c]["median"]) for c in cams])
        foc = r.get("focal_by_camera", {})
        if foc and not r["variant"].endswith("_gtfocal"):
            rows.append(["", "pred/true focal"] + [f(foc[c]["median_ratio_pred_over_gt"]) if c in foc else "n/a" for c in cams])
    write("track_a_scale.tex", tabular(hdr, rows, "ll" + "r" * len(cams)))

    hdr = ["model", "group", "AbsRel raw", "scaled", r"$\delta_1$ raw", "scaled", "points"]
    rows = []
    for r in results:
        for i, g in enumerate(("front", "side", "back")):
            ra, sc = r["per_group_raw"][g], r["per_group_scaled"][g]
            rows.append([VARIANT_LABEL.get(r["variant"], tex(r["variant"])) if i == 0 else "", g, f(ra["abs_rel"]),
                         f(sc["abs_rel"]), f(ra["d1"]), f(sc["d1"]), f"{ra['n_points']:,}"])  # fmt: skip
    write("track_a_groups.tex", tabular(hdr, rows, "llrrrrr"))

    hdr = ["model", "bin", "AbsRel raw", "scaled", "RMSE raw", "scaled", "points"]
    rows = []
    for r in results:
        for i, (b, ra) in enumerate(r["per_bin_raw"].items()):
            sc = r["per_bin_scaled"][b]
            rows.append([VARIANT_LABEL.get(r["variant"], tex(r["variant"])) if i == 0 else "", b.replace("m", " m"),
                         f(ra["abs_rel"]), f(sc["abs_rel"]), f(ra["rmse"], 2), f(sc["rmse"], 2), f"{ra['n_points']:,}"])  # fmt: skip
    write("track_a_depth_bins.tex", tabular(hdr, rows, "llrrrrr"))

    fc = load("depth_pro_focal_check.json")
    if fc:
        diffs = [max(c["max_abs_diff_estimated_focal_variant_m"], c["max_abs_diff_known_focal_variant_m"]) for c in fc["cameras"].values()]
        M("FocalCheckMaxDiff", f(max(diffs), 1))
        M("FocalCheckDevice", fc["device"])
    else:
        M("FocalCheckMaxDiff", "not measured")
        M("FocalCheckDevice", "n/a")


# ------------------------------------------------------------------ Track B

ALIGN_LABEL = {
    "vggt_raw": "VGGT raw (relative scale)",
    "vggt_seq_scaled": "VGGT, one scale per sequence",
    "vggt_sim3_scaled": "VGGT, sim(3) scale from cameras",
    "vggt_img_scaled": "VGGT, per-image median scale",
    "depth_pro_raw": "Depth Pro raw (metric)",
    "depth_pro_img_scaled": "Depth Pro, per-image median scale",
    "depth_pro_gtfocal_raw": "Depth Pro raw, true focal",
}
ALIGN_MACRO = {
    "vggt_raw": "VggtRaw",
    "vggt_seq_scaled": "VggtSeq",
    "vggt_sim3_scaled": "VggtSim",
    "vggt_img_scaled": "VggtImg",
    "depth_pro_raw": "DpRaw",
    "depth_pro_img_scaled": "DpImg",
    "depth_pro_gtfocal_raw": "DpGtfRaw",
}
SETTING_MACRO = {"rig6": "Rig", "temporal5": "Temp"}


def track_b() -> None:
    r = load("multiview_vggt.json")
    if not r:
        M("TrackBStatus", "not measured")
        write("track_b.tex", "Track B: not measured (results/multiview\\_vggt.json missing).\n")
        write("track_b_pose.tex", "")
        return
    M("TrackBStatus", "measured")
    M("VggtModelLoadSeconds", f(r["model_load_s"], 0))
    hdr = ["alignment / model", "AbsRel [95\\% CI]", r"$\delta_1$", "RMSE [m]"]
    rows = []
    prow = []
    for key, s in r["summary"].items():
        S = SETTING_MACRO.get(key, key)
        M(f"B{S}nSeq", str(s["n_sequences"]))
        M(f"B{S}nPoints", f"{s['n_ref_points_total']:,}")
        M(f"B{S}msPerSeq", f(s["timing_ms_per_sequence"]["mean"], 0))
        M(f"B{S}scaleSeqMedian", f(s["scale_seq_median"], 1))
        rows.append([r"\multicolumn{4}{l}{\emph{" + ("(i) 6-camera rig, one timestamp, reference CAM\\_FRONT" if key == "rig6"
                     else "(ii) 5 consecutive CAM\\_FRONT keyframes, reference = centre frame") + "}}"])  # fmt: skip
        for k, c in s["abs_rel"].items():
            d1 = s["d1"].get(k)
            rm = s["rmse"].get(k)
            rows.append([ALIGN_LABEL.get(k, tex(k)), ci(c), ci(d1) if d1 else "", ci(rm, 2) if rm else ""])
            A = ALIGN_MACRO.get(k, re.sub("[^A-Za-z]", "", k))
            M(f"B{S}{A}absrel", f(c["mean"]))
            M(f"B{S}{A}absrelCI", ci(c))
            if d1:
                M(f"B{S}{A}dOne", f(d1["mean"]))
            if rm:
                M(f"B{S}{A}rmse", f(rm["mean"], 2))
        rows.append(["MIDRULE"])
        p1 = s["paired_abs_rel_vggt_seq_minus_depthpro_img"]
        p2 = s["paired_abs_rel_vggt_img_minus_depthpro_img"]
        M(f"B{S}pairedSeq", ci(p1))
        M(f"B{S}pairedSeqMean", f(p1["mean"]))
        M(f"B{S}pairedImg", ci(p2))
        M(f"B{S}fracVggtLower", f(100 * p1["frac_a_lower"], 0))
        po = s["pose"]
        M(f"B{S}rotMean", f(po["rot_err_deg_mean"], 2))
        M(f"B{S}rotMedian", f(po["rot_err_deg_median"], 2))
        M(f"B{S}transMean", f(po["trans_dir_err_deg_mean"], 1))
        M(f"B{S}transMedian", f(po["trans_dir_err_deg_median"], 1))
        M(f"B{S}nPairs", str(po["n_pairs"]))
        M(f"B{S}nPairsBaseline", str(po.get("n_pairs_with_baseline", po["n_pairs"])))
        M(f"B{S}nPairsExcluded", str(po.get("n_pairs_excluded_zero_baseline", 0)))
        M(f"B{S}baselineMean", f(po["gt_baseline_m_mean"], 2))
        M(f"B{S}minBaseline", f(po.get("min_baseline_m", 0.0), 1))
        prow.append(["(i) rig" if key == "rig6" else "(ii) temporal", str(po["n_pairs"]), f(po["rot_err_deg_mean"], 2),
                     f(po["rot_err_deg_median"], 2), f(po["trans_dir_err_deg_mean"], 1), f(po["trans_dir_err_deg_median"], 1),
                     str(po.get("n_pairs_excluded_zero_baseline", 0)), f(po["gt_baseline_m_mean"], 2), f(s["scale_seq_median"], 1)])  # fmt: skip
    rows.pop()
    write("track_b.tex", tabular(hdr, rows, "lrrr"))
    write("track_b_pose.tex", tabular(["setting", "pairs", "rot mean", "rot median", "trans mean", "trans median", "excluded",
                                       "baseline [m]", "seq. scale"], prow, "lrrrrrrrr"))  # fmt: skip
    # paired difference table
    rows = []
    for key, s in r["summary"].items():
        p1 = s["paired_abs_rel_vggt_seq_minus_depthpro_img"]
        p2 = s["paired_abs_rel_vggt_img_minus_depthpro_img"]
        rows.append(["(i) rig" if key == "rig6" else "(ii) temporal", ci(p1), f(p1["frac_a_lower"], 2), ci(p2), f(p2["frac_a_lower"], 2)])
    write("track_b_paired.tex", tabular(["setting", "VGGT (seq) minus Depth Pro (img)", "frac VGGT lower",
                                         "VGGT (img) minus Depth Pro (img)", "frac VGGT lower"], rows, "lrrrr"))  # fmt: skip


# ------------------------------------------------------------------ Track C

GS_ORDER = ["random", "lidar", "lidar_depth0.05", "lidar_depth0.2", "random_15k", "lidar_15k", "lidar_depth0.05_15k", "lidar_depth0.2_15k"]
GS_LABEL = {
    "random": "random init",
    "lidar": "LiDAR init",
    "lidar_depth0.05": "LiDAR init + depth (w=0.05)",
    "lidar_depth0.2": "LiDAR init + depth (w=0.2)",
}
GS_MACRO = {"random": "Rand", "lidar": "Lidar", "lidar_depth0.05": "DepthLo", "lidar_depth0.2": "DepthHi"}


def gs_base_and_seed(path: Path) -> tuple[str, int]:
    stem = path.stem[len("gs_"):]
    m = re.search(r"_s(\d+)$", stem)
    if m:
        return stem[: m.start()], int(m.group(1))
    return stem, 0


def track_c() -> None:
    files = sorted(RES.glob("gs_*.json"))
    if not files:
        M("TrackCStatus", "not measured")
        write("track_c.tex", "Track C: not measured (no results/gs\\_*.json).\n")
        write("track_c_per_camera.tex", "")
        return
    M("TrackCStatus", "measured")
    runs: dict[str, dict[int, dict]] = {}
    for p in files:
        base, seed = gs_base_and_seed(p)
        runs.setdefault(base, {})[seed] = json.loads(p.read_text())
    bases = sorted(runs, key=lambda b: GS_ORDER.index(b) if b in GS_ORDER else 99)
    r0 = runs[bases[0]][min(runs[bases[0]])]
    M("GStrainViews", str(r0["n_train_views"]))
    M("GStestViews", str(r0["n_test_views"]))
    M("GSimageW", str(r0["image_size"][0]))
    M("GSimageH", str(r0["image_size"][1]))
    M("GSdownscale", str(r0["downscale"]))
    M("GSholdout", tex(r0["holdout"]))
    multi_seed = any(len(v) > 1 for v in runs.values())
    M("GSseedStatus", "measured" if multi_seed else "not measured")
    seeds_all = sorted({s for v in runs.values() for s in v})
    M("GSnSeeds", str(len(seeds_all)))
    M("GSseedCaption", "The seed columns give mean $\\pm$ sample standard deviation of the per-run means over the seeds that were run (n in brackets)." if multi_seed else "")
    M("GSseedSentence", (f"Seeds {', '.join(str(s) for s in seeds_all)} were run for the variants marked in Table~\\ref{{tab:track_c}}; "
                         "the seed column reports mean and sample standard deviation over seeds.") if multi_seed else
       "Only seed 0 was run; the per-view standard deviations in Table~\\ref{tab:track_c} do not capture seed variance.")
    hdr = ["variant", "steps", "PSNR", "SSIM", "LPIPS", "init", "final", "train min", "ms/view", "PSNR train"]
    if multi_seed:
        hdr.insert(3, r"PSNR, seeds")
        hdr.insert(4, r"LPIPS, seeds")
    rows = []
    by0: dict[str, dict] = {}
    for b in bases:
        r = runs[b][0] if 0 in runs[b] else runs[b][min(runs[b])]
        by0[b] = r
        t = r["test"]
        label = GS_LABEL.get(b.replace("_15k", ""), tex(b))
        steps = f"{r['steps'] // 1000}k"
        row = [label, steps, f"{t['psnr']['mean']:.2f} $\\pm$ {t['psnr']['std']:.2f}", f(t["ssim"]["mean"]),
               f(t["lpips_alex"]["mean"]), f"{r['n_gaussians_init']:,}", f"{r['n_gaussians_final']:,}",
               f(r["train_time_s"] / 60, 1), f(r["render_ms_per_test_view"]["median"], 1), f"{r['train_psnr_subset_mean']:.2f}"]  # fmt: skip
        mac = GS_MACRO.get(b.replace("_15k", ""), re.sub("[^A-Za-z]", "", b)) + ("FifteenK" if b.endswith("_15k") else "")
        M(f"GS{mac}psnr", f"{t['psnr']['mean']:.2f}")
        M(f"GS{mac}psnrStd", f"{t['psnr']['std']:.2f}")
        M(f"GS{mac}ssim", f(t["ssim"]["mean"]))
        M(f"GS{mac}lpips", f(t["lpips_alex"]["mean"]))
        M(f"GS{mac}gaussians", f"{r['n_gaussians_final']:,}")
        M(f"GS{mac}initGaussians", f"{r['n_gaussians_init']:,}")
        M(f"GS{mac}trainMin", f(r["train_time_s"] / 60, 1))
        M(f"GS{mac}psnrTrain", f"{r['train_psnr_subset_mean']:.2f}")
        M(f"GS{mac}renderMs", f(r["render_ms_per_test_view"]["median"], 1))
        M(f"GS{mac}stepsPerS", f(r["steps_per_s"], 1))
        if multi_seed:
            vals = [runs[b][s]["test"]["psnr"]["mean"] for s in sorted(runs[b])]
            cell = f"{np.mean(vals):.2f} $\\pm$ {np.std(vals, ddof=1):.2f} (n={len(vals)})" if len(vals) > 1 else "not measured"
            lv = [runs[b][s]["test"]["lpips_alex"]["mean"] for s in sorted(runs[b])]
            row.insert(3, cell)
            row.insert(4, f"{np.mean(lv):.3f} $\\pm$ {np.std(lv, ddof=1):.3f}" if len(lv) > 1 else "not measured")
            M(f"GS{mac}psnrSeedMean", f"{np.mean(vals):.2f}")
            M(f"GS{mac}psnrSeedStd", f"{np.std(vals, ddof=1):.2f}" if len(vals) > 1 else "not measured")
            M(f"GS{mac}nSeeds", str(len(vals)))
        rows.append(row)
        if b == "lidar_depth0.2":
            rows.append(["MIDRULE"])
    write("track_c.tex", tabular(hdr, rows, "llrrrrrrrr" + ("rr" if multi_seed else "")))

    def delta(a, b, key="psnr"):
        return by0[a]["test"][key]["mean"] - by0[b]["test"][key]["mean"]

    for suf, S in (("", ""), ("_15k", "FifteenK")):
        if f"random{suf}" in by0 and f"lidar{suf}" in by0:
            M(f"GSdeltaLidarRandom{S}", f(delta(f"lidar{suf}", f"random{suf}"), 2))
        for w, mac in (("0.05", "Lo"), ("0.2", "Hi")):
            if f"lidar_depth{w}{suf}" in by0 and f"lidar{suf}" in by0:
                M(f"GSdeltaDepth{mac}Psnr{S}", f(delta(f"lidar_depth{w}{suf}", f"lidar{suf}"), 2))
                M(f"GSdeltaDepth{mac}Lpips{S}", f(delta(f"lidar_depth{w}{suf}", f"lidar{suf}", "lpips_alex"), 3))
    cams = sorted({v["camera"] for v in r0["per_view"]})
    rows = []
    for b in bases:
        r = by0[b]
        row = [GS_LABEL.get(b.replace("_15k", ""), tex(b)) + (" (15k)" if b.endswith("_15k") else " (7k)")]
        for c in cams:
            vals = [v["psnr"] for v in r["per_view"] if v["camera"] == c]
            row.append(f"{sum(vals) / len(vals):.2f}" if vals else "n/a")
        rows.append(row)
    write("track_c_per_camera.tex", tabular(["variant"] + [CAM_SHORT[c] for c in cams], rows, "l" + "r" * len(cams)))


# ------------------------------------------------------------------ Track D


def track_d() -> None:
    r = load("sharp_nvs.json")
    if not r or r.get("status") != "ok":
        M("TrackDStatus", "not measured" if not r else "not run: " + tex(r.get("reason", "")))
        write("track_d.tex", "Track D: not measured.\n")
        return
    M("TrackDStatus", "measured")
    M("SharpImages", str(r["n_images"]))
    M("SharpEvalW", str(r["eval_resolution"][0]))
    M("SharpEvalH", str(r["eval_resolution"][1]))
    M("SharpPredictMs", f(r["predict_ms_mean"], 0))
    M("SharpRenderMs", f(r["render_ms_mean"], 1))
    M("SharpModelLoadSeconds", f(r["model_load_s"], 0))
    ng = [p["n_gaussians"] for p in r["per_image"]]
    M("SharpGaussiansMin", f"{min(ng):,}")
    M("SharpGaussiansMax", f"{max(ng):,}")
    dt_kf = [p["dt_s_next_keyframe"] for p in r["per_image"]]
    M("SharpDtKeyframe", f(float(np.median(dt_kf)), 2))
    LAB = {"identity": "identity (input pose)", "next_sweep": "next 12 Hz sweep", "next_keyframe": "next keyframe"}
    MAC = {"identity": "Id", "next_sweep": "Sweep", "next_keyframe": "Key"}
    rows = []
    for k, s in r["summary"].items():
        rows.append([LAB.get(k, tex(k)), ci(s["psnr"], 2), ci(s["ssim"]), ci(s["lpips_alex"]), f(s["coverage_mean"], 2),
                     f(s.get("translation_m_mean"), 2)])  # fmt: skip
        m = MAC.get(k, re.sub("[^A-Za-z]", "", k))
        M(f"Sharp{m}psnr", f(s["psnr"]["mean"], 2))
        M(f"Sharp{m}psnrCI", ci(s["psnr"], 2))
        M(f"Sharp{m}ssim", f(s["ssim"]["mean"]))
        M(f"Sharp{m}lpips", f(s["lpips_alex"]["mean"]))
        M(f"Sharp{m}lpipsCI", ci(s["lpips_alex"]))
        M(f"Sharp{m}coverage", f(s["coverage_mean"], 2))
        M(f"Sharp{m}motion", f(s.get("translation_m_mean"), 2))
    write("track_d.tex", tabular(["target pose", "PSNR [95\\% CI]", "SSIM [95\\% CI]", "LPIPS [95\\% CI]", "coverage", "motion [m]"],
                                 rows, "lrrrrr"))  # fmt: skip


# ------------------------------------------------------------------ Track E

E_LABEL = {"fm_sample": "flow matching, one Euler sample", "regression": "regression UNet (L2)", "copy_current": "copy current frame"}
E_MACRO = {"fm_sample": "Fm", "regression": "Reg", "copy_current": "Copy"}


def track_e() -> None:
    d = load("nextframe_data.json")
    if d:
        M("NFframes", f"{d['n_frames']:,}")
        M("NFkeyframes", str(d["n_keyframes"]))
        M("NFw", str(d["size_wh"][0]))
        M("NFh", str(d["size_wh"][1]))
        M("NFtrainPairs", f"{d['n_train_pairs']:,}")
        M("NFtestPairs", f"{d['n_test_pairs']:,}")
        M("NFdtMedian", f(d["pair_dt_s_median"], 2))
        M("NFholdoutScenes", ", ".join(d["holdout_scenes"]))
    for mode, mac in (("fm", "Fm"), ("regression", "Reg")):
        t = load(f"nextframe_train_{mode}.json")
        if t:
            M(f"NF{mac}params", f"{t['n_params'] / 1e6:.1f}")
            M(f"NF{mac}steps", f"{t['steps']:,}")
            M(f"NF{mac}stepsRequested", f"{t['steps_requested']:,}")
            M(f"NF{mac}batch", str(t["batch"]))
            M(f"NF{mac}lr", str(t["lr"]))
            M(f"NF{mac}trainMin", f(t["train_time_s"] / 60, 1))
            M(f"NF{mac}maxMinutes", f(t["max_minutes"], 0))
            M(f"NF{mac}stepsPerS", f(t["steps_per_s"], 1))
            M(f"NF{mac}finalLoss", f(t["curve"][-1][1], 4))
    dd = load("ddp_smoke.json")
    M("DDPsteps", str(dd["steps"]) if dd else "not measured")
    M("DDPworld", str(dd["world_size"]) if dd else "not measured")
    files = sorted(RES.glob("synthesis_nextframe*.json"))
    if not files:
        M("TrackEStatus", "not measured")
        write("track_e.tex", "Track E: not measured.\n")
        return
    M("TrackEStatus", "measured")
    runs: dict[int, dict] = {}
    for p in files:
        m = re.search(r"_s(\d+)$", p.stem)
        runs[int(m.group(1)) if m else 0] = json.loads(p.read_text())
    r = runs[0] if 0 in runs else runs[min(runs)]
    multi = len(runs) > 1
    M("NFseedStatus", "measured" if multi else "not measured")
    M("NFnSeeds", str(len(runs)))
    M("NFseedCaption", "The seed columns give mean $\\pm$ sample standard deviation of the per-run means over seeds." if multi else "")
    M("NFseedSentence", (f"Seeds {', '.join(str(s) for s in sorted(runs))} (training and sampling seed) were run; "
                         "Table~\\ref{tab:track_e} adds mean and sample standard deviation over seeds.") if multi else
       "One seed (training and sampling) was run; the bootstrap intervals are over test pairs and do not capture seed variance.")
    M("NFsampler", tex(r["sampler"]))
    M("NFsamplingMs", f(r["sampling_ms_per_frame"], 1))
    hdr = ["method", "PSNR [95\\% CI]", "LPIPS [95\\% CI]"]
    if multi:
        hdr += [r"PSNR mean $\pm$ std (seeds)", r"LPIPS mean $\pm$ std (seeds)"]
    rows = []
    for k, m in r["methods"].items():
        row = [E_LABEL.get(k, tex(k)), ci(m["psnr"], 2), ci(m["lpips_alex"])]
        mac = E_MACRO.get(k, re.sub("[^A-Za-z]", "", k))
        M(f"NF{mac}psnr", f(m["psnr"]["mean"], 2))
        M(f"NF{mac}psnrCI", ci(m["psnr"], 2))
        M(f"NF{mac}lpips", f(m["lpips_alex"]["mean"]))
        M(f"NF{mac}lpipsCI", ci(m["lpips_alex"]))
        if multi:
            ps = [v["methods"][k]["psnr"]["mean"] for v in runs.values() if k in v["methods"]]
            ls = [v["methods"][k]["lpips_alex"]["mean"] for v in runs.values() if k in v["methods"]]
            row += [f"{np.mean(ps):.2f} $\\pm$ {np.std(ps, ddof=1):.2f} (n={len(ps)})", f"{np.mean(ls):.3f} $\\pm$ {np.std(ls, ddof=1):.3f}"]
            M(f"NF{mac}psnrSeedMean", f"{np.mean(ps):.2f}")
            M(f"NF{mac}psnrSeedStd", f"{np.std(ps, ddof=1):.2f}")
            M(f"NF{mac}lpipsSeedMean", f"{np.mean(ls):.3f}")
            M(f"NF{mac}lpipsSeedStd", f"{np.std(ls, ddof=1):.3f}")
        rows.append(row)
    write("track_e.tex", tabular(hdr, rows, "lrr" + ("rr" if multi else "")))
    if "paired_lpips_fm_minus_regression" in r:
        M("NFpairedLpipsFmReg", ci(r["paired_lpips_fm_minus_regression"]))
        M("NFpairedLpipsFmRegMean", f(r["paired_lpips_fm_minus_regression"]["mean"]))
        M("NFfracFmLowerReg", f(100 * r["paired_lpips_fm_minus_regression"]["frac_a_lower"], 0))
    else:
        M("NFpairedLpipsFmReg", "not measured")
        M("NFpairedLpipsFmRegMean", "n/a")
        M("NFfracFmLowerReg", "n/a")
    M("NFpairedLpipsFmCopy", ci(r["paired_lpips_fm_minus_copy"]))
    M("NFpairedPsnrFmCopy", ci(r["paired_psnr_fm_minus_copy"], 2))
    fid = r.get("fid_unconditional", {})
    if fid.get("value") is not None:
        M("NFfid", f(fid["value"], 1))
        M("NFfidN", str(fid["n_generated"]))
    else:
        M("NFfid", "not measured")
        M("NFfidN", "n/a")


# ------------------------------------------------------------------ runtimes


def runtimes() -> None:
    rows = []
    gt = load("lidar_gt_stats.json")
    if gt:
        rows.append(["LiDAR GT projection (CPU)", f"{gt['n_images']:,} images", f(gt["wall_time_s"] / 60, 1)])
    for p in sorted(RES.glob("mono_depth_*.json")):
        r = json.loads(p.read_text())
        if r["variant"].endswith("_gtfocal"):
            continue
        rows.append([f"Track A {VARIANT_LABEL.get(r['variant'], tex(r['variant']))}",
                     f"{r['n_images']:,} images, {f(r['timing_ms_per_image']['median'], 0)} ms/img median", f(r["wall_time_s"] / 60, 1)])  # fmt: skip
    extra_n, extra_min = 0, 0.0
    budgets: dict[int, list[dict]] = {}
    for p in sorted(RES.glob("gs_*.json")):
        r = json.loads(p.read_text())
        _, seed = gs_base_and_seed(p)
        if seed:
            extra_n += 1
            extra_min += (r["train_time_s"] + r["init_time_s"]) / 60
        else:
            budgets.setdefault(r["steps"], []).append(r)
    for steps in sorted(budgets):
        g = budgets[steps]
        sps = [r["steps_per_s"] for r in g]
        tot = sum((r["train_time_s"] + r["init_time_s"]) / 60 for r in g)
        rows.append([f"Track C 3DGS, {len(g)} variants (seed 0)", f"{steps:,} steps each, {f(min(sps), 1)} to {f(max(sps), 1)} steps/s",
                     f"{f(tot, 1)} (sum)"])  # fmt: skip
    if extra_n:
        rows.append(["Track C 3DGS, additional seeds", f"{extra_n} run" + ("s" if extra_n != 1 else ""), f"{f(extra_min, 1)} (sum)"])
    mv = load("multiview_vggt.json")
    if mv:
        n = sum(s["n_sequences"] for s in mv["summary"].values())
        tot = sum(s["n_sequences"] * s["timing_ms_per_sequence"]["mean"] for s in mv["summary"].values()) / 1000
        rows.append(["Track B VGGT (+ Depth Pro on reference views)", f"{n} sequences", f((tot + mv["model_load_s"]) / 60, 1)])
    e_n, e_min = 0, 0.0
    for tname in sorted(p.name for p in RES.glob("nextframe_train_*.json")):
        t = load(tname)
        mode = tname[len("nextframe_train_"):-len(".json")]
        if re.search(r"_s\d+$", mode):
            e_n += 1
            e_min += t["train_time_s"] / 60
            continue
        rows.append([f"Track E train {tex(mode)}", f"{t['steps']:,} steps, {f(t['steps_per_s'], 1)} steps/s (GPU shared with the other training)",
                     f(t["train_time_s"] / 60, 1)])  # fmt: skip
    if e_n:
        rows.append(["Track E training, additional seeds", f"{e_n} run" + ("s" if e_n != 1 else ""), f"{f(e_min, 1)} (sum)"])
    nf = load("synthesis_nextframe.json")
    if nf:
        rows.append(["Track E eval (sampling only)", f"{nf['n_test_pairs']} pairs, {f(nf['sampling_ms_per_frame'], 1)} ms/sample",
                     f(nf["sampling_ms_per_frame"] * nf["n_test_pairs"] / 60000, 1)])  # fmt: skip
    dd = load("ddp_smoke.json")
    if dd:
        rows.append(["Track E DDP smoke (torchrun, gloo, 2 ranks on one GPU)", f"{dd['steps']} steps, world size {dd['world_size']}", f(dd["train_time_s"] / 60, 2)])
    sh = load("sharp_nvs.json")
    if sh and sh.get("status") == "ok":
        rows.append(["Track D SHARP", f"{sh['n_images']} images, {f(sh['predict_ms_mean'], 0)} ms predict, {f(sh['render_ms_mean'], 1)} ms render per view", "n/a"])
    else:
        rows.append(["Track D SHARP", "not measured", "n/a"])
    write("runtimes.tex", tabular(["job", "size", f"minutes ({GPU})"], rows, "llr"))


def camera_intrinsics() -> None:
    r = load("camera_intrinsics.json")
    if not r:
        M("PPsentence", "the nuScenes principal points are off-centre (offset not measured)")
        return
    o = r["overall"]
    M(
        "PPsentence",
        f"the calibrated nuScenes principal points are on average {o['abs_dx_mean_px']:.0f} px (horizontal) and "
        f"{o['abs_dy_mean_px']:.0f} px (vertical) away from the centre of the 1600x900 images, at most "
        f"{max(o['abs_dx_max_px'], o['abs_dy_max_px']):.0f} px",
    )


def main() -> None:
    camera_intrinsics()
    gt_stats()
    scene_dynamics()
    track_a()
    track_b()
    track_c()
    track_d()
    track_e()
    runtimes()
    M("GPUname", GPU)
    lines = ["% generated by paper/make_tables.py from results/*.json; do not edit"]
    for k in sorted(MACROS):
        lines.append(f"\\newcommand{{\\{k}}}{{{MACROS[k]}}}")
    NUM.write_text("\n".join(lines) + "\n")
    for p in list(TAB.glob("*.tex")) + [NUM]:
        if any(ch in p.read_text() for ch in "–—"):
            raise SystemExit(f"long dash found in {p}")
    print(f"wrote {len(list(TAB.glob('*.tex')))} tables and {len(MACROS)} macros")


if __name__ == "__main__":
    main()
