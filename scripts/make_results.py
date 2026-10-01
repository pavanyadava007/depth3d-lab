"""Generate docs/RESULTS.md (and the results block of README.md) from results/*.json.

Every number in the documentation comes from here. Missing files produce "not measured" lines.
Run: python scripts/make_results.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

RES = Path("results")
DOC = Path("docs/RESULTS.md")
README = Path("README.md")
GPU = "NVIDIA L4"


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
    if isinstance(x, int):
        return f"{x:,}"
    if x != x:  # nan
        return "nan"
    return f"{x:.{nd}f}"


def ci(c: dict | None, nd=3) -> str:
    if not c or c.get("n", 0) == 0:
        return "n/a"
    return f"{c['mean']:.{nd}f} [{c['lo']:.{nd}f}, {c['hi']:.{nd}f}]"


SEED_RE = re.compile(r"^(.*)_s(\d+)$")


def split_seed(name: str) -> tuple[str, int]:
    """'lidar_depth0.2_s1' -> ('lidar_depth0.2', 1); names without suffix are seed 0 files."""
    m = SEED_RE.match(name)
    return (m.group(1), int(m.group(2))) if m else (name, 0)


def mean_std(vals: list[float], nd=3) -> str:
    """mean +- sample std (ddof=1) over seeds; a single value has no std."""
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:,.{nd}f}"
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    return f"{m:,.{nd}f} +- {sd:,.{nd}f}"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


METRIC_COLS = ["abs_rel", "sq_rel", "rmse", "rmse_log", "d1", "d2", "d3"]
METRIC_HDR = ["AbsRel", "SqRel", "RMSE [m]", "RMSE_log", "d<1.25", "d<1.25^2", "d<1.25^3"]


def metric_row(label: str, m: dict | None) -> list[str]:
    if not m or not m.get("n_points"):
        return [label] + ["n/a"] * len(METRIC_COLS)
    return [label] + [f(m[k]) for k in METRIC_COLS]


# ------------------------------------------------------------------ Track A


def track_a() -> tuple[str, list[list[str]]]:
    files = sorted(RES.glob("mono_depth_*.json"))
    lines = ["## Track A - monocular metric depth vs LiDAR (zero-shot)", ""]
    headline: list[list[str]] = []
    gt = load("lidar_gt_stats.json")
    if gt:
        lines.append(
            f"LiDAR ground truth: {gt['n_images']:,} images ({gt['n_keyframes']} keyframes x 6 cameras), points with "
            f"1 m < z < 80 m after z-buffering. Per camera (points per image, median depth):"
        )
        lines.append("")
        rows = [
            [c, f(v["points_per_image"], 0), f(v["median_depth_m"], 1), f(v["frac_1_20m"], 2), f(v["frac_20_40m"], 2),
             f(v["frac_40_80m"], 2)]
            for c, v in gt["per_camera"].items()
        ]  # fmt: skip
        lines.append(table(["camera", "pts/img", "median z [m]", "frac 1-20 m", "frac 20-40 m", "frac 40-80 m"], rows))
        lines.append("")
    if not files:
        lines.append("Not measured: no results/mono_depth_*.json found.")
        return "\n".join(lines), headline
    results = [json.loads(p.read_text()) for p in files]
    order = ["depth_pro", "depth_pro_gtfocal", "dav2_metric_outdoor_large", "moge2", "moge2_gtfocal"]
    results.sort(key=lambda r: order.index(r["variant"]) if r["variant"] in order else 99)
    lines.append("### Overall (all LiDAR points, all cameras)")
    lines.append("")
    rows = []
    for r in results:
        n = r["n_images"]
        rows.append(metric_row(f"{r['variant']} raw", r["raw"]) + [f"{n:,}", f(r['timing_ms_per_image']['median'], 0)])
        rows.append(metric_row(f"{r['variant']} median-scaled", r["scaled"]) + [f"{n:,}", ""])
        headline.append(
            [r["variant"], f(r["raw"]["abs_rel"]), f(r["scaled"]["abs_rel"]), f(r["raw"]["d1"]), f(r["scaled"]["d1"]),
             f(r["timing_ms_per_image"]["median"], 0)]
        )  # fmt: skip
    lines.append(table(["variant", *METRIC_HDR, "images", f"ms/img ({GPU})"], rows))
    lines.append("")
    lines.append(
        "'raw' is the metric output as predicted; 'median-scaled' multiplies each image by median(GT)/median(pred) "
        "so that the remaining error is shape error only. The '_gtfocal' variants (Depth Pro, MoGe-2) feed the true nuScenes "
        "focal length instead of the model's own focal estimate and come from the same forward pass as the own-focal "
        "variant. Timing is the median model time per 1600x900 image."
    )
    lines.append("")
    lines.append("### Scale factor and focal estimate per camera")
    lines.append("")
    rows = []
    for r in results:
        for cam, s in r["scale_factor_by_camera"].items():
            foc = r.get("focal_by_camera", {}).get(cam)
            rows.append(
                [r["variant"], cam, f(s["median"]), f"{f(s['p10'])} / {f(s['p90'])}",
                 f(foc["gt_focal_px"], 0) if foc else "n/a",
                 f(foc["pred_focal_px_median"], 0) if foc else "n/a",
                 f(foc["median_ratio_pred_over_gt"]) if foc else "n/a"]
            )  # fmt: skip
    lines.append(table(["variant", "camera", "median scale GT/pred", "p10 / p90", "true f [px]", "pred f median [px]",
                        "pred/true f"], rows))  # fmt: skip
    lines.append("")
    lines.append("A scale factor of 1.0 means the metric output is already on the LiDAR scale for that camera.")
    lines.append("")
    lines.append("### Per camera group (front / side / back) - AbsRel and d<1.25")
    lines.append("")
    rows = []
    for r in results:
        for g in ("front", "side", "back"):
            ra, sc = r["per_group_raw"][g], r["per_group_scaled"][g]
            rows.append([r["variant"], g, f(ra["abs_rel"]), f(sc["abs_rel"]), f(ra["d1"]), f(sc["d1"]), f"{ra['n_points']:,}"])
    lines.append(table(["variant", "group", "AbsRel raw", "AbsRel scaled", "d1 raw", "d1 scaled", "points"], rows))
    lines.append("")
    lines.append("### Per camera - AbsRel raw / scaled")
    lines.append("")
    cams = list(results[0]["per_camera_raw"].keys())
    rows = []
    for r in results:
        rows.append([r["variant"]] + [f"{f(r['per_camera_raw'][c]['abs_rel'])} / {f(r['per_camera_scaled'][c]['abs_rel'])}" for c in cams])
    lines.append(table(["variant", *cams], rows))
    lines.append("")
    lines.append("### Per depth bin - AbsRel and RMSE")
    lines.append("")
    rows = []
    for r in results:
        for b, ra in r["per_bin_raw"].items():
            sc = r["per_bin_scaled"][b]
            rows.append([r["variant"], b, f(ra["abs_rel"]), f(sc["abs_rel"]), f(ra["rmse"], 2), f(sc["rmse"], 2), f"{ra['n_points']:,}"])
    lines.append(table(["variant", "bin", "AbsRel raw", "AbsRel scaled", "RMSE raw [m]", "RMSE scaled [m]", "points"], rows))
    lines.append("")
    lines.append("### Reading the scale error (generated from the tables above)")
    lines.append("")
    for r in results:
        raw, sc = r["raw"]["abs_rel"], r["scaled"]["abs_rel"]
        share = (raw - sc) / raw if raw else float("nan")
        g = r["per_group_raw"]
        worst = max(g, key=lambda k: g[k]["abs_rel"])
        best = min(g, key=lambda k: g[k]["abs_rel"])
        sb = r["scale_factor_by_camera"].get("CAM_BACK", {}).get("median")
        sf = r["scale_factor_by_camera"].get("CAM_FRONT", {}).get("median")
        far = r["raw"].get("frac_pred_above_max")
        txt = (f"{r['variant']}: per-image median scaling lowers AbsRel from {f(raw)} to {f(sc)}, i.e. {f(100 * share, 0)}% of the "
               f"raw error is a per-image scale error and the rest is shape error. Worst camera group raw: {worst} "
               f"(AbsRel {f(g[worst]['abs_rel'])}), best: {best} ({f(g[best]['abs_rel'])}). Median scale factor CAM_FRONT {f(sf, 2)} vs CAM_BACK {f(sb, 2)}.")
        foc = r.get("focal_by_camera", {})
        if foc.get("CAM_BACK") and foc.get("CAM_FRONT"):
            txt += (f" Focal estimate / true focal: CAM_FRONT {f(foc['CAM_FRONT']['median_ratio_pred_over_gt'], 2)}, "
                    f"CAM_BACK {f(foc['CAM_BACK']['median_ratio_pred_over_gt'], 2)}.")
        if far is not None:
            txt += f" Fraction of LiDAR pixels where the raw prediction exceeded 80 m (clipped before scoring): {f(100 * far, 2)}%."
        lines.append(txt)
        lines.append("")
    lines.append("### Per-image AbsRel, mean with 95% bootstrap CI over images")
    lines.append("")
    rows = [[r["variant"], ci(r["per_image_abs_rel_raw_ci"]), ci(r["per_image_abs_rel_scaled_ci"])] for r in results]
    lines.append(table(["variant", "raw", "median-scaled"], rows))
    lines.append("")
    return "\n".join(lines), headline


# ------------------------------------------------------------------ Track B


def track_b() -> tuple[str, list[list[str]]]:
    lines = ["## Track B - feed-forward multi-view reconstruction (VGGT) vs LiDAR", ""]
    r = load("multiview_vggt.json")
    headline: list[list[str]] = []
    if not r:
        lines.append("Not measured: results/multiview_vggt.json missing.")
        return "\n".join(lines), headline
    lines.append(f"Model: {r['model']}. Mono reference: {r['mono_reference']}. GPU: {r['gpu']}.")
    lines.append("")
    names = {"rig6": "(i) 6-camera rig, one timestamp, reference CAM_FRONT",
             "temporal5": "(ii) 5 consecutive CAM_FRONT keyframes, reference = centre frame"}  # fmt: skip
    for key, s in r["summary"].items():
        lines.append(f"### Setting {names.get(key, key)}")
        lines.append("")
        lines.append(f"{s['n_sequences']} sequences, {s['n_ref_points_total']:,} LiDAR points on the reference views, "
                     f"{f(s['timing_ms_per_sequence']['mean'], 0)} ms per sequence ({GPU}, incl. Depth Pro and metrics).")  # fmt: skip
        lines.append("")
        rows = []
        for k, c in s["abs_rel"].items():
            d1 = s["d1"].get(k)
            rm = s["rmse"].get(k)
            rows.append([k, ci(c), ci(d1) if d1 else "", ci(rm, 2) if rm else ""])
        lines.append(table(["alignment / model", "AbsRel mean [95% CI]", "d<1.25", "RMSE [m]"], rows))
        lines.append("")
        p1 = s["paired_abs_rel_vggt_seq_minus_depthpro_img"]
        p2 = s["paired_abs_rel_vggt_img_minus_depthpro_img"]
        lines.append(
            f"Paired difference on identical pixels, VGGT (one scale per sequence) minus Depth Pro (per-image scale): "
            f"{ci(p1)}; VGGT with per-image scale minus Depth Pro: {ci(p2)}. Negative means VGGT is better. "
            f"Fraction of sequences where VGGT (seq scale) is lower: {f(p1['frac_a_lower'], 2)}."
        )
        lines.append("")
        po = s["pose"]
        lines.append(
            f"Relative camera pose vs nuScenes GT over {po['n_pairs']} view pairs: rotation error mean {f(po['rot_err_deg_mean'], 2)} deg "
            f"(median {f(po['rot_err_deg_median'], 2)}), translation direction error mean {f(po['trans_dir_err_deg_mean'], 1)} deg "
            f"(median {f(po['trans_dir_err_deg_median'], 1)}, over the {po.get('n_pairs_with_baseline', po['n_pairs'])} pairs whose GT "
            f"baseline exceeds {f(po.get('min_baseline_m', 0.0), 1)} m; {po.get('n_pairs_excluded_zero_baseline', 0)} pairs with a "
            f"stationary ego vehicle have no defined direction and are excluded); mean GT baseline {f(po['gt_baseline_m_mean'], 2)} m. "
            f"Median sequence scale factor (LiDAR / VGGT units): {f(s['scale_seq_median'], 2)}."
        )
        lines.append("")
        headline.append([key, ci(s["abs_rel"]["vggt_seq_scaled"]), ci(s["abs_rel"]["depth_pro_img_scaled"]),
                         f(po["rot_err_deg_median"], 2), f(po["trans_dir_err_deg_median"], 1)])  # fmt: skip
    return "\n".join(lines), headline


# ------------------------------------------------------------------ Track C


def track_c() -> tuple[str, list[list[str]]]:
    lines = ["## Track C - 3D Gaussian Splatting on one driving scene (gsplat, own training loop)", ""]
    dyn = load("scene_dynamics.json")
    headline: list[list[str]] = []
    if dyn:
        lines.append(f"Scene choice: moving annotations (box velocity > {dyn['velocity_threshold_mps']} m/s) per scene; "
                     f"selected {dyn['selected_scene']}.")  # fmt: skip
        lines.append("")
        rows = [[s, str(d["n_keyframes"]), f"{d['n_annotations']:,}", f"{d['n_moving_annotations_vel']:,}",
                 f(d["frac_moving_vel"], 2), f"{d['n_moving_annotations_attr']:,}"] for s, d in dyn["scenes"].items()]  # fmt: skip
        rows.sort(key=lambda r: int(r[3].replace(",", "")))
        lines.append(table(["scene", "keyframes", "annotations", "moving (velocity)", "frac moving", "moving (attribute)"], rows))
        lines.append("")
    files = sorted(RES.glob("gs_*.json"))
    if not files:
        lines.append("Not measured: no results/gs_*.json found.")
        return "\n".join(lines), headline
    all_rs = [json.loads(p.read_text()) for p in files]
    order = ["random", "lidar", "lidar_depth0.05", "lidar_depth0.2", "random_15k", "lidar_15k", "lidar_depth0.05_15k", "lidar_depth0.2_15k"]
    rs = [r for r in all_rs if split_seed(r["name"])[1] == 0]
    seed_rs = [r for r in all_rs if split_seed(r["name"])[1] != 0]
    rs.sort(key=lambda r: order.index(r["name"]) if r["name"] in order else 99)
    r0 = rs[0]
    lines.append(
        f"Scene {r0['scene']}, {r0['n_train_views']} train / {r0['n_test_views']} held-out views ({r0['holdout']}), "
        f"images {r0['image_size'][0]}x{r0['image_size'][1]} (downscale {r0['downscale']}), {r0['steps']} steps, seed {r0['seed']}, {GPU}."
    )
    lines.append("")
    rows = []
    for r in rs:
        t = r["test"]
        rows.append([
            r["name"], f(r["depth_weight"], 2) if r["variant"] == "lidar_depth" else "-",
            f"{t['psnr']['mean']:.2f} +- {t['psnr']['std']:.2f}", f(t["ssim"]["mean"]), f(t["lpips_alex"]["mean"]),
            f"{r['n_gaussians_init']:,}", f"{r['n_gaussians_final']:,}", f(r["train_time_s"] / 60, 1),
            f(r["render_ms_per_test_view"]["median"], 1), f"{r['train_psnr_subset_mean']:.2f}",
        ])  # fmt: skip
        headline.append([r["name"], f"{t['psnr']['mean']:.2f}", f(t["ssim"]["mean"]), f(t["lpips_alex"]["mean"]),
                         f"{r['n_gaussians_final']:,}", f(r["train_time_s"] / 60, 1)])  # fmt: skip
    lines.append(table(["variant", "depth w", "PSNR test", "SSIM", "LPIPS (alex)", "init gaussians", "final gaussians",
                        f"train min ({GPU})", "render ms/view", "PSNR train subset"], rows))  # fmt: skip
    lines.append("")
    by = {r["name"]: r for r in rs}
    if "random" in by and "lidar" in by:
        d = by["lidar"]["test"]["psnr"]["mean"] - by["random"]["test"]["psnr"]["mean"]
        lines.append(f"LiDAR init vs random init: {f(d, 2)} dB PSNR on held-out views (positive favours LiDAR init).")
    for w in ("lidar_depth0.05", "lidar_depth0.2"):
        if w in by and "lidar" in by:
            d = by[w]["test"]["psnr"]["mean"] - by["lidar"]["test"]["psnr"]["mean"]
            dl = by[w]["test"]["lpips_alex"]["mean"] - by["lidar"]["test"]["lpips_alex"]["mean"]
            lines.append(f"{w} vs lidar: {f(d, 2)} dB PSNR, {f(dl)} LPIPS (negative LPIPS favours depth supervision).")
    lines.append("")
    seed_head, seed_lines = track_c_seeds(rs, seed_rs, order)
    lines += seed_lines
    lines.append("### Held-out PSNR per camera")
    lines.append("")
    cams = sorted({v["camera"] for v in r0["per_view"]})
    rows = []
    for r in rs:
        row = [r["name"]]
        for c in cams:
            vals = [v["psnr"] for v in r["per_view"] if v["camera"] == c]
            row.append(f"{sum(vals) / len(vals):.2f}" if vals else "n/a")
        rows.append(row)
    lines.append(table(["variant", *cams], rows))
    lines.append("")
    return "\n".join(lines), {"seed0": headline, "seeds": seed_head}


def track_c_seeds(rs: list[dict], seed_rs: list[dict], order: list[str]) -> tuple[list[list[str]], list[str]]:
    """Mean +- std over seeds (seed 0 file + gs_<name>_s<k>.json) per variant; empty when no seed files exist."""
    if not seed_rs:
        return [], []
    groups: dict[str, list[dict]] = {}
    for r in rs + seed_rs:
        base, _ = split_seed(r["name"])
        groups.setdefault(base, []).append(r)
    bases = [b for b in order if b in groups and len(groups[b]) > 1]
    rows, head = [], []
    for b in bases:
        g = sorted(groups[b], key=lambda r: r["seed"])
        seeds = ",".join(str(r["seed"]) for r in g)
        ps = mean_std([r["test"]["psnr"]["mean"] for r in g], 2)
        ss = mean_std([r["test"]["ssim"]["mean"] for r in g])
        lp = mean_std([r["test"]["lpips_alex"]["mean"] for r in g])
        ng = mean_std([float(r["n_gaussians_final"]) for r in g], 0)
        tm = mean_std([r["train_time_s"] / 60 for r in g], 1)
        rows.append([b, seeds, ps, ss, lp, ng, tm])
        head.append([b, seeds, ps, ss, lp])
    lines = ["### Seed variance (7k-step variants, seeds as listed; mean +- sample std over seeds)", "",
             table(["variant", "seeds", "PSNR test", "SSIM", "LPIPS (alex)", "final gaussians", f"train min ({GPU})"], rows), ""]
    # per-seed ordering of the three claims in the README
    for a, b in (("lidar", "random"), ("lidar_depth0.05", "lidar"), ("lidar_depth0.2", "lidar_depth0.05")):
        if a in groups and b in groups:
            ga = {r["seed"]: r for r in groups[a]}
            gb = {r["seed"]: r for r in groups[b]}
            common = sorted(set(ga) & set(gb))
            if len(common) > 1:
                dp = [ga[k]["test"]["psnr"]["mean"] - gb[k]["test"]["psnr"]["mean"] for k in common]
                dl = [ga[k]["test"]["lpips_alex"]["mean"] - gb[k]["test"]["lpips_alex"]["mean"] for k in common]
                ds = [ga[k]["test"]["ssim"]["mean"] - gb[k]["test"]["ssim"]["mean"] for k in common]
                lines.append(
                    f"{a} minus {b}, per seed ({', '.join(str(k) for k in common)}): PSNR "
                    f"{', '.join(f'{d:+.2f}' for d in dp)} dB ({sum(d > 0 for d in dp)}/{len(dp)} seeds positive); SSIM "
                    f"{', '.join(f'{d:+.3f}' for d in ds)} ({sum(d > 0 for d in ds)}/{len(ds)} positive); LPIPS "
                    f"{', '.join(f'{d:+.3f}' for d in dl)} ({sum(d < 0 for d in dl)}/{len(dl)} seeds negative, i.e. better)."
                )
                lines.append("")
    return head, lines


# ------------------------------------------------------------------ Track D


def track_d() -> tuple[str, list[list[str]]]:
    lines = ["## Track D - single-image 3D Gaussians (Apple SHARP) rendered at the true next pose", ""]
    r = load("sharp_nvs.json")
    headline: list[list[str]] = []
    if not r:
        lines.append("Not measured: results/sharp_nvs.json missing (see STATUS.md for the reason).")
        return "\n".join(lines), headline
    if r.get("status") != "ok":
        lines.append(f"Not run: {r.get('reason', 'unknown')}")
        return "\n".join(lines), headline
    lines.append(f"{r['n_images']} CAM_FRONT keyframes (one per scene); Gaussians predicted from frame t, rendered at the "
                 f"identity pose and at the true relative pose of the next frame (nuScenes ego pose + calibration). "
                 f"Metrics only on pixels the render covers. GPU: {GPU}.")  # fmt: skip
    lines.append("")
    rows = []
    for k, s in r["summary"].items():
        rows.append([k, ci(s["psnr"], 2), ci(s["ssim"]), ci(s["lpips_alex"]), f(s["coverage_mean"], 2),
                     f(s.get("translation_m_mean"), 2)])  # fmt: skip
        headline.append([k, ci(s["psnr"], 2), ci(s["lpips_alex"]), f(s.get("translation_m_mean"), 2)])
    lines.append(table(["target", "PSNR [95% CI]", "SSIM", "LPIPS (alex)", "coverage", "camera motion [m]"], rows))
    lines.append("")
    lines.append(f"Prediction time {f(r['predict_ms_mean'], 0)} ms per image, render {f(r['render_ms_mean'], 1)} ms per view ({GPU}).")
    lines.append("")
    return "\n".join(lines), headline


# ------------------------------------------------------------------ Track E


def track_e() -> tuple[str, list[list[str]]]:
    lines = ["## Track E - next-frame synthesis with conditional flow matching (plain PyTorch)", ""]
    d = load("nextframe_data.json")
    headline: list[list[str]] = []
    if d:
        lines.append(f"Data: {d['n_frames']:,} CAM_FRONT frames (keyframes + sweeps, {d['n_keyframes']} keyframes) at "
                     f"{d['size_wh'][0]}x{d['size_wh'][1]}; {d['n_train_pairs']:,} train / {d['n_test_pairs']:,} test "
                     f"consecutive pairs (median dt {f(d['pair_dt_s_median'], 2)} s); held-out scenes {', '.join(d['holdout_scenes'])}.")  # fmt: skip
        lines.append("")
    for mode in ("fm", "regression"):
        t = load(f"nextframe_train_{mode}.json")
        if t:
            lines.append(f"Training {mode} (seed {t.get('seed', 0)}): {t['n_params'] / 1e6:.1f}M-parameter UNet, {t['steps']:,} steps, "
                         f"batch {t['batch']}, lr {t['lr']}, {f(t['train_time_s'] / 60, 1)} min on {GPU} (world size {t['world_size']}; "
                         f"the two trainings shared the GPU), final logged loss {f(t['curve'][-1][1], 4)}.")  # fmt: skip
            lines.append("")
    r = load("synthesis_nextframe.json")
    if not r:
        lines.append("Not measured: results/synthesis_nextframe.json missing.")
        return "\n".join(lines), {"seed0": headline, "seeds": []}
    lines.append(f"Evaluation on {r['n_test_pairs']} held-out pairs, sampler {r['sampler']}, "
                 f"{f(r['sampling_ms_per_frame'], 1)} ms per sampled frame ({GPU}).")  # fmt: skip
    lines.append("")
    rows = []
    for k, m in r["methods"].items():
        rows.append([k, ci(m["psnr"], 2), ci(m["lpips_alex"])])
        headline.append([k, ci(m["psnr"], 2), ci(m["lpips_alex"])])
    lines.append(table(["method", "PSNR vs real next frame [95% CI]", "LPIPS (alex) [95% CI]"], rows))
    lines.append("")
    if "paired_lpips_fm_minus_regression" in r:
        lines.append(f"Paired LPIPS, flow-matching sample minus regression UNet: {ci(r['paired_lpips_fm_minus_regression'])} "
                     f"(negative = sample is perceptually closer).")  # fmt: skip
    lines.append(f"Paired LPIPS, sample minus copy-current-frame: {ci(r['paired_lpips_fm_minus_copy'])}; paired PSNR: "
                 f"{ci(r['paired_psnr_fm_minus_copy'], 2)} dB.")  # fmt: skip
    fid = r.get("fid_unconditional", {})
    if fid.get("value") is not None:
        lines.append(f"FID of unconditional samples vs held-out frames: {f(fid['value'], 1)} ({fid['n_generated']} generated, "
                     f"{fid['n_real']} real). {fid['note']}")  # fmt: skip
    else:
        lines.append(f"FID: {fid.get('note', 'not measured')}.")
    lines.append("")
    seed_head, seed_lines = track_e_seeds(r)
    lines += seed_lines
    return "\n".join(lines), {"seed0": headline, "seeds": seed_head}


def track_e_seeds(r0: dict) -> tuple[list[list[str]], list[str]]:
    """Mean +- std over seeds of PSNR / LPIPS per method from synthesis_nextframe_s<k>.json; empty without seed files."""
    seed_files = sorted(RES.glob("synthesis_nextframe_s*.json"))
    if not seed_files:
        return [], []
    runs = [(0, r0)] + [(int(p.stem.rsplit("_s", 1)[1]), json.loads(p.read_text())) for p in seed_files]
    runs.sort(key=lambda x: x[0])
    seeds = [k for k, _ in runs]
    lines = [f"### Seed variance (seeds {', '.join(str(k) for k in seeds)}: training seed and sampling seed; mean +- sample std over seeds)", ""]
    for k in seeds:
        parts = []
        for mode in ("fm", "regression"):
            t = load(f"nextframe_train_{mode}.json" if k == 0 else f"nextframe_train_{mode}_s{k}.json")
            if t:
                parts.append(f"{mode} {t['steps']:,} steps in {f(t['train_time_s'] / 60, 1)} min")
        if parts:
            lines.append(f"Seed {k} training ({GPU}, the two trainings shared the GPU): " + ", ".join(parts) + ".")
    lines.append("")
    rows, head = [], []
    for m in r0["methods"]:
        ps = [r["methods"][m]["psnr"]["mean"] for _, r in runs if m in r["methods"]]
        lp = [r["methods"][m]["lpips_alex"]["mean"] for _, r in runs if m in r["methods"]]
        rows.append([m, ",".join(str(k) for k, r in runs if m in r["methods"]), mean_std(ps, 2), mean_std(lp)])
        head.append(rows[-1])
    lines.append(table(["method", "seeds", "PSNR", "LPIPS (alex)"], rows))
    lines.append("")
    key = "paired_lpips_fm_minus_regression"
    d = [(k, r[key]["mean"]) for k, r in runs if key in r]
    if d:
        cis = [(k, r[key]) for k, r in runs if key in r]
        lines.append(f"Paired LPIPS fm_sample minus regression per seed (mean [95% CI over pairs]): "
                     f"{', '.join(f'seed {k}: {ci(c, 4)}' for k, c in cis)} ({sum(v < 0 for _, v in d)}/{len(d)} seeds negative, "
                     f"i.e. the sample is perceptually closer; CI excludes zero in {sum(c['hi'] < 0 for _, c in cis)}/{len(cis)}).")  # fmt: skip
        dp = [(k, r["methods"]["fm_sample"]["psnr"]["mean"] - r["methods"]["regression"]["psnr"]["mean"]) for k, r in runs
              if "regression" in r["methods"]]
        lines.append(f"PSNR fm_sample minus regression per seed: {', '.join(f'seed {k}: {v:+.2f} dB' for k, v in dp)} "
                     f"({sum(v < 0 for _, v in dp)}/{len(dp)} seeds negative).")  # fmt: skip
        dc = [(k, r["paired_psnr_fm_minus_copy"]["mean"], r["paired_lpips_fm_minus_copy"]["mean"]) for k, r in runs]
        lines.append(f"fm_sample minus copy_current per seed: PSNR {', '.join(f'{v:+.2f}' for _, v, _ in dc)} dB, LPIPS "
                     f"{', '.join(f'{v:+.4f}' for _, _, v in dc)} (copy_current is ahead in every seed where PSNR is negative "
                     f"and LPIPS is positive: {sum(v < 0 and w > 0 for _, v, w in dc)}/{len(dc)}).")  # fmt: skip
        dr = [(k, r["methods"]["regression"]["psnr"]["mean"] - r["methods"]["copy_current"]["psnr"]["mean"],
               r["methods"]["regression"]["lpips_alex"]["mean"] - r["methods"]["copy_current"]["lpips_alex"]["mean"])
              for k, r in runs if "regression" in r["methods"]]
        lines.append(f"regression minus copy_current per seed (difference of means, not a paired test): PSNR "
                     f"{', '.join(f'{v:+.2f}' for _, v, _ in dr)} dB ({sum(v > 0 for _, v, _ in dr)}/{len(dr)} seeds positive), LPIPS "
                     f"{', '.join(f'{v:+.4f}' for _, _, v in dr)} ({sum(v > 0 for _, _, v in dr)}/{len(dr)} seeds positive, i.e. worse).")  # fmt: skip
        fids = [(k, r["fid_unconditional"]["value"]) for k, r in runs if r.get("fid_unconditional", {}).get("value") is not None]
        if fids:
            lines.append(f"FID of unconditional samples per seed: {', '.join(f'seed {k}: {v:.1f}' for k, v in fids)}.")
        lines.append("")
    return head, lines


def write_status_block() -> None:
    """Runtime table for STATUS.md, between markers, from the json files."""
    rows = []
    gt = load("lidar_gt_stats.json")
    if gt:
        rows.append(["LiDAR GT projection (CPU)", f"{gt['n_images']:,} images", f(gt["wall_time_s"] / 60, 1), "ran"])
    for p in sorted(RES.glob("mono_depth_*.json")):
        r = json.loads(p.read_text())
        if r["variant"].endswith("_gtfocal"):
            continue
        rows.append([f"Track A {r['variant']}", f"{r['n_images']:,} images, {f(r['timing_ms_per_image']['median'], 0)} ms/img median",
                     f(r["wall_time_s"] / 60, 1), "ran"])
    gs = [json.loads(p.read_text()) for p in sorted(RES.glob("gs_*.json"))]
    gs.sort(key=lambda r: (split_seed(r["name"])[0], r["seed"]))
    for r in gs:
        rows.append([f"Track C 3DGS {r['name']} (seed {r['seed']})", f"{r['steps']:,} steps, {r['n_gaussians_final']:,} gaussians, {f(r['steps_per_s'], 1)} steps/s",
                     f((r["train_time_s"] + r["init_time_s"]) / 60, 1), "ran"])
    mv = load("multiview_vggt.json")
    if mv:
        n = sum(s["n_sequences"] for s in mv["summary"].values())
        tot = sum(s["n_sequences"] * s["timing_ms_per_sequence"]["mean"] for s in mv["summary"].values()) / 1000
        rows.append(["Track B VGGT", f"{n} sequences", f((tot + mv["model_load_s"]) / 60, 1), "ran"])
    seed_tags = [""] + sorted(p.stem.rsplit("_s", 1)[1] for p in RES.glob("synthesis_nextframe_s*.json"))
    for tag in seed_tags:
        sfx = f"_s{tag}" if tag else ""
        for mode in ("fm", "regression"):
            t = load(f"nextframe_train_{mode}{sfx}.json")
            if t:
                rows.append([f"Track E train {mode} (seed {t.get('seed', 0)})",
                             f"{t['steps']:,} steps, {f(t['steps_per_s'], 1)} steps/s (GPU shared with the other training)",
                             f(t["train_time_s"] / 60, 1), "ran"])
        nf = load(f"synthesis_nextframe{sfx}.json")
        if nf:
            rows.append([f"Track E eval (seed {nf.get('seed', 0)})", f"{nf['n_test_pairs']} pairs, {f(nf['sampling_ms_per_frame'], 1)} ms/sample (sampling time only)",
                         f(nf["sampling_ms_per_frame"] * nf["n_test_pairs"] / 60000, 1), "ran"])
    dd = load("ddp_smoke.json")
    if dd:
        rows.append(["Track E DDP smoke (torchrun, gloo, 2 ranks on ONE GPU)", f"{dd['steps']} steps, world size {dd['world_size']}", f(dd["train_time_s"] / 60, 2), "ran (smoke only, not a multi-GPU test)"])
    sh = load("sharp_nvs.json")
    if sh:
        if sh.get("status") == "ok":
            rows.append(["Track D SHARP", f"{sh['n_images']} images, {f(sh['predict_ms_mean'], 0)} ms predict, {f(sh['render_ms_mean'], 1)} ms render", "n/a", "ran"])
        else:
            rows.append(["Track D SHARP", sh.get("reason", ""), "n/a", "failed"])
    else:
        rows.append(["Track D SHARP", "results/sharp_nvs.json missing", "n/a", "not run"])
    block = ["<!-- STATUS:BEGIN (generated by scripts/make_results.py) -->", "",
             table(["job", "size", f"minutes ({GPU})", "status"], rows), "", "<!-- STATUS:END -->"]
    st = Path("STATUS.md")
    if st.exists():
        txt = st.read_text()
        a, b = txt.find("<!-- STATUS:BEGIN"), txt.find("<!-- STATUS:END -->")
        if a >= 0 and b >= 0:
            st.write_text(txt[:a] + "\n".join(block) + txt[b + len("<!-- STATUS:END -->"):])


def main() -> None:
    parts = []
    parts.append("# Results\n\nGenerated by `python scripts/make_results.py` from `results/*.json`. Do not edit by hand. "
                 f"All timings measured on an {GPU} (23 GB). Dataset: nuScenes v1.0-mini (10 scenes, 404 keyframes).\n")  # fmt: skip
    heads = {}
    for fn in (track_a, track_c, track_b, track_e, track_d):
        text, head = fn()
        parts.append(text)
        heads[fn.__name__] = head
    doc = "\n".join(parts) + "\n"
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(doc)

    # README block
    blk = ["<!-- RESULTS:BEGIN (generated by scripts/make_results.py, do not edit) -->", ""]
    blk.append("**Track A - monocular metric depth (2,424 nuScenes-mini images, LiDAR GT 1-80 m)**")
    blk.append("")
    if heads["track_a"]:
        blk.append(table(["variant", "AbsRel raw", "AbsRel scaled", "d1 raw", "d1 scaled", f"ms/img ({GPU})"], heads["track_a"]))
    else:
        blk.append("not measured")
    blk.append("")
    blk.append("**Track C - 3DGS ablation on the selected scene (held-out views)**")
    blk.append("")
    blk.append(table(["variant", "PSNR", "SSIM", "LPIPS", "gaussians", f"train min ({GPU})"], heads["track_c"]["seed0"]) if heads["track_c"]["seed0"] else "not measured")
    blk.append("")
    if heads["track_c"]["seeds"]:
        blk.append("Seed variance of the 7k-step variants (mean +- sample std over the listed seeds):")
        blk.append("")
        blk.append(table(["variant", "seeds", "PSNR", "SSIM", "LPIPS"], heads["track_c"]["seeds"]))
        blk.append("")
    blk.append("**Track B - VGGT vs Depth Pro on identical reference pixels (AbsRel, 95% CI)**")
    blk.append("")
    blk.append(table(["setting", "VGGT seq-scaled", "Depth Pro img-scaled", "rot err median deg", "trans dir err median deg"], heads["track_b"]) if heads["track_b"] else "not measured")
    blk.append("")
    blk.append("**Track E - next-frame synthesis on 2 held-out scenes**")
    blk.append("")
    blk.append(table(["method", "PSNR", "LPIPS"], heads["track_e"]["seed0"]) if heads["track_e"]["seed0"] else "not measured")
    blk.append("")
    if heads["track_e"]["seeds"]:
        blk.append("Seed variance (training and sampling seed; mean +- sample std over the listed seeds):")
        blk.append("")
        blk.append(table(["method", "seeds", "PSNR", "LPIPS"], heads["track_e"]["seeds"]))
        blk.append("")
    blk.append("**Track D - SHARP single-image Gaussians rendered at the next pose**")
    blk.append("")
    blk.append(table(["target", "PSNR", "LPIPS", "motion [m]"], heads["track_d"]) if heads["track_d"] else "not measured")
    blk.append("")
    blk.append("<!-- RESULTS:END -->")
    block = "\n".join(blk)
    if README.exists():
        txt = README.read_text()
        a, b = txt.find("<!-- RESULTS:BEGIN"), txt.find("<!-- RESULTS:END -->")
        if a >= 0 and b >= 0:
            txt = txt[:a] + block + txt[b + len("<!-- RESULTS:END -->"):]
            README.write_text(txt)
    for p in (DOC, README):
        if p.exists() and any(ch in p.read_text() for ch in "–—"):
            raise SystemExit(f"long dash found in {p}")
    write_status_block()
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
