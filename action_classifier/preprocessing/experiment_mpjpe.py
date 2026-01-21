# compare_vs_gt_mpjpe_report.py
# Hardcoded experiment script:
# Compare MoveNet(RGB) vs MediaPipe(DVS-video) skeletons against GT (data.log),
# compute MPJPE, save CSV + summary + plots for report.

import os
import re
import numpy as np
import matplotlib.pyplot as plt


# ==========================
# 1) HARD-CODE YOUR PATHS
# ==========================
EXPERIMENTS = {
    "eating": {
        "movenet_npy": r"C:\Users\alenm\OneDrive\Desktop\rgb\eating\rgb_pose_17.npy",
        "dvs_mp_npy":  r"C:\Users\alenm\OneDrive\Desktop\rgb\eating\dvs_pose_17.npy",
        "gt_log":      r"C:\Users\alenm\OneDrive\Desktop\rgb\eating\data.log",
    },
    "greeting": {
        "movenet_npy": r"C:\Users\alenm\OneDrive\Desktop\rgb\greeting\rgb_pose_17.npy",
        "dvs_mp_npy":  r"C:\Users\alenm\OneDrive\Desktop\rgb\greeting\dvs_pose_17.npy",
        "gt_log":      r"C:\Users\alenm\OneDrive\Desktop\rgb\greeting\data.log",
    },
    "sitting": {
        "movenet_npy": r"C:\Users\alenm\OneDrive\Desktop\rgb\sitting\rgb_pose_17.npy",
        "dvs_mp_npy":  r"C:\Users\alenm\OneDrive\Desktop\rgb\sitting\dvs_pose_17.npy",
        "gt_log":      r"C:\Users\alenm\OneDrive\Desktop\rgb\sitting\data.log",
    },
}

OUT_DIR = r"C:\Users\alenm\OneDrive\Desktop\rgb\results"
os.makedirs(OUT_DIR, exist_ok=True)


# ===================================
# 2) GT parsing (EH36M data.log SKLT)
# ===================================
# Example line:
# 0 0000.0 SKLT (224  30 182  48 ... 241 377) -1.0 149.88
SKLT_RE = re.compile(r"^\s*\d+\s+([0-9.]+)\s+SKLT\s+\(([^)]+)\)")

def load_gt_from_datalog(log_path: str):
    """
    Returns:
      gt_t:   [T] timestamps in seconds
      gt_xy:  [T, 13, 2]  (13 joints, x/y pixel coords)
    """
    ts = []
    poses = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = SKLT_RE.match(line)
            if not m:
                continue
            t = float(m.group(1))
            coords_str = m.group(2).strip()

            # coords are a list of ints, length should be 26 (13*2)
            nums = [int(x) for x in coords_str.split()]
            if len(nums) < 26:
                continue
            nums = nums[:26]

            xy = np.array(nums, dtype=np.float32).reshape(13, 2)
            ts.append(t)
            poses.append(xy)

    if not poses:
        raise RuntimeError(f"No SKLT lines found in {log_path}")

    gt_t = np.array(ts, dtype=np.float32)
    gt_xy = np.stack(poses, axis=0).astype(np.float32)  # [T,13,2]
    return gt_t, gt_xy


# ============================================
# 3) Map your [T,17,2] pose to GT [T,13,2]
# ============================================
# Your pipeline 17-joint (H36M-ish) indices:
# 0 hip, 1 rhip,2 rknee,3 rankle,4 lhip,5 lknee,6 lankle,
# 7 spine, 8 neck, 9 nose, 10 head,
# 11 lsho,12 lelb,13 lwri,14 rsho,15 relb,16 rwri
#
# GT 13-joint (EH36M-style) order consistent with your EDGES_13:
# 0 neck, 1 rsho,2 relb,3 rwri, 4 lsho,5 lelb,6 lwri,
# 7 rhip,8 rknee,9 rankle, 10 lhip,11 lknee,12 lankle

H36M17_TO_GT13 = [8, 14, 15, 16, 11, 12, 13, 1, 2, 3, 4, 5, 6]

def pose17_to_gt13(pose17: np.ndarray) -> np.ndarray:
    """
    pose17: [T,17,2]
    return: [T,13,2]
    """
    if pose17.ndim != 3 or pose17.shape[1] < 17 or pose17.shape[2] != 2:
        raise ValueError(f"Expected [T,17,2], got {pose17.shape}")
    return pose17[:, H36M17_TO_GT13, :].astype(np.float32)


# ============================================
# 4) Align predicted sequence to GT timestamps
# ============================================
def interp_pose_to_t(pred_pose: np.ndarray, pred_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    """
    pred_pose: [T_pred, J, 2]
    pred_t:    [T_pred]
    target_t:  [T_gt]
    returns:   [T_gt, J, 2] by linear interpolation per joint/coord
    """
    Tgt = target_t.shape[0]
    J = pred_pose.shape[1]
    out = np.zeros((Tgt, J, 2), dtype=np.float32)

    # clamp target_t to pred range
    tt = np.clip(target_t, pred_t[0], pred_t[-1])

    for j in range(J):
        for c in range(2):
            out[:, j, c] = np.interp(tt, pred_t, pred_pose[:, j, c])
    return out


def build_uniform_time_for_pred(T_pred: int, gt_t: np.ndarray) -> np.ndarray:
    """
    If you don't have true timestamps for prediction, assume it spans
    from gt_t[0]..gt_t[-1] uniformly with T_pred frames.
    """
    return np.linspace(float(gt_t[0]), float(gt_t[-1]), T_pred, dtype=np.float32)


# =====================
# 5) MPJPE computation
# =====================
def mpjpe_per_frame(pred_xy: np.ndarray, gt_xy: np.ndarray) -> np.ndarray:
    """
    pred_xy, gt_xy: [T,J,2]
    returns mpjpe_t: [T] where each entry is mean joint L2 distance
    """
    diff = pred_xy - gt_xy
    d = np.sqrt((diff ** 2).sum(axis=-1))  # [T,J]
    return d.mean(axis=1)                  # [T]


# =====================
# 6) Main experiment
# =====================
def run_one(action: str, paths: dict):
    movenet_path = paths["movenet_npy"]
    dvsmp_path = paths["dvs_mp_npy"]
    gtlog_path = paths["gt_log"]

    # Load GT
    gt_t, gt_xy = load_gt_from_datalog(gtlog_path)     # [Tgt], [Tgt,13,2]

    # Load predictions (expect [T,17,2])
    movenet_17 = np.load(movenet_path).astype(np.float32)
    dvsmp_17   = np.load(dvsmp_path).astype(np.float32)

    movenet_13 = pose17_to_gt13(movenet_17)            # [Tm,13,2]
    dvsmp_13   = pose17_to_gt13(dvsmp_17)              # [Td,13,2]

    # Align to GT time axis (uniform time assumption)
    movenet_t = build_uniform_time_for_pred(movenet_13.shape[0], gt_t)
    dvsmp_t   = build_uniform_time_for_pred(dvsmp_13.shape[0], gt_t)

    movenet_aligned = interp_pose_to_t(movenet_13, movenet_t, gt_t)  # [Tgt,13,2]
    dvsmp_aligned   = interp_pose_to_t(dvsmp_13, dvsmp_t, gt_t)

    # MPJPE per-frame + means
    mpjpe_m = mpjpe_per_frame(movenet_aligned, gt_xy)  # [Tgt]
    mpjpe_d = mpjpe_per_frame(dvsmp_aligned, gt_xy)    # [Tgt]

    mean_m = float(mpjpe_m.mean())
    mean_d = float(mpjpe_d.mean())

    # Save CSV
    csv_path = os.path.join(OUT_DIR, f"{action}_mpjpe.csv")
    np.savetxt(
        csv_path,
        np.column_stack([gt_t, mpjpe_m, mpjpe_d]),
        delimiter=",",
        header="t_sec,mpjpe_movenet,mpjpe_dvs_mediapipe",
        comments="",
        fmt="%.6f",
    )

    # Save summary
    summary_path = os.path.join(OUT_DIR, f"{action}_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Action: {action}\n")
        f.write(f"GT frames: {gt_xy.shape[0]}\n")
        f.write(f"MoveNet frames (raw): {movenet_13.shape[0]}\n")
        f.write(f"DVS+MediaPipe frames (raw): {dvsmp_13.shape[0]}\n")
        f.write("\n")
        f.write(f"Mean MPJPE (MoveNet vs GT): {mean_m:.3f} px\n")
        f.write(f"Mean MPJPE (DVS+MediaPipe vs GT): {mean_d:.3f} px\n")

    # Plot curve over time (report figure 1)
    fig1_path = os.path.join(OUT_DIR, f"{action}_mpjpe_curve.png")
    plt.figure()
    plt.plot(gt_t, mpjpe_m, label="MoveNet (RGB) vs GT")
    plt.plot(gt_t, mpjpe_d, label="MediaPipe (DVS) vs GT")
    plt.xlabel("Time (s)")
    plt.ylabel("MPJPE (pixels)")
    plt.title(f"MPJPE over time — {action}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=200)
    plt.close()

    # Plot bar chart (report figure 2)
    fig2_path = os.path.join(OUT_DIR, f"{action}_mpjpe_bar.png")
    plt.figure()
    plt.bar(["MoveNet (RGB)", "MediaPipe (DVS)"], [mean_m, mean_d])
    plt.ylabel("Mean MPJPE (pixels)")
    plt.title(f"Mean MPJPE — {action}")
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=200)
    plt.close()

    print(f"[{action}] Saved:")
    print("  CSV:", csv_path)
    print("  Summary:", summary_path)
    print("  Curve:", fig1_path)
    print("  Bar:", fig2_path)
    print(f"  Mean MPJPE: MoveNet={mean_m:.3f}px | DVS+MP={mean_d:.3f}px\n")


def main():
    for action, paths in EXPERIMENTS.items():
        run_one(action, paths)

    # Also: save a combined “all actions” comparison CSV + plot
    # (useful for the report’s main result table/figure)
    rows = []
    for action, paths in EXPERIMENTS.items():
        gt_t, gt_xy = load_gt_from_datalog(paths["gt_log"])
        movenet_13 = pose17_to_gt13(np.load(paths["movenet_npy"]).astype(np.float32))
        dvsmp_13   = pose17_to_gt13(np.load(paths["dvs_mp_npy"]).astype(np.float32))

        movenet_t = build_uniform_time_for_pred(movenet_13.shape[0], gt_t)
        dvsmp_t   = build_uniform_time_for_pred(dvsmp_13.shape[0], gt_t)

        movenet_aligned = interp_pose_to_t(movenet_13, movenet_t, gt_t)
        dvsmp_aligned   = interp_pose_to_t(dvsmp_13, dvsmp_t, gt_t)

        mean_m = float(mpjpe_per_frame(movenet_aligned, gt_xy).mean())
        mean_d = float(mpjpe_per_frame(dvsmp_aligned, gt_xy).mean())
        rows.append((action, mean_m, mean_d))

    combined_csv = os.path.join(OUT_DIR, "all_actions_mean_mpjpe.csv")
    with open(combined_csv, "w", encoding="utf-8") as f:
        f.write("action,mean_mpjpe_movenet,mean_mpjpe_dvs_mediapipe\n")
        for action, m, d in rows:
            f.write(f"{action},{m:.6f},{d:.6f}\n")

    # combined bar plot
    combined_fig = os.path.join(OUT_DIR, "all_actions_mean_mpjpe.png")
    actions = [r[0] for r in rows]
    mvals = [r[1] for r in rows]
    dvals = [r[2] for r in rows]

    x = np.arange(len(actions))
    width = 0.35

    plt.figure(figsize=(7, 4))
    plt.bar(x - width/2, mvals, width, label="MoveNet (RGB) vs GT")
    plt.bar(x + width/2, dvals, width, label="MediaPipe (DVS) vs GT")
    plt.xticks(x, actions)
    plt.ylabel("Mean MPJPE (pixels)")
    plt.title("Mean MPJPE across actions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(combined_fig, dpi=200)
    plt.close()

    print("Combined results saved:")
    print("  CSV:", combined_csv)
    print("  Plot:", combined_fig)


if __name__ == "__main__":
    main()
