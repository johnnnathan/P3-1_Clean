# MotionBERT/visualize_pose.py

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ===== 13-joint edges (EH36M-style) =====
EDGES_13 = [
    (1, 2), (2, 3),      # Right arm
    (4, 5), (5, 6),      # Left arm
    (0, 1), (0, 4),      # Neck -> shoulders
    (1, 7), (4, 10),     # Shoulders -> hips
    (7, 8), (8, 9),      # Right leg
    (10, 11), (11, 12),  # Left leg
]

EDGES_17 = [
    (0, 1), (1, 2), (2, 3),      # right leg
    (0, 4), (4, 5), (5, 6),      # left leg
    (0, 7), (7, 8),              # pelvis->spine->neck
    (8, 11), (11, 12), (12, 13), # left arm
    (8, 14), (14, 15), (15, 16), # right arm
    (8, 9), (9, 10),             # neck->nose->head
]

EDGE_COLOR = "tab:blue"
POINT_COLOR = "black"


def _to_numpy(pose_seq):
    if torch.is_tensor(pose_seq):
        pose_seq = pose_seq.detach().cpu().numpy()
    return np.asarray(pose_seq, dtype=np.float32)



def animate_pose_sequence(pose_seq, save_path="pose.gif", fps=12, interval=120):
    """
    pose_seq: [T,J,2] where J is 13 or 17
    Saves a GIF.
    """
    pose_seq = _to_numpy(pose_seq)
    if pose_seq.ndim != 3 or pose_seq.shape[2] != 2:
        raise ValueError(f"Expected pose_seq [T,J,2], got {pose_seq.shape}")

    T, J, _ = pose_seq.shape

    if J == 13:
        edges = EDGES_13
    elif J == 17:
        edges = EDGES_17
    else:
        edges = []

    fig, ax = plt.subplots(figsize=(5, 5))

    # Treat coordinates as pixel coords and DON'T invert in plotting.
    # (If your input is already upright in pixel coords, this fixes the upside-down issue.)
    all_x = pose_seq[:, :, 0].reshape(-1)
    all_y = pose_seq[:, :, 1].reshape(-1)

    xmin, xmax = float(all_x.min()) - 20, float(all_x.max()) + 20
    ymin, ymax = float(all_y.min()) - 20, float(all_y.max()) + 20

    def update(t):
        ax.clear()
        pts = pose_seq[t]
        x = pts[:, 0]
        y = pts[:, 1]

        ax.scatter(x, y, c=POINT_COLOR, s=25)

        for i, j in edges:
            if i < J and j < J:
                ax.plot([x[i], x[j]], [y[i], y[j]], color=EDGE_COLOR, linewidth=2)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)  # <-- no inversion
        ax.set_title(f"Frame {t}")
        ax.axis("off")

    ani = animation.FuncAnimation(fig, update, frames=T, interval=interval)
    ani.save(save_path, writer="pillow", fps=fps)
    print("Saved animation to", save_path)



    def update(t):
        ax.clear()
        pts = pose_seq[t]
        x = pts[:, 0]
        y = pts[:, 1]

        ax.scatter(x, y, c=POINT_COLOR, s=25)

        for i, j in edges:
            if i < J and j < J:
                ax.plot([x[i], x[j]], [y[i], y[j]], color=EDGE_COLOR, linewidth=2)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)  # invert y-axis (image-like)
        ax.set_title(f"Frame {t}")
        ax.axis("off")

    ani = animation.FuncAnimation(fig, update, frames=T, interval=interval)
    ani.save(save_path, writer="pillow", fps=fps)
    print("Saved animation to", save_path)
