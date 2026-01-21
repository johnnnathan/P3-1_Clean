import os
import cv2
import numpy as np
import mediapipe as mp

DVS_AVI = r"C:\Users\alenm\OneDrive\Desktop\rgb\sittingv2e\dvs-video.avi"
OUT_NPY = r"C:\Users\alenm\OneDrive\Desktop\rgb\sitting\dvs_pose_17.npy"

MIN_DET_CONF = 0.3
MIN_TRACK_CONF = 0.3
MODEL_COMPLEXITY = 1
FILL_MISSING = True

if not hasattr(mp, "solutions"):
    raise RuntimeError(
        "Your mediapipe package doesn't expose 'mediapipe.solutions'. "
        "Try: pip uninstall mediapipe -y && pip install mediapipe==0.10.14"
    )

mp_pose = mp.solutions.pose

def mp33_to_h36m17(mp33: np.ndarray) -> np.ndarray:
    NOSE = 0
    L_SHO, R_SHO = 11, 12
    L_ELB, R_ELB = 13, 14
    L_WRI, R_WRI = 15, 16
    L_HIP, R_HIP = 23, 24
    L_KNE, R_KNE = 25, 26
    L_ANK, R_ANK = 27, 28

    lhip = mp33[L_HIP]; rhip = mp33[R_HIP]
    lsho = mp33[L_SHO]; rsho = mp33[R_SHO]

    hip = 0.5 * (lhip + rhip)
    neck = 0.5 * (lsho + rsho)
    spine = 0.5 * (hip + neck)
    head = neck + 0.6 * (mp33[NOSE] - neck)

    out = np.zeros((17, 2), dtype=np.float32)
    out[0]  = hip
    out[1]  = mp33[R_HIP]
    out[2]  = mp33[R_KNE]
    out[3]  = mp33[R_ANK]
    out[4]  = mp33[L_HIP]
    out[5]  = mp33[L_KNE]
    out[6]  = mp33[L_ANK]
    out[7]  = spine
    out[8]  = neck
    out[9]  = mp33[NOSE]
    out[10] = head
    out[11] = mp33[L_SHO]
    out[12] = mp33[L_ELB]
    out[13] = mp33[L_WRI]
    out[14] = mp33[R_SHO]
    out[15] = mp33[R_ELB]
    out[16] = mp33[R_WRI]
    return out

def interpolate_missing(seq):
    T = len(seq)
    out = [None] * T
    valid = [i for i, p in enumerate(seq) if p is not None]
    if len(valid) == 0:
        return np.zeros((T, 17, 2), dtype=np.float32)

    first = valid[0]
    for i in range(0, first):
        out[i] = seq[first].copy()

    last = valid[-1]
    for i in range(last, T):
        out[i] = seq[last].copy()

    for a, b in zip(valid[:-1], valid[1:]):
        pa = seq[a]; pb = seq[b]
        out[a] = pa.copy()
        gap = b - a
        for k in range(1, gap):
            t = k / gap
            out[a + k] = (1.0 - t) * pa + t * pb

    return np.stack(out, axis=0).astype(np.float32)

def main():
    cap = cv2.VideoCapture(DVS_AVI)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {DVS_AVI}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pose_est = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        enable_segmentation=False,
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRACK_CONF,
    )

    seq = []
    i = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = pose_est.process(rgb)

        if res.pose_landmarks:
            mp33 = np.array([[lm.x * W, lm.y * H] for lm in res.pose_landmarks.landmark], dtype=np.float32)
            seq.append(mp33_to_h36m17(mp33))
        else:
            seq.append(None)

        i += 1
        if i % 100 == 0:
            print(f"Processed {i}/{total} frames")

    cap.release()
    pose_est.close()

    if FILL_MISSING:
        pose17 = interpolate_missing(seq)
    else:
        pose17 = np.array([p if p is not None else np.full((17,2), np.nan, dtype=np.float32) for p in seq], dtype=np.float32)

    os.makedirs(os.path.dirname(OUT_NPY), exist_ok=True)
    np.save(OUT_NPY, pose17)
    print("Saved:", OUT_NPY, "| shape:", pose17.shape)

if __name__ == "__main__":
    main()
