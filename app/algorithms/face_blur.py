
import cv2
import json
import random
import numpy as np
from pathlib import Path


# ---------------------------
# Core face transformation
# ---------------------------

def shuffle_face(img, box, patch_size=8):
    x_min, y_min, x_max, y_max = map(int, box)
    face = img[y_min:y_max, x_min:x_max].copy()
    h, w, c = face.shape

    patches, positions = [], []

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            i_end = min(i + patch_size, h)
            j_end = min(j + patch_size, w)
            patch = face[i:i_end, j:j_end].copy()
            patches.append(patch)
            positions.append((i, j))

    random.shuffle(patches)

    face_copy = face.copy()

    for patch, (i, j) in zip(patches, positions):
        ph, pw, _ = patch.shape

        # CLIP TO DESTINATION BOUNDS (critical)
        ph_eff = min(ph, h - i)
        pw_eff = min(pw, w - j)

        face_copy[i:i+ph_eff, j:j+pw_eff] = patch[:ph_eff, :pw_eff]

    out = img.copy()
    out[y_min:y_max, x_min:x_max] = face_copy
    return out

def expand_box(box, img_shape, margin=0.25):
    h, w = img_shape[:2]
    x_min, y_min, x_max, y_max = map(int, box)
    bw, bh = x_max - x_min, y_max - y_min

    x_min = max(0, x_min - int(margin * bw))
    y_min = max(0, y_min - int(margin * bh))
    x_max = min(w, x_max + int(margin * bw))
    y_max = min(h, y_max + int(margin * bh))

    return [x_min, y_min, x_max, y_max]


# ---------------------------
# Blur percentage mapping
# ---------------------------

def blur_percentage_to_patch_size(pct):
    """
    pct: 0–100
    returns patch size in pixels
    """
    pct = np.clip(pct, 0, 100)

    # principled mapping:
    # 0%   -> very large patches (almost unchanged)
    # 100% -> very small patches (max destruction)
    max_patch = 32
    min_patch = 4

    scale = 1.0 - (pct / 100.0)
    return int(min_patch + scale * (max_patch - min_patch))


# ---------------------------
# Image processing
# ---------------------------

def anonymize_image(
    img,
    face_model,
    blur_pct=50,
    expand_margin=0.25
):
    patch_size = blur_percentage_to_patch_size(blur_pct)
    detections = 0

    preds = face_model.predict(img, verbose=False)
    if len(preds) == 0 or preds[0].boxes is None:
        return img, detections

    boxes = preds[0].boxes.xyxy.cpu().numpy()
    detections = len(boxes)

    out = img.copy()
    for box in boxes:
        exp_box = expand_box(box, img.shape, expand_margin)
        out = shuffle_face(out, exp_box, patch_size)

    return out, detections


# ---------------------------
# Unified entry point
# ---------------------------

def anonymize_media(
    input_path,
    output_path,
    face_model,
    blur_pct=50,
    metadata_path=None
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    metadata = {
        "input": str(input_path),
        "output": str(output_path),
        "blur_percentage": blur_pct,
        "frames_processed": 0,
        "faces_detected_total": 0,
    }

    if input_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
        img = cv2.imread(str(input_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        out, faces = anonymize_image(img, face_model, blur_pct)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(output_path), out)

        metadata["frames_processed"] = 1
        metadata["faces_detected_total"] = faces

    else:
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out, faces = anonymize_image(rgb, face_model, blur_pct)
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

            writer.write(out)

            metadata["frames_processed"] += 1
            metadata["faces_detected_total"] += faces

        cap.release()
        writer.release()

    if metadata_path:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    return metadata
