import cv2
import matplotlib.pyplot as plt
import random
import numpy as np

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
            positions.append((i, i_end, j, j_end))

    random.shuffle(patches)


    face_copy = np.zeros_like(face)
    for idx, (i, i_end, j, j_end) in enumerate(positions):
        patch = patches[idx]
        h_patch, w_patch, _ = patch.shape
        h_slice = min(h_patch, i_end - i)
        w_slice = min(w_patch, j_end - j)
        face_copy[i:i+h_slice, j:j+w_slice] = patch[:h_slice, :w_slice]

    img_copy = img.copy()
    img_copy[y_min:y_max, x_min:x_max] = face_copy
    return img_copy


def expand_box(box, img_shape, margin=0.25):
    h, w = img_shape[:2]
    x_min, y_min, x_max, y_max = map(int, box)
    w_box, h_box = x_max - x_min, y_max - y_min
    x_min = max(0, x_min - int(margin * w_box))
    y_min = max(0, y_min - int(margin * h_box))
    x_max = min(w, x_max + int(margin * w_box))
    y_max = min(h, y_max + int(margin * h_box))
    return [x_min, y_min, x_max, y_max]

def draw_boxes(img, boxes, color=(0,255,0), thickness=2):
    img = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
    return img

valid_examples = []

for row in results: 
    if row["orig_detections"] > 0:
        valid_examples.append(row)

print(f"Found {len(valid_examples)} frames with face detections")


num_samples = 12
samples = random.sample(valid_examples, num_samples)

for s in samples:
    frame_idx = s["frame"]
    img_path = f"/content/event_frames/{frame_idx}"


    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    preds = face_model.predict(img, verbose=False)
    if len(preds) == 0 or preds[0].boxes is None or len(preds[0].boxes.xyxy) == 0:
        continue
    gt_box = preds[0].boxes.xyxy[0].cpu().numpy()

    exp_box = expand_box(gt_box, img.shape, margin=0.25)

    imgs = {
        "orig": img,
        "white": white_out_face(img.copy(), exp_box),
        "shuffle25": shuffle_face(img.copy(), exp_box, patch_size=16),
        "shuffle100": shuffle_face(img.copy(), exp_box, patch_size=4),
    }

    fig, axs = plt.subplots(1, 4, figsize=(16,4))

    for ax, (title, im) in zip(axs, imgs.items()):
        preds = face_model.predict(im, verbose=False)
        boxes = preds[0].boxes.xyxy.cpu().numpy() if preds[0].boxes is not None else []

        vis = draw_boxes(im, boxes)
        ax.imshow(vis)
        ax.set_title(f"{title}\nfaces: {len(boxes)}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
