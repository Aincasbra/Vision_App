"""
Ops de visión
-------------
- Utilidades de post-procesado (NMS/fusión, etc.).
- Consumidas por `overlay.py`/`yolo_service.py`.
"""
import numpy as np


def merge_overlapping_detections(xyxy, confs, clss, iou_threshold=0.7):
    if len(xyxy) == 0:
        return xyxy, confs, clss
    xyxy = np.asarray(xyxy)
    confs = np.asarray(confs)
    clss = np.asarray(clss)
    keep = []
    used = np.zeros(len(xyxy), dtype=bool)
    for i in range(len(xyxy)):
        if used[i]:
            continue
        xi1, yi1, xi2, yi2 = xyxy[i]
        ai = max(0, (xi2 - xi1)) * max(0, (yi2 - yi1))
        group = [i]
        used[i] = True
        for j in range(i + 1, len(xyxy)):
            if used[j]:
                continue
            xj1, yj1, xj2, yj2 = xyxy[j]
            aj = max(0, (xj2 - xj1)) * max(0, (yj2 - yj1))
            xx1 = max(xi1, xj1)
            yy1 = max(yi1, yj1)
            xx2 = min(xi2, xj2)
            yy2 = min(yi2, yj2)
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            union = ai + aj - inter
            iou = inter / union if union > 0 else 0.0
            if iou >= iou_threshold and clss[i] == clss[j]:
                group.append(j)
                used[j] = True
        best_idx = max(group, key=lambda k: confs[k])
        keep.append(best_idx)
    keep = np.array(keep, dtype=int)
    return xyxy[keep], confs[keep], clss[keep]


def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def circle_crop(img, cx, cy, r):
    h, w = img.shape[:2]
    y1, y2 = max(0, cy-r), min(h, cy+r)
    x1, x2 = max(0, cx-r), min(w, cx+r)
    roi = img[y1:y2, x1:x2]
    mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
    import cv2
    cv2.circle(mask, (cx-x1, cy-y1), r, 255, -1)
    result = roi.copy()
    result[mask == 0] = 0
    return result


