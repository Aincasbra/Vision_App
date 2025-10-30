import numpy as np


object_history = {}
next_stable_id = 1
frame_counter = 0


def calculate_object_similarity(box1, box2, class1, class2):
    x1, y1, x2, y2 = box1
    px1, py1, px2, py2 = box2
    inter_w = max(0, min(x2, px2) - max(x1, px1))
    inter_h = max(0, min(y2, py2) - max(y1, y2 if False else py2))  # guard simple
    inter_h = max(0, min(y2, py2) - max(y1, py1))
    inter = inter_w * inter_h
    area1 = max(0, x2-x1) * max(0, y2-y1)
    area2 = max(0, px2-px1) * max(0, py2-py1)
    union = area1 + area2 - inter
    iou = inter / union if union > 0 else 0.0
    cx1, cy1 = (x1+x2)/2, (y1+y2)/2
    cx2, cy2 = (px1+px2)/2, (py1+py2)/2
    center_dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
    size1 = np.sqrt(area1)
    size2 = np.sqrt(area2)
    size_ratio = min(size1, size2) / max(size1, size2) if max(size1, size2) > 0 else 0
    class_match = 1.0 if class1 == class2 else 0.0
    similarity = (
        0.4 * iou +
        0.2 * max(0, 1 - center_dist/100) +
        0.2 * size_ratio +
        0.2 * class_match
    )
    return similarity, iou, center_dist


def assign_stable_ids(xyxy, confs, clss, current_frame):
    global object_history, next_stable_id, frame_counter
    frame_counter = current_frame
    stable_ids = []
    if len(xyxy) == 0:
        return stable_ids
    to_remove = []
    for obj_id, data in object_history.items():
        if current_frame - data['last_frame'] > 10:
            to_remove.append(obj_id)
    for obj_id in to_remove:
        del object_history[obj_id]
    MAX_OBJECTS = 12
    used_historical_ids = set()
    for i, (box, conf, cls) in enumerate(zip(xyxy, confs, clss)):
        best_match_id = None
        best_similarity = 0.0
        best_iou = 0.0
        for obj_id, data in object_history.items():
            if obj_id in used_historical_ids:
                continue
            hist_box = data['last_box']
            hist_cls = data.get('class', cls)
            similarity, iou, center_dist = calculate_object_similarity(box, hist_box, cls, hist_cls)
            if (similarity > best_similarity and iou >= 0.3 and center_dist <= 100 and similarity >= 0.45):
                best_match_id = obj_id
                best_similarity = similarity
                best_iou = iou
        if best_match_id is not None:
            stable_ids.append(best_match_id)
            used_historical_ids.add(best_match_id)
            object_history[best_match_id].update({
                'last_box': box.copy(),
                'last_frame': current_frame,
                'stable_count': object_history[best_match_id]['stable_count'] + 1,
                'class': cls
            })
        else:
            adopt_id = None
            min_dist = 1e9
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            for obj_id, data in object_history.items():
                px1, py1, px2, py2 = data['last_box']
                pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
                d = np.hypot(cx - pcx, cy - pcy)
                if d < min_dist:
                    min_dist = d
                    adopt_id = obj_id
            if adopt_id is not None and min_dist <= 80:
                stable_ids.append(adopt_id)
                used_historical_ids.add(adopt_id)
                object_history[adopt_id].update({
                    'last_box': box.copy(),
                    'last_frame': current_frame,
                    'stable_count': object_history[adopt_id]['stable_count'] + 1,
                    'class': cls
                })
            elif float(confs[i]) >= 0.5 and len(object_history) < MAX_OBJECTS:
                new_id = next_stable_id
                next_stable_id += 1
                stable_ids.append(new_id)
                object_history[new_id] = {
                    'last_box': box.copy(),
                    'last_frame': current_frame,
                    'stable_count': 1,
                    'class': cls
                }
            else:
                stable_ids.append(-1)
    return stable_ids


