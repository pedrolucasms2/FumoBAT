# src/data/transforms.py
import cv2
import numpy as np

def letterbox(im, new=640, color=(114,114,114)):
    h, w = im.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    imr = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new - nh) // 2
    left = (new - nw) // 2
    out = np.full((new, new, 3), color, dtype=im.dtype)
    out[top:top+nh, left:left+nw] = imr
    return out, r, left, top

def apply_boxes(boxes, r, left, top):
    if boxes.shape == 0:
        return boxes
    boxes = boxes.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * r + left
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * r + top
    return boxes

def simple_transforms(sample, size=640, hflip_prob=0.5):
    im, boxes, labels = sample["image"], sample["boxes"], sample["labels"]
    im, r, left, top = letterbox(im, size)
    boxes = apply_boxes(boxes, r, left, top)
    if np.random.rand() < hflip_prob and boxes.size > 0:
        im = np.ascontiguousarray(np.fliplr(im))
        w = im.shape[1]
        x1 = boxes[:, 0].copy(); x2 = boxes[:, 2].copy()
        boxes[:, 0] = w - x2; boxes[:, 2] = w - x1
    return {"image": im, "boxes": boxes, "labels": labels}
