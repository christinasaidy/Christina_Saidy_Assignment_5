import torch
from torch.utils.data import random_split

def splits(dataset, train_ratio=0.7, seed=42):
    total_len = len(dataset)
    train_len = int(train_ratio * total_len)
    test_len = total_len - train_len
    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len], generator=generator)
    return train_dataset, test_dataset

def trainval_splits(dataset, val_ratio=0.2, seed=42):
    total_len = len(dataset)
    val_len = int(val_ratio * total_len)
    train_len = total_len - val_len
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(dataset, [train_len, val_len], generator=generator)
    return train_subset, val_subset

def intersection_over_union(boxA, boxB):  # Reference: https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    x1, y1, x2, y2 = boxA
    x1g, y1g, x2g, y2g = boxB

    xA = max(x1, x1g)
    yA = max(y1, y1g)
    xB = min(x2, x2g)
    yB = min(y2, y2g)

    interArea = max(0, xB - xA ) * max(0, yB - yA )

    boxAArea = (x2 - x1) * (y2 - y1)
    boxBArea = (x2g - x1g) * (y2g - y1g)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def bbox_to_pix(bbox, img_w, img_h): #transforms normalized bbox into regular coordinates [xmin,ymin, xmax, ymax]
    cx, cy, bw, bh = bbox
    cx *= img_w
    cy *= img_h
    bw *= img_w
    bh *= img_h
    xmin = cx - bw / 2
    ymin = cy - bh / 2
    xmax = cx + bw / 2
    ymax = cy + bh / 2
    return [xmin, ymin, xmax, ymax]

#reference to bbox_to_pix: https://github.com/tanjeffreyz/yolo-v1/blob/main/utils.py