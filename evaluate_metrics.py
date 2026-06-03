"""
Evaluation Metrics & Tracking Benchmark Utility
==================================================
Demonstrates deep understanding of Computer Vision and Object Tracking evaluation metrics,
including Intersection over Union (IoU), Precision, Recall, F1-Score, and tracking-specific
benchmarks (MOTA / IDF1).

Fulfills: "Evaluation Metrics: Familiarity with IoU, mAP, precision/recall, MOTA / IDF1"
"""

import numpy as np

def calculate_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Bounding box format: [x1, y1, x2, y2] (top-left, bottom-right coordinates)
    """
    # 1. Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # 2. Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # 3. Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # 4. Compute the Intersection over Union (IoU)
    # IoU = Area of Intersection / Area of Union
    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea == 0:
        return 0.0
        
    iou = interArea / unionArea
    return iou


def calculate_precision_recall(detections, ground_truths, iou_threshold=0.5):
    """
    Calculate Precision, Recall, and F1-Score for a set of detections vs ground truths.
    detections: List of bounding boxes [x1, y1, x2, y2] predicted by model.
    ground_truths: List of actual bounding boxes [x1, y1, x2, y2] in the frame.
    """
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives

    matched_gt = set()

    for det in detections:
        best_iou = 0
        best_gt_idx = -1
        
        for idx, gt in enumerate(ground_truths):
            if idx in matched_gt:
                continue
            iou = calculate_iou(det, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        # If best IoU matches or exceeds threshold, it's a True Positive
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    # False Negatives are the ground truth boxes that were never matched
    fn = len(ground_truths) - len(matched_gt)

    # Calculations
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return tp, fp, fn, precision, recall, f1


def tracking_metrics_explanation():
    """
    Explanations of advanced tracking evaluation metrics (MOTA, IDF1)
    to demonstrate academic and industrial depth in multi-object tracking.
    """
    explanation = """
======================================================================
                 MULTI-OBJECT TRACKING (MOT) METRICS
======================================================================

1. MOTA (Multiple Object Tracking Accuracy)
-------------------------------------------
MOTA is the standard metric used to measure the overall tracking system accuracy.
It combines three types of errors: False Positives, False Negatives, and ID Switches.

Formula:
  MOTA = 1 - ( sum(FN_t + FP_t + IDSW_t) ) / sum(GT_t)

Where:
  - FN_t: False Negatives (missed targets) at frame t
  - FP_t: False Positives (ghost detections) at frame t
  - IDSW_t: Identity Switches (tracker shifts identity from target A to B) at frame t
  - GT_t: Ground Truth count at frame t

Interpretation:
  - Ranges from -inf to 1 (or 100%).
  - Focuses on frame-level detection accuracy and count consistency.


2. IDF1 (ID F1-Score)
---------------------
IDF1 measures the ratio of correctly identified detections over the average number of 
ground-truth and computed detections. Unlike MOTA (which penalizes identity switches locally),
IDF1 measures how long a tracker successfully maintains correct identity mappings globally.

Formula:
  IDF1 = 2 * IDP * IDR / (IDP + IDR)

Where:
  - IDP (ID Precision): Fraction of computed trajectories that map to a correct identity.
  - IDR (ID Recall): Fraction of ground-truth trajectories that are correctly identified.

Interpretation:
  - Ranges from 0 to 1 (or 100%).
  - Essential for long-term tracking consistency, re-identification (Re-ID), and track persistence.
======================================================================
"""
    return explanation


def main():
    print("="*60)
    print(" Computer Vision Model Evaluation Metric Sandbox")
    print("="*60)

    # Mock Bounding Boxes
    # format: [x1, y1, x2, y2]
    box_gt = [100, 100, 200, 200]
    box_det_good = [110, 110, 205, 205]  # High overlap
    box_det_poor = [150, 150, 250, 250]  # Low overlap

    iou_good = calculate_iou(box_gt, box_det_good)
    iou_poor = calculate_iou(box_gt, box_det_poor)

    print("\n1. Intersection over Union (IoU) Calculations:")
    print(f" -> GT Box: {box_gt}")
    print(f" -> Det Box (Good): {box_det_good} | IoU: {iou_good:.4f}")
    print(f" -> Det Box (Poor): {box_det_poor} | IoU: {iou_poor:.4f}")

    # Mock Dataset Evaluation (1 Frame)
    ground_truths = [
        [100, 100, 200, 200],  # Person A
        [300, 150, 400, 300],  # Person B
        [50, 300, 150, 450],   # Person C
    ]
    detections = [
        [105, 102, 198, 201],  # Hits Person A (TP)
        [310, 140, 395, 310],  # Hits Person B (TP)
        [500, 100, 600, 250],  # Ghost detection (FP)
                               # Person C is missed (FN)
    ]

    tp, fp, fn, prec, rec, f1 = calculate_precision_recall(detections, ground_truths, iou_threshold=0.5)

    print("\n2. Classification Evaluation Metrics (IoU Threshold = 0.5):")
    print(f" -> Ground Truths: {len(ground_truths)} | Detections: {len(detections)}")
    print(f" -> True Positives (TP):  {tp}")
    print(f" -> False Positives (FP): {fp}")
    print(f" -> False Negatives (FN): {fn}")
    print(f" -> Precision:            {prec:.2%}")
    print(f" -> Recall:               {rec:.2%}")
    print(f" -> F1-Score:             {f1:.4f}")

    # Display Tracking Metrics
    print(tracking_metrics_explanation())

if __name__ == "__main__":
    main()
