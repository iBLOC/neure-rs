//! Non-Max Suppression (NMS) for object detection postprocessing.
//!
//! Pure-Rust implementation. Operates on xyxy bounding boxes (top-left +
//! bottom-right corners). Per-class independent suppression.

use super::BBox;

/// Compute Intersection over Union (IoU) of two axis-aligned bounding boxes.
pub fn iou(a: &BBox, b: &BBox) -> f32 {
    let ax2 = a.x + a.w;
    let ay2 = a.y + a.h;
    let bx2 = b.x + b.w;
    let by2 = b.y + b.h;

    let inter_x1 = a.x.max(b.x);
    let inter_y1 = a.y.max(b.y);
    let inter_x2 = ax2.min(bx2);
    let inter_y2 = ay2.min(by2);

    let iw = (inter_x2 - inter_x1).max(0.0);
    let ih = (inter_y2 - inter_y1).max(0.0);
    let inter = iw * ih;
    if inter <= 0.0 {
        return 0.0;
    }
    let area_a = a.w * a.h;
    let area_b = b.w * b.h;
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        return 0.0;
    }
    inter / union
}

/// One detection candidate, sorted by score for NMS.
#[derive(Debug, Clone, Copy)]
pub struct Candidate {
    pub score: f32,
    pub class_id: u32,
    pub bbox: BBox,
}

/// Apply per-class NMS, returning the indices of kept candidates (in input order).
///
/// `iou_threshold` is the IoU above which a lower-scored candidate is suppressed.
/// Candidates with the same `class_id` are only compared against each other.
pub fn nms(candidates: &[Candidate], iou_threshold: f32) -> Vec<usize> {
    if candidates.is_empty() {
        return Vec::new();
    }
    // Sort indices by score descending
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_by(|&a, &b| candidates[b].score.partial_cmp(&candidates[a].score).unwrap_or(std::cmp::Ordering::Equal));

    let mut suppressed = vec![false; candidates.len()];
    let mut kept: Vec<usize> = Vec::with_capacity(candidates.len());

    for &i in &order {
        if suppressed[i] {
            continue;
        }
        kept.push(i);
        let a = &candidates[i];
        for &j in &order {
            if j == i || suppressed[j] {
                continue;
            }
            let b = &candidates[j];
            if a.class_id != b.class_id {
                continue;
            }
            if iou(&a.bbox, &b.bbox) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bbox(x: f32, y: f32, w: f32, h: f32) -> BBox {
        BBox { x, y, w, h }
    }

    #[test]
    fn test_iou_identical_boxes() {
        let a = bbox(0.0, 0.0, 10.0, 10.0);
        let b = bbox(0.0, 0.0, 10.0, 10.0);
        assert!((iou(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_iou_disjoint_boxes() {
        let a = bbox(0.0, 0.0, 10.0, 10.0);
        let b = bbox(20.0, 20.0, 10.0, 10.0);
        assert!(iou(&a, &b) < 1e-6);
    }

    #[test]
    fn test_iou_half_overlap() {
        let a = bbox(0.0, 0.0, 10.0, 10.0);
        let b = bbox(5.0, 0.0, 10.0, 10.0);
        // Intersection = 5*10 = 50, union = 100+100-50 = 150
        assert!((iou(&a, &b) - 50.0 / 150.0).abs() < 1e-6);
    }

    #[test]
    fn test_nms_removes_overlapping_boxes() {
        let candidates = vec![
            Candidate { score: 0.9, class_id: 0, bbox: bbox(0.0, 0.0, 10.0, 10.0) },
            Candidate { score: 0.8, class_id: 0, bbox: bbox(1.0, 1.0, 10.0, 10.0) }, // overlaps
            Candidate { score: 0.7, class_id: 0, bbox: bbox(0.0, 0.0, 10.0, 10.0) }, // same
        ];
        let kept = nms(&candidates, 0.5);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0], 0); // highest score
    }

    #[test]
    fn test_nms_keeps_disjoint_boxes() {
        let candidates = vec![
            Candidate { score: 0.9, class_id: 0, bbox: bbox(0.0, 0.0, 10.0, 10.0) },
            Candidate { score: 0.8, class_id: 0, bbox: bbox(50.0, 50.0, 10.0, 10.0) },
        ];
        let kept = nms(&candidates, 0.5);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_nms_per_class_independent() {
        let candidates = vec![
            Candidate { score: 0.9, class_id: 0, bbox: bbox(0.0, 0.0, 10.0, 10.0) },
            Candidate { score: 0.8, class_id: 1, bbox: bbox(0.0, 0.0, 10.0, 10.0) }, // same bbox, different class
        ];
        let kept = nms(&candidates, 0.5);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_nms_iou_threshold_zero_still_suppresses_perfect_overlap() {
        // Standard NMS uses strict `>` so even at threshold 0, a perfect
        // overlap (iou=1.0 > 0.0) suppresses the lower-scored box.
        let candidates = vec![
            Candidate { score: 0.9, class_id: 0, bbox: bbox(0.0, 0.0, 10.0, 10.0) },
            Candidate { score: 0.8, class_id: 0, bbox: bbox(0.0, 0.0, 10.0, 10.0) },
        ];
        let kept = nms(&candidates, 0.0);
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn test_nms_iou_threshold_one_keeps_all_non_overlapping() {
        // With threshold = 1.0, no iou can exceed 1.0, so disjoint boxes
        // are always kept.
        let candidates = vec![
            Candidate { score: 0.9, class_id: 0, bbox: bbox(0.0, 0.0, 10.0, 10.0) },
            Candidate { score: 0.8, class_id: 0, bbox: bbox(50.0, 50.0, 10.0, 10.0) },
        ];
        let kept = nms(&candidates, 1.0);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_nms_empty_input() {
        let kept = nms(&[], 0.5);
        assert!(kept.is_empty());
    }
}
