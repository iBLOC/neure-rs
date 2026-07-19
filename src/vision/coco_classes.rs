//! COCO 2017 object detection class names.
//!
//! 80 classes. ID range [0, 79]. Used by YOLOv8 to label detections.

/// 80 COCO class names indexed by class ID (0..=79).
pub const COCO_CLASSES: &[&str] = &[
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
];

pub fn num_classes() -> usize {
    COCO_CLASSES.len()
}

pub fn class_name(class_id: u32) -> String {
    COCO_CLASSES
        .get(class_id as usize)
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("class_{class_id}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coco_has_80_classes() {
        assert_eq!(COCO_CLASSES.len(), 80);
        assert_eq!(num_classes(), 80);
    }

    #[test]
    fn test_coco_class_name_known() {
        assert_eq!(class_name(0), "person");
        assert_eq!(class_name(15), "cat");
        assert_eq!(class_name(79), "toothbrush");
    }

    #[test]
    fn test_coco_class_name_unknown_falls_back() {
        assert_eq!(class_name(80), "class_80");
        assert_eq!(class_name(999), "class_999");
    }
}
