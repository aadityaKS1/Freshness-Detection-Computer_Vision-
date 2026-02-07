import cv2

def draw_result(image, bbox, label, text):
    """
    image : original image
    bbox  : (x1, y1, x2, y2)
    label : detected object name (e.g., banana)
    text  : freshness output (e.g., Fresh/0.82)
    """

    x1, y1, x2, y2 = bbox

    # Normalize separator
    text = text.replace("?", "|")

    # Final display text
    display_text = f"{label} | {text}"

    # -------- Dynamic scaling based on image size --------
    h, w = image.shape[:2]
    scale = max(h, w) / 1000.0

    font_scale = max(0.35, 0.7 * scale)
    thickness = max(1, int(2 * scale))

    # Draw bounding box
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        thickness
    )

    # Draw text
    cv2.putText(
        image,
        display_text,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA
    )
