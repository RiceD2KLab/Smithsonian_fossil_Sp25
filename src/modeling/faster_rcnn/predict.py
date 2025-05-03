# predict.py
import os
import torch
import torchvision.ops as ops
from PIL import Image
from torchvision import transforms
from .utils import draw_image

# Global transform
transform = transforms.Compose([transforms.ToTensor()])

def run_on_tile_multi_focus(model, filename, tile_id, device, confidence_threshold=0.8, iou_threshold=0.5, nms_iou_threshold=0.5, draw=True):
    model.eval()

    all_preds_boxes, all_preds_scores, all_preds_labels = [], [], []
    images = []

    for focus in range(24):  # Focal planes 0z.png - 12z.png
        image_path = os.path.join(filename, tile_id, f"{focus}z.png")
        if not os.path.exists(image_path):
            continue

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).to(device)
        images.append(image_tensor)

        with torch.no_grad():
            pred = model([image_tensor])[0]

        boxes = pred["boxes"].cpu()
        scores = pred["scores"].cpu()
        labels = pred["labels"].cpu()

        keep = scores > confidence_threshold
        if keep.sum() > 0:
            all_preds_boxes.append(boxes[keep])
            all_preds_scores.append(scores[keep])
            all_preds_labels.append(labels[keep])

    if not all_preds_boxes:
        print(f"No detections above threshold for {filename}/{tile_id}")
        return None, None, None

    # Stack predictions across focal planes
    combined_boxes = torch.cat(all_preds_boxes)
    combined_scores = torch.cat(all_preds_scores)
    combined_labels = torch.cat(all_preds_labels)

    # Apply Non-Maximum Suppression
    keep_indices = ops.nms(combined_boxes, combined_scores, nms_iou_threshold)

    final_boxes = combined_boxes[keep_indices]
    final_scores = combined_scores[keep_indices]
    final_labels = combined_labels[keep_indices]

    # Optionally draw predictions on the middle focal plane
    if draw and images:
        reverse_class_mapping = {'alg': 1, 'fun': 2, 'pol': 3, 'spo': 4}
        mid_focus_image = images[len(images) // 2].cpu()

        draw_image(
            mid_focus_image,
            final_boxes,
            final_labels,
            final_scores,
            reverse_class_mapping=reverse_class_mapping,
            draw_text=True,
            title=f"Predictions: {filename}/{tile_id}"
        )

    return final_boxes, final_labels, final_scores
