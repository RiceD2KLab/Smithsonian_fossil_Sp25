# utils.py
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_image(
    image_tensor,
    pred_boxes,
    pred_labels,
    pred_scores,
    reverse_class_mapping,
    draw_text=True,
    save_path=None,
    title=None
):
    image_pil = F.to_pil_image(image_tensor)
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_pil)

    # Draw only predicted boxes
    for j, (box, label, score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
        xmin, ymin, xmax, ymax = box.numpy()
        color = "orange"
        ax.add_patch(
            patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                edgecolor=color,
                facecolor='none',
                linewidth=2
            )
        )
        if draw_text:
            class_name = reverse_class_mapping.get(label.item(), "unknown")
            ax.text(
                xmin,
                ymin - 5,
                f"{class_name}: {score:.2f}",
                fontsize=8,
                color='black',
                backgroundcolor='white'
            )

    ax.set_title(title or "Predicted Boxes")
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

    plt.close()

def collate_fn(batch):
    return tuple(zip(*batch))
