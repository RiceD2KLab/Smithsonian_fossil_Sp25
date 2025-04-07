# Pollen Few-Shot Detection & Segmentation

This repository provides PyTorch `Dataset` classes for training deep learning models on pollen grain microscopy stacks. It supports semantic segmentation, object detection, and few-shot learning, all using image volumes (z-stacks) with per-object annotations.

---

## Folder Structure

Each image sample is a folder named like:  
```
6365x_55347y/
├── 0z.png
├── 1z.png
├── ...
└── 24z.png
```

Each folder contains 25 focal plane slices (z=0 to z=24) of a 2048×2048 RGB microscopy image.

---

## Annotation Format

Your `CSV` file must include columns like:

| Column       | Description                             |
|--------------|-----------------------------------------|
| `tile`       | Folder name (e.g., `6365x_55347y`)      |
| `center`     | String like `"(x, y)"` for center point |
| `radius_px`  | Circle radius in **pixels**             |
| `pol_type`   | Label name (e.g., `"paly"`, `"trip"`)   |

---

## Dataset Classes

### `SurangiBaselineDataset`
- For semantic segmentation (binary or multi-class)
- Output: `(25, 3, H, W)` image tensor + `(25, H, W)` mask tensor

### `PollenDetectionDataset`
- For detection models like Faster R-CNN, YOLO, Mask2Former
- Output: image tensor + dictionary of boxes, labels, or masks

### `PollenFewShotDataset`
- For few-shot segmentation (e.g., 5-shot 5-query)
- Output: dictionary with support/query image-mask sets

---

## Example Usage

```python
from dataloader import PollenFewShotDataset

dataset = PollenFewShotDataset(
    csv_file='annotations.csv',
    img_dir='data/',
    k_shot=5,
    q_size=5
)

episode = dataset[0]
print("Support image shape:", episode["support_imgs"].shape)
print("Query mask shape:", episode["query_masks"].shape)
```

---

## Visualization Tip

```python
import matplotlib.pyplot as plt

img = episode["support_imgs"][0, 12].permute(1, 2, 0).numpy()
mask = episode["support_masks"][0, 12].numpy()

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Support Slice z=12")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap='gray')
plt.title("Mask z=12")
plt.show()
```

---

## Getting Started

### Install dependencies:
```bash
pip install torch pandas opencv-python numpy
```

---

## Notes

- All inputs are automatically resized to `512x512` by default.
- Radius and coordinates are scaled from original 2048x2048 dimensions.
- You can modify `img_size` in any dataset constructor.