# dataloader.py

import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import random

##############################
# 1) Surangi’s Baseline Model
##############################

class SurangiBaselineDataset(Dataset):
    """
    U-Net–style dataset for reproducing Surangi’s baseline approach.
    Returns (image, mask) for segmentation (binary or multi-class).
    """

    def __init__(
        self,
        csv_file: str,
        img_dir: str,
        nm_to_pix: float = 229.0,
        img_size: tuple = (512, 512),
        binary: bool = True
    ):
        """
        Args:
            csv_file: Path to CSV with columns [file, x, y, radius, pol_type, ...].
            img_dir: Directory containing .png (or .tif) tiles.
            nm_to_pix: Conversion factor from nanometers to pixels.
            img_size: (width, height) to resize images.
            binary: If True, produce a binary mask (pollen vs. background).
                    If False, produce multi-class mask using pol_type.
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.nm_to_pix = nm_to_pix
        self.img_size = img_size
        self.binary = binary

        # Group rows by 'file' so we can handle multiple pollen grains per image
        self.grouped = self.annotations.groupby("file")
        self.files = list(self.grouped.groups.keys())

        # If multi-class, map pol_type -> integer ID
        if not self.binary:
            unique_types = self.annotations["pol_type"].unique()
            # 0 = background, 1..N for each unique pol_type
            self.label_map = {ptype: i + 1 for i, ptype in enumerate(unique_types)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 1) Load image
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)

        # 2) Build segmentation mask
        H, W = self.img_size[1], self.img_size[0]
        mask = np.zeros((H, W), dtype=np.uint8)

        rows = self.grouped.get_group(fname)
        for _, row in rows.iterrows():
            cx = int(row["x"] / self.nm_to_pix)
            cy = int(row["y"] / self.nm_to_pix)
            r  = int(row["radius"] / self.nm_to_pix)

            if 0 <= cx < W and 0 <= cy < H:
                if self.binary:
                    # Mark pollen as 1
                    cv2.circle(mask, (cx, cy), r, 1, -1)
                else:
                    # Multi-class segmentation
                    class_id = self.label_map.get(row["pol_type"], 0)
                    cv2.circle(mask, (cx, cy), r, class_id, -1)

        # 3) Convert to torch tensors
        image_t = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask_t  = torch.tensor(mask, dtype=torch.long)  # (H, W)

        return image_t, mask_t


###############################################
# 2) Detection Models (Faster R-CNN, RetinaNet,
#    YOLO, Mask2Former in detection mode, etc.)
###############################################

class PollenDetectionDataset(Dataset):
    """
    Returns bounding boxes + labels for object detection.
    - Faster R-CNN / RetinaNet: (x_min, y_min, x_max, y_max) in pixel coords
    - YOLO: optionally (x_center, y_center, w, h) if needed
    - For Mask2Former instance seg, you’d produce instance masks per object.
    """

    def __init__(
        self,
        csv_file: str,
        img_dir: str,
        nm_to_pix: float = 229.0,
        img_size: tuple = (512, 512),
        label_map: dict = None,
        yolo_format: bool = False,
        mask2former: bool = False
    ):
        """
        Args:
            csv_file: Path to CSV [file, pol_type, x, y, radius].
            img_dir: Directory for images.
            nm_to_pix: nm -> px factor.
            img_size: (width, height) for resizing images.
            label_map: e.g. {"pol": 1, "spo": 2, ...}
            yolo_format: If True, output YOLO style [x_center, y_center, w, h] in [0,1].
            mask2former: If True, produce instance masks instead of bounding boxes.
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.nm_to_pix = nm_to_pix
        self.img_size = img_size
        self.label_map = label_map if label_map else {}
        self.yolo_format = yolo_format
        self.mask2former = mask2former

        # Group by file
        self.grouped = self.annotations.groupby("file")
        self.files = list(self.grouped.groups.keys())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 1) Load image
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)

        H, W = self.img_size[1], self.img_size[0]

        # 2) Collect bounding boxes or masks
        rows = self.grouped.get_group(fname)
        boxes, labels = [], []
        instance_masks = []

        for _, row in rows.iterrows():
            cx = row["x"] / self.nm_to_pix
            cy = row["y"] / self.nm_to_pix
            r  = row["radius"] / self.nm_to_pix

            # Convert circle -> bounding box
            x_min = max(cx - r, 0)
            y_min = max(cy - r, 0)
            x_max = min(cx + r, W - 1)
            y_max = min(cy + r, H - 1)

            # Class label
            pol_type = row["pol_type"]
            class_id = self.label_map.get(pol_type, 0)  # 0 = background?

            # YOLO or standard bounding box
            if self.yolo_format:
                # YOLO wants x_center, y_center, width, height normalized to [0,1]
                x_center = (x_min + x_max) / 2.0
                y_center = (y_min + y_max) / 2.0
                box_w    = (x_max - x_min)
                box_h    = (y_max - y_min)

                # Normalize
                x_center /= W
                y_center /= H
                box_w    /= W
                box_h    /= H

                boxes.append([class_id, x_center, y_center, box_w, box_h])
            else:
                # Standard detection (Faster R-CNN / RetinaNet)
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(class_id)

            # If instance segmentation for Mask2Former
            if self.mask2former:
                # Build a single-instance mask
                mask_i = np.zeros((H, W), dtype=np.uint8)
                # Draw circle for each instance
                cv2.circle(
                    mask_i,
                    (int(cx), int(cy)),
                    int(r),
                    1,  # label "1" in this single-instance mask
                    -1
                )
                instance_masks.append(mask_i)

        # 3) Convert to torch
        image_t = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        if self.mask2former:
            # For instance segmentation, each instance mask is separate
            # shape: (num_instances, H, W)
            if len(instance_masks) > 0:
                masks_t = torch.tensor(np.stack(instance_masks), dtype=torch.uint8)
            else:
                # no annotations? just empty
                masks_t = torch.zeros((0, H, W), dtype=torch.uint8)

            # Typically you also store "labels" for each instance
            labels_t = torch.tensor(labels, dtype=torch.int64)

            target = {
                "masks": masks_t,
                "labels": labels_t
            }
            # Optionally also store bounding boxes if needed
            # target["boxes"] = ...
            return image_t, target

        elif self.yolo_format:
            # YOLO format typically returns a list or array of shape (N, 5)
            # [class, x_center, y_center, w, h]
            target = torch.tensor(boxes, dtype=torch.float32)
            return image_t, target

        else:
            # Standard bounding box detection
            boxes_t  = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            target   = {"boxes": boxes_t, "labels": labels_t}
            return image_t, target


############################################
# 3) Few-Shot Learning Dataset
############################################

class PollenFewShotDataset(Dataset):
    """
    Creates few-shot episodes (support/query).
    Example: segmentation-based few-shot, returning (support_imgs, support_masks, query_imgs, query_masks).
    Adapt as needed for bounding boxes or meta-learning approaches.
    """

    def __init__(
        self,
        csv_file: str,
        img_dir: str,
        nm_to_pix: float = 229.0,
        img_size: tuple = (512, 512),
        k_shot: int = 5,
        q_size: int = 5
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.nm_to_pix = nm_to_pix
        self.img_size = img_size
        self.k_shot = k_shot
        self.q_size = q_size

        # Group by file
        self.grouped = self.annotations.groupby("file")
        self.files = list(self.grouped.groups.keys())

    def __len__(self):
        # The length might represent how many "episodes" you want
        return 1000  # arbitrary, or len(self.files)

    def __getitem__(self, idx):
        # 1) Randomly sample k_shot + q_size images
        selected = random.sample(self.files, self.k_shot + self.q_size)
        support_files = selected[:self.k_shot]
        query_files   = selected[self.k_shot:]

        # 2) Load each set
        support_imgs, support_masks = [], []
        for sf in support_files:
            img_t, mask_t = self._load_seg(sf)
            support_imgs.append(img_t)
            support_masks.append(mask_t)

        query_imgs, query_masks = [], []
        for qf in query_files:
            img_t, mask_t = self._load_seg(qf)
            query_imgs.append(img_t)
            query_masks.append(mask_t)

        # 3) Stack them
        support_imgs_t = torch.stack(support_imgs)  # (k_shot, 3, H, W)
        support_masks_t = torch.stack(support_masks) # (k_shot, H, W)
        query_imgs_t = torch.stack(query_imgs)       # (q_size, 3, H, W)
        query_masks_t = torch.stack(query_masks)     # (q_size, H, W)

        return {
            "support_imgs": support_imgs_t,
            "support_masks": support_masks_t,
            "query_imgs": query_imgs_t,
            "query_masks": query_masks_t
        }

    def _load_seg(self, fname):
        """
        Helper function to load a single image + binary segmentation mask.
        """
        img_path = os.path.join(self.img_dir, fname.replace(".ndpa", ".png"))
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)

        H, W = self.img_size[1], self.img_size[0]
        mask = np.zeros((H, W), dtype=np.uint8)

        rows = self.grouped.get_group(fname)
        for _, row in rows.iterrows():
            cx = int(row["x"] / self.nm_to_pix)
            cy = int(row["y"] / self.nm_to_pix)
            r  = int(row["radius"] / self.nm_to_pix)
            if 0 <= cx < W and 0 <= cy < H:
                cv2.circle(mask, (cx, cy), r, 1, -1)

        image_t = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask_t  = torch.tensor(mask, dtype=torch.long)

        return image_t, mask_t