import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import random
import ast

##################################
# 1) Surangi's Baseline Dataset
##################################

class SurangiBaselineDataset(Dataset):
    def __init__(self, csv_file, img_dir, img_size=(512, 512), binary=True):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.binary = binary

        self.grouped = self.annotations.groupby("tile")
        self.files = list(self.grouped.groups.keys())

        if not self.binary:
            unique_types = self.annotations["pol_type"].unique()
            self.label_map = {ptype: i + 1 for i, ptype in enumerate(unique_types)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        folder_name = self.files[idx]
        folder_path = os.path.join(self.img_dir, folder_name)

        images = []
        for z in range(25):
            img_file = f"{z}z.png"
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            images.append(img)

        image_stack = np.stack(images)
        H, W = self.img_size[1], self.img_size[0]
        mask = np.zeros((H, W), dtype=np.uint8)

        rows = self.grouped.get_group(folder_name)
        for _, row in rows.iterrows():
            center = ast.literal_eval(row["center"])
            cx = int(center[0] * W / 2048)
            cy = int(center[1] * H / 2048)
            r = int(row["radius_px"] * W / 2048)

            if 0 <= cx < W and 0 <= cy < H:
                if self.binary:
                    cv2.circle(mask, (cx, cy), r, 1, -1)
                else:
                    class_id = self.label_map.get(row["pol_type"], 0)
                    cv2.circle(mask, (cx, cy), r, class_id, -1)

        mask_stack = np.stack([mask] * 25)
        image_t = torch.tensor(image_stack, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        mask_t = torch.tensor(mask_stack, dtype=torch.long)

        return image_t, mask_t


###############################################
# 2) Detection Models Dataset
###############################################

class PollenDetectionDataset(Dataset):
    def __init__(self, csv_file, img_dir, img_size=(512, 512), label_map=None, yolo_format=False, mask2former=False):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_map = label_map if label_map else {}
        self.yolo_format = yolo_format
        self.mask2former = mask2former

        self.grouped = self.annotations.groupby("tile")
        self.files = list(self.grouped.groups.keys())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        folder_name = self.files[idx]
        folder_path = os.path.join(self.img_dir, folder_name)

        images = []
        for z in range(25):
            img_path = os.path.join(folder_path, f"{z}z.png")
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            images.append(img)

        image_stack = np.stack(images)
        image_t = torch.tensor(image_stack, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

        H, W = self.img_size[1], self.img_size[0]
        rows = self.grouped.get_group(folder_name)
        boxes, labels = [], []
        instance_masks = []

        for _, row in rows.iterrows():
            cx, cy = ast.literal_eval(row["center"])
            r = row["radius_px"]
            cx = cx * W / 2048
            cy = cy * H / 2048
            r = r * W / 2048

            x_min = max(cx - r, 0)
            y_min = max(cy - r, 0)
            x_max = min(cx + r, W - 1)
            y_max = min(cy + r, H - 1)

            class_id = self.label_map.get(row["pol_type"], 0)

            if self.yolo_format:
                x_center = (x_min + x_max) / 2.0 / W
                y_center = (y_min + y_max) / 2.0 / H
                w = (x_max - x_min) / W
                h = (y_max - y_min) / H
                boxes.append([class_id, x_center, y_center, w, h])
            else:
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(class_id)

            if self.mask2former:
                mask_i = np.zeros((H, W), dtype=np.uint8)
                cv2.circle(mask_i, (int(cx), int(cy)), int(r), 1, -1)
                instance_masks.append(mask_i)

        if self.mask2former:
            masks_t = torch.tensor(np.stack(instance_masks), dtype=torch.uint8) if instance_masks else torch.zeros((0, H, W), dtype=torch.uint8)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            return image_t, {"masks": masks_t, "labels": labels_t}

        elif self.yolo_format:
            return image_t, torch.tensor(boxes, dtype=torch.float32)

        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            return image_t, {"boxes": boxes_t, "labels": labels_t}


############################################
# 3) Few-Shot Learning Dataset
############################################

class PollenFewShotDataset(Dataset):
    def __init__(self, csv_file, img_dir, img_size=(512, 512), k_shot=5, q_size=5):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.k_shot = k_shot
        self.q_size = q_size

        self.grouped = self.annotations.groupby("tile")
        self.files = list(self.grouped.groups.keys())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        selected = random.sample(self.files, self.k_shot + self.q_size)
        support_files = selected[:self.k_shot]
        query_files = selected[self.k_shot:]

        support_imgs, support_masks = [], []
        for sf in support_files:
            img, mask = self._load_stack(sf)
            support_imgs.append(img)
            support_masks.append(mask)

        query_imgs, query_masks = [], []
        for qf in query_files:
            img, mask = self._load_stack(qf)
            query_imgs.append(img)
            query_masks.append(mask)

        return {
            "support_imgs": torch.stack(support_imgs),
            "support_masks": torch.stack(support_masks),
            "query_imgs": torch.stack(query_imgs),
            "query_masks": torch.stack(query_masks),
        }

    def _load_stack(self, folder_name):
        folder_path = os.path.join(self.img_dir, folder_name)
        images = []
        for z in range(25):
            img_path = os.path.join(folder_path, f"{z}z.png")
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Missing {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            images.append(img)

        image_stack = np.stack(images)
        H, W = self.img_size[1], self.img_size[0]
        mask = np.zeros((H, W), dtype=np.uint8)

        rows = self.grouped.get_group(folder_name)
        for _, row in rows.iterrows():
            cx, cy = ast.literal_eval(row["center"])
            r = row["radius_px"]
            cx = int(cx * W / 2048)
            cy = int(cy * H / 2048)
            r = int(r * W / 2048)
            cv2.circle(mask, (cx, cy), r, 1, -1)

        image_t = torch.tensor(image_stack, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        mask_t = torch.tensor(np.stack([mask] * 25), dtype=torch.long)

        return image_t, mask_t
