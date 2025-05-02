# dataset.py
import os
import ast
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class TileDataset(Dataset):
    def __init__(self, image_root, annotation_file, transform=None):
        self.image_root = image_root
        self.transform = transform
        self.annotations = pd.read_csv(annotation_file)

        # Remove 'indet' class
        self.annotations = self.annotations[self.annotations["paly_type"] != "indet"]
        self.image_names = sorted(self.annotations["tile_id"].unique())

        # Class mapping
        unique_classes = sorted(self.annotations["paly_type"].unique())
        self.class_mapping = {paly_type: i + 1 for i, paly_type in enumerate(unique_classes)}

    def __len__(self):
        return len(self.image_names)

    def get_index_by_tile(self, filename, tile_id):
        filename = str(filename).strip()
        tile_id = str(tile_id).strip()
        try:
            return self.image_names.index(tile_id)
        except ValueError:
            return None

    def parse_bboxes(self, image_name, idx):
        filtered_data = self.annotations[self.annotations["tile_id"] == image_name]
        boxes, labels = [], []

        for _, row in filtered_data.iterrows():
            tl = ast.literal_eval(row["TL"])
            br = ast.literal_eval(row["BR"])
            xmin, ymin = tl
            xmax, ymax = br

            if xmax > xmin and ymax > ymin:
                width = xmax - xmin
                height = ymax - ymin
                if width * height >= 16:  # Skip tiny boxes
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(self.class_mapping[row["paly_type"]])

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        return {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
        }

    def __getitem__(self, idx):
        image_tile = f"{self.image_names[idx]}/12z.png"
        row = self.annotations[self.annotations["tile_id"] == self.image_names[idx]].iloc[0]
        image_path = os.path.join(self.image_root, row["filename"], image_tile)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        target = self.parse_bboxes(self.image_names[idx], idx)
        return image, target

    def get_tile_metadata(self, idx):
        tile_id = self.image_names[idx]
        row = self.annotations[self.annotations["tile_id"] == tile_id].iloc[0]
        return row["filename"], tile_id
