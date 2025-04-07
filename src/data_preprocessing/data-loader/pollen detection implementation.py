from dataloader import PollenDetectionDataset

dataset = PollenDetectionDataset(
    csv_file='path/to/annotations.csv',
    img_dir='path/to/image_folders',
    yolo_format=False,
    mask2former=False
)

image_stack, target = dataset[0]
print("Image stack shape:", image_stack.shape)     # (25, 3, H, W)
print("Target:", target)                           # Dictionary with 'boxes' and 'labels'