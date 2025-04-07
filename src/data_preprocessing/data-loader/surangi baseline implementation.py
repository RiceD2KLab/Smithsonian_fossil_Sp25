from dataloader import SurangiBaselineDataset
from torch.utils.data import DataLoader

dataset = SurangiBaselineDataset(
    csv_file='path/to/annotations.csv',
    img_dir='path/to/image_folders',
    binary=True  # or False for multi-class
)

loader = DataLoader(dataset, batch_size=1, shuffle=True)

for image_stack, mask_stack in loader:
    print("Image stack shape:", image_stack.shape)  # (B, 25, 3, H, W)
    print("Mask stack shape:", mask_stack.shape)    # (B, 25, H, W)
    break