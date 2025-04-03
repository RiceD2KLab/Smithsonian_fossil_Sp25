from dataloader import PollenFewShotDataset

dataset = PollenFewShotDataset(
    csv_file='/Users/brucew/Downloads/D3283-2_2024_02_06_15_37_28_Kentucky_annotations_transformed.csv',
    img_dir='/Users/brucew/Downloads/D3283-2_2024_02_06_15_37_28_Kentucky',
    k_shot=5,
    q_size=5
)

episode = dataset[0]
print("Support image shape:", episode["support_imgs"].shape)  # (5, 25, 3, H, W)
print("Query mask shape:", episode["query_masks"].shape)      # (5, 25, H, W)

# for visualize the center slice for debugging
import matplotlib.pyplot as plt

img = episode["support_imgs"][0, 12].permute(1, 2, 0).numpy()
mask = episode["support_masks"][0, 12].numpy()

plt.subplot(1, 2, 1)
plt.title("Support Slice z=12")
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.title("Mask z=12")
plt.imshow(mask, cmap="gray")
plt.show()
