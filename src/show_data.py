import matplotlib.pyplot as plt
from torchvision.datasets import OxfordIIITPet

ds = OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="segmentation",
    download=True,
)

image, mask = ds[0]

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask)
plt.title("Mask")
plt.axis("off")

plt.show()
