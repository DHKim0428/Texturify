import os
from pathlib import Path
from torchvision.io import read_image
from torchvision.utils import make_grid, save_image
from torchvision import torch
from torchvision.transforms import transforms
from tqdm import tqdm


folder_list = [
    'baseline',
    'latent',
    'sum',
    'discriminator',
]

transform = transforms.Compose([
    transforms.ConvertImageDtype(dtype=torch.float)
])

# output_dir = Path('./runs/result_compare_car')
output_dir = Path('./runs/result_compare_chair')
output_grid = output_dir / "grid"
output_grid.mkdir(exist_ok=True, parents=True)

image_list = os.listdir(output_dir / folder_list[-1])
image_list.sort()

for i in tqdm(range(len(image_list) // 32), desc="image_grid"):
    for j in range(8):
        images = []
        for k in range(4):
            image_path = "{0}_{1}_{2}.jpg".format(i, j, k)
            for folder in folder_list:
                img = transform(read_image(str(output_dir / folder / image_path)))
                images.append(img)
        grid = make_grid(images, nrow=4)
        save_image(grid, str(output_grid / "{0}_{1}.jpg".format(i, j)))
