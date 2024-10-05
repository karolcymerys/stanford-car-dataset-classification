import math
from typing import Dict

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import StanfordCars
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_pil_image

BATCH_SIZE = 24

def plot_batch(imgs: torch.Tensor,
               predicted: torch.Tensor,
               expected: torch.Tensor,
               target_idx_to_class: Dict[int, str]) -> None:
    rows = 4
    cols = math.ceil(imgs.shape[0]/rows)

    fig, axes = plt.subplots(rows, cols)
    fig.set_figwidth(24)
    fig.set_figheight(12)

    for img_idx in range(imgs.shape[0]):
        img_row = math.floor(img_idx/cols)
        img_col = int(img_idx - img_row*cols)

        ax_img = axes[img_row, img_col]
        ax_img.imshow(to_pil_image(imgs[img_idx, :, :, :]))
        ax_img.set_axis_off()
        predicted_class = predicted[img_idx]
        expected_class = expected[img_idx]

        if predicted_class == expected_class:
            title = target_idx_to_class[predicted_class.item()]
            color = 'green'
        else:
            title = '{} \n ({})'.format(
                target_idx_to_class[predicted_class.item()],
                target_idx_to_class[expected_class.item()]
            )
            color = 'red'

        ax_img.set_title(
            title,
            fontdict = {
                'fontsize': 10,
                'color': color
            }
        )


    plt.show()


if __name__ == '__main__':
    dataset = StanfordCars(
        root='./data',
        split='test',
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
    )

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    class_idx_to_class_name = {
        class_id: class_name for class_name, class_id in dataset.class_to_idx.items()
    }

    for batch_imgs, batch_targets in data_loader:
        plot_batch(batch_imgs, batch_targets, batch_targets, class_idx_to_class_name)
