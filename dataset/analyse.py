from typing import List

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import StanfordCars
from torchvision.transforms import transforms
from tqdm import tqdm

from dataset.train_val_split import VAL_ITEMS_IDX

BATCH_SIZE = 256


def analyse_class_imbalance(data_loaders: List[DataLoader], no_labels: int) -> None:
    fig, axes = plt.subplots(1, 3)
    axes[0].set_title('Train dataset')
    axes[1].set_title('Validation dataset')
    axes[2].set_title('Test dataset')


    for idx, data_loader in enumerate(data_loaders):
        occurences = torch.zeros(no_labels)

        with tqdm(data_loader, total=len(data_loader)) as samples:
            for batch in samples:
                occurences += torch.bincount(batch[1], minlength=no_labels)

        axes[idx].bar(
            [i for i in range(no_labels)],
            occurences.tolist()
        )

    plt.show()


if __name__ == '__main__':
    train_dataset = StanfordCars(
        root='./data',
        split='train',
        transform=transforms.Compose([
            transforms.Resize((1, 1)),
            transforms.ToTensor()
        ])
    )

    dataset = StanfordCars(
        root='./data',
        split='test',
        transform=transforms.Compose([
            transforms.Resize((1, 1)),
            transforms.ToTensor()
        ])
    )

    val_dataset = Subset(dataset, VAL_ITEMS_IDX)
    test_dataset = Subset(dataset, [idx for idx in range(len(dataset)) if idx not in VAL_ITEMS_IDX])

    analyse_class_imbalance(
        [
            DataLoader(train_dataset, BATCH_SIZE),
            DataLoader(val_dataset, BATCH_SIZE),
            DataLoader(test_dataset, BATCH_SIZE),
        ],
        len(dataset.class_to_idx)
    )
