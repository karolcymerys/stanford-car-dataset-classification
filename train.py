from torch.utils.data import Subset, DataLoader
from torchvision.datasets import StanfordCars
from torchvision.transforms import transforms

from dataset.train_val_split import TRAIN_ITEMS_IDX, VAL_ITEMS_IDX

DEVICE = 'cuda:0'
BATCH_SIZE = 256


if __name__ == '__main__':
    dataset = StanfordCars(
        root='./data',
        split='train',
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    )

    train_dataset = Subset(dataset, TRAIN_ITEMS_IDX)
    val_dataset = Subset(dataset, VAL_ITEMS_IDX)

    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    labels = len(dataset.class_to_idx.values())
