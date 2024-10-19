from torch.nn import CrossEntropyLoss
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import StanfordCars
from torchvision.transforms import transforms

from dataset.train_val_split import VAL_ITEMS_IDX
from loss_functions import FocalLoss
from models.resnet.model import resnet18, resnet34, resnet50, resnet101, resnet152
from models.vgg.model import VGG11, VGG13, VGG16, VGG19
from models.inception.model import InceptionV1, InceptionV3
from train import train

DEVICE = 'cuda:0'
BATCH_SIZE = 64

if __name__ == '__main__':
    train_dataset = StanfordCars(
        root='./data',
        split='train',
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor()
        ])
    )

    test_dataset = StanfordCars(
        root='./data',
        split='test',
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
        ])
    )

    val_dataset = Subset(test_dataset, VAL_ITEMS_IDX)

    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_data_loader = DataLoader(val_dataset, batch_size=16)
    labels = len(train_dataset.class_to_idx.values())

    model = VGG11(classes=labels).to(DEVICE)
    model.init_weights()

    model = train(
        model,
        train_data_loader,
        validation_data_loader,
        CrossEntropyLoss(),
        'vgg',
        'vgg11_cross_entropy_loss',
        device=DEVICE,
        learning_rate=3e-4
    )
