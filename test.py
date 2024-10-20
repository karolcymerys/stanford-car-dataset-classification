import torch
from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader
from torchvision.datasets import StanfordCars
from torchvision.transforms import transforms
from tqdm import tqdm

from dataset.visualize import plot_batch
from models.resnet.model import resnet18, resnet34, resnet50, resnet101, resnet152
from models.vgg.model import VGG11, VGG13, VGG16, VGG19

DEVICE = 'cuda:0'
BATCH_SIZE = 32

if __name__ == '__main__':
    test_dataset = StanfordCars(
        root='./data',
        split='test',
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    )

    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    labels = len(test_dataset.class_to_idx.values())

    model = resnet34(classes=labels).to(DEVICE)
    model.load_state_dict(torch.load('./models/resnet/weights/resnet34_cross_entropy_loss.pth'))
    model = model.eval()

    class_idx_to_class_name = {
        class_id: class_name for class_name, class_id in test_dataset.class_to_idx.items()
    }


    with tqdm(test_data_loader, total=len(test_data_loader)) as samples:
        for batch_idx, batch in enumerate(samples, start=1):
            imgs, targets = batch[0].to(DEVICE), batch[1].to(DEVICE)

            outputs = model(imgs)
            predicted_classes = torch.argmax(log_softmax(outputs, dim=1), dim=1)

            plot_batch(imgs, predicted_classes, targets, class_idx_to_class_name)
