import os
from typing import Tuple

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(model: nn.Module,
          train_data_loader: DataLoader,
          validation_data_loader: DataLoader,
          loss_fn: nn.Module,
          model_name: str,
          model_suffix: str,
          epochs: int = 100,
          learning_rate: float = 1e-2,
          device: str = 'cpu') -> nn.Module:
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min')

    epoch_losses = []
    for epoch in range(1, epochs + 1):
        train_epoch_loss = __train(model, train_data_loader, optimizer, loss_fn, device)
        validation_data_loss, validation_accuracy = __validate(model, validation_data_loader, loss_fn, device)
        scheduler.step(validation_data_loss)
        print(
            f'[{epoch}/{epochs}]\t'
            f'Train epoch loss: {train_epoch_loss}\t'
            f'Validation epoch loss: {validation_data_loss}\t'
            f'Validation accuracy: {validation_accuracy * 100}%')

        torch.save(
            model.state_dict(),
            os.path.join('models', model_name, 'weights', f'{model_suffix}_{epoch}.pth')
        )

        epoch_losses.append(validation_data_loss)
    return model


def __train(model: nn.Module,
            data_loader: DataLoader,
            optimizer,
            loss_fn,
            device: str = 'cpu') -> float:
    model = model.train()
    with tqdm(data_loader, total=len(data_loader)) as samples:
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(samples, start=1):
            imgs, targets = batch[0].to(device), batch[1].to(device)

            outputs = model(imgs)

            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            samples.set_postfix({
                'Epoch loss': epoch_loss / batch_idx
            })

        return epoch_loss / batch_idx


def __validate(model: nn.Module,
               data_loader: DataLoader,
               loss_fn,
               device: str = 'cpu') -> Tuple[float, float]:
    model = model.eval()
    with tqdm(data_loader, total=len(data_loader)) as samples:
        epoch_loss = 0.0
        epoch_correct_predictions = 0
        for batch_idx, batch in enumerate(samples, start=1):
            imgs, targets = batch[0].to(device), batch[1].to(device)

            outputs = model(imgs)

            loss = loss_fn(outputs, targets)
            epoch_loss += loss.item()

            predicted_class = torch.argmax(nn.functional.log_softmax(outputs, dim=1), dim=1)
            epoch_correct_predictions += (predicted_class == targets).int().sum().item()

            samples.set_postfix({
                'Epoch loss': epoch_loss / batch_idx
            })

        return epoch_loss / batch_idx, epoch_correct_predictions / len(data_loader.dataset)
