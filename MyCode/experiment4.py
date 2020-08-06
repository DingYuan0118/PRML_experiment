from MyPackages.CIFAT10_Dataset import CIFAR10, Int2Tensor
import os
from torchvision import datasets, models, transforms
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from MyPackages.ConvNets import ForLayerConvNet
import argparse

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def train_model(model, criterion, optimizer, scheduler, writer, num_epochs=25):
    """
    train process for the model
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        epoch_start = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                scheduler.step()
                writer.add_scalar('training loss', epoch_loss, epoch)
                writer.add_scalar('training acc', epoch_acc, epoch)

            if phase == 'test':
                writer.add_scalar('test acc', epoch_acc, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        epoch_end = time.time()
        print('A epoch cost in {:.0f}m {:.0f}s'.format(
            (epoch_end - epoch_start) // 60, (epoch_end - epoch_start) % 60))
        print()
    # torch.save(best_model_wts, "resnet_18_classes_10.pth")
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--type', choices=['four_layer_conv', 'Resnet', 'Resnet_finetune'], help="please input ['four_layer_conv', 'Resnet', 'Resnet—finetune']")
    parse.add_argument('--bs', type=int, help='batch size for train and test')
    parse.add_argument('--norm', action='store_true', help="use weight decay norm")
    arg = parse.parse_args()

    root = os.environ['dataset']
    # reference = torchvision.datasets.CIFAR10(root)
    # print(reference)
    # print(reference[1])

    data_transform = transforms.ToTensor()
    label2tensor = Int2Tensor()
    train_data = CIFAR10(root, train=True, transform=data_transform, target_transform=label2tensor)
    test_data = CIFAR10(root, train=False, transform=data_transform, target_transform=label2tensor)

    print(train_data)
    print(test_data)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # init the for layer convolution network with output 10 class

    dataloaders = {"train": torch.utils.data.DataLoader(train_data, batch_size=arg.bs, shuffle=True, num_workers=0),
                  "test": torch.utils.data.DataLoader(test_data, batch_size=arg.bs, shuffle=True, num_workers=0)}

    class_names = train_data.classes
    class_to_idx = train_data.class_to_idx
    # inputs, classes = next(iter(dataloaders['train']))
    # Make a grid from batch
    dataset_sizes = {'train': len(train_data), 'test': len(test_data)}

    if arg.type == 'four_layer_conv':
        net = ForLayerConvNet(10)
        net.to(device)
        if arg.norm:
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-2)
            writer = SummaryWriter('runs/for_layer_conv_norm')
        else:
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
            writer = SummaryWriter('runs/for_layer_conv')

        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
        criterion = nn.CrossEntropyLoss()
        train_model(net, optimizer=optimizer, criterion=criterion, scheduler=scheduler, writer=writer, num_epochs=500)

    elif arg.type == 'Resnet':
        Resnet = models.resnet18(pretrained=False)
        num_ftrs = Resnet.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        Resnet.fc = nn.Linear(num_ftrs, 10)
        Resnet = Resnet.to(device)
        if arg.norm:
            # Observe that all parameters are being optimized
            optimizer = optim.SGD(Resnet.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-2)
            writer = SummaryWriter('runs/Resnet_norm_0.01')
        else:
            optimizer = optim.SGD(Resnet.parameters(), lr=0.001, momentum=0.9)
            writer = SummaryWriter('runs/Resnet')
        criterion = nn.CrossEntropyLoss()

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
        Resnet = train_model(Resnet, criterion, optimizer, exp_lr_scheduler, writer=writer,
                               num_epochs=100)
        torch.save(Resnet.state_dict(), "log/Resnet_18.pth")

    elif arg.type == 'Resnet_finetune':
        Resnet_ft = models.resnet18(pretrained=True)
        num_ftrs = Resnet_ft.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        Resnet_ft.fc = nn.Linear(num_ftrs, 10)
        Resnet_ft = Resnet_ft.to(device)
        criterion = nn.CrossEntropyLoss()
        # Observe that all parameters are being optimized
        optimizer = optim.SGD(Resnet_ft.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        writer = SummaryWriter('runs/Resnet_ft')
        Resnet_ft = train_model(Resnet_ft, criterion, optimizer, exp_lr_scheduler, writer=writer,
                               num_epochs=30)
        torch.save(Resnet_ft.state_dict(), "log/Resnet_18_ft.pth")

    else:
        raise ValueError("--type must be ['four_layer_conv', 'Resnet', 'Resnet—finetune']")

