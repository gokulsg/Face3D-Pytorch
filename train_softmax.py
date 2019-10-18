from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import copy
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

from dataset import RGBD_Dataset
from dataset import Resize, RandomHorizontalFlip
from models import ResNet50


def train_model(train_dataset, eval_dataset, num_epochs=50, batch_size=16, log_dir='./log',
                pretrained_on_imagenet=True,
                pretrained_model_path=None, pretrained_optim_path=None):
    r"""Train a Model
    Args:
        :param train_dataset: (RGBD_Dataset)
        :param eval_dataset: (RGBD_Dataset)
        :param pretrained_on_imagenet: (bool) Whether to load the imagenet pretrained model.
        :param log_dir: (str) The log directory to save logging files and models.
        :param num_epochs: (int) Number of times the dataset is traversed.
        :param batch_size: (int) The size of each step.
        :param pretrained_model_path: (str)
        :param pretrained_optim_path: (str)
        :return: the model of the best accuracy on validation dataset
    """
    # If you get such a RuntimeError, change the `num_workers=0` instead.
    # RuntimeError: DataLoader worker (pid 83641) is killed by signal: Unknown signal: 0
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # num_of_classes = train_dataset.get_num_of_classes()
    num_of_classes = 1200

    model = ResNet50(train_dataset.input_channels, num_of_classes, pretrained=pretrained_on_imagenet)

    criterion = torch.nn.CrossEntropyLoss()
    # Observe that only parameters of final layer are being optimized as opposed to before.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if (pretrained_model_path is not None) and os.path.exists(pretrained_model_path):
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        if pretrained_on_imagenet:
            print("Pretrained parameters would be overwritten by {}".format(pretrained_model_path))
        else:
            print("Model parameters is loaded from {}".format(pretrained_model_path))
    if (pretrained_optim_path is not None) and os.path.exists(pretrained_optim_path):
        optimizer.load_state_dict(torch.load(pretrained_optim_path, map_location=device))

    model = model.to(device)

    log_file = os.path.join(log_dir, 'training.log')
    log_fd = open(log_file, 'w')
    log_fd.write(f"""| {"epoch":^8s} | {"epoch_loss":^10s} | {"epoch_accu":^10s} """
                 f"""| {"best_accu":^10s} | {"time_elapsed":^15s} |""")
    log_step_interval = 100

    since = time.time()

    writer = SummaryWriter(log_dir)  # tensorboard writer
    best_model_wts = copy.deepcopy(model.state_dict())
    best_accu = 0.0
    saved_model_path = os.path.join(log_dir, 'resnet50-3d-model.pkl')
    saved_optim_path = os.path.join(log_dir, 'resnet50-3d-optim.pkl')

    for epoch in range(num_epochs):
        model.train()

        running_loss, running_corrects, total = 0, 0, 0
        p_bar = tqdm(total=len(train_dataloader))
        for step, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            _, preds = torch.max(outputs, 1)
            # statistics
            total += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            _loss = float(running_loss) / total
            _accu = float(running_corrects) / total * 100

            p_bar.set_description('[TRAIN on Epoch #{:d} Loss: {:.4f} Acc: {:.2f}%]'.format(epoch + 1, _loss, _accu))
            if (step + 1) % log_step_interval == 0:
                global_step = epoch * len(train_dataloader) + step
                writer.add_scalar('data/train_loss', _loss, global_step)
                writer.add_scalar('data/train_accu', _accu, global_step)
            p_bar.update(1)
        p_bar.close()
        scheduler.step(epoch=None)

        running_loss, running_corrects, total = 0, 0, 0
        p_bar = tqdm(total=len(eval_dataloader))
        for step, (inputs, labels) in enumerate(eval_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            # statistics
            total += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            _loss = float(running_loss) / total
            _accu = float(running_corrects) / total * 100

            p_bar.set_description('[EVAL on Epoch #{:d} Loss: {:.4f} Acc: {:.2f}%]'.format(epoch + 1, _loss, _accu))
            p_bar.update(1)
        p_bar.close()
        epoch_loss = float(running_loss) / total
        epoch_accu = float(running_corrects) / total * 100

        writer.add_scalar('data/val_loss', epoch_loss, epoch)
        writer.add_scalar('data/val_accu', epoch_accu, epoch)

        # deep copy the model
        if epoch_accu > best_accu:
            best_accu = epoch_accu
            best_model_wts = copy.deepcopy(model.state_dict())
            optimizer.state_dict()
            torch.save(best_model_wts, saved_model_path)
            torch.save(optimizer, saved_optim_path)

        time_elapsed = time.time() - since
        log_fd.write(f"""| {epoch:8d} | {epoch_loss:10.2f} | {epoch_accu:9.2f}%"""
                     f"""| {best_accu:9.2f}% | {time_elapsed//60:7.0f} minutes |""")

    writer.close()
    log_fd.close()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_accu))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    from datetime import datetime
    TIMESTAMP = "{0:%m-%d.%H-%M}".format(datetime.now())
    log_dir = './tmp/' + TIMESTAMP
    batch_size = 16  # larger batch_size might cause segmentation fault
    num_epochs = 50  # Number of epochs to run.
    input_channels = 4

    train_transform = transforms.Compose([
        Resize(224),
        # transforms.RandomResizedCrop(224),
        RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    eval_transform = transforms.Compose([
        Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_dataset = RGBD_Dataset('~/vggface3d_sm/train.csv',
                                 input_channels=input_channels,
                                 transform=train_transform)
    eval_dataset = RGBD_Dataset('~/vggface3d_sm/eval.csv',
                                input_channels=input_channels,
                                transform=eval_transform)

    model = train_model(train_dataset, eval_dataset, log_dir=log_dir,
                        num_epochs=num_epochs, batch_size=batch_size)
