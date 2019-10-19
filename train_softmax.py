from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import argparse
import time
import copy
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

from dataset import RGBD_Dataset
from dataset import Resize, RandomHorizontalFlip
from models import ResNet50


def train_model(train_dataset, eval_dataset, input_channels, num_of_classes,
                num_of_epochs, batch_size, num_of_workers,
                log_base_dir, pretrained_on_imagenet=True,
                pretrained_model_path=None, pretrained_optim_path=None):
    r"""Train a Model
    Args:
        :param train_dataset: (RGBD_Dataset)
        :param eval_dataset: (RGBD_Dataset)
        :param input_channels: (int)
        :param num_of_classes: (int)
        :param pretrained_on_imagenet: (bool) Whether to load the imagenet pretrained model.
        :param log_base_dir: (str) The log directory to save logging files and models.
        :param num_of_epochs: (int) Number of times the dataset is traversed.
        :param batch_size: (int) The size of each step. large batch_size might cause segmentation fault
        :param num_of_workers: (int)
        :param pretrained_model_path: (str)
        :param pretrained_optim_path: (str)
        :return: the model of the best accuracy on validation dataset
    """
    # If you get such a RuntimeError, change the `num_workers=0` instead.
    # RuntimeError: DataLoader worker (pid 83641) is killed by signal: Unknown signal: 0
    print(num_of_workers)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_of_workers, drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_of_workers, drop_last=True)

    if num_of_classes is None:
        num_of_classes = train_dataset.get_num_of_classes()

    model = ResNet50(input_channels, num_of_classes, pretrained=pretrained_on_imagenet)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    # In `DataParallel` mode, it's to specify the leader to gather parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Observe that only parameters of final layer are being optimized as opposed to before
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    if (pretrained_model_path is not None) and os.path.exists(pretrained_model_path):
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        if pretrained_on_imagenet:
            print("Pretrained parameters would be overwritten by {}".format(pretrained_model_path))
        else:
            print("Model parameters is loaded from {}".format(pretrained_model_path))
    if (pretrained_optim_path is not None) and os.path.exists(pretrained_optim_path):
        optimizer.load_state_dict(torch.load(pretrained_optim_path, map_location=device))

    if not os.path.exists(log_base_dir):
        os.makedirs(log_base_dir)
    log_file = os.path.join(log_base_dir, 'training.log')
    log_fd = open(log_file, 'w')
    log_fd.write(f"""| {"epoch":^8s} | {"epoch_loss":^10s} | {"epoch_accu":^10s} """
                 f"""| {"best_accu":^10s} | {"time_elapsed":^15s} |\n""")
    log_step_interval = 100

    since = time.time()

    writer = SummaryWriter(log_base_dir)  # tensorboard writer
    best_model_wts = copy.deepcopy(model.state_dict())
    best_accu = 0.0
    saved_model_path = os.path.join(log_base_dir, 'resnet50-3d-model.pkl')
    saved_optim_path = os.path.join(log_base_dir, 'resnet50-3d-optim.pkl')

    for epoch in range(num_of_epochs):
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

        model.eval()
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
                     f"""| {best_accu:9.2f}% | {time_elapsed//60:7.0f} minutes |\n""")

    writer.close()
    log_fd.close()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_accu))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_csv', type=str,
                        help='The path of csv file where to write paths of training images.',
                        default='~/vggface3d_sm/train.csv')
    parser.add_argument('--eval_dataset_csv', type=str,
                        help='The path of csv file where to write paths of validation images.',
                        default='~/vggface3d_sm/eval.csv')
    parser.add_argument('--pretrained_on_imagenet',
                        help='(bool) Whether to load the imagenet pretrained model.', action='store_true')
    parser.add_argument('--pretrained_model_path', type=str,
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--pretrained_optim_path', type=str,
                        help='Load a optimizer before training starts.')
    parser.add_argument('--input_channels', type=int,
                        help='Number of channels of the first input layer.', default=4)
    parser.add_argument('--num_of_classes', type=int,
                        help='Number of channels of the last output layer.', default=1200)
    parser.add_argument('--num_of_epochs', type=int,
                        help='Number of epochs to run.', default=50)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=224)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=16)
    parser.add_argument('--num_of_workers', type=int,
                        help='Number of subprocesses to use for data loading.', default=0)
    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs and save models.', default='./logs/')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    from datetime import datetime
    TIMESTAMP = "{0:%m-%d.%H-%M}".format(datetime.now())
    args.logs_base_dir = args.logs_base_dir + TIMESTAMP

    train_transform = transforms.Compose([
        Resize(args.image_size),
        # transforms.RandomResizedCrop(224),
        RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    eval_transform = transforms.Compose([
        Resize(args.image_size),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_dataset = RGBD_Dataset(args.train_dataset_csv,
                                 input_channels=args.input_channels,
                                 transform=train_transform)
    eval_dataset = RGBD_Dataset(args.eval_dataset_csv,
                                input_channels=args.input_channels,
                                transform=eval_transform)

    model = train_model(train_dataset, eval_dataset,
                        input_channels=args.input_channels, num_of_classes=args.num_of_classes,
                        num_of_epochs=args.num_of_epochs, batch_size=args.batch_size,
                        num_of_workers=args.num_of_workers, log_base_dir=args.logs_base_dir,
                        pretrained_on_imagenet=True)
