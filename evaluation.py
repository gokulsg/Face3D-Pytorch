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


def evaluation(eval_dataset, batch_size=16, num_of_classes=1200, pretrained_model_path=None):
    r"""Train a Model
    Args:
        :param eval_dataset: (RGBD_Dataset)
        :param batch_size: (int) The size of each step.
        :param num_of_classes: (int)
        :param pretrained_model_path: (str)
        :return: the model of the best accuracy on validation dataset
    """
    # If you get such a RuntimeError, change the `num_workers=0` instead.
    # RuntimeError: DataLoader worker (pid 83641) is killed by signal: Unknown signal: 0
    assert pretrained_model_path is not None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = ResNet50(eval_dataloader.input_channels, num_of_classes, pretrained=False)

    if os.path.exists(pretrained_model_path):
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        print("Model parameters is loaded from {}".format(pretrained_model_path))
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    since = time.time()

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
    acc = float(running_corrects) / total * 100
    loss = float(running_loss) / total

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Val Acc: {:4f}, Val loss: {:4f}'.format(acc, loss))

    return acc, loss


if __name__ == '__main__':
    batch_size = 16  # larger batch_size might cause segmentation fault
    input_channels = 4

    # train_transform = transforms.Compose([
    #     Resize(224),
    #     # transforms.RandomResizedCrop(224),
    #     RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    # ])
    eval_transform = transforms.Compose([
        Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    test_dataset = RGBD_Dataset('~/vggface3d_sm/test.csv',
                                input_channels=input_channels,
                                transform=eval_transform)

    model = evaluation(test_dataset, batch_size=batch_size)
