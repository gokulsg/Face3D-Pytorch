from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import time
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import RGBD_Dataset
from dataset import Resize, RandomHorizontalFlip
from models import ResNet50


def evaluation(eval_dataset, pretrained_model_path, batch_size=16, num_of_workers=0, num_of_classes=1200):
    r"""Evaluation a Model
    Args:
        :param eval_dataset: (RGBD_Dataset)
        :param pretrained_model_path: (str) The path
        :param batch_size: (int) The size of each step. Large batch_size might cause segmentation fault
        :param num_of_workers: (int) Number of subprocesses to use for data loading.
        :param num_of_classes: (int) how many subprocesses to use for data loading.
        ``0`` means that the data will be loaded in the main process.
        :return: the model of the best accuracy on validation dataset
    """
    # If you get such a RuntimeError, change the `num_workers=0` instead.
    # RuntimeError: DataLoader worker (pid 83641) is killed by signal: Unknown signal: 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_of_workers, drop_last=True)

    model = ResNet50(eval_dataset.input_channels, num_of_classes, pretrained=False)

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
            loss = criterion(outputs, labels)
        # statistics
        _, preds = torch.max(outputs, 1)
        total += inputs.size(0)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        loss = float(running_loss) / total
        accu = float(running_corrects) / total * 100
        p_bar.set_description('[Evaluation Loss: {:.4f} Acc: {:.2f}%]'.format(loss, accu))
        p_bar.update(1)
    p_bar.close()
    acc = float(running_corrects) / total * 100
    loss = float(running_loss) / total

    time_elapsed = time.time() - since  # in seconds
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Inference Speed: {:2f} fps'.format(total / time_elapsed))
    print('Val Acc: {:4f}, Val loss: {:4f}'.format(acc, loss))
    return acc, loss


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset_csv', type=str,
                        help='The path of csv file where to write paths of test images.',
                        default='~/vggface3d_sm/test.csv')
    parser.add_argument('--pretrained_model_path', type=str,
                        help='The path of the pretrained model.')
    parser.add_argument('--input_channels', type=int,
                        help='Number of channels of the first input layer.', default=4)
    parser.add_argument('--num_of_classes', type=int,
                        help='Number of channels of the last output layer.', default=1200)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=224)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=16)
    parser.add_argument('--num_of_workers', type=int,
                        help='Number of subprocesses to use for data loading.', default=0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    eval_transform = transforms.Compose([
        Resize(args.image_size),
        transforms.ToTensor(),
    ])

    test_dataset = RGBD_Dataset(args.test_dataset_csv,
                                input_channels=args.input_channels,
                                transform=eval_transform)

    acc, loss = evaluation(test_dataset, batch_size=args.batch_size,
                           num_of_workers=args.num_of_workers,
                           pretrained_model_path=args.pretrained_model_path)
