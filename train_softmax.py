from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import copy
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tensorboardX import SummaryWriter

from dataset import RGBD_Dataset


def train_model(train_dataset, eval_dataset, pretrained=False, log_dir='./log', num_epochs=25, batch_size=16):
    if pretrained is True:
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    else:
        model = models.resnet18(pretrained=False)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    model_path = os.path.join(log_dir, 'model.pkl')
    if os.path.exists(model_path):
        if pretrained:
            print("pretrained parameters would be overwritten by {}".format(model_path))
        model = model.load_state_dict(torch.load(model_path))

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True),
        'val': DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True),
    }
    log_step_interval = 100
    num_of_classes = train_dataset.get_num_of_classes()

    # Parameters of newly constructed modules have requires_grad=True by default
    model.fc = torch.nn.Linear(model.fc.in_features, num_of_classes)
    criterion = torch.nn.CrossEntropyLoss()
    # Observe that only parameters of final layer are being optimized as opposed to before.
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    since = time.time()

    writer = SummaryWriter(log_dir)    # tensorboard writer
    best_model_wts = copy.deepcopy(model.state_dict())
    best_accu = 0.0

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            running_loss, running_corrects, total = 0, 0, 0
            p_bar = tqdm(total=len(dataloaders[phase]))

            for step, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward, track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                total += inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                _loss = float(running_loss) / total
                _accu = float(running_corrects) / total * 100

                p_bar.set_description('[{} on Epoch #{:d} Loss: {:.4f} Acc: {:.2f}%]'.format(
                    phase, epoch + 1, _loss, _accu))
                if (step + 1) % log_step_interval == 0:
                    if phase == 'train':
                        global_step = epoch * len(dataloaders['train']) + step
                        writer.add_scalar('data/train_loss', _loss, global_step)
                        writer.add_scalar('data/train_accu', _accu, global_step)
                p_bar.update(1)
            p_bar.close()

            epoch_loss = float(running_loss) / total
            epoch_accu = float(running_corrects) / total * 100

            if phase == 'train':
                scheduler.step(epoch=None)
            else:
                writer.add_scalar('data/val_loss', epoch_loss, epoch)
                writer.add_scalar('data/val_accu', epoch_accu, epoch)

            print('{} Loss: {:.4f} Acc: {:.2f}%'.format(
                phase, epoch_loss, epoch_accu))

            # deep copy the model
            if phase == 'val' and epoch_accu > best_accu:
                best_accu = epoch_accu
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, model_path)

    writer.close()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_accu))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    # model_path = '../logs/inception-resnet-v2.h5'
    log_dir = './logs/'
    batch_size = 16  # larger batch_size might cause segmentation fault
    num_epochs = 1000  # Number of epochs to run.
    steps_per_epoch = 2500  # You must specify the `steps_per_epoch` 'cause the training dataset was repeated
    input_image_size = (182, 182)
    input_shape = input_image_size + (4,)  # Model input shape

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(182),
        transforms.RandomHorizontalFlip(160),
        transforms.ToTensor(),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(182),
        transforms.CenterCrop(160),
        transforms.ToTensor(),
    ])

    train_dataset = RGBD_Dataset('~/vggface3d_sm/train.csv', transform=train_transform)
    eval_dataset = RGBD_Dataset('~/vggface3d_sm/eval.csv', transform=eval_transform)

    model_ft = train_model(train_dataset, eval_dataset, log_dir=log_dir,
                           num_epochs=num_epochs, batch_size=batch_size)

