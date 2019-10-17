from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np

from torchvision import transforms
from skimage import transform


class Resize(object):
    """Resize the image in a sample to a given size.
    ** Strongly Recommended use this class only when dealing with RGB-D Images **

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            return transforms.Resize(self.output_size)(image)
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        image = transform.resize(image, (new_h, new_w)).astype(np.float32)
        return image


class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        """
        Args:
            image ( Image): Image to be flipped.

        Returns:
            Image: Randomly flipped image.
        """
        if not isinstance(image, np.ndarray):
            return transforms.RandomHorizontalFlip(p=self.p)(image)
        if random.random() < self.p:
            return np.fliplr(image).astype(np.float32)
        return image

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomCrop(object):
    """Crop randomly the image in a sample.
    ** Strongly Recommended use this class only when dealing with RGB-D Images **

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            return transforms.RandomCrop(self.output_size)
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h, left: left + new_w]
        return image
