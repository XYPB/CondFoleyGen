import torch
import torchaudio
import torchaudio.functional
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.nn as nn
from PIL import Image
import numpy as np
import math
import random


class ResizeShortSide(object):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def __call__(self, x):
        '''
        x must be PIL.Image
        '''
        w, h = x.size
        short_side = min(w, h)
        w_target = int((w / short_side) * self.size)
        h_target = int((h / short_side) * self.size)
        return x.resize((w_target, h_target))


class RandomResizedCrop3D(nn.Module):
    """Crop the given series of images to random size and aspect ratio.
    The image can be a PIL Images or a Tensor, in which case it is expected
    to have [N, ..., H, W] shape, where ... means an arbitrary number of leading dimensions

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
      size (int or sequence): expected output size of each edge. If size is an
        int instead of sequence like (h, w), a square output size ``(size, size)`` is
        made. If provided a tuple or list of length 1, it will be interpreted as (size[0], size[0]).
      scale (tuple of float): range of size of the origin size cropped
      ratio (tuple of float): range of aspect ratio of the origin aspect ratio cropped.
      interpolation (int): Desired interpolation enum defined by `filters`_.
        Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
        and ``PIL.Image.BICUBIC`` are supported.
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=transforms.InterpolationMode.BILINEAR):
        super().__init__()
        if isinstance(size, tuple) and len(size) == 2:
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
          img (PIL Image or Tensor): Input image.
          scale (list): range of scale of the origin size cropped
          ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
          tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        width, height = img.size
        area = height * width

        for _ in range(10):
            target_area = area * \
                torch.empty(1).uniform_(scale[0], scale[1]).item()
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, imgs):
        """
        Args:
          img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
          PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(imgs[0], self.scale, self.ratio)
        return [F.resized_crop(img, i, j, h, w, self.size, self.interpolation) for img in imgs]


class Resize3D(object):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def __call__(self, imgs):
        '''
        x must be PIL.Image
        '''
        return [x.resize((self.size, self.size)) for x in imgs]


class RandomHorizontalFlip3D(object):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, imgs):
        '''
        x must be PIL.Image
        '''
        if np.random.rand() < self.p:
            return [x.transpose(Image.FLIP_LEFT_RIGHT) for x in imgs]
        else:
            return imgs


class ColorJitter3D(torch.nn.Module):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
    brightness (float or tuple of float (min, max)): How much to jitter brightness.
        brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
        or the given [min, max]. Should be non negative numbers.
    contrast (float or tuple of float (min, max)): How much to jitter contrast.
        contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
        or the given [min, max]. Should be non negative numbers.
    saturation (float or tuple of float (min, max)): How much to jitter saturation.
        saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
        or the given [min, max]. Should be non negative numbers.
    hue (float or tuple of float (min, max)): How much to jitter hue.
        hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
        Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = (1-brightness, 1+brightness)
        self.contrast = (1-contrast, 1+contrast)
        self.saturation = (1-saturation, 1+saturation)
        self.hue = (0-hue, 0+hue)

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        tfs = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            tfs.append(transforms.Lambda(
                lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            tfs.append(transforms.Lambda(
                lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            tfs.append(transforms.Lambda(
                lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            tfs.append(transforms.Lambda(
                lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(tfs)
        transform = transforms.Compose(tfs)

        return transform

    def forward(self, imgs):
        """
        Args:
          img (PIL Image or Tensor): Input image.

        Returns:
          PIL Image or Tensor: Color jittered image.
        """
        transform = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue)
        return [transform(img) for img in imgs]


class ToTensor3D(object):
    def __init__(self):
        super().__init__()

    def __call__(self, imgs):
        '''
        x must be PIL.Image
        '''
        return [F.to_tensor(img) for img in imgs]


class Normalize3D(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, imgs):
        '''
        x must be PIL.Image
        '''
        return [F.normalize(img, self.mean, self.std, self.inplace) for img in imgs]


class CenterCrop3D(object):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def __call__(self, imgs):
        '''
        x must be PIL.Image
        '''
        return [F.center_crop(img, self.size) for img in imgs]


class FrequencyMasking(object):
    def __init__(self, freq_mask_param: int, iid_masks: bool = False):
        super().__init__()
        self.masking = torchaudio.transforms.FrequencyMasking(freq_mask_param, iid_masks)

    def __call__(self, item):
        if 'cond_image' in item.keys():
            batched_spec = torch.stack(
                [torch.tensor(item['image']), torch.tensor(item['cond_image'])], dim=0
            )[:, None] # (2, 1, H, W)
            masked = self.masking(batched_spec).numpy()
            item['image'] = masked[0, 0]
            item['cond_image'] = masked[1, 0]
        elif 'image' in item.keys():
            inp = torch.tensor(item['image'])
            item['image'] = self.masking(inp).numpy()
        else:
            raise NotImplementedError()
        return item


class TimeMasking(object):
    def __init__(self, time_mask_param: int, iid_masks: bool = False):
        super().__init__()
        self.masking = torchaudio.transforms.TimeMasking(time_mask_param, iid_masks)

    def __call__(self, item):
        if 'cond_image' in item.keys():
            batched_spec = torch.stack(
                [torch.tensor(item['image']), torch.tensor(item['cond_image'])], dim=0
            )[:, None] # (2, 1, H, W)
            masked = self.masking(batched_spec).numpy()
            item['image'] = masked[0, 0]
            item['cond_image'] = masked[1, 0]
        elif 'image' in item.keys():
            inp = torch.tensor(item['image'])
            item['image'] = self.masking(inp).numpy()
        else:
            raise NotImplementedError()
        return item
