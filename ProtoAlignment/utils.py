

import time
import os
import string
from PIL import Image
import torch

def add_version_path(out_dir, timestamp=False, is_letter = False, timestamp_id = False):

    ''' Add a letter to the output directory of the summaries to avoid overwriting if several jobs are run at the same time'''

    if timestamp:
        out_dir = out_dir + time.strftime("%y%m%d_%H%M", time.gmtime())
    if timestamp_id:
        # job_id  = os.environ.get('LSB_JOBID')
        time_ = time.strftime("%y%m%d_%H%M", time.gmtime())
        job_index = os.environ.get('LSB_JOBINDEX')
        if job_index is not None:
            out_dir = f'{out_dir}{time_}.{job_index}'
        else:
            out_dir = f'{out_dir}{time_}'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            return out_dir

    i = 0
    letter = '0'

    if is_letter:
        list_versions = string.ascii_lowercase
    else:
        list_versions = [str(x) for x in range(1,100000)]
    created = False

    while not created:
        if not os.path.exists(out_dir + letter):
            try:
                os.makedirs(out_dir + letter)
                created = True
            except OSError:
                pass
        # Check if the folder contains any kind of file
        elif len([name for name in os.listdir(out_dir + letter) if os.path.isfile(os.path.join(out_dir + letter, name)) or os.path.isdir(os.path.join(out_dir + letter, name))]) == 0:
            created = True
        else:
            letter = list_versions[i]
            i += 1
    return out_dir + letter


import torchvision.transforms as transforms
from torchvision.transforms import functional as F
 
class ResizeNoCrop(transforms.Resize):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, color=255,colorBW=0,**kwargs):
        super().__init__(**kwargs)
        self.color = color
        self.colorBW = colorBW


    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """

        scale = self.size / max(img.size)

        array0 = F.resize(img, (int(scale * img.size[1]), int(scale * img.size[0])), self.interpolation)

        if img.mode=="RGB":
            array1 = Image.new('RGB', (self.size, self.size), (self.color, self.color, self.color))
        else:
            array1 = Image.new('L', (self.size, self.size), (self.colorBW,))

        array1.paste(array0, (int((self.size - array0.size[0]) / 2), int((self.size - array0.size[1]) / 2)))

        return array1




class NormalizeRBGonly(transforms.Normalize):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if tensor.shape[0] == 3:
            return F.normalize(tensor, self.mean, self.std, self.inplace)
        else:
            return tensor





class ToTensorIfNotTensor:
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor. This transform does not support torchscript.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.

    .. note::
        Because the input image is scaled to [0.0, 1.0], this transformation should not be used when
        transforming target image masks. See the `references`_ for implementing the transforms for image masks.

    .. _references: https://github.com/pytorch/vision/tree/main/references/segmentation
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """

        if not isinstance(pic, torch.Tensor):
            return F.to_tensor(pic)
        else:
            return pic

    def __repr__(self):
        return self.__class__.__name__ + '()'

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


