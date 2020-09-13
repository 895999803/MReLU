from __future__ import division

import cv2
import torch
import random
import numbers

__all__ = ["Compose", "ToTensor", "Normalize", "Resize", "CenterCrop",  "RandomCrop", "RandomHorizontalFlip"]


_cv2_interpolation_to_str = {
    cv2.INTER_NEAREST: 'cv2.INTER_NEAREST',
    cv2.INTER_LINEAR: 'cv2.INTER_LINEAR',
    cv2.INTER_AREA: 'cv2.INTER_AREA',
    cv2.INTER_CUBIC: 'cv2.INTER_CUBIC',
    cv2.INTER_LANCZOS4: 'cv2.INTER_LANCZOS4',
}


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, array):
        """
        Args:
            np.array: array to be converted to tensor.

        Returns:
            Tensor: Converted array.
        """
        array = torch.from_numpy(array.transpose((2, 0, 1)))
        return array.float().div(255.)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor array of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor array.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Resize(object):
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

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        assert isinstance(size, int) or (isinstance(size, tuple) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, array):
        """
        Args:
            array : np.array to be scaled.

        Returns:
            np.array: Rescaled array.
        """
        if isinstance(self.size, int):
            h, w = array.shape[0], array.shape[1]
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return array
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return cv2.resize(array, (ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return cv2.resize(array, (ow, oh), self.interpolation)
        else:
            return cv2.resize(array, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _cv2_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class CenterCrop(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, array):
        """
        Args:
            np.array: array to be cropped.

        Returns:
            np.array: Cropped array.
        """
        h, w = array.shape[0], array.shape[1]
        ow, oh = self.size[0], self.size[1]
        i = (h-oh)//2
        j = (w-ow)//2

        return array[i:i+oh, j:j+ow]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(array, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            array : np array to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """

        h, w = array.shape[0], array.shape[1]
        tw, th = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, array):

        i, j, h, w = self.get_params(array, self.size)

        return array[i:i+h, j:j+w]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomHorizontalFlip(object):
    """Horizontally flip the given np.array randomly with a given probability.

    Args:
        p (float): probability of the np.array being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, array):
        """
        Args:
            array : np.array to be flipped.

        Returns:
            np.array: Randomly flipped np.array.
        """
        if random.random() < self.p:
            for index in range(array.shape[1] // 2):
                tmp = array[:, index].copy()
                array[:, index] = array[:, array.shape[1] - index - 1]
                array[:, array.shape[1] - index - 1] = tmp

        return array

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


if __name__ == '__main__':

    import scipy.misc
    image = scipy.misc.imread("2007_000039.jpg", mode="RGB")
    resize_img = Resize(320)(image)
    cop_img = CenterCrop(300)(image)
    random_cop_img = RandomCrop(300)(image)
    flip_img = RandomHorizontalFlip(p=1.0)(image)

    print(image.shape)
    print(resize_img.shape)
    print(cop_img.shape)
    print(random_cop_img.shape)
    print(flip_img.shape)

    cv2.imshow("image", image)
    cv2.imshow("resize_img", resize_img)
    cv2.imshow("cop_img", cop_img)
    cv2.imshow("random_cop_img", random_cop_img)
    cv2.imshow("flip_img", flip_img)
    cv2.waitKey(0)