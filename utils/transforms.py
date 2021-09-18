import numpy as np
from skimage.transform import rescale, rotate
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage.restoration import denoise_tv_chambolle
from torchvision.transforms import Compose


def transforms(
        flip_prob=0.5,
        scale_prob=0.5,
        rotate_prob=0.5,
        deform_prob=0.5,
        denoise_prob=None,
        ):
    transform_list: list = []

    if flip_prob is not None:
        transform_list.append(HorizontalFlip(prob=flip_prob))
    if scale_prob is not None:
        transform_list.append(Scale(prob=scale_prob))
    if rotate_prob is not None:
        transform_list.append(Rotate(prob=rotate_prob))
    if deform_prob is not None:
        transform_list.append(ElasticTransform(prob=deform_prob))
    if denoise_prob is not None:
        transform_list.append(TVDenoising(prob=denoise_prob))

    return Compose(transform_list)


class Scale(object):

    def __init__(self,
                 scale=0.05,
                 prob=0.5,
                 ):
        self.scale = scale
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.prob:
            return image, mask

        img_size = image.shape[0]

        scale = np.random.uniform(low=1.0 - self.scale, high=1.0 + self.scale)

        image = rescale(image, scale, multichannel=True, preserve_range=True, mode="constant", anti_aliasing=False)
        mask = rescale(
            mask, scale, order=0, multichannel=False, preserve_range=True, mode="constant", anti_aliasing=False)

        if scale < 1.0:
            diff = (img_size - image.shape[0]) / 2.0
            padding = ((int(np.floor(diff)), int(np.ceil(diff))),) * 2 + ((0, 0),)
            image = np.pad(image, padding, mode="constant", constant_values=0)
            mask = np.pad(mask, padding[0:2], mode="constant", constant_values=0)
        else:
            x_min = (image.shape[0] - img_size) // 2
            x_max = x_min + img_size
            image = image[x_min:x_max, x_min:x_max, ...]
            mask = mask[x_min:x_max, x_min:x_max, ...]

        return image, mask


class Rotate(object):

    def __init__(self,
                 angle=5,
                 prob=0.5,
                 ):
        self.angle = angle
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.prob:
            return image, mask

        angle = np.random.uniform(low=-self.angle, high=self.angle)
        image = rotate(image, angle, resize=False, preserve_range=True, mode="constant")
        mask = rotate(
            mask, angle, resize=False, order=0, preserve_range=True, mode="constant"
        )

        return image, mask


class HorizontalFlip(object):

    def __init__(self,
                 prob=0.5,
                 ):
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.prob:
            return image, mask

        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()

        return image, mask


class ElasticTransform(object):

    def __init__(self,
                 alpha=(0, 100),
                 sigma=(10, 12),
                 prob=0.5,
                 ):
        self.alpha = alpha
        self.sigma = sigma
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.prob:
            return image, mask

        alpha = np.random.uniform(low=self.alpha[0], high=self.alpha[1])
        sigma = np.random.uniform(low=self.sigma[0], high=self.sigma[1])
        random_state = np.random.RandomState(None)

        h, w = image.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        dx = gaussian_filter((random_state.rand(h, w) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(h, w) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        if len(image.shape) > 2:
            c = image.shape[2]
            distorted_image = [map_coordinates(image[:, :, i], indices, order=1, mode='reflect') for i in range(c)]
            distorted_image = np.concatenate(distorted_image, axis=1)
        else:
            distorted_image = map_coordinates(image, indices, order=1, mode='reflect')

        distorted_mask = map_coordinates(mask, indices, order=1, mode='reflect')

        image = distorted_image.reshape(image.shape)
        mask = distorted_mask.reshape(mask.shape)

        return image, mask


class TVDenoising(object):

    def __init__(self,
                 denoise_weight=0.1,
                 prob=0.5
                 ):
        self.denoise_weight = denoise_weight
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.prob:
            return image, mask

        weight = np.random.uniform(low=0, high=self.denoise_weight)
        image = denoise_tv_chambolle(image, weight=weight, multichannel=True)

        return image, mask
