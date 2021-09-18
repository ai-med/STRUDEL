import numpy as np
import nibabel as nib

from nibabel.orientations import axcodes2ornt, ornt_transform


def standardize(img):
    if img.ndim == 3:
        mean = img.mean(axis=(0, 1, 2))
        std = img.std(axis=(0, 1, 2))
    else:
        mean = img.mean(axis=(0, 1))
        std = img.std(axis=(0, 1))
    return (img - mean) / std


def normalize(input):
    return (input - input.min()) / (input.max() - input.min())


def crop(img, crop_x, crop_y):
    if img.ndim == 2:
        y, x = img.shape
    elif img.ndim == 3:
        y, x, _ = img.shape
    else:
        raise ValueError('dimension of N-d array must be either 2 or 3')

    y_new = crop_y if y > crop_y else y
    x_new = crop_x if x > crop_x else x
    start_x = x // 2 - (x_new // 2)
    start_y = y // 2 - (y_new // 2)
    if img.ndim == 2:
        return img[start_y:start_y + crop_y, start_x:start_x + crop_x]
    else:
        return img[start_y:start_y + crop_y, start_x:start_x + crop_x, :]


def resize(img, crop_x, crop_y):
    if img.ndim == 2:
        y, x = img.shape
    elif img.ndim == 3:
        y, x, z = img.shape
    else:
        raise ValueError('dimension of N-d array must be either 2 or 3')

    y_new = crop_y if y < crop_y else y
    x_new = crop_x if x < crop_x else x
    if y != y_new or x != x_new:
        offset_y = (y_new - y) // 2
        offset_x = (x_new - x) // 2
        if img.ndim == 2:
            new_img = np.empty([y_new, x_new])
            new_img[:, :] = img.min()
            new_img[offset_y:offset_y + y, offset_x:offset_x + x] = img
        else:
            new_img = np.empty([y_new, x_new, z])
            new_img[:, :, :] = img.min()
            new_img[offset_y:offset_y + y, offset_x:offset_x + x, :] = img
    else:
        new_img = img
    return new_img


def reshape(img, shape):
    crop_x, crop_y = shape
    return crop(resize(img, crop_x, crop_y), crop_x, crop_y)


def set_orientation(img):
    affine = img.affine
    axis_code = nib.aff2axcodes(affine)
    if axis_code == ('L', 'P', 'S'):
        return img

    orig = nib.io_orientation(affine)
    target = axcodes2ornt('LPS')
    transform = ornt_transform(orig, target)
    img = img.as_reoriented(transform)
    return img
