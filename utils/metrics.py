import numpy as np
import scipy.spatial.distance as scipy


def si_estimate(gt, pred):
    if not np.array_equal(gt, gt.astype(bool)):
        raise ValueError('Input ground truth is not binary')
    if not np.array_equal(pred, pred.astype(bool)):
        raise ValueError('Input prediction is not binary')

    def get_cr_x(input):
        rows, cols, slices = np.where(input == 2)
        while True:
            int_len = len(rows)
            for i, row in enumerate(rows):
                local = input[row - 1:row + 2, cols[i] - 1:cols[i] + 2, slices[i] - 1:slices[i] + 2]
                local[local == 1] = 2
                input[row - 1:row + 2, cols[i] - 1:cols[i] + 2, slices[i] - 1:slices[i] + 2] = local
            rows, cols, slices = np.where(input == 2)
            if len(rows) == int_len:
                break
        input[input == 2] = 0
        return input

    union_or: np.ndarray = np.bitwise_or(gt.astype(int), pred.astype(int))
    union_or_gt: np.ndarray = union_or + gt
    union_or_pred: np.ndarray = union_or + pred

    cr_pred: np.ndarray = get_cr_x(union_or_gt)
    cr_gt: np.ndarray = get_cr_x(union_or_pred)

    de_images: np.ndarray = (cr_pred + cr_gt).sum(axis=(1, 2))
    de_mean: float = de_images[de_images != 0].mean()
    mta: float = 0.5 * (gt.sum() + pred.sum())

    cr12_union: np.ndarray = union_or - (cr_pred + cr_gt)
    cr12_intersection: np.ndarray = (gt - cr_gt) * (pred - cr_pred)

    oe_image: np.ndarray = (cr12_union - cr12_intersection).sum(axis=(1, 2))
    mta_image: np.ndarray = 0.5 * (gt.sum(axis=(1, 2)) + pred.sum(axis=(1, 2)))
    mta_image[oe_image == 0] = 0
    oer_mean: float = (oe_image[oe_image != 0] / mta_image[mta_image != 0]).mean()

    si_e: float = 1 - 0.5 * oer_mean - 0.5 * (de_mean / mta)
    return si_e


def si(gt, pred):
    if not np.array_equal(gt, gt.astype(bool)):
        raise ValueError('Input ground truth is not binary')
    if not np.array_equal(pred, pred.astype(bool)):
        raise ValueError('Input prediction is not binary')

    gt: np.ndarray = gt.flatten()
    pred: np.ndarray = pred.flatten()

    si = 1.0 - scipy.dice(gt, pred)
    return si


def hd(gt, pred):
    if not np.array_equal(gt, gt.astype(bool)):
        raise ValueError('Input ground truth is not binary')
    if not np.array_equal(pred, pred.astype(bool)):
        raise ValueError('Input prediction is not binary')

    gt_to_pred = []
    pred_to_gt = []
    for slice in range(gt.shape[2]):
        gt_to_pred.append(scipy.directed_hausdorff(gt[:, :, slice], pred[:, :, slice])[0])
        pred_to_gt.append(scipy.directed_hausdorff(pred[:, :, slice], gt[:, :, slice])[0])

    gt_to_pred = np.array(gt_to_pred)[gt_to_pred != 0]
    pred_to_gt = np.array(pred_to_gt)[pred_to_gt != 0]

    return np.maximum(gt_to_pred.mean(), pred_to_gt.mean())


def f1(gt, pred):
    if not np.array_equal(gt, gt.astype(bool)):
        raise ValueError('Input ground truth is not binary')
    if not np.array_equal(pred, pred.astype(bool)):
        raise ValueError('Input prediction is not binary')

    tp: np.ndarray = gt * pred
    precision: float = tp.sum() / pred.sum()
    recall: float = tp.sum() / gt.sum()

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def avd(gt, pred):
    if not np.array_equal(gt, gt.astype(bool)):
        raise ValueError('Input ground truth is not binary')
    if not np.array_equal(pred, pred.astype(bool)):
        raise ValueError('Input prediction is not binary')

    avd = 100 * ((gt.sum() - pred.sum()) / gt.sum())
    return avd
