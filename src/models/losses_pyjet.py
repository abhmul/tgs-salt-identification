import torch.nn.functional as F
import pyjet.backend as J


def dice_coeff(y_pred, y_true, smooth=1.):
    assert tuple(y_true.size()) == tuple(y_pred.size())
    intersection = J.sum(y_true * y_pred)
    score = (2. * intersection + smooth) / \
        (J.sum(y_true) + J.sum(y_pred) + smooth)
    return score.mean()


def dice_loss(y_pred, y_true):
    return 1. - dice_coeff(y_pred, y_true)


def weighted_dice_coeff(y_pred, y_true, weight):
    smooth = 1.
    w = weight * weight
    intersection = y_true * y_pred
    score = (2. * J.sum(w * intersection) + smooth) / \
        (J.sum(w * y_true) + J.sum(w * y_pred) + smooth)
    return score.mean()


def get_border_weights(y_true):
    # if we want to get same size of output, kernel size must be odd number
    if y_true.size(1) == 128:
        kernel_size = 5
    elif y_true.size(1) == 256:
        kernel_size = 11
    elif y_true.size(1) == 512:
        kernel_size = 11
    elif y_true.size(1) == 1024:
        kernel_size = 21
    else:
        raise ValueError('Unexpected image size')
    padding = (kernel_size - 1) // 2
    averaged_mask = F.avg_pool2d(y_true, kernel_size, stride=1,
                                 padding=padding, count_include_pad=False)
    border = ((averaged_mask > 0.005) & (averaged_mask < 0.995)).float()
    weight = J.ones(*averaged_mask.size())
    w0 = J.sum(weight)
    weight += border * 2
    w1 = J.sum(weight)
    # Reshape the rescaling factor to broadcast
    weight *= (w0 / w1).view(-1, *([1] * (y_true.dim() - 1)))
    return weight


def weighted_dice_loss(y_pred, y_true):
    weight = get_border_weights(y_true)
    loss = 1. - weighted_dice_coeff(y_pred, y_true, weight)
    return loss


def weighted_bce_loss(y_pred, y_true):
    weight = get_border_weights(y_true)
    loss = F.binary_cross_entropy_with_logits(
        y_pred, y_true, weight=weight)
    return loss
