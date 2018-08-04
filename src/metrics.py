import torch
import pyjet.backend as J
from pyjet.metrics import AverageMetric


def iou(y_pred, y_true):
    non_batch_dims = tuple(range(1, y_pred.dim()))
    union = (y_pred | y_true).sum(non_batch_dims)
    intersection = (y_pred & y_true).sum(non_batch_dims)

    # If the union is 0, we both predicted nothing and ground
    # truth was nothing. All other no ground truth cases will
    # be 0.
    no_union = union == 0
    intersection = intersection.masked_fill_(no_union, 1.)
    union = union.masked_fill_(no_union, 1.)
    return intersection.float() / union.float()


class MeanIOU(AverageMetric):

    thresholds = J.arange(0.5, 1.0, 0.05)

    def score(self, y_pred, y_true):
        # Expect two tensors of the same shape
        assert tuple(y_pred.size()) == tuple(y_true.size())
        y_pred = y_pred > 0.5
        y_true = y_true > 0.5  # Casts the ground truth to a byte tensor

        ious = iou(y_pred, y_true)

        # Use the thresholds to calculate the mean precision
        hits = ious.unsqueeze(1) >= self.thresholds.unsqueeze(0)
        tp = hits.float()
        fpfn = (~hits).float()

        # Return the batch score
        return torch.mean(tp / (tp + fpfn))


mean_iou = MeanIOU()
