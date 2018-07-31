import torch
import torch.nn.functional as F

from pyjet.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from pyjet.models import SLModel
from kaggleutils import dump_args


def get_loss(loss):
    # TODO: Fill in with multiple kinds of losses
    return loss


class UNet(SLModel):

    @dump_args
    def __init__(self,
                 num_filters=8,
                 factor=2,
                 num_channels=3):
        super(UNet, self).__init__()
        self.num_filters = num_filters
        self.factor = factor
        self.num_channels = num_channels

        # Add the loss
        self.add_loss(F.binary_cross_entropy_with_logits, inputs='logit')

        # Create the layers
        self.s = Conv2D(num_filters, 1, batchnorm=True)
        self.c11 = Conv2D(num_filters, 3, activation='relu',
                          batchnorm=True, dropout=0.1)
        self.c12 = Conv2D(num_filters, 3, activation='relu', batchnorm=True)

        num_filters *= factor
        self.c21 = Conv2D(num_filters, 3, activation='relu',
                          batchnorm=True, dropout=0.1)
        self.c22 = Conv2D(num_filters, 3, activation='relu', batchnorm=True)

        num_filters *= factor
        self.c31 = Conv2D(num_filters, 3, activation='relu',
                          batchnorm=True, dropout=0.2)
        self.c32 = Conv2D(num_filters, 3, activation='relu', batchnorm=True)

        num_filters *= factor
        self.c41 = Conv2D(num_filters, 3, activation='relu',
                          batchnorm=True, dropout=0.2)
        self.c42 = Conv2D(num_filters, 3, activation='relu', batchnorm=True)

        num_filters *= factor
        self.c51 = Conv2D(num_filters, 3, activation='relu',
                          batchnorm=True, dropout=0.3)
        self.c52 = Conv2D(num_filters, 3, activation='relu', batchnorm=True)

        num_filters //= factor
        self.c61 = Conv2D(num_filters, 3, activation='relu',
                          batchnorm=True, dropout=0.2)
        self.c62 = Conv2D(num_filters, 3, activation='relu', batchnorm=True)

        num_filters //= factor
        self.c71 = Conv2D(num_filters, 3, activation='relu',
                          batchnorm=True, dropout=0.2)
        self.c72 = Conv2D(num_filters, 3, activation='relu', batchnorm=True)

        num_filters //= factor
        self.c81 = Conv2D(num_filters, 3, activation='relu',
                          batchnorm=True, dropout=0.1)
        self.c82 = Conv2D(num_filters, 3, activation='relu', batchnorm=True)

        num_filters //= factor
        self.c91 = Conv2D(num_filters, 3, activation='relu',
                          batchnorm=True, dropout=0.1)
        self.c92 = Conv2D(num_filters, 3, activation='relu', batchnorm=True)

        self.o = Conv2D(1, 1, activation='linear')

        self.p = MaxPooling2D(2)
        self.u = UpSampling2D(scale_factor=2.)

        # Infer the inputs
        self.infer_inputs(Input(128, 128, 3))

        # Add the optimizer
        self.add_optimizer(torch.optim.Adam(self.parameters()))

    def forward(self, x):
        x = self.s(self.s.fix_input(x))

        c1 = self.c12(self.c11(x))
        x = self.p(c1)

        c2 = self.c22(self.c21(x))
        x = self.p(c2)

        c3 = self.c32(self.c31(x))
        x = self.p(c3)

        c4 = self.c42(self.c41(x))
        x = self.p(c4)

        c5 = self.c52(self.c51(x))

        x = torch.cat([self.u(c5), c4], dim=1)
        c6 = self.c62(self.c61(x))

        x = torch.cat([self.u(c6), c3], dim=1)
        c7 = self.c72(self.c71(x))

        x = torch.cat([self.u(c7), c2], dim=1)
        c8 = self.c82(self.c81(x))

        x = torch.cat([self.u(c8), c1], dim=1)
        c9 = self.c92(self.c91(x))

        self.logit = self.o.unfix_input(self.o(c9))
        self.expit = torch.sigmoid(self.logit)

        return self.expit
