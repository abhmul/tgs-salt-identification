import torch
import torch.nn as nn
import torch.nn.functional as F

from pyjet.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from pyjet.models import SLModel
from kaggleutils import dump_args

from .losses_pyjet import weighted_bce_loss, weighted_dice_loss, dice_loss


class EncoderBlock(nn.Module):

    def __init__(self, num_filters, dropout, kernel_size=3, activation='relu',
                 batchnorm=True, pool_size=2):
        super(EncoderBlock, self).__init__()
        self.num_filters = num_filters
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.activation = activation
        self.batchnorm = batchnorm
        self.pool_size = pool_size

        self.c1 = Conv2D(num_filters, kernel_size, activation=activation,
                         batchnorm=batchnorm, dropout=dropout)
        self.c2 = Conv2D(num_filters, kernel_size, activation=activation,
                         batchnorm=batchnorm)
        self.p = MaxPooling2D(pool_size)

    def forward(self, x):
        residual = self.c2(self.c1(x))
        x = self.p(residual)
        return x, residual


class DecoderBlock(nn.Module):

    def __init__(self, num_filters, dropout, kernel_size=3, activation='relu',
                 batchnorm=True, scale_factor=2., pool_size=2):
        super(DecoderBlock, self).__init__()
        self.num_filters = num_filters
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.activation = activation
        self.batchnorm = batchnorm
        self.scale_factor = scale_factor

        self.c1 = Conv2D(num_filters, kernel_size, activation=activation,
                         batchnorm=batchnorm, dropout=dropout)
        self.c2 = Conv2D(num_filters, kernel_size, activation=activation,
                         batchnorm=batchnorm)
        self.u = UpSampling2D(scale_factor=scale_factor)

    def forward(self, x, residual):
        x = torch.cat([self.u(x), residual], dim=1)
        x = self.c2(self.c1(x))
        return x


class BasicNeck(nn.Module):

    def __init__(self, num_filters, dropout=0.3, kernel_size=3,
                 activation='relu', batchnorm=True):
        super(BasicNeck, self).__init__()
        self.num_filters = num_filters
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.activation = activation
        self.batchnorm = batchnorm

        self.c1 = Conv2D(num_filters, kernel_size, activation=activation,
                         batchnorm=batchnorm, dropout=dropout)
        self.c2 = Conv2D(num_filters, kernel_size, activation=activation,
                         batchnorm=batchnorm)

    def forward(self, x):
        return self.c2(self.c1(x))


class DilationNeck(nn.Module):
    def __init__(self, num_filters, dropout=0.3, depth=3, kernel_size=3,
                 activation='relu', batchnorm=True):
        super(DilationNeck, self).__init__()
        self.num_filters = num_filters
        self.dropout = dropout
        self.depth = depth
        self.kernel_size = kernel_size
        self.activation = activation
        self.batchnorm = batchnorm

        self.dilated_layers = nn.ModuleList([
            Conv2D(num_filters, kernel_size, activation=activation,
                   padding='same', batchnorm=batchnorm, dropout=dropout,
                   dilation=2**i)
            for i in range(depth)])

    def forward(self, x):
        dilated_sum = 0.
        for layer in self.dilated_layers:
            x = layer(x)
            dilated_sum += x
        return dilated_sum


class UNet9(SLModel):

    @dump_args
    def __init__(self,
                 num_filters=16,
                 factor=2,
                 num_channels=3):
        super(UNet9, self).__init__()
        self.num_filters = num_filters
        self.factor = factor
        self.num_channels = num_channels

        # Add the loss
        self.add_loss(F.binary_cross_entropy_with_logits, inputs='logit', name="bce")
        self.add_loss(dice_loss, inputs='expit', name="dice")
        # self.add_loss(weighted_bce_loss, inputs='logit', name="bce")
        # self.add_loss(weighted_dice_loss, inputs='expit', name="dice", weight=0.)

        # Create the layers
        self.s = Conv2D(num_filters, 1, batchnorm=True)
        self.upsampler = UpSampling2D(size=(128, 128))
        self.e1 = EncoderBlock(num_filters, dropout=0.1)
        num_filters *= factor
        self.e2 = EncoderBlock(num_filters, dropout=0.1)
        num_filters *= factor
        self.e3 = EncoderBlock(num_filters, dropout=0.2)
        # num_filters *= factor
        # self.e4 = EncoderBlock(num_filters, dropout=0.2)

        num_filters *= factor
        self.neck = DilationNeck(num_filters, dropout=0.0)

        # num_filters //= factor
        # self.d1 = DecoderBlock(num_filters, dropout=0.2)
        num_filters //= factor
        self.d2 = DecoderBlock(num_filters, dropout=0.2)
        num_filters //= factor
        self.d3 = DecoderBlock(num_filters, dropout=0.1)
        num_filters //= factor
        self.d4 = DecoderBlock(num_filters, dropout=0.1)
        self.downsampler = UpSampling2D(size=(101, 101))
        self.o = Conv2D(1, 1, activation='linear')

        # Infer the inputs
        self.infer_inputs(Input(101, 101, 3))

        # Add the optimizer
        self.add_optimizer(torch.optim.Adam(self.parameters()))

    def forward(self, x):
        res0 = self.s(self.s.fix_input(x))
        x = self.upsampler(res0)

        x, res1 = self.e1(x)
        x, res2 = self.e2(x)
        x, res3 = self.e3(x)
        # x, res4 = self.e4(x)

        x = self.neck(x)

        # x = self.d1(x, res4)
        x = self.d2(x, res3)
        x = self.d3(x, res2)
        x = self.d4(x, res1)

        x = torch.cat([self.downsampler(x), res0], dim=1)
        self.logit = self.o.unfix_input(self.o(x))
        self.expit = torch.sigmoid(self.logit)

        return self.expit
