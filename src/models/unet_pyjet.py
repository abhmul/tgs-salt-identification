import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

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
                         batchnorm=batchnorm)
        self.drop = nn.Dropout2d(p=self.dropout)
        self.c2 = Conv2D(num_filters, kernel_size, activation=activation,
                         batchnorm=batchnorm)
        self.p = MaxPooling2D(pool_size)

    def forward(self, x):
        residual = self.c2(self.drop(self.c1(x)))
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
                         batchnorm=batchnorm)
        self.drop = nn.Dropout2d(p=self.dropout)
        self.c2 = Conv2D(num_filters, kernel_size, activation=activation,
                         batchnorm=batchnorm)
        self.u = UpSampling2D(scale_factor=scale_factor)

    def forward(self, x, residual):
        x = torch.cat([self.u(x), residual], dim=1)
        x = self.c2(self.drop(self.c1(x)))
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
                         batchnorm=batchnorm)
        self.drop = nn.Dropout2d(p=self.dropout)
        self.c2 = Conv2D(num_filters, kernel_size, activation=activation,
                         batchnorm=batchnorm)

    def forward(self, x):
        return self.c2(self.drop(self.c1(x)))


class DilationNeck(nn.Module):
    def __init__(self, num_filters, dropout=0.3, depth=6, kernel_size=3,
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
                   padding='same', batchnorm=batchnorm, dilation=2**i)
            for i in range(depth)])
        self.drop = nn.Dropout2d(p=self.dropout)

    def forward(self, x):
        dilated_sum = 0.
        for layer in self.dilated_layers:
            x = self.drop(layer(x))
            dilated_sum += x
        return dilated_sum


class UNet9(SLModel):

    @dump_args
    def __init__(self,
                 num_filters=16,
                 factor=4,
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
        self.e1 = EncoderBlock(num_filters, dropout=0.0625)
        num_filters *= factor
        self.e2 = EncoderBlock(num_filters, dropout=0.125)
        # num_filters *= factor
        # self.e3 = EncoderBlock(num_filters, dropout=0.2)
        # num_filters *= factor
        # self.e4 = EncoderBlock(num_filters, dropout=0.2)

        num_filters *= factor
        self.neck = DilationNeck(num_filters, depth=6, dropout=0.25)

        # num_filters //= factor
        # self.d1 = DecoderBlock(num_filters, dropout=0.2)
        # num_filters //= factor
        # self.d2 = DecoderBlock(num_filters, dropout=0.2)
        num_filters //= factor
        self.d3 = DecoderBlock(num_filters, dropout=0.125)
        num_filters //= factor
        self.d4 = DecoderBlock(num_filters, dropout=0.0625)
        self.downsampler = UpSampling2D(size=(101, 101))
        self.o = Conv2D(1, 1, activation='linear')

        # Infer the inputs
        self.infer_inputs(Input(4, 101, 101))

        # Add the optimizer
        self.add_optimizer(torch.optim.Adam(self.parameters()))

    def cast_input_to_torch(self, x):
        return super().cast_input_to_torch(x.transpose(0, 3, 1, 2))

    def cast_target_to_torch(self, x):
        return super().cast_target_to_torch(x.transpose(0, 3, 1, 2))

    def cast_output_to_numpy(self, preds):
        super().cast_output_to_numpy(preds).transpose(0, 2, 3, 1)

    def forward(self, x):
        # Fix input to channels_first
        # print(x[:, 3:, 0, 0])
        res0 = self.s(x)
        x = self.upsampler(res0)

        x, res1 = self.e1(x)  # 64
        x, res2 = self.e2(x)  # 32
        # x, res3 = self.e3(x)  # 16
        # x, res4 = self.e4(x)

        x = self.neck(x)

        # x = self.d1(x, res4)
        # x = self.d2(x, res3)
        x = self.d3(x, res2)
        x = self.d4(x, res1)

        x = torch.cat([self.downsampler(x), res0], dim=1)
        # Unfix it back to channels_last
        self.logit = self.o(x)
        self.expit = torch.sigmoid(self.logit)

        return self.expit


# Based on NeptuneML unet resnet
class UNetResNet(SLModel):
    """PyTorch U-Net model using ResNet(34, 101 or 152) encoder.
    UNet: https://arxiv.org/abs/1505.04597
    ResNet: https://arxiv.org/abs/1512.03385
    Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
    Args:
            encoder_depth (int): Depth of a ResNet encoder (34, 101 or 152).
            num_classes (int): Number of output classes.
            num_filters (int, optional): Number of filters in the last layer of decoder. Defaults to 32.
            dropout_2d (float, optional): Probability factor of dropout layer before output layer. Defaults to 0.2.
            pretrained (bool, optional):
                False - no pre-trained weights are being used.
                True  - ResNet encoder is pre-trained on ImageNet.
                Defaults to False.
            is_deconv (bool, optional):
                False: bilinear interpolation is used in decoder.
                True: deconvolution is used in decoder.
                Defaults to False.
    """

    def __init__(self, encoder_depth=152, num_filters=32, pretrained=True):
        super().__init__()
        self.encoder_depth = encoder_depth
        self.num_filters = 32
        self.pretrained = pretrained

        # Add the loss
        self.add_loss(F.binary_cross_entropy_with_logits, inputs='logit', name="bce")
        self.add_loss(dice_loss, inputs='expit', name="dice")

        if encoder_depth == 34:
            # Output it 512 filters
            encoder = torchvision.models.resnet34(pretrained=pretrained)
        elif encoder_depth == 101:
            # Output it 2048 filters
            encoder = torchvision.models.resnet101(pretrained=pretrained)
        elif encoder_depth == 152:
            # Output it 2048 filters
            encoder = torchvision.models.resnet152(pretrained=pretrained)
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        self.pool = encoder.maxpool  # Halves input size
        self.relu = nn.ReLU(inplace=True)
        self.upsampler = UpSampling2D(size=(128, 128))
        self.resample = nn.Sequential(encoder.conv1,  # Halves input size
                                      encoder.bn1,
                                      encoder.relu)

        self.encoder1 = encoder.layer1
        self.encoder2 = encoder.layer2
        self.encoder3 = encoder.layer3
        self.encoder4 = encoder.layer4
        # Clear out the rest of the resnet from memory
        encoder = None
        torch.cuda.empty_cache()

        num_filters *= 8
        self.neck = DilationNeck(num_filters, dropout=0.25, depth=3)

        num_filters //= 2
        self.decoder4 = DecoderBlock(num_filters, dropout=0.125)
        num_filters //= 2
        self.decoder3 = DecoderBlock(num_filters, dropout=0.125)
        num_filters //= 2
        self.decoder2 = DecoderBlock(num_filters, dropout=0.0625)
        num_filters //= 2
        self.decoder1 = DecoderBlock(num_filters, dropout=0.0625)

        self.downsampler = UpSampling2D(size=(101, 101))
        self.final = Conv2D(1, 1, activation='linear')

        # Infer the inputs
        self.infer_inputs(Input(4, 101, 101))

        # Add the optimizer
        self.add_optimizer(torch.optim.SGD(self.encoder_params, lr=1e-4))
        self.add_optimizer(torch.optim.Adam(self.other_params))
        # self.add_optimizer(optim.Adam(self.parameters()))

    @property
    def encoder_params(self):
        return list(self.resample.parameters()) \
            + list(self.encoder1.parameters()) \
            + list(self.encoder2.parameters()) \
            + list(self.encoder3.parameters()) \
            + list(self.encoder4.parameters())

    @property
    def other_params(self):
        encoder_param_ids = {id(param) for param in self.encoder_params}
        return [param for param in self.parameters() if id(param) not in
                encoder_param_ids]

    def cast_input_to_torch(self, x):
        return super().cast_input_to_torch(x.transpose(0, 3, 1, 2))

    def cast_target_to_torch(self, x):
        return super().cast_target_to_torch(x.transpose(0, 3, 1, 2))

    def cast_output_to_numpy(self, preds):
        super().cast_output_to_numpy(preds).transpose(0, 2, 3, 1)

    def forward(self, x):
        # Split out the depth
        depths = x[:, 3:]
        orig_img = x[:, :3]  # 101 x 101

        res0 = self.upsampler(orig_img)  # Turns into 128 x 128 image

        res1 = self.encoder1(self.resample(res0))  # 64 x 64
        res2 = self.encoder2(res1)  # 32 x 32
        res3 = self.encoder3(res2)  # 16 x 16

        x = self.neck(self.encoder4(res3))  # 8 x 8

        x = self.decoder4(x, res3)  # 16 x 16
        x = self.decoder3(x, res2)  # 32 x 32
        x = self.decoder2(x, res1)  # 64 x 64
        x = self.decoder1(x, res0)  # 128 x 128

        x = torch.cat([self.downsampler(x), orig_img, depths], dim=1)
        # Unfix it back to channels_last
        self.logit = self.final(x)
        # print(self.logit)
        self.expit = torch.sigmoid(self.logit)

        return self.expit
