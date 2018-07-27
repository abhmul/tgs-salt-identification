import torch.nn.functional as F

from pyjet.layers import Conv2D, BatchNormalization,
from pyjet.models import SLModel
from kaggleutils import dump_args


def get_loss(loss):
    # TODO: Fill in with multiple kinds of losses
    return loss


class UNet(SLModel):

    def __init__(self,
                 num_filters=16,
                 factor=2,
                 num_channels=3):
        self.num_filters = num_filters
        self.factor = factor
        self.num_channels = num_channels

        self.add_loss(F.binary_cross_entropy_with_logits)
        # TODO: Add the optimizer

        # TODO: Create the layers


    def forward(self, x):



@dump_args
def unet(num_filters=16,
         factor=2,
         optimizer="adam",
         loss="binary_crossentropy",
         img_size=(None, None),
         num_channels=3,
         max_val=255.,
         **kwargs):
    inputs = Input(tuple(img_size) + (num_channels, ), name="input")

    s = BatchNormalization()(Conv2D(num_filters, (1, 1), padding='same')(
        Lambda(lambda x: (1. / max_val) * x)(inputs)))
    c1 = Activation('relu')(BatchNormalization()(Conv2D(
        num_filters, (3, 3), padding='same')(s)))
    c1 = Dropout(0.1)(c1)
    c1 = Activation('relu')(BatchNormalization()(Conv2D(
        num_filters, (3, 3), padding='same')(c1)))
    p1 = MaxPooling2D((2, 2))(c1)

    num_filters *= factor
    c2 = Activation('relu')(BatchNormalization()(Conv2D(
        num_filters, (3, 3), padding='same')(p1)))
    c2 = Dropout(0.1)(c2)
    c2 = Activation('relu')(BatchNormalization()(Conv2D(
        num_filters, (3, 3), padding='same')(c2)))
    p2 = MaxPooling2D((2, 2))(c2)

    num_filters *= factor
    c3 = Activation('relu')(BatchNormalization()(Conv2D(
        num_filters, (3, 3), padding='same')(p2)))
    c3 = Dropout(0.2)(c3)
    c3 = Activation('relu')(BatchNormalization()(Conv2D(
        num_filters, (3, 3), padding='same')(c3)))
    p3 = MaxPooling2D((2, 2))(c3)

    num_filters *= factor
    c4 = Activation('relu')(BatchNormalization()(Conv2D(
        num_filters, (3, 3), padding='same')(p3)))
    c4 = Dropout(0.2)(c4)
    c4 = Activation('relu')(BatchNormalization()(Conv2D(
        num_filters, (3, 3), padding='same')(c4)))
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    num_filters *= factor
    c5 = Activation('relu')(BatchNormalization()(Conv2D(
        num_filters, (3, 3), padding='same')(p4)))
    c5 = Dropout(0.3)(c5)
    c5 = Activation('relu')(BatchNormalization()(Conv2D(
        num_filters, (3, 3), padding='same')(c5)))

    num_filters //= factor
    u6 = UpSampling2D(size=(2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Activation('relu')(BatchNormalization()(Conv2D(
        num_filters, (3, 3), padding='same')(u6)))
    c6 = Dropout(0.2)(c6)
    c6 = Activation('relu')(BatchNormalization()(Conv2D(
        num_filters, (3, 3), padding='same')(c6)))

    num_filters //= factor
    u7 = UpSampling2D(size=(2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Activation('relu')(BatchNormalization()(Conv2D(
        num_filters, (3, 3), padding='same')(u7)))
    c7 = Dropout(0.2)(c7)
    c7 = Activation('relu')(BatchNormalization()(Conv2D(
        num_filters, (3, 3), padding='same')(c7)))

    num_filters //= factor
    u8 = UpSampling2D(size=(2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Activation('relu')(BatchNormalization()(Conv2D(
        num_filters, (3, 3), padding='same')(u8)))
    c8 = Dropout(0.1)(c8)
    c8 = Activation('relu')(BatchNormalization()(Conv2D(
        num_filters, (3, 3), padding='same')(c8)))

    num_filters //= factor
    u9 = UpSampling2D(size=(2, 2))(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Activation('relu')(BatchNormalization()(Conv2D(
        num_filters, (3, 3), padding='same')(u9)))
    c9 = Dropout(0.1)(c9)
    c9 = Activation('relu')(BatchNormalization()(Conv2D(
        num_filters, (3, 3), padding='same')(c9)))

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        optimizer=optimizer,
        loss=get_loss(loss),
        metrics=[])
    model.summary()
    return model
