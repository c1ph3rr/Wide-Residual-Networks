from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Add, BatchNormalization, Activation, AvgPool2D, Flatten, Dense


def conv_block1(inp, k=1):
    i = inp
    
    x = BatchNormalization()(inp)
    x = Activation('relu')(x)
    x = Conv2D(filters=16 * k, kernel_size=3, padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=16 * k, kernel_size=3, padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    
    add1 = Add()([i, x])

    return add1


def conv_block2(inp, k=1):
    i = inp
    
    y = BatchNormalization()(inp)
    y = Activation('relu')(y)
    y = Conv2D(filters=32 * k, kernel_size=3, padding='same', kernel_initializer='he_normal', use_bias=False)(y)
    
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(filters=32 * k, kernel_size=3, padding='same', kernel_initializer='he_normal', use_bias=False)(y)
    
    add2 = Add()([i, y])

    return add2


def conv_block3(inp, k=1):
    i = inp
    
    z = BatchNormalization()(inp)
    z = Activation('relu')(z)
    z = Conv2D(filters=64 * k, kernel_size=3, padding='same', kernel_initializer='he_normal', use_bias=False)(z)
    
    z = BatchNormalization()(z)
    z = Activation('relu')(z)
    z = Conv2D(filters=64 * k, kernel_size=3, padding='same', kernel_initializer='he_normal', use_bias=False)(z)
    
    add3 = Add()([i, z])
    
    return add3


def shortcut_conv(inp, filters, k=1, strides=1):
    i = inp
    
    x = Conv2D(filters=filters * k, kernel_size=3, strides=strides, padding='same', kernel_initializer='he_normal', use_bias=False)(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters=filters * k, kernel_size=3, padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    
    shortcut = Conv2D(filters=filters * k, kernel_size=3, strides=strides, padding='same', kernel_initializer='he_normal', use_bias=False)(inp)
    add = Add()([x, shortcut])
    
    return add


def bn_block(inp):
    bn = BatchNormalization()(inp)
    bn = Activation('relu')(bn)
    return bn


inp = Input((32,32,3))

conv1 = Conv2D(16, 3, padding='same', kernel_initializer='he_normal', use_bias=False)(inp)
conv1 = BatchNormalization()(conv1)
conv1 = Activation('relu')(conv1)

conv2 = shortcut_conv(conv1, 16)
conv2 = conv_block1(conv2)
conv2 = bn_block(conv2)

conv3 = shortcut_conv(conv2, 32, 1, 2)
conv3 = conv_block2(conv3)
conv3 = bn_block(conv3)

conv4 = shortcut_conv(conv3, 64, 1, 2)
conv4 = conv_block3(conv4)
conv4 = bn_block(conv4)

pool = AvgPool2D(8)(conv4)

flatten = Flatten()(pool)

dense = Dense(10, activation='softmax')(flatten)

model = Model(inp, dense).summary()
plot_model(model, 'wrn.png', show_shapes=True)