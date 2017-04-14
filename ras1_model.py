import inspect

import keras
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Dropout, Dense, Conv2D, \
    LeakyReLU
from keras.models import Sequential
from keras.optimizers import SGD


def create_model(im_ch, im_row, im_col):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(im_ch, im_row, im_col), dim_ordering='th'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))

    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model


def create_model_2(im_ch, im_row, im_col):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(im_ch, im_row, im_col), dim_ordering='th'))
    model.add(Convolution2D(64, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(64, 2, 2, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(128, 2, 2, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(128, 2, 2, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(256, 2, 2, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(256, 2, 2, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(256, 2, 2, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(256, 2, 2, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(512, 2, 2, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(512, 2, 2, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(512, 2, 2, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(512, 2, 2, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))

    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model


def create_model_3(im_ch, im_row, im_col):
    activation = LeakyReLU(alpha=0.3)  # todo: @hyperparameter: relu, leakyrelu
    input_shape = [im_ch, im_row, im_col]

    try:
        keras.backend.image_data_format() == 'channels_first'
    except:
        print('{} requires "channel_first" data format.'.format(inspect.currentframe().f_code.co_name))
        raise

    model = Sequential()
    # model.add(Activation(lambda x: (x - be.mean(x))/be.std(x), input_shape=(IMG_ROWS, IMG_COLS, IMG_CH)))
    # model.add(BatchNormalization(input_shape=(im_ch, im_row, im_col)))

    model.add(Conv2D(64, (3, 3), padding='same', activation=activation, input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), padding='same', activation=activation))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (2, 2), padding='same', activation=activation))
    model.add(Conv2D(128, (2, 2), padding='same', activation=activation))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(256, (2, 2), padding='same', activation=activation))
    model.add(Conv2D(256, (2, 2), padding='same', activation=activation))
    model.add(Conv2D(256, (2, 2), padding='same', activation=activation))
    model.add(Conv2D(256, (2, 2), padding='same', activation=activation))
    model.add(Conv2D(256, (1, 1), padding='same', activation=activation))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(512, (2, 2), padding='same', activation=activation))
    model.add(Conv2D(512, (2, 2), padding='same', activation=activation))
    model.add(Conv2D(512, (2, 2), padding='same', activation=activation))
    model.add(Conv2D(512, (2, 2), padding='same', activation=activation))
    model.add(Conv2D(256, (1, 1), padding='same', activation=activation))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation=activation, kernel_initializer='uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation=activation, kernel_initializer='uniform'))
    model.add(Dropout(0.5))

    # todo: @hyperparameter: softmax, sigmoid
    # todo: hardcoded num_classes
    model.add(Dense(8, activation='softmax', kernel_initializer='uniform'))

    # todo: @hyperparameters: optimizer, learning_rate, decay, momentum
    # my_opt = Adam(lr=1e-4, decay=1e-6)
    my_opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    # my_opt = RMSprop(lr=1e-4)

    model.compile(optimizer=my_opt, loss='categorical_crossentropy')

    return model
