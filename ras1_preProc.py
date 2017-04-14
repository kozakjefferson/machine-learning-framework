import inspect

import cv2
import numpy as np
import time
from keras.utils import np_utils
from matplotlib import pyplot as plt


def reshape_data():
    """
    reshape numpy array to channels-first or channels-last format
    :return: 
    """
    print(inspect.currentframe().f_code.co_name)
    pass


def normalise_data(d):
    print(inspect.currentframe().f_code.co_name)
    print('... convert to numpy array ...')
    d = np.array(d, dtype=np.uint8)
    print('... converted to array {} ...'.format(d.shape))

    print('... reshape ...')

    try:
        len(d.shape) == 4
    except:
        print('... 4 dims not detected!')
        raise

    d = d.transpose((0, 3, 1, 2))

    print('... convert to float & normalise ...')
    d = d.astype('float32')
    d = d / 255  # todo: hardcoded: divide by max value
    print('... returning normalised array {}'.format(d.shape))
    return d


def convert_to_onehot(d):
    print(inspect.currentframe().f_code.co_name)
    print('... convert to numpy ...')
    d = np.array(d, dtype=np.uint8)

    print('... convert to float ...')
    d = np_utils.to_categorical(d, 8)  # todo: hardcoded num_classes

    print('... returning onehot {}'.format(d.shape))
    return d


def im_proc_edges(imgs):
    print(inspect.currentframe().f_code.co_name)
    start_time = time.time()

    edges = []
    for idx, im in enumerate(imgs):
        dst = cv2.Canny(im, 100, 200, apertureSize=3, L2gradient=True)
        edges.append(dst)
        if idx % 100 == 0:
            plt.subplot(121), plt.imshow(im, cmap='gray')
            plt.title('Original Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(dst, cmap='gray')
            plt.title('Processed Image'), plt.xticks([]), plt.yticks([])
            plt.ion()
            plt.show()

    # rand_dst = random.choice(edges)
    # plt.imshow(rand_dst, cmap='gray')
    # plt.show()

    print('... completed processing images in: {} seconds ...'.format(round(time.time() - start_time, 2)))
    print('... returning processed images {}'.format(np.array(edges).shape))
    return edges


def im_proc_histeq(imgs):
    print(inspect.currentframe().f_code.co_name)
    start_time = time.time()

    pimgs = []
    for idx, im in enumerate(imgs):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        dst = cv2.equalizeHist(im)
        pimgs.append(dst)
        if idx % 100 == 0:
            plt.subplot(121), plt.imshow(im, cmap='gray')
            plt.title('Original Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(dst, cmap='gray')
            plt.title('Processed Image'), plt.xticks([]), plt.yticks([])
            plt.ion()
            plt.show()

    pimgs = np.expand_dims(pimgs, axis=3)

    # rand_dst = random.choice(edges)
    # plt.imshow(rand_dst, cmap='gray')
    # plt.show()

    print('... completed processing images in: {} seconds ...'.format(round(time.time() - start_time, 2)))
    print('... returning processed images {}'.format(np.array(pimgs).shape))
    return pimgs


def im_proc_clahe(imgs):
    print(inspect.currentframe().f_code.co_name)
    start_time = time.time()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    pimgs = []
    print('... converting images ...')
    for idx, im in enumerate(imgs):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        dst = clahe.apply(im)
        pimgs.append(dst)
        if idx % 100 == 0:
            plt.subplot(121), plt.imshow(im, cmap='gray')
            plt.title('Original Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(dst, cmap='gray')
            plt.title('Processed Image'), plt.xticks([]), plt.yticks([])
            plt.ion()
            plt.show()

    pimgs = np.expand_dims(pimgs, axis=3)

    # rand_dst = random.choice(edges)
    # plt.imshow(rand_dst, cmap='gray')
    # plt.show()

    print('... completed processing images in: {} seconds ...'.format(round(time.time() - start_time, 2)))
    print('... returning processed images {}'.format(np.array(pimgs).shape))
    return pimgs
