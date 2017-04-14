import glob
import inspect
import os
import time
import cv2
import numpy as np
from zipfile import ZipFile


def get_im_cv2(im_row, im_col, im_path, is_zip=False):
    """
    
    :param im_row: 
    :param im_col: 
    :param im_path: 
    :param is_zip: 
    :return: 
    """
    if is_zip:
        img = cv2.imdecode(np.frombuffer(ZipFile.read(im_path), np.uint8), flags=cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(im_path)

    resized = cv2.resize(img, (im_row, im_col), cv2.INTER_AREA)  # INTER_AREA preferred for shrinking

    return resized


def load_train_im_folder(im_row, im_col, path, folders):
    """
    
    :param folders: 
    :param im_row: image rows (spatial dimension of image)
    :param im_col: image columns (spatial dimension of image)
    :param path: 
    :return: 
    """
    print(inspect.currentframe().f_code.co_name)
    start_time = time.time()

    x = []
    x_id = []
    y = []
    print('... read training samples ...')
    for fld in folders:
        index = folders.index(fld)
        print('... load folder {} (Index: {}) ...'.format(fld, index))
        files = glob.glob(os.path.join(path, fld, '*jpg'))
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(im_row, im_col, fl)
            x.append(img)
            x_id.append(flbase)
            y.append(index)
    print('... completed reading training data in: {} seconds ...'.format(round(time.time() - start_time, 2)))
    print('... returning x, y, x_id (x shape: {}, y shape: {}, x_id shape: {})'
          .format(np.array(x).shape, np.array(y).shape, np.array(x_id).shape))
    return x, y, x_id


def load_test_im_folder(im_row, im_col, path):
    """
    
    :param im_row: 
    :param im_col: 
    :param path: 
    :return: 
    """
    print(inspect.currentframe().f_code.co_name)
    start_time = time.time()

    path = os.path.join(path, '*.jpg')
    files = sorted(glob.glob(path))

    x_test = []
    x_test_id = []
    print('... read test samples ...')
    for idx, fl in enumerate(files):
        flbase = os.path.basename(fl)
        img = get_im_cv2(im_row, im_col, fl)

        if idx % 250 == 0:
            print('loaded {} of {}'.format(idx, len(files)))

        x_test.append(img)
        x_test_id.append(flbase)

    print('... completed reading test data in: {} seconds ...'.format(round(time.time() - start_time, 2)))
    print('... returning x_test, x_test_id (x_test shape: {}, x_test_id shape: {})'
          .format(np.array(x_test).shape, np.array(x_test_id).shape))
    return x_test, x_test_id
