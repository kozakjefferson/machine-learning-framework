import inspect
import time

import keras
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

from ras1_model import create_model_3


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def train_models_with_crossval(train_samples, train_labels, train_ids,
                               cv_folds=3, epochs=5, batch_size=64):
    print(inspect.currentframe().f_code.co_name)
    print('... start CV training ...')
    start_time = time.time()
    random_state = 201704

    try:
        len(train_samples.shape) == 4
    except:
        print('... training data must have 4 dims!')
        raise

    if keras.backend.image_data_format() == 'channels_first':
        _, im_ch, im_row, im_col = [x for x in train_samples.shape]
    else:
        _, im_row, im_col, im_ch = [x for x in train_samples.shape]

    yfull_train = dict()

    # todo: @hyperparameter: Cross validation approach: KFold, StratifiedKFold, LeaveOneOut, ShuffleSplit
    # in case of StratifiedKFold split(X_id, y), in KFold split(X_id)
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    # cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)  # todo: bug here

    num_fold = 0
    sum_score = 0
    models = []

    for x_idx, y_idx in cv.split(train_ids):
        model = create_model_3(im_ch, im_row, im_col)
        x_train = train_samples[x_idx]
        y_train = train_labels[x_idx]
        x_valid = train_samples[y_idx]
        y_valid = train_labels[y_idx]

        num_fold += 1
        print('. Start KFold number {} from {} ...'.format(num_fold, cv_folds))
        print('... split train: ', len(x_train), len(y_train))
        print('... split valid: ', len(x_valid), len(y_valid))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=0),
        ]
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                  shuffle=True, verbose=2, validation_data=(x_valid, y_valid),
                  callbacks=callbacks)

        predictions_valid = model.predict(x_valid.astype('float32'), batch_size=batch_size, verbose=2)

        score = log_loss(y_valid, predictions_valid)
        print('... score log_loss: ', score)
        sum_score += score * len(y_idx)

        # Store valid predictions
        for i in range(len(y_idx)):
            yfull_train[y_idx[i]] = predictions_valid[i]

        models.append(model)

    score = sum_score / len(train_samples)
    print("... Log_loss train independent avg: ", score)

    info_string = 'loss_' + str(np.round(score, 6)) + '_folds_' + str(cv_folds) + '_ep_' + str(epochs)
    print('... completed CV training in {} secs, returning info and {} models'
          .format(round(time.time() - start_time, 2), len(models)))
    return info_string, models


def run_cv_test_data(test_samples, test_ids, info_string, models, batch_size=16):
    nfolds = len(models)

    fold_number = 0
    yfull_test = []

    for i in range(nfolds):
        model = models[i]
        fold_number += 1
        print('Start KFold number {} from {}'.format(fold_number, nfolds))

        test_prediction = model.predict(test_samples, batch_size=batch_size, verbose=2)
        yfull_test.append(test_prediction)

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    info_string = 'loss_' + info_string + '_folds_' + str(nfolds)

    return info_string, test_res, test_ids
