import os

import keras

from ras1_loadData import load_train_im_folder, load_test_im_folder
from ras1_postProc import create_dataframe, create_submission_v2, \
    clip_dataframe
from ras1_preProc import normalise_data, convert_to_onehot, im_proc_clahe, im_proc_edges
from ras1_runCV import train_models_with_crossval, run_cv_test_data

# todo: @hyperparameters
IM_ROW = 70
IM_COL = 70
BATCH_SIZE = 64
CV_FOLDS = 10  # number for folds for cross validation
EPOCHS = 50  # number of epochs per training cycle (per fold)

# Environment settings
dataPath = r'D:\ranaalisaeed\Documents\datasets\fisheries'
trainDir = r'train'
testDirStg1 = r'test_stg1'
testDirStg2 = r'test_stg2'
trainFolders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
# trainFolders = ['BET']
colNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
numClasses = 8
dataFormat = 'channels_first'

if __name__ == '__main__':
    print('Keras version: {}'.format(keras.__version__))
    train_samples, train_labels, train_ids = load_train_im_folder(IM_ROW, IM_COL,
                                                                  os.path.join(dataPath, trainDir),
                                                                  trainFolders)
    train_samples = im_proc_edges(train_samples)
    # train_samples = im_proc_histeq(train_samples)
    # train_samples = im_proc_clahe(train_samples)

    train_samples = normalise_data(train_samples)
    train_labels = convert_to_onehot(train_labels)

    info_string, cv_models = train_models_with_crossval(train_samples, train_labels, train_ids,
                                                        CV_FOLDS, EPOCHS, BATCH_SIZE)

    # PROCESS TEST STAGE 1
    test_X_1, test_ids_1 = load_test_im_folder(IM_ROW, IM_COL, os.path.join(dataPath, testDirStg1))
    test_X_1 = im_proc_edges(test_X_1)
    #test_X_1 = im_proc_clahe(test_X_1)
    test_X_1 = normalise_data(test_X_1)

    string, test_res_1, test_ids_1 = run_cv_test_data(test_X_1, test_ids_1, info_string, cv_models, BATCH_SIZE)

    # PROCESS TEST STAGE 2
    test_X_2, test_ids_2 = load_test_im_folder(IM_ROW, IM_COL, os.path.join(dataPath, testDirStg2))
    test_X_2 = im_proc_edges(test_X_2)
    #test_X_2 = im_proc_clahe(test_X_2)
    test_X_2 = normalise_data(test_X_2)

    string, test_res_2, test_ids_2 = run_cv_test_data(test_X_2, test_ids_2, info_string, cv_models, BATCH_SIZE)

    # COMBINE STAGE 1 AND 2 RESULTS
    test_res = test_res_1 + test_res_2
    test_ids_2 = [testDirStg2 + '/' + x for x in test_ids_2]
    test_ids = test_ids_1 + test_ids_2

    sub_df = create_dataframe(test_res, colNames, test_ids, ['image'])
    sub_df = clip_dataframe(sub_df, numClasses, colNames, clip_val=0.9)
    create_submission_v2(sub_df, info_string)

    print('x')
