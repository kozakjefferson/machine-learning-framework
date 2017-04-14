import datetime
import inspect

import pandas as pd


def create_submission_from_pred(predictions, test_id, info):
    print(inspect.currentframe().f_code.co_name)
    # todo: hardcoded columns
    submission = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    submission.loc[:, 'image'] = pd.Series(test_id, index=submission.index)
    # submission.loc['image', :] = pd.Series(test_id, index=submission.index)
    print(submission.head())
    print('Creating submission ...')
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    submission.to_csv(sub_file, index=False)
    print('Completed submission file')


def create_submission_v2(df, info):
    print(inspect.currentframe().f_code.co_name)
    print(df.head())
    print('Creating submission ...')
    now = datetime.datetime.now()
    sub_file = 'subm_' + info + '_' + str(now.strftime("%Y%m%d-%H%M")) + '.csv'
    df.to_csv(sub_file, index=False)
    print('Completed submission file')


def create_dataframe(res, res_cname, res_id, id_cname):
    """
    Create a combined dataframe from predictions and their ids. This dataframe could then be fed to clip or sub
    :param res: predictions or results to be put in dataframe
    :param res_cname: column names of predictions, must be a list e.g. ['c1','c2','c3']
    :param res_id: IDs or filenames for each row
    :param id_cname: column name for IDs column, must be a list e.g. ['xyz']
    """
    print(inspect.currentframe().f_code.co_name)
    df1 = pd.DataFrame(res, columns=res_cname)
    df2 = pd.DataFrame(res_id, columns=id_cname)
    return pd.concat([df2, df1], axis=1)


def clip_dataframe(df, classes, col_names=[], clip_val=0.9):
    print(inspect.currentframe().f_code.co_name)

    if len(col_names) != 0:
        f = df[col_names]
    else:
        f = df

    f = f.clip(lower=(1.0 - clip_val) / float(classes - 1), upper=clip_val)
    f = (f.div(df.sum(axis=1), axis=0)).round(6)
    df[col_names] = f
    print(df.head())
    return df


def clip_csv(csv_file, classes, clip_val=0.9):
    """
    Clipping is a simple operation on predictions where we set a maximum and a minimum certainty. 
    This avoids really hard punishment in case we're wrong. This means while your model gets better, 
    the less clipping will help you improve your score. For example in the earlier stages of this 
    competition, a clip of 0.90 improved our score from 0.94380 to 0.91815.
    :param clip_val: 
    :param csv_file: 
    :param classes: 
    """
    print(inspect.currentframe().f_code.co_name)
    df = pd.read_csv(csv_file, index_col=0)

    # Clip the values
    df = df.clip(lower=(1.0 - clip_val) / float(classes - 1), upper=clip_val)

    # Normalize the values to 1
    df = df.div(df.sum(axis=1), axis=0)

    # Save the new clipped values
    df.to_csv('clip.csv')
    print(df.head(10))


def blend_csv(csv_paths):
    """
    When you have different models or different model configurations, then it could be that some models are experts 
    at recognizing all kinds of tuna, while others are better at distinguishing fish vs no fish. Good specialist 
    models are only very certain in their own area. In this case it helps to let them work together to a solution. 
    A way of combining the outputs of multiple models or model settings is blending. It's a very simple procedure 
    where all predictions are added to each other for each image, class pair and then divided by the number of models.
    
    Blending can for example be used for test augmentation: all test image are augmented with several operations 
    (flipping, rotation, zooming etc.). When you augment each image a couple of times, you can use them as separate 
    submission files, which can be combined using blending afterwards. This approach improved our score from 
    0.89893 to 0.86401 for this competition.
    
    An other simple alternative for blending is majority voting, where every model is allowed to make one 
    prediction per image.
    :type csv_paths: object
    """
    print(inspect.currentframe().f_code.co_name)
    if len(csv_paths) < 2:
        print("Blending takes two or more csv files!")
        return

    # Read the first file
    df_blend = pd.read_csv(csv_paths[0], index_col=0)

    # Loop over all files and add them
    for csv_file in csv_paths[1:]:
        df = pd.read_csv(csv_file, index_col=0)
        df_blend = df_blend.add(df)

    # Divide by the number of files
    df_blend = df_blend.div(len(csv_paths))

    # Save the blend file
    df_blend.to_csv('blend.csv')
    print(df_blend.head(10))


def log_entry_to_file():
    # todo: write function to make entry of all hyperparameters and result to log file
    return 0
