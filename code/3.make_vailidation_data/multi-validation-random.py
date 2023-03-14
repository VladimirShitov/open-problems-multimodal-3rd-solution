# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# make validation sample: 'summary/input/fold/multi_fold_val_df.pickle'

# +
import numpy as np
import pandas as pd


# %matplotlib inline
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# -

# ## data load

def correlation_score(y_true, y_pred):
    """Scores the predictions according to the competition rules.

    It is assumed that the predictions are not constant.

    Returns the average of each sample's Pearson correlation coefficient"""
    if type(y_true) == pd.DataFrame: y_true = y_true.values
    if type(y_pred) == pd.DataFrame: y_pred = y_pred.values
    if y_true.shape != y_pred.shape: raise ValueError("Shapes are different.")
    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)


# +
raw_path = '../../input/raw/'
preprocess_path = '../../input/preprocess/multi/'
validation_path = '../../input/fold/'

target_path = '../../input/target/'
feature_path = '../../input/features/multi/'
model_path = '../../model/validation/multi/'
# -

df_meta = pd.read_csv(raw_path + "metadata.csv")
df_meta = df_meta[df_meta['day'] == 10][['cell_id']].reset_index(drop=True)

# ### feature path

# +
train_file, test_file = ['multi_train_okapi_feature_128.pickle', 'multi_test_okapi_feature_128.pickle']
# -

# ## 

fold_df = pd.read_pickle(validation_path + 'fold_df.pickle')

X = pd.read_pickle(feature_path  + train_file)
X_test = pd.read_pickle(feature_path  + test_file)

train_num = len(X)

X_test = X_test.reset_index().rename(columns = {'index': 'cell_id'})
X_test = df_meta.merge(X_test, on = 'cell_id', how = 'left')
X_test.drop(['cell_id'], axis = 1, inplace = True)

X = pd.concat([X.reset_index(drop=True),
                X_test.reset_index(drop=True),
                ]).reset_index(drop=True)

oof = np.random.uniform(X.shape[0])

X_result = pd.DataFrame(oof, columns = ['pred'])
X_result = pd.concat([X, X_result], axis = 1)

X_index = pd.read_pickle(feature_path  + train_file)

X_train_result = X_result[:len(fold_df)][['pred']]
X_train_result['cell_id'] = list(X_index.index)

df_meta_raw = pd.read_csv(raw_path + "metadata.csv")
X_train_result = X_train_result.merge(df_meta_raw, on = 'cell_id', how = 'left')

df_meta_raw[df_meta_raw['day'] == 10]['donor'].value_counts(normalize = True)

# ### make validation
# - Use 3000 people close to pb in each donor as validation.

X_train_result['rank_per_donor'] = X_train_result.groupby(['donor'])['pred'].rank(ascending=False)
X_train_result['rank_all'] = X_train_result['pred'].rank(ascending=False)

X_train_result['flg_donor_val'] = np.where(X_train_result['rank_per_donor'] < 3000, 1, 0)
X_train_result['flg_all_val'] = np.where(X_train_result['rank_all'] < 10000, 1, 0)

print(X_train_result[X_train_result['flg_donor_val'] == 1]['donor'].value_counts())
print(X_train_result[X_train_result['flg_all_val'] == 1]['donor'].value_counts())

X_train_result.to_pickle(validation_path + 'multi_fold_random_val_df.pickle')
