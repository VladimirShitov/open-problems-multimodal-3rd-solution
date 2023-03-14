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

# make validation sample: 'summary/input/fold/cite_fold_val_df.pickle'

# +
import numpy as np
import pandas as pd

# -

# ## data load

# +
raw_path = '../../input/raw/'
preprocess_path = '../../input/preprocess/cite/'
validation_path = '../../input/fold/'

target_path = '../../input/target/'
feature_path = '../../input/features/cite/'
model_path = '../../model/validation/cite/'
# -

train_ids = np.load(preprocess_path + "train_cite_raw_inputs_idxcol.npz", allow_pickle=True)
test_ids = np.load(preprocess_path + "test_cite_raw_inputs_idxcol.npz", allow_pickle=True)
df_meta = pd.read_csv(raw_path + "metadata.csv")
df_meta = df_meta[df_meta['day'] == 7][['cell_id']].reset_index(drop=True)

# ### feature path

feature_dict = {}
train_file, test_file = ['X_best_128.pickle', 'X_test_best_128.pickle']

X = pd.read_pickle(feature_path  + train_file)
X_test = pd.read_pickle(feature_path  + test_file)

X.index = list(train_ids['index'])
X_test.index = list(test_ids['index'])

X_test = X_test.reset_index().rename(columns = {'index': 'cell_id'})
X_test = df_meta.merge(X_test, on = 'cell_id', how = 'left')
X_test = X_test[X_test['base_svd_0'].isnull() == False]

X_test.drop(['cell_id'], axis = 1, inplace = True)

X = pd.concat([X.reset_index(drop=True),
                X_test.reset_index(drop=True),
                ]).reset_index(drop=True)

oof = np.random.uniform(X.shape[0])

X_result = pd.DataFrame(oof, columns = ['pred'])
X_result = pd.concat([X, X_result], axis = 1)

X_train_result = X_result[:70988][['pred']]
X_train_result['cell_id'] = list(train_ids['index'])

df_meta_raw = pd.read_csv(raw_path + "metadata.csv")
X_train_result = X_train_result.merge(df_meta_raw, on = 'cell_id', how = 'left')

# ### make validation
# - Use 2000 people close to pb in each donor as validation.

X_train_result.shape

X_train_result['rank_per_donor'] = X_train_result.groupby(['donor'])['pred'].rank(ascending=False)
X_train_result['rank_per_donor'] = np.random.randint(X_train_result.shape[0], )
X_train_result['rank_all'] = X_train_result['pred'].rank(ascending=False)

X_train_result['flg_donor_val'] = np.where(X_train_result['rank_per_donor'] < 2000, 1, 0)
X_train_result['flg_all_val'] = np.where(X_train_result['rank_all'] < 5000, 1, 0)

print(X_train_result[X_train_result['flg_donor_val'] == 1]['donor'].value_counts())
print(X_train_result[X_train_result['flg_all_val'] == 1]['donor'].value_counts())

X_train_result.to_pickle(validation_path + 'cite_fold_val_df.pickle')

X_train_result.shape
