import pandas as pd
import numpy as np
import pickle

dtypes={
    'elapsed_time':np.int32,
    'event_name':'category',
    'name':'category',
    'level':np.int8,
    'room_coor_x':np.float32,
    'room_coor_y':np.float32,
    'screen_coor_x':np.float32,
    'screen_coor_y':np.float32,
    'hover_duration':np.float32,
    'text':'category',
    'fqid':'category',
    'room_fqid':'category',
    'text_fqid':'category',
    'fullscreen':'category',
    'hq':'category',
    'music':'category',
    'level_group':'category'
}

dataset_df = pd.read_csv('./input/train.csv', dtype=dtypes)
with open('./data/dataset_df.pkl', 'wb') as file:
    pickle.dump(dataset_df, file)

labels = pd.read_csv('./input/train_labels.csv')
labels['session'] = labels.session_id.apply(lambda x: int(x.split('_')[0]))
labels['q'] = labels.session_id.apply(lambda x: int(x.split('_')[-1][1:]))
with open('./data/labels.pkl', 'wb') as file:
    pickle.dump(labels, file)

ALL_USERS = dataset_df['session_id'].drop_duplicates().values
true = pd.DataFrame(data=np.zeros((len(ALL_USERS),18)), index=ALL_USERS)
for k in range(18):
    tmp = labels.loc[labels.q == k+1].set_index('session').loc[ALL_USERS]
    true[k] = tmp.correct.values
true.to_csv("./data/true.csv", index=True)