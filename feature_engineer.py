import pandas as pd

CATEGORICAL = ['event_name', 'fqid', 'room_fqid', 'text_fqid', 'text', 'name']
NUMERICAL = ['elapsed_time','level','page','room_coor_x', 'room_coor_y', 
             'screen_coor_x', 'screen_coor_y','hover_duration']
EVENTS = ['navigate_click','person_click','cutscene_click','object_click',
          'map_hover','notification_click','map_click','observation_click',
          'checkpoint']

def feature_engineer1(dataset_df):
    dfs = []
    for c in CATEGORICAL:
        tmp = dataset_df.groupby(['session_id','level_group'])[c].agg('nunique')
        tmp.name = tmp.name + '_nunique'
        dfs.append(tmp)
    for c in NUMERICAL:
        tmp = dataset_df.groupby(['session_id','level_group'])[c].agg('mean')
        tmp.name = tmp.name + '_mean'
        dfs.append(tmp)
    for c in NUMERICAL:
        tmp = dataset_df.groupby(['session_id','level_group'])[c].agg('std')
        tmp.name = tmp.name + '_std'
        dfs.append(tmp)

    dataset_df = pd.concat(dfs,axis=1)
    dataset_df = dataset_df.fillna(-1)
    dataset_df = dataset_df.reset_index()
    dataset_df = dataset_df.set_index('session_id')
    return dataset_df

def feature_engineer2(dataset_df):
    dfs = []
    for c in CATEGORICAL:
        tmp = dataset_df.groupby(['session_id','level_group'])[c].agg('nunique')
        tmp.name = tmp.name + '_nunique'
        dfs.append(tmp)
    for c in NUMERICAL:
        tmp = dataset_df.groupby(['session_id','level_group'])[c].agg('mean')
        tmp.name = tmp.name + '_mean'
        dfs.append(tmp)
    for c in NUMERICAL:
        tmp = dataset_df.groupby(['session_id','level_group'])[c].agg('std')
        tmp.name = tmp.name + '_std'
        dfs.append(tmp)
    for c in NUMERICAL:
        tmp = dataset_df.groupby(['session_id', 'level_group'])[c].agg(lambda x: x.max() - x.min())
        tmp.name = tmp.name + '_range'
        dfs.append(tmp)
    for c in NUMERICAL: 
        tmp = dataset_df.groupby(['session_id', 'level_group'])[c].agg('median')
        tmp.name = tmp.name + '_median'
        dfs.append(tmp)   
    for c in EVENTS: 
        dataset_df[c] = (dataset_df.event_name == c).astype('int8') 
    for c in EVENTS + ['elapsed_time']: 
        tmp = dataset_df.groupby(['session_id','level_group'])[c].agg('sum') 
        tmp.name = tmp.name + '_sum' 
        dfs.append(tmp) 
    for c in EVENTS :
        tmp = dataset_df.groupby(['session_id', 'level_group'])[c].agg(lambda x: x.mode().values[0])
        tmp.name = tmp.name + '_mode'
        dfs.append(tmp)

    dataset_df = dataset_df.drop(EVENTS,axis=1)
    dataset_df = pd.concat(dfs,axis=1)
    dataset_df = dataset_df.fillna(-1)
    dataset_df = dataset_df.reset_index()
    dataset_df = dataset_df.set_index('session_id')
    return dataset_df