import pandas as pd
'''
Some Util funcs adapted from MIMIC-Extract:
https://github.com/MLforHealth/MIMIC_Extract
Shirly Wang, Matthew B. A. McDermott, Geeticka Chauhan, Michael C. Hughes, Tristan Naumann, 
and Marzyeh Ghassemi. MIMIC-Extract: A Data Extraction, Preprocessing, and Representation 
Pipeline for MIMIC-III. arXiv:1907.08322. 
'''

def combine_cols(a, b):
    b.columns.names = ['LEVEL2', 'Aggregation Function']
    a = a.droplevel(level=0, axis=1)
    b = b.droplevel(level=0, axis=1)

    or_filled = len(b) - b.loc[:, 'mean'].isnull().sum()
    or_mean = b.loc[:, 'mean'].dropna().mean()

    row_mask = (a.loc[:, 'count'] > 0).values
    mask = (b.loc[:, 'count'] > 0).values

    c = b.loc[row_mask * mask, 'count'].mul(b.loc[row_mask * mask, 'mean'].values) + \
        a.loc[row_mask * mask, 'count'].mul(a.loc[row_mask * mask, 'mean'].values)
    d = b.loc[row_mask * mask, 'count'] + a.loc[row_mask * mask, 'count']

    b.loc[row_mask * mask, 'mean'] = c / d
    b.loc[:, 'count'] = b.loc[:, 'count'] + a.loc[:, 'count']

    b.loc[~mask, 'mean'] = a.loc[~mask, 'mean']

    c_filled = len(b) - b.loc[:, 'mean'].isnull().sum()
    c_mean = b.loc[:, 'mean'].dropna().mean()

    print('Original mean is %.3f, original filled is %d\nCombined mean is %.3f, combined filled is %d\n' % (
    or_mean, or_filled, c_mean, c_filled))
    return b


def range_unnest(df, col, out_col_name=None, reset_index=False):
    assert len(df.index.names) == 1, "Does not support multi-index."
    if out_col_name is None: out_col_name = col

    col_flat = pd.DataFrame(
        [[i, x] for i, y in df[col].iteritems() for x in range(y + 1)],
        columns=[df.index.names[0], out_col_name]
    )

    if not reset_index: col_flat = col_flat.set_index(df.index.names[0])
    return col_flat


def process_query_results(df, fill_df):
    df = df.groupby(ID_COLS + ['hours_in']).agg(['mean', 'count'])
    df.index = df.index.set_levels(df.index.levels[1].astype(int), level=1)
    df = df.reindex(fill_df.index)
    return df


ID_COLS = ['subject_id', 'hadm_id', 'stay_id']
ITEM_COLS = ['itemid', 'label', 'LEVEL1', 'LEVEL2']


def compile_intervention(df_copy, c):
    # df_copy = df.copy(deep=True)
    df_copy['max_hours'] = (df_copy['icu_outtime'] - df_copy['icu_intime']).apply(to_hours)
    df_copy.loc[:, 'starttime'] = df_copy.loc[:, ['starttime', 'icu_intime']].max(axis=1)
    df_copy.loc[:, 'endtime'] = df_copy.loc[:, ['endtime', 'icu_outtime']].min(axis=1)
    df_copy['starttime'] = df_copy['starttime'] - df_copy['icu_intime']
    df_copy['starttime'] = df_copy.starttime.apply(lambda x: x.days * 24 + x.seconds // 3600)
    df_copy['endtime'] = df_copy['endtime'] - df_copy['icu_intime']
    df_copy['endtime'] = df_copy.endtime.apply(lambda x: x.days * 24 + x.seconds // 3600)
    if c == 'antibiotics':
        df_copy = df_copy.groupby('stay_id').apply(add_antibitics_indicators)
    else:
        df_copy = df_copy.groupby('stay_id').apply(add_outcome_indicators)

    df_copy.rename(columns={'on': c}, inplace=True)
    # heparin_2.rename(columns={'values': c + ' conc'}, inplace=True)
    df_copy = df_copy.reset_index(level='stay_id')
    return df_copy


def add_outcome_indicators(out_gb):
    subject_id = out_gb['subject_id'].unique()[0]
    hadm_id = out_gb['hadm_id'].unique()[0]
    # icustay_id = out_gb['stay_id'].unique()[0]
    max_hrs = out_gb['max_hours'].unique()[0]
    on_hrs = set()
    on_values = []

    # p_set = on_hrs.copy()
    for index, row in out_gb.iterrows():
        on_hrs.update(range(row['starttime'], row['endtime'] + 1))
        # if on_hrs - p_set:
        # only when sets updates, append a value
        # on_values.append([row['values']]*len(on_hrs - p_set))
        # p_set = on_hrs.copy()

    off_hrs = set(range(max_hrs + 1)) - on_hrs
    ##values flatten a nested list
    # values = [0]*len(off_hrs) + [item for sublist in on_values for item in sublist]
    on_vals = [0] * len(off_hrs) + [1] * len(on_hrs)
    hours = list(off_hrs) + list(on_hrs)
    return pd.DataFrame({'subject_id': subject_id, 'hadm_id': hadm_id,
                         'hours_in': hours, 'on': on_vals})  # icustay_id': icustay_id})


def add_antibitics_indicators(out_gb):
    subject_id = out_gb['subject_id'].unique()[0]
    hadm_id = out_gb['hadm_id'].unique()[0]
    # icustay_id = out_gb['stay_id'].unique()[0]
    max_hrs = out_gb['max_hours'].unique()[0]
    on_hrs = set()
    on_values = []
    on_route = []

    p_set = on_hrs.copy()
    for index, row in out_gb.iterrows():
        on_hrs.update(range(row['starttime'], row['endtime'] + 1))
        if on_hrs - p_set:
            # only when sets updates, append a value
            on_values.append([row['antibiotic']] * len(on_hrs - p_set))
            on_route.append([row['route']] * len(on_hrs - p_set))

        p_set = on_hrs.copy()

    off_hrs = set(range(max_hrs + 1)) - on_hrs
    ##values flatten a nested list
    values = [np.nan] * len(off_hrs) + [item for sublist in on_values for item in sublist]
    route = [np.nan] * len(off_hrs) + [item for sublist in on_route for item in sublist]
    # on_vals = [0]*len(off_hrs) + [1]*len(on_hrs)
    hours = list(off_hrs) + list(on_hrs)
    return pd.DataFrame({'subject_id': subject_id, 'hadm_id': hadm_id,
                         'hours_in': hours, 'antibiotic': values, 'route': route})  # icustay_id': icustay_id})


def add_blank_indicators(out_gb):
    subject_id = out_gb['subject_id'].unique()[0]
    hadm_id = out_gb['hadm_id'].unique()[0]
    # icustay_id = out_gb['icustay_id'].unique()[0]
    max_hrs = out_gb['max_hours'].unique()[0]

    hrs = range(max_hrs + 1)
    vals = list([0] * len(hrs))
    return pd.DataFrame({'subject_id': subject_id, 'hadm_id': hadm_id,
                         'hours_in': hrs, 'on': vals})  # 'icustay_id': icustay_id,


def continuous_outcome_processing(out_data, data, icustay_timediff):
    """
    Args
    ----
    out_data : pd.DataFrame
        index=None
        Contains subset of icustay_id corresp to specific sessions where outcome observed.
    data : pd.DataFrame
        index=icustay_id
        Contains full population of static demographic data
    Returns
    -------
    out_data : pd.DataFrame
    """
    out_data['icu_intime'] = out_data['stay_id'].map(data['icu_intime'].to_dict())
    out_data['icu_outtime'] = out_data['stay_id'].map(data['icu_outtime'].to_dict())
    out_data['max_hours'] = out_data['stay_id'].map(icustay_timediff)
    out_data['starttime'] = out_data['starttime'] - out_data['icu_intime']
    out_data['starttime'] = out_data.starttime.apply(lambda x: x.days * 24 + x.seconds // 3600)
    out_data['endtime'] = out_data['endtime'] - out_data['icu_intime']
    out_data['endtime'] = out_data.endtime.apply(lambda x: x.days * 24 + x.seconds // 3600)
    out_data = out_data.groupby(['stay_id'])

    return out_data