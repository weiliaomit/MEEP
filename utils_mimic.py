import pandas as pd

def range_unnest(df, col, out_col_name=None, reset_index=False):
    '''

    :param df:
    :param col:
    :param out_col_name:
    :param reset_index:
    :return:
    '''
    assert len(df.index.names) == 1, "Does not support multi-index."
    if out_col_name is None: out_col_name = col

    col_flat = pd.DataFrame(
        [[i, x] for i, y in df[col].iteritems() for x in range(y+1)],
        columns=[df.index.names[0], out_col_name]
    )

    if not reset_index: col_flat = col_flat.set_index(df.index.names[0])
    return col_flat

def process_query_results(df, fill_df):
    '''

    :param df:
    :param fill_df:
    :return:
    '''
    df = df.groupby(ID_COLS + ['hours_in']).agg(['mean', 'count'])
    df.index = df.index.set_levels(df.index.levels[1].astype(int), level=1)
    df = df.reindex(fill_df.index)
    return df