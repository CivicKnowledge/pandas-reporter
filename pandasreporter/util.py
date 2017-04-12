


def melt(df):
    """Melt a census dataframe into two value columns, for the estimate and margin"""
    import pandas as pd

    # Intial melt
    melted = pd.melt(df, id_vars=list(df.columns[:9]), value_vars=list(df.columns[9:]))
    melted = melted[['gvid', 'variable', 'value']]

    # Make two seperate frames for estimates and margins.
    estimates = melted[~melted.variable.str.contains('_m90')].set_index(['gvid', 'variable'])
    margins = melted[melted.variable.str.contains('_m90')].copy()

    margins.columns = ['gvid', 'ovariable', 'm90']
    margins['variable'] = margins.ovariable.str.replace('_m90', '')

    # Join the estimates to the margins.
    final = estimates.join(margins.set_index(['gvid', 'variable']).drop('ovariable', 1))

    return final