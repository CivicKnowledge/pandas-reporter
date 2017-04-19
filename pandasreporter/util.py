


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

# From http://stackoverflow.com/a/295466
def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.type(
    """
    import re
    import unicodedata
    from six import text_type
    value = text_type(value)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('utf8')
    value = re.sub(r'[^\w\s-]', '-', value).strip().lower()
    value = re.sub(r'[-\s]+', '-', value)
    return value


CACHE_NAME = 'pandasreporter'

def get_cache(cache_name=CACHE_NAME):
    """Return the path to a file cache"""

    from fs.osfs import OSFS
    from fs.appfs import UserDataFS
    import os

    env_var = (cache_name+'_cache').upper()

    cache_dir = os.getenv(env_var, None)

    if cache_dir:
        return OSFS(cache_dir)
    else:
        return UserDataFS(cache_name.lower())

def clean_cache(cache = None, cache_name=CACHE_NAME):
    """Clean out a named cache"""
    import datetime

    cache = cache if cache else get_cache(cache_name)

    for step in cache.walk.info():
        details = cache.getdetails(step[0])
        mod = details.modified
        now = datetime.datetime.now(tz=mod.tzinfo)
        age = (now - mod).total_seconds()
        if age > (60 * 60 * 4) and details.is_file:
            cache.remove(step[0])
