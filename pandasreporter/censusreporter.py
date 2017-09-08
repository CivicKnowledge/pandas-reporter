# Copyright (c) 2017 Civic Knowledge. This file is licensed under the terms of the
# MIT License, included in this distribution as LICENSE
"""
Return dataframes from the Census Reporter API
"""

import requests
from operator import itemgetter
from rowgenerators import SourceSpec
from rowgenerators import Url
from rowgenerators import Source
from rowgenerators import register_proto


def split_url(url):
    if isinstance(url, SourceSpec):
        ss = url
    else:
        ss = SourceSpec(str(url))

    table_id = ss.url_parts.netloc

    summary_level, geoid = ss.url_parts.path.strip('/').split('/')

    return table_id, summary_level, geoid


def format_url(table_id, summary_level, geoid):
    return "http://api.censusreporter.org/1.0/data/show/latest?table_ids={table_id}&geo_ids={sl}|{geoid}" \
        .format(table_id=table_id, sl=summary_level, geoid=geoid)


def convert_url(url):
    return format_url(*split_url(url))


try:

    class CensusReporterSource(Source):
        """A RowGenerator source that can be registered for Census REporter URLs.

        To install it:

        > from rowgenerators import register_proto
        > register_proto('censusreporter', CensusReporterSource)

        Then, this class will be used for urls of the form:

            censusreporter:B17001/140/05000US06073

        or, Generically:

            censusreporter:<table_id>/<summary_level>/<geoid>

        """

        # noinspection PyUnusedLocal
        def __init__(self, spec, download_f, cache):
            super(CensusReporterSource, self).__init__(spec, cache)

            self.spec = spec

            self.table_id = spec.url_parts.netloc

            self.summary_level, self.geoid = spec.url_parts.path.strip('/').split('/')

            self.columns = []

        # noinspection PyUnusedLocal
        def dataframe(self, limit=None):
            """
            Return a CensusReporterDataframe
            :param limit: Limit is ignored
            :return:
            """
            return get_cr_dataframe(self.table_id, self.summary_level, self.geoid, cache=self.cache)

        def __iter__(self):
            rows, self.columns, release = get_cr_rows(self.spec, cache=self.cache)

            yield [e['code'] for e in self.columns]

            for row in rows:
                yield row


    register_proto('censusreporter', CensusReporterSource)


except ImportError:
    pass


def get_cr_rows(spec, cache=True, **kwargs):
    return _get_cr_rows(*split_url(spec), cache=cache, **kwargs)


# noinspection PyUnusedLocal
def _get_cr_rows(table_id, summary_level, geoid, cache=True, **kwargs):
    """

    :param table_id: Census table id, ex: 'B01001'
    :param summary_level: A summary level number or string, ex: 140
    :param geoid: Geoid of the containing region. ex '05000US06073' for San Diego county
    :param cache: If true, cache the response from Census Reporter ( Fast and Friendly! )
    :param kwargs: Catchall so dict can be expanded into the signature.
    :return:
    """
    import json
    from itertools import repeat
    from .util import get_cache, slugify

    if cache is True:
        cache_fs = get_cache()
    elif cache:
        cache_fs = cache
    else:
        cache_fs = False

    url = format_url(table_id, summary_level, geoid)

    cache_key = slugify(url)

    if cache and cache_fs.exists(cache_key):
        data = json.loads(cache_fs.gettext(cache_key))
    else:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()

        if cache:
            cache_fs.settext(cache_key, json.dumps(data, indent=4))

    # It looks like the JSON dicts may be properly sorted, but I'm not sure I can rely on that.
    # So, sort the column id values, then make a columns title list in the same order

    columns = [
        {
            'name': 'geoid',
            'code': 'geoid',
            'title': 'geoid',
            'code_title': 'geoid',
            'indent': 0,
            'index': '   ',  # Index in census table
            'position': 0  # Index in dataframe
        }, {
            'name': 'name',
            'code': 'name',
            'title': 'name',
            'code_title': 'name',
            'indent': 0,
            'index': '   ',
            'position': 1
        }
    ]

    title_stack = []

    if 'tables' not in data:
        print(json.dumps(data, indent=4))

    # SOme of the column codes have '.' in them; those are supposed to be headers, not real columns
    column_codes = sorted(c for c in data['tables'][table_id]['columns'].keys() if '.' not in c)

    for column in column_codes:

        name = data['tables'][table_id]['columns'][column]['name']
        indent = data['tables'][table_id]['columns'][column]['indent']

        index = column[-3:]

        if len(title_stack) <= indent:
            title_stack.extend(repeat('', indent - len(title_stack) + 1))
        elif len(title_stack) > indent:
            title_stack = title_stack[:indent + 1]

        title_stack[indent] = name.replace(':', '')

        columns.append({
            'name': name,
            'title': ' '.join(title_stack),
            'code': column,
            'code_title': column + " " + ' '.join(title_stack),
            'indent': indent,
            'index': index,
            'position': len(columns)})

        columns.append({
            'name': "Margins for " + name,
            'title': "Margins for " + ' '.join(title_stack),
            'code': column + "_m90",
            'code_title': "Margins for " + column + " " + ' '.join(title_stack),
            'indent': indent,
            'index': index,
            'position': len(columns)

        })

    rows = []

    row_ig = itemgetter(*column_codes)

    d = data['data']

    for geo in data['data'].keys():

        row = [geo, data['geography'][geo]['name']]

        ests = row_ig(d[geo][table_id]['estimate'])
        errs = row_ig(d[geo][table_id]['error'])

        # Some tables have only one column
        if not isinstance(ests, (list, tuple)):
            ests = [ests]

        if not isinstance(errs, (list, tuple)):
            errs = [errs]

        for e, m in zip(ests, errs):
            row.append(e)
            row.append(m)
        rows.append(row)

    assert len(rows) == 0 or len(columns) == len(rows[0])

    return rows, columns, data['release']


def get_cr_dataframe(table_id, summary_level, geoid, cache=True, **kwargs):
    from .dataframe import CensusDataFrame

    rows, columns, release = _get_cr_rows(table_id, summary_level, geoid, cache=cache, **kwargs)

    df = CensusDataFrame(rows, schema=columns)

    df.release = release

    return df


def make_citation_dict(t):
    """
    Return a dict with BibText key/values
    :param t:
    :return:
    """

    from nameparser import HumanName
    import datetime

    try:
        if Url(t.url).proto == 'censusreporter':

            try:
                url = str(t.resolved_url.url)
            except AttributeError:
                url = t.url


            return {
                'type': 'dataset',
                'name': t.name,
                'origin': 'United States Census Bureau',
                'publisher': 'CensusReporter.org',
                'title': "2010 - 2015 American Community Survey, Table {}: {}".format(t.name.split('_', 1).pop(0), t.description),
                'year': 2015,
                'accessDate': '{}'.format(datetime.datetime.now().strftime('%Y-%m-%d')),
                'url': convert_url(url)
            }
    except (AttributeError, KeyError) as e:

        pass


    return False
