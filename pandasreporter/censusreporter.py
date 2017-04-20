# Copyright (c) 2017 Civic Knowledge. This file is licensed under the terms of the
# MIT License, included in this distribution as LICENSE
"""
Return dataframes from the Census Reporter API
"""

import requests
from operator import itemgetter



def get_cr_dataframe(table_id, summary_level,geoid, cache=True):
    """

    :param table_id: Census table id, ex: 'B01001'
    :param summary_level: A summary level number or string, ex: 140
    :param geoid: Geoid of the containing region. ex '05000US06073' for San Diego county
    :param cache: If true, cache the response from Census Reporter ( Fast and Friendly! )
    :return:
    """
    import json
    from itertools import repeat
    from .util import get_cache, slugify
    from .dataframe import CensusDataFrame

    cache_fs = get_cache()

    url = "http://api.censusreporter.org/1.0/data/show/latest?table_ids={table_id}&geo_ids={sl}|{geoid}"\
            .format(table_id=table_id, sl=summary_level, geoid=geoid)

    cache_key = slugify(url)


    if cache and cache_fs.exists(cache_key):
        data = json.loads(cache_fs.gettext(cache_key))
    else:
        data = requests.get(url).json()

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
            'index': '   ', # Index in census table
            'position': 0 # Index in dataframe
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

    if not 'tables' in data:
        print(json.dumps(data, indent=4))

    column_codes = sorted(data['tables'][table_id]['columns'].keys())

    for column in column_codes:

        name = data['tables'][table_id]['columns'][column]['name']
        indent = data['tables'][table_id]['columns'][column]['indent']

        index =  column[-3:]

        if len(title_stack) <= indent:
            title_stack.extend(repeat('', indent-len(title_stack)+1))
        elif len(title_stack) > indent:
            title_stack = title_stack[:indent+1]

        title_stack[indent] = name.replace(':','')

        columns.append({
            'name': name,
            'title': ' '.join(title_stack),
            'code': column,
            'code_title': column+" "+' '.join(title_stack),
            'indent': indent,
            'index': index,
            'position': len(columns)})

        columns.append({
            'name': "Margins for " + name,
            'title': "Margins for " + ' '.join(title_stack),
            'code': column+"_m90",
            'code_title': "Margins for "+column + " " + ' '.join(title_stack),
            'indent': indent,
            'index': index,
            'position': len(columns)

        })

    rows = []

    row_ig = itemgetter(*column_codes)

    d = data['data']

    for geo in data['data'].keys():
        row = [geo, data['geography'][geo]['name']]
        for e, m in zip(row_ig(d[geo][table_id]['estimate']), row_ig(d[geo][table_id]['error'])):
            row.append(e)
            row.append(m)
        rows.append(row)

    assert len(rows) == 0 or len(columns) == len(rows[0])

    return CensusDataFrame(rows, schema=columns)

