# Copyright (c) 2017 Civic Knowledge. This file is licensed under the terms of the
# MIT License, included in this distribution as LICENSE
"""

"""

import six
import pandas as pd
import numpy as np
import requests
from operator import itemgetter
from .util import get_cache, slugify

def get_dataframe(table_id, summary_level,geoid, cache=True):
    """

    :param table_id: Census table id, ex: 'B01001'
    :param summary_level: A summary level number or string, ex: 140
    :param geoid: Geoid of the containing region. ex '05000US06073' for San Diego county
    :param cache: If true, cache the response from Census Reporter ( Fast and Friendly! )
    :return:
    """
    import json
    from itertools import repeat

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


class CensusDataFrame(pd.DataFrame):

    _metadata = ['schema']

    def __init__(self, data=None, index=None,  columns=None, dtype=None, copy=False, schema=None):

        self.schema = schema

        if columns is None and self.schema is not None:
            columns = [c['code'] for c in self.schema]

        super(CensusDataFrame, self).__init__(data, index, columns, dtype, copy)


    @property
    def titled_columns(self):
        """Return a copy that uses titles for column headings"""
        if not self.schema:
            return self

        return self.rename(index=str,
                           columns = dict(zip(self.columns, [c['title'] for c in self.schema])),
                           inplace=False)

    @property
    def coded_columns(self):
        """Return a copy that uses codes for column headings"""
        if not self.schema:
            return self

        return self.rename(index=str,
                           columns=dict(zip(self.columns, [c['code'] for c in self.schema])),
                           inplace=False)

    @property
    def ct_columns(self):
        """Return a copy that uses codes and titles for column headings"""

        if not self.schema:
            return self

        return self.rename(index=str,
                           columns=dict(zip(self.columns, [c['code_title'] for c in self.schema])),
                           inplace=False)

    def lookup_schema(self, key):
        """Return a colum either by it's actuall column name, or it's position in the census table,
        the last three digits of the column code"""

        if self.schema:
            for c in self.schema:
                if (key == c['name'] or key == c['code'] or key == c['title'] or
                            key == c['index'] or key == c['position']):
                    return c
        else:
            for i, c in enumerate(self.columns):
                if key == c:
                    return {
                        'name': c,
                        'title': c,
                        'code': c,
                        'code_title': c,
                        'indent': 0,
                        'index': str(i).zfill(3),
                        'position': i
                    }


        raise KeyError("did not find key {}".format(key))


    def lookup(self, key):
        """Return a column either by it's actuall column name, or it's position in the census table,
        the last three digits of the column code"""

        schema = self.lookup_schema(key)
        c = self.ix[:, schema['position']]
        c.parent_frame = self
        c.schema = schema

        return c

    @property
    def rows(self):
        """Yield rows like a partition does, with a header first, then rows. """

        yield [self.index.name] + list(self.columns)

        for t in self.itertuples():
            yield list(t)


    def sum_m(self, *cols):
        """Sum a set of Dataframe series and return the summed series and margin. The series must have names"""

        # See the ACS General Handbook, Appendix A, "Calculating Margins of Error for Derived Estimates".
        # (https://www.census.gov/content/dam/Census/library/publications/2008/acs/ACSGeneralHandbook.pdf)
        # for a guide to these calculations.

        if len(cols) == 1 and isinstance(cols[0], (list, tuple)):
            cols = cols[0]

        cols = [ self.lookup(c) for c in cols]

        estimates = sum(cols)

        margins = np.sqrt(sum(c.m90**2 for c in cols))

        return estimates, margins

    def add_sum_m(self, col_name, *cols):
        """
        Add new columns for the sum, plus error margins, for 2 or more other columns

        The routine will add two new columns, one named for col_name, and one for <col_name>_m90

        :param col_name: The base name of the new column
        :param cols:
        :return:
        """

        self[col_name], self[col_name+'_m90'] = self.sum_m(*cols)


    def add_rse(self, *col_name):
        """
        Create a new column, <col_name>_rse for Relative Standard Error, using <col_name> and <col_name>_m90

        :param col_name:
        :return:

        """

        for cn in col_name:
            self[cn + '_rse'] = self[cn].rse

    def sum_col_range(self, first, last):
        """Sum a contiguous group of columns, and return the sum and the new margins.  """

        c1 = self.lookup(first)
        c2 = self.lookup(last)

        cols = self.ix[:,c1.col_position:c2.col_position+1]

        estimates = sum(cols)

        margins = np.sqrt(np.sum(c.m90**2 for c in cols))

        return estimates, margins

    def ratio(self, n, d):
        """
        Calculate a proportion. The numerator should not be a subset of the denominator,
        such as the ratio of males to females. If it is  a subset, use proportion().

        :param n: The Numerator, a string, CensusSeries or tuple
        :param d: The Denominator, a string, CensusSeries or tuple
        :return: a tuple of series, the estimates and the margins
        """

        return self._ratio(n,d,subset = False)

    def proportion(self, n, d):
        """
        Calculate a proportion. The numerator must be a subset of the denominator,  such
        as the proportion of females to the total population. If it is not a subset, use ratio().

        ( I think "subset" mostly means that the numerator < denominator )

        :param n: The Numerator, a string, CensusSeries or tuple
        :param d: The Denominator, a string, CensusSeries or tuple
        :return: a tuple of series, the estimates and the margins
        """

        return self._ratio(n, d, subset=True)

    def normalize(self, x):
        """Convert any of the numerator and denominator forms into a consisten
        tuple form"""

        if isinstance(x, tuple):
            return self.lookup(x[0]), self.lookup(x[1])

        elif isinstance(x, six.string_types):
            return self.lookup(x).value, self.lookup(x).m90

        elif isinstance(x, CensusSeries):
            return x.value, x.m90

        else:
            raise ValueError("Don't know what to do with a {}".format(type(x)))

    def _ratio(self, n, d, subset=True):
        """
        Compute a ratio of a numerator and denominator, propagating errors

        Both arguments may be one of:
        * A CensusSeries for the estimate
        * a string that can be resolved to a colum with .lookup()
        * A tuple of names that resolve with .lookup()

        In the tuple form, the first entry is the estimate and the second is the 90% margin

        :param n: The Numerator, a string, CensusSeries or tuple
        :param d: The Denominator, a string, CensusSeries or tuple
        :return: a tuple of series, the estimates and the margins
        """

        n, n_m90 = self.normalize(n)
        d, d_m90 = self.normalize(d)

        rate = n.astype(float) / d.astype(float)

        if subset:
            try:
                # From external_documentation.acs_handbook, Appendix A, "Calculating MOEs for
                # Derived Proportions". This is for the case when the numerator is a subset of the
                # denominator

                # In the case of a neg arg to a square root, the acs_handbook recommends using the
                # method for "Calculating MOEs for Derived Ratios", where the numerator
                # is not a subset of the denominator. Since our numerator is a subset, the
                # handbook says " use the formula for derived ratios in the next section which
                # will provide a conservative estimate of the MOE."
                # The handbook says this case should be rare, but for this calculation, it
                # happens about 50% of the time.

                # Normal calc, from the handbook
                sqr = n_m90 ** 2 - ((rate ** 2) * (d_m90 ** 2))

                # When the sqr value is < 0, the sqrt will fail, so use the other calc in those cases
                sqrn = sqr.where(sqr > 0, n_m90 ** 2 + ((rate ** 2) * (d_m90 ** 2)) )

                # Aw, hell, just punt.
                sqrnz = sqrn.where(sqrn > 0, float('nan'))

                rate_m = np.sqrt(sqrnz) / d

            except ValueError:

                return self._ratio(n, d, False)


        else:
            rate_m = np.sqrt(n_m90 ** 2 + ((rate ** 2) * (d_m90 ** 2))) / d

        return rate, rate_m

    def product(self, a, b):

        a, a_m90 = self.normalize(a)
        b, b_m90 = self.normalize(b)

        p = a * b

        margin = np.sqrt(a**2 *  b_m90**2 + b**2 * a_m90**2)

        return p, margin


    def dim_columns(self, pred):
        """
        Return a list of columns that have a particular value for age,
        sex and race_eth. The `pred` parameter is a string of python
        code which is evaled, with the classification dict as the local
        variable context, so the code string can access these variables:

        - sex
        - age
        - race-eth
        - col_num

        Col_num is the number in the last three digits of the column name

        Some examples of predicate strings:

        - "sex == 'male' and age != 'na' "

        :param pred: A string of python code that is executed to find column matches.

        """

        from .dimensions import classify

        out_cols = []

        for i, c in enumerate(self.partition.table.columns):
            if c.name.endswith('_m90'):
               continue

            if i < 9:
                continue

            cf = classify(c)
            cf['col_num'] = int(c.name[-3:])

            if eval(pred, {}, cf):
                out_cols.append(c.name)

        return out_cols

    def __getitem__(self, key):
        """

        """
        from pandas import DataFrame, Series
        from .series import CensusSeries

        result = super(CensusDataFrame, self).__getitem__(key)

        if isinstance(result, DataFrame):
            result.__class__ = CensusDataFrame
            result._dataframe = self

        elif isinstance(result, Series):
            result.__class__ = CensusSeries
            result._dataframe = self

        return result

    def copy(self, deep=True):

        r = super(CensusDataFrame, self).copy(deep)
        r.__class__ = CensusDataFrame
        r.schema = self.schema

        return r

    def groupby(self, by=None, axis=0, level=None, as_index=True, sort=True,
                group_keys=True, squeeze=False, **kwargs):
        """
        Overrides groupby() to return CensusDataFrameGroupBy

        """
        from .groupby import groupby

        if level is None and by is None:
            raise TypeError("You have to supply one of 'by' and 'level'")
        axis = self._get_axis_number(axis)

        return groupby(self, by=by, axis=axis, level=level, as_index=as_index,
                       sort=sort, group_keys=group_keys, squeeze=squeeze,
                       **kwargs)

    ##
    ## Extension Points
    ##

    @property
    def _constructor(self):
        return CensusDataFrame

    @property
    def _constructor_sliced(self):
        from .series import CensusSeries
        return CensusSeries

    def _getitem_column(self, key):
        """ Return a column from a name

        :param key:
        :return:
        """
        schema = self.lookup_schema(key)

        c = super(CensusDataFrame, self)._getitem_column(self.columns[schema['position']])
        c.parent_frame = self
        c.schema = schema

        return c

    def _getitem_array(self, key):
        """Return a set of columns"""

        if isinstance(key, list):

            augmented_key = []

            for col_name in key:
                augmented_key.append(col_name)

                m90 = col_name+'_m90'

                try:
                    self.lookup(m90)
                    augmented_key.append(m90)
                except KeyError:
                    pass

            return super()._getitem_array(augmented_key)
        else:
            return super()._getitem_array(key)



