import unittest
import pandasreporter as pr
import pandas as pd
import numpy as np


def test_data(*paths):
    from os.path import dirname, join, abspath

    return abspath(join(dirname(abspath(__file__)), 'test_data', *paths))


class BasicTests(unittest.TestCase):

    def test_basic(self):

        df = pr.get_cr_dataframe('B01001', '140',  '05000US06073', cache=True)

        self.assertEquals(37292.0, df.lookup(84).sum())
        self.assertEquals(df.B01001042.sum(), df.lookup('042').sum())

        self.assertEquals('B01001042', df.lookup('042').census_code)
        self.assertEquals('Total Female 60 and 61 years', df.lookup('042').census_title)

        self.assertEquals(2806.0, df.lookup('042').m90.sum())
        self.assertEquals(37292.0, df.lookup('042').m90.value.sum())

    def make_df(self):
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
                'name': 'group',
                'code': 'group',
                'title': 'group',
                'code_title': 'group',
                'indent': 0,
                'index': '   ',
                'position': 1
            }

        ]

        for i in range(4):
            columns.append({
                'name': "col" + str(i),
                'title': "col" + str(i),
                'code': "col" + str(i),
                'code_title': "col" + str(i),
                'indent': 0,
                'index': i,
                'position': len(columns)})

            columns.append({
                'name': "col" + str(i) + "_m90",
                'title': "Margins for col" + str(i),
                'code': "col" + str(i) + "_m90",
                'code_title': "Margins for col" + str(i),
                'indent': 0,
                'index': i,
                'position': len(columns)
            })

        rows = []
        for i in range(1,11):
            row = [str(i), str(i%3)]
            for j in range(1,5):
                row.append(i*j)
                row.append(7*i/(3*j))
            rows.append(row)

        return pr.CensusDataFrame(rows, schema=columns)

    def test_basic_math(self):

        pd.set_option('display.width', 120)
        pd.set_option('display.precision', 3)
        df = self.make_df()

        odf = pr.CensusDataFrame()

        odf['col0'], odf['col0_m90'] = df['col0'], df['col0_m90']
        odf['col1'], odf['col1_m90'] = df['col1'], df['col1_m90']
        odf['r'], odf['r_m90'] = df.ratio('col0', 'col1')
        odf['p'],odf['p_m90'] = df.proportion('col0', 'col1')
        odf['s1'], odf['s1_m90'] = df.sum_m('col0','col1')
        odf['m'], odf['m_m90'] = df.product('col0', 'col1')

        odf.add_rse('s1')

        #print(odf)

        self.assertAlmostEquals(52.8621, odf.s1_rse.mean(), places=4)
        self.assertAlmostEqual(14.3481, odf.s1_m90.mean(), places=4)
        self.assertEquals(0.5, odf.r.mean().round(3))

        # For aggregates, the MOE is just root sum of squares.
        moe = np.sqrt( odf['col0_m90']**2 + odf['col1_m90']**2)
        self.assertEquals(list(moe), list(odf['s1_m90']))

        # For proportions ( N < D )
        p = df.col0 / df.col1
        moe = np.sqrt(df.col0_m90**2 - ( p**2 * df.col1_m90**2 )) / df.col1

        self.assertEquals(list(moe), list(odf['p_m90']))

        # For ratios
        r = df.col0 / df.col1
        moe = np.sqrt(df.col0_m90 ** 2 + (r ** 2 * df.col1_m90 ** 2)) / df.col1
        self.assertEquals(list(moe), list(odf['r_m90']))

    def test_basic_math_inv(self):
        """Just check that there are no runtime warnings"""
        pd.set_option('display.width', 120)
        pd.set_option('display.precision', 3)
        df = self.make_df()

        odf = pr.CensusDataFrame()

        odf['col0'], odf['col0_m90'] = df['col0'], df['col0_m90']
        odf['col1'], odf['col1_m90'] = df['col1'], df['col1_m90']

        # Inverting the numerator and denom of the previous
        #odf['ri'], odf['r_m90'] = df.ratio('col1', 'col0')
        odf['pi'], odf['p_m90'] = df.proportion('col1', 'col0')

    def test_math_handbook(self):
        """Test using the examples from the ACS Handbook, using
        https://www.census.gov/content/dam/Census/library/publications/2008/acs/ACSGeneralHandbook.pdf"""

        # Aggregates, Table 1, Page A-14

        df = pr.CensusDataFrame(pd.read_csv(test_data('agg.csv')))
        df['agg'], df['agg_m90'] = df.sum_m('a','b','c')

        self.assertAlmostEqual(89008, df.ix[0,'agg'])
        self.assertAlmostEqual(4289, df.ix[0,'agg_m90'], places=0)

        # Proportions, Table 2, Page A-15

        df = pr.CensusDataFrame(pd.read_csv(test_data('prop.csv')))
        df['prop'], df['prop_m90'] = df.proportion('a', 'b')

        self.assertAlmostEqual(0.1461, df.ix[0, 'prop'], places=3)
        self.assertAlmostEqual(0.0311, df.ix[0, 'prop_m90'], places=4)

        # Ratio, Table 3, Page A-16

        df = pr.CensusDataFrame(pd.read_csv(test_data('ratio.csv')))
        df['ratio'], df['ratio_m90'] = df.ratio('a', 'b')

        # The handbook has .7200 and 0.2135, but the rounding error in
        # calculating with .7200 gives them 0.2135 instead of 0.2136
        self.assertAlmostEqual(.719565, df.ix[0, 'ratio'], places=4)
        self.assertAlmostEqual(0.213545, df.ix[0, 'ratio_m90'], places=4)

        # Product, Table 4, Page a-16

        df = pr.CensusDataFrame(pd.read_csv(test_data('product.csv')))
        df['product'], df['product_m90'] = df.product('a', 'b')

        # The handbook has .7200 and 0.2135, but the rounding error in
        # calculating with .7200 gives them 0.2135 instead of 0.2136
        self.assertAlmostEqual(6784, df.ix[0, 'product'], places=0)
        self.assertAlmostEqual(1405, df.ix[0, 'product_m90'], places=0)


    def test_array_index(self):
        """Test that array indexing brings along margin columns"""
        pd.set_option('display.width', 120)
        pd.set_option('display.precision', 3)
        df = self.make_df()

        df2 = df[['geoid','col0', 'col1']]

        columns = list(df2.columns)

        self.assertEquals(['geoid', 'col0', 'col0_m90', 'col1', 'col1_m90'], columns)

        # But not if it is re-cast to a normal Dataframe.
        df3 = pd.DataFrame(df)

        df4 = df3[['geoid', 'col0', 'col1']]

        columns = list(df4.columns)

        self.assertEquals(['geoid', 'col0', 'col1'], columns)

    def test_row_agg(self):

        from pandasreporter.func import  sum_rs
        pd.set_option('display.width', 120)

        df = self.make_df()

        # Don't really know how to determine the correct values ....
        # so ... faith ...
        print(df.groupby('group').sum())

        print(df.groupby('group').mean())

    def test_index(self):

        df = pr.get_cr_dataframe('B01001', '140', '05000US06073', cache=True).ct_columns

        df2 = df[['geoid','B01001001', 'B01001002']]

        print(df2.head())

        df3 = df['B01001001']

        print(df3.head())

    def test_varrep(self):

        df = pr.get_varrep_dataframe(2015,  'B01001', '140' , state='11', cache=False) # Washington DC -- the smallest file

        self.assertEqual(8771, len(df))

        self.assertEqual(13, pr.get_ave_weight(11))

        f = pr.get_k_val_f()

        self.assertEqual(4, f(1000))
        self.assertEqual(10, f(10000))
        self.assertEqual(18, f(49999))
        self.assertEqual(22, f(50001))

if __name__ == '__main__':
    unittest.main()
