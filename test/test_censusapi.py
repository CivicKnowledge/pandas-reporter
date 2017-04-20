import unittest
import pandasreporter as pr
import pandas as pd
import numpy as np


def test_data(*paths):
    from os.path import dirname, join, abspath

    return abspath(join(dirname(abspath(__file__)), 'test_data', *paths))


class BasicTests(unittest.TestCase):
    def test_basic(self):
        api = pr.CensusApi()

        # print(api.metadata)

        # for d in api.search_datasets('poverty'):
        #    print(d.identifier, d.title)

        print(api.search_datasets('Vintage 2015 Population Estimates'))

        ds = api.get_dataset('POPESTpop2015')

        print(ds)

        print(ds.variables)

        print(ds.fetch_url('PLACE','GEONAME','POP', geo_in='state:06+county:073', geo_for='place:*'))

        print(ds.fetch('PLACE', 'GEONAME', 'POP', geo_in='state:06+county:073', geo_for='place:*'))

        print(ds.fetch_dataframe('PLACE', 'GEONAME', 'POP', geo_in='state:06+county:073', geo_for='place:*'))

if __name__ == '__main__':
    unittest.main()
