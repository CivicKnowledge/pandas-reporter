import unittest
from appurl import parse_app_url, get_cache, match_url_classes, AppUrlError

from rowgenerators import get_generator
import pandasreporter as pr
import pandas as pd
import numpy as np
from pkg_resources import iter_entry_points

def test_data(*paths):
    from os.path import dirname, join, abspath

    return abspath(join(dirname(abspath(__file__)), 'test_data', *paths))

class BasicTests(unittest.TestCase):

    def test_basic(self):


        for us in ('censusreporter:B17001/140/05000US06073','censusreporter:/B17001/140/05000US06073',
                   'censusreporter://B17001/140/05000US06073','censusreporter://B17001/140/05000US06073/'):

            u = parse_app_url(us)
            self.assertEqual(str(u), str(parse_app_url(str(u))),us)
            self.assertEqual('B17001', u.table_id,us)
            self.assertEqual('140', u.summary_level,us)
            self.assertEqual('05000US06073', u.geoid,us)

        for us in ('censusreporter:B17001', 'censusreporter:/B17001/140/',
                   'censusreporter://B17001/', 'censusreporter://B17001/140/',
                   'censusreporter://B17001/140/05000US06073/foobar'):

            with self.assertRaises(AppUrlError):
                u = parse_app_url(us)

        self.assertIsInstance(u, pr.CensusReporterURL)

        self.assertEqual(str(u), str(parse_app_url(str(u))))

        r = u.get_resource()

        self.assertTrue(r.exists())

        g = get_generator(r)

        rows, columns, release =  g.get_cr_rows()
        self.assertEqual(628, len(rows))
        self.assertEqual(120, len(columns))
        self.assertEqual('acs2015_5yr', release['id'])

        self.assertEqual(629, len(list(g)))

if __name__ == '__main__':
    unittest.main()
