import unittest
import pandasreporter as pr

class BasicTests(unittest.TestCase):

    def test_something(self):

        df = pr.get_dataframe('B01001', '140',  '05000US06073')

        self.assertEquals(37292.0, df.lookup(84).sum())
        self.assertEquals(df.B01001042.sum(), df.lookup('042').sum())

        self.assertEquals('B01001042', df.lookup('042').census_code)
        self.assertEquals('Total Female 60 and 61 years', df.lookup('042').census_title)

        self.assertEquals(2806.0, df.lookup('042').m90.sum())
        self.assertEquals(37292.0, df.lookup('042').m90.value.sum())


if __name__ == '__main__':
    unittest.main()
