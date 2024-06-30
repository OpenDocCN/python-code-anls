# `D:\src\scipysrc\scipy\scipy\io\arff\tests\test_arffread.py`

```
import datetime  # 导入处理日期和时间的模块
import os  # 导入操作系统功能的模块
import sys  # 导入系统相关的模块
from os.path import join as pjoin  # 导入 os.path 模块中的 join 函数，并重命名为 pjoin

from io import StringIO  # 导入用于内存中读写的 StringIO 类

import numpy as np  # 导入数值计算库 numpy

from numpy.testing import (assert_array_almost_equal,  # 从 numpy.testing 导入断言函数
                           assert_array_equal, assert_equal, assert_)
from pytest import raises as assert_raises  # 导入 pytest 中的 raises 函数，并重命名为 assert_raises

from scipy.io.arff import loadarff  # 导入 scipy 库中处理 ARFF 文件的函数
from scipy.io.arff._arffread import read_header, ParseArffError  # 导入 ARFF 文件读取相关函数

data_path = pjoin(os.path.dirname(__file__), 'data')  # 设置数据文件夹的路径为当前脚本文件夹下的 data 文件夹路径

test1 = pjoin(data_path, 'test1.arff')  # 构建测试文件路径
test2 = pjoin(data_path, 'test2.arff')
test3 = pjoin(data_path, 'test3.arff')

test4 = pjoin(data_path, 'test4.arff')
test5 = pjoin(data_path, 'test5.arff')
test6 = pjoin(data_path, 'test6.arff')
test7 = pjoin(data_path, 'test7.arff')
test8 = pjoin(data_path, 'test8.arff')
test9 = pjoin(data_path, 'test9.arff')
test10 = pjoin(data_path, 'test10.arff')
test11 = pjoin(data_path, 'test11.arff')
test_quoted_nominal = pjoin(data_path, 'quoted_nominal.arff')
test_quoted_nominal_spaces = pjoin(data_path, 'quoted_nominal_spaces.arff')

expect4_data = [(0.1, 0.2, 0.3, 0.4, 'class1'),  # 预期的测试数据
                (-0.1, -0.2, -0.3, -0.4, 'class2'),
                (1, 2, 3, 4, 'class3')]
expected_types = ['numeric', 'numeric', 'numeric', 'numeric', 'nominal']  # 预期的数据类型

missing = pjoin(data_path, 'missing.arff')  # 缺失数据文件路径
expect_missing_raw = np.array([[1, 5], [2, 4], [np.nan, np.nan]])  # 预期的原始缺失数据
expect_missing = np.empty(3, [('yop', float), ('yap', float)])  # 预期的缺失数据结构

expect_missing['yop'] = expect_missing_raw[:, 0]  # 填充预期的缺失数据
expect_missing['yap'] = expect_missing_raw[:, 1]


class TestData:
    def test1(self):
        # Parsing trivial file with nothing.
        self._test(test4)  # 测试加载 test4.arff 文件

    def test2(self):
        # Parsing trivial file with some comments in the data section.
        self._test(test5)  # 测试加载 test5.arff 文件

    def test3(self):
        # Parsing trivial file with nominal attribute of 1 character.
        self._test(test6)  # 测试加载 test6.arff 文件

    def test4(self):
        # Parsing trivial file with trailing spaces in attribute declaration.
        self._test(test11)  # 测试加载 test11.arff 文件

    def _test(self, test_file):
        data, meta = loadarff(test_file)  # 调用 loadarff 函数加载 ARFF 文件数据
        for i in range(len(data)):  # 遍历数据
            for j in range(4):
                assert_array_almost_equal(expect4_data[i][j], data[i][j])  # 断言数据准确性
        assert_equal(meta.types(), expected_types)  # 断言元数据的类型与预期一致

    def test_filelike(self):
        # Test reading from file-like object (StringIO)
        with open(test1) as f1:
            data1, meta1 = loadarff(f1)  # 从文件对象中读取 ARFF 数据
        with open(test1) as f2:
            data2, meta2 = loadarff(StringIO(f2.read()))  # 使用 StringIO 读取 ARFF 数据
        assert_(data1 == data2)  # 断言两种读取方式得到的数据一致
        assert_(repr(meta1) == repr(meta2))  # 断言两种读取方式得到的元数据一致

    def test_path(self):
        # Test reading from `pathlib.Path` object
        from pathlib import Path

        with open(test1) as f1:
            data1, meta1 = loadarff(f1)  # 从文件对象中读取 ARFF 数据

        data2, meta2 = loadarff(Path(test1))  # 使用 pathlib.Path 对象读取 ARFF 数据

        assert_(data1 == data2)  # 断言两种读取方式得到的数据一致
        assert_(repr(meta1) == repr(meta2))  # 断言两种读取方式得到的元数据一致


class TestMissingData:
    # 定义一个名为 test_missing 的测试方法，该方法属于某个类，self 参数表示当前实例
    def test_missing(self):
        # 调用 loadarff 函数加载数据和元数据，其中 missing 变量是一个参数，可能是文件路径或数据源
        data, meta = loadarff(missing)
        # 遍历列表 ['yop', 'yap'] 中的每个元素，依次赋值给变量 i
        for i in ['yop', 'yap']:
            # 断言 data[i] 与 expect_missing[i] 的数组近似相等
            assert_array_almost_equal(data[i], expect_missing[i])
class TestNoData:
    def test_nodata(self):
        # The file nodata.arff has no data in the @DATA section.
        # Reading it should result in an array with length 0.
        # 构建文件路径：数据路径下的 nodata.arff 文件
        nodata_filename = os.path.join(data_path, 'nodata.arff')
        # 载入 ARFF 文件的数据和元数据
        data, meta = loadarff(nodata_filename)
        # 根据系统的字节顺序确定数据结尾的符号
        if sys.byteorder == 'big':
            end = '>'
        else:
            end = '<'
        # 期望的数据类型，使用 NumPy 的 dtype 定义
        expected_dtype = np.dtype([('sepallength', f'{end}f8'),
                                   ('sepalwidth', f'{end}f8'),
                                   ('petallength', f'{end}f8'),
                                   ('petalwidth', f'{end}f8'),
                                   ('class', 'S15')])
        # 断言数据的 dtype 符合期望的 dtype
        assert_equal(data.dtype, expected_dtype)
        # 断言数据的大小为 0，因为 nodata.arff 文件中没有数据
        assert_equal(data.size, 0)


class TestHeader:
    def test_type_parsing(self):
        # Test parsing type of attribute from their value.
        # 打开 test2 文件并读取其头部信息
        with open(test2) as ofile:
            rel, attrs = read_header(ofile)

        # 预期的属性类型列表
        expected = ['numeric', 'numeric', 'numeric', 'numeric', 'numeric',
                    'numeric', 'string', 'string', 'nominal', 'nominal']

        # 对每个属性进行断言，验证其类型是否符合预期
        for i in range(len(attrs)):
            assert_(attrs[i].type_name == expected[i])

    def test_badtype_parsing(self):
        # Test parsing wrong type of attribute from their value.
        # 定义一个函数用来读取 test3 文件，并期望抛出 ParseArffError 异常
        def badtype_read():
            with open(test3) as ofile:
                _, _ = read_header(ofile)

        # 断言调用 badtype_read 函数会抛出 ParseArffError 异常
        assert_raises(ParseArffError, badtype_read)

    def test_fullheader1(self):
        # Parsing trivial header with nothing.
        # 打开 test1 文件并读取其头部信息
        with open(test1) as ofile:
            rel, attrs = read_header(ofile)

        # 测试关系名称是否为 'test1'
        assert_(rel == 'test1')

        # 测试属性数量是否为 5
        assert_(len(attrs) == 5)
        # 对前四个数值属性进行断言：名称和类型是否符合预期
        for i in range(4):
            assert_(attrs[i].name == 'attr%d' % i)
            assert_(attrs[i].type_name == 'numeric')

        # 测试名义属性是否符合预期
        assert_(attrs[4].name == 'class')
        assert_(attrs[4].values == ('class0', 'class1', 'class2', 'class3'))

    def test_dateheader(self):
        # 打开 test7 文件并读取其头部信息
        with open(test7) as ofile:
            rel, attrs = read_header(ofile)

        # 测试关系名称是否为 'test7'
        assert_(rel == 'test7')

        # 测试属性数量是否为 5
        assert_(len(attrs) == 5)

        # 对每个日期属性进行断言：名称和日期格式是否符合预期
        assert_(attrs[0].name == 'attr_year')
        assert_(attrs[0].date_format == '%Y')

        assert_(attrs[1].name == 'attr_month')
        assert_(attrs[1].date_format == '%Y-%m')

        assert_(attrs[2].name == 'attr_date')
        assert_(attrs[2].date_format == '%Y-%m-%d')

        assert_(attrs[3].name == 'attr_datetime_local')
        assert_(attrs[3].date_format == '%Y-%m-%d %H:%M')

        assert_(attrs[4].name == 'attr_datetime_missing')
        assert_(attrs[4].date_format == '%Y-%m-%d %H:%M')

    def test_dateheader_unsupported(self):
        # 定义一个函数用来读取 test8 文件，并期望抛出 ValueError 异常
        def read_dateheader_unsupported():
            with open(test8) as ofile:
                _, _ = read_header(ofile)

        # 断言调用 read_dateheader_unsupported 函数会抛出 ValueError 异常
        assert_raises(ValueError, read_dateheader_unsupported)
    # 在测试运行前设置测试环境的方法
    def setup_method(self):
        # 调用loadarff函数加载测试文件test7，将返回的数据和元数据分别赋值给self.data和self.meta
        self.data, self.meta = loadarff(test7)

    # 测试attr_year属性是否符合预期
    def test_year_attribute(self):
        # 期望的数据，包括多个年份的日期，以datetime64类型存储
        expected = np.array([
            '1999',
            '2004',
            '1817',
            '2100',
            '2013',
            '1631'
        ], dtype='datetime64[Y]')
        # 断言self.data["attr_year"]与expected数组相等
        assert_array_equal(self.data["attr_year"], expected)

    # 测试attr_month属性是否符合预期
    def test_month_attribute(self):
        # 期望的数据，包括多个年份和月份的日期，以datetime64类型存储
        expected = np.array([
            '1999-01',
            '2004-12',
            '1817-04',
            '2100-09',
            '2013-11',
            '1631-10'
        ], dtype='datetime64[M]')
        # 断言self.data["attr_month"]与expected数组相等
        assert_array_equal(self.data["attr_month"], expected)

    # 测试attr_date属性是否符合预期
    def test_date_attribute(self):
        # 期望的数据，包括多个年份、月份和日期的完整日期，以datetime64类型存储
        expected = np.array([
            '1999-01-31',
            '2004-12-01',
            '1817-04-28',
            '2100-09-10',
            '2013-11-30',
            '1631-10-15'
        ], dtype='datetime64[D]')
        # 断言self.data["attr_date"]与expected数组相等
        assert_array_equal(self.data["attr_date"], expected)

    # 测试attr_datetime_local属性是否符合预期
    def test_datetime_local_attribute(self):
        # 期望的数据，包括多个年份、月份、日期、小时和分钟的完整日期时间，以datetime64类型存储
        expected = np.array([
            datetime.datetime(year=1999, month=1, day=31, hour=0, minute=1),
            datetime.datetime(year=2004, month=12, day=1, hour=23, minute=59),
            datetime.datetime(year=1817, month=4, day=28, hour=13, minute=0),
            datetime.datetime(year=2100, month=9, day=10, hour=12, minute=0),
            datetime.datetime(year=2013, month=11, day=30, hour=4, minute=55),
            datetime.datetime(year=1631, month=10, day=15, hour=20, minute=4)
        ], dtype='datetime64[m]')
        # 断言self.data["attr_datetime_local"]与expected数组相等
        assert_array_equal(self.data["attr_datetime_local"], expected)

    # 测试attr_datetime_missing属性是否符合预期
    def test_datetime_missing(self):
        # 期望的数据，包括"nat"（not a time）和部分具体日期时间的字符串，以datetime64类型存储
        expected = np.array([
            'nat',
            '2004-12-01T23:59',
            'nat',
            'nat',
            '2013-11-30T04:55',
            '1631-10-15T20:04'
        ], dtype='datetime64[m]')
        # 断言self.data["attr_datetime_missing"]与expected数组相等
        assert_array_equal(self.data["attr_datetime_missing"], expected)

    # 测试加载包含时区信息的数据时是否引发ParseArffError异常
    def test_datetime_timezone(self):
        # 断言调用loadarff函数加载test8时，会引发ParseArffError异常
        assert_raises(ParseArffError, loadarff, test8)
class TestRelationalAttribute:
    # 测试关系属性类

    def setup_method(self):
        # 设置测试方法
        self.data, self.meta = loadarff(test9)
        # 载入测试数据和元数据

    def test_attributes(self):
        # 测试属性
        assert_equal(len(self.meta._attributes), 1)
        # 断言元数据中属性数量为1

        relational = list(self.meta._attributes.values())[0]
        # 获取关系属性

        assert_equal(relational.name, 'attr_date_number')
        # 断言关系属性名称为'attr_date_number'
        assert_equal(relational.type_name, 'relational')
        # 断言关系属性类型为'relational'
        assert_equal(len(relational.attributes), 2)
        # 断言关系属性包含的子属性数量为2
        assert_equal(relational.attributes[0].name,
                     'attr_date')
        # 断言第一个子属性名称为'attr_date'
        assert_equal(relational.attributes[0].type_name,
                     'date')
        # 断言第一个子属性类型为'date'
        assert_equal(relational.attributes[1].name,
                     'attr_number')
        # 断言第二个子属性名称为'attr_number'
        assert_equal(relational.attributes[1].type_name,
                     'numeric')
        # 断言第二个子属性类型为'numeric'

    def test_data(self):
        # 测试数据
        dtype_instance = [('attr_date', 'datetime64[D]'),
                          ('attr_number', np.float64)]
        # 定义数据类型实例，包括日期和数字类型

        expected = [
            np.array([('1999-01-31', 1), ('1935-11-27', 10)],
                     dtype=dtype_instance),
            np.array([('2004-12-01', 2), ('1942-08-13', 20)],
                     dtype=dtype_instance),
            np.array([('1817-04-28', 3)],
                     dtype=dtype_instance),
            np.array([('2100-09-10', 4), ('1957-04-17', 40),
                      ('1721-01-14', 400)],
                     dtype=dtype_instance),
            np.array([('2013-11-30', 5)],
                     dtype=dtype_instance),
            np.array([('1631-10-15', 6)],
                     dtype=dtype_instance)
        ]
        # 定义预期的数据数组列表

        for i in range(len(self.data["attr_date_number"])):
            # 遍历数据中的'attr_date_number'键对应的所有元素
            assert_array_equal(self.data["attr_date_number"][i],
                               expected[i])
            # 断言每个元素与预期数组中的对应元素相等


class TestRelationalAttributeLong:
    # 测试长关系属性类

    def setup_method(self):
        # 设置测试方法
        self.data, self.meta = loadarff(test10)
        # 载入测试数据和元数据

    def test_attributes(self):
        # 测试属性
        assert_equal(len(self.meta._attributes), 1)
        # 断言元数据中属性数量为1

        relational = list(self.meta._attributes.values())[0]
        # 获取关系属性

        assert_equal(relational.name, 'attr_relational')
        # 断言关系属性名称为'attr_relational'
        assert_equal(relational.type_name, 'relational')
        # 断言关系属性类型为'relational'
        assert_equal(len(relational.attributes), 1)
        # 断言关系属性包含的子属性数量为1
        assert_equal(relational.attributes[0].name,
                     'attr_number')
        # 断言第一个子属性名称为'attr_number'
        assert_equal(relational.attributes[0].type_name, 'numeric')
        # 断言第一个子属性类型为'numeric'

    def test_data(self):
        # 测试数据
        dtype_instance = [('attr_number', np.float64)]
        # 定义数据类型实例，包括数字类型

        expected = np.array([(n,) for n in range(30000)],
                            dtype=dtype_instance)
        # 定义预期的数据数组

        assert_array_equal(self.data["attr_relational"][0],
                           expected)
        # 断言数据中'attr_relational'键对应的第一个元素与预期数组相等


class TestQuotedNominal:
    """
    Regression test for issue #10232:
    
    Exception in loadarff with quoted nominal attributes.
    """
    # 引用名义属性的回归测试

    def setup_method(self):
        # 设置测试方法
        self.data, self.meta = loadarff(test_quoted_nominal)
        # 载入测试数据和元数据
    # 测试属性的断言，确保 self.meta._attributes 包含两个元素
    assert_equal(len(self.meta._attributes), 2)

    # 将 self.meta._attributes 的值解包到 age 和 smoker 变量中
    age, smoker = self.meta._attributes.values()

    # 断言 age 对象的属性 name 为 'age'，type_name 为 'numeric'
    assert_equal(age.name, 'age')
    assert_equal(age.type_name, 'numeric')

    # 断言 smoker 对象的属性 name 为 'smoker'，type_name 为 'nominal'，values 为 ['yes', 'no']
    assert_equal(smoker.name, 'smoker')
    assert_equal(smoker.type_name, 'nominal')
    assert_equal(smoker.values, ['yes', 'no'])

    # 设置预期的 age 和 smoker 数据类型实例
    age_dtype_instance = np.float64
    smoker_dtype_instance = '<S3'

    # 设置预期的 age 和 smoker 数据值
    age_expected = np.array([
        18,
        24,
        44,
        56,
        89,
        11,
    ], dtype=age_dtype_instance)

    smoker_expected = np.array([
        'no',
        'yes',
        'no',
        'no',
        'yes',
        'no',
    ], dtype=smoker_dtype_instance)

    # 断言 self.data 字典中 "age" 键对应的值与预期的 age_expected 数组相等
    assert_array_equal(self.data["age"], age_expected)

    # 断言 self.data 字典中 "smoker" 键对应的值与预期的 smoker_expected 数组相等
    assert_array_equal(self.data["smoker"], smoker_expected)
    # 定义一个名为 TestQuotedNominalSpaces 的测试类，用于回归测试问题 #10232
    """
    Regression test for issue #10232:
    
    Exception in loadarff with quoted nominal attributes.
    """

    # 设置每个测试方法的前置条件，加载指定的 ARFF 数据文件并将数据与元数据赋值给实例变量
    def setup_method(self):
        self.data, self.meta = loadarff(test_quoted_nominal_spaces)

    # 测试属性的方法，验证元数据中属性的数量和具体内容
    def test_attributes(self):
        assert_equal(len(self.meta._attributes), 2)

        # 解构元数据中的两个属性对象，分别验证其名称和类型
        age, smoker = self.meta._attributes.values()

        assert_equal(age.name, 'age')
        assert_equal(age.type_name, 'numeric')
        assert_equal(smoker.name, 'smoker')
        assert_equal(smoker.type_name, 'nominal')
        assert_equal(smoker.values, ['  yes', 'no  '])

    # 测试数据的方法，验证从加载的数据中提取的特定字段的值是否与预期的值数组相匹配
    def test_data(self):

        # 定义预期的数据类型实例
        age_dtype_instance = np.float64
        smoker_dtype_instance = '<S5'

        # 定义预期的年龄数据和吸烟者数据
        age_expected = np.array([
            18,
            24,
            44,
            56,
            89,
            11,
        ], dtype=age_dtype_instance)

        smoker_expected = np.array([
            'no  ',
            '  yes',
            'no  ',
            'no  ',
            '  yes',
            'no  ',
        ], dtype=smoker_dtype_instance)

        # 断言加载的数据中特定字段的实际值与预期值数组相等
        assert_array_equal(self.data["age"], age_expected)
        assert_array_equal(self.data["smoker"], smoker_expected)
```