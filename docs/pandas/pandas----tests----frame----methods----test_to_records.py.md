# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_to_records.py`

```
from collections import abc  # 导入abc模块中的集合类型抽象基类
import email  # 导入email模块，用于处理电子邮件相关操作
from email.parser import Parser  # 从email.parser模块导入Parser类，用于解析电子邮件

import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest模块，用于编写和执行测试用例

from pandas import (  # 从pandas库中导入多个子模块或函数
    CategoricalDtype,
    DataFrame,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm  # 导入pandas._testing模块，用于测试相关的辅助函数

class TestDataFrameToRecords:
    def test_to_records_timeseries(self):
        index = date_range("1/1/2000", periods=10)  # 创建一个包含日期索引的日期范围
        df = DataFrame(  # 创建一个DataFrame对象，包含随机数据
            np.random.default_rng(2).standard_normal((10, 3)),
            index=index,
            columns=["a", "b", "c"],
        )

        result = df.to_records()  # 将DataFrame转换为结构化数组
        assert result["index"].dtype == "M8[ns]"  # 断言索引的数据类型为日期时间类型

        result = df.to_records(index=False)  # 再次将DataFrame转换为结构化数组，但不包含索引

    def test_to_records_dt64(self):
        df = DataFrame(  # 创建一个DataFrame对象，包含字符串数据和日期索引
            [["one", "two", "three"], ["four", "five", "six"]],
            index=date_range("2012-01-01", "2012-01-02"),
        )

        expected = df.index.values[0]  # 获取预期的索引值
        result = df.to_records()["index"][0]  # 将DataFrame转换为结构化数组后，获取第一个索引值
        assert expected == result  # 断言预期的索引值与结果相等

    def test_to_records_dt64tz_column(self):
        # GH#32535 dont less tz in to_records
        df = DataFrame({"A": date_range("2012-01-01", "2012-01-02", tz="US/Eastern")})  # 创建带有时区的日期索引的DataFrame对象

        result = df.to_records()  # 将DataFrame转换为结构化数组

        assert result.dtype["A"] == object  # 断言结构化数组中"A"列的数据类型为对象
        val = result[0][1]  # 获取结构化数组中第一行第二列的值
        assert isinstance(val, Timestamp)  # 断言该值是Timestamp类型
        assert val == df.loc[0, "A"]  # 断言该值与DataFrame中相应位置的值相等

    def test_to_records_with_multindex(self):
        # GH#3189
        index = [  # 创建一个多级索引
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        data = np.zeros((8, 4))  # 创建一个8行4列的全零数组
        df = DataFrame(data, index=index)  # 创建一个包含多级索引的DataFrame对象
        r = df.to_records(index=True)["level_0"]  # 将DataFrame转换为结构化数组，并获取其中的"level_0"列
        assert "bar" in r  # 断言"bar"在结果中
        assert "one" not in r  # 断言"one"不在结果中

    def test_to_records_with_Mapping_type(self):
        abc.Mapping.register(email.message.Message)  # 将email.message.Message注册为abc.Mapping类型

        headers = Parser().parsestr(  # 解析邮件头部信息字符串，创建email.message.Message对象
            "From: <user@example.com>\n"
            "To: <someone_else@example.com>\n"
            "Subject: Test message\n"
            "\n"
            "Body would go here\n"
        )

        frame = DataFrame.from_records([headers])  # 从记录列表创建DataFrame对象
        all(x in frame for x in ["Type", "Subject", "From"])  # 断言DataFrame包含特定的列名

    def test_to_records_floats(self):
        df = DataFrame(np.random.default_rng(2).random((10, 10)))  # 创建一个包含随机浮点数的DataFrame对象
        df.to_records()  # 将DataFrame转换为结构化数组
    # 测试函数：测试将DataFrame转换为记录数组时是否正确处理索引名称
    def test_to_records_index_name(self):
        # 创建一个随机数据的DataFrame，形状为(3, 3)
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)))
        # 设置DataFrame的索引名称为"X"
        df.index.name = "X"
        # 将DataFrame转换为记录数组
        rs = df.to_records()
        # 断言：确保记录数组的数据类型中包含"X"字段
        assert "X" in rs.dtype.fields

        # 创建另一个随机数据的DataFrame，形状为(3, 3)
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)))
        # 将DataFrame转换为记录数组
        rs = df.to_records()
        # 断言：确保记录数组的数据类型中包含"index"字段
        assert "index" in rs.dtype.fields

        # 创建一个带有多级索引的DataFrame
        df.index = MultiIndex.from_tuples([("a", "x"), ("a", "y"), ("b", "z")])
        # 设置DataFrame的索引名称为["A", None]
        df.index.names = ["A", None]
        # 将DataFrame转换为记录数组
        result = df.to_records()
        # 期望的记录数组，使用numpy.rec.fromarrays创建
        expected = np.rec.fromarrays(
            [np.array(["a", "a", "b"]), np.array(["x", "y", "z"])]
            + [np.asarray(df.iloc[:, i]) for i in range(3)],
            dtype={
                "names": ["A", "level_1", "0", "1", "2"],
                "formats": [
                    "O",
                    "O",
                    f"{tm.ENDIAN}f8",
                    f"{tm.ENDIAN}f8",
                    f"{tm.ENDIAN}f8",
                ],
            },
        )
        # 断言：确保转换结果与期望的记录数组一致
        tm.assert_numpy_array_equal(result, expected)

    # 测试函数：测试在存在Unicode索引时是否正确处理转换为记录数组
    def test_to_records_with_unicode_index(self):
        # GH#13172：测试Unicode文字与to_records的冲突
        result = DataFrame([{"a": "x", "b": "y"}]).set_index("a").to_records()
        # 期望的记录数组，使用numpy.rec.array创建
        expected = np.rec.array([("x", "y")], dtype=[("a", "O"), ("b", "O")])
        # 断言：确保转换结果与期望的记录数组一致
        tm.assert_almost_equal(result, expected)

    # 测试函数：测试索引数据类型的一致性
    def test_to_records_index_dtype(self):
        # GH 47263: 确保索引和多级索引的数据类型一致性
        # 创建一个包含日期范围的DataFrame
        df = DataFrame(
            {
                1: date_range("2022-01-01", periods=2),
                2: date_range("2022-01-01", periods=2),
                3: date_range("2022-01-01", periods=2),
            }
        )

        # 期望的记录数组，使用numpy.rec.array创建，数据类型包含日期的精确格式
        expected = np.rec.array(
            [
                ("2022-01-01", "2022-01-01", "2022-01-01"),
                ("2022-01-02", "2022-01-02", "2022-01-02"),
            ],
            dtype=[
                ("1", f"{tm.ENDIAN}M8[ns]"),
                ("2", f"{tm.ENDIAN}M8[ns]"),
                ("3", f"{tm.ENDIAN}M8[ns]"),
            ],
        )

        # 将DataFrame转换为记录数组，不包括索引
        result = df.to_records(index=False)
        # 断言：确保转换结果与期望的记录数组一致
        tm.assert_almost_equal(result, expected)

        # 将DataFrame设置第一列为索引后转换为记录数组
        result = df.set_index(1).to_records(index=True)
        # 断言：确保转换结果与期望的记录数组一致
        tm.assert_almost_equal(result, expected)

        # 将DataFrame设置多级索引后转换为记录数组
        result = df.set_index([1, 2]).to_records(index=True)
        # 断言：确保转换结果与期望的记录数组一致
        tm.assert_almost_equal(result, expected)
    def test_to_records_with_unicode_column_names(self):
        # xref issue: https://github.com/numpy/numpy/issues/2407
        # Issue GH#11879. to_records used to raise an exception when used
        # with column names containing non-ascii characters in Python 2
        
        # 创建一个包含非ASCII字符列名的DataFrame，并使用to_records方法将其转换为记录数组
        result = DataFrame(data={"accented_name_é": [1.0]}).to_records()

        # 注意，numpy允许使用Unicode字段名，但需要使用字典而不是元组列表来指定dtype。
        expected = np.rec.array(
            [(0, 1.0)],
            dtype={"names": ["index", "accented_name_é"], "formats": ["=i8", "=f8"]},
        )
        # 使用tm.assert_almost_equal函数比较result和expected是否几乎相等
        tm.assert_almost_equal(result, expected)

    def test_to_records_with_categorical(self):
        # GH#8626
        
        # 创建一个包含分类数据类型的DataFrame，使用to_records方法将其转换为记录数组
        df = DataFrame({"A": list("abc")}, dtype="category")
        expected = Series(list("abc"), dtype="category", name="A")
        # 使用tm.assert_series_equal函数比较df["A"]和expected是否相等
        tm.assert_series_equal(df["A"], expected)

        # 创建一个包含分类数据类型的DataFrame（类似列表创建方式），使用to_records方法将其转换为记录数组
        df = DataFrame(list("abc"), dtype="category")
        expected = Series(list("abc"), dtype="category", name=0)
        # 使用tm.assert_series_equal函数比较df[0]和expected是否相等
        tm.assert_series_equal(df[0], expected)

        # 使用to_records方法将DataFrame转换为记录数组，并比较转换后的结果是否与预期的记录数组几乎相等
        result = df.to_records()
        expected = np.rec.array(
            [(0, "a"), (1, "b"), (2, "c")], dtype=[("index", "=i8"), ("0", "O")]
        )
        # 使用tm.assert_almost_equal函数比较result和expected是否几乎相等
        tm.assert_almost_equal(result, expected)

    def test_to_records_dtype(self, kwargs, expected):
        # see GH#18146
        
        # 创建一个包含不同数据类型的DataFrame
        df = DataFrame({"A": [1, 2], "B": [0.2, 1.5], "C": ["a", "bc"]})

        if not isinstance(expected, np.rec.recarray):
            # 如果预期结果不是np.rec.recarray类型，则预期会引发异常，使用pytest.raises捕获并检查是否匹配预期的异常类型和消息
            with pytest.raises(expected[0], match=expected[1]):
                df.to_records(**kwargs)
        else:
            # 使用to_records方法将DataFrame转换为记录数组，并比较转换后的结果是否几乎与预期的记录数组相等
            result = df.to_records(**kwargs)
            # 使用tm.assert_almost_equal函数比较result和expected是否几乎相等
            tm.assert_almost_equal(result, expected)
    @pytest.mark.parametrize(
        "df,kwargs,expected",
        [
            # MultiIndex in the index.
            (
                DataFrame(
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=list("abc")
                ).set_index(["a", "b"]),
                {"column_dtypes": "float64", "index_dtypes": {0: "int32", 1: "int8"}},
                np.rec.array(
                    [(1, 2, 3.0), (4, 5, 6.0), (7, 8, 9.0)],
                    dtype=[
                        ("a", f"{tm.ENDIAN}i4"),
                        ("b", "i1"),
                        ("c", f"{tm.ENDIAN}f8"),
                    ],
                ),
            ),
            # MultiIndex in the columns.
            (
                DataFrame(
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    columns=MultiIndex.from_tuples(
                        [("a", "d"), ("b", "e"), ("c", "f")]
                    ),
                ),
                {
                    "column_dtypes": {0: f"{tm.ENDIAN}U1", 2: "float32"},
                    "index_dtypes": "float32",
                },
                np.rec.array(
                    [(0.0, "1", 2, 3.0), (1.0, "4", 5, 6.0), (2.0, "7", 8, 9.0)],
                    dtype=[
                        ("index", f"{tm.ENDIAN}f4"),
                        ("('a', 'd')", f"{tm.ENDIAN}U1"),
                        ("('b', 'e')", f"{tm.ENDIAN}i8"),
                        ("('c', 'f')", f"{tm.ENDIAN}f4"),
                    ],
                ),
            ),
            # MultiIndex in both the columns and index.
            (
                DataFrame(
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    columns=MultiIndex.from_tuples(
                        [("a", "d"), ("b", "e"), ("c", "f")], names=list("ab")
                    ),
                    index=MultiIndex.from_tuples(
                        [("d", -4), ("d", -5), ("f", -6)], names=list("cd")
                    ),
                ),
                {
                    "column_dtypes": "float64",
                    "index_dtypes": {0: f"{tm.ENDIAN}U2", 1: "int8"},
                },
                np.rec.array(
                    [
                        ("d", -4, 1.0, 2.0, 3.0),
                        ("d", -5, 4.0, 5.0, 6.0),
                        ("f", -6, 7, 8, 9.0),
                    ],
                    dtype=[
                        ("c", f"{tm.ENDIAN}U2"),
                        ("d", "i1"),
                        ("('a', 'd')", f"{tm.ENDIAN}f8"),
                        ("('b', 'e')", f"{tm.ENDIAN}f8"),
                        ("('c', 'f')", f"{tm.ENDIAN}f8"),
                    ],
                ),
            ),
        ],
    )
    def test_to_records_dtype_mi(self, df, kwargs, expected):
        # 用于测试将 DataFrame 转换为记录数组的方法，针对具有多级索引的情况
        result = df.to_records(**kwargs)
        # 使用测试工具方法验证转换结果准确性
        tm.assert_almost_equal(result, expected)
    # 定义测试函数：将DataFrame转换为记录数组，并使用自定义的字典类模拟数据类型映射
    def test_to_records_dict_like(self):
        # 标注问题GH#18146，可能是GitHub上的问题编号
        class DictLike:
            # 构造函数，接受关键字参数并复制到内部字典
            def __init__(self, **kwargs) -> None:
                self.d = kwargs.copy()

            # 重载索引操作符[]，委托给内部字典
            def __getitem__(self, key):
                return self.d.__getitem__(key)

            # 检查键是否存在于内部字典中
            def __contains__(self, key) -> bool:
                return key in self.d

            # 返回内部字典的键集合
            def keys(self):
                return self.d.keys()

        # 创建测试用的DataFrame对象
        df = DataFrame({"A": [1, 2], "B": [0.2, 1.5], "C": ["a", "bc"]})

        # 定义数据类型映射字典
        dtype_mappings = {
            "column_dtypes": DictLike(A=np.int8, B=np.float32),
            "index_dtypes": f"{tm.ENDIAN}U2",
        }

        # 调用DataFrame的to_records方法，将DataFrame转换为记录数组，应用数据类型映射
        result = df.to_records(**dtype_mappings)

        # 期望的记录数组，定义了每列的名称和数据类型
        expected = np.rec.array(
            [("0", "1", "0.2", "a"), ("1", "2", "1.5", "bc")],
            dtype=[
                ("index", f"{tm.ENDIAN}U2"),
                ("A", "i1"),
                ("B", f"{tm.ENDIAN}f4"),
                ("C", "O"),
            ],
        )

        # 使用测试工具集中的assert_almost_equal方法验证结果与期望值的近似性
        tm.assert_almost_equal(result, expected)

    # 使用pytest的参数化标记，定义带时区的日期时间索引转换测试函数
    @pytest.mark.parametrize("tz", ["UTC", "GMT", "US/Eastern"])
    def test_to_records_datetimeindex_with_tz(self, tz):
        # 标注问题GH#13937
        # 创建一个日期时间范围，带有时区信息
        dr = date_range("2016-01-01", periods=10, freq="s", tz=tz)

        # 创建包含日期时间索引的DataFrame对象
        df = DataFrame({"datetime": dr}, index=dr)

        # 获取原始的记录数组
        expected = df.to_records()

        # 将DataFrame的时区转换为UTC，并获取记录数组
        result = df.tz_convert("UTC").to_records()

        # 使用测试工具集中的assert_numpy_array_equal方法验证结果与期望值的完全相等性
        # 因为两者都转换为UTC时区，所以它们应该相等
        tm.assert_numpy_array_equal(result, expected)
```