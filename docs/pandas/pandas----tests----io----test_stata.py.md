# `D:\src\scipysrc\pandas\pandas\tests\io\test_stata.py`

```
import bz2  # 导入bz2模块，用于处理bzip2压缩格式
import datetime as dt  # 导入datetime模块并重命名为dt，用于处理日期时间
from datetime import datetime  # 导入datetime模块中的datetime类，用于处理日期时间
import gzip  # 导入gzip模块，用于处理gzip压缩格式
import io  # 导入io模块，提供了核心的IO功能
import os  # 导入os模块，提供了与操作系统交互的功能
import struct  # 导入struct模块，用于处理字节数据和二进制数据的转换
import tarfile  # 导入tarfile模块，用于处理tar压缩文件格式
import zipfile  # 导入zipfile模块，用于处理zip压缩文件格式

import numpy as np  # 导入numpy库并重命名为np，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试用例

import pandas.util._test_decorators as td  # 导入pandas库的测试装饰器

import pandas as pd  # 导入pandas库并重命名为pd，用于数据分析和处理
from pandas import CategoricalDtype  # 从pandas中导入CategoricalDtype类，用于处理分类数据类型
import pandas._testing as tm  # 导入pandas库的测试模块
from pandas.core.frame import (  # 从pandas的核心框架中导入DataFrame和Series类
    DataFrame,
    Series,
)

from pandas.io.parsers import read_csv  # 从pandas的IO解析模块中导入read_csv函数，用于读取CSV文件
from pandas.io.stata import (  # 从pandas的IO Stata模块中导入多个类和函数
    CategoricalConversionWarning,
    InvalidColumnName,
    PossiblePrecisionLoss,
    StataMissingValue,
    StataReader,
    StataWriter,
    StataWriterUTF8,
    ValueLabelTypeMismatch,
    read_stata,
)


@pytest.fixture
def mixed_frame():
    return DataFrame(  # 返回一个DataFrame对象，包含三列数据：a、b、c
        {
            "a": [1, 2, 3, 4],
            "b": [1.0, 3.0, 27.0, 81.0],
            "c": ["Atlanta", "Birmingham", "Cincinnati", "Detroit"],
        }
    )


@pytest.fixture
def parsed_114(datapath):
    dta14_114 = datapath("io", "data", "stata", "stata5_114.dta")  # 拼接数据路径，指定Stata数据文件
    parsed_114 = read_stata(dta14_114, convert_dates=True)  # 使用read_stata函数读取Stata数据文件，并转换日期
    parsed_114.index.name = "index"  # 设置数据的索引名称为'index'
    return parsed_114  # 返回读取并处理后的数据


class TestStata:
    def read_dta(self, file):
        # Legacy default reader configuration
        return read_stata(file, convert_dates=True)  # 使用read_stata函数读取Stata数据文件，并转换日期

    def read_csv(self, file):
        return read_csv(file, parse_dates=True)  # 使用read_csv函数读取CSV文件，并解析日期时间

    @pytest.mark.parametrize("version", [114, 117, 118, 119, None])
    def test_read_empty_dta(self, version, temp_file):
        empty_ds = DataFrame(columns=["unit"])  # 创建一个空的DataFrame对象，包含一列名为'unit'
        # GH 7369, make sure can read a 0-obs dta file
        path = temp_file  # 将temp_file路径赋值给变量path
        empty_ds.to_stata(path, write_index=False, version=version)  # 将空的DataFrame对象写入Stata文件
        empty_ds2 = read_stata(path)  # 使用read_stata函数读取刚写入的Stata文件
        tm.assert_frame_equal(empty_ds, empty_ds2)  # 使用tm.assert_frame_equal断言确保两个DataFrame相等

    @pytest.mark.parametrize("version", [114, 117, 118, 119, None])
    # 测试读取带有指定数据类型的空数据框
    def test_read_empty_dta_with_dtypes(self, version, temp_file):
        # GH 46240
        # 修复上述错误揭示了当写入空数据框时类型未正确保留的问题
        empty_df_typed = DataFrame(
            {
                "i8": np.array([0], dtype=np.int8),
                "i16": np.array([0], dtype=np.int16),
                "i32": np.array([0], dtype=np.int32),
                "i64": np.array([0], dtype=np.int64),
                "u8": np.array([0], dtype=np.uint8),
                "u16": np.array([0], dtype=np.uint16),
                "u32": np.array([0], dtype=np.uint32),
                "u64": np.array([0], dtype=np.uint64),
                "f32": np.array([0], dtype=np.float32),
                "f64": np.array([0], dtype=np.float64),
            }
        )
        # GH 7369, 确保能够读取包含0行观测的 dta 文件
        path = temp_file
        empty_df_typed.to_stata(path, write_index=False, version=version)
        empty_reread = read_stata(path)

        expected = empty_df_typed
        # 无 uint# 支持。由于值在 int# 的范围内，进行类型降级
        expected["u8"] = expected["u8"].astype(np.int8)
        expected["u16"] = expected["u16"].astype(np.int16)
        expected["u32"] = expected["u32"].astype(np.int32)
        # 不支持 int64。由于值在 int32 的范围内，进行类型降级
        expected["u64"] = expected["u64"].astype(np.int32)
        expected["i64"] = expected["i64"].astype(np.int32)

        # 断言重新读取的数据框与预期的数据框相等
        tm.assert_frame_equal(expected, empty_reread)
        # 断言预期的数据类型与重新读取的数据框的数据类型相等
        tm.assert_series_equal(expected.dtypes, empty_reread.dtypes)

    @pytest.mark.parametrize("version", [114, 117, 118, 119, None])
    # 测试读取索引列为 None 的情况
    def test_read_index_col_none(self, version, temp_file):
        df = DataFrame({"a": range(5), "b": ["b1", "b2", "b3", "b4", "b5"]})
        # GH 7369, 确保能够读取包含0行观测的 dta 文件
        path = temp_file
        df.to_stata(path, write_index=False, version=version)
        read_df = read_stata(path)

        # 断言读取的数据框索引是 pd.RangeIndex 类型
        assert isinstance(read_df.index, pd.RangeIndex)
        expected = df
        expected["a"] = expected["a"].astype(np.int32)
        # 断言重新读取的数据框与预期的数据框相等，检查索引类型
        tm.assert_frame_equal(read_df, expected, check_index_type=True)

    @pytest.mark.parametrize("file", ["stata1_114", "stata1_117"])
    # 测试读取特定的 dta 文件
    def test_read_dta1(self, file, datapath):
        file = datapath("io", "data", "stata", f"{file}.dta")
        parsed = self.read_dta(file)

        # Pandas 使用 np.nan 作为缺失值。
        # 因此，所有列的类型都将是 float，无论它们的名称是什么。
        expected = DataFrame(
            [(np.nan, np.nan, np.nan, np.nan, np.nan)],
            columns=["float_miss", "double_miss", "byte_miss", "int_miss", "long_miss"],
        )

        # 这是一个奇怪的情况，实际上 np.nan 应该是 float64，但是
        # 转换不会失败，因此需要与 stata 匹配
        expected["float_miss"] = expected["float_miss"].astype(np.float32)

        # 断言解析的数据框与预期的数据框相等
        tm.assert_frame_equal(parsed, expected)
    # 使用 pytest 的 parametrize 装饰器为该测试方法提供多组参数化输入
    @pytest.mark.parametrize(
        "file", ["stata3_113", "stata3_114", "stata3_115", "stata3_117"]
    )
    # 定义测试方法 test_read_dta3，接受 file 和 datapath 两个参数
    def test_read_dta3(self, file, datapath):
        # 构建完整的文件路径
        file = datapath("io", "data", "stata", f"{file}.dta")
        # 调用 self.read_dta 方法解析数据文件
        parsed = self.read_dta(file)

        # 准备预期的数据集，从 CSV 文件中读取，并将数据类型转换为 np.float32
        expected = self.read_csv(datapath("io", "data", "stata", "stata3.csv"))
        expected = expected.astype(np.float32)
        # 将 "year" 和 "quarter" 列的数据类型转换为 np.int16 和 np.int8
        expected["year"] = expected["year"].astype(np.int16)
        expected["quarter"] = expected["quarter"].astype(np.int8)

        # 使用 pandas.testing 中的 assert_frame_equal 方法比较 parsed 和 expected 数据框
        tm.assert_frame_equal(parsed, expected)

    # 使用 pytest 的 parametrize 装饰器为该测试方法提供多组参数化输入
    @pytest.mark.parametrize("version", [110, 111, 113, 114, 115, 117])
    # 定义测试方法 test_read_dta4，接受 version 和 datapath 两个参数
    def test_read_dta4(self, version, datapath):
        # 构建完整的文件路径
        file = datapath("io", "data", "stata", f"stata4_{version}.dta")
        # 调用 self.read_dta 方法解析数据文件
        parsed = self.read_dta(file)

        # 准备预期的数据集，使用 DataFrame.from_records 创建数据框
        expected = DataFrame.from_records(
            [
                ["one", "ten", "one", "one", "one"],
                ["two", "nine", "two", "two", "two"],
                ["three", "eight", "three", "three", "three"],
                ["four", "seven", 4, "four", "four"],
                ["five", "six", 5, np.nan, "five"],
                ["six", "five", 6, np.nan, "six"],
                ["seven", "four", 7, np.nan, "seven"],
                ["eight", "three", 8, np.nan, "eight"],
                ["nine", "two", 9, np.nan, "nine"],
                ["ten", "one", "ten", np.nan, "ten"],
            ],
            columns=[
                "fully_labeled",
                "fully_labeled2",
                "incompletely_labeled",
                "labeled_with_missings",
                "float_labelled",
            ],
        )

        # 对预期数据集中的所有列进行处理，将其转换为分类数据类型
        for col in expected:
            orig = expected[col].copy()

            # 如果列名为 "incompletely_labeled"，则使用原始数据
            categories = np.asarray(expected["fully_labeled"][orig.notna()])
            if col == "incompletely_labeled":
                categories = orig

            # 将列转换为有序的分类数据类型，并更新 expected 数据框中的该列
            cat = orig.astype("category")._values
            cat = cat.set_categories(categories, ordered=True)
            cat.categories.rename(None, inplace=True)

            expected[col] = cat

        # 使用 pandas.testing 中的 assert_frame_equal 方法比较 parsed 和 expected 数据框
        # 注意：Stata 不保存 .category 元数据
        tm.assert_frame_equal(parsed, expected)
    # 使用版本号和数据路径作为参数，测试读取旧版本的 STATA 数据文件
    def test_readold_dta4(self, version, datapath):
        # 这个测试与上面的 test_read_dta4 相同，不过列名必须重命名以匹配旧文件格式的限制
        file = datapath("io", "data", "stata", f"stata4_{version}.dta")
        # 调用 read_dta 方法解析数据文件
        parsed = self.read_dta(file)

        # 预期的 DataFrame 包含以下记录和列名
        expected = DataFrame.from_records(
            [
                ["one", "ten", "one", "one", "one"],
                ["two", "nine", "two", "two", "two"],
                ["three", "eight", "three", "three", "three"],
                ["four", "seven", 4, "four", "four"],
                ["five", "six", 5, np.nan, "five"],
                ["six", "five", 6, np.nan, "six"],
                ["seven", "four", 7, np.nan, "seven"],
                ["eight", "three", 8, np.nan, "eight"],
                ["nine", "two", 9, np.nan, "nine"],
                ["ten", "one", "ten", np.nan, "ten"],
            ],
            columns=[
                "fulllab",
                "fulllab2",
                "incmplab",
                "misslab",
                "floatlab",
            ],
        )

        # 这些列都是分类数据
        for col in expected:
            orig = expected[col].copy()

            # 从 fulllab 列中获取分类的值
            categories = np.asarray(expected["fulllab"][orig.notna()])
            if col == "incmplab":
                categories = orig

            # 将列转换为分类类型，并设置分类的值和顺序
            cat = orig.astype("category")._values
            cat = cat.set_categories(categories, ordered=True)
            cat.categories.rename(None, inplace=True)

            expected[col] = cat

        # STATA 不保存 .category 元数据，因此这里不检查 dtype
        tm.assert_frame_equal(parsed, expected)

    # 包含字符串标签的数据文件
    @pytest.mark.parametrize(
        "file",
        [
            "stata12_117",
            "stata12_be_117",
            "stata12_118",
            "stata12_be_118",
            "stata12_119",
            "stata12_be_119",
        ],
    )
    def test_read_dta_strl(self, file, datapath):
        # 解析包含字符串标签的 STATA 数据文件
        parsed = self.read_dta(datapath("io", "data", "stata", f"{file}.dta"))
        # 预期的 DataFrame 包含以下记录和列名
        expected = DataFrame.from_records(
            [
                [1, "abc", "abcdefghi"],
                [3, "cba", "qwertywertyqwerty"],
                [93, "", "strl"],
            ],
            columns=["x", "y", "z"],
        )

        # 不检查 dtype，直接比较解析结果和预期结果的 DataFrame
        tm.assert_frame_equal(parsed, expected, check_dtype=False)

    # 117 不包含在列表中，因为它使用 ASCII 字符串
    @pytest.mark.parametrize(
        "file",
        [
            "stata14_118",
            "stata14_be_118",
            "stata14_119",
            "stata14_be_119",
        ],
    )
    # 定义测试函数，用于测试读取特定数据文件并验证其内容
    def test_read_dta118_119(self, file, datapath):
        # 调用read_dta方法读取特定路径下的Stata数据文件并解析
        parsed_118 = self.read_dta(datapath("io", "data", "stata", f"{file}.dta"))
        
        # 将解析后的数据中的"Bytes"列转换为Python对象类型（object）
        parsed_118["Bytes"] = parsed_118["Bytes"].astype("O")
        
        # 创建预期结果的DataFrame，包含各种数据类型和特定值
        expected = DataFrame.from_records(
            [
                ["Cat", "Bogota", "Bogotá", 1, 1.0, "option b Ünicode", 1.0],
                ["Dog", "Boston", "Uzunköprü", np.nan, np.nan, np.nan, np.nan],
                ["Plane", "Rome", "Tromsø", 0, 0.0, "option a", 0.0],
                ["Potato", "Tokyo", "Elâzığ", -4, 4.0, 4, 4],  # noqa: RUF001
                ["", "", "", 0, 0.3332999, "option a", 1 / 3.0],
            ],
            columns=[
                "Things",
                "Cities",
                "Unicode_Cities_Strl",
                "Ints",
                "Floats",
                "Bytes",
                "Longs",
            ],
        )
        
        # 将预期结果中的"Floats"列转换为np.float32类型
        expected["Floats"] = expected["Floats"].astype(np.float32)
        
        # 遍历解析后的数据的每列，使用测试工具库中的方法验证与预期结果的近似相等性
        for col in parsed_118.columns:
            tm.assert_almost_equal(parsed_118[col], expected[col])

        # 使用StataReader打开数据文件，并获取变量标签信息
        with StataReader(datapath("io", "data", "stata", f"{file}.dta")) as rdr:
            # 获取实际的变量标签字典
            vl = rdr.variable_labels()
            
            # 预期的变量标签字典
            vl_expected = {
                "Unicode_Cities_Strl": "Here are some strls with Ünicode chars",
                "Longs": "long data",
                "Things": "Here are some things",
                "Bytes": "byte data",
                "Ints": "int data",
                "Cities": "Here are some cities",
                "Floats": "float data",
            }
            
            # 使用测试工具库中的方法验证实际变量标签与预期的字典是否相等
            tm.assert_dict_equal(vl, vl_expected)

            # 断言数据文件的数据标签是否与预期一致
            assert rdr.data_label == "This is a  Ünicode data label"

    # 定义测试函数，测试将DataFrame写入并读取Stata数据文件的操作
    def test_read_write_dta5(self, temp_file):
        # 创建包含NaN值的DataFrame
        original = DataFrame(
            [(np.nan, np.nan, np.nan, np.nan, np.nan)],
            columns=["float_miss", "double_miss", "byte_miss", "int_miss", "long_miss"],
        )
        
        # 设置DataFrame的索引名为"index"
        original.index.name = "index"

        # 将DataFrame写入Stata格式的临时文件
        path = temp_file
        original.to_stata(path, convert_dates=None)
        
        # 读取并再次解析Stata文件，得到写入并读取后的DataFrame
        written_and_read_again = self.read_dta(path)

        # 设置预期结果为原始的DataFrame，且索引类型转换为np.int32
        expected = original
        expected.index = expected.index.astype(np.int32)
        
        # 使用测试工具库中的方法验证写入并读取后的DataFrame与预期结果的相等性
        tm.assert_frame_equal(written_and_read_again.set_index("index"), expected)

    # 定义测试函数，测试将DataFrame写入并读取Stata数据文件的操作（另一种测试用例）
    def test_write_dta6(self, datapath, temp_file):
        # 从CSV文件读取DataFrame作为原始数据
        original = self.read_csv(datapath("io", "data", "stata", "stata3.csv"))
        
        # 设置DataFrame的索引名为"index"，并将索引和特定列转换为np.int32类型
        original.index.name = "index"
        original.index = original.index.astype(np.int32)
        original["year"] = original["year"].astype(np.int32)
        original["quarter"] = original["quarter"].astype(np.int32)

        # 将DataFrame写入Stata格式的临时文件
        path = temp_file
        original.to_stata(path, convert_dates=None)
        
        # 读取并再次解析Stata文件，得到写入并读取后的DataFrame
        written_and_read_again = self.read_dta(path)
        
        # 使用测试工具库中的方法验证写入并读取后的DataFrame与原始DataFrame的相等性（忽略索引类型检查）
        tm.assert_frame_equal(
            written_and_read_again.set_index("index"),
            original,
            check_index_type=False,
        )

    # 使用pytest的参数化功能，定义版本参数化测试
    @pytest.mark.parametrize("version", [114, 117, 118, 119, None])
    # 定义测试函数，用于测试读写 DTA10 文件格式
    def test_read_write_dta10(self, version, temp_file):
        # 创建包含数据的 DataFrame 对象
        original = DataFrame(
            data=[["string", "object", 1, 1.1, np.datetime64("2003-12-25")]],
            columns=["string", "object", "integer", "floating", "datetime"],
        )
        # 将 "object" 列转换为对象类型
        original["object"] = Series(original["object"], dtype=object)
        # 设置索引的名称为 "index"
        original.index.name = "index"
        # 将索引类型转换为 np.int32
        original.index = original.index.astype(np.int32)
        # 将 "integer" 列的类型转换为 np.int32
        original["integer"] = original["integer"].astype(np.int32)

        # 将 DataFrame 对象写入指定路径的 DTA10 文件，将 datetime 列以 "tc" 方式转换为毫秒级别
        path = temp_file
        original.to_stata(path, convert_dates={"datetime": "tc"}, version=version)
        # 从文件中读取写入的 DTA10 文件，并返回读取的 DataFrame 对象
        written_and_read_again = self.read_dta(path)

        # 创建期望的 DataFrame 对象，与原始数据一致
        expected = original[:]
        # 将 "datetime" 列的类型转换为 "M8[ms]"
        expected["datetime"] = expected["datetime"].astype("M8[ms]")

        # 断言写入并再次读取的 DataFrame 与期望的 DataFrame 在索引上完全相等
        tm.assert_frame_equal(
            written_and_read_again.set_index("index"),
            expected,
        )

    # 测试函数，验证 Stata 文件的文档示例
    def test_stata_doc_examples(self, temp_file):
        # 创建包含随机数据的 DataFrame 对象
        path = temp_file
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)), columns=list("AB")
        )
        # 将 DataFrame 对象写入指定路径的 Stata 文件
        df.to_stata(path)

    # 测试函数，验证写入 Stata 文件后原始数据保持不变
    def test_write_preserves_original(self, temp_file):
        # 创建包含随机数据的 DataFrame 对象
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 4)), columns=list("abcd")
        )
        # 将部分数据设置为 NaN，并创建数据的副本
        df.loc[2, "a":"c"] = np.nan
        df_copy = df.copy()
        path = temp_file
        # 将 DataFrame 对象写入指定路径的 Stata 文件，不写入索引
        df.to_stata(path, write_index=False)
        # 断言写入的 DataFrame 与原始的副本 DataFrame 在所有方面完全相等
        tm.assert_frame_equal(df, df_copy)

    # 使用参数化测试，验证编码处理的正确性
    @pytest.mark.parametrize("version", [114, 117, 118, 119, None])
    def test_encoding(self, version, datapath, temp_file):
        # 从指定路径读取原始的 Stata 文件
        raw = read_stata(datapath("io", "data", "stata", "stata1_encoding.dta"))
        # 从指定路径读取编码后的 Stata 文件
        encoded = read_stata(datapath("io", "data", "stata", "stata1_encoding.dta"))
        # 获取编码后的 DataFrame 的特定值
        result = encoded.kreis1849[0]

        # 获取期望的原始 DataFrame 的特定值
        expected = raw.kreis1849[0]
        # 断言结果值与期望值相等
        assert result == expected
        # 断言结果值的类型为字符串
        assert isinstance(result, str)

        # 将编码后的 DataFrame 写入指定路径的 Stata 文件，不写入索引
        path = temp_file
        encoded.to_stata(path, write_index=False, version=version)
        # 重新读取写入的 Stata 文件作为 DataFrame 对象
        reread_encoded = read_stata(path)
        # 断言编码后的 DataFrame 与重新读取的 DataFrame 在所有方面完全相等
        tm.assert_frame_equal(encoded, reread_encoded)
    # 定义一个测试函数，测试读写 Stata 数据文件功能，使用 pytest 框架
    def test_read_write_dta11(self, temp_file):
        # 创建一个原始数据框，包含一个元组作为数据，列名为字符串和非 ASCII 字符
        original = DataFrame(
            [(1, 2, 3, 4)],
            columns=[
                "good",
                "b\u00e4d",
                "8number",
                "astringwithmorethan32characters______",
            ],
        )
        # 创建一个格式化后的数据框，调整列名以符合 Stata 变量名的规范
        formatted = DataFrame(
            [(1, 2, 3, 4)],
            columns=["good", "b_d", "_8number", "astringwithmorethan32characters_"],
        )
        # 设置格式化后的数据框的索引名为 "index"，并将所有数据转换为 np.int32 类型
        formatted.index.name = "index"
        formatted = formatted.astype(np.int32)

        # 指定临时文件路径
        path = temp_file
        # 设置预期的警告消息
        msg = "Not all pandas column names were valid Stata variable names"
        # 使用 pytest 的 assert_produces_warning 上下文，测试原始数据框转换为 Stata 文件时是否会产生特定警告
        with tm.assert_produces_warning(InvalidColumnName, match=msg):
            original.to_stata(path, convert_dates=None)

        # 从指定路径读取并解析写入的 Stata 数据文件
        written_and_read_again = self.read_dta(path)

        # 设置期望的格式化后的数据框
        expected = formatted
        expected.index = expected.index.astype(np.int32)
        # 使用 pytest 的 assert_frame_equal 函数比较读取的数据框与期望的格式化数据框
        tm.assert_frame_equal(written_and_read_again.set_index("index"), expected)

    # 使用 pytest 的参数化标记定义多个测试用例，测试不同版本的 Stata 数据文件的读写功能
    @pytest.mark.parametrize("version", [114, 117, 118, 119, None])
    def test_read_write_dta12(self, version, temp_file):
        # 创建一个原始数据框，包含一个元组作为数据，列名为包含特殊字符的字符串
        original = DataFrame(
            [(1, 2, 3, 4, 5, 6)],
            columns=[
                "astringwithmorethan32characters_1",
                "astringwithmorethan32characters_2",
                "+",
                "-",
                "short",
                "delete",
            ],
        )
        # 创建一个格式化后的数据框，调整列名以符合 Stata 变量名的规范
        formatted = DataFrame(
            [(1, 2, 3, 4, 5, 6)],
            columns=[
                "astringwithmorethan32characters_",
                "_0astringwithmorethan32character",
                "_",
                "_1_",
                "_short",
                "_delete",
            ],
        )
        # 设置格式化后的数据框的索引名为 "index"，并将所有数据转换为 np.int32 类型
        formatted.index.name = "index"
        formatted = formatted.astype(np.int32)

        # 指定临时文件路径
        path = temp_file
        # 设置预期的警告消息
        msg = "Not all pandas column names were valid Stata variable names"
        # 使用 pytest 的 assert_produces_warning 上下文，测试原始数据框转换为 Stata 文件时是否会产生特定警告
        with tm.assert_produces_warning(InvalidColumnName, match=msg):
            # 转换原始数据框为 Stata 文件，指定 Stata 文件版本号
            original.to_stata(path, convert_dates=None, version=version)
            # 应该会因为格式不符合要求而产生警告

        # 从指定路径读取并解析写入的 Stata 数据文件
        written_and_read_again = self.read_dta(path)

        # 设置期望的格式化后的数据框
        expected = formatted
        expected.index = expected.index.astype(np.int32)
        # 使用 pytest 的 assert_frame_equal 函数比较读取的数据框与期望的格式化数据框
        tm.assert_frame_equal(written_and_read_again.set_index("index"), expected)
    # 定义测试函数，测试读取和写入 STATA 格式文件的功能
    def test_read_write_dta13(self, temp_file):
        # 创建三个不同数据类型的 Pandas Series 对象
        s1 = Series(2**9, dtype=np.int16)
        s2 = Series(2**17, dtype=np.int32)
        s3 = Series(2**33, dtype=np.int64)
        # 创建包含这些 Series 的原始 DataFrame
        original = DataFrame({"int16": s1, "int32": s2, "int64": s3})
        # 设置 DataFrame 的索引名称为 "index"
        original.index.name = "index"

        # 将 formatted 指向 original 的引用
        formatted = original
        # 将 "int64" 列的数据类型转换为 np.float64
        formatted["int64"] = formatted["int64"].astype(np.float64)

        # 将 DataFrame 写入临时文件中
        path = temp_file
        original.to_stata(path)
        # 读取并返回写入的 STATA 文件内容
        written_and_read_again = self.read_dta(path)

        # 期望的 DataFrame 是转换后的 formatted
        expected = formatted
        # 将 expected 的索引数据类型转换为 np.int32
        expected.index = expected.index.astype(np.int32)
        # 断言写入和再次读取的 DataFrame 与期望的 DataFrame 相等
        tm.assert_frame_equal(written_and_read_again.set_index("index"), expected)

    @pytest.mark.parametrize("version", [114, 117, 118, 119, None])
    @pytest.mark.parametrize(
        "file", ["stata5_113", "stata5_114", "stata5_115", "stata5_117"]
    )
    # 定义测试函数，测试读取、写入和再次读取 STATA 格式文件的功能
    def test_read_write_reread_dta14(
        self, file, parsed_114, version, datapath, temp_file
    ):
        # 从数据路径读取特定文件的内容并解析
        file = datapath("io", "data", "stata", f"{file}.dta")
        parsed = self.read_dta(file)
        # 设置解析后 DataFrame 的索引名称为 "index"
        parsed.index.name = "index"

        # 断言解析后的 DataFrame 与预解析的 parsed_114 相等
        tm.assert_frame_equal(parsed_114, parsed)

        # 将 parsed_114 DataFrame 写入临时文件，并指定转换日期格式和版本号
        path = temp_file
        parsed_114.to_stata(path, convert_dates={"date_td": "td"}, version=version)
        # 读取并返回写入的 STATA 文件内容
        written_and_read_again = self.read_dta(path)

        # 期望的 DataFrame 是 parsed_114 的副本
        expected = parsed_114.copy()
        # 断言写入和再次读取的 DataFrame 与期望的 DataFrame 相等
        tm.assert_frame_equal(written_and_read_again.set_index("index"), expected)

    @pytest.mark.parametrize(
        "file", ["stata6_113", "stata6_114", "stata6_115", "stata6_117"]
    )
    # 定义测试函数，测试读取、写入和再次读取 STATA 格式文件的功能
    def test_read_write_reread_dta15(self, file, datapath):
        # 从 CSV 文件读取预期的 DataFrame
        expected = self.read_csv(datapath("io", "data", "stata", "stata6.csv"))
        # 将 "byte_", "int_", "long_", "float_", "double_" 列的数据类型转换为指定类型
        expected["byte_"] = expected["byte_"].astype(np.int8)
        expected["int_"] = expected["int_"].astype(np.int16)
        expected["long_"] = expected["long_"].astype(np.int32)
        expected["float_"] = expected["float_"].astype(np.float32)
        expected["double_"] = expected["double_"].astype(np.float64)

        # 将 "date_td" 列的数据类型转换为 "Period[D]"，然后转换为 "M8[s]" 类型
        arr = expected["date_td"].astype("Period[D]")._values.asfreq("s", how="S")
        expected["date_td"] = arr.view("M8[s]")

        # 从数据路径读取特定文件的内容并解析
        file = datapath("io", "data", "stata", f"{file}.dta")
        parsed = self.read_dta(file)

        # 断言解析后的 DataFrame 与预期的 DataFrame 相等
        tm.assert_frame_equal(expected, parsed)

    @pytest.mark.parametrize("version", [114, 117, 118, 119, None])
    # 定义测试函数，测试写入 STATA 格式文件时的时间戳和数据标签功能
    def test_timestamp_and_label(self, version, temp_file):
        # 创建包含一个变量的原始 DataFrame
        original = DataFrame([(1,)], columns=["variable"])
        # 设置时间戳和数据标签
        time_stamp = datetime(2000, 2, 29, 14, 21)
        data_label = "This is a data file."
        path = temp_file
        # 将 DataFrame 写入临时文件，包含时间戳和数据标签，指定版本号
        original.to_stata(
            path, time_stamp=time_stamp, data_label=data_label, version=version
        )

        # 使用 StataReader 打开临时文件并断言时间戳和数据标签
        with StataReader(path) as reader:
            assert reader.time_stamp == "29 Feb 2000 14:21"
            assert reader.data_label == data_label

    @pytest.mark.parametrize("version", [114, 117, 118, 119, None])
    # 测试在时间戳为无效格式时抛出 ValueError 异常，并验证文件未生成
    def test_invalid_timestamp(self, version, temp_file):
        # 创建一个包含单列数据的 DataFrame
        original = DataFrame([(1,)], columns=["variable"])
        # 设定一个无效的时间戳字符串
        time_stamp = "01 Jan 2000, 00:00:00"
        # 获取临时文件路径
        path = temp_file
        # 用于匹配的错误消息
        msg = "time_stamp should be datetime type"
        # 使用 pytest 断言检查是否抛出了 ValueError 异常，并且异常消息符合预期
        with pytest.raises(ValueError, match=msg):
            original.to_stata(path, time_stamp=time_stamp, version=version)
        # 确保临时文件未生成
        assert not os.path.isfile(path)

    # 测试在存在数值列名无效格式时是否会产生警告
    def test_numeric_column_names(self, temp_file):
        # 创建一个包含浮点数的 DataFrame
        original = DataFrame(np.reshape(np.arange(25.0), (5, 5)))
        # 设定 DataFrame 的索引名
        original.index.name = "index"
        # 获取临时文件路径
        path = temp_file
        # 用于匹配的警告消息
        msg = "Not all pandas column names were valid Stata variable names"
        # 使用 tm.assert_produces_warning 断言检查是否产生了指定类型的警告，并且警告消息符合预期
        with tm.assert_produces_warning(InvalidColumnName, match=msg):
            original.to_stata(path)

        # 读取并再次加载写入的数据文件
        written_and_read_again = self.read_dta(path)

        # 重新设置索引名
        written_and_read_again = written_and_read_again.set_index("index")
        # 获取列名列表
        columns = list(written_and_read_again.columns)
        # 定义一个函数用于转换列名格式
        convert_col_name = lambda x: int(x[1])
        # 将列名转换为指定格式
        written_and_read_again.columns = map(convert_col_name, columns)

        # 设置期望的 DataFrame
        expected = original
        # 使用 tm.assert_frame_equal 断言验证写入和再次读取的 DataFrame 是否相等
        tm.assert_frame_equal(expected, written_and_read_again)

    # 使用 pytest 参数化测试不同的 Stata 版本是否正确处理 NaN 值
    @pytest.mark.parametrize("version", [114, 117, 118, 119, None])
    def test_nan_to_missing_value(self, version, temp_file):
        # 创建两个包含 NaN 值的 Series
        s1 = Series(np.arange(4.0), dtype=np.float32)
        s2 = Series(np.arange(4.0), dtype=np.float64)
        s1[::2] = np.nan
        s2[1::2] = np.nan
        # 创建一个包含这两个 Series 的 DataFrame
        original = DataFrame({"s1": s1, "s2": s2})
        # 设定 DataFrame 的索引名
        original.index.name = "index"

        # 获取临时文件路径
        path = temp_file
        # 将 DataFrame 写入 Stata 文件
        original.to_stata(path, version=version)
        # 读取并再次加载写入的数据文件
        written_and_read_again = self.read_dta(path)

        # 重新设置索引名
        written_and_read_again = written_and_read_again.set_index("index")
        # 设置期望的 DataFrame
        expected = original
        # 使用 tm.assert_frame_equal 断言验证写入和再次读取的 DataFrame 是否相等
        tm.assert_frame_equal(written_and_read_again, expected)

    # 测试在不写入索引的情况下是否抛出 KeyError 异常
    def test_no_index(self, temp_file):
        # 创建包含两列数据的 DataFrame
        columns = ["x", "y"]
        original = DataFrame(np.reshape(np.arange(10.0), (5, 2)), columns=columns)
        # 设定 DataFrame 的索引名
        original.index.name = "index_not_written"
        # 获取临时文件路径
        path = temp_file
        # 将 DataFrame 写入 Stata 文件，但不包含索引
        original.to_stata(path, write_index=False)
        # 读取并再次加载写入的数据文件
        written_and_read_again = self.read_dta(path)
        # 使用 pytest 断言检查是否抛出了 KeyError 异常，并且异常消息符合预期
        with pytest.raises(KeyError, match=original.index.name):
            written_and_read_again["index_not_written"]

    # 测试在 DataFrame 中包含字符串列和无日期数据时是否能正确写入和读取
    def test_string_no_dates(self, temp_file):
        # 创建包含字符串和浮点数列的 DataFrame
        s1 = Series(["a", "A longer string"])
        s2 = Series([1.0, 2.0], dtype=np.float64)
        original = DataFrame({"s1": s1, "s2": s2})
        # 设定 DataFrame 的索引名
        original.index.name = "index"
        # 获取临时文件路径
        path = temp_file
        # 将 DataFrame 写入 Stata 文件
        original.to_stata(path)
        # 读取并再次加载写入的数据文件
        written_and_read_again = self.read_dta(path)

        # 设置期望的 DataFrame
        expected = original
        # 使用 tm.assert_frame_equal 断言验证写入和再次读取的 DataFrame 是否相等
        tm.assert_frame_equal(written_and_read_again.set_index("index"), expected)
    # 测试大数值的转换
    def test_large_value_conversion(self, temp_file):
        # 创建包含不同数据类型和数值范围的 Series 对象
        s0 = Series([1, 99], dtype=np.int8)
        s1 = Series([1, 127], dtype=np.int8)
        s2 = Series([1, 2**15 - 1], dtype=np.int16)
        s3 = Series([1, 2**63 - 1], dtype=np.int64)
        # 将这些 Series 组成一个 DataFrame 对象
        original = DataFrame({"s0": s0, "s1": s1, "s2": s2, "s3": s3})
        # 设置 DataFrame 的索引名称为 "index"
        original.index.name = "index"
        # 将 DataFrame 写入到临时文件 temp_file 中，同时期望产生一个警告
        path = temp_file
        with tm.assert_produces_warning(PossiblePrecisionLoss, match="from int64 to"):
            original.to_stata(path)

        # 从写入的 Stata 文件中再次读取数据
        written_and_read_again = self.read_dta(path)

        # 修改原始 DataFrame
        modified = original
        # 将 "s1" 列的数据类型改为 np.int16
        modified["s1"] = Series(modified["s1"], dtype=np.int16)
        # 将 "s2" 列的数据类型改为 np.int32
        modified["s2"] = Series(modified["s2"], dtype=np.int32)
        # 将 "s3" 列的数据类型改为 np.float64
        modified["s3"] = Series(modified["s3"], dtype=np.float64)
        # 断言经过写入和再次读取的数据与修改后的 DataFrame 相等
        tm.assert_frame_equal(written_and_read_again.set_index("index"), modified)

    # 测试日期无效列名
    def test_dates_invalid_column(self, temp_file):
        # 创建一个包含日期时间对象的 DataFrame
        original = DataFrame([datetime(2006, 11, 19, 23, 13, 20)])
        # 设置 DataFrame 的索引名称为 "index"
        original.index.name = "index"
        # 将 DataFrame 写入到临时文件 temp_file 中，同时期望产生一个特定的警告信息
        path = temp_file
        msg = "Not all pandas column names were valid Stata variable names"
        with tm.assert_produces_warning(InvalidColumnName, match=msg):
            original.to_stata(path, convert_dates={0: "tc"})

        # 从写入的 Stata 文件中再次读取数据
        written_and_read_again = self.read_dta(path)

        # 构建预期的 DataFrame
        expected = original.copy()
        expected.columns = ["_0"]
        expected.index = original.index.astype(np.int32)
        expected["_0"] = expected["_0"].astype("M8[ms]")
        # 断言经过写入和再次读取的数据与预期的 DataFrame 相等
        tm.assert_frame_equal(written_and_read_again.set_index("index"), expected)

    # 测试读取特定数据集中的数据
    def test_105(self, datapath):
        # 使用给定路径读取特定的 Stata 数据文件
        dpath = datapath("io", "data", "stata", "S4_EDUC1.dta")
        df = read_stata(dpath)
        # 创建一个预期的 DataFrame 对象，包含特定的数据和列名
        df0 = [[1, 1, 3, -2], [2, 1, 2, -2], [4, 1, 1, -2]]
        df0 = DataFrame(df0)
        df0.columns = ["clustnum", "pri_schl", "psch_num", "psch_dis"]
        df0["clustnum"] = df0["clustnum"].astype(np.int16)
        df0["pri_schl"] = df0["pri_schl"].astype(np.int8)
        df0["psch_num"] = df0["psch_num"].astype(np.int8)
        df0["psch_dis"] = df0["psch_dis"].astype(np.float32)
        # 断言读取的前三行数据与预期的 DataFrame 相等
        tm.assert_frame_equal(df.head(3), df0)

    # 测试旧格式的值标签处理
    def test_value_labels_old_format(self, datapath):
        # GH 19417
        #
        # 测试当文件格式不支持值标签时，value_labels() 是否返回空字典
        dpath = datapath("io", "data", "stata", "S4_EDUC1.dta")
        with StataReader(dpath) as reader:
            assert reader.value_labels() == {}
    # 测试不同的日期导出格式
    def test_date_export_formats(self, temp_file):
        # 定义列名
        columns = ["tc", "td", "tw", "tm", "tq", "th", "ty"]
        # 列名到自身的映射，用于日期格式转换
        conversions = {c: c for c in columns}
        # 创建包含相同日期数据的 DataFrame
        data = [datetime(2006, 11, 20, 23, 13, 20)] * len(columns)
        original = DataFrame([data], columns=columns)
        # 设置 DataFrame 的索引名
        original.index.name = "index"
        # 预期的日期数值，对应每一列不同的日期单位
        expected_values = [
            datetime(2006, 11, 20, 23, 13, 20),  # Time
            datetime(2006, 11, 20),  # Day
            datetime(2006, 11, 19),  # Week
            datetime(2006, 11, 1),  # Month
            datetime(2006, 10, 1),  # Quarter year
            datetime(2006, 7, 1),  # Half year
            datetime(2006, 1, 1),
        ]  # Year

        # 创建预期的 DataFrame，设置数据类型为 datetime64[s]
        expected = DataFrame(
            [expected_values],
            index=pd.Index([0], dtype=np.int32, name="index"),
            columns=columns,
            dtype="M8[s]",
        )
        # 将 "tc" 列的数据类型转换为 datetime64[ms]
        expected["tc"] = expected["tc"].astype("M8[ms]")

        # 临时文件路径
        path = temp_file
        # 将原始 DataFrame 以 STATA 格式写入到文件
        original.to_stata(path, convert_dates=conversions)
        # 读取并再次写入的数据框
        written_and_read_again = self.read_dta(path)

        # 使用 pandas 测试模块比较写入和读取后的数据框是否相等
        tm.assert_frame_equal(written_and_read_again.set_index("index"), expected)

    # 测试写入缺失字符串
    def test_write_missing_strings(self, temp_file):
        # 创建包含一个字符串和一个缺失值的 DataFrame
        original = DataFrame([["1"], [None]], columns=["foo"])

        # 预期的 DataFrame，将缺失值替换为空字符串
        expected = DataFrame(
            [["1"], [""]],
            index=pd.RangeIndex(2, name="index"),
            columns=["foo"],
        )

        # 临时文件路径
        path = temp_file
        # 将原始 DataFrame 以 STATA 格式写入到文件
        original.to_stata(path)
        # 读取并再次写入的数据框
        written_and_read_again = self.read_dta(path)

        # 使用 pandas 测试模块比较写入和读取后的数据框是否相等
        tm.assert_frame_equal(written_and_read_again.set_index("index"), expected)

    # 使用参数化测试不同的版本和字节顺序
    @pytest.mark.parametrize("version", [114, 117, 118, 119, None])
    @pytest.mark.parametrize("byteorder", [">", "<"])
    def test_bool_uint(self, byteorder, version, temp_file):
        # 创建包含不同类型数据的 Series
        s0 = Series([0, 1, True], dtype=np.bool_)
        s1 = Series([0, 1, 100], dtype=np.uint8)
        s2 = Series([0, 1, 255], dtype=np.uint8)
        s3 = Series([0, 1, 2**15 - 100], dtype=np.uint16)
        s4 = Series([0, 1, 2**16 - 1], dtype=np.uint16)
        s5 = Series([0, 1, 2**31 - 100], dtype=np.uint32)
        s6 = Series([0, 1, 2**32 - 1], dtype=np.uint32)

        # 创建包含上述 Series 的 DataFrame
        original = DataFrame(
            {"s0": s0, "s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6}
        )
        # 设置 DataFrame 的索引名
        original.index.name = "index"

        # 临时文件路径
        path = temp_file
        # 将原始 DataFrame 以 STATA 格式写入到文件，指定版本和字节顺序
        original.to_stata(path, byteorder=byteorder, version=version)
        # 读取并再次写入的数据框
        written_and_read_again = self.read_dta(path)

        # 将读取后的数据框设置索引，并与原始数据框进行比较
        written_and_read_again = written_and_read_again.set_index("index")

        # 准备预期的 DataFrame，将列数据类型转换为指定类型
        expected = original
        expected_types = (
            np.int8,
            np.int8,
            np.int16,
            np.int16,
            np.int32,
            np.int32,
            np.float64,
        )
        for c, t in zip(expected.columns, expected_types):
            expected[c] = expected[c].astype(t)

        # 使用 pandas 测试模块比较写入和读取后的数据框是否相等
        tm.assert_frame_equal(written_and_read_again, expected)
    # 测试变量标签方法
    def test_variable_labels(self, datapath):
        # 使用 StataReader 读取给定路径下的 stata7_115.dta 文件，并获取其变量标签字典
        with StataReader(datapath("io", "data", "stata", "stata7_115.dta")) as rdr:
            sr_115 = rdr.variable_labels()
        # 使用 StataReader 读取给定路径下的 stata7_117.dta 文件，并获取其变量标签字典
        with StataReader(datapath("io", "data", "stata", "stata7_117.dta")) as rdr:
            sr_117 = rdr.variable_labels()
        # 预定义键和标签，用于后续断言
        keys = ("var1", "var2", "var3")
        labels = ("label1", "label2", "label3")
        # 检查两个数据集中变量标签是否一致，并且存在于预定义的键和标签中
        for k, v in sr_115.items():
            assert k in sr_117
            assert v == sr_117[k]
            assert k in keys
            assert v in labels

    # 测试最小尺寸列
    def test_minimal_size_col(self, temp_file):
        # 定义字符串长度列表
        str_lens = (1, 100, 244)
        s = {}
        # 根据字符串长度创建 Series 对象，并添加到字典中
        for str_len in str_lens:
            s["s" + str(str_len)] = Series(
                ["a" * str_len, "b" * str_len, "c" * str_len]
            )
        # 创建原始 DataFrame
        original = DataFrame(s)
        # 将 DataFrame 写入 Stata 文件
        path = temp_file
        original.to_stata(path, write_index=False)

        # 使用 StataReader 打开路径下的 Stata 文件
        with StataReader(path) as sr:
            sr._ensure_open()  # 确保 StataReader 已经打开文件，初始化 `_varlist` 等变量
            # 遍历 `_varlist`、`_fmtlist` 和 `_typlist`，检查格式是否正确
            for variable, fmt, typ in zip(sr._varlist, sr._fmtlist, sr._typlist):
                assert int(variable[1:]) == int(fmt[1:-1])
                assert int(variable[1:]) == typ

    # 测试过长字符串
    def test_excessively_long_string(self, temp_file):
        # 定义字符串长度列表
        str_lens = (1, 244, 500)
        s = {}
        # 根据字符串长度创建 Series 对象，并添加到字典中
        for str_len in str_lens:
            s["s" + str(str_len)] = Series(
                ["a" * str_len, "b" * str_len, "c" * str_len]
            )
        # 创建原始 DataFrame
        original = DataFrame(s)
        # 定义预期的错误消息，用于检查是否引发 ValueError 异常
        msg = (
            r"Fixed width strings in Stata \.dta files are limited to 244 "
            r"\(or fewer\)\ncharacters\.  Column 's500' does not satisfy "
            r"this restriction\. Use the\n'version=117' parameter to write "
            r"the newer \(Stata 13 and later\) format\."
        )
        # 使用 pytest 检查是否引发预期的 ValueError 异常，并检查其错误消息
        with pytest.raises(ValueError, match=msg):
            path = temp_file
            original.to_stata(path)
    # 测试生成缺失值的情况，使用临时文件路径 temp_file
    def test_missing_value_generator(self, temp_file):
        # 定义三种数据类型
        types = ("b", "h", "l")
        # 创建包含单个浮点数列的 DataFrame
        df = DataFrame([[0.0]], columns=["float_"])
        # 设置文件路径为临时文件路径 temp_file，将 DataFrame 写入为 Stata 文件格式
        path = temp_file
        df.to_stata(path)
        # 使用 StataReader 打开路径下的文件，并获取其有效范围
        with StataReader(path) as rdr:
            valid_range = rdr.VALID_RANGE
        # 生成期待的缺失值列表，以 ".a" 到 ".z" 的形式
        expected_values = ["." + chr(97 + i) for i in range(26)]
        expected_values.insert(0, ".")
        # 遍历三种数据类型
        for t in types:
            # 获取当前数据类型的偏移量
            offset = valid_range[t][1]
            # 遍历生成对应类型的 27 个缺失值对象，并验证其字符串表示与期待值是否一致
            for i in range(27):
                val = StataMissingValue(offset + 1 + i)
                assert val.string == expected_values[i]

        # 测试浮点数的极端情况下的缺失值
        val = StataMissingValue(struct.unpack("<f", b"\x00\x00\x00\x7f")[0])
        assert val.string == "."
        val = StataMissingValue(struct.unpack("<f", b"\x00\xd0\x00\x7f")[0])
        assert val.string == ".z"

        # 测试双精度浮点数的极端情况下的缺失值
        val = StataMissingValue(
            struct.unpack("<d", b"\x00\x00\x00\x00\x00\x00\xe0\x7f")[0]
        )
        assert val.string == "."
        val = StataMissingValue(
            struct.unpack("<d", b"\x00\x00\x00\x00\x00\x1a\xe0\x7f")[0]
        )
        assert val.string == ".z"

    # 使用参数化测试测试缺失值的转换情况，file 参数为文件名列表
    @pytest.mark.parametrize("file", ["stata8_113", "stata8_115", "stata8_117"])
    def test_missing_value_conversion(self, file, datapath):
        # 定义列名
        columns = ["int8_", "int16_", "int32_", "float32_", "float64_"]
        # 创建 StataMissingValue 对象并获取其 MISSING_VALUES 字典的键列表
        smv = StataMissingValue(101)
        keys = sorted(smv.MISSING_VALUES.keys())
        data = []
        # 遍历生成 27 行数据，每行包含不同数据类型的缺失值对象
        for i in range(27):
            row = [StataMissingValue(keys[i + (j * 27)]) for j in range(5)]
            data.append(row)
        # 创建期待的 DataFrame，包含不同数据类型的缺失值
        expected = DataFrame(data, columns=columns)

        # 读取 Stata 文件，并转换其中的缺失值为期待的格式，与 expected 进行比较
        parsed = read_stata(
            datapath("io", "data", "stata", f"{file}.dta"), convert_missing=True
        )
        # 使用测试工具验证 parsed 和 expected 的数据框是否相等
        tm.assert_frame_equal(parsed, expected)
    # 定义一个测试方法，用于测试处理大日期的函数
    def test_big_dates(self, datapath, temp_file):
        # 年份列表
        yr = [1960, 2000, 9999, 100, 2262, 1677]
        # 月份列表
        mo = [1, 1, 12, 1, 4, 9]
        # 日列表
        dd = [1, 1, 31, 1, 22, 23]
        # 小时列表
        hr = [0, 0, 23, 0, 0, 0]
        # 分钟列表
        mm = [0, 0, 59, 0, 0, 0]
        # 秒列表
        ss = [0, 0, 59, 0, 0, 0]
        # 预期输出的日期列表
        expected = []
        # 遍历日期的各个部分，并生成预期的日期列表
        for year, month, day, hour, minute, second in zip(yr, mo, dd, hr, mm, ss):
            row = []
            for j in range(7):
                if j == 0:
                    # 根据给定的年月日时分秒生成日期时间对象
                    row.append(datetime(year, month, day, hour, minute, second))
                elif j == 6:
                    # 对于索引为6的列，生成仅有年份的日期时间对象
                    row.append(datetime(year, 1, 1))
                else:
                    # 对于其他列，生成仅有年月日的日期时间对象
                    row.append(datetime(year, month, day))
            expected.append(row)
        # 在预期输出的末尾添加一个全为 NaT 的行
        expected.append([pd.NaT] * 7)
        
        # 列名列表
        columns = [
            "date_tc",
            "date_td",
            "date_tw",
            "date_tm",
            "date_tq",
            "date_th",
            "date_ty",
        ]

        # 修正周、季、半年、年的日期值
        expected[2][2] = datetime(9999, 12, 24)
        expected[2][3] = datetime(9999, 12, 1)
        expected[2][4] = datetime(9999, 10, 1)
        expected[2][5] = datetime(9999, 7, 1)
        expected[4][2] = datetime(2262, 4, 16)
        expected[4][3] = expected[4][4] = datetime(2262, 4, 1)
        expected[4][5] = expected[4][6] = datetime(2262, 1, 1)
        expected[5][2] = expected[5][3] = expected[5][4] = datetime(1677, 10, 1)
        expected[5][5] = expected[5][6] = datetime(1678, 1, 1)

        # 将预期输出转换为 DataFrame 对象，并指定列名和数据类型
        expected = DataFrame(expected, columns=columns, dtype=object)
        expected["date_tc"] = expected["date_tc"].astype("M8[ms]")
        expected["date_td"] = expected["date_td"].astype("M8[s]")
        expected["date_tm"] = expected["date_tm"].astype("M8[s]")
        expected["date_tw"] = expected["date_tw"].astype("M8[s]")
        expected["date_tq"] = expected["date_tq"].astype("M8[s]")
        expected["date_th"] = expected["date_th"].astype("M8[s]")
        expected["date_ty"] = expected["date_ty"].astype("M8[s]")

        # 解析读取给定路径下的两个 Stata 数据文件
        parsed_115 = read_stata(datapath("io", "data", "stata", "stata9_115.dta"))
        parsed_117 = read_stata(datapath("io", "data", "stata", "stata9_117.dta"))

        # 断言预期输出与解析结果相等
        tm.assert_frame_equal(expected, parsed_115)
        tm.assert_frame_equal(expected, parsed_117)

        # 创建一个字典，将列名映射为后两个字符的字符串
        date_conversion = {c: c[-2:] for c in columns}
        # 将预期输出的索引命名为 "index"
        expected.index.name = "index"
        # 将预期输出写入 Stata 数据文件
        expected.to_stata(path, convert_dates=date_conversion)
        # 从文件中读取并再次解析数据
        written_and_read_again = self.read_dta(path)

        # 断言写入并再次读取的结果与预期输出相等
        tm.assert_frame_equal(
            written_and_read_again.set_index("index"),
            expected.set_index(expected.index.astype(np.int32)),
        )
    # 定义测试函数，用于测试数据类型转换功能，接收datapath作为参数
    def test_dtype_conversion(self, datapath):
        # 从指定路径读取并解析 CSV 文件，返回数据框对象 expected
        expected = self.read_csv(datapath("io", "data", "stata", "stata6.csv"))
        # 将 "byte_" 列的数据类型转换为 np.int8
        expected["byte_"] = expected["byte_"].astype(np.int8)
        # 将 "int_" 列的数据类型转换为 np.int16
        expected["int_"] = expected["int_"].astype(np.int16)
        # 将 "long_" 列的数据类型转换为 np.int32
        expected["long_"] = expected["long_"].astype(np.int32)
        # 将 "float_" 列的数据类型转换为 np.float32
        expected["float_"] = expected["float_"].astype(np.float32)
        # 将 "double_" 列的数据类型转换为 np.float64
        expected["double_"] = expected["double_"].astype(np.float64)
        # 将 "date_td" 列的数据类型转换为 "M8[s]"，即 pandas 的日期时间类型
        expected["date_td"] = expected["date_td"].astype("M8[s]")

        # 调用 read_stata 函数，读取指定路径下的 stata 文件，转换日期类型
        no_conversion = read_stata(
            datapath("io", "data", "stata", "stata6_117.dta"), convert_dates=True
        )
        # 使用 pytest 比较预期的数据框 expected 和读取的数据框 no_conversion 是否相等
        tm.assert_frame_equal(expected, no_conversion)

        # 调用 read_stata 函数，读取指定路径下的 stata 文件，同时转换日期类型和保持数据类型不变
        conversion = read_stata(
            datapath("io", "data", "stata", "stata6_117.dta"),
            convert_dates=True,
            preserve_dtypes=False,
        )

        # 创建新的预期数据框 expected2，用于比较 read_stata 函数返回的数据框 conversion 中的 "date_td" 列
        expected2 = self.read_csv(datapath("io", "data", "stata", "stata6.csv"))
        expected2["date_td"] = expected["date_td"]

        # 使用 pytest 比较预期的数据框 expected2 和读取的数据框 conversion 是否相等
        tm.assert_frame_equal(expected2, conversion)

    # 定义测试函数，用于测试删除列的功能，接收datapath作为参数
    def test_drop_column(self, datapath):
        # 从指定路径读取并解析 CSV 文件，返回数据框对象 expected
        expected = self.read_csv(datapath("io", "data", "stata", "stata6.csv"))
        # 将 "byte_" 列的数据类型转换为 np.int8
        expected["byte_"] = expected["byte_"].astype(np.int8)
        # 将 "int_" 列的数据类型转换为 np.int16
        expected["int_"] = expected["int_"].astype(np.int16)
        # 将 "long_" 列的数据类型转换为 np.int32
        expected["long_"] = expected["long_"].astype(np.int32)
        # 将 "float_" 列的数据类型转换为 np.float32
        expected["float_"] = expected["float_"].astype(np.float32)
        # 将 "double_" 列的数据类型转换为 np.float64
        expected["double_"] = expected["double_"].astype(np.float64)
        # 将 "date_td" 列的数据类型转换为 datetime 对象，使用指定的日期格式
        expected["date_td"] = expected["date_td"].apply(
            datetime.strptime, args=("%Y-%m-%d",)
        )

        # 定义需要保留的列名列表
        columns = ["byte_", "int_", "long_"]
        # 从指定路径读取并解析 stata 文件，返回数据框对象 dropped，只保留指定的列
        expected = expected[columns]
        dropped = read_stata(
            datapath("io", "data", "stata", "stata6_117.dta"),
            convert_dates=True,
            columns=columns,
        )

        # 使用 pytest 比较预期的数据框 expected 和读取的数据框 dropped 是否相等
        tm.assert_frame_equal(expected, dropped)

        # 定义需要重新排序的列名列表
        columns = ["int_", "long_", "byte_"]
        # 从指定路径读取并解析 stata 文件，返回数据框对象 reordered，保留指定的列并按指定顺序排列
        reordered = read_stata(
            datapath("io", "data", "stata", "stata6_117.dta"),
            convert_dates=True,
            columns=columns,
        )
        # 使用 pytest 比较预期的数据框 expected 和读取的数据框 reordered 是否相等
        tm.assert_frame_equal(expected, reordered)

        # 定义期望引发 ValueError 异常的错误消息
        msg = "columns contains duplicate entries"
        # 使用 pytest 检查是否引发了预期的 ValueError 异常，并匹配指定的错误消息
        with pytest.raises(ValueError, match=msg):
            read_stata(
                datapath("io", "data", "stata", "stata6_117.dta"),
                convert_dates=True,
                columns=["byte_", "byte_"],
            )

        # 定义期望引发 ValueError 异常的错误消息
        msg = "The following columns were not found in the Stata data set: not_found"
        # 使用 pytest 检查是否引发了预期的 ValueError 异常，并匹配指定的错误消息
        with pytest.raises(ValueError, match=msg):
            read_stata(
                datapath("io", "data", "stata", "stata6_117.dta"),
                convert_dates=True,
                columns=["byte_", "int_", "long_", "not_found"],
            )
    @pytest.mark.filterwarnings(
        "ignore:\\nStata value:pandas.io.stata.ValueLabelTypeMismatch"
    )
    # 定义测试方法，忽略特定警告
    def test_categorical_writing(self, version, temp_file):
        # 创建包含分类数据的原始 DataFrame
        original = DataFrame.from_records(
            [
                ["one", "ten", "one", "one", "one", 1],
                ["two", "nine", "two", "two", "two", 2],
                ["three", "eight", "three", "three", "three", 3],
                ["four", "seven", 4, "four", "four", 4],
                ["five", "six", 5, np.nan, "five", 5],
                ["six", "five", 6, np.nan, "six", 6],
                ["seven", "four", 7, np.nan, "seven", 7],
                ["eight", "three", 8, np.nan, "eight", 8],
                ["nine", "two", 9, np.nan, "nine", 9],
                ["ten", "one", "ten", np.nan, "ten", 10],
            ],
            columns=[
                "fully_labeled",
                "fully_labeled2",
                "incompletely_labeled",
                "labeled_with_missings",
                "float_labelled",
                "unlabeled",
            ],
        )
        # 将原始 DataFrame 转换为 Stata 格式并写入临时文件
        path = temp_file
        original.astype("category").to_stata(path, version=version)
        # 从写入的 Stata 文件中读取并重新创建 DataFrame
        written_and_read_again = self.read_dta(path)

        # 设置读取的 DataFrame 的索引为 "index"
        res = written_and_read_again.set_index("index")

        # 准备预期的 DataFrame 结果
        expected = original
        expected.index = expected.index.set_names("index")

        # 将部分列转换为字符串类型
        expected["incompletely_labeled"] = expected["incompletely_labeled"].apply(str)
        expected["unlabeled"] = expected["unlabeled"].apply(str)
        # 遍历所有列，将其转换为分类类型
        for col in expected:
            orig = expected[col]

            cat = orig.astype("category")._values
            cat = cat.as_ordered()
            if col == "unlabeled":
                cat = cat.set_categories(orig, ordered=True)

            # 移除分类类型的标签名称
            cat.categories.rename(None, inplace=True)

            expected[col] = cat

        # 使用测试工具验证读取的 DataFrame 与预期结果是否一致
        tm.assert_frame_equal(res, expected)

    # 定义测试方法，检查分类数据的警告和错误处理
    def test_categorical_warnings_and_errors(self, temp_file):
        # 创建包含非字符串标签的原始 DataFrame，验证其长度是否超过限制
        original = DataFrame.from_records(
            [["a" * 10000], ["b" * 10000], ["c" * 10000], ["d" * 10000]],
            columns=["Too_long"],
        )

        original = original.astype("category")
        path = temp_file
        msg = (
            "Stata value labels for a single variable must have "
            r"a combined length less than 32,000 characters\."
        )
        # 使用 pytest 来验证是否引发了特定的 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            original.to_stata(path)

        # 创建包含混合内容的原始 DataFrame，验证是否引发了警告
        original = DataFrame.from_records(
            [["a"], ["b"], ["c"], ["d"], [1]], columns=["Too_long"]
        ).astype("category")

        msg = "data file created has not lost information due to duplicate labels"
        # 使用 pytest 来验证是否产生了特定的警告类型
        with tm.assert_produces_warning(ValueLabelTypeMismatch, match=msg):
            original.to_stata(path)
            # 应该会因为混合内容而得到警告

    @pytest.mark.parametrize("version", [114, 117, 118, 119, None])
    def test_categorical_with_stata_missing_values(self, version, temp_file):
        # 创建一个包含字符串和一个 NaN 值的列表，作为测试数据
        values = [["a" + str(i)] for i in range(120)]
        values.append([np.nan])
        
        # 从二维列表创建 DataFrame，列名为 "many_labels"
        original = DataFrame.from_records(values, columns=["many_labels"])
        
        # 将 DataFrame 中的每一列转换为 category 类型，并进行拼接
        original = pd.concat(
            [original[col].astype("category") for col in original], axis=1
        )
        
        # 设置 DataFrame 的索引名称为 "index"
        original.index.name = "index"
        
        # 将 DataFrame 写入到 Stata 文件
        path = temp_file
        original.to_stata(path, version=version)
        
        # 从生成的 Stata 文件中读取数据
        written_and_read_again = self.read_dta(path)
        
        # 将读取的数据设置索引为 "index"
        res = written_and_read_again.set_index("index")
        
        # 将预期结果设置为 original
        expected = original
        
        # 遍历 expected 的每一列，处理 category 类型的列
        for col in expected:
            cat = expected[col]._values
            # 移除未使用的 category，并获取新的 category 列表
            new_cats = cat.remove_unused_categories().categories
            # 设置 category 的新顺序为有序
            cat = cat.set_categories(new_cats, ordered=True)
            expected[col] = cat
        
        # 断言 res 和 expected 的 DataFrame 是否相等
        tm.assert_frame_equal(res, expected)

    @pytest.mark.parametrize("file", ["stata10_115", "stata10_117"])
    def test_categorical_order(self, file, datapath):
        # 定义预期的测试数据列表
        expected = [
            (True, "ordered", ["a", "b", "c", "d", "e"], np.arange(5)),
            (True, "reverse", ["a", "b", "c", "d", "e"], np.arange(5)[::-1]),
            (True, "noorder", ["a", "b", "c", "d", "e"], np.array([2, 1, 4, 0, 3])),
            (True, "floating", ["a", "b", "c", "d", "e"], np.arange(0, 5)),
            (True, "float_missing", ["a", "d", "e"], np.array([0, 1, 2, -1, -1])),
            (False, "nolabel", [1.0, 2.0, 3.0, 4.0, 5.0], np.arange(5)),
            (True, "int32_mixed", ["d", 2, "e", "b", "a"], np.arange(5)),
        ]
        
        # 初始化一个空列表用于存放 DataFrame 的列
        cols = []
        
        # 遍历预期数据列表，根据每个条目构造列数据
        for is_cat, col, labels, codes in expected:
            if is_cat:
                # 如果是 category 类型，则使用 pd.Categorical.from_codes 构造列
                cols.append(
                    (col, pd.Categorical.from_codes(codes, labels, ordered=True))
                )
            else:
                # 如果不是 category 类型，则使用 Series 构造列
                cols.append((col, Series(labels, dtype=np.float32)))
        
        # 从字典构建 DataFrame
        expected = DataFrame.from_dict(dict(cols))

        # 读取 Stata 文件，将解析结果存入 parsed
        file = datapath("io", "data", "stata", f"{file}.dta")
        parsed = read_stata(file)
        
        # 断言 expected 和 parsed 的 DataFrame 是否相等
        tm.assert_frame_equal(expected, parsed)

        # 检查 category 类型列的编码是否相同
        for col in expected:
            if isinstance(expected[col].dtype, CategoricalDtype):
                tm.assert_series_equal(expected[col].cat.codes, parsed[col].cat.codes)
                tm.assert_index_equal(
                    expected[col].cat.categories, parsed[col].cat.categories
                )
    # 定义一个测试方法，用于测试分类数据的排序
    def test_categorical_sorting(self, file, datapath):
        # 使用指定的数据路径读取 Stata 文件，并解析数据
        parsed = read_stata(datapath("io", "data", "stata", f"{file}.dta"))

        # 根据 'srh' 列的值进行排序，确保按照代码而非字符串进行排序，缺失值优先显示
        parsed = parsed.sort_values("srh", na_position="first")

        # 将索引重置为一个连续的整数范围，而不是原始数据的索引
        parsed.index = pd.RangeIndex(len(parsed))
        
        # 定义一个包含代码和对应分类名称的列表，用于创建分类数据
        codes = [-1, -1, 0, 1, 1, 1, 2, 2, 3, 4]
        categories = ["Poor", "Fair", "Good", "Very good", "Excellent"]
        
        # 使用 codes 和 categories 创建一个有序的分类数据
        cat = pd.Categorical.from_codes(
            codes=codes, categories=categories, ordered=True
        )
        
        # 创建一个预期的 Series 对象，以验证 'srh' 列的分类顺序
        expected = Series(cat, name="srh")
        
        # 使用 pytest 的断言函数验证预期的 Series 和实际的 parsed["srh"] 是否相等
        tm.assert_series_equal(expected, parsed["srh"])

    # 使用 pytest 的 parametrize 标记，测试多个文件的分类顺序
    @pytest.mark.parametrize("file", ["stata10_115", "stata10_117"])
    def test_categorical_ordering(self, file, datapath):
        # 构造文件的完整路径
        file = datapath("io", "data", "stata", f"{file}.dta")
        
        # 使用指定的数据路径读取 Stata 文件，并解析数据
        parsed = read_stata(file)

        # 使用 order_categoricals=False 参数读取未排序的数据
        parsed_unordered = read_stata(file, order_categoricals=False)
        
        # 遍历数据集中的每一列，检查是否是分类数据类型
        for col in parsed:
            if not isinstance(parsed[col].dtype, CategoricalDtype):
                continue
            # 断言分类数据是有序的
            assert parsed[col].cat.ordered
            # 断言未排序的分类数据是无序的
            assert not parsed_unordered[col].cat.ordered

    # 使用 pytest 的 parametrize 标记和 filterwarnings 忽略特定警告类型
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.parametrize(
        "file",
        [
            "stata1_117",
            "stata2_117",
            "stata3_117",
            "stata4_117",
            "stata5_117",
            "stata6_117",
            "stata7_117",
            "stata8_117",
            "stata9_117",
            "stata10_117",
            "stata11_117",
        ],
    )
    @pytest.mark.parametrize("chunksize", [1, 2])
    @pytest.mark.parametrize("convert_categoricals", [False, True])
    @pytest.mark.parametrize("convert_dates", [False, True])
    # 定义一个测试方法，用于测试分块读取和处理 Stata 数据文件
    def test_read_chunks_117(
        self, file, chunksize, convert_categoricals, convert_dates, datapath
    ):
        # 构造文件的完整路径
        fname = datapath("io", "data", "stata", f"{file}.dta")

        # 使用指定的数据路径读取 Stata 文件，并解析数据
        parsed = read_stata(
            fname,
            convert_categoricals=convert_categoricals,
            convert_dates=convert_dates,
        )

        # 使用迭代器模式读取 Stata 文件，以支持分块读取
        with read_stata(
            fname,
            iterator=True,
            convert_categoricals=convert_categoricals,
            convert_dates=convert_dates,
        ) as itr:
            pos = 0
            # 循环读取前5个块数据
            for j in range(5):
                try:
                    chunk = itr.read(chunksize)
                except StopIteration:
                    break
                # 从整体数据集中抽取与当前块相同大小的数据块，并进行深拷贝
                from_frame = parsed.iloc[pos : pos + chunksize, :].copy()
                # 对数据块中的分类数据进行转换
                from_frame = self._convert_categorical(from_frame)
                # 使用 pytest 的断言函数验证预期的数据块与从迭代器中读取的块是否相等
                tm.assert_frame_equal(
                    from_frame,
                    chunk,
                    check_dtype=False,
                )
                pos += chunksize

    # 定义一个静态方法
    @staticmethod
    def _convert_categorical(from_frame: DataFrame) -> DataFrame:
        """
        Emulate the categorical casting behavior we expect from roundtripping.
        模拟我们期望在往返过程中的分类转换行为。
        """
        for col in from_frame:
            # 遍历数据框的每一列
            ser = from_frame[col]
            # 获取列对应的序列
            if isinstance(ser.dtype, CategoricalDtype):
                # 检查序列是否是分类数据类型
                cat = ser._values.remove_unused_categories()
                # 移除未使用的分类
                if cat.categories.dtype == object:
                    # 如果分类的数据类型是对象类型
                    categories = pd.Index._with_infer(cat.categories._values)
                    # 推断对象类型的分类数据
                    cat = cat.set_categories(categories)
                    # 设置分类数据
                from_frame[col] = cat
                # 更新数据框中的列为新的分类序列
        return from_frame
        # 返回更新后的数据框

    def test_iterator(self, datapath):
        """
        Test case for iterating over chunks of data from a Stata file.
        对从 Stata 文件中迭代获取数据块的测试用例。
        """
        fname = datapath("io", "data", "stata", "stata3_117.dta")
        # 获取测试数据文件的路径

        parsed = read_stata(fname)
        # 使用 read_stata 函数解析数据文件

        with read_stata(fname, iterator=True) as itr:
            # 使用 read_stata 函数打开文件迭代器模式
            chunk = itr.read(5)
            # 从迭代器中读取 5 行数据块
            tm.assert_frame_equal(parsed.iloc[0:5, :], chunk)
            # 比较读取的数据块与预期的数据块是否相等

        with read_stata(fname, chunksize=5) as itr:
            # 使用 read_stata 函数打开文件，并指定数据块大小为 5
            chunk = list(itr)
            # 将迭代器中的数据块转换为列表
            tm.assert_frame_equal(parsed.iloc[0:5, :], chunk[0])
            # 比较读取的数据块与预期的数据块是否相等

        with read_stata(fname, iterator=True) as itr:
            # 使用 read_stata 函数打开文件迭代器模式
            chunk = itr.get_chunk(5)
            # 从迭代器中获取特定大小的数据块
            tm.assert_frame_equal(parsed.iloc[0:5, :], chunk)
            # 比较读取的数据块与预期的数据块是否相等

        with read_stata(fname, chunksize=5) as itr:
            # 使用 read_stata 函数打开文件，并指定数据块大小为 5
            chunk = itr.get_chunk()
            # 从迭代器中获取下一个数据块
            tm.assert_frame_equal(parsed.iloc[0:5, :], chunk)
            # 比较读取的数据块与预期的数据块是否相等

        # GH12153
        with read_stata(fname, chunksize=4) as itr:
            # 使用 read_stata 函数打开文件，并指定数据块大小为 4
            from_chunks = pd.concat(itr)
            # 将迭代器中的数据块连接起来
        tm.assert_frame_equal(parsed, from_chunks)
        # 比较连接后的数据与预期的数据是否相等

    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.parametrize(
        "file",
        [
            "stata2_115",
            "stata3_115",
            "stata4_115",
            "stata5_115",
            "stata6_115",
            "stata7_115",
            "stata8_115",
            "stata9_115",
            "stata10_115",
            "stata11_115",
        ],
    )
    @pytest.mark.parametrize("chunksize", [1, 2])
    @pytest.mark.parametrize("convert_categoricals", [False, True])
    @pytest.mark.parametrize("convert_dates", [False, True])
    def test_read_chunks_115(
        self, file, chunksize, convert_categoricals, convert_dates, datapath
        ):
        """
        Parameterized test case for reading chunks from various Stata files.
        对从各种 Stata 文件中读取数据块的参数化测试用例。
        """
    ):
        # 构建要读取的文件路径
        fname = datapath("io", "data", "stata", f"{file}.dta")

        # 读取整个文件内容
        parsed = read_stata(
            fname,
            convert_categoricals=convert_categoricals,
            convert_dates=convert_dates,
        )

        # 逐块读取文件内容，并与整体读取结果进行比较
        with read_stata(
            fname,
            iterator=True,
            convert_dates=convert_dates,
            convert_categoricals=convert_categoricals,
        ) as itr:
            pos = 0
            # 循环读取前5个块
            for j in range(5):
                try:
                    chunk = itr.read(chunksize)
                except StopIteration:
                    break
                # 从整体读取结果中切出当前块大小的数据帧，并复制
                from_frame = parsed.iloc[pos : pos + chunksize, :].copy()
                # 对数据帧中的分类变量进行转换
                from_frame = self._convert_categorical(from_frame)
                # 使用测试框架验证当前块的数据帧与预期是否相等
                tm.assert_frame_equal(
                    from_frame,
                    chunk,
                    check_dtype=False,
                )
                pos += chunksize

    def test_read_chunks_columns(self, datapath):
        # 获取要读取的文件路径
        fname = datapath("io", "data", "stata", "stata3_117.dta")
        # 指定要读取的列名
        columns = ["quarter", "cpi", "m1"]
        chunksize = 2

        # 读取指定列的文件内容
        parsed = read_stata(fname, columns=columns)
        # 逐块读取文件内容，并与整体读取结果进行比较
        with read_stata(fname, iterator=True) as itr:
            pos = 0
            # 循环读取前5个块
            for j in range(5):
                # 逐块读取指定列的数据
                chunk = itr.read(chunksize, columns=columns)
                if chunk is None:
                    break
                # 从整体读取结果中切出当前块大小的数据帧
                from_frame = parsed.iloc[pos : pos + chunksize, :]
                # 使用测试框架验证当前块的数据帧与预期是否相等
                tm.assert_frame_equal(from_frame, chunk, check_dtype=False)
                pos += chunksize

    @pytest.mark.parametrize("version", [114, 117, 118, 119, None])
    def test_write_variable_labels(self, version, mixed_frame, temp_file):
        # GH 13631, 增加对变量标签的写入支持
        # 设置混合数据框的索引名
        mixed_frame.index.name = "index"
        # 定义变量标签
        variable_labels = {"a": "City Rank", "b": "City Exponent", "c": "City"}
        # 将混合数据框写入 Stata 文件，并附加变量标签
        path = temp_file
        mixed_frame.to_stata(path, variable_labels=variable_labels, version=version)
        # 使用 StataReader 读取文件，并获取变量标签
        with StataReader(path) as sr:
            read_labels = sr.variable_labels()
        # 预期的变量标签
        expected_labels = {
            "index": "",
            "a": "City Rank",
            "b": "City Exponent",
            "c": "City",
        }
        # 验证读取的变量标签与预期是否一致
        assert read_labels == expected_labels

        # 修改索引的变量标签
        variable_labels["index"] = "The Index"
        # 将更新后的混合数据框写入 Stata 文件，并附加更新后的变量标签
        path = temp_file
        mixed_frame.to_stata(path, variable_labels=variable_labels, version=version)
        # 重新使用 StataReader 读取文件，并获取变量标签
        with StataReader(path) as sr:
            read_labels = sr.variable_labels()
        # 验证读取的变量标签与更新后的预期是否一致
        assert read_labels == variable_labels

    @pytest.mark.parametrize("version", [114, 117, 118, 119, None])
    # 定义一个测试方法，用于测试无效的变量标签
    def test_invalid_variable_labels(self, version, mixed_frame, temp_file):
        # 设置混合框架的索引名称为 "index"
        mixed_frame.index.name = "index"
        # 定义变量标签字典，其中包含三个标签
        variable_labels = {"a": "very long" * 10, "b": "City Exponent", "c": "City"}
        # 设置文件路径为临时文件路径
        path = temp_file
        # 设置错误消息字符串
        msg = "Variable labels must be 80 characters or fewer"
        # 使用 pytest 来验证是否会引发 ValueError 异常，且异常消息与设定的 msg 匹配
        with pytest.raises(ValueError, match=msg):
            # 调用 mixed_frame 对象的 to_stata 方法，传入路径和变量标签字典作为参数
            mixed_frame.to_stata(path, variable_labels=variable_labels, version=version)

    # 使用 pytest 的参数化装饰器，测试无效的变量标签编码
    @pytest.mark.parametrize("version", [114, 117])
    def test_invalid_variable_label_encoding(self, version, mixed_frame, temp_file):
        # 设置混合框架的索引名称为 "index"
        mixed_frame.index.name = "index"
        # 定义变量标签字典，其中包含三个标签，其中一个标签包含无效字符 'Œ'
        variable_labels = {"a": "very long" * 10, "b": "City Exponent", "c": "City"}
        variable_labels["a"] = "invalid character Œ"
        # 设置文件路径为临时文件路径
        path = temp_file
        # 使用 pytest 来验证是否会引发 ValueError 异常，且异常消息包含 "Variable labels must contain only characters"
        with pytest.raises(
            ValueError, match="Variable labels must contain only characters"
        ):
            # 调用 mixed_frame 对象的 to_stata 方法，传入路径和变量标签字典作为参数
            mixed_frame.to_stata(path, variable_labels=variable_labels, version=version)

    # 测试写入变量标签时的错误情况
    def test_write_variable_label_errors(self, mixed_frame, temp_file):
        # 定义一个包含 Unicode 字符串的列表
        values = ["\u03a1", "\u0391", "\u039d", "\u0394", "\u0391", "\u03a3"]

        # 定义一个包含 UTF-8 编码的变量标签字典
        variable_labels_utf8 = {
            "a": "City Rank",
            "b": "City Exponent",
            "c": "".join(values),
        }

        # 设置错误消息字符串
        msg = (
            "Variable labels must contain only characters that can be "
            "encoded in Latin-1"
        )
        # 使用 pytest 来验证是否会引发 ValueError 异常，且异常消息与设定的 msg 匹配
        with pytest.raises(ValueError, match=msg):
            # 设置文件路径为临时文件路径
            path = temp_file
            # 调用 mixed_frame 对象的 to_stata 方法，传入路径和变量标签字典作为参数
            mixed_frame.to_stata(path, variable_labels=variable_labels_utf8)

        # 定义一个包含超过 80 个字符的变量标签字典
        variable_labels_long = {
            "a": "City Rank",
            "b": "City Exponent",
            "c": "A very, very, very long variable label "
            "that is too long for Stata which means "
            "that it has more than 80 characters",
        }

        # 设置错误消息字符串
        msg = "Variable labels must be 80 characters or fewer"
        # 使用 pytest 来验证是否会引发 ValueError 异常，且异常消息与设定的 msg 匹配
        with pytest.raises(ValueError, match=msg):
            # 设置文件路径为临时文件路径
            path = temp_file
            # 调用 mixed_frame 对象的 to_stata 方法，传入路径和变量标签字典作为参数
            mixed_frame.to_stata(path, variable_labels=variable_labels_long)
    # 定义测试方法，用于默认日期转换的测试
    def test_default_date_conversion(self, temp_file):
        # GH 12259
        # 创建包含几个日期时间对象的列表
        dates = [
            dt.datetime(1999, 12, 31, 12, 12, 12, 12000),
            dt.datetime(2012, 12, 21, 12, 21, 12, 21000),
            dt.datetime(1776, 7, 4, 7, 4, 7, 4000),
        ]
        # 创建原始数据框架对象
        original = DataFrame(
            {
                "nums": [1.0, 2.0, 3.0],
                "strs": ["apple", "banana", "cherry"],
                "dates": dates,
            }
        )

        # 复制原始数据框架对象以备后用
        expected = original[:]
        # 将"dates"列转换为毫秒精度的日期时间类型
        expected["dates"] = expected["dates"].astype("M8[ms]")

        # 获取临时文件路径
        path = temp_file
        # 将原始数据框架对象写入 Stata 格式文件，不包含索引
        original.to_stata(path, write_index=False)
        # 从 Stata 文件重新读取数据，执行日期转换
        reread = read_stata(path, convert_dates=True)
        # 断言重新读取的数据与预期数据框架对象相等
        tm.assert_frame_equal(expected, reread)

        # 将原始数据框架对象写入 Stata 格式文件，指定日期转换为"tc"
        original.to_stata(path, write_index=False, convert_dates={"dates": "tc"})
        # 从 Stata 文件重新读取数据，执行日期转换
        direct = read_stata(path, convert_dates=True)
        # 断言重新读取的数据与直接读取的数据相等
        tm.assert_frame_equal(reread, direct)

        # 获取"dates"列的索引值
        dates_idx = original.columns.tolist().index("dates")
        # 将原始数据框架对象写入 Stata 格式文件，指定索引值日期转换为"tc"
        original.to_stata(path, write_index=False, convert_dates={dates_idx: "tc"})
        # 从 Stata 文件重新读取数据，执行日期转换
        direct = read_stata(path, convert_dates=True)
        # 断言重新读取的数据与直接读取的数据相等
        tm.assert_frame_equal(reread, direct)

    # 定义测试方法，用于不支持的数据类型测试
    def test_unsupported_type(self, temp_file):
        # 创建包含复数类型的原始数据框架对象
        original = DataFrame({"a": [1 + 2j, 2 + 4j]})

        # 期望引发未实现错误，匹配错误消息为"Data type complex128 not supported"
        msg = "Data type complex128 not supported"
        with pytest.raises(NotImplementedError, match=msg):
            # 获取临时文件路径
            path = temp_file
            # 将原始数据框架对象写入 Stata 格式文件
            original.to_stata(path)

    # 定义测试方法，用于不支持的日期类型测试
    def test_unsupported_datetype(self, temp_file):
        # 创建包含几个日期时间对象的列表
        dates = [
            dt.datetime(1999, 12, 31, 12, 12, 12, 12000),
            dt.datetime(2012, 12, 21, 12, 21, 12, 21000),
            dt.datetime(1776, 7, 4, 7, 4, 7, 4000),
        ]
        # 创建原始数据框架对象
        original = DataFrame(
            {
                "nums": [1.0, 2.0, 3.0],
                "strs": ["apple", "banana", "cherry"],
                "dates": dates,
            }
        )

        # 期望引发未实现错误，匹配错误消息为"Format %tC not implemented"
        msg = "Format %tC not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            # 获取临时文件路径
            path = temp_file
            # 将原始数据框架对象写入 Stata 格式文件，指定日期转换格式为"tC"
            original.to_stata(path, convert_dates={"dates": "tC"})

        # 创建包含日期范围的 Pandas 日期时间对象
        dates = pd.date_range("1-1-1990", periods=3, tz="Asia/Hong_Kong")
        # 创建原始数据框架对象
        original = DataFrame(
            {
                "nums": [1.0, 2.0, 3.0],
                "strs": ["apple", "banana", "cherry"],
                "dates": dates,
            }
        )
        # 期望引发未实现错误，匹配错误消息为"Data type datetime64"
        with pytest.raises(NotImplementedError, match="Data type datetime64"):
            # 获取临时文件路径
            path = temp_file
            # 将原始数据框架对象写入 Stata 格式文件
            original.to_stata(path)

    # 定义测试方法，用于重复列标签的测试
    def test_repeated_column_labels(self, datapath):
        # GH 13923, 25772
        # 指定测试的错误消息
        msg = """
        with pytest.raises(ValueError, match=msg):
            # 使用 pytest 框架检测 ValueError 异常，并匹配特定的错误消息
            read_stata(
                datapath("io", "data", "stata", "stata15.dta"),
                convert_categoricals=True,
            )
    
    def test_stata_111(self, datapath):
        # 111 是一个旧版本，但目前的 SAS 版本在导出到 Stata 格式时仍在使用。我们没有找到此版本的在线文档。
        # 从指定路径读取 Stata 文件，将其转换为 DataFrame
        df = read_stata(datapath("io", "data", "stata", "stata7_111.dta"))
        # 创建原始 DataFrame
        original = DataFrame(
            {
                "y": [1, 1, 1, 1, 1, 0, 0, np.nan, 0, 0],
                "x": [1, 2, 1, 3, np.nan, 4, 3, 5, 1, 6],
                "w": [2, np.nan, 5, 2, 4, 4, 3, 1, 2, 3],
                "z": ["a", "b", "c", "d", "e", "", "g", "h", "i", "j"],
            }
        )
        # 选择特定列形成原始数据的子集
        original = original[["y", "x", "w", "z"]]
        # 使用测试框架验证读取的 DataFrame 与原始 DataFrame 相等
        tm.assert_frame_equal(original, df)

    def test_out_of_range_double(self, temp_file):
        # GH 14618
        # 创建包含浮点数列的 DataFrame，其中一个列的最大值超出 Stata 支持的范围
        df = DataFrame(
            {
                "ColumnOk": [0.0, np.finfo(np.double).eps, 4.49423283715579e307],
                "ColumnTooBig": [0.0, np.finfo(np.double).eps, np.finfo(np.double).max],
            }
        )
        # 定义错误消息的正则表达式
        msg = (
            r"Column ColumnTooBig has a maximum value \(.+\) outside the range "
            r"supported by Stata \(.+\)"
        )
        # 使用 pytest 框架检测 ValueError 异常，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            path = temp_file
            # 将 DataFrame 写入 Stata 格式的文件
            df.to_stata(path)

    def test_out_of_range_float(self, temp_file):
        # 创建包含浮点数列的原始 DataFrame
        original = DataFrame(
            {
                "ColumnOk": [
                    0.0,
                    np.finfo(np.float32).eps,
                    np.finfo(np.float32).max / 10.0,
                ],
                "ColumnTooBig": [
                    0.0,
                    np.finfo(np.float32).eps,
                    np.finfo(np.float32).max,
                ],
            }
        )
        # 设置索引名称
        original.index.name = "index"
        # 将所有列转换为 np.float32 类型
        for col in original:
            original[col] = original[col].astype(np.float32)

        # 将原始 DataFrame 写入 Stata 格式的临时文件
        path = temp_file
        original.to_stata(path)
        # 重新读取 Stata 文件并转换为 DataFrame
        reread = read_stata(path)

        # 将特定列转换为 np.float64 类型
        original["ColumnTooBig"] = original["ColumnTooBig"].astype(np.float64)
        # 预期的 DataFrame 结果
        expected = original
        # 使用测试框架验证读取的 DataFrame 与预期的 DataFrame 相等
        tm.assert_frame_equal(reread.set_index("index"), expected)

    @pytest.mark.parametrize("infval", [np.inf, -np.inf])
    # 测试函数，用于测试处理包含无穷大值的情况
    def test_inf(self, infval, temp_file):
        # GH 45350
        # 创建一个包含两列的DataFrame，其中一列包含0.0和1.0，另一列包含2.0和infval（无穷大值）
        df = DataFrame({"WithoutInf": [0.0, 1.0], "WithInf": [2.0, infval]})
        # 定义错误消息，用于捕获 ValueError 异常，匹配包含无穷大或负无穷大的列名
        msg = (
            "Column WithInf contains infinity or -infinity"
            "which is outside the range supported by Stata."
        )
        # 使用 pytest 来检测是否会抛出 ValueError 异常，并且异常消息与预期的 msg 匹配
        with pytest.raises(ValueError, match=msg):
            # 生成临时文件路径
            path = temp_file
            # 将DataFrame写入 Stata 格式的文件
            df.to_stata(path)

    # 测试函数，测试使用 pathlib 作为路径参数的情况
    def test_path_pathlib(self):
        # 创建一个DataFrame，包含特定的数据和列名、行索引名
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=pd.Index(list("ABCD"), dtype=object),
            index=pd.Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        df.index.name = "index"
        # 定义一个 lambda 函数，用于读取 Stata 格式文件并设置索引为 "index"
        reader = lambda x: read_stata(x).set_index("index")
        # 执行路径相关的 round-trip 测试，即写入再读取，比较结果
        result = tm.round_trip_pathlib(df.to_stata, reader)
        tm.assert_frame_equal(df, result)

    # 测试函数，测试写入 Stata 文件时使用写索引选项的情况
    @pytest.mark.parametrize("write_index", [True, False])
    def test_value_labels_iterator(self, write_index, temp_file):
        # GH 16923
        # 创建一个简单的 DataFrame
        d = {"A": ["B", "E", "C", "A", "E"]}
        df = DataFrame(data=d)
        # 将列 'A' 转换为分类类型
        df["A"] = df["A"].astype("category")
        # 生成临时文件路径
        path = temp_file
        # 将DataFrame写入 Stata 格式的文件，指定是否写入索引
        df.to_stata(path, write_index=write_index)

        # 使用 StataReader 打开文件，以迭代器模式读取数据
        with read_stata(path, iterator=True) as dta_iter:
            # 获取值标签信息
            value_labels = dta_iter.value_labels()
        # 断言值标签是否符合预期
        assert value_labels == {"A": {0: "A", 1: "B", 2: "C", 3: "E"}}

    # 测试函数，测试设置索引的情况
    def test_set_index(self, temp_file):
        # GH 17328
        # 创建一个DataFrame
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=pd.Index(list("ABCD"), dtype=object),
            index=pd.Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        df.index.name = "index"
        # 生成临时文件路径
        path = temp_file
        # 将DataFrame写入 Stata 格式的文件
        df.to_stata(path)
        # 重新读取并设置索引为 'index'，用于后续比较
        reread = read_stata(path, index_col="index")
        tm.assert_frame_equal(df, reread)

    # 测试函数，测试日期解析时忽略格式细节的情况
    @pytest.mark.parametrize(
        "column", ["ms", "day", "week", "month", "qtr", "half", "yr"]
    )
    def test_date_parsing_ignores_format_details(self, column, datapath):
        # GH 17797
        #
        # 测试在确定数字列是否为日期值时忽略显示格式的情况。
        #
        # 所有日期类型都以数字存储，与列关联的格式同时表示日期类型和显示格式。
        #
        # STATA 支持 9 种日期类型，每种类型都有不同的单位。我们测试其中的 7 种类型，忽略 %tC 和 %tb。
        # %tC 是 %tc 的一种变体，考虑了闰秒；%tb 依赖于 STATA 的工作日历。
        df = read_stata(datapath("io", "data", "stata", "stata13_dates.dta"))
        # 获取未格式化和格式化的特定列值
        unformatted = df.loc[0, column]
        formatted = df.loc[0, column + "_fmt"]
        # 断言未格式化的值与格式化的值相等
        assert unformatted == formatted

    # 测试函数，测试字节顺序的参数化情况
    @pytest.mark.parametrize("byteorder", ["little", "big"])
    # 定义测试方法，用于测试写入 Stata 格式的数据文件
    def test_writer_117(self, byteorder, temp_file):
        # 创建包含数据的原始 DataFrame
        original = DataFrame(
            data=[
                [
                    "string",                # 第一列数据类型为字符串
                    "object",                # 第二列数据类型为对象
                    1,                       # 第三列数据类型为 int8
                    1,                       # 第四列数据类型为 int16
                    1,                       # 第五列数据类型为 int32
                    1.1,                     # 第六列数据类型为 float32
                    1.1,                     # 第七列数据类型为 float64
                    np.datetime64("2003-12-25"),  # 第八列数据类型为日期时间
                    "a",                     # 第九列数据类型为字符串
                    "a" * 2045,              # 第十列数据类型为字符串，长度为 2045
                    "a" * 5000,              # 第十一列数据类型为字符串，长度为 5000
                    "a",                     # 第十二列数据类型为字符串
                ],
                [
                    "string-1",              # 第一列数据类型为字符串
                    "object-1",              # 第二列数据类型为对象
                    1,                       # 第三列数据类型为 int8
                    1,                       # 第四列数据类型为 int16
                    1,                       # 第五列数据类型为 int32
                    1.1,                     # 第六列数据类型为 float32
                    1.1,                     # 第七列数据类型为 float64
                    np.datetime64("2003-12-26"),  # 第八列数据类型为日期时间
                    "b",                     # 第九列数据类型为字符串
                    "b" * 2045,              # 第十列数据类型为字符串，长度为 2045
                    "",                      # 第十一列数据类型为空字符串
                    "",                      # 第十二列数据类型为空字符串
                ],
            ],
            columns=[
                "string",                # 设置列名为 string
                "object",                # 设置列名为 object
                "int8",                  # 设置列名为 int8
                "int16",                 # 设置列名为 int16
                "int32",                 # 设置列名为 int32
                "float32",               # 设置列名为 float32
                "float64",               # 设置列名为 float64
                "datetime",              # 设置列名为 datetime
                "s1",                    # 设置列名为 s1
                "s2045",                 # 设置列名为 s2045
                "srtl",                  # 设置列名为 srtl
                "forced_strl",           # 设置列名为 forced_strl
            ],
        )
        # 将 object 列数据转换为对象类型
        original["object"] = Series(original["object"], dtype=object)
        # 将 int8 列数据转换为 np.int8 类型
        original["int8"] = Series(original["int8"], dtype=np.int8)
        # 将 int16 列数据转换为 np.int16 类型
        original["int16"] = Series(original["int16"], dtype=np.int16)
        # 将 int32 列数据转换为 np.int32 类型
        original["int32"] = original["int32"].astype(np.int32)
        # 将 float32 列数据转换为 np.float32 类型
        original["float32"] = Series(original["float32"], dtype=np.float32)
        # 设置索引名称为 "index"
        original.index.name = "index"
        # 复制原始 DataFrame
        copy = original.copy()
        # 将数据写入临时文件 path，格式为 Stata，指定日期类型转换为 "tc"，指定字节顺序为 byteorder，指定强制转换为字符串长度的列为 "forced_strl"，版本号为 117
        original.to_stata(
            path,
            convert_dates={"datetime": "tc"},
            byteorder=byteorder,
            convert_strl=["forced_strl"],
            version=117,
        )
        # 读取并再次验证写入的数据
        written_and_read_again = self.read_dta(path)

        # 准备预期的数据
        expected = original[:]
        # 将 datetime 列数据转换为 "M8[ms]" 类型
        expected["datetime"] = expected["datetime"].astype("M8[ms]")

        # 断言写入和再次读取的数据与预期数据相等，设置索引为 "index"
        tm.assert_frame_equal(
            written_and_read_again.set_index("index"),
            expected,
        )
        # 断言原始数据与复制的数据相等
        tm.assert_frame_equal(original, copy)

    # 定义测试方法，用于测试转换字符串长度参数名称交换的情况
    def test_convert_strl_name_swap(self, temp_file):
        # 创建包含数据的原始 DataFrame
        original = DataFrame(
            [["a" * 3000, "A", "apple"], ["b" * 1000, "B", "banana"]],
            columns=["long1" * 10, "long", 1],
        )
        # 设置索引名称为 "index"
        original.index.name = "index"

        # 准备警告信息
        msg = "Not all pandas column names were valid Stata variable names"
        # 验证是否产生预期的警告信息 InvalidColumnName，且内容匹配 msg
        with tm.assert_produces_warning(InvalidColumnName, match=msg):
            # 将数据写入临时文件 path，指定转换字符串长度的列为 ["long", 1]，版本号为 117
            path = temp_file
            original.to_stata(path, convert_strl=["long", 1], version=117)
            # 重新读取已写入的数据
            reread = self.read_dta(path)
            # 设置索引为 "index"
            reread = reread.set_index("index")
            # 设置重新读取的列名称与原始数据相同
            reread.columns = original.columns
            # 断言重新读取的数据与原始数据相等，不检查索引类型
            tm.assert_frame_equal(reread, original, check_index_type=False)
    def test_invalid_date_conversion(self, temp_file):
        # GH 12259
        # 定义日期列表
        dates = [
            dt.datetime(1999, 12, 31, 12, 12, 12, 12000),
            dt.datetime(2012, 12, 21, 12, 21, 12, 21000),
            dt.datetime(1776, 7, 4, 7, 4, 7, 4000),
        ]
        # 创建包含数值、字符串和日期列的 DataFrame 对象
        original = DataFrame(
            {
                "nums": [1.0, 2.0, 3.0],
                "strs": ["apple", "banana", "cherry"],
                "dates": dates,
            }
        )

        # 设置临时文件路径
        path = temp_file
        # 设置异常消息
        msg = "convert_dates key must be a column or an integer"
        # 断言抛出 ValueError 异常，并验证异常消息
        with pytest.raises(ValueError, match=msg):
            original.to_stata(path, convert_dates={"wrong_name": "tc"})

    @pytest.mark.parametrize("version", [114, 117, 118, 119, None])
    def test_nonfile_writing(self, version, temp_file):
        # GH 21041
        # 创建 BytesIO 对象
        bio = io.BytesIO()
        # 创建 DataFrame 对象
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=pd.Index(list("ABCD"), dtype=object),
            index=pd.Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        df.index.name = "index"
        # 设置临时文件路径
        path = temp_file
        # 将 DataFrame 对象写入到 Stata 格式的 BytesIO 对象中
        df.to_stata(bio, version=version)
        # 将写入位置移动到起始位置
        bio.seek(0)
        # 打开临时文件，并将 BytesIO 对象的内容写入其中
        with open(path, "wb") as dta:
            dta.write(bio.read())
        # 重新读取写入的 Stata 文件为 DataFrame 对象
        reread = read_stata(path, index_col="index")
        # 断言原始 DataFrame 和重新读取的 DataFrame 相等
        tm.assert_frame_equal(df, reread)

    def test_gzip_writing(self, temp_file):
        # writing version 117 requires seek and cannot be used with gzip
        # 创建 DataFrame 对象
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=pd.Index(list("ABCD"), dtype=object),
            index=pd.Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        df.index.name = "index"
        # 设置临时文件路径
        path = temp_file
        # 使用 GzipFile 将 DataFrame 对象写入到 gzip 压缩文件中
        with gzip.GzipFile(path, "wb") as gz:
            df.to_stata(gz, version=114)
        # 从 gzip 压缩文件中读取数据，并转换为 DataFrame 对象
        with gzip.GzipFile(path, "rb") as gz:
            reread = read_stata(gz, index_col="index")
        # 断言原始 DataFrame 和重新读取的 DataFrame 相等
        tm.assert_frame_equal(df, reread)

    # 117 is not included in this list as it uses ASCII strings
    @pytest.mark.parametrize(
        "file",
        [
            "stata16_118",
            "stata16_be_118",
            "stata16_119",
            "stata16_be_119",
        ],
    )
    def test_unicode_dta_118_119(self, file, datapath):
        # 从指定路径读取 Stata 文件，并转换为 DataFrame 对象
        unicode_df = self.read_dta(datapath("io", "data", "stata", f"{file}.dta"))

        # 定义列名和期望值列表
        columns = ["utf8", "latin1", "ascii", "utf8_strl", "ascii_strl"]
        values = [
            ["ραηδας", "PÄNDÄS", "p", "ραηδας", "p"],
            ["ƤĀńĐąŜ", "Ö", "a", "ƤĀńĐąŜ", "a"],
            ["ᴘᴀᴎᴅᴀS", "Ü", "n", "ᴘᴀᴎᴅᴀS", "n"],
            ["      ", "      ", "d", "      ", "d"],
            [" ", "", "a", " ", "a"],
            ["", "", "s", "", "s"],
            ["", "", " ", "", " "],
        ]
        # 创建期望的 DataFrame 对象
        expected = DataFrame(values, columns=columns)

        # 断言读取的 Unicode DataFrame 和期望的 DataFrame 相等
        tm.assert_frame_equal(unicode_df, expected)
    # 测试函数，用于测试包含混合字符串和数字的输出
    def test_mixed_string_strl(self, temp_file):
        # GH 23633
        # 创建一个包含混合字符串和数字的输出列表
        output = [{"mixed": "string" * 500, "number": 0}, {"mixed": None, "number": 1}]
        # 转换输出为DataFrame对象
        output = DataFrame(output)
        # 将DataFrame中的number列转换为int32类型
        output.number = output.number.astype("int32")

        # 将DataFrame写入 Stata 文件
        path = temp_file
        output.to_stata(path, write_index=False, version=117)
        # 从 Stata 文件中重新读取数据
        reread = read_stata(path)
        # 生成期望的DataFrame，将NaN值填充为空字符串
        expected = output.fillna("")
        # 断言重新读取的数据与期望的数据一致
        tm.assert_frame_equal(reread, expected)

        # 检查 strl 是否支持全部为 None 的情况
        output["mixed"] = None
        output.to_stata(path, write_index=False, convert_strl=["mixed"], version=117)
        reread = read_stata(path)
        expected = output.fillna("")
        tm.assert_frame_equal(reread, expected)

    @pytest.mark.parametrize("version", [114, 117, 118, 119, None])
    # 测试函数，测试当列全部为 None 时是否会抛出异常
    def test_all_none_exception(self, version, temp_file):
        output = [{"none": "none", "number": 0}, {"none": None, "number": 1}]
        output = DataFrame(output)
        output["none"] = None
        # 使用 pytest 检查是否会抛出 ValueError 异常，并匹配特定错误消息
        with pytest.raises(ValueError, match="Column `none` cannot be exported"):
            output.to_stata(temp_file, version=version)

    @pytest.mark.parametrize("version", [114, 117, 118, 119, None])
    # 测试函数，测试当文件中存在无效字符时是否会抛出 UnicodeEncodeError 异常
    def test_invalid_file_not_written(self, version, temp_file):
        content = "Here is one __�__ Another one __·__ Another one __½__"
        df = DataFrame([content], columns=["invalid"])
        # 定义两个可能的错误消息
        msg1 = (
            r"'latin-1' codec can't encode character '\\ufffd' "
            r"in position 14: ordinal not in range\(256\)"
        )
        msg2 = (
            "'ascii' codec can't decode byte 0xef in position 14: "
            r"ordinal not in range\(128\)"
        )
        # 使用 pytest 检查是否会抛出 UnicodeEncodeError 异常，并匹配其中任意一条错误消息
        with pytest.raises(UnicodeEncodeError, match=f"{msg1}|{msg2}"):
            df.to_stata(temp_file)

    # 测试函数，测试在使用 Latin-1 编码时是否能正确处理 strl
    def test_strl_latin1(self, temp_file):
        # GH 23573, correct GSO data to reflect correct size
        # 创建包含两行数据的DataFrame，包括 Latin-1 编码的字符
        output = DataFrame(
            [["pandas"] * 2, ["þâÑÐÅ§"] * 2], columns=["var_str", "var_strl"]
        )

        # 将DataFrame写入 Stata 文件，同时转换 var_strl 列为 strl 格式
        output.to_stata(temp_file, version=117, convert_strl=["var_strl"])
        # 打开并读取 Stata 文件内容
        with open(temp_file, "rb") as reread:
            content = reread.read()
            expected = "þâÑÐÅ§"
            # 断言 Latin-1 编码的字符在文件内容中存在
            assert expected.encode("latin-1") in content
            assert expected.encode("utf-8") in content
            # 解析文件中的 GSO 数据段，验证其大小与预期一致
            gsos = content.split(b"strls")[1][1:-2]
            for gso in gsos.split(b"GSO")[1:]:
                val = gso.split(b"\x00")[-2]
                size = gso[gso.find(b"\x82") + 1]
                assert len(val) == size - 1

    # 测试函数，测试在使用 Latin-1 编码时是否能正确处理
    def test_encoding_latin1_118(self, datapath):
        # GH 25960
        # 定义一个多行的消息字符串
        msg = """
        # 使用 Latin-1 编码来解码 dta 文件中无法用 utf-8 解码的一个或多个字符串。
        # 这种情况可能发生在 Stata 或其他软件不正确编码文件时。应确保返回的字符串值是正确的。
        path = datapath("io", "data", "stata", "stata1_encoding_118.dta")
        # 使用 tm.assert_produces_warning 函数来验证 read_stata 函数是否会产生 UnicodeWarning 警告，
        # 并设置 filter_level="once"，确保只检查到一次警告。
        with tm.assert_produces_warning(UnicodeWarning, filter_level="once") as w:
            # 调用 read_stata 函数读取指定路径的数据文件
            encoded = read_stata(path)
            # 断言只有一条警告产生
            assert len(w) == 1
            # 断言警告的消息内容与预期的 msg 变量相符
            assert w[0].message.args[0] == msg

        # 创建一个预期的 DataFrame，包含 151 个 "Düsseldorf" 字符串，列名为 "kreis1849"
        expected = DataFrame([["Düsseldorf"]] * 151, columns=["kreis1849"])
        # 使用 tm.assert_frame_equal 函数验证 encoded 和 expected 是否相等
        tm.assert_frame_equal(encoded, expected)

    @pytest.mark.slow
    def test_stata_119(self, datapath):
        # 由于包含 32,999 个变量，未压缩时大小为 20MiB，因此此文件经过 gzip 压缩
        # 仅验证读取器是否正确报告变量的数量，以避免内存使用过高的峰值
        with gzip.open(
            datapath("io", "data", "stata", "stata1_119.dta.gz"), "rb"
        ) as gz:
            # 使用 StataReader 类读取 gzip 压缩的数据文件
            with StataReader(gz) as reader:
                # 确保读取器已打开
                reader._ensure_open()
                # 断言读取器的变量数量 _nvar 是否等于 32999
                assert reader._nvar == 32999

    @pytest.mark.parametrize("version", [118, 119, None])
    @pytest.mark.parametrize("byteorder", ["little", "big"])
    # 定义一个测试方法，测试 StataWriterUTF8 类的功能，检验 UTF-8 编码的写入
    def test_utf8_writer(self, version, byteorder, temp_file):
        # 创建一个有序的分类数据，包含三个元素："a", "β", "ĉ"
        cat = pd.Categorical(["a", "β", "ĉ"], ordered=True)
        
        # 创建一个 DataFrame 包含四列数据，其中包括不同类型的数据
        data = DataFrame(
            [
                [1.0, 1, "ᴬ", "ᴀ relatively long ŝtring"],
                [2.0, 2, "ᴮ", ""],
                [3.0, 3, "ᴰ", None],
            ],
            columns=["Å", "β", "ĉ", "strls"],
        )
        
        # 向 DataFrame 添加一个新列，列名为 "ᴐᴬᵀ"，数据类型为分类数据 cat
        data["ᴐᴬᵀ"] = cat
        
        # 创建一个变量标签的字典，将列名映射到描述性标签
        variable_labels = {
            "Å": "apple",
            "β": "ᵈᵉᵊ",
            "ĉ": "ᴎტჄႲႳႴႶႺ",
            "strls": "Long Strings",
            "ᴐᴬᵀ": "",
        }
        
        # 设置数据的标签为 "ᴅaᵀa-label"
        data_label = "ᴅaᵀa-label"
        
        # 设置值标签的字典，对列名为 "β" 的数据进行值标签的映射
        value_labels = {"β": {1: "label", 2: "æøå", 3: "ŋot valid latin-1"}}
        
        # 将列名为 "β" 的数据类型转换为整数型，并创建一个 StataWriterUTF8 对象
        writer = StataWriterUTF8(
            temp_file,
            data,
            data_label=data_label,
            convert_strl=["strls"],
            variable_labels=variable_labels,
            write_index=False,
            byteorder=byteorder,
            version=version,
            value_labels=value_labels,
        )
        
        # 调用写文件方法
        writer.write_file()
        
        # 重新读取并编码数据
        reread_encoded = read_stata(temp_file)
        
        # 将 DataFrame 中的空值替换为空字符串
        data["strls"] = data["strls"].fillna("")
        
        # 将具有值标签的变量重新读取为分类数据类型
        data["β"] = (
            data["β"].replace(value_labels["β"]).astype("category").cat.as_ordered()
        )
        
        # 断言 DataFrame 和重新读取的编码数据相等
        tm.assert_frame_equal(data, reread_encoded)
        
        # 使用 StataReader 读取临时文件
        with StataReader(temp_file) as reader:
            # 断言数据标签与预期的数据标签相等
            assert reader.data_label == data_label
            # 断言变量标签与预期的变量标签字典相等
            assert reader.variable_labels() == variable_labels
        
        # 将数据写入 Stata 文件并重新读取
        data.to_stata(temp_file, version=version, write_index=False)
        reread_to_stata = read_stata(temp_file)
        
        # 断言 DataFrame 和重新读取的 Stata 数据相等
        tm.assert_frame_equal(data, reread_to_stata)
    
    # 定义一个测试方法，测试 StataWriterUTF8 类在特定异常情况下的行为
    def test_writer_118_exceptions(self, temp_file):
        # 创建一个包含一行数据的 DataFrame，列数为 33000，数据类型为 int8
        df = DataFrame(np.zeros((1, 33000), dtype=np.int8))
        
        # 使用 pytest 检测是否抛出 ValueError 异常，异常消息包含 "version must be either 118 or 119."
        with pytest.raises(ValueError, match="version must be either 118 or 119."):
            StataWriterUTF8(temp_file, df, version=117)
        
        # 使用 pytest 检测是否抛出 ValueError 异常，异常消息包含 "You must use version 119"
        with pytest.raises(ValueError, match="You must use version 119"):
            StataWriterUTF8(temp_file, df, version=118)
    
    # 使用 pytest 参数化装饰器，定义一个参数化测试方法，测试不同的 dtype_backend 参数
    @pytest.mark.parametrize(
        "dtype_backend",
        ["numpy_nullable", pytest.param("pyarrow", marks=td.skip_if_no("pyarrow"))],
    )
    # 定义一个测试方法，用于测试读写不同数据类型的数据框
    def test_read_write_ea_dtypes(self, dtype_backend, temp_file, tmp_path):
        # 创建一个包含不同数据类型的数据框
        df = DataFrame(
            {
                "a": [1, 2, None],  # 整数列，包含空值
                "b": ["a", "b", "c"],  # 字符串列
                "c": [True, False, None],  # 布尔列，包含空值
                "d": [1.5, 2.5, 3.5],  # 浮点数列
                "e": pd.date_range("2020-12-31", periods=3, freq="D"),  # 日期时间列
            },
            index=pd.Index([0, 1, 2], name="index"),  # 指定索引
        )
        # 将数据框转换为指定的数据类型后的版本
        df = df.convert_dtypes(dtype_backend=dtype_backend)
        # 创建 Stata 文件的路径
        stata_path = tmp_path / "test_stata.dta"
        # 将数据框保存为 Stata 文件
        df.to_stata(stata_path, version=118)

        # 将数据框保存到临时文件
        df.to_stata(temp_file)
        # 从临时文件中读取并返回读取后的数据框
        written_and_read_again = self.read_dta(temp_file)

        # 创建预期的数据框，包含与 Stata 文件存储格式相对应的数据
        expected = DataFrame(
            {
                "a": [1, 2, np.nan],  # 整数列，包含 NaN
                "b": ["a", "b", "c"],  # 字符串列
                "c": [1.0, 0, np.nan],  # 布尔列，包含 NaN
                "d": [1.5, 2.5, 3.5],  # 浮点数列
                # Stata 存储单位为毫秒，因此日期时间列的精度不会完全保留
                "e": pd.date_range("2020-12-31", periods=3, freq="D", unit="ms"),
            },
            index=pd.RangeIndex(range(3), name="index"),  # 指定索引
        )

        # 使用测试框架断言函数，比较读取后的数据框和预期的数据框是否相等
        tm.assert_frame_equal(written_and_read_again.set_index("index"), expected)
@pytest.mark.parametrize("version", [114, 117, 118, 119, None])
# 使用 pytest 的参数化装饰器，用来多次运行此测试函数，测试不同的版本和参数组合
@pytest.mark.parametrize("use_dict", [True, False])
# 使用 pytest 的参数化装饰器，用来多次运行此测试函数，测试不同的布尔值参数组合
@pytest.mark.parametrize("infer", [True, False])
# 使用 pytest 的参数化装饰器，用来多次运行此测试函数，测试不同的布尔值参数组合
def test_compression(
    # 从模块中导入多个变量或函数，包括 compression、version、use_dict、infer、compression_to_extension 和 tmp_path
    compression, version, use_dict, infer, compression_to_extension, tmp_path
# 定义一个测试函数，用于测试读取压缩数据集后是否与预期一致
def test_compression(tmp_path):
    # 设置默认的文件名
    file_name = "dta_inferred_compression.dta"
    
    # 如果 compression 参数不为空
    if compression:
        # 如果 use_dict 为真，直接使用 compression 作为文件扩展名
        if use_dict:
            file_ext = compression
        # 否则根据 compression_to_extension 字典获取对应的文件扩展名
        else:
            file_ext = compression_to_extension[compression]
        # 更新文件名以包含压缩类型的扩展名
        file_name += f".{file_ext}"
    
    # 设置 compression_arg 为 compression 参数的值
    compression_arg = compression
    
    # 如果 infer 为真，compression_arg 更新为 "infer"
    if infer:
        compression_arg = "infer"
    
    # 如果 use_dict 为真，compression_arg 更新为包含 compression 方法的字典
    if use_dict:
        compression_arg = {"method": compression}
    
    # 创建一个 DataFrame，包含随机生成的数据，2列10行
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 2)), columns=list("AB")
    )
    
    # 设置 DataFrame 的索引名为 "index"
    df.index.name = "index"
    
    # 设置路径为临时路径加上文件名
    path = tmp_path / file_name
    
    # 创建文件路径
    path.touch()
    
    # 将 DataFrame 写入到 Stata 格式文件中，指定版本和压缩参数
    df.to_stata(path, version=version, compression=compression_arg)
    
    # 如果压缩类型为 "gzip"
    if compression == "gzip":
        # 使用 gzip 打开路径并读取内容，将结果写入到 BytesIO 对象中
        with gzip.open(path, "rb") as comp:
            fp = io.BytesIO(comp.read())
    
    # 如果压缩类型为 "zip"
    elif compression == "zip":
        # 使用 zipfile 打开路径，读取第一个文件，并将结果写入到 BytesIO 对象中
        with zipfile.ZipFile(path, "r") as comp:
            fp = io.BytesIO(comp.read(comp.filelist[0]))
    
    # 如果压缩类型为 "tar"
    elif compression == "tar":
        # 使用 tarfile 打开路径，读取第一个文件，并将结果写入到 BytesIO 对象中
        with tarfile.open(path) as tar:
            fp = io.BytesIO(tar.extractfile(tar.getnames()[0]).read())
    
    # 如果压缩类型为 "bz2"
    elif compression == "bz2":
        # 使用 bz2 打开路径并读取内容，将结果写入到 BytesIO 对象中
        with bz2.open(path, "rb") as comp:
            fp = io.BytesIO(comp.read())
    
    # 如果压缩类型为 "zstd"
    elif compression == "zstd":
        # 导入 zstandard 模块，并使用 zstd 打开路径，读取内容并将结果写入到 BytesIO 对象中
        zstd = pytest.importorskip("zstandard")
        with zstd.open(path, "rb") as comp:
            fp = io.BytesIO(comp.read())
    
    # 如果压缩类型为 "xz"
    elif compression == "xz":
        # 导入 lzma 模块，并使用 lzma 打开路径，读取内容并将结果写入到 BytesIO 对象中
        lzma = pytest.importorskip("lzma")
        with lzma.open(path, "rb") as comp:
            fp = io.BytesIO(comp.read())
    
    # 如果 compression 为空，直接将路径赋给 fp
    elif compression is None:
        fp = path
    
    # 调用 read_stata 函数读取 fp 中的数据，指定索引列为 "index"
    reread = read_stata(fp, index_col="index")
    
    # 设置期望的 DataFrame 为 df
    expected = df
    
    # 断言 reread 与 expected 相等
    tm.assert_frame_equal(reread, expected)


# 参数化测试函数，method 参数为 "zip" 或 "infer"，file_ext 参数为 None, "dta", "zip" 中的一个
@pytest.mark.parametrize("method", ["zip", "infer"])
@pytest.mark.parametrize("file_ext", [None, "dta", "zip"])
def test_compression_dict(method, file_ext, tmp_path):
    # 设置文件名为 "test" 加上 file_ext
    file_name = f"test.{file_ext}"
    
    # 设置存档名为 "test.dta"
    archive_name = "test.dta"
    
    # 创建一个 DataFrame，包含随机生成的数据，2列10行
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 2)), columns=list("AB")
    )
    
    # 设置 DataFrame 的索引名为 "index"
    df.index.name = "index"
    
    # 设置 compression 参数为包含 method 和 archive_name 的字典
    compression = {"method": method, "archive_name": archive_name}
    
    # 设置路径为临时路径加上文件名
    path = tmp_path / file_name
    
    # 创建文件路径
    path.touch()
    
    # 将 DataFrame 写入到 Stata 格式文件中，指定压缩参数为 compression
    df.to_stata(path, compression=compression)
    
    # 如果 method 为 "zip" 或者 file_ext 为 "zip"
    if method == "zip" or file_ext == "zip":
        # 使用 zipfile 打开路径
        with zipfile.ZipFile(path, "r") as zp:
            # 断言文件列表长度为 1
            assert len(zp.filelist) == 1
            # 断言第一个文件的文件名为 archive_name
            assert zp.filelist[0].filename == archive_name
            # 读取第一个文件的内容并将结果写入到 BytesIO 对象中
            fp = io.BytesIO(zp.read(zp.filelist[0]))
    
    # 否则直接将路径赋给 fp
    else:
        fp = path
    
    # 调用 read_stata 函数读取 fp 中的数据，指定索引列为 "index"
    reread = read_stata(fp, index_col="index")
    
    # 设置期望的 DataFrame 为 df
    expected = df
    
    # 断言 reread 与 expected 相等
    tm.assert_frame_equal(reread, expected)


# 参数化测试函数，version 参数为 114, 117, 118, 119, None 中的一个
@pytest.mark.parametrize("version", [114, 117, 118, 119, None])
def test_chunked_categorical(version, temp_file):
    # 创建一个包含分类数据的 DataFrame
    df = DataFrame({"cats": Series(["a", "b", "a", "b", "c"], dtype="category")})
    
    # 设置 DataFrame 的索引名为 "index"
    df.index.name = "index"
    
    # 设置期望的 DataFrame 为 df 的副本
    expected = df.copy()
    
    # 将 DataFrame 写入到 Stata 格式文件中，指定版本号为 version
    df.to_stata(temp_file, version=version)
    # 使用 StataReader 打开临时文件，并设置每次读取块大小为2，不排序分类变量
    with StataReader(temp_file, chunksize=2, order_categoricals=False) as reader:
        # 遍历 StataReader 返回的每个块，同时获取块的索引号 i
        for i, block in enumerate(reader):
            # 将当前块按索引列 "index" 设置为索引
            block = block.set_index("index")
            # 断言确保块中包含名为 "cats" 的列
            assert "cats" in block
            # 使用 tm.assert_series_equal 检查块中的 "cats" 列与期望值的部分是否相等
            tm.assert_series_equal(
                block.cats,
                expected.cats.iloc[2 * i : 2 * (i + 1)],  # 从期望值中选择相应部分进行比较
                check_index_type=len(block) > 1,  # 根据块的长度确定是否检查索引类型
            )
# 测试部分被部分标记的 Stata 数据文件的块迭代
def test_chunked_categorical_partial(datapath):
    # 获取数据文件的路径
    dta_file = datapath("io", "data", "stata", "stata-dta-partially-labeled.dta")
    # 预期的分类变量值列表
    values = ["a", "b", "a", "b", 3.0]
    # 警告消息
    msg = "series with value labels are not fully labeled"
    
    # 使用 StataReader 对象读取数据文件，每次读取块大小为 2
    with StataReader(dta_file, chunksize=2) as reader:
        # 断言会产生 CategoricalConversionWarning 警告，匹配指定的消息
        with tm.assert_produces_warning(CategoricalConversionWarning, match=msg):
            # 迭代读取的每个块
            for i, block in enumerate(reader):
                # 断言当前块的分类变量与预期值匹配
                assert list(block.cats) == values[2 * i : 2 * (i + 1)]
                # 根据块的索引 i 选择不同的索引对象
                if i < 2:
                    idx = pd.Index(["a", "b"])
                else:
                    idx = pd.Index([3.0], dtype="float64")
                # 断言当前块的分类变量的分类与预期的索引相等
                tm.assert_index_equal(block.cats.cat.categories, idx)
    
    # 再次断言会产生 CategoricalConversionWarning 警告，匹配指定的消息
    with tm.assert_produces_warning(CategoricalConversionWarning, match=msg):
        # 使用 StataReader 对象读取数据文件，每次读取块大小为 5
        with StataReader(dta_file, chunksize=5) as reader:
            # 直接获取大块数据
            large_chunk = reader.__next__()
    
    # 直接读取整个 Stata 数据文件为 DataFrame
    direct = read_stata(dta_file)
    # 断言直接读取的 DataFrame 与大块数据相等
    tm.assert_frame_equal(direct, large_chunk)


# 使用不同的 chunksize 参数进行错误测试
@pytest.mark.parametrize("chunksize", (-1, 0, "apple"))
def test_iterator_errors(datapath, chunksize):
    # 获取数据文件的路径
    dta_file = datapath("io", "data", "stata", "stata-dta-partially-labeled.dta")
    # 断言会产生 ValueError 异常，匹配指定的消息
    with pytest.raises(ValueError, match="chunksize must be a positive"):
        # 使用 StataReader 对象读取数据文件，使用给定的 chunksize 参数
        with StataReader(dta_file, chunksize=chunksize):
            pass


# 测试带有值标签的迭代器
def test_iterator_value_labels(temp_file):
    # 创建带有值标签的 DataFrame
    values = ["c_label", "b_label"] + ["a_label"] * 500
    df = DataFrame({f"col{k}": pd.Categorical(values, ordered=True) for k in range(2)})
    # 将 DataFrame 写入 Stata 格式的临时文件，不包含索引
    df.to_stata(temp_file, write_index=False)
    # 预期的分类变量的索引
    expected = pd.Index(["a_label", "b_label", "c_label"], dtype="object")
    
    # 使用 read_stata 函数读取临时文件，每次读取块大小为 100
    with read_stata(temp_file, chunksize=100) as reader:
        # 迭代读取的每个块
        for j, chunk in enumerate(reader):
            # 断言当前块中每列的分类变量的分类与预期索引相等
            for i in range(2):
                tm.assert_index_equal(chunk.dtypes.iloc[i].categories, expected)
            # 断言当前块与原始 DataFrame 的部分相等
            tm.assert_frame_equal(chunk, df.iloc[j * 100 : (j + 1) * 100])


# 测试精度损失的情况
def test_precision_loss(temp_file):
    # 创建包含大整数的 DataFrame
    df = DataFrame(
        [[sum(2**i for i in range(60)), sum(2**i for i in range(52))]],
        columns=["big", "little"],
    )
    # 断言会产生 PossiblePrecisionLoss 警告，匹配指定的消息
    with tm.assert_produces_warning(
        PossiblePrecisionLoss, match="Column converted from int64 to float64"
    ):
        # 将 DataFrame 写入 Stata 格式的临时文件，不包含索引
        df.to_stata(temp_file, write_index=False)
    
    # 重新读取临时文件为 DataFrame
    reread = read_stata(temp_file)
    # 预期的列数据类型
    expected_dt = Series([np.float64, np.float64], index=["big", "little"])
    # 断言重新读取的 DataFrame 的列数据类型与预期相等
    tm.assert_series_equal(reread.dtypes, expected_dt)
    # 断言重新读取的 DataFrame 中指定位置的值与原始 DataFrame 相等
    assert reread.loc[0, "little"] == df.loc[0, "little"]
    assert reread.loc[0, "big"] == float(df.loc[0, "big"])


# 测试压缩往返
def test_compression_roundtrip(compression, temp_file):
    # 创建包含浮点数的 DataFrame
    df = DataFrame(
        [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
        index=["A", "B"],
        columns=["X", "Y", "Z"],
    )
    # 设置索引名称
    df.index.name = "index"
    
    # 将 DataFrame 写入 Stata 格式的临时文件，使用指定的压缩方式
    df.to_stata(temp_file, compression=compression)
    # 使用 read_stata 函数读取临时文件，指定索引列名称
    reread = read_stata(temp_file, compression=compression, index_col="index")
    # 断言读取的 DataFrame 与原始 DataFrame 相等
    tm.assert_frame_equal(df, reread)
    # 明确确保文件已经被压缩。
    with tm.decompress_file(temp_file, compression) as fh:
        # 将文件内容读取到字节流中
        contents = io.BytesIO(fh.read())
    # 使用 read_stata 函数读取字节流中的数据，指定 index_col 为 "index"
    reread = read_stata(contents, index_col="index")
    # 使用 tm.assert_frame_equal 检查 df 和 reread 是否相等
    tm.assert_frame_equal(df, reread)
# 使用 pytest.mark.parametrize 装饰器为 test_stata_compression 函数创建参数化测试
@pytest.mark.parametrize("to_infer", [True, False])
@pytest.mark.parametrize("read_infer", [True, False])
def test_stata_compression(
    compression_only, read_infer, to_infer, compression_to_extension, tmp_path
):
    # 设置 compression 变量为 compression_only 参数的值
    compression = compression_only

    # 根据 compression_to_extension 字典选择相应的文件扩展名
    ext = compression_to_extension[compression]
    # 使用文件扩展名创建文件名
    filename = f"test.{ext}"

    # 创建测试用的 DataFrame
    df = DataFrame(
        [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
        index=["A", "B"],
        columns=["X", "Y", "Z"],
    )
    # 设置 DataFrame 的索引名称为 "index"
    df.index.name = "index"

    # 根据 to_infer 和 read_infer 的值选择是否使用 "infer" 作为压缩参数
    to_compression = "infer" if to_infer else compression
    read_compression = "infer" if read_infer else compression

    # 创建临时文件路径
    path = tmp_path / filename
    # 创建空文件
    path.touch()
    # 将 DataFrame 写入 Stata 格式的文件，使用指定的压缩参数
    df.to_stata(path, compression=to_compression)
    # 从 Stata 格式的文件中读取数据到 DataFrame，使用指定的压缩参数和索引列
    result = read_stata(path, compression=read_compression, index_col="index")
    # 断言读取的 DataFrame 与原始 DataFrame 相等
    tm.assert_frame_equal(result, df)


# 定义用于测试非分类值标签的函数
def test_non_categorical_value_labels(temp_file):
    # 创建包含不同数据类型列的 DataFrame
    data = DataFrame(
        {
            "fully_labelled": [1, 2, 3, 3, 1],
            "partially_labelled": [1.0, 2.0, np.nan, 9.0, np.nan],
            "Y": [7, 7, 9, 8, 10],
            "Z": pd.Categorical(["j", "k", "l", "k", "j"]),
        }
    )

    # 设置临时文件路径
    path = temp_file
    # 创建值标签字典
    value_labels = {
        "fully_labelled": {1: "one", 2: "two", 3: "three"},
        "partially_labelled": {1.0: "one", 2.0: "two"},
    }
    # 创建期望的值标签字典，包括原始的值标签和对分类数据列 "Z" 的处理
    expected = {**value_labels, "Z": {0: "j", 1: "k", 2: "l"}}

    # 创建 StataWriter 对象，将数据和值标签写入到指定的 Stata 格式文件
    writer = StataWriter(path, data, value_labels=value_labels)
    writer.write_file()

    # 使用 StataReader 读取文件，并获取其值标签
    with StataReader(path) as reader:
        reader_value_labels = reader.value_labels()
        # 断言读取的值标签与预期的值标签相等
        assert reader_value_labels == expected

    # 验证处理未在数据集中找到的值标签时引发 KeyError 异常
    msg = "Can't create value labels for notY, it wasn't found in the dataset."
    value_labels = {"notY": {7: "label1", 8: "label2"}}
    with pytest.raises(KeyError, match=msg):
        StataWriter(path, data, value_labels=value_labels)

    # 验证尝试应用于非数值列 "Z" 的值标签时引发 ValueError 异常
    msg = (
        "Can't create value labels for Z, value labels "
        "can only be applied to numeric columns."
    )
    value_labels = {"Z": {1: "a", 2: "k", 3: "j", 4: "i"}}
    with pytest.raises(ValueError, match=msg):
        StataWriter(path, data, value_labels=value_labels)


# 定义用于测试非分类值标签名称转换的函数
def test_non_categorical_value_label_name_conversion(temp_file):
    # 检查无效变量名的转换
    data = DataFrame(
        {
            "invalid~!": [1, 1, 2, 3, 5, 8],  # 只允许使用字母数字和下划线
            "6_invalid": [1, 1, 2, 3, 5, 8],  # 必须以字母或下划线开头
            "invalid_name_longer_than_32_characters": [8, 8, 9, 9, 8, 8],  # 名称过长
            "aggregate": [2, 5, 5, 6, 6, 9],  # 保留词汇
            (1, 2): [1, 2, 3, 4, 5, 6],  # 不是字符串的可散列对象
        }
    )

    # 创建值标签字典，包含不同的列名及其对应的标签
    value_labels = {
        "invalid~!": {1: "label1", 2: "label2"},
        "6_invalid": {1: "label1", 2: "label2"},
        "invalid_name_longer_than_32_characters": {8: "eight", 9: "nine"},
        "aggregate": {5: "five"},
        (1, 2): {3: "three"},
    }
    # 定义预期的字典，包含多个键值对，每个键是一个可能无效的列名，对应值是一个字典，用于测试目的
    expected = {
        "invalid__": {1: "label1", 2: "label2"},
        "_6_invalid": {1: "label1", 2: "label2"},
        "invalid_name_longer_than_32_char": {8: "eight", 9: "nine"},
        "_aggregate": {5: "five"},
        "_1__2_": {3: "three"},
    }

    # 设置警告消息，用于在测试中验证抛出特定异常类（InvalidColumnName）时使用
    msg = "Not all pandas column names were valid Stata variable names"

    # 使用断言上下文（assert_produces_warning），验证在执行特定操作（to_stata）时是否会抛出指定异常（InvalidColumnName）
    with tm.assert_produces_warning(InvalidColumnName, match=msg):
        # 将数据转换为 Stata 格式并写入临时文件，同时传入值标签（value_labels）
        data.to_stata(temp_file, value_labels=value_labels)

    # 使用 StataReader 打开临时文件，以便后续验证读取的值标签（value_labels）
    with StataReader(temp_file) as reader:
        # 获取 StataReader 实例中的值标签
        reader_value_labels = reader.value_labels()
        # 断言读取的值标签与预期的字典内容是否相等
        assert reader_value_labels == expected
# 测试非分类值标签转换为分类值时的错误情况
def test_non_categorical_value_label_convert_categoricals_error(temp_file):
    # 定义值标签字典，其中一个标签映射多个值，这在 Stata 标签中是有效的，
    # 但在 convert_categoricals=True 的情况下无法读取
    value_labels = {
        "repeated_labels": {10: "Ten", 20: "More than ten", 40: "More than ten"}
    }

    # 创建包含重复值的数据帧
    data = DataFrame(
        {
            "repeated_labels": [10, 10, 20, 20, 40, 40],
        }
    )

    # 将数据帧写入 Stata 格式的临时文件，并附带值标签
    data.to_stata(temp_file, value_labels=value_labels)

    # 使用 StataReader 读取临时文件，不进行分类值转换
    with StataReader(temp_file, convert_categoricals=False) as reader:
        # 获取读取器中的值标签
        reader_value_labels = reader.value_labels()
    # 断言读取的值标签与预期的值标签相同
    assert reader_value_labels == value_labels

    # 准备用于断言错误消息的相关信息
    col = "repeated_labels"
    repeats = "-" * 80 + "\n" + "\n".join(["More than ten"])

    # 构建错误消息，用于断言异常抛出时的匹配
    msg = f"""
Value labels for column {col} are not unique. These cannot be converted to
pandas categoricals.

Either read the file with `convert_categoricals` set to False or use the
low level interface in `StataReader` to separately read the values and the
value_labels.

The repeated labels are:
{repeats}
"""
    # 使用 pytest 断言抛出指定消息的 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        read_stata(temp_file, convert_categoricals=True)


# 使用参数化测试不同版本和数据类型的组合
@pytest.mark.parametrize("version", [114, 117, 118, 119, None])
@pytest.mark.parametrize(
    "dtype",
    [
        pd.BooleanDtype,
        pd.Int8Dtype,
        pd.Int16Dtype,
        pd.Int32Dtype,
        pd.Int64Dtype,
        pd.UInt8Dtype,
        pd.UInt16Dtype,
        pd.UInt32Dtype,
        pd.UInt64Dtype,
    ],
)
def test_nullable_support(dtype, version, temp_file):
    # 创建包含不同数据类型和空值的数据帧
    df = DataFrame(
        {
            "a": Series([1.0, 2.0, 3.0]),
            "b": Series([1, pd.NA, pd.NA], dtype=dtype.name),
            "c": Series(["a", "b", None]),
        }
    )
    # 获取列 b 的 numpy 数据类型名称
    dtype_name = df.b.dtype.numpy_dtype.name
    # 如果数据类型名称包含 "u"，则将其替换为空字符串
    dtype_name = dtype_name.replace("u", "")
    # 如果数据类型名称为 "int64"，则将其替换为 "int32"
    if dtype_name == "int64":
        dtype_name = "int32"
    # 如果数据类型名称为 "bool"，则将其替换为 "int8"
    elif dtype_name == "bool":
        dtype_name = "int8"
    # 根据处理后的数据类型名称获取 Stata 缺失值常量
    value = StataMissingValue.BASE_MISSING_VALUES[dtype_name]
    # 创建 StataMissingValue 实例
    smv = StataMissingValue(value)
    # 创建预期的 Series 对象，用于比较
    expected_b = Series([1, smv, smv], dtype=object, name="b")
    expected_c = Series(["a", "b", ""], name="c")
    # 将数据帧写入 Stata 格式的临时文件，指定写入版本号
    df.to_stata(temp_file, write_index=False, version=version)
    # 重新读取并解析 Stata 文件
    reread = read_stata(temp_file, convert_missing=True)
    # 使用 pandas 测试工具比较数据帧中的列
    tm.assert_series_equal(df.a, reread.a)
    tm.assert_series_equal(reread.b, expected_b)
    tm.assert_series_equal(reread.c, expected_c)


# 测试空数据帧的读写
def test_empty_frame(temp_file):
    # 创建一个包含 int64 和 float64 类型的空数据帧
    df = DataFrame(data={"a": range(3), "b": [1.0, 2.0, 3.0]}).head(0)
    # 指定临时文件路径
    path = temp_file
    # 将空数据帧写入 Stata 格式的临时文件，指定写入版本号 117
    df.to_stata(path, write_index=False, version=117)
    # 读取整个 Stata 文件作为数据帧
    df2 = read_stata(path)
    # 断言列 "b" 在读取的数据帧中存在
    assert "b" in df2
    # 断言数据帧的数据类型，因为 int32 不受支持，所以会被替换为 int64
    dtypes = Series({"a": np.dtype("int32"), "b": np.dtype("float64")})
    tm.assert_series_equal(df2.dtypes, dtypes)
    # 使用自定义的函数 read_stata 读取指定路径下的 Stata 文件，并只选择列 "a" 的数据
    df3 = read_stata(path, columns=["a"])
    
    # 断言检查 DataFrame df3 中是否不包含列 "b"
    assert "b" not in df3
    
    # 使用 testtools 库中的 assert_series_equal 函数，验证 df3 的数据类型与预期的 dtypes 中列 "a" 的数据类型相等
    tm.assert_series_equal(df3.dtypes, dtypes.loc[["a"]])
```