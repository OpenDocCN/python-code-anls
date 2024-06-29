# `D:\src\scipysrc\pandas\pandas\tests\arrays\categorical\test_repr.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试

from pandas._config import using_pyarrow_string_dtype  # 从 pandas._config 模块导入 using_pyarrow_string_dtype 变量

from pandas import (  # 从 pandas 库中导入以下类和函数
    Categorical,
    CategoricalDtype,
    CategoricalIndex,
    Index,
    Series,
    date_range,
    option_context,
    period_range,
    timedelta_range,
)


class TestCategoricalReprWithFactor:
    def test_print(self, using_infer_string):
        factor = Categorical(["a", "b", "b", "a", "a", "c", "c", "c"], ordered=True)
        if using_infer_string:
            expected = [
                "['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c']",  # 预期输出的第一行字符串表示
                "Categories (3, string): [a < b < c]",  # 预期输出的第二行字符串表示
            ]
        else:
            expected = [
                "['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c']",  # 预期输出的第一行字符串表示
                "Categories (3, object): ['a' < 'b' < 'c']",  # 预期输出的第二行字符串表示
            ]
        expected = "\n".join(expected)  # 将预期输出列表转换为换行连接的字符串格式
        actual = repr(factor)  # 获取实际输出的字符串表示
        assert actual == expected  # 断言实际输出与预期输出相同


class TestCategoricalRepr:
    def test_big_print(self):
        codes = np.array([0, 1, 2, 0, 1, 2] * 100)  # 创建一个重复模式的 NumPy 数组
        dtype = CategoricalDtype(categories=Index(["a", "b", "c"], dtype=object))  # 创建分类数据类型对象
        factor = Categorical.from_codes(codes, dtype=dtype)  # 根据编码和数据类型创建分类对象
        expected = [
            "['a', 'b', 'c', 'a', 'b', ..., 'b', 'c', 'a', 'b', 'c']",  # 预期输出的第一行字符串表示
            "Length: 600",  # 预期输出的第二行字符串表示
            "Categories (3, object): ['a', 'b', 'c']",  # 预期输出的第三行字符串表示
        ]
        expected = "\n".join(expected)  # 将预期输出列表转换为换行连接的字符串格式

        actual = repr(factor)  # 获取实际输出的字符串表示

        assert actual == expected  # 断言实际输出与预期输出相同

    def test_empty_print(self):
        factor = Categorical([], Index(["a", "b", "c"], dtype=object))  # 创建空的分类对象
        expected = "[], Categories (3, object): ['a', 'b', 'c']"  # 预期输出的字符串表示
        actual = repr(factor)  # 获取实际输出的字符串表示
        assert actual == expected  # 断言实际输出与预期输出相同

        assert expected == actual  # 再次断言实际输出与预期输出相同
        factor = Categorical([], Index(["a", "b", "c"], dtype=object), ordered=True)  # 创建有序的空分类对象
        expected = "[], Categories (3, object): ['a' < 'b' < 'c']"  # 预期输出的字符串表示
        actual = repr(factor)  # 获取实际输出的字符串表示
        assert expected == actual  # 断言实际输出与预期输出相同

        factor = Categorical([], [])  # 创建空的分类对象
        expected = "[], Categories (0, object): []"  # 预期输出的字符串表示
        assert expected == repr(factor)  # 断言实际输出与预期输出相同

    def test_print_none_width(self):
        # GH10087
        a = Series(Categorical([1, 2, 3, 4]))  # 创建包含分类数据的 Series 对象
        exp = (
            "0    1\n1    2\n2    3\n3    4\n"
            "dtype: category\nCategories (4, int64): [1, 2, 3, 4]"  # 预期输出的字符串表示
        )

        with option_context("display.width", None):  # 设置显示宽度为 None
            assert exp == repr(a)  # 断言实际输出与预期输出相同

    @pytest.mark.skipif(
        using_pyarrow_string_dtype(),  # 如果 using_pyarrow_string_dtype 返回 True
        reason="Change once infer_string is set to True by default",  # 测试跳过的原因
    )
    def test_unicode_print(self):
        c = Categorical(["aaaaa", "bb", "cccc"] * 20)  # 创建包含 Unicode 数据的分类对象
        expected = """\
['aaaaa', 'bb', 'cccc', 'aaaaa', 'bb', ..., 'bb', 'cccc', 'aaaaa', 'bb', 'cccc']
Length: 60
Categories (3, object): ['aaaaa', 'bb', 'cccc']"""  # 预期输出的字符串表示

        assert repr(c) == expected  # 断言实际输出与预期输出相同

        c = Categorical(["ああああ", "いいいいい", "ううううううう"] * 20)  # 创建包含 Unicode 数据的分类对象
        expected = """\
['ああああ', 'いいいいい', 'ううううううう', 'ああああ', 'いいいいい', ..., 'いいいいい', 'ううううううう', 'ああああ', 'いいいいい', 'ううううううう']"""  # 预期输出的字符串表示
    def test_categorical_repr(self):
        # 创建一个分类变量 c，包含整数 1, 2, 3
        c = Categorical([1, 2, 3])
        # 预期的字符串表示，显示分类变量 c 的内容和分类信息
        exp = """[1, 2, 3]
Categories (3, int64): [1, 2, 3]"""
        
        # 断言分类变量 c 的字符串表示是否与预期的 exp 相同
        assert repr(c) == exp

        # 创建一个分类变量 c，包含整数 1, 2, 3，并指定其分类为 [1, 2, 3]
        c = Categorical([1, 2, 3, 1, 2, 3], categories=[1, 2, 3])
        # 预期的字符串表示，显示分类变量 c 的内容和指定的分类信息
        exp = """[1, 2, 3, 1, 2, 3]
Categories (3, int64): [1, 2, 3]"""
        
        # 断言分类变量 c 的字符串表示是否与预期的 exp 相同
        assert repr(c) == exp

        # 创建一个分类变量 c，包含重复 1, 2, 3, 4, 5 十次的数据
        c = Categorical([1, 2, 3, 4, 5] * 10)
        # 预期的字符串表示，显示分类变量 c 的内容和长度信息，以及其分类信息
        exp = """[1, 2, 3, 4, 5, ..., 1, 2, 3, 4, 5]
Length: 50
Categories (5, int64): [1, 2, 3, 4, 5]"""
        
        # 断言分类变量 c 的字符串表示是否与预期的 exp 相同
        assert repr(c) == exp

        # 创建一个分类变量 c，包含整数范围为 0 到 19
        c = Categorical(np.arange(20, dtype=np.int64))
        # 预期的字符串表示，显示分类变量 c 的内容和长度信息，以及其分类信息
        exp = """[0, 1, 2, 3, 4, ..., 15, 16, 17, 18, 19]
Length: 20
Categories (20, int64): [0, 1, 2, 3, ..., 16, 17, 18, 19]"""
        
        # 断言分类变量 c 的字符串表示是否与预期的 exp 相同
        assert repr(c) == exp

    def test_categorical_repr_ordered(self):
        # 创建一个有序的分类变量 c，包含整数 1, 2, 3
        c = Categorical([1, 2, 3], ordered=True)
        # 预期的字符串表示，显示分类变量 c 的内容和有序分类信息
        exp = """[1, 2, 3]
Categories (3, int64): [1 < 2 < 3]"""
        
        # 断言分类变量 c 的字符串表示是否与预期的 exp 相同
        assert repr(c) == exp

        # 创建一个有序的分类变量 c，包含整数 1, 2, 3，并指定其分类为 [1, 2, 3]
        c = Categorical([1, 2, 3, 1, 2, 3], categories=[1, 2, 3], ordered=True)
        # 预期的字符串表示，显示分类变量 c 的内容和有序分类信息
        exp = """[1, 2, 3, 1, 2, 3]
Categories (3, int64): [1 < 2 < 3]"""
        
        # 断言分类变量 c 的字符串表示是否与预期的 exp 相同
        assert repr(c) == exp

        # 创建一个有序的分类变量 c，包含重复 1, 2, 3, 4, 5 十次的数据
        c = Categorical([1, 2, 3, 4, 5] * 10, ordered=True)
        # 预期的字符串表示，显示分类变量 c 的内容和长度信息，以及其有序分类信息
        exp = """[1, 2, 3, 4, 5, ..., 1, 2, 3, 4, 5]
Length: 50
Categories (5, int64): [1 < 2 < 3 < 4 < 5]"""
        
        # 断言分类变量 c 的字符串表示是否与预期的 exp 相同
        assert repr(c) == exp

        # 创建一个有序的分类变量 c，包含整数范围为 0 到 19
        c = Categorical(np.arange(20, dtype=np.int64), ordered=True)
        # 预期的字符串表示，显示分类变量 c 的内容和长度信息，以及其有序分类信息
        exp = """[0, 1, 2, 3, 4, ..., 15, 16, 17, 18, 19]
Length: 20
Categories (20, int64): [0 < 1 < 2 < 3 ... 16 < 17 < 18 < 19]"""
        
        # 断言分类变量 c 的字符串表示是否与预期的 exp 相同
        assert repr(c) == exp
    def test_categorical_repr_datetime(self):
        # 创建时间范围对象，从"2011-01-01 09:00"开始，按小时频率，总共5个时间点
        idx = date_range("2011-01-01 09:00", freq="h", periods=5)
        # 创建分类变量对象，使用时间范围对象idx
        c = Categorical(idx)

        # 预期的字符串表示形式，展示了时间范围内各个时间点的字符串表示及分类信息
        exp = (
            "[2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, "
            "2011-01-01 12:00:00, 2011-01-01 13:00:00]\n"
            "Categories (5, datetime64[ns]): [2011-01-01 09:00:00, "
            "2011-01-01 10:00:00, 2011-01-01 11:00:00,\n"
            "                                 2011-01-01 12:00:00, "
            "2011-01-01 13:00:00]"
            ""
        )
        # 断言，确保分类变量对象的字符串表示形式与预期一致
        assert repr(c) == exp

        # 将时间范围对象idx扩展为包含两倍长度的新对象，使用原始时间范围作为类别
        c = Categorical(idx.append(idx), categories=idx)
        # 新的预期字符串表示形式，包括扩展后的时间范围和相关的分类信息
        exp = (
            "[2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, "
            "2011-01-01 12:00:00, 2011-01-01 13:00:00, 2011-01-01 09:00:00, "
            "2011-01-01 10:00:00, 2011-01-01 11:00:00, 2011-01-01 12:00:00, "
            "2011-01-01 13:00:00]\n"
            "Categories (5, datetime64[ns]): [2011-01-01 09:00:00, "
            "2011-01-01 10:00:00, 2011-01-01 11:00:00,\n"
            "                                 2011-01-01 12:00:00, "
            "2011-01-01 13:00:00]"
        )

        # 断言，确保分类变量对象的字符串表示形式与新的预期形式一致
        assert repr(c) == exp

        # 创建带有时区信息的时间范围对象idx，从"2011-01-01 09:00"开始，按小时频率，总共5个时间点，使用"US/Eastern"时区
        idx = date_range("2011-01-01 09:00", freq="h", periods=5, tz="US/Eastern")
        # 创建分类变量对象，使用带有时区信息的时间范围对象idx
        c = Categorical(idx)
        # 带有时区信息的预期字符串表示形式，显示了时间范围内各个时间点的字符串表示及分类信息
        exp = (
            "[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, "
            "2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, "
            "2011-01-01 13:00:00-05:00]\n"
            "Categories (5, datetime64[ns, US/Eastern]): "
            "[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00,\n"
            "                                             "
            "2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00,\n"
            "                                             "
            "2011-01-01 13:00:00-05:00]"
        )

        # 断言，确保分类变量对象的字符串表示形式与带有时区信息的预期形式一致
        assert repr(c) == exp

        # 将带有时区信息的时间范围对象idx扩展为包含两倍长度的新对象，使用原始时间范围作为类别
        c = Categorical(idx.append(idx), categories=idx)
        # 新的预期字符串表示形式，包括扩展后的时间范围和相关的分类信息
        exp = (
            "[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, "
            "2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, "
            "2011-01-01 13:00:00-05:00, 2011-01-01 09:00:00-05:00, "
            "2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, "
            "2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00]\n"
            "Categories (5, datetime64[ns, US/Eastern]): "
            "[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00,\n"
            "                                             "
            "2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00,\n"
            "                                             "
            "2011-01-01 13:00:00-05:00]"
        )

        # 断言，确保分类变量对象的字符串表示形式与新的预期形式一致
        assert repr(c) == exp
    # 定义测试函数，用于测试有序分类数据的表示方式
    def test_categorical_repr_datetime_ordered(self):
        # 创建一个日期范围，从 "2011-01-01 09:00" 开始，频率为每小时，共计5个时间点
        idx = date_range("2011-01-01 09:00", freq="h", periods=5)
        # 使用日期时间索引创建一个有序分类对象
        c = Categorical(idx, ordered=True)
        # 期望的字符串表示，展示了每小时的时间点
        exp = """[2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, 2011-01-01 12:00:00, 2011-01-01 13:00:00]
        assert repr(c) == exp
        # 断言当前对象的字符串表示应该等于期望值exp

        c = Categorical(idx.append(idx), categories=idx, ordered=True)
        # 创建一个分类变量c，使用idx两次追加生成，指定类别和有序性为True

        exp = """[2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, 2011-01-01 12:00:00, 2011-01-01 13:00:00, 2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, 2011-01-01 12:00:00, 2011-01-01 13:00:00]
Categories (5, datetime64[ns]): [2011-01-01 09:00:00 < 2011-01-01 10:00:00 < 2011-01-01 11:00:00 <
                                 2011-01-01 12:00:00 < 2011-01-01 13:00:00]"""
        # 设置期望的字符串表示exp，包括多次出现的时间戳和类别信息

        assert repr(c) == exp
        # 断言当前对象c的字符串表示应该等于期望值exp

        idx = date_range("2011-01-01 09:00", freq="h", periods=5, tz="US/Eastern")
        # 生成一个包含时区信息的时间范围idx，频率为每小时，共5个时段

        c = Categorical(idx, ordered=True)
        # 创建一个分类变量c，使用生成的时间范围idx，并指定有序性为True

        exp = """[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00]
Categories (5, datetime64[ns, US/Eastern]): [2011-01-01 09:00:00-05:00 < 2011-01-01 10:00:00-05:00 <
                                             2011-01-01 11:00:00-05:00 < 2011-01-01 12:00:00-05:00 <
                                             2011-01-01 13:00:00-05:00]"""
        # 设置期望的字符串表示exp，包括时间戳和带有时区信息的类别

        assert repr(c) == exp
        # 断言当前对象c的字符串表示应该等于期望值exp

        c = Categorical(idx.append(idx), categories=idx, ordered=True)
        # 创建一个分类变量c，使用idx两次追加生成，指定类别和有序性为True

        exp = """[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00, 2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00]
Categories (5, datetime64[ns, US/Eastern]): [2011-01-01 09:00:00-05:00 < 2011-01-01 10:00:00-05:00 <
                                             2011-01-01 11:00:00-05:00 < 2011-01-01 12:00:00-05:00 <
                                             2011-01-01 13:00:00-05:00]"""
        # 设置期望的字符串表示exp，包括多次出现的时间戳和带有时区信息的类别

        assert repr(c) == exp
        # 断言当前对象c的字符串表示应该等于期望值exp
    def test_categorical_repr_period_ordered(self):
        # 创建一个时间段范围，每小时一个数据点，共5个数据点
        idx = period_range("2011-01-01 09:00", freq="h", periods=5)
        # 创建一个有序的分类对象
        c = Categorical(idx, ordered=True)
        # 预期的字符串表示，包含时间段数据和有序的分类信息
        exp = """[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00]
Categories (5, period[h]): [2011-01-01 09:00 < 2011-01-01 10:00 < 2011-01-01 11:00 < 2011-01-01 12:00 <
                            2011-01-01 13:00]"""  # noqa: E501

        # 断言预期字符串和对象的字符串表示一致
        assert repr(c) == exp

        # 将索引数据重复一次并创建一个有序的分类对象
        c = Categorical(idx.append(idx), categories=idx, ordered=True)
        # 预期的字符串表示，包含重复的时间段数据和有序的分类信息
        exp = """[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00, 2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00]
Categories (5, period[h]): [2011-01-01 09:00 < 2011-01-01 10:00 < 2011-01-01 11:00 < 2011-01-01 12:00 <
                            2011-01-01 13:00]"""  # noqa: E501

        # 断言预期字符串和对象的字符串表示一致
        assert repr(c) == exp

        # 创建一个时间段范围，每月一个数据点，共5个数据点
        idx = period_range("2011-01", freq="M", periods=5)
        # 创建一个有序的分类对象
        c = Categorical(idx, ordered=True)
        # 预期的字符串表示，包含月份数据和有序的分类信息
        exp = """[2011-01, 2011-02, 2011-03, 2011-04, 2011-05]
Categories (5, period[M]): [2011-01 < 2011-02 < 2011-03 < 2011-04 < 2011-05]"""

        # 断言预期字符串和对象的字符串表示一致
        assert repr(c) == exp

        # 将索引数据重复一次并创建一个有序的分类对象
        c = Categorical(idx.append(idx), categories=idx, ordered=True)
        # 预期的字符串表示，包含重复的月份数据和有序的分类信息
        exp = """[2011-01, 2011-02, 2011-03, 2011-04, 2011-05, 2011-01, 2011-02, 2011-03, 2011-04, 2011-05]
Categories (5, period[M]): [2011-01 < 2011-02 < 2011-03 < 2011-04 < 2011-05]"""  # noqa: E501

        # 断言预期字符串和对象的字符串表示一致
        assert repr(c) == exp

    def test_categorical_repr_timedelta(self):
        # 创建一个时间增量范围，每增量一天一个数据点，共5个数据点
        idx = timedelta_range("1 days", periods=5)
        # 创建一个分类对象
        c = Categorical(idx)
        # 预期的字符串表示，包含时间增量数据
        exp = """[1 days, 2 days, 3 days, 4 days, 5 days]
        idx = timedelta_range("1 days", periods=5)
        # 创建一个时间增量序列，从1天开始，包含5个周期
        c = Categorical(idx)
        # 使用时间增量序列创建一个分类变量
        exp = """[1 days, 2 days, 3 days, 4 days, 5 days]
Categories (5, timedelta64[ns]): [1 days, 2 days, 3 days, 4 days, 5 days]"""
        # 期望的字符串表示，显示时间增量序列的内容和分类信息

        assert repr(c) == exp
        # 断言：确保分类变量的字符串表示与期望的一致

        c = Categorical(idx.append(idx), categories=idx)
        # 创建一个包含两个相同时间增量序列的分类变量，并指定分类
        exp = """[1 days, 2 days, 3 days, 4 days, 5 days, 1 days, 2 days, 3 days, 4 days, 5 days]
Categories (5, timedelta64[ns]): [1 days, 2 days, 3 days, 4 days, 5 days]"""  # noqa: E501
        # 期望的字符串表示，显示合并后的时间增量序列内容和分类信息

        assert repr(c) == exp
        # 断言：确保分类变量的字符串表示与期望的一致

    def test_categorical_repr_timedelta_ordered(self):
        idx = timedelta_range("1 days", periods=5)
        # 创建一个时间增量序列，从1天开始，包含5个周期
        c = Categorical(idx, ordered=True)
        # 使用时间增量序列创建一个有序的分类变量
        exp = """[1 days, 2 days, 3 days, 4 days, 5 days]
Categories (5, timedelta64[ns]): [1 days < 2 days < 3 days < 4 days < 5 days]"""
        # 期望的字符串表示，显示时间增量序列的内容和有序分类信息

        assert repr(c) == exp
        # 断言：确保有序分类变量的字符串表示与期望的一致

        c = Categorical(idx.append(idx), categories=idx, ordered=True)
        # 创建一个包含两个相同时间增量序列的有序分类变量，并指定分类
        exp = """[1 days, 2 days, 3 days, 4 days, 5 days, 1 days, 2 days, 3 days, 4 days, 5 days]
Categories (5, timedelta64[ns]): [1 days < 2 days < 3 days < 4 days < 5 days]"""  # noqa: E501
        # 期望的字符串表示，显示合并后的时间增量序列内容和有序分类信息

        assert repr(c) == exp
        # 断言：确保有序分类变量的字符串表示与期望的一致

        idx = timedelta_range("1 hours", periods=20)
        # 创建一个时间增量序列，从1小时开始，包含20个周期
        c = Categorical(idx, ordered=True)
        # 使用时间增量序列创建一个有序的分类变量
        exp = """[0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00, 3 days 01:00:00, 4 days 01:00:00, ..., 15 days 01:00:00, 16 days 01:00:00, 17 days 01:00:00, 18 days 01:00:00, 19 days 01:00:00]
Length: 20
Categories (20, timedelta64[ns]): [0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00,
                                   3 days 01:00:00, ..., 16 days 01:00:00, 17 days 01:00:00,
                                   18 days 01:00:00, 19 days 01:00:00]"""
        # 期望的字符串表示，显示时间增量序列的内容和有序分类信息

        assert repr(c) == exp
        # 断言：确保有序分类变量的字符串表示与期望的一致
        assert repr(c) == exp


# 断言语句：验证变量 c 的表示形式是否等于期望值 exp
assert repr(c) == exp



        c = Categorical(idx.append(idx), categories=idx, ordered=True)
        exp = """[0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00, 3 days 01:00:00, 4 days 01:00:00, ..., 15 days 01:00:00, 16 days 01:00:00, 17 days 01:00:00, 18 days 01:00:00, 19 days 01:00:00]
Length: 40
Categories (20, timedelta64[ns]): [0 days 01:00:00 < 1 days 01:00:00 < 2 days 01:00:00 <
                                   3 days 01:00:00 ... 16 days 01:00:00 < 17 days 01:00:00 <
                                   18 days 01:00:00 < 19 days 01:00:00]"""  # noqa: E501


# 创建 Categorical 对象 c，使用重复的索引 idx，指定类别和有序性为 True
c = Categorical(idx.append(idx), categories=idx, ordered=True)
# 期望的字符串表示形式 exp，展示了长度为 40 的序列以及类别的详细信息
exp = """[0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00, 3 days 01:00:00, 4 days 01:00:00, ..., 15 days 01:00:00, 16 days 01:00:00, 17 days 01:00:00, 18 days 01:00:00, 19 days 01:00:00]
Length: 40
Categories (20, timedelta64[ns]): [0 days 01:00:00 < 1 days 01:00:00 < 2 days 01:00:00 <
                                   3 days 01:00:00 ... 16 days 01:00:00 < 17 days 01:00:00 <
                                   18 days 01:00:00 < 19 days 01:00:00]"""  # noqa: E501



        assert repr(c) == exp


# 断言语句：验证变量 c 的字符串表示形式是否等于期望值 exp
assert repr(c) == exp



    def test_categorical_index_repr(self):


# 定义测试方法 test_categorical_index_repr，用于测试 CategoricalIndex 的字符串表示形式
def test_categorical_index_repr(self):



        idx = CategoricalIndex(Categorical([1, 2, 3]))
        exp = """CategoricalIndex([1, 2, 3], categories=[1, 2, 3], ordered=False, dtype='category')"""  # noqa: E501


# 创建 CategoricalIndex 对象 idx，包含分类变量 [1, 2, 3]
idx = CategoricalIndex(Categorical([1, 2, 3]))
# 期望的字符串表示形式 exp，展示了包含的值、类别、有序性和数据类型信息
exp = """CategoricalIndex([1, 2, 3], categories=[1, 2, 3], ordered=False, dtype='category')"""  # noqa: E501



        assert repr(idx) == exp


# 断言语句：验证变量 idx 的字符串表示形式是否等于期望值 exp
assert repr(idx) == exp



        i = CategoricalIndex(Categorical(np.arange(10, dtype=np.int64)))
        exp = """CategoricalIndex([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], categories=[0, 1, 2, 3, ..., 6, 7, 8, 9], ordered=False, dtype='category')"""  # noqa: E501


# 创建 CategoricalIndex 对象 i，包含整数范围 [0, 9] 的分类变量
i = CategoricalIndex(Categorical(np.arange(10, dtype=np.int64)))
# 期望的字符串表示形式 exp，展示了包含的值、类别、有序性和数据类型信息
exp = """CategoricalIndex([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], categories=[0, 1, 2, 3, ..., 6, 7, 8, 9], ordered=False, dtype='category')"""  # noqa: E501



        assert repr(i) == exp


# 断言语句：验证变量 i 的字符串表示形式是否等于期望值 exp
assert repr(i) == exp



    def test_categorical_index_repr_ordered(self):


# 定义测试方法 test_categorical_index_repr_ordered，用于测试有序的 CategoricalIndex 的字符串表示形式
def test_categorical_index_repr_ordered(self):



        i = CategoricalIndex(Categorical([1, 2, 3], ordered=True))
        exp = """CategoricalIndex([1, 2, 3], categories=[1, 2, 3], ordered=True, dtype='category')"""  # noqa: E501


# 创建有序的 CategoricalIndex 对象 i，包含有序的分类变量 [1, 2, 3]
i = CategoricalIndex(Categorical([1, 2, 3], ordered=True))
# 期望的字符串表示形式 exp，展示了包含的值、类别、有序性和数据类型信息
exp = """CategoricalIndex([1, 2, 3], categories=[1, 2, 3], ordered=True, dtype='category')"""  # noqa: E501



        assert repr(i) == exp


# 断言语句：验证变量 i 的字符串表示形式是否等于期望值 exp
assert repr(i) == exp
    # 定义一个测试方法，用于测试处理分类索引、表示和日期时间的情况
    def test_categorical_index_repr_datetime(self):
        # 创建一个日期时间范围，从 "2011-01-01 09:00" 开始，频率为每小时，共5个时间点
        idx = date_range("2011-01-01 09:00", freq="h", periods=5)
        # 使用日期时间创建一个分类索引对象
        i = CategoricalIndex(Categorical(idx))
        # 期望的字符串表示形式，包含了日期时间字符串列表及其相关的元信息
        exp = """CategoricalIndex(['2011-01-01 09:00:00', '2011-01-01 10:00:00',
                  '2011-01-01 11:00:00', '2011-01-01 12:00:00',
                  '2011-01-01 13:00:00'],
                 categories=[2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, 2011-01-01 12:00:00, 2011-01-01 13:00:00], ordered=False, dtype='category')"""  # noqa: E501

        # 断言实际的 repr 结果与期望的字符串表示形式一致
        assert repr(i) == exp

        # 创建带有时区信息的日期时间范围，从 "2011-01-01 09:00" 开始，频率为每小时，共5个时间点，时区为 "US/Eastern"
        idx = date_range("2011-01-01 09:00", freq="h", periods=5, tz="US/Eastern")
        # 使用带有时区信息的日期时间创建一个分类索引对象
        i = CategoricalIndex(Categorical(idx))
        # 期望的字符串表示形式，包含了带有时区信息的日期时间字符串列表及其相关的元信息
        exp = """CategoricalIndex(['2011-01-01 09:00:00-05:00', '2011-01-01 10:00:00-05:00',
                  '2011-01-01 11:00:00-05:00', '2011-01-01 12:00:00-05:00',
                  '2011-01-01 13:00:00-05:00'],
                 categories=[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00], ordered=False, dtype='category')"""  # noqa: E501

        # 断言实际的 repr 结果与期望的字符串表示形式一致
        assert repr(i) == exp
    # 定义一个测试方法，用于测试分类索引的表现，包括日期时间和排序
    def test_categorical_index_repr_datetime_ordered(self):
        # 创建一个日期范围对象，从"2011-01-01 09:00"开始，每小时一个数据点，总共5个数据点
        idx = date_range("2011-01-01 09:00", freq="h", periods=5)
        # 使用日期时间创建一个有序的分类对象，并将其作为参数创建一个分类索引对象
        i = CategoricalIndex(Categorical(idx, ordered=True))
        # 期望的字符串表示，展示了分类索引的结构和元数据信息
        exp = """CategoricalIndex(['2011-01-01 09:00:00', '2011-01-01 10:00:00',
                  '2011-01-01 11:00:00', '2011-01-01 12:00:00',
                  '2011-01-01 13:00:00'],
                 categories=[2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, 2011-01-01 12:00:00, 2011-01-01 13:00:00], ordered=True, dtype='category')"""  # noqa: E501

        # 断言实际的分类索引对象的字符串表示与期望的字符串表示一致
        assert repr(i) == exp

        # 在具有时区信息的情况下，再次创建分类索引对象，并进行字符串表示的比较
        idx = date_range("2011-01-01 09:00", freq="h", periods=5, tz="US/Eastern")
        i = CategoricalIndex(Categorical(idx, ordered=True))
        exp = """CategoricalIndex(['2011-01-01 09:00:00-05:00', '2011-01-01 10:00:00-05:00',
                  '2011-01-01 11:00:00-05:00', '2011-01-01 12:00:00-05:00',
                  '2011-01-01 13:00:00-05:00'],
                 categories=[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00], ordered=True, dtype='category')"""  # noqa: E501

        # 再次断言实际的分类索引对象的字符串表示与期望的字符串表示一致
        assert repr(i) == exp

        # 将两个相同的日期范围对象连接起来，并创建一个新的分类索引对象，再次进行字符串表示的比较
        i = CategoricalIndex(Categorical(idx.append(idx), ordered=True))
        exp = """CategoricalIndex(['2011-01-01 09:00:00-05:00', '2011-01-01 10:00:00-05:00',
                  '2011-01-01 11:00:00-05:00', '2011-01-01 12:00:00-05:00',
                  '2011-01-01 13:00:00-05:00', '2011-01-01 09:00:00-05:00',
                  '2011-01-01 10:00:00-05:00', '2011-01-01 11:00:00-05:00',
                  '2011-01-01 12:00:00-05:00', '2011-01-01 13:00:00-05:00'],
                 categories=[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00], ordered=True, dtype='category')"""  # noqa: E501

        # 最后断言实际的分类索引对象的字符串表示与期望的字符串表示一致
        assert repr(i) == exp
    def test_categorical_index_repr_period(self):
        # 创建一个时期范围对象，起始时间为 "2011-01-01 09:00"，频率为每小时，包含1个时期
        idx = period_range("2011-01-01 09:00", freq="h", periods=1)
        # 创建一个分类索引对象，其内容是时期索引 idx 的分类版本
        i = CategoricalIndex(Categorical(idx))
        # 预期的字符串表示形式，描述了 CategoricalIndex 对象的结构和内容
        exp = """CategoricalIndex(['2011-01-01 09:00'], categories=[2011-01-01 09:00], ordered=False, dtype='category')"""  # noqa: E501
        # 断言实际的 repr(i) 结果与预期的 exp 字符串相等
        assert repr(i) == exp

        # 创建包含2个时期的时期范围对象
        idx = period_range("2011-01-01 09:00", freq="h", periods=2)
        # 创建对应的分类索引对象
        i = CategoricalIndex(Categorical(idx))
        # 预期的字符串表示形式，描述了 CategoricalIndex 对象的结构和内容
        exp = """CategoricalIndex(['2011-01-01 09:00', '2011-01-01 10:00'], categories=[2011-01-01 09:00, 2011-01-01 10:00], ordered=False, dtype='category')"""  # noqa: E501
        # 断言实际的 repr(i) 结果与预期的 exp 字符串相等
        assert repr(i) == exp

        # 创建包含3个时期的时期范围对象
        idx = period_range("2011-01-01 09:00", freq="h", periods=3)
        # 创建对应的分类索引对象
        i = CategoricalIndex(Categorical(idx))
        # 预期的字符串表示形式，描述了 CategoricalIndex 对象的结构和内容
        exp = """CategoricalIndex(['2011-01-01 09:00', '2011-01-01 10:00', '2011-01-01 11:00'], categories=[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00], ordered=False, dtype='category')"""  # noqa: E501
        # 断言实际的 repr(i) 结果与预期的 exp 字符串相等
        assert repr(i) == exp

        # 创建包含5个时期的时期范围对象
        idx = period_range("2011-01-01 09:00", freq="h", periods=5)
        # 创建对应的分类索引对象
        i = CategoricalIndex(Categorical(idx))
        # 预期的字符串表示形式，描述了 CategoricalIndex 对象的结构和内容
        exp = """CategoricalIndex(['2011-01-01 09:00', '2011-01-01 10:00', '2011-01-01 11:00',
                  '2011-01-01 12:00', '2011-01-01 13:00'],
                 categories=[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00], ordered=False, dtype='category')"""  # noqa: E501
        # 断言实际的 repr(i) 结果与预期的 exp 字符串相等
        assert repr(i) == exp

        # 将时期范围对象 idx 追加到其自身，并创建相应的分类索引对象
        i = CategoricalIndex(Categorical(idx.append(idx)))
        # 预期的字符串表示形式，描述了 CategoricalIndex 对象的结构和内容
        exp = """CategoricalIndex(['2011-01-01 09:00', '2011-01-01 10:00', '2011-01-01 11:00',
                  '2011-01-01 12:00', '2011-01-01 13:00', '2011-01-01 09:00',
                  '2011-01-01 10:00', '2011-01-01 11:00', '2011-01-01 12:00',
                  '2011-01-01 13:00'],
                 categories=[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00], ordered=False, dtype='category')"""  # noqa: E501
        # 断言实际的 repr(i) 结果与预期的 exp 字符串相等
        assert repr(i) == exp

        # 创建包含5个月份的时期范围对象
        idx = period_range("2011-01", freq="M", periods=5)
        # 创建对应的分类索引对象
        i = CategoricalIndex(Categorical(idx))
        # 预期的字符串表示形式，描述了 CategoricalIndex 对象的结构和内容
        exp = """CategoricalIndex(['2011-01', '2011-02', '2011-03', '2011-04', '2011-05'], categories=[2011-01, 2011-02, 2011-03, 2011-04, 2011-05], ordered=False, dtype='category')"""  # noqa: E501
        # 断言实际的 repr(i) 结果与预期的 exp 字符串相等
        assert repr(i) == exp
    # 测试方法：测试有序时间周期的分类索引的字符串表示形式
    def test_categorical_index_repr_period_ordered(self):
        # 创建一个时间周期范围，从"2011-01-01 09:00"开始，每小时(freq="h")生成5个时间点(periods=5)
        idx = period_range("2011-01-01 09:00", freq="h", periods=5)
        # 将时间周期索引转换为有序分类索引
        i = CategoricalIndex(Categorical(idx, ordered=True))
        # 期望的字符串表示形式，包含有序时间点和其类别
        exp = """CategoricalIndex(['2011-01-01 09:00', '2011-01-01 10:00', '2011-01-01 11:00',
                  '2011-01-01 12:00', '2011-01-01 13:00'],
                 categories=[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00], ordered=True, dtype='category')"""  # noqa: E501

        # 断言测试的字符串表示形式是否与期望的一致
        assert repr(i) == exp

        # 创建一个月份频率的时间周期范围，从"2011-01"开始，每月(freq="M")生成5个时间点(periods=5)
        idx = period_range("2011-01", freq="M", periods=5)
        # 将时间周期索引转换为有序分类索引
        i = CategoricalIndex(Categorical(idx, ordered=True))
        # 期望的字符串表示形式，包含有序时间点和其类别
        exp = """CategoricalIndex(['2011-01', '2011-02', '2011-03', '2011-04', '2011-05'], categories=[2011-01, 2011-02, 2011-03, 2011-04, 2011-05], ordered=True, dtype='category')"""  # noqa: E501
        # 断言测试的字符串表示形式是否与期望的一致
        assert repr(i) == exp

    # 测试方法：测试时间增量的分类索引的字符串表示形式
    def test_categorical_index_repr_timedelta(self):
        # 创建一个时间增量范围，从"1 days"开始，每天(periods=5)生成5个时间增量
        idx = timedelta_range("1 days", periods=5)
        # 将时间增量索引转换为分类索引
        i = CategoricalIndex(Categorical(idx))
        # 期望的字符串表示形式，包含时间增量和其类别
        exp = """CategoricalIndex(['1 days', '2 days', '3 days', '4 days', '5 days'], categories=[1 days, 2 days, 3 days, 4 days, 5 days], ordered=False, dtype='category')"""  # noqa: E501
        # 断言测试的字符串表示形式是否与期望的一致
        assert repr(i) == exp

        # 创建一个小时增量范围，从"1 hours"开始，每小时(periods=10)生成10个时间增量
        idx = timedelta_range("1 hours", periods=10)
        # 将时间增量索引转换为分类索引
        i = CategoricalIndex(Categorical(idx))
        # 期望的字符串表示形式，包含时间增量和其类别
        exp = """CategoricalIndex(['0 days 01:00:00', '1 days 01:00:00', '2 days 01:00:00',
                  '3 days 01:00:00', '4 days 01:00:00', '5 days 01:00:00',
                  '6 days 01:00:00', '7 days 01:00:00', '8 days 01:00:00',
                  '9 days 01:00:00'],
                 categories=[0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00, 3 days 01:00:00, ..., 6 days 01:00:00, 7 days 01:00:00, 8 days 01:00:00, 9 days 01:00:00], ordered=False, dtype='category')"""  # noqa: E501

        # 断言测试的字符串表示形式是否与期望的一致
        assert repr(i) == exp

    # 测试方法：测试有序时间增量的分类索引的字符串表示形式
    def test_categorical_index_repr_timedelta_ordered(self):
        # 创建一个时间增量范围，从"1 days"开始，每天(periods=5)生成5个时间增量
        idx = timedelta_range("1 days", periods=5)
        # 将时间增量索引转换为有序分类索引
        i = CategoricalIndex(Categorical(idx, ordered=True))
        # 期望的字符串表示形式，包含时间增量和其类别
        exp = """CategoricalIndex(['1 days', '2 days', '3 days', '4 days', '5 days'], categories=[1 days, 2 days, 3 days, 4 days, 5 days], ordered=True, dtype='category')"""  # noqa: E501
        # 断言测试的字符串表示形式是否与期望的一致
        assert repr(i) == exp

        # 创建一个小时增量范围，从"1 hours"开始，每小时(periods=10)生成10个时间增量
        idx = timedelta_range("1 hours", periods=10)
        # 将时间增量索引转换为有序分类索引
        i = CategoricalIndex(Categorical(idx, ordered=True))
        # 期望的字符串表示形式，包含时间增量和其类别
        exp = """CategoricalIndex(['0 days 01:00:00', '1 days 01:00:00', '2 days 01:00:00',
                  '3 days 01:00:00', '4 days 01:00:00', '5 days 01:00:00',
                  '6 days 01:00:00', '7 days 01:00:00', '8 days 01:00:00',
                  '9 days 01:00:00'],
                 categories=[0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00, 3 days 01:00:00, ..., 6 days 01:00:00, 7 days 01:00:00, 8 days 01:00:00, 9 days 01:00:00], ordered=True, dtype='category')"""  # noqa: E501

        # 断言测试的字符串表示形式是否与期望的一致
        assert repr(i) == exp
    def test_categorical_str_repr(self):
        # GH 33676
        # 创建一个 Categorical 对象，包含整数和字符串作为类别
        result = repr(Categorical([1, "2", 3, 4]))
        # 期望的字符串表示形式，包括原始列表和类别信息
        expected = "[1, '2', 3, 4]\nCategories (4, object): [1, 3, 4, '2']"
        # 断言结果与期望相同
        assert result == expected
```