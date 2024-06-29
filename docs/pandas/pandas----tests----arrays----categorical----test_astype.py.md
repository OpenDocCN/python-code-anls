# `D:\src\scipysrc\pandas\pandas\tests\arrays\categorical\test_astype.py`

```
# 导入所需的库
import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 测试框架

from pandas import (  # 导入 pandas 库中的多个类和函数
    Categorical,
    CategoricalDtype,
    CategoricalIndex,
    DatetimeIndex,
    Interval,
    NaT,
    Period,
    Timestamp,
    array,
    to_datetime,
)
import pandas._testing as tm  # 导入 pandas 内部测试模块

# 定义测试类 TestAstype
class TestAstype:

    # 使用 pytest.mark.parametrize 注册参数化测试
    @pytest.mark.parametrize("cls", [Categorical, CategoricalIndex])
    @pytest.mark.parametrize("values", [[1, np.nan], [Timestamp("2000"), NaT]])
    def test_astype_nan_to_int(self, cls, values):
        # GH#28406
        # 使用给定的类和值创建对象
        obj = cls(values)

        msg = "Cannot (cast|convert)"
        # 使用 pytest.raises 检查是否抛出指定异常
        with pytest.raises((ValueError, TypeError), match=msg):
            obj.astype(int)

    # 使用 pytest.mark.parametrize 注册参数化测试
    @pytest.mark.parametrize(
        "expected",
        [
            array(["2019", "2020"], dtype="datetime64[ns, UTC]"),
            array([0, 0], dtype="timedelta64[ns]"),
            array([Period("2019"), Period("2020")], dtype="period[Y-DEC]"),
            array([Interval(0, 1), Interval(1, 2)], dtype="interval"),
            array([1, np.nan], dtype="Int64"),
        ],
    )
    def test_astype_category_to_extension_dtype(self, expected):
        # GH#28668
        # 将预期结果先转换为 "category" 类型，再转换回原来的类型
        result = expected.astype("category").astype(expected.dtype)

        # 使用 tm.assert_extension_array_equal 检查结果是否与预期相等
        tm.assert_extension_array_equal(result, expected)

    # 使用 pytest.mark.parametrize 注册参数化测试
    @pytest.mark.parametrize(
        "dtype, expected",
        [
            (
                "datetime64[ns]",
                np.array(["2015-01-01T00:00:00.000000000"], dtype="datetime64[ns]"),
            ),
            (
                "datetime64[ns, MET]",
                DatetimeIndex([Timestamp("2015-01-01 00:00:00+0100", tz="MET")]).array,
            ),
        ],
    )
    def test_astype_to_datetime64(self, dtype, expected):
        # GH#28448
        # 将字符串数组转换为指定的 datetime64 类型
        result = Categorical(["2015-01-01"]).astype(dtype)
        # 使用断言检查结果是否与预期相等
        assert result == expected

    # 测试将字符串和整数类别转换为可空整数
    def test_astype_str_int_categories_to_nullable_int(self):
        # GH#39616
        # 创建包含指定类别的 dtype 对象
        dtype = CategoricalDtype([str(i) for i in range(5)])
        # 使用随机数生成器创建随机整数 codes
        codes = np.random.default_rng(2).integers(5, size=20)
        # 使用指定的 dtype 创建分类数组
        arr = Categorical.from_codes(codes, dtype=dtype)

        # 将分类数组转换为 "Int64" 类型
        res = arr.astype("Int64")
        expected = array(codes, dtype="Int64")
        # 使用 tm.assert_extension_array_equal 检查结果是否与预期相等
        tm.assert_extension_array_equal(res, expected)

    # 测试将字符串和整数类别转换为可空浮点数
    def test_astype_str_int_categories_to_nullable_float(self):
        # GH#39616
        # 创建包含指定类别的 dtype 对象
        dtype = CategoricalDtype([str(i / 2) for i in range(5)])
        # 使用随机数生成器创建随机整数 codes
        codes = np.random.default_rng(2).integers(5, size=20)
        # 使用指定的 dtype 创建分类数组
        arr = Categorical.from_codes(codes, dtype=dtype)

        # 将分类数组转换为 "Float64" 类型
        res = arr.astype("Float64")
        expected = array(codes, dtype="Float64") / 2
        # 使用 tm.assert_extension_array_equal 检查结果是否与预期相等
        tm.assert_extension_array_equal(res, expected)
    # 定义测试方法 test_astype，接受一个参数 ordered
    def test_astype(self, ordered):
        # 创建一个有序或无序的分类数据对象 cat，其元素为字符串列表 "abbaaccc"
        cat = Categorical(list("abbaaccc"), ordered=ordered)
        # 将 cat 转换为 object 类型的数组并赋值给 result
        result = cat.astype(object)
        # 使用 np.array 创建一个期望的数组，类型与 cat 相同
        expected = np.array(cat)
        # 断言 result 与 expected 数组内容相等
        tm.assert_numpy_array_equal(result, expected)

        # 设置错误消息，用于捕获 ValueError 异常
        msg = r"Cannot cast object|string dtype to float64"
        # 使用 pytest 的 raises 函数，检查是否抛出 ValueError 异常，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            cat.astype(float)

        # 创建一个包含数字的分类数据对象 cat，元素为列表 [0, 1, 2, 2, 1, 0, 1, 0, 2]
        cat = Categorical([0, 1, 2, 2, 1, 0, 1, 0, 2], ordered=ordered)
        # 将 cat 转换为 object 类型的数组并赋值给 result
        result = cat.astype(object)
        # 使用 np.array 创建一个期望的数组，类型为 object，与 cat 相同
        expected = np.array(cat, dtype=object)
        # 断言 result 与 expected 数组内容相等
        tm.assert_numpy_array_equal(result, expected)

        # 将 cat 转换为 int 类型的数组并赋值给 result
        result = cat.astype(int)
        # 使用 np.array 创建一个期望的数组，类型为 int，与 cat 相同
        expected = np.array(cat, dtype="int")
        # 断言 result 与 expected 数组内容相等
        tm.assert_numpy_array_equal(result, expected)

        # 将 cat 转换为 float 类型的数组并赋值给 result
        result = cat.astype(float)
        # 使用 np.array 创建一个期望的数组，类型为 float，与 cat 相同
        expected = np.array(cat, dtype=float)
        # 断言 result 与 expected 数组内容相等
        tm.assert_numpy_array_equal(result, expected)

    # 使用 pytest 的 parametrize 装饰器定义多个参数化的测试用例
    @pytest.mark.parametrize("dtype_ordered", [True, False])
    def test_astype_category(self, dtype_ordered, ordered):
        # GH#10696/GH#18593
        # 创建一个字符列表 data，包含 "abcaacbab"
        data = list("abcaacbab")
        # 使用指定的类别列表创建一个有序或无序的分类数据对象 cat
        cat = Categorical(data, categories=list("bac"), ordered=ordered)

        # 创建一个 CategoricalDtype 对象，指定是否有序
        dtype = CategoricalDtype(ordered=dtype_ordered)
        # 将 cat 转换为指定 dtype 类型的分类数据对象，并赋值给 result
        result = cat.astype(dtype)
        # 创建一个期望的分类数据对象 expected，与 result 相同
        expected = Categorical(data, categories=cat.categories, ordered=dtype_ordered)
        # 断言 result 与 expected 对象相等
        tm.assert_categorical_equal(result, expected)

        # 创建一个自定义类别顺序的 CategoricalDtype 对象
        dtype = CategoricalDtype(list("adc"), dtype_ordered)
        # 将 cat 转换为指定 dtype 类型的分类数据对象，并赋值给 result
        result = cat.astype(dtype)
        # 创建一个期望的分类数据对象 expected，与 result 相同
        expected = Categorical(data, dtype=dtype)
        # 断言 result 与 expected 对象相等
        tm.assert_categorical_equal(result, expected)

        # 如果 dtype_ordered 是 False
        if dtype_ordered is False:
            # 将 cat 转换为 "category" 类型的分类数据对象，并赋值给 result
            result = cat.astype("category")
            # 创建一个期望的分类数据对象 expected，与 cat 相同
            expected = cat
            # 断言 result 与 expected 对象相等
            tm.assert_categorical_equal(result, expected)

    # 定义测试方法 test_astype_object_datetime_categories
    def test_astype_object_datetime_categories(self):
        # GH#40754
        # 创建一个包含日期时间数据的分类数据对象 cat
        cat = Categorical(to_datetime(["2021-03-27", NaT]))
        # 将 cat 转换为 object 类型的数组并赋值给 result
        result = cat.astype(object)
        # 创建一个期望的数组 expected，包含日期时间对象
        expected = np.array([Timestamp("2021-03-27 00:00:00"), NaT], dtype="object")
        # 断言 result 与 expected 数组内容相等
        tm.assert_numpy_array_equal(result, expected)

    # 定义测试方法 test_astype_object_timestamp_categories
    def test_astype_object_timestamp_categories(self):
        # GH#18024
        # 创建一个包含时间戳数据的分类数据对象 cat
        cat = Categorical([Timestamp("2014-01-01")])
        # 将 cat 转换为 object 类型的数组并赋值给 result
        result = cat.astype(object)
        # 创建一个期望的数组 expected，包含时间戳对象
        expected = np.array([Timestamp("2014-01-01 00:00:00")], dtype="object")
        # 断言 result 与 expected 数组内容相等
        tm.assert_numpy_array_equal(result, expected)

    # 定义测试方法 test_astype_category_readonly_mask_values
    def test_astype_category_readonly_mask_values(self):
        # GH#53658
        # 创建一个整数数组 arr，包含 [0, 1, 2]，类型为 "Int64"
        arr = array([0, 1, 2], dtype="Int64")
        # 设置 arr._mask 的 WRITEABLE 标志为 False，表示 arr._mask 是只读的
        arr._mask.flags["WRITEABLE"] = False
        # 将 arr 转换为 "category" 类型的数组并赋值给 result
        result = arr.astype("category")
        # 创建一个期望的数组 expected，类型为 "Int64"，转换为 "category" 类型
        expected = array([0, 1, 2], dtype="Int64").astype("category")
        # 断言 result 与 expected 数组内容相等
        tm.assert_extension_array_equal(result, expected)
```