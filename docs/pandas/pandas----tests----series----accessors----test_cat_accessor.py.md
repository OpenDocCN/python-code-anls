# `D:\src\scipysrc\pandas\pandas\tests\series\accessors\test_cat_accessor.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas import (  # 从 pandas 库中导入多个模块和类
    Categorical,  # 用于处理分类数据的类
    DataFrame,  # 用于处理二维表格数据的类
    Index,  # pandas 中的索引类
    Series,  # 用于处理一维序列数据的类
    Timestamp,  # 表示时间戳的类
    date_range,  # 生成日期范围的函数
    period_range,  # 生成周期范围的函数
    timedelta_range,  # 生成时间间隔范围的函数
)
import pandas._testing as tm  # 导入 pandas 测试工具，别名为 tm
from pandas.core.arrays.categorical import CategoricalAccessor  # 导入分类数据访问器
from pandas.core.indexes.accessors import Properties  # 导入索引属性访问器


class TestCatAccessor:
    @pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，定义测试参数
        "method",  # 参数名
        [  # 参数值列表，包含多个 Lambda 函数
            lambda x: x.cat.set_categories([1, 2, 3]),  # 设置分类数据的类别
            lambda x: x.cat.reorder_categories([2, 3, 1], ordered=True),  # 重新排序分类数据的类别
            lambda x: x.cat.rename_categories([1, 2, 3]),  # 重命名分类数据的类别
            lambda x: x.cat.remove_unused_categories(),  # 移除未使用的分类类别
            lambda x: x.cat.remove_categories([2]),  # 移除指定的分类类别
            lambda x: x.cat.add_categories([4]),  # 添加新的分类类别
            lambda x: x.cat.as_ordered(),  # 将分类数据标记为有序
            lambda x: x.cat.as_unordered(),  # 将分类数据标记为无序
        ],
    )
    def test_getname_categorical_accessor(self, method):  # 测试分类数据访问器的方法
        # GH#17509
        ser = Series([1, 2, 3], name="A").astype("category")  # 创建一个分类类型的 Series 对象
        expected = "A"  # 预期结果为字符串 "A"
        result = method(ser).name  # 调用指定的分类方法，并获取其名称属性
        assert result == expected  # 断言结果与预期相符

    def test_cat_accessor(self):
        ser = Series(Categorical(["a", "b", np.nan, "a"]))  # 创建一个包含分类数据的 Series 对象
        tm.assert_index_equal(ser.cat.categories, Index(["a", "b"]))  # 使用测试工具检查分类的类别
        assert not ser.cat.ordered, False  # 断言分类数据不是有序的

        exp = Categorical(["a", "b", np.nan, "a"], categories=["b", "a"])  # 创建预期的分类数据对象

        res = ser.cat.set_categories(["b", "a"])  # 设置新的分类类别
        tm.assert_categorical_equal(res.values, exp)  # 使用测试工具检查分类数据的一致性

        ser[:] = "a"  # 将所有值设置为 "a"
        ser = ser.cat.remove_unused_categories()  # 移除未使用的分类类别
        tm.assert_index_equal(ser.cat.categories, Index(["a"]))  # 使用测试工具检查分类的类别

    def test_cat_accessor_api(self):
        # GH#9322

        assert Series.cat is CategoricalAccessor  # 断言 Series 的 cat 属性是 CategoricalAccessor 类
        ser = Series(list("aabbcde")).astype("category")  # 创建一个包含分类数据的 Series 对象
        assert isinstance(ser.cat, CategoricalAccessor)  # 断言 ser.cat 是 CategoricalAccessor 的实例

        invalid = Series([1])  # 创建一个不包含分类数据的 Series 对象
        with pytest.raises(AttributeError, match="only use .cat accessor"):  # 捕获预期的 AttributeError 异常
            invalid.cat  # 尝试访问不存在的 cat 属性
        assert not hasattr(invalid, "cat")  # 断言 invalid 对象没有 cat 属性

    def test_cat_accessor_no_new_attributes(self):
        # https://github.com/pandas-dev/pandas/issues/10673
        cat = Series(list("aabbcde")).astype("category")  # 创建一个包含分类数据的 Series 对象
        with pytest.raises(AttributeError, match="You cannot add any new attribute"):  # 捕获预期的 AttributeError 异常
            cat.cat.xlabel = "a"  # 尝试给 cat 属性添加新的属性

    @pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，定义测试参数
        "idx",  # 参数名
        [  # 参数值列表，包含多个日期、周期和时间间隔范围
            date_range("1/1/2015", periods=5),  # 生成日期范围
            date_range("1/1/2015", periods=5, tz="MET"),  # 生成带时区的日期范围
            period_range("1/1/2015", freq="D", periods=5),  # 生成周期范围
            timedelta_range("1 days", "10 days"),  # 生成时间间隔范围
        ],
    )
    # 测试用例：检查对分类数据的日期时间访问器 API 的行为
    def test_dt_accessor_api_for_categorical(self, idx):
        # 引用问题跟踪 GitHub Issue 10661
        # 创建一个 Series 对象
        ser = Series(idx)
        # 将 Series 对象转换为分类数据类型
        cat = ser.astype("category")

        # 获取日期时间操作的属性名列表
        attr_names = type(ser._values)._datetimelike_ops

        # 断言分类数据的 .dt 属性是 Properties 类的实例
        assert isinstance(cat.dt, Properties)

        # 定义特殊函数的参数列表
        special_func_defs = [
            ("strftime", ("%Y-%m-%d",), {}),
            ("round", ("D",), {}),
            ("floor", ("D",), {}),
            ("ceil", ("D",), {}),
            ("asfreq", ("D",), {}),
            ("as_unit", ("s"), {}),
        ]

        # 根据 idx 的数据类型决定是否添加特殊函数
        if idx.dtype == "M8[ns]":
            # 排除已经本地化的 dt64tz，因为会引发异常
            tup = ("tz_localize", ("UTC",), {})
            special_func_defs.append(tup)
        elif idx.dtype.kind == "M":
            # 排除未本地化的 dt64，因为会引发异常
            tup = ("tz_convert", ("EST",), {})
            special_func_defs.append(tup)

        # 获取特殊函数名列表
        _special_func_names = [f[0] for f in special_func_defs]

        # 忽略的函数名列表
        _ignore_names = ["components", "tz_localize", "tz_convert"]

        # 获取可用的函数名列表，过滤掉私有函数、属性名、特殊函数和忽略的函数
        func_names = [
            fname
            for fname in dir(ser.dt)
            if not (
                fname.startswith("_")
                or fname in attr_names
                or fname in _special_func_names
                or fname in _ignore_names
            )
        ]

        # 根据函数名列表生成函数定义列表
        func_defs = [(fname, (), {}) for fname in func_names]
        func_defs.extend(
            f_def for f_def in special_func_defs if f_def[0] in dir(ser.dt)
        )

        # 遍历函数定义列表进行测试
        for func, args, kwargs in func_defs:
            warn_cls = []

            # 特定条件下给出警告类别
            if func == "to_period" and getattr(idx, "tz", None) is not None:
                # 去除时区信息时警告
                warn_cls.append(UserWarning)
            elif func == "to_pytimedelta":
                # GH 57463
                warn_cls.append(FutureWarning)

            # 将警告类别转换为元组或者置为空
            if warn_cls:
                warn_cls = tuple(warn_cls)
            else:
                warn_cls = None

            # 使用 pytest 的断言确保警告正常产生
            with tm.assert_produces_warning(warn_cls):
                res = getattr(cat.dt, func)(*args, **kwargs)
                exp = getattr(ser.dt, func)(*args, **kwargs)

            # 使用 pytest 的断言检查结果是否与期望相等
            tm.assert_equal(res, exp)

        # 对于日期时间操作属性，逐个进行断言检查
        for attr in attr_names:
            res = getattr(cat.dt, attr)
            exp = getattr(ser.dt, attr)

            # 使用 pytest 的断言检查结果是否与期望相等
            tm.assert_equal(res, exp)

    # 测试用例：检查对分类数据的非法日期时间访问器 API 的行为
    def test_dt_accessor_api_for_categorical_invalid(self):
        # 创建一个包含非日期时间数据的 Series 对象，并将其转换为分类数据类型
        invalid = Series([1, 2, 3]).astype("category")
        # 预期抛出的错误消息
        msg = "Can only use .dt accessor with datetimelike"

        # 使用 pytest 的断言检查是否抛出指定类型的异常，并匹配特定的错误消息
        with pytest.raises(AttributeError, match=msg):
            invalid.dt

        # 使用普通的断言检查是否没有名为 "str" 的属性
        assert not hasattr(invalid, "str")
    # 定义一个测试函数，用于测试设置类别的项目
    def test_set_categories_setitem(self):
        # GH#43334

        # 创建一个包含两列的 DataFrame 对象，其中 "Survived" 和 "Sex" 列的数据类型为 category
        df = DataFrame({"Survived": [1, 0, 1], "Sex": [0, 1, 1]}, dtype="category")

        # 使用 df["Survived"] 的分类属性，将其类别重命名为 ["No", "Yes"]
        df["Survived"] = df["Survived"].cat.rename_categories(["No", "Yes"])
        # 使用 df["Sex"] 的分类属性，将其类别重命名为 ["female", "male"]
        df["Sex"] = df["Sex"].cat.rename_categories(["female", "male"])

        # 检查值是否未被强制转换为 NaN
        assert list(df["Sex"]) == ["female", "male", "male"]
        assert list(df["Survived"]) == ["Yes", "No", "Yes"]

        # 使用 Categorical 对象重新设置 df["Sex"] 的类别为 ["female", "male"]，有序标记为 False
        df["Sex"] = Categorical(df["Sex"], categories=["female", "male"], ordered=False)
        # 使用 Categorical 对象重新设置 df["Survived"] 的类别为 ["No", "Yes"]，有序标记为 False
        df["Survived"] = Categorical(
            df["Survived"], categories=["No", "Yes"], ordered=False
        )

        # 再次检查值是否未被强制转换为 NaN
        assert list(df["Sex"]) == ["female", "male", "male"]
        assert list(df["Survived"]) == ["Yes", "No", "Yes"]

    # 定义一个测试函数，验证布尔值的分类是否为布尔类型
    def test_categorical_of_booleans_is_boolean(self):
        # https://github.com/pandas-dev/pandas/issues/46313

        # 创建一个包含两列的 DataFrame 对象，其中 "int_cat" 列为整数，"bool_cat" 列为布尔类型的 category
        df = DataFrame(
            {"int_cat": [1, 2, 3], "bool_cat": [True, False, False]}, dtype="category"
        )
        # 获取 df["bool_cat"] 列的分类属性的数据类型
        value = df["bool_cat"].cat.categories.dtype
        # 期望的数据类型为 numpy 的布尔类型
        expected = np.dtype(np.bool_)
        # 断言获取的数据类型与期望的数据类型相同
        assert value is expected
```