# `D:\src\scipysrc\pandas\pandas\tests\generic\test_generic.py`

```
from copy import (
    copy,  # 导入 copy 函数
    deepcopy,  # 导入 deepcopy 函数
)

import numpy as np  # 导入 numpy 库，并重命名为 np
import pytest  # 导入 pytest 测试框架

from pandas.core.dtypes.common import is_scalar  # 从 pandas 库中导入 is_scalar 函数

from pandas import (  # 从 pandas 库中导入多个类和函数
    DataFrame,  # 导入 DataFrame 类
    Index,  # 导入 Index 类
    Series,  # 导入 Series 类
    date_range,  # 导入 date_range 函数
)
import pandas._testing as tm  # 导入 pandas._testing 模块，并重命名为 tm

# ----------------------------------------------------------------------
# Generic types test cases


def construct(box, shape, value=None, dtype=None, **kwargs):
    """
    construct an object for the given shape
    if value is specified use that if its a scalar
    if value is an array, repeat it as needed
    """
    if isinstance(shape, int):
        shape = tuple([shape] * box._AXIS_LEN)  # 将 shape 转换为元组，元素为 shape 长度的 box._AXIS_LEN 倍

    if value is not None:
        if is_scalar(value):  # 如果 value 是标量
            if value == "empty":  # 如果 value 是 "empty"
                arr = None  # 将 arr 设为 None
                dtype = np.float64  # 将 dtype 设为 np.float64

                # 移除 info axis 的信息
                kwargs.pop(box._info_axis_name, None)
            else:
                arr = np.empty(shape, dtype=dtype)  # 创建一个空数组 arr，指定形状和数据类型
                arr.fill(value)  # 用 value 填充 arr
        else:
            fshape = np.prod(shape)  # 计算 shape 的所有元素的乘积
            arr = value.ravel()  # 将 value 展平为一维数组
            new_shape = fshape / arr.shape[0]  # 计算新形状
            if fshape % arr.shape[0] != 0:  # 如果不能整除
                raise Exception("invalid value passed in construct")  # 抛出异常，传入 construct 函数的值无效

            arr = np.repeat(arr, new_shape).reshape(shape)  # 重复数组 arr 并重塑为指定形状
    else:
        arr = np.random.default_rng(2).standard_normal(shape)  # 使用标准正态分布生成指定形状的随机数组
    return box(arr, dtype=dtype, **kwargs)  # 使用 box 构造对象，并返回该对象


class TestGeneric:
    @pytest.mark.parametrize(
        "func",
        [
            str.lower,  # 字符串转小写函数
            {x: x.lower() for x in list("ABCD")},  # 字典推导式，将字符转小写
            Series({x: x.lower() for x in list("ABCD")}),  # 创建 Series 对象，键为大写字母，值为小写字母
        ],
    )
    def test_rename(self, frame_or_series, func):
        # single axis
        idx = list("ABCD")  # 创建字符列表 ['A', 'B', 'C', 'D']

        for axis in frame_or_series._AXIS_ORDERS:  # 遍历 frame_or_series 的 _AXIS_ORDERS
            kwargs = {axis: idx}  # 创建关键字参数字典
            obj = construct(frame_or_series, 4, **kwargs)  # 构造对象 obj

            # rename a single axis
            result = obj.rename(**{axis: func})  # 对 obj 的一个轴进行重命名
            expected = obj.copy()  # 复制 obj 到 expected
            setattr(expected, axis, list("abcd"))  # 设置 expected 的轴属性为 ['a', 'b', 'c', 'd']
            tm.assert_equal(result, expected)  # 使用 tm 模块的 assert_equal 函数断言 result 等于 expected

    def test_get_numeric_data(self, frame_or_series):
        n = 4  # 设定 n 为 4
        kwargs = {
            frame_or_series._get_axis_name(i): list(range(n))  # 创建关键字参数字典，键为轴名，值为 [0, 1, 2, 3]
            for i in range(frame_or_series._AXIS_LEN)  # 遍历 frame_or_series 的轴长度
        }

        # get the numeric data
        o = construct(frame_or_series, n, **kwargs)  # 构造对象 o
        result = o._get_numeric_data()  # 获取 o 的数值数据
        tm.assert_equal(result, o)  # 使用 tm 模块的 assert_equal 函数断言 result 等于 o

        # non-inclusion
        result = o._get_bool_data()  # 获取 o 的布尔数据
        expected = construct(frame_or_series, n, value="empty", **kwargs)  # 构造预期的对象 expected，值为 "empty"
        if isinstance(o, DataFrame):  # 如果 o 是 DataFrame 类型
            # preserve columns dtype
            expected.columns = o.columns[:0]  # 保留列的数据类型

        tm.assert_equal(result, expected)  # 使用 tm 模块的 assert_equal 函数断言 result 等于 expected

        # get the bool data
        arr = np.array([True, True, False, True])  # 创建布尔数组 arr
        o = construct(frame_or_series, n, value=arr, **kwargs)  # 构造对象 o，值为 arr
        result = o._get_numeric_data()  # 获取 o 的数值数据
        tm.assert_equal(result, o)  # 使用 tm 模块的 assert_equal 函数断言 result 等于 o
    def test_get_bool_data_empty_preserve_index(self):
        # 创建一个空的 Series 对象作为预期结果，数据类型为 bool
        expected = Series([], dtype="bool")
        # 调用 Series 对象的 _get_bool_data 方法，获取结果
        result = expected._get_bool_data()
        # 使用 pytest 模块的 assert_series_equal 函数比较 result 和 expected，确保它们相等，并检查索引类型
        tm.assert_series_equal(result, expected, check_index_type=True)

    def test_nonzero(self, frame_or_series):
        # GH 4633
        # 测试对象在布尔/非零行为上的表现
        obj = construct(frame_or_series, shape=4)
        msg = f"The truth value of a {frame_or_series.__name__} is ambiguous"
        # 使用 pytest 模块的 raises 函数确保比较 obj == 0、obj == 1、以及 obj 都会抛出 ValueError 异常，并且异常消息匹配 msg
        with pytest.raises(ValueError, match=msg):
            bool(obj == 0)
        with pytest.raises(ValueError, match=msg):
            bool(obj == 1)
        with pytest.raises(ValueError, match=msg):
            bool(obj)

        obj = construct(frame_or_series, shape=4, value=1)
        # 同上，确保比较 obj == 0、obj == 1、以及 obj 都会抛出 ValueError 异常，并且异常消息匹配 msg
        with pytest.raises(ValueError, match=msg):
            bool(obj == 0)
        with pytest.raises(ValueError, match=msg):
            bool(obj == 1)
        with pytest.raises(ValueError, match=msg):
            bool(obj)

        obj = construct(frame_or_series, shape=4, value=np.nan)
        # 同上，确保比较 obj == 0、obj == 1、以及 obj 都会抛出 ValueError 异常，并且异常消息匹配 msg
        with pytest.raises(ValueError, match=msg):
            bool(obj == 0)
        with pytest.raises(ValueError, match=msg):
            bool(obj == 1)
        with pytest.raises(ValueError, match=msg):
            bool(obj)

        # 空对象
        obj = construct(frame_or_series, shape=0)
        # 确保比较 obj 会抛出 ValueError 异常，并且异常消息匹配 msg
        with pytest.raises(ValueError, match=msg):
            bool(obj)

        # 无效行为

        obj1 = construct(frame_or_series, shape=4, value=1)
        obj2 = construct(frame_or_series, shape=4, value=1)

        # 确保 obj1 作为条件表达式时会抛出 ValueError 异常，并且异常消息匹配 msg
        with pytest.raises(ValueError, match=msg):
            if obj1:
                pass

        # 确保 obj1 and obj2 会抛出 ValueError 异常，并且异常消息匹配 msg
        with pytest.raises(ValueError, match=msg):
            obj1 and obj2
        # 确保 obj1 or obj2 会抛出 ValueError 异常，并且异常消息匹配 msg
        with pytest.raises(ValueError, match=msg):
            obj1 or obj2
        # 确保 not obj1 会抛出 ValueError 异常，并且异常消息匹配 msg
        with pytest.raises(ValueError, match=msg):
            not obj1

    def test_frame_or_series_compound_dtypes(self, frame_or_series):
        # see gh-5191
        # 复合数据类型应该抛出 NotImplementedError 异常

        def f(dtype):
            # 调用 construct 函数创建一个 frame_or_series 对象，形状为 3，值为 1，数据类型为 dtype
            return construct(frame_or_series, shape=3, value=1, dtype=dtype)

        # 错误消息
        msg = (
            "compound dtypes are not implemented "
            f"in the {frame_or_series.__name__} constructor"
        )

        # 确保调用 f 函数时，使用复合数据类型会抛出 NotImplementedError 异常，并且异常消息匹配 msg
        with pytest.raises(NotImplementedError, match=msg):
            f([("A", "datetime64[h]"), ("B", "str"), ("C", "int32")])

        # 这些调用应该成功（尽管结果可能不符合预期）
        f("int64")
        f("float64")
        f("M8[ns]")
    # 检查元数据在结果操作中是否匹配

    # 使用给定的 frame_or_series 构造对象 o 和 o2，并设置它们的形状为 3
    o = construct(frame_or_series, shape=3)
    o.name = "foo"
    o2 = construct(frame_or_series, shape=3)
    o2.name = "bar"

    # ----------
    # 保持不变
    # ----------

    # 对于简单的标量操作
    for op in ["__add__", "__sub__", "__truediv__", "__mul__"]:
        # 执行 o 对象的操作 op，并将结果存储在 result 中
        result = getattr(o, op)(1)
        # 断言 o 和 result 的元数据是等效的
        tm.assert_metadata_equivalent(o, result)

    # 对于与自身类似对象的操作
    for op in ["__add__", "__sub__", "__truediv__", "__mul__"]:
        # 执行 o 对象与自身的操作 op，并将结果存储在 result 中
        result = getattr(o, op)(o)
        # 断言 o 和 result 的元数据是等效的
        tm.assert_metadata_equivalent(o, result)

    # 简单的布尔运算
    for op in ["__eq__", "__le__", "__ge__"]:
        # 执行 o 对象的操作 op，并将结果存储在 v1 中
        v1 = getattr(o, op)(o)
        # 断言 o 和 v1 的元数据是等效的
        tm.assert_metadata_equivalent(o, v1)
        tm.assert_metadata_equivalent(o, v1 & v1)  # 逻辑与操作
        tm.assert_metadata_equivalent(o, v1 | v1)  # 逻辑或操作

    # combine_first 操作
    result = o.combine_first(o2)
    # 断言 o 和 result 的元数据是等效的
    tm.assert_metadata_equivalent(o, result)

    # ---------------------------
    # 非保持（默认情况下）
    # ---------------------------

    # 非类似对象的加法操作
    result = o + o2
    # 断言结果的元数据与 o 的元数据是等效的
    tm.assert_metadata_equivalent(result)

    # 简单的布尔运算
    for op in ["__eq__", "__le__", "__ge__"]:
        # 执行 o 对象的操作 op 与 o2，并将结果存储在 v1 和 v2 中
        # 这是一个名称匹配操作
        v1 = getattr(o, op)(o)
        v2 = getattr(o, op)(o2)
        # 断言 v2 的元数据与 o 的元数据是等效的
        tm.assert_metadata_equivalent(v2)
        tm.assert_metadata_equivalent(v1 & v2)  # 逻辑与操作
        tm.assert_metadata_equivalent(v1 | v2)  # 逻辑或操作

def test_size_compat(self, frame_or_series):
    # GH8846
    # size 属性应该被定义

    # 使用给定的 frame_or_series 构造对象 o，并设置其形状为 10
    o = construct(frame_or_series, shape=10)
    # 断言 o 的 size 等于其形状各维度大小的乘积
    assert o.size == np.prod(o.shape)
    assert o.size == 10 ** len(o.axes)

def test_split_compat(self, frame_or_series):
    # xref GH8846
    # 使用给定的 frame_or_series 构造对象 o，并设置其形状为 10
    o = construct(frame_or_series, shape=10)
    # 断言将 o 分割成 5 部分后的长度为 5
    assert len(np.array_split(o, 5)) == 5
    # 断言将 o 分割成 2 部分后的长度为 2
    assert len(np.array_split(o, 2)) == 2

# See gh-12301
def test_stat_unexpected_keyword(self, frame_or_series):
    # 使用给定的 frame_or_series 构造对象 obj，并设置其形状为 5
    obj = construct(frame_or_series, 5)
    starwars = "Star Wars"
    errmsg = "unexpected keyword"

    # 使用 pytest 检查是否会引发 TypeError 异常，并匹配错误消息 errmsg
    with pytest.raises(TypeError, match=errmsg):
        obj.max(epic=starwars)  # stat_function
    with pytest.raises(TypeError, match=errmsg):
        obj.var(epic=starwars)  # stat_function_ddof
    with pytest.raises(TypeError, match=errmsg):
        obj.sum(epic=starwars)  # cum_function
    with pytest.raises(TypeError, match=errmsg):
        obj.any(epic=starwars)  # logical_function

@pytest.mark.parametrize("func", ["sum", "cumsum", "any", "var"])
    # 测试函数，用于验证 API 的兼容性
    def test_api_compat(self, func, frame_or_series):
        # GH 12021
        # 检查 __name__, __qualname__ 的兼容性

        # 根据给定的 frame_or_series 构造对象
        obj = construct(frame_or_series, 5)
        # 获取对象的特定方法 func
        f = getattr(obj, func)
        # 断言方法的名称与给定的 func 相同
        assert f.__name__ == func
        # 断言方法的限定名称以 func 结尾
        assert f.__qualname__.endswith(func)

    # 测试统计函数的非默认参数
    def test_stat_non_defaults_args(self, frame_or_series):
        # 根据给定的 frame_or_series 构造对象
        obj = construct(frame_or_series, 5)
        # 定义一个输出数组
        out = np.array([0])
        # 定义错误消息
        errmsg = "the 'out' parameter is not supported"

        # 使用 pytest 来验证各个统计函数的参数 out 是否触发 ValueError 异常
        with pytest.raises(ValueError, match=errmsg):
            obj.max(out=out)  # stat_function
        with pytest.raises(ValueError, match=errmsg):
            obj.var(out=out)  # stat_function_ddof
        with pytest.raises(ValueError, match=errmsg):
            obj.sum(out=out)  # cum_function
        with pytest.raises(ValueError, match=errmsg):
            obj.any(out=out)  # logical_function

    # 测试 truncate 方法在超出边界时的行为
    def test_truncate_out_of_bounds(self, frame_or_series):
        # GH11382

        # 对于小型数据
        # 构造一个形状为 [2000, 1, ..., 1] 的数据对象
        shape = [2000] + ([1] * (frame_or_series._AXIS_LEN - 1))
        small = construct(frame_or_series, shape, dtype="int8", value=1)
        # 断言 truncate 方法的行为符合预期
        tm.assert_equal(small.truncate(), small)
        tm.assert_equal(small.truncate(before=0, after=3e3), small)
        tm.assert_equal(small.truncate(before=-1, after=2e3), small)

        # 对于大型数据
        # 构造一个形状为 [2_000_000, 1, ..., 1] 的数据对象
        shape = [2_000_000] + ([1] * (frame_or_series._AXIS_LEN - 1))
        big = construct(frame_or_series, shape, dtype="int8", value=1)
        # 断言 truncate 方法的行为符合预期
        tm.assert_equal(big.truncate(), big)
        tm.assert_equal(big.truncate(before=0, after=3e6), big)
        tm.assert_equal(big.truncate(before=-1, after=2e6), big)

    # 参数化测试，验证复制和深度复制函数的行为
    @pytest.mark.parametrize(
        "func",
        [copy, deepcopy, lambda x: x.copy(deep=False), lambda x: x.copy(deep=True)],
    )
    @pytest.mark.parametrize("shape", [0, 1, 2])
    def test_copy_and_deepcopy(self, frame_or_series, shape, func):
        # GH 15444
        # 根据给定的 frame_or_series 和 shape 构造对象
        obj = construct(frame_or_series, shape)
        # 使用 func 对象复制 obj
        obj_copy = func(obj)
        # 断言复制后的对象与原对象不同
        assert obj_copy is not obj
        # 断言复制后的对象与原对象相等
        tm.assert_equal(obj_copy, obj)
class TestNDFrame:
    # tests that don't fit elsewhere

    @pytest.mark.parametrize(
        "ser",
        [
            Series(range(10), dtype=np.float64),  # 创建一个包含浮点数序列的 Series 对象
            Series([str(i) for i in range(10)], dtype=object),  # 创建一个包含字符串的 Series 对象
        ],
    )
    def test_squeeze_series_noop(self, ser):
        # noop，对序列进行 squeeze 操作，不改变结果
        tm.assert_series_equal(ser.squeeze(), ser)

    def test_squeeze_frame_noop(self):
        # noop，对数据帧进行 squeeze 操作，不改变结果
        df = DataFrame(np.eye(2))
        tm.assert_frame_equal(df.squeeze(), df)

    def test_squeeze_frame_reindex(self):
        # squeezing，对数据帧进行 squeeze 操作，并重新索引
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        ).reindex(columns=["A"])
        tm.assert_series_equal(df.squeeze(), df["A"])

    def test_squeeze_0_len_dim(self):
        # don't fail with 0 length dimensions GH11229 & GH8999
        # 创建空 Series 和空数据帧，并对它们进行 squeeze 操作，确保不会出现错误
        empty_series = Series([], name="five", dtype=np.float64)
        empty_frame = DataFrame([empty_series])
        tm.assert_series_equal(empty_series, empty_series.squeeze())
        tm.assert_series_equal(empty_series, empty_frame.squeeze())

    def test_squeeze_axis(self):
        # axis argument，测试带有 axis 参数的 squeeze 操作
        df = DataFrame(
            np.random.default_rng(2).standard_normal((1, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=1, freq="B"),
        ).iloc[:, :1]
        assert df.shape == (1, 1)
        tm.assert_series_equal(df.squeeze(axis=0), df.iloc[0])
        tm.assert_series_equal(df.squeeze(axis="index"), df.iloc[0])
        tm.assert_series_equal(df.squeeze(axis=1), df.iloc[:, 0])
        tm.assert_series_equal(df.squeeze(axis="columns"), df.iloc[:, 0])
        assert df.squeeze() == df.iloc[0, 0]
        msg = "No axis named 2 for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            df.squeeze(axis=2)
        msg = "No axis named x for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            df.squeeze(axis="x")

    def test_squeeze_axis_len_3(self):
        # 对长度为 3 的数据帧进行 squeeze 操作，不改变结果
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=3, freq="B"),
        )
        tm.assert_frame_equal(df.squeeze(axis=0), df)

    def test_numpy_squeeze(self):
        # 测试使用 numpy.squeeze 对 Series 和数据帧进行 squeeze 操作
        s = Series(range(2), dtype=np.float64)
        tm.assert_series_equal(np.squeeze(s), s)

        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        ).reindex(columns=["A"])
        tm.assert_series_equal(np.squeeze(df), df["A"])
    @pytest.mark.parametrize(
        "ser",
        [
            Series(range(10), dtype=np.float64),  # 创建一个包含浮点数的 Pandas Series 对象
            Series([str(i) for i in range(10)], dtype=object),  # 创建一个包含字符串的 Pandas Series 对象
        ],
    )
    def test_transpose_series(self, ser):
        # 调用 pandas/core/base.py 中的实现
        tm.assert_series_equal(ser.transpose(), ser)  # 断言调用 Series 对象的 transpose 方法后与原对象相等

    def test_transpose_frame(self):
        # 创建一个包含随机数的 Pandas DataFrame 对象
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        # 断言连续两次对 DataFrame 对象调用 transpose 方法后与原对象相等
        tm.assert_frame_equal(df.transpose().transpose(), df)

    def test_numpy_transpose(self, frame_or_series):
        # 创建一个包含随机数的 Pandas DataFrame 对象或 Series 对象
        obj = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        obj = tm.get_obj(obj, frame_or_series)  # 获取符合 frame_or_series 类型的对象

        if frame_or_series is Series:
            # 对于 1 维数据，np.transpose 是一个空操作
            tm.assert_series_equal(np.transpose(obj), obj)

        # 断言 np.transpose 方法的双重应用后与原对象相等，即保持数据不变
        tm.assert_equal(np.transpose(np.transpose(obj)), obj)

        # 预期抛出 ValueError 异常，因为 'axes' 参数不受支持
        msg = "the 'axes' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.transpose(obj, axes=1)

    @pytest.mark.parametrize(
        "ser",
        [
            Series(range(10), dtype=np.float64),  # 创建一个包含浮点数的 Pandas Series 对象
            Series([str(i) for i in range(10)], dtype=object),  # 创建一个包含字符串的 Pandas Series 对象
        ],
    )
    def test_take_series(self, ser):
        indices = [1, 5, -2, 6, 3, -1]
        out = ser.take(indices)  # 根据给定索引从 Series 中获取对应元素
        expected = Series(
            data=ser.values.take(indices),  # 获取按索引取出的数据
            index=ser.index.take(indices),  # 获取按索引取出的索引
            dtype=ser.dtype,
        )
        tm.assert_series_equal(out, expected)  # 断言取出的结果与预期的 Series 对象相等

    def test_take_frame(self):
        indices = [1, 5, -2, 6, 3, -1]
        # 创建一个包含随机数的 Pandas DataFrame 对象
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        out = df.take(indices)  # 根据给定的索引从 DataFrame 中获取对应行
        expected = DataFrame(
            data=df.values.take(indices, axis=0),  # 获取按索引取出的行数据
            index=df.index.take(indices),  # 获取按索引取出的行索引
            columns=df.columns,
        )
        tm.assert_frame_equal(out, expected)  # 断言取出的结果与预期的 DataFrame 对象相等

    def test_take_invalid_kwargs(self, frame_or_series):
        indices = [-3, 2, 0, 1]

        obj = DataFrame(range(5))
        obj = tm.get_obj(obj, frame_or_series)  # 获取符合 frame_or_series 类型的对象

        # 预期抛出 TypeError 异常，因为 'foo' 是一个未知的关键字参数
        msg = r"take\(\) got an unexpected keyword argument 'foo'"
        with pytest.raises(TypeError, match=msg):
            obj.take(indices, foo=2)

        # 预期抛出 ValueError 异常，因为 'out' 参数不受支持
        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            obj.take(indices, out=indices)

        # 预期抛出 ValueError 异常，因为 'mode' 参数不受支持
        msg = "the 'mode' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            obj.take(indices, mode="clip")
    # 测试类方法，验证轴对象的行为
    def test_axis_classmethods(self, frame_or_series):
        # 将输入参数赋值给变量 box
        box = frame_or_series
        # 使用 box 创建一个对象 obj，类型为 object
        obj = box(dtype=object)
        # 获取 box 对象的轴到轴编号的映射的键集合
        values = box._AXIS_TO_AXIS_NUMBER.keys()
        # 遍历每个键
        for v in values:
            # 断言 obj 对象的 _get_axis_number 方法返回值与 box 对象的 _get_axis_number 方法返回值相等
            assert obj._get_axis_number(v) == box._get_axis_number(v)
            # 断言 obj 对象的 _get_axis_name 方法返回值与 box 对象的 _get_axis_name 方法返回值相等
            assert obj._get_axis_name(v) == box._get_axis_name(v)
            # 断言 obj 对象的 _get_block_manager_axis 方法返回值与 box 对象的 _get_block_manager_axis 方法返回值相等
            assert obj._get_block_manager_axis(v) == box._get_block_manager_axis(v)
    
    # 验证对象标志的一致性
    def test_flags_identity(self, frame_or_series):
        # 创建一个包含整数 1 和 2 的 Series 对象 obj
        obj = Series([1, 2])
        # 如果 frame_or_series 是 DataFrame 类型，则将 obj 转换为 DataFrame 对象
        if frame_or_series is DataFrame:
            obj = obj.to_frame()
    
        # 断言 obj 对象的标志属性（flags）与自身的标志属性相等
        assert obj.flags is obj.flags
        # 复制 obj 对象到 obj2
        obj2 = obj.copy()
        # 断言 obj2 对象的标志属性（flags）与 obj 对象的标志属性（flags）不相等
        assert obj2.flags is not obj.flags
```