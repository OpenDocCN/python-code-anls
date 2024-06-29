# `D:\src\scipysrc\pandas\pandas\tests\extension\base\dim2.py`

```
"""
Tests for 2D compatibility.
"""

# 导入必要的库
import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 库

# 导入 pandas 相关模块和函数
from pandas._libs.missing import is_matching_na  
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_integer_dtype,
)
import pandas as pd  # 导入 Pandas 库
import pandas._testing as tm  # 导入 Pandas 测试工具模块
from pandas.core.arrays.integer import NUMPY_INT_TO_DTYPE  # 导入整型数组相关函数

# 定义测试类 Dim2CompatTests
class Dim2CompatTests:
    # 注意：以下测试仅适用于支持二维数组的 ExtensionArray 子类
    # 即不适用于由 pyarrow 支持的 EAs.

    # 设置 pytest 的 fixture，如果不支持二维，则跳过测试
    @pytest.fixture(autouse=True)
    def skip_if_doesnt_support_2d(self, dtype, request):
        if not dtype._supports_2d:
            node = request.node
            # 对于混合到 ExtensionTests 中的情况，只跳过 Dim2CompatTests 中定义的测试
            test_func = node._obj
            if test_func.__qualname__.startswith("Dim2CompatTests"):
                # TODO: 是否有更少 hacky 的方法来检查这一点？
                pytest.skip(f"{dtype} does not support 2D.")

    # 测试数组转置功能
    def test_transpose(self, data):
        arr2d = data.repeat(2).reshape(-1, 2)
        shape = arr2d.shape
        assert shape[0] != shape[-1]  # 否则测试的其余部分无意义

        assert arr2d.T.shape == shape[::-1]

    # 测试从二维数组创建 DataFrame
    def test_frame_from_2d_array(self, data):
        arr2d = data.repeat(2).reshape(-1, 2)

        df = pd.DataFrame(arr2d)
        expected = pd.DataFrame({0: arr2d[:, 0], 1: arr2d[:, 1]})
        tm.assert_frame_equal(df, expected)

    # 测试数组轴对换功能
    def test_swapaxes(self, data):
        arr2d = data.repeat(2).reshape(-1, 2)

        result = arr2d.swapaxes(0, 1)
        expected = arr2d.T
        tm.assert_extension_array_equal(result, expected)

    # 测试删除二维数组中的元素
    def test_delete_2d(self, data):
        arr2d = data.repeat(3).reshape(-1, 3)

        # axis = 0
        result = arr2d.delete(1, axis=0)
        expected = data.delete(1).repeat(3).reshape(-1, 3)
        tm.assert_extension_array_equal(result, expected)

        # axis = 1
        result = arr2d.delete(1, axis=1)
        expected = data.repeat(2).reshape(-1, 2)
        tm.assert_extension_array_equal(result, expected)

    # 测试从二维数组中取值
    def test_take_2d(self, data):
        arr2d = data.reshape(-1, 1)

        result = arr2d.take([0, 0, -1], axis=0)

        expected = data.take([0, 0, -1]).reshape(-1, 1)
        tm.assert_extension_array_equal(result, expected)

    # 测试二维数组的字符串表示
    def test_repr_2d(self, data):
        # 在元素包含类型名的特殊情况下，可能会失败
        res = repr(data.reshape(1, -1))
        assert res.count(f"<{type(data).__name__}") == 1

        res = repr(data.reshape(-1, 1))
        assert res.count(f"<{type(data).__name__}") == 1
    # 定义测试函数，用于测试数据的reshape操作
    def test_reshape(self, data):
        # 将数据reshape为(-1, 1)，表示列数为1，行数自动计算
        arr2d = data.reshape(-1, 1)
        # 断言reshape后的形状为(data.size, 1)
        assert arr2d.shape == (data.size, 1)
        # 断言arr2d的长度与原始数据data的长度相同
        assert len(arr2d) == len(data)

        # 再次reshape为((-1, 1))，与上面的操作等价
        arr2d = data.reshape((-1, 1))
        # 断言reshape后的形状为(data.size, 1)
        assert arr2d.shape == (data.size, 1)
        # 再次断言arr2d的长度与原始数据data的长度相同
        assert len(arr2d) == len(data)

        # 使用tm.external_error_raised(ValueError)上下文，测试异常情况下的reshape
        with tm.external_error_raised(ValueError):
            data.reshape((data.size, 2))
        with tm.external_error_raised(ValueError):
            data.reshape(data.size, 2)

    # 定义测试函数，测试二维数组的getitem操作
    def test_getitem_2d(self, data):
        # 将data reshape为(1, -1)，即行数为1，列数自动计算
        arr2d = data.reshape(1, -1)

        # 测试获取单行数据，结果应与原始data相等
        result = arr2d[0]
        tm.assert_extension_array_equal(result, data)

        # 测试超出索引范围的情况，应该抛出IndexError异常
        with pytest.raises(IndexError):
            arr2d[1]
        with pytest.raises(IndexError):
            arr2d[-2]

        # 测试切片操作，结果应与原始arr2d相等
        result = arr2d[:]
        tm.assert_extension_array_equal(result, arr2d)

        # 测试二维切片操作，结果应与原始arr2d相等
        result = arr2d[:, :]
        tm.assert_extension_array_equal(result, arr2d)

        # 测试获取特定列的数据，结果应与原始data的第一行数据相等
        result = arr2d[:, 0]
        expected = data[[0]]
        tm.assert_extension_array_equal(result, expected)

        # 在一维数据上进行扩展维度操作，结果应与arr2d的转置相等
        result = data[:, np.newaxis]
        tm.assert_extension_array_equal(result, arr2d.T)

    # 定义测试函数，测试二维数组的迭代操作
    def test_iter_2d(self, data):
        # 将data reshape为(1, -1)，即行数为1，列数自动计算
        arr2d = data.reshape(1, -1)

        # 将arr2d转换为迭代对象objs，并断言objs的长度与arr2d的行数相同
        objs = list(iter(arr2d))
        assert len(objs) == arr2d.shape[0]

        # 对迭代对象objs中的每个元素进行类型、dtype、维度和长度的断言
        for obj in objs:
            assert isinstance(obj, type(data))
            assert obj.dtype == data.dtype
            assert obj.ndim == 1
            assert len(obj) == arr2d.shape[1]

    # 定义测试函数，测试二维数组的tolist方法
    def test_tolist_2d(self, data):
        # 将data reshape为(1, -1)，即行数为1，列数自动计算
        arr2d = data.reshape(1, -1)

        # 调用tolist方法，将arr2d转换为列表result
        result = arr2d.tolist()
        # 构建期望的结果列表expected，与data.tolist()的结果相同
        expected = [data.tolist()]

        # 断言result是列表类型，且result中的每个元素也是列表类型
        assert isinstance(result, list)
        assert all(isinstance(x, list) for x in result)

        # 断言result与expected相等
        assert result == expected

    # 定义测试函数，测试二维数组的concatenate操作
    def test_concat_2d(self, data):
        # 使用type(data)._concat_same_type([data, data])进行concatenate，并reshape为(-1, 2)
        left = type(data)._concat_same_type([data, data]).reshape(-1, 2)
        right = left.copy()

        # 在axis=0上进行concatenate操作
        result = left._concat_same_type([left, right], axis=0)
        # 构建期望的结果expected，与data._concat_same_type([data] * 4).reshape(-1, 2)相同
        expected = data._concat_same_type([data] * 4).reshape(-1, 2)
        tm.assert_extension_array_equal(result, expected)

        # 在axis=1上进行concatenate操作
        result = left._concat_same_type([left, right], axis=1)
        # 断言结果的形状为(len(data), 4)
        assert result.shape == (len(data), 4)
        # 断言结果的左半部分与left相等，右半部分与right相等
        tm.assert_extension_array_equal(result[:, :2], left)
        tm.assert_extension_array_equal(result[:, 2:], right)

        # 测试axis > 1的情况，应该抛出ValueError异常，匹配消息为"axis 2 is out of bounds for array of dimension 2"
        msg = "axis 2 is out of bounds for array of dimension 2"
        with pytest.raises(ValueError, match=msg):
            left._concat_same_type([left, right], axis=2)

    # 使用pytest.mark.parametrize装饰器，参数化method参数为["backfill", "pad"]
    # 定义一个测试方法，用于测试在二维数据上填充缺失值的功能
    def test_fillna_2d_method(self, data_missing, method):
        # 将 data_missing 数据在第一个轴向上复制两倍，然后重新形状为 2x2 的数组
        arr = data_missing.repeat(2).reshape(2, 2)
        # 断言第一行所有元素都是缺失值
        assert arr[0].isna().all()
        # 断言第二行没有任何缺失值
        assert not arr[1].isna().any()

        # 调用填充或反向填充方法，返回结果
        result = arr._pad_or_backfill(method=method, limit=None)

        # 期望结果是对 data_missing 调用填充或反向填充方法后，再复制两倍并形状为 2x2 的数组
        expected = data_missing._pad_or_backfill(method=method).repeat(2).reshape(2, 2)
        # 使用测试工具方法验证 result 和 expected 是否相等
        tm.assert_extension_array_equal(result, expected)

        # 将数组 arr 反转，以确保反向填充不是无操作
        arr2 = arr[::-1]
        # 断言反转后第一行没有缺失值
        assert not arr2[0].isna().any()
        # 断言反转后第二行所有元素都是缺失值
        assert arr2[1].isna().all()

        # 调用填充或反向填充方法，返回结果
        result2 = arr2._pad_or_backfill(method=method, limit=None)

        # 期望结果是对 data_missing 反向后调用填充或反向填充方法，再复制两倍并形状为 2x2 的数组
        expected2 = (
            data_missing[::-1]._pad_or_backfill(method=method).repeat(2).reshape(2, 2)
        )
        # 使用测试工具方法验证 result2 和 expected2 是否相等
        tm.assert_extension_array_equal(result2, expected2)

    # 使用 pytest 的参数化功能，测试在二维数组上进行无轴向缩减操作
    @pytest.mark.parametrize("method", ["mean", "median", "var", "std", "sum", "prod"])
    def test_reductions_2d_axis_none(self, data, method):
        # 将 data 数据重新形状为 1x(-1) 的二维数组
        arr2d = data.reshape(1, -1)

        err_expected = None
        err_result = None
        try:
            # 尝试获取 data 对象使用 method 方法的结果
            expected = getattr(data, method)()
        except Exception as err:
            # 如果在一维减少过程中发生错误，预期在二维减少过程中也应该出错
            err_expected = err
            try:
                # 尝试在 arr2d 上调用 method 方法，axis=None
                result = getattr(arr2d, method)(axis=None)
            except Exception as err2:
                # 记录在二维减少过程中发生的错误
                err_result = err2
        else:
            # 在未发生异常时，在 arr2d 上调用 method 方法，axis=None
            result = getattr(arr2d, method)(axis=None)

        # 如果 err_result 或 err_expected 有一个不是 None，则断言它们的类型相同
        if err_result is not None or err_expected is not None:
            assert type(err_result) == type(err_expected)
            return

        # 断言 result 和 expected 的缺失值匹配，或者它们的数值相等
        assert is_matching_na(result, expected) or result == expected

    # 使用 pytest 的参数化功能，测试在二维数组上进行减少操作时的最小计数限制
    @pytest.mark.parametrize("method", ["mean", "median", "var", "std", "sum", "prod"])
    @pytest.mark.parametrize("min_count", [0, 1])
    # 定义一个测试方法，用于对二维数据进行沿轴0的归约操作测试
    def test_reductions_2d_axis0(self, data, method, min_count):
        # 如果 min_count 等于 1 并且 method 不是 "sum" 或 "prod"，则跳过该测试
        if min_count == 1 and method not in ["sum", "prod"]:
            pytest.skip(f"min_count not relevant for {method}")

        # 将数据重塑为1行，自动计算列数
        arr2d = data.reshape(1, -1)

        kwargs = {}
        # 如果 method 是 "std" 或 "var"，传递 ddof=0 以获取全零标准差而不是全NA标准差
        if method in ["std", "var"]:
            kwargs["ddof"] = 0
        # 如果 method 是 "prod" 或 "sum"，传递 min_count 参数
        elif method in ["prod", "sum"]:
            kwargs["min_count"] = min_count

        try:
            # 使用 getattr 根据 method 调用 arr2d 的相应归约方法，沿轴0进行归约
            result = getattr(arr2d, method)(axis=0, **kwargs)
        except Exception as err:
            try:
                # 如果出错，尝试在原始数据上调用相同的 method 方法
                getattr(data, method)()
            except Exception as err2:
                # 如果两者抛出的异常类型相同，则断言通过，否则抛出 AssertionError
                assert type(err) == type(err2)
                return
            else:
                raise AssertionError("Both reductions should raise or neither")

        # 定义一个函数，根据 dtype 返回归约结果的数据类型
        def get_reduction_result_dtype(dtype):
            # 在某些情况下，Windows 和 32位编译可能会有 int32/uint32，而其他编译则有 int64/uint64
            if dtype.itemsize == 8:
                return dtype
            elif dtype.kind in "ib":
                return NUMPY_INT_TO_DTYPE[np.dtype(int)]
            else:
                # 即 dtype.kind == "u"
                return NUMPY_INT_TO_DTYPE[np.dtype("uint")]

        # 如果 method 是 "sum" 或 "prod"
        if method in ["sum", "prod"]:
            # std 和 var 不保持 dtype
            expected = data
            # 如果原始数据的 dtype 是整数或布尔类型，则转换为对应的 dtype
            if data.dtype.kind in "iub":
                dtype = get_reduction_result_dtype(data.dtype)
                expected = data.astype(dtype)
                assert dtype == expected.dtype

            # 如果 min_count 为 0，则用 1 填充 NaN 值（对于 "prod" 方法用 1，对于 "sum" 方法用 0）
            if min_count == 0:
                fill_value = 1 if method == "prod" else 0
                expected = expected.fillna(fill_value)

            # 断言归约后的结果与预期结果相等
            tm.assert_extension_array_equal(result, expected)
        # 如果 method 是 "median"
        elif method == "median":
            # std 和 var 不保持 dtype
            expected = data
            # 断言归约后的结果与预期结果相等
            tm.assert_extension_array_equal(result, expected)
        # 如果 method 是 "mean", "std", "var"
        elif method in ["mean", "std", "var"]:
            # 如果数据是整数类型或布尔类型，则将其转换为 "Float64" 类型
            if is_integer_dtype(data) or is_bool_dtype(data):
                data = data.astype("Float64")
            # 如果 method 是 "mean"，则断言归约后的结果与原始数据相等，否则断言结果为原始数据减去自身
            if method == "mean":
                tm.assert_extension_array_equal(result, data)
            else:
                tm.assert_extension_array_equal(result, data - data)
    # 定义一个测试函数，用于在二维数据上测试给定的方法
    def test_reductions_2d_axis1(self, data, method):
        # 将输入数据重新调整形状为1行多列的二维数组
        arr2d = data.reshape(1, -1)

        # 尝试在二维数组上调用指定的方法，指定沿第二个轴（axis=1）进行计算
        try:
            result = getattr(arr2d, method)(axis=1)
        except Exception as err:
            # 如果在二维数组上调用方法时出现异常，尝试在原始数据上直接调用方法
            try:
                getattr(data, method)()
            except Exception as err2:
                # 检查两次异常是否是同一类型，如果是，则通过测试
                assert type(err) == type(err2)
                return
            else:
                # 如果在原始数据上调用方法时没有异常，但在二维数组上有异常，则抛出断言错误
                raise AssertionError("Both reductions should raise or neither")

        # 如果成功执行了方法调用，以下是一些弱化的断言
        # 检查结果的形状是否为 (1,)
        assert result.shape == (1,)
        # 获取原始数据上调用方法后期望的标量值
        expected_scalar = getattr(data, method)()
        # 获取实际计算结果的第一个元素（因为是在二维数组上计算，结果应该是一个标量）
        res = result[0]
        # 断言实际结果与预期标量值相匹配，或者实际结果等于预期标量值
        assert is_matching_na(res, expected_scalar) or res == expected_scalar
class NDArrayBacked2DTests(Dim2CompatTests):
    # 继承自Dim2CompatTests的NDArrayBackedExtensionArray子类的更具体测试

    def test_copy_order(self, data):
        # 测试"copy"方法中关于"order"参数的行为是否符合numpy的语义

        # 创建一个重复并重塑为2维的数据数组
        arr2d = data.repeat(2).reshape(-1, 2)
        # 断言数组是C连续的
        assert arr2d._ndarray.flags["C_CONTIGUOUS"]

        # 复制数组并断言结果是C连续的
        res = arr2d.copy()
        assert res._ndarray.flags["C_CONTIGUOUS"]

        # 对数组的切片进行复制，并断言结果是C连续的
        res = arr2d[::2, ::2].copy()
        assert res._ndarray.flags["C_CONTIGUOUS"]

        # 使用"F"顺序复制数组，并断言结果不是C连续而是F连续的
        res = arr2d.copy("F")
        assert not res._ndarray.flags["C_CONTIGUOUS"]
        assert res._ndarray.flags["F_CONTIGUOUS"]

        # 使用"K"顺序复制数组，并断言结果是C连续的
        res = arr2d.copy("K")
        assert res._ndarray.flags["C_CONTIGUOUS"]

        # 对数组的转置进行"K"顺序复制，并断言结果不是C连续而是F连续的
        res = arr2d.T.copy("K")
        assert not res._ndarray.flags["C_CONTIGUOUS"]
        assert res._ndarray.flags["F_CONTIGUOUS"]

        # 测试不被numpy接受的顺序参数
        msg = r"order must be one of 'C', 'F', 'A', or 'K' \(got 'Q'\)"
        with pytest.raises(ValueError, match=msg):
            arr2d.copy("Q")

        # 测试非连续的数组切片
        arr_nc = arr2d[::2]
        # 断言切片数组不是C连续也不是F连续
        assert not arr_nc._ndarray.flags["C_CONTIGUOUS"]
        assert not arr_nc._ndarray.flags["F_CONTIGUOUS"]

        # 复制非连续切片数组并断言结果是C连续的，但不是F连续的
        assert arr_nc.copy()._ndarray.flags["C_CONTIGUOUS"]
        assert not arr_nc.copy()._ndarray.flags["F_CONTIGUOUS"]

        # 使用"C"顺序复制非连续切片数组并断言结果是C连续的，但不是F连续的
        assert arr_nc.copy("C")._ndarray.flags["C_CONTIGUOUS"]
        assert not arr_nc.copy("C")._ndarray.flags["F_CONTIGUOUS"]

        # 使用"F"顺序复制非连续切片数组并断言结果不是C连续而是F连续的
        assert not arr_nc.copy("F")._ndarray.flags["C_CONTIGUOUS"]
        assert arr_nc.copy("F")._ndarray.flags["F_CONTIGUOUS"]

        # 使用"K"顺序复制非连续切片数组并断言结果是C连续的，但不是F连续的
        assert arr_nc.copy("K")._ndarray.flags["C_CONTIGUOUS"]
        assert not arr_nc.copy("K")._ndarray.flags["F_CONTIGUOUS"]
```