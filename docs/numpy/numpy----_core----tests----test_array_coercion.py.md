# `.\numpy\numpy\_core\tests\test_array_coercion.py`

```
"""
Tests for array coercion, mainly through testing `np.array` results directly.
Note that other such tests exist, e.g., in `test_api.py` and many corner-cases
are tested (sometimes indirectly) elsewhere.
"""

# 导入必要的库
from itertools import permutations, product

import pytest
from pytest import param

# 导入 NumPy 库及其子模块
import numpy as np
import numpy._core._multiarray_umath as ncu
from numpy._core._rational_tests import rational

# 导入 NumPy 测试工具
from numpy.testing import (
    assert_array_equal, assert_warns, IS_PYPY)


def arraylikes():
    """
    Generator for functions converting an array into various array-likes.
    If full is True (default) it includes array-likes not capable of handling
    all dtypes.
    """
    # 定义基本的数组生成函数
    def ndarray(a):
        return a

    # 生成并返回 ndarray 函数作为 pytest 的参数化测试
    yield param(ndarray, id="ndarray")

    # 定义一个继承自 np.ndarray 的子类
    class MyArr(np.ndarray):
        pass

    # 将数组转换为 MyArr 类型的实例
    def subclass(a):
        return a.view(MyArr)

    # 生成并返回 subclass 函数作为 pytest 的参数化测试
    yield subclass

    # 定义一个类似序列的对象，模拟旧版 NumPy 中的行为
    class _SequenceLike():
        # 在旧版 NumPy 中，有时会关注协议数组是否也是 _SequenceLike
        def __len__(self):
            raise TypeError

        def __getitem__(self):
            raise TypeError

    # 模拟数组接口
    class ArrayDunder(_SequenceLike):
        def __init__(self, a):
            self.a = a

        # 定义 __array__ 方法以实现数组接口
        def __array__(self, dtype=None, copy=None):
            if dtype is None:
                return self.a
            return self.a.astype(dtype)

    # 生成并返回 ArrayDunder 类作为 pytest 的参数化测试
    yield param(ArrayDunder, id="__array__")

    # 生成并返回 memoryview 函数作为 pytest 的参数化测试
    yield param(memoryview, id="memoryview")

    # 模拟数组接口
    class ArrayInterface:
        def __init__(self, a):
            self.a = a  # 需要保持引用以保持接口有效
            self.__array_interface__ = a.__array_interface__

    # 生成并返回 ArrayInterface 类作为 pytest 的参数化测试
    yield param(ArrayInterface, id="__array_interface__")

    # 模拟数组结构
    class ArrayStruct:
        def __init__(self, a):
            self.a = a  # 需要保持引用以保持结构有效
            self.__array_struct__ = a.__array_struct__

    # 生成并返回 ArrayStruct 类作为 pytest 的参数化测试
    yield param(ArrayStruct, id="__array_struct__")


def scalar_instances(times=True, extended_precision=True, user_dtype=True):
    # 预定义的标量实例列表
    # 浮点数:
    yield param(np.sqrt(np.float16(5)), id="float16")
    yield param(np.sqrt(np.float32(5)), id="float32")
    yield param(np.sqrt(np.float64(5)), id="float64")
    if extended_precision:
        yield param(np.sqrt(np.longdouble(5)), id="longdouble")

    # 复数:
    yield param(np.sqrt(np.complex64(2+3j)), id="complex64")
    yield param(np.sqrt(np.complex128(2+3j)), id="complex128")
    if extended_precision:
        yield param(np.sqrt(np.clongdouble(2+3j)), id="clongdouble")

    # 布尔:
    # XFAIL: Bool 应该添加，但在处理字符串时存在一些问题，参见 gh-9875
    # yield param(np.bool(0), id="bool")

    # 整数:
    yield param(np.int8(2), id="int8")


这里的注释对代码中每个函数和类的定义进行了解释，确保符合题目要求的格式和注意事项。
    # Yield an np.int16 parameter with value 2 and identifier "int16"
    yield param(np.int16(2), id="int16")
    
    # Yield an np.int32 parameter with value 2 and identifier "int32"
    yield param(np.int32(2), id="int32")
    
    # Yield an np.int64 parameter with value 2 and identifier "int64"
    yield param(np.int64(2), id="int64")

    # Yield an np.uint8 parameter with value 2 and identifier "uint8"
    yield param(np.uint8(2), id="uint8")
    
    # Yield an np.uint16 parameter with value 2 and identifier "uint16"
    yield param(np.uint16(2), id="uint16")
    
    # Yield an np.uint32 parameter with value 2 and identifier "uint32"
    yield param(np.uint32(2), id="uint32")
    
    # Yield an np.uint64 parameter with value 2 and identifier "uint64"
    yield param(np.uint64(2), id="uint64")

    # Rational:
    # If user_dtype is True, yield a rational parameter with value 1/2 and identifier "rational"
    if user_dtype:
        yield param(rational(1, 2), id="rational")

    # Cannot create a structured void scalar directly:
    # Create a structured np.array with one element of type ("i,i"), and extract the first element
    structured = np.array([(1, 3)], "i,i")[0]
    
    # Assert that structured is indeed of type np.void
    assert isinstance(structured, np.void)
    
    # Assert that structured's dtype is np.dtype("i,i")
    assert structured.dtype == np.dtype("i,i")
    
    # Yield a param with structured as value and identifier "structured"
    yield param(structured, id="structured")

    # If times is True:
    if times:
        # Datetimes and timedelta
        # Yield a np.timedelta64 parameter with value 2 and identifier "timedelta64[generic]"
        yield param(np.timedelta64(2), id="timedelta64[generic]")
        
        # Yield a np.timedelta64 parameter with value 23 seconds and identifier "timedelta64[s]"
        yield param(np.timedelta64(23, "s"), id="timedelta64[s]")
        
        # Yield a np.timedelta64 parameter with NaT (Not a Time) value in seconds and identifier "timedelta64[s](NaT)"
        yield param(np.timedelta64("NaT", "s"), id="timedelta64[s](NaT)")

        # Yield a np.datetime64 parameter with NaT (Not a Time) value and identifier "datetime64[generic](NaT)"
        yield param(np.datetime64("NaT"), id="datetime64[generic](NaT)")
        
        # Yield a np.datetime64 parameter with specific datetime value and identifier "datetime64[ms]"
        yield param(np.datetime64("2020-06-07 12:43", "ms"), id="datetime64[ms]")

    # Strings and unstructured void:
    # Yield a np.bytes_ parameter with value b"1234" and identifier "bytes"
    yield param(np.bytes_(b"1234"), id="bytes")
    
    # Yield a np.str_ parameter with value "2345" and identifier "unicode"
    yield param(np.str_("2345"), id="unicode")
    
    # Yield a np.void parameter with value b"4321" and identifier "unstructured_void"
    yield param(np.void(b"4321"), id="unstructured_void")
def is_parametric_dtype(dtype):
    """Returns True if the dtype is a parametric legacy dtype (itemsize
    is 0, or a datetime without units)
    """
    # 检查 dtype 的 itemsize 是否为 0
    if dtype.itemsize == 0:
        return True
    # 检查 dtype 是否为 np.datetime64 或 np.timedelta64 的子类，并且以 "64" 结尾
    if issubclass(dtype.type, (np.datetime64, np.timedelta64)):
        if dtype.name.endswith("64"):
            # 返回 True，表示是通用的时间单位
            return True
    # 否则返回 False
    return False


class TestStringDiscovery:
    @pytest.mark.parametrize("obj",
            [object(), 1.2, 10**43, None, "string"],
            ids=["object", "1.2", "10**43", "None", "string"])
    def test_basic_stringlength(self, obj):
        # 获取对象 obj 的字符串长度
        length = len(str(obj))
        # 创建一个预期的 dtype，以字符串长度为长度的字节类型
        expected = np.dtype(f"S{length}")

        # 断言将 obj 转换为字节字符串后的 dtype 是否等于预期的 dtype
        assert np.array(obj, dtype="S").dtype == expected
        # 断言将包含 obj 的数组转换为字节字符串后的 dtype 是否等于预期的 dtype
        assert np.array([obj], dtype="S").dtype == expected

        # 检查嵌套数组也能正确地被发现
        arr = np.array(obj, dtype="O")
        assert np.array(arr, dtype="S").dtype == expected
        # 如果使用 dtype 类来指定，也应该能正确地被发现
        assert np.array(arr, dtype=type(expected)).dtype == expected
        # 检查 .astype() 方法的行为是否一致
        assert arr.astype("S").dtype == expected
        # `.astype()` 方法也能接受 DType 类型
        assert arr.astype(type(np.dtype("S"))).dtype == expected

    @pytest.mark.parametrize("obj",
            [object(), 1.2, 10**43, None, "string"],
            ids=["object", "1.2", "10**43", "None", "string"])
    def test_nested_arrays_stringlength(self, obj):
        # 获取对象 obj 的字符串长度
        length = len(str(obj))
        # 创建一个预期的 dtype，以字符串长度为长度的字节类型
        expected = np.dtype(f"S{length}")
        arr = np.array(obj, dtype="O")
        # 断言将包含 arr 和 arr 的数组转换为字节字符串后的 dtype 是否等于预期的 dtype
        assert np.array([arr, arr], dtype="S").dtype == expected

    @pytest.mark.parametrize("arraylike", arraylikes())
    def test_unpack_first_level(self, arraylike):
        # 我们解包一级数组
        obj = np.array([None])
        obj[0] = np.array(1.2)
        # 获取 obj[0] 的字符串表示的长度
        length = len(str(obj[0]))
        # 创建一个预期的 dtype，以字符串长度为长度的字节类型
        expected = np.dtype(f"S{length}")

        obj = arraylike(obj)
        # 将 obj 转换为字节字符串后，检查其形状是否为 (1, 1)
        arr = np.array([obj], dtype="S")
        assert arr.shape == (1, 1)
        # 检查转换后的 dtype 是否等于预期的 dtype
        assert arr.dtype == expected


class TestScalarDiscovery:
    def test_void_special_case(self):
        # 空类型 dtype 可以将元组识别为元素
        arr = np.array((1, 2, 3), dtype="i,i,i")
        # 检查数组形状是否为空
        assert arr.shape == ()
        arr = np.array([(1, 2, 3)], dtype="i,i,i")
        # 检查数组形状是否为 (1,)
        assert arr.shape == (1,)

    def test_char_special_case(self):
        # 字符类型的特殊情况
        arr = np.array("string", dtype="c")
        # 检查数组形状是否为 (6,)
        assert arr.shape == (6,)
        # 检查数组的字符类型是否为 'c'
        assert arr.dtype.char == "c"
        arr = np.array(["string"], dtype="c")
        # 检查数组形状是否为 (1, 6)
        assert arr.shape == (1, 6)
        # 检查数组的字符类型是否为 'c'
        assert arr.dtype.char == "c"
    # 定义一个测试方法，用于检查特殊字符情况是否能正确处理错误，如果数组维度太深会报错：
    def test_char_special_case_deep(self):
        # 嵌套列表，初始为包含一个字符串的列表（由于字符串是一个序列，所以是二维）
        nested = ["string"]  
        # 循环创建更深层次的嵌套，直到达到设定的最大维度减去2
        for i in range(ncu.MAXDIMS - 2):
            nested = [nested]

        # 创建一个 NumPy 数组，数据类型为 'c'（字符型）
        arr = np.array(nested, dtype='c')
        # 断言数组的形状为 (1,) * (ncu.MAXDIMS - 1) + (6,)
        assert arr.shape == (1,) * (ncu.MAXDIMS - 1) + (6,)
        # 使用 pytest 断言应该会抛出 ValueError 异常
        with pytest.raises(ValueError):
            # 创建一个 NumPy 数组，包含一个嵌套列表，数据类型为 'c'（字符型）
            np.array([nested], dtype="c")

    # 定义一个测试方法，用于检查创建未知对象的 NumPy 数组的情况：
    def test_unknown_object(self):
        # 创建一个包含未知对象的 NumPy 数组
        arr = np.array(object())
        # 断言数组的形状为空
        assert arr.shape == ()
        # 断言数组的数据类型为对象类型（np.dtype("O")）
        assert arr.dtype == np.dtype("O")

    # 使用 pytest.mark.parametrize 参数化测试方法，参数为 scalar_instances() 函数返回的值
    @pytest.mark.parametrize("scalar", scalar_instances())
    # 定义一个测试方法，用于检查标量情况的 NumPy 数组创建：
    def test_scalar(self, scalar):
        # 创建一个包含标量的 NumPy 数组
        arr = np.array(scalar)
        # 断言数组的形状为空
        assert arr.shape == ()
        # 断言数组的数据类型与标量的数据类型相同
        assert arr.dtype == scalar.dtype

        # 创建一个包含嵌套列表的 NumPy 数组，每个元素为相同的标量
        arr = np.array([[scalar, scalar]])
        # 断言数组的形状为 (1, 2)
        assert arr.shape == (1, 2)
        # 断言数组的数据类型与标量的数据类型相同
        assert arr.dtype == scalar.dtype

    # 使用 pytest.mark.filterwarnings 标记测试方法，忽略 "Promotion of numbers:FutureWarning" 警告
    # 定义一个测试方法，用于检查标量升级的情况：
    def test_scalar_promotion(self):
        # 对 scalar_instances() 返回的标量实例进行笛卡尔积的遍历
        for sc1, sc2 in product(scalar_instances(), scalar_instances()):
            # 从每个标量实例中取出第一个值
            sc1, sc2 = sc1.values[0], sc2.values[0]
            # 尝试创建一个包含 sc1 和 sc2 的 NumPy 数组
            try:
                arr = np.array([sc1, sc2])
            except (TypeError, ValueError):
                # 可能由于时间类型之间的升级问题而失败
                # XFAIL (ValueError): Some object casts are currently undefined
                continue
            # 断言数组的形状为 (2,)
            assert arr.shape == (2,)
            # 尝试确定 sc1 和 sc2 升级后的数据类型
            try:
                dt1, dt2 = sc1.dtype, sc2.dtype
                expected_dtype = np.promote_types(dt1, dt2)
                # 断言数组的数据类型与预期的升级数据类型相同
                assert arr.dtype == expected_dtype
            except TypeError as e:
                # 当前会导致数组的数据类型总是变为对象类型
                assert arr.dtype == np.dtype("O")
    # 定义一个测试方法，用于测试标量强制转换的不同路径，主要针对数值类型。
    # 包括一些与 `np.array` 直接相关的路径。
    def test_scalar_coercion(self, scalar):
        # 如果标量是浮点数类型，确保使用完整精度的数字
        if isinstance(scalar, np.inexact):
            scalar = type(scalar)((scalar * 2)**0.5)

        # 如果标量类型是 rational，通常由于缺少强制转换而失败。
        # 未来应基于 `setitem` 自动定义对象转换。
        if type(scalar) is rational:
            pytest.xfail("Rational to object cast is undefined currently.")

        # 使用对象类型的强制转换创建一个 numpy 数组
        arr = np.array(scalar, dtype=object).astype(scalar.dtype)

        # 测试创建包含此标量的数组的各种方法：
        arr1 = np.array(scalar).reshape(1)
        arr2 = np.array([scalar])
        arr3 = np.empty(1, dtype=scalar.dtype)
        arr3[0] = scalar
        arr4 = np.empty(1, dtype=scalar.dtype)
        arr4[:] = [scalar]
        # 所有这些方法应该产生相同的结果
        assert_array_equal(arr, arr1)
        assert_array_equal(arr, arr2)
        assert_array_equal(arr, arr3)
        assert_array_equal(arr, arr4)

    # 标记测试用例为预期失败（XFAIL），如果运行在 PyPy 环境下则跳过，
    # 原因是 `int(np.complex128(3))` 在 PyPy 上会失败。
    @pytest.mark.xfail(IS_PYPY, reason="`int(np.complex128(3))` fails on PyPy")
    # 忽略 numpy 的 ComplexWarning 警告
    @pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning")
    # 使用 scalar_instances() 的参数化标记
    @pytest.mark.parametrize("cast_to", scalar_instances())
    # 测试标量强制转换是否与类型转换和赋值行为相同
    def test_scalar_coercion_same_as_cast_and_assignment(self, cast_to):
        """
        Test that in most cases:
           * `np.array(scalar, dtype=dtype)`
           * `np.empty((), dtype=dtype)[()] = scalar`
           * `np.array(scalar).astype(dtype)`
        should behave the same.  The only exceptions are parametric dtypes
        (mainly datetime/timedelta without unit) and void without fields.
        """
        dtype = cast_to.dtype  # 用于参数化目标数据类型

        # 对于标量实例的循环测试
        for scalar in scalar_instances(times=False):
            scalar = scalar.values[0]

            # 如果数据类型是 void
            if dtype.type == np.void:
               if scalar.dtype.fields is not None and dtype.fields is None:
                    # 在这种情况下，强制转换为 "V6" 是有效的，但类型转换会失败。
                    # 因为类型相同，SETITEM 会处理这个问题，但其规则与类型转换不同。
                    with pytest.raises(TypeError):
                        np.array(scalar).astype(dtype)
                    np.array(scalar, dtype=dtype)
                    np.array([scalar], dtype=dtype)
                    continue

            # 主要的测试路径，首先尝试使用类型转换，如果成功则继续下面的测试
            try:
                cast = np.array(scalar).astype(dtype)
            except (TypeError, ValueError, RuntimeError):
                # 强制转换也应该引发异常（错误类型可能会改变）
                with pytest.raises(Exception):
                    np.array(scalar, dtype=dtype)

                if (isinstance(scalar, rational) and
                        np.issubdtype(dtype, np.signedinteger)):
                    return

                with pytest.raises(Exception):
                    np.array([scalar], dtype=dtype)
                # 赋值也应该引发异常
                res = np.zeros((), dtype=dtype)
                with pytest.raises(Exception):
                    res[()] = scalar

                return

            # 非错误路径：
            arr = np.array(scalar, dtype=dtype)
            assert_array_equal(arr, cast)
            # 赋值行为相同
            ass = np.zeros((), dtype=dtype)
            ass[()] = scalar
            assert_array_equal(ass, cast)

    @pytest.mark.parametrize("pyscalar", [10, 10.32, 10.14j, 10**100])
    def test_pyscalar_subclasses(self, pyscalar):
        """NumPy arrays are read/write which means that anything but invariant
        behaviour is on thin ice.  However, we currently are happy to discover
        subclasses of Python float, int, complex the same as the base classes.
        This should potentially be deprecated.
        """
        class MyScalar(type(pyscalar)):
            pass

        res = np.array(MyScalar(pyscalar))
        expected = np.array(pyscalar)
        assert_array_equal(res, expected)
    # 使用 pytest 的参数化功能，对参数 dtype_char 逐一测试
    @pytest.mark.parametrize("dtype_char", np.typecodes["All"])
    def test_default_dtype_instance(self, dtype_char):
        # 如果 dtype_char 是 "SU" 中的字符之一
        if dtype_char in "SU":
            # 创建一个特定类型的 NumPy dtype 对象，例如 "S1" 或 "U1"
            dtype = np.dtype(dtype_char + "1")
        elif dtype_char == "V":
            # 在旧版行为中使用 V8，由于 float64 是默认的 dtype，并且占用 8 字节
            dtype = np.dtype("V8")
        else:
            # 创建一个普通的 NumPy dtype 对象，根据给定的字符 dtype_char
            dtype = np.dtype(dtype_char)

        # 调用 ncu._discover_array_parameters 函数，返回一个元组，其中包含发现的 dtype 和一个未使用的占位符
        discovered_dtype, _ = ncu._discover_array_parameters([], type(dtype))

        # 断言发现的 dtype 和预期的 dtype 相同
        assert discovered_dtype == dtype
        # 断言发现的 dtype 的字节大小与预期的 dtype 的字节大小相同
        assert discovered_dtype.itemsize == dtype.itemsize

    # 使用 pytest 的参数化功能，对参数 dtype 和一组特定的测试参数进行组合测试
    @pytest.mark.parametrize("dtype", np.typecodes["Integer"])
    @pytest.mark.parametrize(["scalar", "error"],
            [(np.float64(np.nan), ValueError),
             (np.array(-1).astype(np.ulonglong)[()], OverflowError)])
    def test_scalar_to_int_coerce_does_not_cast(self, dtype, scalar, error):
        """
        Signed integers are currently different in that they do not cast other
        NumPy scalar, but instead use scalar.__int__(). The hardcoded
        exception to this rule is `np.array(scalar, dtype=integer)`.
        """
        # 创建一个特定类型的 NumPy dtype 对象，根据参数 dtype
        dtype = np.dtype(dtype)

        # 在特定的错误状态下执行以下逻辑，忽略 "invalid" 错误
        with np.errstate(invalid="ignore"):
            # 使用 casting 逻辑创建 coerced 数组
            coerced = np.array(scalar, dtype=dtype)
            # 使用 astype 方法创建 cast 数组
            cast = np.array(scalar).astype(dtype)
        # 断言 coerced 和 cast 数组相等
        assert_array_equal(coerced, cast)

        # 然而以下情况会引发异常:
        with pytest.raises(error):
            # 使用特定的 dtype 创建包含 scalar 的数组，预期会抛出 error 异常
            np.array([scalar], dtype=dtype)
        with pytest.raises(error):
            # 使用特定的 dtype 对 cast 数组的元素进行赋值操作，预期会抛出 error 异常
            cast[()] = scalar
# 定义一个名为 TestTimeScalars 的测试类
class TestTimeScalars:
    
    # 使用 pytest.mark.parametrize 装饰器，为参数 dtype 指定多个取值
    # 取值为 np.int64 和 np.float32
    @pytest.mark.parametrize("dtype", [np.int64, np.float32])
    # 使用 pytest.mark.parametrize 装饰器，为参数 scalar 指定多个取值
    # 这些取值是 param() 对象，每个对象代表一个测试参数组合
    @pytest.mark.parametrize("scalar",
            [param(np.timedelta64("NaT", "s"), id="timedelta64[s](NaT)"),
             param(np.timedelta64(123, "s"), id="timedelta64[s]"),
             param(np.datetime64("NaT", "generic"), id="datetime64[generic](NaT)"),
             param(np.datetime64(1, "D"), id="datetime64[D]")],)
    # 定义测试方法 test_coercion_basic，接受参数 dtype 和 scalar
    def test_coercion_basic(self, dtype, scalar):
        # 注释：在这里加上 `[scalar]` 是因为 np.array(scalar) 使用更严格的
        # `scalar.__int__()` 规则，以保持向后兼容性。
        
        # 创建一个数组 arr，使用 scalar 和指定的 dtype
        arr = np.array(scalar, dtype=dtype)
        # 创建一个数组 cast，使用 scalar 并转换为指定的 dtype
        cast = np.array(scalar).astype(dtype)
        # 断言数组 arr 和 cast 相等
        assert_array_equal(arr, cast)

        # 创建一个全为 1 的数组 ass，dtype 为指定的 dtype
        ass = np.ones((), dtype=dtype)
        # 如果 dtype 是 np.integer 的子类
        if issubclass(dtype, np.integer):
            # 使用 pytest.raises 检查是否会抛出 TypeError 异常
            with pytest.raises(TypeError):
                # 注释：抛出异常，就像 np.array([scalar], dtype=dtype) 一样，
                # 这是时间的转换，但整数的行为。
                ass[()] = scalar
        else:
            # 将 ass 数组的第一个元素赋值为 scalar
            ass[()] = scalar
            # 断言数组 ass 和 cast 相等
            assert_array_equal(ass, cast)

    # 使用 pytest.mark.parametrize 装饰器，为参数 dtype 指定多个取值
    # 取值为 np.int64 和 np.float32
    @pytest.mark.parametrize("dtype", [np.int64, np.float32])
    # 使用 pytest.mark.parametrize 装饰器，为参数 scalar 指定多个取值
    # 这些取值是 param() 对象，每个对象代表一个测试参数组合
    @pytest.mark.parametrize("scalar",
            [param(np.timedelta64(123, "ns"), id="timedelta64[ns]"),
             param(np.timedelta64(12, "generic"), id="timedelta64[generic]")])
    # 定义测试方法 test_coercion_timedelta_convert_to_number，接受参数 dtype 和 scalar
    def test_coercion_timedelta_convert_to_number(self, dtype, scalar):
        # 注释：只有 "ns" 和 "generic" 类型的 timedelta 可以转换为数字，
        # 所以这些稍微特殊一些。
        
        # 创建一个数组 arr，使用 scalar 和指定的 dtype
        arr = np.array(scalar, dtype=dtype)
        # 创建一个数组 cast，使用 scalar 并转换为指定的 dtype
        cast = np.array(scalar).astype(dtype)
        # 创建一个全为 1 的数组 ass，dtype 为指定的 dtype
        ass = np.ones((), dtype=dtype)
        # 将 ass 数组的第一个元素赋值为 scalar，会抛出异常，就像 np.array([scalar], dtype=dtype) 一样
        ass[()] = scalar  

        # 断言数组 arr 和 cast 相等
        assert_array_equal(arr, cast)
        # 断言数组 cast 和 cast 相等
        assert_array_equal(cast, cast)

    # 使用 pytest.mark.parametrize 装饰器，为参数 dtype 指定多个取值
    # 取值为 "S6" 和 "U6"
    @pytest.mark.parametrize("dtype", ["S6", "U6"])
    # 使用 pytest.mark.parametrize 装饰器，为参数 val 和 unit 指定多个取值
    # 这些取值是 param() 对象，每个对象代表一个测试参数组合
    @pytest.mark.parametrize(["val", "unit"],
            [param(123, "s", id="[s]"), param(123, "D", id="[D]")])
    # 定义一个测试方法，用于验证 datetime64 类型数据的赋值和强制类型转换
    def test_coercion_assignment_datetime(self, val, unit, dtype):
        # 对于从 datetime64 赋值的字符串，目前特别处理，不会进行强制类型转换。
        # 这是因为在这种情况下强制转换会导致错误，而传统上大多数情况下保持这种行为。
        # (`np.array(scalar, dtype="U6")` 在此之前会失败)
        # TODO: 应该解决这种差异，可以通过放宽强制转换或者弃用第一部分来解决。
        scalar = np.datetime64(val, unit)
        dtype = np.dtype(dtype)
        # 从 dtype 中提取 datetime64 的字符串表示，并截取前6个字符
        cut_string = dtype.type(str(scalar)[:6])

        # 使用 scalar 创建一个 dtype 类型的数组 arr
        arr = np.array(scalar, dtype=dtype)
        # 验证数组 arr 中索引为 () 的值与截取的字符串相等
        assert arr[()] == cut_string

        # 创建一个元素为 () 的 dtype 类型数组 ass，并将 scalar 赋值给该数组
        ass = np.ones((), dtype=dtype)
        ass[()] = scalar
        # 验证数组 ass 中索引为 () 的值与截取的字符串相等
        assert ass[()] == cut_string

        # 使用 pytest 检查以下代码会抛出 RuntimeError 异常
        with pytest.raises(RuntimeError):
            # 然而，与使用 `str(scalar)[:6]` 进行赋值不同，因为它由字符串类型处理而不是强制类型转换，
            # 显式强制类型转换失败：
            np.array(scalar).astype(dtype)


    @pytest.mark.parametrize(["val", "unit"],
            # 参数化测试用例，用于验证 timedelta64 类型数据的赋值和强制类型转换
            [param(123, "s", id="[s]"), param(123, "D", id="[D]")])
    def test_coercion_assignment_timedelta(self, val, unit):
        # 使用给定的 val 和 unit 创建一个 timedelta64 类型的 scalar
        scalar = np.timedelta64(val, unit)

        # 与 datetime64 不同，timedelta 允许不安全的强制转换：
        np.array(scalar, dtype="S6")
        # 使用 np.array 创建一个 scalar 的 S 类型的数组 cast
        cast = np.array(scalar).astype("S6")
        # 创建一个元素为 () 的 S 类型数组 ass，并将 scalar 赋值给该数组
        ass = np.ones((), dtype="S6")
        ass[()] = scalar
        # 预期的结果是将 scalar 强制转换为 S 类型后取前6个字符
        expected = scalar.astype("S")[:6]
        # 验证 cast 数组中索引为 () 的值与 expected 相等
        assert cast[()] == expected
        # 验证 ass 数组中索引为 () 的值与 expected 相等
        assert ass[()] == expected
    @pytest.mark.parametrize("arraylike", arraylikes())
    # 使用参数化测试，依次传入各种类似数组的对象进行测试
    def test_nested_arraylikes(self, arraylike):
        # 创建一个形状为 (1, 1) 的数组对象，并将其赋给 initial
        initial = arraylike(np.ones((1, 1)))

        # 使用 initial 初始化 nested 变量
        nested = initial
        # 循环创建 ncu.MAXDIMS - 1 层嵌套结构
        for i in range(ncu.MAXDIMS - 1):
            nested = [nested]

        # 断言尝试创建 np.array(nested, dtype="float64") 时抛出 ValueError 异常
        with pytest.raises(ValueError, match=".*would exceed the maximum"):
            np.array(nested, dtype="float64")

        # 创建一个 np.array，将 nested 对象放入，指定 dtype 为 object
        arr = np.array(nested, dtype=object)
        # 断言 arr 的数据类型为 np.dtype("O")
        assert arr.dtype == np.dtype("O")
        # 断言 arr 的形状为 (1,) * ncu.MAXDIMS
        assert arr.shape == (1,) * ncu.MAXDIMS
        # 断言 arr 的单元素与 initial 相同
        assert arr.item() == np.array(initial).item()
    def test_empty_sequence(self):
        # 创建一个包含空数组的对象数组，第一个数组为空，影响后续维度的发现
        arr = np.array([[], [1], [[1]]], dtype=object)
        # 断言数组的形状为 (3,)
        assert arr.shape == (3,)

        # 空序列会阻止进一步的维度发现，因此结果的形状将是 (0,)，
        # 在以下操作中将导致错误:
        with pytest.raises(ValueError):
            # 创建包含空数组和空对象的数组，会引发异常
            np.array([[], np.empty((0, 1))], dtype=object)

    def test_array_of_different_depths(self):
        # 当序列中包含深度不同的多个数组（或类数组对象）时，当前方法会发现它们共享的维度。
        # 参见也 gh-17224
        # 创建一个形状为 (3, 2) 的全零数组
        arr = np.zeros((3, 2))
        # 创建一个与 arr 第一维度不匹配的全零数组
        mismatch_first_dim = np.zeros((1, 2))
        # 创建一个与 arr 第二维度不匹配的全零数组
        mismatch_second_dim = np.zeros((3, 3))

        # 使用 ncu._discover_array_parameters 函数发现数组的参数，指定数据类型为对象类型
        dtype, shape = ncu._discover_array_parameters(
            [arr, mismatch_second_dim], dtype=np.dtype("O"))
        # 断言发现的形状为 (2, 3)
        assert shape == (2, 3)

        # 再次使用 ncu._discover_array_parameters 函数发现数组的参数，指定数据类型为对象类型
        dtype, shape = ncu._discover_array_parameters(
            [arr, mismatch_first_dim], dtype=np.dtype("O"))
        # 断言发现的形状为 (2,)
        assert shape == (2,)
        
        # 第二种情况目前受支持，因为这些数组可以存储为对象：
        # 使用 np.asarray 将数组 arr 和 mismatch_first_dim 转换为对象数组
        res = np.asarray([arr, mismatch_first_dim], dtype=np.dtype("O"))
        # 断言 res 的第一个元素是 arr
        assert res[0] is arr
        # 断言 res 的第二个元素是 mismatch_first_dim
        assert res[1] is mismatch_first_dim
class TestBadSequences:
    # 这些是对传递给 `np.array` 的不良对象进行测试的用例，通常会导致未定义的行为。
    # 在旧代码中，它们部分工作，但在现在它们将会失败。
    # 我们可以（也许应该）复制所有序列来防范不良行为者。

    def test_growing_list(self):
        # 要强制转换的列表，`mylist` 在强制转换期间会向其追加内容
        obj = []
        class mylist(list):
            def __len__(self):
                obj.append([1, 2])
                return super().__len__()

        obj.append(mylist([1, 2]))

        # 断言会抛出 RuntimeError 异常
        with pytest.raises(RuntimeError):
            np.array(obj)

    # 注意：我们没有测试缩减列表。这些会做非常恶意的事情，
    #       要修复它们的唯一方法是复制所有序列。
    #       （这在未来可能是一个真实的选项。）

    def test_mutated_list(self):
        # 要强制转换的列表，`mylist` 会改变第一个元素
        obj = []
        class mylist(list):
            def __len__(self):
                obj[0] = [2, 3]  # 用另一个列表替换第一个元素
                return super().__len__()

        obj.append([2, 3])
        obj.append(mylist([1, 2]))
        # 不会导致崩溃：
        np.array(obj)

    def test_replace_0d_array(self):
        # 要强制转换的列表，`baditem` 会改变第一个元素
        obj = []
        class baditem:
            def __len__(self):
                obj[0][0] = 2  # 用另一个列表替换第一个元素
                raise ValueError("not actually a sequence!")

            def __getitem__(self):
                pass

        # 在新代码中遇到一个边缘情况，`array(2)` 被缓存，
        # 因此替换它会使缓存无效。
        obj.append([np.array(2), baditem()])
        # 断言会抛出 RuntimeError 异常
        with pytest.raises(RuntimeError):
            np.array(obj)


class TestArrayLikes:
    @pytest.mark.parametrize("arraylike", arraylikes())
    def test_0d_object_special_case(self, arraylike):
        arr = np.array(0.)
        obj = arraylike(arr)
        # 单个类数组总是被转换：
        res = np.array(obj, dtype=object)
        assert_array_equal(arr, res)

        # 但单个 0-D 嵌套类数组永远不会：
        res = np.array([obj], dtype=object)
        assert res[0] is obj

    @pytest.mark.parametrize("arraylike", arraylikes())
    @pytest.mark.parametrize("arr", [np.array(0.), np.arange(4)])
    def test_object_assignment_special_case(self, arraylike, arr):
        obj = arraylike(arr)
        empty = np.arange(1, dtype=object)
        empty[:] = [obj]
        assert empty[0] is obj
    def test_0d_generic_special_case(self):
        # 定义一个继承自 np.ndarray 的数组子类，重写 __float__ 方法抛出 TypeError
        class ArraySubclass(np.ndarray):
            def __float__(self):
                raise TypeError("e.g. quantities raise on this")

        # 创建一个包含单个浮点数 0.0 的 numpy 数组
        arr = np.array(0.)
        # 通过视图将 arr 转换为 ArraySubclass 类型的对象
        obj = arr.view(ArraySubclass)
        # 再次将 obj 转换为 numpy 数组
        res = np.array(obj)
        # 断言两个数组内容相等
        assert_array_equal(arr, res)

        # 如果包含了 0 维数组，当前保证会使用 __float__ 方法抛出 TypeError 异常
        # 这里可能会考虑改变这种行为，因为 quantities 和 masked arrays 部分依赖于这一特性
        with pytest.raises(TypeError):
            np.array([obj])

        # 对于 memoryview 类型的对象也是一样的
        obj = memoryview(arr)
        # 再次尝试将其转换为 numpy 数组
        res = np.array(obj)
        # 断言两个数组内容相等
        assert_array_equal(arr, res)
        with pytest.raises(ValueError):
            # 这里错误类型并不重要
            np.array([obj])

    def test_arraylike_classes(self):
        # 各种数组类通常应该能够被存储在 numpy 的对象数组中
        # 这里测试了所有特殊属性（因为在强制转换期间会检查所有属性）
        
        # 创建一个包含 np.int64 类型的 numpy 数组
        arr = np.array(np.int64)
        # 断言索引 () 处的元素是 np.int64 类型
        assert arr[()] is np.int64
        
        # 创建一个包含 [np.int64] 的 numpy 数组
        arr = np.array([np.int64])
        # 断言第一个元素是 np.int64 类型
        assert arr[0] is np.int64

        # 对于属性/未绑定方法同样适用
        class ArrayLike:
            @property
            def __array_interface__(self):
                pass

            @property
            def __array_struct__(self):
                pass

            def __array__(self, dtype=None, copy=None):
                pass

        # 创建一个包含 ArrayLike 类型的 numpy 数组
        arr = np.array(ArrayLike)
        # 断言索引 () 处的元素是 ArrayLike 类型
        assert arr[()] is ArrayLike

        # 创建一个包含 [ArrayLike] 的 numpy 数组
        arr = np.array([ArrayLike])
        # 断言第一个元素是 ArrayLike 类型
        assert arr[0] is ArrayLike

    @pytest.mark.skipif(
            np.dtype(np.intp).itemsize < 8, reason="Needs 64bit platform")
    def test_too_large_array_error_paths(self):
        """Test the error paths, including for memory leaks"""
        # 创建一个 uint8 类型的元素为 0 的 numpy 数组
        arr = np.array(0, dtype="uint8")
        # 确保一个连续的拷贝不会起作用
        arr = np.broadcast_to(arr, 2**62)

        # 循环测试，确保缓存无法产生影响
        for i in range(5):
            # 期望内存错误异常被抛出，因为无法分配如此大的内存
            with pytest.raises(MemoryError):
                np.array(arr)
            with pytest.raises(MemoryError):
                np.array([arr])

    @pytest.mark.parametrize("attribute",
        ["__array_interface__", "__array__", "__array_struct__"])
    @pytest.mark.parametrize("error", [RecursionError, MemoryError])
    # 测试处理具有类似数组的不良属性的情况，使用指定的属性和错误类型

    # 定义一个名为 BadInterface 的类，用于模拟具有不良属性的接口
    class BadInterface:
        # 当调用不存在的属性时，如果属性名与传入的 attribute 相同，则抛出指定的错误
        def __getattr__(self, attr):
            if attr == attribute:
                raise error
            # 否则调用父类的同名方法，但是这里实际上应该使用 super().__getattribute__(attr)
            super().__getattr__(attr)

    # 使用 pytest 的上下文管理器检查是否会抛出指定的错误
    with pytest.raises(error):
        # 将 BadInterface 实例化为 NumPy 数组，期望抛出错误
        np.array(BadInterface())

@pytest.mark.parametrize("error", [RecursionError, MemoryError])
def test_bad_array_like_bad_length(self, error):
    # 测试处理具有类似数组但长度错误的情况，使用指定的错误类型

    # 定义一个名为 BadSequence 的类，用于模拟具有错误长度的序列
    class BadSequence:
        # 当调用 __len__ 方法时，抛出指定的错误
        def __len__(self):
            raise error
        # 必须实现 __getitem__ 方法以符合序列的要求，这里只是简单返回一个值
        def __getitem__(self):
            return 1

    # 使用 pytest 的上下文管理器检查是否会抛出指定的错误
    with pytest.raises(error):
        # 将 BadSequence 实例化为 NumPy 数组，期望抛出错误
        np.array(BadSequence())
# 定义一个名为 TestAsArray 的类，用于测试函数 asarray 的预期行为
class TestAsArray:
    """Test expected behaviors of ``asarray``."""
    def test_dtype_identity(self):
        """
        确认 *dtype* 关键字参数的预期行为。

        当使用 ``asarray()`` 时，通过关键字参数提供的 dtype 应该被应用到结果数组上。
        这会强制对于唯一的 np.dtype 对象产生唯一的数组处理方式，但对于等价的 dtypes，底层数据（即基础对象）与原始数组对象是共享的。

        参考 https://github.com/numpy/numpy/issues/1468
        """
        # 创建一个整数数组 int_array
        int_array = np.array([1, 2, 3], dtype='i')
        # 断言 np.asarray(int_array) 返回的是 int_array 本身
        assert np.asarray(int_array) is int_array

        # 字符代码解析为由 numpy 包提供的单例 dtype 对象。
        assert np.asarray(int_array, dtype='i') is int_array

        # 从 n.dtype('i') 派生一个 dtype，但添加一个元数据对象以强制使 dtype 不同。
        unequal_type = np.dtype('i', metadata={'spam': True})
        annotated_int_array = np.asarray(int_array, dtype=unequal_type)
        # 断言 annotated_int_array 不是 int_array
        assert annotated_int_array is not int_array
        # 断言 annotated_int_array 的基础对象是 int_array
        assert annotated_int_array.base is int_array
        # 创建一个具有新的不同 dtype 实例的等价描述符。
        equivalent_requirement = np.dtype('i', metadata={'spam': True})
        annotated_int_array_alt = np.asarray(annotated_int_array,
                                             dtype=equivalent_requirement)
        # 断言 unequal_type 与 equivalent_requirement 相等
        assert unequal_type == equivalent_requirement
        # 断言 unequal_type 不是 equivalent_requirement
        assert unequal_type is not equivalent_requirement
        # 断言 annotated_int_array_alt 不是 annotated_int_array
        assert annotated_int_array_alt is not annotated_int_array
        # 断言 annotated_int_array_alt 的 dtype 是 equivalent_requirement
        assert annotated_int_array_alt.dtype is equivalent_requirement

        # 检查一对 C 类型的相同逻辑，它们在计算环境之间的等效性可能有所不同。
        # 找到一个等价对。
        integer_type_codes = ('i', 'l', 'q')
        integer_dtypes = [np.dtype(code) for code in integer_type_codes]
        typeA = None
        typeB = None
        for typeA, typeB in permutations(integer_dtypes, r=2):
            if typeA == typeB:
                assert typeA is not typeB
                break
        assert isinstance(typeA, np.dtype) and isinstance(typeB, np.dtype)

        # 这些 ``asarray()`` 调用可能产生一个新视图或副本，但绝不是相同的对象。
        long_int_array = np.asarray(int_array, dtype='l')
        long_long_int_array = np.asarray(int_array, dtype='q')
        assert long_int_array is not int_array
        assert long_long_int_array is not int_array
        assert np.asarray(long_int_array, dtype='q') is not long_int_array
        array_a = np.asarray(int_array, dtype=typeA)
        assert typeA == typeB
        assert typeA is not typeB
        assert array_a.dtype is typeA
        assert array_a is not np.asarray(array_a, dtype=typeB)
        assert np.asarray(array_a, dtype=typeB).dtype is typeB
        assert array_a is np.asarray(array_a, dtype=typeB).base
class TestSpecialAttributeLookupFailure:
    # 定义一个测试类用于特殊属性查找失败的情况

    class WeirdArrayLike:
        @property
        def __array__(self, dtype=None, copy=None):
            # 定义一个属性方法__array__，抛出运行时异常"oops!"
            raise RuntimeError("oops!")

    class WeirdArrayInterface:
        @property
        def __array_interface__(self):
            # 定义一个属性方法__array_interface__，抛出运行时异常"oops!"
            raise RuntimeError("oops!")

    def test_deprecated(self):
        # 定义测试方法test_deprecated
        with pytest.raises(RuntimeError):
            # 使用pytest断言捕获RuntimeError异常
            np.array(self.WeirdArrayLike())
        with pytest.raises(RuntimeError):
            # 使用pytest断言捕获RuntimeError异常
            np.array(self.WeirdArrayInterface())


def test_subarray_from_array_construction():
    # 定义测试函数test_subarray_from_array_construction
    # 创建一个数组arr，包含元素[1, 2]
    arr = np.array([1, 2])

    # 将arr转换为指定dtype的数组res，预期结果是[[1, 1], [2, 2]]
    res = arr.astype("2i")
    assert_array_equal(res, [[1, 1], [2, 2]])

    # 使用arr构造一个新数组res，指定dtype为"(2,)i"，预期结果是[[1, 1], [2, 2]]
    res = np.array(arr, dtype="(2,)i")
    assert_array_equal(res, [[1, 1], [2, 2]])

    # 使用arr和其他元素构造一个多维数组res，指定dtype为"2i"，预期结果是[[[1, 1], [2, 2]], [[1, 1], [2, 2]]]
    res = np.array([[(1,), (2,)], arr], dtype="2i")
    assert_array_equal(res, [[[1, 1], [2, 2]], [[1, 1], [2, 2]]])

    # 再次尝试多维示例:
    # 创建一个5行2列的数组arr
    arr = np.arange(5 * 2).reshape(5, 2)
    # 根据arr的形状广播到新形状(5, 2, 2, 2)，并赋值给expected
    expected = np.broadcast_to(arr[:, :, np.newaxis, np.newaxis], (5, 2, 2, 2))

    # 将arr转换为指定dtype的数组res，预期结果是广播到expected的形状
    res = arr.astype("(2,2)f")
    assert_array_equal(res, expected)

    # 使用arr构造一个新数组res，指定dtype为"(2,2)f"，预期结果是广播到expected的形状
    res = np.array(arr, dtype="(2,2)f")
    assert_array_equal(res, expected)


def test_empty_string():
    # 定义测试函数test_empty_string
    # 创建一个长度为10的空字符串数组，dtype为"S"，预期结果是包含"\0"的S1数组
    res = np.array([""] * 10, dtype="S")
    assert_array_equal(res, np.array("\0", "S1"))
    assert res.dtype == "S1"

    # 创建一个长度为10的空字符串数组arr，dtype为object
    arr = np.array([""] * 10, dtype=object)

    # 将arr转换为指定dtype的数组res，预期结果是包含b""的S1数组
    res = arr.astype("S")
    assert_array_equal(res, b"")
    assert res.dtype == "S1"

    # 使用arr构造一个新数组res，dtype为"S"，预期结果是包含b""的S1数组
    res = np.array(arr, dtype="S")
    assert_array_equal(res, b"")
    # TODO: This is arguably weird/wrong, but seems old:
    # 断言结果的dtype可能会与旧版本的S1不同，但现在是S1
    assert res.dtype == f"S{np.dtype('O').itemsize}"

    # 使用arr和另一个空字符串数组构造新数组res，dtype为"S"，预期结果是包含b""的S1数组，形状为(2, 10)
    res = np.array([[""] * 10, arr], dtype="S")
    assert_array_equal(res, b"")
    assert res.shape == (2, 10)
    assert res.dtype == "S1"
```