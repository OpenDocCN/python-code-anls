# `.\numpy\numpy\_core\tests\test_casting_floatingpoint_errors.py`

```
# 导入 pytest 库，用于编写和运行测试用例
import pytest
# 从 pytest 库中导入 param 对象，用于参数化测试用例
from pytest import param
# 导入 numpy.testing 模块中的 IS_WASM
from numpy.testing import IS_WASM
# 导入 numpy 库，并使用 np 别名
import numpy as np


def values_and_dtypes():
    """
    生成会导致浮点错误的数值和数据类型对，包括整数到浮点数的无效转换（会产生"invalid"警告）
    和浮点数转换的溢出（会产生"overflow"警告）。

    （Python 中 int/float 的路径不需要在所有相同情况下进行测试，但也不会有害。）
    """
    # 转换为 float16：
    yield param(70000, "float16", id="int-to-f2")
    yield param("70000", "float16", id="str-to-f2")
    yield param(70000.0, "float16", id="float-to-f2")
    yield param(np.longdouble(70000.), "float16", id="longdouble-to-f2")
    yield param(np.float64(70000.), "float16", id="double-to-f2")
    yield param(np.float32(70000.), "float16", id="float-to-f2")
    # 转换为 float32：
    yield param(10**100, "float32", id="int-to-f4")
    yield param(1e100, "float32", id="float-to-f2")
    yield param(np.longdouble(1e300), "float32", id="longdouble-to-f2")
    yield param(np.float64(1e300), "float32", id="double-to-f2")
    # 转换为 float64：
    # 如果 longdouble 是 double-double，其最大值可能被舍入到 double 的最大值。
    # 所以我们校正了 double 间距（有点奇怪，不过）：
    max_ld = np.finfo(np.longdouble).max
    spacing = np.spacing(np.nextafter(np.finfo("f8").max, 0))
    if max_ld - spacing > np.finfo("f8").max:
        yield param(np.finfo(np.longdouble).max, "float64",
                    id="longdouble-to-f8")

    # 转换为 complex32：
    yield param(2e300, "complex64", id="float-to-c8")
    yield param(2e300+0j, "complex64", id="complex-to-c8")
    yield param(2e300j, "complex64", id="complex-to-c8")
    yield param(np.longdouble(2e300), "complex64", id="longdouble-to-c8")

    # 无效的浮点到整数转换：
    with np.errstate(over="ignore"):
        for to_dt in np.typecodes["AllInteger"]:
            for value in [np.inf, np.nan]:
                for from_dt in np.typecodes["AllFloat"]:
                    from_dt = np.dtype(from_dt)
                    from_val = from_dt.type(value)

                    yield param(from_val, to_dt, id=f"{from_val}-to-{to_dt}")


def check_operations(dtype, value):
    """
    NumPy 中有许多专用路径进行类型转换，应检查在这些转换过程中发生的浮点错误。
    """
    if dtype.kind != 'i':
        # 这些赋值使用更严格的 setitem 逻辑：
        def assignment():
            arr = np.empty(3, dtype=dtype)
            arr[0] = value

        yield assignment

        def fill():
            arr = np.empty(3, dtype=dtype)
            arr.fill(value)

        yield fill

    def copyto_scalar():
        arr = np.empty(3, dtype=dtype)
        np.copyto(arr, value, casting="unsafe")

    yield copyto_scalar

    def copyto():
        arr = np.empty(3, dtype=dtype)
        np.copyto(arr, np.array([value, value, value]), casting="unsafe")

    yield copyto
    yield copyto

    # 定义一个函数 copyto_scalar_masked，用于将 value 复制到长度为 3 的 arr 中的指定位置
    def copyto_scalar_masked():
        arr = np.empty(3, dtype=dtype)
        np.copyto(arr, value, casting="unsafe",
                  where=[True, False, True])

    yield copyto_scalar_masked

    # 定义一个函数 copyto_masked，将 value 复制到长度为 3 的 arr 中的指定位置，支持广播
    def copyto_masked():
        arr = np.empty(3, dtype=dtype)
        np.copyto(arr, np.array([value, value, value]), casting="unsafe",
                  where=[True, False, True])

    yield copyto_masked

    # 定义一个函数 direct_cast，创建一个包含三个相同值的数组，并进行类型转换
    def direct_cast():
        np.array([value, value, value]).astype(dtype)

    yield direct_cast

    # 定义一个函数 direct_cast_nd_strided，创建一个形状为 (5, 5, 5) 的数组，并进行类型转换
    def direct_cast_nd_strided():
        arr = np.full((5, 5, 5), fill_value=value)[:, ::2, :]
        arr.astype(dtype)

    yield direct_cast_nd_strided

    # 定义一个函数 boolean_array_assignment，将 value 复制到长度为 3 的 arr 中的布尔索引位置
    def boolean_array_assignment():
        arr = np.empty(3, dtype=dtype)
        arr[[True, False, True]] = np.array([value, value])

    yield boolean_array_assignment

    # 定义一个函数 integer_array_assignment，将 value 复制到长度为 3 的 arr 中的整数索引位置
    def integer_array_assignment():
        arr = np.empty(3, dtype=dtype)
        values = np.array([value, value])

        arr[[0, 1]] = values

    yield integer_array_assignment

    # 定义一个函数 integer_array_assignment_with_subspace，将 value 复制到形状为 (5, 3) 的 arr 中的整数索引位置
    def integer_array_assignment_with_subspace():
        arr = np.empty((5, 3), dtype=dtype)
        values = np.array([value, value, value])

        arr[[0, 2]] = values

    yield integer_array_assignment_with_subspace

    # 定义一个函数 flat_assignment，将 value 平铺到长度为 3 的 arr 中
    def flat_assignment():
        arr = np.empty((3,), dtype=dtype)
        values = np.array([value, value, value])
        arr.flat[:] = values

    yield flat_assignment
# 如果运行环境是 WebAssembly（WASM），则跳过此测试用例，因为不支持浮点异常
@pytest.mark.skipif(IS_WASM, reason="no wasm fp exception support")

# 使用参数化测试，参数来自 values_and_dtypes 函数返回的值对(value, dtype)
@pytest.mark.parametrize(["value", "dtype"], values_and_dtypes())

# 忽略 numpy 的 ComplexWarning 警告
@pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning")
def test_floatingpoint_errors_casting(dtype, value):
    # 将 dtype 转换为 numpy 的 dtype 对象
    dtype = np.dtype(dtype)

    # 对于给定的 dtype 和 value，获取其支持的操作列表，并逐个执行
    for operation in check_operations(dtype, value):
        # 重新确认 dtype 类型，虽然此处似乎无实际改变
        dtype = np.dtype(dtype)

        # 确定匹配条件，根据 dtype 的种类决定匹配的异常类型
        match = "invalid" if dtype.kind in 'iu' else "overflow"

        # 使用 pytest.warns 检查是否会引发 RuntimeWarning，匹配特定的异常类型
        with pytest.warns(RuntimeWarning, match=match):
            operation()

        # 使用 np.errstate 设置浮点运算状态，使得所有异常都会被抛出
        with np.errstate(all="raise"):
            # 使用 pytest.raises 检查是否会引发 FloatingPointError，匹配特定的异常类型
            with pytest.raises(FloatingPointError, match=match):
                operation()
```