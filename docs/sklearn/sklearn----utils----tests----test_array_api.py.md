# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_array_api.py`

```
# 导入必要的库和模块
import re  # 导入正则表达式模块
from functools import partial  # 导入偏函数功能

import numpy  # 导入NumPy库
import pytest  # 导入pytest测试框架
from numpy.testing import assert_allclose  # 导入NumPy测试框架中的数组比较函数

from sklearn._config import config_context  # 导入配置上下文
from sklearn.base import BaseEstimator  # 导入基础估计器类
from sklearn.utils._array_api import (
    _ArrayAPIWrapper,  # 导入数组API包装器
    _asarray_with_order,  # 导入按顺序转换为数组的功能
    _atol_for_type,  # 导入根据类型返回绝对容差的功能
    _average,  # 导入计算平均值的功能
    _convert_to_numpy,  # 导入转换为NumPy数组的功能
    _estimator_with_converted_arrays,  # 导入使用转换数组的估计器
    _is_numpy_namespace,  # 导入判断是否为NumPy命名空间的功能
    _isin,  # 导入判断元素是否在数组中的功能
    _nanmax,  # 导入计算最大值（忽略NaN）的功能
    _nanmin,  # 导入计算最小值（忽略NaN）的功能
    _NumPyAPIWrapper,  # 导入NumPy API包装器
    _ravel,  # 导入数组展平的功能
    device,  # 导入设备标识符
    get_namespace,  # 导入获取命名空间的函数
    get_namespace_and_device,  # 导入获取命名空间和设备的函数
    indexing_dtype,  # 导入索引数据类型
    supported_float_dtypes,  # 导入支持的浮点数数据类型
    yield_namespace_device_dtype_combinations,  # 导入生成命名空间、设备和数据类型组合的生成器
)
from sklearn.utils._testing import (
    _array_api_for_tests,  # 导入用于测试的数组API
    assert_array_equal,  # 导入断言数组相等的功能
    skip_if_array_api_compat_not_configured,  # 导入如果未配置数组API兼容性则跳过的装饰器
)
from sklearn.utils.fixes import _IS_32BIT  # 导入32位系统修复

# 使用pytest装饰器定义一个参数化测试，测试NumPy数组和列表的情况
@pytest.mark.parametrize("X", [numpy.asarray([1, 2, 3]), [1, 2, 3]])
def test_get_namespace_ndarray_default(X):
    """Check that get_namespace returns NumPy wrapper"""
    # 获取命名空间和数组API兼容性标志
    xp_out, is_array_api_compliant = get_namespace(X)
    # 断言返回的命名空间是_NumPyAPIWrapper类型
    assert isinstance(xp_out, _NumPyAPIWrapper)
    # 断言数组API兼容性为False
    assert not is_array_api_compliant


# 定义测试函数，验证在设备和创建函数下的预期行为
def test_get_namespace_ndarray_creation_device():
    """Check expected behavior with device and creation functions."""
    # 创建NumPy数组
    X = numpy.asarray([1, 2, 3])
    # 获取命名空间和数组API兼容性标志
    xp_out, _ = get_namespace(X)

    # 使用xp_out创建全填充数组，指定设备为cpu
    full_array = xp_out.full(10, fill_value=2.0, device="cpu")
    # 断言全填充数组的值接近于2.0
    assert_allclose(full_array, [2.0] * 10)

    # 使用pytest断言，期望抛出值错误异常，匹配字符串"Unsupported device"
    with pytest.raises(ValueError, match="Unsupported device"):
        xp_out.zeros(10, device="cuda")


# 使用装饰器跳过如果未配置数组API兼容性的测试
@skip_if_array_api_compat_not_configured
def test_get_namespace_ndarray_with_dispatch():
    """Test get_namespace on NumPy ndarrays."""
    # 导入并检查array_api_compat模块
    array_api_compat = pytest.importorskip("array_api_compat")

    # 创建NumPy数组
    X_np = numpy.asarray([[1, 2, 3]])

    # 在数组API分派为真的上下文中
    with config_context(array_api_dispatch=True):
        # 获取命名空间和数组API兼容性标志
        xp_out, is_array_api_compliant = get_namespace(X_np)
        # 断言数组API兼容性为True
        assert is_array_api_compliant
        # 断言返回的命名空间与array_api_compat.numpy相同
        assert xp_out is array_api_compat.numpy


# 使用装饰器跳过如果未配置数组API兼容性的测试
@skip_if_array_api_compat_not_configured
def test_get_namespace_array_api():
    """Test get_namespace for ArrayAPI arrays."""
    # 导入并检查array_api_strict模块
    xp = pytest.importorskip("array_api_strict")

    # 创建NumPy数组
    X_np = numpy.asarray([[1, 2, 3]])
    # 将NumPy数组转换为ArrayAPI数组
    X_xp = xp.asarray(X_np)

    # 在数组API分派为真的上下文中
    with config_context(array_api_dispatch=True):
        # 获取命名空间和数组API兼容性标志
        xp_out, is_array_api_compliant = get_namespace(X_xp)
        # 断言数组API兼容性为True
        assert is_array_api_compliant

        # 使用pytest断言，期望抛出类型错误异常
        with pytest.raises(TypeError):
            get_namespace(X_xp, X_np)


class _AdjustableNameAPITestWrapper(_ArrayAPIWrapper):
    """API wrapper that has an adjustable name. Used for testing."""

    def __init__(self, array_namespace, name):
        # 调用父类构造函数，初始化数组API包装器
        super().__init__(array_namespace=array_namespace)
        # 设置私有属性__name__为指定的名称
        self.__name__ = name


# 定义测试函数，测试不是NumPy的ArrayAPI的_ArrayAPIWrapper
def test_array_api_wrapper_astype():
    """Test _ArrayAPIWrapper for ArrayAPIs that is not NumPy."""
    # 导入并检查array_api_strict模块
    array_api_strict = pytest.importorskip("array_api_strict")
    # 创建_AdjustableNameAPITestWrapper对象，使用array_api_strict命名空间和名称"array_api_strict"
    xp_ = _AdjustableNameAPITestWrapper(array_api_strict, "array_api_strict")
    # 创建_ArrayAPIWrapper对象，使用xp_作为数组API命名空间
    xp = _ArrayAPIWrapper(xp_)
    # 创建一个 NumPy 数组 X，内容为两个子数组，每个子数组包含三个元素，数据类型为 float64
    X = xp.asarray([[1, 2, 3], [3, 4, 5]], dtype=xp.float64)
    
    # 将数组 X 转换为数据类型为 float32 的 NumPy 数组，并将结果赋给 X_converted
    X_converted = xp.astype(X, xp.float32)
    
    # 使用断言检查转换后的数组 X_converted 的数据类型是否为 float32
    assert X_converted.dtype == xp.float32
    
    # 将数组 X 转换为数据类型为 float32 的 NumPy 数组，并将结果赋给 X_converted
    X_converted = xp.asarray(X, dtype=xp.float32)
    
    # 使用断言再次检查转换后的数组 X_converted 的数据类型是否为 float32
    assert X_converted.dtype == xp.float32
# 使用 pytest.mark.parametrize 装饰器，参数化测试函数 test_asarray_with_order，为两种数组 API 运行测试
@pytest.mark.parametrize("array_api", ["numpy", "array_api_strict"])
def test_asarray_with_order(array_api):
    """Test _asarray_with_order passes along order for NumPy arrays."""
    # 导入指定的数组 API 模块，若导入失败则跳过测试
    xp = pytest.importorskip(array_api)

    # 创建 NumPy 数组 X
    X = xp.asarray([1.2, 3.4, 5.1])
    # 调用被测试的函数 _asarray_with_order，期望返回一个按照指定顺序的数组 X_new
    X_new = _asarray_with_order(X, order="F", xp=xp)

    # 将 X_new 转换为 NumPy 数组
    X_new_np = numpy.asarray(X_new)
    # 断言 X_new_np 是列优先（Fortran）连续的
    assert X_new_np.flags["F_CONTIGUOUS"]


# 测试 _asarray_with_order 函数在通用数组 API 下忽略顺序
def test_asarray_with_order_ignored():
    """Test _asarray_with_order ignores order for Generic ArrayAPI."""
    # 导入严格数组 API 模块，若导入失败则跳过测试
    xp = pytest.importorskip("array_api_strict")
    # 创建适配器，用于模拟 array_api_strict 模块的行为
    xp_ = _AdjustableNameAPITestWrapper(xp, "array_api_strict")

    # 创建 NumPy 数组 X，指定 C 顺序（行优先）
    X = numpy.asarray([[1.2, 3.4, 5.1], [3.4, 5.5, 1.2]], order="C")
    # 使用 xp_ 对象的 asarray 方法将 X 转换为 array_api_strict 模块的数组表示
    X = xp_.asarray(X)

    # 调用 _asarray_with_order 函数，期望忽略指定的顺序要求
    X_new = _asarray_with_order(X, order="F", xp=xp_)

    # 将 X_new 转换为 NumPy 数组
    X_new_np = numpy.asarray(X_new)
    # 断言 X_new_np 是行优先（C）连续的，且不是列优先（Fortran）连续的
    assert X_new_np.flags["C_CONTIGUOUS"]
    assert not X_new_np.flags["F_CONTIGUOUS"]


# 使用多个参数化装饰器，为测试函数 test_average 提供各种参数组合的输入
@pytest.mark.parametrize(
    "array_namespace, device, dtype_name", yield_namespace_device_dtype_combinations()
)
@pytest.mark.parametrize(
    "weights, axis, normalize, expected",
    [
        # normalize = True
        (None, None, True, 3.5),
        (None, 0, True, [2.5, 3.5, 4.5]),
        (None, 1, True, [2, 5]),
        ([True, False], 0, True, [1, 2, 3]),  # boolean weights
        ([True, True, False], 1, True, [1.5, 4.5]),  # boolean weights
        ([0.4, 0.1], 0, True, [1.6, 2.6, 3.6]),
        ([0.4, 0.2, 0.2], 1, True, [1.75, 4.75]),
        ([1, 2], 0, True, [3, 4, 5]),
        ([1, 1, 2], 1, True, [2.25, 5.25]),
        ([[1, 2, 3], [1, 2, 3]], 0, True, [2.5, 3.5, 4.5]),
        ([[1, 2, 1], [2, 2, 2]], 1, True, [2, 5]),
        # normalize = False
        (None, None, False, 21),
        (None, 0, False, [5, 7, 9]),
        (None, 1, False, [6, 15]),
        ([True, False], 0, False, [1, 2, 3]),  # boolean weights
        ([True, True, False], 1, False, [3, 9]),  # boolean weights
        ([0.4, 0.1], 0, False, [0.8, 1.3, 1.8]),
        ([0.4, 0.2, 0.2], 1, False, [1.4, 3.8]),
        ([1, 2], 0, False, [9, 12, 15]),
        ([1, 1, 2], 1, False, [9, 21]),
        ([[1, 2, 3], [1, 2, 3]], 0, False, [5, 14, 27]),
        ([[1, 2, 1], [2, 2, 2]], 1, False, [8, 30]),
    ],
)
def test_average(
    array_namespace, device, dtype_name, weights, axis, normalize, expected
):
    # 根据 array_namespace 和 device 获取相应的数组 API
    xp = _array_api_for_tests(array_namespace, device)
    # 创建 NumPy 数组 array_in，并使用指定的数据类型
    array_in = numpy.asarray([[1, 2, 3], [4, 5, 6]], dtype=dtype_name)
    # 将 array_in 转换为指定的数组 API 表示
    array_in = xp.asarray(array_in, device=device)
    if weights is not None:
        # 如果 weights 不为空，则创建权重数组并转换为指定的数组 API 表示
        weights = numpy.asarray(weights, dtype=dtype_name)
        weights = xp.asarray(weights, device=device)

    # 使用配置上下文，设置数组 API 分发为 True
    with config_context(array_api_dispatch=True):
        # 调用被测试的函数 _average，返回计算的平均值结果
        result = _average(array_in, axis=axis, weights=weights, normalize=normalize)

    # 断言输入数组和结果数组具有相同的设备属性
    assert getattr(array_in, "device", None) == getattr(result, "device", None)

    # 将结果转换为 NumPy 数组，并使用指定的数组 API
    result = _convert_to_numpy(result, xp)
    # 断言结果与期望结果在指定的类型容差范围内相等
    assert_allclose(result, expected, atol=_atol_for_type(dtype_name))
    # 创建一个字符串，包含三个逗号分隔的变量名
    "array_namespace, device, dtype_name",
    # 调用函数 yield_namespace_device_dtype_combinations，生成包含不包括 numpy 命名空间的命名空间、设备和数据类型组合的生成器
    yield_namespace_device_dtype_combinations(include_numpy_namespaces=False),
# 定义一个测试函数，用于测试当输入数据类型错误时是否会引发异常
def test_average_raises_with_wrong_dtype(array_namespace, device, dtype_name):
    # 获取测试所需的数组 API 接口
    xp = _array_api_for_tests(array_namespace, device)

    # 创建一个复数数组，包括实部和虚部，数据类型由参数 dtype_name 指定
    array_in = numpy.asarray([2, 0], dtype=dtype_name) + 1j * numpy.asarray(
        [4, 3], dtype=dtype_name
    )
    # 获取复数数据类型名称
    complex_type_name = array_in.dtype.name

    # 如果数组 API 接口中不包含当前复数数据类型，跳过测试
    if not hasattr(xp, complex_type_name):
        pytest.skip(f"{array_namespace} does not support {complex_type_name}")

    # 将 array_in 转换为 xp 上的数组表示，使用特定设备
    array_in = xp.asarray(array_in, device=device)

    # 准备错误消息，指出平均函数不支持复数浮点数值
    err_msg = "Complex floating point values are not supported by average."
    
    # 在特定的配置环境下，期望调用 _average 函数时抛出 NotImplementedError 异常，并匹配特定错误消息
    with (
        config_context(array_api_dispatch=True),
        pytest.raises(NotImplementedError, match=err_msg),
    ):
        _average(array_in)


# 使用参数化测试装饰器，测试 _average 函数在不同参数组合下是否会抛出异常
@pytest.mark.parametrize(
    "array_namespace, device, dtype_name",
    yield_namespace_device_dtype_combinations(include_numpy_namespaces=True),
)
@pytest.mark.parametrize(
    "axis, weights, error, error_msg",
    (
        (
            None,
            [1, 2],
            TypeError,
            "Axis must be specified",
        ),
        (
            0,
            [[1, 2]],
            TypeError,
            "1D weights expected",
        ),
        (
            0,
            [1, 2, 3, 4],
            ValueError,
            "Length of weights",
        ),
        (0, [-1, 1], ZeroDivisionError, "Weights sum to zero, can't be normalized"),
    ),
)
# 定义测试函数，用于验证 _average 函数在使用非法参数时是否会引发异常
def test_average_raises_with_invalid_parameters(
    array_namespace, device, dtype_name, axis, weights, error, error_msg
):
    # 获取测试所需的数组 API 接口
    xp = _array_api_for_tests(array_namespace, device)

    # 创建一个二维数组作为输入数据，数据类型由参数 dtype_name 指定
    array_in = numpy.asarray([[1, 2, 3], [4, 5, 6]], dtype=dtype_name)
    # 将 array_in 转换为 xp 上的数组表示，使用特定设备
    array_in = xp.asarray(array_in, device=device)

    # 创建权重数组，数据类型由参数 weights 指定
    weights = numpy.asarray(weights, dtype=dtype_name)
    # 将 weights 转换为 xp 上的数组表示，使用特定设备
    weights = xp.asarray(weights, device=device)

    # 在特定的配置环境下，期望调用 _average 函数时抛出特定异常 error，并匹配特定错误消息 error_msg
    with config_context(array_api_dispatch=True), pytest.raises(error, match=error_msg):
        _average(array_in, axis=axis, weights=weights)


# 定义测试函数，用于验证 device 函数在没有输入时是否会引发 ValueError 异常
def test_device_raises_if_no_input():
    # 准备错误消息的正则表达式模式，指出至少需要一个输入数组，但没有输入
    err_msg = re.escape(
        "At least one input array expected after filtering with remove_none=True, "
        "remove_types=[str]. Got none. Original types: []."
    )
    # 在特定的配置环境下，期望调用 device 函数时抛出 ValueError 异常，并匹配特定错误消息 err_msg
    with pytest.raises(ValueError, match=err_msg):
        device()

    # 准备错误消息的正则表达式模式，指出至少需要一个输入数组，但实际得到的是两个输入
    err_msg = re.escape(
        "At least one input array expected after filtering with remove_none=True, "
        "remove_types=[str]. Got none. Original types: [NoneType, str]."
    )
    # 在特定的配置环境下，期望调用 device 函数时抛出 ValueError 异常，并匹配特定错误消息 err_msg
    with pytest.raises(ValueError, match=err_msg):
        device(None, "name")


# 定义一个内部类 Device 和内部类 Array，用于测试设备对象的比较和字符串表示
def test_device_inspection():
    class Device:
        def __init__(self, name):
            self.name = name

        def __eq__(self, device):
            return self.name == device.name

        def __hash__(self):
            raise TypeError("Device object is not hashable")

        def __str__(self):
            return self.name

    class Array:
        def __init__(self, device_name):
            self.device = Device(device_name)
    # 对设备模拟类进行健全性检查，确保其不可哈希，以准确处理某些数组库中的非可哈希设备对象。
    # 这导致`device`检查函数不应使用哈希查找表（特别是不应使用`set`）。
    with pytest.raises(TypeError):
        hash(Array("device").device)

    # 测试如果输入数组在不同设备上会引发错误
    err_msg = "Input arrays use different devices: cpu, mygpu"
    with pytest.raises(ValueError, match=err_msg):
        device(Array("cpu"), Array("mygpu"))

    # 测试预期的值返回情况
    array1 = Array("device")
    array2 = Array("device")

    assert array1.device == device(array1)
    assert array1.device == device(array1, array2)
    assert array1.device == device(array1, array1, array2)
# 跳过测试，如果数组 API 兼容性未配置好
@skip_if_array_api_compat_not_configured
# 使用参数化装饰器定义测试参数，测试库包括 "numpy", "array_api_strict", "torch"
@pytest.mark.parametrize("library", ["numpy", "array_api_strict", "torch"])
# 使用参数化装饰器定义测试参数：X 为输入数据，reduction 为函数，expected 为预期结果
@pytest.mark.parametrize(
    "X,reduction,expected",
    [
        # 测试 _nanmin 函数
        ([1, 2, numpy.nan], _nanmin, 1),
        ([1, -2, -numpy.nan], _nanmin, -2),
        ([numpy.inf, numpy.inf], _nanmin, numpy.inf),
        ([[1, 2, 3], [numpy.nan, numpy.nan, numpy.nan], [4, 5, 6.0]],
         partial(_nanmin, axis=0),
         [1.0, 2.0, 3.0]),
        ([[1, 2, 3], [numpy.nan, numpy.nan, numpy.nan], [4, 5, 6.0]],
         partial(_nanmin, axis=1),
         [1.0, numpy.nan, 4.0]),
        # 测试 _nanmax 函数
        ([1, 2, numpy.nan], _nanmax, 2),
        ([1, 2, numpy.nan], _nanmax, 2),
        ([-numpy.inf, -numpy.inf], _nanmax, -numpy.inf),
        ([[1, 2, 3], [numpy.nan, numpy.nan, numpy.nan], [4, 5, 6.0]],
         partial(_nanmax, axis=0),
         [4.0, 5.0, 6.0]),
        ([[1, 2, 3], [numpy.nan, numpy.nan, numpy.nan], [4, 5, 6.0]],
         partial(_nanmax, axis=1),
         [3.0, numpy.nan, 6.0]),
    ],
)
# 测试 NaN 降维操作，如 _nanmin 和 _nanmax
def test_nan_reductions(library, X, reduction, expected):
    """Check NaN reductions like _nanmin and _nanmax"""
    # 导入必要的库并跳过测试，如果库不可用
    xp = pytest.importorskip(library)

    # 使用 array_api_dispatch=True 的配置上下文
    with config_context(array_api_dispatch=True):
        # 执行降维操作，并获得结果
        result = reduction(xp.asarray(X))

    # 将结果转换为 NumPy 数组形式
    result = _convert_to_numpy(result, xp)
    # 断言结果与预期结果的近似性
    assert_allclose(result, expected)


# 使用参数化装饰器定义测试参数，测试不同的命名空间、设备和数据类型组合
@pytest.mark.parametrize(
    "namespace, _device, _dtype", yield_namespace_device_dtype_combinations()
)
# 测试 _ravel 函数
def test_ravel(namespace, _device, _dtype):
    # 获取适当的数组 API 对象
    xp = _array_api_for_tests(namespace, _device)

    # 创建输入数组
    array = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    # 使用适当的数组 API 对象创建数组
    array_xp = xp.asarray(array, device=_device)
    
    # 使用 array_api_dispatch=True 的配置上下文
    with config_context(array_api_dispatch=True):
        # 执行数组展平操作，并获得结果
        result = _ravel(array_xp)

    # 将结果转换为 NumPy 数组形式
    result = _convert_to_numpy(result, xp)
    # 计算预期的展平结果
    expected = numpy.ravel(array, order="C")

    # 断言结果与预期结果的近似性
    assert_allclose(expected, result)

    # 如果使用的是 NumPy 命名空间，验证结果是否是 C 连续的
    if _is_numpy_namespace(xp):
        assert numpy.asarray(result).flags["C_CONTIGUOUS"]


# 跳过测试，如果数组 API 兼容性未配置好
@skip_if_array_api_compat_not_configured
# 使用参数化装饰器定义测试参数，测试库包括 "cupy", "torch", "cupy.array_api"
@pytest.mark.parametrize("library", ["cupy", "torch", "cupy.array_api"])
# 测试在 GPU 支持的库中的 convert_to_numpy 函数
def test_convert_to_numpy_gpu(library):  # pragma: nocover
    """Check convert_to_numpy for GPU backed libraries."""
    # 导入必要的库并跳过测试，如果库不可用
    xp = pytest.importorskip(library)

    # 如果是 torch 库，并且 CUDA 没有构建，则跳过测试
    if library == "torch":
        if not xp.backends.cuda.is_built():
            pytest.skip("test requires cuda")
        # 创建在 CUDA 设备上的输入数据
        X_gpu = xp.asarray([1.0, 2.0, 3.0], device="cuda")
    else:
        # 创建在 CPU 或默认设备上的输入数据
        X_gpu = xp.asarray([1.0, 2.0, 3.0])

    # 使用 convert_to_numpy 函数将结果转换为 NumPy 数组形式
    X_cpu = _convert_to_numpy(X_gpu, xp=xp)
    # 创建预期输出的 NumPy 数组形式
    expected_output = numpy.asarray([1.0, 2.0, 3.0])
    # 断言结果与预期结果的近似性
    assert_allclose(X_cpu, expected_output)


# 测试在 PyTorch CPU 数组上的 convert_to_numpy 函数
def test_convert_to_numpy_cpu():
    """Check convert_to_numpy for PyTorch CPU arrays."""
    # 导入 pytest 模块，并确保导入成功，如果导入失败则跳过测试
    torch = pytest.importorskip("torch")
    
    # 创建一个 Torch 张量，包含数据 [1.0, 2.0, 3.0]，存储在 CPU 上
    X_torch = torch.asarray([1.0, 2.0, 3.0], device="cpu")
    
    # 将 Torch 张量转换为 NumPy 数组，使用了自定义函数 _convert_to_numpy，并传入 torch 作为参数
    X_cpu = _convert_to_numpy(X_torch, xp=torch)
    
    # 创建一个期望的 NumPy 数组，内容与 X_cpu 相同
    expected_output = numpy.asarray([1.0, 2.0, 3.0])
    
    # 使用 assert_allclose 断言函数来验证 X_cpu 与 expected_output 数组近似相等
    assert_allclose(X_cpu, expected_output)
class SimpleEstimator(BaseEstimator):
    # 定义一个简单的估算器类 SimpleEstimator，继承自 BaseEstimator

    def fit(self, X, y=None):
        # 定义 fit 方法，用于训练模型
        self.X_ = X
        # 将输入数据 X 赋值给实例变量 self.X_
        self.n_features_ = X.shape[0]
        # 计算输入数据 X 的特征数，并赋值给实例变量 self.n_features_
        return self
        # 返回当前实例对象本身

@skip_if_array_api_compat_not_configured
@pytest.mark.parametrize(
    "array_namespace, converter",
    [
        ("torch", lambda array: array.cpu().numpy()),
        ("array_api_strict", lambda array: numpy.asarray(array)),
        ("cupy.array_api", lambda array: array._array.get()),
    ],
)
def test_convert_estimator_to_ndarray(array_namespace, converter):
    """Convert estimator attributes to ndarray."""
    # 测试函数：将估算器属性转换为 ndarray 格式

    xp = pytest.importorskip(array_namespace)
    # 导入指定的数组命名空间模块，跳过如果配置未兼容数组 API

    X = xp.asarray([[1.3, 4.5]])
    # 创建一个 xp 数组，存储为 ndarray 格式

    est = SimpleEstimator().fit(X)
    # 创建 SimpleEstimator 实例并拟合数据 X

    new_est = _estimator_with_converted_arrays(est, converter)
    # 调用函数 _estimator_with_converted_arrays，将估算器属性转换为指定格式

    assert isinstance(new_est.X_, numpy.ndarray)
    # 断言新的估算器属性 X_ 的类型为 numpy.ndarray

@skip_if_array_api_compat_not_configured
def test_convert_estimator_to_array_api():
    """Convert estimator attributes to ArrayAPI arrays."""
    # 测试函数：将估算器属性转换为 ArrayAPI 数组

    xp = pytest.importorskip("array_api_strict")
    # 导入 ArrayAPI 严格模式，跳过如果配置未兼容数组 API

    X_np = numpy.asarray([[1.3, 4.5]])
    # 创建一个 numpy 数组 X_np

    est = SimpleEstimator().fit(X_np)
    # 创建 SimpleEstimator 实例并拟合数据 X_np

    new_est = _estimator_with_converted_arrays(est, lambda array: xp.asarray(array))
    # 调用函数 _estimator_with_converted_arrays，将估算器属性转换为 ArrayAPI 数组

    assert hasattr(new_est.X_, "__array_namespace__")
    # 断言新的估算器属性 X_ 具有 "__array_namespace__" 属性

def test_reshape_behavior():
    """Check reshape behavior with copy and is strict with non-tuple shape."""
    # 测试函数：检查重塑操作的行为，包括复制和对非元组形状的严格检查

    xp = _NumPyAPIWrapper()
    # 创建 NumPyAPIWrapper 实例 xp

    X = xp.asarray([[1, 2, 3], [3, 4, 5]])
    # 创建一个 xp 数组 X

    X_no_copy = xp.reshape(X, (-1,), copy=False)
    # 使用不复制的方式对 X 进行重塑操作，将结果存储在 X_no_copy 中
    assert X_no_copy.base is X
    # 断言 X_no_copy 的基础数据是 X

    X_copy = xp.reshape(X, (6, 1), copy=True)
    # 使用复制的方式对 X 进行重塑操作，将结果存储在 X_copy 中
    assert X_copy.base is not X.base
    # 断言 X_copy 的基础数据不是 X 的基础数据

    with pytest.raises(TypeError, match="shape must be a tuple"):
        xp.reshape(X, -1)
    # 使用 xp 对象的 reshape 方法，尝试对 X 进行形状为 -1 的重塑操作，预期会引发 TypeError 异常，且异常信息匹配 "shape must be a tuple"

@pytest.mark.parametrize("wrapper", [_ArrayAPIWrapper, _NumPyAPIWrapper])
def test_get_namespace_array_api_isdtype(wrapper):
    """Test isdtype implementation from _ArrayAPIWrapper and _NumPyAPIWrapper."""
    # 测试函数：测试 _ArrayAPIWrapper 和 _NumPyAPIWrapper 的 isdtype 方法实现

    if wrapper == _ArrayAPIWrapper:
        xp_ = pytest.importorskip("array_api_strict")
        xp = _ArrayAPIWrapper(xp_)
    else:
        xp = _NumPyAPIWrapper()

    assert xp.isdtype(xp.float32, xp.float32)
    # 断言 xp.float32 是否为 xp.float32 类型

    assert xp.isdtype(xp.float32, "real floating")
    # 断言 xp.float32 是否为实部浮点类型

    assert xp.isdtype(xp.float64, "real floating")
    # 断言 xp.float64 是否为实部浮点类型

    assert not xp.isdtype(xp.int32, "real floating")
    # 断言 xp.int32 是否不是实部浮点类型

    for dtype in supported_float_dtypes(xp):
        assert xp.isdtype(dtype, "real floating")
        # 断言 dtype 是否为实部浮点类型

    assert xp.isdtype(xp.bool, "bool")
    # 断言 xp.bool 是否为布尔类型

    assert not xp.isdtype(xp.float32, "bool")
    # 断言 xp.float32 是否不是布尔类型

    assert xp.isdtype(xp.int16, "signed integer")
    # 断言 xp.int16 是否为有符号整数类型

    assert not xp.isdtype(xp.uint32, "signed integer")
    # 断言 xp.uint32 是否不是有符号整数类型

    assert xp.isdtype(xp.uint16, "unsigned integer")
    # 断言 xp.uint16 是否为无符号整数类型

    assert not xp.isdtype(xp.int64, "unsigned integer")
    # 断言 xp.int64 是否不是无符号整数类型

    assert xp.isdtype(xp.int64, "numeric")
    # 断言 xp.int64 是否为数值类型

    assert xp.isdtype(xp.float32, "numeric")
    # 断言 xp.float32 是否为数值类型

    assert xp.isdtype(xp.uint32, "numeric")
    # 断言 xp.uint32 是否为数值类型

    assert not xp.isdtype(xp.float32, "complex floating")
    # 断言 xp.float32 是否不是复数浮点类型
    # 如果变量 wrapper 等于 _NumPyAPIWrapper
    if wrapper == _NumPyAPIWrapper:
        # 断言检查 xp.isdtype(xp.int8, "complex floating") 返回 False
        assert not xp.isdtype(xp.int8, "complex floating")
        # 断言检查 xp.isdtype(xp.complex64, "complex floating") 返回 True
        assert xp.isdtype(xp.complex64, "complex floating")
        # 断言检查 xp.isdtype(xp.complex128, "complex floating") 返回 True
        assert xp.isdtype(xp.complex128, "complex floating")
    
    # 使用 pytest.raises 捕获 ValueError 异常，且异常消息匹配 "Unrecognized data type"
    with pytest.raises(ValueError, match="Unrecognized data type"):
        # 断言检查 xp.isdtype(xp.int16, "unknown") 会引发异常
        assert xp.isdtype(xp.int16, "unknown")
# 使用 pytest 的参数化装饰器，将 yield_namespace_device_dtype_combinations() 的返回值分配给 namespace, _device, _dtype
@pytest.mark.parametrize(
    "namespace, _device, _dtype", yield_namespace_device_dtype_combinations()
)
# 定义一个测试函数 test_indexing_dtype，接受参数 namespace, _device, _dtype
def test_indexing_dtype(namespace, _device, _dtype):
    # 根据 namespace 和 _device 获取适合测试的数组 API
    xp = _array_api_for_tests(namespace, _device)

    # 如果是 32 位系统，断言 indexing_dtype 返回 xp.int32
    if _IS_32BIT:
        assert indexing_dtype(xp) == xp.int32
    # 否则断言 indexing_dtype 返回 xp.int64
    else:
        assert indexing_dtype(xp) == xp.int64


# 使用 pytest 的参数化装饰器，将 yield_namespace_device_dtype_combinations() 的返回值分配给 array_namespace, device, _
# 使用多个参数化装饰器，分别设置 invert、assume_unique、element_size、int_dtype 的参数组合
@pytest.mark.parametrize(
    "array_namespace, device, _", yield_namespace_device_dtype_combinations()
)
@pytest.mark.parametrize("invert", [True, False])
@pytest.mark.parametrize("assume_unique", [True, False])
@pytest.mark.parametrize("element_size", [6, 10, 14])
@pytest.mark.parametrize("int_dtype", ["int16", "int32", "int64", "uint8"])
# 定义一个测试函数 test_isin，接受参数 array_namespace, device, _, invert, assume_unique, element_size, int_dtype
def test_isin(
    array_namespace, device, _, invert, assume_unique, element_size, int_dtype
):
    # 根据 array_namespace 和 device 获取适合测试的数组 API
    xp = _array_api_for_tests(array_namespace, device)
    # 计算 element_size 的一半，并使用 int_dtype 类型创建一个数组 element
    r = element_size // 2
    element = 2 * numpy.arange(element_size).reshape((r, 2)).astype(int_dtype)
    # 创建一个测试用的元素数组 test_elements，类型为 int_dtype
    test_elements = numpy.array(numpy.arange(14), dtype=int_dtype)
    # 使用 xp.asarray 将 element 转换为数组对象 element_xp，并指定 device
    element_xp = xp.asarray(element, device=device)
    # 使用 xp.asarray 将 test_elements 转换为数组对象 test_elements_xp，并指定 device
    test_elements_xp = xp.asarray(test_elements, device=device)
    # 使用 numpy.isin 计算预期结果 expected
    expected = numpy.isin(
        element=element,
        test_elements=test_elements,
        assume_unique=assume_unique,
        invert=invert,
    )
    # 使用 config_context 开启 array_api_dispatch 上下文
    with config_context(array_api_dispatch=True):
        # 调用 _isin 函数计算结果 result
        result = _isin(
            element=element_xp,
            test_elements=test_elements_xp,
            xp=xp,
            assume_unique=assume_unique,
            invert=invert,
        )
    # 断言 _convert_to_numpy 函数将 result 转换为 NumPy 数组后与 expected 相等
    assert_array_equal(_convert_to_numpy(result, xp=xp), expected)


# 定义一个测试函数 test_get_namespace_and_device
def test_get_namespace_and_device():
    # 导入并检查 torch 库和 array_api_compat.torch，如果不存在则跳过测试
    torch = pytest.importorskip("torch")
    xp_torch = pytest.importorskip("array_api_compat.torch")
    # 创建一个在 CPU 上的 torch 张量 some_torch_tensor
    some_torch_tensor = torch.arange(3, device="cpu")
    # 创建一个 NumPy 数组 some_numpy_array
    some_numpy_array = numpy.arange(3)

    # 当 array API dispatch 禁用时，get_namespace_and_device 应该返回默认的 NumPy 封装和无设备信息
    namespace, is_array_api, device = get_namespace_and_device(some_torch_tensor)
    assert namespace is get_namespace(some_numpy_array)[0]
    assert not is_array_api
    assert device is None

    # 否则，使用 array API dispatch 启用时，get_namespace_and_device 应该返回 torch 的命名空间和设备信息
    with config_context(array_api_dispatch=True):
        namespace, is_array_api, device = get_namespace_and_device(some_torch_tensor)
        assert namespace is xp_torch
        assert is_array_api
        assert device == some_torch_tensor.device
```