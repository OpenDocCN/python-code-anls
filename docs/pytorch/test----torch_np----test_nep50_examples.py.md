# `.\pytorch\test\torch_np\test_nep50_examples.py`

```
# Owner(s): ["module: dynamo"]

"""Test examples for NEP 50."""

# 导入必要的库和模块
import itertools
from unittest import skipIf as skipif, SkipTest

try:
    # 尝试导入 NumPy 并检查版本信息
    import numpy as _np

    v = _np.__version__.split(".")
    HAVE_NUMPY = int(v[0]) >= 1 and int(v[1]) >= 24
except ImportError:
    HAVE_NUMPY = False

# 导入 torch._numpy 模块及其部分函数
import torch._numpy as tnp
from torch._numpy import (  # noqa: F401
    array,
    bool_,  # noqa: F401
    complex128,
    complex64,
    float32,
    float64,
    inf,
    int16,
    int32,  # noqa: F401
    int64,
    uint8,
)
# 导入必要的测试函数和类
from torch._numpy.testing import assert_allclose

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)

# 为 uint16 定义一个别名，可以是任何类型，与 uint8 相同
uint16 = uint8  # can be anything here, see below


# from numpy import array, uint8, uint16, int64, float32, float64, inf
# from numpy.testing import assert_allclose
# import numpy as np
# np._set_promotion_state('weak')

from pytest import raises as assert_raises

# 初始化一个未更改的变量
unchanged = None

# 定义测试用例的表达式及其预期结果
examples = {
    "uint8(1) + 2": (int64(3), uint8(3)),
    "array([1], uint8) + int64(1)": (array([2], uint8), array([2], int64)),
    "array([1], uint8) + array(1, int64)": (array([2], uint8), array([2], int64)),
    "array([1.], float32) + float64(1.)": (
        array([2.0], float32),
        array([2.0], float64),
    ),
    "array([1.], float32) + array(1., float64)": (
        array([2.0], float32),
        array([2.0], float64),
    ),
    "array([1], uint8) + 1": (array([2], uint8), unchanged),
    "array([1], uint8) + 200": (array([201], uint8), unchanged),
    "array([100], uint8) + 200": (array([44], uint8), unchanged),
    "array([1], uint8) + 300": (array([301], uint16), Exception),
    "uint8(1) + 300": (int64(301), Exception),
    "uint8(100) + 200": (int64(301), uint8(44)),  # and RuntimeWarning
    "float32(1) + 3e100": (float64(3e100), float32(inf)),  # and RuntimeWarning [T7]
    "array([1.0], float32) + 1e-14 == 1.0": (array([True]), unchanged),
    "array([0.1], float32) == float64(0.1)": (array([True]), array([False])),
    "array(1.0, float32) + 1e-14 == 1.0": (array(False), array(True)),
    "array([1.], float32) + 3": (array([4.0], float32), unchanged),
    "array([1.], float32) + int64(3)": (array([4.0], float32), array([4.0], float64)),
    "3j + array(3, complex64)": (array(3 + 3j, complex128), array(3 + 3j, complex64)),
    "float32(1) + 1j": (array(1 + 1j, complex128), array(1 + 1j, complex64)),
    "int32(1) + 5j": (array(1 + 5j, complex128), unchanged),
    # additional examples from the NEP text
    "int16(2) + 2": (int64(4), int16(4)),
    "int16(4) + 4j": (complex128(4 + 4j), unchanged),
    "float32(5) + 5j": (complex128(5 + 5j), complex64(5 + 5j)),
    "bool_(True) + 1": (int64(2), unchanged),
    "True + uint8(2)": (uint8(3), unchanged),
}

# 根据是否有 NumPy，决定是否跳过测试
@skipif(not HAVE_NUMPY, reason="NumPy not found")
# 实例化参数化测试类
@instantiate_parametrized_tests
class TestNEP50Table(TestCase):
    @parametrize("example", examples)
    # 测试参数化的例子
    # 定义一个测试函数，用于测试 NEP50 中的异常情况
    def test_nep50_exceptions(self, example):
        # 从给定的例子中获取旧值和新值
        old, new = examples[example]

        # 如果新值是 Exception 类型
        if new == Exception:
            # 使用 assert_raises 上下文，检测是否会抛出 OverflowError 异常
            with assert_raises(OverflowError):
                # 执行给定的例子代码
                eval(example)

        else:
            # 否则，执行例子代码，并获取结果
            result = eval(example)

            # 如果新值是未更改的（意味着预期结果应该与旧值相同）
            if new is unchanged:
                new = old

            # 使用 assert_allclose 检查计算结果与新值的接近程度，设置允许的误差范围为 1e-16
            assert_allclose(result, new, atol=1e-16)
            # 检查结果的数据类型是否与新值的数据类型相同
            assert result.dtype == new.dtype
# ### Directly compare to numpy ###

# 弱类型数据示例
weaks = (True, 1, 2.0, 3j)
# 非弱类型数据示例，使用了tnp模块创建的各种数据类型的数组
non_weaks = (
    tnp.asarray(True),
    tnp.uint8(1),
    tnp.int8(1),
    tnp.int32(1),
    tnp.int64(1),
    tnp.float32(1),
    tnp.float64(1),
    tnp.complex64(1),
    tnp.complex128(1),
)
# 如果有NumPy库可用，定义NumPy数据类型数组，否则为None
if HAVE_NUMPY:
    dtypes = (
        None,
        _np.bool_,
        _np.uint8,
        _np.int8,
        _np.int32,
        _np.int64,
        _np.float32,
        _np.float64,
        _np.complex64,
        _np.complex128,
    )
else:
    dtypes = (None,)

# ufunc名称到支持的数据类型列表的映射字典
corners = {
    "true_divide": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "divide": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "arctan2": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "copysign": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "heaviside": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "ldexp": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "power": ["uint8"],
    "nextafter": ["float32"],
}

# 如果没有NumPy库，则使用skipif装饰器跳过测试类
@skipif(not HAVE_NUMPY, reason="NumPy not found")
# 实例化参数化测试的装饰器
@instantiate_parametrized_tests
class TestCompareToNumpy(TestCase):
    # 参数化测试方法，测试直接比较操作
    @parametrize("scalar, array, dtype", itertools.product(weaks, non_weaks, dtypes))
    def test_direct_compare(self, scalar, array, dtype):
        # 使用NumPy进行比较，考虑NEP 50的影响
        try:
            state = _np._get_promotion_state()
            _np._set_promotion_state("weak")

            if dtype is not None:
                kwargs = {"dtype": dtype}
            try:
                # 使用NumPy进行加法运算
                result_numpy = _np.add(scalar, array.tensor.numpy(), **kwargs)
            except Exception:
                return

            kwargs = {}
            if dtype is not None:
                kwargs = {"dtype": getattr(tnp, dtype.__name__)}
            # 使用tnp模块进行加法运算，并获取其NumPy数组表示
            result = tnp.add(scalar, array, **kwargs).tensor.numpy()
            # 断言结果的数据类型应该相同
            assert result.dtype == result_numpy.dtype
            # 断言结果数组应该相等
            assert result == result_numpy

        finally:
            _np._set_promotion_state(state)

    # 参数化测试方法，测试tnp模块的二元ufuncs函数
    @parametrize("name", tnp._ufuncs._binary)
    @parametrize("scalar, array", itertools.product(weaks, non_weaks))
    # 定义一个测试函数，用于比较通用函数（ufuncs）的行为
    def test_compare_ufuncs(self, name, scalar, array):
        # 如果函数名在corners字典中，并且数组的dtype或标量的dtype在corners[name]中，则跳过测试
        if name in corners and (
            array.dtype.name in corners[name]
            or tnp.asarray(scalar).dtype.name in corners[name]
        ):
            # 抛出跳过测试的异常，显示跳过的原因
            raise SkipTest(f"{name}(..., dtype=array.dtype)")

        try:
            # 获取当前numpy的提升状态并保存
            state = _np._get_promotion_state()
            # 设置numpy的提升状态为"weak"
            _np._set_promotion_state("weak")

            # 对于一些特殊的ufunc，直接返回，不进行进一步操作
            if name in ["matmul", "modf", "divmod", "ldexp"]:
                return

            # 获取名为name的ufunc的对象
            ufunc = getattr(tnp, name)
            # 获取numpy中相同名字的ufunc对象
            ufunc_numpy = getattr(_np, name)

            try:
                # 使用ufunc对标量和数组进行操作，得到结果
                result = ufunc(scalar, array)
            except RuntimeError:
                # 如果出现RuntimeError，则记录结果为None
                # 例如："bitwise_xor_cpu" 在 'ComplexDouble' 等类型上未实现
                result = None

            try:
                # 使用numpy的ufunc对标量和数组进行操作，得到numpy的结果
                result_numpy = ufunc_numpy(scalar, array.tensor.numpy())
            except TypeError:
                # 如果出现TypeError，则记录结果为None
                # 例如：ufunc 'hypot' 不支持输入的类型
                result_numpy = None

            # 如果结果和numpy结果都不为None，则断言它们的数据类型一致
            if result is not None and result_numpy is not None:
                assert result.tensor.numpy().dtype == result_numpy.dtype

        finally:
            # 恢复numpy的提升状态为先前保存的状态
            _np._set_promotion_state(state)
# 如果当前脚本作为主程序运行，执行以下代码块
if __name__ == "__main__":
    # 调用函数运行测试函数
    run_tests()
```