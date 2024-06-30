# `D:\src\scipysrc\scipy\scipy\special\tests\test_support_alternative_backends.py`

```
import pytest  # 导入 pytest 库

from scipy.special._support_alternative_backends import (get_array_special_func,
                                                         array_special_func_map)  # 从 scipy.special._support_alternative_backends 导入函数和映射
from scipy.conftest import array_api_compatible  # 从 scipy.conftest 导入 array_api_compatible 函数
from scipy import special  # 导入 scipy 的 special 模块
from scipy._lib._array_api import xp_assert_close, is_jax  # 从 scipy._lib._array_api 导入 xp_assert_close 和 is_jax
from scipy._lib.array_api_compat import numpy as np  # 从 scipy._lib.array_api_compat 导入 numpy 并重命名为 np

try:
    import array_api_strict  # 尝试导入 array_api_strict 模块
    HAVE_ARRAY_API_STRICT = True  # 如果导入成功，则标记为 True
except ImportError:
    HAVE_ARRAY_API_STRICT = False  # 如果导入失败，则标记为 False


@pytest.mark.skipif(not HAVE_ARRAY_API_STRICT,
                    reason="`array_api_strict` not installed")  # 使用 pytest.mark.skipif 装饰器，如果没有安装 array_api_strict 则跳过该测试
def test_dispatch_to_unrecognize_library():
    xp = array_api_strict  # 将 array_api_strict 赋值给 xp
    f = get_array_special_func('ndtr', xp=xp, n_array_args=1)  # 调用 get_array_special_func 函数获取 'ndtr' 的特殊函数，指定使用 xp，有一个数组参数
    x = [1, 2, 3]  # 定义输入数组 x
    res = f(xp.asarray(x))  # 调用获取到的函数 f，传入 xp.asarray 转换后的 x，计算结果 res
    ref = xp.asarray(special.ndtr(np.asarray(x)))  # 调用 scipy.special.ndtr 计算参考结果 ref，使用 xp.asarray 转换后的输入 x
    xp_assert_close(res, ref, xp=xp)  # 调用 xp_assert_close 检查 res 和 ref 的近似性，指定使用 xp


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int64'])  # 参数化测试，测试数据类型包括 'float32', 'float64', 'int64'
@pytest.mark.skipif(not HAVE_ARRAY_API_STRICT,
                    reason="`array_api_strict` not installed")  # 如果没有安装 array_api_strict，则跳过该测试
def test_rel_entr_generic(dtype):
    xp = array_api_strict  # 将 array_api_strict 赋值给 xp
    f = get_array_special_func('rel_entr', xp=xp, n_array_args=2)  # 调用 get_array_special_func 函数获取 'rel_entr' 的特殊函数，指定使用 xp，有两个数组参数
    dtype_np = getattr(np, dtype)  # 获取 np 中对应 dtype 的数据类型对象
    dtype_xp = getattr(xp, dtype)  # 获取 xp 中对应 dtype 的数据类型对象
    x, y = [-1, 0, 0, 1], [1, 0, 2, 3]  # 定义输入数组 x 和 y

    x_xp, y_xp = xp.asarray(x, dtype=dtype_xp), xp.asarray(y, dtype=dtype_xp)  # 使用 xp.asarray 将 x 和 y 转换为 xp 中对应 dtype 的数组
    res = f(x_xp, y_xp)  # 调用获取到的函数 f，传入转换后的 x_xp 和 y_xp，计算结果 res

    x_np, y_np = np.asarray(x, dtype=dtype_np), np.asarray(y, dtype=dtype_np)  # 使用 np.asarray 将 x 和 y 转换为 np 中对应 dtype 的数组
    ref = special.rel_entr(x_np[-1], y_np[-1])  # 调用 scipy.special.rel_entr 计算参考结果 ref，使用 np 中对应的最后一个元素
    ref = np.asarray([np.inf, 0, 0, ref], dtype=ref.dtype)  # 使用 np.asarray 将参考结果 ref 转换为对应的数据类型

    xp_assert_close(res, xp.asarray(ref), xp=xp)  # 调用 xp_assert_close 检查 res 和 ref 的近似性，指定使用 xp


@pytest.mark.fail_slow(5)  # 使用 pytest.mark.fail_slow 装饰器，标记该测试为 "fail_slow" 类型，阈值为 5
@array_api_compatible  # 使用 array_api_compatible 装饰器
# @pytest.mark.skip_xp_backends('numpy', reasons=['skip while debugging'])
# @pytest.mark.usefixtures("skip_xp_backends")
# `reversed` is for developer convenience: test new function first = less waiting
@pytest.mark.parametrize('f_name_n_args', reversed(array_special_func_map.items()))  # 参数化测试，遍历 array_special_func_map 的键值对，反转顺序
@pytest.mark.parametrize('dtype', ['float32', 'float64'])  # 参数化测试，测试数据类型包括 'float32', 'float64'
@pytest.mark.parametrize('shapes', [[(0,)]*4, [tuple()]*4, [(10,)]*4,
                                    [(10,), (11, 1), (12, 1, 1), (13, 1, 1, 1)]])  # 参数化测试，测试不同的 shapes 组合
def test_support_alternative_backends(xp, f_name_n_args, dtype, shapes):
    f_name, n_args = f_name_n_args  # 解包 f_name_n_args，获取函数名称和参数个数
    shapes = shapes[:n_args]  # 将 shapes 切片，保留与参数个数相符合的部分
    f = getattr(special, f_name)  # 获取 special 模块中的函数对象 f_name

    dtype_np = getattr(np, dtype)  # 获取 np 中对应 dtype 的数据类型对象
    dtype_xp = getattr(xp, dtype)  # 获取 xp 中对应 dtype 的数据类型对象

    # # To test the robustness of the alternative backend's implementation,
    # # use Hypothesis to generate arguments
    # from hypothesis import given, strategies, reproduce_failure, assume
    # import hypothesis.extra.numpy as npst
    # @given(data=strategies.data())
    # mbs = npst.mutually_broadcastable_shapes(num_shapes=n_args)
    # shapes, final_shape = data.draw(mbs)
    # elements = dict(allow_subnormal=False)  # consider min_value, max_value
    # args_np = [np.asarray(data.draw(npst.arrays(dtype_np, shape, elements=elements)),
    #                       dtype=dtype_np)
    # 对于每个形状，使用随机数生成器生成符合正态分布的参数
    rng = np.random.default_rng(984254252920492019)
    args_np = [rng.standard_normal(size=shape, dtype=dtype_np) for shape in shapes]

    # 如果是 JAX 环境并且函数名是 'gammaincc'，或者函数名是 'chdtrc'，
    # 对第一个和第二个参数取绝对值，以确保参数符合函数要求
    if (is_jax(xp) and f_name == 'gammaincc'  # google/jax#20699
            or f_name == 'chdtrc'):  # gh-20972
        args_np[0] = np.abs(args_np[0])
        args_np[1] = np.abs(args_np[1])

    # 将生成的 NumPy 数组参数转换为对应的 xp 数组参数
    args_xp = [xp.asarray(arg[()], dtype=dtype_xp) for arg in args_np]

    # 使用 xp 数组参数调用函数 f，并获取结果
    res = f(*args_xp)

    # 使用生成的 NumPy 数组参数调用函数 f，并转换为 xp 数组，获取参考结果
    ref = xp.asarray(f(*args_np), dtype=dtype_xp)

    # 计算数值精度的阈值
    eps = np.finfo(dtype_np).eps

    # 使用 xp_assert_close 函数比较计算结果 res 和参考结果 ref，确保它们在数值上接近
    xp_assert_close(res, ref, atol=10*eps)
```