# `D:\src\scipysrc\scipy\scipy\fft\tests\test_real_transforms.py`

```
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import math

from scipy.fft import dct, idct, dctn, idctn, dst, idst, dstn, idstn
import scipy.fft as fft
from scipy import fftpack
from scipy.conftest import array_api_compatible
from scipy._lib._array_api import copy, xp_assert_close

pytestmark = [array_api_compatible, pytest.mark.usefixtures("skip_xp_backends")]
skip_xp_backends = pytest.mark.skip_xp_backends

SQRT_2 = math.sqrt(2)

# scipy.fft wraps the fftpack versions but with normalized inverse transforms.
# So, the forward transforms and definitions are already thoroughly tested in
# fftpack/test_real_transforms.py

# 定义一个常量 SQRT_2，表示2的平方根
SQRT_2 = math.sqrt(2)

# 使用 pytestmark 标记装饰器，确保兼容数组 API，并跳过某些后端执行测试
pytestmark = [array_api_compatible, pytest.mark.usefixtures("skip_xp_backends")]

# 定义一个跳过某些后端的装饰器
skip_xp_backends = pytest.mark.skip_xp_backends

# 定义一个装饰器，用于标记跳过某些后端的测试用例
@skip_xp_backends(cpu_only=True)
# 参数化测试用例，测试一维变换的逆变换
@pytest.mark.parametrize("forward, backward", [(dct, idct), (dst, idst)])
# 参数化测试用例，测试不同类型的变换
@pytest.mark.parametrize("type", [1, 2, 3, 4])
# 参数化测试用例，测试不同长度的信号
@pytest.mark.parametrize("n", [2, 3, 4, 5, 10, 16])
# 参数化测试用例，测试不同轴向上的变换
@pytest.mark.parametrize("axis", [0, 1])
# 参数化测试用例，测试不同的归一化方式
@pytest.mark.parametrize("norm", [None, 'backward', 'ortho', 'forward'])
# 参数化测试用例，测试是否正交化
@pytest.mark.parametrize("orthogonalize", [False, True])
def test_identity_1d(forward, backward, type, n, axis, norm, orthogonalize, xp):
    # 测试恒等式 f^-1(f(x)) == x

    # 生成一个随机数组，作为输入信号 x
    x = xp.asarray(np.random.rand(n, n))

    # 进行前向变换，得到变换后的信号 y
    y = forward(x, type, axis=axis, norm=norm, orthogonalize=orthogonalize)
    # 进行逆变换，得到恢复的信号 z
    z = backward(y, type, axis=axis, norm=norm, orthogonalize=orthogonalize)
    # 使用 xp_assert_close 断言函数检查 z 是否与原始信号 x 近似相等
    xp_assert_close(z, x)

    # 对 y 进行边界填充
    pad = [(0, 0)] * 2
    pad[axis] = (0, 4)

    # 对填充后的 y2 进行逆变换，得到恢复的信号 z2
    y2 = xp.asarray(np.pad(np.asarray(y), pad, mode='edge'))
    z2 = backward(y2, type, n, axis, norm, orthogonalize=orthogonalize)
    # 使用 xp_assert_close 断言函数检查 z2 是否与原始信号 x 近似相等
    xp_assert_close(z2, x)


# 参数化测试用例，测试支持覆写原始输入信号 x 的情况
@skip_xp_backends(np_only=True,
                   reasons=['`overwrite_x` only supported for NumPy backend.'])
# 参数化测试用例，测试一维变换的逆变换
@pytest.mark.parametrize("forward, backward", [(dct, idct), (dst, idst)])
# 参数化测试用例，测试不同类型的变换
@pytest.mark.parametrize("type", [1, 2, 3, 4])
# 参数化测试用例，测试不同数据类型的信号
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64,
                                   np.complex64, np.complex128])
# 参数化测试用例，测试不同轴向上的变换
@pytest.mark.parametrize("axis", [0, 1])
# 参数化测试用例，测试不同的归一化方式
@pytest.mark.parametrize("norm", [None, 'backward', 'ortho', 'forward'])
# 参数化测试用例，测试是否覆写原始输入信号 x
@pytest.mark.parametrize("overwrite_x", [True, False])
def test_identity_1d_overwrite(forward, backward, type, dtype, axis, norm,
                               overwrite_x):
    # 测试恒等式 f^-1(f(x)) == x

    # 生成一个随机数组，作为输入信号 x
    x = np.random.rand(7, 8).astype(dtype)
    # 备份原始输入信号 x
    x_orig = x.copy()

    # 进行前向变换，得到变换后的信号 y
    y = forward(x, type, axis=axis, norm=norm, overwrite_x=overwrite_x)
    # 备份变换后的信号 y
    y_orig = y.copy()
    # 进行逆变换，得到恢复的信号 z
    z = backward(y, type, axis=axis, norm=norm, overwrite_x=overwrite_x)
    # 如果不覆写原始信号 x，则使用 assert_allclose 断言函数检查 z 是否与原始信号 x 近似相等
    if not overwrite_x:
        assert_allclose(z, x, rtol=1e-6, atol=1e-6)
        assert_array_equal(x, x_orig)
        assert_array_equal(y, y_orig)
    else:
        # 如果覆写原始信号 x，则使用 assert_allclose 断言函数检查 z 是否与原始信号 x_orig 近似相等
        assert_allclose(z, x_orig, rtol=1e-6, atol=1e-6)


# 跳过某些后端的装饰器
@skip_xp_backends(cpu_only=True)
# 参数化测试用例，测试多维变换的逆变换
@pytest.mark.parametrize("forward, backward", [(dctn, idctn), (dstn, idstn)])
# 参数化测试用例，测试不同类型的变换
@pytest.mark.parametrize("type", [1, 2, 3, 4])
# 使用 pytest 的 mark.parametrize 装饰器，为 test_identity_nd 函数设置参数化测试
@pytest.mark.parametrize("shape, axes",
                         [
                             ((4, 4), 0),        # 测试形状为 (4, 4)，轴为 0 的情况
                             ((4, 4), 1),        # 测试形状为 (4, 4)，轴为 1 的情况
                             ((4, 4), None),     # 测试形状为 (4, 4)，无指定轴的情况
                             ((4, 4), (0, 1)),   # 测试形状为 (4, 4)，指定多轴 (0, 1) 的情况
                             ((10, 12), None),   # 测试形状为 (10, 12)，无指定轴的情况
                             ((10, 12), (0, 1)), # 测试形状为 (10, 12)，指定多轴 (0, 1) 的情况
                             ((4, 5, 6), None),   # 测试形状为 (4, 5, 6)，无指定轴的情况
                             ((4, 5, 6), 1),      # 测试形状为 (4, 5, 6)，指定轴为 1 的情况
                             ((4, 5, 6), (0, 2)), # 测试形状为 (4, 5, 6)，指定多轴 (0, 2) 的情况
                         ])
@pytest.mark.parametrize("norm", [None, 'backward', 'ortho', 'forward'])  # 参数化正交化方式的测试
@pytest.mark.parametrize("orthogonalize", [False, True])  # 参数化正交化的测试

# 定义测试函数 test_identity_nd，测试 f^-1(f(x)) == x 的身份验证
def test_identity_nd(forward, backward, type, shape, axes, norm,
                     orthogonalize, xp):
    # 生成指定形状的随机数组 x
    x = xp.asarray(np.random.random(shape))

    # 如果指定了轴，则更新 shape 变量为指定轴的形状
    if axes is not None:
        shape = np.take(shape, axes)

    # 调用 forward 函数计算 y
    y = forward(x, type, axes=axes, norm=norm, orthogonalize=orthogonalize)
    # 调用 backward 函数计算 z
    z = backward(y, type, axes=axes, norm=norm, orthogonalize=orthogonalize)
    # 断言 z 与 x 在数值上接近
    xp_assert_close(z, x)

    # 根据轴的情况设定填充 pad 的方式
    if axes is None:
        pad = [(0, 4)] * x.ndim   # 对所有维度都进行 0 到 4 的填充
    elif isinstance(axes, int):
        pad = [(0, 0)] * x.ndim   # 对指定轴不进行填充
        pad[axes] = (0, 4)       # 对指定轴进行 0 到 4 的填充
    else:
        pad = [(0, 0)] * x.ndim   # 对所有维度都不进行填充

        # 针对每一个指定轴进行填充
        for a in axes:
            pad[a] = (0, 4)

    # TODO 编写一个与数组类型无关的 pad 函数

    # 将 y 转换为 xp 数组并进行边缘模式填充
    y2 = xp.asarray(np.pad(np.asarray(y), pad, mode='edge'))
    # 使用 backward 函数计算 z2
    z2 = backward(y2, type, shape, axes, norm, orthogonalize=orthogonalize)
    # 断言 z2 与 x 在数值上接近
    xp_assert_close(z2, x)


# 使用 skip_xp_backends 装饰器，指定仅在 NumPy 后端跳过测试，原因为 overwrite_x 仅支持 NumPy 后端
@skip_xp_backends(np_only=True,
                   reasons=['`overwrite_x` only supported for NumPy backend.'])
# 参数化 forward 和 backward 函数，测试 dctn/idctn 和 dstn/idstn 的情况
@pytest.mark.parametrize("forward, backward", [(dctn, idctn), (dstn, idstn)])
# 参数化 type 参数，测试值为 1, 2, 3, 4 的情况
@pytest.mark.parametrize("type", [1, 2, 3, 4])
# 参数化 shape 和 axes 参数，测试不同形状和轴的情况
@pytest.mark.parametrize("shape, axes",
                         [
                             ((4, 5), 0),        # 测试形状为 (4, 5)，轴为 0 的情况
                             ((4, 5), 1),        # 测试形状为 (4, 5)，轴为 1 的情况
                             ((4, 5), None),     # 测试形状为 (4, 5)，无指定轴的情况
                         ])
# 参数化 dtype 参数，测试不同数据类型的情况
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64,
                                   np.complex64, np.complex128])
# 参数化 norm 参数，测试不同正交化方式的情况
@pytest.mark.parametrize("norm", [None, 'backward', 'ortho', 'forward'])
# 参数化 overwrite_x 参数，测试覆盖 x 的情况
@pytest.mark.parametrize("overwrite_x", [False, True])

# 定义测试函数 test_identity_nd_overwrite，测试 f^-1(f(x)) == x 的身份验证，并考虑 overwrite_x 参数
def test_identity_nd_overwrite(forward, backward, type, shape, axes, dtype,
                               norm, overwrite_x):
    # 生成指定形状和数据类型的随机数组 x，并转换为 NumPy 数组
    x = np.random.random(shape).astype(dtype)
    # 创建 x 的副本 x_orig
    x_orig = x.copy()

    # 如果指定了轴，则更新 shape 变量为指定轴的形状
    if axes is not None:
        shape = np.take(shape, axes)

    # 调用 forward 函数计算 y
    y = forward(x, type, axes=axes, norm=norm)
    # 创建 y 的副本 y_orig
    y_orig = y.copy()
    # 调用 backward 函数计算 z
    z = backward(y, type, axes=axes, norm=norm)
    # 如果 overwrite_x 为 True，则断言 z 与 x_orig 在数值上接近
    if overwrite_x:
        assert_allclose(z, x_orig, rtol=1e-6, atol=1e-6)
    else:
        # 否则断言 z 与 x 在数值上接近，并检查 x 和 x_orig 是否相等，y 和 y_orig 是否相等
        assert_allclose(z, x, rtol=1e-6, atol=1e-6)
        assert_array_equal(x, x_orig)
        assert_array_equal(y, y_orig)


# 使用 skip_xp_backends 装饰器，指定仅在 CPU 后端跳过测试
@skip_xp_backends(cpu_only=True)
# 参数化 func 参数，测试 dct、dst、dctn、dstn 四种函数的情况
@pytest.mark.parametrize("func", ['dct', 'dst', 'dctn', 'dstn'])
# 使用 pytest 的装饰器，为函数 test_fftpack_equivalience 添加参数化测试
@pytest.mark.parametrize("type", [1, 2, 3, 4])
@pytest.mark.parametrize("norm", [None, 'backward', 'ortho', 'forward'])
def test_fftpack_equivalience(func, type, norm, xp):
    # 创建一个 8x16 的随机数组 x
    x = np.random.rand(8, 16)
    # 使用 xp.asarray 将 fftpack 模块中指定的函数 func 应用于 x，得到 fftpack_res
    fftpack_res = xp.asarray(getattr(fftpack, func)(x, type, norm=norm))
    # 将 x 转换为 xp.ndarray 类型
    x = xp.asarray(x)
    # 使用 getattr 调用 fft 模块中指定的函数 func，得到 fft_res
    fft_res = getattr(fft, func)(x, type, norm=norm)

    # 断言 fft_res 与 fftpack_res 的近似性
    xp_assert_close(fft_res, fftpack_res)


# 使用装饰器 skip_xp_backends，并为函数 test_orthogonalize_default 添加参数化测试
@skip_xp_backends(cpu_only=True)
@pytest.mark.parametrize("func", [dct, dst, dctn, dstn])
@pytest.mark.parametrize("type", [1, 2, 3, 4])
def test_orthogonalize_default(func, type, xp):
    # 测试当 norm="ortho" 时 orthogonalize 是默认行为，但其他情况下不是
    x = xp.asarray(np.random.rand(100))

    for norm, ortho in [
            ("forward", False),
            ("backward", False),
            ("ortho", True),
    ]:
        # 分别调用 func 函数，其中 norm 和 orthogonalize 参数根据循环设定
        a = func(x, type=type, norm=norm, orthogonalize=ortho)
        b = func(x, type=type, norm=norm)
        # 断言 a 和 b 的近似性
        xp_assert_close(a, b)


# 使用装饰器 skip_xp_backends，并为函数 test_orthogonalize_noop 添加参数化测试
@skip_xp_backends(cpu_only=True)
@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
@pytest.mark.parametrize("func, type", [
    (dct, 4), (dst, 1), (dst, 4)])
def test_orthogonalize_noop(func, type, norm, xp):
    # 测试 orthogonalize 对应函数的情况
    x = xp.asarray(np.random.rand(100))
    # 使用 orthogonalize=True 调用 func 函数得到 y1
    y1 = func(x, type=type, norm=norm, orthogonalize=True)
    # 使用 orthogonalize=False 调用 func 函数得到 y2
    y2 = func(x, type=type, norm=norm, orthogonalize=False)
    # 断言 y1 和 y2 的近似性
    xp_assert_close(y1, y2)


# 使用装饰器 skip_xp_backends，并为函数 test_orthogonalize_dct1 添加参数化测试
@skip_xp_backends('jax.numpy',
                  reasons=['jax arrays do not support item assignment'],
                  cpu_only=True)
@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
def test_orthogonalize_dct1(norm, xp):
    # 创建一个长度为 100 的随机数组 x
    x = xp.asarray(np.random.rand(100))

    # 复制 x 到 x2，并进行修改
    x2 = copy(x, xp=xp)
    x2[0] *= SQRT_2
    x2[-1] *= SQRT_2

    # 使用 orthogonalize=True 调用 dct 函数得到 y1
    y1 = dct(x, type=1, norm=norm, orthogonalize=True)
    # 使用 orthogonalize=False 调用 dct 函数得到 y2
    y2 = dct(x2, type=1, norm=norm, orthogonalize=False)

    # 对 y2 进行修正
    y2[0] /= SQRT_2
    y2[-1] /= SQRT_2

    # 断言 y1 和 y2 的近似性
    xp_assert_close(y1, y2)


# 使用装饰器 skip_xp_backends，并为函数 test_orthogonalize_dcst2 添加参数化测试
@skip_xp_backends('jax.numpy',
                  reasons=['jax arrays do not support item assignment'],
                  cpu_only=True)
@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
@pytest.mark.parametrize("func", [dct, dst])
def test_orthogonalize_dcst2(func, norm, xp):
    # 创建一个长度为 100 的随机数组 x
    x = xp.asarray(np.random.rand(100))
    # 使用 orthogonalize=True 调用 func 函数得到 y1
    y1 = func(x, type=2, norm=norm, orthogonalize=True)
    # 使用 orthogonalize=False 调用 func 函数得到 y2
    y2 = func(x, type=2, norm=norm, orthogonalize=False)

    # 根据 func 的类型对 y2 进行修正
    y2[0 if func == dct else -1] /= SQRT_2

    # 断言 y1 和 y2 的近似性
    xp_assert_close(y1, y2)


# 使用装饰器 skip_xp_backends，并为函数 test_orthogonalize_dcst3 添加参数化测试
@skip_xp_backends('jax.numpy',
                  reasons=['jax arrays do not support item assignment'],
                  cpu_only=True)
@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
@pytest.mark.parametrize("func", [dct, dst])
def test_orthogonalize_dcst3(func, norm, xp):
    # 创建一个长度为 100 的随机数组 x
    x = xp.asarray(np.random.rand(100))
    # 复制 x 到 x2，并根据 func 的类型进行修正
    x2 = copy(x, xp=xp)
    x2[0 if func == dct else -1] *= SQRT_2

    # 使用 orthogonalize=True 调用 func 函数得到 y1
    y1 = func(x, type=3, norm=norm, orthogonalize=True)
    # 使用 orthogonalize=False 调用 func 函数得到 y2
    y2 = func(x2, type=3, norm=norm, orthogonalize=False)
    xp_assert_close(y1, y2)


注释：


    # 使用自定义的断言函数 xp_assert_close 检查 y1 和 y2 的接近程度
```