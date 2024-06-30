# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_cython_blas.py`

```
# 导入必要的库
import numpy as np
import pytest

# 导入 Cython 实现的 BLAS 函数
from sklearn.utils._cython_blas import (
    ColMajor,
    NoTrans,
    RowMajor,
    Trans,
    _asum_memview,
    _axpy_memview,
    _copy_memview,
    _dot_memview,
    _gemm_memview,
    _gemv_memview,
    _ger_memview,
    _nrm2_memview,
    _rot_memview,
    _rotg_memview,
    _scal_memview,
)

# 导入用于测试的工具函数
from sklearn.utils._testing import assert_allclose

# 将 NumPy 的数据类型映射到 Cython 的数据类型
def _numpy_to_cython(dtype):
    cython = pytest.importorskip("cython")
    if dtype == np.float32:
        return cython.float
    elif dtype == np.float64:
        return cython.double

# 定义每种数据类型的相对误差容限
RTOL = {np.float32: 1e-6, np.float64: 1e-12}

# 定义行优先和列优先顺序的映射关系
ORDER = {RowMajor: "C", ColMajor: "F"}

# 空操作函数，直接返回输入值
def _no_op(x):
    return x

# 使用 pytest 的参数化装饰器，对 dot 函数进行测试
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dot(dtype):
    dot = _dot_memview[_numpy_to_cython(dtype)]

    # 创建随机数生成器
    rng = np.random.RandomState(0)
    # 生成随机数组并指定数据类型
    x = rng.random_sample(10).astype(dtype, copy=False)
    y = rng.random_sample(10).astype(dtype, copy=False)

    # 计算期望结果
    expected = x.dot(y)
    # 调用 Cython 实现的 dot 函数计算实际结果
    actual = dot(x, y)

    # 断言实际结果与期望结果之间的近似程度
    assert_allclose(actual, expected, rtol=RTOL[dtype])

# 使用 pytest 的参数化装饰器，对 asum 函数进行测试
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_asum(dtype):
    asum = _asum_memview[_numpy_to_cython(dtype)]

    # 创建随机数生成器
    rng = np.random.RandomState(0)
    # 生成随机数组并指定数据类型
    x = rng.random_sample(10).astype(dtype, copy=False)

    # 计算期望结果
    expected = np.abs(x).sum()
    # 调用 Cython 实现的 asum 函数计算实际结果
    actual = asum(x)

    # 断言实际结果与期望结果之间的近似程度
    assert_allclose(actual, expected, rtol=RTOL[dtype])

# 使用 pytest 的参数化装饰器，对 axpy 函数进行测试
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_axpy(dtype):
    axpy = _axpy_memview[_numpy_to_cython(dtype)]

    # 创建随机数生成器
    rng = np.random.RandomState(0)
    # 生成随机数组并指定数据类型
    x = rng.random_sample(10).astype(dtype, copy=False)
    y = rng.random_sample(10).astype(dtype, copy=False)
    alpha = 2.5

    # 计算期望结果
    expected = alpha * x + y
    # 调用 Cython 实现的 axpy 函数进行计算
    axpy(alpha, x, y)

    # 断言结果数组 y 与期望结果之间的近似程度
    assert_allclose(y, expected, rtol=RTOL[dtype])

# 使用 pytest 的参数化装饰器，对 nrm2 函数进行测试
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nrm2(dtype):
    nrm2 = _nrm2_memview[_numpy_to_cython(dtype)]

    # 创建随机数生成器
    rng = np.random.RandomState(0)
    # 生成随机数组并指定数据类型
    x = rng.random_sample(10).astype(dtype, copy=False)

    # 计算期望结果
    expected = np.linalg.norm(x)
    # 调用 Cython 实现的 nrm2 函数计算实际结果
    actual = nrm2(x)

    # 断言实际结果与期望结果之间的近似程度
    assert_allclose(actual, expected, rtol=RTOL[dtype])

# 使用 pytest 的参数化装饰器，对 copy 函数进行测试
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_copy(dtype):
    copy = _copy_memview[_numpy_to_cython(dtype)]

    # 创建随机数生成器
    rng = np.random.RandomState(0)
    # 生成随机数组并指定数据类型
    x = rng.random_sample(10).astype(dtype, copy=False)
    y = np.empty_like(x)

    # 计算期望结果
    expected = x.copy()
    # 调用 Cython 实现的 copy 函数执行复制操作
    copy(x, y)

    # 断言结果数组 y 与期望结果之间的近似程度
    assert_allclose(y, expected, rtol=RTOL[dtype])

# 使用 pytest 的参数化装饰器，对 scal 函数进行测试
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scal(dtype):
    scal = _scal_memview[_numpy_to_cython(dtype)]

    # 创建随机数生成器
    rng = np.random.RandomState(0)
    # 生成随机数组并指定数据类型
    x = rng.random_sample(10).astype(dtype, copy=False)
    alpha = 2.5

    # 计算期望结果
    expected = alpha * x
    # 调用 Cython 实现的 scal 函数进行缩放操作
    scal(alpha, x)

    # 断言结果数组 x 与期望结果之间的近似程度
    assert_allclose(x, expected, rtol=RTOL[dtype])

# 使用 pytest 的参数化装饰器，对 rotg 函数进行测试
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rotg(dtype):
    rotg = _rotg_memview[_numpy_to_cython(dtype)]
    # 使用种子值0初始化随机数生成器
    rng = np.random.RandomState(0)
    # 从随机数生成器中生成一个随机数，并将其转换为指定的数据类型dtype
    a = dtype(rng.randn())
    # 从随机数生成器中生成另一个随机数，并将其转换为指定的数据类型dtype
    b = dtype(rng.randn())
    # 初始化变量c和s为0.0
    c, s = 0.0, 0.0

    # 定义函数expected_rotg，计算给定a和b的旋转givens矩阵的期望值
    def expected_rotg(a, b):
        # 选择a和b中绝对值较大的那个数作为roe
        roe = a if abs(a) > abs(b) else b
        # 如果a和b都为0，则设定c=1, s=0, r=0, z=0
        if a == 0 and b == 0:
            c, s, r, z = (1, 0, 0, 0)
        else:
            # 计算r为(a^2 + b^2)的平方根乘以roe的符号
            r = np.sqrt(a**2 + b**2) * (1 if roe >= 0 else -1)
            # 计算c和s分别为a/r和b/r
            c, s = a / r, b / r
            # 计算z为s如果roe等于a，否则为1/c（注意要避免除以0错误）
            z = s if roe == a else (1 if c == 0 else 1 / c)
        return r, z, c, s

    # 调用expected_rotg函数，计算期望的旋转givens矩阵
    expected = expected_rotg(a, b)
    # 调用名为rotg的函数，计算实际的旋转givens矩阵
    actual = rotg(a, b, c, s)

    # 使用assert_allclose函数断言实际值与期望值的近似程度，允许的相对误差为RTOL[dtype]
    assert_allclose(actual, expected, rtol=RTOL[dtype])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
# 定义测试函数 test_rot，参数为 dtype
def test_rot(dtype):
    # 获取对应数据类型的旋转函数 _rot_memview
    rot = _rot_memview[_numpy_to_cython(dtype)]

    # 创建随机数生成器 rng，种子为 0
    rng = np.random.RandomState(0)
    # 生成长度为 10 的随机浮点数数组 x 和 y，类型为 dtype，且在赋值时不进行拷贝
    x = rng.random_sample(10).astype(dtype, copy=False)
    y = rng.random_sample(10).astype(dtype, copy=False)
    # 随机生成类型为 dtype 的常数 c 和 s
    c = dtype(rng.randn())
    s = dtype(rng.randn())

    # 计算期望的旋转后的数组
    expected_x = c * x + s * y
    expected_y = c * y - s * x

    # 调用旋转函数 rot 对 x 和 y 进行原地旋转操作
    rot(x, y, c, s)

    # 断言 x 和 expected_x 很接近
    assert_allclose(x, expected_x)
    # 断言 y 和 expected_y 很接近
    assert_allclose(y, expected_y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
# 定义测试函数 test_gemv，参数为 dtype, opA, transA, order
@pytest.mark.parametrize(
    "opA, transA", [(_no_op, NoTrans), (np.transpose, Trans)], ids=["NoTrans", "Trans"]
)
# 参数化 opA 和 transA
@pytest.mark.parametrize("order", [RowMajor, ColMajor], ids=["RowMajor", "ColMajor"])
# 参数化 order
def test_gemv(dtype, opA, transA, order):
    # 获取对应数据类型的矩阵向量乘函数 _gemv_memview
    gemv = _gemv_memview[_numpy_to_cython(dtype)]

    # 创建随机数生成器 rng，种子为 0
    rng = np.random.RandomState(0)
    # 生成随机矩阵 A，形状为 (20, 10)，类型为 dtype，不进行拷贝，根据 opA 进行变换，根据 order 指定顺序
    A = np.asarray(
        opA(rng.random_sample((20, 10)).astype(dtype, copy=False)), order=ORDER[order]
    )
    # 生成长度为 10 的随机浮点数数组 x，类型为 dtype，且在赋值时不进行拷贝
    x = rng.random_sample(10).astype(dtype, copy=False)
    # 生成长度为 20 的随机浮点数数组 y，类型为 dtype，且在赋值时不进行拷贝
    y = rng.random_sample(20).astype(dtype, copy=False)
    # 设置 alpha 和 beta 的值
    alpha, beta = 2.5, -0.5

    # 计算期望的结果
    expected = alpha * opA(A).dot(x) + beta * y
    # 调用矩阵向量乘函数 gemv 进行操作
    gemv(transA, alpha, A, x, beta, y)

    # 断言 y 和 expected 很接近，相对误差为 RTOL[dtype]
    assert_allclose(y, expected, rtol=RTOL[dtype])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
# 定义测试函数 test_ger，参数为 dtype, order
@pytest.mark.parametrize("order", [RowMajor, ColMajor], ids=["RowMajor", "ColMajor"])
# 参数化 order
def test_ger(dtype, order):
    # 获取对应数据类型的外积更新函数 _ger_memview
    ger = _ger_memview[_numpy_to_cython(dtype)]

    # 创建随机数生成器 rng，种子为 0
    rng = np.random.RandomState(0)
    # 生成长度为 10 的随机浮点数数组 x，类型为 dtype，且在赋值时不进行拷贝
    x = rng.random_sample(10).astype(dtype, copy=False)
    # 生成长度为 20 的随机浮点数数组 y，类型为 dtype，且在赋值时不进行拷贝
    y = rng.random_sample(20).astype(dtype, copy=False)
    # 生成形状为 (10, 20) 的随机浮点数数组 A，类型为 dtype，且在赋值时不进行拷贝，根据 order 指定顺序
    A = np.asarray(
        rng.random_sample((10, 20)).astype(dtype, copy=False), order=ORDER[order]
    )
    # 设置 alpha 的值
    alpha = 2.5

    # 计算期望的结果
    expected = alpha * np.outer(x, y) + A
    # 调用外积更新函数 ger 进行操作
    ger(alpha, x, y, A)

    # 断言 A 和 expected 很接近，相对误差为 RTOL[dtype]
    assert_allclose(A, expected, rtol=RTOL[dtype])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
# 定义测试函数 test_gemm，参数为 dtype, opA, transA, opB, transB, order
@pytest.mark.parametrize(
    "opB, transB", [(_no_op, NoTrans), (np.transpose, Trans)], ids=["NoTrans", "Trans"]
)
# 参数化 opB 和 transB
@pytest.mark.parametrize(
    "opA, transA", [(_no_op, NoTrans), (np.transpose, Trans)], ids=["NoTrans", "Trans"]
)
# 参数化 opA 和 transA
@pytest.mark.parametrize("order", [RowMajor, ColMajor], ids=["RowMajor", "ColMajor"])
# 参数化 order
def test_gemm(dtype, opA, transA, opB, transB, order):
    # 获取对应数据类型的矩阵乘函数 _gemm_memview
    gemm = _gemm_memview[_numpy_to_cython(dtype)]

    # 创建随机数生成器 rng，种子为 0
    rng = np.random.RandomState(0)
    # 生成随机矩阵 A，形状为 (30, 10)，类型为 dtype，不进行拷贝，根据 opA 进行变换，根据 order 指定顺序
    A = np.asarray(
        opA(rng.random_sample((30, 10)).astype(dtype, copy=False)), order=ORDER[order]
    )
    # 生成随机矩阵 B，形状为 (10, 20)，类型为 dtype，不进行拷贝，根据 opB 进行变换，根据 order 指定顺序
    B = np.asarray(
        opB(rng.random_sample((10, 20)).astype(dtype, copy=False)), order=ORDER[order]
    )
    # 生成随机矩阵 C，形状为 (30, 20)，类型为 dtype，不进行拷贝，根据 order 指定顺序
    C = np.asarray(
        rng.random_sample((30, 20)).astype(dtype, copy=False), order=ORDER[order]
    )
    # 设置 alpha 和 beta 的值
    alpha, beta = 2.5, -0.5

    # 计算期望的结果
    expected = alpha * opA(A).dot(opB(B)) + beta * C
    # 调用矩阵乘函数 gemm 进行操作
    gemm(transA, transB, alpha, A, B, beta, C)

    # 断言 C 和 expected 很接近，相对误差为 RTOL[dtype]
    assert_allclose(C, expected, rtol=RTOL[dtype])
```