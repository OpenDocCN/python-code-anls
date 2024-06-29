# `.\numpy\numpy\linalg\tests\test_linalg.py`

```py
# 导入所需的标准库和第三方库模块
import os
import sys
import itertools
import traceback
import textwrap
import subprocess
import pytest

# 导入 NumPy 库及其部分子模块和函数
import numpy as np
from numpy import array, single, double, csingle, cdouble, dot, identity, matmul
from numpy._core import swapaxes
from numpy.exceptions import AxisError
from numpy import multiply, atleast_2d, inf, asarray
from numpy import linalg
from numpy.linalg import matrix_power, norm, matrix_rank, multi_dot, LinAlgError
from numpy.linalg._linalg import _multi_dot_matrix_chain_order
from numpy.testing import (
    assert_, assert_equal, assert_raises, assert_array_equal,
    assert_almost_equal, assert_allclose, suppress_warnings,
    assert_raises_regex, HAS_LAPACK64, IS_WASM
    )

try:
    import numpy.linalg.lapack_lite
except ImportError:
    # 在没有 BLAS/LAPACK 的情况下可能会导致 ImportError，需要处理这种情况
    # 应确保整个测试套件不会因此而中断
    pass


def consistent_subclass(out, in_):
    # 对于 ndarray 的子类输入，输出应保持相同的子类类型
    # （非 ndarray 输入将转换为 ndarray）。
    return type(out) is (type(in_) if isinstance(in_, np.ndarray)
                         else np.ndarray)


old_assert_almost_equal = assert_almost_equal


def assert_almost_equal(a, b, single_decimal=6, double_decimal=12, **kw):
    # 根据输入数组的数据类型确定精度，执行数值近似相等的断言
    if asarray(a).dtype.type in (single, csingle):
        decimal = single_decimal
    else:
        decimal = double_decimal
    old_assert_almost_equal(a, b, decimal=decimal, **kw)


def get_real_dtype(dtype):
    # 根据复数数据类型确定其对应的实数数据类型
    return {single: single, double: double,
            csingle: single, cdouble: double}[dtype]


def get_complex_dtype(dtype):
    # 根据实数数据类型确定其对应的复数数据类型
    return {single: csingle, double: cdouble,
            csingle: csingle, cdouble: cdouble}[dtype]


def get_rtol(dtype):
    # 根据数据类型选择一个安全的相对误差阈值
    if dtype in (single, csingle):
        return 1e-5
    else:
        return 1e-11


# 用于对测试进行分类的标签集合
all_tags = {
  'square', 'nonsquare', 'hermitian',  # 互斥的基本类别
  'generalized', 'size-0', 'strided' # 可选的附加类别
}


class LinalgCase:
    def __init__(self, name, a, b, tags=set()):
        """
        用于测试用例的一组参数，包括标识名称、操作数 a 和 b，以及一组标签用于过滤测试
        """
        assert_(isinstance(name, str))
        self.name = name
        self.a = a
        self.b = b
        self.tags = frozenset(tags)  # 使用 frozenset 防止标签共享

    def check(self, do):
        """
        在此测试用例上运行函数 `do`，并扩展参数
        """
        do(self.a, self.b, tags=self.tags)

    def __repr__(self):
        return f'<LinalgCase: {self.name}>'


def apply_tag(tag, cases):
    """
    将给定的标签（字符串）添加到每个测试用例（LinalgCase 对象列表）中
    """
    assert tag in all_tags, "Invalid tag"
    # 遍历列表中的每个案例对象
    for case in cases:
        # 将当前案例对象的标签与给定的标签合并
        case.tags = case.tags | {tag}
    # 返回更新后的案例对象列表
    return cases
#
# Base test cases
#

# 设置随机种子，以便重现结果
np.random.seed(1234)

# 创建一个空列表来存储测试用例
CASES = []

# square test cases
# 向 CASES 列表添加“square”标签的测试用例
CASES += apply_tag('square', [
    # 创建一个 LinalgCase 对象，包含单精度数组和预期结果
    LinalgCase("single",
               array([[1., 2.], [3., 4.]], dtype=single),
               array([2., 1.], dtype=single)),
    LinalgCase("double",
               array([[1., 2.], [3., 4.]], dtype=double),
               array([2., 1.], dtype=double)),
    LinalgCase("double_2",
               array([[1., 2.], [3., 4.]], dtype=double),
               array([[2., 1., 4.], [3., 4., 6.]], dtype=double)),
    LinalgCase("csingle",
               array([[1. + 2j, 2 + 3j], [3 + 4j, 4 + 5j]], dtype=csingle),
               array([2. + 1j, 1. + 2j], dtype=csingle)),
    LinalgCase("cdouble",
               array([[1. + 2j, 2 + 3j], [3 + 4j, 4 + 5j]], dtype=cdouble),
               array([2. + 1j, 1. + 2j], dtype=cdouble)),
    LinalgCase("cdouble_2",
               array([[1. + 2j, 2 + 3j], [3 + 4j, 4 + 5j]], dtype=cdouble),
               array([[2. + 1j, 1. + 2j, 1 + 3j], [1 - 2j, 1 - 3j, 1 - 6j]], dtype=cdouble)),
    LinalgCase("0x0",
               np.empty((0, 0), dtype=double),
               np.empty((0,), dtype=double),
               tags={'size-0'}),
    LinalgCase("8x8",
               np.random.rand(8, 8),
               np.random.rand(8)),
    LinalgCase("1x1",
               np.random.rand(1, 1),
               np.random.rand(1)),
    LinalgCase("nonarray",
               [[1, 2], [3, 4]],
               [2, 1]),
])

# non-square test-cases
# 向 CASES 列表添加“nonsquare”标签的测试用例
CASES += apply_tag('nonsquare', [
    LinalgCase("single_nsq_1",
               array([[1., 2., 3.], [3., 4., 6.]], dtype=single),
               array([2., 1.], dtype=single)),
    LinalgCase("single_nsq_2",
               array([[1., 2.], [3., 4.], [5., 6.]], dtype=single),
               array([2., 1., 3.], dtype=single)),
    LinalgCase("double_nsq_1",
               array([[1., 2., 3.], [3., 4., 6.]], dtype=double),
               array([2., 1.], dtype=double)),
    LinalgCase("double_nsq_2",
               array([[1., 2.], [3., 4.], [5., 6.]], dtype=double),
               array([2., 1., 3.], dtype=double)),
    LinalgCase("csingle_nsq_1",
               array(
                   [[1. + 1j, 2. + 2j, 3. - 3j], [3. - 5j, 4. + 9j, 6. + 2j]], dtype=csingle),
               array([2. + 1j, 1. + 2j], dtype=csingle)),
    LinalgCase("csingle_nsq_2",
               array(
                   [[1. + 1j, 2. + 2j], [3. - 3j, 4. - 9j], [5. - 4j, 6. + 8j]], dtype=csingle),
               array([2. + 1j, 1. + 2j, 3. - 3j], dtype=csingle)),
    LinalgCase("cdouble_nsq_1",
               array(
                   [[1. + 1j, 2. + 2j, 3. - 3j], [3. - 5j, 4. + 9j, 6. + 2j]], dtype=cdouble),
               array([2. + 1j, 1. + 2j], dtype=cdouble)),
    LinalgCase("cdouble_nsq_2",
               array(
                   [[1. + 1j, 2. + 2j], [3. - 3j, 4. - 9j], [5. - 4j, 6. + 8j]], dtype=cdouble),
               array([2. + 1j, 1. + 2j, 3. - 3j], dtype=cdouble)),
    # 创建一个 LinalgCase 对象，用于表示线性代数操作的测试用例，名称为 "cdouble_nsq_1_2"
    LinalgCase("cdouble_nsq_1_2",
               # 创建一个复数数组，包含两行三列的复数值矩阵
               array([[1. + 1j, 2. + 2j, 3. - 3j], [3. - 5j, 4. + 9j, 6. + 2j]], dtype=cdouble),
               # 创建一个复数数组，包含两行两列的复数值矩阵
               array([[2. + 1j, 1. + 2j], [1 - 1j, 2 - 2j]], dtype=cdouble)),
    
    # 创建一个 LinalgCase 对象，名称为 "cdouble_nsq_2_2"
    LinalgCase("cdouble_nsq_2_2",
               # 创建一个复数数组，包含三行两列的复数值矩阵
               array([[1. + 1j, 2. + 2j], [3. - 3j, 4. - 9j], [5. - 4j, 6. + 8j]], dtype=cdouble),
               # 创建一个复数数组，包含三行两列的复数值矩阵
               array([[2. + 1j, 1. + 2j], [1 - 1j, 2 - 2j], [1 - 1j, 2 - 2j]], dtype=cdouble)),
    
    # 创建一个 LinalgCase 对象，名称为 "8x11"
    LinalgCase("8x11",
               # 创建一个 8x11 的随机浮点数数组
               np.random.rand(8, 11),
               # 创建一个包含 8 个随机浮点数的一维数组
               np.random.rand(8)),
    
    # 创建一个 LinalgCase 对象，名称为 "1x5"
    LinalgCase("1x5",
               # 创建一个 1x5 的随机浮点数数组
               np.random.rand(1, 5),
               # 创建一个包含 1 个随机浮点数的一维数组
               np.random.rand(1)),
    
    # 创建一个 LinalgCase 对象，名称为 "5x1"
    LinalgCase("5x1",
               # 创建一个 5x1 的随机浮点数数组
               np.random.rand(5, 1),
               # 创建一个包含 5 个随机浮点数的一维数组
               np.random.rand(5)),
    
    # 创建一个 LinalgCase 对象，名称为 "0x4"，带有额外的标签 {'size-0'}
    LinalgCase("0x4",
               # 创建一个 0x4 的随机浮点数数组
               np.random.rand(0, 4),
               # 创建一个空的一维数组
               np.random.rand(0),
               tags={'size-0'}),
    
    # 创建一个 LinalgCase 对象，名称为 "4x0"，带有额外的标签 {'size-0'}
    LinalgCase("4x0",
               # 创建一个 4x0 的随机浮点数数组
               np.random.rand(4, 0),
               # 创建一个包含 4 个随机浮点数的一维数组
               np.random.rand(4),
               tags={'size-0'}),
# 添加测试用例到 hermitian 类别
CASES += apply_tag('hermitian', [
    LinalgCase("hsingle",
               array([[1., 2.], [2., 1.]], dtype=single),
               None),
    LinalgCase("hdouble",
               array([[1., 2.], [2., 1.]], dtype=double),
               None),
    LinalgCase("hcsingle",
               array([[1., 2 + 3j], [2 - 3j, 1]], dtype=csingle),
               None),
    LinalgCase("hcdouble",
               array([[1., 2 + 3j], [2 - 3j, 1]], dtype=cdouble),
               None),
    LinalgCase("hempty",
               np.empty((0, 0), dtype=double),
               None,
               tags={'size-0'}),
    LinalgCase("hnonarray",
               [[1, 2], [2, 1]],
               None),
    LinalgCase("matrix_b_only",
               array([[1., 2.], [2., 1.]]),
               None),
    LinalgCase("hmatrix_1x1",
               np.random.rand(1, 1),
               None),
])

# 生成通用测试案例
def _make_generalized_cases():
    new_cases = []

    for case in CASES:
        if not isinstance(case.a, np.ndarray):  # 如果 a 不是 ndarray 类型则跳过
            continue

        a = np.array([case.a, 2 * case.a, 3 * case.a])  # 构造 a 的新数组
        if case.b is None:  # 如果 b 为空，则设为 None
            b = None
        elif case.b.ndim == 1:
            b = case.b
        else:
            b = np.array([case.b, 7 * case.b, 6 * case.b])
        new_case = LinalgCase(case.name + "_tile3", a, b, tags=case.tags | {'generalized'})  # 创建新的 LinalgCase
        new_cases.append(new_case)  # 将新的测试案例添加到列表中

        a = np.array([case.a] * 2 * 3).reshape((3, 2) + case.a.shape)  # 构造 a 的新数组
        if case.b is None:  # 如果 b 为空，则设为 None
            b = None
        elif case.b.ndim == 1:
            b = np.array([case.b] * 2 * 3 * a.shape[-1]).reshape((3, 2) + case.a.shape[-2:])
        else:
            b = np.array([case.b] * 2 * 3).reshape((3, 2) + case.b.shape)
        new_case = LinalgCase(case.name + "_tile213", a, b, tags=case.tags | {'generalized'})  # 创建新的 LinalgCase
        new_cases.append(new_case)  # 将新的测试案例添加到列表中

    return new_cases  # 返回新的测试案例列表

CASES += _make_generalized_cases()  # 将通用测试案例添加到 CASES 列表中

def _stride_comb_iter(x):
    """
    生成所有轴的步幅的笛卡尔积
    """

    if not isinstance(x, np.ndarray):  # 如果 x 不是 ndarray 类型
        yield x, "nop"  # 返回 x 和 "nop"
        return
    # 初始化步幅组合
    stride_set = [(1,)] * x.ndim
    stride_set[-1] = (1, 3, -4)
    if x.ndim > 1:
        stride_set[-2] = (1, 3, -4)
    if x.ndim > 2:
        stride_set[-3] = (1, -4)
    # 使用 itertools.product 生成多个迭代器元组，每个迭代器从 stride_set 中取值
    for repeats in itertools.product(*tuple(stride_set)):
        # 根据 repeats 计算新的数组形状
        new_shape = [abs(a * b) for a, b in zip(x.shape, repeats)]
        # 根据 repeats 创建切片对象 slices
        slices = tuple([slice(None, None, repeat) for repeat in repeats])

        # 创建一个新的数组 xi，具有不同的步幅，但是与 x 具有相同的数据
        xi = np.empty(new_shape, dtype=x.dtype)
        # 将 xi 视为无符号 32 位整数数组，并填充固定的值 0xdeadbeef
        xi.view(np.uint32).fill(0xdeadbeef)
        # 根据 slices 切片选择 xi 的数据
        xi = xi[slices]
        # 将 x 的数据复制到 xi
        xi[...] = x
        # 将 xi 视为与 x 相同的类别
        xi = xi.view(x.__class__)
        # 断言 xi 与 x 全部元素相等
        assert_(np.all(xi == x))
        # 生成一个生成器，返回 xi 和其对应的重复步幅命名
        yield xi, "stride_" + "_".join(["%+d" % j for j in repeats])

        # 如果可能，生成零步幅的数组
        if x.ndim >= 1 and x.shape[-1] == 1:
            # 复制 x 的步幅，并将最后一个步幅设为 0
            s = list(x.strides)
            s[-1] = 0
            # 使用 np.lib.stride_tricks.as_strided 创建零步幅的数组 xi
            xi = np.lib.stride_tricks.as_strided(x, strides=s)
            yield xi, "stride_xxx_0"
        if x.ndim >= 2 and x.shape[-2] == 1:
            # 复制 x 的步幅，并将倒数第二个步幅设为 0
            s = list(x.strides)
            s[-2] = 0
            xi = np.lib.stride_tricks.as_strided(x, strides=s)
            yield xi, "stride_xxx_0_x"
        if x.ndim >= 2 and x.shape[:-2] == (1, 1):
            # 复制 x 的步幅，并将倒数第一和倒数第二个步幅设为 0
            s = list(x.strides)
            s[-1] = 0
            s[-2] = 0
            xi = np.lib.stride_tricks.as_strided(x, strides=s)
            yield xi, "stride_xxx_0_0"
def _make_strided_cases():
    """
    构造所有可能的步进测试用例，并添加到新的测试用例列表中
    """
    new_cases = []
    for case in CASES:
        for a, a_label in _stride_comb_iter(case.a):
            for b, b_label in _stride_comb_iter(case.b):
                # 创建带有步进标签的新测试用例对象，并加入到新测试用例列表中
                new_case = LinalgCase(case.name + "_" + a_label + "_" + b_label, a, b,
                                      tags=case.tags | {'strided'})
                new_cases.append(new_case)
    return new_cases


CASES += _make_strided_cases()
"""
将生成的步进测试用例添加到全局测试用例集合中
"""


#
# Test different routines against the above cases
#
class LinalgTestCase:
    TEST_CASES = CASES

    def check_cases(self, require=set(), exclude=set()):
        """
        对每个测试用例执行检查函数，根据require和exclude参数过滤测试用例
        """
        for case in self.TEST_CASES:
            # 根据require和exclude标签过滤测试用例
            if case.tags & require != require:
                continue
            if case.tags & exclude:
                continue

            try:
                # 执行测试用例的检查函数
                case.check(self.do)
            except Exception as e:
                # 在发生异常时生成详细的错误消息
                msg = f'In test case: {case!r}\n\n'
                msg += traceback.format_exc()
                raise AssertionError(msg) from e


class LinalgSquareTestCase(LinalgTestCase):

    def test_sq_cases(self):
        """
        运行方阵测试用例，排除广义和零尺寸的情况
        """
        self.check_cases(require={'square'},
                         exclude={'generalized', 'size-0'})

    def test_empty_sq_cases(self):
        """
        运行空方阵测试用例，排除广义情况但包括零尺寸的情况
        """
        self.check_cases(require={'square', 'size-0'},
                         exclude={'generalized'})


class LinalgNonsquareTestCase(LinalgTestCase):

    def test_nonsq_cases(self):
        """
        运行非方阵测试用例，排除广义和零尺寸的情况
        """
        self.check_cases(require={'nonsquare'},
                         exclude={'generalized', 'size-0'})

    def test_empty_nonsq_cases(self):
        """
        运行空非方阵测试用例，排除广义情况但包括零尺寸的情况
        """
        self.check_cases(require={'nonsquare', 'size-0'},
                         exclude={'generalized'})


class HermitianTestCase(LinalgTestCase):

    def test_herm_cases(self):
        """
        运行厄米特测试用例，排除广义和零尺寸的情况
        """
        self.check_cases(require={'hermitian'},
                         exclude={'generalized', 'size-0'})

    def test_empty_herm_cases(self):
        """
        运行空厄米特测试用例，排除广义情况但包括零尺寸的情况
        """
        self.check_cases(require={'hermitian', 'size-0'},
                         exclude={'generalized'})


class LinalgGeneralizedSquareTestCase(LinalgTestCase):

    @pytest.mark.slow
    def test_generalized_sq_cases(self):
        """
        运行广义方阵测试用例，包括广义和方阵，但排除零尺寸的情况
        """
        self.check_cases(require={'generalized', 'square'},
                         exclude={'size-0'})

    @pytest.mark.slow
    def test_generalized_empty_sq_cases(self):
        """
        运行空广义方阵测试用例，包括广义、方阵和零尺寸的情况
        """
        self.check_cases(require={'generalized', 'square', 'size-0'})


class LinalgGeneralizedNonsquareTestCase(LinalgTestCase):

    @pytest.mark.slow
    def test_generalized_nonsq_cases(self):
        """
        运行广义非方阵测试用例，包括广义和非方阵，但排除零尺寸的情况
        """
        self.check_cases(require={'generalized', 'nonsquare'},
                         exclude={'size-0'})

    @pytest.mark.slow
    def test_generalized_empty_nonsq_cases(self):
        """
        运行空广义非方阵测试用例，包括广义、非方阵和零尺寸的情况
        """
        self.check_cases(require={'generalized', 'nonsquare', 'size-0'})


class HermitianGeneralizedTestCase(LinalgTestCase):

    @pytest.mark.slow
    # 定义一个测试方法，用于测试通用广义厄米特矩阵的情况
    def test_generalized_herm_cases(self):
        # 调用自定义的方法 check_cases，传入需要的条件集合和排除的条件集合作为参数
        self.check_cases(require={'generalized', 'hermitian'},
                         exclude={'size-0'})
    
    # 使用 pytest 的装饰器标记，将该测试标记为慢速测试
    @pytest.mark.slow
    # 定义另一个测试方法，用于测试空的通用广义厄米特矩阵的情况
    def test_generalized_empty_herm_cases(self):
        # 调用自定义的方法 check_cases，传入需要的条件集合、包括空大小的集合和排除的条件集合作为参数
        self.check_cases(require={'generalized', 'hermitian', 'size-0'},
                         exclude={'none'})
# 定义一个函数 `identity_like_generalized`，用于根据输入数组 `a` 的维度返回一个类似单位矩阵的数组
def identity_like_generalized(a):
    # 将输入数组转换为 ndarray 类型
    a = asarray(a)
    # 如果输入数组的维度大于等于3
    if a.ndim >= 3:
        # 创建一个与输入数组形状相同的空数组，数据类型与输入数组相同
        r = np.empty(a.shape, dtype=a.dtype)
        # 使用 `identity` 函数填充数组，该函数返回形状为最后两个维度大小的单位矩阵
        r[...] = identity(a.shape[-2])
        return r
    else:
        # 如果输入数组的维度小于3，返回一个以输入数组第一个维度大小为形状的单位矩阵
        return identity(a.shape[0])


# 定义一个类 `SolveCases`，继承自 `LinalgSquareTestCase` 和 `LinalgGeneralizedSquareTestCase`，用于解决线性代数方程组的测试用例
class SolveCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):
    # `do` 方法，用于执行测试
    # 参数：a - 系数矩阵，b - 常数向量，tags - 标签
    def do(self, a, b, tags):
        # 使用 `linalg.solve` 求解线性方程组 `a x = b`，返回解向量 `x`
        x = linalg.solve(a, b)
        # 如果常数向量 `b` 的维度为1
        if np.array(b).ndim == 1:
            # 计算 `a x`，其中 `b` 视为列向量，结果是 `a x` 的第一维度
            adotx = matmul(a, x[..., None])[..., 0]
            # 断言 `a x` 扩展到 `b` 的形状后与 `adotx` 相等
            assert_almost_equal(np.broadcast_to(b, adotx.shape), adotx)
        else:
            # 计算 `a x`，直接与 `b` 比较
            adotx = matmul(a, x)
            # 断言 `a x` 与 `b` 相等
            assert_almost_equal(b, adotx)
        # 断言 `x` 和 `b` 具有相同的子类
        assert_(consistent_subclass(x, b))


# 定义一个类 `TestSolve`，继承自 `SolveCases`，用于测试 `linalg.solve` 的不同数据类型
class TestSolve(SolveCases):
    # 测试不同数据类型的参数 `dtype`
    @pytest.mark.parametrize('dtype', [single, double, csingle, cdouble])
    def test_types(self, dtype):
        # 创建一个数据类型为 `dtype` 的二维数组 `x`
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        # 断言解 `linalg.solve(x, x)` 的数据类型与输入 `dtype` 相同
        assert_equal(linalg.solve(x, x).dtype, dtype)

    # 测试输入数组 `a` 和 `b` 的一维情况
    def test_1_d(self):
        # 定义一个类 `ArraySubclass`，继承自 `np.ndarray`
        class ArraySubclass(np.ndarray):
            pass
        # 创建一个三维数组 `a`
        a = np.arange(8).reshape(2, 2, 2)
        # 创建一个一维数组 `b`，视图为 `ArraySubclass`
        b = np.arange(2).view(ArraySubclass)
        # 使用 `linalg.solve` 解方程组 `a x = b`，返回解 `result`
        result = linalg.solve(a, b)
        # 断言解 `result` 的形状为 (2, 2)
        assert result.shape == (2, 2)

        # 创建一个二维数组 `b`，视图为 `ArraySubclass`
        b = np.arange(4).reshape(2, 2).view(ArraySubclass)
        # 使用 `linalg.solve` 解方程组 `a x = b`，返回解 `result`
        result = linalg.solve(a, b)
        # 断言解 `result` 的形状为 (2, 2, 2)
        assert result.shape == (2, 2, 2)

        # 创建一个二维数组 `b`，视图为 `ArraySubclass`，形状为 (1, 2)
        b = np.arange(2).reshape(1, 2).view(ArraySubclass)
        # 断言 `linalg.solve` 抛出 `ValueError` 异常
        assert_raises(ValueError, linalg.solve, a, b)
    def test_0_size(self):
        class ArraySubclass(np.ndarray):
            pass
        # 创建一个名为 ArraySubclass 的类，继承自 np.ndarray
        a = np.arange(8).reshape(2, 2, 2)
        # 创建一个 2x2x2 的 NumPy 数组 a
        b = np.arange(6).reshape(1, 2, 3).view(ArraySubclass)
        # 创建一个 1x2x3 的 NumPy 数组 b，并将其视图类型设置为 ArraySubclass

        expected = linalg.solve(a, b)[:, 0:0, :]
        # 使用 linalg.solve 求解线性方程组 a * x = b，返回结果的第一维在全部情况下、第二维为 0、第三维在全部情况下的部分
        result = linalg.solve(a[:, 0:0, 0:0], b[:, 0:0, :])
        # 使用 linalg.solve 求解线性方程组 a[:, 0:0, 0:0] * x = b[:, 0:0, :]，即对 0x0 子数组求解线性方程组

        assert_array_equal(result, expected)
        # 断言 result 和 expected 数组相等
        assert_(isinstance(result, ArraySubclass))
        # 断言 result 是 ArraySubclass 类的实例

        # 测试非方阵和仅 b 维度为 0 时的错误
        assert_raises(linalg.LinAlgError, linalg.solve, a[:, 0:0, 0:1], b)
        # 断言在求解时出现 linalg.LinAlgError 异常，因为 a[:, 0:0, 0:1] 不是方阵
        assert_raises(ValueError, linalg.solve, a, b[:, 0:0, :])
        # 断言在求解时出现 ValueError 异常，因为 b[:, 0:0, :] 维度为 0

        # 测试广播错误
        b = np.arange(6).reshape(1, 3, 2)  # 广播错误
        assert_raises(ValueError, linalg.solve, a, b)
        # 断言在求解时出现 ValueError 异常，因为 a 和 b 无法广播
        assert_raises(ValueError, linalg.solve, a[0:0], b[0:0])
        # 断言在求解时出现 ValueError 异常，因为 a[0:0] 和 b[0:0] 维度为 0

        # 测试 0x0 矩阵的零“单方程”
        b = np.arange(2).view(ArraySubclass)
        # 创建一个长度为 2 的 ArraySubclass 类型的数组 b
        expected = linalg.solve(a, b)[:, 0:0]
        # 使用 linalg.solve 求解线性方程组 a * x = b，返回结果的第一维在全部情况下、第二维为 0 的部分
        result = linalg.solve(a[:, 0:0, 0:0], b[0:0])
        # 使用 linalg.solve 求解线性方程组 a[:, 0:0, 0:0] * x = b[0:0]，即对 0x0 子数组求解线性方程组

        assert_array_equal(result, expected)
        # 断言 result 和 expected 数组相等
        assert_(isinstance(result, ArraySubclass))
        # 断言 result 是 ArraySubclass 类的实例

        b = np.arange(3).reshape(1, 3)
        # 创建一个 1x3 的 NumPy 数组 b
        assert_raises(ValueError, linalg.solve, a, b)
        # 断言在求解时出现 ValueError 异常，因为 a 和 b 维度不匹配
        assert_raises(ValueError, linalg.solve, a[0:0], b[0:0])
        # 断言在求解时出现 ValueError 异常，因为 a[0:0] 和 b[0:0] 维度为 0
        assert_raises(ValueError, linalg.solve, a[:, 0:0, 0:0], b)
        # 断言在求解时出现 ValueError 异常，因为 a[:, 0:0, 0:0] 和 b 维度不匹配
class InvCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):
    # 定义测试用例类 InvCases，继承自 LinalgSquareTestCase 和 LinalgGeneralizedSquareTestCase

    def do(self, a, b, tags):
        # 实现测试方法 do，接受参数 a, b, tags
        a_inv = linalg.inv(a)
        # 计算矩阵 a 的逆
        assert_almost_equal(matmul(a, a_inv),
                            identity_like_generalized(a))
        # 断言验证 matmul(a, a_inv) 几乎等于 generalized 版本的单位矩阵
        assert_(consistent_subclass(a_inv, a))
        # 断言验证 a_inv 的类型与 a 一致

class TestInv(InvCases):
    # 定义测试类 TestInv，继承自 InvCases

    @pytest.mark.parametrize('dtype', [single, double, csingle, cdouble])
    def test_types(self, dtype):
        # 定义测试方法 test_types，参数 dtype 通过 pytest 参数化传入
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        # 创建 dtype 类型的数组 x
        assert_equal(linalg.inv(x).dtype, dtype)
        # 断言验证 linalg.inv(x) 的数据类型为 dtype

    def test_0_size(self):
        # 定义测试方法 test_0_size，测试处理各种大小为 0 的数组情况
        # 检查所有类型的大小为 0 的数组是否正常工作
        class ArraySubclass(np.ndarray):
            pass
        a = np.zeros((0, 1, 1), dtype=np.int_).view(ArraySubclass)
        # 创建大小为 (0, 1, 1) 的 int 类型零数组，视图为 ArraySubclass
        res = linalg.inv(a)
        # 计算 a 的逆
        assert_(res.dtype.type is np.float64)
        # 断言验证结果 res 的数据类型是 np.float64
        assert_equal(a.shape, res.shape)
        # 断言验证 a 和 res 的形状相同
        assert_(isinstance(res, ArraySubclass))
        # 断言验证 res 是 ArraySubclass 的实例

        a = np.zeros((0, 0), dtype=np.complex64).view(ArraySubclass)
        # 创建大小为 (0, 0) 的 complex64 类型零数组，视图为 ArraySubclass
        res = linalg.inv(a)
        # 计算 a 的逆
        assert_(res.dtype.type is np.complex64)
        # 断言验证结果 res 的数据类型是 np.complex64
        assert_equal((0,), res.shape)
        # 断言验证 res 的形状为 (0,)
        assert_(isinstance(res, ArraySubclass))
        # 断言验证 res 是 ArraySubclass 的实例


class EigvalsCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):
    # 定义测试用例类 EigvalsCases，继承自 LinalgSquareTestCase 和 LinalgGeneralizedSquareTestCase

    def do(self, a, b, tags):
        # 实现测试方法 do，接受参数 a, b, tags
        ev = linalg.eigvals(a)
        # 计算矩阵 a 的特征值
        evalues, evectors = linalg.eig(a)
        # 计算矩阵 a 的特征值和特征向量
        assert_almost_equal(ev, evalues)
        # 断言验证计算得到的特征值 ev 与 linalg.eig 返回的特征值 evalues 几乎相等


class TestEigvals(EigvalsCases):
    # 定义测试类 TestEigvals，继承自 EigvalsCases

    @pytest.mark.parametrize('dtype', [single, double, csingle, cdouble])
    def test_types(self, dtype):
        # 定义测试方法 test_types，参数 dtype 通过 pytest 参数化传入
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        # 创建 dtype 类型的数组 x
        assert_equal(linalg.eigvals(x).dtype, dtype)
        # 断言验证 linalg.eigvals(x) 的数据类型为 dtype
        x = np.array([[1, 0.5], [-1, 1]], dtype=dtype)
        # 创建 dtype 类型的数组 x
        assert_equal(linalg.eigvals(x).dtype, get_complex_dtype(dtype))
        # 断言验证 linalg.eigvals(x) 的数据类型为复数 dtype 的实部类型

    def test_0_size(self):
        # 定义测试方法 test_0_size，测试处理各种大小为 0 的数组情况
        # 检查所有类型的大小为 0 的数组是否正常工作
        class ArraySubclass(np.ndarray):
            pass
        a = np.zeros((0, 1, 1), dtype=np.int_).view(ArraySubclass)
        # 创建大小为 (0, 1, 1) 的 int 类型零数组，视图为 ArraySubclass
        res = linalg.eigvals(a)
        # 计算 a 的特征值
        assert_(res.dtype.type is np.float64)
        # 断言验证结果 res 的数据类型是 np.float64
        assert_equal((0, 1), res.shape)
        # 断言验证 res 的形状为 (0, 1)
        # This is just for documentation, it might make sense to change:
        assert_(isinstance(res, np.ndarray))
        # 断言验证 res 是 np.ndarray 的实例

        a = np.zeros((0, 0), dtype=np.complex64).view(ArraySubclass)
        # 创建大小为 (0, 0) 的 complex64 类型零数组，视图为 ArraySubclass
        res = linalg.eigvals(a)
        # 计算 a 的特征值
        assert_(res.dtype.type is np.complex64)
        # 断言验证结果 res 的数据类型是 np.complex64
        assert_equal((0,), res.shape)
        # 断言验证 res 的形状为 (0,)
        # This is just for documentation, it might make sense to change:
        assert_(isinstance(res, np.ndarray))
        # 断言验证 res 是 np.ndarray 的实例


class EigCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):
    # 定义测试用例类 EigCases，继承自 LinalgSquareTestCase 和 LinalgGeneralizedSquareTestCase

    def do(self, a, b, tags):
        # 实现测试方法 do，接受参数 a, b, tags
        res = linalg.eig(a)
        # 计算矩阵 a 的特征值和特征向量
        eigenvalues, eigenvectors = res.eigenvalues, res.eigenvectors
        # 获取计算得到的特征值和特征向量
        assert_allclose(matmul(a, eigenvectors),
                        np.asarray(eigenvectors) * np.asarray(eigenvalues)[..., None, :],
                        rtol=get_rtol(eigenvalues.dtype))
        # 断言验证 matmul(a, eigenvectors) 几乎等于 eigenvectors * eigenvalues，考虑浮点数的相对误差
        assert_(consistent_subclass(eigenvectors, a))
        # 断言验证 eigenvectors 的类型与 a 一致


class TestEig(EigCases):
    # 定义测试类 TestEig，继承自 EigCases
    # 使用 pytest 的参数化装饰器，以便为每种数据类型运行此测试函数
    @pytest.mark.parametrize('dtype', [single, double, csingle, cdouble])
    def test_types(self, dtype):
        # 创建一个二维数组 x，指定数据类型为 dtype
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        # 计算 x 的特征值和特征向量
        w, v = np.linalg.eig(x)
        # 断言特征值的数据类型与指定的 dtype 相同
        assert_equal(w.dtype, dtype)
        # 断言特征向量的数据类型与指定的 dtype 相同
        assert_equal(v.dtype, dtype)

        # 创建另一个二维数组 x，指定数据类型为 dtype
        x = np.array([[1, 0.5], [-1, 1]], dtype=dtype)
        # 计算 x 的特征值和特征向量
        w, v = np.linalg.eig(x)
        # 断言特征值的数据类型为由 get_complex_dtype 函数返回的复数数据类型
        assert_equal(w.dtype, get_complex_dtype(dtype))
        # 断言特征向量的数据类型为由 get_complex_dtype 函数返回的复数数据类型
        assert_equal(v.dtype, get_complex_dtype(dtype))

    def test_0_size(self):
        # 检查各种零大小的数组是否正常工作
        # 定义一个继承自 np.ndarray 的数组子类
        class ArraySubclass(np.ndarray):
            pass
        # 创建一个形状为 (0, 1, 1)，数据类型为 np.int_ 的全零数组，并将其视图转换为 ArraySubclass 类型
        a = np.zeros((0, 1, 1), dtype=np.int_).view(ArraySubclass)
        # 计算 a 的特征值和特征向量
        res, res_v = linalg.eig(a)
        # 断言特征向量的数据类型为 np.float64
        assert_(res_v.dtype.type is np.float64)
        # 断言特征值的数据类型为 np.float64
        assert_(res.dtype.type is np.float64)
        # 断言 a 的形状与 res_v 的形状相等
        assert_equal(a.shape, res_v.shape)
        # 断言 res 的形状为 (0, 1)
        assert_equal((0, 1), res.shape)
        # 这仅用于文档，可能需要更改：
        # 断言 a 是 np.ndarray 类型的实例
        assert_(isinstance(a, np.ndarray))

        # 创建一个形状为 (0, 0)，数据类型为 np.complex64 的全零数组，并将其视图转换为 ArraySubclass 类型
        a = np.zeros((0, 0), dtype=np.complex64).view(ArraySubclass)
        # 计算 a 的特征值和特征向量
        res, res_v = linalg.eig(a)
        # 断言特征向量的数据类型为 np.complex64
        assert_(res_v.dtype.type is np.complex64)
        # 断言特征值的数据类型为 np.complex64
        assert_(res.dtype.type is np.complex64)
        # 断言 a 的形状与 res_v 的形状相等
        assert_equal(a.shape, res_v.shape)
        # 断言 res 的形状为 (0,)
        assert_equal((0,), res.shape)
        # 这仅用于文档，可能需要更改：
        # 断言 a 是 np.ndarray 类型的实例
        assert_(isinstance(a, np.ndarray))
class SVDBaseTests:
    hermitian = False

    @pytest.mark.parametrize('dtype', [single, double, csingle, cdouble])
    # 使用 pytest 的参数化装饰器，针对不同的 dtype 参数化测试类型
    def test_types(self, dtype):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        # 创建一个 numpy 数组 x，指定数据类型为 dtype
        res = linalg.svd(x)
        # 对 x 进行奇异值分解，并将结果存储在 res 中
        U, S, Vh = res.U, res.S, res.Vh
        # 从 res 中获取奇异值分解的结果 U、S、Vh
        assert_equal(U.dtype, dtype)
        # 断言 U 的数据类型与指定的 dtype 相同
        assert_equal(S.dtype, get_real_dtype(dtype))
        # 断言 S 的数据类型为 dtype 对应的实数类型
        assert_equal(Vh.dtype, dtype)
        # 断言 Vh 的数据类型与指定的 dtype 相同
        s = linalg.svd(x, compute_uv=False, hermitian=self.hermitian)
        # 对 x 进行奇异值分解，但不计算 U 和 Vh，同时考虑是否为共轭（当 hermitian=True 时）
        assert_equal(s.dtype, get_real_dtype(dtype))
        # 断言 s 的数据类型为 dtype 对应的实数类型

class SVDCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):

    def do(self, a, b, tags):
        u, s, vt = linalg.svd(a, False)
        # 对输入的矩阵 a 进行奇异值分解，不计算 U 和 V^T
        assert_allclose(a, matmul(np.asarray(u) * np.asarray(s)[..., None, :],
                                           np.asarray(vt)),
                        rtol=get_rtol(u.dtype))
        # 断言 a 与通过 U、S、V^T 重构的矩阵相似
        assert_(consistent_subclass(u, a))
        # 断言 u 是 a 的一致子类
        assert_(consistent_subclass(vt, a))
        # 断言 vt 是 a 的一致子类

class TestSVD(SVDCases, SVDBaseTests):

    def test_empty_identity(self):
        """ Empty input should put an identity matrix in u or vh """
        x = np.empty((4, 0))
        # 创建一个空的 numpy 数组 x，形状为 (4, 0)
        u, s, vh = linalg.svd(x, compute_uv=True, hermitian=self.hermitian)
        # 对 x 进行奇异值分解，计算 U 和 V^H，同时考虑是否为共轭（当 hermitian=True 时）
        assert_equal(u.shape, (4, 4))
        # 断言 U 的形状为 (4, 4)
        assert_equal(vh.shape, (0, 0))
        # 断言 V^H 的形状为 (0, 0)
        assert_equal(u, np.eye(4))
        # 断言 U 与 4x4 单位矩阵相等

        x = np.empty((0, 4))
        # 创建一个空的 numpy 数组 x，形状为 (0, 4)
        u, s, vh = linalg.svd(x, compute_uv=True, hermitian=self.hermitian)
        # 对 x 进行奇异值分解，计算 U 和 V^H，同时考虑是否为共轭（当 hermitian=True 时）
        assert_equal(u.shape, (0, 0))
        # 断言 U 的形状为 (0, 0)
        assert_equal(vh.shape, (4, 4))
        # 断言 V^H 的形状为 (4, 4)
        assert_equal(vh, np.eye(4))
        # 断言 V^H 与 4x4 单位矩阵相等

    def test_svdvals(self):
        x = np.array([[1, 0.5], [0.5, 1]])
        # 创建一个 numpy 数组 x
        s_from_svd = linalg.svd(x, compute_uv=False, hermitian=self.hermitian)
        # 对 x 进行奇异值分解，但不计算 U 和 V^H，同时考虑是否为共轭（当 hermitian=True 时）
        s_from_svdvals = linalg.svdvals(x)
        # 计算 x 的奇异值
        assert_almost_equal(s_from_svd, s_from_svdvals)
        # 断言通过奇异值分解和 svdvals 函数得到的奇异值近似相等

class SVDHermitianCases(HermitianTestCase, HermitianGeneralizedTestCase):

    def do(self, a, b, tags):
        u, s, vt = linalg.svd(a, False, hermitian=True)
        # 对输入的矩阵 a 进行奇异值分解，不计算 U 和 V^H，并且假设为共轭
        assert_allclose(a, matmul(np.asarray(u) * np.asarray(s)[..., None, :],
                                           np.asarray(vt)),
                        rtol=get_rtol(u.dtype))
        # 断言 a 与通过 U、S、V^H 重构的矩阵相似
        def hermitian(mat):
            axes = list(range(mat.ndim))
            axes[-1], axes[-2] = axes[-2], axes[-1]
            return np.conj(np.transpose(mat, axes=axes))

        assert_almost_equal(np.matmul(u, hermitian(u)), np.broadcast_to(np.eye(u.shape[-1]), u.shape))
        # 断言 U 与其共轭转置的乘积近似为单位矩阵
        assert_almost_equal(np.matmul(vt, hermitian(vt)), np.broadcast_to(np.eye(vt.shape[-1]), vt.shape))
        # 断言 V^H 与其共轭转置的乘积近似为单位矩阵
        assert_equal(np.sort(s)[..., ::-1], s)
        # 断言 s 为非增序列
        assert_(consistent_subclass(u, a))
        # 断言 u 是 a 的一致子类
        assert_(consistent_subclass(vt, a))
        # 断言 vt 是 a 的一致子类

class TestSVDHermitian(SVDHermitianCases, SVDBaseTests):
    hermitian = True

class CondCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):
    # cond(x, p) for p in (None, 2, -2)
    # 定义一个方法 `do`，接受三个参数 `a`, `b`, `tags`
    def do(self, a, b, tags):
        # 将参数 `a` 转换为 NumPy 数组 `c`，假设 `a` 可能是一个矩阵
        c = asarray(a)  # a might be a matrix
        
        # 如果标签中包含 'size-0'
        if 'size-0' in tags:
            # 断言会抛出 LinAlgError 异常，测试矩阵 `c` 的条件数
            assert_raises(LinAlgError, linalg.cond, c)
            return

        # 计算矩阵 `c` 的奇异值，不计算左右奇异向量
        s = linalg.svd(c, compute_uv=False)
        
        # 断言矩阵 `a` 的条件数近似等于最大和最小奇异值比值
        assert_almost_equal(
            linalg.cond(a), s[..., 0] / s[..., -1],
            single_decimal=5, double_decimal=11)
        
        # 断言矩阵 `a` 的 2-范数条件数近似等于最大和最小奇异值比值
        assert_almost_equal(
            linalg.cond(a, 2), s[..., 0] / s[..., -1],
            single_decimal=5, double_decimal=11)
        
        # 断言矩阵 `a` 的 -2-范数条件数近似等于最小和最大奇异值比值的倒数
        assert_almost_equal(
            linalg.cond(a, -2), s[..., -1] / s[..., 0],
            single_decimal=5, double_decimal=11)

        # 计算矩阵 `c` 的逆矩阵
        cinv = np.linalg.inv(c)
        
        # 断言矩阵 `a` 的 1-范数条件数近似等于绝对值矩阵 `c` 按行求和后的最大值乘以逆矩阵的相同计算结果
        assert_almost_equal(
            linalg.cond(a, 1),
            abs(c).sum(-2).max(-1) * abs(cinv).sum(-2).max(-1),
            single_decimal=5, double_decimal=11)
        
        # 断言矩阵 `a` 的 -1-范数条件数近似等于绝对值矩阵 `c` 按行求和后的最小值乘以逆矩阵的相同计算结果
        assert_almost_equal(
            linalg.cond(a, -1),
            abs(c).sum(-2).min(-1) * abs(cinv).sum(-2).min(-1),
            single_decimal=5, double_decimal=11)
        
        # 断言矩阵 `a` 的无穷范数条件数近似等于绝对值矩阵 `c` 按列求和后的最大值乘以逆矩阵的相同计算结果
        assert_almost_equal(
            linalg.cond(a, np.inf),
            abs(c).sum(-1).max(-1) * abs(cinv).sum(-1).max(-1),
            single_decimal=5, double_decimal=11)
        
        # 断言矩阵 `a` 的负无穷范数条件数近似等于绝对值矩阵 `c` 按列求和后的最小值乘以逆矩阵的相同计算结果
        assert_almost_equal(
            linalg.cond(a, -np.inf),
            abs(c).sum(-1).min(-1) * abs(cinv).sum(-1).min(-1),
            single_decimal=5, double_decimal=11)
        
        # 断言矩阵 `a` 的 Frobenius 范数条件数近似等于矩阵 `c` 的 Frobenius 范数平方和乘以逆矩阵的相同计算结果的平方根
        assert_almost_equal(
            linalg.cond(a, 'fro'),
            np.sqrt((abs(c)**2).sum(-1).sum(-1)
                    * (abs(cinv)**2).sum(-1).sum(-1)),
            single_decimal=5, double_decimal=11)
# 定义一个名为 TestCond 的类，继承自 CondCases
class TestCond(CondCases):
    # 定义一个名为 test_basic_nonsvd 的方法
    def test_basic_nonsvd(self):
        # 对非奇异值分解进行烟雾测试
        A = array([[1., 0, 1], [0, -2., 0], [0, 0, 3.]])
        # 使用无穷范数计算矩阵 A 的条件数，并与 4 进行差值比较
        assert_almost_equal(linalg.cond(A, inf), 4)
        # 使用负无穷范数计算矩阵 A 的条件数，并与 2/3 进行差值比较
        assert_almost_equal(linalg.cond(A, -inf), 2/3)
        # 使用 1 范数计算矩阵 A 的条件数，并与 4 进行差值比较
        assert_almost_equal(linalg.cond(A, 1), 4)
        # 使用 -1 范数计算矩阵 A 的条件数，并与 0.5 进行差值比较
        assert_almost_equal(linalg.cond(A, -1), 0.5)
        # 使用 Frobenius 范数计算矩阵 A 的条件数，并与 sqrt(265 / 12) 进行差值比较
        assert_almost_equal(linalg.cond(A, 'fro'), np.sqrt(265 / 12))

    # 定义一个名为 test_singular 的方法
    def test_singular(self):
        # 奇异矩阵对于正范数有无限条件数，而负范数不应触发异常
        As = [np.zeros((2, 2)), np.ones((2, 2))]
        p_pos = [None, 1, 2, 'fro']
        p_neg = [-1, -2]
        for A, p in itertools.product(As, p_pos):
            # 反转可能不会达到确切的无穷大，因此只需检查数值非常大
            assert_(linalg.cond(A, p) > 1e15)
        for A, p in itertools.product(As, p_neg):
            linalg.cond(A, p)

    # 定义一个名为 test_nan 的方法
    @pytest.mark.xfail(True, run=False,
                       reason="Platform/LAPACK-dependent failure, "
                              "see gh-18914")
    def test_nan(self):
        # nan 应该被传递，而不是转换为无穷大
        ps = [None, 1, -1, 2, -2, 'fro']
        p_pos = [None, 1, 2, 'fro']
        A = np.ones((2, 2))
        A[0,1] = np.nan
        for p in ps:
            c = linalg.cond(A, p)
            assert_(isinstance(c, np.float64))
            assert_(np.isnan(c))
        A = np.ones((3, 2, 2))
        A[1,0,1] = np.nan
        for p in ps:
            c = linalg.cond(A, p)
            assert_(np.isnan(c[1]))
            if p in p_pos:
                assert_(c[0] > 1e15)
                assert_(c[2] > 1e15)
            else:
                assert_(not np.isnan(c[0]))
                assert_(not np.isnan(c[2]))

    # 定义一个名为 test_stacked_singular 的方法
    def test_stacked_singular(self):
        # 当只有部分堆叠矩阵是奇异时，检查行为
        np.random.seed(1234)
        A = np.random.rand(2, 2, 2, 2)
        A[0,0] = 0
        A[1,1] = 0
        for p in (None, 1, 2, 'fro', -1, -2):
            c = linalg.cond(A, p)
            assert_equal(c[0,0], np.inf)
            assert_equal(c[1,1], np.inf)
            assert_(np.isfinite(c[0,1]))
            assert_(np.isfinite(c[1,0]))

# 定义一个名为 PinvCases 的类，继承自多个测试用例类
class PinvCases(LinalgSquareTestCase,
                LinalgNonsquareTestCase,
                LinalgGeneralizedSquareTestCase,
                LinalgGeneralizedNonsquareTestCase):
    # 定义一个 do 方法，接受 a, b, tags 作为参数
    def do(self, a, b, tags):
        # 计算矩阵 a 的伪逆
        a_ginv = linalg.pinv(a)
        # `a @ a_ginv == I` 如果 a 是奇异的话可能不成立
        dot = matmul
        # 断言矩阵乘积 a @ a_ginv 与 a 之间的关系
        assert_almost_equal(dot(dot(a, a_ginv), a), a, single_decimal=5, double_decimal=11)
        # 断言 a_ginv 与 a 具有一致的子类
        assert_(consistent_subclass(a_ginv, a))

# 定义一个名为 TestPinv 的类，继承自 PinvCases
class TestPinv(PinvCases):
    pass

# 定义一个名为 PinvHermitianCases 的类，继承自 HermitianTestCase 和 HermitianGeneralizedTestCase
class PinvHermitianCases(HermitianTestCase, HermitianGeneralizedTestCase):
    def do(self, a, b, tags):
        # 计算矩阵 a 的广义逆
        a_ginv = linalg.pinv(a, hermitian=True)
        # 检查 `a @ a_ginv == I` 是否成立，如果矩阵 a 是奇异的，则不成立
        dot = matmul
        # 断言验证 dot(dot(a, a_ginv), a) 接近于 a，精确到小数点后五位和十一位
        assert_almost_equal(dot(dot(a, a_ginv), a), a, single_decimal=5, double_decimal=11)
        # 断言验证 a_ginv 是否是 a 的一致子类
        assert_(consistent_subclass(a_ginv, a))
class TestPinvHermitian(PinvHermitianCases):
    pass



def test_pinv_rtol_arg():
    # 创建一个测试用的二维数组 `a`
    a = np.array([[1, 2, 3], [4, 1, 1], [2, 3, 1]])

    # 使用 `rcond` 参数计算广义逆矩阵，并使用 `rtol` 参数计算广义逆矩阵，比较它们的近似值
    assert_almost_equal(
        np.linalg.pinv(a, rcond=0.5),
        np.linalg.pinv(a, rtol=0.5),
    )

    # 当 `rtol` 和 `rcond` 同时设置时，使用 `pytest.raises` 检查是否会引发 ValueError 异常
    with pytest.raises(
        ValueError, match=r"`rtol` and `rcond` can't be both set."
    ):
        np.linalg.pinv(a, rcond=0.5, rtol=0.5)



class DetCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):

    def do(self, a, b, tags):
        # 计算矩阵的行列式值 `d`
        d = linalg.det(a)
        # 计算矩阵的行列式的符号和对数绝对值
        res = linalg.slogdet(a)
        s, ld = res.sign, res.logabsdet
        # 根据矩阵的数据类型，转换为适当的复数类型
        if asarray(a).dtype.type in (single, double):
            ad = asarray(a).astype(double)
        else:
            ad = asarray(a).astype(cdouble)
        # 计算矩阵的特征值
        ev = linalg.eigvals(ad)
        # 检查行列式的计算结果与特征值乘积的一致性
        assert_almost_equal(d, multiply.reduce(ev, axis=-1))
        # 检查行列式符号乘以指数对数绝对值是否与特征值乘积的一致性
        assert_almost_equal(s * np.exp(ld), multiply.reduce(ev, axis=-1))

        # 将符号 `s` 和对数绝对值 `ld` 至少转换为一维数组
        s = np.atleast_1d(s)
        ld = np.atleast_1d(ld)
        # 根据符号 `s` 的非零判断条件，检查其绝对值是否为 1
        m = (s != 0)
        assert_almost_equal(np.abs(s[m]), 1)
        # 对于非零符号 `s`，检查对数绝对值 `ld` 是否为负无穷
        assert_equal(ld[~m], -inf)



class TestDet(DetCases):
    def test_zero(self):
        # 对于零矩阵的行列式和特殊情况进行检查
        assert_equal(linalg.det([[0.0]]), 0.0)
        assert_equal(type(linalg.det([[0.0]])), double)
        assert_equal(linalg.det([[0.0j]]), 0.0)
        assert_equal(type(linalg.det([[0.0j]])), cdouble)

        assert_equal(linalg.slogdet([[0.0]]), (0.0, -inf))
        assert_equal(type(linalg.slogdet([[0.0]])[0]), double)
        assert_equal(type(linalg.slogdet([[0.0]])[1]), double)
        assert_equal(linalg.slogdet([[0.0j]]), (0.0j, -inf))
        assert_equal(type(linalg.slogdet([[0.0j]])[0]), cdouble)
        assert_equal(type(linalg.slogdet([[0.0j]])[1]), double)

    @pytest.mark.parametrize('dtype', [single, double, csingle, cdouble])
    def test_types(self, dtype):
        # 对于不同数据类型的矩阵进行行列式和特征值计算的数据类型检查
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        assert_equal(np.linalg.det(x).dtype, dtype)
        ph, s = np.linalg.slogdet(x)
        assert_equal(s.dtype, get_real_dtype(dtype))
        assert_equal(ph.dtype, dtype)

    def test_0_size(self):
        # 对于大小为 0 的矩阵，行列式和特征值计算的特殊情况检查
        a = np.zeros((0, 0), dtype=np.complex64)
        res = linalg.det(a)
        assert_equal(res, 1.)
        assert_(res.dtype.type is np.complex64)
        res = linalg.slogdet(a)
        assert_equal(res, (1, 0))
        assert_(res[0].dtype.type is np.complex64)
        assert_(res[1].dtype.type is np.float32)

        a = np.zeros((0, 0), dtype=np.float64)
        res = linalg.det(a)
        assert_equal(res, 1.)
        assert_(res.dtype.type is np.float64)
        res = linalg.slogdet(a)
        assert_equal(res, (1, 0))
        assert_(res[0].dtype.type is np.float64)
        assert_(res[1].dtype.type is np.float64)
    # 定义一个方法，接受三个参数：a，b，tags
    def do(self, a, b, tags):
        # 将a转换为NumPy数组
        arr = np.asarray(a)
        # 获取数组的形状 m, n
        m, n = arr.shape
        # 对a进行奇异值分解，返回结果是u, s, vt
        u, s, vt = linalg.svd(a, False)
        # 使用最小二乘法求解线性方程组a*x=b，返回结果是x, residuals, rank, sv
        x, residuals, rank, sv = linalg.lstsq(a, b, rcond=-1)
        # 如果m等于0
        if m == 0:
            # 断言x的所有元素都为0
            assert_((x == 0).all())
        # 如果m小于等于n
        if m <= n:
            # 断言b与a*x的点积近似相等
            assert_almost_equal(b, dot(a, x))
            # 断言rank等于m
            assert_equal(rank, m)
        else:
            # 断言rank等于n
            assert_equal(rank, n)
        # 断言sv近似等于s
        assert_almost_equal(sv, sv.__array_wrap__(s))
        # 如果rank等于n且m大于n
        if rank == n and m > n:
            # 预期残差的平方和
            expect_resids = (
                np.asarray(abs(np.dot(a, x) - b)) ** 2).sum(axis=0)
            # 将expect_resids转换为NumPy数组
            expect_resids = np.asarray(expect_resids)
            # 如果b是一维数组
            if np.asarray(b).ndim == 1:
                # 调整expect_resids的形状
                expect_resids.shape = (1,)
                # 断言residuals与expect_resids的形状相同
                assert_equal(residuals.shape, expect_resids.shape)
        else:
            # 创建一个空的NumPy数组，其数据类型与x相同
            expect_resids = np.array([]).view(type(x))
        # 断言residuals近似等于expect_resids
        assert_almost_equal(residuals, expect_resids)
        # 断言residuals的数据类型为浮点数
        assert_(np.issubdtype(residuals.dtype, np.floating))
        # 断言x和b具有一致的子类类型
        assert_(consistent_subclass(x, b))
        # 断言residuals和b具有一致的子类类型
        assert_(consistent_subclass(residuals, b))
class TestLstsq(LstsqCases):
    # 继承自 LstsqCases 类的测试类 TestLstsq

    def test_rcond(self):
        # 测试函数，用于测试 linalg.lstsq 函数的 rcond 参数
        a = np.array([[0., 1.,  0.,  1.,  2.,  0.],
                      [0., 2.,  0.,  0.,  1.,  0.],
                      [1., 0.,  1.,  0.,  0.,  4.],
                      [0., 0.,  0.,  2.,  3.,  0.]]).T
        # 创建一个 numpy 数组 a，表示系数矩阵的转置

        b = np.array([1, 0, 0, 0, 0, 0])
        # 创建一个 numpy 数组 b，表示常数向量

        x, residuals, rank, s = linalg.lstsq(a, b, rcond=-1)
        # 使用 linalg.lstsq 函数求解线性方程组，rcond 参数为 -1
        assert_(rank == 4)
        # 断言 rank 等于 4
        x, residuals, rank, s = linalg.lstsq(a, b)
        # 再次使用 linalg.lstsq 函数求解线性方程组，默认 rcond 参数
        assert_(rank == 3)
        # 断言 rank 等于 3
        x, residuals, rank, s = linalg.lstsq(a, b, rcond=None)
        # 第三次使用 linalg.lstsq 函数求解线性方程组，rcond 参数为 None
        assert_(rank == 3)
        # 断言 rank 等于 3

    @pytest.mark.parametrize(["m", "n", "n_rhs"], [
        (4, 2, 2),
        (0, 4, 1),
        (0, 4, 2),
        (4, 0, 1),
        (4, 0, 2),
        (4, 2, 0),
        (0, 0, 0)
    ])
    def test_empty_a_b(self, m, n, n_rhs):
        # 参数化测试函数，用于测试当 a 或 b 为空时的情况
        a = np.arange(m * n).reshape(m, n)
        # 创建一个 m 行 n 列的 numpy 数组 a
        b = np.ones((m, n_rhs))
        # 创建一个元素全为 1 的 m 行 n_rhs 列的 numpy 数组 b
        x, residuals, rank, s = linalg.lstsq(a, b, rcond=None)
        # 使用 linalg.lstsq 函数求解线性方程组，rcond 参数为 None
        if m == 0:
            assert_((x == 0).all())
        # 如果 m 为 0，则断言 x 中所有元素为 0
        assert_equal(x.shape, (n, n_rhs))
        # 断言 x 的形状为 (n, n_rhs)
        assert_equal(residuals.shape, ((n_rhs,) if m > n else (0,)))
        # 断言 residuals 的形状符合预期 ((n_rhs,) 或 (0,))
        if m > n and n_rhs > 0:
            # 如果 m 大于 n 并且 n_rhs 大于 0
            # residuals 正好是 b 列的平方范数
            r = b - np.dot(a, x)
            assert_almost_equal(residuals, (r * r).sum(axis=-2))
            # 断言 residuals 与 (r * r).sum(axis=-2) 几乎相等
        assert_equal(rank, min(m, n))
        # 断言 rank 等于 m 和 n 中的较小值
        assert_equal(s.shape, (min(m, n),))
        # 断言 s 的形状为 (min(m, n),)

    def test_incompatible_dims(self):
        # 测试函数，用于测试当维度不兼容时的情况
        x = np.array([0, 1, 2, 3])
        # 创建一个 numpy 数组 x
        y = np.array([-1, 0.2, 0.9, 2.1, 3.3])
        # 创建一个 numpy 数组 y
        A = np.vstack([x, np.ones(len(x))]).T
        # 创建一个由 x 和一个全为 1 的数组堆叠而成的 numpy 数组 A
        with assert_raises_regex(LinAlgError, "Incompatible dimensions"):
            linalg.lstsq(A, y, rcond=None)
            # 使用 linalg.lstsq 函数求解线性方程组，rcond 参数为 None


@pytest.mark.parametrize('dt', [np.dtype(c) for c in '?bBhHiIqQefdgFDGO'])
class TestMatrixPower:
    # 参数化测试类，用于测试 matrix_power 函数的不同数据类型

    rshft_0 = np.eye(4)
    # 创建一个 4x4 的单位矩阵赋值给 rshft_0
    rshft_1 = rshft_0[[3, 0, 1, 2]]
    # 创建 rshft_0 的第 4 行换到第 1 行的矩阵赋值给 rshft_1
    rshft_2 = rshft_0[[2, 3, 0, 1]]
    # 创建 rshft_0 的第 3 行换到第 0 行的矩阵赋值给 rshft_2
    rshft_3 = rshft_0[[1, 2, 3, 0]]
    # 创建 rshft_0 的第 2 行换到第 3 行的矩阵赋值给 rshft_3
    rshft_all = [rshft_0, rshft_1, rshft_2, rshft_3]
    # 创建包含 rshft_0、rshft_1、rshft_2 和 rshft_3 的列表赋值给 rshft_all
    noninv = array([[1, 0], [0, 0]])
    # 创建一个非逆矩阵赋值给 noninv
    stacked = np.block([[[rshft_0]]]*2)
    # 创建一个堆叠 rshft_0 两次的 numpy 数组赋值给 stacked
    # FIXME the 'e' dtype might work in future
    dtnoinv = [object, np.dtype('e'), np.dtype('g'), np.dtype('G')]
    # 创建包含 object、'e'、'g' 和 'G' 数据类型的列表赋值给 dtnoinv

    def test_large_power(self, dt):
        # 测试函数，用于测试 matrix_power 函数对于大数幂次的计算
        rshft = self.rshft_1.astype(dt)
        # 将 rshft_1 转换为指定数据类型 dt
        assert_equal(
            matrix_power(rshft, 2**100 + 2**10 + 2**5 + 0), self.rshft_0)
        # 断言 matrix_power 计算结果与 rshft_0 在指定幂次下的相等性
        assert_equal(
            matrix_power(rshft, 2**100 + 2**10 + 2**5 + 1), self.rshft_1)
        # 断言 matrix_power 计算结果与 rshft_1 在指定幂次下的相等性
        assert_equal(
            matrix_power(rshft, 2**100 + 2**10 + 2**5 + 2), self.rshft_2)
        # 断言 matrix_power 计算结果与 rshft_2 在指定幂次下的相等性
        assert_equal(
            matrix_power(rshft, 2**100 + 2**10 + 2**5 + 3), self.rshft_3)
        # 断言 matrix_power 计算结果与 rshft_3 在指定幂次下的相等性

    def test_power_is_zero(self, dt):
        # 测试函数，用于测试 matrix_power 函数对于幂次为 0 的情况
        def tz(M):
            mz = matrix_power(M, 0)
            # 计算矩阵 M 的零次幂
            assert_equal(mz, identity_like_generalized(M))
            # 断言 mz 等于 M 的广义单位矩阵
            assert_equal(mz.dtype, M.dtype)
            # 断言 mz
    # 定义一个测试方法，用于测试矩阵乘幂为1的情况
    def test_power_is_one(self, dt):
        # 定义内部函数 tz，接受一个矩阵 mat 作为参数
        def tz(mat):
            # 计算矩阵 mat 的一次幂
            mz = matrix_power(mat, 1)
            # 断言计算结果 mz 等于原始矩阵 mat
            assert_equal(mz, mat)
            # 断言计算结果 mz 的数据类型等于原始矩阵 mat 的数据类型
            assert_equal(mz.dtype, mat.dtype)

        # 遍历 self.rshft_all 中的每一个矩阵
        for mat in self.rshft_all:
            # 调用 tz 函数，传入 mat 转换为指定数据类型 dt 后的结果
            tz(mat.astype(dt))
            # 如果数据类型 dt 不是对象类型
            if dt != object:
                # 调用 tz 函数，传入 self.stacked 转换为指定数据类型 dt 后的结果
                tz(self.stacked.astype(dt))

    # 定义一个测试方法，用于测试矩阵乘幂为2的情况
    def test_power_is_two(self, dt):
        # 定义内部函数 tz，接受一个矩阵 mat 作为参数
        def tz(mat):
            # 计算矩阵 mat 的二次幂
            mz = matrix_power(mat, 2)
            # 根据矩阵 mat 的数据类型选择使用 matmul 或 dot 函数进行矩阵乘法
            mmul = matmul if mat.dtype != object else dot
            # 断言计算结果 mz 等于 mat 与自身的乘积
            assert_equal(mz, mmul(mat, mat))
            # 断言计算结果 mz 的数据类型等于原始矩阵 mat 的数据类型
            assert_equal(mz.dtype, mat.dtype)

        # 遍历 self.rshft_all 中的每一个矩阵
        for mat in self.rshft_all:
            # 调用 tz 函数，传入 mat 转换为指定数据类型 dt 后的结果
            tz(mat.astype(dt))
            # 如果数据类型 dt 不是对象类型
            if dt != object:
                # 调用 tz 函数，传入 self.stacked 转换为指定数据类型 dt 后的结果
                tz(self.stacked.astype(dt))

    # 定义一个测试方法，用于测试矩阵乘幂为-1的情况
    def test_power_is_minus_one(self, dt):
        # 定义内部函数 tz，接受一个矩阵 mat 作为参数
        def tz(mat):
            # 计算矩阵 mat 的逆矩阵
            invmat = matrix_power(mat, -1)
            # 根据矩阵 mat 的数据类型选择使用 matmul 或 dot 函数进行矩阵乘法
            mmul = matmul if mat.dtype != object else dot
            # 断言计算结果 invmat 与 mat 的乘积接近单位矩阵
            assert_almost_equal(
                mmul(invmat, mat), identity_like_generalized(mat))

        # 遍历 self.rshft_all 中的每一个矩阵
        for mat in self.rshft_all:
            # 如果数据类型 dt 不在 self.dtnoinv 中
            if dt not in self.dtnoinv:
                # 调用 tz 函数，传入 mat 转换为指定数据类型 dt 后的结果
                tz(mat.astype(dt))

    # 定义一个测试方法，用于测试非法的乘幂情况
    def test_exceptions_bad_power(self, dt):
        # 将 self.rshft_0 转换为指定数据类型 dt 后赋值给 mat
        mat = self.rshft_0.astype(dt)
        # 断言对于非整数乘幂会抛出 TypeError 异常
        assert_raises(TypeError, matrix_power, mat, 1.5)
        assert_raises(TypeError, matrix_power, mat, [1])

    # 定义一个测试方法，用于测试非方阵矩阵的乘幂情况
    def test_exceptions_non_square(self, dt):
        # 断言对于非方阵矩阵会抛出 LinAlgError 异常
        assert_raises(LinAlgError, matrix_power, np.array([1], dt), 1)
        assert_raises(LinAlgError, matrix_power, np.array([[1], [2]], dt), 1)
        assert_raises(LinAlgError, matrix_power, np.ones((4, 3, 2), dt), 1)

    # 使用 pytest.mark.skipif 装饰器标记一个测试方法，用于测试不可逆矩阵的乘幂情况
    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_exceptions_not_invertible(self, dt):
        # 如果数据类型 dt 在 self.dtnoinv 中，则直接返回
        if dt in self.dtnoinv:
            return
        # 将 self.noninv 转换为指定数据类型 dt 后赋值给 mat
        mat = self.noninv.astype(dt)
        # 断言对于不可逆矩阵会抛出 LinAlgError 异常
        assert_raises(LinAlgError, matrix_power, mat, -1)
class TestEigvalshCases(HermitianTestCase, HermitianGeneralizedTestCase):

    def do(self, a, b, tags):
        # note that eigenvalue arrays returned by eig must be sorted since
        # their order isn't guaranteed.
        # 使用 np.linalg.eigvalsh 计算矩阵 a 的特征值，并要求以升序排列
        ev = linalg.eigvalsh(a, 'L')
        
        # 使用 np.linalg.eig 计算矩阵 a 的特征值和特征向量
        evalues, evectors = linalg.eig(a)
        
        # 对通过 np.linalg.eig 计算得到的特征值进行排序
        evalues.sort(axis=-1)
        
        # 断言 np.linalg.eigvalsh 计算得到的特征值数组与 np.linalg.eig 计算得到的特征值数组近似相等
        assert_allclose(ev, evalues, rtol=get_rtol(ev.dtype))

        # 使用 np.linalg.eigvalsh 计算矩阵 a 的特征值，并要求以降序排列
        ev2 = linalg.eigvalsh(a, 'U')
        
        # 断言 np.linalg.eigvalsh 计算得到的特征值数组与先前计算的 evalues 近似相等
        assert_allclose(ev2, evalues, rtol=get_rtol(ev.dtype))


class TestEigvalsh:
    @pytest.mark.parametrize('dtype', [single, double, csingle, cdouble])
    def test_types(self, dtype):
        # 创建一个指定数据类型的数组 x
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        
        # 使用 np.linalg.eigvalsh 计算数组 x 的特征值
        w = np.linalg.eigvalsh(x)
        
        # 断言计算得到的特征值数组的数据类型与预期的实数类型一致
        assert_equal(w.dtype, get_real_dtype(dtype))

    def test_invalid(self):
        # 创建一个指定数据类型的数组 x
        x = np.array([[1, 0.5], [0.5, 1]], dtype=np.float32)
        
        # 断言当使用无效的 UPLO 参数 'lrong' 调用 np.linalg.eigvalsh 时会引发 ValueError
        assert_raises(ValueError, np.linalg.eigvalsh, x, UPLO="lrong")
        
        # 断言当使用 UPLO 参数为 "lower" 调用 np.linalg.eigvalsh 时会引发 ValueError
        assert_raises(ValueError, np.linalg.eigvalsh, x, "lower")
        
        # 断言当使用 UPLO 参数为 "upper" 调用 np.linalg.eigvalsh 时会引发 ValueError
        assert_raises(ValueError, np.linalg.eigvalsh, x, "upper")

    def test_UPLO(self):
        # 创建两个指定数据类型的矩阵 Klo 和 Kup，分别用于测试 'L' 和 'U' 选项
        Klo = np.array([[0, 0], [1, 0]], dtype=np.double)
        Kup = np.array([[0, 1], [0, 0]], dtype=np.double)
        
        # 创建一个目标特征值数组 tgt 和比较的相对误差 rtol
        tgt = np.array([-1, 1], dtype=np.double)
        rtol = get_rtol(np.double)

        # 检查默认情况下 UPLO 参数为 'L' 时 np.linalg.eigvalsh 返回的特征值与目标 tgt 近似相等
        w = np.linalg.eigvalsh(Klo)
        assert_allclose(w, tgt, rtol=rtol)
        
        # 检查 UPLO 参数为 'L' 时 np.linalg.eigvalsh 返回的特征值与目标 tgt 近似相等
        w = np.linalg.eigvalsh(Klo, UPLO='L')
        assert_allclose(w, tgt, rtol=rtol)
        
        # 检查 UPLO 参数为 'l' 时 np.linalg.eigvalsh 返回的特征值与目标 tgt 近似相等
        w = np.linalg.eigvalsh(Klo, UPLO='l')
        assert_allclose(w, tgt, rtol=rtol)
        
        # 检查 UPLO 参数为 'U' 时 np.linalg.eigvalsh 返回的特征值与目标 tgt 近似相等
        w = np.linalg.eigvalsh(Kup, UPLO='U')
        assert_allclose(w, tgt, rtol=rtol)
        
        # 检查 UPLO 参数为 'u' 时 np.linalg.eigvalsh 返回的特征值与目标 tgt 近似相等
        w = np.linalg.eigvalsh(Kup, UPLO='u')
        assert_allclose(w, tgt, rtol=rtol)

    def test_0_size(self):
        # 检查所有大小为 0 的数组类型都能正常工作
        
        # 创建一个 ArraySubclass 类型的零大小数组 a，并将其转换为 np.ndarray 类型
        class ArraySubclass(np.ndarray):
            pass
        
        a = np.zeros((0, 1, 1), dtype=np.int_).view(ArraySubclass)
        
        # 使用 np.linalg.eigvalsh 计算零大小数组 a 的特征值
        res = linalg.eigvalsh(a)
        
        # 断言计算得到的特征值数组的数据类型为 np.float64
        assert_(res.dtype.type is np.float64)
        
        # 断言计算得到的特征值数组的形状为 (0, 1)
        assert_equal((0, 1), res.shape)
        
        # 这只是为了说明，可能需要更改：
        # 断言计算得到的 res 是 np.ndarray 类型的实例
        assert_(isinstance(res, np.ndarray))

        a = np.zeros((0, 0), dtype=np.complex64).view(ArraySubclass)
        
        # 使用 np.linalg.eigvalsh 计算零大小数组 a 的特征值
        res = linalg.eigvalsh(a)
        
        # 断言计算得到的特征值数组的数据类型为 np.float32
        assert_(res.dtype.type is np.float32)
        
        # 断言计算得到的特征值数组的形状为 (0,)
        assert_equal((0,), res.shape)
        
        # 这只是为了说明，可能需要更改：
        # 断言计算得到的 res 是 np.ndarray 类型的实例
        assert_(isinstance(res, np.ndarray))


class TestEighCases(HermitianTestCase, HermitianGeneralizedTestCase):
    # 未完待续，注释继续在下一个类的注释中添加
    # 定义一个方法 `do`，接受参数 `a`, `b`, `tags`
    def do(self, a, b, tags):
        # 警告：eig 返回的特征值数组需要排序，因为它们的顺序不保证一致。
        # 使用 linalg.eigh 计算矩阵 a 的特征值和特征向量
        res = linalg.eigh(a)
        # 分别获取特征值和特征向量
        ev, evc = res.eigenvalues, res.eigenvectors
        # 使用 linalg.eig 计算矩阵 a 的特征值和特征向量
        evalues, evectors = linalg.eig(a)
        # 对特征值进行排序，沿着最后一个轴进行排序
        evalues.sort(axis=-1)
        # 断言 ev（通过 linalg.eigh 得到的特征值）与 evalues（通过 linalg.eig 得到并排序的特征值）几乎相等
        assert_almost_equal(ev, evalues)

        # 断言矩阵 a 与特征向量 evc 的乘积，等于 evc 的每个列向量乘以对应的特征值，这里使用 matmul 进行矩阵乘法
        assert_allclose(matmul(a, evc),
                        np.asarray(ev)[..., None, :] * np.asarray(evc),
                        rtol=get_rtol(ev.dtype))

        # 使用 linalg.eigh 计算矩阵 a 的特征值和特征向量，参数 'U' 表示返回的特征向量是未正交化的
        ev2, evc2 = linalg.eigh(a, 'U')
        # 断言 ev2（通过 linalg.eigh 得到的未正交化的特征值）与 evalues（通过 linalg.eig 得到并排序的特征值）几乎相等
        assert_almost_equal(ev2, evalues)

        # 断言矩阵 a 与特征向量 evc2 的乘积，等于 evc2 的每个列向量乘以对应的特征值，这里使用 matmul 进行矩阵乘法
        assert_allclose(matmul(a, evc2),
                        np.asarray(ev2)[..., None, :] * np.asarray(evc2),
                        rtol=get_rtol(ev.dtype), err_msg=repr(a))
class TestEigh:
    # 使用 pytest 的参数化装饰器，测试不同的数据类型
    @pytest.mark.parametrize('dtype', [single, double, csingle, cdouble])
    def test_types(self, dtype):
        # 创建一个二维数组 x，指定数据类型为 dtype
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        # 计算对称矩阵 x 的特征值和特征向量
        w, v = np.linalg.eigh(x)
        # 断言特征值的数据类型与给定 dtype 对应的实数部分的数据类型相同
        assert_equal(w.dtype, get_real_dtype(dtype))
        # 断言特征向量的数据类型与给定 dtype 相同
        assert_equal(v.dtype, dtype)

    # 测试处理无效输入的情况
    def test_invalid(self):
        # 创建一个二维数组 x，指定数据类型为 np.float32
        x = np.array([[1, 0.5], [0.5, 1]], dtype=np.float32)
        # 断言当 UPLO 参数为 "lrong" 时，调用 np.linalg.eigh(x) 抛出 ValueError 异常
        assert_raises(ValueError, np.linalg.eigh, x, UPLO="lrong")
        # 断言当 UPLO 参数为 "lower" 时，调用 np.linalg.eigh(x) 抛出 ValueError 异常
        assert_raises(ValueError, np.linalg.eigh, x, "lower")
        # 断言当 UPLO 参数为 "upper" 时，调用 np.linalg.eigh(x) 抛出 ValueError 异常
        assert_raises(ValueError, np.linalg.eigh, x, "upper")

    # 测试不同的 UPLO 参数对结果的影响
    def test_UPLO(self):
        # 创建两个双精度二维数组 Klo 和 Kup
        Klo = np.array([[0, 0], [1, 0]], dtype=np.double)
        Kup = np.array([[0, 1], [0, 0]], dtype=np.double)
        # 创建目标特征值数组 tgt，数据类型为双精度
        tgt = np.array([-1, 1], dtype=np.double)
        # 获取双精度数值的相对误差容限
        rtol = get_rtol(np.double)

        # 检查默认情况下 UPLO 参数为 'L' 时的特征值计算结果
        w, v = np.linalg.eigh(Klo)
        assert_allclose(w, tgt, rtol=rtol)
        # 检查显式指定 UPLO 参数为 'L' 时的特征值计算结果
        w, v = np.linalg.eigh(Klo, UPLO='L')
        assert_allclose(w, tgt, rtol=rtol)
        # 检查 UPLO 参数为 'l' 时的特征值计算结果
        w, v = np.linalg.eigh(Klo, UPLO='l')
        assert_allclose(w, tgt, rtol=rtol)
        # 检查 UPLO 参数为 'U' 时的特征值计算结果
        w, v = np.linalg.eigh(Kup, UPLO='U')
        assert_allclose(w, tgt, rtol=rtol)
        # 检查 UPLO 参数为 'u' 时的特征值计算结果
        w, v = np.linalg.eigh(Kup, UPLO='u')
        assert_allclose(w, tgt, rtol=rtol)

    # 测试处理大小为零的情况
    def test_0_size(self):
        # 检查各种零大小数组的特征值和特征向量计算
        class ArraySubclass(np.ndarray):
            pass
        # 创建一个 shape 为 (0, 1, 1) 的整数类型的零数组 a，视图类型为 ArraySubclass
        a = np.zeros((0, 1, 1), dtype=np.int_).view(ArraySubclass)
        res, res_v = linalg.eigh(a)
        # 断言特征向量的数据类型为 np.float64
        assert_(res_v.dtype.type is np.float64)
        # 断言特征值的数据类型为 np.float64
        assert_(res.dtype.type is np.float64)
        # 断言数组 a 的形状与特征向量 res_v 的形状相同
        assert_equal(a.shape, res_v.shape)
        # 断言特征值 res 的形状为 (0, 1)
        assert_equal((0, 1), res.shape)
        # 这是为了文档说明，可能会有变化：
        assert_(isinstance(a, np.ndarray))

        # 创建一个 shape 为 (0, 0) 的复数类型的零数组 a，视图类型为 ArraySubclass
        a = np.zeros((0, 0), dtype=np.complex64).view(ArraySubclass)
        res, res_v = linalg.eigh(a)
        # 断言特征向量的数据类型为 np.complex64
        assert_(res_v.dtype.type is np.complex64)
        # 断言特征值的数据类型为 np.float32
        assert_(res.dtype.type is np.float32)
        # 断言数组 a 的形状与特征向量 res_v 的形状相同
        assert_equal(a.shape, res_v.shape)
        # 断言特征值 res 的形状为 (0,)
        assert_equal((0,), res.shape)
        # 这是为了文档说明，可能会有变化：
        assert_(isinstance(a, np.ndarray))


class _TestNormBase:
    dt = None
    dec = None

    # 静态方法：检查输入数组的数据类型
    @staticmethod
    def check_dtype(x, res):
        # 如果输入数组 x 的数据类型是 np.inexact 的子类
        if issubclass(x.dtype.type, np.inexact):
            # 断言结果数组 res 的数据类型与 x 实数部分的数据类型相同
            assert_equal(res.dtype, x.real.dtype)
        else:
            # 对于整数输入，不必测试输出的浮点精度。
            assert_(issubclass(res.dtype.type, np.floating))


class _TestNormGeneral(_TestNormBase):

    # 测试空数组的情况
    def test_empty(self):
        # 断言对空列表计算的范数为 0.0
        assert_equal(norm([]), 0.0)
        # 断言对 shape 为空的数组计算的范数为 0.0
        assert_equal(norm(array([], dtype=self.dt)), 0.0)
        # 断言对至少二维 shape 为空的数组计算的范数为 0.0
        assert_equal(norm(atleast_2d(array([], dtype=self.dt))), 0.0)
    # 定义测试向量返回类型的方法
    def test_vector_return_type(self):
        # 创建一个 NumPy 数组 [1, 0, 1]
        a = np.array([1, 0, 1])

        # 获取所有精确类型的类型码
        exact_types = np.typecodes['AllInteger']
        # 获取所有非精确类型的类型码
        inexact_types = np.typecodes['AllFloat']

        # 合并精确和非精确类型的类型码
        all_types = exact_types + inexact_types

        # 遍历所有类型
        for each_type in all_types:
            # 将数组 a 转换为当前遍历的类型
            at = a.astype(each_type)

            # 计算 L-infinity 范数
            an = norm(at, -np.inf)
            self.check_dtype(at, an)
            # 断言 L-infinity 范数近似为 0.0
            assert_almost_equal(an, 0.0)

            # 忽略运行时警告并计算 L-1 范数
            with suppress_warnings() as sup:
                sup.filter(RuntimeWarning, "divide by zero encountered")
                an = norm(at, -1)
                self.check_dtype(at, an)
                # 断言 L-1 范数近似为 0.0
                assert_almost_equal(an, 0.0)

            # 计算 L-0 范数
            an = norm(at, 0)
            self.check_dtype(at, an)
            # 断言 L-0 范数为 2
            assert_almost_equal(an, 2)

            # 计算 L-1 范数
            an = norm(at, 1)
            self.check_dtype(at, an)
            # 断言 L-1 范数为 2.0
            assert_almost_equal(an, 2.0)

            # 计算 L-2 范数
            an = norm(at, 2)
            self.check_dtype(at, an)
            # 断言 L-2 范数近似为 sqrt(2)
            assert_almost_equal(an, an.dtype.type(2.0)**an.dtype.type(1.0/2.0))

            # 计算 L-4 范数
            an = norm(at, 4)
            self.check_dtype(at, an)
            # 断言 L-4 范数近似为 2^(1/4)
            assert_almost_equal(an, an.dtype.type(2.0)**an.dtype.type(1.0/4.0))

            # 计算 L-infinity 范数
            an = norm(at, np.inf)
            self.check_dtype(at, an)
            # 断言 L-infinity 范数近似为 1.0
            assert_almost_equal(an, 1.0)

    # 定义测试向量的方法
    def test_vector(self):
        # 定义向量 a, b, c
        a = [1, 2, 3, 4]
        b = [-1, -2, -3, -4]
        c = [-1, 2, -3, 4]

        # 定义内部函数 _test，用于测试向量的不同范数
        def _test(v):
            # 断言计算向量 v 的范数近似为 sqrt(30)，小数位数为 self.dec
            np.testing.assert_almost_equal(norm(v), 30 ** 0.5,
                                           decimal=self.dec)
            # 断言计算向量 v 的 L-infinity 范数近似为 4.0，小数位数为 self.dec
            np.testing.assert_almost_equal(norm(v, np.inf), 4.0,
                                           decimal=self.dec)
            # 断言计算向量 v 的 L-norm 范数近似为 1.0，小数位数为 self.dec
            np.testing.assert_almost_equal(norm(v, -np.inf), 1.0,
                                           decimal=self.dec)
            # 断言计算向量 v 的 L-1 范数近似为 10.0，小数位数为 self.dec
            np.testing.assert_almost_equal(norm(v, 1), 10.0,
                                           decimal=self.dec)
            # 断言计算向量 v 的 L-(-1) 范数近似为 12.0 / 25，小数位数为 self.dec
            np.testing.assert_almost_equal(norm(v, -1), 12.0 / 25,
                                           decimal=self.dec)
            # 断言计算向量 v 的 L-2 范数近似为 sqrt(30)，小数位数为 self.dec
            np.testing.assert_almost_equal(norm(v, 2), 30 ** 0.5,
                                           decimal=self.dec)
            # 断言计算向量 v 的 L-(-2) 范数近似为 (205. / 144) ** -0.5，小数位数为 self.dec
            np.testing.assert_almost_equal(norm(v, -2), ((205. / 144) ** -0.5),
                                           decimal=self.dec)
            # 断言计算向量 v 的 L-0 范数近似为 4，小数位数为 self.dec
            np.testing.assert_almost_equal(norm(v, 0), 4,
                                           decimal=self.dec)

        # 遍历向量 a, b, c，并分别调用 _test 方法进行范数测试
        for v in (a, b, c,):
            _test(v)

        # 使用指定的数据类型 self.dt 创建数组，并对每个数组调用 _test 方法进行范数测试
        for v in (array(a, dtype=self.dt), array(b, dtype=self.dt),
                  array(c, dtype=self.dt)):
            _test(v)
    def test_axis(self):
        # 测试轴向操作函数
        # 比较使用 `axis` 参数与分别计算每行或每列范数的方法。

        A = array([[1, 2, 3], [4, 5, 6]], dtype=self.dt)
        # 对于不同的范数参数进行迭代测试
        for order in [None, -1, 0, 1, 2, 3, np.inf, -np.inf]:
            # 计算每列的预期范数
            expected0 = [norm(A[:, k], ord=order) for k in range(A.shape[1])]
            # 断言计算出的每列范数与预期值相近
            assert_almost_equal(norm(A, ord=order, axis=0), expected0)

            # 计算每行的预期范数
            expected1 = [norm(A[k, :], ord=order) for k in range(A.shape[0])]
            # 断言计算出的每行范数与预期值相近
            assert_almost_equal(norm(A, ord=order, axis=1), expected1)

        B = np.arange(1, 25, dtype=self.dt).reshape(2, 3, 4)
        nd = B.ndim
        # 对于不同的范数参数和轴组合进行迭代测试
        for order in [None, -2, 2, -1, 1, np.inf, -np.inf, 'fro']:
            for axis in itertools.combinations(range(-nd, nd), 2):
                row_axis, col_axis = axis
                if row_axis < 0:
                    row_axis += nd
                if col_axis < 0:
                    col_axis += nd
                if row_axis == col_axis:
                    # 断言当行轴和列轴相同时会抛出 ValueError 异常
                    assert_raises(ValueError, norm, B, ord=order, axis=axis)
                else:
                    # 计算给定轴的范数
                    n = norm(B, ord=order, axis=axis)

                    # 根据轴的组合确定 k_index 的逻辑（仅在 nd = 3 时有效）
                    # 如果 nd 增加，这部分逻辑需要调整
                    k_index = nd - (row_axis + col_axis)
                    if row_axis < col_axis:
                        # 计算预期的轴范数值
                        expected = [norm(B[:].take(k, axis=k_index), ord=order)
                                    for k in range(B.shape[k_index])]
                    else:
                        expected = [norm(B[:].take(k, axis=k_index).T, ord=order)
                                    for k in range(B.shape[k_index])]
                    # 断言计算出的范数与预期值相近
                    assert_almost_equal(n, expected)
    # 定义一个测试方法，用于测试保持维度参数的效果
    def test_keepdims(self):
        # 创建一个形状为 (2, 3, 4) 的 NumPy 数组 A，数据类型由类的成员变量 self.dt 指定
        A = np.arange(1, 25, dtype=self.dt).reshape(2, 3, 4)

        # 错误消息格式字符串，用于在 assert_allclose 失败时输出
        allclose_err = 'order {0}, axis = {1}'
        # 形状不匹配的错误消息格式字符串，用于在 assert_ 失败时输出
        shape_err = 'Shape mismatch found {0}, expected {1}, order={2}, axis={3}'

        # 检查 order=None, axis=None 的情况
        expected = norm(A, ord=None, axis=None)
        found = norm(A, ord=None, axis=None, keepdims=True)
        # 使用 assert_allclose 检查 found 是否接近于 expected，将其展平以匹配形状
        assert_allclose(np.squeeze(found), expected,
                        err_msg=allclose_err.format(None, None))
        # 预期的形状应为 (1, 1, 1)
        expected_shape = (1, 1, 1)
        # 使用 assert_ 检查 found 的形状是否与 expected_shape 相匹配
        assert_(found.shape == expected_shape,
                shape_err.format(found.shape, expected_shape, None, None))

        # 向量范数。
        for order in [None, -1, 0, 1, 2, 3, np.inf, -np.inf]:
            for k in range(A.ndim):
                expected = norm(A, ord=order, axis=k)
                found = norm(A, ord=order, axis=k, keepdims=True)
                # 使用 assert_allclose 检查 found 是否接近于 expected，将其展平以匹配形状
                assert_allclose(np.squeeze(found), expected,
                                err_msg=allclose_err.format(order, k))
                # 计算预期的形状，将对应轴的长度设置为 1
                expected_shape = list(A.shape)
                expected_shape[k] = 1
                expected_shape = tuple(expected_shape)
                # 使用 assert_ 检查 found 的形状是否与 expected_shape 相匹配
                assert_(found.shape == expected_shape,
                        shape_err.format(found.shape, expected_shape, order, k))

        # 矩阵范数。
        import itertools
        for order in [None, -2, 2, -1, 1, np.inf, -np.inf, 'fro', 'nuc']:
            for k in itertools.permutations(range(A.ndim), 2):
                expected = norm(A, ord=order, axis=k)
                found = norm(A, ord=order, axis=k, keepdims=True)
                # 使用 assert_allclose 检查 found 是否接近于 expected，将其展平以匹配形状
                assert_allclose(np.squeeze(found), expected,
                                err_msg=allclose_err.format(order, k))
                # 计算预期的形状，将对应轴的长度设置为 1
                expected_shape = list(A.shape)
                expected_shape[k[0]] = 1
                expected_shape[k[1]] = 1
                expected_shape = tuple(expected_shape)
                # 使用 assert_ 检查 found 的形状是否与 expected_shape 相匹配
                assert_(found.shape == expected_shape,
                        shape_err.format(found.shape, expected_shape, order, k))
class _TestNorm2D(_TestNormBase):
    # 2维数组的规范化测试类，继承自_TestNormBase

    # Define the part for 2d arrays separately, so we can subclass this
    # and run the tests using np.matrix in matrixlib.tests.test_matrix_linalg.
    # 将2维数组的部分定义为单独的部分，以便我们可以在matrixlib.tests.test_matrix_linalg中使用np.matrix进行子类化和测试运行。

    array = np.array  # 使用np.array作为数组创建的函数

    def test_matrix_empty(self):
        # 测试空矩阵的规范化结果是否为0
        assert_equal(norm(self.array([[]], dtype=self.dt)), 0.0)

    def test_matrix_return_type(self):
        a = self.array([[1, 0, 1], [0, 1, 1]])

        exact_types = np.typecodes['AllInteger']  # 获取所有精确整数类型的类型码

        # float32, complex64, float64, complex128 types are the only types
        # allowed by `linalg`, which performs the matrix operations used
        # within `norm`.
        # `linalg`只允许float32、complex64、float64、complex128类型，这些类型用于执行规范化中使用的矩阵操作。

        inexact_types = 'fdFD'  # 不精确类型的类型码

        all_types = exact_types + inexact_types  # 所有可能的数据类型

        for each_type in all_types:
            at = a.astype(each_type)  # 将数组a转换为当前遍历到的数据类型

            an = norm(at, -np.inf)  # 计算规范化（范数），使用负无穷范数
            self.check_dtype(at, an)  # 检查计算后的数据类型
            assert_almost_equal(an, 2.0)  # 断言计算结果接近2.0

            with suppress_warnings() as sup:
                sup.filter(RuntimeWarning, "divide by zero encountered")
                an = norm(at, -1)  # 计算规范化，使用-1范数
                self.check_dtype(at, an)  # 检查计算后的数据类型
                assert_almost_equal(an, 1.0)  # 断言计算结果接近1.0

            an = norm(at, 1)  # 计算规范化，使用1范数
            self.check_dtype(at, an)  # 检查计算后的数据类型
            assert_almost_equal(an, 2.0)  # 断言计算结果接近2.0

            an = norm(at, 2)  # 计算规范化，使用2范数
            self.check_dtype(at, an)  # 检查计算后的数据类型
            assert_almost_equal(an, 3.0**(1.0/2.0))  # 断言计算结果接近sqrt(3)

            an = norm(at, -2)  # 计算规范化，使用-2范数
            self.check_dtype(at, an)  # 检查计算后的数据类型
            assert_almost_equal(an, 1.0)  # 断言计算结果接近1.0

            an = norm(at, np.inf)  # 计算规范化，使用无穷范数
            self.check_dtype(at, an)  # 检查计算后的数据类型
            assert_almost_equal(an, 2.0)  # 断言计算结果接近2.0

            an = norm(at, 'fro')  # 计算规范化，使用Frobenius范数
            self.check_dtype(at, an)  # 检查计算后的数据类型
            assert_almost_equal(an, 2.0)  # 断言计算结果接近2.0

            an = norm(at, 'nuc')  # 计算规范化，使用核范数
            self.check_dtype(at, an)  # 检查计算后的数据类型
            # 需要一个较低的界限以支持低精度浮点数。
            # 它们在第7位上有一个偏差。
            np.testing.assert_almost_equal(an, 2.7320508075688772, decimal=6)  # 断言计算结果接近给定的值

    def test_matrix_2x2(self):
        A = self.array([[1, 3], [5, 7]], dtype=self.dt)
        assert_almost_equal(norm(A), 84 ** 0.5)  # 断言计算规范化后的结果接近sqrt(84)
        assert_almost_equal(norm(A, 'fro'), 84 ** 0.5)  # 断言计算Frobenius范数后的结果接近sqrt(84)
        assert_almost_equal(norm(A, 'nuc'), 10.0)  # 断言计算核范数后的结果接近10.0
        assert_almost_equal(norm(A, inf), 12.0)  # 断言计算无穷范数后的结果接近12.0
        assert_almost_equal(norm(A, -inf), 4.0)  # 断言计算负无穷范数后的结果接近4.0
        assert_almost_equal(norm(A, 1), 10.0)  # 断言计算1范数后的结果接近10.0
        assert_almost_equal(norm(A, -1), 6.0)  # 断言计算-1范数后的结果接近6.0
        assert_almost_equal(norm(A, 2), 9.1231056256176615)  # 断言计算2范数后的结果接近给定的值
        assert_almost_equal(norm(A, -2), 0.87689437438234041)  # 断言计算-2范数后的结果接近给定的值

        assert_raises(ValueError, norm, A, 'nofro')  # 断言计算非法Frobenius范数时抛出ValueError异常
        assert_raises(ValueError, norm, A, -3)  # 断言计算非法范数时抛出ValueError异常
        assert_raises(ValueError, norm, A, 0)  # 断言计算范数0时抛出ValueError异常
    def test_matrix_3x3(self):
        # This test has been added because the 2x2 example
        # happened to have equal nuclear norm and induced 1-norm.
        # The 1/10 scaling factor accommodates the absolute tolerance
        # used in assert_almost_equal.

        # 创建一个3x3的矩阵A，使用self.array方法，数据类型为self.dt
        A = (1 / 10) * \
            self.array([[1, 2, 3], [6, 0, 5], [3, 2, 1]], dtype=self.dt)

        # 检查A的2范数是否与给定值几乎相等，绝对容差由1/10缩放
        assert_almost_equal(norm(A), (1 / 10) * 89 ** 0.5)

        # 检查A的Frobenius范数是否与给定值几乎相等，绝对容差由1/10缩放
        assert_almost_equal(norm(A, 'fro'), (1 / 10) * 89 ** 0.5)

        # 检查A的核范数是否与给定值几乎相等
        assert_almost_equal(norm(A, 'nuc'), 1.3366836911774836)

        # 检查A的无穷大范数是否与给定值几乎相等
        assert_almost_equal(norm(A, inf), 1.1)

        # 检查A的负无穷大范数是否与给定值几乎相等
        assert_almost_equal(norm(A, -inf), 0.6)

        # 检查A的1范数是否与给定值几乎相等
        assert_almost_equal(norm(A, 1), 1.0)

        # 检查A的负1范数是否与给定值几乎相等
        assert_almost_equal(norm(A, -1), 0.4)

        # 检查A的2范数是否与给定值几乎相等
        assert_almost_equal(norm(A, 2), 0.88722940323461277)

        # 检查A的负2范数是否与给定值几乎相等
        assert_almost_equal(norm(A, -2), 0.19456584790481812)

    def test_bad_args(self):
        # Check that bad arguments raise the appropriate exceptions.

        # 创建矩阵A和张量B，数据类型为self.dt
        A = self.array([[1, 2, 3], [4, 5, 6]], dtype=self.dt)
        B = np.arange(1, 25, dtype=self.dt).reshape(2, 3, 4)

        # 当ord为'fro'或'nuc'时，使用axis=<integer>或传递1-D数组会导致
        # 抛出ValueError异常
        assert_raises(ValueError, norm, A, 'fro', 0)
        assert_raises(ValueError, norm, A, 'nuc', 0)
        assert_raises(ValueError, norm, [3, 4], 'fro', None)
        assert_raises(ValueError, norm, [3, 4], 'nuc', None)
        assert_raises(ValueError, norm, [3, 4], 'test', None)

        # 当ord为除1, 2, -1或-2之外的任何有限数时，计算矩阵范数时，norm应抛出异常
        for order in [0, 3]:
            assert_raises(ValueError, norm, A, order, None)
            assert_raises(ValueError, norm, A, order, (0, 1))
            assert_raises(ValueError, norm, B, order, (1, 2))

        # 无效的axis值应该抛出AxisError异常
        assert_raises(AxisError, norm, B, None, 3)
        assert_raises(AxisError, norm, B, None, (2, 3))
        assert_raises(ValueError, norm, B, None, (0, 1, 2))
class _TestNorm(_TestNorm2D, _TestNormGeneral):
    pass


class TestNorm_NonSystematic:

    def test_longdouble_norm(self):
        # 非回归测试：longdouble 类型的 p-范数以前会引发 UnboundLocalError。
        x = np.arange(10, dtype=np.longdouble)
        old_assert_almost_equal(norm(x, ord=3), 12.65, decimal=2)

    def test_intmin(self):
        # 非回归测试：有符号整数的 p-范数以前的顺序中进行了 float 转换和错误的绝对值。
        x = np.array([-2 ** 31], dtype=np.int32)
        old_assert_almost_equal(norm(x, ord=3), 2 ** 31, decimal=5)

    def test_complex_high_ord(self):
        # gh-4156
        # 高阶复数测试
        d = np.empty((2,), dtype=np.clongdouble)
        d[0] = 6 + 7j
        d[1] = -6 + 7j
        res = 11.615898132184
        old_assert_almost_equal(np.linalg.norm(d, ord=3), res, decimal=10)
        d = d.astype(np.complex128)
        old_assert_almost_equal(np.linalg.norm(d, ord=3), res, decimal=9)
        d = d.astype(np.complex64)
        old_assert_almost_equal(np.linalg.norm(d, ord=3), res, decimal=5)


# Separate definitions so we can use them for matrix tests.
class _TestNormDoubleBase(_TestNormBase):
    dt = np.double
    dec = 12


class _TestNormSingleBase(_TestNormBase):
    dt = np.float32
    dec = 6


class _TestNormInt64Base(_TestNormBase):
    dt = np.int64
    dec = 12


class TestNormDouble(_TestNorm, _TestNormDoubleBase):
    pass


class TestNormSingle(_TestNorm, _TestNormSingleBase):
    pass


class TestNormInt64(_TestNorm, _TestNormInt64Base):
    pass


class TestMatrixRank:

    def test_matrix_rank(self):
        # Full rank matrix
        # 满秩矩阵
        assert_equal(4, matrix_rank(np.eye(4)))
        # rank deficient matrix
        # 排名不足的矩阵
        I = np.eye(4)
        I[-1, -1] = 0.
        assert_equal(matrix_rank(I), 3)
        # All zeros - zero rank
        # 全零矩阵 - 零秩
        assert_equal(matrix_rank(np.zeros((4, 4))), 0)
        # 1 dimension - rank 1 unless all 0
        # 1 维度 - 秩为 1，除非全为 0
        assert_equal(matrix_rank([1, 0, 0, 0]), 1)
        assert_equal(matrix_rank(np.zeros((4,))), 0)
        # accepts array-like
        # 接受类数组输入
        assert_equal(matrix_rank([1]), 1)
        # greater than 2 dimensions treated as stacked matrices
        # 大于 2 维的数组被视为堆叠矩阵
        ms = np.array([I, np.eye(4), np.zeros((4,4))])
        assert_equal(matrix_rank(ms), np.array([3, 4, 0]))
        # works on scalar
        # 对标量也适用
        assert_equal(matrix_rank(1), 1)

        with assert_raises_regex(
            ValueError, "`tol` and `rtol` can\'t be both set."
        ):
            matrix_rank(I, tol=0.01, rtol=0.01)
    # 定义一个测试函数，用于测试 matrix_rank 函数在对称矩阵情况下的行列式计算
    def test_symmetric_rank(self):
        # 断言对单位矩阵求秩应该得到4，hermitian=True 表示输入为对称矩阵
        assert_equal(4, matrix_rank(np.eye(4), hermitian=True))
        # 断言对全1矩阵求秩应该得到1，hermitian=True 表示输入为对称矩阵
        assert_equal(1, matrix_rank(np.ones((4, 4)), hermitian=True))
        # 断言对全0矩阵求秩应该得到0，hermitian=True 表示输入为对称矩阵
        assert_equal(0, matrix_rank(np.zeros((4, 4)), hermitian=True))
        
        # 创建一个秩亏矩阵 I，将其最后一个元素置为0
        I = np.eye(4)
        I[-1, -1] = 0.
        # 断言对秩亏矩阵求秩应该得到3，hermitian=True 表示输入为对称矩阵
        assert_equal(3, matrix_rank(I, hermitian=True))
        
        # 手动设置一个较小的容差值
        I[-1, -1] = 1e-8
        # 断言对秩亏矩阵求秩应该得到4，hermitian=True，tol=0.99e-8 表示输入为对称矩阵，并且使用指定的容差值
        assert_equal(4, matrix_rank(I, hermitian=True, tol=0.99e-8))
        # 断言对秩亏矩阵求秩应该得到3，hermitian=True，tol=1.01e-8 表示输入为对称矩阵，并且使用指定的容差值
        assert_equal(3, matrix_rank(I, hermitian=True, tol=1.01e-8))
def test_reduced_rank():
    # Test matrices with reduced rank
    rng = np.random.RandomState(20120714)
    for i in range(100):
        # Make a rank deficient matrix
        X = rng.normal(size=(40, 10))
        X[:, 0] = X[:, 1] + X[:, 2]
        # Assert that matrix_rank detected deficiency
        assert_equal(matrix_rank(X), 9)
        X[:, 3] = X[:, 4] + X[:, 5]
        assert_equal(matrix_rank(X), 8)


class TestQR:
    # Define the array class here, so run this on matrices elsewhere.
    array = np.array

    def check_qr(self, a):
        # This test expects the argument `a` to be an ndarray or
        # a subclass of an ndarray of inexact type.
        a_type = type(a)
        a_dtype = a.dtype
        m, n = a.shape
        k = min(m, n)

        # mode == 'complete'
        res = linalg.qr(a, mode='complete')
        Q, R = res.Q, res.R
        assert_(Q.dtype == a_dtype)
        assert_(R.dtype == a_dtype)
        assert_(isinstance(Q, a_type))
        assert_(isinstance(R, a_type))
        assert_(Q.shape == (m, m))
        assert_(R.shape == (m, n))
        assert_almost_equal(dot(Q, R), a)
        assert_almost_equal(dot(Q.T.conj(), Q), np.eye(m))
        assert_almost_equal(np.triu(R), R)

        # mode == 'reduced'
        q1, r1 = linalg.qr(a, mode='reduced')
        assert_(q1.dtype == a_dtype)
        assert_(r1.dtype == a_dtype)
        assert_(isinstance(q1, a_type))
        assert_(isinstance(r1, a_type))
        assert_(q1.shape == (m, k))
        assert_(r1.shape == (k, n))
        assert_almost_equal(dot(q1, r1), a)
        assert_almost_equal(dot(q1.T.conj(), q1), np.eye(k))
        assert_almost_equal(np.triu(r1), r1)

        # mode == 'r'
        r2 = linalg.qr(a, mode='r')
        assert_(r2.dtype == a_dtype)
        assert_(isinstance(r2, a_type))
        assert_almost_equal(r2, r1)


    @pytest.mark.parametrize(["m", "n"], [
        (3, 0),
        (0, 3),
        (0, 0)
    ])
    def test_qr_empty(self, m, n):
        k = min(m, n)
        a = np.empty((m, n))

        self.check_qr(a)

        # Verify QR decomposition for empty matrix
        h, tau = np.linalg.qr(a, mode='raw')
        assert_equal(h.dtype, np.double)
        assert_equal(tau.dtype, np.double)
        assert_equal(h.shape, (n, m))
        assert_equal(tau.shape, (k,))
    # 定义一个测试方法，用于测试 QR 分解函数在 mode='raw' 模式下的行为
    def test_mode_raw(self):
        # 由于因式分解在不同库之间不唯一，结果可能会有所不同，
        # 因此无法根据已知值进行检查。功能测试是一种可能性，
        # 但需要更多 lapack_lite 中函数的曝光。因此，这个测试的范围非常有限。
        # 注意结果是按照 FORTRAN 顺序排列的，因此 h 数组是转置的。
        
        # 创建一个 3x2 的双精度浮点数数组 a
        a = self.array([[1, 2], [3, 4], [5, 6]], dtype=np.double)

        # 测试双精度浮点数
        # 使用 linalg.qr 函数进行 QR 分解，返回 h 和 tau
        h, tau = linalg.qr(a, mode='raw')
        # 断言 h 的数据类型为双精度浮点数
        assert_(h.dtype == np.double)
        # 断言 tau 的数据类型为双精度浮点数
        assert_(tau.dtype == np.double)
        # 断言 h 的形状为 (2, 3)
        assert_(h.shape == (2, 3))
        # 断言 tau 的形状为 (2,)
        assert_(tau.shape == (2,))

        # 对 a 的转置进行 QR 分解测试
        h, tau = linalg.qr(a.T, mode='raw')
        # 断言 h 的数据类型为双精度浮点数
        assert_(h.dtype == np.double)
        # 断言 tau 的数据类型为双精度浮点数
        assert_(tau.dtype == np.double)
        # 断言 h 的形状为 (3, 2)
        assert_(h.shape == (3, 2))
        # 断言 tau 的形状为 (2,)
        assert_(tau.shape == (2,))

    # 定义一个测试方法，用于测试 QR 分解函数在除经济模式以外的所有模式下的行为
    def test_mode_all_but_economic(self):
        # 创建两个不同形状的数组 a 和 b
        a = self.array([[1, 2], [3, 4]])
        b = self.array([[1, 2], [3, 4], [5, 6]])
        
        # 遍历浮点数类型 'f' 和 'd'
        for dt in "fd":
            # 将 a 和 b 转换为当前浮点数类型 dt 的数组 m1 和 m2
            m1 = a.astype(dt)
            m2 = b.astype(dt)
            # 对 m1 和 m2 分别进行 QR 分解并检查结果
            self.check_qr(m1)
            self.check_qr(m2)
            # 对 m2 的转置进行 QR 分解并检查结果
            self.check_qr(m2.T)

        # 遍历复数类型 'f' 和 'd'
        for dt in "fd":
            # 创建复数类型为 1 + 1j 的数组 m1 和 m2
            m1 = 1 + 1j * a.astype(dt)
            m2 = 1 + 1j * b.astype(dt)
            # 对 m1 和 m2 分别进行 QR 分解并检查结果
            self.check_qr(m1)
            self.check_qr(m2)
            # 对 m2 的转置进行 QR 分解并检查结果
            self.check_qr(m2.T)
    # 定义一个方法，用于检查 QR 分解在不同模式下的行为
    def check_qr_stacked(self, a):
        # 此测试期望参数 `a` 是一个 ndarray 或其子类，数据类型是不精确的
        a_type = type(a)
        a_dtype = a.dtype
        m, n = a.shape[-2:]
        k = min(m, n)

        # mode == 'complete' 模式下的 QR 分解
        q, r = linalg.qr(a, mode='complete')
        # 断言结果的数据类型与输入一致
        assert_(q.dtype == a_dtype)
        assert_(r.dtype == a_dtype)
        # 断言结果的类型与输入一致
        assert_(isinstance(q, a_type))
        assert_(isinstance(r, a_type))
        # 断言 Q 和 R 的形状与预期相符
        assert_(q.shape[-2:] == (m, m))
        assert_(r.shape[-2:] == (m, n))
        # 断言 QR 分解的乘积等于原始矩阵
        assert_almost_equal(matmul(q, r), a)
        # 创建单位矩阵
        I_mat = np.identity(q.shape[-1])
        # 扩展单位矩阵以匹配 Q 的形状
        stack_I_mat = np.broadcast_to(I_mat,
                        q.shape[:-2] + (q.shape[-1],)*2)
        # 断言 Q 的共轭转置与自身的乘积等于扩展后的单位矩阵
        assert_almost_equal(matmul(swapaxes(q, -1, -2).conj(), q), stack_I_mat)
        # 断言 R 的上三角部分与 R 相等
        assert_almost_equal(np.triu(r[..., :, :]), r)

        # mode == 'reduced' 模式下的 QR 分解
        q1, r1 = linalg.qr(a, mode='reduced')
        assert_(q1.dtype == a_dtype)
        assert_(r1.dtype == a_dtype)
        assert_(isinstance(q1, a_type))
        assert_(isinstance(r1, a_type))
        # 断言 Q 和 R 的形状与预期相符
        assert_(q1.shape[-2:] == (m, k))
        assert_(r1.shape[-2:] == (k, n))
        # 断言 QR 分解的乘积等于原始矩阵
        assert_almost_equal(matmul(q1, r1), a)
        # 创建单位矩阵
        I_mat = np.identity(q1.shape[-1])
        # 扩展单位矩阵以匹配 Q 的形状
        stack_I_mat = np.broadcast_to(I_mat,
                        q1.shape[:-2] + (q1.shape[-1],)*2)
        # 断言 Q 的共轭转置与自身的乘积等于扩展后的单位矩阵
        assert_almost_equal(matmul(swapaxes(q1, -1, -2).conj(), q1),
                            stack_I_mat)
        # 断言 R 的上三角部分与 R 相等
        assert_almost_equal(np.triu(r1[..., :, :]), r1)

        # mode == 'r' 模式下的 QR 分解
        r2 = linalg.qr(a, mode='r')
        assert_(r2.dtype == a_dtype)
        assert_(isinstance(r2, a_type))
        # 断言 R2 与 mode=='reduced' 下的 R1 相等
        assert_almost_equal(r2, r1)

    # 使用 pytest.mark.parametrize 进行参数化测试
    @pytest.mark.parametrize("size", [
        (3, 4), (4, 3), (4, 4),
        (3, 0), (0, 3)])
    @pytest.mark.parametrize("outer_size", [
        (2, 2), (2,), (2, 3, 4)])
    @pytest.mark.parametrize("dt", [
        np.single, np.double,
        np.csingle, np.cdouble])
    # 定义一个测试方法，测试不同尺寸、数据类型的输入
    def test_stacked_inputs(self, outer_size, size, dt):
        # 使用随机数生成器创建正态分布的 ndarray，并转换为指定数据类型
        rng = np.random.default_rng(123)
        A = rng.normal(size=outer_size + size).astype(dt)
        B = rng.normal(size=outer_size + size).astype(dt)
        # 调用 check_qr_stacked 方法，验证 QR 分解的行为
        self.check_qr_stacked(A)
        self.check_qr_stacked(A + 1.j*B)
class TestCholesky:

    @pytest.mark.parametrize(
        'shape', [(1, 1), (2, 2), (3, 3), (50, 50), (3, 10, 10)]
    )
    @pytest.mark.parametrize(
        'dtype', (np.float32, np.float64, np.complex64, np.complex128)
    )
    @pytest.mark.parametrize(
        'upper', [False, True])
    def test_basic_property(self, shape, dtype, upper):
        np.random.seed(1)
        a = np.random.randn(*shape)
        if np.issubdtype(dtype, np.complexfloating):
            a = a + 1j*np.random.randn(*shape)

        t = list(range(len(shape)))
        t[-2:] = -1, -2

        # Compute the product a^H * a or a * a^H based on transpose permutation
        a = np.matmul(a.transpose(t).conj(), a)
        # Convert 'a' to specified dtype
        a = np.asarray(a, dtype=dtype)

        # Compute Cholesky decomposition of 'a'
        c = np.linalg.cholesky(a, upper=upper)

        # Check A = L L^H or A = U^H U depending on 'upper' flag
        if upper:
            b = np.matmul(c.transpose(t).conj(), c)
        else:
            b = np.matmul(c, c.transpose(t).conj())

        # Set absolute tolerance based on matrix dimensions and dtype precision
        with np._no_nep50_warning():
            atol = 500 * a.shape[0] * np.finfo(dtype).eps
        # Assert the equality of matrices 'b' and 'a' within tolerance 'atol'
        assert_allclose(b, a, atol=atol, err_msg=f'{shape} {dtype}\n{a}\n{c}')

        # Check if diagonal elements of 'c' (L or U) are real and non-negative
        d = np.diagonal(c, axis1=-2, axis2=-1)
        assert_(np.all(np.isreal(d)))
        assert_(np.all(d >= 0))

    def test_0_size(self):
        # Define a subclass of np.ndarray with no additional functionality
        class ArraySubclass(np.ndarray):
            pass
        # Create a zero-sized array of integer dtype and cast it to ArraySubclass
        a = np.zeros((0, 1, 1), dtype=np.int_).view(ArraySubclass)
        # Compute Cholesky decomposition of 'a'
        res = linalg.cholesky(a)
        # Assert that shapes of 'a' and 'res' are equal
        assert_equal(a.shape, res.shape)
        # Assert that 'res' has dtype np.float64
        assert_(res.dtype.type is np.float64)
        # Assert that 'res' is an instance of np.ndarray for documentation purposes
        assert_(isinstance(res, np.ndarray))

        # Create a zero-sized array of complex dtype and cast it to ArraySubclass
        a = np.zeros((1, 0, 0), dtype=np.complex64).view(ArraySubclass)
        # Compute Cholesky decomposition of 'a'
        res = linalg.cholesky(a)
        # Assert that shapes of 'a' and 'res' are equal
        assert_equal(a.shape, res.shape)
        # Assert that 'res' has dtype np.complex64
        assert_(res.dtype.type is np.complex64)
        # Assert that 'res' is an instance of np.ndarray
        assert_(isinstance(res, np.ndarray))

    def test_upper_lower_arg(self):
        # Explicitly test the 'upper' argument of Cholesky decomposition
        a = np.array([[1+0j, 0-2j], [0+2j, 5+0j]])

        # Assert that Cholesky decomposition with 'upper=True' is equal to
        # the conjugate transpose of Cholesky decomposition with default 'upper=False'
        assert_equal(linalg.cholesky(a), linalg.cholesky(a, upper=False))

        # Assert that Cholesky decomposition with 'upper=True' is equal to
        # the conjugate transpose of default Cholesky decomposition
        assert_equal(
            linalg.cholesky(a, upper=True),
            linalg.cholesky(a).T.conj()
        )


class TestOuter:
    arr1 = np.arange(3)
    arr2 = np.arange(3)
    expected = np.array(
        [[0, 0, 0],
         [0, 1, 2],
         [0, 2, 4]]
    )

    # Assert that the outer product of arr1 and arr2 matches expected
    assert_array_equal(np.linalg.outer(arr1, arr2), expected)

    # Assert that ValueError with specific message is raised for invalid input arrays
    with assert_raises_regex(
        ValueError, "Input arrays must be one-dimensional"
    ):
        np.linalg.outer(arr1[:, np.newaxis], arr2)


def test_byteorder_check():
    # Check native byte order and assign appropriate character
    if sys.byteorder == 'little':
        native = '<'
    else:
        native = '>'
    # 遍历数据类型元组，包括 np.float32 和 np.float64
    for dtt in (np.float32, np.float64):
        # 创建一个 4x4 的单位矩阵，数据类型为 dtt
        arr = np.eye(4, dtype=dtt)
        # 创建一个新的数组 n_arr，其字节顺序与本地平台一致
        n_arr = arr.view(arr.dtype.newbyteorder(native))
        # 创建一个新的数组 sw_arr，其字节顺序为小端序并进行字节交换
        sw_arr = arr.view(arr.dtype.newbyteorder("S")).byteswap()
        # 断言数组 arr 的字节顺序为 '='，即本地平台的默认顺序
        assert_equal(arr.dtype.byteorder, '=')
        # 遍历线性代数函数元组，包括 linalg.inv, linalg.det, linalg.pinv
        for routine in (linalg.inv, linalg.det, linalg.pinv):
            # 普通调用，计算 routine(arr) 的结果
            res = routine(arr)
            # 使用本地字节顺序 n_arr 调用 routine 函数，断言结果与普通调用结果一致
            assert_array_equal(res, routine(n_arr))
            # 使用字节交换后的 sw_arr 调用 routine 函数，断言结果与普通调用结果一致
            assert_array_equal(res, routine(sw_arr))
@pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
# 标记为跳过测试，如果在 WebAssembly 环境下，原因是浮点错误在 wasm 中不起作用

def test_generalized_raise_multiloop():
    # 测试函数：测试通用异常抛出在多重循环中的情况
    # 应该会在 ufunc 内部循环的最后一个迭代之前抛出错误

    invertible = np.array([[1, 2], [3, 4]])
    non_invertible = np.array([[1, 1], [1, 1]])

    x = np.zeros([4, 4, 2, 2])[1::2]  # 创建一个4x4x2x2的零数组，并取出一部分
    x[...] = invertible  # 将可逆矩阵赋值给选定的部分
    x[0, 0] = non_invertible  # 将不可逆矩阵赋值给特定位置

    assert_raises(np.linalg.LinAlgError, np.linalg.inv, x)


def test_xerbla_override():
    # 测试函数：检查我们的 xerbla 是否成功链接
    # 如果没有成功链接，则会调用默认的 xerbla 程序，该程序会向 stdout 打印消息，
    # 并且根据 LAPACK 包可能会中止进程

    XERBLA_OK = 255

    try:
        pid = os.fork()
    except (OSError, AttributeError):
        # fork 失败，或者不在 POSIX 上运行
        pytest.skip("Not POSIX or fork failed.")

    if pid == 0:
        # 子进程；关闭输入输出文件句柄
        os.close(1)
        os.close(0)
        # 避免生成核心文件
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        # 下面的调用可能会中止进程
        try:
            np.linalg.lapack_lite.xerbla()
        except ValueError:
            pass
        except Exception:
            os._exit(os.EX_CONFIG)

        try:
            a = np.array([[1.]])
            np.linalg.lapack_lite.dorgqr(
                1, 1, 1, a,
                0,  # <- 无效的值
                a, a, 0, 0)
        except ValueError as e:
            if "DORGQR parameter number 5" in str(e):
                # 成功，重用错误码以标记成功，因为 FORTRAN STOP 返回的是成功
                os._exit(XERBLA_OK)

        # 没有中止，但我们的 xerbla 没有被链接
        os._exit(os.EX_CONFIG)
    else:
        # 父进程
        pid, status = os.wait()
        if os.WEXITSTATUS(status) != XERBLA_OK:
            pytest.skip('Numpy xerbla not linked in.')


@pytest.mark.skipif(IS_WASM, reason="Cannot start subprocess")
@pytest.mark.slow
def test_sdot_bug_8577():
    # 测试函数：回归测试，确保加载某些其他库不会导致 float32 线性代数产生错误结果
    #
    # 在 MacOS 上，存在 gh-8577 可能触发此问题，可能还有其他情况也会出现这个问题
    #
    # 在单独的进程中进行检查

    bad_libs = ['PyQt5.QtWidgets', 'IPython']

    template = textwrap.dedent("""
    import sys
    {before}
    try:
        import {bad_lib}
    except ImportError:
        sys.exit(0)
    {after}
    x = np.ones(2, dtype=np.float32)
    sys.exit(0 if np.allclose(x.dot(x), 2.0) else 1)
    """)
    # 对于每一个不良库 bad_lib，在模板代码中插入 import numpy as np 之前和之后的位置
    for bad_lib in bad_libs:
        # 构建完整的代码模板，将不良库 bad_lib 插入到 import numpy np 之前
        code = template.format(before="import numpy as np", after="",
                               bad_lib=bad_lib)
        # 使用 subprocess 调用 Python 解释器执行该代码
        subprocess.check_call([sys.executable, "-c", code])

        # 构建完整的代码模板，将不良库 bad_lib 插入到 import numpy np 之后
        code = template.format(after="import numpy as np", before="",
                               bad_lib=bad_lib)
        # 使用 subprocess 调用 Python 解释器执行该代码
        subprocess.check_call([sys.executable, "-c", code])
class TestMultiDot:

    def test_basic_function_with_three_arguments(self):
        # multi_dot with three arguments uses a fast hand coded algorithm to
        # determine the optimal order. Therefore test it separately.
        # 生成随机的 6x2 矩阵 A
        A = np.random.random((6, 2))
        # 生成随机的 2x6 矩阵 B
        B = np.random.random((2, 6))
        # 生成随机的 6x2 矩阵 C
        C = np.random.random((6, 2))

        # 断言 multi_dot([A, B, C]) 的结果与 A.dot(B).dot(C) 几乎相等
        assert_almost_equal(multi_dot([A, B, C]), A.dot(B).dot(C))
        # 断言 multi_dot([A, B, C]) 的结果与 np.dot(A, np.dot(B, C)) 几乎相等
        assert_almost_equal(multi_dot([A, B, C]), np.dot(A, np.dot(B, C)))

    def test_basic_function_with_two_arguments(self):
        # separate code path with two arguments
        # 生成随机的 6x2 矩阵 A
        A = np.random.random((6, 2))
        # 生成随机的 2x6 矩阵 B
        B = np.random.random((2, 6))

        # 断言 multi_dot([A, B]) 的结果与 A.dot(B) 几乎相等
        assert_almost_equal(multi_dot([A, B]), A.dot(B))
        # 断言 multi_dot([A, B]) 的结果与 np.dot(A, B) 几乎相等
        assert_almost_equal(multi_dot([A, B]), np.dot(A, B))

    def test_basic_function_with_dynamic_programming_optimization(self):
        # multi_dot with four or more arguments uses the dynamic programming
        # optimization and therefore deserve a separate
        # 生成随机的 6x2 矩阵 A
        A = np.random.random((6, 2))
        # 生成随机的 2x6 矩阵 B
        B = np.random.random((2, 6))
        # 生成随机的 6x2 矩阵 C
        C = np.random.random((6, 2))
        # 生成随机的 2x1 矩阵 D
        D = np.random.random((2, 1))

        # 断言 multi_dot([A, B, C, D]) 的结果与 A.dot(B).dot(C).dot(D) 几乎相等
        assert_almost_equal(multi_dot([A, B, C, D]), A.dot(B).dot(C).dot(D))

    def test_vector_as_first_argument(self):
        # The first argument can be 1-D
        # 生成随机的长度为 2 的一维数组 A1d
        A1d = np.random.random(2)  # 1-D
        # 生成随机的 2x6 矩阵 B
        B = np.random.random((2, 6))
        # 生成随机的 6x2 矩阵 C
        C = np.random.random((6, 2))
        # 生成随机的 2x2 矩阵 D
        D = np.random.random((2, 2))

        # 断言 multi_dot([A1d, B, C, D]) 的形状为 (2,)
        assert_equal(multi_dot([A1d, B, C, D]).shape, (2,))

    def test_vector_as_last_argument(self):
        # The last argument can be 1-D
        # 生成随机的 6x2 矩阵 A
        A = np.random.random((6, 2))
        # 生成随机的 2x6 矩阵 B
        B = np.random.random((2, 6))
        # 生成随机的 6x2 矩阵 C
        C = np.random.random((6, 2))
        # 生成随机的长度为 2 的一维数组 D1d
        D1d = np.random.random(2)  # 1-D

        # 断言 multi_dot([A, B, C, D1d]) 的形状为 (6,)
        assert_equal(multi_dot([A, B, C, D1d]).shape, (6,))

    def test_vector_as_first_and_last_argument(self):
        # The first and last arguments can be 1-D
        # 生成随机的长度为 2 的一维数组 A1d
        A1d = np.random.random(2)  # 1-D
        # 生成随机的 2x6 矩阵 B
        B = np.random.random((2, 6))
        # 生成随机的 6x2 矩阵 C
        C = np.random.random((6, 2))
        # 生成随机的长度为 2 的一维数组 D1d
        D1d = np.random.random(2)  # 1-D

        # 断言 multi_dot([A1d, B, C, D1d]) 的形状为 ()
        assert_equal(multi_dot([A1d, B, C, D1d]).shape, ())

    def test_three_arguments_and_out(self):
        # multi_dot with three arguments uses a fast hand coded algorithm to
        # determine the optimal order. Therefore test it separately.
        # 生成随机的 6x2 矩阵 A
        A = np.random.random((6, 2))
        # 生成随机的 2x6 矩阵 B
        B = np.random.random((2, 6))
        # 生成随机的 6x2 矩阵 C
        C = np.random.random((6, 2))

        # 生成全零的 6x2 矩阵 out
        out = np.zeros((6, 2))
        # 调用 multi_dot([A, B, C], out=out) 并将结果存储在 ret 中
        ret = multi_dot([A, B, C], out=out)
        # 断言 out 是 ret 的引用
        assert out is ret
        # 断言 out 的值几乎等于 A.dot(B).dot(C)
        assert_almost_equal(out, A.dot(B).dot(C))
        # 断言 out 的值几乎等于 np.dot(A, np.dot(B, C))
        assert_almost_equal(out, np.dot(A, np.dot(B, C)))
    def test_two_arguments_and_out(self):
        # 生成随机的 6x2 和 2x6 的矩阵 A 和 B
        A = np.random.random((6, 2))
        B = np.random.random((2, 6))
        # 创建一个 6x6 的全零矩阵 out
        out = np.zeros((6, 6))
        # 使用 multi_dot 函数计算 A 和 B 的乘积，并将结果保存到 out 中
        ret = multi_dot([A, B], out=out)
        # 检查 out 和返回值 ret 是同一个对象
        assert out is ret
        # 检查 out 的值接近于 A 点乘 B 的结果
        assert_almost_equal(out, A.dot(B))
        # 检查 out 的值接近于 np.dot(A, B) 的结果
        assert_almost_equal(out, np.dot(A, B))

    def test_dynamic_programming_optimization_and_out(self):
        # multi_dot 函数在传入四个或更多参数时会使用动态规划优化，
        # 因此需要单独进行测试
        A = np.random.random((6, 2))
        B = np.random.random((2, 6))
        C = np.random.random((6, 2))
        D = np.random.random((2, 1))
        # 创建一个 6x1 的全零矩阵 out
        out = np.zeros((6, 1))
        # 使用 multi_dot 函数计算 A、B、C 和 D 的乘积，并将结果保存到 out 中
        ret = multi_dot([A, B, C, D], out=out)
        # 检查 out 和返回值 ret 是同一个对象
        assert out is ret
        # 检查 out 的值接近于 A 点乘 B 点乘 C 点乘 D 的结果
        assert_almost_equal(out, A.dot(B).dot(C).dot(D))

    def test_dynamic_programming_logic(self):
        # 测试动态规划部分的逻辑
        # 这个测试直接来自于 Cormen 书中第 376 页。
        arrays = [np.random.random((30, 35)),
                  np.random.random((35, 15)),
                  np.random.random((15, 5)),
                  np.random.random((5, 10)),
                  np.random.random((10, 20)),
                  np.random.random((20, 25))]
        m_expected = np.array([[0., 15750., 7875., 9375., 11875., 15125.],
                               [0.,     0., 2625., 4375.,  7125., 10500.],
                               [0.,     0.,    0.,  750.,  2500.,  5375.],
                               [0.,     0.,    0.,    0.,  1000.,  3500.],
                               [0.,     0.,    0.,    0.,     0.,  5000.],
                               [0.,     0.,    0.,    0.,     0.,     0.]])
        s_expected = np.array([[0,  1,  1,  3,  3,  3],
                               [0,  0,  2,  3,  3,  3],
                               [0,  0,  0,  3,  3,  3],
                               [0,  0,  0,  0,  4,  5],
                               [0,  0,  0,  0,  0,  5],
                               [0,  0,  0,  0,  0,  0]], dtype=int)
        s_expected -= 1  # Cormen 使用基于 1 的索引，Python 不是。

        # 调用 _multi_dot_matrix_chain_order 函数计算 s 和 m
        s, m = _multi_dot_matrix_chain_order(arrays, return_costs=True)

        # 只有上三角部分（不包括对角线）是感兴趣的部分。
        assert_almost_equal(np.triu(s[:-1, 1:]),
                            np.triu(s_expected[:-1, 1:]))
        assert_almost_equal(np.triu(m), np.triu(m_expected))

    def test_too_few_input_arrays(self):
        # 检查当输入数组过少时是否会引发 ValueError 异常
        assert_raises(ValueError, multi_dot, [])
        assert_raises(ValueError, multi_dot, [np.random.random((3, 3))])
# 定义一个测试类 TestTensorinv，用于测试 tensorinv 函数的各种情况
class TestTensorinv:

    # 使用 pytest 的参数化装饰器，指定多组参数进行测试
    @pytest.mark.parametrize("arr, ind", [
        (np.ones((4, 6, 8, 2)), 2),   # 参数1：4维数组全为1，指数为2
        (np.ones((3, 3, 2)), 1),      # 参数2：3维数组全为1，指数为1
        ])
    # 测试非方阵处理函数，期望引发 LinAlgError 异常
    def test_non_square_handling(self, arr, ind):
        with assert_raises(LinAlgError):
            linalg.tensorinv(arr, ind=ind)

    # 使用 pytest 的参数化装饰器，指定多组参数进行形状测试
    @pytest.mark.parametrize("shape, ind", [
        # 文档字符串中的示例
        ((4, 6, 8, 3), 2),   # 参数1：形状为 (4, 6, 8, 3)，指数为2
        ((24, 8, 3), 1),     # 参数2：形状为 (24, 8, 3)，指数为1
        ])
    # 测试 tensorinv 函数返回的数组形状是否符合预期
    def test_tensorinv_shape(self, shape, ind):
        a = np.eye(24)
        a.shape = shape
        ainv = linalg.tensorinv(a=a, ind=ind)
        expected = a.shape[ind:] + a.shape[:ind]
        actual = ainv.shape
        assert_equal(actual, expected)

    # 使用 pytest 的参数化装饰器，指定多组参数进行指数限制测试
    @pytest.mark.parametrize("ind", [
        0, -2,   # 参数1：指数为0和-2
        ])
    # 测试 tensorinv 函数对于超出指数范围的值是否引发 ValueError 异常
    def test_tensorinv_ind_limit(self, ind):
        a = np.eye(24)
        a.shape = (4, 6, 8, 3)
        with assert_raises(ValueError):
            linalg.tensorinv(a=a, ind=ind)

    # 测试 tensorinv 函数的结果是否与文档字符串中的示例一致
    def test_tensorinv_result(self):
        # 模仿文档字符串中的示例
        a = np.eye(24)
        a.shape = (24, 8, 3)
        ainv = linalg.tensorinv(a, ind=1)
        b = np.ones(24)
        assert_allclose(np.tensordot(ainv, b, 1), np.linalg.tensorsolve(a, b))


# 定义一个测试类 TestTensorsolve，用于测试 tensorsolve 函数的各种情况
class TestTensorsolve:

    # 使用 pytest 的参数化装饰器，指定多组参数进行测试
    @pytest.mark.parametrize("a, axes", [
        (np.ones((4, 6, 8, 2)), None),   # 参数1：4维数组全为1，axes为None
        (np.ones((3, 3, 2)), (0, 2)),    # 参数2：3维数组全为1，axes为(0, 2)
        ])
    # 测试非方阵处理函数，期望引发 LinAlgError 异常
    def test_non_square_handling(self, a, axes):
        with assert_raises(LinAlgError):
            b = np.ones(a.shape[:2])
            linalg.tensorsolve(a, b, axes=axes)

    # 使用 pytest 的参数化装饰器，指定多组形状参数进行结果测试
    @pytest.mark.parametrize("shape",
        [(2, 3, 6), (3, 4, 4, 3), (0, 3, 3, 0)],
    )
    # 测试 tensorsolve 函数的结果是否符合预期
    def test_tensorsolve_result(self, shape):
        a = np.random.randn(*shape)
        b = np.ones(a.shape[:2])
        x = np.linalg.tensorsolve(a, b)
        assert_allclose(np.tensordot(a, x, axes=len(x.shape)), b)


# 定义一个测试函数 test_unsupported_commontype，测试 linalg 是否优雅处理不支持的类型
def test_unsupported_commontype():
    # 创建一个浮点16位数组，期望捕获 TypeError 异常，错误信息中包含 "unsupported in linalg"
    arr = np.array([[1, -2], [2, 5]], dtype='float16')
    with assert_raises_regex(TypeError, "unsupported in linalg"):
        linalg.cholesky(arr)


# 定义一个测试函数 test_blas64_dot，测试 64 位 BLAS 下的矩阵乘法
@pytest.mark.skip(reason="Bad memory reports lead to OOM in ci testing")
def test_blas64_dot():
    n = 2**32
    a = np.zeros([1, n], dtype=np.float32)
    b = np.ones([1, 1], dtype=np.float32)
    a[0,-1] = 1
    c = np.dot(b, a)
    assert_equal(c[0,-1], 1)


# 定义一个测试函数 test_blas64_geqrf_lwork_smoketest，烟雾测试 LAPACK geqrf 函数的 lwork 调用，使用 64 位整数
@pytest.mark.xfail(not HAS_LAPACK64,
                   reason="Numpy not compiled with 64-bit BLAS/LAPACK")
def test_blas64_geqrf_lwork_smoketest():
    # 使用 64 位浮点数类型
    dtype = np.float64
    lapack_routine = np.linalg.lapack_lite.dgeqrf

    m = 2**32 + 1
    n = 2**32 + 1
    lda = m

    # 虚拟数组，不需要正确大小，仅供 LAPACK 函数调用
    a = np.zeros([1, 1], dtype=dtype)
    # 创建一个dtype类型的大小为1的零数组，用于存储工作空间
    work = np.zeros([1], dtype=dtype)
    # 创建一个dtype类型的大小为1的零数组，用于存储tau参数
    tau = np.zeros([1], dtype=dtype)

    # 调用lapack_routine函数进行大小查询，获取计算结果
    results = lapack_routine(m, n, a, lda, tau, work, -1, 0)
    # 确保返回结果中的信息标志为0，表示调用成功
    assert_equal(results['info'], 0)
    # 确保返回结果中的m字段与输入参数m相等
    assert_equal(results['m'], m)
    # 确保返回结果中的n字段与输入参数m相等
    assert_equal(results['n'], m)

    # 将工作空间大小转换为整数，应该是一个合理大小的整数
    lwork = int(work.item())
    # 确保lwork在2**32和2**42之间，保证大小合理
    assert_(2**32 < lwork < 2**42)
def test_diagonal():
    # 在这里我们只测试选择的轴是否与数组 API 兼容（最后两个）。`diagonal` 的核心实现在 `test_multiarray.py` 中进行测试。
    x = np.arange(60).reshape((3, 4, 5))
    actual = np.linalg.diagonal(x)
    expected = np.array(
        [
            [0,  6, 12, 18],
            [20, 26, 32, 38],
            [40, 46, 52, 58],
        ]
    )
    assert_equal(actual, expected)


def test_trace():
    # 在这里我们只测试选择的轴是否与数组 API 兼容（最后两个）。`trace` 的核心实现在 `test_multiarray.py` 中进行测试。
    x = np.arange(60).reshape((3, 4, 5))
    actual = np.linalg.trace(x)
    expected = np.array([36, 116, 196])

    assert_equal(actual, expected)


def test_cross():
    x = np.arange(9).reshape((3, 3))
    actual = np.linalg.cross(x, x + 1)
    expected = np.array([
        [-1, 2, -1],
        [-1, 2, -1],
        [-1, 2, -1],
    ])

    assert_equal(actual, expected)

    # We test that lists are converted to arrays.
    u = [1, 2, 3]
    v = [4, 5, 6]
    actual = np.linalg.cross(u, v)
    expected = array([-3,  6, -3])

    assert_equal(actual, expected)

    with assert_raises_regex(
        ValueError,
        r"input arrays must be \(arrays of\) 3-dimensional vectors"
    ):
        x_2dim = x[:, 1:]
        np.linalg.cross(x_2dim, x_2dim)


def test_tensordot():
    # np.linalg.tensordot is just an alias for np.tensordot
    x = np.arange(6).reshape((2, 3))

    assert np.linalg.tensordot(x, x) == 55
    assert np.linalg.tensordot(x, x, axes=[(0, 1), (0, 1)]) == 55


def test_matmul():
    # np.linalg.matmul and np.matmul only differs in the number
    # of arguments in the signature
    x = np.arange(6).reshape((2, 3))
    actual = np.linalg.matmul(x, x.T)
    expected = np.array([[5, 14], [14, 50])

    assert_equal(actual, expected)


def test_matrix_transpose():
    x = np.arange(6).reshape((2, 3))
    actual = np.linalg.matrix_transpose(x)
    expected = x.T

    assert_equal(actual, expected)

    with assert_raises_regex(
        ValueError, "array must be at least 2-dimensional"
    ):
        np.linalg.matrix_transpose(x[:, 0])


def test_matrix_norm():
    x = np.arange(9).reshape((3, 3))
    actual = np.linalg.matrix_norm(x)

    assert_almost_equal(actual, np.float64(14.2828), double_decimal=3)

    actual = np.linalg.matrix_norm(x, keepdims=True)

    assert_almost_equal(actual, np.array([[14.2828]]), double_decimal=3)


def test_vector_norm():
    x = np.arange(9).reshape((3, 3))
    actual = np.linalg.vector_norm(x)

    assert_almost_equal(actual, np.float64(14.2828), double_decimal=3)

    actual = np.linalg.vector_norm(x, axis=0)

    assert_almost_equal(
        actual, np.array([6.7082, 8.124, 9.6436]), double_decimal=3
    )

    actual = np.linalg.vector_norm(x, keepdims=True)
    expected = np.full((1, 1), 14.2828, dtype='float64')
    assert_equal(actual.shape, expected.shape)
    # 使用断言来验证两个浮点数的近似相等性，允许误差到小数点后第三位
    assert_almost_equal(actual, expected, double_decimal=3)
```