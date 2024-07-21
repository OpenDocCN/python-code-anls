# `.\pytorch\test\torch_np\numpy_tests\linalg\test_linalg.py`

```
# Owner(s): ["module: dynamo"]

""" Test functions for linalg module

"""
# 导入必要的模块和库
import functools  # 导入 functools 模块，用于创建偏函数
import itertools  # 导入 itertools 模块，用于创建迭代器
import os  # 导入 os 模块，用于与操作系统交互
import subprocess  # 导入 subprocess 模块，用于执行系统命令
import sys  # 导入 sys 模块，提供对 Python 解释器的访问
import textwrap  # 导入 textwrap 模块，用于文本格式化和填充
import traceback  # 导入 traceback 模块，用于提取和格式化异常的堆栈跟踪信息

# 导入 unittest 中的各种装饰器和异常类
from unittest import expectedFailure as xfail, skipIf as skipif, SkipTest

# 导入 NumPy 库
import numpy

# 导入 pytest 库中的断言函数
import pytest

# 导入 NumPy 线性代数相关的函数和类
from numpy.linalg.linalg import _multi_dot_matrix_chain_order

# 导入 pytest 中的异常断言函数
from pytest import raises as assert_raises

# 导入 TorchDynamo 相关测试工具函数和类
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 实例化参数化测试
    parametrize,  # 参数化装饰器
    run_tests,  # 运行测试
    slowTest as slow,  # 标记为慢速测试的装饰器
    TEST_WITH_TORCHDYNAMO,  # TorchDynamo 的测试标志
    TestCase,  # 测试用例类
    xpassIfTorchDynamo,  # 当 TorchDynamo 可用时跳过测试的装饰器
)


# 如果要跟踪这些测试，我们应该使用 NumPy
# 如果在 eager 模式下测试，我们使用 torch._numpy
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy import (
        array,  # 创建数组
        asarray,  # 将输入转换为数组
        atleast_2d,  # 确保至少是二维数组
        cdouble,  # 复数类型的双精度数组
        csingle,  # 复数类型的单精度数组
        dot,  # 矩阵乘法
        double,  # 双精度浮点数类型
        identity,  # 单位矩阵
        inf,  # 无穷大
        linalg,  # 线性代数函数集
        matmul,  # 矩阵乘法
        single,  # 单精度浮点数类型
        swapaxes,  # 交换轴
    )
    from numpy.linalg import (
        LinAlgError,  # 线性代数错误
        matrix_power,  # 矩阵的整数幂
        matrix_rank,  # 矩阵的秩
        multi_dot,  # 多个矩阵乘法
        norm,  # 向量或矩阵的范数
    )
    from numpy.testing import (
        assert_,  # 断言为真
        assert_allclose,  # 断言所有元素近似相等
        assert_almost_equal,  # 断言两个对象近似相等
        assert_array_equal,  # 断言两个数组相等
        assert_equal,  # 断言两个对象相等
        suppress_warnings,  # 抑制警告
        #  assert_raises_regex, HAS_LAPACK64, IS_WASM
    )

else:
    import torch._numpy as np
    from torch._numpy import (
        array,  # 创建数组
        asarray,  # 将输入转换为数组
        atleast_2d,  # 确保至少是二维数组
        cdouble,  # 复数类型的双精度数组
        csingle,  # 复数类型的单精度数组
        dot,  # 矩阵乘法
        double,  # 双精度浮点数类型
        identity,  # 单位矩阵
        inf,  # 无穷大
        linalg,  # 线性代数函数集
        matmul,  # 矩阵乘法
        single,  # 单精度浮点数类型
        swapaxes,  # 交换轴
    )
    from torch._numpy.linalg import (
        LinAlgError,  # 线性代数错误
        matrix_power,  # 矩阵的整数幂
        matrix_rank,  # 矩阵的秩
        multi_dot,  # 多个矩阵乘法
        norm,  # 向量或矩阵的范数
    )
    from torch._numpy.testing import (
        assert_,  # 断言为真
        assert_allclose,  # 断言所有元素近似相等
        assert_almost_equal,  # 断言两个对象近似相等
        assert_array_equal,  # 断言两个数组相等
        assert_equal,  # 断言两个对象相等
        suppress_warnings,  # 抑制警告
        #  assert_raises_regex, HAS_LAPACK64, IS_WASM
    )


# 创建一个跳过装饰器，用于条件性跳过测试
skip = functools.partial(skipif, True)

# 初始化一些全局变量
IS_WASM = False
HAS_LAPACK64 = False


def consistent_subclass(out, in_):
    # 对于 ndarray 的子类输入，输出应该保持相同的子类类型
    # （非 ndarray 输入将被转换为 ndarray）。
    return type(out) is (type(in_) if isinstance(in_, np.ndarray) else np.ndarray)


# 保存原始的 assert_almost_equal 函数
old_assert_almost_equal = assert_almost_equal


def assert_almost_equal(a, b, single_decimal=6, double_decimal=12, **kw):
    # 根据输入数组的数据类型选择适当的小数位数
    if asarray(a).dtype.type in (single, csingle):
        decimal = single_decimal
    else:
        decimal = double_decimal
    # 调用原始的 assert_almost_equal 函数
    old_assert_almost_equal(a, b, decimal=decimal, **kw)


def get_real_dtype(dtype):
    # 根据输入的 dtype 返回对应的实数类型
    return {single: single, double: double, csingle: single, cdouble: double}[dtype]


def get_complex_dtype(dtype):
    # 根据输入的 dtype 返回对应的复数类型
    return {single: csingle, double: cdouble, csingle: csingle, cdouble: cdouble}[dtype]


def get_rtol(dtype):
    # 选择一个安全的相对容差值
    if dtype in (single, csingle):
        return 1e-5
    else:
        # 如果条件不满足，则返回一个非常接近零的小数，表示特定的错误条件
        return 1e-11
# 用于分类测试标签
all_tags = {
    "square",       # 方阵
    "nonsquare",    # 非方阵
    "hermitian",    # Hermite矩阵，与其他标签互斥
    "generalized",  # 广义矩阵
    "size-0",       # 大小为0的矩阵
    "strided",      # 可选的额外标签
}


class LinalgCase:
    def __init__(self, name, a, b, tags=None):
        """
        一个用于测试用例的参数捆绑，包括标识名称、操作数a和b，以及一组用于筛选测试的标签
        """
        if tags is None:
            tags = set()
        assert_(isinstance(name, str))  # 断言确保名称是字符串类型
        self.name = name
        self.a = a
        self.b = b
        self.tags = frozenset(tags)  # 使用不可变集合存储标签，防止标签被修改

    def check(self, do):
        """
        对该测试用例运行函数 `do`，扩展参数
        """
        do(self.a, self.b, tags=self.tags)

    def __repr__(self):
        return f"<LinalgCase: {self.name}>"


def apply_tag(tag, cases):
    """
    给每个测试用例（LinalgCase对象的列表）添加指定的标签（字符串）
    """
    assert tag in all_tags, "Invalid tag"  # 断言确保标签在预定义的标签集合中
    for case in cases:
        case.tags = case.tags | {tag}  # 向每个测试用例的标签集合中添加新标签
    return cases


#
# 基础测试用例
#

np.random.seed(1234)

CASES = []

# 方阵测试用例
CASES += apply_tag(
    "square",
    [
        LinalgCase(
            "single",
            array([[1.0, 2.0], [3.0, 4.0]], dtype=single),
            array([2.0, 1.0], dtype=single),
        ),
        LinalgCase(
            "double",
            array([[1.0, 2.0], [3.0, 4.0]], dtype=double),
            array([2.0, 1.0], dtype=double),
        ),
        LinalgCase(
            "double_2",
            array([[1.0, 2.0], [3.0, 4.0]], dtype=double),
            array([[2.0, 1.0, 4.0], [3.0, 4.0, 6.0]], dtype=double),
        ),
        LinalgCase(
            "csingle",
            array([[1.0 + 2j, 2 + 3j], [3 + 4j, 4 + 5j]], dtype=csingle),
            array([2.0 + 1j, 1.0 + 2j], dtype=csingle),
        ),
        LinalgCase(
            "cdouble",
            array([[1.0 + 2j, 2 + 3j], [3 + 4j, 4 + 5j]], dtype=cdouble),
            array([2.0 + 1j, 1.0 + 2j], dtype=cdouble),
        ),
        LinalgCase(
            "cdouble_2",
            array([[1.0 + 2j, 2 + 3j], [3 + 4j, 4 + 5j]], dtype=cdouble),
            array(
                [[2.0 + 1j, 1.0 + 2j, 1 + 3j], [1 - 2j, 1 - 3j, 1 - 6j]], dtype=cdouble
            ),
        ),
        LinalgCase(
            "0x0",
            np.empty((0, 0), dtype=double),
            np.empty((0,), dtype=double),
            tags={"size-0"},
        ),
        LinalgCase("8x8", np.random.rand(8, 8), np.random.rand(8)),
        LinalgCase("1x1", np.random.rand(1, 1), np.random.rand(1)),
        LinalgCase("nonarray", [[1, 2], [3, 4]], [2, 1]),
    ],
)

# 非方阵测试用例
CASES += apply_tag(
    "nonsquare",
    [
        LinalgCase(
            "single_nsq_1",  # 第一个线性代数测试案例，单精度，形状为 (2, 3)
            array([[1.0, 2.0, 3.0], [3.0, 4.0, 6.0]], dtype=single),  # 二维数组，单精度
            array([2.0, 1.0], dtype=single),  # 一维数组，单精度
        ),
        LinalgCase(
            "single_nsq_2",  # 第二个线性代数测试案例，单精度，形状为 (3, 2)
            array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=single),  # 二维数组，单精度
            array([2.0, 1.0, 3.0], dtype=single),  # 一维数组，单精度
        ),
        LinalgCase(
            "double_nsq_1",  # 第三个线性代数测试案例，双精度，形状为 (2, 3)
            array([[1.0, 2.0, 3.0], [3.0, 4.0, 6.0]], dtype=double),  # 二维数组，双精度
            array([2.0, 1.0], dtype=double),  # 一维数组，双精度
        ),
        LinalgCase(
            "double_nsq_2",  # 第四个线性代数测试案例，双精度，形状为 (3, 2)
            array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=double),  # 二维数组，双精度
            array([2.0, 1.0, 3.0], dtype=double),  # 一维数组，双精度
        ),
        LinalgCase(
            "csingle_nsq_1",  # 第五个线性代数测试案例，单精度复数，形状为 (2, 3)
            array(
                [[1.0 + 1j, 2.0 + 2j, 3.0 - 3j], [3.0 - 5j, 4.0 + 9j, 6.0 + 2j]],
                dtype=csingle,
            ),  # 二维数组，单精度复数
            array([2.0 + 1j, 1.0 + 2j], dtype=csingle),  # 一维数组，单精度复数
        ),
        LinalgCase(
            "csingle_nsq_2",  # 第六个线性代数测试案例，单精度复数，形状为 (3, 2)
            array(
                [[1.0 + 1j, 2.0 + 2j], [3.0 - 3j, 4.0 - 9j], [5.0 - 4j, 6.0 + 8j]],
                dtype=csingle,
            ),  # 二维数组，单精度复数
            array([2.0 + 1j, 1.0 + 2j, 3.0 - 3j], dtype=csingle),  # 一维数组，单精度复数
        ),
        LinalgCase(
            "cdouble_nsq_1",  # 第七个线性代数测试案例，双精度复数，形状为 (2, 3)
            array(
                [[1.0 + 1j, 2.0 + 2j, 3.0 - 3j], [3.0 - 5j, 4.0 + 9j, 6.0 + 2j]],
                dtype=cdouble,
            ),  # 二维数组，双精度复数
            array([2.0 + 1j, 1.0 + 2j], dtype=cdouble),  # 一维数组，双精度复数
        ),
        LinalgCase(
            "cdouble_nsq_2",  # 第八个线性代数测试案例，双精度复数，形状为 (3, 2)
            array(
                [[1.0 + 1j, 2.0 + 2j], [3.0 - 3j, 4.0 - 9j], [5.0 - 4j, 6.0 + 8j]],
                dtype=cdouble,
            ),  # 二维数组，双精度复数
            array(
                [[2.0 + 1j, 1.0 + 2j], [1 - 1j, 2 - 2j], [1 - 1j, 2 - 2j]],
                dtype=cdouble,
            ),  # 二维数组，双精度复数
        ),
        LinalgCase(
            "cdouble_nsq_1_2",  # 第九个线性代数测试案例，双精度复数，形状为 (2, 3)
            array(
                [[1.0 + 1j, 2.0 + 2j, 3.0 - 3j], [3.0 - 5j, 4.0 + 9j, 6.0 + 2j]],
                dtype=cdouble,
            ),  # 二维数组，双精度复数
            array([[2.0 + 1j, 1.0 + 2j], [1 - 1j, 2 - 2j]], dtype=cdouble),  # 二维数组，双精度复数
        ),
        LinalgCase(
            "cdouble_nsq_2_2",  # 第十个线性代数测试案例，双精度复数，形状为 (3, 2)
            array(
                [[1.0 + 1j, 2.0 + 2j], [3.0 - 3j, 4.0 - 9j], [5.0 - 4j, 6.0 + 8j]],
                dtype=cdouble,
            ),  # 二维数组，双精度复数
            array(
                [[2.0 + 1j, 1.0 + 2j], [1 - 1j, 2 - 2j], [1 - 1j, 2 - 2j]],
                dtype=cdouble,
            ),  # 二维数组，双精度复数
        ),
        LinalgCase("8x11", np.random.rand(8, 11), np.random.rand(8)),  # 8行11列的随机数组
        LinalgCase("1x5", np.random.rand(1, 5), np.random.rand(1)),  # 1行5列的随机数组
        LinalgCase("5x1", np.random.rand(5, 1), np.random.rand(5)),  # 5行1列的随机数组
        LinalgCase("0x4", np.random.rand(0, 4), np.random.rand(0), tags={"size-0"}),  # 0行4列的随机数组，带有标签 size-0
        LinalgCase("4x0", np.random.rand(4, 0), np.random.rand(4), tags={"size-0"}),  # 4行0列的随机数组，带有标签 size-0
    ],
# 将"hermitian"标签应用到一系列线性代数测试用例上，并将它们添加到全局变量CASES中
CASES += apply_tag(
    "hermitian",
    [
        # 创建单精度矩阵测试用例，数据为[[1.0, 2.0], [2.0, 1.0]]
        LinalgCase("hsingle", array([[1.0, 2.0], [2.0, 1.0]], dtype=single), None),
        # 创建双精度矩阵测试用例，数据为[[1.0, 2.0], [2.0, 1.0]]
        LinalgCase("hdouble", array([[1.0, 2.0], [2.0, 1.0]], dtype=double), None),
        # 创建复数单精度矩阵测试用例，数据为[[1.0, 2+3j], [2-3j, 1]]
        LinalgCase(
            "hcsingle", array([[1.0, 2 + 3j], [2 - 3j, 1]], dtype=csingle), None
        ),
        # 创建复数双精度矩阵测试用例，数据为[[1.0, 2+3j], [2-3j, 1]]
        LinalgCase(
            "hcdouble", array([[1.0, 2 + 3j], [2 - 3j, 1]], dtype=cdouble), None
        ),
        # 创建空矩阵测试用例，大小为(0, 0)，数据类型为双精度
        LinalgCase("hempty", np.empty((0, 0), dtype=double), None, tags={"size-0"}),
        # 创建非数组矩阵测试用例，数据为[[1, 2], [2, 1]]
        LinalgCase("hnonarray", [[1, 2], [2, 1]], None),
        # 创建矩阵测试用例，数据为[[1.0, 2.0], [2.0, 1.0]]
        LinalgCase("matrix_b_only", array([[1.0, 2.0], [2.0, 1.0]]), None),
        # 创建1x1随机矩阵测试用例
        LinalgCase("hmatrix_1x1", np.random.rand(1, 1), None),
    ],
)

# 定义一个函数用于生成扩展的测试用例，基于已有的CASES全局变量中的测试用例
def _make_generalized_cases():
    new_cases = []

    # 遍历现有的测试用例
    for case in CASES:
        # 如果case.a不是ndarray类型，则跳过当前迭代
        if not isinstance(case.a, np.ndarray):
            continue

        # 将case.a扩展为包含原始矩阵、原始矩阵的两倍和三倍的数组
        a = np.stack([case.a, 2 * case.a, 3 * case.a])
        # 根据case.b是否为None，设置b变量为None或者原始矩阵、原始矩阵的七倍和六倍的数组
        if case.b is None:
            b = None
        else:
            b = np.stack([case.b, 7 * case.b, 6 * case.b])
        # 创建新的LinalgCase对象，标签为case.tags和{"generalized"}的并集
        new_case = LinalgCase(
            case.name + "_tile3", a, b, tags=case.tags | {"generalized"}
        )
        # 将新的测试用例添加到new_cases列表中
        new_cases.append(new_case)

        # 将case.a和case.b扩展为3x2数组
        a = np.array([case.a] * 2 * 3).reshape((3, 2) + case.a.shape)
        if case.b is None:
            b = None
        else:
            b = np.array([case.b] * 2 * 3).reshape((3, 2) + case.b.shape)
        # 创建新的LinalgCase对象，标签为case.tags和{"generalized"}的并集
        new_case = LinalgCase(
            case.name + "_tile213", a, b, tags=case.tags | {"generalized"}
        )
        # 将新的测试用例添加到new_cases列表中
        new_cases.append(new_case)

    return new_cases

# 将_make_generalized_cases函数生成的扩展测试用例添加到CASES全局变量中
CASES += _make_generalized_cases()

# 定义一个线性代数测试类LinalgTestCase
class LinalgTestCase:
    # 类变量TEST_CASES初始化为全局变量CASES
    TEST_CASES = CASES

    # 定义一个方法，用于在指定条件下运行测试用例
    def check_cases(self, require=None, exclude=None):
        """
        Run func on each of the cases with all of the tags in require, and none
        of the tags in exclude
        """
        # 如果require为None，则将其设为空集合
        if require is None:
            require = set()
        # 如果exclude为None，则将其设为空集合
        if exclude is None:
            exclude = set()
        
        # 遍历所有的测试用例
        for case in self.TEST_CASES:
            # 根据require和exclude条件过滤测试用例
            if case.tags & require != require:
                continue
            if case.tags & exclude:
                continue

            try:
                # 调用当前实例的do方法检查测试用例case
                case.check(self.do)
            except Exception as e:
                # 捕获异常并抛出详细错误信息
                msg = f"In test case: {case!r}\n\n"
                msg += traceback.format_exc()
                raise AssertionError(msg) from e

# 定义一个继承自LinalgTestCase的子类LinalgSquareTestCase，用于测试方阵相关的测试用例
class LinalgSquareTestCase(LinalgTestCase):
    # 测试仅包含方阵的测试用例
    def test_sq_cases(self):
        self.check_cases(require={"square"}, exclude={"generalized", "size-0"})

    # 测试仅包含空方阵的测试用例
    def test_empty_sq_cases(self):
        self.check_cases(require={"square", "size-0"}, exclude={"generalized"})

# 定义一个继承自LinalgTestCase的子类LinalgNonsquareTestCase，用于测试非方阵相关的测试用例
class LinalgNonsquareTestCase(LinalgTestCase):
    # 测试仅包含非方阵的测试用例
    def test_nonsq_cases(self):
        self.check_cases(require={"nonsquare"}, exclude={"generalized", "size-0"})
    # 定义一个测试方法，用于检查空的非方阵案例
    def test_empty_nonsq_cases(self):
        # 调用自定义的 check_cases 方法，传入要求包含 "nonsquare" 和 "size-0" 的条件，
        # 并排除 "generalized" 条件
        self.check_cases(require={"nonsquare", "size-0"}, exclude={"generalized"})
class HermitianTestCase(LinalgTestCase):
    # HermitianTestCase 类继承自 LinalgTestCase，用于测试 Hermitian 矩阵相关的测试用例
    def test_herm_cases(self):
        # 调用 check_cases 方法进行测试，要求测试用例包含 "hermitian" 特性，但不包括 "generalized" 和 "size-0"
        self.check_cases(require={"hermitian"}, exclude={"generalized", "size-0"})

    def test_empty_herm_cases(self):
        # 测试空的 Hermitian 矩阵相关用例，要求包含 "hermitian" 和 "size-0" 特性，但不包括 "generalized"
        self.check_cases(require={"hermitian", "size-0"}, exclude={"generalized"})


class LinalgGeneralizedSquareTestCase(LinalgTestCase):
    @slow
    def test_generalized_sq_cases(self):
        # 一般化的方阵测试用例，要求包含 "generalized" 和 "square" 特性，但不包括 "size-0"
        self.check_cases(require={"generalized", "square"}, exclude={"size-0"})

    @slow
    def test_generalized_empty_sq_cases(self):
        # 空的一般化方阵测试用例，要求包含 "generalized"、"square" 和 "size-0" 特性
        self.check_cases(require={"generalized", "square", "size-0"})


class LinalgGeneralizedNonsquareTestCase(LinalgTestCase):
    @slow
    def test_generalized_nonsq_cases(self):
        # 一般化的非方阵测试用例，要求包含 "generalized" 和 "nonsquare" 特性，但不包括 "size-0"
        self.check_cases(require={"generalized", "nonsquare"}, exclude={"size-0"})

    @slow
    def test_generalized_empty_nonsq_cases(self):
        # 空的一般化非方阵测试用例，要求包含 "generalized"、"nonsquare" 和 "size-0" 特性，但不包括 "none"
        self.check_cases(require={"generalized", "nonsquare", "size-0"})


class HermitianGeneralizedTestCase(LinalgTestCase):
    @slow
    def test_generalized_herm_cases(self):
        # 一般化的 Hermitian 矩阵测试用例，要求包含 "generalized" 和 "hermitian" 特性，但不包括 "size-0"
        self.check_cases(require={"generalized", "hermitian"}, exclude={"size-0"})

    @slow
    def test_generalized_empty_herm_cases(self):
        # 空的一般化的 Hermitian 矩阵测试用例，要求包含 "generalized"、"hermitian" 和 "size-0" 特性，但不包括 "none"
        self.check_cases(
            require={"generalized", "hermitian", "size-0"}, exclude={"none"}
        )


def dot_generalized(a, b):
    # 将输入数组 a 转换为 ndarray
    a = asarray(a)
    # 如果 a 的维度大于等于 3
    if a.ndim >= 3:
        # 如果 a 和 b 的维度相同
        if a.ndim == b.ndim:
            # 矩阵乘以矩阵，确定新的形状
            new_shape = a.shape[:-1] + b.shape[-1:]
        elif a.ndim == b.ndim + 1:
            # 矩阵乘以向量，确定新的形状
            new_shape = a.shape[:-1]
        else:
            # 抛出错误，未实现的操作
            raise ValueError("Not implemented...")
        # 创建新的数组 r，用于存储计算结果，类型为 a 和 b 的公共类型
        r = np.empty(new_shape, dtype=np.common_type(a, b))
        # 使用 itertools.product 生成 a.shape[:-2] 中各维度的组合
        for c in itertools.product(*map(range, a.shape[:-2])):
            # 计算 dot(a[c], b[c]) 并存入 r 的对应位置
            r[c] = dot(a[c], b[c])
        return r
    else:
        # 维度不满足条件时，直接计算 dot(a, b)
        return dot(a, b)


def identity_like_generalized(a):
    # 将输入数组 a 转换为 ndarray
    a = asarray(a)
    # 如果 a 的维度大于等于 3
    if a.ndim >= 3:
        # 创建与 a 形状相同的单位矩阵，类型与 a 相同
        r = np.empty(a.shape, dtype=a.dtype)
        r[...] = identity(a.shape[-2])
        return r
    else:
        # 创建与 a 形状相同的单位矩阵，形状为 a.shape[0]
        return identity(a.shape[0])


class SolveCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):
    # 用于测试求解器与矩阵的互动情况，与 TestSolve 分开以便使用矩阵测试
    def do(self, a, b, tags):
        # 求解线性方程组 a * x = b
        x = linalg.solve(a, b)
        # 断言 b 与 a * x 的近似相等
        assert_almost_equal(b, dot_generalized(a, x))
        # 断言 x 是 b 的一致子类
        assert_(consistent_subclass(x, b))


@instantiate_parametrized_tests
class TestSolve(SolveCases, TestCase):
    @parametrize("dtype", [single, double, csingle, cdouble])
    def test_types(self, dtype):
        # 创建指定数据类型的测试矩阵 x
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        # 断言解 x 的数据类型与指定的 dtype 相同
        assert_equal(linalg.solve(x, x).dtype, dtype)

    @skip(reason="subclass")
    def test_0_size(self):
        # 定义一个继承自 np.ndarray 的子类 ArraySubclass
        class ArraySubclass(np.ndarray):
            pass

        # 测试 0x0 矩阵系统
        a = np.arange(8).reshape(2, 2, 2)
        b = np.arange(6).reshape(1, 2, 3).view(ArraySubclass)

        # 求解线性方程组 linalg.solve(a, b)，其中 a 是非 0x0 子集，b 的第一维度为 0
        expected = linalg.solve(a, b)[:, 0:0, :]
        result = linalg.solve(a[:, 0:0, 0:0], b[:, 0:0, :])
        assert_array_equal(result, expected)
        assert_(isinstance(result, ArraySubclass))

        # 测试非方阵和仅 b 维度为 0 时的错误
        assert_raises(linalg.LinAlgError, linalg.solve, a[:, 0:0, 0:1], b)
        assert_raises(ValueError, linalg.solve, a, b[:, 0:0, :])

        # 测试广播错误
        b = np.arange(6).reshape(1, 3, 2)  # 广播错误
        assert_raises(ValueError, linalg.solve, a, b)
        assert_raises(ValueError, linalg.solve, a[0:0], b[0:0])

        # 测试在 0x0 矩阵中的零 "单一方程"
        b = np.arange(2).reshape(1, 2).view(ArraySubclass)
        expected = linalg.solve(a, b)[:, 0:0]
        result = linalg.solve(a[:, 0:0, 0:0], b[:, 0:0])
        assert_array_equal(result, expected)
        assert_(isinstance(result, ArraySubclass))

        b = np.arange(3).reshape(1, 3)
        assert_raises(ValueError, linalg.solve, a, b)
        assert_raises(ValueError, linalg.solve, a[0:0], b[0:0])
        assert_raises(ValueError, linalg.solve, a[:, 0:0, 0:0], b)

    @skip(reason="subclass")
    def test_0_size_k(self):
        # 测试零多方程 (K=0) 情况
        class ArraySubclass(np.ndarray):
            pass

        a = np.arange(4).reshape(1, 2, 2)
        b = np.arange(6).reshape(3, 2, 1).view(ArraySubclass)

        # 求解线性方程组 linalg.solve(a, b)，其中 b 的第三维度为 0
        expected = linalg.solve(a, b)[:, :, 0:0]
        result = linalg.solve(a, b[:, :, 0:0])
        assert_array_equal(result, expected)
        assert_(isinstance(result, ArraySubclass))

        # 测试两者均为零的情况
        expected = linalg.solve(a, b)[:, 0:0, 0:0]
        result = linalg.solve(a[:, 0:0, 0:0], b[:, 0:0, 0:0])
        assert_array_equal(result, expected)
        assert_(isinstance(result, ArraySubclass))
class InvCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):
    # 定义一个测试用例类，继承自LinalgSquareTestCase和LinalgGeneralizedSquareTestCase
    def do(self, a, b, tags):
        # 执行测试方法，计算矩阵a的逆矩阵
        a_inv = linalg.inv(a)
        # 断言：验证a与其逆矩阵相乘后是否接近单位矩阵
        assert_almost_equal(dot_generalized(a, a_inv), identity_like_generalized(a))
        # 断言：验证a的逆矩阵是否与a具有一致的子类
        assert_(consistent_subclass(a_inv, a))


@instantiate_parametrized_tests
# 实例化参数化测试类装饰器
class TestInv(InvCases, TestCase):
    # 测试逆矩阵的主测试类，继承自InvCases和TestCase
    @parametrize("dtype", [single, double, csingle, cdouble])
    # 参数化装饰器，对不同的数据类型进行测试
    def test_types(self, dtype):
        # 创建一个2x2的数组x，数据类型为dtype
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        # 断言：验证x的逆矩阵的数据类型是否为dtype
        assert_equal(linalg.inv(x).dtype, dtype)

    @skip(reason="subclass")
    # 跳过装饰器，理由是"subclass"
    def test_0_size(self):
        # 测试处理所有类型的大小为0的数组
        class ArraySubclass(np.ndarray):
            pass

        # 创建一个形状为(0,1,1)的整型数组a，视图类型为ArraySubclass
        a = np.zeros((0, 1, 1), dtype=np.int_).view(ArraySubclass)
        # 计算a的逆矩阵
        res = linalg.inv(a)
        # 断言：验证res的数据类型是否为np.float64
        assert_(res.dtype.type is np.float64)
        # 断言：验证a和res的形状是否相同
        assert_equal(a.shape, res.shape)
        # 断言：验证res是否是ArraySubclass的实例
        assert_(isinstance(res, ArraySubclass))

        # 创建一个形状为(0,0)的复数数组a，视图类型为ArraySubclass
        a = np.zeros((0, 0), dtype=np.complex64).view(ArraySubclass)
        # 计算a的逆矩阵
        res = linalg.inv(a)
        # 断言：验证res的数据类型是否为np.complex64
        assert_(res.dtype.type is np.complex64)
        # 断言：验证a和res的形状是否相同
        assert_equal(a.shape, res.shape)
        # 断言：验证res是否是ArraySubclass的实例
        assert_(isinstance(res, ArraySubclass))


class EigvalsCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):
    # 定义特征值测试用例类，继承自LinalgSquareTestCase和LinalgGeneralizedSquareTestCase
    def do(self, a, b, tags):
        # 计算矩阵a的特征值
        ev = linalg.eigvals(a)
        # 计算矩阵a的特征值和特征向量
        evalues, evectors = linalg.eig(a)
        # 断言：验证计算得到的特征值ev是否与特征值矩阵evalues相近
        assert_almost_equal(ev, evalues)


@instantiate_parametrized_tests
# 实例化参数化测试类装饰器
class TestEigvals(EigvalsCases, TestCase):
    # 特征值测试的主测试类，继承自EigvalsCases和TestCase
    @parametrize("dtype", [single, double, csingle, cdouble])
    # 参数化装饰器，对不同的数据类型进行测试
    def test_types(self, dtype):
        # 创建一个2x2的数组x，数据类型为dtype
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        # 断言：验证x的特征值的数据类型是否为dtype
        assert_equal(linalg.eigvals(x).dtype, dtype)
        # 创建一个2x2的数组x，数据类型为dtype
        x = np.array([[1, 0.5], [-1, 1]], dtype=dtype)
        # 断言：验证x的特征值的数据类型是否为get_complex_dtype(dtype)返回的复数类型
        assert_equal(linalg.eigvals(x).dtype, get_complex_dtype(dtype))

    @skip(reason="subclass")
    # 跳过装饰器，理由是"subclass"
    def test_0_size(self):
        # 测试处理所有类型的大小为0的数组
        class ArraySubclass(np.ndarray):
            pass

        # 创建一个形状为(0,1,1)的整型数组a，视图类型为ArraySubclass
        a = np.zeros((0, 1, 1), dtype=np.int_).view(ArraySubclass)
        # 计算a的特征值
        res = linalg.eigvals(a)
        # 断言：验证res的数据类型是否为np.float64
        assert_(res.dtype.type is np.float64)
        # 断言：验证res的形状为(0, 1)
        assert_equal((0, 1), res.shape)
        # 文档用的注释：这可能有意义修改
        assert_(isinstance(res, np.ndarray))

        # 创建一个形状为(0,0)的复数数组a，视图类型为ArraySubclass
        a = np.zeros((0, 0), dtype=np.complex64).view(ArraySubclass)
        # 计算a的特征值
        res = linalg.eigvals(a)
        # 断言：验证res的数据类型是否为np.complex64
        assert_(res.dtype.type is np.complex64)
        # 断言：验证res的形状为(0,)
        assert_equal((0,), res.shape)
        # 文档用的注释：这可能有意义修改
        assert_(isinstance(res, np.ndarray))


class EigCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):
    # 定义特征向量测试用例类，继承自LinalgSquareTestCase和LinalgGeneralizedSquareTestCase
    def do(self, a, b, tags):
        # 计算矩阵a的特征值和特征向量
        evalues, evectors = linalg.eig(a)
        # 断言：验证计算得到的特征向量与特征值的乘积是否接近a的一般化乘积
        assert_allclose(
            dot_generalized(a, evectors),
            np.asarray(evectors) * np.asarray(evalues)[..., None, :],
            rtol=get_rtol(evalues.dtype),
        )
        # 断言：验证evectors是否与a具有一致的子类
        assert_(consistent_subclass(evectors, a))


@instantiate_parametrized_tests
# 实例化参数化测试类装饰器
# 定义一个测试类，继承自EigCases和TestCase，用于测试特定情况下的特征值和特征向量
class TestEig(EigCases, TestCase):

    # 参数化测试方法，测试不同的数据类型(dtype)
    @parametrize("dtype", [single, double, csingle, cdouble])
    def test_types(self, dtype):
        # 创建一个二维数组x，指定数据类型为dtype
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        # 计算x的特征值w和特征向量v
        w, v = np.linalg.eig(x)
        # 断言特征值w的数据类型为dtype
        assert_equal(w.dtype, dtype)
        # 断言特征向量v的数据类型为dtype
        assert_equal(v.dtype, dtype)

        # 创建另一个二维数组x，指定数据类型为dtype
        x = np.array([[1, 0.5], [-1, 1]], dtype=dtype)
        # 计算x的特征值w和特征向量v
        w, v = np.linalg.eig(x)
        # 断言特征值w的数据类型为dtype对应的复数类型
        assert_equal(w.dtype, get_complex_dtype(dtype))
        # 断言特征向量v的数据类型为dtype对应的复数类型
        assert_equal(v.dtype, get_complex_dtype(dtype))

    # 跳过测试的方法，理由是“subclass”
    @skip(reason="subclass")
    def test_0_size(self):
        # 检查各种0大小的数组是否正常工作
        # 定义一个数组子类ArraySubclass，继承自np.ndarray
        class ArraySubclass(np.ndarray):
            pass

        # 创建一个0大小的整数类型的二维数组a，视图转换为ArraySubclass类型
        a = np.zeros((0, 1, 1), dtype=np.int_).view(ArraySubclass)
        # 计算a的特征值res和特征向量res_v
        res, res_v = linalg.eig(a)
        # 断言特征向量res_v的数据类型为np.float64
        assert_(res_v.dtype.type is np.float64)
        # 断言特征值res的数据类型为np.float64
        assert_(res.dtype.type is np.float64)
        # 断言数组a的形状与特征向量res_v的形状相等
        assert_equal(a.shape, res_v.shape)
        # 断言特征值res的形状为(0, 1)
        assert_equal((0, 1), res.shape)
        # 这仅仅是文档说明，可能需要修改：
        # 断言a是np.ndarray的实例
        assert_(isinstance(a, np.ndarray))

        # 创建一个0大小的复数类型的二维数组a，视图转换为ArraySubclass类型
        a = np.zeros((0, 0), dtype=np.complex64).view(ArraySubclass)
        # 计算a的特征值res和特征向量res_v
        res, res_v = linalg.eig(a)
        # 断言特征向量res_v的数据类型为np.complex64
        assert_(res_v.dtype.type is np.complex64)
        # 断言特征值res的数据类型为np.complex64
        assert_(res.dtype.type is np.complex64)
        # 断言数组a的形状与特征向量res_v的形状相等
        assert_equal(a.shape, res_v.shape)
        # 断言特征值res的形状为(0,)
        assert_equal((0,), res.shape)
        # 这仅仅是文档说明，可能需要修改：
        # 断言a是np.ndarray的实例
        assert_(isinstance(a, np.ndarray))


# 实例化参数化测试类的测试方法
@instantiate_parametrized_tests
class SVDBaseTests:
    # 声明属性hermitian为False

    # 参数化测试方法，测试不同的数据类型(dtype)
    @parametrize("dtype", [single, double, csingle, cdouble])
    def test_types(self, dtype):
        # 创建一个二维数组x，指定数据类型为dtype
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        # 计算x的奇异值分解u, s, vh
        u, s, vh = linalg.svd(x)
        # 断言矩阵u的数据类型为dtype
        assert_equal(u.dtype, dtype)
        # 断言奇异值s的数据类型为dtype对应的实数类型
        assert_equal(s.dtype, get_real_dtype(dtype))
        # 断言矩阵vh的数据类型为dtype
        assert_equal(vh.dtype, dtype)
        # 计算x的奇异值分解，只计算奇异值s，不计算u和vh，要求是否共轭自适应
        s = linalg.svd(x, compute_uv=False, hermitian=self.hermitian)
        # 断言奇异值s的数据类型为dtype对应的实数类型
        assert_equal(s.dtype, get_real_dtype(dtype))


# 继承自LinalgSquareTestCase和LinalgGeneralizedSquareTestCase的测试类SVDCases
class SVDCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):

    # 执行方法，接受a、b、tags三个参数
    def do(self, a, b, tags):
        # 计算a的奇异值分解，不计算u和vt
        u, s, vt = linalg.svd(a, False)
        # 断言a与u*s*vt的广义乘积近似相等
        assert_allclose(
            a,
            dot_generalized(
                np.asarray(u) * np.asarray(s)[..., None, :], np.asarray(vt)
            ),
            rtol=get_rtol(u.dtype),
        )
        # 断言u是a的一致子类
        assert_(consistent_subclass(u, a))
        # 断言vt是a的一致子类
        assert_(consistent_subclass(vt, a))


# 继承自SVDCases、SVDBaseTests和TestCase的测试类TestSVD
class TestSVD(SVDCases, SVDBaseTests, TestCase):

    # 测试空输入应该在u或vh中放置一个单位矩阵
    def test_empty_identity(self):
        # 创建一个空的4x0数组x
        x = np.empty((4, 0))
        # 计算x的奇异值分解u, s, vh，要求计算u和vh，是否共轭自适应
        u, s, vh = linalg.svd(x, compute_uv=True, hermitian=self.hermitian)
        # 断言矩阵u的形状为(4, 4)
        assert_equal(u.shape, (4, 4))
        # 断言矩阵vh的形状为(0, 0)
        assert_equal(vh.shape, (0, 0))
        # 断言矩阵u与单位矩阵相等
        assert_equal(u, np.eye(4))

        # 创建一个空的0x4数组x
        x = np.empty((0, 4))
        # 计算x的奇异值分解u, s, vh，要求计算u和vh，是否共轭自适应
        u, s, vh = linalg.svd(x, compute_uv=True, hermitian=self.hermitian)
        # 断言矩阵u的形状为(0, 0)
        assert_equal(u.shape, (0, 0))
        # 断言矩阵vh的形状为(4, 4)
        assert_equal(vh.shape, (4, 4))
        # 断言矩阵vh与单位矩阵相等
        assert_equal(vh, np.eye(4))
class SVDHermitianCases(HermitianTestCase, HermitianGeneralizedTestCase):
    def do(self, a, b, tags):
        # 执行奇异值分解，指定使用 Hermite 转置（共轭转置）
        u, s, vt = linalg.svd(a, False, hermitian=True)
        # 断言近似相等，验证 SVD 分解结果的正确性
        assert_allclose(
            a,
            dot_generalized(
                np.asarray(u) * np.asarray(s)[..., None, :], np.asarray(vt)
            ),
            rtol=get_rtol(u.dtype),
        )

        # 定义 Hermite 转置函数
        def hermitian(mat):
            axes = list(range(mat.ndim))
            axes[-1], axes[-2] = axes[-2], axes[-1]
            return np.conj(np.transpose(mat, axes=axes))

        # 断言近似相等，验证 U 的 Hermite 转置乘以 U 等于单位矩阵
        assert_almost_equal(
            np.matmul(u, hermitian(u)), np.broadcast_to(np.eye(u.shape[-1]), u.shape)
        )
        # 断言近似相等，验证 V 的 Hermite 转置乘以 V 等于单位矩阵
        assert_almost_equal(
            np.matmul(vt, hermitian(vt)),
            np.broadcast_to(np.eye(vt.shape[-1]), vt.shape),
        )
        # 断言 S 奇异值从大到小排序
        assert_equal(np.sort(s), np.flip(s, -1))
        # 断言 U 和 V 是输入矩阵的一致子类
        assert_(consistent_subclass(u, a))
        assert_(consistent_subclass(vt, a))


class TestSVDHermitian(SVDHermitianCases, SVDBaseTests, TestCase):
    hermitian = True


class CondCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):
    # cond(x, p) for p in (None, 2, -2)

    def do(self, a, b, tags):
        c = asarray(a)  # 将输入转换为数组，以应对可能的矩阵输入
        if "size-0" in tags:
            # 如果标签中包含 "size-0"，则断言线性代数错误异常
            assert_raises(LinAlgError, linalg.cond, c)
            return

        # +-2 范数
        s = linalg.svd(c, compute_uv=False)
        # 断言近似相等，验证条件数的计算是否正确
        assert_almost_equal(
            linalg.cond(a), s[..., 0] / s[..., -1], single_decimal=5, double_decimal=11
        )
        # 断言近似相等，验证条件数的计算是否正确（使用 2 范数）
        assert_almost_equal(
            linalg.cond(a, 2),
            s[..., 0] / s[..., -1],
            single_decimal=5,
            double_decimal=11,
        )
        # 断言近似相等，验证条件数的计算是否正确（使用 -2 范数）
        assert_almost_equal(
            linalg.cond(a, -2),
            s[..., -1] / s[..., 0],
            single_decimal=5,
            double_decimal=11,
        )

        # 其他范数
        cinv = np.linalg.inv(c)
        # 断言近似相等，验证条件数的计算是否正确（使用 1 范数）
        assert_almost_equal(
            linalg.cond(a, 1),
            abs(c).sum(-2).max(-1) * abs(cinv).sum(-2).max(-1),
            single_decimal=5,
            double_decimal=11,
        )
        # 断言近似相等，验证条件数的计算是否正确（使用 -1 范数）
        assert_almost_equal(
            linalg.cond(a, -1),
            abs(c).sum(-2).min(-1) * abs(cinv).sum(-2).min(-1),
            single_decimal=5,
            double_decimal=11,
        )
        # 断言近似相等，验证条件数的计算是否正确（使用无穷大范数）
        assert_almost_equal(
            linalg.cond(a, np.inf),
            abs(c).sum(-1).max(-1) * abs(cinv).sum(-1).max(-1),
            single_decimal=5,
            double_decimal=11,
        )
        # 断言近似相等，验证条件数的计算是否正确（使用负无穷大范数）
        assert_almost_equal(
            linalg.cond(a, -np.inf),
            abs(c).sum(-1).min(-1) * abs(cinv).sum(-1).min(-1),
            single_decimal=5,
            double_decimal=11,
        )
        # 断言近似相等，验证条件数的计算是否正确（使用 Frobenius 范数）
        assert_almost_equal(
            linalg.cond(a, "fro"),
            np.sqrt((abs(c) ** 2).sum(-1).sum(-1) * (abs(cinv) ** 2).sum(-1).sum(-1)),
            single_decimal=5,
            double_decimal=11,
        )


class TestCond(CondCases, TestCase):
    def test_basic_nonsvd(self):
        # 测试非奇异值分解的基本情况
        A = array([[1.0, 0, 1], [0, -2.0, 0], [0, 0, 3.0]])
        # 检查无穷范数下的条件数
        assert_almost_equal(linalg.cond(A, inf), 4)
        # 检查负无穷范数下的条件数
        assert_almost_equal(linalg.cond(A, -inf), 2 / 3)
        # 检查 1-范数下的条件数
        assert_almost_equal(linalg.cond(A, 1), 4)
        # 检查负 1-范数下的条件数
        assert_almost_equal(linalg.cond(A, -1), 0.5)
        # 检查 Frobenius 范数下的条件数
        assert_almost_equal(linalg.cond(A, "fro"), np.sqrt(265 / 12))

    def test_singular(self):
        # 奇异矩阵在正范数下具有无穷的条件数，负范数不应引发异常
        As = [np.zeros((2, 2)), np.ones((2, 2))]
        p_pos = [None, 1, 2, "fro"]
        p_neg = [-1, -2]
        for A, p in itertools.product(As, p_pos):
            # 反转可能不会精确达到无穷大，所以只检查数值是否很大
            assert_(linalg.cond(A, p) > 1e15)
        for A, p in itertools.product(As, p_neg):
            linalg.cond(A, p)

    @skip(reason="NP_VER: fails on CI")  # (
    #    True, run=False, reason="Platform/LAPACK-dependent failure, see gh-18914"
    # )
    def test_nan(self):
        # NaN 应该被保留，而不是转换为无穷大
        ps = [None, 1, -1, 2, -2, "fro"]
        p_pos = [None, 1, 2, "fro"]

        A = np.ones((2, 2))
        A[0, 1] = np.nan
        for p in ps:
            c = linalg.cond(A, p)
            assert_(isinstance(c, np.float_))
            assert_(np.isnan(c))

        A = np.ones((3, 2, 2))
        A[1, 0, 1] = np.nan
        for p in ps:
            c = linalg.cond(A, p)
            assert_(np.isnan(c[1]))
            if p in p_pos:
                assert_(c[0] > 1e15)
                assert_(c[2] > 1e15)
            else:
                assert_(not np.isnan(c[0]))
                assert_(not np.isnan(c[2]))

    def test_stacked_singular(self):
        # 当堆叠矩阵中只有部分是奇异的时候的行为检查
        np.random.seed(1234)
        A = np.random.rand(2, 2, 2, 2)
        A[0, 0] = 0
        A[1, 1] = 0

        for p in (None, 1, 2, "fro", -1, -2):
            c = linalg.cond(A, p)
            assert_equal(c[0, 0], np.inf)
            assert_equal(c[1, 1], np.inf)
            assert_(np.isfinite(c[0, 1]))
            assert_(np.isfinite(c[1, 0]))
class PinvCases(
    LinalgSquareTestCase,  # 继承自 LinalgSquareTestCase 类
    LinalgNonsquareTestCase,  # 继承自 LinalgNonsquareTestCase 类
    LinalgGeneralizedSquareTestCase,  # 继承自 LinalgGeneralizedSquareTestCase 类
    LinalgGeneralizedNonsquareTestCase,  # 继承自 LinalgGeneralizedNonsquareTestCase 类
):
    def do(self, a, b, tags):
        a_ginv = linalg.pinv(a)
        # 如果 a 是奇异矩阵，`a @ a_ginv == I` 不成立
        dot = dot_generalized  # dot_generalized 函数的引用
        assert_almost_equal(
            dot(dot(a, a_ginv), a), a, single_decimal=5, double_decimal=11
        )  # 断言 a @ a_ginv @ a ≈ a，精确到小数点后5位和11位
        assert_(consistent_subclass(a_ginv, a))  # 断言 a_ginv 与 a 的类型一致


class TestPinv(PinvCases, TestCase):  # 继承自 PinvCases 和 TestCase 类
    pass


class PinvHermitianCases(HermitianTestCase, HermitianGeneralizedTestCase):
    def do(self, a, b, tags):
        a_ginv = linalg.pinv(a, hermitian=True)
        # 如果 a 是奇异矩阵，`a @ a_ginv == I` 不成立
        dot = dot_generalized  # dot_generalized 函数的引用
        assert_almost_equal(
            dot(dot(a, a_ginv), a), a, single_decimal=5, double_decimal=11
        )  # 断言 a @ a_ginv @ a ≈ a，精确到小数点后5位和11位
        assert_(consistent_subclass(a_ginv, a))  # 断言 a_ginv 与 a 的类型一致


class TestPinvHermitian(PinvHermitianCases, TestCase):  # 继承自 PinvHermitianCases 和 TestCase 类
    pass


class DetCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):
    def do(self, a, b, tags):
        d = linalg.det(a)  # 计算矩阵 a 的行列式值
        (s, ld) = linalg.slogdet(a)  # 计算矩阵 a 的行列式的符号和自然对数值
        if asarray(a).dtype.type in (single, double):
            ad = asarray(a).astype(double)  # 将矩阵 a 转换为 double 类型的数组
        else:
            ad = asarray(a).astype(cdouble)  # 将矩阵 a 转换为 cdouble 类型的数组
        ev = linalg.eigvals(ad)  # 计算数组 ad 的特征值
        assert_almost_equal(d, np.prod(ev, axis=-1))  # 断言 d 等于 ev 中所有特征值的乘积，最后一个维度上求乘积
        assert_almost_equal(s * np.exp(ld), np.prod(ev, axis=-1), single_decimal=5)  # 断言 s * e^ld 等于 ev 的乘积，精确到小数点后5位

        s = np.atleast_1d(s)  # 将 s 转换为至少一维的数组
        ld = np.atleast_1d(ld)  # 将 ld 转换为至少一维的数组
        m = s != 0  # 创建一个布尔数组，指示 s 中非零元素的位置
        assert_almost_equal(np.abs(s[m]), 1)  # 断言 s 中非零元素的绝对值近似为1
        assert_equal(ld[~m], -inf)  # 断言 ld 中对应 s 中零元素的位置为 -inf


@instantiate_parametrized_tests
class TestDet(DetCases, TestCase):  # 继承自 DetCases 和 TestCase 类
    def test_zero(self):
        # 注意：以下测试被注释掉，因为返回的是零维数组，类型检查无法通过
        assert_equal(linalg.det([[0.0]]), 0.0)
        #    assert_equal(type(linalg.det([[0.0]])), double)
        assert_equal(linalg.det([[0.0j]]), 0.0)
        #    assert_equal(type(linalg.det([[0.0j]])), cdouble)

        assert_equal(linalg.slogdet([[0.0]]), (0.0, -inf))
        #    assert_equal(type(linalg.slogdet([[0.0]])[0]), double)
        #    assert_equal(type(linalg.slogdet([[0.0]])[1]), double)
        assert_equal(linalg.slogdet([[0.0j]]), (0.0j, -inf))

    #    assert_equal(type(linalg.slogdet([[0.0j]])[0]), cdouble)
    #    assert_equal(type(linalg.slogdet([[0.0j]])[1]), double)

    @parametrize("dtype", [single, double, csingle, cdouble])
    def test_types(self, dtype):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)  # 创建指定类型 dtype 的数组 x
        assert_equal(np.linalg.det(x).dtype, dtype)  # 断言 np.linalg.det(x) 的数据类型为 dtype
        ph, s = np.linalg.slogdet(x)  # 计算数组 x 的行列式的符号和自然对数值
        assert_equal(s.dtype, get_real_dtype(dtype))  # 断言 s 的数据类型为 dtype 的实部类型
        assert_equal(ph.dtype, dtype)  # 断言 ph 的数据类型为 dtype
    # 定义一个测试方法，用于测试零维空数组的情况
    def test_0_size(self):
        # 创建一个大小为 (0, 0) 的复数类型的零数组
        a = np.zeros((0, 0), dtype=np.complex64)
        # 计算数组的行列式值
        res = linalg.det(a)
        # 断言行列式的结果为 1.0
        assert_equal(res, 1.0)
        # 断言结果的数据类型为 np.complex64
        assert_(res.dtype.type is np.complex64)
        # 使用 slogdet 函数计算数组的符号和对数行列式
        res = linalg.slogdet(a)
        # 断言 slogdet 的结果为 (1, 0)
        assert_equal(res, (1, 0))
        # 断言结果中第一个元素的数据类型为 np.complex64
        assert_(res[0].dtype.type is np.complex64)
        # 断言结果中第二个元素的数据类型为 np.float32
        assert_(res[1].dtype.type is np.float32)

        # 创建一个大小为 (0, 0) 的双精度浮点数类型的零数组
        a = np.zeros((0, 0), dtype=np.float64)
        # 计算数组的行列式值
        res = linalg.det(a)
        # 断言行列式的结果为 1.0
        assert_equal(res, 1.0)
        # 断言结果的数据类型为 np.float64
        assert_(res.dtype.type is np.float64)
        # 使用 slogdet 函数计算数组的符号和对数行列式
        res = linalg.slogdet(a)
        # 断言 slogdet 的结果为 (1, 0)
        assert_equal(res, (1, 0))
        # 断言结果中第一个元素的数据类型为 np.float64
        assert_(res[0].dtype.type is np.float64)
        # 断言结果中第二个元素的数据类型为 np.float64
        assert_(res[1].dtype.type is np.float64)
# 继承两个测试用例类 LinalgSquareTestCase 和 LinalgNonsquareTestCase，用于测试最小二乘法
class LstsqCases(LinalgSquareTestCase, LinalgNonsquareTestCase):
    # 执行最小二乘法测试的方法，接受参数 a、b 和 tags
    def do(self, a, b, tags):
        # 将 a 转换为 NumPy 数组
        arr = np.asarray(a)
        # 获取数组 a 的形状 m, n
        m, n = arr.shape
        # 对 a 进行奇异值分解，得到 u, s, vt
        u, s, vt = linalg.svd(a, False)
        # 对 a 和 b 进行最小二乘法求解，返回 x (解)，residuals (残差)，rank (秩)，sv (奇异值)
        x, residuals, rank, sv = linalg.lstsq(a, b, rcond=-1)
        # 如果 m == 0，断言 x 全为 0
        if m == 0:
            assert_((x == 0).all())
        # 如果 m <= n，断言 b 等于 dot(a, x)，并且 rank 等于 m
        if m <= n:
            assert_almost_equal(b, dot(a, x), single_decimal=5)
            assert_equal(rank, m)
        else:
            # 如果 m > n，断言 rank 等于 n
            assert_equal(rank, n)
        # 如果 rank 等于 n 并且 m > n
        if rank == n and m > n:
            # 计算预期残差 expect_resids
            expect_resids = (np.asarray(abs(np.dot(a, x) - b)) ** 2).sum(axis=0)
            expect_resids = np.asarray(expect_resids)
            if np.asarray(b).ndim == 1:
                expect_resids = expect_resids.reshape(
                    1,
                )
                # 断言 residuals 的形状与 expect_resids 相同
                assert_equal(residuals.shape, expect_resids.shape)
        else:
            # 如果不满足上述条件，expect_resids 为空数组
            expect_resids = np.array([])  # .view(type(x))
        # 断言 residuals 与 expect_resids 的值接近
        assert_almost_equal(residuals, expect_resids, single_decimal=5)
        # 断言 residuals 的数据类型为浮点型
        assert_(np.issubdtype(residuals.dtype, np.floating))
        # 断言 x 和 b 具有一致的子类
        assert_(consistent_subclass(x, b))
        # 断言 residuals 和 b 具有一致的子类
        assert_(consistent_subclass(residuals, b))


# 实例化参数化测试类，测试最小二乘法
@instantiate_parametrized_tests
class TestLstsq(LstsqCases, TestCase):
    # 测试未来默认 rcond 参数的情况
    @xpassIfTorchDynamo  # (reason="Lstsq: we use the future default =None")
    def test_future_rcond(self):
        # 创建 NumPy 数组 a 和 b
        a = np.array(
            [
                [0.0, 1.0, 0.0, 1.0, 2.0, 0.0],
                [0.0, 2.0, 0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0, 4.0],
                [0.0, 0.0, 0.0, 2.0, 3.0, 0.0],
            ]
        ).T

        b = np.array([1, 0, 0, 0, 0, 0])
        # 忽略警告信息
        with suppress_warnings() as sup:
            # 记录 FutureWarning 类型的警告信息
            w = sup.record(FutureWarning, "`rcond` parameter will change")
            # 使用默认 rcond 参数进行最小二乘法求解
            x, residuals, rank, s = linalg.lstsq(a, b)
            assert_(rank == 4)
            # 使用 rcond=-1 参数进行最小二乘法求解
            x, residuals, rank, s = linalg.lstsq(a, b, rcond=-1)
            assert_(rank == 4)
            # 使用 rcond=None 参数进行最小二乘法求解
            x, residuals, rank, s = linalg.lstsq(a, b, rcond=None)
            assert_(rank == 3)
            # 断言警告信息只被触发一次（在第一次命令时）
            assert_(len(w) == 1)

    # 参数化测试 m, n, n_rhs 的不同取值
    @parametrize(
        "m, n, n_rhs",
        [
            (4, 2, 2),
            (0, 4, 1),
            (0, 4, 2),
            (4, 0, 1),
            (4, 0, 2),
            #    (4, 2, 0),    # Intel MKL ERROR: Parameter 4 was incorrect on entry to DLALSD.
            (0, 0, 0),
        ],
    )
    # 定义一个测试函数，用于测试特定条件下的线性最小二乘解法
    def test_empty_a_b(self, m, n, n_rhs):
        # 创建一个 m x n 的数组 a，其中包含 0 到 m*n-1 的整数，并重塑为 m 行 n 列的矩阵
        a = np.arange(m * n).reshape(m, n)
        # 创建一个 m 行 n_rhs 列的数组 b，其中所有元素为 1
        b = np.ones((m, n_rhs))
        # 调用线性代数模块的 lstsq 函数，求解方程组 a * x = b 的最小二乘解
        x, residuals, rank, s = linalg.lstsq(a, b, rcond=None)
        # 如果 m 等于 0，则断言 x 的所有元素均为 0
        if m == 0:
            assert_((x == 0).all())
        # 断言 x 的形状为 (n, n_rhs)
        assert_equal(x.shape, (n, n_rhs))
        # 根据 m 和 n_rhs 的关系断言 residuals 的形状
        assert_equal(residuals.shape, ((n_rhs,) if m > n else (0,)))
        # 如果 m 大于 n 且 n_rhs 大于 0，则验证 residuals 是否等于 b 列的平方和
        if m > n and n_rhs > 0:
            # 计算残差 r = b - a * x
            r = b - np.dot(a, x)
            # 断言 residuals 是否接近于 r 的平方和
            assert_almost_equal(residuals, (r * r).sum(axis=-2))
        # 断言 rank 的值为 m 和 n 中较小的那个
        assert_equal(rank, min(m, n))
        # 断言 s 的形状为 (min(m, n),)
        assert_equal(s.shape, (min(m, n),))

    # 定义一个测试函数，用于测试不兼容维度的情况
    def test_incompatible_dims(self):
        # 创建两个 NumPy 数组 x 和 y，分别包含一组数据
        x = np.array([0, 1, 2, 3])
        y = np.array([-1, 0.2, 0.9, 2.1, 3.3])
        # 创建矩阵 A，其中第一列是 x，第二列是长度为 x 的全为 1 的向量
        A = np.vstack([x, np.ones(len(x))]).T
        # 使用 lstsq 函数求解线性方程组 A * x = y 的最小二乘解
        # 当 A 和 y 的维度不兼容时，预期会引发 RuntimeError 或 LinAlgError 异常
        with assert_raises((RuntimeError, LinAlgError)):
            linalg.lstsq(A, y, rcond=None)
# @xfail  #(reason="no block()")
@skip  # FIXME: otherwise fails in setUp calling np.block
@instantiate_parametrized_tests
class TestMatrixPower(TestCase):
    def setUp(self):
        self.rshft_0 = np.eye(4)
        self.rshft_1 = self.rshft_0[[3, 0, 1, 2]]
        self.rshft_2 = self.rshft_0[[2, 3, 0, 1]]
        self.rshft_3 = self.rshft_0[[1, 2, 3, 0]]
        self.rshft_all = [self.rshft_0, self.rshft_1, self.rshft_2, self.rshft_3]
        self.noninv = array([[1, 0], [0, 0]])
        # 使用 np.block 创建一个嵌套的矩阵，由 rshft_0 组成的数组重复两次
        self.stacked = np.block([[[self.rshft_0]]] * 2)
        # FIXME 'e' 类型可能在未来生效
        # 定义不可逆数组的数据类型列表
        self.dtnoinv = [object, np.dtype("e"), np.dtype("g"), np.dtype("G")]

    @parametrize("dt", [np.dtype(c) for c in "?bBhilefdFD"])
    def test_large_power(self, dt):
        # 将 self.rshft_1 转换为指定数据类型 dt
        rshft = self.rshft_1.astype(dt)
        # 断言 matrix_power(rshft, 2**100 + 2**10 + 2**5 + 0) 等于 self.rshft_0
        assert_equal(matrix_power(rshft, 2**100 + 2**10 + 2**5 + 0), self.rshft_0)
        # 断言 matrix_power(rshft, 2**100 + 2**10 + 2**5 + 1) 等于 self.rshft_1
        assert_equal(matrix_power(rshft, 2**100 + 2**10 + 2**5 + 1), self.rshft_1)
        # 断言 matrix_power(rshft, 2**100 + 2**10 + 2**5 + 2) 等于 self.rshft_2
        assert_equal(matrix_power(rshft, 2**100 + 2**10 + 2**5 + 2), self.rshft_2)
        # 断言 matrix_power(rshft, 2**100 + 2**10 + 2**5 + 3) 等于 self.rshft_3
        assert_equal(matrix_power(rshft, 2**100 + 2**10 + 2**5 + 3), self.rshft_3)

    @parametrize("dt", [np.dtype(c) for c in "?bBhilefdFD"])
    def test_power_is_zero(self, dt):
        def tz(M):
            # 测试矩阵 M 的零次幂
            mz = matrix_power(M, 0)
            # 断言 mz 等于 M 的单位矩阵版本
            assert_equal(mz, identity_like_generalized(M))
            # 断言 mz 的数据类型等于 M 的数据类型
            assert_equal(mz.dtype, M.dtype)

        # 遍历 self.rshft_all 中的每个矩阵 mat，并测试其零次幂
        for mat in self.rshft_all:
            tz(mat.astype(dt))
            # 如果数据类型不是 object，则测试 self.stacked 的零次幂
            if dt != object:
                tz(self.stacked.astype(dt))

    @parametrize("dt", [np.dtype(c) for c in "?bBhilefdFD"])
    def test_power_is_one(self, dt):
        def tz(mat):
            # 测试矩阵 mat 的一次幂
            mz = matrix_power(mat, 1)
            # 断言 mz 等于 mat
            assert_equal(mz, mat)
            # 断言 mz 的数据类型等于 mat 的数据类型
            assert_equal(mz.dtype, mat.dtype)

        # 遍历 self.rshft_all 中的每个矩阵 mat，并测试其一次幂
        for mat in self.rshft_all:
            tz(mat.astype(dt))
            # 如果数据类型不是 object，则测试 self.stacked 的一次幂
            if dt != object:
                tz(self.stacked.astype(dt))

    @parametrize("dt", [np.dtype(c) for c in "?bBhilefdFD"])
    def test_power_is_two(self, dt):
        def tz(mat):
            # 测试矩阵 mat 的二次幂
            mz = matrix_power(mat, 2)
            # 根据矩阵数据类型选择 matmul 或 dot 运算
            mmul = matmul if mat.dtype != object else dot
            # 断言 mz 等于 mat 与自身的矩阵乘积
            assert_equal(mz, mmul(mat, mat))
            # 断言 mz 的数据类型等于 mat 的数据类型
            assert_equal(mz.dtype, mat.dtype)

        # 遍历 self.rshft_all 中的每个矩阵 mat，并测试其二次幂
        for mat in self.rshft_all:
            tz(mat.astype(dt))
            # 如果数据类型不是 object，则测试 self.stacked 的二次幂
            if dt != object:
                tz(self.stacked.astype(dt))

    @parametrize("dt", [np.dtype(c) for c in "?bBhilefdFD"])
    def test_power_is_minus_one(self, dt):
        def tz(mat):
            # 计算矩阵 mat 的逆矩阵
            invmat = matrix_power(mat, -1)
            # 根据矩阵数据类型选择 matmul 或 dot 运算
            mmul = matmul if mat.dtype != object else dot
            # 断言 invmat 与 mat 的矩阵乘积接近单位矩阵
            assert_almost_equal(mmul(invmat, mat), identity_like_generalized(mat))

        # 遍历 self.rshft_all 中的每个矩阵 mat，并测试其逆矩阵（若数据类型允许）
        for mat in self.rshft_all:
            if dt not in self.dtnoinv:
                tz(mat.astype(dt))

    @parametrize("dt", [np.dtype(c) for c in "?bBhilefdFD"])
    # 定义一个测试方法，用于测试在给定数据类型下的矩阵幂函数的异常情况（错误的幂值）
    def test_exceptions_bad_power(self, dt):
        # 将self.rshft_0转换为指定数据类型dt的矩阵
        mat = self.rshft_0.astype(dt)
        # 断言调用matrix_power函数时会抛出TypeError异常，因为幂值为1.5不合法
        assert_raises(TypeError, matrix_power, mat, 1.5)
        # 断言调用matrix_power函数时会抛出TypeError异常，因为幂值为列表[1]不合法
        assert_raises(TypeError, matrix_power, mat, [1])
    
    # 使用装饰器@parametrize标记的测试方法，针对非方阵输入的异常情况进行测试
    @parametrize("dt", [np.dtype(c) for c in "?bBhilefdFD"])
    def test_exceptions_non_square(self, dt):
        # 断言调用matrix_power函数时会抛出LinAlgError异常，因为输入数组是形状为(1,)的非方阵
        assert_raises(LinAlgError, matrix_power, np.array([1], dt), 1)
        # 断言调用matrix_power函数时会抛出LinAlgError异常，因为输入数组是形状为(2,1)的非方阵
        assert_raises(LinAlgError, matrix_power, np.array([[1], [2]], dt), 1)
        # 断言调用matrix_power函数时会抛出LinAlgError异常，因为输入数组是形状为(4,3,2)的非方阵
        assert_raises(LinAlgError, matrix_power, np.ones((4, 3, 2), dt), 1)
    
    # 使用装饰器@skipif标记的测试方法，在WebAssembly平台（IS_WASM为True）上跳过测试
    # 原因是浮点数误差在WebAssembly中无法正常工作
    @parametrize("dt", [np.dtype(c) for c in "?bBhilefdFD"])
    def test_exceptions_not_invertible(self, dt):
        # 如果指定的数据类型dt在self.dtnoinv列表中，则直接返回，不进行测试
        if dt in self.dtnoinv:
            return
        # 将self.noninv转换为指定数据类型dt的矩阵
        mat = self.noninv.astype(dt)
        # 断言调用matrix_power函数时会抛出LinAlgError异常，因为输入矩阵mat不可逆且试图求逆幂
        assert_raises(LinAlgError, matrix_power, mat, -1)
# 创建一个测试类 TestEigvalshCases，继承自 HermitianTestCase 和 HermitianGeneralizedTestCase
class TestEigvalshCases(HermitianTestCase, HermitianGeneralizedTestCase):
    
    # 定义一个方法 do，接受参数 a, b, tags
    def do(self, a, b, tags):
        # 标记为预期失败，原因是“sort complex”
        pytest.xfail(reason="sort complex")
        
        # 注意：由于 eig 返回的特征值数组顺序不保证，所以它们必须排序。
        # 计算特征值的下限，用于 Hermitian 或实对称矩阵
        ev = linalg.eigvalsh(a, "L")
        
        # 计算特征值和特征向量，用于通用方阵 a
        evalues, evectors = linalg.eig(a)
        
        # 在最后一个轴上对特征值数组进行排序
        evalues.sort(axis=-1)
        
        # 断言 ev 与排序后的 evalues 在相对误差 rtol 下非常接近
        assert_allclose(ev, evalues, rtol=get_rtol(ev.dtype))

        # 计算特征值的上限，用于 Hermitian 或实对称矩阵
        ev2 = linalg.eigvalsh(a, "U")
        
        # 断言 ev2 与排序后的 evalues 在相对误差 rtol 下非常接近
        assert_allclose(ev2, evalues, rtol=get_rtol(ev.dtype))


# 使用参数化测试实例化 TestEigvalsh 类
@instantiate_parametrized_tests
class TestEigvalsh(TestCase):
    
    # 参数化测试方法，测试不同的 dtype
    @parametrize("dtype", [single, double, csingle, cdouble])
    def test_types(self, dtype):
        # 创建一个 2x2 的矩阵 x，指定数据类型为 dtype
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        
        # 计算矩阵 x 的特征值
        w = np.linalg.eigvalsh(x)
        
        # 断言 w 的数据类型等于 dtype 的实部数据类型
        assert_equal(w.dtype, get_real_dtype(dtype))

    # 测试无效参数的情况
    def test_invalid(self):
        # 创建一个 2x2 的 float32 类型矩阵 x
        x = np.array([[1, 0.5], [0.5, 1]], dtype=np.float32)
        
        # 断言调用 np.linalg.eigvalsh(x, UPLO="lrong") 会引发 RuntimeError 或 ValueError 异常
        assert_raises((RuntimeError, ValueError), np.linalg.eigvalsh, x, UPLO="lrong")
        
        # 断言调用 np.linalg.eigvalsh(x, "lower") 会引发 RuntimeError 或 ValueError 异常
        assert_raises((RuntimeError, ValueError), np.linalg.eigvalsh, x, "lower")
        
        # 断言调用 np.linalg.eigvalsh(x, "upper") 会引发 RuntimeError 或 ValueError 异常
        assert_raises((RuntimeError, ValueError), np.linalg.eigvalsh, x, "upper")

    # 测试 UPLO 参数
    def test_UPLO(self):
        # 创建两个 2x2 的 double 类型矩阵 Klo 和 Kup，以及目标特征值数组 tgt 和相对误差 rtol
        Klo = np.array([[0, 0], [1, 0]], dtype=np.double)
        Kup = np.array([[0, 1], [0, 0]], dtype=np.double)
        tgt = np.array([-1, 1], dtype=np.double)
        rtol = get_rtol(np.double)

        # 检查默认情况下 UPLO 参数为 'L'
        w = np.linalg.eigvalsh(Klo)
        assert_allclose(w, tgt, rtol=rtol)
        
        # 检查指定 UPLO 参数为 'L'
        w = np.linalg.eigvalsh(Klo, UPLO="L")
        assert_allclose(w, tgt, rtol=rtol)
        
        # 检查指定 UPLO 参数为 'l'
        w = np.linalg.eigvalsh(Klo, UPLO="l")
        assert_allclose(w, tgt, rtol=rtol)
        
        # 检查指定 UPLO 参数为 'U'
        w = np.linalg.eigvalsh(Kup, UPLO="U")
        assert_allclose(w, tgt, rtol=rtol)
        
        # 检查指定 UPLO 参数为 'u'
        w = np.linalg.eigvalsh(Kup, UPLO="u")
        assert_allclose(w, tgt, rtol=rtol)

    # 测试零大小数组的情况
    def test_0_size(self):
        # 检查所有类型的零大小数组的工作情况
        
        # 创建一个 0x1x1 的 int 类型零数组 a
        a = np.zeros((0, 1, 1), dtype=np.int_)  # .view(ArraySubclass)
        
        # 计算数组 a 的特征值
        res = linalg.eigvalsh(a)
        
        # 断言 res 的数据类型为 float64
        assert_(res.dtype.type is np.float64)
        
        # 断言 res 的形状为 (0, 1)
        assert_equal((0, 1), res.shape)
        
        # 这仅供文档使用，可能有改变的可能性
        # 断言 res 是 np.ndarray 的实例
        assert_(isinstance(res, np.ndarray))

        # 创建一个 0x0 的 complex64 类型零数组 a
        a = np.zeros((0, 0), dtype=np.complex64)  # .view(ArraySubclass)
        
        # 计算数组 a 的特征值
        res = linalg.eigvalsh(a)
        
        # 断言 res 的数据类型为 float32
        assert_(res.dtype.type is np.float32)
        
        # 断言 res 的形状为 (0,)
        assert_equal((0,), res.shape)
        
        # 这仅供文档使用，可能有改变的可能性
        # 断言 res 是 np.ndarray 的实例
        assert_(isinstance(res, np.ndarray))
    # 定义一个方法 `do`，接受参数 `a`、`b` 和 `tags`
    def do(self, a, b, tags):
        # 声明当前测试为预期失败状态，原因是“排序复杂”
        pytest.xfail(reason="sort complex")
        # 注意：eig 返回的特征值数组需要排序，因为它们的顺序是不保证的。

        # 使用 `linalg.eigh` 计算矩阵 `a` 的特征值 `ev` 和特征向量 `evc`
        ev, evc = linalg.eigh(a)
        
        # 使用 `linalg.eig` 计算矩阵 `a` 的特征值 `evalues` 和特征向量 `evectors`
        evalues, evectors = linalg.eig(a)
        
        # 对特征值 `evalues` 进行排序，沿着最后一个轴（即最后一个维度）
        evalues.sort(axis=-1)
        
        # 断言：`ev` 应该与排序后的 `evalues` 很接近
        assert_almost_equal(ev, evalues)

        # 断言：使用自定义函数 `dot_generalized` 计算结果与特征值 `ev` 乘以特征向量 `evc` 的结果非常接近
        assert_allclose(
            dot_generalized(a, evc),
            np.asarray(ev)[..., None, :] * np.asarray(evc),
            rtol=get_rtol(ev.dtype),
        )

        # 使用参数 `"U"` 调用 `linalg.eigh`，得到特征值 `ev2` 和特征向量 `evc2`
        ev2, evc2 = linalg.eigh(a, "U")
        
        # 断言：`ev2` 应该与排序后的 `evalues` 很接近
        assert_almost_equal(ev2, evalues)

        # 断言：使用自定义函数 `dot_generalized` 计算结果与特征值 `ev2` 乘以特征向量 `evc2` 的结果非常接近
        assert_allclose(
            dot_generalized(a, evc2),
            np.asarray(ev2)[..., None, :] * np.asarray(evc2),
            rtol=get_rtol(ev.dtype),
            err_msg=repr(a),
        )
# 实例化参数化测试装饰器，用于为测试类添加参数化测试方法
@instantiate_parametrized_tests
# 定义测试类 TestEigh，继承自 TestCase 类
class TestEigh(TestCase):

    # 参数化测试方法，测试不同数据类型的输入
    @parametrize("dtype", [single, double, csingle, cdouble])
    def test_types(self, dtype):
        # 创建一个 numpy 数组 x，指定数据类型为 dtype
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        # 使用 np.linalg.eigh 计算矩阵 x 的特征值和特征向量
        w, v = np.linalg.eigh(x)
        # 断言特征值的数据类型与 get_real_dtype 函数返回的实数数据类型相等
        assert_equal(w.dtype, get_real_dtype(dtype))
        # 断言特征向量的数据类型与输入的数据类型 dtype 相等
        assert_equal(v.dtype, dtype)

    # 测试无效参数的情况
    def test_invalid(self):
        # 创建一个 numpy 数组 x，指定数据类型为 np.float32
        x = np.array([[1, 0.5], [0.5, 1]], dtype=np.float32)
        # 断言调用 np.linalg.eigh 函数时传入无效参数 UPLO="lrong" 会抛出 RuntimeError 或 ValueError 异常
        assert_raises((RuntimeError, ValueError), np.linalg.eigh, x, UPLO="lrong")
        # 断言调用 np.linalg.eigh 函数时传入无效参数 UPLO="lower" 会抛出 RuntimeError 或 ValueError 异常
        assert_raises((RuntimeError, ValueError), np.linalg.eigh, x, "lower")
        # 断言调用 np.linalg.eigh 函数时传入无效参数 UPLO="upper" 会抛出 RuntimeError 或 ValueError 异常
        assert_raises((RuntimeError, ValueError), np.linalg.eigh, x, "upper")

    # 测试不同 UPLO 参数的情况
    def test_UPLO(self):
        # 创建两个对称矩阵 Klo 和 Kup，数据类型为 np.double
        Klo = np.array([[0, 0], [1, 0]], dtype=np.double)
        Kup = np.array([[0, 1], [0, 0]], dtype=np.double)
        # 目标特征值数组 tgt，数据类型为 np.double
        tgt = np.array([-1, 1], dtype=np.double)
        # 获取与 np.double 数据类型对应的相对误差容限 rtol
        rtol = get_rtol(np.double)

        # 检查默认 UPLO 参数 'L' 的情况
        w, v = np.linalg.eigh(Klo)
        assert_allclose(w, tgt, rtol=rtol)
        # 检查显式指定 UPLO 参数为 'L' 的情况
        w, v = np.linalg.eigh(Klo, UPLO="L")
        assert_allclose(w, tgt, rtol=rtol)
        # 检查 UPLO 参数为 'l' 的情况
        w, v = np.linalg.eigh(Klo, UPLO="l")
        assert_allclose(w, tgt, rtol=rtol)
        # 检查 UPLO 参数为 'U' 的情况
        w, v = np.linalg.eigh(Kup, UPLO="U")
        assert_allclose(w, tgt, rtol=rtol)
        # 检查 UPLO 参数为 'u' 的情况
        w, v = np.linalg.eigh(Kup, UPLO="u")
        assert_allclose(w, tgt, rtol=rtol)

    # 测试零尺寸数组的情况
    def test_0_size(self):
        # 创建一个零尺寸的整数数组 a，数据类型为 np.int_
        a = np.zeros((0, 1, 1), dtype=np.int_)  # .view(ArraySubclass)
        # 调用 linalg.eigh 计算零尺寸数组 a 的特征值和特征向量
        res, res_v = linalg.eigh(a)
        # 断言特征向量的数据类型是 np.float64
        assert_(res_v.dtype.type is np.float64)
        # 断言特征值的数据类型是 np.float64
        assert_(res.dtype.type is np.float64)
        # 断言数组 a 的形状与特征向量 res_v 的形状相等
        assert_equal(a.shape, res_v.shape)
        # 断言特征值数组 res 的形状为 (0, 1)
        assert_equal((0, 1), res.shape)
        # 用于文档说明，可能需要更改的断言
        assert_(isinstance(a, np.ndarray))

        # 创建一个零尺寸的复数数组 a，数据类型为 np.complex64
        a = np.zeros((0, 0), dtype=np.complex64)  # .view(ArraySubclass)
        # 调用 linalg.eigh 计算零尺寸数组 a 的特征值和特征向量
        res, res_v = linalg.eigh(a)
        # 断言特征向量的数据类型是 np.complex64
        assert_(res_v.dtype.type is np.complex64)
        # 断言特征值的数据类型是 np.float32
        assert_(res.dtype.type is np.float32)
        # 断言数组 a 的形状与特征向量 res_v 的形状相等
        assert_equal(a.shape, res_v.shape)
        # 断言特征值数组 res 的形状为 (0,)
        assert_equal((0,), res.shape)
        # 用于文档说明，可能需要更改的断言
        assert_(isinstance(a, np.ndarray))


# 定义基类 _TestNormBase
class _TestNormBase:
    # 数据类型 dt 和 dec 的初始值为 None
    dt = None
    dec = None

    # 静态方法，用于检查数组 x 和结果 res 的数据类型
    @staticmethod
    def check_dtype(x, res):
        # 如果 x 的数据类型为浮点数类型
        if issubclass(x.dtype.type, np.inexact):
            # 断言结果 res 的数据类型与 x 实部的数据类型相等
            assert_equal(res.dtype, x.real.dtype)
        else:
            # 对于整数输入，不必测试输出的浮点数精度
            assert_(issubclass(res.dtype.type, np.floating))


# 派生自 _TestNormBase 的测试类 _TestNormGeneral
class _TestNormGeneral(_TestNormBase):
    # 测试空数组的情况
    def test_empty(self):
        # 断言使用 norm 函数计算空列表的范数为 0.0
        assert_equal(norm([]), 0.0)
        # 断言使用 norm 函数计算空数组的范数为 0.0
        assert_equal(norm(array([], dtype=self.dt)), 0.0)
        # 断言使用 norm 函数计算至少二维空数组的范数为 0.0
        assert_equal(norm(atleast_2d(array([], dtype=self.dt))), 0.0)
    # 定义一个测试方法，用于验证向量的返回类型
    def test_vector_return_type(self):
        # 创建一个NumPy数组，包含整数类型的元素
        a = np.array([1, 0, 1])

        # 精确类型的字符串，包含所有整数类型的字符码
        exact_types = "Bbhil"  # np.typecodes["AllInteger"]
        # 非精确类型的字符串，包含所有浮点数类型的字符码
        inexact_types = "efdFD"  # np.typecodes["AllFloat"]

        # 合并精确和非精确类型的字符码字符串
        all_types = exact_types + inexact_types

        # 对于每种数据类型字符码进行循环
        for each_type in all_types:
            # 将数组 a 转换为当前数据类型 each_type 的数组
            at = a.astype(each_type)

            # 如果当前数据类型是 np.dtype("float16")
            if each_type == np.dtype("float16"):
                # 抛出跳过测试的异常，并提供原因说明
                raise SkipTest("float16**float64 => float64 (?)")

            # 计算数组 at 的负无穷范数
            an = norm(at, -np.inf)
            # 检查负无穷范数计算的数据类型
            self.check_dtype(at, an)
            # 断言负无穷范数的计算结果接近于 0.0

            # 使用警告抑制器，捕获运行时警告并过滤掉“除以零”的警告
            with suppress_warnings() as sup:
                sup.filter(RuntimeWarning, "divide by zero encountered")
                # 计算数组 at 的 -1 范数
                an = norm(at, -1)
                # 检查 -1 范数计算的数据类型
                self.check_dtype(at, an)
                # 断言 -1 范数的计算结果接近于 0.0

            # 计算数组 at 的 0 范数
            an = norm(at, 0)
            # 检查 0 范数计算的数据类型
            self.check_dtype(at, an)
            # 断言 0 范数的计算结果接近于 2

            # 计算数组 at 的 1 范数
            an = norm(at, 1)
            # 检查 1 范数计算的数据类型
            self.check_dtype(at, an)
            # 断言 1 范数的计算结果接近于 2.0

            # 计算数组 at 的 2 范数
            an = norm(at, 2)
            # 检查 2 范数计算的数据类型
            self.check_dtype(at, an)
            # 断言 2 范数的计算结果接近于 数组元素类型为 float 的 2.0 的平方根

            # 计算数组 at 的 4 范数
            an = norm(at, 4)
            # 检查 4 范数计算的数据类型
            self.check_dtype(at, an)
            # 断言 4 范数的计算结果接近于 数组元素类型为 float 的 2.0 的四分之一次方

            # 计算数组 at 的正无穷范数
            an = norm(at, np.inf)
            # 检查正无穷范数计算的数据类型
            self.check_dtype(at, an)
            # 断言正无穷范数的计算结果接近于 1.0

    # 定义一个测试方法，用于验证向量的计算
    def test_vector(self):
        # 定义三个测试向量 a, b, c
        a = [1, 2, 3, 4]
        b = [-1, -2, -3, -4]
        c = [-1, 2, -3, 4]

        # 内部测试函数，用于测试给定向量的范数计算
        def _test(v):
            # 断言计算的向量范数接近于数值 30 的平方根
            np.testing.assert_almost_equal(norm(v), 30**0.5, decimal=self.dec)
            # 断言计算的向量无穷范数接近于 4.0
            np.testing.assert_almost_equal(norm(v, np.inf), 4.0, decimal=self.dec)
            # 断言计算的向量负无穷范数接近于 1.0
            np.testing.assert_almost_equal(norm(v, -np.inf), 1.0, decimal=self.dec)
            # 断言计算的向量 1 范数接近于 10.0
            np.testing.assert_almost_equal(norm(v, 1), 10.0, decimal=self.dec)
            # 断言计算的向量 -1 范数接近于 12.0 / 25
            np.testing.assert_almost_equal(norm(v, -1), 12.0 / 25, decimal=self.dec)
            # 断言计算的向量 2 范数接近于数值 30 的平方根
            np.testing.assert_almost_equal(norm(v, 2), 30**0.5, decimal=self.dec)
            # 断言计算的向量 -2 范数接近于 ((205.0 / 144) ** -0.5)
            np.testing.assert_almost_equal(
                norm(v, -2), ((205.0 / 144) ** -0.5), decimal=self.dec
            )
            # 断言计算的向量 0 范数等于 4
            np.testing.assert_almost_equal(norm(v, 0), 4, decimal=self.dec)

        # 对三个测试向量分别执行测试函数 _test
        for v in (
            a,
            b,
            c,
        ):
            _test(v)

        # 对三个测试向量分别创建 NumPy 数组，使用指定的数据类型 self.dt，然后执行测试函数 _test
        for v in (
            np.array(a, dtype=self.dt),
            np.array(b, dtype=self.dt),
            np.array(c, dtype=self.dt),
        ):
            _test(v)
    def test_axis(self):
        # 测试轴向计算

        # 向量范数
        # 比较使用 `axis` 参数与分别计算每行或每列范数的区别
        A = array([[1, 2, 3], [4, 5, 6]], dtype=self.dt)

        for order in [None, -1, 0, 1, 2, 3, np.inf, -np.inf]:
            # 计算每列的预期范数
            expected0 = [norm(A[:, k], ord=order) for k in range(A.shape[1])]
            assert_almost_equal(norm(A, ord=order, axis=0), expected0)

            # 计算每行的预期范数
            expected1 = [norm(A[k, :], ord=order) for k in range(A.shape[0])]
            assert_almost_equal(norm(A, ord=order, axis=1), expected1)

        # 矩阵范数
        B = np.arange(1, 25, dtype=self.dt).reshape(2, 3, 4)
        nd = B.ndim

        for order in [None, -2, 2, -1, 1, np.inf, -np.inf, "fro"]:
            # 遍历所有轴的组合
            for axis in itertools.combinations(range(-nd, nd), 2):
                row_axis, col_axis = axis
                if row_axis < 0:
                    row_axis += nd
                if col_axis < 0:
                    col_axis += nd

                if row_axis == col_axis:
                    # 如果行轴与列轴相同，期望引发异常
                    assert_raises(
                        (RuntimeError, ValueError), norm, B, ord=order, axis=axis
                    )
                else:
                    # 否则计算范数 n
                    n = norm(B, ord=order, axis=axis)

                    # 根据 k_index 的逻辑仅适用于 nd = 3
                    # 如果 nd 增加，这部分需要修改
                    k_index = nd - (row_axis + col_axis)
                    if row_axis < col_axis:
                        # 计算预期值，取轴 k_index 的不同切片
                        expected = [
                            norm(B[:].take(k, axis=k_index), ord=order)
                            for k in range(B.shape[k_index])
                        ]
                    else:
                        # 计算预期值，对轴 k_index 的切片转置后计算范数
                        expected = [
                            norm(B[:].take(k, axis=k_index).T, ord=order)
                            for k in range(B.shape[k_index])
                        ]
                    assert_almost_equal(n, expected)
    def test_keepdims(self):
        # 创建一个二维数组 A，范围为 1 到 24，使用指定的数据类型 self.dt，并且形状为 2x3x4
        A = np.arange(1, 25, dtype=self.dt).reshape(2, 3, 4)

        # 错误消息模板，用于 allclose 断言
        allclose_err = "order {0}, axis = {1}"
        # 错误消息模板，用于形状断言
        shape_err = "Shape mismatch found {0}, expected {1}, order={2}, axis={3}"

        # 检查 order=None, axis=None 的情况
        # 计算期望值，不指定 order 和 axis 的 L2 范数
        expected = norm(A, ord=None, axis=None)
        # 计算找到的值，保持维度为 True 的情况下计算 L2 范数
        found = norm(A, ord=None, axis=None, keepdims=True)
        # 使用 allclose 函数断言两个值的近似程度，将 found 值展平以便比较，错误消息指定 order 和 axis
        assert_allclose(
            np.squeeze(found), expected, err_msg=allclose_err.format(None, None)
        )
        # 预期的形状是 (1, 1, 1)
        expected_shape = (1, 1, 1)
        # 使用 assert_ 函数断言 found 的形状与预期形状相符，错误消息指定形状和期望形状，不指定 order 和 axis
        assert_(
            found.shape == expected_shape,
            shape_err.format(found.shape, expected_shape, None, None),
        )

        # 向量范数。
        for order in [None, -1, 0, 1, 2, 3, np.inf, -np.inf]:
            for k in range(A.ndim):
                # 计算指定 order 和 axis 的 Lp 范数的期望值
                expected = norm(A, ord=order, axis=k)
                # 计算找到的值，保持维度为 True 的情况下计算 Lp 范数
                found = norm(A, ord=order, axis=k, keepdims=True)
                # 使用 allclose 函数断言两个值的近似程度，将 found 值展平以便比较，错误消息指定 order 和 k
                assert_allclose(
                    np.squeeze(found), expected, err_msg=allclose_err.format(order, k)
                )
                # 预期的形状是 A 的形状，但在第 k 轴上为 1
                expected_shape = list(A.shape)
                expected_shape[k] = 1
                expected_shape = tuple(expected_shape)
                # 使用 assert_ 函数断言 found 的形状与预期形状相符，错误消息指定形状和期望形状，指定 order 和 k
                assert_(
                    found.shape == expected_shape,
                    shape_err.format(found.shape, expected_shape, order, k),
                )

        # 矩阵范数。
        for order in [None, -2, 2, -1, 1, np.inf, -np.inf, "fro", "nuc"]:
            for k in itertools.permutations(range(A.ndim), 2):
                # 计算指定 order 和 axis 的矩阵范数的期望值
                expected = norm(A, ord=order, axis=k)
                # 计算找到的值，保持维度为 True 的情况下计算矩阵范数
                found = norm(A, ord=order, axis=k, keepdims=True)
                # 使用 allclose 函数断言两个值的近似程度，将 found 值展平以便比较，错误消息指定 order 和 k
                assert_allclose(
                    np.squeeze(found), expected, err_msg=allclose_err.format(order, k)
                )
                # 预期的形状是 A 的形状，但在 k[0] 和 k[1] 轴上为 1
                expected_shape = list(A.shape)
                expected_shape[k[0]] = 1
                expected_shape[k[1]] = 1
                expected_shape = tuple(expected_shape)
                # 使用 assert_ 函数断言 found 的形状与预期形状相符，错误消息指定形状和期望形状，指定 order 和 k
                assert_(
                    found.shape == expected_shape,
                    shape_err.format(found.shape, expected_shape, order, k),
                )
class _TestNorm2D(_TestNormBase):
    # 定义2D数组的部分，以便我们可以在matrixlib.tests.test_matrix_linalg中使用np.matrix子类化此类并运行测试。

    def test_matrix_empty(self):
        # 测试空矩阵的情况，期望其范数为0
        assert_equal(norm(np.array([[]], dtype=self.dt)), 0.0)

    def test_matrix_return_type(self):
        a = np.array([[1, 0, 1], [0, 1, 1]])

        exact_types = "Bbhil"  # np.typecodes["AllInteger"]

        # 仅有float32、complex64、float64、complex128类型被`linalg`允许，
        # 这些类型在`norm`函数内执行矩阵操作时使用。
        inexact_types = "fdFD"

        all_types = exact_types + inexact_types

        for each_type in all_types:
            at = a.astype(each_type)

            # 使用不同的范数计算矩阵at，并进行断言比较
            an = norm(at, -np.inf)
            self.check_dtype(at, an)
            assert_almost_equal(an, 2.0)

            with suppress_warnings() as sup:
                sup.filter(RuntimeWarning, "divide by zero encountered")
                an = norm(at, -1)
                self.check_dtype(at, an)
                assert_almost_equal(an, 1.0)

            an = norm(at, 1)
            self.check_dtype(at, an)
            assert_almost_equal(an, 2.0)

            an = norm(at, 2)
            self.check_dtype(at, an)
            assert_almost_equal(an, 3.0 ** (1.0 / 2.0))

            an = norm(at, -2)
            self.check_dtype(at, an)
            assert_almost_equal(an, 1.0)

            an = norm(at, np.inf)
            self.check_dtype(at, an)
            assert_almost_equal(an, 2.0)

            an = norm(at, "fro")
            self.check_dtype(at, an)
            assert_almost_equal(an, 2.0)

            an = norm(at, "nuc")
            self.check_dtype(at, an)
            # 需要更低的精度来支持低精度浮点数。
            # 它们的值在第7位上相差1。
            np.testing.assert_almost_equal(an, 2.7320508075688772, decimal=6)

    def test_matrix_2x2(self):
        A = np.array([[1, 3], [5, 7]], dtype=self.dt)
        assert_almost_equal(norm(A), 84**0.5)
        assert_almost_equal(norm(A, "fro"), 84**0.5)
        assert_almost_equal(norm(A, "nuc"), 10.0)
        assert_almost_equal(norm(A, inf), 12.0)
        assert_almost_equal(norm(A, -inf), 4.0)
        assert_almost_equal(norm(A, 1), 10.0)
        assert_almost_equal(norm(A, -1), 6.0)
        assert_almost_equal(norm(A, 2), 9.1231056256176615)
        assert_almost_equal(norm(A, -2), 0.87689437438234041)

        # 测试不支持的范数和无效的p值
        assert_raises((RuntimeError, ValueError), norm, A, "nofro")
        assert_raises((RuntimeError, ValueError), norm, A, -3)
        assert_raises((RuntimeError, ValueError), norm, A, 0)
    def test_matrix_3x3(self):
        # This test has been added because the 2x2 example
        # happened to have equal nuclear norm and induced 1-norm.
        # The 1/10 scaling factor accommodates the absolute tolerance
        # used in assert_almost_equal.
        # 定义一个3x3的矩阵A，使用self.dt指定数据类型
        A = (1 / 10) * np.array([[1, 2, 3], [6, 0, 5], [3, 2, 1]], dtype=self.dt)
        # 断言A的普通矩阵范数近似于 (1 / 10) * 89 的平方根
        assert_almost_equal(norm(A), (1 / 10) * 89**0.5)
        # 断言A的Frobenius范数近似于 (1 / 10) * 89 的平方根
        assert_almost_equal(norm(A, "fro"), (1 / 10) * 89**0.5)
        # 断言A的核范数近似于给定的数值
        assert_almost_equal(norm(A, "nuc"), 1.3366836911774836)
        # 断言A的无穷范数近似于给定的数值
        assert_almost_equal(norm(A, np.inf), 1.1)
        # 断言A的负无穷范数近似于给定的数值
        assert_almost_equal(norm(A, -np.inf), 0.6)
        # 断言A的1范数近似于给定的数值
        assert_almost_equal(norm(A, 1), 1.0)
        # 断言A的负1范数近似于给定的数值
        assert_almost_equal(norm(A, -1), 0.4)
        # 断言A的2范数近似于给定的数值
        assert_almost_equal(norm(A, 2), 0.88722940323461277)
        # 断言A的负2范数近似于给定的数值
        assert_almost_equal(norm(A, -2), 0.19456584790481812)

    def test_bad_args(self):
        # Check that bad arguments raise the appropriate exceptions.

        # 创建一个2x3的矩阵A和一个形状为(2, 3, 4)的张量B，使用self.dt指定数据类型
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=self.dt)
        B = np.arange(1, 25, dtype=self.dt).reshape(2, 3, 4)

        # 当使用 `ord='fro'` 或 `ord='nuc'` 或任何其它字符串时，
        # 如果使用 `axis=<integer>` 或传递一个1维数组，意味着正在计算向量范数，
        # 这时会抛出 ValueError 异常。
        assert_raises((RuntimeError, ValueError), norm, A, "fro", 0)
        assert_raises((RuntimeError, ValueError), norm, A, "nuc", 0)
        assert_raises((RuntimeError, ValueError), norm, [3, 4], "fro", None)
        assert_raises((RuntimeError, ValueError), norm, [3, 4], "nuc", None)
        assert_raises((RuntimeError, ValueError), norm, [3, 4], "test", None)

        # 同样地，当计算矩阵范数时，如果 `ord` 是除了1、2、-1或-2以外的有限数，
        # norm 应该抛出异常。
        for order in [0, 3]:
            assert_raises((RuntimeError, ValueError), norm, A, order, None)
            assert_raises((RuntimeError, ValueError), norm, A, order, (0, 1))
            assert_raises((RuntimeError, ValueError), norm, B, order, (1, 2))

        # 无效的轴
        assert_raises((IndexError, np.AxisError), norm, B, None, 3)
        assert_raises((IndexError, np.AxisError), norm, B, None, (2, 3))
        assert_raises((RuntimeError, ValueError), norm, B, None, (0, 1, 2))
class _TestNorm(_TestNorm2D, _TestNormGeneral):
    pass



class TestNorm_NonSystematic(TestCase):
    def test_intmin(self):
        # 非回归测试：先前对带符号整数的 p-范数进行了错误的浮点转换和绝对值操作顺序。
        x = np.array([-(2**31)], dtype=np.int32)
        old_assert_almost_equal(norm(x, ord=3), 2**31, decimal=5)



# 将基类 _TestNormBase 拆分为单独的类，以便用于矩阵测试。
class _TestNormDoubleBase(_TestNormBase, TestCase):
    dt = np.double
    dec = 12



class _TestNormSingleBase(_TestNormBase, TestCase):
    dt = np.float32
    dec = 6



class _TestNormInt64Base(_TestNormBase, TestCase):
    dt = np.int64
    dec = 12



class TestNormDouble(_TestNorm, _TestNormDoubleBase, TestCase):
    pass



class TestNormSingle(_TestNorm, _TestNormSingleBase, TestCase):
    pass



class TestNormInt64(_TestNorm, _TestNormInt64Base):
    pass



class TestMatrixRank(TestCase):
    def test_matrix_rank(self):
        # 完全秩矩阵
        assert_equal(4, matrix_rank(np.eye(4)))
        # 秩不足矩阵
        I = np.eye(4)
        I[-1, -1] = 0.0
        assert_equal(matrix_rank(I), 3)
        # 全零矩阵 - 秩为零
        assert_equal(matrix_rank(np.zeros((4, 4))), 0)
        # 一维数组 - 除非全为零，否则秩为1
        assert_equal(matrix_rank([1, 0, 0, 0]), 1)
        assert_equal(matrix_rank(np.zeros((4,))), 0)
        # 接受类数组作为参数
        assert_equal(matrix_rank([1]), 1)
        # 大于2维的数组被视为堆叠的矩阵
        ms = np.array([I, np.eye(4), np.zeros((4, 4))])
        assert_equal(matrix_rank(ms), np.array([3, 4, 0]))
        # 对标量也有效
        assert_equal(matrix_rank(1), 1)

    def test_symmetric_rank(self):
        assert_equal(4, matrix_rank(np.eye(4), hermitian=True))
        assert_equal(1, matrix_rank(np.ones((4, 4)), hermitian=True))
        assert_equal(0, matrix_rank(np.zeros((4, 4)), hermitian=True))
        # 秩不足矩阵
        I = np.eye(4)
        I[-1, -1] = 0.0
        assert_equal(3, matrix_rank(I, hermitian=True))
        # 手动提供容差
        I[-1, -1] = 1e-8
        assert_equal(4, matrix_rank(I, hermitian=True, tol=0.99e-8))
        assert_equal(3, matrix_rank(I, hermitian=True, tol=1.01e-8))

    def test_reduced_rank(self):
        # 测试具有降秩的矩阵
        # np.random.RandomState(20120714)用于生成伪随机数
        np.random.seed(20120714)
        for i in range(100):
            # 生成一个降秩矩阵
            X = np.random.normal(size=(40, 10))
            X[:, 0] = X[:, 1] + X[:, 2]
            # 断言矩阵秩被正确检测到
            assert_equal(matrix_rank(X), 9)
            X[:, 3] = X[:, 4] + X[:, 5]
            assert_equal(matrix_rank(X), 8)

@instantiate_parametrized_tests
class TestQR(TestCase):
    # 定义一个方法用于检查 QR 分解的结果是否符合预期
    def check_qr(self, a):
        # This test expects the argument `a` to be an ndarray or
        # a subclass of an ndarray of inexact type.
        # 获取参数 `a` 的类型
        a_type = type(a)
        # 获取参数 `a` 的数据类型
        a_dtype = a.dtype
        # 获取矩阵 `a` 的行数 m 和列数 n
        m, n = a.shape
        # 取 m 和 n 中的较小值，作为 QR 分解时的维度 k

        # mode == 'complete'
        # 使用完全模式进行 QR 分解，返回 Q 和 R
        q, r = linalg.qr(a, mode="complete")
        # 断言 Q 和 R 的数据类型与参数 `a` 的数据类型相同
        assert_(q.dtype == a_dtype)
        assert_(r.dtype == a_dtype)
        # 断言 Q 和 R 的类型是参数 `a` 的类型或其子类
        assert_(isinstance(q, a_type))
        assert_(isinstance(r, a_type))
        # 断言 Q 的形状为 (m, m)，R 的形状为 (m, n)
        assert_(q.shape == (m, m))
        assert_(r.shape == (m, n))
        # 断言 Q*R 等于原始矩阵 a，精确到小数点后五位
        assert_almost_equal(dot(q, r), a, single_decimal=5)
        # 断言 Q 的共轭转置乘以 Q 等于单位矩阵
        assert_almost_equal(dot(q.T.conj(), q), np.eye(m))
        # 断言 R 的上三角部分等于 R 自身
        assert_almost_equal(np.triu(r), r)

        # mode == 'reduced'
        # 使用简化模式进行 QR 分解，返回 Q1 和 R1
        q1, r1 = linalg.qr(a, mode="reduced")
        # 断言 Q1 和 R1 的数据类型与参数 `a` 的数据类型相同
        assert_(q1.dtype == a_dtype)
        assert_(r1.dtype == a_dtype)
        # 断言 Q1 和 R1 的类型是参数 `a` 的类型或其子类
        assert_(isinstance(q1, a_type))
        assert_(isinstance(r1, a_type))
        # 断言 Q1 的形状为 (m, k)，R1 的形状为 (k, n)，其中 k=min(m, n)
        assert_(q1.shape == (m, k))
        assert_(r1.shape == (k, n))
        # 断言 Q1*R1 等于原始矩阵 a，精确到小数点后五位
        assert_almost_equal(dot(q1, r1), a, single_decimal=5)
        # 断言 Q1 的共轭转置乘以 Q1 等于单位矩阵
        assert_almost_equal(dot(q1.T.conj(), q1), np.eye(k))
        # 断言 R1 的上三角部分等于 R1 自身
        assert_almost_equal(np.triu(r1), r1)

        # mode == 'r'
        # 使用 R 模式进行 QR 分解，只返回 R2
        r2 = linalg.qr(a, mode="r")
        # 断言 R2 的数据类型与参数 `a` 的数据类型相同
        assert_(r2.dtype == a_dtype)
        # 断言 R2 的类型是参数 `a` 的类型或其子类
        assert_(isinstance(r2, a_type))
        # 断言 R2 等于简化模式下的 R1
        assert_almost_equal(r2, r1)

    @xpassIfTorchDynamo  # (reason="torch does not allow qr(..., mode='raw'")
    @parametrize("m, n", [(3, 0), (0, 3), (0, 0)])
    # 定义一个测试函数，用于测试空矩阵的 QR 分解
    def test_qr_empty(self, m, n):
        # 取 m 和 n 中的较小值，作为 QR 分解时的维度 k
        k = min(m, n)
        # 创建一个空的 NumPy 数组 a，形状为 (m, n)
        a = np.empty((m, n))

        # 调用 check_qr 方法，检查空矩阵的 QR 分解结果
        self.check_qr(a)

        # 对空矩阵进行原始模式下的 QR 分解，返回 H 和 tau
        h, tau = np.linalg.qr(a, mode="raw")
        # 断言 H 和 tau 的数据类型为双精度浮点数
        assert_equal(h.dtype, np.double)
        assert_equal(tau.dtype, np.double)
        # 断言 H 的形状为 (n, m)，tau 的形状为 (k,)
        assert_equal(h.shape, (n, m))
        assert_equal(tau.shape, (k,))

    @xpassIfTorchDynamo  # (reason="torch does not allow qr(..., mode='raw'")
    # 定义一个测试函数，用于测试原始模式下的 QR 分解
    def test_mode_raw(self):
        # 因子分解结果在不同库之间可能不唯一，无法与已知值进行比较
        # 函数测试是一种可能性，但需要等待更多 lapack_lite 中函数的公开。
        # 因此，此测试的范围非常有限。注意结果是按 Fortran 顺序排列的，
        # 因此 H 数组是转置的。
        # 创建一个双精度浮点数类型的二维 NumPy 数组 a
        a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.double)

        # 测试双精度浮点数类型下的原始模式下的 QR 分解，返回 H 和 tau
        h, tau = linalg.qr(a, mode="raw")
        # 断言 H 和 tau 的数据类型为双精度浮点数
        assert_(h.dtype == np.double)
        assert_(tau.dtype == np.double)
        # 断言 H 的形状为 (2, 3)，tau 的形状为 (2,)
        assert_(h.shape == (2, 3))
        assert_(tau.shape == (2,))

        # 对 a 的转置进行双精度浮点数类型下的原始模式下的 QR 分解，返回 H 和 tau
        h, tau = linalg.qr(a.T, mode="raw")
        # 断言 H 和 tau 的数据类型为双精度浮点数
        assert_(h.dtype == np.double)
        assert_(tau.dtype == np.double)
        # 断言 H 的形状为 (3, 2)，tau 的形状为 (2,)
        assert_(h.shape == (3, 2))
        assert_(tau.shape == (2,))
    # 定义一个测试方法，测试所有除经济模式外的模式
    def test_mode_all_but_economic(self):
        # 创建两个二维数组
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[1, 2], [3, 4], [5, 6]])
        
        # 第一个循环，对于字符串"fd"中的每个字符进行迭代
        for dt in "fd":
            # 将数组a转换为指定数据类型dt，并赋值给m1
            m1 = a.astype(dt)
            # 将数组b转换为指定数据类型dt，并赋值给m2
            m2 = b.astype(dt)
            # 对m1进行QR分解检查
            self.check_qr(m1)
            # 对m2进行QR分解检查
            self.check_qr(m2)
            # 对m2的转置进行QR分解检查
            self.check_qr(m2.T)
        
        # 第二个循环，对于字符串"fd"中的每个字符进行迭代
        for dt in "fd":
            # 将数组a转换为复数形式（1 + 1j * a.astype(dt)），并赋值给m1
            m1 = 1 + 1j * a.astype(dt)
            # 将数组b转换为复数形式（1 + 1j * b.astype(dt)），并赋值给m2
            m2 = 1 + 1j * b.astype(dt)
            # 对m1进行QR分解检查
            self.check_qr(m1)
            # 对m2进行QR分解检查
            self.check_qr(m2)
            # 对m2的转置进行QR分解检查
            self.check_qr(m2.T)

    # 定义一个检查QR分解结果的方法，期望参数a为精确类型的ndarray或其子类
    def check_qr_stacked(self, a):
        # 获取参数a的类型
        a_type = type(a)
        # 获取参数a的数据类型
        a_dtype = a.dtype
        # 获取参数a的形状的最后两个维度大小
        m, n = a.shape[-2:]
        # 计算QR分解后的秩
        k = min(m, n)

        # mode == 'complete'，完全模式的QR分解
        q, r = linalg.qr(a, mode="complete")
        # 断言q的数据类型与a的数据类型相同
        assert_(q.dtype == a_dtype)
        # 断言r的数据类型与a的数据类型相同
        assert_(r.dtype == a_dtype)
        # 断言q是a_type的实例
        assert_(isinstance(q, a_type))
        # 断言r是a_type的实例
        assert_(isinstance(r, a_type))
        # 断言q的形状的最后两个维度为(m, m)
        assert_(q.shape[-2:] == (m, m))
        # 断言r的形状的最后两个维度为(m, n)
        assert_(r.shape[-2:] == (m, n))
        # 断言matmul(q, r)与a的乘积近似相等，精确到小数点后5位
        assert_almost_equal(matmul(q, r), a, single_decimal=5)
        # 创建单位矩阵I_mat
        I_mat = np.identity(q.shape[-1])
        # 将I_mat广播扩展到与q相同的维度
        stack_I_mat = np.broadcast_to(I_mat, q.shape[:-2] + (q.shape[-1],) * 2)
        # 断言matmul(swapaxes(q, -1, -2).conj(), q)与stack_I_mat近似相等
        assert_almost_equal(matmul(swapaxes(q, -1, -2).conj(), q), stack_I_mat)
        # 断言上三角部分r[..., :, :]与r近似相等
        assert_almost_equal(np.triu(r[..., :, :]), r)

        # mode == 'reduced'，减少模式的QR分解
        q1, r1 = linalg.qr(a, mode="reduced")
        # 断言q1的数据类型与a的数据类型相同
        assert_(q1.dtype == a_dtype)
        # 断言r1的数据类型与a的数据类型相同
        assert_(r1.dtype == a_dtype)
        # 断言q1是a_type的实例
        assert_(isinstance(q1, a_type))
        # 断言r1是a_type的实例
        assert_(isinstance(r1, a_type))
        # 断言q1的形状的最后两个维度为(m, k)
        assert_(q1.shape[-2:] == (m, k))
        # 断言r1的形状的最后两个维度为(k, n)
        assert_(r1.shape[-2:] == (k, n))
        # 断言matmul(q1, r1)与a的乘积近似相等，精确到小数点后5位
        assert_almost_equal(matmul(q1, r1), a, single_decimal=5)
        # 创建单位矩阵I_mat
        I_mat = np.identity(q1.shape[-1])
        # 将I_mat广播扩展到与q1相同的维度
        stack_I_mat = np.broadcast_to(I_mat, q1.shape[:-2] + (q1.shape[-1],) * 2)
        # 断言matmul(swapaxes(q1, -1, -2).conj(), q1)与stack_I_mat近似相等
        assert_almost_equal(matmul(swapaxes(q1, -1, -2).conj(), q1), stack_I_mat)
        # 断言上三角部分r1[..., :, :]与r1近似相等
        assert_almost_equal(np.triu(r1[..., :, :]), r1)

        # mode == 'r'，R模式的QR分解
        r2 = linalg.qr(a, mode="r")
        # 断言r2的数据类型与a的数据类型相同
        assert_(r2.dtype == a_dtype)
        # 断言r2是a_type的实例
        assert_(isinstance(r2, a_type))
        # 断言r2与r1近似相等
        assert_almost_equal(r2, r1)

    # 使用装饰器设置条件跳过测试，要求numpy版本大于等于1.22，否则跳过
    @skipif(numpy.__version__ < "1.22", reason="NP_VER: fails on CI with numpy 1.21.2")
    # 参数化测试，测试不同的尺寸和数据类型
    @parametrize("size", [(3, 4), (4, 3), (4, 4), (3, 0), (0, 3)])
    # 参数化测试，测试不同的外部尺寸
    @parametrize("outer_size", [(2, 2), (2,), (2, 3, 4)])
    # 参数化测试，测试不同的数据类型
    @parametrize("dt", [np.single, np.double, np.csingle, np.cdouble])
    # 定义一个测试堆叠输入的方法，接受外部尺寸、尺寸和数据类型作为参数
    def test_stacked_inputs(self, outer_size, size, dt):
        # 生成服从正态分布的随机数数组A，并将其转换为指定数据类型dt
        A = np.random.normal(size=outer_size + size).astype(dt)
        # 生成服从正态分布的随机数数组B，并将其转换为指定数据类型dt
        B = np.random.normal(size=outer_size + size).astype(dt)
        # 对数组A进行检查QR分解堆叠
        self.check_qr_stacked(A)
        # 对数组A + 1.0j * B进行检查QR分解堆叠
        self.check_qr_stacked(A + 1.0j * B)
@instantiate_parametrized_tests
class TestCholesky(TestCase):
    # 为 Cholesky 分解编写测试类，支持参数化测试

    @parametrize("shape", [(1, 1), (2, 2), (3, 3), (50, 50), (3, 10, 10)])
    @parametrize("dtype", (np.float32, np.float64, np.complex64, np.complex128))
    def test_basic_property(self, shape, dtype):
        # 测试基本性质：检查矩阵是否满足 A = L L^H

        np.random.seed(1)
        a = np.random.randn(*shape)
        if np.issubdtype(dtype, np.complexfloating):
            a = a + 1j * np.random.randn(*shape)

        t = list(range(len(shape)))
        t[-2:] = -1, -2

        # 转置和共轭转置操作
        a = np.matmul(a.transpose(t).conj(), a)
        a = np.asarray(a, dtype=dtype)

        # 计算 Cholesky 分解
        c = np.linalg.cholesky(a)

        # 重构原始矩阵
        b = np.matmul(c, c.transpose(t).conj())

        # 设置数值容差
        atol = 500 * a.shape[0] * np.finfo(dtype).eps

        # 断言重构后的矩阵与原始矩阵在指定容差范围内相等
        assert_allclose(b, a, atol=atol, err_msg=f"{shape} {dtype}\n{a}\n{c}")

    def test_0_size(self):
        # 测试零大小矩阵的 Cholesky 分解

        a = np.zeros((0, 1, 1), dtype=np.int_)  # 创建一个零大小的整数类型数组
        res = linalg.cholesky(a)

        # 断言结果与输入形状相同
        assert_equal(a.shape, res.shape)
        # 断言结果类型为 np.float64
        assert_(res.dtype.type is np.float64)
        # 用于文档目的：断言结果为 np.ndarray 类型
        assert_(isinstance(res, np.ndarray))

        a = np.zeros((1, 0, 0), dtype=np.complex64)  # 创建一个零大小的复数类型数组
        res = linalg.cholesky(a)

        # 断言结果与输入形状相同
        assert_equal(a.shape, res.shape)
        # 断言结果类型为 np.complex64
        assert_(res.dtype.type is np.complex64)
        # 用于文档目的：断言结果为 np.ndarray 类型
        assert_(isinstance(res, np.ndarray))


class TestMisc(TestCase):
    @xpassIfTorchDynamo  # (reason="endianness")
    def test_byteorder_check(self):
        # 检查字节顺序是否与本机顺序匹配

        if sys.byteorder == "little":
            native = "<"
        else:
            native = ">"

        for dtt in (np.float32, np.float64):
            arr = np.eye(4, dtype=dtt)
            n_arr = arr.newbyteorder(native)
            sw_arr = arr.newbyteorder("S").byteswap()

            # 检查多种线性代数例程的结果
            assert_equal(arr.dtype.byteorder, "=")
            for routine in (linalg.inv, linalg.det, linalg.pinv):
                # 正常调用
                res = routine(arr)
                # 本机字节顺序，但不是 '='
                assert_array_equal(res, routine(n_arr))
                # 字节交换
                assert_array_equal(res, routine(sw_arr))

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_generalized_raise_multiloop(self):
        # 测试通用的多重循环错误引发情况

        invertible = np.array([[1, 2], [3, 4]])
        non_invertible = np.array([[1, 1], [1, 1]])

        x = np.zeros([4, 4, 2, 2])[1::2]
        x[...] = invertible
        x[0, 0] = non_invertible

        # 断言对非可逆矩阵求逆会引发 LinAlgError
        assert_raises(np.linalg.LinAlgError, np.linalg.inv, x)
    def test_xerbla_override(self):
        # 检查我们的 xerbla 是否成功链接进来。如果没有成功，将调用默认的 xerbla 程序，
        # 默认程序会向 stdout 打印消息，并根据 LAPACK 包的具体情况中止或不中止进程。

        XERBLA_OK = 255  # 定义 xerbla 成功的返回码

        try:
            # 尝试创建子进程
            pid = os.fork()
        except (OSError, AttributeError):
            # 如果 fork 失败，或者不在 POSIX 环境下
            raise SkipTest("Not POSIX or fork failed.")  # 声明测试跳过的异常情况，标记为 B904

        if pid == 0:
            # 子进程；关闭 I/O 文件句柄
            os.close(1)
            os.close(0)
            # 避免生成核心文件
            import resource

            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            # 以下调用可能会中止进程
            try:
                np.linalg.lapack_lite.xerbla()  # 调用自定义的 xerbla 函数
            except ValueError:
                pass
            except Exception:
                os._exit(os.EX_CONFIG)

            try:
                a = np.array([[1.0]])
                np.linalg.lapack_lite.dorgqr(
                    1, 1, 1, a, 0, a, a, 0, 0
                )  # <- 无效的值
            except ValueError as e:
                if "DORGQR parameter number 5" in str(e):
                    # 成功，重用错误码表示成功，因为 FORTRAN STOP 返回也表示成功
                    os._exit(XERBLA_OK)

            # 没有中止，但我们的 xerbla 没有链接进来
            os._exit(os.EX_CONFIG)
        else:
            # 父进程
            pid, status = os.wait()
            if os.WEXITSTATUS(status) != XERBLA_OK:
                raise SkipTest("Numpy xerbla not linked in.")

    @pytest.mark.skipif(IS_WASM, reason="Cannot start subprocess")
    @slow
    def test_sdot_bug_8577(self):
        # 回归测试，确保加载某些其他库不会导致 float32 线性代数出现错误结果。
        #
        # 在 macOS 上可能会触发 gh-8577 的 bug，也可能存在其他情况会引发此问题。
        #
        # 在单独的进程中执行检查。

        bad_libs = ["PyQt5.QtWidgets", "IPython"]

        template = textwrap.dedent(
            """
        import sys
        {before}
        try:
            import {bad_lib}
        except ImportError:
            sys.exit(0)
        {after}
        x = np.ones(2, dtype=np.float32)
        sys.exit(0 if np.allclose(x.dot(x), 2.0) else 1)
        """
        )

        for bad_lib in bad_libs:
            # 使用模板创建代码并执行检查
            code = template.format(
                before="import numpy as np", after="", bad_lib=bad_lib
            )
            subprocess.check_call([sys.executable, "-c", code])

            # 交换导入顺序并执行检查
            code = template.format(
                after="import numpy as np", before="", bad_lib=bad_lib
            )
            subprocess.check_call([sys.executable, "-c", code])
class TestMultiDot(TestCase):
    def test_basic_function_with_three_arguments(self):
        # multi_dot with three arguments uses a fast hand coded algorithm to
        # determine the optimal order. Therefore test it separately.
        A = np.random.random((6, 2))   # 创建一个 6x2 的随机数组 A
        B = np.random.random((2, 6))   # 创建一个 2x6 的随机数组 B
        C = np.random.random((6, 2))   # 创建一个 6x2 的随机数组 C

        assert_almost_equal(multi_dot([A, B, C]), A.dot(B).dot(C))   # 断言 multi_dot 的结果与 A.dot(B).dot(C) 相近
        assert_almost_equal(multi_dot([A, B, C]), np.dot(A, np.dot(B, C)))   # 断言 multi_dot 的结果与 np.dot(A, np.dot(B, C)) 相近

    def test_basic_function_with_two_arguments(self):
        # separate code path with two arguments
        A = np.random.random((6, 2))   # 创建一个 6x2 的随机数组 A
        B = np.random.random((2, 6))   # 创建一个 2x6 的随机数组 B

        assert_almost_equal(multi_dot([A, B]), A.dot(B))   # 断言 multi_dot 的结果与 A.dot(B) 相近
        assert_almost_equal(multi_dot([A, B]), np.dot(A, B))   # 断言 multi_dot 的结果与 np.dot(A, B) 相近

    def test_basic_function_with_dynamic_programming_optimization(self):
        # multi_dot with four or more arguments uses the dynamic programming
        # optimization and therefore deserve a separate
        A = np.random.random((6, 2))   # 创建一个 6x2 的随机数组 A
        B = np.random.random((2, 6))   # 创建一个 2x6 的随机数组 B
        C = np.random.random((6, 2))   # 创建一个 6x2 的随机数组 C
        D = np.random.random((2, 1))   # 创建一个 2x1 的随机数组 D

        assert_almost_equal(multi_dot([A, B, C, D]), A.dot(B).dot(C).dot(D))   # 断言 multi_dot 的结果与 A.dot(B).dot(C).dot(D) 相近

    def test_vector_as_first_argument(self):
        # The first argument can be 1-D
        A1d = np.random.random(2)   # 创建一个长度为 2 的随机一维数组 A1d
        B = np.random.random((2, 6))   # 创建一个 2x6 的随机数组 B
        C = np.random.random((6, 2))   # 创建一个 6x2 的随机数组 C
        D = np.random.random((2, 2))   # 创建一个 2x2 的随机数组 D

        # the result should be 1-D
        assert_equal(multi_dot([A1d, B, C, D]).shape, (2,))   # 断言 multi_dot 的结果形状为 (2,)

    def test_vector_as_last_argument(self):
        # The last argument can be 1-D
        A = np.random.random((6, 2))   # 创建一个 6x2 的随机数组 A
        B = np.random.random((2, 6))   # 创建一个 2x6 的随机数组 B
        C = np.random.random((6, 2))   # 创建一个 6x2 的随机数组 C
        D1d = np.random.random(2)   # 创建一个长度为 2 的随机一维数组 D1d

        # the result should be 1-D
        assert_equal(multi_dot([A, B, C, D1d]).shape, (6,))   # 断言 multi_dot 的结果形状为 (6,)

    def test_vector_as_first_and_last_argument(self):
        # The first and last arguments can be 1-D
        A1d = np.random.random(2)   # 创建一个长度为 2 的随机一维数组 A1d
        B = np.random.random((2, 6))   # 创建一个 2x6 的随机数组 B
        C = np.random.random((6, 2))   # 创建一个 6x2 的随机数组 C
        D1d = np.random.random(2)   # 创建一个长度为 2 的随机一维数组 D1d

        # the result should be a scalar
        assert_equal(multi_dot([A1d, B, C, D1d]).shape, ())   # 断言 multi_dot 的结果形状为 ()

    def test_three_arguments_and_out(self):
        # multi_dot with three arguments uses a fast hand coded algorithm to
        # determine the optimal order. Therefore test it separately.
        A = np.random.random((6, 2))   # 创建一个 6x2 的随机数组 A
        B = np.random.random((2, 6))   # 创建一个 2x6 的随机数组 B
        C = np.random.random((6, 2))   # 创建一个 6x2 的随机数组 C

        out = np.zeros((6, 2))   # 创建一个全零的 6x2 数组 out
        ret = multi_dot([A, B, C], out=out)   # 调用 multi_dot，将结果存入 out
        assert out is ret   # 断言 out 和返回值 ret 是同一个对象
        assert_almost_equal(out, A.dot(B).dot(C))   # 断言 out 与 A.dot(B).dot(C) 相近
        assert_almost_equal(out, np.dot(A, np.dot(B, C)))   # 断言 out 与 np.dot(A, np.dot(B, C)) 相近
    def test_two_arguments_and_out(self):
        # 创建随机矩阵 A 和 B，分别为 6x2 和 2x6
        A = np.random.random((6, 2))
        B = np.random.random((2, 6))
        # 创建一个全零的输出矩阵 out，大小为 6x6
        out = np.zeros((6, 6))
        # 使用 multi_dot 计算 A 和 B 的乘积，并将结果存入 out 中
        ret = multi_dot([A, B], out=out)
        # 断言 out 和 ret 是同一个对象
        assert out is ret
        # 断言 out 和 A.dot(B) 几乎相等
        assert_almost_equal(out, A.dot(B))
        # 断言 out 和 np.dot(A, B) 几乎相等
        assert_almost_equal(out, np.dot(A, B))

    def test_dynamic_programming_optimization_and_out(self):
        # multi_dot 使用四个或更多参数时，采用动态规划优化，需要单独测试
        A = np.random.random((6, 2))
        B = np.random.random((2, 6))
        C = np.random.random((6, 2))
        D = np.random.random((2, 1))
        # 创建一个全零的输出矩阵 out，大小为 6x1
        out = np.zeros((6, 1))
        # 使用 multi_dot 计算 A, B, C, D 的乘积，并将结果存入 out 中
        ret = multi_dot([A, B, C, D], out=out)
        # 断言 out 和 ret 是同一个对象
        assert out is ret
        # 断言 out 和 A.dot(B).dot(C).dot(D) 几乎相等
        assert_almost_equal(out, A.dot(B).dot(C).dot(D))

    def test_dynamic_programming_logic(self):
        # 测试动态规划部分
        # 此测试直接来自 Cormen 的第 376 页。
        # 创建包含多个随机矩阵的数组
        arrays = [
            np.random.random((30, 35)),
            np.random.random((35, 15)),
            np.random.random((15, 5)),
            np.random.random((5, 10)),
            np.random.random((10, 20)),
            np.random.random((20, 25)),
        ]
        # 期望的最优化乘积结果矩阵 m_expected
        m_expected = np.array(
            [
                [0.0, 15750.0, 7875.0, 9375.0, 11875.0, 15125.0],
                [0.0, 0.0, 2625.0, 4375.0, 7125.0, 10500.0],
                [0.0, 0.0, 0.0, 750.0, 2500.0, 5375.0],
                [0.0, 0.0, 0.0, 0.0, 1000.0, 3500.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 5000.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        # 期望的分割点矩阵 s_expected
        s_expected = np.array(
            [
                [0, 1, 1, 3, 3, 3],
                [0, 0, 2, 3, 3, 3],
                [0, 0, 0, 3, 3, 3],
                [0, 0, 0, 0, 4, 5],
                [0, 0, 0, 0, 0, 5],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=int,
        )
        # 调整 s_expected，因为 Cormen 使用基于 1 的索引，Python 使用基于 0 的索引
        s_expected -= 1  

        # 调用 _multi_dot_matrix_chain_order 函数计算结果
        s, m = _multi_dot_matrix_chain_order(arrays, return_costs=True)

        # 断言 s 的上三角部分（不包括对角线）与期望的 s_expected 上三角部分几乎相等
        assert_almost_equal(np.triu(s[:-1, 1:]), np.triu(s_expected[:-1, 1:]))
        # 断言 m 和期望的 m_expected 的上三角部分几乎相等
        assert_almost_equal(np.triu(m), np.triu(m_expected))

    def test_too_few_input_arrays(self):
        # 断言当输入数组数量过少时，multi_dot 函数会引发 RuntimeError 或 ValueError 异常
        assert_raises((RuntimeError, ValueError), multi_dot, [])
        assert_raises((RuntimeError, ValueError), multi_dot, [np.random.random((3, 3))])
@instantiate_parametrized_tests
class TestTensorinv(TestCase):
    @parametrize(
        "arr, ind",
        [
            (np.ones((4, 6, 8, 2)), 2),
            (np.ones((3, 3, 2)), 1),
        ],
    )
    def test_non_square_handling(self, arr, ind):
        # 断言在调用 linalg.tensorinv 函数时，期望引发 LinAlgError 或 RuntimeError 异常
        with assert_raises((LinAlgError, RuntimeError)):
            linalg.tensorinv(arr, ind=ind)

    @parametrize(
        "shape, ind",
        [
            # 来自文档字符串的示例
            ((4, 6, 8, 3), 2),
            ((24, 8, 3), 1),
        ],
    )
    def test_tensorinv_shape(self, shape, ind):
        # 创建一个单位矩阵并根据给定的形状重塑它
        a = np.eye(24).reshape(shape)
        # 调用 linalg.tensorinv 函数，计算矩阵的逆
        ainv = linalg.tensorinv(a=a, ind=ind)
        # 计算预期的逆矩阵形状
        expected = a.shape[ind:] + a.shape[:ind]
        # 获取实际计算得到的逆矩阵形状
        actual = ainv.shape
        # 断言实际的逆矩阵形状与预期形状相等
        assert_equal(actual, expected)

    @parametrize(
        "ind",
        [
            0,
            -2,
        ],
    )
    def test_tensorinv_ind_limit(self, ind):
        # 创建一个单位矩阵并根据给定的形状重塑它
        a = np.eye(24).reshape(4, 6, 8, 3)
        # 断言在调用 linalg.tensorinv 函数时，期望引发 ValueError 或 RuntimeError 异常
        with assert_raises((ValueError, RuntimeError)):
            linalg.tensorinv(a=a, ind=ind)

    def test_tensorinv_result(self):
        # 模拟一个文档字符串中的示例
        a = np.eye(24).reshape(24, 8, 3)
        # 调用 linalg.tensorinv 函数，计算矩阵的逆
        ainv = linalg.tensorinv(a, ind=1)
        # 创建一个全为1的向量
        b = np.ones(24)
        # 使用 np.tensordot 函数检查矩阵逆与原始矩阵的矩阵方程解是否接近
        assert_allclose(np.tensordot(ainv, b, 1), np.linalg.tensorsolve(a, b))


@instantiate_parametrized_tests
class TestTensorsolve(TestCase):
    @parametrize(
        "a, axes",
        [
            (np.ones((4, 6, 8, 2)), None),
            (np.ones((3, 3, 2)), (0, 2)),
        ],
    )
    def test_non_square_handling(self, a, axes):
        # 断言在调用 linalg.tensorsolve 函数时，期望引发 LinAlgError 或 RuntimeError 异常
        with assert_raises((LinAlgError, RuntimeError)):
            b = np.ones(a.shape[:2])
            linalg.tensorsolve(a, b, axes=axes)

    @skipif(numpy.__version__ < "1.22", reason="NP_VER: fails on CI with numpy 1.21.2")
    @parametrize(
        "shape",
        [(2, 3, 6), (3, 4, 4, 3), (0, 3, 3, 0)],
    )
    def test_tensorsolve_result(self, shape):
        # 创建一个具有随机值的数组
        a = np.random.randn(*shape)
        b = np.ones(a.shape[:2])
        # 调用 np.linalg.tensorsolve 函数，计算张量方程的解
        x = np.linalg.tensorsolve(a, b)
        # 使用 np.tensordot 函数检查张量方程的解是否正确
        assert_allclose(np.tensordot(a, x, axes=len(x.shape)), b)


class TestMisc2(TestCase):
    @xpassIfTorchDynamo  # (reason="TODO")
    def test_unsupported_commontype(self):
        # linalg 优雅地处理不支持的数据类型
        arr = np.array([[1, -2], [2, 5]], dtype="float16")
        # 断言在调用 linalg.cholesky 函数时，期望引发 TypeError 异常
        with assert_raises(TypeError):
            linalg.cholesky(arr)

    # @slow
    # @pytest.mark.xfail(not HAS_LAPACK64, run=False,
    #                   reason="Numpy not compiled with 64-bit BLAS/LAPACK")
    # @requires_memory(free_bytes=16e9)
    @skip(reason="Bad memory reports lead to OOM in ci testing")
    def test_blas64_dot(self):
        # 创建一个非常大的零数组
        n = 2**32
        a = np.zeros([1, n], dtype=np.float32)
        b = np.ones([1, 1], dtype=np.float32)
        a[0, -1] = 1
        # 执行大规模的矩阵乘法运算
        c = np.dot(b, a)
        # 断言最后一个元素的值是否为1
        assert_equal(c[0, -1], 1)

    @skip(reason="lapack-lite specific")
    @xfail  # 标记为预期失败的测试用例，因为没有使用64位的BLAS/LAPACK支持
    #    not HAS_LAPACK64, reason="Numpy not compiled with 64-bit BLAS/LAPACK"
    # )
    
    def test_blas64_geqrf_lwork_smoketest(self):
        # 对 LAPACK 中的 geqrf 函数使用64位整数进行烟雾测试
        dtype = np.float64
        lapack_routine = np.linalg.lapack_lite.dgeqrf
    
        m = 2**32 + 1  # 设置矩阵的行数 m，略大于 2^32
        n = 2**32 + 1  # 设置矩阵的列数 n，略大于 2^32
        lda = m
    
        # 创建虚拟数组，虽然不会被 LAPACK 函数引用，所以大小不需要完全匹配
        a = np.zeros([1, 1], dtype=dtype)
        work = np.zeros([1], dtype=dtype)
        tau = np.zeros([1], dtype=dtype)
    
        # 查询所需工作空间大小
        results = lapack_routine(m, n, a, lda, tau, work, -1, 0)
        assert_equal(results["info"], 0)  # 断言结果信息为 0，表示成功执行
        assert_equal(results["m"], m)     # 断言返回的行数 m 正确
        assert_equal(results["n"], m)     # 断言返回的列数 n 正确
    
        # 工作空间大小应当是一个合理的整数
        lwork = int(work.item())
        assert_(2**32 < lwork < 2**42)   # 断言工作空间大小在 2^32 和 2^42 之间
# 如果这个脚本被直接执行（而不是被导入为模块），则执行以下代码
if __name__ == "__main__":
    # 调用名为 run_tests 的函数来运行测试
    run_tests()
```