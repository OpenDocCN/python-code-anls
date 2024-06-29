# `.\numpy\numpy\_core\tests\test_umath.py`

```
# 导入标准库和第三方库
import platform             # 导入 platform 库，用于获取平台信息
import warnings             # 导入 warnings 库，用于警告处理
import fnmatch              # 导入 fnmatch 库，用于 Unix 文件名匹配
import itertools            # 导入 itertools 库，用于迭代操作
import pytest               # 导入 pytest 库，用于编写和运行测试用例
import sys                  # 导入 sys 库，用于系统相关的参数和函数
import os                   # 导入 os 库，提供了操作系统相关的功能
import operator             # 导入 operator 模块，提供了一系列对应 Python 操作符的函数
from fractions import Fraction  # 导入 Fraction 类，用于处理有理数
from functools import reduce  # 导入 reduce 函数，用于对列表进行累积计算
from collections import namedtuple  # 导入 namedtuple，用于创建命名元组

# 导入 NumPy 相关库
import numpy._core.umath as ncu  # 导入 NumPy 核心的数学函数
from numpy._core import _umath_tests as ncu_tests, sctypes  # 导入 NumPy 的内部测试函数和类型定义
import numpy as np              # 导入 NumPy 库
from numpy.testing import (     # 导入 NumPy 测试模块的多个断言函数
    assert_, assert_equal, assert_raises, assert_raises_regex,
    assert_array_equal, assert_almost_equal, assert_array_almost_equal,
    assert_array_max_ulp, assert_allclose, assert_no_warnings, suppress_warnings,
    _gen_alignment_data, assert_array_almost_equal_nulp, IS_WASM, IS_MUSL,
    IS_PYPY, HAS_REFCOUNT
)

from numpy.testing._private.utils import _glibc_older_than  # 导入 NumPy 测试私有模块中的一个函数

# 获取所有的 NumPy 通用函数 (ufuncs)
UFUNCS = [obj for obj in np._core.umath.__dict__.values()
         if isinstance(obj, np.ufunc)]

# 将 NumPy 一元通用函数筛选出来
UFUNCS_UNARY = [
    uf for uf in UFUNCS if uf.nin == 1
]

# 进一步筛选出 NumPy 浮点数一元通用函数
UFUNCS_UNARY_FP = [
    uf for uf in UFUNCS_UNARY if 'f->f' in uf.types
]

# 将 NumPy 二元通用函数筛选出来
UFUNCS_BINARY = [
    uf for uf in UFUNCS if uf.nin == 2
]

# 筛选出具有累积属性的 NumPy 二元通用函数
UFUNCS_BINARY_ACC = [
    uf for uf in UFUNCS_BINARY if hasattr(uf, "accumulate") and uf.nout == 1
]

def interesting_binop_operands(val1, val2, dtype):
    """
    Helper to create "interesting" operands to cover common code paths:
    * scalar inputs
    * only first "values" is an array (e.g. scalar division fast-paths)
    * Longer array (SIMD) placing the value of interest at different positions
    * Oddly strided arrays which may not be SIMD compatible

    It does not attempt to cover unaligned access or mixed dtypes.
    These are normally handled by the casting/buffering machinery.

    This is not a fixture (currently), since I believe a fixture normally
    only yields once?
    """
    fill_value = 1  # 可能是一个参数，但可能不是一个可选参数

    # 创建长度为 10003 的数组，并填充为指定的值和数据类型
    arr1 = np.full(10003, dtype=dtype, fill_value=fill_value)
    arr2 = np.full(10003, dtype=dtype, fill_value=fill_value)

    arr1[0] = val1  # 在数组中的第一个位置设置值 val1
    arr2[0] = val2  # 在数组中的第一个位置设置值 val2

    # 定义一个结果提取函数
    extractor = lambda res: res
    # 返回包含不同情况下的操作数及其描述的生成器对象
    yield arr1[0], arr2[0], extractor, "scalars"

    # 重新定义结果提取函数
    extractor = lambda res: res
    # 返回包含不同情况下的操作数及其描述的生成器对象
    yield arr1[0, ...], arr2[0, ...], extractor, "scalar-arrays"

    # 将数组值重置为 fill_value
    arr1[0] = fill_value
    arr2[0] = fill_value

    # 遍历指定的位置列表
    for pos in [0, 1, 2, 3, 4, 5, -1, -2, -3, -4]:
        arr1[pos] = val1  # 在指定位置设置值 val1
        arr2[pos] = val2  # 在指定位置设置值 val2

        # 定义一个结果提取函数
        extractor = lambda res: res[pos]
        # 返回包含不同情况下的操作数及其描述的生成器对象
        yield arr1, arr2, extractor, f"off-{pos}"
        yield arr1, arr2[pos], extractor, f"off-{pos}-with-scalar"

        # 将数组值重置为 fill_value
        arr1[pos] = fill_value
        arr2[pos] = fill_value

    # 遍历指定的步长列表
    for stride in [-1, 113]:
        op1 = arr1[::stride]
        op2 = arr2[::stride]
        op1[10] = val1  # 在指定位置设置值 val1
        op2[10] = val2  # 在指定位置设置值 val2

        # 定义一个结果提取函数
        extractor = lambda res: res[10]
        # 返回包含不同情况下的操作数及其描述的生成器对象
        yield op1, op2, extractor, f"stride-{stride}"

        # 将数组值重置为 fill_value
        op1[10] = fill_value
        op2[10] = fill_value


def on_powerpc():
    """ True if we are running on a Power PC platform."""
    # 检查当前平台的处理器是否为 'powerpc'，或者是否机器类型以 'ppc' 开头
    return platform.processor() == 'powerpc' or \
           platform.machine().startswith('ppc')
# 定义函数 bad_arcsinh，用于检查不同平台上的 arcsinh 函数的精度问题
def bad_arcsinh():
    # 检查当前平台是否为 aarch64
    if platform.machine() == 'aarch64':
        # 如果是 aarch64，设置 x 的值
        x = 1.78e-10
    elif on_powerpc():
        # 如果是 PowerPC 平台，设置 x 的值
        x = 2.16e-10
    else:
        # 如果不是以上两种平台，返回 False
        return False
    
    # 计算 np.arcsinh(np.float128(x)) 的值
    v1 = np.arcsinh(np.float128(x))
    # 计算 np.arcsinh(np.complex256(x)).real 的值
    v2 = np.arcsinh(np.complex256(x)).real
    
    # 比较 v1 和 v2 的绝对误差是否大于给定阈值
    # The eps for float128 is 1-e33, so this is way bigger
    return abs((v1 / v2) - 1.0) > 1e-23


# 定义一个类 _FilterInvalids 用于处理无效值的过滤器设置和恢复
class _FilterInvalids:
    # 设置方法，在测试前设置 np 的错误处理方式，将无效值设为忽略
    def setup_method(self):
        self.olderr = np.seterr(invalid='ignore')

    # 拆卸方法，在测试后恢复 np 的错误处理方式
    def teardown_method(self):
        np.seterr(**self.olderr)


# 定义一个测试类 TestConstants，用于测试数学常数
class TestConstants:
    # 测试 pi 值是否接近预期值
    def test_pi(self):
        assert_allclose(ncu.pi, 3.141592653589793, 1e-15)

    # 测试 e 值是否接近预期值
    def test_e(self):
        assert_allclose(ncu.e, 2.718281828459045, 1e-15)

    # 测试 Euler's gamma 值是否接近预期值
    def test_euler_gamma(self):
        assert_allclose(ncu.euler_gamma, 0.5772156649015329, 1e-15)
    # 定义一个测试方法 `test_out_subok`，用于测试 `np.add` 和 `np.frexp` 函数的输出参数功能
    def test_out_subok(self):
        # 对于 `subok` 参数分别为 True 和 False 两种情况进行测试
        for subok in (True, False):
            # 创建一个包含单个浮点数 0.5 的 NumPy 数组 `a`
            a = np.array(0.5)
            # 创建一个空的 NumPy 数组 `o`
            o = np.empty(())

            # 使用 `np.add` 函数，将 `a` 和标量 2 相加，将结果存储在 `o` 中，指定 `subok` 参数
            r = np.add(a, 2, o, subok=subok)
            # 断言 `r` 和 `o` 是同一个对象
            assert_(r is o)
            
            # 再次使用 `np.add` 函数，将 `a` 和标量 2 相加，将结果存储在 `o` 中，指定 `out` 和 `subok` 参数
            r = np.add(a, 2, out=o, subok=subok)
            # 断言 `r` 和 `o` 是同一个对象
            assert_(r is o)
            
            # 继续使用 `np.add` 函数，将 `a` 和标量 2 相加，将结果存储在元组 `(o,)` 中的第一个元素中，指定 `out` 和 `subok` 参数
            r = np.add(a, 2, out=(o,), subok=subok)
            # 断言 `r` 和 `o` 是同一个对象
            assert_(r is o)

            # 创建一个包含单个浮点数 5.7 的 NumPy 数组 `d`
            d = np.array(5.7)
            # 创建一个空的 NumPy 数组 `o1`
            o1 = np.empty(())
            # 创建一个空的 NumPy 数组 `o2`，指定数据类型为 `np.int32`
            o2 = np.empty((), dtype=np.int32)

            # 使用 `np.frexp` 函数，对 `d` 进行浮点数表示的分解，将结果存储在 `o1` 和 `None` 中，指定 `subok` 参数
            r1, r2 = np.frexp(d, o1, None, subok=subok)
            # 断言 `r1` 和 `o1` 是同一个对象
            assert_(r1 is o1)
            
            # 再次使用 `np.frexp` 函数，对 `d` 进行浮点数表示的分解，将结果存储在 `None` 和 `o2` 中，指定 `subok` 参数
            r1, r2 = np.frexp(d, None, o2, subok=subok)
            # 断言 `r2` 和 `o2` 是同一个对象
            assert_(r2 is o2)
            
            # 继续使用 `np.frexp` 函数，对 `d` 进行浮点数表示的分解，将结果存储在 `o1` 和 `o2` 中，指定 `subok` 参数
            r1, r2 = np.frexp(d, o1, o2, subok=subok)
            # 断言 `r1` 和 `o1` 是同一个对象
            assert_(r1 is o1)
            # 断言 `r2` 和 `o2` 是同一个对象
            assert_(r2 is o2)

            # 使用 `np.frexp` 函数，对 `d` 进行浮点数表示的分解，将结果存储在元组 `(o1, None)` 中的第一个元素中，指定 `out` 和 `subok` 参数
            r1, r2 = np.frexp(d, out=(o1, None), subok=subok)
            # 断言 `r1` 和 `o1` 是同一个对象
            assert_(r1 is o1)
            
            # 再次使用 `np.frexp` 函数，对 `d` 进行浮点数表示的分解，将结果存储在元组 `(None, o2)` 中的第二个元素中，指定 `out` 和 `subok` 参数
            r1, r2 = np.frexp(d, out=(None, o2), subok=subok)
            # 断言 `r2` 和 `o2` 是同一个对象
            assert_(r2 is o2)
            
            # 继续使用 `np.frexp` 函数，对 `d` 进行浮点数表示的分解，将结果存储在元组 `(o1, o2)` 中，指定 `out` 和 `subok` 参数
            r1, r2 = np.frexp(d, out=(o1, o2), subok=subok)
            # 断言 `r1` 和 `o1` 是同一个对象
            assert_(r1 is o1)
            # 断言 `r2` 和 `o2` 是同一个对象
            assert_(r2 is o2)

            # 使用 `assert_raises` 上下文管理器，断言 `np.frexp` 函数在传递单个输出参数 `o1` 时会抛出 `TypeError` 异常
            with assert_raises(TypeError):
                r1, r2 = np.frexp(d, out=o1, subok=subok)

            # 使用 `assert_raises` 函数，断言 `np.add` 函数在传递多个输出参数 `o, o` 时会抛出 `TypeError` 异常
            assert_raises(TypeError, np.add, a, 2, o, o, subok=subok)
            # 使用 `assert_raises` 函数，断言 `np.add` 函数在传递多个输出参数 `o, out=o` 时会抛出 `TypeError` 异常
            assert_raises(TypeError, np.add, a, 2, o, out=o, subok=subok)
            # 使用 `assert_raises` 函数，断言 `np.add` 函数在传递多个输出参数 `None, out=o` 时会抛出 `TypeError` 异常
            assert_raises(TypeError, np.add, a, 2, None, out=o, subok=subok)
            # 使用 `assert_raises` 函数，断言 `np.add` 函数在传递多个输出参数 `(o, o)` 时会抛出 `ValueError` 异常
            assert_raises(ValueError, np.add, a, 2, out=(o, o), subok=subok)
            # 使用 `assert_raises` 函数，断言 `np.add` 函数在传递多个输出参数 `()` 时会抛出 `ValueError` 异常
            assert_raises(ValueError, np.add, a, 2, out=(), subok=subok)
            # 使用 `assert_raises` 函数，断言 `np.add` 函数在传递多个输出参数 `[]` 时会抛出 `TypeError` 异常
            assert_raises(TypeError, np.add, a, 2, [], subok=subok)
            # 使用 `assert_raises` 函数，断言 `np.add` 函数在传递多个输出参数 `out=[]` 时会抛出 `TypeError` 异常
            assert_raises(TypeError, np.add, a, 2, out=[], subok=subok)
            # 使用 `assert_raises` 函数，断言 `np.add` 函数在传递多个输出参数 `out=([],)` 时会抛出 `TypeError` 异常
            assert_raises(TypeError, np.add, a, 2, out=([],), subok=subok)
            
            # 设置 `o` 的标志 `writeable` 为 `False`
            o.flags.writeable = False
            # 使用 `assert_raises` 函数，断言 `np.add` 函数在输出参数 `o` 不可写时会抛出 `ValueError` 异常
            assert_raises(ValueError, np.add, a, 2, o, subok=subok)
            # 使用 `assert_raises` 函数，断言 `np.add` 函数在输出参数 `out=o` 不可写时会抛出 `ValueError` 异常
            assert_raises(ValueError, np.add, a, 2, out=o, subok=subok)
            # 使用 `assert_raises` 函数，断言 `np.add` 函数在输出参数 `out=(o,)` 中的 `o` 不可写时会抛出 `ValueError` 异常
            assert_raises(ValueError, np.add, a, 2, out=(o,), subok=subok)
    # 定义测试函数 test_out_wrap_subok，用于测试带有自定义行为的子类 ArrayWrap
    def test_out_wrap_subok(self):
        # 定义一个数组包装类 ArrayWrap，继承自 np.ndarray
        class ArrayWrap(np.ndarray):
            # 设置数组包装类的优先级
            __array_priority__ = 10

            # 构造函数，将输入数组 arr 转换为 ArrayWrap 类型的实例并复制其内容
            def __new__(cls, arr):
                return np.asarray(arr).view(cls).copy()

            # 定义数组包装类的 __array_wrap__ 方法，以自定义的方式处理数组操作的返回值
            def __array_wrap__(self, arr, context=None, return_scalar=False):
                return arr.view(type(self))

        # 循环遍历 subok 取值为 True 和 False
        for subok in (True, False):
            # 创建 ArrayWrap 实例 a，包装 [0.5]
            a = ArrayWrap([0.5])

            # 使用 np.add 进行数组加法操作，测试 subok 参数的影响
            r = np.add(a, 2, subok=subok)
            # 如果 subok 为 True，断言返回值 r 类型为 ArrayWrap
            if subok:
                assert_(isinstance(r, ArrayWrap))
            else:
                # 如果 subok 为 False，断言返回值 r 类型为 np.ndarray
                assert_(type(r) == np.ndarray)

            # 同上，使用 np.add 进行数组加法操作，测试 None 参数和 subok 的影响
            r = np.add(a, 2, None, subok=subok)
            if subok:
                assert_(isinstance(r, ArrayWrap))
            else:
                assert_(type(r) == np.ndarray)

            # 同上，使用 np.add 进行数组加法操作，测试 out=None 和 subok 的影响
            r = np.add(a, 2, out=None, subok=subok)
            if subok:
                assert_(isinstance(r, ArrayWrap))
            else:
                assert_(type(r) == np.ndarray)

            # 同上，使用 np.add 进行数组加法操作，测试 out=(None,) 和 subok 的影响
            r = np.add(a, 2, out=(None,), subok=subok)
            if subok:
                assert_(isinstance(r, ArrayWrap))
            else:
                assert_(type(r) == np.ndarray)

            # 创建 ArrayWrap 实例 d，包装 [5.7]
            d = ArrayWrap([5.7])
            # 创建 np.ndarray 实例 o1 和 o2
            o1 = np.empty((1,))
            o2 = np.empty((1,), dtype=np.int32)

            # 使用 np.frexp 对 d 进行浮点数拆分操作，测试 subok 参数的影响
            r1, r2 = np.frexp(d, o1, subok=subok)
            if subok:
                assert_(isinstance(r2, ArrayWrap))
            else:
                assert_(type(r2) == np.ndarray)

            # 同上，使用 np.frexp 对 d 进行浮点数拆分操作，测试 None 参数和 subok 的影响
            r1, r2 = np.frexp(d, o1, None, subok=subok)
            if subok:
                assert_(isinstance(r2, ArrayWrap))
            else:
                assert_(type(r2) == np.ndarray)

            # 同上，使用 np.frexp 对 d 进行浮点数拆分操作，测试 None 和 o2 参数的影响
            r1, r2 = np.frexp(d, None, o2, subok=subok)
            if subok:
                assert_(isinstance(r1, ArrayWrap))
            else:
                assert_(type(r1) == np.ndarray)

            # 同上，使用 np.frexp 对 d 进行浮点数拆分操作，测试 out=(o1, None) 和 subok 的影响
            r1, r2 = np.frexp(d, out=(o1, None), subok=subok)
            if subok:
                assert_(isinstance(r2, ArrayWrap))
            else:
                assert_(type(r2) == np.ndarray)

            # 同上，使用 np.frexp 对 d 进行浮点数拆分操作，测试 out=(None, o2) 和 subok 的影响
            r1, r2 = np.frexp(d, out=(None, o2), subok=subok)
            if subok:
                assert_(isinstance(r1, ArrayWrap))
            else:
                assert_(type(r1) == np.ndarray)

            # 使用 assert_raises 断言捕获 TypeError 异常，测试对于非 tuple 类型的 out 参数的处理
            with assert_raises(TypeError):
                # 运行 np.frexp，其中 out 参数为非 tuple 类型，预期抛出 TypeError 异常
                r1, r2 = np.frexp(d, out=o1, subok=subok)
# 定义一个测试类 TestComparisons，用于测试比较函数的行为
class TestComparisons:
    # 导入 operator 模块，用于比较操作符的测试
    import operator

    # 使用 pytest 的参数化装饰器，测试不同数据类型的比较
    @pytest.mark.parametrize('dtype', sctypes['uint'] + sctypes['int'] +
                             sctypes['float'] + [np.bool])
    # 参数化测试比较函数，比较 Python 内建操作符与 NumPy 操作函数
    @pytest.mark.parametrize('py_comp,np_comp', [
        (operator.lt, np.less),
        (operator.le, np.less_equal),
        (operator.gt, np.greater),
        (operator.ge, np.greater_equal),
        (operator.eq, np.equal),
        (operator.ne, np.not_equal)
    ])
    # 定义测试比较函数的方法
    def test_comparison_functions(self, dtype, py_comp, np_comp):
        # 初始化输入数组
        if dtype == np.bool:
            a = np.random.choice(a=[False, True], size=1000)
            b = np.random.choice(a=[False, True], size=1000)
            scalar = True
        else:
            a = np.random.randint(low=1, high=10, size=1000).astype(dtype)
            b = np.random.randint(low=1, high=10, size=1000).astype(dtype)
            scalar = 5
        
        # 使用 NumPy 类型创建标量
        np_scalar = np.dtype(dtype).type(scalar)
        # 转换数组为列表
        a_lst = a.tolist()
        b_lst = b.tolist()

        # (Binary) Comparison (x1=array, x2=array)
        # 进行二进制比较，并将结果视图转换为无符号整数列表
        comp_b = np_comp(a, b).view(np.uint8)
        comp_b_list = [int(py_comp(x, y)) for x, y in zip(a_lst, b_lst)]

        # (Scalar1) Comparison (x1=scalar, x2=array)
        # 进行标量与数组的比较，并将结果视图转换为无符号整数列表
        comp_s1 = np_comp(np_scalar, b).view(np.uint8)
        comp_s1_list = [int(py_comp(scalar, x)) for x in b_lst]

        # (Scalar2) Comparison (x1=array, x2=scalar)
        # 进行数组与标量的比较，并将结果视图转换为无符号整数列表
        comp_s2 = np_comp(a, np_scalar).view(np.uint8)
        comp_s2_list = [int(py_comp(x, scalar)) for x in a_lst]

        # 检查三种比较的结果是否符合预期
        assert_(comp_b.tolist() == comp_b_list,
            f"Failed comparison ({py_comp.__name__})")
        assert_(comp_s1.tolist() == comp_s1_list,
            f"Failed comparison ({py_comp.__name__})")
        assert_(comp_s2.tolist() == comp_s2_list,
            f"Failed comparison ({py_comp.__name__})")

    # 测试在相等比较中忽略对象的身份
    def test_ignore_object_identity_in_equal(self):
        # 检查比较相同对象时抛出异常，例如数组的元素逐个比较
        a = np.array([np.array([1, 2, 3]), None], dtype=object)
        assert_raises(ValueError, np.equal, a, a)

        # 检查比较不可比较的相同对象时抛出异常
        class FunkyType:
            def __eq__(self, other):
                raise TypeError("I won't compare")

        a = np.array([FunkyType()])
        assert_raises(TypeError, np.equal, a, a)

        # 检查 NaN 的身份不会覆盖比较不匹配
        a = np.array([np.nan], dtype=object)
        assert_equal(np.equal(a, a), [False])
    def test_ignore_object_identity_in_not_equal(self):
        # 检查对比较复杂的对象进行比较时，如逐元素比较的数组。
        a = np.array([np.array([1, 2, 3]), None], dtype=object)
        # 断言应该抛出 ValueError 异常，因为比较的对象是相同的。
        assert_raises(ValueError, np.not_equal, a, a)

        # 检查当比较不可比较的相同对象时是否抛出错误。
        class FunkyType:
            def __ne__(self, other):
                raise TypeError("I won't compare")

        a = np.array([FunkyType()])
        # 断言应该抛出 TypeError 异常，因为比较的对象不可比较。
        assert_raises(TypeError, np.not_equal, a, a)

        # 检查即使是相同的对象，其身份也不会覆盖比较不匹配。
        a = np.array([np.nan], dtype=object)
        # 断言 np.not_equal 返回 [True]，即使比较的对象相同也应该如此。
        assert_equal(np.not_equal(a, a), [True])

    def test_error_in_equal_reduce(self):
        # gh-20929
        # 确保当传递未指定 dtype 的数组给 np.equal.reduce 时会引发 TypeError。
        a = np.array([0, 0])
        assert_equal(np.equal.reduce(a, dtype=bool), True)
        assert_raises(TypeError, np.equal.reduce, a)

    def test_object_dtype(self):
        # 断言 np.equal 的结果 dtype 应该是 object。
        assert np.equal(1, [1], dtype=object).dtype == object
        # 使用 signature 参数检查 np.equal 的结果 dtype 是否为 object。
        assert np.equal(1, [1], signature=(None, None, "O")).dtype == object

    def test_object_nonbool_dtype_error(self):
        # bool 输出 dtype 是正常的：
        assert np.equal(1, [1], dtype=bool).dtype == bool

        # 但以下示例没有匹配的循环：
        with pytest.raises(TypeError, match="No loop matching"):
            np.equal(1, 1, dtype=np.int64)

        with pytest.raises(TypeError, match="No loop matching"):
            np.equal(1, 1, sig=(None, None, "l"))

    @pytest.mark.parametrize("dtypes", ["qQ", "Qq"])
    @pytest.mark.parametrize('py_comp, np_comp', [
        (operator.lt, np.less),
        (operator.le, np.less_equal),
        (operator.gt, np.greater),
        (operator.ge, np.greater_equal),
        (operator.eq, np.equal),
        (operator.ne, np.not_equal)
    ])
    @pytest.mark.parametrize("vals", [(2**60, 2**60+1), (2**60+1, 2**60)])
    def test_large_integer_direct_comparison(
            self, dtypes, py_comp, np_comp, vals):
        # 注意 float(2**60) + 1 == float(2**60)。
        a1 = np.array([2**60], dtype=dtypes[0])
        a2 = np.array([2**60 + 1], dtype=dtypes[1])
        expected = py_comp(2**60, 2**60+1)

        assert py_comp(a1, a2) == expected
        assert np_comp(a1, a2) == expected
        # 还要检查标量：
        s1 = a1[0]
        s2 = a2[0]
        # 断言 s1 和 s2 是 np.integer 类型。
        assert isinstance(s1, np.integer)
        assert isinstance(s2, np.integer)
        # Python 操作符在这里是主要的兴趣点：
        assert py_comp(s1, s2) == expected
        assert np_comp(s1, s2) == expected

    @pytest.mark.parametrize("dtype", np.typecodes['UnsignedInteger'])
    # 使用 pytest 的参数化功能，定义了多组测试参数，每组参数包含一个 Python 操作符和一个 NumPy 操作符
    @pytest.mark.parametrize('py_comp_func, np_comp_func', [
        (operator.lt, np.less),
        (operator.le, np.less_equal),
        (operator.gt, np.greater),
        (operator.ge, np.greater_equal),
        (operator.eq, np.equal),
        (operator.ne, np.not_equal)
    ])
    # 再次使用 pytest 的参数化功能，为 flip 参数定义了两个测试用例：True 和 False
    @pytest.mark.parametrize("flip", [True, False])
    # 定义了一个测试方法，用于测试无符号和有符号整数的直接比较
    def test_unsigned_signed_direct_comparison(
            self, dtype, py_comp_func, np_comp_func, flip):
        # 根据 flip 参数的值，选择性地反转 py_comp_func 和 np_comp_func 的参数位置
        if flip:
            py_comp = lambda x, y: py_comp_func(y, x)
            np_comp = lambda x, y: np_comp_func(y, x)
        else:
            py_comp = py_comp_func
            np_comp = np_comp_func

        # 创建一个包含单个最大值的 NumPy 数组，使用指定的 dtype 类型
        arr = np.array([np.iinfo(dtype).max], dtype=dtype)
        # 计算预期结果，用 Python 操作符对数组的最大值和 -1 进行比较
        expected = py_comp(int(arr[0]), -1)

        # 断言语句：使用 Python 操作符和 NumPy 操作符对数组和 -1 进行比较，验证结果是否符合预期
        assert py_comp(arr, -1) == expected
        assert np_comp(arr, -1) == expected

        # 获取数组的第一个元素作为标量
        scalar = arr[0]
        # 断言语句：验证标量是否为 NumPy 的整数类型
        assert isinstance(scalar, np.integer)
        # 断言语句：再次使用 Python 操作符对标量和 -1 进行比较，验证结果是否符合预期
        # 这里主要关注 Python 操作符的使用
        assert py_comp(scalar, -1) == expected
        assert np_comp(scalar, -1) == expected
```python`
class TestAdd:
    def test_reduce_alignment(self):
        # 测试用例标识为 gh-9876
        # 确保具有奇怪步幅的数组在 pairwise_sum_@TYPE@ 优化中能正常工作。
        # 在 x86 上，'b' 字段将以 4 字节偏移量对齐，尽管其 itemsize 是 8。
        # 创建一个包含两个元素的零数组，数据类型为 [('a', np.int32), ('b', np.float64)]
        a = np.zeros(2, dtype=[('a', np.int32), ('b', np.float64)])
        # 将 'a' 字段全部设为 -1
        a['a'] = -1
        # 断言 'b' 字段的总和为 0
        assert_equal(a['b'].sum(), 0)


class TestDivision:
    def test_division_int(self):
        # 整数除法应遵循 Python 的规则
        # 创建一个包含各种整数的数组
        x = np.array([5, 10, 90, 100, -5, -10, -90, -100, -120])
        # 如果 5 / 10 等于 0.5，则断言 x / 100 的结果符合预期
        if 5 / 10 == 0.5:
            assert_equal(x / 100, [0.05, 0.1, 0.9, 1,
                                   -0.05, -0.1, -0.9, -1, -1.2])
        else:
            # 否则，断言 x / 100 的结果应为 [0, 0, 0, 1, -1, -1, -1, -1, -2]
            assert_equal(x / 100, [0, 0, 0, 1, -1, -1, -1, -1, -2])
        # 断言 x // 100 的结果应为 [0, 0, 0, 1, -1, -1, -1, -1, -2]
        assert_equal(x // 100, [0, 0, 0, 1, -1, -1, -1, -1, -2])
        # 断言 x % 100 的结果应为 [5, 10, 90, 0, 95, 90, 10, 0, 80]

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize("dtype,ex_val", itertools.product(
        sctypes['int'] + sctypes['uint'], (
            (
                # 被除数
                "np.array(range(fo.max-lsize, fo.max)).astype(dtype),"
                # 除数
                "np.arange(lsize).astype(dtype),"
                # 标量除数
                "range(15)"
            ),
            (
                # 被除数
                "np.arange(fo.min, fo.min+lsize).astype(dtype),"
                # 除数
                "np.arange(lsize//-2, lsize//2).astype(dtype),"
                # 标量除数
                "range(fo.min, fo.min + 15)"
            ), (
                # 被除数
                "np.array(range(fo.max-lsize, fo.max)).astype(dtype),"
                # 除数
                "np.arange(lsize).astype(dtype),"
                # 标量除数
                "[1,3,9,13astype(dtype),"
                # 注释：除数数组
                "np.arange(lsize).astype(dtype),"
                # 注释：标量除数
                "[1,3,9,13,neg, fo.min+1, fo.min//2, fo.max//3, fo.max//4]"
            )
        )
    ))
    # 定义一个测试方法，用于测试整数除法的边界情况
    def test_division_int_boundary(self, dtype, ex_val):
        # 获取指定数据类型的整数信息
        fo = np.iinfo(dtype)
        # 确定负数的符号，如果fo.min小于0则为-1，否则为1
        neg = -1 if fo.min < 0 else 1
        # 设置一个足够大的数组大小，以测试SIMD循环和余数元素
        lsize = 512 + 7
        # 解析传入的测试数据，分别为数组a、b和除数列表divisors
        a, b, divisors = eval(ex_val)
        # 将数组a和b转换为列表形式
        a_lst, b_lst = a.tolist(), b.tolist()

        # 定义一个匿名函数c_div，用于计算整数除法，处理除数为0或特定边界情况
        c_div = lambda n, d: (
            0 if d == 0 else (
                fo.min if (n and n == fo.min and d == -1) else n//d
            )
        )

        # 在处理除零错误时，忽略除法相关的错误
        with np.errstate(divide='ignore'):
            # 复制数组a，对其执行原地整除运算
            ac = a.copy()
            ac //= b
            # 计算a与b的整数除法结果
            div_ab = a // b

        # 使用列表推导式计算a_lst与b_lst的每个元素的整数除法结果，存储在div_lst中
        div_lst = [c_div(x, y) for x, y in zip(a_lst, b_lst)]

        # 断言整数数组的整数除法结果与预期的div_lst相等
        msg = "Integer arrays floor division check (//)"
        assert all(div_ab == div_lst), msg
        # 断言原地整数除法的结果与预期的div_lst相等
        msg_eq = "Integer arrays floor division check (//=)"
        assert all(ac == div_lst), msg_eq

        # 遍历除数列表，对数组a执行整数除法操作，再次验证结果与预期相等
        for divisor in divisors:
            ac = a.copy()
            with np.errstate(divide='ignore', over='ignore'):
                div_a = a // divisor
                ac //= divisor
            div_lst = [c_div(i, divisor) for i in a_lst]

            # 断言整数数组的整数除法结果与预期的div_lst相等
            assert all(div_a == div_lst), msg
            # 断言原地整数除法的结果与预期的div_lst相等
            assert all(ac == div_lst), msg_eq

        # 在处理除零或溢出错误时，引发特定异常
        with np.errstate(divide='raise', over='raise'):
            if 0 in b:
                # 如果b中包含0，则验证除零错误情况
                with pytest.raises(FloatingPointError,
                        match="divide by zero encountered in floor_divide"):
                    a // b
            else:
                a // b

            if fo.min and fo.min in a:
                # 如果fo.min不为0且在a中，则验证溢出错误情况
                with pytest.raises(FloatingPointError,
                        match='overflow encountered in floor_divide'):
                    a // -1
            elif fo.min:
                a // -1

            # 验证除零错误情况
            with pytest.raises(FloatingPointError,
                    match="divide by zero encountered in floor_divide"):
                a // 0

            # 验证原地除零错误情况
            with pytest.raises(FloatingPointError,
                    match="divide by zero encountered in floor_divide"):
                ac = a.copy()
                ac //= 0

            # 创建一个空数组，并验证其与零的整数除法
            np.array([], dtype=dtype) // 0
    # 定义一个测试方法，用于测试整数除法的缩减操作，传入数据类型和表达式值
    def test_division_int_reduce(self, dtype, ex_val):
        # 获取给定数据类型的整数信息对象
        fo = np.iinfo(dtype)
        # 使用给定表达式值计算得到数组 a
        a = eval(ex_val)
        # 将数组 a 转换为列表形式
        lst = a.tolist()
        
        # 定义一个 lambda 函数 c_div，用于整数除法，处理特殊情况
        c_div = lambda n, d: (
            0 if d == 0 or (n and n == fo.min and d == -1) else n//d
        )

        # 忽略除法警告，计算数组 a 的缩减操作结果 div_a
        with np.errstate(divide='ignore'):
            div_a = np.floor_divide.reduce(a)
        
        # 使用 reduce 函数和自定义的 c_div 函数计算列表 lst 的缩减操作结果 div_lst
        div_lst = reduce(c_div, lst)
        
        # 设置断言消息
        msg = "Reduce floor integer division check"
        
        # 断言 div_a 和 div_lst 相等，如果不相等则抛出断言错误
        assert div_a == div_lst, msg

        # 恢复默认的除法错误处理方式，测试浮点错误的情况
        with np.errstate(divide='raise', over='raise'):
            # 检查在数组范围内的整数除法操作，是否会引发浮点错误
            with pytest.raises(FloatingPointError,
                    match="divide by zero encountered in reduce"):
                np.floor_divide.reduce(np.arange(-100, 100).astype(dtype))
            
            # 如果数据类型的最小值不为零，再次测试是否会引发浮点错误
            if fo.min:
                with pytest.raises(FloatingPointError,
                        match='overflow encountered in reduce'):
                    np.floor_divide.reduce(
                        np.array([fo.min, 1, -1], dtype=dtype)
                    )

    @pytest.mark.parametrize(
            "dividend,divisor,quotient",
            [(np.timedelta64(2,'Y'), np.timedelta64(2,'M'), 12),
             (np.timedelta64(2,'Y'), np.timedelta64(-2,'M'), -12),
             (np.timedelta64(-2,'Y'), np.timedelta64(2,'M'), -12),
             (np.timedelta64(-2,'Y'), np.timedelta64(-2,'M'), 12),
             (np.timedelta64(2,'M'), np.timedelta64(-2,'Y'), -1),
             (np.timedelta64(2,'Y'), np.timedelta64(0,'M'), 0),
             (np.timedelta64(2,'Y'), 2, np.timedelta64(1,'Y')),
             (np.timedelta64(2,'Y'), -2, np.timedelta64(-1,'Y')),
             (np.timedelta64(-2,'Y'), 2, np.timedelta64(-1,'Y')),
             (np.timedelta64(-2,'Y'), -2, np.timedelta64(1,'Y')),
             (np.timedelta64(-2,'Y'), -2, np.timedelta64(1,'Y')),
             (np.timedelta64(-2,'Y'), -3, np.timedelta64(0,'Y')),
             (np.timedelta64(-2,'Y'), 0, np.timedelta64('Nat','Y')),
            ])
    # 定义一个测试方法，用于测试时间增量的整数除法操作
    def test_division_int_timedelta(self, dividend, divisor, quotient):
        # 如果除数不为零且商不是 NaN 时间，进行除法测试
        if divisor and (isinstance(quotient, int) or not np.isnat(quotient)):
            # 设置断言消息
            msg = "Timedelta floor division check"
            
            # 断言时间增量 dividend 除以 divisor 的结果等于 quotient
            assert dividend // divisor == quotient, msg

            # 进一步测试数组形式的时间增量除法
            msg = "Timedelta arrays floor division check"
            dividend_array = np.array([dividend]*5)
            quotient_array = np.array([quotient]*5)
            # 断言数组形式的时间增量除法结果与预期相符
            assert all(dividend_array // divisor == quotient_array), msg
        else:
            # 如果是在 WASM 环境下，跳过浮点错误测试
            if IS_WASM:
                pytest.skip("fp errors don't work in wasm")
            # 恢复默认的除法错误处理方式，测试除数为零的情况
            with np.errstate(divide='raise', invalid='raise'):
                with pytest.raises(FloatingPointError):
                    dividend // divisor
    # 定义一个测试函数，用于测试复数除法
    def test_division_complex(self):
        # 设置测试的错误消息
        msg = "Complex division implementation check"
        # 创建一个复数类型的 NumPy 数组，用于测试
        x = np.array([1. + 1.*1j, 1. + .5*1j, 1. + 2.*1j], dtype=np.complex128)
        # 断言复数的平方除以自身应该等于自身，用于检查复数除法实现的正确性
        assert_almost_equal(x**2/x, x, err_msg=msg)
        
        # 检查复数除法可能出现的溢出和下溢情况
        msg = "Complex division overflow/underflow check"
        # 创建一个复数类型的 NumPy 数组，用于测试溢出和下溢
        x = np.array([1.e+110, 1.e-110], dtype=np.complex128)
        y = x**2/x
        # 断言除以自身应该得到 1，用于检查溢出和下溢情况
        assert_almost_equal(y/x, [1, 1], err_msg=msg)

    # 定义一个测试函数，用于测试复数除以零的情况
    def test_zero_division_complex(self):
        # 忽略复数除以零的警告和错误
        with np.errstate(invalid="ignore", divide="ignore"):
            # 创建一个复数类型的 NumPy 数组，测试除以零的各种情况
            x = np.array([0.0], dtype=np.complex128)
            y = 1.0/x
            # 断言除以零得到无穷大
            assert_(np.isinf(y)[0])
            y = complex(np.inf, np.nan)/x
            assert_(np.isinf(y)[0])
            y = complex(np.nan, np.inf)/x
            assert_(np.isinf(y)[0])
            y = complex(np.inf, np.inf)/x
            assert_(np.isinf(y)[0])
            y = 0.0/x
            # 断言除以零得到 NaN
            assert_(np.isnan(y)[0])

    # 定义一个测试函数，用于测试复数的地板除法
    def test_floor_division_complex(self):
        # 检查地板除法、divmod 和取余操作是否会引发类型错误
        x = np.array([.9 + 1j, -.1 + 1j, .9 + .5*1j, .9 + 2.*1j], dtype=np.complex128)
        with pytest.raises(TypeError):
            x // 7
        with pytest.raises(TypeError):
            np.divmod(x, 7)
        with pytest.raises(TypeError):
            np.remainder(x, 7)

    # 定义一个测试函数，用于测试地板除法中的带符号零
    def test_floor_division_signed_zero(self):
        # 检查当正零和负零除以一时，符号位是否设置正确
        x = np.zeros(10)
        assert_equal(np.signbit(x//1), 0)
        assert_equal(np.signbit((-x)//1), 1)

    # 根据条件标记跳过测试
    @pytest.mark.skipif(hasattr(np.__config__, "blas_ssl2_info"),
            reason="gh-22982")
    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize('dtype', np.typecodes['Float'])
    # 定义一个测试函数，用于测试地板除法中的错误情况
    def test_floor_division_errors(self, dtype):
        # 创建各种类型的 NumPy 数组，用于测试地板除法中的错误情况
        fnan = np.array(np.nan, dtype=dtype)
        fone = np.array(1.0, dtype=dtype)
        fzer = np.array(0.0, dtype=dtype)
        finf = np.array(np.inf, dtype=dtype)
        
        # 检查除以零是否会引发浮点数错误
        with np.errstate(divide='raise', invalid='ignore'):
            assert_raises(FloatingPointError, np.floor_divide, fone, fzer)
        with np.errstate(divide='ignore', invalid='raise'):
            np.floor_divide(fone, fzer)

        # 下面的操作已经包含 NaN，不应该警告
        with np.errstate(all='raise'):
            np.floor_divide(fnan, fone)
            np.floor_divide(fone, fnan)
            np.floor_divide(fnan, fzer)
            np.floor_divide(fzer, fnan)

    # 根据参数化设定，使用不同类型的浮点数进行测试
    @pytest.mark.parametrize('dtype', np.typecodes['Float'])
    # 定义一个测试函数，用于测试整除操作的边界情况
    def test_floor_division_corner_cases(self, dtype):
        # 创建一个长度为10的零数组，指定数据类型为参数 dtype
        x = np.zeros(10, dtype=dtype)
        # 创建一个长度为10的全一数组，指定数据类型为参数 dtype
        y = np.ones(10, dtype=dtype)
        # 创建一个包含 NaN 的数组，指定数据类型为参数 dtype
        fnan = np.array(np.nan, dtype=dtype)
        # 创建一个包含浮点数 1.0 的数组，指定数据类型为参数 dtype
        fone = np.array(1.0, dtype=dtype)
        # 创建一个包含浮点数 0.0 的数组，指定数据类型为参数 dtype
        fzer = np.array(0.0, dtype=dtype)
        # 创建一个包含正无穷大的数组，指定数据类型为参数 dtype
        finf = np.array(np.inf, dtype=dtype)
        # 使用 suppress_warnings 上下文管理器，过滤特定的运行时警告信息
        with suppress_warnings() as sup:
            # 过滤在 floor_divide 操作中遇到的无效值警告
            sup.filter(RuntimeWarning, "invalid value encountered in floor_divide")
            # 对 NaN 除以 1.0 进行 floor_divide 操作，预期结果应为 NaN
            div = np.floor_divide(fnan, fone)
            assert(np.isnan(div)), "dt: %s, div: %s" % (dt, div)
            # 对 1.0 除以 NaN 进行 floor_divide 操作，预期结果应为 NaN
            div = np.floor_divide(fone, fnan)
            assert(np.isnan(div)), "dt: %s, div: %s" % (dt, div)
            # 对 NaN 除以 0.0 进行 floor_divide 操作，预期结果应为 NaN
            div = np.floor_divide(fnan, fzer)
            assert(np.isnan(div)), "dt: %s, div: %s" % (dt, div)
        # 在忽略除法运算中的除以零警告的情况下，验证 1.0 除以 0.0 的 floor_divide 操作结果是否为正无穷
        with np.errstate(divide='ignore'):
            z = np.floor_divide(y, x)
            assert_(np.isinf(z).all())
# 定义一个函数，用于对两个数执行 floor division 和 remainder 操作，返回结果的元组
def floor_divide_and_remainder(x, y):
    return (np.floor_divide(x, y), np.remainder(x, y))


# 根据数据类型判断是否为无符号整数，返回相应的符号元组
def _signs(dt):
    if dt in np.typecodes['UnsignedInteger']:
        return (+1,)
    else:
        return (+1, -1)


# 定义一个测试类 TestRemainder
class TestRemainder:

    # 测试基本的 remainder 操作
    def test_remainder_basic(self):
        # 定义需要测试的数据类型组合
        dt = np.typecodes['AllInteger'] + np.typecodes['Float']
        # 遍历操作函数和数据类型的笛卡尔积
        for op in [floor_divide_and_remainder, np.divmod]:
            for dt1, dt2 in itertools.product(dt, dt):
                # 遍历数据类型的符号组合
                for sg1, sg2 in itertools.product(_signs(dt1), _signs(dt2)):
                    # 格式化测试消息
                    fmt = 'op: %s, dt1: %s, dt2: %s, sg1: %s, sg2: %s'
                    msg = fmt % (op.__name__, dt1, dt2, sg1, sg2)
                    # 创建两个数组 a 和 b，分别代表两个操作数
                    a = np.array(sg1*71, dtype=dt1)
                    b = np.array(sg2*19, dtype=dt2)
                    # 执行操作并获取结果
                    div, rem = op(a, b)
                    # 断言操作结果满足等式 div*b + rem == a
                    assert_equal(div*b + rem, a, err_msg=msg)
                    # 对于 sg2 为 -1 的情况，断言 rem 在区间 (b, 0] 中
                    if sg2 == -1:
                        assert_(b < rem <= 0, msg)
                    else:
                        assert_(b > rem >= 0, msg)

    # 测试浮点数的 exact remainder 操作
    def test_float_remainder_exact(self):
        # 定义测试用的小整数列表
        nlst = list(range(-127, 0))
        plst = list(range(1, 128))
        dividend = nlst + [0] + plst
        divisor = nlst + plst
        arg = list(itertools.product(dividend, divisor))
        tgt = list(divmod(*t) for t in arg)

        # 将参数转换为数组
        a, b = np.array(arg, dtype=int).T
        # 将 Python 的 exact integer 结果转换为浮点数以便处理 signed zero
        tgtdiv, tgtrem = np.array(tgt, dtype=float).T
        tgtdiv = np.where((tgtdiv == 0.0) & ((b < 0) ^ (a < 0)), -0.0, tgtdiv)
        tgtrem = np.where((tgtrem == 0.0) & (b < 0), -0.0, tgtrem)

        # 遍历操作函数和浮点数数据类型
        for op in [floor_divide_and_remainder, np.divmod]:
            for dt in np.typecodes['Float']:
                msg = 'op: %s, dtype: %s' % (op.__name__, dt)
                # 将数组 a 和 b 转换为指定数据类型的浮点数
                fa = a.astype(dt)
                fb = b.astype(dt)
                # 执行操作并获取结果
                div, rem = op(fa, fb)
                # 断言操作结果与预期结果相等
                assert_equal(div, tgtdiv, err_msg=msg)
                assert_equal(rem, tgtrem, err_msg=msg)
    # 定义测试方法，用于测试浮点数的余数和舍入
    def test_float_remainder_roundoff(self):
        # 标识问题编号为 gh-6127
        dt = np.typecodes['Float']
        # 遍历操作函数列表，包括 floor_divide_and_remainder 和 np.divmod
        for op in [floor_divide_and_remainder, np.divmod]:
            # 遍历浮点数类型组合，使用 itertools.product 进行排列组合
            for dt1, dt2 in itertools.product(dt, dt):
                # 遍历符号组合，+1 和 -1
                for sg1, sg2 in itertools.product((+1, -1), (+1, -1)):
                    # 格式化消息字符串，显示操作函数名、浮点数类型、符号信息
                    fmt = 'op: %s, dt1: %s, dt2: %s, sg1: %s, sg2: %s'
                    msg = fmt % (op.__name__, dt1, dt2, sg1, sg2)
                    # 创建数组 a 和 b，分别使用指定的浮点数类型和符号
                    a = np.array(sg1*78*6e-8, dtype=dt1)
                    b = np.array(sg2*6e-8, dtype=dt2)
                    # 调用操作函数，获取除法结果和余数
                    div, rem = op(a, b)
                    # 使用 assert_equal 断言，验证等式 div*b + rem == a 是否成立，显示错误消息 msg
                    assert_equal(div*b + rem, a, err_msg=msg)
                    # 如果 sg2 为 -1，则额外断言 b < rem <= 0，显示消息 msg
                    if sg2 == -1:
                        assert_(b < rem <= 0, msg)
                    else:
                        assert_(b > rem >= 0, msg)

    # 使用 pytest.mark.skipif 标记，条件是 IS_WASM 为真，原因是 wasm 平台上的浮点数错误不起作用
    # 同时使用 pytest.mark.xfail 标记，对 macOS 平台做标记，因为它似乎不能正确显示 `fmod` 的 'invalid' 警告
    # 参数化测试方法，对于每种浮点数类型进行测试
    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.xfail(sys.platform.startswith("darwin"),
            reason="MacOS seems to not give the correct 'invalid' warning for "
                   "`fmod`.  Hopefully, others always do.")
    @pytest.mark.parametrize('dtype', np.typecodes['Float'])
    def test_float_divmod_errors(self, dtype):
        # 检查 divmod 和 remainder 是否引发有效的错误
        fzero = np.array(0.0, dtype=dtype)
        fone = np.array(1.0, dtype=dtype)
        finf = np.array(np.inf, dtype=dtype)
        fnan = np.array(np.nan, dtype=dtype)
        # 使用 np.errstate 设置浮点数错误状态
        # 第一个断言：使用 np.divmod 计算 fone / fzero 时应引发 FloatingPointError
        with np.errstate(divide='raise', invalid='ignore'):
            assert_raises(FloatingPointError, np.divmod, fone, fzero)
        # 第二个断言：使用 np.divmod 计算 fone / fzero 时应引发 FloatingPointError
        with np.errstate(divide='ignore', invalid='raise'):
            assert_raises(FloatingPointError, np.divmod, fone, fzero)
        # 第三个断言：使用 np.divmod 计算 fzero / fzero 时应引发 FloatingPointError
        with np.errstate(invalid='raise'):
            assert_raises(FloatingPointError, np.divmod, fzero, fzero)
        # 第四个断言：使用 np.divmod 计算 finf / finf 时应引发 FloatingPointError
        with np.errstate(invalid='raise'):
            assert_raises(FloatingPointError, np.divmod, finf, finf)
        # 第五个断言：使用 np.divmod 计算 finf / fzero 时应引发 FloatingPointError
        with np.errstate(divide='ignore', invalid='raise'):
            assert_raises(FloatingPointError, np.divmod, finf, fzero)
        # 第六个断言：np.divmod(finf, fzero) 时，inf / 0 不会设置任何标志，只有模数会创建 NaN
        with np.errstate(divide='raise', invalid='ignore'):
            np.divmod(finf, fzero)

    # 使用 pytest.mark.skipif 标记，条件是 np.__config__ 中存在 "blas_ssl2_info" 属性，原因是问题 gh-22982
    # 使用 pytest.mark.skipif 标记，条件是 IS_WASM 为真，原因是 wasm 平台上的浮点数错误不起作用
    # 同时使用 pytest.mark.xfail 标记，对 macOS 平台做标记，因为它似乎不能正确显示 `fmod` 的 'invalid' 警告
    # 参数化测试方法，对于每种浮点数类型以及 np.fmod 和 np.remainder 两个函数进行测试
    @pytest.mark.skipif(hasattr(np.__config__, "blas_ssl2_info"),
            reason="gh-22982")
    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.xfail(sys.platform.startswith("darwin"),
           reason="MacOS seems to not give the correct 'invalid' warning for "
                  "`fmod`.  Hopefully, others always do.")
    @pytest.mark.parametrize('dtype', np.typecodes['Float'])
    @pytest.mark.parametrize('fn', [np.fmod, np.remainder])
    def test_float_divmod_corner_cases(self):
        # 针对浮点数除法和取模函数的边界情况进行测试

        # 遍历所有浮点数类型
        for dt in np.typecodes['Float']:
            fnan = np.array(np.nan, dtype=dt)
            fone = np.array(1.0, dtype=dt)
            fzer = np.array(0.0, dtype=dt)
            finf = np.array(np.inf, dtype=dt)

            # 忽略特定警告信息
            with suppress_warnings() as sup:
                sup.filter(RuntimeWarning, "invalid value encountered in divmod")
                sup.filter(RuntimeWarning, "divide by zero encountered in divmod")

                # 测试 np.divmod 函数的行为
                div, rem = np.divmod(fone, fzer)
                assert(np.isinf(div)), 'dt: %s, div: %s' % (dt, rem)
                assert(np.isnan(rem)), 'dt: %s, rem: %s' % (dt, rem)

                div, rem = np.divmod(fzer, fzer)
                assert(np.isnan(rem)), 'dt: %s, rem: %s' % (dt, rem)
                assert_(np.isnan(div)), 'dt: %s, rem: %s' % (dt, rem)

                div, rem = np.divmod(finf, finf)
                assert(np.isnan(div)), 'dt: %s, rem: %s' % (dt, rem)
                assert(np.isnan(rem)), 'dt: %s, rem: %s' % (dt, rem)

                div, rem = np.divmod(finf, fzer)
                assert(np.isinf(div)), 'dt: %s, rem: %s' % (dt, rem)
                assert(np.isnan(rem)), 'dt: %s, rem: %s' % (dt, rem)

                div, rem = np.divmod(fnan, fone)
                assert(np.isnan(rem)), "dt: %s, rem: %s" % (dt, rem)
                assert(np.isnan(div)), "dt: %s, rem: %s" % (dt, rem)

                div, rem = np.divmod(fone, fnan)
                assert(np.isnan(rem)), "dt: %s, rem: %s" % (dt, rem)
                assert(np.isnan(div)), "dt: %s, rem: %s" % (dt, rem)

                div, rem = np.divmod(fnan, fzer)
                assert(np.isnan(rem)), "dt: %s, rem: %s" % (dt, rem)
                assert(np.isnan(div)), "dt: %s, rem: %s" % (dt, rem)
    def test_float_remainder_corner_cases(self):
        # 检查浮点数取余的边缘情况。

        # 对于每种浮点数类型进行迭代检查
        for dt in np.typecodes['Float']:
            # 创建值为 1.0 的浮点数组，指定数据类型为 dt
            fone = np.array(1.0, dtype=dt)
            # 创建值为 0.0 的浮点数组，指定数据类型为 dt
            fzer = np.array(0.0, dtype=dt)
            # 创建 NaN（非数字）的浮点数组，指定数据类型为 dt
            fnan = np.array(np.nan, dtype=dt)
            # 创建值为 1.0 的浮点数组，指定数据类型为 dt
            b = np.array(1.0, dtype=dt)
            # 创建尽量接近 0.0 且小于 0.0 的浮点数数组，指定数据类型为 dt
            a = np.nextafter(np.array(0.0, dtype=dt), -b)
            # 计算 a 对 b 的取余
            rem = np.remainder(a, b)
            # 断言余数 rem 小于等于 b，如果不成立则抛出异常，附带数据类型信息 dt
            assert_(rem <= b, 'dt: %s' % dt)
            # 计算 -a 对 -b 的取余
            rem = np.remainder(-a, -b)
            # 断言余数 rem 大于等于 -b，如果不成立则抛出异常，附带数据类型信息 dt
            assert_(rem >= -b, 'dt: %s' % dt)

        # 检查 NaN 和无穷大的情况
        with suppress_warnings() as sup:
            # 忽略特定的运行时警告，如遇到取余操作中的无效值或浮点模操作中的无效值
            sup.filter(RuntimeWarning, "invalid value encountered in remainder")
            sup.filter(RuntimeWarning, "invalid value encountered in fmod")
            # 对于每种浮点数类型进行迭代检查
            for dt in np.typecodes['Float']:
                # 创建值为 1.0 的浮点数组，指定数据类型为 dt
                fone = np.array(1.0, dtype=dt)
                # 创建值为 0.0 的浮点数组，指定数据类型为 dt
                fzer = np.array(0.0, dtype=dt)
                # 创建正无穷大的浮点数组，指定数据类型为 dt
                finf = np.array(np.inf, dtype=dt)
                # 创建 NaN（非数字）的浮点数组，指定数据类型为 dt
                fnan = np.array(np.nan, dtype=dt)
                # 计算 fone 对 fzer 的取余
                rem = np.remainder(fone, fzer)
                # 断言余数 rem 是 NaN，如果不成立则抛出异常，附带数据类型信息 dt 和余数值 rem
                assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
                # MSVC 2008 在这里返回 NaN，因此禁用此检查。
                #rem = np.remainder(fone, finf)
                #assert_(rem == fone, 'dt: %s, rem: %s' % (dt, rem))
                # 计算 finf 对 fone 的取余
                rem = np.remainder(finf, fone)
                # 计算 finf 对 fone 的浮点模操作
                fmod = np.fmod(finf, fone)
                # 断言 fmod 是 NaN，如果不成立则抛出异常，附带数据类型信息 dt 和模值 fmod
                assert_(np.isnan(fmod), 'dt: %s, fmod: %s' % (dt, fmod))
                # 断言 rem 是 NaN，如果不成立则抛出异常，附带数据类型信息 dt 和余数值 rem
                assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
                # 计算 finf 对 finf 的取余
                rem = np.remainder(finf, finf)
                # 计算 finf 对 fone 的浮点模操作
                fmod = np.fmod(finf, fone)
                # 断言 rem 是 NaN，如果不成立则抛出异常，附带数据类型信息 dt 和余数值 rem
                assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
                # 断言 fmod 是 NaN，如果不成立则抛出异常，附带数据类型信息 dt 和模值 fmod
                assert_(np.isnan(fmod), 'dt: %s, fmod: %s' % (dt, fmod))
                # 计算 finf 对 fzer 的取余
                rem = np.remainder(finf, fzer)
                # 计算 finf 对 fzer 的浮点模操作
                fmod = np.fmod(finf, fzer)
                # 断言 rem 是 NaN，如果不成立则抛出异常，附带数据类型信息 dt 和余数值 rem
                assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
                # 断言 fmod 是 NaN，如果不成立则抛出异常，附带数据类型信息 dt 和模值 fmod
                assert_(np.isnan(fmod), 'dt: %s, fmod: %s' % (dt, fmod))
                # 计算 fone 对 fnan 的取余
                rem = np.remainder(fone, fnan)
                # 计算 fone 对 fnan 的浮点模操作
                fmod = np.fmod(fone, fnan)
                # 断言 rem 是 NaN，如果不成立则抛出异常，附带数据类型信息 dt 和余数值 rem
                assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
                # 断言 fmod 是 NaN，如果不成立则抛出异常，附带数据类型信息 dt 和模值 fmod
                assert_(np.isnan(fmod), 'dt: %s, fmod: %s' % (dt, fmod))
                # 计算 fnan 对 fzer 的取余
                rem = np.remainder(fnan, fzer)
                # 计算 fnan 对 fzer 的浮点模操作
                fmod = np.fmod(fnan, fzer)
                # 断言 rem 是 NaN，如果不成立则抛出异常，附带数据类型信息 dt 和余数值 rem
                assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
                # 断言 fmod 是 NaN，如果不成立则抛出异常，附带数据类型信息 dt 和模值 fmod
                assert_(np.isnan(fmod), 'dt: %s, fmod: %s' % (dt, rem))
                # 计算 fnan 对 fone 的取余
                rem = np.remainder(fnan, fone)
                # 计算 fnan 对 fone 的浮点模操作
                fmod = np.fmod(fnan, fone)
                # 断言 rem 是 NaN，如果不成立则抛出异常，附带数据类型信息 dt 和余数值 rem
                assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
                # 断言 fmod 是 NaN，如果不成立则抛出异常，附带数据类型信息 dt 和模值 fmod
                assert_(np.isnan(fmod), 'dt: %s, fmod: %s' % (dt, rem))
# 定义一个测试类，用于测试整数溢出和除以零的情况
class TestDivisionIntegerOverflowsAndDivideByZero:
    # 定义一个命名元组 result_type，包含两个字段：nocast 和 casted
    result_type = namedtuple('result_type',
            ['nocast', 'casted'])

    # 定义一个包含不同 lambda 函数的字典，用于生成特定数值
    helper_lambdas = {
        'zero': lambda dtype: 0,
        'min': lambda dtype: np.iinfo(dtype).min,
        'neg_min': lambda dtype: -np.iinfo(dtype).min,
        'min-zero': lambda dtype: (np.iinfo(dtype).min, 0),
        'neg_min-zero': lambda dtype: (-np.iinfo(dtype).min, 0),
    }

    # 定义一个包含各种算术操作和对应结果的字典
    overflow_results = {
        np.remainder: result_type(
            helper_lambdas['zero'], helper_lambdas['zero']),
        np.fmod: result_type(
            helper_lambdas['zero'], helper_lambdas['zero']),
        operator.mod: result_type(
            helper_lambdas['zero'], helper_lambdas['zero']),
        operator.floordiv: result_type(
            helper_lambdas['min'], helper_lambdas['neg_min']),
        np.floor_divide: result_type(
            helper_lambdas['min'], helper_lambdas['neg_min']),
        np.divmod: result_type(
            helper_lambdas['min-zero'], helper_lambdas['neg_min-zero'])
    }

    # 使用 pytest.mark.skipif 装饰器标记条件跳过的测试用例（当 IS_WASM 为真时跳过）
    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    # 使用 pytest.mark.parametrize 标记参数化测试用例，参数为 np.typecodes["Integer"] 中的数据类型
    def test_signed_division_overflow(self, dtype):
        # 获取一组感兴趣的操作数对，用于测试
        to_check = interesting_binop_operands(np.iinfo(dtype).min, -1, dtype)
        # 对每一组操作数进行测试
        for op1, op2, extractor, operand_identifier in to_check:
            # 使用 pytest.warns 检查是否会发出 RuntimeWarning 警告，匹配 "overflow encountered"
            with pytest.warns(RuntimeWarning, match="overflow encountered"):
                res = op1 // op2

            # 断言结果的数据类型与操作数 op1 的数据类型相同
            assert res.dtype == op1.dtype
            # 断言提取的结果与操作数 op1 的最小值相同
            assert extractor(res) == np.iinfo(op1.dtype).min

            # 对取余操作进行测试，不应发出警告
            res = op1 % op2
            assert res.dtype == op1.dtype
            assert extractor(res) == 0

            # 对 np.fmod 函数进行测试
            res = np.fmod(op1, op2)
            assert extractor(res) == 0

            # 对 divmod 函数进行测试，需要发出警告
            with pytest.warns(RuntimeWarning, match="overflow encountered"):
                res1, res2 = np.divmod(op1, op2)

            # 断言结果的数据类型与操作数 op1 和 op2 的数据类型相同
            assert res1.dtype == res2.dtype == op1.dtype
            # 断言提取的结果与操作数 op1 的最小值相同
            assert extractor(res1) == np.iinfo(op1.dtype).min
            assert extractor(res2) == 0

    # 使用 pytest.mark.skipif 装饰器标记条件跳过的测试用例（当 IS_WASM 为真时跳过）
    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    # 使用 pytest.mark.parametrize 标记参数化测试用例，参数为 np.typecodes["AllInteger"] 中的数据类型
    def test_signed_division(self, dtype):
    # 定义一个测试方法，用于测试在除以零时的行为
    def test_divide_by_zero(self, dtype):
        # 注意：这里返回值的定义并不明确，但是 NumPy 目前一致使用 0。这可能会改变。
        
        # 获取一组感兴趣的操作数对，其中包含了除法运算的操作数
        to_check = interesting_binop_operands(1, 0, dtype)
        
        # 遍历每对操作数及其相关的信息
        for op1, op2, extractor, operand_identifier in to_check:
            # 使用 pytest 捕获 RuntimeWarning，匹配 "divide by zero"
            with pytest.warns(RuntimeWarning, match="divide by zero"):
                res = op1 // op2  # 执行整数除法操作

            # 断言结果的数据类型与 op1 的数据类型一致
            assert res.dtype == op1.dtype
            # 断言提取的结果为 0，即整数除法的结果应为 0
            assert extractor(res) == 0

            # 再次使用 pytest 捕获 RuntimeWarning，匹配 "divide by zero"
            with pytest.warns(RuntimeWarning, match="divide by zero"):
                res1, res2 = np.divmod(op1, op2)  # 执行 divmod 操作

            # 断言结果 res1 和 res2 的数据类型与 op1 的数据类型一致
            assert res1.dtype == res2.dtype == op1.dtype
            # 断言提取的结果为 0，即 divmod 操作的结果应为 0
            assert extractor(res1) == 0
            assert extractor(res2) == 0

    # 使用 pytest 的标记来跳过在 WASM 环境下的测试，原因是浮点数错误在 WASM 中无法正常工作
    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    # 使用 sctypes['int'] 的参数化测试被除数和除数的数据类型
    @pytest.mark.parametrize("dividend_dtype", sctypes['int'])
    @pytest.mark.parametrize("divisor_dtype", sctypes['int'])
    # 使用一组操作函数进行参数化测试，包括 remainder, fmod, divmod, floor_divide,
    # operator.mod, operator.floordiv
    @pytest.mark.parametrize("operation",
            [np.remainder, np.fmod, np.divmod, np.floor_divide,
             operator.mod, operator.floordiv])
    # 设置 NumPy 的错误状态，当除法操作出现警告时发出警告
    @np.errstate(divide='warn', over='warn')
    # 定义一个测试函数，用于测试溢出情况
    def test_overflows(self, dividend_dtype, divisor_dtype, operation):
        # SIMD 尝试在尽可能多的元素上执行操作，
        # 这些元素的数量是寄存器大小的倍数。对于剩余的元素，我们使用默认的实现。
        # 我们在此尝试覆盖所有可能的路径。

        # 创建一系列数组，每个数组包含从最小值开始递增的整数，
        # 数据类型为 dividend_dtype，数组长度从 1 到 128
        arrays = [np.array([np.iinfo(dividend_dtype).min]*i,
                           dtype=dividend_dtype) for i in range(1, 129)]
        
        # 创建一个包含单个元素 -1 的数组，数据类型为 divisor_dtype
        divisor = np.array([-1], dtype=divisor_dtype)
        
        # 如果 dividend_dtype 的数据类型大小大于等于 divisor_dtype 的数据类型大小，
        # 且 operation 是 divmod、floor_divide 或者 operator.floordiv 中的一种，
        # 则执行以下代码块
        if np.dtype(dividend_dtype).itemsize >= np.dtype(
                divisor_dtype).itemsize and operation in (
                        np.divmod, np.floor_divide, operator.floordiv):
            
            # 使用 pytest.warns 检测 RuntimeWarning，确保输出中包含 "overflow encountered in"
            with pytest.warns(
                    RuntimeWarning,
                    match="overflow encountered in"):
                # 执行操作，传入最小值和 -1 作为参数
                result = operation(
                            dividend_dtype(np.iinfo(dividend_dtype).min),
                            divisor_dtype(-1)
                        )
                # 断言结果等于预期结果，使用 self.overflow_results 获取预期值
                assert result == self.overflow_results[operation].nocast(
                        dividend_dtype)

            # 遍历数组
            for a in arrays:
                # 对于 divmod 操作，需要先展开结果列，
                # 因为结果是商和余数的列向量，而预期结果是展开后的数组
                with pytest.warns(
                        RuntimeWarning,
                        match="overflow encountered in"):
                    # 执行操作，传入数组和除数，然后展开结果
                    result = np.array(operation(a, divisor)).flatten('f')
                    # 构建预期结果数组
                    expected_array = np.array(
                            [self.overflow_results[operation].nocast(
                                dividend_dtype)]*len(a)).flatten()
                    # 断言结果数组等于预期数组
                    assert_array_equal(result, expected_array)
        else:
            # 如果 dividend_dtype 的数据类型大小小于 divisor_dtype 的数据类型大小，
            # 或者 operation 不是 divmod、floor_divide 或 operator.floordiv 中的一种，
            # 则执行以下代码块
            
            # 执行操作，传入最小值和 -1 作为参数
            result = operation(
                        dividend_dtype(np.iinfo(dividend_dtype).min),
                        divisor_dtype(-1)
                    )
            # 断言结果等于预期结果，使用 self.overflow_results 获取预期值
            assert result == self.overflow_results[operation].casted(
                    dividend_dtype)

            # 遍历数组
            for a in arrays:
                # 同上，参考上面的展开结果的注释
                result = np.array(operation(a, divisor)).flatten('f')
                expected_array = np.array(
                        [self.overflow_results[operation].casted(
                            dividend_dtype)]*len(a)).flatten()
                assert_array_equal(result, expected_array)
class TestCbrt:
    # 测试立方根函数对标量值的计算
    def test_cbrt_scalar(self):
        # 断言近似相等：计算 -2.5 的立方根是否近似等于 -2.5
        assert_almost_equal((np.cbrt(np.float32(-2.5)**3)), -2.5)

    # 测试立方根函数对数组的计算
    def test_cbrt(self):
        # 创建包含不同类型数值的数组
        x = np.array([1., 2., -3., np.inf, -np.inf])
        # 断言近似相等：计算数组每个元素的立方根后应该近似等于原始数组
        assert_almost_equal(np.cbrt(x**3), x)

        # 断言 NaN 的立方根应该是 NaN
        assert_(np.isnan(np.cbrt(np.nan)))
        # 断言正无穷的立方根应该是正无穷
        assert_equal(np.cbrt(np.inf), np.inf)
        # 断言负无穷的立方根应该是负无穷
        assert_equal(np.cbrt(-np.inf), -np.inf)


class TestPower:
    # 测试幂函数对浮点数的计算
    def test_power_float(self):
        # 创建浮点数数组
        x = np.array([1., 2., 3.])
        # 断言：任何数的零次幂应该是 1
        assert_equal(x**0, [1., 1., 1.])
        # 断言：任何数的一次幂应该是它自己
        assert_equal(x**1, x)
        # 断言：计算平方
        assert_equal(x**2, [1., 4., 9.])
        # 创建副本并计算平方
        y = x.copy()
        y **= 2
        assert_equal(y, [1., 4., 9.])
        # 断言：计算倒数
        assert_almost_equal(x**(-1), [1., 0.5, 1./3])
        # 断言：计算平方根，使用 numpy 的数值计算库
        assert_almost_equal(x**(0.5), [1., ncu.sqrt(2), ncu.sqrt(3)])

        # 对于每组数据，生成对齐数据并进行测试
        for out, inp, msg in _gen_alignment_data(dtype=np.float32,
                                                 type='unary',
                                                 max_size=11):
            # 期望的结果是输入的每个元素的平方根
            exp = [ncu.sqrt(i) for i in inp]
            assert_almost_equal(inp**(0.5), exp, err_msg=msg)
            # 使用 numpy 计算平方根，将结果与期望进行比较
            np.sqrt(inp, out=out)
            assert_equal(out, exp, err_msg=msg)

        # 对于每组数据，生成对齐数据并进行测试
        for out, inp, msg in _gen_alignment_data(dtype=np.float64,
                                                 type='unary',
                                                 max_size=7):
            # 期望的结果是输入的每个元素的平方根
            exp = [ncu.sqrt(i) for i in inp]
            assert_almost_equal(inp**(0.5), exp, err_msg=msg)
            # 使用 numpy 计算平方根，将结果与期望进行比较
            np.sqrt(inp, out=out)
            assert_equal(out, exp, err_msg=msg)

    # 测试幂函数对复数的计算
    def test_power_complex(self):
        # 创建复数数组
        x = np.array([1+2j, 2+3j, 3+4j])
        # 断言：任何数的零次幂应该是 1
        assert_equal(x**0, [1., 1., 1.])
        # 断言：任何数的一次幂应该是它自己
        assert_equal(x**1, x)
        # 断言：计算平方
        assert_almost_equal(x**2, [-3+4j, -5+12j, -7+24j])
        # 断言：计算立方
        assert_almost_equal(x**3, [(1+2j)**3, (2+3j)**3, (3+4j)**3])
        # 断言：计算四次方
        assert_almost_equal(x**4, [(1+2j)**4, (2+3j)**4, (3+4j)**4])
        # 断言：计算倒数
        assert_almost_equal(x**(-1), [1/(1+2j), 1/(2+3j), 1/(3+4j)])
        # 断言：计算平方的倒数
        assert_almost_equal(x**(-2), [1/(1+2j)**2, 1/(2+3j)**2, 1/(3+4j)**2])
        # 断言：计算立方的倒数
        assert_almost_equal(x**(-3), [(-11+2j)/125, (-46-9j)/2197,
                                      (-117-44j)/15625])
        # 断言：计算平方根，使用 numpy 的数值计算库
        assert_almost_equal(x**(0.5), [ncu.sqrt(1+2j), ncu.sqrt(2+3j),
                                       ncu.sqrt(3+4j)])
        # 计算标准化因子，使 x 的 14 次幂第一个元素等于标准化因子
        norm = 1./((x**14)[0])
        assert_almost_equal(x**14 * norm,
                [i * norm for i in [-76443+16124j, 23161315+58317492j,
                                    5583548873 + 2465133864j]])

        # Ticket #836
        # 定义一个函数来断言复数相等
        def assert_complex_equal(x, y):
            assert_array_equal(x.real, y.real)
            assert_array_equal(x.imag, y.imag)

        # 对于每个复数 z 进行测试
        for z in [complex(0, np.inf), complex(1, np.inf)]:
            # 将 z 转换为复数数组，并忽略无效值错误
            z = np.array([z], dtype=np.complex128)
            with np.errstate(invalid="ignore"):
                # 断言 z 的一次幂应该等于 z
                assert_complex_equal(z**1, z)
                # 断言 z 的平方等于 z*z
                assert_complex_equal(z**2, z*z)
                # 断言 z 的立方等于 z*z*z
                assert_complex_equal(z**3, z*z*z)
    # 定义一个测试用例，验证零次幂的计算
    def test_power_zero(self):
        # ticket #1271
        # 创建一个复数数组，包含一个复数0j
        zero = np.array([0j])
        # 创建一个复数数组，包含一个实数1+0j
        one = np.array([1+0j])
        # 创建一个复数数组，包含一个复数NaN + NaNj
        cnan = np.array([complex(np.nan, np.nan)])
        # FIXME cinf not tested.
        # cinf = np.array([complex(np.inf, 0)])

        # 定义一个函数，用来断言两个复数数组相等
        def assert_complex_equal(x, y):
            x, y = np.asarray(x), np.asarray(y)
            # 断言两个复数数组的实部和虚部相等
            assert_array_equal(x.real, y.real)
            assert_array_equal(x.imag, y.imag)

        # 正幂的测试
        # 对于指数p列表中的每个值，验证零的p次幂等于零
        for p in [0.33, 0.5, 1, 1.5, 2, 3, 4, 5, 6.6]:
            assert_complex_equal(np.power(zero, p), zero)

        # 零次幂
        # 验证零的零次幂等于一
        assert_complex_equal(np.power(zero, 0), one)
        with np.errstate(invalid="ignore"):
            # 验证零的复数次幂生成NaN
            assert_complex_equal(np.power(zero, 0+1j), cnan)

            # 负幂的测试
            # 对于指数p列表中的每个值，验证零的-p次幂生成NaN
            for p in [0.33, 0.5, 1, 1.5, 2, 3, 4, 5, 6.6]:
                assert_complex_equal(np.power(zero, -p), cnan)
            # 验证零的负1+0.2j次幂生成NaN
            assert_complex_equal(np.power(zero, -1+0.2j), cnan)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    # 测试用例，验证零的非零次幂
    def test_zero_power_nonzero(self):
        # Testing 0^{Non-zero} issue 18378
        # 创建一个复数数组，包含一个实数0.0+0.0j
        zero = np.array([0.0+0.0j])
        # 创建一个复数数组，包含一个复数NaN + NaNj
        cnan = np.array([complex(np.nan, np.nan)])

        # 定义一个函数，用来断言两个复数数组相等
        def assert_complex_equal(x, y):
            # 断言两个复数数组的实部和虚部相等
            assert_array_equal(x.real, y.real)
            assert_array_equal(x.imag, y.imag)

        # 复数次幂，当实部为正时不会生成警告
        assert_complex_equal(np.power(zero, 1+4j), zero)
        assert_complex_equal(np.power(zero, 2-3j), zero)
        # 测试零值当实部大于零时
        assert_complex_equal(np.power(zero, 1+1j), zero)
        assert_complex_equal(np.power(zero, 1+0j), zero)
        assert_complex_equal(np.power(zero, 1-1j), zero)
        # 复数次幂，当实部为负或零（且虚部非零）时生成NaN，会触发RUNTIME警告
        with pytest.warns(expected_warning=RuntimeWarning) as r:
            assert_complex_equal(np.power(zero, -1+1j), cnan)
            assert_complex_equal(np.power(zero, -2-3j), cnan)
            assert_complex_equal(np.power(zero, -7+0j), cnan)
            assert_complex_equal(np.power(zero, 0+1j), cnan)
            assert_complex_equal(np.power(zero, 0-1j), cnan)
        # 断言收到5个警告
        assert len(r) == 5

    # 测试用例，验证快速幂运算
    def test_fast_power(self):
        # 创建一个整数数组
        x = np.array([1, 2, 3], np.int16)
        # 计算x的2.0次幂
        res = x**2.0
        # 断言结果的数据类型与res的数据类型相同
        assert_((x**2.00001).dtype is res.dtype)
        # 断言结果与期望结果相等
        assert_array_equal(res, [1, 4, 9])
        # 检查在类型转换后的复制上进行原位操作不会影响x
        assert_(not np.may_share_memory(res, x))
        assert_array_equal(x, [1, 2, 3])

        # 检查快速路径是否忽略1元素非0维数组
        res = x ** np.array([[[2]]])
        assert_equal(res.shape, (1, 1, 3))

    # 测试用例，验证整数幂运算
    def test_integer_power(self):
        # 创建一个长整型数组
        a = np.array([15, 15], 'i8')
        # 计算a的a次幂
        b = np.power(a, a)
        # 断言结果与期望结果相等
        assert_equal(b, [437893890380859375, 437893890380859375])
    # 定义测试函数，验证整数以0为指数的幂运算
    def test_integer_power_with_integer_zero_exponent(self):
        # 获取所有整数类型的数据类型码
        dtypes = np.typecodes['Integer']
        # 遍历每种数据类型
        for dt in dtypes:
            # 创建一个从-10到9的数组，使用当前数据类型
            arr = np.arange(-10, 10, dtype=dt)
            # 断言数组元素求0次幂后结果为全1数组
            assert_equal(np.power(arr, 0), np.ones_like(arr))

        # 获取所有无符号整数类型的数据类型码
        dtypes = np.typecodes['UnsignedInteger']
        # 遍历每种数据类型
        for dt in dtypes:
            # 创建一个从0到9的数组，使用当前数据类型
            arr = np.arange(10, dtype=dt)
            # 断言数组元素求0次幂后结果为全1数组
            assert_equal(np.power(arr, 0), np.ones_like(arr))

    # 定义测试函数，验证整数以1为底数的幂运算
    def test_integer_power_of_1(self):
        # 获取所有整数类型的数据类型码
        dtypes = np.typecodes['AllInteger']
        # 遍历每种数据类型
        for dt in dtypes:
            # 创建一个从0到9的数组，使用当前数据类型
            arr = np.arange(10, dtype=dt)
            # 断言1的任意整数次幂结果为全1数组
            assert_equal(np.power(1, arr), np.ones_like(arr))

    # 定义测试函数，验证整数以0为底数的幂运算
    def test_integer_power_of_zero(self):
        # 获取所有整数类型的数据类型码
        dtypes = np.typecodes['AllInteger']
        # 遍历每种数据类型
        for dt in dtypes:
            # 创建一个从1到9的数组，使用当前数据类型
            arr = np.arange(1, 10, dtype=dt)
            # 断言0的任意正整数次幂结果为全0数组
            assert_equal(np.power(0, arr), np.zeros_like(arr))

    # 定义测试函数，验证整数以负整数为指数的幂运算
    def test_integer_to_negative_power(self):
        # 获取所有整数类型的数据类型码
        dtypes = np.typecodes['Integer']
        # 遍历每种数据类型
        for dt in dtypes:
            # 创建一个包含0到3的整数数组，使用当前数据类型
            a = np.array([0, 1, 2, 3], dtype=dt)
            # 创建一个包含0、1、2和-3的整数数组，使用当前数据类型
            b = np.array([0, 1, 2, -3], dtype=dt)
            # 创建一个整数1，使用当前数据类型
            one = np.array(1, dtype=dt)
            # 创建一个整数-1，使用当前数据类型
            minusone = np.array(-1, dtype=dt)
            # 断言对于负指数或1为底数，幂运算会引发值错误
            assert_raises(ValueError, np.power, a, b)
            assert_raises(ValueError, np.power, a, minusone)
            assert_raises(ValueError, np.power, one, b)
            assert_raises(ValueError, np.power, one, minusone)

    # 定义测试函数，验证浮点数到无穷大为指数的幂运算
    def test_float_to_inf_power(self):
        # 遍历每种浮点数数据类型
        for dt in [np.float32, np.float64]:
            # 创建一个包含1、1、2、2、-2、-2、正无穷和负无穷的浮点数组，使用当前数据类型
            a = np.array([1, 1, 2, 2, -2, -2, np.inf, -np.inf], dt)
            # 创建一个包含正无穷、负无穷、正无穷、负无穷、正无穷、负无穷、正无穷、负无穷的浮点数组，使用当前数据类型
            b = np.array([np.inf, -np.inf, np.inf, -np.inf,
                                np.inf, -np.inf, np.inf, -np.inf], dt)
            # 创建一个包含1、1、正无穷、0、正无穷、0、正无穷、0的浮点数组，使用当前数据类型
            r = np.array([1, 1, np.inf, 0, np.inf, 0, np.inf, 0], dt)
            # 断言浮点数到无穷大为指数的幂运算结果为预期的结果数组
            assert_equal(np.power(a, b), r)

    # 定义测试函数，验证快速路径的幂运算
    def test_power_fast_paths(self):
        # gh-26055
        # 遍历每种浮点数数据类型
        for dt in [np.float32, np.float64]:
            # 创建一个包含0、1.1、2、12e12、-10.、正无穷、负无穷的浮点数组，使用当前数据类型
            a = np.array([0, 1.1, 2, 12e12, -10., np.inf, -np.inf], dt)
            # 创建一个包含0.0、1.21、4.、1.44e+26、100、正无穷、正无穷的浮点数组，使用当前数据类型
            expected = np.array([0.0, 1.21, 4., 1.44e+26, 100, np.inf, np.inf])
            # 对数组进行平方运算，使用当前数据类型
            result = np.power(a, 2.)
            # 断言数组按最大ULP的数组与预期类型的最大ULP为1
            assert_array_max_ulp(result, expected.astype(dt), maxulp=1)

            # 创建一个包含0、1.1、2、12e12的浮点数组，使用当前数据类型
            a = np.array([0, 1.1, 2, 12e12], dt)
            # 创建一个对a数组进行平方根运算并转换成当前数据类型的数组
            expected = np.sqrt(a).astype(dt)
            # 对数组进行0.5次方运算，使用当前数据类型
            result = np.power(a, 0.5)
            # 断言数组按最大ULP的数组与预期数组进行最大ULP为1的检验
            assert_array_max_ulp(result, expected, maxulp=1)
class TestFloat_power:
    # 测试类，用于测试 np.float_power 函数的类型转换功能
    def test_type_conversion(self):
        # 输入参数类型和输出结果类型的字符串表示
        arg_type = '?bhilBHILefdgFDG'
        res_type = 'ddddddddddddgDDG'
        # 遍历输入参数类型和输出结果类型，逐一测试
        for dtin, dtout in zip(arg_type, res_type):
            # 构建测试消息
            msg = "dtin: %s, dtout: %s" % (dtin, dtout)
            # 创建具有给定数据类型的包含一个元素的数组
            arg = np.ones(1, dtype=dtin)
            # 计算 np.float_power 的结果
            res = np.float_power(arg, arg)
            # 断言结果的数据类型与期望的输出数据类型相符
            assert_(res.dtype.name == np.dtype(dtout).name, msg)


class TestLog2:
    # 测试类，用于测试 np.log2 函数的各种情况
    @pytest.mark.parametrize('dt', ['f', 'd', 'g'])
    def test_log2_values(self, dt):
        # 输入数组和预期输出结果
        x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # 将输入数组转换为指定数据类型的 numpy 数组
        xf = np.array(x, dtype=dt)
        yf = np.array(y, dtype=dt)
        # 断言 np.log2 的计算结果与预期输出结果相近
        assert_almost_equal(np.log2(xf), yf)

    @pytest.mark.parametrize("i", range(1, 65))
    def test_log2_ints(self, i):
        # 测试对整数进行 np.log2 计算的正确性
        v = np.log2(2.**i)
        assert_equal(v, float(i), err_msg='at exponent %d' % i)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_log2_special(self):
        # 测试 np.log2 函数在特殊情况下的行为
        assert_equal(np.log2(1.), 0.)
        assert_equal(np.log2(np.inf), np.inf)
        assert_(np.isnan(np.log2(np.nan)))

        # 测试在警告情况下的 np.log2 的行为
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            assert_(np.isnan(np.log2(-1.)))
            assert_(np.isnan(np.log2(-np.inf)))
            assert_equal(np.log2(0.), -np.inf)
            # 检查警告类型
            assert_(w[0].category is RuntimeWarning)
            assert_(w[1].category is RuntimeWarning)
            assert_(w[2].category is RuntimeWarning)


class TestExp2:
    # 测试类，用于测试 np.exp2 函数的功能
    def test_exp2_values(self):
        # 输入数组和预期输出结果
        x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # 遍历数据类型，逐一测试
        for dt in ['f', 'd', 'g']:
            # 将输入数组转换为指定数据类型的 numpy 数组
            xf = np.array(x, dtype=dt)
            yf = np.array(y, dtype=dt)
            # 断言 np.exp2 的计算结果与预期输出结果相近
            assert_almost_equal(np.exp2(yf), xf)


class TestLogAddExp2(_FilterInvalids):
    # 测试类，继承自 _FilterInvalids，用于测试 np.logaddexp2 函数的功能
    # 需要测试中间精度的情况
    def test_logaddexp2_values(self):
        # 输入数组和预期输出结果
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        z = [6, 6, 6, 6, 6]
        # 遍历数据类型和精度值，逐一测试
        for dt, dec_ in zip(['f', 'd', 'g'], [6, 15, 15]):
            # 计算 np.log2 的结果
            xf = np.log2(np.array(x, dtype=dt))
            yf = np.log2(np.array(y, dtype=dt))
            zf = np.log2(np.array(z, dtype=dt))
            # 断言 np.logaddexp2 的计算结果与预期输出结果相近
            assert_almost_equal(np.logaddexp2(xf, yf), zf, decimal=dec_)

    def test_logaddexp2_range(self):
        # 输入数组和预期输出结果
        x = [1000000, -1000000, 1000200, -1000200]
        y = [1000200, -1000200, 1000000, -1000000]
        z = [1000200, -1000000, 1000200, -1000000]
        # 遍历数据类型，逐一测试
        for dt in ['f', 'd', 'g']:
            # 计算 np.log2 的结果
            logxf = np.array(x, dtype=dt)
            logyf = np.array(y, dtype=dt)
            logzf = np.array(z, dtype=dt)
            # 断言 np.logaddexp2 的计算结果与预期输出结果相近
            assert_almost_equal(np.logaddexp2(logxf, logyf), logzf)
    # 定义一个测试函数，用于测试 np.logaddexp2 函数在不同情况下的行为
    def test_inf(self):
        # 定义无穷大常量
        inf = np.inf
        # 创建包含各种无穷大、有限值的列表 x
        x = [inf, -inf,  inf, -inf, inf, 1,  -inf,  1]
        # 创建包含各种无穷大、有限值的列表 y
        y = [inf,  inf, -inf, -inf, 1,   inf, 1,   -inf]
        # 创建包含各种无穷大、有限值的列表 z
        z = [inf,  inf,  inf, -inf, inf, inf, 1,    1]
        
        # 在处理中遇到无效操作时抛出异常
        with np.errstate(invalid='raise'):
            # 遍历数据类型列表 ['f', 'd', 'g']
            for dt in ['f', 'd', 'g']:
                # 使用指定数据类型创建数组 logxf，并赋予 x 的值
                logxf = np.array(x, dtype=dt)
                # 使用指定数据类型创建数组 logyf，并赋予 y 的值
                logyf = np.array(y, dtype=dt)
                # 使用指定数据类型创建数组 logzf，并赋予 z 的值
                logzf = np.array(z, dtype=dt)
                # 断言 np.logaddexp2(logxf, logyf) 等于 logzf
                assert_equal(np.logaddexp2(logxf, logyf), logzf)

    # 定义测试函数，用于测试 np.logaddexp2 函数处理 NaN 值的行为
    def test_nan(self):
        # 断言 np.logaddexp2(np.nan, np.inf) 的结果是 NaN
        assert_(np.isnan(np.logaddexp2(np.nan, np.inf)))
        # 断言 np.logaddexp2(np.inf, np.nan) 的结果是 NaN
        assert_(np.isnan(np.logaddexp2(np.inf, np.nan)))
        # 断言 np.logaddexp2(np.nan, 0) 的结果是 NaN
        assert_(np.isnan(np.logaddexp2(np.nan, 0)))
        # 断言 np.logaddexp2(0, np.nan) 的结果是 NaN
        assert_(np.isnan(np.logaddexp2(0, np.nan)))
        # 断言 np.logaddexp2(np.nan, np.nan) 的结果是 NaN
        assert_(np.isnan(np.logaddexp2(np.nan, np.nan)))

    # 定义测试函数，用于测试 np.logaddexp2 函数的 reduce 方法
    def test_reduce(self):
        # 断言 np.logaddexp2.identity 等于负无穷大
        assert_equal(np.logaddexp2.identity, -np.inf)
        # 断言对空列表使用 np.logaddexp2.reduce 的结果是负无穷大
        assert_equal(np.logaddexp2.reduce([]), -np.inf)
        # 断言对包含单个 -np.inf 的列表使用 np.logaddexp2.reduce 的结果是负无穷大
        assert_equal(np.logaddexp2.reduce([-np.inf]), -np.inf)
        # 断言对包含 -np.inf 和 0 的列表使用 np.logaddexp2.reduce 的结果是 0
        assert_equal(np.logaddexp2.reduce([-np.inf, 0]), 0)
# 定义一个测试日志类 TestLog
class TestLog:

    # 测试对数函数的值
    def test_log_values(self):
        # 定义 x 和 y 数组
        x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # 遍历不同的数据类型
        for dt in ['f', 'd', 'g']:
            # 定义常数 log2_
            log2_ = 0.69314718055994530943
            # 将 x 转换为指定数据类型的数组 xf
            xf = np.array(x, dtype=dt)
            # 计算 y 数组，每个元素乘以 log2_
            yf = np.array(y, dtype=dt) * log2_
            # 断言计算的自然对数与预期的 yf 数组接近
            assert_almost_equal(np.log(xf), yf)

        # 测试别名问题（问题编号 #17761）
        x = np.array([2, 0.937500, 3, 0.947500, 1.054697])
        # 计算 xf 数组的自然对数
        xf = np.log(x)
        # 将计算的结果放入现有数组 x，然后断言结果与 xf 数组接近
        assert_almost_equal(np.log(x, out=x), xf)

    # 测试对数函数在各数据类型的最大值上的行为
    def test_log_values_maxofdtype(self):
        # 定义数据类型列表
        dtypes = [np.float32, np.float64]
        # 在非 x86-64 平台上，对 longdouble 的检查并不太有用
        if platform.machine() == 'x86_64':
            dtypes += [np.longdouble]

        # 遍历各种数据类型
        for dt in dtypes:
            # 恢复所有错误状态为默认行为，并在运算过程中引发错误
            with np.errstate(all='raise'):
                # 获取当前数据类型的最大值 x
                x = np.finfo(dt).max
                # 计算 x 的自然对数
                np.log(x)

    # 测试对数函数在不同步长和数据大小下的行为
    def test_log_strides(self):
        np.random.seed(42)
        # 定义步长数组和尺寸数组
        strides = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
        sizes = np.arange(2, 100)
        
        # 遍历尺寸数组
        for ii in sizes:
            # 生成指定范围内均匀分布的随机数，以 float64 类型存储在 x_f64
            x_f64 = np.float64(np.random.uniform(low=0.01, high=100.0, size=ii))
            # 创建 x_special，复制自 x_f64
            x_special = x_f64.copy()
            # 将 x_special 中指定范围的元素设置为 1.0
            x_special[3:-1:4] = 1.0
            # 计算 x_f64 和 x_special 的自然对数
            y_true = np.log(x_f64)
            y_special = np.log(x_special)
            
            # 遍历步长数组
            for jj in strides:
                # 断言对 np.log(x_f64[::jj]) 和 y_true[::jj] 进行近似相等性检查
                assert_array_almost_equal_nulp(np.log(x_f64[::jj]), y_true[::jj], nulp=2)
                # 断言对 np.log(x_special[::jj]) 和 y_special[::jj] 进行近似相等性检查
                assert_array_almost_equal_nulp(np.log(x_special[::jj]), y_special[::jj], nulp=2)

    # 使用 mpmath 计算的参考值进行测试精度（数据类型为 float64）
    @pytest.mark.parametrize(
        'z, wref',
        [(1 + 1e-12j, 5e-25 + 1e-12j),
         (1.000000000000001 + 3e-08j,
          1.5602230246251546e-15 + 2.999999999999996e-08j),
         (0.9999995000000417 + 0.0009999998333333417j,
          7.831475869017683e-18 + 0.001j),
         (0.9999999999999996 + 2.999999999999999e-08j,
          5.9107901499372034e-18 + 3e-08j),
         (0.99995000042 - 0.009999833j,
          -7.015159763822903e-15 - 0.009999999665816696j)],
    )
    def test_log_precision_float64(self, z, wref):
        # 计算 z 的自然对数 w
        w = np.log(z)
        # 断言计算的结果 w 与参考值 wref 的接近程度
        assert_allclose(w, wref, rtol=1e-15)

    # 使用 mpmath 计算的参考值进行测试精度（数据类型为 float32）
    @pytest.mark.parametrize(
        'z, wref',
        [(np.complex64(1.0 + 3e-6j), np.complex64(4.5e-12+3e-06j)),
         (np.complex64(1.0 - 2e-5j), np.complex64(1.9999999e-10 - 2e-5j)),
         (np.complex64(0.9999999 + 1e-06j),
          np.complex64(-1.192088e-07+1.0000001e-06j))],
    )
    def test_log_precision_float32(self, z, wref):
        # 计算 z 的自然对数 w
        w = np.log(z)
        # 断言计算的结果 w 与参考值 wref 的接近程度
        assert_allclose(w, wref, rtol=1e-6)
    # 定义一个测试函数，用于验证对指定数据类型的指数函数计算是否正确
    def test_exp_values(self):
        # 定义输入的 x 和对应的指数值 y
        x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # 对于每种数据类型 ['f', 'd', 'g'] 进行测试
        for dt in ['f', 'd', 'g']:
            # 定义常数 log2_
            log2_ = 0.69314718055994530943
            # 使用 numpy 创建包含 x 数组的指定数据类型的数组 xf
            xf = np.array(x, dtype=dt)
            # 根据对应的数据类型和 log2_ 常数创建 yf 数组
            yf = np.array(y, dtype=dt) * log2_
            # 断言 numpy 的 exp 函数应用于 yf 后结果与 xf 相等
            assert_almost_equal(np.exp(yf), xf)

    # 定义一个测试函数，用于验证对指定步长下的指数函数计算是否正确
    def test_exp_strides(self):
        # 设定随机数种子，确保每次运行结果一致
        np.random.seed(42)
        # 定义步长数组 strides
        strides = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
        # 创建从 2 到 99 的数组 sizes
        sizes = np.arange(2, 100)
        # 对于 sizes 中的每个元素 ii 进行测试
        for ii in sizes:
            # 生成指定范围内的随机浮点数数组 x_f64
            x_f64 = np.float64(np.random.uniform(low=0.01, high=709.1, size=ii))
            # 计算 x_f64 的指数函数结果作为参考结果 y_true
            y_true = np.exp(x_f64)
            # 对于 strides 中的每个元素 jj 进行测试
            for jj in strides:
                # 断言使用指定步长 jj 下，对 x_f64 的指数函数结果应与 y_true[::jj] 相近
                assert_array_almost_equal_nulp(np.exp(x_f64[::jj]), y_true[::jj], nulp=2)
# 定义一个测试类 TestSpecialFloats，用于测试处理特殊浮点数的函数
class TestSpecialFloats:
    
    # 定义测试函数 test_exp_values，测试指数函数 np.exp 对特殊浮点数的处理
    def test_exp_values(self):
        # 设置 numpy 错误状态，当发生特定错误时抛出异常
        with np.errstate(under='raise', over='raise'):
            # 定义输入数组 x 和 y，包含 NaN、正负无穷和零
            x = [np.nan,  np.nan, np.inf, 0.]
            y = [np.nan, -np.nan, np.inf, -np.inf]
            # 遍历数据类型 ['e', 'f', 'd', 'g']
            for dt in ['e', 'f', 'd', 'g']:
                # 将 x 和 y 转换为指定数据类型的 numpy 数组
                xf = np.array(x, dtype=dt)
                yf = np.array(y, dtype=dt)
                # 断言 np.exp(yf) 的结果与 xf 相等
                assert_equal(np.exp(yf), xf)

    # 标记为预期失败的测试函数，参考 https://github.com/numpy/numpy/issues/19192
    @pytest.mark.xfail(
        _glibc_older_than("2.17"),
        reason="Older glibc versions may not raise appropriate FP exceptions"
    )
    # 定义测试函数 test_exp_exceptions，测试 np.exp 在特定情况下是否会引发浮点数异常
    def test_exp_exceptions(self):
        # 设置 numpy 错误状态，当发生上溢出错误时抛出异常
        with np.errstate(over='raise'):
            # 断言 np.exp 对各种浮点数类型的输入会引发 FloatingPointError 异常
            assert_raises(FloatingPointError, np.exp, np.float16(11.0899))
            assert_raises(FloatingPointError, np.exp, np.float32(100.))
            assert_raises(FloatingPointError, np.exp, np.float32(1E19))
            assert_raises(FloatingPointError, np.exp, np.float64(800.))
            assert_raises(FloatingPointError, np.exp, np.float64(1E19))

        # 设置 numpy 错误状态，当发生下溢出错误时抛出异常
        with np.errstate(under='raise'):
            # 断言 np.exp 对各种浮点数类型的输入会引发 FloatingPointError 异常
            assert_raises(FloatingPointError, np.exp, np.float16(-17.5))
            assert_raises(FloatingPointError, np.exp, np.float32(-1000.))
            assert_raises(FloatingPointError, np.exp, np.float32(-1E19))
            assert_raises(FloatingPointError, np.exp, np.float64(-1000.))
            assert_raises(FloatingPointError, np.exp, np.float64(-1E19))

    # 标记测试函数为在 WASM 环境下跳过执行的条件
    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    # 定义一个测试方法，用于测试对数函数在不同数据类型和异常条件下的行为
    def test_log_values(self):
        # 忽略所有的数值错误
        with np.errstate(all='ignore'):
            # 定义不同的测试数据集合
            x = [np.nan, np.nan, np.inf, np.nan, -np.inf, np.nan]
            y = [np.nan, -np.nan, np.inf, -np.inf, 0.0, -1.0]
            y1p = [np.nan, -np.nan, np.inf, -np.inf, -1.0, -2.0]
            # 遍历不同的数据类型
            for dt in ['e', 'f', 'd', 'g']:
                # 将数据转换为指定数据类型的数组
                xf = np.array(x, dtype=dt)
                yf = np.array(y, dtype=dt)
                yf1p = np.array(y1p, dtype=dt)
                # 断言对数函数的计算结果与预期相等
                assert_equal(np.log(yf), xf)
                assert_equal(np.log2(yf), xf)
                assert_equal(np.log10(yf), xf)
                assert_equal(np.log1p(yf1p), xf)

        # 抛出除法相关的浮点数错误
        with np.errstate(divide='raise'):
            # 再次遍历不同的数据类型
            for dt in ['e', 'f', 'd']:
                # 断言对数函数对零值进行计算时会抛出浮点数错误
                assert_raises(FloatingPointError, np.log,
                              np.array(0.0, dtype=dt))
                assert_raises(FloatingPointError, np.log2,
                              np.array(0.0, dtype=dt))
                assert_raises(FloatingPointError, np.log10,
                              np.array(0.0, dtype=dt))
                assert_raises(FloatingPointError, np.log1p,
                              np.array(-1.0, dtype=dt))

        # 抛出无效操作相关的浮点数错误
        with np.errstate(invalid='raise'):
            # 再次遍历不同的数据类型
            for dt in ['e', 'f', 'd']:
                # 断言对数函数对负无穷大和负数进行计算时会抛出浮点数错误
                assert_raises(FloatingPointError, np.log,
                              np.array(-np.inf, dtype=dt))
                assert_raises(FloatingPointError, np.log,
                              np.array(-1.0, dtype=dt))
                assert_raises(FloatingPointError, np.log2,
                              np.array(-np.inf, dtype=dt))
                assert_raises(FloatingPointError, np.log2,
                              np.array(-1.0, dtype=dt))
                assert_raises(FloatingPointError, np.log10,
                              np.array(-np.inf, dtype=dt))
                assert_raises(FloatingPointError, np.log10,
                              np.array(-1.0, dtype=dt))
                assert_raises(FloatingPointError, np.log1p,
                              np.array(-np.inf, dtype=dt))
                assert_raises(FloatingPointError, np.log1p,
                              np.array(-2.0, dtype=dt))

        # 参考：https://github.com/numpy/numpy/issues/18005
        # 确保在没有警告的情况下运行以下代码块
        with assert_no_warnings():
            # 创建一个 float32 类型的数组 a，其值为 1e9
            a = np.array(1e9, dtype='float32')
            # 对数组 a 进行对数运算
            np.log(a)

    # 使用 pytest 的标记功能，条件为 IS_WASM 为真时跳过此测试方法
    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    # 参数化测试方法，参数为不同的数据类型 ['e', 'f', 'd', 'g']
    def test_sincos_values(self, dtype):
        # 忽略所有的数值错误
        with np.errstate(all='ignore'):
            # 定义测试数据集合
            x = [np.nan, np.nan, np.nan, np.nan]
            y = [np.nan, -np.nan, np.inf, -np.inf]
            # 将测试数据转换为指定数据类型的数组
            xf = np.array(x, dtype=dtype)
            yf = np.array(y, dtype=dtype)
            # 断言正弦和余弦函数的计算结果与预期相等
            assert_equal(np.sin(yf), xf)
            assert_equal(np.cos(yf), xf)

    # 使用 pytest 的标记功能，条件为 IS_WASM 为真时跳过此测试方法
    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    # 使用 pytest 的 xfail 标记，当运行在 macOS 上时标记为预期失败，原因是触发了标量 'sin' 的下溢
    @pytest.mark.xfail(
        sys.platform.startswith("darwin"),
        reason="underflow is triggered for scalar 'sin'"
    )
    # 测试处理 'sin' 和 'cos' 函数下溢的情况
    def test_sincos_underflow(self):
        # 设置浮点错误状态为下溢时抛出异常
        with np.errstate(under='raise'):
            # 创建一个浮点数数组，使用十六进制表示的小数值，作为下溢触发器
            underflow_trigger = np.array(
                float.fromhex("0x1.f37f47a03f82ap-511"),
                dtype=np.float64
            )
            # 调用 sin 函数
            np.sin(underflow_trigger)
            # 调用 cos 函数
            np.cos(underflow_trigger)

    # 使用 pytest 的 skipif 标记，在运行在 wasm 环境下跳过测试，原因是浮点错误在 wasm 中无效
    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    # 测试处理 'sin' 和 'cos' 函数的浮点错误情况
    @pytest.mark.parametrize('callable', [np.sin, np.cos])
    @pytest.mark.parametrize('dtype', ['e', 'f', 'd'])
    @pytest.mark.parametrize('value', [np.inf, -np.inf])
    def test_sincos_errors(self, callable, dtype, value):
        # 设置浮点错误状态为无效时抛出异常
        with np.errstate(invalid='raise'):
            # 断言调用 callable 函数时，传入包含特定值的数组会引发浮点错误异常
            assert_raises(FloatingPointError, callable,
                np.array([value], dtype=dtype))

    # 测试 'sin' 和 'cos' 函数处理重叠数据的情况
    @pytest.mark.parametrize('callable', [np.sin, np.cos])
    @pytest.mark.parametrize('dtype', ['f', 'd'])
    @pytest.mark.parametrize('stride', [-1, 1, 2, 4, 5])
    def test_sincos_overlaps(self, callable, dtype, stride):
        # 定义数组大小为 N
        N = 100
        # 根据步幅计算 M
        M = N // abs(stride)
        # 使用随机数生成器创建随机正态分布的数组 x
        rng = np.random.default_rng(42)
        x = rng.standard_normal(N, dtype)
        # 调用 callable 函数处理数组 x 的子集，并将结果存储到 y 中
        y = callable(x[::stride])
        # 再次调用 callable 函数，使用相同的子集作为输入，并将结果存储回 x 的前 M 个元素
        callable(x[::stride], out=x[:M])
        # 断言 x 的前 M 个元素等于 y
        assert_equal(x[:M], y)

    # 测试处理 'sqrt' 函数的异常值情况
    @pytest.mark.parametrize('dt', ['e', 'f', 'd', 'g'])
    def test_sqrt_values(self, dt):
        # 忽略所有浮点错误状态
        with np.errstate(all='ignore'):
            # 创建包含 NaN, Inf 和 0 的数组 x 和 y
            x = [np.nan, np.nan, np.inf, np.nan, 0.]
            y = [np.nan, -np.nan, np.inf, -np.inf, 0.]
            # 创建对应数据类型的数组 xf 和 yf
            xf = np.array(x, dtype=dt)
            yf = np.array(y, dtype=dt)
            # 断言对 yf 使用 sqrt 函数的结果等于 xf
            assert_equal(np.sqrt(yf), xf)

        # 以下代码段被注释掉，原本是测试传入负数时 sqrt 函数抛出浮点错误异常的情况

    # 测试处理 'abs' 函数的异常值情况
    def test_abs_values(self):
        # 创建包含 NaN, Inf 和 0 的数组 x 和 y
        x = [np.nan,  np.nan, np.inf, np.inf, 0., 0., 1.0, 1.0]
        y = [np.nan, -np.nan, np.inf, -np.inf, 0., -0., -1.0, 1.0]
        # 遍历不同的数据类型
        for dt in ['e', 'f', 'd', 'g']:
            # 创建对应数据类型的数组 xf 和 yf
            xf = np.array(x, dtype=dt)
            yf = np.array(y, dtype=dt)
            # 断言对 yf 使用 abs 函数的结果等于 xf

    # 使用 pytest 的 skipif 标记，在运行在 wasm 环境下跳过测试，原因是浮点错误在 wasm 中无效
    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    # 测试处理 'square' 函数的异常值情况
    def test_square_values(self):
        # 创建包含 NaN, Inf 的数组 x 和 y
        x = [np.nan,  np.nan, np.inf, np.inf]
        y = [np.nan, -np.nan, np.inf, -np.inf]
        # 忽略所有浮点错误状态
        with np.errstate(all='ignore'):
            # 遍历不同的数据类型
            for dt in ['e', 'f', 'd', 'g']:
                # 创建对应数据类型的数组 xf 和 yf
                xf = np.array(x, dtype=dt)
                yf = np.array(y, dtype=dt)
                # 断言对 yf 使用 square 函数的结果等于 xf

        # 设置浮点错误状态为溢出时抛出异常
        with np.errstate(over='raise'):
            # 断言对传入特定值时调用 square 函数会引发浮点错误异常
            assert_raises(FloatingPointError, np.square,
                          np.array(1E3, dtype='e'))
            assert_raises(FloatingPointError, np.square,
                          np.array(1E32, dtype='f'))
            assert_raises(FloatingPointError, np.square,
                          np.array(1E200, dtype='d'))
    # 跳过在 WebAssembly 环境下执行，因为浮点数错误不起作用
    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    # 测试 np.reciprocal 函数的行为
    def test_reciprocal_values(self):
        # 忽略所有错误状态
        with np.errstate(all='ignore'):
            # 定义输入和期望输出数组
            x = [np.nan,  np.nan, 0.0, -0.0, np.inf, -np.inf]
            y = [np.nan, -np.nan, np.inf, -np.inf, 0., -0.]
            # 对于每种数据类型进行测试
            for dt in ['e', 'f', 'd', 'g']:
                # 将输入转换为指定数据类型的数组
                xf = np.array(x, dtype=dt)
                yf = np.array(y, dtype=dt)
                # 断言 np.reciprocal(yf) 的结果等于 xf
                assert_equal(np.reciprocal(yf), xf)

        # 设置除法错误状态为抛出异常
        with np.errstate(divide='raise'):
            # 对于每种数据类型进行测试
            for dt in ['e', 'f', 'd', 'g']:
                # 断言对于 np.array(-0.0, dtype=dt)，调用 np.reciprocal 会抛出 FloatingPointError 异常
                assert_raises(FloatingPointError, np.reciprocal,
                              np.array(-0.0, dtype=dt))

    # 跳过在 WebAssembly 环境下执行，因为浮点数错误不起作用
    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    # 测试 np.tan 函数的行为
    def test_tan(self):
        # 忽略所有错误状态
        with np.errstate(all='ignore'):
            # 定义输入和期望输出数组
            in_ = [np.nan, -np.nan, 0.0, -0.0, np.inf, -np.inf]
            out = [np.nan, np.nan, 0.0, -0.0, np.nan, np.nan]
            # 对于每种数据类型进行测试
            for dt in ['e', 'f', 'd']:
                # 将输入转换为指定数据类型的数组
                in_arr = np.array(in_, dtype=dt)
                out_arr = np.array(out, dtype=dt)
                # 断言 np.tan(in_arr) 的结果等于 out_arr
                assert_equal(np.tan(in_arr), out_arr)

        # 设置无效输入状态为抛出异常
        with np.errstate(invalid='raise'):
            # 对于每种数据类型进行测试
            for dt in ['e', 'f', 'd']:
                # 断言对于 np.array(np.inf, dtype=dt)，调用 np.tan 会抛出 FloatingPointError 异常
                assert_raises(FloatingPointError, np.tan,
                              np.array(np.inf, dtype=dt))
                # 断言对于 np.array(-np.inf, dtype=dt)，调用 np.tan 会抛出 FloatingPointError 异常
                assert_raises(FloatingPointError, np.tan,
                              np.array(-np.inf, dtype=dt))

    # 跳过在 WebAssembly 环境下执行，因为浮点数错误不起作用
    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    # 测试 np.arcsin 和 np.arccos 函数的行为
    def test_arcsincos(self):
        # 忽略所有错误状态
        with np.errstate(all='ignore'):
            # 定义输入和期望输出数组
            in_ = [np.nan, -np.nan, np.inf, -np.inf]
            out = [np.nan, np.nan, np.nan, np.nan]
            # 对于每种数据类型进行测试
            for dt in ['e', 'f', 'd']:
                # 将输入转换为指定数据类型的数组
                in_arr = np.array(in_, dtype=dt)
                out_arr = np.array(out, dtype=dt)
                # 断言 np.arcsin(in_arr) 的结果等于 out_arr
                assert_equal(np.arcsin(in_arr), out_arr)
                # 断言 np.arccos(in_arr) 的结果等于 out_arr
                assert_equal(np.arccos(in_arr), out_arr)

        # 对于 np.arcsin 和 np.arccos 函数，使用不合法的值进行测试
        for callable in [np.arcsin, np.arccos]:
            for value in [np.inf, -np.inf, 2.0, -2.0]:
                # 对于每种数据类型进行测试
                for dt in ['e', 'f', 'd']:
                    # 设置无效输入状态为抛出异常
                    with np.errstate(invalid='raise'):
                        # 断言对于 np.array(value, dtype=dt)，调用 callable 会抛出 FloatingPointError 异常
                        assert_raises(FloatingPointError, callable,
                                      np.array(value, dtype=dt))

    # 测试 np.arctan 函数的行为
    def test_arctan(self):
        # 忽略所有错误状态
        with np.errstate(all='ignore'):
            # 定义输入和期望输出数组
            in_ = [np.nan, -np.nan]
            out = [np.nan, np.nan]
            # 对于每种数据类型进行测试
            for dt in ['e', 'f', 'd']:
                # 将输入转换为指定数据类型的数组
                in_arr = np.array(in_, dtype=dt)
                out_arr = np.array(out, dtype=dt)
                # 断言 np.arctan(in_arr) 的结果等于 out_arr
                assert_equal(np.arctan(in_arr), out_arr)
    def test_sinh(self):
        # 输入数组，包含 NaN 和无穷大等特殊值
        in_ = [np.nan, -np.nan, np.inf, -np.inf]
        # 预期输出数组，与输入相同的特殊值
        out = [np.nan, np.nan, np.inf, -np.inf]
        # 对于每种数据类型 'e', 'f', 'd'，执行以下测试
        for dt in ['e', 'f', 'd']:
            # 创建指定数据类型的输入数组
            in_arr = np.array(in_, dtype=dt)
            # 创建预期输出数组
            out_arr = np.array(out, dtype=dt)
            # 断言 np.sinh(in_arr) 等于 out_arr
            assert_equal(np.sinh(in_arr), out_arr)

        # 在上下文中处理浮点数异常，设置 'over' 为 'raise' 模式
        with np.errstate(over='raise'):
            # 断言对于浮点数 12.0，使用 'e' 数据类型，调用 np.sinh 会引发 FloatingPointError
            assert_raises(FloatingPointError, np.sinh,
                          np.array(12.0, dtype='e'))
            # 断言对于浮点数 120.0，使用 'f' 数据类型，调用 np.sinh 会引发 FloatingPointError
            assert_raises(FloatingPointError, np.sinh,
                          np.array(120.0, dtype='f'))
            # 断言对于浮点数 1200.0，使用 'd' 数据类型，调用 np.sinh 会引发 FloatingPointError
            assert_raises(FloatingPointError, np.sinh,
                          np.array(1200.0, dtype='d'))

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.skipif('bsd' in sys.platform,
            reason="fallback implementation may not raise, see gh-2487")
    def test_cosh(self):
        # 输入数组，包含 NaN 和无穷大等特殊值
        in_ = [np.nan, -np.nan, np.inf, -np.inf]
        # 预期输出数组，其中无穷大的输出值不同
        out = [np.nan, np.nan, np.inf, np.inf]
        # 对于每种数据类型 'e', 'f', 'd'，执行以下测试
        for dt in ['e', 'f', 'd']:
            # 创建指定数据类型的输入数组
            in_arr = np.array(in_, dtype=dt)
            # 创建预期输出数组
            out_arr = np.array(out, dtype=dt)
            # 断言 np.cosh(in_arr) 等于 out_arr
            assert_equal(np.cosh(in_arr), out_arr)

        # 在上下文中处理浮点数异常，设置 'over' 为 'raise' 模式
        with np.errstate(over='raise'):
            # 断言对于浮点数 12.0，使用 'e' 数据类型，调用 np.cosh 会引发 FloatingPointError
            assert_raises(FloatingPointError, np.cosh,
                          np.array(12.0, dtype='e'))
            # 断言对于浮点数 120.0，使用 'f' 数据类型，调用 np.cosh 会引发 FloatingPointError
            assert_raises(FloatingPointError, np.cosh,
                          np.array(120.0, dtype='f'))
            # 断言对于浮点数 1200.0，使用 'd' 数据类型，调用 np.cosh 会引发 FloatingPointError
            assert_raises(FloatingPointError, np.cosh,
                          np.array(1200.0, dtype='d'))

    def test_tanh(self):
        # 输入数组，包含 NaN 和无穷大等特殊值
        in_ = [np.nan, -np.nan, np.inf, -np.inf]
        # 预期输出数组，对应于 np.tanh 的输出
        out = [np.nan, np.nan, 1.0, -1.0]
        # 对于每种数据类型 'e', 'f', 'd'，执行以下测试
        for dt in ['e', 'f', 'd']:
            # 创建指定数据类型的输入数组
            in_arr = np.array(in_, dtype=dt)
            # 创建预期输出数组
            out_arr = np.array(out, dtype=dt)
            # 断言 np.tanh(in_arr) 与 out_arr 之间的最大ulp误差不超过3
            assert_array_max_ulp(np.tanh(in_arr), out_arr, 3)

    def test_arcsinh(self):
        # 输入数组，包含 NaN 和无穷大等特殊值
        in_ = [np.nan, -np.nan, np.inf, -np.inf]
        # 预期输出数组，与输入相同的特殊值
        out = [np.nan, np.nan, np.inf, -np.inf]
        # 对于每种数据类型 'e', 'f', 'd'，执行以下测试
        for dt in ['e', 'f', 'd']:
            # 创建指定数据类型的输入数组
            in_arr = np.array(in_, dtype=dt)
            # 创建预期输出数组
            out_arr = np.array(out, dtype=dt)
            # 断言 np.arcsinh(in_arr) 等于 out_arr
            assert_equal(np.arcsinh(in_arr), out_arr)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_arccosh(self):
        # 在上下文中忽略所有错误
        with np.errstate(all='ignore'):
            # 输入数组，包含 NaN 和无穷大等特殊值，以及额外的 1.0 和 0.0
            in_ = [np.nan, -np.nan, np.inf, -np.inf, 1.0, 0.0]
            # 预期输出数组，与输入相同的特殊值，以及额外的 0.0 和 NaN
            out = [np.nan, np.nan, np.inf, np.nan, 0.0, np.nan]
            # 对于每种数据类型 'e', 'f', 'd'，执行以下测试
            for dt in ['e', 'f', 'd']:
                # 创建指定数据类型的输入数组
                in_arr = np.array(in_, dtype=dt)
                # 创建预期输出数组
                out_arr = np.array(out, dtype=dt)
                # 断言 np.arccosh(in_arr) 等于 out_arr
                assert_equal(np.arccosh(in_arr), out_arr)

        # 对于值为 0.0 和 -无穷大，在上下文中设置无效操作 'invalid' 为 'raise' 模式
        for value in [0.0, -np.inf]:
            with np.errstate(invalid='raise'):
                # 对于每种数据类型 'e', 'f', 'd'，断言调用 np.arccosh(np.array(value, dtype=dt)) 会引发 FloatingPointError
                for dt in ['e', 'f', 'd']:
                    assert_raises(FloatingPointError, np.arccosh,
                                  np.array(value, dtype=dt))

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_arctanh(self):
        # 忽略所有的浮点错误
        with np.errstate(all='ignore'):
            # 输入数组，包括 NaN、无限大和常规浮点数
            in_ = [np.nan, -np.nan, np.inf, -np.inf, 1.0, -1.0, 2.0]
            # 预期输出数组，与输入对应位置有关的预期输出值
            out = [np.nan, np.nan, np.nan, np.nan, np.inf, -np.inf, np.nan]
            # 对于每种数据类型进行测试
            for dt in ['e', 'f', 'd']:
                # 创建输入数组，指定数据类型
                in_arr = np.array(in_, dtype=dt)
                # 创建预期输出数组，指定数据类型
                out_arr = np.array(out, dtype=dt)
                # 断言 np.arctanh(in_arr) 等于 out_arr
                assert_equal(np.arctanh(in_arr), out_arr)

        # 对于特定数值，验证是否会引发浮点错误异常
        for value in [1.01, np.inf, -np.inf, 1.0, -1.0]:
            # 设置无效（invalid）和除零（divide）错误时引发异常
            with np.errstate(invalid='raise', divide='raise'):
                # 对于每种数据类型进行测试
                for dt in ['e', 'f', 'd']:
                    # 断言调用 np.arctanh(np.array(value, dtype=dt)) 会引发 FloatingPointError 异常
                    assert_raises(FloatingPointError, np.arctanh,
                                  np.array(value, dtype=dt))

        # 确保 glibc < 2.18 的情况下不会使用 atanh，参考 issue 25087
        assert np.signbit(np.arctanh(-1j).real)

    # 查看：https://github.com/numpy/numpy/issues/20448
    @pytest.mark.xfail(
        _glibc_older_than("2.17"),
        reason="Older glibc versions may not raise appropriate FP exceptions"
    )
    def test_exp2(self):
        # 忽略所有的浮点错误
        with np.errstate(all='ignore'):
            # 输入数组，包括 NaN、无限大
            in_ = [np.nan, -np.nan, np.inf, -np.inf]
            # 预期输出数组，与输入对应位置有关的预期输出值
            out = [np.nan, np.nan, np.inf, 0.0]
            # 对于每种数据类型进行测试
            for dt in ['e', 'f', 'd']:
                # 创建输入数组，指定数据类型
                in_arr = np.array(in_, dtype=dt)
                # 创建预期输出数组，指定数据类型
                out_arr = np.array(out, dtype=dt)
                # 断言 np.exp2(in_arr) 等于 out_arr
                assert_equal(np.exp2(in_arr), out_arr)

        # 对于特定数值，验证是否会引发浮点错误异常
        for value in [2000.0, -2000.0]:
            # 设置溢出（over）和下溢（under）错误时引发异常
            with np.errstate(over='raise', under='raise'):
                # 对于每种数据类型进行测试
                for dt in ['e', 'f', 'd']:
                    # 断言调用 np.exp2(np.array(value, dtype=dt)) 会引发 FloatingPointError 异常
                    assert_raises(FloatingPointError, np.exp2,
                                  np.array(value, dtype=dt))

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_expm1(self):
        # 忽略所有的浮点错误
        with np.errstate(all='ignore'):
            # 输入数组，包括 NaN、无限大
            in_ = [np.nan, -np.nan, np.inf, -np.inf]
            # 预期输出数组，与输入对应位置有关的预期输出值
            out = [np.nan, np.nan, np.inf, -1.0]
            # 对于每种数据类型进行测试
            for dt in ['e', 'f', 'd']:
                # 创建输入数组，指定数据类型
                in_arr = np.array(in_, dtype=dt)
                # 创建预期输出数组，指定数据类型
                out_arr = np.array(out, dtype=dt)
                # 断言 np.expm1(in_arr) 等于 out_arr
                assert_equal(np.expm1(in_arr), out_arr)

        # 对于特定数值，验证是否会引发浮点错误异常
        for value in [200.0, 2000.0]:
            # 设置溢出（over）错误时引发异常
            with np.errstate(over='raise'):
                # 对于每种数据类型进行测试
                for dt in ['e', 'f']:
                    # 断言调用 np.expm1(np.array(value, dtype=dt)) 会引发 FloatingPointError 异常
                    assert_raises(FloatingPointError, np.expm1,
                                  np.array(value, dtype=dt))

    # 测试以确保 SIMD 不会由于浮点错误引发意外异常
    INF_INVALID_ERR = [
        np.cos, np.sin, np.tan, np.arccos, np.arcsin, np.spacing, np.arctanh
    ]
    NEG_INVALID_ERR = [
        np.log, np.log2, np.log10, np.log1p, np.sqrt, np.arccosh,
        np.arctanh
    ]
    ONE_INVALID_ERR = [
        np.arctanh,
    ]
    LTONE_INVALID_ERR = [
        np.arccosh,
    ]
    BYZERO_ERR = [
        np.log, np.log2, np.log10, np.reciprocal, np.arccosh
    ]

    @pytest.mark.parametrize("ufunc", UFUNCS_UNARY_FP)
    @pytest.mark.parametrize("dtype", ('e', 'f', 'd'))
    # 使用 pytest 的参数化装饰器，为单元测试函数提供不同的测试数据和期望结果
    @pytest.mark.parametrize("data, escape", (
        # 测试小于或等于0.03的浮点数数组，期望触发LTONE_INVALID_ERR异常
        ([0.03], LTONE_INVALID_ERR),
        ([0.03]*32, LTONE_INVALID_ERR),
        # 测试负数数组，期望触发NEG_INVALID_ERR异常
        ([-1.0], NEG_INVALID_ERR),
        ([-1.0]*32, NEG_INVALID_ERR),
        # 测试正数等于1.0的数组，期望触发ONE_INVALID_ERR异常
        ([1.0], ONE_INVALID_ERR),
        ([1.0]*32, ONE_INVALID_ERR),
        # 测试零值数组，期望触发BYZERO_ERR异常
        ([0.0], BYZERO_ERR),
        ([0.0]*32, BYZERO_ERR),
        ([-0.0], BYZERO_ERR),
        ([-0.0]*32, BYZERO_ERR),
        # 测试包含NaN的数组，期望触发LTONE_INVALID_ERR异常
        ([0.5, 0.5, 0.5, np.nan], LTONE_INVALID_ERR),
        ([0.5, 0.5, 0.5, np.nan]*32, LTONE_INVALID_ERR),
        ([np.nan, 1.0, 1.0, 1.0], ONE_INVALID_ERR),
        ([np.nan, 1.0, 1.0, 1.0]*32, ONE_INVALID_ERR),
        ([np.nan], []),
        ([np.nan]*32, []),
        # 测试包含正无穷大的数组，期望触发INF_INVALID_ERR异常或其组合
        ([0.5, 0.5, 0.5, np.inf], INF_INVALID_ERR + LTONE_INVALID_ERR),
        ([0.5, 0.5, 0.5, np.inf]*32, INF_INVALID_ERR + LTONE_INVALID_ERR),
        ([np.inf, 1.0, 1.0, 1.0], INF_INVALID_ERR),
        ([np.inf, 1.0, 1.0, 1.0]*32, INF_INVALID_ERR),
        ([np.inf], INF_INVALID_ERR),
        ([np.inf]*32, INF_INVALID_ERR),
        # 测试包含负无穷大的数组，期望触发NEG_INVALID_ERR、INF_INVALID_ERR和LTONE_INVALID_ERR的组合
        ([0.5, 0.5, 0.5, -np.inf],
         NEG_INVALID_ERR + INF_INVALID_ERR + LTONE_INVALID_ERR),
        ([0.5, 0.5, 0.5, -np.inf]*32,
         NEG_INVALID_ERR + INF_INVALID_ERR + LTONE_INVALID_ERR),
        ([-np.inf, 1.0, 1.0, 1.0], NEG_INVALID_ERR + INF_INVALID_ERR),
        ([-np.inf, 1.0, 1.0, 1.0]*32, NEG_INVALID_ERR + INF_INVALID_ERR),
        ([-np.inf], NEG_INVALID_ERR + INF_INVALID_ERR),
        ([-np.inf]*32, NEG_INVALID_ERR + INF_INVALID_ERR),
    ))
    # 定义测试浮点数异常处理的单元测试函数
    def test_unary_spurious_fpexception(self, ufunc, dtype, data, escape):
        # 如果 escape 非空且 ufunc 在 escape 列表中，则跳过测试
        if escape and ufunc in escape:
            return
        # TODO: 如果 ufunc 是 np.spacing 或 np.ceil 且 dtype 是 'e'，则跳过测试
        if ufunc in (np.spacing, np.ceil) and dtype == 'e':
            return
        # 创建 numpy 数组，使用给定的数据和数据类型
        array = np.array(data, dtype=dtype)
        # 使用 assert_no_warnings 上下文确保在运行 ufunc 时不会触发警告
        with assert_no_warnings():
            ufunc(array)

    # 使用 pytest 的参数化装饰器，为单元测试函数提供不同的数据类型
    @pytest.mark.parametrize("dtype", ('e', 'f', 'd'))
    # 定义测试除法异常处理的单元测试函数
    def test_divide_spurious_fpexception(self, dtype):
        # 获取给定数据类型的信息
        dt = np.dtype(dtype)
        dt_info = np.finfo(dt)
        # 获取最小子正常数值
        subnorm = dt_info.smallest_subnormal
        # 使用 assert_no_warnings 上下文确保在运行除法时不会触发警告
        with assert_no_warnings():
            # 创建一个具有128 + 1个元素的全零数组，并使用给定数据类型进行除法运算
            np.zeros(128 + 1, dtype=dt) / subnorm
# 定义一个测试类 TestFPClass，用于测试浮点数处理的各种情况
class TestFPClass:
    # 使用 pytest 的参数化装饰器，对 test_fpclass 方法进行参数化测试，参数为 stride
    @pytest.mark.parametrize("stride", [-5, -4, -3, -2, -1, 1,
                                2, 4, 5, 6, 7, 8, 9, 10])
    # 测试方法，测试浮点数分类函数
    def test_fpclass(self, stride):
        # 创建一个双精度浮点数数组 arr_f64，包含 NaN、±∞、正负数等特殊值
        arr_f64 = np.array([np.nan, -np.nan, np.inf, -np.inf, -1.0, 1.0, -0.0, 0.0, 2.2251e-308, -2.2251e-308], dtype='d')
        # 创建一个单精度浮点数数组 arr_f32，包含 NaN、±∞、正负数等特殊值
        arr_f32 = np.array([np.nan, -np.nan, np.inf, -np.inf, -1.0, 1.0, -0.0, 0.0, 1.4013e-045, -1.4013e-045], dtype='f')
        # 创建一个布尔数组 nan，用于比较 arr_f32 和 arr_f64 中的 NaN 值
        nan     = np.array([True, True, False, False, False, False, False, False, False, False])
        # 创建一个布尔数组 inf，用于比较 arr_f32 和 arr_f64 中的 ±∞ 值
        inf     = np.array([False, False, True, True, False, False, False, False, False, False])
        # 创建一个布尔数组 sign，用于比较 arr_f32 和 arr_f64 中的符号位
        sign    = np.array([False, True, False, True, True, False, True, False, False, True])
        # 创建一个布尔数组 finite，用于比较 arr_f32 和 arr_f64 中的有限数值
        finite  = np.array([False, False, False, False, True, True, True, True, True, True])
        
        # 断言函数，验证 arr_f32 和 arr_f64 中提取出来的 NaN 值是否与预期的 nan 数组一致
        assert_equal(np.isnan(arr_f32[::stride]), nan[::stride])
        assert_equal(np.isnan(arr_f64[::stride]), nan[::stride])
        # 断言函数，验证 arr_f32 和 arr_f64 中提取出来的 ±∞ 值是否与预期的 inf 数组一致
        assert_equal(np.isinf(arr_f32[::stride]), inf[::stride])
        assert_equal(np.isinf(arr_f64[::stride]), inf[::stride])
        
        # 如果运行环境是 RISC-V 64 位架构
        if platform.machine() == 'riscv64':
            # 在 RISC-V 架构上，许多产生 NaN 的操作（例如从 f64 转换为 f32 的 -NaN），返回规范化的 NaN。
            # 规范化的 NaN 总是正数。详见 RISC-V Unprivileged ISA 第 11.3 节 NaN 生成和传播。
            # 在这些测试中，我们禁用 riscv64 上 -np.nan 的符号位测试，因为我们不能假设其符号在这些测试中被尊重。
            # 复制 arr_f64 和 arr_f32 数组
            arr_f64_rv = np.copy(arr_f64)
            arr_f32_rv = np.copy(arr_f32)
            arr_f64_rv[1] = -1.0
            arr_f32_rv[1] = -1.0
            # 断言函数，验证在 RISC-V 环境下，arr_f32_rv 和 arr_f64_rv 中提取出来的符号位是否与预期的 sign 数组一致
            assert_equal(np.signbit(arr_f32_rv[::stride]), sign[::stride])
            assert_equal(np.signbit(arr_f64_rv[::stride]), sign[::stride])
        else:
            # 断言函数，验证在非 RISC-V 环境下，arr_f32 和 arr_f64 中提取出来的符号位是否与预期的 sign 数组一致
            assert_equal(np.signbit(arr_f32[::stride]), sign[::stride])
            assert_equal(np.signbit(arr_f64[::stride]), sign[::stride])
        
        # 断言函数，验证 arr_f32 和 arr_f64 中提取出来的有限数值是否与预期的 finite 数组一致
        assert_equal(np.isfinite(arr_f32[::stride]), finite[::stride])
        assert_equal(np.isfinite(arr_f64[::stride]), finite[::stride])

    # 使用 pytest 的参数化装饰器，对 TestLDExp 类的 test_ldexp 方法进行参数化测试，参数为 dtype 和 stride
    @pytest.mark.parametrize("dtype", ['d', 'f'])
class TestLDExp:
    # 使用 pytest 的参数化装饰器，对 test_ldexp 方法进行参数化测试，参数为 stride 和 dtype
    @pytest.mark.parametrize("stride", [-4,-2,-1,1,2,4])
    @pytest.mark.parametrize("dtype", ['f', 'd'])
    # 测试方法，测试 ldexp 函数
    def test_ldexp(self, dtype, stride):
        # 创建一个浮点数数组 mant，用于 ldexp 函数的尾数参数，根据 dtype 决定精度
        mant = np.array([0.125, 0.25, 0.5, 1., 1., 2., 4., 8.], dtype=dtype)
        # 创建一个整数数组 exp，用于 ldexp 函数的指数参数
        exp  = np.array([3, 2, 1, 0, 0, -1, -2, -3], dtype='i')
        # 创建一个输出数组 out，用于接收 ldexp 函数的输出结果
        out  = np.zeros(8, dtype=dtype)
        
        # 断言函数，验证 ldexp 函数的输出是否与预期的全为 1 的数组一致
        assert_equal(np.ldexp(mant[::stride], exp[::stride], out=out[::stride]), np.ones(8, dtype=dtype)[::stride])
        # 断言函数，验证 out 数组是否与预期的全为 1 的数组一致
        assert_equal(out[::stride], np.ones(8, dtype=dtype)[::stride])

# 定义一个测试类 TestFRExp，用于测试 frexp 函数，但本代码中未提供完整的实现和测试代码
class TestFRExp:
    @pytest.mark.parametrize("stride", [-4,-2,-1,1,2,4])
    @pytest.mark.parametrize("dtype", ['f', 'd'])
    # 标记测试用例，仅在运行环境为 Linux 时执行；否则跳过测试并给出理由
    @pytest.mark.skipif(not sys.platform.startswith('linux'),
                        reason="np.frexp gives different answers for NAN/INF on windows and linux")
    # 标记测试预期会失败，针对 Musl libc（一个 C 库）中的问题 gh23049
    @pytest.mark.xfail(IS_MUSL, reason="gh23049")
    # 定义测试函数，接受数据类型和步长作为参数
    def test_frexp(self, dtype, stride):
        # 创建包含各种特殊浮点数和常规浮点数的 numpy 数组
        arr = np.array([np.nan, np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0], dtype=dtype)
        # 创建预期的 mantissa 数组，与 arr 数组对应
        mant_true = np.array([np.nan, np.nan, np.inf, -np.inf, 0.0, -0.0, 0.5, -0.5], dtype=dtype)
        # 创建预期的 exponent 数组，与 arr 数组对应
        exp_true  = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype='i')
        # 创建输出 mantissa 的数组，初始化为全 1
        out_mant  = np.ones(8, dtype=dtype)
        # 创建输出 exponent 的数组，初始化为全 2
        out_exp   = 2*np.ones(8, dtype='i')
        # 调用 numpy 中的 frexp 函数，计算 arr 数组的 mantissa 和 exponent，
        # 将结果存储到 out_mant 和 out_exp 数组中
        mant, exp = np.frexp(arr[::stride], out=(out_mant[::stride], out_exp[::stride]))
        # 断言计算得到的 mantissa 与预期的 mantissa 相等
        assert_equal(mant_true[::stride], mant)
        # 断言计算得到的 exponent 与预期的 exponent 相等
        assert_equal(exp_true[::stride], exp)
        # 断言输出的 mantissa 与预期的 mantissa 相等
        assert_equal(out_mant[::stride], mant_true[::stride])
        # 断言输出的 exponent 与预期的 exponent 相等
        assert_equal(out_exp[::stride], exp_true[::stride])
# 定义 AVX 支持的数学函数及其测试参数
avx_ufuncs = {'sqrt'        :[1,  0.,   100.],  # 平方根函数的最大误差、最小值和最大值范围
              'absolute'    :[0, -100., 100.],  # 绝对值函数的最大误差、最小值和最大值范围
              'reciprocal'  :[1,  1.,   100.],  # 倒数函数的最大误差、最小值和最大值范围
              'square'      :[1, -100., 100.],  # 平方函数的最大误差、最小值和最大值范围
              'rint'        :[0, -100., 100.],  # 最接近整数的函数的最大误差、最小值和最大值范围
              'floor'       :[0, -100., 100.],  # 向下取整函数的最大误差、最小值和最大值范围
              'ceil'        :[0, -100., 100.],  # 向上取整函数的最大误差、最小值和最大值范围
              'trunc'       :[0, -100., 100.]}  # 截断函数的最大误差、最小值和最大值范围

class TestAVXUfuncs:
    def test_avx_based_ufunc(self):
        strides = np.array([-4,-3,-2,-1,1,2,3,4])  # 不同步长用于测试 AVX 中的掩码指令
        np.random.seed(42)
        for func, prop in avx_ufuncs.items():
            maxulperr = prop[0]  # 最大单位最小误差
            minval = prop[1]      # 输入数组的最小值
            maxval = prop[2]      # 输入数组的最大值
            # 不同大小的数组确保在 AVX 中的掩码测试
            for size in range(1,32):
                myfunc = getattr(np, func)
                x_f32 = np.random.uniform(low=minval, high=maxval,
                                          size=size).astype(np.float32)
                x_f64 = x_f32.astype(np.float64)
                x_f128 = x_f32.astype(np.longdouble)
                y_true128 = myfunc(x_f128)
                if maxulperr == 0:
                    assert_equal(myfunc(x_f32), y_true128.astype(np.float32))  # 如果最大单位最小误差为零，检查单精度浮点数的相等性
                    assert_equal(myfunc(x_f64), y_true128.astype(np.float64))  # 检查双精度浮点数的相等性
                else:
                    assert_array_max_ulp(myfunc(x_f32),
                                         y_true128.astype(np.float32),
                                         maxulp=maxulperr)  # 否则，使用最大单位最小误差检查单精度浮点数数组的最大单位最小误差
                    assert_array_max_ulp(myfunc(x_f64),
                                         y_true128.astype(np.float64),
                                         maxulp=maxulperr)  # 使用最大单位最小误差检查双精度浮点数数组的最大单位最小误差
                # 不同的步长来测试 gather 指令
                if size > 1:
                    y_true32 = myfunc(x_f32)
                    y_true64 = myfunc(x_f64)
                    for jj in strides:
                        assert_equal(myfunc(x_f64[::jj]), y_true64[::jj])  # 检查使用不同步长的双精度浮点数数组的相等性
                        assert_equal(myfunc(x_f32[::jj]), y_true32[::jj])  # 检查使用不同步长的单精度浮点数数组的相等性

class TestAVXFloat32Transcendental:
    def test_exp_float32(self):
        np.random.seed(42)
        x_f32 = np.float32(np.random.uniform(low=0.0,high=88.1,size=1000000))
        x_f64 = np.float64(x_f32)
        assert_array_max_ulp(np.exp(x_f32), np.float32(np.exp(x_f64)), maxulp=3)  # 检查单精度浮点数的指数函数的最大单位最小误差

    def test_log_float32(self):
        np.random.seed(42)
        x_f32 = np.float32(np.random.uniform(low=0.0,high=1000,size=1000000))
        x_f64 = np.float64(x_f32)
        assert_array_max_ulp(np.log(x_f32), np.float32(np.log(x_f64)), maxulp=4)  # 检查单精度浮点数的对数函数的最大单位最小误差
    # 定义一个测试函数，用于测试单精度浮点数的 sin 和 cos 函数
    def test_sincos_float32(self):
        # 设定随机数种子，确保可重复性
        np.random.seed(42)
        # 设定数组大小
        N = 1000000
        # 计算 M 的值，M 是 N 的二十分之一
        M = np.int_(N/20)
        # 从 [0, N) 的范围内生成 M 个随机整数作为索引
        index = np.random.randint(low=0, high=N, size=M)
        # 生成 N 个在 [-100.0, 100.0) 范围内的随机单精度浮点数数组
        x_f32 = np.float32(np.random.uniform(low=-100., high=100., size=N))
        
        # 如果不是旧版 glibc (小于 2.17)，则覆盖大于 117435.992 的元素，这些情况下会使用 glibc
        if not _glibc_older_than("2.17"):
            # 将索引处的元素设置为较大的随机单精度浮点数，避免旧版 glibc 的问题
            x_f32[index] = np.float32(10E+10 * np.random.rand(M))
        
        # 将单精度浮点数数组转换为双精度浮点数数组
        x_f64 = np.float64(x_f32)
        
        # 断言单精度浮点数数组和双精度浮点数数组的 sin 值最大误差不超过 2 个单位最后位 (ULP)
        assert_array_max_ulp(np.sin(x_f32), np.float32(np.sin(x_f64)), maxulp=2)
        # 断言单精度浮点数数组和双精度浮点数数组的 cos 值最大误差不超过 2 个单位最后位 (ULP)
        assert_array_max_ulp(np.cos(x_f32), np.float32(np.cos(x_f64)), maxulp=2)
        
        # 测试别名问题(issue #17761)
        # 复制 x_f32 到 tx_f32
        tx_f32 = x_f32.copy()
        # 将 sin 计算结果存入 x_f32 本身，断言其与双精度浮点数的 sin 值的最大误差不超过 2 个 ULP
        assert_array_max_ulp(np.sin(x_f32, out=x_f32), np.float32(np.sin(x_f64)), maxulp=2)
        # 将 cos 计算结果存入 tx_f32 本身，断言其与双精度浮点数的 cos 值的最大误差不超过 2 个 ULP
        assert_array_max_ulp(np.cos(tx_f32, out=tx_f32), np.float32(np.cos(x_f64)), maxulp=2)

    # 定义一个测试函数，用于测试步长变化下的单精度浮点数函数
    def test_strided_float32(self):
        # 设定随机数种子，确保可重复性
        np.random.seed(42)
        # 设定步长数组
        strides = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
        # 设定尺寸数组，从 2 到 99
        sizes = np.arange(2, 100)
        
        # 对每个尺寸进行循环
        for ii in sizes:
            # 生成一个包含 ii 个在 [0.01, 88.1) 范围内的随机单精度浮点数数组
            x_f32 = np.float32(np.random.uniform(low=0.01, high=88.1, size=ii))
            # 将 x_f32 复制到 x_f32_large
            x_f32_large = x_f32.copy()
            # 将 x_f32_large 中索引为 3 到倒数第二个的每隔 4 个元素设置为 120000.0
            x_f32_large[3:-1:4] = 120000.0
            
            # 计算期望的 exp、log、sin、cos 值
            exp_true = np.exp(x_f32)
            log_true = np.log(x_f32)
            sin_true = np.sin(x_f32_large)
            cos_true = np.cos(x_f32_large)
            
            # 对步长数组进行循环
            for jj in strides:
                # 断言 exp 函数的步长计算结果和期望值的最大 ULP 误差不超过 2 个单位最后位
                assert_array_almost_equal_nulp(np.exp(x_f32[::jj]), exp_true[::jj], nulp=2)
                # 断言 log 函数的步长计算结果和期望值的最大 ULP 误差不超过 2 个单位最后位
                assert_array_almost_equal_nulp(np.log(x_f32[::jj]), log_true[::jj], nulp=2)
                # 断言 sin 函数的步长计算结果和期望值的最大 ULP 误差不超过 2 个单位最后位
                assert_array_almost_equal_nulp(np.sin(x_f32_large[::jj]), sin_true[::jj], nulp=2)
                # 断言 cos 函数的步长计算结果和期望值的最大 ULP 误差不超过 2 个单位最后位
                assert_array_almost_equal_nulp(np.cos(x_f32_large[::jj]), cos_true[::jj], nulp=2)
class TestLogAddExp(_FilterInvalids):
    # 测试 logaddexp 函数的值
    def test_logaddexp_values(self):
        # 定义输入数组
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        z = [6, 6, 6, 6, 6]
        # 对于不同的数据类型和小数精度
        for dt, dec_ in zip(['f', 'd', 'g'], [6, 15, 15]):
            # 将 x, y, z 数组转换为对数值数组
            xf = np.log(np.array(x, dtype=dt))
            yf = np.log(np.array(y, dtype=dt))
            zf = np.log(np.array(z, dtype=dt))
            # 断言 logaddexp 函数计算的结果与期望的 zf 数组接近
            assert_almost_equal(np.logaddexp(xf, yf), zf, decimal=dec_)

    # 测试 logaddexp 函数的边界情况
    def test_logaddexp_range(self):
        # 定义输入数组
        x = [1000000, -1000000, 1000200, -1000200]
        y = [1000200, -1000200, 1000000, -1000000]
        z = [1000200, -1000000, 1000200, -1000000]
        # 对于不同的数据类型
        for dt in ['f', 'd', 'g']:
            # 将 x, y, z 数组转换为对数值数组
            logxf = np.array(x, dtype=dt)
            logyf = np.array(y, dtype=dt)
            logzf = np.array(z, dtype=dt)
            # 断言 logaddexp 函数计算的结果与期望的 logzf 数组相等
            assert_almost_equal(np.logaddexp(logxf, logyf), logzf)

    # 测试包含无穷值的情况
    def test_inf(self):
        inf = np.inf
        # 定义输入数组
        x = [inf, -inf,  inf, -inf, inf, 1,  -inf,  1]
        y = [inf,  inf, -inf, -inf, 1,   inf, 1,   -inf]
        z = [inf,  inf,  inf, -inf, inf, inf, 1,    1]
        # 在错误状态为无效时
        with np.errstate(invalid='raise'):
            # 对于不同的数据类型
            for dt in ['f', 'd', 'g']:
                # 将 x, y, z 数组转换为对数值数组
                logxf = np.array(x, dtype=dt)
                logyf = np.array(y, dtype=dt)
                logzf = np.array(z, dtype=dt)
                # 断言 logaddexp 函数计算的结果与期望的 logzf 数组相等
                assert_equal(np.logaddexp(logxf, logyf), logzf)

    # 测试包含 NaN 的情况
    def test_nan(self):
        # 断言 logaddexp 函数处理 NaN 值时返回 NaN
        assert_(np.isnan(np.logaddexp(np.nan, np.inf)))
        assert_(np.isnan(np.logaddexp(np.inf, np.nan)))
        assert_(np.isnan(np.logaddexp(np.nan, 0)))
        assert_(np.isnan(np.logaddexp(0, np.nan)))
        assert_(np.isnan(np.logaddexp(np.nan, np.nan)))

    # 测试 logaddexp 函数的 reduce 方法
    def test_reduce(self):
        # 断言 logaddexp 函数的 identity 属性为负无穷大
        assert_equal(np.logaddexp.identity, -np.inf)
        # 断言空数组调用 reduce 方法后返回负无穷大
        assert_equal(np.logaddexp.reduce([]), -np.inf)


class TestLog1p:
    # 测试 log1p 函数
    def test_log1p(self):
        # 断言 log1p 函数的正确性
        assert_almost_equal(ncu.log1p(0.2), ncu.log(1.2))
        assert_almost_equal(ncu.log1p(1e-6), ncu.log(1+1e-6))

    # 测试 log1p 函数的特殊情况
    def test_special(self):
        # 在忽略无效和除法错误状态下，断言 log1p 处理 NaN 和无穷大时的返回值
        with np.errstate(invalid="ignore", divide="ignore"):
            assert_equal(ncu.log1p(np.nan), np.nan)
            assert_equal(ncu.log1p(np.inf), np.inf)
            assert_equal(ncu.log1p(-1.), -np.inf)
            assert_equal(ncu.log1p(-2.), np.nan)
            assert_equal(ncu.log1p(-np.inf), np.nan)


class TestExpm1:
    # 测试 expm1 函数
    def test_expm1(self):
        # 断言 expm1 函数的正确性
        assert_almost_equal(ncu.expm1(0.2), ncu.exp(0.2)-1)
        assert_almost_equal(ncu.expm1(1e-6), ncu.exp(1e-6)-1)

    # 测试 expm1 函数的特殊情况
    def test_special(self):
        # 断言 expm1 处理特殊输入时的返回值
        assert_equal(ncu.expm1(np.inf), np.inf)
        assert_equal(ncu.expm1(0.), 0.)
        assert_equal(ncu.expm1(-0.), -0.)
        assert_equal(ncu.expm1(np.inf), np.inf)
        assert_equal(ncu.expm1(-np.inf), -1.)

    # 测试复数输入的情况
    def test_complex(self):
        x = np.asarray(1e-12)
        # 断言复数输入时 expm1 的正确性
        assert_allclose(x, ncu.expm1(x))
        x = x.astype(np.complex128)
        assert_allclose(x, ncu.expm1(x))
    # 定义单元测试方法，测试简单情况下的 hypot 函数调用
    def test_simple(self):
        # 断言 hypot 函数计算结果接近 sqrt(2)
        assert_almost_equal(ncu.hypot(1, 1), ncu.sqrt(2))
        # 断言 hypot 函数计算 (0, 0) 时结果为 0
        assert_almost_equal(ncu.hypot(0, 0), 0)

    # 定义单元测试方法，测试 hypot 函数的 reduce 方法
    def test_reduce(self):
        # 断言 hypot 函数对于数组 [3.0, 4.0] 的 reduce 结果接近 5.0
        assert_almost_equal(ncu.hypot.reduce([3.0, 4.0]), 5.0)
        # 断言 hypot 函数对于数组 [3.0, 4.0, 0] 的 reduce 结果接近 5.0
        assert_almost_equal(ncu.hypot.reduce([3.0, 4.0, 0]), 5.0)
        # 断言 hypot 函数对于数组 [9.0, 12.0, 20.0] 的 reduce 结果接近 25.0
        assert_almost_equal(ncu.hypot.reduce([9.0, 12.0, 20.0]), 25.0)
        # 断言 hypot 函数对于空数组 [] 的 reduce 结果为 0.0
        assert_equal(ncu.hypot.reduce([]), 0.0)
# 定义一个函数，用于检查通过忽略无效操作状态后计算的 hypot(x, y) 是否为 NaN
def assert_hypot_isnan(x, y):
    with np.errstate(invalid='ignore'):  # 设置 numpy 的错误状态为忽略无效操作
        assert_(np.isnan(ncu.hypot(x, y)),  # 断言 hypot(x, y) 的结果是否为 NaN
                "hypot(%s, %s) is %s, not nan" % (x, y, ncu.hypot(x, y)))  # 如果断言失败，输出错误信息

# 定义一个函数，用于检查通过忽略无效操作状态后计算的 hypot(x, y) 是否为无穷大
def assert_hypot_isinf(x, y):
    with np.errstate(invalid='ignore'):  # 设置 numpy 的错误状态为忽略无效操作
        assert_(np.isinf(ncu.hypot(x, y)),  # 断言 hypot(x, y) 的结果是否为无穷大
                "hypot(%s, %s) is %s, not inf" % (x, y, ncu.hypot(x, y)))  # 如果断言失败，输出错误信息

# 定义一个测试类，用于测试 hypot 函数对特殊值的处理
class TestHypotSpecialValues:
    # 测试当输入为 NaN 时，hypot 函数的输出是否为 NaN
    def test_nan_outputs(self):
        assert_hypot_isnan(np.nan, np.nan)
        assert_hypot_isnan(np.nan, 1)

    # 测试当输入为特定组合包括 NaN 和无穷大时，hypot 函数的输出是否为无穷大
    def test_nan_outputs2(self):
        assert_hypot_isinf(np.nan, np.inf)
        assert_hypot_isinf(np.inf, np.nan)
        assert_hypot_isinf(np.inf, 0)
        assert_hypot_isinf(0, np.inf)
        assert_hypot_isinf(np.inf, np.inf)
        assert_hypot_isinf(np.inf, 23.0)

    # 测试当输入不引发浮点异常时，hypot 函数的输出是否正常，不会产生浮点异常
    def test_no_fpe(self):
        assert_no_warnings(ncu.hypot, np.inf, 0)

# 定义一个函数，用于检查通过断言 arctan2(x, y) 是否为 NaN
def assert_arctan2_isnan(x, y):
    assert_(np.isnan(ncu.arctan2(x, y)),  # 断言 arctan2(x, y) 的结果是否为 NaN
            "arctan(%s, %s) is %s, not nan" % (x, y, ncu.arctan2(x, y)))  # 如果断言失败，输出错误信息

# 定义一个函数，用于检查通过断言 arctan2(x, y) 是否为正无穷
def assert_arctan2_ispinf(x, y):
    assert_((np.isinf(ncu.arctan2(x, y)) and ncu.arctan2(x, y) > 0),  # 断言 arctan2(x, y) 的结果是否为正无穷
            "arctan(%s, %s) is %s, not +inf" % (x, y, ncu.arctan2(x, y)))  # 如果断言失败，输出错误信息

# 定义一个函数，用于检查通过断言 arctan2(x, y) 是否为负无穷
def assert_arctan2_isninf(x, y):
    assert_((np.isinf(ncu.arctan2(x, y)) and ncu.arctan2(x, y) < 0),  # 断言 arctan2(x, y) 的结果是否为负无穷
            "arctan(%s, %s) is %s, not -inf" % (x, y, ncu.arctan2(x, y)))  # 如果断言失败，输出错误信息

# 定义一个函数，用于检查通过断言 arctan2(x, y) 是否为正零
def assert_arctan2_ispzero(x, y):
    assert_((ncu.arctan2(x, y) == 0 and not np.signbit(ncu.arctan2(x, y))),  # 断言 arctan2(x, y) 的结果是否为正零
            "arctan(%s, %s) is %s, not +0" % (x, y, ncu.arctan2(x, y)))  # 如果断言失败，输出错误信息

# 定义一个函数，用于检查通过断言 arctan2(x, y) 是否为负零
def assert_arctan2_isnzero(x, y):
    assert_((ncu.arctan2(x, y) == 0 and np.signbit(ncu.arctan2(x, y))),  # 断言 arctan2(x, y) 的结果是否为负零
            "arctan(%s, %s) is %s, not -0" % (x, y, ncu.arctan2(x, y)))  # 如果断言失败，输出错误信息

# 定义一个测试类，用于测试 arctan2 函数对特殊值的处理
class TestArctan2SpecialValues:
    # 测试 arctan2(1, 1) 的输出是否接近 pi/4
    def test_one_one(self):
        assert_almost_equal(ncu.arctan2(1, 1), 0.25 * np.pi)
        assert_almost_equal(ncu.arctan2(-1, 1), -0.25 * np.pi)
        assert_almost_equal(ncu.arctan2(1, -1), 0.75 * np.pi)

    # 测试特定的零值组合输入时，arctan2 函数的输出是否符合预期
    def test_zero_nzero(self):
        assert_almost_equal(ncu.arctan2(ncu.PZERO, ncu.NZERO), np.pi)
        assert_almost_equal(ncu.arctan2(ncu.NZERO, ncu.NZERO), -np.pi)

    # 测试特定的零值组合输入时，arctan2 函数的输出是否符合预期
    def test_zero_pzero(self):
        assert_arctan2_ispzero(ncu.PZERO, ncu.PZERO)
        assert_arctan2_isnzero(ncu.NZERO, ncu.PZERO)

    # 测试当零值与负数组合输入时，arctan2 函数的输出是否符合预期
    def test_zero_negative(self):
        assert_almost_equal(ncu.arctan2(ncu.PZERO, -1), np.pi)
        assert_almost_equal(ncu.arctan2(ncu.NZERO, -1), -np.pi)

    # 测试当零值与正数组合输入时，arctan2 函数的输出是否符合预期
    def test_zero_positive(self):
        assert_arctan2_ispzero(ncu.PZERO, 1)
        assert_arctan2_isnzero(ncu.NZERO, 1)

    # 测试当正数与零值组合输入时，arctan2 函数的输出是否接近 +pi/2
    def test_positive_zero(self):
        assert_almost_equal(ncu.arctan2(1, ncu.PZERO), 0.5 * np.pi)
        assert_almost_equal(ncu.arctan2(1, ncu.NZERO), 0.5 * np.pi)
    def test_negative_zero(self):
        # 测试 atan2(y, +-0)，对于 y < 0 返回 -pi/2
        assert_almost_equal(ncu.arctan2(-1, ncu.PZERO), -0.5 * np.pi)
        assert_almost_equal(ncu.arctan2(-1, ncu.NZERO), -0.5 * np.pi)

    def test_any_ninf(self):
        # 测试 atan2(+-y, -infinity)，对于有限的 y > 0 返回 +-pi
        assert_almost_equal(ncu.arctan2(1, -np.inf),  np.pi)
        assert_almost_equal(ncu.arctan2(-1, -np.inf), -np.pi)

    def test_any_pinf(self):
        # 测试 atan2(+-y, +infinity)，对于有限的 y > 0 返回 +-0
        assert_arctan2_ispzero(1, np.inf)
        assert_arctan2_isnzero(-1, np.inf)

    def test_inf_any(self):
        # 测试 atan2(+-infinity, x)，对于有限的 x 返回 +-pi/2
        assert_almost_equal(ncu.arctan2( np.inf, 1),  0.5 * np.pi)
        assert_almost_equal(ncu.arctan2(-np.inf, 1), -0.5 * np.pi)

    def test_inf_ninf(self):
        # 测试 atan2(+-infinity, -infinity)，返回 +-3*pi/4
        assert_almost_equal(ncu.arctan2( np.inf, -np.inf),  0.75 * np.pi)
        assert_almost_equal(ncu.arctan2(-np.inf, -np.inf), -0.75 * np.pi)

    def test_inf_pinf(self):
        # 测试 atan2(+-infinity, +infinity)，返回 +-pi/4
        assert_almost_equal(ncu.arctan2( np.inf, np.inf),  0.25 * np.pi)
        assert_almost_equal(ncu.arctan2(-np.inf, np.inf), -0.25 * np.pi)

    def test_nan_any(self):
        # 测试 atan2(nan, x)，对于任何 x（包括 inf）返回 nan
        assert_arctan2_isnan(np.nan, np.inf)
        assert_arctan2_isnan(np.inf, np.nan)
        assert_arctan2_isnan(np.nan, np.nan)
class TestLdexp:
    # 定义测试类 TestLdexp，用于测试 ldexp 函数

    def _check_ldexp(self, tp):
        # 定义辅助函数 _check_ldexp，用于检查 ldexp 函数在不同类型上的计算结果
        assert_almost_equal(ncu.ldexp(np.array(2., np.float32),
                                      np.array(3, tp)), 16.)
        # 断言 ldexp 在 np.float32 类型上的计算结果接近于 16.0
        assert_almost_equal(ncu.ldexp(np.array(2., np.float64),
                                      np.array(3, tp)), 16.)
        # 断言 ldexp 在 np.float64 类型上的计算结果接近于 16.0
        assert_almost_equal(ncu.ldexp(np.array(2., np.longdouble),
                                      np.array(3, tp)), 16.)
        # 断言 ldexp 在 np.longdouble 类型上的计算结果接近于 16.0

    def test_ldexp(self):
        # 定义测试 ldexp 函数的方法 test_ldexp

        # 默认的 Python int 类型应该正常工作
        assert_almost_equal(ncu.ldexp(2., 3),  16.)
        # 断言 ldexp 在输入参数为 2.0 和 3 的情况下的计算结果接近于 16.0
        
        # 以下不同类型的整数应该都能被接受
        self._check_ldexp(np.int8)
        # 调用 _check_ldexp 函数，检查 np.int8 类型的 ldexp 计算结果
        self._check_ldexp(np.int16)
        # 调用 _check_ldexp 函数，检查 np.int16 类型的 ldexp 计算结果
        self._check_ldexp(np.int32)
        # 调用 _check_ldexp 函数，检查 np.int32 类型的 ldexp 计算结果
        self._check_ldexp('i')
        # 调用 _check_ldexp 函数，检查 'i' 类型的 ldexp 计算结果
        self._check_ldexp('l')
        # 调用 _check_ldexp 函数，检查 'l' 类型的 ldexp 计算结果

    def test_ldexp_overflow(self):
        # 定义测试 ldexp 函数在溢出情况下的方法 test_ldexp_overflow

        # 在溢出时忽略警告信息
        with np.errstate(over="ignore"):
            imax = np.iinfo(np.dtype('l')).max
            # 获取 np.long 类型的最大值
            imin = np.iinfo(np.dtype('l')).min
            # 获取 np.long 类型的最小值
            
            # 断言 ldexp 在输入参数为 2.0 和 imax 时的计算结果为正无穷大
            assert_equal(ncu.ldexp(2., imax), np.inf)
            # 断言 ldexp 在输入参数为 2.0 和 imin 时的计算结果为 0
            assert_equal(ncu.ldexp(2., imin), 0)


class TestMaximum(_FilterInvalids):
    # 定义测试类 TestMaximum，继承自 _FilterInvalids 类

    def test_reduce(self):
        # 定义测试 reduce 方法

        dflt = np.typecodes['AllFloat']
        # 获取所有浮点数类型的类型码
        dint = np.typecodes['AllInteger']
        # 获取所有整数类型的类型码
        seq1 = np.arange(11)
        # 创建一个包含 0 到 10 的整数序列
        seq2 = seq1[::-1]
        # 将 seq1 反向排列得到 seq2
        func = np.maximum.reduce
        # 获取 np.maximum 函数的 reduce 方法

        for dt in dint:
            tmp1 = seq1.astype(dt)
            # 将 seq1 转换为指定类型 dt 的数组 tmp1
            tmp2 = seq2.astype(dt)
            # 将 seq2 转换为指定类型 dt 的数组 tmp2

            # 断言对 tmp1 使用 np.maximum.reduce 方法的结果为 10
            assert_equal(func(tmp1), 10)
            # 断言对 tmp2 使用 np.maximum.reduce 方法的结果为 10

        for dt in dflt:
            tmp1 = seq1.astype(dt)
            # 将 seq1 转换为指定类型 dt 的数组 tmp1
            tmp2 = seq2.astype(dt)
            # 将 seq2 转换为指定类型 dt 的数组 tmp2

            # 断言对 tmp1 使用 np.maximum.reduce 方法的结果为 10
            assert_equal(func(tmp1), 10)
            # 断言对 tmp2 使用 np.maximum.reduce 方法的结果为 10
            
            tmp1[::2] = np.nan
            # 将 tmp1 中偶数索引位置的值设置为 NaN
            tmp2[::2] = np.nan
            # 将 tmp2 中偶数索引位置的值设置为 NaN
            
            # 断言对 tmp1 使用 np.maximum.reduce 方法的结果为 NaN
            assert_equal(func(tmp1), np.nan)
            # 断言对 tmp2 使用 np.maximum.reduce 方法的结果为 NaN

    def test_reduce_complex(self):
        # 定义测试复数类型的 reduce 方法

        assert_equal(np.maximum.reduce([1, 2j]), 1)
        # 断言对 [1, 2j] 使用 np.maximum.reduce 方法的结果为 1
        assert_equal(np.maximum.reduce([1+3j, 2j]), 1+3j)
        # 断言对 [1+3j, 2j] 使用 np.maximum.reduce 方法的结果为 1+3j

    def test_float_nans(self):
        # 定义测试包含 NaN 的浮点数数组的方法

        nan = np.nan
        # 定义 NaN 常量
        arg1 = np.array([0,   nan, nan])
        # 创建包含 NaN 的数组 arg1
        arg2 = np.array([nan, 0,   nan])
        # 创建包含 NaN 的数组 arg2
        out = np.array([nan, nan, nan])
        # 创建期望的输出数组 out

        # 断言对 arg1 和 arg2 使用 np.maximum 方法的结果与 out 相等
        assert_equal(np.maximum(arg1, arg2), out)

    def test_object_nans(self):
        # 定义测试包含 NaN 的对象数组的方法

        # 多次检查以确保如果使用了比较而不是富比较，则可能失败
        for i in range(1):
            x = np.array(float('nan'), object)
            # 创建包含 NaN 的对象数组 x
            y = 1.0
            # 定义一个浮点数 y
            z = np.array(float('nan'), object)
            # 创建包含 NaN 的对象数组 z
            
            # 断言对 x 和 y 使用 np.maximum 方法的结果为 1.0
            assert_(np.maximum(x, y) == 1.0)
            # 断言对 z 和 y 使用 np.maximum 方法的结果为 1.0

    def test_complex_nans(self):
        # 定义测试包含复数 NaN 的方法

        nan = np.nan
        # 定义 NaN 常量
        for cnan in [complex(nan, 0), complex(0, nan), complex(nan, nan)]:
            arg1 = np.array([0, cnan, cnan], dtype=complex)
            # 创建包含复数 NaN 的数组 arg1
            arg2 = np.array([cnan, 0, cnan], dtype=complex)
            # 创建包含复数 NaN 的数组 arg2
            out = np.array([nan, nan, nan], dtype=complex)
            # 创建期望的输出数组 out
            
            # 断言对 arg1 和 arg2 使用 np.maximum 方法的结果与 out 相等
            assert_equal(np.maximum(arg1, arg2), out)
    def test_object_array(self):
        # 创建一个包含对象类型的 NumPy 数组，从 0 到 4
        arg1 = np.arange(5, dtype=object)
        # 将 arg1 中每个元素加 1，形成新的数组 arg2
        arg2 = arg1 + 1
        # 断言 np.maximum 函数对 arg1 和 arg2 的操作结果等于 arg2
        assert_equal(np.maximum(arg1, arg2), arg2)

    def test_strided_array(self):
        # 创建包含特殊浮点值的 NumPy 数组
        arr1 = np.array([-4.0, 1.0, 10.0,  0.0, np.nan, -np.nan, np.inf, -np.inf])
        arr2 = np.array([-2.0,-1.0, np.nan, 1.0, 0.0,    np.nan, 1.0,    -3.0])
        maxtrue  = np.array([-2.0, 1.0, np.nan, 1.0, np.nan, np.nan, np.inf, -3.0])
        # 创建初始输出数组
        out = np.ones(8)
        # 验证 np.maximum 函数的多个断言
        assert_equal(np.maximum(arr1,arr2), maxtrue)
        assert_equal(np.maximum(arr1[::2],arr2[::2]), maxtrue[::2])
        assert_equal(np.maximum(arr1[:4:], arr2[::2]), np.array([-2.0, np.nan, 10.0, 1.0]))
        assert_equal(np.maximum(arr1[::3], arr2[:3:]), np.array([-2.0, 0.0, np.nan]))
        assert_equal(np.maximum(arr1[:6:2], arr2[::3], out=out[::3]), np.array([-2.0, 10., np.nan]))
        assert_equal(out, out_maxtrue)

    def test_precision(self):
        # 定义多种浮点类型
        dtypes = [np.float16, np.float32, np.float64, np.longdouble]

        for dt in dtypes:
            # 获取当前数据类型的最小值和最大值
            dtmin = np.finfo(dt).min
            dtmax = np.finfo(dt).max
            d1 = dt(0.1)
            # 找到 d1 的下一个浮点数
            d1_next = np.nextafter(d1, np.inf)

            # 定义测试用例列表
            test_cases = [
                # v1    v2          expected
                (dtmin, -np.inf,    dtmin),
                (dtmax, -np.inf,    dtmax),
                (d1,    d1_next,    d1_next),
                (dtmax, np.nan,     np.nan),
            ]

            for v1, v2, expected in test_cases:
                # 断言 np.maximum 函数对各个测试用例的操作结果
                assert_equal(np.maximum([v1], [v2]), [expected])
                assert_equal(np.maximum.reduce([v1, v2]), expected)
class TestMinimum(_FilterInvalids):
    # 继承自_FilterInvalids类的TestMinimum测试类，用于测试np.minimum函数

    def test_reduce(self):
        # 定义test_reduce方法，用于测试np.minimum.reduce功能
        dflt = np.typecodes['AllFloat']
        # 获取所有浮点类型码
        dint = np.typecodes['AllInteger']
        # 获取所有整数类型码
        seq1 = np.arange(11)
        # 创建一个包含0到10的数组seq1
        seq2 = seq1[::-1]
        # 创建seq1的逆序数组seq2
        func = np.minimum.reduce
        # 将np.minimum.reduce函数赋给变量func，用于最小值的reduce操作

        for dt in dint:
            # 遍历所有整数类型码
            tmp1 = seq1.astype(dt)
            # 将seq1转换为当前整数类型dt的数组tmp1
            tmp2 = seq2.astype(dt)
            # 将seq2转换为当前整数类型dt的数组tmp2
            assert_equal(func(tmp1), 0)
            # 断言np.minimum.reduce(tmp1)的结果为0
            assert_equal(func(tmp2), 0)
            # 断言np.minimum.reduce(tmp2)的结果为0

        for dt in dflt:
            # 遍历所有浮点类型码
            tmp1 = seq1.astype(dt)
            # 将seq1转换为当前浮点类型dt的数组tmp1
            tmp2 = seq2.astype(dt)
            # 将seq2转换为当前浮点类型dt的数组tmp2
            assert_equal(func(tmp1), 0)
            # 断言np.minimum.reduce(tmp1)的结果为0
            assert_equal(func(tmp2), 0)
            # 断言np.minimum.reduce(tmp2)的结果为0

            tmp1[::2] = np.nan
            # 将tmp1数组的偶数索引位置设置为NaN
            tmp2[::2] = np.nan
            # 将tmp2数组的偶数索引位置设置为NaN
            assert_equal(func(tmp1), np.nan)
            # 断言np.minimum.reduce(tmp1)的结果为NaN
            assert_equal(func(tmp2), np.nan)
            # 断言np.minimum.reduce(tmp2)的结果为NaN

    def test_reduce_complex(self):
        # 定义test_reduce_complex方法，用于测试复数情况下的np.minimum.reduce功能
        assert_equal(np.minimum.reduce([1, 2j]), 2j)
        # 断言np.minimum.reduce([1, 2j])的结果为2j
        assert_equal(np.minimum.reduce([1+3j, 2j]), 2j)
        # 断言np.minimum.reduce([1+3j, 2j])的结果为2j

    def test_float_nans(self):
        # 定义test_float_nans方法，用于测试浮点数中NaN的情况
        nan = np.nan
        # 定义变量nan为NaN
        arg1 = np.array([0,   nan, nan])
        # 创建包含0和两个NaN的数组arg1
        arg2 = np.array([nan, 0,   nan])
        # 创建包含NaN、0和NaN的数组arg2
        out = np.array([nan, nan, nan])
        # 创建期望输出的数组out，全为NaN
        assert_equal(np.minimum(arg1, arg2), out)
        # 断言np.minimum(arg1, arg2)的结果与out相等

    def test_object_nans(self):
        # 定义test_object_nans方法，用于测试对象中NaN的情况
        # 多次检查以确保使用富比较而不是cmp，否则可能失败
        for i in range(1):
            # 循环一次（实际上只执行一次）
            x = np.array(float('nan'), object)
            # 创建一个包含NaN的对象数组x
            y = 1.0
            # 创建一个包含1.0的变量y
            z = np.array(float('nan'), object)
            # 创建一个包含NaN的对象数组z
            assert_(np.minimum(x, y) == 1.0)
            # 断言np.minimum(x, y)的结果为1.0
            assert_(np.minimum(z, y) == 1.0)
            # 断言np.minimum(z, y)的结果为1.0

    def test_complex_nans(self):
        # 定义test_complex_nans方法，用于测试复数中NaN的情况
        nan = np.nan
        # 定义变量nan为NaN
        for cnan in [complex(nan, 0), complex(0, nan), complex(nan, nan)]:
            # 遍历包含不同NaN组合的列表
            arg1 = np.array([0, cnan, cnan], dtype=complex)
            # 创建一个包含0和复数NaN的数组arg1
            arg2 = np.array([cnan, 0, cnan], dtype=complex)
            # 创建一个包含复数NaN、0和复数NaN的数组arg2
            out = np.array([nan, nan, nan], dtype=complex)
            # 创建期望输出的复数数组out，全为NaN
            assert_equal(np.minimum(arg1, arg2), out)
            # 断言np.minimum(arg1, arg2)的结果与out相等

    def test_object_array(self):
        # 定义test_object_array方法，用于测试对象数组的情况
        arg1 = np.arange(5, dtype=object)
        # 创建一个包含0到4的对象数组arg1
        arg2 = arg1 + 1
        # 创建一个arg1每个元素加1的数组arg2
        assert_equal(np.minimum(arg1, arg2), arg1)
        # 断言np.minimum(arg1, arg2)的结果与arg1相等

    def test_strided_array(self):
        # 定义test_strided_array方法，用于测试跨步数组的情况
        arr1 = np.array([-4.0, 1.0, 10.0,  0.0, np.nan, -np.nan, np.inf, -np.inf])
        # 创建包含各种数值的数组arr1
        arr2 = np.array([-2.0,-1.0, np.nan, 1.0, 0.0,    np.nan, 1.0,    -3.0])
        # 创建包含各种数值的数组arr2
        mintrue  = np.array([-4.0, -1.0, np.nan, 0.0, np.nan, np.nan, 1.0, -np.inf])
        # 创建期望最小值的数组mintrue
        out = np.ones(8)
        # 创建全为1的数组out，长度为8
        out_mintrue = np.array([-4.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan, 1.0])
        # 创建期望输出的数组out_mintrue
        assert_equal(np.minimum(arr1,arr2), mintrue)
        # 断言np.minimum(arr1,arr2)的结果与mintrue相等
        assert_equal(np.minimum(arr1[::2],arr2[::2]), mintrue[::2])
        # 断言np.minimum(arr1[::2],arr2[::2])的结果与mintrue[::2]相等
        assert_equal(np.minimum(arr1[:4:], arr2[::2]), np.array([-4.0, np.nan, 0.0, 0.0]))
        # 断言np.minimum(arr1[:4:], arr2[::2])的结果与预期数组相等
        assert_equal(np.minimum(arr1[::3], arr2[:3:]), np.array([-4.0, -1.0, np.nan]))
        # 断言np.minimum(arr1[::3], arr2[:3:])的结果与预期数组相等
        assert_equal(np.minimum(arr1[:6:2], arr2[::3], out=out[::3]), np.array([-4.0, 1.0, np.nan]))
        # 断言np.minimum(arr1[:6:2], arr2[::3], out=out[::3])的结果与预期数组相等
        assert_equal(out, out_mintrue)
        # 断言out与out_mintrue相等
    # 定义测试精度的方法
    def test_precision(self):
        # 定义浮点数类型列表
        dtypes = [np.float16, np.float32, np.float64, np.longdouble]

        # 遍历每种浮点数类型
        for dt in dtypes:
            # 获取当前浮点数类型的最小值和最大值
            dtmin = np.finfo(dt).min
            dtmax = np.finfo(dt).max
            # 创建一个浮点数值为 0.1 的对象
            d1 = dt(0.1)
            # 找到比 d1 大但最接近的浮点数
            d1_next = np.nextafter(d1, np.inf)

            # 定义测试用例列表
            test_cases = [
                # 测试用例 (v1, v2, expected)
                (dtmin, np.inf,     dtmin),  # 当 v1 是 dtmin, v2 是正无穷大时，预期结果是 dtmin
                (dtmax, np.inf,     dtmax),  # 当 v1 是 dtmax, v2 是正无穷大时，预期结果是 dtmax
                (d1,    d1_next,    d1),     # 当 v1 是 d1, v2 是 d1_next 时，预期结果是 d1
                (dtmin, np.nan,     np.nan), # 当 v1 是 dtmin, v2 是 NaN 时，预期结果是 NaN
            ]

            # 遍历每个测试用例
            for v1, v2, expected in test_cases:
                # 断言调用 np.minimum 函数，将 v1 和 v2 作为输入，预期输出是一个列表 [expected]
                assert_equal(np.minimum([v1], [v2]), [expected])
                # 断言调用 np.minimum.reduce 函数，将 v1 和 v2 作为输入，预期输出是 expected
                assert_equal(np.minimum.reduce([v1, v2]), expected)
# 定义一个名为 TestFmax 的测试类，继承自 _FilterInvalids 类
class TestFmax(_FilterInvalids):

    # 定义 test_reduce 方法，用于测试 np.fmax.reduce 函数
    def test_reduce(self):
        # 获取所有整数类型码
        dint = np.typecodes['AllInteger']
        # 创建一个长度为 11 的整数序列
        seq1 = np.arange(11)
        # 将 seq1 倒序排列得到 seq2
        seq2 = seq1[::-1]
        # func 设置为 np.fmax.reduce 函数
        func = np.fmax.reduce

        # 遍历所有整数类型码
        for dt in dint:
            # 将 seq1 和 seq2 转换为当前整数类型 dt
            tmp1 = seq1.astype(dt)
            tmp2 = seq2.astype(dt)
            # 断言 np.fmax.reduce(tmp1) 的结果为 10
            assert_equal(func(tmp1), 10)
            # 断言 np.fmax.reduce(tmp2) 的结果为 10
            assert_equal(func(tmp2), 10)

        # 获取所有浮点数类型码
        dflt = np.typecodes['AllFloat']

        # 再次遍历所有浮点数类型码
        for dt in dflt:
            # 将 seq1 和 seq2 转换为当前浮点数类型 dt
            tmp1 = seq1.astype(dt)
            tmp2 = seq2.astype(dt)
            # 断言 np.fmax.reduce(tmp1) 的结果为 10
            assert_equal(func(tmp1), 10)
            # 断言 np.fmax.reduce(tmp2) 的结果为 10
            assert_equal(func(tmp2), 10)
            # 将 tmp1 和 tmp2 中的偶数索引位置设置为 NaN
            tmp1[::2] = np.nan
            tmp2[::2] = np.nan
            # 断言 np.fmax.reduce(tmp1) 的结果为 9
            assert_equal(func(tmp1), 9)
            # 断言 np.fmax.reduce(tmp2) 的结果为 9
            assert_equal(func(tmp2), 9)

    # 定义 test_reduce_complex 方法，测试复数数组的 np.fmax.reduce 函数
    def test_reduce_complex(self):
        # 断言 np.fmax.reduce([1, 2j]) 的结果为 1
        assert_equal(np.fmax.reduce([1, 2j]), 1)
        # 断言 np.fmax.reduce([1+3j, 2j]) 的结果为 1+3j
        assert_equal(np.fmax.reduce([1+3j, 2j]), 1+3j)

    # 定义 test_float_nans 方法，测试带 NaN 的浮点数数组的 np.fmax 函数
    def test_float_nans(self):
        # 定义 NaN
        nan = np.nan
        # 创建包含 NaN 的两个浮点数数组
        arg1 = np.array([0,   nan, nan])
        arg2 = np.array([nan, 0,   nan])
        # 预期的输出数组
        out = np.array([0,   0,   nan])
        # 断言 np.fmax(arg1, arg2) 的结果等于 out
        assert_equal(np.fmax(arg1, arg2), out)

    # 定义 test_complex_nans 方法，测试带 NaN 的复数数组的 np.fmax 函数
    def test_complex_nans(self):
        # 定义 NaN
        nan = np.nan
        # 遍历包含不同 NaN 复数的列表
        for cnan in [complex(nan, 0), complex(0, nan), complex(nan, nan)]:
            # 创建包含 NaN 的复数数组 arg1 和 arg2
            arg1 = np.array([0, cnan, cnan], dtype=complex)
            arg2 = np.array([cnan, 0, cnan], dtype=complex)
            # 预期的输出数组
            out = np.array([0,    0, nan], dtype=complex)
            # 断言 np.fmax(arg1, arg2) 的结果等于 out
            assert_equal(np.fmax(arg1, arg2), out)

    # 定义 test_precision 方法，测试浮点数精度相关的 np.fmax 函数
    def test_precision(self):
        # 定义一组浮点数类型
        dtypes = [np.float16, np.float32, np.float64, np.longdouble]

        # 遍历每种浮点数类型
        for dt in dtypes:
            # 获取当前浮点数类型的最小值和最大值
            dtmin = np.finfo(dt).min
            dtmax = np.finfo(dt).max
            # 创建一个浮点数值和其下一个可表示值
            d1 = dt(0.1)
            d1_next = np.nextafter(d1, np.inf)

            # 测试用例列表
            test_cases = [
                # v1    v2          期望的结果
                (dtmin, -np.inf,    dtmin),
                (dtmax, -np.inf,    dtmax),
                (d1,    d1_next,    d1_next),
                (dtmax, np.nan,     dtmax),
            ]

            # 遍历每个测试用例
            for v1, v2, expected in test_cases:
                # 断言 np.fmax([v1], [v2]) 的结果等于 [expected]
                assert_equal(np.fmax([v1], [v2]), [expected])
                # 断言 np.fmax.reduce([v1, v2]) 的结果等于 expected
                assert_equal(np.fmax.reduce([v1, v2]), expected)


# 定义一个名为 TestFmin 的测试类，继承自 _FilterInvalids 类
class TestFmin(_FilterInvalids):

    # 定义 test_reduce 方法，用于测试 np.fmin.reduce 函数
    def test_reduce(self):
        # 获取所有整数类型码
        dint = np.typecodes['AllInteger']
        # 创建一个长度为 11 的整数序列
        seq1 = np.arange(11)
        # 将 seq1 倒序排列得到 seq2
        seq2 = seq1[::-1]
        # func 设置为 np.fmin.reduce 函数
        func = np.fmin.reduce

        # 遍历所有整数类型码
        for dt in dint:
            # 将 seq1 和 seq2 转换为当前整数类型 dt
            tmp1 = seq1.astype(dt)
            tmp2 = seq2.astype(dt)
            # 断言 np.fmin.reduce(tmp1) 的结果为 0
            assert_equal(func(tmp1), 0)
            # 断言 np.fmin.reduce(tmp2) 的结果为 0
            assert_equal(func(tmp2), 0)

        # 获取所有浮点数类型码
        dflt = np.typecodes['AllFloat']

        # 再次遍历所有浮点数类型码
        for dt in dflt:
            # 将 seq1 和 seq2 转换为当前浮点数类型 dt
            tmp1 = seq1.astype(dt)
            tmp2 = seq2.astype(dt)
            # 断言 np.fmin.reduce(tmp1) 的结果为 0
            assert_equal(func(tmp1), 0)
            # 断言 np.fmin.reduce(tmp2) 的结果为 0
            assert_equal(func(tmp2), 0)
            # 将 tmp1 和 tmp2 中的偶数索引位置设置为 NaN
            tmp1[::2] = np.nan
            tmp2[::2] = np.nan
            # 断言 np.fmin.reduce(tmp1) 的结果为 1
            assert_equal(func(tmp1), 1)
            # 断言 np.fmin.reduce(tmp2) 的结果为 1
            assert_equal(func(tmp2), 1)
    # 定义一个测试函数，用于测试 np.fmin.reduce 函数的行为
    def test_reduce_complex(self):
        # 断言：对复数列表进行 np.fmin.reduce 操作，返回最小的复数
        assert_equal(np.fmin.reduce([1, 2j]), 2j)
        # 断言：对复数列表进行 np.fmin.reduce 操作，返回最小的复数
        assert_equal(np.fmin.reduce([1+3j, 2j]), 2j)

    # 定义一个测试函数，用于测试处理浮点数包含 NaN 的情况
    def test_float_nans(self):
        # 定义 NaN
        nan = np.nan
        # 创建包含 NaN 的数组
        arg1 = np.array([0,   nan, nan])
        arg2 = np.array([nan, 0,   nan])
        out = np.array([0,   0,   nan])
        # 断言：对包含 NaN 的数组进行 np.fmin 操作，返回预期的输出
        assert_equal(np.fmin(arg1, arg2), out)

    # 定义一个测试函数，用于测试处理复数包含 NaN 的情况
    def test_complex_nans(self):
        # 定义 NaN
        nan = np.nan
        # 循环处理各种包含复数 NaN 的情况
        for cnan in [complex(nan, 0), complex(0, nan), complex(nan, nan)]:
            arg1 = np.array([0, cnan, cnan], dtype=complex)
            arg2 = np.array([cnan, 0, cnan], dtype=complex)
            out = np.array([0,    0, nan], dtype=complex)
            # 断言：对包含复数 NaN 的数组进行 np.fmin 操作，返回预期的输出
            assert_equal(np.fmin(arg1, arg2), out)

    # 定义一个测试函数，用于测试 np.fmin 函数在不同精度下的行为
    def test_precision(self):
        # 定义各种浮点数类型
        dtypes = [np.float16, np.float32, np.float64, np.longdouble]

        # 遍历不同的浮点数类型
        for dt in dtypes:
            # 获取当前浮点数类型的最小值和最大值
            dtmin = np.finfo(dt).min
            dtmax = np.finfo(dt).max
            d1 = dt(0.1)
            d1_next = np.nextafter(d1, np.inf)

            # 定义测试用例：(值1, 值2, 预期结果)
            test_cases = [
                (dtmin, np.inf,     dtmin),
                (dtmax, np.inf,     dtmax),
                (d1,    d1_next,    d1),
                (dtmin, np.nan,     dtmin),
            ]

            # 遍历测试用例
            for v1, v2, expected in test_cases:
                # 断言：对包含两个值的数组进行 np.fmin 操作，返回预期的输出
                assert_equal(np.fmin([v1], [v2]), [expected])
                # 断言：对列表中的值进行 np.fmin.reduce 操作，返回预期的输出
                assert_equal(np.fmin.reduce([v1, v2]), expected)
class TestBool:
    def test_exceptions(self):
        # 创建一个长度为1且元素为True的布尔型数组
        a = np.ones(1, dtype=np.bool)
        # 断言对np.negative应用于布尔数组会引发TypeError异常
        assert_raises(TypeError, np.negative, a)
        # 断言对np.positive应用于布尔数组会引发TypeError异常
        assert_raises(TypeError, np.positive, a)
        # 断言对np.subtract应用于布尔数组和自身会引发TypeError异常
        assert_raises(TypeError, np.subtract, a, a)

    def test_truth_table_logical(self):
        # 2、3和4被视为True值
        input1 = [0, 0, 3, 2]
        input2 = [0, 4, 0, 2]

        # 创建一个包含所有浮点和整数类型的typecodes，并加上布尔类型'?' 
        typecodes = (np.typecodes['AllFloat']
                     + np.typecodes['AllInteger']
                     + '?')     # boolean
        for dtype in map(np.dtype, typecodes):
            # 将input1和input2转换为指定dtype的ndarray
            arg1 = np.asarray(input1, dtype=dtype)
            arg2 = np.asarray(input2, dtype=dtype)

            # 对于逻辑或和最大值函数，验证其结果与预期输出相等
            out = [False, True, True, True]
            for func in (np.logical_or, np.maximum):
                assert_equal(func(arg1, arg2).astype(bool), out)
            # 对于逻辑与和最小值函数，验证其结果与预期输出相等
            out = [False, False, False, True]
            for func in (np.logical_and, np.minimum):
                assert_equal(func(arg1, arg2).astype(bool), out)
            # 对于逻辑异或和不等于函数，验证其结果与预期输出相等
            out = [False, True, True, False]
            for func in (np.logical_xor, np.not_equal):
                assert_equal(func(arg1, arg2).astype(bool), out)

    def test_truth_table_bitwise(self):
        # 设置两个布尔数组作为操作数
        arg1 = [False, False, True, True]
        arg2 = [False, True, False, True]

        # 预期的按位或运算结果
        out = [False, True, True, True]
        assert_equal(np.bitwise_or(arg1, arg2), out)

        # 预期的按位与运算结果
        out = [False, False, False, True]
        assert_equal(np.bitwise_and(arg1, arg2), out)

        # 预期的按位异或运算结果
        out = [False, True, True, False]
        assert_equal(np.bitwise_xor(arg1, arg2), out)

    def test_reduce(self):
        # 不包含True值的布尔数组
        none = np.array([0, 0, 0, 0], bool)
        # 包含部分True值的布尔数组
        some = np.array([1, 0, 1, 1], bool)
        # 全部是True值的布尔数组
        every = np.array([1, 1, 1, 1], bool)
        # 空的布尔数组
        empty = np.array([], bool)

        # 各种布尔数组的集合
        arrs = [none, some, every, empty]

        # 对每个布尔数组，验证逻辑与的reduce结果等于all函数的结果
        for arr in arrs:
            assert_equal(np.logical_and.reduce(arr), all(arr))

        # 对每个布尔数组，验证逻辑或的reduce结果等于any函数的结果
        for arr in arrs:
            assert_equal(np.logical_or.reduce(arr), any(arr))

        # 对每个布尔数组，验证逻辑异或的reduce结果等于对元素求和后模2等于1的结果
        for arr in arrs:
            assert_equal(np.logical_xor.reduce(arr), arr.sum() % 2 == 1)


class TestBitwiseUFuncs:

    _all_ints_bits = [
        np.dtype(c).itemsize * 8 for c in np.typecodes["AllInteger"]]
    # 定义按位操作函数要处理的数据类型列表
    bitwise_types = [
        np.dtype(c) for c in '?' + np.typecodes["AllInteger"] + 'O']
    # 每种按位操作函数的预期位数列表
    bitwise_bits = [
        2,  # boolean type
        *_all_ints_bits,  # All integers
        max(_all_ints_bits) + 1,  # Object_ type
    ]
    # 定义测试函数，用于验证不同数据类型的位运算函数的正确性
    def test_values(self):
        # 遍历每种位运算数据类型
        for dt in self.bitwise_types:
            # 创建一个包含单个零的数组，数据类型为当前迭代的数据类型
            zeros = np.array([0], dtype=dt)
            # 创建一个包含单个负一的数组，数据类型同样为当前迭代的数据类型
            ones = np.array([-1]).astype(dt)
            # 创建错误消息，用于测试失败时的输出
            msg = "dt = '%s'" % dt.char

            # 测试 np.bitwise_not 函数，验证其对零的取反是否等于负一
            assert_equal(np.bitwise_not(zeros), ones, err_msg=msg)
            # 测试 np.bitwise_not 函数，验证其对负一的取反是否等于零
            assert_equal(np.bitwise_not(ones), zeros, err_msg=msg)

            # 测试 np.bitwise_or 函数，验证其对两个零的按位或操作结果是否等于零
            assert_equal(np.bitwise_or(zeros, zeros), zeros, err_msg=msg)
            # 测试 np.bitwise_or 函数，验证其对零和负一的按位或操作结果是否等于负一
            assert_equal(np.bitwise_or(zeros, ones), ones, err_msg=msg)
            # 测试 np.bitwise_or 函数，验证其对负一和零的按位或操作结果是否等于负一
            assert_equal(np.bitwise_or(ones, zeros), ones, err_msg=msg)
            # 测试 np.bitwise_or 函数，验证其对两个负一的按位或操作结果是否等于负一
            assert_equal(np.bitwise_or(ones, ones), ones, err_msg=msg)

            # 测试 np.bitwise_xor 函数，验证其对两个零的按位异或操作结果是否等于零
            assert_equal(np.bitwise_xor(zeros, zeros), zeros, err_msg=msg)
            # 测试 np.bitwise_xor 函数，验证其对零和负一的按位异或操作结果是否等于负一
            assert_equal(np.bitwise_xor(zeros, ones), ones, err_msg=msg)
            # 测试 np.bitwise_xor 函数，验证其对负一和零的按位异或操作结果是否等于负一
            assert_equal(np.bitwise_xor(ones, zeros), ones, err_msg=msg)
            # 测试 np.bitwise_xor 函数，验证其对两个负一的按位异或操作结果是否等于零
            assert_equal(np.bitwise_xor(ones, ones), zeros, err_msg=msg)

            # 测试 np.bitwise_and 函数，验证其对两个零的按位与操作结果是否等于零
            assert_equal(np.bitwise_and(zeros, zeros), zeros, err_msg=msg)
            # 测试 np.bitwise_and 函数，验证其对零和负一的按位与操作结果是否等于零
            assert_equal(np.bitwise_and(zeros, ones), zeros, err_msg=msg)
            # 测试 np.bitwise_and 函数，验证其对负一和零的按位与操作结果是否等于零
            assert_equal(np.bitwise_and(ones, zeros), zeros, err_msg=msg)
            # 测试 np.bitwise_and 函数，验证其对两个负一的按位与操作结果是否等于负一
            assert_equal(np.bitwise_and(ones, ones), ones, err_msg=msg)

    # 定义测试函数，用于验证不同数据类型的位运算函数返回结果的数据类型是否正确
    def test_types(self):
        # 遍历每种位运算数据类型
        for dt in self.bitwise_types:
            # 创建一个包含单个零的数组，数据类型为当前迭代的数据类型
            zeros = np.array([0], dtype=dt)
            # 创建一个包含单个负一的数组，数据类型同样为当前迭代的数据类型
            ones = np.array([-1]).astype(dt)
            # 创建错误消息，用于测试失败时的输出
            msg = "dt = '%s'" % dt.char

            # 测试 np.bitwise_not 函数返回的结果的数据类型是否与当前数据类型相符
            assert_(np.bitwise_not(zeros).dtype == dt, msg)
            # 测试 np.bitwise_or 函数返回的结果的数据类型是否与当前数据类型相符
            assert_(np.bitwise_or(zeros, zeros).dtype == dt, msg)
            # 测试 np.bitwise_xor 函数返回的结果的数据类型是否与当前数据类型相符
            assert_(np.bitwise_xor(zeros, zeros).dtype == dt, msg)
            # 测试 np.bitwise_and 函数返回的结果的数据类型是否与当前数据类型相符
            assert_(np.bitwise_and(zeros, zeros).dtype == dt, msg)

    # 定义测试函数，用于验证位或、位异或和位与的身份元素是否正确
    def test_identity(self):
        # 验证 np.bitwise_or 函数的身份元素是否为零
        assert_(np.bitwise_or.identity == 0, 'bitwise_or')
        # 验证 np.bitwise_xor 函数的身份元素是否为零
        assert_(np.bitwise_xor.identity == 0, 'bitwise_xor')
        # 验证 np.bitwise_and 函数的身份元素是否为负一
        assert_(np.bitwise_and.identity == -1, 'bitwise_and')
    # 定义测试函数 test_reduction，用于测试位运算函数的归约操作
    def test_reduction(self):
        # 定义三个位运算函数：按位或、按位异或、按位与
        binary_funcs = (np.bitwise_or, np.bitwise_xor, np.bitwise_and)

        # 遍历位运算类型列表
        for dt in self.bitwise_types:
            # 创建一个包含单个零的数组，数据类型为 dt
            zeros = np.array([0], dtype=dt)
            # 创建一个包含单个 -1 的数组，转换为数据类型 dt
            ones = np.array([-1]).astype(dt)
            # 遍历位运算函数列表
            for f in binary_funcs:
                # 构造测试消息
                msg = "dt: '%s', f: '%s'" % (dt, f)
                # 断言归约函数应用在 zeros 上的结果等于 zeros，错误消息为 msg
                assert_equal(f.reduce(zeros), zeros, err_msg=msg)
                # 断言归约函数应用在 ones 上的结果等于 ones，错误消息为 msg
                assert_equal(f.reduce(ones), ones, err_msg=msg)

        # 测试空数组的归约操作，排除对象类型数组
        for dt in self.bitwise_types[:-1]:
            # 创建一个空数组，数据类型为 dt
            empty = np.array([], dtype=dt)
            # 遍历位运算函数列表
            for f in binary_funcs:
                # 构造测试消息
                msg = "dt: '%s', f: '%s'" % (dt, f)
                # 获取归约函数的单位元素，并转换为数据类型 dt
                tgt = np.array(f.identity).astype(dt)
                # 对空数组应用归约函数，比较结果和目标值 tgt，错误消息为 msg
                res = f.reduce(empty)
                assert_equal(res, tgt, err_msg=msg)
                # 断言结果的数据类型与目标值的数据类型相同，错误消息为 msg
                assert_(res.dtype == tgt.dtype, msg)

        # 对象类型数组中的空数组使用单位元素进行归约操作
        for f in binary_funcs:
            # 构造测试消息
            msg = "dt: '%s'" % (f,)
            # 创建一个空的对象类型数组
            empty = np.array([], dtype=object)
            # 获取归约函数的单位元素
            tgt = f.identity
            # 对空数组应用归约函数，比较结果和目标值 tgt，错误消息为 msg
            res = f.reduce(empty)
            assert_equal(res, tgt, err_msg=msg)

        # 非空对象类型数组不使用单位元素进行归约操作
        for f in binary_funcs:
            # 构造测试消息
            msg = "dt: '%s'" % (f,)
            # 创建一个包含 True 的对象类型数组
            btype = np.array([True], dtype=object)
            # 断言归约函数应用在 btype 上的结果的类型为布尔型，错误消息为 msg
            assert_(type(f.reduce(btype)) is bool, msg)

    # 使用 pytest 的参数化装饰器，定义测试函数 test_bitwise_count，用于测试位计数函数
    @pytest.mark.parametrize("input_dtype_obj, bitsize",
            zip(bitwise_types, bitwise_bits))
    def test_bitwise_count(self, input_dtype_obj, bitsize):
        # 获取输入数据类型的基本类型
        input_dtype = input_dtype_obj.type

        # 如果 Python 版本小于 3.10 并且数据类型为对象类型，则跳过测试
        if sys.version_info < (3, 10) and input_dtype == np.object_:
            pytest.skip("Required Python >=3.10")

        # 对指定范围内的数进行位计数测试
        for i in range(1, bitsize):
            # 计算 2 的 i 次方减 1
            num = 2**i - 1
            # 构造测试消息
            msg = f"bitwise_count for {num}"
            # 断言位计数函数应用在 num 上的结果等于 i，错误消息为 msg
            assert i == np.bitwise_count(input_dtype(num)), msg
            # 如果数据类型是有符号整数或对象类型，则进一步断言应用在 -num 上的结果等于 i，错误消息为 msg
            if np.issubdtype(input_dtype, np.signedinteger) or input_dtype == np.object_:
                assert i == np.bitwise_count(input_dtype(-num)), msg

        # 创建一个数组，包含从 1 到 bitsize-1 的整数，数据类型为 input_dtype
        a = np.array([2**i-1 for i in range(1, bitsize)], dtype=input_dtype)
        # 对数组应用位计数函数
        bitwise_count_a = np.bitwise_count(a)
        # 生成期望的结果数组，包含从 1 到 bitsize-1 的整数，数据类型为 input_dtype
        expected = np.arange(1, bitsize, dtype=input_dtype)

        # 构造测试消息
        msg = f"array bitwise_count for {input_dtype}"
        # 断言位计数函数应用在数组 a 上的结果与期望的结果数组相等，错误消息为 msg
        assert all(bitwise_count_a == expected), msg
class TestInt:
    # 测试逻辑非操作函数
    def test_logical_not(self):
        # 创建一个长度为10的numpy数组，元素类型为int16，值均为1
        x = np.ones(10, dtype=np.int16)
        # 创建一个长度为20的numpy数组，元素类型为bool，值均为True
        o = np.ones(10 * 2, dtype=bool)
        # 复制o数组，生成tgt数组
        tgt = o.copy()
        # 将tgt数组中偶数索引位置的元素改为False
        tgt[::2] = False
        # 从o数组中选取偶数索引位置的元素形成os数组
        os = o[::2]
        # 断言np.logical_not函数应用于x数组，结果应与False数组相等
        assert_array_equal(np.logical_not(x, out=os), False)
        # 断言o数组应与tgt数组相等
        assert_array_equal(o, tgt)


class TestFloatingPoint:
    # 测试浮点数支持情况
    def test_floating_point(self):
        # 断言浮点数支持常量的值为1
        assert_equal(ncu.FLOATING_POINT_SUPPORT, 1)


class TestDegrees:
    # 测试角度转换为度数
    def test_degrees(self):
        # 断言将π转换为度数后的值接近180.0
        assert_almost_equal(ncu.degrees(np.pi), 180.0)
        # 断言将-0.5π转换为度数后的值接近-90.0
        assert_almost_equal(ncu.degrees(-0.5*np.pi), -90.0)


class TestRadians:
    # 测试度数转换为弧度
    def test_radians(self):
        # 断言将180.0度转换为弧度后的值接近π
        assert_almost_equal(ncu.radians(180.0), np.pi)
        # 断言将-90.0度转换为弧度后的值接近-0.5π
        assert_almost_equal(ncu.radians(-90.0), -0.5*np.pi)


class TestHeavside:
    # 测试Heaviside函数
    def test_heaviside(self):
        # 创建包含不同类型浮点数的二维numpy数组x
        x = np.array([[-30.0, -0.1, 0.0, 0.2], [7.5, np.nan, np.inf, -np.inf]])
        # 预期的结果数组，使用0.5作为阈值
        expectedhalf = np.array([[0.0, 0.0, 0.5, 1.0], [1.0, np.nan, 1.0, 0.0]])
        # 复制expectedhalf数组生成expected1数组，并将其第一行第三列元素改为1
        expected1 = expectedhalf.copy()
        expected1[0, 2] = 1

        # 计算x数组在阈值0.5下的Heaviside函数结果，并与expectedhalf数组进行断言
        h = ncu.heaviside(x, 0.5)
        assert_equal(h, expectedhalf)

        # 计算x数组在阈值1.0下的Heaviside函数结果，并与expected1数组进行断言
        h = ncu.heaviside(x, 1.0)
        assert_equal(h, expected1)

        # 将x数组转换为float32类型后再次计算Heaviside函数结果，并与相应的float32类型的expectedhalf数组进行断言
        x = x.astype(np.float32)
        h = ncu.heaviside(x, np.float32(0.5))
        assert_equal(h, expectedhalf.astype(np.float32))

        h = ncu.heaviside(x, np.float32(1.0))
        assert_equal(h, expected1.astype(np.float32))


class TestSign:
    # 测试符号函数
    def test_sign(self):
        # 创建包含不同类型元素的numpy数组a
        a = np.array([np.inf, -np.inf, np.nan, 0.0, 3.0, -3.0])
        # 创建与a相同形状的零数组out
        out = np.zeros(a.shape)
        # 预期的结果数组tgt
        tgt = np.array([1., -1., np.nan, 0.0, 1.0, -1.0])

        # 使用np.errstate(invalid='ignore')上下文管理器处理无效值异常
        with np.errstate(invalid='ignore'):
            # 计算a数组的符号函数结果并与tgt数组进行断言
            res = ncu.sign(a)
            assert_equal(res, tgt)
            # 使用out数组计算a数组的符号函数结果并与tgt数组进行断言
            res = ncu.sign(a, out)
            assert_equal(res, tgt)
            # 断言out数组与tgt数组相等
            assert_equal(out, tgt)

    # 测试复数类型输入的符号函数
    def test_sign_complex(self):
        # 创建包含复数元素的numpy数组a
        a = np.array([
            np.inf, -np.inf, complex(0, np.inf), complex(0, -np.inf),
            complex(np.inf, np.inf), complex(np.inf, -np.inf),  # nan
            np.nan, complex(0, np.nan), complex(np.nan, np.nan),  # nan
            0.0,  # 0.
            3.0, -3.0, -2j, 3.0+4.0j, -8.0+6.0j
        ])
        # 创建与a相同形状和数据类型的零数组out
        out = np.zeros(a.shape, a.dtype)
        # 预期的结果数组tgt
        tgt = np.array([
            1., -1., 1j, -1j,
            ] + [complex(np.nan, np.nan)] * 5 + [
            0.0,
            1.0, -1.0, -1j, 0.6+0.8j, -0.8+0.6j])

        # 使用np.errstate(invalid='ignore')上下文管理器处理无效值异常
        with np.errstate(invalid='ignore'):
            # 计算a数组的符号函数结果并与tgt数组进行断言
            res = ncu.sign(a)
            assert_equal(res, tgt)
            # 使用out数组计算a数组的符号函数结果，并断言res与out是同一个对象
            res = ncu.sign(a, out)
            assert_(res is out)
            # 断言out数组与tgt数组相等
            assert_equal(res, tgt)

    # 测试对象类型输入的符号函数
    def test_sign_dtype_object(self):
        # 参考github问题＃6229
        # 创建包含浮点数元素的numpy数组foo
        foo = np.array([-.1, 0, .1])
        # 使用object类型计算foo数组的符号函数结果，并赋值给数组a
        a = np.sign(foo.astype(object))
        # 使用默认类型计算foo数组的符号函数结果，并赋值给数组b
        b = np.sign(foo)

        # 断言a数组与b数组相等
        assert_array_equal(a, b)
    def test_sign_dtype_nan_object(self):
        # 在参考 GitHub 问题 #6229 的情况下定义了一个测试函数 test_nan
        def test_nan():
            # 创建一个包含 NaN 值的 NumPy 数组
            foo = np.array([np.nan])
            # FIXME: a not used
            # 对数组进行类型转换为 object 后，计算其符号
            a = np.sign(foo.astype(object))

        # 断言捕获 TypeError 异常，确保 test_nan 函数抛出异常
        assert_raises(TypeError, test_nan)
class TestMinMax:
    def test_minmax_blocked(self):
        # SIMD tests on max/min, testing all alignments, although slow, it's crucial
        # Calculation for loop iterations: 2 * vz + 2 * (vs - 1) + 1 (unrolled once)
        for dt, sz in [(np.float32, 15), (np.float64, 7)]:
            # Generate alignment data for unary operations with specified dtype and size
            for out, inp, msg in _gen_alignment_data(dtype=dt, type='unary', max_size=sz):
                # Iterate over elements in input array
                for i in range(inp.size):
                    # Populate inp with array indices of dtype
                    inp[:] = np.arange(inp.size, dtype=dt)
                    # Set inp[i] to NaN
                    inp[i] = np.nan
                    # Custom error message function
                    emsg = lambda: '%r\n%s' % (inp, msg)
                    # Suppress warnings related to invalid values in reduce operations
                    with suppress_warnings() as sup:
                        sup.filter(RuntimeWarning, "invalid value encountered in reduce")
                        # Assert that maximum value in inp is NaN
                        assert_(np.isnan(inp.max()), msg=emsg)
                        # Assert that minimum value in inp is NaN
                        assert_(np.isnan(inp.min()), msg=emsg)

                    # Set inp[i] to 1e10 and assert its maximum value
                    inp[i] = 1e10
                    assert_equal(inp.max(), 1e10, err_msg=msg)
                    # Set inp[i] to -1e10 and assert its minimum value
                    inp[i] = -1e10
                    assert_equal(inp.min(), -1e10, err_msg=msg)

    def test_lower_align(self):
        # Check data that is not aligned to element size, e.g., doubles on i386
        d = np.zeros(23 * 8, dtype=np.int8)[4:-4].view(np.float64)
        # Assert that maximum value in d equals its first element
        assert_equal(d.max(), d[0])
        # Assert that minimum value in d equals its first element
        assert_equal(d.min(), d[0])

    def test_reduce_reorder(self):
        # GitHub issues 10370, 11029: Some compilers may reorder npy_getfloatstatus
        # causing invalid status to be set before intrinsic function calls.
        # Also ensure no warnings are emitted during operations.
        for n in (2, 4, 8, 16, 32):
            for dt in (np.float32, np.float16, np.complex64):
                # Create diagonal matrix with NaN values of specified dtype
                for r in np.diagflat(np.array([np.nan] * n, dtype=dt)):
                    # Assert that minimum value in r is NaN
                    assert_equal(np.min(r), np.nan)

    def test_minimize_no_warns(self):
        # Calculate minimum of NaN and 1 without generating warnings
        a = np.minimum(np.nan, 1)
        # Assert that result equals NaN
        assert_equal(a, np.nan)


class TestAbsoluteNegative:
    def test_abs_neg_blocked(self):
        # 进行 SIMD 测试，测试绝对值和取负操作在不同对齐方式下的表现
        for dt, sz in [(np.float32, 11), (np.float64, 5)]:
            # 生成指定数据类型和最大大小的数据集，用于一元操作的测试
            for out, inp, msg in _gen_alignment_data(dtype=dt, type='unary',
                                                     max_size=sz):
                # 计算目标结果，使用 ncu 模块计算每个输入元素的绝对值
                tgt = [ncu.absolute(i) for i in inp]
                # 使用 NumPy 计算每个输入元素的绝对值，将结果存储在输出数组中
                np.absolute(inp, out=out)
                # 断言输出数组与目标结果数组相等，用于验证测试通过
                assert_equal(out, tgt, err_msg=msg)
                # 断言输出数组中所有元素均为非负数
                assert_((out >= 0).all())

                # 计算目标结果，使用 ncu 模块计算每个输入元素的相反数
                tgt = [-1*(i) for i in inp]
                # 使用 NumPy 计算每个输入元素的相反数，将结果存储在输出数组中
                np.negative(inp, out=out)
                # 断言输出数组与目标结果数组相等，用于验证测试通过
                assert_equal(out, tgt, err_msg=msg)

                # 对特定值（NaN、-Inf、Inf）进行测试，验证对应情况下的数组操作
                for v in [np.nan, -np.inf, np.inf]:
                    for i in range(inp.size):
                        d = np.arange(inp.size, dtype=dt)
                        inp[:] = -d
                        inp[i] = v
                        d[i] = -v if v == -np.inf else v
                        # 使用 NumPy 计算每个输入数组的绝对值，与预期结果进行比较
                        assert_array_equal(np.abs(inp), d, err_msg=msg)
                        np.abs(inp, out=out)
                        assert_array_equal(out, d, err_msg=msg)

                        # 断言取反操作后数组元素与预期相符
                        assert_array_equal(-inp, -1*inp, err_msg=msg)
                        d = -1 * inp
                        np.negative(inp, out=out)
                        # 断言输出数组与预期结果相等，用于验证测试通过
                        assert_array_equal(out, d, err_msg=msg)

    def test_lower_align(self):
        # 检查未按元素大小对齐的数据情况
        # 即在 i386 上，double 类型数据按 4 字节对齐
        d = np.zeros(23 * 8, dtype=np.int8)[4:-4].view(np.float64)
        # 断言对数组取绝对值后结果与原数组相等
        assert_equal(np.abs(d), d)
        # 断言对数组取负后结果与负原数组相等
        assert_equal(np.negative(d), -d)
        # 在原数组上执行取负操作，并将结果存储在原数组中
        np.negative(d, out=d)
        # 在原数组上执行取负操作，将结果存储在原数组中
        np.negative(np.ones_like(d), out=d)
        # 在原数组上执行取绝对值操作，并将结果存储在原数组中
        np.abs(d, out=d)
        # 在原数组上执行取绝对值操作，将结果存储在原数组中
        np.abs(np.ones_like(d), out=d)

    @pytest.mark.parametrize("dtype", ['d', 'f', 'int32', 'int64'])
    @pytest.mark.parametrize("big", [True, False])
    # 定义测试方法，用于测试非连续数组的操作
    def test_noncontiguous(self, dtype, big):
        # 创建包含特定数据的 NumPy 数组，指定数据类型
        data = np.array([-1.0, 1.0, -0.0, 0.0, 2.2251e-308, -2.5, 2.5, -6,
                            6, -2.2251e-308, -8, 10], dtype=dtype)
        # 创建期望的结果数组，与 data 数组的内容相反
        expect = np.array([1.0, -1.0, 0.0, -0.0, -2.2251e-308, 2.5, -2.5, 6,
                            -6, 2.2251e-308, 8, -10], dtype=dtype)
        # 如果 big 为 True，则对数据和期望结果数组进行重复操作
        if big:
            data = np.repeat(data, 10)
            expect = np.repeat(expect, 10)
        # 创建与 data 相同形状的空数组 out
        out = np.ndarray(data.shape, dtype=dtype)
        # 从 data 中选取非连续元素，作为输入的非连续数组 ncontig_in
        ncontig_in = data[1::2]
        # 从 out 中选取非连续元素，作为输出的非连续数组 ncontig_out
        ncontig_out = out[1::2]
        # 将 ncontig_in 转换为 NumPy 数组，作为连续数组 contig_in
        contig_in = np.array(ncontig_in)
        # 断言：对 contig_in 中的元素取负，与期望的非连续数组元素 expect[1::2] 相等
        assert_array_equal(np.negative(contig_in), expect[1::2])
        # 断言：对 contig_in 中的元素取负，输出到 ncontig_out 中，与期望的非连续数组元素 expect[1::2] 相等
        assert_array_equal(np.negative(contig_in, out=ncontig_out),
                                expect[1::2])
        # 断言：对 ncontig_in 中的元素取负，与期望的非连续数组元素 expect[1::2] 相等
        assert_array_equal(np.negative(ncontig_in), expect[1::2])
        # 断言：对 ncontig_in 中的元素取负，输出到 ncontig_out 中，与期望的非连续数组元素 expect[1::2] 相等
        assert_array_equal(np.negative(ncontig_in, out=ncontig_out),
                                expect[1::2])
        # 断言：对 data 分割成两部分后，分别取负，与期望的分割后的结果 expect_split 相等
        data_split = np.array(np.array_split(data, 2))
        expect_split = np.array(np.array_split(expect, 2))
        assert_equal(np.negative(data_split), expect_split)
# 定义一个测试类 TestPositive，用于测试 np.positive 函数的行为
class TestPositive:
    
    # 定义一个测试方法 test_valid，用于测试 np.positive 对于有效数据类型的行为
    def test_valid(self):
        # 定义有效的数据类型列表
        valid_dtypes = [int, float, complex, object]
        # 遍历每种数据类型
        for dtype in valid_dtypes:
            # 创建一个长度为 5 的数组，指定数据类型为当前遍历的数据类型 dtype
            x = np.arange(5, dtype=dtype)
            # 对数组 x 应用 np.positive 函数
            result = np.positive(x)
            # 使用 assert_equal 断言 x 和 result 相等，如果不等则输出错误信息，并附带数据类型的字符串表示
            assert_equal(x, result, err_msg=str(dtype))
    
    # 定义一个测试方法 test_invalid，用于测试 np.positive 对于无效数据类型抛出 TypeError 的行为
    def test_invalid(self):
        # 使用 assert_raises 断言调用 np.positive(True) 会抛出 TypeError 异常
        with assert_raises(TypeError):
            np.positive(True)
        # 使用 assert_raises 断言调用 np.positive(np.datetime64('2000-01-01')) 会抛出 TypeError 异常
        with assert_raises(TypeError):
            np.positive(np.datetime64('2000-01-01'))
        # 使用 assert_raises 断言调用 np.positive(np.array(['foo'], dtype=str)) 会抛出 TypeError 异常
        with assert_raises(TypeError):
            np.positive(np.array(['foo'], dtype=str))
        # 使用 assert_raises 断言调用 np.positive(np.array(['bar'], dtype=object)) 会抛出 TypeError 异常
        with assert_raises(TypeError):
            np.positive(np.array(['bar'], dtype=object))


# 定义一个测试类 TestSpecialMethods，用于测试自定义类实现特殊方法的行为
class TestSpecialMethods:
    
    # 定义一个测试方法 test_wrap，用于测试自定义类实现 __array__ 和 __array_wrap__ 方法的包装行为
    def test_wrap(self):
        
        # 定义一个内嵌类 with_wrap，实现 __array__ 和 __array_wrap__ 方法
        class with_wrap:
            # 定义 __array__ 方法，返回一个包含一个零的 numpy 数组
            def __array__(self, dtype=None, copy=None):
                return np.zeros(1)

            # 定义 __array_wrap__ 方法，接受一个数组 arr、上下文 context 和返回标量 return_scalar，返回一个新的 with_wrap 对象
            def __array_wrap__(self, arr, context, return_scalar):
                r = with_wrap()
                r.arr = arr  # 将 arr 存储在新对象的 arr 属性中
                r.context = context  # 将 context 存储在新对象的 context 属性中
                return r
        
        # 创建一个 with_wrap 的实例 a
        a = with_wrap()
        # 调用 ncu.minimum 函数，传入 a 和 a 作为参数，并将结果存储在变量 x 中
        x = ncu.minimum(a, a)
        # 使用 assert_equal 断言 x.arr 等于一个包含一个零的 numpy 数组
        assert_equal(x.arr, np.zeros(1))
        # 解构 x.context，并将结果存储在 func、args、i 变量中
        func, args, i = x.context
        # 使用 assert_ 断言 func 是 ncu.minimum 函数
        assert_(func is ncu.minimum)
        # 使用 assert_equal 断言 args 的长度为 2
        assert_equal(len(args), 2)
        # 使用 assert_equal 断言 args[0] 等于 a
        assert_equal(args[0], a)
        # 使用 assert_equal 断言 args[1] 等于 a
        assert_equal(args[1], a)
        # 使用 assert_equal 断言 i 等于 0
        assert_equal(i, 0)
    def test_wrap_out(self):
        # out 参数的调用约定不应影响特殊方法的调用方式

        class StoreArrayPrepareWrap(np.ndarray):
            _wrap_args = None  # 初始化 _wrap_args 为 None
            _prepare_args = None  # 初始化 _prepare_args 为 None

            def __new__(cls):
                # 创建一个全零的 numpy 数组，并以当前类的视图返回
                return np.zeros(()).view(cls)

            def __array_wrap__(self, obj, context, return_scalar):
                # 在数组对象被返回之前，记录下 context 中的第二个参数到 _wrap_args
                self._wrap_args = context[1]
                return obj  # 返回对象本身

            @property
            def args(self):
                # 确保在任何其他 ufunc 被调用之前获取这些参数
                return self._wrap_args

            def __repr__(self):
                return "a"  # 用于简短的测试输出

        def do_test(f_call, f_expected):
            a = StoreArrayPrepareWrap()  # 创建 StoreArrayPrepareWrap 的实例

            f_call(a)  # 调用传入的函数 f_call

            w = a.args  # 获取存储的参数
            expected = f_expected(a)  # 获取预期的结果
            try:
                assert w == expected  # 断言参数与预期结果相等
            except AssertionError as e:
                # assert_equal 生成的错误信息非常无用
                raise AssertionError("\n".join([
                    "Bad arguments passed in ufunc call",
                    " expected:              {}".format(expected),
                    " __array_wrap__ got:    {}".format(w)
                ]))

        # 不涉及 out 参数的方法调用
        do_test(lambda a: np.add(a, 0),              lambda a: (a, 0))
        do_test(lambda a: np.add(a, 0, None),        lambda a: (a, 0))
        do_test(lambda a: np.add(a, 0, out=None),    lambda a: (a, 0))
        do_test(lambda a: np.add(a, 0, out=(None,)), lambda a: (a, 0))

        # 涉及 out 参数的方法调用
        do_test(lambda a: np.add(0, 0, a),           lambda a: (0, 0, a))
        do_test(lambda a: np.add(0, 0, out=a),       lambda a: (0, 0, a))
        do_test(lambda a: np.add(0, 0, out=(a,)),    lambda a: (0, 0, a))

        # 同时检查 where mask 处理：
        do_test(lambda a: np.add(a, 0, where=False), lambda a: (a, 0))
        do_test(lambda a: np.add(0, 0, a, where=False), lambda a: (0, 0, a))

    def test_wrap_with_iterable(self):
        # 测试修复 Bug #1026：

        class with_wrap(np.ndarray):
            __array_priority__ = 10  # 设置数组优先级为 10

            def __new__(cls):
                # 创建一个视图为 cls 类型的数组，初始为 np.asarray(1) 的拷贝
                return np.asarray(1).view(cls).copy()

            def __array_wrap__(self, arr, context, return_scalar):
                # 在数组被返回前，将其视图转换为与当前类相同的类型
                return arr.view(type(self))

        a = with_wrap()  # 创建 with_wrap 的实例
        x = ncu.multiply(a, (1, 2, 3))  # 使用 ncu 对 a 与 (1, 2, 3) 进行乘法操作
        assert_(isinstance(x, with_wrap))  # 断言 x 是 with_wrap 的实例
        assert_array_equal(x, np.array((1, 2, 3)))  # 断言 x 与 np.array((1, 2, 3)) 相等

    def test_priority_with_scalar(self):
        # 测试修复 Bug #826：

        class A(np.ndarray):
            __array_priority__ = 10  # 设置数组优先级为 10

            def __new__(cls):
                # 创建一个 float64 类型的数组，初始为 np.asarray(1.0, 'float64') 的拷贝
                return np.asarray(1.0, 'float64').view(cls).copy()

        a = A()  # 创建 A 的实例
        x = np.float64(1)*a  # 使用 np.float64(1) 乘以 a
        assert_(isinstance(x, A))  # 断言 x 是 A 的实例
        assert_array_equal(x, np.array(1))  # 断言 x 与 np.array(1) 相等
    def test_priority(self):
        # 定义测试方法 test_priority，用于测试优先级相关功能

        class A:
            def __array__(self, dtype=None, copy=None):
                # 定义 __array__ 方法，返回一个包含一个元素的零数组
                return np.zeros(1)

            def __array_wrap__(self, arr, context, return_scalar):
                # 定义 __array_wrap__ 方法，返回一个新的类型为 self 的实例，包含传入的数组和上下文信息
                r = type(self)()
                r.arr = arr
                r.context = context
                return r

        class B(A):
            __array_priority__ = 20.  # 设置数组优先级为 20.0

        class C(A):
            __array_priority__ = 40.  # 设置数组优先级为 40.0

        x = np.zeros(1)
        a = A()
        b = B()
        c = C()
        f = ncu.minimum
        assert_(type(f(x, x)) is np.ndarray)  # 断言调用 ncu.minimum 返回的是 np.ndarray 类型
        assert_(type(f(x, a)) is A)  # 断言调用 ncu.minimum 返回的是 A 类型
        assert_(type(f(x, b)) is B)  # 断言调用 ncu.minimum 返回的是 B 类型
        assert_(type(f(x, c)) is C)  # 断言调用 ncu.minimum 返回的是 C 类型
        assert_(type(f(a, x)) is A)  # 断言调用 ncu.minimum 返回的是 A 类型
        assert_(type(f(b, x)) is B)  # 断言调用 ncu.minimum 返回的是 B 类型
        assert_(type(f(c, x)) is C)  # 断言调用 ncu.minimum 返回的是 C 类型

        assert_(type(f(a, a)) is A)  # 断言调用 ncu.minimum 返回的是 A 类型
        assert_(type(f(a, b)) is B)  # 断言调用 ncu.minimum 返回的是 B 类型
        assert_(type(f(b, a)) is B)  # 断言调用 ncu.minimum 返回的是 B 类型
        assert_(type(f(b, b)) is B)  # 断言调用 ncu.minimum 返回的是 B 类型
        assert_(type(f(b, c)) is C)  # 断言调用 ncu.minimum 返回的是 C 类型
        assert_(type(f(c, b)) is C)  # 断言调用 ncu.minimum 返回的是 C 类型
        assert_(type(f(c, c)) is C)  # 断言调用 ncu.minimum 返回的是 C 类型

        assert_(type(ncu.exp(a)) is A)  # 断言调用 ncu.exp 返回的是 A 类型
        assert_(type(ncu.exp(b)) is B)  # 断言调用 ncu.exp 返回的是 B 类型
        assert_(type(ncu.exp(c)) is C)  # 断言调用 ncu.exp 返回的是 C 类型

    def test_failing_wrap(self):
        # 定义测试方法 test_failing_wrap，用于测试异常情况下的包装器

        class A:
            def __array__(self, dtype=None, copy=None):
                # 定义 __array__ 方法，返回一个包含两个元素的零数组
                return np.zeros(2)

            def __array_wrap__(self, arr, context, return_scalar):
                # 定义 __array_wrap__ 方法，抛出 RuntimeError 异常
                raise RuntimeError

        a = A()
        assert_raises(RuntimeError, ncu.maximum, a, a)  # 断言调用 ncu.maximum 抛出 RuntimeError 异常
        assert_raises(RuntimeError, ncu.maximum.reduce, a)  # 断言调用 ncu.maximum.reduce 抛出 RuntimeError 异常

    def test_failing_out_wrap(self):
        # 定义测试方法 test_failing_out_wrap，用于测试异常情况下的输出包装器

        singleton = np.array([1.0])

        class Ok(np.ndarray):
            def __array_wrap__(self, obj, context, return_scalar):
                # 定义 __array_wrap__ 方法，返回 singleton
                return singleton

        class Bad(np.ndarray):
            def __array_wrap__(self, obj, context, return_scalar):
                # 定义 __array_wrap__ 方法，抛出 RuntimeError 异常
                raise RuntimeError

        ok = np.empty(1).view(Ok)
        bad = np.empty(1).view(Bad)
        # 断言调用 ncu.frexp 函数时，若 bad 引发异常，则导致 "ok" 双重释放（段错误）
        for i in range(10):
            assert_raises(RuntimeError, ncu.frexp, 1, ok, bad)

    def test_none_wrap(self):
        # 定义测试方法 test_none_wrap，测试解决问题 #8507。先前，这会导致段错误。

        class A:
            def __array__(self, dtype=None, copy=None):
                # 定义 __array__ 方法，返回一个包含一个元素的零数组
                return np.zeros(1)

            def __array_wrap__(self, arr, context=None, return_scalar=False):
                # 定义 __array_wrap__ 方法，返回 None
                return None

        a = A()
        assert_equal(ncu.maximum(a, a), None)  # 断言调用 ncu.maximum 返回 None

    def test_default_prepare(self):
        # 定义测试方法 test_default_prepare，测试默认准备功能

        class with_wrap:
            __array_priority__ = 10  # 设置数组优先级为 10

            def __array__(self, dtype=None, copy=None):
                # 定义 __array__ 方法，返回一个包含一个元素的零数组
                return np.zeros(1)

            def __array_wrap__(self, arr, context, return_scalar):
                # 定义 __array_wrap__ 方法，返回传入的数组 arr
                return arr

        a = with_wrap()
        x = ncu.minimum(a, a)
        assert_equal(x, np.zeros(1))  # 断言 x 等于包含一个元素的零数组
        assert_equal(type(x), np.ndarray)  # 断言 x 的类型是 np.ndarray
    def test_ufunc_override(self):
        # 定义一个测试函数，用于验证重载 __array_ufunc__ 方法的效果。

        # 定义一个类 A，该类重载了 __array_ufunc__ 方法
        class A:
            def __array_ufunc__(self, func, method, *inputs, **kwargs):
                return self, func, method, inputs, kwargs
        
        # 定义一个类 MyNDArray，继承自 numpy.ndarray，设置了较高的 __array_priority__ 值为 100
        class MyNDArray(np.ndarray):
            __array_priority__ = 100

        # 创建一个类 A 的实例
        a = A()
        # 创建一个 numpy 数组 b，并将其视图转换为 MyNDArray 类型
        b = np.array([1]).view(MyNDArray)

        # 使用 np.multiply 函数进行数组元素相乘操作，分别赋值给 res0 和 res1
        res0 = np.multiply(a, b)
        res1 = np.multiply(b, b, out=a)

        # 断言语句，验证 res0[0] 和 res1[0] 均为对象 a
        assert_equal(res0[0], a)
        assert_equal(res1[0], a)
        # 断言语句，验证 res0[1] 和 res1[1] 均为 np.multiply 函数
        assert_equal(res0[1], np.multiply)
        assert_equal(res1[1], np.multiply)
        # 断言语句，验证 res0[2] 和 res1[2] 均为 '__call__'
        assert_equal(res0[2], '__call__')
        assert_equal(res1[2], '__call__')
        # 断言语句，验证 res0[3] 和 res1[3] 均为元组 (a, b) 和 (b, b)
        assert_equal(res0[3], (a, b))
        assert_equal(res1[3], (b, b))
        # 断言语句，验证 res0[4] 和 res1[4] 均为空字典和 {'out': (a,)} 字典
        assert_equal(res0[4], {})
        assert_equal(res1[4], {'out': (a,)})
    # 定义一个测试函数 test_ufunc_override_out
    def test_ufunc_override_out(self):

        # 定义类 A，实现 __array_ufunc__ 方法，返回关键字参数 kwargs
        class A:
            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                return kwargs

        # 定义类 B，实现 __array_ufunc__ 方法，返回关键字参数 kwargs
        class B:
            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                return kwargs

        # 创建 A 类的实例 a 和 B 类的实例 b
        a = A()
        b = B()

        # 使用 np.multiply 进行乘法运算，测试不同形式的输出参数 'out_arg'
        res0 = np.multiply(a, b, 'out_arg')
        res1 = np.multiply(a, b, out='out_arg')
        res2 = np.multiply(2, b, 'out_arg')
        res3 = np.multiply(3, b, out='out_arg')
        res4 = np.multiply(a, 4, 'out_arg')
        res5 = np.multiply(a, 5, out='out_arg')

        # 断言检查每个乘法运算结果的 'out' 键是否为 'out_arg'
        assert_equal(res0['out'][0], 'out_arg')
        assert_equal(res1['out'][0], 'out_arg')
        assert_equal(res2['out'][0], 'out_arg')
        assert_equal(res3['out'][0], 'out_arg')
        assert_equal(res4['out'][0], 'out_arg')
        assert_equal(res5['out'][0], 'out_arg')

        # 使用 np.modf 和 np.frexp 函数，测试多输出模式 'out0' 和 'out1'
        res6 = np.modf(a, 'out0', 'out1')
        res7 = np.frexp(a, 'out0', 'out1')

        # 断言检查每个函数结果的 'out' 键对应的值是否符合预期 'out0' 和 'out1'
        assert_equal(res6['out'][0], 'out0')
        assert_equal(res6['out'][1], 'out1')
        assert_equal(res7['out'][0], 'out0')
        assert_equal(res7['out'][1], 'out1')

        # 检查默认输出参数为 None 时的返回值
        assert_(np.sin(a, None) == {})
        assert_(np.sin(a, out=None) == {})
        assert_(np.sin(a, out=(None,)) == {})
        assert_(np.modf(a, None) == {})
        assert_(np.modf(a, None, None) == {})
        assert_(np.modf(a, out=(None, None)) == {})

        # 使用 assert_raises 检查 TypeError 异常，确保 out 参数为元组形式
        with assert_raises(TypeError):
            np.modf(a, out=None)

        # 使用 assert_raises 检查 TypeError 和 ValueError 异常，确保参数组合正确性
        assert_raises(TypeError, np.multiply, a, b, 'one', out='two')
        assert_raises(TypeError, np.multiply, a, b, 'one', 'two')
        assert_raises(ValueError, np.multiply, a, b, out=('one', 'two'))
        assert_raises(TypeError, np.multiply, a, out=())
        assert_raises(TypeError, np.modf, a, 'one', out=('two', 'three'))
        assert_raises(TypeError, np.modf, a, 'one', 'two', 'three')
        assert_raises(ValueError, np.modf, a, out=('one', 'two', 'three'))
        assert_raises(ValueError, np.modf, a, out=('one',))
    def test_ufunc_override_where(self):
        # 定义一个继承自 np.ndarray 的类 OverriddenArrayOld，用于重载特定的数组操作
        class OverriddenArrayOld(np.ndarray):

            # 解包方法，用于将输入对象解包成适当的数组形式
            def _unwrap(self, objs):
                cls = type(self)
                result = []
                for obj in objs:
                    if isinstance(obj, cls):
                        obj = np.array(obj)
                    elif type(obj) != np.ndarray:
                        return NotImplemented
                    result.append(obj)
                return result

            # 重载 __array_ufunc__ 方法，处理数组操作符重载
            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                # 解包输入参数
                inputs = self._unwrap(inputs)
                if inputs is NotImplemented:
                    return NotImplemented

                kwargs = kwargs.copy()
                # 如果输出指定了 out 参数，也进行解包处理
                if "out" in kwargs:
                    kwargs["out"] = self._unwrap(kwargs["out"])
                    if kwargs["out"] is NotImplemented:
                        return NotImplemented

                # 调用父类的 __array_ufunc__ 方法处理数组操作
                r = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
                if r is not NotImplemented:
                    r = r.view(type(self))

                return r

        # 定义一个继承自 OverriddenArrayOld 的类 OverriddenArrayNew，进一步重载 __array_ufunc__ 方法
        class OverriddenArrayNew(OverriddenArrayOld):
            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                kwargs = kwargs.copy()
                # 如果 kwargs 中包含 where 参数，解包并处理
                if "where" in kwargs:
                    kwargs["where"] = self._unwrap((kwargs["where"], ))
                    if kwargs["where"] is NotImplemented:
                        return NotImplemented
                    else:
                        kwargs["where"] = kwargs["where"][0]

                # 调用父类的 __array_ufunc__ 方法处理数组操作
                r = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
                if r is not NotImplemented:
                    r = r.view(type(self))

                return r

        # 定义一个被重载的数组操作函数 ufunc
        ufunc = np.negative

        # 创建一个普通的 numpy 数组
        array = np.array([1, 2, 3])
        # 创建一个用于 where 参数的布尔数组
        where = np.array([True, False, True])
        # 计算预期的操作结果
        expected = ufunc(array, where=where)

        # 使用 pytest 来测试预期的 TypeError 异常
        with pytest.raises(TypeError):
            ufunc(array, where=where.view(OverriddenArrayOld))

        # 测试第一种重载情况，期望返回值为 OverriddenArrayNew 类型，并且结果符合预期
        result_1 = ufunc(
            array,
            where=where.view(OverriddenArrayNew)
        )
        assert isinstance(result_1, OverriddenArrayNew)
        assert np.all(np.array(result_1) == expected, where=where)

        # 测试第二种重载情况，输入为 OverriddenArrayNew 类型的数组，并且结果符合预期
        result_2 = ufunc(
            array.view(OverriddenArrayNew),
            where=where.view(OverriddenArrayNew)
        )
        assert isinstance(result_2, OverriddenArrayNew)
        assert np.all(np.array(result_2) == expected, where=where)

    def test_ufunc_override_exception(self):
        # 定义一个简单的类 A，实现 __array_ufunc__ 方法抛出 ValueError 异常
        class A:
            def __array_ufunc__(self, *a, **kwargs):
                raise ValueError("oops")

        # 创建类 A 的实例
        a = A()
        # 使用 assert_raises 测试 np.negative 在指定的输出 out=a 时抛出 ValueError 异常
        assert_raises(ValueError, np.negative, 1, out=a)
        # 使用 assert_raises 测试 np.negative 在输入为 a 时抛出 ValueError 异常
        assert_raises(ValueError, np.negative, a)
        # 使用 assert_raises 测试 np.divide 在除数为 a 时抛出 ValueError 异常
        assert_raises(ValueError, np.divide, 1., a)
    def test_ufunc_override_not_implemented(self):
        # 定义一个测试方法，验证在自定义类中未实现 __array_ufunc__ 方法时的行为

        class A:
            def __array_ufunc__(self, *args, **kwargs):
                return NotImplemented
            # 定义一个特殊类 A，其 __array_ufunc__ 方法返回 NotImplemented

        # 第一个测试用例：验证对 np.negative 函数的行为
        msg = ("operand type(s) all returned NotImplemented from "
               "__array_ufunc__(<ufunc 'negative'>, '__call__', <*>): 'A'")
        with assert_raises_regex(TypeError, fnmatch.translate(msg)):
            np.negative(A())
            # 断言调用 np.negative(A()) 会引发 TypeError，错误消息符合预期格式

        # 第二个测试用例：验证对 np.add 函数的行为
        msg = ("operand type(s) all returned NotImplemented from "
               "__array_ufunc__(<ufunc 'add'>, '__call__', <*>, <object *>, "
               "out=(1,)): 'A', 'object', 'int'")
        with assert_raises_regex(TypeError, fnmatch.translate(msg)):
            np.add(A(), object(), out=1)
            # 断言调用 np.add(A(), object(), out=1) 会引发 TypeError，错误消息符合预期格式

    def test_ufunc_override_disabled(self):
        # 定义一个测试方法，验证当 __array_ufunc__ 被设置为 None 时的行为

        class OptOut:
            __array_ufunc__ = None
            # 定义一个类 OptOut，其 __array_ufunc__ 属性为 None

        opt_out = OptOut()

        # 第一个测试用例：验证 ufuncs 对 OptOut 类型的行为
        msg = "operand 'OptOut' does not support ufuncs"
        with assert_raises_regex(TypeError, msg):
            np.add(opt_out, 1)
            # 断言调用 np.add(opt_out, 1) 会引发 TypeError，错误消息符合预期格式
        with assert_raises_regex(TypeError, msg):
            np.add(1, opt_out)
            # 断言调用 np.add(1, opt_out) 会引发 TypeError，错误消息符合预期格式
        with assert_raises_regex(TypeError, msg):
            np.negative(opt_out)
            # 断言调用 np.negative(opt_out) 会引发 TypeError，错误消息符合预期格式

        # 第二个测试用例：验证即使其他参数具有异常的 __array_ufunc__ 实现时，OptOut 类型的行为仍然有效
        class GreedyArray:
            def __array_ufunc__(self, *args, **kwargs):
                return self
            # 定义一个类 GreedyArray，其 __array_ufunc__ 方法返回 self

        greedy = GreedyArray()
        assert_(np.negative(greedy) is greedy)
        # 断言调用 np.negative(greedy) 返回的是 greedy 本身
        with assert_raises_regex(TypeError, msg):
            np.add(greedy, opt_out)
            # 断言调用 np.add(greedy, opt_out) 会引发 TypeError，错误消息符合预期格式
        with assert_raises_regex(TypeError, msg):
            np.add(greedy, 1, out=opt_out)
            # 断言调用 np.add(greedy, 1, out=opt_out) 会引发 TypeError，错误消息符合预期格式

    def test_gufunc_override(self):
        # 定义一个测试方法，验证 gufunc 的行为

        # gufunc 实际上是 ufunc 实例，但是它们遵循不同的路径，因此需要检查 __array_ufunc__ 是否正确覆盖
        class A:
            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                return self, ufunc, method, inputs, kwargs
            # 定义一个类 A，其 __array_ufunc__ 方法返回自身及其他参数

        inner1d = ncu_tests.inner1d
        a = A()

        # 第一个测试用例：验证 inner1d(a, a) 的行为
        res = inner1d(a, a)
        assert_equal(res[0], a)
        assert_equal(res[1], inner1d)
        assert_equal(res[2], '__call__')
        assert_equal(res[3], (a, a))
        assert_equal(res[4], {})
        # 断言调用 inner1d(a, a) 返回的结果符合预期

        # 第二个测试用例：验证 inner1d(1, 1, out=a) 的行为
        res = inner1d(1, 1, out=a)
        assert_equal(res[0], a)
        assert_equal(res[1], inner1d)
        assert_equal(res[2], '__call__')
        assert_equal(res[3], (1, 1))
        assert_equal(res[4], {'out': (a,)})
        # 断言调用 inner1d(1, 1, out=a) 返回的结果符合预期

        # 错误的参数数量是一个错误
        assert_raises(TypeError, inner1d, a, out='two')
        assert_raises(TypeError, inner1d, a, a, 'one', out='two')
        assert_raises(TypeError, inner1d, a, a, 'one', 'two')
        assert_raises(ValueError, inner1d, a, a, out=('one', 'two'))
        assert_raises(ValueError, inner1d, a, a, out=())
        # 断言调用 inner1d 时传入错误参数会引发预期的异常
    # 定义一个测试方法，用于直接调用数组的 __array_ufunc__ 方法进行功能回归测试
    def test_array_ufunc_direct_call(self):
        # 这是对 gh-24023 的回归测试，主要是防止出现段错误
        a = np.array(1)
        # 使用 pytest 的断言检查是否会抛出 TypeError 异常
        with pytest.raises(TypeError):
            a.__array_ufunc__()

        # 当不传入关键字参数时，kwargs 在 C 级别可能为 NULL
        with pytest.raises(TypeError):
            a.__array_ufunc__(1, 2)

        # 同样的情况适用于有效调用的情况：
        # 调用 a 对象的 __array_ufunc__ 方法，使用 np.add 函数，调用 "__call__" 方法，对 a 和 a 进行操作
        res = a.__array_ufunc__(np.add, "__call__", a, a)
        # 使用断言检查结果是否与预期的 a + a 相等
        assert_array_equal(res, a + a)
class TestChoose:
    # 测试选择函数 np.choose 的混合用例
    def test_mixed(self):
        # 创建布尔选择数组 c 和备选数组 a
        c = np.array([True, True])
        a = np.array([True, True])
        # 断言选择函数的结果与期望结果相等
        assert_equal(np.choose(c, (a, 1)), np.array([1, 1]))


class TestRationalFunctions:
    # 有关有理数函数的测试类

    # 测试最小公倍数函数 np.lcm 的基本用例
    def test_lcm(self):
        # 使用指定数据类型进行最小公倍数函数的内部测试
        self._test_lcm_inner(np.int16)
        self._test_lcm_inner(np.uint16)

    # 测试最小公倍数函数 np.lcm 在对象数组上的应用
    def test_lcm_object(self):
        # 使用对象数组进行最小公倍数函数的内部测试
        self._test_lcm_inner(np.object_)

    # 测试最大公约数函数 np.gcd 的基本用例
    def test_gcd(self):
        # 使用指定数据类型进行最大公约数函数的内部测试
        self._test_gcd_inner(np.int16)
        self._test_lcm_inner(np.uint16)

    # 测试最大公约数函数 np.gcd 在对象数组上的应用
    def test_gcd_object(self):
        # 使用对象数组进行最大公约数函数的内部测试
        self._test_gcd_inner(np.object_)

    # 内部函数，用于测试最小公倍数函数 np.lcm 的不同情况
    def _test_lcm_inner(self, dtype):
        # 基本用例：使用指定数据类型的数组进行测试
        a = np.array([12, 120], dtype=dtype)
        b = np.array([20, 200], dtype=dtype)
        assert_equal(np.lcm(a, b), [60, 600])

        if not issubclass(dtype, np.unsignedinteger):
            # 负数被忽略的情况
            a = np.array([12, -12,  12, -12], dtype=dtype)
            b = np.array([20,  20, -20, -20], dtype=dtype)
            assert_equal(np.lcm(a, b), [60]*4)

        # 缩减用例
        a = np.array([3, 12, 20], dtype=dtype)
        assert_equal(np.lcm.reduce([3, 12, 20]), 60)

        # 广播以及包含 0 的测试
        a = np.arange(6).astype(dtype)
        b = 20
        assert_equal(np.lcm(a, b), [0, 20, 20, 60, 20, 20])

    # 内部函数，用于测试最大公约数函数 np.gcd 的不同情况
    def _test_gcd_inner(self, dtype):
        # 基本用例：使用指定数据类型的数组进行测试
        a = np.array([12, 120], dtype=dtype)
        b = np.array([20, 200], dtype=dtype)
        assert_equal(np.gcd(a, b), [4, 40])

        if not issubclass(dtype, np.unsignedinteger):
            # 负数被忽略的情况
            a = np.array([12, -12,  12, -12], dtype=dtype)
            b = np.array([20,  20, -20, -20], dtype=dtype)
            assert_equal(np.gcd(a, b), [4]*4)

        # 缩减用例
        a = np.array([15, 25, 35], dtype=dtype)
        assert_equal(np.gcd.reduce(a), 5)

        # 广播以及包含 0 的测试
        a = np.arange(6).astype(dtype)
        b = 20
        assert_equal(np.gcd(a, b), [20,  1,  2,  1,  4,  5])

    # 测试最小公倍数函数 np.lcm 在溢出情况下的行为
    def test_lcm_overflow(self):
        # 验证在 a*b 溢出时，最小公倍数函数不会溢出
        big = np.int32(np.iinfo(np.int32).max // 11)
        a = 2*big
        b = 5*big
        assert_equal(np.lcm(a, b), 10*big)

    # 测试最大公约数函数 np.gcd 在溢出情况下的行为
    def test_gcd_overflow(self):
        for dtype in (np.int32, np.int64):
            # 验证在取 abs(x) 时不会溢出
            # 对于最小公倍数函数而言，结果无法表示
            a = dtype(np.iinfo(dtype).min)  # 负的二次幂
            q = -(a // 4)
            assert_equal(np.gcd(a,  q*3), q)
            assert_equal(np.gcd(a, -q*3), q)

    # 测试十进制数的最大公约数和最小公倍数函数
    def test_decimal(self):
        from decimal import Decimal
        # 创建十进制数组 a 和 b
        a = np.array([1,  1, -1, -1]) * Decimal('0.20')
        b = np.array([1, -1,  1, -1]) * Decimal('0.12')

        # 断言最大公约数函数的结果与期望结果相等
        assert_equal(np.gcd(a, b), 4*[Decimal('0.04')])
        # 断言最小公倍数函数的结果与期望结果相等
        assert_equal(np.lcm(a, b), 4*[Decimal('0.60')])
    def test_float(self):
        # 浮点数在计算中由于舍入误差而不精确定义
        assert_raises(TypeError, np.gcd, 0.3, 0.4)
        assert_raises(TypeError, np.lcm, 0.3, 0.4)

    def test_huge_integers(self):
        # 将整数转换为数组后，由于显式对象数据类型的存在，行为略有不同：
        assert_equal(np.array(2**200), 2**200)
        # 特殊的提升规则应确保即使对两个 Python 整数也有效（尽管速度较慢）。
        # （我们在比较时这样做，因为结果始终是布尔值，我们还特别处理数组与Python整数的比较）
        np.equal(2**200, 2**200)

        # 但是，当影响结果的数据类型时，我们无法这样做：
        with pytest.raises(OverflowError):
            np.gcd(2**100, 3**100)

        # 明确要求使用 `object` 类型是可以的：
        assert np.gcd(2**100, 3**100, dtype=object) == 1

        # 到目前为止，以下操作是有效的，因为它们使用了数组（将是对象数组）
        a = np.array(2**100 * 3**5)
        b = np.array([2**100 * 5**7, 2**50 * 3**10])
        assert_equal(np.gcd(a, b), [2**100,               2**50 * 3**5])
        assert_equal(np.lcm(a, b), [2**100 * 3**5 * 5**7, 2**100 * 3**10])
# 定义一个测试类 TestRoundingFunctions，用于测试四舍五入相关函数
class TestRoundingFunctions:

    # 定义测试对象直接实现魔术方法的方法
    def test_object_direct(self):
        """ test direct implementation of these magic methods """
        # 定义一个内部类 C，实现 __floor__、__ceil__ 和 __trunc__ 方法
        class C:
            def __floor__(self):
                return 1
            def __ceil__(self):
                return 2
            def __trunc__(self):
                return 3

        # 创建一个包含两个 C 类对象的 NumPy 数组 arr
        arr = np.array([C(), C()])
        # 断言 np.floor(arr) 返回值与预期 [1, 1] 相等
        assert_equal(np.floor(arr), [1, 1])
        # 断言 np.ceil(arr) 返回值与预期 [2, 2] 相等
        assert_equal(np.ceil(arr),  [2, 2])
        # 断言 np.trunc(arr) 返回值与预期 [3, 3] 相等
        assert_equal(np.trunc(arr), [3, 3])

    # 定义测试对象间接通过 __float__ 方法实现的方法
    def test_object_indirect(self):
        """ test implementations via __float__ """
        # 定义一个内部类 C，实现 __float__ 方法返回 -2.5
        class C:
            def __float__(self):
                return -2.5

        # 创建一个包含两个 C 类对象的 NumPy 数组 arr
        arr = np.array([C(), C()])
        # 断言 np.floor(arr) 返回值与预期 [-3, -3] 相等
        assert_equal(np.floor(arr), [-3, -3])
        # 断言 np.ceil(arr) 返回值与预期 [-2, -2] 相等
        assert_equal(np.ceil(arr),  [-2, -2])
        # 使用 pytest.raises 检查 np.trunc(arr) 抛出 TypeError 异常，与 math.trunc 保持一致
        with pytest.raises(TypeError):
            np.trunc(arr)  # consistent with math.trunc

    # 定义测试 Fraction 对象的方法
    def test_fraction(self):
        # 创建 Fraction 对象 f，值为 -4/3
        f = Fraction(-4, 3)
        # 断言 np.floor(f) 返回值与预期 -2 相等
        assert_equal(np.floor(f), -2)
        # 断言 np.ceil(f) 返回值与预期 -1 相等
        assert_equal(np.ceil(f), -1)
        # 断言 np.trunc(f) 返回值与预期 -1 相等
        assert_equal(np.trunc(f), -1)


# 定义测试复数函数的测试类 TestComplexFunctions
class TestComplexFunctions:
    # 定义包含多个函数的列表 funcs
    funcs = [np.arcsin,  np.arccos,  np.arctan, np.arcsinh, np.arccosh,
             np.arctanh, np.sin,     np.cos,    np.tan,     np.exp,
             np.exp2,    np.log,     np.sqrt,   np.log10,   np.log2,
             np.log1p]

    # 定义测试方法 test_it，测试 funcs 中所有函数的实现
    def test_it(self):
        for f in self.funcs:
            # 根据不同的函数 f 设置变量 x 的值
            if f is np.arccosh:
                x = 1.5
            else:
                x = .5
            # 分别计算函数 f 在实数 x 和复数 complex(x) 上的结果
            fr = f(x)
            fz = f(complex(x))
            # 断言 fz 的实部与 fr 相等，错误消息包含函数名
            assert_almost_equal(fz.real, fr, err_msg='real part %s' % f)
            # 断言 fz 的虚部为 0.，错误消息包含函数名
            assert_almost_equal(fz.imag, 0., err_msg='imag part %s' % f)

    # 定义测试方法 test_precisions_consistent，测试 funcs 中所有函数在不同精度下的一致性
    @pytest.mark.xfail(IS_WASM, reason="doesn't work")
    def test_precisions_consistent(self):
        # 定义复数 z
        z = 1 + 1j
        for f in self.funcs:
            # 分别计算 f 在 np.csingle(z)、np.cdouble(z) 和 np.clongdouble(z) 上的结果
            fcf = f(np.csingle(z))
            fcd = f(np.cdouble(z))
            fcl = f(np.clongdouble(z))
            # 断言 fcf 和 fcd 的结果在小数点后 6 位精度下相等，错误消息包含函数名
            assert_almost_equal(fcf, fcd, decimal=6, err_msg='fch-fcd %s' % f)
            # 断言 fcl 和 fcd 的结果在小数点后 15 位精度下相等，错误消息包含函数名
            assert_almost_equal(fcl, fcd, decimal=15, err_msg='fch-fcl %s' % f)

    # 定义测试方法 test_precisions_consistent，测试 funcs 中所有函数在不同精度下的一致性
    @pytest.mark.xfail(IS_WASM, reason="doesn't work")
    def test_branch_cuts(self):
        # 检查分支切割和它们的连续性
        _check_branch_cut(np.log,   -0.5, 1j, 1, -1, True)
        _check_branch_cut(np.log2,  -0.5, 1j, 1, -1, True)
        _check_branch_cut(np.log10, -0.5, 1j, 1, -1, True)
        _check_branch_cut(np.log1p, -1.5, 1j, 1, -1, True)
        _check_branch_cut(np.sqrt,  -0.5, 1j, 1, -1, True)

        _check_branch_cut(np.arcsin, [ -2, 2],   [1j, 1j], 1, -1, True)
        _check_branch_cut(np.arccos, [ -2, 2],   [1j, 1j], 1, -1, True)
        _check_branch_cut(np.arctan, [0-2j, 2j],  [1,  1], -1, 1, True)

        _check_branch_cut(np.arcsinh, [0-2j,  2j], [1,   1], -1, 1, True)
        _check_branch_cut(np.arccosh, [ -1, 0.5], [1j,  1j], 1, -1, True)
        _check_branch_cut(np.arctanh, [ -2,   2], [1j, 1j], 1, -1, True)

        # 检查错误的分支切割：断言象限之间的连续性
        _check_branch_cut(np.arcsin, [0-2j, 2j], [ 1,  1], 1, 1)
        _check_branch_cut(np.arccos, [0-2j, 2j], [ 1,  1], 1, 1)
        _check_branch_cut(np.arctan, [ -2,  2], [1j, 1j], 1, 1)

        _check_branch_cut(np.arcsinh, [ -2,  2, 0], [1j, 1j, 1], 1, 1)
        _check_branch_cut(np.arccosh, [0-2j, 2j, 2], [1,  1,  1j], 1, 1)
        _check_branch_cut(np.arctanh, [0-2j, 2j, 0], [1,  1,  1j], 1, 1)

    @pytest.mark.xfail(IS_WASM, reason="doesn't work")
    def test_branch_cuts_complex64(self):
        # 检查复数类型 complex64 的分支切割和连续性
        _check_branch_cut(np.log,   -0.5, 1j, 1, -1, True, np.complex64)
        _check_branch_cut(np.log2,  -0.5, 1j, 1, -1, True, np.complex64)
        _check_branch_cut(np.log10, -0.5, 1j, 1, -1, True, np.complex64)
        _check_branch_cut(np.log1p, -1.5, 1j, 1, -1, True, np.complex64)
        _check_branch_cut(np.sqrt,  -0.5, 1j, 1, -1, True, np.complex64)

        _check_branch_cut(np.arcsin, [ -2, 2],   [1j, 1j], 1, -1, True, np.complex64)
        _check_branch_cut(np.arccos, [ -2, 2],   [1j, 1j], 1, -1, True, np.complex64)
        _check_branch_cut(np.arctan, [0-2j, 2j],  [1,  1], -1, 1, True, np.complex64)

        _check_branch_cut(np.arcsinh, [0-2j,  2j], [1,   1], -1, 1, True, np.complex64)
        _check_branch_cut(np.arccosh, [ -1, 0.5], [1j,  1j], 1, -1, True, np.complex64)
        _check_branch_cut(np.arctanh, [ -2,   2], [1j, 1j], 1, -1, True, np.complex64)

        # 检查错误的分支切割：断言复数类型 complex64 在象限之间的连续性
        _check_branch_cut(np.arcsin, [0-2j, 2j], [ 1,  1], 1, 1, False, np.complex64)
        _check_branch_cut(np.arccos, [0-2j, 2j], [ 1,  1], 1, 1, False, np.complex64)
        _check_branch_cut(np.arctan, [ -2,  2], [1j, 1j], 1, 1, False, np.complex64)

        _check_branch_cut(np.arcsinh, [ -2,  2, 0], [1j, 1j, 1], 1, 1, False, np.complex64)
        _check_branch_cut(np.arccosh, [0-2j, 2j, 2], [1,  1,  1j], 1, 1, False, np.complex64)
        _check_branch_cut(np.arctanh, [0-2j, 2j, 0], [1,  1,  1j], 1, 1, False, np.complex64)
    # 测试自定义函数与cmath库中对应函数的精度
    def test_against_cmath(self):
        # 导入cmath库
        import cmath

        # 定义复数坐标点
        points = [-1-1j, -1+1j, +1-1j, +1+1j]
        # 函数名映射，用于处理cmath库中函数名与自定义函数可能不同的情况
        name_map = {'arcsin': 'asin', 'arccos': 'acos', 'arctan': 'atan',
                    'arcsinh': 'asinh', 'arccosh': 'acosh', 'arctanh': 'atanh'}
        # 计算绝对误差的阈值，使用numpy库中复数类型的机器精度
        atol = 4*np.finfo(complex).eps
        
        # 遍历所有待测试的自定义函数
        for func in self.funcs:
            # 获取函数名，并提取最后一部分（去除命名空间）
            fname = func.__name__.split('.')[-1]
            # 根据映射表获取对应的cmath函数名，如果没有映射，则使用原函数名
            cname = name_map.get(fname, fname)
            try:
                # 获取cmath库中对应的函数对象
                cfunc = getattr(cmath, cname)
            except AttributeError:
                # 如果cmath库中不存在对应函数，则跳过当前函数的测试
                continue
            
            # 对每个复数坐标点执行测试
            for p in points:
                # 调用自定义函数并转换为复数类型
                a = complex(func(np.complex128(p)))
                # 调用cmath库中对应函数
                b = cfunc(p)
                # 断言自定义函数与cmath库计算结果的绝对误差小于预设阈值
                assert_(
                    abs(a - b) < atol,
                    "%s %s: %s; cmath: %s" % (fname, p, a, b)
                )

    @pytest.mark.xfail(
        # manylinux2014 使用 glibc2.17
        _glibc_older_than("2.18"),
        reason="旧版glibc版本不精确（也许使用SIMD能通过？）"
    )
    @pytest.mark.xfail(IS_WASM, reason="不适用于WASM")
    @pytest.mark.parametrize('dtype', [
        np.complex64, np.complex128, np.clongdouble
    ])
    @np.errstate(all="ignore")
    # 测试特殊情况下的类型提升
    def test_promotion_corner_cases(self):
        # 对每个自定义函数执行测试
        for func in self.funcs:
            # 断言自定义函数对np.float16(1)的输出类型为np.float16
            assert func(np.float16(1)).dtype == np.float16
            # 断言整数转低精度浮点数的提升是一个值得商榷的选择
            assert func(np.uint8(1)).dtype == np.float16
            assert func(np.int16(1)).dtype == np.float32
class TestAttributes:
    # 测试属性的类
    def test_attributes(self):
        # 获取全局变量 ncu.add 的引用
        add = ncu.add
        # 断言 add 函数的名称为 'add'
        assert_equal(add.__name__, 'add')
        # 断言 add 函数支持的类型数至少为 18
        assert_(add.ntypes >= 18)  # 如果添加类型不会失败
        # 断言 'ii->i' 在 add 函数的类型列表中
        assert_('ii->i' in add.types)
        # 断言 add 函数的输入参数个数为 2
        assert_equal(add.nin, 2)
        # 断言 add 函数的输出参数个数为 1
        assert_equal(add.nout, 1)
        # 断言 add 函数的身份元素为 0
        assert_equal(add.identity, 0)

    def test_doc(self):
        # 测试文档字符串的方法
        # 不需要检查可能会变化的大量关键字参数
        assert_(ncu.add.__doc__.startswith(
            "add(x1, x2, /, out=None, *, where=True"))
        assert_(ncu.frexp.__doc__.startswith(
            "frexp(x[, out1, out2], / [, out=(None, None)], *, where=True"))


class TestSubclass:
    # 测试子类的类
    def test_subclass_op(self):
        # 测试子类操作的方法

        class simple(np.ndarray):
            # 简单的子类，继承自 np.ndarray
            def __new__(subtype, shape):
                # 根据指定形状创建新的实例
                self = np.ndarray.__new__(subtype, shape, dtype=object)
                # 将实例填充为 0
                self.fill(0)
                return self

        # 创建一个 simple 类的实例 a，形状为 (3, 4)
        a = simple((3, 4))
        # 断言 a 加上自身等于 a
        assert_equal(a+a, a)


class TestFrompyfunc:
    # 测试从 Python 函数创建 ufunc 的类
    def test_identity(self):
        # 测试标识性参数的方法
        def mul(a, b):
            return a * b

        # 使用 identity=1 创建 mul 函数的 ufunc
        mul_ufunc = np.frompyfunc(mul, nin=2, nout=1, identity=1)
        # 断言对数组 [2, 3, 4] 使用 mul_ufunc.reduce 的结果为 24
        assert_equal(mul_ufunc.reduce([2, 3, 4]), 24)
        # 断言对 np.ones((2, 2)) 使用 mul_ufunc.reduce 沿着轴 (0, 1) 的结果为 1
        assert_equal(mul_ufunc.reduce(np.ones((2, 2)), axis=(0, 1)), 1)
        # 断言对空数组使用 mul_ufunc.reduce 会引发 ValueError
        assert_equal(mul_ufunc.reduce([]), 1)

        # 使用 identity=None 创建 mul 函数的 ufunc（可重新排序）
        mul_ufunc = np.frompyfunc(mul, nin=2, nout=1, identity=None)
        # 断言对数组 [2, 3, 4] 使用 mul_ufunc.reduce 的结果为 24
        assert_equal(mul_ufunc.reduce([2, 3, 4]), 24)
        # 断言对 np.ones((2, 2)) 使用 mul_ufunc.reduce 沿着轴 (0, 1) 的结果为 1
        assert_equal(mul_ufunc.reduce(np.ones((2, 2)), axis=(0, 1)), 1)
        # 断言对空数组使用 mul_ufunc.reduce 会引发 ValueError
        assert_raises(ValueError, lambda: mul_ufunc.reduce([]))

        # 使用无 identity 创建 mul 函数的 ufunc（不可重新排序）
        mul_ufunc = np.frompyfunc(mul, nin=2, nout=1)
        # 断言对数组 [2, 3, 4] 使用 mul_ufunc.reduce 的结果为 24
        assert_equal(mul_ufunc.reduce([2, 3, 4]), 24)
        # 断言对 np.ones((2, 2)) 使用 mul_ufunc.reduce 沿着轴 (0, 1) 会引发 ValueError
        assert_raises(ValueError, lambda: mul_ufunc.reduce(np.ones((2, 2)), axis=(0, 1)))
        # 断言对空数组使用 mul_ufunc.reduce 会引发 ValueError
        assert_raises(ValueError, lambda: mul_ufunc.reduce([]))


def _check_branch_cut(f, x0, dx, re_sign=1, im_sign=-1, sig_zero_ok=False,
                      dtype=complex):
    """
    Check for a branch cut in a function.

    Assert that `x0` lies on a branch cut of function `f` and `f` is
    continuous from the direction `dx`.

    Parameters
    ----------
    f : func
        Function to check
    x0 : array-like
        Point on branch cut
    dx : array-like
        Direction to check continuity in
    re_sign, im_sign : {1, -1}
        Change of sign of the real or imaginary part expected
    sig_zero_ok : bool
        Whether to check if the branch cut respects signed zero (if applicable)
    dtype : dtype
        Dtype to check (should be complex)

    """
    # 将 x0 和 dx 转换为至少一维数组，并指定 dtype
    x0 = np.atleast_1d(x0).astype(dtype)
    dx = np.atleast_1d(dx).astype(dtype)

    # 如果 dtype 是复数类型
    if np.dtype(dtype).char == 'F':
        # 设置 scale 和 atol 的值
        scale = np.finfo(dtype).eps * 1e2
        atol = np.float32(1e-2)
    else:
        scale = np.finfo(dtype).eps * 1e3
        atol = 1e-4

    # 计算 f(x0) 的值并赋给 y0
    y0 = f(x0)
    # 计算正向偏移点的函数值
    yp = f(x0 + dx * scale * np.absolute(x0) / np.absolute(dx))
    # 计算反向偏移点的函数值
    ym = f(x0 - dx * scale * np.absolute(x0) / np.absolute(dx))

    # 断言：验证正向偏移的实部与原始点的实部之差小于给定的误差容限
    assert_(np.all(np.absolute(y0.real - yp.real) < atol), (y0, yp))
    # 断言：验证正向偏移的虚部与原始点的虚部之差小于给定的误差容限
    assert_(np.all(np.absolute(y0.imag - yp.imag) < atol), (y0, yp))
    # 断言：验证反向偏移的实部与原始点的实部之差小于给定的误差容限
    assert_(np.all(np.absolute(y0.real - ym.real * re_sign) < atol), (y0, ym))
    # 断言：验证反向偏移的虚部与原始点的虚部之差小于给定的误差容限
    assert_(np.all(np.absolute(y0.imag - ym.imag * im_sign) < atol), (y0, ym))

    if sig_zero_ok:
        # 检查有符号零值是否可以作为位移
        # 对于实部为零但偏移不为零的情况
        jr = (x0.real == 0) & (dx.real != 0)
        if np.any(jr):
            # 取出实部为零的元素，并将其设为零
            x = x0[jr]
            x.real = ncu.NZERO
            # 计算函数在零实部元素上的值
            ym = f(x)
            # 断言：验证在零实部元素上，反向偏移的实部与原始点的实部之差小于给定的误差容限
            assert_(np.all(np.absolute(y0[jr].real - ym.real * re_sign) < atol), (y0[jr], ym))
            # 断言：验证在零实部元素上，反向偏移的虚部与原始点的虚部之差小于给定的误差容限
            assert_(np.all(np.absolute(y0[jr].imag - ym.imag * im_sign) < atol), (y0[jr], ym))

        # 对于虚部为零但偏移不为零的情况
        ji = (x0.imag == 0) & (dx.imag != 0)
        if np.any(ji):
            # 取出虚部为零的元素，并将其设为零
            x = x0[ji]
            x.imag = ncu.NZERO
            # 计算函数在零虚部元素上的值
            ym = f(x)
            # 断言：验证在零虚部元素上，反向偏移的实部与原始点的实部之差小于给定的误差容限
            assert_(np.all(np.absolute(y0[ji].real - ym.real * re_sign) < atol), (y0[ji], ym))
            # 断言：验证在零虚部元素上，反向偏移的虚部与原始点的虚部之差小于给定的误差容限
            assert_(np.all(np.absolute(y0[ji].imag - ym.imag * im_sign) < atol), (y0[ji], ym))
# 定义一个测试函数，用于验证 np.copysign() 的功能
def test_copysign():
    # 断言 np.copysign(1, -1) 的结果应该是 -1
    assert_(np.copysign(1, -1) == -1)
    # 在忽略除法错误的情况下，验证 1 / np.copysign(0, -1) 结果小于 0
    with np.errstate(divide="ignore"):
        assert_(1 / np.copysign(0, -1) < 0)
        # 验证 1 / np.copysign(0, 1) 结果大于 0
        assert_(1 / np.copysign(0, 1) > 0)
    # 验证 np.copysign(np.nan, -1) 结果为负数
    assert_(np.signbit(np.copysign(np.nan, -1)))
    # 验证 np.copysign(np.nan, 1) 结果为非负数
    assert_(not np.signbit(np.copysign(np.nan, 1)))

# 定义一个内部测试函数，用于测试 np.nextafter() 的功能
def _test_nextafter(t):
    one = t(1)
    two = t(2)
    zero = t(0)
    eps = np.finfo(t).eps
    # 验证 np.nextafter(one, two) - one 结果等于 eps
    assert_(np.nextafter(one, two) - one == eps)
    # 验证 np.nextafter(one, zero) - one 结果小于 0
    assert_(np.nextafter(one, zero) - one < 0)
    # 验证 np.nextafter(np.nan, one) 和 np.nextafter(one, np.nan) 结果为 NaN
    assert_(np.isnan(np.nextafter(np.nan, one)))
    assert_(np.isnan(np.nextafter(one, np.nan)))
    # 验证 np.nextafter(one, one) 结果等于 one
    assert_(np.nextafter(one, one) == one)

# 定义测试函数，调用 _test_nextafter() 并返回结果
def test_nextafter():
    return _test_nextafter(np.float64)

# 定义测试函数，调用 _test_nextafter() 并返回结果
def test_nextafterf():
    return _test_nextafter(np.float32)

# 根据特定条件跳过或标记失败的测试函数，验证 np.longdouble 的 np.nextafter() 功能
@pytest.mark.skipif(np.finfo(np.double) == np.finfo(np.longdouble),
                    reason="long double is same as double")
@pytest.mark.xfail(condition=platform.machine().startswith("ppc64"),
                    reason="IBM double double")
def test_nextafterl():
    return _test_nextafter(np.longdouble)

# 定义测试函数，验证 np.nextafter() 在边界值情况下的行为
def test_nextafter_0():
    for t, direction in itertools.product(np._core.sctypes['float'], (1, -1)):
        # 对于双倍精度浮点数的极小值，需要特别处理
        with suppress_warnings() as sup:
            sup.filter(UserWarning)
            if not np.isnan(np.finfo(t).tiny):
                tiny = np.finfo(t).tiny
                # 验证 direction * np.nextafter(t(0), t(direction)) 结果在 0 和 tiny 之间
                assert_(0. < direction * np.nextafter(t(0), t(direction)) < tiny)
        # 验证 np.nextafter(t(0), t(direction)) / t(2.1) 结果等于 direction * 0.0
        assert_equal(np.nextafter(t(0), t(direction)) / t(2.1), direction * 0.0)

# 定义内部测试函数，用于验证 np.spacing() 的功能
def _test_spacing(t):
    one = t(1)
    eps = np.finfo(t).eps
    nan = t(np.nan)
    inf = t(np.inf)
    with np.errstate(invalid='ignore'):
        # 验证 np.spacing(one) 结果等于 eps
        assert_equal(np.spacing(one), eps)
        # 验证 np.spacing(nan) 和 np.spacing(inf) 结果为 NaN
        assert_(np.isnan(np.spacing(nan)))
        assert_(np.isnan(np.spacing(inf)))
        assert_(np.isnan(np.spacing(-inf)))
        # 验证 np.spacing(t(1e30)) 不等于 0
        assert_(np.spacing(t(1e30)) != 0)

# 定义测试函数，调用 _test_spacing() 并返回结果
def test_spacing():
    return _test_spacing(np.float64)

# 定义测试函数，调用 _test_spacing() 并返回结果
def test_spacingf():
    return _test_spacing(np.float32)

# 根据特定条件跳过或标记失败的测试函数，验证 np.longdouble 的 np.spacing() 功能
@pytest.mark.skipif(np.finfo(np.double) == np.finfo(np.longdouble),
                    reason="long double is same as double")
@pytest.mark.xfail(condition=platform.machine().startswith("ppc64"),
                    reason="IBM double double")
def test_spacingl():
    return _test_spacing(np.longdouble)

# 定义测试函数，验证 np.spacing() 在不同条件下的行为
def test_spacing_gfortran():
    # 参考自 gfortran 编译的 Fortran 文件，在不同情况下验证 np.spacing() 结果
    # 这段注释仅作为参考，不需要在代码中体现
    pass
    # 创建一个字典 `ref`，键为 np.float64 和 np.float32，对应的值是列表，分别包含特定精度下的 np.spacing(x) 的参考值
    ref = {np.float64: [1.69406589450860068E-021,
                        2.22044604925031308E-016,
                        1.13686837721616030E-013,
                        1.81898940354585648E-012],
           np.float32: [9.09494702E-13,
                        1.19209290E-07,
                        6.10351563E-05,
                        9.76562500E-04]}

    # 针对每个数据类型 `dt` 和相应的小数精度 `dec_`，执行以下操作
    for dt, dec_ in zip([np.float32, np.float64], (10, 20)):
        # 创建一个 numpy 数组 `x`，包含指定数据类型 `dt` 的四个不同数值
        x = np.array([1e-5, 1, 1000, 10500], dtype=dt)
        # 使用 assert_array_almost_equal 函数验证 np.spacing(x) 的结果几乎等于 `ref[dt]` 中的值，使用 `decimal=dec_` 的精度
        assert_array_almost_equal(np.spacing(x), ref[dt], decimal=dec_)
def test_nextafter_vs_spacing():
    # XXX: spacing does not handle long double yet
    # 循环测试不同类型（np.float32, np.float64）和不同浮点数值（1, 1e-5, 1000）
    for t in [np.float32, np.float64]:
        for _f in [1, 1e-5, 1000]:
            f = t(_f)
            f1 = t(_f + 1)
            # 断言 np.nextafter(f, f1) - f 等于 np.spacing(f)
            assert_(np.nextafter(f, f1) - f == np.spacing(f))

def test_pos_nan():
    """Check np.nan is a positive nan."""
    # 断言 np.nan 的符号位为 0，即它是一个正的 NaN
    assert_(np.signbit(np.nan) == 0)

def test_reduceat():
    """Test bug in reduceat when structured arrays are not copied."""
    # 定义结构化数据类型 db，并创建空数组 a
    db = np.dtype([('name', 'S11'), ('time', np.int64), ('value', np.float32)])
    a = np.empty([100], dtype=db)
    a['name'] = 'Simple'
    a['time'] = 10
    a['value'] = 100
    indx = [0, 7, 15, 25]

    h2 = []
    val1 = indx[0]
    # 对于 indx 中的索引，使用 np.add.reduce 计算 a['value'] 的和，存入 h2
    for val2 in indx[1:]:
        h2.append(np.add.reduce(a['value'][val1:val2]))
        val1 = val2
    h2.append(np.add.reduce(a['value'][val1:]))
    h2 = np.array(h2)

    # 断言 np.add.reduceat(a['value'], indx) 的结果与 h2 数组几乎相等
    h1 = np.add.reduceat(a['value'], indx)
    assert_array_almost_equal(h1, h2)

    # 设置缓冲区大小为 32，再次计算 reduceat 结果，断言几乎相等
    np.setbufsize(32)
    h1 = np.add.reduceat(a['value'], indx)
    np.setbufsize(ncu.UFUNC_BUFSIZE_DEFAULT)
    assert_array_almost_equal(h1, h2)

def test_reduceat_empty():
    """Reduceat should work with empty arrays"""
    indices = np.array([], 'i4')
    x = np.array([], 'f8')
    # 对空数组 x 使用 reduceat，断言结果的数据类型与 x 相同且形状为 (0,)
    result = np.add.reduceat(x, indices)
    assert_equal(result.dtype, x.dtype)
    assert_equal(result.shape, (0,))
    # 另一种情况，x 是一个全为 1 的 5x2 数组，对其使用 reduceat，断言结果形状为 (0, 2)
    x = np.ones((5, 2))
    result = np.add.reduceat(x, [], axis=0)
    assert_equal(result.dtype, x.dtype)
    assert_equal(result.shape, (0, 2))
    result = np.add.reduceat(x, [], axis=1)
    assert_equal(result.dtype, x.dtype)
    assert_equal(result.shape, (5, 0))

def test_complex_nan_comparisons():
    nans = [complex(np.nan, 0), complex(0, np.nan), complex(np.nan, np.nan)]
    fins = [complex(1, 0), complex(-1, 0), complex(0, 1), complex(0, -1),
            complex(1, 1), complex(-1, -1), complex(0, 0)]

    with np.errstate(invalid='ignore'):
        for x in nans + fins:
            x = np.array([x])
            for y in nans + fins:
                y = np.array([y])

                # 如果 x 和 y 都是有限的数，则跳过
                if np.isfinite(x) and np.isfinite(y):
                    continue

                # 断言 x < y, x > y, x <= y, x >= y, x == y 都为 False
                assert_equal(x < y, False, err_msg="%r < %r" % (x, y))
                assert_equal(x > y, False, err_msg="%r > %r" % (x, y))
                assert_equal(x <= y, False, err_msg="%r <= %r" % (x, y))
                assert_equal(x >= y, False, err_msg="%r >= %r" % (x, y))
                assert_equal(x == y, False, err_msg="%r == %r" % (x, y))

def test_rint_big_int():
    # np.rint bug for large integer values on Windows 32-bit and MKL
    # https://github.com/numpy/numpy/issues/6685
    val = 4607998452777363968
    # 这个值在浮点数中可以精确表示
    assert_equal(val, int(float(val)))
    # np.rint 不应改变这个值
    assert_equal(val, np.rint(val))
@pytest.mark.parametrize('ftype', [np.float32, np.float64])
# 使用 pytest 的 parametrize 装饰器，为 test_memoverlap_accumulate 函数创建多个参数化的测试实例
def test_memoverlap_accumulate(ftype):
    # 重现 bug https://github.com/numpy/numpy/issues/15597
    # 创建一个包含指定数据类型的 numpy 数组
    arr = np.array([0.61, 0.60, 0.77, 0.41, 0.19], dtype=ftype)
    # 创建预期的最大值累积结果数组
    out_max = np.array([0.61, 0.61, 0.77, 0.77, 0.77], dtype=ftype)
    # 创建预期的最小值累积结果数组
    out_min = np.array([0.61, 0.60, 0.60, 0.41, 0.19], dtype=ftype)
    # 使用 assert_equal 断言函数比较 numpy 的最大值累积和最小值累积函数的结果
    assert_equal(np.maximum.accumulate(arr), out_max)
    assert_equal(np.minimum.accumulate(arr), out_min)

@pytest.mark.parametrize("ufunc, dtype", [
    # 为 test_memoverlap_accumulate_cmp 函数创建多个参数化的测试实例，用于比较二进制操作函数的累积结果
    (ufunc, t[0])
    for ufunc in UFUNCS_BINARY_ACC
    for t in ufunc.types
    if t[-1] == '?' and t[0] not in 'DFGMmO'
])
def test_memoverlap_accumulate_cmp(ufunc, dtype):
    if ufunc.signature:
        # 如果 ufunc 有特定签名，跳过该测试
        pytest.skip('For generic signatures only')
    # 对于指定的大小进行循环测试
    for size in (2, 8, 32, 64, 128, 256):
        # 创建包含指定数据类型的 numpy 数组
        arr = np.array([0, 1, 1]*size, dtype=dtype)
        # 调用 ufunc 的累积函数，将结果转换为 uint8 类型
        acc = ufunc.accumulate(arr, dtype='?')
        # 将累积结果视图转换为 uint8 类型
        acc_u8 = acc.view(np.uint8)
        # 创建预期的累积结果数组
        exp = np.array(list(itertools.accumulate(arr, ufunc)), dtype=np.uint8)
        # 使用 assert_equal 断言函数比较预期结果和累积结果数组
        assert_equal(exp, acc_u8)

@pytest.mark.parametrize("ufunc, dtype", [
    # 为 test_memoverlap_accumulate_symmetric 函数创建多个参数化的测试实例，用于比较对称数据类型的累积结果
    (ufunc, t[0])
    for ufunc in UFUNCS_BINARY_ACC
    for t in ufunc.types
    if t[0] == t[1] and t[0] == t[-1] and t[0] not in 'DFGMmO?'
])
def test_memoverlap_accumulate_symmetric(ufunc, dtype):
    if ufunc.signature:
        # 如果 ufunc 有特定签名，跳过该测试
        pytest.skip('For generic signatures only')
    # 忽略所有的 numpy 错误
    with np.errstate(all='ignore'):
        # 对于指定的大小进行循环测试
        for size in (2, 8, 32, 64, 128, 256):
            # 创建包含指定数据类型的 numpy 数组
            arr = np.array([0, 1, 2]*size).astype(dtype)
            # 调用 ufunc 的累积函数，将结果转换为指定数据类型
            acc = ufunc.accumulate(arr, dtype=dtype)
            # 创建预期的累积结果数组
            exp = np.array(list(itertools.accumulate(arr, ufunc)), dtype=dtype)
            # 使用 assert_equal 断言函数比较预期结果和累积结果数组
            assert_equal(exp, acc)

def test_signaling_nan_exceptions():
    # 使用 assert_no_warnings 上下文管理器来测试 numpy 是否正确处理信号 NaN 异常
    with assert_no_warnings():
        # 创建一个包含信号 NaN 的 float32 类型的 numpy 数组
        a = np.ndarray(shape=(), dtype='float32', buffer=b'\x00\xe0\xbf\xff')
        # 使用 np.isnan 函数来检测是否存在 NaN 值
        np.isnan(a)

@pytest.mark.parametrize("arr", [
    # 为 test_outer_subclass_preserve 函数创建多个参数化的测试实例，用于测试外部子类的保留
    np.arange(2),
    np.matrix([0, 1]),
    np.matrix([[0, 1], [2, 5]]),
    ])
def test_outer_subclass_preserve(arr):
    # 为了 gh-8661
    # 创建一个名为 foo 的 numpy 子类
    class foo(np.ndarray): pass
    # 调用 np.multiply.outer 函数生成外部乘积，并使用 foo 类型的视图
    actual = np.multiply.outer(arr.view(foo), arr.view(foo))
    # 使用 assert 断言函数比较实际结果的类名是否为 'foo'
    assert actual.__class__.__name__ == 'foo'

def test_outer_bad_subclass():
    # 创建 BadArr1 类作为 np.ndarray 的子类
    class BadArr1(np.ndarray):
        def __array_finalize__(self, obj):
            # 外部调用会将形状重塑为 3 维，尝试进行错误的重塑
            if self.ndim == 3:
                self.shape = self.shape + (1,)

    # 创建 BadArr2 类作为 np.ndarray 的子类
    class BadArr2(np.ndarray):
        def __array_finalize__(self, obj):
            if isinstance(obj, BadArr2):
                # 外部插入 1 大小的维度，干扰这些维度
                if self.shape[-1] == 1:
                    self.shape = self.shape[::-1]
    for cls in [BadArr1, BadArr2]:
        # 使用给定的类（BadArr1 或 BadArr2）创建一个形状为 (2, 3) 的全 1 数组，并将其视图转换为指定类的对象
        arr = np.ones((2, 3)).view(cls)
        
        # 使用 assert_raises 来断言在执行下面的操作时会抛出 TypeError 异常
        with assert_raises(TypeError) as a:
            # 对第一个数组进行外积运算（第二个数组不会被重新形状）
            np.add.outer(arr, [1, 2])

        # 这个断言可以成功，因为我们只会看到重新形状的错误：
        # 再次创建一个形状为 (2, 3) 的全 1 数组，并将其视图转换为指定类的对象
        arr = np.ones((2, 3)).view(cls)
        # 使用 np.add.outer 对数组 [1, 2] 和 arr 进行外积运算，并断言结果的类型是指定的类对象
        assert type(np.add.outer([1, 2], arr)) is cls
def test_outer_exceeds_maxdims():
    # 创建一个形状为 (1,)*33 的多维数组，所有元素为1，深度为33
    deep = np.ones((1,) * 33)
    # 使用 assert_raises 检查是否会引发 ValueError 异常
    with assert_raises(ValueError):
        # 对 deep 数组使用 np.add.outer 进行外积操作
        np.add.outer(deep, deep)

def test_bad_legacy_ufunc_silent_errors():
    # legacy ufuncs 无法报告错误，并且 NumPy 无法检查 GIL 是否已释放。
    # 因此，NumPy 必须在释放 GIL 后检查，以确保覆盖所有情况。
    # `np.power` 之前/现在使用了这种方式。
    # 创建一个长度为3的浮点型数组 arr
    arr = np.arange(3).astype(np.float64)

    # 使用 pytest.raises 检查是否会引发 RuntimeError 异常，并且异常信息匹配指定的正则表达式
    with pytest.raises(RuntimeError, match=r"How unexpected :\)!"):
        # 调用 ncu_tests.always_error 函数，传入 arr 两次作为参数
        ncu_tests.always_error(arr, arr)

    with pytest.raises(RuntimeError, match=r"How unexpected :\)!"):
        # 创建一个非连续的数组 non_contig，使得无法采用快速路径
        non_contig = arr.repeat(20).reshape(-1, 6)[:, ::2]
        # 调用 ncu_tests.always_error 函数，传入 non_contig 和 arr 作为参数
        ncu_tests.always_error(non_contig, arr)

    with pytest.raises(RuntimeError, match=r"How unexpected :\)!"):
        # 调用 ncu_tests.always_error.outer 函数，传入 arr 两次作为参数
        ncu_tests.always_error.outer(arr, arr)

    with pytest.raises(RuntimeError, match=r"How unexpected :\)!"):
        # 调用 ncu_tests.always_error.reduce 函数，传入 arr 作为参数
        ncu_tests.always_error.reduce(arr)

    with pytest.raises(RuntimeError, match=r"How unexpected :\)!"):
        # 调用 ncu_tests.always_error.reduceat 函数，传入 arr 和 [0, 1] 作为参数
        ncu_tests.always_error.reduceat(arr, [0, 1])

    with pytest.raises(RuntimeError, match=r"How unexpected :\)!"):
        # 调用 ncu_tests.always_error.accumulate 函数，传入 arr 作为参数
        ncu_tests.always_error.accumulate(arr)

    with pytest.raises(RuntimeError, match=r"How unexpected :\)!"):
        # 调用 ncu_tests.always_error.at 函数，传入 arr、[0, 1, 2] 和 arr 作为参数
        ncu_tests.always_error.at(arr, [0, 1, 2], arr)


@pytest.mark.parametrize('x1', [np.arange(3.0), [0.0, 1.0, 2.0]])
def test_bad_legacy_gufunc_silent_errors(x1):
    # 验证 gufunc 循环中引发的异常是否正确传播。
    # always_error_gufunc 的签名为 '(i),()->()'
    with pytest.raises(RuntimeError, match=r"How unexpected :\)!"):
        # 调用 ncu_tests.always_error_gufunc 函数，传入 x1 和 0.0 作为参数
        ncu_tests.always_error_gufunc(x1, 0.0)


class TestAddDocstring:
    @pytest.mark.skipif(sys.flags.optimize == 2, reason="Python running -OO")
    @pytest.mark.skipif(IS_PYPY, reason="PyPy does not modify tp_doc")
    def test_add_same_docstring(self):
        # 测试属性（这些是 C 级别定义的）
        # 调用 ncu.add_docstring 函数，传入 np.ndarray.flat 和 np.ndarray.flat.__doc__ 作为参数
        ncu.add_docstring(np.ndarray.flat, np.ndarray.flat.__doc__)

        # 典型函数：
        def func():
            """docstring"""
            return

        # 调用 ncu.add_docstring 函数，传入 func 和 func.__doc__ 作为参数
        ncu.add_docstring(func, func.__doc__)

    @pytest.mark.skipif(sys.flags.optimize == 2, reason="Python running -OO")
    def test_different_docstring_fails(self):
        # 测试属性（这些是 C 级别定义的）
        with assert_raises(RuntimeError):
            # 调用 ncu.add_docstring 函数，传入 np.ndarray.flat 和 "different docstring" 作为参数
            ncu.add_docstring(np.ndarray.flat, "different docstring")

        # 典型函数：
        def func():
            """docstring"""
            return

        with assert_raises(RuntimeError):
            # 调用 ncu.add_docstring 函数，传入 func 和 "different docstring" 作为参数
            ncu.add_docstring(func, "different docstring")


class TestAdd_newdoc_ufunc:
    def test_ufunc_arg(self):
        # 检查是否会引发 TypeError 异常，传入 2 和 "blah" 作为参数
        assert_raises(TypeError, ncu._add_newdoc_ufunc, 2, "blah")
        # 检查是否会引发 ValueError 异常，传入 np.add 和 "blah" 作为参数
        assert_raises(ValueError, ncu._add_newdoc_ufunc, np.add, "blah")
    # 定义一个测试方法 `test_string_arg(self)`
    def test_string_arg(self):
        # 使用 assert_raises 检查是否会引发 TypeError 异常
        assert_raises(TypeError, ncu._add_newdoc_ufunc, np.add, 3)
```