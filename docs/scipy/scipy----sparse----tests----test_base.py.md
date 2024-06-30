# `D:\src\scipysrc\scipy\scipy\sparse\tests\test_base.py`

```
#
# Authors: Travis Oliphant, Ed Schofield, Robert Cimrman, Nathan Bell, and others
#
# 用于稀疏矩阵测试的测试函数。在“基于矩阵类的测试”部分的每个类都成为“通用测试”部分类的子类。
# 这是通过“为通用测试定制的基类”部分中的函数完成的。
#

import contextlib  # 上下文管理工具模块，用于创建和管理上下文
import functools  # 函数工具模块，提供函数操作的工具
import operator  # 操作符模块，提供Python中的各种运算符操作函数
import platform  # 平台模块，提供获取底层操作系统信息的函数
import itertools  # 迭代工具模块，提供用于创建和操作迭代器的函数

import sys  # 系统模块，提供访问与Python解释器相关的变量和函数

import pytest  # 测试工具模块，用于编写和运行测试用例
from pytest import raises as assert_raises  # 导入 raises 方法并重命名为 assert_raises

import numpy as np  # 导入 NumPy 库，并重命名为 np
from numpy import (arange, zeros, array, dot, asarray,  # 从 NumPy 导入多个函数和对象
                   vstack, ndarray, transpose, diag, kron, inf, conjugate,
                   int8)

import random  # 随机数模块，提供生成随机数的函数
from numpy.testing import (assert_equal, assert_array_equal,  # 从 NumPy 测试模块导入断言函数
        assert_array_almost_equal, assert_almost_equal, assert_,
        assert_allclose, suppress_warnings)

import scipy.linalg  # SciPy 线性代数模块，提供线性代数相关函数

import scipy.sparse as sparse  # SciPy 稀疏矩阵模块，重命名为 sparse
from scipy.sparse import (csc_matrix, csr_matrix, dok_matrix,  # 导入稀疏矩阵类型
        coo_matrix, lil_matrix, dia_matrix, bsr_matrix,
        eye, issparse, SparseEfficiencyWarning, sparray)
from scipy.sparse._base import _formats  # 导入稀疏矩阵基类模块中的 _formats 函数
from scipy.sparse._sputils import (supported_dtypes, isscalarlike,  # 导入稀疏矩阵工具函数
                                   get_index_dtype, asmatrix, matrix)
from scipy.sparse.linalg import splu, expm, inv  # 导入稀疏矩阵线性代数函数

from scipy._lib.decorator import decorator  # 装饰器模块，提供装饰器函数
from scipy._lib._util import ComplexWarning  # SciPy 工具模块，导入复杂警告类


IS_COLAB = ('google.colab' in sys.modules)  # 检测当前是否在 Google Colab 环境中


def assert_in(member, collection, msg=None):
    # 断言函数，验证 member 是否在 collection 中
    message = msg if msg is not None else f"{member!r} not found in {collection!r}"
    assert_(member in collection, msg=message)


def assert_array_equal_dtype(x, y, **kwargs):
    # 断言函数，验证 x 和 y 的数据类型相同，并且它们相等
    assert_(x.dtype == y.dtype)
    assert_array_equal(x, y, **kwargs)


NON_ARRAY_BACKED_FORMATS = frozenset(['dok'])

def sparse_may_share_memory(A, B):
    # 检查 A 和 B 是否共享任何 NumPy 数组内存

    def _underlying_arrays(x):
        # 给定任何对象（例如稀疏数组），返回所有存储在其属性中的 NumPy 数组

        arrays = []
        for a in x.__dict__.values():
            if isinstance(a, (np.ndarray, np.generic)):
                arrays.append(a)
        return arrays

    for a in _underlying_arrays(A):
        for b in _underlying_arrays(B):
            if np.may_share_memory(a, b):
                return True
    return False


sup_complex = suppress_warnings()
sup_complex.filter(ComplexWarning)


def with_64bit_maxval_limit(maxval_limit=None, random=False, fixed_dtype=None,
                            downcast_maxval=None, assert_32bit=False):
    """
    Monkeypatch the maxval threshold at which scipy.sparse switches to
    64-bit index arrays, or make it (pseudo-)random.

    """
    if maxval_limit is None:
        maxval_limit = np.int64(10)
    else:
        # 确保使用 NumPy 标量而不是 Python 标量（对 NEP 50 转换规则变更很重要）
        maxval_limit = np.int64(maxval_limit)
    # 如果 assert_32bit 为真，定义一个新的函数 new_get_index_dtype
    def new_get_index_dtype(arrays=(), maxval=None, check_contents=False):
        # 获取索引的数据类型
        tp = get_index_dtype(arrays, maxval, check_contents)
        # 断言索引的最大值等于 np.int32 的最大值
        assert_equal(np.iinfo(tp).max, np.iinfo(np.int32).max)
        # 断言索引的类型是 np.int32 或者 np.intc
        assert_(tp == np.int32 or tp == np.intc)
        return tp
    
    # 如果 fixed_dtype 不为 None，定义一个新的函数 new_get_index_dtype
    elif fixed_dtype is not None:
        def new_get_index_dtype(arrays=(), maxval=None, check_contents=False):
            return fixed_dtype
    
    # 如果 random 为真，使用 seed 1234 创建一个随机状态对象 counter
    elif random:
        counter = np.random.RandomState(seed=1234)
        # 定义一个新的函数 new_get_index_dtype
        def new_get_index_dtype(arrays=(), maxval=None, check_contents=False):
            # 返回 np.int32 或 np.int64 中的一个，由 counter 随机确定
            return (np.int32, np.int64)[counter.randint(2)]
    
    # 如果上述条件都不满足，定义一个新的函数 new_get_index_dtype
    else:
        def new_get_index_dtype(arrays=(), maxval=None, check_contents=False):
            # 默认使用 np.int32 作为索引的数据类型
            dtype = np.int32
            # 如果指定了 maxval，并且超过了 maxval_limit，则使用 np.int64
            if maxval is not None:
                if maxval > maxval_limit:
                    dtype = np.int64
            # 遍历 arrays 中的每个数组
            for arr in arrays:
                # 将数组转换为 NumPy 数组
                arr = np.asarray(arr)
                # 如果数组的数据类型超过 np.int32
                if arr.dtype > np.int32:
                    # 如果需要检查数组内容
                    if check_contents:
                        # 如果数组为空，则不需要更大的数据类型
                        if arr.size == 0:
                            continue
                        # 如果数组是整数类型，检查其最大值和最小值是否在范围内
                        elif np.issubdtype(arr.dtype, np.integer):
                            maxval = arr.max()
                            minval = arr.min()
                            # 如果最小值大于等于 -maxval_limit，最大值小于等于 maxval_limit，则不需要更大的数据类型
                            if minval >= -maxval_limit and maxval <= maxval_limit:
                                continue
                    # 否则，默认使用 np.int64
                    dtype = np.int64
            return dtype
    
    # 如果 downcast_maxval 不为 None，定义一个新的函数 new_downcast_intp_index
    if downcast_maxval is not None:
        def new_downcast_intp_index(arr):
            # 如果数组的最大值超过 downcast_maxval，则引发 AssertionError
            if arr.max() > downcast_maxval:
                raise AssertionError("downcast limited")
            return arr.astype(np.intp)
    
    # 定义一个装饰器函数 deco
    @decorator
    def deco(func, *a, **kw):
        backup = []
        # 模块列表，包含了需要备份和替换的模块
        modules = [scipy.sparse._bsr, scipy.sparse._coo, scipy.sparse._csc,
                   scipy.sparse._csr, scipy.sparse._dia, scipy.sparse._dok,
                   scipy.sparse._lil, scipy.sparse._sputils,
                   scipy.sparse._compressed, scipy.sparse._construct]
        try:
            # 替换模块中的 get_index_dtype 函数为 new_get_index_dtype
            for mod in modules:
                backup.append((mod, 'get_index_dtype',
                               getattr(mod, 'get_index_dtype', None)))
                setattr(mod, 'get_index_dtype', new_get_index_dtype)
                # 如果 downcast_maxval 不为 None，替换模块中的 downcast_intp_index 函数为 new_downcast_intp_index
                if downcast_maxval is not None:
                    backup.append((mod, 'downcast_intp_index',
                                   getattr(mod, 'downcast_intp_index', None)))
                    setattr(mod, 'downcast_intp_index', new_downcast_intp_index)
            # 执行被装饰的函数 func，并返回其结果
            return func(*a, **kw)
        finally:
            # 恢复替换前的函数定义
            for mod, name, oldfunc in backup:
                if oldfunc is not None:
                    setattr(mod, name, oldfunc)
    
    # 返回装饰器函数 deco
    return deco
# 定义函数 toarray，将输入转换为 numpy 数组
def toarray(a):
    # 如果 a 是 numpy 数组或者类似标量，则直接返回 a
    if isinstance(a, np.ndarray) or isscalarlike(a):
        return a
    # 否则，将 a 转换为稀疏数组的密集表示并返回
    return a.toarray()

# 定义一个用于测试稀疏矩阵二元操作的类 BinopTester
class BinopTester:
    # 自定义类型，用于测试在稀疏矩阵上的二元操作

    # 定义右加法操作，返回字符串 "matrix on the right"
    def __add__(self, mat):
        return "matrix on the right"

    # 定义右乘法操作，返回字符串 "matrix on the right"
    def __mul__(self, mat):
        return "matrix on the right"

    # 定义右减法操作，返回字符串 "matrix on the right"
    def __sub__(self, mat):
        return "matrix on the right"

    # 定义左加法操作，返回字符串 "matrix on the left"
    def __radd__(self, mat):
        return "matrix on the left"

    # 定义左乘法操作，返回字符串 "matrix on the left"
    def __rmul__(self, mat):
        return "matrix on the left"

    # 定义左减法操作，返回字符串 "matrix on the left"
    def __rsub__(self, mat):
        return "matrix on the left"

    # 定义矩阵乘法操作，返回字符串 "matrix on the right"
    def __matmul__(self, mat):
        return "matrix on the right"

    # 定义矩阵右乘操作，返回字符串 "matrix on the left"
    def __rmatmul__(self, mat):
        return "matrix on the left"

# 定义一个带有形状属性的 BinopTester_with_shape 类
class BinopTester_with_shape:
    # 自定义类型，用于测试在带有形状属性的对象上的二元操作

    # 初始化方法，接受形状参数 shape，并将其保存在实例变量 _shape 中
    def __init__(self, shape):
        self._shape = shape

    # 返回存储在实例变量 _shape 中的形状信息
    def shape(self):
        return self._shape

    # 返回实例变量 _shape 的维度数
    def ndim(self):
        return len(self._shape)

    # 定义右加法操作，返回字符串 "matrix on the right"
    def __add__(self, mat):
        return "matrix on the right"

    # 定义右乘法操作，返回字符串 "matrix on the right"
    def __mul__(self, mat):
        return "matrix on the right"

    # 定义右减法操作，返回字符串 "matrix on the right"
    def __sub__(self, mat):
        return "matrix on the right"

    # 定义左加法操作，返回字符串 "matrix on the left"
    def __radd__(self, mat):
        return "matrix on the left"

    # 定义左乘法操作，返回字符串 "matrix on the left"
    def __rmul__(self, mat):
        return "matrix on the left"

    # 定义左减法操作，返回字符串 "matrix on the left"
    def __rsub__(self, mat):
        return "matrix on the left"

    # 定义矩阵乘法操作，返回字符串 "matrix on the right"
    def __matmul__(self, mat):
        return "matrix on the right"

    # 定义矩阵右乘操作，返回字符串 "matrix on the left"
    def __rmatmul__(self, mat):
        return "matrix on the left"

# 定义一个用于测试稀疏矩阵比较操作的类 ComparisonTester
class ComparisonTester:
    # 自定义类型，用于测试在稀疏矩阵上的比较操作

    # 定义等于比较操作，返回字符串 "eq"
    def __eq__(self, other):
        return "eq"

    # 定义不等于比较操作，返回字符串 "ne"
    def __ne__(self, other):
        return "ne"

    # 定义小于比较操作，返回字符串 "lt"
    def __lt__(self, other):
        return "lt"

    # 定义小于等于比较操作，返回字符串 "le"
    def __le__(self, other):
        return "le"

    # 定义大于比较操作，返回字符串 "gt"
    def __gt__(self, other):
        return "gt"

    # 定义大于等于比较操作，返回字符串 "ge"
    def __ge__(self, other):
        return "ge"


#------------------------------------------------------------------------------
# Generic tests
#------------------------------------------------------------------------------

# TODO test prune
# TODO test has_sorted_indices

# 定义一个测试通用功能的基类 _TestCommon
class _TestCommon:
    """test common functionality shared by all sparse formats"""
    # 数学数据类型包括所有支持的数据类型
    math_dtypes = supported_dtypes

    # @classmethod 修饰的方法定义开始
    @classmethod
    # 初始化类的静态数据
    def init_class(cls):
        # 定义一个标准数据数组
        cls.dat = array([[1, 0, 0, 2], [3, 0, 1, 0], [0, 2, 0, 0]], 'd')
        # 使用类的方法创建稀疏矩阵
        cls.datsp = cls.spcreator(cls.dat)

        # 创建一些包含不同数据类型的稀疏和密集矩阵集合
        # 这个集合并操作是为了解决 numpy#6295 的问题，即两个 np.int64 类型不会散列到相同的值
        cls.checked_dtypes = set(supported_dtypes).union(cls.math_dtypes)
        cls.dat_dtypes = {}
        cls.datsp_dtypes = {}
        for dtype in cls.checked_dtypes:
            # 将标准数据数组转换为指定数据类型的数组并存储
            cls.dat_dtypes[dtype] = cls.dat.astype(dtype)
            # 使用类的方法创建指定数据类型的稀疏矩阵并存储
            cls.datsp_dtypes[dtype] = cls.spcreator(cls.dat.astype(dtype))

        # 检查原始数据与相应的 dat_dtypes 和 datsp_dtypes 是否等价
        assert_equal(cls.dat, cls.dat_dtypes[np.float64])
        assert_equal(cls.datsp.toarray(),
                     cls.datsp_dtypes[np.float64].toarray())

    # 测试布尔操作
    def test_bool(self):
        # 内部函数，检查特定数据类型的稀疏矩阵布尔值
        def check(dtype):
            datsp = self.datsp_dtypes[dtype]

            # 断言会抛出 ValueError，检查稀疏矩阵的布尔值
            assert_raises(ValueError, bool, datsp)
            # 断言某些条件下的布尔真值
            assert_(self.spcreator([[1]]))
            # 断言某些条件下的布尔假值
            assert_(not self.spcreator([[0]]))

        # 如果当前实例属于 TestDOK 类，则跳过测试
        if isinstance(self, TestDOK):
            pytest.skip("Cannot create a rank <= 2 DOK matrix.")
        
        # 对每种数据类型进行布尔操作检查
        for dtype in self.checked_dtypes:
            check(dtype)

    # 测试布尔值溢出
    def test_bool_rollover(self):
        # 布尔类型的底层数据类型是 1 字节，检查不会在 256 时 True -> False
        dat = array([[True, False]])
        datsp = self.spcreator(dat)

        # 多次迭代加法，检查布尔值溢出
        for _ in range(10):
            datsp = datsp + datsp
            dat = dat + dat
        # 断言数组是否相等
        assert_array_equal(dat, datsp.toarray())
    # 定义测试方法 test_eq，用于测试稀疏矩阵的相等性和比较操作
    def test_eq(self):
        # 创建警告抑制器对象
        sup = suppress_warnings()
        # 过滤稀疏效率警告
        sup.filter(SparseEfficiencyWarning)

        # 使用警告抑制器装饰下面的函数 check
        @sup
        @sup_complex
        def check(dtype):
            # 获取相应数据类型的数据
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            # 复制数据 dat 到 dat2
            dat2 = dat.copy()
            # 修改 dat2 的第一列为 0
            dat2[:,0] = 0
            # 使用 spcreator 方法创建 datsp2 稀疏矩阵
            datsp2 = self.spcreator(dat2)
            # 创建 BSR 格式的稀疏矩阵 datbsr
            datbsr = bsr_matrix(dat)
            # 创建 CSR 格式的稀疏矩阵 datcsr
            datcsr = csr_matrix(dat)
            # 创建 CSC 格式的稀疏矩阵 datcsc
            datcsc = csc_matrix(dat)
            # 创建 LIL 格式的稀疏矩阵 datlil
            datlil = lil_matrix(dat)

            # 测试稀疏/稀疏矩阵相等性
            assert_array_equal_dtype(dat == dat2, (datsp == datsp2).toarray())
            # 测试不同稀疏类型之间的相等性
            assert_array_equal_dtype(dat == dat2, (datbsr == datsp2).toarray())
            assert_array_equal_dtype(dat == dat2, (datcsr == datsp2).toarray())
            assert_array_equal_dtype(dat == dat2, (datcsc == datsp2).toarray())
            assert_array_equal_dtype(dat == dat2, (datlil == datsp2).toarray())
            # 测试稀疏/稠密矩阵相等性
            assert_array_equal_dtype(dat == datsp2, datsp2 == dat)
            # 测试稀疏/标量比较
            assert_array_equal_dtype(dat == 0, (datsp == 0).toarray())
            assert_array_equal_dtype(dat == 1, (datsp == 1).toarray())
            assert_array_equal_dtype(dat == np.nan,
                                     (datsp == np.nan).toarray())

        # 如果 self 不是 TestBSR、TestCSC 或 TestCSR 的实例，则跳过测试
        if not isinstance(self, (TestBSR, TestCSC, TestCSR)):
            pytest.skip("Bool comparisons only implemented for BSR, CSC, and CSR.")
        # 遍历待检查的数据类型列表
        for dtype in self.checked_dtypes:
            # 调用 check 函数进行测试
            check(dtype)
    # 定义一个测试方法来测试不相等情况
    def test_ne(self):
        # 获取一个警告抑制器实例
        sup = suppress_warnings()
        # 过滤稀疏效率警告
        sup.filter(SparseEfficiencyWarning)

        # 使用 sup 和 sup_complex 修饰符定义一个内部函数 check(dtype)
        @sup
        @sup_complex
        def check(dtype):
            # 获取特定数据类型的数据和稀疏数据
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            
            # 复制 dat 数据并将第一列设为 0
            dat2 = dat.copy()
            dat2[:,0] = 0
            
            # 通过 self.spcreator 方法创建稀疏数据 datsp2
            datsp2 = self.spcreator(dat2)
            
            # 将 dat 转换为 BSR 矩阵、CSC 矩阵、CSR 矩阵和 LIL 矩阵
            datbsr = bsr_matrix(dat)
            datcsc = csc_matrix(dat)
            datcsr = csr_matrix(dat)
            datlil = lil_matrix(dat)

            # 断言不相等的数组是否相等，比较稀疏和稠密数据
            assert_array_equal_dtype(dat != dat2, (datsp != datsp2).toarray())
            assert_array_equal_dtype(dat != dat2, (datbsr != datsp2).toarray())
            assert_array_equal_dtype(dat != dat2, (datcsc != datsp2).toarray())
            assert_array_equal_dtype(dat != dat2, (datcsr != datsp2).toarray())
            assert_array_equal_dtype(dat != dat2, (datlil != datsp2).toarray())

            # 比较稀疏和稀疏数据的不相等性
            assert_array_equal_dtype(dat != datsp2, datsp2 != dat)
            
            # 比较稠密和稀疏数据与标量的不相等性
            assert_array_equal_dtype(dat != 0, (datsp != 0).toarray())
            assert_array_equal_dtype(dat != 1, (datsp != 1).toarray())
            assert_array_equal_dtype(0 != dat, (0 != datsp).toarray())
            assert_array_equal_dtype(1 != dat, (1 != datsp).toarray())
            
            # 比较数据与 NaN 的不相等性
            assert_array_equal_dtype(dat != np.nan, (datsp != np.nan).toarray())

        # 如果 self 不是 TestBSR、TestCSC 或 TestCSR 的实例，则跳过测试
        if not isinstance(self, (TestBSR, TestCSC, TestCSR)):
            pytest.skip("Bool comparisons only implemented for BSR, CSC, and CSR.")
        
        # 对于每种被检查的数据类型，调用 check(dtype)
        for dtype in self.checked_dtypes:
            check(dtype)
    # 定义一个测试方法，用于测试小于操作
    def test_lt(self):
        # 创建一个警告抑制器对象
        sup = suppress_warnings()
        # 过滤掉稀疏矩阵效率警告
        sup.filter(SparseEfficiencyWarning)

        # 使用警告抑制器装饰下面的函数
        @sup
        @sup_complex
        def check(dtype):
            # data
            # 从 self.dat_dtypes 中获取指定类型的数据
            dat = self.dat_dtypes[dtype]
            # 从 self.datsp_dtypes 中获取指定类型的稀疏数据
            datsp = self.datsp_dtypes[dtype]
            # 复制 dat，并将第一列置为 0
            dat2 = dat.copy()
            dat2[:,0] = 0
            # 用 self.spcreator 方法创建 dat2 的稀疏版本
            datsp2 = self.spcreator(dat2)
            # 将 dat 转换为复数类型，并将第一列置为 1 + 1j
            datcomplex = dat.astype(complex)
            datcomplex[:,0] = 1 + 1j
            # 将 datcomplex 转换为稀疏复数类型
            datspcomplex = self.spcreator(datcomplex)
            # 创建 BSR 格式的稀疏矩阵
            datbsr = bsr_matrix(dat)
            # 创建 CSC 格式的稀疏矩阵
            datcsc = csc_matrix(dat)
            # 创建 CSR 格式的稀疏矩阵
            datcsr = csr_matrix(dat)
            # 创建 LIL 格式的稀疏矩阵
            datlil = lil_matrix(dat)

            # sparse/sparse
            # 检查稀疏与稀疏矩阵的小于比较结果
            assert_array_equal_dtype(dat < dat2, (datsp < datsp2).toarray())
            assert_array_equal_dtype(datcomplex < dat2,
                                     (datspcomplex < datsp2).toarray())
            # mix sparse types
            # 检查不同稀疏类型之间的小于比较结果
            assert_array_equal_dtype(dat < dat2, (datbsr < datsp2).toarray())
            assert_array_equal_dtype(dat < dat2, (datcsc < datsp2).toarray())
            assert_array_equal_dtype(dat < dat2, (datcsr < datsp2).toarray())
            assert_array_equal_dtype(dat < dat2, (datlil < datsp2).toarray())

            # 反向检查稀疏与稀疏矩阵的小于比较结果
            assert_array_equal_dtype(dat2 < dat, (datsp2 < datbsr).toarray())
            assert_array_equal_dtype(dat2 < dat, (datsp2 < datcsc).toarray())
            assert_array_equal_dtype(dat2 < dat, (datsp2 < datcsr).toarray())
            assert_array_equal_dtype(dat2 < dat, (datsp2 < datlil).toarray())
            # sparse/dense
            # 检查稀疏与稠密矩阵的小于比较结果
            assert_array_equal_dtype(dat < dat2, datsp < dat2)
            assert_array_equal_dtype(datcomplex < dat2, datspcomplex < dat2)
            # sparse/scalar
            # 对于不同的标量值，检查稀疏矩阵与标量的小于比较结果
            for val in [2, 1, 0, -1, -2]:
                val = np.int64(val)  # 避免 Python 标量（由于 NEP 50 改变）
                assert_array_equal_dtype((datsp < val).toarray(), dat < val)
                assert_array_equal_dtype((val < datsp).toarray(), val < dat)

            # 在忽略无效值错误的情况下，检查稀疏矩阵与 NaN 的小于比较结果
            with np.errstate(invalid='ignore'):
                assert_array_equal_dtype((datsp < np.nan).toarray(),
                                         dat < np.nan)

            # data
            # 重新加载数据，准备下一次循环
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            dat2 = dat.copy()
            dat2[:,0] = 0
            datsp2 = self.spcreator(dat2)

            # dense rhs
            # 检查稀疏矩阵与稠密矩阵的小于比较结果
            assert_array_equal_dtype(dat < datsp2, datsp < dat2)

        # 如果当前对象不是 TestBSR、TestCSC 或 TestCSR 的实例，则跳过测试
        if not isinstance(self, (TestBSR, TestCSC, TestCSR)):
            pytest.skip("Bool comparisons only implemented for BSR, CSC, and CSR.")
        # 针对每种被检查的数据类型执行测试
        for dtype in self.checked_dtypes:
            check(dtype)
    # 定义一个测试方法，用于测试大于运算符的功能
    def test_gt(self):
        # 创建一个警告抑制器对象
        sup = suppress_warnings()
        # 过滤稀疏效率警告
        sup.filter(SparseEfficiencyWarning)

        # 定义一个装饰器函数check，接收一个数据类型参数dtype
        @sup
        @sup_complex
        def check(dtype):
            # 获取普通数据和稀疏数据
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]

            # 复制普通数据，并对其进行修改
            dat2 = dat.copy()
            dat2[:,0] = 0

            # 使用self.spcreator创建修改后的稀疏数据
            datsp2 = self.spcreator(dat2)

            # 将普通数据转换为复数类型，并对其进行修改
            datcomplex = dat.astype(complex)
            datcomplex[:,0] = 1 + 1j

            # 使用self.spcreator创建修改后的复数类型稀疏数据
            datspcomplex = self.spcreator(datcomplex)

            # 使用scipy.sparse创建不同格式的稀疏矩阵
            datbsr = bsr_matrix(dat)
            datcsc = csc_matrix(dat)
            datcsr = csr_matrix(dat)
            datlil = lil_matrix(dat)

            # 检查稀疏矩阵与稀疏矩阵的大于比较
            assert_array_equal_dtype(dat > dat2, (datsp > datsp2).toarray())
            assert_array_equal_dtype(datcomplex > dat2,
                                     (datspcomplex > datsp2).toarray())

            # 检查不同稀疏类型之间的大于比较
            assert_array_equal_dtype(dat > dat2, (datbsr > datsp2).toarray())
            assert_array_equal_dtype(dat > dat2, (datcsc > datsp2).toarray())
            assert_array_equal_dtype(dat > dat2, (datcsr > datsp2).toarray())
            assert_array_equal_dtype(dat > dat2, (datlil > datsp2).toarray())

            # 检查反向稀疏矩阵与稀疏矩阵的大于比较
            assert_array_equal_dtype(dat2 > dat, (datsp2 > datbsr).toarray())
            assert_array_equal_dtype(dat2 > dat, (datsp2 > datcsc).toarray())
            assert_array_equal_dtype(dat2 > dat, (datsp2 > datcsr).toarray())
            assert_array_equal_dtype(dat2 > dat, (datsp2 > datlil).toarray())

            # 检查稀疏矩阵与普通矩阵的大于比较
            assert_array_equal_dtype(dat > dat2, datsp > dat2)
            assert_array_equal_dtype(datcomplex > dat2, datspcomplex > dat2)

            # 检查稀疏矩阵与标量的大于比较
            for val in [2, 1, 0, -1, -2]:
                val = np.int64(val)  # 避免Python标量 (因为NEP 50更改)
                assert_array_equal_dtype((datsp > val).toarray(), dat > val)
                assert_array_equal_dtype((val > datsp).toarray(), val > dat)

            # 忽略无效值的情况下，检查稀疏矩阵与NaN的大于比较
            with np.errstate(invalid='ignore'):
                assert_array_equal_dtype((datsp > np.nan).toarray(),
                                         dat > np.nan)

            # 重新获取数据，准备进行下一轮测试
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            dat2 = dat.copy()
            dat2[:,0] = 0
            datsp2 = self.spcreator(dat2)

            # 检查稠密矩阵与修改后的稀疏矩阵的大于比较
            assert_array_equal_dtype(dat > datsp2, datsp > dat2)

        # 如果self不属于TestBSR、TestCSC或TestCSR类型，则跳过测试
        if not isinstance(self, (TestBSR, TestCSC, TestCSR)):
            pytest.skip("Bool comparisons only implemented for BSR, CSC, and CSR.")

        # 对self.checked_dtypes中的每个数据类型执行check函数
        for dtype in self.checked_dtypes:
            check(dtype)
    # 定义一个测试方法来测试小于等于运算符的行为
    def test_le(self):
        # 获取警告抑制器对象
        sup = suppress_warnings()
        # 过滤掉稀疏效率警告
        sup.filter(SparseEfficiencyWarning)

        # 定义内部函数 check，用于测试不同数据类型的小于等于比较
        @sup
        @sup_complex
        def check(dtype):
            # 获取测试数据和稀疏矩阵数据
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            
            # 复制测试数据
            dat2 = dat.copy()
            # 修改复制后的数据的第一列为0
            dat2[:,0] = 0
            # 创建基于修改后的数据的稀疏矩阵
            datsp2 = self.spcreator(dat2)
            
            # 将测试数据转换为复数类型，并修改第一列为复数值
            datcomplex = dat.astype(complex)
            datcomplex[:,0] = 1 + 1j
            # 创建基于复数数据的稀疏矩阵
            datspcomplex = self.spcreator(datcomplex)
            
            # 将测试数据转换为 BSR、CSC、CSR 和 LIL 格式的稀疏矩阵
            datbsr = bsr_matrix(dat)
            datcsc = csc_matrix(dat)
            datcsr = csr_matrix(dat)
            datlil = lil_matrix(dat)

            # sparse/sparse 比较
            assert_array_equal_dtype(dat <= dat2, (datsp <= datsp2).toarray())
            assert_array_equal_dtype(datcomplex <= dat2,
                                     (datspcomplex <= datsp2).toarray())
            
            # 不同稀疏类型之间的比较
            assert_array_equal_dtype((datbsr <= datsp2).toarray(), dat <= dat2)
            assert_array_equal_dtype((datcsc <= datsp2).toarray(), dat <= dat2)
            assert_array_equal_dtype((datcsr <= datsp2).toarray(), dat <= dat2)
            assert_array_equal_dtype((datlil <= datsp2).toarray(), dat <= dat2)

            # 逆向稀疏/稀疏比较
            assert_array_equal_dtype((datsp2 <= datbsr).toarray(), dat2 <= dat)
            assert_array_equal_dtype((datsp2 <= datcsc).toarray(), dat2 <= dat)
            assert_array_equal_dtype((datsp2 <= datcsr).toarray(), dat2 <= dat)
            assert_array_equal_dtype((datsp2 <= datlil).toarray(), dat2 <= dat)
            
            # 稀疏/稠密比较
            assert_array_equal_dtype(datsp <= dat2, dat <= dat2)
            assert_array_equal_dtype(datspcomplex <= dat2, datcomplex <= dat2)
            
            # 稀疏/标量比较
            for val in [2, 1, -1, -2]:
                val = np.int64(val)  # 避免使用 Python 标量 (由于 NEP 50 的更改)
                assert_array_equal_dtype((datsp <= val).toarray(), dat <= val)
                assert_array_equal_dtype((val <= datsp).toarray(), val <= dat)

            # 重新获取数据，准备下一轮测试
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            dat2 = dat.copy()
            dat2[:,0] = 0
            datsp2 = self.spcreator(dat2)

            # 稠密右侧比较
            assert_array_equal_dtype(dat <= datsp2, datsp <= dat2)

        # 如果测试类不是 BSR、CSC 或 CSR 类型，则跳过测试
        if not isinstance(self, (TestBSR, TestCSC, TestCSR)):
            pytest.skip("Bool comparisons only implemented for BSR, CSC, and CSR.")
        
        # 遍历所选的数据类型进行测试
        for dtype in self.checked_dtypes:
            check(dtype)
    # 定义一个测试方法，用于测试大于等于比较操作
    def test_ge(self):
        # 获取警告抑制器对象
        sup = suppress_warnings()
        # 设置要过滤的警告类型为SparseEfficiencyWarning
        sup.filter(SparseEfficiencyWarning)

        # 使用警告抑制器修饰的函数，用于检查指定数据类型的操作结果
        @sup
        @sup_complex
        def check(dtype):
            # 获取密集数据和对应的稀疏数据
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]

            # 复制密集数据，并修改其第一列为0
            dat2 = dat.copy()
            dat2[:,0] = 0

            # 创建修改后的稀疏数据
            datsp2 = self.spcreator(dat2)

            # 将密集数据转换为复数类型，并修改其第一列为1 + 1j
            datcomplex = dat.astype(complex)
            datcomplex[:,0] = 1 + 1j

            # 创建修改后的复数稀疏数据
            datspcomplex = self.spcreator(datcomplex)

            # 将密集数据转换为 BSR、CSC、CSR 和 LIL 格式的稀疏矩阵
            datbsr = bsr_matrix(dat)
            datcsc = csc_matrix(dat)
            datcsr = csr_matrix(dat)
            datlil = lil_matrix(dat)

            # 对稀疏矩阵与稀疏矩阵进行大于等于比较
            assert_array_equal_dtype(dat >= dat2, (datsp >= datsp2).toarray())
            assert_array_equal_dtype(datcomplex >= dat2,
                                     (datspcomplex >= datsp2).toarray())

            # 对不同类型的稀疏矩阵与密集矩阵进行大于等于比较
            assert_array_equal_dtype((datbsr >= datsp2).toarray(), dat >= dat2)
            assert_array_equal_dtype((datcsc >= datsp2).toarray(), dat >= dat2)
            assert_array_equal_dtype((datcsr >= datsp2).toarray(), dat >= dat2)
            assert_array_equal_dtype((datlil >= datsp2).toarray(), dat >= dat2)

            # 对稀疏矩阵与稀疏矩阵进行大于等于比较（反向）
            assert_array_equal_dtype((datsp2 >= datbsr).toarray(), dat2 >= dat)
            assert_array_equal_dtype((datsp2 >= datcsc).toarray(), dat2 >= dat)
            assert_array_equal_dtype((datsp2 >= datcsr).toarray(), dat2 >= dat)
            assert_array_equal_dtype((datsp2 >= datlil).toarray(), dat2 >= dat)

            # 对稀疏矩阵与密集矩阵进行大于等于比较
            assert_array_equal_dtype(datsp >= dat2, dat >= dat2)
            assert_array_equal_dtype(datspcomplex >= dat2, datcomplex >= dat2)

            # 对稀疏矩阵与标量进行大于等于比较
            for val in [2, 1, -1, -2]:
                val = np.int64(val)  # 避免使用Python标量（根据 NEP 50 更改）
                assert_array_equal_dtype((datsp >= val).toarray(), dat >= val)
                assert_array_equal_dtype((val >= datsp).toarray(), val >= dat)

            # 重新获取密集数据和对应的稀疏数据
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]

            # 复制密集数据，并修改其第一列为0
            dat2 = dat.copy()
            dat2[:,0] = 0

            # 创建修改后的稀疏数据
            datsp2 = self.spcreator(dat2)

            # 对稀疏数据与修改后的稀疏数据进行大于等于比较
            assert_array_equal_dtype(dat >= datsp2, datsp >= dat2)

        # 如果当前测试类不是 BSR、CSC 或 CSR，则跳过该测试
        if not isinstance(self, (TestBSR, TestCSC, TestCSR)):
            pytest.skip("Bool comparisons only implemented for BSR, CSC, and CSR.")

        # 对每种被检查的数据类型执行检查函数
        for dtype in self.checked_dtypes:
            check(dtype)

    # 定义一个测试方法，用于测试创建空矩阵的操作
    def test_empty(self):
        # 断言创建的稀疏矩阵与全零矩阵相等
        assert_equal(self.spcreator((3, 3)).toarray(), zeros((3, 3)))
        
        # 断言创建的稀疏矩阵的非零元素数为0
        assert_equal(self.spcreator((3, 3)).nnz, 0)
        
        # 断言创建的稀疏矩阵的非零元素数为0
        assert_equal(self.spcreator((3, 3)).count_nonzero(), 0)
        
        # 如果稀疏数据的格式为 "coo"、"csr"、"csc" 或 "lil"，则断言沿着指定轴的非零元素数为全零数组
        if self.datsp.format in ["coo", "csr", "csc", "lil"]:
            assert_equal(self.spcreator((3, 3)).count_nonzero(axis=0), array([0, 0, 0]))
    # 测试稀疏矩阵对象是否支持非零元素计数功能，根据格式类型选择不同的轴参数
    def test_count_nonzero(self):
        # 检查稀疏矩阵对象支持的格式是否在["coo", "csr", "csc", "lil"]中
        axis_support = self.datsp.format in ["coo", "csr", "csc", "lil"]
        # 如果支持，定义不同的轴参数列表；否则只使用None作为轴参数
        axes = [None, 0, 1, -1, -2] if axis_support else [None]

        # 对于矩阵及其转置，遍历所有轴参数组合
        for A in (self.datsp, self.datsp.T):
            for ax in axes:
                # 计算预期的非零元素个数
                expected = np.count_nonzero(A.toarray(), axis=ax)
                # 使用assert_equal断言实际计数结果与预期结果相等
                assert_equal(A.count_nonzero(axis=ax), expected)

        # 如果不支持任何轴参数，测试是否抛出NotImplementedError异常
        if not axis_support:
            with assert_raises(NotImplementedError, match="not implemented .* format"):
                self.datsp.count_nonzero(axis=0)

    # 测试稀疏矩阵对象在非法形状输入时是否抛出异常
    def test_invalid_shapes(self):
        # 断言创建稀疏矩阵时传入负数形状会引发ValueError异常
        assert_raises(ValueError, self.spcreator, (-1,3))
        assert_raises(ValueError, self.spcreator, (3,-1))
        assert_raises(ValueError, self.spcreator, (-1,-1))

    # 测试稀疏矩阵对象的字符串表示是否符合预期
    def test_repr(self):
        # 创建稀疏矩阵对象datsp
        datsp = self.spcreator([[1, 0, 0], [0, 0, 0], [0, 0, -2]])
        # 根据稀疏矩阵格式选择额外信息
        extra = (
            "(1 diagonals) " if datsp.format == "dia"
            else "(blocksize=1x1) " if datsp.format == "bsr"
            else ""
        )
        # 获取格式化字符串和数据类型信息
        _, fmt = _formats[datsp.format]
        # 期望的字符串表示形式
        expected = (
            f"<{fmt} sparse matrix of dtype '{datsp.dtype}'\n"
            f"\twith {datsp.nnz} stored elements {extra}and shape {datsp.shape}>"
        )
        # 使用assert断言字符串表示符合预期
        assert repr(datsp) == expected

    # 测试稀疏矩阵对象的字符串输出是否在限制的最大行数内
    def test_str_maxprint(self):
        # 创建稀疏矩阵对象datsp
        datsp = self.spcreator(np.arange(75).reshape(5, 15))
        # 断言最大打印行数是否为50
        assert datsp.maxprint == 50
        # 断言字符串表示的行数是否符合预期（51行数据 + 3行格式化信息）

        dat = np.arange(15).reshape(5,3)
        datsp = self.spcreator(dat)
        # 根据稀疏矩阵格式选择合适的非零元素个数（dia格式为14）
        nnz_small = 14 if datsp.format == 'dia' else datsp.nnz
        # 创建最大打印行数为6的稀疏矩阵对象datsp_mp6
        datsp_mp6 = self.spcreator(dat, maxprint=6)

        # 使用assert断言字符串表示的行数是否符合预期（非零元素个数 + 3行格式化信息）
        assert len(str(datsp).split('\n')) == nnz_small + 3
        assert len(str(datsp_mp6).split('\n')) == 6 + 4

        # 检查参数`maxprint`是否只能作为关键字参数传入
        datsp = self.spcreator(dat, shape=(5, 3), dtype='i', copy=False, maxprint=4)
        datsp = self.spcreator(dat, (5, 3), 'i', False, maxprint=4)
        with pytest.raises(TypeError, match="positional argument|unpack non-iterable"):
            self.spcreator(dat, (5, 3), 'i', False, 4)

    # 测试稀疏矩阵对象的字符串输出是否符合预期
    def test_str(self):
        # 创建稀疏矩阵对象datsp
        datsp = self.spcreator([[1, 0, 0], [0, 0, 0], [0, 0, -2]])
        # 如果非零元素个数不等于2，直接返回，不进行断言测试
        if datsp.nnz != 2:
            return
        # 根据稀疏矩阵格式选择合适的额外信息
        extra = (
            "(1 diagonals) " if datsp.format == "dia"
            else "(blocksize=1x1) " if datsp.format == "bsr"
            else ""
        )
        # 获取格式化字符串和数据类型信息
        _, fmt = _formats[datsp.format]
        # 期望的字符串表示形式，包括非零元素的坐标和值
        expected = (
            f"<{fmt} sparse matrix of dtype '{datsp.dtype}'\n"
            f"\twith {datsp.nnz} stored elements {extra}and shape {datsp.shape}>"
            "\n  Coords\tValues"
            "\n  (0, 0)\t1"
            "\n  (2, 2)\t-2"
        )
        # 使用assert断言字符串表示符合预期
        assert str(datsp) == expected
    def test_empty_arithmetic(self):
        # 测试对空矩阵进行算术操作。在 SciPy SVN <= r1768 版本中会失败
        shape = (5, 5)
        for mytype in [np.dtype('int32'), np.dtype('float32'),
                np.dtype('float64'), np.dtype('complex64'),
                np.dtype('complex128')]:
            # 使用给定的形状和数据类型创建稀疏矩阵 a
            a = self.spcreator(shape, dtype=mytype)
            # 对稀疏矩阵 a 进行加法操作得到 b
            b = a + a
            # 对稀疏矩阵 a 进行数乘操作得到 c
            c = 2 * a
            # 计算稀疏矩阵 a 与其转置的乘积，结果存入 d
            d = a @ a.tocsc()
            # 计算稀疏矩阵 a 与其转置的乘积，结果存入 e
            e = a @ a.tocsr()
            # 计算稀疏矩阵 a 与其转置的乘积，结果存入 f
            f = a @ a.tocoo()
            for m in [a,b,c,d,e,f]:
                # 断言稀疏矩阵 m 转为普通数组后与 a 的乘积相等
                assert_equal(m.toarray(), a.toarray()@a.toarray())
                # 断言稀疏矩阵 m 的数据类型与 mytype 相等
                # 这些断言在所有版本 <= r1768 中均会失败：
                assert_equal(m.dtype, mytype)
                assert_equal(m.toarray().dtype, mytype)

    def test_abs(self):
        # 创建一个数组 A
        A = array([[-1, 0, 17], [0, -5, 0], [1, -4, 0], [0, 0, 0]], 'd')
        # 断言数组 A 的绝对值等于使用 self.spcreator(A) 创建的稀疏矩阵的绝对值转为数组后的值
        assert_equal(abs(A), abs(self.spcreator(A)).toarray())

    def test_round(self):
        # 指定小数位数为 1
        decimal = 1
        # 创建一个数组 A
        A = array([[-1.35, 0.56], [17.25, -5.98]], 'd')
        # 断言使用 numpy.around 函数对数组 A 进行四舍五入后的结果与使用 self.spcreator(A) 创建的稀疏矩阵进行指定小数位数四舍五入后的结果相等
        assert_equal(np.around(A, decimals=decimal),
                     round(self.spcreator(A), ndigits=decimal).toarray())

    def test_elementwise_power(self):
        # 创建一个数组 A
        A = array([[-4, -3, -2], [-1, 0, 1], [2, 3, 4]], 'd')
        # 断言使用 numpy.power 函数对数组 A 进行元素级的平方操作后的结果与使用 self.spcreator(A) 创建的稀疏矩阵进行元素级的平方操作后的结果相等
        assert_equal(np.power(A, 2), self.spcreator(A).power(2).toarray())

        # 这是元素级的幂函数，输入必须是一个标量
        assert_raises(NotImplementedError, self.spcreator(A).power, A)

    def test_neg(self):
        # 创建一个数组 A
        A = array([[-1, 0, 17], [0, -5, 0], [1, -4, 0], [0, 0, 0]], 'd')
        # 断言数组 A 的负值等于使用 self.spcreator(A) 创建的稀疏矩阵的负值转为数组后的值
        assert_equal(-A, (-self.spcreator(A)).toarray())

        # 查看 gh-5843
        # 创建一个布尔类型的数组 A
        A = array([[True, False, False], [False, False, True]])
        # 断言调用 self.spcreator(A).__neg__() 会引发 NotImplementedError 异常
        assert_raises(NotImplementedError, self.spcreator(A).__neg__)

    def test_real(self):
        # 创建一个复数数组 D
        D = array([[1 + 3j, 2 - 4j]])
        # 使用 self.spcreator(D) 创建稀疏矩阵 A
        A = self.spcreator(D)
        # 断言稀疏矩阵 A 的实部转为数组后的值与数组 D 的实部相等
        assert_equal(A.real.toarray(), D.real)

    def test_imag(self):
        # 创建一个复数数组 D
        D = array([[1 + 3j, 2 - 4j]])
        # 使用 self.spcreator(D) 创建稀疏矩阵 A
        A = self.spcreator(D)
        # 断言稀疏矩阵 A 的虚部转为数组后的值与数组 D 的虚部相等
        assert_equal(A.imag.toarray(), D.imag)
    def test_diagonal(self):
        # 测试矩阵的 .diagonal() 方法是否正常工作
        mats = []
        mats.append([[1,0,2]])  # 添加一个3x1的矩阵
        mats.append([[1],[0],[2]])  # 添加一个1x3的矩阵
        mats.append([[0,1],[0,2],[0,3]])  # 添加一个3x2的矩阵
        mats.append([[0,0,1],[0,0,2],[0,3,0]])  # 添加一个3x3的矩阵
        mats.append([[1,0],[0,0]])  # 添加一个2x2的矩阵

        mats.append(kron(mats[0],[[1,2]]))  # 将第一个矩阵与一个2x1的矩阵进行 Kronecker 乘积
        mats.append(kron(mats[0],[[1],[2]]))  # 将第一个矩阵与一个1x2的矩阵进行 Kronecker 乘积
        mats.append(kron(mats[1],[[1,2],[3,4]]))  # 将第二个矩阵与一个2x2的矩阵进行 Kronecker 乘积
        mats.append(kron(mats[2],[[1,2],[3,4]]))  # 将第三个矩阵与一个2x2的矩阵进行 Kronecker 乘积
        mats.append(kron(mats[3],[[1,2],[3,4]]))  # 将第四个矩阵与一个2x2的矩阵进行 Kronecker 乘积
        mats.append(kron(mats[3],[[1,2,3,4]]))  # 将第四个矩阵与一个1x4的矩阵进行 Kronecker 乘积

        for m in mats:
            rows, cols = array(m).shape  # 获取矩阵 m 的行数和列数
            sparse_mat = self.spcreator(m)  # 使用 spcreator 方法创建稀疏矩阵
            for k in range(-rows-1, cols+2):  # 遍历一个范围内的 k 值
                assert_equal(sparse_mat.diagonal(k=k), diag(m, k=k))  # 断言稀疏矩阵的对角线元素与 m 矩阵的对角线元素相等
            # 测试超出边界的 k 值(issue #11949)
            assert_equal(sparse_mat.diagonal(k=10), diag(m, k=10))  # 断言稀疏矩阵的对角线元素与 m 矩阵的对角线元素相等
            assert_equal(sparse_mat.diagonal(k=-99), diag(m, k=-99))  # 断言稀疏矩阵的对角线元素与 m 矩阵的对角线元素相等

        # 测试全零矩阵
        assert_equal(self.spcreator((40, 16130)).diagonal(), np.zeros(40))  # 断言稀疏矩阵的对角线元素为全零
        # 测试空矩阵
        # https://github.com/scipy/scipy/issues/11949
        assert_equal(self.spcreator((0, 0)).diagonal(), np.empty(0))  # 断言空稀疏矩阵的对角线元素为空数组
        assert_equal(self.spcreator((15, 0)).diagonal(), np.empty(0))  # 断言空稀疏矩阵的对角线元素为空数组
        assert_equal(self.spcreator((0, 5)).diagonal(10), np.empty(0))  # 断言空稀疏矩阵的对角线元素为空数组

    def test_trace(self):
        # 对于方阵
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        B = self.spcreator(A)
        for k in range(-2, 3):  # 遍历一定范围内的 k 值
            assert_equal(A.trace(offset=k), B.trace(offset=k))  # 断言矩阵 A 的迹与稀疏矩阵 B 的迹相等

        # 对于非方阵
        A = np.array([[1, 2, 3], [4, 5, 6]])
        B = self.spcreator(A)
        for k in range(-1, 3):  # 遍历一定范围内的 k 值
            assert_equal(A.trace(offset=k), B.trace(offset=k))  # 断言矩阵 A 的迹与稀疏矩阵 B 的迹相等
    # 定义名为 test_reshape 的测试方法，用于测试稀疏矩阵的重塑功能
    def test_reshape(self):
        # 以下是从 lil_matrix 重塑测试中获取的第一个示例。
        # 使用 spcreator 方法创建稀疏矩阵 x
        x = self.spcreator([[1, 0, 7], [0, 0, 0], [0, 3, 0], [0, 0, 5]])
        # 嵌套循环，测试不同的顺序和形状参数
        for order in ['C', 'F']:
            for s in [(12, 1), (1, 12)]:
                # 断言重塑后的稀疏矩阵 x 的数组表示与原始数组重塑后的数组表示相等
                assert_array_equal(x.reshape(s, order=order).toarray(),
                                   x.toarray().reshape(s, order=order))

        # 以下示例来自于 stackoverflow 上的一个回答，网址为 https://stackoverflow.com/q/16511879
        # 使用 spcreator 方法创建稀疏矩阵 x
        x = self.spcreator([[0, 10, 0, 0], [0, 0, 0, 0], [0, 20, 30, 40]])
        # 使用默认顺序 'C' 对 x 进行重塑
        y = x.reshape((2, 6))
        # 断言重塑后的稀疏矩阵 y 的数组表示与期望的数组表示相等
        desired = [[0, 10, 0, 0, 0, 0], [0, 0, 0, 20, 30, 40]]
        assert_array_equal(y.toarray(), desired)

        # 使用负索引进行重塑
        y = x.reshape((2, -1))
        assert_array_equal(y.toarray(), desired)
        y = x.reshape((-1, 6))
        assert_array_equal(y.toarray(), desired)
        # 使用两个负索引应该会引发 ValueError 异常
        assert_raises(ValueError, x.reshape, (-1, -1))

        # 使用星号参数进行重塑
        y = x.reshape(2, 6)
        assert_array_equal(y.toarray(), desired)
        # 不支持 'not_an_arg' 参数，应该引发 TypeError 异常
        assert_raises(TypeError, x.reshape, 2, 6, not_an_arg=1)

        # 如果未设置 copy=True，则同样大小的重塑操作不会创建新对象
        y = x.reshape((3, 4))
        assert_(y is x)
        # 使用 copy=True 应该创建一个新的对象
        y = x.reshape((3, 4), copy=True)
        assert_(y is not x)

        # 确保重塑操作没有改变原始稀疏矩阵的形状
        assert_array_equal(x.shape, (3, 4))

        # 就地重塑操作
        x.shape = (2, 6)
        assert_array_equal(x.toarray(), desired)

        # 尝试将稀疏矩阵重塑为不正确的维度应该会引发 ValueError 异常
        assert_raises(ValueError, x.reshape, (x.size,))
        assert_raises(ValueError, x.reshape, (1, x.size, 1))

    @pytest.mark.slow
    def test_setdiag_comprehensive(self):
        # 定义一个测试函数，用于全面测试设置对角线元素的功能

        def dense_setdiag(a, v, k):
            # 在密集矩阵 `a` 上设置对角线元素为 `v`，偏移量为 `k`
            v = np.asarray(v)
            if k >= 0:
                # 当偏移量 `k` 大于等于 0 时
                n = min(a.shape[0], a.shape[1] - k)
                if v.ndim != 0:
                    n = min(n, len(v))
                    v = v[:n]
                i = np.arange(0, n)
                j = np.arange(k, k + n)
                # 设置对角线元素
                a[i,j] = v
            elif k < 0:
                # 当偏移量 `k` 小于 0 时，递归调用以设置转置矩阵 `a.T` 的对角线元素
                dense_setdiag(a.T, v, -k)

        def check_setdiag(a, b, k):
            # 检查使用标量、长度正确的向量以及过短或过长的向量来设置对角线元素
            for r in [-1, len(np.diag(a, k)), 2, 30]:
                if r < 0:
                    v = np.random.choice(range(1, 20))
                else:
                    v = np.random.randint(1, 20, size=r)

                # 使用 `dense_setdiag` 设置 `a` 的对角线
                dense_setdiag(a, v, k)
                with suppress_warnings() as sup:
                    sup.filter(SparseEfficiencyWarning, "Changing the sparsity structu")
                    # 使用 `b.setdiag` 设置稀疏矩阵 `b` 的对角线
                    b.setdiag(v, k)

                # 检查 `dense_setdiag` 是否有效
                d = np.diag(a, k)
                if np.asarray(v).ndim == 0:
                    assert_array_equal(d, v, err_msg="%s %d" % (msg, r))
                else:
                    n = min(len(d), len(v))
                    assert_array_equal(d[:n], v[:n], err_msg="%s %d" % (msg, r))
                # 检查稀疏矩阵 `b` 是否正确设置对角线
                assert_array_equal(b.toarray(), a, err_msg="%s %d" % (msg, r))

        # comprehensive test
        # 全面测试开始
        np.random.seed(1234)
        shapes = [(0,5), (5,0), (1,5), (5,1), (5,5)]
        # 不同形状的矩阵
        for dtype in [np.int8, np.float64]:
            # 不同数据类型的测试
            for m,n in shapes:
                # 不同形状的矩阵
                ks = np.arange(-m+1, n-1)
                # 不同的偏移量 `k`
                for k in ks:
                    # 生成消息描述
                    msg = repr((dtype, m, n, k))
                    # 创建零矩阵 `a` 和稀疏矩阵 `b`
                    a = np.zeros((m, n), dtype=dtype)
                    b = self.spcreator((m, n), dtype=dtype)

                    # 检查设置对角线功能
                    check_setdiag(a, b, k)

                    # 检查覆盖等情况
                    for k2 in np.random.choice(ks, size=min(len(ks), 5)):
                        check_setdiag(a, b, k2)
    def test_setdiag(self):
        # 简单的测试用例
        m = self.spcreator(np.eye(3))  # 使用 np.eye 创建一个稀疏矩阵 m
        m2 = self.spcreator((4, 4))  # 使用 (4, 4) 创建一个空的稀疏矩阵 m2
        values = [3, 2, 1]  # 设置一个列表 values

        with suppress_warnings() as sup:  # 使用 suppress_warnings 上下文管理器来抑制警告
            sup.filter(SparseEfficiencyWarning, "Changing the sparsity structure")  # 过滤稀疏矩阵效率警告，内容为 "Changing the sparsity structure"
            
            # 断言错误：调用 m.setdiag(values, k=4) 应该引发 ValueError 异常
            assert_raises(ValueError, m.setdiag, values, k=4)
            
            # 调用 m.setdiag(values) 设置 m 的对角线为 values，然后断言 m 的对角线与 values 相等
            m.setdiag(values)
            assert_array_equal(m.diagonal(), values)
            
            # 调用 m.setdiag(values, k=1) 设置 m 的第一条对角线为 values，然后断言 m 转为数组后与给定数组相等
            m.setdiag(values, k=1)
            assert_array_equal(m.toarray(), np.array([[3, 3, 0],
                                                      [0, 2, 2],
                                                      [0, 0, 1]]))
            
            # 调用 m.setdiag(values, k=-2) 设置 m 的倒数第二条对角线为 values，然后断言 m 转为数组后与给定数组相等
            m.setdiag(values, k=-2)
            assert_array_equal(m.toarray(), np.array([[3, 3, 0],
                                                      [0, 2, 2],
                                                      [3, 0, 1]]))
            
            # 调用 m.setdiag((9,), k=2) 设置 m 的第二条超出范围的对角线为 (9,)，然后断言 m 的特定位置与给定值相等
            m.setdiag((9,), k=2)
            assert_array_equal(m.toarray()[0,2], 9)
            
            # 调用 m.setdiag((9,), k=-2) 设置 m 的倒数第二条超出范围的对角线为 (9,)，然后断言 m 的特定位置与给定值相等
            m.setdiag((9,), k=-2)
            assert_array_equal(m.toarray()[2,0], 9)
            
            # 测试空矩阵上的短值设置：调用 m2.setdiag([1], k=2)，然后断言 m2 的特定行与给定数组相等
            m2.setdiag([1], k=2)
            assert_array_equal(m2.toarray()[0], [0, 0, 1, 0])
            
            # 测试覆盖相同对角线：调用 m2.setdiag([1, 1], k=2)，然后断言 m2 的前两行与给定数组相等
            m2.setdiag([1, 1], k=2)
            assert_array_equal(m2.toarray()[:2], [[0, 0, 1, 0],
                                                  [0, 0, 0, 1]])

    def test_nonzero(self):
        A = array([[1, 0, 1],[0, 1, 1],[0, 0, 1]])  # 创建一个数组 A
        Asp = self.spcreator(A)  # 使用 self.spcreator 创建稀疏矩阵 Asp

        A_nz = {tuple(ij) for ij in transpose(A.nonzero())}  # 获取数组 A 的非零元素的位置，并将其转换为元组集合
        Asp_nz = {tuple(ij) for ij in transpose(Asp.nonzero())}  # 获取稀疏矩阵 Asp 的非零元素的位置，并将其转换为元组集合

        assert_equal(A_nz, Asp_nz)  # 断言数组 A 的非零元素位置集合与稀疏矩阵 Asp 的非零元素位置集合相等

    def test_numpy_nonzero(self):
        # 见 gh-5987
        A = array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])  # 创建一个数组 A
        Asp = self.spcreator(A)  # 使用 self.spcreator 创建稀疏矩阵 Asp

        A_nz = {tuple(ij) for ij in transpose(np.nonzero(A))}  # 获取数组 A 的非零元素的位置，并将其转换为元组集合
        Asp_nz = {tuple(ij) for ij in transpose(np.nonzero(Asp))}  # 获取稀疏矩阵 Asp 的非零元素的位置，并将其转换为元组集合

        assert_equal(A_nz, Asp_nz)  # 断言数组 A 的非零元素位置集合与稀疏矩阵 Asp 的非零元素位置集合相等

    def test_getrow(self):
        assert_array_equal(self.datsp.getrow(1).toarray(), self.dat[[1], :])  # 断言稀疏矩阵 self.datsp 的第一行与数组 self.dat 的第一行相等
        assert_array_equal(self.datsp.getrow(-1).toarray(), self.dat[[-1], :])  # 断言稀疏矩阵 self.datsp 的倒数第一行与数组 self.dat 的倒数第一行相等

    def test_getcol(self):
        assert_array_equal(self.datsp.getcol(1).toarray(), self.dat[:, [1]])  # 断言稀疏矩阵 self.datsp 的第一列与数组 self.dat 的第一列相等
        assert_array_equal(self.datsp.getcol(-1).toarray(), self.dat[:, [-1]])  # 断言稀疏矩阵 self.datsp 的倒数第一列与数组 self.dat 的倒数第一列相等
    # 定义一个测试函数，用于测试矩阵的求和操作
    def test_sum(self):
        # 设定随机种子以确保可复现的随机数生成
        np.random.seed(1234)
        # 创建一个普通的矩阵 dat_1
        dat_1 = matrix([[0, 1, 2],
                        [3, -4, 5],
                        [-6, 7, 9]])
        # 使用 NumPy 随机生成一个 5x5 的矩阵 dat_2
        dat_2 = np.random.rand(5, 5)
        # 创建一个空的 NumPy 数组 dat_3
        dat_3 = np.array([[]])
        # 创建一个全零的 40x40 NumPy 数组 dat_4
        dat_4 = np.zeros((40, 40))
        # 使用 sparse 库生成一个稀疏矩阵 dat_5，然后转换为稠密数组
        dat_5 = sparse.rand(5, 5, density=1e-2).toarray()
        # 将所有矩阵放入列表中
        matrices = [dat_1, dat_2, dat_3, dat_4, dat_5]

        # 定义一个内部函数 check，用于检查不同数据类型下的矩阵求和操作
        def check(dtype, j):
            # 使用指定数据类型创建一个矩阵 dat
            dat = matrix(matrices[j], dtype=dtype)
            # 使用 self.spcreator 方法创建一个对应的稀疏矩阵 datsp
            datsp = self.spcreator(dat, dtype=dtype)
            # 在忽略溢出错误的上下文中进行以下断言
            with np.errstate(over='ignore'):
                # 检查整体求和结果是否近似相等
                assert_array_almost_equal(dat.sum(), datsp.sum())
                # 检查整体求和结果的数据类型是否相等
                assert_equal(dat.sum().dtype, datsp.sum().dtype)
                # 断言稀疏矩阵 datsp 的整体求和结果是标量
                assert_(np.isscalar(datsp.sum(axis=None)))
                # 检查在指定轴上的求和结果是否近似相等
                assert_array_almost_equal(dat.sum(axis=None),
                                          datsp.sum(axis=None))
                # 检查在指定轴上的求和结果的数据类型是否相等
                assert_equal(dat.sum(axis=None).dtype,
                             datsp.sum(axis=None).dtype)
                # 检查在第 0 轴上的求和结果是否近似相等
                assert_array_almost_equal(dat.sum(axis=0), datsp.sum(axis=0))
                # 检查在第 0 轴上的求和结果的数据类型是否相等
                assert_equal(dat.sum(axis=0).dtype, datsp.sum(axis=0).dtype)
                # 检查在第 1 轴上的求和结果是否近似相等
                assert_array_almost_equal(dat.sum(axis=1), datsp.sum(axis=1))
                # 检查在第 1 轴上的求和结果的数据类型是否相等
                assert_equal(dat.sum(axis=1).dtype, datsp.sum(axis=1).dtype)
                # 检查在倒数第 2 轴上的求和结果是否近似相等
                assert_array_almost_equal(dat.sum(axis=-2), datsp.sum(axis=-2))
                # 检查在倒数第 2 轴上的求和结果的数据类型是否相等
                assert_equal(dat.sum(axis=-2).dtype, datsp.sum(axis=-2).dtype)
                # 检查在倒数第 1 轴上的求和结果是否近似相等
                assert_array_almost_equal(dat.sum(axis=-1), datsp.sum(axis=-1))
                # 检查在倒数第 1 轴上的求和结果的数据类型是否相等
                assert_equal(dat.sum(axis=-1).dtype, datsp.sum(axis=-1).dtype)

        # 遍历待检查的数据类型列表
        for dtype in self.checked_dtypes:
            # 遍历矩阵列表中的每个矩阵
            for j in range(len(matrices)):
                # 调用 check 函数进行检查
                check(dtype, j)

    # 定义一个测试函数，用于测试求和操作中的无效参数
    def test_sum_invalid_params(self):
        # 创建一个输出数组 out
        out = np.zeros((1, 3))
        # 创建一个普通的 NumPy 数组 dat
        dat = array([[0, 1, 2],
                     [3, -4, 5],
                     [-6, 7, 9]])
        # 使用 self.spcreator 方法创建一个对应的稀疏矩阵 datsp
        datsp = self.spcreator(dat)

        # 断言以下操作会引发 ValueError 异常：指定超出范围的轴参数
        assert_raises(ValueError, datsp.sum, axis=3)
        # 断言以下操作会引发 TypeError 异常：指定不合法的轴参数形式
        assert_raises(TypeError, datsp.sum, axis=(0, 1))
        # 断言以下操作会引发 TypeError 异常：指定不合法的轴参数形式
        assert_raises(TypeError, datsp.sum, axis=1.5)
        # 断言以下操作会引发 ValueError 异常：指定输出数组的形状与求和结果不匹配
        assert_raises(ValueError, datsp.sum, axis=1, out=out)

    # 定义一个测试函数，用于测试求和操作的数据类型
    def test_sum_dtype(self):
        # 创建一个普通的 NumPy 数组 dat
        dat = array([[0, 1, 2],
                     [3, -4, 5],
                     [-6, 7, 9]])
        # 使用 self.spcreator 方法创建一个对应的稀疏矩阵 datsp
        datsp = self.spcreator(dat)

        # 定义一个内部函数 check，用于检查不同数据类型下的平均值计算
        def check(dtype):
            # 计算普通数组 dat 的均值，并指定数据类型为 dtype
            dat_mean = dat.mean(dtype=dtype)
            # 计算稀疏矩阵 datsp 的均值，并指定数据类型为 dtype
            datsp_mean = datsp.mean(dtype=dtype)

            # 断言普通数组和稀疏矩阵在指定数据类型下的均值近似相等
            assert_array_almost_equal(dat_mean, datsp_mean)
            # 断言普通数组和稀疏矩阵在指定数据类型下的均值的数据类型相等
            assert_equal(dat_mean.dtype, datsp_mean.dtype)

        # 遍历待检查的数据类型列表
        for dtype in self.checked_dtypes:
            # 调用 check 函数进行检查
            check(dtype)
    # 定义一个测试方法，用于测试数组的求和操作，并验证结果的准确性
    def test_sum_out(self):
        # 创建一个3x3的数组dat
        dat = array([[0, 1, 2],
                     [3, -4, 5],
                     [-6, 7, 9]])
        # 使用self.spcreator方法创建一个稀疏矩阵datsp，与dat相对应
        datsp = self.spcreator(dat)

        # 创建一个形状为(1,1)的数组dat_out，用于存储dat的求和结果
        dat_out = array([[0]])
        # 创建一个形状为(1,1)的矩阵datsp_out，用于存储datsp的求和结果
        datsp_out = matrix([[0]])

        # 对dat进行求和操作，结果存储在dat_out中，保持原有的维度
        dat.sum(out=dat_out, keepdims=True)
        # 对datsp进行求和操作，结果存储在datsp_out中
        datsp.sum(out=datsp_out)
        # 断言dat_out与datsp_out的数值几乎相等
        assert_array_almost_equal(dat_out, datsp_out)

        # 重新分配dat_out和datsp_out的形状为(3,1)的全零数组和矩阵
        dat_out = np.zeros((3, 1))
        datsp_out = asmatrix(np.zeros((3, 1)))

        # 按照axis=1对dat进行求和操作，结果存储在dat_out中，保持原有的维度
        dat.sum(axis=1, out=dat_out, keepdims=True)
        # 按照axis=1对datsp进行求和操作，结果存储在datsp_out中
        datsp.sum(axis=1, out=datsp_out)
        # 断言dat_out与datsp_out的数值几乎相等
        assert_array_almost_equal(dat_out, datsp_out)

    # 定义一个测试方法，用于测试numpy的sum函数
    def test_numpy_sum(self):
        # 创建一个3x3的数组dat
        dat = array([[0, 1, 2],
                     [3, -4, 5],
                     [-6, 7, 9]])
        # 使用self.spcreator方法创建一个稀疏矩阵datsp，与dat相对应
        datsp = self.spcreator(dat)

        # 计算数组dat的总和
        dat_mean = np.sum(dat)
        # 计算稀疏矩阵datsp的总和
        datsp_mean = np.sum(datsp)

        # 断言dat_mean与datsp_mean的数值几乎相等
        assert_array_almost_equal(dat_mean, datsp_mean)
        # 断言dat_mean和datsp_mean的数据类型相等
        assert_equal(dat_mean.dtype, datsp_mean.dtype)

    # 定义一个测试方法，用于测试数组的均值操作
    def test_mean(self):
        # 定义内部函数check，用于检查指定数据类型的数组和稀疏矩阵的均值计算结果
        def check(dtype):
            # 创建一个3x3的数组dat，数据类型为dtype
            dat = array([[0, 1, 2],
                         [3, 4, 5],
                         [6, 7, 9]], dtype=dtype)
            # 使用self.spcreator方法创建一个数据类型为dtype的稀疏矩阵datsp，与dat相对应
            datsp = self.spcreator(dat, dtype=dtype)

            # 断言dat和datsp的均值几乎相等
            assert_array_almost_equal(dat.mean(), datsp.mean())
            # 断言dat和datsp的均值数据类型相等
            assert_equal(dat.mean().dtype, datsp.mean().dtype)
            # 断言datsp.mean(axis=None)返回一个标量
            assert_(np.isscalar(datsp.mean(axis=None)))
            # 断言计算dat和datsp在axis=None时的均值结果几乎相等
            assert_array_almost_equal(
                dat.mean(axis=None, keepdims=True), datsp.mean(axis=None)
            )
            # 断言dat和datsp在axis=None时的均值数据类型相等
            assert_equal(dat.mean(axis=None).dtype, datsp.mean(axis=None).dtype)
            # 断言计算dat和datsp在axis=0时的均值结果几乎相等
            assert_array_almost_equal(
                dat.mean(axis=0, keepdims=True), datsp.mean(axis=0)
            )
            # 断言dat和datsp在axis=0时的均值数据类型相等
            assert_equal(dat.mean(axis=0).dtype, datsp.mean(axis=0).dtype)
            # 断言计算dat和datsp在axis=1时的均值结果几乎相等
            assert_array_almost_equal(
                dat.mean(axis=1, keepdims=True), datsp.mean(axis=1)
            )
            # 断言dat和datsp在axis=1时的均值数据类型相等
            assert_equal(dat.mean(axis=1).dtype, datsp.mean(axis=1).dtype)
            # 断言计算dat和datsp在axis=-2时的均值结果几乎相等
            assert_array_almost_equal(
                dat.mean(axis=-2, keepdims=True), datsp.mean(axis=-2)
            )
            # 断言dat和datsp在axis=-2时的均值数据类型相等
            assert_equal(dat.mean(axis=-2).dtype, datsp.mean(axis=-2).dtype)
            # 断言计算dat和datsp在axis=-1时的均值结果几乎相等
            assert_array_almost_equal(
                dat.mean(axis=-1, keepdims=True), datsp.mean(axis=-1)
            )
            # 断言dat和datsp在axis=-1时的均值数据类型相等
            assert_equal(dat.mean(axis=-1).dtype, datsp.mean(axis=-1).dtype)

        # 遍历self.checked_dtypes中的每种数据类型，并调用check方法进行检查
        for dtype in self.checked_dtypes:
            check(dtype)

    # 定义一个测试方法，用于测试均值操作的无效参数
    def test_mean_invalid_params(self):
        # 创建一个形状为(1,3)的全零矩阵out
        out = asmatrix(np.zeros((1, 3)))
        # 创建一个3x3的数组dat
        dat = array([[0, 1, 2],
                     [3, -4, 5],
                     [-6, 7, 9]])
        # 使用self.spcreator方法创建一个稀疏矩阵datsp，与dat相对应
        datsp = self.spcreator(dat)

        # 断言调用datsp.mean(axis=3)时会抛出ValueError异常
        assert_raises(ValueError, datsp.mean, axis=3)
        # 断言调用datsp.mean(axis=(0, 1))时会抛出TypeError异常
        assert_raises(TypeError, datsp.mean, axis=(0, 1))
        # 断言调用datsp.mean(axis=1.5)时会抛出TypeError异常
        assert_raises(TypeError, datsp.mean, axis=1.5)
        # 断言调用datsp.mean(axis=1, out=out)时会抛出ValueError异常
        assert_raises(ValueError, datsp.mean, axis=1, out=out)
    # 定义一个测试方法，用于测试数组均值计算的数据类型
    def test_mean_dtype(self):
        # 创建一个二维数组 dat
        dat = array([[0, 1, 2],
                     [3, -4, 5],
                     [-6, 7, 9]])
        # 使用 self.spcreator 方法创建一个稀疏矩阵 datsp
        datsp = self.spcreator(dat)

        # 定义一个内部函数 check，用于检查给定数据类型的均值计算结果
        def check(dtype):
            # 计算原始数组 dat 的均值，指定数据类型为 dtype
            dat_mean = dat.mean(dtype=dtype)
            # 计算稀疏矩阵 datsp 的均值，指定数据类型为 dtype
            datsp_mean = datsp.mean(dtype=dtype)

            # 断言两者的均值近似相等
            assert_array_almost_equal(dat_mean, datsp_mean)
            # 断言两者的均值数据类型相同
            assert_equal(dat_mean.dtype, datsp_mean.dtype)

        # 遍历预定义的数据类型列表 self.checked_dtypes
        for dtype in self.checked_dtypes:
            # 调用 check 函数进行测试
            check(dtype)

    # 定义一个测试方法，用于测试数组均值计算的输出设置
    def test_mean_out(self):
        # 创建一个二维数组 dat
        dat = array([[0, 1, 2],
                     [3, -4, 5],
                     [-6, 7, 9]])
        # 使用 self.spcreator 方法创建一个稀疏矩阵 datsp
        datsp = self.spcreator(dat)

        # 创建一个输出数组 dat_out，用于存储计算结果
        dat_out = array([[0]])
        # 创建一个输出矩阵 datsp_out，用于存储计算结果
        datsp_out = matrix([[0]])

        # 计算原始数组 dat 的均值，并将结果存储到 dat_out 中，保持维度
        dat.mean(out=dat_out, keepdims=True)
        # 计算稀疏矩阵 datsp 的均值，并将结果存储到 datsp_out 中
        datsp.mean(out=datsp_out)
        # 断言两者计算结果近似相等
        assert_array_almost_equal(dat_out, datsp_out)

        # 重新初始化输出数组和矩阵，用于下一轮测试
        dat_out = np.zeros((3, 1))
        datsp_out = matrix(np.zeros((3, 1)))

        # 计算原始数组 dat 按行的均值，并将结果存储到 dat_out 中，保持维度
        dat.mean(axis=1, out=dat_out, keepdims=True)
        # 计算稀疏矩阵 datsp 按行的均值，并将结果存储到 datsp_out 中
        datsp.mean(axis=1, out=datsp_out)
        # 断言两者计算结果近似相等
        assert_array_almost_equal(dat_out, datsp_out)

    # 定义一个测试方法，用于测试 numpy 的均值计算
    def test_numpy_mean(self):
        # 创建一个二维数组 dat
        dat = array([[0, 1, 2],
                     [3, -4, 5],
                     [-6, 7, 9]])
        # 使用 self.spcreator 方法创建一个稀疏矩阵 datsp
        datsp = self.spcreator(dat)

        # 计算原始数组 dat 的全局均值
        dat_mean = np.mean(dat)
        # 计算稀疏矩阵 datsp 的全局均值
        datsp_mean = np.mean(datsp)

        # 断言两者的全局均值近似相等
        assert_array_almost_equal(dat_mean, datsp_mean)
        # 断言两者的全局均值数据类型相同
        assert_equal(dat_mean.dtype, datsp_mean.dtype)

    # 定义一个测试方法，用于测试指数矩阵的计算
    def test_expm(self):
        # 创建一个二维数组 M
        M = array([[1, 0, 2], [0, 0, 3], [-4, 5, 6]], float)
        # 使用 self.spcreator 方法创建一个稀疏矩阵 sM，指定形状和数据类型
        sM = self.spcreator(M, shape=(3,3), dtype=float)
        # 计算原始数组 M 的指数矩阵
        Mexp = scipy.linalg.expm(M)

        # 创建一个二维数组 N
        N = array([[3., 0., 1.], [0., 2., 0.], [0., 0., 0.]])
        # 使用 self.spcreator 方法创建一个稀疏矩阵 sN，指定形状和数据类型
        sN = self.spcreator(N, shape=(3,3), dtype=float)
        # 计算原始数组 N 的指数矩阵
        Nexp = scipy.linalg.expm(N)

        # 使用 suppress_warnings 上下文管理器，抑制稀疏矩阵计算的警告信息
        with suppress_warnings() as sup:
            # 过滤稀疏矩阵计算中的效率警告
            sup.filter(
                SparseEfficiencyWarning,
                "splu converted its input to CSC format",
            )
            sup.filter(
                SparseEfficiencyWarning,
                "spsolve is more efficient when sparse b is in the CSC matrix format",
            )
            sup.filter(
                SparseEfficiencyWarning,
                "spsolve requires A be CSC or CSR matrix format",
            )
            # 计算稀疏矩阵 sM 的指数矩阵，并转换为密集数组
            sMexp = expm(sM).toarray()
            # 计算稀疏矩阵 sN 的指数矩阵，并转换为密集数组
            sNexp = expm(sN).toarray()

        # 断言稀疏矩阵计算结果与原始数组计算结果的近似相等
        assert_array_almost_equal((sMexp - Mexp), zeros((3, 3)))
        assert_array_almost_equal((sNexp - Nexp), zeros((3, 3)))
    def test_inv(self):
        # 定义内部函数check，用于测试不同数据类型的矩阵的逆运算
        def check(dtype):
            # 创建一个二维数组M，指定数据类型dtype
            M = array([[1, 0, 2], [0, 0, 3], [-4, 5, 6]], dtype)
            # 使用suppress_warnings上下文管理器，过滤稀疏效率警告
            with suppress_warnings() as sup:
                # 过滤特定警告信息
                sup.filter(SparseEfficiencyWarning,
                           "spsolve requires A be CSC or CSR matrix format",)
                sup.filter(SparseEfficiencyWarning,
                           "spsolve is more efficient when sparse b "
                           "is in the CSC matrix format",)
                sup.filter(SparseEfficiencyWarning,
                           "splu converted its input to CSC format",)
                # 使用spcreator函数创建稀疏矩阵sM，指定形状和数据类型
                sM = self.spcreator(M, shape=(3,3), dtype=dtype)
                # 计算稀疏矩阵sM的逆矩阵sMinv
                sMinv = inv(sM)
            # 断言sMinv乘以sM的结果近似等于单位矩阵
            assert_array_almost_equal(sMinv.dot(sM).toarray(), np.eye(3))
            # 断言使用inv函数对M进行操作会引发TypeError异常
            assert_raises(TypeError, inv, M)
        # 针对浮点数类型进行check函数测试
        for dtype in [float]:
            check(dtype)

    @sup_complex
    def test_from_array(self):
        # 创建二维数组A
        A = array([[1,0,0],[2,3,4],[0,5,0],[0,0,0]])
        # 断言使用spcreator函数创建的稀疏矩阵与原始数组A的转换为稀疏矩阵后相等
        assert_array_equal(self.spcreator(A).toarray(), A)

        # 创建包含复数的二维数组A
        A = array([[1.0 + 3j, 0, 0],
                   [0, 2.0 + 5, 0],
                   [0, 0, 0]])
        # 断言使用spcreator函数创建的稀疏矩阵与原始数组A的转换为稀疏矩阵后相等
        assert_array_equal(self.spcreator(A).toarray(), A)
        # 断言使用spcreator函数创建的稀疏矩阵（指定dtype为'int16'）与A转换为int16类型后的数组相等
        assert_array_equal(self.spcreator(A, dtype='int16').toarray(),A.astype('int16'))

    @sup_complex
    def test_from_matrix(self):
        # 创建矩阵A
        A = matrix([[1, 0, 0], [2, 3, 4], [0, 5, 0], [0, 0, 0]])
        # 断言使用spcreator函数创建的稀疏矩阵与原始矩阵A的密集矩阵表示相等
        assert_array_equal(self.spcreator(A).todense(), A)

        # 创建包含复数的矩阵A
        A = matrix([[1.0 + 3j, 0, 0],
                    [0, 2.0 + 5, 0],
                    [0, 0, 0]])
        # 断言使用spcreator函数创建的稀疏矩阵与原始矩阵A的密集矩阵表示相等
        assert_array_equal(self.spcreator(A).todense(), A)
        # 断言使用spcreator函数创建的稀疏矩阵（指定dtype为'int16'）与A转换为int16类型后的密集矩阵相等
        assert_array_equal(
            self.spcreator(A, dtype='int16').todense(), A.astype('int16')
        )

    @sup_complex
    def test_from_list(self):
        # 创建二维列表A
        A = [[1,0,0],[2,3,4],[0,5,0],[0,0,0]]
        # 断言使用spcreator函数创建的稀疏矩阵与原始二维列表A的转换为稀疏矩阵后相等
        assert_array_equal(self.spcreator(A).toarray(), A)

        # 创建包含复数的二维列表A
        A = [[1.0 + 3j, 0, 0],
             [0, 2.0 + 5, 0],
             [0, 0, 0]]
        # 断言使用spcreator函数创建的稀疏矩阵与原始二维列表A的数组转换为稀疏矩阵后相等
        assert_array_equal(self.spcreator(A).toarray(), array(A))
        # 断言使用spcreator函数创建的稀疏矩阵（指定dtype为'int16'）与A转换为int16类型后的数组相等
        assert_array_equal(
            self.spcreator(A, dtype='int16').toarray(), array(A).astype('int16')
        )

    @sup_complex
    def test_from_sparse(self):
        # 创建二维数组D
        D = array([[1,0,0],[2,3,4],[0,5,0],[0,0,0]])
        # 将数组D转换为CSR格式的稀疏矩阵S
        S = csr_matrix(D)
        # 断言使用spcreator函数创建的稀疏矩阵与原始数组D的转换为稀疏矩阵后相等
        assert_array_equal(self.spcreator(S).toarray(), D)
        # 将数组D转换为稀疏矩阵S，并再次使用spcreator函数创建稀疏矩阵，断言与原始数组D的稀疏矩阵表示相等
        S = self.spcreator(D)
        assert_array_equal(self.spcreator(S).toarray(), D)

        # 创建包含复数的二维数组D
        D = array([[1.0 + 3j, 0, 0],
                   [0, 2.0 + 5, 0],
                   [0, 0, 0]])
        # 将数组D转换为CSR格式的稀疏矩阵S
        S = csr_matrix(D)
        # 断言使用spcreator函数创建的稀疏矩阵与原始数组D的转换为稀疏矩阵后相等
        assert_array_equal(self.spcreator(S).toarray(), D)
        # 断言使用spcreator函数创建的稀疏矩阵（指定dtype为'int16'）与D转换为int16类型后的数组相等
        assert_array_equal(self.spcreator(S, dtype='int16').toarray(),
                           D.astype('int16'))
        # 将数组D转换为稀疏矩阵S，并再次使用spcreator函数创建稀疏矩阵，断言与原始数组D的稀疏矩阵表示相等
        S = self.spcreator(D)
        assert_array_equal(self.spcreator(S).toarray(), D)
        # 断言使用spcreator函数创建的稀疏矩阵（指定dtype为'int16'）与D转换为int16类型后的数组相等
        assert_array_equal(self.spcreator(S, dtype='int16').toarray(),
                           D.astype('int16'))

    # def test_array(self):
    #    """test array(A) where A is in sparse format"""
    #    assert_equal( array(self.datsp), self.dat )

    # 定义测试函数，验证稀疏矩阵转换为密集矩阵的正确性
    def test_todense(self):
        # 检查 C- 或 F-连续性（默认）
        chk = self.datsp.todense()
        assert isinstance(chk, np.matrix)
        assert_array_equal(chk, self.dat)
        assert_(chk.flags.c_contiguous != chk.flags.f_contiguous)
        
        # 检查 C-连续性（使用参数指定）
        chk = self.datsp.todense(order='C')
        assert_array_equal(chk, self.dat)
        assert_(chk.flags.c_contiguous)
        assert_(not chk.flags.f_contiguous)
        
        # 检查 F-连续性（使用参数指定）
        chk = self.datsp.todense(order='F')
        assert_array_equal(chk, self.dat)
        assert_(not chk.flags.c_contiguous)
        assert_(chk.flags.f_contiguous)
        
        # 检查带有 out 参数的情况（数组形式）
        out = np.zeros(self.datsp.shape, dtype=self.datsp.dtype)
        chk = self.datsp.todense(out=out)
        assert_array_equal(self.dat, out)
        assert_array_equal(self.dat, chk)
        assert np.may_share_memory(chk, out)
        
        # 检查带有 out 参数的情况（矩阵形式）
        out = asmatrix(np.zeros(self.datsp.shape, dtype=self.datsp.dtype))
        chk = self.datsp.todense(out=out)
        assert_array_equal(self.dat, out)
        assert_array_equal(self.dat, chk)
        assert np.may_share_memory(chk, out)
        
        # 验证稀疏矩阵与密集矩阵的乘法运算
        a = array([[1.,2.,3.]])
        dense_dot_dense = a @ self.dat
        check = a @ self.datsp.todense()
        assert_array_equal(dense_dot_dense, check)
        
        b = array([[1.,2.,3.,4.]]).T
        dense_dot_dense = self.dat @ b
        check2 = self.datsp.todense() @ b
        assert_array_equal(dense_dot_dense, check2)
        
        # 检查布尔型数据是否正常工作
        spbool = self.spcreator(self.dat, dtype=bool)
        matbool = self.dat.astype(bool)
        assert_array_equal(spbool.todense(), matbool)
    # 定义一个测试方法，用于验证稀疏矩阵转换为密集数组的各种情况

    # 将self.dat转换为数组（可能是稀疏或密集的）
    dat = asarray(self.dat)

    # 将稀疏矩阵self.datsp转换为密集数组
    chk = self.datsp.toarray()

    # 断言chk与dat相等
    assert_array_equal(chk, dat)

    # 断言chk是C连续的并且不是F连续的
    assert_(chk.flags.c_contiguous != chk.flags.f_contiguous)

    # 将稀疏矩阵self.datsp以C顺序转换为密集数组
    chk = self.datsp.toarray(order='C')

    # 断言chk与dat相等
    assert_array_equal(chk, dat)

    # 断言chk是C连续的
    assert_(chk.flags.c_contiguous)

    # 断言chk不是F连续的
    assert_(not chk.flags.f_contiguous)

    # 将稀疏矩阵self.datsp以F顺序转换为密集数组
    chk = self.datsp.toarray(order='F')

    # 断言chk与dat相等
    assert_array_equal(chk, dat)

    # 断言chk不是C连续的
    assert_(not chk.flags.c_contiguous)

    # 断言chk是F连续的
    assert_(chk.flags.f_contiguous)

    # 创建一个形状与self.datsp相同、dtype与self.datsp相同的零数组
    out = np.zeros(self.datsp.shape, dtype=self.datsp.dtype)

    # 将self.datsp转换为密集数组，结果放入out中
    self.datsp.toarray(out=out)

    # 断言out与chk相等
    assert_array_equal(chk, dat)

    # 将out数组所有元素设为1
    out[...] = 1.

    # 再次将self.datsp转换为密集数组，结果放入out中
    self.datsp.toarray(out=out)

    # 断言out与chk相等
    assert_array_equal(chk, dat)

    # 创建一个密集数组a
    a = array([1.,2.,3.])

    # 计算稠密数组a与dat的点积
    dense_dot_dense = dot(a, dat)

    # 计算稠密数组a与self.datsp转换为密集数组后的点积
    check = dot(a, self.datsp.toarray())

    # 断言dense_dot_dense与check相等
    assert_array_equal(dense_dot_dense, check)

    # 创建一个密集数组b
    b = array([1.,2.,3.,4.])

    # 计算dat与密集数组b的点积
    dense_dot_dense = dot(dat, b)

    # 计算self.datsp转换为密集数组后与密集数组b的点积
    check2 = dot(self.datsp.toarray(), b)

    # 断言dense_dot_dense与check2相等
    assert_array_equal(dense_dot_dense, check2)

    # 创建一个布尔数据的稀疏矩阵
    spbool = self.spcreator(self.dat, dtype=bool)

    # 将dat转换为布尔数组
    arrbool = dat.astype(bool)

    # 断言稀疏布尔矩阵spbool转换为密集数组与数组布尔arrbool相等
    assert_array_equal(spbool.toarray(), arrbool)
    # 定义一个测试函数，用于测试数据类型转换功能
    def test_astype(self):
        # 创建一个复杂数据类型的二维数组 D
        D = array([[2.0 + 3j, 0, 0],
                   [0, 4.0 + 5j, 0],
                   [0, 0, 0]])
        # 使用 self.spcreator 方法创建一个稀疏矩阵 S，其基于数组 D
        S = self.spcreator(D)

        # 遍历支持的数据类型列表 supported_dtypes
        for x in supported_dtypes:
            # 检查正确转换后的数组 D_casted
            D_casted = D.astype(x)
            # 遍历是否复制的选项（True 和 False）
            for copy in (True, False):
                # 将稀疏矩阵 S 转换为指定数据类型 x，选项为 copy
                S_casted = S.astype(x, copy=copy)
                # 断言稀疏矩阵 S_casted 的数据类型与数组 D_casted 相同
                assert_equal(S_casted.dtype, D_casted.dtype)  # 正确的数据类型
                # 断言稀疏矩阵 S_casted 转换为数组后与 D_casted 相同
                assert_equal(S_casted.toarray(), D_casted)    # 正确的数值
                # 断言稀疏矩阵 S_casted 的格式与 S 相同
                assert_equal(S_casted.format, S.format)       # 保留格式信息
            # 检查未复制情况下的稀疏矩阵转换
            assert_(S_casted.astype(x, copy=False) is S_casted)
            # 复制稀疏矩阵 S_casted 并断言它不是同一个对象
            S_copied = S_casted.astype(x, copy=True)
            assert_(S_copied is not S_casted)

            # 定义函数检查属性在复制时不同但相等
            def check_equal_but_not_same_array_attribute(attribute):
                # 获取属性 a 和 b
                a = getattr(S_casted, attribute)
                b = getattr(S_copied, attribute)
                # 断言属性 a 和 b 相等但不是同一个对象
                assert_array_equal(a, b)
                assert_(a is not b)
                # 修改 b 的第一个元素，并确保 a 和 b 的第一个元素不相等
                i = (0,) * b.ndim
                b_i = b[i]
                b[i] = not b[i]
                assert_(a[i] != b[i])
                b[i] = b_i

            # 根据稀疏矩阵的格式，检查不同属性在复制时的行为
            if S_casted.format in ('csr', 'csc', 'bsr'):
                for attribute in ('indices', 'indptr', 'data'):
                    check_equal_but_not_same_array_attribute(attribute)
            elif S_casted.format == 'coo':
                for attribute in ('row', 'col', 'data'):
                    check_equal_but_not_same_array_attribute(attribute)
            elif S_casted.format == 'dia':
                for attribute in ('offsets', 'data'):
                    check_equal_but_not_same_array_attribute(attribute)

    # 装饰器函数，标记为复杂类型支持
    @sup_complex
    # 定义一个测试函数，用于测试不可变类型转换功能
    def test_astype_immutable(self):
        # 创建一个复杂数据类型的二维数组 D
        D = array([[2.0 + 3j, 0, 0],
                   [0, 4.0 + 5j, 0],
                   [0, 0, 0]])
        # 使用 self.spcreator 方法创建一个稀疏矩阵 S，其基于数组 D
        S = self.spcreator(D)
        # 如果稀疏矩阵 S 具有 'data' 属性，则将其设置为不可写
        if hasattr(S, 'data'):
            S.data.flags.writeable = False
        # 如果稀疏矩阵 S 的格式属于 ('csr', 'csc', 'bsr')，则将其 'indptr' 和 'indices' 属性设置为不可写
        if S.format in ('csr', 'csc', 'bsr'):
            S.indptr.flags.writeable = False
            S.indices.flags.writeable = False
        # 遍历支持的数据类型列表 supported_dtypes
        for x in supported_dtypes:
            # 将数组 D 转换为指定数据类型 x
            D_casted = D.astype(x)
            # 将稀疏矩阵 S 转换为指定数据类型 x
            S_casted = S.astype(x)
            # 断言稀疏矩阵 S_casted 的数据类型与数组 D_casted 相同
            assert_equal(S_casted.dtype, D_casted.dtype)

    # 定义一个测试函数，用于测试数据类型转换后的浮点类型转换
    def test_asfptype(self):
        # 使用 self.spcreator 方法创建一个稀疏矩阵 A，其基于 0 到 5 的整数数组，类型为 'int32'
        A = self.spcreator(arange(6,dtype='int32').reshape(2,3))

        # 断言稀疏矩阵 A 的数据类型为 'int32'
        assert_equal(A.dtype, np.dtype('int32'))
        # 断言将稀疏矩阵 A 转换为浮点数后的数据类型为 'float64'
        assert_equal(A.asfptype().dtype, np.dtype('float64'))
        # 断言稀疏矩阵 A 转换为浮点数后的格式与原始格式相同
        assert_equal(A.asfptype().format, A.format)
        # 断言将稀疏矩阵 A 转换为 'int16' 类型后再转换为浮点数的数据类型为 'float32'
        assert_equal(A.astype('int16').asfptype().dtype, np.dtype('float32'))
        # 断言将稀疏矩阵 A 转换为 'complex128' 类型后再转换为浮点数的数据类型为 'complex128'
        assert_equal(A.astype('complex128').asfptype().dtype, np.dtype('complex128'))

        # 将稀疏矩阵 A 转换为浮点数 B 和 C，断言 B 和 C 是同一个对象
        B = A.asfptype()
        C = B.asfptype()
        assert_(B is C)
    # 定义测试函数，用于测试对稀疏和密集数据进行标量乘法操作
    def test_mul_scalar(self):
        # 定义内部函数，检查给定数据类型的操作结果是否正确
        def check(dtype):
            # 获取稠密数据和对应的稀疏数据
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]

            # 断言稠密数据乘以标量2与稀疏数据乘以标量2后的稀疏表示相等
            assert_array_equal(dat*2, (datsp*2).toarray())
            # 断言稠密数据乘以标量17.3与稀疏数据乘以标量17.3后的稀疏表示相等
            assert_array_equal(dat*17.3, (datsp*17.3).toarray())

        # 遍历数学数据类型进行测试
        for dtype in self.math_dtypes:
            check(dtype)

    # 定义测试函数，用于测试标量乘法的反向操作
    def test_rmul_scalar(self):
        # 定义内部函数，检查给定数据类型的反向操作结果是否正确
        def check(dtype):
            # 获取稠密数据和对应的稀疏数据
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]

            # 断言标量2乘以稠密数据与标量2乘以稀疏数据后的稀疏表示相等
            assert_array_equal(2*dat, (2*datsp).toarray())
            # 断言标量17.3乘以稠密数据与标量17.3乘以稀疏数据后的稀疏表示相等
            assert_array_equal(17.3*dat, (17.3*datsp).toarray())

        # 遍历数学数据类型进行测试
        for dtype in self.math_dtypes:
            check(dtype)

    # github问题 #15210 的测试函数，用于测试在类型错误时的标量乘法操作
    def test_rmul_scalar_type_error(self):
        # 获取np.float64类型的稀疏数据
        datsp = self.datsp_dtypes[np.float64]
        # 使用assert_raises检查类型错误是否被引发
        with assert_raises(TypeError):
            None * datsp

    # 定义测试函数，用于测试稀疏数据的加法操作
    def test_add(self):
        # 定义内部函数，检查给定数据类型的加法操作结果是否正确
        def check(dtype):
            # 获取稠密数据和对应的稀疏数据
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]

            # 对稠密数据进行修改，并与稀疏数据进行加法运算，断言结果相等
            a = dat.copy()
            a[0,2] = 2.0
            b = datsp
            c = b + a
            assert_array_equal(c, b.toarray() + a)

            # 对稀疏数据与其CSR格式的自身进行加法运算，断言结果相等
            c = b + b.tocsr()
            assert_array_equal(c.toarray(),
                               b.toarray() + b.toarray())

            # 测试广播特性，对稀疏数据与稠密数据的部分行进行加法运算，断言结果相等
            c = b + a[0]
            assert_array_equal(c, b.toarray() + a[0])

        # 遍历数学数据类型进行测试
        for dtype in self.math_dtypes:
            check(dtype)

    # 定义测试函数，用于测试稀疏数据的反向加法操作
    def test_radd(self):
        # 定义内部函数，检查给定数据类型的反向加法操作结果是否正确
        def check(dtype):
            # 获取稠密数据和对应的稀疏数据
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]

            # 对稠密数据进行修改，并与稀疏数据进行反向加法运算，断言结果相等
            a = dat.copy()
            a[0,2] = 2.0
            b = datsp
            c = a + b
            assert_array_equal(c, a + b.toarray())

        # 遍历数学数据类型进行测试
        for dtype in self.math_dtypes:
            check(dtype)

    # 定义测试函数，用于测试稀疏数据的减法操作
    def test_sub(self):
        # 定义内部函数，检查给定数据类型的减法操作结果是否正确
        def check(dtype):
            # 获取稠密数据和对应的稀疏数据
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]

            # 断言稀疏数据与自身的减法结果为全零矩阵
            assert_array_equal((datsp - datsp).toarray(), np.zeros((3, 4)))
            # 断言稀疏数据减去0后的结果与稠密数据相等
            assert_array_equal((datsp - 0).toarray(), dat)

            # 创建一个稀疏矩阵A，并与稀疏数据进行减法运算，断言结果与稠密数据减去A后相等
            A = self.spcreator(
                np.array([[1, 0, 0, 4], [-1, 0, 0, 0], [0, 8, 0, -5]], 'd')
            )
            assert_array_equal((datsp - A).toarray(), dat - A.toarray())
            # 断言稀疏矩阵A减去稀疏数据后的结果与稠密矩阵A减去稠密数据后的结果相等
            assert_array_equal((A - datsp).toarray(), A.toarray() - dat)

            # 测试广播特性，断言稀疏数据与稠密数据的部分行进行减法运算后的结果相等
            assert_array_equal(datsp - dat[0], dat - dat[0])

        # 遍历数学数据类型进行测试，跳过布尔类型的测试（在1.9.0版本后布尔数组的减法已不建议使用）
        for dtype in self.math_dtypes:
            if dtype == np.dtype('bool'):
                continue

            check(dtype)
    def test_rsub(self):
        # 定义内部函数check，用于检查不同数据类型下的操作结果
        def check(dtype):
            # 从测试数据中获取对应数据类型的稠密矩阵和稀疏矩阵
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]

            # 断言稠密矩阵减去稀疏矩阵的结果为全零矩阵
            assert_array_equal((dat - datsp), [[0,0,0,0],[0,0,0,0],[0,0,0,0]])
            # 断言稀疏矩阵减去稠密矩阵的结果为全零矩阵
            assert_array_equal((datsp - dat), [[0,0,0,0],[0,0,0,0],[0,0,0,0]])
            # 断言0减去稀疏矩阵转为稠密矩阵的结果为负稠密矩阵
            assert_array_equal((0 - datsp).toarray(), -dat)

            # 创建一个稀疏矩阵A，进行稠密矩阵和稀疏矩阵的运算比较
            A = self.spcreator(matrix([[1,0,0,4],[-1,0,0,0],[0,8,0,-5]], 'd'))
            # 断言稠密矩阵减去稀疏矩阵A的结果与稠密矩阵减去A转为稠密矩阵的结果相等
            assert_array_equal((dat - A), dat - A.toarray())
            # 断言稀疏矩阵A减去稠密矩阵的结果与A转为稠密矩阵减去稠密矩阵的结果相等
            assert_array_equal((A - dat), A.toarray() - dat)
            # 断言稀疏矩阵A转为稠密矩阵减去稀疏矩阵datsp的结果与稠密矩阵A转为稠密矩阵减去稠密矩阵dat的结果相等
            assert_array_equal(A.toarray() - datsp, A.toarray() - dat)
            # 断言稀疏矩阵datsp减去稀疏矩阵A转为稠密矩阵的结果与稠密矩阵dat减去稀疏矩阵A转为稠密矩阵的结果相等
            assert_array_equal(datsp - A.toarray(), dat - A.toarray())

            # 测试广播操作
            assert_array_equal(dat[0] - datsp, dat[0] - dat)

        # 遍历数学数据类型进行检查
        for dtype in self.math_dtypes:
            # 跳过布尔类型，因为在1.9.0版本后布尔数组减法已被弃用
            if dtype == np.dtype('bool'):
                continue

            # 调用check函数检查当前数据类型的操作
            check(dtype)

    def test_add0(self):
        # 定义内部函数check，用于检查不同数据类型下的加零操作
        def check(dtype):
            # 从测试数据中获取对应数据类型的稠密矩阵和稀疏矩阵
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]

            # 断言将零加到稀疏矩阵的结果转为稠密矩阵等于原始稠密矩阵dat
            assert_array_equal((datsp + 0).toarray(), dat)
            # 使用sum函数对稀疏矩阵进行求和，预期结果等于稠密矩阵的和
            sumS = sum([k * datsp for k in range(1, 3)])
            sumD = sum([k * dat for k in range(1, 3)])
            assert_almost_equal(sumS.toarray(), sumD)

        # 遍历数学数据类型进行检查
        for dtype in self.math_dtypes:
            # 调用check函数检查当前数据类型的加零操作
            check(dtype)

    def test_elementwise_multiply(self):
        # 测试实数/实数的乘法
        A = array([[4,0,9],[2,-3,5]])
        B = array([[0,7,0],[0,-4,0]])
        Asp = self.spcreator(A)
        Bsp = self.spcreator(B)
        assert_almost_equal(Asp.multiply(Bsp).toarray(), A*B)  # 稀疏/稀疏
        assert_almost_equal(Asp.multiply(B).toarray(), A*B)    # 稀疏/稠密

        # 测试复数/复数的乘法
        C = array([[1-2j,0+5j,-1+0j],[4-3j,-3+6j,5]])
        D = array([[5+2j,7-3j,-2+1j],[0-1j,-4+2j,9]])
        Csp = self.spcreator(C)
        Dsp = self.spcreator(D)
        assert_almost_equal(Csp.multiply(Dsp).toarray(), C*D)  # 稀疏/稀疏
        assert_almost_equal(Csp.multiply(D).toarray(), C*D)    # 稀疏/稠密

        # 测试实数/复数的乘法
        assert_almost_equal(Asp.multiply(Dsp).toarray(), A*D)   # 稀疏/稀疏
        assert_almost_equal(Asp.multiply(D).toarray(), A*D)     # 稀疏/稠密
    def test_elementwise_multiply_broadcast(self):
        A = array([4])  # 创建包含单个元素的一维数组 A
        B = array([[-9]])  # 创建包含单个元素的二维数组 B
        C = array([1,-1,0])  # 创建一维数组 C
        D = array([[7,9,-9]])  # 创建二维数组 D
        E = array([[3],[2],[1]])  # 创建二维数组 E
        F = array([[8,6,3],[-4,3,2],[6,6,6]])  # 创建二维数组 F
        G = [1, 2, 3]  # 创建列表 G
        H = np.ones((3, 4))  # 创建形状为 (3, 4) 的全为 1 的二维数组 H
        J = H.T  # 计算 H 的转置，得到二维数组 J
        K = array([[0]])  # 创建包含单个元素的二维数组 K
        L = array([[[1,2],[0,1]]])  # 创建三维数组 L

        # Some arrays can't be cast as spmatrices (A,C,L) so leave
        # them out.
        # 创建 B 的稀疏矩阵版本 Bsp，调用 spcreator 函数
        Bsp = self.spcreator(B)
        # 创建 D 的稀疏矩阵版本 Dsp，调用 spcreator 函数
        Dsp = self.spcreator(D)
        # 创建 E 的稀疏矩阵版本 Esp，调用 spcreator 函数
        Esp = self.spcreator(E)
        # 创建 F 的稀疏矩阵版本 Fsp，调用 spcreator 函数
        Fsp = self.spcreator(F)
        # 创建 H 的稀疏矩阵版本 Hsp，调用 spcreator 函数
        Hsp = self.spcreator(H)
        # 创建 H 第一行的稀疏矩阵版本 Hspp，调用 spcreator 函数
        Hspp = self.spcreator(H[0,None])
        # 创建 J 的稀疏矩阵版本 Jsp，调用 spcreator 函数
        Jsp = self.spcreator(J)
        # 创建 J 第一列的稀疏矩阵版本 Jspp，调用 spcreator 函数
        Jspp = self.spcreator(J[:,0,None])
        # 创建 K 的稀疏矩阵版本 Ksp，调用 spcreator 函数
        Ksp = self.spcreator(K)

        matrices = [A, B, C, D, E, F, G, H, J, K, L]  # 创建数组列表 matrices
        spmatrices = [Bsp, Dsp, Esp, Fsp, Hsp, Hspp, Jsp, Jspp, Ksp]  # 创建稀疏矩阵列表 spmatrices

        # sparse/sparse
        # 对稀疏矩阵进行乘法操作
        for i in spmatrices:
            for j in spmatrices:
                try:
                    dense_mult = i.toarray() * j.toarray()  # 尝试计算稀疏矩阵 i 和 j 的密集乘积
                except ValueError:
                    assert_raises(ValueError, i.multiply, j)  # 如果出现 ValueError，则断言调用 i.multiply(j) 引发 ValueError
                    continue
                sp_mult = i.multiply(j)  # 计算稀疏矩阵 i 和 j 的逐元素乘积
                assert_almost_equal(sp_mult.toarray(), dense_mult)  # 断言稀疏乘积的数组表示接近于密集乘积

        # sparse/dense
        # 对稀疏矩阵和密集数组进行乘法操作
        for i in spmatrices:
            for j in matrices:
                try:
                    dense_mult = i.toarray() * j  # 尝试计算稀疏矩阵 i 和密集数组 j 的乘积
                except TypeError:
                    continue
                except ValueError:
                    assert_raises(ValueError, i.multiply, j)  # 如果出现 ValueError，则断言调用 i.multiply(j) 引发 ValueError
                    continue
                sp_mult = i.multiply(j)  # 计算稀疏矩阵 i 和密集数组 j 的逐元素乘积
                if issparse(sp_mult):
                    assert_almost_equal(sp_mult.toarray(), dense_mult)  # 断言稀疏乘积的数组表示接近于密集乘积
                else:
                    assert_almost_equal(sp_mult, dense_mult)  # 断言稀疏乘积与密集乘积接近
    # 定义测试元素级除法的方法
    def test_elementwise_divide(self):
        # 预期的结果矩阵，包含 NaN 和 inf
        expected = [[1,np.nan,np.nan,1],
                    [1,np.nan,1,np.nan],
                    [np.nan,1,np.nan,np.nan]]
        # 断言稀疏矩阵除法的结果与预期值相等
        assert_array_equal(toarray(self.datsp / self.datsp), expected)

        # 创建一个特定的稀疏矩阵作为除数
        denom = self.spcreator(matrix([[1,0,0,4],[-1,0,0,0],[0,8,0,-5]],'d'))
        # 预期的结果矩阵，包含 NaN、inf 和计算结果
        expected = [[1,np.nan,np.nan,0.5],
                    [-3,np.nan,inf,np.nan],
                    [np.nan,0.25,np.nan,0]]
        # 断言稀疏矩阵除法的结果与预期值相等
        assert_array_equal(toarray(self.datsp / denom), expected)

        # 复数运算
        A = array([[1-2j,0+5j,-1+0j],[4-3j,-3+6j,5]])
        B = array([[5+2j,7-3j,-2+1j],[0-1j,-4+2j,9]])
        # 将数组转换为稀疏矩阵
        Asp = self.spcreator(A)
        Bsp = self.spcreator(B)
        # 断言稀疏矩阵除法的结果与 numpy 数组除法的结果近似相等
        assert_almost_equal(toarray(Asp / Bsp), A/B)

        # 整数运算
        A = array([[1,2,3],[-3,2,1]])
        B = array([[0,1,2],[0,-2,3]])
        # 将数组转换为稀疏矩阵
        Asp = self.spcreator(A)
        Bsp = self.spcreator(B)
        # 使用 numpy.errstate 忽略除法警告，断言稀疏矩阵除法的结果与 numpy 数组除法的结果相等
        with np.errstate(divide='ignore'):
            assert_array_equal(toarray(Asp / Bsp), A / B)

        # 不匹配的稀疏矩阵模式
        A = array([[0,1],[1,0]])
        B = array([[1,0],[1,0]])
        # 将数组转换为稀疏矩阵
        Asp = self.spcreator(A)
        Bsp = self.spcreator(B)
        # 使用 numpy.errstate 忽略除法和无效值警告，断言稀疏矩阵除法的结果与 numpy 数组除法的结果相等
        with np.errstate(divide='ignore', invalid='ignore'):
            assert_array_equal(np.array(toarray(Asp / Bsp)), A / B)

    # 定义测试矩阵乘幂的方法
    def test_pow(self):
        # 创建一个数组
        A = array([[1, 0, 2, 0], [0, 3, 4, 0], [0, 5, 0, 0], [0, 6, 7, 8]])
        # 将数组转换为稀疏矩阵
        B = self.spcreator(A)

        # 遍历指数值进行测试
        for exponent in [0,1,2,3]:
            # 计算稀疏矩阵的指数幂
            ret_sp = B**exponent
            # 使用 numpy 计算数组的指数幂
            ret_np = np.linalg.matrix_power(A, exponent)
            # 断言稀疏矩阵的幂运算结果与 numpy 数组的结果相等
            assert_array_equal(ret_sp.toarray(), ret_np)
            # 断言稀疏矩阵的数据类型与 numpy 数组的数据类型相等
            assert_equal(ret_sp.dtype, ret_np.dtype)

        # 测试无效的指数值
        for exponent in [-1, 2.2, 1 + 3j]:
            # 断言对于无效的指数值，抛出 ValueError 异常
            assert_raises(ValueError, B.__pow__, exponent)

        # 测试非方阵矩阵
        B = self.spcreator(A[:3,:])
        # 断言对于非方阵矩阵，抛出 TypeError 异常
        assert_raises(TypeError, B.__pow__, 1)

    # 定义测试右乘向量的方法
    def test_rmatvec(self):
        # 创建一个稀疏矩阵
        M = self.spcreator(matrix([[3,0,0],[0,1,0],[2,0,3.0],[2,3,0]]))
        # 断言稀疏矩阵与向量的右乘结果近似相等
        assert_array_almost_equal([1,2,3,4] @ M, dot([1,2,3,4], M.toarray()))
        # 创建一个行向量
        row = array([[1,2,3,4]])
        # 断言行向量与稀疏矩阵的右乘结果近似相等
        assert_array_almost_equal(row @ M, row @ M.toarray())

    # 定义测试小规模矩阵乘法的方法
    def test_small_multiplication(self):
        # 测试对于形状为 ()、(1,)、(1,1) 和 (1,0) 的 x，A*x 的工作情况
        A = self.spcreator([[1],[2],[3]])

        # 断言稀疏矩阵与标量乘积的结果是稀疏矩阵
        assert_(issparse(A * array(1)))
        # 断言稀疏矩阵与标量乘积的结果与预期相等
        assert_equal((A * array(1)).toarray(), [[1], [2], [3]])

        # 断言稀疏矩阵与向量乘积的结果与预期相等
        assert_equal(A @ array([1]), array([1, 2, 3]))
        # 断言稀疏矩阵与二维向量乘积的结果与预期相等
        assert_equal(A @ array([[1]]), array([[1], [2], [3]]))
        # 断言稀疏矩阵与形状为 (1,1) 的全一矩阵乘积的结果与预期相等
        assert_equal(A @ np.ones((1, 1)), array([[1], [2], [3]]))
        # 断言稀疏矩阵与形状为 (1,0) 的全一矩阵乘积的结果与预期相等
        assert_equal(A @ np.ones((1, 0)), np.ones((3, 0)))
    # 测试星号在稀疏矩阵（spmatrix）和稀疏数组（sparray）上的不同行为
    def test_star_vs_at_sign_for_sparray_and_spmatrix(self):
        # 创建稀疏矩阵或稀疏数组 A，根据输入的列表创建
        A = self.spcreator([[1],[2],[3]])

        # 如果 A 是 sparray 类型，则执行以下断言
        if isinstance(A, sparray):
            # 断言 A 与一个全为1的列向量相乘的结果接近 A 自身
            assert_array_almost_equal(A * np.ones((3,1)), A)
            # 断言 A 与一个二维数组 [[1]] 相乘的结果接近 A 自身
            assert_array_almost_equal(A * array([[1]]), A)
            # 断言 A 与一个全为1的列向量相乘的结果接近 A 自身
            assert_array_almost_equal(A * np.ones((3,1)), A)
        else:
            # 断言 A 与一个数组 [1] 相乘的结果等于数组 [1, 2, 3]
            assert_equal(A * array([1]), array([1, 2, 3]))
            # 断言 A 与一个二维数组 [[1]] 相乘的结果等于二维数组 [[1], [2], [3]]
            assert_equal(A * array([[1]]), array([[1], [2], [3]]))
            # 断言 A 与一个形状为 (1, 0) 的数组相乘的结果等于一个全为1的 3x0 矩阵
            assert_equal(A * np.ones((1, 0)), np.ones((3, 0)))

    # 测试自定义类型的二元运算
    def test_binop_custom_type(self):
        # 创建稀疏矩阵或稀疏数组 A，根据输入的列表创建
        A = self.spcreator([[1], [2], [3]])
        # 创建 BinopTester 类的实例 B
        B = BinopTester()

        # 断言 A 与 B 相加的结果是"matrix on the left"
        assert_equal(A + B, "matrix on the left")
        # 断言 A 与 B 相减的结果是"matrix on the left"
        assert_equal(A - B, "matrix on the left")
        # 断言 A 与 B 相乘的结果是"matrix on the left"
        assert_equal(A * B, "matrix on the left")
        # 断言 B 与 A 相加的结果是"matrix on the right"
        assert_equal(B + A, "matrix on the right")
        # 断言 B 与 A 相减的结果是"matrix on the right"
        assert_equal(B - A, "matrix on the right")
        # 断言 B 与 A 相乘的结果是"matrix on the right"
        assert_equal(B * A, "matrix on the right")

        # 断言 A 与 B 的矩阵乘积的结果是"matrix on the left"
        assert_equal(A @ B, "matrix on the left")
        # 断言 B 与 A 的矩阵乘积的结果是"matrix on the right"
        assert_equal(B @ A, "matrix on the right")

    # 测试带形状的自定义类型的二元运算
    def test_binop_custom_type_with_shape(self):
        # 创建稀疏矩阵或稀疏数组 A，根据输入的列表创建
        A = self.spcreator([[1], [2], [3]])
        # 创建形状为 (3,1) 的 BinopTester_with_shape 实例 B
        B = BinopTester_with_shape((3,1))

        # 断言 A 与 B 相加的结果是"matrix on the left"
        assert_equal(A + B, "matrix on the left")
        # 断言 A 与 B 相减的结果是"matrix on the left"
        assert_equal(A - B, "matrix on the left")
        # 断言 A 与 B 相乘的结果是"matrix on the left"
        assert_equal(A * B, "matrix on the left")
        # 断言 B 与 A 相加的结果是"matrix on the right"
        assert_equal(B + A, "matrix on the right")
        # 断言 B 与 A 相减的结果是"matrix on the right"
        assert_equal(B - A, "matrix on the right")
        # 断言 B 与 A 相乘的结果是"matrix on the right"
        assert_equal(B * A, "matrix on the right")

        # 断言 A 与 B 的矩阵乘积的结果是"matrix on the left"
        assert_equal(A @ B, "matrix on the left")
        # 断言 B 与 A 的矩阵乘积的结果是"matrix on the right"
        assert_equal(B @ A, "matrix on the right")

    # 测试自定义类型的乘法运算
    def test_mul_custom_type(self):
        # 定义一个自定义类 Custom
        class Custom:
            def __init__(self, scalar):
                self.scalar = scalar
                
            # 定义右乘法运算符的重载方法
            def __rmul__(self, other):
                return other * self.scalar
        
        # 定义一个标量
        scalar = 2
        # 创建稀疏矩阵或稀疏数组 A，根据输入的列表创建
        A = self.spcreator([[1],[2],[3]])
        # 创建 Custom 类的实例 c
        c = Custom(scalar)
        # 计算 A 与标量 scalar 相乘的结果
        A_scalar = A * scalar
        # 计算 A 与自定义类实例 c 相乘的结果
        A_c = A * c
        # 断言 A_scalar 和 A_c 的数组元素类型相同
        assert_array_equal_dtype(A_scalar.toarray(), A_c.toarray())
        # 断言 A_scalar 和 A_c 的格式相同
        assert_equal(A_scalar.format, A_c.format)

    # 测试自定义类型的比较运算
    def test_comparisons_custom_type(self):
        # 创建稀疏矩阵或稀疏数组 A，根据输入的列表创建
        A = self.spcreator([[1], [2], [3]])
        # 创建 ComparisonTester 类的实例 B
        B = ComparisonTester()

        # 断言 A 与 B 的相等比较结果是"eq"
        assert_equal(A == B, "eq")
        # 断言 A 与 B 的不等比较结果是"ne"
        assert_equal(A != B, "ne")
        # 断言 A 大于 B 的比较结果是"lt"
        assert_equal(A > B, "lt")
        # 断言 A 大于等于 B 的比较结果是"le"
        assert_equal(A >= B, "le")
        # 断言 A 小于 B 的比较结果是"gt"
        assert_equal(A < B, "gt")
        # 断言 A 小于等于 B 的比较结果是"ge"
        assert_equal(A <= B, "ge")

    # 测试稀疏矩阵的标量乘法
    def test_dot_scalar(self):
        # 创建稀疏矩阵或稀疏数组 M，根据输入的数组创建
        M = self.spcreator(array([[3,0,0],[0,1,0],[2,0,3.0],[2,3,0]]))
        # 定义一个标量
        scalar = 10
        # 计算 M 与标量 scalar 的点乘结果
        actual = M.dot(scalar)
        # 计算 M 与标量 scalar 相乘的结果
        expected = M * scalar

        # 断言 actual 和 expected 的数组元素接近
        assert_allclose(actual.toarray(), expected.toarray())
    def test_matmul(self):
        # 创建稀疏矩阵 M，使用 self.spcreator 方法创建，传入一个数组
        M = self.spcreator(array([[3,0,0],[0,1,0],[2,0,3.0],[2,3,0]]))
        # 创建稀疏矩阵 B，使用 self.spcreator 方法创建，传入一个双精度数组
        B = self.spcreator(array([[0,1],[1,0],[0,2]],'d'))
        # 创建一个列向量 col，转置一个包含 [1,2,3] 的数组
        col = array([[1,2,3]]).T

        # 获取矩阵乘法的操作符
        matmul = operator.matmul
        # 检查矩阵-向量乘法结果的近似相等性
        assert_array_almost_equal(matmul(M, col), M.toarray() @ col)

        # 检查矩阵-矩阵乘法结果的近似相等性
        assert_array_almost_equal(matmul(M, B).toarray(), (M @ B).toarray())
        assert_array_almost_equal(matmul(M.toarray(), B), (M @ B).toarray())
        assert_array_almost_equal(matmul(M, B.toarray()), (M @ B).toarray())

        # 检查矩阵-标量乘法时是否引发 ValueError 异常
        assert_raises(ValueError, matmul, M, 1)
        assert_raises(ValueError, matmul, 1, M)

    def test_matvec(self):
        # 创建稀疏矩阵 M，使用 self.spcreator 方法创建，传入一个矩阵
        M = self.spcreator(matrix([[3,0,0],[0,1,0],[2,0,3.0],[2,3,0]]))
        # 创建一个列向量 col，转置一个包含 [1,2,3] 的数组
        col = array([[1,2,3]]).T

        # 检查矩阵乘向量的结果的近似相等性
        assert_array_almost_equal(M @ col, M.toarray() @ col)

        # 检查结果的维度是否如预期 (ticket #514)
        assert_equal((M @ array([1,2,3])).shape,(4,))
        assert_equal((M @ array([[1],[2],[3]])).shape,(4,1))
        assert_equal((M @ matrix([[1],[2],[3]])).shape,(4,1))

        # 检查结果的类型
        assert_(isinstance(M @ array([1,2,3]), ndarray))
        assert_(isinstance(M @ matrix([1,2,3]).T, np.matrix))

        # 确保对于不正确的维度会引发异常
        bad_vecs = [array([1,2]), array([1,2,3,4]), array([[1],[2]]),
                    matrix([1,2,3]), matrix([[1],[2]])]
        for x in bad_vecs:
            assert_raises(ValueError, M.__matmul__, x)

        # 稀疏矩阵乘法与数组乘法的关系如下：
        assert_array_almost_equal(M@array([1,2,3]), dot(M.toarray(),[1,2,3]))
        assert_array_almost_equal(M@[[1],[2],[3]], asmatrix(dot(M.toarray(),[1,2,3])).T)
        # 注意，如果 x 的维度是单一的，M * x 的结果将是稠密的。

        # 目前 M.matvec(asarray(col)) 是秩为 1，而 M.matvec(col) 是秩为 2。这样做是否合理？
    # 定义测试稀疏矩阵乘法的函数
    def test_matmat_sparse(self):
        # 创建一个矩阵 a，使用 matrix 函数定义
        a = matrix([[3,0,0],[0,1,0],[2,0,3.0],[2,3,0]])
        # 创建一个数组 a2，使用 array 函数定义
        a2 = array([[3,0,0],[0,1,0],[2,0,3.0],[2,3,0]])
        # 创建一个矩阵 b，指定类型为双精度浮点型
        b = matrix([[0,1],[1,0],[0,2]], 'd')
        # 使用 self.spcreator 方法创建稀疏矩阵 asp 和 bsp
        asp = self.spcreator(a)
        bsp = self.spcreator(b)
        # 断言稀疏矩阵乘积的结果与稠密矩阵乘积结果的近似相等
        assert_array_almost_equal((asp @ bsp).toarray(), a @ b)
        assert_array_almost_equal(asp @ b, a @ b)
        assert_array_almost_equal(a @ bsp, a @ b)
        assert_array_almost_equal(a2 @ bsp, a @ b)

        # 现在尝试执行不同类型之间的矩阵乘法：
        csp = bsp.tocsc()
        c = b
        want = a @ c
        # 断言稀疏矩阵乘积的结果与期望结果相等
        assert_array_almost_equal((asp @ csp).toarray(), want)
        assert_array_almost_equal(asp @ c, want)

        assert_array_almost_equal(a @ csp, want)
        assert_array_almost_equal(a2 @ csp, want)
        csp = bsp.tocsr()
        # 断言稀疏矩阵乘积的结果与期望结果相等
        assert_array_almost_equal((asp @ csp).toarray(), want)
        assert_array_almost_equal(asp @ c, want)

        assert_array_almost_equal(a @ csp, want)
        assert_array_almost_equal(a2 @ csp, want)
        csp = bsp.tocoo()
        # 断言稀疏矩阵乘积的结果与期望结果相等
        assert_array_almost_equal((asp @ csp).toarray(), want)
        assert_array_almost_equal(asp @ c, want)

        assert_array_almost_equal(a @ csp, want)
        assert_array_almost_equal(a2 @ csp, want)

        # Andy Fraser 提供的测试案例，日期为 2006-03-26
        L = 30
        frac = .3
        random.seed(0)  # 使运行结果可重复
        # 创建一个维度为 (L, 2) 的全零数组 A
        A = zeros((L,2))
        for i in range(L):
            for j in range(2):
                r = random.random()
                if r < frac:
                    A[i,j] = r/frac

        # 使用 self.spcreator 方法创建稀疏矩阵 A
        A = self.spcreator(A)
        # 计算 A 与其转置的乘积 B
        B = A @ A.T
        # 断言稀疏矩阵乘积的结果与稠密矩阵乘积结果的近似相等
        assert_array_almost_equal(B.toarray(), A.toarray() @ A.T.toarray())
        assert_array_almost_equal(B.toarray(), A.toarray() @ A.toarray().T)

        # 检查维度不匹配情况下的异常：2x2 矩阵乘以 3x2 矩阵
        A = self.spcreator([[1,2],[3,4]])
        B = self.spcreator([[1,2],[3,4],[5,6]])
        # 断言引发 ValueError 异常
        assert_raises(ValueError, A.__matmul__, B)
        if isinstance(A, sparray):
            # 如果 A 是稀疏矩阵类型，断言引发 ValueError 异常
            assert_raises(ValueError, A.__mul__, B)

    # 定义测试稠密矩阵乘法的函数
    def test_matmat_dense(self):
        # 创建一个矩阵 a，使用 matrix 函数定义
        a = matrix([[3,0,0],[0,1,0],[2,0,3.0],[2,3,0]])
        # 使用 self.spcreator 方法创建稀疏矩阵 asp
        asp = self.spcreator(a)

        # 检查数组和矩阵类型的情况
        bs = [array([[1,2],[3,4],[5,6]]), matrix([[1,2],[3,4],[5,6]])]

        for b in bs:
            # 计算稀疏矩阵 asp 与 b 的乘积结果
            result = asp @ b
            # 断言结果的类型与 b 的类型相同
            assert_(isinstance(result, type(b)))
            # 断言结果的形状为 (4,2)
            assert_equal(result.shape, (4,2))
            # 断言结果与 dot(a, b) 的结果相等
            assert_equal(result, dot(a,b))
    # 定义测试稀疏格式转换的函数
    def test_sparse_format_conversions(self):
        # 创建一个稀疏克罗内克积矩阵A
        A = sparse.kron([[1,0,2],[0,3,4],[5,0,0]], [[1,2],[0,3]])
        # 将稀疏矩阵A转换为密集数组D
        D = A.toarray()
        # 使用自定义的spcreator方法处理A
        A = self.spcreator(A)

        # 遍历不同的稀疏格式进行测试
        for format in ['bsr','coo','csc','csr','dia','dok','lil']:
            # 将A转换为当前格式，并检查格式是否正确
            a = A.asformat(format)
            assert_equal(a.format, format)
            # 检查转换后的数组是否与D相等
            assert_array_equal(a.toarray(), D)

            # 将复数版本的A转换为当前格式，并检查格式是否正确
            b = self.spcreator(D+3j).asformat(format)
            assert_equal(b.format, format)
            assert_array_equal(b.toarray(), D+3j)

            # 将A转换为当前格式的稀疏矩阵，并检查格式是否正确
            c = eval(format + '_matrix')(A)
            assert_equal(c.format, format)
            assert_array_equal(c.toarray(), D)

        # 对于稠密数组和普通数组格式，进行额外的测试
        for format in ['array', 'dense']:
            # 将A转换为当前格式，并检查数组是否相等
            a = A.asformat(format)
            assert_array_equal(a, D)

            # 将复数版本的D转换为当前格式，并检查数组是否相等
            b = self.spcreator(D+3j).asformat(format)
            assert_array_equal(b, D+3j)

    # 定义测试转换为BSR格式的函数
    def test_tobsr(self):
        # 定义两个二维数组x和y
        x = array([[1,0,2,0],[0,0,0,0],[0,0,4,5]])
        y = array([[0,1,2],[3,0,5]])
        # 计算它们的克罗内克积得到稀疏矩阵A
        A = kron(x,y)
        # 使用自定义的spcreator方法处理A
        Asp = self.spcreator(A)
        # 遍历'bsr'格式进行测试
        for format in ['bsr']:
            # 获取to_bsr方法
            fn = getattr(Asp, 'to' + format)

            # 遍历不同的块大小进行测试
            for X in [1, 2, 3, 6]:
                for Y in [1, 2, 3, 4, 6, 12]:
                    # 将Asp转换为BSR格式，并检查是否与A相等
                    assert_equal(fn(blocksize=(X, Y)).toarray(), A)

    # 定义测试转置操作的函数
    def test_transpose(self):
        # 获取self.dat和np.array([[]])两个矩阵
        dat_1 = self.dat
        dat_2 = np.array([[]])
        matrices = [dat_1, dat_2]

        # 定义检查函数，输入参数dtype和j
        def check(dtype, j):
            # 将矩阵转换为指定dtype的数组dat
            dat = array(matrices[j], dtype=dtype)
            # 使用自定义的spcreator方法处理dat
            datsp = self.spcreator(dat)

            # 对datsp进行转置操作，并检查结果是否与dat相等
            a = datsp.transpose()
            b = dat.transpose()
            assert_array_equal(a.toarray(), b)
            assert_array_equal(a.transpose().toarray(), dat)
            assert_array_equal(datsp.transpose(axes=(1, 0)).toarray(), b)
            assert_equal(a.dtype, b.dtype)

        # 进行特定测试，见gh-5987
        empty = self.spcreator((3, 4))
        assert_array_equal(np.transpose(empty).toarray(),
                           np.transpose(zeros((3, 4))))
        assert_array_equal(empty.T.toarray(), zeros((4, 3)))
        assert_raises(ValueError, empty.transpose, axes=0)

        # 对于self.checked_dtypes中的每个dtype和matrices中的每个矩阵，执行检查函数
        for dtype in self.checked_dtypes:
            for j in range(len(matrices)):
                check(dtype, j)

    # 定义测试稠密矩阵加法的函数
    def test_add_dense(self):
        # 定义检查函数，输入参数dtype
        def check(dtype):
            # 获取self.dat_dtypes和self.datsp_dtypes中的数据
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]

            # 将稠密矩阵dat与稀疏矩阵datsp相加，并检查结果是否正确
            sum1 = dat + datsp
            assert_array_equal(sum1, dat + dat)
            sum2 = datsp + dat
            assert_array_equal(sum2, dat + dat)

        # 对于self.math_dtypes中的每个dtype，执行检查函数
        for dtype in self.math_dtypes:
            check(dtype)
    def test_sub_dense(self):
        # subtracting a dense matrix to/from a sparse matrix
        # 定义一个测试函数，用于测试稠密矩阵与稀疏矩阵的减法操作

        def check(dtype):
            # 获取数据和稀疏数据
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]

            # 对于布尔型数据，行为不同
            if dat.dtype == bool:
                # 计算稠密数据与稀疏数据的差，并验证结果
                sum1 = dat - datsp
                assert_array_equal(sum1, dat - dat)
                sum2 = datsp - dat
                assert_array_equal(sum2, dat - dat)
            else:
                # 手动计算，避免标量乘法引起的类型提升
                sum1 = (dat + dat + dat) - datsp
                assert_array_equal(sum1, dat + dat)
                sum2 = (datsp + datsp + datsp) - dat
                assert_array_equal(sum2, dat + dat)

        # 遍历数学数据类型进行测试
        for dtype in self.math_dtypes:
            if dtype == np.dtype('bool'):
                # 布尔数组减法在版本1.9.0中已废弃
                continue

            check(dtype)

    def test_maximum_minimum(self):
        # 定义测试函数，测试最大最小值操作
        A_dense = np.array([[1, 0, 3], [0, 4, 5], [0, 0, 0]])
        B_dense = np.array([[1, 1, 2], [0, 3, 6], [1, -1, 0]])

        A_dense_cpx = np.array([[1, 0, 3], [0, 4+2j, 5], [0, 1j, -1j]])

        def check(dtype, dtype2, btype):
            # 如果dtype是复数类型，将稠密矩阵A_dense_cpx转换为dtype类型的稀疏矩阵A
            # 否则将稠密矩阵A_dense转换为dtype类型的稀疏矩阵A
            if np.issubdtype(dtype, np.complexfloating):
                A = self.spcreator(A_dense_cpx.astype(dtype))
            else:
                A = self.spcreator(A_dense.astype(dtype))

            # 根据btype类型，初始化B矩阵
            if btype == 'scalar':
                B = dtype2.type(1)
            elif btype == 'scalar2':
                B = dtype2.type(-1)
            elif btype == 'dense':
                B = B_dense.astype(dtype2)
            elif btype == 'sparse':
                B = self.spcreator(B_dense.astype(dtype2))
            else:
                raise ValueError()

            # 忽略警告，捕获稀疏效率警告
            with suppress_warnings() as sup:
                sup.filter(SparseEfficiencyWarning,
                           "Taking maximum .minimum. with > 0 .< 0. number "
                           "results to a dense matrix")

                # 计算稀疏矩阵A与B的最大值和最小值
                max_s = A.maximum(B)
                min_s = A.minimum(B)

            # 将稀疏矩阵A与B转换为稠密矩阵的最大值和最小值
            max_d = np.maximum(toarray(A), toarray(B))
            assert_array_equal(toarray(max_s), max_d)
            assert_equal(max_s.dtype, max_d.dtype)

            min_d = np.minimum(toarray(A), toarray(B))
            assert_array_equal(toarray(min_s), min_d)
            assert_equal(min_s.dtype, min_d.dtype)

        # 遍历数学数据类型和数据类型2进行测试
        for dtype in self.math_dtypes:
            for dtype2 in [np.int8, np.float64, np.complex128]:
                for btype in ['scalar', 'scalar2', 'dense', 'sparse']:
                    check(np.dtype(dtype), np.dtype(dtype2), btype)
    # 定义一个测试方法，用于测试复制功能
    def test_copy(self):
        # 检查 copy=True 和 copy=False 参数的工作方式
        A = self.datsp

        # 检查复制是否保留了格式
        assert_equal(A.copy().format, A.format)
        assert_equal(A.__class__(A, copy=True).format, A.format)
        assert_equal(A.__class__(A, copy=False).format, A.format)

        # 检查复制后的稀疏矩阵是否与原始矩阵的数组表示相同
        assert_equal(A.copy().toarray(), A.toarray())
        assert_equal(A.__class__(A, copy=True).toarray(), A.toarray())
        assert_equal(A.__class__(A, copy=False).toarray(), A.toarray())

        # 检查 XXX_matrix.toXXX() 方法是否正常工作
        toself = getattr(A, 'to' + A.format)
        assert_(toself() is A)
        assert_(toself(copy=False) is A)
        assert_equal(toself(copy=True).format, A.format)
        assert_equal(toself(copy=True).toarray(), A.toarray())

        # 检查数据是否被复制了？
        assert_(not sparse_may_share_memory(A.copy(), A))

    # 测试 __iter__ 方法是否与 NumPy 矩阵兼容
    def test_iterator(self):
        B = matrix(np.arange(50).reshape(5, 10))
        A = self.spcreator(B)

        # 迭代测试，比较稀疏矩阵 A 和密集矩阵 B 的每一行是否相等
        for x, y in zip(A, B):
            assert_equal(x.toarray(), y)
    def test_size_zero_matrix_arithmetic(self):
        # 测试对空矩阵进行基本的矩阵运算，如形状为 (0,0), (10,0), (0,3) 等。
        mat = array([])  # 创建一个空的 NumPy 数组
        a = mat.reshape((0, 0))  # 将数组重新形状为 (0,0) 的矩阵
        b = mat.reshape((0, 1))  # 将数组重新形状为 (0,1) 的矩阵
        c = mat.reshape((0, 5))  # 将数组重新形状为 (0,5) 的矩阵
        d = mat.reshape((1, 0))  # 将数组重新形状为 (1,0) 的矩阵
        e = mat.reshape((5, 0))  # 将数组重新形状为 (5,0) 的矩阵
        f = np.ones([5, 5])  # 创建一个 5x5 全为 1 的 NumPy 数组

        asp = self.spcreator(a)  # 使用 spcreator 方法创建稀疏矩阵对象 asp
        bsp = self.spcreator(b)  # 使用 spcreator 方法创建稀疏矩阵对象 bsp
        csp = self.spcreator(c)  # 使用 spcreator 方法创建稀疏矩阵对象 csp
        dsp = self.spcreator(d)  # 使用 spcreator 方法创建稀疏矩阵对象 dsp
        esp = self.spcreator(e)  # 使用 spcreator 方法创建稀疏矩阵对象 esp
        fsp = self.spcreator(f)  # 使用 spcreator 方法创建稀疏矩阵对象 fsp

        # 矩阵乘法。
        assert_array_equal(asp.dot(asp).toarray(), np.dot(a, a))
        assert_array_equal(bsp.dot(dsp).toarray(), np.dot(b, d))
        assert_array_equal(dsp.dot(bsp).toarray(), np.dot(d, b))
        assert_array_equal(csp.dot(esp).toarray(), np.dot(c, e))
        assert_array_equal(csp.dot(fsp).toarray(), np.dot(c, f))
        assert_array_equal(esp.dot(csp).toarray(), np.dot(e, c))
        assert_array_equal(dsp.dot(csp).toarray(), np.dot(d, c))
        assert_array_equal(fsp.dot(esp).toarray(), np.dot(f, e))

        # 错误的矩阵乘法
        assert_raises(ValueError, dsp.dot, e)
        assert_raises(ValueError, asp.dot, d)

        # 逐元素乘法
        assert_array_equal(asp.multiply(asp).toarray(), np.multiply(a, a))
        assert_array_equal(bsp.multiply(bsp).toarray(), np.multiply(b, b))
        assert_array_equal(dsp.multiply(dsp).toarray(), np.multiply(d, d))

        assert_array_equal(asp.multiply(a).toarray(), np.multiply(a, a))
        assert_array_equal(bsp.multiply(b).toarray(), np.multiply(b, b))
        assert_array_equal(dsp.multiply(d).toarray(), np.multiply(d, d))

        assert_array_equal(asp.multiply(6).toarray(), np.multiply(a, 6))
        assert_array_equal(bsp.multiply(6).toarray(), np.multiply(b, 6))
        assert_array_equal(dsp.multiply(6).toarray(), np.multiply(d, 6))

        # 错误的逐元素乘法
        assert_raises(ValueError, asp.multiply, c)
        assert_raises(ValueError, esp.multiply, c)

        # 加法
        assert_array_equal(asp.__add__(asp).toarray(), a.__add__(a))
        assert_array_equal(bsp.__add__(bsp).toarray(), b.__add__(b))
        assert_array_equal(dsp.__add__(dsp).toarray(), d.__add__(d))

        # 错误的加法
        assert_raises(ValueError, asp.__add__, dsp)
        assert_raises(ValueError, bsp.__add__, asp)

    def test_size_zero_conversions(self):
        mat = array([])
        a = mat.reshape((0, 0))
        b = mat.reshape((0, 5))
        c = mat.reshape((5, 0))

        for m in [a, b, c]:
            spm = self.spcreator(m)
            assert_array_equal(spm.tocoo().toarray(), m)
            assert_array_equal(spm.tocsr().toarray(), m)
            assert_array_equal(spm.tocsc().toarray(), m)
            assert_array_equal(spm.tolil().toarray(), m)
            assert_array_equal(spm.todok().toarray(), m)
            assert_array_equal(spm.tobsr().toarray(), m)
    # 定义一个测试方法，用于测试 pickle 序列化和反序列化功能
    def test_pickle(self):
        # 导入 pickle 库
        import pickle
        # 创建一个警告抑制对象
        sup = suppress_warnings()
        # 添加一个警告过滤器，抑制稀疏矩阵效率警告
        sup.filter(SparseEfficiencyWarning)

        # 使用警告抑制对象装饰的函数，用于检查 pickle 操作
        @sup
        def check():
            # 复制当前测试环境的稀疏矩阵数据
            datsp = self.datsp.copy()
            # 对每个 pickle 协议版本进行测试
            for protocol in range(pickle.HIGHEST_PROTOCOL):
                # 使用 pickle 序列化和反序列化数据，并加载为 sploaded 对象
                sploaded = pickle.loads(pickle.dumps(datsp, protocol=protocol))
                # 断言原始数据与反序列化后的数据的形状相同
                assert_equal(datsp.shape, sploaded.shape)
                # 断言原始数据的稀疏数组内容与反序列化后的稀疏数组内容相同
                assert_array_equal(datsp.toarray(), sploaded.toarray())
                # 断言原始数据的格式与反序列化后的格式相同
                assert_equal(datsp.format, sploaded.format)

                # Hacky check for class member equality. This assumes that
                # all instance variables are one of:
                #  1. Plain numpy ndarrays
                #  2. Tuples of ndarrays
                #  3. Types that support equality comparison with ==
                # 针对类成员的相等性进行检查。这假设所有实例变量为以下类型之一：
                #  1. 普通的 numpy 数组
                #  2. 数组元组
                #  3. 支持使用 == 进行相等性比较的类型
                for key, val in datsp.__dict__.items():
                    if isinstance(val, np.ndarray):
                        # 如果成员是 numpy 数组，则断言其在 sploaded 对象中的对应成员相等
                        assert_array_equal(val, sploaded.__dict__[key])
                    elif (isinstance(val, tuple) and val
                          and isinstance(val[0], np.ndarray)):
                        # 如果成员是数组元组，则断言其在 sploaded 对象中的对应成员相等
                        assert_array_equal(val, sploaded.__dict__[key])
                    else:
                        # 否则，直接比较成员的值是否相等
                        assert_(val == sploaded.__dict__[key])

        # 执行检查函数
        check()

    # 定义一个测试方法，用于测试一元通用函数的覆盖行为
    def test_unary_ufunc_overrides(self):
        # 定义一个内部函数，用于检查特定的一元通用函数
        def check(name):
            # 如果函数名为 "sign"，则跳过该测试，因为它与 Numpy 的比较操作有冲突
            if name == "sign":
                pytest.skip("sign conflicts with comparison op support on Numpy")
            # 如果当前的稀疏矩阵创建函数是 dok_matrix 或 lil_matrix，则跳过测试，
            # 因为它们未实现一元操作
            if self.spcreator in (dok_matrix, lil_matrix):
                pytest.skip("Unary ops not implemented for dok/lil")
            # 获取 numpy 中对应名称的一元通用函数对象
            ufunc = getattr(np, name)

            # 创建一个稀疏矩阵 X，包含从 0 到 19 的数列，形状为 (4, 5)，并将其元素除以 20
            X = self.spcreator(np.arange(20).reshape(4, 5) / 20.)
            # 使用 ufunc 对 X 的密集数组部分进行一元操作，得到 X0
            X0 = ufunc(X.toarray())

            # 使用 ufunc 对稀疏矩阵 X 进行一元操作，得到 X2
            X2 = ufunc(X)
            # 断言 X2 的密集数组部分与 X0 相等
            assert_array_equal(X2.toarray(), X0)

        # 遍历指定的一元通用函数名称列表，逐个进行检查
        for name in ["sin", "tan", "arcsin", "arctan", "sinh", "tanh",
                     "arcsinh", "arctanh", "rint", "sign", "expm1", "log1p",
                     "deg2rad", "rad2deg", "floor", "ceil", "trunc", "sqrt",
                     "abs"]:
            # 调用 check 函数，检查当前名称对应的一元通用函数
            check(name)
    # 定义测试方法 test_resize，用于测试稀疏矩阵的 resize 方法
    def test_resize(self):
        # 创建一个 NumPy 数组 D，表示初始的稀疏矩阵
        D = np.array([[1, 0, 3, 4],
                      [2, 0, 0, 0],
                      [3, 0, 0, 0]])
        # 使用 self.spcreator(D) 创建稀疏矩阵 S
        S = self.spcreator(D)
        # 调用 S.resize((3, 2)) 进行矩阵 resize 操作，期望返回 None
        assert_(S.resize((3, 2)) is None)
        # 验证 resize 后的稀疏矩阵 S 是否与预期的稀疏矩阵形状一致
        assert_array_equal(S.toarray(), [[1, 0],
                                         [2, 0],
                                         [3, 0]])
        # 再次调用 S.resize((2, 2)) 进行 resize 操作
        S.resize((2, 2))
        # 验证 resize 后的稀疏矩阵 S 是否与预期的稀疏矩阵形状一致
        assert_array_equal(S.toarray(), [[1, 0],
                                         [2, 0]])
        # 继续调用 S.resize((3, 2)) 进行 resize 操作
        S.resize((3, 2))
        # 验证 resize 后的稀疏矩阵 S 是否与预期的稀疏矩阵形状一致
        assert_array_equal(S.toarray(), [[1, 0],
                                         [2, 0],
                                         [0, 0]])
        # 再次调用 S.resize((3, 3)) 进行 resize 操作
        S.resize((3, 3))
        # 验证 resize 后的稀疏矩阵 S 是否与预期的稀疏矩阵形状一致
        assert_array_equal(S.toarray(), [[1, 0, 0],
                                         [2, 0, 0],
                                         [0, 0, 0]])
        # 测试无操作情况，调用 S.resize((3, 3))，即使形状相同，也应该不改变矩阵
        S.resize((3, 3))
        # 验证 resize 后的稀疏矩阵 S 是否与预期的稀疏矩阵形状一致
        assert_array_equal(S.toarray(), [[1, 0, 0],
                                         [2, 0, 0],
                                         [0, 0, 0]])

        # 测试 *args 形式的 resize 方法调用，使用 S.resize(3, 2) 进行 resize 操作
        S.resize(3, 2)
        # 验证 resize 后的稀疏矩阵 S 是否与预期的稀疏矩阵形状一致
        assert_array_equal(S.toarray(), [[1, 0],
                                         [2, 0],
                                         [0, 0]])

        # 遍历测试不合法的形状参数，应该抛出 ValueError 异常
        for bad_shape in [1, (-1, 2), (2, -1), (1, 2, 3)]:
            assert_raises(ValueError, S.resize, bad_shape)

    # 定义测试方法 test_constructor1_base，用于测试稀疏矩阵的构造方法
    def test_constructor1_base(self):
        # 获取测试数据 self.datsp
        A = self.datsp

        # 获取 A 的格式信息
        self_format = A.format

        # 测试使用 copy=False 的构造方法
        C = A.__class__(A, copy=False)
        # 验证构造出的稀疏矩阵 C 的数组形式是否与 A 的数组形式相同
        assert_array_equal_dtype(A.toarray(), C.toarray())
        # 如果 A 的格式不在 NON_ARRAY_BACKED_FORMATS 中，则验证是否共享内存
        if self_format not in NON_ARRAY_BACKED_FORMATS:
            assert_(sparse_may_share_memory(A, C))

        # 测试使用 dtype=A.dtype, copy=False 的构造方法
        C = A.__class__(A, dtype=A.dtype, copy=False)
        # 验证构造出的稀疏矩阵 C 的数组形式是否与 A 的数组形式相同
        assert_array_equal_dtype(A.toarray(), C.toarray())
        # 如果 A 的格式不在 NON_ARRAY_BACKED_FORMATS 中，则验证是否共享内存
        if self_format not in NON_ARRAY_BACKED_FORMATS:
            assert_(sparse_may_share_memory(A, C))

        # 测试使用 dtype=np.float32, copy=False 的构造方法
        C = A.__class__(A, dtype=np.float32, copy=False)
        # 验证构造出的稀疏矩阵 C 的数组形式是否与 A 的数组形式相同
        assert_array_equal(A.toarray(), C.toarray())

        # 测试使用 copy=True 的构造方法
        C = A.__class__(A, copy=True)
        # 验证构造出的稀疏矩阵 C 的数组形式是否与 A 的数组形式相同
        assert_array_equal_dtype(A.toarray(), C.toarray())
        # 验证构造出的稀疏矩阵 C 是否与 A 不共享内存
        assert_(not sparse_may_share_memory(A, C))

        # 遍历测试使用不同格式 other_format 构造方法的情况
        for other_format in ['csr', 'csc', 'coo', 'dia', 'dok', 'lil']:
            # 如果 other_format 与 self_format 相同，则跳过
            if other_format == self_format:
                continue
            # 将 A 转换成 other_format 格式的稀疏矩阵 B
            B = A.asformat(other_format)
            # 测试使用 copy=False 的构造方法
            C = A.__class__(B, copy=False)
            # 验证构造出的稀疏矩阵 C 的数组形式是否与 A 的数组形式相同
            assert_array_equal_dtype(A.toarray(), C.toarray())

            # 测试使用 copy=True 的构造方法
            C = A.__class__(B, copy=True)
            # 验证构造出的稀疏矩阵 C 的数组形式是否与 A 的数组形式相同
            assert_array_equal_dtype(A.toarray(), C.toarray())
            # 验证构造出的稀疏矩阵 C 是否与 B 不共享内存
            assert_(not sparse_may_share_memory(B, C))
# 定义一个名为 _TestInplaceArithmetic 的测试类，用于测试就地运算的功能
class _TestInplaceArithmetic:
    
    # 定义一个测试方法 test_inplace_dense，用于测试稠密数组的就地运算
    def test_inplace_dense(self):
        # 创建一个3行4列的全1数组 a
        a = np.ones((3, 4))
        # 调用 self.spcreator 方法创建一个稀疏数组 b
        b = self.spcreator(a)

        # 复制数组 a 到 x 和 y
        x = a.copy()
        y = a.copy()
        # 就地加法操作：x = x + a
        x += a
        # 就地加法操作：y = y + b
        y += b
        # 断言数组 x 和 y 是否相等
        assert_array_equal(x, y)

        # 复制数组 a 到 x 和 y
        x = a.copy()
        y = a.copy()
        # 就地减法操作：x = x - a
        x -= a
        # 就地减法操作：y = y - b
        y -= b
        # 断言数组 x 和 y 是否相等
        assert_array_equal(x, y)

        # 如果 b 是 sparray 类型的实例
        if isinstance(b, sparray):
            # 对应元素相乘操作，使用 __rmul__ 方法
            x = a.copy()
            y = a.copy()
            # 断言引发 ValueError 异常，异常消息包含 "dimension mismatch"
            with assert_raises(ValueError, match="dimension mismatch"):
                x *= b.T
            # 普通乘法操作：x = x * a
            x = x * a
            # 就地乘法操作：y *= b
            y *= b
            # 断言数组 x 和 y 是否相等
            assert_array_equal(x, y)
        else:
            # 矩阵乘法操作，使用 __rmul__ 方法
            x = a.copy()
            y = a.copy()
            # 断言引发 ValueError 异常，异常消息包含 "dimension mismatch"
            with assert_raises(ValueError, match="dimension mismatch"):
                x *= b
            # 矩阵乘法操作：x = x.dot(a.T)
            x = x.dot(a.T)
            # 就地乘法操作：y *= b.T
            y *= b.T
            # 断言数组 x 和 y 是否相等
            assert_array_equal(x, y)

        # 矩阵乘法操作，使用 __rmatmul__ 方法
        y = a.copy()
        # 如果 numpy 不支持 __imatmul__ 方法，则跳过此测试
        # 一旦 numpy 1.24 不再支持，就不需要在 try/except 中使用
        try:
            y @= b.T
        except TypeError:
            pass
        else:
            x = a.copy()
            y = a.copy()
            # 断言引发 ValueError 异常，异常消息包含 "dimension mismatch"
            with assert_raises(ValueError, match="dimension mismatch"):
                x @= b
            # 矩阵乘法操作：x = x.dot(a.T)
            x = x.dot(a.T)
            # 就地矩阵乘法操作：y @= b.T
            y @= b.T
            # 断言数组 x 和 y 是否相等
            assert_array_equal(x, y)

        # 整数地板除法操作，由于不支持，预期引发 TypeError 异常
        with assert_raises(TypeError, match="unsupported operand"):
            x //= b

    # 定义一个测试方法 test_imul_scalar，用于测试标量乘法的就地运算
    def test_imul_scalar(self):
        # 定义一个嵌套函数 check，用于检查不同数据类型的乘法操作
        def check(dtype):
            # 获取特定数据类型的稠密数组 dat 和稀疏数组 datsp
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]

            # 避免隐式类型转换
            if np.can_cast(int, dtype, casting='same_kind'):
                a = datsp.copy()
                # 就地乘法操作：a *= 2
                a *= 2
                b = dat.copy()
                # 普通乘法操作：b = b * 2
                b *= 2
                # 断言数组 b 和 a 转换为稠密数组后是否相等
                assert_array_equal(b, a.toarray())

            if np.can_cast(float, dtype, casting='same_kind'):
                a = datsp.copy()
                # 就地乘法操作：a *= 17.3
                a *= 17.3
                b = dat.copy()
                # 普通乘法操作：b = b * 17.3
                b *= 17.3
                # 断言数组 b 和 a 转换为稠密数组后是否相等
                assert_array_equal(b, a.toarray())

        # 遍历 self.math_dtypes 中的数据类型，分别调用 check 函数进行测试
        for dtype in self.math_dtypes:
            check(dtype)
    # 定义测试方法，用于测试整数除法运算
    def test_idiv_scalar(self):
        # 定义内部函数 check，接受数据类型 dtype 作为参数
        def check(dtype):
            # dat 和 datsp 是从 self.dat_dtypes 和 self.datsp_dtypes 中获取的数据
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]

            # 如果可以将 int 类型转换为 dtype，使用稳定类型转换
            if np.can_cast(int, dtype, casting='same_kind'):
                # 复制 datsp 并进行除以 2 的操作
                a = datsp.copy()
                a /= 2
                # 复制 dat 并进行除以 2 的操作
                b = dat.copy()
                b /= 2
                # 断言数组 b 与稀疏数组 a 的稠密表示相等
                assert_array_equal(b, a.toarray())

            # 如果可以将 float 类型转换为 dtype，同样使用稳定类型转换
            if np.can_cast(float, dtype, casting='same_kind'):
                # 复制 datsp 并进行除以 17.3 的操作
                a = datsp.copy()
                a /= 17.3
                # 复制 dat 并进行除以 17.3 的操作
                b = dat.copy()
                b /= 17.3
                # 断言数组 b 与稀疏数组 a 的稠密表示相等
                assert_array_equal(b, a.toarray())

        # 遍历 self.math_dtypes 中的数据类型
        for dtype in self.math_dtypes:
            # 只有当不能将 dtype 转换为 int 类型时才调用 check 函数
            if not np.can_cast(dtype, np.dtype(int)):
                check(dtype)

    # 定义测试方法，测试就地操作的成功性
    def test_inplace_success(self):
        # 就地操作应该正常工作，即使没有实现专门的版本，应该退回到 x = x <op> y 的操作
        # 创建 np.eye(5) 的稀疏矩阵和对应的稀疏矩阵创建器
        a = self.spcreator(np.eye(5))
        b = self.spcreator(np.eye(5))
        bp = self.spcreator(np.eye(5))

        # b += a，就地加法操作
        b += a
        # bp = bp + a，常规加法操作
        bp = bp + a
        # 断言数组 b 和 bp 的稠密表示相等
        assert_allclose(b.toarray(), bp.toarray())

        # 如果 b 是 sparray 类型
        if isinstance(b, sparray):
            # b *= a，就地乘法操作
            b *= a
            # bp = bp * a，常规乘法操作
            bp = bp * a
            # 断言数组 b 和 bp 的稠密表示相等
            assert_allclose(b.toarray(), bp.toarray())

        # b @= a，就地矩阵乘法操作
        b @= a
        # bp = bp @ a，常规矩阵乘法操作
        bp = bp @ a
        # 断言数组 b 和 bp 的稠密表示相等
        assert_allclose(b.toarray(), bp.toarray())

        # b -= a，就地减法操作
        b -= a
        # bp = bp - a，常规减法操作
        bp = bp - a
        # 断言数组 b 和 bp 的稠密表示相等
        assert_allclose(b.toarray(), bp.toarray())

        # 使用 assert_raises 检查 TypeError 异常是否被抛出，匹配 "unsupported operand" 字符串
        with assert_raises(TypeError, match="unsupported operand"):
            # a //= b，整数除法操作，应该抛出异常
            a //= b
# 定义一个名为 _TestGetSet 的测试类
class _TestGetSet:
    
    # 定义一个测试方法 test_getelement，用于测试获取元素的功能
    def test_getelement(self):
        
        # 定义一个内部函数 check，用于具体检查不同数据类型的数组操作
        def check(dtype):
            # 创建一个二维数组 D，指定数据类型为 dtype
            D = array([[1,0,0],
                       [4,3,0],
                       [0,2,0],
                       [0,0,0]], dtype=dtype)
            
            # 通过 self.spcreator 方法创建一个稀疏矩阵 A，与数组 D 一致
            A = self.spcreator(D)

            # 获取数组 D 的行数 M 和列数 N
            M,N = D.shape

            # 循环遍历数组 D 的所有索引
            for i in range(-M, M):
                for j in range(-N, N):
                    # 断言稀疏矩阵 A 的元素 A[i,j] 等于数组 D 的对应元素 D[i,j]
                    assert_equal(A[i,j], D[i,j])

            # 断言稀疏矩阵 A[1,1] 的类型为 dtype
            assert_equal(type(A[1,1]), dtype)

            # 针对不合法索引情况，逐一断言会触发 IndexError 或 TypeError 异常
            for ij in [(0,3),(-1,3),(4,0),(4,3),(4,-1), (1, 2, 3)]:
                assert_raises((IndexError, TypeError), A.__getitem__, ij)

        # 遍历所有支持的数据类型，对每种类型调用 check 函数进行测试
        for dtype in supported_dtypes:
            check(np.dtype(dtype))

    # 定义一个测试方法 test_setelement，用于测试设置元素的功能
    def test_setelement(self):
        
        # 定义一个内部函数 check，用于具体检查不同数据类型的数组设置操作
        def check(dtype):
            # 通过 self.spcreator 方法创建一个形状为 (3,4) 的稀疏矩阵 A，指定数据类型为 dtype
            A = self.spcreator((3,4), dtype=dtype)
            
            # 使用 suppress_warnings 上下文管理器，过滤 SparseEfficiencyWarning 警告信息
            with suppress_warnings() as sup:
                sup.filter(SparseEfficiencyWarning, "Changing the sparsity structure")
                
                # 分别设置稀疏矩阵 A 中的几个元素
                A[0, 0] = dtype.type(0)  # bug 870
                A[1, 2] = dtype.type(4.0)
                A[0, 1] = dtype.type(3)
                A[2, 0] = dtype.type(2.0)
                A[0,-1] = dtype.type(8)
                A[-1,-2] = dtype.type(7)
                A[0, 1] = dtype.type(5)

            # 如果数据类型不是布尔类型，断言稀疏矩阵 A 转换为密集数组后的结果
            if dtype != np.bool_:
                assert_array_equal(
                    A.toarray(),
                    [
                        [0, 5, 0, 8],
                        [0, 0, 4, 0],
                        [2, 0, 7, 0]
                    ]
                )

            # 针对不合法索引情况，逐一断言会触发 IndexError 异常
            for ij in [(0,4),(-1,4),(3,0),(3,4),(3,-1)]:
                assert_raises(IndexError, A.__setitem__, ij, 123.0)

            # 针对设置不合法值的情况，逐一断言会触发 ValueError 异常
            for v in [[1,2,3], array([1,2,3])]:
                assert_raises(ValueError, A.__setitem__, (0,0), v)

            # 如果数据类型不是复数类型或布尔类型，针对设置不合法值的情况，逐一断言会触发 TypeError 异常
            if (not np.issubdtype(dtype, np.complexfloating) and
                    dtype != np.bool_):
                for v in [3j]:
                    assert_raises(TypeError, A.__setitem__, (0,0), v)

        # 遍历所有支持的数据类型，对每种类型调用 check 函数进行测试
        for dtype in supported_dtypes:
            check(np.dtype(dtype))

    # 定义一个测试方法 test_negative_index_assignment，用于测试负索引设置元素的功能
    def test_negative_index_assignment(self):
        # GitHub 问题 4428 的回归测试

        # 定义一个内部函数 check，用于具体检查不同数据类型的负索引设置操作
        def check(dtype):
            # 通过 self.spcreator 方法创建一个形状为 (3,10) 的稀疏矩阵 A，指定数据类型为 dtype
            A = self.spcreator((3, 10), dtype=dtype)
            
            # 使用 suppress_warnings 上下文管理器，过滤 SparseEfficiencyWarning 警告信息
            with suppress_warnings() as sup:
                sup.filter(SparseEfficiencyWarning, "Changing the sparsity structure")
                
                # 设置稀疏矩阵 A 中的一个负索引元素
                A[0, -4] = 1
            
            # 断言稀疏矩阵 A[0, -4] 的值为 1
            assert_equal(A[0, -4], 1)

        # 遍历 self.math_dtypes 中的所有数据类型，对每种类型调用 check 函数进行测试
        for dtype in self.math_dtypes:
            check(np.dtype(dtype))
    # 定义一个测试方法，用于测试稀疏矩阵的标量赋值
    def test_scalar_assign_2(self):
        # 定义变量 n 和 m，并赋值为 (5, 10)
        n, m = (5, 10)

        # 定义一个内部函数 _test_set，用于设置稀疏矩阵的元素并进行断言验证
        def _test_set(i, j, nitems):
            # 生成测试消息，包括 i、j 和 nitems 的值
            msg = f"{i!r} ; {j!r} ; {nitems!r}"
            # 创建一个稀疏矩阵 A，形状为 (n, m)
            A = self.spcreator((n, m))
            # 使用 suppress_warnings 上下文管理器，过滤 SparseEfficiencyWarning，避免警告输出
            with suppress_warnings() as sup:
                # 过滤掉 "Changing the sparsity structure" 的警告信息
                sup.filter(SparseEfficiencyWarning, "Changing the sparsity structure")
                # 设置矩阵 A 中索引 (i, j) 处的值为 1
                A[i, j] = 1
            # 断言稀疏矩阵 A 的元素总和为 nitems
            assert_almost_equal(A.sum(), nitems, err_msg=msg)
            # 再次断言稀疏矩阵 A 中索引 (i, j) 处的值为 1
            assert_almost_equal(A[i, j], 1, err_msg=msg)

        # 对于每对 (i, j) 进行测试
        for i, j in [(2, 3), (-1, 8), (-1, -2), (array(-1), -2), (-1, array(-2)),
                     (array(-1), array(-2))]:
            # 调用 _test_set 方法，设置 (i, j) 处的值为 1，nitems 为 1
            _test_set(i, j, 1)

    # 定义一个测试方法，用于测试索引赋值
    def test_index_scalar_assign(self):
        # 创建一个稀疏矩阵 A，形状为 (5, 5)
        A = self.spcreator((5, 5))
        # 创建一个全零矩阵 B，形状为 (5, 5)
        B = np.zeros((5, 5))
        # 使用 suppress_warnings 上下文管理器，过滤 SparseEfficiencyWarning，避免警告输出
        with suppress_warnings() as sup:
            # 过滤掉 "Changing the sparsity structure" 的警告信息
            sup.filter(SparseEfficiencyWarning, "Changing the sparsity structure")
            # 对于矩阵 A 和 B 中的每一个，设置特定索引处的值
            for C in [A, B]:
                C[0, 1] = 1   # 设置索引 (0, 1) 处的值为 1
                C[3, 0] = 4   # 设置索引 (3, 0) 处的值为 4
                C[3, 0] = 9   # 再次设置索引 (3, 0) 处的值为 9，覆盖之前的值
        # 断言稀疏矩阵 A 转换为普通矩阵后与 B 相等
        assert_array_equal(A.toarray(), B)
class _TestSolve:
    def test_solve(self):
        # 测试 lu_solve 命令是否会崩溃，这是由 Nils Wagner 在 2005 年 3 月 2 日报告的问题，针对 64 位机器 (EJS)
        n = 20
        np.random.seed(0)  # 使测试结果可重复
        A = zeros((n,n), dtype=complex)
        x = np.random.rand(n)
        y = np.random.rand(n-1)+1j*np.random.rand(n-1)
        r = np.random.rand(n)
        for i in range(len(x)):
            A[i,i] = x[i]
        for i in range(len(y)):
            A[i,i+1] = y[i]
            A[i+1,i] = conjugate(y[i])
        A = self.spcreator(A)  # 使用 spcreator 方法创建稀疏矩阵 A
        with suppress_warnings() as sup:
            sup.filter(SparseEfficiencyWarning,
                       "splu converted its input to CSC format")
            x = splu(A).solve(r)  # 使用 splu 求解方程组 A @ x = r
        assert_almost_equal(A @ x,r)  # 断言 A @ x 等于 r


class _TestSlicing:
    def test_dtype_preservation(self):
        assert_equal(self.spcreator((1,10), dtype=np.int16)[0,1:5].dtype, np.int16)
        assert_equal(self.spcreator((1,10), dtype=np.int32)[0,1:5].dtype, np.int32)
        assert_equal(self.spcreator((1,10), dtype=np.float32)[0,1:5].dtype, np.float32)
        assert_equal(self.spcreator((1,10), dtype=np.float64)[0,1:5].dtype, np.float64)

    def test_dtype_preservation_empty_slice(self):
        # 这里应该使用 pytest 参数化，但是在这个文件中的父类创建中有东西破坏了 pytest.mark.parametrize。
        for dt in [np.int16, np.int32, np.float32, np.float64]:
            A = self.spcreator((3, 2), dtype=dt)
            assert_equal(A[:, 0:0:2].dtype, dt)
            assert_equal(A[0:0:2, :].dtype, dt)
            assert_equal(A[0, 0:0:2].dtype, dt)
            assert_equal(A[0:0:2, 0].dtype, dt)

    def test_get_horiz_slice(self):
        B = asmatrix(arange(50.).reshape(5,10))
        A = self.spcreator(B)
        assert_array_equal(B[1, :], A[1, :].toarray())
        assert_array_equal(B[1, 2:5], A[1, 2:5].toarray())

        C = matrix([[1, 2, 1], [4, 0, 6], [0, 0, 0], [0, 0, 1]])
        D = self.spcreator(C)
        assert_array_equal(C[1, 1:3], D[1, 1:3].toarray())

        # 当一行仅包含零时测试切片
        E = matrix([[1, 2, 1], [4, 0, 0], [0, 0, 0], [0, 0, 1]])
        F = self.spcreator(E)
        assert_array_equal(E[1, 1:3], F[1, 1:3].toarray())
        assert_array_equal(E[2, -2:], F[2, -2:].toarray())

        # 以下应该引发异常:
        assert_raises(IndexError, A.__getitem__, (slice(None), 11))
        assert_raises(IndexError, A.__getitem__, (6, slice(3, 7)))
    # 定义测试函数，用于测试垂直切片的获取
    def test_get_vert_slice(self):
        # 创建一个 5x10 的二维数组 B，并传入 spcreator 函数创建稀疏矩阵 A
        B = arange(50.).reshape(5, 10)
        A = self.spcreator(B)
        # 断言：稀疏矩阵 A 的垂直切片 [2:5, [0]] 应当等于数组 B 的相应切片，并转换为稀疏表示后比较
        assert_array_equal(B[2:5, [0]], A[2:5, 0].toarray())
        # 断言：稀疏矩阵 A 的垂直切片 [:, [1]] 应当等于数组 B 的相应切片，并转换为稀疏表示后比较
        assert_array_equal(B[:, [1]], A[:, 1].toarray())

        # 创建一个 4x3 的二维数组 C，并传入 spcreator 函数创建稀疏矩阵 D
        C = array([[1, 2, 1], [4, 0, 6], [0, 0, 0], [0, 0, 1]])
        D = self.spcreator(C)
        # 断言：稀疏矩阵 D 的垂直切片 [1:3, [1]] 应当等于数组 C 的相应切片，并转换为稀疏表示后比较
        assert_array_equal(C[1:3, [1]], D[1:3, 1].toarray())
        # 断言：稀疏矩阵 D 的垂直切片 [:, [2]] 应当等于数组 C 的相应切片，并转换为稀疏表示后比较
        assert_array_equal(C[:, [2]], D[:, 2].toarray())

        # 创建一个 4x3 的二维数组 E，并传入 spcreator 函数创建稀疏矩阵 F
        E = array([[1, 0, 1], [4, 0, 0], [0, 0, 0], [0, 0, 1]])
        F = self.spcreator(E)
        # 断言：稀疏矩阵 F 的垂直切片 [:, [1]] 应当等于数组 E 的相应切片，并转换为稀疏表示后比较
        assert_array_equal(E[:, [1]], F[:, 1].toarray())
        # 断言：稀疏矩阵 F 的垂直切片 [-2:, [2]] 应当等于数组 E 的相应切片，并转换为稀疏表示后比较
        assert_array_equal(E[-2:, [2]], F[-2:, 2].toarray())

        # 下面的代码段应该引发异常：
        # 断言：当尝试访问超出索引范围的列时，应该引发 IndexError 异常
        assert_raises(IndexError, A.__getitem__, (slice(None), 11))
        # 断言：当尝试访问超出索引范围的行时，应该引发 IndexError 异常
        assert_raises(IndexError, A.__getitem__, (6, slice(3, 7)))

    # 定义测试函数，用于测试多种切片操作
    def test_get_slices(self):
        # 创建一个 5x10 的二维数组 B，并传入 spcreator 函数创建稀疏矩阵 A
        B = arange(50.).reshape(5, 10)
        A = self.spcreator(B)
        # 断言：稀疏矩阵 A 的切片 [2:5, 0:3] 应当等于数组 B 的相应切片
        assert_array_equal(A[2:5, 0:3].toarray(), B[2:5, 0:3])
        # 断言：稀疏矩阵 A 的切片 [1:, :-1] 应当等于数组 B 的相应切片
        assert_array_equal(A[1:, :-1].toarray(), B[1:, :-1])
        # 断言：稀疏矩阵 A 的切片 [:-1, 1:] 应当等于数组 B 的相应切片
        assert_array_equal(A[:-1, 1:].toarray(), B[:-1, 1:])

        # 创建一个 4x3 的二维数组 E，并传入 spcreator 函数创建稀疏矩阵 F
        E = array([[1, 0, 1], [4, 0, 0], [0, 0, 0], [0, 0, 1]])
        F = self.spcreator(E)
        # 断言：稀疏矩阵 F 的切片 [1:2, 1:2] 应当等于数组 E 的相应切片
        assert_array_equal(E[1:2, 1:2], F[1:2, 1:2].toarray())
        # 断言：稀疏矩阵 F 的切片 [:, 1:] 应当等于数组 E 的相应切片
        assert_array_equal(E[:, 1:], F[:, 1:].toarray())

    # 定义测试函数，用于测试二维索引时的非单位步长
    def test_non_unit_stride_2d_indexing(self):
        # 创建一个 50x50 的随机数组 v0
        v0 = np.random.rand(50, 50)
        try:
            # 尝试使用 spcreator 函数对 v0 进行切片操作，并捕获 ValueError 异常
            v = self.spcreator(v0)[0:25:2, 2:30:3]
        except ValueError:
            # 如果不支持该操作，抛出 pytest 跳过异常
            raise pytest.skip("feature not implemented")

        # 断言：稀疏矩阵 v 的内容应当等于数组 v0 的相应切片
        assert_array_equal(v.toarray(), v0[0:25:2, 2:30:3])
    # 定义一个测试函数，测试切片操作
    def test_slicing_2(self):
        # 创建一个5x10的矩阵B，并将其转换为矩阵对象
        B = asmatrix(arange(50).reshape(5,10))
        # 调用self.spcreator方法，将矩阵B转换为A
        A = self.spcreator(B)

        # 测试索引操作 A[2,3] 是否等于 B[2,3]
        assert_equal(A[2,3], B[2,3])
        # 测试索引操作 A[-1,8] 是否等于 B[-1,8]
        assert_equal(A[-1,8], B[-1,8])
        # 测试索引操作 A[-1,-2] 是否等于 B[-1,-2]
        assert_equal(A[-1,-2], B[-1,-2])
        # 测试索引操作 A[array(-1),-2] 是否等于 B[-1,-2]
        assert_equal(A[array(-1),-2], B[-1,-2])
        # 测试索引操作 A[-1,array(-2)] 是否等于 B[-1,-2]
        assert_equal(A[-1,array(-2)], B[-1,-2])
        # 测试索引操作 A[array(-1),array(-2)] 是否等于 B[-1,-2]
        assert_equal(A[array(-1),array(-2)], B[-1,-2])

        # 测试切片操作 A[2, :] 是否等于 B[2, :]
        assert_equal(A[2, :].toarray(), B[2, :])
        # 测试切片操作 A[2, 5:-2] 是否等于 B[2, 5:-2]
        assert_equal(A[2, 5:-2].toarray(), B[2, 5:-2])
        # 测试切片操作 A[array(2), 5:-2] 是否等于 B[2, 5:-2]
        assert_equal(A[array(2), 5:-2].toarray(), B[2, 5:-2])

        # 测试切片操作 A[:, 2] 是否等于 B[:, 2]
        assert_equal(A[:, 2].toarray(), B[:, 2])
        # 测试切片操作 A[3:4, 9] 是否等于 B[3:4, 9]
        assert_equal(A[3:4, 9].toarray(), B[3:4, 9])
        # 测试切片操作 A[1:4, -5] 是否等于 B[1:4, -5]
        assert_equal(A[1:4, -5].toarray(), B[1:4, -5])
        # 测试切片操作 A[2:-1, 3] 是否等于 B[2:-1, 3]
        assert_equal(A[2:-1, 3].toarray(), B[2:-1, 3])
        # 测试切片操作 A[2:-1, array(3)] 是否等于 B[2:-1, 3]
        assert_equal(A[2:-1, array(3)].toarray(), B[2:-1, 3])

        # 测试切片操作 A[1:2, 1:2] 是否等于 B[1:2, 1:2]
        assert_equal(A[1:2, 1:2].toarray(), B[1:2, 1:2])
        # 测试切片操作 A[4:, 3:] 是否等于 B[4:, 3:]
        assert_equal(A[4:, 3:].toarray(), B[4:, 3:])
        # 测试切片操作 A[:4, :5] 是否等于 B[:4, :5]
        assert_equal(A[:4, :5].toarray(), B[:4, :5])
        # 测试切片操作 A[2:-1, :5] 是否等于 B[2:-1, :5]
        assert_equal(A[2:-1, :5].toarray(), B[2:-1, :5])

        # 测试切片操作 A[1, :] 是否等于 B[1, :]
        assert_equal(A[1, :].toarray(), B[1, :])
        # 测试切片操作 A[-2, :] 是否等于 B[-2, :]
        assert_equal(A[-2, :].toarray(), B[-2, :])
        # 测试切片操作 A[array(-2), :] 是否等于 B[-2, :]
        assert_equal(A[array(-2), :].toarray(), B[-2, :])

        # 测试切片操作 A[1:4] 是否等于 B[1:4]
        assert_equal(A[1:4].toarray(), B[1:4])
        # 测试切片操作 A[1:-2] 是否等于 B[1:-2]
        assert_equal(A[1:-2].toarray(), B[1:-2])

        # 检查由Robert Cimrman报告的Bug
        # 使用int8类型的切片s，测试切片操作 A[s, :] 是否等于 B[2:4, :]
        s = slice(int8(2),int8(4),None)
        assert_equal(A[s, :].toarray(), B[2:4, :])
        # 测试切片操作 A[:, s] 是否等于 B[:, 2:4]
        assert_equal(A[:, s].toarray(), B[:, 2:4])

    # 标记该测试为fail_slow，并设置最大失败次数为2
    @pytest.mark.fail_slow(2)
    # 定义一个测试方法，测试切片操作（三维）的正确性
    def test_slicing_3(self):
        # 创建一个 5x10 的矩阵 B
        B = asmatrix(arange(50).reshape(5,10))
        # 调用 self.spcreator 方法，将 B 转换为另一个对象 A
        A = self.spcreator(B)

        # 定义 np.s_ 切片对象的别名
        s_ = np.s_
        # 定义多个切片操作
        slices = [s_[:2], s_[1:2], s_[3:], s_[3::2],
                  s_[15:20], s_[3:2],
                  s_[8:3:-1], s_[4::-2], s_[:5:-1],
                  0, 1, s_[:], s_[1:5], -1, -2, -5,
                  array(-1), np.int8(-3)]

        # 定义一个函数，检查对给定切片 a 的索引操作是否正确
        def check_1(a):
            # 从对象 A 中获取切片 a 对应的数据
            x = A[a]
            # 从矩阵 B 中获取切片 a 对应的数据
            y = B[a]
            # 如果 y 的形状是空的元组，断言 x 和 y 相等
            if y.shape == ():
                assert_equal(x, y, repr(a))
            else:
                # 如果 x 和 y 都是空的或者 x 和 y 的数组形状相等，则通过数组断言检查它们是否相等
                if x.size == 0 and y.size == 0:
                    pass
                else:
                    assert_array_equal(x.toarray(), y, repr(a))

        # 对于 slices 中的每个切片 a，调用 check_1 函数进行检查
        for j, a in enumerate(slices):
            check_1(a)

        # 定义一个函数，检查对给定切片 a 和 b 的索引操作是否正确
        def check_2(a, b):
            # 如果 a 是 ndarray 类型，则将其转换为整数，否则保持原样
            if isinstance(a, np.ndarray):
                ai = int(a)
            else:
                ai = a
            # 如果 b 是 ndarray 类型，则将其转换为整数，否则保持原样
            if isinstance(b, np.ndarray):
                bi = int(b)
            else:
                bi = b

            # 从对象 A 中获取切片 (a, b) 对应的数据
            x = A[a, b]
            # 从矩阵 B 中获取切片 (ai, bi) 对应的数据
            y = B[ai, bi]

            # 如果 y 的形状是空的元组，断言 x 和 y 相等
            if y.shape == ():
                assert_equal(x, y, repr((a, b)))
            else:
                # 如果 x 和 y 都是空的或者 x 和 y 的数组形状相等，则通过数组断言检查它们是否相等
                if x.size == 0 and y.size == 0:
                    pass
                else:
                    assert_array_equal(x.toarray(), y, repr((a, b)))

        # 对于 slices 中的每个切片对 (a, b)，调用 check_2 函数进行检查
        for i, a in enumerate(slices):
            for j, b in enumerate(slices):
                check_2(a, b)

        # 系统性地检查越界等情况的额外切片
        extra_slices = []
        # 使用 itertools.product 生成多个切片的组合
        for a, b, c in itertools.product(*([(None, 0, 1, 2, 5, 15,
                                             -1, -2, 5, -15)]*3)):
            if c == 0:
                continue
            extra_slices.append(slice(a, b, c))

        # 对于 extra_slices 中的每个切片 a，调用 check_2 函数进行检查
        for a in extra_slices:
            check_2(a, a)
            check_2(a, -2)
            check_2(-2, a)

    # 定义一个测试方法，测试省略号切片操作的正确性
    def test_ellipsis_slicing(self):
        # 创建一个 5x10 的矩阵 b
        b = asmatrix(arange(50).reshape(5,10))
        # 调用 self.spcreator 方法，将 b 转换为另一个对象 a
        a = self.spcreator(b)

        # 断言省略号切片操作返回的稀疏矩阵 a 的稠密数组等于矩阵 b 的稠密数组
        assert_array_equal(a[...].toarray(), b[...].A)
        assert_array_equal(a[...,].toarray(), b[...,].A)

        # 断言带有一维索引的省略号切片操作返回的稀疏矩阵 a 的稠密数组等于矩阵 b 的稠密数组
        assert_array_equal(a[1, ...].toarray(), b[1, ...].A)
        assert_array_equal(a[..., 1].toarray(), b[..., 1].A)
        assert_array_equal(a[1:, ...].toarray(), b[1:, ...].A)
        assert_array_equal(a[..., 1:].toarray(), b[..., 1:].A)

        # 断言带有两个一维索引的省略号切片操作返回的稀疏矩阵 a 的稠密数组等于矩阵 b 的稠密数组
        assert_array_equal(a[1:, 1, ...].toarray(), b[1:, 1, ...].A)
        assert_array_equal(a[1, ..., 1:].toarray(), b[1, ..., 1:].A)
        # 这些操作返回整数
        assert_equal(a[1, 1, ...], b[1, 1, ...])
        assert_equal(a[1, ..., 1], b[1, ..., 1])
    # 定义一个测试方法，用于测试多个省略号的切片操作
    def test_multiple_ellipsis_slicing(self):
        # 调用 self.spcreator 方法，创建一个数组 a，形状为 (3, 2)，其中包含从 0 到 5 的整数
        a = self.spcreator(arange(6).reshape(3, 2))

        # 使用 pytest 的 assertRaises 函数捕获 IndexError 异常，确保切片中只有一个省略号
        with pytest.raises(IndexError,
                           match='an index can only have a single ellipsis'):
            a[..., ...]
        
        # 再次使用 pytest 的 assertRaises 函数捕获 IndexError 异常，确保切片中只有一个省略号
        with pytest.raises(IndexError,
                           match='an index can only have a single ellipsis'):
            a[..., 1, ...]
class _TestSlicingAssign:
    # 测试切片赋值的类
    def test_slice_scalar_assign(self):
        # 测试标量切片赋值
        A = self.spcreator((5, 5))
        # 创建一个稀疏矩阵 A
        B = np.zeros((5, 5))
        # 创建一个全零的 NumPy 数组 B
        with suppress_warnings() as sup:
            # 忽略警告
            sup.filter(SparseEfficiencyWarning, "Changing the sparsity structure")
            # 设置警告过滤器
            for C in [A, B]:
                # 对 A 和 B 进行循环操作
                C[0:1,1] = 1
                # 将 C 的部分切片赋值为 1
                C[3:0,0] = 4
                # 尝试将 C 的部分切片赋值为 4（无效，因为切片范围错误）
                C[3:4,0] = 9
                # 将 C 的部分切片赋值为 9
                C[0,4:] = 1
                # 将 C 的部分切片赋值为 1
                C[3::-1,4:] = 9
                # 将 C 的部分切片赋值为 9
        assert_array_equal(A.toarray(), B)
        # 断言 A 转换为数组后与 B 相等

    def test_slice_assign_2(self):
        # 测试切片赋值的另一种情况
        n, m = (5, 10)

        def _test_set(i, j):
            # 内部函数，测试设置特定位置 (i, j)
            msg = f"i={i!r}; j={j!r}"
            # 打印消息
            A = self.spcreator((n, m))
            # 创建稀疏矩阵 A
            with suppress_warnings() as sup:
                # 忽略警告
                sup.filter(SparseEfficiencyWarning, "Changing the sparsity structure")
                # 设置警告过滤器
                A[i, j] = 1
                # 将 A 的位置 (i, j) 赋值为 1
            B = np.zeros((n, m))
            # 创建一个全零的 NumPy 数组 B
            B[i, j] = 1
            # 将 B 的位置 (i, j) 赋值为 1
            assert_array_almost_equal(A.toarray(), B, err_msg=msg)
            # 断言 A 转换为数组后与 B 相近，打印错误消息

        # 对不同的 (i, j) 组合进行测试
        for i, j in [(2, slice(3)), (2, slice(None, 10, 4)), (2, slice(5, -2)),
                     (array(2), slice(5, -2))]:
            _test_set(i, j)

    def test_self_self_assignment(self):
        # 测试自我赋值
        # Tests whether a row of one lil_matrix can be assigned to
        # another.
        # 测试一行 lil_matrix 是否可以赋值给另一个
        B = self.spcreator((4,3))
        # 创建一个稀疏矩阵 B
        with suppress_warnings() as sup:
            # 忽略警告
            sup.filter(SparseEfficiencyWarning, "Changing the sparsity structure")
            # 设置警告过滤器
            B[0,0] = 2
            # 将 B 的位置 (0,0) 赋值为 2
            B[1,2] = 7
            # 将 B 的位置 (1,2) 赋值为 7
            B[2,1] = 3
            # 将 B 的位置 (2,1) 赋值为 3
            B[3,0] = 10
            # 将 B 的位置 (3,0) 赋值为 10

            A = B / 10
            # 将 B 的每个元素除以 10，结果赋给 A
            B[0,:] = A[0,:]
            # 将 A 的第一行赋值给 B 的第一行
            assert_array_equal(A[0,:].toarray(), B[0,:].toarray())
            # 断言 A 的第一行转换为数组后与 B 的第一行转换为数组后相等

            A = B / 10
            # 将 B 的每个元素除以 10，结果赋给 A
            B[:,:] = A[:1,:1]
            # 将 A 的第一个元素赋值给 B
            assert_array_equal(np.zeros((4,3)) + A[0,0], B.toarray())
            # 断言 A 的第一个元素与 B 转换为数组后相等

            A = B / 10
            # 将 B 的每个元素除以 10，结果赋给 A
            B[:-1,0] = A[0,:].T
            # 将 A 的第一行转置后赋值给 B 的第一列
            assert_array_equal(A[0,:].toarray().T, B[:-1,0].toarray())
            # 断言 A 的第一行转置后与 B 的第一列转换为数组后相等

    def test_slice_assignment(self):
        # 测试切片赋值
        B = self.spcreator((4,3))
        # 创建一个稀疏矩阵 B
        expected = array([[10,0,0],
                          [0,0,6],
                          [0,14,0],
                          [0,0,0]])
        # 预期的数组
        block = [[1,0],[0,4]]
        # 块数组

        with suppress_warnings() as sup:
            # 忽略警告
            sup.filter(SparseEfficiencyWarning, "Changing the sparsity structure")
            # 设置警告过滤器
            B[0,0] = 5
            # 将 B 的位置 (0,0) 赋值为 5
            B[1,2] = 3
            # 将 B 的位置 (1,2) 赋值为 3
            B[2,1] = 7
            # 将 B 的位置 (2,1) 赋值为 7
            B[:,:] = B+B
            # 将 B 的每个元素加上自身的值，结果赋给 B
            assert_array_equal(B.toarray(), expected)
            # 断言 B 转换为数组后与预期数组相等

            B[:2,:2] = csc_matrix(array(block))
            # 将 block 数组转换为 csc_matrix 后赋值给 B 的左上角部分
            assert_array_equal(B.toarray()[:2, :2], block)
            # 断言 B 的左上角部分转换为数组后与 block 相等

    def test_sparsity_modifying_assignment(self):
        # 测试修改稀疏性的赋值操作
        B = self.spcreator((4,3))
        # 创建一个稀疏矩阵 B
        with suppress_warnings() as sup:
            # 忽略警告
            sup.filter(SparseEfficiencyWarning, "Changing the sparsity structure")
            # 设置警告过滤器
            B[0,0] = 5
            # 将 B 的位置 (0,0) 赋值为 5
            B[1,2] = 3
            # 将 B 的位置 (1,2) 赋值为 3
            B[2,1] = 7
            # 将 B 的位置 (2,1) 赋值为 7
            B[3,0] = 10
            # 将 B 的位置 (3,0) 赋值为 10
            B[:3] = csr_matrix(np.eye(3))
            # 将单位矩阵的前三行赋值给 B
        expected = array([[1,0,0],[0,1,0],[0,0,1],[10,0,0]])
        # 预期的数组
        assert_array_equal(B.toarray(), expected)
        # 断言 B 转换为数组后与预期数组相等
    # 定义一个测试方法，用于测试稀疏矩阵的切片设置操作
    def test_set_slice(self):
        # 创建一个稀疏矩阵 A，形状为 (5, 10)
        A = self.spcreator((5,10))
        # 创建一个全零的普通数组 B，形状为 (5, 10)
        B = array(zeros((5, 10), float))
        # 获取切片对象的别名 s_
        s_ = np.s_
        # 定义一组切片列表
        slices = [s_[:2], s_[1:2], s_[3:], s_[3::2],
                  s_[8:3:-1], s_[4::-2], s_[:5:-1],
                  0, 1, s_[:], s_[1:5], -1, -2, -5,
                  array(-1), np.int8(-3)]

        # 忽略警告的上下文管理器
        with suppress_warnings() as sup:
            # 过滤掉稀疏效率警告 "Changing the sparsity structure"
            sup.filter(SparseEfficiencyWarning, "Changing the sparsity structure")
            # 对每个切片执行赋值操作，并验证结果与 B 的数组是否相等
            for j, a in enumerate(slices):
                A[a] = j
                B[a] = j
                assert_array_equal(A.toarray(), B, repr(a))

            # 嵌套循环，对每对切片执行赋值操作，并验证结果与 B 的数组是否相等
            for i, a in enumerate(slices):
                for j, b in enumerate(slices):
                    A[a,b] = 10*i + 1000*(j+1)
                    B[a,b] = 10*i + 1000*(j+1)
                    assert_array_equal(A.toarray(), B, repr((a, b)))

            # 对特定的切片进行赋值操作，并验证结果与 B 的数组是否相等
            A[0, 1:10:2] = range(1, 10, 2)
            B[0, 1:10:2] = range(1, 10, 2)
            assert_array_equal(A.toarray(), B)
            A[1:5:2, 0] = np.arange(1, 5, 2)[:, None]
            B[1:5:2, 0] = np.arange(1, 5, 2)[:]
            assert_array_equal(A.toarray(), B)

        # 下面的命令应该引发异常
        # 验证设置超出稀疏矩阵范围的操作是否会引发 ValueError 异常
        assert_raises(ValueError, A.__setitem__, (0, 0), list(range(100)))
        assert_raises(ValueError, A.__setitem__, (0, 0), arange(100))
        assert_raises(ValueError, A.__setitem__, (0, slice(None)),
                      list(range(100)))
        assert_raises(ValueError, A.__setitem__, (slice(None), 1),
                      list(range(100)))
        assert_raises(ValueError, A.__setitem__, (slice(None), 1), A.copy())
        assert_raises(ValueError, A.__setitem__,
                      ([[1, 2, 3], [0, 3, 4]], [1, 2, 3]), [1, 2, 3, 4])
        assert_raises(ValueError, A.__setitem__,
                      ([[1, 2, 3], [0, 3, 4], [4, 1, 3]],
                       [[1, 2, 4], [0, 1, 3]]), [2, 3, 4])
        assert_raises(ValueError, A.__setitem__, (slice(4), 0),
                      [[1, 2], [3, 4]])

    # 定义一个测试方法，用于测试稀疏矩阵的赋值操作（对应情况：分配空数组）
    def test_assign_empty(self):
        # 创建一个稀疏矩阵 A，形状为 (2, 3)
        A = self.spcreator(np.ones((2, 3)))
        # 创建一个稀疏矩阵 B，形状为 (1, 2)
        B = self.spcreator((1, 2))
        # 对 A 的特定切片进行赋值操作，并验证结果是否符合预期
        A[1, :2] = B
        assert_array_equal(A.toarray(), [[1, 1, 1], [0, 0, 1]])

    # 定义一个测试方法，用于测试稀疏矩阵的赋值操作（对应情况：一维切片分配）
    def test_assign_1d_slice(self):
        # 创建一个稀疏矩阵 A，形状为 (3, 3)
        A = self.spcreator(np.ones((3, 3)))
        # 创建一个全零的一维数组 x，长度为 3
        x = np.zeros(3)
        # 对 A 的列向切片进行赋值操作，并验证结果是否符合预期
        A[:, 0] = x
        # 对 A 的行向切片进行赋值操作，并验证结果是否符合预期
        A[1, :] = x
        assert_array_equal(A.toarray(), [[0, 1, 1], [0, 0, 0], [0, 1, 1]])
class _TestFancyIndexing:
    """Tests fancy indexing features.  The tests for any matrix formats
    that implement these features should derive from this class.
    """

    def test_dtype_preservation_empty_index(self):
        # This should be parametrized with pytest, but something in the parent
        # class creation used in this file breaks pytest.mark.parametrize.
        # 循环测试不同的数据类型
        for dt in [np.int16, np.int32, np.float32, np.float64]:
            # 创建一个稀疏矩阵 A，指定数据类型为 dt
            A = self.spcreator((3, 2), dtype=dt)
            # 断言通过空的索引选择子集后，数据类型仍为 dt
            assert_equal(A[:, [False, False]].dtype, dt)
            assert_equal(A[[False, False, False], :].dtype, dt)
            assert_equal(A[:, []].dtype, dt)
            assert_equal(A[[], :].dtype, dt)

    def test_bad_index(self):
        # 创建一个全零矩阵 A
        A = self.spcreator(np.zeros([5, 5]))
        # 断言对 A 使用错误的索引引发 IndexError、ValueError 或 TypeError 异常
        assert_raises((IndexError, ValueError, TypeError), A.__getitem__, "foo")
        assert_raises((IndexError, ValueError, TypeError), A.__getitem__, (2, "foo"))
        assert_raises((IndexError, ValueError), A.__getitem__,
                      ([1, 2, 3], [1, 2, 3, 4]))

    def test_fancy_indexing_randomized(self):
        # 设置随机数种子以便结果可重复
        np.random.seed(1234)
        # 定义常量
        NUM_SAMPLES = 50
        M = 6
        N = 4

        # 创建一个 MxN 的随机矩阵 D，并将小于 0.5 的元素置零
        D = asmatrix(np.random.rand(M,N))
        D = np.multiply(D, D > 0.5)

        # 随机生成索引数组 I 和 J
        I = np.random.randint(-M + 1, M, size=NUM_SAMPLES)
        J = np.random.randint(-N + 1, N, size=NUM_SAMPLES)

        # 使用 spcreator 创建稀疏矩阵 S
        S = self.spcreator(D)

        # 使用索引数组 I 和 J 选择稀疏矩阵 S 的子集 SIJ
        SIJ = S[I,J]
        # 如果 SIJ 是稀疏矩阵，转换成密集矩阵
        if issparse(SIJ):
            SIJ = SIJ.toarray()
        # 断言 SIJ 与 D[I,J] 相等
        assert_equal(SIJ, D[I,J])

        # 创建超出边界的索引数组 I_bad 和 J_bad
        I_bad = I + M
        J_bad = J - N

        # 断言对 S 使用超出边界的索引引发 IndexError 异常
        assert_raises(IndexError, S.__getitem__, (I_bad,J))
        assert_raises(IndexError, S.__getitem__, (I,J_bad))

    def test_missized_masking(self):
        # 定义常量 M 和 N
        M, N = 5, 10

        # 创建一个 MxN 的矩阵 B，并将其转换为稀疏矩阵 A
        B = asmatrix(arange(M * N).reshape(M, N))
        A = self.spcreator(B)

        # 创建不同尺寸的布尔索引数组
        row_long = np.ones(M + 1, dtype=bool)
        row_short = np.ones(M - 1, dtype=bool)
        col_long = np.ones(N + 2, dtype=bool)
        col_short = np.ones(N - 2, dtype=bool)

        match="bool index .* has shape .* instead of .*"
        # 使用 itertools 生成不同组合的索引 i 和 j 进行测试
        for i, j in itertools.product(
            (row_long, row_short, slice(None)),
            (col_long, col_short, slice(None)),
        ):
            # 如果 i 和 j 都是切片，则跳过
            if isinstance(i, slice) and isinstance(j, slice):
                continue
            # 断言对 A 使用不匹配尺寸的索引引发 IndexError 异常，且异常信息匹配特定模式
            with pytest.raises(IndexError, match=match):
                _ = A[i, j]
    def test_fancy_indexing_boolean(self):
        np.random.seed(1234)  # 设置随机种子以使运行结果可重复

        B = asmatrix(arange(50).reshape(5,10))  # 创建一个5行10列的矩阵B
        A = self.spcreator(B)  # 使用spcreator方法创建一个稀疏矩阵A

        I = np.array(np.random.randint(0, 2, size=5), dtype=bool)  # 创建一个大小为5的随机布尔数组I
        J = np.array(np.random.randint(0, 2, size=10), dtype=bool)  # 创建一个大小为10的随机布尔数组J
        X = np.array(np.random.randint(0, 2, size=(5, 10)), dtype=bool)  # 创建一个大小为5x10的随机布尔矩阵X

        assert_equal(toarray(A[I]), B[I])  # 断言A中布尔数组I对应的元素与B中对应的元素相等
        assert_equal(toarray(A[:, J]), B[:, J])  # 断言A中所有行的布尔数组J对应的元素与B中对应的元素相等
        assert_equal(toarray(A[X]), B[X])  # 断言A中布尔矩阵X对应的元素与B中对应的元素相等
        assert_equal(toarray(A[B > 9]), B[B > 9])  # 断言A中大于9的元素对应的元素与B中对应的元素相等

        I = np.array([True, False, True, True, False])  # 创建一个指定布尔数组I
        J = np.array([False, True, True, False, True,
                      False, False, False, False, False])  # 创建一个指定布尔数组J

        assert_equal(toarray(A[I, J]), B[I, J])  # 断言A中布尔索引(I, J)对应的元素与B中对应的元素相等

        Z1 = np.zeros((6, 11), dtype=bool)  # 创建一个6行11列的全零布尔矩阵Z1
        Z2 = np.zeros((6, 11), dtype=bool)  # 创建一个6行11列的全零布尔矩阵Z2
        Z2[0,-1] = True  # 修改Z2的第一行最后一列为True
        Z3 = np.zeros((6, 11), dtype=bool)  # 创建一个6行11列的全零布尔矩阵Z3
        Z3[-1,0] = True  # 修改Z3的最后一行第一列为True

        assert_raises(IndexError, A.__getitem__, Z1)  # 断言尝试使用Z1索引A时抛出IndexError异常
        assert_raises(IndexError, A.__getitem__, Z2)  # 断言尝试使用Z2索引A时抛出IndexError异常
        assert_raises(IndexError, A.__getitem__, Z3)  # 断言尝试使用Z3索引A时抛出IndexError异常
        assert_raises((IndexError, ValueError), A.__getitem__, (X, 1))  # 断言尝试使用(X, 1)索引A时抛出IndexError或ValueError异常

    def test_fancy_indexing_sparse_boolean(self):
        np.random.seed(1234)  # 设置随机种子以使运行结果可重复

        B = asmatrix(arange(50).reshape(5,10))  # 创建一个5行10列的矩阵B
        A = self.spcreator(B)  # 使用spcreator方法创建一个稀疏矩阵A

        X = np.array(np.random.randint(0, 2, size=(5, 10)), dtype=bool)  # 创建一个大小为5x10的随机布尔矩阵X

        Xsp = csr_matrix(X)  # 将X转换为稀疏矩阵Xsp

        assert_equal(toarray(A[Xsp]), B[X])  # 断言A中稀疏矩阵Xsp对应的元素与B中对应的元素相等
        assert_equal(toarray(A[A > 9]), B[B > 9])  # 断言A中大于9的元素对应的元素与B中对应的元素相等

        Z = np.array(np.random.randint(0, 2, size=(5, 11)), dtype=bool)  # 创建一个大小为5x11的随机布尔矩阵Z
        Y = np.array(np.random.randint(0, 2, size=(6, 10)), dtype=bool)  # 创建一个大小为6x10的随机布尔矩阵Y

        Zsp = csr_matrix(Z)  # 将Z转换为稀疏矩阵Zsp
        Ysp = csr_matrix(Y)  # 将Y转换为稀疏矩阵Ysp

        assert_raises(IndexError, A.__getitem__, Zsp)  # 断言尝试使用Zsp索引A时抛出IndexError异常
        assert_raises(IndexError, A.__getitem__, Ysp)  # 断言尝试使用Ysp索引A时抛出IndexError异常
        assert_raises((IndexError, ValueError), A.__getitem__, (Xsp, 1))  # 断言尝试使用(Xsp, 1)索引A时抛出IndexError或ValueError异常

    def test_fancy_indexing_regression_3087(self):
        mat = self.spcreator(array([[1, 0, 0], [0,1,0], [1,0,0]]))  # 创建一个特定的稀疏矩阵mat
        desired_cols = np.ravel(mat.sum(0)) > 0  # 找出mat中每列和大于0的布尔值
        assert_equal(mat[:, desired_cols].toarray(), [[1, 0], [0, 1], [1, 0]])  # 断言提取mat中符合desired_cols条件的列后的结果与预期相等

    def test_fancy_indexing_seq_assign(self):
        mat = self.spcreator(array([[1, 0], [0, 1]]))  # 创建一个特定的稀疏矩阵mat
        assert_raises(ValueError, mat.__setitem__, (0, 0), np.array([1,2]))  # 断言尝试给mat中特定位置赋值时抛出ValueError异常

    def test_fancy_indexing_2d_assign(self):
        # 对于gh-10695的回归测试
        mat = self.spcreator(array([[1, 0], [2, 3]]))  # 创建一个特定的稀疏矩阵mat
        with suppress_warnings() as sup:  # 使用抑制警告的上下文管理器
            sup.filter(SparseEfficiencyWarning, "Changing the sparsity structure")  # 过滤特定的稀疏效率警告信息
            mat[[0, 1], [1, 1]] = mat[[1, 0], [0, 0]]  # 修改mat的特定位置的值
        assert_equal(toarray(mat), array([[1, 2], [2, 1]]))  # 断言修改后的mat与预期的数组相等
    # 定义一个测试方法，用于测试空情况下的复杂索引操作
    def test_fancy_indexing_empty(self):
        # 创建一个包含数字 0 到 49 的矩阵，并重新形状为 5 行 10 列的矩阵 B
        B = asmatrix(arange(50).reshape(5,10))
        # 将矩阵 B 第二行所有元素设为 0
        B[1,:] = 0
        # 将矩阵 B 第三列所有元素设为 0
        B[:,2] = 0
        # 将矩阵 B 第四行第七列的元素设为 0
        B[3,6] = 0
        # 调用 self 对象的 spcreator 方法，使用 B 创建一个新的对象 A
        A = self.spcreator(B)

        # 创建一个布尔类型的数组 K，所有元素为 False
        K = np.array([False, False, False, False, False])
        # 断言将 A 中 K 所对应的部分转换为数组后与 B 中 K 所对应的部分相等
        assert_equal(toarray(A[K]), B[K])
        
        # 创建一个空的整数类型数组 K
        K = np.array([], dtype=int)
        # 断言将 A 中 K 所对应的部分转换为数组后与 B 中 K 所对应的部分相等
        assert_equal(toarray(A[K]), B[K])
        
        # 断言将 A 中 K, K 所对应的部分转换为数组后与 B 中 K, K 所对应的部分相等
        assert_equal(toarray(A[K, K]), B[K, K])
        
        # 创建一个整数类型的数组 J，包含元素 0 到 4，每个元素作为列向量
        J = np.array([0, 1, 2, 3, 4], dtype=int)[:,None]
        # 断言将 A 中 K, J 所对应的部分转换为数组后与 B 中 K, J 所对应的部分相等
        assert_equal(toarray(A[K, J]), B[K, J])
        
        # 断言将 A 中 J, K 所对应的部分转换为数组后与 B 中 J, K 所对应的部分相等
        assert_equal(toarray(A[J, K]), B[J, K])
# 定义一个上下文管理器函数，用于检查操作过程中是否保持了排序索引的属性
@contextlib.contextmanager
def check_remains_sorted(X):
    """Checks that sorted indices property is retained through an operation
    """
    # 如果对象 X 没有属性 'has_sorted_indices' 或者 'has_sorted_indices' 不为真，则直接返回
    if not hasattr(X, 'has_sorted_indices') or not X.has_sorted_indices:
        yield
        return
    # 否则继续执行
    yield
    # 备份当前的索引数组
    indices = X.indices.copy()
    # 修改对象 X 的 'has_sorted_indices' 属性为 False
    X.has_sorted_indices = False
    # 对象 X 执行排序操作
    X.sort_indices()
    # 断言当前的索引数组与备份的索引数组相等，用于确保索引排序没有改变
    assert_array_equal(indices, X.indices,
                       'Expected sorted indices, found unsorted')


class _TestFancyIndexingAssign:
    def test_bad_index_assign(self):
        # 创建一个大小为 5x5 的稀疏矩阵 A
        A = self.spcreator(np.zeros([5, 5]))
        # 断言对 A 进行非法索引赋值会抛出 IndexError、ValueError 或 TypeError 异常
        assert_raises((IndexError, ValueError, TypeError), A.__setitem__, "foo", 2)
        assert_raises((IndexError, ValueError, TypeError), A.__setitem__, (2, "foo"), 5)

    def test_fancy_indexing_set(self):
        # 定义 n 和 m 的值为 5 和 10
        n, m = (5, 10)

        def _test_set_slice(i, j):
            # 创建一个大小为 (n, m) 的稀疏矩阵 A
            A = self.spcreator((n, m))
            # 创建一个大小为 (n, m) 的稠密矩阵 B
            B = asmatrix(np.zeros((n, m)))
            # 忽略警告信息，其中包括稀疏矩阵效率警告
            with suppress_warnings() as sup:
                sup.filter(SparseEfficiencyWarning, "Changing the sparsity structure")
                # 在 B 的索引 i, j 处赋值为 1
                B[i, j] = 1
                # 使用 check_remains_sorted 上下文管理器检查 A 的索引 i, j 处赋值为 1
                with check_remains_sorted(A):
                    A[i, j] = 1
            # 断言 A 转为稠密矩阵后与 B 相等
            assert_array_almost_equal(A.toarray(), B)

        # 循环测试不同的切片索引组合
        for i, j in [((2, 3, 4), slice(None, 10, 4)),
                     (np.arange(3), slice(5, -2)),
                     (slice(2, 5), slice(5, -2))]:
            _test_set_slice(i, j)
        for i, j in [(np.arange(3), np.arange(3)), ((0, 3, 4), (1, 2, 4))]:
            _test_set_slice(i, j)

    def test_fancy_assignment_dtypes(self):
        # 定义一个函数 check，用于检查指定 dtype 的情况
        def check(dtype):
            # 创建一个大小为 (5, 5) 的稀疏矩阵 A，指定 dtype
            A = self.spcreator((5, 5), dtype=dtype)
            # 忽略警告信息，其中包括稀疏矩阵效率警告
            with suppress_warnings() as sup:
                sup.filter(SparseEfficiencyWarning, "Changing the sparsity structure")
                # 对 A 的索引 [0,1] 赋值为指定 dtype 类型的 1
                A[[0,1],[0,1]] = dtype.type(1)
                # 断言 A 所有元素之和等于指定 dtype 类型的 2
                assert_equal(A.sum(), dtype.type(1)*2)
                # 对 A 的切片 [0:2,0:2] 赋值为指定 dtype 类型的 1.0
                A[0:2,0:2] = dtype.type(1.0)
                # 断言 A 所有元素之和等于指定 dtype 类型的 4
                assert_equal(A.sum(), dtype.type(1)*4)
                # 对 A 的索引 [2,2] 赋值为指定 dtype 类型的 1.0
                A[2,2] = dtype.type(1.0)
                # 断言 A 所有元素之和等于指定 dtype 类型的 4 + 1
                assert_equal(A.sum(), dtype.type(1)*4 + dtype.type(1))

        # 遍历支持的 dtype 类型，分别调用 check 函数进行检查
        for dtype in supported_dtypes:
            check(np.dtype(dtype))
    # 测试稀疏矩阵的序列赋值功能
    def test_sequence_assignment(self):
        # 创建稀疏矩阵 A 和 B
        A = self.spcreator((4,3))
        B = self.spcreator(eye(3,4))

        # 定义索引 i0, i1, i2
        i0 = [0,1,2]
        i1 = (0,1,2)
        i2 = array(i0)

        # 忽略警告
        with suppress_warnings() as sup:
            # 过滤稀疏矩阵效率警告
            sup.filter(SparseEfficiencyWarning, "Changing the sparsity structure")
            # 检查 A 的排序保持不变
            with check_remains_sorted(A):
                # 对 A 的特定位置赋值
                A[0,i0] = B[i0,0].T
                A[1,i1] = B[i1,1].T
                A[2,i2] = B[i2,2].T
            # 断言 A 转为稠密数组后与 B 的转置相等
            assert_array_equal(A.toarray(), B.T.toarray())

            # 列切片赋值
            A = self.spcreator((2,3))
            with check_remains_sorted(A):
                A[1,1:3] = [10,20]
            assert_array_equal(A.toarray(), [[0, 0, 0], [0, 10, 20]])

            # 行切片赋值
            A = self.spcreator((3,2))
            with check_remains_sorted(A):
                A[1:3,1] = [[10],[20]]
            assert_array_equal(A.toarray(), [[0, 0], [0, 10], [0, 20]])

            # 同时对行和列进行切片赋值
            A = self.spcreator((3,3))
            B = asmatrix(np.zeros((3,3)))
            with check_remains_sorted(A):
                for C in [A, B]:
                    C[[0,1,2], [0,1,2]] = [4,5,6]
            assert_array_equal(A.toarray(), B)

            # 同时对行和列进行切片赋值（第二组）
            A = self.spcreator((4, 3))
            with check_remains_sorted(A):
                A[(1, 2, 3), (0, 1, 2)] = [1, 2, 3]
            assert_almost_equal(A.sum(), 6)
            B = asmatrix(np.zeros((4, 3)))
            B[(1, 2, 3), (0, 1, 2)] = [1, 2, 3]
            assert_array_equal(A.toarray(), B)

    # 测试对稀疏矩阵进行空索引赋值操作
    def test_fancy_assign_empty(self):
        # 创建普通矩阵 B
        B = asmatrix(arange(50).reshape(5,10))
        
        # 对 B 的特定位置赋值为 0
        B[1,:] = 0
        B[:,2] = 0
        B[3,6] = 0
        
        # 使用 B 创建稀疏矩阵 A
        A = self.spcreator(B)

        # 定义布尔数组 K
        K = np.array([False, False, False, False, False])
        
        # 对 A 中 K 对应的位置赋值为 42
        A[K] = 42
        # 断言 A 转为稠密数组后与 B 相等
        assert_equal(toarray(A), B)

        # 空索引赋值，应该不改变 A
        K = np.array([], dtype=int)
        A[K] = 42
        assert_equal(toarray(A), B)
        
        # 对角线索引赋值为 42，应该不改变 A
        A[K,K] = 42
        assert_equal(toarray(A), B)

        # 使用 J 索引赋值为 42，应该不改变 A
        J = np.array([0, 1, 2, 3, 4], dtype=int)[:,None]
        A[K,J] = 42
        assert_equal(toarray(A), B)
        # 使用 K 索引赋值为 42，应该不改变 A
        A[J,K] = 42
        assert_equal(toarray(A), B)
class _TestFancyMultidim:
    # 定义测试类 _TestFancyMultidim

    def test_fancy_indexing_ndarray(self):
        # 定义测试方法 test_fancy_indexing_ndarray

        sets = [
            (np.array([[1], [2], [3]]), np.array([3, 4, 2])),
            (np.array([[1], [2], [3]]), np.array([[3, 4, 2]])),
            (np.array([[1, 2, 3]]), np.array([[3], [4], [2]])),
            (np.array([1, 2, 3]), np.array([[3], [4], [2]])),
            (np.array([[1, 2, 3], [3, 4, 2]]),
             np.array([[5, 6, 3], [2, 3, 1]]))
        ]
        # 定义多组测试数据 sets，每组包含两个 NumPy 数组

        # These inputs generate 3-D outputs
        #    (np.array([[[1], [2], [3]], [[3], [4], [2]]]),
        #     np.array([[[5], [6], [3]], [[2], [3], [1]]])),
        # 说明这些输入会生成 3 维的输出

        for I, J in sets:
            # 遍历 sets 中的每对数组 I 和 J
            np.random.seed(1234)
            # 设置随机数种子
            D = asmatrix(np.random.rand(5, 7))
            # 生成一个 5x7 的随机矩阵，并将其转换为矩阵对象
            S = self.spcreator(D)
            # 使用 spcreator 方法创建稀疏矩阵 S

            SIJ = S[I,J]
            # 从稀疏矩阵 S 中获取索引为 I 和 J 的部分
            if issparse(SIJ):
                # 如果 SIJ 是稀疏矩阵
                SIJ = SIJ.toarray()
                # 将稀疏矩阵转换为密集数组
            assert_equal(SIJ, D[I,J])
            # 断言 SIJ 与 D[I,J] 相等

            I_bad = I + 5
            J_bad = J + 7

            assert_raises(IndexError, S.__getitem__, (I_bad,J))
            # 断言对 S 的索引 (I_bad, J) 会引发 IndexError 异常
            assert_raises(IndexError, S.__getitem__, (I,J_bad))
            # 断言对 S 的索引 (I, J_bad) 会引发 IndexError 异常

            # This would generate 3-D arrays -- not supported
            assert_raises(IndexError, S.__getitem__, ([I, I], slice(None)))
            # 断言对 S 的索引 ([I, I], slice(None)) 会引发 IndexError 异常
            assert_raises(IndexError, S.__getitem__, (slice(None), [J, J]))
            # 断言对 S 的索引 (slice(None), [J, J]) 会引发 IndexError 异常


class _TestFancyMultidimAssign:
    # 定义测试类 _TestFancyMultidimAssign

    def test_fancy_assign_ndarray(self):
        # 定义测试方法 test_fancy_assign_ndarray

        np.random.seed(1234)
        # 设置随机数种子

        D = asmatrix(np.random.rand(5, 7))
        # 生成一个 5x7 的随机矩阵，并将其转换为矩阵对象
        S = self.spcreator(D)
        # 使用 spcreator 方法创建稀疏矩阵 S
        X = np.random.rand(2, 3)
        # 生成一个 2x3 的随机数组 X

        I = np.array([[1, 2, 3], [3, 4, 2]])
        # 定义一个二维 NumPy 数组 I
        J = np.array([[5, 6, 3], [2, 3, 1]])
        # 定义一个二维 NumPy 数组 J

        with check_remains_sorted(S):
            # 使用 check_remains_sorted 上下文管理器

            S[I,J] = X
            # 将 X 赋值给稀疏矩阵 S 中索引为 I 和 J 的部分
        D[I,J] = X
        # 将 X 赋值给矩阵 D 中索引为 I 和 J 的部分
        assert_equal(S.toarray(), D)
        # 断言稀疏矩阵 S 转换为密集数组后与矩阵 D 相等

        I_bad = I + 5
        J_bad = J + 7

        C = [1, 2, 3]

        with check_remains_sorted(S):
            # 使用 check_remains_sorted 上下文管理器

            S[I,J] = C
            # 将列表 C 赋值给稀疏矩阵 S 中索引为 I 和 J 的部分
        D[I,J] = C
        # 将列表 C 赋值给矩阵 D 中索引为 I 和 J 的部分
        assert_equal(S.toarray(), D)
        # 断言稀疏矩阵 S 转换为密集数组后与矩阵 D 相等

        with check_remains_sorted(S):
            # 使用 check_remains_sorted 上下文管理器

            S[I,J] = 3
            # 将数值 3 赋值给稀疏矩阵 S 中索引为 I 和 J 的部分
        D[I,J] = 3
        # 将数值 3 赋值给矩阵 D 中索引为 I 和 J 的部分
        assert_equal(S.toarray(), D)
        # 断言稀疏矩阵 S 转换为密集数组后与矩阵 D 相等

        assert_raises(IndexError, S.__setitem__, (I_bad,J), C)
        # 断言对 S 的索引 (I_bad, J) 赋值 C 会引发 IndexError 异常
        assert_raises(IndexError, S.__setitem__, (I,J_bad), C)
        # 断言对 S 的索引 (I, J_bad) 赋值 C 会引发 IndexError 异常

    def test_fancy_indexing_multidim_set(self):
        # 定义测试方法 test_fancy_indexing_multidim_set

        n, m = (5, 10)
        # 定义变量 n 和 m 分别赋值为 5 和 10

        def _test_set_slice(i, j):
            # 定义内部函数 _test_set_slice，接受参数 i 和 j

            A = self.spcreator((n, m))
            # 使用 spcreator 方法创建稀疏矩阵 A，尺寸为 (n, m)
            with check_remains_sorted(A), suppress_warnings() as sup:
                # 使用 check_remains_sorted 上下文管理器和 suppress_warnings 上下文管理器
                sup.filter(SparseEfficiencyWarning, "Changing the sparsity structure")
                # 过滤稀疏效率警告中的 "Changing the sparsity structure"
                A[i, j] = 1
                # 将值 1 赋值给稀疏矩阵 A 中索引为 i 和 j 的部分
            B = asmatrix(np.zeros((n, m)))
            # 生成一个全零的矩阵 B，尺寸为 (n, m)
            B[i, j] = 1
            # 将值 1 赋值给矩阵 B 中索引为 i 和 j 的部分
            assert_array_almost_equal(A.toarray(), B)
            # 断言稀疏矩阵 A 转换为密集数组后与矩阵 B 相等

        # [[[1, 2], [1, 2]], [1, 2]]
        for i, j in [(np.array([[1, 2], [1, 3]]), [1, 3]),
                        (np.array([0, 4]), [[0, 3], [1, 2]]),
                        ([[1, 2, 3], [0, 2, 4]], [[0, 4, 3], [4, 1, 2]])]:
            # 遍历多组测试数据 (i, j)
            _test_set
    def test_fancy_assign_list(self):
        # 设置随机种子为1234，确保结果可重复
        np.random.seed(1234)

        # 创建一个 5x7 的随机矩阵 D
        D = asmatrix(np.random.rand(5, 7))
        # 通过 spcreator 方法创建稀疏矩阵 S，基于矩阵 D
        S = self.spcreator(D)
        # 创建一个 2x3 的随机矩阵 X
        X = np.random.rand(2, 3)

        # 定义索引数组 I 和 J
        I = [[1, 2, 3], [3, 4, 2]]
        J = [[5, 6, 3], [2, 3, 1]]

        # 将矩阵 X 赋值给 S 的指定索引位置
        S[I,J] = X
        # 将矩阵 X 赋值给 D 的指定索引位置
        D[I,J] = X
        # 断言稀疏矩阵 S 转换为普通矩阵后与 D 相等
        assert_equal(S.toarray(), D)

        # 创建索引数组 I_bad 和 J_bad，这些索引超出了矩阵的范围
        I_bad = [[ii + 5 for ii in i] for i in I]
        J_bad = [[jj + 7 for jj in j] for j in J]
        # 创建数组 C
        C = [1, 2, 3]

        # 尝试将数组 C 赋值给 S 的超出范围的索引位置，预期会引发 IndexError
        assert_raises(IndexError, S.__setitem__, (I_bad,J), C)
        # 尝试将数组 C 赋值给 S 的超出范围的索引位置，预期会引发 IndexError
        assert_raises(IndexError, S.__setitem__, (I,J_bad), C)

    def test_fancy_assign_slice(self):
        # 设置随机种子为1234，确保结果可重复
        np.random.seed(1234)

        # 创建一个 5x7 的随机矩阵 D
        D = asmatrix(np.random.rand(5, 7))
        # 通过 spcreator 方法创建稀疏矩阵 S，基于矩阵 D
        S = self.spcreator(D)

        # 定义索引数组 I 和 J
        I = [1, 2, 3, 3, 4, 2]
        J = [5, 6, 3, 2, 3, 1]

        # 创建索引数组 I_bad 和 J_bad，这些索引超出了矩阵的范围
        I_bad = [ii + 5 for ii in I]
        J_bad = [jj + 7 for jj in J]

        # 创建数组 C1 和 C2
        C1 = [1, 2, 3, 4, 5, 6, 7]
        C2 = np.arange(5)[:, None]

        # 尝试将数组 C1 赋值给 S 的超出范围的索引位置，预期会引发 IndexError
        assert_raises(IndexError, S.__setitem__, (I_bad, slice(None)), C1)
        # 尝试将数组 C2 赋值给 S 的超出范围的索引位置，预期会引发 IndexError
        assert_raises(IndexError, S.__setitem__, (slice(None), J_bad), C2)
class _TestArithmetic:
    """
    Test real/complex arithmetic
    """
    def __arith_init(self):
        # 定义浮点类型的矩阵 __A，包含负数、整数和小数
        self.__A = array([[-1.5, 6.5, 0, 2.25, 0, 0],
                          [3.125, -7.875, 0.625, 0, 0, 0],
                          [0, 0, -0.125, 1.0, 0, 0],
                          [0, 0, 8.375, 0, 0, 0]], 'float64')
        # 定义复数类型的矩阵 __B，包含实部和虚部
        self.__B = array([[0.375, 0, 0, 0, -5, 2.5],
                          [14.25, -3.75, 0, 0, -0.125, 0],
                          [0, 7.25, 0, 0, 0, 0],
                          [18.5, -0.0625, 0, 0, 0, 0]], 'complex128')
        # 设置 __B 的虚部为一个浮点类型的矩阵
        self.__B.imag = array([[1.25, 0, 0, 0, 6, -3.875],
                               [2.25, 4.125, 0, 0, 0, 2.75],
                               [0, 4.125, 0, 0, 0, 0],
                               [-0.0625, 0, 0, 0, 0, 0]], 'float64')

        # 确保浮点数乘以16后转换为整数类型的矩阵
        assert_array_equal((self.__A*16).astype('int32'), 16*self.__A)
        assert_array_equal((self.__B.real*16).astype('int32'), 16*self.__B.real)
        assert_array_equal((self.__B.imag*16).astype('int32'), 16*self.__B.imag)

        # 使用 self.spcreator 方法创建稀疏矩阵 __Asp 和 __Bsp
        self.__Asp = self.spcreator(self.__A)
        self.__Bsp = self.spcreator(self.__B)

    @pytest.mark.fail_slow(20)
    def test_add_sub(self):
        self.__arith_init()

        # 基本测试，验证稀疏矩阵加法和减法
        assert_array_equal(
            (self.__Asp + self.__Bsp).toarray(), self.__A + self.__B
        )

        # 检查类型转换
        for x in supported_dtypes:
            with np.errstate(invalid="ignore"):
                A = self.__A.astype(x)
            Asp = self.spcreator(A)
            for y in supported_dtypes:
                if not np.issubdtype(y, np.complexfloating):
                    with np.errstate(invalid="ignore"):
                        B = self.__B.real.astype(y)
                else:
                    B = self.__B.astype(y)
                Bsp = self.spcreator(B)

                # 加法
                D1 = A + B
                S1 = Asp + Bsp

                assert_equal(S1.dtype, D1.dtype)
                assert_array_equal(S1.toarray(), D1)
                assert_array_equal(Asp + B, D1)   # 检查稀疏矩阵 + 密集矩阵
                assert_array_equal(A + Bsp, D1)   # 检查密集矩阵 + 稀疏矩阵

                # 减法
                if np.dtype('bool') in [x, y]:
                    # 在1.9.0版本中，布尔类型数组的减法操作被弃用
                    continue

                D1 = A - B
                S1 = Asp - Bsp

                assert_equal(S1.dtype, D1.dtype)
                assert_array_equal(S1.toarray(), D1)
                assert_array_equal(Asp - B, D1)   # 检查稀疏矩阵 - 密集矩阵
                assert_array_equal(A - Bsp, D1)   # 检查密集矩阵 - 稀疏矩阵
    def test_mu(self):
        self.__arith_init()

        # 基本测试
        # 使用稀疏矩阵乘法计算，比较结果与稠密矩阵乘法的结果
        assert_array_equal((self.__Asp @ self.__Bsp.T).toarray(),
                           self.__A @ self.__B.T)

        # 循环遍历支持的数据类型
        for x in supported_dtypes:
            with np.errstate(invalid="ignore"):
                A = self.__A.astype(x)
            Asp = self.spcreator(A)
            for y in supported_dtypes:
                # 根据是否为复数类型，选择相应的数据转换方式
                if np.issubdtype(y, np.complexfloating):
                    B = self.__B.astype(y)
                else:
                    with np.errstate(invalid="ignore"):
                        B = self.__B.real.astype(y)
                Bsp = self.spcreator(B)

                # 计算稠密矩阵乘法结果和稀疏矩阵乘法结果
                D1 = A @ B.T
                S1 = Asp @ Bsp.T

                # 断言稀疏矩阵乘法结果与稠密矩阵乘法结果的接近程度
                assert_allclose(S1.toarray(), D1,
                                atol=1e-14*abs(D1).max())
                # 断言稀疏矩阵的数据类型与稠密矩阵的数据类型相同
                assert_equal(S1.dtype, D1.dtype)
# 定义一个用于测试最小值和最大值计算的测试类 _TestMinMax
class _TestMinMax:
    
    # 定义测试最小值和最大值计算的方法 test_minmax
    def test_minmax(self):
        
        # 遍历不同数据类型
        for dtype in [np.float32, np.float64, np.int32, np.int64, np.complex128]:
            
            # 创建一个包含20个元素的数组 D，使用给定的数据类型 dtype，并将其reshape为5行4列的数组
            D = np.arange(20, dtype=dtype).reshape(5,4)

            # 使用 self.spcreator 方法创建稀疏矩阵 X
            X = self.spcreator(D)
            
            # 断言稀疏矩阵 X 的最小值为 0
            assert_equal(X.min(), 0)
            
            # 断言稀疏矩阵 X 的最大值为 19
            assert_equal(X.max(), 19)
            
            # 断言稀疏矩阵 X 的最小值的数据类型与 dtype 相同
            assert_equal(X.min().dtype, dtype)
            
            # 断言稀疏矩阵 X 的最大值的数据类型与 dtype 相同
            assert_equal(X.max().dtype, dtype)

            # 将数组 D 中的所有元素乘以 -1
            D *= -1
            
            # 使用 self.spcreator 方法重新创建稀疏矩阵 X
            X = self.spcreator(D)
            
            # 断言稀疏矩阵 X 的最小值为 -19
            assert_equal(X.min(), -19)
            
            # 断言稀疏矩阵 X 的最大值为 0
            assert_equal(X.max(), 0)

            # 将数组 D 中的所有元素加上 5
            D += 5
            
            # 使用 self.spcreator 方法重新创建稀疏矩阵 X
            X = self.spcreator(D)
            
            # 断言稀疏矩阵 X 的最小值为 -14
            assert_equal(X.min(), -14)
            
            # 断言稀疏矩阵 X 的最大值为 5
            assert_equal(X.max(), 5)

        # 尝试创建一个完全密集的矩阵 X
        X = self.spcreator(np.arange(1, 10).reshape(3, 3))
        
        # 断言密集矩阵 X 的最小值为 1
        assert_equal(X.min(), 1)
        
        # 断言密集矩阵 X 的最小值的数据类型与 X 的数据类型相同
        assert_equal(X.min().dtype, X.dtype)

        # 将矩阵 X 中的所有元素取反
        X = -X
        
        # 断言矩阵 X 的最大值为 -1
        assert_equal(X.max(), -1)

        # 创建一个完全稀疏的矩阵 Z
        Z = self.spcreator(np.zeros((1, 1)))
        
        # 断言稀疏矩阵 Z 的最小值为 0
        assert_equal(Z.min(), 0)
        
        # 断言稀疏矩阵 Z 的最大值为 0
        assert_equal(Z.max(), 0)
        
        # 断言稀疏矩阵 Z 的最大值的数据类型与 Z 的数据类型相同
        assert_equal(Z.max().dtype, Z.dtype)

        # 另一个测试
        # 创建一个包含20个元素的浮点数数组 D，并将其reshape为5行4列的数组
        D = np.arange(20, dtype=float).reshape(5,4)
        
        # 将数组 D 的前两行所有元素设为 0
        D[0:2, :] = 0
        
        # 使用 self.spcreator 方法重新创建稀疏矩阵 X
        X = self.spcreator(D)
        
        # 断言稀疏矩阵 X 的最小值为 0
        assert_equal(X.min(), 0)
        
        # 断言稀疏矩阵 X 的最大值为 19
        assert_equal(X.max(), 19)

        # 零大小矩阵的测试
        # 遍历不同大小的零矩阵
        for D in [np.zeros((0, 0)), np.zeros((0, 10)), np.zeros((10, 0))]:
            
            # 使用 self.spcreator 方法创建稀疏矩阵 X
            X = self.spcreator(D)
            
            # 断言调用稀疏矩阵 X 的 min 方法会抛出 ValueError 异常
            assert_raises(ValueError, X.min)
            
            # 断言调用稀疏矩阵 X 的 max 方法会抛出 ValueError 异常
            assert_raises(ValueError, X.max)
    # 定义测试函数 test_minmax_axis，用于测试最大和最小函数在不同轴上的行为
    def test_minmax_axis(self):
        # 创建一个包含50个元素的数组，并将其重塑为5行10列的矩阵
        D = np.arange(50).reshape(5, 10)
        # 将第二行所有元素置为0，使其成为完全空的行
        D[1, :] = 0
        # 将第10列所有元素置为0，用于测试 reduceat 函数的空列情况
        D[:, 9] = 0
        # 将第四行第四列的元素置为0，创建部分空的行和列
        D[3, 3] = 0
        # 将第三行第三列的元素置为-1，创建左右各有元素的情况
        D[2, 2] = -1
        # 使用 self.spcreator 函数根据数组 D 创建稀疏矩阵 X
        X = self.spcreator(D)

        # 定义测试轴列表
        axes = [-2, -1, 0, 1]
        # 遍历每个轴进行断言测试
        for axis in axes:
            # 断言稀疏矩阵 X 按指定轴计算最大值，并与原始数组 D 的最大值进行比较
            assert_array_equal(
                X.max(axis=axis).toarray(), D.max(axis=axis, keepdims=True)
            )
            # 断言稀疏矩阵 X 按指定轴计算最小值，并与原始数组 D 的最小值进行比较
            assert_array_equal(
                X.min(axis=axis).toarray(), D.min(axis=axis, keepdims=True)
            )

        # 创建一个完整填充的矩阵 D
        D = np.arange(1, 51).reshape(10, 5)
        # 使用 self.spcreator 函数根据数组 D 创建稀疏矩阵 X
        X = self.spcreator(D)
        # 再次遍历每个轴进行断言测试
        for axis in axes:
            # 断言稀疏矩阵 X 按指定轴计算最大值，并与原始数组 D 的最大值进行比较
            assert_array_equal(
                X.max(axis=axis).toarray(), D.max(axis=axis, keepdims=True)
            )
            # 断言稀疏矩阵 X 按指定轴计算最小值，并与原始数组 D 的最小值进行比较
            assert_array_equal(
                X.min(axis=axis).toarray(), D.min(axis=axis, keepdims=True)
            )

        # 创建一个全零的矩阵 D
        D = np.zeros((10, 5))
        # 使用 self.spcreator 函数根据数组 D 创建稀疏矩阵 X
        X = self.spcreator(D)
        # 再次遍历每个轴进行断言测试
        for axis in axes:
            # 断言稀疏矩阵 X 按指定轴计算最大值，并与原始数组 D 的最大值进行比较
            assert_array_equal(
                X.max(axis=axis).toarray(), D.max(axis=axis, keepdims=True)
            )
            # 断言稀疏矩阵 X 按指定轴计算最小值，并与原始数组 D 的最小值进行比较
            assert_array_equal(
                X.min(axis=axis).toarray(), D.min(axis=axis, keepdims=True)
            )

        # 定义轴列表，包含偶数轴
        axes_even = [0, -2]
        # 定义轴列表，包含奇数轴
        axes_odd = [1, -1]

        # 创建一个零大小的矩阵 D
        D = np.zeros((0, 10))
        # 使用 self.spcreator 函数根据数组 D 创建稀疏矩阵 X
        X = self.spcreator(D)
        # 遍历偶数轴进行异常值断言测试
        for axis in axes_even:
            # 断言在偶数轴上计算稀疏矩阵 X 的最小值和最大值会引发 ValueError 异常
            assert_raises(ValueError, X.min, axis=axis)
            assert_raises(ValueError, X.max, axis=axis)
        # 遍历奇数轴进行零大小矩阵断言测试
        for axis in axes_odd:
            # 断言在奇数轴上计算稀疏矩阵 X 的最小值和最大值为零矩阵
            assert_array_equal(np.zeros((0, 1)), X.min(axis=axis).toarray())
            assert_array_equal(np.zeros((0, 1)), X.max(axis=axis).toarray())

        # 创建一个零大小的矩阵 D
        D = np.zeros((10, 0))
        # 使用 self.spcreator 函数根据数组 D 创建稀疏矩阵 X
        X = self.spcreator(D)
        # 遍历奇数轴进行异常值断言测试
        for axis in axes_odd:
            # 断言在奇数轴上计算稀疏矩阵 X 的最小值和最大值会引发 ValueError 异常
            assert_raises(ValueError, X.min, axis=axis)
            assert_raises(ValueError, X.max, axis=axis)
        # 遍历偶数轴进行零大小矩阵断言测试
        for axis in axes_even:
            # 断言在偶数轴上计算稀疏矩阵 X 的最小值和最大值为零矩阵
            assert_array_equal(np.zeros((1, 0)), X.min(axis=axis).toarray())
            assert_array_equal(np.zeros((1, 0)), X.max(axis=axis).toarray())
    def test_nanminmax(self):
        # 创建一个5行10列的浮点数矩阵D，其中元素从0到49
        D = matrix(np.arange(50).reshape(5,10), dtype=float)
        # 将第1行所有元素置为0
        D[1, :] = 0
        # 将第9列所有元素置为0
        D[:, 9] = 0
        # 将第3行第3列的元素置为0
        D[3, 3] = 0
        # 将第2行第2列的元素置为-1
        D[2, 2] = -1
        # 将第4行第2列的元素置为NaN
        D[4, 2] = np.nan
        # 将第1行第4列的元素置为NaN
        D[1, 4] = np.nan
        # 使用self.spcreator方法创建一个稀疏矩阵X
        X = self.spcreator(D)

        # 计算X中NaN值的最大值
        X_nan_maximum = X.nanmax()
        # 断言X_nan_maximum是标量
        assert np.isscalar(X_nan_maximum)
        # 断言X_nan_maximum等于D中的NaN最大值
        assert X_nan_maximum == np.nanmax(D)

        # 计算X中NaN值的最小值
        X_nan_minimum = X.nanmin()
        # 断言X_nan_minimum是标量
        assert np.isscalar(X_nan_minimum)
        # 断言X_nan_minimum等于D中的NaN最小值
        assert X_nan_minimum == np.nanmin(D)

        # 定义测试轴
        axes = [-2, -1, 0, 1]
        for axis in axes:
            # 计算沿指定轴的X中的NaN值的最大值
            X_nan_maxima = X.nanmax(axis=axis)
            # 断言X_nan_maxima是coo_matrix类型
            assert isinstance(X_nan_maxima, coo_matrix)
            # 断言X_nan_maxima.toarray()近似等于D沿指定轴的NaN最大值
            assert_allclose(X_nan_maxima.toarray(),
                            np.nanmax(D, axis=axis))

            # 计算沿指定轴的X中的NaN值的最小值
            X_nan_minima = X.nanmin(axis=axis)
            # 断言X_nan_minima是coo_matrix类型
            assert isinstance(X_nan_minima, coo_matrix)
            # 断言X_nan_minima.toarray()近似等于D沿指定轴的NaN最小值
            assert_allclose(X_nan_minima.toarray(),
                            np.nanmin(D, axis=axis))

    def test_minmax_invalid_params(self):
        # 创建一个包含特定数据的数组dat
        dat = array([[0, 1, 2],
                     [3, -4, 5],
                     [-6, 7, 9]])
        # 使用self.spcreator方法创建一个稀疏矩阵datsp
        datsp = self.spcreator(dat)

        # 遍历('min', 'max')，对每个函数名fname进行测试
        for fname in ('min', 'max'):
            # 获取datsp中的fname方法
            func = getattr(datsp, fname)
            # 断言在axis=3时会引发ValueError异常
            assert_raises(ValueError, func, axis=3)
            # 断言在axis=(0, 1)时会引发TypeError异常
            assert_raises(TypeError, func, axis=(0, 1))
            # 断言在axis=1.5时会引发TypeError异常
            assert_raises(TypeError, func, axis=1.5)
            # 断言在axis=1且out=1时会引发ValueError异常
            assert_raises(ValueError, func, axis=1, out=1)

    def test_numpy_minmax(self):
        # 查看gh-5987
        # 参考gh-7460中的'numpy'
        from scipy.sparse import _data

        # 创建一个包含特定数据的数组dat
        dat = array([[0, 1, 2],
                     [3, -4, 5],
                     [-6, 7, 9]])
        # 使用self.spcreator方法创建一个稀疏矩阵datsp
        datsp = self.spcreator(dat)

        # 我们仅测试已实现'min'和'max'的稀疏矩阵
        # 因为它们与'numpy'实现存在兼容性问题
        if isinstance(datsp, _data._minmax_mixin):
            # 断言np.min(datsp)近似等于np.min(dat)
            assert_array_equal(np.min(datsp), np.min(dat))
            # 断言np.max(datsp)近似等于np.max(dat)
            assert_array_equal(np.max(datsp), np.max(dat))
    # 定义一个测试方法，用于测试 argmax 和 argmin 方法的功能
    def test_argmax(self):
        # 导入所需的模块 _data
        from scipy.sparse import _data
        # 创建一个二维数组 D1
        D1 = np.array([
            [-1, 5, 2, 3],
            [0, 0, -1, -2],
            [-1, -2, -3, -4],
            [1, 2, 3, 4],
            [1, 2, 0, 0],
        ])
        # 计算 D1 的转置矩阵并赋给 D2
        D2 = D1.transpose()
        # 用于 gh-16929 的非回归测试用例
        D3 = np.array([[4, 3], [7, 5]])
        D4 = np.array([[4, 3], [7, 0]])
        D5 = np.array([[5, 5, 3], [4, 9, 10], [3, 4, 9]])

        # 对于每个数组 D，创建稀疏矩阵 mat，并进行以下测试
        for D in [D1, D2, D3, D4, D5]:
            mat = self.spcreator(D)
            # 如果 mat 不是 _data._minmax_mixin 的实例，则继续下一次循环
            if not isinstance(mat, _data._minmax_mixin):
                continue

            # 断言稀疏矩阵 mat 的 argmax() 方法结果与 np.argmax(D) 相等
            assert_equal(mat.argmax(), np.argmax(D))
            # 断言稀疏矩阵 mat 的 argmin() 方法结果与 np.argmin(D) 相等

            assert_equal(mat.argmin(), np.argmin(D))

            # 断言稀疏矩阵 mat 沿着 axis=0 的 argmax() 方法结果与 np.argmax(D, axis=0) 相等
            assert_equal(mat.argmax(axis=0),
                         asmatrix(np.argmax(D, axis=0)))
            # 断言稀疏矩阵 mat 沿着 axis=0 的 argmin() 方法结果与 np.argmin(D, axis=0) 相等
            assert_equal(mat.argmin(axis=0),
                         asmatrix(np.argmin(D, axis=0)))

            # 断言稀疏矩阵 mat 沿着 axis=1 的 argmax() 方法结果与 np.argmax(D, axis=1).reshape(-1, 1) 相等
            assert_equal(mat.argmax(axis=1),
                         asmatrix(np.argmax(D, axis=1).reshape(-1, 1)))
            # 断言稀疏矩阵 mat 沿着 axis=1 的 argmin() 方法结果与 np.argmin(D, axis=1).reshape(-1, 1) 相等
            assert_equal(mat.argmin(axis=1),
                         asmatrix(np.argmin(D, axis=1).reshape(-1, 1)))

        # 创建空的 D1 和 D2 矩阵
        D1 = np.empty((0, 5))
        D2 = np.empty((5, 0))

        # 对于每个 axis 值，创建稀疏矩阵 mat，并断言在 axis 为 None 或 0 时会引发 ValueError
        for axis in [None, 0]:
            mat = self.spcreator(D1)
            assert_raises(ValueError, mat.argmax, axis=axis)
            assert_raises(ValueError, mat.argmin, axis=axis)

        # 对于每个 axis 值，创建稀疏矩阵 mat，并断言在 axis 为 None 或 1 时会引发 ValueError
        for axis in [None, 1]:
            mat = self.spcreator(D2)
            assert_raises(ValueError, mat.argmax, axis=axis)
            assert_raises(ValueError, mat.argmin, axis=axis)
class _TestGetNnzAxis:
    # 定义测试类 _TestGetNnzAxis
    def test_getnnz_axis(self):
        # 定义测试方法 test_getnnz_axis，self 指向当前实例
        dat = array([[0, 2],
                     [3, 5],
                     [-6, 9]])
        # 创建一个 NumPy 数组 dat，包含整数值
        bool_dat = dat.astype(bool)
        # 将 dat 转换为布尔类型的数组 bool_dat
        datsp = self.spcreator(dat)
        # 使用 self.spcreator 方法创建稀疏矩阵 datsp

        accepted_return_dtypes = (np.int32, np.int64)
        # 接受的返回数据类型为 np.int32 和 np.int64 的元组

        # 检查沿着 None 轴的布尔类型数组求和结果是否与稀疏矩阵 datsp 相等
        assert_array_equal(bool_dat.sum(axis=None), datsp.getnnz(axis=None))
        # 检查整个布尔类型数组的求和结果是否与稀疏矩阵 datsp 相等
        assert_array_equal(bool_dat.sum(), datsp.getnnz())
        # 检查沿着 axis=0 轴的布尔类型数组求和结果是否与稀疏矩阵 datsp 相等
        assert_array_equal(bool_dat.sum(axis=0), datsp.getnnz(axis=0))
        # 检查沿着 axis=0 轴的稀疏矩阵 datsp 的非零元素数量的数据类型是否为接受的类型之一
        assert_in(datsp.getnnz(axis=0).dtype, accepted_return_dtypes)
        # 检查沿着 axis=1 轴的布尔类型数组求和结果是否与稀疏矩阵 datsp 相等
        assert_array_equal(bool_dat.sum(axis=1), datsp.getnnz(axis=1))
        # 检查沿着 axis=1 轴的稀疏矩阵 datsp 的非零元素数量的数据类型是否为接受的类型之一
        assert_in(datsp.getnnz(axis=1).dtype, accepted_return_dtypes)
        # 检查沿着 axis=-2 轴的布尔类型数组求和结果是否与稀疏矩阵 datsp 相等
        assert_array_equal(bool_dat.sum(axis=-2), datsp.getnnz(axis=-2))
        # 检查沿着 axis=-2 轴的稀疏矩阵 datsp 的非零元素数量的数据类型是否为接受的类型之一
        assert_in(datsp.getnnz(axis=-2).dtype, accepted_return_dtypes)
        # 检查沿着 axis=-1 轴的布尔类型数组求和结果是否与稀疏矩阵 datsp 相等
        assert_array_equal(bool_dat.sum(axis=-1), datsp.getnnz(axis=-1))
        # 检查沿着 axis=-1 轴的稀疏矩阵 datsp 的非零元素数量的数据类型是否为接受的类型之一

        # 检查在 axis=2 时是否引发 ValueError 异常
        assert_raises(ValueError, datsp.getnnz, axis=2)


#------------------------------------------------------------------------------
# Tailored base class for generic tests
#------------------------------------------------------------------------------

def _possibly_unimplemented(cls, require=True):
    """
    Construct a class that either runs tests as usual (require=True),
    or each method skips if it encounters a common error.
    """
    if require:
        return cls
    else:
        def wrap(fc):
            @functools.wraps(fc)
            def wrapper(*a, **kw):
                try:
                    return fc(*a, **kw)
                except (NotImplementedError, TypeError, ValueError,
                        IndexError, AttributeError):
                    raise pytest.skip("feature not implemented")

            return wrapper

        new_dict = dict(cls.__dict__)
        for name, func in cls.__dict__.items():
            if name.startswith('test_'):
                new_dict[name] = wrap(func)
        return type(cls.__name__ + "NotImplemented",
                    cls.__bases__,
                    new_dict)


def sparse_test_class(getset=True, slicing=True, slicing_assign=True,
                      fancy_indexing=True, fancy_assign=True,
                      fancy_multidim_indexing=True, fancy_multidim_assign=True,
                      minmax=True, nnz_axis=True):
    """
    Construct a base class, optionally converting some of the tests in
    the suite to check that the feature is not implemented.
    """
    # 定义基类列表，包括多个测试类和可能未实现的测试类
    bases = (_TestCommon,                      # 通用测试类
             _possibly_unimplemented(_TestGetSet, getset),  # 可能未实现的获取/设置测试类
             _TestSolve,                       # 解决测试类
             _TestInplaceArithmetic,           # 就地算术操作测试类
             _TestArithmetic,                  # 算术操作测试类
             _possibly_unimplemented(_TestSlicing, slicing),  # 可能未实现的切片测试类
             _possibly_unimplemented(_TestSlicingAssign, slicing_assign),  # 可能未实现的切片赋值测试类
             _possibly_unimplemented(_TestFancyIndexing, fancy_indexing),  # 可能未实现的高级索引测试类
             _possibly_unimplemented(_TestFancyIndexingAssign, fancy_assign),  # 可能未实现的高级索引赋值测试类
             _possibly_unimplemented(_TestFancyMultidim,  # 可能未实现的多维高级索引测试类
                                     fancy_indexing and fancy_multidim_indexing),
             _possibly_unimplemented(_TestFancyMultidimAssign,  # 可能未实现的多维高级索引赋值测试类
                                     fancy_multidim_assign and fancy_assign),
             _possibly_unimplemented(_TestMinMax, minmax),  # 可能未实现的最小/最大值测试类
             _possibly_unimplemented(_TestGetNnzAxis, nnz_axis))  # 可能未实现的获取非零元素数轴测试类
    
    # 检查测试类名称不冲突
    names = {}
    for cls in bases:
        for name in cls.__dict__:
            if not name.startswith('test_'):  # 如果名称不以'test_'开头则跳过
                continue
            old_cls = names.get(name)
            if old_cls is not None:
                # 如果发现同名测试方法，则抛出异常
                raise ValueError(f"Test class {cls.__name__} overloads test "
                                 f"{name} defined in {old_cls.__name__}")
            names[name] = cls
    
    # 返回由给定基类生成的新类型对象
    return type("TestBase", bases, {})
#------------------------------------------------------------------------------
# Matrix class based tests
#------------------------------------------------------------------------------

class TestCSR(sparse_test_class()):
    # 定义一个测试类 TestCSR，继承自 sparse_test_class()

    @classmethod
    def spcreator(cls, *args, **kwargs):
        # 定义一个类方法 spcreator，用于创建 CSR 矩阵对象
        with suppress_warnings() as sup:
            # 使用 suppress_warnings 上下文管理器来抑制特定警告
            sup.filter(SparseEfficiencyWarning, "Changing the sparsity structure")
            # 过滤 SparseEfficiencyWarning 警告消息 "Changing the sparsity structure"
            return csr_matrix(*args, **kwargs)
            # 返回一个 CSR 矩阵对象，使用传入的参数 args 和 kwargs

    math_dtypes = [np.bool_, np.int_, np.float64, np.complex128]
    # 定义数学数据类型列表，包括布尔型、整型、双精度浮点型和复数双精度浮点型

    def test_constructor1(self):
        # 测试用例1：测试 CSR 矩阵的构造方法1
        b = array([[0, 4, 0],
                   [3, 0, 0],
                   [0, 2, 0]], 'd')
        # 创建一个双精度浮点型数组 b，表示一个稀疏矩阵
        bsp = csr_matrix(b)
        # 使用数组 b 创建一个 CSR 矩阵 bsp

        assert_array_almost_equal(bsp.data,[4,3,2])
        # 检查 bsp 矩阵的数据部分是否准确
        assert_array_equal(bsp.indices,[1,0,1])
        # 检查 bsp 矩阵的列索引部分是否准确
        assert_array_equal(bsp.indptr,[0,1,2,3])
        # 检查 bsp 矩阵的指针索引部分是否准确
        assert_equal(bsp.getnnz(),3)
        # 检查 bsp 矩阵的非零元素个数是否为3
        assert_equal(bsp.format,'csr')
        # 检查 bsp 矩阵的格式是否为 CSR
        assert_array_equal(bsp.toarray(), b)
        # 检查 bsp 矩阵转为数组后是否与原始数组 b 相同

    def test_constructor2(self):
        # 测试用例2：测试 CSR 矩阵的构造方法2
        b = zeros((6,6),'d')
        # 创建一个 6x6 的双精度浮点型零矩阵 b
        b[3,4] = 5
        # 设置矩阵 b 中第(3,4)个元素为 5
        bsp = csr_matrix(b)
        # 使用数组 b 创建一个 CSR 矩阵 bsp

        assert_array_almost_equal(bsp.data,[5])
        # 检查 bsp 矩阵的数据部分是否准确
        assert_array_equal(bsp.indices,[4])
        # 检查 bsp 矩阵的列索引部分是否准确
        assert_array_equal(bsp.indptr,[0,0,0,0,1,1,1])
        # 检查 bsp 矩阵的指针索引部分是否准确
        assert_array_almost_equal(bsp.toarray(), b)
        # 检查 bsp 矩阵转为数组后是否与原始数组 b 相同

    def test_constructor3(self):
        # 测试用例3：测试 CSR 矩阵的构造方法3
        b = array([[1, 0],
                   [0, 2],
                   [3, 0]], 'd')
        # 创建一个双精度浮点型数组 b，表示一个稀疏矩阵
        bsp = csr_matrix(b)
        # 使用数组 b 创建一个 CSR 矩阵 bsp

        assert_array_almost_equal(bsp.data,[1,2,3])
        # 检查 bsp 矩阵的数据部分是否准确
        assert_array_equal(bsp.indices,[0,1,0])
        # 检查 bsp 矩阵的列索引部分是否准确
        assert_array_equal(bsp.indptr,[0,1,2,3])
        # 检查 bsp 矩阵的指针索引部分是否准确
        assert_array_almost_equal(bsp.toarray(), b)
        # 检查 bsp 矩阵转为数组后是否与原始数组 b 相同

    def test_constructor4(self):
        # 测试用例4：测试 CSR 矩阵的构造方法4
        # using (data, ij) format
        row = array([2, 3, 1, 3, 0, 1, 3, 0, 2, 1, 2])
        col = array([0, 1, 0, 0, 1, 1, 2, 2, 2, 2, 1])
        data = array([6., 10., 3., 9., 1., 4.,
                              11., 2., 8., 5., 7.])
        # 定义行、列和数据数组，用于构建 CSR 矩阵

        ij = vstack((row,col))
        # 垂直堆叠行和列数组，构成索引数组 ij
        csr = csr_matrix((data,ij),(4,3))
        # 使用数据和索引数组构建一个 CSR 矩阵 csr

        assert_array_equal(arange(12).reshape(4, 3), csr.toarray())
        # 检查 csr 矩阵转为数组后是否与预期的数组相同

        # using Python lists and a specified dtype
        csr = csr_matrix(([2**63 + 1, 1], ([0, 1], [0, 1])), dtype=np.uint64)
        # 使用 Python 列表和指定的数据类型创建 CSR 矩阵 csr
        dense = array([[2**63 + 1, 0], [0, 1]], dtype=np.uint64)
        # 创建一个双精度整型数组 dense

        assert_array_equal(dense, csr.toarray())
        # 检查 csr 矩阵转为数组后是否与预期的数组 dense 相同

        # with duplicates (should sum the duplicates)
        csr = csr_matrix(([1,1,1,1], ([0,2,2,0], [0,1,1,0])))
        # 使用数据和索引数组创建一个 CSR 矩阵 csr，包含重复值

        assert csr.nnz == 2
        # 检查 csr 矩阵的非零元素个数是否为 2

    def test_constructor5(self):
        # 测试用例5：测试 CSR 矩阵的构造方法5
        # infer dimensions from arrays
        indptr = array([0,1,3,3])
        indices = array([0,5,1,2])
        data = array([1,2,3,4])
        # 定义指针、索引和数据数组，用于构建 CSR 矩阵

        csr = csr_matrix((data, indices, indptr))
        # 使用数据、索引和指针数组构建一个 CSR 矩阵 csr

        assert_array_equal(csr.shape,(3,6))
        # 检查 csr 矩阵的形状是否为 (3, 6)
    def test_constructor6(self):
        # 推断维度和数据类型从列表中
        indptr = [0, 1, 3, 3]  # CSR矩阵的行指针
        indices = [0, 5, 1, 2]  # 非零元素的列索引
        data = [1, 2, 3, 4]  # 非零元素的值
        csr = csr_matrix((data, indices, indptr))  # 创建CSR稀疏矩阵对象
        assert_array_equal(csr.shape, (3,6))  # 验证稀疏矩阵的形状
        assert_(np.issubdtype(csr.dtype, np.signedinteger))  # 验证稀疏矩阵的数据类型是否为有符号整数

    def test_constructor_smallcol(self):
        # 不需要int64类型的索引
        data = arange(6) + 1  # 非零元素的值
        col = array([1, 2, 1, 0, 0, 2], dtype=np.int64)  # 非零元素的列索引
        ptr = array([0, 2, 4, 6], dtype=np.int64)  # CSR矩阵的行指针

        a = csr_matrix((data, col, ptr), shape=(3, 3))  # 创建CSR稀疏矩阵对象

        b = array([[0, 1, 2],
                   [4, 3, 0],
                   [5, 0, 6]], 'd')  # 预期的稀疏矩阵数组表示

        assert_equal(a.indptr.dtype, np.dtype(np.int32))  # 验证稀疏矩阵行指针的数据类型
        assert_equal(a.indices.dtype, np.dtype(np.int32))  # 验证稀疏矩阵非零元素列索引的数据类型
        assert_array_equal(a.toarray(), b)  # 验证稀疏矩阵的稠密表示与预期的数组b相等

    def test_constructor_largecol(self):
        # 需要int64类型的索引
        data = arange(6) + 1  # 非零元素的值
        large = np.iinfo(np.int32).max + 100  # 大于int32范围的整数
        col = array([0, 1, 2, large, large+1, large+2], dtype=np.int64)  # 非零元素的列索引
        ptr = array([0, 2, 4, 6], dtype=np.int64)  # CSR矩阵的行指针

        a = csr_matrix((data, col, ptr))  # 创建CSR稀疏矩阵对象

        assert_equal(a.indptr.dtype, np.dtype(np.int64))  # 验证稀疏矩阵行指针的数据类型
        assert_equal(a.indices.dtype, np.dtype(np.int64))  # 验证稀疏矩阵非零元素列索引的数据类型
        assert_array_equal(a.shape, (3, max(col)+1))  # 验证稀疏矩阵的形状与预期的最大列索引相等

    def test_sort_indices(self):
        data = arange(5)  # 非零元素的值
        indices = array([7, 2, 1, 5, 4])  # 非零元素的列索引
        indptr = array([0, 3, 5])  # CSR矩阵的行指针
        asp = csr_matrix((data, indices, indptr), shape=(2,10))  # 创建CSR稀疏矩阵对象
        bsp = asp.copy()  # 复制稀疏矩阵对象asp

        asp.sort_indices()  # 对稀疏矩阵的列索引排序

        assert_array_equal(asp.indices,[1, 2, 7, 4, 5])  # 验证排序后的列索引
        assert_array_equal(asp.toarray(), bsp.toarray())  # 验证排序后的稀疏矩阵与原始的稀疏矩阵bsp相等

    def test_eliminate_zeros(self):
        data = array([1, 0, 0, 0, 2, 0, 3, 0])  # 非零元素的值
        indices = array([1, 2, 3, 4, 5, 6, 7, 8])  # 非零元素的列索引
        indptr = array([0, 3, 8])  # CSR矩阵的行指针
        asp = csr_matrix((data, indices, indptr), shape=(2,10))  # 创建CSR稀疏矩阵对象
        bsp = asp.copy()  # 复制稀疏矩阵对象asp

        asp.eliminate_zeros()  # 清除稀疏矩阵中的零元素

        assert_array_equal(asp.nnz, 3)  # 验证清除零元素后的非零元素个数
        assert_array_equal(asp.data,[1, 2, 3])  # 验证清除零元素后的非零元素的值
        assert_array_equal(asp.toarray(), bsp.toarray())  # 验证清除零元素后的稀疏矩阵与原始的稀疏矩阵bsp相等

    def test_ufuncs(self):
        X = csr_matrix(np.arange(20).reshape(4, 5) / 20.)  # 创建CSR稀疏矩阵对象X
        for f in ["sin", "tan", "arcsin", "arctan", "sinh", "tanh",
                  "arcsinh", "arctanh", "rint", "sign", "expm1", "log1p",
                  "deg2rad", "rad2deg", "floor", "ceil", "trunc", "sqrt"]:
            assert_equal(hasattr(csr_matrix, f), True)  # 验证CSR稀疏矩阵类是否具有当前ufunc函数f
            X2 = getattr(X, f)()  # 对X中的数据应用ufunc函数f
            assert_equal(X.shape, X2.shape)  # 验证结果X2的形状与X相等
            assert_array_equal(X.indices, X2.indices)  # 验证结果X2的列索引与X相等
            assert_array_equal(X.indptr, X2.indptr)  # 验证结果X2的行指针与X相等
            assert_array_equal(X2.toarray(), getattr(np, f)(X.toarray()))  # 验证结果X2的稠密表示与numpy库中的相应函数np.f(X的稠密表示)相等
    # 定义测试函数，测试未排序的稀疏矩阵加法
    def test_unsorted_arithmetic(self):
        # 创建包含0到4的数组
        data = arange(5)
        # 创建包含索引[7, 2, 1, 5, 4]的数组
        indices = array([7, 2, 1, 5, 4])
        # 创建包含指针[0, 3, 5]的数组
        indptr = array([0, 3, 5])
        # 创建一个2行10列的Compressed Sparse Row (CSR)矩阵，使用data, indices, indptr作为数据
        asp = csr_matrix((data, indices, indptr), shape=(2,10))
        # 创建包含0到5的数组
        data = arange(6)
        # 创建包含索引[8, 1, 5, 7, 2, 4]的数组
        indices = array([8, 1, 5, 7, 2, 4])
        # 创建包含指针[0, 2, 6]的数组
        indptr = array([0, 2, 6])
        # 创建一个2行10列的Compressed Sparse Row (CSR)矩阵，使用data, indices, indptr作为数据
        bsp = csr_matrix((data, indices, indptr), shape=(2,10))
        # 断言两个稀疏矩阵相加后的稠密表示等于它们各自的稠密表示相加的结果
        assert_equal((asp + bsp).toarray(), asp.toarray() + bsp.toarray())

    # 定义测试函数，测试索引广播（fancy indexing broadcast）模式
    def test_fancy_indexing_broadcast(self):
        # 创建二维数组[[1], [2], [3]]
        I = np.array([[1], [2], [3]])
        # 创建一维数组[3, 4, 2]
        J = np.array([3, 4, 2])

        # 设置随机种子为1234
        np.random.seed(1234)
        # 创建一个5行7列的随机矩阵并转换为矩阵形式
        D = asmatrix(np.random.rand(5, 7))
        # 使用给定的稀疏矩阵创建函数创建稀疏矩阵S
        S = self.spcreator(D)

        # 对稀疏矩阵S进行索引操作，使用I, J作为索引
        SIJ = S[I,J]
        # 如果SIJ是稀疏矩阵，则将其转换为稠密数组形式
        if issparse(SIJ):
            SIJ = SIJ.toarray()
        # 断言SIJ与矩阵D的相同索引位置处的元素相等
        assert_equal(SIJ, D[I,J])

    # 定义测试函数，测试稀疏矩阵是否具有已排序索引的特性
    def test_has_sorted_indices(self):
        # 确保has_sorted_indices方法在sort_indices之后能够缓存已排序状态
        sorted_inds = np.array([0, 1])
        unsorted_inds = np.array([1, 0])
        data = np.array([1, 1])
        indptr = np.array([0, 2])

        # 创建包含已排序索引的CSR矩阵M
        M = csr_matrix((data, sorted_inds, indptr)).copy()
        assert_equal(True, M.has_sorted_indices)
        assert isinstance(M.has_sorted_indices, bool)

        # 创建包含未排序索引的CSR矩阵M
        M = csr_matrix((data, unsorted_inds, indptr)).copy()
        assert_equal(False, M.has_sorted_indices)

        # 通过排序手动设置has_sorted_indices为True
        M.sort_indices()
        assert_equal(True, M.has_sorted_indices)
        assert_array_equal(M.indices, sorted_inds)

        # 再次创建包含未排序索引的CSR矩阵M
        M = csr_matrix((data, unsorted_inds, indptr)).copy()
        # 尽管底层未排序，但手动设置has_sorted_indices为True
        M.has_sorted_indices = True
        assert_equal(True, M.has_sorted_indices)
        assert_array_equal(M.indices, unsorted_inds)

        # 确保在has_sorted_indices为True时跳过排序操作
        M.sort_indices()
        assert_array_equal(M.indices, unsorted_inds)
    def test_has_canonical_format(self):
        "Ensure has_canonical_format memoizes state for sum_duplicates"

        M = csr_matrix((np.array([2]), np.array([0]), np.array([0, 1])))
        assert_equal(True, M.has_canonical_format)  # 检查稀疏矩阵 M 是否具有规范格式

        indices = np.array([0, 0])  # contains duplicate
        data = np.array([1, 1])
        indptr = np.array([0, 2])

        M = csr_matrix((data, indices, indptr)).copy()
        assert_equal(False, M.has_canonical_format)  # 检查稀疏矩阵 M 是否不具有规范格式
        assert isinstance(M.has_canonical_format, bool)  # 确保 M.has_canonical_format 的类型为布尔值

        # set by deduplicating
        M.sum_duplicates()  # 对 M 进行重复元素求和，设置其具有规范格式
        assert_equal(True, M.has_canonical_format)  # 检查稀疏矩阵 M 是否具有规范格式
        assert_equal(1, len(M.indices))  # 检查 M.indices 的长度是否为 1

        M = csr_matrix((data, indices, indptr)).copy()
        # set manually (although underlyingly duplicated)
        M.has_canonical_format = True  # 手动设置 M 具有规范格式，尽管其底层数据有重复
        assert_equal(True, M.has_canonical_format)  # 检查稀疏矩阵 M 是否具有规范格式
        assert_equal(2, len(M.indices))  # 检查 M.indices 的长度是否为 2，内容不受影响

        # ensure deduplication bypassed when has_canonical_format == True
        M.sum_duplicates()  # 确保当 M.has_canonical_format == True 时不执行重复元素求和操作
        assert_equal(2, len(M.indices))  # 检查 M.indices 的长度是否为 2，内容不受影响

    def test_scalar_idx_dtype(self):
        # Check that index dtype takes into account all parameters
        # passed to sparsetools, including the scalar ones
        indptr = np.zeros(2, dtype=np.int32)
        indices = np.zeros(0, dtype=np.int32)
        vals = np.zeros(0)
        a = csr_matrix((vals, indices, indptr), shape=(1, 2**31-1))
        b = csr_matrix((vals, indices, indptr), shape=(1, 2**31))
        ij = np.zeros((2, 0), dtype=np.int32)
        c = csr_matrix((vals, ij), shape=(1, 2**31-1))
        d = csr_matrix((vals, ij), shape=(1, 2**31))
        e = csr_matrix((1, 2**31-1))
        f = csr_matrix((1, 2**31))
        assert_equal(a.indptr.dtype, np.int32)  # 检查稀疏矩阵 a 的 indptr 数组的数据类型是否为 np.int32
        assert_equal(b.indptr.dtype, np.int64)  # 检查稀疏矩阵 b 的 indptr 数组的数据类型是否为 np.int64
        assert_equal(c.indptr.dtype, np.int32)  # 检查稀疏矩阵 c 的 indptr 数组的数据类型是否为 np.int32
        assert_equal(d.indptr.dtype, np.int64)  # 检查稀疏矩阵 d 的 indptr 数组的数据类型是否为 np.int64
        assert_equal(e.indptr.dtype, np.int32)  # 检查稀疏矩阵 e 的 indptr 数组的数据类型是否为 np.int32
        assert_equal(f.indptr.dtype, np.int64)  # 检查稀疏矩阵 f 的 indptr 数组的数据类型是否为 np.int64

        # These shouldn't fail
        for x in [a, b, c, d, e, f]:
            x + x  # 对稀疏矩阵 x 执行加法操作，确保不会引发错误

    def test_binop_explicit_zeros(self):
        # Check that binary ops don't introduce spurious explicit zeros.
        # See gh-9619 for context.
        a = csr_matrix([[0, 1, 0]])
        b = csr_matrix([[1, 1, 0]])
        assert (a + b).nnz == 2  # 检查稀疏矩阵 a 和 b 相加后非零元素的数量为 2
        assert a.multiply(b).nnz == 1  # 检查稀疏矩阵 a 和 b 逐元素相乘后非零元素的数量为 1
TestCSR.init_class()  # 初始化 TestCSR 类

# 定义一个名为 TestCSC 的测试类，继承自 sparse_test_class()
class TestCSC(sparse_test_class()):

    # 类方法，用于创建 csc_matrix 实例
    @classmethod
    def spcreator(cls, *args, **kwargs):
        # 使用 suppress_warnings 上下文管理器来忽略 SparseEfficiencyWarning
        with suppress_warnings() as sup:
            # 过滤掉 SparseEfficiencyWarning 中包含 "Changing the sparsity structure" 的警告
            sup.filter(SparseEfficiencyWarning, "Changing the sparsity structure")
            # 返回 csc_matrix 的实例
            return csc_matrix(*args, **kwargs)

    # 数学数据类型列表
    math_dtypes = [np.bool_, np.int_, np.float64, np.complex128]

    # 测试构造函数1
    def test_constructor1(self):
        # 创建一个二维数组 b
        b = array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 2, 0, 3]], 'd')
        # 将二维数组 b 转换为 csc_matrix 格式
        bsp = csc_matrix(b)
        # 断言 csc_matrix 的数据与预期的数据近似相等
        assert_array_almost_equal(bsp.data,[1,2,1,3])
        # 断言 csc_matrix 的列索引与预期的索引相等
        assert_array_equal(bsp.indices,[0,2,1,2])
        # 断言 csc_matrix 的行指针与预期的指针相等
        assert_array_equal(bsp.indptr,[0,1,2,3,4])
        # 断言 csc_matrix 的非零元素个数与预期相等
        assert_equal(bsp.getnnz(),4)
        # 断言 csc_matrix 的形状与原始数组 b 的形状相等
        assert_equal(bsp.shape,b.shape)
        # 断言 csc_matrix 的格式为 'csc'
        assert_equal(bsp.format,'csc')

    # 测试构造函数2
    def test_constructor2(self):
        # 创建一个全零的6x6数组 b，并设置其特定位置的值
        b = zeros((6,6),'d')
        b[2,4] = 5
        # 将数组 b 转换为 csc_matrix 格式
        bsp = csc_matrix(b)
        # 断言 csc_matrix 的数据与预期的数据近似相等
        assert_array_almost_equal(bsp.data,[5])
        # 断言 csc_matrix 的列索引与预期的索引相等
        assert_array_equal(bsp.indices,[2])
        # 断言 csc_matrix 的行指针与预期的指针相等
        assert_array_equal(bsp.indptr,[0,0,0,0,0,1,1])

    # 测试构造函数3
    def test_constructor3(self):
        # 创建一个二维数组 b
        b = array([[1, 0], [0, 0], [0, 2]], 'd')
        # 将二维数组 b 转换为 csc_matrix 格式
        bsp = csc_matrix(b)
        # 断言 csc_matrix 的数据与预期的数据近似相等
        assert_array_almost_equal(bsp.data,[1,2])
        # 断言 csc_matrix 的列索引与预期的索引相等
        assert_array_equal(bsp.indices,[0,2])
        # 断言 csc_matrix 的行指针与预期的指针相等
        assert_array_equal(bsp.indptr,[0,1,2])

    # 测试构造函数4
    def test_constructor4(self):
        # 使用 (data, ij) 格式创建 csc_matrix
        row = array([2, 3, 1, 3, 0, 1, 3, 0, 2, 1, 2])
        col = array([0, 1, 0, 0, 1, 1, 2, 2, 2, 2, 1])
        data = array([6., 10., 3., 9., 1., 4., 11., 2., 8., 5., 7.])

        ij = vstack((row,col))
        csc = csc_matrix((data,ij),(4,3))
        # 断言 csc_matrix 转换为数组后与预期的数组相等
        assert_array_equal(arange(12).reshape(4, 3), csc.toarray())

        # 包含重复值的情况（应该对重复值求和）
        csc = csc_matrix(([1,1,1,1], ([0,2,2,0], [0,1,1,0])))
        # 断言 csc_matrix 的非零元素个数与预期相等
        assert csc.nnz == 2

    # 测试构造函数5
    def test_constructor5(self):
        # 从数组推断维度
        indptr = array([0,1,3,3])
        indices = array([0,5,1,2])
        data = array([1,2,3,4])
        csc = csc_matrix((data, indices, indptr))
        # 断言 csc_matrix 的形状与预期相等
        assert_array_equal(csc.shape,(6,3))

    # 测试构造函数6
    def test_constructor6(self):
        # 从列表推断维度和数据类型
        indptr = [0, 1, 3, 3]
        indices = [0, 5, 1, 2]
        data = [1, 2, 3, 4]
        csc = csc_matrix((data, indices, indptr))
        # 断言 csc_matrix 的形状与预期相等
        assert_array_equal(csc.shape,(6,3))
        # 断言 csc_matrix 的数据类型为有符号整数
        assert_(np.issubdtype(csc.dtype, np.signedinteger))

    # 测试消除零元素
    def test_eliminate_zeros(self):
        data = array([1, 0, 0, 0, 2, 0, 3, 0])
        indices = array([1, 2, 3, 4, 5, 6, 7, 8])
        indptr = array([0, 3, 8])
        asp = csc_matrix((data, indices, indptr), shape=(10,2))
        bsp = asp.copy()
        # 消除 csc_matrix 中的零元素
        asp.eliminate_zeros()
        # 断言消除零元素后 csc_matrix 的非零元素个数与预期相等
        assert_array_equal(asp.nnz, 3)
        # 断言消除零元素后 csc_matrix 的数据与预期的数据相等
        assert_array_equal(asp.data,[1, 2, 3])
        # 断言消除零元素后 csc_matrix 转换为数组与未消除零元素前的数组相等
        assert_array_equal(asp.toarray(), bsp.toarray())
    # 定义一个测试函数，用于测试稀疏矩阵的排序功能
    def test_sort_indices(self):
        # 创建一维数组，内容为 [0, 1, 2, 3, 4]
        data = arange(5)
        # 创建一维数组，内容为 [7, 2, 1, 5, 4]
        row = array([7, 2, 1, 5, 4])
        # 创建一维数组，内容为 [0, 3, 5]
        ptr = [0, 3, 5]
        # 使用稀疏矩阵的构造方法创建 csc_matrix 对象 asp
        asp = csc_matrix((data, row, ptr), shape=(10,2))
        # 创建 bsp 作为 asp 的副本
        bsp = asp.copy()
        # 调用稀疏矩阵的排序方法，按照索引排序
        asp.sort_indices()
        # 断言排序后的 indices 数组是否与预期相等
        assert_array_equal(asp.indices,[1, 2, 7, 4, 5])
        # 断言排序后的稀疏矩阵转换成稠密数组后是否与 bsp 相等
        assert_array_equal(asp.toarray(), bsp.toarray())

    # 定义一个测试函数，用于测试稀疏矩阵支持的数学函数（ufuncs）
    def test_ufuncs(self):
        # 创建一个稀疏矩阵 X，内容为 np.arange(21).reshape(7, 3) / 21.
        X = csc_matrix(np.arange(21).reshape(7, 3) / 21.)
        # 遍历一系列数学函数名字的列表
        for f in ["sin", "tan", "arcsin", "arctan", "sinh", "tanh",
                  "arcsinh", "arctanh", "rint", "sign", "expm1", "log1p",
                  "deg2rad", "rad2deg", "floor", "ceil", "trunc", "sqrt"]:
            # 断言 csr_matrix 类是否具有当前数学函数的属性
            assert_equal(hasattr(csr_matrix, f), True)
            # 调用当前稀疏矩阵 X 的数学函数，并获取结果 X2
            X2 = getattr(X, f)()
            # 断言 X 和 X2 的形状是否相同
            assert_equal(X.shape, X2.shape)
            # 断言 X 和 X2 的 indices 数组是否相同
            assert_array_equal(X.indices, X2.indices)
            # 断言 X 和 X2 的 indptr 数组是否相同
            assert_array_equal(X.indptr, X2.indptr)
            # 断言 X2 转换成稠密数组后是否与 np 函数的结果相等
            assert_array_equal(X2.toarray(), getattr(np, f)(X.toarray()))

    # 定义一个测试函数，用于测试稀疏矩阵在未排序情况下的算术运算
    def test_unsorted_arithmetic(self):
        # 创建两组数据数组和索引数组，构造两个 csc_matrix 对象 asp 和 bsp
        data = arange(5)
        indices = array([7, 2, 1, 5, 4])
        indptr = array([0, 3, 5])
        asp = csc_matrix((data, indices, indptr), shape=(10,2))
        data = arange(6)
        indices = array([8, 1, 5, 7, 2, 4])
        indptr = array([0, 2, 6])
        bsp = csc_matrix((data, indices, indptr), shape=(10,2))
        # 断言稀疏矩阵 asp 和 bsp 的加法结果是否与其稠密数组相加结果相等
        assert_equal((asp + bsp).toarray(), asp.toarray() + bsp.toarray())

    # 定义一个测试函数，用于测试稀疏矩阵的高级索引与广播功能
    def test_fancy_indexing_broadcast(self):
        # 创建两个数组 I 和 J 作为高级索引，以及一个随机数组 D
        I = np.array([[1], [2], [3]])
        J = np.array([3, 4, 2])
        np.random.seed(1234)
        D = asmatrix(np.random.rand(5, 7))
        # 使用自定义方法 spcreator 创建稀疏矩阵 S
        S = self.spcreator(D)
        # 使用高级索引 I 和 J 访问稀疏矩阵 S 的元素 SIJ
        SIJ = S[I,J]
        # 如果 SIJ 是稀疏矩阵，则转换成稠密数组
        if issparse(SIJ):
            SIJ = SIJ.toarray()
        # 断言 SIJ 的值与矩阵 D 中对应索引的值是否相等
        assert_equal(SIJ, D[I,J])

    # 定义一个测试函数，用于检查稀疏矩阵的标量索引数据类型
    def test_scalar_idx_dtype(self):
        # 创建零填充的一维数组和空数组，构造多个不同形状的 csc_matrix 对象
        indptr = np.zeros(2, dtype=np.int32)
        indices = np.zeros(0, dtype=np.int32)
        vals = np.zeros(0)
        a = csc_matrix((vals, indices, indptr), shape=(2**31-1, 1))
        b = csc_matrix((vals, indices, indptr), shape=(2**31, 1))
        ij = np.zeros((2, 0), dtype=np.int32)
        c = csc_matrix((vals, ij), shape=(2**31-1, 1))
        d = csc_matrix((vals, ij), shape=(2**31, 1))
        e = csr_matrix((1, 2**31-1))
        f = csr_matrix((1, 2**31))
        # 断言各稀疏矩阵的 indptr 数组的数据类型是否符合预期
        assert_equal(a.indptr.dtype, np.int32)
        assert_equal(b.indptr.dtype, np.int64)
        assert_equal(c.indptr.dtype, np.int32)
        assert_equal(d.indptr.dtype, np.int64)
        assert_equal(e.indptr.dtype, np.int32)
        assert_equal(f.indptr.dtype, np.int64)
        # 对所有稀疏矩阵执行加法操作，确保不会出现异常
        for x in [a, b, c, d, e, f]:
            x + x
TestCSC.init_class()



# 初始化 TestCSC 类，执行其初始化类方法
TestCSC.init_class()



class TestDOK(sparse_test_class(minmax=False, nnz_axis=False)):



# 定义 TestDOK 类，继承 sparse_test_class 类，并禁用 minmax 和 nnz_axis 参数
class TestDOK(sparse_test_class(minmax=False, nnz_axis=False)):



    spcreator = dok_matrix
    math_dtypes = [np.int_, np.float64, np.complex128]



# 定义 spcreator 属性为 dok_matrix 类，定义 math_dtypes 属性为整数、双精度浮点数和复数类型的列表
    spcreator = dok_matrix
    math_dtypes = [np.int_, np.float64, np.complex128]



    def test_mult(self):
        A = dok_matrix((10, 12))
        A[0, 3] = 10
        A[5, 6] = 20
        D = A @ A.T
        E = A @ A.T.conjugate()
        assert_array_equal(D.toarray(), E.toarray())



# 定义 test_mult 方法，在稀疏矩阵 A 上进行乘法操作，并比较结果 D 和 E 的数组表示是否相等
    def test_mult(self):
        A = dok_matrix((10, 12))
        A[0, 3] = 10
        A[5, 6] = 20
        D = A @ A.T
        E = A @ A.T.conjugate()
        assert_array_equal(D.toarray(), E.toarray())



    def test_add_nonzero(self):
        A = self.spcreator((3,2))
        A[0,1] = -10
        A[2,0] = 20
        A = A + 10
        B = array([[10, 0], [10, 10], [30, 10]])
        assert_array_equal(A.toarray(), B)

        A = A + 1j
        B = B + 1j
        assert_array_equal(A.toarray(), B)



# 定义 test_add_nonzero 方法，测试在稀疏矩阵 A 上的加法操作和复数加法操作，并进行结果的断言验证
    def test_add_nonzero(self):
        A = self.spcreator((3,2))
        A[0,1] = -10
        A[2,0] = 20
        A = A + 10
        B = array([[10, 0], [10, 10], [30, 10]])
        assert_array_equal(A.toarray(), B)

        A = A + 1j
        B = B + 1j
        assert_array_equal(A.toarray(), B)



    def test_dok_divide_scalar(self):
        A = self.spcreator((3,2))
        A[0,1] = -10
        A[2,0] = 20

        assert_array_equal((A/1j).toarray(), A.toarray()/1j)
        assert_array_equal((A/9).toarray(), A.toarray()/9)



# 定义 test_dok_divide_scalar 方法，测试稀疏矩阵 A 对标量进行除法操作，并进行结果的断言验证
    def test_dok_divide_scalar(self):
        A = self.spcreator((3,2))
        A[0,1] = -10
        A[2,0] = 20

        assert_array_equal((A/1j).toarray(), A.toarray()/1j)
        assert_array_equal((A/9).toarray(), A.toarray()/9)



    def test_convert(self):
        # Test provided by Andrew Straw.  Fails in SciPy <= r1477.
        (m, n) = (6, 7)
        a = dok_matrix((m, n))

        # set a few elements, but none in the last column
        a[2,1] = 1
        a[0,2] = 2
        a[3,1] = 3
        a[1,5] = 4
        a[4,3] = 5
        a[4,2] = 6

        # assert that the last column is all zeros
        assert_array_equal(a.toarray()[:,n-1], zeros(m,))

        # make sure it still works for CSC format
        csc = a.tocsc()
        assert_array_equal(csc.toarray()[:,n-1], zeros(m,))

        # now test CSR
        (m, n) = (n, m)
        b = a.transpose()
        assert_equal(b.shape, (m, n))
        # assert that the last row is all zeros
        assert_array_equal(b.toarray()[m-1,:], zeros(n,))

        # make sure it still works for CSR format
        csr = b.tocsr()
        assert_array_equal(csr.toarray()[m-1,:], zeros(n,))



# 定义 test_convert 方法，测试稀疏矩阵的转换操作，并进行多个结果的断言验证
    def test_convert(self):
        # Test provided by Andrew Straw.  Fails in SciPy <= r1477.
        (m, n) = (6, 7)
        a = dok_matrix((m, n))

        # set a few elements, but none in the last column
        a[2,1] = 1
        a[0,2] = 2
        a[3,1] = 3
        a[1,5] = 4
        a[4,3] = 5
        a[4,2] = 6

        # assert that the last column is all zeros
        assert_array_equal(a.toarray()[:,n-1], zeros(m,))

        # make sure it still works for CSC format
        csc = a.tocsc()
        assert_array_equal(csc.toarray()[:,n-1], zeros(m,))

        # now test CSR
        (m, n) = (n, m)
        b = a.transpose()
        assert_equal(b.shape, (m, n))
        # assert that the last row is all zeros
        assert_array_equal(b.toarray()[m-1,:], zeros(n,))

        # make sure it still works for CSR format
        csr = b.tocsr()
        assert_array_equal(csr.toarray()[m-1,:], zeros(n,))



    def test_ctor(self):
        # Empty ctor
        assert_raises(TypeError, dok_matrix)

        # Dense ctor
        b = array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 2, 0, 3]], 'd')
        A = dok_matrix(b)
        assert_equal(b.dtype, A.dtype)
        assert_equal(A.toarray(), b)

        # Sparse ctor
        c = csr_matrix(b)
        assert_equal(A.toarray(), c.toarray())

        data = [[0, 1, 2], [3, 0, 0]]
        d = dok_matrix(data, dtype=np.float32)
        assert_equal(d.dtype, np.float32)
        da = d.toarray()
        assert_equal(da.dtype, np.float32)
        assert_array_equal(da, data)



# 定义 test_ctor 方法，测试稀疏矩阵的构造函数，并进行多个结果的断言验证
    def test_ctor(self):
        # Empty ctor
        assert_raises(TypeError, dok_matrix)

        # Dense ctor
        b = array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 2, 0, 3]], 'd')
        A = dok_matrix(b)
        assert_equal(b.dtype, A.dtype)
        assert_equal(A.toarray(), b)

        # Sparse ctor
        c = csr_matrix(b)
        assert_equal(A.toarray(), c.toarray())

        data = [[0, 1, 2], [3, 0, 0]]
        d = dok_matrix(data, dtype=np.float32)
        assert_equal(d.dtype, np.float32)
        da = d.toarray()
        assert_equal(da.dtype, np.float32)
        assert_array_equal(da, data)



    def test_ticket1160(self):
        # Regression test for ticket #1160.
        a = dok_matrix((3,3))
        a[0,0] = 0
        # This assert would fail, because the above assignment would
        # incorrectly call __set_item__ even though the value was 0.
        assert_((0,0) not in a.keys(), "Unexpected entry (0,0) in keys")

        # Slice assignments were also affected.
        b = dok_matrix((3,3))
        b[:,0] = 0
        assert_(len(b
class TestLIL(sparse_test_class(minmax=False)):
    # 定义一个测试类 TestLIL，继承自 sparse_test_class，且禁用最小最大值的设置
    spcreator = lil_matrix
    # 设置 spcreator 属性为 lil_matrix，表示稀疏矩阵的创建器
    math_dtypes = [np.int_, np.float64, np.complex128]
    # 设置 math_dtypes 属性为包含整数、双精度浮点数和复数的列表

    def test_dot(self):
        # 定义一个测试方法 test_dot
        A = zeros((10, 10), np.complex128)
        # 创建一个 10x10 的复数类型零矩阵 A
        A[0, 3] = 10
        # 在 A 中设置索引为 (0, 3) 的元素为 10
        A[5, 6] = 20j
        # 在 A 中设置索引为 (5, 6) 的元素为 20j（复数）

        B = lil_matrix((10, 10), dtype=np.complex128)
        # 使用 lil_matrix 创建一个 10x10 的复数类型稀疏矩阵 B
        B[0, 3] = 10
        # 在 B 中设置索引为 (0, 3) 的元素为 10
        B[5, 6] = 20j
        # 在 B 中设置索引为 (5, 6) 的元素为 20j（复数）

        # TODO: properly handle this assertion on ppc64le
        # 如果当前平台不是 ppc64le，则执行以下断言
        if platform.machine() != 'ppc64le':
            assert_array_equal(A @ A.T, (B @ B.T).toarray())

        # 断言 A 乘以其转置的结果与 B 乘以其转置的结果转换为数组后相等
        assert_array_equal(A @ A.conjugate().T, (B @ B.conjugate().T).toarray())

    def test_scalar_mul(self):
        # 定义一个测试方法 test_scalar_mul
        x = lil_matrix((3, 3))
        # 创建一个 3x3 的稀疏矩阵 x
        x[0, 0] = 2
        # 在 x 中设置索引为 (0, 0) 的元素为 2

        x = x*2
        # 将 x 的每个元素乘以 2
        assert_equal(x[0, 0], 4)
        # 断言 x 的索引为 (0, 0) 的元素是否为 4

        x = x*0
        # 将 x 的每个元素乘以 0
        assert_equal(x[0, 0], 0)
        # 断言 x 的索引为 (0, 0) 的元素是否为 0

    def test_truediv_scalar(self):
        # 定义一个测试方法 test_truediv_scalar
        A = self.spcreator((3, 2))
        # 使用 spcreator 属性创建一个大小为 (3, 2) 的稀疏矩阵 A
        A[0, 1] = -10
        # 在 A 中设置索引为 (0, 1) 的元素为 -10
        A[2, 0] = 20
        # 在 A 中设置索引为 (2, 0) 的元素为 20

        assert_array_equal((A / 1j).toarray(), A.toarray() / 1j)
        # 断言 A 除以虚数单位 1j 后的结果数组与 A 除以 1j 后的稀疏矩阵转换为数组的结果相等
        assert_array_equal((A / 9).toarray(), A.toarray() / 9)
        # 断言 A 除以 9 后的结果数组与 A 除以 9 后的稀疏矩阵转换为数组的结果相等

    def test_inplace_ops(self):
        # 定义一个测试方法 test_inplace_ops
        A = lil_matrix([[0, 2, 3], [4, 0, 6]])
        # 使用 lil_matrix 创建一个稀疏矩阵 A
        B = lil_matrix([[0, 1, 0], [0, 2, 3]])
        # 使用 lil_matrix 创建一个稀疏矩阵 B

        data = {'add': (B, A + B),
                'sub': (B, A - B),
                'mul': (3, A * 3)}
        # 创建一个包含不同操作和对应预期结果的字典 data

        for op, (other, expected) in data.items():
            # 遍历 data 字典中的每个操作和对应的预期结果
            result = A.copy()
            # 复制稀疏矩阵 A 到 result
            getattr(result, f'__i{op}__')(other)
            # 使用 getattr 动态调用 result 对象的对应操作方法（例如 __iadd__、__isub__、__imul__）

            assert_array_equal(result.toarray(), expected.toarray())
            # 断言 result 转换为数组后与预期结果 expected 转换为数组后的结果相等

        # Ticket 1604.
        A = lil_matrix((1, 3), dtype=np.dtype('float64'))
        # 创建一个大小为 (1, 3) 的浮点数类型稀疏矩阵 A
        B = array([0.1, 0.1, 0.1])
        # 创建一个数组 B 包含 [0.1, 0.1, 0.1]

        A[0, :] += B
        # 将数组 B 加到 A 的第一行

        assert_array_equal(A[0, :].toarray().squeeze(), B)
        # 断言 A 的第一行转换为数组并压缩后与数组 B 相等

    def test_lil_iteration(self):
        # 定义一个测试方法 test_lil_iteration
        row_data = [[1, 2, 3], [4, 5, 6]]
        # 创建一个包含两个子列表的列表 row_data
        B = lil_matrix(array(row_data))
        # 使用 array 函数创建一个数组，然后将其转换为稀疏矩阵 B

        for r, row in enumerate(B):
            # 遍历稀疏矩阵 B 中的每一行及其索引
            assert_array_equal(row.toarray(), array(row_data[r], ndmin=2))
            # 断言当前行转换为数组后与 row_data 中对应行的数组（以二维形式）相等

    def test_lil_from_csr(self):
        # 定义一个测试方法 test_lil_from_csr，测试是否能从 csr_matrix 构造出 lil_matrix
        B = lil_matrix((10, 10))
        # 创建一个大小为 (10, 10) 的稀疏矩阵 B
        B[0, 3] = 10
        # 在 B 中设置索引为 (0, 3) 的元素为 10
        B[5, 6] = 20
        # 在 B 中设置索引为 (5, 6) 的元素为 20
        B[8, 3] = 30
        # 在 B 中设置索引为 (8, 3) 的元素为 30
        B[3, 8] = 40
        # 在 B 中设置索引为 (3, 8) 的元素为 40
        B[8, 9] = 50
        # 在 B 中设置索引为 (8, 9) 的元素为 50

        C = B.tocsr()
        # 将稀疏矩阵 B 转换为 csr_matrix 格式的 C
        D = lil_matrix(C)
        # 使用 csr_matrix C 创建一个新的 lil_matrix D

        assert_array_equal(C.toarray(), D.toarray())
        # 断言 C 转换为数组后与 D 转换为数组后的结果相等

    def test_fancy_indexing_lil(self):
        # 定义一个测试方法 test_fancy_indexing_lil，测试稀疏矩阵的高级索引
        M = asmatrix(arange(25).reshape(5, 5))
        # 创建一个 5x5 的矩
    def test_point_wise_multiply(self):
        # 创建一个 4x3 的稀疏矩阵 l
        l = lil_matrix((4, 3))
        # 设置 l 的特定元素值
        l[0, 0] = 1
        l[1, 1] = 2
        l[2, 2] = 3
        l[3, 1] = 4

        # 创建一个 4x3 的稀疏矩阵 m
        m = lil_matrix((4, 3))
        # 设置 m 的特定元素值
        m[0, 0] = 1
        m[0, 1] = 2
        m[2, 2] = 3
        m[3, 1] = 4
        m[3, 2] = 4

        # 断言 l 与 m 的逐元素乘积矩阵与 m 与 l 的逐元素乘积矩阵相等
        assert_array_equal(l.multiply(m).toarray(),
                           m.multiply(l).toarray())

        # 断言 l 与 m 的逐元素乘积矩阵的数组表示
        assert_array_equal(l.multiply(m).toarray(),
                           [[1, 0, 0],
                            [0, 0, 0],
                            [0, 0, 9],
                            [0, 16, 0]])

    def test_lil_multiply_removal(self):
        # Ticket #1427.
        # 创建一个全为 1 的 3x3 稀疏矩阵 a
        a = lil_matrix(np.ones((3, 3)))
        # 将 a 的所有元素乘以 2.0
        a *= 2.
        # 将 a 的第一行所有元素设为 0
        a[0, :] = 0
TestLIL.init_class()

# 定义一个名为 TestCOO 的测试类，继承自 sparse_test_class，禁用了 getset、slicing、slicing_assign、fancy_indexing 和 fancy_assign 功能
class TestCOO(sparse_test_class(getset=False,
                                slicing=False, slicing_assign=False,
                                fancy_indexing=False, fancy_assign=False)):
    # 设定类变量
    spcreator = coo_matrix  # 指定 spcreator 变量为 coo_matrix 类
    math_dtypes = [np.int_, np.float64, np.complex128]  # 数学数据类型包括整型、双精度浮点型和复数类型

    # 定义测试构造函数1
    def test_constructor1(self):
        # unsorted triplet format
        row = array([2, 3, 1, 3, 0, 1, 3, 0, 2, 1, 2])  # 行索引数组
        col = array([0, 1, 0, 0, 1, 1, 2, 2, 2, 2, 1])  # 列索引数组
        data = array([6., 10., 3., 9., 1., 4., 11., 2., 8., 5., 7.])  # 数据数组

        # 创建 COO 矩阵对象
        coo = coo_matrix((data,(row,col)),(4,3))
        assert_array_equal(arange(12).reshape(4, 3), coo.toarray())

        # 使用 Python 列表和指定的数据类型创建 COO 矩阵对象
        coo = coo_matrix(([2**63 + 1, 1], ([0, 1], [0, 1])), dtype=np.uint64)
        dense = array([[2**63 + 1, 0], [0, 1]], dtype=np.uint64)
        assert_array_equal(dense, coo.toarray())

    # 定义测试构造函数2
    def test_constructor2(self):
        # unsorted triplet format with duplicates (which are summed)
        row = array([0,1,2,2,2,2,0,0,2,2])  # 行索引数组
        col = array([0,2,0,2,1,1,1,0,0,2])  # 列索引数组
        data = array([2,9,-4,5,7,0,-1,2,1,-5])  # 数据数组
        coo = coo_matrix((data,(row,col)),(3,3))  # 创建 COO 矩阵对象

        mat = array([[4, -1, 0], [0, 0, 9], [-3, 7, 0]])  # 预期的稀疏矩阵密集表示

        assert_array_equal(mat, coo.toarray())

    # 定义测试构造函数3
    def test_constructor3(self):
        # empty matrix
        coo = coo_matrix((4,3))  # 创建一个空的 COO 矩阵对象

        assert_array_equal(coo.shape,(4,3))  # 验证形状是否匹配
        assert_array_equal(coo.row,[])  # 验证行索引是否为空列表
        assert_array_equal(coo.col,[])  # 验证列索引是否为空列表
        assert_array_equal(coo.data,[])  # 验证数据是否为空列表
        assert_array_equal(coo.toarray(), zeros((4, 3)))  # 验证转换为密集表示后是否为全零矩阵

    # 定义测试构造函数4
    def test_constructor4(self):
        # from dense matrix
        mat = array([[0,1,0,0],
                     [7,0,3,0],
                     [0,4,0,0]])  # 密集矩阵对象
        coo = coo_matrix(mat)  # 从密集矩阵创建 COO 矩阵对象
        assert_array_equal(coo.toarray(), mat)  # 验证转换为密集表示后是否与原始矩阵相等

        # upgrade rank 1 arrays to row matrix
        mat = array([0,1,0,0])  # 一维数组对象
        coo = coo_matrix(mat)  # 从一维数组创建 COO 矩阵对象
        assert_array_equal(coo.toarray(), mat.reshape(1, -1))  # 验证转换为密集表示后是否为行矩阵

        # error if second arg interpreted as shape (gh-9919)
        with pytest.raises(TypeError, match=r'object cannot be interpreted'):
            coo_matrix([0, 11, 22, 33], ([0, 1, 2, 3], [0, 0, 0, 0]))  # 引发类型错误异常

        # error if explicit shape arg doesn't match the dense matrix
        with pytest.raises(ValueError, match=r'inconsistent shapes'):
            coo_matrix([0, 11, 22, 33], shape=(4, 4))  # 引发形状不一致异常

    # 定义测试构造函数 data_ij_dtypeNone
    def test_constructor_data_ij_dtypeNone(self):
        data = [1]  # 数据列表
        coo = coo_matrix((data, ([0], [0])), dtype=None)  # 创建 COO 矩阵对象
        assert coo.dtype == np.array(data).dtype  # 验证数据类型是否正确

    @pytest.mark.xfail(run=False, reason='COO does not have a __getitem__')
    def test_iterator(self):
        pass

    # 定义测试函数 test_todia_all_zeros
    def test_todia_all_zeros(self):
        zeros = [[0, 0]]  # 全零列表
        dia = coo_matrix(zeros).todia()  # 创建 COO 矩阵对象并转换为 DIA 格式
        assert_array_equal(dia.toarray(), zeros)  # 验证转换为密集表示后是否为全零矩阵
    # 测试稀疏矩阵的 sum_duplicates 方法
    def test_sum_duplicates(self):
        # 创建一个 4x3 的稀疏矩阵
        coo = coo_matrix((4,3))
        # 移除矩阵中的重复元素
        coo.sum_duplicates()
        
        # 创建一个新的稀疏矩阵，包含元素及其位置信息
        coo = coo_matrix(([1,2], ([1,0], [1,0])))
        # 移除新矩阵中的重复元素
        coo.sum_duplicates()
        
        # 断言：将稀疏矩阵转为密集数组，并检查是否与预期的数组相等
        assert_array_equal(coo.toarray(), [[2,0],[0,1]])
        
        # 创建一个新的稀疏矩阵，包含元素及其位置信息
        coo = coo_matrix(([1,2], ([1,1], [1,1])))
        # 移除新矩阵中的重复元素
        coo.sum_duplicates()
        
        # 断言：将稀疏矩阵转为密集数组，并检查是否与预期的数组相等
        assert_array_equal(coo.toarray(), [[0,0],[0,3]])
        
        # 断言：检查稀疏矩阵的行索引是否与预期相等
        assert_array_equal(coo.row, [1])
        
        # 断言：检查稀疏矩阵的列索引是否与预期相等
        assert_array_equal(coo.col, [1])
        
        # 断言：检查稀疏矩阵的数据数组是否与预期相等
        assert_array_equal(coo.data, [3])

    # 测试稀疏矩阵的 todok 方法
    def test_todok_duplicates(self):
        # 创建一个稀疏矩阵，包含元素及其位置信息
        coo = coo_matrix(([1,1,1,1], ([0,2,2,0], [0,1,1,0])))
        # 转换为 dok 格式的稀疏矩阵
        dok = coo.todok()
        
        # 断言：将 dok 格式的稀疏矩阵转为密集数组，并检查是否与原始稀疏矩阵相等
        assert_array_equal(dok.toarray(), coo.toarray())

    # 测试稀疏矩阵的 tocsr 和 tocsc 方法
    def test_tocompressed_duplicates(self):
        # 创建一个稀疏矩阵，包含元素及其位置信息
        coo = coo_matrix(([1,1,1,1], ([0,2,2,0], [0,1,1,0])))
        # 转换为 csr 格式的稀疏矩阵
        csr = coo.tocsr()
        
        # 断言：检查 csr 格式的稀疏矩阵的非零元素个数加2是否等于原始稀疏矩阵的非零元素个数
        assert_equal(csr.nnz + 2, coo.nnz)
        
        # 转换为 csc 格式的稀疏矩阵
        csc = coo.tocsc()
        
        # 断言：检查 csc 格式的稀疏矩阵的非零元素个数加2是否等于原始稀疏矩阵的非零元素个数
        assert_equal(csc.nnz + 2, coo.nnz)

    # 测试稀疏矩阵的 eliminate_zeros 方法
    def test_eliminate_zeros(self):
        # 创建一个稀疏矩阵，包含数据、行索引和列索引
        data = array([1, 0, 0, 0, 2, 0, 3, 0])
        row = array([0, 0, 0, 1, 1, 1, 1, 1])
        col = array([1, 2, 3, 4, 5, 6, 7, 8])
        asp = coo_matrix((data, (row, col)), shape=(2,10))
        
        # 复制稀疏矩阵
        bsp = asp.copy()
        
        # 移除稀疏矩阵中的零元素
        asp.eliminate_zeros()
        
        # 断言：检查移除零元素后的稀疏矩阵的所有数据元素是否都不为零
        assert_((asp.data != 0).all())
        
        # 断言：将移除零元素后的稀疏矩阵转为密集数组，并检查是否与原始稀疏矩阵相等
        assert_array_equal(asp.toarray(), bsp.toarray())

    # 测试稀疏矩阵的 reshape 方法
    def test_reshape_copy(self):
        # 创建一个二维数组
        arr = [[0, 10, 0, 0], [0, 0, 0, 0], [0, 20, 30, 40]]
        # 指定新的形状
        new_shape = (2, 6)
        # 创建一个稀疏矩阵
        x = coo_matrix(arr)

        # 将稀疏矩阵重塑为新形状的稀疏矩阵，不进行复制
        y = x.reshape(new_shape)
        # 断言：检查重塑后的稀疏矩阵的数据是否与原始稀疏矩阵的数据相同
        assert_(y.data is x.data)

        # 将稀疏矩阵重塑为新形状的稀疏矩阵，不进行复制
        y = x.reshape(new_shape, copy=False)
        # 断言：检查重塑后的稀疏矩阵的数据是否与原始稀疏矩阵的数据相同
        assert_(y.data is x.data)

        # 将稀疏矩阵重塑为新形状的稀疏矩阵，进行复制
        y = x.reshape(new_shape, copy=True)
        # 断言：检查重塑后的稀疏矩阵的数据是否与原始稀疏矩阵的数据不共享内存
        assert_(not np.may_share_memory(y.data, x.data))

    # 测试大尺寸稀疏矩阵的 reshape 方法
    def test_large_dimensions_reshape(self):
        # 创建一个大尺寸的稀疏矩阵
        mat1 = coo_matrix(([1], ([3000000], [1000])), (3000001, 1001))
        mat2 = coo_matrix(([1], ([1000], [3000000])), (1001, 3000001))

        # 断言：检查重塑后的稀疏矩阵是否与另一个稀疏矩阵相等
        assert_((mat1.reshape((1001, 3000001), order='C') != mat2).nnz == 0)
        assert_((mat2.reshape((3000001, 1001), order='F') != mat1).nnz == 0)
TestCOO.init_class()

# 定义 TestDIA 类，继承自 sparse_test_class，禁用了一些特性（getset, slicing, slicing_assign, fancy_indexing, fancy_assign, minmax, nnz_axis）
class TestDIA(sparse_test_class(getset=False, slicing=False, slicing_assign=False,
                                fancy_indexing=False, fancy_assign=False,
                                minmax=False, nnz_axis=False)):
    spcreator = dia_matrix  # 设置 spcreator 属性为 dia_matrix
    math_dtypes = [np.int_, np.float64, np.complex128]  # 定义 math_dtypes 属性为包含 np.int_, np.float64, np.complex128 的列表

    # 测试构造函数1
    def test_constructor1(self):
        D = array([[1, 0, 3, 0],
                   [1, 2, 0, 4],
                   [0, 2, 3, 0],
                   [0, 0, 3, 4]])
        data = np.array([[1,2,3,4]]).repeat(3,axis=0)  # 创建数据数组
        offsets = np.array([0,-1,2])  # 创建偏移数组
        assert_equal(dia_matrix((data, offsets), shape=(4, 4)).toarray(), D)  # 断言使用给定数据和偏移创建的 dia_matrix 的数组形式与 D 相等

    # 标记为预期失败的测试用例，不运行，原因是 DIA 类没有 __getitem__ 方法
    @pytest.mark.xfail(run=False, reason='DIA does not have a __getitem__')
    def test_iterator(self):
        pass

    # 使用 64 位最大值限制装饰器的测试用例
    @with_64bit_maxval_limit(3)
    def test_setdiag_dtype(self):
        m = dia_matrix(np.eye(3))  # 创建一个对角 dia_matrix
        assert_equal(m.offsets.dtype, np.int32)  # 断言 offsets 的数据类型为 np.int32
        m.setdiag((3,), k=2)  # 在偏移 k=2 处设置对角元素为 3
        assert_equal(m.offsets.dtype, np.int32)  # 再次断言 offsets 的数据类型为 np.int32

        m = dia_matrix(np.eye(4))  # 创建一个对角 dia_matrix
        assert_equal(m.offsets.dtype, np.int64)  # 断言 offsets 的数据类型为 np.int64
        m.setdiag((3,), k=3)  # 在偏移 k=3 处设置对角元素为 3
        assert_equal(m.offsets.dtype, np.int64)  # 再次断言 offsets 的数据类型为 np.int64

    # 跳过测试，原因是 DIA 存储了额外的零元素
    @pytest.mark.skip(reason='DIA stores extra zeros')
    def test_getnnz_axis(self):
        pass

    # 测试转换方法，修复了 gh-14555 的问题
    def test_convert_gh14555(self):
        m = dia_matrix(([[1, 1, 0]], [-1]), shape=(4, 2))  # 创建一个 dia_matrix
        expected = m.toarray()  # 获取该 dia_matrix 的数组表示
        assert_array_equal(m.tocsc().toarray(), expected)  # 断言转换为 csc_matrix 后的数组与预期相等
        assert_array_equal(m.tocsr().toarray(), expected)  # 断言转换为 csr_matrix 后的数组与预期相等

    # 测试 gh-10050 的转换方法
    def test_tocoo_gh10050(self):
        m = dia_matrix([[1, 2], [3, 4]]).tocoo()  # 创建一个 dia_matrix 并转换为 coo_matrix
        flat_inds = np.ravel_multi_index((m.row, m.col), m.shape)  # 计算扁平化索引
        inds_are_sorted = np.all(np.diff(flat_inds) > 0)  # 检查扁平化索引是否已排序
        assert m.has_canonical_format == inds_are_sorted  # 断言 dia_matrix 是否具有规范格式

    # 测试 gh-19245 中的 index_dtype 与 tocoo、tocsr、tocsc 方法
    def test_tocoo_tocsr_tocsc_gh19245(self):
        data = np.array([[1, 2, 3, 4]]).repeat(3, axis=0)  # 创建数据数组
        offsets = np.array([0, -1, 2], dtype=np.int32)  # 创建偏移数组，指定数据类型为 np.int32
        dia = sparse.dia_array((data, offsets), shape=(4, 4))  # 创建一个 dia_array

        coo = dia.tocoo()  # 转换为 coo_matrix
        assert coo.col.dtype == np.int32  # 断言列索引的数据类型为 np.int32
        csr = dia.tocsr()  # 转换为 csr_matrix
        assert csr.indices.dtype == np.int32  # 断言非零元素索引的数据类型为 np.int32
        csc = dia.tocsc()  # 转换为 csc_matrix
        assert csc.indices.dtype == np.int32  # 断言非零元素索引的数据类型为 np.int32

    # 乘法运算的测试用例，修复了 gh-20434 的问题
    def test_mul_scalar(self):
        m = dia_matrix([[1, 2], [0, 4]])  # 创建一个 dia_matrix
        res = m * 3  # 乘以标量 3
        assert isinstance(res, dia_matrix)  # 断言结果的类型为 dia_matrix
        assert_array_equal(res.toarray(), [[3, 6], [0, 12]])  # 断言结果的数组表示与预期相等

        res2 = m.multiply(3)  # 使用 multiply 方法乘以标量 3
        assert isinstance(res2, dia_matrix)  # 断言结果的类型为 dia_matrix
        assert_array_equal(res2.toarray(), [[3, 6], [0, 12]])  # 断言结果的数组表示与预期相等

TestDIA.init_class()

# 定义 TestBSR 类，继承自 sparse_test_class，禁用了一些特性（getset, slicing, slicing_assign, fancy_indexing, fancy_assign, nnz_axis）
class TestBSR(sparse_test_class(getset=False,
                                slicing=False, slicing_assign=False,
                                fancy_indexing=False, fancy_assign=False,
                                nnz_axis=False)):
    spcreator = bsr_matrix  # 设置 spcreator 属性为 bsr_matrix
    # 定义包含三种数据类型的列表，分别为整数、双精度浮点数和双精度复数
    math_dtypes = [np.int_, np.float64, np.complex128]

    # 第一个构造函数测试方法
    def test_constructor1(self):
        # 检查原生的 BSR 格式构造函数
        indptr = array([0,2,2,4])  # 定义行指针数组
        indices = array([0,2,2,3])  # 定义列索引数组
        data = zeros((4,2,3))  # 创建一个全零的三维数组

        data[0] = array([[0, 1, 2],  # 填充数据的第一个子数组
                         [3, 0, 5]])
        data[1] = array([[0, 2, 4],  # 填充数据的第二个子数组
                         [6, 0, 10]])
        data[2] = array([[0, 4, 8],  # 填充数据的第三个子数组
                         [12, 0, 20]])
        data[3] = array([[0, 5, 10],  # 填充数据的第四个子数组
                         [15, 0, 25]])

        # 创建 Kronicker 乘积矩阵 A
        A = kron([[1,0,2,0],[0,0,0,0],[0,0,4,5]], [[0,1,2],[3,0,5]])
        # 使用给定的数据、索引和行指针创建 BSR 稀疏矩阵 Asp
        Asp = bsr_matrix((data,indices,indptr),shape=(6,12))
        # 断言 Asp 转换为稠密数组与 A 相等
        assert_equal(Asp.toarray(), A)

        # 从数组推断形状
        Asp = bsr_matrix((data,indices,indptr))
        # 断言 Asp 转换为稠密数组与 A 相等
        assert_equal(Asp.toarray(), A)

    # 第二个构造函数测试方法
    def test_constructor2(self):
        # 从密集矩阵构造

        # 测试零矩阵
        for shape in [(1,1), (5,1), (1,10), (10,4), (3,7), (2,1)]:
            A = zeros(shape)  # 创建指定形状的全零数组 A
            # 断言 BSR 矩阵从 A 构造后转换为稠密数组与 A 相等
            assert_equal(bsr_matrix(A).toarray(), A)
        A = zeros((4,6))  # 创建一个 4x6 的全零数组 A
        # 断言 BSR 矩阵从 A 构造并指定块大小后转换为稠密数组与 A 相等
        assert_equal(bsr_matrix(A, blocksize=(2, 2)).toarray(), A)
        assert_equal(bsr_matrix(A, blocksize=(2, 3)).toarray(), A)

        # 创建 Kronicker 乘积矩阵 A
        A = kron([[1,0,2,0],[0,0,0,0],[0,0,4,5]], [[0,1,2],[3,0,5]])
        # 断言 BSR 矩阵从 A 构造后转换为稠密数组与 A 相等
        assert_equal(bsr_matrix(A).toarray(), A)
        assert_equal(bsr_matrix(A, shape=(6, 12)).toarray(), A)
        assert_equal(bsr_matrix(A, blocksize=(1, 1)).toarray(), A)
        assert_equal(bsr_matrix(A, blocksize=(2, 3)).toarray(), A)
        assert_equal(bsr_matrix(A, blocksize=(2, 6)).toarray(), A)
        assert_equal(bsr_matrix(A, blocksize=(2, 12)).toarray(), A)
        assert_equal(bsr_matrix(A, blocksize=(3, 12)).toarray(), A)
        assert_equal(bsr_matrix(A, blocksize=(6, 12)).toarray(), A)

        # 创建一个不同的 Kronicker 乘积矩阵 A
        A = kron([[1,0,2,0],[0,1,0,0],[0,0,0,0]], [[0,1,2],[3,0,5]])
        # 断言 BSR 矩阵从 A 构造并指定块大小后转换为稠密数组与 A 相等
        assert_equal(bsr_matrix(A, blocksize=(2, 3)).toarray(), A)

    # 第三个构造函数测试方法
    def test_constructor3(self):
        # 从类似 COO 格式 (data,(row,col)) 构造

        arg = ([1,2,3], ([0,1,1], [0,0,1]))  # 定义参数 arg，包含数据和坐标信息
        A = array([[1,0],[2,3]])  # 创建一个二维数组 A
        # 断言 BSR 矩阵从参数 arg 构造并指定块大小后转换为稠密数组与 A 相等
        assert_equal(bsr_matrix(arg, blocksize=(2, 2)).toarray(), A)

    # 第四个构造函数测试方法
    def test_constructor4(self):
        # GH-6292 的回归测试: bsr_matrix((data, indices, indptr)) 试图比较 int 和 None

        n = 8  # 定义整数 n
        data = np.ones((n, n, 1), dtype=np.int8)  # 创建一个全为 1 的三维数组
        indptr = np.array([0, n], dtype=np.int32)  # 创建行指针数组
        indices = np.arange(n, dtype=np.int32)  # 创建索引数组
        bsr_matrix((data, indices, indptr), blocksize=(n, 1), copy=False)
        # 使用给定的数据、索引和行指针创建 BSR 稀疏矩阵，并设置不复制原始数据
    def test_constructor5(self):
        # 检查在 gh-13400 中引入的验证
        n = 8
        # 创建一个长度为 n 的全为 1 的一维数组
        data_1dim = np.ones(n)
        # 创建一个 n x n x n 大小的全为 1 的三维数组
        data = np.ones((n, n, n))
        # 创建一个长度为 2 的数组作为 indptr
        indptr = np.array([0, n])
        # 创建一个长度为 n 的数组作为 indices
        indices = np.arange(n)

        # 使用 assert_raises 检查 ValueError 是否被抛出
        with assert_raises(ValueError):
            # 检查数据的维度是否正确
            bsr_matrix((data_1dim, indices, indptr))

        with assert_raises(ValueError):
            # 检查是否使用了无效的块大小
            bsr_matrix((data, indices, indptr), blocksize=(1, 1, 1))

        with assert_raises(ValueError):
            # 检查块大小是否不匹配
            bsr_matrix((data, indices, indptr), blocksize=(1, 1))

    def test_default_dtype(self):
        # `values` 作为一个 numpy 数组，形状为 (2, 2, 1)
        values = [[[1], [1]], [[1], [1]]]
        # 创建一个长度为 2 的 int32 类型数组作为 indptr
        indptr = np.array([0, 2], dtype=np.int32)
        # 创建一个长度为 2 的 int32 类型数组作为 indices
        indices = np.array([0, 1], dtype=np.int32)
        # 使用给定的 values, indices 和 indptr 创建一个 BSR 矩阵，块大小为 (2, 1)
        b = bsr_matrix((values, indices, indptr), blocksize=(2, 1))
        # 断言矩阵的数据类型与 values 的数据类型相同
        assert b.dtype == np.array(values).dtype

    def test_bsr_tocsr(self):
        # 检查从 BSR 到 CSR 的本地转换
        indptr = array([0, 2, 2, 4])
        indices = array([0, 2, 2, 3])
        # 创建一个形状为 (4, 2, 3) 的全零数组作为 data
        data = zeros((4, 2, 3))

        data[0] = array([[0, 1, 2],
                         [3, 0, 5]])
        data[1] = array([[0, 2, 4],
                         [6, 0, 10]])
        data[2] = array([[0, 4, 8],
                         [12, 0, 20]])
        data[3] = array([[0, 5, 10],
                         [15, 0, 25]])

        # 使用给定的 data, indices 和 indptr 创建一个 BSR 矩阵，形状为 (6, 12)
        Absr = bsr_matrix((data, indices, indptr), shape=(6, 12))
        # 将 BSR 矩阵转换为 CSR 格式
        Acsr = Absr.tocsr()
        # 通过 COO 格式转换为 CSR 格式
        Acsr_via_coo = Absr.tocoo().tocsr()
        # 断言两个 CSR 矩阵的数组表示相等
        assert_equal(Acsr.toarray(), Acsr_via_coo.toarray())

    def test_eliminate_zeros(self):
        # 创建一个按照指定规则生成的数据数组
        data = kron([1, 0, 0, 0, 2, 0, 3, 0], [[1,1],[1,1]]).T
        # 将数据数组重新形状为 (-1, 2, 2)
        data = data.reshape(-1, 2, 2)
        # 创建一个长度为 8 的数组作为 indices
        indices = array([1, 2, 3, 4, 5, 6, 7, 8])
        # 创建一个长度为 3 的数组作为 indptr
        indptr = array([0, 3, 8])
        # 使用给定的 data, indices 和 indptr 创建一个 BSR 矩阵，形状为 (4, 20)
        asp = bsr_matrix((data, indices, indptr), shape=(4, 20))
        # 复制 asp 到 bsp
        bsp = asp.copy()
        # 消除 asp 中的零元素
        asp.eliminate_zeros()
        # 断言消除零元素后的非零元素数量是否正确
        assert_array_equal(asp.nnz, 3*4)
        # 断言消除零元素后的数组表示与原始矩阵 bsp 的数组表示相等
        assert_array_equal(asp.toarray(), bsp.toarray())

    # github issue #9687
    # 定义一个测试函数，用于测试在所有元素均为零的情况下消除零值
    def test_eliminate_zeros_all_zero(self):
        np.random.seed(0)
        # 创建一个稀疏矩阵，随机填充数据，使用块大小为 (2, 3)
        m = bsr_matrix(np.random.random((12, 12)), blocksize=(2, 3))

        # 消除部分块的零值，但不是全部
        m.data[m.data <= 0.9] = 0
        # 调用稀疏矩阵的方法，消除零值
        m.eliminate_zeros()
        # 断言稀疏矩阵非零元素的数量为 66
        assert_equal(m.nnz, 66)
        # 断言稀疏矩阵数据的形状为 (11, 2, 3)
        assert_array_equal(m.data.shape, (11, 2, 3))

        # 消除剩余的所有块的零值
        m.data[m.data <= 1.0] = 0
        m.eliminate_zeros()
        # 断言稀疏矩阵非零元素的数量为 0
        assert_equal(m.nnz, 0)
        # 断言稀疏矩阵数据的形状为 (0, 2, 3)
        assert_array_equal(m.data.shape, (0, 2, 3))
        # 断言稀疏矩阵转换为密集数组后与全零数组相等
        assert_array_equal(m.toarray(), np.zeros((12, 12)))

        # 测试快速路径
        m.eliminate_zeros()
        # 再次断言稀疏矩阵非零元素的数量为 0
        assert_equal(m.nnz, 0)
        # 再次断言稀疏矩阵数据的形状为 (0, 2, 3)
        assert_array_equal(m.data.shape, (0, 2, 3))
        # 再次断言稀疏矩阵转换为密集数组后与全零数组相等
        assert_array_equal(m.toarray(), np.zeros((12, 12)))

    # 定义一个测试函数，用于测试 BSR 矩阵向量乘法
    def test_bsr_matvec(self):
        # 创建一个 BSR 矩阵，使用给定的块大小 (4, 5)
        A = bsr_matrix(arange(2*3*4*5).reshape(2*4, 3*5), blocksize=(4, 5))
        # 创建一个向量 x，长度与 A 的列数相同
        x = arange(A.shape[1]).reshape(-1, 1)
        # 断言 BSR 矩阵与向量 x 的乘积结果与其密集表示的乘积结果相等
        assert_equal(A @ x, A.toarray() @ x)

    # 定义一个测试函数，用于测试 BSR 矩阵多向量乘法
    def test_bsr_matvecs(self):
        # 创建一个 BSR 矩阵，使用给定的块大小 (4, 5)
        A = bsr_matrix(arange(2*3*4*5).reshape(2*4, 3*5), blocksize=(4, 5))
        # 创建一个多列的向量 x，长度是 A 列数的 6 倍
        x = arange(A.shape[1] * 6).reshape(-1, 6)
        # 断言 BSR 矩阵与多向量 x 的乘积结果与其密集表示的乘积结果相等
        assert_equal(A @ x, A.toarray() @ x)

    # 标记此测试为预期失败，原因是 BSR 类型矩阵没有 __getitem__ 方法
    @pytest.mark.xfail(run=False, reason='BSR does not have a __getitem__')
    def test_iterator(self):
        pass

    # 标记此测试为预期失败，原因是 BSR 类型矩阵没有 __setitem__ 方法
    @pytest.mark.xfail(run=False, reason='BSR does not have a __setitem__')
    def test_setdiag(self):
        pass

    # 定义一个测试函数，用于测试调整尺寸时的块矩阵行为
    def test_resize_blocked(self):
        # 使用指定的块大小创建一个稀疏矩阵
        # 测试 resize() 方法，块大小不为 (1, 1)
        D = np.array([[1, 0, 3, 4],
                      [2, 0, 0, 0],
                      [3, 0, 0, 0]])
        S = self.spcreator(D, blocksize=(1, 2))
        # 断言调整大小为 (3, 2) 后稀疏矩阵返回 None
        assert_(S.resize((3, 2)) is None)
        # 断言稀疏矩阵转换为密集数组后与预期数组相等
        assert_array_equal(S.toarray(), [[1, 0],
                                         [2, 0],
                                         [3, 0]])
        # 再次调整大小为 (2, 2)
        S.resize((2, 2))
        assert_array_equal(S.toarray(), [[1, 0],
                                         [2, 0]])
        # 再次调整大小为 (3, 2)
        S.resize((3, 2))
        assert_array_equal(S.toarray(), [[1, 0],
                                         [2, 0],
                                         [0, 0]])
        # 再次调整大小为 (3, 4)
        S.resize((3, 4))
        assert_array_equal(S.toarray(), [[1, 0, 0, 0],
                                         [2, 0, 0, 0],
                                         [0, 0, 0, 0]])
        # 断言调整大小为 (2, 3) 时引发 ValueError
        assert_raises(ValueError, S.resize, (2, 3))

    # 标记此测试为预期失败，原因是 BSR 类型矩阵没有 __setitem__ 方法
    @pytest.mark.xfail(run=False, reason='BSR does not have a __setitem__')
    def test_setdiag_comprehensive(self):
        pass

    # 根据条件跳过测试，如果在 Colab 环境下运行，因为超出了内存限制
    @pytest.mark.skipif(IS_COLAB, reason="exceeds memory limit")
    def test_scalar_idx_dtype(self):
        # 检查索引数据类型是否考虑了传递给 sparsetools 的所有参数，包括标量参数
        indptr = np.zeros(2, dtype=np.int32)
        indices = np.zeros(0, dtype=np.int32)
        vals = np.zeros((0, 1, 1))
        # 创建 BSR 稀疏矩阵对象 a 和 b，分别设置形状为 (1, 2^31-1) 和 (1, 2^31)
        a = bsr_matrix((vals, indices, indptr), shape=(1, 2**31-1))
        b = bsr_matrix((vals, indices, indptr), shape=(1, 2**31))
        # 创建 BSR 稀疏矩阵对象 c 和 d，分别设置数据的形状为 (1, 2^31-1) 和 (1, 2^31)
        c = bsr_matrix((1, 2**31-1))
        d = bsr_matrix((1, 2**31))
        # 断言检查 a、b、c、d 对象的 indptr 属性的数据类型是否符合预期
        assert_equal(a.indptr.dtype, np.int32)
        assert_equal(b.indptr.dtype, np.int64)
        assert_equal(c.indptr.dtype, np.int32)
        assert_equal(d.indptr.dtype, np.int64)

        try:
            # 尝试创建 BSR 稀疏矩阵对象 e 和 f，设置形状分别为 (1, 2^31-1) 和 (1, 2^31)
            vals2 = np.zeros((0, 1, 2**31-1))
            vals3 = np.zeros((0, 1, 2**31))
            e = bsr_matrix((vals2, indices, indptr), shape=(1, 2**31-1))
            f = bsr_matrix((vals3, indices, indptr), shape=(1, 2**31))
            # 断言检查 e 和 f 对象的 indptr 属性的数据类型是否符合预期
            assert_equal(e.indptr.dtype, np.int32)
            assert_equal(f.indptr.dtype, np.int64)
        except (MemoryError, ValueError):
            # 在 32 位 Python 下可能会失败
            e = 0
            f = 0

        # 这些断言不应该失败
        # 对于列表中的每个对象 x，执行加法操作
        for x in [a, b, c, d, e, f]:
            x + x
TestBSR.init_class()
# 调用TestBSR类的静态方法init_class()

#------------------------------------------------------------------------------
# Tests for non-canonical representations (with duplicates, unsorted indices)
#------------------------------------------------------------------------------

def _same_sum_duplicate(data, *inds, **kwargs):
    """Duplicates entries to produce the same matrix"""
    # 如果数据类型是布尔型或无符号整数型，则处理索引指针（indptr），返回处理后的数据和索引
    indptr = kwargs.pop('indptr', None)
    if np.issubdtype(data.dtype, np.bool_) or \
       np.issubdtype(data.dtype, np.unsignedinteger):
        if indptr is None:
            return (data,) + inds
        else:
            return (data,) + inds + (indptr,)

    # 找出所有值为0的位置
    zeros_pos = (data == 0).nonzero()

    # 复制数据条目以产生相同的矩阵
    data = data.repeat(2, axis=0)  # 沿着0轴（行）复制数据
    data[::2] -= 1  # 奇数索引位置减1
    data[1::2] = 1  # 偶数索引位置设为1

    # 不破坏所有显式的零
    if zeros_pos[0].size > 0:
        pos = tuple(p[0] for p in zeros_pos)  # 找到第一个零的位置
        pos1 = (2*pos[0],) + pos[1:]  # 对应位置设置为0
        pos2 = (2*pos[0]+1,) + pos[1:]  # 对应位置设置为0
        data[pos1] = 0
        data[pos2] = 0

    # 复制所有索引
    inds = tuple(indices.repeat(2) for indices in inds)

    if indptr is None:
        return (data,) + inds
    else:
        return (data,) + inds + (indptr * 2,)


class _NonCanonicalMixin:
    def spcreator(self, D, *args, sorted_indices=False, **kwargs):
        """Replace D with a non-canonical equivalent: containing
        duplicate elements and explicit zeros"""
        construct = super().spcreator
        # 调用父类的spcreator方法构造一个非规范的等价矩阵M
        M = construct(D, *args, **kwargs)

        # 找出M中所有值为0的位置
        zero_pos = (M.toarray() == 0).nonzero()
        has_zeros = (zero_pos[0].size > 0)  # 判断是否存在零元素

        # 如果存在零元素，则在M中插入一个显式的零
        if has_zeros:
            k = zero_pos[0].size//2  # 找到第k个零元素的位置
            with suppress_warnings() as sup:
                sup.filter(SparseEfficiencyWarning, "Changing the sparsity structure")
                M = self._insert_explicit_zero(M, zero_pos[0][k], zero_pos[1][k])

        # 使用M构造一个非规范的矩阵NC
        arg1 = self._arg1_for_noncanonical(M, sorted_indices)
        if 'shape' not in kwargs:
            kwargs['shape'] = M.shape
        NC = construct(arg1, **kwargs)

        # 检查结果是否有效
        if NC.dtype in [np.float32, np.complex64]:
            # 对于单精度浮点数，由于构造NC所涉及的额外操作引入的差异，需要比默认更宽松的容差水平
            rtol = 1e-05
        else:
            rtol = 1e-07
        assert_allclose(NC.toarray(), M.toarray(), rtol=rtol)

        # 检查至少存在一个显式的零元素
        if has_zeros:
            assert_((NC.data == 0).any())
        # TODO 检查NC是否包含重复项（非显式的零元素）

        return NC

    @pytest.mark.skip(reason='bool(matrix) counts explicit zeros')
    def test_bool(self):
        pass

    @pytest.mark.skip(reason='getnnz-axis counts explicit zeros')
    def test_getnnz_axis(self):
        pass

    @pytest.mark.skip(reason='nnz counts explicit zeros')
    def test_empty(self):
        pass
class _NonCanonicalCompressedMixin(_NonCanonicalMixin):
    # 继承自 _NonCanonicalMixin 的非规范压缩混合类

    def _arg1_for_noncanonical(self, M, sorted_indices=False):
        """Return non-canonical constructor arg1 equivalent to M"""
        # 返回与 M 等效的非规范构造函数参数 arg1
        # 调用 _same_sum_duplicate 函数，处理 M 的数据、索引和行指针
        data, indices, indptr = _same_sum_duplicate(M.data, M.indices,
                                                    indptr=M.indptr)
        # 如果不需要按索引排序
        if not sorted_indices:
            # 反转每个行的索引和数据
            for start, stop in zip(indptr, indptr[1:]):
                indices[start:stop] = indices[start:stop][::-1].copy()
                data[start:stop] = data[start:stop][::-1].copy()
        return data, indices, indptr
        # 返回处理后的数据、索引和行指针

    def _insert_explicit_zero(self, M, i, j):
        # 将矩阵 M 的位置 (i, j) 设置为 0
        M[i,j] = 0
        return M
        # 返回修改后的矩阵 M


class _NonCanonicalCSMixin(_NonCanonicalCompressedMixin):
    # 继承自 _NonCanonicalCompressedMixin 的非规范压缩稀疏混合类

    def test_getelement(self):
        # 定义内部函数 check，用于测试特定数据类型和索引排序方式下的行为
        def check(dtype, sorted_indices):
            # 创建一个数组 D，根据指定的数据类型 dtype
            D = array([[1,0,0],
                       [4,3,0],
                       [0,2,0],
                       [0,0,0]], dtype=dtype)
            # 使用 self.spcreator 创建稀疏矩阵 A，基于数组 D，并指定是否排序索引
            A = self.spcreator(D, sorted_indices=sorted_indices)

            # 获取数组 D 的形状 M,N
            M,N = D.shape

            # 遍历所有可能的索引范围
            for i in range(-M, M):
                for j in range(-N, N):
                    # 断言稀疏矩阵 A 的元素与数组 D 相同
                    assert_equal(A[i,j], D[i,j])

            # 断言对于不合法索引对 (i, j)，会引发 IndexError 或 TypeError
            for ij in [(0,3),(-1,3),(4,0),(4,3),(4,-1), (1, 2, 3)]:
                assert_raises((IndexError, TypeError), A.__getitem__, ij)

        # 对于所有支持的数据类型，以及排序和非排序的索引方式，进行测试
        for dtype in supported_dtypes:
            for sorted_indices in [False, True]:
                check(np.dtype(dtype), sorted_indices)

    def test_setitem_sparse(self):
        # 创建单位矩阵 D
        D = np.eye(3)
        # 使用 self.spcreator 创建稀疏矩阵 A，基于数组 D
        A = self.spcreator(D)
        # 使用 self.spcreator 创建稀疏矩阵 B，基于列表 [[1,2,3]]
        B = self.spcreator([[1,2,3]])

        # 修改 D 的第 1 行为 B 的稀疏表示
        D[1,:] = B.toarray()
        # 使用 A[1,:] = B 修改稀疏矩阵 A 的第 1 行
        with suppress_warnings() as sup:
            sup.filter(SparseEfficiencyWarning, "Changing the sparsity structure")
            A[1,:] = B
        # 断言稀疏矩阵 A 转换为密集表示后与 D 相同
        assert_array_equal(A.toarray(), D)

        # 修改 D 的第 2 列为 B 的稀疏表示的扁平形式
        D[:,2] = B.toarray().ravel()
        # 使用 A[:,2] = B.T 修改稀疏矩阵 A 的第 2 列
        with suppress_warnings() as sup:
            sup.filter(SparseEfficiencyWarning, "Changing the sparsity structure")
            A[:,2] = B.T
        # 断言稀疏矩阵 A 转换为密集表示后与 D 相同
        assert_array_equal(A.toarray(), D)

    @pytest.mark.xfail(run=False, reason='inverse broken with non-canonical matrix')
    def test_inv(self):
        # 标记为预期失败测试，因为逆矩阵在非规范矩阵下不工作
        pass

    @pytest.mark.xfail(run=False, reason='solve broken with non-canonical matrix')
    def test_solve(self):
        # 标记为预期失败测试，因为求解函数在非规范矩阵下不工作
        pass


class TestCSRNonCanonical(_NonCanonicalCSMixin, TestCSR):
    # 测试非规范 CSR 稀疏矩阵，继承自 _NonCanonicalCSMixin 和 TestCSR
    pass


class TestCSCNonCanonical(_NonCanonicalCSMixin, TestCSC):
    # 测试非规范 CSC 稀疏矩阵，继承自 _NonCanonicalCSMixin 和 TestCSC
    pass


class TestBSRNonCanonical(_NonCanonicalCompressedMixin, TestBSR):
    # 测试非规范 BSR 压缩矩阵，继承自 _NonCanonicalCompressedMixin 和 TestBSR

    def _insert_explicit_zero(self, M, i, j):
        # 将矩阵 M 转换为 CSR 格式，并将位置 (i, j) 设置为 0
        x = M.tocsr()
        x[i,j] = 0
        # 将修改后的 CSR 格式矩阵转换为 BSR 格式返回
        return x.tobsr(blocksize=M.blocksize)

    @pytest.mark.xfail(run=False, reason='diagonal broken with non-canonical BSR')
    def test_diagonal(self):
        # 标记为预期失败测试，因为对角线函数在非规范 BSR 矩阵下不工作
        pass

    @pytest.mark.xfail(run=False, reason='expm broken with non-canonical BSR')
    def test_expm(self):
        # 标记为预期失败测试，因为指数函数在非规范 BSR 矩阵下不工作
        pass


class TestCOONonCanonical(_NonCanonicalMixin, TestCOO):
    # 测试非规范 COO 稀疏矩阵，继承自 _NonCanonicalMixin 和 TestCOO
    pass
    # 返回一个与输入矩阵 M 非规范构造函数参数 arg1 等效的数据，以及行列索引
    def _arg1_for_noncanonical(self, M, sorted_indices=None):
        # 调用 _same_sum_duplicate 函数，返回数据、行索引和列索引
        data, row, col = _same_sum_duplicate(M.data, M.row, M.col)
        # 返回数据和元组 (行索引, 列索引)
        return data, (row, col)
    
    # 向矩阵 M 的指定位置 (i, j) 插入显式的零值
    def _insert_explicit_zero(self, M, i, j):
        # 在 M.data 数组的开头插入一个与 M.data.dtype 相同类型的零值
        M.data = np.r_[M.data.dtype.type(0), M.data]
        # 在 M.row 数组的开头插入一个与 M.row.dtype 相同类型的 i 值
        M.row = np.r_[M.row.dtype.type(i), M.row]
        # 在 M.col 数组的开头插入一个与 M.col.dtype 相同类型的 j 值
        M.col = np.r_[M.col.dtype.type(j), M.col]
        # 返回修改后的矩阵 M
        return M
    
    # 测试非规范矩阵的 setdiag 方法
    def test_setdiag_noncanonical(self):
        # 使用 self.spcreator 函数创建一个稀疏矩阵 m，初始化为单位矩阵的稀疏表示
        m = self.spcreator(np.eye(3))
        # 确保矩阵 m 没有重复的元素
        m.sum_duplicates()
        # 在矩阵 m 的次对角线上设置对角线元素为 [3, 2]
        m.setdiag([3, 2], k=1)
        # 再次确保矩阵 m 没有重复的元素
        m.sum_duplicates()
        # 断言：验证矩阵 m 的列索引是递增的
        assert_(np.all(np.diff(m.col) >= 0))
def cases_64bit():
    TEST_CLASSES = [TestBSR, TestCOO, TestCSC, TestCSR, TestDIA,
                    # lil/dok->other conversion operations have get_index_dtype
                    TestDOK, TestLIL
                    ]
    # 定义一个测试类列表，包括各种稀疏矩阵类型的测试类

    SKIP_TESTS = {
        'test_expm': 'expm for 64-bit indices not available',
        'test_inv': 'linsolve for 64-bit indices not available',
        'test_solve': 'linsolve for 64-bit indices not available',
        'test_scalar_idx_dtype': 'test implemented in base class',
        'test_large_dimensions_reshape': 'test actually requires 64-bit to work',
        'test_constructor_smallcol': 'test verifies int32 indexes',
        'test_constructor_largecol': 'test verifies int64 indexes',
        'test_tocoo_tocsr_tocsc_gh19245': 'test verifies int32 indexes',
    }
    # 定义一个字典，包含跳过测试的方法名及其对应的原因说明

    for cls in TEST_CLASSES:
        for method_name in sorted(dir(cls)):
            method = getattr(cls, method_name)
            # 遍历每个测试类的方法名列表，获取方法对象

            if (method_name.startswith('test_') and
                    not getattr(method, 'slow', False)):
                # 如果方法名以'test_'开头且不是slow测试

                marks = []

                msg = SKIP_TESTS.get(method_name)
                # 获取跳过测试的原因消息

                if bool(msg):
                    marks += [pytest.mark.skip(reason=msg)]
                    # 如果有跳过消息，添加skip标记到marks列表

                markers = getattr(method, 'pytestmark', [])
                # 获取方法可能包含的pytest标记列表

                for mark in markers:
                    if mark.name in ('skipif', 'skip', 'xfail', 'xslow'):
                        marks.append(mark)
                        # 将标记为'skipif', 'skip', 'xfail', 'xslow'的标记添加到marks列表

                yield pytest.param(cls, method_name, marks=marks)
                # 生成pytest参数化的测试用例
                    

class Test64Bit:
    MAT_CLASSES = [bsr_matrix, coo_matrix, csc_matrix, csr_matrix, dia_matrix]
    # 定义一组稀疏矩阵类列表

    def _create_some_matrix(self, mat_cls, m, n):
        return mat_cls(np.random.rand(m, n))
        # 创建一个指定类型的随机数值稀疏矩阵

    def _compare_index_dtype(self, m, dtype):
        dtype = np.dtype(dtype)
        # 将输入的dtype参数转换为NumPy的数据类型对象

        if isinstance(m, (csc_matrix, csr_matrix, bsr_matrix)):
            return (m.indices.dtype == dtype) and (m.indptr.dtype == dtype)
            # 如果是CSC、CSR或BSR类型的稀疏矩阵，比较其索引数据类型和给定dtype是否相同
        elif isinstance(m, coo_matrix):
            return (m.row.dtype == dtype) and (m.col.dtype == dtype)
            # 如果是COO类型的稀疏矩阵，比较其行和列的数据类型和给定dtype是否相同
        elif isinstance(m, dia_matrix):
            return (m.offsets.dtype == dtype)
            # 如果是DIA类型的稀疏矩阵，比较其偏移量的数据类型和给定dtype是否相同
        else:
            raise ValueError(f"matrix {m!r} has no integer indices")
            # 如果输入的矩阵类型不在支持的范围内，则引发值错误异常

    def test_decorator_maxval_limit(self):
        # Test that the with_64bit_maxval_limit decorator works
        # 测试 with_64bit_maxval_limit 装饰器是否有效

        @with_64bit_maxval_limit(maxval_limit=10)
        # 使用 with_64bit_maxval_limit 装饰器，设置最大值限制为10

        def check(mat_cls):
            m = mat_cls(np.random.rand(10, 1))
            assert_(self._compare_index_dtype(m, np.int32))
            # 使用 mat_cls 创建一个大小为(10, 1)的矩阵，验证其索引数据类型是否为np.int32

            m = mat_cls(np.random.rand(11, 1))
            assert_(self._compare_index_dtype(m, np.int64))
            # 使用 mat_cls 创建一个大小为(11, 1)的矩阵，验证其索引数据类型是否为np.int64

        for mat_cls in self.MAT_CLASSES:
            check(mat_cls)
            # 针对 MAT_CLASSES 中的每个稀疏矩阵类型，调用 check 函数进行测试
    def test_decorator_maxval_random(self):
        # Test that the with_64bit_maxval_limit decorator works (2)
        
        # 定义一个被with_64bit_maxval_limit装饰的函数check，用于检查装饰器的功能
        @with_64bit_maxval_limit(random=True)
        def check(mat_cls):
            seen_32 = False
            seen_64 = False
            # 进行100次循环检查
            for k in range(100):
                # 创建一个 mat_cls 类型的 9x9 矩阵
                m = self._create_some_matrix(mat_cls, 9, 9)
                # 检查矩阵的索引类型是否为 np.int32
                seen_32 = seen_32 or self._compare_index_dtype(m, np.int32)
                # 检查矩阵的索引类型是否为 np.int64
                seen_64 = seen_64 or self._compare_index_dtype(m, np.int64)
                # 如果已经同时看到了32位和64位索引类型，则退出循环
                if seen_32 and seen_64:
                    break
            else:
                # 如果没有同时看到32位和64位索引类型，则抛出断言错误
                raise AssertionError("both 32 and 64 bit indices not seen")

        # 对每个 self.MAT_CLASSES 中的类进行测试
        for mat_cls in self.MAT_CLASSES:
            check(mat_cls)

    def _check_resiliency(self, cls, method_name, **kw):
        # Resiliency test, to check that sparse matrices deal reasonably
        # with varying index data types.
        
        # 定义一个检查函数check，用于测试稀疏矩阵在不同索引数据类型下的稳健性
        @with_64bit_maxval_limit(**kw)
        def check(cls, method_name):
            instance = cls()
            # 如果实例有 setup_method 方法，则调用它
            if hasattr(instance, 'setup_method'):
                instance.setup_method()
            try:
                # 调用实例的 method_name 方法
                getattr(instance, method_name)()
            finally:
                # 如果实例有 teardown_method 方法，则调用它
                if hasattr(instance, 'teardown_method'):
                    instance.teardown_method()

        # 使用给定的 cls 和 method_name 调用 check 函数
        check(cls, method_name)

    @pytest.mark.parametrize('cls,method_name', cases_64bit())
    def test_resiliency_limit_10(self, cls, method_name):
        # 测试在 maxval_limit=10 条件下的稳健性
        self._check_resiliency(cls, method_name, maxval_limit=10)

    @pytest.mark.fail_slow(2)
    @pytest.mark.parametrize('cls,method_name', cases_64bit())
    def test_resiliency_random(self, cls, method_name):
        # bsr_matrix.eliminate_zeros 依赖于 csr_matrix 构造函数
        # 不复制索引数组 --- 当随机选择索引数据类型时，这个前提不一定成立
        # 测试在 random=True 条件下的稳健性
        self._check_resiliency(cls, method_name, random=True)

    @pytest.mark.parametrize('cls,method_name', cases_64bit())
    def test_resiliency_all_32(self, cls, method_name):
        # 测试在 fixed_dtype=np.int32 条件下的稳健性
        self._check_resiliency(cls, method_name, fixed_dtype=np.int32)

    @pytest.mark.parametrize('cls,method_name', cases_64bit())
    def test_resiliency_all_64(self, cls, method_name):
        # 测试在 fixed_dtype=np.int64 条件下的稳健性
        self._check_resiliency(cls, method_name, fixed_dtype=np.int64)

    @pytest.mark.fail_slow(5)
    @pytest.mark.parametrize('cls,method_name', cases_64bit())
    def test_no_64(self, cls, method_name):
        # 测试在 assert_32bit=True 条件下的稳健性
        self._check_resiliency(cls, method_name, assert_32bit=True)
    # 定义一个测试方法 test_downcast_intp，用于验证整数下溢的情况
    def test_downcast_intp(self):
        # 检查在处理 bincount 和 ufunc.reduceat 的 intp 下溢时的情况
        # 这里的目的是触发在32位系统上可能出现的问题，因为使用了64位索引
        # 由于使用仅适用于 intp 大小索引的函数可能会导致失败

        # 定义一个装饰器函数 with_64bit_maxval_limit，使用 np.int64 作为固定数据类型
        # downcast_maxval 设置为 1，限制最大值为1
        @with_64bit_maxval_limit(fixed_dtype=np.int64,
                                 downcast_maxval=1)
        def check_limited():
            # 下面的代码涉及到比 downcast_maxval 更大的索引
            # 创建一个压缩稀疏列矩阵 csc_matrix
            a = csc_matrix([[1, 2], [3, 4], [5, 6]])
            # 断言异常，期望触发 AssertionError
            assert_raises(AssertionError, a.getnnz, axis=1)
            assert_raises(AssertionError, a.sum, axis=0)

            # 创建一个压缩行矩阵 csr_matrix
            a = csr_matrix([[1, 2, 3], [3, 4, 6]])
            assert_raises(AssertionError, a.getnnz, axis=0)

            # 创建一个坐标型矩阵 coo_matrix
            a = coo_matrix([[1, 2, 3], [3, 4, 5]])
            assert_raises(AssertionError, a.getnnz, axis=0)

        # 定义另一个装饰器函数 with_64bit_maxval_limit，使用 np.int64 作为固定数据类型
        # 这里没有设置 downcast_maxval，即不限制最大值
        @with_64bit_maxval_limit(fixed_dtype=np.int64)
        def check_unlimited():
            # 下面的代码同样涉及到比 downcast_maxval 更大的索引
            a = csc_matrix([[1, 2], [3, 4], [5, 6]])
            a.getnnz(axis=1)
            a.sum(axis=0)

            a = csr_matrix([[1, 2, 3], [3, 4, 6]])
            a.getnnz(axis=0)

            a = coo_matrix([[1, 2, 3], [3, 4, 5]])
            a.getnnz(axis=0)

        # 调用 check_limited 函数，测试有限制情况下的处理
        check_limited()
        # 调用 check_unlimited 函数，测试无限制情况下的处理
        check_unlimited()
```