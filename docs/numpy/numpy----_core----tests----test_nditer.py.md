# `.\numpy\numpy\_core\tests\test_nditer.py`

```py
# 导入系统和 pytest 模块
import sys
import pytest

# 导入文本包装和子进程模块
import textwrap
import subprocess

# 导入 NumPy 库及其子模块
import numpy as np
import numpy._core.umath as ncu
import numpy._core._multiarray_tests as _multiarray_tests
# 导入 NumPy 的特定函数和类
from numpy import array, arange, nditer, all
# 导入 NumPy 测试相关的函数和类
from numpy.testing import (
    assert_, assert_equal, assert_array_equal, assert_raises,
    IS_WASM, HAS_REFCOUNT, suppress_warnings, break_cycles,
    )

# 定义一个函数，用于迭代多维数组的索引
def iter_multi_index(i):
    ret = []
    while not i.finished:
        ret.append(i.multi_index)
        i.iternext()
    return ret

# 定义一个函数，用于迭代数组的索引
def iter_indices(i):
    ret = []
    while not i.finished:
        ret.append(i.index)
        i.iternext()
    return ret

# 定义一个函数，用于迭代迭代器的索引
def iter_iterindices(i):
    ret = []
    while not i.finished:
        ret.append(i.iterindex)
        i.iternext()
    return ret

# 标记测试函数，如果 Python 不支持引用计数，则跳过该测试
@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
def test_iter_refcount():
    # 确保迭代器不会泄漏

    # 基本测试
    a = arange(6)
    dt = np.dtype('f4').newbyteorder()
    rc_a = sys.getrefcount(a)
    rc_dt = sys.getrefcount(dt)
    # 使用 nditer 迭代器处理数组 a
    with nditer(a, [],
                [['readwrite', 'updateifcopy']],
                casting='unsafe',
                op_dtypes=[dt]) as it:
        assert_(not it.iterationneedsapi)
        assert_(sys.getrefcount(a) > rc_a)
        assert_(sys.getrefcount(dt) > rc_dt)
    # 释放迭代器 'it'
    it = None
    assert_equal(sys.getrefcount(a), rc_a)
    assert_equal(sys.getrefcount(dt), rc_dt)

    # 带有复制的测试
    a = arange(6, dtype='f4')
    dt = np.dtype('f4')
    rc_a = sys.getrefcount(a)
    rc_dt = sys.getrefcount(dt)
    it = nditer(a, [],
                [['readwrite']],
                op_dtypes=[dt])
    rc2_a = sys.getrefcount(a)
    rc2_dt = sys.getrefcount(dt)
    it2 = it.copy()
    assert_(sys.getrefcount(a) > rc2_a)
    if sys.version_info < (3, 13):
        # 在 Python 3.13 之前，np.dtype('f4') 在创建后不会被回收
        assert_(sys.getrefcount(dt) > rc2_dt)
    it = None
    assert_equal(sys.getrefcount(a), rc2_a)
    assert_equal(sys.getrefcount(dt), rc2_dt)
    it2 = None
    assert_equal(sys.getrefcount(a), rc_a)
    assert_equal(sys.getrefcount(dt), rc_dt)

    del it2  # 避免 pyflakes 未使用变量警告

def test_iter_best_order():
    # 迭代器应该始终找到内存地址增加的迭代顺序

    # 测试 1 维到 5 维数组的迭代顺序
    # 遍历不同形状的数组
    for shape in [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 3), (2, 3, 2, 2, 3)]:
        # 创建包含所有元素的一维数组
        a = arange(np.prod(shape))
        
        # 测试每种正负步长组合
        for dirs in range(2**len(shape)):
            # 根据位掩码设置切片方向
            dirs_index = [slice(None)] * len(shape)
            for bit in range(len(shape)):
                if ((2**bit) & dirs):
                    dirs_index[bit] = slice(None, None, -1)
            dirs_index = tuple(dirs_index)

            # 根据切片方向创建视图
            aview = a.reshape(shape)[dirs_index]
            
            # 使用 C 顺序迭代
            i = nditer(aview, [], [['readonly']])
            assert_equal([x for x in i], a)
            
            # 使用 Fortran 顺序迭代
            i = nditer(aview.T, [], [['readonly']])
            assert_equal([x for x in i], a)
            
            # 如果数组维度大于 2，则使用其他顺序迭代
            if len(shape) > 2:
                i = nditer(aview.swapaxes(0, 1), [], [['readonly']])
                assert_equal([x for x in i], a)
# 测试强制使用 C 顺序

# 测试从1维到5维形状的顺序
for shape in [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 3), (2, 3, 2, 2, 3)]:
    # 创建包含指定形状的连续数组
    a = arange(np.prod(shape))
    # 测试每种正向和反向步幅的组合
    for dirs in range(2**len(shape)):
        dirs_index = [slice(None)]*len(shape)
        # 根据 dirs 的每一位设置切片方向
        for bit in range(len(shape)):
            if ((2**bit) & dirs):
                dirs_index[bit] = slice(None, None, -1)
        dirs_index = tuple(dirs_index)

        # 创建视图，根据 dirs_index 切片
        aview = a.reshape(shape)[dirs_index]
        # 使用 C 顺序迭代
        i = nditer(aview, order='C')
        # 断言迭代器中的值与按 C 顺序展平后的数组相等
        assert_equal([x for x in i], aview.ravel(order='C'))
        # 使用 Fortran 顺序迭代
        i = nditer(aview.T, order='C')
        # 断言迭代器中的值与按 C 顺序展平后的转置数组相等
        assert_equal([x for x in i], aview.T.ravel(order='C'))
        # 如果形状长度大于2，测试其他顺序
        if len(shape) > 2:
            i = nditer(aview.swapaxes(0, 1), order='C')
            # 断言迭代器中的值与按 C 顺序展平后的轴对换数组相等
            assert_equal([x for x in i], aview.swapaxes(0, 1).ravel(order='C'))

def test_iter_f_order():
    # 测试强制使用 F 顺序

    # 测试从1维到5维形状的顺序
    for shape in [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 3), (2, 3, 2, 2, 3)]:
        # 创建包含指定形状的连续数组
        a = arange(np.prod(shape))
        # 测试每种正向和反向步幅的组合
        for dirs in range(2**len(shape)):
            dirs_index = [slice(None)]*len(shape)
            # 根据 dirs 的每一位设置切片方向
            for bit in range(len(shape)):
                if ((2**bit) & dirs):
                    dirs_index[bit] = slice(None, None, -1)
            dirs_index = tuple(dirs_index)

            # 创建视图，根据 dirs_index 切片
            aview = a.reshape(shape)[dirs_index]
            # 使用 F 顺序迭代
            i = nditer(aview, order='F')
            # 断言迭代器中的值与按 F 顺序展平后的数组相等
            assert_equal([x for x in i], aview.ravel(order='F'))
            # 使用 Fortran 顺序迭代
            i = nditer(aview.T, order='F')
            # 断言迭代器中的值与按 F 顺序展平后的转置数组相等
            assert_equal([x for x in i], aview.T.ravel(order='F'))
            # 如果形状长度大于2，测试其他顺序
            if len(shape) > 2:
                i = nditer(aview.swapaxes(0, 1), order='F')
                # 断言迭代器中的值与按 F 顺序展平后的轴对换数组相等
                assert_equal([x for x in i], aview.swapaxes(0, 1).ravel(order='F'))

def test_iter_c_or_f_order():
    # 测试强制使用任意连续（C 或 F）顺序

    # 测试从1维到5维形状的顺序
    # 对于每个形状进行迭代，形状分别为 (5,), (3, 4), (2, 3, 4), (2, 3, 4, 3), (2, 3, 2, 2, 3)
    for shape in [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 3), (2, 3, 2, 2, 3)]:
        # 创建一个包含 np.prod(shape) 个元素的数组 a
        a = arange(np.prod(shape))
        
        # 测试每种正负步长组合
        for dirs in range(2**len(shape)):
            # 初始化用于选择方向的索引列表
            dirs_index = [slice(None)] * len(shape)
            
            # 根据 dirs 中的每一位设置对应的索引方向
            for bit in range(len(shape)):
                if ((2**bit) & dirs):
                    dirs_index[bit] = slice(None, None, -1)
            dirs_index = tuple(dirs_index)

            # 根据 dirs_index 创建视图 aview
            aview = a.reshape(shape)[dirs_index]
            
            # 使用 C 顺序进行迭代
            i = nditer(aview, order='A')
            # 断言迭代结果与 C 顺序的扁平化视图一致
            assert_equal([x for x in i], aview.ravel(order='A'))
            
            # 使用 Fortran 顺序进行迭代
            i = nditer(aview.T, order='A')
            # 断言迭代结果与 Fortran 顺序的扁平化视图一致
            assert_equal([x for x in i], aview.T.ravel(order='A'))
            
            # 如果形状长度大于 2，则测试其他顺序
            if len(shape) > 2:
                i = nditer(aview.swapaxes(0, 1), order='A')
                # 断言迭代结果与交换轴后的扁平化视图一致
                assert_equal([x for x in i],
                                    aview.swapaxes(0, 1).ravel(order='A'))
# 测试多索引设置功能
def test_nditer_multi_index_set():
    # 创建一个形状为(2, 3)的NumPy数组，内容为0到5的连续整数
    a = np.arange(6).reshape(2, 3)
    # 创建一个NumPy迭代器，以多索引模式遍历数组a
    it = np.nditer(a, flags=['multi_index'])

    # 将迭代器的多索引设置为(0, 2)，即a[0]中的后两个元素
    it.multi_index = (0, 2,)

    # 断言迭代器遍历结果是否为[2, 3, 4, 5]
    assert_equal([i for i in it], [2, 3, 4, 5])

# 根据条件跳过测试，如果系统不支持引用计数功能则跳过测试
@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
def test_nditer_multi_index_set_refcount():
    # 测试索引变量的引用计数是否减少

    # 初始化索引变量为0
    index = 0
    # 创建一个NumPy迭代器，以多索引模式遍历数组[111, 222, 333, 444]
    i = np.nditer(np.array([111, 222, 333, 444]), flags=['multi_index'])

    # 获取设置索引前后的引用计数
    start_count = sys.getrefcount(index)
    i.multi_index = (index,)
    end_count = sys.getrefcount(index)

    # 断言设置索引后的引用计数与设置前相同
    assert_equal(start_count, end_count)

# 测试一维数组最佳遍历顺序下的多索引
def test_iter_best_order_multi_index_1d():
    # 数组a为长度为4的一维数组
    a = arange(4)
    # 创建一个NumPy迭代器，以多索引模式遍历数组a，只读模式
    i = nditer(a, ['multi_index'], [['readonly']])
    # 断言迭代器遍历结果是否为 [(0,), (1,), (2,), (3,)]
    assert_equal(iter_multi_index(i), [(0,), (1,), (2,), (3,)])
    
    # 数组a按照反向顺序重排
    i = nditer(a[::-1], ['multi_index'], [['readonly']])
    # 断言迭代器遍历结果是否为 [(3,), (2,), (1,), (0,)]
    assert_equal(iter_multi_index(i), [(3,), (2,), (1,), (0,)])

# 测试二维数组最佳遍历顺序下的多索引
def test_iter_best_order_multi_index_2d():
    # 数组a为长度为6的一维数组
    a = arange(6)
    
    # 创建一个NumPy迭代器，以多索引模式遍历形状为(2, 3)的C顺序数组a
    i = nditer(a.reshape(2, 3), ['multi_index'], [['readonly']])
    # 断言迭代器遍历结果是否为 [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert_equal(iter_multi_index(i), [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])
    
    # 创建一个NumPy迭代器，以多索引模式遍历形状为(2, 3)的Fortran顺序数组a
    i = nditer(a.reshape(2, 3).copy(order='F'), ['multi_index'], [['readonly']])
    # 断言迭代器遍历结果是否为 [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]
    assert_equal(iter_multi_index(i), [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)])
    
    # 创建一个NumPy迭代器，以多索引模式遍历反向顺序的形状为(2, 3)的C顺序数组a
    i = nditer(a.reshape(2, 3)[::-1], ['multi_index'], [['readonly']])
    # 断言迭代器遍历结果是否为 [(1, 0), (1, 1), (1, 2), (0, 0), (0, 1), (0, 2)]
    assert_equal(iter_multi_index(i), [(1, 0), (1, 1), (1, 2), (0, 0), (0, 1), (0, 2)])
    
    # 创建一个NumPy迭代器，以多索引模式遍历反向顺序的形状为(2, 3)的C顺序数组a的列反向顺序
    i = nditer(a.reshape(2, 3)[:, ::-1], ['multi_index'], [['readonly']])
    # 断言迭代器遍历结果是否为 [(0, 2), (0, 1), (0, 0), (1, 2), (1, 1), (1, 0)]
    assert_equal(iter_multi_index(i), [(0, 2), (0, 1), (0, 0), (1, 2), (1, 1), (1, 0)])
    
    # 创建一个NumPy迭代器，以多索引模式遍历反向顺序的形状为(2, 3)的C顺序数组a的行列反向顺序
    i = nditer(a.reshape(2, 3)[::-1, ::-1], ['multi_index'], [['readonly']])
    # 断言迭代器遍历结果是否为 [(1, 2), (1, 1), (1, 0), (0, 2), (0, 1), (0, 0)]
    assert_equal(iter_multi_index(i), [(1, 2), (1, 1), (1, 0), (0, 2), (0, 1), (0, 0)])
    
    # 创建一个NumPy迭代器，以多索引模式遍历反向顺序的形状为(2, 3)的Fortran顺序数组a
    i = nditer(a.reshape(2, 3).copy(order='F')[::-1], ['multi_index'], [['readonly']])
    # 断言迭代器遍历结果是否为 [(1, 0), (0, 0), (1, 1), (0, 1), (1, 2), (0, 2)]
    assert_equal(iter_multi_index(i), [(1, 0), (0, 0), (1, 1), (0, 1), (1, 2), (0, 2)])
    
    # 创建一个NumPy迭代器，以多索引模式遍历反向顺序的形状为(2, 3)的Fortran顺序数组a的列反向顺序
    i = nditer(a.reshape(2, 3).copy(order='F')[:, ::-1], ['multi_index'], [['readonly']])
    # 断言迭代器遍历结果是否为 [(0, 2), (1, 2), (0, 1), (1, 1), (0, 0), (1, 0)]
    assert_equal(iter_multi_index(i), [(0, 2), (1, 2), (0, 1), (1, 1), (0, 0), (1, 0)])
    
    # 创建一个NumPy迭代器，以多索引模式遍历反向顺序的形状为(2, 3)的Fortran顺序数组a的行列反向顺序
    i = nditer(a.reshape(2, 3).copy(order='F')[::-1, ::-1], ['multi_index'], [['readonly']])
    # 断言迭代器遍历结果是否为 [(1, 2), (0, 2), (1, 1), (0, 1), (1, 0), (0, 0)]
    assert_equal(iter_multi_index(i), [(1, 2), (0, 2), (1, 1), (0, 1), (1, 0), (0, 0
    # 以迭代器的方式获取多维数组 `a` 的所有索引，并进行断言比较
    assert_equal(iter_multi_index(i),
                            [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1),
                             (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1), (1, 2, 0), (1, 2, 1)])
    # 使用 Fortran 顺序重塑数组 `a` 的形状为 (2, 3, 2)，并创建一个以 'F' 顺序排列的副本，返回一个迭代器 `i`
    i = nditer(a.reshape(2, 3, 2).copy(order='F'), ['multi_index'], [['readonly']])
    # 再次使用迭代器获取多维数组 `a` 的索引，并进行断言比较
    assert_equal(iter_multi_index(i),
                            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 2, 0), (1, 2, 0),
                             (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1), (1, 2, 1)])
    # 使用 C 顺序反转数组 `a` 的形状为 (2, 3, 2)，创建一个迭代器 `i`
    i = nditer(a.reshape(2, 3, 2)[::-1], ['multi_index'], [['readonly']])
    # 再次使用迭代器获取多维数组 `a` 的索引，并进行断言比较
    assert_equal(iter_multi_index(i),
                            [(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1), (1, 2, 0), (1, 2, 1),
                             (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1)])
    # 使用迭代器获取多维数组 `a` 的索引，并进行断言比较，对第二维进行逆序
    i = nditer(a.reshape(2, 3, 2)[:, ::-1], ['multi_index'], [['readonly']])
    # 再次使用迭代器获取多维数组 `a` 的索引，并进行断言比较
    assert_equal(iter_multi_index(i),
                            [(0, 2, 0), (0, 2, 1), (0, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1),
                             (1, 2, 0), (1, 2, 1), (1, 1, 0), (1, 1, 1), (1, 0, 0), (1, 0, 1)])
    # 使用迭代器获取多维数组 `a` 的索引，并进行断言比较，对第三维进行逆序
    i = nditer(a.reshape(2, 3, 2)[:,:, ::-1], ['multi_index'], [['readonly']])
    # 再次使用迭代器获取多维数组 `a` 的索引，并进行断言比较
    assert_equal(iter_multi_index(i),
                            [(0, 0, 1), (0, 0, 0), (0, 1, 1), (0, 1, 0), (0, 2, 1), (0, 2, 0),
                             (1, 0, 1), (1, 0, 0), (1, 1, 1), (1, 1, 0), (1, 2, 1), (1, 2, 0)])
    # 使用 Fortran 逆序重塑数组 `a` 的形状为 (2, 3, 2)，创建一个迭代器 `i`
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[::-1],
                                                    ['multi_index'], [['readonly']])
    # 再次使用迭代器获取多维数组 `a` 的索引，并进行断言比较
    assert_equal(iter_multi_index(i),
                            [(1, 0, 0), (0, 0, 0), (1, 1, 0), (0, 1, 0), (1, 2, 0), (0, 2, 0),
                             (1, 0, 1), (0, 0, 1), (1, 1, 1), (0, 1, 1), (1, 2, 1), (0, 2, 1)])
    # 使用 Fortran 逆序重塑数组 `a` 的形状为 (2, 3, 2)，创建一个迭代器 `i`，对第二维进行逆序
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[:, ::-1],
                                                    ['multi_index'], [['readonly']])
    # 再次使用迭代器获取多维数组 `a` 的索引，并进行断言比较
    assert_equal(iter_multi_index(i),
                            [(0, 2, 0), (1, 2, 0), (0, 1, 0), (1, 1, 0), (0, 0, 0), (1, 0, 0),
                             (0, 2, 1), (1, 2, 1), (0, 1, 1), (1, 1, 1), (0, 0, 1), (1, 0, 1)])
    # 使用 Fortran 逆序重塑数组 `a` 的形状为 (2, 3, 2)，创建一个迭代器 `i`，对第三维进行逆序
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[:,:, ::-1],
                                                    ['multi_index'], [['readonly']])
    # 再次使用迭代器获取多维数组 `a` 的索引，并进行断言比较
    assert_equal(iter_multi_index(i),
                            [(0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1), (1, 2, 1),
                             (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 2, 0), (1, 2, 0)])
def test_iter_best_order_c_index_1d():
    # The C index should be correct with any reordering

    a = arange(4)
    # 创建一个包含四个元素的一维数组

    # 1D order
    i = nditer(a, ['c_index'], [['readonly']])
    # 使用迭代器nditer遍历数组a，按C顺序访问元素
    assert_equal(iter_indices(i), [0, 1, 2, 3])
    # 断言迭代器返回的索引顺序与预期的一致

    # 1D reversed order
    i = nditer(a[::-1], ['c_index'], [['readonly']])
    # 使用迭代器nditer遍历数组a的逆序，按C顺序访问元素
    assert_equal(iter_indices(i), [3, 2, 1, 0])
    # 断言迭代器返回的索引顺序与预期的一致

def test_iter_best_order_c_index_2d():
    # The C index should be correct with any reordering

    a = arange(6)
    # 创建一个包含六个元素的一维数组

    # 2D C-order
    i = nditer(a.reshape(2, 3), ['c_index'], [['readonly']])
    # 使用迭代器nditer遍历将a重塑为2x3的二维数组，按C顺序访问元素
    assert_equal(iter_indices(i), [0, 1, 2, 3, 4, 5])
    # 断言迭代器返回的索引顺序与预期的一致

    # 2D Fortran-order
    i = nditer(a.reshape(2, 3).copy(order='F'),
                                    ['c_index'], [['readonly']])
    # 使用迭代器nditer遍历将a重塑为2x3的二维数组，按Fortran顺序访问元素
    assert_equal(iter_indices(i), [0, 3, 1, 4, 2, 5])
    # 断言迭代器返回的索引顺序与预期的一致

    # 2D reversed C-order
    i = nditer(a.reshape(2, 3)[::-1], ['c_index'], [['readonly']])
    # 使用迭代器nditer遍历将a重塑为2x3的二维数组的逆序，按C顺序访问元素
    assert_equal(iter_indices(i), [3, 4, 5, 0, 1, 2])
    # 断言迭代器返回的索引顺序与预期的一致
    i = nditer(a.reshape(2, 3)[:, ::-1], ['c_index'], [['readonly']])
    # 使用迭代器nditer遍历将a重塑为2x3的二维数组的列逆序，按C顺序访问元素
    assert_equal(iter_indices(i), [2, 1, 0, 5, 4, 3])
    # 断言迭代器返回的索引顺序与预期的一致
    i = nditer(a.reshape(2, 3)[::-1, ::-1], ['c_index'], [['readonly']])
    # 使用迭代器nditer遍历将a重塑为2x3的二维数组的逆序及列逆序，按C顺序访问元素
    assert_equal(iter_indices(i), [5, 4, 3, 2, 1, 0])
    # 断言迭代器返回的索引顺序与预期的一致

    # 2D reversed Fortran-order
    i = nditer(a.reshape(2, 3).copy(order='F')[::-1],
                                    ['c_index'], [['readonly']])
    # 使用迭代器nditer遍历将a重塑为2x3的二维数组的逆序，按Fortran顺序访问元素
    assert_equal(iter_indices(i), [3, 0, 4, 1, 5, 2])
    # 断言迭代器返回的索引顺序与预期的一致
    i = nditer(a.reshape(2, 3).copy(order='F')[:, ::-1],
                                    ['c_index'], [['readonly']])
    # 使用迭代器nditer遍历将a重塑为2x3的二维数组的列逆序，按Fortran顺序访问元素
    assert_equal(iter_indices(i), [2, 5, 1, 4, 0, 3])
    # 断言迭代器返回的索引顺序与预期的一致
    i = nditer(a.reshape(2, 3).copy(order='F')[::-1, ::-1],
                                    ['c_index'], [['readonly']])
    # 使用迭代器nditer遍历将a重塑为2x3的二维数组的逆序及列逆序，按Fortran顺序访问元素
    assert_equal(iter_indices(i), [5, 2, 4, 1, 3, 0])
    # 断言迭代器返回的索引顺序与预期的一致

def test_iter_best_order_c_index_3d():
    # The C index should be correct with any reordering

    a = arange(12)
    # 创建一个包含十二个元素的一维数组

    # 3D C-order
    i = nditer(a.reshape(2, 3, 2), ['c_index'], [['readonly']])
    # 使用迭代器nditer遍历将a重塑为2x3x2的三维数组，按C顺序访问元素
    assert_equal(iter_indices(i),
                            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    # 断言迭代器返回的索引顺序与预期的一致

    # 3D Fortran-order
    i = nditer(a.reshape(2, 3, 2).copy(order='F'),
                                    ['c_index'], [['readonly']])
    # 使用迭代器nditer遍历将a重塑为2x3x2的三维数组，按Fortran顺序访问元素
    assert_equal(iter_indices(i),
                            [0, 6, 2, 8, 4, 10, 1, 7, 3, 9, 5, 11])
    # 断言迭代器返回的索引顺序与预期的一致

    # 3D reversed C-order
    i = nditer(a.reshape(2, 3, 2)[::-1], ['c_index'], [['readonly']])
    # 使用迭代器nditer遍历将a重塑为2x3x2的三维数组的逆序，按C顺序访问元素
    assert_equal(iter_indices(i),
                            [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5])
    # 断言迭代器返回的索引顺序与预期的一致
    i = nditer(a.reshape(2, 3, 2)[:, ::-1], ['c_index'], [['readonly']])
    # 使用迭代器nditer遍历将a重塑为2x3x2的三维数组的列逆序，按C顺序访问元素
    assert_equal(iter_indices(i),
                            [4, 5, 2, 3, 0, 1, 10, 11, 8, 9, 6, 7])
    # 断言迭代器返回的索引顺序与预期的一致
    i = nditer(a.reshape(2, 3, 2)[:,:, ::-1], ['c_index'], [['readonly']])
    # 使用迭代器nditer遍历将a重塑为2x3x2的三维数组的层逆序，按C顺序访问元素
    assert_equal(iter_indices(i),
                            [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10])
    # 断言迭代器返回的索引顺序与预期的一致

    # 3D reversed Fortran-order
    # 断言：验证 iter_indices 函数对于参数 i 的返回结果是否与指定的列表相等
    assert_equal(iter_indices(i),
                            [6, 0, 8, 2, 10, 4, 7, 1, 9, 3, 11, 5])
    # 使用 nditer 函数对数组 a 进行迭代，首先对其进行形状重塑并按列主序复制，然后对结果进行逆序切片迭代
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[:, ::-1],
                                    ['c_index'], [['readonly']])
    # 断言：验证 iter_indices 函数对于参数 i 的返回结果是否与指定的列表相等
    assert_equal(iter_indices(i),
                            [4, 10, 2, 8, 0, 6, 5, 11, 3, 9, 1, 7])
    # 使用 nditer 函数对数组 a 进行迭代，首先对其进行形状重塑并按列主序复制，然后对所有维度进行逆序切片迭代
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[:,:, ::-1],
                                    ['c_index'], [['readonly']])
    # 断言：验证 iter_indices 函数对于参数 i 的返回结果是否与指定的列表相等
    assert_equal(iter_indices(i),
                            [1, 7, 3, 9, 5, 11, 0, 6, 2, 8, 4, 10])
def test_iter_best_order_f_index_1d():
    # 定义测试函数，用于验证 nditer 在 1 维数组上的 Fortran 索引行为

    a = arange(4)
    # 创建一个长度为 4 的一维数组 a

    i = nditer(a, ['f_index'], [['readonly']])
    # 使用 nditer 迭代器创建一个迭代对象 i，要求使用 Fortran 索引，并只读访问数组内容

    assert_equal(iter_indices(i), [0, 1, 2, 3])
    # 验证迭代器 i 的索引序列是否与预期的 [0, 1, 2, 3] 相符

    i = nditer(a[::-1], ['f_index'], [['readonly']])
    # 创建一个对数组 a 反转后的迭代器 i，仍然使用 Fortran 索引并只读访问

    assert_equal(iter_indices(i), [3, 2, 1, 0])
    # 验证迭代器 i 的索引序列是否与预期的 [3, 2, 1, 0] 相符

def test_iter_best_order_f_index_2d():
    # 定义测试函数，用于验证 nditer 在 2 维数组上的 Fortran 索引行为

    a = arange(6)
    # 创建一个长度为 6 的一维数组 a

    i = nditer(a.reshape(2, 3), ['f_index'], [['readonly']])
    # 使用 nditer 迭代器创建一个迭代对象 i，将一维数组 a 重塑为 2x3 的二维数组，并要求使用 Fortran 索引

    assert_equal(iter_indices(i), [0, 2, 4, 1, 3, 5])
    # 验证迭代器 i 的索引序列是否与预期的 [0, 2, 4, 1, 3, 5] 相符

    i = nditer(a.reshape(2, 3).copy(order='F'), ['f_index'], [['readonly']])
    # 创建一个按 Fortran 顺序排列的二维数组的迭代器 i，要求使用 Fortran 索引并只读访问

    assert_equal(iter_indices(i), [0, 1, 2, 3, 4, 5])
    # 验证迭代器 i 的索引序列是否与预期的 [0, 1, 2, 3, 4, 5] 相符

    i = nditer(a.reshape(2, 3)[::-1], ['f_index'], [['readonly']])
    # 创建一个对数组 a 逆序排列后的二维数组的迭代器 i，要求使用 Fortran 索引并只读访问

    assert_equal(iter_indices(i), [1, 3, 5, 0, 2, 4])
    # 验证迭代器 i 的索引序列是否与预期的 [1, 3, 5, 0, 2, 4] 相符

    i = nditer(a.reshape(2, 3)[:, ::-1], ['f_index'], [['readonly']])
    # 创建一个对数组 a 每行逆序排列后的二维数组的迭代器 i，要求使用 Fortran 索引并只读访问

    assert_equal(iter_indices(i), [4, 2, 0, 5, 3, 1])
    # 验证迭代器 i 的索引序列是否与预期的 [4, 2, 0, 5, 3, 1] 相符

    i = nditer(a.reshape(2, 3)[::-1, ::-1], ['f_index'], [['readonly']])
    # 创建一个对数组 a 完全逆序排列后的二维数组的迭代器 i，要求使用 Fortran 索引并只读访问

    assert_equal(iter_indices(i), [5, 3, 1, 4, 2, 0])
    # 验证迭代器 i 的索引序列是否与预期的 [5, 3, 1, 4, 2, 0] 相符

    i = nditer(a.reshape(2, 3).copy(order='F')[::-1], ['f_index'], [['readonly']])
    # 创建一个按 Fortran 顺序排列的二维数组，并对其逆序排列后的迭代器 i，要求使用 Fortran 索引并只读访问

    assert_equal(iter_indices(i), [1, 0, 3, 2, 5, 4])
    # 验证迭代器 i 的索引序列是否与预期的 [1, 0, 3, 2, 5, 4] 相符

    i = nditer(a.reshape(2, 3).copy(order='F')[:, ::-1], ['f_index'], [['readonly']])
    # 创建一个按 Fortran 顺序排列的二维数组，并对每行逆序排列后的迭代器 i，要求使用 Fortran 索引并只读访问

    assert_equal(iter_indices(i), [4, 5, 2, 3, 0, 1])
    # 验证迭代器 i 的索引序列是否与预期的 [4, 5, 2, 3, 0, 1] 相符

    i = nditer(a.reshape(2, 3).copy(order='F')[::-1, ::-1], ['f_index'], [['readonly']])
    # 创建一个按 Fortran 顺序排列的二维数组，并对其完全逆序排列后的迭代器 i，要求使用 Fortran 索引并只读访问

    assert_equal(iter_indices(i), [5, 4, 3, 2, 1, 0])
    # 验证迭代器 i 的索引序列是否与预期的 [5, 4, 3, 2, 1, 0] 相符

def test_iter_best_order_f_index_3d():
    # 定义测试函数，用于验证 nditer 在 3 维数组上的 Fortran 索引行为

    a = arange(12)
    # 创建一个长度为 12 的一维数组 a

    i = nditer(a.reshape(2, 3, 2), ['f_index'], [['readonly']])
    # 使用 nditer 迭代器创建一个迭代对象 i，将一维数组 a 重塑为 2x3x2 的三维数组，并要求使用 Fortran 索引

    assert_equal(iter_indices(i),
                            [0, 6, 2, 8, 4, 10, 1, 7, 3, 9, 5, 11])
    # 验证迭代器 i 的索引序列是否与预期的 [0, 6, 2, 8, 4, 10, 1, 7, 3, 9, 5, 11] 相符

    i = nditer(a.reshape(2, 3, 2).copy(order='F'), ['f_index'], [['readonly']])
    # 创建一个按 Fortran 顺序排列的三维数组的迭代器 i，要求使用 Fortran 索引并只读访问

    assert_equal(iter_indices(i),
                            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    # 验证迭代器 i 的索引序列是否与预期的 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] 相符

    i = nditer(a.reshape(2, 3, 2)[::-1], ['f_index'], [['readonly']])
    # 创建一个对数组 a 逆序排列后的三维数组的迭代器 i，要求使用 Fortran 索引并只读访问

    assert_equal(iter_indices(i),
                            [1, 7, 3, 9, 5, 11, 0, 6, 2, 8, 4, 10])
    # 验证迭代器 i 的索引序列是否与预期的 [1, 7, 3, 9, 5, 11, 0, 6, 2, 8, 4, 10] 相符

    i
    # 使用 nditer 函数迭代数组 a 的拷贝，reshape 后的形状为 (2, 3, 2)，按列优先 ('F' order) 排列，然后进行逆序操作
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[::-1],
               ['f_index'], [['readonly']])
    # 断言迭代后的索引顺序是否符合预期
    assert_equal(iter_indices(i),
                 [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10])

    # 使用 nditer 函数迭代数组 a 的拷贝，reshape 后的形状为 (2, 3, 2)，按列优先 ('F' order) 排列，然后在第二维度进行逆序操作
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[:, ::-1],
               ['f_index'], [['readonly']])
    # 断言迭代后的索引顺序是否符合预期
    assert_equal(iter_indices(i),
                 [4, 5, 2, 3, 0, 1, 10, 11, 8, 9, 6, 7])

    # 使用 nditer 函数迭代数组 a 的拷贝，reshape 后的形状为 (2, 3, 2)，按列优先 ('F' order) 排列，然后在第三维度进行逆序操作
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[:, :, ::-1],
               ['f_index'], [['readonly']])
    # 断言迭代后的索引顺序是否符合预期
    assert_equal(iter_indices(i),
                 [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5])
def test_iter_no_inner_full_coalesce():
    # Check no_inner iterators which coalesce into a single inner loop

    # Iterate over various shapes
    for shape in [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 3), (2, 3, 2, 2, 3)]:
        size = np.prod(shape)
        a = arange(size)
        # Test each combination of forward and backwards indexing
        for dirs in range(2**len(shape)):
            dirs_index = [slice(None)]*len(shape)
            # Create index based on directions bitmask
            for bit in range(len(shape)):
                if ((2**bit) & dirs):
                    dirs_index[bit] = slice(None, None, -1)
            dirs_index = tuple(dirs_index)

            # Apply indexing to reshape view of array
            aview = a.reshape(shape)[dirs_index]

            # Test with C-order
            i = nditer(aview, ['external_loop'], [['readonly']])
            assert_equal(i.ndim, 1)
            assert_equal(i[0].shape, (size,))

            # Test with Fortran-order
            i = nditer(aview.T, ['external_loop'], [['readonly']])
            assert_equal(i.ndim, 1)
            assert_equal(i[0].shape, (size,))

            # Test with other orders if array has more than 2 dimensions
            if len(shape) > 2:
                i = nditer(aview.swapaxes(0, 1),
                                    ['external_loop'], [['readonly']])
                assert_equal(i.ndim, 1)
                assert_equal(i[0].shape, (size,))

def test_iter_no_inner_dim_coalescing():
    # Check no_inner iterators whose dimensions may not coalesce completely

    # Test case where last element in a dimension is skipped
    a = arange(24).reshape(2, 3, 4)[:,:, :-1]
    i = nditer(a, ['external_loop'], [['readonly']])
    assert_equal(i.ndim, 2)
    assert_equal(i[0].shape, (3,))

    a = arange(24).reshape(2, 3, 4)[:, :-1,:]
    i = nditer(a, ['external_loop'], [['readonly']])
    assert_equal(i.ndim, 2)
    assert_equal(i[0].shape, (8,))

    a = arange(24).reshape(2, 3, 4)[:-1,:,:]
    i = nditer(a, ['external_loop'], [['readonly']])
    assert_equal(i.ndim, 1)
    assert_equal(i[0].shape, (12,))

    # Even with many 1-sized dimensions, should still coalesce
    a = arange(24).reshape(1, 1, 2, 1, 1, 3, 1, 1, 4, 1, 1)
    i = nditer(a, ['external_loop'], [['readonly']])
    assert_equal(i.ndim, 1)
    assert_equal(i[0].shape, (24,))

def test_iter_dim_coalescing():
    # Check that the correct number of dimensions are coalesced

    # Testing with multi-index tracking which disables coalescing
    a = arange(24).reshape(2, 3, 4)
    i = nditer(a, ['multi_index'], [['readonly']])
    assert_equal(i.ndim, 3)

    # Testing with C-index tracking, should coalesce into 1 dimension
    a3d = arange(24).reshape(2, 3, 4)
    i = nditer(a3d, ['c_index'], [['readonly']])
    assert_equal(i.ndim, 1)

    # Testing with array transpose and C-index tracking, should coalesce into 3 dimensions
    i = nditer(a3d.swapaxes(0, 1), ['c_index'], [['readonly']])
    assert_equal(i.ndim, 3)

    # Testing with array transpose and F-index tracking, should coalesce into 1 dimension
    i = nditer(a3d.T, ['f_index'], [['readonly']])
    assert_equal(i.ndim, 1)

    # Testing with array transpose, swap axes, and F-index tracking, should coalesce into 3 dimensions
    i = nditer(a3d.T.swapaxes(0, 1), ['f_index'], [['readonly']])
    assert_equal(i.ndim, 3)
    # 当强制使用 C 或 F 排序时，仍然可能发生合并操作
    a3d = arange(24).reshape(2, 3, 4)
    # 创建一个迭代器对象 i，迭代顺序为 C（按行）
    i = nditer(a3d, order='C')
    # 断言迭代器的维度应为 1
    assert_equal(i.ndim, 1)
    
    # 对 a3d 的转置进行迭代，顺序为 C（按行）
    i = nditer(a3d.T, order='C')
    # 断言迭代器的维度应为 3
    assert_equal(i.ndim, 3)
    
    # 对 a3d 进行迭代，顺序为 F（按列）
    i = nditer(a3d, order='F')
    # 断言迭代器的维度应为 3
    assert_equal(i.ndim, 3)
    
    # 对 a3d 的转置进行迭代，顺序为 F（按列）
    i = nditer(a3d.T, order='F')
    # 断言迭代器的维度应为 1
    assert_equal(i.ndim, 1)
    
    # 使用默认顺序（A），对 a3d 进行迭代
    i = nditer(a3d, order='A')
    # 断言迭代器的维度应为 1
    assert_equal(i.ndim, 1)
    
    # 对 a3d 的转置进行迭代，使用默认顺序（A）
    i = nditer(a3d.T, order='A')
    # 断言迭代器的维度应为 1
    assert_equal(i.ndim, 1)
def test_iter_broadcasting():
    # Standard NumPy broadcasting rules

    # 1D with scalar
    # 使用nditer迭代器处理1维数组和标量
    i = nditer([arange(6), np.int32(2)], ['multi_index'], [['readonly']]*2)
    assert_equal(i.itersize, 6)  # 断言迭代器的大小为6
    assert_equal(i.shape, (6,))  # 断言迭代器的形状为(6,)

    # 2D with scalar
    # 使用nditer迭代器处理2维数组和标量
    i = nditer([arange(6).reshape(2, 3), np.int32(2)],
                        ['multi_index'], [['readonly']]*2)
    assert_equal(i.itersize, 6)  # 断言迭代器的大小为6
    assert_equal(i.shape, (2, 3))  # 断言迭代器的形状为(2, 3)

    # 2D with 1D
    # 使用nditer迭代器处理2维数组和1维数组
    i = nditer([arange(6).reshape(2, 3), arange(3)],
                        ['multi_index'], [['readonly']]*2)
    assert_equal(i.itersize, 6)  # 断言迭代器的大小为6
    assert_equal(i.shape, (2, 3))  # 断言迭代器的形状为(2, 3)
    i = nditer([arange(2).reshape(2, 1), arange(3)],
                        ['multi_index'], [['readonly']]*2)
    assert_equal(i.itersize, 6)  # 断言迭代器的大小为6
    assert_equal(i.shape, (2, 3))  # 断言迭代器的形状为(2, 3)

    # 2D with 2D
    # 使用nditer迭代器处理2维数组和2维数组
    i = nditer([arange(2).reshape(2, 1), arange(3).reshape(1, 3)],
                        ['multi_index'], [['readonly']]*2)
    assert_equal(i.itersize, 6)  # 断言迭代器的大小为6
    assert_equal(i.shape, (2, 3))  # 断言迭代器的形状为(2, 3)

    # 3D with scalar
    # 使用nditer迭代器处理3维数组和标量
    i = nditer([np.int32(2), arange(24).reshape(4, 2, 3)],
                        ['multi_index'], [['readonly']]*2)
    assert_equal(i.itersize, 24)  # 断言迭代器的大小为24
    assert_equal(i.shape, (4, 2, 3))  # 断言迭代器的形状为(4, 2, 3)

    # 3D with 1D
    # 使用nditer迭代器处理3维数组和1维数组
    i = nditer([arange(3), arange(24).reshape(4, 2, 3)],
                        ['multi_index'], [['readonly']]*2)
    assert_equal(i.itersize, 24)  # 断言迭代器的大小为24
    assert_equal(i.shape, (4, 2, 3))  # 断言迭代器的形状为(4, 2, 3)
    i = nditer([arange(3), arange(8).reshape(4, 2, 1)],
                        ['multi_index'], [['readonly']]*2)
    assert_equal(i.itersize, 24)  # 断言迭代器的大小为24
    assert_equal(i.shape, (4, 2, 3))  # 断言迭代器的形状为(4, 2, 3)

    # 3D with 2D
    # 使用nditer迭代器处理3维数组和2维数组
    i = nditer([arange(6).reshape(2, 3), arange(24).reshape(4, 2, 3)],
                        ['multi_index'], [['readonly']]*2)
    assert_equal(i.itersize, 24)  # 断言迭代器的大小为24
    assert_equal(i.shape, (4, 2, 3))  # 断言迭代器的形状为(4, 2, 3)
    i = nditer([arange(2).reshape(2, 1), arange(24).reshape(4, 2, 3)],
                        ['multi_index'], [['readonly']]*2)
    assert_equal(i.itersize, 24)  # 断言迭代器的大小为24
    assert_equal(i.shape, (4, 2, 3))  # 断言迭代器的形状为(4, 2, 3)
    i = nditer([arange(3).reshape(1, 3), arange(8).reshape(4, 2, 1)],
                        ['multi_index'], [['readonly']]*2)
    assert_equal(i.itersize, 24)  # 断言迭代器的大小为24
    assert_equal(i.shape, (4, 2, 3))  # 断言迭代器的形状为(4, 2, 3)

    # 3D with 3D
    # 使用nditer迭代器处理3维数组和3维数组
    i = nditer([arange(2).reshape(1, 2, 1), arange(3).reshape(1, 1, 3),
                        arange(4).reshape(4, 1, 1)],
                        ['multi_index'], [['readonly']]*3)
    assert_equal(i.itersize, 24)  # 断言迭代器的大小为24
    assert_equal(i.shape, (4, 2, 3))  # 断言迭代器的形状为(4, 2, 3)
    i = nditer([arange(6).reshape(1, 2, 3), arange(4).reshape(4, 1, 1)],
                        ['multi_index'], [['readonly']]*2)
    assert_equal(i.itersize, 24)  # 断言迭代器的大小为24
    assert_equal(i.shape, (4, 2, 3))  # 断言迭代器的形状为(4, 2, 3)
    i = nditer([arange(24).reshape(4, 2, 3), arange(12).reshape(4, 1, 3)],
                        ['multi_index'], [['readonly']]*2)
    assert_equal(i.itersize, 24)  # 断言迭代器的大小为24
    assert_equal(i.shape, (4, 2, 3))  # 断言迭代器的形状为(4, 2, 3)

def test_iter_itershape():
    # Check that allocated outputs work with a specified shape
    # 检查分配的输出是否与指定的形状匹配
    # 创建一个形状为 (2, 3) 的 int16 类型的 NumPy 数组，并按行优先顺序填充 0 到 5 的整数
    a = np.arange(6, dtype='i2').reshape(2, 3)
    
    # 使用 nditer 对象迭代数组 a 和一个空数组，设置对 a 只读，对空数组可写且需要分配空间
    i = nditer([a, None], [], [['readonly'], ['writeonly', 'allocate']],
               op_axes=[[0, 1, None], None],
               itershape=(-1, -1, 4))
    
    # 断言空数组的形状应为 (2, 3, 4)
    assert_equal(i.operands[1].shape, (2, 3, 4))
    
    # 断言空数组的步长应为 (24, 8, 2)
    assert_equal(i.operands[1].strides, (24, 8, 2))
    
    # 使用 nditer 对象迭代数组 a 的转置和一个空数组，设置对 a 的转置只读，对空数组可写且需要分配空间
    i = nditer([a.T, None], [], [['readonly'], ['writeonly', 'allocate']],
               op_axes=[[0, 1, None], None],
               itershape=(-1, -1, 4))
    
    # 断言空数组的形状应为 (3, 2, 4)
    assert_equal(i.operands[1].shape, (3, 2, 4))
    
    # 断言空数组的步长应为 (8, 24, 2)
    assert_equal(i.operands[1].strides, (8, 24, 2))
    
    # 使用 nditer 对象迭代数组 a 的转置和一个空数组，设置对 a 的转置只读，对空数组可写且需要分配空间，按 Fortran（列优先）顺序
    i = nditer([a.T, None], [], [['readonly'], ['writeonly', 'allocate']],
               order='F',
               op_axes=[[0, 1, None], None],
               itershape=(-1, -1, 4))
    
    # 断言空数组的形状应为 (3, 2, 4)
    assert_equal(i.operands[1].shape, (3, 2, 4))
    
    # 断言空数组的步长应为 (2, 6, 12)
    assert_equal(i.operands[1].strides, (2, 6, 12))
    
    # 如果在 itershape 中指定为 1，那么不允许将该维度广播为更大的值，应抛出 ValueError 异常
    assert_raises(ValueError, nditer, [a, None], [],
                  [['readonly'], ['writeonly', 'allocate']],
                  op_axes=[[0, 1, None], None],
                  itershape=(-1, 1, 4))
    
    # 测试修复的 bug：在没有 op_axes 但有 itershape 的情况下，它们被正确地设为 NULL
    i = np.nditer([np.ones(2), None, None], itershape=(2,))
def test_iter_broadcasting_errors():
    # Check that errors are thrown for bad broadcasting shapes

    # 1D with 1D
    assert_raises(ValueError, nditer, [arange(2), arange(3)],
                    [], [['readonly']]*2)
    # 2D with 1D
    assert_raises(ValueError, nditer,
                    [arange(6).reshape(2, 3), arange(2)],
                    [], [['readonly']]*2)
    # 2D with 2D
    assert_raises(ValueError, nditer,
                    [arange(6).reshape(2, 3), arange(9).reshape(3, 3)],
                    [], [['readonly']]*2)
    assert_raises(ValueError, nditer,
                    [arange(6).reshape(2, 3), arange(4).reshape(2, 2)],
                    [], [['readonly']]*2)
    # 3D with 3D
    assert_raises(ValueError, nditer,
                    [arange(36).reshape(3, 3, 4), arange(24).reshape(2, 3, 4)],
                    [], [['readonly']]*2)
    assert_raises(ValueError, nditer,
                    [arange(8).reshape(2, 4, 1), arange(24).reshape(2, 3, 4)],
                    [], [['readonly']]*2)

    # Verify that the error message mentions the right shapes
    try:
        nditer([arange(2).reshape(1, 2, 1),
                arange(3).reshape(1, 3),
                arange(6).reshape(2, 3)],
               [],
               [['readonly'], ['readonly'], ['writeonly', 'no_broadcast']])
        raise AssertionError('Should have raised a broadcast error')
    except ValueError as e:
        msg = str(e)
        # The message should contain the shape of the 3rd operand
        assert_(msg.find('(2,3)') >= 0,
                'Message "%s" doesn\'t contain operand shape (2,3)' % msg)
        # The message should contain the broadcast shape
        assert_(msg.find('(1,2,3)') >= 0,
                'Message "%s" doesn\'t contain broadcast shape (1,2,3)' % msg)

    try:
        nditer([arange(6).reshape(2, 3), arange(2)],
               [],
               [['readonly'], ['readonly']],
               op_axes=[[0, 1], [0, np.newaxis]],
               itershape=(4, 3))
        raise AssertionError('Should have raised a broadcast error')
    except ValueError as e:
        msg = str(e)
        # The message should contain "shape->remappedshape" for each operand
        assert_(msg.find('(2,3)->(2,3)') >= 0,
            'Message "%s" doesn\'t contain operand shape (2,3)->(2,3)' % msg)
        assert_(msg.find('(2,)->(2,newaxis)') >= 0,
                ('Message "%s" doesn\'t contain remapped operand shape' +
                '(2,)->(2,newaxis)') % msg)
        # The message should contain the itershape parameter
        assert_(msg.find('(4,3)') >= 0,
                'Message "%s" doesn\'t contain itershape parameter (4,3)' % msg)

    try:
        nditer([np.zeros((2, 1, 1)), np.zeros((2,))],
               [],
               [['writeonly', 'no_broadcast'], ['readonly']])
        raise AssertionError('Should have raised a broadcast error')
    except ValueError as e:
        msg = str(e)
        # The message should contain details about the broadcasting error
        assert_('broadcast' in msg.lower(),
                'Message "%s" should mention broadcasting' % msg)
    # 捕获 ValueError 异常并将异常信息转换为字符串
    except ValueError as e:
        msg = str(e)
        # 断言异常信息中包含特定的操作数形状 (2,1,1)
        assert_(msg.find('(2,1,1)') >= 0,
            'Message "%s" doesn\'t contain operand shape (2,1,1)' % msg)
        # 断言异常信息中包含特定的广播形状 (2,1,2)
        assert_(msg.find('(2,1,2)') >= 0,
                'Message "%s" doesn\'t contain the broadcast shape (2,1,2)' % msg)
def test_iter_flags_errors():
    # 检查错误标志组合是否会产生错误

    a = arange(6)
    # 操作数不足
    assert_raises(ValueError, nditer, [], [], [])
    # 操作数过多
    assert_raises(ValueError, nditer, [a]*100, [], [['readonly']]*100)
    # 错误的全局标志
    assert_raises(ValueError, nditer, [a], ['bad flag'], [['readonly']])
    # 错误的操作标志
    assert_raises(ValueError, nditer, [a], [], [['readonly', 'bad flag']])
    # 错误的顺序参数
    assert_raises(ValueError, nditer, [a], [], [['readonly']], order='G')
    # 错误的类型转换参数
    assert_raises(ValueError, nditer, [a], [], [['readonly']], casting='noon')
    # 操作标志数量必须与操作数一致
    assert_raises(ValueError, nditer, [a]*3, [], [['readonly']]*2)
    # 不能同时跟踪 C 和 F 索引
    assert_raises(ValueError, nditer, a, ['c_index', 'f_index'], [['readonly']])
    # 内部迭代和多索引/索引不兼容
    assert_raises(ValueError, nditer, a, ['external_loop', 'multi_index'], [['readonly']])
    assert_raises(ValueError, nditer, a, ['external_loop', 'c_index'], [['readonly']])
    assert_raises(ValueError, nditer, a, ['external_loop', 'f_index'], [['readonly']])
    # 每个操作数必须指定 readwrite/readonly/writeonly 中的一种
    assert_raises(ValueError, nditer, a, [], [[]])
    assert_raises(ValueError, nditer, a, [], [['readonly', 'writeonly']])
    assert_raises(ValueError, nditer, a, [], [['readonly', 'readwrite']])
    assert_raises(ValueError, nditer, a, [], [['writeonly', 'readwrite']])
    assert_raises(ValueError, nditer, a, [], [['readonly', 'writeonly', 'readwrite']])
    # Python 标量总是只读的
    assert_raises(TypeError, nditer, 1.5, [], [['writeonly']])
    assert_raises(TypeError, nditer, 1.5, [], [['readwrite']])
    # 数组标量总是只读的
    assert_raises(TypeError, nditer, np.int32(1), [], [['writeonly']])
    assert_raises(TypeError, nditer, np.int32(1), [], [['readwrite']])
    # 检查只读数组
    a.flags.writeable = False
    assert_raises(ValueError, nditer, a, [], [['writeonly']])
    assert_raises(ValueError, nditer, a, [], [['readwrite']])
    a.flags.writeable = True
    # 多索引仅在使用 multi_index 标志时可用
    i = nditer(arange(6), [], [['readonly']])
    assert_raises(ValueError, lambda i:i.multi_index, i)
    # 索引仅在使用 index 标志时可用
    assert_raises(ValueError, lambda i:i.index, i)

    # GotoCoords 和 GotoIndex 与缓冲或 no_inner 不兼容
    def assign_multi_index(i):
        i.multi_index = (0,)

    def assign_index(i):
        i.index = 0

    def assign_iterindex(i):
        i.iterindex = 0

    def assign_iterrange(i):
        i.iterrange = (0, 1)

    i = nditer(arange(6), ['external_loop'])
    assert_raises(ValueError, assign_multi_index, i)
    assert_raises(ValueError, assign_index, i)
    # 使用 assert_raises 检查 assign_iterindex 函数对于 ValueError 的异常处理
    assert_raises(ValueError, assign_iterindex, i)
    
    # 使用 assert_raises 检查 assign_iterrange 函数对于 ValueError 的异常处理
    assert_raises(ValueError, assign_iterrange, i)
    
    # 使用 nditer 函数创建一个迭代器 i，对 arange(6) 的数组进行迭代，使用 'buffered' 模式
    i = nditer(arange(6), ['buffered'])
    
    # 使用 assert_raises 检查 assign_multi_index 函数对于 ValueError 的异常处理
    assert_raises(ValueError, assign_multi_index, i)
    
    # 使用 assert_raises 检查 assign_index 函数对于 ValueError 的异常处理
    assert_raises(ValueError, assign_index, i)
    
    # 再次使用 assert_raises 检查 assign_iterrange 函数对于 ValueError 的异常处理
    assert_raises(ValueError, assign_iterrange, i)
    
    # 当数组大小为零时无法进行迭代，使用 assert_raises 检查 nditer 函数对于 ValueError 的异常处理
    assert_raises(ValueError, nditer, np.array([]))
# 定义一个测试函数，用于测试nditer迭代器的切片操作
def test_iter_slice():
    # 创建三个NumPy数组a, b, c，分别包含0到2的整数和0.0到2.0的浮点数
    a, b, c = np.arange(3), np.arange(3), np.arange(3.)
    # 使用nditer迭代器同时遍历a, b, c数组，设置为可读写模式
    i = nditer([a, b, c], [], ['readwrite'])
    # 进入nditer的上下文管理器，确保资源正确释放
    with i:
        # 修改i的前两个元素为(3, 3)
        i[0:2] = (3, 3)
        # 断言a的值变为[3, 1, 2]
        assert_equal(a, [3, 1, 2])
        # 断言b的值变为[3, 1, 2]
        assert_equal(b, [3, 1, 2])
        # 断言c的值未变，仍为[0, 1, 2]
        assert_equal(c, [0, 1, 2])
        # 修改i的第二个元素为12
        i[1] = 12
        # 断言i的前两个元素的值为[3, 12]
        assert_equal(i[0:2], [3, 12])

# 定义一个测试函数，用于测试nditer迭代器的映射赋值操作
def test_iter_assign_mapping():
    # 创建一个形状为(4, 3, 2)的三维浮点数组a
    a = np.arange(24, dtype='f8').reshape(2, 3, 4).T
    # 使用nditer迭代器遍历a，设置为可读写和更新时复制模式，转换为float32类型
    it = np.nditer(a, [], [['readwrite', 'updateifcopy']],
                       casting='same_kind', op_dtypes=[np.dtype('f4')])
    # 进入nditer的上下文管理器，确保资源正确释放
    with it:
        # 将a的所有元素设置为3
        it.operands[0][...] = 3
        # 将a的所有元素设置为14
        it.operands[0][...] = 14
    # 断言a的所有元素值都为14
    assert_equal(a, 14)
    # 重新创建一个nditer迭代器，遍历a，设置为可读写和更新时复制模式，转换为float32类型
    it = np.nditer(a, [], [['readwrite', 'updateifcopy']],
                       casting='same_kind', op_dtypes=[np.dtype('f4')])
    # 进入nditer的上下文管理器，确保资源正确释放
    with it:
        # 获取a的倒数第二个到第二个元素的切片x
        x = it.operands[0][-1:1]
        # 将x的所有元素设置为14
        x[...] = 14
        # 将a的所有元素设置为-1234
        it.operands[0][...] = -1234
    # 断言a的所有元素值都为-1234
    assert_equal(a, -1234)
    # 检查dealloc时没有警告
    x = None
    it = None

# 定义一个测试函数，用于测试nditer迭代器在不同字节顺序、对齐和连续性变化下的工作
def test_iter_nbo_align_contig():
    # 检查通过请求特定dtype来改变字节顺序
    a = np.arange(6, dtype='f4')
    # 字节交换数组a，并创建一个视图以请求新的字节顺序
    au = a.byteswap().view(a.dtype.newbyteorder())
    # 断言a和au的字节顺序不同
    assert_(a.dtype.byteorder != au.dtype.byteorder)
    # 使用nditer迭代器遍历au，设置为可读写和更新时复制模式，类型转换为float32
    i = nditer(au, [], [['readwrite', 'updateifcopy']],
                        casting='equiv',
                        op_dtypes=[np.dtype('f4')])
    # 进入nditer的上下文管理器，确保资源正确释放
    with i:
        # 上下文管理器在退出时触发WRITEBACKIFCOPY模式
        # 断言i的dtype字节顺序与a相同
        assert_equal(i.dtypes[0].byteorder, a.dtype.byteorder)
        # 断言i的操作数的dtype字节顺序与a相同
        assert_equal(i.operands[0].dtype.byteorder, a.dtype.byteorder)
        # 断言i的操作数与a相同
        assert_equal(i.operands[0], a)
        # 将i的操作数的所有元素设置为2
        i.operands[0][:] = 2
    # 断言au的所有元素值都为2
    assert_equal(au, [2]*6)
    # 删除i，不应引发警告
    del i

    # 通过请求NBO来改变字节顺序
    a = np.arange(6, dtype='f4')
    # 字节交换数组a，并创建一个视图以请求新的字节顺序
    au = a.byteswap().view(a.dtype.newbyteorder())
    # 断言a和au的字节顺序不同
    assert_(a.dtype.byteorder != au.dtype.byteorder)
    # 使用nditer迭代器遍历au，设置为可读写、更新时复制和NBO模式
    with nditer(au, [], [['readwrite', 'updateifcopy', 'nbo']],
                        casting='equiv') as i:
        # 上下文管理器在退出时触发UPDATEIFCOPY模式
        # 断言i的dtype字节顺序与a相同
        assert_equal(i.dtypes[0].byteorder, a.dtype.byteorder)
        # 断言i的操作数的dtype字节顺序与a相同
        assert_equal(i.operands[0].dtype.byteorder, a.dtype.byteorder)
        # 断言i的操作数与a相同
        assert_equal(i.operands[0], a)
        # 将i的操作数的所有元素设置为2
        i.operands[0][:] = 2
    # 断言au的所有元素值都为2
    assert_equal(au, [2]*6)

    # 未对齐的输入
    # 创建一个未对齐的字节的数组a，将其转换为float32类型，并设置其值为0到5
    a = np.zeros((6*4+1,), dtype='i1')[1:]
    a.dtype = 'f4'
    a[:] = np.arange(6, dtype='f4')
    # 断言a不是对齐的
    assert_(not a.flags.aligned)
    # 没有'aligned'标志时，不应复制
    i = nditer(a, [], [['readonly']])
    # 断言i的操作数不是对齐的
    assert_(not i.operands[0].flags.aligned)
    # 断言i的操作数与a相同
    assert_equal(i.operands[0], a)
    # 使用'aligned'标志时，应该复制
    with nditer(a, [], [['readwrite', 'updateifcopy', 'aligned']]) as i:
        # 断言i的操作数是对齐的
        assert_(i.operands[0].flags.aligned)
        # 上下文管理器在退出时触发UPDATEIFCOPY模式
        # 断言i的操作数与a相同
        assert_equal(i.operands[0], a)
        # 将i的操作数的所有元素设置为3
        i.operands[0][:] = 3
    # 断言检查数组 a 是否等于 [3]*6
    assert_equal(a, [3]*6)

    # 不连续的输入
    # 创建一个包含 12 个元素的数组 a
    a = arange(12)

    # 如果数组是连续的，nditer 不会复制数据
    # 创建一个 nditer 对象 i，迭代数组 a 的前 6 个元素，只读模式
    i = nditer(a[:6], [], [['readonly']])
    # 断言操作数的连续性
    assert_(i.operands[0].flags.contiguous)
    # 断言操作数的值与数组 a 的前 6 个元素相等
    assert_equal(i.operands[0], a[:6])

    # 如果数组不连续，应该进行缓冲
    # 创建一个 nditer 对象 i，迭代数组 a 的每隔两个元素的部分，使用缓冲和外部循环
    i = nditer(a[::2], ['buffered', 'external_loop'],
                        [['readonly', 'contig']],
                        buffersize=10)
    # 断言操作数的连续性
    assert_(i[0].flags.contiguous)
    # 断言操作数的值与数组 a 每隔两个元素的部分相等
    assert_equal(i[0], a[::2])
# 检查数组是否按要求进行类型转换

# 没有类型转换，'f4' -> 'f4'
a = np.arange(6, dtype='f4').reshape(2, 3)
# 创建一个迭代器对象，允许读写操作，并指定操作数的数据类型为 'f4'
i = nditer(a, [], [['readwrite']], op_dtypes=[np.dtype('f4')])
with i:
    # 断言操作数和原始数组相同
    assert_equal(i.operands[0], a)
    # 断言操作数的数据类型为 'f4'
    assert_equal(i.operands[0].dtype, np.dtype('f4'))

# 字节顺序转换，'<f4' -> '>f4'
a = np.arange(6, dtype='<f4').reshape(2, 3)
with nditer(a, [], [['readwrite', 'updateifcopy']],
        casting='equiv',
        op_dtypes=[np.dtype('>f4')]) as i:
    # 断言操作数和原始数组相同
    assert_equal(i.operands[0], a)
    # 断言操作数的数据类型为 '>f4'
    assert_equal(i.operands[0].dtype, np.dtype('>f4'))

# 安全的类型转换，'f4' -> 'f8'
a = np.arange(24, dtype='f4').reshape(2, 3, 4).swapaxes(1, 2)
# 创建一个迭代器对象，只读访问，需要复制数据，类型转换为 'f8'
i = nditer(a, [], [['readonly', 'copy']],
        casting='safe',
        op_dtypes=[np.dtype('f8')])
assert_equal(i.operands[0], a)
assert_equal(i.operands[0].dtype, np.dtype('f8'))
# 临时数据的内存布局应该与 a 相同，但负步长会转换为正步长
assert_equal(i.operands[0].strides, (96, 8, 32))
# 修改数组 a 的视图，以反转其内容
a = a[::-1,:, ::-1]
# 使用相同的参数创建另一个迭代器对象
i = nditer(a, [], [['readonly', 'copy']],
        casting='safe',
        op_dtypes=[np.dtype('f8')])
assert_equal(i.operands[0], a)
assert_equal(i.operands[0].dtype, np.dtype('f8'))
assert_equal(i.operands[0].strides, (96, 8, 32))

# 相同类型的转换链，'f8' -> 'f4' -> 'f8'
a = np.arange(24, dtype='f8').reshape(2, 3, 4).T
with nditer(a, [],
        [['readwrite', 'updateifcopy']],
        casting='same_kind',
        op_dtypes=[np.dtype('f4')]) as i:
    assert_equal(i.operands[0], a)
    assert_equal(i.operands[0].dtype, np.dtype('f4'))
    assert_equal(i.operands[0].strides, (4, 16, 48))
    # 检查 WRITEBACKIFCOPY 在退出时是否激活
    i.operands[0][2, 1, 1] = -12.5
    assert_(a[2, 1, 1] != -12.5)
assert_equal(a[2, 1, 1], -12.5)

# 不安全的类型转换，'i4' -> 'f4'
a = np.arange(6, dtype='i4')[::-2]
with nditer(a, [],
        [['writeonly', 'updateifcopy']],
        casting='unsafe',
        op_dtypes=[np.dtype('f4')]) as i:
    # 断言操作数的数据类型为 'f4'
    assert_equal(i.operands[0].dtype, np.dtype('f4'))
    # 即使在 'a' 中步长是负数，在临时数组中步长也会变成正数
    assert_equal(i.operands[0].strides, (4,))
    # 将临时数组的数据设置为 [1, 2, 3]
    i.operands[0][:] = [1, 2, 3]
assert_equal(a, [1, 2, 3])

# 检查类型转换时是否捕获无效的转换错误

# 需要允许复制以进行类型转换
assert_raises(TypeError, nditer, arange(2, dtype='f4'), [],
            [['readonly']], op_dtypes=[np.dtype('f8')])
# 需要允许类型转换以进行类型转换
assert_raises(TypeError, nditer, arange(2, dtype='f4'), [],
            [['readonly', 'copy']], casting='no',
            op_dtypes=[np.dtype('f8')])
    # 抛出 TypeError 异常，测试 numpy 的 nditer 函数
    assert_raises(TypeError, nditer, arange(2, dtype='f4'), [],
                    [['readonly', 'copy']], casting='equiv',
                    op_dtypes=[np.dtype('f8')])
    # 抛出 TypeError 异常，测试 numpy 的 nditer 函数
    assert_raises(TypeError, nditer, arange(2, dtype='f8'), [],
                    [['writeonly', 'updateifcopy']],
                    casting='no',
                    op_dtypes=[np.dtype('f4')])
    # 抛出 TypeError 异常，测试 numpy 的 nditer 函数
    assert_raises(TypeError, nditer, arange(2, dtype='f8'), [],
                    [['writeonly', 'updateifcopy']],
                    casting='equiv',
                    op_dtypes=[np.dtype('f4')])
    # 抛出 TypeError 异常，测试 numpy 的 nditer 函数，要求 'f4' -> '>f4' 在 casting='no' 时不能工作
    assert_raises(TypeError, nditer, arange(2, dtype='<f4'), [],
                    [['readonly', 'copy']], casting='no',
                    op_dtypes=[np.dtype('>f4')])
    # 抛出 TypeError 异常，测试 numpy 的 nditer 函数，'f4' -> 'f8' 是安全转换，但 'f8' -> 'f4' 不是
    assert_raises(TypeError, nditer, arange(2, dtype='f4'), [],
                    [['readwrite', 'updateifcopy']],
                    casting='safe',
                    op_dtypes=[np.dtype('f8')])
    # 抛出 TypeError 异常，测试 numpy 的 nditer 函数，'f8' -> 'f4' 是安全转换，但 'f4' -> 'f8' 不是
    assert_raises(TypeError, nditer, arange(2, dtype='f8'), [],
                    [['readwrite', 'updateifcopy']],
                    casting='safe',
                    op_dtypes=[np.dtype('f4')])
    # 抛出 TypeError 异常，测试 numpy 的 nditer 函数，'f4' -> 'i4' 不是安全转换也不是同类别转换
    assert_raises(TypeError, nditer, arange(2, dtype='f4'), [],
                    [['readonly', 'copy']],
                    casting='same_kind',
                    op_dtypes=[np.dtype('i4')])
    # 抛出 TypeError 异常，测试 numpy 的 nditer 函数，'i4' -> 'f4' 不是安全转换也不是同类别转换
    assert_raises(TypeError, nditer, arange(2, dtype='i4'), [],
                    [['writeonly', 'updateifcopy']],
                    casting='same_kind',
                    op_dtypes=[np.dtype('f4')])
# 测试标量转换迭代器的功能

# 没有类型转换 'f4' -> 'f4'
i = nditer(np.float32(2.5), [], [['readonly']],
            op_dtypes=[np.dtype('f4')])
assert_equal(i.dtypes[0], np.dtype('f4'))  # 确保类型为 'f4'
assert_equal(i.value.dtype, np.dtype('f4'))  # 确保值的类型为 'f4'
assert_equal(i.value, 2.5)  # 确保值为 2.5

# 安全类型转换 'f4' -> 'f8'
i = nditer(np.float32(2.5), [],
            [['readonly', 'copy']],
            casting='safe',
            op_dtypes=[np.dtype('f8')])
assert_equal(i.dtypes[0], np.dtype('f8'))  # 确保类型为 'f8'
assert_equal(i.value.dtype, np.dtype('f8'))  # 确保值的类型为 'f8'
assert_equal(i.value, 2.5)  # 确保值为 2.5

# 同类型安全转换 'f8' -> 'f4'
i = nditer(np.float64(2.5), [],
            [['readonly', 'copy']],
            casting='same_kind',
            op_dtypes=[np.dtype('f4')])
assert_equal(i.dtypes[0], np.dtype('f4'))  # 确保类型为 'f4'
assert_equal(i.value.dtype, np.dtype('f4'))  # 确保值的类型为 'f4'
assert_equal(i.value, 2.5)  # 确保值为 2.5

# 不安全转换 'f8' -> 'i4'
i = nditer(np.float64(3.0), [],
            [['readonly', 'copy']],
            casting='unsafe',
            op_dtypes=[np.dtype('i4')])
assert_equal(i.dtypes[0], np.dtype('i4'))  # 确保类型为 'i4'
assert_equal(i.value.dtype, np.dtype('i4'))  # 确保值的类型为 'i4'
assert_equal(i.value, 3)  # 确保值为 3

# 只读标量即使未设置 COPY 或 BUFFERED 也可以进行类型转换
i = nditer(3, [], [['readonly']], op_dtypes=[np.dtype('f8')])
assert_equal(i[0].dtype, np.dtype('f8'))  # 确保类型为 'f8'
assert_equal(i[0], 3.)  # 确保值为 3.0


# 检查标量类型转换中的错误

# 需要允许复制/缓冲以便对标量进行写入转换
assert_raises(TypeError, nditer, np.float32(2), [],
            [['readwrite']], op_dtypes=[np.dtype('f8')])
assert_raises(TypeError, nditer, 2.5, [],
            [['readwrite']], op_dtypes=[np.dtype('f4')])

# 如果值会溢出，'f8' -> 'f4' 不是安全转换
assert_raises(TypeError, nditer, np.float64(1e60), [],
            [['readonly']],
            casting='safe',
            op_dtypes=[np.dtype('f4')])

# 'f4' -> 'i4' 既不是安全转换也不是同类型安全转换
assert_raises(TypeError, nditer, np.float32(2), [],
            [['readonly']],
            casting='same_kind',
            op_dtypes=[np.dtype('i4')])


# 检查对象数组的基本操作

# 检查对象数组是否工作
obj = {'a': 3, 'b': 'd'}
a = np.array([[1, 2, 3], None, obj, None], dtype='O')
if HAS_REFCOUNT:
    rc = sys.getrefcount(obj)

# 需要允许引用以处理对象数组
assert_raises(TypeError, nditer, a)
if HAS_REFCOUNT:
    assert_equal(sys.getrefcount(obj), rc)

# 使用 'refs_ok' 标志和 'readonly' 模式迭代对象数组
i = nditer(a, ['refs_ok'], ['readonly'])
vals = [x_[()] for x_ in i]  # 收集迭代器生成的值
assert_equal(np.array(vals, dtype='O'), a)  # 确保值与原始数组一致
vals, i, x = [None]*3  # 清理变量
if HAS_REFCOUNT:
    assert_equal(sys.getrefcount(obj), rc)  # 确保引用计数不变
    # 使用nditer迭代数组a的转置视图，设置refs_ok和buffered标志，只读访问，按行优先顺序
    i = nditer(a.reshape(2, 2).T, ['refs_ok', 'buffered'],
                        ['readonly'], order='C')
    # 断言迭代器需要API支持
    assert_(i.iterationneedsapi)
    # 从迭代器中提取所有的值到列表vals中
    vals = [x_[()] for x_ in i]
    # 断言vals转换为dtype='O'的NumPy数组与按列优先顺序展开的数组a相等
    assert_equal(np.array(vals, dtype='O'), a.reshape(2, 2).ravel(order='F'))
    # 将vals, i, x分别设置为None
    vals, i, x = [None]*3
    # 如果支持引用计数，则断言对象obj的引用计数等于rc
    if HAS_REFCOUNT:
        assert_equal(sys.getrefcount(obj), rc)

    # 再次使用nditer迭代数组a的转置视图，设置refs_ok和buffered标志，读写访问，按行优先顺序
    i = nditer(a.reshape(2, 2).T, ['refs_ok', 'buffered'],
                        ['readwrite'], order='C')
    # 进入上下文管理器i
    with i:
        # 遍历迭代器i中的每个元素x
        for x in i:
            # 将x的内容设置为None
            x[...] = None
        # 将vals, i, x分别设置为None
        vals, i, x = [None]*3
    # 如果支持引用计数，则断言对象obj的引用计数减1后与rc相等
    if HAS_REFCOUNT:
        assert_(sys.getrefcount(obj) == rc-1)
    # 断言数组a等于dtype='O'的NumPy数组，其所有元素都为None，按行优先顺序
    assert_equal(a, np.array([None]*4, dtype='O'))
# 定义一个测试函数，用于验证迭代器在对象数组转换中的行为
def test_iter_object_arrays_conversions():
    # Conversions to/from objects
    
    # 创建一个包含6个对象的NumPy数组，数据类型为对象
    a = np.arange(6, dtype='O')
    # 创建一个NumPy迭代器，指定refs_ok和buffered标志，并设置为读写模式
    i = nditer(a, ['refs_ok', 'buffered'], ['readwrite'],
                    casting='unsafe', op_dtypes='i4')
    # 进入迭代器的上下文环境
    with i:
        # 迭代处理每个元素，将其值加1
        for x in i:
            x[...] += 1
    # 验证数组a中的元素是否按预期被加1
    assert_equal(a, np.arange(6)+1)

    # 创建一个包含6个整数的NumPy数组，数据类型为32位整数
    a = np.arange(6, dtype='i4')
    # 创建一个NumPy迭代器，指定refs_ok和buffered标志，并设置为读写模式
    i = nditer(a, ['refs_ok', 'buffered'], ['readwrite'],
                    casting='unsafe', op_dtypes='O')
    # 进入迭代器的上下文环境
    with i:
        # 迭代处理每个元素，将其值加1
        for x in i:
            x[...] += 1
    # 验证数组a中的元素是否按预期被加1
    assert_equal(a, np.arange(6)+1)

    # 创建一个包含6个元素的非连续对象数组的NumPy数组，数据类型为对象
    a = np.zeros((6,), dtype=[('p', 'i1'), ('a', 'O')])
    a = a['a']  # 获取数组a中名为'a'的字段
    a[:] = np.arange(6)  # 将数组a的所有元素设置为0到5的整数
    # 创建一个NumPy迭代器，指定refs_ok和buffered标志，并设置为读写模式
    i = nditer(a, ['refs_ok', 'buffered'], ['readwrite'],
                    casting='unsafe', op_dtypes='i4')
    # 进入迭代器的上下文环境
    with i:
        # 迭代处理每个元素，将其值加1
        for x in i:
            x[...] += 1
    # 验证数组a中的元素是否按预期被加1
    assert_equal(a, np.arange(6)+1)

    # 创建一个包含6个元素的非连续值数组的NumPy数组，数据类型包含两个字段
    a = np.zeros((6,), dtype=[('p', 'i1'), ('a', 'i4')])
    a = a['a']  # 获取数组a中名为'a'的字段
    a[:] = np.arange(6) + 98172488  # 将数组a的所有元素设置为98172488到98172493的整数
    # 创建一个NumPy迭代器，指定refs_ok和buffered标志，并设置为读写模式
    i = nditer(a, ['refs_ok', 'buffered'], ['readwrite'],
                    casting='unsafe', op_dtypes='O')
    # 进入迭代器的上下文环境
    with i:
        ob = i[0][()]  # 获取迭代器的第一个元素
        if HAS_REFCOUNT:  # 如果系统支持引用计数
            rc = sys.getrefcount(ob)  # 获取对象ob的引用计数
        # 迭代处理每个元素，将其值加1
        for x in i:
            x[...] += 1
    # 如果系统支持引用计数，则验证对象ob的引用计数是否符合预期
    if HAS_REFCOUNT:
        assert_(sys.getrefcount(ob) == rc-1)
    # 验证数组a中的元素是否按预期被加1
    assert_equal(a, np.arange(6)+98172489)

# 定义一个测试函数，用于验证迭代器在查找公共数据类型时的行为
def test_iter_common_dtype():
    # Check that the iterator finds a common data type correctly
    # (some checks are somewhat duplicate after adopting NEP 50)

    # 创建一个NumPy迭代器，指定两个数组，并使用common_dtype标志
    i = nditer([array([3], dtype='f4'), array([0], dtype='f8')],
                    ['common_dtype'],
                    [['readonly', 'copy']]*2,
                    casting='safe')
    # 验证迭代器的第一个数据类型是否符合预期
    assert_equal(i.dtypes[0], np.dtype('f8'))
    # 验证迭代器的第二个数据类型是否符合预期
    assert_equal(i.dtypes[1], np.dtype('f8'))

    # 创建一个NumPy迭代器，指定两个数组，并使用common_dtype标志
    i = nditer([array([3], dtype='i4'), array([0], dtype='f4')],
                    ['common_dtype'],
                    [['readonly', 'copy']]*2,
                    casting='safe')
    # 验证迭代器的第一个数据类型是否符合预期
    assert_equal(i.dtypes[0], np.dtype('f8'))
    # 验证迭代器的第二个数据类型是否符合预期
    assert_equal(i.dtypes[1], np.dtype('f8'))

    # 创建一个NumPy迭代器，指定两个数组，并使用common_dtype标志
    i = nditer([array([3], dtype='f4'), array(0, dtype='f8')],
                    ['common_dtype'],
                    [['readonly', 'copy']]*2,
                    casting='same_kind')
    # 验证迭代器的第一个数据类型是否符合预期
    assert_equal(i.dtypes[0], np.dtype('f8'))
    # 验证迭代器的第二个数据类型是否符合预期
    assert_equal(i.dtypes[1], np.dtype('f8'))

    # 创建一个NumPy迭代器，指定两个数组，并使用common_dtype标志
    i = nditer([array([3], dtype='u4'), array(0, dtype='i4')],
                    ['common_dtype'],
                    [['readonly', 'copy']]*2,
                    casting='safe')
    # 验证迭代器的第一个数据类型是否符合预期
    assert_equal(i.dtypes[0], np.dtype('i8'))
    # 验证迭代器的第二个数据类型是否符合预期
    assert_equal(i.dtypes[1], np.dtype('i8'))

    # 创建一个NumPy迭代器，指定两个数组，并使用common_dtype标志
    i = nditer([array([3], dtype='u4'), array(-12, dtype='i4')],
                    ['common_dtype'],
                    [['readonly', 'copy']]*2,
                    casting='safe')
    # 验证迭代器的第一个数据类型是否符合预期
    assert_equal(i.dtypes[0], np.dtype('i8'))
    # 验证迭代器的第二个数据类型是否符合预期
    assert_equal(i.dtypes[1], np.dtype('i8'))
    # 创建一个迭代器对象 `i`，用于同时迭代四个不同的数组
    i = nditer([array([3], dtype='u4'), array(-12, dtype='i4'),
                 array([2j], dtype='c8'), array([9], dtype='f8')],
                    ['common_dtype'],  # 使用最常见的数据类型进行迭代
                    [['readonly', 'copy']]*4,  # 四个数组都是只读且需要复制
                    casting='safe')
    # 断言各个数组元素的数据类型为复数类型 (complex)
    assert_equal(i.dtypes[0], np.dtype('c16'))
    assert_equal(i.dtypes[1], np.dtype('c16'))
    assert_equal(i.dtypes[2], np.dtype('c16'))
    assert_equal(i.dtypes[3], np.dtype('c16'))
    # 断言迭代器的值符合预期的元组
    assert_equal(i.value, (3, -12, 2j, 9))

    # 当分配输出时，不考虑其他输出的影响
    i = nditer([array([3], dtype='i4'), None, array([2j], dtype='c16')], [],
                    [['readonly', 'copy'],
                     ['writeonly', 'allocate'],
                     ['writeonly']],
                    casting='safe')
    # 断言各个数组元素的数据类型为预期类型
    assert_equal(i.dtypes[0], np.dtype('i4'))
    assert_equal(i.dtypes[1], np.dtype('i4'))
    assert_equal(i.dtypes[2], np.dtype('c16'))

    # 但是，如果请求使用常见的数据类型，那么它们会被考虑进去
    i = nditer([array([3], dtype='i4'), None, array([2j], dtype='c16')],
                    ['common_dtype'],
                    [['readonly', 'copy'],
                     ['writeonly', 'allocate'],
                     ['writeonly']],
                    casting='safe')
    # 断言各个数组元素的数据类型为复数类型 (complex)
    assert_equal(i.dtypes[0], np.dtype('c16'))
    assert_equal(i.dtypes[1], np.dtype('c16'))
    assert_equal(i.dtypes[2], np.dtype('c16'))
# 确保迭代器在需要时在读写重叠时进行复制

# Copy not needed, 1 op
for flag in ['readonly', 'writeonly', 'readwrite']:
    # 创建长度为10的数组a
    a = arange(10)
    # 创建迭代器i，指定在读写重叠时复制（copy_if_overlap）或不复制
    i = nditer([a], ['copy_if_overlap'], [[flag]])
    # 进入迭代器上下文
    with i:
        # 断言a与迭代器操作数的第一个元素相同
        assert_(i.operands[0] is a)

# Copy needed, 2 ops, read-write overlap
x = arange(10)
# 创建x的切片a和b，有重叠的读写操作
a = x[1:]
b = x[:-1]
# 创建迭代器i，指定在读写重叠时复制（copy_if_overlap）或不复制
with nditer([a, b], ['copy_if_overlap'], [['readonly'], ['readwrite']]) as i:
    # 断言a与b不共享内存
    assert_(not np.shares_memory(*i.operands))

# Copy not needed with elementwise, 2 ops, exactly same arrays
x = arange(10)
# 创建数组x的两个引用a和b
a = x
b = x
# 创建迭代器i，指定在读写重叠时复制（copy_if_overlap）或不复制，假设元素级重叠
i = nditer([a, b], ['copy_if_overlap'], [['readonly', 'overlap_assume_elementwise'],
                                         ['readwrite', 'overlap_assume_elementwise']])
with i:
    # 断言a与b是迭代器操作数的第一个和第二个元素
    assert_(i.operands[0] is a and i.operands[1] is b)
# 创建迭代器i，指定在读写重叠时复制（copy_if_overlap）或不复制
with nditer([a, b], ['copy_if_overlap'], [['readonly'], ['readwrite']]) as i:
    # 断言a与迭代器操作数的第一个元素相同，且b与迭代器操作数的第二个元素不共享内存
    assert_(i.operands[0] is a and not np.shares_memory(i.operands[1], b))

# Copy not needed, 2 ops, no overlap
x = arange(10)
# 创建x的偶数索引切片a和奇数索引切片b
a = x[::2]
b = x[1::2]
# 创建迭代器i，指定在读写重叠时复制（copy_if_overlap）或不复制
i = nditer([a, b], ['copy_if_overlap'], [['readonly'], ['writeonly']])
# 断言a与迭代器操作数的第一个元素相同，且b与迭代器操作数的第二个元素相同
assert_(i.operands[0] is a and i.operands[1] is b)

# Copy needed, 2 ops, read-write overlap
x = arange(4, dtype=np.int8)
# 创建x的切片a和b，有重叠的读写操作
a = x[3:]
b = x.view(np.int32)[:1]
# 创建迭代器i，指定在读写重叠时复制（copy_if_overlap）或不复制
with nditer([a, b], ['copy_if_overlap'], [['readonly'], ['writeonly']]) as i:
    # 断言a与b不共享内存
    assert_(not np.shares_memory(*i.operands))

# Copy needed, 3 ops, read-write overlap
for flag in ['writeonly', 'readwrite']:
    # 创建全1的10x10数组x，创建x和其转置的引用a和b，和x的另一个引用c
    x = np.ones([10, 10])
    a = x
    b = x.T
    c = x
    # 创建迭代器i，指定在读写重叠时复制（copy_if_overlap）或不复制
    with nditer([a, b, c], ['copy_if_overlap'],
               [['readonly'], ['readonly'], [flag]]) as i:
        # 断言a与c，b与c不共享内存
        a2, b2, c2 = i.operands
        assert_(not np.shares_memory(a2, c2))
        assert_(not np.shares_memory(b2, c2))

# Copy not needed, 3 ops, read-only overlap
x = np.ones([10, 10])
# 创建x和其转置的引用a和b，和x的另一个引用c
a = x
b = x.T
c = x
# 创建迭代器i，指定在读写重叠时复制（copy_if_overlap）或不复制
i = nditer([a, b, c], ['copy_if_overlap'],
           [['readonly'], ['readonly'], ['readonly']])
a2, b2, c2 = i.operands
# 断言a、b、c与它们的相应操作数相同
assert_(a is a2)
assert_(b is b2)
assert_(c is c2)

# Copy not needed, 3 ops, read-only overlap
x = np.ones([10, 10])
# 创建全1的10x10数组x和另一个全1的10x10数组b，创建x和其转置的引用a和c
a = x
b = np.ones([10, 10])
c = x.T
# 创建迭代器i，指定在读写重叠时复制（copy_if_overlap）或不复制
i = nditer([a, b, c], ['copy_if_overlap'],
           [['readonly'], ['writeonly'], ['readonly']])
a2, b2, c2 = i.operands
# 断言a、b、c与它们的相应操作数相同
assert_(a is a2)
assert_(b is b2)
assert_(c is c2)

# Copy not needed, 3 ops, write-only overlap
x = np.arange(7)
# 创建x的三个不同切片a、b、c，有重叠的写操作
a = x[:3]
b = x[3:6]
c = x[4:7]
# 创建迭代器i，指定在读写重叠时复制（copy_if_overlap）或不复制
i = nditer([a, b, c], ['copy_if_overlap'],
           [['readonly'], ['writeonly'], ['writeonly']])
a2, b2, c2 = i.operands
# 断言a、b、c与它们的相应操作数相同
assert_(a is a2)
assert_(b is b2)
assert_(c is c2)
    # 创建一个 2x3 的数组，并且按照指定形状排列
    a = arange(6).reshape(2, 3)
    
    # 创建一个迭代器，同时迭代数组 a 和其转置，声明两者均为只读，指定操作轴
    i = nditer([a, a.T], [], [['readonly']]*2, op_axes=[[0, 1], [1, 0]])
    
    # 断言所有对应元素相等，即 a 和 a 的转置在所有对应位置上的元素值相等
    assert_(all([x == y for (x, y) in i]))
    
    # 创建一个 2x3x4 的数组，并且按照指定形状排列
    a = arange(24).reshape(2, 3, 4)
    
    # 创建一个迭代器，同时迭代数组 a 的转置和数组 a 本身，声明两者均为只读，指定操作轴
    i = nditer([a.T, a], [], [['readonly']]*2, op_axes=[[2, 1, 0], None])
    
    # 断言所有对应元素相等，即 a 的转置和 a 在所有对应位置上的元素值相等
    assert_(all([x == y for (x, y) in i]))
    
    # 将 1D 数组广播到任意维度
    a = arange(1, 31).reshape(2, 3, 5)
    b = arange(1, 3)
    
    # 创建一个迭代器，同时迭代数组 a 和数组 b，声明两者均为只读，指定操作轴
    i = nditer([a, b], [], [['readonly']]*2, op_axes=[None, [0, -1, -1]])
    
    # 断言迭代结果的每个元素乘积等于将 a 和 b 广播后展平的结果
    assert_equal([x*y for (x, y) in i], (a*b.reshape(2, 1, 1)).ravel())
    
    b = arange(1, 4)
    i = nditer([a, b], [], [['readonly']]*2, op_axes=[None, [-1, 0, -1]])
    assert_equal([x*y for (x, y) in i], (a*b.reshape(1, 3, 1)).ravel())
    
    b = arange(1, 6)
    i = nditer([a, b], [], [['readonly']]*2, op_axes=[None, [np.newaxis, np.newaxis, 0]])
    assert_equal([x*y for (x, y) in i], (a*b.reshape(1, 1, 5)).ravel())
    
    # 使用内积风格的广播
    a = arange(24).reshape(2, 3, 4)
    b = arange(40).reshape(5, 2, 4)
    
    # 创建一个迭代器，同时迭代数组 a 和数组 b，声明多重索引，两者均为只读，指定操作轴
    i = nditer([a, b], ['multi_index'], [['readonly']]*2, op_axes=[[0, 1, -1, -1], [-1, -1, 0, 1]])
    
    # 断言迭代器的形状与预期相等，即 (2, 3, 5, 2)
    assert_equal(i.shape, (2, 3, 5, 2))
    
    # 使用矩阵乘积风格的广播
    a = arange(12).reshape(3, 4)
    b = arange(20).reshape(4, 5)
    
    # 创建一个迭代器，同时迭代数组 a 和数组 b，声明多重索引，两者均为只读，指定操作轴
    i = nditer([a, b], ['multi_index'], [['readonly']]*2, op_axes=[[0, -1], [-1, 1]])
    
    # 断言迭代器的形状与预期相等，即 (3, 5)
    assert_equal(i.shape, (3, 5))
def test_iter_op_axes_errors():
    # 检查自定义轴是否对错误输入抛出异常

    # op_axes 中项目数目不正确
    a = arange(6).reshape(2, 3)
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']]*2,
                                    op_axes=[[0], [1], [0]])
    # op_axes 中存在超出界限的项目
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']]*2,
                                    op_axes=[[2, 1], [0, 1]])
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']]*2,
                                    op_axes=[[0, 1], [2, -1]])
    # op_axes 中存在重复的项目
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']]*2,
                                    op_axes=[[0, 0], [0, 1]])
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']]*2,
                                    op_axes=[[0, 1], [1, 1]])

    # op_axes 中数组大小不一致
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']]*2,
                                    op_axes=[[0, 1], [0, 1, 0]])

    # 结果中存在无法广播的维度
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']]*2,
                                    op_axes=[[0, 1], [1, 0]])

def test_iter_copy():
    # 检查迭代器复制功能是否正确

    a = arange(24).reshape(2, 3, 4)

    # 简单迭代器
    i = nditer(a)
    j = i.copy()
    assert_equal([x[()] for x in i], [x[()] for x in j])

    i.iterindex = 3
    j = i.copy()
    assert_equal([x[()] for x in i], [x[()] for x in j])

    # 缓冲迭代器
    i = nditer(a, ['buffered', 'ranged'], order='F', buffersize=3)
    j = i.copy()
    assert_equal([x[()] for x in i], [x[()] for x in j])

    i.iterindex = 3
    j = i.copy()
    assert_equal([x[()] for x in i], [x[()] for x in j])

    i.iterrange = (3, 9)
    j = i.copy()
    assert_equal([x[()] for x in i], [x[()] for x in j])

    i.iterrange = (2, 18)
    next(i)
    next(i)
    j = i.copy()
    assert_equal([x[()] for x in i], [x[()] for x in j])

    # 强制类型转换迭代器
    with nditer(a, ['buffered'], order='F', casting='unsafe',
                op_dtypes='f8', buffersize=5) as i:
        j = i.copy()
    assert_equal([x[()] for x in j], a.ravel(order='F'))

    a = arange(24, dtype='<i4').reshape(2, 3, 4)
    with nditer(a, ['buffered'], order='F', casting='unsafe',
                op_dtypes='>f8', buffersize=5) as i:
        j = i.copy()
    assert_equal([x[()] for x in j], a.ravel(order='F'))


@pytest.mark.parametrize("dtype", np.typecodes["All"])
@pytest.mark.parametrize("loop_dtype", np.typecodes["All"])
@pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning")
def test_iter_copy_casts(dtype, loop_dtype):
    # 确保 dtype 永远不是灵活的：

    if loop_dtype.lower() == "m":
        loop_dtype = loop_dtype + "[ms]"
    elif np.dtype(loop_dtype).itemsize == 0:
        loop_dtype = loop_dtype + "50"
    # 创建一个由全1数组组成的NumPy数组，数据类型为指定的dtype，并且需要进行字节交换
    arr = np.ones(1000, dtype=np.dtype(dtype).newbyteorder())
    try:
        # 尝试将arr数组转换为指定的loop_dtype数据类型
        expected = arr.astype(loop_dtype)
    except Exception:
        # 捕获异常：某些类型转换不可行，不需要处理这些异常情况
        return

    # 创建一个NumPy迭代器对象it，迭代arr数组，设置迭代选项为"buffered", "external_loop", "refs_ok"，
    # 操作数据类型为loop_dtype，禁止类型安全检查
    it = np.nditer((arr,), ["buffered", "external_loop", "refs_ok"],
                   op_dtypes=[loop_dtype], casting="unsafe")

    # 如果loop_dtype的数据类型是数值类型
    if np.issubdtype(np.dtype(loop_dtype), np.number):
        # 对于简单的数据类型，转换为字符串可能会出现奇怪的结果，但不依赖于转换是否正确
        assert_array_equal(expected, np.ones(1000, dtype=loop_dtype))

    # 创建it对象的副本it_copy，并获取迭代器的下一个元素res
    it_copy = it.copy()
    res = next(it)
    # 删除原始迭代器对象it
    del it
    # 获取it_copy副本的下一个元素res_copy，并删除it_copy副本
    res_copy = next(it_copy)
    del it_copy

    # 断言迭代器的结果res与预期的expected相等
    assert_array_equal(res, expected)
    # 断言副本迭代器的结果res_copy与预期的expected相等
    assert_array_equal(res_copy, expected)
# 检查迭代器是否能正确分配输出

# 简单情况
a = arange(6)
# 创建一个迭代器，迭代器有两个操作数，第一个是只读的，第二个是只写且分配空间的，类型为 np.dtype('f4')
i = nditer([a, None], [], [['readonly'], ['writeonly', 'allocate']],
            op_dtypes=[None, np.dtype('f4')])
# 确认迭代器第二个操作数的形状与 a 相同
assert_equal(i.operands[1].shape, a.shape)
# 确认迭代器第二个操作数的数据类型为 np.dtype('f4')
assert_equal(i.operands[1].dtype, np.dtype('f4'))
    # 分配输出数组并使用缓冲和延迟缓冲分配

    # 创建一个包含6个元素的数组a
    a = arange(6)
    # 使用nditer迭代器迭代数组a和一个None数组，使用缓冲和延迟缓冲分配
    i = nditer([a, None], ['buffered', 'delay_bufalloc'],
                        [['readonly'], ['allocate', 'readwrite']])
    # 进入迭代器的上下文管理器
    with i:
        # 将第二个操作数数组的所有元素设置为1
        i.operands[1][:] = 1
        # 重置迭代器到初始状态
        i.reset()
        # 遍历迭代器
        for x in i:
            # 将第二个操作数数组的每个元素增加对应的第一个操作数数组的元素值
            x[1][...] += x[0][...]
        # 断言第二个操作数数组的值与数组a每个元素加1后的结果相等
        assert_equal(i.operands[1], a+1)
def test_iter_allocate_output_itorder():
    # The allocated output should match the iteration order

    # C-order input, best iteration order
    # 创建一个形状为 (2, 3) 的整数数组 a，按 C 顺序进行排列
    a = arange(6, dtype='i4').reshape(2, 3)
    # 使用 nditer 创建迭代器 i，操作数包括 a 和一个写入-分配的数组，结果类型为 float32
    i = nditer([a, None], [], [['readonly'], ['writeonly', 'allocate']],
                        op_dtypes=[None, np.dtype('f4')])
    # 断言分配的输出数组的形状与 a 相同
    assert_equal(i.operands[1].shape, a.shape)
    # 断言分配的输出数组的步幅与 a 相同
    assert_equal(i.operands[1].strides, a.strides)
    # 断言分配的输出数组的数据类型为 float32
    assert_equal(i.operands[1].dtype, np.dtype('f4'))

    # F-order input, best iteration order
    # 创建一个形状为 (2, 3, 4) 的整数数组 a，按 F 顺序进行排列
    a = arange(24, dtype='i4').reshape(2, 3, 4).T
    # 使用 nditer 创建迭代器 i，操作数包括 a 和一个写入-分配的数组，结果类型为 float32
    i = nditer([a, None], [], [['readonly'], ['writeonly', 'allocate']],
                        op_dtypes=[None, np.dtype('f4')])
    # 断言分配的输出数组的形状与 a 相同
    assert_equal(i.operands[1].shape, a.shape)
    # 断言分配的输出数组的步幅与 a 相同
    assert_equal(i.operands[1].strides, a.strides)
    # 断言分配的输出数组的数据类型为 float32
    assert_equal(i.operands[1].dtype, np.dtype('f4'))

    # Non-contiguous input, C iteration order
    # 创建一个形状为 (2, 3, 4) 的整数数组 a，并在轴 0 和 1 上交换，按 C 顺序进行排列
    a = arange(24, dtype='i4').reshape(2, 3, 4).swapaxes(0, 1)
    # 使用 nditer 创建迭代器 i，操作数包括 a 和一个写入-分配的数组，按 C 顺序进行排列，结果类型为 float32
    i = nditer([a, None], [],
                        [['readonly'], ['writeonly', 'allocate']],
                        order='C',
                        op_dtypes=[None, np.dtype('f4')])
    # 断言分配的输出数组的形状与 a 相同
    assert_equal(i.operands[1].shape, a.shape)
    # 断言分配的输出数组的步幅与预期的 (32, 16, 4) 相同
    assert_equal(i.operands[1].strides, (32, 16, 4))
    # 断言分配的输出数组的数据类型为 float32
    assert_equal(i.operands[1].dtype, np.dtype('f4'))

def test_iter_allocate_output_opaxes():
    # Specifying op_axes should work

    # 创建一个形状为 (2, 3, 4) 的整数数组 a
    a = arange(24, dtype='i4').reshape(2, 3, 4)
    # 使用 nditer 创建迭代器 i，操作数包括一个写入-分配的数组和 a，指定操作轴，结果类型为 uint32
    i = nditer([None, a], [], [['writeonly', 'allocate'], ['readonly']],
                        op_dtypes=[np.dtype('u4'), None],
                        op_axes=[[1, 2, 0], None])
    # 断言分配的输出数组的形状为 (4, 2, 3)
    assert_equal(i.operands[0].shape, (4, 2, 3))
    # 断言分配的输出数组的步幅为 (4, 48, 16)
    assert_equal(i.operands[0].strides, (4, 48, 16))
    # 断言分配的输出数组的数据类型为 uint32
    assert_equal(i.operands[0].dtype, np.dtype('u4'))

def test_iter_allocate_output_types_promotion():
    # Check type promotion of automatic outputs (this was more interesting
    # before NEP 50...)

    # 使用 nditer 创建迭代器 i，操作数包括两个浮点数组和一个写入-分配的数组
    i = nditer([array([3], dtype='f4'), array([0], dtype='f8'), None], [],
                    [['readonly']]*2+[['writeonly', 'allocate']])
    # 断言分配的输出数组的数据类型为 float64
    assert_equal(i.dtypes[2], np.dtype('f8'))

    # 使用 nditer 创建迭代器 i，操作数包括一个整数数组、一个浮点数组和一个写入-分配的数组
    i = nditer([array([3], dtype='i4'), array([0], dtype='f4'), None], [],
                    [['readonly']]*2+[['writeonly', 'allocate']])
    # 断言分配的输出数组的数据类型为 float64
    assert_equal(i.dtypes[2], np.dtype('f8'))

    # 使用 nditer 创建迭代器 i，操作数包括一个浮点数组、一个浮点数和一个写入-分配的数组
    i = nditer([array([3], dtype='f4'), array(0, dtype='f8'), None], [],
                    [['readonly']]*2+[['writeonly', 'allocate']])
    # 断言分配的输出数组的数据类型为 float64
    assert_equal(i.dtypes[2], np.dtype('f8'))

    # 使用 nditer 创建迭代器 i，操作数包括一个无符号整数数组、一个整数和一个写入-分配的数组
    i = nditer([array([3], dtype='u4'), array(0, dtype='i4'), None], [],
                    [['readonly']]*2+[['writeonly', 'allocate']])
    # 断言分配的输出数组的数据类型为 int64
    assert_equal(i.dtypes[2], np.dtype('i8'))

    # 使用 nditer 创建迭代器 i，操作数包括一个无符号整数数组、一个整数和一个写入-分配的数组
    i = nditer([array([3], dtype='u4'), array(-12, dtype='i4'), None], [],
                    [['readonly']]*2+[['writeonly', 'allocate']])
    # 断言分配的输出数组的数据类型为 int64
    assert_equal(i.dtypes[2], np.dtype('i8'))

def test_iter_allocate_output_types_byte_order():
    # Verify the rules for byte order changes
    # 创建一个长度为1的无符号32位整数数组
    a = array([3], dtype='u4')
    # 将数组视图转换为与当前系统字节顺序相同的格式
    a = a.view(a.dtype.newbyteorder())
    # 创建一个迭代器对象，迭代数组a和一个空数组，不需要操作数组的索引
    i = nditer([a, None], [],
                    [['readonly'], ['writeonly', 'allocate']])
    # 断言迭代器中第一个输入和第二个输出的数据类型相同
    assert_equal(i.dtypes[0], i.dtypes[1])
    # 当输入大于等于两个时，输出的数据类型是本机字节顺序
    i = nditer([a, a, None], [],
                    [['readonly'], ['readonly'], ['writeonly', 'allocate']])
    # 断言迭代器中第一个输入和第三个输出的数据类型不同
    assert_(i.dtypes[0] != i.dtypes[2])
    # 断言迭代器中第一个输入的本机字节顺序与第三个输出的数据类型的本机字节顺序相同
    assert_equal(i.dtypes[0].newbyteorder('='), i.dtypes[2])
# 如果所有输入都是标量，则输出应该是一个标量
def test_iter_allocate_output_types_scalar():
    i = nditer([None, 1, 2.3, np.float32(12), np.complex128(3)], [],
                [['writeonly', 'allocate']] + [['readonly']]*4)
    # 断言输出操作数的数据类型为 complex128
    assert_equal(i.operands[0].dtype, np.dtype('complex128'))
    # 断言输出操作数的维度为 0
    assert_equal(i.operands[0].ndim, 0)

# 确保具有优先级的子类型能够胜出
def test_iter_allocate_output_subtype():
    class MyNDArray(np.ndarray):
        __array_priority__ = 15

    # ndarray 的子类 vs ndarray
    a = np.array([[1, 2], [3, 4]]).view(MyNDArray)
    b = np.arange(4).reshape(2, 2).T
    i = nditer([a, b, None], [],
               [['readonly'], ['readonly'], ['writeonly', 'allocate']])
    # 断言第三个操作数的类型与 a 相同
    assert_equal(type(a), type(i.operands[2]))
    # 断言第三个操作数的类型不同于 b
    assert_(type(b) is not type(i.operands[2]))
    # 断言第三个操作数的形状为 (2, 2)
    assert_equal(i.operands[2].shape, (2, 2))

    # 如果禁用子类型，应该返回一个 ndarray
    i = nditer([a, b, None], [],
               [['readonly'], ['readonly'],
                ['writeonly', 'allocate', 'no_subtype']])
    # 断言第三个操作数的类型与 b 相同
    assert_equal(type(b), type(i.operands[2]))
    # 断言第三个操作数的类型不同于 a
    assert_(type(a) is not type(i.operands[2]))
    # 断言第三个操作数的形状为 (2, 2)
    assert_equal(i.operands[2].shape, (2, 2))

# 检查迭代器在错误的输出分配时是否会抛出错误
def test_iter_allocate_output_errors():
    # 如果没有指定输出数据类型，则需要一个输入
    a = arange(6)
    assert_raises(TypeError, nditer, [a, None], [],
                        [['writeonly'], ['writeonly', 'allocate']])
    # 分配的输出应该标记为可写
    assert_raises(ValueError, nditer, [a, None], [],
                        [['readonly'], ['allocate', 'readonly']])
    # 分配的输出不能有缓冲区，除非延迟分配缓冲区
    assert_raises(ValueError, nditer, [a, None], ['buffered'],
                                            ['allocate', 'readwrite'])
    # 如果没有输入，则必须指定 dtype（不能提升现有的 dtype；可能应该在这里使用 'f4'，但历史上并没有这样做）
    assert_raises(TypeError, nditer, [None, None], [],
                        [['writeonly', 'allocate'],
                         ['writeonly', 'allocate']],
                        op_dtypes=[None, np.dtype('f4')])
    # 如果使用 op_axes，则必须指定所有轴
    a = arange(24, dtype='i4').reshape(2, 3, 4)
    assert_raises(ValueError, nditer, [a, None], [],
                        [['readonly'], ['writeonly', 'allocate']],
                        op_dtypes=[None, np.dtype('f4')],
                        op_axes=[None, [0, np.newaxis, 1]])
    # 如果使用 op_axes，则轴必须在范围内
    assert_raises(ValueError, nditer, [a, None], [],
                        [['readonly'], ['writeonly', 'allocate']],
                        op_dtypes=[None, np.dtype('f4')],
                        op_axes=[None, [0, 3, 1]])
    # 如果使用 op_axes，则不能有重复的轴
    # 使用 assert_raises 函数验证调用 nditer 函数时是否会引发 ValueError 异常
    assert_raises(ValueError, nditer, [a, None], [],
                        [['readonly'], ['writeonly', 'allocate']],
                        op_dtypes=[None, np.dtype('f4')],
                        op_axes=[None, [0, 2, 1, 0]])
    # 如果 op_axes 中存在空值或缺失，则会导致错误，因此需要确保每个轴都有明确的定义

    # 创建一个形状为 (2, 3, 4) 的整数数组 a，范围从 0 到 23
    a = arange(24, dtype='i4').reshape(2, 3, 4)
    
    # 使用 assert_raises 函数验证调用 nditer 函数时是否会引发 ValueError 异常
    assert_raises(ValueError, nditer, [a, None], ["reduce_ok"],
                        [['readonly'], ['readwrite', 'allocate']],
                        op_dtypes=[None, np.dtype('f4')],
                        op_axes=[None, [0, np.newaxis, 2]])
    # 如果 op_axes 中存在空值或缺失，则会导致错误，因此需要确保每个轴都有明确的定义
def test_all_allocated():
    # 当没有输出和形状给定时，默认使用 `()` 作为形状。
    # 创建一个 `np.nditer` 迭代器对象 `i`，用于遍历数组，指定数据类型为 "int64"
    i = np.nditer([None], op_dtypes=["int64"])
    # 断言迭代器操作数的形状为 `()`
    assert i.operands[0].shape == ()
    # 断言迭代器的数据类型为 `np.dtype("int64")`
    assert i.dtypes == (np.dtype("int64"),)

    # 创建另一个 `np.nditer` 迭代器对象 `i`，指定形状为 (2, 3, 4)，数据类型为 "int64"
    i = np.nditer([None], op_dtypes=["int64"], itershape=(2, 3, 4))
    # 断言迭代器操作数的形状为 (2, 3, 4)
    assert i.operands[0].shape == (2, 3, 4)

def test_iter_remove_axis():
    # 创建一个形状为 (2, 3, 4) 的数组 `a`
    a = arange(24).reshape(2, 3, 4)

    # 创建一个 `nditer` 迭代器对象 `i`，只遍历 'multi_index'
    i = nditer(a, ['multi_index'])
    # 移除第二个轴（axis=1）
    i.remove_axis(1)
    # 断言迭代器输出的扁平化结果与 `a[:, 0,:]` 的扁平化结果相等
    assert_equal([x for x in i], a[:, 0,:].ravel())

    # 将数组 `a` 在第一个轴上逆序
    a = a[::-1,:,:]
    # 创建一个新的 `nditer` 迭代器对象 `i`，只遍历 'multi_index'
    i = nditer(a, ['multi_index'])
    # 移除第一个轴（axis=0）
    i.remove_axis(0)
    # 断言迭代器输出的扁平化结果与 `a[0,:,:]` 的扁平化结果相等
    assert_equal([x for x in i], a[0,:,:].ravel())

def test_iter_remove_multi_index_inner_loop():
    # 检查移除多重索引支持是否有效

    # 创建一个形状为 (2, 3, 4) 的数组 `a`
    a = arange(24).reshape(2, 3, 4)

    # 创建一个 `nditer` 迭代器对象 `i`，遍历 'multi_index'
    i = nditer(a, ['multi_index'])
    # 断言迭代器的维度为 3
    assert_equal(i.ndim, 3)
    # 断言迭代器的形状为 (2, 3, 4)
    assert_equal(i.shape, (2, 3, 4))
    # 断言迭代器的视图形状为 (2, 3, 4)
    assert_equal(i.itviews[0].shape, (2, 3, 4))

    # 移除多重索引追踪
    before = [x for x in i]
    i.remove_multi_index()
    after = [x for x in i]

    # 断言移除多重索引后的输出与移除前相同
    assert_equal(before, after)
    # 断言迭代器的维度变为 1
    assert_equal(i.ndim, 1)
    # 断言尝试获取迭代器形状时引发 ValueError 异常
    assert_raises(ValueError, lambda i:i.shape, i)
    # 断言迭代器的视图形状为 (24,)
    assert_equal(i.itviews[0].shape, (24,))

    # 重置迭代器
    i.reset()
    # 断言迭代器的迭代大小为 24
    assert_equal(i.itersize, 24)
    # 断言迭代器索引为 0 的元素形状为 `tuple()`
    assert_equal(i[0].shape, tuple())
    # 启用外部循环
    i.enable_external_loop()
    # 断言迭代器的迭代大小为 24
    assert_equal(i.itersize, 24)
    # 断言迭代器索引为 0 的元素形状为 (24,)
    assert_equal(i[0].shape, (24,))
    # 断言迭代器的值与 `arange(24)` 相等
    assert_equal(i.value, arange(24))

def test_iter_iterindex():
    # 确保 iterindex 正常工作

    buffersize = 5
    # 创建一个形状为 (4, 3, 2) 的数组 `a`
    a = arange(24).reshape(4, 3, 2)
    for flags in ([], ['buffered']):
        # 创建一个 `nditer` 迭代器对象 `i`，指定缓冲区大小为 5
        i = nditer(a, flags, buffersize=buffersize)
        # 断言迭代器的 iterindex 数组与 `range(24)` 相等
        assert_equal(iter_iterindices(i), list(range(24)))
        # 设置 iterindex 为 2
        i.iterindex = 2
        # 断言迭代器的 iterindex 数组从 2 开始，与 `range(2, 24)` 相等
        assert_equal(iter_iterindices(i), list(range(2, 24)))

        # 创建一个按列优先（Fortran顺序）遍历的 `nditer` 迭代器对象 `i`，指定缓冲区大小为 5
        i = nditer(a, flags, order='F', buffersize=buffersize)
        # 断言迭代器的 iterindex 数组与 `range(24)` 相等
        assert_equal(iter_iterindices(i), list(range(24)))
        # 设置 iterindex 为 5
        i.iterindex = 5
        # 断言迭代器的 iterindex 数组从 5 开始，与 `range(5, 24)` 相等
        assert_equal(iter_iterindices(i), list(range(5, 24)))

        # 创建一个逆序遍历的 `nditer` 迭代器对象 `i`，按列优先（Fortran顺序），指定缓冲区大小为 5
        i = nditer(a[::-1], flags, order='F', buffersize=buffersize)
        # 断言迭代器的 iterindex 数组与 `range(24)` 相等
        assert_equal(iter_iterindices(i), list(range(24)))
        # 设置 iterindex 为 9
        i.iterindex = 9
        # 断言迭代器的 iterindex 数组从 9 开始，与 `range(9, 24)` 相等
        assert_equal(iter_iterindices(i), list(range(9, 24)))

        # 创建一个逆序遍历的 `nditer` 迭代器对象 `i`，按行优先（C顺序），指定缓冲区大小为 5
        i = nditer(a[::-1, ::-1], flags, order='C', buffersize=buffersize)
        # 断言迭代器的 iterindex 数组与 `range(24)` 相等
        assert_equal(iter_iterindices(i), list(range(24)))
        # 设置 iterindex 为 13
        i.iterindex = 13
        # 断言迭代器的 iterindex 数组从 13 开始，与 `range(13, 24)` 相等
        assert_equal(iter_iterindices(i), list(range(13, 24)))

        # 创建一个逆序遍历的 `nditer` 迭代器对象 `i`，按列优先（Fortran顺序），指定缓冲区大小为 5
        i = nditer(a[::1, ::-1], flags, buffersize=buffersize)
        # 断言迭代器的 iterindex 数组与 `range(24)` 相等
        assert_equal(iter_iterindices(i), list(range(24)))
        # 设置 iterindex 为 23
        i.iterindex = 23
        # 断言迭代器的 iterindex 数组从 23 开始，与 `range(23, 24)` 相等
        assert_equal(iter_iterindices(i), list(range(23, 24)))
        # 重置迭代器
        i.reset()
    # 将数组 a 按列（Fortran）顺序展平为一维数组
    a_fort = a.ravel(order='F')

    # 创建一个以 Fortran（列优先）顺序迭代数组 a 的迭代器
    i = nditer(a, ['ranged'], ['readonly'], order='F',
                buffersize=buffersize)
    # 断言迭代器的范围是否为 (0, 24)
    assert_equal(i.iterrange, (0, 24))
    # 断言迭代器中的每个元素与 a_fort 中对应位置的元素相等
    assert_equal([x[()] for x in i], a_fort)
    # 对给定的多个范围进行迭代，设置迭代器的范围并进行断言
    for r in [(0, 24), (1, 2), (3, 24), (5, 5), (0, 20), (23, 24)]:
        i.iterrange = r
        assert_equal(i.iterrange, r)
        # 断言迭代器中的每个元素与 a_fort 中对应范围的元素相等
        assert_equal([x[()] for x in i], a_fort[r[0]:r[1]])

    # 创建一个以 Fortran（列优先）顺序迭代数组 a 的迭代器，使用双缓冲
    i = nditer(a, ['ranged', 'buffered'], ['readonly'], order='F',
                op_dtypes='f8', buffersize=buffersize)
    # 断言迭代器的范围是否为 (0, 24)
    assert_equal(i.iterrange, (0, 24))
    # 断言迭代器中的每个元素与 a_fort 中对应位置的元素相等
    assert_equal([x[()] for x in i], a_fort)
    # 对给定的多个范围进行迭代，设置迭代器的范围并进行断言
    for r in [(0, 24), (1, 2), (3, 24), (5, 5), (0, 20), (23, 24)]:
        i.iterrange = r
        assert_equal(i.iterrange, r)
        # 断言迭代器中的每个元素与 a_fort 中对应范围的元素相等
        assert_equal([x[()] for x in i], a_fort[r[0]:r[1]])

    # 定义一个函数 get_array，用于将迭代器 i 中的元素拼接成一个 NumPy 数组
    def get_array(i):
        val = np.array([], dtype='f8')
        for x in i:
            val = np.concatenate((val, x))
        return val

    # 创建一个以 Fortran（列优先）顺序迭代数组 a 的迭代器，使用双缓冲和外部循环
    i = nditer(a, ['ranged', 'buffered', 'external_loop'],
                ['readonly'], order='F',
                op_dtypes='f8', buffersize=buffersize)
    # 断言迭代器的范围是否为 (0, 24)
    assert_equal(i.iterrange, (0, 24))
    # 断言迭代器中的所有元素合并为一个数组后与 a_fort 相等
    assert_equal(get_array(i), a_fort)
    # 对给定的多个范围进行迭代，设置迭代器的范围并进行断言
    for r in [(0, 24), (1, 2), (3, 24), (5, 5), (0, 20), (23, 24)]:
        i.iterrange = r
        assert_equal(i.iterrange, r)
        # 断言迭代器中的所有元素合并为一个数组后与 a_fort 对应范围的元素相等
        assert_equal(get_array(i), a_fort[r[0]:r[1]])
# 测试使用不同的缓冲大小和类型进行缓冲操作
def test_iter_buffering():
    arrays = []
    # 创建一个 F-order 的交换数组
    _tmp = np.arange(24, dtype='c16').reshape(2, 3, 4).T
    _tmp = _tmp.view(_tmp.dtype.newbyteorder()).byteswap()
    arrays.append(_tmp)
    # 创建一个连续的一维数组
    arrays.append(np.arange(10, dtype='f4'))
    # 创建一个非对齐的数组
    a = np.zeros((4*16+1,), dtype='i1')[1:]
    a.dtype = 'i4'
    a[:] = np.arange(16, dtype='i4')
    arrays.append(a)
    # 创建一个 4-D 的 F-order 数组
    arrays.append(np.arange(120, dtype='i4').reshape(5, 3, 2, 4).T)
    # 对于每个数组进行迭代
    for a in arrays:
        # 对于每个缓冲大小进行迭代
        for buffersize in (1, 2, 3, 5, 8, 11, 16, 1024):
            vals = []
            # 创建一个 nditer 迭代器
            i = nditer(a, ['buffered', 'external_loop'],
                           [['readonly', 'nbo', 'aligned']],
                           order='C',
                           casting='equiv',
                           buffersize=buffersize)
            # 迭代直到完成
            while not i.finished:
                # 断言当前元素大小不超过缓冲大小
                assert_(i[0].size <= buffersize)
                vals.append(i[0].copy())
                i.iternext()
            # 断言迭代后的结果与数组按顺序展平后的结果相等
            assert_equal(np.concatenate(vals), a.ravel(order='C'))

# 测试写入缓冲是否有效
def test_iter_write_buffering():
    # 创建一个 F-order 交换数组
    a = np.arange(24).reshape(2, 3, 4).T
    a = a.view(a.dtype.newbyteorder()).byteswap()
    # 创建一个 nditer 迭代器，用于写入缓冲
    i = nditer(a, ['buffered'],
                   [['readwrite', 'nbo', 'aligned']],
                   casting='equiv',
                   order='C',
                   buffersize=16)
    x = 0
    # 使用 with 语句确保正确处理迭代器
    with i:
        while not i.finished:
            i[0] = x
            x += 1
            i.iternext()
    # 断言按顺序展平后的数组与预期的顺序数组相等
    assert_equal(a.ravel(order='C'), np.arange(24))

# 测试延迟分配缓冲是否有效
def test_iter_buffering_delayed_alloc():
    # 创建一个数组 a
    a = np.arange(6)
    # 创建一个数组 b，包含单个元素
    b = np.arange(1, dtype='f4')
    # 创建一个 nditer 迭代器，测试缓冲、延迟缓冲分配、多索引和允许归约操作
    i = nditer([a, b], ['buffered', 'delay_bufalloc', 'multi_index', 'reduce_ok'],
                    ['readwrite'],
                    casting='unsafe',
                    op_dtypes='f4')
    # 断言是否有延迟缓冲分配
    assert_(i.has_delayed_bufalloc)
    # 断言使用 lambda 函数是否会抛出 ValueError 异常
    assert_raises(ValueError, lambda i:i.multi_index, i)
    assert_raises(ValueError, lambda i:i[0], i)
    assert_raises(ValueError, lambda i:i[0:2], i)

    def assign_iter(i):
        i[0] = 0
    # 断言使用自定义函数 assign_iter 是否会抛出 ValueError 异常
    assert_raises(ValueError, assign_iter, i)

    # 重置迭代器
    i.reset()
    # 断言现在没有延迟缓冲分配
    assert_(not i.has_delayed_bufalloc)
    # 断言多索引的值为 (0,)
    assert_equal(i.multi_index, (0,))
    with i:
        # 断言当前元素的值为 0
        assert_equal(i[0], 0)
        i[1] = 1
        # 断言前两个元素的值为 [0, 1]
        assert_equal(i[0:2], [0, 1])
        # 断言迭代器的结果与预期的结果相等
        assert_equal([[x[0][()], x[1][()]] for x in i], list(zip(range(6), [1]*6)))

# 测试缓冲是否能处理简单的类型转换
def test_iter_buffered_cast_simple():
    # 创建一个浮点型的数组 a
    a = np.arange(10, dtype='f4')
    # 创建一个 nditer 迭代器，测试缓冲、外部循环、类型转换、相同类型转换、缓冲大小
    i = nditer(a, ['buffered', 'external_loop'],
                   [['readwrite', 'nbo', 'aligned']],
                   casting='same_kind',
                   op_dtypes=[np.dtype('f8')],
                   buffersize=3)
    # 使用上下文管理器 i 开始一个代码块，确保在其范围内操作 i
    with i:
        # 遍历 i 中的每一个元素 v
        for v in i:
            # 将 v 中的所有元素乘以 2
            v[...] *= 2
    
    # 断言检查：确保数组 a 等于一个由 np.arange 生成的浮点数数组，每个元素乘以 2
    assert_equal(a, 2*np.arange(10, dtype='f4'))
# 测试缓冲区能否处理需要交换->转换->交换的类型转换
def test_iter_buffered_cast_byteswapped():
    # 创建一个长度为10的单精度浮点数数组
    a = np.arange(10, dtype='f4')
    # 将数组视图转换为新的字节顺序，并进行字节交换
    a = a.view(a.dtype.newbyteorder()).byteswap()
    # 创建一个迭代器对象，指定使用缓冲区模式和外部循环模式
    i = nditer(a, ['buffered', 'external_loop'],
                   [['readwrite', 'nbo', 'aligned']],
                   casting='same_kind',
                   op_dtypes=[np.dtype('f8').newbyteorder()],
                   buffersize=3)
    with i:
        for v in i:
            # 对迭代器中的每个元素进行乘以2的操作
            v[...] *= 2

    # 断言数组a的内容与期望值相等
    assert_equal(a, 2*np.arange(10, dtype='f4'))

    # 屏蔽特定警告类型的上下文管理器
    with suppress_warnings() as sup:
        sup.filter(np.exceptions.ComplexWarning)

        # 创建一个长度为10的双精度浮点数数组
        a = np.arange(10, dtype='f8')
        # 将数组视图转换为新的字节顺序，并进行字节交换
        a = a.view(a.dtype.newbyteorder()).byteswap()
        # 创建一个迭代器对象，指定使用缓冲区模式和外部循环模式
        i = nditer(a, ['buffered', 'external_loop'],
                       [['readwrite', 'nbo', 'aligned']],
                       casting='unsafe',
                       op_dtypes=[np.dtype('c8').newbyteorder()],
                       buffersize=3)
        with i:
            for v in i:
                # 对迭代器中的每个元素进行乘以2的操作
                v[...] *= 2

        # 断言数组a的内容与期望值相等
        assert_equal(a, 2*np.arange(10, dtype='f8'))

def test_iter_buffered_cast_byteswapped_complex():
    # 测试缓冲区能否处理需要交换->转换->复制的复杂类型转换

    # 创建一个长度为10的复数双精度浮点数数组
    a = np.arange(10, dtype='c8')
    # 将数组视图转换为新的字节顺序，并进行字节交换
    a = a.view(a.dtype.newbyteorder()).byteswap()
    # 将数组中每个元素加上虚数部分为2的复数
    a += 2j
    # 创建一个迭代器对象，指定使用缓冲区模式和外部循环模式
    i = nditer(a, ['buffered', 'external_loop'],
                   [['readwrite', 'nbo', 'aligned']],
                   casting='same_kind',
                   op_dtypes=[np.dtype('c16')],
                   buffersize=3)
    with i:
        for v in i:
            # 对迭代器中的每个元素进行乘以2的操作
            v[...] *= 2
    # 断言数组a的内容与期望值相等
    assert_equal(a, 2*np.arange(10, dtype='c8') + 4j)

    # 创建一个长度为10的复数双精度浮点数数组
    a = np.arange(10, dtype='c8')
    # 将数组中每个元素加上虚数部分为2的复数
    a += 2j
    # 创建一个迭代器对象，指定使用缓冲区模式和外部循环模式
    i = nditer(a, ['buffered', 'external_loop'],
                   [['readwrite', 'nbo', 'aligned']],
                   casting='same_kind',
                   op_dtypes=[np.dtype('c16').newbyteorder()],
                   buffersize=3)
    with i:
        for v in i:
            # 对迭代器中的每个元素进行乘以2的操作
            v[...] *= 2
    # 断言数组a的内容与期望值相等
    assert_equal(a, 2*np.arange(10, dtype='c8') + 4j)

    # 创建一个长度为10的复杂长双精度浮点数数组
    a = np.arange(10, dtype=np.clongdouble)
    # 将数组视图转换为新的字节顺序，并进行字节交换
    a = a.view(a.dtype.newbyteorder()).byteswap()
    # 将数组中每个元素加上虚数部分为2的复数
    a += 2j
    # 创建一个迭代器对象，指定使用缓冲区模式和外部循环模式
    i = nditer(a, ['buffered', 'external_loop'],
                   [['readwrite', 'nbo', 'aligned']],
                   casting='same_kind',
                   op_dtypes=[np.dtype('c16')],
                   buffersize=3)
    with i:
        for v in i:
            # 对迭代器中的每个元素进行乘以2的操作
            v[...] *= 2
    # 断言数组a的内容与期望值相等
    assert_equal(a, 2*np.arange(10, dtype=np.clongdouble) + 4j)

    # 创建一个长度为10的长双精度浮点数数组
    a = np.arange(10, dtype=np.longdouble)
    # 将数组视图转换为新的字节顺序，并进行字节交换
    a = a.view(a.dtype.newbyteorder()).byteswap()
    # 创建一个迭代器对象，指定使用缓冲区模式和外部循环模式
    i = nditer(a, ['buffered', 'external_loop'],
                   [['readwrite', 'nbo', 'aligned']],
                   casting='same_kind',
                   op_dtypes=[np.dtype('f4')],
                   buffersize=7)
    with i:
        for v in i:
            # 对迭代器中的每个元素进行乘以2的操作
            v[...] *= 2
    # 断言数组a的内容与期望值相等
    assert_equal(a, 2*np.arange(10, dtype=np.longdouble))

def test_iter_buffered_cast_structured_type():
    # 待完成：测试缓冲区处理结构化类型的类型转换
    pass
    # Tests buffering of structured types

    # simple -> struct type (duplicates the value)
    # 定义结构化数据类型sdt，包含四个字段：'a'为float32，'b'为int64，'c'为complex128的数组，'d'为对象类型
    sdt = [('a', 'f4'), ('b', 'i8'), ('c', 'c8', (2, 3)), ('d', 'O')]
    # 创建一个float32类型的数组a，内容为[0.5, 1.5, 2.5]
    a = np.arange(3, dtype='f4') + 0.5
    # 使用nditer迭代器处理数组a，采用缓冲方式和允许引用，只读访问，类型转换为不安全操作，操作数据类型为sdt
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],
                    casting='unsafe',
                    op_dtypes=sdt)
    # 将迭代器i的每个元素转换为数组形式，存储在列表vals中
    vals = [np.array(x) for x in i]
    # 验证各字段值是否符合预期
    assert_equal(vals[0]['a'], 0.5)
    assert_equal(vals[0]['b'], 0)
    assert_equal(vals[0]['c'], [[(0.5)]*3]*2)
    assert_equal(vals[0]['d'], 0.5)
    assert_equal(vals[1]['a'], 1.5)
    assert_equal(vals[1]['b'], 1)
    assert_equal(vals[1]['c'], [[(1.5)]*3]*2)
    assert_equal(vals[1]['d'], 1.5)
    # 验证vals[0]的数据类型是否与sdt相符
    assert_equal(vals[0].dtype, np.dtype(sdt))

    # object -> struct type
    # 定义结构化数据类型sdt，包含四个字段：'a'为float32，'b'为int64，'c'为complex128的数组，'d'为对象类型
    sdt = [('a', 'f4'), ('b', 'i8'), ('c', 'c8', (2, 3)), ('d', 'O')]
    # 创建一个对象类型的数组a，长度为3
    a = np.zeros((3,), dtype='O')
    # 向数组a的每个元素赋值，每个元素都是一个元组，符合sdt的描述
    a[0] = (0.5, 0.5, [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], 0.5)
    a[1] = (1.5, 1.5, [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]], 1.5)
    a[2] = (2.5, 2.5, [[2.5, 2.5, 2.5], [2.5, 2.5, 2.5]], 2.5)
    # 如果支持引用计数，获取a[0]的引用计数
    if HAS_REFCOUNT:
        rc = sys.getrefcount(a[0])
    # 使用nditer迭代器处理数组a，采用缓冲方式和允许引用，只读访问，类型转换为不安全操作，操作数据类型为sdt
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],
                    casting='unsafe',
                    op_dtypes=sdt)
    # 将迭代器i的每个元素拷贝为新的对象，存储在列表vals中
    vals = [x.copy() for x in i]
    # 验证各字段值是否符合预期
    assert_equal(vals[0]['a'], 0.5)
    assert_equal(vals[0]['b'], 0)
    assert_equal(vals[0]['c'], [[(0.5)]*3]*2)
    assert_equal(vals[0]['d'], 0.5)
    assert_equal(vals[1]['a'], 1.5)
    assert_equal(vals[1]['b'], 1)
    assert_equal(vals[1]['c'], [[(1.5)]*3]*2)
    assert_equal(vals[1]['d'], 1.5)
    # 验证vals[0]的数据类型是否与sdt相符
    assert_equal(vals[0].dtype, np.dtype(sdt))
    # 清空vals、i和x变量
    vals, i, x = [None]*3
    # 如果支持引用计数，验证a[0]的引用计数是否恢复到之前的值
    if HAS_REFCOUNT:
        assert_equal(sys.getrefcount(a[0]), rc)

    # single-field struct type -> simple
    # 定义结构化数据类型sdt，包含一个字段：'a'为float32
    sdt = [('a', 'f4')]
    # 创建一个sdt类型的数组a，包含两个元素：(5.5,)和(8,)
    a = np.array([(5.5,), (8,)], dtype=sdt)
    # 使用nditer迭代器处理数组a，采用缓冲方式和允许引用，只读访问，类型转换为不安全操作，操作数据类型为'i4'
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],
                    casting='unsafe',
                    op_dtypes='i4')
    # 验证迭代器i的每个元素，提取出的值为[5, 8]
    assert_equal([x_[()] for x_ in i], [5, 8])

    # make sure multi-field struct type -> simple doesn't work
    # 定义结构化数据类型sdt，包含三个字段：'a'为float32，'b'为int64，'d'为对象类型
    sdt = [('a', 'f4'), ('b', 'i8'), ('d', 'O')]
    # 创建一个sdt类型的数组a，包含两个元素：(5.5, 7, 'test')和(8, 10, 11)
    a = np.array([(5.5, 7, 'test'), (8, 10, 11)], dtype=sdt)
    # 断言会抛出TypeError异常，因为无法将多字段结构类型转换为简单类型'i4'
    assert_raises(TypeError, lambda: (
        nditer(a, ['buffered', 'refs_ok'], ['readonly'],
               casting='unsafe',
               op_dtypes='i4')))

    # struct type -> struct type (field-wise copy)
    # 定义两个结构化数据类型sdt1和sdt2
    sdt1 = [('a', 'f4'), ('b', 'i8'), ('d', 'O')]
    sdt2 = [('d', 'u2'), ('a', 'O'), ('b', 'f8')]
    # 创建一个sdt1类型的数组a，包含两个元素：(1, 2, 3)和(4, 5, 6)
    a = np.array([(1, 2, 3), (4, 5, 6)], dtype=sdt1)
    # 使用nditer迭代器处理数组a，采用缓冲方式和允许引用，只读访问，类型转换为不安全操作，操作数据类型为sdt2
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],
                    casting='unsafe',
                    op_dtypes=sdt2)
    # 验证迭代器i的第一个元素的数据类型是否与sdt2相符
    assert_equal(i[0].dtype, np.dtype(sdt2))
    # 验证迭代器i的每个元素，转换为数组后是否与预期值相等
    assert_equal([np.array(x_) for x_ in i],
                 [np.array((1, 2, 3), dtype=sdt2),
                  np.array((4, 5, 6), dtype=sdt2)])
def test_iter_buffered_cast_structured_type_failure_with_cleanup():
    # 确保结构类型到结构类型，字段数量不同时会失败
    sdt1 = [('a', 'f4'), ('b', 'i8'), ('d', 'O')]  # 定义结构化数据类型 sdt1
    sdt2 = [('b', 'O'), ('a', 'f8')]  # 定义结构化数据类型 sdt2
    a = np.array([(1, 2, 3), (4, 5, 6)], dtype=sdt1)  # 创建一个包含 sdt1 类型的数组 a

    for intent in ["readwrite", "readonly", "writeonly"]:
        # 这个测试最初设计用于检测不同地方的错误，
        # 但现在由于转换不可能发生而提前抛出异常：
        # `assert np.can_cast(a.dtype, sdt2, casting="unsafe")` 会失败。
        # 如果没有错误的 DType，可能没有可靠的方式获得最初测试的行为。
        simple_arr = np.array([1, 2], dtype="i,i")  # 需要清理的数组
        with pytest.raises(TypeError):
            nditer((simple_arr, a), ['buffered', 'refs_ok'], [intent, intent],
                   casting='unsafe', op_dtypes=["f,f", sdt2])


def test_buffered_cast_error_paths():
    with pytest.raises(ValueError):
        # 输入被转换为 `S3` 缓冲区
        np.nditer((np.array("a", dtype="S1"),), op_dtypes=["i"],
                  casting="unsafe", flags=["buffered"])

    # `M8[ns]` 被转换为 `S3` 输出
    it = np.nditer((np.array(1, dtype="i"),), op_dtypes=["S1"],
                   op_flags=["writeonly"], casting="unsafe", flags=["buffered"])
    with pytest.raises(ValueError):
        with it:
            buf = next(it)
            buf[...] = "a"  # 无法转换为整数


@pytest.mark.skipif(IS_WASM, reason="Cannot start subprocess")
@pytest.mark.skipif(not HAS_REFCOUNT, reason="PyPy seems to not hit this.")
def test_buffered_cast_error_paths_unraisable():
    # 下面的代码会导致无法处理的错误。有时 pytest 会捕获它
    # (取决于 Python 和/或 pytest 版本)。因此，在 Python>=3.8 中，
    # 可能可以清除以检查 pytest.PytestUnraisableExceptionWarning：
    code = textwrap.dedent("""
        import numpy as np

        it = np.nditer((np.array(1, dtype="i"),), op_dtypes=["S1"],
                       op_flags=["writeonly"], casting="unsafe", flags=["buffered"])
        buf = next(it)
        buf[...] = "a"
        del buf, it  # 目前只在 deallocate 期间进行刷新。
        """)
    res = subprocess.check_output([sys.executable, "-c", code],
                                  stderr=subprocess.STDOUT, text=True)
    assert "ValueError" in res


def test_iter_buffered_cast_subarray():
    # 测试子数组的缓冲

    # 一个元素 -> 多个元素 (复制到所有元素)
    sdt1 = [('a', 'f4')]
    sdt2 = [('a', 'f8', (3, 2, 2))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'] = np.arange(6)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],
               casting='unsafe',
               op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    for x, count in zip(i, list(range(6))):
        assert_(np.all(x['a'] == count))
    # one element -> many -> back (copies it to all)
    sdt1 = [('a', 'O', (1, 1))]  # 定义结构化数据类型 sdt1，包含一个元素 'a'，类型为对象，形状为 (1, 1)
    sdt2 = [('a', 'O', (3, 2, 2))]  # 定义结构化数据类型 sdt2，包含一个元素 'a'，类型为对象，形状为 (3, 2, 2)
    a = np.zeros((6,), dtype=sdt1)  # 创建一个形状为 (6,) 的数组 a，数据类型为 sdt1，初始值都为 0
    a['a'][:, 0, 0] = np.arange(6)  # 在 a 数组的 'a' 元素中按照指定索引分配值
    i = nditer(a, ['buffered', 'refs_ok'], ['readwrite'],  # 创建一个迭代器 i 来遍历 a 数组
                    casting='unsafe',
                    op_dtypes=sdt2)
    with i:  # 使用 with 块来确保迭代器正确释放资源
        assert_equal(i[0].dtype, np.dtype(sdt2))  # 断言迭代器的第一个元素的数据类型与 sdt2 相同
        count = 0
        for x in i:  # 遍历迭代器 i
            assert_(np.all(x['a'] == count))  # 断言迭代器中当前元素的 'a' 值是否与 count 相等
            x['a'][0] += 2  # 修改迭代器中当前元素的 'a' 值的第一个元素
            count += 1
    assert_equal(a['a'], np.arange(6).reshape(6, 1, 1)+2)  # 断言数组 a 的 'a' 元素是否与给定值相等

    # many -> one element -> back (copies just element 0)
    sdt1 = [('a', 'O', (3, 2, 2))]  # 定义结构化数据类型 sdt1，包含一个元素 'a'，类型为对象，形状为 (3, 2, 2)
    sdt2 = [('a', 'O', (1,))]  # 定义结构化数据类型 sdt2，包含一个元素 'a'，类型为对象，形状为 (1,)
    a = np.zeros((6,), dtype=sdt1)  # 创建一个形状为 (6,) 的数组 a，数据类型为 sdt1，初始值都为 0
    a['a'][:, 0, 0, 0] = np.arange(6)  # 在 a 数组的 'a' 元素中按照指定索引分配值
    i = nditer(a, ['buffered', 'refs_ok'], ['readwrite'],  # 创建一个迭代器 i 来遍历 a 数组
                    casting='unsafe',
                    op_dtypes=sdt2)
    with i:  # 使用 with 块来确保迭代器正确释放资源
        assert_equal(i[0].dtype, np.dtype(sdt2))  # 断言迭代器的第一个元素的数据类型与 sdt2 相同
        count = 0
        for x in i:  # 遍历迭代器 i
            assert_equal(x['a'], count)  # 断言迭代器中当前元素的 'a' 值是否与 count 相等
            x['a'] += 2  # 修改迭代器中当前元素的 'a' 值
            count += 1
    assert_equal(a['a'], np.arange(6).reshape(6, 1, 1, 1)*np.ones((1, 3, 2, 2))+2)  # 断言数组 a 的 'a' 元素是否与给定值相等

    # many -> one element -> back (copies just element 0)
    sdt1 = [('a', 'f8', (3, 2, 2))]  # 定义结构化数据类型 sdt1，包含一个元素 'a'，类型为 float64，形状为 (3, 2, 2)
    sdt2 = [('a', 'O', (1,))]  # 定义结构化数据类型 sdt2，包含一个元素 'a'，类型为对象，形状为 (1,)
    a = np.zeros((6,), dtype=sdt1)  # 创建一个形状为 (6,) 的数组 a，数据类型为 sdt1，初始值都为 0
    a['a'][:, 0, 0, 0] = np.arange(6)  # 在 a 数组的 'a' 元素中按照指定索引分配值
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],  # 创建一个只读迭代器 i 来遍历 a 数组
                    casting='unsafe',
                    op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))  # 断言迭代器的第一个元素的数据类型与 sdt2 相同
    count = 0
    for x in i:  # 遍历迭代器 i
        assert_equal(x['a'], count)  # 断言迭代器中当前元素的 'a' 值是否与 count 相等
        count += 1

    # many -> one element (copies just element 0)
    sdt1 = [('a', 'O', (3, 2, 2))]  # 定义结构化数据类型 sdt1，包含一个元素 'a'，类型为对象，形状为 (3, 2, 2)
    sdt2 = [('a', 'f4', (1,))]  # 定义结构化数据类型 sdt2，包含一个元素 'a'，类型为 float32，形状为 (1,)
    a = np.zeros((6,), dtype=sdt1)  # 创建一个形状为 (6,) 的数组 a，数据类型为 sdt1，初始值都为 0
    a['a'][:, 0, 0, 0] = np.arange(6)  # 在 a 数组的 'a' 元素中按照指定索引分配值
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],  # 创建一个只读迭代器 i 来遍历 a 数组
                    casting='unsafe',
                    op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))  # 断言迭代器的第一个元素的数据类型与 sdt2 相同
    count = 0
    for x in i:  # 遍历迭代器 i
        assert_equal(x['a'], count)  # 断言迭代器中当前元素的 'a' 值是否与 count 相等
        count += 1

    # many -> matching shape (straightforward copy)
    sdt1 = [('a', 'O', (3, 2, 2))]  # 定义结构化数据类型 sdt1，包含一个元素 'a'，类型为对象，形状为 (3, 2, 2)
    sdt2 = [('a', 'f4', (3, 2, 2))]  # 定义结构化数据类型 sdt2，包含一个元素 'a'，类型为 float32，形状为 (3, 2, 2)
    a = np.zeros((6,), dtype=sdt1)  # 创建一个形状为 (6,) 的数组 a，数据类型为 sdt1，初始值都为 0
    a['a'] = np.arange(6*3*2*2).reshape(6, 3, 2, 2)  # 在 a 数组的 'a' 元素中按照指定形状分配值
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],  # 创建一个只读迭代器 i 来遍历 a 数组
                    casting='unsafe',
                    op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))  # 断言迭代器的第一个元素的数据类型与 sdt2 相同
    count = 0
    for x in i:  # 遍历迭代
    # 遍历迭代器 i 中的每个元素 x
    for x in i:
        # 断言 x 字典中键 'a' 的值等于数组 a 中对应索引的元素 'a' 的前两个元素
        assert_equal(x['a'], a[count]['a'][:2])
        # 计数器加一
        count += 1

    # 定义结构数据类型 sdt1 和 sdt2，sdt1 是长度为 2 的浮点数数组，sdt2 是长度为 6 的浮点数数组
    sdt1 = [('a', 'f8', (2,))]
    sdt2 = [('a', 'f4', (6,))]
    # 创建一个全零数组 a，数据类型为 sdt1
    a = np.zeros((6,), dtype=sdt1)
    # 在数组 a 中的字段 'a' 中填充 0 到 11 的数值，形状为 (6, 2)
    a['a'] = np.arange(6*2).reshape(6, 2)
    # 使用 nditer 迭代 a，设置选项为 'buffered'、'refs_ok'，只读模式，类型转换不安全，操作数据类型为 sdt2
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],
                    casting='unsafe',
                    op_dtypes=sdt2)
    # 断言迭代器 i 的第一个元素的数据类型与 sdt2 相等
    assert_equal(i[0].dtype, np.dtype(sdt2))
    # 初始化计数器
    count = 0
    # 遍历迭代器 i 中的每个元素 x
    for x in i:
        # 断言 x 字典中键 'a' 的前两个元素等于数组 a 中对应索引的元素 'a'
        assert_equal(x['a'][:2], a[count]['a'])
        # 断言 x 字典中键 'a' 的后四个元素等于 [0, 0, 0, 0]
        assert_equal(x['a'][2:], [0, 0, 0, 0])
        # 计数器加一
        count += 1

    # 定义结构数据类型 sdt1 和 sdt2，sdt1 是长度为 2 的浮点数数组，sdt2 是 2x2 的浮点数数组
    sdt1 = [('a', 'f8', (2,))]
    sdt2 = [('a', 'f4', (2, 2))]
    # 创建一个全零数组 a，数据类型为 sdt1
    a = np.zeros((6,), dtype=sdt1)
    # 在数组 a 中的字段 'a' 中填充 0 到 11 的数值，形状为 (6, 2)
    a['a'] = np.arange(6*2).reshape(6, 2)
    # 使用 nditer 迭代 a，设置选项为 'buffered'、'refs_ok'，只读模式，类型转换不安全，操作数据类型为 sdt2
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],
                    casting='unsafe',
                    op_dtypes=sdt2)
    # 断言迭代器 i 的第一个元素的数据类型与 sdt2 相等
    assert_equal(i[0].dtype, np.dtype(sdt2))
    # 初始化计数器
    count = 0
    # 遍历迭代器 i 中的每个元素 x
    for x in i:
        # 断言 x 字典中键 'a' 的第一个元素等于数组 a 中对应索引的元素 'a'
        assert_equal(x['a'][0], a[count]['a'])
        # 断言 x 字典中键 'a' 的第二个元素等于数组 a 中对应索引的元素 'a'
        assert_equal(x['a'][1], a[count]['a'])
        # 计数器加一
        count += 1

    # 定义结构数据类型 sdt1 和 sdt2，sdt1 是形状为 (2, 1) 的浮点数数组，sdt2 是 3x2 的浮点数数组
    sdt1 = [('a', 'f8', (2, 1))]
    sdt2 = [('a', 'f4', (3, 2))]
    # 创建一个全零数组 a，数据类型为 sdt1
    a = np.zeros((6,), dtype=sdt1)
    # 在数组 a 中的字段 'a' 中填充 0 到 11 的数值，形状为 (6, 2, 1)
    a['a'] = np.arange(6*2).reshape(6, 2, 1)
    # 使用 nditer 迭代 a，设置选项为 'buffered'、'refs_ok'，只读模式，类型转换不安全，操作数据类型为 sdt2
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],
                    casting='unsafe',
                    op_dtypes=sdt2)
    # 断言迭代器 i 的第一个元素的数据类型与 sdt2 相等
    assert_equal(i[0].dtype, np.dtype(sdt2))
    # 初始化计数器
    count = 0
    # 遍历迭代器 i 中的每个元素 x
    for x in i:
        # 断言 x 字典中键 'a' 的前两行第一列元素等于数组 a 中对应索引的元素 'a' 的所有行第一列元素
        assert_equal(x['a'][:2, 0], a[count]['a'][:, 0])
        # 断言 x 字典中键 'a' 的前两行第二列元素等于数组 a 中对应索引的元素 'a' 的所有行第一列元素
        assert_equal(x['a'][:2, 1], a[count]['a'][:, 0])
        # 断言 x 字典中键 'a' 的第三行等于 [0, 0]
        assert_equal(x['a'][2,:], [0, 0])
        # 计数器加一
        count += 1

    # 定义结构数据类型 sdt1 和 sdt2，sdt1 是形状为 (2, 3) 的浮点数数组，sdt2 是 3x2 的浮点数数组
    sdt1 = [('a', 'f8', (2, 3))]
    sdt2 = [('a', 'f4', (3, 2))]
    # 创建一个全零数组 a，数据类型为 sdt1
    a = np.zeros((6,), dtype=sdt1)
    # 在数组 a 中的字段 'a' 中填充 0 到 35 的数值，形状为 (6, 2, 3)
    a['a'] = np.arange(6*2*3).reshape(6, 2, 3)
    # 使用 nditer 迭代 a，设置选项为 'buffered'、'refs_ok'，只读模式，类型转换不安全，操作数据类型为 sdt2
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],
                    casting='unsafe',
                    op_dtypes=sdt2)
    # 断言迭代器 i 的第一个元素的数据类型与 sdt2 相等
    assert_equal(i[0].dtype, np.dtype(sdt2))
    # 初始化计数器
    count = 0
    # 遍历迭代器 i 中的每个元素 x
    for x in i:
        # 断言 x 字典中键 'a' 的前两行第一列元素等于数组 a 中对应索引的元素 'a' 的所有行第一列元素
        assert_equal(x['a'][:2, 0], a[count]['a'][:, 0])
        # 断言 x 字典中键 'a' 的前两行第二列元素等于数组 a 中对应索引的元素 'a' 的所有行第二列元素
        assert_equal(x['a'][:2, 1], a[count]['a'][:, 1])
        # 断言 x 字典中键 'a' 的第三行等于 [0, 0]
        assert_equal(x['a'][2,:], [0, 0])
        # 计数器加一
        count += 1
# Writing back from a buffer cannot combine elements

# a needs write buffering, but had a broadcast dimension
a = np.arange(6).reshape(2, 3, 1)
b = np.arange(12).reshape(2, 3, 2)
# 断言：如果使用了缓冲区并且外部循环，则抛出值错误异常
assert_raises(ValueError, nditer, [a, b],
              ['buffered', 'external_loop'],
              [['readwrite'], ['writeonly']],
              order='C')

# But if a is readonly, it's fine
nditer([a, b], ['buffered', 'external_loop'],
       [['readonly'], ['writeonly']],
       order='C')

# If a has just one element, it's fine too (constant 0 stride, a reduction)
a = np.arange(1).reshape(1, 1, 1)
nditer([a, b], ['buffered', 'external_loop', 'reduce_ok'],
       [['readwrite'], ['writeonly']],
       order='C')

# check that it fails on other dimensions too
a = np.arange(6).reshape(1, 3, 2)
# 断言：如果使用了缓冲区并且外部循环，则抛出值错误异常
assert_raises(ValueError, nditer, [a, b],
              ['buffered', 'external_loop'],
              [['readwrite'], ['writeonly']],
              order='C')
a = np.arange(4).reshape(2, 1, 2)
# 断言：如果使用了缓冲区并且外部循环，则抛出值错误异常
assert_raises(ValueError, nditer, [a, b],
              ['buffered', 'external_loop'],
              [['readwrite'], ['writeonly']],
              order='C')

# Safe casting disallows shrinking strings
a = np.array(['abc', 'a', 'abcd'], dtype=np.bytes_)
assert_equal(a.dtype, np.dtype('S4'))
# 断言：使用缓冲区并且只读时，强制操作类型 'S2' 会引发类型错误异常
assert_raises(TypeError, nditer, a, ['buffered'], ['readonly'],
              op_dtypes='S2')
i = nditer(a, ['buffered'], ['readonly'], op_dtypes='S6')
assert_equal(i[0], b'abc')
assert_equal(i[0].dtype, np.dtype('S6'))

a = np.array(['abc', 'a', 'abcd'], dtype=np.str_)
assert_equal(a.dtype, np.dtype('U4'))
# 断言：使用缓冲区并且只读时，强制操作类型 'U2' 会引发类型错误异常
assert_raises(TypeError, nditer, a, ['buffered'], ['readonly'],
              op_dtypes='U2')
i = nditer(a, ['buffered'], ['readonly'], op_dtypes='U6')
assert_equal(i[0], 'abc')
assert_equal(i[0].dtype, np.dtype('U6'))

# Test that the inner loop grows when no buffering is needed
a = np.arange(30)
i = nditer(a, ['buffered', 'growinner', 'external_loop'],
           buffersize=5)
# Should end up with just one inner loop here
assert_equal(i[0].size, a.size)


@pytest.mark.slow
def test_iter_buffered_reduce_reuse():
    # large enough array for all views, including negative strides.
    a = np.arange(2*3**5)[3**5:3**5+1]
    flags = ['buffered', 'delay_bufalloc', 'multi_index', 'reduce_ok', 'refs_ok']
    op_flags = [('readonly',), ('readwrite', 'allocate')]
    op_axes_list = [[(0, 1, 2), (0, 1, -1)], [(0, 1, 2), (0, -1, -1)]]
    # wrong dtype to force buffering
    op_dtypes = [float, a.dtype]
    # 定义生成器函数，用于生成测试参数
    def get_params():
        # 迭代 xs 的取值范围从 -9 到 9，共 19 个值
        for xs in range(-3**2, 3**2 + 1):
            # 迭代 ys 的取值范围从 xs 到 9，根据 xs 的不同而变化
            for ys in range(xs, 3**2 + 1):
                # 迭代操作轴列表中的每一个元素
                for op_axes in op_axes_list:
                    # 根据 xs, ys 和数组 a 的字节大小计算步长
                    strides = (xs * a.itemsize, ys * a.itemsize, a.itemsize)
                    # 使用 np.lib.stride_tricks.as_strided 函数创建一个视图数组 arr
                    arr = np.lib.stride_tricks.as_strided(a, (3, 3, 3), strides)

                    # 迭代 skip 取值为 0 和 1
                    for skip in [0, 1]:
                        # 生成器返回 arr, op_axes, skip 作为一组参数
                        yield arr, op_axes, skip

    # 使用 get_params 函数迭代获取测试参数，并执行相应的操作
    for arr, op_axes, skip in get_params():
        # 创建一个多维迭代器对象 nditer2，用于处理 arr 和 None
        nditer2 = np.nditer([arr.copy(), None],
                            op_axes=op_axes, flags=flags, op_flags=op_flags,
                            op_dtypes=op_dtypes)
        with nditer2:
            # 将 nditer2 迭代对象的最后一个操作数置为 0
            nditer2.operands[-1][...] = 0
            nditer2.reset()
            # 设置 nditer2 的迭代索引为 skip
            nditer2.iterindex = skip

            # 遍历 nditer2 中的每对元素 (a2_in, b2_in)
            for (a2_in, b2_in) in nditer2:
                # 将 a2_in 转换为 np.int_ 类型后加到 b2_in 中
                b2_in += a2_in.astype(np.int_)

            # 获取 nditer2 的最后一个操作数，用于后续的比较
            comp_res = nditer2.operands[-1]

        # 迭代 bufsize 的取值范围从 0 到 27
        for bufsize in range(0, 3**3):
            # 创建一个多维迭代器对象 nditer1，用于处理 arr 和 None
            nditer1 = np.nditer([arr, None],
                                op_axes=op_axes, flags=flags, op_flags=op_flags,
                                buffersize=bufsize, op_dtypes=op_dtypes)
            with nditer1:
                # 将 nditer1 迭代对象的最后一个操作数置为 0
                nditer1.operands[-1][...] = 0
                nditer1.reset()
                # 设置 nditer1 的迭代索引为 skip
                nditer1.iterindex = skip

                # 遍历 nditer1 中的每对元素 (a1_in, b1_in)
                for (a1_in, b1_in) in nditer1:
                    # 将 a1_in 转换为 np.int_ 类型后加到 b1_in 中
                    b1_in += a1_in.astype(np.int_)

                # 获取 nditer1 的最后一个操作数，用于断言比较
                res = nditer1.operands[-1]
            # 使用 assert_array_equal 函数断言 res 与 comp_res 相等
            assert_array_equal(res, comp_res)
def test_iter_no_broadcast():
    # 测试 no_broadcast 标志是否起作用
    # 创建三个 NumPy 数组 a, b, c
    a = np.arange(24).reshape(2, 3, 4)
    b = np.arange(6).reshape(2, 3, 1)
    c = np.arange(12).reshape(3, 4)

    # 测试 nditer 函数的使用，验证 no_broadcast 标志是否正确处理
    nditer([a, b, c], [],
           [['readonly', 'no_broadcast'],  # a 使用 no_broadcast
            ['readonly'],  # b 没有 no_broadcast
            ['readonly']])  # c 没有 no_broadcast

    # 预期会抛出 ValueError 异常，因为 b 中没有 no_broadcast 标志
    assert_raises(ValueError, nditer, [a, b, c], [],
                  [['readonly'], ['readonly', 'no_broadcast'], ['readonly']])

    # 预期会抛出 ValueError 异常，因为 c 中没有 no_broadcast 标志
    assert_raises(ValueError, nditer, [a, b, c], [],
                  [['readonly'], ['readonly'], ['readonly', 'no_broadcast']])


class TestIterNested:

    def test_basic(self):
        # 测试嵌套迭代的基本用法
        a = arange(12).reshape(2, 3, 2)

        # 使用 nested_iters 函数，迭代数组 a
        # 第一次迭代的索引是 [0], 第二次迭代的索引是 [1, 2]
        i, j = np.nested_iters(a, [[0], [1, 2]])
        # 收集迭代结果并验证是否符合预期
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])

        # 第一次迭代的索引是 [0, 1], 第二次迭代的索引是 [2]
        i, j = np.nested_iters(a, [[0, 1], [2]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])

        # 第一次迭代的索引是 [0, 2], 第二次迭代的索引是 [1]
        i, j = np.nested_iters(a, [[0, 2], [1]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 2, 4], [1, 3, 5], [6, 8, 10], [7, 9, 11]])

    def test_reorder(self):
        # 测试嵌套迭代的基本用法
        a = arange(12).reshape(2, 3, 2)

        # 使用 nested_iters 函数，迭代数组 a
        # 在 'K' order 下，重新排序
        i, j = np.nested_iters(a, [[0], [2, 1]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])

        # 第一次迭代的索引是 [1, 0], 第二次迭代的索引是 [2]
        i, j = np.nested_iters(a, [[1, 0], [2]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])

        # 第一次迭代的索引是 [2, 0], 第二次迭代的索引是 [1]
        i, j = np.nested_iters(a, [[2, 0], [1]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 2, 4], [1, 3, 5], [6, 8, 10], [7, 9, 11]])

        # 在 'C' order 下，不会重新排序
        i, j = np.nested_iters(a, [[0], [2, 1]], order='C')
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 2, 4, 1, 3, 5], [6, 8, 10, 7, 9, 11]])

        # 第一次迭代的索引是 [1, 0], 第二次迭代的索引是 [2]
        i, j = np.nested_iters(a, [[1, 0], [2]], order='C')
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 1], [6, 7], [2, 3], [8, 9], [4, 5], [10, 11]])

        # 第一次迭代的索引是 [2, 0], 第二次迭代的索引是 [1]
        i, j = np.nested_iters(a, [[2, 0], [1]], order='C')
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 2, 4], [6, 8, 10], [1, 3, 5], [7, 9, 11]])
    def test_flip_axes(self):
        # Test nested iteration with negative axes
        # 创建一个12个元素的一维数组，并将其形状调整为(2, 3, 2)，然后沿着各个轴向进行反转操作，生成数组a
        a = arange(12).reshape(2, 3, 2)[::-1, ::-1, ::-1]

        # In 'K' order (default), the axes all get flipped
        # 使用 'K' 顺序（默认顺序），对a进行嵌套迭代，第一个轴为[0]，第二个轴为[1, 2]
        i, j = np.nested_iters(a, [[0], [1, 2]])
        # 获取所有迭代值并存储在vals列表中
        vals = [list(j) for _ in i]
        # 断言vals的值与期望值相等
        assert_equal(vals, [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])

        # 对a进行嵌套迭代，第一个轴为[0, 1]，第二个轴为[2]
        i, j = np.nested_iters(a, [[0, 1], [2]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])

        # 对a进行嵌套迭代，第一个轴为[0, 2]，第二个轴为[1]
        i, j = np.nested_iters(a, [[0, 2], [1]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 2, 4], [1, 3, 5], [6, 8, 10], [7, 9, 11]])

        # In 'C' order, flipping axes is disabled
        # 使用 'C' 顺序，对a进行嵌套迭代，第一个轴为[0]，第二个轴为[1, 2]
        i, j = np.nested_iters(a, [[0], [1, 2]], order='C')
        vals = [list(j) for _ in i]
        assert_equal(vals, [[11, 10, 9, 8, 7, 6], [5, 4, 3, 2, 1, 0]])

        # 对a进行嵌套迭代，第一个轴为[0, 1]，第二个轴为[2]
        i, j = np.nested_iters(a, [[0, 1], [2]], order='C')
        vals = [list(j) for _ in i]
        assert_equal(vals, [[11, 10], [9, 8], [7, 6], [5, 4], [3, 2], [1, 0]])

        # 对a进行嵌套迭代，第一个轴为[0, 2]，第二个轴为[1]
        i, j = np.nested_iters(a, [[0, 2], [1]], order='C')
        vals = [list(j) for _ in i]
        assert_equal(vals, [[11, 9, 7], [10, 8, 6], [5, 3, 1], [4, 2, 0]])

    def test_broadcast(self):
        # Test nested iteration with broadcasting
        # 创建两个一维数组a和b，并分别将其形状调整为(2, 1)和(1, 3)
        a = arange(2).reshape(2, 1)
        b = arange(3).reshape(1, 3)

        # 对[a, b]进行嵌套迭代，第一个轴为[0]，第二个轴为[1]
        i, j = np.nested_iters([a, b], [[0], [1]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[[0, 0], [0, 1], [0, 2]], [[1, 0], [1, 1], [1, 2]]])

        # 对[a, b]进行嵌套迭代，第一个轴为[1]，第二个轴为[0]
        i, j = np.nested_iters([a, b], [[1], [0]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[[0, 0], [1, 0]], [[0, 1], [1, 1]], [[0, 2], [1, 2]]])
    def test_dtype_copy(self):
        # Test nested iteration with a copy to change dtype

        # 创建一个整数类型的数组，并进行形状变换
        a = arange(6, dtype='i4').reshape(2, 3)
        # 使用 nested_iters 函数创建迭代器 i 和 j，其中 j 是一个复制的只读迭代器
        i, j = np.nested_iters(a, [[0], [1]],
                            op_flags=['readonly', 'copy'],
                            op_dtypes='f8')
        # 断言复制后的 j 的数据类型为 'f8'
        assert_equal(j[0].dtype, np.dtype('f8'))
        # 收集迭代器 i 的所有值，作为嵌套列表 vals
        vals = [list(j) for _ in i]
        # 断言 vals 的值与预期的嵌套列表相等
        assert_equal(vals, [[0, 1, 2], [3, 4, 5]])
        vals = None

        # writebackifcopy - 使用上下文管理器的方式
        # 创建一个单精度浮点类型的数组，并进行形状变换
        a = arange(6, dtype='f4').reshape(2, 3)
        # 使用 nested_iters 函数创建迭代器 i 和 j，j 是一个更新副本的读写迭代器
        i, j = np.nested_iters(a, [[0], [1]],
                            op_flags=['readwrite', 'updateifcopy'],
                            casting='same_kind',
                            op_dtypes='f8')
        # 使用上下文管理器，确保在迭代结束后正确更新数组 a
        with i, j:
            # 断言 j 的数据类型为 'f8'
            assert_equal(j[0].dtype, np.dtype('f8'))
            # 遍历 i 和 j，将 j 中的每个元素加 1
            for x in i:
                for y in j:
                    y[...] += 1
            # 断言数组 a 的值已更新为预期值
            assert_equal(a, [[0, 1, 2], [3, 4, 5]])
        # 再次断言数组 a 的值确保没有变化
        assert_equal(a, [[1, 2, 3], [4, 5, 6]])

        # writebackifcopy - 使用 close() 方法的方式
        # 创建一个单精度浮点类型的数组，并进行形状变换
        a = arange(6, dtype='f4').reshape(2, 3)
        # 使用 nested_iters 函数创建迭代器 i 和 j，j 是一个更新副本的读写迭代器
        i, j = np.nested_iters(a, [[0], [1]],
                            op_flags=['readwrite', 'updateifcopy'],
                            casting='same_kind',
                            op_dtypes='f8')
        # 断言 j 的数据类型为 'f8'
        assert_equal(j[0].dtype, np.dtype('f8'))
        # 遍历 i 和 j，将 j 中的每个元素加 1
        for x in i:
            for y in j:
                y[...] += 1
        # 断言数组 a 的值已更新为预期值
        assert_equal(a, [[0, 1, 2], [3, 4, 5]])
        # 关闭迭代器 i 和 j
        i.close()
        j.close()
        # 再次断言数组 a 的值确保已更新
        assert_equal(a, [[1, 2, 3], [4, 5, 6]])

    def test_dtype_buffered(self):
        # Test nested iteration with buffering to change dtype

        # 创建一个单精度浮点类型的数组，并进行形状变换
        a = arange(6, dtype='f4').reshape(2, 3)
        # 使用 nested_iters 函数创建带缓冲的迭代器 i 和 j，j 是一个读写迭代器
        i, j = np.nested_iters(a, [[0], [1]],
                            flags=['buffered'],
                            op_flags=['readwrite'],
                            casting='same_kind',
                            op_dtypes='f8')
        # 断言 j 的数据类型为 'f8'
        assert_equal(j[0].dtype, np.dtype('f8'))
        # 遍历 i 和 j，将 j 中的每个元素加 1
        for x in i:
            for y in j:
                y[...] += 1
        # 断言数组 a 的值已更新为预期值
        assert_equal(a, [[1, 2, 3], [4, 5, 6]])

    def test_0d(self):
        # 创建一个三维数组
        a = np.arange(12).reshape(2, 3, 2)
        # 使用 nested_iters 函数创建迭代器 i 和 j，其中 j 遍历第二维和第一维的交错索引
        i, j = np.nested_iters(a, [[], [1, 0, 2]])
        # 收集迭代器 i 的所有值，作为嵌套列表 vals
        vals = [list(j) for _ in i]
        # 断言 vals 的值与预期的嵌套列表相等
        assert_equal(vals, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])

        # 使用 nested_iters 函数创建迭代器 i 和 j，其中 j 遍历第一维和第二维的交错索引
        i, j = np.nested_iters(a, [[1, 0, 2], []])
        # 收集迭代器 i 的所有值，作为嵌套列表 vals
        vals = [list(j) for _ in i]
        # 断言 vals 的值与预期的嵌套列表相等
        assert_equal(vals, [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]])

        # 使用 nested_iters 函数创建三个迭代器 i、j 和 k，j 和 k 遍历第三维和第一维的交错索引
        i, j, k = np.nested_iters(a, [[2, 0], [], [1]])
        # 初始化一个空列表 vals，用于收集每个迭代器的值
        vals = []
        # 遍历迭代器 i、j 和 k，收集每个组合的值作为嵌套列表并添加到 vals 中
        for x in i:
            for y in j:
                vals.append([z for z in k])
        # 断言 vals 的值与预期的嵌套列表相等
        assert_equal(vals, [[0, 2, 4], [1, 3, 5], [6, 8, 10], [7, 9, 11]])
    def test_iter_nested_iters_dtype_buffered(self):
        # 定义测试函数 test_iter_nested_iters_dtype_buffered，用于测试嵌套迭代和缓冲以改变数据类型

        # 创建一个包含6个单精度浮点数的数组，并reshape成2行3列的矩阵
        a = arange(6, dtype='f4').reshape(2, 3)
        
        # 使用 np.nested_iters 函数创建两个迭代器 i 和 j
        i, j = np.nested_iters(a, [[0], [1]],
                            flags=['buffered'],  # 使用缓冲标志
                            op_flags=['readwrite'],  # 操作标志设为读写
                            casting='same_kind',  # 强制转换方式为相同种类
                            op_dtypes='f8')  # 操作数据类型为双精度浮点数
        
        # 使用 with 语句管理迭代器 i 和 j
        with i, j:
            # 断言 j[0] 的数据类型为双精度浮点数
            assert_equal(j[0].dtype, np.dtype('f8'))
            
            # 嵌套循环迭代 i 和 j
            for x in i:
                for y in j:
                    # 对 y 的所有元素加1
                    y[...] += 1
        
        # 断言数组 a 的值是否变为 [[1, 2, 3], [4, 5, 6]]
        assert_equal(a, [[1, 2, 3], [4, 5, 6]])
def test_iter_reduction_error():
    # 定义测试函数，用于测试迭代器在减少操作时的错误处理

    a = np.arange(6)
    # 断言会引发 ValueError 异常，因为 op_axes 的维度与操作数不匹配
    assert_raises(ValueError, nditer, [a, None], [],
                    [['readonly'], ['readwrite', 'allocate']],
                    op_axes=[[0], [-1]])

    a = np.arange(6).reshape(2, 3)
    # 断言会引发 ValueError 异常，因为 op_axes 的维度与操作数不匹配
    assert_raises(ValueError, nditer, [a, None], ['external_loop'],
                    [['readonly'], ['readwrite', 'allocate']],
                    op_axes=[[0, 1], [-1, -1]])

def test_iter_reduction():
    # 测试使用迭代器进行减少操作

    a = np.arange(6)
    # 创建迭代器对象 i，用于执行 reduce_ok 操作
    i = nditer([a, None], ['reduce_ok'],
                    [['readonly'], ['readwrite', 'allocate']],
                    op_axes=[[0], [-1]])
    # 进入迭代器上下文
    with i:
        # 初始化输出操作数为加法单位
        i.operands[1][...] = 0
        # 执行减少操作
        for x, y in i:
            y[...] += x
        # 由于未指定轴，应分配一个标量
        assert_equal(i.operands[1].ndim, 0)
        assert_equal(i.operands[1], np.sum(a))

    a = np.arange(6).reshape(2, 3)
    # 创建迭代器对象 i，用于执行 reduce_ok 和 external_loop 操作
    i = nditer([a, None], ['reduce_ok', 'external_loop'],
                    [['readonly'], ['readwrite', 'allocate']],
                    op_axes=[[0, 1], [-1, -1]])
    # 进入迭代器上下文
    with i:
        # 初始化输出操作数为加法单位
        i.operands[1][...] = 0
        # 断言输出的形状和步幅
        assert_equal(i[1].shape, (6,))
        assert_equal(i[1].strides, (0,))
        # 执行减少操作
        for x, y in i:
            # 使用 for 循环而不是 ``y[...] += x``，
            # 因为 y 的步幅为零，我们使用这种方式进行减少
            for j in range(len(y)):
                y[j] += x[j]
        # 由于未指定轴，应分配一个标量
        assert_equal(i.operands[1].ndim, 0)
        assert_equal(i.operands[1], np.sum(a))

    # 这是一个复杂的减少情况，需要缓冲双循环处理
    a = np.ones((2, 3, 5))
    it1 = nditer([a, None], ['reduce_ok', 'external_loop'],
                    [['readonly'], ['readwrite', 'allocate']],
                    op_axes=[None, [0, -1, 1]])
    it2 = nditer([a, None], ['reduce_ok', 'external_loop',
                            'buffered', 'delay_bufalloc'],
                    [['readonly'], ['readwrite', 'allocate']],
                    op_axes=[None, [0, -1, 1]], buffersize=10)
    with it1, it2:
        # 初始化输出操作数为零
        it1.operands[1].fill(0)
        it2.operands[1].fill(0)
        it2.reset()
        # 执行迭代并进行减少操作
        for x in it1:
            x[1][...] += x[0]
        for x in it2:
            x[1][...] += x[0]
        # 断言两个迭代器的输出操作数相等
        assert_equal(it1.operands[1], it2.operands[1])
        # 断言 it2 的输出操作数的总和等于 a 的大小
        assert_equal(it2.operands[1].sum(), a.size)

def test_iter_buffering_reduction():
    # 测试使用迭代器进行带缓冲区的减少操作

    a = np.arange(6)
    b = np.array(0., dtype='f8').byteswap()
    b = b.view(b.dtype.newbyteorder())
    # 创建一个多维迭代器 `i`，用于同时迭代数组 `a` 和 `b`
    i = nditer([a, b], ['reduce_ok', 'buffered'],
                    [['readonly'], ['readwrite', 'nbo']],
                    op_axes=[[0], [-1]])
    # 使用迭代器 `i` 进行迭代
    with i:
        # 断言迭代器中第二个操作数 `i[1]` 的数据类型为双精度浮点型
        assert_equal(i[1].dtype, np.dtype('f8'))
        # 断言迭代器中第二个操作数 `i[1]` 的数据类型不等于 `b` 的数据类型
        assert_(i[1].dtype != b.dtype)
        # 执行归约操作
        # 使用迭代器 `i` 迭代每对元素 `x, y`
        for x, y in i:
            # 将 `x` 加到 `y` 上
            y[...] += x
    # 断言数组 `b` 等于数组 `a` 的所有元素之和
    # 因为未指定轴，应该分配一个标量
    assert_equal(b, np.sum(a))

    # 创建数组 `a` 和数组 `b`
    a = np.arange(6).reshape(2, 3)
    b = np.array([0, 0], dtype='f8').byteswap()
    b = b.view(b.dtype.newbyteorder())
    # 创建一个新的多维迭代器 `i`，用于同时迭代数组 `a` 和 `b`
    i = nditer([a, b], ['reduce_ok', 'external_loop', 'buffered'],
                    [['readonly'], ['readwrite', 'nbo']],
                    op_axes=[[0, 1], [0, -1]])
    # 用于输出的形状和步幅
    with i:
        # 断言迭代器中第二个操作数 `i[1]` 的形状为 (3,)
        assert_equal(i[1].shape, (3,))
        # 断言迭代器中第二个操作数 `i[1]` 的步幅为 (0,)
        assert_equal(i[1].strides, (0,))
        # 执行归约操作
        # 使用迭代器 `i` 迭代每对元素 `x, y`
        for x, y in i:
            # 使用 for 循环代替 `y[...] += x`，因为 `y` 的步幅为零，用于归约
            for j in range(len(y)):
                y[j] += x[j]
    # 断言数组 `b` 等于数组 `a` 沿轴 1 的和
    assert_equal(b, np.sum(a, axis=1))

    # 修正此迭代器内部的双重循环
    p = np.arange(2) + 1
    # 创建迭代器 `it`，用于迭代 `p` 和 `None`
    it = np.nditer([p, None],
            ['delay_bufalloc', 'reduce_ok', 'buffered', 'external_loop'],
            [['readonly'], ['readwrite', 'allocate']],
            op_axes=[[-1, 0], [-1, -1]],
            itershape=(2, 2))
    # 使用迭代器 `it`
    with it:
        # 将 `it` 中的第二个操作数填充为 0
        it.operands[1].fill(0)
        # 重置迭代器 `it`
        it.reset()
        # 断言迭代器中第一个元素 `it[0]` 为 [1, 2, 1, 2]
        assert_equal(it[0], [1, 2, 1, 2])

    # 迭代器内部循环应考虑参数的连续性
    x = np.ones((7, 13, 8), np.int8)[4:6,1:11:6,1:5].transpose(1, 2, 0)
    x[...] = np.arange(x.size).reshape(x.shape)
    y_base = np.arange(4*4, dtype=np.int8).reshape(4, 4)
    y_base_copy = y_base.copy()
    y = y_base[::2,:,None]

    # 创建迭代器 `it`，用于迭代 `y` 和 `x`
    it = np.nditer([y, x],
                   ['buffered', 'external_loop', 'reduce_ok'],
                   [['readwrite'], ['readonly']])
    # 使用迭代器 `it`
    with it:
        # 对于迭代器 `it` 中的每对元素 `a, b`
        for a, b in it:
            # 将 `a` 填充为 2
            a.fill(2)

    # 断言 `y_base` 中每隔一个元素的子数组等于 `y_base_copy` 的相应部分
    assert_equal(y_base[1::2], y_base_copy[1::2])
    # 断言 `y_base` 中每隔两个元素的子数组等于 2
    assert_equal(y_base[::2], 2)
def test_iter_buffering_reduction_reuse_reduce_loops():
    # 修复了重复使用 reduce 循环的 bug，导致处理过程中分块过小，超出缓冲区限制

    # 创建两个二维全零数组
    a = np.zeros((2, 7))
    b = np.zeros((1, 7))
    # 创建迭代器，使用缓冲和外部循环，并允许 reduce 操作
    it = np.nditer([a, b], flags=['reduce_ok', 'external_loop', 'buffered'],
                    op_flags=[['readonly'], ['readwrite']],
                    buffersize=5)

    with it:
        # 获取迭代过程中每个缓冲区的大小
        bufsizes = [x.shape[0] for x, y in it]
    # 断言检查每个缓冲区的大小列表
    assert_equal(bufsizes, [5, 2, 5, 2])
    # 断言检查所有缓冲区大小的总和是否等于数组 a 的大小
    assert_equal(sum(bufsizes), a.size)

def test_iter_writemasked_badinput():
    # 创建两个二维全零数组和其他掩码数组
    a = np.zeros((2, 3))
    b = np.zeros((3,))
    m = np.array([[True, True, False], [False, True, False]])
    m2 = np.array([True, True, False])
    m3 = np.array([0, 1, 1], dtype='u1')
    mbad1 = np.array([0, 1, 1], dtype='i1')
    mbad2 = np.array([0, 1, 1], dtype='f4')

    # 如果任何操作数是 writemasked，需要一个 arraymask
    assert_raises(ValueError, nditer, [a, m], [],
                    [['readwrite', 'writemasked'], ['readonly']])

    # writemasked 操作数不能是 readonly
    assert_raises(ValueError, nditer, [a, m], [],
                    [['readonly', 'writemasked'], ['readonly', 'arraymask']])

    # writemasked 和 arraymask 不能同时使用
    assert_raises(ValueError, nditer, [a, m], [],
                    [['readonly'], ['readwrite', 'arraymask', 'writemasked']])

    # arraymask 只能被指定一次
    assert_raises(ValueError, nditer, [a, m, m2], [],
                    [['readwrite', 'writemasked'],
                     ['readonly', 'arraymask'],
                     ['readonly', 'arraymask']])

    # 如果没有 writemasked，arraymask 也没有意义
    assert_raises(ValueError, nditer, [a, m], [],
                    [['readwrite'], ['readonly', 'arraymask']])

    # writemasked 的 reduce 操作需要一个相对较小的掩码
    assert_raises(ValueError, nditer, [a, b, m], ['reduce_ok'],
                    [['readonly'],
                     ['readwrite', 'writemasked'],
                     ['readonly', 'arraymask']])
    # 但如果掩码与 reduce 操作的操作数大小相等或更小，则应该可以工作
    np.nditer([a, b, m2], ['reduce_ok'],
                    [['readonly'],
                     ['readwrite', 'writemasked'],
                     ['readonly', 'arraymask']])
    # arraymask 本身不能是 reduce 操作
    assert_raises(ValueError, nditer, [a, b, m2], ['reduce_ok'],
                    [['readonly'],
                     ['readwrite', 'writemasked'],
                     ['readwrite', 'arraymask']])

    # uint8 类型的掩码也是可以的
    np.nditer([a, m3], ['buffered'],
                    [['readwrite', 'writemasked'],
                     ['readonly', 'arraymask']],
                    op_dtypes=['f4', None],
                    casting='same_kind')
    # int8 类型的掩码则不可以
    # 检查是否会引发 TypeError 异常，当尝试使用 np.nditer 对象迭代器时，
    # 在给定的参数中包括一个类型错误的浮点32位数掩码。
    assert_raises(TypeError, np.nditer, [a, mbad1], ['buffered'],
                    [['readwrite', 'writemasked'],
                     ['readonly', 'arraymask']],
                    op_dtypes=['f4', None],
                    casting='same_kind')
    # 检查是否会引发 TypeError 异常，当尝试使用 np.nditer 对象迭代器时，
    # 在给定的参数中包括另一个类型错误的浮点32位数掩码。
    assert_raises(TypeError, np.nditer, [a, mbad2], ['buffered'],
                    [['readwrite', 'writemasked'],
                     ['readonly', 'arraymask']],
                    op_dtypes=['f4', None],
                    casting='same_kind')
python
# 检查迭代器是否支持缓冲，如果不支持则返回 True
def _is_buffered(iterator):
    try:
        iterator.itviews
    except ValueError:
        return True
    # 支持缓冲则返回 False
    return False

# 使用参数化测试框架，针对不同的数组 a 执行多组测试
@pytest.mark.parametrize("a",
        [np.zeros((3,), dtype='f8'),
         np.zeros((9876, 3*5), dtype='f8')[::2, :],
         np.zeros((4, 312, 124, 3), dtype='f8')[::2, :, ::2, :],
         # Also test with the last dimension strided (so it does not fit if
         # there is repeated access)
         np.zeros((9,), dtype='f8')[::3],
         np.zeros((9876, 3*10), dtype='f8')[::2, ::5],
         np.zeros((4, 312, 124, 3), dtype='f8')[::2, :, ::2, ::-1]])
def test_iter_writemasked(a):
    # 注意，上面的切片操作旨在确保 nditer 无法将多个轴合并为一个。重复操作使得测试更加有趣。
    shape = a.shape
    reps = shape[-1] // 3
    msk = np.empty(shape, dtype=bool)
    msk[...] = [True, True, False] * reps

    # 当未使用缓冲时，'writemasked' 效果上什么也不做。
    # 用户需要遵守请求的语义。
    it = np.nditer([a, msk], [],
                [['readwrite', 'writemasked'],
                 ['readonly', 'arraymask']])
    with it:
        for x, m in it:
            x[...] = 1
    # 因为我们违反了语义，所有的值变成了 1
    assert_equal(a, np.broadcast_to([1, 1, 1] * reps, shape))

    # 即使启用了缓冲，我们仍然可能直接访问数组。
    it = np.nditer([a, msk], ['buffered'],
                [['readwrite', 'writemasked'],
                 ['readonly', 'arraymask']])
    # @seberg: 目前我不明白为什么一个“buffered”迭代器在这里使用“writemasked”时会不使用缓冲，这似乎令人困惑... 通过测试实际内存重叠来检查！
    is_buffered = True
    with it:
        for x, m in it:
            x[...] = 2.5
            if np.may_share_memory(x, a):
                is_buffered = False

    if not is_buffered:
        # 因为我们违反了语义，所有的值变成了 2.5
        assert_equal(a, np.broadcast_to([2.5, 2.5, 2.5] * reps, shape))
    else:
        # 对于大尺寸的情况，迭代器可能会使用缓冲：
        assert_equal(a, np.broadcast_to([2.5, 2.5, 1] * reps, shape))
        a[...] = 2.5

    # 如果肯定会发生缓冲，例如由于强制转换，只有由掩码选择的项目将从缓冲区复制回来。
    it = np.nditer([a, msk], ['buffered'],
                [['readwrite', 'writemasked'],
                 ['readonly', 'arraymask']],
                op_dtypes=['i8', None],
                casting='unsafe')
    with it:
        for x, m in it:
            x[...] = 3
    # 即使我们违反了语义，只有选定的值被复制回来
    assert_equal(a, np.broadcast_to([3, 3, 2.5] * reps, shape))
# 使用 pytest.mark.parametrize 装饰器来定义多个参数化测试用例
@pytest.mark.parametrize(["mask", "mask_axes"], [
    # 第一个测试用例：没有指定 mask，mask_axes 包含 [-1, 0]
    (None, [-1, 0]),
    # 第二个测试用例：mask 是一个 1x4 的布尔数组，mask_axes 包含 [0, 1]
    (np.zeros((1, 4), dtype="bool"), [0, 1]),
    # 第三个测试用例：mask 是一个 1x4 的布尔数组，没有指定 mask_axes
    (np.zeros((1, 4), dtype="bool"), None),
    # 第四个测试用例：mask 是一个长度为 4 的布尔数组，mask_axes 包含 [-1, 0]
    (np.zeros(4, dtype="bool"), [-1, 0]),
    # 第五个测试用例：mask 是一个标量（0维数组），mask_axes 包含 [-1, -1]
    (np.zeros((), dtype="bool"), [-1, -1]),
    # 第六个测试用例：mask 是一个标量（0维数组），没有指定 mask_axes
    (np.zeros((), dtype="bool"), None)
])
def test_iter_writemasked_broadcast_error(mask, mask_axes):
    # 如果 mask_axes 是 None，则 op_axes 也设置为 None；否则，op_axes 是一个包含 mask_axes 和 [0, 1] 的列表
    if mask_axes is None:
        op_axes = None
    else:
        op_axes = [mask_axes, [0, 1]]

    # 使用 assert_raises 检查 ValueError 是否被抛出
    with assert_raises(ValueError):
        # 使用 np.nditer 创建一个迭代器，迭代 mask 和 arr
        np.nditer((mask, arr), flags=["reduce_ok"], op_flags=[["arraymask", "readwrite", "allocate"], ["writeonly", "writemasked"]],
                  op_axes=op_axes)


def test_iter_writemasked_decref():
    # 创建一个结构化数据类型数组 arr，包含 10000 个元素，其中每个元素包含一个大整数和一个对象引用
    arr = np.arange(10000).astype(">i,O")
    original = arr.copy()
    # 创建一个长度为 10000 的布尔数组 mask
    mask = np.random.randint(0, 2, size=10000).astype(bool)

    # 使用 np.nditer 创建一个迭代器 it，迭代 arr 和 mask
    it = np.nditer([arr, mask], flags=['buffered', 'refs_ok'],
                   op_flags=[['readwrite', 'writemasked'], ['readonly', 'arraymask']],
                   op_dtypes=["<i,O", "?"])
    singleton = object()
    # 如果支持引用计数，则获取当前 singleton 的引用计数
    if HAS_REFCOUNT:
        count = sys.getrefcount(singleton)
    # 遍历迭代器
    for buf, mask_buf in it:
        # 将 buf 的内容设置为 (3, singleton)
        buf[...] = (3, singleton)

    # 删除 buf 和 mask_buf 以确保正确清理
    del buf, mask_buf, it

    # 如果支持引用计数，则检查 singleton 的引用计数是否增加了与 mask 中非零元素数量相同的值
    if HAS_REFCOUNT:
        assert sys.getrefcount(singleton) - count == np.count_nonzero(mask)

    # 检查 arr 中未被 mask 掩码的元素是否保持不变
    assert_array_equal(arr[~mask], original[~mask])
    # 检查 arr 中被 mask 掩码的元素是否被正确设置为 (3, singleton)
    assert (arr[mask] == np.array((3, singleton), arr.dtype)).all()
    # 删除 arr
    del arr

    # 如果支持引用计数，则检查 singleton 的引用计数是否与之前相同
    if HAS_REFCOUNT:
        assert sys.getrefcount(singleton) == count


def test_iter_non_writable_attribute_deletion():
    # 创建一个 np.nditer 对象 it，迭代一个包含两个元素的数组
    it = np.nditer(np.ones(2))
    # 定义需要删除的属性列表 attr
    attr = ["value", "shape", "operands", "itviews", "has_delayed_bufalloc",
            "iterationneedsapi", "has_multi_index", "has_index", "dtypes",
            "ndim", "nop", "itersize", "finished"]

    # 遍历 attr 列表，使用 assert_raises 检查是否会抛出 AttributeError 异常
    for s in attr:
        assert_raises(AttributeError, delattr, it, s)


def test_iter_writable_attribute_deletion():
    # 创建一个 np.nditer 对象 it，迭代一个包含两个元素的数组
    it = np.nditer(np.ones(2))
    # 定义需要删除的属性列表 attr
    attr = ["multi_index", "index", "iterrange", "iterindex"]
    # 遍历 attr 列表，使用 assert_raises 检查是否会抛出 AttributeError 异常
    for s in attr:
        assert_raises(AttributeError, delattr, it, s)


def test_iter_element_deletion():
    # 创建一个 np.nditer 对象 it，迭代一个包含三个元素的数组
    it = np.nditer(np.ones(3))
    try:
        # 尝试删除迭代器中的一个元素和一个切片
        del it[1]
        del it[1:2]
    except TypeError:
        pass
    except Exception:
        raise AssertionError

def test_iter_allocated_array_dtypes():
    # 创建一个迭代器对象 `it`，用于迭代输入的数组和输出的数组
    it = np.nditer(([1, 3, 20], None), op_dtypes=[None, ('i4', (2,))])
    # 迭代器遍历开始，对每一对元素执行以下操作
    for a, b in it:
        # 设置输出数组 `b` 的第一个元素为 `a - 1`
        b[0] = a - 1
        # 设置输出数组 `b` 的第二个元素为 `a + 1`
        b[1] = a + 1
    # 断言检查迭代器的第二个操作数的值是否符合预期
    assert_equal(it.operands[1], [[0, 2], [2, 4], [19, 21]])

    # 使用带有 `op_axes` 参数为 -1 的 `np.nditer` 来检查相同的操作，此时更不敏感
    it = np.nditer(([[1, 3, 20]], None), op_dtypes=[None, ('i4', (2,))],
                   flags=["reduce_ok"], op_axes=[None, (-1, 0)])
    # 迭代器遍历开始，对每一对元素执行以下操作
    for a, b in it:
        # 设置输出数组 `b` 的第一个元素为 `a - 1`
        b[0] = a - 1
        # 设置输出数组 `b` 的第二个元素为 `a + 1`
        b[1] = a + 1
    # 断言检查迭代器的第二个操作数的值是否符合预期
    assert_equal(it.operands[1], [[0, 2], [2, 4], [19, 21]])

    # 确保标量同样可以正常工作
    it = np.nditer((10, 2, None), op_dtypes=[None, None, ('i4', (2, 2))])
    # 迭代器遍历开始，对每一组元素执行以下操作
    for a, b, c in it:
        # 设置输出数组 `c` 的第一个元素为 `a - b`
        c[0, 0] = a - b
        # 设置输出数组 `c` 的第二个元素为 `a + b`
        c[0, 1] = a + b
        # 设置输出数组 `c` 的第三个元素为 `a * b`
        c[1, 0] = a * b
        # 设置输出数组 `c` 的第四个元素为 `a / b`
        c[1, 1] = a / b
    # 断言检查迭代器的第三个操作数的值是否符合预期
    assert_equal(it.operands[2], [[8, 12], [20, 5]])
# Basic test function for iterating over 0-dimensional arrays using nditer.
def test_0d_iter():
    # Basic test for iteration of 0-d arrays:
    # Create an iterator over a list containing the values 2 and 3, with multi-indexing enabled and read-only mode.
    i = nditer([2, 3], ['multi_index'], [['readonly']]*2)
    # Assert that the iterator's dimensionality is 0.
    assert_equal(i.ndim, 0)
    # Retrieve the next item from the iterator and assert it matches (2, 3).
    assert_equal(next(i), (2, 3))
    # Assert that the multi_index attribute of the iterator is an empty tuple, as it is 0-dimensional.
    assert_equal(i.multi_index, ())
    # Assert that the current iteration index is 0.
    assert_equal(i.iterindex, 0)
    # Assert that attempting to retrieve the next item raises a StopIteration exception.
    assert_raises(StopIteration, next, i)
    # Test iterator reset functionality:
    # Reset the iterator to its initial state.
    i.reset()
    # Assert that retrieving the next item again yields (2, 3).
    assert_equal(next(i), (2, 3))
    # Assert that trying to retrieve another item raises StopIteration.
    assert_raises(StopIteration, next, i)

    # Test forcing the iterator to be 0-dimensional:
    # Create an iterator over a 1-dimensional array (arange(5)), with multi-indexing and read-only mode,
    # but force it to be treated as 0-dimensional by specifying op_axes=[()].
    i = nditer(np.arange(5), ['multi_index'], [['readonly']], op_axes=[()])
    # Assert that the iterator's dimensionality is 0.
    assert_equal(i.ndim, 0)
    # Assert that the length of the iterator is 1 (since it's 0-dimensional).
    assert_equal(len(i), 1)

    # Another way to force 0-dimensional iteration using itershape=().
    i = nditer(np.arange(5), ['multi_index'], [['readonly']],
               op_axes=[()], itershape=())
    # Assert that the iterator's dimensionality is 0.
    assert_equal(i.ndim, 0)
    # Assert that the length of the iterator is 1.
    assert_equal(len(i), 1)

    # Testing an invalid case where itershape alone is provided without op_axes.
    with assert_raises(ValueError):
        nditer(np.arange(5), ['multi_index'], [['readonly']], itershape=())

    # Test a more complex buffered casting case with structured datatypes.
    sdt = [('a', 'f4'), ('b', 'i8'), ('c', 'c8', (2, 3)), ('d', 'O')]
    a = np.array(0.5, dtype='f4')
    # Create an iterator over 'a', with buffering and allowing references, in read-only mode,
    # unsafe casting, and using the structured datatype defined by sdt.
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],
               casting='unsafe', op_dtypes=sdt)
    # Retrieve the next item from the iterator and assert the values of its fields.
    vals = next(i)
    assert_equal(vals['a'], 0.5)
    assert_equal(vals['b'], 0)
    assert_equal(vals['c'], [[(0.5)]*3]*2)
    assert_equal(vals['d'], 0.5)

# Test cases for object array iteration and cleanup scenarios.
def test_object_iter_cleanup():
    # See GitHub issue #18450.
    # Assert that attempting to perform arithmetic operations on an object array with None values raises a TypeError.
    assert_raises(TypeError, lambda: np.zeros((17000, 2), dtype='f4') * None)

    # More explicit test case triggering TypeError with object arrays.
    arr = np.arange(ncu.BUFSIZE * 10).reshape(10, -1).astype(str)
    oarr = arr.astype(object)
    oarr[:, -1] = None
    # Assert that attempting to add arrays with reversed column order raises a TypeError.
    assert_raises(TypeError, lambda: np.add(oarr[:, ::-1], arr[:, ::-1]))

    # Follow-up test for a bug related to TypeError fallthrough in GitHub issue #18450.
    class T:
        def __bool__(self):
            raise TypeError("Ambiguous")
    # Assert that attempting logical OR on an array containing instances of T raises a TypeError.
    assert_raises(TypeError, np.logical_or.reduce,
                             np.array([T(), T()], dtype='O'))

# Test cases for object array iteration and cleanup in reduction operations.
def test_object_iter_cleanup_reduce():
    # Similar to test_object_iter_cleanup but focusing on reduction cases.
    # See GitHub issue #18810.
    # Define an array that cannot be flattened.
    arr = np.array([[None, 1], [-1, -1], [None, 2], [-1, -1]])[::2]
    # Assert that attempting to sum this array raises a TypeError.
    with pytest.raises(TypeError):
        np.sum(arr)

# Parameterized test cases for large object array reduction scenarios.
@pytest.mark.parametrize("arr", [
        np.ones((8000, 4, 2), dtype=object)[:, ::2, :],
        np.ones((8000, 4, 2), dtype=object, order="F")[:, ::2, :],
        np.ones((8000, 4, 2), dtype=object)[:, ::2, :].copy("F")])
def test_object_iter_cleanup_large_reduce(arr):
    # Test cases for large object array reductions, parameterized with different array configurations.
    pass  # Placeholder for actual test implementation
    # 创建一个包含8000个元素且数据类型为np.intp的数组，所有元素初始化为1
    out = np.ones(8000, dtype=np.intp)
    # 对数组arr沿着第1和第2维度进行求和，强制使用dtype=object进行数据类型转换，并将结果存储到out数组中
    res = np.sum(arr, axis=(1, 2), dtype=object, out=out)
    # 断言计算结果res与包含8000个元素且所有元素为4的数组相等，数据类型为object
    assert_array_equal(res, np.full(8000, 4, dtype=object))
# 定义一个测试函数，用于测试当迭代器大小超出最大整数限制时的行为
def test_iter_too_large():
    # 计算迭代器的大小，确保不超过最大整数限制的 1/1024，以确保数组合法
    size = np.iinfo(np.intp).max // 1024
    # 使用 np.lib.stride_tricks.as_strided 创建一个新的数组，模拟超大迭代器
    arr = np.lib.stride_tricks.as_strided(np.zeros(1), (size,), (0,))
    # 断言应该抛出 ValueError 异常，因为迭代器超出了可接受范围
    assert_raises(ValueError, nditer, (arr, arr[:, None]))
    
    # 测试带有多索引的情况，当移除零维轴允许时可能更有趣
    assert_raises(ValueError, nditer,
                  (arr, arr[:, None]), flags=['multi_index'])


# 定义另一个测试函数，用于测试带有多索引的迭代器在超大情况下的行为
def test_iter_too_large_with_multiindex():
    # 当跟踪多索引时，错误消息会被延迟，这段代码检查延迟错误消息并通过移除轴来解决
    base_size = 2**10
    num = 1
    while base_size**num < np.iinfo(np.intp).max:
        num += 1

    shape_template = [1, 1] * num
    arrays = []
    for i in range(num):
        shape = shape_template[:]
        shape[i * 2] = 2**10
        arrays.append(np.empty(shape))
    arrays = tuple(arrays)

    # 现在数组太大无法广播。不同的模式测试了不同的 nditer 功能，有或无 GIL。
    for mode in range(6):
        with assert_raises(ValueError):
            _multiarray_tests.test_nditer_too_large(arrays, -1, mode)
    
    # 但是如果我们对 nditer 什么都不做，它可以被构造：
    _multiarray_tests.test_nditer_too_large(arrays, -1, 7)

    # 当移除一个轴时，事情应该重新工作（一半的时间）：
    for i in range(num):
        for mode in range(6):
            # 移除大小为 1024 的轴：
            _multiarray_tests.test_nditer_too_large(arrays, i*2, mode)
            # 移除大小为 1 的轴应该抛出 ValueError 异常：
            with assert_raises(ValueError):
                _multiarray_tests.test_nditer_too_large(arrays, i*2 + 1, mode)


# 定义测试写回行为的函数
def test_writebacks():
    a = np.arange(6, dtype='f4')
    au = a.byteswap()
    au = au.view(au.dtype.newbyteorder())
    assert_(a.dtype.byteorder != au.dtype.byteorder)
    
    # 使用 nditer 迭代器进行读写操作
    it = nditer(au, [], [['readwrite', 'updateifcopy']],
                        casting='equiv', op_dtypes=[np.dtype('f4')])
    with it:
        it.operands[0][:] = 100
    
    assert_equal(au, 100)
    
    # 再次进行相同操作，这次预期会引发错误
    it = nditer(au, [], [['readwrite', 'updateifcopy']],
                        casting='equiv', op_dtypes=[np.dtype('f4')])
    try:
        with it:
            assert_equal(au.flags.writeable, False)
            it.operands[0][:] = 0
            raise ValueError('exit context manager on exception')
    except:
        pass
    
    assert_equal(au, 0)
    assert_equal(au.flags.writeable, True)
    
    # 在上下文管理器外部无法重用迭代器 i
    assert_raises(ValueError, getattr, it, 'operands')

    it = nditer(au, [], [['readwrite', 'updateifcopy']],
                        casting='equiv', op_dtypes=[np.dtype('f4')])
    # 使用上下文管理器 it 进行迭代
    with it:
        # 获取迭代器 it 的第一个操作数
        x = it.operands[0]
        # 将 x 的所有元素设置为 6
        x[:] = 6
        # 断言 x 的标志位 writebackifcopy 已设置
        assert_(x.flags.writebackifcopy)
    # 断言 au 的值为 6
    assert_equal(au, 6)
    # 断言 x 的标志位 writebackifcopy 未设置
    assert_(not x.flags.writebackifcopy)
    # 将 x 的所有元素设置为 123，x.data 仍然有效
    x[:] = 123
    # 断言 au 的值为 6，但不再与 au 关联

    # 使用特定参数创建新的 nditer 迭代器 it
    it = nditer(au, [],
                 [['readwrite', 'updateifcopy']],
                 casting='equiv', op_dtypes=[np.dtype('f4')])
    # 可以重新进入迭代器
    with it:
        with it:
            # 迭代并将每个元素设置为 123
            for x in it:
                x[...] = 123

    # 使用特定参数再次创建新的 nditer 迭代器 it
    it = nditer(au, [],
                 [['readwrite', 'updateifcopy']],
                 casting='equiv', op_dtypes=[np.dtype('f4')])
    # 确保退出内部上下文管理器会关闭迭代器
    with it:
        with it:
            for x in it:
                x[...] = 123
        # 断言调用 getattr(it, 'operands') 会引发 ValueError 异常
        assert_raises(ValueError, getattr, it, 'operands')

    # 使用特定参数再次创建新的 nditer 迭代器 it
    it = nditer(au, [],
                 [['readwrite', 'updateifcopy']],
                 casting='equiv', op_dtypes=[np.dtype('f4')])
    # 删除数组 au
    del au
    # 尝试迭代已关闭的迭代器 it，不会导致崩溃
    with it:
        for x in it:
            x[...] = 123
    # 确保无法重新进入已关闭的迭代器
    enter = it.__enter__
    # 断言调用 it.__enter__() 会引发 RuntimeError 异常
    assert_raises(RuntimeError, enter)
def test_close_equivalent():
    ''' using a context amanger and using nditer.close are equivalent
    '''
    # 定义一个函数 add_close，用于执行数组的加法操作
    def add_close(x, y, out=None):
        addop = np.add
        # 创建一个迭代器 it，用于迭代数组 x, y 和输出数组 out
        it = np.nditer([x, y, out], [],
                    [['readonly'], ['readonly'], ['writeonly','allocate']])
        # 遍历迭代器并执行加法操作
        for (a, b, c) in it:
            addop(a, b, out=c)
        # 获取并返回输出数组
        ret = it.operands[2]
        # 手动关闭迭代器
        it.close()
        return ret

    # 定义一个函数 add_context，用于执行数组的加法操作
    def add_context(x, y, out=None):
        addop = np.add
        # 创建一个迭代器 it，用于迭代数组 x, y 和输出数组 out
        it = np.nditer([x, y, out], [],
                    [['readonly'], ['readonly'], ['writeonly','allocate']])
        # 使用上下文管理器处理迭代器
        with it:
            # 遍历迭代器并执行加法操作
            for (a, b, c) in it:
                addop(a, b, out=c)
            # 返回输出数组
            return it.operands[2]

    # 测试使用 add_close 函数进行迭代加法操作
    z = add_close(range(5), range(5))
    assert_equal(z, range(0, 10, 2))
    # 测试使用 add_context 函数进行迭代加法操作
    z = add_context(range(5), range(5))
    assert_equal(z, range(0, 10, 2))

# 测试在迭代器关闭后抛出异常的情况
def test_close_raises():
    # 创建一个迭代器 it，迭代数组 np.arange(3)
    it = np.nditer(np.arange(3))
    assert_equal(next(it), 0)
    # 显式关闭迭代器
    it.close()
    # 断言迭代器关闭后抛出 StopIteration 异常
    assert_raises(StopIteration, next, it)
    # 断言迭代器关闭后无法访问 operands 属性，抛出 ValueError 异常
    assert_raises(ValueError, getattr, it, 'operands')

# 测试迭代器关闭函数的参数
def test_close_parameters():
    # 创建一个迭代器 it，迭代数组 np.arange(3)
    it = np.nditer(np.arange(3))
    # 断言关闭迭代器时传递参数会抛出 TypeError 异常
    assert_raises(TypeError, it.close, 1)

# 使用 pytest.mark.skipif 装饰器标记，当条件满足时跳过测试
@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
# 测试在不关闭迭代器时发出警告
def test_warn_noclose():
    # 创建一个浮点类型的数组 a
    a = np.arange(6, dtype='f4')
    # 字节顺序转换并创建视图
    au = a.byteswap()
    au = au.view(au.dtype.newbyteorder())
    # 使用警告抑制上下文管理器记录 RuntimeWarning
    with suppress_warnings() as sup:
        sup.record(RuntimeWarning)
        # 创建一个迭代器 it，迭代数组 au
        it = np.nditer(au, [], [['readwrite', 'updateifcopy']],
                        casting='equiv', op_dtypes=[np.dtype('f4')])
        # 删除迭代器对象
        del it
        # 断言警告记录数量为 1
        assert len(sup.log) == 1

# 使用 pytest.mark.skipif 装饰器标记，当条件满足时跳过测试
@pytest.mark.skipif(sys.version_info[:2] == (3, 9) and sys.platform == "win32",
                    reason="Errors with Python 3.9 on Windows")
# 使用 pytest.mark.parametrize 装饰器定义多组参数化测试用例
@pytest.mark.parametrize(["in_dtype", "buf_dtype"],
        [("i", "O"), ("O", "i"),  # most simple cases
         ("i,O", "O,O"),  # structured partially only copying O
         ("O,i", "i,O"),  # structured casting to and from O
         ])
# 使用 pytest.mark.parametrize 装饰器定义参数化测试用例
@pytest.mark.parametrize("steps", [1, 2, 3])
# 测试部分迭代器清理过程中的引用计数泄漏情况
def test_partial_iteration_cleanup(in_dtype, buf_dtype, steps):
    """
    Checks for reference counting leaks during cleanup.  Using explicit
    reference counts lead to occasional false positives (at least in parallel
    test setups).  This test now should still test leaks correctly when
    run e.g. with pytest-valgrind or pytest-leaks
    """
    # 创建一个大数组 arr，用于测试
    value = 2**30 + 1  # just a random value that Python won't intern
    arr = np.full(int(ncu.BUFSIZE * 2.5), value).astype(in_dtype)

    # 创建一个迭代器 it，迭代数组 arr
    it = np.nditer(arr, op_dtypes=[np.dtype(buf_dtype)],
            flags=["buffered", "external_loop", "refs_ok"], casting="unsafe")
    # 迭代几步以测试部分迭代
    for step in range(steps):
        next(it)

    # 删除迭代器对象，测试清理过程
    del it
    # 使用 NumPy 中的 nditer 迭代器遍历数组 arr
    it = np.nditer(arr, op_dtypes=[np.dtype(buf_dtype)],
                   flags=["buffered", "external_loop", "refs_ok"], casting="unsafe")
    # 对迭代器进行指定次数的迭代，步数由 steps 变量指定
    for step in range(steps):
        it.iternext()
    
    # 清理操作：删除迭代器对象 it，尽管这一步不是必需的，但我们在此测试清理过程
    del it  # not necessary, but we test the cleanup
# 标记测试函数，如果没有引用计数（refcount）功能则跳过测试，给出原因为“Python lacks refcounts”
@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
# 参数化测试，指定不同的输入数据类型和缓冲区数据类型
@pytest.mark.parametrize(["in_dtype", "buf_dtype"],
         [("O", "i"),  # 最简单的情况
          ("O,i", "i,O"),  # 结构化转换为和从O的情况
          ])
# 定义测试函数，测试部分迭代中的错误处理
def test_partial_iteration_error(in_dtype, buf_dtype):
    # 设置一个整数值，依赖于Python的缓存（内存泄漏检查仍然会发现它）
    value = 123
    # 创建一个numpy数组，将其填充为指定值，并转换为指定的输入数据类型
    arr = np.full(int(ncu.BUFSIZE * 2.5), value).astype(in_dtype)
    # 如果输入数据类型为"O"，则将数组中某个位置设为None
    if in_dtype == "O":
        arr[int(ncu.BUFSIZE * 1.5)] = None
    else:
        arr[int(ncu.BUFSIZE * 1.5)]["f0"] = None

    # 获取value的引用计数
    count = sys.getrefcount(value)

    # 创建一个numpy迭代器对象，指定操作数据类型和迭代标志，并设置类型转换为不安全模式
    it = np.nditer(arr, op_dtypes=[np.dtype(buf_dtype)],
            flags=["buffered", "external_loop", "refs_ok"], casting="unsafe")
    # 使用pytest断言捕获特定类型的错误
    with pytest.raises(TypeError):
        # 手动逐个迭代，因为pytest.raises在for循环中可能会有问题
        next(it)
        next(it)  # 触发TypeError异常

    # 重置迭代器，再次使用iternext，预期也会触发TypeError异常
    it.reset()
    with pytest.raises(TypeError):
        it.iternext()
        it.iternext()

    # 断言value的引用计数没有变化
    assert count == sys.getrefcount(value)


# 定义测试函数，用于调试打印输出
def test_debug_print(capfd):
    """
    检查调试打印的输出是否符合预期。
    注意，迭代器转储的格式不稳定，这个测试主要是确保打印不会导致崩溃。

    目前使用子进程来避免处理C级别的printf。
    """
    # 期望的输出，移除所有地址和大小（因为它们会变化或者是平台相关）。
    expected = """
    ------ BEGIN ITERATOR DUMP ------
    | Iterator Address:
    | ItFlags: BUFFER REDUCE REUSE_REDUCE_LOOPS
    | NDim: 2
    | NOp: 2
    | IterSize: 50
    | IterStart: 0
    | IterEnd: 50
    | IterIndex: 0
    | Iterator SizeOf:
    | BufferData SizeOf:
    | AxisData SizeOf:
    |
    | Perm: 0 1
    | DTypes:
    | DTypes: dtype('float64') dtype('int32')
    | InitDataPtrs:
    | BaseOffsets: 0 0
    | Operands:
    | Operand DTypes: dtype('int64') dtype('float64')
    | OpItFlags:
    |   Flags[0]: READ CAST ALIGNED
    |   Flags[1]: READ WRITE CAST ALIGNED REDUCE
    |
    | BufferData:
    |   BufferSize: 50
    |   Size: 5
    |   BufIterEnd: 5
    |   REDUCE Pos: 0
    |   REDUCE OuterSize: 10
    |   REDUCE OuterDim: 1
    |   Strides: 8 4
    |   Ptrs:
    |   REDUCE Outer Strides: 40 0
    |   REDUCE Outer Ptrs:
    |   ReadTransferFn:
    |   ReadTransferData:
    |   WriteTransferFn:
    |   WriteTransferData:
    |   Buffers:
    |
    | AxisData[0]:
    |   Shape: 5
    |   Index: 0
    |   Strides: 16 8
    |   Ptrs:
    | AxisData[1]:
    |   Shape: 10
    |   Index: 0
    |   Strides: 80 0
    |   Ptrs:
    ------- END ITERATOR DUMP -------
    """.strip().splitlines()

    # 创建两个numpy数组用于测试
    arr1 = np.arange(100, dtype=np.int64).reshape(10, 10)[:, ::2]
    arr2 = np.arange(5.)
    # 使用 np.nditer() 创建一个迭代器，同时迭代 arr1 和 arr2 两个数组
    # 设置操作数的数据类型为 ["d", "i4"]，即第一个数组为双精度浮点数，第二个数组为32位整数
    # 设置不安全转换，允许缓冲区读取，可以减少操作
    it = np.nditer((arr1, arr2), op_dtypes=["d", "i4"], casting="unsafe",
                   flags=["reduce_ok", "buffered"],
                   op_flags=[["readonly"], ["readwrite"]])

    # 调试打印迭代器的信息
    it.debug_print()

    # 从 capfd 的输出中读取所有内容
    res = capfd.readouterr().out

    # 去除末尾的空白字符并按行分割结果
    res = res.strip().splitlines()

    # 断言实际输出的行数与期望输出的行数相等
    assert len(res) == len(expected)

    # 逐行比较实际输出和期望输出
    for res_line, expected_line in zip(res, expected):
        # 实际输出可能包含额外的指针信息，这些信息在示例输出中被剥离：
        # 断言实际输出的每一行都以去除空白后的期望输出行开头
        assert res_line.startswith(expected_line.strip())
```