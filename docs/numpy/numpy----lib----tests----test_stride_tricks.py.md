# `.\numpy\numpy\lib\tests\test_stride_tricks.py`

```py
# 导入必要的库
import numpy as np
# 从 numpy._core._rational_tests 模块中导入 rational 函数
from numpy._core._rational_tests import rational
# 从 numpy.testing 模块中导入多个断言函数
from numpy.testing import (
    assert_equal, assert_array_equal, assert_raises, assert_,
    assert_raises_regex, assert_warns,
    )
# 从 numpy.lib._stride_tricks_impl 模块中导入多个函数
from numpy.lib._stride_tricks_impl import (
    as_strided, broadcast_arrays, _broadcast_shape, broadcast_to,
    broadcast_shapes, sliding_window_view,
    )
# 导入 pytest 模块
import pytest


def assert_shapes_correct(input_shapes, expected_shape):
    # 对给定的输入形状列表进行广播，检查广播后的输出形状是否与期望形状相同。

    # 创建一个由零数组组成的列表，每个数组使用给定的形状
    inarrays = [np.zeros(s) for s in input_shapes]
    # 对列表中的数组进行广播，得到广播后的数组列表
    outarrays = broadcast_arrays(*inarrays)
    # 获取广播后每个数组的形状
    outshapes = [a.shape for a in outarrays]
    # 创建一个期望的形状列表，长度与输入形状列表相同
    expected = [expected_shape] * len(inarrays)
    # 断言广播后的形状与期望形状列表相同
    assert_equal(outshapes, expected)


def assert_incompatible_shapes_raise(input_shapes):
    # 对给定的（不兼容的）输入形状列表进行广播，检查是否引发 ValueError 异常。

    # 创建一个由零数组组成的列表，每个数组使用给定的形状
    inarrays = [np.zeros(s) for s in input_shapes]
    # 断言对列表中的数组进行广播时会引发 ValueError 异常
    assert_raises(ValueError, broadcast_arrays, *inarrays)


def assert_same_as_ufunc(shape0, shape1, transposed=False, flipped=False):
    # 对两个形状进行广播，检查数据布局是否与 ufunc 执行广播时的相同。

    # 创建一个形状为 shape0 的零数组
    x0 = np.zeros(shape0, dtype=int)
    # 根据 shape1 创建一个一维数组，并将其重塑为 shape1 形状的二维数组
    n = int(np.multiply.reduce(shape1))
    x1 = np.arange(n).reshape(shape1)
    # 如果 transposed 为 True，则对 x0 和 x1 进行转置
    if transposed:
        x0 = x0.T
        x1 = x1.T
    # 如果 flipped 为 True，则对 x0 和 x1 进行反转
    if flipped:
        x0 = x0[::-1]
        x1 = x1[::-1]
    # 使用 add ufunc 进行广播操作。由于我们在 x1 上加的是零数组 x0，因此结果应该与 x1 的广播视图完全相同。
    y = x0 + x1
    # 对 x0 和 x1 进行广播操作，并断言广播后的结果与 y 相同
    b0, b1 = broadcast_arrays(x0, x1)
    assert_array_equal(y, b1)


def test_same():
    # 测试相同输入的广播

    # 创建一个长度为 10 的一维数组 x 和 y
    x = np.arange(10)
    y = np.arange(10)
    # 对 x 和 y 进行广播操作
    bx, by = broadcast_arrays(x, y)
    # 断言广播后的 x 和 y 与原始 x 和 y 相同
    assert_array_equal(x, bx)
    assert_array_equal(y, by)


def test_broadcast_kwargs():
    # 测试使用非 'subok' 关键字参数调用 np.broadcast_arrays() 是否引发 TypeError 异常

    # 创建一个长度为 10 的一维数组 x 和 y
    x = np.arange(10)
    y = np.arange(10)

    # 使用 assert_raises_regex 上下文管理器断言调用 broadcast_arrays() 时使用非 'subok' 关键字参数会引发 TypeError 异常
    with assert_raises_regex(TypeError, 'got an unexpected keyword'):
        broadcast_arrays(x, y, dtype='float64')


def test_one_off():
    # 测试特殊的广播情况

    # 创建一个形状为 (1, 3) 的二维数组 x 和一个形状为 (3, 1) 的二维数组 y
    x = np.array([[1, 2, 3]])
    y = np.array([[1], [2], [3]])
    # 对 x 和 y 进行广播操作
    bx, by = broadcast_arrays(x, y)
    # 创建一个期望的二维数组 bx0 和 by0
    bx0 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    by0 = bx0.T
    # 断言广播后的 bx 和 by 与 bx0 和 by0 相同
    assert_array_equal(bx0, bx)
    assert_array_equal(by0, by)


def test_same_input_shapes():
    # 检查最终形状是否与输入形状相同

    # 定义一个包含各种输入形状的列表
    data = [
        (),
        (1,),
        (3,),
        (0, 1),
        (0, 3),
        (1, 0),
        (3, 0),
        (1, 3),
        (3, 1),
        (3, 3),
    ]
    # 遍历数据列表中的每个形状对象
    for shape in data:
        # 构建只包含当前形状对象的列表作为输入
        input_shapes = [shape]
        # 调用函数以确保输入的形状与给定的形状对象相匹配
        assert_shapes_correct(input_shapes, shape)
        # 构建包含两个当前形状对象的列表作为输入
        input_shapes2 = [shape, shape]
        # 调用函数以确保输入的形状与给定的形状对象相匹配
        assert_shapes_correct(input_shapes2, shape)
        # 构建包含三个当前形状对象的列表作为输入
        input_shapes3 = [shape, shape, shape]
        # 调用函数以确保输入的形状与给定的形状对象相匹配
        assert_shapes_correct(input_shapes3, shape)
def test_two_compatible_by_ones_input_shapes():
    # 检查两个不同输入形状，但长度相同且某些部分为1的情况，是否广播到正确的形状。

    data = [
        [[(1,), (3,)], (3,)],
        [[(1, 3), (3, 3)], (3, 3)],
        [[(3, 1), (3, 3)], (3, 3)],
        [[(1, 3), (3, 1)], (3, 3)],
        [[(1, 1), (3, 3)], (3, 3)],
        [[(1, 1), (1, 3)], (1, 3)],
        [[(1, 1), (3, 1)], (3, 1)],
        [[(1, 0), (0, 0)], (0, 0)],
        [[(0, 1), (0, 0)], (0, 0)],
        [[(1, 0), (0, 1)], (0, 0)],
        [[(1, 1), (0, 0)], (0, 0)],
        [[(1, 1), (1, 0)], (1, 0)],
        [[(1, 1), (0, 1)], (0, 1)],
    ]
    for input_shapes, expected_shape in data:
        assert_shapes_correct(input_shapes, expected_shape)
        # 反转输入形状，因为广播应该是对称的。
        assert_shapes_correct(input_shapes[::-1], expected_shape)


def test_two_compatible_by_prepending_ones_input_shapes():
    # 检查两个不同长度的输入形状，但以1开头的情况是否广播到正确的形状。

    data = [
        [[(), (3,)], (3,)],
        [[(3,), (3, 3)], (3, 3)],
        [[(3,), (3, 1)], (3, 3)],
        [[(1,), (3, 3)], (3, 3)],
        [[(), (3, 3)], (3, 3)],
        [[(1, 1), (3,)], (1, 3)],
        [[(1,), (3, 1)], (3, 1)],
        [[(1,), (1, 3)], (1, 3)],
        [[(), (1, 3)], (1, 3)],
        [[(), (3, 1)], (3, 1)],
        [[(), (0,)], (0,)],
        [[(0,), (0, 0)], (0, 0)],
        [[(0,), (0, 1)], (0, 0)],
        [[(1,), (0, 0)], (0, 0)],
        [[(), (0, 0)], (0, 0)],
        [[(1, 1), (0,)], (1, 0)],
        [[(1,), (0, 1)], (0, 1)],
        [[(1,), (1, 0)], (1, 0)],
        [[(), (1, 0)], (1, 0)],
        [[(), (0, 1)], (0, 1)],
    ]
    for input_shapes, expected_shape in data:
        assert_shapes_correct(input_shapes, expected_shape)
        # 反转输入形状，因为广播应该是对称的。
        assert_shapes_correct(input_shapes[::-1], expected_shape)


def test_incompatible_shapes_raise_valueerror():
    # 检查对于不兼容的形状是否引发 ValueError 异常。

    data = [
        [(3,), (4,)],
        [(2, 3), (2,)],
        [(3,), (3,), (4,)],
        [(1, 3, 4), (2, 3, 3)],
    ]
    for input_shapes in data:
        assert_incompatible_shapes_raise(input_shapes)
        # 反转输入形状，因为广播应该是对称的。
        assert_incompatible_shapes_raise(input_shapes[::-1])


def test_same_as_ufunc():
    # 检查数据布局是否与 ufunc 执行的操作相同。
    
    data = [
        [[(1,), (3,)], (3,)],  # 示例数据：包含两个元组作为输入形状，期望输出形状为 (3,)
        [[(1, 3), (3, 3)], (3, 3)],  # 示例数据：包含两个元组作为输入形状，期望输出形状为 (3, 3)
        [[(3, 1), (3, 3)], (3, 3)],  # 示例数据：包含两个元组作为输入形状，期望输出形状为 (3, 3)
        [[(1, 3), (3, 1)], (3, 3)],  # 示例数据：包含两个元组作为输入形状，期望输出形状为 (3, 3)
        [[(1, 1), (3, 3)], (3, 3)],  # 示例数据：包含两个元组作为输入形状，期望输出形状为 (3, 3)
        [[(1, 1), (1, 3)], (1, 3)],  # 示例数据：包含两个元组作为输入形状，期望输出形状为 (1, 3)
        [[(1, 1), (3, 1)], (3, 1)],  # 示例数据：包含两个元组作为输入形状，期望输出形状为 (3, 1)
        [[(1, 0), (0, 0)], (0, 0)],  # 示例数据：包含两个元组作为输入形状，期望输出形状为 (0, 0)
        [[(0, 1), (0, 0)], (0, 0)],  # 示例数据：包含两个元组作为输入形状，期望输出形状为 (0, 0)
        [[(1, 0), (0, 1)], (0, 0)],  # 示例数据：包含两个元组作为输入形状，期望输出形状为 (0, 0)
        [[(1, 1), (0, 0)], (0, 0)],  # 示例数据：包含两个元组作为输入形状，期望输出形状为 (0, 0)
        [[(1, 1), (1, 0)], (1, 0)],  # 示例数据：包含两个元组作为输入形状，期望输出形状为 (1, 0)
        [[(1, 1), (0, 1)], (0, 1)],  # 示例数据：包含两个元组作为输入形状，期望输出形状为 (0, 1)
        [[(), (3,)], (3,)],  # 示例数据：包含一个元组和一个标量作为输入形状，期望输出形状为 (3,)
        [[(3,), (3, 3)], (3, 3)],  # 示例数据：包含一个元组和一个向量作为输入形状，期望输出形状为 (3, 3)
        [[(3,), (3, 1)], (3, 3)],  # 示例数据：包含一个元组和一个向量作为输入形状，期望输出形状为 (3, 3)
        [[(1,), (3, 3)], (3, 3)],  # 示例数据：包含一个标量和一个向量作为输入形状，期望输出形状为 (3, 3)
        [[(), (3, 3)], (3, 3)],  # 示例数据：包含一个标量和一个矩阵作为输入形状，期望输出形状为 (3, 3)
        [[(1, 1), (3,)], (1, 3)],  # 示例数据：包含一个元组和一个标量作为输入形状，期望输出形状为 (1, 3)
        [[(1,), (3, 1)], (3, 1)],  # 示例数据：包含一个标量和一个向量作为输入形状，期望输出形状为 (3, 1)
        [[(1,), (1, 3)], (1, 3)],  # 示例数据：包含一个标量和一个矩阵作为输入形状，期望输出形状为 (1, 3)
        [[(), (1, 3)], (1, 3)],  # 示例数据：包含一个标量和一个矩阵作为输入形状，期望输出形状为 (1, 3)
        [[(), (3, 1)], (3, 1)],  # 示例数据：包含一个标量和一个向量作为输入形状，期望输出形状为 (3, 1)
        [[(), (0,)], (0,)],  # 示例数据：包含一个标量和一个向量作为输入形状，期望输出形状为 (0,)
        [[(0,), (0, 0)], (0, 0)],  # 示例数据：包含一个向量和一个标量作为输入形状，期望输出形状为 (0, 0)
        [[(0,), (0, 1)], (0, 0)],  # 示例数据：包含一个向量和一个向量作为输入形状，期望输出形状为 (0, 0)
        [[(1,), (0, 0)], (0, 0)],  # 示例数据：包含一个标量和一个标量作为输入形状，期望输出形状为 (0, 0)
        [[(), (0, 0)], (0, 0)],  # 示例数据：包含一个标量和一个标量作为输入形状，期望输出形状为 (0, 0)
        [[(1, 1), (0,)], (1, 0)],  # 示例数据：包含一个元组和一个标量作为输入形状，期望输出形状为 (1, 0)
        [[(1,), (0, 1)], (0, 1)],  # 示例数据：包含一个标量和一个向量作为输入形状，期望输出形状为 (0, 1)
        [[(1,), (1, 0)], (1, 0)],  # 示例数据：包含一个标量和一个标量作为输入形状，期望输出形状为 (1, 0)
        [[(), (1, 0)], (1, 0)],  # 示例数据：包含一个标量和一个标量作为输入形状，期望输出形状为 (1, 0)
        [[(), (0, 1)], (0, 1)],  # 示例数据：包含一个标量和一个向量作为输入形状，期望输出形状为 (0, 1)
    ]
    for input_shapes, expected_shape in data:
        assert_same_as_ufunc(input_shapes[0], input_shapes[1],
                             "Shapes: %s %s" % (input_shapes[0], input_shapes[1]))
        # 反转输入形状，因为广播应该是对称的。
        assert_same_as_ufunc(input_shapes[1], input_shapes[0])
        # 也尝试转置它们。
        assert_same_as_ufunc(input_shapes[0], input_shapes[1], True)
        # ... 并且对于非秩为0的输入，也进行翻转以测试负步长。
        if () not in input_shapes:
            assert_same_as_ufunc(input_shapes[0], input_shapes[1], False, True)
            assert_same_as_ufunc(input_shapes[0], input_shapes[1], True, True)
# 定义函数用于测试 broadcast_to 函数的成功情况
def test_broadcast_to_succeeds():
    # 定义测试数据，每个元素包含输入数组、目标形状和期望的输出数组
    data = [
        [np.array(0), (0,), np.array(0)],  # 输入数组为0维，目标形状为(0,)，期望输出为0
        [np.array(0), (1,), np.zeros(1)],  # 输入数组为0维，目标形状为(1,)，期望输出为[0.]
        [np.array(0), (3,), np.zeros(3)],  # 输入数组为0维，目标形状为(3,)，期望输出为[0. 0. 0.]
        [np.ones(1), (1,), np.ones(1)],    # 输入数组为1维[1.]，目标形状为(1,)，期望输出为[1.]
        [np.ones(1), (2,), np.ones(2)],    # 输入数组为1维[1.]，目标形状为(2,)，期望输出为[1. 1.]
        [np.ones(1), (1, 2, 3), np.ones((1, 2, 3))],  # 输入数组为1维[1.]，目标形状为(1, 2, 3)，期望输出为[[[1. 1. 1.] [1. 1. 1.]]]
        [np.arange(3), (3,), np.arange(3)],  # 输入数组为1维[0 1 2]，目标形状为(3,)，期望输出为[0 1 2]
        [np.arange(3), (1, 3), np.arange(3).reshape(1, -1)],  # 输入数组为1维[0 1 2]，目标形状为(1, 3)，期望输出为[[0 1 2]]
        [np.arange(3), (2, 3), np.array([[0, 1, 2], [0, 1, 2]])],  # 输入数组为1维[0 1 2]，目标形状为(2, 3)，期望输出为[[0 1 2] [0 1 2]]
        # 测试目标形状为非元组的情况
        [np.ones(0), 0, np.ones(0)],  # 输入数组为0维，目标形状为0，期望输出为空数组
        [np.ones(1), 1, np.ones(1)],  # 输入数组为1维[1.]，目标形状为1，期望输出为[1.]
        [np.ones(1), 2, np.ones(2)],  # 输入数组为1维[1.]，目标形状为2，期望输出为[1. 1.]
        # 下面这些大小为0的情况看似奇怪，但是复现了与 ufuncs 广播的行为（参见上面的 test_same_as_ufunc）
        [np.ones(1), (0,), np.ones(0)],  # 输入数组为1维[1.]，目标形状为(0,)，期望输出为空数组
        [np.ones((1, 2)), (0, 2), np.ones((0, 2))],  # 输入数组为2维[[1. 1.]]，目标形状为(0, 2)，期望输出为二维空数组
        [np.ones((2, 1)), (2, 0), np.ones((2, 0))],  # 输入数组为2维[[1.] [1.]]，目标形状为(2, 0)，期望输出为二维空数组
    ]
    # 遍历测试数据
    for input_array, shape, expected in data:
        # 调用 broadcast_to 函数得到实际输出
        actual = broadcast_to(input_array, shape)
        # 断言实际输出与期望输出相等
        assert_array_equal(expected, actual)


# 定义函数用于测试 broadcast_to 函数的异常情况
def test_broadcast_to_raises():
    # 定义测试数据，每个元素包含原始形状和目标形状
    data = [
        [(0,), ()],        # 原始形状为(0,)，目标形状为空元组，期望抛出 ValueError 异常
        [(1,), ()],        # 原始形状为(1,)，目标形状为空元组，期望抛出 ValueError 异常
        [(3,), ()],        # 原始形状为(3,)，目标形状为空元组，期望抛出 ValueError 异常
        [(3,), (1,)],      # 原始形状为(3,)，目标形状为(1,)，期望抛出 ValueError 异常
        [(3,), (2,)],      # 原始形状为(3,)，目标形状为(2,)，期望抛出 ValueError 异常
        [(3,), (4,)],      # 原始形状为(3,)，目标形状为(4,)，期望抛出 ValueError 异常
        [(1, 2), (2, 1)],  # 原始形状为(1, 2)，目标形状为(2, 1)，期望抛出 ValueError 异常
        [(1, 1), (1,)],    # 原始形状为(1, 1)，目标形状为(1,)，期望抛出 ValueError 异常
        [(1,), -1],        # 原始形状为(1,)，目标形状为-1，期望抛出 ValueError 异常
        [(1,), (-1,)],     # 原始形状为(1,)，目标形状为(-1,)，期望抛出 ValueError 异常
        [(1, 2), (-1, 2)],  # 原始形状为(1, 2)，目标形状为(-1, 2)，期望抛出 ValueError 异常
    ]
    # 遍历测试数据
    for orig_shape, target_shape in data:
        # 创建具有原始形状的零数组
        arr = np.zeros(orig_shape)
        # 使用 lambda 表达式和 assert_raises 断言抛出 ValueError 异常
        assert_raises(ValueError, lambda: broadcast_to(arr, target_shape))


# 定义函数用于测试 broadcast_shapes 函数
def test_broadcast_shape():
    # 测试 _broadcast_shape 内部函数
    # _broadcast_shape 已经通过 broadcast_arrays 间接测试
    # _broadcast_shape 也通过公共 broadcast_shapes 函数测试
    assert_equal(_broadcast_shape(), ())  # 确保没有参数时返回空元组
    assert_equal(_broadcast_shape([1, 2]), (2,))  # 测试一维数组[1, 2]，期望返回元组(2,)
    assert_equal(_broadcast_shape(np.ones((1, 1))), (1, 1))  # 测试形状为(1, 1)的全一数组，期望返回元组(1, 1)
    assert_equal(_broadcast_shape(np.ones((1, 1)), np.ones((3, 4))), (3, 4))  # 测试两个不同形状的全一数组，期望返回元组(3, 4)
    assert_equal(_broadcast_shape(*([np.ones((1, 2))] * 32)), (1, 2))  # 测试32个形状为(1, 2)的全一数组，期望返回元组(1, 2)
    assert_equal(_broadcast_shape(*([np.ones((1, 2))] * 100)), (1, 2))  # 测试100个形状为(1, 2)的全一数组，期望返回元组(1, 2)

    # gh-5862 的回归测试
    assert_equal(_broadcast_shape(*([np.ones(2)] * 32 + [1])), (2,))  # 测试32个形状为(2,)的全一数组和一个形状为(1,)的数组，期
    data = [
        [[], ()],                     # 空列表作为输入，预期输出是空元组
        [[()], ()],                   # 包含一个空元组作为输入，预期输出是空元组
        [[(7,)], (7,)],               # 包含一个包含一个元素的元组作为输入，预期输出是包含一个元素的元组
        [[(1, 2), (2,)], (1, 2)],     # 包含两个元组作为输入，预期输出是第一个元组
        [[(1, 1)], (1, 1)],           # 包含一个包含相同元素的元组作为输入，预期输出是包含相同元素的元组
        [[(1, 1), (3, 4)], (3, 4)],   # 包含两个不同元素的元组作为输入，预期输出是第二个元组
        [[(6, 7), (5, 6, 1), (7,), (5, 1, 7)], (5, 6, 7)],  # 多个不同长度元组的输入，预期输出是长度最长的元组
        [[(5, 6, 1)], (5, 6, 1)],     # 包含一个包含三个元素的元组作为输入，预期输出是包含三个元素的元组
        [[(1, 3), (3, 1)], (3, 3)],   # 包含两个不同顺序的元组作为输入，预期输出是第二个元组
        [[(1, 0), (0, 0)], (0, 0)],   # 包含两个不同元素的元组作为输入，预期输出是包含两个零的元组
        [[(0, 1), (0, 0)], (0, 0)],   # 包含两个不同元素的元组作为输入，预期输出是包含两个零的元组
        [[(1, 0), (0, 1)], (0, 0)],   # 包含两个不同元素的元组作为输入，预期输出是包含两个零的元组
        [[(1, 1), (0, 0)], (0, 0)],   # 包含两个不同元素的元组作为输入，预期输出是包含两个零的元组
        [[(1, 1), (1, 0)], (1, 0)],   # 包含两个不同元素的元组作为输入，预期输出是包含一个和零的元组
        [[(1, 1), (0, 1)], (0, 1)],   # 包含两个不同元素的元组作为输入，预期输出是包含一个和一的元组
        [[(), (0,)], (0,)],           # 包含一个空元组和一个非空元组作为输入，预期输出是非空元组
        [[(0,), (0, 0)], (0, 0)],     # 包含一个单元素元组和一个包含两个元素的元组作为输入，预期输出是包含两个零的元组
        [[(0,), (0, 1)], (0, 0)],     # 包含一个单元素元组和一个包含两个元素的元组作为输入，预期输出是包含两个零的元组
        [[(1,), (0, 0)], (0, 0)],     # 包含一个单元素元组和一个包含两个元素的元组作为输入，预期输出是包含两个零的元组
        [[(), (0, 0)], (0, 0)],       # 包含一个空元组和一个包含两个元素的元组作为输入，预期输出是包含两个零的元组
        [[(1, 1), (0,)], (1, 0)],     # 包含一个包含两个元素的元组和一个单元素元组作为输入，预期输出是包含一个和零的元组
        [[(1,), (0, 1)], (0, 1)],     # 包含一个单元素元组和一个包含两个元素的元组作为输入，预期输出是包含一个和一的元组
        [[(1,), (1, 0)], (1, 0)],     # 包含一个单元素元组和一个包含两个元素的元组作为输入，预期输出是包含一个和一的元组
        [[(), (1, 0)], (1, 0)],       # 包含一个空元组和一个包含两个元素的元组作为输入，预期输出是包含一个和一的元组
        [[(), (0, 1)], (0, 1)],       # 包含一个空元组和一个包含两个元素的元组作为输入，预期输出是包含一个和一的元组
        [[(1,), (3,)], (3,)],         # 包含一个单元素元组和一个包含一个元素的元组作为输入，预期输出是包含一个元素的元组
        [[2, (3, 2)], (3, 2)],        # 包含一个整数和一个包含两个元素的元组作为输入，预期输出是包含两个元素的元组
    ]

    for input_shapes, target_shape in data:
        assert_equal(broadcast_shapes(*input_shapes), target_shape)

    assert_equal(broadcast_shapes(*([(1, 2)] * 32)), (1, 2))      # 对包含32个相同元组的列表进行广播，预期输出是这个元组
    assert_equal(broadcast_shapes(*([(1, 2)] * 100)), (1, 2))    # 对包含100个相同元组的列表进行广播，预期输出是这个元组

    # 用于测试 gh-5862 的回归测试
    assert_equal(broadcast_shapes(*([(2,)] * 32)), (2,))         # 对包含32个包含一个元素的元组的列表进行广播，预期输出是包含一个元素的元组
def test_broadcast_shapes_raises():
    # 测试公共函数 broadcast_shapes
    # 定义测试数据，包含多组不同形状的输入
    data = [
        [(3,), (4,)],           # 输入形状为 (3,) 和 (4,)
        [(2, 3), (2,)],         # 输入形状为 (2, 3) 和 (2,)
        [(3,), (3,), (4,)],     # 输入形状为 (3,)、(3,) 和 (4,)
        [(1, 3, 4), (2, 3, 3)],  # 输入形状为 (1, 3, 4) 和 (2, 3, 3)
        [(1, 2), (3, 1), (3, 2), (10, 5)],  # 多组输入形状
        [2, (2, 3)],            # 输入形状为 2 和 (2, 3)
    ]
    
    # 遍历测试数据，对每组数据调用 broadcast_shapes 应当抛出 ValueError 异常
    for input_shapes in data:
        assert_raises(ValueError, lambda: broadcast_shapes(*input_shapes))
    
    # 构造一个包含多个相同形状的参数的列表，长度为 64
    bad_args = [(2,)] * 32 + [(3,)] * 32
    
    # 调用 broadcast_shapes 应当抛出 ValueError 异常
    assert_raises(ValueError, lambda: broadcast_shapes(*bad_args))


def test_as_strided():
    # 创建一个包含单个元素的 numpy 数组 a
    a = np.array([None])
    # 调用 as_strided 函数获取数组 a 的视图
    a_view = as_strided(a)
    # 预期的结果是包含单个 None 的 numpy 数组
    expected = np.array([None])
    # 断言 a_view 等于预期结果 expected
    assert_array_equal(a_view, expected)
    
    # 创建一个包含元素 [1, 2, 3, 4] 的 numpy 数组 a
    a = np.array([1, 2, 3, 4])
    # 使用 as_strided 函数创建形状为 (2,)、步长为 (2 * a.itemsize,) 的数组视图 a_view
    a_view = as_strided(a, shape=(2,), strides=(2 * a.itemsize,))
    # 预期的结果是包含元素 [1, 3] 的 numpy 数组
    expected = np.array([1, 3])
    # 断言 a_view 等于预期结果 expected
    assert_array_equal(a_view, expected)
    
    # 创建一个包含元素 [1, 2, 3, 4] 的 numpy 数组 a
    a = np.array([1, 2, 3, 4])
    # 使用 as_strided 函数创建形状为 (3, 4)、步长为 (0, 1 * a.itemsize) 的数组视图 a_view
    a_view = as_strided(a, shape=(3, 4), strides=(0, 1 * a.itemsize))
    # 预期的结果是包含元素 [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]] 的 numpy 数组
    expected = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    # 断言 a_view 等于预期结果 expected
    assert_array_equal(a_view, expected)
    
    # 回归测试 gh-5081
    # 创建一个自定义结构的 dtype
    dt = np.dtype([('num', 'i4'), ('obj', 'O')])
    # 创建一个 dtype 为 dt、形状为 (4,) 的空数组 a
    a = np.empty((4,), dtype=dt)
    # 将 a['num'] 的值设为 [1, 2, 3, 4]
    a['num'] = np.arange(1, 5)
    # 使用 as_strided 函数创建形状为 (3, 4)、步长为 (0, a.itemsize) 的数组视图 a_view
    a_view = as_strided(a, shape=(3, 4), strides=(0, a.itemsize))
    # 预期的 num 数组是 [[1, 2, 3, 4]] 的重复三次，obj 数组是包含四个 None 的列表，组成的列表
    expected_num = [[1, 2, 3, 4]] * 3
    expected_obj = [[None] * 4] * 3
    # 断言 a_view 的 dtype 等于预期的 dt
    assert_equal(a_view.dtype, dt)
    # 断言 a_view['num'] 等于预期结果 expected_num
    assert_array_equal(expected_num, a_view['num'])
    # 断言 a_view['obj'] 等于预期结果 expected_obj
    assert_array_equal(expected_obj, a_view['obj'])
    
    # 确保没有字段的空类型保持不变
    # 创建一个 dtype 为 'V4'、形状为 (4,) 的空数组 a
    a = np.empty((4,), dtype='V4')
    # 使用 as_strided 函数创建形状为 (3, 4)、步长为 (0, a.itemsize) 的数组视图 a_view
    a_view = as_strided(a, shape=(3, 4), strides=(0, a.itemsize))
    # 断言 a 的 dtype 等于 a_view 的 dtype
    assert_equal(a.dtype, a_view.dtype)
    
    # 确保唯一可能失败的类型被正确处理
    # 创建一个自定义结构的 dtype，只包含一个 'V4' 类型的字段
    dt = np.dtype({'names': [''], 'formats': ['V4']})
    # 创建一个 dtype 为 dt、形状为 (4,) 的空数组 a
    a = np.empty((4,), dtype=dt)
    # 使用 as_strided 函数创建形状为 (3, 4)、步长为 (0, a.itemsize) 的数组视图 a_view
    a_view = as_strided(a, shape=(3, 4), strides=(0, a.itemsize))
    # 断言 a 的 dtype 等于 a_view 的 dtype
    assert_equal(a.dtype, a_view.dtype)
    
    # 自定义 dtype 不应该丢失 (gh-9161)
    # 创建一个有理数类的实例列表 r，包含 [rational(0), rational(1), rational(2), rational(3)]
    r = [rational(i) for i in range(4)]
    # 创建一个 dtype 为 rational 的数组 a，包含 r 中的有理数实例
    a = np.array(r, dtype=rational)
    # 使用 as_strided 函数创建形状为 (3, 4)、步长为 (0, a.itemsize) 的数组视图 a_view
    a_view = as_strided(a, shape=(3, 4), strides=(0, a.itemsize))
    # 断言 a 的 dtype 等于 a_view 的 dtype
    assert_equal(a.dtype, a_view.dtype)
    # 断言 a_view 中包含 r 的重复三次的数组
    assert_array_equal([r] * 3, a_view)


class TestSlidingWindowView:
    def test_1d(self):
        # 创建一个包含元素 [0, 1, 2, 3, 4] 的 numpy 数组 arr
        arr = np.arange(5)
        # 使用 sliding_window_view 函数创建 arr 的滑动窗口视图 arr_view，窗口大小为 2
        arr_view = sliding_window_view(arr, 2)
        # 预期的结果是包含元素 [[0, 1], [1, 2], [2, 3], [3, 4]] 的 numpy 数组
        expected = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
        # 断言 arr_view 等于预期结果 expected
        assert_array_equal(arr_view, expected)
    # 定义一个测试方法，测试二维情况下的滑动窗口视图
    def test_2d(self):
        # 创建两个二维的坐标网格，i 为行，j 为列，形状为 (3, 4)
        i, j = np.ogrid[:3, :4]
        # 根据公式 arr = 10*i + j 创建一个二维数组 arr
        arr = 10*i + j
        # 定义窗口的形状为 (2, 2)
        shape = (2, 2)
        # 使用 sliding_window_view 函数创建 arr 的滑动窗口视图
        arr_view = sliding_window_view(arr, shape)
        # 期望的结果数组，包含所有可能的滑动窗口视图
        expected = np.array([[[[0, 1], [10, 11]],
                              [[1, 2], [11, 12]],
                              [[2, 3], [12, 13]]],
                             [[[10, 11], [20, 21]],
                              [[11, 12], [21, 22]],
                              [[12, 13], [22, 23]]]])
        # 断言 arr_view 与期望结果 expected 相等
        assert_array_equal(arr_view, expected)

    # 定义一个测试方法，测试带有指定轴的二维滑动窗口视图
    def test_2d_with_axis(self):
        # 创建两个二维的坐标网格，i 为行，j 为列，形状为 (3, 4)
        i, j = np.ogrid[:3, :4]
        # 根据公式 arr = 10*i + j 创建一个二维数组 arr
        arr = 10*i + j
        # 使用 sliding_window_view 函数创建 arr 的滑动窗口视图，指定窗口形状为 3，轴为 0
        arr_view = sliding_window_view(arr, 3, 0)
        # 期望的结果数组，包含按照指定轴生成的滑动窗口视图
        expected = np.array([[[0, 10, 20],
                              [1, 11, 21],
                              [2, 12, 22],
                              [3, 13, 23]]])
        # 断言 arr_view 与期望结果 expected 相等
        assert_array_equal(arr_view, expected)

    # 定义一个测试方法，测试带有重复轴的二维滑动窗口视图
    def test_2d_repeated_axis(self):
        # 创建两个二维的坐标网格，i 为行，j 为列，形状为 (3, 4)
        i, j = np.ogrid[:3, :4]
        # 根据公式 arr = 10*i + j 创建一个二维数组 arr
        arr = 10*i + j
        # 使用 sliding_window_view 函数创建 arr 的滑动窗口视图，指定窗口形状为 (2, 3)，轴为 (1, 1)
        arr_view = sliding_window_view(arr, (2, 3), (1, 1))
        # 期望的结果数组，包含按照指定轴生成的滑动窗口视图
        expected = np.array([[[[0, 1, 2],
                               [1, 2, 3]]],
                             [[[10, 11, 12],
                               [11, 12, 13]]],
                             [[[20, 21, 22],
                               [21, 22, 23]]]])
        # 断言 arr_view 与期望结果 expected 相等
        assert_array_equal(arr_view, expected)

    # 定义一个测试方法，测试在不指定轴的情况下的二维滑动窗口视图
    def test_2d_without_axis(self):
        # 创建两个二维的坐标网格，i 为行，j 为列，形状为 (4, 4)
        i, j = np.ogrid[:4, :4]
        # 根据公式 arr = 10*i + j 创建一个二维数组 arr
        arr = 10*i + j
        # 定义窗口的形状为 (2, 3)
        shape = (2, 3)
        # 使用 sliding_window_view 函数创建 arr 的滑动窗口视图
        arr_view = sliding_window_view(arr, shape)
        # 期望的结果数组，包含所有可能的滑动窗口视图
        expected = np.array([[[[0, 1, 2], [10, 11, 12]],
                              [[1, 2, 3], [11, 12, 13]]],
                             [[[10, 11, 12], [20, 21, 22]],
                              [[11, 12, 13], [21, 22, 23]]],
                             [[[20, 21, 22], [30, 31, 32]],
                              [[21, 22, 23], [31, 32, 33]]]])
        # 断言 arr_view 与期望结果 expected 相等
        assert_array_equal(arr_view, expected)

    # 定义一个测试方法，测试各种错误情况下的异常处理
    def test_errors(self):
        # 创建两个二维的坐标网格，i 为行，j 为列，形状为 (4, 4)
        i, j = np.ogrid[:4, :4]
        # 根据公式 arr = 10*i + j 创建一个二维数组 arr
        arr = 10*i + j
        # 测试 ValueError 异常，窗口形状包含负值
        with pytest.raises(ValueError, match='cannot contain negative values'):
            sliding_window_view(arr, (-1, 3))
        # 测试 ValueError 异常，未为所有维度提供窗口形状
        with pytest.raises(
                ValueError,
                match='must provide window_shape for all dimensions of `x`'):
            sliding_window_view(arr, (1,))
        # 测试 ValueError 异常，窗口形状与轴的长度不匹配
        with pytest.raises(
                ValueError,
                match='Must provide matching length window_shape and axis'):
            sliding_window_view(arr, (1, 3, 4), axis=(0, 1))
        # 测试 ValueError 异常，窗口形状大于输入数组的尺寸
        with pytest.raises(
                ValueError,
                match='window shape cannot be larger than input array'):
            sliding_window_view(arr, (5, 5))
    # 定义一个名为 test_writeable 的测试方法，用于测试 sliding_window_view 函数的 writeable 参数
    def test_writeable(self):
        # 创建一个包含 0 到 4 的 NumPy 数组
        arr = np.arange(5)
        # 使用 sliding_window_view 函数创建一个窗口视图，窗口大小为 2，设置为不可写
        view = sliding_window_view(arr, 2, writeable=False)
        # 断言视图不可写
        assert_(not view.flags.writeable)
        # 使用 pytest 检查写入不可写视图的操作是否引发 ValueError 异常，并检查异常消息
        with pytest.raises(
                ValueError,
                match='assignment destination is read-only'):
            view[0, 0] = 3
        # 使用 sliding_window_view 函数创建一个窗口视图，窗口大小为 2，设置为可写
        view = sliding_window_view(arr, 2, writeable=True)
        # 断言视图可写
        assert_(view.flags.writeable)
        # 修改可写视图的元素
        view[0, 1] = 3
        # 断言原始数组被正确修改
        assert_array_equal(arr, np.array([0, 3, 2, 3, 4]))

    # 定义一个名为 test_subok 的测试方法，用于测试 sliding_window_view 函数的 subok 参数
    def test_subok(self):
        # 定义一个继承自 np.ndarray 的子类 MyArray
        class MyArray(np.ndarray):
            pass

        # 创建一个包含 0 到 4 的 NumPy 数组，并将其视图转换为 MyArray 类型
        arr = np.arange(5).view(MyArray)
        # 断言不使用 subok 参数时，生成的窗口视图不是 MyArray 类型
        assert_(not isinstance(sliding_window_view(arr, 2,
                                                   subok=False),
                               MyArray))
        # 断言使用 subok 参数时，生成的窗口视图是 MyArray 类型
        assert_(isinstance(sliding_window_view(arr, 2, subok=True), MyArray))
        # 默认行为下，断言生成的窗口视图不是 MyArray 类型
        assert_(not isinstance(sliding_window_view(arr, 2), MyArray))
def as_strided_writeable():
    # 创建一个长度为10的全为1的 NumPy 数组
    arr = np.ones(10)
    # 创建一个 arr 的视图，但设置为不可写
    view = as_strided(arr, writeable=False)
    # 断言视图的可写标志为 False
    assert_(not view.flags.writeable)

    # 检查可写性是否正常：
    view = as_strided(arr, writeable=True)
    # 断言视图的可写标志为 True
    assert_(view.flags.writeable)
    # 修改视图中的所有元素为3
    view[...] = 3
    # 断言原始数组 arr 的所有元素都被修改为3
    assert_array_equal(arr, np.full_like(arr, 3))

    # 测试只读模式下的情况：
    arr.flags.writeable = False
    view = as_strided(arr, writeable=False)
    # 创建一个 arr 的视图，此时设置为可写
    view = as_strided(arr, writeable=True)
    # 断言视图的可写标志为 False
    assert_(not view.flags.writeable)


class VerySimpleSubClass(np.ndarray):
    def __new__(cls, *args, **kwargs):
        # 创建一个 np.array，并将其视图化为当前类的实例
        return np.array(*args, subok=True, **kwargs).view(cls)


class SimpleSubClass(VerySimpleSubClass):
    def __new__(cls, *args, **kwargs):
        # 创建一个 np.array，并将其视图化为当前类的实例
        self = np.array(*args, subok=True, **kwargs).view(cls)
        # 为实例添加附加信息
        self.info = 'simple'
        return self

    def __array_finalize__(self, obj):
        # 如果 obj 中有 'info' 属性，则将其赋给当前实例的 'info' 属性
        self.info = getattr(obj, 'info', '') + ' finalized'


def test_subclasses():
    # 测试只有当 subok=True 时，子类才能被保留
    a = VerySimpleSubClass([1, 2, 3, 4])
    assert_(type(a) is VerySimpleSubClass)
    # 创建一个 a 的视图，但不指定类型，应返回基本的 ndarray 类型
    a_view = as_strided(a, shape=(2,), strides=(2 * a.itemsize,))
    assert_(type(a_view) is np.ndarray)
    # 创建一个 a 的视图，并指定 subok=True，应返回子类 VerySimpleSubClass 的类型
    a_view = as_strided(a, shape=(2,), strides=(2 * a.itemsize,), subok=True)
    assert_(type(a_view) is VerySimpleSubClass)
    
    # 测试如果子类定义了 __array_finalize__ 方法，它会被调用
    a = SimpleSubClass([1, 2, 3, 4])
    # 创建一个 a 的视图，并指定 subok=True，应返回子类 SimpleSubClass 的类型
    a_view = as_strided(a, shape=(2,), strides=(2 * a.itemsize,), subok=True)
    assert_(type(a_view) is SimpleSubClass)
    assert_(a_view.info == 'simple finalized')

    # 类似的测试用于 broadcast_arrays
    b = np.arange(len(a)).reshape(-1, 1)
    # 对 a 和 b 进行 broadcast，预期返回的是普通的 ndarray 类型
    a_view, b_view = broadcast_arrays(a, b)
    assert_(type(a_view) is np.ndarray)
    assert_(type(b_view) is np.ndarray)
    assert_(a_view.shape == b_view.shape)
    # 对 a 和 b 进行 broadcast，指定 subok=True，预期返回的是子类 SimpleSubClass 类型
    a_view, b_view = broadcast_arrays(a, b, subok=True)
    assert_(type(a_view) is SimpleSubClass)
    assert_(a_view.info == 'simple finalized')
    assert_(type(b_view) is np.ndarray)
    assert_(a_view.shape == b_view.shape)

    # 还有对 broadcast_to 的测试
    shape = (2, 4)
    # 对 a 进行 broadcast 到指定 shape，预期返回的是普通的 ndarray 类型
    a_view = broadcast_to(a, shape)
    assert_(type(a_view) is np.ndarray)
    assert_(a_view.shape == shape)
    # 对 a 进行 broadcast 到指定 shape，指定 subok=True，预期返回的是子类 SimpleSubClass 类型
    a_view = broadcast_to(a, shape, subok=True)
    assert_(type(a_view) is SimpleSubClass)
    assert_(a_view.info == 'simple finalized')
    assert_(a_view.shape == shape)


def test_writeable():
    # broadcast_to 应返回一个只读数组
    original = np.array([1, 2, 3])
    result = broadcast_to(original, (2, 3))
    assert_equal(result.flags.writeable, False)
    assert_raises(ValueError, result.__setitem__, slice(None), 0)

    # 但 broadcast_arrays 的结果需要是可写的，以保持向后兼容性
    test_cases = [((False,), broadcast_arrays(original,)),
                  ((True, False), broadcast_arrays(0, original))]
    for is_broadcast, results in test_cases:
        for array_is_broadcast, result in zip(is_broadcast, results):
            # 遍历测试用例，每个测试用例包括是否广播和结果列表
            # 如果数组被广播，则执行以下操作：
            if array_is_broadcast:
                # 发出未来版本警告，即将改为 False
                with assert_warns(FutureWarning):
                    # 检查结果的可写标志是否为 True
                    assert_equal(result.flags.writeable, True)
                # 发出弃用警告
                with assert_warns(DeprecationWarning):
                    # 将结果数组全部设为 0
                    result[:] = 0
                # 警告未发出，写入数组会重置可写状态为 True
                assert_equal(result.flags.writeable, True)
            else:
                # 没有警告：
                # 检查结果的可写标志是否为 True
                assert_equal(result.flags.writeable, True)

    for results in [broadcast_arrays(original),
                    broadcast_arrays(0, original)]:
        for result in results:
            # 重置 warn_on_write 弃用警告
            result.flags.writeable = True
            # 检查：没有警告被发出
            assert_equal(result.flags.writeable, True)
            # 将结果数组全部设为 0
            result[:] = 0

    # 保持只读输入的只读状态
    original.flags.writeable = False
    # 使用广播数组函数返回的结果
    _, result = broadcast_arrays(0, original)
    # 检查结果的可写标志是否为 False
    assert_equal(result.flags.writeable, False)

    # GH6491 的回归测试
    # 创建一个形状为 (2,)，步幅为 [0] 的数组
    shape = (2,)
    strides = [0]
    tricky_array = as_strided(np.array(0), shape, strides)
    # 创建一个形状为 (1,) 的零数组
    other = np.zeros((1,))
    # 执行广播数组操作
    first, second = broadcast_arrays(tricky_array, other)
    # 检查第一个和第二个数组的形状是否相同
    assert_(first.shape == second.shape)
def test_writeable_memoryview():
    # 创建一个原始的 NumPy 数组，包含整数 [1, 2, 3]
    original = np.array([1, 2, 3])

    # 定义测试用例列表，每个元素是一个元组，包含是否进行广播的布尔值和广播后的结果
    test_cases = [((False, ), broadcast_arrays(original,)),
                  ((True, False), broadcast_arrays(0, original))]
    # 遍历测试用例
    for is_broadcast, results in test_cases:
        # 遍历每个结果和对应的是否广播标志
        for array_is_broadcast, result in zip(is_broadcast, results):
            # 如果数组是广播的，断言结果的 memoryview 是只读的
            if array_is_broadcast:
                # 此处在未来版本中会更改为 False
                assert memoryview(result).readonly
            else:
                # 如果数组不是广播的，断言结果的 memoryview 不是只读的
                assert not memoryview(result).readonly


def test_reference_types():
    # 创建一个包含单个字符 'a' 的对象数组
    input_array = np.array('a', dtype=object)
    # 创建预期的对象数组，包含三个 'a' 字符串
    expected = np.array(['a'] * 3, dtype=object)
    # 使用 broadcast_to 函数将输入数组广播到形状为 (3,) 的数组
    actual = broadcast_to(input_array, (3,))
    # 断言广播后的数组与预期结果相等
    assert_array_equal(expected, actual)

    # 使用 broadcast_arrays 函数广播输入数组和一个包含三个 1 的数组
    actual, _ = broadcast_arrays(input_array, np.ones(3))
    # 断言广播后的数组与预期结果相等
    assert_array_equal(expected, actual)
```