# `D:\src\scipysrc\scipy\scipy\ndimage\tests\test_measurements.py`

```
import os.path  # 导入处理文件路径的模块

import numpy as np  # 导入NumPy库
from numpy.testing import (  # 导入NumPy测试模块的多个函数
    assert_,  # 断言函数
    assert_allclose,  # 断言数组接近
    assert_almost_equal,  # 断言数组近似相等
    assert_array_almost_equal,  # 断言数组近似相等
    assert_array_equal,  # 断言数组完全相等
    assert_equal,  # 断言对象相等
    suppress_warnings,  # 抑制警告
)
import pytest  # 导入pytest测试框架
from pytest import raises as assert_raises  # 导入pytest的raises别名为assert_raises

import scipy.ndimage as ndimage  # 导入SciPy的图像处理模块

from . import types  # 从当前包导入types模块

class Test_measurements_stats:
    """ndimage._measurements._stats() is a utility used by other functions."""
    
    def test_a(self):
        x = [0, 1, 2, 6]  # 定义列表x
        labels = [0, 0, 1, 1]  # 定义列表labels
        index = [0, 1]  # 定义列表index，索引值为0和1
        for shp in [(4,), (2, 2)]:  # 遍历不同的形状(shp)，分别是(4,)和(2,2)
            x = np.array(x).reshape(shp)  # 将列表x转换为NumPy数组，并reshape成当前形状
            labels = np.array(labels).reshape(shp)  # 将列表labels转换为NumPy数组，并reshape成当前形状
            counts, sums = ndimage._measurements._stats(  # 调用ndimage._measurements._stats函数，计算统计信息
                x, labels=labels, index=index)
            assert_array_equal(counts, [2, 2])  # 断言counts数组与预期结果相等
            assert_array_equal(sums, [1.0, 8.0])  # 断言sums数组与预期结果相等

    def test_b(self):
        # Same data as test_a, but different labels.  The label 9 exceeds the
        # length of 'labels', so this test will follow a different code path.
        x = [0, 1, 2, 6]  # 定义列表x
        labels = [0, 0, 9, 9]  # 定义列表labels，其中包含一个超出labels长度的标签9
        index = [0, 9]  # 定义列表index，索引值为0和9
        for shp in [(4,), (2, 2)]:  # 遍历不同的形状(shp)，分别是(4,)和(2,2)
            x = np.array(x).reshape(shp)  # 将列表x转换为NumPy数组，并reshape成当前形状
            labels = np.array(labels).reshape(shp)  # 将列表labels转换为NumPy数组，并reshape成当前形状
            counts, sums = ndimage._measurements._stats(  # 调用ndimage._measurements._stats函数，计算统计信息
                x, labels=labels, index=index)
            assert_array_equal(counts, [2, 2])  # 断言counts数组与预期结果相等
            assert_array_equal(sums, [1.0, 8.0])  # 断言sums数组与预期结果相等

    def test_a_centered(self):
        x = [0, 1, 2, 6]  # 定义列表x
        labels = [0, 0, 1, 1]  # 定义列表labels
        index = [0, 1]  # 定义列表index，索引值为0和1
        for shp in [(4,), (2, 2)]:  # 遍历不同的形状(shp)，分别是(4,)和(2,2)
            x = np.array(x).reshape(shp)  # 将列表x转换为NumPy数组，并reshape成当前形状
            labels = np.array(labels).reshape(shp)  # 将列表labels转换为NumPy数组，并reshape成当前形状
            counts, sums, centers = ndimage._measurements._stats(  # 调用ndimage._measurements._stats函数，计算统计信息并返回中心值
                x, labels=labels, index=index, centered=True)
            assert_array_equal(counts, [2, 2])  # 断言counts数组与预期结果相等
            assert_array_equal(sums, [1.0, 8.0])  # 断言sums数组与预期结果相等
            assert_array_equal(centers, [0.5, 8.0])  # 断言centers数组与预期结果相等

    def test_b_centered(self):
        x = [0, 1, 2, 6]  # 定义列表x
        labels = [0, 0, 9, 9]  # 定义列表labels，其中包含一个超出labels长度的标签9
        index = [0, 9]  # 定义列表index，索引值为0和9
        for shp in [(4,), (2, 2)]:  # 遍历不同的形状(shp)，分别是(4,)和(2,2)
            x = np.array(x).reshape(shp)  # 将列表x转换为NumPy数组，并reshape成当前形状
            labels = np.array(labels).reshape(shp)  # 将列表labels转换为NumPy数组，并reshape成当前形状
            counts, sums, centers = ndimage._measurements._stats(  # 调用ndimage._measurements._stats函数，计算统计信息并返回中心值
                x, labels=labels, index=index, centered=True)
            assert_array_equal(counts, [2, 2])  # 断言counts数组与预期结果相等
            assert_array_equal(sums, [1.0, 8.0])  # 断言sums数组与预期结果相等
            assert_array_equal(centers, [0.5, 8.0])  # 断言centers数组与预期结果相等
    # 定义一个测试方法，用于测试非整数标签的情况
    def test_nonint_labels(self):
        # 初始化一个列表 x，包含整数元素
        x = [0, 1, 2, 6]
        # 初始化一个列表 labels，包含浮点数元素
        labels = [0.0, 0.0, 9.0, 9.0]
        # 初始化一个列表 index，包含浮点数元素
        index = [0.0, 9.0]
        
        # 对于每个元组 shp 在 [(4,), (2, 2)] 中
        for shp in [(4,), (2, 2)]:
            # 将列表 x 转换为 NumPy 数组，并按照 shp 元组的形状重新排列
            x = np.array(x).reshape(shp)
            # 将列表 labels 转换为 NumPy 数组，并按照 shp 元组的形状重新排列
            labels = np.array(labels).reshape(shp)
            
            # 使用 ndimage._measurements._stats 函数计算统计信息，传入参数 x, labels, index, centered=True
            counts, sums, centers = ndimage._measurements._stats(
                x, labels=labels, index=index, centered=True)
            
            # 断言 counts 数组与预期的数组 [2, 2] 相等
            assert_array_equal(counts, [2, 2])
            # 断言 sums 数组与预期的数组 [1.0, 8.0] 相等
            assert_array_equal(sums, [1.0, 8.0])
            # 断言 centers 数组与预期的数组 [0.5, 8.0] 相等
            assert_array_equal(centers, [0.5, 8.0])
class Test_measurements_select:
    """ndimage._measurements._select() is a utility used by other functions."""

    def test_basic(self):
        # 示例数据
        x = [0, 1, 6, 2]
        # 不同测试用例，包含标签和索引
        cases = [
            ([0, 0, 1, 1], [0, 1]),           # "Small" integer labels
            ([0, 0, 9, 9], [0, 9]),           # A label larger than len(labels)
            ([0.0, 0.0, 7.0, 7.0], [0.0, 7.0]),   # Non-integer labels
        ]
        # 遍历测试用例
        for labels, index in cases:
            # 调用 _select 函数，测试是否返回空列表
            result = ndimage._measurements._select(
                x, labels=labels, index=index)
            assert_(len(result) == 0)
            # 调用 _select 函数，测试是否返回包含最大值的数组
            result = ndimage._measurements._select(
                x, labels=labels, index=index, find_max=True)
            assert_(len(result) == 1)
            assert_array_equal(result[0], [1, 6])
            # 调用 _select 函数，测试是否返回包含最小值的数组
            result = ndimage._measurements._select(
                x, labels=labels, index=index, find_min=True)
            assert_(len(result) == 1)
            assert_array_equal(result[0], [0, 2])
            # 调用 _select 函数，测试是否返回包含最小值位置的数组
            result = ndimage._measurements._select(
                x, labels=labels, index=index, find_min=True,
                find_min_positions=True)
            assert_(len(result) == 2)
            assert_array_equal(result[0], [0, 2])
            assert_array_equal(result[1], [0, 3])
            assert_equal(result[1].dtype.kind, 'i')
            # 调用 _select 函数，测试是否返回包含最大值位置的数组
            result = ndimage._measurements._select(
                x, labels=labels, index=index, find_max=True,
                find_max_positions=True)
            assert_(len(result) == 2)
            assert_array_equal(result[0], [1, 6])
            assert_array_equal(result[1], [1, 2])
            assert_equal(result[1].dtype.kind, 'i')


def test_label01():
    # 测试用例：全为1的数据
    data = np.ones([])
    out, n = ndimage.label(data)
    assert_array_almost_equal(out, 1)
    assert_equal(n, 1)


def test_label02():
    # 测试用例：全为0的数据
    data = np.zeros([])
    out, n = ndimage.label(data)
    assert_array_almost_equal(out, 0)
    assert_equal(n, 0)


def test_label03():
    # 测试用例：包含一个元素的全为1的数组
    data = np.ones([1])
    out, n = ndimage.label(data)
    assert_array_almost_equal(out, [1])
    assert_equal(n, 1)


def test_label04():
    # 测试用例：包含一个元素的全为0的数组
    data = np.zeros([1])
    out, n = ndimage.label(data)
    assert_array_almost_equal(out, [0])
    assert_equal(n, 0)


def test_label05():
    # 测试用例：包含五个元素的全为1的数组
    data = np.ones([5])
    out, n = ndimage.label(data)
    assert_array_almost_equal(out, [1, 1, 1, 1, 1])
    assert_equal(n, 1)


def test_label06():
    # 测试用例：包含多个元素的混合数组
    data = np.array([1, 0, 1, 1, 0, 1])
    out, n = ndimage.label(data)
    assert_array_almost_equal(out, [1, 0, 2, 2, 0, 3])
    assert_equal(n, 3)


def test_label07():
    # 测试用例：全为0的二维数组
    data = np.array([[0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]])
    out, n = ndimage.label(data)
    # 使用断言检查变量 `out` 的值是否几乎等于一个全零的二维数组
    assert_array_almost_equal(out, [[0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0]])
    # 使用断言检查变量 `n` 的值是否等于 0
    assert_equal(n, 0)
def test_label08():
    data = np.array([[1, 0, 0, 0, 0, 0],   # 创建一个2D NumPy数组，表示待标记的数据
                     [0, 0, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 0],
                     [1, 1, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 0]])
    out, n = ndimage.label(data)  # 使用ndimage库的label函数对数据进行标记，并返回标记后的数组和标记数
    assert_array_almost_equal(out, [[1, 0, 0, 0, 0, 0],    # 断言标记后的数组与期望的数组几乎相等
                                    [0, 0, 2, 2, 0, 0],
                                    [0, 0, 2, 2, 2, 0],
                                    [3, 3, 0, 0, 0, 0],
                                    [3, 3, 0, 0, 0, 0],
                                    [0, 0, 0, 4, 4, 0]])
    assert_equal(n, 4)   # 断言标记的数量与期望值相等


def test_label09():
    data = np.array([[1, 0, 0, 0, 0, 0],   # 创建一个2D NumPy数组，表示待标记的数据
                     [0, 0, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 0],
                     [1, 1, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 0]])
    struct = ndimage.generate_binary_structure(2, 2)   # 创建一个指定结构的二进制结构数组
    out, n = ndimage.label(data, struct)   # 使用指定的结构对数据进行标记，并返回标记后的数组和标记数
    assert_array_almost_equal(out, [[1, 0, 0, 0, 0, 0],    # 断言标记后的数组与期望的数组几乎相等
                                    [0, 0, 2, 2, 0, 0],
                                    [0, 0, 2, 2, 2, 0],
                                    [2, 2, 0, 0, 0, 0],
                                    [2, 2, 0, 0, 0, 0],
                                    [0, 0, 0, 3, 3, 0]])
    assert_equal(n, 3)   # 断言标记的数量与期望值相等


def test_label10():
    data = np.array([[0, 0, 0, 0, 0, 0],   # 创建一个2D NumPy数组，表示待标记的数据
                     [0, 1, 1, 0, 1, 0],
                     [0, 1, 1, 1, 1, 0],
                     [0, 0, 0, 0, 0, 0]])
    struct = ndimage.generate_binary_structure(2, 2)   # 创建一个指定结构的二进制结构数组
    out, n = ndimage.label(data, struct)   # 使用指定的结构对数据进行标记，并返回标记后的数组和标记数
    assert_array_almost_equal(out, [[0, 0, 0, 0, 0, 0],    # 断言标记后的数组与期望的数组几乎相等
                                    [0, 1, 1, 0, 1, 0],
                                    [0, 1, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0]])
    assert_equal(n, 1)   # 断言标记的数量与期望值相等


def test_label11():
    for type in types:   # 对types列表中的每个类型执行以下操作
        data = np.array([[1, 0, 0, 0, 0, 0],   # 创建一个2D NumPy数组，表示待标记的数据
                         [0, 0, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 0],
                         [1, 1, 0, 0, 0, 0],
                         [1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 0]], type)
        out, n = ndimage.label(data)   # 使用ndimage库的label函数对数据进行标记，并返回标记后的数组和标记数
        expected = [[1, 0, 0, 0, 0, 0],    # 预期的标记后的数组
                    [0, 0, 2, 2, 0, 0],
                    [0, 0, 2, 2, 2, 0],
                    [3, 3, 0, 0, 0, 0],
                    [3, 3, 0, 0, 0, 0],
                    [0, 0, 0, 4, 4, 0]]
        assert_array_almost_equal(out, expected)   # 断言标记后的数组与期望的数组几乎相等
        assert_equal(n, 4)   # 断言标记的数量与期望值相等


def test_label11_inplace():
    # 遍历类型列表中的每个类型
    for type in types:
        # 创建一个二维数组，表示图像数据，根据当前类型
        data = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 0],
                         [1, 1, 0, 0, 0, 0],
                         [1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 0]], type)
        # 对图像数据进行标签化，修改数据本身作为输出
        n = ndimage.label(data, output=data)
        # 预期的标签化结果
        expected = [[1, 0, 0, 0, 0, 0],
                    [0, 0, 2, 2, 0, 0],
                    [0, 0, 2, 2, 2, 0],
                    [3, 3, 0, 0, 0, 0],
                    [3, 3, 0, 0, 0, 0],
                    [0, 0, 0, 4, 4, 0]]
        # 断言实际标签化后的数据与预期结果相等
        assert_array_almost_equal(data, expected)
        # 断言标签的数量与预期的标签数量相等
        assert_equal(n, 4)
def test_label12():
    # 遍历类型列表中的每种类型
    for type in types:
        # 创建一个二维数组，并指定类型
        data = np.array([[0, 0, 0, 0, 1, 1],
                         [0, 0, 0, 0, 0, 1],
                         [0, 0, 1, 0, 1, 1],
                         [0, 0, 1, 1, 1, 1],
                         [0, 0, 0, 1, 1, 0]], type)
        # 对数据应用标签化函数，返回标签化后的结果和标签数
        out, n = ndimage.label(data)
        # 期望的标签化结果
        expected = [[0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 1, 0, 1, 1],
                    [0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 0]]
        # 检查实际输出是否与期望相符
        assert_array_almost_equal(out, expected)
        # 检查标签数是否正确
        assert_equal(n, 1)


def test_label13():
    # 遍历类型列表中的每种类型
    for type in types:
        # 创建一个二维数组，并指定类型
        data = np.array([[1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                         [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                        type)
        # 对数据应用标签化函数，返回标签化后的结果和标签数
        out, n = ndimage.label(data)
        # 期望的标签化结果
        expected = [[1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                    [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        # 检查实际输出是否与期望相符
        assert_array_almost_equal(out, expected)
        # 检查标签数是否正确
        assert_equal(n, 1)


def test_label_output_typed():
    # 创建一个全一的一维数组
    data = np.ones([5])
    # 遍历类型列表中的每种类型
    for t in types:
        # 创建一个全零的一维数组，类型为当前迭代的类型 t
        output = np.zeros([5], dtype=t)
        # 对数据应用标签化函数，返回标签化后的标签数
        n = ndimage.label(data, output=output)
        # 检查输出数组是否全为 1
        assert_array_almost_equal(output, 1)
        # 检查标签数是否为 1
        assert_equal(n, 1)


def test_label_output_dtype():
    # 创建一个全一的一维数组
    data = np.ones([5])
    # 遍历类型列表中的每种类型
    for t in types:
        # 对数据应用标签化函数，返回标签化后的输出数组和标签数
        output, n = ndimage.label(data, output=t)
        # 检查输出数组是否全为 1
        assert_array_almost_equal(output, 1)
        # 检查输出数组的数据类型是否为当前迭代的类型 t
        assert output.dtype == t


def test_label_output_wrong_size():
    # 创建一个全一的一维数组
    data = np.ones([5])
    # 遍历类型列表中的每种类型
    for t in types:
        # 创建一个全零的一维数组，类型为当前迭代的类型 t
        output = np.zeros([10], t)
        # 检查当输出数组大小与输入数据不匹配时是否抛出异常
        assert_raises((RuntimeError, ValueError),
                      ndimage.label, data, output=output)


def test_label_structuring_elements():
    # 从文件加载数据，用于标签化输入
    data = np.loadtxt(os.path.join(os.path.dirname(
        __file__), "data", "label_inputs.txt"))
    # 从文件加载结构元素数据，用于标签化
    strels = np.loadtxt(os.path.join(
        os.path.dirname(__file__), "data", "label_strels.txt"))
    # 从文件加载预期结果数据
    results = np.loadtxt(os.path.join(
        os.path.dirname(__file__), "data", "label_results.txt"))
    # 重新调整数据的形状
    data = data.reshape((-1, 7, 7))
    strels = strels.reshape((-1, 3, 3))
    results = results.reshape((-1, 7, 7))
    r = 0
    # 遍历输入数据的每个子数组
    for i in range(data.shape[0]):
        d = data[i, :, :]
        # 遍历结构元素数据的每个子数组
        for j in range(strels.shape[0]):
            s = strels[j, :, :]
            # 断言标签化函数的输出与预期结果是否一致
            assert_equal(ndimage.label(d, s)[0], results[r, :, :])
            r += 1


def test_ticket_742():
    # 定义一个函数，接收图像和阈值参数，并生成二进制掩码
    def SE(img, thresh=.7, size=4):
        mask = img > thresh
        rank = len(mask.shape)
        # 根据二进制结构生成二进制结构元素，并进行标签化
        la, co = ndimage.label(mask,
                               ndimage.generate_binary_structure(rank, rank))
        # 查找标签化结果中的对象，并忽略返回值
        _ = ndimage.find_objects(la)
    # 检查 NumPy 中的整数类型是否为默认的 'int' 类型，如果不是则执行以下操作
    if np.dtype(np.intp) != np.dtype('i'):
        # 定义一个形状为 (3, 1240, 1240) 的数组
        shape = (3, 1240, 1240)
        # 使用随机数填充一个与指定形状相匹配的数组
        a = np.random.rand(np.prod(shape)).reshape(shape)
        # 调用 SE 函数，传递数组 a 作为参数，期望它不会导致程序崩溃
        # SE 函数的具体功能和作用未在当前上下文中提供
        SE(a)
def test_gh_issue_3025():
    """Github issue #3025 - improper merging of labels"""
    # 创建一个60x320的全零数组
    d = np.zeros((60, 320))
    # 将第一列到第257列的所有行设置为1
    d[:, :257] = 1
    # 将第260列到最后一列的所有行设置为1
    d[:, 260:] = 1
    # 设置特定位置的元素为1，形成特定的模式
    d[36, 257] = 1
    d[35, 258] = 1
    d[35, 259] = 1
    # 断言语句，验证标签合并操作后的结果是否为1
    assert ndimage.label(d, np.ones((3, 3)))[1] == 1


def test_label_default_dtype():
    # 创建一个大小为10x10的随机数组
    test_array = np.random.rand(10, 10)
    # 对数组中大于0.5的元素进行标签化处理
    label, no_features = ndimage.label(test_array > 0.5)
    # 断言语句，验证标签数组的数据类型为np.int32或np.int64之一
    assert_(label.dtype in (np.int32, np.int64))
    # 不应该引发异常
    ndimage.find_objects(label)


def test_find_objects01():
    # 创建一个空的一维整数数组
    data = np.ones([], dtype=int)
    # 使用ndimage.find_objects查找数组中的对象
    out = ndimage.find_objects(data)
    # 断言语句，验证结果是否为[()]
    assert_(out == [()])


def test_find_objects02():
    # 创建一个空的一维整数数组
    data = np.zeros([], dtype=int)
    # 使用ndimage.find_objects查找数组中的对象
    out = ndimage.find_objects(data)
    # 断言语句，验证结果是否为空列表
    assert_(out == [])


def test_find_objects03():
    # 创建一个包含单个元素的一维整数数组
    data = np.ones([1], dtype=int)
    # 使用ndimage.find_objects查找数组中的对象
    out = ndimage.find_objects(data)
    # 断言语句，验证结果是否为[(slice(0, 1, None),)]
    assert_equal(out, [(slice(0, 1, None),)])


def test_find_objects04():
    # 创建一个包含单个元素的一维整数数组
    data = np.zeros([1], dtype=int)
    # 使用ndimage.find_objects查找数组中的对象
    out = ndimage.find_objects(data)
    # 断言语句，验证结果是否为空列表
    assert_equal(out, [])


def test_find_objects05():
    # 创建一个包含5个元素的一维整数数组
    data = np.ones([5], dtype=int)
    # 使用ndimage.find_objects查找数组中的对象
    out = ndimage.find_objects(data)
    # 断言语句，验证结果是否为[(slice(0, 5, None),)]
    assert_equal(out, [(slice(0, 5, None),)])


def test_find_objects06():
    # 创建一个一维整数数组
    data = np.array([1, 0, 2, 2, 0, 3])
    # 使用ndimage.find_objects查找数组中的对象
    out = ndimage.find_objects(data)
    # 断言语句，验证结果是否为[(slice(0, 1, None),), (slice(2, 4, None),), (slice(5, 6, None),)]
    assert_equal(out, [(slice(0, 1, None),),
                       (slice(2, 4, None),),
                       (slice(5, 6, None),)])


def test_find_objects07():
    # 创建一个全零的二维整数数组
    data = np.array([[0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]])
    # 使用ndimage.find_objects查找数组中的对象
    out = ndimage.find_objects(data)
    # 断言语句，验证结果是否为空列表
    assert_equal(out, [])


def test_find_objects08():
    # 创建一个包含不同整数的二维整数数组
    data = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 0, 2, 2, 0, 0],
                     [0, 0, 2, 2, 2, 0],
                     [3, 3, 0, 0, 0, 0],
                     [3, 3, 0, 0, 0, 0],
                     [0, 0, 0, 4, 4, 0]])
    # 使用ndimage.find_objects查找数组中的对象
    out = ndimage.find_objects(data)
    # 断言语句，验证结果是否为指定的切片列表
    assert_equal(out, [(slice(0, 1, None), slice(0, 1, None)),
                       (slice(1, 3, None), slice(2, 5, None)),
                       (slice(3, 5, None), slice(0, 2, None)),
                       (slice(5, 6, None), slice(3, 5, None))])


def test_find_objects09():
    # 创建一个包含不同整数的二维整数数组
    data = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 0, 2, 2, 0, 0],
                     [0, 0, 2, 2, 2, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 4, 4, 0]])
    # 使用ndimage.find_objects查找数组中的对象
    out = ndimage.find_objects(data)
    # 断言语句，验证结果是否为指定的切片列表
    assert_equal(out, [(slice(0, 1, None), slice(0, 1, None)),
                       (slice(1, 3, None), slice(2, 5, None)),
                       None,
                       (slice(5, 6, None), slice(3, 5, None))])


def test_value_indices01():
    "Test dictionary keys and entries"
    # 创建一个 NumPy 数组，表示一个二维的稀疏矩阵
    data = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 0, 2, 2, 0, 0],
                     [0, 0, 2, 2, 2, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 4, 4, 0]])
    
    # 使用 SciPy 的 ndimage 库的 value_indices 函数，找出非零元素的索引
    vi = ndimage.value_indices(data, ignore_value=0)
    
    # 真实的键集合，即非零元素可能的值
    true_keys = [1, 2, 4]
    
    # 使用断言确保 value_indices 返回的键列表与预期的 true_keys 相同
    assert_equal(list(vi.keys()), true_keys)
    
    # 初始化一个空字典，用于存储真实的值索引
    truevi = {}
    
    # 遍历真实的键集合 true_keys
    for k in true_keys:
        # 将 data 中等于 k 的位置索引存储到 truevi[k] 中
        truevi[k] = np.where(data == k)
    
    # 重新计算 vi，以确保它与 truevi 相匹配
    vi = ndimage.value_indices(data, ignore_value=0)
    
    # 使用断言确保 vi 与 truevi 相等
    assert_equal(vi, truevi)
def test_value_indices02():
    "Test input checking"
    # 创建一个 5x4 的全零数组，数据类型为 np.float32
    data = np.zeros((5, 4), dtype=np.float32)
    # 指定错误信息
    msg = "Parameter 'arr' must be an integer array"
    # 使用 assert_raises 来检查数值索引函数对错误输入的处理
    with assert_raises(ValueError, match=msg):
        ndimage.value_indices(data)


def test_value_indices03():
    "Test different input array shapes, from 1-D to 4-D"
    # 针对不同的数组形状进行测试，从一维到四维
    for shape in [(36,), (18, 2), (3, 3, 4), (3, 3, 2, 2)]:
        # 创建具有指定形状的数组，并填充特定的数据
        a = np.array((12*[1]+12*[2]+12*[3]), dtype=np.int32).reshape(shape)
        # 获取数组中唯一的键（即唯一的值）
        trueKeys = np.unique(a)
        # 使用数值索引函数获取值对应的索引
        vi = ndimage.value_indices(a)
        # 断言 vi 的键列表与 trueKeys 相同
        assert_equal(list(vi.keys()), list(trueKeys))
        # 对于每个唯一值 k，验证数值索引是否正确
        for k in trueKeys:
            trueNdx = np.where(a == k)
            assert_equal(vi[k], trueNdx)


def test_sum01():
    # 对于每种数据类型进行测试
    for type in types:
        # 创建空数组，指定数据类型
        input = np.array([], type)
        # 使用 ndimage.sum 计算数组的和
        output = ndimage.sum(input)
        # 断言输出的和为 0.0
        assert_equal(output, 0.0)


# 以下函数 test_sum02 到 test_sum12 的注释模式类似，主要测试不同的输入和标签组合以及数据类型处理的正确性。
    # 对于每种数据类型执行以下操作
    for type in types:
        # 创建一个包含特定数据类型的NumPy数组
        input = np.array([[1, 2], [3, 4]], type)
        # 使用ndimage模块计算输入数组的和，根据给定的标签和索引
        output = ndimage.sum(input, labels=labels, index=[4, 8, 2])
        # 断言计算得到的输出与预期的几乎相等
        assert_array_almost_equal(output, [4.0, 0.0, 5.0])
# 测试函数，用于验证 sum_labels 函数的正确性
def test_sum_labels():
    # 创建一个包含指定元素的 NumPy 数组，元素类型为 np.int8
    labels = np.array([[1, 2], [2, 4]], np.int8)
    # 遍历 types 列表中的每一种类型
    for type in types:
        # 创建一个包含指定元素和类型的 NumPy 数组
        input = np.array([[1, 2], [3, 4]], type)
        # 计算输入数组的指定标签区域的像素和
        output_sum = ndimage.sum(input, labels=labels, index=[4, 8, 2])
        # 使用 sum_labels 函数计算输入数组的指定标签区域的像素和
        output_labels = ndimage.sum_labels(input, labels=labels, index=[4, 8, 2])

        # 断言两者结果数组的所有元素均相等
        assert (output_sum == output_labels).all()
        # 断言输出标签区域的像素和与预期值接近
        assert_array_almost_equal(output_labels, [4.0, 0.0, 5.0])


# 测试函数，验证 mean 函数在给定标签下的正确性
def test_mean01():
    # 创建一个包含指定元素的 NumPy 数组，元素类型为 bool
    labels = np.array([1, 0], bool)
    # 遍历 types 列表中的每一种类型
    for type in types:
        # 创建一个包含指定元素和类型的 NumPy 数组
        input = np.array([[1, 2], [3, 4]], type)
        # 计算输入数组在指定标签下的像素均值
        output = ndimage.mean(input, labels=labels)
        # 断言输出均值与预期值接近
        assert_almost_equal(output, 2.0)


# 测试函数，验证 mean 函数在全为 bool 类型输入下的正确性
def test_mean02():
    # 创建一个包含指定元素的 NumPy 数组，元素类型为 bool
    labels = np.array([1, 0], bool)
    # 创建一个包含指定元素的 NumPy 数组，元素类型为 bool
    input = np.array([[1, 2], [3, 4]], bool)
    # 计算输入数组在指定标签下的像素均值
    output = ndimage.mean(input, labels=labels)
    # 断言输出均值与预期值接近
    assert_almost_equal(output, 1.0)


# 测试函数，验证 mean 函数在给定标签和索引下的正确性
def test_mean03():
    # 创建一个包含指定元素的 NumPy 数组
    labels = np.array([1, 2])
    # 遍历 types 列表中的每一种类型
    for type in types:
        # 创建一个包含指定元素和类型的 NumPy 数组
        input = np.array([[1, 2], [3, 4]], type)
        # 计算输入数组在指定标签和索引下的像素均值
        output = ndimage.mean(input, labels=labels, index=2)
        # 断言输出均值与预期值接近
        assert_almost_equal(output, 3.0)


# 测试函数，验证 mean 函数在给定标签、索引和忽略警告条件下的正确性
def test_mean04():
    # 创建一个包含指定元素的 NumPy 数组，元素类型为 np.int8
    labels = np.array([[1, 2], [2, 4]], np.int8)
    # 忽略 NumPy 的全部警告
    with np.errstate(all='ignore'):
        # 遍历 types 列表中的每一种类型
        for type in types:
            # 创建一个包含指定元素和类型的 NumPy 数组
            input = np.array([[1, 2], [3, 4]], type)
            # 计算输入数组在指定标签、索引下的像素均值
            output = ndimage.mean(input, labels=labels, index=[4, 8, 2])
            # 断言输出数组的部分元素与预期值接近
            assert_array_almost_equal(output[[0, 2]], [4.0, 2.5])
            # 断言输出数组的指定位置元素为 NaN
            assert_(np.isnan(output[1]))


# 测试函数，验证 minimum 函数在给定标签下的正确性
def test_minimum01():
    # 创建一个包含指定元素的 NumPy 数组，元素类型为 bool
    labels = np.array([1, 0], bool)
    # 遍历 types 列表中的每一种类型
    for type in types:
        # 创建一个包含指定元素和类型的 NumPy 数组
        input = np.array([[1, 2], [3, 4]], type)
        # 计算输入数组在指定标签下的最小像素值
        output = ndimage.minimum(input, labels=labels)
        # 断言输出最小值与预期值接近
        assert_almost_equal(output, 1.0)


# 测试函数，验证 minimum 函数在全为 bool 类型输入下的正确性
def test_minimum02():
    # 创建一个包含指定元素的 NumPy 数组，元素类型为 bool
    labels = np.array([1, 0], bool)
    # 创建一个包含指定元素的 NumPy 数组，元素类型为 bool
    input = np.array([[2, 2], [2, 4]], bool)
    # 计算输入数组在指定标签下的最小像素值
    output = ndimage.minimum(input, labels=labels)
    # 断言输出最小值与预期值接近
    assert_almost_equal(output, 1.0)


# 测试函数，验证 minimum 函数在给定标签和索引下的正确性
def test_minimum03():
    # 创建一个包含指定元素的 NumPy 数组
    labels = np.array([1, 2])
    # 遍历 types 列表中的每一种类型
    for type in types:
        # 创建一个包含指定元素和类型的 NumPy 数组
        input = np.array([[1, 2], [3, 4]], type)
        # 计算输入数组在指定标签和索引下的最小像素值
        output = ndimage.minimum(input, labels=labels, index=2)
        # 断言输出最小值与预期值接近
        assert_almost_equal(output, 2.0)


# 测试函数，验证 minimum 函数在给定标签、索引和多个索引下的正确性
def test_minimum04():
    # 创建一个包含指定元素的 NumPy 数组
    labels = np.array([[1, 2], [2, 3]])
    # 遍历 types 列表中的每一种类型
    for type in types:
        # 创建一个包含指定元素和类型的 NumPy 数组
        input = np.array([[1, 2], [3, 4]], type)
        # 计算输入数组在指定标签、索引和多个索引下的最小像素值
        output = ndimage.minimum(input, labels=labels, index=[2, 3, 8])
        # 断言输出数组与预期值接近
        assert_array_almost_equal(output, [2.0, 4.0, 0.0])


# 测试函数，验证 maximum 函数在给定标签下的正确性
def test_maximum01():
    # 创建一个包含指定元素的 NumPy 数组，元素类型为 bool
    labels = np.array([1, 0], bool)
    # 遍历 types 列表中的每一种类型
    for type in types:
        # 创建一个包含指定元素和类型的 NumPy 数组
        input = np.array([[1, 2], [3, 4]], type)
        # 计算输入数组在指定标签下的最大像素值
        output = ndimage.maximum(input, labels=labels)
        # 断言输出最大值与预期值接近
        assert_almost_equal(output, 3.0)


# 测试函数，验证 maximum 函数在全为 bool 类型输入下的正确性
def test_maximum02():
    # 创建一个包含指定元素的 NumPy 数组，元素类型为 bool
    labels =
    # 对于每种数据类型 type 在 types 中
    for type in types:
        # 创建一个 2x2 的 NumPy 数组，数据类型为 type
        input = np.array([[1, 2], [3, 4]], type)
        # 使用 ndimage.maximum 函数计算输入数组的最大值
        # labels 参数指定标签数组，index 参数指定要计算最大值的索引
        output = ndimage.maximum(input, labels=labels, index=2)
        # 使用 assert_almost_equal 断言函数验证输出结果接近于 4.0
        assert_almost_equal(output, 4.0)
def test_maximum04():
    # 创建一个二维数组作为标签
    labels = np.array([[1, 2], [2, 3]])
    # 遍历类型列表中的每个类型
    for type in types:
        # 创建一个指定类型的二维数组作为输入
        input = np.array([[1, 2], [3, 4]], type)
        # 调用 ndimage 库的 maximum 函数，计算输入数组中的最大值
        # 使用预设的标签和索引参数
        output = ndimage.maximum(input, labels=labels,
                                 index=[2, 3, 8])
        # 断言计算结果与预期的近似值相等
        assert_array_almost_equal(output, [3.0, 4.0, 0.0])


def test_maximum05():
    # 对于 ticket #501 (Trac) 的回归测试
    # 创建一个包含负数的一维数组
    x = np.array([-3, -2, -1])
    # 断言 ndimage 库计算的最大值与预期值相等
    assert_equal(ndimage.maximum(x), -1)


def test_median01():
    # 创建一个二维数组作为输入
    a = np.array([[1, 2, 0, 1],
                  [5, 3, 0, 4],
                  [0, 0, 0, 7],
                  [9, 3, 0, 0]])
    # 创建一个二维数组作为标签
    labels = np.array([[1, 1, 0, 2],
                       [1, 1, 0, 2],
                       [0, 0, 0, 2],
                       [3, 3, 0, 0]])
    # 调用 ndimage 库的 median 函数，计算输入数组的中位数
    # 使用预设的标签和索引参数
    output = ndimage.median(a, labels=labels, index=[1, 2, 3])
    # 断言计算结果与预期的近似值相等
    assert_array_almost_equal(output, [2.5, 4.0, 6.0])


def test_median02():
    # 创建一个二维数组作为输入
    a = np.array([[1, 2, 0, 1],
                  [5, 3, 0, 4],
                  [0, 0, 0, 7],
                  [9, 3, 0, 0]])
    # 调用 ndimage 库的 median 函数，计算输入数组的中位数
    output = ndimage.median(a)
    # 断言计算结果与预期的近似值相等
    assert_almost_equal(output, 1.0)


def test_median03():
    # 创建一个二维数组作为输入
    a = np.array([[1, 2, 0, 1],
                  [5, 3, 0, 4],
                  [0, 0, 0, 7],
                  [9, 3, 0, 0]])
    # 创建一个二维数组作为标签
    labels = np.array([[1, 1, 0, 2],
                       [1, 1, 0, 2],
                       [0, 0, 0, 2],
                       [3, 3, 0, 0]])
    # 调用 ndimage 库的 median 函数，计算输入数组的中位数
    # 使用预设的标签参数
    output = ndimage.median(a, labels=labels)
    # 断言计算结果与预期的近似值相等
    assert_almost_equal(output, 3.0)


def test_median_gh12836_bool():
    # 对 gh-12836 中的布尔值加法修复进行测试
    # 创建一个布尔类型的一维数组
    a = np.asarray([1, 1], dtype=bool)
    # 调用 ndimage 库的 median 函数，计算输入数组的中位数
    # 使用预设的标签和索引参数
    output = ndimage.median(a, labels=np.ones((2,)), index=[1])
    # 断言计算结果与预期的近似值相等
    assert_array_almost_equal(output, [1.0])


def test_median_no_int_overflow():
    # 对 gh-12836 中的整数溢出修复进行测试
    # 创建一个 int8 类型的一维数组
    a = np.asarray([65, 70], dtype=np.int8)
    # 调用 ndimage 库的 median 函数，计算输入数组的中位数
    # 使用预设的标签和索引参数
    output = ndimage.median(a, labels=np.ones((2,)), index=[1])
    # 断言计算结果与预期的近似值相等
    assert_array_almost_equal(output, [67.5])


def test_variance01():
    # 忽略所有错误状态
    with np.errstate(all='ignore'):
        # 遍历类型列表中的每个类型
        for type in types:
            # 创建一个空的输入数组
            input = np.array([], type)
            # 使用警告抑制器
            with suppress_warnings() as sup:
                # 过滤特定的运行时警告信息
                sup.filter(RuntimeWarning, "Mean of empty slice")
                # 调用 ndimage 库的 variance 函数，计算输入数组的方差
                output = ndimage.variance(input)
            # 断言输出为 NaN
            assert_(np.isnan(output))


def test_variance02():
    # 遍历类型列表中的每个类型
    for type in types:
        # 创建一个包含单个元素的输入数组
        input = np.array([1], type)
        # 调用 ndimage 库的 variance 函数，计算输入数组的方差
        output = ndimage.variance(input)
        # 断言计算结果与预期的近似值相等
        assert_almost_equal(output, 0.0)


def test_variance03():
    # 遍历类型列表中的每个类型
    for type in types:
        # 创建一个包含两个元素的输入数组
        input = np.array([1, 3], type)
        # 调用 ndimage 库的 variance 函数，计算输入数组的方差
        output = ndimage.variance(input)
        # 断言计算结果与预期的近似值相等
        assert_almost_equal(output, 1.0)


def test_variance04():
    # 创建一个布尔类型的输入数组
    input = np.array([1, 0], bool)
    # 调用 ndimage 库的 variance 函数，计算输入数组的方差
    output = ndimage.variance(input)
    # 断言计算结果与预期的近似值相等
    assert_almost_equal(output, 0.25)


def test_variance05():
    # 创建一个包含三个标签的列表
    labels = [2, 2, 3]
    # 遍历类型列表中的每个类型
    for type in types:
        # 创建一个包含三个元素的输入数组
        input = np.array([1, 3, 8], type)
        # 调用 ndimage 库的 variance 函数，计算输入数组的方差
        # 使用预设的标签和索引参数
        output = ndimage.variance(input, labels, 2)
        # 断言计算结果与预期的近似值相等
        assert_almost_equal(output, 1.0)


def test_variance06():
    # 待实现
    pass
    # 定义一个列表，包含标签数据
    labels = [2, 2, 3, 3, 4]
    # 忽略 NumPy 的所有错误
    with np.errstate(all='ignore'):
        # 对于类型列表中的每一个类型
        for type in types:
            # 创建一个 NumPy 数组，包含指定类型的数据
            input = np.array([1, 3, 8, 10, 8], type)
            # 计算输入数据的方差，使用给定的标签和输出形状参数
            output = ndimage.variance(input, labels, [2, 3, 4])
            # 断言计算结果与预期结果几乎相等
            assert_array_almost_equal(output, [1.0, 1.0, 0.0])
# 定义测试函数，用于测试 ndimage.standard_deviation 函数的不同情况
def test_standard_deviation01():
    # 忽略所有 NumPy 的错误和警告
    with np.errstate(all='ignore'):
        # 遍历 types 列表中的每种数据类型
        for type in types:
            # 创建空的 NumPy 数组，并指定数据类型
            input = np.array([], type)
            # 使用 suppress_warnings 上下文管理器来过滤特定的运行时警告
            with suppress_warnings() as sup:
                # 过滤 RuntimeWarning 类型的警告信息 "Mean of empty slice"
                sup.filter(RuntimeWarning, "Mean of empty slice")
                # 计算输入数组的标准差
                output = ndimage.standard_deviation(input)
            # 使用 assert_ 断言函数验证输出结果是否为 NaN
            assert_(np.isnan(output))

# 定义测试函数，用于测试 ndimage.standard_deviation 函数的不同情况
def test_standard_deviation02():
    # 遍历 types 列表中的每种数据类型
    for type in types:
        # 创建包含单个元素的 NumPy 数组，并指定数据类型
        input = np.array([1], type)
        # 计算输入数组的标准差
        output = ndimage.standard_deviation(input)
        # 使用 assert_almost_equal 断言函数验证输出结果是否接近于 0.0
        assert_almost_equal(output, 0.0)

# 定义测试函数，用于测试 ndimage.standard_deviation 函数的不同情况
def test_standard_deviation03():
    # 遍历 types 列表中的每种数据类型
    for type in types:
        # 创建包含两个元素的 NumPy 数组，并指定数据类型
        input = np.array([1, 3], type)
        # 计算输入数组的标准差
        output = ndimage.standard_deviation(input)
        # 使用 assert_almost_equal 断言函数验证输出结果是否接近于 1.0 的平方根
        assert_almost_equal(output, np.sqrt(1.0))

# 定义测试函数，用于测试 ndimage.standard_deviation 函数的不同情况
def test_standard_deviation04():
    # 创建包含两个布尔值元素的 NumPy 数组
    input = np.array([1, 0], bool)
    # 计算输入数组的标准差
    output = ndimage.standard_deviation(input)
    # 使用 assert_almost_equal 断言函数验证输出结果是否接近于 0.5
    assert_almost_equal(output, 0.5)

# 定义测试函数，用于测试 ndimage.standard_deviation 函数的不同情况
def test_standard_deviation05():
    # 定义标签列表
    labels = [2, 2, 3]
    # 遍历 types 列表中的每种数据类型
    for type in types:
        # 创建包含三个元素的 NumPy 数组，并指定数据类型
        input = np.array([1, 3, 8], type)
        # 计算输入数组的标准差，指定标签和轴
        output = ndimage.standard_deviation(input, labels, 2)
        # 使用 assert_almost_equal 断言函数验证输出结果是否接近于 1.0
        assert_almost_equal(output, 1.0)

# 定义测试函数，用于测试 ndimage.standard_deviation 函数的不同情况
def test_standard_deviation06():
    # 定义标签列表
    labels = [2, 2, 3, 3, 4]
    # 忽略所有 NumPy 的错误和警告
    with np.errstate(all='ignore'):
        # 遍历 types 列表中的每种数据类型
        for type in types:
            # 创建包含五个元素的 NumPy 数组，并指定数据类型
            input = np.array([1, 3, 8, 10, 8], type)
            # 计算输入数组的标准差，指定标签和轴
            output = ndimage.standard_deviation(input, labels, [2, 3, 4])
            # 使用 assert_array_almost_equal 断言函数验证输出结果是否接近于给定数组
            assert_array_almost_equal(output, [1.0, 1.0, 0.0])

# 定义测试函数，用于测试 ndimage.standard_deviation 函数的不同情况
def test_standard_deviation07():
    # 定义标签列表
    labels = [1]
    # 忽略所有 NumPy 的错误和警告
    with np.errstate(all='ignore'):
        # 遍历 types 列表中的每种数据类型
        for type in types:
            # 创建包含单个元素的 NumPy 数组，并指定数据类型
            input = np.array([-0.00619519], type)
            # 计算输入数组的标准差，指定标签和轴
            output = ndimage.standard_deviation(input, labels, [1])
            # 使用 assert_array_almost_equal 断言函数验证输出结果是否接近于给定数组
            assert_array_almost_equal(output, [0])

# 定义测试函数，用于测试 ndimage.minimum_position 函数的不同情况
def test_minimum_position01():
    # 创建包含两个布尔值元素的 NumPy 数组
    labels = np.array([1, 0], bool)
    # 遍历 types 列表中的每种数据类型
    for type in types:
        # 创建包含两行两列的 NumPy 数组，并指定数据类型
        input = np.array([[1, 2], [3, 4]], type)
        # 计算输入数组的最小位置，指定标签
        output = ndimage.minimum_position(input, labels=labels)
        # 使用 assert_equal 断言函数验证输出结果是否等于预期的元组 (0, 0)
        assert_equal(output, (0, 0))

# 定义测试函数，用于测试 ndimage.minimum_position 函数的不同情况
def test_minimum_position02():
    # 遍历 types 列表中的每种数据类型
    for type in types:
        # 创建包含三行四列的 NumPy 数组，并指定数据类型
        input = np.array([[5, 4, 2, 5],
                          [3, 7, 0, 2],
                          [1, 5, 1, 1]], type)
        # 计算输入数组的最小位置
        output = ndimage.minimum_position(input)
        # 使用 assert_equal 断言函数验证输出结果是否等于预期的元组 (1, 2)
        assert_equal(output, (1, 2))

# 定义测试函数，用于测试 ndimage.minimum_position 函数的不同情况
def test_minimum_position03():
    # 创建包含三行四列的布尔值 NumPy 数组
    input = np.array([[5, 4, 2, 5],
                      [3, 7, 0, 2],
                      [1, 5, 1, 1]], bool)
    # 计算输入数组的最小位置
    output = ndimage.minimum_position(input)
    # 使用 assert_equal 断言函数验证输出结果是否等于预期的元组 (1, 2)
    assert_equal(output, (1, 2))

# 定义测试函数，用于测试 ndimage.minimum_position 函数的不同情况
def test_minimum_position04():
    # 创建包含三行四列的布尔值 NumPy 数组
    input = np.array([[5, 4, 2, 5],
                      [3, 7, 1, 2],
                      [1, 5, 1, 1]], bool)
    # 计算输入数组的最小位置
    output = ndimage.minimum_position(input)
    # 使用 assert_equal 断言函数验证输出结果是否等于预期的元组 (0, 0)
    assert_equal(output, (0, 0))

# 定义测试函数，用于测试 ndimage.minimum_position 函数的不同情况
def test_minimum_position05():
    # 定义标签列表
    labels = [1, 2, 0, 4]
    # 遍历 types 列表中的每种数据类型
    for type in types:
        # 创建包含三行四列的 NumPy 数组，并指定数据类型
        input = np.array([[5, 4, 2, 5],
                          [3, 7, 0, 2],
                          [1, 5, 2, 3]], type)
        #
# 定义一个测试函数，用于测试 ndimage 库中的 minimum_position 函数的不同情况
def test_minimum_position06():
    # 设定标签列表
    labels = [1, 2, 3, 4]
    # 对于 types 列表中的每种类型
    for type in types:
        # 创建一个二维 numpy 数组作为输入，数据类型由 type 决定
        input = np.array([[5, 4, 2, 5],
                          [3, 7, 0, 2],
                          [1, 5, 1, 1]], type)
        # 调用 ndimage 库中的 minimum_position 函数，返回最小值的位置信息
        output = ndimage.minimum_position(input, labels, 2)
        # 使用 assert_equal 断言函数验证输出是否符合预期
        assert_equal(output, (0, 1))


# 定义另一个测试函数，测试 ndimage 库中的 minimum_position 函数的另一种情况
def test_minimum_position07():
    # 设定标签列表
    labels = [1, 2, 3, 4]
    # 对于 types 列表中的每种类型
    for type in types:
        # 创建一个二维 numpy 数组作为输入，数据类型由 type 决定
        input = np.array([[5, 4, 2, 5],
                          [3, 7, 0, 2],
                          [1, 5, 1, 1]], type)
        # 调用 ndimage 库中的 minimum_position 函数，返回多个最小值的位置信息
        output = ndimage.minimum_position(input, labels,
                                          [2, 3])
        # 使用 assert_equal 断言函数验证输出的每个位置是否符合预期
        assert_equal(output[0], (0, 1))
        assert_equal(output[1], (1, 2))


# 定义一个测试函数，用于测试 ndimage 库中的 maximum_position 函数的不同情况
def test_maximum_position01():
    # 创建一个布尔类型的标签数组
    labels = np.array([1, 0], bool)
    # 对于 types 列表中的每种类型
    for type in types:
        # 创建一个二维 numpy 数组作为输入，数据类型由 type 决定
        input = np.array([[1, 2], [3, 4]], type)
        # 调用 ndimage 库中的 maximum_position 函数，返回最大值的位置信息
        output = ndimage.maximum_position(input,
                                          labels=labels)
        # 使用 assert_equal 断言函数验证输出是否符合预期
        assert_equal(output, (1, 0))


# 定义另一个测试函数，测试 ndimage 库中的 maximum_position 函数的另一种情况
def test_maximum_position02():
    # 对于 types 列表中的每种类型
    for type in types:
        # 创建一个二维 numpy 数组作为输入，数据类型由 type 决定
        input = np.array([[5, 4, 2, 5],
                          [3, 7, 8, 2],
                          [1, 5, 1, 1]], type)
        # 调用 ndimage 库中的 maximum_position 函数，返回最大值的位置信息
        output = ndimage.maximum_position(input)
        # 使用 assert_equal 断言函数验证输出是否符合预期
        assert_equal(output, (1, 2))


# 定义另一个测试函数，测试 ndimage 库中的 maximum_position 函数的另一种情况
def test_maximum_position03():
    # 创建一个布尔类型的二维 numpy 数组作为输入
    input = np.array([[5, 4, 2, 5],
                      [3, 7, 8, 2],
                      [1, 5, 1, 1]], bool)
    # 调用 ndimage 库中的 maximum_position 函数，返回最大值的位置信息
    output = ndimage.maximum_position(input)
    # 使用 assert_equal 断言函数验证输出是否符合预期
    assert_equal(output, (0, 0))


# 定义另一个测试函数，测试 ndimage 库中的 maximum_position 函数的另一种情况
def test_maximum_position04():
    # 设定标签列表
    labels = [1, 2, 0, 4]
    # 对于 types 列表中的每种类型
    for type in types:
        # 创建一个二维 numpy 数组作为输入，数据类型由 type 决定
        input = np.array([[5, 4, 2, 5],
                          [3, 7, 8, 2],
                          [1, 5, 1, 1]], type)
        # 调用 ndimage 库中的 maximum_position 函数，返回最大值的位置信息
        output = ndimage.maximum_position(input, labels)
        # 使用 assert_equal 断言函数验证输出是否符合预期
        assert_equal(output, (1, 1))


# 定义另一个测试函数，测试 ndimage 库中的 maximum_position 函数的另一种情况
def test_maximum_position05():
    # 设定标签列表
    labels = [1, 2, 0, 4]
    # 对于 types 列表中的每种类型
    for type in types:
        # 创建一个二维 numpy 数组作为输入，数据类型由 type 决定
        input = np.array([[5, 4, 2, 5],
                          [3, 7, 8, 2],
                          [1, 5, 1, 1]], type)
        # 调用 ndimage 库中的 maximum_position 函数，返回多个最大值的位置信息
        output = ndimage.maximum_position(input, labels, 1)
        # 使用 assert_equal 断言函数验证输出是否符合预期
        assert_equal(output, (0, 0))


# 定义另一个测试函数，测试 ndimage 库中的 maximum_position 函数的另一种情况
def test_maximum_position06():
    # 设定标签列表
    labels = [1, 2, 0, 4]
    # 对于 types 列表中的每种类型
    for type in types:
        # 创建一个二维 numpy 数组作为输入，数据类型由 type 决定
        input = np.array([[5, 4, 2, 5],
                          [3, 7, 8, 2],
                          [1, 5, 1, 1]], type)
        # 调用 ndimage 库中的 maximum_position 函数，返回多个最大值的位置信息
        output = ndimage.maximum_position(input, labels,
                                          [1, 2])
        # 使用 assert_equal 断言函数验证输出的每个位置是否符合预期
        assert_equal(output[0], (0, 0))
        assert_equal(output[1], (1, 1))


# 定义另一个测试函数，测试 ndimage 库中的 maximum_position 函数的另一种情况
def test_maximum_position07():
    # 测试浮点类型的标签
    labels = np.array([1.0, 2.5, 0.0, 4.5])
    # 对于 types 列表中的每种类型
    for type in types:
        # 创建一个二维 numpy 数组作为输入，数据类型由 type 决定
        input = np.array([[5, 4, 2, 5],
                          [3, 7, 8, 2],
                          [1, 5, 1, 1]], type)
        # 调用 ndimage 库中的 maximum_position 函数，返回多个最大值的位置信息
        output = ndimage.maximum_position(input, labels,
                                          [1.0, 4.5])
        # 使用 assert_equal 断言函数验证输出的每个位置是否符合预期
        assert_equal(output[0], (0, 0))
        assert_equal(output[1], (0, 3))


# 定义一个测试函数，测试 ndimage 库中的 extrema 函数的第一种情况
def test_extrema01():
    # 创建一个布尔类型的标签数组
    labels = np.array([1, 0], bool)
    # 对于每种数据类型执行以下操作
    for type in types:
        # 创建一个二维数组，内容为 [[1, 2], [3, 4]]，数据类型为当前循环中的 type
        input = np.array([[1, 2], [3, 4]], type)
        # 计算数组的极值并返回结果
        output1 = ndimage.extrema(input, labels=labels)
        # 计算数组的最小值并返回结果
        output2 = ndimage.minimum(input, labels=labels)
        # 计算数组的最大值并返回结果
        output3 = ndimage.maximum(input, labels=labels)
        # 找到数组中的最小值位置并返回结果
        output4 = ndimage.minimum_position(input,
                                           labels=labels)
        # 找到数组中的最大值位置并返回结果
        output5 = ndimage.maximum_position(input,
                                           labels=labels)
        # 断言输出的极值和位置结果是否符合预期，即是否相等
        assert_equal(output1, (output2, output3, output4, output5))
# 测试函数，计算输入数组的极值及其位置
def test_extrema02():
    # 创建标签数组，用于标识输入数组中的区域
    labels = np.array([1, 2])
    # 遍历 types 列表中的每个类型
    for type in types:
        # 创建指定类型的输入数组
        input = np.array([[1, 2], [3, 4]], type)
        # 计算输入数组的极值，使用给定的标签数组和索引值
        output1 = ndimage.extrema(input, labels=labels,
                                  index=2)
        # 计算输入数组的最小值，使用给定的标签数组和索引值
        output2 = ndimage.minimum(input, labels=labels,
                                  index=2)
        # 计算输入数组的最大值，使用给定的标签数组和索引值
        output3 = ndimage.maximum(input, labels=labels,
                                  index=2)
        # 计算输入数组的最小值的位置，使用给定的标签数组和索引值
        output4 = ndimage.minimum_position(input,
                                           labels=labels, index=2)
        # 计算输入数组的最大值的位置，使用给定的标签数组和索引值
        output5 = ndimage.maximum_position(input,
                                           labels=labels, index=2)
        # 断言极值计算的输出是否符合预期
        assert_equal(output1, (output2, output3, output4, output5))


# 测试函数，计算输入数组的极值及其位置
def test_extrema03():
    # 创建标签数组，用于标识输入数组中的区域
    labels = np.array([[1, 2], [2, 3]])
    # 遍历 types 列表中的每个类型
    for type in types:
        # 创建指定类型的输入数组
        input = np.array([[1, 2], [3, 4]], type)
        # 计算输入数组的极值，使用给定的标签数组和多个索引值
        output1 = ndimage.extrema(input, labels=labels,
                                  index=[2, 3, 8])
        # 计算输入数组的最小值，使用给定的标签数组和多个索引值
        output2 = ndimage.minimum(input, labels=labels,
                                  index=[2, 3, 8])
        # 计算输入数组的最大值，使用给定的标签数组和多个索引值
        output3 = ndimage.maximum(input, labels=labels,
                                  index=[2, 3, 8])
        # 计算输入数组的最小值的位置，使用给定的标签数组和多个索引值
        output4 = ndimage.minimum_position(input,
                                           labels=labels, index=[2, 3, 8])
        # 计算输入数组的最大值的位置，使用给定的标签数组和多个索引值
        output5 = ndimage.maximum_position(input,
                                           labels=labels, index=[2, 3, 8])
        # 断言极值计算的输出是否符合预期
        assert_array_almost_equal(output1[0], output2)
        assert_array_almost_equal(output1[1], output3)
        assert_array_almost_equal(output1[2], output4)
        assert_array_almost_equal(output1[3], output5)


# 测试函数，计算输入数组的极值及其位置
def test_extrema04():
    # 创建标签列表，用于标识输入数组中的区域
    labels = [1, 2, 0, 4]
    # 遍历 types 列表中的每个类型
    for type in types:
        # 创建指定类型的输入数组
        input = np.array([[5, 4, 2, 5],
                          [3, 7, 8, 2],
                          [1, 5, 1, 1]], type)
        # 计算输入数组的极值，使用给定的标签列表和多个索引值
        output1 = ndimage.extrema(input, labels, [1, 2])
        # 计算输入数组的最小值，使用给定的标签列表和多个索引值
        output2 = ndimage.minimum(input, labels, [1, 2])
        # 计算输入数组的最大值，使用给定的标签列表和多个索引值
        output3 = ndimage.maximum(input, labels, [1, 2])
        # 计算输入数组的最小值的位置，使用给定的标签列表和多个索引值
        output4 = ndimage.minimum_position(input, labels,
                                           [1, 2])
        # 计算输入数组的最大值的位置，使用给定的标签列表和多个索引值
        output5 = ndimage.maximum_position(input, labels,
                                           [1, 2])
        # 断言极值计算的输出是否符合预期
        assert_array_almost_equal(output1[0], output2)
        assert_array_almost_equal(output1[1], output3)
        assert_array_almost_equal(output1[2], output4)
        assert_array_almost_equal(output1[3], output5)


# 测试函数，计算输入数组的质心
def test_center_of_mass01():
    # 期望的质心位置
    expected = [0.0, 0.0]
    # 遍历 types 列表中的每个类型
    for type in types:
        # 创建指定类型的输入数组
        input = np.array([[1, 0], [0, 0]], type)
        # 计算输入数组的质心
        output = ndimage.center_of_mass(input)
        # 断言质心计算的输出是否符合预期
        assert_array_almost_equal(output, expected)


# 测试函数，计算输入数组的质心
def test_center_of_mass02():
    # 期望的质心位置
    expected = [1, 0]
    # 遍历 types 列表中的每个类型
    for type in types:
        # 创建指定类型的输入数组
        input = np.array([[0, 0], [1, 0]], type)
        # 计算输入数组的质心
        output = ndimage.center_of_mass(input)
        # 断言质心计算的输出是否符合预期
        assert_array_almost_equal(output, expected)


# 测试函数，计算输入数组的质心
def test_center_of_mass03():
    expected = [0, 1]
    # 对于给定的每种数据类型进行循环处理
    for type in types:
        # 使用 numpy 创建一个 2x2 的数组，类型为当前循环中的 type
        input = np.array([[0, 1], [0, 0]], type)
        # 使用 ndimage.center_of_mass 计算输入数组的质心坐标
        output = ndimage.center_of_mass(input)
        # 断言计算得到的输出与预期结果的数组几乎相等
        assert_array_almost_equal(output, expected)
def test_center_of_mass04():
    expected = [1, 1]  # 预期的质心坐标
    for type in types:  # 遍历类型列表
        input = np.array([[0, 0], [0, 1]], type)  # 创建指定类型的输入数组
        output = ndimage.center_of_mass(input)  # 计算输入数组的质心
        assert_array_almost_equal(output, expected)  # 断言计算结果与预期值的近似性


def test_center_of_mass05():
    expected = [0.5, 0.5]  # 预期的质心坐标
    for type in types:  # 遍历类型列表
        input = np.array([[1, 1], [1, 1]], type)  # 创建指定类型的输入数组
        output = ndimage.center_of_mass(input)  # 计算输入数组的质心
        assert_array_almost_equal(output, expected)  # 断言计算结果与预期值的近似性


def test_center_of_mass06():
    expected = [0.5, 0.5]  # 预期的质心坐标
    input = np.array([[1, 2], [3, 1]], bool)  # 创建布尔类型的输入数组
    output = ndimage.center_of_mass(input)  # 计算输入数组的质心
    assert_array_almost_equal(output, expected)  # 断言计算结果与预期值的近似性


def test_center_of_mass07():
    labels = [1, 0]  # 标签列表
    expected = [0.5, 0.0]  # 预期的质心坐标
    input = np.array([[1, 2], [3, 1]], bool)  # 创建布尔类型的输入数组
    output = ndimage.center_of_mass(input, labels)  # 计算指定标签的输入数组的质心
    assert_array_almost_equal(output, expected)  # 断言计算结果与预期值的近似性


def test_center_of_mass08():
    labels = [1, 2]  # 标签列表
    expected = [0.5, 1.0]  # 预期的质心坐标
    input = np.array([[5, 2], [3, 1]], bool)  # 创建布尔类型的输入数组
    output = ndimage.center_of_mass(input, labels, 2)  # 计算指定标签和轴的输入数组的质心
    assert_array_almost_equal(output, expected)  # 断言计算结果与预期值的近似性


def test_center_of_mass09():
    labels = [1, 2]  # 标签列表
    expected = [(0.5, 0.0), (0.5, 1.0)]  # 预期的质心坐标列表
    input = np.array([[1, 2], [1, 1]], bool)  # 创建布尔类型的输入数组
    output = ndimage.center_of_mass(input, labels, [1, 2])  # 计算指定标签列表的输入数组的质心
    assert_array_almost_equal(output, expected)  # 断言计算结果与预期值的近似性


def test_histogram01():
    expected = np.ones(10)  # 预期的直方图数组
    input = np.arange(10)  # 创建输入数组
    output = ndimage.histogram(input, 0, 10, 10)  # 计算输入数组的直方图
    assert_array_almost_equal(output, expected)  # 断言计算结果与预期值的近似性


def test_histogram02():
    labels = [1, 1, 1, 1, 2, 2, 2, 2]  # 标签列表
    expected = [0, 2, 0, 1, 1]  # 预期的直方图数组
    input = np.array([1, 1, 3, 4, 3, 3, 3, 3])  # 创建输入数组
    output = ndimage.histogram(input, 0, 4, 5, labels, 1)  # 计算带有标签和最小像素值的直方图
    assert_array_almost_equal(output, expected)  # 断言计算结果与预期值的近似性


def test_histogram03():
    labels = [1, 0, 1, 1, 2, 2, 2, 2]  # 标签列表
    expected1 = [0, 1, 0, 1, 1]  # 预期的直方图数组1
    expected2 = [0, 0, 0, 3, 0]  # 预期的直方图数组2
    input = np.array([1, 1, 3, 4, 3, 5, 3, 3])  # 创建输入数组
    output = ndimage.histogram(input, 0, 4, 5, labels, (1, 2))  # 计算带有标签列表和轴的直方图
    assert_array_almost_equal(output[0], expected1)  # 断言计算结果与预期值的近似性
    assert_array_almost_equal(output[1], expected2)  # 断言计算结果与预期值的近似性


def test_stat_funcs_2d():
    a = np.array([[5, 6, 0, 0, 0], [8, 9, 0, 0, 0], [0, 0, 0, 3, 5]])  # 创建二维数组
    lbl = np.array([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 2, 2]])  # 创建标签数组

    mean = ndimage.mean(a, labels=lbl, index=[1, 2])  # 计算指定标签和索引的平均值
    assert_array_equal(mean, [7.0, 4.0])  # 断言计算结果与预期值的相等性

    var = ndimage.variance(a, labels=lbl, index=[1, 2])  # 计算指定标签和索引的方差
    assert_array_equal(var, [2.5, 1.0])  # 断言计算结果与预期值的相等性

    std = ndimage.standard_deviation(a, labels=lbl, index=[1, 2])  # 计算指定标签和索引的标准差
    assert_array_almost_equal(std, np.sqrt([2.5, 1.0]))  # 断言计算结果与预期值的近似性

    med = ndimage.median(a, labels=lbl, index=[1, 2])  # 计算指定标签和索引的中位数
    assert_array_equal(med, [7.0, 4.0])  # 断言计算结果与预期值的相等性

    min = ndimage.minimum(a, labels=lbl, index=[1, 2])  # 计算指定标签和索引的最小值
    assert_array_equal(min, [5, 3])  # 断言计算结果与预期值的相等性

    max = ndimage.maximum(a, labels=lbl, index=[1, 2])  # 计算指定标签和索引的最大值
    assert_array_equal(max, [9, 5])  # 断言计算结果与预期值的相等性
    # 定义一个名为 test_watershed_ift01 的测试方法
    def test_watershed_ift01(self):
        # 创建一个 8x7 的二维数组作为输入数据，数据类型为无符号8位整数
        data = np.array([[0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 1, 0, 0, 0, 1, 0],
                         [0, 1, 0, 0, 0, 1, 0],
                         [0, 1, 0, 0, 0, 1, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        # 创建一个 8x7 的二维数组作为标记（markers），数据类型为有符号8位整数
        markers = np.array([[-1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]], np.int8)
        # 调用 ndimage 模块的 watershed_ift 函数进行图像分水岭变换，使用给定的数据和标记，结构设定为3x3的全1矩阵
        out = ndimage.watershed_ift(data, markers, structure=[[1, 1, 1],
                                                              [1, 1, 1],
                                                              [1, 1, 1]])
        # 期望的输出结果，与分水岭变换后的预期标记值对比
        expected = [[-1, -1, -1, -1, -1, -1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1]]
        # 使用 NumPy 的 assert_array_almost_equal 函数进行输出结果和期望结果的比较
        assert_array_almost_equal(out, expected)

    # 定义一个名为 test_watershed_ift02 的测试方法
    def test_watershed_ift02(self):
        # 创建一个 8x7 的二维数组作为输入数据，数据类型为无符号8位整数
        data = np.array([[0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 1, 0, 0, 0, 1, 0],
                         [0, 1, 0, 0, 0, 1, 0],
                         [0, 1, 0, 0, 0, 1, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        # 创建一个 8x7 的二维数组作为标记（markers），数据类型为有符号8位整数
        markers = np.array([[-1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]], np.int8)
        # 调用 ndimage 模块的 watershed_ift 函数进行图像分水岭变换，使用给定的数据和标记
        out = ndimage.watershed_ift(data, markers)
        # 期望的输出结果，与分水岭变换后的预期标记值对比
        expected = [[-1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, 1, 1, 1, -1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, -1, 1, 1, 1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1]]
        # 使用 NumPy 的 assert_array_almost_equal 函数进行输出结果和期望结果的比较
        assert_array_almost_equal(out, expected)
    # 定义测试函数 test_watershed_ift03，用于测试图像分水岭算法的结果
    def test_watershed_ift03(self):
        # 创建一个 7x7 的二维数组作为输入数据，表示图像的像素值（0表示背景，1表示前景）
        data = np.array([[0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        # 创建一个 7x7 的二维数组作为标记（markers），用于指定图像分割的初始标记
        markers = np.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 2, 0, 3, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, -1]], np.int8)
        # 调用图像分水岭算法 ndimage.watershed_ift 进行图像分割
        out = ndimage.watershed_ift(data, markers)
        # 预期的分割结果，即预期的输出结果
        expected = [[-1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, 2, -1, 3, -1, -1],
                    [-1, 2, 2, 3, 3, 3, -1],
                    [-1, 2, 2, 3, 3, 3, -1],
                    [-1, 2, 2, 3, 3, 3, -1],
                    [-1, -1, 2, -1, 3, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1]]
        # 断言输出结果与预期结果的近似性
        assert_array_almost_equal(out, expected)

    # 定义测试函数 test_watershed_ift04，与 test_watershed_ift03 类似，用于测试带有自定义结构的图像分水岭算法的结果
    def test_watershed_ift04(self):
        # 创建一个 7x7 的二维数组作为输入数据，表示图像的像素值（0表示背景，1表示前景）
        data = np.array([[0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        # 创建一个 7x7 的二维数组作为标记（markers），用于指定图像分割的初始标记
        markers = np.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 2, 0, 3, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, -1]],
                           np.int8)
        # 调用图像分水岭算法 ndimage.watershed_ift，传入自定义的 3x3 结构作为参数进行图像分割
        out = ndimage.watershed_ift(data, markers,
                                    structure=[[1, 1, 1],
                                               [1, 1, 1],
                                               [1, 1, 1]])
        # 预期的分割结果，即预期的输出结果
        expected = [[-1, -1, -1, -1, -1, -1, -1],
                    [-1, 2, 2, 3, 3, 3, -1],
                    [-1, 2, 2, 3, 3, 3, -1],
                    [-1, 2, 2, 3, 3, 3, -1],
                    [-1, 2, 2, 3, 3, 3, -1],
                    [-1, 2, 2, 3, 3, 3, -1],
                    [-1, -1, -1, -1, -1, -1, -1]]
        # 断言输出结果与预期结果的近似性
        assert_array_almost_equal(out, expected)
    # 定义测试函数 test_watershed_ift05，用于测试图像分水岭算法
    def test_watershed_ift05(self):
        # 创建一个 7x7 的二维数组作为图像数据，数据类型为 uint8
        data = np.array([[0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        # 创建一个 7x7 的二维数组作为标记图，数据类型为 int8
        markers = np.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 3, 0, 2, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, -1]],
                           np.int8)
        # 调用 ndimage 模块中的 watershed_ift 函数，对图像数据进行分水岭算法计算
        out = ndimage.watershed_ift(data, markers,
                                    structure=[[1, 1, 1],
                                               [1, 1, 1],
                                               [1, 1, 1]])
        # 预期输出的分水岭算法结果，与计算得到的 out 数组进行比较
        expected = [[-1, -1, -1, -1, -1, -1, -1],
                    [-1, 3, 3, 2, 2, 2, -1],
                    [-1, 3, 3, 2, 2, 2, -1],
                    [-1, 3, 3, 2, 2, 2, -1],
                    [-1, 3, 3, 2, 2, 2, -1],
                    [-1, 3, 3, 2, 2, 2, -1],
                    [-1, -1, -1, -1, -1, -1, -1]]
        # 使用 assert_array_almost_equal 函数断言，验证 out 与 expected 数组近似相等
        assert_array_almost_equal(out, expected)

    # 定义测试函数 test_watershed_ift06，用于测试另一组图像分水岭算法
    def test_watershed_ift06(self):
        # 创建一个 6x7 的二维数组作为图像数据，数据类型为 uint8
        data = np.array([[0, 1, 0, 0, 0, 1, 0],
                         [0, 1, 0, 0, 0, 1, 0],
                         [0, 1, 0, 0, 0, 1, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        # 创建一个 6x7 的二维数组作为标记图，数据类型为 int8
        markers = np.array([[-1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]], np.int8)
        # 调用 ndimage 模块中的 watershed_ift 函数，对图像数据进行分水岭算法计算
        out = ndimage.watershed_ift(data, markers,
                                    structure=[[1, 1, 1],
                                               [1, 1, 1],
                                               [1, 1, 1]])
        # 预期输出的分水岭算法结果，与计算得到的 out 数组进行比较
        expected = [[-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1]]
        # 使用 assert_array_almost_equal 函数断言，验证 out 与 expected 数组近似相等
        assert_array_almost_equal(out, expected)
    def test_watershed_ift07(self):
        # 定义一个形状为 (7, 6) 的二维数组，并初始化为全零的无符号整数类型
        shape = (7, 6)
        data = np.zeros(shape, dtype=np.uint8)
        # 对数组进行转置操作
        data = data.transpose()
        # 将特定的值赋给数组的部分位置
        data[...] = np.array([[0, 1, 0, 0, 0, 1, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 1, 1, 1, 1, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        # 定义一个标记数组，包含特定的整数值
        markers = np.array([[-1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]], np.int8)
        # 定义一个形状为 (7, 6) 的二维数组，并初始化为全零的有符号整数类型
        out = np.zeros(shape, dtype=np.int16)
        # 对数组进行转置操作
        out = out.transpose()
        # 调用 ndimage 库的 watershed_ift 函数，传入数据数组、标记数组和结构数组作为参数，将结果存入 out 数组中
        ndimage.watershed_ift(data, markers,
                              structure=[[1, 1, 1],
                                         [1, 1, 1],
                                         [1, 1, 1]],
                              output=out)
        # 定义预期的结果数组
        expected = [[-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1]]
        # 使用 assert_array_almost_equal 函数断言 out 数组和预期结果数组几乎相等
        assert_array_almost_equal(out, expected)

    def test_watershed_ift08(self):
        # 测试情况：成本值大于 uint8 的情况，见 gh-10069
        # 定义一个形状为 (2, 2) 的二维数组，并初始化为指定的无符号整数类型
        data = np.array([[256, 0],
                         [0, 0]], np.uint16)
        # 定义一个形状为 (2, 2) 的二维数组，并初始化为指定的整数类型
        markers = np.array([[1, 0],
                            [0, 0]], np.int8)
        # 调用 ndimage 库的 watershed_ift 函数，传入数据数组和标记数组作为参数，返回结果数组
        out = ndimage.watershed_ift(data, markers)
        # 定义预期的结果数组
        expected = [[1, 1],
                    [1, 1]]
        # 使用 assert_array_almost_equal 函数断言 out 数组和预期结果数组几乎相等
        assert_array_almost_equal(out, expected)

    def test_watershed_ift09(self):
        # 测试情况：大成本值的情况，见 gh-19575
        # 定义一个形状为 (2, 2) 的二维数组，并初始化为指定的无符号整数类型
        data = np.array([[np.iinfo(np.uint16).max, 0],
                         [0, 0]], np.uint16)
        # 定义一个形状为 (2, 2) 的二维数组，并初始化为指定的整数类型
        markers = np.array([[1, 0],
                            [0, 0]], np.int8)
        # 调用 ndimage 库的 watershed_ift 函数，传入数据数组和标记数组作为参数，返回结果数组
        out = ndimage.watershed_ift(data, markers)
        # 定义预期的结果数组
        expected = [[1, 1],
                    [1, 1]]
        # 使用 assert_allclose 函数断言 out 数组和预期结果数组全部相等
        assert_allclose(out, expected)
# 使用 pytest 模块的 parametrize 装饰器，为测试用例提供参数化的功能，参数为 np.intc 和 np.uintc
@pytest.mark.parametrize("dt", [np.intc, np.uintc])
# 定义名为 test_gh_19423 的测试函数，参数 dt 会依次取 np.intc 和 np.uintc 两个值
def test_gh_19423(dt):
    # 使用种子为 123 的随机数生成器创建 rng 对象
    rng = np.random.default_rng(123)
    # 定义变量 max_val，并赋值为 8
    max_val = 8
    # 使用 rng 对象生成一个形状为 (10, 12) 的随机整数数组 image，数值范围为 [0, max_val)，并将其转换为指定类型 dt
    image = rng.integers(low=0, high=max_val, size=(10, 12)).astype(dtype=dt)
    # 调用 ndimage.value_indices 函数，返回图像中不同值的索引
    val_idx = ndimage.value_indices(image)
    # 断言不同值的索引的数量等于 max_val
    assert len(val_idx.keys()) == max_val
```