# `D:\src\scipysrc\scipy\scipy\ndimage\tests\test_morphology.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数组和数值计算
from numpy.testing import (assert_, assert_equal, assert_array_equal,
                           assert_array_almost_equal)  # 导入 NumPy 测试模块的断言函数

import pytest  # 导入 pytest 测试框架
from pytest import raises as assert_raises  # 导入 pytest 的 raises 断言别名

from scipy import ndimage  # 导入 SciPy 的图像处理模块

from . import types  # 导入当前包中的 types 模块

class TestNdimageMorphology:  # 定义测试类 TestNdimageMorphology

    @pytest.mark.parametrize('dtype', types)  # 使用 pytest 的参数化装饰器，参数为 types 模块中的类型

    def test_distance_transform_bf01(self, dtype):  # 定义测试方法 test_distance_transform_bf01，接受一个 dtype 参数
        # brute force (bf) distance transform
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],  # 创建一个 NumPy 数组作为测试数据
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)  # 使用 dtype 参数作为数组的数据类型

        out, ft = ndimage.distance_transform_bf(data, 'euclidean',  # 调用 SciPy 的 bf 距离变换函数
                                                return_indices=True)  # 返回变换结果和距离指数

        expected = [[0, 0, 0, 0, 0, 0, 0, 0, 0],  # 预期的距离变换结果
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 2, 4, 2, 1, 0, 0],
                    [0, 0, 1, 4, 8, 4, 1, 0, 0],
                    [0, 0, 1, 2, 4, 2, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        assert_array_almost_equal(out * out, expected)  # 断言 out 的平方与预期结果的近似相等性

        expected = [[[0, 0, 0, 0, 0, 0, 0, 0, 0],  # 预期的距离变换结果的指数
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [2, 2, 2, 2, 1, 2, 2, 2, 2],
                     [3, 3, 3, 2, 1, 2, 3, 3, 3],
                     [4, 4, 4, 4, 6, 4, 4, 4, 4],
                     [5, 5, 6, 6, 7, 6, 6, 5, 5],
                     [6, 6, 6, 7, 7, 7, 6, 6, 6],
                     [7, 7, 7, 7, 7, 7, 7, 7, 7],
                     [8, 8, 8, 8, 8, 8, 8, 8, 8]],
                    [[0, 1, 2, 3, 4, 5, 6, 7, 8],  # 第二个预期的距离变换结果的指数
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 2, 4, 6, 6, 7, 8],
                     [0, 1, 1, 2, 4, 6, 7, 7, 8],
                     [0, 1, 1, 1, 6, 7, 7, 7, 8],
                     [0, 1, 2, 2, 4, 6, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8]]]
        assert_array_almost_equal(ft, expected)  # 断言 ft 与预期结果的近似相等性

    @pytest.mark.parametrize('dtype', types)  # 继续使用 pytest 的参数化装饰器，参数为 types 模块中的类型
    # 定义测试函数，测试 distance_transform_bf 函数
    def test_distance_transform_bf02(self, dtype):
        # 创建一个二维 numpy 数组作为测试数据
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        
        # 调用 distance_transform_bf 函数计算距离变换，并返回距离变换结果和索引
        out, ft = ndimage.distance_transform_bf(data, 'cityblock',
                                                return_indices=True)
        
        # 预期的距离变换结果
        expected = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 2, 2, 2, 1, 0, 0],
                    [0, 0, 1, 2, 3, 2, 1, 0, 0],
                    [0, 0, 1, 2, 2, 2, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        
        # 断言计算得到的距离变换结果与预期结果相近
        assert_array_almost_equal(out, expected)
        
        # 预期的距离变换索引
        expected = [[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [2, 2, 2, 2, 1, 2, 2, 2, 2],
                     [3, 3, 3, 3, 1, 3, 3, 3, 3],
                     [4, 4, 4, 4, 7, 4, 4, 4, 4],
                     [5, 5, 6, 7, 7, 7, 6, 5, 5],
                     [6, 6, 6, 7, 7, 7, 6, 6, 6],
                     [7, 7, 7, 7, 7, 7, 7, 7, 7],
                     [8, 8, 8, 8, 8, 8, 8, 8, 8]],
                    [[0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 2, 4, 6, 6, 7, 8],
                     [0, 1, 1, 1, 4, 7, 7, 7, 8],
                     [0, 1, 1, 1, 4, 7, 7, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8]]]
        
        # 断言计算得到的距离变换索引与预期结果相近
        assert_array_almost_equal(expected, ft)
        
    @pytest.mark.parametrize('dtype', types)


这段代码是一个用于测试 `ndimage.distance_transform_bf` 函数的测试函数。其中包含了输入数据的定义、函数调用以及对函数输出的预期结果的断言。
    # 定义一个测试方法，用于测试 distance_transform_bf 函数
    def test_distance_transform_bf03(self, dtype):
        # 创建一个二维 numpy 数组作为测试数据
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        # 调用 ndimage 中的 distance_transform_bf 函数，使用 chessboard 距离度量，并返回距离变换结果和索引
        out, ft = ndimage.distance_transform_bf(data, 'chessboard',
                                                return_indices=True)

        # 预期的距离变换结果
        expected = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 2, 1, 1, 0, 0],
                    [0, 0, 1, 2, 2, 2, 1, 0, 0],
                    [0, 0, 1, 1, 2, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        # 断言计算得到的距离变换结果与预期结果相近
        assert_array_almost_equal(out, expected)

        # 预期的距离变换结果中的索引数组
        expected = [[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [2, 2, 2, 2, 1, 2, 2, 2, 2],
                     [3, 3, 4, 2, 2, 2, 4, 3, 3],
                     [4, 4, 5, 6, 6, 6, 5, 4, 4],
                     [5, 5, 6, 6, 7, 6, 6, 5, 5],
                     [6, 6, 6, 7, 7, 7, 6, 6, 6],
                     [7, 7, 7, 7, 7, 7, 7, 7, 7],
                     [8, 8, 8, 8, 8, 8, 8, 8, 8]],
                    [[0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 2, 5, 6, 6, 7, 8],
                     [0, 1, 1, 2, 6, 6, 7, 7, 8],
                     [0, 1, 1, 2, 6, 7, 7, 7, 8],
                     [0, 1, 2, 2, 6, 6, 7, 7, 8],
                     [0, 1, 2, 4, 5, 6, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8]]]
        # 断言计算得到的索引数组与预期结果相近
        assert_array_almost_equal(ft, expected)

    # 使用 pytest 的参数化测试，测试不同数据类型的情况
    @pytest.mark.parametrize('dtype', types)
    # 定义一个测试方法，用于测试 distance_transform_bf 函数
    def test_distance_transform_bf04(self, dtype):
        # 创建一个二维 NumPy 数组作为测试数据
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        # 调用 distance_transform_bf 函数，返回距离变换后的距离和索引
        tdt, tft = ndimage.distance_transform_bf(data, return_indices=1)
        
        # 初始化空列表，用于存储距离变换的结果
        dts = []
        fts = []
        
        # 创建一个全零的 NumPy 数组，作为距离变换的距离结果
        dt = np.zeros(data.shape, dtype=np.float64)
        
        # 调用 distance_transform_bf 函数，将距离结果存入 dt 数组
        ndimage.distance_transform_bf(data, distances=dt)
        dts.append(dt)
        
        # 调用 distance_transform_bf 函数，返回索引结果
        ft = ndimage.distance_transform_bf(
            data, return_distances=False, return_indices=1)
        fts.append(ft)
        
        # 创建一个包含所有索引的 NumPy 数组
        ft = np.indices(data.shape, dtype=np.int32)
        
        # 调用 distance_transform_bf 函数，传入自定义索引数组，返回索引结果
        ndimage.distance_transform_bf(
            data, return_distances=False, return_indices=True, indices=ft)
        fts.append(ft)
        
        # 再次调用 distance_transform_bf 函数，返回距离和索引
        dt, ft = ndimage.distance_transform_bf(
            data, return_indices=1)
        dts.append(dt)
        fts.append(ft)
        
        # 创建一个全零的 NumPy 数组，作为距离结果
        dt = np.zeros(data.shape, dtype=np.float64)
        
        # 调用 distance_transform_bf 函数，将距离结果存入 dt 数组，并返回索引
        ft = ndimage.distance_transform_bf(
            data, distances=dt, return_indices=True)
        dts.append(dt)
        fts.append(ft)
        
        # 创建一个包含所有索引的 NumPy 数组
        ft = np.indices(data.shape, dtype=np.int32)
        
        # 调用 distance_transform_bf 函数，传入自定义距离和索引数组，返回距离结果
        dt = ndimage.distance_transform_bf(
            data, return_indices=True, indices=ft)
        dts.append(dt)
        fts.append(ft)
        
        # 再次创建全零的 NumPy 数组，作为距离结果
        dt = np.zeros(data.shape, dtype=np.float64)
        
        # 创建一个包含所有索引的 NumPy 数组
        ft = np.indices(data.shape, dtype=np.int32)
        
        # 调用 distance_transform_bf 函数，传入自定义距离和索引数组，返回距离结果
        ndimage.distance_transform_bf(
            data, distances=dt, return_indices=True, indices=ft)
        dts.append(dt)
        fts.append(ft)
        
        # 遍历所有距离结果，检查是否与 tdt 相等
        for dt in dts:
            assert_array_almost_equal(tdt, dt)
        
        # 遍历所有索引结果，检查是否与 tft 相等
        for ft in fts:
            assert_array_almost_equal(tft, ft)
    # 定义测试方法，测试 distance_transform_bf 函数的行为
    def test_distance_transform_bf05(self, dtype):
        # 创建一个二维 NumPy 数组作为测试数据
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        # 调用 distance_transform_bf 函数，计算距离变换，并返回结果和距离场的索引
        out, ft = ndimage.distance_transform_bf(
            data, 'euclidean', return_indices=True, sampling=[2, 2])
        # 期望的距离变换结果
        expected = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 4, 4, 4, 0, 0, 0],
                    [0, 0, 4, 8, 16, 8, 4, 0, 0],
                    [0, 0, 4, 16, 32, 16, 4, 0, 0],
                    [0, 0, 4, 8, 16, 8, 4, 0, 0],
                    [0, 0, 0, 4, 4, 4, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        # 断言距离变换的输出是否与期望值几乎相等
        assert_array_almost_equal(out * out, expected)

        # 期望的距离场的索引
        expected = [[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [2, 2, 2, 2, 1, 2, 2, 2, 2],
                     [3, 3, 3, 2, 1, 2, 3, 3, 3],
                     [4, 4, 4, 4, 6, 4, 4, 4, 4],
                     [5, 5, 6, 6, 7, 6, 6, 5, 5],
                     [6, 6, 6, 7, 7, 7, 6, 6, 6],
                     [7, 7, 7, 7, 7, 7, 7, 7, 7],
                     [8, 8, 8, 8, 8, 8, 8, 8, 8]],
                    [[0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 2, 4, 6, 6, 7, 8],
                     [0, 1, 1, 2, 4, 6, 7, 7, 8],
                     [0, 1, 1, 1, 6, 7, 7, 7, 8],
                     [0, 1, 2, 2, 4, 6, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8]]]
        # 断言距离场的索引是否与期望值几乎相等
        assert_array_almost_equal(ft, expected)

    # 使用参数化装饰器，以支持多种数据类型的测试
    @pytest.mark.parametrize('dtype', types)
    # 定义测试函数 test_distance_transform_bf06，接受一个参数 dtype
    def test_distance_transform_bf06(self, dtype):
        # 创建一个二维 NumPy 数组作为测试数据
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        # 调用 ndimage.distance_transform_bf 函数进行距离变换，返回距离变换结果和特征跟踪数组
        out, ft = ndimage.distance_transform_bf(
            data, 'euclidean', return_indices=True, sampling=[2, 1])
        # 预期的距离变换结果
        expected = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 4, 1, 0, 0, 0],
                    [0, 0, 1, 4, 8, 4, 1, 0, 0],
                    [0, 0, 1, 4, 9, 4, 1, 0, 0],
                    [0, 0, 1, 4, 8, 4, 1, 0, 0],
                    [0, 0, 0, 1, 4, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        # 使用 assert_array_almost_equal 断言函数检查距离变换结果是否与预期相符
        assert_array_almost_equal(out * out, expected)

        # 预期的特征跟踪数组
        expected = [[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [2, 2, 2, 2, 2, 2, 2, 2, 2],
                     [3, 3, 3, 3, 2, 3, 3, 3, 3],
                     [4, 4, 4, 4, 4, 4, 4, 4, 4],
                     [5, 5, 5, 5, 6, 5, 5, 5, 5],
                     [6, 6, 6, 6, 7, 6, 6, 6, 6],
                     [7, 7, 7, 7, 7, 7, 7, 7, 7],
                     [8, 8, 8, 8, 8, 8, 8, 8, 8]],
                    [[0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 2, 6, 6, 6, 7, 8],
                     [0, 1, 1, 1, 6, 7, 7, 7, 8],
                     [0, 1, 1, 1, 7, 7, 7, 7, 8],
                     [0, 1, 1, 1, 6, 7, 7, 7, 8],
                     [0, 1, 2, 2, 4, 6, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8]]]
        # 使用 assert_array_almost_equal 断言函数检查特征跟踪数组是否与预期相符
        assert_array_almost_equal(ft, expected)

    # 定义测试函数 test_distance_transform_bf07
    def test_distance_transform_bf07(self):
        # 测试输入数据的有效性，根据 PR #13302 中的讨论
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        # 使用 assert_raises 检查是否引发 RuntimeError 异常
        with assert_raises(RuntimeError):
            ndimage.distance_transform_bf(
                data, return_distances=False, return_indices=False
            )

    # 使用 pytest 的参数化装饰器，参数化 dtype 参数
    @pytest.mark.parametrize('dtype', types)
    # 定义一个测试函数，用于测试距离变换中的城市块距离类型（cdt01）
    def test_distance_transform_cdt01(self, dtype):
        # 创建一个二维NumPy数组作为测试数据，表示一个二值图像
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        # 进行城市块距离变换，返回变换结果和距离变换的索引
        out, ft = ndimage.distance_transform_cdt(
            data, 'cityblock', return_indices=True)
        # 使用暴力方法计算城市块距离变换作为对照
        bf = ndimage.distance_transform_bf(data, 'cityblock')
        # 断言变换结果与暴力计算结果近似相等
        assert_array_almost_equal(bf, out)

        # 预期的距离变换结果，包括两个二维数组的列表
        expected = [[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [2, 2, 2, 1, 1, 1, 2, 2, 2],
                     [3, 3, 2, 1, 1, 1, 2, 3, 3],
                     [4, 4, 4, 4, 1, 4, 4, 4, 4],
                     [5, 5, 5, 5, 7, 7, 6, 5, 5],
                     [6, 6, 6, 6, 7, 7, 6, 6, 6],
                     [7, 7, 7, 7, 7, 7, 7, 7, 7],
                     [8, 8, 8, 8, 8, 8, 8, 8, 8]],
                    [[0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 1, 1, 4, 7, 7, 7, 8],
                     [0, 1, 1, 1, 4, 5, 6, 7, 8],
                     [0, 1, 2, 2, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8]]]
        # 断言距离变换的索引与预期结果近似相等
        assert_array_almost_equal(ft, expected)

    # 使用参数化测试，对不同数据类型执行上述测试函数
    @pytest.mark.parametrize('dtype', types)
    # 定义测试方法，用于测试 distance_transform_cdt 函数的行为
    def test_distance_transform_cdt02(self, dtype):
        # 创建一个二维数组作为测试数据
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        # 使用 distance_transform_cdt 函数计算距离变换，并返回结果和特征变换图像
        out, ft = ndimage.distance_transform_cdt(data, 'chessboard',
                                                 return_indices=True)
        # 使用 distance_transform_bf 函数计算距离变换的基准值
        bf = ndimage.distance_transform_bf(data, 'chessboard')
        # 断言计算结果与基准值的近似相等性
        assert_array_almost_equal(bf, out)

        # 预期的特征变换结果，包含两个子数组
        expected = [[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [2, 2, 2, 1, 1, 1, 2, 2, 2],
                     [3, 3, 2, 2, 1, 2, 2, 3, 3],
                     [4, 4, 3, 2, 2, 2, 3, 4, 4],
                     [5, 5, 4, 6, 7, 6, 4, 5, 5],
                     [6, 6, 6, 6, 7, 7, 6, 6, 6],
                     [7, 7, 7, 7, 7, 7, 7, 7, 7],
                     [8, 8, 8, 8, 8, 8, 8, 8, 8]],
                    [[0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 2, 3, 4, 6, 7, 8],
                     [0, 1, 1, 2, 2, 6, 6, 7, 8],
                     [0, 1, 1, 1, 2, 6, 7, 7, 8],
                     [0, 1, 1, 2, 6, 6, 7, 7, 8],
                     [0, 1, 2, 2, 5, 6, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8]]]
        # 断言特征变换结果与预期结果的近似相等性
        assert_array_almost_equal(ft, expected)

    # 使用 pytest 的参数化装饰器，为测试方法传递不同的数据类型进行多次测试
    @pytest.mark.parametrize('dtype', types)
    # 定义测试函数，用于测试 distance_transform_cdt 函数的多个参数组合和返回值
    def test_distance_transform_cdt03(self, dtype):
        # 创建二维 NumPy 数组作为测试数据
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        # 调用 distance_transform_cdt 函数，返回距离变换的结果和指数
        tdt, tft = ndimage.distance_transform_cdt(data, return_indices=True)

        # 初始化空列表用于存储各种参数组合下的距离变换结果
        dts = []
        fts = []

        # 创建与输入数据形状相同的全零数组，用作存储距离变换的距离信息的目标数组
        dt = np.zeros(data.shape, dtype=np.int32)
        # 执行距离变换，将结果存储在 dt 中，并将 dt 添加到 dts 列表中
        ndimage.distance_transform_cdt(data, distances=dt)
        dts.append(dt)

        # 执行距离变换，返回不带距离信息的指数，并将结果添加到 fts 列表中
        ft = ndimage.distance_transform_cdt(
            data, return_distances=False, return_indices=True)
        fts.append(ft)

        # 创建与输入数据形状相同的索引数组，并将其添加到 fts 列表中
        ft = np.indices(data.shape, dtype=np.int32)
        # 执行距离变换，将结果存储在 ft 中，并将 ft 添加到 fts 列表中
        ndimage.distance_transform_cdt(
            data, return_distances=False, return_indices=True, indices=ft)
        fts.append(ft)

        # 执行距离变换，返回距离和指数，并将结果添加到 dts 和 fts 列表中
        dt, ft = ndimage.distance_transform_cdt(data, return_indices=True)
        dts.append(dt)
        fts.append(ft)

        # 创建与输入数据形状相同的全零数组，用作存储距离变换的距离信息的目标数组
        dt = np.zeros(data.shape, dtype=np.int32)
        # 执行距离变换，将结果存储在 dt 中，并将 dt 添加到 dts 列表中
        ft = ndimage.distance_transform_cdt(
            data, distances=dt, return_indices=True)
        dts.append(dt)
        fts.append(ft)

        # 创建与输入数据形状相同的索引数组，并将其添加到 fts 列表中
        ft = np.indices(data.shape, dtype=np.int32)
        # 执行距离变换，将结果存储在 dt 中，并将 dt 添加到 dts 列表中
        ndimage.distance_transform_cdt(
            data, return_indices=True, indices=ft)
        fts.append(ft)

        # 创建与输入数据形状相同的全零数组，用作存储距离变换的距离信息的目标数组
        dt = np.zeros(data.shape, dtype=np.int32)
        # 创建与输入数据形状相同的索引数组，并将其添加到 fts 列表中
        ft = np.indices(data.shape, dtype=np.int32)
        # 执行距离变换，将结果存储在 dt 和 ft 中，并将它们分别添加到 dts 和 fts 列表中
        ndimage.distance_transform_cdt(data, distances=dt,
                                       return_indices=True, indices=ft)
        dts.append(dt)
        fts.append(ft)

        # 遍历所有距离变换结果，确保其与预期结果 tdt 和 tft 相近
        for dt in dts:
            assert_array_almost_equal(tdt, dt)
        for ft in fts:
            assert_array_almost_equal(tft, ft)

    # 定义输入验证的测试函数，用于测试 distance_transform_bf 函数的异常情况
    def test_distance_transform_cdt04(self):
        # 准备测试数据，一个二维 NumPy 数组
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]])

        # 创建与输入数据形状相同的全零数组，用作存储距离变换的索引数组
        indices_out = np.zeros((data.ndim,) + data.shape, dtype=np.int32)

        # 使用 assert_raises 检查 distance_transform_bf 在给定参数下是否引发了 RuntimeError 异常
        with assert_raises(RuntimeError):
            ndimage.distance_transform_bf(
                data,
                return_distances=True,
                return_indices=False,
                indices=indices_out
            )

    # 使用 pytest 的参数化标记，针对测试函数 test_distance_transform_cdt03 中的参数 dtype 进行参数化测试
    @pytest.mark.parametrize('dtype', types)
    def test_distance_transform_cdt05(self, dtype):
        # 测试距离转换函数 distance_transform_cdt，使用自定义的度量类型，根据问题讨论的 issue #17381
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        # 定义距离转换的度量参数
        metric_arg = np.ones((3, 3))
        # 调用 distance_transform_cdt 进行距离转换
        actual = ndimage.distance_transform_cdt(data, metric=metric_arg)
        # 断言距离转换结果的总和为 -21
        assert actual.sum() == -21

    @pytest.mark.parametrize('dtype', types)
    def test_distance_transform_edt01(self, dtype):
        # 欧氏距离转换 (edt)
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        # 使用 return_indices=True 调用 distance_transform_edt，返回距离转换结果和索引
        out, ft = ndimage.distance_transform_edt(data, return_indices=True)
        # 使用 brute-force 方法计算欧氏距离转换
        bf = ndimage.distance_transform_bf(data, 'euclidean')
        # 断言 bf 与 out 的数组几乎相等
        assert_array_almost_equal(bf, out)

        # 计算 ft 与其索引的差值，并将其平方化
        dt = ft - np.indices(ft.shape[1:], dtype=ft.dtype)
        dt = dt.astype(np.float64)
        np.multiply(dt, dt, dt)
        dt = np.add.reduce(dt, axis=0)
        np.sqrt(dt, dt)

        # 断言 bf 与 dt 的数组几乎相等
        assert_array_almost_equal(bf, dt)

    @pytest.mark.parametrize('dtype', types)
    # 定义一个测试方法，用于测试距离转换函数 distance_transform_edt 的不同用例
    def test_distance_transform_edt02(self, dtype):
        # 创建一个二维 numpy 数组作为测试数据
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        # 调用 ndimage.distance_transform_edt 函数，返回距离变换的结果和转换后的坐标
        tdt, tft = ndimage.distance_transform_edt(data, return_indices=True)
        
        # 初始化空列表用于存储不同的距离变换结果和坐标
        dts = []
        fts = []
        
        # 创建一个全零的 numpy 数组，作为存储距离变换结果的容器
        dt = np.zeros(data.shape, dtype=np.float64)
        
        # 调用 ndimage.distance_transform_edt 函数，将距离变换结果存储到 dt 中，并添加到 dts 列表
        ndimage.distance_transform_edt(data, distances=dt)
        dts.append(dt)
        
        # 调用 ndimage.distance_transform_edt 函数，返回转换后的坐标，不返回距离，并添加到 fts 列表
        ft = ndimage.distance_transform_edt(
            data, return_distances=0, return_indices=True)
        fts.append(ft)
        
        # 使用 np.indices 函数创建一个索引数组，并添加到 fts 列表
        ft = np.indices(data.shape, dtype=np.int32)
        ndimage.distance_transform_edt(
            data, return_distances=False, return_indices=True, indices=ft)
        fts.append(ft)
        
        # 调用 ndimage.distance_transform_edt 函数，返回距离变换结果和转换后的坐标，并添加到对应列表
        dt, ft = ndimage.distance_transform_edt(
            data, return_indices=True)
        dts.append(dt)
        fts.append(ft)
        
        # 创建一个全零的 numpy 数组，作为存储距离变换结果的容器
        dt = np.zeros(data.shape, dtype=np.float64)
        
        # 调用 ndimage.distance_transform_edt 函数，将距离变换结果存储到 dt 中，并添加到 dts 列表
        ft = ndimage.distance_transform_edt(
            data, distances=dt, return_indices=True)
        dts.append(dt)
        fts.append(ft)
        
        # 使用 np.indices 函数创建一个索引数组，并添加到 fts 列表
        ft = np.indices(data.shape, dtype=np.int32)
        dt = ndimage.distance_transform_edt(
            data, return_indices=True, indices=ft)
        dts.append(dt)
        fts.append(ft)
        
        # 创建一个全零的 numpy 数组，作为存储距离变换结果的容器
        dt = np.zeros(data.shape, dtype=np.float64)
        
        # 使用 np.indices 函数创建一个索引数组，并添加到 fts 列表
        ft = np.indices(data.shape, dtype=np.int32)
        
        # 调用 ndimage.distance_transform_edt 函数，将距离变换结果存储到 dt 中，并添加到 dts 列表
        ndimage.distance_transform_edt(
            data, distances=dt, return_indices=True, indices=ft)
        dts.append(dt)
        fts.append(ft)
        
        # 遍历所有距离变换结果，与预期结果 tdt 做近似数组相等性断言
        for dt in dts:
            assert_array_almost_equal(tdt, dt)
        
        # 遍历所有坐标变换结果，与预期结果 tft 做近似数组相等性断言
        for ft in fts:
            assert_array_almost_equal(tft, ft)

    # 使用 pytest 的参数化标记，对 distance_transform_edt03 方法进行参数化测试
    @pytest.mark.parametrize('dtype', types)
    def test_distance_transform_edt03(self, dtype):
        # 创建一个二维 numpy 数组作为测试数据
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        
        # 使用 brute-force 方法计算参考距离变换结果
        ref = ndimage.distance_transform_bf(data, 'euclidean', sampling=[2, 2])
        
        # 使用 edt 方法计算距离变换结果
        out = ndimage.distance_transform_edt(data, sampling=[2, 2])
        
        # 断言距离变换结果近似相等
        assert_array_almost_equal(ref, out)
    def test_distance_transform_edt4(self, dtype):
        # 创建一个二维的 numpy 数组作为测试数据，表示一个二值图像
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        # 使用暴力方法计算欧几里得距离变换的参考结果
        ref = ndimage.distance_transform_bf(data, 'euclidean', sampling=[2, 1])
        # 使用快速方法计算欧几里得距离变换的输出结果
        out = ndimage.distance_transform_edt(data, sampling=[2, 1])
        # 断言两种方法的输出结果应该几乎相等
        assert_array_almost_equal(ref, out)

    def test_distance_transform_edt5(self):
        # 确认 Ticket #954 的回归测试
        out = ndimage.distance_transform_edt(False)
        assert_array_almost_equal(out, [0.])

    def test_distance_transform_edt6(self):
        # 根据 PR #13302 上的讨论进行输入验证测试
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        distances_out = np.zeros(data.shape, dtype=np.float64)
        # 使用 assert_raises 来检查是否会引发 RuntimeError
        with assert_raises(RuntimeError):
            ndimage.distance_transform_bf(
                data,
                return_indices=True,
                return_distances=False,
                distances=distances_out
            )

    def test_generate_structure01(self):
        # 生成一个二进制结构元素，维度为 0，形状为 1
        struct = ndimage.generate_binary_structure(0, 1)
        assert_array_almost_equal(struct, 1)

    def test_generate_structure02(self):
        # 生成一个二进制结构元素，维度为 1，形状为 [1, 1, 1]
        struct = ndimage.generate_binary_structure(1, 1)
        assert_array_almost_equal(struct, [1, 1, 1])

    def test_generate_structure03(self):
        # 生成一个二进制结构元素，维度为 2，形状为 [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        struct = ndimage.generate_binary_structure(2, 1)
        assert_array_almost_equal(struct, [[0, 1, 0],
                                           [1, 1, 1],
                                           [0, 1, 0]])

    def test_generate_structure04(self):
        # 生成一个二进制结构元素，维度为 2，形状为 [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        struct = ndimage.generate_binary_structure(2, 2)
        assert_array_almost_equal(struct, [[1, 1, 1],
                                           [1, 1, 1],
                                           [1, 1, 1]])
    def test_iterate_structure01(self):
        # 定义一个二维结构数组作为测试输入
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        # 调用 ndimage.iterate_structure 函数，对结构进行迭代，迭代深度为2
        out = ndimage.iterate_structure(struct, 2)
        # 断言迭代结果与预期的数组几乎相等
        assert_array_almost_equal(out, [[0, 0, 1, 0, 0],
                                        [0, 1, 1, 1, 0],
                                        [1, 1, 1, 1, 1],
                                        [0, 1, 1, 1, 0],
                                        [0, 0, 1, 0, 0]])

    def test_iterate_structure02(self):
        # 定义另一个二维结构数组作为测试输入
        struct = [[0, 1],
                  [1, 1],
                  [0, 1]]
        # 调用 ndimage.iterate_structure 函数，对结构进行迭代，迭代深度为2
        out = ndimage.iterate_structure(struct, 2)
        # 断言迭代结果与预期的数组几乎相等
        assert_array_almost_equal(out, [[0, 0, 1],
                                        [0, 1, 1],
                                        [1, 1, 1],
                                        [0, 1, 1],
                                        [0, 0, 1]])

    def test_iterate_structure03(self):
        # 定义一个二维结构数组作为测试输入
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        # 调用 ndimage.iterate_structure 函数，对结构进行迭代，迭代深度为2，边界值为1
        out = ndimage.iterate_structure(struct, 2, 1)
        # 定义预期的迭代结果数组
        expected = [[0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0]]
        # 断言迭代结果的第一个元素与预期的数组几乎相等
        assert_array_almost_equal(out[0], expected)
        # 断言迭代结果的第二个元素为 [2, 2]
        assert_equal(out[1], [2, 2])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion01(self, dtype):
        # 创建一个元素为空的一维数组，并使用 ndimage.binary_erosion 进行处理
        data = np.ones([], dtype)
        out = ndimage.binary_erosion(data)
        # 断言处理结果几乎与预期值 1 相等
        assert_array_almost_equal(out, 1)

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion02(self, dtype):
        # 创建一个元素为空的一维数组，并使用 ndimage.binary_erosion 进行处理，边界值设为1
        data = np.ones([], dtype)
        out = ndimage.binary_erosion(data, border_value=1)
        # 断言处理结果几乎与预期值 1 相等
        assert_array_almost_equal(out, 1)

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion03(self, dtype):
        # 创建一个包含一个元素的一维数组，并使用 ndimage.binary_erosion 进行处理
        data = np.ones([1], dtype)
        out = ndimage.binary_erosion(data)
        # 断言处理结果几乎与预期值 [0] 相等
        assert_array_almost_equal(out, [0])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion04(self, dtype):
        # 创建一个包含一个元素的一维数组，并使用 ndimage.binary_erosion 进行处理，边界值设为1
        data = np.ones([1], dtype)
        out = ndimage.binary_erosion(data, border_value=1)
        # 断言处理结果几乎与预期值 [1] 相等
        assert_array_almost_equal(out, [1])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion05(self, dtype):
        # 创建一个包含三个元素的一维数组，并使用 ndimage.binary_erosion 进行处理
        data = np.ones([3], dtype)
        out = ndimage.binary_erosion(data)
        # 断言处理结果几乎与预期值 [0, 1, 0] 相等
        assert_array_almost_equal(out, [0, 1, 0])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion06(self, dtype):
        # 创建一个包含三个元素的一维数组，并使用 ndimage.binary_erosion 进行处理，边界值设为1
        data = np.ones([3], dtype)
        out = ndimage.binary_erosion(data, border_value=1)
        # 断言处理结果几乎与预期值 [1, 1, 1] 相等
        assert_array_almost_equal(out, [1, 1, 1])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion07(self, dtype):
        # 创建一个包含五个元素的一维数组，并使用 ndimage.binary_erosion 进行处理
        data = np.ones([5], dtype)
        out = ndimage.binary_erosion(data)
        # 断言处理结果几乎与预期值 [0, 1, 1, 1, 0] 相等
        assert_array_almost_equal(out, [0, 1, 1, 1, 0])

    @pytest.mark.parametrize('dtype', types)
    # 定义一个名为 test_binary_erosion08 的测试方法，接受一个 dtype 参数
    def test_binary_erosion08(self, dtype):
        # 创建一个包含五个元素的全为1的数组，并指定数据类型为参数 dtype
        data = np.ones([5], dtype)
        # 对数据进行二进制侵蚀操作，指定边界值为1，结果存储在 out 中
        out = ndimage.binary_erosion(data, border_value=1)
        # 断言输出数组 out 与期望的结果 [1, 1, 1, 1, 1] 几乎相等
        assert_array_almost_equal(out, [1, 1, 1, 1, 1])

    # 使用 pytest 的参数化标记，对 test_binary_erosion09 方法进行参数化
    @pytest.mark.parametrize('dtype', types)
    # 定义一个名为 test_binary_erosion09 的测试方法，接受一个 dtype 参数
    def test_binary_erosion09(self, dtype):
        # 创建一个包含五个元素的全为1的数组，并指定数据类型为参数 dtype
        data = np.ones([5], dtype)
        # 将数组中索引为2的元素设置为0
        data[2] = 0
        # 对数据进行二进制侵蚀操作，结果存储在 out 中
        out = ndimage.binary_erosion(data)
        # 断言输出数组 out 与期望的结果 [0, 0, 0, 0, 0] 几乎相等
        assert_array_almost_equal(out, [0, 0, 0, 0, 0])

    # 使用 pytest 的参数化标记，对 test_binary_erosion10 方法进行参数化
    @pytest.mark.parametrize('dtype', types)
    # 定义一个名为 test_binary_erosion10 的测试方法，接受一个 dtype 参数
    def test_binary_erosion10(self, dtype):
        # 创建一个包含五个元素的全为1的数组，并指定数据类型为参数 dtype
        data = np.ones([5], dtype)
        # 将数组中索引为2的元素设置为0
        data[2] = 0
        # 对数据进行二进制侵蚀操作，指定边界值为1，结果存储在 out 中
        out = ndimage.binary_erosion(data, border_value=1)
        # 断言输出数组 out 与期望的结果 [1, 0, 0, 0, 1] 几乎相等
        assert_array_almost_equal(out, [1, 0, 0, 0, 1])

    # 使用 pytest 的参数化标记，对 test_binary_erosion11 方法进行参数化
    @pytest.mark.parametrize('dtype', types)
    # 定义一个名为 test_binary_erosion11 的测试方法，接受一个 dtype 参数
    def test_binary_erosion11(self, dtype):
        # 创建一个包含五个元素的全为1的数组，并指定数据类型为参数 dtype
        data = np.ones([5], dtype)
        # 将数组中索引为2的元素设置为0
        data[2] = 0
        # 定义一个结构元素为 [1, 0, 1]
        struct = [1, 0, 1]
        # 对数据进行二进制侵蚀操作，指定结构元素和边界值为1，结果存储在 out 中
        out = ndimage.binary_erosion(data, struct, border_value=1)
        # 断言输出数组 out 与期望的结果 [1, 0, 1, 0, 1] 几乎相等
        assert_array_almost_equal(out, [1, 0, 1, 0, 1])

    # 使用 pytest 的参数化标记，对 test_binary_erosion12 方法进行参数化
    @pytest.mark.parametrize('dtype', types)
    # 定义一个名为 test_binary_erosion12 的测试方法，接受一个 dtype 参数
    def test_binary_erosion12(self, dtype):
        # 创建一个包含五个元素的全为1的数组，并指定数据类型为参数 dtype
        data = np.ones([5], dtype)
        # 将数组中索引为2的元素设置为0
        data[2] = 0
        # 定义一个结构元素为 [1, 0, 1]
        struct = [1, 0, 1]
        # 对数据进行二进制侵蚀操作，指定结构元素、边界值为1和原点为-1，结果存储在 out 中
        out = ndimage.binary_erosion(data, struct, border_value=1, origin=-1)
        # 断言输出数组 out 与期望的结果 [0, 1, 0, 1, 1] 几乎相等
        assert_array_almost_equal(out, [0, 1, 0, 1, 1])

    # 使用 pytest 的参数化标记，对 test_binary_erosion13 方法进行参数化
    @pytest.mark.parametrize('dtype', types)
    # 定义一个名为 test_binary_erosion13 的测试方法，接受一个 dtype 参数
    def test_binary_erosion13(self, dtype):
        # 创建一个包含五个元素的全为1的数组，并指定数据类型为参数 dtype
        data = np.ones([5], dtype)
        # 将数组中索引为2的元素设置为0
        data[2] = 0
        # 定义一个结构元素为 [1, 0, 1]
        struct = [1, 0, 1]
        # 对数据进行二进制侵蚀操作，指定结构元素、边界值为1和原点为1，结果存储在 out 中
        out = ndimage.binary_erosion(data, struct, border_value=1, origin=1)
        # 断言输出数组 out 与期望的结果 [1, 1, 0, 1, 0] 几乎相等
        assert_array_almost_equal(out, [1, 1, 0, 1, 0])

    # 使用 pytest 的参数化标记，对 test_binary_erosion14 方法进行参数化
    @pytest.mark.parametrize('dtype', types)
    # 定义一个名为 test_binary_erosion14 的测试方法，接受一个 dtype 参数
    def test_binary_erosion14(self, dtype):
        # 创建一个包含五个元素的全为1的数组，并指定数据类型为参数 dtype
        data = np.ones([5], dtype)
        # 将数组中索引为2的元素设置为0
        data[2] = 0
        # 定义一个结构元素为 [1, 1]
        struct = [1, 1]
        # 对数据进行二进制侵蚀操作，指定结构元素和边界值为1，结果存储在 out 中
        out = ndimage.binary_erosion(data, struct, border_value=1)
        # 断言输出数组 out 与期望的结果 [1, 1, 0, 0, 1] 几乎相等
        assert_array_almost_equal(out, [1, 1, 0, 0, 1])

    # 使用 pytest 的参数化标记，对 test_binary_erosion15 方法进行参数化
    @pytest.mark.parametrize('dtype', types)
    # 定义一个名为 test_binary_erosion15 的测试方法，接受一个 dtype 参数
    def test_binary_erosion15(self, dtype):
        # 创建一个包含五个元素的全为1的数组，并指定数据类型为参数 dtype
        data = np.ones([5], dtype)
        # 将数组中索引为2的元素设置为0
        data[2] = 0
        # 定义一个结构元素为 [1, 1]
        struct = [1, 1]
        # 对数据进行二进制侵蚀操作，指定结构元素、边界值为1和原点为-1，结果存储在 out 中
        out = ndimage.binary_erosion(data, struct, border_value=1, origin=-1)
        # 断言输出数组 out 与期望的结果 [1, 0, 0, 1, 1] 几乎相等
        assert_array_almost_equal(out, [1, 0, 0, 1, 1])

    # 使用 pytest 的参数化标
    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion19(self, dtype):
        # 创建一个形状为 (1, 3) 的数组，元素类型为 dtype，所有元素为 1
        data = np.ones([1, 3], dtype)
        # 对二进制数组进行侵蚀操作，使用边界值为 1
        out = ndimage.binary_erosion(data, border_value=1)
        # 断言输出结果与期望值 [[1, 1, 1]] 近似相等
        assert_array_almost_equal(out, [[1, 1, 1]])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion20(self, dtype):
        # 创建一个形状为 (3, 3) 的数组，元素类型为 dtype，所有元素为 1
        data = np.ones([3, 3], dtype)
        # 对二进制数组进行侵蚀操作，默认边界值为 0
        out = ndimage.binary_erosion(data)
        # 断言输出结果与期望值 [[0, 0, 0], [0, 1, 0], [0, 0, 0]] 近似相等
        assert_array_almost_equal(out, [[0, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 0]])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion21(self, dtype):
        # 创建一个形状为 (3, 3) 的数组，元素类型为 dtype，所有元素为 1
        data = np.ones([3, 3], dtype)
        # 对二进制数组进行侵蚀操作，使用边界值为 1
        out = ndimage.binary_erosion(data, border_value=1)
        # 断言输出结果与期望值 [[1, 1, 1], [1, 1, 1], [1, 1, 1]] 近似相等
        assert_array_almost_equal(out, [[1, 1, 1],
                                        [1, 1, 1],
                                        [1, 1, 1]])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion22(self, dtype):
        # 期望的二维数组输出
        expected = [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        # 创建一个指定形状和元素类型的二维数组
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 1, 1, 1, 1, 1, 1],
                         [0, 0, 1, 1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 1, 0],
                         [0, 1, 1, 0, 0, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        # 对二维数组进行侵蚀操作，使用边界值为 1
        out = ndimage.binary_erosion(data, border_value=1)
        # 断言输出结果与期望值 expected 近似相等
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion23(self, dtype):
        # 生成一个二进制结构元素
        struct = ndimage.generate_binary_structure(2, 2)
        # 期望的二维数组输出
        expected = [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        # 创建一个指定形状和元素类型的二维数组
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 1, 1, 1, 1, 1, 1],
                         [0, 0, 1, 1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 1, 0],
                         [0, 1, 1, 0, 0, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        # 对二维数组进行侵蚀操作，使用指定的结构元素和边界值为 1
        out = ndimage.binary_erosion(data, struct, border_value=1)
        # 断言输出结果与期望值 expected 近似相等
        assert_array_almost_equal(out, expected)
    # 测试函数，用于测试二进制侵蚀函数 binary_erosion
    def test_binary_erosion24(self, dtype):
        # 结构元素定义，表示二维数组的结构
        struct = [[0, 1],
                  [1, 1]]
        # 预期输出结果，表示二维数组的预期结果
        expected = [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        # 输入数据，表示要进行二进制侵蚀的二维数组数据
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 1, 1, 1, 1, 1, 1],
                         [0, 0, 1, 1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 1, 0],
                         [0, 1, 1, 0, 0, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        # 使用 ndimage 库中的 binary_erosion 函数进行二进制侵蚀操作，设定边界值为 1
        out = ndimage.binary_erosion(data, struct, border_value=1)
        # 断言输出结果与预期结果几乎相等
        assert_array_almost_equal(out, expected)
    
    @pytest.mark.parametrize('dtype', types)
    # 参数化测试函数，用于测试不同数据类型的二进制侵蚀效果
    def test_binary_erosion25(self, dtype):
        # 结构元素定义，表示二维数组的结构
        struct = [[0, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0]]
        # 预期输出结果，表示二维数组的预期结果
        expected = [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        # 输入数据，表示要进行二进制侵蚀的二维数组数据
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 1, 1, 1, 0, 1, 1],
                         [0, 0, 1, 0, 1, 1, 0, 0],
                         [0, 1, 0, 1, 1, 1, 1, 0],
                         [0, 1, 1, 0, 0, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        # 使用 ndimage 库中的 binary_erosion 函数进行二进制侵蚀操作，设定边界值为 1
        out = ndimage.binary_erosion(data, struct, border_value=1)
        # 断言输出结果与预期结果几乎相等
        assert_array_almost_equal(out, expected)
    # 定义测试函数，用于测试二进制侵蚀操作，接受数据类型作为参数
    def test_binary_erosion26(self, dtype):
        # 定义结构元素（结构化数组），用于二进制侵蚀操作
        struct = [[0, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0]]
        # 预期输出的二进制侵蚀结果
        expected = [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0, 0, 1],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1]]
        # 创建一个 NumPy 数组作为测试输入数据
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 1, 1, 1, 0, 1, 1],
                         [0, 0, 1, 0, 1, 1, 0, 0],
                         [0, 1, 0, 1, 1, 1, 1, 0],
                         [0, 1, 1, 0, 0, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        # 对输入数据进行二进制侵蚀操作，使用给定的结构元素和参数
        out = ndimage.binary_erosion(data, struct, border_value=1,
                                     origin=(-1, -1))
        # 断言操作结果与预期输出是否相近
        assert_array_almost_equal(out, expected)

    # 定义测试函数，用于测试二进制侵蚀操作（不指定数据类型）
    def test_binary_erosion27(self):
        # 定义结构元素，用于二进制侵蚀操作
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        # 预期输出的二进制侵蚀结果
        expected = [[0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]
        # 创建一个布尔类型的 NumPy 数组作为测试输入数据
        data = np.array([[0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0]], bool)
        # 对输入数据进行二进制侵蚀操作，使用给定的结构元素和参数
        out = ndimage.binary_erosion(data, struct, border_value=1,
                                     iterations=2)
        # 断言操作结果与预期输出是否相近
        assert_array_almost_equal(out, expected)

    # 定义测试函数，用于测试二进制侵蚀操作（指定输出数组）
    def test_binary_erosion28(self):
        # 定义结构元素，用于二进制侵蚀操作
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        # 预期输出的二进制侵蚀结果
        expected = [[0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]
        # 创建一个布尔类型的 NumPy 数组作为测试输入数据
        data = np.array([[0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0]], bool)
        # 创建一个全零的布尔类型 NumPy 数组作为输出数组
        out = np.zeros(data.shape, bool)
        # 对输入数据进行二进制侵蚀操作，结果存储到指定的输出数组中
        ndimage.binary_erosion(data, struct, border_value=1,
                               iterations=2, output=out)
        # 断言操作结果与预期输出是否相近
        assert_array_almost_equal(out, expected)
    # 定义一个测试函数，用于测试二进制侵蚀操作的功能
    def test_binary_erosion30(self):
        # 结构元素，用于定义侵蚀操作的模板
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        # 预期输出，侵蚀操作后的预期结果
        expected = [[0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]
        # 输入数据，用 numpy 数组表示的二维布尔值矩阵
        data = np.array([[0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0]], bool)
        # 创建一个与输入数据形状相同的全零布尔值数组，用于存储侵蚀操作的输出
        out = np.zeros(data.shape, bool)
        # 使用 ndimage 模块进行二进制侵蚀操作，输出结果存储在 out 中
        ndimage.binary_erosion(data, struct, border_value=1,
                               iterations=3, output=out)
        # 断言输出结果与预期结果几乎相等
        assert_array_almost_equal(out, expected)

        # 测试带有输出内存重叠的情况
        # 在原始数据上直接进行侵蚀操作，结果存储在 data 中
        ndimage.binary_erosion(data, struct, border_value=1,
                               iterations=3, output=data)
        # 断言原始数据经过侵蚀操作后与预期结果几乎相等
        assert_array_almost_equal(data, expected)
    def test_binary_erosion31(self):
        # 定义结构元素（用于形态学运算）
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        # 预期的输出结果
        expected = [[0, 0, 1, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 1],
                    [0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1]]
        # 输入的数据数组
        data = np.array([[0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0]], bool)
        # 创建一个与数据数组形状相同的全零数组
        out = np.zeros(data.shape, bool)
        # 执行二进制侵蚀运算，将结果存储到 out 数组中
        ndimage.binary_erosion(data, struct, border_value=1,
                               iterations=1, output=out, origin=(-1, -1))
        # 断言输出结果与预期结果几乎相等
        assert_array_almost_equal(out, expected)

    def test_binary_erosion32(self):
        # 定义结构元素（用于形态学运算）
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        # 预期的输出结果
        expected = [[0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]
        # 输入的数据数组
        data = np.array([[0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0]], bool)
        # 执行二进制侵蚀运算，返回结果数组
        out = ndimage.binary_erosion(data, struct,
                                     border_value=1, iterations=2)
        # 断言输出结果与预期结果几乎相等
        assert_array_almost_equal(out, expected)
    # 定义一个名为 test_binary_erosion33 的测试函数
    def test_binary_erosion33(self):
        # 定义结构元素（二维数组），用于执行二值侵蚀操作
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        # 预期的输出结果（二维数组），是二值侵蚀操作后的期望结果
        expected = [[0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]
        # 定义掩模（二维数组），用于指定哪些位置的像素参与二值侵蚀
        mask = [[1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1]]
        # 定义输入数据（二维数组），表示进行二值侵蚀操作的原始数据
        data = np.array([[0, 0, 0, 0, 0, 1, 1],
                         [0, 0, 0, 1, 0, 0, 1],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0]], bool)
        # 执行二值侵蚀操作，将结果赋给 out 变量
        out = ndimage.binary_erosion(data, struct,
                                     border_value=1, mask=mask, iterations=-1)
        # 断言输出结果 out 和预期结果 expected 近似相等
        assert_array_almost_equal(out, expected)

    # 定义一个名为 test_binary_erosion34 的测试函数
    def test_binary_erosion34(self):
        # 定义结构元素（二维数组），用于执行二值侵蚀操作
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        # 预期的输出结果（二维数组），是二值侵蚀操作后的期望结果
        expected = [[0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]
        # 定义掩模（二维数组），用于指定哪些位置的像素参与二值侵蚀
        mask = [[0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]]
        # 定义输入数据（二维数组），表示进行二值侵蚀操作的原始数据
        data = np.array([[0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0]], bool)
        # 执行二值侵蚀操作，将结果赋给 out 变量
        out = ndimage.binary_erosion(data, struct,
                                     border_value=1, mask=mask)
        # 断言输出结果 out 和预期结果 expected 近似相等
        assert_array_almost_equal(out, expected)
    # 定义测试函数，测试二进制侵蚀功能，使用结构元素和掩码进行计算
    def test_binary_erosion35(self):
        # 定义结构元素，一个二维数组
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        # 定义掩码，也是一个二维数组
        mask = [[0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]]
        # 定义数据，使用NumPy创建布尔类型的二维数组
        data = np.array([[0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0]], bool)
        # 定义预期输出，进行逻辑与和或操作得到的结果
        tmp = [[0, 0, 1, 0, 0, 0, 0],
               [0, 1, 1, 1, 0, 0, 0],
               [1, 1, 1, 1, 1, 0, 1],
               [0, 1, 1, 1, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 1]]
        expected = np.logical_and(tmp, mask)
        # 对数据应用二进制侵蚀函数，指定结构元素、掩码以及其他参数
        out = np.zeros(data.shape, bool)
        ndimage.binary_erosion(data, struct, border_value=1,
                               iterations=1, output=out,
                               origin=(-1, -1), mask=mask)
        # 断言输出与预期结果相等
        assert_array_almost_equal(out, expected)

    # 定义另一个测试函数，测试不同的二进制侵蚀功能
    def test_binary_erosion36(self):
        # 定义结构元素，一个二维数组
        struct = [[0, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0]]
        # 定义掩码，也是一个二维数组
        mask = [[0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]]
        # 定义数据，使用NumPy创建二维数组
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 1, 1, 1, 0, 1, 1],
                         [0, 0, 1, 0, 1, 1, 0, 0],
                         [0, 1, 0, 1, 1, 1, 1, 0],
                         [0, 1, 1, 0, 0, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]])
        # 定义预期输出，进行逻辑与和或操作得到的结果
        tmp = [[0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 1, 0, 0, 1],
               [0, 0, 1, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1]]
        expected = np.logical_and(tmp, mask)
        # 对数据应用二进制侵蚀函数，指定结构元素、掩码以及其他参数
        out = ndimage.binary_erosion(data, struct, mask=mask,
                                     border_value=1, origin=(-1, -1))
        # 断言输出与预期结果相等
        assert_array_almost_equal(out, expected)
    # 定义测试函数test_binary_erosion37，用于测试二进制腐蚀操作
    def test_binary_erosion37(self):
        # 创建一个布尔类型的二维数组a
        a = np.array([[1, 0, 1],
                      [0, 1, 0],
                      [1, 0, 1]], dtype=bool)
        # 创建一个与a相同形状的全零数组b
        b = np.zeros_like(a)
        # 使用ndimage模块中的binary_erosion函数对数组a进行腐蚀操作，并将结果存入数组b中
        out = ndimage.binary_erosion(a, structure=a, output=b, iterations=0,
                                     border_value=True, brute_force=True)
        # 断言操作后的out数组与b是同一对象
        assert_(out is b)
        # 断言使用不同的参数再次调用binary_erosion函数得到的结果与数组b相等
        assert_array_equal(
            ndimage.binary_erosion(a, structure=a, iterations=0,
                                   border_value=True),
            b)

    # 定义测试函数test_binary_erosion38，用于测试非法参数类型情况下的异常处理
    def test_binary_erosion38(self):
        # 创建一个布尔类型的二维数组data
        data = np.array([[1, 0, 1],
                        [0, 1, 0],
                        [1, 0, 1]], dtype=bool)
        # 定义一个非法的iterations参数（浮点数），期望会抛出TypeError异常
        iterations = 2.0
        # 使用assert_raises上下文管理器来断言调用binary_erosion函数时会抛出TypeError异常
        with assert_raises(TypeError):
            _ = ndimage.binary_erosion(data, iterations=iterations)

    # 定义测试函数test_binary_erosion39，用于测试二进制腐蚀操作的期望输出
    def test_binary_erosion39(self):
        # 定义一个整数类型的iterations参数
        iterations = np.int32(3)
        # 定义一个3x3的结构元素struct
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        # 定义期望的输出数组expected，包含了预期的腐蚀结果
        expected = [[0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]
        # 创建一个布尔类型的二维数组data
        data = np.array([[0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0]], bool)
        # 创建一个与data形状相同的全零数组out
        out = np.zeros(data.shape, bool)
        # 使用ndimage模块中的binary_erosion函数对data进行腐蚀操作，将结果存入out中
        ndimage.binary_erosion(data, struct, border_value=1,
                               iterations=iterations, output=out)
        # 断言out数组的结果与预期的expected数组相近
        assert_array_almost_equal(out, expected)

    # 定义测试函数test_binary_erosion40，用于测试二进制腐蚀操作的期望输出（使用64位整数类型的iterations）
    def test_binary_erosion40(self):
        # 定义一个64位整数类型的iterations参数
        iterations = np.int64(3)
        # 定义一个3x3的结构元素struct
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        # 定义期望的输出数组expected，包含了预期的腐蚀结果
        expected = [[0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]
        # 创建一个布尔类型的二维数组data
        data = np.array([[0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0]], bool)
        # 创建一个与data形状相同的全零数组out
        out = np.zeros(data.shape, bool)
        # 使用ndimage模块中的binary_erosion函数对data进行腐蚀操作，将结果存入out中
        ndimage.binary_erosion(data, struct, border_value=1,
                               iterations=iterations, output=out)
        # 断言out数组的结果与预期的expected数组相近
        assert_array_almost_equal(out, expected)

    # 使用pytest的参数化装饰器对'types'参数进行测试
    @pytest.mark.parametrize('dtype', types)
    @pytest.mark.parametrize('dtype', types)
    # 使用 pytest 的 parametrize 标记，允许在测试方法中使用不同的数据类型
    def test_binary_dilation01(self, dtype):
        # 创建一个空数组，数据类型为传入的 dtype
        data = np.ones([], dtype)
        # 对数据进行二值膨胀操作
        out = ndimage.binary_dilation(data)
        # 断言输出的结果与预期的值几乎相等（在此情况下，输出应该为1）
        assert_array_almost_equal(out, 1)

    @pytest.mark.parametrize('dtype', types)
    # 使用 pytest 的 parametrize 标记，允许在测试方法中使用不同的数据类型
    def test_binary_dilation02(self, dtype):
        # 创建一个空数组，数据类型为传入的 dtype
        data = np.zeros([], dtype)
        # 对数据进行二值膨胀操作
        out = ndimage.binary_dilation(data)
        # 断言输出的结果与预期的值几乎相等（在此情况下，输出应该为0）
        assert_array_almost_equal(out, 0)

    @pytest.mark.parametrize('dtype', types)
    # 使用 pytest 的 parametrize 标记，允许在测试方法中使用不同的数据类型
    def test_binary_dilation03(self, dtype):
        # 创建一个长度为1的数组，数据类型为传入的 dtype
        data = np.ones([1], dtype)
        # 对数据进行二值膨胀操作
        out = ndimage.binary_dilation(data)
        # 断言输出的结果与预期的值几乎相等（在此情况下，输出应该为 [1]）
        assert_array_almost_equal(out, [1])

    @pytest.mark.parametrize('dtype', types)
    # 使用 pytest 的 parametrize 标记，允许在测试方法中使用不同的数据类型
    def test_binary_dilation04(self, dtype):
        # 创建一个长度为1的数组，数据类型为传入的 dtype
        data = np.zeros([1], dtype)
        # 对数据进行二值膨胀操作
        out = ndimage.binary_dilation(data)
        # 断言输出的结果与预期的值几乎相等（在此情况下，输出应该为 [0]）
        assert_array_almost_equal(out, [0])

    @pytest.mark.parametrize('dtype', types)
    # 使用 pytest 的 parametrize 标记，允许在测试方法中使用不同的数据类型
    def test_binary_dilation05(self, dtype):
        # 创建一个长度为3的数组，数据类型为传入的 dtype
        data = np.ones([3], dtype)
        # 对数据进行二值膨胀操作
        out = ndimage.binary_dilation(data)
        # 断言输出的结果与预期的值几乎相等（在此情况下，输出应该为 [1, 1, 1]）
        assert_array_almost_equal(out, [1, 1, 1])

    @pytest.mark.parametrize('dtype', types)
    # 使用 pytest 的 parametrize 标记，允许在测试方法中使用不同的数据类型
    def test_binary_dilation06(self, dtype):
        # 创建一个长度为3的数组，数据类型为传入的 dtype
        data = np.zeros([3], dtype)
        # 对数据进行二值膨胀操作
        out = ndimage.binary_dilation(data)
        # 断言输出的结果与预期的值几乎相等（在此情况下，输出应该为 [0, 0, 0]）
        assert_array_almost_equal(out, [0, 0, 0])

    @pytest.mark.parametrize('dtype', types)
    # 使用 pytest 的 parametrize 标记，允许在测试方法中使用不同的数据类型
    def test_binary_dilation07(self, dtype):
        # 创建一个长度为3的数组，数据类型为传入的 dtype
        data = np.zeros([3], dtype)
        # 将数组的第二个元素设置为1
        data[1] = 1
        # 对数据进行二值膨胀操作
        out = ndimage.binary_dilation(data)
        # 断言输出的结果与预期的值几乎相等（在此情况下，输出应该为 [1, 1, 1]）
        assert_array_almost_equal(out, [1, 1, 1])

    @pytest.mark.parametrize('dtype', types)
    # 使用 pytest 的 parametrize 标记，允许在测试方法中使用不同的数据类型
    def test_binary_dilation08(self, dtype):
        # 创建一个长度为5的数组，数据类型为传入的 dtype
        data = np.zeros([5], dtype)
        # 将数组的第二个和第四个元素设置为1
        data[1] = 1
        data[3] = 1
        # 对数据进行二值膨胀操作
        out = ndimage.binary_dilation(data)
        # 断言输出的结果与预期的值几乎相等（在此情况下，输出应该为 [1, 1, 1, 1, 1]）
        assert_array_almost_equal(out, [1, 1, 1, 1, 1])

    @pytest.mark.parametrize('dtype', types)
    # 使用 pytest 的 parametrize 标记，允许在测试方法中使用不同的数据类型
    def test_binary_dilation09(self, dtype):
        # 创建一个长度为5的数组，数据类型为传入的 dtype
        data = np.zeros([5], dtype)
        # 将数组的第二个元素设置为1
        data[1] = 1
        # 对数据进行二值膨胀操作
        out = ndimage.binary_dilation(data)
        # 断言输出的结果与预期的值几乎相等（在此情况下，输出应该为 [1, 1, 1, 0, 0]）
        assert_array_almost_equal(out, [1, 1, 1, 0, 0])

    @pytest.mark.parametrize('dtype', types)
    # 使用 pytest 的 parametrize 标记，允许在测试方法中使用不同的数据类型
    def test_binary_dilation10(self, dtype):
        # 创建一个长度为5的数组，数据类型为传入的 dtype
        data = np.zeros([5], dtype)
        # 将数组的第二个元素设置为1
        data[1] = 1
        # 对数据进行二值膨胀操作，并指定起始位置为-1
        out = ndimage.binary_dilation(data, origin=-1)
        # 断言输出的结果与预期的值几乎相等（在此情况下，输出应该为 [0, 1, 1, 1, 0]）
        assert_array_almost_equal(out, [0, 1, 1, 1, 0])

    @pytest.mark.parametrize('dtype', types)
    # 使用 pytest 的 parametrize 标记，允许在测试方法中使用不同的数据类型
    def test_binary_dilation11(self, dtype):
        # 创建一个长度为5的数组，数据类型为传入的 dtype
        data = np.zeros([5], dtype)
        # 将数组的第二个元素设置为1
        data[1] = 1
        # 对数据进行二值膨胀操作，并指定起始位置为1
        out = ndimage.binary_dilation(data, origin=1)
        # 断言输出的结果与预期的值几乎相等（在此情况下，输出应该为 [1, 1, 0, 0, 0]）
        assert_array_almost_equal(out, [1, 1, 0, 0, 0])

    @pytest.mark.parametrize('dtype', types)
    # 使用 pytest 的 parametrize 标记，允许在测试方法中使用不同的数据类型
    def test_binary_dilation12(self, dtype):
        # 创建一个长度为5的数组，数据类型为传入的 dtype
        data = np.zeros([5], dtype)
        # 将数组的第二个元素设置为1
        data[1] = 1
        # 定义结构元素为 [1, 0, 1]
        struct = [1, 0, 1]
        # 对数据进行二值膨胀操作，并使用指定的结构元素
        out = ndimage.binary_dilation(data, struct)
        # 断言输出的结果与预期的值几乎相等
    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation13(self, dtype):
        # 创建一个长度为5的零数组，使用给定的数据类型
        data = np.zeros([5], dtype)
        # 将数组第二个元素设置为1
        data[1] = 1
        # 定义一个结构元素作为列表
        struct = [1, 0, 1]
        # 对输入数据进行二值膨胀操作，使用指定的结构元素和边界值
        out = ndimage.binary_dilation(data, struct, border_value=1)
        # 断言输出数组与期望的数组几乎相等
        assert_array_almost_equal(out, [1, 0, 1, 0, 1])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation14(self, dtype):
        # 创建一个长度为5的零数组，使用给定的数据类型
        data = np.zeros([5], dtype)
        # 将数组第二个元素设置为1
        data[1] = 1
        # 定义一个结构元素作为列表
        struct = [1, 0, 1]
        # 对输入数据进行二值膨胀操作，使用指定的结构元素和原点位置
        out = ndimage.binary_dilation(data, struct, origin=-1)
        # 断言输出数组与期望的数组几乎相等
        assert_array_almost_equal(out, [0, 1, 0, 1, 0])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation15(self, dtype):
        # 创建一个长度为5的零数组，使用给定的数据类型
        data = np.zeros([5], dtype)
        # 将数组第二个元素设置为1
        data[1] = 1
        # 定义一个结构元素作为列表
        struct = [1, 0, 1]
        # 对输入数据进行二值膨胀操作，使用指定的结构元素、原点位置和边界值
        out = ndimage.binary_dilation(data, struct,
                                      origin=-1, border_value=1)
        # 断言输出数组与期望的数组几乎相等
        assert_array_almost_equal(out, [1, 1, 0, 1, 0])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation16(self, dtype):
        # 创建一个大小为1x1的全1数组，使用给定的数据类型
        data = np.ones([1, 1], dtype)
        # 对输入数据进行二值膨胀操作，不指定结构元素
        out = ndimage.binary_dilation(data)
        # 断言输出数组与期望的数组几乎相等
        assert_array_almost_equal(out, [[1]])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation17(self, dtype):
        # 创建一个大小为1x1的零数组，使用给定的数据类型
        data = np.zeros([1, 1], dtype)
        # 对输入数据进行二值膨胀操作，不指定结构元素
        out = ndimage.binary_dilation(data)
        # 断言输出数组与期望的数组几乎相等
        assert_array_almost_equal(out, [[0]])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation18(self, dtype):
        # 创建一个大小为1x3的全1数组，使用给定的数据类型
        data = np.ones([1, 3], dtype)
        # 对输入数据进行二值膨胀操作，不指定结构元素
        out = ndimage.binary_dilation(data)
        # 断言输出数组与期望的数组几乎相等
        assert_array_almost_equal(out, [[1, 1, 1]])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation19(self, dtype):
        # 创建一个大小为3x3的全1数组，使用给定的数据类型
        data = np.ones([3, 3], dtype)
        # 对输入数据进行二值膨胀操作，不指定结构元素
        out = ndimage.binary_dilation(data)
        # 断言输出数组与期望的数组几乎相等
        assert_array_almost_equal(out, [[1, 1, 1],
                                        [1, 1, 1],
                                        [1, 1, 1]])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation20(self, dtype):
        # 创建一个大小为3x3的零数组，使用给定的数据类型
        data = np.zeros([3, 3], dtype)
        # 将数组中心位置的元素设为1
        data[1, 1] = 1
        # 对输入数据进行二值膨胀操作，不指定结构元素
        out = ndimage.binary_dilation(data)
        # 断言输出数组与期望的数组几乎相等
        assert_array_almost_equal(out, [[0, 1, 0],
                                        [1, 1, 1],
                                        [0, 1, 0]])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation21(self, dtype):
        # 使用给定维度和连接结构生成一个二进制结构元素
        struct = ndimage.generate_binary_structure(2, 2)
        # 创建一个大小为3x3的零数组，使用给定的数据类型
        data = np.zeros([3, 3], dtype)
        # 将数组中心位置的元素设为1
        data[1, 1] = 1
        # 对输入数据进行二值膨胀操作，使用生成的二进制结构元素
        out = ndimage.binary_dilation(data, struct)
        # 断言输出数组与期望的数组几乎相等
        assert_array_almost_equal(out, [[1, 1, 1],
                                        [1, 1, 1],
                                        [1, 1, 1]])
    def test_binary_dilation22(self, dtype):
        expected = [[0, 1, 0, 0, 0, 0, 0, 0],  # 预期输出的二维数组
                    [1, 1, 1, 0, 0, 0, 0, 0],  # 预期输出的二维数组
                    [0, 1, 0, 0, 0, 1, 0, 0],  # 预期输出的二维数组
                    [0, 0, 0, 1, 1, 1, 1, 0],  # 预期输出的二维数组
                    [0, 0, 1, 1, 1, 1, 0, 0],  # 预期输出的二维数组
                    [0, 1, 1, 1, 1, 1, 1, 0],  # 预期输出的二维数组
                    [0, 0, 1, 0, 0, 1, 0, 0],  # 预期输出的二维数组
                    [0, 0, 0, 0, 0, 0, 0, 0]]  # 预期输出的二维数组
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0],  # 输入的二维数组
                         [0, 1, 0, 0, 0, 0, 0, 0],  # 输入的二维数组
                         [0, 0, 0, 0, 0, 0, 0, 0],  # 输入的二维数组
                         [0, 0, 0, 0, 0, 1, 0, 0],  # 输入的二维数组
                         [0, 0, 0, 1, 1, 0, 0, 0],  # 输入的二维数组
                         [0, 0, 1, 0, 0, 1, 0, 0],  # 输入的二维数组
                         [0, 0, 0, 0, 0, 0, 0, 0],  # 输入的二维数组
                         [0, 0, 0, 0, 0, 0, 0, 0]], dtype)  # 输入数组的数据类型
        out = ndimage.binary_dilation(data)  # 对输入的二进制数组进行二值膨胀处理
        assert_array_almost_equal(out, expected)  # 断言输出是否与预期一致

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation23(self, dtype):
        expected = [[1, 1, 1, 1, 1, 1, 1, 1],  # 预期输出的二维数组
                    [1, 1, 1, 0, 0, 0, 0, 1],  # 预期输出的二维数组
                    [1, 1, 0, 0, 0, 1, 0, 1],  # 预期输出的二维数组
                    [1, 0, 0, 1, 1, 1, 1, 1],  # 预期输出的二维数组
                    [1, 0, 1, 1, 1, 1, 0, 1],  # 预期输出的二维数组
                    [1, 1, 1, 1, 1, 1, 1, 1],  # 预期输出的二维数组
                    [1, 0, 1, 0, 0, 1, 0, 1],  # 预期输出的二维数组
                    [1, 1, 1, 1, 1, 1, 1, 1]]  # 预期输出的二维数组
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0],  # 输入的二维数组
                         [0, 1, 0, 0, 0, 0, 0, 0],  # 输入的二维数组
                         [0, 0, 0, 0, 0, 0, 0, 0],  # 输入的二维数组
                         [0, 0, 0, 0, 0, 1, 0, 0],  # 输入的二维数组
                         [0, 0, 0, 1, 1, 0, 0, 0],  # 输入的二维数组
                         [0, 0, 1, 0, 0, 1, 0, 0],  # 输入的二维数组
                         [0, 0, 0, 0, 0, 0, 0, 0],  # 输入的二维数组
                         [0, 0, 0, 0, 0, 0, 0, 0]], dtype)  # 输入数组的数据类型
        out = ndimage.binary_dilation(data, border_value=1)  # 对输入的二进制数组进行二值膨胀处理，边界值设为1
        assert_array_almost_equal(out, expected)  # 断言输出是否与预期一致

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation24(self, dtype):
        expected = [[1, 1, 0, 0, 0, 0, 0, 0],  # 预期输出的二维数组
                    [1, 0, 0, 0, 1, 0, 0, 0],  # 预期输出的二维数组
                    [0, 0, 1, 1, 1, 1, 0, 0],  # 预期输出的二维数组
                    [0, 1, 1, 1, 1, 0, 0, 0],  # 预期输出的二维数组
                    [1, 1, 1, 1, 1, 1, 0, 0],  # 预期输出的二维数组
                    [0, 1, 0, 0, 1, 0, 0, 0],  # 预期输出的二维数组
                    [0, 0, 0, 0, 0, 0, 0, 0],  # 预期输出的二维数组
                    [0, 0, 0, 0, 0, 0, 0, 0]]  # 预期输出的二维数组
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0],  # 输入的二维数组
                         [0, 1, 0, 0, 0, 0, 0, 0],  # 输入的二维数组
                         [0, 0, 0, 0, 0, 0, 0, 0],  # 输入的二维数组
                         [0, 0, 0, 0, 0, 1, 0, 0],  # 输入的二维数组
                         [0, 0, 0, 1, 1, 0, 0, 0],  # 输入的二维数组
                         [0, 0, 1, 0, 0, 1, 0, 0],  # 输入的二维数组
                         [0, 0, 0, 0, 0, 0, 0, 0],  # 输入的二维数组
                         [0, 0, 0, 0, 0, 0, 0, 0]], dtype)  # 输入数组的数据类型
        out = ndimage.binary_dilation(data, origin=(1, 1))  # 对输入的二进制数组进行二值膨胀处理，指定起始点为(1, 1)
        assert_array_almost_equal(out, expected)  # 断言输出是否与预期一致
    # 定义一个名为 test_binary_dilation25 的测试方法，带有一个参数 dtype
    def test_binary_dilation25(self, dtype):
        # 预期输出的二维数组
        expected = [[1, 1, 0, 0, 0, 0, 1, 1],
                    [1, 0, 0, 0, 1, 0, 1, 1],
                    [0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 0, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 0, 0, 1, 0, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1]]
        # 创建一个 numpy 数组，表示二维数据矩阵
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 1, 1, 0, 0, 0],
                         [0, 0, 1, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        # 使用 ndimage 库中的 binary_dilation 函数对 data 进行形态学膨胀操作
        out = ndimage.binary_dilation(data, origin=(1, 1), border_value=1)
        # 断言输出的结果与预期结果相近
        assert_array_almost_equal(out, expected)

    # 使用 pytest 的参数化装饰器，对 test_binary_dilation26 方法进行参数化
    @pytest.mark.parametrize('dtype', types)
    # 定义一个名为 test_binary_dilation26 的测试方法，带有一个参数 dtype
    def test_binary_dilation26(self, dtype):
        # 生成一个二进制结构用于形态学操作
        struct = ndimage.generate_binary_structure(2, 2)
        # 预期输出的二维数组
        expected = [[1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        # 创建一个 numpy 数组，表示二维数据矩阵
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 1, 1, 0, 0, 0],
                         [0, 0, 1, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        # 使用 ndimage 库中的 binary_dilation 函数对 data 进行形态学膨胀操作，结构由 struct 定义
        out = ndimage.binary_dilation(data, struct)
        # 断言输出的结果与预期结果相近
        assert_array_almost_equal(out, expected)
    # 定义一个测试函数，用于测试二进制膨胀操作（使用不同的结构元素和参数）
    def test_binary_dilation27(self, dtype):
        # 定义一个二维结构元素，用于膨胀操作
        struct = [[0, 1],
                  [1, 1]]
        # 预期的膨胀结果
        expected = [[0, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        # 定义一个测试数据数组，用于进行二进制膨胀操作
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 1, 1, 0, 0, 0],
                         [0, 0, 1, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        # 执行二进制膨胀操作，获取实际输出
        out = ndimage.binary_dilation(data, struct)
        # 断言实际输出与预期输出相近
        assert_array_almost_equal(out, expected)

    # 使用pytest的参数化标记，定义另一个测试函数，用于测试带有边界值的二进制膨胀操作
    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation28(self, dtype):
        # 预期的带有边界值的膨胀结果
        expected = [[1, 1, 1, 1],
                    [1, 0, 0, 1],
                    [1, 0, 0, 1],
                    [1, 1, 1, 1]]
        # 定义一个测试数据数组，用于进行带有边界值的二进制膨胀操作
        data = np.array([[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]], dtype)
        # 执行带有边界值的二进制膨胀操作，获取实际输出
        out = ndimage.binary_dilation(data, border_value=1)
        # 断言实际输出与预期输出相近
        assert_array_almost_equal(out, expected)

    # 定义一个测试函数，用于测试带有迭代次数参数的二进制膨胀操作
    def test_binary_dilation29(self):
        # 定义一个二维结构元素，用于膨胀操作
        struct = [[0, 1],
                  [1, 1]]
        # 预期的膨胀结果（经过两次迭代）
        expected = [[0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0]]
        # 定义一个布尔类型的测试数据数组，用于进行带有迭代次数参数的二进制膨胀操作
        data = np.array([[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0]], bool)
        # 执行带有迭代次数参数的二进制膨胀操作，获取实际输出
        out = ndimage.binary_dilation(data, struct, iterations=2)
        # 断言实际输出与预期输出相近
        assert_array_almost_equal(out, expected)

    # 定义一个测试函数，用于测试将输出结果写入预定义的输出数组的二进制膨胀操作
    def test_binary_dilation30(self):
        # 定义一个二维结构元素，用于膨胀操作
        struct = [[0, 1],
                  [1, 1]]
        # 预期的膨胀结果（经过两次迭代）
        expected = [[0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0]]
        # 定义一个布尔类型的测试数据数组，用于进行输出结果写入预定义的输出数组的二进制膨胀操作
        data = np.array([[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0]], bool)
        # 创建一个与测试数据数组形状相同的布尔类型数组，作为输出的预定义数组
        out = np.zeros(data.shape, bool)
        # 执行将输出结果写入预定义的输出数组的二进制膨胀操作
        ndimage.binary_dilation(data, struct, iterations=2, output=out)
        # 断言实际输出与预期输出相近
        assert_array_almost_equal(out, expected)
    def test_binary_dilation31(self):
        struct = [[0, 1],                # 结构元素定义，用于二值膨胀操作
                  [1, 1]]
        expected = [[0, 0, 0, 1, 0],      # 预期输出结果的二维数组
                    [0, 0, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0]]

        data = np.array([[0, 0, 0, 0, 0],  # 输入的二值数组数据
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0]], bool)
        out = ndimage.binary_dilation(data, struct, iterations=3)  # 执行二值膨胀操作
        assert_array_almost_equal(out, expected)  # 断言输出是否与预期结果相符

    def test_binary_dilation32(self):
        struct = [[0, 1],                # 结构元素定义，用于二值膨胀操作
                  [1, 1]]
        expected = [[0, 0, 0, 1, 0],      # 预期输出结果的二维数组
                    [0, 0, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0]]

        data = np.array([[0, 0, 0, 0, 0],  # 输入的二值数组数据
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0]], bool)
        out = np.zeros(data.shape, bool)  # 创建一个与输入数据相同形状的空数组
        ndimage.binary_dilation(data, struct, iterations=3, output=out)  # 执行二值膨胀操作，输出到指定数组
        assert_array_almost_equal(out, expected)  # 断言输出是否与预期结果相符

    def test_binary_dilation33(self):
        struct = [[0, 1, 0],             # 结构元素定义，用于二值膨胀操作
                  [1, 1, 1],
                  [0, 1, 0]]
        expected = np.array([[0, 1, 0, 0, 0, 0, 0, 0],   # 预期输出结果的二维数组
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 1, 0, 0],
                             [0, 0, 1, 1, 1, 0, 0, 0],
                             [0, 1, 1, 0, 1, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0]], bool)
        mask = np.array([[0, 1, 0, 0, 0, 0, 0, 0],         # 遮罩数组，指定哪些区域应用膨胀
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 1, 1, 0, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]], bool)
        data = np.array([[0, 1, 0, 0, 0, 0, 0, 0],         # 输入的二值数组数据
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]], bool)

        out = ndimage.binary_dilation(data, struct, iterations=-1,   # 执行二值膨胀操作，指定参数
                                      mask=mask, border_value=0)
        assert_array_almost_equal(out, expected)   # 断言输出是否与预期结果相符
    # 定义名为 test_binary_dilation34 的测试方法，用于测试二进制膨胀功能
    def test_binary_dilation34(self):
        # 定义结构元素，表示膨胀操作的结构
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        # 预期输出的二进制膨胀结果
        expected = [[0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        # 定义用作遮罩的布尔数组
        mask = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 1, 1, 0, 0, 0],
                         [0, 0, 1, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]], bool)
        # 创建与遮罩相同形状的布尔数组，用于存储膨胀结果
        data = np.zeros(mask.shape, bool)
        # 调用二进制膨胀函数，传入数据、结构元素、迭代次数、遮罩和边界值
        out = ndimage.binary_dilation(data, struct, iterations=-1,
                                      mask=mask, border_value=1)
        # 断言输出结果与预期结果几乎相等
        assert_array_almost_equal(out, expected)

    # 使用 pytest 的参数化装饰器，传入不同的数据类型进行测试
    @pytest.mark.parametrize('dtype', types)
    # 定义一个测试函数，用于测试 binary_dilation 方法，输入参数为数据类型 dtype
    def test_binary_dilation35(self, dtype):
        # 定义一个临时的二维列表 tmp，表示二值化的输入数据
        tmp = [[1, 1, 0, 0, 0, 0, 1, 1],
               [1, 0, 0, 0, 1, 0, 1, 1],
               [0, 0, 1, 1, 1, 1, 1, 1],
               [0, 1, 1, 1, 1, 0, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1],
               [0, 1, 0, 0, 1, 0, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1]]
        # 定义一个二维 NumPy 数组 data，表示待处理的数据
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 1, 1, 0, 0, 0],
                         [0, 0, 1, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]])
        # 定义一个二维列表 mask，表示 dilation 操作中的掩码
        mask = [[0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]]
        # 计算预期的输出结果，即 tmp 和 mask 的逻辑与操作的结果
        expected = np.logical_and(tmp, mask)
        # 对 data 应用 dilation 操作，使用给定的 mask 和原点 (1, 1)，边界值为 1
        out = ndimage.binary_dilation(data, mask=mask,
                                      origin=(1, 1), border_value=1)
        # 断言实际输出结果与预期结果的近似相等性
        assert_array_almost_equal(out, expected)
    def test_binary_propagation01(self):
        # 定义结构元素，表示形状为一个三行三列的二维数组
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        # 定义预期输出的二维布尔数组，形状为8行8列
        expected = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 1, 0, 0],
                             [0, 0, 1, 1, 1, 0, 0, 0],
                             [0, 1, 1, 0, 1, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0]], bool)
        # 定义掩码数组，形状为8行8列的二维布尔数组
        mask = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 1, 1, 0, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]], bool)
        # 定义数据数组，形状为8行8列的二维布尔数组，全为False
        data = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]], bool)

        # 调用ndimage模块的binary_propagation函数，传入数据、结构元素、掩码和边界值参数
        out = ndimage.binary_propagation(data, struct,
                                         mask=mask, border_value=0)
        # 断言输出数组与预期结果数组几乎相等
        assert_array_almost_equal(out, expected)

    def test_binary_propagation02(self):
        # 定义结构元素，表示形状为一个三行三列的二维数组
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        # 定义预期输出的二维数组
        expected = [[0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        # 定义掩码数组，形状为8行8列的二维布尔数组
        mask = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 1, 1, 0, 0, 0],
                         [0, 0, 1, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]], bool)
        # 创建与掩码数组形状相同的全False的二维布尔数组作为数据数组
        data = np.zeros(mask.shape, bool)
        
        # 调用ndimage模块的binary_propagation函数，传入数据、结构元素、掩码和边界值参数
        out = ndimage.binary_propagation(data, struct,
                                         mask=mask, border_value=1)
        # 断言输出数组与预期结果数组几乎相等
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('dtype', types)
    @pytest.mark.parametrize('dtype', types)
    # 使用 pytest 的参数化装饰器，允许在测试函数中多次运行同一个测试用例，每次使用不同的参数值
    def test_binary_opening01(self, dtype):
        # 预期的二值化开运算结果
        expected = [[0, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        # 定义一个 numpy 数组作为测试输入数据
        data = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1, 0, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 1, 0],
                         [0, 0, 1, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        # 执行二值化开运算
        out = ndimage.binary_opening(data)
        # 使用 numpy 的断言方法检查实际输出是否与预期结果相近
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('dtype', types)
    # 使用 pytest 的参数化装饰器，允许在测试函数中多次运行同一个测试用例，每次使用不同的参数值
    def test_binary_opening02(self, dtype):
        # 生成二维结构元素
        struct = ndimage.generate_binary_structure(2, 2)
        # 预期的二值化开运算结果
        expected = [[1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        # 定义一个 numpy 数组作为测试输入数据
        data = np.array([[1, 1, 1, 0, 0, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0, 1, 1, 0],
                         [0, 1, 1, 1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        # 执行二值化开运算
        out = ndimage.binary_opening(data, struct)
        # 使用 numpy 的断言方法检查实际输出是否与预期结果相近
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('dtype', types)
    # 使用 pytest 的参数化装饰器，允许在测试函数中多次运行同一个测试用例，每次使用不同的参数值
    def test_binary_closing01(self, dtype):
        # 预期的二值化闭运算结果
        expected = [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        # 定义一个 numpy 数组作为测试输入数据
        data = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1, 0, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 1, 0],
                         [0, 0, 1, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        # 执行二值化闭运算
        out = ndimage.binary_closing(data)
        # 使用 numpy 的断言方法检查实际输出是否与预期结果相近
        assert_array_almost_equal(out, expected)
    # 定义一个测试函数，用于测试二进制闭运算函数 binary_closing
    def test_binary_closing02(self, dtype):
        # 生成一个二维的二进制结构，这里是一个 3x3 的方形结构
        struct = ndimage.generate_binary_structure(2, 2)
        # 期望的输出结果，一个预定义的二维数组
        expected = [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        # 输入数据，一个与期望形状相同的二维 numpy 数组
        data = np.array([[1, 1, 1, 0, 0, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0, 1, 1, 0],
                         [0, 1, 1, 1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        # 调用 ndimage 库中的 binary_closing 函数，对输入数据进行二进制闭运算，使用上面定义的结构
        out = ndimage.binary_closing(data, struct)
        # 断言输出结果与期望结果几乎相等
        assert_array_almost_equal(out, expected)

    # 定义一个测试函数，用于测试二进制填充孔函数 binary_fill_holes
    def test_binary_fill_holes01(self):
        # 期望的输出结果，一个预定义的二维布尔数组
        expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 1, 1, 1, 0, 0],
                             [0, 0, 1, 1, 1, 1, 0, 0],
                             [0, 0, 1, 1, 1, 1, 0, 0],
                             [0, 0, 1, 1, 1, 1, 0, 0],
                             [0, 0, 1, 1, 1, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0]], bool)
        # 输入数据，一个与期望形状相同的二维布尔数组
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]], bool)
        # 调用 ndimage 库中的 binary_fill_holes 函数，对输入数据进行二进制填充孔操作
        out = ndimage.binary_fill_holes(data)
        # 断言输出结果与期望结果几乎相等
        assert_array_almost_equal(out, expected)

    # 定义一个测试函数，用于测试二进制填充孔函数 binary_fill_holes 的另一个情况
    def test_binary_fill_holes02(self):
        # 期望的输出结果，一个预定义的二维布尔数组
        expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 1, 0, 0, 0],
                             [0, 0, 1, 1, 1, 1, 0, 0],
                             [0, 0, 1, 1, 1, 1, 0, 0],
                             [0, 0, 1, 1, 1, 1, 0, 0],
                             [0, 0, 0, 1, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0]], bool)
        # 输入数据，一个与期望形状相同的二维布尔数组
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 0, 0, 0],
                         [0, 0, 1, 0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0, 1, 0, 0],
                         [0, 0, 0, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]], bool)
        # 调用 ndimage 库中的 binary_fill_holes 函数，对输入数据进行二进制填充孔操作
        out = ndimage.binary_fill_holes(data)
        # 断言输出结果与期望结果几乎相等
        assert_array_almost_equal(out, expected)
    def test_binary_fill_holes03(self):
        expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 0, 1, 1, 1],
                             [0, 1, 1, 1, 0, 1, 1, 1],
                             [0, 1, 1, 1, 0, 1, 1, 1],
                             [0, 0, 1, 0, 0, 1, 1, 1],
                             [0, 0, 0, 0, 0, 0, 0, 0]], bool)
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 1, 0, 1, 1, 1],
                         [0, 1, 0, 1, 0, 1, 0, 1],
                         [0, 1, 0, 1, 0, 1, 0, 1],
                         [0, 0, 1, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0]], bool)
        out = ndimage.binary_fill_holes(data)
        assert_array_almost_equal(out, expected)

    def test_grey_erosion01(self):
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        output = ndimage.grey_erosion(array, footprint=footprint)
        assert_array_almost_equal([[2, 2, 1, 1, 1],
                                   [2, 3, 1, 3, 1],
                                   [5, 5, 3, 3, 1]], output)

    def test_grey_erosion01_overlap(self):
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        # 在原始数组上应用灰度侵蚀，并将结果存储回原始数组
        ndimage.grey_erosion(array, footprint=footprint, output=array)
        assert_array_almost_equal([[2, 2, 1, 1, 1],
                                   [2, 3, 1, 3, 1],
                                   [5, 5, 3, 3, 1]], array)

    def test_grey_erosion02(self):
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        structure = [[0, 0, 0], [0, 0, 0]]
        output = ndimage.grey_erosion(array, footprint=footprint,
                                      structure=structure)
        assert_array_almost_equal([[2, 2, 1, 1, 1],
                                   [2, 3, 1, 3, 1],
                                   [5, 5, 3, 3, 1]], output)

    def test_grey_erosion03(self):
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        structure = [[1, 1, 1], [1, 1, 1]]
        output = ndimage.grey_erosion(array, footprint=footprint,
                                      structure=structure)
        assert_array_almost_equal([[1, 1, 0, 0, 0],
                                   [1, 2, 0, 2, 0],
                                   [4, 4, 2, 2, 0]], output)
    # 定义一个测试函数，用于测试灰度图像的膨胀操作（第一种情况）
    def test_grey_dilation01(self):
        # 创建一个二维 NumPy 数组作为输入图像
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        # 定义一个结构元素（足迹），用于指定膨胀操作的邻域
        footprint = [[0, 1, 1], [1, 0, 1]]
        # 调用 ndimage 库中的灰度膨胀函数，进行图像处理
        output = ndimage.grey_dilation(array, footprint=footprint)
        # 断言输出的数组与预期的数组近似相等
        assert_array_almost_equal([[7, 7, 9, 9, 5],
                                   [7, 9, 8, 9, 7],
                                   [8, 8, 8, 7, 7]], output)

    # 定义一个测试函数，用于测试灰度图像的膨胀操作（第二种情况）
    def test_grey_dilation02(self):
        # 创建一个二维 NumPy 数组作为输入图像
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        # 定义一个结构元素（足迹），用于指定膨胀操作的邻域
        footprint = [[0, 1, 1], [1, 0, 1]]
        # 定义一个结构元素（结构），用于指定膨胀操作的邻域形状
        structure = [[0, 0, 0], [0, 0, 0]]
        # 调用 ndimage 库中的灰度膨胀函数，传入自定义的结构元素
        output = ndimage.grey_dilation(array, footprint=footprint,
                                       structure=structure)
        # 断言输出的数组与预期的数组近似相等
        assert_array_almost_equal([[7, 7, 9, 9, 5],
                                   [7, 9, 8, 9, 7],
                                   [8, 8, 8, 7, 7]], output)

    # 定义一个测试函数，用于测试灰度图像的膨胀操作（第三种情况）
    def test_grey_dilation03(self):
        # 创建一个二维 NumPy 数组作为输入图像
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        # 定义一个结构元素（足迹），用于指定膨胀操作的邻域
        footprint = [[0, 1, 1], [1, 0, 1]]
        # 定义一个结构元素（结构），用于指定膨胀操作的邻域形状
        structure = [[1, 1, 1], [1, 1, 1]]
        # 调用 ndimage 库中的灰度膨胀函数，传入自定义的结构元素
        output = ndimage.grey_dilation(array, footprint=footprint,
                                       structure=structure)
        # 断言输出的数组与预期的数组近似相等
        assert_array_almost_equal([[8, 8, 10, 10, 6],
                                   [8, 10, 9, 10, 8],
                                   [9, 9, 9, 8, 8]], output)

    # 定义一个测试函数，用于测试灰度图像的开操作（第一种情况）
    def test_grey_opening01(self):
        # 创建一个二维 NumPy 数组作为输入图像
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        # 定义一个结构元素（足迹），用于指定开操作的邻域
        footprint = [[1, 0, 1], [1, 1, 0]]
        # 使用灰度侵蚀函数计算临时结果
        tmp = ndimage.grey_erosion(array, footprint=footprint)
        # 使用灰度膨胀函数计算期望的开操作结果
        expected = ndimage.grey_dilation(tmp, footprint=footprint)
        # 调用 ndimage 库中的灰度开操作函数
        output = ndimage.grey_opening(array, footprint=footprint)
        # 断言期望的输出与实际输出近似相等
        assert_array_almost_equal(expected, output)

    # 定义一个测试函数，用于测试灰度图像的开操作（第二种情况）
    def test_grey_opening02(self):
        # 创建一个二维 NumPy 数组作为输入图像
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        # 定义一个结构元素（足迹），用于指定开操作的邻域
        footprint = [[1, 0, 1], [1, 1, 0]]
        # 定义一个结构元素（结构），用于指定开操作的邻域形状
        structure = [[0, 0, 0], [0, 0, 0]]
        # 使用灰度侵蚀函数计算临时结果
        tmp = ndimage.grey_erosion(array, footprint=footprint,
                                   structure=structure)
        # 使用灰度膨胀函数计算期望的开操作结果
        expected = ndimage.grey_dilation(tmp, footprint=footprint,
                                         structure=structure)
        # 调用 ndimage 库中的灰度开操作函数，传入自定义的结构元素
        output = ndimage.grey_opening(array, footprint=footprint,
                                      structure=structure)
        # 断言期望的输出与实际输出近似相等
        assert_array_almost_equal(expected, output)
    # 定义测试函数 test_grey_closing01，用于测试灰度图像的闭运算
    def test_grey_closing01(self):
        # 创建一个二维 NumPy 数组作为输入图像
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        # 定义形态学操作的结构元素（脚印）
        footprint = [[1, 0, 1], [1, 1, 0]]
        # 对输入图像进行灰度膨胀操作
        tmp = ndimage.grey_dilation(array, footprint=footprint)
        # 对膨胀后的图像进行灰度腐蚀操作，作为预期输出
        expected = ndimage.grey_erosion(tmp, footprint=footprint)
        # 对输入图像进行灰度闭运算
        output = ndimage.grey_closing(array, footprint=footprint)
        # 断言预期输出与实际输出的数组近似相等
        assert_array_almost_equal(expected, output)

    # 定义测试函数 test_grey_closing02，用于测试带有结构元素的灰度闭运算
    def test_grey_closing02(self):
        # 创建一个二维 NumPy 数组作为输入图像
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        # 定义形态学操作的脚印和结构元素
        footprint = [[1, 0, 1], [1, 1, 0]]
        structure = [[0, 0, 0], [0, 0, 0]]
        # 对输入图像进行灰度膨胀操作，指定结构元素
        tmp = ndimage.grey_dilation(array, footprint=footprint,
                                    structure=structure)
        # 对膨胀后的图像进行灰度腐蚀操作，指定结构元素，作为预期输出
        expected = ndimage.grey_erosion(tmp, footprint=footprint,
                                        structure=structure)
        # 对输入图像进行灰度闭运算，指定结构元素
        output = ndimage.grey_closing(array, footprint=footprint,
                                      structure=structure)
        # 断言预期输出与实际输出的数组近似相等
        assert_array_almost_equal(expected, output)

    # 定义测试函数 test_morphological_gradient01，用于测试形态学梯度操作
    def test_morphological_gradient01(self):
        # 创建一个二维 NumPy 数组作为输入图像
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        # 定义形态学操作的脚印和结构元素
        footprint = [[1, 0, 1], [1, 1, 0]]
        structure = [[0, 0, 0], [0, 0, 0]]
        # 对输入图像进行灰度膨胀操作，指定结构元素
        tmp1 = ndimage.grey_dilation(array, footprint=footprint,
                                     structure=structure)
        # 对输入图像进行灰度腐蚀操作，指定结构元素
        tmp2 = ndimage.grey_erosion(array, footprint=footprint,
                                    structure=structure)
        # 计算形态学梯度，即膨胀后图像减去腐蚀后图像
        expected = tmp1 - tmp2
        # 创建一个和输入图像相同形状的零数组，作为输出
        output = np.zeros(array.shape, array.dtype)
        # 计算形态学梯度并将结果存储到输出数组中
        ndimage.morphological_gradient(array, footprint=footprint,
                                       structure=structure, output=output)
        # 断言预期输出与实际输出的数组近似相等
        assert_array_almost_equal(expected, output)

    # 定义测试函数 test_morphological_gradient02，用于测试带有结构元素的形态学梯度操作
    def test_morphological_gradient02(self):
        # 创建一个二维 NumPy 数组作为输入图像
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        # 定义形态学操作的脚印和结构元素
        footprint = [[1, 0, 1], [1, 1, 0]]
        structure = [[0, 0, 0], [0, 0, 0]]
        # 对输入图像进行灰度膨胀操作，指定结构元素
        tmp1 = ndimage.grey_dilation(array, footprint=footprint,
                                     structure=structure)
        # 对输入图像进行灰度腐蚀操作，指定结构元素
        tmp2 = ndimage.grey_erosion(array, footprint=footprint,
                                    structure=structure)
        # 计算形态学梯度，即膨胀后图像减去腐蚀后图像
        expected = tmp1 - tmp2
        # 计算形态学梯度并将结果作为输出
        output = ndimage.morphological_gradient(array, footprint=footprint,
                                                structure=structure)
        # 断言预期输出与实际输出的数组近似相等
        assert_array_almost_equal(expected, output)
    def test_morphological_laplace01(self):
        # 创建一个二维数组作为测试数据
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        # 定义腐蚀和膨胀的结构元素
        footprint = [[1, 0, 1], [1, 1, 0]]
        structure = [[0, 0, 0], [0, 0, 0]]
        # 进行灰度膨胀操作
        tmp1 = ndimage.grey_dilation(array, footprint=footprint,
                                     structure=structure)
        # 进行灰度腐蚀操作
        tmp2 = ndimage.grey_erosion(array, footprint=footprint,
                                    structure=structure)
        # 计算期望的膨胀和腐蚀的差，用于后续 Laplace 变换的预期输出
        expected = tmp1 + tmp2 - 2 * array
        # 创建一个与输入数组相同形状和数据类型的零数组，用于接收 Laplace 变换的输出
        output = np.zeros(array.shape, array.dtype)
        # 进行形态学 Laplace 变换，并将结果存入 output 中
        ndimage.morphological_laplace(array, footprint=footprint,
                                      structure=structure, output=output)
        # 使用 NumPy 的数组比较函数检查预期输出与实际输出的近似程度
        assert_array_almost_equal(expected, output)

    def test_morphological_laplace02(self):
        # 创建一个二维数组作为测试数据
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        # 定义腐蚀和膨胀的结构元素
        footprint = [[1, 0, 1], [1, 1, 0]]
        structure = [[0, 0, 0], [0, 0, 0]]
        # 进行灰度膨胀操作
        tmp1 = ndimage.grey_dilation(array, footprint=footprint,
                                     structure=structure)
        # 进行灰度腐蚀操作
        tmp2 = ndimage.grey_erosion(array, footprint=footprint,
                                    structure=structure)
        # 计算期望的膨胀和腐蚀的差，用于后续 Laplace 变换的预期输出
        expected = tmp1 + tmp2 - 2 * array
        # 进行形态学 Laplace 变换，并将结果作为函数返回值
        output = ndimage.morphological_laplace(array, footprint=footprint,
                                               structure=structure)
        # 使用 NumPy 的数组比较函数检查预期输出与实际输出的近似程度
        assert_array_almost_equal(expected, output)

    def test_white_tophat01(self):
        # 创建一个二维数组作为测试数据
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        # 定义开运算的结构元素
        footprint = [[1, 0, 1], [1, 1, 0]]
        structure = [[0, 0, 0], [0, 0, 0]]
        # 进行灰度开运算操作
        tmp = ndimage.grey_opening(array, footprint=footprint,
                                   structure=structure)
        # 计算期望的白帽变换输出，即原始数组与开运算结果的差
        expected = array - tmp
        # 创建一个与输入数组相同形状和数据类型的零数组，用于接收白帽变换的输出
        output = np.zeros(array.shape, array.dtype)
        # 进行白帽变换，并将结果存入 output 中
        ndimage.white_tophat(array, footprint=footprint,
                             structure=structure, output=output)
        # 使用 NumPy 的数组比较函数检查预期输出与实际输出的近似程度
        assert_array_almost_equal(expected, output)

    def test_white_tophat02(self):
        # 创建一个二维数组作为测试数据
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        # 定义开运算的结构元素
        footprint = [[1, 0, 1], [1, 1, 0]]
        structure = [[0, 0, 0], [0, 0, 0]]
        # 进行灰度开运算操作
        tmp = ndimage.grey_opening(array, footprint=footprint,
                                   structure=structure)
        # 计算期望的白帽变换输出，即原始数组与开运算结果的差
        expected = array - tmp
        # 进行白帽变换，并将结果作为函数返回值
        output = ndimage.white_tophat(array, footprint=footprint,
                                      structure=structure)
        # 使用 NumPy 的数组比较函数检查预期输出与实际输出的近似程度
        assert_array_almost_equal(expected, output)
    def test_white_tophat03(self):
        # 创建一个布尔类型的二维数组作为测试输入
        array = np.array([[1, 0, 0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 1, 1, 0],
                          [0, 1, 1, 1, 1, 1, 0],
                          [0, 1, 1, 1, 1, 1, 0],
                          [0, 1, 1, 1, 0, 1, 0],
                          [0, 1, 1, 1, 1, 1, 0],
                          [0, 0, 0, 0, 0, 0, 1]], dtype=np.bool_)
        # 创建一个布尔类型的 3x3 全为 True 的结构元素
        structure = np.ones((3, 3), dtype=np.bool_)
        # 预期的输出结果，布尔类型的二维数组
        expected = np.array([[0, 1, 1, 0, 0, 0, 0],
                             [1, 0, 0, 1, 1, 1, 0],
                             [1, 0, 0, 1, 1, 1, 0],
                             [0, 1, 1, 0, 0, 0, 1],
                             [0, 1, 1, 0, 1, 0, 1],
                             [0, 1, 1, 0, 0, 0, 1],
                             [0, 0, 0, 1, 1, 1, 1]], dtype=np.bool_)

        # 使用 ndimage 库中的 white_tophat 函数进行操作，输出与预期结果进行比较
        output = ndimage.white_tophat(array, structure=structure)
        assert_array_equal(expected, output)

    def test_white_tophat04(self):
        # 创建一个对角线为 True 的 5x5 布尔类型数组作为测试输入
        array = np.eye(5, dtype=np.bool_)
        # 创建一个布尔类型的 3x3 全为 True 的结构元素
        structure = np.ones((3, 3), dtype=np.bool_)

        # 创建一个与输入数组类型不匹配的空数组作为输出
        output = np.empty_like(array, dtype=np.float64)
        # 调用 ndimage 库中的 white_tophat 函数，检查类型不匹配的处理
        ndimage.white_tophat(array, structure=structure, output=output)

    def test_black_tophat01(self):
        # 创建一个 3x5 的整数数组作为测试输入
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        # 创建一个自定义的 2x3 结构元素
        footprint = [[1, 0, 1], [1, 1, 0]]
        # 创建一个 2x3 的零矩阵作为结构元素
        structure = [[0, 0, 0], [0, 0, 0]]
        # 使用 ndimage 库中的 grey_closing 函数进行操作，保存临时结果
        tmp = ndimage.grey_closing(array, footprint=footprint,
                                   structure=structure)
        # 计算期望的输出结果
        expected = tmp - array
        # 创建一个与输入数组相同大小和类型的零数组作为输出
        output = np.zeros(array.shape, array.dtype)
        # 调用 ndimage 库中的 black_tophat 函数，输出与期望结果进行比较
        ndimage.black_tophat(array, footprint=footprint,
                             structure=structure, output=output)
        assert_array_almost_equal(expected, output)

    def test_black_tophat02(self):
        # 创建一个 3x5 的整数数组作为测试输入
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        # 创建一个自定义的 2x3 结构元素
        footprint = [[1, 0, 1], [1, 1, 0]]
        # 创建一个 2x3 的零矩阵作为结构元素
        structure = [[0, 0, 0], [0, 0, 0]]
        # 使用 ndimage 库中的 grey_closing 函数进行操作，保存临时结果
        tmp = ndimage.grey_closing(array, footprint=footprint,
                                   structure=structure)
        # 计算期望的输出结果
        expected = tmp - array
        # 调用 ndimage 库中的 black_tophat 函数，直接返回结果与期望结果进行比较
        output = ndimage.black_tophat(array, footprint=footprint,
                                      structure=structure)
        assert_array_almost_equal(expected, output)
    `
        def test_black_tophat03(self):
            # 创建一个二维布尔数组作为输入数据
            array = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 1, 1, 1, 1, 0],
                              [0, 1, 1, 1, 1, 1, 0],
                              [0, 1, 1, 1, 1, 1, 0],
                              [0, 1, 1, 1, 0, 1, 0],
                              [0, 1, 1, 1, 1, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]], dtype=np.bool_)
            # 创建一个二维布尔结构元素作为结构参数
            structure = np.ones((3, 3), dtype=np.bool_)
            # 创建一个预期的输出结果作为参考
            expected = np.array([[0, 1, 1, 1, 1, 1, 1],
                                 [1, 0, 0, 0, 0, 0, 1],
                                 [1, 0, 0, 0, 0, 0, 1],
                                 [1, 0, 0, 0, 0, 0, 1],
                                 [1, 0, 0, 0, 1, 0, 1],
                                 [1, 0, 0, 0, 0, 0, 1],
                                 [1, 1, 1, 1, 1, 1, 0]], dtype=np.bool_)
            # 调用黑帽操作函数并验证输出是否符合预期
            output = ndimage.black_tophat(array, structure=structure)
            assert_array_equal(expected, output)
    
        def test_black_tophat04(self):
            # 创建一个对角线布尔矩阵作为输入数据
            array = np.eye(5, dtype=np.bool_)
            # 创建一个二维布尔结构元素作为结构参数
            structure = np.ones((3, 3), dtype=np.bool_)
    
            # 检查类型不匹配情况下的处理是否正确
            output = np.empty_like(array, dtype=np.float64)
            ndimage.black_tophat(array, structure=structure, output=output)
    
        @pytest.mark.parametrize('dtype', types)
        def test_hit_or_miss01(self, dtype):
            # 定义一个二值化结构元素
            struct = [[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]]
            # 创建一个预期的输出结果作为参考
            expected = [[0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]]
            # 创建一个二维数组作为输入数据
            data = np.array([[0, 1, 0, 0, 0],
                             [1, 1, 1, 0, 0],
                             [0, 1, 0, 1, 1],
                             [0, 0, 1, 1, 1],
                             [0, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1],
                             [0, 1, 1, 1, 1],
                             [0, 0, 0, 0, 0]], dtype)
            # 创建一个布尔类型的输出数组
            out = np.zeros(data.shape, bool)
            # 调用二值化命中或者未命中操作函数
            ndimage.binary_hit_or_miss(data, struct, output=out)
            # 验证输出结果是否与预期一致
            assert_array_almost_equal(expected, out)
    
        @pytest.mark.parametrize('dtype', types)
        def test_hit_or_miss02(self, dtype):
            # 定义一个二值化结构元素
            struct = [[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]]
            # 创建一个预期的输出结果作为参考
            expected = [[0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0]]
            # 创建一个二维数组作为输入数据
            data = np.array([[0, 1, 0, 0, 1, 1, 1, 0],
                             [1, 1, 1, 0, 0, 1, 0, 0],
                             [0, 1, 0, 1, 1, 1, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
            # 调用二值化命中或者未命中操作函数并返回结果
            out = ndimage.binary_hit_or_miss(data, struct)
            # 验证输出结果是否与预期一致
            assert_array_almost_equal(expected, out)
    
        @pytest.mark.parametrize('dtype', types)
    # 定义一个测试方法，用于测试二值图像的击中或不击中操作
    def test_hit_or_miss03(self, dtype):
        # 定义结构元素1，这是一个3x3的二维数组
        struct1 = [[0, 0, 0],
                   [1, 1, 1],
                   [0, 0, 0]]
        # 定义结构元素2，也是一个3x3的二维数组
        struct2 = [[1, 1, 1],
                   [0, 0, 0],
                   [1, 1, 1]]
        # 预期输出，一个8x8的二维数组
        expected = [[0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        # 定义一个8x8的二维数组，表示输入的数据
        data = np.array([[0, 1, 0, 0, 1, 1, 1, 0],
                         [1, 1, 1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0, 1, 1, 0],
                         [0, 0, 0, 0, 1, 1, 1, 0],
                         [0, 1, 1, 1, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        # 使用 ndimage 模块中的二值击中或不击中函数进行处理
        out = ndimage.binary_hit_or_miss(data, struct1, struct2)
        # 断言输出的结果与预期结果几乎相等
        assert_array_almost_equal(expected, out)
class TestDilateFix:

    def setup_method(self):
        # dilation related setup
        self.array = np.array([[0, 0, 0, 0, 0],  # 创建一个5x5的NumPy数组，表示二进制图像
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0],
                               [0, 0, 1, 1, 0],
                               [0, 0, 0, 0, 0]], dtype=np.uint8)

        self.sq3x3 = np.ones((3, 3))  # 创建一个3x3的全为1的结构元素
        # 对self.array进行二进制膨胀操作，使用3x3的结构元素
        dilated3x3 = ndimage.binary_dilation(self.array, structure=self.sq3x3)
        self.dilated3x3 = dilated3x3.view(np.uint8)  # 将结果转换为uint8类型保存

    def test_dilation_square_structure(self):
        result = ndimage.grey_dilation(self.array, structure=self.sq3x3)
        # 断言结果与预期的二进制膨胀结果相加1后近似相等
        assert_array_almost_equal(result, self.dilated3x3 + 1)

    def test_dilation_scalar_size(self):
        result = ndimage.grey_dilation(self.array, size=3)  # 使用标量尺寸进行灰度膨胀
        assert_array_almost_equal(result, self.dilated3x3)


class TestBinaryOpeningClosing:

    def setup_method(self):
        a = np.zeros((5, 5), dtype=bool)  # 创建一个5x5的布尔类型全0数组
        a[1:4, 1:4] = True  # 设置部分区域为True
        a[4, 4] = True  # 设置单个元素为True
        self.array = a  # 保存布尔数组到实例变量
        self.sq3x3 = np.ones((3, 3))  # 创建一个3x3的全为1的结构元素
        # 对self.array进行二进制开运算，使用3x3的结构元素
        self.opened_old = ndimage.binary_opening(self.array, self.sq3x3,
                                                 1, None, 0)
        # 对self.array进行二进制闭运算，使用3x3的结构元素
        self.closed_old = ndimage.binary_closing(self.array, self.sq3x3,
                                                 1, None, 0)

    def test_opening_new_arguments(self):
        opened_new = ndimage.binary_opening(self.array, self.sq3x3, 1, None,
                                            0, None, 0, False)
        # 断言新参数设置下的开运算结果与旧的相等
        assert_array_equal(opened_new, self.opened_old)

    def test_closing_new_arguments(self):
        closed_new = ndimage.binary_closing(self.array, self.sq3x3, 1, None,
                                            0, None, 0, False)
        # 断言新参数设置下的闭运算结果与旧的相等
        assert_array_equal(closed_new, self.closed_old)


def test_binary_erosion_noninteger_iterations():
    # 回归测试gh-9905, gh-9909：测试对非整数迭代次数抛出TypeError异常
    data = np.ones([1])
    assert_raises(TypeError, ndimage.binary_erosion, data, iterations=0.5)
    assert_raises(TypeError, ndimage.binary_erosion, data, iterations=1.5)


def test_binary_dilation_noninteger_iterations():
    # 回归测试gh-9905, gh-9909：测试对非整数迭代次数抛出TypeError异常
    data = np.ones([1])
    assert_raises(TypeError, ndimage.binary_dilation, data, iterations=0.5)
    assert_raises(TypeError, ndimage.binary_dilation, data, iterations=1.5)


def test_binary_opening_noninteger_iterations():
    # 回归测试gh-9905, gh-9909：测试对非整数迭代次数抛出TypeError异常
    data = np.ones([1])
    assert_raises(TypeError, ndimage.binary_opening, data, iterations=0.5)
    assert_raises(TypeError, ndimage.binary_opening, data, iterations=1.5)


def test_binary_closing_noninteger_iterations():
    # 回归测试gh-9905, gh-9909：测试对非整数迭代次数抛出TypeError异常
    data = np.ones([1])
    assert_raises(TypeError, ndimage.binary_closing, data, iterations=0.5)
    assert_raises(TypeError, ndimage.binary_closing, data, iterations=1.5)
    # 断言测试：检查调用 `ndimage.binary_closing` 函数时参数 `iterations` 为非整数时是否引发 TypeError 异常
    assert_raises(TypeError, ndimage.binary_closing, data, iterations=0.5)
    # 断言测试：检查调用 `ndimage.binary_closing` 函数时参数 `iterations` 为非整数时是否引发 TypeError 异常
    assert_raises(TypeError, ndimage.binary_closing, data, iterations=1.5)
def test_binary_closing_noninteger_brute_force_passes_when_true():
    # 回归测试 gh-9905, gh-9909：对于非整数迭代，确保不会引发 ValueError
    data = np.ones([1])

    # 测试当 brute_force 参数为 1.5 时的二值侵蚀操作
    assert ndimage.binary_erosion(
        data, iterations=2, brute_force=1.5
    ) == ndimage.binary_erosion(data, iterations=2, brute_force=bool(1.5))
    
    # 测试当 brute_force 参数为 0.0 时的二值侵蚀操作
    assert ndimage.binary_erosion(
        data, iterations=2, brute_force=0.0
    ) == ndimage.binary_erosion(data, iterations=2, brute_force=bool(0.0))


@pytest.mark.parametrize(
    'function',
    ['binary_erosion', 'binary_dilation', 'binary_opening', 'binary_closing'],
)
@pytest.mark.parametrize('iterations', [1, 5])
@pytest.mark.parametrize('brute_force', [False, True])
def test_binary_input_as_output(function, iterations, brute_force):
    rstate = np.random.RandomState(123)
    data = rstate.randint(low=0, high=2, size=100).astype(bool)
    ndi_func = getattr(ndimage, function)

    # 检查输入数据是否没有被修改
    data_orig = data.copy()
    
    # 获取预期结果
    expected = ndi_func(data, brute_force=brute_force, iterations=iterations)
    assert_array_equal(data, data_orig)

    # 现在 data 应包含预期结果
    ndi_func(data, brute_force=brute_force, iterations=iterations, output=data)
    assert_array_equal(expected, data)


def test_binary_hit_or_miss_input_as_output():
    rstate = np.random.RandomState(123)
    data = rstate.randint(low=0, high=2, size=100).astype(bool)

    # 检查输入数据是否没有被修改
    data_orig = data.copy()
    
    # 获取预期结果
    expected = ndimage.binary_hit_or_miss(data)
    assert_array_equal(data, data_orig)

    # 现在 data 应包含预期结果
    ndimage.binary_hit_or_miss(data, output=data)
    assert_array_equal(expected, data)


def test_distance_transform_cdt_invalid_metric():
    msg = 'invalid metric provided'
    
    # 使用 pytest 检查是否会引发 ValueError，并匹配特定消息
    with pytest.raises(ValueError, match=msg):
        ndimage.distance_transform_cdt(np.ones((5, 5)),
                                       metric="garbage")
```