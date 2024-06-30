# `D:\src\scipysrc\scipy\scipy\ndimage\tests\test_interpolation.py`

```
# 导入系统模块
import sys

# 导入 NumPy 库，并且从中导入多个断言函数
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
                           assert_array_almost_equal, assert_allclose,
                           suppress_warnings)

# 导入 pytest 库，并且从中导入 raises 函数
import pytest
from pytest import raises as assert_raises

# 导入 SciPy 的图像处理模块
import scipy.ndimage as ndimage

# 导入当前包中的 types 模块
from . import types

# 定义一个极小的值 eps
eps = 1e-12

# 定义一个字典，将 SciPy 中的图像处理模块的插值模式映射到 NumPy 中对应的模式
ndimage_to_numpy_mode = {
    'mirror': 'reflect',
    'reflect': 'symmetric',
    'grid-mirror': 'symmetric',
    'grid-wrap': 'wrap',
    'nearest': 'edge',
    'grid-constant': 'constant',
}

# 定义测试类 TestNdimageInterpolation
class TestNdimageInterpolation:

    # 使用 pytest.mark.parametrize 装饰器标记参数化测试
    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [1.5, 2.5, 3.5, 4, 4, 4, 4]),
         ('wrap', [1.5, 2.5, 3.5, 1.5, 2.5, 3.5, 1.5]),
         ('grid-wrap', [1.5, 2.5, 3.5, 2.5, 1.5, 2.5, 3.5]),
         ('mirror', [1.5, 2.5, 3.5, 3.5, 2.5, 1.5, 1.5]),
         ('reflect', [1.5, 2.5, 3.5, 4, 3.5, 2.5, 1.5]),
         ('constant', [1.5, 2.5, 3.5, -1, -1, -1, -1]),
         ('grid-constant', [1.5, 2.5, 3.5, 1.5, -1, -1, -1])]
    )
    # 定义测试方法 test_boundaries
    def test_boundaries(self, mode, expected_value):
        # 定义内部函数 shift，用于数据偏移
        def shift(x):
            return (x[0] + 0.5,)

        # 创建 NumPy 数组 data
        data = np.array([1, 2, 3, 4.])
        
        # 调用 ndimage.geometric_transform 函数进行几何变换，验证期望结果
        assert_array_equal(
            expected_value,
            ndimage.geometric_transform(data, shift, cval=-1, mode=mode,
                                        output_shape=(7,), order=1))

    # 使用 pytest.mark.parametrize 装饰器标记参数化测试
    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [1, 1, 2, 3]),
         ('wrap', [3, 1, 2, 3]),
         ('grid-wrap', [4, 1, 2, 3]),
         ('mirror', [2, 1, 2, 3]),
         ('reflect', [1, 1, 2, 3]),
         ('constant', [-1, 1, 2, 3]),
         ('grid-constant', [-1, 1, 2, 3])]
    )
    # 定义测试方法 test_boundaries2
    def test_boundaries2(self, mode, expected_value):
        # 定义内部函数 shift，用于数据偏移
        def shift(x):
            return (x[0] - 0.9,)

        # 创建 NumPy 数组 data
        data = np.array([1, 2, 3, 4])
        
        # 调用 ndimage.geometric_transform 函数进行几何变换，验证期望结果
        assert_array_equal(
            expected_value,
            ndimage.geometric_transform(data, shift, cval=-1, mode=mode,
                                        output_shape=(4,)))

    # 使用 pytest.mark.parametrize 装饰器标记参数化测试
    @pytest.mark.parametrize('mode', ['mirror', 'reflect', 'grid-mirror',
                                      'grid-wrap', 'grid-constant',
                                      'nearest'])
    @pytest.mark.parametrize('order', range(6))
    # 定义测试方法 test_boundary_spline_accuracy
    def test_boundary_spline_accuracy(self, mode, order):
        """Tests based on examples from gh-2640"""
        # 创建 NumPy 数组 data
        data = np.arange(-6, 7, dtype=float)
        # 创建数组 x 和 y，调用 ndimage.map_coordinates 进行插值
        x = np.linspace(-8, 15, num=1000)
        y = ndimage.map_coordinates(data, [x], order=order, mode=mode)

        # 使用 np.pad 对数据进行显式填充，计算期望值
        npad = 32
        pad_mode = ndimage_to_numpy_mode.get(mode)
        padded = np.pad(data, npad, mode=pad_mode)
        expected = ndimage.map_coordinates(padded, [npad + x], order=order,
                                           mode=mode)

        # 根据不同的模式设置容差 atol
        atol = 1e-5 if mode == 'grid-constant' else 1e-12
        # 使用 assert_allclose 断言实际值与期望值的接近程度
        assert_allclose(y, expected, rtol=1e-7, atol=atol)
    `
        # 使用 pytest 的参数化功能，传入 order 参数范围 2 到 5
        @pytest.mark.parametrize('order', range(2, 6))
        # 使用 pytest 的参数化功能，传入 dtype 参数，值为 types
        @pytest.mark.parametrize('dtype', types)
        # 定义测试函数 test_spline01，接收 dtype 和 order 两个参数
        def test_spline01(self, dtype, order):
            # 创建一个空数组，数据类型为 dtype
            data = np.ones([], dtype)
            # 调用 ndimage.spline_filter 函数进行样条滤波，指定 order 为参数
            out = ndimage.spline_filter(data, order=order)
            # 断言输出结果与 1 的数组近似相等
            assert_array_almost_equal(out, 1)
    
        # 使用 pytest 的参数化功能，传入 order 参数范围 2 到 5
        @pytest.mark.parametrize('order', range(2, 6))
        # 使用 pytest 的参数化功能，传入 dtype 参数，值为 types
        @pytest.mark.parametrize('dtype', types)
        # 定义测试函数 test_spline02，接收 dtype 和 order 两个参数
        def test_spline02(self, dtype, order):
            # 创建一个包含一个元素的数组，数据类型为 dtype
            data = np.array([1], dtype)
            # 调用 ndimage.spline_filter 函数进行样条滤波，指定 order 为参数
            out = ndimage.spline_filter(data, order=order)
            # 断言输出结果与 [1] 的数组近似相等
            assert_array_almost_equal(out, [1])
    
        # 使用 pytest 的参数化功能，传入 order 参数范围 2 到 5
        @pytest.mark.parametrize('order', range(2, 6))
        # 使用 pytest 的参数化功能，传入 dtype 参数，值为 types
        @pytest.mark.parametrize('dtype', types)
        # 定义测试函数 test_spline03，接收 dtype 和 order 两个参数
        def test_spline03(self, dtype, order):
            # 创建一个空数组，数据类型为 dtype
            data = np.ones([], dtype)
            # 调用 ndimage.spline_filter 函数进行样条滤波，指定 order 和 output 为 dtype
            out = ndimage.spline_filter(data, order, output=dtype)
            # 断言输出结果与 1 的数组近似相等
            assert_array_almost_equal(out, 1)
    
        # 使用 pytest 的参数化功能，传入 order 参数范围 2 到 5
        @pytest.mark.parametrize('order', range(2, 6))
        # 使用 pytest 的参数化功能，传入 dtype 参数，值为 types
        @pytest.mark.parametrize('dtype', types)
        # 定义测试函数 test_spline04，接收 dtype 和 order 两个参数
        def test_spline04(self, dtype, order):
            # 创建一个包含四个元素的数组，数据类型为 dtype
            data = np.ones([4], dtype)
            # 调用 ndimage.spline_filter 函数进行样条滤波，指定 order 为参数
            out = ndimage.spline_filter(data, order)
            # 断言输出结果与 [1, 1, 1, 1] 的数组近似相等
            assert_array_almost_equal(out, [1, 1, 1, 1])
    
        # 使用 pytest 的参数化功能，传入 order 参数范围 2 到 5
        @pytest.mark.parametrize('order', range(2, 6))
        # 使用 pytest 的参数化功能，传入 dtype 参数，值为 types
        @pytest.mark.parametrize('dtype', types)
        # 定义测试函数 test_spline05，接收 dtype 和 order 两个参数
        def test_spline05(self, dtype, order):
            # 创建一个包含四行四列的数组，数据类型为 dtype
            data = np.ones([4, 4], dtype)
            # 调用 ndimage.spline_filter 函数进行样条滤波，指定 order 为参数
            out = ndimage.spline_filter(data, order=order)
            # 断言输出结果与指定的二维数组近似相等
            assert_array_almost_equal(out, [[1, 1, 1, 1],
                                            [1, 1, 1, 1],
                                            [1, 1, 1, 1],
                                            [1, 1, 1, 1]])
    
        # 使用 pytest 的参数化功能，传入 order 参数范围 0 到 5
        @pytest.mark.parametrize('order', range(0, 6))
        # 定义测试函数 test_geometric_transform01，接收 order 参数
        def test_geometric_transform01(self, order):
            # 创建一个包含一个元素的数组
            data = np.array([1])
    
            # 定义一个映射函数，返回输入值 x
            def mapping(x):
                return x
    
            # 调用 ndimage.geometric_transform 函数，执行几何变换，传入数据，映射函数，数据形状和 order
            out = ndimage.geometric_transform(data, mapping, data.shape,
                                              order=order)
            # 断言输出结果与 [1] 的数组近似相等
            assert_array_almost_equal(out, [1])
    
        # 使用 pytest 的参数化功能，传入 order 参数范围 0 到 5
        @pytest.mark.parametrize('order', range(0, 6))
        # 定义测试函数 test_geometric_transform02，接收 order 参数
        def test_geometric_transform02(self, order):
            # 创建一个包含四个元素的数组，数据类型为 dtype
            data = np.ones([4])
    
            # 定义一个映射函数，返回输入值 x
            def mapping(x):
                return x
    
            # 调用 ndimage.geometric_transform 函数，执行几何变换，传入数据，映射函数，数据形状和 order
            out = ndimage.geometric_transform(data, mapping, data.shape,
                                              order=order)
            # 断言输出结果与 [1, 1, 1, 1] 的数组近似相等
            assert_array_almost_equal(out, [1, 1, 1, 1])
    
        # 使用 pytest 的参数化功能，传入 order 参数范围 0 到 5
        @pytest.mark.parametrize('order', range(0, 6))
        # 定义测试函数 test_geometric_transform03，接收 order 参数
        def test_geometric_transform03(self, order):
            # 创建一个包含四个元素的数组，数据类型为 dtype
            data = np.ones([4])
    
            # 定义一个映射函数，返回输入值 x 的第一个元素减去 1 的元组
            def mapping(x):
                return (x[0] - 1,)
    
            # 调用 ndimage.geometric_transform 函数，执行几何变换，传入数据，映射函数，数据形状和 order
            out = ndimage.geometric_transform(data, mapping, data.shape,
                                              order=order)
            # 断言输出结果与 [0, 1, 1, 1] 的数组近似相等
            assert_array_almost_equal(out, [0, 1, 1, 1])
    
        # 使用 pytest 的参数化功能，传入 order 参数范围 0 到 5
        @pytest.mark.parametrize('order', range(0, 6))
        # 定义测试函数 test_geometric_transform04，接收 order 参数
        def test_geometric_transform04(self, order):
            # 创建一个包含四个元素的数组，数据类型为 dtype
            data = np.array([4, 1, 3, 2])
    
            # 定义一个映射函数，返回输入值 x 的第一个元素减去 1 的元组
            def mapping(x):
                return (x[0] - 1,)
    
            # 调用 ndimage.geometric_transform 函数，执行几何变换，传入数据，映射函数，数据形状和 order
            out = ndimage.geometric_transform(data, mapping, data.shape,
                                              order=order)
            # 断言输出结果与 [0, 4, 1, 3] 的数组近似相等
            assert_array_almost_equal(out, [0, 4, 1, 3])
    # 使用 pytest 的参数化装饰器，为 'order' 参数指定范围从 0 到 5
    # 同时为 'dtype' 参数指定了两种类型：np.float64 和 np.complex128
    @pytest.mark.parametrize('order', range(0, 6))
    @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
    # 定义测试函数 test_geometric_transform05，接受 'order' 和 'dtype' 作为参数
    def test_geometric_transform05(self, order, dtype):
        # 创建一个 numpy 数组 'data'，数据类型为 'dtype'
        data = np.array([[1, 1, 1, 1],
                         [1, 1, 1, 1],
                         [1, 1, 1, 1]], dtype=dtype)
        # 创建一个预期结果的 numpy 数组 'expected'，数据类型与 'data' 相同
        expected = np.array([[0, 1, 1, 1],
                             [0, 1, 1, 1],
                             [0, 1, 1, 1]], dtype=dtype)
        # 如果 'data' 的数据类型为复数，则对 'data' 和 'expected' 进行调整
        if data.dtype.kind == 'c':
            data -= 1j * data
            expected -= 1j * expected
    
        # 定义一个内部函数 'mapping'，接受参数 x，返回元组 (x[0], x[1] - 1)
        def mapping(x):
            return (x[0], x[1] - 1)
    
        # 使用 ndimage.geometric_transform 对 'data' 应用 'mapping' 函数进行几何变换，指定变换顺序为 'order'
        out = ndimage.geometric_transform(data, mapping, data.shape, order=order)
        # 断言 'out' 和 'expected' 的元素近似相等
        assert_array_almost_equal(out, expected)
    
    
    # 使用 pytest 的参数化装饰器，为 'order' 参数指定范围从 0 到 5
    @pytest.mark.parametrize('order', range(0, 6))
    # 定义测试函数 test_geometric_transform06，接受 'order' 作为参数
    def test_geometric_transform06(self, order):
        # 创建一个 numpy 数组 'data'
        data = np.array([[4, 1, 3, 2],
                         [7, 6, 8, 5],
                         [3, 5, 3, 6]])
    
        # 定义一个内部函数 'mapping'，接受参数 x，返回元组 (x[0], x[1] - 1)
        def mapping(x):
            return (x[0], x[1] - 1)
    
        # 使用 ndimage.geometric_transform 对 'data' 应用 'mapping' 函数进行几何变换，指定变换顺序为 'order'
        out = ndimage.geometric_transform(data, mapping, data.shape, order=order)
        # 断言 'out' 和预期的二维数组的元素近似相等
        assert_array_almost_equal(out, [[0, 4, 1, 3],
                                        [0, 7, 6, 8],
                                        [0, 3, 5, 3]])
    
    
    # 使用 pytest 的参数化装饰器，为 'order' 参数指定范围从 0 到 5
    @pytest.mark.parametrize('order', range(0, 6))
    # 定义测试函数 test_geometric_transform07，接受 'order' 作为参数
    def test_geometric_transform07(self, order):
        # 创建一个 numpy 数组 'data'
        data = np.array([[4, 1, 3, 2],
                         [7, 6, 8, 5],
                         [3, 5, 3, 6]])
    
        # 定义一个内部函数 'mapping'，接受参数 x，返回元组 (x[0] - 1, x[1])
        def mapping(x):
            return (x[0] - 1, x[1])
    
        # 使用 ndimage.geometric_transform 对 'data' 应用 'mapping' 函数进行几何变换，指定变换顺序为 'order'
        out = ndimage.geometric_transform(data, mapping, data.shape, order=order)
        # 断言 'out' 和预期的二维数组的元素近似相等
        assert_array_almost_equal(out, [[0, 0, 0, 0],
                                        [4, 1, 3, 2],
                                        [7, 6, 8, 5]])
    
    
    # 使用 pytest 的参数化装饰器，为 'order' 参数指定范围从 0 到 5
    @pytest.mark.parametrize('order', range(0, 6))
    # 定义测试函数 test_geometric_transform08，接受 'order' 作为参数
    def test_geometric_transform08(self, order):
        # 创建一个 numpy 数组 'data'
        data = np.array([[4, 1, 3, 2],
                         [7, 6, 8, 5],
                         [3, 5, 3, 6]])
    
        # 定义一个内部函数 'mapping'，接受参数 x，返回元组 (x[0] - 1, x[1] - 1)
        def mapping(x):
            return (x[0] - 1, x[1] - 1)
    
        # 使用 ndimage.geometric_transform 对 'data' 应用 'mapping' 函数进行几何变换，指定变换顺序为 'order'
        out = ndimage.geometric_transform(data, mapping, data.shape, order=order)
        # 断言 'out' 和预期的二维数组的元素近似相等
        assert_array_almost_equal(out, [[0, 0, 0, 0],
                                        [0, 4, 1, 3],
                                        [0, 7, 6, 8]])
    # 定义一个测试函数，用于测试几何变换函数的效果
    def test_geometric_transform10(self, order):
        # 创建一个二维数组作为测试数据
        data = np.array([[4, 1, 3, 2],
                         [7, 6, 8, 5],
                         [3, 5, 3, 6]])

        # 定义一个映射函数，将每个元素的坐标减去1
        def mapping(x):
            return (x[0] - 1, x[1] - 1)

        # 根据变换阶数选择是否进行样条滤波处理
        if (order > 1):
            filtered = ndimage.spline_filter(data, order=order)
        else:
            filtered = data

        # 对数据进行几何变换，使用定义好的映射函数，并进行断言验证结果
        out = ndimage.geometric_transform(filtered, mapping, data.shape,
                                          order=order, prefilter=False)
        assert_array_almost_equal(out, [[0, 0, 0, 0],
                                        [0, 4, 1, 3],
                                        [0, 7, 6, 8]])

    # 使用参数化测试标记，定义第二个测试函数
    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform13(self, order):
        # 创建一个长度为2的全1浮点数数组作为测试数据
        data = np.ones([2], np.float64)

        # 定义一个映射函数，将每个元素的值除以2
        def mapping(x):
            return (x[0] // 2,)

        # 对数据进行几何变换，使用定义好的映射函数，并进行断言验证结果
        out = ndimage.geometric_transform(data, mapping, [4], order=order)
        assert_array_almost_equal(out, [1, 1, 1, 1])

    # 使用参数化测试标记，定义第三个测试函数
    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform14(self, order):
        # 创建一个包含8个整数的列表作为测试数据
        data = [1, 5, 2, 6, 3, 7, 4, 4]

        # 定义一个映射函数，将每个元素的值乘以2
        def mapping(x):
            return (2 * x[0],)

        # 对数据进行几何变换，使用定义好的映射函数，并进行断言验证结果
        out = ndimage.geometric_transform(data, mapping, [4], order=order)
        assert_array_almost_equal(out, [1, 2, 3, 4])

    # 使用参数化测试标记，定义第四个测试函数
    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform15(self, order):
        # 创建一个包含4个整数的列表作为测试数据
        data = [1, 2, 3, 4]

        # 定义一个映射函数，将每个元素的值除以2
        def mapping(x):
            return (x[0] / 2,)

        # 对数据进行几何变换，使用定义好的映射函数，并进行断言验证结果
        out = ndimage.geometric_transform(data, mapping, [8], order=order)
        assert_array_almost_equal(out[::2], [1, 2, 3, 4])

    # 使用参数化测试标记，定义第五个测试函数
    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform16(self, order):
        # 创建一个包含3个子列表的二维数组作为测试数据
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9.0, 10, 11, 12]]

        # 定义一个映射函数，将每个元素的第二个坐标乘以2
        def mapping(x):
            return (x[0], x[1] * 2)

        # 对数据进行几何变换，使用定义好的映射函数，并进行断言验证结果
        out = ndimage.geometric_transform(data, mapping, (3, 2),
                                          order=order)
        assert_array_almost_equal(out, [[1, 3], [5, 7], [9, 11]])

    # 使用参数化测试标记，定义第六个测试函数
    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform17(self, order):
        # 创建一个包含3个子列表的二维数组作为测试数据
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]

        # 定义一个映射函数，将每个元素的第一个坐标乘以2
        def mapping(x):
            return (x[0] * 2, x[1])

        # 对数据进行几何变换，使用定义好的映射函数，并进行断言验证结果
        out = ndimage.geometric_transform(data, mapping, (1, 4),
                                          order=order)
        assert_array_almost_equal(out, [[1, 2, 3, 4]])

    # 使用参数化测试标记，定义第七个测试函数
    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform18(self, order):
        # 创建一个包含3个子列表的二维数组作为测试数据
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]

        # 定义一个映射函数，将每个元素的两个坐标都乘以2
        def mapping(x):
            return (x[0] * 2, x[1] * 2)

        # 对数据进行几何变换，使用定义好的映射函数，并进行断言验证结果
        out = ndimage.geometric_transform(data, mapping, (1, 2),
                                          order=order)
        assert_array_almost_equal(out, [[1, 3]])

    # 使用参数化测试标记，定义第八个测试函数
    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform19(self, order):
        # 创建一个包含3个子列表的二维数组作为测试数据
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]

        # 定义一个映射函数，将每个元素的第一个坐标乘以2，第二个坐标乘以3
        def mapping(x):
            return (x[0] * 2, x[1] * 3)

        # 对数据进行几何变换，使用定义好的映射函数，并进行断言验证结果
        out = ndimage.geometric_transform(data, mapping, (1, 2),
                                          order=order)
        assert_array_almost_equal(out, [[1, 3]])
    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform19(self, order):
        # 定义输入数据
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]

        # 定义映射函数，将每个元素的第二个维度值除以2
        def mapping(x):
            return (x[0], x[1] / 2)

        # 执行几何变换，将数据按照映射函数进行变换，输出结果的第二个维度长度设置为8
        out = ndimage.geometric_transform(data, mapping, (3, 8),
                                          order=order)
        # 断言输出结果的每一行，步长为2的切片与原始数据相等
        assert_array_almost_equal(out[..., ::2], data)

    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform20(self, order):
        # 定义输入数据
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]

        # 定义映射函数，将每个元素的第一个维度值除以2
        def mapping(x):
            return (x[0] / 2, x[1])

        # 执行几何变换，将数据按照映射函数进行变换，输出结果的第一个维度长度设置为6
        out = ndimage.geometric_transform(data, mapping, (6, 4),
                                          order=order)
        # 断言输出结果的每一列，步长为2的切片与原始数据相等
        assert_array_almost_equal(out[::2, ...], data)

    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform21(self, order):
        # 定义输入数据
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]

        # 定义映射函数，将每个元素的两个维度值都除以2
        def mapping(x):
            return (x[0] / 2, x[1] / 2)

        # 执行几何变换，将数据按照映射函数进行变换，输出结果的第一个和第二个维度长度分别为6和8
        out = ndimage.geometric_transform(data, mapping, (6, 8),
                                          order=order)
        # 断言输出结果的每一行和每一列，步长为2的切片与原始数据相等
        assert_array_almost_equal(out[::2, ::2], data)

    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform22(self, order):
        # 定义输入数据，使用numpy数组和np.float64类型
        data = np.array([[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12]], np.float64)

        # 定义两个映射函数
        def mapping1(x):
            return (x[0] / 2, x[1] / 2)

        def mapping2(x):
            return (x[0] * 2, x[1] * 2)

        # 执行两次几何变换，先按照mapping1进行变换，再按照mapping2进行变换
        out = ndimage.geometric_transform(data, mapping1,
                                          (6, 8), order=order)
        out = ndimage.geometric_transform(out, mapping2,
                                          (3, 4), order=order)
        # 断言输出结果与原始数据相等
        assert_array_almost_equal(out, data)

    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform23(self, order):
        # 定义输入数据
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]

        # 定义映射函数，将每个元素的第一个维度设置为1，第二个维度乘以2
        def mapping(x):
            return (1, x[0] * 2)

        # 执行几何变换，将数据按照映射函数进行变换，输出结果的长度设置为2
        out = ndimage.geometric_transform(data, mapping, (2,), order=order)
        # 将输出结果转换为np.int32类型，然后断言与给定的数组相等
        out = out.astype(np.int32)
        assert_array_almost_equal(out, [5, 7])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform24(self, order):
        # 定义输入数据
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]

        # 定义映射函数，其中包含两个额外参数a和b，返回结果的第一个维度设置为a，第二个维度乘以b
        def mapping(x, a, b):
            return (a, x[0] * b)

        # 执行几何变换，将数据按照映射函数进行变换，输出结果的长度设置为2，并传入额外的参数1和关键字参数'b': 2
        out = ndimage.geometric_transform(
            data, mapping, (2,), order=order, extra_arguments=(1,),
            extra_keywords={'b': 2})
        # 断言输出结果与给定的数组相等
        assert_array_almost_equal(out, [5, 7])
    def test_geometric_transform_grid_constant_order1(self):
        # 验证在原始边界之外的插值
        x = np.array([[1, 2, 3],
                      [4, 5, 6]], dtype=float)

        # 定义一个映射函数，用于减去0.5的偏移
        def mapping(x):
            return (x[0] - 0.5), (x[1] - 0.5)

        # 预期的结果数组
        expected_result = np.array([[0.25, 0.75, 1.25],
                                    [1.25, 3.00, 4.00]])
        # 使用 order=1 进行几何变换，使用 grid-constant 模式
        assert_array_almost_equal(
            ndimage.geometric_transform(x, mapping, mode='grid-constant',
                                        order=1),
            expected_result,
        )

    @pytest.mark.parametrize('mode', ['grid-constant', 'grid-wrap', 'nearest',
                                      'mirror', 'reflect'])
    @pytest.mark.parametrize('order', range(6))
    def test_geometric_transform_vs_padded(self, order, mode):
        # 创建一个12x12的浮点数数组 x
        x = np.arange(144, dtype=float).reshape(12, 12)

        # 定义一个映射函数，将数组中每个元素的坐标做减0.4和加2.3的映射
        def mapping(x):
            return (x[0] - 0.4), (x[1] + 2.3)

        # 手动填充数组，然后在变换后提取中心部分以获取预期结果
        npad = 24
        pad_mode = ndimage_to_numpy_mode.get(mode)
        xp = np.pad(x, npad, mode=pad_mode)
        center_slice = tuple([slice(npad, -npad)] * x.ndim)
        # 使用指定的 mode 和 order 进行几何变换
        expected_result = ndimage.geometric_transform(
            xp, mapping, mode=mode, order=order)[center_slice]

        assert_allclose(
            ndimage.geometric_transform(x, mapping, mode=mode,
                                        order=order),
            expected_result,
            rtol=1e-7,
        )

    def test_geometric_transform_endianness_with_output_parameter(self):
        # 对给定具有非本机字节顺序的输出 ndarray 或 dtype 进行几何变换
        # 参见问题 #4127
        data = np.array([1])

        # 定义一个映射函数，直接返回输入
        def mapping(x):
            return x

        # 循环测试不同的输出类型
        for out in [data.dtype, data.dtype.newbyteorder(),
                    np.empty_like(data),
                    np.empty_like(data).astype(data.dtype.newbyteorder())]:
            # 执行几何变换，将输出指定为 out
            returned = ndimage.geometric_transform(data, mapping, data.shape,
                                                   output=out)
            # 如果返回值为 None，则结果为 out；否则结果为返回值
            result = out if returned is None else returned
            assert_array_almost_equal(result, [1])

    def test_geometric_transform_with_string_output(self):
        data = np.array([1])

        # 定义一个映射函数，直接返回输入
        def mapping(x):
            return x

        # 使用 'f' 作为输出类型进行几何变换
        out = ndimage.geometric_transform(data, mapping, output='f')
        # 断言输出的数据类型为 'f' 类型的浮点数
        assert_(out.dtype is np.dtype('f'))
        assert_array_almost_equal(out, [1])

    @pytest.mark.parametrize('order', range(0, 6))
    @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
    # 定义测试方法，测试 map_coordinates 函数，接受 order 和 dtype 两个参数
    def test_map_coordinates01(self, order, dtype):
        # 创建一个二维 NumPy 数组作为测试数据
        data = np.array([[4, 1, 3, 2],
                         [7, 6, 8, 5],
                         [3, 5, 3, 6]])
        # 创建一个期望的二维 NumPy 数组，用于预期结果对比
        expected = np.array([[0, 0, 0, 0],
                             [0, 4, 1, 3],
                             [0, 7, 6, 8]])
        # 如果数据的类型的种类是 'c'，则进行复数处理
        if data.dtype.kind == 'c':
            data = data - 1j * data
            expected = expected - 1j * expected

        # 创建一个索引数组，包含与 data 相同的形状
        idx = np.indices(data.shape)
        # 将索引数组每个元素减一
        idx -= 1

        # 使用 ndimage 库中的 map_coordinates 函数对 data 进行插值映射，按给定的 order 参数
        out = ndimage.map_coordinates(data, idx, order=order)
        # 断言计算得到的 out 数组与期望的 expected 数组近似相等
        assert_array_almost_equal(out, expected)

    # 使用 pytest.mark.parametrize 装饰器进行参数化测试，测试 map_coordinates 函数
    @pytest.mark.parametrize('order', range(0, 6))
    def test_map_coordinates02(self, order):
        # 创建一个二维 NumPy 数组作为测试数据
        data = np.array([[4, 1, 3, 2],
                         [7, 6, 8, 5],
                         [3, 5, 3, 6]])
        # 创建一个浮点型索引数组，包含与 data 相同的形状
        idx = np.indices(data.shape, np.float64)
        # 将索引数组每个元素减 0.5
        idx -= 0.5

        # 使用 ndimage 库中的 shift 函数对 data 进行平移，按给定的 order 参数
        out1 = ndimage.shift(data, 0.5, order=order)
        # 使用 ndimage 库中的 map_coordinates 函数对 data 进行插值映射，按给定的 order 参数
        out2 = ndimage.map_coordinates(data, idx, order=order)
        # 断言计算得到的 out1 数组与 out2 数组近似相等
        assert_array_almost_equal(out1, out2)

    # 定义测试方法，测试 map_coordinates 函数，不接受任何参数
    def test_map_coordinates03(self):
        # 创建一个按列主序存储的二维 NumPy 数组作为测试数据
        data = np.array([[4, 1, 3, 2],
                         [7, 6, 8, 5],
                         [3, 5, 3, 6]], order='F')
        # 创建一个索引数组，包含与 data 相同的形状，将每个元素减一
        idx = np.indices(data.shape) - 1
        # 使用 ndimage 库中的 map_coordinates 函数对 data 进行插值映射
        out = ndimage.map_coordinates(data, idx)
        # 断言计算得到的 out 数组与期望的二维 NumPy 数组近似相等
        assert_array_almost_equal(out, [[0, 0, 0, 0],
                                        [0, 4, 1, 3],
                                        [0, 7, 6, 8]])
        # 断言计算得到的 out 数组与使用 shift 函数进行平移后的结果近似相等
        assert_array_almost_equal(out, ndimage.shift(data, (1, 1)))
        
        # 对 data 的每隔两行的子数组进行处理
        idx = np.indices(data[::2].shape) - 1
        out = ndimage.map_coordinates(data[::2], idx)
        # 断言计算得到的 out 数组与期望的二维 NumPy 数组近似相等
        assert_array_almost_equal(out, [[0, 0, 0, 0],
                                        [0, 4, 1, 3]])
        # 断言计算得到的 out 数组与使用 shift 函数进行平移后的结果近似相等
        assert_array_almost_equal(out, ndimage.shift(data[::2], (1, 1)))
        
        # 对 data 的每隔两列的子数组进行处理
        idx = np.indices(data[:, ::2].shape) - 1
        out = ndimage.map_coordinates(data[:, ::2], idx)
        # 断言计算得到的 out 数组与期望的二维 NumPy 数组近似相等
        assert_array_almost_equal(out, [[0, 0], [0, 4], [0, 7]])
        # 断言计算得到的 out 数组与使用 shift 函数进行平移后的结果近似相等
        assert_array_almost_equal(out, ndimage.shift(data[:, ::2], (1, 1)))

    # 定义测试方法，测试 map_coordinates 函数对输出参数的字节序进行处理
    def test_map_coordinates_endianness_with_output_parameter(self):
        # 创建一个二维 NumPy 数组作为测试数据
        data = np.array([[1, 2], [7, 6]])
        # 创建一个期望的二维 NumPy 数组，用于预期结果对比
        expected = np.array([[0, 0], [0, 1]])
        # 创建一个索引数组，包含与 data 相同的形状，将每个元素减一
        idx = np.indices(data.shape)
        idx -= 1

        # 对于不同类型的输出参数进行迭代
        for out in [
            data.dtype,  # 使用 data 的数据类型作为输出参数
            data.dtype.newbyteorder(),  # 使用 data 的数据类型反序作为输出参数
            np.empty_like(expected),  # 使用与 expected 相同形状的空数组作为输出参数
            np.empty_like(expected).astype(expected.dtype.newbyteorder())  # 使用反序的 expected 形状空数组作为输出参数
        ]:
            # 使用 ndimage 库中的 map_coordinates 函数对 data 进行插值映射，输出结果存入 out
            returned = ndimage.map_coordinates(data, idx, output=out)
            # 如果返回值为空，则使用 out 本身作为结果
            result = out if returned is None else returned
            # 断言计算得到的 result 数组与期望的 expected 数组近似相等
            assert_array_almost_equal(result, expected)
    def test_map_coordinates_with_string_output(self):
        # 创建一个包含单个元素的 NumPy 数组
        data = np.array([[1]])
        # 使用 np.indices 函数获取数组的索引
        idx = np.indices(data.shape)
        # 使用 ndimage.map_coordinates 函数进行坐标映射，并将输出设置为浮点数
        out = ndimage.map_coordinates(data, idx, output='f')
        # 断言输出的数据类型为单精度浮点数
        assert_(out.dtype is np.dtype('f'))
        # 断言输出与预期值几乎相等
        assert_array_almost_equal(out, [[1]])

    @pytest.mark.skipif('win32' in sys.platform or np.intp(0).itemsize < 8,
                        reason='do not run on 32 bit or windows '
                               '(no sparse memory)')
    def test_map_coordinates_large_data(self):
        # 检查大数据集合上的程序崩溃情况
        try:
            # 设置数据集大小为 30000 x 30000，并创建一个空的浮点数数组
            n = 30000
            a = np.empty(n**2, dtype=np.float32).reshape(n, n)
            # 填充可能读取的部分
            a[n - 3:, n - 3:] = 0
            # 使用 ndimage.map_coordinates 函数进行坐标映射，测试是否会引发内存错误
            ndimage.map_coordinates(a, [[n - 1.5], [n - 1.5]], order=1)
        except MemoryError as e:
            # 如果内存不足，抛出 pytest.skip 异常
            raise pytest.skip('Not enough memory available') from e

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform01(self, order):
        # 创建一个包含单个元素的 NumPy 数组
        data = np.array([1])
        # 使用 ndimage.affine_transform 函数进行仿射变换，保持顺序为 order
        out = ndimage.affine_transform(data, [[1]], order=order)
        # 断言输出与预期值几乎相等
        assert_array_almost_equal(out, [1])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform02(self, order):
        # 创建一个包含四个元素的全一数组
        data = np.ones([4])
        # 使用 ndimage.affine_transform 函数进行仿射变换，保持顺序为 order
        out = ndimage.affine_transform(data, [[1]], order=order)
        # 断言输出与预期值几乎相等
        assert_array_almost_equal(out, [1, 1, 1, 1])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform03(self, order):
        # 创建一个包含四个元素的全一数组
        data = np.ones([4])
        # 使用 ndimage.affine_transform 函数进行仿射变换，保持顺序为 order，偏移量为 -1
        out = ndimage.affine_transform(data, [[1]], -1, order=order)
        # 断言输出与预期值几乎相等
        assert_array_almost_equal(out, [0, 1, 1, 1])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform04(self, order):
        # 创建一个包含四个元素的 NumPy 数组
        data = np.array([4, 1, 3, 2])
        # 使用 ndimage.affine_transform 函数进行仿射变换，保持顺序为 order，偏移量为 -1
        out = ndimage.affine_transform(data, [[1]], -1, order=order)
        # 断言输出与预期值几乎相等
        assert_array_almost_equal(out, [0, 4, 1, 3])

    @pytest.mark.parametrize('order', range(0, 6))
    @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
    def test_affine_transform05(self, order, dtype):
        # 创建一个包含全一矩阵的 NumPy 数组，根据 dtype 设置数据类型
        data = np.array([[1, 1, 1, 1],
                         [1, 1, 1, 1],
                         [1, 1, 1, 1]], dtype=dtype)
        # 创建一个预期输出的 NumPy 数组，根据 dtype 设置数据类型
        expected = np.array([[0, 1, 1, 1],
                             [0, 1, 1, 1],
                             [0, 1, 1, 1]], dtype=dtype)
        # 如果数据类型为复数类型，进行特定的处理
        if data.dtype.kind == 'c':
            data -= 1j * data
            expected -= 1j * expected
        # 使用 ndimage.affine_transform 函数进行仿射变换，保持顺序为 order，偏移量为 [0, -1]
        out = ndimage.affine_transform(data, [[1, 0], [0, 1]],
                                       [0, -1], order=order)
        # 断言输出与预期值几乎相等
        assert_array_almost_equal(out, expected)
    @pytest.mark.parametrize('order', range(0, 6))
    # 使用 pytest 的参数化功能，对 order 参数进行范围为 0 到 5 的测试
    def test_affine_transform06(self, order):
        # 创建一个 3x4 的 NumPy 数组作为测试数据
        data = np.array([[4, 1, 3, 2],
                         [7, 6, 8, 5],
                         [3, 5, 3, 6]])
        # 调用 ndimage 库的 affine_transform 函数进行仿射变换
        # 变换矩阵为单位矩阵，偏移量为 (0, -1)，使用指定的 order 参数
        out = ndimage.affine_transform(data, [[1, 0], [0, 1]],
                                       [0, -1], order=order)
        # 断言输出数组与预期结果的近似相等
        assert_array_almost_equal(out, [[0, 4, 1, 3],
                                        [0, 7, 6, 8],
                                        [0, 3, 5, 3]])

    @pytest.mark.parametrize('order', range(0, 6))
    # 使用 pytest 的参数化功能，对 order 参数进行范围为 0 到 5 的测试
    def test_affine_transform07(self, order):
        # 创建一个 3x4 的 NumPy 数组作为测试数据
        data = np.array([[4, 1, 3, 2],
                         [7, 6, 8, 5],
                         [3, 5, 3, 6]])
        # 调用 ndimage 库的 affine_transform 函数进行仿射变换
        # 变换矩阵为单位矩阵，偏移量为 (-1, 0)，使用指定的 order 参数
        out = ndimage.affine_transform(data, [[1, 0], [0, 1]],
                                       [-1, 0], order=order)
        # 断言输出数组与预期结果的近似相等
        assert_array_almost_equal(out, [[0, 0, 0, 0],
                                        [4, 1, 3, 2],
                                        [7, 6, 8, 5]])

    @pytest.mark.parametrize('order', range(0, 6))
    # 使用 pytest 的参数化功能，对 order 参数进行范围为 0 到 5 的测试
    def test_affine_transform08(self, order):
        # 创建一个 3x4 的 NumPy 数组作为测试数据
        data = np.array([[4, 1, 3, 2],
                         [7, 6, 8, 5],
                         [3, 5, 3, 6]])
        # 调用 ndimage 库的 affine_transform 函数进行仿射变换
        # 变换矩阵为单位矩阵，偏移量为 (-1, -1)，使用指定的 order 参数
        out = ndimage.affine_transform(data, [[1, 0], [0, 1]],
                                       [-1, -1], order=order)
        # 断言输出数组与预期结果的近似相等
        assert_array_almost_equal(out, [[0, 0, 0, 0],
                                        [0, 4, 1, 3],
                                        [0, 7, 6, 8]])

    @pytest.mark.parametrize('order', range(0, 6))
    # 使用 pytest 的参数化功能，对 order 参数进行范围为 0 到 5 的测试
    def test_affine_transform09(self, order):
        # 创建一个 3x4 的 NumPy 数组作为测试数据
        data = np.array([[4, 1, 3, 2],
                         [7, 6, 8, 5],
                         [3, 5, 3, 6]])
        # 如果 order 大于 1，则对 data 应用 ndimage 的 spline_filter 函数
        if (order > 1):
            filtered = ndimage.spline_filter(data, order=order)
        else:
            filtered = data
        # 调用 ndimage 库的 affine_transform 函数进行仿射变换
        # 变换矩阵为单位矩阵，偏移量为 (-1, -1)，使用指定的 order 参数和 prefilter 参数
        out = ndimage.affine_transform(filtered, [[1, 0], [0, 1]],
                                       [-1, -1], order=order,
                                       prefilter=False)
        # 断言输出数组与预期结果的近似相等
        assert_array_almost_equal(out, [[0, 0, 0, 0],
                                        [0, 4, 1, 3],
                                        [0, 7, 6, 8]])

    @pytest.mark.parametrize('order', range(0, 6))
    # 使用 pytest 的参数化功能，对 order 参数进行范围为 0 到 5 的测试
    def test_affine_transform10(self, order):
        # 创建一个长度为 2 的全为 1.0 的 NumPy 浮点数数组作为测试数据
        data = np.ones([2], np.float64)
        # 调用 ndimage 库的 affine_transform 函数进行仿射变换
        # 变换矩阵为 [[0.5]]，输出形状为 (4,)，使用指定的 order 参数
        out = ndimage.affine_transform(data, [[0.5]], output_shape=(4,),
                                       order=order)
        # 断言输出数组与预期结果的近似相等
        assert_array_almost_equal(out, [1, 1, 1, 0])

    @pytest.mark.parametrize('order', range(0, 6))
    # 使用 pytest 的参数化功能，对 order 参数进行范围为 0 到 5 的测试
    def test_affine_transform11(self, order):
        # 创建一个长度为 8 的整数列表作为测试数据
        data = [1, 5, 2, 6, 3, 7, 4, 4]
        # 调用 ndimage 库的 affine_transform 函数进行仿射变换
        # 变换矩阵为 [[2]]，偏移量为 0，输出形状为 (4,)，使用指定的 order 参数
        out = ndimage.affine_transform(data, [[2]], 0, (4,), order=order)
        # 断言输出数组与预期结果的近似相等
        assert_array_almost_equal(out, [1, 2, 3, 4])

    @pytest.mark.parametrize('order', range(0, 6))
    # 使用 pytest 的参数化功能，对 order 参数进行范围为 0 到 5 的测试
    def test_affine_transform12(self, order):
        # 创建一个长度为 4 的整数列表作为测试数据
        data = [1, 2, 3, 4]
        # 调用 ndimage 库的 affine_transform 函数进行仿射变换
        # 变换矩阵为 [[0.5]]，偏移量为 0，输出形状为 (8,)，使用指定的 order 参数
        out = ndimage.affine_transform(data, [[0.5]], 0, (8,), order=order)
        # 断言输出数组的偶数索引元素与预期结果的近似相等
        assert_array_almost_equal(out[::2], [1, 2, 3, 4])
    # 使用 pytest 的 parametrize 装饰器为该方法生成多个参数化的测试用例
    @pytest.mark.parametrize('order', range(0, 6))
    # 定义测试方法 test_affine_transform13，接受参数 order
    def test_affine_transform13(self, order):
        # 创建一个包含浮点数和整数的二维数组作为测试数据
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9.0, 10, 11, 12]]
        # 使用 ndimage 库中的 affine_transform 函数进行仿射变换，变换矩阵为 [[1, 0], [0, 2]]，偏移为 0，输出形状为 (3, 2)，指定变换次序为 order
        out = ndimage.affine_transform(data, [[1, 0], [0, 2]], 0, (3, 2),
                                       order=order)
        # 使用 assert_array_almost_equal 检查变换后的输出是否与期望的结果近似相等
        assert_array_almost_equal(out, [[1, 3], [5, 7], [9, 11]])
    
    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform14(self, order):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        out = ndimage.affine_transform(data, [[2, 0], [0, 1]], 0, (1, 4),
                                       order=order)
        assert_array_almost_equal(out, [[1, 2, 3, 4]])
    
    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform15(self, order):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        out = ndimage.affine_transform(data, [[2, 0], [0, 2]], 0, (1, 2),
                                       order=order)
        assert_array_almost_equal(out, [[1, 3]])
    
    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform16(self, order):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        out = ndimage.affine_transform(data, [[1, 0.0], [0, 0.5]], 0,
                                       (3, 8), order=order)
        assert_array_almost_equal(out[..., ::2], data)
    
    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform17(self, order):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        out = ndimage.affine_transform(data, [[0.5, 0], [0, 1]], 0,
                                       (6, 4), order=order)
        assert_array_almost_equal(out[::2, ...], data)
    
    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform18(self, order):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        out = ndimage.affine_transform(data, [[0.5, 0], [0, 0.5]], 0,
                                       (6, 8), order=order)
        assert_array_almost_equal(out[::2, ::2], data)
    
    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform19(self, order):
        data = np.array([[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12]], np.float64)
        out = ndimage.affine_transform(data, [[0.5, 0], [0, 0.5]], 0,
                                       (6, 8), order=order)
        # 对先前变换的输出再次进行仿射变换，变换矩阵为 [[2.0, 0], [0, 2.0]]，偏移为 0，输出形状为 (3, 4)，指定变换次序为 order
        out = ndimage.affine_transform(out, [[2.0, 0], [0, 2.0]], 0,
                                       (3, 4), order=order)
        # 使用 assert_array_almost_equal 检查最终的输出是否与原始输入数据近似相等
        assert_array_almost_equal(out, data)
    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform20(self, order):
        # 定义测试函数 test_affine_transform20，参数为 order
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        # 定义测试数据 data，一个二维列表
        out = ndimage.affine_transform(data, [[0], [2]], 0, (2,),
                                       order=order)
        # 调用 ndimage 库中的 affine_transform 函数，对 data 进行仿射变换
        # 参数包括变换矩阵 [[0], [2]]，偏移值 0，输出形状 (2,)，以及 order 参数
        assert_array_almost_equal(out, [1, 3])
        # 断言输出 out 与预期结果 [1, 3] 几乎相等

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform21(self, order):
        # 定义测试函数 test_affine_transform21，参数为 order
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        # 定义测试数据 data，一个二维列表
        out = ndimage.affine_transform(data, [[2], [0]], 0, (2,),
                                       order=order)
        # 调用 ndimage 库中的 affine_transform 函数，对 data 进行仿射变换
        # 参数包括变换矩阵 [[2], [0]]，偏移值 0，输出形状 (2,)，以及 order 参数
        assert_array_almost_equal(out, [1, 9])
        # 断言输出 out 与预期结果 [1, 9] 几乎相等

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform22(self, order):
        # 定义测试函数 test_affine_transform22，参数为 order
        # shift and offset interaction; see issue #1547
        # 移动和偏移交互问题，参见问题 #1547
        data = np.array([4, 1, 3, 2])
        # 定义测试数据 data，一个 NumPy 数组
        out = ndimage.affine_transform(data, [[2]], [-1], (3,),
                                       order=order)
        # 调用 ndimage 库中的 affine_transform 函数，对 data 进行仿射变换
        # 参数包括变换矩阵 [[2]]，偏移值为 [-1]，输出形状 (3,)，以及 order 参数
        assert_array_almost_equal(out, [0, 1, 2])
        # 断言输出 out 与预期结果 [0, 1, 2] 几乎相等

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform23(self, order):
        # 定义测试函数 test_affine_transform23，参数为 order
        # shift and offset interaction; see issue #1547
        # 移动和偏移交互问题，参见问题 #1547
        data = np.array([4, 1, 3, 2])
        # 定义测试数据 data，一个 NumPy 数组
        out = ndimage.affine_transform(data, [[0.5]], [-1], (8,),
                                       order=order)
        # 调用 ndimage 库中的 affine_transform 函数，对 data 进行仿射变换
        # 参数包括变换矩阵 [[0.5]]，偏移值为 [-1]，输出形状 (8,)，以及 order 参数
        assert_array_almost_equal(out[::2], [0, 4, 1, 3])
        # 断言输出 out 的每隔一个元素与预期结果 [0, 4, 1, 3] 几乎相等

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform24(self, order):
        # 定义测试函数 test_affine_transform24，参数为 order
        # consistency between diagonal and non-diagonal case; see issue #1547
        # 对角线和非对角线情况的一致性，参见问题 #1547
        data = np.array([4, 1, 3, 2])
        # 定义测试数据 data，一个 NumPy 数组
        with suppress_warnings() as sup:
            # 使用 suppress_warnings 上下文管理器来抑制警告
            sup.filter(UserWarning,
                       'The behavior of affine_transform with a 1-D array .* '
                       'has changed')
            # 过滤特定的 UserWarning，提示信息为匹配正则表达式
            out1 = ndimage.affine_transform(data, [2], -1, order=order)
            # 第一次调用 affine_transform 函数，变换矩阵为 [2]，偏移值为 -1，以及 order 参数
        out2 = ndimage.affine_transform(data, [[2]], -1, order=order)
        # 第二次调用 affine_transform 函数，变换矩阵为 [[2]]，偏移值为 -1，以及 order 参数
        assert_array_almost_equal(out1, out2)
        # 断言两次调用的输出 out1 和 out2 几乎相等

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform25(self, order):
        # 定义测试函数 test_affine_transform25，参数为 order
        # consistency between diagonal and non-diagonal case; see issue #1547
        # 对角线和非对角线情况的一致性，参见问题 #1547
        data = np.array([4, 1, 3, 2])
        # 定义测试数据 data，一个 NumPy 数组
        with suppress_warnings() as sup:
            # 使用 suppress_warnings 上下文管理器来抑制警告
            sup.filter(UserWarning,
                       'The behavior of affine_transform with a 1-D array .* '
                       'has changed')
            # 过滤特定的 UserWarning，提示信息为匹配正则表达式
            out1 = ndimage.affine_transform(data, [0.5], -1, order=order)
            # 第一次调用 affine_transform 函数，变换矩阵为 [0.5]，偏移值为 -1，以及 order 参数
        out2 = ndimage.affine_transform(data, [[0.5]], -1, order=order)
        # 第二次调用 affine_transform 函数，变换矩阵为 [[0.5]]，偏移值为 -1，以及 order 参数
        assert_array_almost_equal(out1, out2)
        # 断言两次调用的输出 out1 和 out2 几乎相等
    def test_affine_transform26(self, order):
        # 测试仿射变换，处理均匀坐标
        data = np.array([[4, 1, 3, 2],
                         [7, 6, 8, 5],
                         [3, 5, 3, 6]])
        # 如果 order 大于 1，则使用样条插值滤波器处理数据
        if (order > 1):
            filtered = ndimage.spline_filter(data, order=order)
        else:
            filtered = data
        # 创建原始仿射变换矩阵和偏移矩阵
        tform_original = np.eye(2)
        offset_original = -np.ones((2, 1))
        # 创建包含偏移的仿射变换矩阵
        tform_h1 = np.hstack((tform_original, offset_original))
        # 创建包含额外行和列的仿射变换矩阵
        tform_h2 = np.vstack((tform_h1, [[0, 0, 1]]))
        # 对滤波后的数据进行仿射变换，使用原始仿射变换和偏移
        out1 = ndimage.affine_transform(filtered, tform_original,
                                        offset_original.ravel(),
                                        order=order, prefilter=False)
        # 对滤波后的数据进行仿射变换，使用包含偏移的仿射变换矩阵
        out2 = ndimage.affine_transform(filtered, tform_h1, order=order,
                                        prefilter=False)
        # 对滤波后的数据进行仿射变换，使用包含额外行和列的仿射变换矩阵
        out3 = ndimage.affine_transform(filtered, tform_h2, order=order,
                                        prefilter=False)
        # 断言输出结果是否与预期数组近似相等
        for out in [out1, out2, out3]:
            assert_array_almost_equal(out, [[0, 0, 0, 0],
                                            [0, 4, 1, 3],
                                            [0, 7, 6, 8]])

    def test_affine_transform27(self):
        # 测试有效的均匀变换矩阵
        data = np.array([[4, 1, 3, 2],
                         [7, 6, 8, 5],
                         [3, 5, 3, 6]])
        # 创建包含偏移的仿射变换矩阵
        tform_h1 = np.hstack((np.eye(2), -np.ones((2, 1))))
        # 创建包含额外行和列的仿射变换矩阵，这里故意设置一个无效的变换矩阵
        tform_h2 = np.vstack((tform_h1, [[5, 2, 1]]))
        # 断言调用 affine_transform 函数时是否抛出 ValueError 异常
        assert_raises(ValueError, ndimage.affine_transform, data, tform_h2)

    def test_affine_transform_1d_endianness_with_output_parameter(self):
        # 测试给定输出参数的一维仿射变换，包括不同的字节顺序
        data = np.ones((2, 2))
        for out in [np.empty_like(data),
                    np.empty_like(data).astype(data.dtype.newbyteorder()),
                    data.dtype, data.dtype.newbyteorder()]:
            with suppress_warnings() as sup:
                sup.filter(UserWarning,
                           'The behavior of affine_transform with a 1-D array '
                           '.* has changed')
                # 执行一维仿射变换，并对返回结果进行断言
                returned = ndimage.affine_transform(data, [1, 1], output=out)
            result = out if returned is None else returned
            assert_array_almost_equal(result, [[1, 1], [1, 1]])

    def test_affine_transform_multi_d_endianness_with_output_parameter(self):
        # 测试给定输出参数的多维仿射变换，包括不同的字节顺序
        data = np.array([1])
        for out in [data.dtype, data.dtype.newbyteorder(),
                    np.empty_like(data),
                    np.empty_like(data).astype(data.dtype.newbyteorder())]:
            # 执行多维仿射变换，并对返回结果进行断言
            returned = ndimage.affine_transform(data, [[1]], output=out)
            result = out if returned is None else returned
            assert_array_almost_equal(result, [1])
    # 定义测试函数，测试仿射变换的输出形状是否正确
    def test_affine_transform_output_shape(self):
        # 创建一个包含8个元素的浮点数数组
        data = np.arange(8, dtype=np.float64)
        # 创建一个形状为(16,)的全1数组作为输出
        out = np.ones((16,))

        # 执行仿射变换，将data应用于out，不需要指定output_shape
        ndimage.affine_transform(data, [[1]], output=out)
        # 断言输出的前8个元素与原始data数组相等
        assert_array_almost_equal(out[:8], data)

        # 如果输出形状不匹配，应该抛出RuntimeError错误
        with pytest.raises(RuntimeError):
            # 执行仿射变换，指定输出为out，并且指定输出形状为(12,)
            ndimage.affine_transform(
                data, [[1]], output=out, output_shape=(12,))

    # 定义测试函数，测试仿射变换的字符串类型输出
    def test_affine_transform_with_string_output(self):
        # 创建一个包含单个元素1的数组
        data = np.array([1])
        # 执行仿射变换，将data应用于'f'类型的输出
        out = ndimage.affine_transform(data, [[1]], output='f')
        # 断言输出的数据类型为float32
        assert_(out.dtype is np.dtype('f'))
        # 断言输出与原始data数组几乎相等
        assert_array_almost_equal(out, [1])

    # 使用pytest参数化标记定义测试函数，测试通过grid-wrap模式实现的仿射变换位移
    @pytest.mark.parametrize('shift',
                             [(1, 0), (0, 1), (-1, 1), (3, -5), (2, 7)])
    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform_shift_via_grid_wrap(self, shift, order):
        # 创建一个2x2的数组x
        x = np.array([[0, 1],
                      [2, 3]])
        # 创建一个2x3的零矩阵affine，并将单位矩阵复制到其前两行前两列
        affine = np.zeros((2, 3))
        affine[:2, :2] = np.eye(2)
        # 将shift的值赋给affine的第三列
        affine[:, 2] = shift
        # 断言通过grid-wrap模式实现的仿射变换后的结果与np.roll函数应用于x数组后的结果几乎相等
        assert_array_almost_equal(
            ndimage.affine_transform(x, affine, mode='grid-wrap', order=order),
            np.roll(x, shift, axis=(0, 1)),
        )

    # 使用pytest参数化标记定义测试函数，测试通过reflect模式实现的仿射变换位移
    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform_shift_reflect(self, order):
        # 创建一个2x3的数组x
        x = np.array([[0, 1, 2],
                      [3, 4, 5]])
        # 创建一个2x3的零矩阵affine，并将单位矩阵复制到其前两行前两列
        affine = np.zeros((2, 3))
        affine[:2, :2] = np.eye(2)
        # 将x的形状作为位移值赋给affine的第三列
        affine[:, 2] = x.shape
        # 断言通过reflect模式实现的仿射变换后的结果与x数组的水平翻转和垂直翻转结果几乎相等
        assert_array_almost_equal(
            ndimage.affine_transform(x, affine, mode='reflect', order=order),
            x[::-1, ::-1],
        )

    # 使用pytest参数化标记定义测试函数，测试通过shift函数实现的位移
    @pytest.mark.parametrize('order', range(0, 6))
    def test_shift01(self, order):
        # 创建一个包含单个元素1的数组
        data = np.array([1])
        # 执行位移操作，将data数组向右平移1个单位
        out = ndimage.shift(data, [1], order=order)
        # 断言位移后的结果与期望的结果几乎相等
        assert_array_almost_equal(out, [0])

    # 使用pytest参数化标记定义测试函数，测试通过shift函数实现的位移
    @pytest.mark.parametrize('order', range(0, 6))
    def test_shift02(self, order):
        # 创建一个包含四个元素的全1数组
        data = np.ones([4])
        # 执行位移操作，将data数组向右平移1个单位
        out = ndimage.shift(data, [1], order=order)
        # 断言位移后的结果与期望的结果几乎相等
        assert_array_almost_equal(out, [0, 1, 1, 1])

    # 使用pytest参数化标记定义测试函数，测试通过shift函数实现的位移
    @pytest.mark.parametrize('order', range(0, 6))
    def test_shift03(self, order):
        # 创建一个包含四个元素的全1数组
        data = np.ones([4])
        # 执行位移操作，将data数组向左平移1个单位
        out = ndimage.shift(data, -1, order=order)
        # 断言位移后的结果与期望的结果几乎相等
        assert_array_almost_equal(out, [1, 1, 1, 0])

    # 使用pytest参数化标记定义测试函数，测试通过shift函数实现的位移
    @pytest.mark.parametrize('order', range(0, 6))
    def test_shift04(self, order):
        # 创建一个包含四个元素的数组
        data = np.array([4, 1, 3, 2])
        # 执行位移操作，将data数组向右平移1个单位
        out = ndimage.shift(data, 1, order=order)
        # 断言位移后的结果与期望的结果几乎相等
        assert_array_almost_equal(out, [0, 4, 1, 3])

    # 使用pytest参数化标记定义测试函数，测试通过shift函数实现的位移，并指定数据类型
    @pytest.mark.parametrize('order', range(0, 6))
    @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
    # 定义一个测试函数，测试在给定顺序和数据类型下的数组平移操作
    def test_shift05(self, order, dtype):
        # 创建一个二维数组作为测试数据，数据类型由参数指定
        data = np.array([[1, 1, 1, 1],
                         [1, 1, 1, 1],
                         [1, 1, 1, 1]], dtype=dtype)
        # 创建一个预期结果的数组，数据类型与测试数据相同
        expected = np.array([[0, 1, 1, 1],
                             [0, 1, 1, 1],
                             [0, 1, 1, 1]], dtype=dtype)
        # 如果数据类型是复数类型，则进行特定操作
        if data.dtype.kind == 'c':
            data -= 1j * data
            expected -= 1j * expected
        # 对测试数据进行平移操作，沿着第二个维度平移一个单位
        out = ndimage.shift(data, [0, 1], order=order)
        # 断言平移后的结果与预期结果近似相等
        assert_array_almost_equal(out, expected)

    # 标记测试参数化，测试多种参数组合下的平移操作
    @pytest.mark.parametrize('order', range(0, 6))
    @pytest.mark.parametrize('mode', ['constant', 'grid-constant'])
    @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
    def test_shift_with_nonzero_cval(self, order, mode, dtype):
        # 创建一个二维数组作为测试数据，数据类型由参数指定
        data = np.array([[1, 1, 1, 1],
                         [1, 1, 1, 1],
                         [1, 1, 1, 1]], dtype=dtype)

        # 创建一个预期结果的数组，数据类型与测试数据相同
        expected = np.array([[0, 1, 1, 1],
                             [0, 1, 1, 1],
                             [0, 1, 1, 1]], dtype=dtype)

        # 如果数据类型是复数类型，则进行特定操作
        if data.dtype.kind == 'c':
            data -= 1j * data
            expected -= 1j * expected
        
        # 设置一个常数值用于填充平移后结果的指定列
        cval = 5.0
        expected[:, 0] = cval  # 特定于下面使用的 [0, 1] 平移操作
        # 对测试数据进行平移操作，指定平移顺序、模式和填充值
        out = ndimage.shift(data, [0, 1], order=order, mode=mode, cval=cval)
        # 断言平移后的结果与预期结果近似相等
        assert_array_almost_equal(out, expected)

    # 标记测试参数化，测试在给定顺序下的数组平移操作
    @pytest.mark.parametrize('order', range(0, 6))
    def test_shift06(self, order):
        # 创建一个二维数组作为测试数据
        data = np.array([[4, 1, 3, 2],
                         [7, 6, 8, 5],
                         [3, 5, 3, 6]])
        # 对测试数据进行平移操作，沿着第二个维度平移一个单位
        out = ndimage.shift(data, [0, 1], order=order)
        # 断言平移后的结果与预期结果近似相等
        assert_array_almost_equal(out, [[0, 4, 1, 3],
                                        [0, 7, 6, 8],
                                        [0, 3, 5, 3]])

    # 标记测试参数化，测试在给定顺序下的数组平移操作
    @pytest.mark.parametrize('order', range(0, 6))
    def test_shift07(self, order):
        # 创建一个二维数组作为测试数据
        data = np.array([[4, 1, 3, 2],
                         [7, 6, 8, 5],
                         [3, 5, 3, 6]])
        # 对测试数据进行平移操作，沿着第一个维度平移一个单位
        out = ndimage.shift(data, [1, 0], order=order)
        # 断言平移后的结果与预期结果近似相等
        assert_array_almost_equal(out, [[0, 0, 0, 0],
                                        [4, 1, 3, 2],
                                        [7, 6, 8, 5]])

    # 标记测试参数化，测试在给定顺序下的数组平移操作
    @pytest.mark.parametrize('order', range(0, 6))
    def test_shift08(self, order):
        # 创建一个二维数组作为测试数据
        data = np.array([[4, 1, 3, 2],
                         [7, 6, 8, 5],
                         [3, 5, 3, 6]])
        # 对测试数据进行平移操作，沿着两个维度均平移一个单位
        out = ndimage.shift(data, [1, 1], order=order)
        # 断言平移后的结果与预期结果近似相等
        assert_array_almost_equal(out, [[0, 0, 0, 0],
                                        [0, 4, 1, 3],
                                        [0, 7, 6, 8]])
    # 定义一个测试函数，测试 ndimage.shift 方法在给定不同偏移和阶数下的行为
    def test_shift09(self, order):
        # 创建一个包含整数的 NumPy 数组
        data = np.array([[4, 1, 3, 2],
                         [7, 6, 8, 5],
                         [3, 5, 3, 6]])
        # 根据阶数 order 的值选择进行样条滤波或保持原数据
        if (order > 1):
            filtered = ndimage.spline_filter(data, order=order)
        else:
            filtered = data
        # 对 filtered 进行平移操作，偏移量为 [1, 1]，使用给定的阶数 order 进行插值，不进行预滤波
        out = ndimage.shift(filtered, [1, 1], order=order, prefilter=False)
        # 断言 out 数组的内容与预期的值相近
        assert_array_almost_equal(out, [[0, 0, 0, 0],
                                        [0, 4, 1, 3],
                                        [0, 7, 6, 8]])

    @pytest.mark.parametrize('shift',
                             [(1, 0), (0, 1), (-1, 1), (3, -5), (2, 7)])
    @pytest.mark.parametrize('order', range(0, 6))
    # 测试 'grid-wrap' 模式下的 ndimage.shift 方法
    def test_shift_grid_wrap(self, shift, order):
        # 创建一个小型的二维 NumPy 数组 x
        x = np.array([[0, 1],
                      [2, 3]])
        # 断言使用 'grid-wrap' 模式的平移结果与 np.roll 函数的结果相近
        assert_array_almost_equal(
            ndimage.shift(x, shift, mode='grid-wrap', order=order),
            np.roll(x, shift, axis=(0, 1)),
        )

    @pytest.mark.parametrize('shift',
                             [(1, 0), (0, 1), (-1, 1), (3, -5), (2, 7)])
    @pytest.mark.parametrize('order', range(0, 6))
    # 测试 'grid-constant' 模式与 'constant' 模式下整数偏移的等效性
    def test_shift_grid_constant1(self, shift, order):
        # 创建一个二维数组 x
        x = np.arange(20).reshape((5, 4))
        # 断言在 'grid-constant' 模式和 'constant' 模式下整数偏移的结果近似相等
        assert_array_almost_equal(
            ndimage.shift(x, shift, mode='grid-constant', order=order),
            ndimage.shift(x, shift, mode='constant', order=order),
        )

    # 测试 'grid-constant' 模式下的 ndimage.shift 方法，使用阶数为 1 的结果进行验证
    def test_shift_grid_constant_order1(self):
        # 创建一个包含浮点数的二维数组 x 和期望的结果数组 expected_result
        x = np.array([[1, 2, 3],
                      [4, 5, 6]], dtype=float)
        expected_result = np.array([[0.25, 0.75, 1.25],
                                    [1.25, 3.00, 4.00]])
        # 断言使用 'grid-constant' 模式和阶数为 1 的结果与预期的值相近
        assert_array_almost_equal(
            ndimage.shift(x, (0.5, 0.5), mode='grid-constant', order=1),
            expected_result,
        )

    @pytest.mark.parametrize('order', range(0, 6))
    # 测试 'reflect' 模式下的 ndimage.shift 方法
    def test_shift_reflect(self, order):
        # 创建一个二维数组 x
        x = np.array([[0, 1, 2],
                      [3, 4, 5]])
        # 断言使用 'reflect' 模式和给定阶数的结果与 x 矩阵的反转相近
        assert_array_almost_equal(
            ndimage.shift(x, x.shape, mode='reflect', order=order),
            x[::-1, ::-1],
        )

    @pytest.mark.parametrize('order', range(0, 6))
    @pytest.mark.parametrize('prefilter', [False, True])
    # 测试 'nearest' 边界条件下的 ndimage.shift 方法
    def test_shift_nearest_boundary(self, order, prefilter):
        # 创建一个长度为 16 的一维数组 x
        x = np.arange(16)
        kwargs = dict(mode='nearest', order=order, prefilter=prefilter)
        # 断言向数组末尾偏移至少 order // 2 的位置时，得到的值等于边界值 x[0]
        assert_array_almost_equal(
            ndimage.shift(x, order // 2 + 1, **kwargs)[0], x[0],
        )
        # 断言向数组开头偏移至少 order // 2 的位置时，得到的值等于边界值 x[-1]
        assert_array_almost_equal(
            ndimage.shift(x, -order // 2 - 1, **kwargs)[-1], x[-1],
        )
    @pytest.mark.parametrize('mode', ['grid-constant', 'grid-wrap', 'nearest',
                                      'mirror', 'reflect'])
    @pytest.mark.parametrize('order', range(6))
    def test_shift_vs_padded(self, order, mode):
        # 创建一个 12x12 的浮点数数组
        x = np.arange(144, dtype=float).reshape(12, 12)
        # 定义平移量
        shift = (0.4, -2.3)

        # 手动填充数组以获取期望结果的中心
        npad = 32  # 填充数量
        pad_mode = ndimage_to_numpy_mode.get(mode)  # 根据给定模式获取填充方式
        xp = np.pad(x, npad, mode=pad_mode)  # 对数组 x 进行填充
        center_slice = tuple([slice(npad, -npad)] * x.ndim)  # 提取中心部分的切片
        expected_result = ndimage.shift(
            xp, shift, mode=mode, order=order)[center_slice]  # 使用 ndimage.shift 进行平移并提取中心部分的预期结果

        # 断言实际平移结果与期望结果相近
        assert_allclose(
            ndimage.shift(x, shift, mode=mode, order=order),
            expected_result,
            rtol=1e-7,
        )

    @pytest.mark.parametrize('order', range(0, 6))
    def test_zoom1(self, order):
        for z in [2, [2, 2]]:
            # 创建一个 5x5 的浮点数数组，然后按照给定的缩放因子进行缩放
            arr = np.array(list(range(25))).reshape((5, 5)).astype(float)
            arr = ndimage.zoom(arr, z, order=order)
            # 断言缩放后数组的形状为 (10, 10)
            assert_equal(arr.shape, (10, 10))
            # 断言最后一行的所有元素不为 0
            assert_(np.all(arr[-1, :] != 0))
            # 断言最后一行的所有元素大于等于 (20 - eps)
            assert_(np.all(arr[-1, :] >= (20 - eps)))
            # 断言第一行的所有元素小于等于 (5 + eps)
            assert_(np.all(arr[0, :] <= (5 + eps)))
            # 断言数组的所有元素大于等于 (0 - eps)
            assert_(np.all(arr >= (0 - eps)))
            # 断言数组的所有元素小于等于 (24 + eps)
            assert_(np.all(arr <= (24 + eps)))

    def test_zoom2(self):
        arr = np.arange(12).reshape((3, 4))
        # 对数组进行两次连续的缩放操作，然后再次缩小回原始大小
        out = ndimage.zoom(ndimage.zoom(arr, 2), 0.5)
        # 断言最终结果与原始数组相等
        assert_array_equal(out, arr)

    def test_zoom3(self):
        arr = np.array([[1, 2]])
        # 对数组分别按照 (2, 1) 和 (1, 2) 的缩放因子进行缩放
        out1 = ndimage.zoom(arr, (2, 1))
        out2 = ndimage.zoom(arr, (1, 2))

        # 断言缩放后的结果与预期结果相近
        assert_array_almost_equal(out1, np.array([[1, 2], [1, 2]]))
        assert_array_almost_equal(out2, np.array([[1, 1, 2, 2]]))

    @pytest.mark.parametrize('order', range(0, 6))
    @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
    def test_zoom_affine01(self, order, dtype):
        # 创建一个二维数组，根据数据类型进行处理
        data = np.asarray([[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12]], dtype=dtype)
        if data.dtype.kind == 'c':
            data -= 1j * data
        with suppress_warnings() as sup:
            sup.filter(UserWarning,
                       'The behavior of affine_transform with a 1-D array .* '
                       'has changed')
            # 使用仿射变换对数组进行变换
            out = ndimage.affine_transform(data, [0.5, 0.5], 0,
                                           (6, 8), order=order)
        # 断言每隔两行两列取出的结果与原始数组相等
        assert_array_almost_equal(out[::2, ::2], data)

    def test_zoom_infinity(self):
        # Ticket #1419 的回归测试，对零数组进行缩放操作
        dim = 8
        ndimage.zoom(np.zeros((dim, dim)), 1. / dim, mode='nearest')

    def test_zoom_zoomfactor_one(self):
        # Ticket #1122 的回归测试，对一维 5x5 零数组进行缩放操作
        arr = np.zeros((1, 5, 5))
        zoom = (1.0, 2.0, 2.0)

        # 对数组进行缩放，未填充部分使用 cval=7 填充
        out = ndimage.zoom(arr, zoom, cval=7)
        ref = np.zeros((1, 10, 10))
        # 断言缩放后的结果与预期结果相等
        assert_array_almost_equal(out, ref)
    # 定义测试函数，验证 ndimage.zoom 的输出形状是否正确舍入
    def test_zoom_output_shape_roundoff(self):
        # 创建一个形状为 (3, 11, 25) 的全零数组
        arr = np.zeros((3, 11, 25))
        # 设置缩放比例为 (4/3, 15/11, 29/25)
        zoom = (4.0 / 3, 15.0 / 11, 29.0 / 25)
        # 使用 ndimage.zoom 进行缩放操作
        out = ndimage.zoom(arr, zoom)
        # 断言输出数组的形状是否为 (4, 15, 29)
        assert_array_equal(out.shape, (4, 15, 29))

    @pytest.mark.parametrize('zoom', [(1, 1), (3, 5), (8, 2), (8, 8)])
    @pytest.mark.parametrize('mode', ['nearest', 'constant', 'wrap', 'reflect',
                                      'mirror', 'grid-wrap', 'grid-mirror',
                                      'grid-constant'])
    def test_zoom_by_int_order0(self, zoom, mode):
        # 对于 order 0 的缩放，应当等同于通过 np.kron 复制 x
        # 注意：当 x.shape = (2, 2) 时，这并不适用于所有通用的 x 形状，
        #       但对于所有模式来说，因为大小比例恰好总是整数，所以在这里适用。
        x = np.array([[0, 1],
                      [2, 3]], dtype=float)
        # 断言 ndimage.zoom 的结果与 np.kron(x, np.ones(zoom)) 几乎相等
        assert_array_almost_equal(
            ndimage.zoom(x, zoom, order=0, mode=mode),
            np.kron(x, np.ones(zoom))
        )

    @pytest.mark.parametrize('shape', [(2, 3), (4, 4)])
    @pytest.mark.parametrize('zoom', [(1, 1), (3, 5), (8, 2), (8, 8)])
    @pytest.mark.parametrize('mode', ['nearest', 'reflect', 'mirror',
                                      'grid-wrap', 'grid-constant'])
    def test_zoom_grid_by_int_order0(self, shape, zoom, mode):
        # 当 grid_mode 为 True 时，order 0 的缩放应当等同于通过 np.kron 复制 x。
        # 唯一的例外是非 grid 模式 'constant' 和 'wrap'。
        x = np.arange(np.prod(shape), dtype=float).reshape(shape)
        # 断言 ndimage.zoom 的结果与 np.kron(x, np.ones(zoom)) 几乎相等
        assert_array_almost_equal(
            ndimage.zoom(x, zoom, order=0, mode=mode, grid_mode=True),
            np.kron(x, np.ones(zoom))
        )

    @pytest.mark.parametrize('mode', ['constant', 'wrap'])
    def test_zoom_grid_mode_warnings(self, mode):
        # 当 grid_mode 为 True 时，使用非 grid 模式时会发出警告。
        x = np.arange(9, dtype=float).reshape((3, 3))
        # 使用 pytest.warns 检查是否会发出 UserWarning 警告信息
        with pytest.warns(UserWarning,
                          match="It is recommended to use mode"):
            ndimage.zoom(x, 2, mode=mode, grid_mode=True),

    @pytest.mark.parametrize('order', range(0, 6))
    def test_rotate01(self, order):
        # 创建一个形状为 (3, 4) 的浮点类型数组
        data = np.array([[0, 0, 0, 0],
                         [0, 1, 1, 0],
                         [0, 0, 0, 0]], dtype=np.float64)
        # 使用 ndimage.rotate 进行旋转，角度为 0 度，顺序为 order
        out = ndimage.rotate(data, 0, order=order)
        # 断言输出的结果与原始数据几乎相等
        assert_array_almost_equal(out, data)

    @pytest.mark.parametrize('order', range(0, 6))
    # 定义测试函数 test_rotate02，用于测试图像旋转功能，接受一个参数 order 用于指定旋转顺序
    def test_rotate02(self, order):
        # 创建一个 3x4 的二维数组，数据类型为 np.float64
        data = np.array([[0, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 0]], dtype=np.float64)
        # 创建一个预期结果的 4x3 的二维数组，数据类型为 np.float64
        expected = np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]], dtype=np.float64)
        # 调用 ndimage.rotate 进行图像旋转，角度为 90 度，使用指定的旋转顺序 order
        out = ndimage.rotate(data, 90, order=order)
        # 断言计算出的结果 out 与预期结果 expected 几乎相等
        assert_array_almost_equal(out, expected)

    # 标记该测试函数为参数化测试，参数为 order 取值范围为 0 到 5
    @pytest.mark.parametrize('order', range(0, 6))
    # 标记该测试函数为参数化测试，参数为 dtype 取值为 np.float64 和 np.complex128
    @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
    # 定义测试函数 test_rotate03，用于测试图像旋转功能，接受两个参数 order 和 dtype
    def test_rotate03(self, order, dtype):
        # 创建一个 3x5 的二维数组，数据类型为参数 dtype
        data = np.array([[0, 0, 0, 0, 0],
                         [0, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0]], dtype=dtype)
        # 创建一个预期结果的 5x3 的二维数组，数据类型为参数 dtype
        expected = np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 1, 0],
                            [0, 1, 0],
                            [0, 0, 0]], dtype=dtype)
        # 如果数据类型是复数，对数据进行特定处理
        if data.dtype.kind == 'c':
            data -= 1j * data
            expected -= 1j * expected
        # 调用 ndimage.rotate 进行图像旋转，角度为 90 度，使用指定的旋转顺序 order
        out = ndimage.rotate(data, 90, order=order)
        # 断言计算出的结果 out 与预期结果 expected 几乎相等
        assert_array_almost_equal(out, expected)

    # 标记该测试函数为参数化测试，参数为 order 取值范围为 0 到 5
    @pytest.mark.parametrize('order', range(0, 6))
    # 定义测试函数 test_rotate04，用于测试图像旋转功能，接受一个参数 order
    def test_rotate04(self, order):
        # 创建一个 3x5 的二维数组，数据类型为 np.float64
        data = np.array([[0, 0, 0, 0, 0],
                         [0, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0]], dtype=np.float64)
        # 创建一个预期结果的 3x5 的二维数组，数据类型为 np.float64
        expected = np.array([[0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 1, 0, 0]], dtype=np.float64)
        # 调用 ndimage.rotate 进行图像旋转，角度为 90 度，不改变形状，使用指定的旋转顺序 order
        out = ndimage.rotate(data, 90, reshape=False, order=order)
        # 断言计算出的结果 out 与预期结果 expected 几乎相等
        assert_array_almost_equal(out, expected)

    # 标记该测试函数为参数化测试，参数为 order 取值范围为 0 到 5
    @pytest.mark.parametrize('order', range(0, 6))
    # 定义测试函数 test_rotate05，用于测试图像旋转功能，接受一个参数 order
    def test_rotate05(self, order):
        # 创建一个 4x3x3 的三维数组
        data = np.empty((4, 3, 3))
        # 循环填充三维数组的每个通道
        for i in range(3):
            data[:, :, i] = np.array([[0, 0, 0],
                                      [0, 1, 0],
                                      [0, 1, 0],
                                      [0, 0, 0]], dtype=np.float64)
        # 创建一个预期结果的 3x4 的二维数组，数据类型为 np.float64
        expected = np.array([[0, 0, 0, 0],
                             [0, 1, 1, 0],
                             [0, 0, 0, 0]], dtype=np.float64)
        # 调用 ndimage.rotate 进行图像旋转，角度为 90 度，使用指定的旋转顺序 order
        out = ndimage.rotate(data, 90, order=order)
        # 遍历每个通道，断言计算出的结果与预期结果 expected 几乎相等
        for i in range(3):
            assert_array_almost_equal(out[:, :, i], expected)

    # 标记该测试函数为参数化测试，参数为 order 取值范围为 0 到 5
    @pytest.mark.parametrize('order', range(0, 6))
    # 定义测试函数 test_rotate06，用于测试图像旋转功能，接受一个参数 order
    def test_rotate06(self, order):
        # 创建一个 3x4x3 的三维数组
        data = np.empty((3, 4, 3))
        # 循环填充三维数组的每个通道
        for i in range(3):
            data[:, :, i] = np.array([[0, 0, 0, 0],
                                      [0, 1, 1, 0],
                                      [0, 0, 0, 0]], dtype=np.float64)
        # 创建一个预期结果的 4x3 的二维数组，数据类型为 np.float64
        expected = np.array([[0, 0, 0],
                             [0, 1, 0],
                             [0, 1, 0],
                             [0, 0, 0]], dtype=np.float64)
        # 调用 ndimage.rotate 进行图像旋转，角度为 90 度，使用指定的旋转顺序 order
        out = ndimage.rotate(data, 90, order=order)
        # 遍历每个通道，断言计算出的结果与预期结果 expected 几乎相等
        for i in range(3):
            assert_array_almost_equal(out[:, :, i], expected)
    # 使用 pytest 的参数化功能，遍历 order 参数从 0 到 5
    @pytest.mark.parametrize('order', range(0, 6))
    # 定义测试函数 test_rotate07，参数包括 order
    def test_rotate07(self, order):
        # 创建一个 3x3x2 的三维 numpy 数组，数据类型为 np.float64
        data = np.array([[[0, 0, 0, 0, 0],
                          [0, 1, 1, 0, 0],
                          [0, 0, 0, 0, 0]]] * 2, dtype=np.float64)
        # 对数据进行转置操作
        data = data.transpose()
        # 创建期望输出的 3x5x2 三维 numpy 数组，数据类型为 np.float64
        expected = np.array([[[0, 0, 0],
                              [0, 1, 0],
                              [0, 1, 0],
                              [0, 0, 0],
                              [0, 0, 0]]] * 2, dtype=np.float64)
        # 对期望输出数据进行转置操作
        expected = expected.transpose([2, 1, 0])
        # 使用 ndimage.rotate 函数进行数据旋转，角度为 90 度，沿着轴 (0, 1)，使用给定的 order 参数
        out = ndimage.rotate(data, 90, axes=(0, 1), order=order)
        # 断言输出数据与期望数据几乎相等
        assert_array_almost_equal(out, expected)

    # 使用 pytest 的参数化功能，遍历 order 参数从 0 到 5
    @pytest.mark.parametrize('order', range(0, 6))
    # 定义测试函数 test_rotate08，参数包括 order
    def test_rotate08(self, order):
        # 创建一个 3x3x2 的三维 numpy 数组，数据类型为 np.float64
        data = np.array([[[0, 0, 0, 0, 0],
                          [0, 1, 1, 0, 0],
                          [0, 0, 0, 0, 0]]] * 2, dtype=np.float64)
        # 对数据进行转置操作
        data = data.transpose()
        # 创建期望输出的 3x3x2 三维 numpy 数组，数据类型为 np.float64
        expected = np.array([[[0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0]]] * 2, dtype=np.float64)
        # 对期望输出数据进行转置操作
        expected = expected.transpose()
        # 使用 ndimage.rotate 函数进行数据旋转，角度为 90 度，沿着轴 (0, 1)，不重塑形状，使用给定的 order 参数
        out = ndimage.rotate(data, 90, axes=(0, 1), reshape=False, order=order)
        # 断言输出数据与期望数据几乎相等
        assert_array_almost_equal(out, expected)

    # 定义测试函数 test_rotate09
    def test_rotate09(self):
        # 创建一个 3x5 的二维 numpy 数组，数据类型为 np.float64
        data = np.array([[0, 0, 0, 0, 0],
                         [0, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0]] * 2, dtype=np.float64)
        # 使用 assert_raises 断言，测试在旋转时沿着维度超出数据维度的情况会引发 ValueError
        with assert_raises(ValueError):
            ndimage.rotate(data, 90, axes=(0, data.ndim))

    # 定义测试函数 test_rotate10
    def test_rotate10(self):
        # 创建一个形状为 (3, 5, 3) 的三维 numpy 数组，数据类型为 np.float64，包含 45 个元素
        data = np.arange(45, dtype=np.float64).reshape((3, 5, 3))

        # 创建期望输出的三维 numpy 数组，数据类型为 np.float64，包含特定的数值
        expected = np.array([[[0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0],
                              [6.54914793, 7.54914793, 8.54914793],
                              [10.84520162, 11.84520162, 12.84520162],
                              [0.0, 0.0, 0.0]],
                             [[6.19286575, 7.19286575, 8.19286575],
                              [13.4730712, 14.4730712, 15.4730712],
                              [21.0, 22.0, 23.0],
                              [28.5269288, 29.5269288, 30.5269288],
                              [35.80713425, 36.80713425, 37.80713425]],
                             [[0.0, 0.0, 0.0],
                              [31.15479838, 32.15479838, 33.15479838],
                              [35.45085207, 36.45085207, 37.45085207],
                              [0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0]]])

        # 使用 ndimage.rotate 函数进行数据旋转，角度为 12 度，不重塑形状
        out = ndimage.rotate(data, angle=12, reshape=False)
        # 断言输出数据与期望数据几乎相等
        assert_array_almost_equal(out, expected)

    # 定义测试函数 test_rotate_exact_180
    def test_rotate_exact_180(self):
        # 创建一个二维 numpy 数组，数据为重复排列的 arange(5) 行向量
        a = np.tile(np.arange(5), (5, 1))
        # 使用 ndimage.rotate 函数两次对数组进行 180 度旋转，并检查旋转后的数组与原始数组是否相等
        b = ndimage.rotate(ndimage.rotate(a, 180), -180)
        assert_equal(a, b)
# 定义一个测试函数，用于验证 ndimage.zoom 函数的输出形状是否符合预期
def test_zoom_output_shape():
    """Ticket #643"""
    # 创建一个 3x4 的 NumPy 数组 x，其中包含 0 到 11 的整数
    x = np.arange(12).reshape((3, 4))
    # 使用 ndimage.zoom 函数对数组 x 进行放大为原来的两倍，将结果输出到一个 6x8 的零数组中
    ndimage.zoom(x, 2, output=np.zeros((6, 8)))
```