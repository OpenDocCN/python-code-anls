# `D:\src\scipysrc\scipy\scipy\ndimage\tests\test_c_api.py`

```
# 导入 NumPy 库并重命名为 np
import numpy as np
# 导入 numpy.testing 模块中的 assert_allclose 函数，用于比较两个数组是否接近
from numpy.testing import assert_allclose

# 导入 scipy 库中的 ndimage 模块，用于图像处理
from scipy import ndimage
# 导入 scipy.ndimage 模块中的 _ctest 和 _cytest，这是 C 扩展模块
from scipy.ndimage import _ctest
from scipy.ndimage import _cytest
# 导入 scipy._lib._ccallback 中的 LowLevelCallable 类，用于封装低级回调函数
from scipy._lib._ccallback import LowLevelCallable

# 定义 FILTER1D_FUNCTIONS 列表，包含四个 lambda 函数，每个函数生成不同的 filter1d 函数实例
FILTER1D_FUNCTIONS = [
    lambda filter_size: _ctest.filter1d(filter_size),
    lambda filter_size: _cytest.filter1d(filter_size, with_signature=False),
    lambda filter_size: LowLevelCallable(
                            _cytest.filter1d(filter_size, with_signature=True)
                        ),
    lambda filter_size: LowLevelCallable.from_cython(
                            _cytest, "_filter1d",
                            _cytest.filter1d_capsule(filter_size),
                        ),
]

# 定义 FILTER2D_FUNCTIONS 列表，包含四个 lambda 函数，每个函数生成不同的 filter2d 函数实例
FILTER2D_FUNCTIONS = [
    lambda weights: _ctest.filter2d(weights),
    lambda weights: _cytest.filter2d(weights, with_signature=False),
    lambda weights: LowLevelCallable(_cytest.filter2d(weights, with_signature=True)),
    lambda weights: LowLevelCallable.from_cython(_cytest,
                                                 "_filter2d",
                                                 _cytest.filter2d_capsule(weights),),
]

# 定义 TRANSFORM_FUNCTIONS 列表，包含四个 lambda 函数，每个函数生成不同的 transform 函数实例
TRANSFORM_FUNCTIONS = [
    lambda shift: _ctest.transform(shift),
    lambda shift: _cytest.transform(shift, with_signature=False),
    lambda shift: LowLevelCallable(_cytest.transform(shift, with_signature=True)),
    lambda shift: LowLevelCallable.from_cython(_cytest,
                                               "_transform",
                                               _cytest.transform_capsule(shift),),
]

# 定义 test_generic_filter 函数，用于测试通用滤波器
def test_generic_filter():
    # 定义 filter2d 函数，用于计算二维滤波
    def filter2d(footprint_elements, weights):
        return (weights * footprint_elements).sum()

    # 定义 check 函数，用于检查 FILTER2D_FUNCTIONS 中的函数
    def check(j):
        func = FILTER2D_FUNCTIONS[j]

        # 创建一个 20x20 的全一数组
        im = np.ones((20, 20))
        # 将左上角的 10x10 区域设为零
        im[:10, :10] = 0
        # 定义一个二维数组作为足迹
        footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        # 计算足迹中非零元素的个数
        footprint_size = np.count_nonzero(footprint)
        # 创建一个权重数组，元素为足迹的均值
        weights = np.ones(footprint_size) / footprint_size

        # 使用 ndimage.generic_filter 对图像进行通用滤波处理
        res = ndimage.generic_filter(im, func(weights), footprint=footprint)
        # 调用 ndimage.generic_filter 用于标准化处理
        std = ndimage.generic_filter(im, filter2d, footprint=footprint, extra_arguments=(weights,))
        # 使用 assert_allclose 函数比较两者的结果，如果不接近则抛出错误信息
        assert_allclose(res, std, err_msg=f"#{j} failed")

    # 遍历 FILTER2D_FUNCTIONS 列表，并对每个函数调用 check 函数
    for j, func in enumerate(FILTER2D_FUNCTIONS):
        check(j)


# 定义 test_generic_filter1d 函数，用于测试一维通用滤波器
def test_generic_filter1d():
    # 定义 filter1d 函数，实现一维滤波器
    def filter1d(input_line, output_line, filter_size):
        for i in range(output_line.size):
            output_line[i] = 0
            for j in range(filter_size):
                output_line[i] += input_line[i + j]
        output_line /= filter_size
    # 定义一个名为 check 的函数，接受一个参数 j
    def check(j):
        # 从全局变量 FILTER1D_FUNCTIONS 中获取指定索引 j 处的函数对象
        func = FILTER1D_FUNCTIONS[j]

        # 创建一个 10x20 的二维数组 im，左半部分是10个零，右半部分是10个一，然后将其沿着水平方向堆叠为一行
        im = np.tile(np.hstack((np.zeros(10), np.ones(10))), (10, 1))
        
        # 定义滤波器的大小为 3
        filter_size = 3

        # 使用 ndimage 模块中的 generic_filter1d 函数对输入图像 im 进行一维通用滤波
        # 使用 FILTER1D_FUNCTIONS[j] 所对应的函数作为滤波器函数，filter_size 作为滤波器大小
        res = ndimage.generic_filter1d(im, func(filter_size), filter_size)
        
        # 再次使用 generic_filter1d 函数对输入图像 im 进行一维通用滤波
        # 使用全局变量 filter1d 作为滤波器函数，filter_size 作为滤波器大小，
        # extra_arguments=(filter_size,) 传递额外的参数 filter_size 给滤波器函数
        std = ndimage.generic_filter1d(im, filter1d, filter_size, extra_arguments=(filter_size,))
        
        # 断言 res 和 std 的值非常接近，如果不接近则抛出错误信息
        assert_allclose(res, std, err_msg=f"#{j} failed")

    # 使用 enumerate 函数遍历 FILTER1D_FUNCTIONS 列表，同时获取索引 j 和函数对象 func
    for j, func in enumerate(FILTER1D_FUNCTIONS):
        # 调用 check 函数，传入当前索引 j，执行函数内部的一系列操作
        check(j)
# 定义一个测试几何变换的函数
def test_geometric_transform():
    
    # 定义一个局部函数transform，用于根据输出坐标和平移量进行变换
    def transform(output_coordinates, shift):
        return output_coordinates[0] - shift, output_coordinates[1] - shift
    
    # 定义一个局部函数check，用于检查特定变换函数的输出是否正确
    def check(j):
        # 从全局变量TRANSFORM_FUNCTIONS中获取第j个变换函数
        func = TRANSFORM_FUNCTIONS[j]

        # 创建一个4x3的浮点型数组，数值为0到11
        im = np.arange(12).reshape(4, 3).astype(np.float64)
        
        # 设置平移量为0.5
        shift = 0.5

        # 使用ndimage.geometric_transform对数组im进行变换，使用func(shift)作为变换函数
        res = ndimage.geometric_transform(im, func(shift))
        
        # 使用标准的transform函数和额外参数(shift,)调用ndimage.geometric_transform
        std = ndimage.geometric_transform(im, transform, extra_arguments=(shift,))
        
        # 断言res和std的所有元素是否近似相等，若不等则输出失败信息
        assert_allclose(res, std, err_msg=f"#{j} failed")

    # 遍历TRANSFORM_FUNCTIONS全局变量中的所有函数，并逐个调用check函数进行测试
    for j, func in enumerate(TRANSFORM_FUNCTIONS):
        check(j)
```