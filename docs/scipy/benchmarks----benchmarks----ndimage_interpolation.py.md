# `D:\src\scipysrc\scipy\benchmarks\benchmarks\ndimage_interpolation.py`

```
# 导入必要的库
import numpy as np

# 从本地模块中导入Benchmark类
from .common import Benchmark

# 尝试导入SciPy的图像处理模块中的函数
try:
    from scipy.ndimage import (geometric_transform, affine_transform, rotate,
                               zoom, shift, map_coordinates)
except ImportError:
    pass  # 如果导入失败，则忽略异常继续执行

# 定义一个用于二维平移函数的辅助函数
def shift_func_2d(c):
    return (c[0] - 0.5, c[1] - 0.5)

# 定义一个用于三维平移函数的辅助函数
def shift_func_3d(c):
    return (c[0] - 0.5, c[1] - 0.5, c[2] - 0.5)

# 继承Benchmark类，用于测试NdimageInterpolation类的性能
class NdimageInterpolation(Benchmark):
    # 参数名列表
    param_names = ['shape', 'order', 'mode']
    # 参数值列表，包含不同的形状、阶数和模式
    params = [
        [(64, 64), (512, 512), (2048, 2048), (16, 16, 16), (128, 128, 128)],
        [0, 1, 3, 5],
        ['mirror', 'constant']
    ]

    # 初始化方法，在每个测试开始之前调用
    def setup(self, shape, order, mode):
        # 使用随机种子为5的随机状态创建随机正态分布数组
        rstate = np.random.RandomState(5)
        self.x = rstate.standard_normal(shape)  # 生成指定形状的随机数组
        # 定义二维仿射变换矩阵
        self.matrix_2d = np.asarray([[0.8, 0, 1.5],
                                     [0, 1.2, -5.]])
        # 定义三维仿射变换矩阵
        self.matrix_3d = np.asarray([[0.8, 0, 0, 1.5],
                                     [0, 1.2, 0, -5.],
                                     [0, 0, 1, 0]])

    # 测试二维仿射变换的性能
    def time_affine_transform(self, shape, order, mode):
        # 根据数组的维度选择合适的仿射变换矩阵
        if self.x.ndim == 2:
            matrix = self.matrix_2d
        else:
            matrix = self.matrix_3d
        affine_transform(self.x, matrix, order=order, mode=mode)

    # 测试数组旋转的性能
    def time_rotate(self, shape, order, mode):
        rotate(self.x, 15, order=order, mode=mode)

    # 测试数组平移的性能
    def time_shift(self, shape, order, mode):
        shift(self.x, (-2.5,) * self.x.ndim, order=order, mode=mode)

    # 测试数组缩放的性能
    def time_zoom(self, shape, order, mode):
        zoom(self.x, (1.3,) * self.x.ndim, order=order, mode=mode)

    # 测试几何变换映射的性能
    def time_geometric_transform_mapping(self, shape, order, mode):
        # 根据数组的维度选择合适的映射函数
        if self.x.ndim == 2:
            mapping = shift_func_2d
        if self.x.ndim == 3:
            mapping = shift_func_3d
        geometric_transform(self.x, mapping, order=order, mode=mode)

    # 测试坐标映射的性能
    def time_map_coordinates(self, shape, order, mode):
        # 生成坐标网格
        coords = np.meshgrid(*[np.arange(0, s, 2) + 0.3 for s in self.x.shape])
        map_coordinates(self.x, coords, order=order, mode=mode)

    # 峰值内存测试：数组旋转的内存使用情况
    def peakmem_rotate(self, shape, order, mode):
        rotate(self.x, 15, order=order, mode=mode)

    # 峰值内存测试：数组平移的内存使用情况
    def peakmem_shift(self, shape, order, mode):
        shift(self.x, 3, order=order, mode=mode)
```