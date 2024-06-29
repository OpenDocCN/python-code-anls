# `.\numpy\benchmarks\benchmarks\bench_app.py`

```py
from .common import Benchmark  # 导入Benchmark类，来自.common模块

import numpy as np  # 导入NumPy库，用于科学计算

class LaplaceInplace(Benchmark):
    params = ['inplace', 'normal']  # 参数列表，用于测试两种更新方式
    param_names = ['update']  # 参数名称列表，表示参数含义

    def setup(self, update):
        N = 150  # 网格尺寸
        Niter = 1000  # 迭代次数
        dx = 0.1  # x方向步长
        dy = 0.1  # y方向步长
        dx2 = (dx * dx)  # x方向步长的平方
        dy2 = (dy * dy)  # y方向步长的平方

        def num_update(u, dx2, dy2):
            u[1:(-1), 1:(-1)] = ((((u[2:, 1:(-1)] + u[:(-2), 1:(-1)]) * dy2) +
                                  ((u[1:(-1), 2:] + u[1:(-1), :(-2)]) * dx2))
                                 / (2 * (dx2 + dy2)))
            # Laplace更新函数，计算每个网格点的新值

        def num_inplace(u, dx2, dy2):
            tmp = u[:(-2), 1:(-1)].copy()
            np.add(tmp, u[2:, 1:(-1)], out=tmp)
            np.multiply(tmp, dy2, out=tmp)
            tmp2 = u[1:(-1), 2:].copy()
            np.add(tmp2, u[1:(-1), :(-2)], out=tmp2)
            np.multiply(tmp2, dx2, out=tmp2)
            np.add(tmp, tmp2, out=tmp)
            np.multiply(tmp, (1.0 / (2.0 * (dx2 + dy2))),
                        out=u[1:(-1), 1:(-1)])
            # 原地更新的 Laplace 函数，优化内存使用和性能

        def laplace(N, Niter=100, func=num_update, args=()):
            u = np.zeros([N, N], order='C')  # 创建大小为N*N的零数组
            u[0] = 1  # 设置初始条件
            for i in range(Niter):
                func(u, *args)  # 执行指定的更新函数
            return u  # 返回更新后的数组

        func = {'inplace': num_inplace, 'normal': num_update}[update]  # 根据参数选择更新函数

        def run():
            laplace(N, Niter, func, args=(dx2, dy2))  # 运行 Laplace 方程求解函数

        self.run = run  # 将运行函数绑定到类的实例变量

    def time_it(self, update):
        self.run()  # 执行运行函数


class MaxesOfDots(Benchmark):
    def setup(self):
        np.random.seed(1)  # 设置随机种子，确保结果可重复
        nsubj = 5  # 数据集数量
        nfeat = 100  # 特征数量
        ntime = 200  # 时间步数

        self.arrays = [np.random.normal(size=(ntime, nfeat))
                       for i in range(nsubj)]
        # 创建包含随机数据的数组列表

    def maxes_of_dots(self, arrays):
        """
        计算每个数据集中每个特征的特征分数

        :ref:`Haxby et al., Neuron (2011) <HGC+11>`.
        如果在计算前对数组进行了列方向的标准化（zscore-d），结果将表现为每个数组中每个列与其他数组中相应列的最大相关性之和。

        数组只需要在第一维上一致。

        NumPy 使用这个函数来同时评估 1) 点积 和 2) max(<array>, axis=<int>) 的性能。
        """
        feature_scores = ([0] * len(arrays))  # 初始化特征分数列表
        for (i, sd) in enumerate(arrays):
            for (j, sd2) in enumerate(arrays[(i + 1):]):
                corr_temp = np.dot(sd.T, sd2)  # 计算两个数组之间的点积
                feature_scores[i] += np.max(corr_temp, axis=1)  # 计算每列最大值的和
                feature_scores[((j + i) + 1)] += np.max(corr_temp, axis=0)  # 计算每行最大值的和
        return feature_scores  # 返回特征分数列表

    def time_it(self):
        self.maxes_of_dots(self.arrays)  # 执行特征分数计算函数
```