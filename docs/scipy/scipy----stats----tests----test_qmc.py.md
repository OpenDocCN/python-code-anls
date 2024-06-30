# `D:\src\scipysrc\scipy\scipy\stats\tests\test_qmc.py`

```
# 导入必要的库
import os  # 提供与操作系统交互的功能
from collections import Counter  # 提供计数器的数据结构
from itertools import combinations, product  # 提供组合和笛卡尔积的功能

import pytest  # 提供用于编写和运行测试的框架
import numpy as np  # 提供数值计算的支持
from numpy.testing import (assert_allclose, assert_equal, assert_array_equal, 
    assert_array_less)  # 提供用于数组测试的函数

from scipy.spatial import distance  # 提供空间距离计算的功能
from scipy.stats import shapiro  # 提供统计分析中的正态性检验
from scipy.stats._sobol import _test_find_index  # 提供 Sobol 序列的索引查找功能
from scipy.stats import qmc  # 提供用于 QMC (Quasi-Monte Carlo) 方法的支持
from scipy.stats._qmc import (
    van_der_corput, n_primes, primes_from_2_to,
    update_discrepancy, QMCEngine, _l1_norm,
    _perturb_discrepancy, _lloyd_centroidal_voronoi_tessellation
)  # 提供 QMC 方法的具体实现函数和类

class TestUtils:
    def test_scale(self):
        # 1d scalar
        space = [[0], [1], [0.5]]
        out = [[-2], [6], [2]]
        scaled_space = qmc.scale(space, l_bounds=-2, u_bounds=6)

        assert_allclose(scaled_space, out)  # 断言 scaled_space 是否与 out 接近

        # 2d space
        space = [[0, 0], [1, 1], [0.5, 0.5]]
        bounds = np.array([[-2, 0], [6, 5]])
        out = [[-2, 0], [6, 5], [2, 2.5]]

        scaled_space = qmc.scale(space, l_bounds=bounds[0], u_bounds=bounds[1])

        assert_allclose(scaled_space, out)  # 断言 scaled_space 是否与 out 接近

        scaled_back_space = qmc.scale(scaled_space, l_bounds=bounds[0],
                                      u_bounds=bounds[1], reverse=True)
        assert_allclose(scaled_back_space, space)  # 断言 scaled_back_space 是否与 space 接近

        # broadcast
        space = [[0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5]]
        l_bounds, u_bounds = 0, [6, 5, 3]
        out = [[0, 0, 0], [6, 5, 3], [3, 2.5, 1.5]]

        scaled_space = qmc.scale(space, l_bounds=l_bounds, u_bounds=u_bounds)

        assert_allclose(scaled_space, out)  # 断言 scaled_space 是否与 out 接近

    def test_scale_random(self):
        rng = np.random.default_rng(317589836511269190194010915937762468165)
        sample = rng.random((30, 10))  # 生成随机样本数据
        a = -rng.random(10) * 10  # 生成下界
        b = rng.random(10) * 10  # 生成上界
        scaled = qmc.scale(sample, a, b, reverse=False)  # 对样本数据进行缩放
        unscaled = qmc.scale(scaled, a, b, reverse=True)  # 反向缩放操作
        assert_allclose(unscaled, sample)  # 断言 unscaled 是否与 sample 接近
    # 定义测试方法 test_scale_errors，用于测试 qmc.scale 函数在不同情况下是否抛出预期的 ValueError 异常
    def test_scale_errors(self):
        # 第一个测试用例：检查如果 space 不是二维数组，则应该引发 ValueError 异常
        with pytest.raises(ValueError, match=r"Sample is not a 2D array"):
            space = [0, 1, 0.5]
            qmc.scale(space, l_bounds=-2, u_bounds=6)

        # 第二个测试用例：检查如果 bounds 不一致，则应该引发 ValueError 异常
        with pytest.raises(ValueError, match=r"Bounds are not consistent"):
            space = [[0, 0], [1, 1], [0.5, 0.5]]
            bounds = np.array([[-2, 6], [6, 5]])
            qmc.scale(space, l_bounds=bounds[0], u_bounds=bounds[1])

        # 第三个测试用例：检查如果 l_bounds 和 u_bounds 无法广播，则应该引发 ValueError 异常
        with pytest.raises(ValueError, match=r"'l_bounds' and 'u_bounds' must be broadcastable"):
            space = [[0, 0], [1, 1], [0.5, 0.5]]
            l_bounds, u_bounds = [-2, 0, 2], [6, 5]
            qmc.scale(space, l_bounds=l_bounds, u_bounds=u_bounds)

        # 第四个测试用例：检查如果 bounds 不一致无法广播，则应该引发 ValueError 异常
        with pytest.raises(ValueError, match=r"'l_bounds' and 'u_bounds' must be broadcastable"):
            space = [[0, 0], [1, 1], [0.5, 0.5]]
            bounds = np.array([[-2, 0, 2], [6, 5, 5]])
            qmc.scale(space, l_bounds=bounds[0], u_bounds=bounds[1])

        # 第五个测试用例：检查如果样本不在单位超立方体内，则应该引发 ValueError 异常
        with pytest.raises(ValueError, match=r"Sample is not in unit hypercube"):
            space = [[0, 0], [1, 1.5], [0.5, 0.5]]
            bounds = np.array([[-2, 0], [6, 5]])
            qmc.scale(space, l_bounds=bounds[0], u_bounds=bounds[1])

        # 第六个测试用例：检查如果样本超出边界范围，则应该引发 ValueError 异常
        with pytest.raises(ValueError, match=r"Sample is out of bounds"):
            out = [[-2, 0], [6, 5], [8, 2.5]]
            bounds = np.array([[-2, 0], [6, 5]])
            qmc.scale(out, l_bounds=bounds[0], u_bounds=bounds[1], reverse=True)
    # 定义一个测试方法，用于验证不同空间的差异度计算
    def test_discrepancy(self):
        # 创建第一个空间的二维数组并进行归一化处理
        space_1 = np.array([[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]])
        space_1 = (2.0 * space_1 - 1.0) / (2.0 * 6.0)
        
        # 创建第二个空间的二维数组并进行归一化处理
        space_2 = np.array([[1, 5], [2, 4], [3, 3], [4, 2], [5, 1], [6, 6]])
        space_2 = (2.0 * space_2 - 1.0) / (2.0 * 6.0)

        # 使用 Fang et al. (2006) 的方法计算并验证第一个空间的差异度
        assert_allclose(qmc.discrepancy(space_1), 0.0081, atol=1e-4)
        # 使用 Fang et al. (2006) 的方法计算并验证第二个空间的差异度
        assert_allclose(qmc.discrepancy(space_2), 0.0105, atol=1e-4)

        # 使用 Zhou Y.-D. et al. (2013) 的方法计算并验证样本的差异度
        # 这是 Journal of Complexity 中第 29 卷的第 283-301 页的 Example 4
        sample = np.array([[2, 1, 1, 2, 2, 2],
                           [1, 2, 2, 2, 2, 2],
                           [2, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 2, 2],
                           [1, 2, 2, 2, 1, 1],
                           [2, 2, 2, 2, 1, 1],
                           [2, 2, 2, 1, 2, 2]])
        sample = (2.0 * sample - 1.0) / (2.0 * 2.0)

        # 使用不同的方法计算并验证样本的差异度
        assert_allclose(qmc.discrepancy(sample, method='MD'), 2.5000, atol=1e-4)
        assert_allclose(qmc.discrepancy(sample, method='WD'), 1.3680, atol=1e-4)
        assert_allclose(qmc.discrepancy(sample, method='CD'), 0.3172, atol=1e-4)

        # 使用 Tim P. et al. (2005) 的方法计算并验证单位超立方体中单点的 L2 和 Linf 星差异度
        # 这是 JCAM 中第 283 页的 Table 1
        for dim in [2, 4, 8, 16, 32, 64]:
            ref = np.sqrt(3**(-dim))
            assert_allclose(qmc.discrepancy(np.array([[1]*dim]), method='L2-star'), ref)

    # 定义一个测试方法，用于验证差异度计算中的错误情况
    def test_discrepancy_errors(self):
        # 创建一个二维数组样本
        sample = np.array([[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]])

        # 验证样本是否在单位超立方体中，应该引发 ValueError 异常
        with pytest.raises(ValueError, match=r"Sample is not in unit hypercube"):
            qmc.discrepancy(sample)

        # 验证样本是否为二维数组，应该引发 ValueError 异常
        with pytest.raises(ValueError, match=r"Sample is not a 2D array"):
            qmc.discrepancy([1, 3])

        # 创建一个样本列表，验证方法参数是否有效，应该引发 ValueError 异常
        sample = [[0, 0], [1, 1], [0.5, 0.5]]
        with pytest.raises(ValueError, match=r"'toto' is not a valid ..."):
            qmc.discrepancy(sample, method="toto")
    # 测试并行计算不同方法下样本的离散度
    def test_discrepancy_parallel(self, monkeypatch):
        # 创建一个示例样本，是一个 6x6 的二维数组
        sample = np.array([[2, 1, 1, 2, 2, 2],
                           [1, 2, 2, 2, 2, 2],
                           [2, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 2, 2],
                           [1, 2, 2, 2, 1, 1],
                           [2, 2, 2, 2, 1, 1],
                           [2, 2, 2, 1, 2, 2]])
        # 将样本标准化到 [-1, 1] 范围内
        sample = (2.0 * sample - 1.0) / (2.0 * 2.0)

        # 使用 assert_allclose 检查使用不同方法计算离散度的结果是否接近预期值
        assert_allclose(qmc.discrepancy(sample, method='MD', workers=8),
                        2.5000,
                        atol=1e-4)
        assert_allclose(qmc.discrepancy(sample, method='WD', workers=8),
                        1.3680,
                        atol=1e-4)
        assert_allclose(qmc.discrepancy(sample, method='CD', workers=8),
                        0.3172,
                        atol=1e-4)

        # 输出参考文献中的结果，用于验证单点在单位超立方体中的 L2-star 离散度
        # 参考文献：Tim P. et al. Minimizing the L2 and Linf star discrepancies
        # of a single point in the unit hypercube. JCAM, 2005
        # 第 283 页的表格 1
        for dim in [2, 4, 8, 16, 32, 64]:
            ref = np.sqrt(3 ** (-dim))
            assert_allclose(qmc.discrepancy(np.array([[1] * dim]),
                                            method='L2-star', workers=-1), ref)

        # 使用 monkeypatch 模块模拟 os.cpu_count() 返回 None
        monkeypatch.setattr(os, 'cpu_count', lambda: None)
        # 检查在 workers=-1 时是否会引发 NotImplementedError
        with pytest.raises(NotImplementedError, match="Cannot determine the"):
            qmc.discrepancy(sample, workers=-1)

        # 检查在 workers=-2 时是否会引发 ValueError
        with pytest.raises(ValueError, match="Invalid number of workers..."):
            qmc.discrepancy(sample, workers=-2)

    # 测试几何离散度函数的错误处理
    def test_geometric_discrepancy_errors(self):
        # 创建一个示例样本，不在单位超立方体中
        sample = np.array([[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]])

        # 检查样本不在单位超立方体中时是否引发 ValueError
        with pytest.raises(ValueError, match=r"Sample is not in unit hypercube"):
            qmc.geometric_discrepancy(sample)

        # 检查样本不是二维数组时是否引发 ValueError
        with pytest.raises(ValueError, match=r"Sample is not a 2D array"):
            qmc.geometric_discrepancy([1, 3])

        # 检查传入无效方法时是否引发 ValueError
        sample = [[0, 0], [1, 1], [0.5, 0.5]]
        with pytest.raises(ValueError, match=r"'toto' is not a valid ..."):
            qmc.geometric_discrepancy(sample, method="toto")

        # 检查样本包含重复点时是否会发出警告
        sample = np.array([[0, 0], [0, 0], [0, 1]])
        with pytest.warns(UserWarning, match="Sample contains duplicate points."):
            qmc.geometric_discrepancy(sample)

        # 检查样本包含少于两个点时是否引发 ValueError
        sample = np.array([[0.5, 0.5]])
        with pytest.raises(ValueError, match="Sample must contain at least two points"):
            qmc.geometric_discrepancy(sample)
    # 定义一个测试方法，用于测试几何差异度函数的计算
    def test_geometric_discrepancy(self):
        # 创建一个二维数组作为样本数据，包含两个点
        sample = np.array([[0, 0], [1, 1]])
        # 断言计算的几何差异度与预期的平方根(2)相近
        assert_allclose(qmc.geometric_discrepancy(sample), np.sqrt(2))
        # 断言使用最小生成树方法计算的几何差异度与预期的平方根(2)相近
        assert_allclose(qmc.geometric_discrepancy(sample, method="mst"), np.sqrt(2))

        # 创建一个包含三个点的二维数组作为样本数据
        sample = np.array([[0, 0], [0, 1], [0.5, 1]])
        # 断言计算的几何差异度与预期的0.5相近
        assert_allclose(qmc.geometric_discrepancy(sample), 0.5)
        # 断言使用最小生成树方法计算的几何差异度与预期的0.75相近
        assert_allclose(qmc.geometric_discrepancy(sample, method="mst"), 0.75)

        # 创建一个包含三个点的二维数组作为样本数据
        sample = np.array([[0, 0], [0.25, 0.25], [1, 1]])
        # 断言计算的几何差异度与预期的平方根(2)/4相近
        assert_allclose(qmc.geometric_discrepancy(sample), np.sqrt(2) / 4)
        # 断言使用最小生成树方法计算的几何差异度与预期的平方根(2)/2相近
        assert_allclose(qmc.geometric_discrepancy(sample, method="mst"), np.sqrt(2) / 2)
        # 断言使用切比雪夫距离计算的几何差异度与预期的0.25相近
        assert_allclose(qmc.geometric_discrepancy(sample, metric="chebyshev"), 0.25)
        # 断言使用最小生成树方法和切比雪夫距离计算的几何差异度与预期的0.5相近
        assert_allclose(
            qmc.geometric_discrepancy(sample, method="mst", metric="chebyshev"), 0.5
        )

        # 使用指定种子创建随机数生成器对象
        rng = np.random.default_rng(191468432622931918890291693003068437394)
        # 使用拉丁超立方体方法生成维数为3的50个随机样本数据点
        sample = qmc.LatinHypercube(d=3, seed=rng).random(50)
        # 断言计算的几何差异度与预期的0.05106012076093356相近
        assert_allclose(qmc.geometric_discrepancy(sample), 0.05106012076093356)
        # 断言使用最小生成树方法计算的几何差异度与预期的0.19704396643366182相近
        assert_allclose(
            qmc.geometric_discrepancy(sample, method='mst'), 0.19704396643366182
        )

    # 使用 xfail 标记的测试方法，测试在存在零距离时最小生成树方法的行为
    @pytest.mark.xfail(
            reason="minimum_spanning_tree ignores zero distances (#18892)",
            strict=True,
    )
    def test_geometric_discrepancy_mst_with_zero_distances(self):
        # 创建一个包含三个点的二维数组作为样本数据，其中两个点距离为零
        sample = np.array([[0, 0], [0, 0], [0, 1]])
        # 断言使用最小生成树方法计算的几何差异度与预期的0.5相近
        assert_allclose(qmc.geometric_discrepancy(sample, method='mst'), 0.5)
    def test_update_discrepancy(self):
        # 引用自文献 Fang et al. Design and modeling for computer experiments, 2006

        # 创建一个二维数组表示空间点集，每个点有两个维度
        space_1 = np.array([[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]])
        # 将空间点集进行线性转换和缩放，使其位于单位超立方体内
        space_1 = (2.0 * space_1 - 1.0) / (2.0 * 6.0)

        # 计算初始化的 QMC 不一致度
        disc_init = qmc.discrepancy(space_1[:-1], iterative=True)
        # 更新 QMC 不一致度，加入新的点并计算新的不一致度
        disc_iter = update_discrepancy(space_1[-1], space_1[:-1], disc_init)

        # 断言更新后的 QMC 不一致度接近于预期值 0.0081
        assert_allclose(disc_iter, 0.0081, atol=1e-4)

        # 当 n<d 时的情况，生成随机数种子并创建新的空间点集
        rng = np.random.default_rng(241557431858162136881731220526394276199)
        space_1 = rng.random((4, 10))

        # 计算参考的 QMC 不一致度
        disc_ref = qmc.discrepancy(space_1)
        # 计算初始化的 QMC 不一致度
        disc_init = qmc.discrepancy(space_1[:-1], iterative=True)
        # 更新 QMC 不一致度，加入新的点并计算新的不一致度
        disc_iter = update_discrepancy(space_1[-1], space_1[:-1], disc_init)

        # 断言更新后的 QMC 不一致度接近于参考的不一致度
        assert_allclose(disc_iter, disc_ref, atol=1e-4)

        # 引发错误的测试用例：加入的样本点不在单位超立方体内
        with pytest.raises(ValueError, match=r"Sample is not in unit "
                                             r"hypercube"):
            update_discrepancy(space_1[-1], space_1[:-1] + 1, disc_init)

        # 引发错误的测试用例：加入的样本点不是二维数组
        with pytest.raises(ValueError, match=r"Sample is not a 2D array"):
            update_discrepancy(space_1[-1], space_1[0], disc_init)

        # 引发错误的测试用例：加入的样本点不在单位超立方体内
        x_new = [1, 3]
        with pytest.raises(ValueError, match=r"x_new is not in unit "
                                             r"hypercube"):
            update_discrepancy(x_new, space_1[:-1], disc_init)

        # 引发错误的测试用例：加入的样本点不是一维数组
        x_new = [[0.5, 0.5]]
        with pytest.raises(ValueError, match=r"x_new is not a 1D array"):
            update_discrepancy(x_new, space_1[:-1], disc_init)

        # 引发错误的测试用例：加入的样本点和现有样本点不能广播
        x_new = [0.3, 0.1, 0]
        with pytest.raises(ValueError, match=r"x_new and sample must be "
                                             r"broadcastable"):
            update_discrepancy(x_new, space_1[:-1], disc_init)

    def test_perm_discrepancy(self):
        # 随机数种子
        rng = np.random.default_rng(46449423132557934943847369749645759997)
        # 创建拉丁超立方体 QMC 生成器
        qmc_gen = qmc.LatinHypercube(5, seed=rng)
        # 生成 QMC 样本集合
        sample = qmc_gen.random(10)
        # 计算初始 QMC 不一致度
        disc = qmc.discrepancy(sample)

        # 对 QMC 样本进行置换不一致度测试
        for i in range(100):
            # 随机选择两行和一列进行置换
            row_1 = rng.integers(10)
            row_2 = rng.integers(10)
            col = rng.integers(5)

            # 更新 QMC 样本并计算置换后的不一致度
            disc = _perturb_discrepancy(sample, row_1, row_2, col, disc)
            # 实际进行样本置换
            sample[row_1, col], sample[row_2, col] = (
                sample[row_2, col], sample[row_1, col])
            # 计算参考的 QMC 不一致度
            disc_reference = qmc.discrepancy(sample)
            # 断言置换后的不一致度接近于参考的不一致度
            assert_allclose(disc, disc_reference)
    def test_discrepancy_alternative_implementation(self):
        """Alternative definitions from Matt Haberland."""
        
        # 定义第一种离差函数
        def disc_c2(x):
            # 获取输入数组的形状
            n, s = x.shape
            # 将输入数组赋值给新变量 xij
            xij = x
            # 计算离差函数的第一部分
            disc1 = np.sum(np.prod((1
                                    + 1/2*np.abs(xij-0.5)
                                    - 1/2*np.abs(xij-0.5)**2), axis=1))
            # 将 x 的新维度版本赋值给 xij 和 xkj
            xij = x[None, :, :]
            xkj = x[:, None, :]
            # 计算离差函数的第二部分
            disc2 = np.sum(np.sum(np.prod(1
                                          + 1/2*np.abs(xij - 0.5)
                                          + 1/2*np.abs(xkj - 0.5)
                                          - 1/2*np.abs(xij - xkj), axis=2),
                                  axis=0))
            # 返回最终的离差值
            return (13/12)**s - 2/n * disc1 + 1/n**2*disc2

        # 定义第二种离差函数
        def disc_wd(x):
            # 获取输入数组的形状
            n, s = x.shape
            # 创建新的 xij 和 xkj 变量
            xij = x[None, :, :]
            xkj = x[:, None, :]
            # 计算离差函数的值
            disc = np.sum(np.sum(np.prod(3/2
                                         - np.abs(xij - xkj)
                                         + np.abs(xij - xkj)**2, axis=2),
                                 axis=0))
            # 返回最终的离差值
            return -(4/3)**s + 1/n**2 * disc

        # 定义第三种离差函数
        def disc_md(x):
            # 获取输入数组的形状
            n, s = x.shape
            # 将输入数组赋值给新变量 xij
            xij = x
            # 计算离差函数的第一部分
            disc1 = np.sum(np.prod((5/3
                                    - 1/4*np.abs(xij-0.5)
                                    - 1/4*np.abs(xij-0.5)**2), axis=1))
            # 将 x 的新维度版本赋值给 xij 和 xkj
            xij = x[None, :, :]
            xkj = x[:, None, :]
            # 计算离差函数的第二部分
            disc2 = np.sum(np.sum(np.prod(15/8
                                          - 1/4*np.abs(xij - 0.5)
                                          - 1/4*np.abs(xkj - 0.5)
                                          - 3/4*np.abs(xij - xkj)
                                          + 1/2*np.abs(xij - xkj)**2,
                                          axis=2), axis=0))
            # 返回最终的离差值
            return (19/12)**s - 2/n * disc1 + 1/n**2*disc2

        # 定义第四种离差函数
        def disc_star_l2(x):
            # 获取输入数组的形状
            n, s = x.shape
            # 计算离差函数的值
            return np.sqrt(
                3 ** (-s) - 2 ** (1 - s) / n
                * np.sum(np.prod(1 - x ** 2, axis=1))
                + np.sum([
                    np.prod(1 - np.maximum(x[k, :], x[j, :]))
                    for k in range(n) for j in range(n)
                ]) / n ** 2
            )

        # 使用指定的种子创建随机数生成器
        rng = np.random.default_rng(117065081482921065782761407107747179201)
        # 生成指定形状的随机样本
        sample = rng.random((30, 10))

        # 使用 'CD' 方法计算当前离差值
        disc_curr = qmc.discrepancy(sample, method='CD')
        # 使用 disc_c2 函数计算替代的离差值
        disc_alt = disc_c2(sample)
        # 断言两者非常接近
        assert_allclose(disc_curr, disc_alt)

        # 使用 'WD' 方法计算当前离差值
        disc_curr = qmc.discrepancy(sample, method='WD')
        # 使用 disc_wd 函数计算替代的离差值
        disc_alt = disc_wd(sample)
        # 断言两者非常接近
        assert_allclose(disc_curr, disc_alt)

        # 使用 'MD' 方法计算当前离差值
        disc_curr = qmc.discrepancy(sample, method='MD')
        # 使用 disc_md 函数计算替代的离差值
        disc_alt = disc_md(sample)
        # 断言两者非常接近
        assert_allclose(disc_curr, disc_alt)

        # 使用 'L2-star' 方法计算当前离差值
        disc_curr = qmc.discrepancy(sample, method='L2-star')
        # 使用 disc_star_l2 函数计算替代的离差值
        disc_alt = disc_star_l2(sample)
        # 断言两者非常接近
        assert_allclose(disc_curr, disc_alt)
    # 测试函数：测试生成前 n 个质数的函数 n_primes()
    def test_n_primes(self):
        # 调用 n_primes() 函数生成前 10 个质数
        primes = n_primes(10)
        # 断言最后一个生成的质数是否为 29
        assert primes[-1] == 29

        # 调用 n_primes() 函数生成前 168 个质数
        primes = n_primes(168)
        # 断言最后一个生成的质数是否为 997
        assert primes[-1] == 997

        # 调用 n_primes() 函数生成前 350 个质数
        primes = n_primes(350)
        # 断言最后一个生成的质数是否为 2357
        assert primes[-1] == 2357

    # 测试函数：测试生成从2到指定上限的所有质数的函数 primes_from_2_to()
    def test_primes(self):
        # 调用 primes_from_2_to() 函数生成从2到50的所有质数
        primes = primes_from_2_to(50)
        # 预期的输出结果列表
        out = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        # 使用 assert_allclose() 函数断言生成的质数列表是否与预期输出相符
        assert_allclose(primes, out)
class TestVDC:
    # 测试 van_der_corput 函数的基本功能
    def test_van_der_corput(self):
        # 使用默认参数调用 van_der_corput 函数
        sample = van_der_corput(10)
        # 预期输出的样本值
        out = [0.0, 0.5, 0.25, 0.75, 0.125, 0.625,
               0.375, 0.875, 0.0625, 0.5625]
        # 使用 assert_allclose 函数检查实际输出与预期输出是否接近
        assert_allclose(sample, out)

        # 使用指定 workers 参数调用 van_der_corput 函数
        sample = van_der_corput(10, workers=4)
        assert_allclose(sample, out)

        sample = van_der_corput(10, workers=8)
        assert_allclose(sample, out)

        # 使用指定 start_index 参数调用 van_der_corput 函数
        sample = van_der_corput(7, start_index=3)
        assert_allclose(sample, out[3:])

    # 测试 van_der_corput 函数的混淆（scramble）功能
    def test_van_der_corput_scramble(self):
        # 设定种子值
        seed = 338213789010180879520345496831675783177
        # 使用混淆（scramble）参数和指定种子值调用 van_der_corput 函数
        out = van_der_corput(10, scramble=True, seed=seed)

        # 使用混淆（scramble）参数、指定种子值和 start_index 参数调用 van_der_corput 函数
        sample = van_der_corput(7, start_index=3, scramble=True, seed=seed)
        assert_allclose(sample, out[3:])

        # 使用混淆（scramble）参数、指定种子值、start_index 参数和 workers 参数调用 van_der_corput 函数
        sample = van_der_corput(
            7, start_index=3, scramble=True, seed=seed, workers=4
        )
        assert_allclose(sample, out[3:])

        sample = van_der_corput(
            7, start_index=3, scramble=True, seed=seed, workers=8
        )
        assert_allclose(sample, out[3:])

    # 测试当 base 参数为无效值时是否会引发 ValueError 异常
    def test_invalid_base_error(self):
        with pytest.raises(ValueError, match=r"'base' must be at least 2"):
            van_der_corput(10, base=1)


class RandomEngine(qmc.QMCEngine):
    # RandomEngine 类的构造函数，继承自 QMCEngine 类
    def __init__(self, d, optimization=None, seed=None):
        super().__init__(d=d, optimization=optimization, seed=seed)

    # 生成随机数样本的方法
    def _random(self, n=1, *, workers=1):
        # 使用 RNG 实例生成 n 行 self.d 列的随机数样本
        sample = self.rng.random((n, self.d))
        return sample


# 测试 RandomEngine 类的子类化是否正常工作
def test_subclassing_QMCEngine():
    # 创建 RandomEngine 的实例
    engine = RandomEngine(2, seed=175180605424926556207367152557812293274)

    # 调用 random 方法生成样本
    sample_1 = engine.random(n=5)
    sample_2 = engine.random(n=7)
    # 检查生成的样本数量是否正确
    assert engine.num_generated == 12

    # 重置引擎并重新生成样本
    engine.reset()
    assert engine.num_generated == 0

    # 再次调用 random 方法生成样本，比较是否与之前生成的样本相同
    sample_1_test = engine.random(n=5)
    assert_equal(sample_1, sample_1_test)

    # 重复重置和快速前进操作
    engine.reset()
    engine.fast_forward(n=5)
    sample_2_test = engine.random(n=7)

    # 检查生成的样本是否正确，以及生成的总数是否正确
    assert_equal(sample_2, sample_2_test)
    assert engine.num_generated == 12


# 测试异常情况是否会正确引发异常
def test_raises():
    # 输入验证
    with pytest.raises(ValueError, match=r"d must be a non-negative integer"):
        RandomEngine((2,))

    with pytest.raises(ValueError, match=r"d must be a non-negative integer"):
        RandomEngine(-1)

    # 检查 'u_bounds' 和 'l_bounds' 参数是否为整数
    msg = r"'u_bounds' and 'l_bounds' must be integers"
    with pytest.raises(ValueError, match=msg):
        engine = RandomEngine(1)
        engine.integers(l_bounds=1, u_bounds=1.1)


# 测试整数生成方法 integers 的功能
def test_integers():
    # 创建 RandomEngine 的实例
    engine = RandomEngine(1, seed=231195739755290648063853336582377368684)

    # 基本测试
    sample = engine.integers(1, n=10)
    assert_equal(np.unique(sample), [0])

    # 检查生成的样本数据类型是否为 int64
    assert sample.dtype == np.dtype('int64')

    # 使用 endpoint 参数测试
    sample = engine.integers(1, n=10, endpoint=True)
    assert_equal(np.unique(sample), [0, 1])

    # 测试 scaling 逻辑
    low = -5
    high = 7

    # 重置引擎并生成随机数样本
    engine.reset()
    ref_sample = engine.random(20)
    # 将 ref_sample 数组元素缩放到指定的范围 [low, high)，并转换为整型
    ref_sample = ref_sample * (high - low) + low
    
    # 将 ref_sample 数组元素向下取整并转换为64位整型
    ref_sample = np.floor(ref_sample).astype(np.int64)
    
    # 重置随机数生成器 engine 的状态
    engine.reset()
    
    # 使用 engine 生成20个在指定范围 [low, high) 内的随机整数，不包括 high
    sample = engine.integers(low, u_bounds=high, n=20, endpoint=False)
    
    # 断言 sample 数组与 ref_sample 数组相等
    assert_equal(sample, ref_sample)
    
    # 使用 engine 生成100个在指定范围 [low, high) 内的随机整数，不包括 high
    sample = engine.integers(low, u_bounds=high, n=100, endpoint=False)
    
    # 断言 sample 数组的最小值为 low，最大值为 high-1
    assert_equal((sample.min(), sample.max()), (low, high-1))
    
    # 使用 engine 生成100个在指定范围 [low, high] 内的随机整数，包括 high
    sample = engine.integers(low, u_bounds=high, n=100, endpoint=True)
    
    # 断言 sample 数组的最小值为 low，最大值为 high
    assert_equal((sample.min(), sample.max()), (low, high))
def test_integers_nd():
    # 设定维度为 10
    d = 10
    # 使用指定种子创建随机数生成器对象
    rng = np.random.default_rng(3716505122102428560615700415287450951)
    # 生成位于 [-5, -1) 范围内的 d 维整数数组
    low = rng.integers(low=-5, high=-1, size=d)
    # 生成位于 [1, 5] 范围内的 d 维整数数组
    high = rng.integers(low=1, high=5, size=d, endpoint=True)
    # 使用随机数引擎创建 RandomEngine 对象
    engine = RandomEngine(d, seed=rng)

    # 获取从 engine 中生成的整数样本，范围在 low 到 high-1 之间，生成 100 个样本
    sample = engine.integers(low, u_bounds=high, n=100, endpoint=False)
    # 断言样本中每列的最小值等于 low
    assert_equal(sample.min(axis=0), low)
    # 断言样本中每列的最大值等于 high-1
    assert_equal(sample.max(axis=0), high-1)

    # 获取从 engine 中生成的整数样本，范围在 low 到 high 之间，生成 100 个样本
    sample = engine.integers(low, u_bounds=high, n=100, endpoint=True)
    # 断言样本中每列的最小值等于 low
    assert_equal(sample.min(axis=0), low)
    # 断言样本中每列的最大值等于 high
    assert_equal(sample.max(axis=0), high)


class QMCEngineTests:
    """Generic tests for QMC engines."""
    qmce = NotImplemented
    can_scramble = NotImplemented
    unscramble_nd = NotImplemented
    scramble_nd = NotImplemented

    scramble = [True, False]
    ids = ["Scrambled", "Unscrambled"]

    def engine(
        self, scramble: bool,
        seed=170382760648021597650530316304495310428,
        **kwargs
    ) -> QMCEngine:
        # 如果引擎支持混淆，则使用指定的混淆和种子创建 QMCEngine 对象
        if self.can_scramble:
            return self.qmce(scramble=scramble, seed=seed, **kwargs)
        else:
            # 如果引擎不支持混淆，根据参数决定是否跳过测试
            if scramble:
                pytest.skip()
            else:
                return self.qmce(seed=seed, **kwargs)

    # 根据是否混淆返回相应的参考值数组
    def reference(self, scramble: bool) -> np.ndarray:
        return self.scramble_nd if scramble else self.unscramble_nd

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    # 测试零维情况下的随机生成
    def test_0dim(self, scramble):
        engine = self.engine(d=0, scramble=scramble)
        sample = engine.random(4)
        # 断言返回的样本是一个空的 (4, 0) 数组
        assert_array_equal(np.empty((4, 0)), sample)

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    # 测试零样本情况下的随机生成
    def test_0sample(self, scramble):
        engine = self.engine(d=2, scramble=scramble)
        sample = engine.random(0)
        # 断言返回的样本是一个空的 (0, 2) 数组
        assert_array_equal(np.empty((0, 2)), sample)

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    # 测试单个样本的随机生成
    def test_1sample(self, scramble):
        engine = self.engine(d=2, scramble=scramble)
        sample = engine.random(1)
        # 断言返回的样本的形状为 (1, 2)
        assert (1, 2) == sample.shape

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    # 测试边界情况下的随机生成
    def test_bounds(self, scramble):
        engine = self.engine(d=100, scramble=scramble)
        sample = engine.random(512)
        # 断言返回的样本所有元素大于等于 0
        assert np.all(sample >= 0)
        # 断言返回的样本所有元素小于等于 1
        assert np.all(sample <= 1)

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    # 测试生成样本是否与参考样本接近
    def test_sample(self, scramble):
        ref_sample = self.reference(scramble=scramble)
        engine = self.engine(d=2, scramble=scramble)
        sample = engine.random(n=len(ref_sample))

        # 断言返回的样本与参考样本在指定精度范围内接近
        assert_allclose(sample, ref_sample, atol=1e-1)
        # 断言生成的样本数量等于参考样本的长度
        assert engine.num_generated == len(ref_sample)

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    # 定义测试方法，用于测试在给定的混淆参数下的引擎行为
    def test_continuing(self, scramble):
        # 创建引擎对象，设置混淆参数，并生成一个参考样本
        engine = self.engine(d=2, scramble=scramble)
        ref_sample = engine.random(n=8)

        # 重新创建引擎对象，设置相同的混淆参数
        engine = self.engine(d=2, scramble=scramble)

        # 计算参考样本的一半长度
        n_half = len(ref_sample) // 2

        # 使用引擎对象生成一半长度的随机样本
        _ = engine.random(n=n_half)
        sample = engine.random(n=n_half)
        
        # 断言生成的样本与参考样本的后半部分在给定的误差范围内相等
        assert_allclose(sample, ref_sample[n_half:], atol=1e-1)

    # 使用参数化测试，对引擎的重置功能进行测试
    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    @pytest.mark.parametrize(
        "seed",
        (
            170382760648021597650530316304495310428,
            np.random.default_rng(170382760648021597650530316304495310428),
            None,
        ),
    )
    def test_reset(self, scramble, seed):
        # 创建引擎对象，设置混淆参数和种子，并生成一个参考样本
        engine = self.engine(d=2, scramble=scramble, seed=seed)
        ref_sample = engine.random(n=8)

        # 重置引擎对象，断言生成计数器被重置为0
        engine.reset()
        assert engine.num_generated == 0

        # 使用重置后的引擎对象生成新的样本，并断言其与参考样本相等
        sample = engine.random(n=8)
        assert_allclose(sample, ref_sample)

    # 使用参数化测试，对引擎的快进功能进行测试
    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    def test_fast_forward(self, scramble):
        # 创建引擎对象，设置混淆参数，并生成一个参考样本
        engine = self.engine(d=2, scramble=scramble)
        ref_sample = engine.random(n=8)

        # 重新创建引擎对象，设置相同的混淆参数
        engine = self.engine(d=2, scramble=scramble)

        # 快进引擎对象4步，生成一个长度为4的样本
        engine.fast_forward(4)
        sample = engine.random(n=4)

        # 断言生成的样本与参考样本从第4个位置开始在给定的误差范围内相等
        assert_allclose(sample, ref_sample[4:], atol=1e-1)

        # 使用交替的快进和抽样生成样本
        engine.reset()
        even_draws = []
        for i in range(8):
            if i % 2 == 0:
                even_draws.append(engine.random())
            else:
                engine.fast_forward(1)
        
        # 断言偶数索引位置的样本与参考样本中相应位置的值在给定的误差范围内相等
        assert_allclose(
            ref_sample[[i for i in range(8) if i % 2 == 0]],
            np.concatenate(even_draws),
            atol=1e-5
        )

    # 使用参数化测试，对引擎生成的样本分布进行测试
    @pytest.mark.parametrize("scramble", [True])
    def test_distribution(self, scramble):
        d = 50
        # 创建引擎对象，设置维度和混淆参数，并生成一个长度为1024的样本
        engine = self.engine(d=d, scramble=scramble)
        sample = engine.random(1024)
        
        # 断言样本各维度的平均值接近0.5，允许的误差为1e-2
        assert_allclose(
            np.mean(sample, axis=0), np.repeat(0.5, d), atol=1e-2
        )
        
        # 断言样本各维度的第25百分位数接近0.25，允许的误差为1e-2
        assert_allclose(
            np.percentile(sample, 25, axis=0), np.repeat(0.25, d), atol=1e-2
        )
        
        # 断言样本各维度的第75百分位数接近0.75，允许的误差为1e-2
        assert_allclose(
            np.percentile(sample, 75, axis=0), np.repeat(0.75, d), atol=1e-2
        )

    # 测试引擎在不支持的优化方法下是否会引发 ValueError 异常
    def test_raises_optimizer(self):
        message = r"'toto' is not a valid optimization method"
        with pytest.raises(ValueError, match=message):
            self.engine(d=1, scramble=False, optimization="toto")

    # 使用参数化测试，对引擎的优化方法和度量进行测试
    @pytest.mark.parametrize(
        "optimization,metric",
        [
            ("random-CD", qmc.discrepancy),
            ("lloyd", lambda sample: -_l1_norm(sample))]
    )
    # 测试优化器的功能，验证优化后的采样结果的度量值小于未优化的采样结果的度量值
    def test_optimizers(self, optimization, metric):
        # 创建引擎对象，设置维度为2，禁用随机化
        engine = self.engine(d=2, scramble=False)
        # 生成参考样本
        sample_ref = engine.random(n=64)
        # 对参考样本计算度量值
        metric_ref = metric(sample_ref)
    
        # 创建使用优化器的引擎对象，设置维度为2，禁用随机化，并应用给定的优化策略
        optimal_ = self.engine(d=2, scramble=False, optimization=optimization)
        # 生成经优化后的样本
        sample_ = optimal_.random(n=64)
        # 对优化后的样本计算度量值
        metric_ = metric(sample_)
    
        # 断言优化后的度量值小于参考度量值
        assert metric_ < metric_ref
    
    # 测试消费伪随机数生成器状态的功能
    def test_consume_prng_state(self):
        # 使用指定种子创建 NumPy 的伪随机数生成器对象
        rng = np.random.default_rng(0xa29cabb11cfdf44ff6cac8bec254c2a0)
        # 初始化样本列表
        sample = []
        # 迭代三次
        for i in range(3):
            # 创建引擎对象，设置维度为2，启用随机化，并使用指定的伪随机数生成器对象作为种子
            engine = self.engine(d=2, scramble=True, seed=rng)
            # 生成随机样本并添加到样本列表中
            sample.append(engine.random(4))
    
        # 使用 pytest 断言检查第一个和第二个样本是否相等，如果不相等则抛出 AssertionError 异常
        with pytest.raises(AssertionError, match="Arrays are not equal"):
            assert_equal(sample[0], sample[1])
        # 使用 pytest 断言检查第一个和第三个样本是否相等，如果不相等则抛出 AssertionError 异常
        with pytest.raises(AssertionError, match="Arrays are not equal"):
            assert_equal(sample[0], sample[2])
# QMCEngineTests 类的一个子类，用于测试 Halton 序列生成器
class TestHalton(QMCEngineTests):
    # 使用 Halton 序列生成器
    qmce = qmc.Halton
    # 可以进行混淆（scramble）
    can_scramble = True

    # Van der Corput 理论值，已知的理论数值
    unscramble_nd = np.array([[0, 0], [1 / 2, 1 / 3],
                              [1 / 4, 2 / 3], [3 / 4, 1 / 9],
                              [1 / 8, 4 / 9], [5 / 8, 7 / 9],
                              [3 / 8, 2 / 9], [7 / 8, 5 / 9]])
    # 未知的理论值：收敛性质已经验证
    scramble_nd = np.array([[0.50246036, 0.93382481],
                            [0.00246036, 0.26715815],
                            [0.75246036, 0.60049148],
                            [0.25246036, 0.8227137 ],
                            [0.62746036, 0.15604704],
                            [0.12746036, 0.48938037],
                            [0.87746036, 0.71160259],
                            [0.37746036, 0.04493592]])

    # 测试多线程（workers）功能
    def test_workers(self):
        # 获取参考样本，使用混淆（scramble=True）
        ref_sample = self.reference(scramble=True)
        # 创建一个引擎对象，维度为2，并进行混淆
        engine = self.engine(d=2, scramble=True)
        # 生成样本，数量与参考样本相同，使用8个工作线程
        sample = engine.random(n=len(ref_sample), workers=8)

        # 断言样本与参考样本在给定的误差范围内相等
        assert_allclose(sample, ref_sample, atol=1e-3)

        # 重置引擎，生成整数样本
        engine.reset()
        ref_sample = engine.integers(10)
        engine.reset()
        # 使用8个工作线程生成整数样本
        sample = engine.integers(10, workers=8)
        # 断言样本与参考样本相等
        assert_equal(sample, ref_sample)


# QMCEngineTests 类的一个子类，用于测试 LatinHypercube 序列生成器
class TestLHS(QMCEngineTests):
    # 使用 LatinHypercube 序列生成器
    qmce = qmc.LatinHypercube
    # 可以进行混淆（scramble）
    can_scramble = True

    # 测试不适用的方法：不是一个序列
    def test_continuing(self, *args):
        pytest.skip("Not applicable: not a sequence.")

    # 测试不适用的方法：不是一个序列
    def test_fast_forward(self, *args):
        pytest.skip("Not applicable: not a sequence.")

    # 测试不适用的方法：参考样本的值依赖于具体实现
    def test_sample(self, *args):
        pytest.skip("Not applicable: the value of reference sample is"
                    " implementation dependent.")

    # 参数化测试：strength 参数取值为1和2，scramble 参数取值为False和True，optimization 参数取值为None和"random-CD"
    @pytest.mark.parametrize("strength", [1, 2])
    @pytest.mark.parametrize("scramble", [False, True])
    @pytest.mark.parametrize("optimization", [None, "random-CD"])
    def test_sample_stratified(self, optimization, scramble, strength):
        # 使用给定的种子初始化随机数生成器
        seed = np.random.default_rng(37511836202578819870665127532742111260)
        # 设置参数 p
        p = 5
        # 计算样本数 n
        n = p**2
        # 设置维度 d
        d = 6

        # 创建 Latin Hypercube 引擎对象
        engine = qmc.LatinHypercube(d=d, scramble=scramble,
                                    strength=strength,
                                    optimization=optimization,
                                    seed=seed)
        # 生成随机样本
        sample = engine.random(n=n)
        # 断言生成的样本形状为 (n, d)
        assert sample.shape == (n, d)
        # 断言引擎生成的样本数量为 n
        assert engine.num_generated == n

        # centering stratifies samples in the middle of equal segments:
        # * inter-sample distance is constant in 1D sub-projections
        # * after ordering, columns are equal
        # 构造预期的一维样本
        expected1d = (np.arange(n) + 0.5) / n
        # 广播预期的样本形状为 (n, d)
        expected = np.broadcast_to(expected1d, (d, n)).T
        # 断言生成的样本与预期的样本不完全相等
        assert np.any(sample != expected)

        # 对样本按列排序
        sorted_sample = np.sort(sample, axis=0)
        # 设置容差
        tol = 0.5 / n if scramble else 0

        # 断言所有元素在容差范围内的相等性
        assert_allclose(sorted_sample, expected, atol=tol)
        # 断言生成的样本与预期样本之间存在大于容差的差异
        assert np.any(sample - expected > tol)

        # 若 strength 为 2 且 optimization 为 None
        if strength == 2 and optimization is None:
            # 创建唯一元素集合
            unique_elements = np.arange(p)
            # 生成期望的组合集合
            desired = set(product(unique_elements, unique_elements))

            # 对所有可能的组合进行断言
            for i, j in combinations(range(engine.d), 2):
                samples_2d = sample[:, [i, j]]
                res = (samples_2d * p).astype(int)
                res_set = {tuple(row) for row in res}
                assert_equal(res_set, desired)

    def test_optimizer_1d(self):
        # discrepancy measures are invariant under permuting factors and runs
        # 创建不可变的引擎对象
        engine = self.engine(d=1, scramble=False)
        # 生成参考样本
        sample_ref = engine.random(n=64)

        # 创建优化引擎对象
        optimal_ = self.engine(d=1, scramble=False, optimization="random-CD")
        # 生成优化后的样本
        sample_ = optimal_.random(n=64)

        # 断言参考样本与优化后的样本相等
        assert_array_equal(sample_ref, sample_)

    def test_raises(self):
        # 异常消息
        message = r"not a valid strength"
        # 断言抛出 ValueError 异常，并匹配给定的消息
        with pytest.raises(ValueError, match=message):
            qmc.LatinHypercube(1, strength=3)

        # 异常消息
        message = r"n is not the square of a prime number"
        # 断言抛出 ValueError 异常，并匹配给定的消息
        with pytest.raises(ValueError, match=message):
            engine = qmc.LatinHypercube(d=2, strength=2)
            engine.random(16)

        # 异常消息
        message = r"n is not the square of a prime number"
        # 断言抛出 ValueError 异常，并匹配给定的消息
        with pytest.raises(ValueError, match=message):
            engine = qmc.LatinHypercube(d=2, strength=2)
            engine.random(5)  # because int(sqrt(5)) would result in 2

        # 异常消息
        message = r"n is too small for d"
        # 断言抛出 ValueError 异常，并匹配给定的消息
        with pytest.raises(ValueError, match=message):
            engine = qmc.LatinHypercube(d=5, strength=2)
            engine.random(9)
# 定义一个测试类 TestSobol，继承自 QMCEngineTests
class TestSobol(QMCEngineTests):
    # 设置 qmce 属性为 Sobol 类
    qmce = qmc.Sobol
    # 允许进行混淆（scramble）
    can_scramble = True

    # 理论数值，来自 Joe Kuo2010
    unscramble_nd = np.array([[0., 0.],
                              [0.5, 0.5],
                              [0.75, 0.25],
                              [0.25, 0.75],
                              [0.375, 0.375],
                              [0.875, 0.875],
                              [0.625, 0.125],
                              [0.125, 0.625]])

    # 理论数值未知：收敛性质已验证
    scramble_nd = np.array([[0.25331921, 0.41371179],
                            [0.8654213, 0.9821167],
                            [0.70097554, 0.03664616],
                            [0.18027647, 0.60895735],
                            [0.10521339, 0.21897069],
                            [0.53019685, 0.66619033],
                            [0.91122276, 0.34580743],
                            [0.45337471, 0.78912079]])

    # 定义一个测试函数 test_warning
    def test_warning(self):
        # 检查是否会发出 UserWarning，并匹配特定的警告信息
        with pytest.warns(UserWarning, match=r"The balance properties of "
                                             r"Sobol' points"):
            # 创建一个 Sobol 引擎对象，维度为 1
            engine = qmc.Sobol(1)
            # 生成 10 个随机数
            engine.random(10)

    # 定义一个测试函数 test_random_base2
    def test_random_base2(self):
        # 创建一个 Sobol 引擎对象，维度为 2，禁用混淆
        engine = qmc.Sobol(2, scramble=False)
        # 生成一个基于 2 的随机样本，长度为 2
        sample = engine.random_base2(2)
        # 检查生成的样本与预期的非混淆样本是否一致
        assert_array_equal(self.unscramble_nd[:4], sample)

        # 再次生成基于 2 的随机样本，长度为 2
        sample = engine.random_base2(2)
        # 检查生成的样本与预期的非混淆样本的后半部分是否一致
        assert_array_equal(self.unscramble_nd[4:8], sample)

        # 再次尝试生成基于 2 的随机样本，但会导致 N 不等于 2**n
        with pytest.raises(ValueError, match=r"The balance properties of "
                                             r"Sobol' points"):
            engine.random_base2(2)

    # 定义一个测试函数 test_raise
    def test_raise(self):
        # 检查是否会引发 ValueError，并匹配特定的异常信息
        with pytest.raises(ValueError, match=r"Maximum supported "
                                             r"dimensionality"):
            # 创建一个维度超过最大支持值的 Sobol 引擎对象
            qmc.Sobol(qmc.Sobol.MAXDIM + 1)

        # 检查是否会引发 ValueError，并匹配特定的异常信息
        with pytest.raises(ValueError, match=r"Maximum supported "
                                             r"'bits' is 64"):
            # 创建一个 bits 超过最大支持值的 Sobol 引擎对象，维度为 1
            qmc.Sobol(1, bits=65)

    # 定义一个测试函数 test_high_dim
    def test_high_dim(self):
        # 创建一个维度为 1111 的 Sobol 引擎对象，禁用混淆
        engine = qmc.Sobol(1111, scramble=False)
        # 统计生成的随机数的分布情况，期望所有随机数为 0.0
        count1 = Counter(engine.random().flatten().tolist())
        # 统计再次生成的随机数的分布情况，期望所有随机数为 0.5
        count2 = Counter(engine.random().flatten().tolist())
        # 断言两次生成的随机数分布符合预期
        assert_equal(count1, Counter({0.0: 1111}))
        assert_equal(count2, Counter({0.5: 1111}))

    # 使用 pytest.mark.parametrize 参数化测试函数 test_bits
    @pytest.mark.parametrize("bits", [2, 3])
    def test_bits(self, bits):
        # 创建一个维度为 2 的 Sobol 引擎对象，禁用混淆，设置 bits 值
        engine = qmc.Sobol(2, scramble=False, bits=bits)
        # 计算样本数 ns，应为 2 的 bits 次方
        ns = 2**bits
        # 生成 ns 个随机样本
        sample = engine.random(ns)
        # 检查生成的随机样本与预期的非混淆样本是否一致
        assert_array_equal(self.unscramble_nd[:ns], sample)

        # 尝试生成未指定 bits 值的随机样本，应该引发 ValueError
        with pytest.raises(ValueError, match="increasing `bits`"):
            engine.random()

    # 定义一个测试函数 test_64bits
    def test_64bits(self):
        # 创建一个维度为 2 的 Sobol 引擎对象，禁用混淆，设置 bits 值为 64
        engine = qmc.Sobol(2, scramble=False, bits=64)
        # 生成 8 个随机样本
        sample = engine.random(8)
        # 检查生成的随机样本与预期的非混淆样本是否一致
        assert_array_equal(self.unscramble_nd, sample)
    # 将类 qmc.PoissonDisk 赋值给变量 qmce
    qmce = qmc.PoissonDisk
    # 设置变量 can_scramble 为 False
    can_scramble = False

    def test_bounds(self, *args):
        # 跳过测试并提示原因：内存开销过大
        pytest.skip("Too costly in memory.")

    def test_fast_forward(self, *args):
        # 跳过测试并提示原因：不适用于递归过程
        pytest.skip("Not applicable: recursive process.")

    def test_sample(self, *args):
        # 跳过测试并提示原因：参考样本值依赖于具体实现
        pytest.skip("Not applicable: the value of reference sample is"
                    " implementation dependent.")

    def test_continuing(self, *args):
        # 可以继续采样，但不保证相同顺序，因为候选点可能会丢失，所以不会选择相同的中心点
        radius = 0.05
        ns = 6
        # 使用给定的引擎创建采样引擎对象，设置维度为2，半径为radius，禁用混淆
        engine = self.engine(d=2, radius=radius, scramble=False)

        # 初始采样
        sample_init = engine.random(n=ns)
        assert len(sample_init) <= ns
        assert l2_norm(sample_init) >= radius

        # 继续采样
        sample_continued = engine.random(n=ns)
        assert len(sample_continued) <= ns
        assert l2_norm(sample_continued) >= radius

        # 合并两次采样结果
        sample = np.concatenate([sample_init, sample_continued], axis=0)
        assert len(sample) <= ns * 2
        assert l2_norm(sample) >= radius

    def test_mindist(self):
        # 创建用于生成随机数的 RNG 对象
        rng = np.random.default_rng(132074951149370773672162394161442690287)
        ns = 50

        # 设置半径的范围
        low, high = 0.08, 0.2
        radii = (high - low) * rng.random(5) + low

        # 设置维度和超球面方法的组合
        dimensions = [1, 3, 4]
        hypersphere_methods = ["volume", "surface"]

        # 创建维度、半径、超球面方法的组合生成器
        gen = product(dimensions, radii, hypersphere_methods)

        # 遍历组合并进行测试
        for d, radius, hypersphere in gen:
            # 使用给定的 qmce 类型创建采样引擎对象
            engine = self.qmce(
                d=d, radius=radius, hypersphere=hypersphere, seed=rng
            )
            # 进行随机采样
            sample = engine.random(ns)

            assert len(sample) <= ns
            assert l2_norm(sample) >= radius

    def test_fill_space(self):
        # 设置半径
        radius = 0.2
        # 使用给定的 qmce 类型创建采样引擎对象
        engine = self.qmce(d=2, radius=radius)

        # 填充空间并进行 l2 范数测试
        sample = engine.fill_space()
        # 圆装填问题较为复杂
        assert l2_norm(sample) >= radius

    @pytest.mark.parametrize("l_bounds", [[-1, -2, -1], [1, 2, 1]])
    def test_sample_inside_lower_bounds(self, l_bounds):
        # 设置半径
        radius = 0.2
        u_bounds=[3, 3, 2]
        # 使用给定的 qmce 类型创建采样引擎对象，设置维度为3，半径为radius，下界为l_bounds，上界为u_bounds
        engine = self.qmce(
            d=3, radius=radius, l_bounds=l_bounds, u_bounds=u_bounds
        )
        # 进行随机采样
        sample = engine.random(30)

        # 验证采样点在上界和下界内
        for point in sample:
            assert_array_less(point, u_bounds) 
            assert_array_less(l_bounds, point) 

    @pytest.mark.parametrize("u_bounds", [[-1, -2, -1], [1, 2, 1]])
    def test_sample_inside_upper_bounds(self, u_bounds):
        # 设置半径
        radius = 0.2
        l_bounds=[-3, -3, -2]
        # 使用给定的 qmce 类型创建采样引擎对象，设置维度为3，半径为radius，下界为l_bounds，上界为u_bounds
        engine = self.qmce(
            d=3, radius=radius, l_bounds=l_bounds, u_bounds=u_bounds
        )
        # 进行随机采样
        sample = engine.random(30)

        # 验证采样点在上界和下界内
        for point in sample:
            assert_array_less(point, u_bounds) 
            assert_array_less(l_bounds, point) 
    # 定义一个测试函数，用于测试不一致的边界值情况
    def test_inconsistent_bound_value(self):
        # 设置半径为0.2
        radius = 0.2
        # 设置下界列表
        l_bounds=[3, 2, 1]
        # 设置上界列表
        u_bounds=[-1, -2, -1]
        # 使用 pytest 的断言检查是否会引发 ValueError 异常，并匹配特定错误消息
        with pytest.raises(
            ValueError, 
            match="Bounds are not consistent 'l_bounds' < 'u_bounds'"):
            # 调用 qmce 方法，期望引发异常
            self.qmce(d=3, radius=radius, l_bounds=l_bounds, u_bounds=u_bounds)

    # 使用 pytest 的参数化装饰器定义另一个测试函数，测试不一致的边界情况
    @pytest.mark.parametrize("u_bounds", [[-1, -2, -1], [-1, -2]])
    @pytest.mark.parametrize("l_bounds", [[3, 2]])
    def test_inconsistent_bounds(self, u_bounds, l_bounds):
        # 设置半径为0.2
        radius = 0.2
        # 使用 pytest 的断言检查是否会引发 ValueError 异常，并匹配特定错误消息
        with pytest.raises(
            ValueError, 
            match="'l_bounds' and 'u_bounds' must be broadcastable and respect" 
            " the sample dimension"):
            # 调用 qmce 方法，期望引发异常
            self.qmce(
                d=3, radius=radius, 
                l_bounds=l_bounds, u_bounds=u_bounds
            )
        
    # 定义一个测试函数，用于测试特定异常情况
    def test_raises(self):
        # 设置错误消息
        message = r"'toto' is not a valid hypersphere sampling"
        # 使用 pytest 的断言检查是否会引发 ValueError 异常，并匹配特定错误消息
        with pytest.raises(ValueError, match=message):
            # 创建 PoissonDisk 对象时传入非法参数，期望引发异常
            qmc.PoissonDisk(1, hypersphere="toto")
class TestMultinomialQMC:
    # 测试类，用于测试 MultinomialQMC 类的功能

    def test_validations(self):
        # 测试 MultinomialQMC 类中的验证功能

        # 测试负数概率值引发 ValueError 异常
        p = np.array([0.12, 0.26, -0.05, 0.35, 0.22])
        with pytest.raises(ValueError, match=r"Elements of pvals must "
                                             r"be non-negative."):
            qmc.MultinomialQMC(p, n_trials=10)

        # 测试概率和不等于 1 引发 ValueError 异常
        p = np.array([0.12, 0.26, 0.1, 0.35, 0.22])
        message = r"Elements of pvals must sum to 1."
        with pytest.raises(ValueError, match=message):
            qmc.MultinomialQMC(p, n_trials=10)

        # 测试指定的 qmc.MultinomialQMC 引擎维度不符合要求引发 ValueError 异常
        p = np.array([0.12, 0.26, 0.05, 0.35, 0.22])
        message = r"Dimension of `engine` must be 1."
        with pytest.raises(ValueError, match=message):
            qmc.MultinomialQMC(p, n_trials=10, engine=qmc.Sobol(d=2))

        # 测试指定的 qmc.MultinomialQMC 引擎类型不符合要求引发 ValueError 异常
        message = r"`engine` must be an instance of..."
        with pytest.raises(ValueError, match=message):
            qmc.MultinomialQMC(p, n_trials=10, engine=np.random.default_rng())

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_MultinomialBasicDraw(self):
        # 忽略 UserWarning 类型的警告

        # 测试基本抽样功能
        seed = np.random.default_rng(6955663962957011631562466584467607969)
        p = np.array([0.12, 0.26, 0.05, 0.35, 0.22])
        n_trials = 100
        expected = np.atleast_2d(n_trials * p).astype(int)
        engine = qmc.MultinomialQMC(p, n_trials=n_trials, seed=seed)
        assert_allclose(engine.random(1), expected, atol=1)

    def test_MultinomialDistribution(self):
        # 测试多项式分布的抽样结果是否符合期望

        seed = np.random.default_rng(77797854505813727292048130876699859000)
        p = np.array([0.12, 0.26, 0.05, 0.35, 0.22])
        engine = qmc.MultinomialQMC(p, n_trials=8192, seed=seed)
        draws = engine.random(1)
        assert_allclose(draws / np.sum(draws), np.atleast_2d(p), atol=1e-4)

    def test_FindIndex(self):
        # 测试 _test_find_index 函数的功能

        p_cumulative = np.array([0.1, 0.4, 0.45, 0.6, 0.75, 0.9, 0.99, 1.0])
        size = len(p_cumulative)
        # 测试不同的概率值定位索引的准确性
        assert_equal(_test_find_index(p_cumulative, size, 0.0), 0)
        assert_equal(_test_find_index(p_cumulative, size, 0.4), 2)
        assert_equal(_test_find_index(p_cumulative, size, 0.44999), 2)
        assert_equal(_test_find_index(p_cumulative, size, 0.45001), 3)
        assert_equal(_test_find_index(p_cumulative, size, 1.0), size - 1)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_other_engine(self):
        # 忽略 UserWarning 类型的警告

        # 测试使用不同引擎的基本抽样功能
        seed = np.random.default_rng(283753519042773243071753037669078065412)
        p = np.array([0.12, 0.26, 0.05, 0.35, 0.22])
        n_trials = 100
        expected = np.atleast_2d(n_trials * p).astype(int)
        base_engine = qmc.Sobol(1, scramble=True, seed=seed)
        engine = qmc.MultinomialQMC(p, n_trials=n_trials, engine=base_engine,
                                    seed=seed)
        assert_allclose(engine.random(1), expected, atol=1)
    def test_NormalQMC(self):
        # 定义测试函数 test_NormalQMC，用于测试 MultivariateNormalQMC 类

        # 测试维度 d = 1 的情况
        engine = qmc.MultivariateNormalQMC(mean=np.zeros(1))
        # 生成一个随机样本
        samples = engine.random()
        assert_equal(samples.shape, (1, 1))
        # 生成 n=5 个随机样本
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 1))

        # 测试维度 d = 2 的情况
        engine = qmc.MultivariateNormalQMC(mean=np.zeros(2))
        # 生成一个随机样本
        samples = engine.random()
        assert_equal(samples.shape, (1, 2))
        # 生成 n=5 个随机样本
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 2))

    def test_NormalQMCInvTransform(self):
        # 定义测试函数 test_NormalQMCInvTransform，用于测试带逆变换的 MultivariateNormalQMC 类

        # 测试维度 d = 1 的情况
        engine = qmc.MultivariateNormalQMC(
            mean=np.zeros(1), inv_transform=True)
        # 生成一个随机样本
        samples = engine.random()
        assert_equal(samples.shape, (1, 1))
        # 生成 n=5 个随机样本
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 1))

        # 测试维度 d = 2 的情况
        engine = qmc.MultivariateNormalQMC(
            mean=np.zeros(2), inv_transform=True)
        # 生成一个随机样本
        samples = engine.random()
        assert_equal(samples.shape, (1, 2))
        # 生成 n=5 个随机样本
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 2))

    def test_NormalQMCSeeded(self):
        # 定义测试函数 test_NormalQMCSeeded，用于测试带种子的 MultivariateNormalQMC 类

        # 测试偶数维度的情况
        seed = np.random.default_rng(274600237797326520096085022671371676017)
        engine = qmc.MultivariateNormalQMC(
            mean=np.zeros(2), inv_transform=False, seed=seed)
        samples = engine.random(n=2)
        samples_expected = np.array([[-0.932001, -0.522923],
                                     [-1.477655, 0.846851]])
        assert_allclose(samples, samples_expected, atol=1e-4)

        # 测试奇数维度的情况
        seed = np.random.default_rng(274600237797326520096085022671371676017)
        engine = qmc.MultivariateNormalQMC(
            mean=np.zeros(3), inv_transform=False, seed=seed)
        samples = engine.random(n=2)
        samples_expected = np.array([[-0.932001, -0.522923, 0.036578],
                                     [-1.778011, 0.912428, -0.065421]])
        assert_allclose(samples, samples_expected, atol=1e-4)

        # 使用另一个引擎再次进行相同的测试
        seed = np.random.default_rng(274600237797326520096085022671371676017)
        base_engine = qmc.Sobol(4, scramble=True, seed=seed)
        engine = qmc.MultivariateNormalQMC(
            mean=np.zeros(3), inv_transform=False,
            engine=base_engine, seed=seed
        )
        samples = engine.random(n=2)
        samples_expected = np.array([[-0.932001, -0.522923, 0.036578],
                                     [-1.778011, 0.912428, -0.065421]])
        assert_allclose(samples, samples_expected, atol=1e-4)
    def test_NormalQMCSeededInvTransform(self):
        # 测试偶数维度

        # 使用指定种子创建随机数生成器对象
        seed = np.random.default_rng(288527772707286126646493545351112463929)
        # 创建 QMC 引擎对象，指定均值为零向量，使用逆变换方法
        engine = qmc.MultivariateNormalQMC(
            mean=np.zeros(2), seed=seed, inv_transform=True)
        # 生成随机样本
        samples = engine.random(n=2)
        # 预期的样本值
        samples_expected = np.array([[-0.913237, -0.964026],
                                     [0.255904, 0.003068]])
        # 检查生成的样本是否与预期值在给定的绝对误差范围内相似
        assert_allclose(samples, samples_expected, atol=1e-4)

        # 测试奇数维度

        # 使用相同的种子创建新的随机数生成器对象
        seed = np.random.default_rng(288527772707286126646493545351112463929)
        # 创建 QMC 引擎对象，指定均值为零向量，使用逆变换方法
        engine = qmc.MultivariateNormalQMC(
            mean=np.zeros(3), seed=seed, inv_transform=True)
        # 生成随机样本
        samples = engine.random(n=2)
        # 预期的样本值
        samples_expected = np.array([[-0.913237, -0.964026, 0.355501],
                                     [0.699261, 2.90213 , -0.6418]])
        # 检查生成的样本是否与预期值在给定的绝对误差范围内相似
        assert_allclose(samples, samples_expected, atol=1e-4)

    def test_other_engine(self):
        # 遍历维度列表 (0, 1, 2)
        for d in (0, 1, 2):
            # 创建 Sobol 引擎对象，指定维度和禁用混洗选项
            base_engine = qmc.Sobol(d=d, scramble=False)
            # 创建 QMC 引擎对象，指定均值为零向量，使用基础引擎和逆变换方法
            engine = qmc.MultivariateNormalQMC(mean=np.zeros(d),
                                               engine=base_engine,
                                               inv_transform=True)
            # 生成随机样本
            samples = engine.random()
            # 检查生成的样本形状是否符合预期 (1, d)
            assert_equal(samples.shape, (1, d))

    def test_NormalQMCShapiro(self):
        # 使用指定种子创建随机数生成器对象
        rng = np.random.default_rng(13242)
        # 创建 QMC 引擎对象，指定均值为零向量，使用指定种子
        engine = qmc.MultivariateNormalQMC(mean=np.zeros(2), seed=rng)
        # 生成随机样本
        samples = engine.random(n=256)
        # 检查样本均值是否接近零向量
        assert all(np.abs(samples.mean(axis=0)) < 1e-2)
        # 检查样本标准差是否接近 1
        assert all(np.abs(samples.std(axis=0) - 1) < 1e-2)
        # 对每个维度执行 Shapiro-Wilk 正态性检验
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            # 确保 p 值大于 0.9，样本满足正态性假设
            assert pval > 0.9
        # 确保样本各维度之间无显著相关性
        cov = np.cov(samples.transpose())
        assert np.abs(cov[0, 1]) < 1e-2

    def test_NormalQMCShapiroInvTransform(self):
        # 使用指定种子创建随机数生成器对象
        rng = np.random.default_rng(32344554)
        # 创建 QMC 引擎对象，指定均值为零向量，使用逆变换方法和指定种子
        engine = qmc.MultivariateNormalQMC(
            mean=np.zeros(2), inv_transform=True, seed=rng)
        # 生成随机样本
        samples = engine.random(n=256)
        # 检查样本均值是否接近零向量
        assert all(np.abs(samples.mean(axis=0)) < 1e-2)
        # 检查样本标准差是否接近 1
        assert all(np.abs(samples.std(axis=0) - 1) < 1e-2)
        # 对每个维度执行 Shapiro-Wilk 正态性检验
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            # 确保 p 值大于 0.9，样本满足正态性假设
            assert pval > 0.9
        # 确保样本各维度之间无显著相关性
        cov = np.cov(samples.transpose())
        assert np.abs(cov[0, 1]) < 1e-2
class TestMultivariateNormalQMC:

    def test_validations(self):
        # 定义错误信息
        message = r"Dimension of `engine` must be consistent"
        # 使用 pytest 来验证抛出 ValueError 异常，并匹配指定的错误信息
        with pytest.raises(ValueError, match=message):
            # 创建 MultivariateNormalQMC 对象，传入错误维度的引擎对象
            qmc.MultivariateNormalQMC([0], engine=qmc.Sobol(d=2))

        message = r"Dimension of `engine` must be consistent"
        with pytest.raises(ValueError, match=message):
            qmc.MultivariateNormalQMC([0, 0, 0], engine=qmc.Sobol(d=4))

        message = r"`engine` must be an instance of..."
        with pytest.raises(ValueError, match=message):
            # 创建 MultivariateNormalQMC 对象，传入不是 QMC 引擎的对象
            qmc.MultivariateNormalQMC([0, 0], engine=np.random.default_rng())

        message = r"Covariance matrix not PSD."
        with pytest.raises(ValueError, match=message):
            # 创建 MultivariateNormalQMC 对象，传入不是正定半定的协方差矩阵
            qmc.MultivariateNormalQMC([0, 0], [[1, 2], [2, 1]])

        message = r"Covariance matrix is not symmetric."
        with pytest.raises(ValueError, match=message):
            # 创建 MultivariateNormalQMC 对象，传入不对称的协方差矩阵
            qmc.MultivariateNormalQMC([0, 0], [[1, 0], [2, 1]])

        message = r"Dimension mismatch between mean and covariance."
        with pytest.raises(ValueError, match=message):
            # 创建 MultivariateNormalQMC 对象，传入维度不匹配的均值和协方差矩阵
            qmc.MultivariateNormalQMC([0], [[1, 0], [0, 1]])

    def test_MultivariateNormalQMCNonPD(self):
        # 使用非正定但半正定的协方差矩阵，应该能正常工作
        engine = qmc.MultivariateNormalQMC(
            [0, 0, 0], [[1, 0, 1], [0, 1, 1], [1, 1, 2]],
        )
        # 断言引擎的相关矩阵不为 None
        assert engine._corr_matrix is not None

    def test_MultivariateNormalQMC(self):
        # d = 1 scalar
        engine = qmc.MultivariateNormalQMC(mean=0, cov=5)
        # 生成一个随机样本
        samples = engine.random()
        # 断言样本形状为 (1, 1)
        assert_equal(samples.shape, (1, 1))
        # 生成多个随机样本
        samples = engine.random(n=5)
        # 断言样本形状为 (5, 1)
        assert_equal(samples.shape, (5, 1))

        # d = 2 list
        engine = qmc.MultivariateNormalQMC(mean=[0, 1], cov=[[1, 0], [0, 1]])
        samples = engine.random()
        assert_equal(samples.shape, (1, 2))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 2))

        # d = 3 np.array
        mean = np.array([0, 1, 2])
        cov = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        engine = qmc.MultivariateNormalQMC(mean, cov)
        samples = engine.random()
        assert_equal(samples.shape, (1, 3))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 3))
    # 定义一个测试方法，测试多变量正态分布的 QMC 逆变换抽样
    def test_MultivariateNormalQMCInvTransform(self):
        # d = 1，标量情况
        engine = qmc.MultivariateNormalQMC(mean=0, cov=5, inv_transform=True)
        # 生成一个随机样本
        samples = engine.random()
        # 断言样本形状为 (1, 1)
        assert_equal(samples.shape, (1, 1))
        # 生成 n=5 个随机样本
        samples = engine.random(n=5)
        # 断言样本形状为 (5, 1)
        assert_equal(samples.shape, (5, 1))

        # d = 2，列表情况
        engine = qmc.MultivariateNormalQMC(
            mean=[0, 1], cov=[[1, 0], [0, 1]], inv_transform=True,
        )
        # 生成一个随机样本
        samples = engine.random()
        # 断言样本形状为 (1, 2)
        assert_equal(samples.shape, (1, 2))
        # 生成 n=5 个随机样本
        samples = engine.random(n=5)
        # 断言样本形状为 (5, 2)
        assert_equal(samples.shape, (5, 2))

        # d = 3，numpy 数组情况
        mean = np.array([0, 1, 2])
        cov = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        engine = qmc.MultivariateNormalQMC(mean, cov, inv_transform=True)
        # 生成一个随机样本
        samples = engine.random()
        # 断言样本形状为 (1, 3)
        assert_equal(samples.shape, (1, 3))
        # 生成 n=5 个随机样本
        samples = engine.random(n=5)
        # 断言样本形状为 (5, 3)
        assert_equal(samples.shape, (5, 3))

    # 定义一个测试方法，测试多变量正态分布的 QMC 使用种子生成
    def test_MultivariateNormalQMCSeeded(self):
        # 测试偶数维度
        rng = np.random.default_rng(180182791534511062935571481899241825000)
        a = rng.standard_normal((2, 2))
        A = a @ a.transpose() + np.diag(rng.random(2))
        engine = qmc.MultivariateNormalQMC(np.array([0, 0]), A,
                                           inv_transform=False, seed=rng)
        # 生成 2 个随机样本
        samples = engine.random(n=2)
        # 期望的样本值
        samples_expected = np.array([[-0.64419, -0.882413],
                                     [0.837199, 2.045301]])
        # 断言生成的样本与期望的样本非常接近，允许的误差为 1e-4
        assert_allclose(samples, samples_expected, atol=1e-4)

        # 测试奇数维度
        rng = np.random.default_rng(180182791534511062935571481899241825000)
        a = rng.standard_normal((3, 3))
        A = a @ a.transpose() + np.diag(rng.random(3))
        engine = qmc.MultivariateNormalQMC(np.array([0, 0, 0]), A,
                                           inv_transform=False, seed=rng)
        # 生成 2 个随机样本
        samples = engine.random(n=2)
        # 期望的样本值
        samples_expected = np.array([[-0.693853, -1.265338, -0.088024],
                                     [1.620193, 2.679222, 0.457343]])
        # 断言生成的样本与期望的样本非常接近，允许的误差为 1e-4
        assert_allclose(samples, samples_expected, atol=1e-4)
    # 定义测试函数 test_MultivariateNormalQMCSeededInvTransform
    def test_MultivariateNormalQMCSeededInvTransform(self):
        # 测试偶数维度情况

        # 使用指定种子创建随机数生成器
        rng = np.random.default_rng(224125808928297329711992996940871155974)
        # 生成一个 2x2 的标准正态分布矩阵 a
        a = rng.standard_normal((2, 2))
        # 计算协方差矩阵 A
        A = a @ a.transpose() + np.diag(rng.random(2))
        # 创建 QMC 引擎对象，设置种子和反变换为真
        engine = qmc.MultivariateNormalQMC(
            np.array([0, 0]), A, seed=rng, inv_transform=True
        )
        # 生成随机样本，期望的样本值为指定的值
        samples = engine.random(n=2)
        samples_expected = np.array([[0.682171, -3.114233],
                                     [-0.098463, 0.668069]])
        # 断言生成的样本与期望值在指定容差范围内相等
        assert_allclose(samples, samples_expected, atol=1e-4)

        # 测试奇数维度情况

        # 使用相同的种子创建新的随机数生成器
        rng = np.random.default_rng(224125808928297329711992996940871155974)
        # 生成一个 3x3 的标准正态分布矩阵 a
        a = rng.standard_normal((3, 3))
        # 计算协方差矩阵 A
        A = a @ a.transpose() + np.diag(rng.random(3))
        # 创建 QMC 引擎对象，设置种子和反变换为真
        engine = qmc.MultivariateNormalQMC(
            np.array([0, 0, 0]), A, seed=rng, inv_transform=True
        )
        # 生成随机样本，期望的样本值为指定的值
        samples = engine.random(n=2)
        samples_expected = np.array([[0.988061, -1.644089, -0.877035],
                                     [-1.771731, 1.096988, 2.024744]])
        # 断言生成的样本与期望值在指定容差范围内相等
        assert_allclose(samples, samples_expected, atol=1e-4)

    # 定义测试函数 test_MultivariateNormalQMCShapiro
    def test_MultivariateNormalQMCShapiro(self):
        # 测试标准情况

        # 使用指定种子创建随机数生成器
        seed = np.random.default_rng(188960007281846377164494575845971640)
        # 创建 QMC 引擎对象，设置均值、协方差矩阵和种子
        engine = qmc.MultivariateNormalQMC(
            mean=[0, 0], cov=[[1, 0], [0, 1]], seed=seed
        )
        # 生成 256 个随机样本
        samples = engine.random(n=256)
        # 断言样本均值的绝对值小于指定容差
        assert all(np.abs(samples.mean(axis=0)) < 1e-2)
        # 断言样本标准差与 1 的差的绝对值小于指定容差
        assert all(np.abs(samples.std(axis=0) - 1) < 1e-2)
        # 对每一维度进行 Shapiro-Wilk 正态性检验
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            # 断言 p 值大于指定阈值
            assert pval > 0.9
        # 确保样本之间的相关性接近零
        cov = np.cov(samples.transpose())
        # 断言相关系数的绝对值小于指定容差
        assert np.abs(cov[0, 1]) < 1e-2

        # 测试相关性的非零均值情况

        # 使用相同的种子创建新的 QMC 引擎对象
        engine = qmc.MultivariateNormalQMC(
            mean=[1.0, 2.0], cov=[[1.5, 0.5], [0.5, 1.5]], seed=seed
        )
        # 生成 256 个随机样本
        samples = engine.random(n=256)
        # 断言样本均值与指定均值的绝对值小于指定容差
        assert all(np.abs(samples.mean(axis=0) - [1, 2]) < 1e-2)
        # 断言样本标准差与 sqrt(1.5) 的差的绝对值小于指定容差
        assert all(np.abs(samples.std(axis=0) - np.sqrt(1.5)) < 1e-2)
        # 对每一维度进行 Shapiro-Wilk 正态性检验
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            # 断言 p 值大于指定阈值
            assert pval > 0.9
        # 检查样本之间的协方差
        cov = np.cov(samples.transpose())
        # 断言协方差的绝对值与指定值的差小于指定容差
        assert np.abs(cov[0, 1] - 0.5) < 1e-2
    def test_MultivariateNormalQMCShapiroInvTransform(self):
        # 测试多元正态分布的 QMC 方法与逆变换的 Shapiro-Wilk 正态性检验

        # 设置随机种子
        seed = np.random.default_rng(200089821034563288698994840831440331329)
        
        # 创建 QMC 引擎，设定均值和协方差矩阵，使用逆变换方法
        engine = qmc.MultivariateNormalQMC(
            mean=[0, 0], cov=[[1, 0], [0, 1]], seed=seed, inv_transform=True
        )
        
        # 生成随机样本
        samples = engine.random(n=256)
        
        # 断言样本均值接近零
        assert all(np.abs(samples.mean(axis=0)) < 1e-2)
        
        # 断言样本标准差接近 1
        assert all(np.abs(samples.std(axis=0) - 1) < 1e-2)
        
        # 对每一维度进行 Shapiro-Wilk 正态性检验
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            assert pval > 0.9
        
        # 确保样本各维度之间的相关性很小
        cov = np.cov(samples.transpose())
        assert np.abs(cov[0, 1]) < 1e-2

        # 测试相关且有非零均值的情况
        engine = qmc.MultivariateNormalQMC(
            mean=[1.0, 2.0],
            cov=[[1.5, 0.5], [0.5, 1.5]],
            seed=seed,
            inv_transform=True,
        )
        samples = engine.random(n=256)
        
        # 断言样本均值接近设定的均值
        assert all(np.abs(samples.mean(axis=0) - [1, 2]) < 1e-2)
        
        # 断言样本标准差接近 sqrt(1.5)
        assert all(np.abs(samples.std(axis=0) - np.sqrt(1.5)) < 1e-2)
        
        # 对每一维度进行 Shapiro-Wilk 正态性检验
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            assert pval > 0.9
        
        # 检查样本的协方差矩阵
        cov = np.cov(samples.transpose())
        assert np.abs(cov[0, 1] - 0.5) < 1e-2

    def test_MultivariateNormalQMCDegenerate(self):
        # 测试多元正态分布的 QMC 方法对退化分布的处理

        # 设置随机种子
        seed = np.random.default_rng(16320637417581448357869821654290448620)
        
        # 创建 QMC 引擎，设定均值和协方差矩阵
        engine = qmc.MultivariateNormalQMC(
            mean=[0.0, 0.0, 0.0],
            cov=[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0]],
            seed=seed,
        )
        
        # 生成随机样本
        samples = engine.random(n=512)
        
        # 断言样本均值接近零
        assert all(np.abs(samples.mean(axis=0)) < 1e-2)
        
        # 断言第一维样本标准差接近 1
        assert np.abs(np.std(samples[:, 0]) - 1) < 1e-2
        
        # 断言第二维样本标准差接近 1
        assert np.abs(np.std(samples[:, 1]) - 1) < 1e-2
        
        # 断言第三维样本标准差接近 sqrt(2)
        assert np.abs(np.std(samples[:, 2]) - np.sqrt(2)) < 1e-2
        
        # 对每一维度进行 Shapiro-Wilk 正态性检验
        for i in (0, 1, 2):
            _, pval = shapiro(samples[:, i])
            assert pval > 0.8
        
        # 检查样本的协方差矩阵
        cov = np.cov(samples.transpose())
        assert np.abs(cov[0, 1]) < 1e-2
        assert np.abs(cov[0, 2] - 1) < 1e-2
        
        # 检查 X + Y 几乎等于 Z
        assert all(np.abs(samples[:, 0] + samples[:, 1] - samples[:, 2]) < 1e-5)
class TestLloyd:
    def test_lloyd(self):
        # 使用种子值1809831创建一个伪随机数生成器对象
        rng = np.random.RandomState(1809831)
        # 生成一个大小为(128, 2)的均匀分布的随机样本
        sample = rng.uniform(0, 1, size=(128, 2))
        # 计算样本的L1范数作为基准值
        base_l1 = _l1_norm(sample)
        # 计算样本的L2范数作为基准值
        base_l2 = l2_norm(sample)

        # 进行4次迭代
        for _ in range(4):
            # 对样本进行Lloyd算法的重心泰森多边形剖分
            sample_lloyd = _lloyd_centroidal_voronoi_tessellation(
                    sample, maxiter=1,
            )
            # 计算重心泰森多边形剖分后样本的L1范数和L2范数
            curr_l1 = _l1_norm(sample_lloyd)
            curr_l2 = l2_norm(sample_lloyd)

            # 断言新的L1范数和L2范数比基准值更大（更优）
            assert base_l1 < curr_l1
            assert base_l2 < curr_l2

            # 更新基准值为当前L1范数和L2范数
            base_l1 = curr_l1
            base_l2 = curr_l2

            # 更新样本为重心泰森多边形剖分后的样本
            sample = sample_lloyd

    def test_lloyd_non_mutating(self):
        """
        Verify that the input samples are not mutated in place and that they do
        not share memory with the output.
        """
        # 创建原始样本和其深拷贝
        sample_orig = np.array([[0.1, 0.1],
                                [0.1, 0.2],
                                [0.2, 0.1],
                                [0.2, 0.2]])
        sample_copy = sample_orig.copy()
        # 调用Lloyd算法，验证原始样本未被原地修改且与输出不共享内存
        new_sample = _lloyd_centroidal_voronoi_tessellation(
            sample=sample_orig
        )
        # 断言原始样本与深拷贝相等
        assert_allclose(sample_orig, sample_copy)
        # 断言原始样本与输出不共享内存
        assert not np.may_share_memory(sample_orig, new_sample)

    def test_lloyd_errors(self):
        # 检测当输入样本不是2维数组时是否引发值错误异常
        with pytest.raises(ValueError, match=r"`sample` is not a 2D array"):
            sample = [0, 1, 0.5]
            _lloyd_centroidal_voronoi_tessellation(sample)

        # 检测当输入样本维度小于2时是否引发值错误异常
        msg = r"`sample` dimension is not >= 2"
        with pytest.raises(ValueError, match=msg):
            sample = [[0], [0.4], [1]]
            _lloyd_centroidal_voronoi_tessellation(sample)

        # 检测当输入样本不在单位超立方体内时是否引发值错误异常
        msg = r"`sample` is not in unit hypercube"
        with pytest.raises(ValueError, match=msg):
            sample = [[-1.1, 0], [0.1, 0.4], [1, 2]]
            _lloyd_centroidal_voronoi_tessellation(sample)


# 计算样本的L2范数（最小距离）
def l2_norm(sample):
    return distance.pdist(sample).min()
```