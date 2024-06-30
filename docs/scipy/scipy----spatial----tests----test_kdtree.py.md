# `D:\src\scipysrc\scipy\scipy\spatial\tests\test_kdtree.py`

```
# 导入标准库和第三方库模块
import os
from numpy.testing import (assert_equal, assert_array_equal, assert_,
                           assert_almost_equal, assert_array_almost_equal,
                           assert_allclose)
from pytest import raises as assert_raises
import pytest
from platform import python_implementation
import numpy as np
# 导入 scipy.spatial 库中的 KDTree, Rectangle, distance_matrix, cKDTree 等类和函数
from scipy.spatial import KDTree, Rectangle, distance_matrix, cKDTree
# 导入 scipy.spatial._ckdtree 库中的 cKDTreeNode 类
from scipy.spatial._ckdtree import cKDTreeNode
# 导入 scipy.spatial 库中的 minkowski_distance 函数
from scipy.spatial import minkowski_distance

# 导入 itertools 模块
import itertools

# 定义 pytest 的装饰器，参数为 KDTree 和 cKDTree 类
@pytest.fixture(params=[KDTree, cKDTree])
def kdtree_type(request):
    return request.param

# 定义一个装饰器函数 KDTreeTest，用于创建 KDTree 和 cKDTree 的测试用例
def KDTreeTest(kls):
    """Class decorator to create test cases for KDTree and cKDTree
    
    Tests use the class variable ``kdtree_type`` as the tree constructor.
    """
    # 检查类名是否以 _Test 开头
    if not kls.__name__.startswith('_Test'):
        raise RuntimeError("Expected a class name starting with _Test")

    # 遍历 KDTree 和 cKDTree
    for tree in (KDTree, cKDTree):
        # 构造测试名称
        test_name = kls.__name__[1:] + '_' + tree.__name__

        # 检查是否存在重复的测试名称
        if test_name in globals():
            raise RuntimeError("Duplicated test name: " + test_name)

        # 创建一个新的子类，其中 kdtree_type 被定义为当前的 tree 类型
        test_case = type(test_name, (kls,), {'kdtree_type': tree})
        # 将新创建的测试类添加到全局变量中
        globals()[test_name] = test_case
    return kls

# 定义一个计算带周期边界条件的距离函数
def distance_box(a, b, p, boxsize):
    # 计算 a 和 b 之间的差值
    diff = a - b
    # 处理差值超过一半 boxsize 的情况
    diff[diff > 0.5 * boxsize] -= boxsize
    diff[diff < -0.5 * boxsize] += boxsize
    # 计算 Minkowski 距离
    d = minkowski_distance(diff, 0, p)
    return d

# 定义一个一致性测试类 ConsistencyTests
class ConsistencyTests:
    # 定义计算 Minkowski 距离的方法
    def distance(self, a, b, p):
        return minkowski_distance(a, b, p)

    # 测试最近邻搜索的方法
    def test_nearest(self):
        x = self.x
        # 使用 KDTree 查询 x 的最近邻
        d, i = self.kdtree.query(x, 1)
        # 断言距离的平方等于 (x-self.data[i]) 的平方和
        assert_almost_equal(d**2, np.sum((x-self.data[i])**2))
        eps = 1e-8
        # 断言所有点到 x 的距离平方都大于 d**2-eps
        assert_(np.all(np.sum((self.data-x[np.newaxis, :])**2, axis=1) > d**2-eps))

    # 测试多个最近邻搜索的方法
    def test_m_nearest(self):
        x = self.x
        m = self.m
        # 使用 KDTree 查询 x 的前 m 个最近邻
        dd, ii = self.kdtree.query(x, m)
        # 计算最远邻的距离和索引
        d = np.amax(dd)
        i = ii[np.argmax(dd)]
        # 断言距离的平方等于 (x-self.data[i]) 的平方和
        assert_almost_equal(d**2, np.sum((x-self.data[i])**2))
        eps = 1e-8
        # 断言距离小于 d**2+eps 的点的数量等于 m
        assert_equal(
            np.sum(np.sum((self.data-x[np.newaxis, :])**2, axis=1) < d**2+eps),
            m,
        )

    # 测试在给定距离范围内的点的搜索方法
    def test_points_near(self):
        x = self.x
        d = self.d
        # 使用 KDTree 查询距离 x 最近的 self.kdtree.n 个点，且距离小于等于 d
        dd, ii = self.kdtree.query(x, k=self.kdtree.n, distance_upper_bound=d)
        eps = 1e-8
        hits = 0
        # 遍历查询结果
        for near_d, near_i in zip(dd, ii):
            if near_d == np.inf:
                continue
            hits += 1
            # 断言距离的平方等于 (x-self.data[near_i]) 的平方和
            assert_almost_equal(near_d**2, np.sum((x-self.data[near_i])**2))
            # 断言距离 near_d 小于 d+eps
            assert_(near_d < d+eps, f"near_d={near_d:g} should be less than {d:g}")
        # 断言距离小于 d**2+eps 的点的数量等于 hits
        assert_equal(np.sum(self.distance(self.data, x, 2) < d**2+eps), hits)
    # 定义一个测试函数，用于测试在L1范数下的近邻点查找
    def test_points_near_l1(self):
        # 从类属性中获取数据点 x
        x = self.x
        # 从类属性中获取距离阈值 d
        d = self.d
        # 使用 kdtree 对象查询距离在 d 以内的所有点，返回距离数组 dd 和索引数组 ii
        dd, ii = self.kdtree.query(x, k=self.kdtree.n, p=1, distance_upper_bound=d)
        # 设定一个极小的误差值
        eps = 1e-8
        # 初始化命中计数器
        hits = 0
        # 遍历距离数组 dd 和索引数组 ii
        for near_d, near_i in zip(dd, ii):
            # 如果距离为无穷大，则跳过当前循环
            if near_d == np.inf:
                continue
            # 命中数加一，并断言近邻点距离近似于给定的 L1 距离
            hits += 1
            assert_almost_equal(near_d, self.distance(x, self.data[near_i], 1))
            # 断言近邻距离小于 d+eps，否则抛出异常
            assert_(near_d < d+eps, f"near_d={near_d:g} should be less than {d:g}")
        # 断言实际距离小于 d+eps 的数据点个数等于 hits
        assert_equal(np.sum(self.distance(self.data, x, 1) < d+eps), hits)

    # 定义一个测试函数，用于测试在L∞范数下的近邻点查找
    def test_points_near_linf(self):
        # 从类属性中获取数据点 x
        x = self.x
        # 从类属性中获取距离阈值 d
        d = self.d
        # 使用 kdtree 对象查询距离在 d 以内的所有点，返回距离数组 dd 和索引数组 ii
        dd, ii = self.kdtree.query(x, k=self.kdtree.n, p=np.inf, distance_upper_bound=d)
        # 设定一个极小的误差值
        eps = 1e-8
        # 初始化命中计数器
        hits = 0
        # 遍历距离数组 dd 和索引数组 ii
        for near_d, near_i in zip(dd, ii):
            # 如果距离为无穷大，则跳过当前循环
            if near_d == np.inf:
                continue
            # 命中数加一，并断言近邻点距离近似于给定的 L∞ 距离
            hits += 1
            assert_almost_equal(near_d, self.distance(x, self.data[near_i], np.inf))
            # 断言近邻距离小于 d+eps，否则抛出异常
            assert_(near_d < d+eps, f"near_d={near_d:g} should be less than {d:g}")
        # 断言实际距离小于 d+eps 的数据点个数等于 hits
        assert_equal(np.sum(self.distance(self.data, x, np.inf) < d+eps), hits)

    # 定义一个测试函数，用于测试近似最近邻点查找
    def test_approx(self):
        # 从类属性中获取数据点 x 和近邻点数量 k
        x = self.x
        k = self.k
        # 设定一个相对误差 eps
        eps = 0.1
        # 使用 kdtree 对象查询真实的 k 个最近邻点的距离和索引
        d_real, i_real = self.kdtree.query(x, k)
        # 使用 kdtree 对象查询近似的 k 个最近邻点的距离和索引，设置相对误差 eps
        d, i = self.kdtree.query(x, k, eps=eps)
        # 断言近似距离数组 d 中的每个元素都小于等于真实距离数组 d_real 的对应元素乘以 (1+eps)
        assert_(np.all(d <= d_real*(1+eps)))
@KDTreeTest
class _Test_random(ConsistencyTests):
    # 定义一个测试类 _Test_random，继承自 ConsistencyTests 类，并标记为 KDTreeTest
    def setup_method(self):
        # 设置每个测试方法的初始化方法
        self.n = 100
        self.m = 4
        np.random.seed(1234)  # 设置随机数种子为 1234
        self.data = np.random.randn(self.n, self.m)  # 生成一个随机数据集
        self.kdtree = self.kdtree_type(self.data, leafsize=2)  # 使用生成的数据集创建 KD 树对象，leafsize 参数设置为 2
        self.x = np.random.randn(self.m)  # 生成一个 m 维的随机向量 x
        self.d = 0.2  # 设置一个距离阈值 d
        self.k = 10  # 设置近邻数量 k


@KDTreeTest
class _Test_random_far(_Test_random):
    # 定义一个测试类 _Test_random_far，继承自 _Test_random 类，并标记为 KDTreeTest
    def setup_method(self):
        # 设置每个测试方法的初始化方法
        super().setup_method()  # 调用父类的 setup_method 方法，继承父类的初始化设置
        self.x = np.random.randn(self.m) + 10  # 重新生成一个 m 维的随机向量 x，并在每个分量上加上 10


@KDTreeTest
class _Test_small(ConsistencyTests):
    # 定义一个测试类 _Test_small，继承自 ConsistencyTests 类，并标记为 KDTreeTest
    def setup_method(self):
        # 设置每个测试方法的初始化方法
        self.data = np.array([[0, 0, 0],  # 创建一个小数据集，包含 8 个点，每个点有 3 个维度
                              [0, 0, 1],
                              [0, 1, 0],
                              [0, 1, 1],
                              [1, 0, 0],
                              [1, 0, 1],
                              [1, 1, 0],
                              [1, 1, 1]])
        self.kdtree = self.kdtree_type(self.data)  # 使用小数据集创建 KD 树对象
        self.n = self.kdtree.n  # 记录 KD 树中的点数 n
        self.m = self.kdtree.m  # 记录 KD 树中的维度 m
        np.random.seed(1234)  # 设置随机数种子为 1234
        self.x = np.random.randn(3)  # 生成一个 3 维的随机向量 x
        self.d = 0.5  # 设置一个距离阈值 d
        self.k = 4  # 设置近邻数量 k

    def test_nearest(self):
        # 定义一个测试方法 test_nearest，测试 KD 树查询最近点功能
        assert_array_equal(
                self.kdtree.query((0, 0, 0.1), 1),  # 查询距离点 (0, 0, 0.1) 最近的 1 个点
                (0.1, 0))  # 断言查询结果是否与期望相符

    def test_nearest_two(self):
        # 定义一个测试方法 test_nearest_two，测试 KD 树查询最近点功能
        assert_array_equal(
                self.kdtree.query((0, 0, 0.1), 2),  # 查询距离点 (0, 0, 0.1) 最近的 2 个点
                ([0.1, 0.9], [0, 1]))  # 断言查询结果是否与期望相符


@KDTreeTest
class _Test_small_nonleaf(_Test_small):
    # 定义一个测试类 _Test_small_nonleaf，继承自 _Test_small 类，并标记为 KDTreeTest
    def setup_method(self):
        # 设置每个测试方法的初始化方法
        super().setup_method()  # 调用父类的 setup_method 方法，继承父类的初始化设置
        self.kdtree = self.kdtree_type(self.data, leafsize=1)  # 使用小数据集创建 KD 树对象，leafsize 参数设置为 1


class Test_vectorization_KDTree:
    # 定义一个测试类 Test_vectorization_KDTree
    def setup_method(self):
        # 设置每个测试方法的初始化方法
        self.data = np.array([[0, 0, 0],  # 创建一个数据集，包含 8 个点，每个点有 3 个维度
                              [0, 0, 1],
                              [0, 1, 0],
                              [0, 1, 1],
                              [1, 0, 0],
                              [1, 0, 1],
                              [1, 1, 0],
                              [1, 1, 1]])
        self.kdtree = KDTree(self.data)  # 使用数据集创建 KD 树对象

    def test_single_query(self):
        # 定义一个测试方法 test_single_query，测试 KD 树单点查询功能
        d, i = self.kdtree.query(np.array([0, 0, 0]))  # 查询距离点 (0, 0, 0) 最近的点及其索引
        assert_(isinstance(d, float))  # 断言返回的距离值 d 是浮点型
        assert_(np.issubdtype(i, np.signedinteger))  # 断言返回的索引 i 是有符号整数类型

    def test_vectorized_query(self):
        # 定义一个测试方法 test_vectorized_query，测试 KD 树向量化查询功能
        d, i = self.kdtree.query(np.zeros((2, 4, 3)))  # 向 KD 树查询多个点的最近邻点
        assert_equal(np.shape(d), (2, 4))  # 断言返回的距离数组 d 的形状是否为 (2, 4)
        assert_equal(np.shape(i), (2, 4))  # 断言返回的索引数组 i 的形状是否为 (2, 4)

    def test_single_query_multiple_neighbors(self):
        # 定义一个测试方法 test_single_query_multiple_neighbors，测试 KD 树单点查询多个邻居功能
        s = 23
        kk = self.kdtree.n + s  # 计算要查询的邻居点数量
        d, i = self.kdtree.query(np.array([0, 0, 0]), k=kk)  # 查询距离点 (0, 0, 0) 最近的 kk 个点
        assert_equal(np.shape(d), (kk,))  # 断言返回的距离数组 d 的形状是否为 (kk,)
        assert_equal(np.shape(i), (kk,))  # 断言返回的索引数组 i 的形状是否为 (kk,)
        assert_(np.all(~np.isfinite(d[-s:])))  # 断言最后 s 个邻居的距离值是否为无穷大
        assert_(np.all(i[-s:] == self.kdtree.n))  # 断言最后 s 个邻居的索引是否为数据集中的虚拟点索引
    # 定义测试函数，用于测试向量化查询多个邻居的情况
    def test_vectorized_query_multiple_neighbors(self):
        # 设置变量 s 为 23
        s = 23
        # 计算查询邻居数 kk，为 kdtree 中邻居数 n 加上 s
        kk = self.kdtree.n + s
        # 使用 kdtree 对象执行查询，返回距离 d 和索引 i
        d, i = self.kdtree.query(np.zeros((2, 4, 3)), k=kk)
        # 断言距离 d 的形状为 (2, 4, kk)
        assert_equal(np.shape(d), (2, 4, kk))
        # 断言索引 i 的形状为 (2, 4, kk)
        assert_equal(np.shape(i), (2, 4, kk))
        # 断言最后 s 个距离 d 的所有元素都不是有限数
        assert_(np.all(~np.isfinite(d[:, :, -s:])))
        # 断言最后 s 个索引 i 的所有元素等于 kdtree 中的邻居数 n
        assert_(np.all(i[:, :, -s:] == self.kdtree.n))

    # 定义测试函数，用于测试查询中 k 为 None 时引发异常的情况
    def test_query_raises_for_k_none(self):
        # 设置变量 x 为 1.0
        x = 1.0
        # 使用 pytest 检查调用 kdtree 对象的 query 函数时，当 k 为 None 时是否引发 ValueError 异常，
        # 并且异常消息匹配指定的正则表达式模式
        with pytest.raises(ValueError, match="k must be an integer or*"):
            self.kdtree.query(x, k=None)
# 定义一个测试类 Test_vectorization_cKDTree，用于测试 cKDTree 的向量化查询功能
class Test_vectorization_cKDTree:
    
    # 在每个测试方法运行之前设置测试数据和 cKDTree 实例
    def setup_method(self):
        self.data = np.array([[0, 0, 0],
                              [0, 0, 1],
                              [0, 1, 0],
                              [0, 1, 1],
                              [1, 0, 0],
                              [1, 0, 1],
                              [1, 1, 0],
                              [1, 1, 1]])
        self.kdtree = cKDTree(self.data)

    # 测试单点查询功能
    def test_single_query(self):
        d, i = self.kdtree.query([0, 0, 0])
        assert_(isinstance(d, float))  # 确保返回的距离是浮点数类型
        assert_(isinstance(i, int))    # 确保返回的索引是整数类型

    # 测试向量化查询功能
    def test_vectorized_query(self):
        d, i = self.kdtree.query(np.zeros((2, 4, 3)))
        assert_equal(np.shape(d), (2, 4))  # 确保距离数组的形状为 (2, 4)
        assert_equal(np.shape(i), (2, 4))  # 确保索引数组的形状为 (2, 4)

    # 测试向量化查询包含非连续值的情况
    def test_vectorized_query_noncontiguous_values(self):
        np.random.seed(1234)
        qs = np.random.randn(3, 1000).T
        ds, i_s = self.kdtree.query(qs)
        for q, d, i in zip(qs, ds, i_s):
            assert_equal(self.kdtree.query(q), (d, i))  # 确保每个查询的结果正确

    # 测试单点查询多个最近邻的功能
    def test_single_query_multiple_neighbors(self):
        s = 23
        kk = self.kdtree.n+s
        d, i = self.kdtree.query([0, 0, 0], k=kk)
        assert_equal(np.shape(d), (kk,))   # 确保距离数组的形状为 (kk,)
        assert_equal(np.shape(i), (kk,))   # 确保索引数组的形状为 (kk,)
        assert_(np.all(~np.isfinite(d[-s:])))  # 确保最后 s 个距离值不是有限的
        assert_(np.all(i[-s:] == self.kdtree.n))  # 确保最后 s 个索引值为 n

    # 测试向量化查询多个最近邻的功能
    def test_vectorized_query_multiple_neighbors(self):
        s = 23
        kk = self.kdtree.n+s
        d, i = self.kdtree.query(np.zeros((2, 4, 3)), k=kk)
        assert_equal(np.shape(d), (2, 4, kk))  # 确保距离数组的形状为 (2, 4, kk)
        assert_equal(np.shape(i), (2, 4, kk))  # 确保索引数组的形状为 (2, 4, kk)
        assert_(np.all(~np.isfinite(d[:, :, -s:])))  # 确保最后 s 个距离值不是有限的
        assert_(np.all(i[:, :, -s:] == self.kdtree.n))  # 确保最后 s 个索引值为 n

# 定义球形一致性测试类 ball_consistency
class ball_consistency:
    tol = 0.0  # 设置容差值为 0.0

    # 计算 Minkowski 距离的方法
    def distance(self, a, b, p):
        return minkowski_distance(a * 1.0, b * 1.0, p)

    # 测试球内点的方法
    def test_in_ball(self):
        x = np.atleast_2d(self.x)
        d = np.broadcast_to(self.d, x.shape[:-1])
        l = self.T.query_ball_point(x, self.d, p=self.p, eps=self.eps)
        for i, ind in enumerate(l):
            # 计算距离和规范化值
            dist = self.distance(self.data[ind], x[i], self.p) - d[i]*(1.+self.eps)
            norm = self.distance(self.data[ind], x[i], self.p) + d[i]*(1.+self.eps)
            # 确保球内条件满足
            assert_array_equal(dist < self.tol * norm, True)

    # 测试找到所有点的方法
    def test_found_all(self):
        x = np.atleast_2d(self.x)
        d = np.broadcast_to(self.d, x.shape[:-1])
        l = self.T.query_ball_point(x, self.d, p=self.p, eps=self.eps)
        for i, ind in enumerate(l):
            c = np.ones(self.T.n, dtype=bool)
            c[ind] = False
            # 计算距离和规范化值
            dist = self.distance(self.data[c], x[i], self.p) - d[i]/(1.+self.eps)
            norm = self.distance(self.data[c], x[i], self.p) + d[i]/(1.+self.eps)
            # 确保找到所有点的条件满足
            assert_array_equal(dist > -self.tol * norm, True)

# 标记为 KDTreeTest 的随机球测试类，继承自球形一致性测试类 ball_consistency
@KDTreeTest
class _Test_random_ball(ball_consistency):
    pass  # 无需额外的代码，继承了 ball_consistency 的所有方法和属性
    # 定义测试方法的设置方法，用于初始化测试环境
    def setup_method(self):
        # 设置样本数量为100
        n = 100
        # 设置数据维度为4
        m = 4
        # 使用固定种子1234初始化随机数生成器，保证结果可复现性
        np.random.seed(1234)
        # 生成一个大小为(n, m)的随机正态分布数据矩阵，并将其赋给实例变量self.data
        self.data = np.random.randn(n, m)
        # 使用给定的数据创建一个kdtree_type类型的对象self.T，叶子节点大小设置为2
        self.T = self.kdtree_type(self.data, leafsize=2)
        # 生成一个长度为m的随机正态分布向量，并将其赋给实例变量self.x
        self.x = np.random.randn(m)
        # 设置距离度量的参数p为2，代表欧几里得距离
        self.p = 2.
        # 设置近似搜索的容差为0
        self.eps = 0
        # 设置距离的上限为0.2
        self.d = 0.2
@KDTreeTest
class _Test_random_ball_periodic(ball_consistency):
    # _Test_random_ball_periodic 类的定义，继承自 ball_consistency 类
    def distance(self, a, b, p):
        # 定义 distance 方法，计算 a 和 b 之间的距离，使用 p 范数
        return distance_box(a, b, p, 1.0)

    def setup_method(self):
        # 设置测试方法的初始化操作
        n = 10000
        m = 4
        np.random.seed(1234)
        # 生成 n 行 m 列的随机数据
        self.data = np.random.uniform(size=(n, m))
        # 创建 KD 树对象 T，使用 self.data 作为数据，leafsize=2，boxsize=1
        self.T = self.kdtree_type(self.data, leafsize=2, boxsize=1)
        # 初始化 self.x 为 m 维全为 0.1 的数组
        self.x = np.full(m, 0.1)
        self.p = 2.
        self.eps = 0
        self.d = 0.2

    def test_in_ball_outside(self):
        # 测试方法：验证球体内部和外部点的查询结果
        # 查询球体内部距离 self.x + 1.0 小于等于 self.d*(1.+self.eps) 的点列表
        l = self.T.query_ball_point(self.x + 1.0, self.d, p=self.p, eps=self.eps)
        for i in l:
            # 验证距离条件是否成立
            assert_(self.distance(self.data[i], self.x, self.p) <= self.d*(1.+self.eps))
        # 查询球体外部距离 self.x - 1.0 小于等于 self.d*(1.+self.eps) 的点列表
        l = self.T.query_ball_point(self.x - 1.0, self.d, p=self.p, eps=self.eps)
        for i in l:
            # 验证距离条件是否成立
            assert_(self.distance(self.data[i], self.x, self.p) <= self.d*(1.+self.eps))

    def test_found_all_outside(self):
        # 测试方法：验证找到所有球体外部的点
        c = np.ones(self.T.n, dtype=bool)
        # 查询球体内部距离 self.x + 1.0 小于等于 self.d*(1.+self.eps) 的点列表
        l = self.T.query_ball_point(self.x + 1.0, self.d, p=self.p, eps=self.eps)
        c[l] = False
        # 验证所有 c 中为 False 的点，与 self.x 的距离大于等于 self.d/(1.+self.eps)
        assert np.all(
            self.distance(self.data[c], self.x, self.p) >= self.d/(1.+self.eps)
        )

        # 查询球体外部距离 self.x - 1.0 小于等于 self.d*(1.+self.eps) 的点列表
        l = self.T.query_ball_point(self.x - 1.0, self.d, p=self.p, eps=self.eps)
        c[l] = False
        # 验证所有 c 中为 False 的点，与 self.x 的距离大于等于 self.d/(1.+self.eps)
        assert np.all(
            self.distance(self.data[c], self.x, self.p) >= self.d/(1.+self.eps)
        )


@KDTreeTest
class _Test_random_ball_largep_issue9890(ball_consistency):
    # _Test_random_ball_largep_issue9890 类的定义，继承自 ball_consistency 类

    # 允许由于数值问题导致的舍入误差
    tol = 1e-13

    def setup_method(self):
        # 设置测试方法的初始化操作
        n = 1000
        m = 2
        np.random.seed(123)
        # 生成 n 行 m 列的随机整数数据，范围在 [100, 1000)
        self.data = np.random.randint(100, 1000, size=(n, m))
        # 创建 KD 树对象 T，使用 self.data 作为数据
        self.T = self.kdtree_type(self.data)
        self.x = self.data
        self.p = 100
        self.eps = 0
        self.d = 10


@KDTreeTest
class _Test_random_ball_approx(_Test_random_ball):
    # _Test_random_ball_approx 类的定义，继承自 _Test_random_ball 类

    def setup_method(self):
        # 调用父类的 setup_method 方法
        super().setup_method()
        self.eps = 0.1


@KDTreeTest
class _Test_random_ball_approx_periodic(_Test_random_ball):
    # _Test_random_ball_approx_periodic 类的定义，继承自 _Test_random_ball 类

    def setup_method(self):
        # 调用父类的 setup_method 方法
        super().setup_method()
        self.eps = 0.1


@KDTreeTest
class _Test_random_ball_far(_Test_random_ball):
    # _Test_random_ball_far 类的定义，继承自 _Test_random_ball 类

    def setup_method(self):
        # 调用父类的 setup_method 方法
        super().setup_method()
        self.d = 2.


@KDTreeTest
class _Test_random_ball_far_periodic(_Test_random_ball_periodic):
    # _Test_random_ball_far_periodic 类的定义，继承自 _Test_random_ball_periodic 类

    def setup_method(self):
        # 调用父类的 setup_method 方法
        super().setup_method()
        self.d = 2.


@KDTreeTest
class _Test_random_ball_l1(_Test_random_ball):
    # _Test_random_ball_l1 类的定义，继承自 _Test_random_ball 类

    def setup_method(self):
        # 调用父类的 setup_method 方法
        super().setup_method()
        self.p = 1


@KDTreeTest
class _Test_random_ball_linf(_Test_random_ball):
    # _Test_random_ball_linf 类的定义，继承自 _Test_random_ball 类

    def setup_method(self):
        # 调用父类的 setup_method 方法
        super().setup_method()
        self.p = np.inf


def test_random_ball_vectorized(kdtree_type):
    # 测试函数：测试球体向量化查询
    n = 20
    m = 5
    np.random.seed(1234)
    # 创建 KD 树对象 T，使用 np.random.randn(n, m) 作为数据
    T = kdtree_type(np.random.randn(n, m))

    # 查询多个点与球体的关系，期望结果是一个形状为 (2, 3) 的数组
    r = T.query_ball_point(np.random.randn(2, 3, m), 1)
    # 验证查询结果的形状是否符合预期
    assert_equal(r.shape, (2, 3))
    # 验证 r[0, 0] 是否是一个列表
    assert_(isinstance(r[0, 0], list))
# 定义一个测试函数，测试多线程下的 kdtree 查询效果
def test_query_ball_point_multithreading(kdtree_type):
    # 设定随机种子以保证可复现性
    np.random.seed(0)
    # 设定点的数量和维度
    n = 5000
    k = 2
    # 生成随机点集合
    points = np.random.randn(n, k)
    # 创建 kdtree 实例 T
    T = kdtree_type(points)
    
    # 使用不同的线程数执行查询，生成结果列表
    l1 = T.query_ball_point(points, 0.003, workers=1)
    l2 = T.query_ball_point(points, 0.003, workers=64)
    l3 = T.query_ball_point(points, 0.003, workers=-1)
    
    # 对查询结果进行比较和断言
    for i in range(n):
        if l1[i] or l2[i]:
            assert_array_equal(l1[i], l2[i])
    
    for i in range(n):
        if l1[i] or l3[i]:
            assert_array_equal(l1[i], l3[i])


# 定义一个类用于测试两棵 kdtree 之间的一致性
class two_trees_consistency:

    # 定义距离函数，使用 Minkowski 距离
    def distance(self, a, b, p):
        return minkowski_distance(a, b, p)

    # 测试球形区域内是否所有点都被找到
    def test_all_in_ball(self):
        # 查询 T1 中距离 T2 中每个点在距离 d 以内的点
        r = self.T1.query_ball_tree(self.T2, self.d, p=self.p, eps=self.eps)
        # 对查询结果进行遍历和断言
        for i, l in enumerate(r):
            for j in l:
                assert (self.distance(self.data1[i], self.data2[j], self.p)
                        <= self.d*(1.+self.eps))
    
    # 测试是否找到所有符合条件的点
    def test_found_all(self):
        # 查询 T1 中距离 T2 中每个点在距离 d 以内的点
        r = self.T1.query_ball_tree(self.T2, self.d, p=self.p, eps=self.eps)
        # 对查询结果进行遍历和断言
        for i, l in enumerate(r):
            c = np.ones(self.T2.n, dtype=bool)
            c[l] = False
            assert np.all(self.distance(self.data2[c], self.data1[i], self.p)
                          >= self.d/(1.+self.eps))


# 用于测试两个随机 kdtree 的一致性，继承自 two_trees_consistency 类
@KDTreeTest
class _Test_two_random_trees(two_trees_consistency):

    # 初始化方法，设置数据和参数
    def setup_method(self):
        n = 50
        m = 4
        np.random.seed(1234)
        # 生成随机数据集
        self.data1 = np.random.randn(n, m)
        # 创建第一个 kdtree 实例 T1
        self.T1 = self.kdtree_type(self.data1, leafsize=2)
        # 生成随机数据集
        self.data2 = np.random.randn(n, m)
        # 创建第二个 kdtree 实例 T2
        self.T2 = self.kdtree_type(self.data2, leafsize=2)
        # 设定距离参数 p, eps, d
        self.p = 2.
        self.eps = 0
        self.d = 0.2


# 用于测试两个周期性 kdtree 的一致性，继承自 two_trees_consistency 类
@KDTreeTest
class _Test_two_random_trees_periodic(two_trees_consistency):
    
    # 距离函数的重定义，使用 distance_box 函数
    def distance(self, a, b, p):
        return distance_box(a, b, p, 1.0)
    
    # 初始化方法，设置数据和参数
    def setup_method(self):
        n = 50
        m = 4
        np.random.seed(1234)
        # 生成均匀分布的随机数据集
        self.data1 = np.random.uniform(size=(n, m))
        # 创建第一个周期性 kdtree 实例 T1
        self.T1 = self.kdtree_type(self.data1, leafsize=2, boxsize=1.0)
        # 生成均匀分布的随机数据集
        self.data2 = np.random.uniform(size=(n, m))
        # 创建第二个周期性 kdtree 实例 T2
        self.T2 = self.kdtree_type(self.data2, leafsize=2, boxsize=1.0)
        # 设定距离参数 p, eps, d
        self.p = 2.
        self.eps = 0
        self.d = 0.2


# 用于测试两个远距离 kdtree 的一致性，继承自 _Test_two_random_trees 类
@KDTreeTest
class _Test_two_random_trees_far(_Test_two_random_trees):

    # 重载初始化方法，设置距离参数 d
    def setup_method(self):
        super().setup_method()
        self.d = 2


# 用于测试两个周期性、远距离 kdtree 的一致性，继承自 _Test_two_random_trees_periodic 类
@KDTreeTest
class _Test_two_random_trees_far_periodic(_Test_two_random_trees_periodic):

    # 重载初始化方法，设置距离参数 d
    def setup_method(self):
        super().setup_method()
        self.d = 2


# 用于测试使用 Linf 范数的 kdtree 的一致性，继承自 _Test_two_random_trees 类
@KDTreeTest
class _Test_two_random_trees_linf(_Test_two_random_trees):

    # 重载初始化方法，设置距离参数 p 为无穷大
    def setup_method(self):
        super().setup_method()
        self.p = np.inf


# 用于测试使用 Linf 范数的周期性 kdtree 的一致性，继承自 _Test_two_random_trees_periodic 类
@KDTreeTest
class _Test_two_random_trees_linf_periodic(_Test_two_random_trees_periodic):

    # 重载初始化方法，设置距离参数 p 为无穷大
    def setup_method(self):
        super().setup_method()
        self.p = np.inf


# 定义一个用于测试 Rectangle 类的测试类
class Test_rectangle:

    # 初始化方法，创建一个 Rectangle 对象
    def setup_method(self):
        self.rect = Rectangle([0, 0], [1, 1])
    # 测试最小距离点函数在矩形内部的情况，预期返回接近于 0 的值
    def test_min_inside(self):
        assert_almost_equal(self.rect.min_distance_point([0.5, 0.5]), 0)

    # 测试最小距离点函数在矩形一侧的情况，预期返回接近于 0.5 的值
    def test_min_one_side(self):
        assert_almost_equal(self.rect.min_distance_point([0.5, 1.5]), 0.5)

    # 测试最小距离点函数在矩形两侧的情况，预期返回接近于 sqrt(2) 的值
    def test_min_two_sides(self):
        assert_almost_equal(self.rect.min_distance_point([2, 2]), np.sqrt(2))

    # 测试最大距离点函数在矩形内部的情况，预期返回接近于 1/√2 的值
    def test_max_inside(self):
        assert_almost_equal(self.rect.max_distance_point([0.5, 0.5]), 1/np.sqrt(2))

    # 测试最大距离点函数在矩形一侧的情况，预期返回接近于 hypot(0.5, 1.5) 的值
    def test_max_one_side(self):
        assert_almost_equal(self.rect.max_distance_point([0.5, 1.5]),
                            np.hypot(0.5, 1.5))

    # 测试最大距离点函数在矩形两侧的情况，预期返回接近于 2*sqrt(2) 的值
    def test_max_two_sides(self):
        assert_almost_equal(self.rect.max_distance_point([2, 2]), 2*np.sqrt(2))

    # 测试矩形切分函数，按给定的维度和切分值分割矩形，验证切分后的最大和最小值数组是否符合预期
    def test_split(self):
        less, greater = self.rect.split(0, 0.1)
        assert_array_equal(less.maxes, [0.1, 1])
        assert_array_equal(less.mins, [0, 0])
        assert_array_equal(greater.maxes, [1, 1])
        assert_array_equal(greater.mins, [0.1, 0])
def test_distance_l2():
    # 测试使用 Minkowski 距离计算两点之间的欧几里得距离
    assert_almost_equal(minkowski_distance([0, 0], [1, 1], 2), np.sqrt(2))


def test_distance_l1():
    # 测试使用 Minkowski 距离计算两点之间的曼哈顿距离
    assert_almost_equal(minkowski_distance([0, 0], [1, 1], 1), 2)


def test_distance_linf():
    # 测试使用 Minkowski 距离计算两点之间的切比雪夫距离
    assert_almost_equal(minkowski_distance([0, 0], [1, 1], np.inf), 1)


def test_distance_vectorization():
    # 测试 Minkowski 距离的向量化计算
    np.random.seed(1234)
    x = np.random.randn(10, 1, 3)
    y = np.random.randn(1, 7, 3)
    assert_equal(minkowski_distance(x, y).shape, (10, 7))


class count_neighbors_consistency:
    def test_one_radius(self):
        # 测试在给定半径下邻居数量的一致性
        r = 0.2
        assert_equal(self.T1.count_neighbors(self.T2, r),
                np.sum([len(l) for l in self.T1.query_ball_tree(self.T2, r)]))

    def test_large_radius(self):
        # 测试在大半径下邻居数量的一致性
        r = 1000
        assert_equal(self.T1.count_neighbors(self.T2, r),
                np.sum([len(l) for l in self.T1.query_ball_tree(self.T2, r)]))

    def test_multiple_radius(self):
        # 测试多个半径值下邻居数量的一致性
        rs = np.exp(np.linspace(np.log(0.01), np.log(10), 3))
        results = self.T1.count_neighbors(self.T2, rs)
        assert_(np.all(np.diff(results) >= 0))
        for r, result in zip(rs, results):
            assert_equal(self.T1.count_neighbors(self.T2, r), result)


@KDTreeTest
class _Test_count_neighbors(count_neighbors_consistency):
    def setup_method(self):
        # 设置测试方法的初始化条件
        n = 50
        m = 2
        np.random.seed(1234)
        self.T1 = self.kdtree_type(np.random.randn(n, m), leafsize=2)
        self.T2 = self.kdtree_type(np.random.randn(n, m), leafsize=2)


class sparse_distance_matrix_consistency:

    def distance(self, a, b, p):
        # 计算使用 Minkowski 距离计算两点之间的距离
        return minkowski_distance(a, b, p)

    def test_consistency_with_neighbors(self):
        # 测试稀疏距离矩阵与邻居一致性
        M = self.T1.sparse_distance_matrix(self.T2, self.r)
        r = self.T1.query_ball_tree(self.T2, self.r)
        for i, l in enumerate(r):
            for j in l:
                assert_almost_equal(
                    M[i, j],
                    self.distance(self.T1.data[i], self.T2.data[j], self.p),
                    decimal=14
                )
        for ((i, j), d) in M.items():
            assert_(j in r[i])

    def test_zero_distance(self):
        # 测试稀疏距离矩阵对角线为零的情况
        # raises an exception for bug 870 (FIXME: Does it?)
        self.T1.sparse_distance_matrix(self.T1, self.r)

    def test_consistency(self):
        # 测试稀疏距离矩阵与密集距离矩阵的一致性
        M1 = self.T1.sparse_distance_matrix(self.T2, self.r)
        expected = distance_matrix(self.T1.data, self.T2.data)
        expected[expected > self.r] = 0
        assert_array_almost_equal(M1.toarray(), expected, decimal=14)

    def test_against_logic_error_regression(self):
        # 回归测试 gh-5077 的逻辑错误
        np.random.seed(0)
        too_many = np.array(np.random.randn(18, 2), dtype=int)
        tree = self.kdtree_type(
            too_many, balanced_tree=False, compact_nodes=False)
        d = tree.sparse_distance_matrix(tree, 3).toarray()
        assert_array_almost_equal(d, d.T, decimal=14)
    # 定义测试方法：检查稀疏距离矩阵的返回类型是否正确
    def test_ckdtree_return_types(self):
        # 创建一个全零矩阵作为参考
        ref = np.zeros((self.n, self.n))
        # 使用暴力法计算距离矩阵的参考值
        for i in range(self.n):
            for j in range(self.n):
                # 计算向量差并计算平方和
                v = self.data1[i, :] - self.data2[j, :]
                ref[i, j] = np.dot(v, v)
        # 对参考值进行平方根处理
        ref = np.sqrt(ref)
        # 将大于阈值的距离设为零
        ref[ref > self.r] = 0.
        
        # 测试返回类型为 'dict' 的情况
        dist = np.zeros((self.n, self.n))
        # 调用稀疏距离矩阵计算方法，返回类型为字典
        r = self.T1.sparse_distance_matrix(self.T2, self.r, output_type='dict')
        # 将字典中的数据填充到距离矩阵中
        for i, j in r.keys():
            dist[i, j] = r[(i, j)]
        # 断言计算得到的距离矩阵与参考值矩阵的近似相等性
        assert_array_almost_equal(ref, dist, decimal=14)
        
        # 测试返回类型为 'ndarray' 的情况
        dist = np.zeros((self.n, self.n))
        # 调用稀疏距离矩阵计算方法，返回类型为 ndarray
        r = self.T1.sparse_distance_matrix(self.T2, self.r,
            output_type='ndarray')
        # 将 ndarray 中的数据填充到距离矩阵中
        for k in range(r.shape[0]):
            i = r['i'][k]
            j = r['j'][k]
            v = r['v'][k]
            dist[i, j] = v
        # 断言计算得到的距离矩阵与参考值矩阵的近似相等性
        assert_array_almost_equal(ref, dist, decimal=14)
        
        # 测试返回类型为 'dok_matrix' 的情况
        # 调用稀疏距离矩阵计算方法，返回类型为 dok_matrix
        r = self.T1.sparse_distance_matrix(self.T2, self.r,
            output_type='dok_matrix')
        # 将 dok_matrix 转换为 ndarray 并断言其与参考值矩阵的近似相等性
        assert_array_almost_equal(ref, r.toarray(), decimal=14)
        
        # 测试返回类型为 'coo_matrix' 的情况
        # 调用稀疏距离矩阵计算方法，返回类型为 coo_matrix
        r = self.T1.sparse_distance_matrix(self.T2, self.r,
            output_type='coo_matrix')
        # 将 coo_matrix 转换为 ndarray 并断言其与参考值矩阵的近似相等性
        assert_array_almost_equal(ref, r.toarray(), decimal=14)
# 定义一个测试类，继承自sparse_distance_matrix_consistency，用于测试KD树的稀疏距离矩阵一致性
@KDTreeTest
class _Test_sparse_distance_matrix(sparse_distance_matrix_consistency):
    # 设置测试方法的初始化
    def setup_method(self):
        n = 50
        m = 4
        np.random.seed(1234)
        # 生成随机数据
        data1 = np.random.randn(n, m)
        data2 = np.random.randn(n, m)
        # 创建KD树对象T1和T2，分别用data1和data2初始化，设置叶子大小为2
        self.T1 = self.kdtree_type(data1, leafsize=2)
        self.T2 = self.kdtree_type(data2, leafsize=2)
        self.r = 0.5  # 设置距离阈值r为0.5
        self.p = 2    # 设置闵可夫斯基距离的p值为2
        self.data1 = data1  # 存储data1
        self.data2 = data2  # 存储data2
        self.n = n    # 存储数据行数n
        self.m = m    # 存储数据列数m


# 测试距离矩阵函数
def test_distance_matrix():
    m = 10
    n = 11
    k = 4
    np.random.seed(1234)
    # 生成随机数据集xs和ys，每个数据集包含m或n个k维向量
    xs = np.random.randn(m, k)
    ys = np.random.randn(n, k)
    # 计算xs和ys之间的距离矩阵ds
    ds = distance_matrix(xs, ys)
    # 断言距离矩阵ds的形状为(m, n)
    assert_equal(ds.shape, (m, n))
    # 遍历距离矩阵ds的每个元素，断言计算的闵可夫斯基距离与ds中的距离一致
    for i in range(m):
        for j in range(n):
            assert_almost_equal(minkowski_distance(xs[i], ys[j]), ds[i, j])


# 测试带循环的距离矩阵函数
def test_distance_matrix_looping():
    m = 10
    n = 11
    k = 4
    np.random.seed(1234)
    # 生成随机数据集xs和ys，每个数据集包含m或n个k维向量
    xs = np.random.randn(m, k)
    ys = np.random.randn(n, k)
    # 计算xs和ys之间的距离矩阵ds
    ds = distance_matrix(xs, ys)
    # 计算带有阈值的距离矩阵dsl
    dsl = distance_matrix(xs, ys, threshold=1)
    # 断言两个距离矩阵ds和dsl相等
    assert_equal(ds, dsl)


# 检查单棵树的查询函数
def check_onetree_query(T, d):
    # 使用树T进行球形查询，找到距离小于d的所有点的索引对
    r = T.query_ball_tree(T, d)
    s = set()
    for i, l in enumerate(r):
        for j in l:
            if i < j:
                s.add((i, j))
    # 断言球形查询的结果与T的查询对结果相等
    assert_(s == T.query_pairs(d))


# 测试单棵树的查询函数
def test_onetree_query(kdtree_type):
    np.random.seed(0)
    n = 50
    k = 4
    points = np.random.randn(n, k)
    T = kdtree_type(points)
    # 使用check_onetree_query函数测试树T，查询距离阈值为0.1
    check_onetree_query(T, 0.1)

    points = np.random.randn(3*n, k)
    points[:n] *= 0.001
    points[n:2*n] += 2
    T = kdtree_type(points)
    # 使用check_onetree_query函数测试树T，分别查询距离阈值为0.1、0.001、0.00001和1e-6
    check_onetree_query(T, 0.1)
    check_onetree_query(T, 0.001)
    check_onetree_query(T, 0.00001)
    check_onetree_query(T, 1e-6)


# 测试单节点查询对函数
def test_query_pairs_single_node(kdtree_type):
    # 创建仅包含一个节点的树tree
    tree = kdtree_type([[0, 1]])
    # 断言树tree的query_pairs方法返回空集合
    assert_equal(tree.query_pairs(0.5), set())


# 测试KD树查询对函数
def test_kdtree_query_pairs(kdtree_type):
    np.random.seed(0)
    n = 50
    k = 2
    r = 0.1
    r2 = r**2
    # 生成随机数据点集points，每个点有k维度
    points = np.random.randn(n, k)
    T = kdtree_type(points)
    # 使用暴力方法brute计算距离小于r的所有点对
    brute = set()
    for i in range(n):
        for j in range(i+1, n):
            v = points[i, :] - points[j, :]
            if np.dot(v, v) <= r2:
                brute.add((i, j))
    l0 = sorted(brute)  # 对brute集合排序得到列表l0
    # 测试默认返回类型的query_pairs方法
    s = T.query_pairs(r)
    l1 = sorted(s)
    # 断言计算得到的结果与暴力计算结果一致
    assert_array_equal(l0, l1)
    # 测试返回类型为'set'的query_pairs方法
    s = T.query_pairs(r, output_type='set')
    l1 = sorted(s)
    # 断言计算得到的结果与暴力计算结果一致
    assert_array_equal(l0, l1)
    # 测试返回类型为'ndarray'的query_pairs方法
    s = set()
    arr = T.query_pairs(r, output_type='ndarray')
    for i in range(arr.shape[0]):
        s.add((int(arr[i, 0]), int(arr[i, 1])))
    l2 = sorted(s)
    # 断言计算得到的结果与暴力计算结果一致
    assert_array_equal(l0, l2)


# 测试带eps的查询对函数
def test_query_pairs_eps(kdtree_type):
    spacing = np.sqrt(2)
    # 使用不规则间距生成x_range和y_range
    x_range = np.linspace(0, 3 * spacing, 4)
    y_range = np.linspace(0, 3 * spacing, 4)
    # 创建一个包含所有 (xi, yi) 对的列表，其中 xi 属于 x_range，yi 属于 y_range
    xy_array = [(xi, yi) for xi in x_range for yi in y_range]
    # 使用 xy_array 构建一个 KD 树
    tree = kdtree_type(xy_array)
    # 查询 KD 树中所有距离小于 spacing 且误差不超过 0.1 的点对
    pairs_eps = tree.query_pairs(r=spacing, eps=.1)
    # 由于浮点数舍入的影响，带有 eps 的查询结果是 24，没有 eps 的查询结果是 16
    pairs = tree.query_pairs(r=spacing * 1.01)
    # 使用断言检查两个查询结果是否相等
    assert_equal(pairs, pairs_eps)
# 测试球点整数的回归测试，针对 #1373。
def test_ball_point_ints(kdtree_type):
    # 创建一个 2D 网格，x 和 y 分别是 0 到 3 的整数坐标点
    x, y = np.mgrid[0:4, 0:4]
    # 将网格坐标点展平并组成一个点列表
    points = list(zip(x.ravel(), y.ravel()))
    # 使用给定的 kdtree_type 构建 KD 树
    tree = kdtree_type(points)
    # 断言查询球形范围内的点 (2, 0) 半径为 1 的结果是否与预期相符
    assert_equal(sorted([4, 8, 9, 12]),
                 sorted(tree.query_ball_point((2, 0), 1)))
    # 将点坐标转换为浮点数数组
    points = np.asarray(points, dtype=float)
    # 使用浮点数数组构建 KD 树
    tree = kdtree_type(points)
    # 再次断言查询球形范围内的点 (2, 0) 半径为 1 的结果是否与预期相符
    assert_equal(sorted([4, 8, 9, 12]),
                 sorted(tree.query_ball_point((2, 0), 1)))


# KD 树节点比较的回归测试：在 0.12 版本中，与 Python 3 的兼容性问题。
def test_kdtree_comparisons():
    # 创建包含三个 KD 树节点的列表
    nodes = [KDTree.node() for _ in range(3)]
    # 断言反转列表后的节点排序结果是否与原列表排序结果相同
    assert_equal(sorted(nodes), sorted(nodes[::-1]))


# KD 树不同构建模式的回归测试：检查不同构建模式是否会产生类似的查询结果
def test_kdtree_build_modes(kdtree_type):
    # 设定随机种子为 0
    np.random.seed(0)
    n = 5000  # 点的数量
    k = 4     # 点的维度
    # 生成服从标准正态分布的随机点集合
    points = np.random.randn(n, k)
    # 测试不同构建模式下的 KD 树对相同点集的查询结果是否一致
    T1 = kdtree_type(points).query(points, k=5)[-1]
    T2 = kdtree_type(points, compact_nodes=False).query(points, k=5)[-1]
    T3 = kdtree_type(points, balanced_tree=False).query(points, k=5)[-1]
    T4 = kdtree_type(points, compact_nodes=False,
                     balanced_tree=False).query(points, k=5)[-1]
    assert_array_equal(T1, T2)
    assert_array_equal(T1, T3)
    assert_array_equal(T1, T4)


# 测试 KD 树是否可以序列化
def test_kdtree_pickle(kdtree_type):
    # 导入 pickle 模块
    import pickle
    # 设定随机种子为 0
    np.random.seed(0)
    n = 50   # 点的数量
    k = 4    # 点的维度
    # 生成服从标准正态分布的随机点集合
    points = np.random.randn(n, k)
    # 构建 KD 树
    T1 = kdtree_type(points)
    # 将 KD 树对象序列化为字节流
    tmp = pickle.dumps(T1)
    # 从序列化的字节流中恢复 KD 树对象
    T2 = pickle.loads(tmp)
    # 对原始点集和反序列化后的 KD 树对象进行查询操作，并比较结果是否一致
    T1 = T1.query(points, k=5)[-1]
    T2 = T2.query(points, k=5)[-1]
    assert_array_equal(T1, T2)


# 测试带有盒子大小参数的 KD 树是否可以序列化
def test_kdtree_pickle_boxsize(kdtree_type):
    # 导入 pickle 模块
    import pickle
    # 设定随机种子为 0
    np.random.seed(0)
    n = 50   # 点的数量
    k = 4    # 点的维度
    # 生成在 [0, 1) 范围内均匀分布的随机点集合
    points = np.random.uniform(size=(n, k))
    # 构建带有盒子大小参数的 KD 树
    T1 = kdtree_type(points, boxsize=1.0)
    # 将 KD 树对象序列化为字节流
    tmp = pickle.dumps(T1)
    # 从序列化的字节流中恢复 KD 树对象
    T2 = pickle.loads(tmp)
    # 对原始点集和反序列化后的 KD 树对象进行查询操作，并比较结果是否一致
    T1 = T1.query(points, k=5)[-1]
    T2 = T2.query(points, k=5)[-1]
    assert_array_equal(T1, T2)


# 测试复制数据时是否使 KD 树对数据修改不敏感
def test_kdtree_copy_data(kdtree_type):
    # 设定随机种子为 0
    np.random.seed(0)
    n = 5000  # 点的数量
    k = 4     # 点的维度
    # 生成服从标准正态分布的随机点集合
    points = np.random.randn(n, k)
    # 使用 copy_data=True 构建 KD 树
    T = kdtree_type(points, copy_data=True)
    # 复制原始点集合
    q = points.copy()
    # 对 KD 树分别使用原始点集和修改后的点集进行查询操作，并比较结果是否一致
    T1 = T.query(q, k=5)[-1]
    points[...] = np.random.randn(n, k)
    T2 = T.query(q, k=5)[-1]
    assert_array_equal(T1, T2)


# 测试并行查询是否生成正确的结果
def test_ckdtree_parallel(kdtree_type, monkeypatch):
    # 设定随机种子为 0
    np.random.seed(0)
    n = 5000  # 点的数量
    k = 4     # 点的维度
    # 生成服从标准正态分布的随机点集合
    points = np.random.randn(n, k)
    # 构建 KD 树
    T = kdtree_type(points)
    # 使用不同的并行工作线程数进行查询，并比较结果是否一致
    T1 = T.query(points, k=5, workers=64)[-1]
    T2 = T.query(points, k=5, workers=-1)[-1]
    T3 = T.query(points, k=5)[-1]
    assert_array_equal(T1, T2)
    assert_array_equal(T1, T3)
    # 用 monkeypatch 设置 os.cpu_count 函数的返回值为 None
    monkeypatch.setattr(os, 'cpu_count', lambda: None)
    # 使用 pytest 模块中的 `raises` 方法来测试是否会抛出 NotImplementedError 异常，并检查异常消息是否包含特定字符串
    with pytest.raises(NotImplementedError, match="Cannot determine the"):
        # 调用 T 对象的 query 方法，传入 points 和 1 作为参数，同时指定 workers 参数为 -1
        T.query(points, 1, workers=-1)
# 测试函数，验证 cKDTree 中节点能否正确从 Python 中查看。
# 此测试还会对 cKDTree 中的每个节点进行健全性检查，从而验证 kd 树的内部结构。
def test_ckdtree_view():
    # 设置随机种子以确保可重复性
    np.random.seed(0)
    # 定义点的数量和维度
    n = 100
    k = 4
    # 生成随机点集
    points = np.random.randn(n, k)
    # 创建 cKDTree 对象
    kdtree = cKDTree(points)

    # 递归遍历整个 kd 树并对每个节点进行健全性检查
    def recurse_tree(n):
        # 断言节点类型为 cKDTreeNode
        assert_(isinstance(n, cKDTreeNode))
        if n.split_dim == -1:
            # 对叶子节点进行健全性检查
            assert_(n.lesser is None)
            assert_(n.greater is None)
            assert_(n.indices.shape[0] <= kdtree.leafsize)
        else:
            # 递归遍历左右子节点
            recurse_tree(n.lesser)
            recurse_tree(n.greater)
            # 检查分裂维度的左右子节点数据的合理性
            x = n.lesser.data_points[:, n.split_dim]
            y = n.greater.data_points[:, n.split_dim]
            assert_(x.max() < y.min())

    recurse_tree(kdtree.tree)

    # 检查索引是否被正确检索
    n = kdtree.tree
    assert_array_equal(np.sort(n.indices), range(100))

    # 检查数据点是否被正确检索
    assert_array_equal(kdtree.data[n.indices, :], n.data_points)

# KDTree 专用于双精度点类型，因此不需要创建对应 test_ball_point_ints() 的单元测试

# 测试函数，检查 kdtree 周期边界
def test_kdtree_list_k(kdtree_type):
    n = 200
    m = 2
    klist = [1, 2, 3]
    kint = 3

    np.random.seed(1234)
    data = np.random.uniform(size=(n, m))
    # 创建 kdtree 对象
    kdtree = kdtree_type(data, leafsize=1)

    # 检查 arange(1, k+1) 和 k 之间的一致性
    dd, ii = kdtree.query(data, klist)
    dd1, ii1 = kdtree.query(data, kint)
    assert_equal(dd, dd1)
    assert_equal(ii, ii1)

    # 检查跳过一个元素后的情况
    klist = np.array([1, 3])
    kint = 3
    dd, ii = kdtree.query(data, kint)
    dd1, ii1 = kdtree.query(data, klist)
    assert_equal(dd1, dd[..., klist - 1])
    assert_equal(ii1, ii[..., klist - 1])

    # 检查 k == 1 的特殊情况和 k == [1] 的非特殊情况
    dd, ii = kdtree.query(data, 1)
    dd1, ii1 = kdtree.query(data, [1])
    assert_equal(len(dd.shape), 1)
    assert_equal(len(dd1.shape), 2)
    assert_equal(dd, np.ravel(dd1))
    assert_equal(ii, np.ravel(ii1))

@pytest.mark.fail_slow(10)
# 测试函数，检查 ckdtree 周期边界
def test_kdtree_box(kdtree_type):
    n = 2000
    m = 3
    k = 3
    np.random.seed(1234)
    data = np.random.uniform(size=(n, m))
    # 创建 kdtree 对象，设置 leafsize 和 boxsize
    kdtree = kdtree_type(data, leafsize=1, boxsize=1.0)

    # 使用标准 Python KDTree 模拟周期边界框
    kdtree2 = kdtree_type(data, leafsize=1)
    # 对于给定的每个距离度量参数 p，执行以下操作：

    # 使用 KD 树查询数据集 data 中每个点的最近 k 个邻居和它们的距离，返回距离和邻居索引
    dd, ii = kdtree.query(data, k, p=p)

    # 检查将数据集 data 中的每个点的每个坐标增加 1.0 后的查询结果是否与原始结果几乎相等
    dd1, ii1 = kdtree.query(data + 1.0, k, p=p)
    assert_almost_equal(dd, dd1)
    assert_equal(ii, ii1)

    # 检查将数据集 data 中的每个点的每个坐标减少 1.0 后的查询结果是否与原始结果几乎相等
    dd1, ii1 = kdtree.query(data - 1.0, k, p=p)
    assert_almost_equal(dd, dd1)
    assert_equal(ii, ii1)

    # 使用周期性边界条件 simulate_periodic_box 函数模拟在周期性盒子中查询数据的结果
    # 查询结果与原始数据集 data 在距离度量参数 p 下的查询结果进行比较
    dd2, ii2 = simulate_periodic_box(kdtree2, data, k, boxsize=1.0, p=p)
    assert_almost_equal(dd, dd2)
    assert_equal(ii, ii2)
# 测试函数，用于检查在非周期性边界的条件下，CKD 树的行为模拟
def test_kdtree_box_0boxsize(kdtree_type):
    # 设置数据点的数量
    n = 2000
    # 设置数据的维度
    m = 2
    # 设置最近邻居的数量
    k = 3
    # 设置随机数生成的种子，确保结果可重复
    np.random.seed(1234)
    # 生成一个随机的数据集
    data = np.random.uniform(size=(n, m))
    # 创建一个 CKD 树对象，模拟非周期性边界，设置叶子节点大小为 1，盒子大小为 0.0
    kdtree = kdtree_type(data, leafsize=1, boxsize=0.0)

    # 使用标准的 Python KD 树来模拟周期性边界
    kdtree2 = kdtree_type(data, leafsize=1)

    # 对每种距离度量 p 进行迭代
    for p in [1, 2, np.inf]:
        # 查询 CKD 树，返回每个点的最近 k 个距离和对应的索引
        dd, ii = kdtree.query(data, k, p=p)

        # 查询标准 KD 树，返回每个点的最近 k 个距离和对应的索引
        dd1, ii1 = kdtree2.query(data, k, p=p)
        # 断言 CKD 树和标准 KD 树的查询结果近似相等
        assert_almost_equal(dd, dd1)
        assert_equal(ii, ii1)

# 测试函数，用于检查 CKD 树的盒子大小的上界情况
def test_kdtree_box_upper_bounds(kdtree_type):
    # 创建一个简单的二维数据集，y 方向上加 10
    data = np.linspace(0, 2, 10).reshape(-1, 2)
    data[:, 1] += 10
    # 使用 pytest 检测是否会抛出 ValueError 异常，盒子大小设为 1.0
    with pytest.raises(ValueError):
        kdtree_type(data, leafsize=1, boxsize=1.0)
    # 使用 pytest 检测是否会抛出 ValueError 异常，盒子大小设为 (0.0, 2.0)
    with pytest.raises(ValueError):
        kdtree_type(data, leafsize=1, boxsize=(0.0, 2.0))
    # 创建一个 CKD 树对象，盒子大小设为 (2.0, 0.0)，跳过一个维度
    kdtree_type(data, leafsize=1, boxsize=(2.0, 0.0))

# 测试函数，用于检查 CKD 树的盒子大小的下界情况
def test_kdtree_box_lower_bounds(kdtree_type):
    # 创建一个简单的一维数据集，数据范围在 [-1, 1]
    data = np.linspace(-1, 1, 10)
    # 使用 assert_raises 检测是否会抛出 ValueError 异常，盒子大小设为 1.0
    assert_raises(ValueError, kdtree_type, data, leafsize=1, boxsize=1.0)

# 模拟周期性边界的函数，用于在给定盒子大小和距离度量的条件下，查询 CKD 树
def simulate_periodic_box(kdtree, data, k, boxsize, p):
    # 初始化空列表存储查询结果的距离和索引
    dd = []
    ii = []
    # 生成一个用于模拟周期性边界的格点
    x = np.arange(3 ** data.shape[1])
    nn = np.array(np.unravel_index(x, [3] * data.shape[1])).T
    nn = nn - 1.0
    # 对每个格点进行迭代
    for n in nn:
        # 创建偏移后的图像数据
        image = data + n * 1.0 * boxsize
        # 查询 CKD 树，返回每个点的最近 k 个距离和对应的索引
        dd2, ii2 = kdtree.query(image, k, p=p)
        # 将查询结果按照 k 重构成矩阵形式
        dd2 = dd2.reshape(-1, k)
        ii2 = ii2.reshape(-1, k)
        # 将结果添加到 dd 和 ii 列表中
        dd.append(dd2)
        ii.append(ii2)
    # 将 dd 和 ii 列表按列连接成 numpy 数组
    dd = np.concatenate(dd, axis=-1)
    ii = np.concatenate(ii, axis=-1)

    # 创建一个结构化数组存储最终结果
    result = np.empty([len(data), len(nn) * k], dtype=[
            ('ii', 'i8'),
            ('dd', 'f8')])
    result['ii'][:] = ii
    result['dd'][:] = dd
    # 根据距离 'dd' 对结果数组进行排序
    result.sort(order='dd')
    # 返回排序后的最近 k 个距离和索引
    return result['dd'][:, :k], result['ii'][:, :k]

# 标记测试，如果运行环境为 PyPy，则跳过测试
@pytest.mark.skipif(python_implementation() == 'PyPy',
                    reason="Fails on PyPy CI runs. See #9507")
def test_ckdtree_memuse():
    # 单元测试，用于检查内存使用情况，适用于 gh-5630 的适配

    try:
        import resource
    except ImportError:
        # Windows 环境下没有 resource 模块，直接返回
        return
    # 生成一个简单的二维数据集
    dx, dy = 0.05, 0.05
    y, x = np.mgrid[slice(1, 5 + dy, dy),
                    slice(1, 5 + dx, dx)]
    z = np.sin(x)**10 + np.cos(10 + y*x) * np.cos(x)
    z_copy = np.empty_like(z)
    z_copy[:] = z
    # 在 z_copy 中随机选择几个位置放置 FILLVAL
    FILLVAL = 99.
    mask = np.random.randint(0, z.size, np.random.randint(50) + 5)
    z_copy.flat[mask] = FILLVAL
    # 找到非 FILLVAL 值的索引
    igood = np.vstack(np.nonzero(x != FILLVAL)).T
    ibad = np.vstack(np.nonzero(x == FILLVAL)).T
    # 获取当前进程的最大内存使用情况
    mem_use = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # 构建 CKD 树对象并进行初始化
    for i in range(10):
        tree = cKDTree(igood)
    # 计数构建和查询 CKD 树时的内存泄漏次数
    num_leaks = 0
    # 循环执行100次以下操作，检测内存使用情况是否存在泄漏
    for i in range(100):
        # 获取当前进程的最大内存使用情况
        mem_use = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # 使用 igood 数据创建 cKDTree 对象
        tree = cKDTree(igood)
        # 查询 cKDTree 中与 ibad 最接近的 4 个点的距离和索引，使用 L2 范数
        dist, iquery = tree.query(ibad, k=4, p=2)
        # 获取执行查询后的最大内存使用情况
        new_mem_use = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # 如果新的内存使用超过了旧的内存使用，则计数泄漏次数
        if new_mem_use > mem_use:
            num_leaks += 1
    # 断言：泄漏次数应该小于 10，理想情况下为零，但可能存在误差导致的泄漏
    assert_(num_leaks < 10)
def test_kdtree_weights(kdtree_type):
    # 创建一个包含四个点的一维数据集
    data = np.linspace(0, 1, 4).reshape(-1, 1)
    # 使用给定的 kdtree_type 构建 KD 树
    tree1 = kdtree_type(data, leafsize=1)
    # 创建权重数组，每个点权重为1
    weights = np.ones(len(data), dtype='f4')

    # 测试 KD 树的权重构建方法
    nw = tree1._build_weights(weights)
    # 断言权重构建结果与预期值相等
    assert_array_equal(nw, [4, 2, 1, 1, 2, 1, 1])

    # 断言当权重数组长度少于数据点数量时抛出 ValueError 异常
    assert_raises(ValueError, tree1._build_weights, weights[:-1])

    for i in range(10):
        # 由于权重是均匀的，这些结果应该一致：
        c1 = tree1.count_neighbors(tree1, np.linspace(0, 10, i))
        c2 = tree1.count_neighbors(tree1, np.linspace(0, 10, i),
                weights=(weights, weights))
        c3 = tree1.count_neighbors(tree1, np.linspace(0, 10, i),
                weights=(weights, None))
        c4 = tree1.count_neighbors(tree1, np.linspace(0, 10, i),
                weights=(None, weights))
        tree1.count_neighbors(tree1, np.linspace(0, 10, i),
                weights=weights)

        # 断言不同权重设置下的邻居计数结果一致
        assert_array_equal(c1, c2)
        assert_array_equal(c1, c3)
        assert_array_equal(c1, c4)

    for i in range(len(data)):
        # 通过将权重设置为0来测试移除一个数据点的情况
        w1 = weights.copy()
        w1[i] = 0
        data2 = data[w1 != 0]
        tree2 = kdtree_type(data2)

        # 使用修改后的权重数组计算邻居数
        c1 = tree1.count_neighbors(tree1, np.linspace(0, 10, 100),
                weights=(w1, w1))
        # "c2 is correct"
        c2 = tree2.count_neighbors(tree2, np.linspace(0, 10, 100))

        # 断言移除数据点后的邻居计数结果一致
        assert_array_equal(c1, c2)

        # 断言当使用不同的 KD 树对象和相同的权重数组时，抛出 ValueError 异常
        assert_raises(ValueError, tree1.count_neighbors,
            tree2, np.linspace(0, 10, 100), weights=w1)

@pytest.mark.fail_slow(10)
def test_kdtree_count_neighbous_multiple_r(kdtree_type):
    # 设置随机种子
    np.random.seed(1234)
    n = 2000
    m = 2
    # 创建二维正态分布数据集
    data = np.random.normal(size=(n, m))
    # 使用给定的 kdtree_type 构建 KD 树
    kdtree = kdtree_type(data, leafsize=1)
    # 设置查询半径列表
    r0 = [0, 0.01, 0.01, 0.02, 0.05]
    i0 = np.arange(len(r0))
    # 计算不同半径下的邻居数量
    n0 = kdtree.count_neighbors(kdtree, r0)
    nnc = kdtree.count_neighbors(kdtree, r0, cumulative=False)
    # 断言不同参数下的邻居数量累积结果一致
    assert_equal(n0, nnc.cumsum())

    for i, r in zip(itertools.permutations(i0),
                    itertools.permutations(r0)):
        # 使用不同排列的半径计算邻居数量
        n = kdtree.count_neighbors(kdtree, r)
        # 断言不同排列的半径计算结果一致
        assert_array_equal(n, n0[list(i)])

def test_len0_arrays(kdtree_type):
    # 确保长度为0的数组在范围查询中被正确处理
    # 创建随机数据集
    np.random.seed(1234)
    X = np.random.rand(10, 2)
    Y = np.random.rand(10, 2)
    # 使用给定的 kdtree_type 构建 KD 树
    tree = kdtree_type(X)
    # 查询单点的球形邻居
    d, i = tree.query([.5, .5], k=1)
    z = tree.query_ball_point([.5, .5], 0.1*d)
    # 断言空数组与预期结果一致
    assert_array_equal(z, [])
    # 查询多点的球形邻居
    d, i = tree.query(Y, k=1)
    mind = d.min()
    z = tree.query_ball_point(Y, 0.1*mind)
    y = np.empty(shape=(10, ), dtype=object)
    y.fill([])
    # 断言空数组与预期结果一致
    assert_array_equal(y, z)
    # 使用另一棵 KD 树进行球形邻居查询
    other = kdtree_type(Y)
    y = tree.query_ball_tree(other, 0.1*mind)
    # 断言数组应该等于 10 个空列表
    assert_array_equal(10*[[]], y)
    
    # 使用 count_neighbors 方法计算与给定距离范围内的邻居数量
    y = tree.count_neighbors(other, 0.1*mind)
    # 断言结果应该为 0
    assert_(y == 0)
    
    # 使用 sparse_distance_matrix 方法计算稀疏距离矩阵，输出类型为 'dok_matrix'
    y = tree.sparse_distance_matrix(other, 0.1*mind, output_type='dok_matrix')
    # 断言结果应该是一个 10x10 的零矩阵
    assert_array_equal(y == np.zeros((10, 10)), True)
    
    # 使用 sparse_distance_matrix 方法计算稀疏距离矩阵，输出类型为 'coo_matrix'
    y = tree.sparse_distance_matrix(other, 0.1*mind, output_type='coo_matrix')
    # 断言结果应该是一个 10x10 的零矩阵
    assert_array_equal(y == np.zeros((10, 10)), True)
    
    # 使用 sparse_distance_matrix 方法计算稀疏距离矩阵，输出类型为 'dict'
    y = tree.sparse_distance_matrix(other, 0.1*mind, output_type='dict')
    # 断言结果应该是空字典
    assert_equal(y, {})
    
    # 使用 sparse_distance_matrix 方法计算稀疏距离矩阵，输出类型为 'ndarray'
    y = tree.sparse_distance_matrix(other, 0.1*mind, output_type='ndarray')
    # 定义预期的 dtype 结构
    _dtype = [('i', np.intp), ('j', np.intp), ('v', np.float64)]
    res_dtype = np.dtype(_dtype, align=True)
    # 创建一个空的数组 z，dtype 为 res_dtype
    z = np.empty(shape=(0, ), dtype=res_dtype)
    # 断言计算得到的 y 应该等于 z
    assert_array_equal(y, z)
    
    # 使用 query 方法查询最近邻居对
    d, i = tree.query(X, k=2)
    # 计算距离的最小值
    mind = d[:, -1].min()
    # 使用 query_pairs 方法查询距离在 0.1*mind 范围内的邻居对，输出类型为 'set'
    y = tree.query_pairs(0.1*mind, output_type='set')
    # 断言结果应该是空集合
    assert_equal(y, set())
    
    # 使用 query_pairs 方法查询距离在 0.1*mind 范围内的邻居对，输出类型为 'ndarray'
    y = tree.query_pairs(0.1*mind, output_type='ndarray')
    # 创建一个空的二维数组 z，形状为 (0, 2)
    z = np.empty(shape=(0, 2), dtype=np.intp)
    # 断言结果应该是空数组 z
    assert_array_equal(y, z)
# 测试 KD 树对于具有重复输入的情况
def test_kdtree_duplicated_inputs(kdtree_type):
    # 数据维度设置为 n=1024, m 变化从 1 到 7
    n = 1024
    for m in range(1, 8):
        # 创建数据数组，全部为 1 的部分和后半部分为 2
        data = np.ones((n, m))
        data[n//2:] = 2

        # 生成平衡和紧凑选项的排列组合
        for balanced, compact in itertools.product((False, True), repeat=2):
            # 创建 KD 树对象，根据平衡和紧凑选项设置叶子节点大小为 1
            kdtree = kdtree_type(data, balanced_tree=balanced,
                                 compact_nodes=compact, leafsize=1)
            # 断言 KD 树的大小为 3
            assert kdtree.size == 3

            # 获取 KD 树的根节点
            tree = (kdtree.tree if kdtree_type is cKDTree else
                    kdtree.tree._node)

            # 断言左子树节点的索引按顺序为 0 到 n//2-1
            assert_equal(
                np.sort(tree.lesser.indices),
                np.arange(0, n // 2))
            # 断言右子树节点的索引按顺序为 n//2 到 n-1
            assert_equal(
                np.sort(tree.greater.indices),
                np.arange(n // 2, n))


# 测试 KD 树对于非累积非递减的情况
def test_kdtree_noncumulative_nondecreasing(kdtree_type):
    # 创建包含单个元素的数据集，初始化 KD 树对象
    kdtree = kdtree_type([[0]], leafsize=1)

    # 断言调用 count_neighbors 方法时会抛出 ValueError 异常
    assert_raises(ValueError, kdtree.count_neighbors,
        kdtree, [0.1, 0], cumulative=False)


# 测试 KD 树对于短距离最近邻搜索的情况
def test_short_knn(kdtree_type):
    # 根据给定的坐标数组创建 KD 树对象
    xyz = np.array([
        [0., 0., 0.],
        [1.01, 0., 0.],
        [0., 1., 0.],
        [0., 1.01, 0.],
        [1., 0., 0.],
        [1., 1., 0.]],
    dtype='float64')

    ckdt = kdtree_type(xyz)

    # 查询 KD 树中每个点的最近邻点，限制最大距离为 0.2
    deq, ieq = ckdt.query(xyz, k=4, distance_upper_bound=0.2)

    # 断言返回的距离数组与预期的数组几乎相等
    assert_array_almost_equal(deq,
            [[0., np.inf, np.inf, np.inf],
            [0., 0.01, np.inf, np.inf],
            [0., 0.01, np.inf, np.inf],
            [0., 0.01, np.inf, np.inf],
            [0., 0.01, np.inf, np.inf],
            [0., np.inf, np.inf, np.inf]])


# 测试 KD 树对于球形区域内点的查询
def test_query_ball_point_vector_r(kdtree_type):
    # 使用正态分布生成数据和查询点
    np.random.seed(1234)
    data = np.random.normal(size=(100, 3))
    query = np.random.normal(size=(100, 3))

    # 创建 KD 树对象
    tree = kdtree_type(data)

    # 生成随机距离数组，查询球形区域内的点索引列表
    d = np.random.uniform(0, 0.3, size=len(query))
    rvector = tree.query_ball_point(query, d)

    # 逐个比较向量化查询和标量查询的结果
    rscalar = [tree.query_ball_point(qi, di) for qi, di in zip(query, d)]
    for a, b in zip(rvector, rscalar):
        assert_array_equal(sorted(a), sorted(b))


# 测试 KD 树对于球形区域内点的查询（返回查询到的点的数量）
def test_query_ball_point_length(kdtree_type):
    # 使用正态分布生成数据和查询点
    np.random.seed(1234)
    data = np.random.normal(size=(100, 3))
    query = np.random.normal(size=(100, 3))

    # 创建 KD 树对象
    tree = kdtree_type(data)

    # 设置球形区域的半径
    d = 0.3

    # 查询球形区域内点的索引列表，并返回结果长度
    length = tree.query_ball_point(query, d, return_length=True)

    # 使用不同方式执行查询并比较结果长度
    length2 = [len(ind) for ind in tree.query_ball_point(query, d, return_length=False)]
    length3 = [len(tree.query_ball_point(qi, d)) for qi in query]
    length4 = [tree.query_ball_point(qi, d, return_length=True) for qi in query]

    # 断言所有查询结果长度相等
    assert_array_equal(length, length2)
    assert_array_equal(length, length3)
    assert_array_equal(length, length4)


# 测试 KD 树对于不连续数据集的建立
def test_discontiguous(kdtree_type):
    # 使用正态分布生成不连续的数据集
    np.random.seed(1234)
    data = np.random.normal(size=(100, 3))
    d_contiguous = np.arange(100) * 0.04
    # 创建一个非连续的 NumPy 数组，其中元素是从 99 到 0 的整数乘以 0.04，然后再反向排列
    d_discontiguous = np.ascontiguousarray(
                          np.arange(100)[::-1] * 0.04)[::-1]
    # 创建一个连续的 NumPy 数组，包含从标准正态分布中随机抽取的大小为 (100, 3) 的样本
    query_contiguous = np.random.normal(size=(100, 3))
    # 将连续的 NumPy 数组转换为非连续的数组
    query_discontiguous = np.ascontiguousarray(query_contiguous.T).T
    # 断言非连续数组的最后一个轴的步长与连续数组不同
    assert query_discontiguous.strides[-1] != query_contiguous.strides[-1]
    # 断言非连续的 d_discontiguous 数组的最后一个轴的步长与连续的 d_contiguous 数组不同

    # 使用给定的数据创建一个 KD 树对象
    tree = kdtree_type(data)

    # 使用 KD 树查询连续数组 query_contiguous 中距离 d_contiguous 范围内的点，返回长度
    length1 = tree.query_ball_point(query_contiguous,
                                    d_contiguous, return_length=True)
    # 使用 KD 树查询非连续数组 query_discontiguous 中距离 d_discontiguous 范围内的点，返回长度
    length2 = tree.query_ball_point(query_discontiguous,
                                    d_discontiguous, return_length=True)

    # 断言两个查询的结果长度相等
    assert_array_equal(length1, length2)

    # 使用 KD 树查询连续数组 query_contiguous 中与每个点最近的邻居距离和索引
    d1, i1 = tree.query(query_contiguous, 1)
    # 使用 KD 树查询非连续数组 query_discontiguous 中与每个点最近的邻居距离和索引
    d2, i2 = tree.query(query_discontiguous, 1)

    # 断言两个查询的距离结果数组相等
    assert_array_equal(d1, d2)
    # 断言两个查询的索引结果数组相等
    assert_array_equal(i1, i2)
@pytest.mark.parametrize("balanced_tree, compact_nodes",
    [(True, False),
     (True, True),
     (False, False),
     (False, True)])
# 使用 pytest 的参数化功能，为 test_kdtree_empty_input 函数定义多组参数组合进行测试
def test_kdtree_empty_input(kdtree_type, balanced_tree, compact_nodes):
    # 解决 GitHub 上的问题链接
    # 设置随机种子为 1234
    np.random.seed(1234)
    # 创建一个形状为空的 NumPy 数组，用作空输入
    empty_v3 = np.empty(shape=(0, 3))
    # 创建一个形状为 (1, 3) 的全为 1 的 NumPy 数组，用作查询向量
    query_v3 = np.ones(shape=(1, 3))
    # 创建一个形状为 (2, 3) 的全为 1 的 NumPy 数组，用作查询向量
    query_v2 = np.ones(shape=(2, 3))

    # 使用给定类型的 kdtree_type 构建 KD 树对象，传入空输入数组和其他参数
    tree = kdtree_type(empty_v3, balanced_tree=balanced_tree,
                       compact_nodes=compact_nodes)
    # 查询球形区域内距离小于 0.3 的点的数量，期望结果为 0
    length = tree.query_ball_point(query_v3, 0.3, return_length=True)
    assert length == 0

    # 查询最近的两个点及其距离
    dd, ii = tree.query(query_v2, 2)
    # 断言返回的索引数组的形状为 (2, 2)
    assert ii.shape == (2, 2)
    # 断言返回的距离数组的形状为 (2, 2)，且所有元素为无穷大
    assert dd.shape == (2, 2)
    assert np.isinf(dd).all()

    # 计算树中与指定点距离小于 1 的邻居点的数量，期望结果为 [0, 0]
    N = tree.count_neighbors(tree, [0, 1])
    assert_array_equal(N, [0, 0])

    # 构建稀疏距离矩阵，期望结果为形状为 (0, 0)
    M = tree.sparse_distance_matrix(tree, 0.3)
    assert M.shape == (0, 0)


@KDTreeTest
# 使用 KDTreeTest 装饰器标记测试类 _Test_sorted_query_ball_point
class _Test_sorted_query_ball_point:
    def setup_method(self):
        # 设置随机种子为 1234
        np.random.seed(1234)
        # 创建一个形状为 (100, 1) 的随机数数组
        self.x = np.random.randn(100, 1)
        # 使用给定类型的 kdtree_type 构建 KD 树对象，传入随机数数组
        self.ckdt = self.kdtree_type(self.x)

    def test_return_sorted_True(self):
        # 查询球形区域内距离小于 1 的点的索引列表，期望返回的索引列表是排序的
        idxs_list = self.ckdt.query_ball_point(self.x, 1., return_sorted=True)
        for idxs in idxs_list:
            assert_array_equal(idxs, sorted(idxs))

        for xi in self.x:
            # 对每个点查询球形区域内距离小于 1 的点的索引，期望返回的索引列表是排序的
            idxs = self.ckdt.query_ball_point(xi, 1., return_sorted=True)
            assert_array_equal(idxs, sorted(idxs))

    def test_return_sorted_None(self):
        """Previous behavior was to sort the returned indices if there were
        multiple points per query but not sort them if there was a single point
        per query."""
        # 查询球形区域内距离小于 1 的点的索引列表，期望返回的索引列表是排序的
        idxs_list = self.ckdt.query_ball_point(self.x, 1.)
        for idxs in idxs_list:
            assert_array_equal(idxs, sorted(idxs))

        # 对每个点查询球形区域内距离小于 1 的点的索引列表，期望返回的索引列表与排序后的单点查询结果一致
        idxs_list_single = [self.ckdt.query_ball_point(xi, 1.) for xi in self.x]
        # 查询球形区域内距离小于 1 的点的索引列表，不排序返回的索引列表应与单点查询结果一致
        idxs_list_False = self.ckdt.query_ball_point(self.x, 1., return_sorted=False)
        for idxs0, idxs1 in zip(idxs_list_False, idxs_list_single):
            assert_array_equal(idxs0, idxs1)


def test_kdtree_complex_data():
    # 测试 KD 树能否拒绝复数输入点 (gh-9108)
    points = np.random.rand(10, 2).view(complex)

    # 使用 pytest 断言应抛出 TypeError 异常，异常信息包含 "complex data"
    with pytest.raises(TypeError, match="complex data"):
        t = KDTree(points)

    # 构建 KD 树对象，传入复数数组的实部
    t = KDTree(points.real)

    # 使用 pytest 断言应抛出 TypeError 异常，异常信息包含 "complex data"
    with pytest.raises(TypeError, match="complex data"):
        t.query(points)

    # 使用 pytest 断言应抛出 TypeError 异常，异常信息包含 "complex data"
    with pytest.raises(TypeError, match="complex data"):
        t.query_ball_point(points, r=1)


def test_kdtree_tree_access():
    # 测试 KD 树的 tree 属性能否用于遍历 KD 树
    np.random.seed(1234)
    # 创建一个形状为 (100, 4) 的随机数数组
    points = np.random.rand(100, 4)
    # 构建 KD 树对象，传入随机数数组
    t = KDTree(points)
    # 获取根节点
    root = t.tree

    # 使用断言检查根节点是 KDTree.innernode 类型
    assert isinstance(root, KDTree.innernode)
    # 使用断言检查根节点的子节点数等于点的总数
    assert root.children == points.shape[0]

    # 访问树并断言每个节点的基本属性
    nodes = [root]
    # 当节点列表非空时循环进行以下操作
    while nodes:
        # 从节点列表中取出最后一个节点
        n = nodes.pop(-1)

        # 如果当前节点是叶节点
        if isinstance(n, KDTree.leafnode):
            # 断言该叶节点的 children 属性是整数类型
            assert isinstance(n.children, int)
            # 断言叶节点的 children 数量等于其包含的索引数量
            assert n.children == len(n.idx)
            # 断言数据点集合中叶节点索引对应的点与节点的 data_points 属性相等
            assert_array_equal(points[n.idx], n._node.data_points)
        else:
            # 如果当前节点是内部节点
            assert isinstance(n, KDTree.innernode)
            # 断言分割维度 split_dim 是整数类型
            assert isinstance(n.split_dim, int)
            # 断言分割维度在有效范围内（大于等于 0 且小于 t.m）
            assert 0 <= n.split_dim < t.m
            # 断言分割值 split 是浮点数类型
            assert isinstance(n.split, float)
            # 断言该内部节点的 children 属性是整数类型
            assert isinstance(n.children, int)
            # 断言内部节点的 children 数量等于其左右子节点的 children 之和
            assert n.children == n.less.children + n.greater.children
            # 将内部节点的左子节点和右子节点依次添加到节点列表中
            nodes.append(n.greater)
            nodes.append(n.less)
# 测试 KDTree 类的属性是否可用
def test_kdtree_attributes():
    # 设定随机种子以确保可重复性
    np.random.seed(1234)
    # 生成一个形状为 (100, 4) 的随机数组作为点集
    points = np.random.rand(100, 4)
    # 创建 KDTree 对象 t，传入点集作为参数
    t = KDTree(points)

    # 断言 t.m 是整数类型，并检查其值
    assert isinstance(t.m, int)
    assert t.n == points.shape[0]

    # 断言 t.n 是整数类型，并检查其值
    assert isinstance(t.n, int)
    assert t.m == points.shape[1]

    # 断言 t.leafsize 是整数类型，并检查其值为 10
    assert isinstance(t.leafsize, int)
    assert t.leafsize == 10

    # 断言 t.maxes 与 np.amax(points, axis=0) 相等
    assert_array_equal(t.maxes, np.amax(points, axis=0))
    # 断言 t.mins 与 np.amin(points, axis=0) 相等
    assert_array_equal(t.mins, np.amin(points, axis=0))
    # 断言 t.data 与 points 相同
    assert t.data is points


@pytest.mark.parametrize("kdtree_class", [KDTree, cKDTree])
def test_kdtree_count_neighbors_weighted(kdtree_class):
    # 设定随机种子以确保可重复性
    np.random.seed(1234)
    # 创建一个从 0.05 到 1，步长为 0.05 的数组 r
    r = np.arange(0.05, 1, 0.05)

    # 生成两组随机数作为点集 A 和 B，并将它们重新形状为 (7, 3) 和 (15, 3)
    A = np.random.random(21).reshape((7, 3))
    B = np.random.random(45).reshape((15, 3))

    # 生成与 A 和 B 对应的随机权重 wA 和 wB
    wA = np.random.random(7)
    wB = np.random.random(15)

    # 使用给定的 kdtree_class 创建 KDTree 对象 kdA 和 kdB
    kdA = kdtree_class(A)
    kdB = kdtree_class(B)

    # 计算 kdA 和 kdB 之间在距离范围 r 内的邻居数量，带权重
    nAB = kdA.count_neighbors(kdB, r, cumulative=False, weights=(wA, wB))

    # 与暴力方法比较结果
    # 计算权重矩阵
    weights = wA[None, :] * wB[:, None]
    # 计算 A 和 B 之间的距离
    dist = np.linalg.norm(A[None, :, :] - B[:, None, :], axis=-1)
    # 期望结果
    expect = [np.sum(weights[(prev_radius < dist) & (dist <= radius)])
              for prev_radius, radius in zip(itertools.chain([0], r[:-1]), r)]
    # 断言 nAB 与期望结果相近
    assert_allclose(nAB, expect)


def test_kdtree_nan():
    # 创建一个包含有限数和 NaN 的值列表 vals
    vals = [1, 5, -10, 7, -4, -16, -6, 6, 3, -11]
    n = len(vals)
    # 创建一个包含 vals 和 NaN 的数组 data，并将其转置为列向量
    data = np.concatenate([vals, np.full(n, np.nan)])[:, None]
    # 使用 pytest 断言应该引发 ValueError 异常，且异常消息应包含 "must be finite"
    with pytest.raises(ValueError, match="must be finite"):
        # 尝试创建包含 NaN 的 KDTree 对象
        KDTree(data)


def test_nonfinite_inputs_gh_18223():
    # 创建一个随机数生成器 rng
    rng = np.random.default_rng(12345)
    # 生成一个形状为 (100, 3) 的随机坐标数组 coords
    coords = rng.uniform(size=(100, 3), low=0.0, high=0.1)
    # 创建一个 KDTree 对象 t，传入 coords 作为参数，并关闭平衡和紧凑节点选项
    t = KDTree(coords, balanced_tree=False, compact_nodes=False)
    # 创建一个包含 NaN 的坏坐标列表 bad_coord
    bad_coord = [np.nan for _ in range(3)]

    # 使用 pytest 断言应该引发 ValueError 异常，且异常消息应包含 "must be finite"
    with pytest.raises(ValueError, match="must be finite"):
        # 尝试查询包含 NaN 坐标的 t
        t.query(bad_coord)
    with pytest.raises(ValueError, match="must be finite"):
        t.query_ball_point(bad_coord, 1)

    # 将 coords 的第一行全部设置为 NaN
    coords[0, :] = np.nan
    # 使用 pytest 断言应该引发 ValueError 异常，且异常消息应包含 "must be finite"
    with pytest.raises(ValueError, match="must be finite"):
        # 尝试创建包含 NaN 的 KDTree 对象，开启平衡选项，关闭紧凑节点选项
        KDTree(coords, balanced_tree=True, compact_nodes=False)
    with pytest.raises(ValueError, match="must be finite"):
        # 尝试创建包含 NaN 的 KDTree 对象，关闭平衡选项，开启紧凑节点选项
        KDTree(coords, balanced_tree=False, compact_nodes=True)
    with pytest.raises(ValueError, match="must be finite"):
        # 尝试创建包含 NaN 的 KDTree 对象，开启平衡和紧凑节点选项
        KDTree(coords, balanced_tree=True, compact_nodes=True)
    with pytest.raises(ValueError, match="must be finite"):
        # 尝试创建包含 NaN 的 KDTree 对象，关闭平衡和紧凑节点选项
        KDTree(coords, balanced_tree=False, compact_nodes=False)


@pytest.mark.parametrize("incantation", [cKDTree, KDTree])
def test_gh_18800(incantation):
    # 我们在 kd 树工作流中禁止非有限值，因此需要强制转换为 NumPy 数组
    # 定义一个继承自 np.ndarray 的自定义类 ArrLike
    class ArrLike(np.ndarray):
        # __new__ 方法用于创建一个新实例
        def __new__(cls, input_array):
            # 将输入数组 input_array 转换为 np.ndarray 类型，并将其视图赋给 obj
            obj = np.asarray(input_array).view(cls)
            # 设置对象的 all 属性为 None，用以模拟在 gh-18800 中遇到的 pandas DataFrame 问题
            obj.all = None
            return obj
    
        # __array_finalize__ 方法用于在数组创建后进行对象的最终初始化
        def __array_finalize__(self, obj):
            # 如果 obj 为 None，则直接返回，无需进一步处理
            if obj is None:
                return
            # 获取 obj 对象的 all 属性，并将其赋值给当前对象的 self.all
            self.all = getattr(obj, 'all', None)
    
    # 创建一个包含三个点坐标的列表
    points = [
        [66.22, 32.54],
        [22.52, 22.39],
        [31.01, 81.21],
    ]
    # 使用 np.array 将 points 转换为 NumPy 数组
    arr = np.array(points)
    # 使用自定义类 ArrLike 创建一个新的数组实例 arr_like
    arr_like = ArrLike(arr)
    # 使用 incantation 函数处理 points 和参数 10，返回一个 tree 对象
    tree = incantation(points, 10)
    # 对 tree 对象执行 query 方法，查询与 arr_like 最接近的 1 个点
    tree.query(arr_like, 1)
    # 对 tree 对象执行 query_ball_point 方法，查询与 arr_like 中的点距离小于 200 的点集合
    tree.query_ball_point(arr_like, 200)
```