# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_quadratic_assignment.py`

```
import pytest
import numpy as np
from scipy.optimize import quadratic_assignment, OptimizeWarning
from scipy.optimize._qap import _calc_score as _score
from numpy.testing import assert_equal, assert_, assert_warns

################
# Common Tests #
################

def chr12c():
    # 定义矩阵 A 和 B，分别表示问题实例的两个成本矩阵
    A = [
        [0, 90, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [90, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0],
        [10, 0, 0, 0, 43, 0, 0, 0, 0, 0, 0, 0],
        [0, 23, 0, 0, 0, 88, 0, 0, 0, 0, 0, 0],
        [0, 0, 43, 0, 0, 0, 26, 0, 0, 0, 0, 0],
        [0, 0, 0, 88, 0, 0, 0, 16, 0, 0, 0, 0],
        [0, 0, 0, 0, 26, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 16, 0, 0, 0, 96, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 29, 0],
        [0, 0, 0, 0, 0, 0, 0, 96, 0, 0, 0, 37],
        [0, 0, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 0, 0],
    ]
    B = [
        [0, 36, 54, 26, 59, 72, 9, 34, 79, 17, 46, 95],
        [36, 0, 73, 35, 90, 58, 30, 78, 35, 44, 79, 36],
        [54, 73, 0, 21, 10, 97, 58, 66, 69, 61, 54, 63],
        [26, 35, 21, 0, 93, 12, 46, 40, 37, 48, 68, 85],
        [59, 90, 10, 93, 0, 64, 5, 29, 76, 16, 5, 76],
        [72, 58, 97, 12, 64, 0, 96, 55, 38, 54, 0, 34],
        [9, 30, 58, 46, 5, 96, 0, 83, 35, 11, 56, 37],
        [34, 78, 66, 40, 29, 55, 83, 0, 44, 12, 15, 80],
        [79, 35, 69, 37, 76, 38, 35, 44, 0, 64, 39, 33],
        [17, 44, 61, 48, 16, 54, 11, 12, 64, 0, 70, 86],
        [46, 79, 54, 68, 5, 0, 56, 15, 39, 70, 0, 18],
        [95, 36, 63, 85, 76, 34, 37, 80, 33, 86, 18, 0],
    ]
    # 将 A 和 B 转换为 NumPy 数组
    A, B = np.array(A), np.array(B)
    n = A.shape[0]

    # Umeyama IVB 文章中给出的全局最优排列
    opt_perm = np.array([7, 5, 1, 3, 10, 4, 8, 6, 9, 11, 2, 12]) - [1] * n

    return A, B, opt_perm


class QAPCommonTests:
    """
    Base class for `quadratic_assignment` tests.
    """
    def setup_method(self):
        # 设置随机种子为 0，确保测试结果的可重复性
        np.random.seed(0)

    # Test global optima of problem from Umeyama IVB
    # https://pcl.sitehost.iu.edu/rgoldsto/papers/weighted%20graph%20match2.pdf
    # Graph matching maximum is in the paper
    # QAP minimum determined by brute force
    # 测试准确性的第一个测试方法
    def test_accuracy_1(self):
        # 除了测试准确性，还检查 A 和 B 是否可以是列表形式
        A = [[0, 3, 4, 2],
             [0, 0, 1, 2],
             [1, 0, 0, 1],
             [0, 0, 1, 0]]

        B = [[0, 4, 2, 4],
             [0, 0, 1, 0],
             [0, 2, 0, 2],
             [0, 1, 2, 0]]

        # 调用 quadratic_assignment 函数计算
        res = quadratic_assignment(A, B, method=self.method,
                                   options={"rng": 0, "maximize": False})
        # 断言优化结果的函数值为 10
        assert_equal(res.fun, 10)
        # 断言优化结果的列指标为指定的数组
        assert_equal(res.col_ind, np.array([1, 2, 3, 0]))

        # 再次调用 quadratic_assignment 函数计算，不同的最大化选项
        res = quadratic_assignment(A, B, method=self.method,
                                   options={"rng": 0, "maximize": True})

        if self.method == 'faq':
            # 当使用 FAQ 方法时，全局最优值为 40，但 FAQ 方法得到 37
            assert_equal(res.fun, 37)
            # 断言优化结果的列指标为指定的数组
            assert_equal(res.col_ind, np.array([0, 2, 3, 1]))
        else:
            # 如果不是 FAQ 方法，断言优化结果的函数值为 40
            assert_equal(res.fun, 40)
            # 断言优化结果的列指标为指定的数组
            assert_equal(res.col_ind, np.array([0, 3, 1, 2]))

        # 再次调用 quadratic_assignment 函数计算，不同的最大化选项
        res = quadratic_assignment(A, B, method=self.method,
                                   options={"rng": 0, "maximize": True})

    # 测试 Umeyama IIIB 论文中问题的全局最优解
    # https://pcl.sitehost.iu.edu/rgoldsto/papers/weighted%20graph%20match2.pdf
    # 图匹配的最大化在论文中有描述
    # QAP 最小化由暴力搜索确定
    def test_accuracy_2(self):

        A = np.array([[0, 5, 8, 6],
                      [5, 0, 5, 1],
                      [8, 5, 0, 2],
                      [6, 1, 2, 0]])

        B = np.array([[0, 1, 8, 4],
                      [1, 0, 5, 2],
                      [8, 5, 0, 5],
                      [4, 2, 5, 0]])

        # 调用 quadratic_assignment 函数计算
        res = quadratic_assignment(A, B, method=self.method,
                                   options={"rng": 0, "maximize": False})
        if self.method == 'faq':
            # 当使用 FAQ 方法时，全局最优值为 176，但 FAQ 方法得到 178
            assert_equal(res.fun, 178)
            # 断言优化结果的列指标为指定的数组
            assert_equal(res.col_ind, np.array([1, 0, 3, 2]))
        else:
            # 如果不是 FAQ 方法，断言优化结果的函数值为 176
            assert_equal(res.fun, 176)
            # 断言优化结果的列指标为指定的数组
            assert_equal(res.col_ind, np.array([1, 2, 3, 0]))

        # 再次调用 quadratic_assignment 函数计算，不同的最大化选项
        res = quadratic_assignment(A, B, method=self.method,
                                   options={"rng": 0, "maximize": True})
        # 断言优化结果的函数值为 286
        assert_equal(res.fun, 286)
        # 断言优化结果的列指标为指定的数组
        assert_equal(res.col_ind, np.array([2, 3, 0, 1]))
    def test_accuracy_3(self):
        # 调用 chr12c 函数获取输入 A, B 和最优置换 opt_perm
        A, B, opt_perm = chr12c()

        # 进行基本的最小化优化
        res = quadratic_assignment(A, B, method=self.method,
                                   options={"rng": 0})
        # 断言目标函数值在特定范围内
        assert_(11156 <= res.fun < 21000)
        # 断言优化结果的目标函数值与得分函数 _score 计算结果相等
        assert_equal(res.fun, _score(A, B, res.col_ind))

        # 进行基本的最大化优化
        res = quadratic_assignment(A, B, method=self.method,
                                   options={"rng": 0, 'maximize': True})
        # 断言目标函数值在特定范围内
        assert_(74000 <= res.fun < 85000)
        # 断言优化结果的目标函数值与得分函数 _score 计算结果相等
        assert_equal(res.fun, _score(A, B, res.col_ind))

        # 使用部分匹配检查目标函数值
        seed_cost = np.array([4, 8, 10])
        seed = np.asarray([seed_cost, opt_perm[seed_cost]]).T
        res = quadratic_assignment(A, B, method=self.method,
                                   options={'partial_match': seed})
        # 断言目标函数值在特定范围内
        assert_(11156 <= res.fun < 21000)
        # 断言部分匹配的结果与最优置换在指定索引上的一致性
        assert_equal(res.col_ind[seed_cost], opt_perm[seed_cost])

        # 检查部分匹配为全局最优时的性能
        seed = np.asarray([np.arange(len(A)), opt_perm]).T
        res = quadratic_assignment(A, B, method=self.method,
                                   options={'partial_match': seed})
        # 断言优化结果的列索引与部分匹配的结果一致
        assert_equal(res.col_ind, seed[:, 1].T)
        # 断言目标函数值等于最小值 11156
        assert_equal(res.fun, 11156)
        # 断言迭代次数为 0
        assert_equal(res.nit, 0)

        # 检查零大小矩阵输入时的性能
        empty = np.empty((0, 0))
        res = quadratic_assignment(empty, empty, method=self.method,
                                   options={"rng": 0})
        # 断言迭代次数为 0
        assert_equal(res.nit, 0)
        # 断言目标函数值为 0

    def test_unknown_options(self):
        A, B, opt_perm = chr12c()

        def f():
            # 调用 quadratic_assignment 函数并传入未知选项 "ekki-ekki"
            quadratic_assignment(A, B, method=self.method,
                                 options={"ekki-ekki": True})
        # 断言 f 函数会引发 OptimizeWarning 警告
        assert_warns(OptimizeWarning, f)
class TestFAQ(QAPCommonTests):
    method = "faq"

    def test_options(self):
        # cost and distance matrices of QAPLIB instance chr12c
        A, B, opt_perm = chr12c()
        n = len(A)

        # check that max_iter is obeying with low input value
        res = quadratic_assignment(A, B,
                                   options={'maxiter': 5})
        assert_equal(res.nit, 5)  # Assert that the number of iterations matches the specified maxiter

        # test with shuffle
        res = quadratic_assignment(A, B,
                                   options={'shuffle_input': True})
        assert_(11156 <= res.fun < 21000)  # Assert that the objective function value falls within a specific range

        # test with randomized init
        res = quadratic_assignment(A, B,
                                   options={'rng': 1, 'P0': "randomized"})
        assert_(11156 <= res.fun < 21000)  # Assert that the objective function value falls within a specific range

        # check with specified P0
        K = np.ones((n, n)) / float(n)
        K = _doubly_stochastic(K)
        res = quadratic_assignment(A, B,
                                   options={'P0': K})
        assert_(11156 <= res.fun < 21000)  # Assert that the objective function value falls within a specific range

    def test_specific_input_validation(self):

        A = np.identity(2)
        B = A

        # method is implicitly faq

        # ValueError Checks: making sure single value parameters are of
        # correct value
        with pytest.raises(ValueError, match="Invalid 'P0' parameter"):
            quadratic_assignment(A, B, options={'P0': "random"})  # Ensure ValueError is raised for invalid 'P0' parameter
        with pytest.raises(
                ValueError, match="'maxiter' must be a positive integer"):
            quadratic_assignment(A, B, options={'maxiter': -1})  # Ensure ValueError is raised for negative 'maxiter'
        with pytest.raises(ValueError, match="'tol' must be a positive float"):
            quadratic_assignment(A, B, options={'tol': -1})  # Ensure ValueError is raised for negative 'tol'

        # TypeError Checks: making sure single value parameters are of
        # correct type
        with pytest.raises(TypeError):
            quadratic_assignment(A, B, options={'maxiter': 1.5})  # Ensure TypeError is raised for non-integer 'maxiter'

        # test P0 matrix input
        with pytest.raises(
                ValueError,
                match="`P0` matrix must have shape m' x m', where m'=n-m"):
            quadratic_assignment(
                np.identity(4), np.identity(4),
                options={'P0': np.ones((3, 3))}
            )  # Ensure ValueError is raised for incorrect shape of 'P0' matrix

        K = [[0.4, 0.2, 0.3],
             [0.3, 0.6, 0.2],
             [0.2, 0.2, 0.7]]
        # matrix that isn't quite doubly stochastic
        with pytest.raises(
                ValueError, match="`P0` matrix must be doubly stochastic"):
            quadratic_assignment(
                np.identity(3), np.identity(3), options={'P0': K}
            )  # Ensure ValueError is raised for non-doubly stochastic 'P0' matrix


class Test2opt(QAPCommonTests):
    method = "2opt"
    # 定义一个测试方法，用于测试确定性问题
    def test_deterministic(self):
        # 设置随机种子为0，确保每次方法调用前都会执行这一步
        np.random.seed(0)
        
        # 定义一个大小为20的变量n
        
        # 生成两个大小为n x n的随机矩阵A和B
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        
        # 调用 quadratic_assignment 函数计算结果 res1
        res1 = quadratic_assignment(A, B, method=self.method)
        
        # 重新设置随机种子为0，以确保再次执行时产生相同的随机数序列
        np.random.seed(0)
        
        # 重新生成随机矩阵A和B
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        
        # 再次调用 quadratic_assignment 函数计算结果 res2
        res2 = quadratic_assignment(A, B, method=self.method)
        
        # 断言两次调用的迭代次数相同
        assert_equal(res1.nit, res2.nit)

    # 定义另一个测试方法，用于测试带部分预测值的情况
    def test_partial_guess(self):
        # 定义一个大小为5的变量n
        n = 5
        
        # 生成两个大小为n x n的随机矩阵A和B
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        
        # 调用 quadratic_assignment 函数计算结果 res1，带有额外选项参数 'rng': 0
        res1 = quadratic_assignment(A, B, method=self.method,
                                    options={'rng': 0})
        
        # 创建一个部分猜测数组 guess，包含原始索引和 res1 结果中的列索引
        guess = np.array([np.arange(5), res1.col_ind]).T
        
        # 再次调用 quadratic_assignment 函数计算结果 res2，带有额外选项参数 'rng': 0 和 'partial_guess': guess
        res2 = quadratic_assignment(A, B, method=self.method,
                                    options={'rng': 0, 'partial_guess': guess})
        
        # 定义一个固定的索引列表 fix
        fix = [2, 4]
        
        # 创建一个部分匹配数组 match，包含 fix 中的原始索引和 res1 结果中的列索引
        match = np.array([np.arange(5)[fix], res1.col_ind[fix]]).T
        
        # 再次调用 quadratic_assignment 函数计算结果 res3，带有额外选项参数 'rng': 0, 'partial_guess': guess 和 'partial_match': match
        res3 = quadratic_assignment(A, B, method=self.method,
                                    options={'rng': 0, 'partial_guess': guess,
                                             'partial_match': match})
        
        # 断言 res1 的迭代次数不等于 n*(n+1)/2
        assert_(res1.nit != n*(n+1)/2)
        
        # 断言 res2 的迭代次数等于 n*(n+1)/2，这会测试每次交换的精确性
        assert_equal(res2.nit, n*(n+1)/2)
        
        # 断言 res3 的迭代次数等于 (n-2)*(n-1)/2，这会测试自由交换的精确性
        assert_equal(res3.nit, (n-2)*(n-1)/2)
    def test_specific_input_validation(self):
        # 确保种子节点数不超过成本/距离节点数
        _rm = _range_matrix
        # 使用 pytest 检测是否引发 ValueError，并检查错误信息是否包含特定文本
        with pytest.raises(
                ValueError,
                match="`partial_guess` can have only as many entries as"):
            # 调用 quadratic_assignment 函数，传入参数和选项字典
            quadratic_assignment(np.identity(3), np.identity(3),
                                 method=self.method,
                                 options={'partial_guess': _rm(5, 2)})
        # 测试只有两列种子节点的情况
        with pytest.raises(
                ValueError, match="`partial_guess` must have two columns"):
            # 再次调用 quadratic_assignment 函数，传入参数和选项字典
            quadratic_assignment(
                np.identity(3), np.identity(3), method=self.method,
                options={'partial_guess': _range_matrix(2, 3)}
            )
        # 测试种子节点的维度不超过两个的情况
        with pytest.raises(
                ValueError, match="`partial_guess` must have exactly two"):
            # 再次调用 quadratic_assignment 函数，传入参数和选项字典
            quadratic_assignment(
                np.identity(3), np.identity(3), method=self.method,
                options={'partial_guess': np.random.rand(3, 2, 2)}
            )
        # 确保种子节点不包含负值
        with pytest.raises(
                ValueError, match="`partial_guess` must contain only pos"):
            # 再次调用 quadratic_assignment 函数，传入参数和选项字典
            quadratic_assignment(
                np.identity(3), np.identity(3), method=self.method,
                options={'partial_guess': -1 * _range_matrix(2, 2)}
            )
        # 确保种子节点的值不超过节点数
        with pytest.raises(
                ValueError,
                match="`partial_guess` entries must be less than number"):
            # 再次调用 quadratic_assignment 函数，传入参数和选项字典
            quadratic_assignment(
                np.identity(5), np.identity(5), method=self.method,
                options={'partial_guess': 2 * _range_matrix(4, 2)}
            )
        # 确保种子矩阵的列是唯一的
        with pytest.raises(
                ValueError,
                match="`partial_guess` column entries must be unique"):
            # 最后一次调用 quadratic_assignment 函数，传入参数和选项字典
            quadratic_assignment(
                np.identity(3), np.identity(3), method=self.method,
                options={'partial_guess': np.ones((2, 2))}
            )
# 定义一个名为 TestQAPOnce 的测试类
class TestQAPOnce:
    # 在每个测试方法执行前的设置方法
    def setup_method(self):
        # 设定随机种子为 0，以确保每次运行的随机结果一致
        np.random.seed(0)

# 生成一个 a 行 b 列的零矩阵，其中每列包含从 0 到 a-1 的整数序列
def _range_matrix(a, b):
    mat = np.zeros((a, b))
    for i in range(b):
        mat[:, i] = np.arange(a)
    return mat

# 计算达到双随机性的概率转移矩阵 P
def _doubly_stochastic(P, tol=1e-3):
    # 设定最大迭代次数为 1000
    max_iter = 1000
    # 计算列向量 c，每个元素为 P 按列求和的倒数
    c = 1 / P.sum(axis=0)
    # 计算行向量 r，每个元素为 P 乘以列向量 c 后再按行求和的倒数
    r = 1 / (P @ c)
    # 将 P 赋值给 P_eps，P_eps 用于迭代计算
    P_eps = P

    # 开始迭代过程
    for it in range(max_iter):
        # 检查是否所有行和列的和都接近于 1，即达到了双随机性的阈值
        if ((np.abs(P_eps.sum(axis=1) - 1) < tol).all() and
                (np.abs(P_eps.sum(axis=0) - 1) < tol).all()):
            # 如果满足条件，结束迭代
            break

        # 更新列向量 c，每个元素为 r 乘以 P 后再按列求和的倒数
        c = 1 / (r @ P)
        # 更新行向量 r，每个元素为 P 乘以列向量 c 后再按行求和的倒数
        r = 1 / (P @ c)
        # 更新 P_eps，根据新的 r 和 c 重新计算 P
        P_eps = r[:, None] * P * c

    # 返回达到双随机性的概率转移矩阵 P_eps
    return P_eps
```