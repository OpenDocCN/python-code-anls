# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_linear_assignment.py`

```
# 从 numpy.testing 模块中导入 assert_array_equal 函数，用于比较数组是否相等
# 导入 pytest 模块，用于编写和运行测试用例

# 导入 numpy 模块，并使用 np 别名
import numpy as np

# 从 scipy.optimize 模块中导入 linear_sum_assignment 函数，用于求解线性和分配问题
# 从 scipy.sparse 模块中导入 random 和 matrix 函数
# 从 scipy.sparse.csgraph 模块中导入 min_weight_full_bipartite_matching 函数
# 从 scipy.sparse.csgraph.tests.test_matching 模块中导入相关测试函数和测试用例
from scipy.optimize import linear_sum_assignment
from scipy.sparse import random
from scipy.sparse._sputils import matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.sparse.csgraph.tests.test_matching import (
    linear_sum_assignment_assertions, linear_sum_assignment_test_cases
)


# 定义测试函数 test_linear_sum_assignment_input_shape，测试线性和分配函数对输入形状的处理
def test_linear_sum_assignment_input_shape():
    # 使用 pytest 的断言检查是否会引发 ValueError 异常，并检查异常信息中是否包含指定字符串
    with pytest.raises(ValueError, match="expected a matrix"):
        linear_sum_assignment([1, 2, 3])


# 定义测试函数 test_linear_sum_assignment_input_object，测试线性和分配函数对不同对象类型输入的处理
def test_linear_sum_assignment_input_object():
    # 定义二维列表 C
    C = [[1, 2, 3], [4, 5, 6]]
    # 使用 assert_array_equal 函数比较不同类型输入下线性和分配函数的输出结果是否一致
    assert_array_equal(linear_sum_assignment(C),
                       linear_sum_assignment(np.asarray(C)))
    assert_array_equal(linear_sum_assignment(C),
                       linear_sum_assignment(matrix(C)))


# 定义测试函数 test_linear_sum_assignment_input_bool，测试线性和分配函数对布尔类型输入的处理
def test_linear_sum_assignment_input_bool():
    # 创建单位矩阵 I，并将其转换为布尔类型后进行测试
    I = np.identity(3)
    assert_array_equal(linear_sum_assignment(I.astype(np.bool_)),
                       linear_sum_assignment(I))


# 定义测试函数 test_linear_sum_assignment_input_string，测试线性和分配函数对字符串类型输入的处理
def test_linear_sum_assignment_input_string():
    # 创建具有字符串类型数据的单位矩阵 I，并验证是否会引发 TypeError 异常
    I = np.identity(3)
    with pytest.raises(TypeError, match="Cannot cast array data"):
        linear_sum_assignment(I.astype(str))


# 定义测试函数 test_linear_sum_assignment_input_nan，测试线性和分配函数对 NaN 值输入的处理
def test_linear_sum_assignment_input_nan():
    # 创建包含 NaN 值的对角矩阵 I，并验证是否会引发 ValueError 异常
    I = np.diag([np.nan, 1, 1])
    with pytest.raises(ValueError, match="contains invalid numeric entries"):
        linear_sum_assignment(I)


# 定义测试函数 test_linear_sum_assignment_input_neginf，测试线性和分配函数对负无穷值输入的处理
def test_linear_sum_assignment_input_neginf():
    # 创建包含负无穷值的对角矩阵 I，并验证是否会引发 ValueError 异常
    I = np.diag([1, -np.inf, 1])
    with pytest.raises(ValueError, match="contains invalid numeric entries"):
        linear_sum_assignment(I)


# 定义测试函数 test_linear_sum_assignment_input_inf，测试线性和分配函数对正无穷值输入的处理
def test_linear_sum_assignment_input_inf():
    # 创建具有正无穷值的单位矩阵 I，并验证是否会引发 ValueError 异常
    I = np.identity(3)
    I[:, 0] = np.inf
    with pytest.raises(ValueError, match="cost matrix is infeasible"):
        linear_sum_assignment(I)


# 定义测试函数 test_constant_cost_matrix，测试线性和分配函数对常数成本矩阵的处理
def test_constant_cost_matrix():
    # 修复 GitHub 问题 #11602
    # 创建全为 1 的常数成本矩阵 C，并验证线性和分配函数的输出是否符合预期
    n = 8
    C = np.ones((n, n))
    row_ind, col_ind = linear_sum_assignment(C)
    assert_array_equal(row_ind, np.arange(n))
    assert_array_equal(col_ind, np.arange(n))


# 使用 pytest.mark.parametrize 装饰器定义测试函数 test_linear_sum_assignment_trivial_cost，测试线性和分配函数对简单成本矩阵的处理
@pytest.mark.parametrize('num_rows,num_cols', [(0, 0), (2, 0), (0, 3)])
def test_linear_sum_assignment_trivial_cost(num_rows, num_cols):
    # 创建空的 num_cols × num_rows 形状的数组 C，并验证线性和分配函数的输出是否符合预期
    C = np.empty(shape=(num_cols, num_rows))
    row_ind, col_ind = linear_sum_assignment(C)
    assert len(row_ind) == 0
    assert len(col_ind) == 0


# 使用 pytest.mark.parametrize 装饰器定义测试函数 test_linear_sum_assignment_small_inputs，测试线性和分配函数对小输入的处理
@pytest.mark.parametrize('sign,test_case', linear_sum_assignment_test_cases)
def test_linear_sum_assignment_small_inputs(sign, test_case):
    # 调用 linear_sum_assignment_assertions 函数进行断言验证
    linear_sum_assignment_assertions(
        linear_sum_assignment, np.array, sign, test_case)


# 定义测试函数 test_two_methods_give_same_result_on_many_sparse_inputs，测试两种方法在稀疏输入上是否给出相同的结果
def test_two_methods_give_same_result_on_many_sparse_inputs():
    # 与前面的测试不同，这里只断言两种方法给出的结果相同，不明确指定预期输出
    # 设定随机数种子，以便结果可重复
    np.random.seed(1234)
    # 执行100次测试，每次测试使用大小为100x100的稀疏矩阵，其中大约有6%的非零元素
    for _ in range(100):
        # 初始化标志，用于指示是否引发异常
        lsa_raises = False
        mwfbm_raises = False
        # 生成稀疏矩阵，数据元素为1到99之间的随机整数
        sparse = random(100, 100, density=0.06,
                        data_rvs=lambda size: np.random.randint(1, 100, size))
        # 在稠密矩阵中，用无穷大(np.inf)替换稀疏矩阵中的零，以便在后续算法中表示缺失边
        dense = np.full(sparse.shape, np.inf)
        dense[sparse.row, sparse.col] = sparse.data
        # 将稀疏矩阵转换为压缩稀疏行格式(csr)以便后续的线性求解
        sparse = sparse.tocsr()
        try:
            # 使用线性求解算法求解稠密矩阵的最小权重匹配
            row_ind, col_ind = linear_sum_assignment(dense)
            lsa_cost = dense[row_ind, col_ind].sum()
        except ValueError:
            # 如果出现值错误异常，设置标志
            lsa_raises = True
        try:
            # 使用最小权重全双部匹配算法求解稀疏矩阵的最小权重匹配
            row_ind, col_ind = min_weight_full_bipartite_matching(sparse)
            mwfbm_cost = sparse[row_ind, col_ind].sum()
        except ValueError:
            # 如果出现值错误异常，设置标志
            mwfbm_raises = True
        # 确保如果一个方法引发异常，另一个方法也应该引发异常
        assert lsa_raises == mwfbm_raises
        if not lsa_raises:
            # 如果没有引发异常，确保两种算法的计算结果相等
            assert lsa_cost == mwfbm_cost
```