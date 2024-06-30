# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_class_weight.py`

```
# 导入所需的库
import numpy as np
import pytest
from numpy.testing import assert_allclose

# 导入用于生成数据和模型的函数
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# 导入用于测试的工具函数和类
from sklearn.utils._testing import assert_almost_equal, assert_array_almost_equal
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.utils.fixes import CSC_CONTAINERS

# 定义测试函数：测试 compute_class_weight 函数
def test_compute_class_weight():
    # 创建样本标签数组
    y = np.asarray([2, 2, 2, 3, 3, 4])
    # 获取唯一的类别标签
    classes = np.unique(y)

    # 调用 compute_class_weight 函数计算类别权重
    cw = compute_class_weight("balanced", classes=classes, y=y)
    
    # 断言：样本的总数效果不变
    class_counts = np.bincount(y)[2:]  # 计算每个类别的样本数
    assert_almost_equal(np.dot(cw, class_counts), y.shape[0])
    # 断言：类别权重按顺序递增
    assert cw[0] < cw[1] < cw[2]

# 参数化测试函数：测试 compute_class_weight 函数在特定情况下的行为
@pytest.mark.parametrize(
    "y_type, class_weight, classes, err_msg",
    [
        (
            "numeric",
            "balanced",
            np.arange(4),
            "classes should have valid labels that are in y",
        ),
        # 对于特定问题的非回归
        (
            "numeric",
            {"label_not_present": 1.0},
            np.arange(4),
            r"The classes, \[0, 1, 2, 3\], are not in class_weight",
        ),
        (
            "numeric",
            "balanced",
            np.arange(2),
            "classes should include all valid labels",
        ),
        (
            "numeric",
            {0: 1.0, 1: 2.0},
            np.arange(2),
            "classes should include all valid labels",
        ),
        (
            "string",
            {"dogs": 3, "cat": 2},
            np.array(["dog", "cat"]),
            r"The classes, \['dog'\], are not in class_weight",
        ),
    ],
)
def test_compute_class_weight_not_present(y_type, class_weight, classes, err_msg):
    # 当 y 不包含所有类别标签时抛出错误
    y = (
        np.asarray([0, 0, 0, 1, 1, 2])
        if y_type == "numeric"
        else np.asarray(["dog", "cat", "dog"])
    )

    # 打印 y 的值
    print(y)
    # 使用 pytest.raises 检查是否引发了预期的 ValueError 异常，并匹配特定错误消息
    with pytest.raises(ValueError, match=err_msg):
        compute_class_weight(class_weight, classes=classes, y=y)

# 测试函数：测试使用字典指定类别权重时的 compute_class_weight 函数行为
def test_compute_class_weight_dict():
    # 定义类别数组
    classes = np.arange(3)
    # 定义类别权重字典
    class_weights = {0: 1.0, 1: 2.0, 2: 3.0}
    # 创建样本标签数组
    y = np.asarray([0, 0, 1, 2])

    # 调用 compute_class_weight 函数计算类别权重
    cw = compute_class_weight(class_weights, classes=classes, y=y)

    # 断言：当用户指定类别权重时，compute_class_weight 应直接返回这些权重
    assert_array_almost_equal(np.asarray([1.0, 2.0, 3.0]), cw)

    # 当指定的类别权重不在类别数组中时，应忽略该权重
    class_weights = {0: 1.0, 1: 2.0, 2: 3.0, 4: 1.5}
    cw = compute_class_weight(class_weights, classes=classes, y=y)
    assert_allclose([1.0, 2.0, 3.0], cw)

    # 当类别权重中的类别标签不在类别数组中时，应忽略该权重
    class_weights = {-1: 5.0, 0: 4.0, 1: 2.0, 2: 3.0}
    cw = compute_class_weight(class_weights, classes=classes, y=y)
    assert_allclose([4.0, 2.0, 3.0], cw)
def test_compute_class_weight_invariance():
    # 测试当 class_weight="balanced" 时，对于相同数量样本的类别不平衡是不变的。
    # 使用一个包含 100 个数据点的平衡二分类数据集进行测试。
    # 创建三个版本的数据集：一个类别 1 的样本被复制，导致类别 1 有 150 个点，类别 0 有 50 个点；
    # 另一个类别 0 的样本被复制，导致类别 1 有 50 个点，类别 0 有 150 个点；
    # 还有一个每个类别各有 100 个点（这是平衡的）。
    # 使用平衡的类别权重，所有三个版本应该得到相同的模型。

    X, y = make_blobs(centers=2, random_state=0)

    # 创建一个数据集，其中类别 1 被复制两次
    X_1 = np.vstack([X] + [X[y == 1]] * 2)
    y_1 = np.hstack([y] + [y[y == 1]] * 2)

    # 创建一个数据集，其中类别 0 被复制两次
    X_0 = np.vstack([X] + [X[y == 0]] * 2)
    y_0 = np.hstack([y] + [y[y == 0]] * 2)

    # 将整个数据集复制一次
    X_ = np.vstack([X] * 2)
    y_ = np.hstack([y] * 2)

    # 结果应该是相同的
    logreg1 = LogisticRegression(class_weight="balanced").fit(X_1, y_1)
    logreg0 = LogisticRegression(class_weight="balanced").fit(X_0, y_0)
    logreg = LogisticRegression(class_weight="balanced").fit(X_, y_)
    assert_array_almost_equal(logreg1.coef_, logreg0.coef_)
    assert_array_almost_equal(logreg.coef_, logreg0.coef_)


def test_compute_class_weight_balanced_negative():
    # 测试当标签为负数时 compute_class_weight 的行为。
    # 使用平衡的类别标签进行测试。

    classes = np.array([-2, -1, 0])
    y = np.asarray([-1, -1, 0, 0, -2, -2])

    cw = compute_class_weight("balanced", classes=classes, y=y)
    assert len(cw) == len(classes)
    assert_array_almost_equal(cw, np.array([1.0, 1.0, 1.0]))

    # 使用不平衡的类别标签进行测试
    y = np.asarray([-1, 0, 0, -2, -2, -2])

    cw = compute_class_weight("balanced", classes=classes, y=y)
    assert len(cw) == len(classes)
    class_counts = np.bincount(y + 2)
    assert_almost_equal(np.dot(cw, class_counts), y.shape[0])
    assert_array_almost_equal(cw, [2.0 / 3, 2.0, 1.0])


def test_compute_class_weight_balanced_unordered():
    # 测试当类别无序时 compute_class_weight 的行为。

    classes = np.array([1, 0, 3])
    y = np.asarray([1, 0, 0, 3, 3, 3])

    cw = compute_class_weight("balanced", classes=classes, y=y)
    class_counts = np.bincount(y)[classes]
    assert_almost_equal(np.dot(cw, class_counts), y.shape[0])
    assert_array_almost_equal(cw, [2.0, 1.0, 2.0 / 3])


def test_compute_class_weight_default():
    # 测试当某些类别没有指定权重时的行为。
    # 当前的行为是给没有指定权重的类别分配权重 1。

    y = np.asarray([2, 2, 2, 3, 3, 4])
    classes = np.unique(y)
    classes_len = len(classes)

    # 测试没有指定权重的情况
    cw = compute_class_weight(None, classes=classes, y=y)
    assert len(cw) == classes_len
    assert_array_almost_equal(cw, np.ones(3))
    # Tests for partly specified class weights using compute_class_weight function
    # 计算部分指定权重的类别的测试
    cw = compute_class_weight({2: 1.5}, classes=classes, y=y)
    # 计算根据指定权重 {2: 1.5} 计算得到的类别权重
    assert len(cw) == classes_len
    # 断言计算得到的类别权重数量与类别数目相等
    assert_array_almost_equal(cw, [1.5, 1.0, 1.0])
    # 断言计算得到的类别权重与预期的值 [1.5, 1.0, 1.0] 几乎相等
    
    cw = compute_class_weight({2: 1.5, 4: 0.5}, classes=classes, y=y)
    # 计算根据指定权重 {2: 1.5, 4: 0.5} 计算得到的类别权重
    assert len(cw) == classes_len
    # 断言计算得到的类别权重数量与类别数目相等
    assert_array_almost_equal(cw, [1.5, 1.0, 0.5])
    # 断言计算得到的类别权重与预期的值 [1.5, 1.0, 0.5] 几乎相等
def test_compute_sample_weight():
    # Test (and demo) compute_sample_weight.
    # Test with balanced classes
    y = np.asarray([1, 1, 1, 2, 2, 2])
    sample_weight = compute_sample_weight("balanced", y)
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Test with user-defined weights
    sample_weight = compute_sample_weight({1: 2, 2: 1}, y)
    assert_array_almost_equal(sample_weight, [2.0, 2.0, 2.0, 1.0, 1.0, 1.0])

    # Test with column vector of balanced classes
    y = np.asarray([[1], [1], [1], [2], [2], [2]])
    sample_weight = compute_sample_weight("balanced", y)
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Test with unbalanced classes
    y = np.asarray([1, 1, 1, 2, 2, 2, 3])
    sample_weight = compute_sample_weight("balanced", y)
    expected_balanced = np.array(
        [0.7777, 0.7777, 0.7777, 0.7777, 0.7777, 0.7777, 2.3333]
    )
    assert_array_almost_equal(sample_weight, expected_balanced, decimal=4)

    # Test with `None` weights
    sample_weight = compute_sample_weight(None, y)
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Test with multi-output of balanced classes
    y = np.asarray([[1, 0], [1, 0], [1, 0], [2, 1], [2, 1], [2, 1]])
    sample_weight = compute_sample_weight("balanced", y)
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Test with multi-output with user-defined weights
    y = np.asarray([[1, 0], [1, 0], [1, 0], [2, 1], [2, 1], [2, 1]])
    sample_weight = compute_sample_weight([{1: 2, 2: 1}, {0: 1, 1: 2}], y)
    assert_array_almost_equal(sample_weight, [2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

    # Test with multi-output of unbalanced classes
    y = np.asarray([[1, 0], [1, 0], [1, 0], [2, 1], [2, 1], [2, 1], [3, -1]])
    sample_weight = compute_sample_weight("balanced", y)
    assert_array_almost_equal(sample_weight, expected_balanced**2, decimal=3)


def test_compute_sample_weight_with_subsample():
    # Test compute_sample_weight with subsamples specified.
    # Test with balanced classes and all samples present
    y = np.asarray([1, 1, 1, 2, 2, 2])
    sample_weight = compute_sample_weight("balanced", y, indices=range(6))
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Test with column vector of balanced classes and all samples present
    y = np.asarray([[1], [1], [1], [2], [2], [2]])
    sample_weight = compute_sample_weight("balanced", y, indices=range(6))
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Test with a subsample
    y = np.asarray([1, 1, 1, 2, 2, 2])
    sample_weight = compute_sample_weight("balanced", y, indices=range(4))
    assert_array_almost_equal(sample_weight, [2.0 / 3, 2.0 / 3, 2.0 / 3, 2.0, 2.0, 2.0])

    # Test with a bootstrap subsample
    y = np.asarray([1, 1, 1, 2, 2, 2])
    # Remaining lines are truncated as they are not directly relevant for explaining the code.
    # 使用 compute_sample_weight 函数计算样本权重，使用"balanced"参数进行平衡处理
    sample_weight = compute_sample_weight("balanced", y, indices=[0, 1, 1, 2, 2, 3])
    # 预期的平衡权重数组
    expected_balanced = np.asarray([0.6, 0.6, 0.6, 3.0, 3.0, 3.0])
    # 断言样本权重与预期平衡权重数组几乎相等
    assert_array_almost_equal(sample_weight, expected_balanced)
    
    # 使用 bootstrap 子样本进行多输出测试
    y = np.asarray([[1, 0], [1, 0], [1, 0], [2, 1], [2, 1], [2, 1]])
    # 重新计算样本权重，indices参数指定了样本的索引
    sample_weight = compute_sample_weight("balanced", y, indices=[0, 1, 1, 2, 2, 3])
    # 断言样本权重与预期平衡权重数组的平方几乎相等
    assert_array_almost_equal(sample_weight, expected_balanced**2)
    
    # 测试包含缺失类别的情况
    y = np.asarray([1, 1, 1, 2, 2, 2, 3])
    # 计算样本权重，使用"balanced"参数，并指定indices范围
    sample_weight = compute_sample_weight("balanced", y, indices=range(6))
    # 断言样本权重与预期数组几乎相等
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    
    # 测试包含缺失类别的多输出情况
    y = np.asarray([[1, 0], [1, 0], [1, 0], [2, 1], [2, 1], [2, 1], [2, 2]])
    # 计算样本权重，使用"balanced"参数，并指定indices范围
    sample_weight = compute_sample_weight("balanced", y, indices=range(6))
    # 断言样本权重与预期数组几乎相等
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
# 使用 pytest.mark.parametrize 装饰器定义多个参数化测试用例，用于测试 test_compute_sample_weight_errors 函数
@pytest.mark.parametrize(
    "y_type, class_weight, indices, err_msg",
    [
        (
            "single-output",                            # 第一个测试用例：单输出情况
            {1: 2, 2: 1},                               # 类别权重为字典
            range(4),                                   # 索引范围为0到3
            "The only valid class_weight for subsampling is 'balanced'.",  # 预期的错误消息
        ),
        (
            "multi-output",                             # 第二个测试用例：多输出情况
            {1: 2, 2: 1},                               # 类别权重为字典
            None,                                       # 索引为None
            "For multi-output, class_weight should be a list of dicts, or the string",  # 预期的错误消息
        ),
        (
            "multi-output",                             # 第三个测试用例：多输出情况
            [{1: 2, 2: 1}],                             # 类别权重为列表中的字典
            None,                                       # 索引为None
            r"Got 1 element\(s\) while having 2 outputs",  # 预期的错误消息，使用原始字符串
        ),
    ],
)
def test_compute_sample_weight_errors(y_type, class_weight, indices, err_msg):
    """Test compute_sample_weight raises errors expected."""
    # 创建单输出和多输出的示例数据
    y_single_output = np.asarray([1, 1, 1, 2, 2, 2])
    y_multi_output = np.asarray([[1, 0], [1, 0], [1, 0], [2, 1], [2, 1], [2, 1]])

    # 根据 y_type 选择要使用的 y 数据
    y = y_single_output if y_type == "single-output" else y_multi_output

    # 使用 pytest 的 raises 断言，验证 compute_sample_weight 函数是否会抛出 ValueError 异常，并匹配 err_msg 中的消息
    with pytest.raises(ValueError, match=err_msg):
        compute_sample_weight(class_weight, y, indices=indices)


def test_compute_sample_weight_more_than_32():
    """Non-regression smoke test for #12146"""
    # 创建一个具有超过32个不同类别的示例 y 数据
    y = np.arange(50)
    indices = np.arange(50)  # 使用子采样索引
    # 调用 compute_sample_weight 函数计算权重，期望所有权重均为1
    weight = compute_sample_weight("balanced", y, indices=indices)
    # 使用 assert_array_almost_equal 断言验证权重是否与期望的一致
    assert_array_almost_equal(weight, np.ones(y.shape[0]))


def test_class_weight_does_not_contains_more_classes():
    """Check that class_weight can contain more labels than in y.

    Non-regression test for #22413
    """
    # 创建一个决策树分类器，指定类别权重为{0: 1, 1: 10, 2: 20}
    tree = DecisionTreeClassifier(class_weight={0: 1, 1: 10, 2: 20})

    # 调用 fit 方法拟合决策树，不期望抛出异常
    tree.fit([[0, 0, 1], [1, 0, 1], [1, 2, 0]], [0, 0, 1])


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_compute_sample_weight_sparse(csc_container):
    """Check that we can compute weight for sparse `y`."""
    # 创建稀疏矩阵 y 的示例数据
    y = csc_container(np.asarray([[0], [1], [1]]))
    # 使用 compute_sample_weight 函数计算样本权重，期望得到的权重列表与指定的列表接近
    sample_weight = compute_sample_weight("balanced", y)
    # 使用 assert_allclose 断言验证计算得到的样本权重与期望的接近
    assert_allclose(sample_weight, [1.5, 0.75, 0.75])
```