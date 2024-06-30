# `D:\src\scipysrc\scikit-learn\sklearn\tree\tests\test_tree.py`

```
"""
Testing for the tree module (sklearn.tree).
"""

import copy  # 导入深拷贝模块
import copyreg  # 导入pickle注册模块
import io  # 导入用于字节流操作的IO模块
import pickle  # 导入用于序列化和反序列化的pickle模块
import struct  # 导入用于处理结构化数据的struct模块
from itertools import chain, product  # 导入用于迭代操作的itertools模块中的chain和product函数

import joblib  # 导入joblib用于并行执行的模块
import numpy as np  # 导入数值计算的核心库numpy
import pytest  # 导入用于单元测试的pytest模块
from joblib.numpy_pickle import NumpyPickler  # 从joblib中导入用于numpy对象序列化的NumpyPickler类
from numpy.testing import assert_allclose  # 从numpy.testing中导入用于比较数组是否接近的函数assert_allclose

from sklearn import clone, datasets, tree  # 从sklearn中导入用于机器学习的函数和类
from sklearn.dummy import DummyRegressor  # 导入用于基本回归的DummyRegressor类
from sklearn.exceptions import NotFittedError  # 导入表示模型未拟合错误的异常类
from sklearn.impute import SimpleImputer  # 导入用于填补缺失值的SimpleImputer类
from sklearn.metrics import accuracy_score, mean_poisson_deviance, mean_squared_error  # 导入用于评估模型性能的指标
from sklearn.model_selection import train_test_split  # 导入用于数据集划分的函数
from sklearn.pipeline import make_pipeline  # 导入用于创建管道的make_pipeline函数
from sklearn.random_projection import _sparse_random_matrix  # 导入用于随机投影的稀疏矩阵生成函数
from sklearn.tree import (
    DecisionTreeClassifier,  # 导入决策树分类器
    DecisionTreeRegressor,  # 导入决策树回归器
    ExtraTreeClassifier,  # 导入额外树分类器
    ExtraTreeRegressor,  # 导入额外树回归器
)
from sklearn.tree._classes import (
    CRITERIA_CLF,  # 导入决策树分类器的评估标准
    CRITERIA_REG,  # 导入决策树回归器的评估标准
    DENSE_SPLITTERS,  # 导入密集数据集的分裂器
    SPARSE_SPLITTERS,  # 导入稀疏数据集的分裂器
)
from sklearn.tree._tree import (
    NODE_DTYPE,  # 导入节点数据类型
    TREE_LEAF,  # 导入表示树节点为叶子节点的常量
    TREE_UNDEFINED,  # 导入表示树节点未定义的常量
    _check_n_classes,  # 导入用于检查类别数量的函数
    _check_node_ndarray,  # 导入用于检查节点数据类型的函数
    _check_value_ndarray,  # 导入用于检查节点值数据类型的函数
)
from sklearn.tree._tree import Tree as CythonTree  # 导入Cython实现的Tree类
from sklearn.utils import compute_sample_weight  # 导入用于计算样本权重的函数
from sklearn.utils._testing import (
    assert_almost_equal,  # 导入用于比较值是否近似相等的函数
    assert_array_almost_equal,  # 导入用于比较数组是否元素近似相等的函数
    assert_array_equal,  # 导入用于比较数组是否相等的函数
    create_memmap_backed_data,  # 导入用于创建支持内存映射的数据的函数
    ignore_warnings,  # 导入用于忽略警告的装饰器
    skip_if_32bit,  # 导入用于在32位系统上跳过测试的装饰器
)
from sklearn.utils.estimator_checks import check_sample_weights_invariance  # 导入用于检查样本权重不变性的函数
from sklearn.utils.fixes import (
    _IS_32BIT,  # 导入用于检查是否为32位系统的常量
    COO_CONTAINERS,  # 导入COO格式的容器
    CSC_CONTAINERS,  # 导入CSC格式的容器
    CSR_CONTAINERS,  # 导入CSR格式的容器
)
from sklearn.utils.validation import check_random_state  # 导入用于检查随机状态的函数

CLF_CRITERIONS = ("gini", "log_loss")  # 定义分类器可用的评估标准列表
REG_CRITERIONS = ("squared_error", "absolute_error", "friedman_mse", "poisson")  # 定义回归器可用的评估标准列表

CLF_TREES = {
    "DecisionTreeClassifier": DecisionTreeClassifier,  # 决策树分类器类的映射
    "ExtraTreeClassifier": ExtraTreeClassifier,  # 额外树分类器类的映射
}

REG_TREES = {
    "DecisionTreeRegressor": DecisionTreeRegressor,  # 决策树回归器类的映射
    "ExtraTreeRegressor": ExtraTreeRegressor,  # 额外树回归器类的映射
}

ALL_TREES: dict = dict()  # 定义包含所有树模型的字典
ALL_TREES.update(CLF_TREES)  # 将分类器树模型加入到ALL_TREES字典中
ALL_TREES.update(REG_TREES)  # 将回归器树模型加入到ALL_TREES字典中

SPARSE_TREES = [
    "DecisionTreeClassifier",  # 包含稀疏数据集支持的决策树分类器
    "DecisionTreeRegressor",  # 包含稀疏数据集支持的决策树回归器
    "ExtraTreeClassifier",  # 包含稀疏数据集支持的额外树分类器
    "ExtraTreeRegressor",  # 包含稀疏数据集支持的额外树回归器
]

X_small = np.array(
    # 定义一个二维列表，包含多个子列表，每个子列表代表一个数据行
    [
        # 第一个子列表
        [0, 0, 4, 0, 0, 0, 1, -14, 0, -4, 0, 0, 0, 0],
        # 第二个子列表
        [0, 0, 5, 3, 0, -4, 0, 0, 1, -5, 0.2, 0, 4, 1],
        # 第三个子列表
        [-1, -1, 0, 0, -4.5, 0, 0, 2.1, 1, 0, 0, -4.5, 0, 1],
        # 第四个子列表
        [-1, -1, 0, -1.2, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 1],
        # 第五个子列表
        [-1, -1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 1],
        # 第六个子列表
        [-1, -2, 0, 4, -3, 10, 4, 0, -3.2, 0, 4, 3, -4, 1],
        # 第七个子列表
        [2.11, 0, -6, -0.5, 0, 11, 0, 0, -3.2, 6, 0.5, 0, -3, 1],
        # 第八个子列表
        [2.11, 0, -6, -0.5, 0, 11, 0, 0, -3.2, 6, 0, 0, -2, 1],
        # 第九个子列表
        [2.11, 8, -6, -0.5, 0, 11, 0, 0, -3.2, 6, 0, 0, -2, 1],
        # 第十个子列表
        [2.11, 8, -6, -0.5, 0, 11, 0, 0, -3.2, 6, 0.5, 0, -1, 0],
        # 第十一个子列表
        [2, 8, 5, 1, 0.5, -4, 10, 0, 1, -5, 3, 0, 2, 0],
        # 第十二个子列表
        [2, 0, 1, 1, 1, -1, 1, 0, 0, -2, 3, 0, 1, 0],
        # 第十三个子列表
        [2, 0, 1, 2, 3, -1, 10, 2, 0, -1, 1, 2, 2, 0],
        # 第十四个子列表
        [1, 1, 0, 2, 2, -1, 1, 2, 0, -5, 1, 2, 3, 0],
        # 第十五个子列表
        [3, 1, 0, 3, 0, -4, 10, 0, 1, -5, 3, 0, 3, 1],
        # 第十六个子列表
        [2.11, 8, -6, -0.5, 0, 1, 0, 0, -3.2, 6, 0.5, 0, -3, 1],
        # 第十七个子列表
        [2.11, 8, -6, -0.5, 0, 1, 0, 0, -3.2, 6, 1.5, 1, -1, -1],
        # 第十八个子列表
        [2.11, 8, -6, -0.5, 0, 10, 0, 0, -3.2, 6, 0.5, 0, -1, -1],
        # 第十九个子列表
        [2, 0, 5, 1, 0.5, -2, 10, 0, 1, -5, 3, 1, 0, -1],
        # 第二十个子列表
        [2, 0, 1, 1, 1, -2, 1, 0, 0, -2, 0, 0, 0, 1],
        # 第二十一个子列表
        [2, 1, 1, 1, 2, -1, 10, 2, 0, -1, 0, 2, 1, 1],
        # 第二十二个子列表
        [1, 1, 0, 0, 1, -3, 1, 2, 0, -5, 1, 2, 1, 1],
        # 第二十三个子列表
        [3, 1, 0, 1, 0, -4, 1, 0, 1, -2, 0, 0, 1, 0],
    ]
# 定义一个包含 23 个元素的列表，表示一个小样本的分类标签
y_small = [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]

# 定义一个包含 23 个元素的列表，表示一个小样本的回归目标值
y_small_reg = [
    1.0,
    2.1,
    1.2,
    0.05,
    10,
    2.4,
    3.1,
    1.01,
    0.01,
    2.98,
    3.1,
    1.1,
    0.0,
    1.2,
    2,
    11,
    0,
    0,
    4.5,
    0.201,
    1.06,
    0.9,
    0,
]

# 定义一个小的二维数据集，包含6个样本，每个样本有2个特征
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]

# 定义与X对应的目标标签
y = [-1, -1, -1, 1, 1, 1]

# 定义3个测试样本
T = [[-1, -1], [2, 2], [3, 2]]

# 定义3个测试样本的真实预期结果
true_result = [-1, 1, 1]

# 加载鸢尾花数据集，然后随机排列它
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# 加载糖尿病数据集，然后随机排列它
diabetes = datasets.load_diabetes()
perm = rng.permutation(diabetes.target.size)
diabetes.data = diabetes.data[perm]
diabetes.target = diabetes.target[perm]

# 加载手写数字数据集，然后随机排列它
digits = datasets.load_digits()
perm = rng.permutation(digits.target.size)
digits.data = digits.data[perm]
digits.target = digits.target[perm]

# 设置一个随机种子为0，生成一个多标签分类的小样本
random_state = check_random_state(0)
X_multilabel, y_multilabel = datasets.make_multilabel_classification(
    random_state=0, n_samples=30, n_features=10
)

# 生成一个稀疏正数矩阵，大多数元素为0，部分元素为随机正数
X_sparse_pos = random_state.uniform(size=(20, 5))
X_sparse_pos[X_sparse_pos <= 0.8] = 0.0

# 生成一个随机数组，元素范围在0到3之间
y_random = random_state.randint(0, 4, size=(20,))

# 生成一个稀疏混合矩阵，部分元素为随机正数，其余为0
X_sparse_mix = _sparse_random_matrix(20, 10, density=0.25, random_state=0).toarray()

# 定义多个数据集字典，每个数据集包含特征矩阵X和目标y
DATASETS = {
    "iris": {"X": iris.data, "y": iris.target},
    "diabetes": {"X": diabetes.data, "y": diabetes.target},
    "digits": {"X": digits.data, "y": digits.target},
    "toy": {"X": X, "y": y},
    "clf_small": {"X": X_small, "y": y_small},
    "reg_small": {"X": X_small, "y": y_small_reg},
    "multilabel": {"X": X_multilabel, "y": y_multilabel},
    "sparse-pos": {"X": X_sparse_pos, "y": y_random},
    "sparse-neg": {"X": -X_sparse_pos, "y": y_random},
    "sparse-mix": {"X": X_sparse_mix, "y": y_random},
    "zeros": {"X": np.zeros((20, 3)), "y": y_random},
}

# 定义一个函数，用于比较两棵树的结构是否相同
def assert_tree_equal(d, s, message):
    # 断言两棵树节点数相等
    assert (
        s.node_count == d.node_count
    ), "{0}: inequal number of node ({1} != {2})".format(
        message, s.node_count, d.node_count
    )

    # 断言两棵树的叶子节点数相等
    assert_array_equal(
        d.children_right, s.children_right, message + ": inequal children_right"
    )

    # 断言两棵树的左子节点数相等
    assert_array_equal(
        d.children_left, s.children_left, message + ": inequal children_left"
    )

    # 获取内部节点和叶子节点的索引
    external = d.children_right == TREE_LEAF
    internal = np.logical_not(external)

    # 断言两棵树的内部节点特征值相等
    assert_array_equal(
        d.feature[internal], s.feature[internal], message + ": inequal features"
    )

    # 断言两棵树的内部节点阈值相等
    assert_array_equal(
        d.threshold[internal], s.threshold[internal], message + ": inequal threshold"
    )

    # 断言两棵树的节点样本数之和相等
    assert_array_equal(
        d.n_node_samples.sum(),
        s.n_node_samples.sum(),
        message + ": inequal sum(n_node_samples)",
    )
    # 检查两个数组是否相等，用于节点样本数（n_node_samples）
    assert_array_equal(
        d.n_node_samples, s.n_node_samples, message + ": inequal n_node_samples"
    )
    
    # 检查两个数值是否几乎相等，用于节点不纯度（impurity）
    assert_almost_equal(
        d.impurity, s.impurity, err_msg=message + ": inequal impurity"
    )
    
    # 检查两个数组的每个元素是否几乎相等，用于外部索引对应的节点值（value[external]）
    assert_array_almost_equal(
        d.value[external], s.value[external], err_msg=message + ": inequal value"
    )
def test_classification_toy():
    # 检查在一个玩具数据集上的分类。
    for name, Tree in CLF_TREES.items():
        # 实例化分类器对象
        clf = Tree(random_state=0)
        # 使用数据集 X, y 进行训练
        clf.fit(X, y)
        # 断言预测结果与真实结果的一致性
        assert_array_equal(clf.predict(T), true_result, "Failed with {0}".format(name))

        # 使用带有限制最大特征数的分类器对象
        clf = Tree(max_features=1, random_state=1)
        # 使用数据集 X, y 进行训练
        clf.fit(X, y)
        # 断言预测结果与真实结果的一致性
        assert_array_equal(clf.predict(T), true_result, "Failed with {0}".format(name))


def test_weighted_classification_toy():
    # 检查在一个带权重的玩具数据集上的分类。
    for name, Tree in CLF_TREES.items():
        # 实例化分类器对象
        clf = Tree(random_state=0)

        # 使用权重为全1数组的数据集 X, y 进行训练
        clf.fit(X, y, sample_weight=np.ones(len(X)))
        # 断言预测结果与真实结果的一致性
        assert_array_equal(clf.predict(T), true_result, "Failed with {0}".format(name))

        # 使用权重为全0.5数组的数据集 X, y 进行训练
        clf.fit(X, y, sample_weight=np.full(len(X), 0.5))
        # 断言预测结果与真实结果的一致性
        assert_array_equal(clf.predict(T), true_result, "Failed with {0}".format(name))


@pytest.mark.parametrize("Tree", REG_TREES.values())
@pytest.mark.parametrize("criterion", REG_CRITERIONS)
def test_regression_toy(Tree, criterion):
    # 检查在一个玩具数据集上的回归。
    if criterion == "poisson":
        # 将目标变为正数，但不影响原始的 y 和 true_result
        a = np.abs(np.min(y)) + 1
        y_train = np.array(y) + a
        y_test = np.array(true_result) + a
    else:
        y_train = y
        y_test = true_result

    # 实例化回归器对象
    reg = Tree(criterion=criterion, random_state=1)
    # 使用数据集 X, y_train 进行训练
    reg.fit(X, y_train)
    # 断言预测结果与测试数据集 y_test 的一致性
    assert_allclose(reg.predict(T), y_test)

    # 使用带有限制最大特征数的分类器对象
    clf = Tree(criterion=criterion, max_features=1, random_state=1)
    # 使用数据集 X, y_train 进行训练
    clf.fit(X, y_train)
    # 断言预测结果与测试数据集 y_test 的一致性
    assert_allclose(reg.predict(T), y_test)


def test_xor():
    # 检查在一个异或问题上的表现
    y = np.zeros((10, 10))
    y[:5, :5] = 1
    y[5:, 5:] = 1

    # 生成网格坐标
    gridx, gridy = np.indices(y.shape)

    # 将网格坐标转换为二维数组
    X = np.vstack([gridx.ravel(), gridy.ravel()]).T
    y = y.ravel()

    for name, Tree in CLF_TREES.items():
        # 实例化分类器对象
        clf = Tree(random_state=0)
        # 使用数据集 X, y 进行训练
        clf.fit(X, y)
        # 断言分类器在整个数据集上的准确率为1.0
        assert clf.score(X, y) == 1.0, "Failed with {0}".format(name)

        # 使用带有限制最大特征数的分类器对象
        clf = Tree(random_state=0, max_features=1)
        # 使用数据集 X, y 进行训练
        clf.fit(X, y)
        # 断言分类器在整个数据集上的准确率为1.0
        assert clf.score(X, y) == 1.0, "Failed with {0}".format(name)


def test_iris():
    # 检查在鸢尾花数据集上的一致性
    for (name, Tree), criterion in product(CLF_TREES.items(), CLF_CRITERIONS):
        # 实例化分类器对象
        clf = Tree(criterion=criterion, random_state=0)
        # 使用鸢尾花数据集进行训练
        clf.fit(iris.data, iris.target)
        # 计算分类器在整个数据集上的准确率
        score = accuracy_score(clf.predict(iris.data), iris.target)
        # 断言分类器在整个数据集上的准确率大于0.9
        assert score > 0.9, "Failed with {0}, criterion = {1} and score = {2}".format(
            name, criterion, score
        )

        # 使用带有限制最大特征数的分类器对象
        clf = Tree(criterion=criterion, max_features=2, random_state=0)
        # 使用鸢尾花数据集进行训练
        clf.fit(iris.data, iris.target)
        # 计算分类器在整个数据集上的准确率
        score = accuracy_score(clf.predict(iris.data), iris.target)
        # 断言分类器在整个数据集上的准确率大于0.5
        assert score > 0.5, "Failed with {0}, criterion = {1} and score = {2}".format(
            name, criterion, score
        )
@pytest.mark.parametrize("criterion", REG_CRITERIONS)
def test_diabetes_overfit(name, Tree, criterion):
    # 检查在糖尿病数据集上过拟合树的一致性
    # 由于树会过拟合，我们期望均方误差为0
    reg = Tree(criterion=criterion, random_state=0)
    reg.fit(diabetes.data, diabetes.target)
    score = mean_squared_error(diabetes.target, reg.predict(diabetes.data))
    assert score == pytest.approx(
        0
    ), f"Failed with {name}, criterion = {criterion} and score = {score}"


@skip_if_32bit
@pytest.mark.parametrize("name, Tree", REG_TREES.items())
@pytest.mark.parametrize(
    "criterion, max_depth, metric, max_loss",
    [
        ("squared_error", 15, mean_squared_error, 60),
        ("absolute_error", 20, mean_squared_error, 60),
        ("friedman_mse", 15, mean_squared_error, 60),
        ("poisson", 15, mean_poisson_deviance, 30),
    ],
)
def test_diabetes_underfit(name, Tree, criterion, max_depth, metric, max_loss):
    # 检查在深度和特征数目受限制时树的一致性
    reg = Tree(criterion=criterion, max_depth=max_depth, max_features=6, random_state=0)
    reg.fit(diabetes.data, diabetes.target)
    loss = metric(diabetes.target, reg.predict(diabetes.data))
    assert 0 < loss < max_loss


def test_probability():
    # 使用DecisionTreeClassifier预测概率。
    for name, Tree in CLF_TREES.items():
        clf = Tree(max_depth=1, max_features=1, random_state=42)
        clf.fit(iris.data, iris.target)

        prob_predict = clf.predict_proba(iris.data)
        assert_array_almost_equal(
            np.sum(prob_predict, 1),
            np.ones(iris.data.shape[0]),
            err_msg="Failed with {0}".format(name),
        )
        assert_array_equal(
            np.argmax(prob_predict, 1),
            clf.predict(iris.data),
            err_msg="Failed with {0}".format(name),
        )
        assert_almost_equal(
            clf.predict_proba(iris.data),
            np.exp(clf.predict_log_proba(iris.data)),
            8,
            err_msg="Failed with {0}".format(name),
        )


def test_arrayrepr():
    # 检查数组的表示形式。
    # 检查大小调整
    X = np.arange(10000)[:, np.newaxis]
    y = np.arange(10000)

    for name, Tree in REG_TREES.items():
        reg = Tree(max_depth=None, random_state=0)
        reg.fit(X, y)


def test_pure_set():
    # 检查当y是纯净的时候。
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
    y = [1, 1, 1, 1, 1, 1]

    for name, TreeClassifier in CLF_TREES.items():
        clf = TreeClassifier(random_state=0)
        clf.fit(X, y)
        assert_array_equal(clf.predict(X), y, err_msg="Failed with {0}".format(name))

    for name, TreeRegressor in REG_TREES.items():
        reg = TreeRegressor(random_state=0)
        reg.fit(X, y)
        assert_almost_equal(reg.predict(X), y, err_msg="Failed with {0}".format(name))


def test_numerical_stability():
    # 检查数值稳定性。
    # 检查数值稳定性。

    # 创建包含浮点数的二维数组 X，用于训练模型
    X = np.array(
        [
            [152.08097839, 140.40744019, 129.75102234, 159.90493774],
            [142.50700378, 135.81935120, 117.82884979, 162.75781250],
            [127.28772736, 140.40744019, 129.75102234, 159.90493774],
            [132.37025452, 143.71923828, 138.35694885, 157.84558105],
            [103.10237122, 143.71928406, 138.35696411, 157.84559631],
            [127.71276855, 143.71923828, 138.35694885, 157.84558105],
            [120.91514587, 140.40744019, 129.75102234, 159.90493774],
        ]
    )

    # 创建包含浮点数的一维数组 y，用作 X 的目标输出
    y = np.array([1.0, 0.70209277, 0.53896582, 0.0, 0.90914464, 0.48026916, 0.49622521])

    # 设置 numpy 的错误状态，使所有错误都会抛出异常
    with np.errstate(all="raise"):
        # 对于 REG_TREES 字典中的每个键值对，分别取出键名和对应的值（通常是一个树模型）
        for name, Tree in REG_TREES.items():
            # 使用指定的随机种子创建树模型对象 reg
            reg = Tree(random_state=0)
            # 使用 X, y 来拟合正向数据
            reg.fit(X, y)
            # 使用 X, -y 来拟合 y 的反向数据
            reg.fit(X, -y)
            # 使用 -X, y 来拟合 X 的反向数据
            reg.fit(-X, y)
            # 使用 -X, -y 来拟合 X 和 y 的反向数据
            reg.fit(-X, -y)
# 检查变量重要性的测试函数
def test_importances():
    # 创建一个具有特定特征和样本数量的分类数据集
    X, y = datasets.make_classification(
        n_samples=5000,
        n_features=10,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        shuffle=False,
        random_state=0,
    )

    # 遍历分类器字典中的每个分类器名称和类
    for name, Tree in CLF_TREES.items():
        # 使用特定种子创建分类器实例
        clf = Tree(random_state=0)

        # 使用数据集 X, y 对分类器进行拟合
        clf.fit(X, y)
        
        # 获取特征重要性
        importances = clf.feature_importances_
        
        # 统计重要性大于0.1的特征数量
        n_important = np.sum(importances > 0.1)

        # 断言特征重要性数组的长度为10，若不符合则抛出异常
        assert importances.shape[0] == 10, "Failed with {0}".format(name)
        
        # 断言重要性大于0.1的特征数量为3，若不符合则抛出异常
        assert n_important == 3, "Failed with {0}".format(name)

    # 检查 iris 数据集上的分类器是否具有相同的特征重要性
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(iris.data, iris.target)
    
    # 使用与前一个分类器相同的参数创建另一个分类器实例
    clf2 = DecisionTreeClassifier(random_state=0, max_leaf_nodes=len(iris.data))
    clf2.fit(iris.data, iris.target)

    # 断言两个分类器的特征重要性数组相等，若不符合则抛出异常
    assert_array_equal(clf.feature_importances_, clf2.feature_importances_)


# 检查特征重要性未拟合时是否引发 ValueError 异常的测试函数
def test_importances_raises():
    # 创建一个决策树分类器实例
    clf = DecisionTreeClassifier()
    
    # 使用 pytest 检查是否在调用未拟合前获取特征重要性时引发 ValueError 异常
    with pytest.raises(ValueError):
        getattr(clf, "feature_importances_")


# 检查基尼系数与均方误差是否等效于二元输出变量的测试函数
def test_importances_gini_equal_squared_error():
    # 创建一个具有特定特征和样本数量的分类数据集
    X, y = datasets.make_classification(
        n_samples=2000,
        n_features=10,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        shuffle=False,
        random_state=0,
    )

    # 使用 gini 指数创建一个深度限制为5的决策树分类器实例并拟合数据集 X, y
    clf = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=0).fit(
        X, y
    )
    
    # 使用均方误差（squared_error）创建一个深度限制为5的决策树回归器实例并拟合数据集 X, y
    reg = DecisionTreeRegressor(
        criterion="squared_error", max_depth=5, random_state=0
    ).fit(X, y)

    # 断言两个模型的特征重要性数组近似相等，若不符合则抛出异常
    assert_almost_equal(clf.feature_importances_, reg.feature_importances_)
    
    # 断言两个模型的决策树特征数组相等，若不符合则抛出异常
    assert_array_equal(clf.tree_.feature, reg.tree_.feature)
    
    # 断言两个模型的左子节点数组相等，若不符合则抛出异常
    assert_array_equal(clf.tree_.children_left, reg.tree_.children_left)
    
    # 断言两个模型的右子节点数组相等，若不符合则抛出异常
    assert_array_equal(clf.tree_.children_right, reg.tree_.children_right)
    
    # 断言两个模型的节点样本数数组相等，若不符合则抛出异常
    assert_array_equal(clf.tree_.n_node_samples, reg.tree_.n_node_samples)


# 检查 max_features 参数的测试函数
def test_max_features():
    # Check max_features.
    # 遍历 ALL_TREES 字典中的每个树估计器的名称和类
    for name, TreeEstimator in ALL_TREES.items():
        # 使用当前树估计器类创建一个估计器对象，设置最大特征数为平方根形式
        est = TreeEstimator(max_features="sqrt")
        # 使用鸢尾花数据 iris.data 和目标 iris.target 来训练估计器
        est.fit(iris.data, iris.target)
        # 断言估计器的最大特征数等于输入特征的平方根取整
        assert est.max_features_ == int(np.sqrt(iris.data.shape[1]))

        # 使用当前树估计器类创建另一个估计器对象，设置最大特征数为对数形式
        est = TreeEstimator(max_features="log2")
        # 使用相同的数据训练估计器
        est.fit(iris.data, iris.target)
        # 断言估计器的最大特征数等于输入特征的对数取整
        assert est.max_features_ == int(np.log2(iris.data.shape[1]))

        # 使用当前树估计器类创建另一个估计器对象，设置最大特征数为1
        est = TreeEstimator(max_features=1)
        # 使用相同的数据训练估计器
        est.fit(iris.data, iris.target)
        # 断言估计器的最大特征数等于1
        assert est.max_features_ == 1

        # 使用当前树估计器类创建另一个估计器对象，设置最大特征数为3
        est = TreeEstimator(max_features=3)
        # 使用相同的数据训练估计器
        est.fit(iris.data, iris.target)
        # 断言估计器的最大特征数等于3
        assert est.max_features_ == 3

        # 使用当前树估计器类创建另一个估计器对象，设置最大特征数为0.01
        est = TreeEstimator(max_features=0.01)
        # 使用相同的数据训练估计器
        est.fit(iris.data, iris.target)
        # 断言估计器的最大特征数等于1（最小值为1）
        assert est.max_features_ == 1

        # 使用当前树估计器类创建另一个估计器对象，设置最大特征数为0.5
        est = TreeEstimator(max_features=0.5)
        # 使用相同的数据训练估计器
        est.fit(iris.data, iris.target)
        # 断言估计器的最大特征数等于输入特征数的一半取整
        assert est.max_features_ == int(0.5 * iris.data.shape[1])

        # 使用当前树估计器类创建另一个估计器对象，设置最大特征数为1.0
        est = TreeEstimator(max_features=1.0)
        # 使用相同的数据训练估计器
        est.fit(iris.data, iris.target)
        # 断言估计器的最大特征数等于输入特征数
        assert est.max_features_ == iris.data.shape[1]

        # 使用当前树估计器类创建另一个估计器对象，设置最大特征数为None（不限制）
        est = TreeEstimator(max_features=None)
        # 使用相同的数据训练估计器
        est.fit(iris.data, iris.target)
        # 断言估计器的最大特征数等于输入特征数（不限制情况下）
        assert est.max_features_ == iris.data.shape[1]
def test_error():
    # Test that it gives proper exception on deficient input.
    for name, TreeEstimator in CLF_TREES.items():
        # 创建一个树估计器实例
        est = TreeEstimator()
        # 预测操作在拟合之前应该引发 NotFittedError 异常
        with pytest.raises(NotFittedError):
            est.predict_proba(X)

        # 拟合树估计器
        est.fit(X, y)
        # 错误的特征形状应该引发 ValueError 异常
        X2 = [[-2, -1, 1]]
        with pytest.raises(ValueError):
            est.predict_proba(X2)

        # 错误的标签维度应该引发 ValueError 异常
        est = TreeEstimator()
        y2 = y[:-1]
        with pytest.raises(ValueError):
            est.fit(X, y2)

        # 使用非连续数组测试
        Xf = np.asfortranarray(X)
        est = TreeEstimator()
        est.fit(Xf, y)
        # 验证预测结果的近似相等性
        assert_almost_equal(est.predict(T), true_result)

        # 拟合之前预测应该引发 NotFittedError 异常
        est = TreeEstimator()
        with pytest.raises(NotFittedError):
            est.predict(T)

        # 使用不同维度的向量预测应该引发 ValueError 异常
        est.fit(X, y)
        t = np.asarray(T)
        with pytest.raises(ValueError):
            est.predict(t[:, 1:])

        # 错误的样本形状应该引发 ValueError 异常
        Xt = np.array(X).T

        est = TreeEstimator()
        est.fit(np.dot(X, Xt), y)
        with pytest.raises(ValueError):
            est.predict(X)
        with pytest.raises(ValueError):
            est.apply(X)

        clf = TreeEstimator()
        clf.fit(X, y)
        with pytest.raises(ValueError):
            clf.predict(Xt)
        with pytest.raises(ValueError):
            clf.apply(Xt)

        # 拟合之前应用应该引发 NotFittedError 异常
        est = TreeEstimator()
        with pytest.raises(NotFittedError):
            est.apply(T)

    # 对于 Poisson 分裂标准，目标值非正数应该引发 ValueError 异常
    est = DecisionTreeRegressor(criterion="poisson")
    with pytest.raises(ValueError, match="y is not positive.*Poisson"):
        est.fit([[0, 1, 2]], [0, 0, 0])
    with pytest.raises(ValueError, match="Some.*y are negative.*Poisson"):
        est.fit([[0, 1, 2]], [5, -0.1, 2])


def test_min_samples_split():
    """Test min_samples_split parameter"""
    X = np.asfortranarray(iris.data, dtype=tree._tree.DTYPE)
    y = iris.target

    # test both DepthFirstTreeBuilder and BestFirstTreeBuilder
    # by setting max_leaf_nodes
    # 使用 product 函数生成两组参数组合：(None, 1000) 和 ALL_TREES 字典中所有键名
    for max_leaf_nodes, name in product((None, 1000), ALL_TREES.keys()):
        # 根据名称获取相应的树估算器类
        TreeEstimator = ALL_TREES[name]

        # 测试整数类型参数
        # 创建树估算器对象，设置参数 min_samples_split=10, max_leaf_nodes=max_leaf_nodes, random_state=0
        est = TreeEstimator(
            min_samples_split=10, max_leaf_nodes=max_leaf_nodes, random_state=0
        )
        # 使用数据集 X 和标签 y 进行拟合
        est.fit(X, y)
        # 统计每个节点的样本数，-1 表示该节点是叶子节点
        node_samples = est.tree_.n_node_samples[est.tree_.children_left != -1]

        # 断言：所有节点中最小样本数应大于9，否则输出错误信息
        assert np.min(node_samples) > 9, "Failed with {0}".format(name)

        # 测试浮点类型参数
        # 创建树估算器对象，设置参数 min_samples_split=0.2, max_leaf_nodes=max_leaf_nodes, random_state=0
        est = TreeEstimator(
            min_samples_split=0.2, max_leaf_nodes=max_leaf_nodes, random_state=0
        )
        # 使用数据集 X 和标签 y 进行拟合
        est.fit(X, y)
        # 统计每个节点的样本数，-1 表示该节点是叶子节点
        node_samples = est.tree_.n_node_samples[est.tree_.children_left != -1]

        # 断言：所有节点中最小样本数应大于9，否则输出错误信息
        assert np.min(node_samples) > 9, "Failed with {0}".format(name)
# 测试函数，检查叶子节点是否包含超过 min_samples_leaf 个训练样本
def test_min_samples_leaf():
    # 将 iris 数据集转换为 Fortran 顺序的数组 X，数据类型为 tree._tree.DTYPE
    X = np.asfortranarray(iris.data, dtype=tree._tree.DTYPE)
    y = iris.target

    # 使用 DepthFirstTreeBuilder 和 BestFirstTreeBuilder 进行测试，设置 max_leaf_nodes 参数
    for max_leaf_nodes, name in product((None, 1000), ALL_TREES.keys()):
        TreeEstimator = ALL_TREES[name]

        # 测试整数参数
        est = TreeEstimator(
            min_samples_leaf=5, max_leaf_nodes=max_leaf_nodes, random_state=0
        )
        est.fit(X, y)
        out = est.tree_.apply(X)
        node_counts = np.bincount(out)
        # 去除内部节点，计算叶子节点的样本数
        leaf_count = node_counts[node_counts != 0]
        assert np.min(leaf_count) > 4, "Failed with {0}".format(name)

        # 测试浮点数参数
        est = TreeEstimator(
            min_samples_leaf=0.1, max_leaf_nodes=max_leaf_nodes, random_state=0
        )
        est.fit(X, y)
        out = est.tree_.apply(X)
        node_counts = np.bincount(out)
        # 去除内部节点，计算叶子节点的样本数
        leaf_count = node_counts[node_counts != 0]
        assert np.min(leaf_count) > 4, "Failed with {0}".format(name)


# 检查函数，测试叶子节点是否包含至少 min_weight_fraction_leaf 比例的训练集样本
def check_min_weight_fraction_leaf(name, datasets, sparse_container=None):
    """Test if leaves contain at least min_weight_fraction_leaf of the
    training set"""
    # 载入指定数据集的特征矩阵 X，转换为 np.float32 类型
    X = DATASETS[datasets]["X"].astype(np.float32)
    if sparse_container is not None:
        X = sparse_container(X)
    y = DATASETS[datasets]["y"]

    # 生成权重数组，总权重为所有权重之和
    weights = rng.rand(X.shape[0])
    total_weight = np.sum(weights)

    TreeEstimator = ALL_TREES[name]

    # 使用 DepthFirstTreeBuilder 和 BestFirstTreeBuilder 进行测试，设置 max_leaf_nodes 和 frac 参数
    for max_leaf_nodes, frac in product((None, 1000), np.linspace(0, 0.5, 6)):
        est = TreeEstimator(
            min_weight_fraction_leaf=frac, max_leaf_nodes=max_leaf_nodes, random_state=0
        )
        est.fit(X, y, sample_weight=weights)

        # 根据是否稀疏，选择合适的数据结构进行预测
        if sparse_container is not None:
            out = est.tree_.apply(X.tocsr())
        else:
            out = est.tree_.apply(X)

        # 计算节点权重分布，去除内部节点，计算叶子节点的权重
        node_weights = np.bincount(out, weights=weights)
        leaf_weights = node_weights[node_weights != 0]
        assert (
            np.min(leaf_weights) >= total_weight * est.min_weight_fraction_leaf
        ), "Failed with {0} min_weight_fraction_leaf={1}".format(
            name, est.min_weight_fraction_leaf
        )

    # 测试未传入权重的情况，总权重为样本数
    total_weight = X.shape[0]
    # 遍历 max_leaf_nodes 和 frac 组合的笛卡尔积
    for max_leaf_nodes, frac in product((None, 1000), np.linspace(0, 0.5, 6)):
        # 使用 TreeEstimator 创建一个决策树估计器对象 est
        est = TreeEstimator(
            min_weight_fraction_leaf=frac, max_leaf_nodes=max_leaf_nodes, random_state=0
        )
        # 使用数据集 X 和标签 y 来训练估计器 est
        est.fit(X, y)

        # 如果 sparse_container 不为 None，则将稀疏矩阵 X 转换为 CSR 格式后应用决策树
        if sparse_container is not None:
            out = est.tree_.apply(X.tocsr())
        else:
            # 否则直接应用决策树到 X
            out = est.tree_.apply(X)

        # 统计各个节点的出现次数，并生成节点权重数组 node_weights
        node_weights = np.bincount(out)
        # 获取叶子节点的权重数组 leaf_weights，排除掉权重为零的内部节点
        leaf_weights = node_weights[node_weights != 0]
        
        # 断言确保所有叶子节点的最小权重不小于总权重乘以估计器的最小叶子权重分数
        assert (
            np.min(leaf_weights) >= total_weight * est.min_weight_fraction_leaf
        ), "Failed with {0} min_weight_fraction_leaf={1}".format(
            name, est.min_weight_fraction_leaf
        )
# 使用 pytest 的 parametrize 装饰器，按照 ALL_TREES 中的每个树的名称依次运行测试函数
@pytest.mark.parametrize("name", ALL_TREES)
def test_min_weight_fraction_leaf_on_dense_input(name):
    # 调用函数 check_min_weight_fraction_leaf，测试密集输入情况下的最小权重分数叶子设置对于 iris 数据集的效果
    check_min_weight_fraction_leaf(name, "iris")


# 使用 pytest 的 parametrize 装饰器，对于稀疏树（SPARSE_TREES）中的每个树名称和 CSC_CONTAINERS 中的每个稀疏容器，依次运行测试函数
@pytest.mark.parametrize("name", SPARSE_TREES)
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_min_weight_fraction_leaf_on_sparse_input(name, csc_container):
    # 调用函数 check_min_weight_fraction_leaf，测试稀疏输入情况下的最小权重分数叶子设置对于 multilabel 数据集的效果，可选使用稀疏容器 csc_container
    check_min_weight_fraction_leaf(name, "multilabel", sparse_container=csc_container)


# 定义函数，测试 min_weight_fraction_leaf 和 min_samples_leaf 之间的交互作用，当未在 fit 中提供 sample_weights 时
def check_min_weight_fraction_leaf_with_min_samples_leaf(
    name, datasets, sparse_container=None
):
    """Test the interaction between min_weight_fraction_leaf and
    min_samples_leaf when sample_weights is not provided in fit."""
    # 获取数据集 X 和 y
    X = DATASETS[datasets]["X"].astype(np.float32)
    if sparse_container is not None:
        X = sparse_container(X)
    y = DATASETS[datasets]["y"]

    # 总权重数为样本数
    total_weight = X.shape[0]
    # 获取指定名称的树估算器类
    TreeEstimator = ALL_TREES[name]

    # 对于每个 max_leaf_nodes 和 frac 的组合，进行测试
    for max_leaf_nodes, frac in product((None, 1000), np.linspace(0, 0.5, 3)):
        # 测试整数 min_samples_leaf
        est = TreeEstimator(
            min_weight_fraction_leaf=frac,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=5,
            random_state=0,
        )
        # 使用数据拟合估算器
        est.fit(X, y)

        if sparse_container is not None:
            out = est.tree_.apply(X.tocsr())
        else:
            out = est.tree_.apply(X)

        # 计算每个节点的权重
        node_weights = np.bincount(out)
        # 去除内部节点，得到叶子节点权重
        leaf_weights = node_weights[node_weights != 0]
        # 断言叶子节点的最小权重大于等于总权重乘以 min_weight_fraction_leaf 和 5 中的较大者
        assert np.min(leaf_weights) >= max(
            (total_weight * est.min_weight_fraction_leaf), 5
        ), "Failed with {0} min_weight_fraction_leaf={1}, min_samples_leaf={2}".format(
            name, est.min_weight_fraction_leaf, est.min_samples_leaf
        )

    # 再次对于每个 max_leaf_nodes 和 frac 的组合，进行测试
    for max_leaf_nodes, frac in product((None, 1000), np.linspace(0, 0.5, 3)):
        # 测试浮点数 min_samples_leaf
        est = TreeEstimator(
            min_weight_fraction_leaf=frac,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=0.1,
            random_state=0,
        )
        # 使用数据拟合估算器
        est.fit(X, y)

        if sparse_container is not None:
            out = est.tree_.apply(X.tocsr())
        else:
            out = est.tree_.apply(X)

        # 计算每个节点的权重
        node_weights = np.bincount(out)
        # 去除内部节点，得到叶子节点权重
        leaf_weights = node_weights[node_weights != 0]
        # 断言叶子节点的最小权重大于等于总权重乘以 min_weight_fraction_leaf 和 min_samples_leaf 中的较大者
        assert np.min(leaf_weights) >= max(
            (total_weight * est.min_weight_fraction_leaf),
            (total_weight * est.min_samples_leaf),
        ), "Failed with {0} min_weight_fraction_leaf={1}, min_samples_leaf={2}".format(
            name, est.min_weight_fraction_leaf, est.min_samples_leaf
        )


# 使用 pytest 的 parametrize 装饰器，按照 ALL_TREES 中的每个树的名称依次运行测试函数
@pytest.mark.parametrize("name", ALL_TREES)
def test_min_weight_fraction_leaf_with_min_samples_leaf_on_dense_input(name):
    # 调用函数 check_min_weight_fraction_leaf_with_min_samples_leaf，测试密集输入情况下的 min_weight_fraction_leaf 和 min_samples_leaf 的效果
    check_min_weight_fraction_leaf_with_min_samples_leaf(name, "iris")


# 使用 pytest 的 parametrize 装饰器，对于稀疏树（SPARSE_TREES）中的每个树名称和 CSC_CONTAINERS 中的每个稀疏容器，依次运行测试函数
@pytest.mark.parametrize("name", SPARSE_TREES)
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
# 定义一个函数，用于测试在稀疏输入上使用最小叶子样本数和最小权重分数叶子的影响
def test_min_weight_fraction_leaf_with_min_samples_leaf_on_sparse_input(
    name, csc_container
):
    # 调用检查函数，验证在多标签情况下，使用稀疏容器进行最小权重分数叶子和最小样本叶子的检查
    check_min_weight_fraction_leaf_with_min_samples_leaf(
        name, "multilabel", sparse_container=csc_container
    )


def test_min_impurity_decrease(global_random_seed):
    # 测试最小不纯度减少值是否确保仅当不纯度减少至少达到该值时才进行分裂
    # 创建一个具有全局随机种子的分类数据集 X, y
    X, y = datasets.make_classification(n_samples=100, random_state=global_random_seed)

    # 测试使用深度优先树构建器和最佳优先树构建器
    # 通过设置最大叶子节点数来测试
    # 遍历所有可能的最大叶子节点数和决策树名称的组合
    for max_leaf_nodes, name in product((None, 1000), ALL_TREES.keys()):
        # 获取当前树的估算器类
        TreeEstimator = ALL_TREES[name]

        # 使用默认的 min_impurity_decrease 值 1e-7 进行估算器初始化
        est1 = TreeEstimator(max_leaf_nodes=max_leaf_nodes, random_state=0)
        # 使用显式值 0.05 进行估算器初始化
        est2 = TreeEstimator(
            max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=0.05, random_state=0
        )
        # 使用较低的值 0.0001 进行估算器初始化
        est3 = TreeEstimator(
            max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=0.0001, random_state=0
        )
        # 使用较低的值 0.1 进行估算器初始化
        est4 = TreeEstimator(
            max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=0.1, random_state=0
        )

        # 遍历每个估算器及其预期的 min_impurity_decrease 值
        for est, expected_decrease in (
            (est1, 1e-7),
            (est2, 0.05),
            (est3, 0.0001),
            (est4, 0.1),
        ):
            # 断言当前的 min_impurity_decrease 不大于预期值，否则输出错误信息
            assert (
                est.min_impurity_decrease <= expected_decrease
            ), "Failed, min_impurity_decrease = {0} > {1}".format(
                est.min_impurity_decrease, expected_decrease
            )
            # 使用数据 X 和标签 y 对估算器进行拟合
            est.fit(X, y)

            # 遍历当前估算器的每个节点
            for node in range(est.tree_.node_count):
                # 如果当前节点不是叶子节点，则检查是否基于 min_impurity_decrease 进行了切分
                if est.tree_.children_left[node] != TREE_LEAF:
                    # 获取当前节点的不纯度和加权节点样本数
                    imp_parent = est.tree_.impurity[node]
                    wtd_n_node = est.tree_.weighted_n_node_samples[node]

                    # 获取左子节点的信息：索引、加权节点样本数、不纯度、加权不纯度
                    left = est.tree_.children_left[node]
                    wtd_n_left = est.tree_.weighted_n_node_samples[left]
                    imp_left = est.tree_.impurity[left]
                    wtd_imp_left = wtd_n_left * imp_left

                    # 获取右子节点的信息：索引、加权节点样本数、不纯度、加权不纯度
                    right = est.tree_.children_right[node]
                    wtd_n_right = est.tree_.weighted_n_node_samples[right]
                    imp_right = est.tree_.impurity[right]
                    wtd_imp_right = wtd_n_right * imp_right

                    # 计算左右节点加权平均不纯度
                    wtd_avg_left_right_imp = wtd_imp_right + wtd_imp_left
                    wtd_avg_left_right_imp /= wtd_n_node

                    # 计算节点权重的分数
                    fractional_node_weight = (
                        est.tree_.weighted_n_node_samples[node] / X.shape[0]
                    )

                    # 计算实际的不纯度减少量
                    actual_decrease = fractional_node_weight * (
                        imp_parent - wtd_avg_left_right_imp
                    )

                    # 断言实际的不纯度减少量不小于预期值，否则输出错误信息
                    assert (
                        actual_decrease >= expected_decrease
                    ), "Failed with {0} expected min_impurity_decrease={1}".format(
                        actual_decrease, expected_decrease
                    )
# 测试序列化与反序列化过程中是否能保持树模型的属性和性能
def test_pickle():
    """Test pickling preserves Tree properties and performance."""
    # 遍历所有的决策树估计器
    for name, TreeEstimator in ALL_TREES.items():
        # 根据估计器的名称选择不同的数据集
        if "Classifier" in name:
            X, y = iris.data, iris.target
        else:
            X, y = diabetes.data, diabetes.target

        # 创建估计器实例并拟合数据
        est = TreeEstimator(random_state=0)
        est.fit(X, y)
        score = est.score(X, y)

        # 测试所有类属性是否被保留
        attributes = [
            "max_depth",
            "node_count",
            "capacity",
            "n_classes",
            "children_left",
            "children_right",
            "n_leaves",
            "feature",
            "threshold",
            "impurity",
            "n_node_samples",
            "weighted_n_node_samples",
            "value",
        ]
        # 提取拟合后的树模型属性
        fitted_attribute = {
            attribute: getattr(est.tree_, attribute) for attribute in attributes
        }

        # 将对象序列化为字节流
        serialized_object = pickle.dumps(est)
        # 从序列化后的字节流中反序列化得到新的对象
        est2 = pickle.loads(serialized_object)
        # 断言反序列化后的对象类型与原对象类型相同
        assert type(est2) == est.__class__

        # 再次计算分数，确保序列化与反序列化后得到的对象具有相同的性能
        score2 = est2.score(X, y)
        assert (
            score == score2
        ), f"Failed to generate same score after pickling with {name}"
        # 检查每个属性是否在序列化与反序列化后保持一致
        for attribute in fitted_attribute:
            assert_array_equal(
                getattr(est2.tree_, attribute),
                fitted_attribute[attribute],
                err_msg=(
                    f"Failed to generate same attribute {attribute} after pickling with"
                    f" {name}"
                ),
            )


def test_multioutput():
    # 在多输出问题上检查估计器
    X = [
        [-2, -1],
        [-1, -1],
        [-1, -2],
        [1, 1],
        [1, 2],
        [2, 1],
        [-2, 1],
        [-1, 1],
        [-1, 2],
        [2, -1],
        [1, -1],
        [1, -2],
    ]

    y = [
        [-1, 0],
        [-1, 0],
        [-1, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [-1, 2],
        [-1, 2],
        [-1, 2],
        [1, 3],
        [1, 3],
        [1, 3],
    ]

    T = [[-1, -1], [1, 1], [-1, 1], [1, -1]]
    y_true = [[-1, 0], [1, 1], [-1, 2], [1, 3]]

    # 玩具分类问题
    for name, TreeClassifier in CLF_TREES.items():
        # 创建分类器实例并拟合数据
        clf = TreeClassifier(random_state=0)
        y_hat = clf.fit(X, y).predict(T)
        # 断言预测结果与真实结果一致
        assert_array_equal(y_hat, y_true)
        # 断言预测结果的形状正确
        assert y_hat.shape == (4, 2)

        # 检查预测概率的形状
        proba = clf.predict_proba(T)
        assert len(proba) == 2
        assert proba[0].shape == (4, 2)
        assert proba[1].shape == (4, 4)

        # 检查预测对数概率的形状
        log_proba = clf.predict_log_proba(T)
        assert len(log_proba) == 2
        assert log_proba[0].shape == (4, 2)
        assert log_proba[1].shape == (4, 4)

    # 玩具回归问题
    for name, TreeRegressor in REG_TREES.items():
        # 创建回归器实例并拟合数据
        reg = TreeRegressor(random_state=0)
        y_hat = reg.fit(X, y).predict(T)
        # 断言预测结果与真实结果接近
        assert_almost_equal(y_hat, y_true)
        # 断言预测结果的形状正确
        assert y_hat.shape == (4, 2)
# 测试分类器的 n_classes_ 和 classes_ 的形状是否正确
def test_classes_shape():
    # 遍历 CLF_TREES 字典中的每个分类器名称和对应的类
    for name, TreeClassifier in CLF_TREES.items():
        # 创建具有随机状态的分类器对象
        clf = TreeClassifier(random_state=0)
        # 使用数据集 X 和标签 y 进行训练
        clf.fit(X, y)

        # 断言分类器的输出类别数量是否为2
        assert clf.n_classes_ == 2
        # 断言分类器的类别是否为 [-1, 1]
        assert_array_equal(clf.classes_, [-1, 1])

        # 创建多输出的分类标签 _y
        _y = np.vstack((y, np.array(y) * 2)).T
        # 创建具有随机状态的分类器对象
        clf = TreeClassifier(random_state=0)
        # 使用数据集 X 和多输出标签 _y 进行训练
        clf.fit(X, _y)
        # 断言分类器的每个输出的类别数量是否为2
        assert len(clf.n_classes_) == 2
        # 断言分类器的每个输出的类别是否为 [[-1, 1], [-2, 2]]
        assert_array_equal(clf.classes_, [[-1, 1], [-2, 2]])


# 检查类别重新平衡
def test_unbalanced_iris():
    # 选取部分鸢尾花数据作为不平衡数据集
    unbalanced_X = iris.data[:125]
    unbalanced_y = iris.target[:125]
    # 计算平衡样本权重
    sample_weight = compute_sample_weight("balanced", unbalanced_y)

    # 遍历 CLF_TREES 字典中的每个分类器名称和对应的类
    for name, TreeClassifier in CLF_TREES.items():
        # 创建具有随机状态的分类器对象
        clf = TreeClassifier(random_state=0)
        # 使用不平衡数据集和样本权重进行训练
        clf.fit(unbalanced_X, unbalanced_y, sample_weight=sample_weight)
        # 断言分类器对不平衡数据集的预测结果与真实标签几乎相等
        assert_almost_equal(clf.predict(unbalanced_X), unbalanced_y)


# 检查不同内存布局下的工作情况
def test_memory_layout():
    # 遍历 ALL_TREES 字典中的每个树估计器和数据类型
    for (name, TreeEstimator), dtype in product(ALL_TREES.items(), [np.float64, np.float32]):
        # 创建具有随机状态的树估计器对象
        est = TreeEstimator(random_state=0)

        # 没有指定布局时的数据集 X 和标签 y
        X = np.asarray(iris.data, dtype=dtype)
        y = iris.target
        # 断言使用C顺序布局时，树估计器的训练和预测结果是否相等
        assert_array_equal(est.fit(X, y).predict(X), y)

        # C顺序布局时的数据集 X 和标签 y
        X = np.asarray(iris.data, order="C", dtype=dtype)
        y = iris.target
        # 断言使用C顺序布局时，树估计器的训练和预测结果是否相等
        assert_array_equal(est.fit(X, y).predict(X), y)

        # F顺序布局时的数据集 X 和标签 y
        X = np.asarray(iris.data, order="F", dtype=dtype)
        y = iris.target
        # 断言使用F顺序布局时，树估计器的训练和预测结果是否相等
        assert_array_equal(est.fit(X, y).predict(X), y)

        # 连续内存布局时的数据集 X 和标签 y
        X = np.ascontiguousarray(iris.data, dtype=dtype)
        y = iris.target
        # 断言使用连续内存布局时，树估计器的训练和预测结果是否相等
        assert_array_equal(est.fit(X, y).predict(X), y)

        # CSR格式时的数据集 X 和标签 y
        for csr_container in CSR_CONTAINERS:
            X = csr_container(iris.data, dtype=dtype)
            y = iris.target
            # 断言使用CSR格式时，树估计器的训练和预测结果是否相等
            assert_array_equal(est.fit(X, y).predict(X), y)

        # CSC格式时的数据集 X 和标签 y
        for csc_container in CSC_CONTAINERS:
            X = csc_container(iris.data, dtype=dtype)
            y = iris.target
            # 断言使用CSC格式时，树估计器的训练和预测结果是否相等
            assert_array_equal(est.fit(X, y).predict(X), y)

        # 步进的数据集 X 和标签 y
        X = np.asarray(iris.data[::3], dtype=dtype)
        y = iris.target[::3]
        # 断言使用步进数据集时，树估计器的训练和预测结果是否相等
        assert_array_equal(est.fit(X, y).predict(X), y)


# 检查样本权重
def test_sample_weight():
    # 检查样本加权
    # 测试零权重样本不被考虑
    X = np.arange(100)[:, np.newaxis]
    y = np.ones(100)
    y[:50] = 0.0

    sample_weight = np.ones(100)
    sample_weight[y == 0] = 0.0

    # 创建具有随机状态的决策树分类器对象
    clf = DecisionTreeClassifier(random_state=0)
    # 使用样本权重进行训练
    clf.fit(X, y, sample_weight=sample_weight)
    # 断言分类器对 X 的预测结果与全为1的数组相等
    assert_array_equal(clf.predict(X), np.ones(100))
    # Test that low weighted samples are not taken into account at low depth
    # 创建一个包含200行的列向量，其中每行是一个整数，范围从0到199
    X = np.arange(200)[:, np.newaxis]
    # 创建一个长度为200的零数组
    y = np.zeros(200)
    # 将索引为50到99的元素设置为1，表示第一个类别的样本
    y[50:100] = 1
    # 将索引为100到199的元素设置为2，表示第二个类别的样本
    y[100:200] = 2
    # 将X的第一列中索引为100到199的行设置为200
    X[100:200, 0] = 200

    # 创建一个长度为200的全1数组作为样本权重
    sample_weight = np.ones(200)

    # 将类别为2的样本权重设置为0.51，使其比其他样本更重要
    sample_weight[y == 2] = 0.51
    # 创建一个深度为1的决策树分类器对象
    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
    # 使用样本权重拟合分类器
    clf.fit(X, y, sample_weight=sample_weight)
    # 断言树的第一个节点的分裂阈值为149.5
    assert clf.tree_.threshold[0] == 149.5

    # 将类别为2的样本权重重新设置为0.5，使其与其他样本相同
    sample_weight[y == 2] = 0.5
    # 创建一个新的深度为1的决策树分类器对象
    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
    # 使用新的样本权重拟合分类器
    clf.fit(X, y, sample_weight=sample_weight)
    # 断言树的第一个节点的分裂阈值为49.5，说明阈值已经移动
    assert clf.tree_.threshold[0] == 49.5  # Threshold should have moved

    # Test that sample weighting is the same as having duplicates
    # 加载鸢尾花数据集中的特征数据和目标值
    X = iris.data
    y = iris.target

    # 从X的行数中随机选择100个整数作为重复样本的索引
    duplicates = rng.randint(0, X.shape[0], 100)

    # 创建一个随机种子为1的决策树分类器对象
    clf = DecisionTreeClassifier(random_state=1)
    # 使用重复样本拟合分类器
    clf.fit(X[duplicates], y[duplicates])

    # 使用np.bincount()函数创建一个与样本数相同长度的样本权重数组
    sample_weight = np.bincount(duplicates, minlength=X.shape[0])
    # 创建一个随机种子为1的新决策树分类器对象
    clf2 = DecisionTreeClassifier(random_state=1)
    # 使用样本权重数组拟合分类器
    clf2.fit(X, y, sample_weight=sample_weight)

    # 检查内部节点的阈值是否几乎相等，以验证样本权重和重复样本方法的等效性
    internal = clf.tree_.children_left != tree._tree.TREE_LEAF
    assert_array_almost_equal(
        clf.tree_.threshold[internal], clf2.tree_.threshold[internal]
    )
# 测试无效的样本权重
def test_sample_weight_invalid():
    # 创建包含100个元素的列向量 X
    X = np.arange(100)[:, np.newaxis]
    # 创建长度为100的全1数组 y，并将前50个元素置为0.0
    y = np.ones(100)
    y[:50] = 0.0

    # 创建一个决策树分类器对象 clf
    clf = DecisionTreeClassifier(random_state=0)

    # 创建一个形状为(100, 1)的随机样本权重数组 sample_weight
    sample_weight = np.random.rand(100, 1)
    # 检查是否会引发 ValueError 异常
    with pytest.raises(ValueError):
        clf.fit(X, y, sample_weight=sample_weight)

    # 创建一个标量值为0的数组 sample_weight
    sample_weight = np.array(0)
    # 准备匹配的错误信息字符串 expected_err
    expected_err = r"Singleton.* cannot be considered a valid collection"
    # 检查是否会引发 TypeError 异常，并且错误信息与 expected_err 匹配
    with pytest.raises(TypeError, match=expected_err):
        clf.fit(X, y, sample_weight=sample_weight)


@pytest.mark.parametrize("name", CLF_TREES)
def test_class_weights(name):
    # 测试 class_weights 是否类似于 sample_weights 的行为
    TreeClassifier = CLF_TREES[name]

    # 对于平衡的鸢尾花数据集，使用 'balanced' 权重不应有影响
    clf1 = TreeClassifier(random_state=0)
    clf1.fit(iris.data, iris.target)
    clf2 = TreeClassifier(class_weight="balanced", random_state=0)
    clf2.fit(iris.data, iris.target)
    # 检查两个分类器的特征重要性是否几乎相等
    assert_almost_equal(clf1.feature_importances_, clf2.feature_importances_)

    # 创建一个包含三个鸢尾花副本的多输出问题数据集 iris_multi
    iris_multi = np.vstack((iris.target, iris.target, iris.target)).T
    # 创建自定义权重，预计能平衡输出
    clf3 = TreeClassifier(
        class_weight=[
            {0: 2.0, 1: 2.0, 2: 1.0},
            {0: 2.0, 1: 1.0, 2: 2.0},
            {0: 1.0, 1: 2.0, 2: 2.0},
        ],
        random_state=0,
    )
    clf3.fit(iris.data, iris_multi)
    # 检查两个分类器的特征重要性是否几乎相等
    assert_almost_equal(clf2.feature_importances_, clf3.feature_importances_)
    # 使用 'auto' 多输出权重检查，预计也不会有影响
    clf4 = TreeClassifier(class_weight="balanced", random_state=0)
    clf4.fit(iris.data, iris_multi)
    # 检查两个分类器的特征重要性是否几乎相等
    assert_almost_equal(clf3.feature_importances_, clf4.feature_importances_)

    # 增加类别 1 的重要性，检查是否与自定义权重匹配
    sample_weight = np.ones(iris.target.shape)
    sample_weight[iris.target == 1] *= 100
    class_weight = {0: 1.0, 1: 100.0, 2: 1.0}
    clf1 = TreeClassifier(random_state=0)
    clf1.fit(iris.data, iris.target, sample_weight)
    clf2 = TreeClassifier(class_weight=class_weight, random_state=0)
    clf2.fit(iris.data, iris.target)
    # 检查两个分类器的特征重要性是否几乎相等
    assert_almost_equal(clf1.feature_importances_, clf2.feature_importances_)

    # 检查 sample_weight 和 class_weight 是否可以相乘
    clf1 = TreeClassifier(random_state=0)
    clf1.fit(iris.data, iris.target, sample_weight**2)
    clf2 = TreeClassifier(class_weight=class_weight, random_state=0)
    clf2.fit(iris.data, iris.target, sample_weight)
    # 检查两个分类器的特征重要性是否几乎相等
    assert_almost_equal(clf1.feature_importances_, clf2.feature_importances_)
    # 创建一个决策树分类器对象 clf，指定了类别权重为 [{-1: 0.5, 1: 1.0}]，并设置了随机数种子为 0
    clf = TreeClassifier(class_weight=[{-1: 0.5, 1: 1.0}], random_state=0)
    # 定义错误消息字符串，用于匹配 pytest.raises 抛出的 ValueError 异常
    err_msg = "number of elements in class_weight should match number of outputs."
    # 使用 pytest 库的 pytest.raises 上下文管理器捕获 ValueError 异常，并验证错误消息是否符合预期
    with pytest.raises(ValueError, match=err_msg):
        # 调用分类器对象的 fit 方法，传入 X 和 _y 进行模型拟合
        clf.fit(X, _y)
def test_max_leaf_nodes():
    # 测试使用 max_depth + 1 个叶子节点的贪婪树。
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    # 设置最大叶子节点数
    k = 4
    # 遍历所有树估计器
    for name, TreeEstimator in ALL_TREES.items():
        # 使用指定参数拟合模型
        est = TreeEstimator(max_depth=None, max_leaf_nodes=k + 1).fit(X, y)
        # 断言叶子节点数目符合预期
        assert est.get_n_leaves() == k + 1


def test_max_leaf_nodes_max_depth():
    # 测试 max_leaf_nodes 对 max_depth 的优先级。
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    # 设置最大叶子节点数
    k = 4
    # 遍历所有树估计器
    for name, TreeEstimator in ALL_TREES.items():
        # 使用指定参数拟合模型
        est = TreeEstimator(max_depth=1, max_leaf_nodes=k).fit(X, y)
        # 断言深度符合预期
        assert est.get_depth() == 1


def test_arrays_persist():
    # 确保在树消失后，属性数组的内存仍然存在
    # 针对问题 #2726 的非回归测试
    for attr in [
        "n_classes",
        "value",
        "children_left",
        "children_right",
        "threshold",
        "impurity",
        "feature",
        "n_node_samples",
    ]:
        # 获取决策树分类器的属性值
        value = getattr(DecisionTreeClassifier().fit([[0], [1]], [0, 1]).tree_, attr)
        # 如果指向释放的内存，内容可能是任意的
        assert -3 <= value.flat[0] < 3, "Array points to arbitrary memory"


def test_only_constant_features():
    random_state = check_random_state(0)
    # 创建零矩阵
    X = np.zeros((10, 20))
    y = random_state.randint(0, 2, (10,))
    # 遍历所有树估计器
    for name, TreeEstimator in ALL_TREES.items():
        # 使用指定参数创建估计器
        est = TreeEstimator(random_state=0)
        est.fit(X, y)
        # 断言树的最大深度为0
        assert est.tree_.max_depth == 0


def test_behaviour_constant_feature_after_splits():
    # 创建特征矩阵和标签
    X = np.transpose(
        np.vstack(([[0, 0, 0, 0, 0, 1, 2, 4, 5, 6, 7]], np.zeros((4, 11))))
    )
    y = [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3]
    # 遍历所有树估计器
    for name, TreeEstimator in ALL_TREES.items():
        # 对 ExtraTree 以外的树进行测试
        if "ExtraTree" not in name:
            # 使用指定参数创建估计器
            est = TreeEstimator(random_state=0, max_features=1)
            est.fit(X, y)
            # 断言树的最大深度为2
            assert est.tree_.max_depth == 2
            # 断言节点数量为5
            assert est.tree_.node_count == 5


def test_with_only_one_non_constant_features():
    # 创建包含一个非常量特征的特征矩阵
    X = np.hstack([np.array([[1.0], [1.0], [0.0], [0.0]]), np.zeros((4, 1000))])

    y = np.array([0.0, 1.0, 0.0, 1.0])
    # 遍历分类树估计器
    for name, TreeEstimator in CLF_TREES.items():
        # 使用指定参数创建估计器
        est = TreeEstimator(random_state=0, max_features=1)
        est.fit(X, y)
        # 断言树的最大深度为1
        assert est.tree_.max_depth == 1
        # 断言预测概率数组符合预期
        assert_array_equal(est.predict_proba(X), np.full((4, 2), 0.5))

    # 遍历回归树估计器
    for name, TreeEstimator in REG_TREES.items():
        # 使用指定参数创建估计器
        est = TreeEstimator(random_state=0, max_features=1)
        est.fit(X, y)
        # 断言树的最大深度为1
        assert est.tree_.max_depth == 1
        # 断言预测数组符合预期
        assert_array_equal(est.predict(X), np.full((4,), 0.5))


def test_big_input():
    # 测试对于过大输入是否会适时发出警告。
    X = np.repeat(10**40.0, 4).astype(np.float64).reshape(-1, 1)
    clf = DecisionTreeClassifier()
    # 使用 pytest 检查是否引发特定的 ValueError 错误，匹配字符串 "float32"
    with pytest.raises(ValueError, match="float32"):
        clf.fit(X, [0, 1, 0, 1])


def test_realloc():
    # 这个测试函数尚未实现，仅作为函数占位使用
    pass
    # 从sklearn.tree._utils模块中导入_realloc_test函数
    from sklearn.tree._utils import _realloc_test
    
    # 使用pytest模块的raises装饰器捕获MemoryError异常
    with pytest.raises(MemoryError):
        # 调用_realloc_test函数，期望其引发MemoryError异常
        _realloc_test()
# 定义一个测试函数，用于测试大内存分配情况
def test_huge_allocations():
    # 计算指针大小所需的位数
    n_bits = 8 * struct.calcsize("P")

    # 创建随机数组 X 和 y
    X = np.random.randn(10, 2)
    y = np.random.randint(0, 2, 10)

    # 检查：不能请求比地址空间大小更多的内存。当前会引发 OverflowError。
    huge = 2 ** (n_bits + 1)
    clf = DecisionTreeClassifier(splitter="best", max_leaf_nodes=huge)
    # 使用 pytest 断言来检查是否抛出异常
    with pytest.raises(Exception):
        clf.fit(X, y)

    # 非回归测试：以前 Cython 会因缺少 "except *" 而丢弃 MemoryError。
    huge = 2 ** (n_bits - 1) - 1
    clf = DecisionTreeClassifier(splitter="best", max_leaf_nodes=huge)
    # 使用 pytest 断言来检查是否抛出 MemoryError
    with pytest.raises(MemoryError):
        clf.fit(X, y)


# 检查稀疏输入的函数
def check_sparse_input(tree, dataset, max_depth=None):
    # 选择正确的树估算器和数据集
    TreeEstimator = ALL_TREES[tree]
    X = DATASETS[dataset]["X"]
    y = DATASETS[dataset]["y"]

    # 加速测试时间
    if dataset in ["digits", "diabetes"]:
        # 减少样本数以加速测试
        n_samples = X.shape[0] // 5
        X = X[:n_samples]
        y = y[:n_samples]

    # 遍历不同的稀疏矩阵容器类型
    for sparse_container in COO_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS:
        # 使用当前容器类型创建稀疏矩阵 X_sparse
        X_sparse = sparse_container(X)

        # 检查默认的深度优先搜索树
        d = TreeEstimator(random_state=0, max_depth=max_depth).fit(X, y)
        s = TreeEstimator(random_state=0, max_depth=max_depth).fit(X_sparse, y)

        # 使用自定义断言函数检查两种树的等价性
        assert_tree_equal(
            d.tree_,
            s.tree_,
            "{0} with dense and sparse format gave different trees".format(tree),
        )

        # 预测 d 的结果并验证稀疏矩阵预测结果与之相等
        y_pred = d.predict(X)
        if tree in CLF_TREES:
            y_proba = d.predict_proba(X)
            y_log_proba = d.predict_log_proba(X)

        # 再次遍历不同的稀疏矩阵容器类型，测试稀疏矩阵 X_sparse_test 的预测结果
        for sparse_container_test in COO_CONTAINERS + CSR_CONTAINERS + CSC_CONTAINERS:
            X_sparse_test = sparse_container_test(X_sparse, dtype=np.float32)

            assert_array_almost_equal(s.predict(X_sparse_test), y_pred)

            if tree in CLF_TREES:
                assert_array_almost_equal(s.predict_proba(X_sparse_test), y_proba)
                assert_array_almost_equal(
                    s.predict_log_proba(X_sparse_test), y_log_proba
                )


# 使用参数化测试来执行不同的稀疏输入测试
@pytest.mark.parametrize("tree_type", SPARSE_TREES)
@pytest.mark.parametrize(
    "dataset",
    (
        "clf_small",
        "toy",
        "digits",
        "multilabel",
        "sparse-pos",
        "sparse-neg",
        "sparse-mix",
        "zeros",
    ),
)
def test_sparse_input(tree_type, dataset):
    max_depth = 3 if dataset == "digits" else None
    check_sparse_input(tree_type, dataset, max_depth)


# 使用参数化测试来执行稀疏输入与回归树的测试
@pytest.mark.parametrize("tree_type", sorted(set(SPARSE_TREES).intersection(REG_TREES)))
@pytest.mark.parametrize("dataset", ["diabetes", "reg_small"])
def test_sparse_input_reg_trees(tree_type, dataset):
    # 由于 MSE 的数值不稳定性和测试严格度，限制最大深度
    check_sparse_input(tree_type, dataset, 2)
# 使用pytest的mark.parametrize装饰器，为test_sparse_parameters函数参数化数据集
@pytest.mark.parametrize("dataset", ["sparse-pos", "sparse-neg", "sparse-mix", "zeros"])
# 使用pytest的mark.parametrize装饰器，为test_sparse_parameters函数参数化稀疏容器类型
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
# 定义测试函数test_sparse_parameters，测试稀疏参数的影响
def test_sparse_parameters(tree_type, dataset, csc_container):
    # 根据tree_type选择对应的树估计器类
    TreeEstimator = ALL_TREES[tree_type]
    # 获取指定数据集的特征矩阵X
    X = DATASETS[dataset]["X"]
    # 使用给定的稀疏容器类型csc_container处理特征矩阵X，转换成稀疏格式
    X_sparse = csc_container(X)
    # 获取指定数据集的标签y
    y = DATASETS[dataset]["y"]

    # 测试max_features参数对比
    # 创建两个树估计器对象，分别使用稠密和稀疏格式数据拟合
    d = TreeEstimator(random_state=0, max_features=1, max_depth=2).fit(X, y)
    s = TreeEstimator(random_state=0, max_features=1, max_depth=2).fit(X_sparse, y)
    # 断言两者的决策树结构相同
    assert_tree_equal(
        d.tree_,
        s.tree_,
        "{0} with dense and sparse format gave different trees".format(tree_type),
    )
    # 断言使用稠密和稀疏格式预测结果几乎相等
    assert_array_almost_equal(s.predict(X), d.predict(X))

    # 测试min_samples_split参数对比
    d = TreeEstimator(random_state=0, max_features=1, min_samples_split=10).fit(X, y)
    s = TreeEstimator(random_state=0, max_features=1, min_samples_split=10).fit(
        X_sparse, y
    )
    assert_tree_equal(
        d.tree_,
        s.tree_,
        "{0} with dense and sparse format gave different trees".format(tree_type),
    )
    assert_array_almost_equal(s.predict(X), d.predict(X))

    # 测试min_samples_leaf参数对比
    d = TreeEstimator(random_state=0, min_samples_leaf=X_sparse.shape[0] // 2).fit(X, y)
    s = TreeEstimator(random_state=0, min_samples_leaf=X_sparse.shape[0] // 2).fit(
        X_sparse, y
    )
    assert_tree_equal(
        d.tree_,
        s.tree_,
        "{0} with dense and sparse format gave different trees".format(tree_type),
    )
    assert_array_almost_equal(s.predict(X), d.predict(X))

    # 测试最佳优先搜索(max_leaf_nodes)参数对比
    d = TreeEstimator(random_state=0, max_leaf_nodes=3).fit(X, y)
    s = TreeEstimator(random_state=0, max_leaf_nodes=3).fit(X_sparse, y)
    assert_tree_equal(
        d.tree_,
        s.tree_,
        "{0} with dense and sparse format gave different trees".format(tree_type),
    )
    assert_array_almost_equal(s.predict(X), d.predict(X))


# 使用pytest的mark.parametrize装饰器，为test_sparse_criteria函数参数化树类型和评估标准
@pytest.mark.parametrize(
    "tree_type, criterion",
    list(product([tree for tree in SPARSE_TREES if tree in REG_TREES], REG_CRITERIONS))
    + list(
        product([tree for tree in SPARSE_TREES if tree in CLF_TREES], CLF_CRITERIONS)
    ),
)
# 使用pytest的mark.parametrize装饰器，为test_sparse_criteria函数参数化数据集
@pytest.mark.parametrize("dataset", ["sparse-pos", "sparse-neg", "sparse-mix", "zeros"])
# 使用pytest的mark.parametrize装饰器，为test_sparse_criteria函数参数化稀疏容器类型
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
# 定义测试函数test_sparse_criteria，测试稀疏数据集和评估标准的影响
def test_sparse_criteria(tree_type, dataset, csc_container, criterion):
    # 根据tree_type选择对应的树估计器类
    TreeEstimator = ALL_TREES[tree_type]
    # 获取指定数据集的特征矩阵X
    X = DATASETS[dataset]["X"]
    # 使用给定的稀疏容器类型csc_container处理特征矩阵X，转换成稀疏格式
    X_sparse = csc_container(X)
    # 获取指定数据集的标签y
    y = DATASETS[dataset]["y"]

    # 创建两个树估计器对象，分别使用稠密和稀疏格式数据拟合
    d = TreeEstimator(random_state=0, max_depth=3, criterion=criterion).fit(X, y)
    s = TreeEstimator(random_state=0, max_depth=3, criterion=criterion).fit(X_sparse, y)

    # 断言两者的决策树结构相同
    assert_tree_equal(
        d.tree_,
        s.tree_,
        "{0} with dense and sparse format gave different trees".format(tree_type),
    )
    # 断言使用稠密和稀疏格式预测结果几乎相等
    assert_array_almost_equal(s.predict(X), d.predict(X))
# 使用 pytest 的 parametrize 装饰器，为测试函数提供多组参数化的输入，每次运行测试函数时会遍历这些参数
@pytest.mark.parametrize("tree_type", SPARSE_TREES)
@pytest.mark.parametrize(
    "csc_container,csr_container", zip(CSC_CONTAINERS, CSR_CONTAINERS)
)
# 定义测试函数，测试稀疏矩阵生成和树模型的行为
def test_explicit_sparse_zeros(tree_type, csc_container, csr_container):
    # 根据树类型获取对应的树估计器类
    TreeEstimator = ALL_TREES[tree_type]
    max_depth = 3
    n_features = 10

    # 设置样本数等于特征数，以便同时构建 csr 和 csc 矩阵
    n_samples = n_features
    samples = np.arange(n_samples)

    # 生成稀疏矩阵 X 和对应的标签 y
    random_state = check_random_state(0)
    indices = []
    data = []
    offset = 0
    indptr = [offset]
    for i in range(n_features):
        # 在每列中生成随机数量的非零元素
        n_nonzero_i = random_state.binomial(n_samples, 0.5)
        indices_i = random_state.permutation(samples)[:n_nonzero_i]
        indices.append(indices_i)
        # 生成随机数据
        data_i = random_state.binomial(3, 0.5, size=(n_nonzero_i,)) - 1
        data.append(data_i)
        offset += n_nonzero_i
        indptr.append(offset)

    # 将生成的数据转换为 numpy 数组，用于构建稀疏矩阵
    indices = np.concatenate(indices).astype(np.int32)
    indptr = np.array(indptr, dtype=np.int32)
    data = np.array(np.concatenate(data), dtype=np.float32)
    X_sparse = csc_container((data, indices, indptr), shape=(n_samples, n_features))
    X = X_sparse.toarray()  # 转换为稠密矩阵
    X_sparse_test = csr_container((data, indices, indptr), shape=(n_samples, n_features))
    X_test = X_sparse_test.toarray()  # 转换为稠密矩阵
    y = random_state.randint(0, 3, size=(n_samples,))

    # 确保 X_sparse_test 拥有自己的数据，索引和指针数组
    X_sparse_test = X_sparse_test.copy()

    # 确保稀疏矩阵 X_sparse 和 X_sparse_test 中有显式的零元素
    assert (X_sparse.data == 0.0).sum() > 0
    assert (X_sparse_test.data == 0.0).sum() > 0

    # 执行比较测试
    d = TreeEstimator(random_state=0, max_depth=max_depth).fit(X, y)
    s = TreeEstimator(random_state=0, max_depth=max_depth).fit(X_sparse, y)

    # 断言两种构建方式（稠密和稀疏）的树模型结构相同
    assert_tree_equal(
        d.tree_,
        s.tree_,
        "{0} with dense and sparse format gave different trees".format(tree_type),
    )

    Xs = (X_test, X_sparse_test)
    for X1, X2 in product(Xs, Xs):
        # 断言预测的类别和决策路径在稀疏和稠密格式下的一致性
        assert_array_almost_equal(s.tree_.apply(X1), d.tree_.apply(X2))
        assert_array_almost_equal(s.apply(X1), d.apply(X2))
        assert_array_almost_equal(s.apply(X1), s.tree_.apply(X1))

        # 断言决策路径在稀疏和稠密格式下的一致性
        assert_array_almost_equal(
            s.tree_.decision_path(X1).toarray(), d.tree_.decision_path(X2).toarray()
        )
        assert_array_almost_equal(
            s.decision_path(X1).toarray(), d.decision_path(X2).toarray()
        )
        assert_array_almost_equal(
            s.decision_path(X1).toarray(), s.tree_.decision_path(X1).toarray()
        )

        # 断言预测结果在稀疏和稠密格式下的一致性
        assert_array_almost_equal(s.predict(X1), d.predict(X2))

        # 如果是分类树模型，断言预测概率在稀疏和稠密格式下的一致性
        if tree_type in CLF_TREES:
            assert_array_almost_equal(s.predict_proba(X1), d.predict_proba(X2))


# 使用 ignore_warnings 装饰器确保在输入为 1 维时能够引发错误
@ignore_warnings
def check_raise_error_on_1d_input(name):
    # 根据模型名称获取对应的树估计器类
    TreeEstimator = ALL_TREES[name]

    # 从鸢尾花数据集中获取一个特征的一维向量和相应的二维版本
    X = iris.data[:, 0].ravel()
    X_2d = iris.data[:, 0].reshape((-1, 1))
    # 从iris数据集中获取目标值（类别标签）
    y = iris.target
    
    # 使用pytest模块检查是否会抛出ValueError异常
    with pytest.raises(ValueError):
        # 创建一个TreeEstimator对象，设置随机种子为0，并尝试使用X和y进行拟合
        TreeEstimator(random_state=0).fit(X, y)
    
    # 创建TreeEstimator对象，设置随机种子为0
    est = TreeEstimator(random_state=0)
    # 使用X_2d和y对est对象进行拟合
    est.fit(X_2d, y)
    
    # 使用pytest模块检查是否会抛出ValueError异常
    with pytest.raises(ValueError):
        # 使用est对象预测单个样本X，并检查是否会抛出异常
        est.predict([X])
# 使用 pytest 的 mark.parametrize 装饰器，对每个测试用例参数化处理
@pytest.mark.parametrize("name", ALL_TREES)
def test_1d_input(name):
    # 忽略警告，执行检查一维输入是否会引发错误的函数
    with ignore_warnings():
        check_raise_error_on_1d_input(name)


# 对于每个树的名称和稀疏容器类型参数化测试
@pytest.mark.parametrize("name", ALL_TREES)
@pytest.mark.parametrize("sparse_container", [None] + CSC_CONTAINERS)
def test_min_weight_leaf_split_level(name, sparse_container):
    # 根据树的名称获取相应的树估计器类
    TreeEstimator = ALL_TREES[name]

    # 创建一个简单的输入数据集 X 和标签 y
    X = np.array([[0], [0], [0], [0], [1]])
    y = [0, 0, 0, 0, 1]
    sample_weight = [0.2, 0.2, 0.2, 0.2, 0.2]

    # 如果稀疏容器不为 None，则将 X 转换为稀疏表示
    if sparse_container is not None:
        X = sparse_container(X)

    # 创建估计器对象，使用 fit 方法拟合数据
    est = TreeEstimator(random_state=0)
    est.fit(X, y, sample_weight=sample_weight)

    # 断言树的最大深度为 1
    assert est.tree_.max_depth == 1

    # 创建另一个估计器对象，设置最小叶子权重分数为 0.4
    est = TreeEstimator(random_state=0, min_weight_fraction_leaf=0.4)
    est.fit(X, y, sample_weight=sample_weight)

    # 断言树的最大深度为 0
    assert est.tree_.max_depth == 0


# 对每个树的名称参数化测试
@pytest.mark.parametrize("name", ALL_TREES)
def test_public_apply_all_trees(name):
    # 将 X_small 转换为指定类型（X_small32），以保证数据类型一致性
    X_small32 = X_small.astype(tree._tree.DTYPE, copy=False)

    # 创建特定树估计器对象，拟合 X_small 和 y_small 数据
    est = ALL_TREES[name]()
    est.fit(X_small, y_small)

    # 断言应用该估计器的 apply 方法结果与其树对象的 apply 方法结果一致
    assert_array_equal(est.apply(X_small), est.tree_.apply(X_small32))


# 对稀疏树的名称和 CSR 容器参数化测试
@pytest.mark.parametrize("name", SPARSE_TREES)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_public_apply_sparse_trees(name, csr_container):
    # 将 X_small 转换为指定类型（X_small32），以保证数据类型一致性
    X_small32 = csr_container(X_small.astype(tree._tree.DTYPE, copy=False))

    # 创建特定树估计器对象，拟合 X_small 和 y_small 数据
    est = ALL_TREES[name]()
    est.fit(X_small, y_small)

    # 断言应用该估计器的 apply 方法结果与其树对象的 apply 方法结果一致
    assert_array_equal(est.apply(X_small), est.tree_.apply(X_small32))


# 测试硬编码决策路径
def test_decision_path_hardcoded():
    # 使用鸢尾花数据集的特征数据 X 和目标数据 y
    X = iris.data
    y = iris.target

    # 创建决策树分类器估计器对象，设置最大深度为 1，拟合数据
    est = DecisionTreeClassifier(random_state=0, max_depth=1).fit(X, y)

    # 获取 X 的前两个样本的决策路径表示，并转换为稀疏矩阵数组
    node_indicator = est.decision_path(X[:2]).toarray()

    # 断言计算得到的决策路径与预期的结果一致
    assert_array_equal(node_indicator, [[1, 1, 0], [1, 0, 1]])


# 对每个树的名称参数化测试
@pytest.mark.parametrize("name", ALL_TREES)
def test_decision_path(name):
    # 使用鸢尾花数据集的特征数据 X 和目标数据 y
    X = iris.data
    y = iris.target
    n_samples = X.shape[0]

    # 根据树的名称获取相应的树估计器类
    TreeEstimator = ALL_TREES[name]
    est = TreeEstimator(random_state=0, max_depth=2)
    est.fit(X, y)

    # 获取 X 的决策路径的稀疏表示
    node_indicator_csr = est.decision_path(X)
    node_indicator = node_indicator_csr.toarray()

    # 断言节点指示器的形状与预期一致
    assert node_indicator.shape == (n_samples, est.tree_.node_count)

    # 断言叶子节点索引正确性
    leaves = est.apply(X)
    leave_indicator = [node_indicator[i, j] for i, j in enumerate(leaves)]
    assert_array_almost_equal(leave_indicator, np.ones(shape=n_samples))

    # 确保每个样本仅有一个叶子节点
    all_leaves = est.tree_.children_left == TREE_LEAF
    assert_array_almost_equal(
        np.dot(node_indicator, all_leaves), np.ones(shape=n_samples)
    )

    # 确保最大深度与节点指示器的和一致
    max_depth = node_indicator.sum(axis=1).max()
    assert est.tree_.max_depth <= max_depth


# 对每个树的名称和 CSR 容器参数化测试
@pytest.mark.parametrize("name", ALL_TREES)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_no_sparse_y_support(name, csr_container):
    # 目前不支持稀疏的 y
    X, y = X_multilabel, csr_container(y_multilabel)
    # 从 ALL_TREES 字典中获取名称为 `name` 的树估算器类
    TreeEstimator = ALL_TREES[name]
    # 使用 pytest 断言检查是否会引发 TypeError 异常
    with pytest.raises(TypeError):
        # 创建一个具有随机状态参数为 0 的 TreeEstimator 实例，并对数据集 X, y 进行拟合
        TreeEstimator(random_state=0).fit(X, y)
# 定义一个测试函数，用于验证均方绝对误差准则在小型测试数据集上的正确性
def test_mae():
    """Check MAE criterion produces correct results on small toy dataset:

    ------------------
    | X | y | weight |
    ------------------
    | 3 | 3 |  0.1   |
    | 5 | 3 |  0.3   |
    | 8 | 4 |  1.0   |
    | 3 | 6 |  0.6   |
    | 5 | 7 |  0.3   |
    ------------------
    |sum wt:|  2.3   |
    ------------------

    因为我们处理的是样本权重，所以不能简单地选择/平均中心值，而是考虑累积权重达到50%的中位数（在排序后的y数据集中）。
    因此，根据这个测试数据集，当y = 4时，累积权重>= 50%。因此：
    中位数 = 4

    对于所有样本，可以通过以下方式计算总误差：
    绝对值(中位数 - y) * weight

    即，总误差 = (|4 - 3| * 0.1)
              + (|4 - 3| * 0.3)
              + (|4 - 4| * 1.0)
              + (|4 - 6| * 0.6)
              + (|4 - 7| * 0.3)
              = 2.5

    杂质 = 总误差 / 总权重
         = 2.5 / 2.3
         = 1.08695652173913
         ------------------

    从根节点开始，下一个最佳分割点是在X值为3和5之间。因此，我们有左右子节点：

    LEFT                    RIGHT
    ------------------      ------------------
    | X | y | weight |      | X | y | weight |
    ------------------      ------------------
    | 3 | 3 |  0.1   |      | 5 | 3 |  0.3   |
    | 3 | 6 |  0.6   |      | 8 | 4 |  1.0   |
    ------------------      | 5 | 7 |  0.3   |
    |sum wt:|  0.7   |      ------------------
    ------------------      |sum wt:|  1.6   |
                            ------------------

    杂质的计算方法相同：
    左节点 中位数 = 6
    总误差 = (|6 - 3| * 0.1)
           + (|6 - 6| * 0.6)
           = 0.3

    左节点 杂质 = 总误差 / 总权重
            = 0.3 / 0.7
            = 0.428571428571429
            -------------------

    同样适用于右节点：
    右节点 中位数 = 4
    总误差 = (|4 - 3| * 0.3)
           + (|4 - 4| * 1.0)
           + (|4 - 7| * 0.3)
           = 1.2

    右节点 杂质 = 总误差 / 总权重
            = 1.2 / 1.6
            = 0.75
            ------
    """

    # 创建一个决策树回归器对象，使用绝对误差准则，最大叶节点数为2
    dt_mae = DecisionTreeRegressor(
        random_state=0, criterion="absolute_error", max_leaf_nodes=2
    )

    # 测试带有非均匀样本权重的均方绝对误差计算（如上所示的测试数据集）
    dt_mae.fit(
        X=[[3], [5], [3], [8], [5]],
        y=[6, 7, 3, 4, 3],
        sample_weight=[0.6, 0.3, 0.1, 1.0, 0.3],
    )

    # 断言树节点的杂质等于预期值
    assert_allclose(dt_mae.tree_.impurity, [2.5 / 2.3, 0.3 / 0.7, 1.2 / 1.6])

    # 断言树节点的值平坦化后等于预期值
    assert_array_equal(dt_mae.tree_.value.flat, [4.0, 6.0, 4.0])

    # 测试所有样本权重均匀的均方绝对误差计算：
    # (此处略去未完成的测试代码，因为不在要求的注释范围内)
    # 使用 MAE 决策树回归器 dt_mae 对数据进行拟合，指定样本特征 X、目标值 y 和样本权重为全为1的数组
    dt_mae.fit(X=[[3], [5], [3], [8], [5]], y=[6, 7, 3, 4, 3], sample_weight=np.ones(5))
    # 断言决策树的不纯度 impurity 数组与预期的值相等
    assert_array_equal(dt_mae.tree_.impurity, [1.4, 1.5, 4.0 / 3.0])
    # 断言决策树的值 value 数组扁平化后与预期的值相等
    assert_array_equal(dt_mae.tree_.value.flat, [4, 4.5, 4.0])

    # 测试不显式提供 `sample_weight` 时的 MAE，此时相当于提供均匀的样本权重，尽管内部逻辑不同：
    # 使用 MAE 决策树回归器 dt_mae 对数据进行拟合，指定样本特征 X 和目标值 y，没有指定样本权重
    dt_mae.fit(X=[[3], [5], [3], [8], [5]], y=[6, 7, 3, 4, 3])
    # 断言决策树的不纯度 impurity 数组与预期的值相等
    assert_array_equal(dt_mae.tree_.impurity, [1.4, 1.5, 4.0 / 3.0])
    # 断言决策树的值 value 数组扁平化后与预期的值相等
    assert_array_equal(dt_mae.tree_.value.flat, [4, 4.5, 4.0])
# 定义测试函数，用于检验复制条件是否保持与原始对象相同的类型和属性
def test_criterion_copy():
    # 定义输出数量
    n_outputs = 3
    # 创建包含整数类型的类别数组
    n_classes = np.arange(3, dtype=np.intp)
    # 设定样本数量
    n_samples = 100

    # 定义用于深度复制对象的函数
    def _pickle_copy(obj):
        return pickle.loads(pickle.dumps(obj))

    # 遍历不同的复制函数：浅复制、深度复制、pickle 复制
    for copy_func in [copy.copy, copy.deepcopy, _pickle_copy]:
        # 遍历分类器字典中的每个键值对
        for _, typename in CRITERIA_CLF.items():
            # 使用类名创建分类器对象，传入输出数量和类别数组
            criteria = typename(n_outputs, n_classes)
            # 对 criteria 进行复制，并获取其__reduce__结果
            result = copy_func(criteria).__reduce__()
            # 解包__reduce__结果，获取类型名、输出数量和类别数组
            typename_, (n_outputs_, n_classes_), _ = result
            # 断言复制后的类型名与原始类型名相同
            assert typename == typename_
            # 断言复制后的输出数量与原始输出数量相同
            assert n_outputs == n_outputs_
            # 断言复制后的类别数组与原始类别数组相同
            assert_array_equal(n_classes, n_classes_)

        # 遍历回归器字典中的每个键值对
        for _, typename in CRITERIA_REG.items():
            # 使用类名创建回归器对象，传入输出数量和样本数量
            criteria = typename(n_outputs, n_samples)
            # 对 criteria 进行复制，并获取其__reduce__结果
            result = copy_func(criteria).__reduce__()
            # 解包__reduce__结果，获取类型名、输出数量和样本数量
            typename_, (n_outputs_, n_samples_), _ = result
            # 断言复制后的类型名与原始类型名相同
            assert typename == typename_
            # 断言复制后的输出数量与原始输出数量相同
            assert n_outputs == n_outputs_
            # 断言复制后的样本数量与原始样本数量相同
            assert n_samples == n_samples_


@pytest.mark.parametrize("sparse_container", [None] + CSC_CONTAINERS)
def test_empty_leaf_infinite_threshold(sparse_container):
    # 尝试通过使用接近无限值来创建空叶节点
    # 创建一个服从标准正态分布的数据矩阵，并乘以接近2e38的值
    data = np.random.RandomState(0).randn(100, 11) * 2e38
    # 将数据中的NaN值替换为0，并转换为float32类型
    data = np.nan_to_num(data.astype("float32"))
    # 提取特征矩阵X和目标变量y
    X = data[:, :-1]
    # 如果指定了稀疏容器类型，则将特征矩阵X转换为稀疏矩阵
    if sparse_container is not None:
        X = sparse_container(X)
    y = data[:, -1]

    # 使用决策树回归器拟合数据
    tree = DecisionTreeRegressor(random_state=0).fit(X, y)
    # 获取每个样本的叶节点索引
    terminal_regions = tree.apply(X)
    # 获取左子树叶节点中的索引
    left_leaf = set(np.where(tree.tree_.children_left == TREE_LEAF)[0])
    # 计算空叶节点的索引
    empty_leaf = left_leaf.difference(terminal_regions)
    # 获取非有限阈值的索引
    infinite_threshold = np.where(~np.isfinite(tree.tree_.threshold))[0]
    # 断言不存在非有限阈值的索引
    assert len(infinite_threshold) == 0
    # 断言不存在空叶节点
    assert len(empty_leaf) == 0


@pytest.mark.parametrize(
    "dataset", sorted(set(DATASETS.keys()) - {"reg_small", "diabetes"})
)
@pytest.mark.parametrize("tree_cls", [DecisionTreeClassifier, ExtraTreeClassifier])
def test_prune_tree_classifier_are_subtrees(dataset, tree_cls):
    # 获取指定名称的数据集
    dataset = DATASETS[dataset]
    # 提取特征矩阵X和目标变量y
    X, y = dataset["X"], dataset["y"]
    # 创建决策树分类器对象，设置最大叶节点数和随机状态
    est = tree_cls(max_leaf_nodes=20, random_state=0)
    # 计算决策树剪枝路径
    info = est.cost_complexity_pruning_path(X, y)

    # 获取剪枝路径的alpha值
    pruning_path = info.ccp_alphas
    # 获取剪枝路径的不纯度值
    impurities = info.impurities
    # 断言剪枝路径中的alpha值单调递增
    assert np.all(np.diff(pruning_path) >= 0)
    # 断言剪枝路径中的不纯度值单调递增
    assert np.all(np.diff(impurities) >= 0)

    # 断言剪枝后的树是原始树的子树
    assert_pruning_creates_subtree(tree_cls, X, y, pruning_path)


@pytest.mark.parametrize("dataset", DATASETS.keys())
@pytest.mark.parametrize("tree_cls", [DecisionTreeRegressor, ExtraTreeRegressor])
def test_prune_tree_regression_are_subtrees(dataset, tree_cls):
    # 获取指定名称的数据集
    dataset = DATASETS[dataset]
    # 提取特征矩阵X和目标变量y
    X, y = dataset["X"], dataset["y"]

    # 创建决策树回归器对象，设置最大叶节点数和随机状态
    est = tree_cls(max_leaf_nodes=20, random_state=0)
    # 计算决策树剪枝路径
    info = est.cost_complexity_pruning_path(X, y)

    # 获取剪枝路径的alpha值
    pruning_path = info.ccp_alphas
    # 获取剪枝路径的不纯度值
    impurities = info.impurities
    # 断言剪枝路径中的alpha值单调递增
    assert np.all(np.diff(pruning_path) >= 0)
    # 断言：验证 `impurities` 数组中相邻元素的差值均大于等于零
    assert np.all(np.diff(impurities) >= 0)
    
    # 断言：验证修剪路径 `pruning_path` 通过使用给定的决策树类 `tree_cls`、输入数据 `X` 和目标变量 `y` 后能够创建子树
    assert_pruning_creates_subtree(tree_cls, X, y, pruning_path)
def test_prune_single_node_tree():
    # 定义测试函数，用于验证单节点树的剪枝操作

    # 创建一个决策树分类器对象，设置随机种子为0
    clf1 = DecisionTreeClassifier(random_state=0)
    # 使用数据 [[0], [1]] 和标签 [0, 0] 对分类器进行训练
    clf1.fit([[0], [1]], [0, 0])

    # 创建一个带有剪枝参数的决策树分类器对象，设置随机种子为0，ccp_alpha参数为10
    clf2 = DecisionTreeClassifier(random_state=0, ccp_alpha=10)
    # 使用数据 [[0], [1]] 和标签 [0, 0] 对分类器进行训练
    clf2.fit([[0], [1]], [0, 0])

    # 验证 clf2 的树结构是 clf1 树结构的子树
    assert_is_subtree(clf1.tree_, clf2.tree_)


def assert_pruning_creates_subtree(estimator_cls, X, y, pruning_path):
    # 确保剪枝操作能够创建子树

    # 使用不同的剪枝参数生成多个决策树估计器对象
    estimators = []
    for ccp_alpha in pruning_path:
        # 创建一个带有最大叶子节点数和给定剪枝参数的估计器对象，设置随机种子为0，对数据 X, y 进行训练
        est = estimator_cls(max_leaf_nodes=20, ccp_alpha=ccp_alpha, random_state=0).fit(
            X, y
        )
        estimators.append(est)

    # 验证每个剪枝后的树是前一个树的子树（前一个树使用了较小的剪枝参数）
    for prev_est, next_est in zip(estimators, estimators[1:]):
        assert_is_subtree(prev_est.tree_, next_est.tree_)


def assert_is_subtree(tree, subtree):
    # 确保 subtree 是 tree 的子树

    # 确保 subtree 的节点数不超过 tree 的节点数
    assert tree.node_count >= subtree.node_count
    # 确保 subtree 的最大深度不超过 tree 的最大深度
    assert tree.max_depth >= subtree.max_depth

    # 获取 tree 和 subtree 的左右子树索引
    tree_c_left = tree.children_left
    tree_c_right = tree.children_right
    subtree_c_left = subtree.children_left
    subtree_c_right = subtree.children_right

    stack = [(0, 0)]
    while stack:
        tree_node_idx, subtree_node_idx = stack.pop()

        # 确保节点值相等
        assert_array_almost_equal(
            tree.value[tree_node_idx], subtree.value[subtree_node_idx]
        )
        # 确保节点不纯度相等
        assert_almost_equal(
            tree.impurity[tree_node_idx], subtree.impurity[subtree_node_idx]
        )
        # 确保节点样本数相等
        assert_almost_equal(
            tree.n_node_samples[tree_node_idx], subtree.n_node_samples[subtree_node_idx]
        )
        # 确保加权节点样本数相等
        assert_almost_equal(
            tree.weighted_n_node_samples[tree_node_idx],
            subtree.weighted_n_node_samples[subtree_node_idx],
        )

        if subtree_c_left[subtree_node_idx] == subtree_c_right[subtree_node_idx]:
            # 如果是叶子节点，确保其阈值为未定义值
            assert_almost_equal(TREE_UNDEFINED, subtree.threshold[subtree_node_idx])
        else:
            # 如果不是叶子节点，确保其阈值与 tree 中对应节点的阈值相等，并将左右子树索引入栈
            assert_almost_equal(
                tree.threshold[tree_node_idx], subtree.threshold[subtree_node_idx]
            )
            stack.append((tree_c_left[tree_node_idx], subtree_c_left[subtree_node_idx]))
            stack.append(
                (tree_c_right[tree_node_idx], subtree_c_right[subtree_node_idx])
            )


@pytest.mark.parametrize("name", ALL_TREES)
@pytest.mark.parametrize("splitter", ["best", "random"])
@pytest.mark.parametrize("sparse_container", [None] + CSC_CONTAINERS + CSR_CONTAINERS)
def test_apply_path_readonly_all_trees(name, splitter, sparse_container):
    # 对所有树应用只读路径的测试

    dataset = DATASETS["clf_small"]
    # 将数据集中的 X 转换为指定类型的数据，确保不复制数据
    X_small = dataset["X"].astype(tree._tree.DTYPE, copy=False)
    if sparse_container is None:
        # 创建只读的内存映射数据 X_readonly
        X_readonly = create_memmap_backed_data(X_small)
    else:
        # 创建稀疏矩阵容器，存储数据集中的特征数据
        X_readonly = sparse_container(dataset["X"])

        # 将特征数据转换为指定类型的 NumPy 数组
        X_readonly.data = np.array(X_readonly.data, dtype=tree._tree.DTYPE)

        # 使用内存映射方式创建数据，返回元组的形式包含了映射后的数据
        (
            X_readonly.data,
            X_readonly.indices,
            X_readonly.indptr,
        ) = create_memmap_backed_data(
            (X_readonly.data, X_readonly.indices, X_readonly.indptr)
        )

    # 使用内存映射方式创建数据，返回映射后的数据作为只读的特征数据
    y_readonly = create_memmap_backed_data(np.array(y_small, dtype=tree._tree.DTYPE))

    # 根据指定名称创建所有决策树的估计器对象
    est = ALL_TREES[name](splitter=splitter)

    # 使用估计器对象拟合特征数据和目标数据
    est.fit(X_readonly, y_readonly)

    # 断言预测结果与小规模数据集的预测结果相等
    assert_array_equal(est.predict(X_readonly), est.predict(X_small))

    # 断言决策路径的稠密表示结果与小规模数据集的决策路径稠密表示结果相等
    assert_array_equal(
        est.decision_path(X_readonly).todense(), est.decision_path(X_small).todense()
    )
@pytest.mark.parametrize("criterion", ["squared_error", "friedman_mse", "poisson"])
# 使用pytest的parametrize装饰器，定义了参数化测试，测试不同的损失函数类型
@pytest.mark.parametrize("Tree", REG_TREES.values())
# 使用pytest的parametrize装饰器，定义了参数化测试，测试不同的回归树模型

def test_balance_property(criterion, Tree):
    # 测试在训练集上预测值的总和是否等于真实值的总和。
    # 如果预测均值，则应该成立（甚至对每个叶子节点也应该成立）。
    # MAE（平均绝对误差）预测中位数，因此在此测试中排除。

    # 选择一个具有非负目标的训练集（适用于poisson损失函数）
    X, y = diabetes.data, diabetes.target
    reg = Tree(criterion=criterion)
    # 使用选择的回归树模型拟合数据
    reg.fit(X, y)
    # 断言预测值的总和近似等于真实值的总和
    assert np.sum(reg.predict(X)) == pytest.approx(np.sum(y))


@pytest.mark.parametrize("seed", range(3))
# 使用pytest的parametrize装饰器，定义了参数化测试，测试不同的随机种子

def test_poisson_zero_nodes(seed):
    # 测试确保在节点上预测值的总和为零，因此预测值应该不为零。
    X = [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 2], [1, 2], [1, 3]]
    y = [0, 0, 0, 0, 1, 2, 3, 4]
    # 注意X[:, 0] == 0完全指示y == 0。树可以轻松学习到这一点。
    reg = DecisionTreeRegressor(criterion="squared_error", random_state=seed)
    # 使用平方误差作为损失函数的决策树回归模型
    reg.fit(X, y)
    # 断言预测值的最小值为0
    assert np.amin(reg.predict(X)) == 0

    # 对于Poisson分布，预测值必须严格为正数
    reg = DecisionTreeRegressor(criterion="poisson", random_state=seed)
    # 使用Poisson损失函数的决策树回归模型
    reg.fit(X, y)
    # 断言所有预测值都大于0
    assert np.all(reg.predict(X) > 0)

    # 测试附加数据集，可能出现问题。
    n_features = 10
    X, y = datasets.make_regression(
        effective_rank=n_features * 2 // 3,
        tail_strength=0.6,
        n_samples=1_000,
        n_features=n_features,
        n_informative=n_features * 2 // 3,
        random_state=seed,
    )
    # 一些多余的零值
    y[(-1 < y) & (y < 0)] = 0
    # 确保目标值为正数
    y = np.abs(y)
    reg = DecisionTreeRegressor(criterion="poisson", random_state=seed)
    # 使用Poisson损失函数的决策树回归模型
    reg.fit(X, y)
    # 断言所有预测值都大于0
    assert np.all(reg.predict(X) > 0)


def test_poisson_vs_mse():
    # 对于Poisson分布的目标，Poisson损失函数应该比平方误差损失函数给出更好的结果，
    # 使用Poisson偏差作为度量方式来衡量。
    # 在sklearn/ensemble/_hist_gradient_boosting/tests/test_gradient_boosting.py中有一个类似的测试test_poisson()。

    rng = np.random.RandomState(42)
    n_train, n_test, n_features = 500, 500, 10
    X = datasets.make_low_rank_matrix(
        n_samples=n_train + n_test, n_features=n_features, random_state=rng
    )
    # 创建一个对数线性的Poisson模型，并缩放coef因为它将会被指数化。
    coef = rng.uniform(low=-2, high=2, size=n_features) / np.max(X, axis=0)
    y = rng.poisson(lam=np.exp(X @ coef))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, random_state=rng
    )
    # 通过设置min_samples_split=10来防止一些过拟合。
    tree_poi = DecisionTreeRegressor(
        criterion="poisson", min_samples_split=10, random_state=rng
    )
    # 使用Poisson损失函数的决策树回归模型
    tree_mse = DecisionTreeRegressor(
        criterion="squared_error", min_samples_split=10, random_state=rng
    )
    # 使用训练数据拟合基于泊松回归的决策树模型
    tree_poi.fit(X_train, y_train)
    # 使用训练数据拟合基于均方误差的决策树模型
    tree_mse.fit(X_train, y_train)
    # 使用均值策略拟合虚拟回归器，用于作为基准模型
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)

    # 对于训练集和测试集的每一组数据 (X, y)，以及对应的标识符（"train" 或 "test"）
    for X, y, val in [(X_train, y_train, "train"), (X_test, y_test, "test")]:
        # 计算基于泊松回归的决策树模型在当前数据集上的平均泊松偏差
        metric_poi = mean_poisson_deviance(y, tree_poi.predict(X))
        # 由于均方误差可能产生非正预测值，因此在预测结果中进行截断处理
        metric_mse = mean_poisson_deviance(y, np.clip(tree_mse.predict(X), 1e-15, None))
        # 计算虚拟回归器在当前数据集上的平均泊松偏差
        metric_dummy = mean_poisson_deviance(y, dummy.predict(X))
        
        # 如果当前数据集为测试集
        if val == "test":
            # 断言：基于泊松回归的模型的偏差应当小于均方误差模型偏差的一半
            assert metric_poi < 0.5 * metric_mse
        # 断言：基于泊松回归的模型的偏差应当小于虚拟回归器模型偏差的百分之七十五
        assert metric_poi < 0.75 * metric_dummy
@pytest.mark.parametrize("criterion", REG_CRITERIONS)
def test_decision_tree_regressor_sample_weight_consistency(criterion):
    """Test that the impact of sample_weight is consistent."""
    # 设置决策树参数，使用给定的评估标准（criterion）
    tree_params = dict(criterion=criterion)
    # 创建决策树回归器对象，使用指定参数和随机种子
    tree = DecisionTreeRegressor(**tree_params, random_state=42)
    
    # 对于每种样本权重类型进行验证
    for kind in ["zeros", "ones"]:
        # 调用函数检查样本权重的不变性
        check_sample_weights_invariance(
            "DecisionTreeRegressor_" + criterion, tree, kind="zeros"
        )

    # 设置随机数生成器和样本数、特征数
    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 5

    # 创建随机样本数据 X 和目标变量 y
    X = rng.rand(n_samples, n_features)
    y = np.mean(X, axis=1) + rng.rand(n_samples)
    # 使目标变量为正，以适应泊松标准
    y += np.min(y) + 0.1

    # 检查通过将样本权重乘以2是否等同于相应样本的重复
    X2 = np.concatenate([X, X[: n_samples // 2]], axis=0)
    y2 = np.concatenate([y, y[: n_samples // 2]])
    sample_weight_1 = np.ones(len(y))
    sample_weight_1[: n_samples // 2] = 2

    # 使用样本权重 sample_weight_1 训练第一个决策树回归器
    tree1 = DecisionTreeRegressor(**tree_params).fit(
        X, y, sample_weight=sample_weight_1
    )

    # 使用无样本权重训练第二个决策树回归器
    tree2 = DecisionTreeRegressor(**tree_params).fit(X2, y2, sample_weight=None)

    # 断言两个决策树的节点数相同
    assert tree1.tree_.node_count == tree2.tree_.node_count
    # 虽然阈值（thresholds）、tree.tree_.threshold 和值（values）、tree.tree_.value 不完全相同，
    # 但在训练集上这些差异不重要，因此预测结果应该相同。
    assert_allclose(tree1.predict(X), tree2.predict(X))


@pytest.mark.parametrize("Tree", [DecisionTreeClassifier, ExtraTreeClassifier])
@pytest.mark.parametrize("n_classes", [2, 4])
def test_criterion_entropy_same_as_log_loss(Tree, n_classes):
    """Test that criterion=entropy gives same as log_loss."""
    # 设置样本数和特征数
    n_samples, n_features = 50, 5
    # 生成分类数据集 X, y
    X, y = datasets.make_classification(
        n_classes=n_classes,
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        random_state=42,
    )
    # 使用 criterion='log_loss' 训练决策树分类器
    tree_log_loss = Tree(criterion="log_loss", random_state=43).fit(X, y)
    # 使用 criterion='entropy' 训练决策树分类器
    tree_entropy = Tree(criterion="entropy", random_state=43).fit(X, y)

    # 断言使用 criterion='entropy' 和 criterion='log_loss' 训练出的树结构相同
    assert_tree_equal(
        tree_log_loss.tree_,
        tree_entropy.tree_,
        f"{Tree!r} with criterion 'entropy' and 'log_loss' gave different trees.",
    )
    # 断言使用 criterion='entropy' 和 criterion='log_loss' 训练出的预测结果相近
    assert_allclose(tree_log_loss.predict(X), tree_entropy.predict(X))


def test_different_endianness_pickle():
    # 生成分类数据集 X, y
    X, y = datasets.make_classification(random_state=0)

    # 创建决策树分类器
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)
    clf.fit(X, y)
    # 计算分类准确率
    score = clf.score(X, y)

    # 定义函数，用于调整 ndarray 的字节顺序
    def reduce_ndarray(arr):
        return arr.byteswap().view(arr.dtype.newbyteorder()).__reduce__()

    # 获取非本地字节顺序的 pickle 文件
    def get_pickle_non_native_endianness():
        f = io.BytesIO()
        p = pickle.Pickler(f)
        p.dispatch_table = copyreg.dispatch_table.copy()
        p.dispatch_table[np.ndarray] = reduce_ndarray

        p.dump(clf)
        f.seek(0)
        return f
    ultimate
    
    # 加载经过处理以适应非本地字节顺序的 pickle 对象，创建一个新的分类器对象
    new_clf = pickle.load(get_pickle_non_native_endianness())
    
    # 使用新的分类器对象评估输入数据 X 对应的预测结果与真实标签 y 之间的得分
    new_score = new_clf.score(X, y)
    
    # 断言原始得分与新得分在数值上接近
    assert np.isclose(score, new_score)
# 使用 scikit-learn 提供的 make_classification 函数生成随机数据集 X 和标签 y
X, y = datasets.make_classification(random_state=0)

# 使用 DecisionTreeClassifier 创建决策树分类器 clf，设定随机种子和最大深度为3，并训练模型
clf = DecisionTreeClassifier(random_state=0, max_depth=3)
clf.fit(X, y)

# 计算分类器在训练数据 X 上的得分
score = clf.score(X, y)

# 定义一个类 NonNativeEndiannessNumpyPickler，继承自 NumpyPickler，用于处理非本机字节序的 Numpy 对象
class NonNativeEndiannessNumpyPickler(NumpyPickler):
    def save(self, obj):
        # 如果对象是 ndarray 类型，则进行字节交换和字节序转换
        if isinstance(obj, np.ndarray):
            obj = obj.byteswap().view(obj.dtype.newbyteorder())
        super().save(obj)

# 定义函数 get_joblib_pickle_non_native_endianness，返回一个存储 clf 序列化后数据的 BytesIO 对象
def get_joblib_pickle_non_native_endianness():
    f = io.BytesIO()
    p = NonNativeEndiannessNumpyPickler(f)
    p.dump(clf)
    f.seek(0)
    return f

# 使用 joblib.load 从 BytesIO 对象中加载序列化后的数据，重新创建 clf 模型
new_clf = joblib.load(get_joblib_pickle_non_native_endianness())

# 计算重新创建的 clf 模型在训练数据 X 上的得分
new_score = new_clf.score(X, y)

# 断言原始模型和重新加载后模型的得分相近
assert np.isclose(score, new_score)


# 定义函数 get_different_bitness_node_ndarray，将 node_ndarray 的数据类型改为与 _IS_32BIT 对应的 int 类型
def get_different_bitness_node_ndarray(node_ndarray):
    new_dtype_for_indexing_fields = np.int64 if _IS_32BIT else np.int32

    # 定义 Node 结构体中 SIZE_t 类型的字段名列表
    indexing_field_names = ["left_child", "right_child", "feature", "n_node_samples"]

    # 创建新的数据类型字典 new_dtype_dict，将 Node 结构体中的字段名和对应的数据类型映射起来
    new_dtype_dict = {
        name: dtype for name, (dtype, _) in node_ndarray.dtype.fields.items()
    }

    # 将 SIZE_t 类型的字段改为 new_dtype_for_indexing_fields 类型
    for name in indexing_field_names:
        new_dtype_dict[name] = new_dtype_for_indexing_fields

    # 根据新的数据类型字典创建新的数据类型 new_dtype，并将 node_ndarray 转换为这种新的数据类型
    new_dtype = np.dtype(
        {"names": list(new_dtype_dict.keys()), "formats": list(new_dtype_dict.values())}
    )
    return node_ndarray.astype(new_dtype, casting="same_kind")


# 定义函数 get_different_alignment_node_ndarray，根据偏移量调整 node_ndarray 的数据类型
def get_different_alignment_node_ndarray(node_ndarray):
    # 创建新的数据类型字典 new_dtype_dict，将 Node 结构体中的字段名和对应的数据类型映射起来
    new_dtype_dict = {
        name: dtype for name, (dtype, _) in node_ndarray.dtype.fields.items()
    }

    # 获取 Node 结构体中每个字段的偏移量列表
    offsets = [offset for dtype, offset in node_ndarray.dtype.fields.values()]

    # 将偏移量列表中的每个偏移量增加 8
    shifted_offsets = [8 + offset for offset in offsets]

    # 根据新的数据类型字典和调整后的偏移量列表创建新的数据类型 new_dtype
    new_dtype = np.dtype(
        {
            "names": list(new_dtype_dict.keys()),
            "formats": list(new_dtype_dict.values()),
            "offsets": shifted_offsets,
        }
    )
    return node_ndarray.astype(new_dtype, casting="same_kind")


# 定义函数 reduce_tree_with_different_bitness，将树结构中的数据类型改为与 _IS_32BIT 对应的 int 类型
def reduce_tree_with_different_bitness(tree):
    new_dtype = np.int64 if _IS_32BIT else np.int32

    # 调用 tree 对象的 __reduce__ 方法获取类名、特征数、类别数和状态信息
    tree_cls, (n_features, n_classes, n_outputs), state = tree.__reduce__()

    # 将 n_classes 数据类型转换为 new_dtype
    new_n_classes = n_classes.astype(new_dtype, casting="same_kind")

    # 复制状态信息，并将状态信息中的 nodes 字段转换为与 _IS_32BIT 对应的数据类型
    new_state = state.copy()
    new_state["nodes"] = get_different_bitness_node_ndarray(new_state["nodes"])

    # 返回新的类名、特征数、新的类别数和输出数，以及更新后的状态信息
    return (tree_cls, (n_features, new_n_classes, n_outputs), new_state)


# 定义函数 pickle_dump_with_different_bitness，使用 pickle 序列化 clf 并修改其数据类型与 _IS_32BIT 对应的 int 类型
def pickle_dump_with_different_bitness():
    f = io.BytesIO()
    p = pickle.Pickler(f)

    # 复制 pickle 序列化器的分发表，并为 CythonTree 类型设置 reduce 函数为 reduce_tree_with_different_bitness
    p.dispatch_table = copyreg.dispatch_table.copy()
    p.dispatch_table[CythonTree] = reduce_tree_with_different_bitness

    # 序列化 clf 对象到 BytesIO 对象 f 中
    p.dump(clf)
    f.seek(0)
    return f
    # 从一个具有不同比特位的 pickle 数据流中加载新的分类器对象
    new_clf = pickle.load(pickle_dump_with_different_bitness())
    # 使用新加载的分类器对象对给定的数据集 X 和标签 y 进行评分
    new_score = new_clf.score(X, y)
    # 使用 pytest 的近似断言检查原来的评分 score 是否与新评分 new_score 近似相等
    assert score == pytest.approx(new_score)
# 测试不同位数的 joblib pickle 文件
def test_different_bitness_joblib_pickle():
    # 创建一个随机分类数据集 X, y
    X, y = datasets.make_classification(random_state=0)

    # 初始化一个深度为3的决策树分类器
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)
    # 使用 X, y 训练分类器
    clf.fit(X, y)
    # 计算分类器在训练数据上的得分
    score = clf.score(X, y)

    # 定义一个函数，用于生成不同位数的 joblib dump
    def joblib_dump_with_different_bitness():
        # 创建一个字节流对象
        f = io.BytesIO()
        # 使用 NumpyPickler 对象对字节流进行序列化
        p = NumpyPickler(f)
        # 复制默认的 dispatch table 并注册 CythonTree 类型的特殊处理函数
        p.dispatch_table = copyreg.dispatch_table.copy()
        p.dispatch_table[CythonTree] = reduce_tree_with_different_bitness

        # 序列化 clf 对象到字节流中
        p.dump(clf)
        # 将文件指针位置移动到文件开头
        f.seek(0)
        # 返回序列化后的字节流对象
        return f

    # 从 joblib dump 加载新的分类器对象
    new_clf = joblib.load(joblib_dump_with_different_bitness())
    # 计算新分类器在相同数据上的得分
    new_score = new_clf.score(X, y)
    # 断言原始分类器和新分类器的得分应该非常接近
    assert score == pytest.approx(new_score)


# 测试检查 n_classes 的函数
def test_check_n_classes():
    # 根据当前机器位数确定预期的数据类型
    expected_dtype = np.dtype(np.int32) if _IS_32BIT else np.dtype(np.int64)
    # 允许的数据类型列表
    allowed_dtypes = [np.dtype(np.int32), np.dtype(np.int64)]
    allowed_dtypes += [dt.newbyteorder() for dt in allowed_dtypes]

    # 创建一个包含两个元素的数组 n_classes，数据类型为 expected_dtype
    n_classes = np.array([0, 1], dtype=expected_dtype)
    # 遍历允许的数据类型，检查 _check_n_classes 函数的行为
    for dt in allowed_dtypes:
        _check_n_classes(n_classes.astype(dt), expected_dtype)

    # 使用错误的维度创建 n_classes 数组，预期会引发 ValueError 异常
    with pytest.raises(ValueError, match="Wrong dimensions.+n_classes"):
        wrong_dim_n_classes = np.array([[0, 1]], dtype=expected_dtype)
        _check_n_classes(wrong_dim_n_classes, expected_dtype)

    # 使用错误的数据类型创建 n_classes 数组，预期会引发 ValueError 异常
    with pytest.raises(ValueError, match="n_classes.+incompatible dtype"):
        wrong_dtype_n_classes = n_classes.astype(np.float64)
        _check_n_classes(wrong_dtype_n_classes, expected_dtype)


# 测试检查 value ndarray 的函数
def test_check_value_ndarray():
    # 预期的数据类型和形状
    expected_dtype = np.dtype(np.float64)
    expected_shape = (5, 1, 2)
    # 创建一个值数组 value_ndarray，数据类型和形状符合预期
    value_ndarray = np.zeros(expected_shape, dtype=expected_dtype)

    # 允许的数据类型列表
    allowed_dtypes = [expected_dtype, expected_dtype.newbyteorder()]

    # 遍历允许的数据类型，检查 _check_value_ndarray 函数的行为
    for dt in allowed_dtypes:
        _check_value_ndarray(
            value_ndarray, expected_dtype=dt, expected_shape=expected_shape
        )

    # 使用错误的形状创建 value_ndarray 数组，预期会引发 ValueError 异常
    with pytest.raises(ValueError, match="Wrong shape.+value array"):
        _check_value_ndarray(
            value_ndarray, expected_dtype=expected_dtype, expected_shape=(1, 2)
        )

    # 使用问题数组创建 value_ndarray，预期会引发 ValueError 异常，因为它不是 C-contiguous
    for problematic_arr in [value_ndarray[:, :, :1], np.asfortranarray(value_ndarray)]:
        with pytest.raises(ValueError, match="value array.+C-contiguous"):
            _check_value_ndarray(
                problematic_arr,
                expected_dtype=expected_dtype,
                expected_shape=problematic_arr.shape,
            )

    # 使用不兼容的数据类型创建 value_ndarray 数组，预期会引发 ValueError 异常
    with pytest.raises(ValueError, match="value array.+incompatible dtype"):
        _check_value_ndarray(
            value_ndarray.astype(np.float32),
            expected_dtype=expected_dtype,
            expected_shape=expected_shape,
        )
    # 定义期望的节点数据类型
    expected_dtype = NODE_DTYPE

    # 创建一个形状为 (5,) 的零数组，使用期望的数据类型
    node_ndarray = np.zeros((5,), dtype=expected_dtype)

    # 创建包含多种有效节点数组的列表
    valid_node_ndarrays = [
        node_ndarray,  # 原始的节点数组
        get_different_bitness_node_ndarray(node_ndarray),  # 具有不同位数的节点数组
        get_different_alignment_node_ndarray(node_ndarray),  # 具有不同对齐方式的节点数组
    ]

    # 将每个数组转换为其新的字节顺序后添加到列表中
    valid_node_ndarrays += [
        arr.astype(arr.dtype.newbyteorder()) for arr in valid_node_ndarrays
    ]

    # 对于每个有效的节点数组，执行节点数组检查函数
    for arr in valid_node_ndarrays:
        _check_node_ndarray(node_ndarray, expected_dtype=expected_dtype)

    # 使用 pytest 来断言抛出值错误异常，匹配指定的错误信息
    with pytest.raises(ValueError, match="Wrong dimensions.+node array"):
        # 创建一个形状为 (5, 2) 的零数组，使用期望的数据类型
        problematic_node_ndarray = np.zeros((5, 2), dtype=expected_dtype)
        # 执行节点数组检查函数，预期抛出值错误异常
        _check_node_ndarray(problematic_node_ndarray, expected_dtype=expected_dtype)

    # 使用 pytest 来断言抛出值错误异常，匹配指定的错误信息
    with pytest.raises(ValueError, match="node array.+C-contiguous"):
        # 创建一个不连续的节点数组，从原始节点数组中获取每隔一个元素
        problematic_node_ndarray = node_ndarray[::2]
        # 执行节点数组检查函数，预期抛出值错误异常
        _check_node_ndarray(problematic_node_ndarray, expected_dtype=expected_dtype)

    # 创建一个字典，映射字段名称到字段的数据类型，从原始节点数组的字段中提取
    dtype_dict = {name: dtype for name, (dtype, _) in node_ndarray.dtype.fields.items()}

    # 创建一个新的字段数据类型字典，复制现有的字段数据类型
    new_dtype_dict = dtype_dict.copy()
    # 将 'threshold' 字段的数据类型更改为 np.int64
    new_dtype_dict["threshold"] = np.int64

    # 创建新的数据类型，使用新的字段数据类型字典
    new_dtype = np.dtype(
        {"names": list(new_dtype_dict.keys()), "formats": list(new_dtype_dict.values())}
    )

    # 将原始节点数组转换为新的数据类型，创建一个具有不兼容数据类型的节点数组
    problematic_node_ndarray = node_ndarray.astype(new_dtype)

    # 使用 pytest 来断言抛出值错误异常，匹配指定的错误信息
    with pytest.raises(ValueError, match="node array.+incompatible dtype"):
        # 执行节点数组检查函数，预期抛出值错误异常
        _check_node_ndarray(problematic_node_ndarray, expected_dtype=expected_dtype)

    # 创建一个新的字段数据类型字典，复制现有的字段数据类型
    new_dtype_dict = dtype_dict.copy()
    # 将 'left_child' 字段的数据类型更改为 np.float64
    new_dtype_dict["left_child"] = np.float64
    # 创建新的数据类型，使用新的字段数据类型字典
    new_dtype = np.dtype(
        {"names": list(new_dtype_dict.keys()), "formats": list(new_dtype_dict.values())}
    )

    # 将原始节点数组转换为新的数据类型，创建一个具有不兼容数据类型的节点数组
    problematic_node_ndarray = node_ndarray.astype(new_dtype)

    # 使用 pytest 来断言抛出值错误异常，匹配指定的错误信息
    with pytest.raises(ValueError, match="node array.+incompatible dtype"):
        # 执行节点数组检查函数，预期抛出值错误异常
        _check_node_ndarray(problematic_node_ndarray, expected_dtype=expected_dtype)
# 使用 pytest 的参数化装饰器，对 test_splitter_serializable 函数进行多组参数化测试
@pytest.mark.parametrize(
    "Splitter", chain(DENSE_SPLITTERS.values(), SPARSE_SPLITTERS.values())
)
def test_splitter_serializable(Splitter):
    """Check that splitters are serializable."""
    # 设置随机数生成器
    rng = np.random.RandomState(42)
    # 设置最大特征数、输出数和类数
    max_features = 10
    n_outputs, n_classes = 2, np.array([3, 2], dtype=np.intp)

    # 根据给定的分类标准创建决策树分割器
    criterion = CRITERIA_CLF["gini"](n_outputs, n_classes)
    # 使用参数创建分割器实例
    splitter = Splitter(criterion, max_features, 5, 0.5, rng, monotonic_cst=None)
    # 对分割器进行序列化
    splitter_serialize = pickle.dumps(splitter)

    # 反序列化分割器
    splitter_back = pickle.loads(splitter_serialize)
    # 断言反序列化后的分割器与原分割器的最大特征数相同
    assert splitter_back.max_features == max_features
    # 断言反序列化后的对象属于原分割器的类型
    assert isinstance(splitter_back, Splitter)


# 测试从只读缓冲区反序列化树模型的功能
def test_tree_deserialization_from_read_only_buffer(tmpdir):
    """Check that Trees can be deserialized with read only buffers.

    Non-regression test for gh-25584.
    """
    # 创建临时文件路径用于存储序列化后的模型
    pickle_path = str(tmpdir.join("clf.joblib"))
    # 创建决策树分类器实例
    clf = DecisionTreeClassifier(random_state=0)
    # 使用小型数据集进行训练
    clf.fit(X_small, y_small)

    # 将模型序列化并保存到文件
    joblib.dump(clf, pickle_path)
    # 从只读模式加载序列化的模型
    loaded_clf = joblib.load(pickle_path, mmap_mode="r")

    # 断言加载后的树结构与原始分类器的树结构相等
    assert_tree_equal(
        loaded_clf.tree_,
        clf.tree_,
        "The trees of the original and loaded classifiers are not equal.",
    )


# 使用参数化装饰器对 test_min_sample_split_1_error 函数进行测试，覆盖所有树模型
@pytest.mark.parametrize("Tree", ALL_TREES.values())
def test_min_sample_split_1_error(Tree):
    """Check that an error is raised when min_sample_split=1.

    non-regression test for issue gh-25481.
    """
    # 创建示例数据
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0, 1])

    # 当 min_samples_split=1.0 时，应当正常运行
    Tree(min_samples_split=1.0).fit(X, y)

    # 当 min_samples_split=1 时，应当抛出值错误异常
    tree = Tree(min_samples_split=1)
    msg = (
        r"'min_samples_split' .* must be an int in the range \[2, inf\) "
        r"or a float in the range \(0.0, 1.0\]"
    )
    with pytest.raises(ValueError, match=msg):
        tree.fit(X, y)


# 使用参数化装饰器对 test_missing_values_on_equal_nodes_no_missing 函数进行测试，覆盖所有不同的判断标准
@pytest.mark.parametrize("criterion", ["squared_error", "friedman_mse"])
def test_missing_values_on_equal_nodes_no_missing(criterion):
    """Check missing values goes to correct node during predictions"""
    # 创建示例数据集
    X = np.array([[0, 1, 2, 3, 8, 9, 11, 12, 15]]).T
    y = np.array([0.1, 0.2, 0.3, 0.2, 1.4, 1.4, 1.5, 1.6, 2.6])

    # 创建决策树回归器实例
    dtc = DecisionTreeRegressor(random_state=42, max_depth=1, criterion=criterion)
    dtc.fit(X, y)

    # 预测缺失值，应当进入正确的节点
    y_pred = dtc.predict([[np.nan]])
    assert_allclose(y_pred, [np.mean(y[-5:])])

    # 创建另一个数据集，确保节点内元素数相等
    X_equal = X[:-1]
    y_equal = y[:-1]

    # 创建另一个决策树回归器实例
    dtc = DecisionTreeRegressor(random_state=42, max_depth=1, criterion=criterion)
    dtc.fit(X_equal, y_equal)

    # 预测缺失值，应当进入正确的节点，因为算法会选择最多数据点的节点
    y_pred = dtc.predict([[np.nan]])
    assert_allclose(y_pred, [np.mean(y_equal[-4:])])


# 使用参数化装饰器对 test_missing_values_best_splitter_three_classes 函数进行测试，覆盖所有不同的判断标准
@pytest.mark.parametrize("criterion", ["entropy", "gini"])
def test_missing_values_best_splitter_three_classes(criterion):
    """Test when missing values are uniquely present in a class among 3 classes."""
    # 定义缺失值所属的类别为第一个类（类别标识为0）
    missing_values_class = 0
    # 创建特征矩阵 X，其中包含一个特征，多行数据，部分值为 NaN，其余为整数
    X = np.array([[np.nan] * 4 + [0, 1, 2, 3, 8, 9, 11, 12]]).T
    # 创建目标向量 y，其中第一部分数据属于缺失值类别，其余数据分别属于第二和第三类
    y = np.array([missing_values_class] * 4 + [1] * 4 + [2] * 4)
    # 初始化决策树分类器对象，设定随机种子、最大深度和划分标准
    dtc = DecisionTreeClassifier(random_state=42, max_depth=2, criterion=criterion)
    # 使用 X 和 y 训练决策树分类器
    dtc.fit(X, y)

    # 创建测试数据集 X_test，包含一个特征，其中一个值为 NaN
    X_test = np.array([[np.nan, 3, 12]]).T
    # 对测试数据集进行预测
    y_nan_pred = dtc.predict(X_test)
    # 断言预测结果与预期结果相等，验证缺失值是否正确地关联到对应的类别
    # 预期结果是 [missing_values_class, 1, 2]
    assert_array_equal(y_nan_pred, [missing_values_class, 1, 2])
@pytest.mark.parametrize("criterion", ["entropy", "gini"])
def test_missing_values_best_splitter_to_left(criterion):
    """Missing values spanning only one class at fit-time must make missing
    values at predict-time be classified has belonging to this class."""
    # 创建一个包含一个特征的数组，有4个NaN值和8个具体值的样本
    X = np.array([[np.nan] * 4 + [0, 1, 2, 3, 4, 5]]).T
    # 创建一个标签数组，前4个为0，后6个为1
    y = np.array([0] * 4 + [1] * 6)

    # 创建一个决策树分类器对象，设定随机种子为42，最大深度为2，判定标准为entropy或gini
    dtc = DecisionTreeClassifier(random_state=42, max_depth=2, criterion=criterion)
    # 使用样本和标签进行训练
    dtc.fit(X, y)

    # 创建一个测试样本，包含一个NaN值和两个具体值
    X_test = np.array([[np.nan, 5, np.nan]]).T
    # 对测试样本进行预测
    y_pred = dtc.predict(X_test)

    # 断言预测结果与预期结果相等
    assert_array_equal(y_pred, [0, 1, 0])


@pytest.mark.parametrize("criterion", ["entropy", "gini"])
def test_missing_values_best_splitter_to_right(criterion):
    """Missing values and non-missing values sharing one class at fit-time
    must make missing values at predict-time be classified has belonging
    to this class."""
    # 创建一个包含一个特征的数组，有4个NaN值和8个具体值的样本
    X = np.array([[np.nan] * 4 + [0, 1, 2, 3, 4, 5]]).T
    # 创建一个标签数组，前4个为1，接着的4个为0，最后2个为1
    y = np.array([1] * 4 + [0] * 4 + [1] * 2)

    # 创建一个决策树分类器对象，设定随机种子为42，最大深度为2，判定标准为entropy或gini
    dtc = DecisionTreeClassifier(random_state=42, max_depth=2, criterion=criterion)
    # 使用样本和标签进行训练
    dtc.fit(X, y)

    # 创建一个测试样本，包含一个NaN值和两个具体值
    X_test = np.array([[np.nan, 1.2, 4.8]]).T
    # 对测试样本进行预测
    y_pred = dtc.predict(X_test)

    # 断言预测结果与预期结果相等
    assert_array_equal(y_pred, [1, 0, 1])


@pytest.mark.parametrize("criterion", ["entropy", "gini"])
def test_missing_values_missing_both_classes_has_nan(criterion):
    """Check behavior of missing value when there is one missing value in each class."""
    # 创建一个包含一个特征的数组，有5个具体值和5个NaN值的样本
    X = np.array([[1, 2, 3, 5, np.nan, 10, 20, 30, 60, np.nan]]).T
    # 创建一个标签数组，前5个为0，后5个为1
    y = np.array([0] * 5 + [1] * 5)

    # 创建一个决策树分类器对象，设定随机种子为42，最大深度为1，判定标准为entropy或gini
    dtc = DecisionTreeClassifier(random_state=42, max_depth=1, criterion=criterion)
    # 使用样本和标签进行训练
    dtc.fit(X, y)
    # 创建一个测试样本，包含一个NaN值和两个具体值
    X_test = np.array([[np.nan, 2.3, 34.2]]).T
    # 对测试样本进行预测
    y_pred = dtc.predict(X_test)

    # 断言预测结果与预期结果相等
    assert_array_equal(y_pred, [1, 0, 1])


@pytest.mark.parametrize("sparse_container", [None] + CSR_CONTAINERS)
@pytest.mark.parametrize(
    "tree",
    [
        DecisionTreeClassifier(splitter="random"),
        DecisionTreeRegressor(criterion="absolute_error"),
    ],
)
def test_missing_value_errors(sparse_container, tree):
    """Check unsupported configurations for missing values."""
    # 创建一个包含一个特征的数组，有5个具体值和5个NaN值的样本
    X = np.array([[1, 2, 3, 5, np.nan, 10, 20, 30, 60, np.nan]]).T
    # 创建一个标签数组，前5个为0，后5个为1
    y = np.array([0] * 5 + [1] * 5)

    if sparse_container is not None:
        X = sparse_container(X)

    # 断言对包含NaN值的输入X会引发ValueError异常，异常消息包含"Input X contains NaN"
    with pytest.raises(ValueError, match="Input X contains NaN"):
        tree.fit(X, y)


def test_missing_values_poisson():
    """Smoke test for poisson regression and missing values."""
    # 复制糖尿病数据集的特征和标签
    X, y = diabetes.data.copy(), diabetes.target

    # 将某些值设置为缺失值
    X[::5, 0] = np.nan
    X[::6, -1] = np.nan

    # 创建一个泊松回归的决策树回归器对象，设定随机种子为42
    reg = DecisionTreeRegressor(criterion="poisson", random_state=42)
    # 使用样本和标签进行训练
    reg.fit(X, y)

    # 对训练集进行预测
    y_pred = reg.predict(X)
    # 断言预测结果均大于等于0
    assert (y_pred >= 0.0).all()


def make_friedman1_classification(*args, **kwargs):
    # 使用make_friedman1函数生成分类问题的特征和标签数据集
    X, y = datasets.make_friedman1(*args, **kwargs)
    # 将回归目标转换为分类问题，基于阈值14
    y = y > 14
    return X, y
# 使用 pytest 的 mark.parametrize 装饰器定义参数化测试，测试函数接受两个参数：make_data 和 Tree
@pytest.mark.parametrize(
    "make_data,Tree",
    [
        # 第一个参数组合：使用 datasets.make_friedman1 函数生成数据，Tree 为 DecisionTreeRegressor
        (datasets.make_friedman1, DecisionTreeRegressor),
        # 第二个参数组合：使用 make_friedman1_classification 函数生成分类数据，Tree 为 DecisionTreeClassifier
        (make_friedman1_classification, DecisionTreeClassifier),
    ],
)
# 参数化测试，对 sample_weight_train 参数进行测试，可能取值为 None 或 "ones"
@pytest.mark.parametrize("sample_weight_train", [None, "ones"])
# 定义测试函数 test_missing_values_is_resilience，接受参数 make_data, Tree, sample_weight_train, global_random_seed
def test_missing_values_is_resilience(
    make_data, Tree, sample_weight_train, global_random_seed
):
    """Check that trees can deal with missing values have decent performance."""
    # 设置数据集大小和特征数量
    n_samples, n_features = 5_000, 10
    # 使用 make_data 生成数据集 X, y
    X, y = make_data(
        n_samples=n_samples, n_features=n_features, random_state=global_random_seed
    )

    # 复制 X 到 X_missing
    X_missing = X.copy()
    # 使用全局随机种子初始化随机数生成器 rng
    rng = np.random.RandomState(global_random_seed)
    # 随机将 X_missing 的部分值设为 NaN
    X_missing[rng.choice([False, True], size=X.shape, p=[0.9, 0.1])] = np.nan
    # 划分训练集和测试集
    X_missing_train, X_missing_test, y_train, y_test = train_test_split(
        X_missing, y, random_state=global_random_seed
    )
    # 根据 sample_weight_train 的取值设置样本权重
    if sample_weight_train == "ones":
        sample_weight = np.ones(X_missing_train.shape[0])
    else:
        sample_weight = None

    # 创建一个原始的决策树模型
    native_tree = Tree(max_depth=10, random_state=global_random_seed)
    # 使用训练集拟合原始决策树模型
    native_tree.fit(X_missing_train, y_train, sample_weight=sample_weight)
    # 计算原始决策树模型在测试集上的得分
    score_native_tree = native_tree.score(X_missing_test, y_test)

    # 创建一个包含缺失值填充器的管道，和决策树模型
    tree_with_imputer = make_pipeline(
        SimpleImputer(), Tree(max_depth=10, random_state=global_random_seed)
    )
    # 使用训练集拟合带有缺失值填充器的管道
    tree_with_imputer.fit(X_missing_train, y_train)
    # 计算带有缺失值填充器的管道在测试集上的得分
    score_tree_with_imputer = tree_with_imputer.score(X_missing_test, y_test)

    # 断言：原始决策树模型的得分应严格大于带有缺失值填充器的管道模型得分
    assert (
        score_native_tree > score_tree_with_imputer
    ), f"{score_native_tree=} should be strictly greater than {score_tree_with_imputer}"


# 定义单独的测试函数 test_missing_value_is_predictive，检查树模型是否能够学习到缺失值对预测的影响
def test_missing_value_is_predictive():
    """Check the tree learns when only the missing value is predictive."""
    # 使用全局随机种子初始化随机数生成器 rng
    rng = np.random.RandomState(0)
    # 设置样本数量
    n_samples = 1000

    # 生成标准正态分布的数据集 X
    X = rng.standard_normal(size=(n_samples, 10))
    # 生成随机的分类标签 y
    y = rng.randint(0, high=2, size=n_samples)

    # 创建一个预测特征，使用 y 和一些噪声生成
    X_random_mask = rng.choice([False, True], size=n_samples, p=[0.95, 0.05])
    y_mask = y.copy().astype(bool)
    y_mask[X_random_mask] = ~y_mask[X_random_mask]

    # 生成一个预测性特征 X_predictive，其中部分值设为 NaN
    X_predictive = rng.standard_normal(size=n_samples)
    X_predictive[y_mask] = np.nan

    # 将 X_predictive 中的数据作为 X 的第 5 列
    X[:, 5] = X_predictive

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)
    # 使用决策树分类器拟合数据
    tree = DecisionTreeClassifier(random_state=rng).fit(X_train, y_train)

    # 断言：训练集上的得分和测试集上的得分都应不低于 0.85
    assert tree.score(X_train, y_train) >= 0.85
    assert tree.score(X_test, y_test) >= 0.85


# 参数化测试，测试函数接受两个参数：make_data 和 Tree
@pytest.mark.parametrize(
    "make_data, Tree",
    [
        # 第一个参数组合：使用 datasets.make_regression 函数生成数据，Tree 为 DecisionTreeRegressor
        (datasets.make_regression, DecisionTreeRegressor),
        # 第二个参数组合：使用 datasets.make_classification 函数生成分类数据，Tree 为 DecisionTreeClassifier
        (datasets.make_classification, DecisionTreeClassifier),
    ],
)
# 定义测试函数 test_sample_weight_non_uniform，检查处理带有缺失值的样本权重是否正确
def test_sample_weight_non_uniform(make_data, Tree):
    """Check sample weight is correctly handled with missing values."""
    # 使用全局随机种子初始化随机数生成器 rng
    rng = np.random.RandomState(0)
    # 设置样本数量和特征数量
    n_samples, n_features = 1000, 10
    # 使用 make_data 生成数据集 X, y
    X, y = make_data(n_samples=n_samples, n_features=n_features, random_state=rng)
    # 创建带有缺失值的数据集
    X[rng.choice([False, True], size=X.shape, p=[0.9, 0.1])] = np.nan
    
    # 设置样本权重，将每隔一个样本的权重设为0，相当于移除这些样本
    sample_weight = np.ones(X.shape[0])
    sample_weight[::2] = 0.0
    
    # 使用带有样本权重的决策树模型，使用固定的随机状态
    tree_with_sw = Tree(random_state=0)
    tree_with_sw.fit(X, y, sample_weight=sample_weight)
    
    # 不使用被移除样本的决策树模型，使用固定的随机状态
    tree_samples_removed = Tree(random_state=0)
    tree_samples_removed.fit(X[1::2, :], y[1::2])
    
    # 断言两个决策树模型在所有数据上的预测结果应该非常接近
    assert_allclose(tree_samples_removed.predict(X), tree_with_sw.predict(X))
# 定义一个测试函数，用于检验决策树在序列化时的确定性。
def test_deterministic_pickle():
    # 非回归测试，关注的问题链接为：
    # https://github.com/scikit-learn/scikit-learn/issues/27268
    # 未初始化的内存可能导致两个 pickle 字符串不同。
    tree1 = DecisionTreeClassifier(random_state=0).fit(iris.data, iris.target)
    tree2 = DecisionTreeClassifier(random_state=0).fit(iris.data, iris.target)

    pickle1 = pickle.dumps(tree1)  # 序列化第一个决策树对象
    pickle2 = pickle.dumps(tree2)  # 序列化第二个决策树对象

    assert pickle1 == pickle2  # 断言两个序列化结果相同


@pytest.mark.parametrize(
    "X",
    [
        # 缺失值将导致贪婪分割时向左移动
        np.array([np.nan, 2, np.nan, 4, 5, 6]),
        np.array([np.nan, np.nan, 3, 4, 5, 6]),
        # 缺失值将导致贪婪分割时向右移动
        np.array([1, 2, 3, 4, np.nan, np.nan]),
        np.array([1, 2, 3, np.nan, 6, np.nan]),
    ],
)
@pytest.mark.parametrize("criterion", ["squared_error", "friedman_mse"])
def test_regression_tree_missing_values_toy(X, criterion):
    """检查回归树在使用玩具数据集中如何处理缺失值。

    此测试的目标是检测当叶子节点只有一个样本时，是否正确处理缺失值导致的问题。
    非回归测试，关注的问题链接为：
    https://github.com/scikit-learn/scikit-learn/issues/28254
    https://github.com/scikit-learn/scikit-learn/issues/28316
    """
    X = X.reshape(-1, 1)
    y = np.arange(6)

    tree = DecisionTreeRegressor(criterion=criterion, random_state=0).fit(X, y)
    tree_ref = clone(tree).fit(y.reshape(-1, 1), y)
    assert all(tree.tree_.impurity >= 0)  # MSE 应始终为正数
    # 检查第一次分割后的不纯度匹配
    assert_allclose(tree.tree_.impurity[:2], tree_ref.tree_.impurity[:2])

    # 找到只有一个样本的叶子节点，其 MSE 应为 0
    leaves_idx = np.flatnonzero(
        (tree.tree_.children_left == -1) & (tree.tree_.n_node_samples == 1)
    )
    assert_allclose(tree.tree_.impurity[leaves_idx], 0.0)


def test_classification_tree_missing_values_toy():
    """检查分类树在使用玩具数据集中如何处理缺失值。

    此测试更为复杂，因为我们使用了检测到的随机森林回归问题的情况。我们因此定义了种子和引导索引，
    以检测其中的一种不经常发生的回归。

    在此处，我们检查叶子节点中不纯度是否为零或正数。

    非回归测试，关注的问题链接为：
    https://github.com/scikit-learn/scikit-learn/issues/28254
    """
    X, y = datasets.load_iris(return_X_y=True)

    rng = np.random.RandomState(42)
    X_missing = X.copy()
    mask = rng.binomial(
        n=np.ones(shape=(1, 4), dtype=np.int32), p=X[:, [2]] / 8
    ).astype(bool)
    X_missing[mask] = np.nan
    # 使用 train_test_split 函数分割数据集 X_missing 和 y，仅保留训练集部分 X_train 和 y_train
    X_train, _, y_train, _ = train_test_split(X_missing, y, random_state=13)

    # fmt: off
    # 禁止 Black 重新格式化此特定数组
    # 定义一个包含特定索引的 NumPy 数组
    indices = np.array([
        2, 81, 39, 97, 91, 38, 46, 31, 101, 13, 89, 82, 100, 42, 69, 27, 81, 16, 73, 74,
        51, 47, 107, 17, 75, 110, 20, 15, 104, 57, 26, 15, 75, 79, 35, 77, 90, 51, 46,
        13, 94, 91, 23, 8, 93, 93, 73, 77, 12, 13, 74, 109, 110, 24, 10, 23, 104, 27,
        92, 52, 20, 109, 8, 8, 28, 27, 35, 12, 12, 7, 43, 0, 30, 31, 78, 12, 24, 105,
        50, 0, 73, 12, 102, 105, 13, 31, 1, 69, 11, 32, 75, 90, 106, 94, 60, 56, 35, 17,
        62, 85, 81, 39, 80, 16, 63, 6, 80, 84, 3, 3, 76, 78
    ], dtype=np.int32)
    # fmt: on

    # 创建一个决策树分类器对象，设定最大深度为 3，特征数为平方根，随机数种子为 1857819720
    tree = DecisionTreeClassifier(
        max_depth=3, max_features="sqrt", random_state=1857819720
    )
    # 使用决策树分类器拟合训练集中特定索引的数据
    tree.fit(X_train[indices], y_train[indices])

    # 检查所有叶子节点的不纯度是否大于等于 0
    assert all(tree.tree_.impurity >= 0)

    # 找到所有叶子节点的索引，这些节点的左子节点为 -1 且节点样本数为 1
    leaves_idx = np.flatnonzero(
        (tree.tree_.children_left == -1) & (tree.tree_.n_node_samples == 1)
    )
    # 断言所有找到的叶子节点的不纯度为 0
    assert_allclose(tree.tree_.impurity[leaves_idx], 0.0)
```