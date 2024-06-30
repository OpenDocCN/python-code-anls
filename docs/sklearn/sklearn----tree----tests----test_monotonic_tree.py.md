# `D:\src\scipysrc\scikit-learn\sklearn\tree\tests\test_monotonic_tree.py`

```
# 导入必要的库
import numpy as np  # 导入NumPy库，用于科学计算
import pytest  # 导入pytest库，用于编写和运行测试用例

# 导入数据集生成和机器学习模型相关的类和函数
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)

# 导入用于测试的辅助函数和类
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import CSC_CONTAINERS

# 定义用于决策树分类器和回归器的类列表
TREE_CLASSIFIER_CLASSES = [DecisionTreeClassifier, ExtraTreeClassifier]
TREE_REGRESSOR_CLASSES = [DecisionTreeRegressor, ExtraTreeRegressor]

# 将决策树和基于树的分类器类别合并为一个列表
TREE_BASED_CLASSIFIER_CLASSES = TREE_CLASSIFIER_CLASSES + [
    RandomForestClassifier,
    ExtraTreesClassifier,
]

# 将决策树和基于树的回归器类别合并为一个列表
TREE_BASED_REGRESSOR_CLASSES = TREE_REGRESSOR_CLASSES + [
    RandomForestRegressor,
    ExtraTreesRegressor,
]

# 参数化测试函数，测试单调约束在分类问题上的应用
@pytest.mark.parametrize("TreeClassifier", TREE_BASED_CLASSIFIER_CLASSES)
@pytest.mark.parametrize("depth_first_builder", (True, False))
@pytest.mark.parametrize("sparse_splitter", (True, False))
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_monotonic_constraints_classifications(
    TreeClassifier,
    depth_first_builder,
    sparse_splitter,
    global_random_seed,
    csc_container,
):
    # 设定样本数量
    n_samples = 1000
    n_samples_train = 900

    # 生成分类数据集
    X, y = make_classification(
        n_samples=n_samples,
        n_classes=2,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        random_state=global_random_seed,
    )

    # 分割训练集和测试集
    X_train, y_train = X[:n_samples_train], y[:n_samples_train]
    X_test, _ = X[n_samples_train:], y[n_samples_train:]

    # 复制测试集用于不同的增减约束
    X_test_0incr, X_test_0decr = np.copy(X_test), np.copy(X_test)
    X_test_1incr, X_test_1decr = np.copy(X_test), np.copy(X_test)
    X_test_0incr[:, 0] += 10
    X_test_0decr[:, 0] -= 10
    X_test_1incr[:, 1] += 10
    X_test_1decr[:, 1] -= 10

    # 创建单调约束数组
    monotonic_cst = np.zeros(X.shape[1])
    monotonic_cst[0] = 1
    monotonic_cst[1] = -1

    # 根据是否使用深度优先构建器选择分类器
    if depth_first_builder:
        est = TreeClassifier(max_depth=None, monotonic_cst=monotonic_cst)
    else:
        est = TreeClassifier(
            max_depth=None,
            monotonic_cst=monotonic_cst,
            max_leaf_nodes=n_samples_train,
        )

    # 如果分类器具有随机状态参数，则设定全局随机种子
    if hasattr(est, "random_state"):
        est.set_params(**{"random_state": global_random_seed})

    # 如果分类器具有估计器数量参数，则设置为5
    if hasattr(est, "n_estimators"):
        est.set_params(**{"n_estimators": 5})

    # 如果使用稀疏分割器，将训练集转换为指定的稀疏容器类型
    if sparse_splitter:
        X_train = csc_container(X_train)

    # 拟合分类器
    est.fit(X_train, y_train)

    # 预测测试集样本的概率
    proba_test = est.predict_proba(X_test)

    # 断言预测概率在合理的范围内
    assert np.logical_and(
        proba_test >= 0.0, proba_test <= 1.0
    ).all(), "Probability should always be in [0, 1] range."

    # 断言预测概率和为1
    assert_allclose(proba_test.sum(axis=1), 1.0)

    # 断言单调递增约束应用于正类别
    assert np.all(est.predict_proba(X_test_0incr)[:, 1] >= proba_test[:, 1])

    # 断言单调递减约束应用于正类别
    assert np.all(est.predict_proba(X_test_0decr)[:, 1] <= proba_test[:, 1])
    # 断言：单调递减约束条件，适用于正类别
    # 检查预测的正类别概率是否在增加数据集（X_test_1incr）上单调递减
    assert np.all(est.predict_proba(X_test_1incr)[:, 1] <= proba_test[:, 1])
    # 检查预测的正类别概率是否在减少数据集（X_test_1decr）上单调递增
    assert np.all(est.predict_proba(X_test_1decr)[:, 1] >= proba_test[:, 1])
# 使用 pytest 的 parametrize 装饰器来为单元测试参数化，每个参数化都会生成一个测试用例

@pytest.mark.parametrize("TreeRegressor", TREE_BASED_REGRESSOR_CLASSES)
# 参数化 TreeRegressor，依次使用 TREE_BASED_REGRESSOR_CLASSES 中的每个回归树类进行测试

@pytest.mark.parametrize("depth_first_builder", (True, False))
# 参数化 depth_first_builder，测试两种不同的建树方式：深度优先和广度优先

@pytest.mark.parametrize("sparse_splitter", (True, False))
# 参数化 sparse_splitter，测试两种不同的稀疏分割器选项

@pytest.mark.parametrize("criterion", ("absolute_error", "squared_error"))
# 参数化 criterion，测试两种不同的评价标准：绝对误差和平方误差

@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
# 参数化 csc_container，使用 CSC_CONTAINERS 中的不同容器类型对稀疏数据进行测试

def test_monotonic_constraints_regressions(
    TreeRegressor,
    depth_first_builder,
    sparse_splitter,
    criterion,
    global_random_seed,
    csc_container,
):
    # 设置样本数量和训练样本数量
    n_samples = 1000
    n_samples_train = 900

    # 创建包含 5 个信息特征的回归任务数据集
    X, y = make_regression(
        n_samples=n_samples,
        n_features=5,
        n_informative=5,
        random_state=global_random_seed,
    )

    # 创建训练索引和测试索引
    train = np.arange(n_samples_train)
    test = np.arange(n_samples_train, n_samples)

    # 切分训练集和测试集
    X_train = X[train]
    y_train = y[train]
    X_test = np.copy(X[test])
    X_test_incr = np.copy(X_test)
    X_test_decr = np.copy(X_test)

    # 增加和减少测试集的第一个特征值
    X_test_incr[:, 0] += 10
    X_test_decr[:, 1] += 10

    # 创建一个零数组，指示特征的单调性约束
    monotonic_cst = np.zeros(X.shape[1])
    monotonic_cst[0] = 1  # 第一个特征单调递增
    monotonic_cst[1] = -1  # 第二个特征单调递减

    # 根据深度优先建树方式选择回归器
    if depth_first_builder:
        est = TreeRegressor(
            max_depth=None,
            monotonic_cst=monotonic_cst,
            criterion=criterion,
        )
    else:
        est = TreeRegressor(
            max_depth=8,
            monotonic_cst=monotonic_cst,
            criterion=criterion,
            max_leaf_nodes=n_samples_train,
        )

    # 如果估计器有 random_state 属性，设置随机种子
    if hasattr(est, "random_state"):
        est.set_params(random_state=global_random_seed)

    # 如果估计器有 n_estimators 属性，设置估计器数量
    if hasattr(est, "n_estimators"):
        est.set_params(**{"n_estimators": 5})

    # 如果选择了稀疏分割器，使用指定的容器类型对训练数据进行转换
    if sparse_splitter:
        X_train = csc_container(X_train)

    # 拟合回归器
    est.fit(X_train, y_train)

    # 对测试集进行预测
    y = est.predict(X_test)

    # 单调递增约束
    y_incr = est.predict(X_test_incr)
    # y_incr 应该始终大于等于 y
    assert np.all(y_incr >= y)

    # 单调递减约束
    y_decr = est.predict(X_test_decr)
    # y_decr 应该始终小于等于 y
    assert np.all(y_decr <= y)


@pytest.mark.parametrize("TreeClassifier", TREE_BASED_CLASSIFIER_CLASSES)
# 参数化 TreeClassifier，依次使用 TREE_BASED_CLASSIFIER_CLASSES 中的每个分类树类进行测试

def test_multiclass_raises(TreeClassifier):
    # 创建一个多分类任务数据集
    X, y = make_classification(
        n_samples=100, n_features=5, n_classes=3, n_informative=3, random_state=0
    )

    # 将第一个样本标签设置为 0
    y[0] = 0

    # 创建一个零数组，指示特征的单调性约束
    monotonic_cst = np.zeros(X.shape[1])
    monotonic_cst[0] = -1  # 第一个特征单调递减
    monotonic_cst[1] = 1   # 第二个特征单调递增

    # 创建一个分类树分类器，尝试应用单调性约束
    est = TreeClassifier(max_depth=None, monotonic_cst=monotonic_cst, random_state=0)

    # 期望抛出 ValueError，并匹配指定消息
    msg = "Monotonicity constraints are not supported with multiclass classification"
    with pytest.raises(ValueError, match=msg):
        est.fit(X, y)


@pytest.mark.parametrize("TreeClassifier", TREE_BASED_CLASSIFIER_CLASSES)
# 参数化 TreeClassifier，依次使用 TREE_BASED_CLASSIFIER_CLASSES 中的每个分类树类进行测试

def test_multiple_output_raises(TreeClassifier):
    # 创建一个多输出回归任务数据集
    X = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    y = [[1, 0, 1, 0, 1], [1, 0, 1, 0, 1]]

    # 创建一个分类树分类器，尝试应用单调性约束
    est = TreeClassifier(
        max_depth=None, monotonic_cst=np.array([-1, 1]), random_state=0
    )
    # 定义错误信息，指出多输出不支持单调性约束
    msg = "Monotonicity constraints are not supported with multiple output"
    
    # 使用 pytest 库的 raises 函数验证是否引发 ValueError 异常，并检查异常消息是否匹配预期
    with pytest.raises(ValueError, match=msg):
        # 调用估算器（estimator）的 fit 方法，传入特征 X 和目标变量 y 进行拟合
        est.fit(X, y)
# 使用 pytest 的 mark.parametrize 装饰器，为测试函数参数化
@pytest.mark.parametrize(
    "DecisionTreeEstimator", [DecisionTreeClassifier, DecisionTreeRegressor]
)
# 定义测试函数，测试当输入数据中包含 NaN 时是否会引发 ValueError 异常
def test_missing_values_raises(DecisionTreeEstimator):
    # 创建一个具有 NaN 值的样本数据集 X 和对应的标签 y
    X, y = make_classification(
        n_samples=100, n_features=5, n_classes=2, n_informative=3, random_state=0
    )
    X[0, 0] = np.nan  # 在第一个样本的第一个特征位置设置 NaN 值
    monotonic_cst = np.zeros(X.shape[1])  # 创建一个全零数组，用于定义单调性约束
    monotonic_cst[0] = 1  # 将第一个特征定义为单调递增
    # 创建决策树估计器对象，设置最大深度为 None，单调性约束为 monotonic_cst，随机种子为 0
    est = DecisionTreeEstimator(
        max_depth=None, monotonic_cst=monotonic_cst, random_state=0
    )

    msg = "Input X contains NaN"  # 定义异常消息字符串
    # 使用 pytest 的 raises 断言检查是否抛出 ValueError 异常，并验证异常消息
    with pytest.raises(ValueError, match=msg):
        est.fit(X, y)


# 使用 pytest 的 mark.parametrize 装饰器，为测试函数参数化
@pytest.mark.parametrize("TreeClassifier", TREE_BASED_CLASSIFIER_CLASSES)
# 定义测试函数，测试当单调性约束不匹配输入数据维度时是否会引发 ValueError 异常
def test_bad_monotonic_cst_raises(TreeClassifier):
    # 创建一个简单的样本数据集 X 和对应的标签 y
    X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    y = [1, 0, 1, 0, 1]

    msg = "monotonic_cst has shape 3 but the input data X has 2 features."
    # 创建树类分类器对象，设置最大深度为 None，单调性约束为 np.array([-1, 1, 0])，随机种子为 0
    est = TreeClassifier(
        max_depth=None, monotonic_cst=np.array([-1, 1, 0]), random_state=0
    )
    # 使用 pytest 的 raises 断言检查是否抛出 ValueError 异常，并验证异常消息
    with pytest.raises(ValueError, match=msg):
        est.fit(X, y)

    msg = "monotonic_cst must be None or an array-like of -1, 0 or 1."
    # 创建树类分类器对象，设置最大深度为 None，单调性约束为 np.array([-2, 2])，随机种子为 0
    est = TreeClassifier(
        max_depth=None, monotonic_cst=np.array([-2, 2]), random_state=0
    )
    # 使用 pytest 的 raises 断言检查是否抛出 ValueError 异常，并验证异常消息
    with pytest.raises(ValueError, match=msg):
        est.fit(X, y)

    # 创建树类分类器对象，设置最大深度为 None，单调性约束为 np.array([-1, 0.8])，随机种子为 0
    est = TreeClassifier(
        max_depth=None, monotonic_cst=np.array([-1, 0.8]), random_state=0
    )
    # 使用 pytest 的 raises 断言检查是否抛出 ValueError 异常，并验证异常消息的部分匹配
    with pytest.raises(ValueError, match=msg + "(.*)0.8]"):
        est.fit(X, y)


# 定义辅助函数，用于验证一维回归树的子节点的单调性和边界
def assert_1d_reg_tree_children_monotonic_bounded(tree_, monotonic_sign):
    values = tree_.value  # 获取树节点的值
    # 遍历树的每个节点
    for i in range(tree_.node_count):
        # 检查是否有左右子节点，并且当前节点不是叶子节点
        if tree_.children_left[i] > i and tree_.children_right[i] > i:
            # 检查子节点的单调性
            i_left = tree_.children_left[i]
            i_right = tree_.children_right[i]
            if monotonic_sign == 1:
                assert values[i_left] <= values[i_right]
            elif monotonic_sign == -1:
                assert values[i_left] >= values[i_right]
            val_middle = (values[i_left] + values[i_right]) / 2
            # 检查孙子节点的边界，过滤掉叶子节点
            if tree_.feature[i_left] >= 0:
                i_left_right = tree_.children_right[i_left]
                if monotonic_sign == 1:
                    assert values[i_left_right] <= val_middle
                elif monotonic_sign == -1:
                    assert values[i_left_right] >= val_middle
            if tree_.feature[i_right] >= 0:
                i_right_left = tree_.children_left[i_right]
                if monotonic_sign == 1:
                    assert val_middle <= values[i_right_left]
                elif monotonic_sign == -1:
                    assert val_middle >= values[i_right_left]


# 定义测试函数，测试一维回归树的子节点的单调性和边界
def test_assert_1d_reg_tree_children_monotonic_bounded():
    # 创建一维输入特征 X 和对应的目标值 y
    X = np.linspace(-1, 1, 7).reshape(-1, 1)
    y = np.sin(2 * np.pi * X.ravel())

    # 创建决策树回归器对象，设置最大深度为 None，随机种子为 0，拟合输入数据 X 和目标值 y
    reg = DecisionTreeRegressor(max_depth=None, random_state=0).fit(X, y)
    # 使用 pytest 模块中的 pytest.raises() 上下文管理器，检测是否抛出 AssertionError 异常
    with pytest.raises(AssertionError):
        # 调用 assert_1d_reg_tree_children_monotonic_bounded 函数验证决策树的子节点单调性和边界条件
        assert_1d_reg_tree_children_monotonic_bounded(reg.tree_, 1)
    
    # 使用 pytest 模块中的 pytest.raises() 上下文管理器，检测是否抛出 AssertionError 异常
    with pytest.raises(AssertionError):
        # 调用 assert_1d_reg_tree_children_monotonic_bounded 函数验证决策树的子节点单调性和边界条件
        assert_1d_reg_tree_children_monotonic_bounded(reg.tree_, -1)
# 定义一个函数，用于检查一维回归模型是否具有单调性。
def assert_1d_reg_monotonic(clf, monotonic_sign, min_x, max_x, n_steps):
    # 生成一个等间距的一维数组，作为输入数据的网格
    X_grid = np.linspace(min_x, max_x, n_steps).reshape(-1, 1)
    # 使用回归模型预测网格上的输出值
    y_pred_grid = clf.predict(X_grid)
    # 如果单调性要求为正向，则断言预测值的一阶差分非负
    if monotonic_sign == 1:
        assert (np.diff(y_pred_grid) >= 0.0).all()
    # 如果单调性要求为负向，则断言预测值的一阶差分非正
    elif monotonic_sign == -1:
        assert (np.diff(y_pred_grid) <= 0.0).all()


# 使用参数化测试装饰器，测试在特定条件下单调性相反的一维数据的情况
@pytest.mark.parametrize("TreeRegressor", TREE_REGRESSOR_CLASSES)
def test_1d_opposite_monotonicity_cst_data(TreeRegressor):
    # 检查带有负单调性约束的正单调数据是否产生常数预测，预测值等于目标值的平均值
    X = np.linspace(-2, 2, 10).reshape(-1, 1)
    y = X.ravel()
    # 使用回归树模型，应用负单调性约束
    clf = TreeRegressor(monotonic_cst=[-1])
    clf.fit(X, y)
    assert clf.tree_.node_count == 1
    assert clf.tree_.value[0] == 0.0

    # 交换单调性约束为正
    clf = TreeRegressor(monotonic_cst=[1])
    clf.fit(X, -y)
    assert clf.tree_.node_count == 1
    assert clf.tree_.value[0] == 0.0


# 使用参数化测试装饰器，测试一维树节点值的情况
@pytest.mark.parametrize("TreeRegressor", TREE_REGRESSOR_CLASSES)
@pytest.mark.parametrize("monotonic_sign", (-1, 1))
@pytest.mark.parametrize("depth_first_builder", (True, False))
@pytest.mark.parametrize("criterion", ("absolute_error", "squared_error"))
def test_1d_tree_nodes_values(
    TreeRegressor, monotonic_sign, depth_first_builder, criterion, global_random_seed
):
    # 从 sklearn.ensemble._hist_gradient_boosting 中的 test_nodes_values 改编而来
    # 构建只包含一个特征的单一树，并确保节点值遵守单调性约束

    # 构建一个随机种子
    rng = np.random.RandomState(global_random_seed)
    n_samples = 1000
    n_features = 1
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)

    if depth_first_builder:
        # 没有最大叶子节点数限制，采用默认的深度优先树构建器
        clf = TreeRegressor(
            monotonic_cst=[monotonic_sign],
            criterion=criterion,
            random_state=global_random_seed,
        )
    else:
        # 设置了最大叶子节点数，触发最佳优先树构建器
        clf = TreeRegressor(
            monotonic_cst=[monotonic_sign],
            max_leaf_nodes=n_samples,
            criterion=criterion,
            random_state=global_random_seed,
        )
    clf.fit(X, y)

    # 断言树的子节点单调有界
    assert_1d_reg_tree_children_monotonic_bounded(clf.tree_, monotonic_sign)
    # 断言一维回归模型具有单调性
    assert_1d_reg_monotonic(clf, monotonic_sign, np.min(X), np.max(X), 100)


# 定义一个函数，用于断言多维回归树的子节点具有单调性约束
def assert_nd_reg_tree_children_monotonic_bounded(tree_, monotonic_cst):
    # 创建一个节点数目大小的上界和下界数组
    upper_bound = np.full(tree_.node_count, np.inf)
    lower_bound = np.full(tree_.node_count, -np.inf)

# 定义一个测试函数，用于测试多维回归树子节点的单调性约束
def test_assert_nd_reg_tree_children_monotonic_bounded():
    # 检查 assert_nd_reg_tree_children_monotonic_bounded 是否能检测非单调的树预测。
    
    # 创建一个包含30个点的一维数组，范围从0到2π，作为输入特征 X
    X = np.linspace(0, 2 * np.pi, 30).reshape(-1, 1)
    # 计算正弦函数并将其展平，作为目标变量 y
    y = np.sin(X).ravel()
    # 使用深度不限的决策树回归器拟合数据
    reg = DecisionTreeRegressor(max_depth=None, random_state=0).fit(X, y)
    
    # 使用 pytest 来验证是否会抛出 AssertionError
    with pytest.raises(AssertionError):
        # 调用 assert_nd_reg_tree_children_monotonic_bounded 函数，期望会抛出 AssertionError
        assert_nd_reg_tree_children_monotonic_bounded(reg.tree_, [1])
    
    with pytest.raises(AssertionError):
        # 同上，这次传入一个负数，也期望会抛出 AssertionError
        assert_nd_reg_tree_children_monotonic_bounded(reg.tree_, [-1])
    
    # 第三次调用 assert_nd_reg_tree_children_monotonic_bounded，传入 0，期望不抛出异常
    assert_nd_reg_tree_children_monotonic_bounded(reg.tree_, [0])
    
    # 检查 assert_nd_reg_tree_children_monotonic_bounded 是否会在数据（以及因此模型）在相反方向上是单调的情况下抛出异常。
    
    # 创建一个包含5个点的一维数组，范围从-5到5，作为输入特征 X
    X = np.linspace(-5, 5, 5).reshape(-1, 1)
    # 计算 X 的立方并展平，作为目标变量 y
    y = X.ravel() ** 3
    # 使用深度不限的决策树回归器拟合数据
    reg = DecisionTreeRegressor(max_depth=None, random_state=0).fit(X, y)
    
    with pytest.raises(AssertionError):
        # 传入负数，期望抛出 AssertionError
        assert_nd_reg_tree_children_monotonic_bounded(reg.tree_, [-1])
    
    # 完整性起见，检查当改变符号时是否得到相反的结果。
    # 使用深度不限的决策树回归器拟合数据，这次传入 -y
    reg = DecisionTreeRegressor(max_depth=None, random_state=0).fit(X, -y)
    
    with pytest.raises(AssertionError):
        # 传入正数，期望抛出 AssertionError
        assert_nd_reg_tree_children_monotonic_bounded(reg.tree_, [1])
# 使用参数化测试框架pytest.mark.parametrize为测试函数test_nd_tree_nodes_values添加参数组合
@pytest.mark.parametrize("TreeRegressor", TREE_REGRESSOR_CLASSES)
@pytest.mark.parametrize("monotonic_sign", (-1, 1))
@pytest.mark.parametrize("depth_first_builder", (True, False))
@pytest.mark.parametrize("criterion", ("absolute_error", "squared_error"))
def test_nd_tree_nodes_values(
    TreeRegressor, monotonic_sign, depth_first_builder, criterion, global_random_seed
):
    # 构建带有多个特征的树，并确保节点的值遵守单调性约束。

    # 考虑具有在X[0]上单调增加约束的以下树，我们应该有：
    #
    #            root
    #           X[0]<=t
    #          /       \
    #         a         b
    #     X[0]<=u   X[1]<=v
    #    /       \   /     \
    #   c        d  e       f
    #
    # i)   a <= root <= b
    # ii)  c <= a <= d <= (a+b)/2
    # iii) (a+b)/2 <= min(e,f)
    # 对于 iii)，我们检查每个节点的值是否在适当的下限和上限之间。

    # 使用全局随机种子创建随机数生成器对象rng
    rng = np.random.RandomState(global_random_seed)
    n_samples = 1000  # 样本数
    n_features = 2  # 特征数
    monotonic_cst = [monotonic_sign, 0]  # 单调性约束列表
    X = rng.rand(n_samples, n_features)  # 生成随机特征数据
    y = rng.rand(n_samples)  # 生成随机目标数据

    if depth_first_builder:
        # 如果使用深度优先构建树，没有max_leaf_nodes限制，默认使用深度优先构建树
        clf = TreeRegressor(
            monotonic_cst=monotonic_cst,
            criterion=criterion,
            random_state=global_random_seed,
        )
    else:
        # 如果不使用深度优先构建树，使用max_leaf_nodes触发最佳优先构建树
        clf = TreeRegressor(
            monotonic_cst=monotonic_cst,
            max_leaf_nodes=n_samples,
            criterion=criterion,
            random_state=global_random_seed,
        )
    clf.fit(X, y)  # 使用数据X和y拟合分类器clf

    # 断言树的子节点是否满足单调性约束
    assert_nd_reg_tree_children_monotonic_bounded(clf.tree_, monotonic_cst)
```