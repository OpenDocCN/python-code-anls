# `D:\src\scipysrc\scikit-learn\sklearn\semi_supervised\tests\test_self_training.py`

```
# 从 math 模块导入 ceil 函数
from math import ceil

# 导入 numpy 库，并使用 np 别名
import numpy as np

# 导入 pytest 库
import pytest

# 从 numpy.testing 模块导入 assert_array_equal 函数
from numpy.testing import assert_array_equal

# 导入 sklearn 中的数据集加载函数和模型
from sklearn.datasets import load_iris, make_blobs
from sklearn.ensemble import StackingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 作者声明和许可证信息
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 加载鸢尾花数据集并随机排列样本顺序
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=0
)

# 定义有标签样本数量
n_labeled_samples = 50

# 复制训练集标签，将一部分标签设为缺失（-1）
y_train_missing_labels = y_train.copy()
y_train_missing_labels[n_labeled_samples:] = -1

# 定义标签映射关系
mapping = {0: "A", 1: "B", 2: "C", -1: "-1"}

# 使用向量化函数 np.vectorize 将标签映射为字符串类型，并转换为对象数组
y_train_missing_strings = np.vectorize(mapping.get)(y_train_missing_labels).astype(
    object
)

# 将缺失标签的字符串表示设为原始标签的整数值 -1
y_train_missing_strings[y_train_missing_labels == -1] = -1


def test_warns_k_best():
    # 测试使用 SelfTrainingClassifier 进行半监督学习，验证是否会发出警告
    st = SelfTrainingClassifier(KNeighborsClassifier(), criterion="k_best", k_best=1000)
    with pytest.warns(UserWarning, match="k_best is larger than"):
        st.fit(X_train, y_train_missing_labels)

    # 断言半监督学习的终止条件为 "all_labeled"
    assert st.termination_condition_ == "all_labeled"


@pytest.mark.parametrize(
    "base_estimator",
    [KNeighborsClassifier(), SVC(gamma="scale", probability=True, random_state=0)],
)
@pytest.mark.parametrize("selection_crit", ["threshold", "k_best"])
def test_classification(base_estimator, selection_crit):
    # 针对不同的参数设置进行分类测试
    # 断言字符串标签和数值标签的预测结果应相等
    # 测试多输出分类

    # 设置阈值和最大迭代次数
    threshold = 0.75
    max_iter = 10

    # 创建 SelfTrainingClassifier 对象进行半监督学习
    st = SelfTrainingClassifier(
        base_estimator, max_iter=max_iter, threshold=threshold, criterion=selection_crit
    )
    st.fit(X_train, y_train_missing_labels)

    # 对测试集进行预测和概率预测
    pred = st.predict(X_test)
    proba = st.predict_proba(X_test)

    # 使用字符串标签进行半监督学习
    st_string = SelfTrainingClassifier(
        base_estimator, max_iter=max_iter, criterion=selection_crit, threshold=threshold
    )
    st_string.fit(X_train, y_train_missing_strings)

    # 对字符串标签的测试集进行预测和概率预测
    pred_string = st_string.predict(X_test)
    proba_string = st_string.predict_proba(X_test)

    # 断言数值标签和字符串标签的预测结果应相等
    assert_array_equal(np.vectorize(mapping.get)(pred), pred_string)

    # 断言数值标签和字符串标签的概率预测结果应相等
    assert_array_equal(proba, proba_string)

    # 断言半监督学习的终止条件应相等
    assert st.termination_condition_ == st_string.termination_condition_

    # 检查有标签样本的迭代次数，n_iter 和 max_iter 的一致性
    labeled = y_train_missing_labels != -1

    # 断言有标签样本的 labeled_iter 应为 0
    assert_array_equal(st.labeled_iter_ == 0, labeled)

    # 断言训练期间有标签样本的标签不应发生变化
    assert_array_equal(y_train_missing_labels[labeled], st.transduction_[labeled])
    # 确保迭代的最大次数不超过总迭代次数的断言检查
    assert np.max(st.labeled_iter_) <= st.n_iter_ <= max_iter
    # 确保字符串标记迭代的最大次数不超过总迭代次数的断言检查
    assert np.max(st_string.labeled_iter_) <= st_string.n_iter_ <= max_iter
    
    # 检查数组形状是否相等的断言检查
    assert st.labeled_iter_.shape == st.transduction_.shape
    # 检查字符串标记数组形状是否相等的断言检查
    assert st_string.labeled_iter_.shape == st_string.transduction_.shape
def`
# 测试K最近邻算法

def test_k_best():
    # 创建一个自训练分类器对象，基础分类器为K最近邻，使用“k_best”标准，选取前10个特征
    st = SelfTrainingClassifier(
        KNeighborsClassifier(n_neighbors=1),
        criterion="k_best",
        k_best=10,
        max_iter=None,
    )
    # 复制训练标签y_train，仅保留第一个标签，其余标签设为-1
    y_train_only_one_label = np.copy(y_train)
    y_train_only_one_label[1:] = -1
    # 获取训练样本数量
    n_samples = y_train.shape[0]

    # 计算预期迭代次数，向上取整((样本数 - 1) / 10)
    n_expected_iter = ceil((n_samples - 1) / 10)
    # 对自训练分类器进行拟合
    st.fit(X_train, y_train_only_one_label)
    # 断言实际迭代次数与预期迭代次数相等
    assert st.n_iter_ == n_expected_iter

    # 检查labeled_iter_
    # 确认迭代0中标记为1的样本有1个
    assert np.sum(st.labeled_iter_ == 0) == 1
    # 对于后续每个迭代，确认标记为i的样本数为10
    for i in range(1, n_expected_iter):
        assert np.sum(st.labeled_iter_ == i) == 10
    # 最后一个迭代确认标记的样本数为(n_samples - 1) % 10
    assert np.sum(st.labeled_iter_ == n_expected_iter) == (n_samples - 1) % 10
    # 断言终止条件为所有样本均已标记
    assert st.termination_condition_ == "all_labeled"


# 测试分类的合理性

def test_sanity_classification():
    # 创建基础支持向量机分类器对象
    base_estimator = SVC(gamma="scale", probability=True)
    # 使用未标记样本拟合基础分类器
    base_estimator.fit(X_train[n_labeled_samples:], y_train[n_labeled_samples:])

    # 创建自训练分类器对象，基础分类器为支持向量机
    st = SelfTrainingClassifier(base_estimator)
    # 使用缺失标签的训练集拟合自训练分类器
    st.fit(X_train, y_train_missing_labels)

    # 获取基础分类器和自训练分类器在测试集上的预测结果
    pred1, pred2 = base_estimator.predict(X_test), st.predict(X_test)
    # 断言基础分类器和自训练分类器的预测结果不完全相同
    assert not np.array_equal(pred1, pred2)
    # 计算基础分类器和自训练分类器在测试集上的准确率
    score_supervised = accuracy_score(base_estimator.predict(X_test), y_test)
    score_self_training = accuracy_score(st.predict(X_test), y_test)

    # 断言自训练分类器的准确率高于基础分类器
    assert score_self_training > score_supervised


# 测试最大迭代次数为None的情况

def test_none_iter():
    # 检查在“合理”的迭代次数后所有样本是否已标记
    st = SelfTrainingClassifier(KNeighborsClassifier(), threshold=0.55, max_iter=None)
    st.fit(X_train, y_train_missing_labels)

    # 断言迭代次数小于10
    assert st.n_iter_ < 10
    # 断言终止条件为所有样本均已标记
    assert st.termination_condition_ == "all_labeled"


# 参数化测试

@pytest.mark.parametrize(
    "base_estimator",
    [KNeighborsClassifier(), SVC(gamma="scale", probability=True, random_state=0)],
)
@pytest.mark.parametrize("y", [y_train_missing_labels, y_train_missing_strings])
def test_zero_iterations(base_estimator, y):
    # 检查零次迭代的分类结果
    # 使用零次迭代的自训练分类器应该与使用监督分类器得到相同结果
    # 同时也断言字符串数组的预期工作方式

    # 创建零次迭代的自训练分类器对象
    clf1 = SelfTrainingClassifier(base_estimator, max_iter=0)

    # 对训练集进行拟合
    clf1.fit(X_train, y)

    # 创建基础分类器对象，并对部分标记样本进行拟合
    clf2 = base_estimator.fit(X_train[:n_labeled_samples], y[:n_labeled_samples])

    # 断言使用自训练分类器和基础分类器预测测试集时结果相同
    assert_array_equal(clf1.predict(X_test), clf2.predict(X_test))
    # 断言终止条件为达到最大迭代次数
    assert clf1.termination_condition_ == "max_iter"


# 测试传递已拟合分类器是否会引发错误

def test_prefitted_throws_error():
    # 测试传递已拟合分类器并调用预测是否会引发错误
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    st = SelfTrainingClassifier(knn)
    with pytest.raises(
        NotFittedError,
        match="This SelfTrainingClassifier instance is not fitted yet",
    ):
        st.predict(X_train)


# 参数化测试

@pytest.mark.parametrize("max_iter", range(1, 5))
def test_labeled_iter(max_iter):
    # 检查迭代0中标记的数据点数量是否相等
    #
    # 创建一个自训练分类器对象，基于K最近邻分类器作为基础分类器，并设定最大迭代次数
    st = SelfTrainingClassifier(KNeighborsClassifier(), max_iter=max_iter)

    # 使用自训练分类器对象拟合训练数据集和带有缺失标签的目标数据集
    st.fit(X_train, y_train_missing_labels)
    
    # 统计第0轮迭代中标记的数据点数量
    amount_iter_0 = len(st.labeled_iter_[st.labeled_iter_ == 0])
    
    # 断言第0轮迭代中标记的数据点数量应与预期的有标签样本数量相等
    assert amount_iter_0 == n_labeled_samples
    
    # 检查迭代次数的范围，确保标记的最大迭代次数不超过总迭代次数，且总迭代次数不超过设定的最大迭代次数
    assert np.max(st.labeled_iter_) <= st.n_iter_ <= max_iter
# 测试确保在完全标记的数据集上训练，与单独训练分类器得到相同结果
def test_no_unlabeled():
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)  # 使用训练数据 X_train 和标签 y_train 训练 KNN 分类器
    st = SelfTrainingClassifier(knn)  # 创建自训练分类器对象，基于训练好的 KNN 分类器
    with pytest.warns(UserWarning, match="y contains no unlabeled samples"):
        st.fit(X_train, y_train)  # 使用自训练分类器进行训练
    assert_array_equal(knn.predict(X_test), st.predict(X_test))  # 断言预测结果一致
    # 断言在第 0 次迭代中所有样本都已标记（因为没有未标记样本）
    assert np.all(st.labeled_iter_ == 0)
    assert st.termination_condition_ == "all_labeled"


# 测试早停机制
def test_early_stopping():
    svc = SVC(gamma="scale", probability=True)  # 创建 SVC 分类器对象
    st = SelfTrainingClassifier(svc)  # 创建自训练分类器对象，基于 SVC 分类器
    X_train_easy = [[1], [0], [1], [0.5]]
    y_train_easy = [1, 0, -1, -1]
    # 因为 X = [[0.5]] 不能以高置信度预测，所以训练过程会早停
    st.fit(X_train_easy, y_train_easy)
    assert st.n_iter_ == 1  # 断言迭代次数为 1
    assert st.termination_condition_ == "no_change"


# 测试标签为字符串类型时的异常处理
def test_strings_dtype():
    clf = SelfTrainingClassifier(KNeighborsClassifier())  # 创建自训练分类器对象
    X, y = make_blobs(n_samples=30, random_state=0, cluster_std=0.1)  # 创建数据集
    labels_multiclass = ["one", "two", "three"]
    y_strings = np.take(labels_multiclass, y)  # 将整数标签转换为字符串标签
    with pytest.raises(ValueError, match="dtype"):  # 检查是否抛出值错误异常，匹配错误消息 "dtype"
        clf.fit(X, y_strings)


# 测试 verbose 参数为 True 和 False 时的输出
@pytest.mark.parametrize("verbose", [True, False])
def test_verbose(capsys, verbose):
    clf = SelfTrainingClassifier(KNeighborsClassifier(), verbose=verbose)  # 创建自训练分类器对象
    clf.fit(X_train, y_train_missing_labels)  # 使用训练数据和缺失标签进行训练
    captured = capsys.readouterr()  # 捕获标准输出和错误输出
    if verbose:
        assert "iteration" in captured.out  # 如果 verbose 为 True，则输出中应包含 "iteration"
    else:
        assert "iteration" not in captured.out  # 如果 verbose 为 False，则输出中不应包含 "iteration"


# 测试使用 k_best 准则选择最佳标签
def test_verbose_k_best(capsys):
    st = SelfTrainingClassifier(
        KNeighborsClassifier(n_neighbors=1),  # 创建 KNN 分类器对象
        criterion="k_best",
        k_best=10,
        verbose=True,
        max_iter=None,
    )
    y_train_only_one_label = np.copy(y_train)
    y_train_only_one_label[1:] = -1  # 将除了第一个样本外的所有标签设为未标记
    n_samples = y_train.shape[0]
    n_expected_iter = ceil((n_samples - 1) / 10)  # 预期的迭代次数
    st.fit(X_train, y_train_only_one_label)  # 使用训练数据和部分标签进行训练
    captured = capsys.readouterr()  # 捕获标准输出和错误输出
    msg = "End of iteration {}, added {} new labels."
    for i in range(1, n_expected_iter):
        assert msg.format(i, 10) in captured.out  # 断言输出中包含预期的迭代信息
    assert msg.format(n_expected_iter, (n_samples - 1) % 10) in captured.out  # 断言输出中包含最后的迭代信息


# 测试使用 k_best 准则时选择的标签是否最佳
def test_k_best_selects_best():
    svc = SVC(gamma="scale", probability=True, random_state=0)  # 创建 SVC 分类器对象
    st = SelfTrainingClassifier(svc, criterion="k_best", max_iter=1, k_best=10)  # 创建自训练分类器对象
    has_label = y_train_missing_labels != -1  # 找出已标记的样本
    st.fit(X_train, y_train_missing_labels)  # 使用训练数据和缺失标签进行训练
    got_label = ~has_label & (st.transduction_ != -1)  # 确定自训练是否添加了标签
    svc.fit(X_train[has_label], y_train_missing_labels[has_label])  # 使用有标签的数据训练 SVC 分类器
    pred = svc.predict_proba(X_train[~has_label])  # 对未标签数据进行预测
    max_proba = np.max(pred, axis=1)  # 获取预测概率的最大值
    # 从 X_train 中选择没有标签的样本，并按照最高预测概率的顺序选取最自信的前十个样本
    most_confident_svc = X_train[~has_label][np.argsort(max_proba)[-10:]]
    
    # 从 X_train 中选择已经被赋予标签的样本，并将其转换为列表形式
    added_by_st = X_train[np.where(got_label)].tolist()
    
    # 遍历最自信的支持向量分类器（SVC）样本，并确保它们都在已经被赋予标签的样本列表中
    for row in most_confident_svc.tolist():
        assert row in added_by_st
# 定义一个测试函数，用于测试基础估计器和元估计器的行为
def test_base_estimator_meta_estimator():
    # 检查依赖于实现 `predict_proba` 方法的估计器的元估计器，即使在拟合之前未公开此方法，也能正常工作。
    # 非回归测试链接：
    # https://github.com/scikit-learn/scikit-learn/issues/19119

    # 使用包含 `probability=True` 的 SVC 作为基础估计器的 StackingClassifier
    base_estimator = StackingClassifier(
        estimators=[
            ("svc_1", SVC(probability=True)),
            ("svc_2", SVC(probability=True)),
        ],
        final_estimator=SVC(probability=True),
        cv=2,
    )

    # 断言 `base_estimator` 对象具有 `predict_proba` 方法
    assert hasattr(base_estimator, "predict_proba")

    # 使用 base_estimator 初始化 SelfTrainingClassifier
    clf = SelfTrainingClassifier(base_estimator=base_estimator)
    # 对训练集进行拟合
    clf.fit(X_train, y_train_missing_labels)
    # 对测试集进行预测概率计算
    clf.predict_proba(X_test)

    # 使用包含 `probability=False` 的 SVC 作为基础估计器的另一个 StackingClassifier
    base_estimator = StackingClassifier(
        estimators=[
            ("svc_1", SVC(probability=False)),
            ("svc_2", SVC(probability=False)),
        ],
        final_estimator=SVC(probability=False),
        cv=2,
    )

    # 断言 `base_estimator` 对象不具有 `predict_proba` 方法
    assert not hasattr(base_estimator, "predict_proba")

    # 使用 base_estimator 初始化 SelfTrainingClassifier
    clf = SelfTrainingClassifier(base_estimator=base_estimator)
    # 期望引发 AttributeError 异常
    with pytest.raises(AttributeError):
        clf.fit(X_train, y_train_missing_labels)


# 定义一个测试函数，检查当 `base_estimator` 未实现 `predict_proba` 方法或 `decision_function` 方法时是否能正确引发 AttributeError
def test_self_training_estimator_attribute_error():
    """Check that we raise the proper AttributeErrors when the `base_estimator`
    does not implement the `predict_proba` method, which is called from within
    `fit`, or `decision_function`, which is decorated with `available_if`.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/28108
    """
    # 使用 `probability=False` 的 SVC，该估计器未实现 `predict_proba` 方法，
    # 但在 `SelfTrainingClassifier` 的 `fit` 方法中需要。我们期望引发 AttributeError 异常。
    base_estimator = SVC(probability=False, gamma="scale")
    self_training = SelfTrainingClassifier(base_estimator)

    with pytest.raises(AttributeError, match="has no attribute 'predict_proba'"):
        self_training.fit(X_train, y_train_missing_labels)

    # `DecisionTreeClassifier` 未实现 'decision_function' 方法，应该引发 AttributeError 异常
    self_training = SelfTrainingClassifier(base_estimator=DecisionTreeClassifier())

    outer_msg = "This 'SelfTrainingClassifier' has no attribute 'decision_function'"
    inner_msg = "'DecisionTreeClassifier' object has no attribute 'decision_function'"
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        self_training.fit(X_train, y_train_missing_labels).decision_function(X_train)
    # 断言异常的原因是 AttributeError
    assert isinstance(exec_info.value.__cause__, AttributeError)
    # 断言异常信息包含正确的内部消息
    assert inner_msg in str(exec_info.value.__cause__)
```