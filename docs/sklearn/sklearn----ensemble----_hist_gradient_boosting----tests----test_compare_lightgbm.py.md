# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_hist_gradient_boosting\tests\test_compare_lightgbm.py`

```
# 导入所需的库和模块
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于单元测试

from sklearn.datasets import make_classification, make_regression  # 导入数据生成函数
from sklearn.ensemble import (  # 导入集成学习模型
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper  # 导入数据分箱工具
from sklearn.ensemble._hist_gradient_boosting.utils import get_equivalent_estimator  # 导入模型等效性工具
from sklearn.metrics import accuracy_score  # 导入准确率评估指标
from sklearn.model_selection import train_test_split  # 导入数据集分割工具


@pytest.mark.parametrize("seed", range(5))  # 参数化标记，测试多个种子值
@pytest.mark.parametrize(
    "loss",
    [
        "squared_error",  # 平方误差损失
        "poisson",  # 泊松损失
        pytest.param(
            "gamma",
            marks=pytest.mark.skip("LightGBM with gamma loss has larger deviation."),
        ),  # gamma损失，跳过该测试用例，并注明LightGBM的gamma损失有较大偏差
    ],
)
@pytest.mark.parametrize("min_samples_leaf", (1, 20))  # 参数化标记，测试多个叶子节点最小样本数
@pytest.mark.parametrize(
    "n_samples, max_leaf_nodes",
    [
        (255, 4096),  # 样本数为255，最大叶子节点数为4096
        (1000, 8),  # 样本数为1000，最大叶子节点数为8
    ],
)
def test_same_predictions_regression(
    seed, loss, min_samples_leaf, n_samples, max_leaf_nodes
):
    # 确保sklearn与LightGBM有相同的预测结果，以便于进行简单的目标对比。
    #
    # 特别是当树的大小受限且样本数足够大时，LightGBM和sklearn找到的预测树结构应完全相同。
    #
    # 注意：
    # - 当节点中样本数较少时（由于浮点误差），几个候选分裂可能具有相等的增益。因此，如果树的结构不完全相同，
    #   则测试集上的预测可能会有所不同。为了避免此问题，我们仅在样本数足够大且最大叶子节点数足够小时比较测试集上的预测。
    # - 如果n_samples > 255，则预先分箱数据以忽略由分箱策略中小差异引起的不一致性。
    # - 我们不在这里检查绝对误差损失。这是因为LightGBM计算中位数（用于原始预测的初始值）时有些偏差
    #   （例如，当不需要时，它们会返回中点）。由于这些测试仅运行1次迭代，初始值之间的差异导致预测有较大差异。
    #   使用更多迭代时，这些差异要小得多。
    pytest.importorskip("lightgbm")  # 如果不存在lightgbm库，则跳过测试

    rng = np.random.RandomState(seed=seed)  # 创建特定种子的随机数生成器
    max_iter = 1  # 设置迭代次数为1
    max_bins = 255  # 设置最大分箱数为255

    X, y = make_regression(
        n_samples=n_samples, n_features=5, n_informative=5, random_state=0
    )  # 生成回归数据集X和y

    if loss in ("gamma", "poisson"):
        # 使目标变为正数
        y = np.abs(y) + np.mean(np.abs(y))

    if n_samples > 255:
        # 对数据进行分箱并将其转换为float32类型，以防止估计器将其视为预先分箱的数据
        X = _BinMapper(n_bins=max_bins + 1).fit_transform(X).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)  # 将数据集拆分为训练集和测试集
    # 使用 HistGradientBoostingRegressor 创建 sklearn 的梯度提升回归器
    est_sklearn = HistGradientBoostingRegressor(
        loss=loss,                   # 损失函数
        max_iter=max_iter,           # 最大迭代次数
        max_bins=max_bins,           # 最大箱数
        learning_rate=1,             # 学习率
        early_stopping=False,        # 不启用早停策略
        min_samples_leaf=min_samples_leaf,   # 叶节点的最小样本数
        max_leaf_nodes=max_leaf_nodes,       # 最大叶节点数
    )
    # 获取与 est_sklearn 等效的 LightGBM 回归器
    est_lightgbm = get_equivalent_estimator(est_sklearn, lib="lightgbm")
    # 设置 LightGBM 回归器的参数 min_sum_hessian_in_leaf 为 0
    est_lightgbm.set_params(min_sum_hessian_in_leaf=0)

    # 使用训练数据 X_train 和 y_train 分别训练 LightGBM 和 sklearn 的回归器
    est_lightgbm.fit(X_train, y_train)
    est_sklearn.fit(X_train, y_train)

    # 将 X_train 和 X_test 的数据类型转换为 np.float32，以便作为数值型数据处理，而不是预先分箱的数据
    # 这是因为部分算法要求输入为数值型数据
    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)

    # 分别使用训练集 X_train 对 LightGBM 和 sklearn 的回归器进行预测
    pred_lightgbm = est_lightgbm.predict(X_train)
    pred_sklearn = est_sklearn.predict(X_train)

    # 如果损失函数为 "gamma" 或 "poisson"
    if loss in ("gamma", "poisson"):
        # 断言：超过 65% 的预测结果在相对误差和绝对误差均小于等于 0.01
        # 这是因为不同算法之间可能存在轻微的算法差异，导致精度不完全一致
        assert (
            np.mean(np.isclose(pred_lightgbm, pred_sklearn, rtol=1e-2, atol=1e-2))
            > 0.65
        )
    else:
        # 断言：超过 99% 的预测结果在相对误差小于等于 0.001
        assert np.mean(np.isclose(pred_lightgbm, pred_sklearn, rtol=1e-3)) > 1 - 0.01

    # 如果最大叶节点数小于 10，并且样本数大于等于 1000，并且损失函数为 "squared_error"
    if max_leaf_nodes < 10 and n_samples >= 1000 and loss in ("squared_error",):
        # 使用测试集 X_test 分别对 LightGBM 和 sklearn 的回归器进行预测
        pred_lightgbm = est_lightgbm.predict(X_test)
        pred_sklearn = est_sklearn.predict(X_test)
        # 断言：超过 99% 的预测结果在相对误差小于等于 0.0001
        assert np.mean(np.isclose(pred_lightgbm, pred_sklearn, rtol=1e-4)) > 1 - 0.01
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("min_samples_leaf", (1, 20))
@pytest.mark.parametrize(
    "n_samples, max_leaf_nodes",
    [
        (255, 4096),
        (1000, 8),
    ],
)
def test_same_predictions_classification(
    seed, min_samples_leaf, n_samples, max_leaf_nodes
):
    # Same as test_same_predictions_regression but for classification
    # 指定测试参数化：种子范围为0到4，最小叶子样本数为1或20
    # n_samples和max_leaf_nodes分别取(255, 4096)和(1000, 8)两组值

    pytest.importorskip("lightgbm")  # 导入lightgbm模块，若失败则跳过当前测试

    rng = np.random.RandomState(seed=seed)  # 使用种子创建随机数生成器rng
    max_iter = 1  # 设定最大迭代次数为1
    n_classes = 2  # 分类任务的类别数为2
    max_bins = 255  # 设定最大箱数为255

    X, y = make_classification(
        n_samples=n_samples,
        n_classes=n_classes,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        random_state=0,
    )
    # 生成分类任务的特征矩阵X和目标向量y，特征数为5，信息特征数为5，无冗余特征，随机种子为0

    if n_samples > 255:
        # 如果样本数大于255，则对数据进行分箱并转换为float32类型，以避免模型将其视为预先分箱数据
        X = _BinMapper(n_bins=max_bins + 1).fit_transform(X).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)
    # 使用随机数生成器rng对数据集进行训练集和测试集的划分

    est_sklearn = HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=max_iter,
        max_bins=max_bins,
        learning_rate=1,
        early_stopping=False,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=max_leaf_nodes,
    )
    # 创建Scikit-Learn的HistGradientBoostingClassifier分类器est_sklearn对象

    est_lightgbm = get_equivalent_estimator(
        est_sklearn, lib="lightgbm", n_classes=n_classes
    )
    # 调用get_equivalent_estimator函数获取与Scikit-Learn兼容的LightGBM分类器est_lightgbm对象

    est_lightgbm.fit(X_train, y_train)  # 使用LightGBM分类器对训练集进行拟合
    est_sklearn.fit(X_train, y_train)  # 使用Scikit-Learn分类器对训练集进行拟合

    # 需要X被视为数值数据，而不是预先分箱数据。
    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)
    # 将训练集和测试集数据类型转换为float32

    pred_lightgbm = est_lightgbm.predict(X_train)  # 使用LightGBM分类器对训练集进行预测
    pred_sklearn = est_sklearn.predict(X_train)  # 使用Scikit-Learn分类器对训练集进行预测
    assert np.mean(pred_sklearn == pred_lightgbm) > 0.89  # 断言两个模型预测结果的准确率均大于0.89

    acc_lightgbm = accuracy_score(y_train, pred_lightgbm)  # 计算LightGBM分类器在训练集上的准确率
    acc_sklearn = accuracy_score(y_train, pred_sklearn)  # 计算Scikit-Learn分类器在训练集上的准确率
    np.testing.assert_almost_equal(acc_lightgbm, acc_sklearn)  # 断言两者准确率几乎相等

    if max_leaf_nodes < 10 and n_samples >= 1000:
        # 如果最大叶子节点数小于10且样本数大于等于1000
        pred_lightgbm = est_lightgbm.predict(X_test)  # 使用LightGBM分类器对测试集进行预测
        pred_sklearn = est_sklearn.predict(X_test)  # 使用Scikit-Learn分类器对测试集进行预测
        assert np.mean(pred_sklearn == pred_lightgbm) > 0.89  # 断言两个模型预测结果的准确率均大于0.89

        acc_lightgbm = accuracy_score(y_test, pred_lightgbm)  # 计算LightGBM分类器在测试集上的准确率
        acc_sklearn = accuracy_score(y_test, pred_sklearn)  # 计算Scikit-Learn分类器在测试集上的准确率
        np.testing.assert_almost_equal(acc_lightgbm, acc_sklearn, decimal=2)  # 断言两者准确率几乎相等
    # 使用 make_classification 函数生成分类数据集 X 和标签 y
    X, y = make_classification(
        n_samples=n_samples,
        n_classes=n_classes,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=0,
    )

    # 如果样本数大于 255，则对数据进行二进制编码，并转换为 float32 类型，
    # 以避免估算器将其视为预先编码的数据
    if n_samples > 255:
        X = _BinMapper(n_bins=max_bins + 1).fit_transform(X).astype(np.float32)

    # 将数据集分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)

    # 使用 HistGradientBoostingClassifier 初始化 sklearn 和 lightgbm 两个分类器
    est_sklearn = HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=max_iter,
        max_bins=max_bins,
        learning_rate=lr,
        early_stopping=False,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=max_leaf_nodes,
    )
    # 获得与 sklearn 相对应的 lightgbm 分类器
    est_lightgbm = get_equivalent_estimator(
        est_sklearn, lib="lightgbm", n_classes=n_classes
    )

    # 分别使用训练集拟合两个分类器
    est_lightgbm.fit(X_train, y_train)
    est_sklearn.fit(X_train, y_train)

    # 需要确保 X_train 数据被视为数值型数据，而不是预先编码的数据
    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)

    # 使用两个分类器对训练集进行预测
    pred_lightgbm = est_lightgbm.predict(X_train)
    pred_sklearn = est_sklearn.predict(X_train)
    # 断言两个分类器的预测准确率至少达到 89%
    assert np.mean(pred_sklearn == pred_lightgbm) > 0.89

    # 获取两个分类器对训练集的预测概率
    proba_lightgbm = est_lightgbm.predict_proba(X_train)
    proba_sklearn = est_sklearn.predict_proba(X_train)
    # 断言至少有 75% 的预测概率在第二位小数处相同
    assert np.mean(np.abs(proba_lightgbm - proba_sklearn) < 1e-2) > 0.75

    # 计算两个分类器在训练集上的准确率
    acc_lightgbm = accuracy_score(y_train, pred_lightgbm)
    acc_sklearn = accuracy_score(y_train, pred_sklearn)

    # 使用 np.testing.assert_allclose 函数断言两个分类器的准确率相等，允许误差范围在 5e-2
    np.testing.assert_allclose(acc_lightgbm, acc_sklearn, rtol=0, atol=5e-2)

    # 如果 max_leaf_nodes 小于 10 并且样本数大于等于 1000，则在测试集上进行额外的测试
    if max_leaf_nodes < 10 and n_samples >= 1000:
        # 使用两个分类器对测试集进行预测
        pred_lightgbm = est_lightgbm.predict(X_test)
        pred_sklearn = est_sklearn.predict(X_test)
        # 断言两个分类器的测试集预测准确率至少达到 89%
        assert np.mean(pred_sklearn == pred_lightgbm) > 0.89

        # 获取两个分类器对训练集的预测概率
        proba_lightgbm = est_lightgbm.predict_proba(X_train)
        proba_sklearn = est_sklearn.predict_proba(X_train)
        # 断言至少有 75% 的测试集预测概率在第二位小数处相同
        assert np.mean(np.abs(proba_lightgbm - proba_sklearn) < 1e-2) > 0.75

        # 计算两个分类器在测试集上的准确率
        acc_lightgbm = accuracy_score(y_test, pred_lightgbm)
        acc_sklearn = accuracy_score(y_test, pred_sklearn)
        # 使用 np.testing.assert_almost_equal 函数断言两个分类器的测试集准确率近似相等，精确到小数点后两位
        np.testing.assert_almost_equal(acc_lightgbm, acc_sklearn, decimal=2)
```