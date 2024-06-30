# `D:\src\scipysrc\scikit-learn\sklearn\tests\test_calibration.py`

```
# 导入所需的库和模块
import numpy as np  # 导入NumPy库
import pytest  # 导入pytest用于单元测试
from numpy.testing import assert_allclose  # 导入NumPy的测试工具函数

from sklearn.base import BaseEstimator, clone  # 导入sklearn基础模块
from sklearn.calibration import (
    CalibratedClassifierCV,  # 导入用于分类器校准的模块
    CalibrationDisplay,  # 导入用于校准显示的模块
    _CalibratedClassifier,  # 导入内部的校准分类器类
    _sigmoid_calibration,  # 导入sigmoid校准函数
    _SigmoidCalibration,  # 导入sigmoid校准类
    calibration_curve,  # 导入校准曲线函数
)
from sklearn.datasets import load_iris, make_blobs, make_classification  # 导入数据集生成和加载模块
from sklearn.dummy import DummyClassifier  # 导入虚拟分类器
from sklearn.ensemble import (
    RandomForestClassifier,  # 导入随机森林分类器
    VotingClassifier,  # 导入投票分类器
)
from sklearn.exceptions import NotFittedError  # 导入未拟合错误类
from sklearn.feature_extraction import DictVectorizer  # 导入字典向量化模块
from sklearn.impute import SimpleImputer  # 导入简单填充模块
from sklearn.isotonic import IsotonicRegression  # 导入保序回归模块
from sklearn.linear_model import LogisticRegression, SGDClassifier  # 导入逻辑回归和随机梯度分类器
from sklearn.metrics import brier_score_loss  # 导入Brier分数损失度量
from sklearn.model_selection import (
    KFold,  # 导入K折交叉验证
    LeaveOneOut,  # 导入留一法交叉验证
    check_cv,  # 导入验证交叉验证函数
    cross_val_predict,  # 导入交叉验证预测函数
    cross_val_score,  # 导入交叉验证得分函数
    train_test_split,  # 导入训练测试集分割函数
)
from sklearn.naive_bayes import MultinomialNB  # 导入多项式朴素贝叶斯分类器
from sklearn.pipeline import Pipeline, make_pipeline  # 导入管道构建函数
from sklearn.preprocessing import LabelEncoder, StandardScaler  # 导入标签编码器和标准化模块
from sklearn.svm import LinearSVC  # 导入线性支持向量分类器
from sklearn.tree import DecisionTreeClassifier  # 导入决策树分类器
from sklearn.utils._mocking import CheckingClassifier  # 导入用于检查的模拟模块
from sklearn.utils._testing import (
    _convert_container,  # 导入转换容器函数
    assert_almost_equal,  # 导入近似相等断言函数
    assert_array_almost_equal,  # 导入数组近似相等断言函数
    assert_array_equal,  # 导入数组相等断言函数
)
from sklearn.utils.extmath import softmax  # 导入softmax函数
from sklearn.utils.fixes import CSR_CONTAINERS  # 导入CSR容器修复模块

N_SAMPLES = 200  # 设置样本数量

@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=N_SAMPLES, n_features=6, random_state=42)
    return X, y

@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
@pytest.mark.parametrize("ensemble", [True, False])
def test_calibration(data, method, csr_container, ensemble):
    # 测试校准对象使用保序和sigmoid方法
    n_samples = N_SAMPLES // 2
    X, y = data
    sample_weight = np.random.RandomState(seed=42).uniform(size=y.size)

    X -= X.min()  # 多项式朴素贝叶斯要求X为正值

    # 分割训练集和测试集
    X_train, y_train, sw_train = X[:n_samples], y[:n_samples], sample_weight[:n_samples]
    X_test, y_test = X[n_samples:], y[n_samples:]

    # 朴素贝叶斯分类器
    clf = MultinomialNB().fit(X_train, y_train, sample_weight=sw_train)
    prob_pos_clf = clf.predict_proba(X_test)[:, 1]

    cal_clf = CalibratedClassifierCV(clf, cv=y.size + 1, ensemble=ensemble)
    with pytest.raises(ValueError):
        cal_clf.fit(X, y)

    # 带校准的朴素贝叶斯
    for this_X_train, this_X_test in [
        (X_train, X_test),
        (csr_container(X_train), csr_container(X_test)),
        cal_clf = CalibratedClassifierCV(clf, method=method, cv=5, ensemble=ensemble)
        # 使用CalibratedClassifierCV对分类器进行校准，设置方法和交叉验证折数
        # 注意：此处的fit会覆盖整个训练集上的fit操作
        cal_clf.fit(this_X_train, y_train, sample_weight=sw_train)
        # 使用校准后的分类器预测测试集的正类概率
        prob_pos_cal_clf = cal_clf.predict_proba(this_X_test)[:, 1]

        # 检查校准后的Brier分数是否比未校准时有所改善
        assert brier_score_loss(y_test, prob_pos_clf) > brier_score_loss(
            y_test, prob_pos_cal_clf
        )

        # 检查对标签进行重新标记（[0, 1] -> [1, 2]）后的不变性
        cal_clf.fit(this_X_train, y_train + 1, sample_weight=sw_train)
        prob_pos_cal_clf_relabeled = cal_clf.predict_proba(this_X_test)[:, 1]
        # 断言校准前后的正类概率数组几乎相等
        assert_array_almost_equal(prob_pos_cal_clf, prob_pos_cal_clf_relabeled)

        # 检查对标签进行重新标记（[0, 1] -> [-1, 1]）后的不变性
        cal_clf.fit(this_X_train, 2 * y_train - 1, sample_weight=sw_train)
        prob_pos_cal_clf_relabeled = cal_clf.predict_proba(this_X_test)[:, 1]
        # 断言校准前后的正类概率数组几乎相等
        assert_array_almost_equal(prob_pos_cal_clf, prob_pos_cal_clf_relabeled)

        # 检查对标签进行重新标记（[0, 1] -> [1, 0]）后的不变性
        cal_clf.fit(this_X_train, (y_train + 1) % 2, sample_weight=sw_train)
        prob_pos_cal_clf_relabeled = cal_clf.predict_proba(this_X_test)[:, 1]
        if method == "sigmoid":
            # 对于sigmoid方法，断言校准前后的正类概率数组几乎互为补数
            assert_array_almost_equal(prob_pos_cal_clf, 1 - prob_pos_cal_clf_relabeled)
        else:
            # 对于其它方法，虽然不满足重新标记的不变性，但应该在两种情况下都有所改善
            assert brier_score_loss(y_test, prob_pos_clf) > brier_score_loss(
                (y_test + 1) % 2, prob_pos_cal_clf_relabeled
            )
# 测试默认情况下的校准估算器是否为 LinearSVC
def test_calibration_default_estimator(data):
    # 解包数据
    X, y = data
    # 创建校准分类器CV对象，默认使用2折交叉验证
    calib_clf = CalibratedClassifierCV(cv=2)
    # 使用数据 X, y 进行拟合
    calib_clf.fit(X, y)

    # 获取基础估算器
    base_est = calib_clf.calibrated_classifiers_[0].estimator
    # 断言基础估算器是否为 LinearSVC 类型
    assert isinstance(base_est, LinearSVC)


@pytest.mark.parametrize("ensemble", [True, False])
def test_calibration_cv_splitter(data, ensemble):
    # 检查当 `cv` 是一个交叉验证分离器时的情况
    X, y = data

    splits = 5
    # 创建 KFold 分离器，将数据分为 splits 份
    kfold = KFold(n_splits=splits)
    # 创建校准分类器CV对象，指定交叉验证分离器和是否集成
    calib_clf = CalibratedClassifierCV(cv=kfold, ensemble=ensemble)
    # 断言校准分类器的交叉验证对象类型为 KFold
    assert isinstance(calib_clf.cv, KFold)
    # 断言交叉验证的分割数是否为 splits
    assert calib_clf.cv.n_splits == splits

    # 使用数据 X, y 进行拟合
    calib_clf.fit(X, y)
    # 期望的校准分类器数量
    expected_n_clf = splits if ensemble else 1
    # 断言校准分类器集合的长度是否符合预期
    assert len(calib_clf.calibrated_classifiers_) == expected_n_clf


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
@pytest.mark.parametrize("ensemble", [True, False])
def test_sample_weight(data, method, ensemble):
    n_samples = N_SAMPLES // 2
    X, y = data

    # 创建随机种子为 42 的随机数生成器，生成样本权重
    sample_weight = np.random.RandomState(seed=42).uniform(size=len(y))
    X_train, y_train, sw_train = X[:n_samples], y[:n_samples], sample_weight[:n_samples]
    X_test = X[n_samples:]

    # 创建 LinearSVC 估算器
    estimator = LinearSVC(random_state=42)
    # 创建校准分类器CV对象，指定估算方法和是否集成
    calibrated_clf = CalibratedClassifierCV(estimator, method=method, ensemble=ensemble)
    # 使用部分样本进行拟合，同时传入样本权重
    calibrated_clf.fit(X_train, y_train, sample_weight=sw_train)
    # 使用带有样本权重的测试数据集进行预测
    probs_with_sw = calibrated_clf.predict_proba(X_test)

    # 因为权重用于校准，所以它们应该产生不同的预测
    # 再次拟合，此时没有样本权重
    calibrated_clf.fit(X_train, y_train)
    # 使用无样本权重的测试数据集进行预测
    probs_without_sw = calibrated_clf.predict_proba(X_test)

    # 计算两者之间的差异
    diff = np.linalg.norm(probs_with_sw - probs_without_sw)
    # 断言差异应大于 0.1
    assert diff > 0.1


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
@pytest.mark.parametrize("ensemble", [True, False])
def test_parallel_execution(data, method, ensemble):
    """测试并行校准"""
    X, y = data
    # 将数据集分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # 创建管道对象，包括数据标准化和 LinearSVC 估算器
    estimator = make_pipeline(StandardScaler(), LinearSVC(random_state=42))

    # 创建并行校准分类器CV对象，指定估算方法、并行任务数和是否集成
    cal_clf_parallel = CalibratedClassifierCV(
        estimator, method=method, n_jobs=2, ensemble=ensemble
    )
    # 使用训练数据进行拟合
    cal_clf_parallel.fit(X_train, y_train)
    # 使用测试数据进行预测
    probs_parallel = cal_clf_parallel.predict_proba(X_test)

    # 创建串行校准分类器CV对象，指定估算方法、并行任务数为 1 和是否集成
    cal_clf_sequential = CalibratedClassifierCV(
        estimator, method=method, n_jobs=1, ensemble=ensemble
    )
    # 使用训练数据进行拟合
    cal_clf_sequential.fit(X_train, y_train)
    # 使用测试数据进行预测
    probs_sequential = cal_clf_sequential.predict_proba(X_test)

    # 断言并行和串行预测结果的近似程度
    assert_allclose(probs_parallel, probs_sequential)


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
@pytest.mark.parametrize("ensemble", [True, False])
# 增加随机数种子的数量，以评估该测试的统计稳定性
@pytest.mark.parametrize("seed", range(2))
def test_calibration_multiclass(method, ensemble, seed):
    """多分类校准测试"""
    # 定义一个函数，计算多类别问题的布里尔分数
    def multiclass_brier(y_true, proba_pred, n_classes):
        # 将真实标签转换为独热编码形式
        Y_onehot = np.eye(n_classes)[y_true]
        # 计算布里尔分数并返回
        return np.sum((Y_onehot - proba_pred) ** 2) / Y_onehot.shape[0]

    # 使用LinearSVC作为分类器，设置随机种子为7
    clf = LinearSVC(random_state=7)
    # 生成一个包含500个样本、100个特征的随机数据集，设置中心点为10个，标准差为15.0
    X, y = make_blobs(
        n_samples=500, n_features=100, random_state=seed, centers=10, cluster_std=15.0
    )

    # 将类别标签大于2的样本归为类别2，构造一个不平衡的数据集
    y[y > 2] = 2
    # 计算类别数目
    n_classes = np.unique(y).shape[0]
    # 将数据集划分为训练集和测试集
    X_train, y_train = X[::2], y[::2]
    X_test, y_test = X[1::2], y[1::2]

    # 使用训练集训练分类器
    clf.fit(X_train, y_train)

    # 对分类器进行校准，使用交叉验证方法为method，cv=5，集成方式为ensemble
    cal_clf = CalibratedClassifierCV(clf, method=method, cv=5, ensemble=ensemble)
    cal_clf.fit(X_train, y_train)
    # 预测测试集的概率
    probas = cal_clf.predict_proba(X_test)
    # 检查预测概率的和是否为1
    assert_allclose(np.sum(probas, axis=1), np.ones(len(X_test)))

    # 检查测试集的分类精度是否在0.65到0.95之间
    assert 0.65 < clf.score(X_test, y_test) < 0.95

    # 检查校准后模型的精度是否未下降超过原始分类器的95%
    assert cal_clf.score(X_test, y_test) > 0.95 * clf.score(X_test, y_test)

    # 检查校准分类器的布里尔损失是否小于通过softmax转换OvR决策函数得到的未校准布里尔损失的1.1倍
    uncalibrated_brier = multiclass_brier(
        y_test, softmax(clf.decision_function(X_test)), n_classes=n_classes
    )
    calibrated_brier = multiclass_brier(y_test, probas, n_classes=n_classes)
    assert calibrated_brier < 1.1 * uncalibrated_brier

    # 使用RandomForestClassifier作为分类器，设置树的数量为30，随机种子为42
    clf = RandomForestClassifier(n_estimators=30, random_state=42)
    clf.fit(X_train, y_train)
    # 预测测试集的概率
    clf_probs = clf.predict_proba(X_test)
    # 计算未校准的布里尔损失
    uncalibrated_brier = multiclass_brier(y_test, clf_probs, n_classes=n_classes)

    # 对随机森林分类器进行校准，方法为method，cv=5，集成方式为ensemble
    cal_clf = CalibratedClassifierCV(clf, method=method, cv=5, ensemble=ensemble)
    cal_clf.fit(X_train, y_train)
    # 预测测试集的概率
    cal_clf_probs = cal_clf.predict_proba(X_test)
    # 计算校准后的布里尔损失
    calibrated_brier = multiclass_brier(y_test, cal_clf_probs, n_classes=n_classes)
    # 检查校准后的布里尔损失是否小于通过softmax转换OvR决策函数得到的未校准布里尔损失的1.1倍
    assert calibrated_brier < 1.1 * uncalibrated_brier
def test_calibration_zero_probability():
    # 测试一个边缘案例，即如果所有校准器同时输出给定样本的概率为零，
    # 则在多类别归一化步骤中避免数值错误，并回退到均匀概率。

    class ZeroCalibrator:
        # 这个函数被从 _CalibratedClassifier.predict_proba 调用。
        def predict(self, X):
            return np.zeros(X.shape[0])

    # 创建一个样本数据集
    X, y = make_blobs(
        n_samples=50, n_features=10, random_state=7, centers=10, cluster_std=15.0
    )

    # 使用 DummyClassifier 拟合数据
    clf = DummyClassifier().fit(X, y)

    # 创建一个 ZeroCalibrator 实例
    calibrator = ZeroCalibrator()

    # 创建 _CalibratedClassifier 对象
    cal_clf = _CalibratedClassifier(
        estimator=clf, calibrators=[calibrator], classes=clf.classes_
    )

    # 预测样本的概率
    probas = cal_clf.predict_proba(X)

    # 检查所有概率是否均匀为 1.0 / clf.n_classes_
    assert_allclose(probas, 1.0 / clf.n_classes_)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_calibration_prefit(csr_container):
    """测试预拟合分类器的校准"""

    # 创建一个样本数据集
    n_samples = 50
    X, y = make_classification(n_samples=3 * n_samples, n_features=6, random_state=42)
    sample_weight = np.random.RandomState(seed=42).uniform(size=y.size)

    X -= X.min()  # MultinomialNB 只允许正数 X

    # 分割训练集和测试集
    X_train, y_train, sw_train = X[:n_samples], y[:n_samples], sample_weight[:n_samples]
    X_calib, y_calib, sw_calib = (
        X[n_samples : 2 * n_samples],
        y[n_samples : 2 * n_samples],
        sample_weight[n_samples : 2 * n_samples],
    )
    X_test, y_test = X[2 * n_samples :], y[2 * n_samples :]

    # 多项式朴素贝叶斯
    clf = MultinomialNB()

    # 如果 clf 没有预拟合，检查错误
    unfit_clf = CalibratedClassifierCV(clf, cv="prefit")
    with pytest.raises(NotFittedError):
        unfit_clf.fit(X_calib, y_calib)

    # 使用训练集拟合分类器
    clf.fit(X_train, y_train, sw_train)
    prob_pos_clf = clf.predict_proba(X_test)[:, 1]

    # 带校准的朴素贝叶斯
    for this_X_calib, this_X_test in [
        (X_calib, X_test),
        (csr_container(X_calib), csr_container(X_test)),
    ]:
        for method in ["isotonic", "sigmoid"]:
            cal_clf = CalibratedClassifierCV(clf, method=method, cv="prefit")

            for sw in [sw_calib, None]:
                cal_clf.fit(this_X_calib, y_calib, sample_weight=sw)
                y_prob = cal_clf.predict_proba(this_X_test)
                y_pred = cal_clf.predict(this_X_test)
                prob_pos_cal_clf = y_prob[:, 1]
                assert_array_equal(y_pred, np.array([0, 1])[np.argmax(y_prob, axis=1)])

                assert brier_score_loss(y_test, prob_pos_clf) > brier_score_loss(
                    y_test, prob_pos_cal_clf
                )


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibration_ensemble_false(data, method):
    # 测试 `ensemble=False` 与使用 `cross_val_predict` 预测训练校准器的结果是否相同。
    # 将数据集拆分为特征向量 X 和目标变量 y
    X, y = data
    # 使用线性支持向量分类器（LinearSVC）创建分类器对象
    clf = LinearSVC(random_state=7)

    # 创建校准分类器，用于对原始分类器进行概率校准
    cal_clf = CalibratedClassifierCV(clf, method=method, cv=3, ensemble=False)
    # 对数据进行拟合，训练校准分类器
    cal_clf.fit(X, y)
    # 使用校准分类器预测每个类别的概率
    cal_probas = cal_clf.predict_proba(X)

    # 手动获取概率预测值
    # 使用交叉验证方法对数据进行预测并获取无偏预测结果
    unbiased_preds = cross_val_predict(clf, X, y, cv=3, method="decision_function")
    # 根据选择的校准方法创建相应的校准器
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
    else:
        calibrator = _SigmoidCalibration()
    # 对无偏预测结果进行校准器拟合
    calibrator.fit(unbiased_preds, y)
    # 使用原始分类器 clf 在全部数据上进行拟合
    clf.fit(X, y)
    # 获取原始分类器在全部数据上的决策函数值
    clf_df = clf.decision_function(X)
    # 使用校准器对决策函数值进行预测，得到手动计算的概率预测值
    manual_probas = calibrator.predict(clf_df)
    # 断言校准后的概率预测值与校准分类器预测的概率值接近
    assert_allclose(cal_probas[:, 1], manual_probas)
def test_sigmoid_calibration():
    """Test calibration values with Platt sigmoid model"""
    # 示例的计算值，从我的 Python 版本的 LibSVM C++ 代码端口计算得出
    exF = np.array([5, -4, 1.0])
    exY = np.array([1, -1, -1])
    AB_lin_libsvm = np.array([-0.20261354391187855, 0.65236314980010512])
    # 断言两个数组几乎相等，精度为 3
    assert_array_almost_equal(AB_lin_libsvm, _sigmoid_calibration(exF, exY), 3)
    # 计算线性概率
    lin_prob = 1.0 / (1.0 + np.exp(AB_lin_libsvm[0] * exF + AB_lin_libsvm[1]))
    # 使用 Sklearn 的 _SigmoidCalibration 拟合数据并预测概率
    sk_prob = _SigmoidCalibration().fit(exF, exY).predict(exF)
    # 断言两个概率数组几乎相等，精度为 6
    assert_array_almost_equal(lin_prob, sk_prob, 6)

    # 检查 _SigmoidCalibration().fit 只接受 1 维数组或 2 维列数组
    with pytest.raises(ValueError):
        _SigmoidCalibration().fit(np.vstack((exF, exF)), exY)


def test_calibration_curve():
    """Check calibration_curve function"""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    # 调用 calibration_curve 函数，生成真实概率和预测概率
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=2)
    # 断言真实概率和预测概率数组长度相等
    assert len(prob_true) == len(prob_pred)
    # 断言真实概率和预测概率数组长度为 2
    assert len(prob_true) == 2
    # 断言真实概率数组几乎等于 [0, 1]
    assert_almost_equal(prob_true, [0, 1])
    # 断言预测概率数组几乎等于 [0.1, 0.9]
    assert_almost_equal(prob_pred, [0.1, 0.9])

    # 概率值超出 [0, 1] 范围应该被拒绝
    with pytest.raises(ValueError):
        calibration_curve([1], [-0.1])

    # 测试分位数策略的行为
    y_true2 = np.array([0, 0, 0, 0, 1, 1])
    y_pred2 = np.array([0.0, 0.1, 0.2, 0.5, 0.9, 1.0])
    # 调用 calibration_curve 函数，生成真实概率和预测概率（使用分位数策略）
    prob_true_quantile, prob_pred_quantile = calibration_curve(
        y_true2, y_pred2, n_bins=2, strategy="quantile"
    )
    # 断言真实概率和预测概率数组长度相等
    assert len(prob_true_quantile) == len(prob_pred_quantile)
    # 断言真实概率和预测概率数组长度为 2
    assert len(prob_true_quantile) == 2
    # 断言真实概率数组几乎等于 [0, 2/3]
    assert_almost_equal(prob_true_quantile, [0, 2 / 3])
    # 断言预测概率数组几乎等于 [0.1, 0.8]
    assert_almost_equal(prob_pred_quantile, [0.1, 0.8])

    # 选择无效策略时应该引发错误
    with pytest.raises(ValueError):
        calibration_curve(y_true2, y_pred2, strategy="percentile")


@pytest.mark.parametrize("ensemble", [True, False])
def test_calibration_nan_imputer(ensemble):
    """Test that calibration can accept nan"""
    # 创建包含 NaN 值的样本和标签
    X, y = make_classification(
        n_samples=10, n_features=2, n_informative=2, n_redundant=0, random_state=42
    )
    # 将第一个样本的第一个特征设置为 NaN
    X[0, 0] = np.nan
    # 创建包含简单填充器和随机森林分类器的 Pipeline
    clf = Pipeline(
        [("imputer", SimpleImputer()), ("rf", RandomForestClassifier(n_estimators=1))]
    )
    # 创建带有分类器交叉验证的校准分类器
    clf_c = CalibratedClassifierCV(clf, cv=2, method="isotonic", ensemble=ensemble)
    # 拟合数据并进行预测
    clf_c.fit(X, y)
    clf_c.predict(X)


@pytest.mark.parametrize("ensemble", [True, False])
def test_calibration_prob_sum(ensemble):
    # 测试概率之和为 1 的校准性质，用于 issue #7796 的非回归测试
    num_classes = 2
    # 创建包含多类别的样本和标签
    X, y = make_classification(n_samples=10, n_features=5, n_classes=num_classes)
    # 创建线性支持向量分类器
    clf = LinearSVC(C=1.0, random_state=7)
    # 创建带有 Sigmoid 方法的校准分类器交叉验证
    clf_prob = CalibratedClassifierCV(
        clf, method="sigmoid", cv=LeaveOneOut(), ensemble=ensemble
    )
    # 拟合数据
    clf_prob.fit(X, y)

    # 预测概率
    probs = clf_prob.predict_proba(X)
    # 使用断言检查二维数组 probs 的每一行的和是否几乎等于 1，即检查每行概率值的总和是否接近 1
    assert_array_almost_equal(probs.sum(axis=1), np.ones(probs.shape[0]))
@pytest.mark.parametrize("ensemble", [True, False])
# 使用 pytest 的参数化装饰器，定义测试函数 test_calibration_less_classes，参数 ensemble 可以为 True 或 False
def test_calibration_less_classes(ensemble):
    # 测试校准在训练集不包含所有类别时的表现
    # 由于这个测试使用 LOO（留一法），每次迭代训练集都不包含一个类标签
    X = np.random.randn(10, 5)
    y = np.arange(10)
    clf = LinearSVC(C=1.0, random_state=7)
    # 创建校准分类器对象，使用 LOO 交叉验证，可能使用集成方法
    cal_clf = CalibratedClassifierCV(
        clf, method="sigmoid", cv=LeaveOneOut(), ensemble=ensemble
    )
    # 使用数据 X, y 来拟合校准分类器
    cal_clf.fit(X, y)

    # 遍历校准分类器的每个校准后的分类器
    for i, calibrated_classifier in enumerate(cal_clf.calibrated_classifiers_):
        # 预测 X 的概率
        proba = calibrated_classifier.predict_proba(X)
        if ensemble:
            # 检查未观察到的类别的概率是否为 0
            assert_array_equal(proba[:, i], np.zeros(len(y)))
            # 检查其他所有类别的概率是否大于 0
            assert np.all(proba[:, :i] > 0)
            assert np.all(proba[:, i + 1 :] > 0)
        else:
            # 检查 `proba` 是否都接近于 1/n_classes
            assert np.allclose(proba, 1 / proba.shape[0])


@pytest.mark.parametrize(
    "X",
    [
        np.random.RandomState(42).randn(15, 5, 2),
        np.random.RandomState(42).randn(15, 5, 2, 6),
    ],
)
# 使用 pytest 的参数化装饰器，定义测试函数 test_calibration_accepts_ndarray，参数 X 是一个数组
def test_calibration_accepts_ndarray(X):
    """Test that calibration accepts n-dimensional arrays as input"""
    y = [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0]

    class MockTensorClassifier(BaseEstimator):
        """A toy estimator that accepts tensor inputs"""

        _estimator_type = "classifier"

        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self

        def decision_function(self, X):
            # 简单的决策函数，只需要正确的形状
            return X.reshape(X.shape[0], -1).sum(axis=1)

    calibrated_clf = CalibratedClassifierCV(MockTensorClassifier())
    # 应能够无错误地拟合该分类器
    calibrated_clf.fit(X, y)


@pytest.fixture
# 定义名为 dict_data 的夹具函数，返回一个包含字典数据和文本标签的元组
def dict_data():
    dict_data = [
        {"state": "NY", "age": "adult"},
        {"state": "TX", "age": "adult"},
        {"state": "VT", "age": "child"},
    ]
    text_labels = [1, 0, 1]
    return dict_data, text_labels


@pytest.fixture
# 定义名为 dict_data_pipeline 的夹具函数，接受 dict_data 作为参数
def dict_data_pipeline(dict_data):
    X, y = dict_data
    # 创建一个 Pipeline 对象，包含 DictVectorizer 和 RandomForestClassifier
    pipeline_prefit = Pipeline(
        [("vectorizer", DictVectorizer()), ("clf", RandomForestClassifier())]
    )
    return pipeline_prefit.fit(X, y)


def test_calibration_dict_pipeline(dict_data, dict_data_pipeline):
    """Test that calibration works in prefit pipeline with transformer

    `X` is not array-like, sparse matrix or dataframe at the start.
    See https://github.com/scikit-learn/scikit-learn/issues/8710

    Also test it can predict without running into validation errors.
    See https://github.com/scikit-learn/scikit-learn/issues/19637
    """
    X, y = dict_data
    clf = dict_data_pipeline
    # 创建校准分类器对象，使用 'prefit' 模式
    calib_clf = CalibratedClassifierCV(clf, cv="prefit")
    # 使用数据 X, y 来拟合校准分类器
    calib_clf.fit(X, y)
    # 检查属性是否从拟合的估计器中获取
    assert_array_equal(calib_clf.classes_, clf.classes_)

    # 确保在这种数据上，管道或校准元估计器均不公开 n_features_in_ 检查。
    assert not hasattr(clf, "n_features_in_")
    assert not hasattr(calib_clf, "n_features_in_")

    # 确保在使用 predict 和 predict_proba 函数时不会抛出错误
    calib_clf.predict(X)
    calib_clf.predict_proba(X)
@pytest.mark.parametrize(
    "clf, cv",
    [
        pytest.param(LinearSVC(C=1), 2),  # 使用参数化测试，测试LinearSVC分类器和cv参数为2的情况
        pytest.param(LinearSVC(C=1), "prefit"),  # 使用参数化测试，测试LinearSVC分类器和cv参数为"prefit"的情况
    ],
)
def test_calibration_attributes(clf, cv):
    # 检查`n_features_in_`和`classes_`属性是否正确创建
    X, y = make_classification(n_samples=10, n_features=5, n_classes=2, random_state=7)
    if cv == "prefit":
        clf = clf.fit(X, y)
    calib_clf = CalibratedClassifierCV(clf, cv=cv)
    calib_clf.fit(X, y)

    if cv == "prefit":
        assert_array_equal(calib_clf.classes_, clf.classes_)  # 断言校准后分类器的类别与原分类器的类别相同
        assert calib_clf.n_features_in_ == clf.n_features_in_  # 断言校准后分类器的输入特征数与原分类器相同
    else:
        classes = LabelEncoder().fit(y).classes_
        assert_array_equal(calib_clf.classes_, classes)  # 断言校准后分类器的类别与标签编码后的类别相同
        assert calib_clf.n_features_in_ == X.shape[1]  # 断言校准后分类器的输入特征数与X的列数相同


def test_calibration_inconsistent_prefit_n_features_in():
    # 检查预拟合基础估计器的`n_features_in_`是否与训练集一致
    X, y = make_classification(n_samples=10, n_features=5, n_classes=2, random_state=7)
    clf = LinearSVC(C=1).fit(X, y)
    calib_clf = CalibratedClassifierCV(clf, cv="prefit")

    msg = "X has 3 features, but LinearSVC is expecting 5 features as input."
    with pytest.raises(ValueError, match=msg):
        calib_clf.fit(X[:, :3], y)  # 断言拟合时会引发预期的值错误


def test_calibration_votingclassifier():
    # 检查`CalibratedClassifier`在`VotingClassifier`中的工作情况
    # `VotingClassifier`的`predict_proba`方法通过属性动态定义，仅在voting="soft"时有效
    X, y = make_classification(n_samples=10, n_features=5, n_classes=2, random_state=7)
    vote = VotingClassifier(
        estimators=[("lr" + str(i), LogisticRegression()) for i in range(3)],
        voting="soft",
    )
    vote.fit(X, y)

    calib_clf = CalibratedClassifierCV(estimator=vote, cv="prefit")
    # 烟雾测试：不应引发错误
    calib_clf.fit(X, y)


@pytest.fixture(scope="module")
def iris_data():
    return load_iris(return_X_y=True)


@pytest.fixture(scope="module")
def iris_data_binary(iris_data):
    X, y = iris_data
    return X[y < 2], y[y < 2]


@pytest.mark.parametrize("n_bins", [5, 10])
@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_calibration_display_compute(pyplot, iris_data_binary, n_bins, strategy):
    # 确保`CalibrationDisplay.from_predictions`和`calibration_curve`
    # 计算相同的结果。还检查`CalibrationDisplay`对象的属性。
    X, y = iris_data_binary

    lr = LogisticRegression().fit(X, y)

    viz = CalibrationDisplay.from_estimator(
        lr, X, y, n_bins=n_bins, strategy=strategy, alpha=0.8
    )

    y_prob = lr.predict_proba(X)[:, 1]
    prob_true, prob_pred = calibration_curve(
        y, y_prob, n_bins=n_bins, strategy=strategy
    )

    assert_allclose(viz.prob_true, prob_true)  # 断言校准显示对象的真实概率与实际计算的真实概率相近
    assert_allclose(viz.prob_pred, prob_pred)  # 断言校准显示对象的预测概率与实际计算的预测概率相近
    assert_allclose(viz.y_prob, y_prob)  # 断言校准显示对象的y_prob与实际预测概率相近
    # 断言检查可视化对象的估计器名称是否为 "LogisticRegression"
    assert viz.estimator_name == "LogisticRegression"

    # 导入 matplotlib 作为 mpl，这里不会因为 pyplot 的装置失败而发生异常
    import matplotlib as mpl  # noqa

    # 断言检查可视化对象的线条是否为 mpl 中的 Line2D 对象
    assert isinstance(viz.line_, mpl.lines.Line2D)
    # 断言检查线条的透明度是否为 0.8
    assert viz.line_.get_alpha() == 0.8
    # 断言检查可视化对象的坐标轴是否为 mpl 中的 Axes 对象
    assert isinstance(viz.ax_, mpl.axes.Axes)
    # 断言检查可视化对象的图形是否为 mpl 中的 Figure 对象
    assert isinstance(viz.figure_, mpl.figure.Figure)

    # 断言检查坐标轴的 x 轴标签是否为指定值
    assert viz.ax_.get_xlabel() == "Mean predicted probability (Positive class: 1)"
    # 断言检查坐标轴的 y 轴标签是否为指定值
    assert viz.ax_.get_ylabel() == "Fraction of positives (Positive class: 1)"

    # 期望的图例标签列表
    expected_legend_labels = ["LogisticRegression", "Perfectly calibrated"]
    # 获取当前图中的图例文本标签
    legend_labels = viz.ax_.get_legend().get_texts()
    # 断言检查图例标签列表的长度是否与期望的长度相同
    assert len(legend_labels) == len(expected_legend_labels)
    # 遍历每个图例标签，断言每个标签的文本是否在期望的图例标签列表中
    for labels in legend_labels:
        assert labels.get_text() in expected_legend_labels
python
def test_plot_calibration_curve_pipeline(pyplot, iris_data_binary):
    # 确保 CalibrationDisplay.from_estimator 支持管道模型
    X, y = iris_data_binary  # 从 iris_data_binary 中获取特征 X 和目标 y
    clf = make_pipeline(StandardScaler(), LogisticRegression())  # 创建包含标准化和逻辑回归的管道模型
    clf.fit(X, y)  # 使用 X 和 y 训练模型
    viz = CalibrationDisplay.from_estimator(clf, X, y)  # 使用训练好的模型创建 CalibrationDisplay 对象

    expected_legend_labels = [viz.estimator_name, "Perfectly calibrated"]  # 预期的图例标签
    legend_labels = viz.ax_.get_legend().get_texts()  # 获取当前图的图例文本
    assert len(legend_labels) == len(expected_legend_labels)  # 断言图例标签数量与预期相同
    for labels in legend_labels:
        assert labels.get_text() in expected_legend_labels  # 断言每个图例文本在预期标签列表中


@pytest.mark.parametrize(
    "name, expected_label", [(None, "_line1"), ("my_est", "my_est")]
)
def test_calibration_display_default_labels(pyplot, name, expected_label):
    prob_true = np.array([0, 1, 1, 0])  # 真实概率标签
    prob_pred = np.array([0.2, 0.8, 0.8, 0.4])  # 预测概率标签
    y_prob = np.array([])  # 空的预测概率标签数组

    viz = CalibrationDisplay(prob_true, prob_pred, y_prob, estimator_name=name)  # 创建 CalibrationDisplay 对象
    viz.plot()  # 绘制校准曲线图

    expected_legend_labels = [] if name is None else [name]  # 预期的图例标签
    expected_legend_labels.append("Perfectly calibrated")  # 添加 "Perfectly calibrated" 到预期标签列表
    legend_labels = viz.ax_.get_legend().get_texts()  # 获取当前图的图例文本
    assert len(legend_labels) == len(expected_legend_labels)  # 断言图例标签数量与预期相同
    for labels in legend_labels:
        assert labels.get_text() in expected_legend_labels  # 断言每个图例文本在预期标签列表中


def test_calibration_display_label_class_plot(pyplot):
    # 检查在实例化 CalibrationDisplay 类后调用 plot 方法时 self.estimator_name 是否正确设置
    prob_true = np.array([0, 1, 1, 0])  # 真实概率标签
    prob_pred = np.array([0.2, 0.8, 0.8, 0.4])  # 预测概率标签
    y_prob = np.array([])  # 空的预测概率标签数组

    name = "name one"  # 设定一个名称
    viz = CalibrationDisplay(prob_true, prob_pred, y_prob, estimator_name=name)  # 创建 CalibrationDisplay 对象
    assert viz.estimator_name == name  # 断言对象的 estimator_name 属性与设置的名称一致
    name = "name two"  # 更新名称为 "name two"
    viz.plot(name=name)  # 绘制校准曲线图，指定新的名称

    expected_legend_labels = [name, "Perfectly calibrated"]  # 预期的图例标签
    legend_labels = viz.ax_.get_legend().get_texts()  # 获取当前图的图例文本
    assert len(legend_labels) == len(expected_legend_labels)  # 断言图例标签数量与预期相同
    for labels in legend_labels:
        assert labels.get_text() in expected_legend_labels  # 断言每个图例文本在预期标签列表中


@pytest.mark.parametrize("constructor_name", ["from_estimator", "from_predictions"])
def test_calibration_display_name_multiple_calls(
    constructor_name, pyplot, iris_data_binary
):
    # 检查在多次调用 CalibrationDisplay.viz.plot() 时，传入的名称是否正确使用
    X, y = iris_data_binary  # 从 iris_data_binary 中获取特征 X 和目标 y
    clf_name = "my hand-crafted name"  # 设定一个自定义名称
    clf = LogisticRegression().fit(X, y)  # 使用逻辑回归模型拟合数据
    y_prob = clf.predict_proba(X)[:, 1]  # 预测类别为 1 的概率

    constructor = getattr(CalibrationDisplay, constructor_name)  # 获取指定的构造函数
    params = (clf, X, y) if constructor_name == "from_estimator" else (y, y_prob)  # 根据构造函数名选择参数

    viz = constructor(*params, name=clf_name)  # 使用指定的构造函数和参数创建 CalibrationDisplay 对象
    assert viz.estimator_name == clf_name  # 断言对象的 estimator_name 属性与设置的名称一致
    pyplot.close("all")  # 关闭所有绘图
    viz.plot()  # 绘制校准曲线图

    expected_legend_labels = [clf_name, "Perfectly calibrated"]  # 预期的图例标签
    legend_labels = viz.ax_.get_legend().get_texts()  # 获取当前图的图例文本
    assert len(legend_labels) == len(expected_legend_labels)  # 断言图例标签数量与预期相同
    for labels in legend_labels:
        assert labels.get_text() in expected_legend_labels  # 断言每个图例文本在预期标签列表中
    # 确保图例标签列表的长度与预期图例标签列表的长度相等
    assert len(legend_labels) == len(expected_legend_labels)
    
    # 遍历图例标签列表中的每个标签对象
    for labels in legend_labels:
        # 确保每个标签对象的文本内容存在于预期的图例标签列表中
        assert labels.get_text() in expected_legend_labels
    
    # 关闭所有 pyplot 绘图窗口
    pyplot.close("all")
    
    # 将分类器名称设置为"another_name"
    clf_name = "another_name"
    
    # 使用可视化模块 viz 绘制指定名称的图形
    viz.plot(name=clf_name)
    
    # 再次确保图例标签列表的长度与预期图例标签列表的长度相等
    assert len(legend_labels) == len(expected_legend_labels)
    
    # 再次遍历图例标签列表中的每个标签对象
    for labels in legend_labels:
        # 确保每个标签对象的文本内容存在于预期的图例标签列表中
        assert labels.get_text() in expected_legend_labels
def test_calibration_display_ref_line(pyplot, iris_data_binary):
    # 检查 `ref_line` 只出现一次
    X, y = iris_data_binary
    lr = LogisticRegression().fit(X, y)  # 使用逻辑回归模型拟合数据
    dt = DecisionTreeClassifier().fit(X, y)  # 使用决策树模型拟合数据

    viz = CalibrationDisplay.from_estimator(lr, X, y)  # 创建逻辑回归模型的校准显示对象
    viz2 = CalibrationDisplay.from_estimator(dt, X, y, ax=viz.ax_)  # 创建决策树模型的校准显示对象，并共享坐标轴

    labels = viz2.ax_.get_legend_handles_labels()[1]  # 获取图例标签
    assert labels.count("Perfectly calibrated") == 1  # 断言"Perfectly calibrated"标签出现一次


@pytest.mark.parametrize("dtype_y_str", [str, object])
def test_calibration_curve_pos_label_error_str(dtype_y_str):
    """检查当目标为字符串类型时，未指定 `pos_label` 时的错误消息。"""
    rng = np.random.RandomState(42)
    y1 = np.array(["spam"] * 3 + ["eggs"] * 2, dtype=dtype_y_str)  # 创建字符串类型的目标数组
    y2 = rng.randint(0, 2, size=y1.size)

    err_msg = (
        "y_true takes value in {'eggs', 'spam'} and pos_label is not "
        "specified: either make y_true take value in {0, 1} or {-1, 1} or "
        "pass pos_label explicitly"
    )
    with pytest.raises(ValueError, match=err_msg):
        calibration_curve(y1, y2)  # 调用校准曲线函数，预期抛出值错误与指定的错误消息匹配的异常


@pytest.mark.parametrize("dtype_y_str", [str, object])
def test_calibration_curve_pos_label(dtype_y_str):
    """检查在显式传递 `pos_label` 时的行为。"""
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])  # 创建二进制目标数组
    classes = np.array(["spam", "egg"], dtype=dtype_y_str)
    y_true_str = classes[y_true]  # 使用字符串类型的类标签生成字符串类型的目标数组
    y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9, 1.0])

    # 默认情况
    prob_true, _ = calibration_curve(y_true, y_pred, n_bins=4)  # 计算默认情况下的校准曲线
    assert_allclose(prob_true, [0, 0.5, 1, 1])  # 断言计算结果与预期值接近
    # 如果 `y_true` 包含字符串，则需要显式指定 `pos_label`
    prob_true, _ = calibration_curve(y_true_str, y_pred, n_bins=4, pos_label="egg")  # 计算指定 `pos_label` 的校准曲线
    assert_allclose(prob_true, [0, 0.5, 1, 1])  # 断言计算结果与预期值接近

    prob_true, _ = calibration_curve(y_true, 1 - y_pred, n_bins=4, pos_label=0)  # 计算反向预测概率的校准曲线
    assert_allclose(prob_true, [0, 0, 0.5, 1])  # 断言计算结果与预期值接近
    prob_true, _ = calibration_curve(y_true_str, 1 - y_pred, n_bins=4, pos_label="spam")  # 计算指定 `pos_label` 的反向预测概率校准曲线
    assert_allclose(prob_true, [0, 0, 0.5, 1])  # 断言计算结果与预期值接近


@pytest.mark.parametrize("pos_label, expected_pos_label", [(None, 1), (0, 0), (1, 1)])
def test_calibration_display_pos_label(
    pyplot, iris_data_binary, pos_label, expected_pos_label
):
    """检查 `CalibrationDisplay` 中 `pos_label` 的行为。"""
    X, y = iris_data_binary

    lr = LogisticRegression().fit(X, y)  # 使用逻辑回归模型拟合数据
    viz = CalibrationDisplay.from_estimator(lr, X, y, pos_label=pos_label)  # 创建带有指定 `pos_label` 的校准显示对象

    y_prob = lr.predict_proba(X)[:, expected_pos_label]  # 获取预测概率
    prob_true, prob_pred = calibration_curve(y, y_prob, pos_label=pos_label)  # 计算校准曲线

    assert_allclose(viz.prob_true, prob_true)  # 断言校准显示对象的真实概率与预期值接近
    assert_allclose(viz.prob_pred, prob_pred)  # 断言校准显示对象的预测概率与预期值接近
    assert_allclose(viz.y_prob, y_prob)  # 断言校准显示对象的预测概率与预期值接近

    assert (
        viz.ax_.get_xlabel()
        == f"Mean predicted probability (Positive class: {expected_pos_label})"
    )  # 断言 x 轴标签内容
    assert (
        viz.ax_.get_ylabel()
        == f"Fraction of positives (Positive class: {expected_pos_label})"
    )  # 断言 y 轴标签内容
    # 预期的图例标签，包括线性回归对象的类名和"Perfectly calibrated"
    expected_legend_labels = [lr.__class__.__name__, "Perfectly calibrated"]
    
    # 获取当前图表(ax_)的图例对象，并提取其中的文本标签
    legend_labels = viz.ax_.get_legend().get_texts()
    
    # 断言：确保图例标签的数量与预期的标签数量相同
    assert len(legend_labels) == len(expected_legend_labels)
    
    # 遍历每个图例标签
    for label in legend_labels:
        # 断言：确保每个标签的文本内容存在于预期的图例标签列表中
        assert label.get_text() in expected_legend_labels
@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
@pytest.mark.parametrize("ensemble", [True, False])
def test_calibrated_classifier_cv_double_sample_weights_equivalence(method, ensemble):
    """Check that passing repeating twice the dataset `X` is equivalent to
    passing a `sample_weight` with a factor 2."""
    X, y = load_iris(return_X_y=True)
    # 数据标准化，避免收敛问题
    X = StandardScaler().fit_transform(X)
    # 只使用前100个样本和对应标签
    X, y = X[:100], y[:100]
    sample_weight = np.ones_like(y) * 2

    # 将数据重复两次，以使二折交叉验证等同于使用原始数据集的两倍样本权重
    X_twice = np.zeros((X.shape[0] * 2, X.shape[1]), dtype=X.dtype)
    X_twice[::2, :] = X
    X_twice[1::2, :] = X
    y_twice = np.zeros(y.shape[0] * 2, dtype=y.dtype)
    y_twice[::2] = y
    y_twice[1::2] = y

    estimator = LogisticRegression()
    # 创建无权重的校准分类器CV对象
    calibrated_clf_without_weights = CalibratedClassifierCV(
        estimator,
        method=method,
        ensemble=ensemble,
        cv=2,
    )
    # 克隆无权重的校准分类器CV对象，以创建有权重的对象
    calibrated_clf_with_weights = clone(calibrated_clf_without_weights)

    # 使用样本权重拟合有权重的校准分类器CV对象
    calibrated_clf_with_weights.fit(X, y, sample_weight=sample_weight)
    # 使用重复数据拟合无权重的校准分类器CV对象
    calibrated_clf_without_weights.fit(X_twice, y_twice)

    # 检查已拟合的估算器是否具有相同的系数
    for est_with_weights, est_without_weights in zip(
        calibrated_clf_with_weights.calibrated_classifiers_,
        calibrated_clf_without_weights.calibrated_classifiers_,
    ):
        assert_allclose(
            est_with_weights.estimator.coef_,
            est_without_weights.estimator.coef_,
        )

    # 检查预测是否相同
    y_pred_with_weights = calibrated_clf_with_weights.predict_proba(X)
    y_pred_without_weights = calibrated_clf_without_weights.predict_proba(X)

    assert_allclose(y_pred_with_weights, y_pred_without_weights)


@pytest.mark.parametrize("fit_params_type", ["list", "array"])
def test_calibration_with_fit_params(fit_params_type, data):
    """Tests that fit_params are passed to the underlying base estimator.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/12384
    """
    X, y = data
    # 根据类型转换fit_params的值
    fit_params = {
        "a": _convert_container(y, fit_params_type),
        "b": _convert_container(y, fit_params_type),
    }

    clf = CheckingClassifier(expected_fit_params=["a", "b"])
    pc_clf = CalibratedClassifierCV(clf)

    pc_clf.fit(X, y, **fit_params)


@pytest.mark.parametrize(
    "sample_weight",
    [
        [1.0] * N_SAMPLES,
        np.ones(N_SAMPLES),
    ],
)
def test_calibration_with_sample_weight_estimator(sample_weight, data):
    """Tests that sample_weight is passed to the underlying base
    estimator.
    """
    X, y = data
    clf = CheckingClassifier(expected_sample_weight=True)
    pc_clf = CalibratedClassifierCV(clf)

    pc_clf.fit(X, y, sample_weight=sample_weight)
# 检查即使估计器不支持样本权重，使用样本权重进行拟合仍然有效。
# 由于样本权重未传递给估计器，应该会有警告。
def test_calibration_without_sample_weight_estimator(data):
    X, y = data
    # 创建与 y 相同形状的全1数组作为样本权重
    sample_weight = np.ones_like(y)

    # 定义一个不支持样本权重的分类器类
    class ClfWithoutSampleWeight(CheckingClassifier):
        def fit(self, X, y, **fit_params):
            # 检查 fit_params 中不包含样本权重参数
            assert "sample_weight" not in fit_params
            return super().fit(X, y, **fit_params)

    # 实例化上述分类器类
    clf = ClfWithoutSampleWeight()
    # 创建校准分类器对象，用于校准上述分类器
    pc_clf = CalibratedClassifierCV(clf)

    # 使用 pytest 来检测是否会有 UserWarning 警告
    with pytest.warns(UserWarning):
        # 使用样本权重拟合校准分类器
        pc_clf.fit(X, y, sample_weight=sample_weight)


# 参数化测试函数，测试不同的方法和集成参数组合
@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
@pytest.mark.parametrize("ensemble", [True, False])
def test_calibrated_classifier_cv_zeros_sample_weights_equivalence(method, ensemble):
    """检查从数据集 `X` 中去除一些样本与传递样本权重为0等效的情况。"""
    # 加载鸢尾花数据集，并返回特征矩阵 X 和目标向量 y
    X, y = load_iris(return_X_y=True)
    # 对数据进行标准化，避免收敛问题
    X = StandardScaler().fit_transform(X)
    # 只使用两类数据，并选择样本以使2折交叉验证分割等效于样本权重为0
    X = np.vstack((X[:40], X[50:90]))
    y = np.hstack((y[:40], y[50:90]))
    # 创建与 y 相同形状的全0数组作为样本权重
    sample_weight = np.zeros_like(y)
    sample_weight[::2] = 1  # 每隔一个样本设置样本权重为1

    # 创建逻辑回归估计器
    estimator = LogisticRegression()
    # 创建校准分类器对象，用于校准上述逻辑回归估计器
    calibrated_clf_without_weights = CalibratedClassifierCV(
        estimator,
        method=method,
        ensemble=ensemble,
        cv=2,
    )
    # 克隆无样本权重的校准分类器对象
    calibrated_clf_with_weights = clone(calibrated_clf_without_weights)

    # 使用样本权重拟合有样本权重的校准分类器
    calibrated_clf_with_weights.fit(X, y, sample_weight=sample_weight)
    # 使用每隔一个样本拟合无样本权重的校准分类器
    calibrated_clf_without_weights.fit(X[::2], y[::2])

    # 检查底层拟合的估计器是否具有相同的系数
    for est_with_weights, est_without_weights in zip(
        calibrated_clf_with_weights.calibrated_classifiers_,
        calibrated_clf_without_weights.calibrated_classifiers_,
    ):
        assert_allclose(
            est_with_weights.estimator.coef_,
            est_without_weights.estimator.coef_,
        )

    # 检查预测结果是否相同
    y_pred_with_weights = calibrated_clf_with_weights.predict_proba(X)
    y_pred_without_weights = calibrated_clf_without_weights.predict_proba(X)

    assert_allclose(y_pred_with_weights, y_pred_without_weights)


def test_calibration_with_non_sample_aligned_fit_param(data):
    """检查 CalibratedClassifierCV 是否不强制适合参数对样本的对齐性。"""

    # 定义一个测试用的逻辑回归分类器类
    class TestClassifier(LogisticRegression):
        def fit(self, X, y, sample_weight=None, fit_param=None):
            # 检查 fit_param 参数不为空
            assert fit_param is not None
            return super().fit(X, y, sample_weight=sample_weight)

    # 使用测试用的逻辑回归分类器实例化 CalibratedClassifierCV 对象
    CalibratedClassifierCV(estimator=TestClassifier()).fit(
        *data, fit_param=np.ones(len(data[1]) + 1)
    )
def test_calibrated_classifier_cv_works_with_large_confidence_scores(
    global_random_seed,
):
    """Test that :class:`CalibratedClassifierCV` works with large confidence
    scores when using the `sigmoid` method, particularly with the
    :class:`SGDClassifier`.

    Non-regression test for issue #26766.
    """
    # 设置正例概率
    prob = 0.67
    # 样本数量
    n = 1000
    # 使用全局随机种子生成正态分布随机数作为随机噪声
    random_noise = np.random.default_rng(global_random_seed).normal(size=n)

    # 创建分类标签 y，其中约有 prob 比例的数据为正例
    y = np.array([1] * int(n * prob) + [0] * (n - int(n * prob)))
    # 构建特征矩阵 X，其中正例特征值偏向于较大值，并加入随机噪声
    X = 1e5 * y.reshape((-1, 1)) + random_noise

    # 验证 SGDClassifier 的决策函数在考虑的数据集下是否产生较大的预测值
    cv = check_cv(cv=None, y=y, classifier=True)
    indices = cv.split(X, y)
    for train, test in indices:
        X_train, y_train = X[train], y[train]
        X_test = X[test]
        sgd_clf = SGDClassifier(loss="squared_hinge", random_state=global_random_seed)
        sgd_clf.fit(X_train, y_train)
        predictions = sgd_clf.decision_function(X_test)
        assert (predictions > 1e4).any()

    # 比较使用 sigmoid 方法的 CalibratedClassifierCV 和使用 isotonic 方法的 CalibratedClassifierCV
    # 使用 isotonic 方法进行比较是因为它在数值上更稳定
    clf_sigmoid = CalibratedClassifierCV(
        SGDClassifier(loss="squared_hinge", random_state=global_random_seed),
        method="sigmoid",
    )
    score_sigmoid = cross_val_score(clf_sigmoid, X, y, scoring="roc_auc")

    # 使用 isotonic 方法进行比较是因为它在数值上更稳定
    clf_isotonic = CalibratedClassifierCV(
        SGDClassifier(loss="squared_hinge", random_state=global_random_seed),
        method="isotonic",
    )
    score_isotonic = cross_val_score(clf_isotonic, X, y, scoring="roc_auc")

    # AUC 分数应该相同，因为它在严格单调条件下是不变的
    assert_allclose(score_sigmoid, score_isotonic)


def test_sigmoid_calibration_max_abs_prediction_threshold(global_random_seed):
    random_state = np.random.RandomState(seed=global_random_seed)
    n = 100
    y = random_state.randint(0, 2, size=n)

    # 检查对于预测值范围在 -2 到 2 之间的较小预测值，阈值值不会对结果产生影响
    predictions_small = random_state.uniform(low=-2, high=2, size=100)

    # 使用低于预测值的最大绝对值的阈值，通过 max(abs(predictions_small)) 进行内部重新缩放
    threshold_1 = 0.1
    a1, b1 = _sigmoid_calibration(
        predictions=predictions_small,
        y=y,
        max_abs_prediction_threshold=threshold_1,
    )

    # 使用较大的阈值禁用重新缩放
    threshold_2 = 10
    a2, b2 = _sigmoid_calibration(
        predictions=predictions_small,
        y=y,
        max_abs_prediction_threshold=threshold_2,
    )

    # 使用默认的阈值 30 也禁用了缩放
    # 调用 _sigmoid_calibration 函数，对 predictions_small 进行逻辑回归校准，得到 a3 和 b3 两个结果
    a3, b3 = _sigmoid_calibration(
        predictions=predictions_small,
        y=y,
    )

    # 设置容差值，用于下面的数值比较操作，控制比较的精度
    atol = 1e-6
    
    # 使用 assert_allclose 函数比较 a1 和 a2 两个值是否在容差范围内相等
    assert_allclose(a1, a2, atol=atol)
    
    # 使用 assert_allclose 函数比较 a2 和 a3 两个值是否在容差范围内相等
    assert_allclose(a2, a3, atol=atol)
    
    # 使用 assert_allclose 函数比较 b1 和 b2 两个值是否在容差范围内相等
    assert_allclose(b1, b2, atol=atol)
    
    # 使用 assert_allclose 函数比较 b2 和 b3 两个值是否在容差范围内相等
    assert_allclose(b2, b3, atol=atol)
# 检查 CalibratedClassifierCV 在 float32 预测概率上的工作是否正常。
# 这是针对 gh-28245 的非回归测试。

def test_float32_predict_proba(data):
    """Check that CalibratedClassifierCV works with float32 predict proba.

    Non-regression test for gh-28245.
    """

    # 创建一个自定义的 DummyClassifier32 类，继承自 DummyClassifier
    class DummyClassifer32(DummyClassifier):
        # 重写 predict_proba 方法，将其返回结果转换为 np.float32 类型
        def predict_proba(self, X):
            return super().predict_proba(X).astype(np.float32)

    # 实例化 DummyClassifer32 类作为模型
    model = DummyClassifer32()
    # 创建 CalibratedClassifierCV 对象，传入模型
    calibrator = CalibratedClassifierCV(model)
    # 拟合模型和校准器，验证不会引发错误
    calibrator.fit(*data)


# 检查 CalibratedClassifierCV 在字符串目标上的工作是否正常。
# 这是针对 issue #28841 的非回归测试。

def test_error_less_class_samples_than_folds():
    """Check that CalibratedClassifierCV works with string targets.

    non-regression test for issue #28841.
    """
    
    # 生成随机数据 X，形状为 (20, 3)
    X = np.random.normal(size=(20, 3))
    # 生成目标标签 y，包含 10 个 'a' 和 10 个 'b'
    y = ["a"] * 10 + ["b"] * 10

    # 创建 CalibratedClassifierCV 对象，设置交叉验证折数为 3
    CalibratedClassifierCV(cv=3).fit(X, y)
```