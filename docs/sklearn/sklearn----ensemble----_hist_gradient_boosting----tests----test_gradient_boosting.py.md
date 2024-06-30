# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_hist_gradient_boosting\tests\test_gradient_boosting.py`

```
import copyreg  # 导入 copyreg 模块，用于注册 pickle 模块的扩展类型
import io  # 导入 io 模块，提供了在内存中读写文本数据的工具
import pickle  # 导入 pickle 模块，用于对象的序列化和反序列化
import re  # 导入 re 模块，提供对正则表达式的支持
import warnings  # 导入 warnings 模块，用于警告控制

from unittest.mock import Mock  # 从 unittest.mock 模块中导入 Mock 类

import joblib  # 导入 joblib 库，用于高效处理 Python 对象持久化的工具
import numpy as np  # 导入 NumPy 库，用于科学计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from joblib.numpy_pickle import NumpyPickler  # 从 joblib.numpy_pickle 模块导入 NumpyPickler 类
from numpy.testing import assert_allclose, assert_array_equal  # 从 numpy.testing 模块导入用于数组测试的函数

import sklearn  # 导入 scikit-learn 库，用于机器学习
from sklearn._loss.loss import (  # 从 scikit-learn 库的 _loss.loss 模块中导入指定损失函数类
    AbsoluteError,
    HalfBinomialLoss,
    HalfSquaredError,
    PinballLoss,
)
from sklearn.base import BaseEstimator, TransformerMixin, clone, is_regressor  # 从 scikit-learn 库的 base 模块中导入基类和相关函数
from sklearn.compose import make_column_transformer  # 从 scikit-learn 库的 compose 模块导入 make_column_transformer 函数
from sklearn.datasets import make_classification, make_low_rank_matrix, make_regression  # 从 scikit-learn 库的 datasets 模块导入数据生成函数
from sklearn.dummy import DummyRegressor  # 从 scikit-learn 库的 dummy 模块导入 DummyRegressor 类
from sklearn.ensemble import (  # 从 scikit-learn 库的 ensemble 模块导入集成学习方法
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper  # 从 scikit-learn 库的 ensemble._hist_gradient_boosting.binning 模块导入 _BinMapper 类
from sklearn.ensemble._hist_gradient_boosting.common import G_H_DTYPE  # 从 scikit-learn 库的 ensemble._hist_gradient_boosting.common 模块导入 G_H_DTYPE 常量
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower  # 从 scikit-learn 库的 ensemble._hist_gradient_boosting.grower 模块导入 TreeGrower 类
from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor  # 从 scikit-learn 库的 ensemble._hist_gradient_boosting.predictor 模块导入 TreePredictor 类
from sklearn.exceptions import NotFittedError  # 从 scikit-learn 库的 exceptions 模块导入 NotFittedError 异常类
from sklearn.metrics import get_scorer, mean_gamma_deviance, mean_poisson_deviance  # 从 scikit-learn 库的 metrics 模块导入评估函数
from sklearn.model_selection import cross_val_score, train_test_split  # 从 scikit-learn 库的 model_selection 模块导入交叉验证和数据集划分函数
from sklearn.pipeline import make_pipeline  # 从 scikit-learn 库的 pipeline 模块导入 make_pipeline 函数
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, OneHotEncoder  # 从 scikit-learn 库的 preprocessing 模块导入数据预处理类
from sklearn.utils import shuffle  # 从 scikit-learn 库的 utils 模块导入 shuffle 函数
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads  # 从 scikit-learn 库的 utils._openmp_helpers 模块导入 _openmp_effective_n_threads 函数
from sklearn.utils._testing import _convert_container  # 从 scikit-learn 库的 utils._testing 模块导入 _convert_container 函数
from sklearn.utils.fixes import _IS_32BIT  # 从 scikit-learn 库的 utils.fixes 模块导入 _IS_32BIT 常量

n_threads = _openmp_effective_n_threads()  # 获取有效的 OpenMP 线程数

X_classification, y_classification = make_classification(random_state=0)  # 生成分类任务数据集
X_regression, y_regression = make_regression(random_state=0)  # 生成回归任务数据集
X_multi_classification, y_multi_classification = make_classification(
    n_classes=3, n_informative=3, random_state=0
)  # 生成多类分类任务数据集


def _make_dumb_dataset(n_samples):
    """Make a dumb dataset to test early stopping."""
    rng = np.random.RandomState(42)  # 创建随机数生成器对象
    X_dumb = rng.randn(n_samples, 1)  # 生成服从标准正态分布的随机样本
    y_dumb = (X_dumb[:, 0] > 0).astype("int64")  # 根据 X_dumb 的第一列生成二分类标签
    return X_dumb, y_dumb  # 返回生成的数据集


@pytest.mark.parametrize(
    "GradientBoosting, X, y",
    [
        (HistGradientBoostingClassifier, X_classification, y_classification),  # 参数化测试：分类梯度提升树模型和分类数据集
        (HistGradientBoostingRegressor, X_regression, y_regression),  # 参数化测试：回归梯度提升树模型和回归数据集
    ],
)
@pytest.mark.parametrize(
    "params, err_msg",
    # 以下是一个包含多个元组的列表，每个元组包含一个字典和一个错误消息字符串
    [
        (
            # 第一个字典包含键 "interaction_cst"，其值是包含数字 0 和 1 的列表
            {"interaction_cst": [0, 1]},
            # 对应的错误消息，指出交互约束必须是元组或列表的序列
            "Interaction constraints must be a sequence of tuples or lists",
        ),
        (
            # 第二个字典包含键 "interaction_cst"，其值是包含集合 {0, 9999} 的列表
            {"interaction_cst": [{0, 9999}]},
            # 对应的错误消息，指出交互约束必须是整数索引，范围在 [0, n_features - 1] 内
            r"Interaction constraints must consist of integer indices in \[0,"
            r" n_features - 1\] = \[.*\], specifying the position of features,",
        ),
        (
            # 第三个字典包含键 "interaction_cst"，其值是包含集合 {-1, 0} 的列表
            {"interaction_cst": [{-1, 0}]},
            # 对应的错误消息，指出交互约束必须是整数索引，范围在 [0, n_features - 1] 内
            r"Interaction constraints must consist of integer indices in \[0,"
            r" n_features - 1\] = \[.*\], specifying the position of features,",
        ),
        (
            # 第四个字典包含键 "interaction_cst"，其值是包含浮点数 0.5 的列表
            {"interaction_cst": [{0.5}]},
            # 对应的错误消息，指出交互约束必须是整数索引，范围在 [0, n_features - 1] 内
            r"Interaction constraints must consist of integer indices in \[0,"
            r" n_features - 1\] = \[.*\], specifying the position of features,",
        ),
    ],
@pytest.mark.parametrize(
    "GradientBoosting, X, y",

这是一个使用 pytest 的参数化测试装饰器，用来定义多个测试参数组合。


def test_init_parameters_validation(GradientBoosting, X, y, params, err_msg):

定义了一个测试函数 `test_init_parameters_validation`，用于验证初始化参数是否有效，如果参数无效则抛出 ValueError 异常。


    with pytest.raises(ValueError, match=err_msg):

使用 pytest 的 `pytest.raises` 上下文管理器来捕获 ValueError 异常，并匹配异常信息为 `err_msg`。


        GradientBoosting(**params).fit(X, y)

使用传入的参数 `params` 创建 `GradientBoosting` 实例，并调用 `fit` 方法拟合模型。


@pytest.mark.parametrize(
    "scoring, validation_fraction, early_stopping, n_iter_no_change, tol",
    [
        ("neg_mean_squared_error", 0.1, True, 5, 1e-7),  # use scorer
        ("neg_mean_squared_error", None, True, 5, 1e-1),  # use scorer on train
        (None, 0.1, True, 5, 1e-7),  # same with default scorer
        (None, None, True, 5, 1e-1),
        ("loss", 0.1, True, 5, 1e-7),  # use loss
        ("loss", None, True, 5, 1e-1),  # use loss on training data
        (None, None, False, 5, 0.0),  # no early stopping
    ],
)

定义了一个参数化测试函数 `test_early_stopping_regression` 的参数，每个元组代表一个测试用例，包含不同的参数组合。


def test_early_stopping_regression(
    scoring, validation_fraction, early_stopping, n_iter_no_change, tol
):

定义了一个测试函数 `test_early_stopping_regression`，用于测试回归模型中的早停功能。


    max_iter = 200

设定最大迭代次数为 200。


    X, y = make_regression(n_samples=50, random_state=0)

生成回归模型的测试数据 `X` 和 `y`。


    gb = HistGradientBoostingRegressor(

创建 `HistGradientBoostingRegressor` 的实例 `gb`，用于测试梯度提升回归器。


        verbose=1,  # just for coverage
        min_samples_leaf=5,  # easier to overfit fast
        scoring=scoring,
        tol=tol,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        max_iter=max_iter,
        n_iter_no_change=n_iter_no_change,
        random_state=0,
    )

设置梯度提升回归器 `gb` 的各种参数，包括是否输出详细信息、叶子最小样本数、评分指标、容忍度、早停、验证分数比例、最大迭代次数、迭代不变次数等。


    gb.fit(X, y)

使用测试数据 `X` 和 `y` 拟合 `gb` 模型。


    if early_stopping:
        assert n_iter_no_change <= gb.n_iter_ < max_iter
    else:
        assert gb.n_iter_ == max_iter

根据早停参数判断测试结果，如果开启了早停，则断言 `gb.n_iter_` 应在指定范围内，否则断言 `gb.n_iter_` 应等于最大迭代次数。


@pytest.mark.parametrize(
    "data",
    (
        make_classification(n_samples=30, random_state=0),
        make_classification(
            n_samples=30, n_classes=3, n_clusters_per_class=1, random_state=0
        ),
    ),
)

定义了一个参数化测试函数 `test_early_stopping_classification` 的 `data` 参数，生成了两种不同的分类数据集。


@pytest.mark.parametrize(
    "scoring, validation_fraction, early_stopping, n_iter_no_change, tol",
    [
        ("accuracy", 0.1, True, 5, 1e-7),  # use scorer
        ("accuracy", None, True, 5, 1e-1),  # use scorer on training data
        (None, 0.1, True, 5, 1e-7),  # same with default scorer
        (None, None, True, 5, 1e-1),
        ("loss", 0.1, True, 5, 1e-7),  # use loss
        ("loss", None, True, 5, 1e-1),  # use loss on training data
        (None, None, False, 5, 0.0),  # no early stopping
    ],
)

定义了一个参数化测试函数 `test_early_stopping_classification` 的参数，每个元组代表一个测试用例，包含不同的参数组合。


def test_early_stopping_classification(
    data, scoring, validation_fraction, early_stopping, n_iter_no_change, tol
):

定义了一个测试函数 `test_early_stopping_classification`，用于测试分类模型中的早停功能。


    max_iter = 50

设定最大迭代次数为 50。


    X, y = data

从参数 `data` 中获取分类模型的测试数据 `X` 和 `y`。


    gb = HistGradientBoostingClassifier(

创建 `HistGradientBoostingClassifier` 的实例 `gb`，用于测试梯度提升分类器。


        verbose=1,  # just for coverage
        min_samples_leaf=5,  # easier to overfit fast
        scoring=scoring,
        tol=tol,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        max_iter=max_iter,
        n_iter_no_change=n_iter_no_change,
        random_state=0,
    )

设置梯度提升分类器 `gb` 的各种参数，包括是否输出详细信息、叶子最小样本数、评分指标、容忍度、早停、验证分数比例、最大迭代次数、迭代不变次数等。


    gb.fit(X, y)

使用测试数据 `X` 和 `y` 拟合 `gb` 模型。


    if early_stopping is True:
        assert n_iter_no_change <= gb.n_iter_ < max_iter
    else:
        assert gb.n_iter_ == max_iter

根据早停参数判断测试结果，如果开启了早停，则断言 `gb.n_iter_` 应在指定范围内，否则断言 `gb.n_iter_` 应等于最大迭代次数。
    [
        # 使用 HistGradientBoostingClassifier 模型，处理包含 10000 条样本的简单数据集
        (HistGradientBoostingClassifier, *_make_dumb_dataset(10000)),
        # 使用 HistGradientBoostingClassifier 模型，处理包含 10001 条样本的简单数据集
        (HistGradientBoostingClassifier, *_make_dumb_dataset(10001)),
        # 使用 HistGradientBoostingRegressor 模型，处理包含 10000 条样本的简单数据集
        (HistGradientBoostingRegressor, *_make_dumb_dataset(10000)),
        # 使用 HistGradientBoostingRegressor 模型，处理包含 10001 条样本的简单数据集
        (HistGradientBoostingRegressor, *_make_dumb_dataset(10001)),
    ],
# 测试默认情况下的早停功能
def test_early_stopping_default(GradientBoosting, X, y):
    # 测试早停功能是否默认启用，仅当样本数超过10000时
    gb = GradientBoosting(max_iter=10, n_iter_no_change=2, tol=1e-1)
    gb.fit(X, y)
    if X.shape[0] > 10000:
        assert gb.n_iter_ < gb.max_iter  # 断言迭代次数小于最大迭代次数
    else:
        assert gb.n_iter_ == gb.max_iter  # 断言迭代次数等于最大迭代次数


@pytest.mark.parametrize(
    "scores, n_iter_no_change, tol, stopping",
    [
        ([], 1, 0.001, False),  # 迭代次数不足
        ([1, 1, 1], 5, 0.001, False),  # 迭代次数不足
        ([1, 1, 1, 1, 1], 5, 0.001, False),  # 迭代次数不足
        ([1, 2, 3, 4, 5, 6], 5, 0.001, False),  # 有显著改善
        ([1, 2, 3, 4, 5, 6], 5, 0.0, False),  # 有显著改善
        ([1, 2, 3, 4, 5, 6], 5, 0.999, False),  # 有显著改善
        ([1, 2, 3, 4, 5, 6], 5, 5 - 1e-5, False),  # 有显著改善
        ([1] * 6, 5, 0.0, True),  # 没有显著改善
        ([1] * 6, 5, 0.001, True),  # 没有显著改善
        ([1] * 6, 5, 5, True),  # 没有显著改善
    ],
)
def test_should_stop(scores, n_iter_no_change, tol, stopping):
    # 测试 `_should_stop` 方法的不同输入情况下的行为
    gbdt = HistGradientBoostingClassifier(n_iter_no_change=n_iter_no_change, tol=tol)
    assert gbdt._should_stop(scores) == stopping  # 断言 `_should_stop` 方法的输出符合预期


def test_absolute_error():
    # 仅用于覆盖率测试
    X, y = make_regression(n_samples=500, random_state=0)
    gbdt = HistGradientBoostingRegressor(loss="absolute_error", random_state=0)
    gbdt.fit(X, y)
    assert gbdt.score(X, y) > 0.9  # 断言预测准确度高于0.9


def test_absolute_error_sample_weight():
    # 非回归测试，用于检查问题 #19400
    # 确保在使用 absolute_error 损失函数并传递样本权重时不会抛出错误
    rng = np.random.RandomState(0)
    n_samples = 100
    X = rng.uniform(-1, 1, size=(n_samples, 2))
    y = rng.uniform(-1, 1, size=n_samples)
    sample_weight = rng.uniform(0, 1, size=n_samples)
    gbdt = HistGradientBoostingRegressor(loss="absolute_error")
    gbdt.fit(X, y, sample_weight=sample_weight)


@pytest.mark.parametrize("y", [([1.0, -2.0, 0.0]), ([0.0, 1.0, 2.0])])
def test_gamma_y_positive(y):
    # 测试当 y_i <= 0 时是否引发 ValueError
    err_msg = r"loss='gamma' requires strictly positive y."
    gbdt = HistGradientBoostingRegressor(loss="gamma", random_state=0)
    with pytest.raises(ValueError, match=err_msg):
        gbdt.fit(np.zeros(shape=(len(y), 1)), y)


def test_gamma():
    # 对于 Gamma 分布的目标变量，期望使用 Gamma 偏差（损失）训练的 HGBT 比使用其他任何损失函数的 HGBT 在外样本 Gamma 偏差上表现更好
    # 注意，平方误差可能会预测出负值，这对于 Gamma 偏差是无效的（np.inf）。具有对数链接的 Poisson HGBT 没有这个缺陷。
    # 使用种子随机数生成器创建一个 RandomState 对象，种子值为 42
    rng = np.random.RandomState(42)
    
    # 定义训练样本数量、测试样本数量和特征数量
    n_train, n_test, n_features = 500, 100, 20
    
    # 生成一个低秩矩阵 X，用于模拟数据集，特征数为 n_features
    X = make_low_rank_matrix(
        n_samples=n_train + n_test,
        n_features=n_features,
        random_state=rng,
    )
    
    # 创建一个系数向量 coef，其值从均匀分布 [-10, 20) 中随机选取，长度为 n_features
    coef = rng.uniform(low=-10, high=20, size=n_features)
    
    # 设定 Gamma 分布的参数 dispersion，用于后续的 Gamma 分布模型
    dispersion = 0.5
    
    # 生成 Gamma 分布的目标值 y，形状参数为 1/dispersion，尺度参数为 dispersion * exp(X @ coef)
    y = rng.gamma(shape=1 / dispersion, scale=dispersion * np.exp(X @ coef))
    
    # 将数据集 X 和目标值 y 划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, random_state=rng
    )
    
    # 创建一个 HistGradientBoostingRegressor 对象，用于 Gamma 损失函数的梯度提升树回归
    gbdt_gamma = HistGradientBoostingRegressor(loss="gamma", random_state=123)
    
    # 创建一个 HistGradientBoostingRegressor 对象，用于均方误差损失函数的梯度提升树回归
    gbdt_mse = HistGradientBoostingRegressor(loss="squared_error", random_state=123)
    
    # 创建一个 DummyRegressor 对象，采用均值策略作为预测基准
    dummy = DummyRegressor(strategy="mean")
    
    # 对三个模型（Gamma 损失 GBDT、均方误差 GBDT、Dummy 模型）进行训练
    for model in (gbdt_gamma, gbdt_mse, dummy):
        model.fit(X_train, y_train)
    
    # 对训练集和测试集分别进行预测并计算 Gamma deviance 损失值
    for X, y in [(X_train, y_train), (X_test, y_test)]:
        # 计算 Gamma 损失 GBDT 的平均 Gamma deviance
        loss_gbdt_gamma = mean_gamma_deviance(y, gbdt_gamma.predict(X))
        
        # 限制均方误差 GBDT 预测值至少为训练时最小的 y 值，以确保预测结果严格为正数
        loss_gbdt_mse = mean_gamma_deviance(
            y, np.maximum(np.min(y_train), gbdt_mse.predict(X))
        )
        
        # 计算 Dummy 模型的平均 Gamma deviance
        loss_dummy = mean_gamma_deviance(y, dummy.predict(X))
        
        # 断言 Gamma 损失 GBDT 的 Gamma deviance 损失应小于 Dummy 模型的损失
        assert loss_gbdt_gamma < loss_dummy
        
        # 断言 Gamma 损失 GBDT 的 Gamma deviance 损失应小于均方误差 GBDT 的损失
        assert loss_gbdt_gamma < loss_gbdt_mse
# 使用 pytest 的 parametrize 装饰器来执行多组参数化测试，每个测试以不同的 quantile 值运行一次
@pytest.mark.parametrize("quantile", [0.2, 0.5, 0.8])
def test_quantile_asymmetric_error(quantile):
    """Test quantile regression for asymmetric distributed targets."""
    # 设定样本数量
    n_samples = 10_000
    # 创建一个指定种子的随机数生成器
    rng = np.random.RandomState(42)
    # 构造特征 X，确保 X @ coef + intercept > 0
    X = np.concatenate(
        (
            np.abs(rng.randn(n_samples)[:, None]),  # 第一列为正态分布的绝对值
            -rng.randint(2, size=(n_samples, 1)),    # 第二列为0或-1的随机整数
        ),
        axis=1,
    )
    # 设置截距
    intercept = 1.23
    # 设置系数
    coef = np.array([0.5, -2])
    
    # 对于速率 lambda 的指数分布，例如 exp(-lambda * x)，在水平 q 处的分位数是：
    #   quantile(q) = - log(1 - q) / lambda
    #   尺度 scale = 1/lambda = -quantile(q) / log(1-q)
    # 生成目标变量 y，根据指定的 quantile 值和 X @ coef + intercept 计算尺度参数
    y = rng.exponential(
        scale=-(X @ coef + intercept) / np.log(1 - quantile), size=n_samples
    )
    
    # 使用 HistGradientBoostingRegressor 拟合模型，损失函数为 quantile 回归
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        max_iter=25,
        random_state=0,
        max_leaf_nodes=10,
    ).fit(X, y)
    
    # 断言预测的平均值大于目标变量 y 的比例等于 quantile
    assert_allclose(np.mean(model.predict(X) > y), quantile, rtol=1e-2)

    # 创建 PinballLoss 对象，用于计算损失值
    pinball_loss = PinballLoss(quantile=quantile)
    # 计算真实 quantile 损失值
    loss_true_quantile = pinball_loss(y, X @ coef + intercept)
    # 计算预测 quantile 损失值
    loss_pred_quantile = pinball_loss(y, model.predict(X))
    # 断言预测损失值小于或等于真实损失值，表明过拟合
    assert loss_pred_quantile <= loss_true_quantile


# 使用 pytest 的 parametrize 装饰器来执行多组参数化测试，每个测试以不同的 y 值运行一次
@pytest.mark.parametrize("y", [([1.0, -2.0, 0.0]), ([0.0, 0.0, 0.0])])
def test_poisson_y_positive(y):
    # 测试如果任一 y_i < 0 或 sum(y_i) <= 0，则引发 ValueError
    err_msg = r"loss='poisson' requires non-negative y and sum\(y\) > 0."
    gbdt = HistGradientBoostingRegressor(loss="poisson", random_state=0)
    with pytest.raises(ValueError, match=err_msg):
        gbdt.fit(np.zeros(shape=(len(y), 1)), y)


# 测试 Poisson 分布目标变量时，Poisson 损失函数应在 Poisson 偏差作为度量时表现更好
def test_poisson():
    # 创建随机数生成器
    rng = np.random.RandomState(42)
    # 设置训练集、测试集的样本数量和特征数量
    n_train, n_test, n_features = 500, 100, 100
    # 创建低秩矩阵作为特征矩阵 X
    X = make_low_rank_matrix(
        n_samples=n_train + n_test, n_features=n_features, random_state=rng
    )
    # 创建一个对数线性的 Poisson 模型，并调整 coef，因为它将被指数化
    coef = rng.uniform(low=-2, high=2, size=n_features) / np.max(X, axis=0)
    # 生成目标变量 y，服从 Poisson 分布
    y = rng.poisson(lam=np.exp(X @ coef))
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, random_state=rng
    )
    # 使用 HistGradientBoostingRegressor 拟合 Poisson 损失函数的模型
    gbdt_pois = HistGradientBoostingRegressor(loss="poisson", random_state=rng)
    gbdt_pois.fit(X_train, y_train)
    # 使用 HistGradientBoostingRegressor 拟合平方误差损失函数的模型
    gbdt_ls = HistGradientBoostingRegressor(loss="squared_error", random_state=rng)
    gbdt_ls.fit(X_train, y_train)
    # 创建一个 DummyRegressor 作为基准模型
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
    # 对训练集和测试集进行迭代，分别赋值给 X 和 y
    for X, y in [(X_train, y_train), (X_test, y_test)]:
        # 计算基于 Poisson 分布的平均偏差指标
        metric_pois = mean_poisson_deviance(y, gbdt_pois.predict(X))
        # 使用 GBDT 模型对 X 进行预测，并对预测结果进行截断，以避免产生非正预测值
        # 这是因为 squared_error 可能会产生非正的预测值
        metric_ls = mean_poisson_deviance(y, np.clip(gbdt_ls.predict(X), 1e-15, None))
        # 使用虚拟预测器对 X 进行预测，计算平均 Poisson 偏差指标
        metric_dummy = mean_poisson_deviance(y, dummy.predict(X))
        # 断言：基于 Poisson 模型的偏差指标应该小于基于最小二乘模型的偏差指标
        assert metric_pois < metric_ls
        # 断言：基于 Poisson 模型的偏差指标应该小于虚拟预测器的偏差指标
        assert metric_pois < metric_dummy
def test_binning_train_validation_are_separated():
    # Make sure training and validation data are binned separately.
    # See issue 13926

    rng = np.random.RandomState(0)  # 创建一个伪随机数生成器对象rng，种子为0
    validation_fraction = 0.2  # 设置验证集比例为0.2
    gb = HistGradientBoostingClassifier(
        early_stopping=True, validation_fraction=validation_fraction, random_state=rng
    )  # 创建一个梯度提升分类器对象gb，启用早停和设置验证集比例，使用rng作为随机数生成器
    gb.fit(X_classification, y_classification)  # 使用分类数据X_classification和标签y_classification训练gb
    mapper_training_data = gb._bin_mapper  # 获取训练数据的分箱映射器对象

    # Note that since the data is small there is no subsampling and the
    # random_state doesn't matter
    mapper_whole_data = _BinMapper(random_state=0)  # 创建一个全数据的分箱映射器对象，种子为0
    mapper_whole_data.fit(X_classification)  # 使用分类数据X_classification拟合全数据的分箱映射器

    n_samples = X_classification.shape[0]  # 获取样本数量
    assert np.all(
        mapper_training_data.n_bins_non_missing_
        == int((1 - validation_fraction) * n_samples)
    )  # 断言训练数据的非缺失值分箱数等于(1 - 验证集比例)乘以样本数量的整数部分
    assert np.all(
        mapper_training_data.n_bins_non_missing_
        != mapper_whole_data.n_bins_non_missing_
    )  # 断言训练数据的非缺失值分箱数不等于全数据的非缺失值分箱数


def test_missing_values_trivial():
    # sanity check for missing values support. With only one feature and
    # y == isnan(X), the gbdt is supposed to reach perfect accuracy on the
    # training set.

    n_samples = 100  # 设置样本数量为100
    n_features = 1  # 设置特征数量为1
    rng = np.random.RandomState(0)  # 创建一个伪随机数生成器对象rng，种子为0

    X = rng.normal(size=(n_samples, n_features))  # 生成服从正态分布的样本数据X
    mask = rng.binomial(1, 0.5, size=X.shape).astype(bool)  # 生成二项分布掩码数组mask，掩盖50%的数据为NaN
    X[mask] = np.nan  # 将mask中为True的位置设置为NaN
    y = mask.ravel()  # 将mask展平作为标签y
    gb = HistGradientBoostingClassifier()  # 创建一个梯度提升分类器对象gb
    gb.fit(X, y)  # 使用X和y训练gb

    assert gb.score(X, y) == pytest.approx(1)  # 断言gb在训练数据上的得分接近1


@pytest.mark.parametrize("problem", ("classification", "regression"))
@pytest.mark.parametrize(
    (
        "missing_proportion, expected_min_score_classification, "
        "expected_min_score_regression"
    ),
    [(0.1, 0.97, 0.89), (0.2, 0.93, 0.81), (0.5, 0.79, 0.52)],
)
def test_missing_values_resilience(
    problem,
    missing_proportion,
    expected_min_score_classification,
    expected_min_score_regression,
):
    # Make sure the estimators can deal with missing values and still yield
    # decent predictions

    rng = np.random.RandomState(0)  # 创建一个伪随机数生成器对象rng，种子为0
    n_samples = 1000  # 设置样本数量为1000
    n_features = 2  # 设置特征数量为2
    if problem == "regression":
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features,
            random_state=rng,
        )  # 生成回归问题的数据集X和标签y
        gb = HistGradientBoostingRegressor()  # 创建一个梯度提升回归器对象gb
        expected_min_score = expected_min_score_regression  # 设置预期的最小得分为回归问题的预期最小得分
    else:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features,
            n_redundant=0,
            n_repeated=0,
            random_state=rng,
        )  # 生成分类问题的数据集X和标签y
        gb = HistGradientBoostingClassifier()  # 创建一个梯度提升分类器对象gb
        expected_min_score = expected_min_score_classification  # 设置预期的最小得分为分类问题的预期最小得分

    mask = rng.binomial(1, missing_proportion, size=X.shape).astype(bool)  # 生成二项分布掩码数组mask，掩盖指定比例的数据为NaN
    X[mask] = np.nan  # 将mask中为True的位置设置为NaN

    gb.fit(X, y)  # 使用X和y训练gb

    assert gb.score(X, y) > expected_min_score  # 断言gb在X和y上的得分大于预期的最小得分
    [
        # 生成一个二分类问题的数据集，使用随机种子 0
        make_classification(random_state=0, n_classes=2),
        # 生成一个三分类问题的数据集，使用随机种子 0，且包含 3 个信息特征
        make_classification(random_state=0, n_classes=3, n_informative=3),
    ],
    # 定义测试用例的标识符，分别对应二分类和多分类的逻辑损失
    ids=["binary_log_loss", "multiclass_log_loss"],
)
# 定义测试函数 test_zero_division_hessians，用于验证非回归问题 #14018
# 确保在计算叶子值时避免零除错误。

# 如果学习率过高，原始预测结果可能很差，并且会使 softmax 函数（或在二分类中的 sigmoid 函数）饱和。
# 这会导致概率值恰好为 0 或 1，梯度保持恒定，海森矩阵为零。
def test_zero_division_hessians(data):
    X, y = data
    # 创建一个学习率很高、迭代次数较少的 HistGradientBoostingClassifier 对象
    gb = HistGradientBoostingClassifier(learning_rate=100, max_iter=10)
    gb.fit(X, y)


def test_small_trainset():
    # 确保小训练集是分层的，并且具有预期的长度（10k 个样本）
    n_samples = 20000
    original_distrib = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples).reshape(n_samples, 1)
    # 根据原始分布生成标签 y
    y = [
        [class_] * int(prop * n_samples) for (class_, prop) in original_distrib.items()
    ]
    y = shuffle(np.concatenate(y))
    gb = HistGradientBoostingClassifier()

    # 计算小训练集
    X_small, y_small, *_ = gb._get_small_trainset(
        X, y, seed=42, sample_weight_train=None
    )

    # 计算小训练集中的类别分布
    unique, counts = np.unique(y_small, return_counts=True)
    small_distrib = {class_: count / 10000 for (class_, count) in zip(unique, counts)}

    # 测试小训练集的长度是否符合预期
    assert X_small.shape[0] == 10000
    assert y_small.shape[0] == 10000

    # 测试整个数据集和小训练集中的类别分布是否相同
    assert small_distrib == pytest.approx(original_distrib)


def test_missing_values_minmax_imputation():
    # 比较直方图梯度提升分类器的内置缺失值处理与一种先验缺失值插补策略的决策函数的一致性。
    #
    # 每个包含 NaN 值的特征被替换为两个特征：
    # - 一个将 NaN 替换为该特征的最小值 - 1
    # - 一个将 NaN 替换为该特征的最大值 + 1
    # 当 NaN 值向左移动时的分裂，在第一个（最小）特征中有一个等效分裂，
    # 当 NaN 值向右移动时的分裂，在第二个（最大）特征中有一个等效分裂。
    #
    # 假设数据永远不会出现在训练过程中选择最佳特征时有多个等分的情况下，
    # 学习到的决策树应该是严格相等的（学习一系列编码相同决策函数的分裂）。
    #
    # MinMaxImputer 转换器旨在作为决策树的“缺失在属性”（MIA）缺失值处理的玩具实现
    # https://www.sciencedirect.com/science/article/abs/pii/S0167865508000305
    # 将 MIA 实现为插补转换器的建议是由“Remark 3”在 :arxiv:'<1902.06931>` 中提出的。
    class MinMaxImputer(TransformerMixin, BaseEstimator):
        # 自定义转换器类，实现了TransformerMixin和BaseEstimator接口

        def fit(self, X, y=None):
            # 拟合方法，用于计算最小最大缩放器，并保存数据的最小和最大值
            mm = MinMaxScaler().fit(X)
            self.data_min_ = mm.data_min_  # 保存数据的最小值
            self.data_max_ = mm.data_max_  # 保存数据的最大值
            return self

        def transform(self, X):
            # 转换方法，用于处理缺失值
            X_min, X_max = X.copy(), X.copy()

            for feature_idx in range(X.shape[1]):
                nan_mask = np.isnan(X[:, feature_idx])  # 找到缺失值位置
                X_min[nan_mask, feature_idx] = self.data_min_[feature_idx] - 1  # 用最小值减去1来填充缺失值
                X_max[nan_mask, feature_idx] = self.data_max_[feature_idx] + 1  # 用最大值加上1来填充缺失值

            return np.concatenate([X_min, X_max], axis=1)  # 返回处理后的数据

    def make_missing_value_data(n_samples=int(1e4), seed=0):
        # 生成带有缺失值的数据集

        rng = np.random.RandomState(seed)
        X, y = make_regression(n_samples=n_samples, n_features=4, random_state=rng)

        # 将数据进行分箱处理，确保对两种策略有确定性的处理，并便于在结构化方式下插入np.nan
        X = KBinsDiscretizer(n_bins=42, encode="ordinal").fit_transform(X)

        # 第一个特征完全随机地缺失值
        rnd_mask = rng.rand(X.shape[0]) > 0.9
        X[rnd_mask, 0] = np.nan

        # 第二和第三个特征在极值处缺失值（对缺失的删减）
        low_mask = X[:, 1] == 0
        X[low_mask, 1] = np.nan

        high_mask = X[:, 2] == X[:, 2].max()
        X[high_mask, 2] = np.nan

        # 让最后一个特征的缺失模式非常有信息
        y_max = np.percentile(y, 70)
        y_max_mask = y >= y_max
        y[y_max_mask] = y_max
        X[y_max_mask, 3] = np.nan

        # 检查每个特征至少有一个缺失值
        for feature_idx in range(X.shape[1]):
            assert any(np.isnan(X[:, feature_idx]))

        # 使用测试集检查学习的决策函数在未见数据上的表现是否相同
        return train_test_split(X, y, random_state=rng)

    # n_samples需要足够大，以最小化在给定树中具有相同增益值的多个候选分裂的可能性
    X_train, X_test, y_train, y_test = make_missing_value_data(
        n_samples=int(1e4), seed=0
    )

    # 使用少量叶子节点和迭代次数以确保模型欠拟合，从而最小化训练模型时出现并列的可能性
    gbm1 = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=5, random_state=0)
    gbm1.fit(X_train, y_train)

    gbm2 = make_pipeline(MinMaxImputer(), clone(gbm1))
    gbm2.fit(X_train, y_train)

    # 检查模型是否达到相同的评分
    assert gbm1.score(X_train, y_train) == pytest.approx(gbm2.score(X_train, y_train))

    assert gbm1.score(X_test, y_test) == pytest.approx(gbm2.score(X_test, y_test))
    # 检查两个梯度提升机模型在训练集上的预测结果是否接近，用于精细粒度的决策函数验证。
    assert_allclose(gbm1.predict(X_train), gbm2.predict(X_train))
    
    # 检查两个梯度提升机模型在测试集上的预测结果是否接近，用于精细粒度的决策函数验证。
    assert_allclose(gbm1.predict(X_test), gbm2.predict(X_test))
def test_infinite_values():
    # Basic test for infinite values

    # 创建包含无限值的numpy数组，reshape为列向量
    X = np.array([-np.inf, 0, 1, np.inf]).reshape(-1, 1)
    # 创建目标变量的numpy数组
    y = np.array([0, 0, 1, 1])

    # 初始化一个最小叶子样本数为1的HistGradientBoostingRegressor对象
    gbdt = HistGradientBoostingRegressor(min_samples_leaf=1)
    # 使用X和y来拟合梯度提升决策树回归模型
    gbdt.fit(X, y)
    # 使用np.testing.assert_allclose检查预测值和真实值的近似性
    np.testing.assert_allclose(gbdt.predict(X), y, atol=1e-4)


def test_consistent_lengths():
    # 创建包含无限值的numpy数组，reshape为列向量
    X = np.array([-np.inf, 0, 1, np.inf]).reshape(-1, 1)
    # 创建目标变量的numpy数组
    y = np.array([0, 0, 1, 1])
    # 创建样本权重的numpy数组
    sample_weight = np.array([0.1, 0.3, 0.1])

    # 初始化一个默认参数的HistGradientBoostingRegressor对象
    gbdt = HistGradientBoostingRegressor()

    # 使用pytest.raises检查是否抛出预期的ValueError异常
    with pytest.raises(ValueError, match=r"sample_weight.shape == \(3,\), expected"):
        gbdt.fit(X, y, sample_weight)

    # 使用pytest.raises检查是否抛出预期的ValueError异常
    with pytest.raises(
        ValueError, match="Found input variables with inconsistent number"
    ):
        gbdt.fit(X, y[1:])


def test_infinite_values_missing_values():
    # High level test making sure that inf and nan values are properly handled
    # when both are present. This is similar to
    # test_split_on_nan_with_infinite_values() in test_grower.py, though we
    # cannot check the predictions for binned values here.

    # 创建包含-inf, 0, 1, inf和nan值的numpy数组，reshape为列向量
    X = np.asarray([-np.inf, 0, 1, np.inf, np.nan]).reshape(-1, 1)
    # 创建标记nan值的numpy数组
    y_isnan = np.isnan(X.ravel())
    # 创建标记inf值的numpy数组
    y_isinf = X.ravel() == np.inf

    # 初始化一个最小叶子样本数为1、最大迭代次数为1、学习率为1、最大深度为2的HistGradientBoostingClassifier对象
    stump_clf = HistGradientBoostingClassifier(
        min_samples_leaf=1, max_iter=1, learning_rate=1, max_depth=2
    )

    # 拟合模型并使用score方法计算模型在X上的准确率
    assert stump_clf.fit(X, y_isinf).score(X, y_isinf) == 1
    assert stump_clf.fit(X, y_isnan).score(X, y_isnan) == 1


@pytest.mark.parametrize("scoring", [None, "loss"])
def test_string_target_early_stopping(scoring):
    # Regression tests for #14709 where the targets need to be encoded before
    # to compute the score

    # 创建服从标准正态分布的随机数生成器
    rng = np.random.RandomState(42)
    # 创建服从标准正态分布的100行10列的numpy数组
    X = rng.randn(100, 10)
    # 创建包含50个"x"和50个"y"字符串的numpy数组，dtype为object
    y = np.array(["x"] * 50 + ["y"] * 50, dtype=object)
    # 初始化一个n_iter_no_change为10、scoring为scoring参数的HistGradientBoostingClassifier对象
    gbrt = HistGradientBoostingClassifier(n_iter_no_change=10, scoring=scoring)
    # 使用X和y拟合梯度提升决策树分类模型
    gbrt.fit(X, y)


def test_zero_sample_weights_regression():
    # Make sure setting a SW to zero amounts to ignoring the corresponding
    # sample

    # 创建包含两个特征的训练数据集列表
    X = [[1, 0], [1, 0], [1, 0], [0, 1]]
    # 创建目标变量的列表
    y = [0, 0, 1, 0]
    # 忽略前两个训练样本的权重设置为0的样本权重列表
    sample_weight = [0, 0, 1, 1]
    # 初始化一个最小叶子样本数为1的HistGradientBoostingRegressor对象
    gb = HistGradientBoostingRegressor(min_samples_leaf=1)
    # 使用X、y和sample_weight拟合梯度提升决策树回归模型
    gb.fit(X, y, sample_weight=sample_weight)
    # 断言预测结果大于0.5
    assert gb.predict([[1, 0]])[0] > 0.5


def test_zero_sample_weights_classification():
    # Make sure setting a SW to zero amounts to ignoring the corresponding
    # sample

    # 创建包含两个特征的训练数据集列表
    X = [[1, 0], [1, 0], [1, 0], [0, 1]]
    # 创建目标变量的列表
    y = [0, 0, 1, 0]
    # 忽略前两个训练样本的权重设置为0的样本权重列表
    sample_weight = [0, 0, 1, 1]
    # 初始化一个损失函数为"log_loss"、最小叶子样本数为1的HistGradientBoostingClassifier对象
    gb = HistGradientBoostingClassifier(loss="log_loss", min_samples_leaf=1)
    # 使用X、y和sample_weight拟合梯度提升决策树分类模型
    gb.fit(X, y, sample_weight=sample_weight)
    # 使用assert_array_equal检查预测结果是否符合期望值
    assert_array_equal(gb.predict([[1, 0]]), [1])

    # 创建包含两个特征的训练数据集列表
    X = [[1, 0], [1, 0], [1, 0], [0, 1], [1, 1]]
    # 创建目标变量的列表
    y = [0, 0, 1, 0, 2]
    # 忽略前两个训练样本的权重设置为0的样本权重列表
    sample_weight = [0, 0, 1, 1, 1]
    # 创建一个梯度提升分类器对象，使用对数损失函数进行训练
    gb = HistGradientBoostingClassifier(loss="log_loss", min_samples_leaf=1)
    # 使用给定的训练数据 X、标签 y 和样本权重 sample_weight 来训练梯度提升分类器
    gb.fit(X, y, sample_weight=sample_weight)
    # 断言预测结果是否与预期的数组 [1] 相等，用于验证模型的预测准确性
    assert_array_equal(gb.predict([[1, 0]]), [1])
# 使用 pytest.mark.parametrize 装饰器设置两组参数，分别为 problem 和 duplication
@pytest.mark.parametrize(
    "problem", ("regression", "binary_classification", "multiclass_classification")
)
@pytest.mark.parametrize("duplication", ("half", "all"))
def test_sample_weight_effect(problem, duplication):
    # 高级别测试，确保复制样本等效于赋予样本权重为 2

    # 当 n_samples > 255 时失败，因为分箱操作不考虑样本权重。
    # 保持 n_samples <= 255 可以确保只使用唯一的值，从而样本权重不影响分箱结果。
    n_samples = 255
    n_features = 2
    if problem == "regression":
        # 如果问题为回归，则生成回归数据集
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features,
            random_state=0,
        )
        Klass = HistGradientBoostingRegressor
    else:
        # 如果问题为分类问题，则生成分类数据集
        n_classes = 2 if problem == "binary_classification" else 3
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features,
            n_redundant=0,
            n_clusters_per_class=1,
            n_classes=n_classes,
            random_state=0,
        )
        Klass = HistGradientBoostingClassifier

    # 当 min_samples_leaf > 1 时，此测试无法通过，因为这将强制 est_sw 中的 2 个样本在同一个节点中，
    # 而在 est_dup 中这些样本可以分开。
    est = Klass(min_samples_leaf=1)

    # 创建包含重复样本及其相应样本权重的数据集
    if duplication == "half":
        lim = n_samples // 2
    else:
        lim = n_samples
    X_dup = np.r_[X, X[:lim]]
    y_dup = np.r_[y, y[:lim]]
    sample_weight = np.ones(shape=(n_samples))
    sample_weight[:lim] = 2

    # 使用 clone 克隆 est，并分别使用 est_sw 和 est_dup 拟合数据集
    est_sw = clone(est).fit(X, y, sample_weight=sample_weight)
    est_dup = clone(est).fit(X_dup, y_dup)

    # 对于分类问题，使用 assert 检查 _raw_predict 更为严格
    assert np.allclose(est_sw._raw_predict(X_dup), est_dup._raw_predict(X_dup))


@pytest.mark.parametrize("Loss", (HalfSquaredError, AbsoluteError))
def test_sum_hessians_are_sample_weight(Loss):
    # 对于具有常数 hessian 的损失函数，直方图的 sum_hessians 字段必须等于相应箱子中样本权重的总和

    rng = np.random.RandomState(0)
    n_samples = 1000
    n_features = 2
    X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=rng)
    bin_mapper = _BinMapper()
    X_binned = bin_mapper.fit_transform(X)

    # 虽然样本权重应为正数，但这个测试仍然有效
    sample_weight = rng.normal(size=n_samples)

    # 初始化损失函数对象，并获取梯度和 hessian 矩阵
    loss = Loss(sample_weight=sample_weight)
    gradients, hessians = loss.init_gradient_and_hessian(
        n_samples=n_samples, dtype=G_H_DTYPE
    )
    gradients, hessians = gradients.reshape((-1, 1)), hessians.reshape((-1, 1))
    # 生成服从正态分布的随机预测结果，形状为 (n_samples, 1)
    raw_predictions = rng.normal(size=(n_samples, 1))
    
    # 计算损失函数的梯度和黑塞矩阵
    loss.gradient_hessian(
        y_true=y,
        raw_prediction=raw_predictions,
        sample_weight=sample_weight,
        gradient_out=gradients,
        hessian_out=hessians,
        n_threads=n_threads,
    )
    
    # 构建 sum_sample_weight 数组，包含每个特征每个箱中的样本权重之和，
    # 这个值必须等于相应直方图的 sum_hessians 字段
    sum_sw = np.zeros(shape=(n_features, bin_mapper.n_bins))
    for feature_idx in range(n_features):
        for sample_idx in range(n_samples):
            # 将每个样本的权重加到对应特征的对应箱中
            sum_sw[feature_idx, X_binned[sample_idx, feature_idx]] += sample_weight[
                sample_idx
            ]
    
    # 构建直方图
    grower = TreeGrower(
        X_binned, gradients[:, 0], hessians[:, 0], n_bins=bin_mapper.n_bins
    )
    histograms = grower.histogram_builder.compute_histograms_brute(
        grower.root.sample_indices
    )
    
    # 验证每个特征每个箱的直方图的 sum_hessians 字段是否与 sum_sw 数组中对应值近似相等
    for feature_idx in range(n_features):
        for bin_idx in range(bin_mapper.n_bins):
            assert histograms[feature_idx, bin_idx]["sum_hessians"] == (
                pytest.approx(sum_sw[feature_idx, bin_idx], rel=1e-5)
            )
def test_max_depth_max_leaf_nodes():
    # 非回归测试，检查 https://github.com/scikit-learn/scikit-learn/issues/16179
    # 当 max_depth 和 max_leaf_nodes 条件同时满足时，存在 bug，导致 max_leaf_nodes 无法被正确遵守。
    X, y = make_classification(random_state=0)
    # 创建并拟合一个 HistGradientBoostingClassifier 对象
    est = HistGradientBoostingClassifier(max_depth=2, max_leaf_nodes=3, max_iter=1).fit(
        X, y
    )
    # 获取第一个树对象
    tree = est._predictors[0][0]
    # 断言树的最大深度为 2
    assert tree.get_max_depth() == 2
    # 断言叶节点的数量为 3（在 bug 修复之前是 4）
    assert tree.get_n_leaf_nodes() == 3


def test_early_stopping_on_test_set_with_warm_start():
    # 非回归测试，检查 #16661，当 warm_start=True，early_stopping=True，没有验证集时，第二次拟合失败的问题。
    X, y = make_classification(random_state=0)
    # 创建并拟合一个 HistGradientBoostingClassifier 对象
    gb = HistGradientBoostingClassifier(
        max_iter=1,
        scoring="loss",
        warm_start=True,
        early_stopping=True,
        n_iter_no_change=1,
        validation_fraction=None,
    )

    gb.fit(X, y)
    # 第二次调用不应该引发异常
    gb.set_params(max_iter=2)
    gb.fit(X, y)


def test_early_stopping_with_sample_weights(monkeypatch):
    """检查样本权重是否传递给评分器，并且不调用 _raw_predict。"""

    # 设置模拟评分器
    mock_scorer = Mock(side_effect=get_scorer("neg_median_absolute_error"))

    # 模拟 check_scoring 函数
    def mock_check_scoring(estimator, scoring):
        assert scoring == "neg_median_absolute_error"
        return mock_scorer

    # 使用 monkeypatch 设置 check_scoring 函数
    monkeypatch.setattr(
        sklearn.ensemble._hist_gradient_boosting.gradient_boosting,
        "check_scoring",
        mock_check_scoring,
    )

    X, y = make_regression(random_state=0)
    sample_weight = np.ones_like(y)
    # 创建 HistGradientBoostingRegressor 对象
    hist = HistGradientBoostingRegressor(
        max_iter=2,
        early_stopping=True,
        random_state=0,
        scoring="neg_median_absolute_error",
    )
    # 设置 _raw_predict 的模拟对象
    mock_raw_predict = Mock(side_effect=hist._raw_predict)
    hist._raw_predict = mock_raw_predict
    # 使用样本权重拟合模型
    hist.fit(X, y, sample_weight=sample_weight)

    # _raw_predict 不应该以字符串形式的评分被调用
    assert mock_raw_predict.call_count == 0

    # 对于每次迭代，评分器应该被调用两次（训练集和验证集），所以对于 `max_iter=2` 总共调用 6 次。
    assert mock_scorer.call_count == 6
    for arg_list in mock_scorer.call_args_list:
        assert "sample_weight" in arg_list[1]


def test_raw_predict_is_called_with_custom_scorer():
    """自定义评分器仍然会调用 _raw_predict。"""

    # 设置模拟评分器
    mock_scorer = Mock(side_effect=get_scorer("neg_median_absolute_error"))

    X, y = make_regression(random_state=0)
    # 创建 HistGradientBoostingRegressor 对象，使用自定义评分器
    hist = HistGradientBoostingRegressor(
        max_iter=2,
        early_stopping=True,
        random_state=0,
        scoring=mock_scorer,
    )
    # 设置 _raw_predict 的模拟对象
    mock_raw_predict = Mock(side_effect=hist._raw_predict)
    hist._raw_predict = mock_raw_predict
    # 拟合模型
    hist.fit(X, y)
    # 确保 `_raw_predict` 和 `scorer` 在基准评分时被调用了两次（训练和验证集各一次），
    # 并在每次迭代后被调用两次（训练和验证集各一次）。因此，对于 `max_iter=2`，总共调用了6次。
    assert mock_raw_predict.call_count == 6
    assert mock_scorer.call_count == 6
@pytest.mark.parametrize(
    "Est", (HistGradientBoostingClassifier, HistGradientBoostingRegressor)
)
# 定义测试函数 test_single_node_trees，用于验证单节点树的构建情况
def test_single_node_trees(Est):
    # 确保仍然可以构建单节点树。在这种情况下，根节点的值被设置为0。
    # 这是正确的值：如果树是单节点的，那是因为从根节点开始就没有满足 min_gain_to_split 的条件，
    # 所以我们不希望树对预测产生任何影响。
    
    # 生成一个简单的分类数据集 X, y
    X, y = make_classification(random_state=0)
    y[:] = 1  # 将目标值全部设为1，导致只有一个根节点的树结构

    # 初始化一个估计器对象 est
    est = Est(max_iter=20)
    # 使用数据集 X, y 进行拟合
    est.fit(X, y)

    # 断言：所有的估计器都只有一个节点
    assert all(len(predictor[0].nodes) == 1 for predictor in est._predictors)
    # 断言：所有估计器的根节点值都为0
    assert all(predictor[0].nodes[0]["value"] == 0 for predictor in est._predictors)
    # 断言：由于基线预测值，仍然可以给出正确的预测结果
    assert_allclose(est.predict(X), y)


@pytest.mark.parametrize(
    "Est, loss, X, y",
    [
        (
            HistGradientBoostingClassifier,
            HalfBinomialLoss(sample_weight=None),
            X_classification,
            y_classification,
        ),
        (
            HistGradientBoostingRegressor,
            HalfSquaredError(sample_weight=None),
            X_regression,
            y_regression,
        ),
    ],
)
# 定义测试函数 test_custom_loss，用于测试自定义损失函数
def test_custom_loss(Est, loss, X, y):
    # 初始化估计器对象 est，指定损失函数 loss 和最大迭代次数 max_iter
    est = Est(loss=loss, max_iter=20)
    # 使用数据集 X, y 进行拟合
    est.fit(X, y)


@pytest.mark.parametrize(
    "HistGradientBoosting, X, y",
    [
        (HistGradientBoostingClassifier, X_classification, y_classification),
        (HistGradientBoostingRegressor, X_regression, y_regression),
        (
            HistGradientBoostingClassifier,
            X_multi_classification,
            y_multi_classification,
        ),
    ],
)
# 定义测试函数 test_staged_predict，用于测试 staged 预测方法
def test_staged_predict(HistGradientBoosting, X, y):
    # 测试 staged 预测是否最终给出相同的预测结果
    
    # 将数据集分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0
    )
    # 初始化 HistGradientBoosting 对象 gb，设置最大迭代次数为 10
    gb = HistGradientBoosting(max_iter=10)

    # 断言：如果未进行拟合，则测试 staged_predict 是否会引发 NotFittedError
    with pytest.raises(NotFittedError):
        next(gb.staged_predict(X_test))

    # 使用训练集进行拟合
    gb.fit(X_train, y_train)

    # 断言：测试每次迭代的 staged 预测是否与从头开始训练的相应预测一致
    # 同时也测试了当 max_iter = 1 时的极端情况
    method_names = (
        ["predict"]
        if is_regressor(gb)
        else ["predict", "predict_proba", "decision_function"]
    )
    # 对于给定的每个方法名，依次执行以下操作
    for method_name in method_names:
        # 使用 getattr 函数获取 gb 对象的对应方法，例如 "staged_method_name"
        staged_method = getattr(gb, "staged_" + method_name)
        
        # 调用获取到的 staged_method 对象，并将结果转换为列表，存储在 staged_predictions 中
        staged_predictions = list(staged_method(X_test))
        
        # 断言 staged_predictions 的长度等于 gb 模型的迭代次数 n_iter_
        assert len(staged_predictions) == gb.n_iter_
        
        # 对 staged_method(X_test) 返回的结果进行遍历，从 1 开始计数每次迭代 n_iter
        for n_iter, staged_predictions in enumerate(staged_method(X_test), 1):
            # 创建一个新的 HistGradientBoosting 对象 aux，设置其最大迭代次数为当前 n_iter
            aux = HistGradientBoosting(max_iter=n_iter)
            
            # 使用 aux 对象拟合训练数据 X_train 和标签 y_train
            aux.fit(X_train, y_train)
            
            # 调用 aux 对象的 method_name 方法来预测测试数据 X_test
            pred_aux = getattr(aux, method_name)(X_test)
            
            # 断言 staged_predictions 和 pred_aux 的值接近
            assert_allclose(staged_predictions, pred_aux)
            
            # 断言 staged_predictions 和 pred_aux 的形状相同
            assert staged_predictions.shape == pred_aux.shape
@pytest.mark.parametrize("insert_missing", [False, True])
@pytest.mark.parametrize(
    "Est", (HistGradientBoostingRegressor, HistGradientBoostingClassifier)
)
@pytest.mark.parametrize("bool_categorical_parameter", [True, False])
@pytest.mark.parametrize("missing_value", [np.nan, -1])
# 定义测试函数，参数化测试插入缺失值、估计器类型、布尔类别参数和缺失值的情况
def test_unknown_categories_nan(
    insert_missing, Est, bool_categorical_parameter, missing_value
):
    # Make sure no error is raised at predict if a category wasn't seen during
    # fit. We also make sure they're treated as nans.
    # 确保在预测过程中，如果在拟合过程中未见到某个类别，不会引发错误，并确保它们被视为缺失值。

    rng = np.random.RandomState(0)
    n_samples = 1000
    f1 = rng.rand(n_samples)
    f2 = rng.randint(4, size=n_samples)
    X = np.c_[f1, f2]
    y = np.zeros(shape=n_samples)
    y[X[:, 1] % 2 == 0] = 1

    if bool_categorical_parameter:
        categorical_features = [False, True]
    else:
        categorical_features = [1]

    if insert_missing:
        mask = rng.binomial(1, 0.01, size=X.shape).astype(bool)
        assert mask.sum() > 0
        X[mask] = missing_value

    # 使用指定的估计器类型和参数进行拟合
    est = Est(max_iter=20, categorical_features=categorical_features).fit(X, y)
    # 断言是否正确标识出分类特征
    assert_array_equal(est.is_categorical_, [False, True])

    # Make sure no error is raised on unknown categories and nans
    # unknown categories will be treated as nans
    # 确保在处理未知类别和缺失值时不会引发错误，未知类别将被视为缺失值
    X_test = np.zeros((10, X.shape[1]), dtype=float)
    X_test[:5, 1] = 30
    X_test[5:, 1] = missing_value
    assert len(np.unique(est.predict(X_test))) == 1


def test_categorical_encoding_strategies():
    # Check native categorical handling vs different encoding strategies. We
    # make sure that native encoding needs only 1 split to achieve a perfect
    # prediction on a simple dataset. In contrast, OneHotEncoded data needs
    # more depth / splits, and treating categories as ordered (just using
    # OrdinalEncoder) requires even more depth.
    # 检查原生分类处理与不同编码策略的区别。确保原生编码在简单数据集上只需一个分割即可实现完美预测。
    # 相比之下，使用OneHot编码的数据需要更深的树或更多的分割，而将类别视为有序（仅使用OrdinalEncoder）则需要更深的树。

    # dataset with one random continuous feature, and one categorical feature
    # with values in [0, 5], e.g. from an OrdinalEncoder.
    # class == 1 iff categorical value in {0, 2, 4}
    # 包含一个随机连续特征和一个分类特征的数据集，分类特征取值范围为[0, 5]，例如使用OrdinalEncoder编码。
    # 如果分类值为{0, 2, 4}，则类别为1

    rng = np.random.RandomState(0)
    n_samples = 10_000
    f1 = rng.rand(n_samples)
    f2 = rng.randint(6, size=n_samples)
    X = np.c_[f1, f2]
    y = np.zeros(shape=n_samples)
    y[X[:, 1] % 2 == 0] = 1

    # make sure dataset is balanced so that the baseline_prediction doesn't
    # influence predictions too much with max_iter = 1
    # 确保数据集平衡，以使基线预测在max_iter = 1时不会对预测产生太大影响
    assert 0.49 < y.mean() < 0.51

    native_cat_specs = [
        [False, True],
        [1],
    ]
    try:
        import pandas as pd

        X = pd.DataFrame(X, columns=["f_0", "f_1"])
        native_cat_specs.append(["f_1"])
    except ImportError:
        pass

    for native_cat_spec in native_cat_specs:
        clf_cat = HistGradientBoostingClassifier(
            max_iter=1, max_depth=1, categorical_features=native_cat_spec
        )
        clf_cat.fit(X, y)

        # Using native categorical encoding, we get perfect predictions with just
        # one split
        # 使用原生分类编码，我们只需一个分割即可实现完美预测
        assert cross_val_score(clf_cat, X, y).mean() == 1
    # 对于位集的快速检查：0, 2, 4 = 2**0 + 2**2 + 2**4 = 21
    expected_left_bitset = [21, 0, 0, 0, 0, 0, 0, 0]
    
    # 从分类器 clf_cat 中获取第一个预测器，然后获取其原始左侧分类位集的第一个元素
    left_bitset = clf_cat.fit(X, y)._predictors[0][0].raw_left_cat_bitsets[0]
    
    # 断言左侧位集与预期的位集相等
    assert_array_equal(left_bitset, expected_left_bitset)
    
    # 将分类看作是有序的，为了获得相同的预测结果，我们需要更多的深度/分割
    clf_no_cat = HistGradientBoostingClassifier(
        max_iter=1, max_depth=4, categorical_features=None
    )
    
    # 断言不使用分类特征时的交叉验证得分的平均值小于 0.9
    assert cross_val_score(clf_no_cat, X, y).mean() < 0.9
    
    # 设置分类器的最大深度为 5
    clf_no_cat.set_params(max_depth=5)
    
    # 断言使用新的最大深度时的交叉验证得分的平均值等于 1
    assert cross_val_score(clf_no_cat, X, y).mean() == 1
    
    # 使用独热编码后的数据，相比纯独热编码的数据，我们需要更少的分割，但仍然比使用原生分类分割时需要更多分割
    ct = make_column_transformer(
        (OneHotEncoder(sparse_output=False), [1]), remainder="passthrough"
    )
    
    # 对 X 进行列变换
    X_ohe = ct.fit_transform(X)
    
    # 设置分类器的最大深度为 2
    clf_no_cat.set_params(max_depth=2)
    
    # 断言使用新的最大深度时的交叉验证得分的平均值小于 0.9
    assert cross_val_score(clf_no_cat, X_ohe, y).mean() < 0.9
    
    # 设置分类器的最大深度为 3
    clf_no_cat.set_params(max_depth=3)
    
    # 断言使用新的最大深度时的交叉验证得分的平均值等于 1
    assert cross_val_score(clf_no_cat, X_ohe, y).mean() == 1
# 使用 pytest.mark.parametrize 装饰器为测试函数参数化，测试 HistGradientBoostingClassifier 和 HistGradientBoostingRegressor 两个估算器类
@pytest.mark.parametrize(
    "Est", (HistGradientBoostingClassifier, HistGradientBoostingRegressor)
)
# 使用 pytest.mark.parametrize 装饰器为测试函数参数化，测试不同情况下的分类特征、单调性约束及预期错误消息
@pytest.mark.parametrize(
    "categorical_features, monotonic_cst, expected_msg",
    [
        (
            [b"hello", b"world"],
            None,
            re.escape(
                "categorical_features must be an array-like of bool, int or str, "
                "got: bytes40."
            ),
        ),
        (
            np.array([b"hello", 1.3], dtype=object),
            None,
            re.escape(
                "categorical_features must be an array-like of bool, int or str, "
                "got: bytes, float."
            ),
        ),
        (
            [0, -1],
            None,
            re.escape(
                "categorical_features set as integer indices must be in "
                "[0, n_features - 1]"
            ),
        ),
        (
            [True, True, False, False, True],
            None,
            re.escape(
                "categorical_features set as a boolean mask must have shape "
                "(n_features,)"
            ),
        ),
        (
            [True, True, False, False],
            [0, -1, 0, 1],
            "Categorical features cannot have monotonic constraints",
        ),
    ],
)
# 定义测试函数 test_categorical_spec_errors，测试当分类特征指定错误时的错误情况
def test_categorical_spec_errors(
    Est, categorical_features, monotonic_cst, expected_msg
):
    # 生成样本数据 X 和目标标签 y
    n_samples = 100
    X, y = make_classification(random_state=0, n_features=4, n_samples=n_samples)
    rng = np.random.RandomState(0)
    # 修改样本数据的第一列和第二列为随机整数
    X[:, 0] = rng.randint(0, 10, size=n_samples)
    X[:, 1] = rng.randint(0, 10, size=n_samples)
    # 使用给定的参数初始化估算器对象 est
    est = Est(categorical_features=categorical_features, monotonic_cst=monotonic_cst)

    # 使用 pytest.raises 检查是否抛出预期的 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=expected_msg):
        est.fit(X, y)


# 使用 pytest.mark.parametrize 装饰器为测试函数参数化，测试 HistGradientBoostingClassifier 和 HistGradientBoostingRegressor 两个估算器类
@pytest.mark.parametrize(
    "Est", (HistGradientBoostingClassifier, HistGradientBoostingRegressor)
)
# 定义测试函数 test_categorical_spec_errors_with_feature_names，测试在使用特征名称时分类特征指定错误时的错误情况
def test_categorical_spec_errors_with_feature_names(Est):
    # 导入 pandas 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")
    n_samples = 10
    # 创建包含特征和特征值的 pandas DataFrame 对象 X
    X = pd.DataFrame(
        {
            "f0": range(n_samples),
            "f1": range(n_samples),
            "f2": [1.0] * n_samples,
        }
    )
    # 创建目标标签 y，交替包含 0 和 1
    y = [0, 1] * (n_samples // 2)

    # 使用给定的参数初始化估算器对象 est
    est = Est(categorical_features=["f0", "f1", "f3"])
    # 定义预期的错误消息
    expected_msg = re.escape(
        "categorical_features has a item value 'f3' which is not a valid "
        "feature name of the training data."
    )
    # 使用 pytest.raises 检查是否抛出预期的 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=expected_msg):
        est.fit(X, y)

    # 使用给定的参数初始化估算器对象 est
    est = Est(categorical_features=["f0", "f1"])
    # 定义预期的错误消息
    expected_msg = re.escape(
        "categorical_features should be passed as an array of integers or "
        "as a boolean mask when the model is fitted on data without feature "
        "names."
    )
    # 使用 pytest.raises 检查是否抛出预期的 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=expected_msg):
        est.fit(X.to_numpy(), y)
@pytest.mark.parametrize("categorical_features", ([False, False], []))
@pytest.mark.parametrize("as_array", (True, False))
def test_categorical_spec_no_categories(Est, categorical_features, as_array):
    # 确保即使 categorical_features 参数不是 None，我们也能正确检测到没有分类特征存在
    X = np.arange(10).reshape(5, 2)
    y = np.arange(5)
    if as_array:
        categorical_features = np.asarray(categorical_features)
    # 使用给定的 Est 实例化对象，传入 categorical_features 参数并拟合数据
    est = Est(categorical_features=categorical_features).fit(X, y)
    assert est.is_categorical_ is None


@pytest.mark.parametrize(
    "Est", (HistGradientBoostingClassifier, HistGradientBoostingRegressor)
)
@pytest.mark.parametrize(
    "use_pandas, feature_name", [(False, "at index 0"), (True, "'f0'")]
)
def test_categorical_bad_encoding_errors(Est, use_pandas, feature_name):
    # 测试当类别编码错误时是否会报错

    # 创建 HistGradientBoostingClassifier 或 HistGradientBoostingRegressor 实例，指定分类特征和最大箱数为2
    gb = Est(categorical_features=[True], max_bins=2)

    if use_pandas:
        # 导入 pandas 库，如果导入失败则跳过测试
        pd = pytest.importorskip("pandas")
        # 创建包含特定列名和数据的 DataFrame
        X = pd.DataFrame({"f0": [0, 1, 2]})
    else:
        # 创建 numpy 数组，每行表示一个特征
        X = np.array([[0, 1, 2]]).T
    y = np.arange(3)
    # 设置错误信息，检查是否符合预期的异常信息
    msg = (
        f"Categorical feature {feature_name} is expected to have a "
        "cardinality <= 2 but actually has a cardinality of 3."
    )
    # 断言是否会抛出 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        gb.fit(X, y)

    # 忽略 NaN 值后进行拟合
    X = np.array([[0, 1, np.nan]]).T
    y = np.arange(3)
    gb.fit(X, y)


@pytest.mark.parametrize(
    "Est", (HistGradientBoostingClassifier, HistGradientBoostingRegressor)
)
def test_uint8_predict(Est):
    # 非回归测试，验证在预测中 X 可以是 uint8 类型（即 X_BINNED_DTYPE），会被转换为 X_DTYPE。

    rng = np.random.RandomState(0)

    # 创建 uint8 类型的随机数数组 X 和 y
    X = rng.randint(0, 100, size=(10, 2)).astype(np.uint8)
    y = rng.randint(0, 2, size=10).astype(np.uint8)
    est = Est()
    # 拟合模型
    est.fit(X, y)
    # 进行预测
    est.predict(X)


@pytest.mark.parametrize(
    "interaction_cst, n_features, result",
    [
        (None, 931, None),
        ([{0, 1}], 2, [{0, 1}]),
        ("pairwise", 2, [{0, 1}]),
        ("pairwise", 4, [{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}]),
        ("no_interactions", 2, [{0}, {1}]),
        ("no_interactions", 4, [{0}, {1}, {2}, {3}]),
        ([(1, 0), [5, 1]], 6, [{0, 1}, {1, 5}, {2, 3, 4}]),
    ],
)
def test_check_interaction_cst(interaction_cst, n_features, result):
    """检查 _check_interaction_cst 是否返回预期的集合列表"""
    est = HistGradientBoostingRegressor()
    est.set_params(interaction_cst=interaction_cst)
    # 断言 _check_interaction_cst 的返回结果是否符合预期
    assert est._check_interaction_cst(n_features) == result


def test_interaction_cst_numerically():
    """检查交互约束是否没有禁止的交互。"""
    rng = np.random.RandomState(42)
    n_samples = 1000
    X = rng.uniform(size=(n_samples, 2))
    # Construct y with a strong interaction term
    # 构造 y，包含强交互项
    # y = x0 + x1 + 5 * x0 * x1
    y = np.hstack((X, 5 * X[:, [0]] * X[:, [1]])).sum(axis=1)

    # Initialize a Histogram-based Gradient Boosting Regressor
    # 使用随机种子初始化直方图梯度提升回归器
    est = HistGradientBoostingRegressor(random_state=42)
    # Fit the model using input X and target y
    # 使用输入 X 和目标 y 来训练模型
    est.fit(X, y)

    # Initialize another Histogram-based Gradient Boosting Regressor
    # 使用交互约束和随机种子初始化直方图梯度提升回归器
    est_no_interactions = HistGradientBoostingRegressor(
        interaction_cst=[{0}, {1}], random_state=42
    )
    # Fit the model with interaction constraints using input X and target y
    # 使用交互约束、输入 X 和目标 y 来训练模型
    est_no_interactions.fit(X, y)

    delta = 0.25
    # Ensure the test set X_test stays within the bounds of the training set,
    # as tree-based estimators perform poorly with extrapolation.
    # 确保测试集 X_test 不会超出训练集的范围，因为基于树的估计器在这种情况下表现很差。
    X_test = X[(X[:, 0] < 1 - delta) & (X[:, 1] < 1 - delta)]
    X_delta_d_0 = X_test + [delta, 0]
    X_delta_0_d = X_test + [0, delta]
    X_delta_d_d = X_test + [delta, delta]

    # Perform assertion to check the effect of interaction constraints on predictions
    # 断言检查交互约束对预测结果的影响
    assert_allclose(
        est_no_interactions.predict(X_delta_d_d)
        + est_no_interactions.predict(X_test)
        - est_no_interactions.predict(X_delta_d_0)
        - est_no_interactions.predict(X_delta_0_d),
        0,
        atol=1e-12,
    )

    # Validate the expected outcome of expressions with positive results
    # 验证表达式的预期结果为正数
    assert np.all(
        est.predict(X_delta_d_d)
        + est.predict(X_test)
        - est.predict(X_delta_d_0)
        - est.predict(X_delta_0_d)
        > 0.01
    )
# 检查是否在设置评分时不会触发 UserWarning 的情况。

# 导入 pytest 库，如果不存在则跳过当前测试
pd = pytest.importorskip("pandas")

# 生成回归数据集 X 和 y
X, y = make_regression(n_samples=50, random_state=0)

# 将 X 转换为 pandas DataFrame，列名为 col0, col1, ...
X_df = pd.DataFrame(X, columns=[f"col{i}" for i in range(X.shape[1])])

# 初始化 HistGradientBoostingRegressor 模型
est = HistGradientBoostingRegressor(
    random_state=0, scoring="neg_mean_absolute_error", early_stopping=True
)

# 使用 warnings 模块捕获 UserWarning
with warnings.catch_warnings():
    warnings.simplefilter("error", UserWarning)
    # 训练模型
    est.fit(X_df, y)


# 高级别测试，检查 class_weights 的使用情况
def test_class_weights():
    """High level test to check class_weights."""
    
    # 定义样本数和特征数
    n_samples = 255
    n_features = 2

    # 生成分类数据集 X 和 y
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=2,
        random_state=0,
    )

    # 创建一个布尔数组，表示 y 是否等于 1
    y_is_1 = y == 1

    # 初始化 HistGradientBoostingClassifier 分类器
    clf = HistGradientBoostingClassifier(
        min_samples_leaf=2, random_state=0, max_depth=2
    )

    # 初始化样本权重数组，所有样本权重初始化为 1，y_is_1 对应位置的样本权重设为 3.0
    sample_weight = np.ones(shape=(n_samples))
    sample_weight[y_is_1] = 3.0

    # 使用样本权重训练分类器 clf
    clf.fit(X, y, sample_weight=sample_weight)

    # 定义类别权重字典，类别 0 的权重为 1.0，类别 1 的权重为 3.0
    class_weight = {0: 1.0, 1: 3.0}

    # 使用类别权重字典创建一个新的分类器 clf_class_weighted
    clf_class_weighted = clone(clf).set_params(class_weight=class_weight)
    clf_class_weighted.fit(X, y)

    # 验证 clf 和 clf_class_weighted 的决策函数是否接近
    assert_allclose(clf.decision_function(X), clf_class_weighted.decision_function(X))

    # 检查样本权重和类别权重的乘法关系
    clf.fit(X, y, sample_weight=sample_weight**2)
    clf_class_weighted.fit(X, y, sample_weight=sample_weight)
    assert_allclose(clf.decision_function(X), clf_class_weighted.decision_function(X))

    # 创建一个不平衡的数据集
    X_imb = np.concatenate((X[~y_is_1], X[y_is_1][:10]))
    y_imb = np.concatenate((y[~y_is_1], y[y_is_1][:10]))

    # 使用 class_weight="balanced" 创建一个新的分类器 clf_balanced
    clf_balanced = clone(clf).set_params(class_weight="balanced")
    clf_balanced.fit(X_imb, y_imb)

    # 计算类别权重
    class_weight = y_imb.shape[0] / (2 * np.bincount(y_imb))
    sample_weight = class_weight[y_imb]

    # 创建一个使用样本权重的分类器 clf_sample_weight
    clf_sample_weight = clone(clf).set_params(class_weight=None)
    clf_sample_weight.fit(X_imb, y_imb, sample_weight=sample_weight)

    # 验证 clf_balanced 和 clf_sample_weight 的决策函数是否接近
    assert_allclose(
        clf_balanced.decision_function(X_imb),
        clf_sample_weight.decision_function(X_imb),
    )


# 检查当未知类别为负数时不会导致错误的情况。

# 创建随机数生成器 rng
rng = np.random.RandomState(42)

# 设置样本数
n_samples = 1000

# 生成特征 X，第一列为 [0, 1) 之间的随机数，第二列为 [0, 4) 之间的随机整数
X = np.c_[rng.rand(n_samples), rng.randint(4, size=n_samples)]

# 设置目标变量 y，初始全为 0，偶数位置的样本设为 1
y = np.zeros(shape=n_samples)
y[X[:, 1] % 2 == 0] = 1

# 使用 HistGradientBoostingRegressor 回归器，设定相关参数并训练模型
hist = HistGradientBoostingRegressor(
    random_state=0,
    categorical_features=[False, True],
    max_iter=10,
).fit(X, y)
    `
        # 检查第二列的负值是否被视为缺失类别
        X_test_neg = np.asarray([[1, -2], [3, -4]])  # 创建一个包含负值的测试数据数组
        X_test_nan = np.asarray([[1, np.nan], [3, np.nan]])  # 创建一个包含 NaN 的测试数据数组
    
        assert_allclose(hist.predict(X_test_neg), hist.predict(X_test_nan))  # 断言负值和 NaN 的预测结果相近，确保负值和 NaN 被视为相同类别
`
@pytest.mark.parametrize("dataframe_lib", ["pandas", "polars"])
@pytest.mark.parametrize(
    "HistGradientBoosting",
    [HistGradientBoostingClassifier, HistGradientBoostingRegressor],
)
def test_dataframe_categorical_results_same_as_ndarray(
    dataframe_lib, HistGradientBoosting
):
    """Check that pandas categorical give the same results as ndarray."""
    # 导入必要的数据分析库，如果缺少则跳过测试
    pytest.importorskip(dataframe_lib)

    # 设置随机数种子和样本数量、类别数量、最大分箱数
    rng = np.random.RandomState(42)
    n_samples = 5_000
    n_cardinality = 50
    max_bins = 100

    # 生成数值特征和分类特征的随机数据
    f_num = rng.rand(n_samples)
    f_cat = rng.randint(n_cardinality, size=n_samples)

    # 生成分类特征的标签
    y = (f_cat % 3 == 0) & (f_num > 0.2)

    # 将数值特征和分类特征合并成为特征矩阵 X
    X = np.c_[f_num, f_cat]

    # 将分类特征转换为带有前缀的字符串格式
    f_cat = [f"cat{c:0>3}" for c in f_cat]

    # 将特征数据转换为数据分析库指定的数据结构
    X_df = _convert_container(
        np.asarray([f_num, f_cat]).T,
        dataframe_lib,
        ["f_num", "f_cat"],
        categorical_feature_names=["f_cat"],
    )

    # 使用训练测试分割函数划分数据集
    X_train, X_test, X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, X_df, y, random_state=0
    )

    # 定义直方图梯度提升模型的参数
    hist_kwargs = dict(max_iter=10, max_bins=max_bins, random_state=0)

    # 使用 numpy 数组初始化直方图梯度提升模型
    hist_np = HistGradientBoosting(categorical_features=[False, True], **hist_kwargs)
    hist_np.fit(X_train, y_train)

    # 使用数据分析库数据结构初始化直方图梯度提升模型
    hist_pd = HistGradientBoosting(categorical_features="from_dtype", **hist_kwargs)
    hist_pd.fit(X_train_df, y_train)

    # 检查分类特征是否正确并且排序正确
    categories = hist_pd._preprocessor.named_transformers_["encoder"].categories_[0]
    assert_array_equal(categories, np.unique(f_cat))

    # 检查两个模型预测器的数量是否相同
    assert len(hist_np._predictors) == len(hist_pd._predictors)

    # 检查每对预测器的节点数量是否相同
    for predictor_1, predictor_2 in zip(hist_np._predictors, hist_pd._predictors):
        assert len(predictor_1[0].nodes) == len(predictor_2[0].nodes)

    # 检查使用 numpy 数组和数据分析库数据结构的模型得分是否接近
    score_np = hist_np.score(X_test, y_test)
    score_pd = hist_pd.score(X_test_df, y_test)
    assert score_np == pytest.approx(score_pd)

    # 检查使用 numpy 数组和数据分析库数据结构的模型预测结果是否接近
    assert_allclose(hist_np.predict(X_test), hist_pd.predict(X_test_df))


@pytest.mark.parametrize("dataframe_lib", ["pandas", "polars"])
@pytest.mark.parametrize(
    "HistGradientBoosting",
    [HistGradientBoostingClassifier, HistGradientBoostingRegressor],
)
def test_dataframe_categorical_errors(dataframe_lib, HistGradientBoosting):
    """Check error cases for pandas categorical feature."""
    # 导入必要的数据分析库，如果缺少则跳过测试
    pytest.importorskip(dataframe_lib)

    # 设置预期的错误消息
    msg = "Categorical feature 'f_cat' is expected to have a cardinality <= 16"

    # 使用最大分箱数为 16 初始化直方图梯度提升模型
    hist = HistGradientBoosting(categorical_features="from_dtype", max_bins=16)

    # 生成字符串类型的分类特征数据
    rng = np.random.RandomState(42)
    f_cat = rng.randint(0, high=100, size=100).astype(str)

    # 将分类特征数据转换为数据分析库指定的数据结构
    X_df = _convert_container(
        f_cat[:, None], dataframe_lib, ["f_cat"], categorical_feature_names=["f_cat"]
    )

    # 生成随机的目标变量数据
    y = rng.randint(0, high=2, size=100)

    # 使用断言检查是否引发了预期的 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        hist.fit(X_df, y)


@pytest.mark.parametrize("dataframe_lib", ["pandas", "polars"])
def test_categorical_different_order_same_model(dataframe_lib):
    """Check that the order of the categorical gives same model."""
    # 导入 dataframe_lib 库，如果不存在则跳过当前测试
    pytest.importorskip(dataframe_lib)
    # 使用种子为42的随机数生成器创建随机数生成器对象
    rng = np.random.RandomState(42)
    # 设定样本数为1000
    n_samples = 1_000
    # 生成0到1之间的随机整数数组，作为模拟特征整数数据
    f_ints = rng.randint(low=0, high=2, size=n_samples)

    # 构造一个目标数组，加入一些噪声
    y = f_ints.copy()
    # 随机选择一部分索引，将对应位置的元素取反
    flipped = rng.choice([True, False], size=n_samples, p=[0.1, 0.9])
    y[flipped] = 1 - y[flipped]

    # 根据整数数组 f_ints 构造两个分类数组，其中 0 -> A, 1 -> B 和 1 -> A, 0 -> B
    f_cat_a_b = np.asarray(["A", "B"])[f_ints]
    f_cat_b_a = np.asarray(["B", "A"])[f_ints]
    
    # 将分类数组转换为 dataframe_lib 中的数据容器对象 df_a_b 和 df_b_a
    df_a_b = _convert_container(
        f_cat_a_b[:, None],
        dataframe_lib,
        ["f_cat"],
        categorical_feature_names=["f_cat"],
    )
    df_b_a = _convert_container(
        f_cat_b_a[:, None],
        dataframe_lib,
        ["f_cat"],
        categorical_feature_names=["f_cat"],
    )

    # 使用 HistGradientBoostingClassifier 创建两个分类器 hist_a_b 和 hist_b_a
    hist_a_b = HistGradientBoostingClassifier(
        categorical_features="from_dtype", random_state=0
    )
    hist_b_a = HistGradientBoostingClassifier(
        categorical_features="from_dtype", random_state=0
    )

    # 分别使用 hist_a_b 和 hist_b_a 对 df_a_b 和 df_b_a 进行拟合
    hist_a_b.fit(df_a_b, y)
    hist_b_a.fit(df_b_a, y)

    # 断言 hist_a_b 和 hist_b_a 的预测器列表长度相等
    assert len(hist_a_b._predictors) == len(hist_b_a._predictors)
    # 对 hist_a_b 和 hist_b_a 的每个预测器进行逐一检查，断言其节点数相等
    for predictor_1, predictor_2 in zip(hist_a_b._predictors, hist_b_a._predictors):
        assert len(predictor_1[0].nodes) == len(predictor_2[0].nodes)
# TODO(1.6): Remove warning and change default in 1.6
# 当输入的 DataFrame 中包含分类特征时，引发警告。
# 对于 Polars，不需要进行此测试，因为 Polars 的分类必须始终是字符串，
# 而字符串只能作为分类处理。因此，当前将分类列视为数字，未来将其视为分类的情况在 Polars 中不会发生。
def test_categorical_features_warn():
    pd = pytest.importorskip("pandas")  # 导入并检查是否安装了 Pandas 库
    X = pd.DataFrame({"a": pd.Series([1, 2, 3], dtype="category"), "b": [4, 5, 6]})  # 创建包含分类特征的 DataFrame
    y = [0, 1, 0]  # 创建目标变量
    hist = HistGradientBoostingClassifier(random_state=0)  # 创建梯度提升分类器对象

    msg = "The categorical_features parameter will change to 'from_dtype' in v1.6"
    # 使用 pytest 的 warn 函数来检查是否有未来警告消息
    with pytest.warns(FutureWarning, match=msg):
        hist.fit(X, y)  # 使用分类器拟合数据


def get_different_bitness_node_ndarray(node_ndarray):
    new_dtype_for_indexing_fields = np.int64 if _IS_32BIT else np.int32  # 根据平台位数选择新的数据类型

    # Node 结构中使用 np.intp 类型的字段名列表（见 sklearn/ensemble/_hist_gradient_boosting/common.pyx）
    indexing_field_names = ["feature_idx"]

    # 创建一个字典，其中包含节点数组 dtype 中每个字段名及其对应的数据类型
    new_dtype_dict = {
        name: dtype for name, (dtype, _) in node_ndarray.dtype.fields.items()
    }
    # 将索引字段名列表中的字段名映射到新的数据类型
    for name in indexing_field_names:
        new_dtype_dict[name] = new_dtype_for_indexing_fields

    # 创建新的数据类型对象，包含字段名列表和对应的数据类型列表
    new_dtype = np.dtype(
        {"names": list(new_dtype_dict.keys()), "formats": list(new_dtype_dict.values())}
    )
    return node_ndarray.astype(new_dtype, casting="same_kind")  # 将节点数组转换为新的数据类型


def reduce_predictor_with_different_bitness(predictor):
    cls, args, state = predictor.__reduce__()  # 调用预测器对象的 __reduce__ 方法获取类、参数和状态信息

    new_state = state.copy()  # 复制状态信息
    new_state["nodes"] = get_different_bitness_node_ndarray(new_state["nodes"])  # 更新状态中的节点数组

    return (cls, args, new_state)  # 返回更新后的预测器元组


def test_different_bitness_pickle():
    X, y = make_classification(random_state=0)  # 生成分类数据

    clf = HistGradientBoostingClassifier(random_state=0, max_depth=3)  # 创建梯度提升分类器对象
    clf.fit(X, y)  # 使用分类器拟合数据
    score = clf.score(X, y)  # 计算分类器的得分

    def pickle_dump_with_different_bitness():
        f = io.BytesIO()  # 创建字节流对象
        p = pickle.Pickler(f)  # 创建 Pickler 对象
        p.dispatch_table = copyreg.dispatch_table.copy()  # 复制默认的 dispatch_table

        p.dispatch_table[TreePredictor] = reduce_predictor_with_different_bitness  # 更新 dispatch_table

        p.dump(clf)  # 将分类器对象序列化并写入字节流
        f.seek(0)  # 将字节流指针设置为开头
        return f  # 返回序列化后的字节流对象

    # 模拟加载同一模型的 pickle，在不同位数平台上训练和使用的情况
    new_clf = pickle.load(pickle_dump_with_different_bitness())  # 加载 pickle 文件并反序列化为新的分类器对象
    new_score = new_clf.score(X, y)  # 计算新分类器对象的得分
    assert score == pytest.approx(new_score)  # 断言原分类器对象得分与新分类器对象得分相近


def test_different_bitness_joblib_pickle():
    # 确保在 64 位平台生成的特定平台 pickle 可以在加载时转换为适用于主机的估算器，
    # 当主机为 32 位平台时，可以使用主机的本机整数精度来索引树数据结构中的节点（反之亦然）。
    #
    # 生成一个用于训练和测试的人工数据集，X 是特征集，y 是目标变量集
    X, y = make_classification(random_state=0)

    # 创建一个 HistGradientBoostingClassifier 分类器对象
    clf = HistGradientBoostingClassifier(random_state=0, max_depth=3)
    # 使用生成的数据集 X, y 进行训练
    clf.fit(X, y)
    # 计算分类器在训练数据上的准确率得分
    score = clf.score(X, y)

    # 定义一个函数，用于将分类器 clf 以不同的位数保存到字节流中
    def joblib_dump_with_different_bitness():
        # 创建一个字节流对象
        f = io.BytesIO()
        # 创建一个 NumpyPickler 对象，用于将对象序列化到字节流
        p = NumpyPickler(f)
        # 复制默认的分发表到自定义的分发表中
        p.dispatch_table = copyreg.dispatch_table.copy()
        # 将 TreePredictor 类的序列化函数设置为 reduce_predictor_with_different_bitness
        p.dispatch_table[TreePredictor] = reduce_predictor_with_different_bitness

        # 将分类器 clf 序列化并写入字节流
        p.dump(clf)
        # 将字节流指针位置设置为起始位置
        f.seek(0)
        # 返回包含序列化后数据的字节流
        return f

    # 从保存的字节流中加载分类器，命名为 new_clf
    new_clf = joblib.load(joblib_dump_with_different_bitness())
    # 计算新加载的分类器在同样数据集 X, y 上的准确率得分
    new_score = new_clf.score(X, y)
    # 使用 pytest 断言检查原始分类器和加载后分类器的得分是否近似相等
    assert score == pytest.approx(new_score)
# 定义测试函数，用于验证 Pandas 支持可空数据类型的问题修复
def test_pandas_nullable_dtype():
    # 导入 pytest 库，如果导入失败则跳过测试，确保环境可以正常运行
    pd = pytest.importorskip("pandas")

    # 创建一个随机数生成器对象 rng
    rng = np.random.default_rng(0)
    # 生成一个包含 'a' 列的 Pandas DataFrame，列数据为随机整数（范围 0-9），并将其转换为 Int64 类型
    X = pd.DataFrame({"a": rng.integers(10, size=100)}).astype(pd.Int64Dtype())
    # 生成一个包含 100 个元素的随机整数数组作为分类模型的目标变量 y
    y = rng.integers(2, size=100)

    # 创建一个 HistGradientBoostingClassifier 分类器的实例
    clf = HistGradientBoostingClassifier()
    # 使用 X 和 y 来训练分类器
    clf.fit(X, y)
```