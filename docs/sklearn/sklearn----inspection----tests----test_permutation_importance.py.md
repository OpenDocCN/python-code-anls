# `D:\src\scipysrc\scikit-learn\sklearn\inspection\tests\test_permutation_importance.py`

```
```python`
import numpy as np  # 导入 NumPy 库，提供数学计算支持
import pytest  # 导入 pytest 库，用于编写和运行测试
from joblib import parallel_backend  # 从 joblib 导入 parallel_backend，用于并行计算
from numpy.testing import assert_allclose  # 从 numpy.testing 导入 assert_allclose，用于断言浮点数值的近似相等

from sklearn.compose import ColumnTransformer  # 从 sklearn.compose 导入 ColumnTransformer，数据列转换
from sklearn.datasets import (  # 从 sklearn.datasets 导入数据集加载函数
    load_diabetes,
    load_iris,
    make_classification,
    make_regression,
)
from sklearn.dummy import DummyClassifier  # 从 sklearn.dummy 导入 DummyClassifier，空分类器
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # 从 sklearn.ensemble 导入随机森林分类器和回归器
from sklearn.impute import SimpleImputer  # 从 sklearn.impute 导入 SimpleImputer，用于数据填补
from sklearn.inspection import permutation_importance  # 从 sklearn.inspection 导入 permutation_importance，用于特征重要性评估
from sklearn.linear_model import LinearRegression, LogisticRegression  # 从 sklearn.linear_model 导入线性回归和逻辑回归
from sklearn.metrics import (  # 从 sklearn.metrics 导入评估指标
    get_scorer,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split  # 从 sklearn.model_selection 导入 train_test_split，数据集划分
from sklearn.pipeline import make_pipeline  # 从 sklearn.pipeline 导入 make_pipeline，创建管道
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler, scale  # 从 sklearn.preprocessing 导入数据预处理工具
from sklearn.utils._testing import _convert_container  # 从 sklearn.utils._testing 导入 _convert_container，用于测试数据转换


@pytest.mark.parametrize("n_jobs", [1, 2])  # 使用 pytest.mark.parametrize 装饰器，设置 n_jobs 参数的多种取值
@pytest.mark.parametrize("max_samples", [0.5, 1.0])  # 使用 pytest.mark.parametrize 装饰器，设置 max_samples 参数的多种取值
@pytest.mark.parametrize("sample_weight", [None, "ones"])  # 使用 pytest.mark.parametrize 装饰器，设置 sample_weight 参数的多种取值
def test_permutation_importance_correlated_feature_regression(
    n_jobs, max_samples, sample_weight
):
    # 确保与目标变量高度相关的特征具有更高的特征重要性
    rng = np.random.RandomState(42)  # 创建一个随机数生成器，种子设置为 42
    n_repeats = 5  # 设置重复次数

    X, y = load_diabetes(return_X_y=True)  # 加载糖尿病数据集，返回特征矩阵 X 和目标变量 y
    y_with_little_noise = (y + rng.normal(scale=0.001, size=y.shape[0])).reshape(-1, 1)  # 添加一些噪声到 y

    X = np.hstack([X, y_with_little_noise])  # 将噪声特征添加到特征矩阵 X 的最后一列

    weights = np.ones_like(y) if sample_weight == "ones" else sample_weight  # 根据 sample_weight 设置样本权重
    clf = RandomForestRegressor(n_estimators=10, random_state=42)  # 创建随机森林回归器，决策树数量为 10
    clf.fit(X, y)  # 拟合模型

    result = permutation_importance(
        clf,
        X,
        y,
        sample_weight=weights,
        n_repeats=n_repeats,
        random_state=rng,
        n_jobs=n_jobs,
        max_samples=max_samples,
    )  # 计算特征重要性

    assert result.importances.shape == (X.shape[1], n_repeats)  # 断言特征重要性矩阵的形状

    # 检查与 y 高度相关的特征是否具有最高的平均重要性
    assert np.all(result.importances_mean[-1] > result.importances_mean[:-1])


@pytest.mark.parametrize("n_jobs", [1, 2])  # 使用 pytest.mark.parametrize 装饰器，设置 n_jobs 参数的多种取值
@pytest.mark.parametrize("max_samples", [0.5, 1.0])  # 使用 pytest.mark.parametrize 装饰器，设置 max_samples 参数的多种取值
def test_permutation_importance_correlated_feature_regression_pandas(
    n_jobs, max_samples
):
    pd = pytest.importorskip("pandas")  # 导入 pandas 库，若导入失败则跳过测试

    # 确保与目标变量高度相关的特征具有更高的特征重要性
    rng = np.random.RandomState(42)  # 创建一个随机数生成器，种子设置为 42
    n_repeats = 5  # 设置重复次数

    dataset = load_iris()  # 加载鸢尾花数据集
    X, y = dataset.data, dataset.target  # 获取特征矩阵 X 和目标变量 y
    y_with_little_noise = (y + rng.normal(scale=0.001, size=y.shape[0])).reshape(-1, 1)  # 添加一些噪声到 y

    # 将数据转换为 pandas DataFrame，并添加高度相关的特征
    X = pd.DataFrame(X, columns=dataset.feature_names)  # 创建 DataFrame，列名为特征名
    X["correlated_feature"] = y_with_little_noise  # 添加高度相关的特征列

    clf = RandomForestClassifier(n_estimators=10, random_state=42)  # 创建随机森林分类器，决策树数量为 10
    clf.fit(X, y)  # 拟合模型
    # 使用 permutation_importance 函数计算特征重要性
    result = permutation_importance(
        clf,                    # 分类器模型
        X,                      # 特征数据
        y,                      # 目标数据
        n_repeats=n_repeats,    # 重复次数
        random_state=rng,       # 随机数种子
        n_jobs=n_jobs,          # 并行计算的任务数
        max_samples=max_samples, # 每次排列的最大样本数
    )

    # 断言检查特征重要性结果的形状是否符合预期
    assert result.importances.shape == (X.shape[1], n_repeats)

    # 断言检查最后一列添加的与目标变量 y 相关的特征具有最高的重要性
    # 应该保证最后一列的平均重要性大于其它所有特征的平均重要性
    assert np.all(result.importances_mean[-1] > result.importances_mean[:-1])
@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("max_samples", [0.5, 1.0])
def test_robustness_to_high_cardinality_noisy_feature(n_jobs, max_samples, seed=42):
    # Permutation variable importance should not be affected by the high
    # cardinality bias of traditional feature importances, especially when
    # computed on a held-out test set:

    # 使用种子初始化随机数生成器
    rng = np.random.RandomState(seed)

    # 定义重复次数和样本数
    n_repeats = 5
    n_samples = 1000
    n_classes = 5
    n_informative_features = 2
    n_noise_features = 1
    n_features = n_informative_features + n_noise_features

    # 生成一个多类别分类数据集，并创建一组信息二进制特征，用于准确预测某些类别
    classes = np.arange(n_classes)
    y = rng.choice(classes, size=n_samples)
    X = np.hstack([(y == c).reshape(-1, 1) for c in classes[:n_informative_features]])
    X = X.astype(np.float32)

    # 不是所有目标类别都由二进制类别指示器特征解释：
    assert n_informative_features < n_classes

    # 添加其他 10 个高基数（数值）的噪声特征，可用于过度拟合训练数据
    X = np.concatenate([X, rng.randn(n_samples, n_noise_features)], axis=1)
    assert X.shape == (n_samples, n_features)

    # 分割数据集以便在保留的测试集上评估。测试集大小应足够大，以确保重要性测量稳定：
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=rng
    )

    # 使用随机森林分类器拟合训练数据
    clf = RandomForestClassifier(n_estimators=5, random_state=rng)
    clf.fit(X_train, y_train)

    # 树节点上使用基于不纯度减少的变量重要性常常会利用噪声特征进行分割。
    # 这可能会给出错误的印象，认为高基数噪声变量是最重要的：
    tree_importances = clf.feature_importances_
    informative_tree_importances = tree_importances[:n_informative_features]
    noisy_tree_importances = tree_importances[n_informative_features:]
    assert informative_tree_importances.max() < noisy_tree_importances.min()

    # 检查置换重要性是否解决了这个问题。
    r = permutation_importance(
        clf,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=rng,
        n_jobs=n_jobs,
        max_samples=max_samples,
    )

    assert r.importances.shape == (X.shape[1], n_repeats)

    # 将重要性分为信息特征和噪声特征
    informative_importances = r.importances_mean[:n_informative_features]
    noisy_importances = r.importances_mean[n_informative_features:]

    # 因为没有一个二进制变量可以解释每个目标类别，RF 模型将不得不使用随机变量来做出某些
    # 确保噪声特征的绝对值大于 1e-7，以避免过拟合，因为最大深度未设置。
    assert max(np.abs(noisy_importances)) > 1e-7
    # 确保噪声特征的最大值小于 0.05，以确保它们具有较小的振荡值。
    assert noisy_importances.max() < 0.05

    # 信息特征的重要性应该比高基数噪声特征更高。
    # 每个信息特征贡献的准确率大约为 0.2，最大测试准确率为 0.4。
    assert informative_importances.min() > 0.15
# 测试排列重要性（permutation importance）对混合类型数据的影响
def test_permutation_importance_mixed_types():
    # 创建随机数生成器对象，并设定种子为42
    rng = np.random.RandomState(42)
    # 设定重复次数为4
    n_repeats = 4

    # 创建包含缺失值的二维数组 X，其中最后一列与 y 相关联
    X = np.array([[1.0, 2.0, 3.0, np.nan], [2, 1, 2, 1]]).T
    # 创建一维数组 y
    y = np.array([0, 1, 0, 1])

    # 使用简单填充器（SimpleImputer）和逻辑回归作为管道创建分类器
    clf = make_pipeline(SimpleImputer(), LogisticRegression(solver="lbfgs"))
    # 使用 X, y 训练分类器
    clf.fit(X, y)
    # 计算排列重要性结果
    result = permutation_importance(clf, X, y, n_repeats=n_repeats, random_state=rng)

    # 断言结果的重要性矩阵的形状符合预期
    assert result.importances.shape == (X.shape[1], n_repeats)

    # 最后一列与 y 相关联，应该具有最高的重要性
    assert np.all(result.importances_mean[-1] > result.importances_mean[:-1])

    # 使用另一个随机状态
    rng = np.random.RandomState(0)
    # 使用不同随机状态再次计算排列重要性结果
    result2 = permutation_importance(clf, X, y, n_repeats=n_repeats, random_state=rng)
    # 断言结果的重要性矩阵的形状符合预期
    assert result2.importances.shape == (X.shape[1], n_repeats)

    # 断言两次计算的重要性结果不完全相等
    assert not np.allclose(result.importances, result2.importances)

    # 最后一列与 y 相关联，应该具有最高的重要性
    assert np.all(result2.importances_mean[-1] > result2.importances_mean[:-1])


# 测试对 Pandas 数据框进行排列重要性分析
def test_permutation_importance_mixed_types_pandas():
    # 导入 pytest，并确保导入成功
    pd = pytest.importorskip("pandas")
    # 创建随机数生成器对象，并设定种子为42
    rng = np.random.RandomState(42)
    # 设定重复次数为5
    n_repeats = 5

    # 创建包含缺失值的 Pandas 数据框 X，其中最后一列与 y 相关联
    X = pd.DataFrame({"col1": [1.0, 2.0, 3.0, np.nan], "col2": ["a", "b", "a", "b"]})
    # 创建一维数组 y
    y = np.array([0, 1, 0, 1])

    # 创建数值类型预处理管道
    num_preprocess = make_pipeline(SimpleImputer(), StandardScaler())
    # 创建列变换器，并指定数值列和分类列的预处理方式
    preprocess = ColumnTransformer(
        [("num", num_preprocess, ["col1"]), ("cat", OneHotEncoder(), ["col2"])]
    )
    # 使用预处理管道和逻辑回归创建分类器
    clf = make_pipeline(preprocess, LogisticRegression(solver="lbfgs"))
    # 使用 X, y 训练分类器
    clf.fit(X, y)

    # 计算排列重要性结果
    result = permutation_importance(clf, X, y, n_repeats=n_repeats, random_state=rng)

    # 断言结果的重要性矩阵的形状符合预期
    assert result.importances.shape == (X.shape[1], n_repeats)

    # 最后一列与 y 相关联，应该具有最高的重要性
    assert np.all(result.importances_mean[-1] > result.importances_mean[:-1])


# 测试线性回归模型的排列重要性
def test_permutation_importance_linear_regresssion():
    # 生成包含500个样本和10个特征的回归数据集
    X, y = make_regression(n_samples=500, n_features=10, random_state=0)

    # 对特征矩阵 X 和目标变量 y 进行缩放
    X = scale(X)
    y = scale(y)

    # 使用线性回归拟合数据
    lr = LinearRegression().fit(X, y)

    # 期望的特征重要性可以通过公式计算得出
    expected_importances = 2 * lr.coef_**2
    # 计算排列重要性结果
    results = permutation_importance(
        lr, X, y, n_repeats=50, scoring="neg_mean_squared_error"
    )
    # 断言计算得到的重要性与预期的重要性接近
    assert_allclose(
        expected_importances, results.importances_mean, rtol=1e-1, atol=1e-6
    )


# 使用参数化测试确保顺序调用和并行调用的排列重要性结果一致
@pytest.mark.parametrize("max_samples", [500, 1.0])
def test_permutation_importance_equivalence_sequential_parallel(max_samples):
    # 回归测试，确保顺序调用和并行调用的排列重要性结果相同
    # 同时测试当 max_samples 等于样本数时，结果等同于 1.0 的情况
    X, y = make_regression(n_samples=500, n_features=10, random_state=0)
    lr = LinearRegression().fit(X, y)
    # 使用 permutation_importance 函数计算特征重要性，使用顺序方式运行
    importance_sequential = permutation_importance(
        lr, X, y, n_repeats=5, random_state=0, n_jobs=1, max_samples=max_samples
    )

    # 首先检查问题是否足够结构化，并且模型复杂度足够高，以避免产生平凡的常数重要性：
    imp_min = importance_sequential["importances"].min()
    imp_max = importance_sequential["importances"].max()
    assert imp_max - imp_min > 0.3

    # 实际上检查并行化是否影响结果，无论是使用共享内存（线程化）还是通过基于进程的独立内存进行并行化，
    # 使用默认的后端（'loky' 或 'multiprocessing'），具体取决于 joblib 版本：

    # 基于进程的并行化（默认）：
    importance_processes = permutation_importance(
        lr, X, y, n_repeats=5, random_state=0, n_jobs=2
    )
    assert_allclose(
        importance_processes["importances"], importance_sequential["importances"]
    )

    # 基于线程的并行化：
    with parallel_backend("threading"):
        importance_threading = permutation_importance(
            lr, X, y, n_repeats=5, random_state=0, n_jobs=2
        )
    assert_allclose(
        importance_threading["importances"], importance_sequential["importances"]
    )
# 使用 pytest 的参数化功能，定义测试函数 test_permutation_importance_equivalence_array_dataframe，分别测试不同的参数组合
@pytest.mark.parametrize("n_jobs", [None, 1, 2])
@pytest.mark.parametrize("max_samples", [0.5, 1.0])
def test_permutation_importance_equivalence_array_dataframe(n_jobs, max_samples):
    # 此测试检查列重排逻辑在 dataframe 和简单的 numpy 数组上具有相同的行为
    pd = pytest.importorskip("pandas")  # 导入 pandas 库，如果导入失败则跳过测试

    # 回归测试，确保顺序调用和并行调用产生相同的结果
    X, y = make_regression(n_samples=100, n_features=5, random_state=0)
    X_df = pd.DataFrame(X)  # 将 numpy 数组 X 转换为 pandas DataFrame

    # 添加一个与 y 统计上相关的分类特征：
    binner = KBinsDiscretizer(n_bins=3, encode="ordinal")
    cat_column = binner.fit_transform(y.reshape(-1, 1))

    # 将额外的列连接到 numpy 数组 X：整数将被转换为浮点数值
    X = np.hstack([X, cat_column])
    assert X.dtype.kind == "f"  # 断言 X 的数据类型为浮点数

    # 将额外的列插入到 DataFrame 中作为非 numpy 本地数据类型（同时保持对旧 pandas 版本的向后兼容性）：
    if hasattr(pd, "Categorical"):
        cat_column = pd.Categorical(cat_column.ravel())
    else:
        cat_column = cat_column.ravel()
    new_col_idx = len(X_df.columns)
    X_df[new_col_idx] = cat_column
    assert X_df[new_col_idx].dtype == cat_column.dtype  # 断言新插入的列数据类型与 cat_column 的数据类型相同

    # 将任意索引附加到 DataFrame：
    X_df.index = np.arange(len(X_df)).astype(str)

    # 使用随机森林回归器训练模型
    rf = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=0)
    rf.fit(X, y)

    n_repeats = 3
    # 计算特征重要性的置换重要性，使用 numpy 数组作为输入
    importance_array = permutation_importance(
        rf,
        X,
        y,
        n_repeats=n_repeats,
        random_state=0,
        n_jobs=n_jobs,
        max_samples=max_samples,
    )

    # 首先检查问题结构是否足够复杂，模型是否足够复杂，确保不会产生微不足道的常数重要性：
    imp_min = importance_array["importances"].min()
    imp_max = importance_array["importances"].max()
    assert imp_max - imp_min > 0.3

    # 现在检查在 DataFrame 上计算的重要性与在相同数据的数组上计算的值是否匹配：
    importance_dataframe = permutation_importance(
        rf,
        X_df,
        y,
        n_repeats=n_repeats,
        random_state=0,
        n_jobs=n_jobs,
        max_samples=max_samples,
    )
    assert_allclose(
        importance_array["importances"], importance_dataframe["importances"]
    )


# 使用 pytest 的参数化功能，定义测试函数 test_permutation_importance_large_memmaped_data，测试不同输入类型的情况
@pytest.mark.parametrize("input_type", ["array", "dataframe"])
def test_permutation_importance_large_memmaped_data(input_type):
    # 烟雾测试，非回归测试，用于验证：
    # https://github.com/scikit-learn/scikit-learn/issues/15810
    n_samples, n_features = int(5e4), 4
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, random_state=0
    )
    assert X.nbytes > 1e6  # 触发 joblib 的内存映射

    X = _convert_container(X, input_type)  # 将 X 转换为指定的输入类型
    clf = DummyClassifier(strategy="prior").fit(X, y)

    # 实际烟雾测试：不应该引发任何错误
    # 定义重复次数
    n_repeats = 5
    # 使用 permutation_importance 函数计算特征重要性，利用 clf 分类器、特征矩阵 X 和目标向量 y，
    # 设定重复次数为 n_repeats，同时使用 2 个工作线程进行处理
    r = permutation_importance(clf, X, y, n_repeats=n_repeats, n_jobs=2)

    # 辅助检查：DummyClassifier 是特征独立的，
    # 对特征进行置换不应该改变预测结果
    # 预期的特征重要性矩阵，维度为 (特征数, 重复次数)，初始化为全零数组
    expected_importances = np.zeros((n_features, n_repeats))
    # 使用 assert_allclose 函数检查预期的特征重要性和实际计算得到的 r.importances 是否相等
    assert_allclose(expected_importances, r.importances)
def test_permutation_importance_sample_weight():
    # 创建包含2个特征和1000个样本的数据，其中目标变量是这两个特征的线性组合，
    # 使得在一半的样本中，特征1的影响是特征2的两倍，另一半相反。
    rng = np.random.RandomState(1)
    n_samples = 1000
    n_features = 2
    n_half_samples = n_samples // 2
    x = rng.normal(0.0, 0.001, (n_samples, n_features))
    y = np.zeros(n_samples)
    y[:n_half_samples] = 2 * x[:n_half_samples, 0] + x[:n_half_samples, 1]
    y[n_half_samples:] = x[n_half_samples:, 0] + 2 * x[n_half_samples:, 1]

    # 使用无拟合截距的线性回归模型进行拟合
    lr = LinearRegression(fit_intercept=False)
    lr.fit(x, y)

    # 当所有样本的权重相同时，特征重要性的比率应该接近1（使用平均绝对误差作为损失函数）
    pi = permutation_importance(
        lr, x, y, random_state=1, scoring="neg_mean_absolute_error", n_repeats=200
    )
    x1_x2_imp_ratio_w_none = pi.importances_mean[0] / pi.importances_mean[1]
    assert x1_x2_imp_ratio_w_none == pytest.approx(1, 0.01)

    # 当将权重向量设为全1时，结果应与未设置样本权重时相同
    w = np.ones(n_samples)
    pi = permutation_importance(
        lr,
        x,
        y,
        random_state=1,
        scoring="neg_mean_absolute_error",
        n_repeats=200,
        sample_weight=w,
    )
    x1_x2_imp_ratio_w_ones = pi.importances_mean[0] / pi.importances_mean[1]
    assert x1_x2_imp_ratio_w_ones == pytest.approx(x1_x2_imp_ratio_w_none, 0.01)

    # 当前半部分样本的权重接近无穷大，后半部分样本的权重为1时，
    # 特征重要性的比率应接近2（使用平均绝对误差作为损失函数）
    w = np.hstack([np.repeat(10.0**10, n_half_samples), np.repeat(1.0, n_half_samples)])
    lr.fit(x, y, w)
    pi = permutation_importance(
        lr,
        x,
        y,
        random_state=1,
        scoring="neg_mean_absolute_error",
        n_repeats=200,
        sample_weight=w,
    )
    x1_x2_imp_ratio_w = pi.importances_mean[0] / pi.importances_mean[1]
    assert x1_x2_imp_ratio_w / x1_x2_imp_ratio_w_none == pytest.approx(2, 0.01)


def test_permutation_importance_no_weights_scoring_function():
    # 创建一个不接受样本权重的评分函数
    def my_scorer(estimator, X, y):
        return 1

    # 创建一些数据和用于置换测试的估计器
    x = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])
    w = np.array([1, 1])
    lr = LinearRegression()
    lr.fit(x, y)

    # 测试当样本权重为None时，置换重要性不返回错误
    try:
        # 尝试运行 permutation_importance 函数，计算特征重要性
        permutation_importance(lr, x, y, random_state=1, scoring=my_scorer, n_repeats=1)
    except TypeError:
        # 如果捕获到 TypeError 异常，则使用 pytest.fail 报错，说明 permutation_test 在使用一个不接受样本权重的评分函数时抛出了异常
        pytest.fail(
            "permutation_test raised an error when using a scorer "
            "function that does not accept sample_weight even though "
            "sample_weight was None"
        )

    # 测试 permutation_importance 函数在 sample_weight 不为 None 时是否会抛出 TypeError 异常
    with pytest.raises(TypeError):
        permutation_importance(
            lr, x, y, random_state=1, scoring=my_scorer, n_repeats=1, sample_weight=w
        )
@pytest.mark.parametrize(
    "list_single_scorer, multi_scorer",
    [  # 参数化测试用例，包括单一和多个评分器的组合
        (["r2", "neg_mean_squared_error"], ["r2", "neg_mean_squared_error"]),  # 第一组参数化测试用例：两个列表形式的评分器
        (
            ["r2", "neg_mean_squared_error"],
            {  # 第二组参数化测试用例：一个字典，包含评分器名称和对应的评分器对象
                "r2": get_scorer("r2"),
                "neg_mean_squared_error": get_scorer("neg_mean_squared_error"),
            },
        ),
        (
            ["r2", "neg_mean_squared_error"],
            lambda estimator, X, y: {  # 第三组参数化测试用例：lambda 函数，返回评分结果字典
                "r2": r2_score(y, estimator.predict(X)),
                "neg_mean_squared_error": -mean_squared_error(y, estimator.predict(X)),
            },
        ),
    ],
)
def test_permutation_importance_multi_metric(list_single_scorer, multi_scorer):
    # 测试多指标情况下的置换重要性

    # 创建用于置换测试的数据和评估器
    x, y = make_regression(n_samples=500, n_features=10, random_state=0)
    lr = LinearRegression().fit(x, y)

    # 进行多指标的置换重要性计算
    multi_importance = permutation_importance(
        lr, x, y, random_state=1, scoring=multi_scorer, n_repeats=2
    )
    assert set(multi_importance.keys()) == set(list_single_scorer)

    # 遍历单一评分器列表，比较结果
    for scorer in list_single_scorer:
        multi_result = multi_importance[scorer]
        single_result = permutation_importance(
            lr, x, y, random_state=1, scoring=scorer, n_repeats=2
        )

        assert_allclose(multi_result.importances, single_result.importances)


def test_permutation_importance_max_samples_error():
    """检查当 `max_samples` 设置为无效值时是否会引发正确的错误消息。"""
    X = np.array([(1.0, 2.0, 3.0, 4.0)]).T
    y = np.array([0, 1, 0, 1])

    clf = LogisticRegression()
    clf.fit(X, y)

    err_msg = r"max_samples must be <= n_samples"

    # 使用 pytest 的断言检查是否会引发 ValueError，并匹配特定的错误消息
    with pytest.raises(ValueError, match=err_msg):
        permutation_importance(clf, X, y, max_samples=5)
```