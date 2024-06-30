# `D:\src\scipysrc\scikit-learn\sklearn\preprocessing\tests\test_common.py`

```
import warnings  # 导入警告模块

import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于测试框架

from sklearn.base import clone  # 导入clone函数，用于克隆对象
from sklearn.datasets import load_iris  # 导入load_iris函数，用于加载鸢尾花数据集
from sklearn.model_selection import train_test_split  # 导入train_test_split函数，用于划分训练集和测试集
from sklearn.preprocessing import (  # 导入数据预处理模块中的各种缩放器和转换器
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    maxabs_scale,
    minmax_scale,
    power_transform,
    quantile_transform,
    robust_scale,
    scale,
)
from sklearn.utils._testing import assert_allclose, assert_array_equal  # 导入用于测试的函数
from sklearn.utils.fixes import (  # 导入修复功能相关的模块
    BSR_CONTAINERS,
    COO_CONTAINERS,
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    DIA_CONTAINERS,
    DOK_CONTAINERS,
    LIL_CONTAINERS,
)

iris = load_iris()  # 加载鸢尾花数据集


def _get_valid_samples_by_column(X, col):
    """Get non NaN samples in column of X"""
    # 获取X中指定列中的非NaN样本
    return X[:, [col]][~np.isnan(X[:, col])]


@pytest.mark.parametrize(  # 使用pytest的parametrize装饰器进行参数化测试
    "est, func, support_sparse, strictly_positive, omit_kwargs",
    [  # 测试用例参数列表
        (MaxAbsScaler(), maxabs_scale, True, False, []),  # 最大绝对值缩放器
        (MinMaxScaler(), minmax_scale, False, False, ["clip"]),  # 最小-最大缩放器
        (StandardScaler(), scale, False, False, []),  # 标准缩放器
        (StandardScaler(with_mean=False), scale, True, False, []),  # 去均值标准缩放器
        (PowerTransformer("yeo-johnson"), power_transform, False, False, []),  # 功率变换器 (Yeo-Johnson)
        (PowerTransformer("box-cox"), power_transform, False, True, []),  # 功率变换器 (Box-Cox)
        (QuantileTransformer(n_quantiles=10), quantile_transform, True, False, []),  # 分位数变换器
        (RobustScaler(), robust_scale, False, False, []),  # 鲁棒缩放器
        (RobustScaler(with_centering=False), robust_scale, True, False, []),  # 不带中心化的鲁棒缩放器
    ],
)
def test_missing_value_handling(
    est, func, support_sparse, strictly_positive, omit_kwargs
):
    # 检查预处理方法是否处理NaN值
    rng = np.random.RandomState(42)  # 创建随机数生成器对象
    X = iris.data.copy()  # 拷贝鸢尾花数据集的特征数据
    n_missing = 50  # 设置缺失值数量
    X[
        rng.randint(X.shape[0], size=n_missing), rng.randint(X.shape[1], size=n_missing)
    ] = np.nan  # 在随机位置生成NaN值
    if strictly_positive:
        X += np.nanmin(X) + 0.1  # 如果需要严格正数，则对数据进行调整
    X_train, X_test = train_test_split(X, random_state=1)  # 划分训练集和测试集
    # 检查点
    assert not np.all(np.isnan(X_train), axis=0).any()  # 确保训练集中没有完全是NaN的列
    assert np.any(np.isnan(X_train), axis=0).all()  # 确保训练集中至少有一列包含NaN
    assert np.any(np.isnan(X_test), axis=0).all()  # 确保测试集中至少有一列包含NaN
    X_test[:, 0] = np.nan  # 确保测试集中的边界情况得到测试

    with warnings.catch_warnings():  # 捕获警告信息
        warnings.simplefilter("error", RuntimeWarning)  # 设置警告过滤器
        Xt = est.fit(X_train).transform(X_test)  # 拟合并转换测试集
    # 确保没有引发警告
    # 缺失值应该仍然缺失，且只有它们
    assert_array_equal(np.isnan(Xt), np.isnan(X_test))

    # 检查函数和类的结果是否一致
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        Xt_class = est.transform(X_train)
    kwargs = est.get_params()  # 获取估计器的参数
    # 删除应省略的参数，因为它们在预处理类的对应函数中没有定义
    for kwarg in omit_kwargs:
        _ = kwargs.pop(kwarg)
    # 对训练集进行函数变换
    Xt_func = func(X_train, **kwargs)
    # 断言两个数组的非NaN部分相等
    assert_array_equal(np.isnan(Xt_func), np.isnan(Xt_class))
    # 断言两个数组的非NaN部分近似相等
    assert_allclose(Xt_func[~np.isnan(Xt_func)], Xt_class[~np.isnan(Xt_class)])

    # 检查逆变换时是否保留NaN值
    Xt_inv = est.inverse_transform(Xt)
    # 断言两个数组的NaN分布相等
    assert_array_equal(np.isnan(Xt_inv), np.isnan(X_test))
    # FIXME: 在最新的numpy版本中可以使用 equal_nan=True。目前我们只检查非NaN值是否几乎相等。
    assert_allclose(Xt_inv[~np.isnan(Xt_inv)], X_test[~np.isnan(X_test)])

    # 遍历数据集的每个特征列
    for i in range(X.shape[1]):
        # 仅在非NaN样本上进行训练
        est.fit(_get_valid_samples_by_column(X_train, i))
        # 检查即使在没有NaN样本训练的情况下，NaN值的转换仍然有效
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            # 使用当前特征列的NaN值进行转换
            Xt_col = est.transform(X_test[:, [i]])
        # 断言转换后的结果与期望值近似相等
        assert_allclose(Xt_col, Xt[:, [i]])
        # 检查处理非NaN值的方式是否与之前一致 - 第一列全部为NaN
        if not np.isnan(X_test[:, i]).all():
            # 使用当前特征列的非NaN样本进行转换
            Xt_col_nonan = est.transform(_get_valid_samples_by_column(X_test, i))
            # 断言处理后的结果与NaN值转换后的结果相等
            assert_array_equal(Xt_col_nonan, Xt_col[~np.isnan(Xt_col.squeeze())])

    # 如果支持稀疏矩阵
    if support_sparse:
        # 克隆一个稠密估计器和一个稀疏估计器
        est_dense = clone(est)
        est_sparse = clone(est)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            # 使用稠密矩阵进行拟合和转换
            Xt_dense = est_dense.fit(X_train).transform(X_test)
            # 使用稠密矩阵进行逆变换
            Xt_inv_dense = est_dense.inverse_transform(Xt_dense)

        # 遍历各种稀疏矩阵容器
        for sparse_container in (
            BSR_CONTAINERS
            + COO_CONTAINERS
            + CSC_CONTAINERS
            + CSR_CONTAINERS
            + DIA_CONTAINERS
            + DOK_CONTAINERS
            + LIL_CONTAINERS
        ):
            # 检查稠密和稀疏输入是否导致相同的结果
            # 预先计算矩阵以避免捕捉到的副作用警告
            X_train_sp = sparse_container(X_train)
            X_test_sp = sparse_container(X_test)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PendingDeprecationWarning)
                warnings.simplefilter("error", RuntimeWarning)
                # 使用稀疏矩阵进行拟合和转换
                Xt_sp = est_sparse.fit(X_train_sp).transform(X_test_sp)

            # 断言稀疏矩阵转换后的稠密表示与稠密矩阵转换的结果近似相等
            assert_allclose(Xt_sp.toarray(), Xt_dense)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PendingDeprecationWarning)
                warnings.simplefilter("error", RuntimeWarning)
                # 使用稀疏矩阵进行逆变换
                Xt_inv_sp = est_sparse.inverse_transform(Xt_sp)

            # 断言稀疏矩阵逆变换后的稠密表示与稠密矩阵逆变换的结果近似相等
            assert_allclose(Xt_inv_sp.toarray(), Xt_inv_dense)
@pytest.mark.parametrize(
    "est, func",
    [  # 参数化测试，定义了多组估算器和相应的转换函数
        (MaxAbsScaler(), maxabs_scale),  # 最大绝对值缩放器和其对应的缩放函数
        (MinMaxScaler(), minmax_scale),  # 最小-最大缩放器和其对应的缩放函数
        (StandardScaler(), scale),  # 标准缩放器和其对应的缩放函数
        (StandardScaler(with_mean=False), scale),  # 关闭均值中心化的标准缩放器和其对应的缩放函数
        (PowerTransformer("yeo-johnson"), power_transform),  # Yeo-Johnson 幂变换器和其对应的幂变换函数
        (
            PowerTransformer("box-cox"),  # Box-Cox 幂变换器
            power_transform,  # 对应的幂变换函数
        ),
        (QuantileTransformer(n_quantiles=3), quantile_transform),  # 分位数转换器和其对应的分位数转换函数
        (RobustScaler(), robust_scale),  # 鲁棒缩放器和其对应的缩放函数
        (RobustScaler(with_centering=False), robust_scale),  # 关闭中心化的鲁棒缩放器和其对应的缩放函数
    ],
)
def test_missing_value_pandas_na_support(est, func):
    # 测试 pandas 的 IntegerArray 中包含 pd.NA 的情况

    pd = pytest.importorskip("pandas")  # 导入 pandas 库，如果不存在则跳过测试

    X = np.array(
        [
            [1, 2, 3, np.nan, np.nan, 4, 5, 1],
            [np.nan, np.nan, 8, 4, 6, np.nan, np.nan, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ]
    ).T

    # 创建包含 IntegerArray 和 pd.NA 的数据框
    X_df = pd.DataFrame(X, dtype="Int16", columns=["a", "b", "c"])
    X_df["c"] = X_df["c"].astype("int")

    X_trans = est.fit_transform(X)  # 对 X 进行拟合和转换
    X_df_trans = est.fit_transform(X_df)  # 对 X_df 进行拟合和转换

    assert_allclose(X_trans, X_df_trans)  # 断言 X_trans 和 X_df_trans 相近
```