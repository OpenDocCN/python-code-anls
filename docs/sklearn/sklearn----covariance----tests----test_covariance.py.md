# `D:\src\scipysrc\scikit-learn\sklearn\covariance\tests\test_covariance.py`

```
# 导入所需的库
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于单元测试

from sklearn import datasets  # 导入sklearn的datasets模块，用于加载数据集
from sklearn.covariance import (  # 导入协方差估计相关的类和函数
    OAS,
    EmpiricalCovariance,
    LedoitWolf,
    ShrunkCovariance,
    empirical_covariance,
    ledoit_wolf,
    ledoit_wolf_shrinkage,
    oas,
    shrunk_covariance,
)
from sklearn.covariance._shrunk_covariance import _ledoit_wolf  # 导入_ledoit_wolf函数
from sklearn.utils._testing import (  # 导入测试相关的辅助函数
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)

from .._shrunk_covariance import _oas  # 导入_oas函数（相对路径）

# 加载糖尿病数据集，只获取特征矩阵X
X, _ = datasets.load_diabetes(return_X_y=True)
X_1d = X[:, 0]  # 获取X的第一列作为一维数组
n_samples, n_features = X.shape  # 获取数据集的样本数和特征数

def test_covariance():
    """测试协方差模块在简单数据集上的表现。"""

    # 使用经验协方差估计器进行拟合
    cov = EmpiricalCovariance()
    cov.fit(X)
    emp_cov = empirical_covariance(X)
    # 检验拟合后的经验协方差矩阵是否准确到小数点后四位
    assert_array_almost_equal(emp_cov, cov.covariance_, 4)
    # 检验误差范数是否为零
    assert_almost_equal(cov.error_norm(emp_cov), 0)
    assert_almost_equal(cov.error_norm(emp_cov, norm="spectral"), 0)
    assert_almost_equal(cov.error_norm(emp_cov, norm="frobenius"), 0)
    assert_almost_equal(cov.error_norm(emp_cov, scaling=False), 0)
    assert_almost_equal(cov.error_norm(emp_cov, squared=False), 0)
    # 测试对于不支持的范数类型是否引发NotImplementedError异常
    with pytest.raises(NotImplementedError):
        cov.error_norm(emp_cov, norm="foo")
    # 马氏距离计算测试
    mahal_dist = cov.mahalanobis(X)
    assert np.amin(mahal_dist) > 0

    # 测试特征数为1的情况
    X_1d = X[:, 0].reshape((-1, 1))
    cov = EmpiricalCovariance()
    cov.fit(X_1d)
    assert_array_almost_equal(empirical_covariance(X_1d), cov.covariance_, 4)
    assert_almost_equal(cov.error_norm(empirical_covariance(X_1d)), 0)
    assert_almost_equal(cov.error_norm(empirical_covariance(X_1d), norm="spectral"), 0)

    # 测试只有一个样本的情况
    # 创建只有一个样本和5个特征的数据集X_1sample
    X_1sample = np.arange(5).reshape(1, 5)
    cov = EmpiricalCovariance()
    warn_msg = "Only one sample available. You may want to reshape your data array"
    # 检查是否会引发UserWarning并匹配警告消息
    with pytest.warns(UserWarning, match=warn_msg):
        cov.fit(X_1sample)
    # 断言经验协方差矩阵为全零矩阵
    assert_array_almost_equal(cov.covariance_, np.zeros(shape=(5, 5), dtype=np.float64))

    # 测试整数类型数据
    X_integer = np.asarray([[0, 1], [1, 0]])
    result = np.asarray([[0.25, -0.25], [-0.25, 0.25]])
    assert_array_almost_equal(empirical_covariance(X_integer), result)

    # 测试数据已居中的情况
    cov = EmpiricalCovariance(assume_centered=True)
    cov.fit(X)
    assert_array_equal(cov.location_, np.zeros(X.shape[1]))


@pytest.mark.parametrize("n_matrices", [1, 3])
def test_shrunk_covariance_func(n_matrices):
    """检查`shrunk_covariance`函数。"""

    n_features = 2
    cov = np.ones((n_features, n_features))  # 创建一个全一的协方差矩阵
    cov_target = np.array([[1, 0.5], [0.5, 1]])  # 设定目标协方差矩阵
    # 如果 n_matrices 大于 1，则复制 cov 和 cov_target 数组，使其在第一个维度上扩展为 n_matrices 个副本
    if n_matrices > 1:
        cov = np.repeat(cov[np.newaxis, ...], n_matrices, axis=0)
        cov_target = np.repeat(cov_target[np.newaxis, ...], n_matrices, axis=0)

    # 使用 shrunk_covariance 函数对 cov 进行收缩，收缩系数为 0.5
    cov_shrunk = shrunk_covariance(cov, 0.5)
    
    # 断言函数，检查 cov_shrunk 和 cov_target 数组是否在数值上非常接近
    assert_allclose(cov_shrunk, cov_target)
def test_shrunk_covariance():
    """Check consistency between `ShrunkCovariance` and `shrunk_covariance`."""

    # Tests ShrunkCovariance module on a simple dataset.
    # 对简单数据集测试ShrunkCovariance模块

    # compare shrunk covariance obtained from data and from MLE estimate
    # 比较从数据和MLE估计中得到的收缩协方差
    cov = ShrunkCovariance(shrinkage=0.5)
    cov.fit(X)
    assert_array_almost_equal(
        shrunk_covariance(empirical_covariance(X), shrinkage=0.5), cov.covariance_, 4
    )

    # same test with shrinkage not provided
    # 没有提供收缩参数的相同测试
    cov = ShrunkCovariance()
    cov.fit(X)
    assert_array_almost_equal(
        shrunk_covariance(empirical_covariance(X)), cov.covariance_, 4
    )

    # same test with shrinkage = 0 (<==> empirical_covariance)
    # 收缩参数为0的相同测试（等同于经验协方差）
    cov = ShrunkCovariance(shrinkage=0.0)
    cov.fit(X)
    assert_array_almost_equal(empirical_covariance(X), cov.covariance_, 4)

    # test with n_features = 1
    # n_features = 1 的测试
    X_1d = X[:, 0].reshape((-1, 1))
    cov = ShrunkCovariance(shrinkage=0.3)
    cov.fit(X_1d)
    assert_array_almost_equal(empirical_covariance(X_1d), cov.covariance_, 4)

    # test shrinkage coeff on a simple data set (without saving precision)
    # 在简单数据集上测试收缩系数（不保存精度）
    cov = ShrunkCovariance(shrinkage=0.5, store_precision=False)
    cov.fit(X)
    assert cov.precision_ is None


def test_ledoit_wolf():
    # Tests LedoitWolf module on a simple dataset.
    # 对简单数据集测试LedoitWolf模块

    # test shrinkage coeff on a simple data set
    # 在简单数据集上测试收缩系数
    X_centered = X - X.mean(axis=0)
    lw = LedoitWolf(assume_centered=True)
    lw.fit(X_centered)
    shrinkage_ = lw.shrinkage_

    score_ = lw.score(X_centered)
    assert_almost_equal(
        ledoit_wolf_shrinkage(X_centered, assume_centered=True), shrinkage_
    )
    assert_almost_equal(
        ledoit_wolf_shrinkage(X_centered, assume_centered=True, block_size=6),
        shrinkage_,
    )
    # compare shrunk covariance obtained from data and from MLE estimate
    # 比较从数据和MLE估计中得到的收缩协方差
    lw_cov_from_mle, lw_shrinkage_from_mle = ledoit_wolf(
        X_centered, assume_centered=True
    )
    assert_array_almost_equal(lw_cov_from_mle, lw.covariance_, 4)
    assert_almost_equal(lw_shrinkage_from_mle, lw.shrinkage_)
    # compare estimates given by LW and ShrunkCovariance
    # 比较LW和ShrunkCovariance给出的估计
    scov = ShrunkCovariance(shrinkage=lw.shrinkage_, assume_centered=True)
    scov.fit(X_centered)
    assert_array_almost_equal(scov.covariance_, lw.covariance_, 4)

    # test with n_features = 1
    # n_features = 1 的测试
    X_1d = X[:, 0].reshape((-1, 1))
    lw = LedoitWolf(assume_centered=True)
    lw.fit(X_1d)
    lw_cov_from_mle, lw_shrinkage_from_mle = ledoit_wolf(X_1d, assume_centered=True)
    assert_array_almost_equal(lw_cov_from_mle, lw.covariance_, 4)
    assert_almost_equal(lw_shrinkage_from_mle, lw.shrinkage_)
    assert_array_almost_equal((X_1d**2).sum() / n_samples, lw.covariance_, 4)

    # test shrinkage coeff on a simple data set (without saving precision)
    # 在简单数据集上测试收缩系数（不保存精度）
    lw = LedoitWolf(store_precision=False, assume_centered=True)
    lw.fit(X_centered)
    assert_almost_equal(lw.score(X_centered), score_, 4)
    assert lw.precision_ is None

    # Same tests without assuming centered data
    # 不假设中心化数据的相同测试
    # 使用 LedoitWolf 类进行缩减系数在简单数据集上的测试
    lw = LedoitWolf()
    # 拟合数据 X
    lw.fit(X)
    # 断言缩减系数与预期值 shrinkage_ 相近，精确到小数点后四位
    assert_almost_equal(lw.shrinkage_, shrinkage_, 4)
    # 断言缩减系数与 ledoit_wolf_shrinkage 函数计算结果相近，精确到小数点后四位
    assert_almost_equal(lw.shrinkage_, ledoit_wolf_shrinkage(X))
    # 断言缩减系数与 ledoit_wolf 函数计算结果的第二项（缩减系数）相近，精确到小数点后四位
    assert_almost_equal(lw.shrinkage_, ledoit_wolf(X)[1])
    # 断言缩减系数与 _ledoit_wolf 函数给出的结果中的第二项（缩减系数）相近，精确到小数点后四位
    assert_almost_equal(
        lw.shrinkage_, _ledoit_wolf(X=X, assume_centered=False, block_size=10000)[1]
    )
    # 断言通过 LW 计算的分数与预期值 score_ 相近，精确到小数点后四位
    assert_almost_equal(lw.score(X), score_, 4)

    # 比较从数据和从 MLE 估计中获得的收缩协方差
    lw_cov_from_mle, lw_shrinkage_from_mle = ledoit_wolf(X)
    # 断言 LW 计算的协方差矩阵与 lw.covariance_ 相近，精确到小数点后四位
    assert_array_almost_equal(lw_cov_from_mle, lw.covariance_, 4)
    # 断言通过 MLE 估计的缩减系数与 LW 计算的缩减系数相近
    assert_almost_equal(lw_shrinkage_from_mle, lw.shrinkage_)

    # 比较由 LW 和 ShrunkCovariance 给出的估计
    scov = ShrunkCovariance(shrinkage=lw.shrinkage_)
    scov.fit(X)
    # 断言 ShrunkCovariance 计算的协方差矩阵与 LW 计算的协方差矩阵相近，精确到小数点后四位
    assert_array_almost_equal(scov.covariance_, lw.covariance_, 4)

    # 在 n_features = 1 的情况下进行测试
    X_1d = X[:, 0].reshape((-1, 1))
    lw = LedoitWolf()
    lw.fit(X_1d)
    # 断言 X_1d 的方差与 _ledoit_wolf 函数给出的结果中的第一项（方差）相近，不考虑自由度校正
    assert_allclose(
        X_1d.var(ddof=0),
        _ledoit_wolf(X=X_1d, assume_centered=False, block_size=10000)[0],
    )
    # 断言 LW 计算的协方差矩阵与通过 MLE 估计的协方差矩阵相近，精确到小数点后四位
    lw_cov_from_mle, lw_shrinkage_from_mle = ledoit_wolf(X_1d)
    assert_array_almost_equal(lw_cov_from_mle, lw.covariance_, 4)
    # 断言通过 MLE 估计的缩减系数与 LW 计算的缩减系数相近
    assert_almost_equal(lw_shrinkage_from_mle, lw.shrinkage_)
    # 断言通过经验法计算的 X_1d 的协方差矩阵与 LW 计算的协方差矩阵相近，精确到小数点后四位
    assert_array_almost_equal(empirical_covariance(X_1d), lw.covariance_, 4)

    # 在只有一个样本的情况下进行测试
    X_1sample = np.arange(5).reshape(1, 5)
    lw = LedoitWolf()

    # 期望在仅有一个样本时会发出警告
    warn_msg = "Only one sample available. You may want to reshape your data array"
    with pytest.warns(UserWarning, match=warn_msg):
        lw.fit(X_1sample)

    # 断言 LW 计算的协方差矩阵为全零矩阵
    assert_array_almost_equal(lw.covariance_, np.zeros(shape=(5, 5), dtype=np.float64))

    # 在简单数据集上测试缩减系数（不保存精度）
    lw = LedoitWolf(store_precision=False)
    lw.fit(X)
    # 断言 LW 计算的分数与预期值 score_ 相近，精确到小数点后四位
    assert_almost_equal(lw.score(X), score_, 4)
    # 断言 LW 的精确度矩阵为 None
    assert lw.precision_ is None
# 导入所需的库和模块
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance, empirical_covariance
from sklearn.utils.validation import check_array

# Ledoit-Wolf 收缩估计的简单实现函数
def _naive_ledoit_wolf_shrinkage(X):
    # 根据 Ledoit & Wolf 的公式进行计算

    # 获取样本数和特征数
    n_samples, n_features = X.shape
    # 计算经验协方差矩阵
    emp_cov = empirical_covariance(X, assume_centered=False)
    # 计算样本协方差的迹的均值
    mu = np.trace(emp_cov) / n_features
    # 创建协方差矩阵的副本并进行中心化
    delta_ = emp_cov.copy()
    delta_.flat[:: n_features + 1] -= mu
    # 计算 delta 的平方和除以特征数
    delta = (delta_**2).sum() / n_features
    # 计算 X 的平方
    X2 = X**2
    # 计算 beta_
    beta_ = (
        1.0
        / (n_features * n_samples)
        * np.sum(np.dot(X2.T, X2) / n_samples - emp_cov**2)
    )
    # 取 beta_ 和 delta 中较小的一个作为 beta
    beta = min(beta_, delta)
    # 计算收缩系数
    shrinkage = beta / delta
    return shrinkage


# 测试小样本的 Ledoit-Wolf 实现是否与 naive 函数一致
def test_ledoit_wolf_small():
    # 选择前4个特征的子集作为 X_small
    X_small = X[:, :4]
    lw = LedoitWolf()
    lw.fit(X_small)
    shrinkage_ = lw.shrinkage_

    # 断言收缩系数与 naive 函数计算的结果几乎相等
    assert_almost_equal(shrinkage_, _naive_ledoit_wolf_shrinkage(X_small))


# 测试大样本的 Ledoit-Wolf 实现
def test_ledoit_wolf_large():
    # 随机生成一个更宽的数据集 X
    rng = np.random.RandomState(0)
    X = rng.normal(size=(10, 20))
    lw = LedoitWolf(block_size=10).fit(X)
    # 检查协方差矩阵是否接近对角矩阵
    assert_almost_equal(lw.covariance_, np.eye(20), 0)
    cov = lw.covariance_

    # 检查结果是否与不分块数据一致
    lw = LedoitWolf(block_size=25).fit(X)
    assert_almost_equal(lw.covariance_, cov)


# 参数化测试，验证 Ledoit-Wolf 在空数组时是否会引发错误
@pytest.mark.parametrize(
    "ledoit_wolf_fitting_function", [LedoitWolf().fit, _naive_ledoit_wolf_shrinkage]
)
def test_ledoit_wolf_empty_array(ledoit_wolf_fitting_function):
    """检查我们在空数组时是否能正确验证 X 并引发适当的错误。"""
    X_empty = np.zeros((0, 2))
    with pytest.raises(ValueError, match="Found array with 0 sample"):
        ledoit_wolf_fitting_function(X_empty)


# 测试 OAS 模块在简单数据集上的表现
def test_oas():
    # 对中心化后的数据 X_centered 进行测试
    X_centered = X - X.mean(axis=0)
    oa = OAS(assume_centered=True)
    oa.fit(X_centered)
    shrinkage_ = oa.shrinkage_
    score_ = oa.score(X_centered)
    # 比较从数据和从MLE估计获得的收缩协方差
    oa_cov_from_mle, oa_shrinkage_from_mle = oas(X_centered, assume_centered=True)
    assert_array_almost_equal(oa_cov_from_mle, oa.covariance_, 4)
    assert_almost_equal(oa_shrinkage_from_mle, oa.shrinkage_)
    # 比较 OAS 和 ShrunkCovariance 给出的估计结果
    scov = ShrunkCovariance(shrinkage=oa.shrinkage_, assume_centered=True)
    scov.fit(X_centered)
    assert_array_almost_equal(scov.covariance_, oa.covariance_, 4)

    # 对于 n_features = 1 的情况进行测试
    X_1d = X[:, 0:1]
    oa = OAS(assume_centered=True)
    oa.fit(X_1d)
    # 使用 OAS 方法计算数据 X_1d 的样本协方差和收缩后的协方差估计
    oa_cov_from_mle, oa_shrinkage_from_mle = oas(X_1d, assume_centered=True)
    # 断言样本协方差估计与 OAS 类中的协方差矩阵近似相等，精度为四位小数
    assert_array_almost_equal(oa_cov_from_mle, oa.covariance_, 4)
    # 断言收缩系数估计值与 OAS 类中的收缩系数近似相等
    assert_almost_equal(oa_shrinkage_from_mle, oa.shrinkage_)
    # 断言计算得到的数据方差与样本数的比例近似等于 OAS 类中的协方差矩阵，精度为四位小数
    assert_array_almost_equal((X_1d**2).sum() / n_samples, oa.covariance_, 4)

    # 在一个简单数据集上测试收缩系数（不保存精度）
    oa = OAS(store_precision=False, assume_centered=True)
    oa.fit(X_centered)
    # 断言使用 OAS 类计算得分与预期得分近似相等，精度为四位小数
    assert_almost_equal(oa.score(X_centered), score_, 4)
    # 断言 OAS 类的精度矩阵为 None
    assert oa.precision_ is None

    # 在未假定数据已居中的情况下进行相同的测试--------------------------------
    # 在一个简单数据集上测试收缩系数
    oa = OAS()
    oa.fit(X)
    # 断言 OAS 类中的收缩系数近似等于预期收缩系数，精度为四位小数
    assert_almost_equal(oa.shrinkage_, shrinkage_, 4)
    # 断言使用 OAS 类计算得分与预期得分近似相等，精度为四位小数
    assert_almost_equal(oa.score(X), score_, 4)
    # 比较从数据和从 MLE 估计得到的收缩协方差
    oa_cov_from_mle, oa_shrinkage_from_mle = oas(X)
    assert_array_almost_equal(oa_cov_from_mle, oa.covariance_, 4)
    # 断言收缩系数估计与 OAS 类中的收缩系数近似相等
    assert_almost_equal(oa_shrinkage_from_mle, oa.shrinkage_)
    # 比较由 OAS 和 ShrunkCovariance 给出的估计
    scov = ShrunkCovariance(shrinkage=oa.shrinkage_)
    scov.fit(X)
    # 断言由 ShrunkCovariance 计算得到的协方差矩阵与 OAS 类中的协方差矩阵近似相等，精度为四位小数
    assert_array_almost_equal(scov.covariance_, oa.covariance_, 4)

    # 在特征数为 1 时进行测试
    X_1d = X[:, 0].reshape((-1, 1))
    oa = OAS()
    oa.fit(X_1d)
    oa_cov_from_mle, oa_shrinkage_from_mle = oas(X_1d)
    # 断言样本协方差估计与 OAS 类中的协方差矩阵近似相等，精度为四位小数
    assert_array_almost_equal(oa_cov_from_mle, oa.covariance_, 4)
    # 断言收缩系数估计与 OAS 类中的收缩系数近似相等
    assert_almost_equal(oa_shrinkage_from_mle, oa.shrinkage_)
    # 断言样本协方差估计与经验协方差矩阵近似相等，精度为四位小数
    assert_array_almost_equal(empirical_covariance(X_1d), oa.covariance_, 4)

    # 在单个样本上进行测试
    # 当只有 1 个样本时应发出警告
    X_1sample = np.arange(5).reshape(1, 5)
    oa = OAS()
    warn_msg = "Only one sample available. You may want to reshape your data array"
    with pytest.warns(UserWarning, match=warn_msg):
        oa.fit(X_1sample)
    # 断言 OAS 类的协方差矩阵为全零矩阵
    assert_array_almost_equal(oa.covariance_, np.zeros(shape=(5, 5), dtype=np.float64))

    # 在一个简单数据集上测试收缩系数（不保存精度）
    oa = OAS(store_precision=False)
    oa.fit(X)
    # 断言使用 OAS 类计算得分与预期得分近似相等，精度为四位小数
    assert_almost_equal(oa.score(X), score_, 4)
    # 断言 OAS 类的精度矩阵为 None
    assert oa.precision_ is None

    # 在未假定数据已居中的情况下测试函数 _oas
    X_1f = X[:, 0:1]
    oa = OAS()
    oa.fit(X_1f)
    # 比较从数据和从 MLE 估计得到的收缩协方差
    _oa_cov_from_mle, _oa_shrinkage_from_mle = _oas(X_1f)
    # 断言样本协方差估计与 OAS 类中的协方差矩阵近似相等，精度为四位小数
    assert_array_almost_equal(_oa_cov_from_mle, oa.covariance_, 4)
    # 断言收缩系数估计与 OAS 类中的收缩系数近似相等
    assert_almost_equal(_oa_shrinkage_from_mle, oa.shrinkage_)
    # 断言计算得到的数据方差与样本数的比例近似等于 OAS 类中的协方差矩阵，精度为四位小数
    assert_array_almost_equal((X_1f**2).sum() / n_samples, oa.covariance_, 4)
# 定义测试函数，验证 EmpiricalCovariance 对象在使用马氏距离时是否能够有效地验证数据。
def test_EmpiricalCovariance_validates_mahalanobis():
    """Checks that EmpiricalCovariance validates data with mahalanobis."""

    # 使用 EmpiricalCovariance 对象拟合数据 X，计算协方差矩阵
    cov = EmpiricalCovariance().fit(X)

    # 准备匹配的错误消息，用于验证异常是否被正确抛出
    msg = f"X has 2 features, but \\w+ is expecting {X.shape[1]} features as input"

    # 使用 pytest 断言语法检查是否会抛出 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        # 调用 EmpiricalCovariance 对象的 mahalanobis 方法，传入部分数据 X[:, :2]
        cov.mahalanobis(X[:, :2])
```