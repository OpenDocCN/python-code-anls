# `D:\src\scipysrc\scikit-learn\sklearn\covariance\_elliptic_envelope.py`

```
# 导入需要的模块和类
from numbers import Real  # 从 numbers 模块导入 Real 类型
import numpy as np  # 导入 NumPy 库，并使用 np 别名

# 导入父类和相关模块
from ..base import OutlierMixin, _fit_context  # 从当前包的 base 模块中导入 OutlierMixin 和 _fit_context 类
from ..metrics import accuracy_score  # 从当前包的 metrics 模块导入 accuracy_score 函数
from ..utils._param_validation import Interval  # 从当前包的 utils 模块中的 _param_validation 导入 Interval 类
from ..utils.validation import check_is_fitted  # 从当前包的 utils 模块中导入 check_is_fitted 函数
from ._robust_covariance import MinCovDet  # 从当前包的 _robust_covariance 模块导入 MinCovDet 类


class EllipticEnvelope(OutlierMixin, MinCovDet):
    """用于检测高斯分布数据集中异常值的对象。

    更多信息请参考 :ref:`用户指南 <outlier_detection>`。

    Parameters
    ----------
    store_precision : bool, default=True
        指定是否存储估计的精度。

    assume_centered : bool, default=False
        如果为 True，则计算健壮位置和协方差估计的支持，并从中重新计算协方差估计，而不是将数据居中。
        对于平均值显著等于零但不完全为零的数据非常有用。
        如果为 False，则直接使用 FastMCD 算法计算健壮位置和协方差，无需额外处理。

    support_fraction : float, default=None
        原始 MCD 估计支持中应包含的点的比例。如果为 None，则算法将使用支持分数的最小值：
        `(n_samples + n_features + 1) / 2 * n_samples`。
        范围为 (0, 1)。

    contamination : float, default=0.1
        数据集中的污染量，即数据集中异常值的比例。范围为 (0, 0.5]。

    random_state : int, RandomState instance or None, default=None
        确定用于对数据进行洗牌的伪随机数生成器。
        传递一个 int 可以确保多次函数调用时产生可复现的结果。
        参见 :term:`术语表 <random_state>`。

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        估计的健壮位置。

    covariance_ : ndarray of shape (n_features, n_features)
        估计的健壮协方差矩阵。

    precision_ : ndarray of shape (n_features, n_features)
        估计的伪逆矩阵。
        (仅在 store_precision 为 True 时存储)

    support_ : ndarray of shape (n_samples,)
        用于计算健壮位置和形状估计的观测点的掩码。

    offset_ : float
        用于从原始分数定义决策函数的偏移量。
        我们有关系：``decision_function = score_samples - offset_``。
        偏移量取决于污染参数，并且定义了我们在训练中获得预期数量的异常值（具有决策函数 < 0 的样本）。

        .. versionadded:: 0.20

    raw_location_ : ndarray of shape (n_features,)
        校正和重新加权之前的原始健壮估计位置。
    """
    # 存储未校正和重新加权之前的原始鲁棒估计协方差矩阵，形状为 (n_features, n_features)
    raw_covariance_ : ndarray of shape (n_features, n_features)
    
    # 存储在校正和重新加权之前用于计算位置和形状的原始鲁棒估计的观测值的掩码，形状为 (n_samples,)
    raw_support_ : ndarray of shape (n_samples,)
    
    # 训练集上观测值的马氏距离（在调用 :meth:`fit` 方法时计算）
    dist_ : ndarray of shape (n_samples,)
    
    # 在 `fit` 过程中看到的特征数量
    n_features_in_ : int
    
    # 在 `fit` 过程中看到的特征名称数组，仅在 `X` 的特征名全为字符串时定义
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
    
    # 相关工具和参考信息
    
    # 相关工具：最大似然协方差估计器
    EmpiricalCovariance : Maximum likelihood covariance estimator.
    
    # 相关工具：带有 l1 惩罚估计器的稀疏逆协方差估计
    GraphicalLasso : Sparse inverse covariance estimation with an l1-penalized estimator.
    
    # 相关工具：LedoitWolf 估计器
    LedoitWolf : LedoitWolf Estimator.
    
    # 相关工具：最小协方差行列式（鲁棒协方差估计器）
    MinCovDet : Minimum Covariance Determinant (robust estimator of covariance).
    
    # 相关工具：Oracle 近似收缩估计器
    OAS : Oracle Approximating Shrinkage Estimator.
    
    # 相关工具：带有收缩的协方差估计器
    ShrunkCovariance : Covariance estimator with shrinkage.
    
    # 注意事项
    Outlier detection from covariance estimation may break or not perform well in high-dimensional settings. 
    In particular, one will always take care to work with ``n_samples > n_features ** 2``.
    
    # 参考文献
    Rousseeuw, P.J., Van Driessen, K. "A fast algorithm for the minimum covariance determinant estimator" Technometrics 41(3), 212 (1999)
    
    # 示例
    >>> import numpy as np
    >>> from sklearn.covariance import EllipticEnvelope
    >>> true_cov = np.array([[.8, .3],
    ...                      [.3, .4]])
    >>> X = np.random.RandomState(0).multivariate_normal(mean=[0, 0],
    ...                                                  cov=true_cov,
    ...                                                  size=500)
    >>> cov = EllipticEnvelope(random_state=0).fit(X)
    >>> # predict returns 1 for an inlier and -1 for an outlier
    >>> cov.predict([[0, 0],
    ...              [3, 3]])
    array([ 1, -1])
    >>> cov.covariance_
    array([[0.7411..., 0.2535...],
           [0.2535..., 0.3053...]])
    >>> cov.location_
    array([0.0813... , 0.0427...])
    """
    
    # 存储参数约束的字典，继承自 `MinCovDet._parameter_constraints`，添加了 "contamination" 参数的约束
    _parameter_constraints: dict = {
        **MinCovDet._parameter_constraints,
        "contamination": [Interval(Real, 0, 0.5, closed="right")],
    }
    
    # 初始化方法，定义了一些参数
    def __init__(
        self,
        *,
        store_precision=True,
        assume_centered=False,
        support_fraction=None,
        contamination=0.1,
        random_state=None,
    ):
        super().__init__(
            store_precision=store_precision,
            assume_centered=assume_centered,
            support_fraction=support_fraction,
            random_state=random_state,
        )
        self.contamination = contamination


# 调用父类的初始化方法，设置参数以及污染率
super().__init__(
    store_precision=store_precision,
    assume_centered=assume_centered,
    support_fraction=support_fraction,
    random_state=random_state,
)
# 设置当前对象的污染率
self.contamination = contamination



    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the EllipticEnvelope model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        super().fit(X)
        self.offset_ = np.percentile(-self.dist_, 100.0 * self.contamination)
        return self


# 使用装饰器设置上下文为fit方法，使得可以选择跳过嵌套验证
@_fit_context(prefer_skip_nested_validation=True)
# 定义fit方法，用于训练EllipticEnvelope模型
def fit(self, X, y=None):
    """Fit the EllipticEnvelope model.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data.

    y : Ignored
        Not used, present for API consistency by convention.

    Returns
    -------
    self : object
        Returns the instance itself.
    """
    # 调用父类的fit方法，传入训练数据X
    super().fit(X)
    # 计算偏移量，使用负的self.dist_的百分位数来设置
    self.offset_ = np.percentile(-self.dist_, 100.0 * self.contamination)
    return self



    def decision_function(self, X):
        """Compute the decision function of the given observations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        decision : ndarray of shape (n_samples,)
            Decision function of the samples.
            It is equal to the shifted Mahalanobis distances.
            The threshold for being an outlier is 0, which ensures a
            compatibility with other outlier detection algorithms.
        """
        check_is_fitted(self)
        negative_mahal_dist = self.score_samples(X)
        return negative_mahal_dist - self.offset_


# 定义decision_function方法，计算给定观测的决策函数值
def decision_function(self, X):
    """Compute the decision function of the given observations.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data matrix.

    Returns
    -------
    decision : ndarray of shape (n_samples,)
        Decision function of the samples.
        It is equal to the shifted Mahalanobis distances.
        The threshold for being an outlier is 0, which ensures a
        compatibility with other outlier detection algorithms.
    """
    # 检查模型是否已拟合
    check_is_fitted(self)
    # 计算负的马氏距离
    negative_mahal_dist = self.score_samples(X)
    # 返回负的马氏距离减去偏移量的结果
    return negative_mahal_dist - self.offset_



    def score_samples(self, X):
        """Compute the negative Mahalanobis distances.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        negative_mahal_distances : array-like of shape (n_samples,)
            Opposite of the Mahalanobis distances.
        """
        check_is_fitted(self)
        return -self.mahalanobis(X)


# 定义score_samples方法，计算负的马氏距离
def score_samples(self, X):
    """Compute the negative Mahalanobis distances.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data matrix.

    Returns
    -------
    negative_mahal_distances : array-like of shape (n_samples,)
        Opposite of the Mahalanobis distances.
    """
    # 检查模型是否已拟合
    check_is_fitted(self)
    # 返回负的马氏距离
    return -self.mahalanobis(X)



    def predict(self, X):
        """
        Predict labels (1 inlier, -1 outlier) of X according to fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            Returns -1 for anomalies/outliers and +1 for inliers.
        """
        values = self.decision_function(X)
        is_inlier = np.full(values.shape[0], -1, dtype=int)
        is_inlier[values >= 0] = 1

        return is_inlier


# 定义predict方法，根据拟合的模型预测X的标签（1表示正常值，-1表示异常值）
def predict(self, X):
    """
    Predict labels (1 inlier, -1 outlier) of X according to fitted model.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data matrix.

    Returns
    -------
    is_inlier : ndarray of shape (n_samples,)
        Returns -1 for anomalies/outliers and +1 for inliers.
    """
    # 计算决策函数值
    values = self.decision_function(X)
    # 初始化is_inlier为全为-1的数组，数据类型为整数
    is_inlier = np.full(values.shape[0], -1, dtype=int)
    # 将决策函数值大于等于0的位置设置为1，表示正常值
    is_inlier[values >= 0] = 1

    return is_inlier
    # 计算给定测试数据和标签的平均准确率（accuracy）。
    # 在多标签分类中，这是子集准确率，是一种严格的度量，因为对于每个样本，需要每个标签集合都正确预测。
    def score(self, X, y, sample_weight=None):
        """Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) w.r.t. y.
        """
        # 调用 self.predict(X) 预测函数，并计算其对应于 y 的准确率
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
```