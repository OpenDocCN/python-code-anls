# `D:\src\scipysrc\scikit-learn\sklearn\_loss\link.py`

```
"""
Module contains classes for invertible (and differentiable) link functions.
"""

# Author: Christian Lorentzen <lorentzen.ch@gmail.com>

# 从 abc 模块导入 ABC 抽象基类和 abstractmethod 抽象方法装饰器
from abc import ABC, abstractmethod

# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass

# 导入 numpy 库，并使用别名 np
import numpy as np

# 从 scipy.special 模块导入 expit 和 logit 函数
from scipy.special import expit, logit

# 从 scipy.stats 模块导入 gmean 函数
from scipy.stats import gmean

# 从 ..utils.extmath 导入 softmax 函数
from ..utils.extmath import softmax


# 使用 dataclass 装饰器创建类 Interval，表示一个区间
@dataclass
class Interval:
    low: float
    high: float
    low_inclusive: bool
    high_inclusive: bool

    # 初始化方法，检查 low 是否小于等于 high
    def __post_init__(self):
        """Check that low <= high"""
        if self.low > self.high:
            raise ValueError(
                f"One must have low <= high; got low={self.low}, high={self.high}."
            )

    # 方法 includes，用于检测数组 x 中的元素是否在区间范围内
    def includes(self, x):
        """Test whether all values of x are in interval range.

        Parameters
        ----------
        x : ndarray
            Array whose elements are tested to be in interval range.

        Returns
        -------
        result : bool
        """
        # 如果 low_inclusive 为 True，则使用 np.greater_equal 函数比较 x 和 low
        if self.low_inclusive:
            low = np.greater_equal(x, self.low)
        else:
            low = np.greater(x, self.low)

        # 如果不是所有元素都在区间内，则返回 False
        if not np.all(low):
            return False

        # 如果 high_inclusive 为 True，则使用 np.less_equal 函数比较 x 和 high
        if self.high_inclusive:
            high = np.less_equal(x, self.high)
        else:
            high = np.less(x, self.high)

        # 返回是否所有元素都在区间内的布尔值
        # 注意：bool(np.all(high)) 将 numpy.bool_ 转换为 Python 的 bool 类型
        return bool(np.all(high))


# 定义函数 _inclusive_low_high，生成在区间范围内的 low 和 high 值
def _inclusive_low_high(interval, dtype=np.float64):
    """Generate values low and high to be within the interval range.

    This is used in tests only.

    Returns
    -------
    low, high : tuple
        The returned values low and high lie within the interval.
    """
    # 计算机器精度 eps
    eps = 10 * np.finfo(dtype).eps

    # 根据 interval.low 的值生成 low
    if interval.low == -np.inf:
        low = -1e10
    elif interval.low < 0:
        low = interval.low * (1 - eps) + eps
    else:
        low = interval.low * (1 + eps) + eps

    # 根据 interval.high 的值生成 high
    if interval.high == np.inf:
        high = 1e10
    elif interval.high < 0:
        high = interval.high * (1 + eps) - eps
    else:
        high = interval.high * (1 - eps) - eps

    # 返回生成的 low 和 high 值
    return low, high


# 定义抽象基类 BaseLink，用于不同iable 和可逆的链接函数
class BaseLink(ABC):
    """Abstract base class for differentiable, invertible link functions.

    Convention:
        - link function g: raw_prediction = g(y_pred)
        - inverse link h: y_pred = h(raw_prediction)

    For (generalized) linear models, `raw_prediction = X @ coef` is the so
    called linear predictor, and `y_pred = h(raw_prediction)` is the predicted
    conditional (on X) expected value of the target `y_true`.

    The methods are not implemented as staticmethods in case a link function needs
    parameters.
    """

    is_multiclass = False  # used for testing only

    # 通常，raw_prediction 可以是任意实数，而 y_pred 是一个开区间
    # interval_raw_prediction = Interval(-np.inf, np.inf, False, False)
    
    # y_pred 的区间设置为 (-∞, +∞)，即所有实数范围
    interval_y_pred = Interval(-np.inf, np.inf, False, False)

    # 抽象方法声明，子类需要实现具体的链接函数
    @abstractmethod
    # 定义一个方法 link，用于计算链接函数 g(y_pred)。
    # 链接函数将（预测的）目标值映射到原始预测值，即 `g(y_pred) = raw_prediction`。
    def link(self, y_pred, out=None):
        """Compute the link function g(y_pred).

        The link function maps (predicted) target values to raw predictions,
        i.e. `g(y_pred) = raw_prediction`.

        Parameters
        ----------
        y_pred : array
            Predicted target values.
        out : array
            A location into which the result is stored. If provided, it must
            have a shape that the inputs broadcast to. If not provided or None,
            a freshly-allocated array is returned.

        Returns
        -------
        out : array
            Output array, element-wise link function.
        """

    # 定义一个抽象方法 inverse，用于计算逆链接函数 h(raw_prediction)。
    # 逆链接函数将原始预测值映射到预测的目标值，即 `h(raw_prediction) = y_pred`。
    @abstractmethod
    def inverse(self, raw_prediction, out=None):
        """Compute the inverse link function h(raw_prediction).

        The inverse link function maps raw predictions to predicted target
        values, i.e. `h(raw_prediction) = y_pred`.

        Parameters
        ----------
        raw_prediction : array
            Raw prediction values (in link space).
        out : array
            A location into which the result is stored. If provided, it must
            have a shape that the inputs broadcast to. If not provided or None,
            a freshly-allocated array is returned.

        Returns
        -------
        out : array
            Output array, element-wise inverse link function.
        """
class IdentityLink(BaseLink):
    """The identity link function g(x)=x."""

    def link(self, y_pred, out=None):
        # 如果提供了输出数组 out，则将 y_pred 复制到 out 中并返回 out
        if out is not None:
            np.copyto(out, y_pred)
            return out
        else:
            # 否则直接返回 y_pred
            return y_pred

    # inverse 方法与 link 方法功能相同，直接返回 y_pred
    inverse = link


class LogLink(BaseLink):
    """The log link function g(x)=log(x)."""

    # interval_y_pred 定义了 y_pred 可取值的区间为 (0, ∞)
    interval_y_pred = Interval(0, np.inf, False, False)

    def link(self, y_pred, out=None):
        # 计算 y_pred 的自然对数，并将结果存入 out（如果提供了 out 的话）
        return np.log(y_pred, out=out)

    def inverse(self, raw_prediction, out=None):
        # 计算原始预测 raw_prediction 的指数，并将结果存入 out（如果提供了 out 的话）
        return np.exp(raw_prediction, out=out)


class LogitLink(BaseLink):
    """The logit link function g(x)=logit(x)."""

    # interval_y_pred 定义了 y_pred 可取值的区间为 (0, 1)
    interval_y_pred = Interval(0, 1, False, False)

    def link(self, y_pred, out=None):
        # 调用 logit 函数计算 y_pred 的 logit 值，并将结果存入 out（如果提供了 out 的话）
        return logit(y_pred, out=out)

    def inverse(self, raw_prediction, out=None):
        # 调用 expit 函数计算原始预测 raw_prediction 的反函数值，并将结果存入 out（如果提供了 out 的话）
        return expit(raw_prediction, out=out)


class HalfLogitLink(BaseLink):
    """Half the logit link function g(x)=1/2 * logit(x).

    Used for the exponential loss.
    """

    # interval_y_pred 定义了 y_pred 可取值的区间为 (0, 1)
    interval_y_pred = Interval(0, 1, False, False)

    def link(self, y_pred, out=None):
        # 计算 y_pred 的 logit 值，并将结果存入 out（如果提供了 out 的话），然后乘以 0.5
        out = logit(y_pred, out=out)
        out *= 0.5
        return out

    def inverse(self, raw_prediction, out=None):
        # 计算原始预测 raw_prediction 的反函数值，并将结果存入 out（如果提供了 out 的话）
        return expit(2 * raw_prediction, out)


class MultinomialLogit(BaseLink):
    """The symmetric multinomial logit function.

    Convention:
        - y_pred.shape = raw_prediction.shape = (n_samples, n_classes)

    Notes:
        - The inverse link h is the softmax function.
        - The sum is over the second axis, i.e. axis=1 (n_classes).

    We have to choose additional constraints in order to make

        y_pred[k] = exp(raw_pred[k]) / sum(exp(raw_pred[k]), k=0..n_classes-1)

    for n_classes classes identifiable and invertible.
    We choose the symmetric side constraint where the geometric mean response
    is set as reference category, see [2]:

    The symmetric multinomial logit link function for a single data point is
    then defined as

        raw_prediction[k] = g(y_pred[k]) = log(y_pred[k]/gmean(y_pred))
        = log(y_pred[k]) - mean(log(y_pred)).

    Note that this is equivalent to the definition in [1] and implies mean
    centered raw predictions:

        sum(raw_prediction[k], k=0..n_classes-1) = 0.

    For linear models with raw_prediction = X @ coef, this corresponds to
    sum(coef[k], k=0..n_classes-1) = 0, i.e. the sum over classes for every
    feature is zero.

    Reference
    ---------
    .. [1] Friedman, Jerome; Hastie, Trevor; Tibshirani, Robert. "Additive
        logistic regression: a statistical view of boosting" Ann. Statist.
        28 (2000), no. 2, 337--407. doi:10.1214/aos/1016218223.
        https://projecteuclid.org/euclid.aos/1016218223

    .. [2] Zahid, Faisal Maqbool and Gerhard Tutz. "Ridge estimation for
        multinomial logit models with symmetric side constraints."
        Computational Statistics 28 (2013): 1017-1034.
        http://epub.ub.uni-muenchen.de/11001/1/tr067.pdf
    """
    
    is_multiclass = True
    interval_y_pred = Interval(0, 1, False, False)

    def symmetrize_raw_prediction(self, raw_prediction):
        # 对原始预测数据进行对称化处理，减去每行的平均值
        return raw_prediction - np.mean(raw_prediction, axis=1)[:, np.newaxis]

    def link(self, y_pred, out=None):
        # 进行链接函数操作，以几何平均值作为参考类别
        gm = gmean(y_pred, axis=1)
        return np.log(y_pred / gm[:, np.newaxis], out=out)

    def inverse(self, raw_prediction, out=None):
        # 如果没有提供输出数组，直接返回 softmax 处理后的原始预测数据副本
        if out is None:
            return softmax(raw_prediction, copy=True)
        else:
            # 否则，将原始预测数据复制到输出数组中，然后对输出数组进行 softmax 处理
            np.copyto(out, raw_prediction)
            softmax(out, copy=False)
            return out
# 定义一个名为 _LINKS 的字典，用于将字符串映射到对应的类
_LINKS = {
    # 将字符串 "identity" 映射到 IdentityLink 类
    "identity": IdentityLink,
    # 将字符串 "log" 映射到 LogLink 类
    "log": LogLink,
    # 将字符串 "logit" 映射到 LogitLink 类
    "logit": LogitLink,
    # 将字符串 "half_logit" 映射到 HalfLogitLink 类
    "half_logit": HalfLogitLink,
    # 将字符串 "multinomial_logit" 映射到 MultinomialLogit 类
    "multinomial_logit": MultinomialLogit,
}
```