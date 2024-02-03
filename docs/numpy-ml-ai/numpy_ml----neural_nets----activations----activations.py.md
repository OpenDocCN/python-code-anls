# `numpy-ml\numpy_ml\neural_nets\activations\activations.py`

```py
# 用于构建神经网络的激活函数对象集合
from math import erf
from abc import ABC, abstractmethod
import numpy as np

# 定义激活函数基类
class ActivationBase(ABC):
    def __init__(self, **kwargs):
        """初始化 ActivationBase 对象"""
        super().__init__()

    def __call__(self, z):
        """将激活函数应用于输入"""
        if z.ndim == 1:
            z = z.reshape(1, -1)
        return self.fn(z)

    @abstractmethod
    def fn(self, z):
        """将激活函数应用于输入"""
        raise NotImplementedError

    @abstractmethod
    def grad(self, x, **kwargs):
        """计算激活函数相对于输入的梯度"""
        raise NotImplementedError

# 定义 Sigmoid 激活函数类，继承自 ActivationBase
class Sigmoid(ActivationBase):
    def __init__(self):
        """逻辑 Sigmoid 激活函数"""
        super().__init__()

    def __str__(self):
        """返回激活函数的字符串表示"""
        return "Sigmoid"

    def fn(self, z):
        """
        计算输入 z 上的逻辑 Sigmoid 函数值

        .. math::

            \sigma(x_i) = \frac{1}{1 + e^{-x_i}}
        """
        return 1 / (1 + np.exp(-z))

    def grad(self, x):
        """
        计算输入 x 上逻辑 Sigmoid 函数的一阶导数

        .. math::

            \frac{\partial \sigma}{\partial x_i} = \sigma(x_i) (1 - \sigma(x_i))
        """
        fn_x = self.fn(x)
        return fn_x * (1 - fn_x)

    def grad2(self, x):
        """
        计算输入 x 上逻辑 Sigmoid 函数的二阶导数

        .. math::

            \frac{\partial^2 \sigma}{\partial x_i^2} =
                \frac{\partial \sigma}{\partial x_i} (1 - 2 \sigma(x_i))
        """
        fn_x = self.fn(x)
        return fn_x * (1 - fn_x) * (1 - 2 * fn_x)

# 定义 ReLU 激活函数类，继承自 ActivationBase
class ReLU(ActivationBase):
    """
    修正线性激活函数
    """
    Notes
    -----
    "ReLU units can be fragile during training and can "die". For example, a
    large gradient flowing through a ReLU neuron could cause the weights to
    update in such a way that the neuron will never activate on any datapoint
    again. If this happens, then the gradient flowing through the unit will
    forever be zero from that point on. That is, the ReLU units can
    irreversibly die during training since they can get knocked off the data
    manifold.

    For example, you may find that as much as 40% of your network can be "dead"
    (i.e. neurons that never activate across the entire training dataset) if
    the learning rate is set too high. With a proper setting of the learning
    rate this is less frequently an issue." [*]_

    References
    ----------
    .. [*] Karpathy, A. "CS231n: Convolutional neural networks for visual recognition."
    """

    # 初始化函数
    def __init__(self):
        super().__init__()

    # 返回激活函数的字符串表示
    def __str__(self):
        """Return a string representation of the activation function"""
        return "ReLU"

    # 计算输入 z 上的 ReLU 函数值
    def fn(self, z):
        r"""
        Evaulate the ReLU function on the elements of input `z`.

        .. math::

            \text{ReLU}(z_i)
                &=  z_i \ \ \ \ &&\text{if }z_i > 0 \\
                &=  0 \ \ \ \ &&\text{otherwise}
        """
        return np.clip(z, 0, np.inf)

    # 计算输入 x 上的 ReLU 函数的一阶导数
    def grad(self, x):
        r"""
        Evaulate the first derivative of the ReLU function on the elements of input `x`.

        .. math::

            \frac{\partial \text{ReLU}}{\partial x_i}
                &=  1 \ \ \ \ &&\text{if }x_i > 0 \\
                &=  0   \ \ \ \ &&\text{otherwise}
        """
        return (x > 0).astype(int)

    # 计算输入 x 上的 ReLU 函数的二阶导数
    def grad2(self, x):
        r"""
        Evaulate the second derivative of the ReLU function on the elements of
        input `x`.

        .. math::

            \frac{\partial^2 \text{ReLU}}{\partial x_i^2}  =  0
        """
        return np.zeros_like(x)
class LeakyReLU(ActivationBase):
    """
    'Leaky' version of a rectified linear unit (ReLU).

    Notes
    -----
    Leaky ReLUs [*]_ are designed to address the vanishing gradient problem in
    ReLUs by allowing a small non-zero gradient when `x` is negative.

    Parameters
    ----------
    alpha: float
        Activation slope when x < 0. Default is 0.3.

    References
    ----------
    .. [*] Mass, L. M., Hannun, A. Y, & Ng, A. Y. (2013). "Rectifier
       nonlinearities improve neural network acoustic models." *Proceedings of
       the 30th International Conference of Machine Learning, 30*.
    """

    def __init__(self, alpha=0.3):
        # 初始化 LeakyReLU 激活函数，设置斜率参数 alpha，默认为 0.3
        self.alpha = alpha
        # 调用父类的初始化方法
        super().__init__()

    def __str__(self):
        """Return a string representation of the activation function"""
        # 返回激活函数的字符串表示形式
        return "Leaky ReLU(alpha={})".format(self.alpha)

    def fn(self, z):
        r"""
        Evaluate the leaky ReLU function on the elements of input `z`.

        .. math::

            \text{LeakyReLU}(z_i)
                &=  z_i \ \ \ \ &&\text{if } z_i > 0 \\
                &=  \alpha z_i \ \ \ \ &&\text{otherwise}
        """
        # 复制输入 z，避免修改原始数据
        _z = z.copy()
        # 对小于 0 的元素应用 Leaky ReLU 函数
        _z[z < 0] = _z[z < 0] * self.alpha
        return _z

    def grad(self, x):
        r"""
        Evaluate the first derivative of the leaky ReLU function on the elements
        of input `x`.

        .. math::

            \frac{\partial \text{LeakyReLU}}{\partial x_i}
                &=  1 \ \ \ \ &&\text{if }x_i > 0 \\
                &=  \alpha \ \ \ \ &&\text{otherwise}
        """
        # 创建与输入 x 相同形状的全为 1 的数组
        out = np.ones_like(x)
        # 对小于 0 的元素应用 Leaky ReLU 函数的导数
        out[x < 0] *= self.alpha
        return out

    def grad2(self, x):
        r"""
        Evaluate the second derivative of the leaky ReLU function on the
        elements of input `x`.

        .. math::

            \frac{\partial^2 \text{LeakyReLU}}{\partial x_i^2}  =  0
        """
        # 返回与输入 x 相同形状的全为 0 的数组
        return np.zeros_like(x)


class GELU(ActivationBase):
    # 初始化函数，创建一个高斯误差线性单元（GELU）对象
    def __init__(self, approximate=True):
        r"""
        A Gaussian error linear unit (GELU). [*]_

        Notes
        -----
        A ReLU alternative. GELU weights inputs by their value, rather than
        gates inputs by their sign, as in vanilla ReLUs.

        References
        ----------
        .. [*] Hendrycks, D., & Gimpel, K. (2016). "Bridging nonlinearities and
           stochastic regularizers with Gaussian error linear units." *CoRR*.

        Parameters
        ----------
        approximate : bool
            Whether to use a faster but less precise approximation to the Gauss
            error function when calculating the unit activation and gradient.
            Default is True.
        """
        # 设置是否使用近似计算高斯误差函数的标志
        self.approximate = True
        # 调用父类的初始化函数
        super().__init__()

    # 返回激活函数的字符串表示
    def __str__(self):
        """Return a string representation of the activation function"""
        return f"GELU(approximate={self.approximate})"

    # 计算输入 z 的 GELU 函数值
    def fn(self, z):
        r"""
        Compute the GELU function on the elements of input `z`.

        .. math::

            \text{GELU}(z_i) = z_i P(Z \leq z_i) = z_i \Phi(z_i)
                = z_i \cdot \frac{1}{2}(1 + \text{erf}(x/\sqrt{2}))
        """
        # 导入数学库中的常数和函数
        pi, sqrt, tanh = np.pi, np.sqrt, np.tanh

        # 如果使用近似计算
        if self.approximate:
            # 计算近似的 GELU 函数值
            return 0.5 * z * (1 + tanh(sqrt(2 / pi) * (z + 0.044715 * z ** 3)))
        # 如果不使用近似计算
        return 0.5 * z * (1 + erf(z / sqrt(2)))
    def grad(self, x):
        r"""
        Evaluate the first derivative of the GELU function on the elements
        of input `x`.

        .. math::

            \frac{\partial \text{GELU}}{\partial x_i}  =
                \frac{1}{2} + \frac{1}{2}\left(\text{erf}(\frac{x}{\sqrt{2}}) +
                    \frac{x + \text{erf}'(\frac{x}{\sqrt{2}})}{\sqrt{2}}\right)

        where :math:`\text{erf}'(x) = \frac{2}{\sqrt{\pi}} \cdot \exp\{-x^2\}`.
        """
        # 导入所需的数学函数库
        pi, exp, sqrt, tanh = np.pi, np.exp, np.sqrt, np.tanh

        # 计算 x/sqrt(2)
        s = x / sqrt(2)
        # 定义 erf' 函数
        erf_prime = lambda x: (2 / sqrt(pi)) * exp(-(x ** 2))  # noqa: E731

        # 如果使用近似计算
        if self.approximate:
            # 计算近似值
            approx = tanh(sqrt(2 / pi) * (x + 0.044715 * x ** 3))
            # 计算一阶导数
            dx = 0.5 + 0.5 * approx + ((0.5 * x * erf_prime(s)) / sqrt(2))
        else:
            # 计算一阶导数
            dx = 0.5 + 0.5 * erf(s) + ((0.5 * x * erf_prime(s)) / sqrt(2))
        return dx

    def grad2(self, x):
        r"""
        Evaluate the second derivative of the GELU function on the elements
        of input `x`.

        .. math::

            \frac{\partial^2 \text{GELU}}{\partial x_i^2} =
                \frac{1}{2\sqrt{2}} \left\[
                    \text{erf}'(\frac{x}{\sqrt{2}}) +
                    \frac{1}{\sqrt{2}} \text{erf}''(\frac{x}{\sqrt{2}})
                \right]

        where :math:`\text{erf}'(x) = \frac{2}{\sqrt{\pi}} \cdot \exp\{-x^2\}` and
        :math:`\text{erf}''(x) = \frac{-4x}{\sqrt{\pi}} \cdot \exp\{-x^2\}`.
        """
        # 导入所需的数学函数库
        pi, exp, sqrt = np.pi, np.exp, np.sqrt
        # 计算 x/sqrt(2)
        s = x / sqrt(2)

        # 定义 erf' 函数
        erf_prime = lambda x: (2 / sqrt(pi)) * exp(-(x ** 2))  # noqa: E731
        # 定义 erf'' 函数
        erf_prime2 = lambda x: -4 * x * exp(-(x ** 2)) / sqrt(pi)  # noqa: E731
        # 计算二阶导数
        ddx = (1 / 2 * sqrt(2)) * (1 + erf_prime(s) + (erf_prime2(s) / sqrt(2)))
        return ddx
class Tanh(ActivationBase):
    def __init__(self):
        """初始化一个双曲正切激活函数。"""
        super().__init__()

    def __str__(self):
        """返回激活函数的字符串表示形式"""
        return "Tanh"

    def fn(self, z):
        """计算输入 `z` 中元素的双曲正切函数。"""
        return np.tanh(z)

    def grad(self, x):
        r"""
        计算输入 `x` 中元素的双曲正切函数的一阶导数。

        .. math::

            \frac{\partial \tanh}{\partial x_i}  =  1 - \tanh(x)^2
        """
        return 1 - np.tanh(x) ** 2

    def grad2(self, x):
        r"""
        计算输入 `x` 中元素的双曲正切函数的二阶导数。

        .. math::

            \frac{\partial^2 \tanh}{\partial x_i^2} =
                -2 \tanh(x) \left(\frac{\partial \tanh}{\partial x_i}\right)
        """
        tanh_x = np.tanh(x)
        return -2 * tanh_x * (1 - tanh_x ** 2)


class Affine(ActivationBase):
    def __init__(self, slope=1, intercept=0):
        """
        一个仿射激活函数。

        参数
        ----------
        slope: float
            激活斜率。默认为1。
        intercept: float
            截距/偏移项。默认为0。
        """
        self.slope = slope
        self.intercept = intercept
        super().__init__()

    def __str__(self):
        """返回激活函数的字符串表示形式"""
        return "Affine(slope={}, intercept={})".format(self.slope, self.intercept)

    def fn(self, z):
        r"""
        计算输入 `z` 中元素的仿射激活函数。

        .. math::

            \text{Affine}(z_i)  =  \text{slope} \times z_i + \text{intercept}
        """
        return self.slope * z + self.intercept
    # 计算 Affine 激活函数在输入 x 的每个元素上的一阶导数
    def grad(self, x):
        # 一阶导数公式：导数等于斜率
        return self.slope * np.ones_like(x)

    # 计算 Affine 激活函数在输入 x 的每个元素上的二阶导数
    def grad2(self, x):
        # 二阶导数公式：二阶导数等于0
        return np.zeros_like(x)
class Identity(Affine):
    def __init__(self):
        """
        Identity activation function.

        Notes
        -----
        :class:`Identity` is syntactic sugar for :class:`Affine` with
        slope = 1 and intercept = 0.
        """
        # 调用父类的构造函数，设置斜率为1，截距为0
        super().__init__(slope=1, intercept=0)

    def __str__(self):
        """Return a string representation of the activation function"""
        # 返回激活函数的字符串表示
        return "Identity"


class ELU(ActivationBase):
    def __init__(self, alpha=1.0):
        r"""
        An exponential linear unit (ELU).

        Notes
        -----
        ELUs are intended to address the fact that ReLUs are strictly nonnegative
        and thus have an average activation > 0, increasing the chances of internal
        covariate shift and slowing down learning. ELU units address this by (1)
        allowing negative values when :math:`x < 0`, which (2) are bounded by a value
        :math:`-\alpha`. Similar to :class:`LeakyReLU`, the negative activation
        values help to push the average unit activation towards 0. Unlike
        :class:`LeakyReLU`, however, the boundedness of the negative activation
        allows for greater robustness in the face of large negative values,
        allowing the function to avoid conveying the *degree* of "absence"
        (negative activation) in the input. [*]_

        Parameters
        ----------
        alpha : float
            Slope of negative segment. Default is 1.

        References
        ----------
        .. [*] Clevert, D. A., Unterthiner, T., Hochreiter, S. (2016). "Fast
           and accurate deep network learning by exponential linear units
           (ELUs)". *4th International Conference on Learning
           Representations*.
        """
        # 设置 alpha 参数
        self.alpha = alpha
        # 调用父类的构造函数
        super().__init__()

    def __str__(self):
        """Return a string representation of the activation function"""
        # 返回激活函数的字符串表示，包括 alpha 参数的值
        return "ELU(alpha={})".format(self.alpha)
    # 定义一个函数 fn，用于计算 ELU 激活函数在输入 z 的元素上的值
    def fn(self, z):
        # ELU 激活函数的定义，根据输入 z 的值进行计算
        return np.where(z > 0, z, self.alpha * (np.exp(z) - 1))

    # 定义一个函数 grad，用于计算 ELU 激活函数在输入 x 的元素上的一阶导数
    def grad(self, x):
        # ELU 激活函数一阶导数的定义，根据输入 x 的值进行计算
        return np.where(x > 0, np.ones_like(x), self.alpha * np.exp(x))

    # 定义一个函数 grad2，用于计算 ELU 激活函数在输入 x 的元素上的二阶导数
    def grad2(self, x):
        # ELU 激活函数二阶导数的定义，根据输入 x 的值进行计算
        return np.where(x >= 0, np.zeros_like(x), self.alpha * np.exp(x))
class Exponential(ActivationBase):
    # 定义指数（以 e 为底）激活函数
    def __init__(self):
        """An exponential (base e) activation function"""
        # 调用父类的构造函数
        super().__init__()

    def __str__(self):
        """Return a string representation of the activation function"""
        # 返回激活函数的字符串表示
        return "Exponential"

    def fn(self, z):
        r"""
        Evaluate the activation function

        .. math::
            \text{Exponential}(z_i) = e^{z_i}
        """
        # 计算激活函数的值
        return np.exp(z)

    def grad(self, x):
        r"""
        Evaluate the first derivative of the exponential activation on the elements
        of input `x`.

        .. math::

            \frac{\partial \text{Exponential}}{\partial x_i}  =  e^{x_i}
        """
        # 计算激活函数的一阶导数
        return np.exp(x)

    def grad2(self, x):
        r"""
        Evaluate the second derivative of the exponential activation on the elements
        of input `x`.

        .. math::

            \frac{\partial^2 \text{Exponential}}{\partial x_i^2}  =  e^{x_i}
        """
        # 计算激活函数的二阶导数
        return np.exp(x)


class SELU(ActivationBase):
    r"""
    A scaled exponential linear unit (SELU).

    Notes
    -----
    SELU units, when used in conjunction with proper weight initialization and
    regularization techniques, encourage neuron activations to converge to
    zero-mean and unit variance without explicit use of e.g., batchnorm.

    For SELU units, the :math:`\alpha` and :math:`\text{scale}` values are
    constants chosen so that the mean and variance of the inputs are preserved
    between consecutive layers. As such the authors propose weights be
    initialized using Lecun-Normal initialization: :math:`w_{ij} \sim
    \mathcal{N}(0, 1 / \text{fan_in})`, and to use the dropout variant
    :math:`\alpha`-dropout during regularization. [*]_

    See the reference for more information (especially the appendix ;-) ).

    References
    ----------
    # 定义 SELU 激活函数类
    """
    [*] Klambauer, G., Unterthiner, T., & Hochreiter, S. (2017).
    "Self-normalizing neural networks." *Advances in Neural Information
    Processing Systems, 30.*
    """

    def __init__(self):
        # 初始化 SELU 激活函数的 alpha 和 scale 参数
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
        # 创建 ELU 激活函数对象
        self.elu = ELU(alpha=self.alpha)
        # 调用父类的初始化方法
        super().__init__()

    def __str__(self):
        """Return a string representation of the activation function"""
        # 返回 SELU 的字符串表示
        return "SELU"

    def fn(self, z):
        r"""
        Evaluate the SELU activation on the elements of input `z`.

        .. math::

            \text{SELU}(z_i)  =  \text{scale} \times \text{ELU}(z_i, \alpha)

        which is simply

        .. math::

            \text{SELU}(z_i)
                &= \text{scale} \times z_i \ \ \ \ &&\text{if }z_i > 0 \\
                &= \text{scale} \times \alpha (e^{z_i} - 1) \ \ \ \ &&\text{otherwise}
        """
        # 计算 SELU 激活函数在输入 z 的元素上的值
        return self.scale * self.elu.fn(z)

    def grad(self, x):
        r"""
        Evaluate the first derivative of the SELU activation on the elements
        of input `x`.

        .. math::

            \frac{\partial \text{SELU}}{\partial x_i}
                &=  \text{scale} \ \ \ \ &&\text{if } x_i > 0 \\
                &=  \text{scale} \times \alpha e^{x_i} \ \ \ \ &&\text{otherwise}
        """
        # 计算 SELU 激活函数在输入 x 的元素上的一阶导数
        return np.where(
            x >= 0, np.ones_like(x) * self.scale, np.exp(x) * self.alpha * self.scale,
        )

    def grad2(self, x):
        r"""
        Evaluate the second derivative of the SELU activation on the elements
        of input `x`.

        .. math::

            \frac{\partial^2 \text{SELU}}{\partial x_i^2}
                &=  0 \ \ \ \ &&\text{if } x_i > 0 \\
                &=  \text{scale} \times \alpha e^{x_i} \ \ \ \ &&\text{otherwise}
        """
        # 计算 SELU 激活函数在输入 x 的元素上的二阶导数
        return np.where(x > 0, np.zeros_like(x), np.exp(x) * self.alpha * self.scale)
class HardSigmoid(ActivationBase):
    def __init__(self):
        """
        A "hard" sigmoid activation function.

        Notes
        -----
        The hard sigmoid is a piecewise linear approximation of the logistic
        sigmoid that is computationally more efficient to compute.
        """
        # 初始化函数，创建一个“hard” sigmoid激活函数对象
        super().__init__()

    def __str__(self):
        """Return a string representation of the activation function"""
        # 返回激活函数的字符串表示
        return "Hard Sigmoid"

    def fn(self, z):
        r"""
        Evaluate the hard sigmoid activation on the elements of input `z`.

        .. math::

            \text{HardSigmoid}(z_i)
                &= 0 \ \ \ \ &&\text{if }z_i < -2.5 \\
                &= 0.2 z_i + 0.5 \ \ \ \ &&\text{if }-2.5 \leq z_i \leq 2.5 \\
                &= 1 \ \ \ \ &&\text{if }z_i > 2.5
        """
        # 计算输入`z`的hard sigmoid激活值
        return np.clip((0.2 * z) + 0.5, 0.0, 1.0)

    def grad(self, x):
        r"""
        Evaluate the first derivative of the hard sigmoid activation on the elements
        of input `x`.

        .. math::

            \frac{\partial \text{HardSigmoid}}{\partial x_i}
                &=  0.2 \ \ \ \ &&\text{if } -2.5 \leq x_i \leq 2.5\\
                &=  0 \ \ \ \ &&\text{otherwise}
        """
        # 计算输入`x`的hard sigmoid激活函数的一阶导数
        return np.where((x >= -2.5) & (x <= 2.5), 0.2, 0)

    def grad2(self, x):
        r"""
        Evaluate the second derivative of the hard sigmoid activation on the elements
        of input `x`.

        .. math::

            \frac{\partial^2 \text{HardSigmoid}}{\partial x_i^2} =  0
        """
        # 计算输入`x`的hard sigmoid激活函数的二阶导数
        return np.zeros_like(x)


class SoftPlus(ActivationBase):
    def __init__(self):
        """
        A softplus activation function.

        Notes
        -----
        In contrast to :class:`ReLU`, the softplus activation is differentiable
        everywhere (including 0). It is, however, less computationally efficient to
        compute.

        The derivative of the softplus activation is the logistic sigmoid.
        """
        # 初始化函数，创建一个softplus激活函数对象
        super().__init__()
    # 返回激活函数的字符串表示形式
    def __str__(self):
        """Return a string representation of the activation function"""
        return "SoftPlus"

    # 计算输入 z 中每个元素的 softplus 激活函数值
    def fn(self, z):
        r"""
        Evaluate the softplus activation on the elements of input `z`.

        .. math::

            \text{SoftPlus}(z_i) = \log(1 + e^{z_i})
        """
        return np.log(np.exp(z) + 1)

    # 计算输入 x 中每个元素 softplus 激活函数的一阶导数值
    def grad(self, x):
        r"""
        Evaluate the first derivative of the softplus activation on the elements
        of input `x`.

        .. math::

            \frac{\partial \text{SoftPlus}}{\partial x_i} = \frac{e^{x_i}}{1 + e^{x_i}}
        """
        exp_x = np.exp(x)
        return exp_x / (exp_x + 1)

    # 计算输入 x 中每个元素 softplus 激活函数的二阶导数值
    def grad2(self, x):
        r"""
        Evaluate the second derivative of the softplus activation on the elements
        of input `x`.

        .. math::

            \frac{\partial^2 \text{SoftPlus}}{\partial x_i^2} =
                \frac{e^{x_i}}{(1 + e^{x_i})^2}
        """
        exp_x = np.exp(x)
        return exp_x / ((exp_x + 1) ** 2)
```