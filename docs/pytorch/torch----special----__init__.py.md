# `.\pytorch\torch\special\__init__.py`

```
# 导入 torch 库
import torch
# 导入 _add_docstr 和 _special，这两个是在 torch._C 模块中定义的，类型提示为 ignore[attr-defined]
from torch._C import _add_docstr, _special

# 以下是导入 torch 文档生成工具模块中定义的一些变量和函数
from torch._torch_docs import common_args, multi_dim_common

# 定义公开的函数和变量列表
__all__ = [
    'airy_ai',
    'bessel_j0',
    'bessel_j1',
    'bessel_y0',
    'bessel_y1',
    'chebyshev_polynomial_t',
    'chebyshev_polynomial_u',
    'chebyshev_polynomial_v',
    'chebyshev_polynomial_w',
    'digamma',
    'entr',
    'erf',
    'erfc',
    'erfcx',
    'erfinv',
    'exp2',
    'expit',
    'expm1',
    'gammainc',
    'gammaincc',
    'gammaln',
    'hermite_polynomial_h',
    'hermite_polynomial_he',
    'i0',
    'i0e',
    'i1',
    'i1e',
    'laguerre_polynomial_l',
    'legendre_polynomial_p',
    'log1p',
    'log_ndtr',
    'log_softmax',
    'logit',
    'logsumexp',
    'modified_bessel_i0',
    'modified_bessel_i1',
    'modified_bessel_k0',
    'modified_bessel_k1',
    'multigammaln',
    'ndtr',
    'ndtri',
    'polygamma',
    'psi',
    'round',
    'shifted_chebyshev_polynomial_t',
    'shifted_chebyshev_polynomial_u',
    'shifted_chebyshev_polynomial_v',
    'shifted_chebyshev_polynomial_w',
    'scaled_modified_bessel_k0',
    'scaled_modified_bessel_k1',
    'sinc',
    'softmax',
    'spherical_bessel_j0',
    'xlog1py',
    'xlogy',
    'zeta',
]

# 定义 Tensor 类型为 torch.Tensor
Tensor = torch.Tensor

# 使用 _add_docstr 函数为 special_entr 函数添加文档字符串
entr = _add_docstr(_special.special_entr,
                   r"""
entr(input, *, out=None) -> Tensor
Computes the entropy on :attr:`input` (as defined below), elementwise.

.. math::
    \begin{align}
    \text{entr(x)} = \begin{cases}
        -x * \ln(x)  & x > 0 \\
        0 &  x = 0.0 \\
        -\infty & x < 0
    \end{cases}
    \end{align}

Args:
   input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::
    >>> a = torch.arange(-0.5, 1, 0.5)
    >>> a
    tensor([-0.5000,  0.0000,  0.5000])
    >>> torch.special.entr(a)
    tensor([  -inf, 0.0000, 0.3466])
""")

# 使用 _add_docstr 函数为 special_psi 函数添加文档字符串，别名为 psi
psi = _add_docstr(_special.special_psi,
                  r"""
psi(input, *, out=None) -> Tensor

Alias for :func:`torch.special.digamma`.
""")

# 使用 _add_docstr 函数为 special_digamma 函数添加文档字符串，别名为 digamma
digamma = _add_docstr(_special.special_digamma,
                      r"""
digamma(input, *, out=None) -> Tensor

Computes the logarithmic derivative of the gamma function on `input`.

.. math::
    \digamma(x) = \frac{d}{dx} \ln\left(\Gamma\left(x\right)\right) = \frac{\Gamma'(x)}{\Gamma(x)}

Args:
    input (Tensor): the tensor to compute the digamma function on

Keyword args:
    {out}

.. note::  This function is similar to SciPy's `scipy.special.digamma`.

.. note::  From PyTorch 1.8 onwards, the digamma function returns `-Inf` for `0`.
           Previously it returned `NaN` for `0`.

Example::

    >>> a = torch.tensor([1, 0.5])
    >>> torch.special.digamma(a)
    tensor([-0.5772, -1.9635])

""".format(**common_args))

# 使用 _add_docstr 函数为 special_gammaln 函数添加文档字符串，别名为 gammaln
gammaln = _add_docstr(_special.special_gammaln,
                      r"""
gammaln(input, *, out=None) -> Tensor
Compute the logarithm of the absolute value of the gamma function on `input`.
""")
# 计算输入的伽马函数绝对值的自然对数
# \text{out}_{i} = \ln \Gamma(|\text{input}_{i}|)
""" + """
Args:
    {input}  # 输入张量

Keyword args:
    {out}  # 输出张量

Example::

    >>> a = torch.arange(0.5, 2, 0.5)
    >>> torch.special.gammaln(a)
    tensor([ 0.5724,  0.0000, -0.1208])
""".format(**common_args))

# 添加多项式伽马函数的文档字符串
polygamma = _add_docstr(_special.special_polygamma,
                        r"""
polygamma(n, input, *, out=None) -> Tensor

Computes the :math:`n^{th}` derivative of the digamma function on :attr:`input`.
:math:`n \geq 0` is called the order of the polygamma function.

.. math::
    \psi^{(n)}(x) = \frac{d^{(n)}}{dx^{(n)}} \psi(x)

.. note::
    This function is implemented only for nonnegative integers :math:`n \geq 0`.
""" + """
Args:
    n (int): the order of the polygamma function
    {input}  # 输入张量

Keyword args:
    {out}  # 输出张量

Example::
    >>> a = torch.tensor([1, 0.5])
    >>> torch.special.polygamma(1, a)
    tensor([1.64493, 4.9348])
    >>> torch.special.polygamma(2, a)
    tensor([ -2.4041, -16.8288])
    >>> torch.special.polygamma(3, a)
    tensor([ 6.4939, 97.4091])
    >>> torch.special.polygamma(4, a)
    tensor([ -24.8863, -771.4742])
""".format(**common_args))

# 添加误差函数的文档字符串
erf = _add_docstr(_special.special_erf,
                  r"""
erf(input, *, out=None) -> Tensor

Computes the error function of :attr:`input`. The error function is defined as follows:

.. math::
    \mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt
""" + r"""
Args:
    {input}  # 输入张量

Keyword args:
    {out}  # 输出张量

Example::

    >>> torch.special.erf(torch.tensor([0, -1., 10.]))
    tensor([ 0.0000, -0.8427,  1.0000])
""".format(**common_args))

# 添加补误差函数的文档字符串
erfc = _add_docstr(_special.special_erfc,
                   r"""
erfc(input, *, out=None) -> Tensor

Computes the complementary error function of :attr:`input`.
The complementary error function is defined as follows:

.. math::
    \mathrm{erfc}(x) = 1 - \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt
""" + r"""
Args:
    {input}  # 输入张量

Keyword args:
    {out}  # 输出张量

Example::

    >>> torch.special.erfc(torch.tensor([0, -1., 10.]))
    tensor([ 1.0000, 1.8427,  0.0000])
""".format(**common_args))

# 添加缩放补误差函数的文档字符串
erfcx = _add_docstr(_special.special_erfcx,
                    r"""
erfcx(input, *, out=None) -> Tensor

Computes the scaled complementary error function for each element of :attr:`input`.
The scaled complementary error function is defined as follows:

.. math::
    \mathrm{erfcx}(x) = e^{x^2} \mathrm{erfc}(x)
""" + r"""

""" + r"""
Args:
    {input}  # 输入张量

Keyword args:
    {out}  # 输出张量

Example::

    >>> torch.special.erfcx(torch.tensor([0, -1., 10.]))
    tensor([ 1.0000, 5.0090, 0.0561])
""".format(**common_args))

# 添加误差函数的反函数的文档字符串
erfinv = _add_docstr(_special.special_erfinv,
                     r"""
erfinv(input, *, out=None) -> Tensor

Computes the inverse error function of :attr:`input`.
The inverse error function is defined in the range :math:`(-1, 1)` as:

.. math::
    \mathrm{erfinv}(\mathrm{erf}(x)) = x
""" + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> torch.special.erfinv(torch.tensor([0, 0.5, -1.]))
    tensor([ 0.0000,  0.4769,    -inf])
""".format(**common_args))



# 给 _special.special_logit 函数添加文档字符串
logit = _add_docstr(_special.special_logit,
                    r"""
logit(input, eps=None, *, out=None) -> Tensor

Returns a new tensor with the logit of the elements of :attr:`input`.
:attr:`input` is clamped to [eps, 1 - eps] when eps is not None.
When eps is None and :attr:`input` < 0 or :attr:`input` > 1, the function will yields NaN.

.. math::
    \begin{align}
    y_{i} &= \ln(\frac{z_{i}}{1 - z_{i}}) \\
    z_{i} &= \begin{cases}
        x_{i} & \text{if eps is None} \\
        \text{eps} & \text{if } x_{i} < \text{eps} \\
        x_{i} & \text{if } \text{eps} \leq x_{i} \leq 1 - \text{eps} \\
        1 - \text{eps} & \text{if } x_{i} > 1 - \text{eps}
    \end{cases}
    \end{align}
""" + r"""
Args:
    {input}
    eps (float, optional): the epsilon for input clamp bound. Default: ``None``

Keyword args:
    {out}

Example::

    >>> a = torch.rand(5)
    >>> a
    tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516])
    >>> torch.special.logit(a, eps=1e-6)
    tensor([-0.9466,  2.6352,  0.6131, -1.7169,  0.6261])
""".format(**common_args))



# 给 _special.special_logsumexp 函数添加文档字符串
logsumexp = _add_docstr(_special.special_logsumexp,
                        r"""
logsumexp(input, dim, keepdim=False, *, out=None)

Alias for :func:`torch.logsumexp`.
""".format(**multi_dim_common))



# 给 _special.special_expit 函数添加文档字符串
expit = _add_docstr(_special.special_expit,
                    r"""
expit(input, *, out=None) -> Tensor

Computes the expit (also known as the logistic sigmoid function) of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \frac{1}{1 + e^{-\text{input}_{i}}}
""" + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> t = torch.randn(4)
    >>> t
    tensor([ 0.9213,  1.0887, -0.8858, -1.7683])
    >>> torch.special.expit(t)
    tensor([ 0.7153,  0.7481,  0.2920,  0.1458])
""".format(**common_args))



# 给 _special.special_exp2 函数添加文档字符串
exp2 = _add_docstr(_special.special_exp2,
                   r"""
exp2(input, *, out=None) -> Tensor

Computes the base two exponential function of :attr:`input`.

.. math::
    y_{i} = 2^{x_{i}}

""" + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> torch.special.exp2(torch.tensor([0, math.log2(2.), 3, 4]))
    tensor([ 1.,  2.,  8., 16.])
""".format(**common_args))



# 给 _special.special_expm1 函数添加文档字符串
expm1 = _add_docstr(_special.special_expm1,
                    r"""
expm1(input, *, out=None) -> Tensor

Computes the exponential of the elements minus 1
of :attr:`input`.

.. math::
    y_{i} = e^{x_{i}} - 1

.. note:: This function provides greater precision than exp(x) - 1 for small values of x.

""" + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> torch.special.expm1(torch.tensor([0, math.log(2.)]))
    tensor([ 0.,  1.])
""".format(**common_args))



# 给 _special.special_xlog1py 函数添加文档字符串
xlog1py = _add_docstr(_special.special_xlog1py,
                      r"""
xlog1py(input, other, *, out=None) -> Tensor
# 定义函数 `i0e`，计算每个元素的指数尺度下的修正第一类零阶贝塞尔函数
i0e = _add_docstr(_special.special_i0e,
                  r"""
i0e(input, *, out=None) -> Tensor

Computes the exponentially scaled zeroth order modified Bessel function of the first kind (as defined below)

.. math::
    \text{out}_{i} = I_{0e}(\text{input}_{i}) = e^{-|\text{input}_{i}|} \cdot I_0(\text{input}_{i})

Args:
    input (Tensor): 输入张量

Keyword args:
    {out}

Example::

    >>> torch.i0e(torch.arange(5, dtype=torch.float32))
    tensor([1.0000, 0.6839, 0.3679, 0.1353, 0.0432])

""".format(**common_args))
# 给定输入的每个元素，计算指数缩放的修正第一类修改 Bessel 函数（如下定义）
for each element of :attr:`input`.

.. math::
    \text{out}_{i} = \exp(-|x|) * i0(x) = \exp(-|x|) * \sum_{k=0}^{\infty} \frac{(\text{input}_{i}^2/4)^k}{(k!)^2}

""" + r"""
Args:
    {input}  # 输入参数，应为 torch.Tensor

Keyword args:
    {out}  # 输出参数，应为 torch.Tensor

Example::
    >>> torch.special.i0e(torch.arange(5, dtype=torch.float32))
    tensor([1.0000, 0.4658, 0.3085, 0.2430, 0.2070])
""".format(**common_args))

# 使用 _special.special_i1 函数为 torch.special.i1 添加文档字符串
i1 = _add_docstr(_special.special_i1,
                 r"""
i1(input, *, out=None) -> Tensor
Computes the first order modified Bessel function of the first kind (as defined below)
for each element of :attr:`input`.

.. math::
    \text{out}_{i} = \frac{(\text{input}_{i})}{2} * \sum_{k=0}^{\infty} \frac{(\text{input}_{i}^2/4)^k}{(k!) * (k+1)!}

""" + r"""
Args:
    {input}  # 输入参数，应为 torch.Tensor

Keyword args:
    {out}  # 输出参数，应为 torch.Tensor

Example::
    >>> torch.special.i1(torch.arange(5, dtype=torch.float32))
    tensor([0.0000, 0.5652, 1.5906, 3.9534, 9.7595])
""".format(**common_args))

# 使用 _special.special_i1e 函数为 torch.special.i1e 添加文档字符串
i1e = _add_docstr(_special.special_i1e,
                  r"""
i1e(input, *, out=None) -> Tensor
Computes the exponentially scaled first order modified Bessel function of the first kind (as defined below)
for each element of :attr:`input`.

.. math::
    \text{out}_{i} = \exp(-|x|) * i1(x) =
        \exp(-|x|) * \frac{(\text{input}_{i})}{2} * \sum_{k=0}^{\infty} \frac{(\text{input}_{i}^2/4)^k}{(k!) * (k+1)!}

""" + r"""
Args:
    {input}  # 输入参数，应为 torch.Tensor

Keyword args:
    {out}  # 输出参数，应为 torch.Tensor

Example::
    >>> torch.special.i1e(torch.arange(5, dtype=torch.float32))
    tensor([0.0000, 0.2079, 0.2153, 0.1968, 0.1788])
""".format(**common_args))

# 使用 _special.special_ndtr 函数为 torch.special.ndtr 添加文档字符串
ndtr = _add_docstr(_special.special_ndtr,
                   r"""
ndtr(input, *, out=None) -> Tensor
Computes the area under the standard Gaussian probability density function,
integrated from minus infinity to :attr:`input`, elementwise.

.. math::
    \text{ndtr}(x) = \frac{1}{\sqrt{2 \pi}}\int_{-\infty}^{x} e^{-\frac{1}{2}t^2} dt

""" + r"""
Args:
    {input}  # 输入参数，应为 torch.Tensor

Keyword args:
    {out}  # 输出参数，应为 torch.Tensor

Example::
    >>> torch.special.ndtr(torch.tensor([-3., -2, -1, 0, 1, 2, 3]))
    tensor([0.0013, 0.0228, 0.1587, 0.5000, 0.8413, 0.9772, 0.9987])
""".format(**common_args))

# 使用 _special.special_ndtri 函数为 torch.special.ndtri 添加文档字符串
ndtri = _add_docstr(_special.special_ndtri,
                    r"""
ndtri(input, *, out=None) -> Tensor
Computes the argument, x, for which the area under the Gaussian probability density function
(integrated from minus infinity to x) is equal to :attr:`input`, elementwise.

.. math::
    \text{ndtri}(p) = \sqrt{2}\text{erf}^{-1}(2p - 1)

.. note::
    Also known as quantile function for Normal Distribution.

""" + r"""
Args:
    {input}  # 输入参数，应为 torch.Tensor

Keyword args:
    {out}  # 输出参数，应为 torch.Tensor

Example::
    >>> torch.special.ndtri(torch.tensor([0, 0.25, 0.5, 0.75, 1]))
    tensor([   -inf, -0.6745,  0.0000,  0.6745,     inf])
""".format(**common_args))

# 使用 _special.special_log_ndtr 函数为 torch.special.log_ndtr 添加文档字符串
log_ndtr = _add_docstr(_special.special_log_ndtr,
                       r"""
log_ndtr(input, *, out=None) -> Tensor
Computes the log of the area under the standard Gaussian probability density function,
log_ndtr = _add_docstr(_special.special_log_ndtr,
                       r"""
log_ndtr(input, *, out=None) -> Tensor

Computes the log of the cumulative distribution function of the standard normal distribution.

.. math::
    \text{log\_ndtr}(x) = \log\left(\frac{1}{\sqrt{2 \pi}} \int_{-\infty}^{x} e^{-\frac{1}{2}t^2} dt \right)

Args:
    input (Tensor): input tensor containing values for which log_ndtr is computed.
    out (Tensor, optional): output tensor. Must be of the same shape as input.

Keyword args:
    {out}

Example::
    >>> torch.special.log_ndtr(torch.tensor([-3., -2, -1, 0, 1, 2, 3]))
    tensor([-6.6077, -3.7832, -1.8410, -0.6931, -0.1728, -0.0230, -0.0014])
""".format(**common_args))

log1p = _add_docstr(_special.special_log1p,
                    r"""
log1p(input, *, out=None) -> Tensor

Alias for :func:`torch.log1p`.

Args:
    input (Tensor): input tensor for which log1p is computed.
    out (Tensor, optional): output tensor. Must be of the same shape as input.

""")

sinc = _add_docstr(_special.special_sinc,
                   r"""
sinc(input, *, out=None) -> Tensor

Computes the normalized sinc function.

.. math::
    \text{sinc}(x) =
    \begin{cases}
      1, & \text{if}\ x=0 \\
      \frac{\sin(\pi x)}{\pi x}, & \text{otherwise}
    \end{cases}

Args:
    input (Tensor): input tensor containing values for which sinc is computed.
    out (Tensor, optional): output tensor. Must be of the same shape as input.

Example::
    >>> t = torch.randn(4)
    >>> t
    tensor([ 0.2252, -0.2948,  1.0267, -1.1566])
    >>> torch.special.sinc(t)
    tensor([ 0.9186,  0.8631, -0.0259, -0.1300])
""".format(**common_args))

round = _add_docstr(_special.special_round,
                    r"""
round(input, *, out=None) -> Tensor

Alias for :func:`torch.round`.

Args:
    input (Tensor): input tensor for which round is computed.
    out (Tensor, optional): output tensor. Must be of the same shape as input.

""")

softmax = _add_docstr(_special.special_softmax,
                      r"""
softmax(input, dim, *, dtype=None) -> Tensor

Computes the softmax function along a specified dimension.

Softmax is defined as:

:math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`

Args:
    input (Tensor): input tensor for which softmax is computed.
    dim (int): A dimension along which softmax will be computed.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is cast to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.

Example::
    >>> t = torch.ones(2, 2)
    >>> torch.special.softmax(t, 0)
    tensor([[0.5000, 0.5000],
            [0.5000, 0.5000]])

""")

log_softmax = _add_docstr(_special.special_log_softmax,
                          r"""
log_softmax(input, dim, *, dtype=None) -> Tensor

Computes log of the softmax function along a specified dimension.

Mathematically equivalent to log(softmax(x)), but computed in a numerically stable way.

Args:
    input (Tensor): input tensor for which log_softmax is computed.
    dim (int): A dimension along which log_softmax will be computed.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is cast to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.

Example::

    >>> t = torch.randn(2, 3)
    >>> torch.special.log_softmax(t, 1)

"""
    # 创建一个2x2的张量t，其中所有元素的值为1
    t = torch.ones(2, 2)
    # 对张量t沿着第一个维度（行）计算log_softmax函数，返回一个张量
    # log_softmax函数对张量进行softmax操作后再计算对数值
    result = torch.special.log_softmax(t, 0)
    # 输出结果张量，其形状为2x2，每个元素是经过log_softmax函数计算后的结果
    print(result)
# 将 _special.special_zeta 函数添加文档字符串
zeta = _add_docstr(_special.special_zeta,
                   r"""
zeta(input, other, *, out=None) -> Tensor

Computes the Hurwitz zeta function, elementwise.

.. math::
    \zeta(x, q) = \sum_{k=0}^{\infty} \frac{1}{(k + q)^x}

""" + r"""
Args:
    input (Tensor): the input tensor corresponding to `x`.
    other (Tensor): the input tensor corresponding to `q`.

.. note::
    The Riemann zeta function corresponds to the case when `q = 1`

Keyword args:
    {out}

Example::
    >>> x = torch.tensor([2., 4.])
    >>> torch.special.zeta(x, 1)
    tensor([1.6449, 1.0823])
    >>> torch.special.zeta(x, torch.tensor([1., 2.]))
    tensor([1.6449, 0.0823])
    >>> torch.special.zeta(2, torch.tensor([1., 2.]))
    tensor([1.6449, 0.6449])
""".format(**common_args))

# 将 _special.special_multigammaln 函数添加文档字符串
multigammaln = _add_docstr(_special.special_multigammaln,
                           r"""
multigammaln(input, p, *, out=None) -> Tensor

Computes the `multivariate log-gamma function
<https://en.wikipedia.org/wiki/Multivariate_gamma_function>`_ with dimension
:math:`p` element-wise, given by

.. math::
    \log(\Gamma_{p}(a)) = C + \displaystyle \sum_{i=1}^{p} \log\left(\Gamma\left(a - \frac{i - 1}{2}\right)\right)

where :math:`C = \log(\pi) \cdot \frac{p (p - 1)}{4}` and :math:`\Gamma(-)` is the Gamma function.

All elements must be greater than :math:`\frac{p - 1}{2}`, otherwise the behavior is undefiend.
""" + """

Args:
    input (Tensor): the tensor to compute the multivariate log-gamma function
    p (int): the number of dimensions

Keyword args:
    {out}

Example::

    >>> a = torch.empty(2, 3).uniform_(1, 2)
    >>> a
    tensor([[1.6835, 1.8474, 1.1929],
            [1.0475, 1.7162, 1.4180]])
    >>> torch.special.multigammaln(a, 2)
    tensor([[0.3928, 0.4007, 0.7586],
            [1.0311, 0.3901, 0.5049]])
""".format(**common_args))

# 将 _special.special_gammainc 函数添加文档字符串
gammainc = _add_docstr(_special.special_gammainc,
                       r"""
gammainc(input, other, *, out=None) -> Tensor

Computes the regularized lower incomplete gamma function:

.. math::
    \text{out}_{i} = \frac{1}{\Gamma(\text{input}_i)} \int_0^{\text{other}_i} t^{\text{input}_i-1} e^{-t} dt

where both :math:`\text{input}_i` and :math:`\text{other}_i` are weakly positive
and at least one is strictly positive.
If both are zero or either is negative then :math:`\text{out}_i=\text{nan}`.
:math:`\Gamma(\cdot)` in the equation above is the gamma function,

.. math::
    \Gamma(\text{input}_i) = \int_0^\infty t^{(\text{input}_i-1)} e^{-t} dt.

See :func:`torch.special.gammaincc` and :func:`torch.special.gammaln` for related functions.

Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`
and float inputs.

.. note::
    The backward pass with respect to :attr:`input` is not yet supported.
    Please open an issue on PyTorch's Github to request it.

""" + r"""
Args:
    input (Tensor): the first non-negative input tensor
    other (Tensor): the second non-negative input tensor

Keyword args:
    {out}

Example::
    >>> a = torch.tensor([1.5, 2.5])
    >>> b = torch.tensor([0.5, 1.0])
    >>> torch.special.gammainc(a, b)
    tensor([0.7812, 0.9375])
""".format(**common_args))
    # 创建一个包含单个浮点数 4.0 的张量 a1
    >>> a1 = torch.tensor([4.0])
    # 创建一个包含三个浮点数 3.0, 4.0, 5.0 的张量 a2
    >>> a2 = torch.tensor([3.0, 4.0, 5.0])
    # 使用 torch.special.gammaincc 函数计算 Gamma 不完全补充函数的值，将结果存储在张量 a 中
    >>> a = torch.special.gammaincc(a1, a2)
    # 打印张量 a 的值
    tensor([0.3528, 0.5665, 0.7350])
    # 再次打印张量 a 的值
    tensor([0.3528, 0.5665, 0.7350])
    # 使用 torch.special.gammainc 函数计算 Gamma 完全补充函数的值，然后将两个张量相加，存储在张量 b 中
    >>> b = torch.special.gammainc(a1, a2) + torch.special.gammaincc(a1, a2)
    # 打印张量 b 的值
    tensor([1., 1., 1.])
# 定义一个字符串，使用.format方法将通用参数插入到字符串中
""".format(**common_args))

# 调用_add_docstr函数，为special_gammaincc函数添加文档字符串
gammaincc = _add_docstr(_special.special_gammaincc,
                        r"""
gammaincc(input, other, *, out=None) -> Tensor

Computes the regularized upper incomplete gamma function:

.. math::
    \text{out}_{i} = \frac{1}{\Gamma(\text{input}_i)} \int_{\text{other}_i}^{\infty} t^{\text{input}_i-1} e^{-t} dt

where both :math:`\text{input}_i` and :math:`\text{other}_i` are weakly positive
and at least one is strictly positive.
If both are zero or either is negative then :math:`\text{out}_i=\text{nan}`.
:math:`\Gamma(\cdot)` in the equation above is the gamma function,

.. math::
    \Gamma(\text{input}_i) = \int_0^\infty t^{(\text{input}_i-1)} e^{-t} dt.

See :func:`torch.special.gammainc` and :func:`torch.special.gammaln` for related functions.

Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`
and float inputs.

.. note::
    The backward pass with respect to :attr:`input` is not yet supported.
    Please open an issue on PyTorch's Github to request it.

""" + r"""
Args:
    input (Tensor): the first non-negative input tensor
    other (Tensor): the second non-negative input tensor

Keyword args:
    {out}

Example::

    >>> a1 = torch.tensor([4.0])
    >>> a2 = torch.tensor([3.0, 4.0, 5.0])
    >>> a = torch.special.gammaincc(a1, a2)
    tensor([0.6472, 0.4335, 0.2650])
    >>> b = torch.special.gammainc(a1, a2) + torch.special.gammaincc(a1, a2)
    tensor([1., 1., 1.])

""".format(**common_args))

# 为special_airy_ai函数添加文档字符串
airy_ai = _add_docstr(_special.special_airy_ai,
                      r"""
airy_ai(input, *, out=None) -> Tensor

Airy function :math:`\text{Ai}\left(\text{input}\right)`.

""" + r"""
Args:
    {input}

Keyword args:
    {out}
""".format(**common_args))

# 为special_bessel_j0函数添加文档字符串
bessel_j0 = _add_docstr(_special.special_bessel_j0,
                        r"""
bessel_j0(input, *, out=None) -> Tensor

Bessel function of the first kind of order :math:`0`.

""" + r"""
Args:
    {input}

Keyword args:
    {out}
""".format(**common_args))

# 为special_bessel_j1函数添加文档字符串
bessel_j1 = _add_docstr(_special.special_bessel_j1,
                        r"""
bessel_j1(input, *, out=None) -> Tensor

Bessel function of the first kind of order :math:`1`.

""" + r"""
Args:
    {input}

Keyword args:
    {out}
""".format(**common_args))

# 为special_bessel_y0函数添加文档字符串
bessel_y0 = _add_docstr(_special.special_bessel_y0,
                        r"""
bessel_y0(input, *, out=None) -> Tensor

Bessel function of the second kind of order :math:`0`.

""" + r"""
Args:
    {input}

Keyword args:
    {out}
""".format(**common_args))

# 为special_bessel_y1函数添加文档字符串
bessel_y1 = _add_docstr(_special.special_bessel_y1,
                        r"""
bessel_y1(input, *, out=None) -> Tensor

Bessel function of the second kind of order :math:`1`.

""" + r"""
Args:
    {input}

Keyword args:
    {out}
""".format(**common_args))

# 为special_chebyshev_polynomial_t函数添加文档字符串
chebyshev_polynomial_t = _add_docstr(_special.special_chebyshev_polynomial_t,
                                     r"""
chebyshev_polynomial_t(input, n, *, out=None) -> Tensor
chebyshev_polynomial_u = _add_docstr(_special.special_chebyshev_polynomial_u,
                                     r"""
chebyshev_polynomial_t(input, n, *, out=None) -> Tensor

Chebyshev polynomial of the second kind :math:`U_{n}(\text{input})`.

If :math:`n = 0`, :math:`1` is returned. If :math:`n = 1`,
:math:`2 \times \text{input}` is returned. If :math:`n < 6` or
:math:`|\text{input}| > 1`, the recursion:

.. math::
    U_{n + 1}(\text{input}) = 2 \times \text{input} \times U_{n}(\text{input}) - U_{n - 1}(\text{input})

is evaluated. Otherwise, the explicit trigonometric formula:

.. math::
    \frac{\text{sin}((n + 1) \times \text{arccos}(\text{input}))}{\text{sin}(\text{arccos}(\text{input}))}

is evaluated.

""" + r"""
Args:
    input (Tensor): Input tensor.
    n (Tensor): Degree of the polynomial.

Keyword args:
    out (Tensor, optional): Output tensor. Default: None.
""".format(**common_args))

chebyshev_polynomial_v = _add_docstr(_special.special_chebyshev_polynomial_v,
                                     r"""
chebyshev_polynomial_v(input, n, *, out=None) -> Tensor

Chebyshev polynomial of the third kind :math:`V_{n}^{\ast}(\text{input})`.

""" + r"""
Args:
    input (Tensor): Input tensor.
    n (Tensor): Degree of the polynomial.

Keyword args:
    out (Tensor, optional): Output tensor. Default: None.
""".format(**common_args))

chebyshev_polynomial_w = _add_docstr(_special.special_chebyshev_polynomial_w,
                                     r"""
chebyshev_polynomial_w(input, n, *, out=None) -> Tensor

Chebyshev polynomial of the fourth kind :math:`W_{n}^{\ast}(\text{input})`.

""" + r"""
Args:
    input (Tensor): Input tensor.
    n (Tensor): Degree of the polynomial.

Keyword args:
    out (Tensor, optional): Output tensor. Default: None.
""".format(**common_args))

hermite_polynomial_h = _add_docstr(_special.special_hermite_polynomial_h,
                                   r"""
hermite_polynomial_h(input, n, *, out=None) -> Tensor

Physicist's Hermite polynomial :math:`H_{n}(\text{input})`.

If :math:`n = 0`, :math:`1` is returned. If :math:`n = 1`, :math:`\text{input}`
is returned. Otherwise, the recursion:

.. math::
    H_{n + 1}(\text{input}) = 2 \times \text{input} \times H_{n}(\text{input}) - H_{n - 1}(\text{input})

is evaluated.

""" + r"""
Args:
    input (Tensor): Input tensor.
    n (Tensor): Degree of the polynomial.

Keyword args:
    out (Tensor, optional): Output tensor. Default: None.
""".format(**common_args))

hermite_polynomial_he = _add_docstr(_special.special_hermite_polynomial_he,
                                    r"""
hermite_polynomial_he(input, n, *, out=None) -> Tensor

Physicist's Hermite polynomial :math:`He_{n}(\text{input})`.

""" + r"""
Args:
    input (Tensor): Input tensor.
    n (Tensor): Degree of the polynomial.

Keyword args:
    out (Tensor, optional): Output tensor. Default: None.
""".format(**common_args))
# 定义 Probabilist's Hermite polynomial 的文档字符串和函数签名
"""
Probabilist's Hermite polynomial :math:`He_{n}(\text{input})`.

If :math:`n = 0`, :math:`1` is returned. If :math:`n = 1`, :math:`\text{input}`
is returned. Otherwise, the recursion:

.. math::
    He_{n + 1}(\text{input}) = 2 \times \text{input} \times He_{n}(\text{input}) - He_{n - 1}(\text{input})

is evaluated.

Args:
    input (Tensor): 输入参数
    n (Tensor): 多项式的次数

Keyword args:
    out (Tensor, optional): 可选参数，输出张量
"""

# 使用 _add_docstr 函数为 Laguerre polynomial 添加文档字符串和函数签名
laguerre_polynomial_l = _add_docstr(_special.special_laguerre_polynomial_l,
                                    r"""
laguerre_polynomial_l(input, n, *, out=None) -> Tensor

Laguerre polynomial :math:`L_{n}(\text{input})`.

If :math:`n = 0`, :math:`1` is returned. If :math:`n = 1`, :math:`\text{input}`
is returned. Otherwise, the recursion:

.. math::
    L_{n + 1}(\text{input}) = 2 \times \text{input} \times L_{n}(\text{input}) - L_{n - 1}(\text{input})

is evaluated.

Args:
    input (Tensor): 输入参数
    n (Tensor): 多项式的次数

Keyword args:
    out (Tensor, optional): 可选参数，输出张量
""")

# 使用 _add_docstr 函数为 Legendre polynomial 添加文档字符串和函数签名
legendre_polynomial_p = _add_docstr(_special.special_legendre_polynomial_p,
                                    r"""
legendre_polynomial_p(input, n, *, out=None) -> Tensor

Legendre polynomial :math:`P_{n}(\text{input})`.

If :math:`n = 0`, :math:`1` is returned. If :math:`n = 1`, :math:`\text{input}`
is returned. Otherwise, the recursion:

.. math::
    P_{n + 1}(\text{input}) = 2 \times \text{input} \times P_{n}(\text{input}) - P_{n - 1}(\text{input})

is evaluated.

Args:
    input (Tensor): 输入参数
    n (Tensor): 多项式的次数

Keyword args:
    out (Tensor, optional): 可选参数，输出张量
""")

# 使用 _add_docstr 函数为 Modified Bessel function I_0 添加文档字符串和函数签名
modified_bessel_i0 = _add_docstr(_special.special_modified_bessel_i0,
                                 r"""
modified_bessel_i0(input, *, out=None) -> Tensor

Modified Bessel function of the first kind of order :math:`0`.

Args:
    input (Tensor): 输入参数

Keyword args:
    out (Tensor, optional): 可选参数，输出张量
""")

# 使用 _add_docstr 函数为 Modified Bessel function I_1 添加文档字符串和函数签名
modified_bessel_i1 = _add_docstr(_special.special_modified_bessel_i1,
                                 r"""
modified_bessel_i1(input, *, out=None) -> Tensor

Modified Bessel function of the first kind of order :math:`1`.

Args:
    input (Tensor): 输入参数

Keyword args:
    out (Tensor, optional): 可选参数，输出张量
""")

# 使用 _add_docstr 函数为 Modified Bessel function K_0 添加文档字符串和函数签名
modified_bessel_k0 = _add_docstr(_special.special_modified_bessel_k0,
                                 r"""
modified_bessel_k0(input, *, out=None) -> Tensor

Modified Bessel function of the second kind of order :math:`0`.

Args:
    input (Tensor): 输入参数

Keyword args:
    out (Tensor, optional): 可选参数，输出张量
""")

# 使用 _add_docstr 函数为 Modified Bessel function K_1 添加文档字符串和函数签名
modified_bessel_k1 = _add_docstr(_special.special_modified_bessel_k1,
                                 r"""
modified_bessel_k1(input, *, out=None) -> Tensor

Modified Bessel function of the second kind of order :math:`1`.

Args:
    input (Tensor): 输入参数

Keyword args:
    out (Tensor, optional): 可选参数，输出张量
""")

# 使用 _add_docstr 函数为 Scaled Modified Bessel function K_0 添加文档字符串和函数签名
scaled_modified_bessel_k0 = _add_docstr(_special.special_scaled_modified_bessel_k0,
                                        r"""
scaled_modified_bessel_k0(input, *, out=None) -> Tensor

Scaled Modified Bessel function of the second kind of order :math:`0`.
""")
# 计算修正的第二类贝塞尔函数的缩放版本，阶数为0。
scaled_modified_bessel_k0(input, *, out=None) -> Tensor

""" + r"""
Args:
    {input}  # 输入张量，用于计算贝塞尔函数

Keyword args:
    {out}  # 可选参数，用于存储计算结果的张量
""".format(**common_args))

# 使用特定的文档字符串格式化函数添加文档注释，描述修正的第二类贝塞尔函数的缩放版本，阶数为1。
scaled_modified_bessel_k1 = _add_docstr(_special.special_scaled_modified_bessel_k1,
                                        r"""
scaled_modified_bessel_k1(input, *, out=None) -> Tensor

Scaled modified Bessel function of the second kind of order :math:`1`.

""" + r"""
Args:
    {input}  # 输入张量，用于计算贝塞尔函数

Keyword args:
    {out}  # 可选参数，用于存储计算结果的张量
""".format(**common_args))

# 使用特定的文档字符串格式化函数添加文档注释，描述位移切比雪夫多项式的第一类版本，阶数为n。
shifted_chebyshev_polynomial_t = _add_docstr(_special.special_shifted_chebyshev_polynomial_t,
                                             r"""
shifted_chebyshev_polynomial_t(input, n, *, out=None) -> Tensor

Chebyshev polynomial of the first kind :math:`T_{n}^{\ast}(\text{input})`.

""" + r"""
Args:
    {input}  # 输入张量，多项式的自变量
    n (Tensor): Degree of the polynomial.  # 多项式的阶数

Keyword args:
    {out}  # 可选参数，用于存储计算结果的张量
""".format(**common_args))

# 使用特定的文档字符串格式化函数添加文档注释，描述位移切比雪夫多项式的第二类版本，阶数为n。
shifted_chebyshev_polynomial_u = _add_docstr(_special.special_shifted_chebyshev_polynomial_u,
                                             r"""
shifted_chebyshev_polynomial_u(input, n, *, out=None) -> Tensor

Chebyshev polynomial of the second kind :math:`U_{n}^{\ast}(\text{input})`.

""" + r"""
Args:
    {input}  # 输入张量，多项式的自变量
    n (Tensor): Degree of the polynomial.  # 多项式的阶数

Keyword args:
    {out}  # 可选参数，用于存储计算结果的张量
""".format(**common_args))

# 使用特定的文档字符串格式化函数添加文档注释，描述位移切比雪夫多项式的第三类版本，阶数为n。
shifted_chebyshev_polynomial_v = _add_docstr(_special.special_shifted_chebyshev_polynomial_v,
                                             r"""
shifted_chebyshev_polynomial_v(input, n, *, out=None) -> Tensor

Chebyshev polynomial of the third kind :math:`V_{n}^{\ast}(\text{input})`.

""" + r"""
Args:
    {input}  # 输入张量，多项式的自变量
    n (Tensor): Degree of the polynomial.  # 多项式的阶数

Keyword args:
    {out}  # 可选参数，用于存储计算结果的张量
""".format(**common_args))

# 使用特定的文档字符串格式化函数添加文档注释，描述位移切比雪夫多项式的第四类版本，阶数为n。
shifted_chebyshev_polynomial_w = _add_docstr(_special.special_shifted_chebyshev_polynomial_w,
                                             r"""
shifted_chebyshev_polynomial_w(input, n, *, out=None) -> Tensor

Chebyshev polynomial of the fourth kind :math:`W_{n}^{\ast}(\text{input})`.

""" + r"""
Args:
    {input}  # 输入张量，多项式的自变量
    n (Tensor): Degree of the polynomial.  # 多项式的阶数

Keyword args:
    {out}  # 可选参数，用于存储计算结果的张量
""".format(**common_args))

# 使用特定的文档字符串格式化函数添加文档注释，描述球形贝塞尔函数的第一类版本，阶数为0。
spherical_bessel_j0 = _add_docstr(_special.special_spherical_bessel_j0,
                                  r"""
spherical_bessel_j0(input, *, out=None) -> Tensor

Spherical Bessel function of the first kind of order :math:`0`.

""" + r"""
Args:
    {input}  # 输入张量，用于计算贝塞尔函数

Keyword args:
    {out}  # 可选参数，用于存储计算结果的张量
""".format(**common_args))
```