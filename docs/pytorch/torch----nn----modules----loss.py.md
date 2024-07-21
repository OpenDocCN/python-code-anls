# `.\pytorch\torch\nn\modules\loss.py`

```
# mypy: allow-untyped-defs
# 导入必要的类型定义和兼容扩展模块
from typing import Callable, Optional
from typing_extensions import deprecated

# 导入 PyTorch 相关模块
from torch import Tensor
from torch.nn import _reduction as _Reduction, functional as F

# 导入自定义模块
from .distance import PairwiseDistance
from .module import Module

# 指定模块导出的公共接口列表
__all__ = [
    "L1Loss",
    "NLLLoss",
    "NLLLoss2d",
    "PoissonNLLLoss",
    "GaussianNLLLoss",
    "KLDivLoss",
    "MSELoss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "HingeEmbeddingLoss",
    "MultiLabelMarginLoss",
    "SmoothL1Loss",
    "HuberLoss",
    "SoftMarginLoss",
    "CrossEntropyLoss",
    "MultiLabelSoftMarginLoss",
    "CosineEmbeddingLoss",
    "MarginRankingLoss",
    "MultiMarginLoss",
    "TripletMarginLoss",
    "TripletMarginWithDistanceLoss",
    "CTCLoss",
]

# 定义损失函数基类 _Loss
class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__()
        # 根据参数设置减少方式
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

# 定义带权重的损失函数基类 _WeightedLoss，继承自 _Loss
class _WeightedLoss(_Loss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        # 注册权重参数为缓冲区
        self.register_buffer("weight", weight)
        self.weight: Optional[Tensor]

# 定义 L1 损失函数类，继承自 _Loss
class L1Loss(_Loss):
    r"""Creates a criterion that measures the mean absolute error (MAE) between each element in
    the input :math:`x` and target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left| x_n - y_n \right|,

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The sum operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

    Supports real-valued and complex-valued inputs.
    """
    """
    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then
          :math:`(*)`, same shape as the input.

    Examples::

        >>> loss = nn.L1Loss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    __constants__ = ["reduction"]

    # 初始化函数，设置损失函数的参数
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)

    # 前向传播函数，计算L1损失
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(input, target, reduction=self.reduction)
# 定义一个名为 NLLLoss 的类，继承自 _WeightedLoss 类
class NLLLoss(_WeightedLoss):
    # 多行字符串文档，描述了负对数似然损失的用途和特点
    r"""The negative log likelihood loss. It is useful to train a classification
    problem with `C` classes.

    If provided, the optional argument :attr:`weight` should be a 1D Tensor assigning
    weight to each of the classes. This is particularly useful when you have an
    unbalanced training set.

    The `input` given through a forward call is expected to contain
    log-probabilities of each class. `input` has to be a Tensor of size either
    :math:`(minibatch, C)` or :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    with :math:`K \geq 1` for the `K`-dimensional case. The latter is useful for
    higher dimension inputs, such as computing NLL loss per-pixel for 2D images.

    Obtaining log-probabilities in a neural network is easily achieved by
    adding a  `LogSoftmax`  layer in the last layer of your network.
    You may use `CrossEntropyLoss` instead, if you prefer not to add an extra
    layer.

    The `target` that this loss expects should be a class index in the range :math:`[0, C-1]`
    where `C = number of classes`; if `ignore_index` is specified, this loss also accepts
    this class index (this index may not necessarily be in the class range).

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_{y_n} x_{n,y_n}, \quad
        w_{c} = \text{weight}[c] \cdot \mathbb{1}\{c \not= \text{ignore\_index}\},

    where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight, and
    :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then

    .. math::
        \ell(x, y) = \begin{cases}
            \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n, &
            \text{if reduction} = \text{`mean';}\\
            \sum_{n=1}^N l_n,  &
            \text{if reduction} = \text{`sum'.}
        \end{cases}
    # 参数 weight (Tensor, optional): 对每个类别的手动重新缩放权重。如果给定，必须是大小为 `C` 的 Tensor。否则，默认为全为1。
    # 参数 size_average (bool, optional): 已弃用（参见 :attr:`reduction`）。默认情况下，损失在批次中每个损失元素上进行平均。注意，对于某些损失，每个样本可能有多个元素。
    #                                  如果 :attr:`size_average` 设置为 ``False``，则损失将按每个小批量汇总。当 :attr:`reduce` 为 ``False`` 时忽略此参数。默认值为 ``None``。
    # 参数 ignore_index (int, optional): 指定一个目标值，该值将被忽略，不会对输入梯度贡献。当 :attr:`size_average` 为 ``True`` 时，损失将在非忽略目标上进行平均。
    # 参数 reduce (bool, optional): 已弃用（参见 :attr:`reduction`）。默认情况下，损失在每个小批量观察中进行平均或汇总，具体取决于 :attr:`size_average`。
    #                              当 :attr:`reduce` 为 ``False`` 时，返回每个批次元素的损失，并忽略 :attr:`size_average`。默认值为 ``None``。
    # 参数 reduction (str, optional): 指定要应用于输出的减少类型：``'none'`` | ``'mean'`` | ``'sum'``。 ``'none'``: 不应用任何减少，
    #                                ``'mean'``: 取输出的加权平均值， ``'sum'``: 输出将被求和。注意： :attr:`size_average` 和 :attr:`reduce` 正在被弃用的过程中，
    #                                同时指定这两个参数将覆盖 :attr:`reduction`。默认值为 ``'mean'``。

    # Shape::
    #   - 输入: :math:`(N, C)` 或 :math:`(C)`，其中 `C` 是类别数，`N` 是批量大小，或者
    #           :math:`(N, C, d_1, d_2, ..., d_K)`，其中 :math:`K \geq 1` 是 `K` 维损失的情况。
    #   - 目标: :math:`(N)` 或 :math:`()`，其中每个值为 :math:`0 \leq \text{targets}[i] \leq C-1`，或者
    #           :math:`(N, d_1, d_2, ..., d_K)`，其中 :math:`K \geq 1` 是 `K` 维损失的情况。
    #   - 输出: 如果 :attr:`reduction` 为 ``'none'``，形状为 :math:`(N)` 或 :math:`(N, d_1, d_2, ..., d_K)`，其中 :math:`K \geq 1` 是 `K` 维损失的情况。
    #           否则为标量。
    Examples::
    
        >>> log_softmax = nn.LogSoftmax(dim=1)
        >>> loss_fn = nn.NLLLoss()
        >>> # input to NLLLoss is of size N x C = 3 x 5
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> # each element in target must have 0 <= value < C
        >>> target = torch.tensor([1, 0, 4])
        >>> loss = loss_fn(log_softmax(input), target)
        >>> loss.backward()
        >>>
        >>>
        >>> # 2D loss example (used, for example, with image inputs)
        >>> N, C = 5, 4
        >>> loss_fn = nn.NLLLoss()
        >>> data = torch.randn(N, 16, 10, 10)
        >>> conv = nn.Conv2d(16, C, (3, 3))
        >>> log_softmax = nn.LogSoftmax(dim=1)
        >>> # output of conv forward is of shape [N, C, 8, 8]
        >>> output = log_softmax(conv(data))
        >>> # each element in target must have 0 <= value < C
        >>> target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
        >>> # input to NLLLoss is of size N x C x height (8) x width (8)
        >>> loss = loss_fn(output, target)
        >>> loss.backward()
    
    """
    __constants__ = ["ignore_index", "reduction"]
    ignore_index: int
    
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        # 调用父类构造函数初始化损失函数对象
        super().__init__(weight, size_average, reduce, reduction)
        # 设置损失函数忽略的索引值
        self.ignore_index = ignore_index
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # 调用函数库 F 中的负对数似然损失函数
        return F.nll_loss(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )
# 标记为过时的类装饰器，提示使用者 `NLLLoss2d` 已被弃用
@deprecated(
    "`NLLLoss2d` has been deprecated. "
    "Please use `NLLLoss` instead as a drop-in replacement and see "
    "https://pytorch.org/docs/main/nn.html#torch.nn.NLLLoss for more details.",
    category=FutureWarning,
)
# 定义一个继承自 `NLLLoss` 的类 `NLLLoss2d`
class NLLLoss2d(NLLLoss):
    # 初始化方法
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        # 调用父类 `NLLLoss` 的初始化方法
        super().__init__(weight, size_average, ignore_index, reduce, reduction)


# 定义一个继承自 `_Loss` 的类 `PoissonNLLLoss`
class PoissonNLLLoss(_Loss):
    # 文档字符串，描述基于泊松分布的负对数似然损失
    r"""Negative log likelihood loss with Poisson distribution of target.

    The loss can be described as:

    .. math::
        \text{target} \sim \mathrm{Poisson}(\text{input})

        \text{loss}(\text{input}, \text{target}) = \text{input} - \text{target} * \log(\text{input})
                                    + \log(\text{target!})

    The last term can be omitted or approximated with Stirling formula. The
    approximation is used for target values more than 1. For targets less or
    equal to 1 zeros are added to the loss.
    Args:
        log_input (bool, optional): 是否使用对数输入。如果为 ``True``，损失计算为
            :math:`\exp(\text{input}) - \text{target}*\text{input}`；如果为 ``False``，损失为
            :math:`\text{input} - \text{target}*\log(\text{input}+\text{eps})`。
        full (bool, optional): 是否计算完整的损失，即添加斯特林近似项

            .. math::
                \text{target}*\log(\text{target}) - \text{target} + 0.5 * \log(2\pi\text{target}).
        size_average (bool, optional): 已弃用（参见 :attr:`reduction`）。默认情况下，
            损失将在批处理中每个损失元素上进行平均。注意，对于某些损失，每个样本可能有多个元素。
            如果 :attr:`size_average` 设置为 ``False``，则损失将在每个小批量中进行求和。当
            :attr:`reduce` 为 ``False`` 时忽略此选项。默认值为 ``True``。
        eps (float, optional): 用于避免在 :attr:`log_input = False` 时计算 :math:`\log(0)` 的小值。
            默认值为 1e-8。
        reduce (bool, optional): 已弃用（参见 :attr:`reduction`）。默认情况下，
            损失将根据 :attr:`size_average` 对每个小批量中的观察结果进行平均或求和。当
            :attr:`reduce` 为 ``False`` 时，返回每个批次元素的损失，并忽略 :attr:`size_average`。
            默认值为 ``True``。
        reduction (str, optional): 指定应用于输出的减少方式：
            ``'none'`` | ``'mean'`` | ``'sum'``。``'none'``: 不应用任何减少，
            ``'mean'``: 输出的总和将除以输出中的元素数量，
            ``'sum'``: 将输出求和。注意： :attr:`size_average` 和 :attr:`reduce` 正在逐步弃用，
            同时指定这两个参数将覆盖 :attr:`reduction`。默认值为 ``'mean'``。

    Examples::

        >>> loss = nn.PoissonNLLLoss()
        >>> log_input = torch.randn(5, 2, requires_grad=True)
        >>> target = torch.randn(5, 2)
        >>> output = loss(log_input, target)
        >>> output.backward()

    Shape:
        - Input: :math:`(*)`，其中 :math:`*` 表示任意数量的维度。
        - Target: :math:`(*)`，与输入形状相同。
        - Output: 默认情况下是标量。如果 :attr:`reduction` 为 ``'none'``，则为 :math:`(*)`，
          与输入形状相同。
    """
    __constants__ = ["log_input", "full", "eps", "reduction"]
    log_input: bool
    full: bool
    eps: float

    def __init__(
        self,
        log_input: bool = True,
        full: bool = False,
        size_average=None,
        eps: float = 1e-8,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        # 调用父类构造函数初始化损失函数的参数
        super().__init__(size_average, reduce, reduction)
        # 初始化损失函数的参数
        self.log_input = log_input
        self.full = full
        self.eps = eps
    # 定义一个方法 `forward`，用于计算泊松负对数似然损失
    def forward(self, log_input: Tensor, target: Tensor) -> Tensor:
        # 使用 PyTorch 中的泊松负对数似然损失函数 `F.poisson_nll_loss` 进行计算
        return F.poisson_nll_loss(
            log_input,               # 输入的对数概率值
            target,                  # 目标值（真实值）
            log_input=self.log_input,  # 参数: 输入的对数概率值
            full=self.full,          # 参数: 是否计算完整的损失，默认为 False
            eps=self.eps,            # 参数: 用于稳定计算的小值，默认为 1e-8
            reduction=self.reduction  # 参数: 损失的减少方式，可选值为 'none', 'mean', 'sum'
        )
# 定义一个 GaussianNLLLoss 类，继承自 _Loss 类
class GaussianNLLLoss(_Loss):
    r"""Gaussian negative log likelihood loss.

    The targets are treated as samples from Gaussian distributions with
    expectations and variances predicted by the neural network. For a
    ``target`` tensor modelled as having Gaussian distribution with a tensor
    of expectations ``input`` and a tensor of positive variances ``var`` the loss is:

    .. math::
        \text{loss} = \frac{1}{2}\left(\log\left(\text{max}\left(\text{var},
        \ \text{eps}\right)\right) + \frac{\left(\text{input} - \text{target}\right)^2}
        {\text{max}\left(\text{var}, \ \text{eps}\right)}\right) + \text{const.}

    where :attr:`eps` is used for stability. By default, the constant term of
    the loss function is omitted unless :attr:`full` is ``True``. If ``var`` is not the same
    size as ``input`` (due to a homoscedastic assumption), it must either have a final dimension
    of 1 or have one fewer dimension (with all other sizes being the same) for correct broadcasting.

    Args:
        full (bool, optional): include the constant term in the loss
            calculation. Default: ``False``.
        eps (float, optional): value used to clamp ``var`` (see note below), for
            stability. Default: 1e-6.
        reduction (str, optional): specifies the reduction to apply to the
            output:``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
            will be applied, ``'mean'``: the output is the average of all batch
            member losses, ``'sum'``: the output is the sum of all batch member
            losses. Default: ``'mean'``.

    Shape:
        - Input: :math:`(N, *)` or :math:`(*)` where :math:`*` means any number of additional
          dimensions
        - Target: :math:`(N, *)` or :math:`(*)`, same shape as the input, or same shape as the input
          but with one dimension equal to 1 (to allow for broadcasting)
        - Var: :math:`(N, *)` or :math:`(*)`, same shape as the input, or same shape as the input but
          with one dimension equal to 1, or same shape as the input but with one fewer
          dimension (to allow for broadcasting)
        - Output: scalar if :attr:`reduction` is ``'mean'`` (default) or
          ``'sum'``. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same
          shape as the input

    Examples::
        >>> loss = nn.GaussianNLLLoss()
        >>> input = torch.randn(5, 2, requires_grad=True)
        >>> target = torch.randn(5, 2)
        >>> var = torch.ones(5, 2, requires_grad=True)  # heteroscedastic
        >>> output = loss(input, target, var)
        >>> output.backward()

        >>> loss = nn.GaussianNLLLoss()
        >>> input = torch.randn(5, 2, requires_grad=True)
        >>> target = torch.randn(5, 2)
        >>> var = torch.ones(5, 1, requires_grad=True)  # homoscedastic
        >>> output = loss(input, target, var)
        >>> output.backward()
    """
    __constants__ = ["full", "eps", "reduction"]
    full: bool
    eps: float

    def __init__(
        self, *, full: bool = False, eps: float = 1e-6, reduction: str = "mean"
    ) -> None:
        # 调用父类的构造函数，初始化父类的属性
        super().__init__(None, None, reduction)
        # 设置对象的属性值
        self.full = full  # 控制是否计算完整的 NLL 损失
        self.eps = eps    # 控制方差的平滑值
        self.reduction = reduction  # 控制损失的归约方式（如平均或总和）

    def forward(self, input: Tensor, target: Tensor, var: Tensor) -> Tensor:
        # 调用 F 模块中的高斯负对数似然损失函数，传入参数和对象的属性
        return F.gaussian_nll_loss(
            input, target, var, full=self.full, eps=self.eps, reduction=self.reduction
        )
# 定义了一个 Kullback-Leibler 散度损失的类，继承自 PyTorch 的 _Loss 类
class KLDivLoss(_Loss):
    r"""The Kullback-Leibler divergence loss.

    # 用于相同形状的张量 y_pred 和 y_true 的 KL 散度计算
    For tensors of the same shape :math:`y_{\text{pred}},\ y_{\text{true}}`,
    where :math:`y_{\text{pred}}` is the :attr:`input` and :math:`y_{\text{true}}` is the
    :attr:`target`, we define the **pointwise KL-divergence** as

    # KL 散度的点对点定义
    .. math::

        L(y_{\text{pred}},\ y_{\text{true}})
            = y_{\text{true}} \cdot \log \frac{y_{\text{true}}}{y_{\text{pred}}}
            = y_{\text{true}} \cdot (\log y_{\text{true}} - \log y_{\text{pred}})

    # 为避免在计算时的下溢问题，该损失函数期望参数 input 是对数空间中的值
    To avoid underflow issues when computing this quantity, this loss expects the argument
    :attr:`input` in the log-space. The argument :attr:`target` may also be provided in the
    log-space if :attr:`log_target`\ `= True`.

    # 总结来说，该函数大致等同于以下计算
    To summarise, this function is roughly equivalent to computing

    .. code-block:: python

        if not log_target: # default
            loss_pointwise = target * (target.log() - input)
        else:
            loss_pointwise = target.exp() * (target - input)

    # 然后根据 reduction 参数来减少这个结果
    and then reducing this result depending on the argument :attr:`reduction` as

    .. code-block:: python

        if reduction == "mean":  # default
            loss = loss_pointwise.mean()
        elif reduction == "batchmean":  # mathematically correct
            loss = loss_pointwise.sum() / input.size(0)
        elif reduction == "sum":
            loss = loss_pointwise.sum()
        else:  # reduction == "none"
            loss = loss_pointwise

    # 提示：与 PyTorch 中的所有其他损失函数一样，该函数期望第一个参数 input 是模型的输出，第二个参数 target 是数据集中的观测值
    .. note::
        As all the other losses in PyTorch, this function expects the first argument,
        :attr:`input`, to be the output of the model (e.g. the neural network)
        and the second, :attr:`target`, to be the observations in the dataset.
        This differs from the standard mathematical notation :math:`KL(P\ ||\ Q)` where
        :math:`P` denotes the distribution of the observations and :math:`Q` denotes the model.

    # 警告：当 reduction="mean" 时，该函数并不返回真正的 KL 散度值，请使用 reduction="batchmean"，它符合数学定义
    .. warning::
        :attr:`reduction`\ `= "mean"` doesn't return the true KL divergence value, please use
        :attr:`reduction`\ `= "batchmean"` which aligns with the mathematical definition.
    Args:
        size_average (bool, optional): 是否对每个损失元素进行平均。注意，某些损失函数每个样本可能有多个元素。
            如果 `size_average` 设置为 `False`，则对每个 minibatch 进行求和。当 `reduce` 为 `False` 时忽略。默认为 `True`。
        reduce (bool, optional): 是否对每个 minibatch 的观测值进行平均或求和，取决于 `size_average`。当 `reduce` 设置为 `False` 时，
            返回每个 batch 元素的损失，并忽略 `size_average`。默认为 `True`。
        reduction (str, optional): 指定应用于输出的减少方式。默认为 `"mean"`。
        log_target (bool, optional): 指定 `target` 是否处于对数空间。默认为 `False`。

    Shape:
        - Input: :math:`(*)`，其中 :math:`*` 表示任意数量的维度。
        - Target: :math:`(*)`，与输入具有相同的形状。
        - Output: 默认情况下为标量。如果 `reduction` 是 `'none'`，则为 :math:`(*)`，与输入具有相同的形状。

    Examples::
        >>> kl_loss = nn.KLDivLoss(reduction="batchmean")
        >>> # input 应该是对数空间中的分布
        >>> input = F.log_softmax(torch.randn(3, 5, requires_grad=True), dim=1)
        >>> # 从数据集中抽样一个 batch 的分布。通常这些数据来自数据集
        >>> target = F.softmax(torch.rand(3, 5), dim=1)
        >>> output = kl_loss(input, target)

        >>> kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        >>> log_target = F.log_softmax(torch.rand(3, 5), dim=1)
        >>> output = kl_loss(input, log_target)
    """
    __constants__ = ["reduction"]

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        log_target: bool = False,
    ) -> None:
        # 调用父类构造函数初始化损失函数的参数
        super().__init__(size_average, reduce, reduction)
        # 设置是否使用对数空间的目标
        self.log_target = log_target

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # 调用 `F.kl_div` 函数计算 KL 散度损失
        return F.kl_div(
            input, target, reduction=self.reduction, log_target=self.log_target
        )
# 定义一个名为 MSELoss 的类，继承自 _Loss 类
class MSELoss(_Loss):
    # 创建一个用于计算均方误差的标准，即每个输入元素 x 与目标元素 y 之间的平方 L2 范数差的平均值或总和
    r"""Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input :math:`x` and target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{`sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The mean operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.

    Examples::

        >>> loss = nn.MSELoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    
    # 定义类属性 __constants__，指定在实例化时具有固定值的常量
    __constants__ = ["reduction"]

    # 初始化函数，用于初始化 MSELoss 类的实例
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        # 调用父类 _Loss 的初始化方法
        super().__init__(size_average, reduce, reduction)
    # 定义一个名为 forward 的方法，接受两个参数 input 和 target，都应为张量类型，返回一个张量
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # 使用 PyTorch 中的均方误差损失函数 F.mse_loss 计算 input 和 target 之间的均方误差损失
        # reduction 参数指定损失的计算方式，可以是 'mean'、'sum' 或 'none'，由 self.reduction 控制
        return F.mse_loss(input, target, reduction=self.reduction)
# 定义 BCELoss 类，继承自 _WeightedLoss 类
class BCELoss(_WeightedLoss):
    r"""Creates a criterion that measures the Binary Cross Entropy between the target and
    the input probabilities:
    
    # 构造函数说明了这个损失函数用来计算目标和输入概率之间的二元交叉熵

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:
    
    # 未减少的损失（即 :attr:`reduction` 设置为 `'none'`）可以描述为:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],

    # 其中 :math:`N` 是批次大小。如果 :attr:`reduction` 不是 `'none'`
    # （默认为 `'mean'`），则

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}

    # 这用于测量例如自动编码器中重建的误差。注意目标 :math:`y` 应为 0 到 1 之间的数值。

    Notice that if :math:`x_n` is either 0 or 1, one of the log terms would be
    mathematically undefined in the above loss equation. PyTorch chooses to set
    :math:`\log (0) = -\infty`, since :math:`\lim_{x\to 0} \log (x) = -\infty`.
    However, an infinite term in the loss equation is not desirable for several reasons.

    # 如果 :math:`x_n` 为 0 或 1，则上述损失方程中的某个对数项在数学上是未定义的。
    # PyTorch 选择将 :math:`\log (0) = -\infty`，因为 :math:`\lim_{x\to 0} \log (x) = -\infty`。
    # 然而，损失方程中的无限项出现并不理想，原因有几点。

    For one, if either :math:`y_n = 0` or :math:`(1 - y_n) = 0`, then we would be
    multiplying 0 with infinity. Secondly, if we have an infinite loss value, then
    we would also have an infinite term in our gradient, since
    :math:`\lim_{x\to 0} \frac{d}{dx} \log (x) = \infty`.
    This would make BCELoss's backward method nonlinear with respect to :math:`x_n`,
    and using it for things like linear regression would not be straight-forward.

    # 首先，如果 :math:`y_n = 0` 或 :math:`(1 - y_n) = 0`，那么我们将会
    # 将 0 乘以无穷大。其次，如果我们有一个无限的损失值，那么我们的梯度中也会有一个无限项，
    # 因为 :math:`\lim_{x\to 0} \frac{d}{dx} \log (x) = \infty`。
    # 这将使得 BCELoss 的反向传播方法对 :math:`x_n` 非线性化，
    # 并且将其用于诸如线性回归之类的事物将不会那么直接。

    Our solution is that BCELoss clamps its log function outputs to be greater than
    or equal to -100. This way, we can always have a finite loss value and a linear
    backward method.
    
    # 我们的解决方案是，BCELoss 将其对数函数的输出夹紧到大于或等于 -100。
    # 这样，我们总是可以得到一个有限的损失值和线性的反向传播方法。
    """
    Args:
        weight (Tensor, optional): 每个批次元素的损失手动重新缩放权重。如果提供，则必须是大小为 `nbatch` 的张量。
        size_average (bool, optional): 已弃用（参见 :attr:`reduction`）。默认情况下，损失在批次中的每个损失元素上进行平均。注意，对于某些损失，每个样本可能有多个元素。如果字段 :attr:`size_average` 设置为 ``False``，则损失将对每个小批量进行求和。当 :attr:`reduce` 为 ``False`` 时忽略。默认值为 ``True``。
        reduce (bool, optional): 已弃用（参见 :attr:`reduction`）。默认情况下，损失对每个小批量的观察进行平均或求和，具体取决于 :attr:`size_average`。当 :attr:`reduce` 为 ``False`` 时，返回每个批次元素的损失，并忽略 :attr:`size_average`。默认值为 ``True``。
        reduction (str, optional): 指定要应用于输出的减少方式：“none” | “mean” | “sum”。``'none'``: 不应用任何减少，“mean”：输出的总和将除以输出中的元素数量，“sum”：输出将被求和。注意： :attr:`size_average` 和 :attr:`reduce` 正在被弃用，在此期间，指定这两个参数之一将覆盖 :attr:`reduction`。默认值为 ``'mean'``
    
    Shape:
        - Input: :math:`(*)`，其中 :math:`*` 表示任意数量的维度。
        - Target: :math:`(*)`，与输入具有相同的形状。
        - Output: 标量。如果 :attr:`reduction` 为 ``'none'``，则为 :math:`(*)`，与输入具有相同的形状。
    
    Examples::
    
        >>> m = nn.Sigmoid()
        >>> loss = nn.BCELoss()
        >>> input = torch.randn(3, 2, requires_grad=True)
        >>> target = torch.rand(3, 2, requires_grad=False)
        >>> output = loss(m(input), target)
        >>> output.backward()
    """
    
    __constants__ = ["reduction"]
    
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        """
        初始化函数，设置损失函数的参数。
    
        Args:
            weight (Optional[Tensor], optional): 权重张量，默认为 None。
            size_average (bool, optional): 已弃用，曾用于指定是否对每个批次元素进行平均。现在由 reduction 参数控制。默认为 None。
            reduce (bool, optional): 已弃用，曾用于指定是否减少输出。现在由 reduction 参数控制。默认为 None。
            reduction (str, optional): 指定输出的减少方式。默认为 "mean"。
        """
        super().__init__(weight, size_average, reduce, reduction)
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        前向传播函数，计算二分类交叉熵损失。
    
        Args:
            input (Tensor): 神经网络模型的输出，经过 Sigmoid 激活函数后的结果。
            target (Tensor): 目标张量，与输入具有相同的形状，用于比较。
    
        Returns:
            Tensor: 计算得到的损失张量。
        """
        return F.binary_cross_entropy(
            input, target, weight=self.weight, reduction=self.reduction
        )
class BCEWithLogitsLoss(_Loss):
    r"""This loss combines a `Sigmoid` layer and the `BCELoss` in one single
    class. This version is more numerically stable than using a plain `Sigmoid`
    followed by a `BCELoss` as, by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log \sigma(x_n)
        + (1 - y_n) \cdot \log (1 - \sigma(x_n)) \right],

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}

    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets `t[i]` should be numbers
    between 0 and 1.

    It's possible to trade off recall and precision by adding weights to positive examples.
    In the case of multi-label classification the loss can be described as:

    .. math::
        \ell_c(x, y) = L_c = \{l_{1,c},\dots,l_{N,c}\}^\top, \quad
        l_{n,c} = - w_{n,c} \left[ p_c y_{n,c} \cdot \log \sigma(x_{n,c})
        + (1 - y_{n,c}) \cdot \log (1 - \sigma(x_{n,c})) \right],

    where :math:`c` is the class number (:math:`c > 1` for multi-label binary classification,
    :math:`c = 1` for single-label binary classification),
    :math:`n` is the number of the sample in the batch and
    :math:`p_c` is the weight of the positive answer for the class :math:`c`.

    :math:`p_c > 1` increases the recall, :math:`p_c < 1` increases the precision.

    For example, if a dataset contains 100 positive and 300 negative examples of a single class,
    then ``pos_weight`` for the class should be equal to :math:`\frac{300}{100}=3`.
    The loss would act as if the dataset contains :math:`3\times 100=300` positive examples.

    Examples::

        >>> target = torch.ones([10, 64], dtype=torch.float32)  # 64 classes, batch size = 10
        >>> output = torch.full([10, 64], 1.5)  # A prediction (logit)
        >>> pos_weight = torch.ones([64])  # All weights are equal to 1
        >>> criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        >>> criterion(output, target)  # -log(sigmoid(1.5))
        tensor(0.20...)

    In the above example, the ``pos_weight`` tensor's elements correspond to the 64 distinct classes
    in a multi-label binary classification scenario. Each element in ``pos_weight`` is designed to adjust the
    loss function based on the imbalance between negative and positive samples for the respective class.
    This approach is useful in datasets with varying levels of class imbalance, ensuring that the loss
    function effectively prioritizes the less frequent positive examples.
    """

    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        """
        Initializes the BCEWithLogitsLoss.

        Args:
            weight (Tensor, optional): a manual rescaling weight given to the loss of each batch element.
                                       If given, has to be a Tensor of size "nbatch".
            size_average (bool, optional): Deprecated (see reduction). By default, this parameter has no effect.
            reduce (bool, optional): Deprecated (see reduction). By default, this parameter has no effect.
            reduction (string, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                                          'none': no reduction will be applied,
                                          'mean': the sum of the output will be divided by the number of elements
                                                  in the output,
                                          'sum': the output will be summed.
                                          Default: 'mean'.
            pos_weight (Tensor, optional): a weight of positive examples. Must be a vector with length equal to the
                                           number of classes.

        Notes:
            - :attr:`size_average` and :attr:`reduce` are in the process of being deprecated,
              and in the meantime, specifying them will override :attr:`reduction`.
        """

        super(BCEWithLogitsLoss, self).__init__(reduction=reduction)
        self.weight = weight
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, input, target):
        """
        Defines the computation performed at every call.

        Args:
            input (Tensor): input tensor of arbitrary shape
            target (Tensor): target tensor of the same shape as input

        Returns:
            Tensor: the computed loss

        Computes the binary cross entropy loss between input tensor `input` and a target tensor `target`.
        """

        return F.binary_cross_entropy_with_logits(input, target,
                                                  self.weight,
                                                  pos_weight=self.pos_weight,
                                                  reduction=self.reduction)
    calculation accurately accounts for the distribution in each class.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size `nbatch`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        pos_weight (Tensor, optional): a weight of positive examples to be broadcasted with target.
            Must be a tensor with equal size along the class dimension to the number of classes.
            Pay close attention to PyTorch's broadcasting semantics in order to achieve the desired
            operations. For a target of size [B, C, H, W] (where B is batch size) pos_weight of
            size [B, C, H, W] will apply different pos_weights to each element of the batch or
            [C, H, W] the same pos_weights across the batch. To apply the same positive weight
            along all spacial dimensions for a 2D multi-class target [C, H, W] use: [C, 1, 1].
            Default: ``None``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(*)`, same
          shape as input.

     Examples::

        >>> loss = nn.BCEWithLogitsLoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    
    # 初始化函数，定义了 BCEWithLogitsLoss 类的构造方法
    def __init__(
        self,
        weight: Optional[Tensor] = None,  # 可选参数，用于对每个批次元素的损失进行手动缩放
        size_average=None,  # 弃用的参数（参见 reduction），默认情况下对每个批次元素的损失进行平均
        reduce=None,  # 弃用的参数（参见 reduction），默认情况下根据 size_average 对每个批次的观察结果进行平均或求和
        reduction: str = "mean",  # 指定输出的缩减方式：'none' | 'mean' | 'sum'，默认为 'mean'
        pos_weight: Optional[Tensor] = None,  # 可选参数，用于与目标进行广播的正例权重
    ) -> None:
        # 调用父类的初始化方法，设定损失函数的计算方式
        super().__init__(size_average, reduce, reduction)
        # 将权重参数注册为缓冲区，使其在模型保存和加载时能够保持状态
        self.register_buffer("weight", weight)
        # 将正权重参数注册为缓冲区
        self.register_buffer("pos_weight", pos_weight)
        # 声明权重和正权重的类型为可选张量
        self.weight: Optional[Tensor]
        self.pos_weight: Optional[Tensor]

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # 调用 PyTorch 中的二元交叉熵函数，计算损失
        return F.binary_cross_entropy_with_logits(
            input,                      # 模型输出的 logits
            target,                     # 目标标签
            self.weight,                # 权重参数
            pos_weight=self.pos_weight, # 正权重参数
            reduction=self.reduction,   # 损失函数的降维方式
        )
# 定义一个名为 HingeEmbeddingLoss 的损失函数类，继承自 _Loss 类
class HingeEmbeddingLoss(_Loss):
    r"""Measures the loss given an input tensor :math:`x` and a labels tensor :math:`y`
    (containing 1 or -1).
    This is usually used for measuring whether two inputs are similar or
    dissimilar, e.g. using the L1 pairwise distance as :math:`x`, and is typically
    used for learning nonlinear embeddings or semi-supervised learning.

    The loss function for :math:`n`-th sample in the mini-batch is

    .. math::
        l_n = \begin{cases}
            x_n, & \text{if}\; y_n = 1,\\
            \max \{0, margin - x_n\}, & \text{if}\; y_n = -1,
        \end{cases}

    and the total loss functions is

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}

    where :math:`L = \{l_1,\dots,l_N\}^\top`.

    Args:
        margin (float, optional): Has a default value of `1`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(*)` where :math:`*` means, any number of dimensions. The sum operation
          operates over all the elements.
        - Target: :math:`(*)`, same shape as the input
        - Output: scalar. If :attr:`reduction` is ``'none'``, then same shape as the input
    """
    # 常量声明：包含 margin 和 reduction
    __constants__ = ["margin", "reduction"]
    margin: float

    # 初始化函数，设置损失函数的参数
    def __init__(
        self,
        margin: float = 1.0,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        # 调用父类 _Loss 的初始化函数
        super().__init__(size_average, reduce, reduction)
        # 设置损失函数的 margin 属性
        self.margin = margin
    # 定义一个方法 `forward`，用于计算 Hinge Embedding Loss
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # 调用 PyTorch 中的 Hinge Embedding Loss 函数 `F.hinge_embedding_loss`
        # 输入参数包括 input（模型的输出）、target（目标值）、margin（边界值）、reduction（减少方式）
        return F.hinge_embedding_loss(
            input, target, margin=self.margin, reduction=self.reduction
        )
class MultiLabelMarginLoss(_Loss):
    r"""Creates a criterion that optimizes a multi-class multi-classification
    hinge loss (margin-based loss) between input :math:`x` (a 2D mini-batch `Tensor`)
    and output :math:`y` (which is a 2D `Tensor` of target class indices).
    For each sample in the mini-batch:

    .. math::
        \text{loss}(x, y) = \sum_{ij}\frac{\max(0, 1 - (x[y[j]] - x[i]))}{\text{x.size}(0)}

    where :math:`x \in \left\{0, \; \cdots , \; \text{x.size}(0) - 1\right\}`, \
    :math:`y \in \left\{0, \; \cdots , \; \text{y.size}(0) - 1\right\}`, \
    :math:`0 \leq y[j] \leq \text{x.size}(0)-1`, \
    and :math:`i \neq y[j]` for all :math:`i` and :math:`j`.

    :math:`y` and :math:`x` must have the same size.

    The criterion only considers a contiguous block of non-negative targets that
    starts at the front.

    This allows for different samples to have variable amounts of target classes.

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(C)` or :math:`(N, C)` where `N` is the batch size and `C`
          is the number of classes.
        - Target: :math:`(C)` or :math:`(N, C)`, label targets padded by -1 ensuring same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(N)`.

    Examples::

        >>> loss = nn.MultiLabelMarginLoss()
        >>> x = torch.FloatTensor([[0.1, 0.2, 0.4, 0.8]])
        >>> # for target y, only consider labels 3 and 0, not after label -1
        >>> y = torch.LongTensor([[3, 0, -1, 1]])
        >>> # 0.25 * ((1-(0.1-0.2)) + (1-(0.1-0.4)) + (1-(0.8-0.2)) + (1-(0.8-0.4)))
        >>> loss(x, y)
        tensor(0.85...)

    """

    def __init__(self, size_average=True, reduce=True, reduction='mean'):
        # 初始化父类 _Loss，设置默认参数
        super(MultiLabelMarginLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # 计算损失函数
        return F.multi_label_margin_loss(input, target, reduction=self.reduction)
    # 定义类的常量，这里只有一个"reduction"
    __constants__ = ["reduction"]
    
    # 初始化函数，用于设置损失函数的参数
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        # 调用父类的初始化方法，传递损失函数的参数
        super().__init__(size_average, reduce, reduction)
    
    # 前向传播函数，计算多标签边界损失函数
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # 调用函数库中的多标签边界损失函数，传递输入数据和目标数据，设定损失函数的减少方式
        return F.multilabel_margin_loss(input, target, reduction=self.reduction)
# 定义 SmoothL1Loss 类，继承自 _Loss 类
class SmoothL1Loss(_Loss):
    r"""Creates a criterion that uses a squared term if the absolute
    element-wise error falls below beta and an L1 term otherwise.
    It is less sensitive to outliers than :class:`torch.nn.MSELoss` and in some cases
    prevents exploding gradients (e.g. see the paper `Fast R-CNN`_ by Ross Girshick).

    For a batch of size :math:`N`, the unreduced loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1, ..., l_N\}^T

    with

    .. math::
        l_n = \begin{cases}
        0.5 (x_n - y_n)^2 / beta, & \text{if } |x_n - y_n| < beta \\
        |x_n - y_n| - 0.5 * beta, & \text{otherwise }
        \end{cases}

    If `reduction` is not `none`, then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{`sum'.}
        \end{cases}

    .. note::
        Smooth L1 loss can be seen as exactly :class:`L1Loss`, but with the :math:`|x - y| < beta`
        portion replaced with a quadratic function such that its slope is 1 at :math:`|x - y| = beta`.
        The quadratic segment smooths the L1 loss near :math:`|x - y| = 0`.

    .. note::
        Smooth L1 loss is closely related to :class:`HuberLoss`, being
        equivalent to :math:`huber(x, y) / beta` (note that Smooth L1's beta hyper-parameter is
        also known as delta for Huber). This leads to the following differences:

        * As beta -> 0, Smooth L1 loss converges to :class:`L1Loss`, while :class:`HuberLoss`
          converges to a constant 0 loss. When beta is 0, Smooth L1 loss is equivalent to L1 loss.
        * As beta -> :math:`+\infty`, Smooth L1 loss converges to a constant 0 loss, while
          :class:`HuberLoss` converges to :class:`MSELoss`.
        * For Smooth L1 loss, as beta varies, the L1 segment of the loss has a constant slope of 1.
          For :class:`HuberLoss`, the slope of the L1 segment is beta.

    .. _`Fast R-CNN`: https://arxiv.org/abs/1504.08083
    """
    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        beta (float, optional): Specifies the threshold at which to change between L1 and L2 loss.
            The value must be non-negative. Default: 1.0
    
    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(*)`, same shape as the input.
    """
    __constants__ = ["reduction"]
    
    def __init__(
        self, size_average=None, reduce=None, reduction: str = "mean", beta: float = 1.0
    ) -> None:
        """
        初始化方法，设置损失函数的参数和属性。
    
        Args:
            size_average (bool, optional): 是否对每个损失元素在批次中进行平均。如果为 ``False``，
                则在每个小批次中对损失进行求和。在 :attr:`reduce` 设置为 ``False`` 时被忽略。默认为 ``True``。
            reduce (bool, optional): 是否在每个小批次中对观测进行平均或求和，取决于 :attr:`size_average`。
                当 :attr:`reduce` 为 ``False`` 时，返回每个批次元素的损失，并忽略 :attr:`size_average`。默认为 ``True``。
            reduction (str, optional): 指定应用于输出的减少方式：
                ``'none'`` | ``'mean'`` | ``'sum'``。 ``'none'``：不应用减少，
                ``'mean'``：输出总和将被输出中的元素数量除以，``'sum'``：输出将被求和。注意：:attr:`size_average`
                和 :attr:`reduce` 正在被弃用，同时指定这两个参数将覆盖 :attr:`reduction`。默认为 ``'mean'``。
            beta (float, optional): 指定从L1损失到L2损失之间转换的阈值。值必须为非负数。默认为 1.0。
        """
        super().__init__(size_average, reduce, reduction)
        self.beta = beta
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        前向传播方法，计算平滑的L1损失。
    
        Args:
            input (Tensor): 模型的预测输出。
            target (Tensor): 真实标签值。
    
        Returns:
            Tensor: 平滑的L1损失值。
        """
        return F.smooth_l1_loss(input, target, reduction=self.reduction, beta=self.beta)
# 定义 Huber Loss 类，继承自 _Loss 类
class HuberLoss(_Loss):
    r"""Creates a criterion that uses a squared term if the absolute
    element-wise error falls below delta and a delta-scaled L1 term otherwise.
    This loss combines advantages of both :class:`L1Loss` and :class:`MSELoss`; the
    delta-scaled L1 region makes the loss less sensitive to outliers than :class:`MSELoss`,
    while the L2 region provides smoothness over :class:`L1Loss` near 0. See
    `Huber loss <https://en.wikipedia.org/wiki/Huber_loss>`_ for more information.

    For a batch of size :math:`N`, the unreduced loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1, ..., l_N\}^T

    with

    .. math::
        l_n = \begin{cases}
        0.5 (x_n - y_n)^2, & \text{if } |x_n - y_n| < delta \\
        delta * (|x_n - y_n| - 0.5 * delta), & \text{otherwise }
        \end{cases}

    If `reduction` is not `none`, then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{`sum'.}
        \end{cases}

    .. note::
        When delta is set to 1, this loss is equivalent to :class:`SmoothL1Loss`.
        In general, this loss differs from :class:`SmoothL1Loss` by a factor of delta (AKA beta
        in Smooth L1).
        See :class:`SmoothL1Loss` for additional discussion on the differences in behavior
        between the two losses.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        delta (float, optional): Specifies the threshold at which to change between delta-scaled L1 and L2 loss.
            The value must be positive.  Default: 1.0

    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(*)`, same shape as the input.
    """
    # 定义常量列表，指定类的常量属性
    __constants__ = ["reduction", "delta"]

    # 初始化方法，接受 reduction 和 delta 两个参数
    def __init__(self, reduction: str = "mean", delta: float = 1.0) -> None:
        # 调用父类 _Loss 的初始化方法，传入 reduction 参数
        super().__init__(reduction=reduction)
        # 设置 delta 属性为传入的 delta 参数
        self.delta = delta

    # 前向传播方法，接受 input 和 target 两个张量参数，返回损失值张量
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # 调用 torch.nn.functional 的 huber_loss 函数计算 Huber 损失
        # 根据传入的 reduction 和 delta 参数进行计算
        return F.huber_loss(input, target, reduction=self.reduction, delta=self.delta)


# 定义 SoftMargin Loss 类，继承自 _Loss 类
class SoftMarginLoss(_Loss):
    r"""Creates a criterion that optimizes a two-class classification
    logistic loss between input tensor :math:`x` and target tensor :math:`y`
    (containing 1 or -1).

    .. math::
        \text{loss}(x, y) = \sum_i \frac{\log(1 + \exp(-y[i]*x[i]))}{\text{x.nelement}()}

    """
    # 此处省略了 __constants__ 属性

    # 初始化方法，不接受任何参数，默认使用 mean reduction
    def __init__(self) -> None:
        # 调用父类 _Loss 的初始化方法，指定 reduction 为 'mean'
        super().__init__()

    # 前向传播方法，接受 input 和 target 两个张量参数，返回损失值张量
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # 返回 Soft Margin Loss，调用 torch.nn.functional 的 soft_margin_loss 函数计算
        # 损失函数为 log(1 + exp(-y * x)) 的平均值
        return F.soft_margin_loss(input, target)
    """
    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
    
    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(*)`, same
          shape as input.
    
    """
    # 定义一个包含常量 "reduction" 的类
    __constants__ = ["reduction"]
    
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        # 调用父类的初始化方法
        super().__init__(size_average, reduce, reduction)
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # 调用函数 F.soft_margin_loss 计算 soft margin loss
        # 根据指定的 reduction 参数来处理输出
        return F.soft_margin_loss(input, target, reduction=self.reduction)
# 定义一个名为 CrossEntropyLoss 的类，它继承自 _WeightedLoss 类
class CrossEntropyLoss(_WeightedLoss):
    r"""This criterion computes the cross entropy loss between input logits
    and target.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    The `input` is expected to contain the unnormalized logits for each class (which do `not` need
    to be positive or sum to 1, in general).
    `input` has to be a Tensor of size :math:`(C)` for unbatched input,
    :math:`(minibatch, C)` or :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1` for the
    `K`-dimensional case. The last being useful for higher dimension inputs, such
    as computing cross entropy loss per-pixel for 2D images.

    The `target` that this criterion expects should contain either:

    - Class indices in the range :math:`[0, C)` where :math:`C` is the number of classes; if
      `ignore_index` is specified, this loss also accepts this class index (this index
      may not necessarily be in the class range). The unreduced (i.e. with :attr:`reduction`
      set to ``'none'``) loss for this case can be described as:

      .. math::
          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - w_{y_n} \log \frac{\exp(x_{n,y_n})}{\sum_{c=1}^C \exp(x_{n,c})}
          \cdot \mathbb{1}\{y_n \not= \text{ignore\_index}\}

      where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
      :math:`C` is the number of classes, and :math:`N` spans the minibatch dimension as well as
      :math:`d_1, ..., d_k` for the `K`-dimensional case. If
      :attr:`reduction` is not ``'none'`` (default ``'mean'``), then

      .. math::
          \ell(x, y) = \begin{cases}
              \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n} \cdot \mathbb{1}\{y_n \not= \text{ignore\_index}\}} l_n, &
               \text{if reduction} = \text{`mean';}\\
                \sum_{n=1}^N l_n,  &
                \text{if reduction} = \text{`sum'.}
            \end{cases}

      Note that this case is equivalent to applying :class:`~torch.nn.LogSoftmax`
      on an input, followed by :class:`~torch.nn.NLLLoss`.
    # 每个类别的概率；在需要每个 minibatch 项超出单一类别标签时有用，比如混合标签、标签平滑等。对应未归约的（即使用 `reduction` 设置为 `'none'` 的）损失可以描述为：
    
    # 这里使用 LaTeX 公式描述损失函数，其中：
    # - :math:`x` 是输入，
    # - :math:`y` 是目标，
    # - :math:`w` 是权重，
    # - :math:`C` 是类别数，
    # - :math:`N` 跨 minibatch 维度，以及对于 `K` 维情况下的 :math:`d_1, ..., d_k`。如果 `reduction` 不是 `'none'`（默认为 `'mean'`），则：
    
    # .. note::
    #     当 `target` 包含类别索引时，此准则的性能通常更好，因为这样可以进行优化计算。考虑在每个 minibatch 项仅含单一类别标签时，将 `target` 提供为类别概率。
   python
    Args:
        weight (Tensor, optional): A manual rescaling weight given to each class.
            If provided, should be a Tensor of size `C` and floating point dtype.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). Indicates whether
            to average the losses over each loss element in the batch (`True`) or sum them (`False`).
            Ignored if :attr:`reduce` is `False`. Default: `True`.
        ignore_index (int, optional): Specifies a target value to ignore, not contributing to gradients.
            Only applicable when the target contains class indices. Default: `-100`.
        reduce (bool, optional): Deprecated (see :attr:`reduction`). Determines whether to
            average or sum losses over observations for each minibatch (`True` for average, `False` for sum).
            Ignored if :attr:`size_average` is `False`. Default: `True`.
        reduction (str, optional): Specifies the reduction to apply to the output: `'none'`, `'mean'`, or `'sum'`.
            `'none'` applies no reduction, `'mean'` computes the weighted mean of output, `'sum'` computes the sum.
            Overrides :attr:`size_average` and :attr:`reduce`. Default: `'mean'`.
        label_smoothing (float, optional): Amount of smoothing in loss computation, where `0.0` means no smoothing.
            Targets are a mix of original ground truth and uniform distribution. Default: `0.0`.
    """
    __constants__ = ["ignore_index", "reduction", "label_smoothing"]
    定义类的常量列表，这些常量在类的整个生命周期中不变
    
    ignore_index: int
    初始化一个整数变量ignore_index，用于指定忽略的类别索引，默认为-100
    
    label_smoothing: float
    初始化一个浮点数变量label_smoothing，用于标签平滑（label smoothing），默认为0.0
    
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
    构造函数初始化CrossEntropyLoss对象的属性：
    
    weight: Optional[Tensor] = None
        权重张量，可选参数，默认为None
    
    size_average=None
        已弃用参数，不再使用
    
    ignore_index: int = -100
        要忽略的类别索引，默认为-100
    
    reduce=None
        已弃用参数，不再使用
    
    reduction: str = "mean"
        损失的减少方式，默认为"mean"
    
    label_smoothing: float = 0.0
        标签平滑参数，默认为0.0
    
    super().__init__(weight, size_average, reduce, reduction)
        调用父类(nn.Module)的构造函数，传递相应的参数
    
    self.ignore_index = ignore_index
        将输入的ignore_index参数赋值给实例的ignore_index属性
    
    self.label_smoothing = label_smoothing
        将输入的label_smoothing参数赋值给实例的label_smoothing属性
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        前向传播方法，计算输入input和目标target之间的交叉熵损失
    
    return F.cross_entropy(
        input,
        target,
        weight=self.weight,
        ignore_index=self.ignore_index,
        reduction=self.reduction,
        label_smoothing=self.label_smoothing,
    )
        调用torch.nn.functional中的cross_entropy函数计算输入和目标之间的交叉熵损失，传递相应的参数
    
    """
# 定义一个继承自 _WeightedLoss 的类 MultiLabelSoftMarginLoss，用于多标签分类问题的损失计算
class MultiLabelSoftMarginLoss(_WeightedLoss):
    r"""Creates a criterion that optimizes a multi-label one-versus-all
    loss based on max-entropy, between input :math:`x` and target :math:`y` of size
    :math:`(N, C)`.
    For each sample in the minibatch:

    .. math::
        loss(x, y) = - \frac{1}{C} * \sum_i y[i] * \log((1 + \exp(-x[i]))^{-1})
                         + (1-y[i]) * \log\left(\frac{\exp(-x[i])}{(1 + \exp(-x[i]))}\right)

    where :math:`i \in \left\{0, \; \cdots , \; \text{x.nElement}() - 1\}`,
    :math:`y[i] \in \left\{0, \; 1\}`.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`. Otherwise, it is
            treated as if having all ones.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, C)` where `N` is the batch size and `C` is the number of classes.
        - Target: :math:`(N, C)`, label targets must have the same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(N)`.

    """
    # 损失函数类的常量声明，指定使用的输出数据缩减方式
    __constants__ = ["reduction"]

    # 初始化函数，设置损失函数的参数
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        # 调用父类的初始化函数
        super().__init__(weight, size_average, reduce, reduction)

    # 前向传播函数，计算损失值
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # 调用 torch.nn.functional 中的多标签软间隔损失函数计算
        return F.multilabel_soft_margin_loss(
            input, target, weight=self.weight, reduction=self.reduction
        )
    """
        :math:`x_1`, :math:`x_2` and a `Tensor` label :math:`y` with values 1 or -1.
        Use (:math:`y=1`) to maximize the cosine similarity of two inputs, and (:math:`y=-1`) otherwise.
        This is typically used for learning nonlinear
        embeddings or semi-supervised learning.
    
        The loss function for each sample is:
    
        .. math::
            \text{loss}(x, y) =
            \begin{cases}
            1 - \cos(x_1, x_2), & \text{if } y = 1 \\
            \max(0, \cos(x_1, x_2) - \text{margin}), & \text{if } y = -1
            \end{cases}
    
        Args:
            margin (float, optional): Should be a number from :math:`-1` to :math:`1`,
                :math:`0` to :math:`0.5` is suggested. If :attr:`margin` is missing, the
                default value is :math:`0`.
            size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
                the losses are averaged over each loss element in the batch. Note that for
                some losses, there are multiple elements per sample. If the field :attr:`size_average`
                is set to ``False``, the losses are instead summed for each minibatch. Ignored
                when :attr:`reduce` is ``False``. Default: ``True``
            reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
                losses are averaged or summed over observations for each minibatch depending
                on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
                batch element instead and ignores :attr:`size_average`. Default: ``True``
            reduction (str, optional): Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
    
        Shape:
            - Input1: :math:`(N, D)` or :math:`(D)`, where `N` is the batch size and `D` is the embedding dimension.
            - Input2: :math:`(N, D)` or :math:`(D)`, same shape as Input1.
            - Target: :math:`(N)` or :math:`()`.
            - Output: If :attr:`reduction` is ``'none'``, then :math:`(N)`, otherwise scalar.
    
        Examples::
    
            >>> loss = nn.CosineEmbeddingLoss()
            >>> input1 = torch.randn(3, 5, requires_grad=True)
            >>> input2 = torch.randn(3, 5, requires_grad=True)
            >>> target = torch.ones(3)
            >>> output = loss(input1, input2, target)
            >>> output.backward()
    """
    __constants__ = ["margin", "reduction"]
    margin: float
    
    def __init__(
        self,
        margin: float = 0.0,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        # 调用父类的构造函数，初始化损失函数的参数
        super().__init__(size_average, reduce, reduction)
        # 设置自定义的 margin（余弦相似度的边界值）
        self.margin = margin

    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        # 调用 PyTorch 中的 cosine_embedding_loss 函数计算余弦嵌入损失
        return F.cosine_embedding_loss(
            input1, input2, target, margin=self.margin, reduction=self.reduction
        )
# MarginRankingLoss 类继承自 _Loss 类，用于计算基于边界排名的损失函数。
class MarginRankingLoss(_Loss):
    r"""Creates a criterion that measures the loss given
    inputs :math:`x1`, :math:`x2`, two 1D mini-batch or 0D `Tensors`,
    and a label 1D mini-batch or 0D `Tensor` :math:`y` (containing 1 or -1).

    If :math:`y = 1` then it assumed the first input should be ranked higher
    (have a larger value) than the second input, and vice-versa for :math:`y = -1`.

    The loss function for each pair of samples in the mini-batch is:

    .. math::
        \text{loss}(x1, x2, y) = \max(0, -y * (x1 - x2) + \text{margin})

    Args:
        margin (float, optional): Has a default value of :math:`0`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input1: :math:`(N)` or :math:`()` where `N` is the batch size.
        - Input2: :math:`(N)` or :math:`()`, same shape as the Input1.
        - Target: :math:`(N)` or :math:`()`, same shape as the inputs.
        - Output: scalar. If :attr:`reduction` is ``'none'`` and Input size is not :math:`()`, then :math:`(N)`.

    Examples::

        >>> loss = nn.MarginRankingLoss()
        >>> input1 = torch.randn(3, requires_grad=True)
        >>> input2 = torch.randn(3, requires_grad=True)
        >>> target = torch.randn(3).sign()
        >>> output = loss(input1, input2, target)
        >>> output.backward()
    """
    # 定义常量 __constants__，包含 margin 和 reduction
    __constants__ = ["margin", "reduction"]
    # margin 属性声明为 float 类型
    margin: float

    def __init__(
        self,
        margin: float = 0.0,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        # 调用父类 _Loss 的构造函数，初始化损失函数的参数
        super().__init__(size_average, reduce, reduction)
        # 设置本类的 margin 属性
        self.margin = margin
    # 定义一个方法 `forward`，用于计算两个输入张量之间的边缘排名损失
    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        # 调用 PyTorch 的 margin_ranking_loss 函数计算边缘排名损失
        return F.margin_ranking_loss(
            input1, input2, target, margin=self.margin, reduction=self.reduction
        )
class MultiMarginLoss(_WeightedLoss):
    r"""Creates a criterion that optimizes a multi-class classification hinge
    loss (margin-based loss) between input :math:`x` (a 2D mini-batch `Tensor`) and
    output :math:`y` (which is a 1D tensor of target class indices,
    :math:`0 \leq y \leq \text{x.size}(1)-1`):

    For each mini-batch sample, the loss in terms of the 1D input :math:`x` and scalar
    output :math:`y` is:

    .. math::
        \text{loss}(x, y) = \frac{\sum_i \max(0, \text{margin} - x[y] + x[i])^p}{\text{x.size}(0)}

    where :math:`i \in \left\{0, \; \cdots , \; \text{x.size}(0) - 1\right\}`
    and :math:`i \neq y`.

    Optionally, you can give non-equal weighting on the classes by passing
    a 1D :attr:`weight` tensor into the constructor.

    The loss function then becomes:

    .. math::
        \text{loss}(x, y) = \frac{\sum_i w[y] * \max(0, \text{margin} - x[y] + x[i])^p}{\text{x.size}(0)}

    Args:
        p (int, optional): Has a default value of :math:`1`. :math:`1` and :math:`2`
            are the only supported values. Defines the power in the formula.
        margin (float, optional): Has a default value of :math:`1`. Defines the margin value.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`. Otherwise, it is
            treated as if having all ones. Weight for each class in the loss calculation.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
    """
    # 定义一个神经网络损失函数，用于多类别边界损失
    class MultiMarginLoss(nn.Module):
        # 以下为类的文档字符串，描述输入、目标和输出的形状和含义
        """
        Shape:
            - Input: :math:`(N, C)` or :math:`(C)`, where :math:`N` is the batch size and :math:`C` is the number of classes.
            - Target: :math:`(N)` or :math:`()`, where each value is :math:`0 \leq \text{targets}[i] \leq C-1`.
            - Output: scalar. If :attr:`reduction` is ``'none'``, then same shape as the target.
    
        Examples::
    
            >>> loss = nn.MultiMarginLoss()
            >>> x = torch.tensor([[0.1, 0.2, 0.4, 0.8]])
            >>> y = torch.tensor([3])
            >>> # 0.25 * ((1-(0.8-0.1)) + (1-(0.8-0.2)) + (1-(0.8-0.4)))
            >>> loss(x, y)
            tensor(0.32...)
        """
        
        # 指定常量列表，包括 p、margin 和 reduction
        __constants__ = ["p", "margin", "reduction"]
        
        # 定义 margin 和 p 的类型注解
        margin: float
        p: int
    
        # 初始化函数，设置损失函数的参数和可选的权重、平均方式、减少方式、减少类型等
        def __init__(
            self,
            p: int = 1,
            margin: float = 1.0,
            weight: Optional[Tensor] = None,
            size_average=None,
            reduce=None,
            reduction: str = "mean",
        ) -> None:
            # 调用父类的初始化方法，传递权重、平均方式、减少方式和减少类型等参数
            super().__init__(weight, size_average, reduce, reduction)
            # 如果 p 不是 1 或 2，则抛出 ValueError
            if p != 1 and p != 2:
                raise ValueError("only p == 1 and p == 2 supported")
            # 如果权重不为 None 且维度不为 1，则抛出 ValueError
            if weight is not None and weight.dim() != 1:
                raise ValueError(
                    f"MultiMarginLoss: expected weight to be None or 1D tensor, got {weight.dim()}D instead"
                )
            # 将 p 和 margin 设置为对象的属性
            self.p = p
            self.margin = margin
    
        # 前向传播函数，接受输入张量 input 和目标张量 target，返回损失值张量
        def forward(self, input: Tensor, target: Tensor) -> Tensor:
            # 调用 torch.nn.functional 中的 multi_margin_loss 函数，传递输入、目标、p、margin、权重和减少方式等参数
            return F.multi_margin_loss(
                input,
                target,
                p=self.p,
                margin=self.margin,
                weight=self.weight,
                reduction=self.reduction,
            )
class TripletMarginLoss(_Loss):
    r"""Creates a criterion that measures the triplet loss given an input
    tensors :math:`x1`, :math:`x2`, :math:`x3` and a margin with a value greater than :math:`0`.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `a`, `p` and `n` (i.e., `anchor`, `positive examples` and `negative
    examples` respectively). The shapes of all input tensors should be
    :math:`(N, D)`.

    The distance swap is described in detail in the paper `Learning shallow
    convolutional feature descriptors with triplet losses`_ by
    V. Balntas, E. Riba et al.

    The loss function for each sample in the mini-batch is:

    .. math::
        L(a, p, n) = \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\}

    where

    .. math::
        d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p

    The norm is calculated using the specified p value and a small constant :math:`\varepsilon` is
    added for numerical stability.

    See also :class:`~torch.nn.TripletMarginWithDistanceLoss`, which computes the
    triplet margin loss for input tensors using a custom distance function.

    Args:
        margin (float, optional): Default: :math:`1`.
            Margin value added to the distance between anchor and negative examples.
        p (int, optional): The norm degree for pairwise distance. Default: :math:`2`.
            Degree of the norm used to compute distance.
        eps (float, optional): Small constant for numerical stability. Default: :math:`1e-6`.
            Small value added to the norm to prevent division by zero.
        swap (bool, optional): The distance swap is described in detail in the paper
            `Learning shallow convolutional feature descriptors with triplet losses` by
            V. Balntas, E. Riba et al. Default: ``False``.
            If True, swaps the positive and negative examples in the loss calculation.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
    """
    Shape:
        - Input: :math:`(N, D)` or :math:`(D)` where :math:`D` is the vector dimension.
        - Output: A Tensor of shape :math:`(N)` if :attr:`reduction` is ``'none'`` and
          input shape is :math:`(N, D)`; a scalar otherwise.

    Examples::

    >>> triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
    >>> anchor = torch.randn(100, 128, requires_grad=True)
    >>> positive = torch.randn(100, 128, requires_grad=True)
    >>> negative = torch.randn(100, 128, requires_grad=True)
    >>> output = triplet_loss(anchor, positive, negative)
    >>> output.backward()

    .. _Learning shallow convolutional feature descriptors with triplet losses:
        http://www.bmva.org/bmvc/2016/papers/paper119/index.html
    """
    __constants__ = ["margin", "p", "eps", "swap", "reduction"]
    margin: float
    p: float
    eps: float
    swap: bool

    # 初始化 TripletMarginLoss 类
    def __init__(
        self,
        margin: float = 1.0,
        p: float = 2.0,
        eps: float = 1e-6,
        swap: bool = False,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ):
        # 调用父类的初始化方法
        super().__init__(size_average, reduce, reduction)
        # 如果 margin 小于等于 0，则抛出异常
        if margin <= 0:
            raise ValueError(
                f"TripletMarginLoss: expected margin to be greater than 0, got {margin} instead"
            )
        # 设置 margin、p、eps、swap 属性
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap

    # 前向传播方法
    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        # 调用 F.triplet_margin_loss 方法计算损失
        return F.triplet_margin_loss(
            anchor,
            positive,
            negative,
            margin=self.margin,
            p=self.p,
            eps=self.eps,
            swap=self.swap,
            reduction=self.reduction,
        )
# 定义一个自定义的损失函数 TripletMarginWithDistanceLoss，继承自 _Loss 类
class TripletMarginWithDistanceLoss(_Loss):
    r"""Creates a criterion that measures the triplet loss given input
    tensors :math:`a`, :math:`p`, and :math:`n` (representing anchor,
    positive, and negative examples, respectively), and a nonnegative,
    real-valued function ("distance function") used to compute the relationship
    between the anchor and positive example ("positive distance") and the
    anchor and negative example ("negative distance").

    The unreduced loss (i.e., with :attr:`reduction` set to ``'none'``)
    can be described as:

    .. math::
        \ell(a, p, n) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_i = \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\}

    where :math:`N` is the batch size; :math:`d` is a nonnegative, real-valued function
    quantifying the closeness of two tensors, referred to as the :attr:`distance_function`;
    and :math:`margin` is a nonnegative margin representing the minimum difference
    between the positive and negative distances that is required for the loss to
    be 0.  The input tensors have :math:`N` elements each and can be of any shape
    that the distance function can handle.

    If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{`sum'.}
        \end{cases}

    See also :class:`~torch.nn.TripletMarginLoss`, which computes the triplet
    loss for input tensors using the :math:`l_p` distance as the distance function.
    Args:
        distance_function (Callable, optional): 一个非负实值函数，用于衡量两个张量之间的接近程度。如果未指定，
            将使用 `nn.PairwiseDistance`。默认值为 ``None``。
        margin (float, optional): 一个非负边界，表示使损失为0所需的正负距离之差的最小值。较大的边界惩罚那些
            负例与锚点不足够远的情况。默认值为 :math:`1`。
        swap (bool, optional): 是否使用论文中描述的距离交换策略，详见
            `Learning shallow convolutional feature descriptors with triplet losses` by
            V. Balntas, E. Riba et al. 如果为 True，并且正例比锚点更接近负例，则在损失计算中交换正例和锚点。
            默认值为 ``False``。
        reduction (str, optional): 指定应用于输出的（可选）减少操作：
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: 不应用减少操作，
            ``'mean'``: 输出总和将被输出元素数量除，``'sum'``: 输出将被求和。默认值为 ``'mean'``


    Shape:
        - Input: :math:`(N, *)`，其中 :math:`*` 表示距离函数支持的任意数量的附加维度。
        - Output: 如果 :attr:`reduction` 是 ``'none'``，则形状为 :math:`(N)` 的张量，否则为标量。

    Examples::

    >>> # 初始化嵌入
    >>> embedding = nn.Embedding(1000, 128)
    >>> anchor_ids = torch.randint(0, 1000, (1,))
    >>> positive_ids = torch.randint(0, 1000, (1,))
    >>> negative_ids = torch.randint(0, 1000, (1,))
    >>> anchor = embedding(anchor_ids)
    >>> positive = embedding(positive_ids)
    >>> negative = embedding(negative_ids)
    >>>
    >>> # 内置距离函数
    >>> triplet_loss = \
    >>>     nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance())
    >>> output = triplet_loss(anchor, positive, negative)
    >>> output.backward()
    >>>
    >>> # 自定义距离函数
    >>> def l_infinity(x1, x2):
    >>>     return torch.max(torch.abs(x1 - x2), dim=1).values
    >>>
    >>> # xdoctest: +SKIP("FIXME: Would call backwards a second time")
    >>> triplet_loss = (
    >>>     nn.TripletMarginWithDistanceLoss(distance_function=l_infinity, margin=1.5))
    >>> output = triplet_loss(anchor, positive, negative)
    >>> output.backward()
    >>>
    >>> # 自定义距离函数（使用 Lambda）
    >>> triplet_loss = (
    >>>     nn.TripletMarginWithDistanceLoss(
    >>>         distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y)))
    >>> output = triplet_loss(anchor, positive, negative)
    >>> output.backward()
    """
    Reference:
        V. Balntas, et al.: Learning shallow convolutional feature descriptors with triplet losses:
        http://www.bmva.org/bmvc/2016/papers/paper119/index.html
    """
    __constants__ = ["margin", "swap", "reduction"]
    # 声明常量列表包含 "margin", "swap", "reduction"
    
    margin: float
    swap: bool
    
    def __init__(
        self,
        *,
        distance_function: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        margin: float = 1.0,
        swap: bool = False,
        reduction: str = "mean",
    ):
        # 继承父类构造函数，传递特定的参数
        super().__init__(size_average=None, reduce=None, reduction=reduction)
        # 如果 margin 小于等于 0，抛出值错误异常
        if margin <= 0:
            raise ValueError(
                f"TripletMarginWithDistanceLoss: expected margin to be greater than 0, got {margin} instead"
            )
        # 根据传入的 distance_function 参数设置距离函数，如果为 None 则使用 PairwiseDistance()
        self.distance_function: Optional[Callable[[Tensor, Tensor], Tensor]] = (
            distance_function if distance_function is not None else PairwiseDistance()
        )
        # 设置 margin 和 swap 属性
        self.margin = margin
        self.swap = swap
    
    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        # 调用 F 模块的 triplet_margin_with_distance_loss 函数，计算三元组损失
        return F.triplet_margin_with_distance_loss(
            anchor,
            positive,
            negative,
            distance_function=self.distance_function,
            margin=self.margin,
            swap=self.swap,
            reduction=self.reduction,
        )
class CTCLoss(_Loss):
    r"""The Connectionist Temporal Classification loss.

    Calculates loss between a continuous (unsegmented) time series and a target sequence. CTCLoss sums over the
    probability of possible alignments of input to target, producing a loss value which is differentiable
    with respect to each input node. The alignment of input to target is assumed to be "many-to-one", which
    limits the length of the target sequence such that it must be :math:`\leq` the input length.

    Args:
        blank (int, optional): blank label. Default :math:`0`.
            指定空白标签的整数值，默认为 :math:`0`。
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output losses will be divided by the target lengths and
            then the mean over the batch is taken, ``'sum'``: the output losses will be summed.
            Default: ``'mean'``
            指定对输出应用的减少操作：
            ``'none'``：不应用任何减少操作，
            ``'mean'``：输出损失将被目标长度除以，然后取批次的平均值，
            ``'sum'``：输出损失将被求和。
            默认值为 ``'mean'``。
        zero_infinity (bool, optional):
            Whether to zero infinite losses and the associated gradients.
            Default: ``False``
            是否将无限损失和相关的梯度归零。
            默认为 ``False``。
            当输入太短以至于无法与目标对齐时，会出现无限损失。

    """
    # 定义函数的文档字符串，描述了输入和输出的张量的形状和含义
    Shape:
        # Log_probs: 大小为 (T, N, C) 或 (T, C) 的张量，
        # 其中 T = 输入长度，N = 批量大小，C = 类别数（包括空白符）。
        # 输出的对数概率（例如，使用 torch.nn.functional.log_softmax 获得）。
        - Log_probs: Tensor of size :math:`(T, N, C)` or :math:`(T, C)`,
          where :math:`T = \text{input length}`,
          :math:`N = \text{batch size}`, and
          :math:`C = \text{number of classes (including blank)}`.
          The logarithmized probabilities of the outputs (e.g. obtained with
          :func:`torch.nn.functional.log_softmax`).
          
        # Targets: 大小为 (N, S) 或 (\operatorname{sum}(\text{target\_lengths})) 的张量，
        # 其中 N = 批量大小，S = 最大目标长度（如果形状为 (N, S)）。代表目标序列。
        # 目标序列中的每个元素是一个类别索引。目标索引不能为空白符（默认为0）。
        # 在形状为 (N, S) 的情况下，目标被填充到最长序列的长度并堆叠。
        # 在形状为 (\operatorname{sum}(\text{target\_lengths})) 的情况下，
        # 假定目标未填充并在1维内连接。
        - Targets: Tensor of size :math:`(N, S)` or
          :math:`(\operatorname{sum}(\text{target\_lengths}))`,
          where :math:`N = \text{batch size}` and
          :math:`S = \text{max target length, if shape is } (N, S)`.
          It represents the target sequences. Each element in the target
          sequence is a class index. And the target index cannot be blank (default=0).
          In the :math:`(N, S)` form, targets are padded to the
          length of the longest sequence, and stacked.
          In the :math:`(\operatorname{sum}(\text{target\_lengths}))` form,
          the targets are assumed to be un-padded and
          concatenated within 1 dimension.
          
        # Input_lengths: 大小为 (N) 或 () 的元组或张量，
        # 其中 N = 批量大小。表示输入的长度（每个序列必须 <= T）。
        # 假设序列被填充到相等长度以实现掩码。
        - Input_lengths: Tuple or tensor of size :math:`(N)` or :math:`()`,
          where :math:`N = \text{batch size}`. It represents the lengths of the
          inputs (must each be :math:`\leq T`). And the lengths are specified
          for each sequence to achieve masking under the assumption that sequences
          are padded to equal lengths.
          
        # Target_lengths: 大小为 (N) 或 () 的元组或张量，
        # 其中 N = 批量大小。表示目标的长度。
        # 长度为每个序列指定以实现掩码。
        # 假设序列被填充到相等长度。如果目标形状为 (N,S)，target_lengths 是每个目标序列的停止索引 s_n，
        # 使得 ``target_n = targets[n,0:s_n]`` 对于批量中的每个目标。
        # 每个长度必须 <= S。
        # 如果目标给出为 1 维张量，该张量的总长度必须等于目标的总长度。
        - Target_lengths: Tuple or tensor of size :math:`(N)` or :math:`()`,
          where :math:`N = \text{batch size}`. It represents lengths of the targets.
          Lengths are specified for each sequence to achieve masking under the
          assumption that sequences are padded to equal lengths. If target shape is
          :math:`(N,S)`, target_lengths are effectively the stop index
          :math:`s_n` for each target sequence, such that ``target_n = targets[n,0:s_n]`` for
          each target in a batch. Lengths must each be :math:`\leq S`
          If the targets are given as a 1d tensor that is the concatenation of individual
          targets, the target_lengths must add up to the total length of the tensor.
          
        # Output: 如果 reduction 是 'mean'（默认）或 'sum'，则为标量。
        # 如果 reduction 是 'none'，则为 (N)（如果输入是批量化的）或 ()（如果输入未批量化）。
        # 其中 N = 批量大小。
        - Output: scalar if :attr:`reduction` is ``'mean'`` (default) or
          ``'sum'``. If :attr:`reduction` is ``'none'``, then :math:`(N)` if input is batched or
          :math:`()` if input is unbatched, where :math:`N = \text{batch size}`.
    Examples::

        >>> # 目标是要填充
        >>> T = 50      # 输入序列长度
        >>> C = 20      # 类别数（包括空白类）
        >>> N = 16      # 批量大小
        >>> S = 30      # 批次中最长目标序列的目标序列长度（填充长度）
        >>> S_min = 10  # 最小目标长度，用于演示目的
        >>>
        >>> # 初始化随机批量的输入向量，大小为 *size = (T,N,C)
        >>> input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
        >>>
        >>> # 初始化随机批量的目标（0 = 空白, 1:C = 类别）
        >>> target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
        >>>
        >>> input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
        >>> target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
        >>> ctc_loss = nn.CTCLoss()
        >>> loss = ctc_loss(input, target, input_lengths, target_lengths)
        >>> loss.backward()
        >>>
        >>>
        >>> # 目标是不进行填充
        >>> T = 50      # 输入序列长度
        >>> C = 20      # 类别数（包括空白类）
        >>> N = 16      # 批量大小
        >>>
        >>> # 初始化随机批量的输入向量，大小为 *size = (T,N,C)
        >>> input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
        >>> input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
        >>>
        >>> # 初始化随机批量的目标（0 = 空白, 1:C = 类别）
        >>> target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
        >>> target = torch.randint(low=1, high=C, size=(sum(target_lengths),), dtype=torch.long)
        >>> ctc_loss = nn.CTCLoss()
        >>> loss = ctc_loss(input, target, input_lengths, target_lengths)
        >>> loss.backward()
        >>>
        >>>
        >>> # 目标是不进行填充且不进行批处理（实际上是 N=1）
        >>> T = 50      # 输入序列长度
        >>> C = 20      # 类别数（包括空白类）
        >>>
        >>> # 初始化随机批量的输入向量，大小为 *size = (T,C)
        >>> # xdoctest: +SKIP("FIXME: doctest 中的错误")
        >>> input = torch.randn(T, C).log_softmax(1).detach().requires_grad_()
        >>> input_lengths = torch.tensor(T, dtype=torch.long)
        >>>
        >>> # 初始化随机批量的目标（0 = 空白, 1:C = 类别）
        >>> target_lengths = torch.randint(low=1, high=T, size=(), dtype=torch.long)
        >>> target = torch.randint(low=1, high=C, size=(target_lengths,), dtype=torch.long)
        >>> ctc_loss = nn.CTCLoss()
        >>> loss = ctc_loss(input, target, input_lengths, target_lengths)
        >>> loss.backward()
    """
    Reference:
        A. Graves et al.: Connectionist Temporal Classification:
        Labelling Unsegmented Sequence Data with Recurrent Neural Networks:
        https://www.cs.toronto.edu/~graves/icml_2006.pdf

    Note:
        In order to use CuDNN, the following must be satisfied: :attr:`targets` must be
        in concatenated format, all :attr:`input_lengths` must be `T`.  :math:`blank=0`,
        :attr:`target_lengths` :math:`\leq 256`, the integer arguments must be of
        dtype :attr:`torch.int32`.

        The regular implementation uses the (more common in PyTorch) `torch.long` dtype.


    Note:
        In some circumstances when using the CUDA backend with CuDNN, this operator
        may select a nondeterministic algorithm to increase performance. If this is
        undesirable, you can try to make the operation deterministic (potentially at
        a performance cost) by setting ``torch.backends.cudnn.deterministic =
        True``.
        Please see the notes on :doc:`/notes/randomness` for background.
    """
    __constants__ = ["blank", "reduction"]
    # 定义类的常量，这些常量是不可变的属性列表，包括了 'blank' 和 'reduction'

    blank: int
    zero_infinity: bool
    # 声明类的属性 blank 和 zero_infinity，分别表示 CTC 损失中的空白标签和是否将无穷大值置零

    def __init__(
        self, blank: int = 0, reduction: str = "mean", zero_infinity: bool = False
    ):
        # 初始化函数，设置类的初始属性
        super().__init__(reduction=reduction)
        # 调用父类的初始化方法，设置 reduction 属性
        self.blank = blank
        self.zero_infinity = zero_infinity
        # 设置类的属性 blank 和 zero_infinity

    def forward(
        self,
        log_probs: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        # 前向传播函数，计算 CTC 损失
        return F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            self.blank,
            self.reduction,
            self.zero_infinity,
        )
        # 调用 PyTorch 的函数 F.ctc_loss 计算 CTC 损失，传入相关参数
# TODO: L1HingeEmbeddingCriterion
# L1HingeEmbeddingCriterion 的具体功能和实现待添加

# TODO: MSECriterion weight
# MSECriterion 中 weight 参数的作用和用法待添加

# TODO: ClassSimplexCriterion
# ClassSimplexCriterion 的具体功能和实现待添加
```