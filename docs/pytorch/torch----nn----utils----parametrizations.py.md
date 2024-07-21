# `.\pytorch\torch\nn\utils\parametrizations.py`

```py
# mypy: allow-untyped-defs
# 引入必要的模块和类
from enum import auto, Enum
from typing import Optional

import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 中的函数模块
from torch import Tensor  # 导入 PyTorch 的张量类
from torch.nn.modules import Module  # 导入 PyTorch 的神经网络模块
from torch.nn.utils import parametrize  # 导入 PyTorch 的参数化工具

__all__ = ["orthogonal", "spectral_norm", "weight_norm"]  # 定义公开的模块成员列表


def _is_orthogonal(Q, eps=None):
    # 检查输入张量 Q 是否为正交矩阵
    n, k = Q.size(-2), Q.size(-1)
    Id = torch.eye(k, dtype=Q.dtype, device=Q.device)
    # 选择一个合理的 eps 值，但不要太大
    eps = 10.0 * n * torch.finfo(Q.dtype).eps
    return torch.allclose(Q.mH @ Q, Id, atol=eps)


def _make_orthogonal(A):
    """Assume that A is a tall matrix.

    Compute the Q factor s.t. A = QR (A may be complex) and diag(R) is real and non-negative.
    """
    # 对给定的 A 计算正交因子 Q，确保 A = QR，其中 R 的对角元素是实数且非负
    X, tau = torch.geqrf(A)
    Q = torch.linalg.householder_product(X, tau)
    # X 的对角线是 R 的对角线（始终为实数），因此我们通过其符号对其进行归一化处理
    Q *= X.diagonal(dim1=-2, dim2=-1).sgn().unsqueeze(-2)
    return Q


class _OrthMaps(Enum):
    # 定义枚举类型，包含三种正交映射方法
    matrix_exp = auto()
    cayley = auto()
    householder = auto()


class _Orthogonal(Module):
    base: Tensor

    def __init__(
        self, weight, orthogonal_map: _OrthMaps, *, use_trivialization=True
    ) -> None:
        super().__init__()

        # Note [Householder complex]
        # 对于复数张量，无法从反射器计算 linalg.householder_product 所需的张量 tau。
        # 因为反射器的形状类似于：
        # 0 0 0
        # * 0 0
        # * * 0
        # 对于复杂矩阵，这会产生 n(n-1)（实数）参数。现在，你需要 n^2 个参数来参数化酉矩阵。
        # 单独保存 tau 也行不通，因为不是每种“(A, tau)”组合都会生成酉矩阵，这意味着如果我们将它们作为独立张量优化，
        # 就无法保持约束条件。对于矩形矩阵，类似的推理适用。
        if weight.is_complex() and orthogonal_map == _OrthMaps.householder:
            raise ValueError(
                "The householder parametrization does not support complex tensors."
            )

        self.shape = weight.shape
        self.orthogonal_map = orthogonal_map
        if use_trivialization:
            self.register_buffer("base", None)
    # 定义前向传播方法，接收一个张量 X 作为输入并返回一个张量作为输出
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # 获取输入张量 X 的维度大小
        n, k = X.size(-2), X.size(-1)
        # 判断是否需要转置输入张量 X
        transposed = n < k
        if transposed:
            # 如果需要转置，将输入张量 X 转置
            X = X.mT
            # 更新 n 和 k 的值
            n, k = k, n
        # 当 n > k 时，输入张量 X 是一个长方形矩阵

        # 根据选择的正交映射方法进行不同的处理
        if (
            self.orthogonal_map == _OrthMaps.matrix_exp
            or self.orthogonal_map == _OrthMaps.cayley
        ):
            # 如果选择的是矩阵指数映射或者卡莱映射
            # 取输入张量 X 的下三角部分
            X = X.tril()
            if n != k:
                # 如果 n 不等于 k，将 X 嵌入到一个方阵中
                X = torch.cat(
                    [X, X.new_zeros(n, n - k).expand(*X.shape[:-2], -1, -1)], dim=-1
                )
            # 计算 A = X - X^H，其中 A 是反对称的（或反-Hermitian的）
            A = X - X.mH

            # 根据选择的正交映射方法计算 Q
            if self.orthogonal_map == _OrthMaps.matrix_exp:
                # 使用矩阵指数计算 Q
                Q = torch.matrix_exp(A)
            elif self.orthogonal_map == _OrthMaps.cayley:
                # 计算卡莱映射的结果 (I+A/2)(I-A/2)^{-1}
                Id = torch.eye(n, dtype=A.dtype, device=A.device)
                Q = torch.linalg.solve(
                    torch.add(Id, A, alpha=-0.5), torch.add(Id, A, alpha=0.5)
                )
            # 现在 Q 是尺寸为 (..., n, n) 的正交矩阵（或酉矩阵）

            if n != k:
                # 如果 n 不等于 k，截取 Q 的前 k 列
                Q = Q[..., :k]
            # 现在 Q 的大小与 X 相同（可能转置）
        else:
            # 如果不支持复数的情况下，X 是实数
            # 取输入张量 X 的下三角部分（副对角线以下）
            A = X.tril(diagonal=-1)
            # 计算 Householder 变换所需的参数 tau
            tau = 2.0 / (1.0 + (A * A).sum(dim=-2))
            # 使用 Householder 乘积算法计算 Q
            Q = torch.linalg.householder_product(A, tau)
            # X 的对角线元素为 1 或 -1
            # 我们不希望通过这些元素进行微分或更新，因此进行类型转换处理
            Q = Q * X.diagonal(dim1=-2, dim2=-1).int().unsqueeze(-2)

        # 如果模型有 "base" 属性，将 Q 与 base 矩阵相乘
        if hasattr(self, "base"):
            Q = self.base @ Q
        # 如果之前进行了转置，将 Q 还原为原始的转置状态
        if transposed:
            Q = Q.mT
        # 返回 Q 作为输出结果，这里可能会有未定义类型的警告
        return Q  # type: ignore[possibly-undefined]

    @torch.autograd.no_grad()
# 定义一个函数 orthogonal，用于给模型中的权重矩阵应用正交或酉变换的参数化
def orthogonal(
    module: Module,
    name: str = "weight",
    orthogonal_map: Optional[str] = None,
    *,
    use_trivialization: bool = True,
) -> Module:
    r"""Apply an orthogonal or unitary parametrization to a matrix or a batch of matrices.

    Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`, the parametrized
    matrix :math:`Q \in \mathbb{K}^{m \times n}` is **orthogonal** as

    .. math::

        \begin{align*}
            Q^{\text{H}}Q &= \mathrm{I}_n \mathrlap{\qquad \text{if }m \geq n}\\
            QQ^{\text{H}} &= \mathrm{I}_m \mathrlap{\qquad \text{if }m < n}
        \end{align*}

    where :math:`Q^{\text{H}}` is the conjugate transpose when :math:`Q` is complex
    and the transpose when :math:`Q` is real-valued, and
    :math:`\mathrm{I}_n` is the `n`-dimensional identity matrix.
    In plain words, :math:`Q` will have orthonormal columns whenever :math:`m \geq n`
    and orthonormal rows otherwise.

    If the tensor has more than two dimensions, we consider it as a batch of matrices of shape `(..., m, n)`.

    The matrix :math:`Q` may be parametrized via three different ``orthogonal_map`` in terms of the original tensor:

    - ``"matrix_exp"``/``"cayley"``:
      the :func:`~torch.matrix_exp` :math:`Q = \exp(A)` and the `Cayley map`_
      :math:`Q = (\mathrm{I}_n + A/2)(\mathrm{I}_n - A/2)^{-1}` are applied to a skew-symmetric
      :math:`A` to give an orthogonal matrix.
    - ``"householder"``: computes a product of Householder reflectors
      (:func:`~torch.linalg.householder_product`).

    ``"matrix_exp"``/``"cayley"`` often make the parametrized weight converge faster than
    ``"householder"``, but they are slower to compute for very thin or very wide matrices.

    If ``use_trivialization=True`` (default), the parametrization implements the "Dynamic Trivialization Framework",
    where an extra matrix :math:`B \in \mathbb{K}^{n \times n}` is stored under
    ``module.parametrizations.weight[0].base``. This helps the
    convergence of the parametrized layer at the expense of some extra memory use.
    See `Trivializations for Gradient-Based Optimization on Manifolds`_ .

    Initial value of :math:`Q`:
    If the original tensor is not parametrized and ``use_trivialization=True`` (default), the initial value
    of :math:`Q` is that of the original tensor if it is orthogonal (or unitary in the complex case)
    and it is orthogonalized via the QR decomposition otherwise (see :func:`torch.linalg.qr`).
    Same happens when it is not parametrized and ``orthogonal_map="householder"`` even when ``use_trivialization=False``.
    Otherwise, the initial value is the result of the composition of all the registered
    parametrizations applied to the original tensor.

    .. note::
        This function is implemented using the parametrization functionality
        in :func:`~torch.nn.utils.parametrize.register_parametrization`.
    weight = getattr(module, name, None)
    # 获取模块 `module` 中名称为 `name` 的参数或缓冲区，如果不存在则为 None

    if not isinstance(weight, Tensor):
        # 如果获取的参数不是 Tensor 类型，则抛出数值错误异常
        raise ValueError(
            f"Module '{module}' has no parameter or buffer with name '{name}'"
        )

    # 如果参数的维度小于 2，则抛出数值错误异常
    if weight.ndim < 2:
        raise ValueError(
            "Expected a matrix or batch of matrices. "
            f"Got a tensor of {weight.ndim} dimensions."
        )

    # 如果未指定正交映射方法，则根据权重张量的形状和是否为复数选择默认的方法
    if orthogonal_map is None:
        orthogonal_map = (
            "matrix_exp"
            if weight.size(-2) == weight.size(-1) or weight.is_complex()
            else "householder"
        )

    # 获取正交映射的枚举对象，如果找不到则抛出数值错误异常
    orth_enum = getattr(_OrthMaps, orthogonal_map, None)
    if orth_enum is None:
        raise ValueError(
            'orthogonal_map has to be one of "matrix_exp", "cayley", "householder". '
            f"Got: {orthogonal_map}"
        )

    # 创建 _Orthogonal 类的实例，用于参数化权重张量
    orth = _Orthogonal(weight, orth_enum, use_trivialization=use_trivialization)
    # 将参数化方法注册到指定的模块和名称上，unsafe=True 表示允许不安全的操作
    parametrize.register_parametrization(module, name, orth, unsafe=True)
    # 返回已注册了正交参数化的原始模块
    return module
class _WeightNorm(Module):
    # 定义带权重归一化的模块类
    def __init__(
        self,
        dim: Optional[int] = 0,
    ) -> None:
        # 初始化函数，设置归一化的维度，默认为0
        super().__init__()
        # 调用父类初始化方法
        if dim is None:
            dim = -1
        # 如果维度为None，则设置为-1
        self.dim = dim
        # 将维度保存在实例变量中

    def forward(self, weight_g, weight_v):
        # 前向传播方法，应用权重归一化
        return torch._weight_norm(weight_v, weight_g, self.dim)
        # 调用PyTorch内部的权重归一化函数，并返回结果

    def right_inverse(self, weight):
        # 计算权重的右逆
        weight_g = torch.norm_except_dim(weight, 2, self.dim)
        # 计算除了指定维度外的2范数
        weight_v = weight
        # 直接将权重作为权重向量返回

        return weight_g, weight_v
        # 返回计算出的权重归一化因子和权重向量


def weight_norm(module: Module, name: str = "weight", dim: int = 0):
    r"""Apply weight normalization to a parameter in the given module.

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by :attr:`name` with two parameters: one specifying the magnitude
    and one specifying the direction.

    By default, with ``dim=0``, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    ``dim=None``.

    See https://arxiv.org/abs/1602.07868

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm

    Returns:
        The original module with the weight norm hook

    Example::

        >>> m = weight_norm(nn.Linear(20, 40), name='weight')
        >>> m
        ParametrizedLinear(
          in_features=20, out_features=40, bias=True
          (parametrizations): ModuleDict(
            (weight): ParametrizationList(
              (0): _WeightNorm()
            )
          )
        )
        >>> m.parametrizations.weight.original0.size()
        torch.Size([40, 1])
        >>> m.parametrizations.weight.original1.size()
        torch.Size([40, 20])

    """
    # 给指定模块的参数应用权重归一化

    _weight_norm = _WeightNorm(dim)
    # 创建一个带有指定维度的权重归一化对象
    parametrize.register_parametrization(module, name, _weight_norm, unsafe=True)
    # 在模块中注册参数化方法，使用不安全模式

    def _weight_norm_compat_hook(
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # 兼容性钩子函数，用于加载状态字典时的处理
        g_key = f"{prefix}{name}_g"
        # 计算归一化因子在状态字典中的键
        v_key = f"{prefix}{name}_v"
        # 计算归一化向量在状态字典中的键
        if g_key in state_dict and v_key in state_dict:
            # 如果键存在于状态字典中
            original0 = state_dict.pop(g_key)
            # 弹出并获取归一化因子的值
            original1 = state_dict.pop(v_key)
            # 弹出并获取归一化向量的值
            state_dict[f"{prefix}parametrizations.{name}.original0"] = original0
            # 将归一化因子的值存入新的状态字典键中
            state_dict[f"{prefix}parametrizations.{name}.original1"] = original1
            # 将归一化向量的值存入新的状态字典键中

    module._register_load_state_dict_pre_hook(_weight_norm_compat_hook)
    # 在模块上注册加载状态字典前的钩子函数
    return module
    # 返回应用了权重归一化的模块类


class _SpectralNorm(Module):
    # 定义带谱范数的模块类
    def __init__(
        self,
        weight: torch.Tensor,
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
    ):
        # 初始化函数，设置谱范数计算所需的参数
    ) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 获取权重张量的维度数
        ndim = weight.ndim
        # 检查给定的维度是否超出范围
        if dim >= ndim or dim < -ndim:
            raise IndexError(
                "Dimension out of range (expected to be in range of "
                f"[-{ndim}, {ndim - 1}] but got {dim})"
            )

        # 检查迭代次数是否合法
        if n_power_iterations <= 0:
            raise ValueError(
                "Expected n_power_iterations to be positive, but "
                f"got n_power_iterations={n_power_iterations}"
            )
        # 根据传入的维度参数确定有效维度
        self.dim = dim if dim >= 0 else dim + ndim
        self.eps = eps
        # 如果权重张量的维度大于1，则需要进行谱归一化的估计
        if ndim > 1:
            # 对于一维情况，不需要进行估计（参见 _SpectralNorm.forward 方法）
            self.n_power_iterations = n_power_iterations
            # 将权重张量重塑为矩阵
            weight_mat = self._reshape_weight_to_matrix(weight)
            h, w = weight_mat.size()

            # 初始化随机向量 u 和 v
            u = weight_mat.new_empty(h).normal_(0, 1)
            v = weight_mat.new_empty(w).normal_(0, 1)
            # 对 u 和 v 进行归一化并注册为缓冲区
            self.register_buffer("_u", F.normalize(u, dim=0, eps=self.eps))
            self.register_buffer("_v", F.normalize(v, dim=0, eps=self.eps))

            # 使用幂方法对 u, v 进行初始化
            self._power_method(weight_mat, 15)

    def _reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        # 先决条件：确保权重张量的维度大于1
        assert weight.ndim > 1

        if self.dim != 0:
            # 如果指定的维度不是第一维度，则将其置换到第一维度
            weight = weight.permute(
                self.dim, *(d for d in range(weight.dim()) if d != self.dim)
            )

        # 将权重张量展平为二维矩阵
        return weight.flatten(1)

    @torch.autograd.no_grad()
    def _power_method(self, weight_mat: torch.Tensor, n_power_iterations: int) -> None:
        # 在 torch/nn/utils/spectral_norm.py 中查看原始注释
        # 注意：如果设置了 `do_power_iteration`，则 `u` 和 `v` 向量会在幂迭代过程中**原地**更新。这非常重要，
        # 因为在 `DataParallel` 的前向传播中，这些向量（作为缓冲区）会从并行化模块广播到每个模块副本，
        # 每个副本都会执行自己的谱范数幂迭代。因此，简单地将更新后的向量分配给该函数所在的模块会导致更新永久丢失。
        # 下次并行模块复制时，相同的随机初始化向量会被广播并使用！

        # 因此，为了使变化传播回来，我们依赖于两个重要行为（也通过测试强制执行）：
        #   1. `DataParallel` 如果广播张量已经在正确的设备上，则不会克隆存储；并确保并行模块已经在 `device[0]` 上。
        #   2. 如果 `out=` 关键字参数的输出张量具有正确的形状，则只会填充值。
        # 因此，由于在所有设备上执行相同的幂迭代，简单地原地更新张量将确保在 `device[0]` 上的模块副本将更新 _u 向量
        # （通过共享存储的方式）。

        # 然而，在原地更新 `u` 和 `v` 后，我们需要在使用它们进行权重归一化之前**克隆**它们。这是为了支持两次前向传播
        # 的反向传播，例如 GAN 训练中常见的模式：loss = D(real) - D(fake)。否则，引擎将抱怨需要在第一次前向传播
        # （即 `u` 和 `v` 向量）中进行反向传播时，第二次前向传播中这些变量已经发生了变化。

        # 前提条件
        assert weight_mat.ndim > 1

        for _ in range(n_power_iterations):
            # 权重的谱范数等于 `u^T W v`，其中 `u` 和 `v` 是左奇异向量和右奇异向量。
            # 这个幂迭代生成 `u` 和 `v` 的近似值。
            self._u = F.normalize(
                torch.mv(weight_mat, self._v),  # type: ignore[has-type]
                dim=0,
                eps=self.eps,
                out=self._u,  # type: ignore[has-type]
            )
            self._v = F.normalize(
                torch.mv(weight_mat.H, self._u),  # type: ignore[has-type]
                dim=0,
                eps=self.eps,
                out=self._v,  # type: ignore[has-type]
            )
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim == 1:
            # 如果权重张量是一维的，使用更快速和更准确的路径，无需近似处理
            return F.normalize(weight, dim=0, eps=self.eps)
        else:
            # 将权重张量重塑为矩阵
            weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                # 如果处于训练模式，执行幂方法来更新内部状态
                self._power_method(weight_mat, self.n_power_iterations)
            # 根据之前的理由，我们需要克隆这些张量
            u = self._u.clone(memory_format=torch.contiguous_format)
            v = self._v.clone(memory_format=torch.contiguous_format)
            # 通过 F.bilinear 计算这个值的正确方式应该是这样，但似乎存在效率问题：
            # https://github.com/pytorch/pytorch/issues/58093
            # 计算 sigma = u^T * (weight_mat * v)
            sigma = torch.vdot(u, torch.mv(weight_mat, v))
            # 返回权重除以 sigma 的结果
            return weight / sigma

    def right_inverse(self, value: torch.Tensor) -> torch.Tensor:
        # 可能希望在这里断言传递的值已经满足约束条件
        # 直接返回传入的值作为右逆的计算结果
        return value
# 定义 spectral_norm 函数，用于对给定模块中的参数应用谱归一化
def spectral_norm(
    module: Module,
    name: str = "weight",
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    dim: Optional[int] = None,
) -> Module:
    r"""Apply spectral normalization to a parameter in the given module.

    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    When applied on a vector, it simplifies to

    .. math::
        \mathbf{x}_{SN} = \dfrac{\mathbf{x}}{\|\mathbf{x}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by reducing the Lipschitz constant
    of the model. :math:`\sigma` is approximated performing one iteration of the
    `power method`_ every time the weight is accessed. If the dimension of the
    weight tensor is greater than 2, it is reshaped to 2D in power iteration
    method to get spectral norm.


    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`power method`: https://en.wikipedia.org/wiki/Power_iteration
    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    .. note::
        This function is implemented using the parametrization functionality
        in :func:`~torch.nn.utils.parametrize.register_parametrization`. It is a
        reimplementation of :func:`torch.nn.utils.spectral_norm`.

    .. note::
        When this constraint is registered, the singular vectors associated to the largest
        singular value are estimated rather than sampled at random. These are then updated
        performing :attr:`n_power_iterations` of the `power method`_ whenever the tensor
        is accessed with the module on `training` mode.

    .. note::
        If the `_SpectralNorm` module, i.e., `module.parametrization.weight[idx]`,
        is in training mode on removal, it will perform another power iteration.
        If you'd like to avoid this iteration, set the module to eval mode
        before its removal.

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter. Default: ``"weight"``.
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm. Default: ``1``.
        eps (float, optional): epsilon for numerical stability in
            calculating norms. Default: ``1e-12``.
        dim (int, optional): dimension corresponding to number of outputs.
            Default: ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with a new parametrization registered to the specified
        weight
    """
    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> snm = spectral_norm(nn.Linear(20, 40))
        >>> snm
        ParametrizedLinear(
          in_features=20, out_features=40, bias=True
          (parametrizations): ModuleDict(
            (weight): ParametrizationList(
              (0): _SpectralNorm()
            )
          )
        )
        >>> torch.linalg.matrix_norm(snm.weight, 2)
        tensor(1.0081, grad_fn=<AmaxBackward0>)
    """
    # 从模块中获取指定名称的参数或缓存
    weight = getattr(module, name, None)
    # 如果获取的对象不是张量，则抛出数值错误
    if not isinstance(weight, Tensor):
        raise ValueError(
            f"Module '{module}' has no parameter or buffer with name '{name}'"
        )

    # 如果未指定维度，则根据模块的类型设置默认维度
    if dim is None:
        if isinstance(
            module,
            (
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            ),
        ):
            dim = 1
        else:
            dim = 0

    # 注册参数化操作，使用 _SpectralNorm 进行参数规范化
    parametrize.register_parametrization(
        module, name, _SpectralNorm(weight, n_power_iterations, dim, eps)
    )
    # 返回已修改的模块对象
    return module
```