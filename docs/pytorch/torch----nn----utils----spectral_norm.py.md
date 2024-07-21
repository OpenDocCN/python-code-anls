# `.\pytorch\torch\nn\utils\spectral_norm.py`

```py
# mypy: allow-untyped-defs
"""Spectral Normalization from https://arxiv.org/abs/1802.05957."""
from typing import Any, Optional, TypeVar

import torch
import torch.nn.functional as F
from torch.nn.modules import Module


__all__ = [
    "SpectralNorm",
    "SpectralNormLoadStateDictPreHook",
    "SpectralNormStateDictHook",
    "spectral_norm",
    "remove_spectral_norm",
]

# 定义一个类，实现了谱归一化
class SpectralNorm:
    # Invariant before and after each forward call:
    #   u = F.normalize(W @ v)
    # NB: At initialization, this invariant is not enforced
    _version: int = 1
    # 在版本 1 中：
    #   将 `W` 设为非缓冲区（non-buffer），
    #   添加 `v` 作为缓冲区（buffer），
    #   使得 eval 模式下使用 `W = u @ W_orig @ v` 而非存储的 `W`。
    
    name: str  # 参数名称，默认为 "weight"
    dim: int  # 操作的维度，默认为 0
    n_power_iterations: int  # 幂迭代的次数
    eps: float  # 计算中的小数值偏移量，默认为 1e-12

    # 初始化方法
    def __init__(
        self,
        name: str = "weight",
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
    ) -> None:
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError(
                "Expected n_power_iterations to be positive, but "
                f"got n_power_iterations={n_power_iterations}"
            )
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    # 将权重张量重塑为矩阵
    def reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        weight_mat = weight
        if self.dim != 0:
            # 将指定维度置换到最前面
            weight_mat = weight_mat.permute(
                self.dim, *[d for d in range(weight_mat.dim()) if d != self.dim]
            )
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    # 移除谱归一化
    def remove(self, module: Module) -> None:
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + "_u")
        delattr(module, self.name + "_v")
        delattr(module, self.name + "_orig")
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    # 在调用模块时触发，计算并设置谱归一化后的权重
    def __call__(self, module: Module, inputs: Any) -> None:
        setattr(
            module,
            self.name,
            self.compute_weight(module, do_power_iteration=module.training),
        )

    # 辅助方法：解出 v 并重新缩放
    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        # Tries to returns a vector `v` s.t. `u = F.normalize(W @ v)`
        # (the invariant at top of this class) and `u @ W @ v = sigma`.
        # This uses pinverse in case W^T W is not invertible.
        v = torch.linalg.multi_dot(
            [weight_mat.t().mm(weight_mat).pinverse(), weight_mat.t(), u.unsqueeze(1)]
        ).squeeze(1)
        return v.mul_(target_sigma / torch.dot(u, torch.mv(weight_mat, v)))

    # 静态方法：应用谱归一化到指定模块
    @staticmethod
    def apply(
        module: Module, name: str, n_power_iterations: int, dim: int, eps: float
    ) -> "SpectralNorm":
        # 遍历模块的前向预处理钩子
        for hook in module._forward_pre_hooks.values():
            # 如果钩子是 SpectralNorm 类型，并且钩子的名称与当前名称相同，则抛出运行时错误
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError(
                    f"Cannot register two spectral_norm hooks on the same parameter {name}"
                )

        # 创建 SpectralNorm 实例
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        # 获取模块参数中指定名称的权重
        weight = module._parameters[name]
        # 如果权重为 None，则抛出值错误
        if weight is None:
            raise ValueError(
                f"`SpectralNorm` cannot be applied as parameter `{name}` is None"
            )
        # 如果权重是未初始化的参数，则抛出值错误
        if isinstance(weight, torch.nn.parameter.UninitializedParameter):
            raise ValueError(
                "The module passed to `SpectralNorm` can't have uninitialized parameters. "
                "Make sure to run the dummy forward before applying spectral normalization"
            )

        # 使用 torch.no_grad() 上下文管理器，初始化权重矩阵和向量 `u` 和 `v`
        with torch.no_grad():
            # 将权重重塑为矩阵形式
            weight_mat = fn.reshape_weight_to_matrix(weight)

            # 获取矩阵的高度和宽度
            h, w = weight_mat.size()
            # 随机初始化 `u` 和 `v`
            u = F.normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = F.normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)

        # 删除模块上现有的与 `fn.name` 相关的属性
        delattr(module, fn.name)
        # 将原始权重注册为模块的参数
        module.register_parameter(fn.name + "_orig", weight)
        # 将权重数据注册为普通属性，因为直接赋值可能会导致将其误认为是 nn.Parameter
        setattr(module, fn.name, weight.data)
        # 将 `u` 和 `v` 注册为模块的缓冲区
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)

        # 注册前向预处理钩子
        module.register_forward_pre_hook(fn)
        # 注册状态字典钩子
        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        # 注册加载状态字典前预处理钩子
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        # 返回 SpectralNorm 实例
        return fn
# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
class SpectralNormLoadStateDictPreHook:
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    # SpectralNormLoadStateDictPreHook类的构造函数，初始化一个实例。
    def __init__(self, fn) -> None:
        self.fn = fn

    # For state_dict with version None, (assuming that it has gone through at
    # least one training forward), we have
    #
    #    u = F.normalize(W_orig @ v)
    #    W = W_orig / sigma, where sigma = u @ W_orig @ v
    #
    # To compute `v`, we solve `W_orig @ x = u`, and let
    #    v = x / (u @ W_orig @ x) * (W / W_orig).
    # SpectralNormLoadStateDictPreHook类的__call__方法，用于加载state_dict时调用。
    def __call__(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        fn = self.fn
        version = local_metadata.get("spectral_norm", {}).get(
            fn.name + ".version", None
        )
        if version is None or version < 1:
            weight_key = prefix + fn.name
            if (
                version is None
                and all(weight_key + s in state_dict for s in ("_orig", "_u", "_v"))
                and weight_key not in state_dict
            ):
                # Detect if it is the updated state dict and just missing metadata.
                # This could happen if the users are crafting a state dict themselves,
                # so we just pretend that this is the newest.
                return
            has_missing_keys = False
            for suffix in ("_orig", "", "_u"):
                key = weight_key + suffix
                if key not in state_dict:
                    has_missing_keys = True
                    if strict:
                        missing_keys.append(key)
            if has_missing_keys:
                return
            with torch.no_grad():
                weight_orig = state_dict[weight_key + "_orig"]
                weight = state_dict.pop(weight_key)
                sigma = (weight_orig / weight).mean()
                weight_mat = fn.reshape_weight_to_matrix(weight_orig)
                u = state_dict[weight_key + "_u"]
                v = fn._solve_v_and_rescale(weight_mat, u, sigma)
                state_dict[weight_key + "_v"] = v


# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
class SpectralNormStateDictHook:
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    # SpectralNormStateDictHook类的构造函数，初始化一个实例。
    def __init__(self, fn) -> None:
        self.fn = fn

    # SpectralNormStateDictHook类的__call__方法，用于处理module的state_dict。
    def __call__(self, module, state_dict, prefix, local_metadata) -> None:
        if "spectral_norm" not in local_metadata:
            local_metadata["spectral_norm"] = {}
        key = self.fn.name + ".version"
        if key in local_metadata["spectral_norm"]:
            raise RuntimeError(f"Unexpected key in metadata['spectral_norm']: {key}")
        local_metadata["spectral_norm"][key] = self.fn._version


# TypeVar用于声明一个泛型类型T_module，其绑定的类型是Module类或其子类。
T_module = TypeVar("T_module", bound=Module)
def spectral_norm(
    module: T_module,
    name: str = "weight",
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    dim: Optional[int] = None,
) -> T_module:
    r"""Apply spectral normalization to a parameter in the given module.

    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.

    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with the spectral norm hook

    .. note::
        This function has been reimplemented as
        :func:`torch.nn.utils.parametrizations.spectral_norm` using the new
        parametrization functionality in
        :func:`torch.nn.utils.parametrize.register_parametrization`. Please use
        the newer version. This function will be deprecated in a future version
        of PyTorch.

    Example::

        >>> m = spectral_norm(nn.Linear(20, 40))
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])

    """
    # Determine the dimension if not explicitly specified
    if dim is None:
        # For certain modules, set dim to 1; otherwise, default to 0
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
    # Apply spectral normalization using SpectralNorm class
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    # Return the modified module
    return module


def remove_spectral_norm(module: T_module, name: str = "weight") -> T_module:
    r"""Remove the spectral normalization reparameterization from a module.

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter

    """
    for k, hook in module._forward_pre_hooks.items():
        # 遍历 module 对象的 _forward_pre_hooks 字典，查找 SpectralNorm 类型的钩子并且钩子的名称为 name
        if isinstance(hook, SpectralNorm) and hook.name == name:
            # 如果找到匹配的钩子，则调用其 remove 方法移除钩子
            hook.remove(module)
            # 删除对应的钩子记录
            del module._forward_pre_hooks[k]
            # 结束循环
            break
    else:
        # 如果未找到匹配的钩子，抛出异常
        raise ValueError(f"spectral_norm of '{name}' not found in {module}")

    for k, hook in module._state_dict_hooks.items():
        # 遍历 module 对象的 _state_dict_hooks 字典，查找 SpectralNormStateDictHook 类型的钩子并且钩子的名称为 name
        if isinstance(hook, SpectralNormStateDictHook) and hook.fn.name == name:
            # 如果找到匹配的钩子，则删除对应的钩子记录
            del module._state_dict_hooks[k]
            # 结束循环
            break

    for k, hook in module._load_state_dict_pre_hooks.items():
        # 遍历 module 对象的 _load_state_dict_pre_hooks 字典，查找 SpectralNormLoadStateDictPreHook 类型的钩子并且钩子的名称为 name
        if isinstance(hook, SpectralNormLoadStateDictPreHook) and hook.fn.name == name:
            # 如果找到匹配的钩子，则删除对应的钩子记录
            del module._load_state_dict_pre_hooks[k]
            # 结束循环
            break

    # 返回经过处理的 module 对象
    return module
```