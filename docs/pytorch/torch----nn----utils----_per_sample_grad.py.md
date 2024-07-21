# `.\pytorch\torch\nn\utils\_per_sample_grad.py`

```
# 引入允许未类型化定义的类型检查设置
# functools 是 Python 标准库中的工具模块，提供了有用的功能
import functools

# 引入 PyTorch 库
import torch

# 从 torch.nn.utils._expanded_weights.expanded_weights_impl 模块中导入 ExpandedWeight 类
from torch.nn.utils._expanded_weights.expanded_weights_impl import ExpandedWeight

# 从 torch.utils 模块中导入 _pytree 模块
from torch.utils import _pytree as pytree

# 由于依赖于 `functional_call`，这个函数不能暴露在 utils 中以避免循环依赖
def call_for_per_sample_grads(
    module,
    *,
    batch_size=None,
    loss_reduction="sum",
    batch_first=True,
):
    r"""
    返回一个模块的前向函数，使得在调用反向传播时可以填充 grad_sample 以获取每个样本的梯度。

    Args:
        module: 需要获取每个样本梯度的 `nn.Module`。所有可训练参数将在调用 `backward` 时计算每个样本的梯度，
          并存储在 `grad_sample` 字段中。
        batch_size: 输入的批量大小。如果传入 `None`，则 `args` 和 `kwargs` 中的所有张量参数必须具有相同的批量大小，
          即第一维的大小。否则，必须手动传入批量大小。默认为 `None`。
        loss_reduction: 表示用于聚合梯度的损失缩减操作是求和还是平均。如果是 "mean"，每个样本的梯度将通过批量大小进行缩放，
          以抵消跨批量交互造成的影响。必须是 "mean" 或 "sum"。默认为 "sum"。
        batch_first: 表示批量维度是否为第一维。如果为 True，则批量维度是第一维；如果为 False，则为第二维。默认为 True。

    Examples::
        >>> # xdoctest: +SKIP
        >>> model = nn.Linear(4, 3)
        >>> batched_input = torch.randn(5, 4)  # 批量大小为 5
        >>> res = call_for_per_sample_grads(model)(batched_input).sum()
        >>> res.backward()
        >>> assert model.weight.shape == (3, 4)
        >>> assert model.weight.grad_sample.shape == (5, 3, 4)
        >>> assert model.weight.grad is None
        >>> assert model.bias.shape == (3,)
        >>> assert model.bias.grad_sample.shape == (5, 3)
        >>> assert model.bias.grad is None

    使用 "mean" 损失缩减的示例。grad_sample 字段将通过 batch_size 进行缩放，与设置 loss_reduction="sum" 下的效果相同。
    这是因为最终的平均操作会通过 1 / batch_size 缩放所有 grad_outputs，以抵消跨批量交互。
        >>> model = nn.Linear(4, 3)
        >>> batched_input = torch.randn(5, 4)  # 批量大小为 5
        >>> res = call_for_per_sample_grads(model, 5, loss_reduction="mean")(batched_input).mean()
        >>> res.backward()

    注意::
        不适用于任何 `nn.RNN`，包括 `nn.GRU` 或 `nn.LSTM`。请使用自定义重写，包装一个 `nn.Linear` 模块。参见 Opacus 的示例。
    """
    # 如果原始张量需要梯度，则使用 ExpandedWeight 类构建一个新的对象，否则直接返回原始张量
    def maybe_build_expanded_weight(og_tensor, batch_size):
        if og_tensor.requires_grad:
            return ExpandedWeight(og_tensor, batch_size, loss_reduction)
        else:
            return og_tensor

    # 计算输入参数中的批量大小
    def compute_batch_size(*args, **kwargs):
        # 将所有参数和关键字参数展开为一个列表
        args_and_kwargs = pytree.arg_tree_leaves(*args, **kwargs)
        batch_size = None
        for arg in args_and_kwargs:
            if not isinstance(arg, torch.Tensor):  # 如果参数不是张量则跳过
                continue

            # 根据参数是否以 batch_first 方式排列获取其批量大小
            arg_batch_size = arg.shape[0] if batch_first else arg.shape[1]
            # 如果之前已经确定了批量大小并且当前参数的批量大小不一致，则抛出错误
            if batch_size is not None and batch_size != arg_batch_size:
                raise RuntimeError(
                    "When computing batch size, found at least one input with batch size "
                    f"{batch_size} and one with batch size {arg_batch_size}. Please specify it "
                    "explicitly using the batch size kwarg in call_for_per_sample_grads"
                )
            batch_size = arg_batch_size
        # 如果未找到张量参数，则抛出错误
        if batch_size is None:
            raise RuntimeError(
                "Unable to find a tensor in the passed args and kwargs. They may not be pytree-able "
                "and so ExpandedWeights cannot compute the batch size from the inputs. Please specify "
                "it explicitly"
            )
        return batch_size

    # 检查损失函数的缩减方式是否为 sum 或 mean，否则抛出错误
    if loss_reduction not in ["sum", "mean"]:
        raise RuntimeError(
            f"Expected loss_reduction argument to be sum or mean, got {loss_reduction}"
        )

    # 检查传入的模块是否为 torch.nn.Module 的实例，否则抛出错误
    if not isinstance(module, torch.nn.Module):
        raise RuntimeError(
            f"Module passed must be nn.Module, got {type(module).__name__}"
        )

    # 检查传入的批量大小参数是否为 None 或整数类型，否则抛出错误
    if not (batch_size is None or isinstance(batch_size, int)):
        raise RuntimeError(
            f"Batch size passed must be None or an integer, got {type(batch_size).__name__}"
        )

    # 如果指定了批量大小且其值小于 1，则抛出错误
    if batch_size is not None and batch_size < 1:
        raise RuntimeError(f"Batch size must be positive, got {batch_size}")

    # 检查模块的每个参数是否存在 grad_sample 属性，并且其值不为 None，否则抛出错误
    for weight in module.parameters():
        if hasattr(weight, "grad_sample") and weight.grad_sample is not None:
            raise RuntimeError(
                "Current Expanded Weights accumulates the gradients, which will be incorrect for multiple "
                f"calls without clearing gradients. Please clear out the grad_sample parameter of {weight} or "
                "post an issue to pytorch/pytorch to prioritize correct behavior"
            )

    # 使用 functools.wraps 装饰器保留原始模块的属性和文档字符串
    @functools.wraps(module.forward)
    def wrapper(*args, **kwargs):
        # 初始化 wrapper_batch_size 为传入的 batch_size
        wrapper_batch_size = batch_size
        # 如果 wrapper_batch_size 为 None，则调用 compute_batch_size 计算批量大小
        if wrapper_batch_size is None:
            wrapper_batch_size = compute_batch_size(*args, **kwargs)

        # 为模块的每个命名参数构建 ExpandedWeight 对象或使用原始张量
        params = {
            name: maybe_build_expanded_weight(value, wrapper_batch_size)
            for (name, value) in module.named_parameters()
        }
        # 调用 torch.func.functional_call 执行模块的前向传播
        return torch.func.functional_call(module, params, args, kwargs)

    # 返回装饰后的 wrapper 函数
    return wrapper
```