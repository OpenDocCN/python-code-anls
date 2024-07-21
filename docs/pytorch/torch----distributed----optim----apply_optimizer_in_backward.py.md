# `.\pytorch\torch\distributed\optim\apply_optimizer_in_backward.py`

```
# 引入必要的模块和类型定义
from typing import Any, Dict, Iterable, List, no_type_check, Type

import torch

# 声明一个空列表，用于存储此模块中公开的所有符号名称
__all__: List[str] = []

# 使用 WeakTensorKeyDictionary 类型创建两个全局变量，用于存储张量/参数的元数据
# 这些字典会根据张量/参数的生命周期自动管理，避免影响其生命周期
param_to_optim_hook_handle_map = torch.utils.weak.WeakTensorKeyDictionary()
param_to_acc_grad_map = torch.utils.weak.WeakTensorKeyDictionary()

# 使用 no_type_check 装饰器声明一个函数，用于在反向传播过程中应用优化器
@no_type_check
def _apply_optimizer_in_backward(
    optimizer_class: Type[torch.optim.Optimizer],
    params: Iterable[torch.nn.Parameter],
    optimizer_kwargs: Dict[str, Any],
    register_hook: bool = True,
) -> None:
    """
    在调用 backward() 后，指定的优化器将在梯度累积到参数后触发。

    注意：这些参数的梯度在 backward() 后将被设置为 None。
    这意味着，如果未通过 _apply_optimizer_in_backward 指定任何其他优化器，则此参数上的任何其他优化器将无效。

    Args:
        optimizer_class: (Type[torch.optim.Optimizer]): 应用于参数的优化器类
        params: (Iterator[nn.Parameter]): 要应用优化器状态的参数
        optimizer_kwargs: (Dict[str, Any]): 传递给优化器构造函数的关键字参数
        register_hook: (bool): 是否注册一个钩子，在累积此参数的梯度后运行优化器。
            这是 backward 中默认实现优化器的方式，但特定用例（如 DDP）可能希望覆盖此以实现自定义行为。
            (默认为 True)

    Example::
        params_generator = model.parameters()
        param_1 = next(params_generator)
        remainder_params = list(params_generator)

        apply_optimizer_in_backward(torch.optim.SGD, [param_1], {"lr": .02})
        apply_optimizer_in_backward(torch.optim.Adam, remainder_params, {"lr": .04})

        model(...).sum().backward() # 在 backward 后，参数将已应用其注册的优化器。

    """
    # 记录一次 API 使用情况，这里是为了跟踪 torch.distributed.optim.apply_optimizer_in_backward 的调用次数
    torch._C._log_api_usage_once("torch.distributed.optim.apply_optimizer_in_backward")

    @no_type_check
    def _apply_optimizer_in_backward_to_param(param: torch.nn.Parameter) -> None:
        # view_as 创建一个节点在自动求导图中，使我们能够访问参数的 AccumulateGrad 自动求导函数对象。
        # 我们在这个对象上注册一个钩子，以便在该参数的梯度准备好时（累积到 .grad 字段中）触发优化器。

        # 如果 param_to_acc_grad_map 中还没有这个参数的累积梯度，就创建一个
        # 这适用于共享参数或将多个优化器附加到一个参数上的情况。
        if param not in param_to_acc_grad_map:
            param_to_acc_grad_map[param] = param.view_as(param).grad_fn.next_functions[
                0
            ][0]

        # 创建一个优化器对象，以 param 作为优化目标，使用 optimizer_kwargs 作为参数
        optimizer = optimizer_class([param], **optimizer_kwargs)

        # 如果 param 没有 _in_backward_optimizers 属性，就初始化它为一个空列表
        if not hasattr(param, "_in_backward_optimizers"):
            param._in_backward_optimizers = []  # type: ignore[attr-defined]
            # TODO: 一旦我们有更好的方法访问参数的优化器类和 kwargs，就可以移除这些属性。
            param._optimizer_classes = []  # type: ignore[attr-defined]
            param._optimizer_kwargs = []  # type: ignore[attr-defined]

        # 将当前的优化器对象添加到 param._in_backward_optimizers 列表中
        param._in_backward_optimizers.append(optimizer)  # type: ignore[attr-defined]
        param._optimizer_classes.append(optimizer_class)  # type: ignore[attr-defined]
        param._optimizer_kwargs.append(optimizer_kwargs)  # type: ignore[attr-defined]

        # 如果不需要注册钩子，直接返回
        if not register_hook:
            return

        # 定义一个 optimizer_hook 函数作为钩子的回调函数
        def optimizer_hook(*_unused) -> None:
            # 遍历 param._in_backward_optimizers 中的所有优化器对象，逐个执行 step() 方法
            for opt in param._in_backward_optimizers:  # type: ignore[attr-defined]
                opt.step()

            # 将 param.grad 设置为 None，清空梯度
            param.grad = None

        # 注册 optimizer_hook 函数到 param 的累积梯度的钩子中
        handle = param_to_acc_grad_map[param].register_hook(optimizer_hook)  # type: ignore[attr-defined]
        # 如果 param 在 param_to_optim_hook_handle_map 中不存在，则添加一个空列表
        if param not in param_to_optim_hook_handle_map:
            param_to_optim_hook_handle_map[param] = []
        # 将 handle 添加到 param_to_optim_hook_handle_map[param] 列表中
        param_to_optim_hook_handle_map[param].append(handle)

    # 遍历 params 列表，对每个参数应用 _apply_optimizer_in_backward_to_param 函数
    for param in params:
        _apply_optimizer_in_backward_to_param(param)
# 返回应用于模块参数的反向优化器列表。这些优化器通常不直接由用户调用其“step”或“zero_grad”方法，而是用于诸如检查点等用途。
def _get_in_backward_optimizers(module: torch.nn.Module) -> List[torch.optim.Optimizer]:
    # 初始化一个空列表，用于存储模块参数的反向优化器
    optims: List[torch.optim.Optimizer] = []
    # 遍历模块的所有参数
    for param in module.parameters():
        # 获取参数的"_in_backward_optimizers"属性，并将其添加到optims列表中
        optims.extend(getattr(param, "_in_backward_optimizers", []))

    # 返回包含反向优化器的列表
    return optims
```