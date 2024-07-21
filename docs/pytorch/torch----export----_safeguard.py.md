# `.\pytorch\torch\export\_safeguard.py`

```py
# mypy: allow-untyped-defs
# 导入 torch 库
import torch
# 导入 ProxyTorchDispatchMode 类型，用于控制代理 torch 调度模式
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode
# 导入 TorchFunctionMode 类，用于重写 Torch 函数模式
from torch.overrides import TorchFunctionMode

# 定义 AutogradStateOpsFailSafeguard 类，继承 TorchFunctionMode 类
class AutogradStateOpsFailSafeguard(TorchFunctionMode):
    """
    在导出图表时检测梯度状态操作，并通过引发错误来中止流程，以避免意外行为。这些梯度模式操作可能包括：
    `torch.no_grad`
    `torch.enable_grad`
    `torch.set_grad_enabled`

    使用 predispatch 模式导出时免除检查。
    """

    # 重写 __torch_function__ 方法
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # 不支持的梯度模式操作列表
        unsupported_grad_mode_ops = [
            torch._C._set_grad_enabled,
        ]
        # 只有在追踪时才启用，通过确认 torch 调度模式是任何活动的 PROXY 来进行验证。这是为了允许追踪外的自动求导操作。
        current_state = torch._C.is_grad_enabled()
        # 如果 func 在不支持的梯度模式操作列表中
        if func in unsupported_grad_mode_ops:
            assert len(args) == 1
            changed_state = args[0]
            # 获取 torch 调度模式
            mode = torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY)
            # 意图是检查它是否不是预调度模式。在预调度模式中允许使用自动求导操作，例如 `torch.no_grad`
            if (
                mode
                and isinstance(mode, ProxyTorchDispatchMode)
                and not mode.pre_dispatch
                and changed_state != current_state
            ):
                raise RuntimeError(
                    f"Encountered autograd state manager op {func} trying to change global autograd state "
                    "while exporting. This is unsafe because we don't capture this op in torch.export "
                    "today, hence we can't reflect the user intention soundly. You can fix this by "
                    "adding a torch.no_grad() context around the export call."
                )
        # 调用原始函数
        return func(*args, **kwargs)
```