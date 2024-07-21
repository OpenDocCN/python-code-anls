# `.\pytorch\torch\autograd\anomaly_mode.py`

```
# 设置 mypy 来允许未类型化的定义
r"""Autograd anomaly mode."""  # 模块的文档字符串，描述此模块的用途

import warnings  # 导入警告模块

import torch  # 导入 PyTorch 库

__all__ = ["detect_anomaly", "set_detect_anomaly"]  # 定义模块的公开接口

# 定义一个上下文管理器 detect_anomaly，用于启用自动求导引擎的异常检测
class detect_anomaly:
    r"""Context-manager that enable anomaly detection for the autograd engine.

    This does two things:

    - Running the forward pass with detection enabled will allow the backward
      pass to print the traceback of the forward operation that created the failing
      backward function.
    - If ``check_nan`` is ``True``, any backward computation that generate "nan"
      value will raise an error. Default ``True``.

    .. warning::
        This mode should be enabled only for debugging as the different tests
        will slow down your program execution.
    """
    # 定义一个类，用于管理异常检测，继承自torch.autograd.Function
    class MyFunc(autograd.Function):

        # 静态方法：前向传播函数，接收上下文对象 ctx 和输入 inp
        @staticmethod
        def forward(ctx, inp):
            # 返回输入的克隆（深拷贝）
            return inp.clone()

        # 静态方法：反向传播函数，接收上下文对象 ctx 和梯度 gO
        @staticmethod
        def backward(ctx, gO):
            # 在反向传播过程中引发运行时异常
            raise RuntimeError("Some error in backward")
            # 返回梯度的克隆（深拷贝）
            return gO.clone()

    # 函数：运行自定义的 MyFunc 类，计算输入张量 a 的和
    def run_fn(a):
        # 使用 MyFunc 的 forward 方法进行前向传播
        out = MyFunc.apply(a)
        # 返回计算结果的和
        return out.sum()

    # 创建一个新的张量，形状为 10x10，要求计算梯度
    inp = torch.rand(10, 10, requires_grad=True)
    # 调用 run_fn 函数进行计算
    out = run_fn(inp)
    # 对计算结果进行反向传播
    out.backward()

    # 构造函数：初始化异常检测类，可以检查 NaN 值，默认为开启状态
    def __init__(self, check_nan=True) -> None:
        # 保存当前的异常检测状态
        self.prev = torch.is_anomaly_enabled()
        # 设置是否检查 NaN 值的标志
        self.check_nan = check_nan
        # 保存之前的 NaN 检查状态
        self.prev_check_nan = torch.is_anomaly_check_nan_enabled()
        # 发出警告，说明已启用异常检测模式，这会增加运行时长，仅用于调试目的
        warnings.warn(
            "Anomaly Detection has been enabled. "
            "This mode will increase the runtime "
            "and should only be enabled for debugging.",
            stacklevel=2,
        )
    def __enter__(self) -> None:  # noqa: D105
        # 进入上下文时设置 PyTorch 异常检测，并根据 self.check_nan 决定是否检测 NaN
        torch.set_anomaly_enabled(True, self.check_nan)

    def __exit__(self, *args: object) -> None:  # noqa: D105
        # 退出上下文时恢复 PyTorch 异常检测设置，回到之前的状态，并恢复 self.prev_check_nan 设置
        torch.set_anomaly_enabled(self.prev, self.prev_check_nan)
class set_detect_anomaly:
    r"""Context-manager that sets the anomaly detection for the autograd engine on or off.

    ``set_detect_anomaly`` will enable or disable the autograd anomaly detection
    based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    See ``detect_anomaly`` above for details of the anomaly detection behaviour.

    Args:
        mode (bool): Flag whether to enable anomaly detection (``True``),
                     or disable (``False``).
        check_nan (bool): Flag whether to raise an error when the backward
                          generate "nan"

    """

    def __init__(self, mode: bool, check_nan: bool = True) -> None:  # noqa: D107
        # 记录当前的异常检测状态和是否检查 NaN
        self.prev = torch.is_anomaly_enabled()
        self.prev_check_nan = torch.is_anomaly_check_nan_enabled()
        # 设置新的异常检测状态和是否检查 NaN
        torch.set_anomaly_enabled(mode, check_nan)

    def __enter__(self) -> None:  # noqa: D105
        # 在进入上下文时不执行任何操作
        pass

    def __exit__(self, *args: object) -> None:  # noqa: D105
        # 在退出上下文时恢复之前的异常检测状态和是否检查 NaN 的设置
        torch.set_anomaly_enabled(self.prev, self.prev_check_nan)
```