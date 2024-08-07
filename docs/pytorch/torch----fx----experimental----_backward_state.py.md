# `.\pytorch\torch\fx\experimental\_backward_state.py`

```py
# 导入 torch.fx 模块，用于处理 PyTorch 中的特定功能
import torch.fx

# 定义 BackwardState 类
class BackwardState:
    """
    BackwardState is used to pass Python hooks from the forwards pass
    into the backwards pass in Dynamo+Compiled Autograd.

    It is created by TorchDynamo and has special handling there.
    Dynamo will pass an empty BackwardState to the forwards, then populate
    members on it (via setattr) only after the forwards graph is finished.
    Later on, in CompileAutograd we will inline and add the needed guards
    on the BackwardState.

    BackwardState is identified and has special handling in AOTAutograd.
    During AOTAutograd:
        1) BackwardState is an input to the forwards graph
        2) It must only be used in the backwards
        3) It will be empty in the forwards
        4) In the forwards we add a wrapper to save it
        5) In the backwards it becomes an input
        6) There can only be one per graph

    BackwardState requires CompiledAutograd.
    """

    # 声明类属性 proxy，类型为 torch.fx.Proxy
    proxy: torch.fx.Proxy
```