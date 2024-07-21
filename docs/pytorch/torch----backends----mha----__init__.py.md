# `.\pytorch\torch\backends\mha\__init__.py`

```py
# 导入 torch 模块，用于配置是否启用 C++ 内核来加速 nn.functional.MHA 和 nn.TransformerEncoder
import torch

# 全局变量，用于标识是否启用快速路径（Fast Path）
_is_fastpath_enabled: bool = True


def get_fastpath_enabled() -> bool:
    """
    返回是否启用 TransformerEncoder 和 MultiHeadAttention 的快速路径，
    或者如果 jit 正在脚本化，则返回 ``True``。

    ..note:
        即使 ``get_fastpath_enabled`` 返回 ``True``，也可能不会运行快速路径，
        除非所有输入条件都满足。
    """
    # 检查当前是否处于 JIT 脚本化模式
    if not torch.jit.is_scripting():
        return _is_fastpath_enabled
    # 如果处于 JIT 脚本化模式，则默认返回 True
    return True


def set_fastpath_enabled(value: bool) -> None:
    """
    设置是否启用快速路径
    """
    # 使用 global 关键字声明全局变量 _is_fastpath_enabled
    global _is_fastpath_enabled
    _is_fastpath_enabled = value
```