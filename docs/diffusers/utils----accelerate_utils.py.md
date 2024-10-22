# `.\diffusers\utils\accelerate_utils.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证，版本 2.0（“许可证”）授权；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，否则根据许可证分发的软件是以“原样”基础提供的，
# 不提供任何形式的明示或暗示的担保或条件。
# 请参阅许可证以了解特定语言的权限和
# 限制。
"""
加速工具：与加速相关的工具
"""

# 导入版本管理模块
from packaging import version

# 从导入工具模块中导入检测加速是否可用的函数
from .import_utils import is_accelerate_available


# 检查加速是否可用
if is_accelerate_available():
    # 如果可用，导入加速模块
    import accelerate


def apply_forward_hook(method):
    """
    装饰器，将注册的 CpuOffload 钩子应用于任意函数而非 `forward`。这对于 PyTorch 模块提供的其他函数（如 `encode` 和 `decode`）非常有用，这些函数应触发移动到适当的加速设备。
    此装饰器检查内部 `_hf_hook` 属性以查找注册的卸载钩子。

    :param method: 要装饰的方法。此方法应为 PyTorch 模块的方法。
    """
    # 如果加速不可用，则直接返回原方法
    if not is_accelerate_available():
        return method
    # 解析当前加速模块的版本
    accelerate_version = version.parse(accelerate.__version__).base_version
    # 如果加速版本小于 0.17.0，则直接返回原方法
    if version.parse(accelerate_version) < version.parse("0.17.0"):
        return method

    # 定义包装器函数
    def wrapper(self, *args, **kwargs):
        # 如果存在 `_hf_hook` 且具有 `pre_forward` 属性，则调用该钩子
        if hasattr(self, "_hf_hook") and hasattr(self._hf_hook, "pre_forward"):
            self._hf_hook.pre_forward(self)
        # 调用原方法并返回结果
        return method(self, *args, **kwargs)

    # 返回包装器函数
    return wrapper
```