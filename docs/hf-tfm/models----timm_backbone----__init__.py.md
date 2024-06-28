# `.\models\timm_backbone\__init__.py`

```py
# flake8: noqa
# 禁止 flake8 对本模块进行检查，因为无法忽略 "F401 '...' imported but unused" 警告，而保留其他警告。

# Copyright 2023 The HuggingFace Team. All rights reserved.
# 版权 2023 年 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本进行许可；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据此许可证分发的软件是基于“按原样”提供的，不附带任何明示或暗示的保证或条件。
# 请查阅许可证了解具体的法律语言和权限限制。

from typing import TYPE_CHECKING

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构
_import_structure = {"configuration_timm_backbone": ["TimmBackboneConfig"]}

try:
    # 检查是否有可用的 Torch
    if not is_torch_available():
        # 如果没有可用的 Torch，则抛出 OptionalDependencyNotAvailable 异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果出现 OptionalDependencyNotAvailable 异常，则什么也不做，继续执行
    pass
else:
    # 如果没有异常，则将 TimmBackbone 添加到导入结构中
    _import_structure["modeling_timm_backbone"] = ["TimmBackbone"]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 导入配置文件的 TimmBackboneConfig 类
    from .configuration_timm_backbone import TimmBackboneConfig

    try:
        # 再次检查是否有可用的 Torch
        if not is_torch_available():
            # 如果没有可用的 Torch，则抛出 OptionalDependencyNotAvailable 异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果出现 OptionalDependencyNotAvailable 异常，则什么也不做，继续执行
        pass
    else:
        # 如果没有异常，则导入 modeling_timm_backbone 模块的 TimmBackbone 类
        from .modeling_timm_backbone import TimmBackbone

# 如果不是类型检查阶段
else:
    # 导入 sys 模块
    import sys

    # 将当前模块替换为 _LazyModule 的实例，延迟加载模块内容
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```