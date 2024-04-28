# `.\transformers\models\timm_backbone\__init__.py`

```
# 禁用 flake8 对当前模块的检查，防止出现 "F401 '...' imported but unused" 警告
# 由于需要保持其他警告，所以无法忽略这种警告，因此完全不检查这个模块

# 版权声明，版权归 2023 年的 HuggingFace 团队所有
#
# 根据 Apache 许可证，版本 2.0 进行许可；
# 除非符合该许可证，否则不得使用此文件
# 你可以在下面链接处获得该许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发基于该许可证的软件,
# 没有任何形式的担保或条件，不管是明示的还是暗示的
# 查看许可证以获取有关特定语言的权限和限制

from typing import TYPE_CHECKING

# 导入必要的依赖
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 初始化导入结构字典，包含配置和模型的键
_import_structure = {"configuration_timm_backbone": ["TimmBackboneConfig"]}

# 检查 Torch 是否可用，如果不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则将模型结构添加到导入结构字典中
    _import_structure["modeling_timm_backbone"] = ["TimmBackbone"]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 从配置文件中导入 TimmBackboneConfig
    from .configuration_timm_backbone import TimmBackboneConfig

    # 检查 Torch 是否可用，如果不可用则抛出异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 Torch 可用，则从模型文件中导入 TimmBackbone
        from .modeling_timm_backbone import TimmBackbone

# 如果不是类型检查阶段
else:
    # 导入 sys 模块
    import sys

    # 将当前模块设置为 LazyModule，传入文件名、导入结构和模块规范
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```