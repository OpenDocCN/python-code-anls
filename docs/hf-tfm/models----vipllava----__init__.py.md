# `.\transformers\models\vipllava\__init__.py`

```
# 版权声明和许可信息
# 版权归 HuggingFace 团队所有，保留所有权利。
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不可使用此文件
# 可以获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 依据适用法律或书面同意，以"现有的"基础分发软件，不提供任何担保或条件，无论是明示的还是暗示的。
# 请参阅许可证以获取有关权限和限制的详细信息

# 导入类型检查相关模块
from typing import TYPE_CHECKING

# 导入必需的依赖
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义各模块的导入结构
_import_structure = {"configuration_vipllava": ["VIPLLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP", "VipLlavaConfig"]}

# 检查是否有必需的 Torch 依赖
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 添加模型导入结构
    _import_structure["modeling_vipllava"] = [
        "VIPLLAVA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "VipLlavaForConditionalGeneration",
        "VipLlavaPreTrainedModel",
    ]

# 如果是类型检查环境
if TYPE_CHECKING:
    # 导入相关的配置和模型内容
    from .configuration_vipllava import VIPLLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP, VipLlavaConfig

    # 检查 Torch 依赖
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关内容
        from .modeling_vipllava import (
            VIPLLAVA_PRETRAINED_MODEL_ARCHIVE_LIST,
            VipLlavaForConditionalGeneration,
            VipLlavaPreTrainedModel,
        )

# 如果不是类型检查环境
else:
    import sys

    # 将当前模块注册到 sys.modules 中
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```  
```