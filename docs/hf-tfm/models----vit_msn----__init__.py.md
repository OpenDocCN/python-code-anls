# `.\transformers\models\vit_msn\__init__.py`

```
# 版权声明及许可证信息
# 版权©2020 The HuggingFace Team。保留所有权利。
# 根据Apache许可证第2.0版（“许可证”）获得许可；
# 您不得使用本文件，除非符合许可证的规定。
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则不得基于许可证分发软件
# 以“原样”为基础，无论明示或暗示的条件，都没有任何担保或条件。
# 有关特定语言的权限和限制，请参阅许可证

# 导入必要的库
from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 导入结构定义
_import_structure = {"configuration_vit_msn": ["VIT_MSN_PRETRAINED_CONFIG_ARCHIVE_MAP", "ViTMSNConfig"]}

# 检查torch是否可用
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 更新结构定义
    _import_structure["modeling_vit_msn"] = [
        "VIT_MSN_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ViTMSNModel",
        "ViTMSNForImageClassification",
        "ViTMSNPreTrainedModel",
    ]

# 类型检查
if TYPE_CHECKING:
    # 导入类型检查相关结构
    from .configuration_vit_msn import VIT_MSN_PRETRAINED_CONFIG_ARCHIVE_MAP, ViTMSNConfig
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入类型检查相关结构
        from .modeling_vit_msn import (
            VIT_MSN_PRETRAINED_MODEL_ARCHIVE_LIST,
            ViTMSNForImageClassification,
            ViTMSNModel,
            ViTMSNPreTrainedModel,
        )
else:
    import sys
    # 设置懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```