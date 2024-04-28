# `.\transformers\models\table_transformer\__init__.py`

```
# 版权声明和许可证信息
# 版权所有 © 2022 年 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证，第 2 版（“许可证”）授权。
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的担保或条件。
# 有关特定语言以及使用许可证的详细信息，请参阅许可证。

# 导入需要的类型检查工具
from typing import TYPE_CHECKING

# 导入自定义的异常类
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块导入结构
_import_structure = {
    "configuration_table_transformer": [
        "TABLE_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "TableTransformerConfig",
        "TableTransformerOnnxConfig",
    ]
}

# 检查是否 torch 库可用，如果不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 可用，增加模型相关模块到导入结构中
    _import_structure["modeling_table_transformer"] = [
        "TABLE_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TableTransformerForObjectDetection",
        "TableTransformerModel",
        "TableTransformerPreTrainedModel",
    ]

# 如果是类型检查，导入相关配置和模型模块
if TYPE_CHECKING:
    from .configuration_table_transformer import (
        TABLE_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TableTransformerConfig,
        TableTransformerOnnxConfig,
    )

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_table_transformer import (
            TABLE_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TableTransformerForObjectDetection,
            TableTransformerModel,
            TableTransformerPreTrainedModel,
        )

# 如果不是类型检查，将当前模块设为懒加载模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```