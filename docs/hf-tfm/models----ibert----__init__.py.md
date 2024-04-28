# `.\models\ibert\__init__.py`

```
# 版权声明和许可证信息
# 版权归 The HuggingFace Team 所有，保留所有权利
# 根据 Apache 许可证版本 2.0 授权
# 除非符合许可证要求，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则软件按"原样"分发，不提供任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的具体语言

# 导入必要的模块和函数
from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块导入结构
_import_structure = {"configuration_ibert": ["IBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "IBertConfig", "IBertOnnxConfig"]}

# 检查是否存在 torch 库，如果不存在则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 torch 库，则添加以下模块到导入结构中
    _import_structure["modeling_ibert"] = [
        "IBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "IBertForMaskedLM",
        "IBertForMultipleChoice",
        "IBertForQuestionAnswering",
        "IBertForSequenceClassification",
        "IBertForTokenClassification",
        "IBertModel",
        "IBertPreTrainedModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入配置相关的模块
    from .configuration_ibert import IBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, IBertConfig, IBertOnnxConfig

    # 检查是否存在 torch 库，如果不存在则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关的模块
        from .modeling_ibert import (
            IBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            IBertForMaskedLM,
            IBertForMultipleChoice,
            IBertForQuestionAnswering,
            IBertForSequenceClassification,
            IBertForTokenClassification,
            IBertModel,
            IBertPreTrainedModel,
        )

# 如果不是类型检查模式
else:
    import sys

    # 将当前模块设置为 LazyModule，延迟导入模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```