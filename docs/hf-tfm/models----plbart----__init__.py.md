# `.\transformers\models\plbart\__init__.py`

```py
# 版权声明和许可证信息
# 版权归 The HuggingFace Team 所有
# 根据 Apache 许可证 2.0 版本授权
# 只有在遵守许可证的情况下才能使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发的
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的详细信息

# 导入必要的模块和函数
from typing import TYPE_CHECKING
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块导入结构
_import_structure = {"configuration_plbart": ["PLBART_PRETRAINED_CONFIG_ARCHIVE_MAP", "PLBartConfig"]}

# 检查是否存在 sentencepiece 库，如果不存在则引发异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_plbart"] = ["PLBartTokenizer"]

# 检查是否存在 torch 库，如果不存在则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_plbart"] = [
        "PLBART_PRETRAINED_MODEL_ARCHIVE_LIST",
        "PLBartForCausalLM",
        "PLBartForConditionalGeneration",
        "PLBartForSequenceClassification",
        "PLBartModel",
        "PLBartPreTrainedModel",
    ]

# 如果是类型检查模式，则导入相关模块
if TYPE_CHECKING:
    from .configuration_plbart import PLBART_PRETRAINED_CONFIG_ARCHIVE_MAP, PLBartConfig

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_plbart import PLBartTokenizer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_plbart import (
            PLBART_PRETRAINED_MODEL_ARCHIVE_LIST,
            PLBartForCausalLM,
            PLBartForConditionalGeneration,
            PLBartForSequenceClassification,
            PLBartModel,
            PLBartPreTrainedModel,
        )

# 如果不是类型检查模式，则将模块定义为 LazyModule
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```