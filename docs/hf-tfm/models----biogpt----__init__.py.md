# `.\transformers\models\biogpt\__init__.py`

```py
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 导入必要的模块和函数
from typing import TYPE_CHECKING
# 导入自定义的异常类
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义模块的导入结构
_import_structure = {
    "configuration_biogpt": ["BIOGPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "BioGptConfig"],
    "tokenization_biogpt": ["BioGptTokenizer"],
}

# 检查是否存在 torch 库，若不存在则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若存在 torch 库，则添加相关模型到导入结构中
    _import_structure["modeling_biogpt"] = [
        "BIOGPT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BioGptForCausalLM",
        "BioGptForTokenClassification",
        "BioGptForSequenceClassification",
        "BioGptModel",
        "BioGptPreTrainedModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入配置和分词器相关模块
    from .configuration_biogpt import BIOGPT_PRETRAINED_CONFIG_ARCHIVE_MAP, BioGptConfig
    from .tokenization_biogpt import BioGptTokenizer

    # 检查是否存在 torch 库，若不存在则抛出异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关模块
        from .modeling_biogpt import (
            BIOGPT_PRETRAINED_MODEL_ARCHIVE_LIST,
            BioGptForCausalLM,
            BioGptForSequenceClassification,
            BioGptForTokenClassification,
            BioGptModel,
            BioGptPreTrainedModel,
        )

# 如果不是类型检查模式
else:
    import sys

    # 将当前模块设置为 LazyModule 类型
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```