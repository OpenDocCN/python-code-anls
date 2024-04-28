# `.\models\gpt_bigcode\__init__.py`

```
# 引入类型检查模块，用于在类型检查时使用
from typing import TYPE_CHECKING
# 引入自定义工具函数和异常类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义模块导入结构
_import_structure = {
    "configuration_gpt_bigcode": ["GPT_BIGCODE_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPTBigCodeConfig"],
}

# 尝试检查是否有 torch 库可用
try:
    # 如果 torch 库不可用，则抛出 OptionalDependencyNotAvailable 异常
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
# 处理 OptionalDependencyNotAvailable 异常
except OptionalDependencyNotAvailable:
    pass
# 如果 torch 库可用
else:
    # 将 GPT BigCode 模型相关的模块添加到模块导入结构中
    _import_structure["modeling_gpt_bigcode"] = [
        "GPT_BIGCODE_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GPTBigCodeForSequenceClassification",
        "GPTBigCodeForTokenClassification",
        "GPTBigCodeForCausalLM",
        "GPTBigCodeModel",
        "GPTBigCodePreTrainedModel",
    ]

# 如果处于类型检查模式
if TYPE_CHECKING:
    # 从 configuration_gpt_bigcode 模块中导入 GPTBigCodeConfig 类和 GPT_BIGCODE_PRETRAINED_CONFIG_ARCHIVE_MAP 字典
    from .configuration_gpt_bigcode import GPT_BIGCODE_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTBigCodeConfig

    try:
        # 如果 torch 库不可用，则抛出 OptionalDependencyNotAvailable 异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    # 处理 OptionalDependencyNotAvailable 异常
    except OptionalDependencyNotAvailable:
        pass
    # 如果 torch 库可用
    else:
        # 从 modeling_gpt_bigcode 模块中导入 GPTBigCode 模型相关的类和函数
        from .modeling_gpt_bigcode import (
            GPT_BIGCODE_PRETRAINED_MODEL_ARCHIVE_LIST,
            GPTBigCodeForCausalLM,
            GPTBigCodeForSequenceClassification,
            GPTBigCodeForTokenClassification,
            GPTBigCodeModel,
            GPTBigCodePreTrainedModel,
        )

# 如果不处于类型检查模式
else:
    # 导入 sys 模块
    import sys
    # 将当前模块设为惰性加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```