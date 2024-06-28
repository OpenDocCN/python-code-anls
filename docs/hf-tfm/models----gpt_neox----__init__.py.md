# `.\models\gpt_neox\__init__.py`

```
# 版权声明，指明代码的版权信息
# 根据 Apache License, Version 2.0 许可证授权使用此文件
# 如果不符合许可证要求，禁止使用此文件
from typing import TYPE_CHECKING

# 从文件工具中导入 LazyModule、is_tokenizers_available 和 is_torch_available 函数
from ...file_utils import _LazyModule, is_tokenizers_available, is_torch_available
# 从工具函数中导入 OptionalDependencyNotAvailable 异常类
from ...utils import OptionalDependencyNotAvailable

# 定义模块导入结构字典
_import_structure = {"configuration_gpt_neox": ["GPT_NEOX_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPTNeoXConfig"]}

# 检查是否存在 tokenizers 库，如果不存在则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若存在 tokenizers 库，则将 GPTNeoXTokenizerFast 导入到导入结构字典中
    _import_structure["tokenization_gpt_neox_fast"] = ["GPTNeoXTokenizerFast"]

# 检查是否存在 torch 库，如果不存在则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若存在 torch 库，则将 GPTNeoX 相关模型导入到导入结构字典中
    _import_structure["modeling_gpt_neox"] = [
        "GPT_NEOX_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GPTNeoXForCausalLM",
        "GPTNeoXForQuestionAnswering",
        "GPTNeoXForSequenceClassification",
        "GPTNeoXForTokenClassification",
        "GPTNeoXLayer",
        "GPTNeoXModel",
        "GPTNeoXPreTrainedModel",
    ]

# 如果类型检查为真，则从 configuration_gpt_neox 模块导入配置映射和配置类
if TYPE_CHECKING:
    from .configuration_gpt_neox import GPT_NEOX_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTNeoXConfig

    # 检查是否存在 tokenizers 库，如果不存在则跳过导入
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若存在 tokenizers 库，则从 tokenization_gpt_neox_fast 模块导入 GPTNeoXTokenizerFast 类
        from .tokenization_gpt_neox_fast import GPTNeoXTokenizerFast

    # 检查是否存在 torch 库，如果不存在则跳过导入
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若存在 torch 库，则从 modeling_gpt_neox 模块导入 GPTNeoX 相关类
        from .modeling_gpt_neox import (
            GPT_NEOX_PRETRAINED_MODEL_ARCHIVE_LIST,
            GPTNeoXForCausalLM,
            GPTNeoXForQuestionAnswering,
            GPTNeoXForSequenceClassification,
            GPTNeoXForTokenClassification,
            GPTNeoXLayer,
            GPTNeoXModel,
            GPTNeoXPreTrainedModel,
        )

# 如果不是类型检查模式，则将当前模块指定为 LazyModule 的代理
else:
    import sys

    # 将当前模块注册为 LazyModule，使用 LazyModule 实现延迟加载模块的功能
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```