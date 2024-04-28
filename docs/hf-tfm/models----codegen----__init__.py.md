# `.\models\codegen\__init__.py`

```py
# 版权声明及许可信息

# 导入必要的模块和函数
from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义模块导入结构
_import_structure = {
    "configuration_codegen": ["CODEGEN_PRETRAINED_CONFIG_ARCHIVE_MAP", "CodeGenConfig", "CodeGenOnnxConfig"],
    "tokenization_codegen": ["CodeGenTokenizer"],
}

# 检查是否存在 tokenizers 包，若不存在则引发异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 tokenizers 包，则增加相应的导入结构
    _import_structure["tokenization_codegen_fast"] = ["CodeGenTokenizerFast"]

# 检查是否存在 torch 包，若不存在则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 torch 包，则增加相应的导入结构
    _import_structure["modeling_codegen"] = [
        "CODEGEN_PRETRAINED_MODEL_ARCHIVE_LIST",
        "CodeGenForCausalLM",
        "CodeGenModel",
        "CodeGenPreTrainedModel",
    ]

# 如果当前环境是类型检查模式
if TYPE_CHECKING:
    # 导入必要的配置和模型类
    from .configuration_codegen import CODEGEN_PRETRAINED_CONFIG_ARCHIVE_MAP, CodeGenConfig, CodeGenOnnxConfig
    from .tokenization_codegen import CodeGenTokenizer

    # 检查是否存在 tokenizers 包，若不存在则引发异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果存在 tokenizers 包，则增加相应的导入结构
        from .tokenization_codegen_fast import CodeGenTokenizerFast

    # 检查是否存在 torch 包，若不存在则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果存在 torch 包，则增加相应的导入结构
        from .modeling_codegen import (
            CODEGEN_PRETRAINED_MODEL_ARCHIVE_LIST,
            CodeGenForCausalLM,
            CodeGenModel,
            CodeGenPreTrainedModel,
        )

# 如果不是类型检查模式
else:
    import sys

    # 动态创建延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```