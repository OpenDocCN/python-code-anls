# `.\models\qwen2\__init__.py`

```py
# 版权声明和许可信息
# 本代码受 Apache 许可证 2.0 版本保护，详细信息可查阅许可证
# http://www.apache.org/licenses/LICENSE-2.0

# 引入类型检查
from typing import TYPE_CHECKING

# 引入依赖的模块和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构，包括配置和标记化
_import_structure = {
    "configuration_qwen2": ["QWEN2_PRETRAINED_CONFIG_ARCHIVE_MAP", "Qwen2Config"],
    "tokenization_qwen2": ["Qwen2Tokenizer"],
}

# 检查 tokenizers 是否可用，若不可用则引发异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，增加快速标记化的导入结构
    _import_structure["tokenization_qwen2_fast"] = ["Qwen2TokenizerFast"]

# 检查 torch 是否可用，若不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，增加模型的导入结构
    _import_structure["modeling_qwen2"] = [
        "Qwen2ForCausalLM",
        "Qwen2Model",
        "Qwen2PreTrainedModel",
        "Qwen2ForSequenceClassification",
    ]

# 如果正在进行类型检查，导入相应的模块和函数
if TYPE_CHECKING:
    from .configuration_qwen2 import QWEN2_PRETRAINED_CONFIG_ARCHIVE_MAP, Qwen2Config
    from .tokenization_qwen2 import Qwen2Tokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_qwen2_fast import Qwen2TokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_qwen2 import (
            Qwen2ForCausalLM,
            Qwen2ForSequenceClassification,
            Qwen2Model,
            Qwen2PreTrainedModel,
        )

# 如果不是类型检查，将模块定义为延迟加载模块
else:
    import sys

    # 使用 LazyModule 将模块定义为延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```