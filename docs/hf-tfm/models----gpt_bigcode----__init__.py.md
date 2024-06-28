# `.\models\gpt_bigcode\__init__.py`

```py
# 引入类型检查模块
from typing import TYPE_CHECKING

# 从工具包中引入异常和懒加载模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义导入结构字典，包含配置和模型
_import_structure = {
    "configuration_gpt_bigcode": ["GPT_BIGCODE_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPTBigCodeConfig"],
}

# 检查是否存在Torch库，若不存在则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若Torch可用，则添加模型相关的导入结构
    _import_structure["modeling_gpt_bigcode"] = [
        "GPT_BIGCODE_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GPTBigCodeForSequenceClassification",
        "GPTBigCodeForTokenClassification",
        "GPTBigCodeForCausalLM",
        "GPTBigCodeModel",
        "GPTBigCodePreTrainedModel",
    ]

# 如果类型检查开启
if TYPE_CHECKING:
    # 从配置文件中导入所需内容
    from .configuration_gpt_bigcode import GPT_BIGCODE_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTBigCodeConfig

    # 检查是否存在Torch库，若不存在则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从模型文件中导入所需内容
        from .modeling_gpt_bigcode import (
            GPT_BIGCODE_PRETRAINED_MODEL_ARCHIVE_LIST,
            GPTBigCodeForCausalLM,
            GPTBigCodeForSequenceClassification,
            GPTBigCodeForTokenClassification,
            GPTBigCodeModel,
            GPTBigCodePreTrainedModel,
        )

# 如果类型检查未开启
else:
    # 引入系统模块
    import sys

    # 将当前模块替换为懒加载模块实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```