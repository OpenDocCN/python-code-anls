# `.\transformers\models\umt5\__init__.py`

```py
# 版权声明和许可信息
# 版权归 The HuggingFace Team 所有
# 根据 Apache 许可证版本 2.0 许可
# 只有在遵守许可证的情况下才能使用此文件
# 您可以在以下网址获得许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或经书面同意，否则按“原样”分发的软件
# 没有任何形式的担保或条件，不管是明示的还是暗示的
# 请查看许可证了解具体语言下的权限和限制

# 导入必要的模块
from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 导入结构
_import_structure = {"configuration_umt5": ["UMT5Config", "UMT5OnnxConfig"]}

# 检查是否导入了 torch 模块
try:
    if not is_torch_available():
        # 如果没有导入 torch 模块，则引发 OptionalDependencyNotAvailable 异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果发生 OptionalDependencyNotAvailable 异常，则不做任何操作
    pass
else:
    # 如果没有发生 OptionalDependencyNotAvailable 异常，则添加以下模块到_import_structure中
    _import_structure["modeling_umt5"] = [
        "UMT5EncoderModel",
        "UMT5ForConditionalGeneration",
        "UMT5ForQuestionAnswering",
        "UMT5ForSequenceClassification",
        "UMT5Model",
        "UMT5PreTrainedModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入UMT5Config和UMT5OnnxConfig模块
    from .configuration_umt5 import UMT5Config, UMT5OnnxConfig

    # 再次检查是否导入了 torch 模块
    try:
        if not is_torch_available():
            # 如果没有导入 torch 模块，则引发 OptionalDependencyNotAvailable 异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果发生 OptionalDependencyNotAvailable 异常，则不做任何操作
        pass
    else:
        # 如果没有发生 OptionalDependencyNotAvailable 异常，则导入以下模块
        from .modeling_umt5 import (
            UMT5EncoderModel,
            UMT5ForConditionalGeneration,
            UMT5ForQuestionAnswering,
            UMT5ForSequenceClassification,
            UMT5Model,
            UMT5PreTrainedModel,
        )
# 如果不是类型检查模式
else:
    # 导入sys模块
    import sys
    # 将模块注册为_LazyModule
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```