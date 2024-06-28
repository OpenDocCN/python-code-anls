# `.\models\umt5\__init__.py`

```py
# 版权声明和许可信息
#
# 版权 2023 年由 HuggingFace 团队所有。保留所有权利。
#
# 根据 Apache 许可证版本 2.0 授权。
# 除非符合许可证的要求，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 没有任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。

# 导入所需的类型检查工具
from typing import TYPE_CHECKING

# 引入自定义的异常和模块加载延迟工具
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义要导入的结构化模块的映射字典
_import_structure = {"configuration_umt5": ["UMT5Config", "UMT5OnnxConfig"]}

# 尝试检查是否有 Torch 库可用，如果不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则定义需要导入的模型相关模块列表
    _import_structure["modeling_umt5"] = [
        "UMT5EncoderModel",
        "UMT5ForConditionalGeneration",
        "UMT5ForQuestionAnswering",
        "UMT5ForSequenceClassification",
        "UMT5ForTokenClassification",
        "UMT5Model",
        "UMT5PreTrainedModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入配置相关的类型
    from .configuration_umt5 import UMT5Config, UMT5OnnxConfig

    # 再次尝试检查 Torch 是否可用，如果不可用则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关的类型
        from .modeling_umt5 import (
            UMT5EncoderModel,
            UMT5ForConditionalGeneration,
            UMT5ForQuestionAnswering,
            UMT5ForSequenceClassification,
            UMT5ForTokenClassification,
            UMT5Model,
            UMT5PreTrainedModel,
        )
# 如果不是类型检查模式
else:
    import sys

    # 将当前模块设置为延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```