# `.\models\ibert\__init__.py`

```
# 版权声明和许可声明，指明此代码的版权和使用许可
# 详细许可信息可以在 Apache License, Version 2.0 网页上找到：http://www.apache.org/licenses/LICENSE-2.0
#
# 如果按照许可证的规定，在没有软件的任何保证或条件的情况下分发此软件
# 请查看许可证以了解更多详细信息。

# 引入类型检查模块
from typing import TYPE_CHECKING

# 引入自定义工具函数和模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义导入结构字典，用于延迟加载模块
_import_structure = {"configuration_ibert": ["IBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "IBertConfig", "IBertOnnxConfig"]}

# 检查是否存在 Torch 库，如果不存在则抛出自定义的依赖未可用异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则添加下列模块到导入结构字典中
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

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 从 configuration_ibert 模块中导入指定的类和变量
    from .configuration_ibert import IBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, IBertConfig, IBertOnnxConfig

    # 再次检查 Torch 是否可用，如果不可用则抛出自定义的依赖未可用异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 Torch 可用，则从 modeling_ibert 模块中导入指定的类和变量
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

# 如果不在类型检查模式下
else:
    # 导入 sys 模块
    import sys

    # 将当前模块替换为延迟加载模块 _LazyModule 的实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```