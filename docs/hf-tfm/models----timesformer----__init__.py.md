# `.\transformers\models\timesformer\__init__.py`

```py
# 2022年 HuggingFace 团队版权所有
#
# 根据 Apache 许可证第 2.0 版（“许可证”）获得许可;
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是在“存在”的基础上分发的，没有任何形式的保证或条件，无论是明示的还是暗示的。
# 请参阅许可证以了解具体语言规定的权限和
# 限制
from typing import TYPE_CHECKING

# 导入必要的依赖项，包括检查类型是否可用的函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义导入结构，包括模块和类的元数据信息
_import_structure = {
    "configuration_timesformer": ["TIMESFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "TimesformerConfig"],
}

# 尝试导入 torch，如果不可用，则引发 OptionalDependencyNotAvailable 错误
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加额外的导入结构
    _import_structure["modeling_timesformer"] = [
        "TIMESFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TimesformerModel",
        "TimesformerForVideoClassification",
        "TimesformerPreTrainedModel",
    ]

# 如果是类型检查模式，则添加类型检查的导入结构
if TYPE_CHECKING:
    from .configuration_timesformer import TIMESFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, TimesformerConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_timesformer import (
            TIMESFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TimesformerForVideoClassification,
            TimesformerModel,
            TimesformerPreTrainedModel,
        )

# 如果不是类型检查模式，则延迟导入结构
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```