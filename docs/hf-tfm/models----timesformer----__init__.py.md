# `.\models\timesformer\__init__.py`

```py
# 版权声明和许可信息
#
# 版权所有 2022 年 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）进行许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于“按原样”提供的，
# 没有任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。

# 引入依赖检查相关模块
from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义导入结构
_import_structure = {
    "configuration_timesformer": ["TIMESFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "TimesformerConfig"],
}

# 检查是否有 Torch 可用
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果有 Torch 可用，则添加 Timesformer 相关模块到导入结构
    _import_structure["modeling_timesformer"] = [
        "TIMESFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TimesformerModel",
        "TimesformerForVideoClassification",
        "TimesformerPreTrainedModel",
    ]

# 如果是类型检查阶段，导入 Timesformer 相关配置和模型
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

# 如果不是类型检查阶段，则动态地将当前模块设置为延迟加载模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```