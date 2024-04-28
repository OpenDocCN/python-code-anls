# `.\transformers\models\trocr\__init__.py`

```
# 2021年版权声明
#
# 根据 Apache 许可证 2.0 版本进行许可; 除非符合许可证的要求，否则不得使用此文件
# 您可以在以下位置获得许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则不得基于 "AS IS" 基础发布软件，
# 没有明示或暗示的担保或条件。
# 查看许可证以获取特定语言中的权限和限制
from typing import TYPE_CHECKING

# 导入工具包中定义的依赖项和类型判断
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_speech_available,
    is_torch_available,
)

# 定义导入结构
_import_structure = {
    "configuration_trocr": ["TROCR_PRETRAINED_CONFIG_ARCHIVE_MAP", "TrOCRConfig"],
    "processing_trocr": ["TrOCRProcessor"],
}

# 尝试导入 torch，如果不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_trocr"] = [
        "TROCR_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TrOCRForCausalLM",
        "TrOCRPreTrainedModel",
    ]

# 如果是类型检查，则导入特定模块和类
if TYPE_CHECKING:
    from .configuration_trocr import TROCR_PRETRAINED_CONFIG_ARCHIVE_MAP, TrOCRConfig
    from .processing_trocr import TrOCRProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_trocr import TROCR_PRETRAINED_MODEL_ARCHIVE_LIST, TrOCRForCausalLM, TrOCRPreTrainedModel

# 如果不是类型检查，则将 LazyModule 添加到当前模块
else:
    import sys
    # 将 LazyModule 添加到当前模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```