# `.\models\deprecated\mctct\__init__.py`

```py
# 版权声明
# 版权所有2022年HuggingFace团队。保留所有权利。
#
# 根据Apache许可证2.0版（以下简称“许可证”）获得许可，您不得使用此文件，除非遵守许可。 
# 您可以在以下网址获取许可的副本
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或经书面同意，否则根据许可分发的软件将按“按原样”基础分发，
# 没有任何明示或默示的保证或条件。请参阅许可证以了解特定的语言管理权限和限制。
from typing import TYPE_CHECKING

from ....utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available


_import_structure = {
    "configuration_mctct": ["MCTCT_PRETRAINED_CONFIG_ARCHIVE_MAP", "MCTCTConfig"],
    "feature_extraction_mctct": ["MCTCTFeatureExtractor"],
    "processing_mctct": ["MCTCTProcessor"],
}

# 尝试导入torch，如果不可用，则引发OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加以下modeling_mctct模块到import结构中
    _import_structure["modeling_mctct"] = [
        "MCTCT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MCTCTForCTC",
        "MCTCTModel",
        "MCTCTPreTrainedModel",
    ]

# 如果是类型检查，导入相应的类型
if TYPE_CHECKING:
    from .configuration_mctct import MCTCT_PRETRAINED_CONFIG_ARCHIVE_MAP, MCTCTConfig
    from .feature_extraction_mctct import MCTCTFeatureExtractor
    from .processing_mctct import MCTCTProcessor

    # 同样地，尝试导入torch，如果不可用，则引发OptionalDependencyNotAvailable异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则导入modeling_mctct模块的内容
        from .modeling_mctct import MCTCT_PRETRAINED_MODEL_ARCHIVE_LIST, MCTCTForCTC, MCTCTModel, MCTCTPreTrainedModel
else:
    import sys

    # 把LazyModule的实例赋值给当前模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```