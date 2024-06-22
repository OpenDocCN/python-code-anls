# `.\models\efficientformer\__init__.py`

```py
# 版权声明，版权归 The HuggingFace Team 所有
#
# 根据 Apache 许可证 2.0 版本进行许可
# 除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是基于 "按原样" 分发，没有任何明示或暗示的担保或条件
# 有关权限和限制的详细内容，请查看许可证
from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_torch_available,
    is_vision_available,
)

# 定义模块导入结构
_import_structure = {
    "configuration_efficientformer": [
        "EFFICIENTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "EfficientFormerConfig",
    ]
}

# 检查是否存在视觉处理依赖
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_efficientformer"] = ["EfficientFormerImageProcessor"]

# 检查是否存在 PyTorch 依赖
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_efficientformer"] = [
        "EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "EfficientFormerForImageClassification",
        "EfficientFormerForImageClassificationWithTeacher",
        "EfficientFormerModel",
        "EfficientFormerPreTrainedModel",
    ]

# 检查是否存在 TensorFlow 依赖
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_efficientformer"] = [
        "TF_EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFEfficientFormerForImageClassification",
        "TFEfficientFormerForImageClassificationWithTeacher",
        "TFEfficientFormerModel",
        "TFEfficientFormerPreTrainedModel",
    ]

# 如果是类型检查，则导入相关模块
if TYPE_CHECKING:
    from .configuration_efficientformer import EFFICIENTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, EfficientFormerConfig

    # 检查是否存在视觉处理依赖
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_efficientformer import EfficientFormerImageProcessor

    # 检查是否存在 PyTorch 依赖
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_efficientformer import (
            EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            EfficientFormerForImageClassification,
            EfficientFormerForImageClassificationWithTeacher,
            EfficientFormerModel,
            EfficientFormerPreTrainedModel,
        )
```  
    # 尝试检查 TensorFlow 是否可用
    try:
        if not is_tf_available():
            # 如果 TensorFlow 不可用，则引发可选依赖不可用的异常
            raise OptionalDependencyNotAvailable()
    # 捕获可选依赖不可用的异常
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有引发异常，则执行以下代码
    else:
        # 导入 TF_EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST 等模块
        from .modeling_tf_efficientformer import (
            TF_EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFEfficientFormerForImageClassification,
            TFEfficientFormerForImageClassificationWithTeacher,
            TFEfficientFormerModel,
            TFEfficientFormerPreTrainedModel,
        )
# 如果不在Jupyter Notebook中，则导入sys模块
import sys
# 将当前模块以延迟加载方式导入，并将其赋值给sys.modules中当前模块名的条目
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```