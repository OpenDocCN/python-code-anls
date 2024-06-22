# `.\transformers\models\sam\__init__.py`

```py
# 版权声明
#
# 本代码版权归 HuggingFace 团队所有。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）许可；
# 除非符合许可证规定，否则您不能使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件按"原样"分发，
# 不附带任何明示或暗示的担保或条件。
# 有关许可证下的特定语言的权限和限制，请参阅许可证。
from typing import TYPE_CHECKING

# 从 utils 模块中导入所需函数和异常
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_torch_available,
    is_vision_available,
)

# 定义模块导入结构
_import_structure = {
    "configuration_sam": [
        "SAM_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "SamConfig",
        "SamMaskDecoderConfig",
        "SamPromptEncoderConfig",
        "SamVisionConfig",
    ],
    "processing_sam": ["SamProcessor"],
}

# 检查是否有 torch 可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若 torch 可用，则添加 modeling_sam 模块到导入结构
    _import_structure["modeling_sam"] = [
        "SAM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SamModel",
        "SamPreTrainedModel",
    ]

# 检查是否有 TensorFlow 可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若 TensorFlow 可用，则添加 modeling_tf_sam 模块到导入结构
    _import_structure["modeling_tf_sam"] = [
        "TF_SAM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFSamModel",
        "TFSamPreTrainedModel",
    ]

# 检查是否有视觉处理库可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若视觉处理库可用，则添加 image_processing_sam 模块到导入结构
    _import_structure["image_processing_sam"] = ["SamImageProcessor"]

# 若类型检查开启
if TYPE_CHECKING:
    # 从 configuration_sam 模块中导入相关类
    from .configuration_sam import (
        SAM_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SamConfig,
        SamMaskDecoderConfig,
        SamPromptEncoderConfig,
        SamVisionConfig,
    )
    # 从 processing_sam 模块中导入 SamProcessor 类
    from .processing_sam import SamProcessor

    # 若 torch 可用，则从 modeling_sam 模块中导入相关类
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_sam import SAM_PRETRAINED_MODEL_ARCHIVE_LIST, SamModel, SamPreTrainedModel

    # 若 TensorFlow 可用，则从 modeling_tf_sam 模块中导入相关类
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_sam import TF_SAM_PRETRAINED_MODEL_ARCHIVE_LIST, TFSamModel, TFSamPreTrainedModel

    # 若视觉处理库可用，则从 image_processing_sam 模块中导入 SamImageProcessor 类
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_sam import SamImageProcessor

# 若非类型检查模式
else:
    import sys

    # 将当前模块指定为惰性加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```  
```