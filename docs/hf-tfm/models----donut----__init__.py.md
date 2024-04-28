# `.\models\donut\__init__.py`

```py
# 版权声明
# 2022年HuggingFace团队。保留所有权利。
#
# 基于Apache许可证2.0版（“许可证”）进行许可;
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获得许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或经书面同意，否则软件
# 根据许可证以“原样”分发，不附带任何明示或
# 暗示的担保或条件。
# 查看许可证以获取特定语言控制权限和
# 许可证下的限制
from typing import TYPE_CHECKING  # 导入TYPE_CHECKING类型检查

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available  # 导入所需的依赖项

# 定义需要动态导入的模块结构
_import_structure = {
    "configuration_donut_swin": ["DONUT_SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP", "DonutSwinConfig"],  # 导入模型配置相关内容
    "processing_donut": ["DonutProcessor"],  # 导入处理器相关内容
}

# 检查是否存在torch库，如果不存在则引发OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass  # 如果出现OptionalDependencyNotAvailable异常则忽略，继续执行

# 如果存在torch库，导入模型和预训练模型相关内容
else:
    _import_structure["modeling_donut_swin"] = [
        "DONUT_SWIN_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DonutSwinModel",
        "DonutSwinPreTrainedModel",
    ]

# 检查是否存在vision库，如果不存在则引发OptionalDependencyNotAvailable异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass  # 如果出现OptionalDependencyNotAvailable异常则忽略，继续执行
else:
    # 如果存在vision库，导入特征提取器和图像处理器相关内容
    _import_structure["feature_extraction_donut"] = ["DonutFeatureExtractor"]
    _import_structure["image_processing_donut"] = ["DonutImageProcessor"]


if TYPE_CHECKING:
    from .configuration_donut_swin import DONUT_SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP, DonutSwinConfig  # 动态导入类型检查所需的内容
    from .processing_donut import DonutProcessor  # 动态导入类型检查所需的内容

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 动态导入类型检查所需的内容
        from .modeling_donut_swin import (
            DONUT_SWIN_PRETRAINED_MODEL_ARCHIVE_LIST,
            DonutSwinModel,
            DonutSwinPreTrainedModel,
        )

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 动态导入类型检查所需的内容
        from .feature_extraction_donut import DonutFeatureExtractor
        from .image_processing_donut import DonutImageProcessor

else:
    import sys

    # 使用_LazyModule类创建LazyModule并将其赋值给当前模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```