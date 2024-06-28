# `.\models\deit\__init__.py`

```py
# 版权声明和许可证声明
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

# 从 HuggingFace 内部的 utils 模块中导入所需的依赖和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_torch_available,
    is_vision_available,
)

# 定义导入结构字典，包含需要导入的模块和类
_import_structure = {"configuration_deit": ["DEIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "DeiTConfig", "DeiTOnnxConfig"]}

# 检查视觉处理是否可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若视觉处理可用，则添加特征提取和图像处理模块到导入结构中
    _import_structure["feature_extraction_deit"] = ["DeiTFeatureExtractor"]
    _import_structure["image_processing_deit"] = ["DeiTImageProcessor"]

# 检查是否 Torch 可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若 Torch 可用，则添加模型建模相关的类到导入结构中
    _import_structure["modeling_deit"] = [
        "DEIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DeiTForImageClassification",
        "DeiTForImageClassificationWithTeacher",
        "DeiTForMaskedImageModeling",
        "DeiTModel",
        "DeiTPreTrainedModel",
    ]

# 检查是否 TensorFlow 可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若 TensorFlow 可用，则添加 TensorFlow 下的模型建模相关的类到导入结构中
    _import_structure["modeling_tf_deit"] = [
        "TF_DEIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFDeiTForImageClassification",
        "TFDeiTForImageClassificationWithTeacher",
        "TFDeiTForMaskedImageModeling",
        "TFDeiTModel",
        "TFDeiTPreTrainedModel",
    ]

# 若当前环境支持类型检查，则进行进一步的导入
if TYPE_CHECKING:
    # 从相应模块中导入具体的类和配置
    from .configuration_deit import DEIT_PRETRAINED_CONFIG_ARCHIVE_MAP, DeiTConfig, DeiTOnnxConfig

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若视觉处理可用，则进一步导入特征提取和图像处理相关类
        from .feature_extraction_deit import DeiTFeatureExtractor
        from .image_processing_deit import DeiTImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若 Torch 可用，则进一步导入模型建模相关的类
        from .modeling_deit import (
            DEIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            DeiTForImageClassification,
            DeiTForImageClassificationWithTeacher,
            DeiTForMaskedImageModeling,
            DeiTModel,
            DeiTPreTrainedModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    # 如果发生 OptionalDependencyNotAvailable 异常，则什么都不做，直接 pass
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有发生异常，则导入以下模块
    else:
        from .modeling_tf_deit import (
            TF_DEIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFDeiTForImageClassification,
            TFDeiTForImageClassificationWithTeacher,
            TFDeiTForMaskedImageModeling,
            TFDeiTModel,
            TFDeiTPreTrainedModel,
        )
else:
    # 导入 sys 模块，用于操作 Python 解释器的系统功能
    import sys

    # 将当前模块添加到 sys.modules 中，使用 _LazyModule 进行延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```