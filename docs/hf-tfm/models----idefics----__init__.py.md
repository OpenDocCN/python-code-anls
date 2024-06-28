# `.\models\idefics\__init__.py`

```
# 版权声明及许可证信息，声明代码版权归 HuggingFace 团队所有，基于 Apache License, Version 2.0 发布
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

# 导入类型检查模块中的 TYPE_CHECKING
from typing import TYPE_CHECKING

# 从工具模块中导入必要的异常和工具函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块导入结构，包含 IDEFiCS 配置和模型相关的结构
_import_structure = {"configuration_idefics": ["IDEFICS_PRETRAINED_CONFIG_ARCHIVE_MAP", "IdeficsConfig"]}

# 检查是否图像处理可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加图像处理相关的导入结构
    _import_structure["image_processing_idefics"] = ["IdeficsImageProcessor"]

# 检查是否 Torch 可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加模型和处理相关的导入结构
    _import_structure["modeling_idefics"] = [
        "IDEFICS_PRETRAINED_MODEL_ARCHIVE_LIST",
        "IdeficsForVisionText2Text",
        "IdeficsModel",
        "IdeficsPreTrainedModel",
    ]
    _import_structure["processing_idefics"] = ["IdeficsProcessor"]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 从 configuration_idefics 模块导入必要的类和对象
    from .configuration_idefics import IDEFICS_PRETRAINED_CONFIG_ARCHIVE_MAP, IdeficsConfig

    # 检查是否图像处理可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，从 image_processing_idefics 模块导入必要的类和对象
        from .image_processing_idefics import IdeficsImageProcessor

    # 检查是否 Torch 可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，从 modeling_idefics 和 processing_idefics 模块导入必要的类和对象
        from .modeling_idefics import (
            IDEFICS_PRETRAINED_MODEL_ARCHIVE_LIST,
            IdeficsForVisionText2Text,
            IdeficsModel,
            IdeficsPreTrainedModel,
        )
        from .processing_idefics import IdeficsProcessor

# 如果不是类型检查阶段
else:
    import sys

    # 将当前模块设置为延迟加载模块，使用 _LazyModule 进行懒加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```