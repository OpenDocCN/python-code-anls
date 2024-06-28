# `.\models\llava_next\__init__.py`

```py
# Copyright 2024 The HuggingFace Team. All rights reserved.
# 版权声明，版权归HuggingFace团队所有

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# 本文件采用Apache许可证2.0版授权，除非符合许可证要求，否则不得使用本文件

# You may obtain a copy of the License at
# 可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非适用法律要求或书面同意，否则按"原样"分发本软件，无任何明示或暗示的保证或条件

# See the License for the specific language governing permissions and
# limitations under the License.
# 请参阅许可证了解具体的语言规定和限制

from typing import TYPE_CHECKING

# 从HuggingFace内部utils模块中导入OptionalDependencyNotAvailable和_LazyModule
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块导入结构
_import_structure = {
    "configuration_llava_next": ["LLAVA_NEXT_PRETRAINED_CONFIG_ARCHIVE_MAP", "LlavaNextConfig"],
    "processing_llava_next": ["LlavaNextProcessor"],
}

# 检查是否Torch可用，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若Torch可用，则更新导入结构中的模型相关模块
    _import_structure["modeling_llava_next"] = [
        "LLAVA_NEXT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "LlavaNextForConditionalGeneration",
        "LlavaNextPreTrainedModel",
    ]

# 检查是否Vision模块可用，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若Vision模块可用，则更新导入结构中的图像处理相关模块
    _import_structure["image_processing_llava_next"] = ["LlavaNextImageProcessor"]

# 如果是类型检查模式（如mypy），则导入相应的模块
if TYPE_CHECKING:
    from .configuration_llava_next import LLAVA_NEXT_PRETRAINED_CONFIG_ARCHIVE_MAP, LlavaNextConfig
    from .processing_llava_next import LlavaNextProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_llava_next import (
            LLAVA_NEXT_PRETRAINED_MODEL_ARCHIVE_LIST,
            LlavaNextForConditionalGeneration,
            LlavaNextPreTrainedModel,
        )

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_llava_next import LlavaNextImageProcessor

# 如果不是类型检查模式，则使用_LazyModule懒加载模块
else:
    import sys

    # 将当前模块注册为_LazyModule，以实现延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```