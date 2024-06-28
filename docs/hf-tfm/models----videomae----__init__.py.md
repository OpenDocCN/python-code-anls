# `.\models\videomae\__init__.py`

```py
# 版权声明及许可证信息，声明代码版权及使用许可
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

# 引入类型检查模块中的 TYPE_CHECKING 类型
from typing import TYPE_CHECKING

# 引入依赖模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块导入结构
_import_structure = {
    "configuration_videomae": ["VIDEOMAE_PRETRAINED_CONFIG_ARCHIVE_MAP", "VideoMAEConfig"],
}

# 检查是否有 torch 库可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 可用，则添加 modeling_videomae 模块到导入结构
    _import_structure["modeling_videomae"] = [
        "VIDEOMAE_PRETRAINED_MODEL_ARCHIVE_LIST",
        "VideoMAEForPreTraining",
        "VideoMAEModel",
        "VideoMAEPreTrainedModel",
        "VideoMAEForVideoClassification",
    ]

# 检查是否有 vision 库可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 vision 可用，则添加 feature_extraction_videomae 和 image_processing_videomae 模块到导入结构
    _import_structure["feature_extraction_videomae"] = ["VideoMAEFeatureExtractor"]
    _import_structure["image_processing_videomae"] = ["VideoMAEImageProcessor"]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 从 configuration_videomae 模块导入指定内容
    from .configuration_videomae import VIDEOMAE_PRETRAINED_CONFIG_ARCHIVE_MAP, VideoMAEConfig

    # 检查是否有 torch 库可用，如果不可用则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从 modeling_videomae 模块导入指定内容
        from .modeling_videomae import (
            VIDEOMAE_PRETRAINED_MODEL_ARCHIVE_LIST,
            VideoMAEForPreTraining,
            VideoMAEForVideoClassification,
            VideoMAEModel,
            VideoMAEPreTrainedModel,
        )

    # 检查是否有 vision 库可用，如果不可用则忽略
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从 feature_extraction_videomae 和 image_processing_videomae 模块导入指定内容
        from .feature_extraction_videomae import VideoMAEFeatureExtractor
        from .image_processing_videomae import VideoMAEImageProcessor

# 如果不是类型检查模式
else:
    import sys

    # 将当前模块映射到 LazyModule，用于懒加载导入结构
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```