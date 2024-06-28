# `.\models\vivit\__init__.py`

```
# flake8: noqa
# 忽略 flake8 检查，因为这里没有办法仅忽略 "F401 '...' imported but unused" 警告而保留其它警告。
# 这样做是为了确保保留其它警告，而不对本模块进行检查。

# Copyright 2023 The HuggingFace Team. All rights reserved.
# 版权声明，版权归 HuggingFace Team 所有。

# Licensed under the Apache License, Version 2.0 (the "License");
# 授权协议，使用 Apache License, Version 2.0 版本。

# you may not use this file except in compliance with the License.
# 只有在遵守协议的情况下才可以使用此文件。

# You may obtain a copy of the License at
# 您可以在以下网址获取协议的副本

#     http://www.apache.org/licenses/LICENSE-2.0
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 没有任何明示或暗示的担保或条件。

# See the License for the specific language governing permissions and
# limitations under the License.
# 详细信息请参阅许可证，以了解权限限制和特定语言的许可。

from typing import TYPE_CHECKING

# 依赖于 isort 来合并导入

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块的导入结构
_import_structure = {
    "configuration_vivit": ["VIVIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "VivitConfig"],
}

# 尝试导入视觉功能，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_vivit"] = ["VivitImageProcessor"]

# 尝试导入 Torch，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_vivit"] = [
        "VIVIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "VivitModel",
        "VivitPreTrainedModel",
        "VivitForVideoClassification",
    ]

# 如果 TYPE_CHECKING 为真，则从相应模块导入特定内容
if TYPE_CHECKING:
    from .configuration_vivit import VIVIT_PRETRAINED_CONFIG_ARCHIVE_MAP, VivitConfig

    # 尝试导入视觉功能，如果不可用则忽略
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_vivit import VivitImageProcessor

    # 尝试导入 Torch，如果不可用则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_vivit import (
            VIVIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            VivitForVideoClassification,
            VivitModel,
            VivitPreTrainedModel,
        )

# 如果不在 TYPE_CHECKING 模式下，则导入 LazyModule 动态生成模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```