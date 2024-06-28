# `.\models\donut\__init__.py`

```py
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入自定义异常和LazyModule类、检查是否有torch和vision库可用的函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块导入结构字典
_import_structure = {
    "configuration_donut_swin": ["DONUT_SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP", "DonutSwinConfig"],
    "processing_donut": ["DonutProcessor"],
}

# 检查是否有torch库可用，如果不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加相关模块到_import_structure字典中
    _import_structure["modeling_donut_swin"] = [
        "DONUT_SWIN_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DonutSwinModel",
        "DonutSwinPreTrainedModel",
    ]

# 检查是否有vision库可用，如果不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加相关模块到_import_structure字典中
    _import_structure["feature_extraction_donut"] = ["DonutFeatureExtractor"]
    _import_structure["image_processing_donut"] = ["DonutImageProcessor"]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 从相应模块导入特定类和变量
    from .configuration_donut_swin import DONUT_SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP, DonutSwinConfig
    from .processing_donut import DonutProcessor

    try:
        # 检查是否有torch库可用，如果不可用则抛出OptionalDependencyNotAvailable异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，从相应模块导入特定类和变量
        from .modeling_donut_swin import (
            DONUT_SWIN_PRETRAINED_MODEL_ARCHIVE_LIST,
            DonutSwinModel,
            DonutSwinPreTrainedModel,
        )

    try:
        # 检查是否有vision库可用，如果不可用则抛出OptionalDependencyNotAvailable异常
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，从相应模块导入特定类和变量
        from .feature_extraction_donut import DonutFeatureExtractor
        from .image_processing_donut import DonutImageProcessor

# 如果不是在类型检查模式下
else:
    # 导入sys模块
    import sys

    # 将当前模块替换为_LazyModule的实例，通过LazyModule进行按需导入
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```