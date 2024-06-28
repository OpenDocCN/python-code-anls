# `.\models\seamless_m4t_v2\__init__.py`

```py
# Copyright 2023 The HuggingFace Team. All rights reserved.
# 版权声明，版权归HuggingFace团队所有

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
# Apache License 2.0许可证声明，指定了在符合许可证条件的情况下可以使用本文件的条款

from typing import TYPE_CHECKING
# 导入TYPE_CHECKING用于类型检查

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)
# 从相对路径导入依赖模块和函数

_import_structure = {
    "configuration_seamless_m4t_v2": ["SEAMLESS_M4T_V2_PRETRAINED_CONFIG_ARCHIVE_MAP", "SeamlessM4Tv2Config"],
}
# 定义要导入的结构字典，包含配置和模型

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
# 如果torch不可用，抛出OptionalDependencyNotAvailable异常
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_seamless_m4t_v2"] = [
        "SEAMLESS_M4T_V2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SeamlessM4Tv2ForTextToSpeech",
        "SeamlessM4Tv2ForSpeechToSpeech",
        "SeamlessM4Tv2ForTextToText",
        "SeamlessM4Tv2ForSpeechToText",
        "SeamlessM4Tv2Model",
        "SeamlessM4Tv2PreTrainedModel",
    ]
# 如果torch可用，则添加模型相关的导入结构到_import_structure字典

if TYPE_CHECKING:
    from .configuration_seamless_m4t_v2 import SEAMLESS_M4T_V2_PRETRAINED_CONFIG_ARCHIVE_MAP, SeamlessM4Tv2Config
    # 如果正在进行类型检查，则从配置文件导入特定的类和映射

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_seamless_m4t_v2 import (
            SEAMLESS_M4T_V2_PRETRAINED_MODEL_ARCHIVE_LIST,
            SeamlessM4Tv2ForSpeechToSpeech,
            SeamlessM4Tv2ForSpeechToText,
            SeamlessM4Tv2ForTextToSpeech,
            SeamlessM4Tv2ForTextToText,
            SeamlessM4Tv2Model,
            SeamlessM4Tv2PreTrainedModel,
        )
        # 如果torch可用，则从模型文件导入特定的类和映射

else:
    import sys
    # 否则，导入sys模块

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
    # 将当前模块注册为懒加载模块，以便在需要时导入指定的结构
```