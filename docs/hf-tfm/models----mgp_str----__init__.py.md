# `.\models\mgp_str\__init__.py`

```
# flake8: noqa
# 由于在此模块中无法忽略 "F401 '...' imported but unused" 警告，但要保留其他警告。
# 因此，完全不检查这个模块。

# Copyright 2023 The HuggingFace Team. All rights reserved.
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

# 定义模块结构的导入方式和依赖关系
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 模块导入结构字典，指定各模块和其对应的导入内容列表
_import_structure = {
    "configuration_mgp_str": ["MGP_STR_PRETRAINED_CONFIG_ARCHIVE_MAP", "MgpstrConfig"],
    "processing_mgp_str": ["MgpstrProcessor"],
    "tokenization_mgp_str": ["MgpstrTokenizer"],
}

# 检查是否有torch可用，若不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果torch可用，则增加额外的模块导入信息到_import_structure字典中
    _import_structure["modeling_mgp_str"] = [
        "MGP_STR_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MgpstrModel",
        "MgpstrPreTrainedModel",
        "MgpstrForSceneTextRecognition",
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 从对应模块中导入特定的类或变量
    from .configuration_mgp_str import MGP_STR_PRETRAINED_CONFIG_ARCHIVE_MAP, MgpstrConfig
    from .processing_mgp_str import MgpstrProcessor
    from .tokenization_mgp_str import MgpstrTokenizer

    # 再次检查torch是否可用，若不可用则抛出异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果torch可用，则从模型相关模块中导入特定的类或变量
        from .modeling_mgp_str import (
            MGP_STR_PRETRAINED_MODEL_ARCHIVE_LIST,
            MgpstrForSceneTextRecognition,
            MgpstrModel,
            MgpstrPreTrainedModel,
        )
else:
    # 如果不在类型检查模式下，则将当前模块设置为懒加载模块
    import sys

    # 使用_LazyModule类封装当前模块，以实现延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```