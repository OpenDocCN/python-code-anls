# `.\models\luke\__init__.py`

```
# 版权声明和许可信息，指出本代码的版权和使用许可
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

# 引入类型检查模块
from typing import TYPE_CHECKING

# 引入自定义工具模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块导入结构
_import_structure = {
    "configuration_luke": ["LUKE_PRETRAINED_CONFIG_ARCHIVE_MAP", "LukeConfig"],
    "tokenization_luke": ["LukeTokenizer"],
}

# 检查是否存在 Torch 库
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则添加以下模块到导入结构中
    _import_structure["modeling_luke"] = [
        "LUKE_PRETRAINED_MODEL_ARCHIVE_LIST",
        "LukeForEntityClassification",
        "LukeForEntityPairClassification",
        "LukeForEntitySpanClassification",
        "LukeForMultipleChoice",
        "LukeForQuestionAnswering",
        "LukeForSequenceClassification",
        "LukeForTokenClassification",
        "LukeForMaskedLM",
        "LukeModel",
        "LukePreTrainedModel",
    ]


# 如果是类型检查模式
if TYPE_CHECKING:
    # 从具体模块导入所需内容
    from .configuration_luke import LUKE_PRETRAINED_CONFIG_ARCHIVE_MAP, LukeConfig
    from .tokenization_luke import LukeTokenizer

    # 再次检查 Torch 库的可用性
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 Torch 可用，则从 modeling_luke 模块导入以下内容
        from .modeling_luke import (
            LUKE_PRETRAINED_MODEL_ARCHIVE_LIST,
            LukeForEntityClassification,
            LukeForEntityPairClassification,
            LukeForEntitySpanClassification,
            LukeForMaskedLM,
            LukeForMultipleChoice,
            LukeForQuestionAnswering,
            LukeForSequenceClassification,
            LukeForTokenClassification,
            LukeModel,
            LukePreTrainedModel,
        )

# 如果不是类型检查模式
else:
    # 导入 sys 模块
    import sys

    # 将当前模块替换为 LazyModule 类的实例，以延迟导入模块内容
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```