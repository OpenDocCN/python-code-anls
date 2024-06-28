# `.\models\yoso\__init__.py`

```
# 版权声明及许可证信息，指明此文件的版权及使用许可
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

# 从 typing 模块导入 TYPE_CHECKING 类型检查工具
from typing import TYPE_CHECKING

# 从 ...utils 中导入相关模块和函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义模块的导入结构字典
_import_structure = {"configuration_yoso": ["YOSO_PRETRAINED_CONFIG_ARCHIVE_MAP", "YosoConfig"]}

# 尝试检查是否有 torch 可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 可用，则添加 modeling_yoso 模块到导入结构字典中
    _import_structure["modeling_yoso"] = [
        "YOSO_PRETRAINED_MODEL_ARCHIVE_LIST",
        "YosoForMaskedLM",
        "YosoForMultipleChoice",
        "YosoForQuestionAnswering",
        "YosoForSequenceClassification",
        "YosoForTokenClassification",
        "YosoLayer",
        "YosoModel",
        "YosoPreTrainedModel",
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 从 .configuration_yoso 中导入 YOSO_PRETRAINED_CONFIG_ARCHIVE_MAP 和 YosoConfig 类
    from .configuration_yoso import YOSO_PRETRAINED_CONFIG_ARCHIVE_MAP, YosoConfig

    # 尝试检查是否有 torch 可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 torch 可用，则从 .modeling_yoso 中导入相关模块和类
        from .modeling_yoso import (
            YOSO_PRETRAINED_MODEL_ARCHIVE_LIST,
            YosoForMaskedLM,
            YosoForMultipleChoice,
            YosoForQuestionAnswering,
            YosoForSequenceClassification,
            YosoForTokenClassification,
            YosoLayer,
            YosoModel,
            YosoPreTrainedModel,
        )

# 如果不是类型检查模式
else:
    # 导入 sys 模块
    import sys

    # 使用 _LazyModule 定义延迟加载的模块，并将其指定给当前模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```