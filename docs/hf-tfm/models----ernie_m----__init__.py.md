# `.\models\ernie_m\__init__.py`

```py
# 版权声明和保留声明，声明了代码版权及许可协议
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

# 导入异常：当依赖项不可用时引发异常
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_sentencepiece_available, is_torch_available

# 定义导入结构的字典
_import_structure = {
    "configuration_ernie_m": ["ERNIE_M_PRETRAINED_CONFIG_ARCHIVE_MAP", "ErnieMConfig"],
}

# 检查是否有句子分割器可用，若不可用则引发异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 将 tokenization_ernie_m 模块添加到导入结构中
    _import_structure["tokenization_ernie_m"] = ["ErnieMTokenizer"]

# 检查是否有 Torch 可用，若不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 将 modeling_ernie_m 模块添加到导入结构中
    _import_structure["modeling_ernie_m"] = [
        "ERNIE_M_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ErnieMForMultipleChoice",
        "ErnieMForQuestionAnswering",
        "ErnieMForSequenceClassification",
        "ErnieMForTokenClassification",
        "ErnieMModel",
        "ErnieMPreTrainedModel",
        "ErnieMForInformationExtraction",
    ]

# 如果是类型检查模式，则进行额外的导入
if TYPE_CHECKING:
    # 导入配置和配置类
    from .configuration_ernie_m import ERNIE_M_PRETRAINED_CONFIG_ARCHIVE_MAP, ErnieMConfig

    try:
        # 再次检查是否有句子分割器可用，若不可用则跳过
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 ErnieMTokenizer 类
        from .tokenization_ernie_m import ErnieMTokenizer

    try:
        # 再次检查是否有 Torch 可用，若不可用则跳过
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 modeling_ernie_m 中的多个类和常量
        from .modeling_ernie_m import (
            ERNIE_M_PRETRAINED_MODEL_ARCHIVE_LIST,
            ErnieMForInformationExtraction,
            ErnieMForMultipleChoice,
            ErnieMForQuestionAnswering,
            ErnieMForSequenceClassification,
            ErnieMForTokenClassification,
            ErnieMModel,
            ErnieMPreTrainedModel,
        )

# 若非类型检查模式，则创建懒加载模块
else:
    import sys

    # 使用 _LazyModule 创建模块，提供导入结构和模块规范
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```