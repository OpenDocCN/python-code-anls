# `.\models\ernie_m\__init__.py`

```
# 版权声明及许可证信息
# Copyright 2023 The HuggingFace and Baidu Team. All rights reserved.
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

# 引入延迟加载模块和一些依赖检查函数
# 使用了相对路径引用
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_sentencepiece_available, is_torch_available

# 定义模块导入结构
_import_structure = {
    "configuration_ernie_m": ["ERNIE_M_PRETRAINED_CONFIG_ARCHIVE_MAP", "ErnieMConfig"],
}

# 检查是否存在 sentencepiece 库，若不存在则引发异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass  # 如果异常发生，则忽略
else:
    _import_structure["tokenization_ernie_m"] = ["ErnieMTokenizer"]  # 导入 ErnieMTokenizer 模块

# 检查是否存在 torch 库，若不存在则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass  # 如果异常发生，则忽略
else:
    # 导入多个与 ERNIE-M 模型相关的模块
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

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 导入配置模块及相关类
    from .configuration_ernie_m import ERNIE_M_PRETRAINED_CONFIG_ARCHIVE_MAP, ErnieMConfig

    # 检查是否存在 sentencepiece 库，若不存在则引发异常
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass  # 如果异常发生，则忽略
    else:
        # 导入 tokenization_ernie_m 模块
        from .tokenization_ernie_m import ErnieMTokenizer

    # 检查是否存在 torch 库，若不存在则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass  # 如果异常发生，则忽略
    else:
        # 导入 modeling_ernie_m 模块及相关类
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

# 如果不是类型检查阶段
else:
    import sys

    # 将当前模块替换为懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```  
```