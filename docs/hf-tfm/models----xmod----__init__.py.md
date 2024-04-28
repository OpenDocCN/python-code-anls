# `.\transformers\models\xmod\__init__.py`

```
# flake8: noqa
# 忽略 flake8 检查，避免"导入但未使用"警告，同时保留其他警告

# 版权声明，声明版权和许可
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

# 导入类型检查相关的模块
from typing import TYPE_CHECKING

# 导入 OptionalDependencyNotAvailable 异常和 _LazyModule 模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义导入结构
_import_structure = {
    "configuration_xmod": [
        "XMOD_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "XmodConfig",
        "XmodOnnxConfig",
    ],
}

# 检查是否有 torch 库可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 可用，则添加 modeling_xmod 模块到导入结构中
    _import_structure["modeling_xmod"] = [
        "XMOD_PRETRAINED_MODEL_ARCHIVE_LIST",
        "XmodForCausalLM",
        "XmodForMaskedLM",
        "XmodForMultipleChoice",
        "XmodForQuestionAnswering",
        "XmodForSequenceClassification",
        "XmodForTokenClassification",
        "XmodModel",
        "XmodPreTrainedModel",
    ]

# 如果类型检查开启
if TYPE_CHECKING:
    # 导入 configuration_xmod 模块中的类和常量
    from .configuration_xmod import XMOD_PRETRAINED_CONFIG_ARCHIVE_MAP, XmodConfig, XmodOnnxConfig

    # 检查是否有 torch 库可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 torch 可用，则导入 modeling_xmod 模块中的类和常量
        from .modeling_xmod import (
            XMOD_PRETRAINED_MODEL_ARCHIVE_LIST,
            XmodForCausalLM,
            XmodForMaskedLM,
            XmodForMultipleChoice,
            XmodForQuestionAnswering,
            XmodForSequenceClassification,
            XmodForTokenClassification,
            XmodModel,
            XmodPreTrainedModel,
        )

# 如果类型检查未开启
else:
    # 导入 sys 模块
    import sys

    # 将当前模块替换为懒加载模块 _LazyModule，确保按需导入
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```