# `.\models\mega\__init__.py`

```
# Copyright 2023 The HuggingFace Team. All rights reserved.
# 版权声明及许可信息，指明该代码的版权归属及使用许可
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache 许可证版本 2.0 进行许可
# you may not use this file except in compliance with the License.
# 除非符合许可证的条件，否则不得使用此文件
# You may obtain a copy of the License at
# 可以在以下网址获得许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 除非法律有明确规定或书面同意，否则按"原样"分发
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 没有任何形式的明示或暗示的保证和条件
# See the License for the specific language governing permissions and
# 请查阅许可证了解具体的语言授权条款及限制。
# limitations under the License.

from typing import TYPE_CHECKING
# 引入类型检查模块

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)
# 从相对路径中引入必要的工具模块及函数

_import_structure = {
    "configuration_mega": ["MEGA_PRETRAINED_CONFIG_ARCHIVE_MAP", "MegaConfig", "MegaOnnxConfig"],
}
# 定义一个字典，包含 Mega 模块的配置信息

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
# 检查是否存在 Torch 库，如果不存在则引发异常
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_mega"] = [
        "MEGA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MegaForCausalLM",
        "MegaForMaskedLM",
        "MegaForMultipleChoice",
        "MegaForQuestionAnswering",
        "MegaForSequenceClassification",
        "MegaForTokenClassification",
        "MegaModel",
        "MegaPreTrainedModel",
    ]
    # 如果 Torch 存在，则将 Mega 模块的建模信息添加到导入结构中

if TYPE_CHECKING:
    from .configuration_mega import MEGA_PRETRAINED_CONFIG_ARCHIVE_MAP, MegaConfig, MegaOnnxConfig
    # 如果在类型检查模式下，从配置模块导入配置映射和配置类

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_mega import (
            MEGA_PRETRAINED_MODEL_ARCHIVE_LIST,
            MegaForCausalLM,
            MegaForMaskedLM,
            MegaForMultipleChoice,
            MegaForQuestionAnswering,
            MegaForSequenceClassification,
            MegaForTokenClassification,
            MegaModel,
            MegaPreTrainedModel,
        )
        # 如果 Torch 存在，从建模模块导入 Mega 模块的各个模型类

else:
    import sys
    # 导入 sys 模块

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
    # 如果不是类型检查模式，将当前模块替换为懒加载模块，实现按需导入
```