# `.\models\xlm_prophetnet\__init__.py`

```py
# 版权声明及许可证信息，声明代码版权及授权许可
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

# 导入自定义的异常和模块延迟加载工具
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_sentencepiece_available, is_torch_available

# 定义模块导入结构字典，包含一些模块及其相关的导入
_import_structure = {
    "configuration_xlm_prophetnet": ["XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "XLMProphetNetConfig"],
}

# 检查是否存在 sentencepiece 库，若不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若存在 sentencepiece 库，则添加 tokenization_xlm_prophetnet 模块到导入结构中
    _import_structure["tokenization_xlm_prophetnet"] = ["XLMProphetNetTokenizer"]

# 检查是否存在 torch 库，若不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若存在 torch 库，则添加 modeling_xlm_prophetnet 模块到导入结构中
    _import_structure["modeling_xlm_prophetnet"] = [
        "XLM_PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "XLMProphetNetDecoder",
        "XLMProphetNetEncoder",
        "XLMProphetNetForCausalLM",
        "XLMProphetNetForConditionalGeneration",
        "XLMProphetNetModel",
        "XLMProphetNetPreTrainedModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 从 configuration_xlm_prophetnet 模块导入特定类和变量
    from .configuration_xlm_prophetnet import XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMProphetNetConfig

    try:
        # 再次检查 sentencepiece 库是否存在
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若存在 sentencepiece 库，则从 tokenization_xlm_prophetnet 模块导入特定类
        from .tokenization_xlm_prophetnet import XLMProphetNetTokenizer

    try:
        # 再次检查 torch 库是否存在
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若存在 torch 库，则从 modeling_xlm_prophetnet 模块导入特定类和变量
        from .modeling_xlm_prophetnet import (
            XLM_PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            XLMProphetNetDecoder,
            XLMProphetNetEncoder,
            XLMProphetNetForCausalLM,
            XLMProphetNetForConditionalGeneration,
            XLMProphetNetModel,
            XLMProphetNetPreTrainedModel,
        )

# 如果不是类型检查模式
else:
    # 导入 sys 模块
    import sys

    # 使用延迟加载模块，将当前模块设置为 LazyModule 类型
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```