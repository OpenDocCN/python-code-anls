# `.\models\mvp\__init__.py`

```
# 版权声明和许可证信息，说明此代码的版权和使用许可
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

# 引入类型检查模块中的 TYPE_CHECKING 类型
from typing import TYPE_CHECKING

# 从 utils 模块中导入必要的工具和函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义模块的导入结构字典，包含不同模块和对应的导入内容列表
_import_structure = {
    "configuration_mvp": ["MVP_PRETRAINED_CONFIG_ARCHIVE_MAP", "MvpConfig", "MvpOnnxConfig"],
    "tokenization_mvp": ["MvpTokenizer"],
}

# 尝试导入 tokenizers_mvp_fast 模块，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_mvp_fast"] = ["MvpTokenizerFast"]

# 尝试导入 torch 模块，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 可用，则将 modeling_mvp 模块添加到导入结构中
    _import_structure["modeling_mvp"] = [
        "MVP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MvpForCausalLM",
        "MvpForConditionalGeneration",
        "MvpForQuestionAnswering",
        "MvpForSequenceClassification",
        "MvpModel",
        "MvpPreTrainedModel",
    ]

# 如果当前处于类型检查状态
if TYPE_CHECKING:
    # 导入配置相关的类和常量
    from .configuration_mvp import MVP_PRETRAINED_CONFIG_ARCHIVE_MAP, MvpConfig, MvpOnnxConfig
    # 导入 tokenizers 模块的相关类，如果不可用则忽略
    from .tokenization_mvp import MvpTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 tokenizers_mvp_fast 模块的快速 tokenizer 类，如果可用的话
        from .tokenization_mvp_fast import MvpTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 modeling_mvp 模块中的各种模型类，如果 torch 可用的话
        from .modeling_mvp import (
            MVP_PRETRAINED_MODEL_ARCHIVE_LIST,
            MvpForCausalLM,
            MvpForConditionalGeneration,
            MvpForQuestionAnswering,
            MvpForSequenceClassification,
            MvpModel,
            MvpPreTrainedModel,
        )

# 如果不处于类型检查状态，则将当前模块设置为 LazyModule 的代理模块
else:
    import sys

    # 将当前模块替换为 LazyModule，实现延迟导入
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```