# `.\models\deberta_v2\__init__.py`

```py
# 版权声明和许可证声明，指明代码的版权和许可条件
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache 许可证版本 2.0 授权使用此文件
# you may not use this file except in compliance with the License.
# 除非符合许可证的规定，否则不得使用本文件
# You may obtain a copy of the License at
# 获取许可证的副本地址
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 除非法律要求或书面同意，否则本软件按"原样"提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 没有明示或暗示的担保或条件
# See the License for the specific language governing permissions and
# 详见许可证，了解特定语言的授权信息
# limitations under the License.
# 许可证下的限制

from typing import TYPE_CHECKING

# 导入必要的依赖模块和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块导入结构
_import_structure = {
    "configuration_deberta_v2": ["DEBERTA_V2_PRETRAINED_CONFIG_ARCHIVE_MAP", "DebertaV2Config", "DebertaV2OnnxConfig"],
    "tokenization_deberta_v2": ["DebertaV2Tokenizer"],
}

# 检查 tokenizers 是否可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将 tokenization_deberta_v2_fast 模块添加到导入结构中
    _import_structure["tokenization_deberta_v2_fast"] = ["DebertaV2TokenizerFast"]

# 检查 TensorFlow 是否可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将 modeling_tf_deberta_v2 模块添加到导入结构中
    _import_structure["modeling_tf_deberta_v2"] = [
        "TF_DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFDebertaV2ForMaskedLM",
        "TFDebertaV2ForQuestionAnswering",
        "TFDebertaV2ForMultipleChoice",
        "TFDebertaV2ForSequenceClassification",
        "TFDebertaV2ForTokenClassification",
        "TFDebertaV2Model",
        "TFDebertaV2PreTrainedModel",
    ]

# 检查 PyTorch 是否可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将 modeling_deberta_v2 模块添加到导入结构中
    _import_structure["modeling_deberta_v2"] = [
        "DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DebertaV2ForMaskedLM",
        "DebertaV2ForMultipleChoice",
        "DebertaV2ForQuestionAnswering",
        "DebertaV2ForSequenceClassification",
        "DebertaV2ForTokenClassification",
        "DebertaV2Model",
        "DebertaV2PreTrainedModel",
    ]

# 如果是类型检查阶段，进行进一步的导入
if TYPE_CHECKING:
    # 导入配置相关的类和变量
    from .configuration_deberta_v2 import (
        DEBERTA_V2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DebertaV2Config,
        DebertaV2OnnxConfig,
    )
    # 导入 tokenizers 相关的类
    from .tokenization_deberta_v2 import DebertaV2Tokenizer

    # 检查 tokenizers 是否可用，若不可用则不导入
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，导入 tokenization_deberta_v2_fast 模块
        from .tokenization_deberta_v2_fast import DebertaV2TokenizerFast

    # 检查 TensorFlow 是否可用，若不可用则不导入
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果前面的条件不满足，则从当前目录下的.modeling_tf_deberta_v2模块中导入以下内容：
    from .modeling_tf_deberta_v2 import (
        TF_DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFDebertaV2ForMaskedLM,
        TFDebertaV2ForMultipleChoice,
        TFDebertaV2ForQuestionAnswering,
        TFDebertaV2ForSequenceClassification,
        TFDebertaV2ForTokenClassification,
        TFDebertaV2Model,
        TFDebertaV2PreTrainedModel,
    )

try:
    # 尝试检查是否有torch库可用，如果不可用则引发OptionalDependencyNotAvailable异常
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果OptionalDependencyNotAvailable异常被引发，什么也不做，直接跳过
    pass
else:
    # 如果上面的try块未引发异常，则从当前目录下的.modeling_deberta_v2模块中导入以下内容：
    from .modeling_deberta_v2 import (
        DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST,
        DebertaV2ForMaskedLM,
        DebertaV2ForMultipleChoice,
        DebertaV2ForQuestionAnswering,
        DebertaV2ForSequenceClassification,
        DebertaV2ForTokenClassification,
        DebertaV2Model,
        DebertaV2PreTrainedModel,
    )
else:
    # 导入 sys 模块，用于操作 Python 解释器的系统功能
    import sys
    
    # 将当前模块注册到 sys.modules 中，使用 _LazyModule 进行延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```