# `.\models\decision_transformer\__init__.py`

```
# 版权声明和许可证信息，说明此文件的版权归HuggingFace团队所有，并遵循Apache License 2.0许可
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

# 引入TYPE_CHECKING用于静态类型检查
from typing import TYPE_CHECKING

# 从utils模块导入OptionalDependencyNotAvailable、_LazyModule和is_torch_available函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构，包括configuration_decision_transformer模块的部分内容
_import_structure = {
    "configuration_decision_transformer": [
        "DECISION_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "DecisionTransformerConfig",
    ],
}

# 检查是否torch可用，如果不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果torch可用，则扩展_import_structure添加modeling_decision_transformer模块的内容
    _import_structure["modeling_decision_transformer"] = [
        "DECISION_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DecisionTransformerGPT2Model",
        "DecisionTransformerGPT2PreTrainedModel",
        "DecisionTransformerModel",
        "DecisionTransformerPreTrainedModel",
    ]

# 如果正在进行类型检查
if TYPE_CHECKING:
    # 从configuration_decision_transformer模块导入特定内容，包括DECISION_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP和DecisionTransformerConfig
    from .configuration_decision_transformer import (
        DECISION_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DecisionTransformerConfig,
    )

    # 再次检查torch是否可用，如果不可用则跳过
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从modeling_decision_transformer模块导入特定内容，包括DECISION_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST和多个DecisionTransformer类
        from .modeling_decision_transformer import (
            DECISION_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            DecisionTransformerGPT2Model,
            DecisionTransformerGPT2PreTrainedModel,
            DecisionTransformerModel,
            DecisionTransformerPreTrainedModel,
        )

# 如果不是在进行类型检查
else:
    # 导入sys模块
    import sys

    # 将当前模块设置为_LazyModule，使用_LazyModule延迟加载模块内容
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```