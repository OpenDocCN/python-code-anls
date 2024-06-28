# `.\models\opt\__init__.py`

```py
# 版权声明和许可信息，声明代码版权和使用许可
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

# 引入必要的类型检查模块
from typing import TYPE_CHECKING

# 引入内部工具函数和异常类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构，用于延迟加载模块
_import_structure = {"configuration_opt": ["OPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "OPTConfig"]}

# 检查是否支持 Torch，若不支持则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果支持 Torch，则添加相关模型的导入结构
    _import_structure["modeling_opt"] = [
        "OPT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "OPTForCausalLM",
        "OPTModel",
        "OPTPreTrainedModel",
        "OPTForSequenceClassification",
        "OPTForQuestionAnswering",
    ]

# 检查是否支持 TensorFlow，若不支持则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果支持 TensorFlow，则添加相关模型的导入结构
    _import_structure["modeling_tf_opt"] = ["TFOPTForCausalLM", "TFOPTModel", "TFOPTPreTrainedModel"]

# 检查是否支持 Flax，若不支持则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果支持 Flax，则添加相关模型的导入结构
    _import_structure["modeling_flax_opt"] = [
        "FlaxOPTForCausalLM",
        "FlaxOPTModel",
        "FlaxOPTPreTrainedModel",
    ]

# 如果是类型检查模式，则进行额外的导入
if TYPE_CHECKING:
    # 导入配置相关的内容
    from .configuration_opt import OPT_PRETRAINED_CONFIG_ARCHIVE_MAP, OPTConfig

    try:
        # 检查是否支持 Torch，若不支持则跳过
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果支持 Torch，则导入 Torch 相关的模型
        from .modeling_opt import (
            OPT_PRETRAINED_MODEL_ARCHIVE_LIST,
            OPTForCausalLM,
            OPTForQuestionAnswering,
            OPTForSequenceClassification,
            OPTModel,
            OPTPreTrainedModel,
        )

    try:
        # 检查是否支持 TensorFlow，若不支持则跳过
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果支持 TensorFlow，则导入 TensorFlow 相关的模型
        from .modeling_tf_opt import TFOPTForCausalLM, TFOPTModel, TFOPTPreTrainedModel

    try:
        # 检查是否支持 Flax，若不支持则跳过
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果支持 Flax，则导入 Flax 相关的模型
        from .modeling_flax_opt import FlaxOPTForCausalLM, FlaxOPTModel, FlaxOPTPreTrainedModel

else:
    # 如果不是类型检查模式，则使用 LazyModule 进行延迟加载
    import sys

    # 将当前模块替换为 LazyModule 实例，实现延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```