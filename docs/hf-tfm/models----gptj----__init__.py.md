# `.\models\gptj\__init__.py`

```py
# 版权声明和许可信息
# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
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

# 引入类型检查
from typing import TYPE_CHECKING

# 引入必要的模块和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)

# 定义模块的导入结构，用于延迟加载模块
_import_structure = {"configuration_gptj": ["GPTJ_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPTJConfig", "GPTJOnnxConfig"]}

# 检查是否支持 torch 库，如果不支持则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果支持 torch 库，则添加 torch 下相关模块到导入结构
    _import_structure["modeling_gptj"] = [
        "GPTJ_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GPTJForCausalLM",
        "GPTJForQuestionAnswering",
        "GPTJForSequenceClassification",
        "GPTJModel",
        "GPTJPreTrainedModel",
    ]

# 检查是否支持 tensorflow 库，如果不支持则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果支持 tensorflow 库，则添加 tensorflow 下相关模块到导入结构
    _import_structure["modeling_tf_gptj"] = [
        "TFGPTJForCausalLM",
        "TFGPTJForQuestionAnswering",
        "TFGPTJForSequenceClassification",
        "TFGPTJModel",
        "TFGPTJPreTrainedModel",
    ]

# 检查是否支持 flax 库，如果不支持则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果支持 flax 库，则添加 flax 下相关模块到导入结构
    _import_structure["modeling_flax_gptj"] = [
        "FlaxGPTJForCausalLM",
        "FlaxGPTJModel",
        "FlaxGPTJPreTrainedModel",
    ]

# 如果是类型检查模式，执行以下导入
if TYPE_CHECKING:
    # 从相应模块导入配置和模型类
    from .configuration_gptj import GPTJ_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTJConfig, GPTJOnnxConfig

    try:
        # 检查是否支持 torch 库，如果不支持则抛出 OptionalDependencyNotAvailable 异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果支持 torch 库，则从 modeling_gptj 模块中导入相关类
        from .modeling_gptj import (
            GPTJ_PRETRAINED_MODEL_ARCHIVE_LIST,
            GPTJForCausalLM,
            GPTJForQuestionAnswering,
            GPTJForSequenceClassification,
            GPTJModel,
            GPTJPreTrainedModel,
        )

    try:
        # 检查是否支持 tensorflow 库，如果不支持则抛出 OptionalDependencyNotAvailable 异常
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果支持 tensorflow 库，则从 modeling_tf_gptj 模块中导入相关类
        from .modeling_tf_gptj import (
            TFGPTJForCausalLM,
            TFGPTJForQuestionAnswering,
            TFGPTJForSequenceClassification,
            TFGPTJModel,
            TFGPTJPreTrainedModel,
        )

    try:
        # 检查是否支持 flax 库，如果不支持则抛出 OptionalDependencyNotAvailable 异常
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果支持 flax 库，则从 modeling_flax_gptj 模块中导入相关类
        from .modeling_flax_gptj import (
            FlaxGPTJForCausalLM,
            FlaxGPTJModel,
            FlaxGPTJPreTrainedModel,
        )
    # 捕获 OptionalDependencyNotAvailable 异常，如果发生则不做任何操作
    except OptionalDependencyNotAvailable:
        pass
    # 如果未发生异常，则导入以下模块
    else:
        from .modeling_flax_gptj import FlaxGPTJForCausalLM, FlaxGPTJModel, FlaxGPTJPreTrainedModel
else:
    # 导入 sys 模块，用于动态设置当前模块为懒加载模块
    import sys
    
    # 使用 sys.modules[__name__] 将当前模块注册为 _LazyModule 的实例，
    # __name__ 是当前模块的名称，__file__ 是当前模块的文件名，
    # _import_structure 是导入结构，module_spec=__spec__ 是模块规范
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```