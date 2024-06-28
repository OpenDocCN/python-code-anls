# `.\models\bloom\__init__.py`

```
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

from ...utils import (
    # 捕获未安装的依赖包的异常
    OptionalDependencyNotAvailable,
    _LazyModule,
    是Flax可获取的,
    是Tokenizers可获取的,
    是Torch可获取的,
)

_import_structure = {
    # 导入BLOOM配置相关结构
    "configuration_bloom": ["BLOOM_PRETRAINED_CONFIG_ARCHIVE_MAP", "BloomConfig", "BloomOnnxConfig"],
}

# 填充已存在的受限包可用性检查后
# 尝试导入和验证是否 Tokenizers 依赖可用
try:
    如果 not 是Tokenizers可获取的():
        抛出OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # NaN 接下来继续执行不导入Tokenizers相关结构的代码
    pass
else:
    _import_structure["tokenization_bloom_fast"] = ["BloomTokenizerFast"]

# 填充已存在的受限包可用性检查后
# 尝试导入和验证Torch是否可用
try:
    If not 是Torch可获取的():
        抛出OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # NaN 接下来继续执行不导入PyTorch相关结构的代码
    pass
else:
    _import_structure["modeling_bloom"] = [
        "BLOOM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BloomForCausalLM",
        "BloomModel",
        "BloomPreTrainedModel",
        "BloomForSequenceClassification",
        "BloomForTokenClassification",
        "BloomForQuestionAnswering",
    ]

# 填充已存在的受限包可用性检查后
# 尝试导入和验证是否Flax可用
try:
    If not 是Flax可获取的():
        抛出OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # NaN 接下来继续执行不导入Flax相关结构的代码
    pass
else:
    _import_structure["modeling_flax_bloom"] = [
        "FlaxBloomForCausalLM",
        "FlaxBloomModel",
        "FlaxBloomPreTrainedModel",
    ]

# 如果代码在模式检查中（例如，导入结构是静态的）
如果 "检查类型":
    从 .configuration_bloom 导入 BLOOM_PRETRAINED_CONFIG_ARCHIVE_MAP, BloomConfig, BloomOnnxConfig

    # 尝试导入 Tokenizers 相关结构的另一个方法
    try:
        If not 是Tokenizers可获取的():
            抛出OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        从 .tokenization_bloom_fast 导入 BloomTokenizerFast

    # 尝试导入 PyTorch 相关结构
    try:
        If not 是Torch可获取的():
            抛出OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        从 .modeling_bloom 导入 (
            BLOOM_PRETRAINED_MODEL_ARCHIVE_LIST,
            BloomForCausalLM,
            BloomForQuestionAnswering,
            BloomForSequenceClassification,
            BloomForTokenClassification,
            BloomModel,
            BloomPreTrainedModel,
        )

    # 尝试导入 Flax 相关结构
    try:
        If not 是Flax可获取的():
            抛出OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        从 .modeling_flax_bloom 导入 (
            FlaxBloomForCausalLM,
            FlaxBloomModel,
            FlaxBloomPreTrainedModel,
        )

else:
    import 系统 as 系统
    # 实例化一个占位符类，用于当代码非类型检查模式时工作
_导入_structure = _懒模块(lambda: _导入_structure(), 属性("__version__"))
    # 将当前模块注册到 sys.modules 中，使用 _LazyModule 进行延迟加载
    # 设置模块名为当前模块的名字，文件路径为当前模块的文件路径
    # 使用 globals()["__file__"] 获取当前模块的文件路径作为参数传递给 _LazyModule
    # 使用 _import_structure 指定模块的导入结构
    # 将模块规范 (__spec__) 作为参数传递给 _LazyModule
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```