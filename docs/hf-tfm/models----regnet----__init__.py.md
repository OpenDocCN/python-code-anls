# `.\models\regnet\__init__.py`

```
# 版权声明和许可证信息
#
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

# 导入类型检查相关模块
from typing import TYPE_CHECKING

# 导入必要的依赖和模块
# 引入了一些特定的异常和工具函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)

# 定义了一个导入结构的字典，包含了模块和其对应的导入内容
_import_structure = {"configuration_regnet": ["REGNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "RegNetConfig"]}

# 检查是否可用 Torch，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 Torch 模型相关的导入内容到导入结构字典中
    _import_structure["modeling_regnet"] = [
        "REGNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "RegNetForImageClassification",
        "RegNetModel",
        "RegNetPreTrainedModel",
    ]

# 类似地检查 TensorFlow 的可用性，并添加相应的导入内容到导入结构字典中
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_regnet"] = [
        "TF_REGNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFRegNetForImageClassification",
        "TFRegNetModel",
        "TFRegNetPreTrainedModel",
    ]

# 类似地检查 Flax 的可用性，并添加相应的导入内容到导入结构字典中
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_regnet"] = [
        "FlaxRegNetForImageClassification",
        "FlaxRegNetModel",
        "FlaxRegNetPreTrainedModel",
    ]

# 如果是类型检查阶段，则导入更多的内容以支持类型检查
if TYPE_CHECKING:
    from .configuration_regnet import REGNET_PRETRAINED_CONFIG_ARCHIVE_MAP, RegNetConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_regnet import (
            REGNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            RegNetForImageClassification,
            RegNetModel,
            RegNetPreTrainedModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_regnet import (
            TF_REGNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFRegNetForImageClassification,
            TFRegNetModel,
            TFRegNetPreTrainedModel,
        )

    # Flax 在类型检查中的导入暂时略过，因为前面已经处理过了
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果前面的条件不满足，则执行以下代码块
        # 从当前目录下的 `modeling_flax_regnet` 模块中导入以下三个类
        from .modeling_flax_regnet import (
            FlaxRegNetForImageClassification,
            FlaxRegNetModel,
            FlaxRegNetPreTrainedModel,
        )
else:
    # 如果条件不满足，导入 sys 模块
    import sys
    # 将当前模块替换为一个懒加载模块，传入当前模块名、文件路径和导入结构
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```