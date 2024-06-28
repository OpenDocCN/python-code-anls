# `.\models\groupvit\__init__.py`

```
# 版权声明和许可证声明，指明此代码的版权和使用许可
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License 2.0 版本许可，除非遵循该许可，否则不得使用此文件。
# You may obtain a copy of the License at
# 你可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# 除非法律要求或书面同意，否则软件
# distributed under the License is distributed on an "AS IS" BASIS,
# 依据许可证的“原样”分发。
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 没有任何形式的明示或暗示保证或条件。
# See the License for the specific language governing permissions and
# 详细信息，请参阅特定语言的许可证。
from typing import TYPE_CHECKING

# 从工具包引入相关模块和函数，检查是否可以导入相关依赖项
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_torch_available

# 定义导入结构字典，用于延迟加载模块
_import_structure = {
    "configuration_groupvit": [
        "GROUPVIT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "GroupViTConfig",
        "GroupViTOnnxConfig",
        "GroupViTTextConfig",
        "GroupViTVisionConfig",
    ],
}

# 检查是否可以导入 torch，如果不行，则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果能导入 torch，则加入模型相关的配置和模型定义到导入结构中
    _import_structure["modeling_groupvit"] = [
        "GROUPVIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GroupViTModel",
        "GroupViTPreTrainedModel",
        "GroupViTTextModel",
        "GroupViTVisionModel",
    ]

# 检查是否可以导入 tensorflow，如果不行，则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果能导入 tensorflow，则加入 tensorflow 模型相关的配置和模型定义到导入结构中
    _import_structure["modeling_tf_groupvit"] = [
        "TF_GROUPVIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFGroupViTModel",
        "TFGroupViTPreTrainedModel",
        "TFGroupViTTextModel",
        "TFGroupViTVisionModel",
    ]

# 如果是类型检查模式，导入具体的配置和模型类
if TYPE_CHECKING:
    from .configuration_groupvit import (
        GROUPVIT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        GroupViTConfig,
        GroupViTOnnxConfig,
        GroupViTTextConfig,
        GroupViTVisionConfig,
    )

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_groupvit import (
            GROUPVIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            GroupViTModel,
            GroupViTPreTrainedModel,
            GroupViTTextModel,
            GroupViTVisionModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_groupvit import (
            TF_GROUPVIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFGroupViTModel,
            TFGroupViTPreTrainedModel,
            TFGroupViTTextModel,
            TFGroupViTVisionModel,
        )

else:
    # 如果不是类型检查模式，使用 LazyModule 进行模块的延迟加载
    import sys

    # 将当前模块映射到 LazyModule，以支持延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```