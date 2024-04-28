# `.\models\groupvit\__init__.py`

```py
# 版权声明和许可证信息
# 版权归 The HuggingFace Team 所有，保留所有权利。
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证，否则不得使用此文件
# 可以在以下链接获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 根据适用法律或书面同意，软件分发基于“原样”基础，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的具体语言

# 导入必要的模块和函数
from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_torch_available

# 定义模块导入结构
_import_structure = {
    "configuration_groupvit": [
        "GROUPVIT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "GroupViTConfig",
        "GroupViTOnnxConfig",
        "GroupViTTextConfig",
        "GroupViTVisionConfig",
    ],
}

# 检查是否存在 torch 库，如果不存在则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 torch 库，则添加相关模型到导入结构中
    _import_structure["modeling_groupvit"] = [
        "GROUPVIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GroupViTModel",
        "GroupViTPreTrainedModel",
        "GroupViTTextModel",
        "GroupViTVisionModel",
    ]

# 检查是否存在 tensorflow 库，如果不存在则引发异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 tensorflow 库，则添加相关模型到导入结构中
    _import_structure["modeling_tf_groupvit"] = [
        "TF_GROUPVIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFGroupViTModel",
        "TFGroupViTPreTrainedModel",
        "TFGroupViTTextModel",
        "TFGroupViTVisionModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入配置相关的内容
    from .configuration_groupvit import (
        GROUPVIT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        GroupViTConfig,
        GroupViTOnnxConfig,
        GroupViTTextConfig,
        GroupViTVisionConfig,
    )

    # 检查是否存在 torch 库，如果不存在则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 torch 模型相关的内容
        from .modeling_groupvit import (
            GROUPVIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            GroupViTModel,
            GroupViTPreTrainedModel,
            GroupViTTextModel,
            GroupViTVisionModel,
        )

    # 检查是否存在 tensorflow 库，如果不存在则引发异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 tensorflow 模型相关的内容
        from .modeling_tf_groupvit import (
            TF_GROUPVIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFGroupViTModel,
            TFGroupViTPreTrainedModel,
            TFGroupViTTextModel,
            TFGroupViTVisionModel,
        )

# 如果不是类型检查模式
else:
    import sys

    # 将当前模块设置为 LazyModule，延迟导入相关内容
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```