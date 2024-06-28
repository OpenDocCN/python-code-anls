# `.\models\align\__init__.py`

```py
# 版权声明和版权信息
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

# 引入类型检查模块
from typing import TYPE_CHECKING

# 从utils中导入必要的异常和模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_align": [
        "ALIGN_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "AlignConfig",
        "AlignTextConfig",
        "AlignVisionConfig",
    ],
    "processing_align": ["AlignProcessor"],
}

# 检查是否Torch可用，如果不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果Torch可用，则添加以下模型相关的导入结构
    _import_structure["modeling_align"] = [
        "ALIGN_PRETRAINED_MODEL_ARCHIVE_LIST",
        "AlignModel",
        "AlignPreTrainedModel",
        "AlignTextModel",
        "AlignVisionModel",
    ]

# 如果是类型检查阶段，导入特定的配置和模型类
if TYPE_CHECKING:
    from .configuration_align import (
        ALIGN_PRETRAINED_CONFIG_ARCHIVE_MAP,
        AlignConfig,
        AlignTextConfig,
        AlignVisionConfig,
    )
    from .processing_align import AlignProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_align import (
            ALIGN_PRETRAINED_MODEL_ARCHIVE_LIST,
            AlignModel,
            AlignPreTrainedModel,
            AlignTextModel,
            AlignVisionModel,
        )

# 如果不是类型检查阶段，则进行Lazy导入的设置
else:
    import sys

    # 将当前模块替换为LazyModule，使用Lazy加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```