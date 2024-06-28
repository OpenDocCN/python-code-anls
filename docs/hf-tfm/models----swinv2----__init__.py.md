# `.\models\swinv2\__init__.py`

```
# 版权声明和许可证信息，指明代码版权归 HuggingFace Team 所有，使用 Apache License, Version 2.0 进行许可
# 如果不符合许可证要求，不能使用此文件中的代码
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

# 导入必要的类型检查模块
from typing import TYPE_CHECKING

# 导入必要的自定义异常和模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义导入结构，包含需要导入的模块和对象
_import_structure = {
    "configuration_swinv2": ["SWINV2_PRETRAINED_CONFIG_ARCHIVE_MAP", "Swinv2Config"],
}

# 检查是否存在 torch 库，如果不存在则抛出自定义异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 存在，添加额外的模块到导入结构中
    _import_structure["modeling_swinv2"] = [
        "SWINV2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Swinv2ForImageClassification",
        "Swinv2ForMaskedImageModeling",
        "Swinv2Model",
        "Swinv2PreTrainedModel",
        "Swinv2Backbone",
    ]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 导入配置和模型相关的对象，用于类型检查
    from .configuration_swinv2 import SWINV2_PRETRAINED_CONFIG_ARCHIVE_MAP, Swinv2Config

    # 再次检查 torch 库是否存在，如果不存在则抛出自定义异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关的对象，用于类型检查
        from .modeling_swinv2 import (
            SWINV2_PRETRAINED_MODEL_ARCHIVE_LIST,
            Swinv2Backbone,
            Swinv2ForImageClassification,
            Swinv2ForMaskedImageModeling,
            Swinv2Model,
            Swinv2PreTrainedModel,
        )

# 如果不是类型检查阶段
else:
    import sys

    # 使用 LazyModule 模式加载模块，延迟导入相关对象
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```