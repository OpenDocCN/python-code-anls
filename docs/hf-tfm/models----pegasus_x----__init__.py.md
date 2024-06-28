# `.\models\pegasus_x\__init__.py`

```
# 版权声明和版权许可信息，标识本代码版权归 HuggingFace 团队所有，受 Apache License, Version 2.0 许可
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

# 导入类型检查模块中的 TYPE_CHECKING 类型
from typing import TYPE_CHECKING

# 从 utils 中导入 OptionalDependencyNotAvailable、_LazyModule 和 is_torch_available 函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块导入结构字典 _import_structure
_import_structure = {
    "configuration_pegasus_x": ["PEGASUS_X_PRETRAINED_CONFIG_ARCHIVE_MAP", "PegasusXConfig"],
}

# 检查是否 Torch 可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若 Torch 可用，则扩展 _import_structure 字典，导入 modeling_pegasus_x 模块中的类和变量
    _import_structure["modeling_pegasus_x"] = [
        "PEGASUS_X_PRETRAINED_MODEL_ARCHIVE_LIST",
        "PegasusXForConditionalGeneration",
        "PegasusXModel",
        "PegasusXPreTrainedModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 从 configuration_pegasus_x 模块中导入 PEGASUS_X_PRETRAINED_CONFIG_ARCHIVE_MAP 和 PegasusXConfig 类
    from .configuration_pegasus_x import PEGASUS_X_PRETRAINED_CONFIG_ARCHIVE_MAP, PegasusXConfig

    # 检查 Torch 是否可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若 Torch 可用，则从 modeling_pegasus_x 模块中导入相关类和变量
        from .modeling_pegasus_x import (
            PEGASUS_X_PRETRAINED_MODEL_ARCHIVE_LIST,
            PegasusXForConditionalGeneration,
            PegasusXModel,
            PegasusXPreTrainedModel,
        )

# 如果不是类型检查模式
else:
    # 导入 sys 模块
    import sys

    # 将当前模块替换为 LazyModule 对象，延迟加载相关模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```