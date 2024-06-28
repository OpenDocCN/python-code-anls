# `.\models\table_transformer\__init__.py`

```py
# 版权声明和许可证信息
# Copyright 2022 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 引入类型检查模块
from typing import TYPE_CHECKING

# 引入自定义的异常和模块延迟加载工具
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块导入结构
_import_structure = {
    "configuration_table_transformer": [
        "TABLE_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "TableTransformerConfig",
        "TableTransformerOnnxConfig",
    ]
}

# 检查是否存在 torch 库，如果不存在则引发自定义的异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 存在，扩展导入结构以包含额外的模块
    _import_structure["modeling_table_transformer"] = [
        "TABLE_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TableTransformerForObjectDetection",
        "TableTransformerModel",
        "TableTransformerPreTrainedModel",
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 从配置模块中导入特定的类和常量
    from .configuration_table_transformer import (
        TABLE_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TableTransformerConfig,
        TableTransformerOnnxConfig,
    )

    # 再次检查 torch 是否可用，如果不可用则捕获异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 torch 可用，则从建模模块中导入特定的类和常量
        from .modeling_table_transformer import (
            TABLE_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TableTransformerForObjectDetection,
            TableTransformerModel,
            TableTransformerPreTrainedModel,
        )

# 如果不是类型检查模式
else:
    # 导入 sys 模块
    import sys

    # 将当前模块注册为一个延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```