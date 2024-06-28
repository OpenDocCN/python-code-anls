# `.\models\cpmant\__init__.py`

```py
# flake8: noqa
# 禁止 flake8 对当前模块执行检查，以避免 "F401 '...' imported but unused" 警告。

# Copyright 2022 The HuggingFace Team and The OpenBMB Team. All rights reserved.
# 版权声明，保留所有权利。

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache 许可证 2.0 版本授权。

# you may not use this file except in compliance with the License.
# 除非符合许可证要求，否则不得使用本文件。

# You may obtain a copy of the License at
# 您可以在以下网址获取许可证副本：

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# 根据适用法律或书面同意，软件

# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 在 "AS IS" 基础上分发，不提供任何担保或条件，无论是明示的还是隐含的。

# See the License for the specific language governing permissions and
# limitations under the License.
# 请查阅许可证了解具体的语言授权和限制。

from typing import TYPE_CHECKING

# rely on isort to merge the imports
# 使用 isort 来合并导入项

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

_import_structure = {
    "configuration_cpmant": ["CPMANT_PRETRAINED_CONFIG_ARCHIVE_MAP", "CpmAntConfig"],
    "tokenization_cpmant": ["CpmAntTokenizer"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 添加模型相关的导入项到 _import_structure 字典中
    _import_structure["modeling_cpmant"] = [
        "CPMANT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "CpmAntForCausalLM",
        "CpmAntModel",
        "CpmAntPreTrainedModel",
    ]

if TYPE_CHECKING:
    # 在类型检查时导入必要的模块和类
    from .configuration_cpmant import CPMANT_PRETRAINED_CONFIG_ARCHIVE_MAP, CpmAntConfig
    from .tokenization_cpmant import CpmAntTokenizer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 在类型检查时导入模型相关的类
        from .modeling_cpmant import (
            CPMANT_PRETRAINED_MODEL_ARCHIVE_LIST,
            CpmAntForCausalLM,
            CpmAntModel,
            CpmAntPreTrainedModel,
        )

else:
    import sys

    # 延迟加载模块的定义
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```