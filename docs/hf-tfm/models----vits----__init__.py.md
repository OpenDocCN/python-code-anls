# `.\models\vits\__init__.py`

```
# 版权声明及许可信息
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

# 从本地模块导入所需的工具函数和类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_speech_available,
    is_torch_available,
)

# 定义导入结构，以便在懒加载时使用
_import_structure = {
    "configuration_vits": [
        "VITS_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "VitsConfig",
    ],
    "tokenization_vits": ["VitsTokenizer"],
}

# 尝试检查是否有必要的 Torch 库可用
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则添加下列模型相关的导入
    _import_structure["modeling_vits"] = [
        "VITS_PRETRAINED_MODEL_ARCHIVE_LIST",
        "VitsModel",
        "VitsPreTrainedModel",
    ]

# 如果是类型检查阶段，则从相应模块中导入特定类和常量
if TYPE_CHECKING:
    from .configuration_vits import (
        VITS_PRETRAINED_CONFIG_ARCHIVE_MAP,
        VitsConfig,
    )
    from .tokenization_vits import VitsTokenizer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_vits import (
            VITS_PRETRAINED_MODEL_ARCHIVE_LIST,
            VitsModel,
            VitsPreTrainedModel,
        )

# 如果不是类型检查阶段，则进行懒加载处理
else:
    import sys

    # 将当前模块替换为懒加载模块对象
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```