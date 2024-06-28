# `.\models\univnet\__init__.py`

```
# 版权声明和许可证信息
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

# 引入必要的类型检查模块
from typing import TYPE_CHECKING

# 引入依赖的模块和异常处理类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_univnet": [
        "UNIVNET_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "UnivNetConfig",
    ],
    "feature_extraction_univnet": ["UnivNetFeatureExtractor"],
}

# 检查是否有torch可用，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若torch可用，则添加对应的模型建模模块到_import_structure中
    _import_structure["modeling_univnet"] = [
        "UNIVNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "UnivNetModel",
    ]

# 如果是类型检查阶段，导入特定的配置、特征提取和模型建模类
if TYPE_CHECKING:
    from .configuration_univnet import (
        UNIVNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
        UnivNetConfig,
    )
    from .feature_extraction_univnet import UnivNetFeatureExtractor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_univnet import (
            UNIVNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            UnivNetModel,
        )

# 如果不是类型检查阶段，则进行模块的懒加载设置
else:
    import sys

    # 使用_LazyModule类来设置懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```