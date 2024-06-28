# `.\models\sew\__init__.py`

```
# 版权声明及许可信息，指明此代码的版权归HuggingFace团队所有，受Apache License, Version 2.0许可
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

# 引入类型检查模块，用于检查类型是否可用
from typing import TYPE_CHECKING

# 引入必要的依赖项，包括OptionalDependencyNotAvailable异常和_LazyModule
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构，用于延迟加载模块
_import_structure = {"configuration_sew": ["SEW_PRETRAINED_CONFIG_ARCHIVE_MAP", "SEWConfig"]}

# 检查是否可以使用torch库，若不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加SEW相关模型到导入结构中
    _import_structure["modeling_sew"] = [
        "SEW_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SEWForCTC",
        "SEWForSequenceClassification",
        "SEWModel",
        "SEWPreTrainedModel",
    ]

# 如果当前环境是类型检查模式
if TYPE_CHECKING:
    # 从configuration_sew模块中导入所需内容，包括SEW_PRETRAINED_CONFIG_ARCHIVE_MAP和SEWConfig
    from .configuration_sew import SEW_PRETRAINED_CONFIG_ARCHIVE_MAP, SEWConfig

    # 再次检查是否可以使用torch库，若不可用则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，从modeling_sew模块中导入SEW相关的模型和类
        from .modeling_sew import (
            SEW_PRETRAINED_MODEL_ARCHIVE_LIST,
            SEWForCTC,
            SEWForSequenceClassification,
            SEWModel,
            SEWPreTrainedModel,
        )

# 如果当前环境不是类型检查模式
else:
    # 导入sys模块
    import sys

    # 将当前模块注册为_LazyModule，用于延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```