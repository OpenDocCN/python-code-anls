# `.\models\hubert\__init__.py`

```py
# 版权声明和许可证信息
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

# 导入依赖检查函数和 LazyModule 类
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_torch_available

# 定义模块的导入结构字典
_import_structure = {"configuration_hubert": ["HUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "HubertConfig"]}

# 检查是否有 torch 可用，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 可用，则添加对应的 modeling_hubert 模块导入结构
    _import_structure["modeling_hubert"] = [
        "HUBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "HubertForCTC",
        "HubertForSequenceClassification",
        "HubertModel",
        "HubertPreTrainedModel",
    ]

# 检查是否有 tensorflow 可用，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 tensorflow 可用，则添加对应的 modeling_tf_hubert 模块导入结构
    _import_structure["modeling_tf_hubert"] = [
        "TF_HUBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFHubertForCTC",
        "TFHubertModel",
        "TFHubertPreTrainedModel",
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 导入配置和模型类
    from .configuration_hubert import HUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, HubertConfig

    # 如果 torch 可用，则导入 torch 版的模型类
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_hubert import (
            HUBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            HubertForCTC,
            HubertForSequenceClassification,
            HubertModel,
            HubertPreTrainedModel,
        )

    # 如果 tensorflow 可用，则导入 tensorflow 版的模型类
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_hubert import (
            TF_HUBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFHubertForCTC,
            TFHubertModel,
            TFHubertPreTrainedModel,
        )

# 如果不在类型检查模式下
else:
    import sys

    # 将当前模块注册为 LazyModule，延迟导入实现
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```