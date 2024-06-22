# `.\models\hubert\__init__.py`

```py
# 版权声明和许可证信息
# 版权声明和许可证信息，指明代码版权和许可证信息
# 根据 Apache 许可证版本 2.0 进行许可
# 在遵守许可证的前提下可以使用该文件
# 获取许可证的副本
# 如果不符合适用法律或书面同意的要求，软件将按"原样"分发
# 没有任何明示或暗示的担保或条件，包括但不限于
# 特定目的的适销性或适用性
# 请查看许可证以获取有关权限和限制的详细信息
# 从类型提示中导入 TYPE_CHECKING
from typing import TYPE_CHECKING

# 从工具模块中导入必要的依赖
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_torch_available

# 定义模块导入结构
_import_structure = {"configuration_hubert": ["HUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "HubertConfig"]}

# 检查是否存在 torch 库，如果不存在则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 torch 库，则添加相关模块到导入结构中
    _import_structure["modeling_hubert"] = [
        "HUBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "HubertForCTC",
        "HubertForSequenceClassification",
        "HubertModel",
        "HubertPreTrainedModel",
    ]

# 检查是否存在 tensorflow 库，如果不存在则引发异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 tensorflow 库，则添加相关模块到导入结构中
    _import_structure["modeling_tf_hubert"] = [
        "TF_HUBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFHubertForCTC",
        "TFHubertModel",
        "TFHubertPreTrainedModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 从配置模块中导入相关内容
    from .configuration_hubert import HUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, HubertConfig

    # 检查是否存在 torch 库，如果不存在则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从模型模块中导入相关内容
        from .modeling_hubert import (
            HUBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            HubertForCTC,
            HubertForSequenceClassification,
            HubertModel,
            HubertPreTrainedModel,
        )

    # 检查是否存在 tensorflow 库，如果不存在则引发异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从 tensorflow 模型模块中导入相关内容
        from .modeling_tf_hubert import (
            TF_HUBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFHubertForCTC,
            TFHubertModel,
            TFHubertPreTrainedModel,
        )

# 如果不是类型检查模式
else:
    # 导入 sys 模块
    import sys

    # 将当前模块设置为 LazyModule 类型
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```