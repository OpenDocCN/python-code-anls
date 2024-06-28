# `.\models\time_series_transformer\__init__.py`

```py
# 引入类型检查器的功能，用于在类型检查时导入特定模块和对象
from typing import TYPE_CHECKING

# 引入自定义的异常类，用于处理依赖不可用的情况，以及 LazyModule 的实现和 torch 是否可用的检查
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块导入结构的字典，包含了配置和模型的名称
_import_structure = {
    "configuration_time_series_transformer": [
        "TIME_SERIES_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "TimeSeriesTransformerConfig",
    ],
}

# 检查是否可用 torch 库，若不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若 torch 可用，则添加相关模型的导入结构到 _import_structure 字典中
    _import_structure["modeling_time_series_transformer"] = [
        "TIME_SERIES_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TimeSeriesTransformerForPrediction",
        "TimeSeriesTransformerModel",
        "TimeSeriesTransformerPreTrainedModel",
    ]

# 如果正在进行类型检查
if TYPE_CHECKING:
    # 导入配置模块中所需的内容
    from .configuration_time_series_transformer import (
        TIME_SERIES_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TimeSeriesTransformerConfig,
    )

    try:
        # 再次检查 torch 是否可用
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型模块中所需的内容
        from .modeling_time_series_transformer import (
            TIME_SERIES_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TimeSeriesTransformerForPrediction,
            TimeSeriesTransformerModel,
            TimeSeriesTransformerPreTrainedModel,
        )

# 如果不是类型检查模式，则进行 LazyModule 的初始化
else:
    import sys

    # 将当前模块替换为 LazyModule，以支持按需导入和延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```