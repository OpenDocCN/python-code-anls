# `.\transformers\models\time_series_transformer\__init__.py`

```py
# 版权声明
# 版权所有 © 2022 年 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版（以下简称“许可证”）许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以获取许可证的拷贝，网址为
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，按照“现状”方式发布的软件，
# 没有任何明示或暗示的担保或条件。
# 详见许可证以获得指定语言下的权限及限制。
from typing import TYPE_CHECKING

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 导入结构定义
_import_structure = {
    "configuration_time_series_transformer": [
        "TIME_SERIES_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "TimeSeriesTransformerConfig",
    ],
}

# 尝试导入 torch 模块，如果不可用，则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可以导入 torch 模块，则继续增加导入结构定义
    _import_structure["modeling_time_series_transformer"] = [
        "TIME_SERIES_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TimeSeriesTransformerForPrediction",
        "TimeSeriesTransformerModel",
        "TimeSeriesTransformerPreTrainedModel",
    ]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 导入时间序列变换器配置及相关内容
    from .configuration_time_series_transformer import (
        TIME_SERIES_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TimeSeriesTransformerConfig,
    )

    # 在此处再次尝试导入 torch 模块，如果不可用，则异常会被捕获
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入时间序列变换器模型相关内容
        from .modeling_time_series_transformer import (
            TIME_SERIES_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TimeSeriesTransformerForPrediction,
            TimeSeriesTransformerModel,
            TimeSeriesTransformerPreTrainedModel,
        )

# 如果不是类型检查阶段
else:
    # 导入 sys 模块
    import sys

    # 将当前模块映射到 LazyModule 类，并指定导入结构
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```