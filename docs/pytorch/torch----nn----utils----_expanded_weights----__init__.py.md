# `.\pytorch\torch\nn\utils\_expanded_weights\__init__.py`

```
# 导入自定义模块中的特定类和函数
from .conv_expanded_weights import ConvPerSampleGrad
from .embedding_expanded_weights import EmbeddingPerSampleGrad
from .expanded_weights_impl import ExpandedWeight
from .group_norm_expanded_weights import GroupNormPerSampleGrad
from .instance_norm_expanded_weights import InstanceNormPerSampleGrad
from .layer_norm_expanded_weights import LayerNormPerSampleGrad
from .linear_expanded_weights import LinearPerSampleGrad

# 将 "ExpandedWeight" 添加到 __all__ 列表中，使其在使用 import * 时被导入
__all__ = ["ExpandedWeight"]
```