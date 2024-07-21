# `.\pytorch\torch\ao\pruning\_experimental\pruner\__init__.py`

```
# 从当前包中导入 BaseStructuredSparsifier 类
from .base_structured_sparsifier import BaseStructuredSparsifier
# 从当前包中导入 FakeStructuredSparsity 和 BiasHook 类
from .parametrization import (
    FakeStructuredSparsity,
    BiasHook,
)
# 从当前包中导入 SaliencyPruner 类
from .saliency_pruner import SaliencyPruner
# 从当前包中导入 LSTMSaliencyPruner 类
from .lstm_saliency_pruner import LSTMSaliencyPruner
# 从当前包中导入 FPGMPruner 类
from .FPGM_pruner import FPGMPruner
```