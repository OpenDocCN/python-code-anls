# `.\pytorch\torch\quantization\fx\graph_module.py`

```py
# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate files under `torch/ao/quantization/fx/`, while adding an import statement
here.
"""
# 从 torch/ao/quantization/fx/graph_module 模块导入以下类和函数
from torch.ao.quantization.fx.graph_module import (
    _is_observed_module,
    _is_observed_standalone_module,
    FusedGraphModule,
    GraphModule,
    ObservedGraphModule,
    ObservedStandaloneGraphModule,
    QuantizedGraphModule,
)
```