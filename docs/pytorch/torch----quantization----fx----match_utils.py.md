# `.\pytorch\torch\quantization\fx\match_utils.py`

```
# flake8: noqa: F401
"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate files under `torch/ao/quantization/fx/`, while adding an import statement
here.
"""

# 从torch.ao.quantization.fx.match_utils模块导入以下函数和类：
# _find_matches: 用于查找匹配项的函数
# _is_match: 判断是否为匹配项的函数
# _MatchResult: 匹配结果的类
# MatchAllNode: 匹配所有节点的类
from torch.ao.quantization.fx.match_utils import (
    _find_matches,
    _is_match,
    _MatchResult,
    MatchAllNode,
)
```