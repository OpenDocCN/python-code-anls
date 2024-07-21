# `.\pytorch\torch\utils\model_zoo.py`

```
# 从 torch.hub 中导入 tqdm 和 load_state_dict_from_url 函数
# `noqa: F401` 是一个特殊的注释，用来告诉 linter 忽略未使用的 import 警告
from torch.hub import tqdm, load_state_dict_from_url as load_url  # noqa: F401
```