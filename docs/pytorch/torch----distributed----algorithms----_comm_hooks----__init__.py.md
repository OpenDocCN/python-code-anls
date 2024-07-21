# `.\pytorch\torch\distributed\algorithms\_comm_hooks\__init__.py`

```
# 从当前目录中导入名为 default 的模块，并将其命名为 default
from . import default_hooks as default
# 定义了一个名为 LOW_PRECISION_HOOKS 的列表，包含了两个函数引用
LOW_PRECISION_HOOKS = [
    # 将 default 模块中的 fp16_compress_hook 函数添加到 LOW_PRECISION_HOOKS 列表中
    default.fp16_compress_hook,
    # 将 default 模块中的 bf16_compress_hook 函数添加到 LOW_PRECISION_HOOKS 列表中
    default.bf16_compress_hook,
]
```