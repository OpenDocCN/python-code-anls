# `.\pytorch\torch\utils\data\backward_compatibility.py`

```py
# 声明使用 mypy 时允许未类型化的函数定义
# 导入 typing_extensions 模块中的 deprecated 别名作为 _deprecated
from typing_extensions import deprecated as _deprecated

# 使用 _deprecated 装饰器标记函数，表示此函数已被弃用
# 给出警告信息说明 `backward_compatibility.worker_init_fn` 的使用已被弃用，
# 因为 `DataLoader` 在每个工作进程中自动应用分片
# 使用 FutureWarning 作为警告的类别
@_deprecated(
    "Usage of `backward_compatibility.worker_init_fn` is deprecated "
    "as `DataLoader` automatically applies sharding in every worker",
    category=FutureWarning,
)
# 定义函数 worker_init_fn，接受一个参数 worker_id，但函数体为空
def worker_init_fn(worker_id):
    pass
```