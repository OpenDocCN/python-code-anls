# `.\pytorch\torch\utils\_stats.py`

```
# mypy: allow-untyped-defs
# NOTE! PLEASE KEEP THIS FILE *FREE* OF TORCH DEPS! IT SHOULD BE IMPORTABLE ANYWHERE.
# IF YOU FEEL AN OVERWHELMING URGE TO ADD A TORCH DEP, MAKE A TRAMPOLINE FILE A LA torch._dynamo.utils
# AND SCRUB AWAY TORCH NOTIONS THERE.

# 导入 collections 模块，用于处理有序字典 OrderedDict
import collections
# 导入 functools 模块，用于函数装饰器和函数包装
import functools
# 导入 typing 模块中的 OrderedDict 类型
from typing import OrderedDict

# 定义一个有序字典 simple_call_counter，用于记录函数调用次数，键为字符串类型的函数名，值为整数类型的调用次数
simple_call_counter: OrderedDict[str, int] = collections.OrderedDict()

# 定义一个函数 count_label，用于增加指定标签对应的调用计数
def count_label(label):
    # 获取当前标签的调用次数，如果不存在则设置为 0
    prev = simple_call_counter.setdefault(label, 0)
    # 将当前标签的调用次数加一
    simple_call_counter[label] = prev + 1

# 定义一个装饰器函数 count，用于装饰其它函数，实现记录函数调用次数
def count(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # 如果函数名在 simple_call_counter 中不存在，则初始化其调用次数为 0
        if fn.__qualname__ not in simple_call_counter:
            simple_call_counter[fn.__qualname__] = 0
        # 将函数名对应的调用次数加一
        simple_call_counter[fn.__qualname__] = simple_call_counter[fn.__qualname__] + 1
        # 调用被装饰的原函数，并返回其结果
        return fn(*args, **kwargs)
    # 返回装饰后的包装函数
    return wrapper
```