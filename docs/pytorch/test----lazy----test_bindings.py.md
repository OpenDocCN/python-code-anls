# `.\pytorch\test\lazy\test_bindings.py`

```
# 导入 torch 库中的 metrics 模块
import torch._lazy.metrics

# 定义一个测试函数 test_metrics
def test_metrics():
    # 调用 torch 库中 metrics 模块的 counter_names 函数，获取计数器的名称列表
    names = torch._lazy.metrics.counter_names()
    # 断言获取到的计数器名称列表长度为 0，如果不是则抛出 AssertionError
    assert len(names) == 0, f"Expected no counter names, but got {names}"
```