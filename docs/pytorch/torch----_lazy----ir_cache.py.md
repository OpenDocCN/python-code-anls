# `.\pytorch\torch\_lazy\ir_cache.py`

```
# 引入 mypy 的特定设置，允许未标注类型的函数定义
import torch._C._lazy

# 定义函数 dump，用于将 TrieCache 转储为 dot 格式
def dump(dot_file_name: str):
    """Dump TrieCache in the dot format"""
    # 调用 torch._C._lazy 模块中的 _dump_ir_cache 函数，将 TrieCache 转储为 dot 文件
    return torch._C._lazy._dump_ir_cache(dot_file_name)

# 定义函数 reset，用于清空 TrieCache，在测试中防止不同测试之间节点的重用
def reset():
    """Clear TrieCache. This is needed in testing to avoid
    node reusing between different tests.
    """
    # 调用 torch._C._lazy 模块中的 _clear_ir_cache 函数，清空 TrieCache
    return torch._C._lazy._clear_ir_cache()
```