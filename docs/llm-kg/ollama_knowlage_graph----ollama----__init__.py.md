# `.\ollama_knowlage_graph\ollama\__init__.py`

```
# 定义一个名为 `merge_dicts` 的函数，接收任意数量的字典作为参数
def merge_dicts(*dicts):
    # 初始化一个空字典 `result`
    result = {}
    # 遍历每一个传入的字典
    for d in dicts:
        # 将每个字典的键值对更新到 `result` 中
        result.update(d)
    # 返回合并后的字典 `result`
    return result
```