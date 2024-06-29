# `D:\src\scipysrc\pandas\pandas\tests\window\moments\__init__.py`

```
# 定义一个名为 merge_dicts 的函数，接受两个字典作为参数
def merge_dicts(d1, d2):
    # 使用字典 d1 的内容创建一个新的字典，避免直接修改原始字典
    merged = dict(d1)
    # 将字典 d2 的键值对逐一添加到 merged 字典中
    merged.update(d2)
    # 返回合并后的字典
    return merged
```