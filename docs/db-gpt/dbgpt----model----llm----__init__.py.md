# `.\DB-GPT-src\dbgpt\model\llm\__init__.py`

```py
# 定义一个函数，名称为merge_dicts，接收两个字典作为参数，将它们合并后返回
def merge_dicts(d1, d2):
    # 创建一个新的字典对象，初始化为空
    merged = {}
    # 将第一个字典d1的所有键值对添加到新字典merged中
    merged.update(d1)
    # 将第二个字典d2的所有键值对添加到新字典merged中
    merged.update(d2)
    # 返回合并后的字典对象merged
    return merged
```