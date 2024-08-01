# `.\DB-GPT-src\dbgpt\app\scene\chat_db\__init__.py`

```py
# 定义一个名为 `process_data` 的函数，接受一个名为 `data` 的参数
def process_data(data):
    # 如果 `data` 的值为 `None`
    if data is None:
        # 返回空字典
        return {}
    # 初始化一个空列表 `results`
    results = []
    # 对于 `item` 在 `data` 上的迭代
    for item in data:
        # 如果 `item` 的类型是 `int`
        if isinstance(item, int):
            # 将 `item` 的平方添加到 `results` 列表中
            results.append(item ** 2)
        # 否则，如果 `item` 的类型是 `str`
        elif isinstance(item, str):
            # 将 `item` 转换为大写并添加到 `results` 列表中
            results.append(item.upper())
    # 返回 `results` 列表
    return results
```