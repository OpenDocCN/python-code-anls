# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\6082.dc4621c4b73d1f81.js`

```py
# 定义一个名为 `process_data` 的函数，接受一个名为 `data` 的参数
def process_data(data):
    # 如果 `data` 的值为 `None`，则返回空列表
    if data is None:
        return []
    # 从 `data` 中获取 `info` 键对应的值，如果不存在则返回空字典
    info = data.get('info', {})
    # 从 `info` 中获取 `records` 键对应的值，如果不存在则返回空列表
    records = info.get('records', [])
    # 通过列表推导式，过滤掉所有 `records` 列表中值为 `None` 或空字典的元素
    processed_records = [rec for rec in records if rec is not None and rec != {}]
    # 返回经过处理后的 `processed_records` 列表
    return processed_records
```