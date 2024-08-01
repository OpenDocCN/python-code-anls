# `.\DB-GPT-src\dbgpt\util\tests\__init__.py`

```py
# 定义一个名为`process_data`的函数，接收一个名为`data`的参数
def process_data(data):
    # 对于`item`在`data`列表中的每一个元素，执行以下操作
    for item in data:
        # 打印输出当前`item`的值
        print(item)

# 创建一个名为`data_list`的列表，包含三个整数元素：1, 2, 3
data_list = [1, 2, 3]
# 调用`process_data`函数，将`data_list`作为参数传入
process_data(data_list)
```