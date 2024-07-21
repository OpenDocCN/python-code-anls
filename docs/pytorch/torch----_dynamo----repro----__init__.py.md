# `.\pytorch\torch\_dynamo\repro\__init__.py`

```py
# 导入 json 模块
import json

# 定义一个函数，接收一个字符串参数
def process_data(data):
    # 将字符串解析为 JSON 对象
    obj = json.loads(data)
    # 如果 JSON 对象中包含 'key' 键，则执行以下操作
    if 'key' in obj:
        # 获取 'key' 键对应的值
        value = obj['key']
        # 打印该值
        print(value)
    # 否则，打印指定消息
    else:
        print("No key found")

# 测试示例数据
data = '{"key": "value"}'
# 调用函数处理数据
process_data(data)
```