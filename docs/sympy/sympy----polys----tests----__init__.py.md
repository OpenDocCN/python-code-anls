# `D:\src\scipysrc\sympy\sympy\polys\tests\__init__.py`

```
# 导入 Python 内置模块 json
import json

# 定义一个函数 greet，接收一个参数 name
def greet(name):
    # 打印欢迎信息，使用传入的 name 参数
    print(f"Hello, {name}!")

# 调用 greet 函数，传入参数 "Alice"
greet("Alice")

# 创建一个字典，包含两个键值对
data = {
    'name': 'John',
    'age': 30
}

# 将字典 data 转换为 JSON 格式的字符串，存储在变量 json_str 中
json_str = json.dumps(data)

# 打印 JSON 字符串
print(json_str)
```