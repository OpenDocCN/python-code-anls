# `.\pytorch\torch\fx\passes\dialect\common\__init__.py`

```py
# 定义一个类 Calculator，用于实现基本的数学运算
class Calculator:
    # 初始化方法，设置初始值
    def __init__(self, init_value=0):
        # 使用 init_value 初始化实例变量 self.result
        self.result = init_value

    # 加法方法，将参数 value 加到实例变量 self.result 上
    def add(self, value):
        self.result += value

    # 减法方法，将参数 value 减去实例变量 self.result
    def subtract(self, value):
        self.result -= value

    # 乘法方法，将参数 value 乘到实例变量 self.result 上
    def multiply(self, value):
        self.result *= value

    # 除法方法，将实例变量 self.result 除以参数 value
    def divide(self, value):
        self.result /= value
```