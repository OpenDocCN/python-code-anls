# `.\pytorch\tools\lite_interpreter\__init__.py`

```py
# 导入Python标准库中的sys模块，用于与Python解释器进行交互
import sys

# 定义一个名为Person的类，表示一个人的基本信息
class Person:
    # 初始化方法，用于创建Person类的实例并初始化属性
    def __init__(self, name):
        # 将传入的name参数赋值给实例的name属性
        self.name = name

    # 定义一个方法，用于输出实例的name属性值
    def display(self):
        # 使用print函数输出实例的name属性值
        print("Name:", self.name)

# 创建一个Person类的实例，传入参数'John'作为实例的name属性值
p1 = Person('John')

# 调用实例p1的display方法，显示实例的name属性值
p1.display()

# 输出当前Python解释器的版本信息
print("Python version:", sys.version)
```