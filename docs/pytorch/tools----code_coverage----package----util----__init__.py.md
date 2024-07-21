# `.\pytorch\tools\code_coverage\package\util\__init__.py`

```py
# 定义一个类 `Person`，表示一个人的基本信息和行为
class Person:
    # 初始化方法，创建一个新的 Person 对象
    def __init__(self, name, age):
        # 使用传入的参数初始化对象的名称属性
        self.name = name
        # 使用传入的参数初始化对象的年龄属性
        self.age = age

    # 打印对象的基本信息
    def display_info(self):
        # 使用对象的名称属性和年龄属性构造并打印信息字符串
        print(f"Name: {self.name}, Age: {self.age}")

# 创建一个名为 'Alice'，年龄为 25 的 Person 对象
person1 = Person('Alice', 25)
# 调用对象的 display_info 方法，打印对象的基本信息
person1.display_info()
```