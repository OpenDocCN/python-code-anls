# `.\pytorch\torchgen\selective_build\__init__.py`

```
# 定义一个类名为 Person
class Person:
    # 初始化方法，接收 name 和 age 两个参数
    def __init__(self, name, age):
        # 将 name 参数赋值给实例变量 self.name
        self.name = name
        # 将 age 参数赋值给实例变量 self.age
        self.age = age

    # 定义一个方法 greet，用于打印问候语
    def greet(self):
        # 打印格式化的问候语，包含实例变量 self.name
        print(f"Hello, my name is {self.name}!")

# 创建一个名为 alice 的 Person 实例，传入 "Alice" 和 30 作为参数
alice = Person("Alice", 30)
# 调用 alice 实例的 greet 方法，打印问候语
alice.greet()
```