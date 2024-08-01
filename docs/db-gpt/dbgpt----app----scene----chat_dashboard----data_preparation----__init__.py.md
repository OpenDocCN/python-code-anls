# `.\DB-GPT-src\dbgpt\app\scene\chat_dashboard\data_preparation\__init__.py`

```py
# 定义一个类名为 Person，表示一个人的信息
class Person:
    # 初始化方法，接受参数 name 和 age，并将它们赋值给实例变量
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    # 定义一个方法 greet，用于打印问候语，包含 self 参数表示该方法属于实例
    def greet(self):
        # 打印问候语，使用 format 方法填充 name 和 age
        print(f"Hello, my name is {self.name} and I am {self.age} years old.")

# 创建一个 Person 类的实例，传入参数 "Alice" 和 30
alice = Person("Alice", 30)
# 调用实例的 greet 方法，打印问候语
alice.greet()
```