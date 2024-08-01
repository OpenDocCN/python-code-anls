# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\2075.e1643aa91d0d5714.js`

```py
# 定义一个类名为 Person，用于表示人员信息
class Person:
    # 类级别的属性，用于记录总人数
    population = 0
    
    # 初始化方法，用于创建一个新的实例并初始化实例变量
    def __init__(self, name):
        # 实例级别的属性，用于记录每个人的名字
        self.name = name
        # 每次创建新实例时，总人数加一
        Person.population += 1
    
    # 实例方法，用于打印每个人的名字
    def print_name(self):
        print(f"My name is {self.name}")
    
    # 类方法，用于打印当前总人数
    @classmethod
    def print_population(cls):
        print(f"We have {cls.population} people")
```