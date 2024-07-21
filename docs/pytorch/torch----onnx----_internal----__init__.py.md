# `.\pytorch\torch\onnx\_internal\__init__.py`

```
# 导入Python标准库中的json模块，用于处理JSON格式数据
import json

# 定义一个Python类，名称为Person，用于表示一个人的信息
class Person:
    # 初始化方法，用于设置Person对象的初始属性
    def __init__(self, name, age):
        # 将传入的name参数赋值给对象的name属性
        self.name = name
        # 将传入的age参数赋值给对象的age属性
        self.age = age

    # 定义一个实例方法，用于将Person对象转换为JSON格式字符串
    def toJSON(self):
        # 使用json.dumps方法将Person对象转换为JSON格式字符串，
        # ensure_ascii=False参数用于确保在输出时不转义非ASCII字符
        return json.dumps(self, default=lambda o: o.__dict__, ensure_ascii=False)

# 创建一个Person对象，传入'name'和'age'参数分别为'John'和30
person = Person('John', 30)

# 调用Person对象的toJSON方法，将其转换为JSON格式字符串并打印输出
print(person.toJSON())
```