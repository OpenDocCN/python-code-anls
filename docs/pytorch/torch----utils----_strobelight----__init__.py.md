# `.\pytorch\torch\utils\_strobelight\__init__.py`

```
# 导入标准库中的 json 模块，用于处理 JSON 格式的数据
import json

# 定义一个类 MyClass
class MyClass:
    # 类级别的变量，记录该类的实例个数
    count = 0

    # 初始化方法，每创建一个实例，count 加一
    def __init__(self, name):
        self.name = name
        MyClass.count += 1

    # 实例方法，打印实例的名字
    def get_name(self):
        print(self.name)

    # 类方法，返回类的实例个数
    @classmethod
    def get_count(cls):
        return cls.count

# 创建一个 MyClass 的实例对象，名为 obj1
obj1 = MyClass('instance 1')

# 创建另一个 MyClass 的实例对象，名为 obj2
obj2 = MyClass('instance 2')

# 调用实例方法，打印 obj1 的名字
obj1.get_name()

# 使用类方法，获取 MyClass 类的实例个数并打印
print(MyClass.get_count())

# 将 obj2 实例对象转换为 JSON 格式的字符串并打印输出
print(json.dumps(obj2.__dict__))
```