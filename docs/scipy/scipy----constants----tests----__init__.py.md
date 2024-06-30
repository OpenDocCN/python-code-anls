# `D:\src\scipysrc\scipy\scipy\constants\tests\__init__.py`

```
# 定义一个类 `Person`，用于表示人员信息
class Person:
    # 类级别的变量 `count`，用于记录创建的 `Person` 对象数量
    count = 0

    # 类的初始化方法，初始化人员的 `name` 和 `age` 属性，并增加 `count` 的值
    def __init__(self, name, age):
        self.name = name  # 设置人员的姓名
        self.age = age    # 设置人员的年龄
        Person.count += 1  # 每次创建 `Person` 对象，`count` 自增1

    # 类方法 `get_count`，返回已创建的 `Person` 对象数量
    @classmethod
    def get_count(cls):
        return cls.count  # 返回类级别的 `count` 属性值

# 创建一个 `Person` 对象，姓名为 'Alice'，年龄为 25
p1 = Person('Alice', 25)
# 创建一个 `Person` 对象，姓名为 'Bob'，年龄为 30
p2 = Person('Bob', 30)

# 打印已创建的 `Person` 对象数量
print(Person.get_count())
```