# `D:\src\scipysrc\sympy\sympy\benchmarks\__init__.py`

```
# 定义一个类 `Person`
class Person:
    # 类级别的变量 `population`，用于跟踪创建的 `Person` 实例数
    population = 0

    # 初始化方法，每次创建 `Person` 实例时调用
    def __init__(self, name):
        # 实例变量 `name`，用于存储每个实例的名字
        self.name = name
        # 增加总人口计数
        Person.population += 1

    # 实例方法 `say_hello`，打印实例的名字和问候语
    def say_hello(self):
        print(f"Hello, my name is {self.name}")

# 创建 `Person` 类的实例 `p1`，传入名字参数 `"Alice"`
p1 = Person("Alice")
# 调用 `say_hello` 方法，打印 `"Hello, my name is Alice"`
p1.say_hello()

# 创建 `Person` 类的实例 `p2`，传入名字参数 `"Bob"`
p2 = Person("Bob")
# 调用 `say_hello` 方法，打印 `"Hello, my name is Bob"`
p2.say_hello()

# 打印 `Person` 类的总人口数量，此时应为 `2`
print(f"Total population: {Person.population}")
```