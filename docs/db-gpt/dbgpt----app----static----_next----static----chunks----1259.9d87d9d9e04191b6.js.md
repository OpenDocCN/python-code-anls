# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\1259.9d87d9d9e04191b6.js`

```py
# 导入Python标准库中的datetime模块
import datetime

# 定义一个名为Person的类
class Person:
    # 初始化方法，用于创建Person对象并设置属性
    def __init__(self, name, birthdate):
        # 将传入的name参数赋给对象的name属性
        self.name = name
        # 将传入的birthdate参数赋给对象的birthdate属性
        self.birthdate = birthdate
    
    # 方法用于计算并返回当前对象的年龄
    def age(self):
        # 使用datetime模块中的date类的today()方法获取当前日期，并从中减去对象的birthdate属性，计算出时间差
        delta = datetime.date.today() - self.birthdate
        # 返回时间差中的年份数
        return delta.days // 365

# 创建一个日期对象，表示1980年1月1日
birthdate = datetime.date(1980, 1, 1)
# 创建一个Person对象，传入姓名为John，出生日期为birthdate对象
john = Person('John', birthdate)
# 调用Person对象的age()方法计算John的年龄并打印输出
print(john.age())
```