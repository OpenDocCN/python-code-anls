# `.\pytorch\tools\alerts\__init__.py`

```py
# 导入标准库中的 datetime 模块
import datetime

# 定义一个名为 Person 的类
class Person:
    # 初始化方法，接受参数 name 和 birthdate
    def __init__(self, name, birthdate):
        # 将 name 参数赋值给实例变量 self.name
        self.name = name
        # 将 birthdate 参数赋值给实例变量 self.birthdate
        self.birthdate = birthdate
    
    # 计算并返回当前实例的年龄
    def calculate_age(self):
        # 使用当前日期和出生日期计算年龄
        today = datetime.date.today()
        age = today.year - self.birthdate.year - ((today.month, today.day) < (self.birthdate.month, self.birthdate.day))
        # 返回计算出的年龄
        return age

# 创建一个名为 john 的 Person 实例，传入姓名和出生日期
john = Person('John', datetime.date(1990, 5, 15))
# 调用实例的 calculate_age 方法，计算并返回 john 的年龄
john_age = john.calculate_age()
# 打印 john 的姓名和年龄
print(f'{john.name} is {john_age} years old.')
```