# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\7835.678c71345b78788c.js`

```py
# 导入 datetime 模块，用于处理日期和时间
import datetime

# 定义函数 calculate_age，计算指定生日的年龄
def calculate_age(birth_date):
    # 获取当前日期
    today = datetime.date.today()
    # 计算年龄，通过当前日期减去生日得到时间差，再取整得到年龄
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    # 返回计算出的年龄
    return age

# 调用 calculate_age 函数，传入生日日期，并打印返回的年龄
print(calculate_age(datetime.date(1990, 8, 1)))
```