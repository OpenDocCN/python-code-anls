# `D:\src\scipysrc\sympy\sympy\vector\tests\__init__.py`

```
# 导入datetime模块，用于处理日期和时间
import datetime

# 定义函数calculate_age，接收参数birth_year和current_year
def calculate_age(birth_year, current_year):
    # 返回当前年份与出生年份的差值，即年龄
    return current_year - birth_year

# 调用函数calculate_age计算1990年出生的人在2024年的年龄
age = calculate_age(1990, 2024)
# 打印计算结果
print(f"The person's age is: {age}")
```