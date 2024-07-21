# `.\pytorch\torch\nested\_internal\__init__.py`

```
# 导入Python内置的datetime模块，用于处理日期和时间
import datetime

# 定义一个函数，名称为calculate_age，接收参数birth_date
def calculate_age(birth_date):
    # 获取当前日期
    today = datetime.date.today()
    # 计算年龄，通过当前日期减去出生日期得到时间差对象，再取其年份差作为年龄
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    # 返回计算出的年龄
    return age
```