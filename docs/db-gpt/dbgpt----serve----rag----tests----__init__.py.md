# `.\DB-GPT-src\dbgpt\serve\rag\tests\__init__.py`

```py
# 导入Python标准库中的datetime模块，用于日期和时间处理
import datetime

# 定义一个名为get_weekday的函数，接受一个日期参数d
def get_weekday(d):
    # 调用datetime模块中的weekday方法，返回星期几的数字表示（0代表星期一，6代表星期日）
    return d.weekday()

# 创建一个datetime对象，表示当前日期和时间
now = datetime.datetime.now()

# 调用get_weekday函数，传入当前日期时间对象now，并将返回值赋给变量weekday
weekday = get_weekday(now)

# 打印输出当前日期的星期几（返回值是0到6的整数）
print(weekday)
```