# `D:\src\scipysrc\pandas\pandas\core\tools\__init__.py`

```
# 导入 Python 内置的 datetime 模块
import datetime

# 定义一个名为 'format_date' 的函数，接收一个日期时间对象作为参数
def format_date(date_obj):
    # 调用日期时间对象的 strftime 方法，使用给定的格式化字符串将其格式化为字符串
    return date_obj.strftime('%Y-%m-%d')

# 创建一个 datetime.date 对象，表示当前日期
current_date = datetime.date.today()

# 调用 'format_date' 函数，将当前日期对象格式化为字符串，并将结果存储在 'formatted_date' 变量中
formatted_date = format_date(current_date)

# 打印格式化后的日期字符串
print(formatted_date)
```