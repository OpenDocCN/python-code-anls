# `.\models\dit\__init__.py`

```py
# 导入标准库中的datetime模块，用于处理日期和时间相关的操作
import datetime

# 定义一个名为format_date的函数，接受一个datetime对象作为参数，返回格式化后的日期字符串
def format_date(dt):
    # 使用datetime对象的strftime方法格式化日期，"%Y-%m-%d"表示年-月-日的格式
    return dt.strftime("%Y-%m-%d")

# 创建一个datetime对象，表示当前日期和时间
current_date = datetime.datetime.now()

# 调用format_date函数，将current_date格式化为字符串并赋值给formatted_date变量
formatted_date = format_date(current_date)

# 打印格式化后的日期字符串
print(formatted_date)
```