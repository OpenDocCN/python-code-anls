# `D:\src\scipysrc\pandas\pandas\tests\io\sas\__init__.py`

```
# 导入所需的模块：datetime 用于日期时间操作
import datetime

# 定义一个名为 `days_between` 的函数，接受两个参数 `d1` 和 `d2`
def days_between(d1, d2):
    # 如果 `d1` 大于 `d2`，交换它们的值
    if d1 > d2:
        d1, d2 = d2, d1
    # 返回两个日期之间的天数差，使用 `days` 方法获取 `timedelta` 对象，再取 `days` 属性
    return (d2 - d1).days

# 使用 `datetime` 模块的 `datetime` 类创建日期对象 `date1`，赋值为 2017 年 3 月 31 日
date1 = datetime.datetime(2017, 3, 31)
# 使用同样的方法创建日期对象 `date2`，赋值为 2017 年 5 月 5 日
date2 = datetime.datetime(2017, 5, 5)

# 调用 `days_between` 函数计算 `date1` 和 `date2` 之间的天数差，存储在 `diff` 变量中
diff = days_between(date1, date2)

# 打印计算结果，显示 `date1` 和 `date2` 之间的天数差
print(f"Days difference: {diff}")
```