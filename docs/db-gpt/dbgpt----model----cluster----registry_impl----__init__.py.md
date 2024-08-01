# `.\DB-GPT-src\dbgpt\model\cluster\registry_impl\__init__.py`

```py
# 导入所需模块：datetime模块用于处理日期和时间
import datetime

# 定义函数get_last_friday，无参数，返回最近的上周五的日期对象
def get_last_friday():
    # 获取今天的日期对象
    today = datetime.date.today()
    # 计算今天是星期几（0代表星期一，6代表星期日）
    today_weekday = today.weekday()
    # 计算距离上周五还有多少天
    days_to_friday = (today_weekday + 1) % 7
    # 计算上周五的日期
    last_friday = today - datetime.timedelta(days=days_to_friday)
    # 返回上周五的日期对象
    return last_friday
```