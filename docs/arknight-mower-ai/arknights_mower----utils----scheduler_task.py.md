# `arknights-mower\arknights_mower\utils\scheduler_task.py`

```py
# 导入 datetime 和 timedelta 模块
from datetime import datetime, timedelta
# 导入 copy 模块
import copy
# 从 arknights_mower.utils.datetime 模块中导入 the_same_time 函数
from arknights_mower.utils.datetime import the_same_time

# 定义 SchedulerTask 类
class SchedulerTask:
    # 初始化类属性
    time = None
    type = ''
    plan = {}
    meta_flag = False

    # 初始化方法
    def __init__(self, time=None, task_plan={}, task_type='', meta_flag=False):
        # 如果时间为空，则设置为当前时间
        if time is None:
            self.time = datetime.now()
        else:
            self.time = time
        # 设置任务计划
        self.plan = task_plan
        # 设置任务类型
        self.type = task_type
        # 设置元标志
        self.meta_flag = meta_flag

    # 时间偏移方法
    def time_offset(self, h):
        # 复制当前对象
        after_offset = copy.deepcopy(self)
        # 对时间进行偏移
        after_offset.time += timedelta(hours=h)
        # 返回偏移后的对象
        return after_offset

    # 定义对象的字符串表示形式
    def __str__(self):
        return f"SchedulerTask(time='{self.time}',task_plan={self.plan},task_type='{self.type}',meta_flag={self.meta_flag})"

    # 定义对象的相等性比较方法
    def __eq__(self, other):
        # 如果 other 是 SchedulerTask 类型的对象
        if isinstance(other, SchedulerTask):
            # 比较任务类型、任务计划、时间是否相同以及元标志是否相同
            return self.type == other.type and self.plan == other.plan and the_same_time(self.time, other.time) and self.meta_flag == other.meta_flag
        # 如果 other 不是 SchedulerTask 类型的对象，则返回 False
        return False
```