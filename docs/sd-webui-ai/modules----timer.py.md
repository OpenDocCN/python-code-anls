# `stable-diffusion-webui\modules\timer.py`

```
# 导入时间模块
import time
# 导入参数解析模块
import argparse

# 定义计时器子类
class TimerSubcategory:
    # 初始化计时器子类
    def __init__(self, timer, category):
        self.timer = timer
        self.category = category
        self.start = None
        self.original_base_category = timer.base_category

    # 进入计时器子类上下文
    def __enter__(self):
        self.start = time.time()
        self.timer.base_category = self.original_base_category + self.category + "/"
        self.timer.subcategory_level += 1

        # 如果需要打印日志，则输出子类别名称
        if self.timer.print_log:
            print(f"{'  ' * self.timer.subcategory_level}{self.category}:")

    # 退出计时器子类上下文
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_for_subcategroy = time.time() - self.start
        self.timer.base_category = self.original_base_category
        self.timer.add_time_to_record(self.original_base_category + self.category, elapsed_for_subcategroy)
        self.timer.subcategory_level -= 1
        self.timer.record(self.category, disable_log=True)

# 定义计时器类
class Timer:
    # 初始化计时器
    def __init__(self, print_log=False):
        self.start = time.time()
        self.records = {}
        self.total = 0
        self.base_category = ''
        self.print_log = print_log
        self.subcategory_level = 0

    # 计算经过的时间
    def elapsed(self):
        end = time.time()
        res = end - self.start
        self.start = end
        return res

    # 将时间添加到记录中
    def add_time_to_record(self, category, amount):
        if category not in self.records:
            self.records[category] = 0

        self.records[category] += amount

    # 记录时间
    def record(self, category, extra_time=0, disable_log=False):
        e = self.elapsed()

        self.add_time_to_record(self.base_category + category, e + extra_time)

        self.total += e + extra_time

        # 如果需要打印日志且不禁用日志，则输出类别名称和所用时间
        if self.print_log and not disable_log:
            print(f"{'  ' * self.subcategory_level}{category}: done in {e + extra_time:.3f}s")

    # 创建子类别
    def subcategory(self, name):
        self.elapsed()

        subcat = TimerSubcategory(self, name)
        return subcat
    # 生成一个包含总时间的字符串
    def summary(self):
        res = f"{self.total:.1f}s"

        # 筛选出时间大于等于0.1且不包含'/'的记录
        additions = [(category, time_taken) for category, time_taken in self.records.items() if time_taken >= 0.1 and '/' not in category]
        # 如果没有符合条件的记录，则直接返回结果字符串
        if not additions:
            return res

        # 在结果字符串后添加括号
        res += " ("
        # 将筛选出的记录格式化为字符串并添加到结果字符串中
        res += ", ".join([f"{category}: {time_taken:.1f}s" for category, time_taken in additions])
        # 在结果字符串后添加右括号
        res += ")"

        # 返回最终结果字符串
        return res

    # 返回一个包含总时间和记录的字典
    def dump(self):
        return {'total': self.total, 'records': self.records}

    # 重置记录
    def reset(self):
        # 调用初始化方法重新初始化对象
        self.__init__()
# 创建一个参数解析器对象，用于解析命令行参数，禁用默认的帮助选项
parser = argparse.ArgumentParser(add_help=False)
# 添加一个命令行参数，--log-startup，当设置时会打印详细的启动日志
parser.add_argument("--log-startup", action='store_true', help="print a detailed log of what's happening at startup")
# 解析命令行参数，并获取第一个元素，即参数的值
args = parser.parse_known_args()[0]

# 创建一个计时器对象，用于记录启动时间，根据参数中的log_startup值决定是否打印日志
startup_timer = Timer(print_log=args.log_startup)

# 初始化一个启动记录对象，暂时赋值为None
startup_record = None
```