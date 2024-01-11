# `arknights-mower\arknights_mower\solvers\schedule.py`

```
# 导入 datetime 模块
import datetime
# 导入 time 模块
import time
# 从 collections.abc 模块中导入 Callable 类
from collections.abc import Callable
# 从 functools 模块中导入 cmp_to_key 函数
from functools import cmp_to_key
# 从 pathlib 模块中导入 Path 类
from pathlib import Path

# 从 ruamel.yaml 模块中导入 yaml_object 函数
from ruamel.yaml import yaml_object

# 从上层目录的 utils 模块中导入 config
from ..utils import config
# 从上层目录的 utils.datetime 模块中导入 the_same_day 函数
from ..utils.datetime import the_same_day
# 从上层目录的 utils.device 模块中导入 Device 类
from ..utils.device import Device
# 从上层目录的 utils.log 模块中导入 logger 对象
from ..utils.log import logger
# 从上层目录的 utils.param 模块中导入 operation_times, parse_operation_params 函数
from ..utils.param import operation_times, parse_operation_params
# 从上层目录的 utils.priority_queue 模块中导入 PriorityQueue 类
from ..utils.priority_queue import PriorityQueue
# 从上层目录的 utils.recognize 模块中导入 Recognizer 类
from ..utils.recognize import Recognizer
# 从上层目录的 utils.solver 模块中导入 BaseSolver 类
from ..utils.solver import BaseSolver
# 从上层目录的 utils.typealias 模块中导入 ParamArgs 类型别名
from ..utils.typealias import ParamArgs
# 从上层目录的 utils.yaml 模块中导入 yaml 对象
from ..utils.yaml import yaml
# 从当前目录的 operation 模块中导入 OpeSolver 类
from .operation import OpeSolver

# 定义任务优先级字典
task_priority = {'base': 0, 'recruit': 1, 'mail': 2,
                 'credit': 3, 'shop': 4, 'mission': 5, 'operation': 6}


# 定义 ScheduleLogError 类，继承自 ValueError
class ScheduleLogError(ValueError):
    """ Schedule log 文件解析错误 """


# 定义 operation_one 函数，参数为 args 和 device，返回值为布尔类型
def operation_one(args: ParamArgs = [], device: Device = None) -> bool:
    """
    只为 schedule 模块使用的单次作战操作
    目前不支持使用源石和体力药

    返回值表示该次作战是否成功
    完成剿灭不算成功
    """
    # 解析操作参数
    level, _, _, _, eliminate = parse_operation_params(args)
    # 运行作战，获取剩余计划
    remain_plan = OpeSolver(device).run(level, 1, 0, 0, eliminate)
    # 遍历剩余计划
    for plan in remain_plan:
        # 如果计划次数不为 0，则作战失败，返回 False
        if plan[1] != 0:
            return False
    # 所有计划次数为 0，作战成功，返回 True
    return True


# 使用 yaml_object 装饰器，将 Task 类注册为 yaml 对象
@yaml_object(yaml)
# 定义 Task 类
class Task(object):
    """
    单个任务
    """

    # 初始化方法，参数为 tag、cmd、args 和 device
    def __init__(self, tag: str = '', cmd: Callable = None, args: ParamArgs = [], device: Device = None):
        # 设置任务的命令和参数
        self.cmd = cmd
        self.cmd_args = args
        # 设置任务的标签、上次运行时间、索引、是否挂起、总次数、完成次数和设备
        self.tag = tag
        self.last_run = None
        self.idx = None
        self.pending = False
        self.total = 1
        self.finish = 0
        self.device = device

        # 如果标签为 'per_hour'，则设置上次运行时间为当前时间
        if tag == 'per_hour':
            self.last_run = datetime.datetime.now()
        # 如果命令名称为 'operation'，则设置总次数为操作次数，并断言总次数不为 0
        if cmd.__name__ == 'operation':
            self.total = operation_times(args)
            assert self.total != 0

    # 类方法
    @classmethod
    # 将数据转换为 YAML 格式
    def to_yaml(cls, representer, data):
        # 初始化最后运行时间为空字符串
        last_run = ''
        # 如果数据的最后运行时间不为空，则格式化为指定格式的字符串
        if data.last_run is not None:
            last_run = data.last_run.strftime('%Y-%m-%d %H:%M:%S')
        # 返回表示任务的映射
        return representer.represent_mapping('task',
                                             {'tag': data.tag,
                                              'cmd': data.cmd.__name__,
                                              'cmd_args': data.cmd_args,
                                              'last_run': last_run,
                                              'idx': data.idx,
                                              'pending': data.pending,
                                              'total': data.total,
                                              'finish': data.finish})

    # 比较任务的优先级
    def __lt__(self, other):
        # 如果任务的命令名称不相等，则比较它们的优先级
        if task_priority[self.cmd.__name__] != task_priority[other.cmd.__name__]:
            return task_priority[self.cmd.__name__] < task_priority[other.cmd.__name__]
        # 否则比较它们的索引
        return self.idx < other.idx

    # 加载任务的属性
    def load(self, last_run: str = '', idx: int = 0, pending: bool = False, total: int = 1, finish: int = 0):
        # 如果最后运行时间为空字符串，则将任务的最后运行时间设置为 None
        if last_run == '':
            self.last_run = None
        # 否则将字符串格式的最后运行时间转换为 datetime 对象
        else:
            self.last_run = datetime.datetime.strptime(
                last_run, '%Y-%m-%d %H:%M:%S')
        # 设置任务的索引、挂起状态、总数和完成数
        self.idx = idx
        self.pending = pending
        self.total = total
        self.finish = finish

    # 重置任务的属性
    def reset(self):
        # 如果任务的标签不是 'per_hour'，则将最后运行时间设置为 None
        if self.tag != 'per_hour':
            self.last_run = None
        # 将任务的挂起状态设置为 False，完成数设置为 0
        self.pending = False
        self.finish = 0

    # 设置任务的索引
    def set_idx(self, idx: int = None):
        # 将任务的索引设置为指定值
        self.idx = idx

    # 判断任务是否为启动任务
    def start_up(self) -> bool:
        # 返回任务的标签是否为 'start_up' 的布尔值
        return self.tag == 'start_up'
    # 检查是否需要运行任务
    def need_run(self, now: datetime.datetime = datetime.datetime.now()) -> bool:
        # 如果任务已经在等待中，则不需要运行
        if self.pending:
            return False
        # 如果任务需要启动
        if self.start_up():
            # 如果上次运行时间不为空，则不需要运行
            if self.last_run is not None:
                return False
            # 设置任务为等待状态，更新上次运行时间，并返回需要运行
            self.pending = True
            self.last_run = now
            return True
        # 如果任务标签以'day_'开头
        if self.tag[:4] == 'day_':
            # 如果上次运行时间不为空且与当前时间在同一天，则不需要运行
            if self.last_run is not None and the_same_day(now, self.last_run):
                return False
            # 如果当前时间还未到达设定的运行时间，则不需要运行
            if now.strftime('%H:%M') < self.tag.replace('_', ':')[4:]:
                return False
            # 设置任务为等待状态，更新上次运行时间，并返回需要运行
            self.pending = True
            self.last_run = now
            return True
        # 如果任务标签为'per_hour'
        if self.tag == 'per_hour':
            # 如果上次运行时间加上一小时小于等于当前时间，则设置任务为等待状态，更新上次运行时间，并返回需要运行
            if self.last_run + datetime.timedelta(hours=1) <= now:
                self.pending = True
                self.last_run = now
                return True
            # 否则不需要运行
            return False
        # 其他情况均不需要运行
        return False

    # 执行任务
    def run(self) -> bool:
        # 记录任务信息
        logger.info(f'task: {self.cmd.__name__} {self.cmd_args}')
        # 如果任务为'operation'，则执行相应操作
        if self.cmd.__name__ == 'operation':
            # 如果操作成功，则更新完成数量，如果完成数量等于总数，则重置完成数量和等待状态，并返回需要运行
            if operation_one(self.cmd_args, self.device):
                self.finish += 1
                if self.finish == self.total:
                    self.finish = 0
                    self.pending = False
                    return True
            # 否则不需要运行
            return False
        # 否则执行相应命令，并设置任务为非等待状态，返回需要运行
        self.cmd(self.cmd_args, self.device)
        self.pending = False
        return True
# 定义一个比较函数，用于初始化任务的比较，返回值为整数类型
def cmp_for_init(task_a: Task = None, task_b: Task = None) -> int:
    # 如果两个任务都启动成功，则返回0
    if task_a.start_up() and task_b.start_up():
        return 0

    # 如果只有任务A启动成功，则返回-1
    if task_a.start_up():
        return -1

    # 如果只有任务B启动成功，则返回1
    if task_b.start_up():
        return 1
    # 如果都没有启动成功，则返回0
    return 0


# 使用yaml_object装饰器，将类ScheduleSolver注册为yaml对象
@yaml_object(yaml)
class ScheduleSolver(BaseSolver):
    """
    按照计划定时、自动完成任务
    """

    # 初始化方法，接受设备和识别器作为参数
    def __init__(self, device: Device = None, recog: Recognizer = None) -> None:
        # 调用父类的初始化方法
        super().__init__(device, recog)
        # 初始化任务列表
        self.tasks = []
        # 初始化待处理任务队列
        self.pending_list = PriorityQueue()
        # 初始化设备
        self.device = device
        # 初始化上次运行时间
        self.last_run = None
        # 初始化日程日志路径
        self.schedule_log_path = Path(config.LOGFILE_PATH).joinpath('schedule.log')

    # 类方法，将对象转换为yaml格式
    @classmethod
    def to_yaml(cls, representer, data):
        return representer.represent_mapping('Schedule', {'last_run': data.last_run.strftime('%Y-%m-%d %H:%M:%S'),
                                                          'tasks': data.tasks})

    # 将对象存储到磁盘
    def dump_to_disk(self):
        with self.schedule_log_path.open('w', encoding='utf8') as f:
            yaml.dump(self, f)
        # 记录日志信息
        logger.info('计划已存档')
    # 从磁盘加载数据，根据命令列表和匹配器返回布尔值
    def load_from_disk(self, cmd_list=None, matcher: Callable = None) -> bool:
        # 如果命令列表为空，则初始化为空列表
        if cmd_list is None:
            cmd_list = []
        try:
            # 以只读方式打开日程日志文件
            with self.schedule_log_path.open('r', encoding='utf8') as f:
                # 从文件中加载数据
                data = yaml.load(f)
            # 将最后运行时间转换为日期时间格式
            self.last_run = datetime.datetime.strptime(
                data['last_run'], '%Y-%m-%d %H:%M:%S')
            # 遍历任务列表
            for task in data['tasks']:
                # 使用匹配器匹配命令，如果匹配不到则抛出异常
                cmd = matcher(task['cmd'], cmd_list)
                if cmd is None:
                    raise ScheduleLogError
                # 创建新的任务对象
                new_task = Task(
                    task['tag'], cmd, task['cmd_args'], self.device
                )
                # 加载任务的相关信息
                new_task.load(
                    task['last_run'], task['idx'], task['pending'], task['total'], task['finish']
                )
                # 将任务添加到任务列表中
                self.tasks.append(new_task)
                # 如果任务是待执行状态，则将其加入待执行列表
                if new_task.pending:
                    self.pending_list.push(new_task)
        except Exception:
            # 加载失败则返回 False
            return False
        # 加载成功则记录日志并返回 True
        logger.info('发现中断的计划，将继续执行')
        return True

    # 添加新任务到任务列表中
    def add_task(self, tag: str = '', cmd: Callable = None, args: ParamArgs = []):
        task = Task(tag, cmd, args, self.device)
        self.tasks.append(task)

    # 对任务列表进行排序，处理优先级相同的情况
    def per_run(self):
        """
        这里是为了处理优先级相同的情况，对于优先级相同时，我们依次考虑：
        1. start_up 优先执行
        2. 按照配置文件的顺序决定先后顺序

        sort 是稳定排序，详见:
        https://docs.python.org/3/library/functions.html#sorted
        """
        self.tasks.sort(key=cmp_to_key(cmp_for_init))
        for idx, task in enumerate(self.tasks):
            task.set_idx(idx)

    # 运行任务
    def run(self):
        logger.info('Start: 计划')

        super().run()

    # 处理新的一天开始时的操作
    def new_day(self):
        for task in self.tasks:
            task.reset()
        self.pending_list = PriorityQueue()
    # 定义一个方法，用于执行状态转换
    def transition(self) -> None:
        # 进入循环，持续执行状态转换
        while True:
            # 获取当前时间
            now = datetime.datetime.now()
            # 如果上次运行时间不为空，并且不是同一天
            if self.last_run is not None and the_same_day(self.last_run, now) is False:
                # 执行新的一天操作
                self.new_day()
            # 更新上次运行时间为当前时间
            self.last_run = now
            # 遍历任务列表，将需要执行的任务加入待执行列表
            for task in self.tasks:
                if task.need_run(now):
                    self.pending_list.push(task)

            # 从待执行列表中取出一个任务
            task = self.pending_list.pop()
            # 如果任务不为空，并且执行失败，则重新加入待执行列表
            if task is not None and task.run() is False:
                self.pending_list.push(task)

            # 将状态信息保存到磁盘
            self.dump_to_disk()
            # 休眠60秒
            time.sleep(60)
```