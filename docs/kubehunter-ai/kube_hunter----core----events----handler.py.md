# `kubehunter\kube_hunter\core\events\handler.py`

```
# 导入日志模块
import logging
# 导入时间模块
import time
# 导入默认字典模块
from collections import defaultdict
# 导入队列模块
from queue import Queue
# 导入线程模块
from threading import Thread

# 导入配置模块
from kube_hunter.conf import config
# 导入核心类型模块
from kube_hunter.core.types import ActiveHunter, HunterBase
# 导入事件类型模块
from kube_hunter.core.events.types import Vulnerability, EventFilterBase

# 获取日志记录器
logger = logging.getLogger(__name__)

# 定义一个继承自队列对象的事件队列类，用于异步处理事件
class EventQueue(Queue, object):
    # 初始化方法
    def __init__(self, num_worker=10):
        # 调用父类的初始化方法
        super(EventQueue, self).__init__()
        # 初始化被动猎手字典
        self.passive_hunters = dict()
        # 初始化主动猎手字典
        self.active_hunters = dict()
        # 初始化所有猎手字典
        self.all_hunters = dict()
# 创建一个默认值为列表的字典，用于存储钩子函数
self.hooks = defaultdict(list)
# 创建一个默认值为列表的字典，用于存储过滤器函数
self.filters = defaultdict(list)
# 设置运行状态为True
self.running = True
# 创建一个空的线程列表
self.workers = list()

# 根据指定的工作线程数量创建线程并启动
for _ in range(num_worker):
    t = Thread(target=self.worker)  # 创建一个线程，目标函数为self.worker
    t.daemon = True  # 将线程设置为守护线程
    t.start()  # 启动线程
    self.workers.append(t)  # 将线程添加到workers列表中

# 创建一个线程，目标函数为self.notifier，并启动
t = Thread(target=self.notifier)
t.daemon = True
t.start()

# 用于订阅事件的装饰器，用于简化订阅过程
def subscribe(self, event, hook=None, predicate=None):
    # 定义一个装饰器函数，用于实际执行订阅操作
    def wrapper(hook):
        # 调用subscribe_event方法进行事件订阅
        self.subscribe_event(event, hook=hook, predicate=predicate)
# 返回一个装饰器函数，用于订阅事件
def subscribe_once(self, event, hook=None, predicate=None):
    # 定义一个内部函数，用于处理订阅一次的机制
    def wrapper(hook):
        # 定义一个内部函数，安装一个 __new__ 魔术方法在 hunter 上
        # 该方法在创建时将从列表中移除 hunter
        def __new__unsubscribe_self(self, cls):
            handler.hooks[event].remove((hook, predicate))
            return object.__new__(self)

        # 将 __new__unsubscribe_self 方法安装到 hook 对象上
        hook.__new__ = __new__unsubscribe_self

        # 调用 subscribe_event 方法订阅事件
        self.subscribe_event(event, hook=hook, predicate=predicate)
        return hook

    return wrapper
# 定义一个订阅事件的方法，接受事件、钩子和条件作为参数
def subscribe_event(self, event, hook=None, predicate=None):
    # 如果钩子是 ActiveHunter 的子类
    if ActiveHunter in hook.__mro__:
        # 如果配置不是活跃的，直接返回
        if not config.active:
            return
        # 将活跃的猎手添加到活跃猎手字典中
        self.active_hunters[hook] = hook.__doc__
    # 如果钩子是 HunterBase 的子类
    elif HunterBase in hook.__mro__:
        # 将被动猎手添加到被动猎手字典中
        self.passive_hunters[hook] = hook.__doc__

    # 将所有猎手添加到所有猎手字典中
    if HunterBase in hook.__mro__:
        self.all_hunters[hook] = hook.__doc__

    # 注册过滤器
    if EventFilterBase in hook.__mro__:
        # 如果钩子不在事件的过滤器列表中，将其添加进去
        if hook not in self.filters[event]:
            self.filters[event].append((hook, predicate))
            logger.debug(f"{hook} filter subscribed to {event}")

    # 注册猎手
    elif hook not in self.hooks[event]:
    # 将钩子和条件添加到事件的钩子列表中
    self.hooks[event].append((hook, predicate))
    # 记录调试信息，显示哪个钩子订阅了哪个事件
    logger.debug(f"{hook} subscribed to {event}")

# 应用过滤器
def apply_filters(self, event):
    # 如果有过滤器订阅了事件，就在事件上应用它们
    for hooked_event in self.filters.keys():
        # 如果事件的类在过滤器的键中
        if hooked_event in event.__class__.__mro__:
            # 遍历过滤器钩子和条件
            for filter_hook, predicate in self.filters[hooked_event]:
                # 如果有条件，并且条件不满足，则继续下一个过滤器
                if predicate and not predicate(event):
                    continue

                # 记录调试信息，显示哪个事件被哪个过滤器过滤
                logger.debug(f"Event {event.__class__} filtered with {filter_hook}")
                # 执行过滤器钩子，并将结果赋给事件
                event = filter_hook(event).execute()
                # 如果过滤器决定移除事件，则返回 None
                if not event:
                    return None
    return event

# 发布事件
# 获取实例化的事件对象
def publish_event(self, event, caller=None):
# 设置事件链
# 如果有调用者，则将调用者的事件设置为当前事件的上一个事件，将调用者的类设置为当前事件的hunter
if caller:
    event.previous = caller.event
    event.hunter = caller.__class__

# 在将事件发布给订阅者之前，对事件应用过滤器
# 如果过滤器返回None，则不继续发布
event = self.apply_filters(event)
if event:
    # 如果事件被重写，确保它与其父事件（'previous'）相关联
    if caller:
        event.previous = caller.event
        event.hunter = caller.__class__

    # 遍历已经挂钩的事件
    for hooked_event in self.hooks.keys():
        # 如果挂钩的事件在当前事件的类层次结构中
        if hooked_event in event.__class__.__mro__:
            # 遍历挂钩的事件的钩子和谓词
            for hook, predicate in self.hooks[hooked_event]:
                # 如果有谓词，并且谓词不满足，则继续下一个钩子
                if predicate and not predicate(event):
                    continue
# 如果配置了统计信息并且存在调用者，则执行以下操作
if config.statistics and caller:
    # 如果事件属于 Vulnerability 类或其父类，则调用者的 publishedVulnerabilities 属性加一
    if Vulnerability in event.__class__.__mro__:
        caller.__class__.publishedVulnerabilities += 1

# 记录调试信息，说明事件被发布，并打印事件的类和内容
logger.debug(f"Event {event.__class__} got published with {event}")
# 将事件传递给回调函数，并将返回值放入队列中
self.put(hook(event))

# 在专用线程上作为守护线程执行回调函数
def worker(self):
    while self.running:
        try:
            # 从队列中获取回调函数
            hook = self.get()
            # 记录调试信息，说明正在执行特定的回调函数，并打印事件的属性
            logger.debug(f"Executing {hook.__class__} with {hook.event.__dict__}")
            # 执行回调函数
            hook.execute()
        except Exception as ex:
            # 记录异常信息
            logger.debug(ex, exc_info=True)
        finally:
            # 标记任务完成
            self.task_done()
    # 记录调试信息，说明线程即将关闭
    logger.debug("closing thread...")
# 定义一个通知器方法，用于通知任务的执行情况
def notifier(self):
    # 等待2秒
    time.sleep(2)
    # 在未完成任务数大于0时循环
    while self.unfinished_tasks > 0:
        # 记录未完成任务数
        logger.debug(f"{self.unfinished_tasks} tasks left")
        # 等待3秒
        time.sleep(3)
        # 如果未完成任务数为1，记录最终挂钩正在挂起
        if self.unfinished_tasks == 1:
            logger.debug("final hook is hanging")

# 停止所有守护进程的执行
def free(self):
    # 将运行状态设置为False
    self.running = False
    # 在互斥锁上加锁
    with self.mutex:
        # 清空队列
        self.queue.clear()

# 创建一个事件队列对象，初始大小为800
handler = EventQueue(800)
```