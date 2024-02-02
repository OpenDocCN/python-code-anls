# `kubehunter\kube_hunter\core\events\handler.py`

```py
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

# 从 kube_hunter.conf 模块中导入 config 对象
from kube_hunter.conf import config
# 从 kube_hunter.core.types 模块中导入 ActiveHunter 和 HunterBase 类
from kube_hunter.core.types import ActiveHunter, HunterBase
# 从 kube_hunter.core.events.types 模块中导入 Vulnerability 和 EventFilterBase 类
from kube_hunter.core.events.types import Vulnerability, EventFilterBase

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 继承自 Queue 对象，用于异步处理事件
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

        # 初始化钩子字典
        self.hooks = defaultdict(list)
        # 初始化过滤器字典
        self.filters = defaultdict(list)
        # 设置运行状态为 True
        self.running = True
        # 初始化工作线程列表
        self.workers = list()

        # 创建指定数量的工作线程
        for _ in range(num_worker):
            t = Thread(target=self.worker)
            t.daemon = True
            t.start()
            self.workers.append(t)

        # 创建通知线程
        t = Thread(target=self.notifier)
        t.daemon = True
        t.start()

    # 用于装饰的订阅方法
    def subscribe(self, event, hook=None, predicate=None):
        def wrapper(hook):
            self.subscribe_event(event, hook=hook, predicate=predicate)
            return hook

        return wrapper

    # 用于装饰的一次性订阅方法
    def subscribe_once(self, event, hook=None, predicate=None):
        def wrapper(hook):
            # 安装一个 __new__ 魔术方法到猎手上，用于在创建时从列表中移除猎手
            def __new__unsubscribe_self(self, cls):
                handler.hooks[event].remove((hook, predicate))
                return object.__new__(self)

            hook.__new__ = __new__unsubscribe_self

            self.subscribe_event(event, hook=hook, predicate=predicate)
            return hook

        return wrapper

    # 获取未实例化的事件对象
    # 订阅事件，将钩子函数和条件添加到相应的事件中
    def subscribe_event(self, event, hook=None, predicate=None):
        # 如果钩子函数在 ActiveHunter 的方法解析顺序中
        if ActiveHunter in hook.__mro__:
            # 如果配置为非活跃状态，则返回
            if not config.active:
                return
            # 将活跃状态的钩子函数添加到活跃猎手字典中
            self.active_hunters[hook] = hook.__doc__
        # 如果钩子函数在 HunterBase 的方法解析顺序中
        elif HunterBase in hook.__mro__:
            # 将钩子函数添加到被动猎手字典中
            self.passive_hunters[hook] = hook.__doc__

        # 如果钩子函数在 HunterBase 的方法解析顺序中
        if HunterBase in hook.__mro__:
            # 将钩子函数添加到所有猎手字典中
            self.all_hunters[hook] = hook.__doc__

        # 注册过滤器
        if EventFilterBase in hook.__mro__:
            # 如果钩子函数不在事件的过滤器列表中，则添加到列表中
            if hook not in self.filters[event]:
                self.filters[event].append((hook, predicate))
                logger.debug(f"{hook} filter subscribed to {event}")

        # 注册猎手函数
        elif hook not in self.hooks[event]:
            # 如果钩子函数不在事件的钩子函数列表中，则添加到列表中
            self.hooks[event].append((hook, predicate))
            logger.debug(f"{hook} subscribed to {event}")

    # 应用过滤器
    def apply_filters(self, event):
        # 如果有订阅的过滤器，对事件应用过滤器
        for hooked_event in self.filters.keys():
            # 如果事件的类在过滤器的键中
            if hooked_event in event.__class__.__mro__:
                # 遍历过滤器列表，应用过滤器
                for filter_hook, predicate in self.filters[hooked_event]:
                    if predicate and not predicate(event):
                        continue

                    logger.debug(f"Event {event.__class__} filtered with {filter_hook}")
                    event = filter_hook(event).execute()
                    # 如果过滤器决定移除事件，则返回 None
                    if not event:
                        return None
        return event

    # 获取实例化的事件对象
    # 发布事件到订阅者
    def publish_event(self, event, caller=None):
        # 设置事件链
        if caller:
            event.previous = caller.event
            event.hunter = caller.__class__

        # 在发布事件到订阅者之前，对事件应用过滤器
        # 如果过滤器返回 None，则不继续发布
        event = self.apply_filters(event)
        if event:
            # 如果事件被重写，确保它链接到其父事件（'previous'）
            if caller:
                event.previous = caller.event
                event.hunter = caller.__class__

            # 遍历钩子事件，检查是否在事件的类层次结构中
            for hooked_event in self.hooks.keys():
                if hooked_event in event.__class__.__mro__:
                    for hook, predicate in self.hooks[hooked_event]:
                        if predicate and not predicate(event):
                            continue

                        # 如果配置了统计信息，并且有调用者
                        if config.statistics and caller:
                            # 如果事件属于 Vulnerability 类的子类
                            if Vulnerability in event.__class__.__mro__:
                                # 增加调用者类的已发布漏洞数量
                                caller.__class__.publishedVulnerabilities += 1

                        # 记录事件发布的调试信息
                        logger.debug(f"Event {event.__class__} got published with {event}")
                        # 将事件传递给钩子并执行
                        self.put(hook(event))

    # 作为守护线程在专用线程上执行回调
    def worker(self):
        while self.running:
            try:
                # 从队列中获取钩子并执行
                hook = self.get()
                logger.debug(f"Executing {hook.__class__} with {hook.event.__dict__}")
                hook.execute()
            except Exception as ex:
                # 记录异常信息
                logger.debug(ex, exc_info=True)
            finally:
                # 标记任务完成
                self.task_done()
        # 记录线程关闭信息
        logger.debug("closing thread...")
    # 定义一个通知器方法，用于通知任务状态
    def notifier(self):
        # 等待2秒
        time.sleep(2)
        # 在未完成任务数大于0时循环
        while self.unfinished_tasks > 0:
            # 记录未完成任务数
            logger.debug(f"{self.unfinished_tasks} tasks left")
            # 等待3秒
            time.sleep(3)
            # 如果未完成任务数为1，记录最终挂起的钩子
            if self.unfinished_tasks == 1:
                logger.debug("final hook is hanging")
    
    # 停止所有守护进程的执行
    def free(self):
        # 将运行状态设置为False
        self.running = False
        # 使用互斥锁清空队列
        with self.mutex:
            self.queue.clear()
# 创建一个事件队列对象，设置队列的最大容量为800
handler = EventQueue(800)
```