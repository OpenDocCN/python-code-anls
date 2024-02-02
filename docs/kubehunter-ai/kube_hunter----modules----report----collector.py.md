# `kubehunter\kube_hunter\modules\report\collector.py`

```py
# 导入 logging 模块
import logging
# 导入 threading 模块
import threading

# 从 kube_hunter.conf 模块中导入 config 对象
from kube_hunter.conf import config
# 从 kube_hunter.core.events 模块中导入 handler 对象
from kube_hunter.core.events import handler
# 从 kube_hunter.core.events.types 模块中导入 Event, Service, Vulnerability, HuntFinished, HuntStarted, ReportDispatched 类
from kube_hunter.core.events.types import (
    Event,
    Service,
    Vulnerability,
    HuntFinished,
    HuntStarted,
    ReportDispatched,
)

# 获取名为 __name__ 的 logger 对象
logger = logging.getLogger(__name__)

# 创建全局变量 services_lock，并赋值为 threading.Lock() 对象
global services_lock
services_lock = threading.Lock()
# 创建全局变量 services，并赋值为空列表
services = list()

# 创建全局变量 vulnerabilities_lock，并赋值为 threading.Lock() 对象
global vulnerabilities_lock
vulnerabilities_lock = threading.Lock()
# 创建全局变量 vulnerabilities，并赋值为空列表
vulnerabilities = list()

# 获取所有的 hunter 对象，并赋值给 hunters 变量
hunters = handler.all_hunters

# 使用 handler.subscribe 装饰器订阅 Service 事件
@handler.subscribe(Service)
# 使用 handler.subscribe 装饰器订阅 Vulnerability 事件
@handler.subscribe(Vulnerability)
# 定义 Collector 类
class Collector(object):
    # 初始化方法，接受 event 参数
    def __init__(self, event=None):
        self.event = event

    # 执行方法，用于收集数据时调用
    def execute(self):
        """function is called only when collecting data"""
        # 声明使用全局变量 services
        global services
        # 声明使用全局变量 vulnerabilities
        global vulnerabilities
        # 获取当前事件的类继承关系
        bases = self.event.__class__.__mro__
        # 如果当前事件是 Service 类的子类
        if Service in bases:
            # 使用 services_lock 锁，将当前事件添加到 services 列表中
            with services_lock:
                services.append(self.event)
            # 记录日志，表示发现了开放的服务
            logger.info(f'Found open service "{self.event.get_name()}" at {self.event.location()}')
        # 如果当前事件是 Vulnerability 类的子类
        elif Vulnerability in bases:
            # 使用 vulnerabilities_lock 锁，将当前事件添加到 vulnerabilities 列表中
            with vulnerabilities_lock:
                vulnerabilities.append(self.event)
            # 记录日志，表示发现了漏洞
            logger.info(f'Found vulnerability "{self.event.get_name()}" in {self.event.location()}')

# 定义 TablesPrinted 类，继承自 Event 类
class TablesPrinted(Event):
    pass

# 使用 handler.subscribe 装饰器订阅 HuntFinished 事件
@handler.subscribe(HuntFinished)
# 定义 SendFullReport 类
class SendFullReport(object):
    # 初始化方法，接受 event 参数
    def __init__(self, event):
        self.event = event

    # 执行方法，用于发送完整报告
    def execute(self):
        # 生成报告，并通过 config.dispatcher 发送报告
        report = config.reporter.get_report(statistics=config.statistics, mapping=config.mapping)
        config.dispatcher.dispatch(report)
        # 发布 ReportDispatched 事件
        handler.publish_event(ReportDispatched())
        # 发布 TablesPrinted 事件
        handler.publish_event(TablesPrinted())

# 使用 handler.subscribe 装饰器订阅 HuntStarted 事件
@handler.subscribe(HuntStarted)
# 定义 StartedInfo 类
class StartedInfo(object):
    # 初始化方法，接受 event 参数
    def __init__(self, event):
        self.event = event

    # 执行方法，用于记录开始信息
    def execute(self):
        # 记录日志，表示开始进行漏洞扫描
        logger.info("Started hunting")
        # 记录日志，表示正在发现开放的 Kubernetes 服务
        logger.info("Discovering Open Kubernetes Services")
```