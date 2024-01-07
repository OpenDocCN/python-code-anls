# `.\kubehunter\kube_hunter\modules\report\collector.py`

```

# 导入 logging 和 threading 模块
import logging
import threading

# 从 kube_hunter.conf 模块中导入 config 变量
from kube_hunter.conf import config
# 从 kube_hunter.core.events 模块中导入 handler 变量
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

# 创建全局变量 services_lock 和 vulnerabilities_lock，并分别初始化为 threading.Lock() 对象
global services_lock
services_lock = threading.Lock()
global vulnerabilities_lock
vulnerabilities_lock = threading.Lock()

# 创建全局变量 services 和 vulnerabilities，并分别初始化为空列表
services = list()
vulnerabilities = list()

# 获取所有的 hunter 对象，并赋值给 hunters 变量
hunters = handler.all_hunters

# 使用 handler.subscribe 装饰器订阅 Service 和 Vulnerability 事件
@handler.subscribe(Service)
@handler.subscribe(Vulnerability)
class Collector(object):
    def __init__(self, event=None):
        self.event = event

    # 定义 execute 方法，用于处理收集数据
    def execute(self):
        """function is called only when collecting data"""
        # 声明使用全局变量 services 和 vulnerabilities
        global services
        global vulnerabilities
        # 获取当前事件的类继承链
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
class SendFullReport(object):
    def __init__(self, event):
        self.event = event

    # 定义 execute 方法，用于发送完整报告
    def execute(self):
        # 生成报告，并通过 config.dispatcher 发送报告
        report = config.reporter.get_report(statistics=config.statistics, mapping=config.mapping)
        config.dispatcher.dispatch(report)
        # 发布 ReportDispatched 和 TablesPrinted 事件
        handler.publish_event(ReportDispatched())
        handler.publish_event(TablesPrinted())

# 使用 handler.subscribe 装饰器订阅 HuntStarted 事件
@handler.subscribe(HuntStarted)
class StartedInfo(object):
    def __init__(self, event):
        self.event = event

    # 定义 execute 方法，用于记录开始信息
    def execute(self):
        # 记录日志，表示开始了探测
        logger.info("Started hunting")
        logger.info("Discovering Open Kubernetes Services")

```