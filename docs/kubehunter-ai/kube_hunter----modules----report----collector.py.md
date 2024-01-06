# `kubehunter\kube_hunter\modules\report\collector.py`

```
# 导入日志和线程模块
import logging
import threading

# 从 kube_hunter.conf 模块中导入配置
from kube_hunter.conf import config
# 从 kube_hunter.core.events 模块中导入事件处理器
from kube_hunter.core.events import handler
# 从 kube_hunter.core.events.types 模块中导入事件、服务、漏洞、搜索完成、搜索开始、报告分发等类型
from kube_hunter.core.events.types import (
    Event,
    Service,
    Vulnerability,
    HuntFinished,
    HuntStarted,
    ReportDispatched,
)

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 创建全局变量 services_lock，并初始化为线程锁对象
global services_lock
services_lock = threading.Lock()
# 创建全局变量 services，并初始化为空列表
services = list()
# 定义全局变量vulnerabilities_lock，用于线程同步
global vulnerabilities_lock
# 创建一个线程锁对象
vulnerabilities_lock = threading.Lock()
# 创建一个空的漏洞列表
vulnerabilities = list()

# 获取所有的hunter对象
hunters = handler.all_hunters

# 订阅Service和Vulnerability事件
@handler.subscribe(Service)
@handler.subscribe(Vulnerability)
class Collector(object):
    def __init__(self, event=None):
        self.event = event

    def execute(self):
        """function is called only when collecting data"""
        # 声明全局变量services和vulnerabilities
        global services
        global vulnerabilities
        # 获取当前事件类的所有父类
        bases = self.event.__class__.__mro__
        # 如果当前事件是Service类的实例
        if Service in bases:
            # 使用services_lock进行线程同步
            with services_lock:
# 将事件添加到服务列表中
services.append(self.event)
# 记录日志，显示找到的开放服务的名称和位置
logger.info(f'Found open service "{self.event.get_name()}" at {self.event.location()}')
# 如果事件的基类是 Vulnerability，则将事件添加到漏洞列表中
elif Vulnerability in bases:
    # 使用漏洞锁，将事件添加到漏洞列表中
    with vulnerabilities_lock:
        vulnerabilities.append(self.event)
    # 记录日志，显示找到的漏洞的名称和位置
    logger.info(f'Found vulnerability "{self.event.get_name()}" in {self.event.location()}')

# 定义一个名为 TablesPrinted 的事件类，继承自 Event 类

# 使用 handler.subscribe 装饰器，将 SendFullReport 类与 HuntFinished 事件关联起来
class SendFullReport(object):
    # 初始化方法，接收一个事件对象
    def __init__(self, event):
        self.event = event

    # 执行方法，生成报告并将其分发
    def execute(self):
        # 从配置中获取报告生成器，生成报告并传入统计和映射参数
        report = config.reporter.get_report(statistics=config.statistics, mapping=config.mapping)
        # 使用配置中的分发器，将报告分发出去
        config.dispatcher.dispatch(report)
# 调用 handler 对象的 publish_event 方法，发布 ReportDispatched 事件
handler.publish_event(ReportDispatched())
# 调用 handler 对象的 publish_event 方法，发布 TablesPrinted 事件
handler.publish_event(TablesPrinted())

# 使用 handler 对象的 subscribe 装饰器，订阅 HuntStarted 事件，并创建 StartedInfo 类
@handler.subscribe(HuntStarted)
class StartedInfo(object):
    # 初始化方法，接收事件对象并保存在 self.event 中
    def __init__(self, event):
        self.event = event

    # 执行方法，记录日志信息 "Started hunting" 和 "Discovering Open Kubernetes Services"
    def execute(self):
        logger.info("Started hunting")
        logger.info("Discovering Open Kubernetes Services")
```