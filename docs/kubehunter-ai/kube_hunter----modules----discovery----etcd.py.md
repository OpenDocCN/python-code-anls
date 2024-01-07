# `.\kubehunter\kube_hunter\modules\discovery\etcd.py`

```

# 从 kube_hunter.core.events 模块中导入 handler 函数
from kube_hunter.core.events import handler
# 从 kube_hunter.core.events.types 模块中导入 Event, OpenPortEvent, Service 类
from kube_hunter.core.events.types import Event, OpenPortEvent, Service
# 从 kube_hunter.core.types 模块中导入 Discovery 类
from kube_hunter.core.types import Discovery

# 创建 EtcdAccessEvent 类，继承自 Service 和 Event 类
class EtcdAccessEvent(Service, Event):
    """Etcd is a DB that stores cluster's data, it contains configuration and current
    state information, and might contain secrets"""

    # 初始化函数
    def __init__(self):
        # 调用父类 Service 的初始化函数，设置服务名称为 "Etcd"
        Service.__init__(self, name="Etcd")

# 使用 handler.subscribe 装饰器注册 OpenPortEvent 事件的处理函数，端口号为 2379
@handler.subscribe(OpenPortEvent, predicate=lambda p: p.port == 2379)
# 创建 EtcdRemoteAccess 类，继承自 Discovery 类
class EtcdRemoteAccess(Discovery):
    """Etcd service
    check for the existence of etcd service
    """

    # 初始化函数
    def __init__(self, event):
        # 设置事件属性
        self.event = event

    # 执行函数
    def execute(self):
        # 发布 EtcdAccessEvent 事件
        self.publish_event(EtcdAccessEvent())

```