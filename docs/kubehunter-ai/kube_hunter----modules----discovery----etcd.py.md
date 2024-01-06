# `kubehunter\kube_hunter\modules\discovery\etcd.py`

```
# 从 kube_hunter.core.events 模块中导入 handler
# 从 kube_hunter.core.events.types 模块中导入 Event, OpenPortEvent, Service
# 从 kube_hunter.core.types 模块中导入 Discovery
from kube_hunter.core.events import handler
from kube_hunter.core.events.types import Event, OpenPortEvent, Service
from kube_hunter.core.types import Discovery

# 创建 EtcdAccessEvent 类，继承自 Service 和 Event 类
# 用于表示 Etcd 数据库的访问事件
class EtcdAccessEvent(Service, Event):
    """Etcd is a DB that stores cluster's data, it contains configuration and current
    state information, and might contain secrets"""

    # 初始化方法
    def __init__(self):
        # 调用父类 Service 的初始化方法，设置服务名称为 "Etcd"
        Service.__init__(self, name="Etcd")

# 使用 handler.subscribe 装饰器注册 OpenPortEvent 事件的处理函数
# 通过 predicate 参数指定端口号为 2379 的事件
class EtcdRemoteAccess(Discovery):
    """Etcd service
    check for the existence of etcd service
    """

    # 初始化方法
    def __init__(self, event):
```
这段代码定义了两个类，一个是用于表示 Etcd 数据库访问事件的 EtcdAccessEvent 类，另一个是用于发现 Etcd 服务的 EtcdRemoteAccess 类。同时使用了装饰器来注册事件处理函数。
# 设置类的属性 event 为传入的 event 参数
self.event = event

# 执行方法，发布 EtcdAccessEvent 事件
def execute(self):
    self.publish_event(EtcdAccessEvent())
```