# `kubehunter\tests\core\test_subscribe.py`

```
# 导入时间模块
import time

# 从 kube_hunter.core.types 模块中导入 Hunter 类
from kube_hunter.core.types import Hunter
# 从 kube_hunter.core.events.types 模块中导入 Event 和 Service 类
from kube_hunter.core.events.types import Event, Service
# 从 kube_hunter.core.events 模块中导入 handler
from kube_hunter.core.events import handler

# 初始化计数器
counter = 0

# 定义一个继承自 Service 和 Event 类的 OnceOnlyEvent 类
class OnceOnlyEvent(Service, Event):
    def __init__(self):
        # 调用父类的初始化方法，设置服务名称为 "Test Once Service"
        Service.__init__(self, "Test Once Service")

# 定义一个继承自 Service 和 Event 类的 RegularEvent 类
class RegularEvent(Service, Event):
    def __init__(self):
        # 调用父类的初始化方法，设置服务名称为 "Test Service"
        Service.__init__(self, "Test Service")

# 使用 handler.subscribe_once 装饰器订阅 OnceOnlyEvent 事件
@handler.subscribe_once(OnceOnlyEvent)
class OnceHunter(Hunter):
    def __init__(self, event):
        # 使用全局变量 counter 记录事件触发次数
        global counter
        counter += 1

# 使用 handler.subscribe 装饰器订阅 RegularEvent 事件
@handler.subscribe(RegularEvent)
class RegularHunter(Hunter):
    def __init__(self, event):
        # 使用全局变量 counter 记录事件触发次数
        global counter
        counter += 1

# 定义测试函数 test_subscribe_mechanism
def test_subscribe_mechanism():
    # 使用全局变量 counter 记录事件触发次数

    # 首先测试正常的订阅和发布是否正常工作
    handler.publish_event(RegularEvent())
    handler.publish_event(RegularEvent())
    handler.publish_event(RegularEvent())

    # 等待一段时间
    time.sleep(0.02)
    # 断言事件触发次数为 3
    assert counter == 3
    # 重置计数器
    counter = 0

    # 测试 subscribe_once 机制
    handler.publish_event(OnceOnlyEvent())
    handler.publish_event(OnceOnlyEvent())
    handler.publish_event(OnceOnlyEvent())

    # 等待一段时间
    time.sleep(0.02)
    # 断言事件触发次数为 1
    assert counter == 1
```