# `.\kubehunter\tests\core\test_subscribe.py`

```

# 导入时间模块
import time

# 导入自定义模块
from kube_hunter.core.types import Hunter
from kube_hunter.core.events.types import Event, Service
from kube_hunter.core.events import handler

# 初始化计数器
counter = 0

# 定义一个只触发一次的事件类
class OnceOnlyEvent(Service, Event):
    def __init__(self):
        Service.__init__(self, "Test Once Service")

# 定义一个常规事件类
class RegularEvent(Service, Event):
    def __init__(self):
        Service.__init__(self, "Test Service")

# 订阅只触发一次的事件类
@handler.subscribe_once(OnceOnlyEvent)
class OnceHunter(Hunter):
    def __init__(self, event):
        global counter
        counter += 1

# 订阅常规事件类
@handler.subscribe(RegularEvent)
class RegularHunter(Hunter):
    def __init__(self, event):
        global counter
        counter += 1

# 测试订阅机制的函数
def test_subscribe_mechanism():
    global counter

    # 测试常规订阅和发布是否正常工作
    handler.publish_event(RegularEvent())
    handler.publish_event(RegularEvent())
    handler.publish_event(RegularEvent())

    # 等待一段时间
    time.sleep(0.02)
    # 断言计数器的值为3
    assert counter == 3
    counter = 0

    # 测试只触发一次订阅机制
    handler.publish_event(OnceOnlyEvent())
    handler.publish_event(OnceOnlyEvent())
    handler.publish_event(OnceOnlyEvent())

    # 等待一段时间
    time.sleep(0.02)
    # 断言计数器的值为1
    assert counter == 1

```