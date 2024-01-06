# `kubehunter\tests\core\test_subscribe.py`

```
# 导入时间模块
import time

# 导入自定义模块
from kube_hunter.core.types import Hunter
from kube_hunter.core.events.types import Event, Service
from kube_hunter.core.events import handler

# 初始化计数器
counter = 0

# 定义一个继承自Service和Event的OnceOnlyEvent类
class OnceOnlyEvent(Service, Event):
    def __init__(self):
        # 调用父类的初始化方法，设置服务名称为"Test Once Service"
        Service.__init__(self, "Test Once Service")

# 定义一个继承自Service和Event的RegularEvent类
class RegularEvent(Service, Event):
    def __init__(self):
        # 调用父类的初始化方法，设置服务名称为"Test Service"
        Service.__init__(self, "Test Service")

# 订阅OnceOnlyEvent事件，只订阅一次
@handler.subscribe_once(OnceOnlyEvent)
# 定义一个名为OnceHunter的类，继承自Hunter类
class OnceHunter(Hunter):
    # 初始化方法，接受一个事件参数
    def __init__(self, event):
        # 声明全局变量counter
        global counter
        # counter加1
        counter += 1

# 使用handler.subscribe装饰器订阅RegularEvent事件
@handler.subscribe(RegularEvent)
# 定义一个名为RegularHunter的类，继承自Hunter类
class RegularHunter(Hunter):
    # 初始化方法，接受一个事件参数
    def __init__(self, event):
        # 声明全局变量counter
        global counter
        # counter加1
        counter += 1

# 定义一个名为test_subscribe_mechanism的函数
def test_subscribe_mechanism():
    # 声明全局变量counter

    # 测试正常订阅和发布事件是否正常工作
    handler.publish_event(RegularEvent())
    handler.publish_event(RegularEvent())
    handler.publish_event(RegularEvent())
# 暂停程序执行 0.02 秒
time.sleep(0.02)
# 断言计数器的值为 3
assert counter == 3
# 将计数器重置为 0
counter = 0

# 测试一次订阅机制
# 发布一次性事件
handler.publish_event(OnceOnlyEvent())
handler.publish_event(OnceOnlyEvent())
handler.publish_event(OnceOnlyEvent())

# 暂停程序执行 0.02 秒
time.sleep(0.02)
# 断言计数器的值应该为 1
# 表明一次性事件应该只被触发一次
assert counter == 1
```