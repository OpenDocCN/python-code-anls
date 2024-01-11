# `ZeroNet\src\util\GreenletManager.py`

```
# 导入 gevent 模块
import gevent
# 从 Debug 模块中导入 Debug 类
from Debug import Debug

# 定义 GreenletManager 类
class GreenletManager:
    # 初始化方法
    def __init__(self):
        # 创建一个空的集合用于存储 greenlet 对象
        self.greenlets = set()

    # 定义 spawnLater 方法
    def spawnLater(self, *args, **kwargs):
        # 创建一个延迟执行的 greenlet 对象
        greenlet = gevent.spawn_later(*args, **kwargs)
        # 绑定一个回调函数，当 greenlet 完成时从集合中移除
        greenlet.link(lambda greenlet: self.greenlets.remove(greenlet))
        # 将 greenlet 对象添加到集合中
        self.greenlets.add(greenlet)
        # 返回创建的 greenlet 对象
        return greenlet

    # 定义 spawn 方法
    def spawn(self, *args, **kwargs):
        # 创建一个 greenlet 对象
        greenlet = gevent.spawn(*args, **kwargs)
        # 绑定一个回调函数，当 greenlet 完成时从集合中移除
        greenlet.link(lambda greenlet: self.greenlets.remove(greenlet))
        # 将 greenlet 对象添加到集合中
        self.greenlets.add(greenlet)
        # 返回创建的 greenlet 对象
        return greenlet

    # 定义 stopGreenlets 方法
    def stopGreenlets(self, reason="Stopping all greenlets"):
        # 获取当前集合中 greenlet 对象的数量
        num = len(self.greenlets)
        # 终止所有 greenlet 对象，并传入终止原因
        gevent.killall(list(self.greenlets), Debug.createNotifyType(reason), block=False)
        # 返回被终止的 greenlet 对象数量
        return num
```