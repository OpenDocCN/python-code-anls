# `ZeroNet\src\Test\TestCached.py`

```py
# 导入时间模块
import time
# 从util模块中导入Cached类
from util import Cached

# 定义CachedObject类
class CachedObject:
    # 初始化方法
    def __init__(self):
        # 初始化计数器
        self.num_called_add = 0
        self.num_called_multiply = 0
        self.num_called_none = 0

    # 使用Cached装饰器，设置超时时间为1秒，实现加法计算
    @Cached(timeout=1)
    def calcAdd(self, a, b):
        # 每次调用增加计数器
        self.num_called_add += 1
        return a + b

    # 使用Cached装饰器，设置超时时间为1秒，实现乘法计算
    @Cached(timeout=1)
    def calcMultiply(self, a, b):
        # 每次调用增加计数器
        self.num_called_multiply += 1
        return a * b

    # 使用Cached装饰器，设置超时时间为1秒，返回None
    @Cached(timeout=1)
    def none(self):
        # 每次调用增加计数器
        self.num_called_none += 1
        return None

# 定义TestCached类
class TestCached:
    # 测试返回None值的情况
    def testNoneValue(self):
        # 创建CachedObject对象
        cached_object = CachedObject()
        # 断言调用none方法返回None
        assert cached_object.none() is None
        # 断言再次调用none方法返回None
        assert cached_object.none() is None
        # 断言none方法被调用了1次
        assert cached_object.num_called_none == 1
        # 等待2秒
        time.sleep(2)
        # 断言再次调用none方法返回None
        assert cached_object.none() is None
        # 断言none方法被调用了2次
        assert cached_object.num_called_none == 2

    # 测试调用计算方法的情况
    def testCall(self):
        # 创建CachedObject对象
        cached_object = CachedObject()
        # 断言调用calcAdd方法返回3
        assert cached_object.calcAdd(1, 2) == 3
        # 断言再次调用calcAdd方法返回3
        assert cached_object.calcAdd(1, 2) == 3
        # 断言调用calcMultiply方法返回2
        assert cached_object.calcMultiply(1, 2) == 2
        # 断言再次调用calcMultiply方法返回2
        assert cached_object.calcMultiply(1, 2) == 2
        # 断言calcAdd方法被调用了1次
        assert cached_object.num_called_add == 1
        # 断言calcMultiply方法被调用了1次

        assert cached_object.num_called_multiply == 1

        # 断言调用calcAdd方法返回5
        assert cached_object.calcAdd(2, 3) == 5
        # 断言再次调用calcAdd方法返回5
        assert cached_object.calcAdd(2, 3) == 5
        # 断言calcAdd方法被调用了2次
        assert cached_object.num_called_add == 2

        # 断言再次调用calcAdd方法返回3
        assert cached_object.calcAdd(1, 2) == 3
        # 断言再次调用calcMultiply方法返回6
        assert cached_object.calcMultiply(2, 3) == 6
        # 断言calcAdd方法被调用了2次
        assert cached_object.num_called_add == 2
        # 断言calcMultiply方法被调用了2次

        # 等待2秒
        time.sleep(2)
        # 断言再次调用calcAdd方法返回3
        assert cached_object.calcAdd(1, 2) == 3
        # 断言calcAdd方法被调用了3次
        assert cached_object.num_called_add == 3
```