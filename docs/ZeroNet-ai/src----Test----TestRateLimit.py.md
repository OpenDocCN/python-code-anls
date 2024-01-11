# `ZeroNet\src\Test\TestRateLimit.py`

```
# 导入时间模块
import time
# 导入协程模块
import gevent
# 从util模块中导入RateLimit类
from util import RateLimit

# 定义一个函数，判断给定时间t是否在限制范围内（limit-0.05到limit+0.05之间）
def around(t, limit):
    return t >= limit - 0.05 and t <= limit + 0.05

# 定义一个示例类ExampleClass
class ExampleClass(object):
    # 初始化方法，初始化counted和last_called属性
    def __init__(self):
        self.counted = 0
        self.last_called = None

    # 定义一个计数方法，back参数默认为"counted"
    def count(self, back="counted"):
        # 计数属性counted加1
        self.counted += 1
        # 更新最后调用属性last_called为back
        self.last_called = back
        # 返回back
        return back

# 定义一个测试类TestRateLimit
class TestRateLimit:
    # 定义一个测试方法
    def testCall(self):
        # 创建两个 ExampleClass 的实例对象
        obj1 = ExampleClass()
        obj2 = ExampleClass()

        # 记录当前时间
        s = time.time()
        # 断言 RateLimit.call 方法调用返回 "counted"，并且立即可以再次调用
        assert RateLimit.call("counting", allowed_again=0.1, func=obj1.count) == "counted"
        assert around(time.time() - s, 0.0)  # 第一次调用立即允许
        assert obj1.counted == 1

        # 再次调用
        assert not RateLimit.isAllowed("counting", 0.1)
        assert RateLimit.isAllowed("something else", 0.1)
        assert RateLimit.call("counting", allowed_again=0.1, func=obj1.count) == "counted"
        assert around(time.time() - s, 0.1)  # 在时间间隔内延迟第二次调用
        assert obj1.counted == 2
        time.sleep(0.1)  # 等待冷却时间

        # 异步调用 3 次
        s = time.time()
        assert obj2.counted == 0
        threads = [
            gevent.spawn(lambda: RateLimit.call("counting", allowed_again=0.1, func=obj2.count)),  # 立即
            gevent.spawn(lambda: RateLimit.call("counting", allowed_again=0.1, func=obj2.count)),  # 0.1 秒延迟
            gevent.spawn(lambda: RateLimit.call("counting", allowed_again=0.1, func=obj2.count))   # 0.2 秒延迟
        ]
        gevent.joinall(threads)
        assert [thread.value for thread in threads] == ["counted", "counted", "counted"]
        assert around(time.time() - s, 0.2)

        # 等待 0.1 秒冷却
        assert not RateLimit.isAllowed("counting", 0.1)
        time.sleep(0.11)
        assert RateLimit.isAllowed("counting", 0.1)

        # 没有队列 = 立即再次调用
        s = time.time()
        assert RateLimit.isAllowed("counting", 0.1)
        assert RateLimit.call("counting", allowed_again=0.1, func=obj2.count) == "counted"
        assert around(time.time() - s, 0.0)

        assert obj2.counted == 4
    # 定义一个测试异步调用的方法
    def testCallAsync(self):
        # 创建两个 ExampleClass 的实例对象
        obj1 = ExampleClass()
        obj2 = ExampleClass()

        # 记录当前时间
        s = time.time()
        # 调用 RateLimit.callAsync 方法，设置允许再次调用的时间间隔为0.1秒，调用 obj1 的 count 方法，back 参数为"call #1"，并等待调用结束
        RateLimit.callAsync("counting async", allowed_again=0.1, func=obj1.count, back="call #1").join()
        # 断言 obj1.counted 的值为1，即第一次调用
        assert obj1.counted == 1  # First instant
        # 断言当前时间与记录的时间差在0.1秒以内
        assert around(time.time() - s, 0.0)

        # 之后的调用会有延迟
        # 记录当前时间
        s = time.time()
        # 调用 RateLimit.callAsync 方法，设置允许再次调用的时间间隔为0.1秒，调用 obj1 的 count 方法，back 参数为"call #2"，并将返回的 Future 对象赋值给 t1
        t1 = RateLimit.callAsync("counting async", allowed_again=0.1, func=obj1.count, back="call #2")  # Dumped by the next call
        # 等待0.03秒
        time.sleep(0.03)
        # 调用 RateLimit.callAsync 方法，设置允许再次调用的时间间隔为0.1秒，调用 obj1 的 count 方法，back 参数为"call #3"，并将返回的 Future 对象赋值给 t2
        t2 = RateLimit.callAsync("counting async", allowed_again=0.1, func=obj1.count, back="call #3")  # Dumped by the next call
        # 等待0.03秒
        time.sleep(0.03)
        # 调用 RateLimit.callAsync 方法，设置允许再次调用的时间间隔为0.1秒，调用 obj1 的 count 方法，back 参数为"call #4"，并将返回的 Future 对象赋值给 t3
        t3 = RateLimit.callAsync("counting async", allowed_again=0.1, func=obj1.count, back="call #4")  # Will be called
        # 断言 obj1.counted 的值为1，即延迟仍在进行中：尚未被调用
        assert obj1.counted == 1  # Delay still in progress: Not called yet
        # 等待 t3 调用结束
        t3.join()
        # 断言 t3 的值为"call #4"
        assert t3.value == "call #4"
        # 断言当前时间与记录的时间差在0.1秒以内
        assert around(time.time() - s, 0.1)

        # 只有最后一个被调用
        # 断言 obj1.counted 的值为2
        assert obj1.counted == 2
        # 断言 obj1.last_called 的值为"call #4"
        assert obj1.last_called == "call #4"

        # 刚刚被调用，不允许再次调用
        # 断言 RateLimit.isAllowed 方法返回值为 False
        assert not RateLimit.isAllowed("counting async", 0.1)
        # 记录当前时间
        s = time.time()
        # 调用 RateLimit.callAsync 方法，设置允许再次调用的时间间隔为0.1秒，调用 obj1 的 count 方法，back 参数为"call #5"，并等待调用结束
        t4 = RateLimit.callAsync("counting async", allowed_again=0.1, func=obj1.count, back="call #5").join()
        # 断言 obj1.counted 的值为3
        assert obj1.counted == 3
        # 断言当前时间与记录的时间差在0.1秒以内
        assert around(time.time() - s, 0.1)
        # 断言 RateLimit.isAllowed 方法返回值为 False
        assert not RateLimit.isAllowed("counting async", 0.1)
        # 等待0.11秒
        time.sleep(0.11)
        # 断言 RateLimit.isAllowed 方法返回值为 True
        assert RateLimit.isAllowed("counting async", 0.1)
```