# `ZeroNet\src\Test\TestNoparallel.py`

```
# 导入时间模块
import time
# 导入 gevent 模块
import gevent
# 导入 pytest 模块
import pytest
# 导入自定义的 util 模块
import util
# 从 util 模块中导入 ThreadPool 类
from util import ThreadPool

# 使用 pytest 的 fixture 装饰器，参数化测试用例
@pytest.fixture(params=['gevent.spawn', 'thread_pool.spawn'])
def queue_spawn(request):
    # 创建一个线程池对象，最大线程数为 10
    thread_pool = ThreadPool.ThreadPool(10)
    # 根据参数值选择返回 gevent.spawn 或 thread_pool.spawn 函数
    if request.param == "gevent.spawn":
        return gevent.spawn
    else:
        return thread_pool.spawn

# 定义一个示例类
class ExampleClass(object):
    def __init__(self):
        self.counted = 0

    # 使用 util 模块中的 Noparallel 装饰器修饰 countBlocking 方法
    @util.Noparallel()
    def countBlocking(self, num=5):
        # 循环计数，每次循环暂停 0.1 秒
        for i in range(1, num + 1):
            time.sleep(0.1)
            self.counted += 1
        return "counted:%s" % i

    # 使用 util 模块中的 Noparallel 装饰器修饰 countQueue 方法
    @util.Noparallel(queue=True, ignore_class=True)
    def countQueue(self, num=5):
        # 循环计数，每次循环暂停 0.1 秒
        for i in range(1, num + 1):
            time.sleep(0.1)
            self.counted += 1
        return "counted:%s" % i

    # 使用 util 模块中的 Noparallel 装饰器修饰 countNoblocking 方法
    @util.Noparallel(blocking=False)
    def countNoblocking(self, num=5):
        # 循环计数，每次循环暂停 0.01 秒
        for i in range(1, num + 1):
            time.sleep(0.01)
            self.counted += 1
        return "counted:%s" % i

# 定义一个测试类
class TestNoparallel:
    # 定义测试方法 testBlocking，使用参数化的 queue_spawn 作为参数
    def testBlocking(self, queue_spawn):
        # 创建 ExampleClass 的两个实例对象
        obj1 = ExampleClass()
        obj2 = ExampleClass()

        # 不允许再次调用，直到其运行并等待其运行
        threads = [
            queue_spawn(obj1.countBlocking),  # 调用 obj1.countBlocking 方法
            queue_spawn(obj1.countBlocking),  # 调用 obj1.countBlocking 方法
            queue_spawn(obj1.countBlocking),  # 调用 obj1.countBlocking 方法
            queue_spawn(obj2.countBlocking)   # 调用 obj2.countBlocking 方法
        ]
        # 断言 obj2.countBlocking() 的返回值为 "counted:5"
        assert obj2.countBlocking() == "counted:5"  # 调用被忽略，因为 obj2.countBlocking 已经在计数，但会阻塞直到计数完成
        # 等待所有线程执行完成
        gevent.joinall(threads)
        # 断言所有线程的返回值为 ["counted:5", "counted:5", "counted:5", "counted:5"]
        assert [thread.value for thread in threads] == ["counted:5", "counted:5", "counted:5", "counted:5"]
        obj2.countBlocking()  # 允许再次调用，因为 obj2.countBlocking 已经完成

        # 断言 obj1.counted 的值为 5
        assert obj1.counted == 5
        # 断言 obj2.counted 的值为 10
        assert obj2.counted == 10
    # 测试非阻塞计数的方法
    def testNoblocking(self):
        # 创建 ExampleClass 的实例
        obj1 = ExampleClass()

        # 调用 countNoblocking 方法，返回一个线程对象
        thread1 = obj1.countNoblocking()
        # 调用 countNoblocking 方法，返回另一个线程对象（被忽略）
        thread2 = obj1.countNoblocking()  # Ignored

        # 断言 obj1.counted 的初始值为 0
        assert obj1.counted == 0
        # 等待 0.1 秒
        time.sleep(0.1)
        # 断言 thread1 的值为 "counted:5"
        assert thread1.value == "counted:5"
        # 断言 thread2 的值为 "counted:5"
        assert thread2.value == "counted:5"
        # 断言 obj1.counted 的值为 5
        assert obj1.counted == 5

        # 调用 countNoblocking 方法并等待其完成
        obj1.countNoblocking().join()  # Allow again and wait until finishes
        # 断言 obj1.counted 的值为 10
        assert obj1.counted == 10

    # 测试队列的方法
    def testQueue(self, queue_spawn):
        # 创建 ExampleClass 的实例
        obj1 = ExampleClass()

        # 使用队列方式调用 countQueue 方法，num 参数为 1
        queue_spawn(obj1.countQueue, num=1)
        # 使用队列方式调用 countQueue 方法，num 参数为 1
        queue_spawn(obj1.countQueue, num=1)
        # 使用队列方式调用 countQueue 方法，num 参数为 1
        queue_spawn(obj1.countQueue, num=1)

        # 等待 0.3 秒
        time.sleep(0.3)
        # 断言 obj1.counted 的值为 2（不支持多队列）
        assert obj1.counted == 2  # No multi-queue supported

        # 创建另一个 ExampleClass 的实例
        obj2 = ExampleClass()
        # 使用队列方式调用 countQueue 方法，num 参数为 10
        queue_spawn(obj2.countQueue, num=10)
        # 使用队列方式调用 countQueue 方法，num 参数为 10
        queue_spawn(obj2.countQueue, num=10)

        # 等待 1.5 秒，第一个调用已完成，第二个调用仍在进行中
        time.sleep(1.5)  # Call 1 finished, call 2 still working
        # 断言 obj2.counted 的值在 10 和 20 之间
        assert 10 < obj2.counted < 20

        # 使用队列方式调用 countQueue 方法，num 参数为 10
        queue_spawn(obj2.countQueue, num=10)
        # 等待 2.0 秒
        time.sleep(2.0)
        # 断言 obj2.counted 的值为 30
        assert obj2.counted == 30

    # 测试队列过载的方法
    def testQueueOverload(self):
        # 创建 ExampleClass 的实例
        obj1 = ExampleClass()

        # 创建一个包含 1000 个线程的列表
        threads = []
        for i in range(1000):
            # 使用协程方式调用 countQueue 方法，num 参数为 5
            thread = gevent.spawn(obj1.countQueue, num=5)
            threads.append(thread)

        # 等待所有线程完成
        gevent.joinall(threads)
        # 断言 obj1.counted 的值为 5 * 2（只调用了两次，不支持多队列）
        assert obj1.counted == 5 * 2  # Only called twice (no multi-queue allowed)
    # 测试忽略类方法，创建两个 ExampleClass 的实例
    def testIgnoreClass(self, queue_spawn):
        obj1 = ExampleClass()
        obj2 = ExampleClass()

        # 创建多个线程，分别调用 obj1 和 obj2 的 countQueue 方法
        threads = [
            queue_spawn(obj1.countQueue),
            queue_spawn(obj1.countQueue),
            queue_spawn(obj1.countQueue),
            queue_spawn(obj2.countQueue),
            queue_spawn(obj2.countQueue)
        ]
        # 记录当前时间
        s = time.time()
        # 等待一段时间
        time.sleep(0.001)
        # 等待所有线程执行完毕
        gevent.joinall(threads)

        # 断言 obj1 和 obj2 的计数之和为 10
        assert obj1.counted + obj2.counted == 10

        # 计算时间差
        taken = time.time() - s
        # 断言时间差在 1.0 到 1.2 之间
        assert 1.2 > taken >= 1.0  # 2 * 0.5s count = ~1s

    # 测试异常处理，定义一个自定义异常类 MyException
    def testException(self, queue_spawn):
        class MyException(Exception):
            pass

        # 定义一个函数，抛出 MyException 异常
        @util.Noparallel()
        def raiseException():
            raise MyException("Test error!")

        # 使用 pytest 断言捕获异常，并验证异常信息
        with pytest.raises(MyException) as err:
            raiseException()
        assert str(err.value) == "Test error!"

        # 使用 pytest 断言捕获异常，并验证异常信息
        with pytest.raises(MyException) as err:
            queue_spawn(raiseException).get()
        assert str(err.value) == "Test error!"

    # 测试多线程混合，创建一个 ExampleClass 的实例
    def testMultithreadMix(self, queue_spawn):
        obj1 = ExampleClass()
        # 使用 ThreadPool 创建一个线程池
        with ThreadPool.ThreadPool(10) as thread_pool:
            # 记录当前时间
            s = time.time()
            # 创建多个协程和线程，分别调用 obj1 的 countBlocking 方法
            t1 = queue_spawn(obj1.countBlocking, 5)
            time.sleep(0.01)
            t2 = thread_pool.spawn(obj1.countBlocking, 5)
            time.sleep(0.01)
            t3 = thread_pool.spawn(obj1.countBlocking, 5)
            time.sleep(0.3)
            t4 = gevent.spawn(obj1.countBlocking, 5)
            threads = [t1, t2, t3, t4]
            # 遍历线程列表，等待线程执行完毕，并断言结果
            for thread in threads:
                assert thread.get() == "counted:5"

            # 计算时间差
            time_taken = time.time() - s
            # 断言 obj1 的计数为 5，时间差在 0.5 到 0.7 之间
            assert obj1.counted == 5
            assert 0.5 < time_taken < 0.7
```