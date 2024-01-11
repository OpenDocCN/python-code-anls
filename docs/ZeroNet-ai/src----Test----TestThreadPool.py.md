# `ZeroNet\src\Test\TestThreadPool.py`

```
# 导入必要的模块
import time
import threading
import gevent
import pytest
from util import ThreadPool

# 定义测试类 TestThreadPool
class TestThreadPool:
    # 测试执行顺序的方法
    def testExecutionOrder(self):
        # 使用线程池，最大线程数为4
        with ThreadPool.ThreadPool(4) as pool:
            events = []

            # 在线程池中包装一个阻塞函数
            @pool.wrap
            def blocker():
                events.append("S")  # 添加事件标记"S"
                out = 0
                for i in range(10000000):
                    if i == 3000000:
                        events.append("M")  # 添加事件标记"M"
                    out += 1
                events.append("D")  # 添加事件标记"D"
                return out

            threads = []
            # 创建3个协程并添加到线程池中
            for i in range(3):
                threads.append(gevent.spawn(blocker))
            gevent.joinall(threads)

            # 断言事件列表的顺序
            assert events == ["S"] * 3 + ["M"] * 3 + ["D"] * 3

            # 调用阻塞函数
            res = blocker()
            # 断言返回值
            assert res == 10000000

    # 测试同一线程中的锁阻塞
    def testLockBlockingSameThread(self):
        lock = ThreadPool.Lock()

        s = time.time()

        # 定义解锁函数
        def unlocker():
            time.sleep(1)
            lock.release()

        # 创建一个协程执行解锁函数
        gevent.spawn(unlocker)
        lock.acquire(True)  # 获取锁
        lock.acquire(True, timeout=2)  # 获取锁，设置超时时间为2秒

        unlock_taken = time.time() - s  # 计算解锁所花费的时间

        # 断言解锁所花费的时间在1.0到1.5秒之间
        assert 1.0 < unlock_taken < 1.5

    # 测试不同线程中的锁阻塞
    def testLockBlockingDifferentThread(self):
        lock = ThreadPool.Lock()

        # 定义加锁函数
        def locker():
            lock.acquire(True)  # 获取锁
            time.sleep(0.5)
            lock.release()

        # 使用线程池，最大线程数为10
        with ThreadPool.ThreadPool(10) as pool:
            threads = [
                pool.spawn(locker),  # 在线程池中创建一个协程执行加锁函数
                pool.spawn(locker),  # 在线程池中创建一个协程执行加锁函数
                gevent.spawn(locker),  # 创建一个独立的协程执行加锁函数
                pool.spawn(locker)  # 在线程池中创建一个协程执行加锁函数
            ]
            time.sleep(0.1)

            s = time.time()

            lock.acquire(True, 5.0)  # 获取锁，设置超时时间为5秒

            unlock_taken = time.time() - s  # 计算解锁所花费的时间

            # 断言解锁所花费的时间在1.8到2.2秒之间
            assert 1.8 < unlock_taken < 2.2

            gevent.joinall(threads)  # 等待所有协程执行完成
    # 测试主循环调用者线程 ID
    def testMainLoopCallerThreadId(self):
        # 获取当前线程的 ID
        main_thread_id = threading.current_thread().ident
        # 使用线程池创建一个包含5个线程的线程池
        with ThreadPool.ThreadPool(5) as pool:
            # 定义一个函数，返回当前线程的 ID
            def getThreadId(*args, **kwargs):
                return threading.current_thread().ident

            # 在线程池中执行 getThreadId 函数，并断言返回的线程 ID 不等于主线程 ID
            t = pool.spawn(getThreadId)
            assert t.get() != main_thread_id

            # 在线程池中执行 lambda 函数，调用 ThreadPool.main_loop.call(getThreadId) 函数，并断言返回的线程 ID 等于主线程 ID
            t = pool.spawn(lambda: ThreadPool.main_loop.call(getThreadId))
            assert t.get() == main_thread_id

    # 测试主循环调用者 Gevent Spawn
    def testMainLoopCallerGeventSpawn(self):
        # 获取当前线程的 ID
        main_thread_id = threading.current_thread().ident
        # 使用线程池创建一个包含5个线程的线程池
        with ThreadPool.ThreadPool(5) as pool:
            # 定义一个等待函数
            def waiter():
                time.sleep(1)
                return threading.current_thread().ident

            # 定义一个 Gevent Spawner 函数
            def geventSpawner():
                # 调用 ThreadPool.main_loop.call(gevent.spawn, waiter) 函数
                event = ThreadPool.main_loop.call(gevent.spawn, waiter)

                # 使用 pytest 断言捕获异常，并判断异常信息是否为 "cannot switch to a different thread"
                with pytest.raises(Exception) as greenlet_err:
                    event.get()
                assert str(greenlet_err.value) == "cannot switch to a different thread"

                # 调用 ThreadPool.main_loop.call(event.get) 函数，返回等待线程的 ID
                waiter_thread_id = ThreadPool.main_loop.call(event.get)
                return waiter_thread_id

            # 记录开始时间
            s = time.time()
            # 在线程池中执行 geventSpawner 函数，并断言主线程 ID 等于等待线程 ID
            waiter_thread_id = pool.apply(geventSpawner)
            assert main_thread_id == waiter_thread_id
            # 计算执行时间
            time_taken = time.time() - s
            # 断言执行时间在0.9到1.2之间
            assert 0.9 < time_taken < 1.2
    # 定义测试事件的方法
    def testEvent(self):
        # 使用线程池并发执行任务
        with ThreadPool.ThreadPool(5) as pool:
            # 创建一个事件对象
            event = ThreadPool.Event()

            # 定义一个设置事件状态的函数
            def setter():
                # 等待1秒
                time.sleep(1)
                # 设置事件状态为"done!"
                event.set("done!")

            # 定义一个获取事件状态的函数
            def getter():
                return event.get()

            # 在线程池中执行设置事件状态的函数
            pool.spawn(setter)
            # 使用gevent创建一个协程执行获取事件状态的函数
            t_gevent = gevent.spawn(getter)
            # 在线程池中创建一个任务执行获取事件状态的函数
            t_pool = pool.spawn(getter)
            # 记录当前时间
            s = time.time()
            # 断言事件状态为"done!"
            assert event.get() == "done!"
            # 计算时间差
            time_taken = time.time() - s
            # 等待所有协程和线程池中的任务执行完成
            gevent.joinall([t_gevent, t_pool])

            # 断言协程中获取的事件状态为"done!"
            assert t_gevent.get() == "done!"
            # 断言线程池中任务获取的事件状态为"done!"
            assert t_pool.get() == "done!"

            # 断言时间差在0.9到1.2之间
            assert 0.9 < time_taken < 1.2

            # 使用pytest断言事件已经有值，不能再设置新的值
            with pytest.raises(Exception) as err:
                event.set("another result")

            # 断言异常信息中包含"Event already has value"
            assert "Event already has value" in str(err.value)

    # 定义测试内存泄漏的方法
    def testMemoryLeak(self):
        # 导入gc模块
        import gc
        # 获取测试方法执行前的线程池对象的id列表
        thread_objs_before = [id(obj) for obj in gc.get_objects() if "threadpool" in str(type(obj))]

        # 定义一个工作函数
        def worker():
            # 等待0.1秒
            time.sleep(0.1)
            return "ok"

        # 定义一个测试线程池的函数
        def poolTest():
            # 使用线程池并发执行worker函数
            with ThreadPool.ThreadPool(5) as pool:
                for i in range(20):
                    pool.spawn(worker)

        # 循环执行5次测试线程池的函数
        for i in range(5):
            poolTest()
            # 获取新创建的线程池对象
            new_thread_objs = [obj for obj in gc.get_objects() if "threadpool" in str(type(obj)) and id(obj) not in thread_objs_before]
            #print("New objs:", new_thread_objs, "run:", num_run)

        # 确保没有线程池对象被遗漏
        assert not new_thread_objs
```