# `ZeroNet\src\util\ThreadPool.py`

```py
# 导入所需的模块
import threading
import time
import queue

import gevent
import gevent.monkey
import gevent.threadpool
import gevent._threading

# 创建线程池类
class ThreadPool:
    # 初始化方法，设置最大线程数和线程池名称
    def __init__(self, max_size, name=None):
        self.setMaxSize(max_size)
        if name:
            self.name = name
        else:
            self.name = "ThreadPool#%s" % id(self)

    # 设置最大线程数
    def setMaxSize(self, max_size):
        self.max_size = max_size
        if max_size > 0:
            self.pool = gevent.threadpool.ThreadPool(max_size)
        else:
            self.pool = None

    # 包装函数，用于在主线程以外的线程中调用函数
    def wrap(self, func):
        if self.pool is None:
            return func

        def wrapper(*args, **kwargs):
            if not isMainThread():  # 如果不在主线程中，则直接调用函数
                return func(*args, **kwargs)
            res = self.apply(func, args, kwargs)
            return res

        return wrapper

    # 在线程池中执行函数
    def spawn(self, *args, **kwargs):
        if not isMainThread() and not self.pool._semaphore.ready():
            # 避免在其他线程中生成信号量错误，并且线程池已满时出现错误
            return main_loop.call(self.spawn, *args, **kwargs)
        res = self.pool.spawn(*args, **kwargs)
        return res

    # 应用函数到线程池中
    def apply(self, func, args=(), kwargs={}):
        t = self.spawn(func, *args, **kwargs)
        if self.pool._apply_immediately():
            return main_loop.call(t.get)
        else:
            return t.get()

    # 关闭线程池
    def kill(self):
        if self.pool is not None and self.pool.size > 0 and main_loop:
            main_loop.call(lambda: gevent.spawn(self.pool.kill).join(timeout=1))

        del self.pool
        self.pool = None

    # 进入上下文管理器
    def __enter__(self):
        return self

    # 退出上下文管理器
    def __exit__(self, *args):
        self.kill()

# 创建全局线程池和主线程 ID
lock_pool = gevent.threadpool.ThreadPool(50)
main_thread_id = threading.current_thread().ident

# 判断当前线程是否为主线程
def isMainThread():
    return threading.current_thread().ident == main_thread_id

# 创建锁类
class Lock:
    # 初始化方法，设置锁、锁状态、释放方法和时间锁
    def __init__(self):
        # 创建一个线程锁对象
        self.lock = gevent._threading.Lock()
        # 获取锁的状态
        self.locked = self.lock.locked
        # 释放锁的方法
        self.release = self.lock.release
        # 时间锁初始化为0
        self.time_lock = 0
    
    # 获取锁的方法，记录获取锁的时间，并根据条件选择在新线程中获取锁或者当前线程中获取锁
    def acquire(self, *args, **kwargs):
        # 记录获取锁的时间
        self.time_lock = time.time()
        # 如果锁已经被获取并且当前是主线程
        if self.locked() and isMainThread():
            # 在新线程中获取锁，避免阻塞gevent循环
            return lock_pool.apply(self.lock.acquire, args, kwargs)
        else:
            # 在当前线程中获取锁
            return self.lock.acquire(*args, **kwargs)
    
    # 析构方法，在对象被销毁时释放锁
    def __del__(self):
        # 当锁被获取时释放锁
        while self.locked():
            self.release()
# 定义一个事件类
class Event:
    # 初始化方法
    def __init__(self):
        # 创建一个锁对象
        self.get_lock = Lock()
        # 初始化结果为None
        self.res = None
        # 尝试获取锁，非阻塞
        self.get_lock.acquire(False)
        # 事件是否完成的标志
        self.done = False

    # 设置事件结果的方法
    def set(self, res):
        # 如果事件已经完成，则抛出异常
        if self.done:
            raise Exception("Event already has value")
        # 设置结果
        self.res = res
        # 释放锁
        self.get_lock.release()
        # 设置完成标志
        self.done = True

    # 获取事件结果的方法
    def get(self):
        # 如果事件未完成，则阻塞等待
        if not self.done:
            self.get_lock.acquire(True)
        # 如果锁被获取，则释放锁
        if self.get_lock.locked():
            self.get_lock.release()
        # 返回结果
        back = self.res
        return back

    # 析构方法
    def __del__(self):
        # 清空结果
        self.res = None
        # 循环直到锁被释放
        while self.get_lock.locked():
            self.get_lock.release()


# 执行来自其他线程的主循环中的函数调用
class MainLoopCaller():
    # 初始化方法
    def __init__(self):
        # 创建一个队列
        self.queue_call = queue.Queue()
        # 创建一个线程池
        self.pool = gevent.threadpool.ThreadPool(1)
        # 直接执行的数量
        self.num_direct = 0
        # 运行标志
        self.running = True

    # 调用函数的方法
    def caller(self, func, args, kwargs, event_done):
        try:
            # 执行函数
            res = func(*args, **kwargs)
            # 设置事件完成标志和结果
            event_done.set((True, res))
        except Exception as err:
            # 设置事件完成标志和错误信息
            event_done.set((False, err))

    # 启动方法
    def start(self):
        # 创建一个协程并运行
        gevent.spawn(self.run)
        # 等待一段时间
        time.sleep(0.001)

    # 运行方法
    def run(self):
        # 循环直到停止标志为False
        while self.running:
            # 如果队列为空，则在新线程中获取队列，避免gevent阻塞
            if self.queue_call.qsize() == 0:
                func, args, kwargs, event_done = self.pool.apply(self.queue_call.get)
            else:
                func, args, kwargs, event_done = self.queue_call.get()
            # 创建一个协程执行调用函数
            gevent.spawn(self.caller, func, args, kwargs, event_done)
            # 删除变量引用
            del func, args, kwargs, event_done
        # 设置运行标志为False
        self.running = False
    # 定义一个方法，用于在当前线程中调用指定的函数，并传入参数
    def call(self, func, *args, **kwargs):
        # 如果当前线程的标识符与主线程的标识符相同，则直接调用函数并返回结果
        if threading.current_thread().ident == main_thread_id:
            return func(*args, **kwargs)
        # 如果当前线程不是主线程，则创建一个事件对象
        else:
            event_done = Event()
            # 将函数、参数、关键字参数和事件对象放入调用队列中
            self.queue_call.put((func, args, kwargs, event_done))
            # 获取事件对象的成功标志和结果
            success, res = event_done.get()
            # 删除事件对象
            del event_done
            # 标记调用队列中的任务已完成
            self.queue_call.task_done()
            # 如果成功，则返回结果
            if success:
                return res
            # 如果失败，则抛出异常
            else:
                raise res
# 定义一个函数用于修复线程中的内存泄漏问题，通过在线程中使用真实的睡眠功能
def patchSleep():  # Fix memory leak by using real sleep in threads
    # 获取原始的睡眠函数
    real_sleep = gevent.monkey.get_original("time", "sleep")

    # 定义一个修补后的睡眠函数
    def patched_sleep(seconds):
        # 如果是主线程，则使用协程的睡眠功能
        if isMainThread():
            gevent.sleep(seconds)
        # 否则使用真实的睡眠功能
        else:
            real_sleep(seconds)
    # 重写time模块中的sleep函数
    time.sleep = patched_sleep

# 创建一个主循环调用器对象
main_loop = MainLoopCaller()
# 启动主循环
main_loop.start()
# 修复线程中的内存泄漏问题
patchSleep()
```