# `stable-diffusion-webui\modules\fifo_lock.py`

```
# 导入 threading 和 collections 模块
import threading
import collections

# 定义一个 FIFO 锁类
# 参考链接：https://gist.github.com/vitaliyp/6d54dd76ca2c3cdfc1149d33007dc34a
class FIFOLock(object):
    # 初始化方法
    def __init__(self):
        # 创建一个线程锁对象
        self._lock = threading.Lock()
        # 创建一个内部线程锁对象
        self._inner_lock = threading.Lock()
        # 创建一个双端队列用于存储等待的线程
        self._pending_threads = collections.deque()

    # 获取锁的方法
    def acquire(self, blocking=True):
        # 使用内部线程锁
        with self._inner_lock:
            # 尝试获取锁，如果成功直接返回 True
            lock_acquired = self._lock.acquire(False)
            if lock_acquired:
                return True
            # 如果不是阻塞模式，直接返回 False
            elif not blocking:
                return False

            # 创建一个线程事件对象
            release_event = threading.Event()
            # 将事件对象添加到等待队列中
            self._pending_threads.append(release_event)

        # 等待事件对象被设置
        release_event.wait()
        # 再次尝试获取锁
        return self._lock.acquire()

    # 释放锁的方法
    def release(self):
        # 使用内部线程锁
        with self._inner_lock:
            # 如果有等待的线程，取出一个并设置事件
            if self._pending_threads:
                release_event = self._pending_threads.popleft()
                release_event.set()

            # 释放锁
            self._lock.release()

    # 进入上下文管理器时调用的方法
    __enter__ = acquire

    # 退出上下文管理器时调用的方法
    def __exit__(self, t, v, tb):
        # 释放锁
        self.release()
```