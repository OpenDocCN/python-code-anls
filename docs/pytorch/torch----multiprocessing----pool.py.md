# `.\pytorch\torch\multiprocessing\pool.py`

```py
import multiprocessing.pool
import multiprocessing.util as util

from .queue import SimpleQueue


def clean_worker(*args, **kwargs):
    import gc
    
    multiprocessing.pool.worker(*args, **kwargs)
    # 普通的 multiprocessing 工作者在结束后不会完全清理资源，
    # 因此我们需要显式触发垃圾回收，以确保调用所有析构函数...
    gc.collect()


class Pool(multiprocessing.pool.Pool):
    """使用我们版本的 SimpleQueue 的 Pool 实现。

    这使我们能够跨进程传递共享内存中的张量，而不是序列化底层数据。
    """

    def _setup_queues(self):
        self._inqueue = SimpleQueue()
        self._outqueue = SimpleQueue()
        self._quick_put = self._inqueue._writer.send
        self._quick_get = self._outqueue._reader.recv

    def _repopulate_pool(self):
        """增加池进程数到指定数量。

        将池进程数增加到指定数量，用于在清理已退出的工作者之后使用。
        """
        for i in range(self._processes - len(self._pool)):
            # 将 worker 改为 clean_worker
            args = (
                self._inqueue,
                self._outqueue,
                self._initializer,
                self._initargs,
                self._maxtasksperchild,
            )
            if hasattr(self, "_wrap_exception"):
                args += (self._wrap_exception,)
            w = self.Process(target=clean_worker, args=args)
            self._pool.append(w)
            w.name = w.name.replace("Process", "PoolWorker")
            w.daemon = True
            w.start()
            util.debug("added worker")
```