# `ZeroNet\src\util\Pooled.py`

```py
# 导入 gevent.pool 模块
import gevent.pool

# 定义 Pooled 类
class Pooled(object):
    # 初始化方法
    def __init__(self, size=100):
        # 创建一个协程池
        self.pool = gevent.pool.Pool(size)
        # 标记协程池是否正在运行
        self.pooler_running = False
        # 创建一个队列
        self.queue = []
        # 函数对象
        self.func = None

    # 等待方法
    def waiter(self, evt, args, kwargs):
        # 调用函数并获取结果
        res = self.func(*args, **kwargs)
        # 如果结果是异步结果对象，则设置事件
        if type(res) == gevent.event.AsyncResult:
            evt.set(res.get())
        else:
            evt.set(res)

    # 协程池方法
    def pooler(self):
        # 循环处理队列中的任务
        while self.queue:
            evt, args, kwargs = self.queue.pop(0)
            self.pool.spawn(self.waiter, evt, args, kwargs)
        self.pooler_running = False

    # 装饰器方法
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            # 创建异步结果对象
            evt = gevent.event.AsyncResult()
            # 将任务添加到队列中
            self.queue.append((evt, args, kwargs))
            # 如果协程池未运行，则启动协程池
            if not self.pooler_running:
                self.pooler_running = True
                gevent.spawn(self.pooler)
            return evt
        wrapper.__name__ = func.__name__
        self.func = func

        return wrapper

# 主程序入口
if __name__ == "__main__":
    # 导入必要的模块
    import gevent
    import gevent.pool
    import gevent.queue
    import gevent.event
    import gevent.monkey
    import time

    # 打补丁
    gevent.monkey.patch_all()

    # 定义 addTask 函数
    def addTask(inner_path):
        # 创建异步结果对象
        evt = gevent.event.AsyncResult()
        # 延迟执行任务
        gevent.spawn_later(1, lambda: evt.set(True))
        return evt

    # 定义 needFile 函数
    def needFile(inner_path):
        return addTask(inner_path)

    # 使用装饰器创建协程池任务
    @Pooled(10)
    def pooledNeedFile(inner_path):
        return needFile(inner_path)

    # 创建任务列表
    threads = []
    for i in range(100):
        threads.append(pooledNeedFile(i))

    # 记录开始时间
    s = time.time()
    # 等待所有任务完成
    gevent.joinall(threads)  # 应该花费 10 秒
    # 打印执行时间
    print(time.time() - s)
```