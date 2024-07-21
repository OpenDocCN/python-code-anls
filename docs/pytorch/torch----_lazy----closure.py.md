# `.\pytorch\torch\_lazy\closure.py`

```
# mypy: allow-untyped-defs
# 引入必要的模块和库
import os  # 导入操作系统相关的功能
import threading  # 导入多线程支持
from queue import Empty as EmptyQueue, Queue  # 导入队列相关功能

from torch._lazy.device_context import get_device_context  # 导入获取设备上下文的函数


class ClosureHandler:
    def __init__(self):
        pass

    def run(self, closure):
        """Run closure function

        Args:
        closure: callable function to run
        """
        closure()  # 执行传入的闭包函数

    def __call__(self, closures):
        for closure in closures:
            self.run(closure)  # 对传入的闭包列表依次执行 run 方法


class AsyncClosureHandler(ClosureHandler):
    """Handler for Asynchronous Step Closures

    Args:
        max_queue_size: The maximum length of the closure queue after which
        the training loop will block until closures are evaluated.
        By default, a reasonable limit of a maximum of 100 on the queue.
        This value can be set using the `XLA_MAX_ASYNC_QUEUE` environment
        variable.
    """

    def __init__(self, max_queue_size=100):
        super().__init__()
        # 初始化异步闭包处理器的属性
        self._closure_queue: Queue = Queue(
            int(os.environ.get("LTC_MAX_ASYNC_QUEUE", max_queue_size))
        )  # 使用环境变量或默认值来设置闭包队列的最大长度
        self._closure_exception: Queue = Queue()  # 异常队列，用于存储处理过程中的异常
        self._closure_lock = threading.Lock()  # 线程锁，用于保护闭包队列的并发访问
        self._closure_event_loop_finished = threading.Event()  # 事件，标志闭包事件循环是否结束
        self._closure_event_loop = None  # 闭包事件循环线程对象，默认为None，表示未启动

    def start_event_loop(self):
        """Start closure event loop if not started"""
        if self._closure_event_loop is None:
            # 定义闭包事件循环的函数
            def event_loop():
                # 循环执行，直到收到结束信号并且闭包队列为空
                while True:
                    try:
                        closure = self._closure_queue.get(block=True, timeout=3)
                        closure()  # 执行闭包函数
                        self._closure_queue.task_done()  # 标记闭包任务完成
                    except EmptyQueue:
                        with self._closure_lock:
                            if self._closure_queue.empty():
                                self._closure_event_loop_finished.set()  # 设置事件循环结束信号
                                return
                    except Exception as e:
                        self._closure_exception.put(e)  # 将捕获的异常放入异常队列
                        return

            self._closure_event_loop = threading.Thread(target=event_loop)  # 创建事件循环线程
            self._closure_event_loop.start()  # 启动事件循环线程

    def run(self, closure):
        with self._closure_lock:
            self._closure_queue.put(closure, block=True)  # 将闭包函数放入队列中
            if (
                self._closure_event_loop is None
                or not self._closure_event_loop.is_alive()
            ):
                try:
                    e = self._closure_exception.get(block=False)
                    raise RuntimeError(
                        "Cannot run asynchronous closure due to previously raised exception"
                    ) from e
                except EmptyQueue:
                    self._closure_event_loop = None
                    self.start_event_loop()  # 如果事件循环未启动或已经结束，重新启动事件循环


def add_step_closure(closure, args=(), run_async=False):
    # To be implemented
    pass  # 占位符，待实现具体功能的闭包函数
    """Adds a closure to the list of the ones to be run at the end of the step.
    Many times during model training there is the need to print/report (print to
    console, post to tensorboard, etc...) information which require the content of
    intermediary tensors to be inspected.
    Inspecting different tensors content in different points of the model code
    requires many executions and typically causes performance issues.
    Adding a step closure will ensure that it will be run after the barrier, when
    all the live tensors will be already materialized to device data.
    Live tensors which will include the ones captured by the closure arguments.
    So using `add_step_closure()` will ensure a single execution will be
    performed, even when multiple closures are queued, requiring multiple tensors
    to be inspected.
    Step closures will be run sequentially in the order they have been queued.
    Note that even though using this API the execution will be optimized, it is
    advised to throttle the printing/reporting events once every N steps.
    Args:
      closure (callable): The function to be called.
      args (tuple): The arguments to be passed to the closure.
      run_async: If True, run the closure asynchronously.
    """
    # 获取当前设备的上下文
    devctx = get_device_context()
    # 根据 run_async 参数确定闭包类型
    closures_type = "async_step_closures" if run_async else "step_closures"
    # 获取当前设备上下文中对应类型的闭包列表，若不存在则创建一个空列表
    step_closures = getattr(devctx, closures_type, None)
    if step_closures is None:
        step_closures = []
        setattr(devctx, closures_type, step_closures)
    # 将一个 lambda 函数添加到闭包列表中，lambda 函数接收 args 作为参数并调用原始闭包函数
    step_closures.append(lambda a=args: closure(*a))
def run_step_closures():
    # 获取设备上下文对象
    devctx = get_device_context()
    # 获取异步步骤闭包列表
    async_step_closures = getattr(devctx, "async_step_closures", None)
    # 如果异步步骤闭包列表不为空
    if async_step_closures is not None:
        # 清空异步步骤闭包列表
        devctx.async_step_closures = []
        # 获取异步闭包处理器对象
        async_closure_handler = getattr(devctx, "async_closure_handler", None)
        # 如果异步闭包处理器对象不存在
        if async_closure_handler is None:
            # 创建一个新的异步闭包处理器对象
            async_closure_handler = AsyncClosureHandler()
            # 将新创建的异步闭包处理器对象赋给设备上下文的属性
            devctx.async_closure_handler = async_closure_handler
        # 调用异步闭包处理器处理异步步骤闭包列表
        async_closure_handler(async_step_closures)

    # 获取步骤闭包列表
    step_closures = getattr(devctx, "step_closures", None)
    # 如果步骤闭包列表不为空
    if step_closures is not None:
        # 清空步骤闭包列表
        devctx.step_closures = []
        # 获取闭包处理器对象
        closure_handler = getattr(devctx, "closure_handler", None)
        # 如果闭包处理器对象不存在
        if closure_handler is None:
            # 创建一个新的闭包处理器对象
            closure_handler = ClosureHandler()
            # 将新创建的闭包处理器对象赋给设备上下文的属性
            devctx.closure_handler = closure_handler
        # 调用闭包处理器处理步骤闭包列表
        closure_handler(step_closures)
    # 返回更新后的设备上下文对象
    return devctx
```