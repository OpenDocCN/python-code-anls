# `stable-diffusion-webui\modules\call_queue.py`

```py
# 从 functools 模块导入 wraps 装饰器
from functools import wraps
# 导入 html 模块
import html
# 导入 time 模块

# 从 modules 模块导入 shared, progress, errors, devices, fifo_lock
from modules import shared, progress, errors, devices, fifo_lock

# 创建一个 FIFO 锁对象
queue_lock = fifo_lock.FIFOLock()

# 定义一个装饰器函数，用于包装被调用的函数
def wrap_queued_call(func):
    # 定义一个内部函数 f，接收任意参数
    def f(*args, **kwargs):
        # 使用队列锁
        with queue_lock:
            # 调用原始函数并返回结果
            res = func(*args, **kwargs)

        return res

    return f

# 定义一个装饰器函数，用于包装 GPU 调用的函数
def wrap_gradio_gpu_call(func, extra_outputs=None):
    # 使用 wraps 装饰器，保留原始函数的元数据
    @wraps(func)
    def f(*args, **kwargs):

        # 如果第一个参数是以 "task(...)" 开头并以 ")" 结尾的字符串，则将其视为作业 ID
        if args and type(args[0]) == str and args[0].startswith("task(") and args[0].endswith(")"):
            id_task = args[0]
            # 将任务 ID 添加到队列中
            progress.add_task_to_queue(id_task)
        else:
            id_task = None

        # 使用队列锁
        with queue_lock:
            # 开始共享状态，传入任务 ID
            shared.state.begin(job=id_task)
            # 开始任务进度追踪
            progress.start_task(id_task)

            try:
                # 调用原始函数并获取结果
                res = func(*args, **kwargs)
                # 记录任务结果
                progress.record_results(id_task, res)
            finally:
                # 完成任务进度追踪
                progress.finish_task(id_task)

            # 结束共享状态
            shared.state.end()

        return res

    # 返回包装后的函数
    return wrap_gradio_call(f, extra_outputs=extra_outputs, add_stats=True)

# 定义一个装饰器函数，用于包装调用的函数
def wrap_gradio_call(func, extra_outputs=None, add_stats=False):
    # 使用 wraps 装饰器，保留原始函数的元数据
    @wraps(func)
    return f
```