# `.\lucidrains\flash-cosine-sim-attention\flash_cosine_sim_attention\benchmark.py`

```
# 导入 torch 库
import torch
# 从 torch.cuda 模块中导入 synchronize 和 Event 类
from torch.cuda import synchronize, Event
# 从 functools 模块中导入 wraps 和 partial 函数
from functools import wraps, partial

# 创建一个名为 timer 的 partial 函数，用于创建启用计时功能的 Event 对象
timer = partial(Event, enable_timing = True)

# 定义一个装饰器函数 benchmark，用于对指定函数进行性能测试
def benchmark(
    fn,
    *,
    num_times = 10,  # 默认测试次数为 10 次
    warmup_iters = 10,  # 默认预热迭代次数为 10 次
    forwards = True,  # 默认进行前向传播
    backwards = False  # 默认不进行反向传播
):
    assert forwards or backwards

    # 定义内部函数 inner，用于实际执行性能测试
    @wraps(fn)
    def inner(*args, **kwargs):
        # 预热阶段

        for _ in range(warmup_iters):
            # 调用被测试函数，并获取损失值
            loss = fn(*args, **kwargs)

            # 如果需要进行反向传播，则计算梯度
            if backwards:
                loss.sum().backward()

        # 计算多次函数调用的平均时间

        all_measured_times_ms = 0.

        for _ in range(num_times):
            # 创建开始和结束事件对象
            start_event = timer()
            end_event = timer()

            # 如果需要进行前向传播，则记录开始事件
            if forwards:
                start_event.record()

            # 调用被测试函数
            o = fn(*args, **kwargs)

            # 如果不需要进行反向传播，则记录结束事件
            if not backwards:
                end_event.record()

            # 如果不需要进行前向传播，则记录开始事件
            if not forwards:
                start_event.record()

            # 如果需要进行反向传播，则计算损失并反向传播，然后记录结束事件
            if backwards:
                loss = o.sum()
                loss.backward()
                end_event.record()

            # 同步事件
            synchronize()

            # 计算经过的时间
            elapsed_time_ms = start_event.elapsed_time(end_event)
            all_measured_times_ms += elapsed_time_ms

        # 返回平均时间
        return all_measured_times_ms / num_times

    return inner
```