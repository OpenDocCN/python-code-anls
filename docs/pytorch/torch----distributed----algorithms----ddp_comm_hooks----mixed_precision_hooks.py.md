# `.\pytorch\torch\distributed\algorithms\ddp_comm_hooks\mixed_precision_hooks.py`

```py
from dataclasses import dataclass
from typing import Any, no_type_check

import torch
import torch.distributed as dist
from torch.autograd import Variable
from torch.distributed.utils import _free_storage


@dataclass
class _AllreduceUpcastHookState:
    """
    State to manage DDP mixed precision in backward / gradient communication.

    This contains a weakref to the DDP module for access to reducer and process
    group, and a stream to run parameter and gradient upcasts.
    """

    ddp_weakref: Any  # 弱引用指向 DDP 模块，用于获取 reducer 和 process group
    upcast_stream: torch.cuda.Stream  # 用于运行参数和梯度升级的 CUDA 流
    wait_for_stream_enqueued: bool = False  # 是否等待流排队


@no_type_check
def _reducer_allreduce_and_upcast_hook(
    hook_state: _AllreduceUpcastHookState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    Perform allreduce in precision ``reduce_dtype``, upcast to prepare for optimizer.

    Performs allreduce in the reduced precision given by DDP's mixed precision
    reduce_dtype, and upcasts parameters and gradients to fp32 in preparation
    to run the optimizer.
    """
    ddp_weakref = hook_state.ddp_weakref
    reducer, process_group = ddp_weakref().reducer, ddp_weakref().process_group
    gradient_is_bucket_view = ddp_weakref().gradient_as_bucket_view
    
    # Cast bucket if different than param_dtype.
    if (
        ddp_weakref().mixed_precision.param_dtype
        != ddp_weakref().mixed_precision.reduce_dtype
    ):
        # 将 bucket 张量转换为 reduce_dtype
        bucket.set_buffer(
            bucket.buffer().to(ddp_weakref().mixed_precision.reduce_dtype)
        )
    
    fut = reducer._run_allreduce_hook(bucket)
    ret_fut = torch.futures.Future()
    stream = hook_state.upcast_stream
    
    with torch.cuda.stream(stream):
        fut.wait()
        bucket.buffer().div_(process_group.size())
        ret_fut.set_result(bucket.buffer())

        # Upcast parameters and gradients so optimizer step can run in fp32.
        params, grads = bucket.parameters(), bucket.gradients()
        for p, g in zip(params, grads):
            p.data = p._fp_param
            # 释放 mixed precision 参数的存储空间，因为它将在下一次前向传播中重新分配。
            _free_storage(p._mp_param)
            p.grad.data = p.grad.to(p.data.dtype)

    # 在反向传播结束时排队一个回调来等待此流
    def wait_for_stream_cb():
        # 等待当前 CUDA 流操作完成
        torch.cuda.current_stream().wait_stream(stream)
        
        # 移除后向传播钩子，因为它们会在下一轮重新安装，类似于FSDP。
        # 不需要梯度的参数仍然需要被转换，因为它们可能参与计算。
        # 然而，由于它们没有梯度钩子安装，所以上述钩子不会重新转换它们，因此在这里重新转换它们。
        for n, p in ddp_weakref().module.named_parameters():
            if hasattr(p, "_ddp_mp_hook_state"):
                p._ddp_mp_hook_state[1].remove()
                delattr(p, "_ddp_mp_hook_state")
            if not p.requires_grad and not hasattr(p, "_ddp_ignored"):
                p.data = p._fp_param

        # 重置为下一次反向传播准备
        hook_state.wait_for_stream_enqueued = False

    if not hook_state.wait_for_stream_enqueued:
        # 将等待流回调函数添加到执行引擎队列中
        Variable._execution_engine.queue_callback(wait_for_stream_cb)
        # 标记回调已经入队
        hook_state.wait_for_stream_enqueued = True

    # 返回异步执行的 Future 对象
    return ret_fut
```