# `.\pytorch\torch\distributed\algorithms\ddp_comm_hooks\optimizer_overlap_hooks.py`

```
# mypy: allow-untyped-defs
# 引入必要的模块和函数，包括数据类定义、部分函数、类型声明和特定的类型检查功能
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, no_type_check

import torch  # 导入 PyTorch 库
import torch.distributed as dist  # 导入分布式训练相关模块
from torch.autograd import Variable  # 导入变量自动求导模块


__all__: List[str] = []  # 初始化空的公开接口列表

_FUNCTIONAL_OPTIM_STEP_METHOD_NAME = "step_param"  # 定义字符串常量，表示优化器对象的优化步骤方法名


class _OptimizerHookState:
    """
    Holds state for running optimizer in-line after DDP communication hook.

    Currently contains only optimizer class which must have a method `step_param`.
    """
    
    __slots__ = ["functional_optimizer", "params_to_optimize"]

    def __init__(self, functional_optim, params=None):
        self.functional_optimizer = functional_optim  # 初始化功能优化器对象
        self._check_valid_functional_optim()  # 检查功能优化器对象是否有效
        self._set_params_to_optimize(params)  # 设置需要优化的参数集合

    def _set_params_to_optimize(self, params):
        if params is not None:
            self.params_to_optimize = set(params)  # 如果参数集合不为空，则设置为集合类型

    def _check_valid_functional_optim(self):
        if not hasattr(self.functional_optimizer, _FUNCTIONAL_OPTIM_STEP_METHOD_NAME):
            # 如果功能优化器对象没有定义指定的优化步骤方法，抛出异常
            raise ValueError(
                f"Class {type(self.functional_optimizer)} must implement method "
                f"{_FUNCTIONAL_OPTIM_STEP_METHOD_NAME}."
            )


@dataclass
class _OptimInBackwardHookState:
    optim_stream: torch.cuda.Stream  # 优化流对象，用于在 GPU 上执行优化操作
    wait_for_optim_stream_enqueued: bool  # 是否等待优化流被入队的标志位


@no_type_check
def _apply_optim_in_backward_hook(
    gradient_is_bucket_view: bool,
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    r"""
    Register hook to apply the optimizer in backward.

    If torch.distributed.optim._apply_optimizer_in_backward is used to overlap
    optimizer with backward pass, DDP will run the below hook to run optimizer
    step for parameters after gradient communication has taken place.
    """
    optim_in_bwd_state = _OptimInBackwardHookState(
        optim_stream=torch.cuda.Stream(),  # 初始化优化流对象
        wait_for_optim_stream_enqueued=False,  # 设置不等待优化流入队
    )

    def apply_optim_in_backward_hook(
        hook_state: Any,
        bucket: dist.GradBucket,  # 梯度桶对象，包含了分布式训练过程中的梯度信息
        optim_stream_state,  # 优化流状态，用于控制优化操作在 GPU 上的执行
    ) -> torch.futures.Future[torch.Tensor]:
        # 定义一个函数签名，接受一个bucket作为参数，并返回一个Future对象，其结果为torch.Tensor类型
        
        # 获取DDP状态的弱引用
        ddp_weakref = hook_state
        # 从弱引用获取DDP实例
        ddp_inst = ddp_weakref()
        # 获取DDP实例中的reducer和process_group
        reducer, process_group = ddp_inst.reducer, ddp_inst.process_group
        # 调用reducer的_run_allreduce_hook方法，并传入bucket，获取一个Future对象
        fut = reducer._run_allreduce_hook(bucket)
        # 获取优化器流的状态
        optimizer_stream = optim_stream_state.optim_stream
        # 在优化器流上设置CUDA流
        with torch.cuda.stream(optimizer_stream):
            # 等待Future对象完成
            fut.wait()
            # 由于C++端只执行allreduce而不执行平均，因此在这里应用梯度划分。
            # TODO: (rohan-varma) 当与join hook一起运行时，除法因子可能不同。
            bucket.buffer().div_(process_group.size())
            # 获取bucket中的模型参数
            model_params = bucket.parameters()
            # 获取bucket中的梯度
            grads = bucket.gradients()
            # 遍历模型参数和梯度，并按需进行类型转换，以支持DDP混合精度
            # TODO (rohan-varma): 一旦支持后向传播 + DDP混合精度优化器，需要进行类型转换。
            for p, g in zip(model_params, grads):
                if hasattr(p, "_in_backward_optimizers"):
                    # 注意: 需要将梯度设置为bucket的梯度，因为运行allreduce会导致bucket的梯度被减少，但grad字段不会。
                    if not gradient_is_bucket_view:
                        p.grad = g
                    for optim in p._in_backward_optimizers:
                        optim.step()

        # 需要返回一个Future[Tensor]以遵守通信钩子API契约。
        ret_fut = torch.futures.Future()
        ret_fut.set_result(bucket.buffer())

        # 在后向传播结束时，排队一个回调函数以等待此优化器流，并将所有DDP管理的梯度设置为None。
        def wait_for_optim_stream_callback():
            torch.cuda.current_stream().wait_stream(optim_stream_state.optim_stream)
            # 将DDP管理的梯度设置为None
            for param in ddp_inst._get_data_parallel_params(ddp_inst.module):
                if hasattr(param, "_in_backward_optimizers"):
                    param.grad = None

            # 为下一个后向传播重置状态
            optim_stream_state.wait_for_optim_stream_enqueued = False

        # 如果还没有排队等待优化器流的回调函数，则将其加入队列
        if not optim_stream_state.wait_for_optim_stream_enqueued:
            Variable._execution_engine.queue_callback(wait_for_optim_stream_callback)
            # 标记回调函数已排队
            optim_stream_state.wait_for_optim_stream_enqueued = True

        # 返回通信钩子函数
        return ret_fut

    # 创建一个partial函数，其中包含apply_optim_in_backward_hook函数和optim_stream_state参数
    comm_hook = partial(
        apply_optim_in_backward_hook, optim_stream_state=optim_in_bwd_state
    )
    # 为了DDP的通信钩子日志记录，需要设置以下属性
    comm_hook.__name__ = apply_optim_in_backward_hook.__name__
    comm_hook.__qualname__ = apply_optim_in_backward_hook.__qualname__

    # 返回通信钩子函数
    return comm_hook
def _hook_then_optimizer(
    hook: Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]],
    optimizer_state: _OptimizerHookState,
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    r"""Run optimizer in a functional fashion after DDP communication hook."""
    # 检查 optimizer_state 是否有 params_to_optimize 属性，并且不为 None
    has_set_params = (
        hasattr(optimizer_state, "params_to_optimize")
        and optimizer_state.params_to_optimize is not None
    )

    def hook_then_optimizer_wrapper(
        hook_state, bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        # 运行原始的 hook 函数
        fut = hook(hook_state, bucket)

        def optimizer_step(fut):
            # 获取梯度张量和模型参数
            gradient_tensors = bucket.gradients()
            model_params = bucket.parameters()
            for grad_tensor, model_param in zip(gradient_tensors, model_params):
                # 如果没有设置 params_to_optimize 或者模型参数在 params_to_optimize 中
                if (
                    not has_set_params
                    or model_param in optimizer_state.params_to_optimize
                ):
                    # 调用 functional_optimizer 的 step_param 方法来更新模型参数
                    optimizer_state.functional_optimizer.step_param(
                        model_param,
                        grad_tensor,
                    )
            # 返回 bucket 的缓冲区
            return bucket.buffer()

        # 使用 fut.then() 来链式调用 optimizer_step 函数
        return fut.then(optimizer_step)

    # 返回包装后的 hook_then_optimizer_wrapper 函数
    return hook_then_optimizer_wrapper
```