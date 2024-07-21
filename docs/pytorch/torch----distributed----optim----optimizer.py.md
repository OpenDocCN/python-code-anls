# `.\pytorch\torch\distributed\optim\optimizer.py`

```py
# mypy: allow-untyped-defs
# 引入日志模块
import logging
# 引入默认字典模块
from collections import defaultdict
# 引入线程锁模块
from threading import Lock
# 引入类型提示相关模块
from typing import List, Optional

# 引入PyTorch相关模块
import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.jit as jit
import torch.nn as nn
from torch import Tensor
from torch.distributed.rpc import RRef

# 引入自定义工具函数
from .utils import functional_optim_map

# 模块的公开接口列表
__all__ = ["DistributedOptimizer"]

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


# XXX: 我们在这里定义一个 _ScriptModuleOptimizer 类，
# 明确将 FunctionalOptimizer 类编译为 TorchScript。
# 这是因为 ScriptClass 实例在 Python 中仍然存在，
# 除非你显式地将其编译为 ScriptModule 的属性或将其传递给 ScriptFunction。
# _ScriptLocalOptimizerInterface 作为 Optimizer ScriptModules 的通用接口类型。
#
# TODO (wanchaol): 一旦我们添加了 TorchScript 类的引用语义，就可以删除这部分内容。
@jit.interface
class _ScriptLocalOptimizerInterface:
    # 定义接口方法，用于在 TorchScript 中执行优化步骤
    def step(self, autograd_ctx_id: int) -> None:
        pass


# _ScriptLocalOptimizer 类继承自 nn.Module 类
class _ScriptLocalOptimizer(nn.Module):
    # TorchScript 不支持多线程并发编译。
    # request_callback 可能会调用并发编译，因此我们使用锁来串行化编译过程。
    compile_lock = Lock()

    # 初始化方法，接受优化器类、本地参数 RRef 的列表以及其他参数和关键字参数
    def __init__(self, optim_cls, local_params_rref, *args, **kwargs):
        super().__init__()
        # 从每个 RRef 中获取其本地数值并存储在 _local_params 列表中
        self._local_params = [rref.local_value() for rref in local_params_rref]
        # 使用给定的参数和关键字参数创建优化器实例
        self.optim = optim_cls(self._local_params, *args, **kwargs)

    # 在 TorchScript 中导出的优化步骤方法
    @jit.export
    def step(self, autograd_ctx_id: int):
        # 获取当前 autograd 上下文 ID 下的所有本地梯度
        all_local_grads = dist_autograd.get_gradients(autograd_ctx_id)
        # 使用梯度列表 grads 来执行函数优化器步骤
        grads: List[Optional[Tensor]] = [
            all_local_grads[p] if p in all_local_grads else None
            for p in self._local_params
        ]
        self.optim.step(grads)


# TODO (wanchaol): 一旦我们在 distributed.optim 中将所有内容转换为函数优化器，就可以删除或合并此部分内容。
class _LocalOptimizer:
    # 理想情况下，我们只需要为处理相同参数的 _LocalOptimizer 实例共享一个锁。
    # 在这里我们做了一个简化假设，即如果每个 worker 中有多个 _LocalOptimizer 实例，
    # 那么它们将优化相同的参数（例如，每个数据并行训练器将创建自己的 _LocalOptimizer 实例，
    # 但它们将在每个 worker 上优化相同的参数）。
    global_lock = Lock()

    # 初始化方法，接受优化器类、本地参数 RRef 的列表以及其他参数和关键字参数
    def __init__(self, optim_cls, local_params_rref, *args, **kwargs):
        # 从每个 RRef 中获取其本地数值并存储在 _local_params 列表中
        self._local_params = [rref.local_value() for rref in local_params_rref]
        # 使用给定的参数和关键字参数创建优化器实例
        self.optim = optim_cls(self._local_params, *args, **kwargs)
    # 定义一个方法 `step`，用于执行优化器的一个步骤
    def step(self, autograd_ctx_id):
        # 获取特定自动求导上下文 `autograd_ctx_id` 中的所有本地梯度
        all_local_grads = dist_autograd.get_gradients(autograd_ctx_id)

        # 使用全局锁确保多线程环境下的安全访问
        with _LocalOptimizer.global_lock:
            # 遍历所有本地梯度，将其分配给对应的参数的梯度属性
            for param, grad in all_local_grads.items():
                param.grad = grad
            # 调用优化器的 `step` 方法，执行参数更新
            self.optim.step()
# 使用给定的优化器类、本地参数的远程引用以及其他参数，创建一个新的本地优化器并返回其远程引用
def _new_local_optimizer(optim_cls, local_params_rref, *args, **kwargs):
    return rpc.RRef(_LocalOptimizer(optim_cls, local_params_rref, *args, **kwargs))


# 从本地优化器的远程引用中获取本地优化器实例，并执行一步优化操作，传入自动求导上下文的ID
def _local_optimizer_step(local_optim_rref, autograd_ctx_id):
    local_optim = local_optim_rref.local_value()
    local_optim.step(autograd_ctx_id)


# 结合 _ScriptLocalOptimizer 提供的新建和步骤函数，以实现在无全局解释器锁（GIL）的情况下执行优化器操作
def _new_script_local_optimizer(optim_cls, local_params_rref, *args, **kwargs):
    # 使用 _ScriptLocalOptimizer 类创建一个优化器实例
    optim = _ScriptLocalOptimizer(optim_cls, local_params_rref, *args, **kwargs)

    # 使用 _ScriptLocalOptimizer 类的编译锁来确保在 TorchScript 下编译优化器
    with _ScriptLocalOptimizer.compile_lock:
        script_optim = jit.script(optim)
        # 返回编译后的优化器的远程引用
        return rpc.RRef(script_optim, _ScriptLocalOptimizerInterface)


# 使用 TorchScript 注解定义的函数，执行脚本化的本地优化器步骤操作
@jit.script
def _script_local_optimizer_step(
    local_optim_rref: RRef[_ScriptLocalOptimizerInterface], autograd_ctx_id: int
) -> None:
    local_optim = local_optim_rref.local_value()
    local_optim.step(autograd_ctx_id)


# 等待一组远程调用的未来对象完成，并处理可能的异常
def _wait_for_all(rpc_futs):
    # TODO: improve error propagation
    exception = None
    results = []
    for fut in rpc_futs:
        try:
            results.append(fut.wait())
        except Exception as e:
            results.append(e)
            exception = e
    if exception is not None:
        raise exception
    return results


class DistributedOptimizer:
    """
    DistributedOptimizer 将分布在多个工作节点上的参数的远程引用，并对每个参数在本地应用给定的优化器。

    该类使用 :meth:`~torch.distributed.autograd.get_gradients` 来检索特定参数的梯度。

    同时调用 :meth:`~torch.distributed.optim.DistributedOptimizer.step`，
    无论来自同一客户端还是不同客户端，都将在每个工作节点上串行化 --
    因为每个工作节点的优化器一次只能处理一组梯度。然而，并不能保证完整的
    前向-反向-优化器序列将一次仅由一个客户端执行。这意味着应用的梯度可能
    不对应于给定工作节点上执行的最新前向传递。此外，跨工作节点也没有保证的顺序。

    `DistributedOptimizer` 默认启用了启用 TorchScript 的本地优化器，
    这样在多线程训练（例如分布式模型并行）中不会被 Python 全局解释器锁（GIL）阻塞。
    大多数优化器当前都支持此功能。您也可以按照 PyTorch 教程中的 `配方`__ 为自定义优化器启用 TorchScript 支持。

    """
    pass  # 类的文档字符串已经提供了相关的详细说明
    Args:
        optimizer_class (optim.Optimizer): 每个 worker 上实例化的优化器类。
        params_rref (list[RRef]): 用于优化的本地或远程参数的 RRef 列表。
        args: 每个 worker 上传递给优化器构造函数的参数。
        kwargs: 每个 worker 上传递给优化器构造函数的关键字参数。
    
    Example::
        >>> # xdoctest: +SKIP("distributed")
        >>> import torch.distributed.autograd as dist_autograd
        >>> import torch.distributed.rpc as rpc
        >>> from torch import optim
        >>> from torch.distributed.optim import DistributedOptimizer
        >>>
        >>> with dist_autograd.context() as context_id:
        >>>   # Forward pass.
        >>>   rref1 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 3))
        >>>   rref2 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 1))
        >>>   loss = rref1.to_here() + rref2.to_here()
        >>>
        >>>   # Backward pass.
        >>>   dist_autograd.backward(context_id, [loss.sum()])
        >>>
        >>>   # Optimizer.
        >>>   dist_optim = DistributedOptimizer(
        >>>      optim.SGD,  # 使用 SGD 优化器
        >>>      [rref1, rref2],  # 优化 rref1 和 rref2 的参数
        >>>      lr=0.05,  # 学习率为 0.05
        >>>   )
        >>>   dist_optim.step(context_id)  # 执行优化步骤
    
    __ https://github.com/pytorch/tutorials/pull/1465
    # 初始化方法，接受优化器类、参数远程引用等参数
    def __init__(self, optimizer_class, params_rref, *args, **kwargs):
        # 记录 API 使用日志
        torch._C._log_api_usage_once("torch.distributed.optim.DistributedOptimizer")
        
        # 将参数按照所属的 worker 分组
        per_worker_params_rref = defaultdict(list)
        for param in params_rref:
            per_worker_params_rref[param.owner()].append(param)

        # 根据条件选择优化器构造函数
        if optimizer_class in functional_optim_map and jit._state._enabled:
            optim_ctor = functional_optim_map.get(optimizer_class)
        else:
            optim_ctor = optimizer_class
        # 判断是否为函数式优化器
        self.is_functional_optim = optim_ctor != optimizer_class

        # 根据优化器类型选择相应的优化器生成函数
        if self.is_functional_optim:
            optimizer_new_func = _new_script_local_optimizer
        else:
            # 如果没有 TorchScript 支持，则发出警告并选择普通的优化器生成函数
            logger.warning(
                "Creating the optimizer %s without TorchScript support, "
                "this might result in slow computation time in multithreading environment"
                "(i.e. Distributed Model Parallel training on CPU) due to the Python's "
                "Global Interpreter Lock (GIL). Please file an issue if you need this "
                "optimizer in TorchScript. ",
                optimizer_class,
            )
            optimizer_new_func = _new_local_optimizer

        # 在远程 worker 上为每组参数创建异步的远程优化器对象
        remote_optim_futs = []
        for worker, param_rrefs in per_worker_params_rref.items():
            remote_optim_rref_fut = rpc.rpc_async(
                worker,
                optimizer_new_func,
                args=(optim_ctor, param_rrefs) + args,
                kwargs=kwargs,
            )
            remote_optim_futs.append(remote_optim_rref_fut)

        # 等待所有远程优化器对象创建完成，并存储在 self.remote_optimizers 中
        self.remote_optimizers = _wait_for_all(remote_optim_futs)

    # 执行单步优化操作的方法
    def step(self, context_id):
        """
        Performs a single optimization step.

        This will call :meth:`torch.optim.Optimizer.step` on each worker
        containing parameters to be optimized, and will block until all workers
        return. The provided ``context_id`` will be used to retrieve the
        corresponding :class:`~torch.distributed.autograd.context` that
        contains the gradients that should be applied to the parameters.

        Args:
            context_id: the autograd context id for which we should run the
                optimizer step.
        """
        # 验证 autograd 上下文的有效性
        dist_autograd._is_valid_context(context_id)

        # 根据是否为函数式优化器选择相应的优化步骤函数
        optimizer_step_func = (
            _script_local_optimizer_step
            if self.is_functional_optim
            else _local_optimizer_step
        )

        # 发起远程调用执行每个远程优化器的优化步骤
        rpc_futs = []
        for optimizer in self.remote_optimizers:
            rpc_futs.append(
                rpc.rpc_async(
                    optimizer.owner(),
                    optimizer_step_func,
                    args=(optimizer, context_id),
                )
            )
        # 等待所有远程优化步骤完成
        _wait_for_all(rpc_futs)
```