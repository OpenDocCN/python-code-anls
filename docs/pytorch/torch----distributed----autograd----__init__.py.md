# `.\pytorch\torch\distributed\autograd\__init__.py`

```
# mypy: allow-untyped-defs

# 导入 torch 库
import torch

# 检查是否存在 torch._C._dist_autograd_init 属性，判断是否支持分布式自动求导
def is_available():
    return hasattr(torch._C, "_dist_autograd_init")

# 如果支持分布式自动求导但初始化失败，则抛出运行时错误
if is_available() and not torch._C._dist_autograd_init():
    raise RuntimeError("Failed to initialize torch.distributed.autograd")

# 如果支持分布式自动求导，则从 torch._C._distributed_autograd 中导入所需的函数和类
if is_available():
    from torch._C._distributed_autograd import (
        _current_context,
        _get_debug_info,
        _get_max_id,
        _init,
        _is_valid_context,
        _new_context,
        _release_context,
        _retrieve_context,
        backward,
        DistAutogradContext,
        get_gradients,
    )

# 定义 context 类，用于封装使用分布式自动求导进行前向和后向传播的上下文
class context:
    """
    Context object to wrap forward and backward passes when using
    distributed autograd. The ``context_id`` generated in the ``with``
    statement  is required to uniquely identify a distributed backward pass
    on all workers. Each worker stores metadata associated with this
    ``context_id``, which is required to correctly execute a distributed
    autograd pass.

    Example::
        >>> # xdoctest: +SKIP
        >>> import torch.distributed.autograd as dist_autograd
        >>> with dist_autograd.context() as context_id:
        >>>     t1 = torch.rand((3, 3), requires_grad=True)
        >>>     t2 = torch.rand((3, 3), requires_grad=True)
        >>>     loss = rpc.rpc_sync("worker1", torch.add, args=(t1, t2)).sum()
        >>>     dist_autograd.backward(context_id, [loss])
    """

    # 进入上下文时创建新的自动求导上下文，并返回上下文 ID
    def __enter__(self):
        self.autograd_context = _new_context()
        return self.autograd_context._context_id()

    # 退出上下文时释放自动求导上下文资源，根据上下文 ID
    def __exit__(self, type, value, traceback):
        _release_context(self.autograd_context._context_id())
```