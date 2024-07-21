# `.\pytorch\torch\distributed\_shard\sharded_tensor\_ops\_common.py`

```py
# 通过注释允许未声明的函数
# 导入 functools 模块，用于处理装饰器相关的功能
import functools

# 导入必要的函数和类
# 从 torch.distributed._shard.common_op_utils 中导入 _basic_validation 函数
from torch.distributed._shard.common_op_utils import _basic_validation
# 从 torch.distributed._shard.sharded_tensor 中导入 _sharded_op_impl, Shard, ShardedTensor 类
from torch.distributed._shard.sharded_tensor import (
    _sharded_op_impl,
    Shard,
    ShardedTensor,
)


def _sharded_op_common(op, early_stop_func, extra_check):
    """
    注册具有常见逻辑的分片张量操作，这些逻辑在执行各种操作之前执行，
    无论是在本地分片还是本地张量上执行。

    示例::
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> op = torch.transpose
        >>> @_sharded_op_impl(op)
        >>> @_sharded_op_common(op, early_stop_func, extra_check)
        >>> def sharded_tensor_op(types, args, kwargs, process_group):
        >>>   ...
        >>>
        >>> st = sharded_tensor.rand(32, 16)
        >>> st.transpose(1, 2)
        >>> # 这将调用 '_sharded_op_common'

    Args:
        op: 要注册并应用于所有分片的操作。
        early_stop_func (Callable, optional): 用于提前停止的函数。
            默认: 如果 ``None``，则不会提前停止。
        extra_check (Callable, optional): 用于额外条件检查的函数。
            默认: 如果 ``None``，则不进行额外检查。

    Return:
        func (Callable): Torch 函数，我们希望提供分片实现的函数（例如：torch.transpose）
    """

    # 装饰器函数，用于包装被装饰的函数
    def decorator_sharded_func(wrapped_func):
        @functools.wraps(wrapped_func)
        def wrapper(types, args=(), kwargs=None, pg=None):
            # 执行基本验证操作
            _basic_validation(op, args, kwargs)

            st = args[0]  # 获取第一个参数作为 ShardedTensor 对象
            if kwargs is None:
                kwargs = {}
            if extra_check:
                extra_check(*args, **kwargs)  # 如果存在额外检查函数，则调用
            if early_stop_func:
                early_stop = early_stop_func(*args, **kwargs)  # 如果存在提前停止函数，则调用
                if early_stop:
                    return st  # 如果提前停止条件满足，则直接返回原始的 ShardedTensor 对象
            return wrapped_func(types, args, kwargs, pg)  # 调用被装饰的函数

        return wrapper

    return decorator_sharded_func


def _register_sharded_op_on_local_shards(
    op, early_stop_func=None, extra_check=None, customized_func=None
):
    """
    处理在分片张量的每个分片上执行的操作的 ``__torch_function__`` 分发，例如像
    ``torch.nn.functional.gelu`` 或 ``torch.nn.functional.relu`` 的逐元素操作。

    对于更复杂的操作，可以使用定制函数生成新的分片和分片张量大小。

    该函数期望保留 ShardedTensor 的原始 ShardingSpec，无论是否使用定制函数。

    """
    # 定义一个装饰器函数，用于注册和应用于分片张量所有分片的操作
    @_sharded_op_impl(op)
    # 另一个装饰器函数，用于分片张量操作的公共逻辑，包括提前停止函数和额外检查函数
    @_sharded_op_common(op, early_stop_func, extra_check)
    # 定义一个函数，用于在本地分片上执行分片张量操作
    def sharded_tensor_op_on_local_shards(types, args=(), kwargs=None, pg=None):
        # 获取传入参数中的分片张量对象
        st = args[0]
        # 获取分片张量的元数据
        st_metadata = st.metadata()
        # 获取分片张量的本地分片
        local_shards = st.local_shards()
        # 存储新的本地分片
        local_shards_new = []
        # 如果定义了定制函数，则使用定制函数生成新的本地分片和更新的元数据
        if customized_func:
            local_shards_new, st_metadata = customized_func(args, kwargs, pg)
        else:
            # 否则，对每个本地分片执行操作并生成新的本地分片
            for local_shard in local_shards:
                args = (local_shard.tensor, *args[1:])
                local_shards_new.append(
                    Shard(op(*args, **kwargs), local_shard.metadata)
                )
        # 使用新的本地分片和全局元数据初始化新的分片张量对象
        return ShardedTensor._init_from_local_shards_and_global_metadata(
            local_shards_new,
            st_metadata,
            process_group=pg,
            init_rrefs=st._init_rrefs,
            sharding_spec=st.sharding_spec(),
        )
```