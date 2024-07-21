# `.\pytorch\torch\distributed\_shard\sharded_tensor\_ops\binary_cmp.py`

```py
# mypy: allow-untyped-defs
# 导入PyTorch和分布式相关模块
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as distributed_c10d
from torch.distributed._shard.sharded_tensor import _sharded_op_impl, ShardedTensor


# 定义函数用于比较两个分布式张量是否相等
def _communicate_result(result, pg):
    # 如果结果存在，创建值为1的张量；否则创建值为0的张量
    if result:
        result_tensor = torch.ones(1, device=torch.device(torch.cuda.current_device()))
    else:
        result_tensor = torch.zeros(1, device=torch.device(torch.cuda.current_device()))

    # 在指定的进程组中对所有进程的结果张量进行全局归约操作
    dist.all_reduce(result_tensor, group=pg)

    # 期望的结果是全1张量，其值为进程组中进程数量的乘积
    expected_result = torch.ones(
        1, device=torch.device(torch.cuda.current_device())
    ) * dist.get_world_size(pg)

    # 检查结果张量是否等于期望的全1张量
    return torch.equal(result_tensor, expected_result)


# 定义函数用于比较两个分布式张量的二进制操作
def binary_cmp(cmp_fun, types, args, kwargs=None, process_group=None):
    # 检查参数数量是否为2，否则引发错误
    if len(args) != 2:
        raise ValueError(f"Expected two arguments for torch.{cmp_fun.__name__}")

    result = True
    st1 = args[0]  # 获取第一个分片张量
    st2 = args[1]  # 获取第二个分片张量

    # 检查两个张量是否都为 ShardedTensor 类型
    if not (isinstance(st1, ShardedTensor) and isinstance(st2, ShardedTensor)):
        raise TypeError(
            f"Both arguments to torch.{cmp_fun.__name__} need to be of type ShardedTensor"
        )

    # 验证两个张量是否属于同一个进程组
    if st1._process_group != st2._process_group:
        return False

    # 验证当前进程是否属于指定的进程组，如果不属于，则返回比较结果
    if distributed_c10d._rank_not_in_group(
        st1._process_group
    ) or distributed_c10d._rank_not_in_group(st2._process_group):
        return distributed_c10d._rank_not_in_group(
            st1._process_group
        ) == distributed_c10d._rank_not_in_group(st2._process_group)

    # 验证元数据是否相等
    if st1.metadata() != st2.metadata():
        return _communicate_result(False, st1._process_group)

    # 验证本地分片数量是否相等
    st1_local_shards = st1.local_shards()
    st2_local_shards = st2.local_shards()
    if len(st1_local_shards) != len(st2_local_shards):
        return _communicate_result(False, st1._process_group)

    # 如果kwargs为None，则设为一个空字典
    if kwargs is None:
        kwargs = {}

    # 逐个验证每个本地分片
    for idx in range(len(st1_local_shards)):
        # 检查本地分片的元数据是否相等
        if st1_local_shards[idx].metadata != st2_local_shards[idx].metadata:
            return _communicate_result(False, st1._process_group)
        # 使用给定的比较函数比较本地分片的张量
        if not cmp_fun(
            st1_local_shards[idx].tensor, st2_local_shards[idx].tensor, **kwargs
        ):
            return _communicate_result(False, st1._process_group)

    # 返回最终的比较结果
    return _communicate_result(True, st1._process_group)


# 使用装饰器注册 torch.equal 函数的分布式实现
@_sharded_op_impl(torch.equal)
def equal(types, args, kwargs, process_group):
    return binary_cmp(torch.equal, types, args, kwargs, process_group)


# 使用装饰器注册 torch.allclose 函数的分布式实现
@_sharded_op_impl(torch.allclose)
def allclose(types, args, kwargs, process_group):
    return binary_cmp(torch.allclose, types, args, kwargs, process_group)
```