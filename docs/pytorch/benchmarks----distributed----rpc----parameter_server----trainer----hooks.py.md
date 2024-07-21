# `.\pytorch\benchmarks\distributed\rpc\parameter_server\trainer\hooks.py`

```
# 导入远程服务器处理桶的工具函数
from utils import process_bucket_with_remote_server

# 导入 PyTorch 相关库
import torch
import torch.distributed as c10d

# 定义一个 DDP 通信钩子，使用进程组的全局归约操作
def allreduce_hook(state, bucket):
    r"""
    使用进程组的全局归约实现的 DDP 通信钩子。
    Args:
        state (object): 维护训练过程中的状态
        bucket (GradBucket): 梯度桶
    """
    # 获取用户定义的回调参考
    cref = state.cref
    # 获取梯度数据张量
    tensor = bucket.buffer()
    # 将张量除以进程组大小，得到需要归约的张量列表
    tensors = [tensor / state.process_group.size()]
    # 获取梯度桶索引对应的键值
    key = state.get_key(bucket.get_index())
    # 如果张量是稀疏的，则进行稀疏张量的压缩操作
    if tensor.is_sparse:
        tensor = tensor.coalesce()
    # 根据张量类型记录起始时间并设置回调函数
    tensor_type = "sparse" if tensor.is_sparse else "dense"
    cref.record_start(
        "hook_future_metric", key, f"{cref.backend}_{tensor_type}_allreduce"
    )
    # 执行全局归约操作并获取其未来
    fut = state.process_group.allreduce(tensors).get_future()

    # 定义归约完成的回调函数
    def callback(fut):
        cref.record_end("hook_future_metric", key)
        return fut.wait()

    # 返回归约的未来对象
    return fut.then(callback)


# 定义一个混合 DDP 通信钩子，对稀疏和密集梯度分别使用不同的进程组
def hybrid_hook(state, bucket):
    r"""
    使用默认的 Gloo 进程组进行稀疏梯度的全局归约，使用非默认的 NCCL 进程组进行密集梯度的全局归约的 DDP 通信钩子。
    Args:
        state (object): 维护训练过程中的状态
        bucket (GradBucket): 梯度桶
    """
    # 获取用户定义的回调参考
    cref = state.cref
    # 获取梯度数据张量
    tensor = bucket.buffer()
    # 获取梯度桶索引对应的键值
    key = state.get_key(bucket.get_index())

    # 如果张量是稀疏的，则使用 Gloo 进程组进行归约
    if tensor.is_sparse:
        cref.record_start("hook_c10d_metric", key, "gloo_sparse_allreduce")
        tensor = tensor.coalesce()
        tensor = tensor / state.process_group.size()
        c10d.all_reduce(tensor, op=c10d.ReduceOp.SUM)
        cref.record_end("hook_c10d_metric", key)
        fut = torch.futures.Future()
        fut.set_result([tensor])
    # 如果张量是密集的，则使用 NCCL 进程组进行归约
    else:
        cref.record_start("hook_future_metric", key, "nccl_dense_allreduce")
        tensors = [bucket.buffer() / state.process_group.size()]
        fut = state.process_group.allreduce(tensors).get_future()

        # 定义归约完成的回调函数
        def callback(fut):
            cref.record_end("hook_future_metric", key)
            return fut.wait()

        fut = fut.then(callback)

    # 返回归约的未来对象
    return fut


# 定义一个 RPC DDP 通信钩子，通过远程服务器方法处理梯度桶
def rpc_hook(state, bucket):
    r"""
    使用 process_bucket_with_remote_server 方法平均稀疏和密集张量的 DDP 通信钩子。
    Args:
        state (object): 维护训练过程中的状态
        bucket (GradBucket): 梯度桶
    """
    return process_bucket_with_remote_server(state, bucket)


# 定义一个稀疏 RPC DDP 通信钩子，对稀疏张量使用当前后端的全局归约实现，对密集张量使用远程服务器处理
def sparse_rpc_hook(state, bucket):
    r"""
    使用当前后端的全局归约实现密集张量，使用服务器处理稀疏张量的 DDP 通信钩子。
    Args:
        state (object): 维护训练过程中的状态
        bucket (GradBucket): 梯度桶
    """
    # 获取梯度数据张量
    tensor = bucket.buffer()
    # 如果张量是稀疏的，则使用远程服务器处理
    if tensor.is_sparse:
        return process_bucket_with_remote_server(state, bucket)
    else:
        # 从状态对象中获取当前计算资源（可能是节点或进程组）
        cref = state.cref
        # 将张量按进程组大小进行归一化处理
        tensor = [tensor / state.process_group.size()]
        # 获取当前桶的索引，用于获取唯一的键
        key = state.get_key(bucket.get_index())
        # 记录开始钩子回调的指标，包括键和后端类型
        cref.record_start("hook_future_metric", key, f"{cref.backend}_dense_allreduce")
        # 执行进程组的全局归约操作，并获取其未来对象
        fut = state.process_group.allreduce(tensor).get_future()

        # 定义异步回调函数
        def callback(fut):
            # 记录结束钩子回调的指标，包括键
            cref.record_end("hook_future_metric", key)
            # 等待并返回异步操作的结果
            return fut.wait()

        # 返回异步操作的未来对象，附带回调函数
        return fut.then(callback)
```