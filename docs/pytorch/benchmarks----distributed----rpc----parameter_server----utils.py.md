# `.\pytorch\benchmarks\distributed\rpc\parameter_server\utils.py`

```
import torch  # 导入PyTorch库

RPC_SPARSE = "rpc_sparse"  # 定义字符串常量RPC_SPARSE，表示稀疏张量类型
RPC_DENSE = "rpc_dense"    # 定义字符串常量RPC_DENSE，表示密集张量类型


def sparse_tensor_to_rpc_format(sparse_tensor):
    r"""
    A helper function creates a list containing the indices, values, and size
    of a coalesced sparse tensor.
    Args:
        sparse_tensor (torch.Tensor): sparse_coo_tensor represented as a list
    """
    sparse_tensor = sparse_tensor.coalesce()  # 将稀疏张量稠密化
    return [sparse_tensor.indices(), sparse_tensor.values(), sparse_tensor.size()]  # 返回包含稀疏张量的索引、值和尺寸的列表


def sparse_rpc_format_to_tensor(sparse_rpc_format):
    r"""
    A helper function creates a sparse_coo_tensor from indices, values, and size.
    Args:
        sparse_rpc_format (list): sparse_coo_tensor represented as a list
    """
    return torch.sparse_coo_tensor(  # 根据给定的索引、值和尺寸创建稀疏张量
        sparse_rpc_format[0], sparse_rpc_format[1], sparse_rpc_format[2]
    ).coalesce()  # 将创建的稀疏张量稠密化


def process_bucket_with_remote_server(state, bucket):
    r"""
    Processes a gradient bucket passed by a DDP communication hook
    during .backward(). The method supports processing sparse and dense
    tensors. It records RPC future completion time metric for the trainer.
    Args:
        state (object): maintains state during the training process
        bucket (GradBucket): gradient bucket
    """
    cref = state.cref  # 获取状态对象中的 cref 属性
    tensor = bucket.buffer()  # 获取梯度桶的缓冲数据
    if not cref.use_cuda_rpc:
        tensor = tensor.cpu()  # 如果不使用 CUDA RPC，则将张量移动到 CPU 上
    sparse = tensor.is_sparse  # 检查张量是否为稀疏张量
    if sparse:
        tensor = sparse_tensor_to_rpc_format(tensor)  # 如果是稀疏张量，则转换为RPC格式
    b_index = bucket.get_index()  # 获取梯度桶的索引
    server_args = [cref.server_rref, state.batch_number, b_index, tensor]  # 构建传递给远程服务器的参数列表
    key = state.get_key(b_index)  # 获取与梯度桶索引相关联的键
    cref.record_start("hook_future_metric", key, RPC_SPARSE if sparse else RPC_DENSE)  # 记录RPC未来完成时间度量
    fut = cref.server_rref.rpc_async().average_gradient(*server_args)  # 异步调用远程服务器方法

    def callback(fut):
        cref.record_end("hook_future_metric", key)  # 记录RPC完成时间度量
        tensor = fut.wait()  # 等待远程调用的结果
        if type(tensor) is list:
            tensor = sparse_rpc_format_to_tensor(tensor)  # 如果结果是列表，则转换为稀疏张量
        tensor = tensor.cuda(cref.rank)  # 将结果移动到指定的CUDA设备
        return [tensor]  # 返回处理后的张量列表

    return fut.then(callback)  # 返回异步调用的Future对象并添加回调函数
```