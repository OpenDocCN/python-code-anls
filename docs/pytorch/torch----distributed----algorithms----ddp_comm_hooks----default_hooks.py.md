# `.\pytorch\torch\distributed\algorithms\ddp_comm_hooks\default_hooks.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和类型定义
from typing import Any, Callable, cast, Tuple

import torch
import torch.distributed as dist

# 定义公开的函数和方法
__all__ = [
    "allreduce_hook",
    "fp16_compress_hook",
    "bf16_compress_hook",
    "fp16_compress_wrapper",
    "bf16_compress_wrapper",
]

# 定义一个内部函数，用于执行梯度张量的全局平均化并返回一个Future对象
def _allreduce_fut(
    process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future[torch.Tensor]:
    """Average the input gradient tensor by allreduce and returns a future."""
    # 如果提供了process_group，则使用它；否则使用默认的WORLD分组
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    # 先进行除法操作，以避免溢出，尤其是对于FP16类型的张量
    tensor.div_(group_to_use.size())

    # 执行异步的全局归约操作，并返回Future对象
    return (
        dist.all_reduce(tensor, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])  # 返回全局归约后的结果张量
    )

# 定义一个函数，通过allreduce来对GradBucket中的梯度张量进行全局平均化
def allreduce_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    Call ``allreduce`` using ``GradBucket`` tensors.

    Once gradient tensors are aggregated across all workers, its ``then``
    callback takes the mean and returns the result.

    If user registers this DDP communication hook,
    DDP results is expected to be same as the case where no hook was registered.
    Hence, this won't change behavior of DDP and user can use this as a reference
    or modify this hook to log useful information or any other purposes while
    unaffecting DDP behavior.

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, allreduce_hook)
    """
    # 调用内部函数_allreduce_fut来执行全局平均化操作，并返回Future对象
    return _allreduce_fut(process_group, bucket.buffer())

# 定义一个函数，实现梯度压缩，将GradBucket中的梯度张量转换为torch.float16类型，并除以进程组大小
def fp16_compress_hook(
    process_group: dist.ProcessGroup,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """
    Compress by casting ``GradBucket`` to ``torch.float16`` divided by process group size.

    This DDP communication hook implements a simple gradient compression
    approach that casts ``GradBucket`` tensor to half-precision floating-point format (``torch.float16``)
    and then divides it by the process group size.
    It allreduces those ``float16`` gradient tensors. Once compressed gradient
    tensors are allreduced, the chained callback ``decompress`` casts it back to the input data type (such as ``float32``).

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, fp16_compress_hook)
    """
    # 如果提供了process_group，则使用它；否则使用默认的WORLD分组
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    # 获取进程组的大小
    world_size = group_to_use.size()

    # 从GradBucket中提取缓冲区的数据，如果bucket是元组则获取第一个张量，否则获取整个缓冲区
    buffer = (
        cast(Tuple[torch.Tensor, ...], bucket)[0]
        if isinstance(bucket, tuple)
        else bucket.buffer()
    )
    
    # 将缓冲区的数据转换为torch.float16类型，并进行除以进程组大小的操作，实现压缩
    compressed_tensor = buffer.to(torch.float16).div_(world_size)
    # 定义一个函数，用于解压缩压缩后的张量
    def decompress(fut):
        # 将压缩后的数据拷贝到解压缩后的张量中，以减少内存峰值的使用
        # 参考：https://github.com/pytorch/pytorch/issues/45968
        decompressed_tensor = buffer
        # 如果输入的 fut 是 torch.Tensor 类型，则直接使用，否则获取其值的第一个元素
        value = fut if isinstance(fut, torch.Tensor) else fut.value()[0]
        # 将解压缩后的数据拷贝到 decompressed_tensor 中
        decompressed_tensor.copy_(value)
        # 返回解压缩后的张量
        return decompressed_tensor

    # 如果正在编译 torch 库，则执行分布式全局归约操作，并返回解压缩后的结果
    if torch._utils.is_compiling():
        # 使用分布式通信库进行全局归约操作，将压缩后的张量按照 "sum" 的方式进行归约
        grad = dist._functional_collectives.all_reduce(
            compressed_tensor, "sum", group_to_use
        )
        # 对归约后的结果进行解压缩，并返回解压缩后的张量
        return decompress(grad)
    else:
        # 如果没有在编译中，则执行异步的分布式全局归约操作，并获取其未来对象
        fut = dist.all_reduce(
            compressed_tensor, group=group_to_use, async_op=True
        ).get_future()
        # 在未来对象完成后，执行解压缩操作，并返回解压缩后的张量
        return fut.then(decompress)
# TODO: create an internal helper function and extract the duplicate code in FP16_compress and BF16_compress.
def bf16_compress_hook(
    process_group: dist.ProcessGroup,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """
    Warning: This API is experimental, and it requires NCCL version later than 2.9.6.

    This DDP communication hook implements a simple gradient compression
    approach that casts ``GradBucket`` tensor to half-precision
    `Brain floating point format <https://en.wikipedia.org/wiki/Bfloat16_floating-point_format>`_ (``torch.bfloat16``)
    and then divides it by the process group size.
    It allreduces those ``bfloat16`` gradient tensors. Once compressed gradient
    tensors are allreduced, the chained callback ``decompress`` casts it back to the input data type (such as ``float32``).

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, bf16_compress_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    # Determine the size of the process group
    world_size = group_to_use.size()

    # Retrieve the buffer tensor from the GradBucket
    buffer = (
        cast(Tuple[torch.Tensor, ...], bucket)[0]
        if isinstance(bucket, tuple)
        else bucket.buffer()
    )

    # Compress the gradient tensor to bfloat16 and divide by the world size
    compressed_tensor = buffer.to(torch.bfloat16).div_(world_size)

    # Define the decompression function
    def decompress(fut):
        # Ensure the decompressed tensor shares memory with the original buffer
        decompressed_tensor = buffer
        # Decompress the tensor in place to save memory
        # See: https://github.com/pytorch/pytorch/issues/45968
        value = fut if isinstance(fut, torch.Tensor) else fut.value()[0]
        decompressed_tensor.copy_(value)
        return decompressed_tensor

    # Check if Torch is in the compilation phase
    if torch._utils.is_compiling():
        # Perform all-reduce using functional collectives
        grad = dist._functional_collectives.all_reduce(
            compressed_tensor, "sum", group_to_use
        )
        # Return the result of decompression
        return decompress(grad)
    else:
        # Perform asynchronous all-reduce operation
        fut = dist.all_reduce(
            compressed_tensor, group=group_to_use, async_op=True
        ).get_future()
        # Chain the decompression function to the future
        return fut.then(decompress)


def fp16_compress_wrapper(
    hook: Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    """
    Cast input tensor to ``torch.float16``, cast result of hook back to input dtype.

    This wrapper casts the input gradient tensor of a given DDP communication hook to half-precision
    floating point format (``torch.float16``), and casts the resulting tensor of the given hook back to
    the input data type, such as ``float32``.
    Therefore, ``fp16_compress_hook`` is equivalent to ``fp16_compress_wrapper(allreduce_hook)``.

    Example::
        >>> # xdoctest: +SKIP
        >>> state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1, start_powerSGD_iter=10)
        >>> ddp_model.register_comm_hook(state, fp16_compress_wrapper(powerSGD_hook))
    """

    def fp16_compress_wrapper_hook(
        hook_state, bucket: dist.GradBucket
    ):
        # Implement wrapper hook logic here
        pass

    return fp16_compress_wrapper_hook
    ) -> torch.futures.Future[torch.Tensor]:
        # 将桶张量转换为FP16格式。
        bucket.set_buffer(bucket.buffer().to(torch.float16))

        # 调用钩子函数处理桶
        fut = hook(hook_state, bucket)

        # 定义解压函数，处理异步任务结果
        def decompress(fut):
            # 获取解压后的张量数据
            decompressed_tensor = bucket.buffer()
            # 原地解压以减少内存峰值。
            # 参考：https://github.com/pytorch/pytorch/issues/45968
            decompressed_tensor.copy_(fut.value())
            return decompressed_tensor

        # 在钩子函数运行后进行解压处理
        return fut.then(decompress)

    # 返回FP16压缩包装的钩子函数
    return fp16_compress_wrapper_hook
# 定义一个装饰器函数 bf16_compress_wrapper，接受一个类型为 Callable 的参数 hook，返回一个类型也为 Callable 的函数
def bf16_compress_wrapper(
    hook: Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    """
    警告：此 API 是实验性的，需要使用 NCCL 版本大于 2.9.6。

    此装饰器将给定 DDP 通信钩子的输入梯度张量转换为半精度 `Brain floating point format`（`torch.bfloat16`），
    并将给定钩子的结果张量转换回输入数据类型，如 `float32`。

    因此，`bf16_compress_hook` 等同于 `bf16_compress_wrapper(allreduce_hook)`。

    示例::
        >>> # xdoctest: +SKIP
        >>> state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1, start_powerSGD_iter=10)
        >>> ddp_model.register_comm_hook(state, bf16_compress_wrapper(powerSGD_hook))
    """

    # 定义内部函数 bf16_compress_wrapper_hook，接受 hook_state 和 bucket 作为参数，返回一个 Future 对象
    def bf16_compress_wrapper_hook(
        hook_state, bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        # 将 bucket 张量转换为 BF16 格式
        bucket.set_buffer(bucket.buffer().to(torch.bfloat16))

        # 调用给定的 hook 函数处理 bucket，并得到一个 Future 对象
        fut = hook(hook_state, bucket)

        # 定义内部函数 decompress，接受 fut 作为参数，返回解压后的张量
        def decompress(fut):
            # 获取原始的 bucket 缓冲区张量
            decompressed_tensor = bucket.buffer()
            # 原地解压以减少峰值内存使用
            # 参见：https://github.com/pytorch/pytorch/issues/45968
            decompressed_tensor.copy_(fut.value())
            return decompressed_tensor

        # 在 hook 运行后进行解压缩
        return fut.then(decompress)

    # 返回内部定义的 bf16_compress_wrapper_hook 函数
    return bf16_compress_wrapper_hook
```