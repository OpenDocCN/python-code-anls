# `.\pytorch\torch\distributed\_tensor\_tp_conv.py`

```
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
# 导入必要的类型声明
from typing import cast, Dict, List, Tuple

# 导入 PyTorch 库
import torch
import torch.distributed as dist
import torch.distributed._tensor.api as dtensor

# 使用 torch.ops.aten 别名创建 aten
aten = torch.ops.aten

# 检查是否需要数据交换的函数，基于填充值的第二项判断
def _requires_data_exchange(padding):
    return padding[1] != 0

# 检查张量并行卷积是否受支持的函数
def _is_supported(input_size, kernel_size, stride, padding, dilation):
    # 如果膨胀率的第二项不为1，则抛出异常
    if dilation[1] != 1:
        raise RuntimeError("Dilation must be 1 for tensor parallel convolution.")
    # 如果填充的第二项不为0
    if padding[1] != 0:
        # 并且当填充时，步长不为1时抛出异常
        if stride[1] != 1:
            raise RuntimeError(
                "Stride must be 1 when there is padding for tensor parallel convolution."
            )
        # 并且如果核大小的第四项除以2大于输入大小的第四项，则抛出异常
        if kernel_size[3] // 2 > input_size[3]:
            raise RuntimeError(
                "kernel_size[3] // 2 should be less than or equal to input_size[3] for tensor parallel convolution."
            )
    else:
        # 否则，要求输入大小的第四项必须能被步长的第一项整除，并且步长的第一项等于核大小的第四项
        if not (input_size[3] % stride[1] == 0 and stride[1] == kernel_size[3]):
            raise RuntimeError(
                "It requires that input_size[3] is divisible by stride[1] and stride[1] equals kernel_size[3] "
                "when there is padding for tensor parallel convolution."
            )
    # 如果通过所有检查，则返回True
    return True

# 执行环形发送和接收，重建本地输入张量的函数
def _ring_send_recv_construct(in_tensor, d1, d2, left, right, rank, size):
    # 将张量的右侧边缘数据发送给右侧节点，左侧边缘数据发送给左侧节点
    send_to_right = in_tensor[:, :, :, -d1:].contiguous()
    send_to_left = in_tensor[:, :, :, :d2].contiguous()
    # 创建用于接收数据的张量
    recv_from_right = torch.zeros_like(send_to_left)
    recv_from_left = torch.zeros_like(send_to_right)

    # 定义发送和接收操作
    send_op_right = dist.P2POp(dist.isend, send_to_right, right)
    send_op_left = dist.P2POp(dist.isend, send_to_left, left)
    recv_op_right = dist.P2POp(dist.irecv, recv_from_right, right)
    recv_op_left = dist.P2POp(dist.irecv, recv_from_left, left)

    # 批量执行发送和接收操作，并等待完成
    reqs = dist.batch_isend_irecv(
        [send_op_right, send_op_left, recv_op_left, recv_op_right]
    )
    for req in reqs:
        req.wait()

    # 根据节点的排名，合并接收到的数据到原始张量中
    if rank == 0:
        in_tensor = torch.cat([in_tensor, recv_from_right], dim=-1)
    elif rank == size - 1:
        in_tensor = torch.cat([recv_from_left, in_tensor], dim=-1)
    else:
        in_tensor = torch.cat([recv_from_left, in_tensor, recv_from_right], dim=-1)

    # 返回更新后的张量
    return in_tensor

# 执行环形发送和接收，聚合梯度的边缘像素的函数
def _ring_send_recv_aggregate(grad_in_tensor, d1, d2, left, right, rank, size):
    # 将梯度张量的右侧边缘数据发送给右侧节点，左侧边缘数据发送给左侧节点
    send_to_right = grad_in_tensor[:, :, :, -d2:].contiguous()
    send_to_left = grad_in_tensor[:, :, :, :d1].contiguous()
    # 创建用于接收数据的张量
    recv_from_right = torch.zeros_like(send_to_left)
    recv_from_left = torch.zeros_like(send_to_right)

    # 定义发送和接收操作
    send_op_right = dist.P2POp(dist.isend, send_to_right, right)
    send_op_left = dist.P2POp(dist.isend, send_to_left, left)
    recv_op_right = dist.P2POp(dist.irecv, recv_from_right, right)

    # 返回更新后的张量
    # 创建一个从左侧进程接收数据的 P2P 操作对象
    recv_op_left = dist.P2POp(dist.irecv, recv_from_left, left)
    
    # 批量发送和接收数据的请求列表，包括右侧发送、左侧发送、左侧接收和右侧接收操作
    reqs = dist.batch_isend_irecv(
        [send_op_right, send_op_left, recv_op_left, recv_op_right]
    )
    # 等待所有请求完成
    for req in reqs:
        req.wait()
    
    # 如果当前进程的排名为 0
    if rank == 0:
        # 调整梯度张量的最后一个维度，截去后面的 d2 个元素
        grad_in_tensor = grad_in_tensor[:, :, :, :-d2]
        # 将接收到的右侧数据添加到梯度张量的最后 d1 列
        grad_in_tensor[:, :, :, -d1:] = torch.add(
            grad_in_tensor[:, :, :, -d1:], recv_from_right
        )
    # 如果当前进程的排名为 size - 1（最后一个进程）
    elif rank == size - 1:
        # 调整梯度张量的最后一个维度，截去前面的 d1 个元素
        grad_in_tensor = grad_in_tensor[:, :, :, d1:]
        # 将接收到的左侧数据添加到梯度张量的前 d2 列
        grad_in_tensor[:, :, :, :d2] = torch.add(
            grad_in_tensor[:, :, :, :d2], recv_from_left
        )
    # 对于其他排名的进程
    else:
        # 调整梯度张量的最后一个维度，保留从 d1 到 -d2 的元素范围
        grad_in_tensor = grad_in_tensor[:, :, :, d1:-d2]
        # 将接收到的右侧数据添加到梯度张量的最后 d1 列
        grad_in_tensor[:, :, :, -d1:] = torch.add(
            grad_in_tensor[:, :, :, -d1:], recv_from_right
        )
        # 将接收到的左侧数据添加到梯度张量的前 d2 列
        grad_in_tensor[:, :, :, :d2] = torch.add(
            grad_in_tensor[:, :, :, :d2], recv_from_left
        )
# 执行 TP 卷积操作，根据传入的操作符调用、本地张量参数和关键字参数来执行计算
def tp_convolution(
    op_call: torch._ops.OpOverload,
    local_tensor_args: Tuple[object, ...],
    local_tensor_kwargs: Dict[str, object],
) -> object:
    # 断言操作符为默认的卷积操作
    assert op_call == aten.convolution.default
    # 断言本地张量参数的数量为9个
    assert len(local_tensor_args) == 9

    # 获取当前进程的排名
    rank = dist.get_rank()
    # 获取世界中所有进程的数量
    size = dist.get_world_size()
    # 强制类型转换输入张量和权重张量
    in_tensor = cast(torch.Tensor, local_tensor_args[0])
    weight = cast(torch.Tensor, local_tensor_args[1])
    # 获取步长、填充和扩张参数
    stride, padding, dilation = local_tensor_args[3:6]

    # 断言输入张量的形状、权重张量的形状、步长、填充和扩张参数均为支持的类型
    assert _is_supported(in_tensor.shape, weight.shape, stride, padding, dilation)
    # 断言填充参数为列表类型
    assert isinstance(padding, List)

    # 如果不需要数据交换，则直接调用操作符处理本地张量参数和关键字参数，并返回结果
    if not _requires_data_exchange(padding):
        local_results = op_call(*local_tensor_args, **local_tensor_kwargs)
        return local_results
    else:
        # 步骤 0：计算输入张量的重叠像素
        d = weight.shape[3] - 1
        d1 = d // 2
        d2 = d - d1
        assert d1 + d2 == d
        right = (rank + 1) % size
        left = (rank - 1 + size) % size

        # 步骤 1：重建本地输入张量
        in_tensor = _ring_send_recv_construct(
            in_tensor, d1, d2, left, right, rank, size
        )

        # 步骤 2：将重建后的本地输入张量传递给操作符调用
        local_tensor_args_list = list(local_tensor_args)
        local_tensor_args_list[0] = in_tensor
        local_tensor_args = cast(Tuple[object, ...], local_tensor_args_list)
        local_results = op_call(*local_tensor_args, **local_tensor_kwargs)

        # 步骤 3：从结果中移除额外的输出
        padding_w = padding[1]
        w = local_results.size(3)
        if rank == 0:
            local_results = local_results[:, :, :, : w - padding_w]
        elif rank == size - 1:
            local_results = local_results[:, :, :, padding_w:]
        else:
            local_results = local_results[:, :, :, padding_w : w - padding_w]

        return local_results


# 执行 TP 卷积的反向传播操作，根据传入的操作符调用、本地张量参数和关键字参数来执行计算
def tp_convolution_backward(
    op_call: torch._ops.OpOverload,
    local_tensor_args: Tuple[object, ...],
    local_tensor_kwargs: Dict[str, object],
) -> object:
    # 断言操作符为默认的卷积反向传播操作
    assert op_call == aten.convolution_backward.default
    # 断言本地张量参数的数量为11个
    assert len(local_tensor_args) == 11

    # 获取当前进程的排名
    rank = dist.get_rank()
    # 获取世界中所有进程的数量
    size = dist.get_world_size()
    # 强制类型转换梯度输出张量、输入张量和权重张量
    grad_out_tensor = cast(torch.Tensor, local_tensor_args[0])
    in_tensor = cast(torch.Tensor, local_tensor_args[1])
    weight = cast(torch.Tensor, local_tensor_args[2])
    # 获取步长、填充和扩张参数
    stride, padding, dilation = local_tensor_args[4:7]

    # 断言输入张量的形状、权重张量的形状、步长、填充和扩张参数均为支持的类型
    assert _is_supported(in_tensor.shape, weight.shape, stride, padding, dilation)
    # 断言填充参数为列表类型
    assert isinstance(padding, List)

    # 如果不需要数据交换，则直接调用操作符处理本地张量参数和关键字参数，并返回结果
    if not _requires_data_exchange(padding):
        local_results = op_call(*local_tensor_args, **local_tensor_kwargs)
        return local_results
    else:
        # step 0 compute the overlap pixels of the input tensor
        # 计算输入张量的重叠像素
        d = weight.shape[3] - 1
        d1 = d // 2
        d2 = d - d1
        assert d1 + d2 == d
        right = (rank + 1) % size
        left = (rank - 1 + size) % size

        # step1 reconstruct local input tensor
        # 步骤1 重构本地输入张量
        in_tensor = _ring_send_recv_construct(
            in_tensor, d1, d2, left, right, rank, size
        )

        # step2 reconstruct local gradient output tensor
        # 步骤2 重构本地梯度输出张量
        N, C_out, H_out, _ = grad_out_tensor.shape
        padding_w = padding[1]
        if rank == 0:
            grad_out_tensor = torch.nn.functional.pad(
                grad_out_tensor, (0, padding_w), "constant", 0
            )
        elif rank == size - 1:
            grad_out_tensor = torch.nn.functional.pad(
                grad_out_tensor, (padding_w, 0), "constant", 0
            )
        else:
            grad_out_tensor = torch.nn.functional.pad(
                grad_out_tensor, (padding_w, padding_w), "constant", 0
            )

        # step3 feed local input tensor to op_call
        # 步骤3 将本地输入张量传递给 op_call 函数
        local_tensor_args_list = list(local_tensor_args)
        local_tensor_args_list[0] = grad_out_tensor
        local_tensor_args_list[1] = in_tensor
        local_tensor_args = cast(Tuple[object, ...], local_tensor_args_list)
        local_results = op_call(*local_tensor_args, **local_tensor_kwargs)

        # step4 aggregate gradients for edge pixels
        # 步骤4 聚合边缘像素的梯度
        grad_in_tensor = local_results[0]
        grad_in_tensor = _ring_send_recv_aggregate(
            grad_in_tensor, d1, d2, left, right, rank, size
        )

        local_results = list(local_results)
        local_results[0] = grad_in_tensor
        local_results = cast(Tuple[object, ...], local_results)

        return local_results
# 处理卷积操作的函数，接受一个操作调用、参数元组和关键字参数字典，并返回一个对象
def convolution_handler(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> object:
    # 解包操作调用、参数和关键字参数到 OpInfo 对象
    op_info = dtensor.DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)

    # 分片传播
    dtensor.DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"

    # 局部传播
    local_results = tp_convolution(
        op_call, tuple(op_info.local_args), op_info.local_kwargs
    )

    # 封装结果并返回
    return dtensor.DTensor._op_dispatcher.wrap(
        local_results, output_sharding.output_spec
    )


# 处理卷积反向传播操作的函数，接受一个操作调用、参数元组和关键字参数字典，并返回一个对象
def convolution_backward_handler(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> object:
    # 将 grad_output 张量重新分布到与输入张量相同的位置
    args = list(args)
    assert isinstance(args[0], dtensor.DTensor) and isinstance(args[1], dtensor.DTensor)
    args[0] = args[0].redistribute(args[1].device_mesh, args[1].placements)
    args = tuple(args)

    # 解包操作调用、参数和关键字参数到 OpInfo 对象
    op_info = dtensor.DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)

    # 分片传播
    dtensor.DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"

    # 局部传播反向操作
    local_results = tp_convolution_backward(
        op_call, tuple(op_info.local_args), op_info.local_kwargs
    )

    # 封装结果并返回
    return dtensor.DTensor._op_dispatcher.wrap(
        local_results, output_sharding.output_spec
    )
```