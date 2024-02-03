# `bitsandbytes\tests\test_linear8bitlt.py`

```py
# 导入所需的库
from contextlib import nullcontext
import os
from tempfile import TemporaryDirectory

import pytest
import torch

# 导入自定义的模块和函数
import bitsandbytes as bnb
from bitsandbytes import functional as F
from bitsandbytes.autograd import get_inverse_transform_indices, undo_layout
from bitsandbytes.nn.modules import Linear8bitLt
from tests.helpers import TRUE_FALSE, id_formatter

# 由 Alex Borzunov 贡献，参见链接
# https://github.com/bigscience-workshop/petals/blob/main/tests/test_linear8bitlt.py

# 标记为跳过测试条件
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (7, 5),
    reason="this test requires a turing-generation or newer GPU, see bitsandbytes docs",
)
def test_layout_exact_match():
    # 创建输入张量 x
    x = (torch.randn(14336 * 3, 14336) * 10).to(torch.int8).cuda()
    # 遍历不同的 tile_size 和 order 组合
    for tile_size, order in ((8, 32), "col_turing"), ((32, 32), "col_ampere"):
        # 定义变换函数 transform
        transform = lambda x: F.transform(x.cuda(), from_order="row", to_order=order)[0].to(x.device)
        # 获取逆变换索引
        tile_indices = get_inverse_transform_indices(transform, tile_size)
        # 对输入张量进行变换
        cxb = transform(x)

        # 同步 CUDA 设备
        torch.cuda.synchronize()
        # 恢复原始布局
        restored_x = undo_layout(cxb, tile_indices)
        # 同步 CUDA 设备
        torch.cuda.synchronize()
        # 断言恢复后的张量是连续的
        assert restored_x.is_contiguous()
        # 断言恢复后的张量与原始张量相等
        assert torch.all(torch.eq(restored_x, x))


def test_linear_no_igemmlt():
    # 创建标准的线性层
    linear = torch.nn.Linear(1024, 3072)
    # 创建输入张量 x
    x = torch.randn(3, 1024, dtype=torch.half)
    # 创建自定义的 Linear8bitLt 层
    linear_custom = Linear8bitLt(
        linear.in_features,
        linear.out_features,
        linear.bias is not None,
        has_fp16_weights=False,
        threshold=6.0,
    )
    # 强制禁用 igemm_lt
    linear_custom.state.force_no_igemmlt = True

    # 设置自定义层的权重和偏置
    linear_custom.weight = bnb.nn.Int8Params(
        linear.weight.data.clone(), requires_grad=False, has_fp16_weights=False
    ).to(linear.weight.dtype)
    linear_custom.bias = linear.bias
    # 将自定义层移动到 CUDA 设备
    linear_custom = linear_custom.cuda()
    # 将标准线性层转换为半精度并移动到 CUDA 设备
    linear = linear.half().cuda()

    # 创建输入张量的副本，并移动到 CUDA 设备，设置为需要梯度
    x_ref = x.clone().cuda().requires_grad_(True)
    x_ours = x.clone().cuda().requires_grad_(True)
    # 计算线性模型在参考输入上的输出
    fx_ref = linear(x_ref).float()
    # 生成一个与 fx_ref 相同形状的随机梯度向量
    grad_proj = torch.randn_like(fx_ref)
    # 计算 fx_ref 与 grad_proj 的点积的均值，并进行反向传播
    (fx_ref * grad_proj).mean().backward()
    
    # 计算自定义线性模型在我们自己的输入上的输出
    fx_ours = linear_custom(x_ours).float()
    # 计算 fx_ours 与 grad_proj 的点积的均值，并进行反向传播
    (fx_ours * grad_proj).mean().backward()
    # 断言 fx_ref 与 fx_ours 之间的值在指定的容差范围内相等
    assert torch.allclose(fx_ref, fx_ours, atol=0.02)
    # 断言 x_ref 的梯度与 x_ours 的梯度之间的值在指定的容差范围内相等
    assert torch.allclose(x_ref.grad, x_ours.grad, atol=0.01)
    # 断言自定义线性模型的状态中没有使用 FP16 权重
    assert not linear_custom.state.has_fp16_weights
    # 断言自定义线性模型的状态中 CB 不为 None
    assert linear_custom.state.CB is not None
    # 断言自定义线性模型的状态中 CxB 为 None
    assert linear_custom.state.CxB is None
# 使用 pytest.mark.parametrize 来为测试用例传递参数，参数为布尔值，用于测试不同情况下的线性序列化
@pytest.mark.parametrize("has_fp16_weights", TRUE_FALSE, ids=id_formatter("has_fp16_weights"))
@pytest.mark.parametrize("serialize_before_forward", TRUE_FALSE, ids=id_formatter("serialize_before_forward"))
@pytest.mark.parametrize("deserialize_before_cuda", TRUE_FALSE, ids=id_formatter("deserialize_before_cuda"))
@pytest.mark.parametrize("force_no_igemmlt", TRUE_FALSE, ids=id_formatter("force_no_igemmlt"))
def test_linear_serialization(has_fp16_weights, serialize_before_forward, deserialize_before_cuda, force_no_igemmlt):
    # 创建一个输入维度为32，输出维度为96的线性层
    linear = torch.nn.Linear(32, 96)
    # 生成一个形状为(3, 32)的随机张量，数据类型为 torch.half
    x = torch.randn(3, 32, dtype=torch.half)

    # 创建一个自定义的 Linear8bitLt 层，参数与 linear 层相同，设置是否有 fp16 权重和阈值为6.0
    linear_custom = Linear8bitLt(
        linear.in_features,
        linear.out_features,
        linear.bias is not None,
        has_fp16_weights=has_fp16_weights,
        threshold=6.0,
    )
    # 如果设置了强制不使用 igemmlt，则将 linear_custom 的状态设置为强制不使用 igemmlt
    if force_no_igemmlt:
        linear_custom.state.force_no_igemmlt = True

    # 将 linear_custom 的权重设置为 Int8Params 类型，克隆 linear 的权重数据，并设置是否需要梯度和是否有 fp16 权重
    linear_custom.weight = bnb.nn.Int8Params(
        linear.weight.data.clone(), requires_grad=has_fp16_weights, has_fp16_weights=has_fp16_weights
    )
    # 将 linear_custom 的偏置设置为 linear 的偏置
    linear_custom.bias = linear.bias
    # 将 linear_custom 移动到 GPU 上
    linear_custom = linear_custom.cuda()

    # 如果设置了在前向传播之前序列化，则获取 linear_custom 的状态字典
    if serialize_before_forward:
        state_dict_8bit = linear_custom.state_dict()

    # 克隆 x 并移动到 GPU 上，并设置需要梯度
    x_first = x.clone().cuda().requires_grad_(True)
    # 对 x_first 进行前向传播得到 fx_first，并转换为 float 类型
    fx_first = linear_custom(x_first).float()
    # 生成一个与 fx_first 形状相同的随机梯度张量 grad_proj
    grad_proj = torch.randn_like(fx_first)
    # 计算 fx_first 和 grad_proj 的点乘的均值，并进行反向传播
    (fx_first * grad_proj).mean().backward()

    # 如果没有设置在前向传播之前序列化，则获取 linear_custom 的状态字典
    if not serialize_before_forward:
        state_dict_8bit = linear_custom.state_dict()

    # 使用临时目录来保存模型状态字典
    with TemporaryDirectory() as tmpdir:
        state_path_8bit = os.path.join(tmpdir, "state_8bit.pth")
        state_path = os.path.join(tmpdir, "state.pth")

        # 保存原始 linear 的状态字典到 state_path
        torch.save(linear.state_dict(), state_path)
        # 保存 state_dict_8bit 到 state_path_8bit
        torch.save(state_dict_8bit, state_path_8bit)

        # 如果没有 fp16 权重，则断言 state_path_8bit 文件大小小于 state_path 文件大小的一半
        if not has_fp16_weights:
            assert os.path.getsize(state_path_8bit) < 0.5 * os.path.getsize(state_path)

        # 加载 state_path_8bit 的状态字典到 new_state_dict
        new_state_dict = torch.load(state_path_8bit)
    # 创建一个新的自定义的8位整数线性层，参数包括输入特征数、输出特征数、是否有偏置、是否有FP16权重、阈值为6.0
    new_linear_custom = Linear8bitLt(
        linear.in_features,
        linear.out_features,
        linear.bias is not None,
        has_fp16_weights=has_fp16_weights,
        threshold=6.0,
    )
    # 如果强制禁用igemmlt，则设置新线性层的状态为强制禁用igemmlt
    if force_no_igemmlt:
        new_linear_custom.state.force_no_igemmlt = True

    # 如果在将模型加载到CUDA设备之前进行反序列化
    if deserialize_before_cuda:
        # 如果有FP16权重，则使用nullcontext()，否则捕获RuntimeError异常
        with nullcontext() if has_fp16_weights else pytest.raises(RuntimeError):
            # 加载新的状态字典到新的线性层，严格模式
            new_linear_custom.load_state_dict(new_state_dict, strict=True)

    # 将新的线性层加载到CUDA设备
    new_linear_custom = new_linear_custom.cuda()

    # 如果不是在将模型加载到CUDA设备之前进行反序列化
    if not deserialize_before_cuda:
        # 加载新的状态字典到新的线性层，严格模式
        new_linear_custom.load_state_dict(new_state_dict, strict=True)

    # 克隆输入张量x，并将其加载到CUDA设备上，并设置requires_grad为True
    x_second = x.clone().cuda().requires_grad_(True)
    # 使用新的线性层对x_second进行前向传播，并将结果转换为浮点数
    fx_second = new_linear_custom(x_second).float()
    # 计算fx_second和grad_proj的点乘的均值，并进行反向传播
    (fx_second * grad_proj).mean().backward()

    # 如果有FP16权重或者不是在将模型加载到CUDA设备之前进行反序列化
    if has_fp16_weights or not deserialize_before_cuda:
        # 断言fx_first和fx_second在指定的误差范围内相等
        assert torch.allclose(fx_first, fx_second, atol=1e-5)
        # 断言x_first的梯度和x_second的梯度在指定的误差范围内相等
        assert torch.allclose(x_first.grad, x_second.grad, atol=1e-5)
```