# `.\pytorch\test\quantization\core\test_workflow_ops.py`

```
# Owner(s): ["oncall: quantization"]

# 引入 PyTorch 相关模块
import torch
import math
from typing import Tuple
from torch.ao.quantization import (
    FakeQuantize,
    MovingAverageMinMaxObserver,
    default_observer,
    default_fixed_qparams_range_0to1_fake_quant,
)

# 引入特定的量化相关模块和函数
from torch.ao.quantization._learnable_fake_quantize import _LearnableFakeQuantize
from torch.testing._internal.common_quantized import (
    _fake_quantize_per_channel_affine_reference,
    _fake_quantize_per_channel_affine_grad_reference,
    to_tensor,
)
import torch.nn as nn

# 标准库引入
import io
import itertools
import unittest
import numpy as np

# 测试工具引入
from hypothesis import given, settings
from hypothesis import strategies as st
import torch.testing._internal.hypothesis_utils as hu
hu.assert_deadline_disabled()  # 禁用测试时限检测
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import TestCase, skipIfTorchDynamo

# 参考方法：对于整张张量的仿量化
# 注意：因为实际内核中的 scale/zero_point 保留为浮点数，这里模拟了对 float16/64 的仿量化操作
def _fake_quantize_per_tensor_affine_reference(X, scale, zero_point, quant_min, quant_max):
    dtype = X.dtype
    # 对输入张量 X 进行仿量化操作，返回仿量化后的结果
    res = ((torch.clamp(torch.round(X.to(torch.float32) * (1.0 / scale) + zero_point), quant_min, quant_max) - zero_point) * scale)
    return res.to(dtype)

# 参考方法：仿量化操作的梯度
# 注意：因为实际内核中的 scale/zero_point 保留为浮点数，这里模拟了对 float16/64 的仿量化梯度操作
def _fake_quantize_per_tensor_affine_grad_reference(dY, X, scale, zero_point, quant_min, quant_max):
    dtype = X.dtype
    Xq = torch.round(X.to(torch.float32) * (1.0 / scale) + zero_point)
    mask = (Xq >= quant_min) * (Xq <= quant_max)
    res = torch.zeros_like(dY)
    res[mask] = dY[mask]
    return res.to(dtype)

# 参考方法：仿量化操作的可学习版本梯度
def _fake_quantize_learnable_per_tensor_affine_grad_reference(dY, X, scale, zero_point, quant_min, quant_max, device):
    r"""This method references the following literatures for back propagation on scale and zero point.
    - https://arxiv.org/pdf/1902.08153.pdf
    - https://arxiv.org/pdf/1903.08066.pdf
    """
    # 对 zero_point 进行四舍五入，并确保在 quant_min 和 quant_max 之间
    zero_point_rounded = int((zero_point + 0.5).clamp(quant_min, quant_max).item())
    Xq = torch.round(X * (1.0 / scale) + zero_point_rounded)

    # 根据仿量化后的 Xq 值，生成对 scale 和 zero_point 的梯度指示
    indicate_small_scale = (Xq < quant_min).float().to(device)
    indicate_big_scale = (Xq > quant_max).float().to(device)
    indicate_middle_scale = torch.ones(indicate_small_scale.shape).to(device) - \
        indicate_small_scale - indicate_big_scale

    indicate_saturate_zp = ((Xq < quant_min).float() + (Xq > quant_max).float()).to(device)
    indicate_unsaturate_zp = torch.ones(indicate_saturate_zp.shape).to(device) - indicate_saturate_zp

    # 将 Xq 限制在 quant_min 和 quant_max 之间，并进行仿量化操作
    Xq = Xq.clamp(quant_min, quant_max)
    Xfq = (Xq - zero_point_rounded) * scale

    # 计算小尺度的 scale 梯度
    grad_small_scale = quant_min - zero_point_rounded
    # 计算梯度的大尺度，即量化最大值减去四舍五入的零点
    grad_big_scale = quant_max - zero_point_rounded
    
    # 计算梯度的中间尺度，即 (Xfq - X) / scale，并将结果移动到指定设备上
    grad_middle_scale = ((Xfq - X) / scale).to(device)

    # 梯度饱和的零点，设为负的量化尺度，并移动到指定设备上
    grad_saturate_zp = -scale.to(device)
    
    # 梯度非饱和的零点，设为零
    grad_unsaturate_zp = 0

    # 计算梯度的尺度，结合指示小尺度、大尺度和中间尺度的梯度
    grad_scale = indicate_small_scale * grad_small_scale + \
        indicate_big_scale * grad_big_scale + \
        indicate_middle_scale * grad_middle_scale
    
    # 计算梯度的零点，结合指示饱和和非饱和的梯度
    grad_zp = indicate_saturate_zp * grad_saturate_zp + \
        indicate_unsaturate_zp * grad_unsaturate_zp
    
    # 计算输入张量 X 的梯度，调用 _fake_quantize_per_tensor_affine_grad_reference 函数
    # 传入相关参数，并将结果移动到指定设备上
    grad_X = _fake_quantize_per_tensor_affine_grad_reference(
        dY, X, scale, zero_point, quant_min, quant_max).to(device)

    # 计算尺度梯度的总和，并在第一个维度上增加一个维度
    grad_scale = (grad_scale * dY).sum().unsqueeze(dim=0)
    
    # 计算零点梯度的总和，并在第一个维度上增加一个维度
    grad_zp = (grad_zp * dY).sum().unsqueeze(dim=0)
    
    # 返回计算得到的梯度：X 的梯度、尺度的梯度、零点的梯度
    return grad_X, grad_scale, grad_zp
# 定义了一个用于量化的每张量方法，基于给定的量化参数对张量 x 进行量化
def _quantize_per_tensor(x, scale, zero_point, quant_min, quant_max):
    return ((x / scale) + zero_point).round().clamp(quant_min, quant_max)

# 定义了一个参考方法，用于可学习的每通道仿真量化操作符的梯度反向传播
def _fake_quantize_learnable_per_channel_affine_grad_reference(
        dY, X, per_channel_scale, per_channel_zero_point, axis, quant_min, quant_max, device):
    r"""This method references the following literatures for back propagation on scale and zero point.
    - https://arxiv.org/pdf/1902.08153.pdf
    - https://arxiv.org/pdf/1903.08066.pdf
    """
    # 将每通道的零点偏移量加上0.5后，限制在量化范围内，并转换为整数类型
    per_channel_zero_point = ((per_channel_zero_point.detach() + 0.5).clamp(quant_min, quant_max)).type(torch.int32)
    # 调用另一个方法计算每通道仿真量化操作的梯度
    grad_X = _fake_quantize_per_channel_affine_grad_reference(
        dY, X, per_channel_scale, per_channel_zero_point, axis, quant_min, quant_max).to(device)
    # 分离每通道的尺度参数，并转换为浮点数类型
    per_channel_scale = per_channel_scale.detach().type(torch.float)

    # 创建用于存储尺度和零点梯度的零张量
    grad_scale = torch.zeros([per_channel_scale.size(0)]).to(device)
    grad_zero_point = torch.zeros([per_channel_zero_point.size(0)]).to(device)

    # 按轴展开张量 X 和 dY
    X_flattened = torch.unbind(X, dim=axis)
    dY_flattened = torch.unbind(dY, dim=axis)

    # 对每个元素进行迭代，计算量化后的张量和仿真后的张量
    for i, X_i in enumerate(torch.unbind(X, dim=axis), 0):
        scale_i = per_channel_scale[i]
        zero_point_i = per_channel_zero_point[i]
        X_i = X_flattened[i]
        dY_i = dY_flattened[i]

        Xq_i = ((X_i / scale_i) + zero_point_i).round()  # 量化操作
        Xfq_i = (Xq_i - zero_point_i) * scale_i  # 仿真操作

        # 计算指示量化小尺度、大尺度和中间尺度的指示器
        indicate_small_scale_i = (Xq_i < quant_min).float().to(device)
        indicate_big_scale_i = (Xq_i > quant_max).float().to(device)
        indicate_middle_scale_i = torch.ones(indicate_small_scale_i.shape).to(device) - \
            indicate_small_scale_i - indicate_big_scale_i

        # 计算指示饱和零点和未饱和零点的指示器
        indicate_saturate_zp_i = ((Xq_i < quant_min).float() +
                                  (Xq_i > quant_max).float()).to(device)
        indicate_unsaturate_zp_i = torch.ones(indicate_saturate_zp_i.shape).to(device) - \
            indicate_saturate_zp_i

        # 将量化后的值限制在量化范围内
        Xq_i = Xq_i.clamp(quant_min, quant_max)
        Xfq_i = (Xq_i - zero_point_i) * scale_i

        # 计算小尺度、大尺度和中间尺度的梯度
        grad_small_scale_i = quant_min - zero_point_i
        grad_big_scale_i = quant_max - zero_point_i
        grad_middle_scale_i = ((Xfq_i - X_i) / scale_i).to(device)

        # 计算饱和零点和未饱和零点的梯度
        grad_saturate_zp_i = -scale_i.to(device)
        grad_unsaturate_zp_i = 0

        # 计算尺度和零点的梯度
        grad_scale_i = indicate_small_scale_i * grad_small_scale_i + \
            indicate_middle_scale_i * grad_middle_scale_i + \
            indicate_big_scale_i * grad_big_scale_i
        grad_zp_i = indicate_saturate_zp_i * grad_saturate_zp_i + \
            indicate_unsaturate_zp_i * grad_unsaturate_zp_i

        # 计算尺度和零点的总梯度
        grad_scale_i = (grad_scale_i * dY_i).sum().unsqueeze(dim=0)
        grad_zp_i = (grad_zp_i * dY_i).sum().unsqueeze(dim=0)

        # 将计算得到的梯度存储在对应的位置
        grad_scale[i] = grad_scale_i
        grad_zero_point[i] = grad_zp_i
    # 返回三个变量 grad_X, grad_scale, grad_zero_point，这些变量分别是梯度相关的数值
    return grad_X, grad_scale, grad_zero_point
# 定义一个函数 _get_tensor_min_max，计算给定张量 X 的最小值和最大值
def _get_tensor_min_max(
        X: torch.Tensor,
        running_min: float = float("inf"),
        running_max: float = float("-inf"),
        averaging_const: float = 0.01) -> Tuple[float, float]:
    # 计算张量 X 的最小值，并将其转换为 float32 类型后获取其数值
    min_val = X.min().to(dtype=torch.float32).item()
    # 计算张量 X 的最大值，并将其转换为 float32 类型后获取其数值
    max_val = X.max().to(dtype=torch.float32).item()

    # 如果 running_min 不是正无穷，则使用移动平均计算新的最小值
    if not math.isinf(running_min):
        min_val = running_min + averaging_const * (min_val - running_min)
    # 如果 running_max 不是负无穷，则使用移动平均计算新的最大值
    if not math.isinf(running_max):
        max_val = running_max + averaging_const * (max_val - running_max)

    # 返回计算得到的最小值和最大值
    return min_val, max_val

# 定义一个函数 _get_per_row_min_max，计算张量 x 沿指定轴的每行的最小值和最大值
def _get_per_row_min_max(
        x: torch.Tensor,
        min_vals: torch.Tensor,
        max_vals: torch.Tensor,
        axis: int = 0,
        averaging_const: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    # 获取张量 x 的维度信息
    x_dim = x.size()
    # 创建一个新的轴顺序列表，用于对 x 进行轴置换
    new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
    new_axis_list[axis] = 0
    new_axis_list[0] = axis
    y = x.permute(*new_axis_list)

    # 将 y 沿第一个维度展平成二维张量
    y = torch.flatten(y, start_dim=1)

    # 如果 min_vals 或 max_vals 中的第一个元素为正无穷，则重新计算全局最小值和最大值
    if math.isinf(min_vals[0]) or math.isinf(max_vals[0]):
        min_vals, max_vals = torch.aminmax(y, dim=1)
    else:
        # 否则，计算当前最小值和最大值，并进行移动平均更新
        min_vals_cur, max_vals_cur = torch.aminmax(y, dim=1)
        min_vals = min_vals + averaging_const * (min_vals_cur - min_vals)
        max_vals = max_vals + averaging_const * (max_vals_cur - max_vals)
    
    # 返回计算得到的每行最小值和最大值张量
    return min_vals, max_vals

# 定义一个函数 _get_scale_zp，计算张量的量化参数（scale, zero_point）
def _get_scale_zp(
        min_val: float,
        max_val: float,
        dtype: torch.dtype,
        reduce_range: bool = False,
        preserve_sparsity: bool = False) -> Tuple[float, int]:
    """
    Calculate the quantization parameters (scale, zero_point)
    based on the min and max element of the tensor
    """
    # 根据张量的数据类型选择量化范围
    if dtype == torch.qint8:
        if reduce_range:
            qmin, qmax = -64, 63
        else:
            qmin, qmax = -128, 127
    else:
        if reduce_range:
            qmin, qmax = 0, 127
        else:
            qmin, qmax = 0, 255

    # 如果最小值小于 0 且最大值大于 0 并且保持稀疏性为真，则根据稀疏性更新最小值和最大值
    if min_val < 0 and max_val > 0 and preserve_sparsity:
        symmetric_qmin = int(-((qmax - qmin) / 2 + 1))
        symmetric_qmax = int((qmax - qmin) / 2)
        max_scale = max(
            abs(min_val / symmetric_qmin), abs(max_val / symmetric_qmax)
        )
        min_val = max_scale * symmetric_qmin
        max_val = max_scale * symmetric_qmax
    
    # 确保最小值和最大值在 0 以下和以上
    min_val = min(min_val, 0.0)
    max_val = max(max_val, 0.0)

    # 计算量化的缩放因子 scale
    scale = (max_val - min_val) / (qmax - qmin)

    # 如果 scale 为 0 或者倒数无穷大，则设定默认的 scale 和 zero_point
    if scale == 0.0 or math.isinf(1.0 / scale):
        scale = 0.1
        zero_point = 0
    else:
        # 否则，根据最小值计算初始的 zero_point
        zero_point_from_min = qmin - min_val / float(scale)
        zero_point_from_max = qmax - max_val / float(scale)
        zero_point_from_min_error = abs(qmin) - abs(min_val / float(scale))
        zero_point_from_max_error = abs(qmax) - abs(max_val / float(scale))
        # 根据误差确定初始的 zero_point
        if zero_point_from_min_error < zero_point_from_max_error:
            initial_zero_point = zero_point_from_min
        else:
            initial_zero_point = zero_point_from_max
    # 如果最小值小于 0，最大值大于 0，并且需要保持稀疏性（即保持零点不变）
    if min_val < 0 and max_val > 0 and preserve_sparsity:
        # 计算初始零点：取量化范围的中点再加1
        initial_zero_point = (qmin + qmax) / 2 + 1
    
    # 初始化零点的变量
    nudged_zero_point = 0
    
    # 调整零点，确保在量化范围内
    if initial_zero_point < qmin:
        # 如果初始零点小于量化范围的最小值，则将调整后的零点设为量化范围的最小值
        nudged_zero_point = qmin
    elif initial_zero_point > qmax:
        # 如果初始零点大于量化范围的最大值，则将调整后的零点设为量化范围的最大值
        nudged_zero_point = qmax
    else:
        # 否则，将初始零点四舍五入后作为调整后的零点
        nudged_zero_point = int(round(initial_zero_point))
    
    # 返回量化参数的缩放因子和调整后的零点值
    return (scale, int(nudged_zero_point))
NP_RANDOM_SEED = 19
tolerance = 1e-6

class TestFakeQuantizeOps(TestCase):
    @given(device=st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']),
           X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       qparams=hu.qparams(dtypes=torch.quint8)))
    def test_forward_per_tensor(self, device, X):
        r"""Tests the forward path of the FakeQuantizePerTensorAffine op.
        """
        np.random.seed(NP_RANDOM_SEED)  # 设置随机种子为19
        X, (scale, zero_point, torch_type) = X  # 解包输入张量 X 和量化参数
        quant_min = torch.iinfo(torch_type).min  # 获取量化的最小值
        quant_max = torch.iinfo(torch_type).max  # 获取量化的最大值

        X = to_tensor(X, device)  # 将输入张量 X 转移到指定设备
        Y = _fake_quantize_per_tensor_affine_reference(X.cpu(), scale, zero_point, quant_min, quant_max)  # 使用参考方法计算仿真量化结果
        Y_prime = torch.fake_quantize_per_tensor_affine(
            X, scale, zero_point, quant_min, quant_max)  # 使用 PyTorch 内置方法计算仿真量化结果
        np.testing.assert_allclose(Y, Y_prime.cpu(), rtol=tolerance, atol=tolerance)  # 检查两种方法的结果是否接近

    @given(device=st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']),
           X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       qparams=hu.qparams(dtypes=torch.quint8)))
    @unittest.skip("temporarily disable the test")  # 暂时禁用该测试
    def test_backward_per_tensor(self, device, X):
        r"""Tests the backward method.
        """
        np.random.seed(NP_RANDOM_SEED)  # 设置随机种子为19
        X, (scale, zero_point, torch_type) = X  # 解包输入张量 X 和量化参数
        quant_min = torch.iinfo(torch_type).min  # 获取量化的最小值
        quant_max = torch.iinfo(torch_type).max  # 获取量化的最大值

        X = to_tensor(X, device)  # 将输入张量 X 转移到指定设备
        X.requires_grad_()  # 设置输入张量 X 需要梯度计算
        Y = _fake_quantize_per_tensor_affine_reference(X.cpu(), scale, zero_point, quant_min, quant_max)  # 使用参考方法计算仿真量化结果
        Y_prime = torch.fake_quantize_per_tensor_affine(
            X, scale, zero_point, quant_min, quant_max)  # 使用 PyTorch 内置方法计算仿真量化结果
        dout = torch.rand_like(X, dtype=torch.float).to(device)  # 生成与 X 形状相同的随机梯度
        dX = _fake_quantize_per_tensor_affine_grad_reference(
            dout, X, scale, zero_point, quant_min, quant_max)  # 使用参考方法计算仿真量化梯度
        Y_prime.backward(dout)  # 对 Y_prime 执行反向传播
        np.testing.assert_allclose(dX.cpu(), X.grad.cpu().detach().numpy(), rtol=tolerance, atol=tolerance)  # 检查计算的梯度是否接近 PyTorch 计算的梯度

    def test_forward_backward_per_tensor_with_amp(self):
        net = nn.Sequential(nn.Conv2d(1, 1, 3))  # 创建包含卷积层的神经网络
        net.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')  # 配置量化感知训练的默认配置

        net_prep = torch.ao.quantization.prepare_qat(net)  # 准备网络进行量化感知训练

        with torch.cuda.amp.autocast():  # 使用自动混合精度加速训练过程
            x = torch.randn(4, 1, 5, 5)  # 创建输入张量 x
            out = net_prep(x).sum()  # 将输入 x 经过准备后的网络 net_prep 运行，并对输出进行求和
            out.backward()  # 对输出进行反向传播
            self.assertTrue(net_prep[0].weight.grad is not None)  # 断言卷积层的权重梯度不为空
    # 测试单张量半精度数值的前向传播
    def test_forward_per_tensor_half_precision_numerics(self):
        scale = .1  # 定义缩放因子
        zero = 0  # 定义零点
        maxi = 255  # 定义最大量化值
        mini = 0  # 定义最小量化值

        for i in range(20):
            X1 = torch.randn(5, 5).to(torch.float16)  # 生成随机张量，并转换为半精度
            Y1 = torch.fake_quantize_per_tensor_affine(X1, scale, zero, mini, maxi)  # 使用仿真量化函数量化张量
            Y1r = _fake_quantize_per_tensor_affine_reference(X1, scale, zero, mini, maxi)  # 调用参考函数进行仿真量化
            self.assertEqual(Y1, Y1r, rtol=tolerance, atol=tolerance)  # 断言量化结果与参考结果相等

        # to force overflow
        X2 = torch.tensor(2**15 + .01).to(torch.float16)  # 创建一个超出半精度范围的张量
        Y2 = torch.fake_quantize_per_tensor_affine(X2, scale, zero, mini, maxi)  # 使用仿真量化函数量化张量
        Y2r = _fake_quantize_per_tensor_affine_reference(X2, scale, zero, mini, maxi)  # 调用参考函数进行仿真量化
        self.assertEqual(Y2, Y2r, rtol=tolerance, atol=tolerance)  # 断言量化结果与参考结果相等

        scale = 10  # 更新缩放因子

        # to force underflow
        X3 = torch.tensor(2**-24).to(torch.float16)  # 创建一个导致欠流的半精度张量
        Y3 = torch.fake_quantize_per_tensor_affine(X3, scale, zero, mini, maxi)  # 使用仿真量化函数量化张量
        Y3r = _fake_quantize_per_tensor_affine_reference(X3, scale, zero, mini, maxi)  # 调用参考函数进行仿真量化
        self.assertEqual(Y3, Y3r, rtol=tolerance, atol=tolerance)  # 断言量化结果与参考结果相等

    # 测试单张量缓存掩码实现的前向传播，根据设备选择不同的数据类型和张量
    def _test_forward_per_tensor_cachemask_impl(self, device):
        float_types = (torch.float32, torch.float16, torch.float64)  # 定义浮点数据类型
        torch_types = (torch.qint8, torch.quint8)  # 定义 PyTorch 张量类型
        Xs = (torch.randn(4, 8, device=device), torch.randn(4, 16, device=device)[:, ::2])  # 创建不同形状的张量
        tensor_qparam = (True, False)  # 定义张量量化参数的选择
        for float_type, torch_type, X, tensor_qparams in itertools.product(float_types, torch_types, Xs, tensor_qparam):
            # pick the scale + zp so that some values get clipped
            X = X.to(float_type)  # 将张量转换为指定的浮点数据类型
            obs = torch.ao.quantization.MinMaxObserver(torch_type)  # 创建观察器对象
            obs.to(device)  # 设置观察器对象所在的设备
            obs(X * 0.75)  # 对输入张量的观察器进行操作
            scale, zero_point = obs.calculate_qparams()  # 计算量化参数的缩放因子和零点
            quant_min, quant_max = obs.quant_min, obs.quant_max  # 获取量化的最小值和最大值
            if not tensor_qparam:
                scale, zero_point = float(scale), int(zero_point)  # 根据需要转换量化参数的类型
            Y_test = torch.fake_quantize_per_tensor_affine(
                X, scale, zero_point, quant_min, quant_max)  # 使用仿真量化函数量化张量
            Y_ref = _fake_quantize_per_tensor_affine_reference(
                X, scale, zero_point, quant_min, quant_max).to(device)  # 调用参考函数进行仿真量化，并设置在指定设备上
            self.assertEqual(Y_test, Y_ref, rtol=tolerance, atol=tolerance)  # 断言量化结果与参考结果相等
            self.assertTrue(Y_test.dtype == float_type)  # 断言量化后的张量数据类型与指定的浮点类型一致

    # 测试 CPU 环境下的单张量缓存掩码实现的前向传播
    def test_forward_per_tensor_cachemask_cpu(self):
        device = torch.device('cpu')  # 设置设备为 CPU
        self._test_forward_per_tensor_cachemask_impl(device)  # 调用实现方法进行测试

    # 如果 CUDA 可用，测试 CUDA 环境下的单张量缓存掩码实现的前向传播
    @unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
    def test_forward_per_tensor_cachemask_cuda(self):
        device = torch.device('cuda')  # 设置设备为 CUDA
        self._test_forward_per_tensor_cachemask_impl(device)  # 调用实现方法进行测试
    # 定义一个测试方法，用于测试基于每个张量的反向传播缓存掩码功能
    def _test_backward_per_tensor_cachemask_impl(self, device):
        # 定义浮点类型的元组
        float_types = (torch.float32, torch.float16, torch.float64)
        # 定义 torch 类型的元组
        torch_types = (torch.qint8, torch.quint8)
        # 定义张量量化参数的元组
        tensor_qparams = (True, False)
        # 使用 itertools.product 创建所有可能的组合
        for float_type, torch_type, tensor_qparam in itertools.product(float_types, torch_types, tensor_qparams):
            # 创建一个形状为 (4, 8) 的随机张量，并将其转移到指定设备并使用指定浮点类型
            X = torch.randn(4, 8).to(device).to(float_type)
            # 设置张量为可求导状态
            X.requires_grad_()
            # 创建一个 MinMaxObserver 实例，基于指定的 torch 类型
            obs = torch.ao.quantization.MinMaxObserver(torch_type)
            # 将 MinMaxObserver 实例移到指定设备
            obs.to(device)
            # 对张量 X 的观察，计算量化参数 scale 和 zero_point，用于一部分值的裁剪
            obs(X * 0.75)
            scale, zero_point = obs.calculate_qparams()
            # 如果不使用张量量化参数，则将 scale 转为 float 类型，zero_point 转为 int 类型
            if not tensor_qparam:
                scale, zero_point = float(scale), int(zero_point)
            # 获取 MinMaxObserver 的量化范围
            quant_min, quant_max = obs.quant_min, obs.quant_max

            # 正向传播
            Y_test = torch.fake_quantize_per_tensor_affine(
                X, scale, zero_point, quant_min, quant_max)
            # 使用参考函数计算 Y 的预期值，并将其转移到指定设备
            Y_ref = _fake_quantize_per_tensor_affine_reference(
                X, scale, zero_point, quant_min, quant_max).to(device)
            # 断言 Y_test 和 Y_ref 在给定容差下相等
            self.assertEqual(Y_test, Y_ref, rtol=tolerance, atol=tolerance)

            # 反向传播
            dout = torch.rand_like(X, dtype=torch.float).to(device)
            # 使用参考函数计算反向传播梯度 dX
            dX = _fake_quantize_per_tensor_affine_grad_reference(
                dout, X, scale, zero_point, quant_min, quant_max)
            # 对 Y_test 进行反向传播，并断言计算得到的梯度与预期值 X.grad 相等
            Y_test.backward(dout)
            self.assertEqual(dX, X.grad)
            # 断言 X.grad 的数据类型为 float_type
            self.assertTrue(X.grad.dtype == float_type)

    # 在 CPU 上执行基于每个张量的反向传播缓存掩码测试
    def test_backward_per_tensor_cachemask_cpu(self):
        device = torch.device('cpu')
        self._test_backward_per_tensor_cachemask_impl(device)

    # 如果 CUDA 可用，则在 CUDA 上执行基于每个张量的反向传播缓存掩码测试
    @unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
    def test_backward_per_tensor_cachemask_cuda(self):
        device = torch.device('cuda')
        self._test_backward_per_tensor_cachemask_impl(device)

    # 定义一个测试方法，用于测试可学习的每张量前向量化
    def _test_learnable_forward_per_tensor(self, X, device, scale_base, zero_point_base):
        # 将输入张量 X 转换为指定设备上的 torch 张量
        X_base = torch.tensor(X).to(device)

        # 遍历不同位数的量化
        for n_bits in (4, 8):
            # 计算量化范围的最小值和最大值
            quant_min, quant_max = 0, 2 ** n_bits - 1

            # 克隆 X_base，并转换为 float 类型的张量
            X = X_base.clone().float()
            # 将 scale_base 转移到指定设备并转换为 float 类型
            scale_base = scale_base.to(device).float()
            # 将 zero_point_base 转移到指定设备并转换为 torch.int32 类型
            zero_point_base = zero_point_base.to(dtype=torch.int32, device=device)
            # 克隆 scale_base 和 zero_point_base
            scale = scale_base.clone()
            zero_point = zero_point_base.clamp(quant_min, quant_max)

            # 使用参考函数计算 Y 的预期值，并将其转移到指定设备
            Y = _fake_quantize_per_tensor_affine_reference(
                X, scale, zero_point, quant_min, quant_max).to(device)
            
            # 遍历不同的梯度因子
            for grad_factor in [0.1, 1.0, 10.0]:
                # 使用可学习的每张量前向量化函数计算 Y_prime
                Y_prime = torch._fake_quantize_learnable_per_tensor_affine(
                    X, scale, zero_point, quant_min, quant_max, grad_factor).to(device)
                # 断言 Y_prime 与 Y 在给定容差下相等
                self.assertTrue(
                    torch.allclose(Y, Y_prime, rtol=tolerance, atol=tolerance),
                    "Expected kernel forward function to have results match the reference forward function")
    # 使用 `hypothesis` 的 `@given` 装饰器定义测试函数，生成输入数据 `X`
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       elements=hu.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                       qparams=hu.qparams(dtypes=torch.quint8)))
    # 使用 `unittest.skip` 装饰器跳过此测试，因为存在问题，需要修改相关代码后再执行
    @unittest.skip(
        "this is broken without changes to any relevant code, "
        "we need to remove hypothesis testing in CI")
    # 定义测试方法 `test_learnable_forward_per_tensor_cpu`，接收参数 `X`
    def test_learnable_forward_per_tensor_cpu(self, X):
        # 解包 `X`，获取第一个元素作为输入数据
        X, (_, _, _) = X
        # 生成标准差为 1 的正态分布随机数，限制在 [1e-4, 100] 范围内，并赋值给 `scale_base`
        scale_base = torch.normal(mean=0, std=1, size=(1,)).clamp(1e-4, 100)
        # 生成均值为 0，标准差为 128 的正态分布随机数，并赋值给 `zero_point_base`
        zero_point_base = torch.normal(mean=0, std=128, size=(1,))
        # 调用内部方法 `_test_learnable_forward_per_tensor` 进行测试
        self._test_learnable_forward_per_tensor(
            X, 'cpu', scale_base, zero_point_base)

    # 使用 `hypothesis` 的 `@given` 装饰器定义测试函数，生成输入数据 `X`
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       elements=hu.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                       qparams=hu.qparams(dtypes=torch.quint8)))
    # 如果未开启 CUDA 测试，则使用 `unittest.skipIf` 装饰器跳过测试
    @unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
    # 定义测试方法 `test_learnable_forward_per_tensor_cuda`，接收参数 `X`
    def test_learnable_forward_per_tensor_cuda(self, X):
        # 解包 `X`，获取第一个元素作为输入数据
        X, (_, _, _) = X
        # 生成标准差为 1 的正态分布随机数，限制在 [1e-4, 100] 范围内，并赋值给 `scale_base`
        scale_base = torch.normal(mean=0, std=1, size=(1,)).clamp(1e-4, 100)
        # 生成均值为 0，标准差为 128 的正态分布随机数，并赋值给 `zero_point_base`
        zero_point_base = torch.normal(mean=0, std=128, size=(1,))
        # 调用内部方法 `_test_learnable_forward_per_tensor` 进行测试
        self._test_learnable_forward_per_tensor(
            X, 'cuda', scale_base, zero_point_base)
    # 定义一个测试方法，用于测试带有每个张量的可学习后向传递的反向传播方法
    def _test_learnable_backward_per_tensor(self, X, device, scale_base, zero_point_base):
        r"""Tests the backward method with additional backprop support for scale and zero point.
        """
        # 将输入数据 X 转换为张量，并移动到指定的设备上
        X_base = torch.tensor(X).to(device)

        # 遍历不同的比特数配置
        for n_bits in (4, 8):
            quant_min, quant_max = 0, 2 ** n_bits - 1

            # 克隆并转换输入 X 为浮点张量，并设置其需要梯度计算
            X = X_base.clone().float().to(device)
            X.requires_grad_()

            # 将规模基础 scale_base 和零点基础 zero_point_base 移动到设备上，并设置需要梯度计算
            scale_base = scale_base.to(device)
            zero_point_base = zero_point_base.to(device)
            scale = scale_base.clone()
            scale.requires_grad_()
            zero_point = zero_point_base.clone().clamp(quant_min, quant_max)
            zero_point.requires_grad_()

            # 遍历梯度因子列表
            for grad_factor in [0.1, 1.0, 10.0]:
                # 调用 _fake_quantize_learnable_per_tensor_affine 函数进行量化仿真
                Y_prime = torch._fake_quantize_learnable_per_tensor_affine(
                    X, scale, zero_point, quant_min, quant_max, grad_factor).to(device)
                
                # 创建与 X 同样形状的随机梯度张量 dout
                dout = torch.rand_like(X, dtype=torch.float).to(device)
                
                # 调用 _fake_quantize_learnable_per_tensor_affine_grad_reference 获取期望的梯度值
                dX, dScale, dZeroPoint = _fake_quantize_learnable_per_tensor_affine_grad_reference(
                    dout, X, scale, zero_point, quant_min, quant_max, device)
                
                # 对 Y_prime 进行反向传播
                Y_prime.backward(dout)

                # 分别获取期望的 X 梯度、实际的 X 梯度，并进行比较
                expected_dX = dX.to(device).detach()
                actual_dX = X.grad.to(device).detach()
                self.assertTrue(
                    torch.allclose(
                        expected_dX, actual_dX, rtol=tolerance, atol=tolerance),
                    "Expected dX to match X.grad")

                # 分别获取期望的 scale 梯度、实际的 scale 梯度，并进行比较
                expected_dScale = dScale.to(device).detach()
                actual_dScale = scale.grad.to(device).detach()
                self.assertTrue(
                    torch.allclose(
                        expected_dScale * grad_factor, actual_dScale, rtol=tolerance, atol=tolerance),
                    "Expected dScale to match scale.grad")

                # 分别获取期望的 zero_point 梯度、实际的 zero_point 梯度，并进行比较
                expected_dZeroPoint = dZeroPoint.to(device).detach()
                actual_dZeroPoint = zero_point.grad.to(device).detach()
                self.assertTrue(
                    torch.allclose(
                        expected_dZeroPoint * grad_factor, actual_dZeroPoint, rtol=tolerance, atol=tolerance),
                    "Expected dZeroPoint to match zero_point.grad")

                # 将 X、scale、zero_point 的梯度数据清零，为下一次迭代做准备
                X.grad.data.zero_()
                scale.grad.data.zero_()
                zero_point.grad.data.zero_()

    # 使用 hypothesis 提供的张量 X 进行测试，运行在 CPU 上
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       elements=hu.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                       qparams=hu.qparams(dtypes=torch.quint8)))
    def test_learnable_backward_per_tensor_cpu(self, X):
        # 设置随机种子
        torch.random.manual_seed(NP_RANDOM_SEED)
        X, (_, _, _) = X
        
        # 从正态分布中生成规模基础 scale_base 和零点基础 zero_point_base，并进行限制
        scale_base = torch.normal(mean=0, std=1, size=(1,)).clamp(1e-4, 100)
        zero_point_base = torch.normal(mean=0, std=128, size=(1,))
        
        # 调用 _test_learnable_backward_per_tensor 方法进行测试，运行在 CPU 上
        self._test_learnable_backward_per_tensor(
            X, 'cpu', scale_base, zero_point_base)
    # 使用 hypothesis 库的 @given 装饰器定义测试函数，参数 X 是根据指定规则生成的张量
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       elements=hu.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                       qparams=hu.qparams(dtypes=torch.quint8)))
    # 如果 CUDA 不可用，则跳过测试
    @unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
    def test_learnable_backward_per_tensor_cuda(self, X):
        # 设置随机种子
        torch.random.manual_seed(NP_RANDOM_SEED)
        # 从 X 中解包出张量数据
        X, (_, _, _) = X
        # 创建一个服从正态分布的张量，用于量化的缩放因子，范围在 [1e-4, 100] 之间，并且将小于 1e-4 的值调整为 1e-4
        scale_base = torch.normal(mean=0, std=1, size=(1,)).clamp(1e-4, 100)
        # 创建一个服从正态分布的张量，用于量化的零点，标准差为 128
        zero_point_base = torch.normal(mean=0, std=128, size=(1,))
        # 调用内部方法，测试可学习的张量级反向传播
        self._test_learnable_backward_per_tensor(
            X, 'cuda', scale_base, zero_point_base)

    # 使用 hypothesis 库的 @given 装饰器定义测试函数，参数 device 是指定的设备（CPU 或 CUDA）
    @given(device=st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']),
           X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       qparams=hu.qparams(dtypes=[torch.quint8])),
           )
    def test_fq_module_per_tensor(self, device, X):
        # 设置随机种子
        np.random.seed(NP_RANDOM_SEED)
        # 从 X 中解包出张量数据以及量化参数（缩放因子、零点、张量类型）
        X, (scale, zero_point, torch_type) = X
        # 获取张量的量化极值范围
        quant_min = torch.iinfo(torch_type).min
        quant_max = torch.iinfo(torch_type).max

        # 将张量 X 移动到指定设备上
        X = to_tensor(X, device)
        # 要求计算梯度
        X.requires_grad_()
        # 创建一个默认的伪量化模块，并将其移动到指定设备上
        fq_module = torch.ao.quantization.default_fake_quant().to(device)
        # 对输入张量 X 应用伪量化模块
        Y_prime = fq_module(X)
        # 断言伪量化模块的缩放因子和零点不为 None
        assert fq_module.scale is not None
        assert fq_module.zero_point is not None
        # 使用参考函数计算伪量化后的张量 Y
        Y = _fake_quantize_per_tensor_affine_reference(X, fq_module.scale, fq_module.zero_point, quant_min, quant_max)
        # 使用 numpy 测试断言，验证 Y 和 Y_prime 在指定的容差范围内相等
        np.testing.assert_allclose(Y.cpu().detach().numpy(), Y_prime.cpu().detach().numpy(), rtol=tolerance, atol=tolerance)

        # 测试反向传播
        # 随机生成一个与 X 形状相同的张量，用作梯度
        dout = torch.rand_like(X, dtype=torch.float, device=device)
        # 对 Y_prime 进行反向传播
        Y_prime.backward(dout)
        # 使用参考函数计算伪量化的梯度 dX
        dX = _fake_quantize_per_tensor_affine_grad_reference(dout, X, fq_module.scale, fq_module.zero_point, quant_min, quant_max)
        # 使用 numpy 测试断言，验证 dX 和 X.grad 在指定的容差范围内相等
        np.testing.assert_allclose(dX.cpu().numpy(), X.grad.cpu().detach().numpy(), rtol=tolerance, atol=tolerance)

    # 使用 hypothesis 库的 @given 装饰器定义测试函数，参数 device 是指定的设备（CPU 或 CUDA）
    @given(device=st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']),
           X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       qparams=hu.qparams(dtypes=torch.quint8)))
    def test_fixed_qparams_fq_module(self, device, X):
        # 从 X 中解包出张量数据以及量化参数（缩放因子、零点、张量类型）
        X, (scale, zero_point, torch_type) = X
        # 将张量 X 移动到指定设备上
        X = to_tensor(X, device)
        # 创建一个固定量化参数范围在 [0, 1] 的伪量化模块
        fq_module = default_fixed_qparams_range_0to1_fake_quant()
        # 将伪量化模块移动到指定设备上
        fq_module.to(device)
        # 复制伪量化模块的缩放因子和零点作为固定的值
        fixed_scale = fq_module.scale.clone()
        fixed_zero_point = fq_module.zero_point.clone()
        # 启用观察者功能，并应用伪量化模块到输入 X 上
        torch.ao.quantization.enable_observer(fq_module)
        fq_module(X)
        # 断言伪量化模块的缩放因子和零点与初始值相等
        self.assertEqual(fixed_scale, fq_module.scale)
        self.assertEqual(fixed_zero_point, fq_module.zero_point)
    # 定义一个测试函数，用于测试针对每个张量的伪量化模块的序列化和反序列化
    def test_fq_serializable_per_tensor(self):
        # 使用默认的观察器
        observer = default_observer
        # 设置量化的最小值和最大值
        quant_min = 0
        quant_max = 127
        # 遍历两种伪量化类：普通伪量化和可学习伪量化
        for FakeQuantizeClass in [FakeQuantize, _LearnableFakeQuantize]:
            # 创建伪量化模块实例
            fq_module = FakeQuantizeClass(observer, quant_min, quant_max)
            # 创建一个张量作为输入数据
            X = torch.tensor([-5, -3.5, -2, 0, 3, 5, 7], dtype=torch.float32)
            # 对输入数据进行伪量化处理
            y_ref = fq_module(X)
            # 获取伪量化模块的状态字典
            state_dict = fq_module.state_dict()
            # 断言状态字典中的scale和zero_point值符合预期
            self.assertEqual(state_dict['scale'], 0.094488)
            self.assertEqual(state_dict['zero_point'], 53)
            # 创建一个字节流对象
            b = io.BytesIO()
            # 将状态字典保存到字节流中
            torch.save(state_dict, b)
            # 针对是否只保存权重进行循环
            for weights_only in [True, False]:
                # 将字节流的读取位置移到开头
                b.seek(0)
                # 从字节流中加载状态字典
                loaded_dict = torch.load(b, weights_only=weights_only)
                # 创建一个新的伪量化模块实例并加载加载的状态字典
                loaded_fq_module = FakeQuantizeClass(observer, quant_min, quant_max)
                loaded_fq_module.load_state_dict(loaded_dict)
                # 遍历原始状态字典中的每个键，断言加载后的状态字典与原始状态字典一致
                for key in state_dict:
                    self.assertEqual(state_dict[key], loaded_fq_module.state_dict()[key])

                # 断言加载后的伪量化模块计算出的量化参数与原始模块一致
                self.assertEqual(loaded_fq_module.calculate_qparams(), fq_module.calculate_qparams())
    # 定义测试方法，用于测试伪量化控制逻辑
    def test_fake_quant_control(self):
        # 对于两种伪量化模块进行迭代测试
        for fq_module in [torch.ao.quantization.default_fake_quant(),
                          _LearnableFakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0,
                                                           quant_max=255,
                                                           dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                                                           reduce_range=True)()]:
            # 设置随机种子以确保可复现性
            torch.manual_seed(42)
            # 创建随机张量 X
            X = torch.rand(20, 10, dtype=torch.float32)
            # 对 X 应用伪量化，得到 Y
            Y = fq_module(X)
            # 断言 Y 与 X 不相等，验证伪量化后的输出不等于输入
            self.assertNotEqual(Y, X)
            # 如果当前模块是 _LearnableFakeQuantize 类型
            if type(fq_module) == _LearnableFakeQuantize:
                # 关闭伪量化
                fq_module.toggle_fake_quant(False)
            else:
                # 禁用伪量化
                torch.ao.quantization.disable_fake_quant(fq_module)
            # 重新生成随机张量 X
            X = torch.rand(20, 10, dtype=torch.float32)
            # 对 X 应用伪量化，得到 Y
            Y = fq_module(X)
            # 断言 Y 与 X 相等，验证伪量化被禁用后输出与输入相同
            self.assertEqual(Y, X)

            # 此处显式复制伪量化模块的内部状态，因为 FakeQuant 保持可变缓冲区中的状态
            scale = fq_module.scale.clone().detach()
            zero_point = fq_module.zero_point.clone().detach()

            # 根据模块类型进行不同的操作
            if type(fq_module) == _LearnableFakeQuantize:
                # 关闭观察者更新，并启用伪量化
                fq_module.toggle_observer_update(False)
                fq_module.toggle_fake_quant(True)
            else:
                # 禁用观察者，并启用伪量化
                torch.ao.quantization.disable_observer(fq_module)
                torch.ao.quantization.enable_fake_quant(fq_module)
            # 创建带有指定范围的随机张量 X
            X = 10.0 * torch.rand(20, 10, dtype=torch.float32) - 5.0
            # 对 X 应用伪量化，得到 Y
            Y = fq_module(X)
            # 断言 Y 与 X 不相等，验证观察者禁用后伪量化有效
            self.assertNotEqual(Y, X)
            # 断言当前的 scale 和 zero_point 与之前保存的相同，验证观察者禁用后状态不变
            self.assertEqual(fq_module.scale, scale)
            self.assertEqual(fq_module.zero_point, zero_point)
            # 如果当前模块是 _LearnableFakeQuantize 类型
            if type(fq_module) == _LearnableFakeQuantize:
                # 启用观察者更新
                fq_module.toggle_observer_update(True)
            else:
                # 启用观察者
                torch.ao.quantization.enable_observer(fq_module)
            # 再次对 X 应用伪量化，得到 Y
            Y = fq_module(X)
            # 断言 Y 与 X 不相等，验证观察者启用后伪量化有效
            self.assertNotEqual(Y, X)
            # 断言当前的 scale 和 zero_point 与之前保存的不同，验证观察者启用后状态变化
            self.assertNotEqual(fq_module.scale, scale)
            self.assertNotEqual(fq_module.zero_point, zero_point)
    def test_fake_quant_preserves_qparam_shapes_for_activations(self):
        # 定义一个继承自 nn.Module 的模型类 Model
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, x):
                # 在模型中执行线性变换
                x = self.linear(x)
                return x

        # 创建一个 Model 的实例 m
        m = Model()

        # 获取默认的量化训练配置 'fbgemm' 并设置为模型 m 的量化配置
        m.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
        # 准备模型 m 进行量化训练，使用 inplace 模式
        torch.ao.quantization.prepare_qat(m, inplace=True)

        # 记录量化前的线性层激活后处理器的尺寸
        scale_shape_before = m.linear.activation_post_process.scale.shape
        zero_point_shape_before = m.linear.activation_post_process.zero_point.shape

        # 创建一个随机张量作为输入
        x = torch.rand(4, 4, 4, 4)
        # 将输入 x 输入到模型 m 中
        m(x)
        # 记录量化后的线性层激活后处理器的尺寸
        scale_shape_after = m.linear.activation_post_process.scale.shape
        zero_point_shape_after = m.linear.activation_post_process.zero_point.shape
        # 断言量化前后激活后处理器的尺寸保持一致
        self.assertEqual(
            scale_shape_before, scale_shape_after,
            msg="FakeQuant scale shape must stay consistent")
        self.assertEqual(
            zero_point_shape_before, zero_point_shape_after,
            msg="FakeQuant zero_point shape must stay consistent")

    def fake_quant_scriptable(self):
        # 使用默认的观察器
        observer = default_observer
        quant_min = 0
        quant_max = 255
        # 对 FakeQuantize 和 _LearnableFakeQuantize 两个类进行迭代
        for FakeQuantizeClass in [FakeQuantize, _LearnableFakeQuantize]:
            # 创建 FakeQuantizeClass 类的实例 fq_module
            fq_module = FakeQuantizeClass(observer, quant_min, quant_max)
            # 对 fq_module 进行脚本化
            scripted_module = torch.jit.script(fq_module)

            # 创建一个测试输入张量 X
            X = torch.tensor([-5, -3.5, -2, 0, 3, 5, 7], dtype=torch.float32)

            # 将 X 输入到 fq_module 和 scripted_module 中
            fq_module(X)
            scripted_module(X)
            # 断言 fq_module 和 scripted_module 计算出的量化参数一致
            self.assertEqual(fq_module.calculate_qparams(), scripted_module.calculate_qparams())

            # 创建一个字节流对象 buf
            buf = io.BytesIO()
            # 将 scripted_module 保存到 buf 中
            torch.jit.save(scripted_module, buf)
            buf.seek(0)
            # 从 buf 中加载模型到 loaded_module
            loaded_module = torch.jit.load(buf)
            # 断言 fq_module 和 loaded_module 计算出的量化参数一致
            self.assertEqual(fq_module.calculate_qparams(), loaded_module.calculate_qparams())

    @given(device=st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']),
           X=hu.per_channel_tensor(shapes=hu.array_shapes(1, 5,),
           qparams=hu.qparams(dtypes=torch.quint8)))
    def test_forward_per_channel(self, device, X):
        r"""Tests the forward path of the FakeQuantizePerTensorAffine op.
        """
        # 设定随机种子
        np.random.seed(NP_RANDOM_SEED)
        X, (scale, zero_point, axis, torch_type) = X
        quant_min = torch.iinfo(torch_type).min
        quant_max = torch.iinfo(torch_type).max

        # 将 X 转换为指定设备上的张量
        X = to_tensor(X, device)
        scale = to_tensor(scale, device)
        # 将 zero_point 转换为 torch.int32 类型，并移到指定设备上
        zero_point = torch.tensor(zero_point).to(dtype=torch.int32, device=device)
        # 使用参考函数计算 per channel 的仿真量化结果 Y
        Y = _fake_quantize_per_channel_affine_reference(X.cpu(), scale.cpu(), zero_point.cpu(), axis, quant_min, quant_max)
        # 使用 torch.fake_quantize_per_channel_affine 计算 per channel 的仿真量化结果 Y_prime
        Y_prime = torch.fake_quantize_per_channel_affine(
            X, scale, zero_point, axis, quant_min, quant_max)
        # 使用 np.testing.assert_allclose 断言 Y 和 Y_prime 在给定的容差范围内一致
        np.testing.assert_allclose(Y, Y_prime.cpu(), rtol=tolerance, atol=tolerance)
    # 定义一个测试方法，用于测试按通道进行缓存掩码操作的实现
    def _test_forward_per_channel_cachemask_impl(self, device):
        # 定义三元组列表，包含不同的 Torch 类型、浮点类型和零点类型组合
        torch_types = (torch.qint8, torch.quint8)
        float_types = (torch.float32, torch.float16, torch.float64)
        zero_point_types = (torch.int, torch.float32, torch.float16)

        # 对三元组进行迭代，生成测试数据 X
        for torch_type, float_type, zero_point_type in itertools.product(torch_types, float_types, zero_point_types):
            X = torch.randn(1, 2, 4, 4, dtype=float_type).to(device)
            # 选择轴向为1，并创建按通道最小最大值观察器
            axis = 1
            obs = torch.ao.quantization.PerChannelMinMaxObserver(axis, torch_type).to(device)
            # 将 X 的缩放因子应用于观察器
            obs(X * 0.75)
            # 计算量化参数的缩放因子和零点值
            scale, zero_point = obs.calculate_qparams()
            # TODO(future PR): 修复 obs.calculate_qparams 中错误的数据类型并移除类型转换
            zero_point = zero_point.to(zero_point_type)
            # 获取观察器中的量化最小值和最大值
            quant_min, quant_max = obs.quant_min, obs.quant_max

            # 使用参考方法进行按通道仿真量化操作 Y
            Y = _fake_quantize_per_channel_affine_reference(
                X.cpu(), scale.cpu(), zero_point.cpu(), axis, quant_min, quant_max)
            # 使用 Torch 内置方法进行按通道仿真量化操作 Y_prime
            Y_prime = torch.fake_quantize_per_channel_affine(
                X, scale, zero_point, axis, quant_min, quant_max)
            # 使用 numpy 测试断言，验证 Y 和 Y_prime 的近似程度
            np.testing.assert_allclose(Y, Y_prime.cpu(), rtol=tolerance, atol=tolerance)
            # 断言 Y 的数据类型与 float_type 相同
            self.assertTrue(Y.dtype == float_type)

    # 测试 CPU 环境下的按通道缓存掩码操作
    def test_forward_per_channel_cachemask_cpu(self):
        self._test_forward_per_channel_cachemask_impl('cpu')

    # 如果 CUDA 测试未禁用，则测试 CUDA 环境下的按通道缓存掩码操作
    @unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
    def test_forward_per_channel_cachemask_cuda(self):
        self._test_forward_per_channel_cachemask_impl('cuda')

    # 测试半精度数字的按通道前向操作
    def test_forward_per_channel_half_precision_numerics(self):
        # 创建长度为5的随机缩放因子和零点数组
        scale = torch.randn(5).abs()
        zero = torch.randn(5).to(dtype=torch.int)
        axis = 1
        mini = 0
        maxi = 255

        # 进行20次迭代，使用半精度浮点数创建 X1，并分别对比仿真量化结果 Y1 和 Y1r
        for i in range(20):
            X1 = torch.randn(4, 5).to(torch.float16)
            Y1 = torch.fake_quantize_per_channel_affine(X1, scale, zero, axis, mini, maxi)
            Y1r = _fake_quantize_per_channel_affine_reference(X1, scale, zero, axis, mini, maxi)
            self.assertEqual(Y1, Y1r, rtol=tolerance, atol=tolerance)

        # 强制溢出情况
        X2 = torch.randn(4, 5).to(torch.float16)
        X2[0, 0] = 2**15 + .01
        Y2 = torch.fake_quantize_per_channel_affine(X2, scale, zero, axis, mini, maxi)
        Y2r = _fake_quantize_per_channel_affine_reference(X2, scale, zero, axis, mini, maxi)
        self.assertEqual(Y2, Y2r, rtol=tolerance, atol=tolerance)

        # 强制下溢情况
        scale = torch.zeros(5) + 10
        X3 = torch.randn(4, 5).to(torch.float16)
        X3[0, 0] = 2**-24
        Y3 = torch.fake_quantize_per_channel_affine(X3, scale, zero, axis, mini, maxi)
        Y3r = _fake_quantize_per_channel_affine_reference(X3, scale, zero, axis, mini, maxi)
        self.assertEqual(Y3, Y3r, rtol=tolerance, atol=tolerance)
    @given(X=hu.per_channel_tensor(shapes=hu.array_shapes(1, 5,),
                                   qparams=hu.qparams(dtypes=torch.quint8)))
    def test_fake_quant_per_channel_qparam_range(self, X):
        # 解包输入张量 X 及其相关的量化参数
        X, (scale, zero_point, axis, torch_type) = X
        # 获取当前量化类型的最小和最大值
        quant_min = torch.iinfo(torch_type).min
        quant_max = torch.iinfo(torch_type).max

        # 根据设备是否支持 CUDA，选择运行在 CPU 还是 CUDA 上
        for device in ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']:
            # 将输入张量 X 和 scale 转换到指定设备上
            X = to_tensor(X, device)
            scale = to_tensor(scale, device)

            # 确保 zero_point 小于 quant_min
            zero_point = torch.full(zero_point.shape, -1 - quant_min).to(dtype=torch.int32, device=device)

            # 对于非浮点型的 zero_point，fakequant 要求其在 quant_min 和 quant_max 之间
            with self.assertRaisesRegex(RuntimeError, "`zero_point` must be between `quant_min` and `quant_max`."):
                Y = torch.fake_quantize_per_channel_affine(X, scale, zero_point, axis, quant_min, quant_max)

            # 对于浮点型的 zero_point，fakequant 允许其超出 quant_min 和 quant_max 的范围
            for zero_point_dtype in [torch.float32, torch.float16]:
                zero_point = zero_point.to(dtype=zero_point_dtype)
                Y = torch.fake_quantize_per_channel_affine(X, scale, zero_point, axis, quant_min, quant_max)
                # 调用参考函数计算期望输出 Y_ref
                Y_ref = _fake_quantize_per_channel_affine_reference(X.cpu(), scale.cpu(), zero_point.cpu(),
                                                                    axis, quant_min, quant_max)
                # 使用 numpy.testing 库验证 Y 与 Y_ref 的近似程度
                np.testing.assert_allclose(Y.cpu().numpy(), Y_ref.cpu().numpy(), rtol=tolerance, atol=tolerance)

    def _test_learnable_forward_per_channel(self, X_base, device, scale_base, zero_point_base, axis):
        r"""Tests the forward path of the learnable FakeQuantizePerTensorAffine op.
        """
        # 针对不同比特数进行测试
        for n_bits in (4, 8):
            quant_min, quant_max = 0, 2 ** (n_bits) - 1

            # 将 scale_base 和 zero_point_base 转换到指定设备上
            scale_base = scale_base.to(device)
            zero_point_base = zero_point_base.to(device)

            # 克隆输入张量和量化参数
            X_curr = X_base.clone()
            scale_curr = scale_base.clone()
            zero_point_curr = zero_point_base.clone()

            # 调用参考函数计算 Y，要求 zero_point 在 quant_min 和 quant_max 之间
            Y = _fake_quantize_per_channel_affine_reference(
                X_curr, scale_curr, zero_point_curr.round().clamp(quant_min, quant_max), axis, quant_min, quant_max).to(device)
            # 对于不同的梯度因子进行测试
            for grad_factor in [0.1, 1.0, 10.0]:
                # 调用学习版本的 fakequant 函数计算 Y_prime
                Y_prime = torch._fake_quantize_learnable_per_channel_affine(
                    X_curr, scale_curr, zero_point_curr, axis, quant_min, quant_max, grad_factor).to(device)
                # 断言 Y 与 Y_prime 的近似程度符合预期
                self.assertTrue(
                    torch.allclose(Y, Y_prime, rtol=tolerance, atol=tolerance),
                    "Expected kernel forward function to have results match the reference forward function")

    @given(X=hu.per_channel_tensor(shapes=hu.array_shapes(1, 5,),
                                   qparams=hu.qparams(dtypes=torch.quint8)))
    # 定义测试方法，用于测试 CPU 上的可学习通道前向传播
    def test_learnable_forward_per_channel_cpu(self, X):
        # 设置随机种子
        torch.random.manual_seed(NP_RANDOM_SEED)
        # 解包输入数据 X，并获取其中的 axis 值
        X, (_, _, axis, _) = X
        # 将 X 转换为 CPU 上的张量
        X_base = torch.tensor(X).to('cpu')
        # 获取指定 axis 上的通道大小
        channel_size = X_base.size(axis)
        # 创建均值为 0，标准差为 1 的正态分布张量作为 scale_base，并限制在范围 [1e-4, 100]
        scale_base = torch.normal(mean=0, std=1, size=(channel_size,)).clamp(1e-4, 100)
        # 创建均值为 0，标准差为 128 的正态分布张量作为 zero_point_base
        zero_point_base = torch.normal(mean=0, std=128, size=(channel_size,))
        # 调用内部方法 _test_learnable_forward_per_channel，测试 CPU 上的可学习通道前向传播
        self._test_learnable_forward_per_channel(
            X_base, 'cpu', scale_base, zero_point_base, axis)

    # 标记测试函数为假设测试，用于 CUDA 上的可学习通道前向传播
    @given(X=hu.per_channel_tensor(shapes=hu.array_shapes(1, 5,),
                                   qparams=hu.qparams(dtypes=torch.quint8)))
    # 如果 CUDA 不可用，则跳过测试
    @unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
    def test_learnable_forward_per_channel_cuda(self, X):
        # 设置随机种子
        torch.random.manual_seed(NP_RANDOM_SEED)
        # 解包输入数据 X，并获取其中的 axis 值
        X, (_, _, axis, _) = X
        # 将 X 转换为 CUDA 上的张量
        X_base = torch.tensor(X).to('cuda')
        # 获取指定 axis 上的通道大小
        channel_size = X_base.size(axis)
        # 创建均值为 0，标准差为 1 的正态分布张量作为 scale_base，并限制在范围 [1e-4, 100]
        scale_base = torch.normal(mean=0, std=1, size=(channel_size,)).clamp(1e-4, 100)
        # 创建均值为 0，标准差为 128 的正态分布张量作为 zero_point_base
        zero_point_base = torch.normal(mean=0, std=128, size=(channel_size,))
        # 调用内部方法 _test_learnable_forward_per_channel，测试 CUDA 上的可学习通道前向传播
        self._test_learnable_forward_per_channel(
            X_base, 'cuda', scale_base, zero_point_base, axis)

    # 标记测试函数为假设测试，用于测试反向传播的通道方法
    @given(device=st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']),
           X=hu.per_channel_tensor(shapes=hu.array_shapes(1, 5,),
           qparams=hu.qparams(dtypes=torch.quint8)))
    # 跳过测试，并附带说明，因为在 CI 环境中存在问题需要修复
    @unittest.skip(
        "this is broken without changes to any relevant code, "
        "we need to remove hypothesis testing in CI")
    def test_backward_per_channel(self, device, X):
        # 测试反向传播方法
        r"""Tests the backward method.
        """
        # 设置随机种子
        np.random.seed(NP_RANDOM_SEED)
        # 解包输入数据 X，并获取其中的 scale, zero_point, axis, torch_type 值
        X, (scale, zero_point, axis, torch_type) = X
        # 获取 torch_type 的数据范围
        quant_min = torch.iinfo(torch_type).min
        quant_max = torch.iinfo(torch_type).max
        # 定义 zero_point 的类型为 torch.int, torch.float, torch.float16
        zero_point_types = (torch.int, torch.float, torch.float16)

        # 遍历 zero_point_types
        for zero_point_type in zero_point_types:
            # 将 X 转换为指定 device 上的张量
            X = to_tensor(X, device)
            # 将 scale 转换为指定 device 上的张量
            scale = to_tensor(scale, device)
            # 将 zero_point 转换为指定 dtype 的张量，并指定为 zero_point_type 类型
            zero_point = to_tensor(zero_point, device).to(dtype=zero_point_type)
            # 设置 X 的梯度信息为可计算
            X.requires_grad_()
            # 使用 torch.fake_quantize_per_channel_affine 方法计算 Y_prime
            Y_prime = torch.fake_quantize_per_channel_affine(
                X, scale, zero_point, axis, quant_min, quant_max)
            # 随机生成与 X 同样大小的浮点数张量 dout，并将其转换为指定 device 上的张量
            dout = torch.rand_like(X, dtype=torch.float).to(device)
            # 使用 _fake_quantize_per_channel_affine_grad_reference 方法计算 dX
            dX = _fake_quantize_per_channel_affine_grad_reference(
                dout, X, scale, zero_point, axis, quant_min, quant_max)
            # 对 Y_prime 进行反向传播计算梯度
            Y_prime.backward(dout)
            # 使用 np.testing.assert_allclose 检查 dX 与 X.grad 是否在容差范围内相等
            np.testing.assert_allclose(dX.cpu().detach().numpy(), X.grad.cpu().detach().numpy(), rtol=tolerance, atol=tolerance)
    # 定义一个测试方法，用于测试在指定设备上的逐通道反向传播
    def _test_backward_per_channel_cachemask_impl(self, device):
        # 定义 torch 数据类型的组合，包括量化整数和浮点数
        torch_types = (torch.qint8, torch.quint8)
        # 定义浮点数数据类型的组合
        float_types = (torch.float32, torch.float16, torch.float64)
        # 定义零点类型的组合，包括整数和浮点数
        zero_point_types = (torch.int, torch.float32, torch.float16)

        # 遍历上述类型组合的笛卡尔积
        for torch_type, float_type, zero_point_type in itertools.product(torch_types, float_types, zero_point_types):
            # 生成一个随机张量 X，设备为指定设备，数据类型为 float_type
            X = torch.randn(1, 2, 4, 4, dtype=float_type).to(device)
            # 选择轴向为 1，创建一个逐通道的最小-最大观察器对象 obs，数据类型为 torch_type，设备为指定设备
            obs = torch.ao.quantization.PerChannelMinMaxObserver(axis, torch_type).to(device)
            # 将 X 的缩放因子和零点调整为使一些值被剪切
            obs(X * 0.75)
            # 计算量化参数的缩放因子和零点
            scale, zero_point = obs.calculate_qparams()
            # TODO（未来的 PR）：修复 obs.calculate_qparams 中错误的数据类型并移除类型转换
            # 将 zero_point 转换为指定的零点类型
            zero_point = zero_point.to(zero_point_type)
            # 获取观察器对象的量化范围的最小值和最大值
            quant_min, quant_max = obs.quant_min, obs.quant_max
            # 设置张量 X 需要计算梯度
            X.requires_grad_()
            # 使用逐通道仿射伪量化函数对张量 X 进行仿射量化
            Y_prime = torch.fake_quantize_per_channel_affine(
                X, scale, zero_point, axis, quant_min, quant_max)
            # 创建一个与 X 形状相同的随机梯度张量 dout，数据类型为 float_type，设备为指定设备
            dout = torch.rand_like(X, dtype=float_type).to(device)
            # 计算逐通道仿射伪量化的梯度参考值
            dX = _fake_quantize_per_channel_affine_grad_reference(
                dout, X, scale, zero_point, axis, quant_min, quant_max)
            # 对 Y_prime 执行反向传播，传入 dout 作为梯度
            Y_prime.backward(dout)
            # 使用 numpy 测试断言，验证计算得到的梯度 dX 与 X.grad 的值相近
            np.testing.assert_allclose(
                dX.cpu().detach().numpy(), X.grad.cpu().detach().numpy(), rtol=tolerance, atol=tolerance)
            # 断言 X.grad 的数据类型与 float_type 相同
            assert X.grad.dtype == float_type

    # 测试在 CPU 上的逐通道缓存掩码反向传播
    def test_backward_per_channel_cachemask_cpu(self):
        self._test_backward_per_channel_cachemask_impl('cpu')

    # 如果没有可用的 GPU，则跳过测试在 CUDA 上的逐通道缓存掩码反向传播
    @unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
    def test_backward_per_channel_cachemask_cuda(self):
        self._test_backward_per_channel_cachemask_impl('cuda')
    # 定义一个测试方法，用于测试可学习的 FakeQuantizePerTensorAffine 操作的反向传播路径
    def _test_learnable_backward_per_channel(self, X_base, device, scale_base, zero_point_base, axis):
        r"""Tests the backward path of the learnable FakeQuantizePerTensorAffine op.
        """
        # 遍历不同的比特位数进行测试
        for n_bits in (4, 8):
            # 计算量化的最小值和最大值
            quant_min, quant_max = 0, 2 ** n_bits - 1

            # 将 scale_base 和 zero_point_base 移动到指定的设备上
            scale_base = scale_base.to(device)
            zero_point_base = zero_point_base.to(device=device)

            # 克隆 X_base 并声明其需要梯度
            X_curr = X_base.clone()
            X_curr.requires_grad_()
            # 克隆 scale_base 并声明其需要梯度
            scale_curr = scale_base.clone()
            scale_curr.requires_grad_()
            # 克隆 zero_point_base 并声明其需要梯度
            zero_point_curr = zero_point_base.clone()
            zero_point_curr.requires_grad_()

            # 遍历不同的梯度因子进行测试
            for grad_factor in [0.1, 1.0, 10.0]:
                # 调用 torch._fake_quantize_learnable_per_channel_affine 方法进行量化
                Y_prime = torch._fake_quantize_learnable_per_channel_affine(
                    X_curr, scale_curr, zero_point_curr, axis, quant_min, quant_max, grad_factor).to(device)

                # 创建随机梯度 dout，类型为 torch.float，并移动到指定设备上
                dout = torch.rand(X_curr.shape, dtype=torch.float).to(device)
                # 调用 _fake_quantize_learnable_per_channel_affine_grad_reference 方法计算梯度
                dX, dScale, dZeroPoint = _fake_quantize_learnable_per_channel_affine_grad_reference(
                    dout, X_curr, scale_curr, zero_point_curr, axis, quant_min, quant_max, device)
                # 对 Y_prime 执行反向传播
                Y_prime.backward(dout)

                # 计算预期的梯度值，并将其移动到指定设备上并且分离出来
                dX_expected = dX.to(device).detach()
                dX_actual = X_curr.to(device).grad.detach()
                dScale_expected = dScale.to(device).detach()
                dScale_actual = scale_curr.to(device).grad.detach()
                dZeroPoint_expected = dZeroPoint.to(device).detach()
                dZeroPoint_actual = zero_point_curr.to(device).grad.detach()
                tolerance = 1e-4

                # 使用断言来验证梯度的正确性，并输出错误信息以便调试
                self.assertTrue(
                    torch.allclose(dX_expected, dX_actual, rtol=tolerance, atol=tolerance),
                    f"Expected dX={dX_expected} to match X.grad={dX_actual}, X={X_curr}, s={scale_curr}, z={zero_point_curr}, dout={dout}, n_bits={n_bits}")  # noqa: B950
                self.assertTrue(
                    torch.allclose(dScale_expected * grad_factor, dScale_actual, rtol=tolerance, atol=tolerance),
                    f"Expected dScale={dScale_expected * grad_factor} to match scale.grad={dScale_actual}, X={X_curr}, s={scale_curr}, z={zero_point_curr}, dout={dout}, n_bits={n_bits}")  # noqa: B950
                self.assertTrue(
                    torch.allclose(dZeroPoint_expected * grad_factor, dZeroPoint_actual, rtol=tolerance, atol=tolerance),
                    f"Expected dZeroPoint={dZeroPoint_expected * grad_factor} to match zero_point.grad={dZeroPoint_actual}, X={X_curr}, s={scale_curr}, z={zero_point_curr}, dout={dout}, n_bits={n_bits}")  # noqa: B950
                # 清零当前的梯度数据，以便进行下一次迭代
                X_curr.grad.data.zero_()
                scale_curr.grad.data.zero_()
                zero_point_curr.grad.data.zero_()
    # 标记测试跳过，说明测试不可用并提供原因
    @unittest.skip(
        "this is broken without changes to any relevant code, "
        "we need to remove hypothesis testing in CI")
    # 定义测试函数，测试反向传播在每个通道上是否可学习（CPU 版本）
    def test_learnable_backward_per_channel_cpu(self, X):
        # 设置随机种子
        torch.random.manual_seed(NP_RANDOM_SEED)
        # 解构 X 变量，并获取相关信息
        X, (_, _, axis, _) = X
        # 将 X 转换为 PyTorch 张量并放置在 CPU 上
        X_base = torch.tensor(X).to('cpu')
        # 获取通道的大小
        channel_size = X_base.size(axis)
        # 创建一个正态分布的张量作为 scale_base，并进行截断处理
        scale_base = torch.normal(mean=0, std=1, size=(channel_size,)).clamp(1e-4, 100)
        # 创建一个正态分布的张量作为 zero_point_base
        zero_point_base = torch.normal(mean=0, std=128, size=(channel_size,))
        # 调用内部方法，测试每个通道上的可学习反向传播
        self._test_learnable_backward_per_channel(
            X_base, 'cpu', scale_base, zero_point_base, axis)
    
    # 使用 hypothesis 库提供的数据生成器定义的测试函数（CUDA 版本）
    @given(X=hu.per_channel_tensor(shapes=hu.array_shapes(2, 5,),
                                   qparams=hu.qparams(dtypes=torch.quint8)))
    # 如果没有 CUDA，跳过测试
    @unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
    # 定义测试函数，测试反向传播在每个通道上是否可学习（CUDA 版本）
    def test_learnable_backward_per_channel_cuda(self, X):
        # 设置随机种子
        torch.random.manual_seed(NP_RANDOM_SEED)
        # 解构 X 变量，并获取相关信息
        X, (scale, zero_point, axis, torch_type) = X
        # 将 X 转换为 PyTorch 张量并放置在 CUDA 上
        X_base = torch.tensor(X).to('cuda')
        # 将 scale 转换为 CUDA 张量
        scale_base = to_tensor(scale, 'cuda')
        # 将 zero_point 转换为 CUDA 张量
        zero_point_base = to_tensor(zero_point, 'cuda')
        # 调用内部方法，测试每个通道上的可学习反向传播
        self._test_learnable_backward_per_channel(
            X_base, 'cuda', scale_base, zero_point_base, axis)
    
    # 测试数值一致性，针对每个张量进行测试
    def test_numerical_consistency_per_tensor(self):
        self._test_numerical_consistency('per_tensor')
    
    # 测试数值一致性，针对每个通道进行测试
    def test_numerical_consistency_per_channel(self):
        self._test_numerical_consistency('per_channel')
    # 定义一个测试函数，用于比较量化/反量化操作与伪量化操作在不同设备和数据类型上的数值一致性
    def _test_numerical_consistency(self, test_type):
        r"""Comparing numerical consistency between quantize/dequantize op and the fake quantize op across devices and dtypes
        """
        # 设置随机种子
        torch.random.manual_seed(NP_RANDOM_SEED)
        # 定义量化类型和浮点类型
        torch_types = [torch.qint8, torch.quint8]
        float_types = [torch.float, torch.float16, torch.float64]
        # 根据测试类型选择零点类型
        if test_type == "per_channel":
            zero_types = [torch.int, torch.float, torch.float16]
        else:
            zero_types = [torch.int]
        # 根据CUDA是否可用选择设备
        devices = [torch.device('cpu'), torch.device('cuda')] if torch.cuda.is_available() else [torch.device('cpu')]
        # 设置通道轴
        axis = 1
        # 循环20次
        for i in range(20):
            # 对每一种组合进行迭代测试
            for torch_type, float_type, device, zero_type in itertools.product(torch_types, float_types, devices, zero_types):
                # 生成随机输入张量X，并移到指定设备和浮点类型
                X = torch.randn(3, 3, device=device).to(float_type)
                # 生成随机的缩放因子，并计算平均值作为scale，转换为float类型
                scales = (10 * torch.randn(3, device=device)).abs()
                scale = scales.mean().to(float).item()
                # 生成随机的零点，并找出最大值后转换为标量
                zeros = (10 * torch.randn(3, device=device)).abs().to(dtype=zero_type)
                zero = zeros.max().view(1).item()
                # 获取量化的最小和最大值
                quant_min = torch.iinfo(torch_type).min
                quant_max = torch.iinfo(torch_type).max

                # 初始化测试是否执行的标志
                test_was_run = False
                # 如果测试类型为"per_tensor"
                if test_type == "per_tensor":
                    test_was_run = True
                    # 执行量化、反量化并转换为指定设备和浮点类型的操作，并比较结果
                    Y = torch.dequantize(torch.quantize_per_tensor(X.to('cpu').to(torch.float),
                                                                   scale, zero, torch_type)).to(device).to(float_type)
                    Y_prime = torch.fake_quantize_per_tensor_affine(X, scale, zero, quant_min, quant_max)
                    self.assertEqual(
                        Y, Y_prime, "Difference found between dequant+quant_per_tensor and fake_quantize_per_tensor")

                # 如果测试类型为"per_channel"
                if test_type == "per_channel":
                    test_was_run = True
                    # 执行通道量化、反量化并转换为指定设备和浮点类型的操作，并比较结果
                    Y = torch.dequantize(torch.quantize_per_channel(X.to('cpu').to(torch.float), scales.to(
                        'cpu'), zeros.to('cpu'), axis, torch_type)).to(device).to(float_type)
                    Y_prime = torch.fake_quantize_per_channel_affine(X, scales, zeros, axis, quant_min, quant_max)
                    self.assertEqual(
                        Y, Y_prime, "Difference found between dequant+quant_per_channel and fake_quantize_per_channel")
                
                # 断言该次测试一定执行过
                self.assertTrue(test_was_run)

    # 标记该测试函数不适用于TorchDynamo，跳过执行
    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_fake_quantize_per_channel_affine_scale_dtypes(self):
        """
        Ensure the error message is more helpful
        """
        # 定义待测试的数据类型列表，包括 torch.float, torch.float64, torch.bfloat16, torch.half
        dtype_list = [torch.float, torch.float64, torch.bfloat16, torch.half]
        
        # 对每种数据类型进行测试
        for scale_dtype in dtype_list:
            # 创建一个形状为 (3, 4, 5, 6) 的随机张量作为输入
            input = torch.randn(3, 4, 5, 6)
            
            # 创建一个尺度张量 scale，其数据类型为 scale_dtype，并包含值 [0.1, 0.2, 0.3, 0.4]
            scale = torch.Tensor([0.1, 0.2, 0.3, 0.4]).to(scale_dtype)
            
            # 创建一个零点张量 zero_point，数据类型为 torch.int32，包含值 [1, 2, 3, 4]
            zero_point = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
            
            # 设置轴 axis 的值为 1
            axis = 1
            
            # 设置量化的最小值和最大值
            quant_min = 0
            quant_max = 255
            
            # 如果 scale 的数据类型不是 torch.float
            if scale_dtype != torch.float:
                # 预期抛出 RuntimeError 异常
                with self.assertRaises(RuntimeError):
                    torch.fake_quantize_per_channel_affine(
                        input, scale, zero_point, axis, quant_min, quant_max
                    )
            else:
                # 否则调用 fake_quantize_per_channel_affine 函数进行量化操作
                torch.fake_quantize_per_channel_affine(
                    input, scale, zero_point, axis, quant_min, quant_max
                )
# 定义测试类 TestFusedObsFakeQuant，继承自 TestCase，用于测试融合操作假量化的反向传播
class TestFusedObsFakeQuant(TestCase):
    
    # 使用 hypothesis 库的 given 装饰器，设定测试参数：设备为 CPU 或 CUDA（若可用），是否对称量化
    @given(device=st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']),
           symmetric_quant=st.booleans())
    # 设定测试设置：没有执行期限
    @settings(deadline=None)
    # 同上一个 given 装饰器，设定测试参数
    @given(device=st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']),
           symmetric_quant=st.booleans())
    # 同上一个 given 装饰器，设定测试设置
    @settings(deadline=None)
    # 使用 hypothesis 库的 given 装饰器，设定测试参数：设备为 CPU 或 CUDA（若可用）
    @given(device=st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']),)
    # 设定测试设置：没有执行期限
    @settings(deadline=None)
    # 定义测试方法 test_fused_obs_fake_quant_backward_op，接收设备参数 device
    def test_fused_obs_fake_quant_backward_op(self, device) -> None:
        # 定义三个变量 n, m, k，均初始化为 10
        n = m = k = 10
        # 定义输入形状 input_shape 为元组 (m, n)
        input_shape = (m, n)
        # 定义输出形状 output_shape 为元组 (m, n)
        output_shape = (m, n)
        
        # 生成一个在指定设备上的随机张量 x，并设置 requires_grad=True，以便计算梯度
        x = torch.randn(input_shape, device=device, requires_grad=True)
        
        # 定义平均常量 avg_const 为 0.01
        avg_const = 0.01
        # 创建一个在指定设备上的标量张量 scale，值为 1.0
        scale = torch.tensor([1.0], device=device)
        # 创建一个在指定设备上的整数张量 zero_point，值为 0
        zero_point = torch.tensor([0], dtype=torch.int, device=device)
        
        # 调用 _get_tensor_min_max 函数获取张量 x 的最小值和最大值
        x_min, x_max = _get_tensor_min_max(x)
        # 调用 _get_scale_zp 函数获取量化的缩放因子和零点值，这里指定量化类型为 torch.quint8
        x_scale, x_zero_point = _get_scale_zp(
            x_min, x_max, torch.quint8
        )
        
        # 创建在指定设备上的张量 x_scale，其值为计算得到的量化缩放因子
        x_scale = torch.tensor(x_scale, device=device)
        # 创建在指定设备上的整数张量 x_zero_point，其值为计算得到的量化零点值
        x_zero_point = torch.tensor(x_zero_point, dtype=torch.int, device=device)
        
        # 使用 torch.fake_quantize_per_tensor_affine 对张量 x 进行仿真量化
        x_fake_quant = torch.fake_quantize_per_tensor_affine(
            x, x_scale, x_zero_point, 0, 255
        )
        
        # 获取 torch.fused_moving_avg_obs_fake_quant 操作的引用
        pt_op = torch.fused_moving_avg_obs_fake_quant
        # 调用 pt_op 运算，传入多个参数，计算结果存入 out
        out = pt_op(
            x,
            torch.tensor(1, device=device),
            torch.tensor(1, device=device),
            torch.tensor(x_min, device=device),
            torch.tensor(x_max, device=device),
            scale,
            zero_point,
            avg_const,
            0,
            255,
            0,
            False,
        )
        
        # 验证 out 和 x_fake_quant 是否接近，用于确认输出是否匹配预期
        torch.testing.assert_close(out, x_fake_quant)
        
        # 验证梯度是否符合 fake_quant 操作的预期
        # 创建一个与 x 形状相同的随机张量 dout，并设置数据类型为 float，放在指定设备上
        dout = torch.rand_like(x, dtype=torch.float).to(device)
        # 对 out 进行反向传播
        out.backward(dout)
        
        # 使用 _fake_quantize_per_tensor_affine_grad_reference 函数计算仿真量化梯度 dX
        dX = _fake_quantize_per_tensor_affine_grad_reference(
            dout, x, x_scale, x_zero_point, 0, 255)
        
        # 断言 dX 与 x.grad 是否相等
        self.assertEqual(dX, x.grad)
        # 断言 x.grad 的数据类型是否为 torch.float32
        self.assertTrue(x.grad.dtype == torch.float32)

    # 使用 hypothesis 库的 given 装饰器，设定测试参数：设备为 CPU 或 CUDA（若可用）
    @given(device=st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']),)
    # 设定测试设置：没有执行期限
    @settings(deadline=None)
    # 定义一个测试函数，用于验证不带量化和融合移动平均的反向传播操作
    def test_fused_backward_op_fake_quant_off(self, device) -> None:
        # 定义输入和输出的维度
        n = m = 4
        input_shape = (m, n)
        output_shape = (m, n)

        # 创建一个张量 x，其形状为 input_shape，需要计算梯度
        x = torch.randn(input_shape, device=device, requires_grad=True)

        # 定义常量和参数，用于量化和反量化操作
        avg_const = 0.01
        scale = torch.tensor([1.0], device=device)
        zero_point = torch.tensor([0], dtype=torch.int, device=device)

        # 获取张量 x 的最小值和最大值
        x_min, x_max = _get_tensor_min_max(x)
        # 根据张量 x 的最小值和最大值，以及量化类型（这里是 torch.quint8），获取量化比例和零点
        x_scale, x_zero_point = _get_scale_zp(
            x_min, x_max, torch.quint8
        )

        # 设置 PyTorch 操作为融合移动平均观察值假量化操作
        pt_op = torch.fused_moving_avg_obs_fake_quant
        # 执行融合操作，传入相关参数，得到输出张量 out
        out = pt_op(
            x,
            torch.tensor(0, device=device),
            torch.tensor(0, device=device),
            torch.tensor(x_min, device=device),
            torch.tensor(x_max, device=device),
            scale,
            zero_point,
            avg_const,
            0,
            255,
            0,
            False,
        )

        # 验证输出与输入张量 x 相匹配
        torch.testing.assert_close(out, x)

        # 验证梯度是否符合假量化操作的预期
        dout = torch.rand_like(x, dtype=torch.float).to(device)
        out.backward(dout)

        # 计算假量化操作的梯度参考值
        dX = _fake_quantize_per_tensor_affine_grad_reference(
            dout, x, x_scale, x_zero_point, 0, 255)
        
        # 断言计算得到的梯度 dX 与 x 的梯度相等
        self.assertEqual(dX, x.grad)
        # 确保 x 的梯度数据类型为 torch.float32
        self.assertTrue(x.grad.dtype == torch.float32)
if __name__ == '__main__':
    # 检查当前模块是否作为主程序直接运行
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_quantization.py TESTNAME\n\n"
                       "instead.")
    # 如果是，则引发运行时错误，提示不应直接运行该测试文件
```