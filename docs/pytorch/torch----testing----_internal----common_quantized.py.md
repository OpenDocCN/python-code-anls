# `.\pytorch\torch\testing\_internal\common_quantized.py`

```
# 忽略类型检查错误（这行注释可能是为了在类型检查工具中跳过错误的检查）
mypy: ignore-errors

r"""导入此文件包括用于检查量化张量和模块的常用实用方法。
"""

# 导入所需的库
import numpy as np
import torch
from contextlib import contextmanager
from torch.testing._internal.common_utils import TEST_WITH_ASAN, TEST_WITH_TSAN, TEST_WITH_UBSAN, IS_PPC, IS_MACOS, IS_WINDOWS

# 获取支持的量化引擎列表
supported_qengines = torch.backends.quantized.supported_engines
supported_qengines.remove('none')

# 注意：我们目前不在 WINDOWS 和 MACOS 上运行 QNNPACK 测试，因为它存在问题。问题编号 #29326
# QNNPACK 不支持 PPC
# QNNPACK 抛出 ASAN 堆缓冲区溢出错误。
if 'qnnpack' in supported_qengines and any([IS_PPC, TEST_WITH_ASAN, TEST_WITH_TSAN, TEST_WITH_UBSAN, IS_MACOS, IS_WINDOWS]):
    supported_qengines.remove('qnnpack')

def _conv_output_shape(input_size, kernel_size, padding, stride, dilation,
                       output_padding=0):
    """根据卷积参数计算输出形状。"""
    return np.floor((input_size + 2 * padding - kernel_size - (kernel_size - 1)
                     * (dilation - 1)) / stride) + 2 * output_padding + 1

# 量化引用
def _quantize(x, scale, zero_point, qmin=None, qmax=None, dtype=np.uint8):
    """对 numpy 数组进行量化。"""
    if qmin is None:
        qmin = np.iinfo(dtype).min
    if qmax is None:
        qmax = np.iinfo(dtype).max
    qx = np.round(x / scale + zero_point).astype(np.int64)
    qx = np.clip(qx, qmin, qmax)
    qx = qx.astype(dtype)
    return qx

def _dequantize(qx, scale, zero_point):
    """对 numpy 数组进行反量化。"""
    x = (qx.astype(float) - zero_point) * scale
    return x

def _requantize(x, multiplier, zero_point, qmin=0, qmax=255, qtype=np.uint8):
    """重新量化 numpy 数组，即将中间的 int32 或 int16 值转换回给定类型。"""
    qx = (x * multiplier).round() + zero_point
    qx = np.clip(qx, qmin, qmax).astype(qtype)
    return qx

def _calculate_dynamic_qparams(X, dtype, reduce_range=False, qscheme=torch.per_tensor_affine):
    """根据张量的最小值和最大值计算动态量化参数（scale, zero_point）。"""
    assert qscheme in (torch.per_tensor_affine, torch.per_tensor_symmetric)
    if qscheme == torch.per_tensor_symmetric:
        assert dtype == torch.qint8
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if dtype == torch.qint8:
        if reduce_range:
            qmin, qmax = -64, 63
        else:
            qmin, qmax = -128, 127
    else:  # dtype == torch.quint8
        if reduce_range:
            qmin, qmax = 0, 127
        else:
            qmin, qmax = 0, 255
    min_val = X.min()
    max_val = X.max()
    is_symmetric = (qscheme == torch.per_tensor_symmetric)
    if min_val == max_val:
        scale = 1.0
        zero_point = 0
    else:
        # 如果不是对称量化，根据最大值和最小值重新计算缩放因子和零点
        if is_symmetric:
            # 更新最大值和最小值
            max_val = max(max_val, -min_val)
            min_val = -max_val
            # 计算缩放因子
            scale = (max_val - min_val) / (qmax - qmin)
            # 确保缩放因子不小于 float32 的最小非零值
            scale = max(scale, np.finfo(np.float32).eps)
            # 零点设为0（对称量化时）
            zero_point = 0
        else:
            # 更新最大值和最小值
            max_val = max(max_val, 0.0)
            min_val = min(min_val, 0.0)
            # 计算缩放因子
            scale = (max_val - min_val) / (qmax - qmin)
            # 确保缩放因子不小于 float32 的最小非零值
            scale = max(scale, np.finfo(np.float32).eps)
            # 根据最小值计算零点，并确保在 [qmin, qmax] 范围内
            zero_point = qmin - round(min_val / scale)
            zero_point = max(qmin, zero_point)
            zero_point = min(qmax, zero_point)
    # 返回缩放因子和零点作为浮点数和整数
    return [float(scale), int(zero_point)]
# 计算动态量化参数（scale, zero_point）的函数
def _calculate_dynamic_per_channel_qparams(X, dtype):
    """Calculate the dynamic quantization parameters (scale, zero_point)
    according to the min and max element of the tensor"""
    # 如果输入是 torch.Tensor，则转换为 numpy 数组
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    # 获取指定数据类型的量化范围
    qmin, qmax = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    # 计算量化级别数
    n_levels = qmax - qmin
    # 初始化 scale 和 zero_point 数组
    scale = np.zeros(X.shape[0], dtype=np.float64)
    zero_point = np.zeros(X.shape[0], dtype=np.int64)
    # 遍历每个通道
    for i in range(zero_point.shape[0]):
        # 计算当前通道的最小值和最大值
        min_val = X.min()
        max_val = X.max()
        # 如果最小值等于最大值，表示所有值相等
        if min_val == max_val:
            scale[i] = 1.0
            zero_point[i] = 0
        else:
            # 确保 max_val 和 min_val 大于等于 0
            max_val = max(max_val, 0.0)
            min_val = min(min_val, 0.0)
            # 计算当前通道的 scale
            scale[i] = (max_val - min_val) / n_levels
            # 确保 scale 大于等于 np.finfo(np.float32).eps
            scale[i] = max(scale[i], np.finfo(np.float32).eps)
            # 计算当前通道的 zero_point
            zero_point[i] = qmin - round(min_val / scale[i])
            zero_point[i] = max(qmin, zero_point[i])
            zero_point[i] = min(qmax, zero_point[i])

    return scale, zero_point

# 计算信噪比（SNR）的函数
def _snr(x, x_hat):
    """Calculates the signal to noise ratio and returns the signal and noise
    power, as well as the SNR in dB.
    If the input is a list/tuple this function is called recursively on each
    element. The result will have the same nested structure as the inputs.

    Args:
        x, x_hat: Either a tensor or a nested list/tuple of tensors.
    Returns:
        signal, noise, SNR(in dB): Either floats or a nested list of floats
    """
    # 如果输入是列表或元组，则递归调用 _snr 函数
    if isinstance(x, (list, tuple)):
        assert len(x) == len(x_hat)
        res = []
        for idx in range(len(x)):
            res.append(_snr(x[idx], x_hat[idx]))
        return res
    # 如果 x_hat 是量化的，则反量化
    if x_hat.is_quantized:
        x_hat = x_hat.dequantize()
    # 如果 x 是量化的，则反量化
    if x.is_quantized:
        x = x.dequantize()
    # 计算噪声
    noise = (x - x_hat).norm()
    if noise == 0:
        return 0.0, float('inf'), float('inf')  # 如果噪声为零，则 SNR 为无穷大
    # 计算信号
    signal = x.norm()
    # 计算 SNR 和 SNR（dB）
    snr = signal / noise
    snr_db = 20 * snr.log10()
    return signal, noise, snr_db

# 上下文管理器：用于覆盖量化引擎设置
@contextmanager
def override_quantized_engine(qengine):
    previous = torch.backends.quantized.engine
    torch.backends.quantized.engine = qengine
    try:
        yield
    finally:
        torch.backends.quantized.engine = previous

# 上下文管理器：用于在 qnnpack 上覆盖 CPU 分配器
@contextmanager
def override_cpu_allocator_for_qnnpack(qengine_is_qnnpack):
    try:
        if qengine_is_qnnpack:
            torch._C._set_default_mobile_cpu_allocator()
        yield
    finally:
        if qengine_is_qnnpack:
            torch._C._unset_default_mobile_cpu_allocator()

# 装饰器函数：用于覆盖所有量化引擎设置
# TODO: 更新所有量化测试以使用此装饰器。
# 目前对于某些测试，似乎在 fbgemm vs qnnpack 上参数不一致。
def override_qengines(qfunction):
    def test_fn(*args, **kwargs):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                # qfunction 应不返回任何内容。
                qfunction(*args, **kwargs)
    return test_fn

# 函数：检查当前量化引擎是否为 fbgemm
def qengine_is_fbgemm():
    # 检查当前 Torch 的量化后端引擎是否为 'fbgemm'
    return torch.backends.quantized.engine == 'fbgemm'
# 检查当前的量化后端引擎是否为 QNNPACK
def qengine_is_qnnpack():
    return torch.backends.quantized.engine == 'qnnpack'

# 检查当前的量化后端引擎是否为 OneDNN
def qengine_is_onednn():
    return torch.backends.quantized.engine == 'onednn'

# 检查当前的量化后端引擎是否为 x86
def qengine_is_x86():
    return torch.backends.quantized.engine == 'x86'

# 辅助函数：用于将指定轴置换到第一个维度
def _permute_to_axis_zero(X, axis):
    new_axis_list = list(range(X.dim()))
    new_axis_list[axis] = 0
    new_axis_list[0] = axis
    y = X.permute(tuple(new_axis_list))
    return y, new_axis_list

# 参考方法：对每个通道仿真的伪量化操作
# 注意：由于在实际内核中，scale/zero_point 保持为浮点数，这里模拟了对 float16/64 的伪量化操作方式
def _fake_quantize_per_channel_affine_reference(X, per_channel_scale, per_channel_zero_point, axis, quant_min, quant_max):
    dtype = X.dtype
    X, permute_axis_list = _permute_to_axis_zero(X.to(torch.float32), axis)
    res = torch.zeros_like(X)

    for i in range(X.size()[0]):
        res[i] = (torch.clamp(torch.round(X[i] * (1.0 / per_channel_scale[i]) +
                  per_channel_zero_point[i]), quant_min, quant_max) - per_channel_zero_point[i]) * per_channel_scale[i]

    out = res.permute(tuple(permute_axis_list))
    return out.to(dtype)

# 参考方法：伪量化操作的梯度
# 注意：由于在实际内核中，scale/zero_point 保持为浮点数，这里模拟了对 float16/64 的伪量化操作方式
def _fake_quantize_per_channel_affine_grad_reference(dY, X, per_channel_scale, per_channel_zero_point, axis, quant_min, quant_max):
    dtype = X.dtype
    X, permute_axis_list = _permute_to_axis_zero(X.to(torch.float32), axis)
    Xq = torch.zeros_like(X)
    for i in range(X.size()[0]):
        Xq[i] = torch.round(X[i] * (1.0 / per_channel_scale[i]) + per_channel_zero_point[i])
    Xq = Xq.permute(tuple(permute_axis_list))
    mask = (Xq >= quant_min) * (Xq <= quant_max)
    res = torch.zeros_like(dY)
    res[mask] = dY[mask]
    return res.to(dtype)

# 将输入数据转换为张量，并移动到指定的设备上
def to_tensor(X, device):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X)
    else:
        X = X.clone().detach()
    return X.to(device=torch.device(device), dtype=torch.float32)
```