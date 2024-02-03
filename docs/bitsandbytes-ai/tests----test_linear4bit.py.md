# `bitsandbytes\tests\test_linear4bit.py`

```py
# 导入必要的库
import os
from tempfile import TemporaryDirectory
import pytest
import torch

# 导入自定义模块
import bitsandbytes as bnb
from tests.helpers import TRUE_FALSE

# 定义存储不同数据类型的字典
storage = {
    'uint8': torch.uint8,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float32': torch.float32
}

# 参数化测试用例
@pytest.mark.parametrize("quant_storage", ['uint8', 'float16', 'bfloat16', 'float32'])
@pytest.mark.parametrize("bias", TRUE_FALSE)
@pytest.mark.parametrize("compress_statistics", TRUE_FALSE)
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
def test_linear_serialization(quant_type, compress_statistics, bias, quant_storage):
    # 定义初始数据类型、计算数据类型、设备和层形状
    original_dtype = torch.float16
    compute_dtype = None
    device = "cuda"
    layer_shape = (300, 400)

    # 创建原始线性层
    linear = torch.nn.Linear(*layer_shape, dtype=original_dtype, device="cpu")

    # 对原始层进行量化
    linear_q = bnb.nn.Linear4bit(
        linear.in_features,
        linear.out_features,
        bias=bias,
        compute_dtype=compute_dtype,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
        device="meta",
    )
    new_weight = bnb.nn.Params4bit(data=linear.weight, quant_type=quant_type, requires_grad=False)
    linear_q.weight = new_weight
    if bias:
        linear_q.bias = torch.nn.Parameter(linear.bias)
    linear_q = linear_q.to(device)

    # 保存状态字典
    sd = linear_q.state_dict()

    # 从状态字典中恢复
    bias_data2 = sd.pop("bias", None)
    weight_data2 = sd.pop("weight")
    weight2 = bnb.nn.Params4bit.from_prequantized(quantized_stats=sd, data=weight_data2)

    # 创建具有相同参数的新层
    linear_q2 = bnb.nn.Linear4bit(
        linear.in_features,
        linear.out_features,
        bias=bias,
        compute_dtype=compute_dtype,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
        device="meta",
    )
    # 从状态字典中加载权重
    linear_q2.weight = weight2
    # 如果存在偏置项，将偏置项数据赋值给linear_q2的偏置项参数
    if bias:
        linear_q2.bias = torch.nn.Parameter(bias_data2)
    # 将linear_q2移动到指定设备上
    linear_q2 = linear_q2.to(device)

    # 匹配两个线性层的权重
    a, b = linear_q.weight, linear_q2.weight

    # 使用指定的量化存储类型对原始层进行量化
    linear_qs = bnb.nn.Linear4bit(
        linear.in_features,
        linear.out_features,
        bias=bias,
        compute_dtype=compute_dtype,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
        quant_storage=storage[quant_storage],
        device="meta",
    )
    # 将原始层的权重赋值给量化层的权重参数
    linear_qs.weight = bnb.nn.Params4bit(data=linear.weight, requires_grad=False, quant_type=quant_type, quant_storage=storage[quant_storage])
    # 如果存在偏置项，将原始层的偏置项赋值给量化层的偏置项参数
    if bias:
        linear_qs.bias = torch.nn.Parameter(linear.bias)
    # 将量化层移动到指定设备上
    linear_qs = linear_qs.to(device)

    # 断言两个权重的设备和数据类型相同，并且值相等
    assert a.device == b.device
    assert a.dtype == b.dtype
    assert torch.equal(a, b)

    # 获取量化状态并进行比较
    q0 = a.quant_state
    q1 = b.quant_state
    for attr in ('code', 'dtype', 'blocksize', 'absmax'):
        c, d = getattr(q0, attr), getattr(q1, attr)
        if isinstance(c, torch.Tensor):
            assert torch.equal(c, d)
        else:
            assert c == d, f"{c} != {d}"

    # 如果存在第二个量化状态，进行比较
    if q0.state2 is not None:
        for attr in ('code', 'dtype', 'blocksize', 'absmax'):
            c, d = getattr(q0.state2, attr), getattr(q1.state2, attr)
            if isinstance(c, torch.Tensor):
                assert torch.equal(c, d)
            else:
                assert c == d, f"{c} != {d}"

    # 如果存在偏置项，比较两个偏置项的设备、数据类型和值
    if bias:
        a, b = linear_q.bias, linear_q2.bias
        assert a.device == b.device
        assert a.dtype == b.dtype
        assert torch.equal(a, b)

    # 前向传播测试
    x = torch.rand(42, layer_shape[0], device=device)
    a = linear_q(x)
    b = linear_q2(x)
    c = linear_qs(x)
    assert a.device == b.device
    assert a.dtype == b.dtype
    assert a.device == c.device
    assert a.dtype == c.dtype
    assert torch.equal(a, b)
    assert torch.equal(a, c)
    # 测试将模型移动到 CPU 再移回 GPU
    linear_q2.to('cpu')
    linear_q2.to(device)
    # 使用移回 GPU 后的模型进行推理
    d = linear_qs(x)
    # 断言结果张量的数据类型与设备与原始张量一致
    assert c.dtype == d.dtype
    assert c.device == d.device
    assert torch.equal(c, d)

    # 保存大小比率测试。目标设置为 layer_shape == (300, 400) 且带有偏置
    with TemporaryDirectory() as tmpdir:
        # 创建临时目录
        state_path_4bit = os.path.join(tmpdir, "state_4bit.pth")
        state_path = os.path.join(tmpdir, "state.pth")
        # 保存原始模型和量化后的模型的状态字典
        torch.save(linear.state_dict(), state_path)
        torch.save(linear_q.state_dict(), state_path_4bit)

        # 获取原始模型和量化后模型的文件大小
        size_orig, size_4 = os.path.getsize(state_path), os.path.getsize(
            state_path_4bit
        )
        # 计算量化后模型大小与原始模型大小的比率
        size_ratio = size_4 / size_orig
        # 根据原始数据类型选择目标压缩比率
        target_compression = 0.143 if original_dtype == torch.float32 else 0.29  # 这些数字随着权重形状的增加而降低
        # 比率错误消息
        ratio_error_msg = f"quantized_size {size_4:,} is larger on disk than {target_compression:.2%} of original size {size_orig:,}"
        # 断言量化后模型大小比率小于目标压缩比率
        assert size_ratio < target_compression, ratio_error_msg
```