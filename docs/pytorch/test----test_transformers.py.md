# `.\pytorch\test\test_transformers.py`

```
# Owner(s): ["module: nn"]

# 引入上下文管理器
import contextlib
# 导入偏函数
from functools import partial
# 导入命名元组
from collections import namedtuple
# 导入 sys 模块
import sys
# 导入 PyTorch 主模块
import torch
# 导入 PyTorch 神经网络模块
import torch.nn as nn
# 导入 PyTorch 的函数式接口
import torch.nn.functional as F
# 从 torch.nn.functional 中导入 scaled_dot_product_attention 函数
from torch.nn.functional import scaled_dot_product_attention
# 导入 SDPA 内核和后端相关模块
from torch.nn.attention import sdpa_kernel, SDPBackend
# 导入因果变体相关模块
from torch.nn.attention.bias import CausalVariant, causal_lower_right, causal_upper_left
# 导入参数模块
from torch.nn.parameter import Parameter
# 导入单元测试模块
import unittest
# 从 unittest.mock 中导入 patch、MagicMock 和 ANY
from unittest.mock import patch, MagicMock, ANY
# 导入数学模块
import math
# 导入 itertools 模块
import itertools
# 导入 PyTorch 的优化模块
import torch.optim as optim
# 导入测试设备类型相关模块
from torch.testing._internal.common_device_type import instantiate_device_type_tests, onlyCUDA, onlyCPU
# 导入类型提示相关模块
from typing import List, Tuple, Optional
# 导入 PyTorch 的 C++ 扩展模块
import torch.utils.cpp_extension
# 从 torch.testing._internal.common_nn 中导入 NNTestCase
from torch.testing._internal.common_nn import NNTestCase
# 从 torch.testing._internal.common_utils 中导入各种测试相关函数和变量
from torch.testing._internal.common_utils import (
    TEST_WITH_ROCM,
    skipIfRocm,
    skipIfTorchDynamo,
    TEST_FAIRSEQ,
    run_tests,
    parametrize,
    freeze_rng_state,
    TEST_WITH_CROSSREF,
    slowTest,
    set_default_dtype,
    gradcheck,
    make_tensor,
    NOTEST_CPU,
    IS_WINDOWS,
    TEST_WITH_TORCHDYNAMO,
)
# 从 torch._dynamo.testing 中导入 CompileCounterWithBackend
from torch._dynamo.testing import CompileCounterWithBackend

# 从 torch.testing._internal.common_methods_invocations 中导入 wrapper_set_seed
from torch.testing._internal.common_methods_invocations import wrapper_set_seed
# 从 torch.testing._internal.common_cuda 中导入相关 CUDA 设备支持模块
from torch.testing._internal.common_cuda import (
    IS_JETSON, SM80OrLater, PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
    PLATFORM_SUPPORTS_FUSED_ATTENTION,
    PLATFORM_SUPPORTS_CUDNN_ATTENTION
)

# 从 test_cpp_extensions_open_device_registration 中导入相关函数
from test_cpp_extensions_open_device_registration import (
    remove_build_path,
    generate_faked_module
)

# 如果 TEST_FAIRSEQ 为真，则导入 fairseq.models.transformer 模块
if TEST_FAIRSEQ:
    import fairseq.models.transformer as fairseq_transformer

# 定义 SdpaShape 命名元组
SdpaShape = namedtuple('Sdpa_Shape', ['batch', 'num_heads', 'seq_len', 'head_dim'])
# 定义 Tolerances 命名元组
Tolerances = namedtuple('Tolerances', ['atol', 'rtol'])

# 定义上下文管理器，用于临时启用或禁用确定性算法
@contextlib.contextmanager
def use_deterministic_algorithims(mode: bool, warn_only: bool):
    r"""
    This context manager can be used to temporarily enable or disable deterministic algorithms.
    Upon exiting the context manager, the previous state of the flag will be restored.
    """
    # 保存当前确定性算法状态
    previous_mode: bool = torch.are_deterministic_algorithms_enabled()
    previous_warn_only: bool = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        # 设置新的确定性算法状态
        torch.use_deterministic_algorithms(mode, warn_only=warn_only)
        # 返回空字典
        yield {}
    finally:
        # 恢复之前的确定性算法状态
        torch.use_deterministic_algorithms(previous_mode, warn_only=previous_warn_only)


# 默认的绝对容差和相对容差，根据不同的浮点数类型设置
default_atol = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float32: 1e-5}
default_rtol = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float32: 1.3e-6}

# 检查当前 CUDA 设备是否为 SM8X 架构或更新版本
isSM8XDevice = torch.cuda.is_available() and torch.cuda.get_device_capability() in [(8, 6), (8, 7), (8, 9)]
# 检查当前 CUDA 设备是否为 SM90 架构
isSM90Device = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
# 检查当前 CUDA 设备是否为 SM5x 架构
isSM5xDevice = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 5
# 检查当前设备是否是SM80或更高版本，并保存结果
isLessThanSM80Device = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8

# 计算两个张量之间的相对容差（relative tolerance）
def get_rtol(true_value: torch.Tensor, computed_value: torch.Tensor) -> float:
    # 计算真实值与计算值之间的偏差
    deviation = true_value - computed_value
    # 计算相对偏差并取绝对值
    deviation = torch.abs(deviation / true_value)
    # 将偏差中的NaN值用默认的rtol填充
    torch.nan_to_num_(deviation, nan=default_rtol[computed_value.dtype])
    return deviation.max().item()

# 计算两个张量之间的绝对容差（absolute tolerance）
def get_atol(true_value: torch.Tensor, computed_value: torch.Tensor) -> float:
    # 计算真实值与计算值之间的偏差并取绝对值后的最大值
    deviation = true_value - computed_value
    atol = torch.abs(deviation).max().item()
    return atol

# 获取用于比较两个张量的绝对容差和相对容差
def get_tolerances(
    true_value: torch.Tensor,
    computed_value: torch.Tensor,
    fudge_factor: Optional[float] = None,
) -> Tuple[float, float]:
    """Returns the absolute and relative tolerances for comparing two tensors."""
    # 如果给定了调整因子，则使用它；否则默认为1.0
    fudge_factor = fudge_factor if fudge_factor is not None else 1.0
    # 计算绝对容差
    atol = get_atol(true_value, computed_value)
    # 计算相对容差
    rtol = get_rtol(true_value, computed_value)

    # 根据调整因子和默认容差，计算最终的绝对容差和相对容差
    atol = fudge_factor * max(atol, default_atol[computed_value.dtype])
    rtol = fudge_factor * max(rtol, default_rtol[computed_value.dtype])

    # 处理torch.isclose()在极端情况下的异常行为，参考链接详细说明
    if rtol > 1e30:
        rtol = default_rtol[computed_value.dtype]

    return atol, rtol

# 克隆查询（query）、键（key）、值（value）张量，并将它们转换为指定的数据类型
def query_key_value_clones(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, dtype: torch.dtype = None):
    """ Clones the query, key, and value tensors and moves them to the specified dtype. """
    # 如果未指定数据类型，则使用查询张量的数据类型
    if dtype is None:
        dtype = query.dtype
    # 克隆并移动查询张量到指定的数据类型，并保留梯度信息
    query_ref = query.clone().detach().to(dtype).requires_grad_(query.requires_grad)
    # 克隆并移动键张量到指定的数据类型，并保留梯度信息
    key_ref = key.clone().detach().to(dtype).requires_grad_(key.requires_grad)
    # 克隆并移动值张量到指定的数据类型，并保留梯度信息
    value_ref = value.clone().detach().to(dtype).requires_grad_(value.requires_grad)
    return query_ref, key_ref, value_ref

# 获取平台特定的SDPA（Structured Data Parallelism Algorithm）后端
def get_platform_specific_sdpa():
    ret = []
    # 如果平台支持FLASH_ATTENTION，则添加到返回列表中
    if PLATFORM_SUPPORTS_FLASH_ATTENTION:
        ret.append(SDPBackend.FLASH_ATTENTION)
    # 如果平台支持EFFICIENT_ATTENTION，则添加到返回列表中
    if PLATFORM_SUPPORTS_MEM_EFF_ATTENTION:
        ret.append(SDPBackend.EFFICIENT_ATTENTION)
    # 如果平台支持CUDNN_ATTENTION，则添加到返回列表中
    if PLATFORM_SUPPORTS_CUDNN_ATTENTION:
        ret.append(SDPBackend.CUDNN_ATTENTION)
    # 如果没有匹配到任何后端，则添加EFFICIENT_ATTENTION作为占位符
    if not ret:
        ret.append(SDPBackend.EFFICIENT_ATTENTION)
    return ret

# 获取平台特定的SDPA后端列表并保存到常量中
PLATFORM_SPECIFIC_SDPA = get_platform_specific_sdpa()

# 表示EFFICIENT_ATTENTION后端能够支持：
# 1. 序列长度大于512
# 2. 头部维度大于64
MEM_EFF_CAPABILITY_MATCHES_SM80 = SM80OrLater or TEST_WITH_ROCM

# 创建具有给定形状、设备、数据类型和类型的随机稠密或嵌套张量
def rand_sdpa_tensor(shape: SdpaShape, device: str, dtype: torch.dtype, type: str,
                     requires_grad: bool = False, packed: bool = False) -> torch.Tensor:
    """Creates rand dense or nested tensor with given shape and type."""
    # 根据给定参数构建一个新的张量（Tensor）
    def construct_tensor(shape, device, dtype, type, requires_grad=False, packed=False):
        # 从 shape 中提取 batch, num_heads, seq_len, head_dim 这些维度信息
        batch, num_heads, seq_len, head_dim = shape.batch, shape.num_heads, shape.seq_len, shape.head_dim
        
        # 如果 type 是 "nested"，则执行以下逻辑
        if type == "nested":
            # 如果 seq_len 是一个列表，则按照列表中的每个值创建张量尺寸，并使用 torch.nested.nested_tensor 进行创建
            if isinstance(seq_len, list):
                # 定义一个内部函数 _size(i)，用于返回张量的尺寸
                def _size(i):
                    return (seq_len[i], num_heads, head_dim) if not packed else (seq_len[i], 3 * num_heads * head_dim)
    
                # 使用列表推导式创建多个张量，每个张量尺寸由 _size(i) 决定，然后放入一个嵌套结构的张量中
                return torch.nested.nested_tensor([
                    torch.randn(_size(i), device=device, dtype=dtype, requires_grad=requires_grad)
                    for i in range(batch)])
            else:
                # 如果 seq_len 不是列表，则使用单一的尺寸创建张量，并放入一个嵌套结构的张量中
                size = (seq_len, num_heads, head_dim) if not packed else (seq_len, 3 * num_heads * head_dim)
                return torch.nested.nested_tensor([
                    torch.randn(size, device=device, dtype=dtype, requires_grad=requires_grad)
                    for _ in range(batch)])
        else:
            # 如果 type 不是 "nested"，则确保 seq_len 是整数类型
            assert isinstance(seq_len, int)
            # 使用给定的尺寸创建张量，根据 packed 参数决定是否创建一个合并的 QKV 张量
            size = (batch, seq_len, num_heads, head_dim) if not packed else (batch, seq_len, 3 * num_heads * head_dim)
            return torch.randn(size, device=device, dtype=dtype, requires_grad=requires_grad)
# 计算 NT 张量容差值
def calculate_nt_tolerances(nt_ref_hp, nt_ref_lp, default_dtype, fudge_factor=1):
    # 默认容差从 default_atol 字典中获取
    ref_atol = default_atol[default_dtype]
    # 默认相对容差从 default_rtol 字典中获取
    ref_rtol = default_rtol[default_dtype]
    # 遍历高优和低优 NT 张量的组件
    for tensor_component_ref, tensor_component_ref_lp in zip(nt_ref_hp.unbind(), nt_ref_lp.unbind()):
        # 计算每个组件的绝对容差，取其最大值
        ref_atol = max((fudge_factor * torch.abs(tensor_component_ref - tensor_component_ref_lp)).max().item(), ref_atol)
        # 计算每个组件的相对容差，取其最大值
        ref_rtol = max(get_rtol(tensor_component_ref, tensor_component_ref_lp), ref_rtol)
    # 返回计算得到的绝对容差和相对容差
    return ref_atol, ref_rtol

class TestTransformers(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    @onlyCUDA
    @unittest.skip("4D mask not supported yet - activate when 4D mask supported")
    def test_self_attn_TxT_attn_mask(self, device):
        embed_dim = 16
        num_heads = 4
        batch_size = 10
        tgt_len = 16

        # 创建随机查询张量 query，形状为 [N, T, D]
        query = torch.rand(batch_size, tgt_len, embed_dim, device=device)
        
        # 创建随机的注意力掩码 attn_mask，形状为 [T, T]
        attn_mask = torch.randint(0, 2, (tgt_len, tgt_len)).cuda().float()
        # 将掩码中的 0 替换为负无穷大，1 替换为 0.0
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, 0.0)

        # 将 2D 注意力掩码扩展为 4D，形状为 [N, num_heads, T, T]
        attn_mask_4d = attn_mask.expand(batch_size, num_heads, tgt_len, tgt_len)

        # 创建一个多头注意力模型，使用 CUDA 运行
        mta_model = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).cuda()
        # 设置模型为评估模式
        mta_model.eval()

        # 在推断模式下生成 4D 结果
        with torch.inference_mode():
            output_mask_4d = mta_model(query, query, query, attn_mask=attn_mask_4d)[0]
            # 调整输出维度顺序为 [N, T, D]
            output_mask_4d = output_mask_4d.transpose(0, 1)

            # 使用 2D 注意力掩码生成 TxT 结果
            output_mask_TxT = mta_model(query, query, query, attn_mask=attn_mask)[0]
            # 调整输出维度顺序为 [N, T, D]
            output_mask_TxT = output_mask_TxT.transpose(0, 1)

            # 断言两个输出是否相等
            self.assertEqual(output_mask_4d, output_mask_TxT)

    @slowTest
    # 定义一个测试方法，用于测试带填充和捕获错误的 Transformer 编码器
    def test_train_with_pad_and_catch_error(self, device):
        # 设定迭代次数
        iters = 100
        # 创建一个填充掩码张量，指定某些位置不参与计算
        pad_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool).to(device)
        # 创建 Transformer 编码器层
        layer = nn.TransformerEncoderLayer(
            d_model=2,  # 模型维度
            dim_feedforward=4,  # 前馈网络的维度
            nhead=2,  # 头数
            batch_first=True,  # 批量维度是否在第一维
            activation="gelu",  # 激活函数类型
            dropout=0,  # dropout 比例
        )
        # 定义均方误差损失函数
        criterion = nn.MSELoss()
        # 创建 Transformer 编码器
        encoder = nn.TransformerEncoder(layer, 2).to(device)
        # 定义优化器
        optimizer = optim.SGD(encoder.parameters(), lr=0.1, momentum=0.9)
        # 设置编码器为训练模式
        encoder.train()
        # 进行多次迭代训练
        for i in range(iters):
            encoder.train()
            # 梯度清零
            optimizer.zero_grad()
            # 创建输入张量，包含随机数和零填充，并移动到设备上
            inputs = torch.cat([torch.randn(1, 2, 2), torch.zeros(1, 2, 2)], dim=1).to(device)

            # 使用编码器进行前向传播，传入填充掩码
            outputs = encoder(inputs, src_key_padding_mask=pad_mask)

            # 计算损失
            loss = criterion(outputs[:, 0:2, :], inputs[:, 0:2, :])
            # 反向传播
            loss.backward()
            # 更新优化器参数
            optimizer.step()

            with torch.no_grad():
                # 创建测试张量，包含随机数和零填充，并移动到设备上
                test = torch.cat([torch.randn(1, 2, 2), torch.zeros(1, 2, 2)], dim=1).to(device)

                # 预期会引发 uint8 类型不支持的异常
                ex = None
                try:
                    test_train_uint8 = encoder(test, src_key_padding_mask=pad_mask.to(torch.uint8))
                except AssertionError as e:
                    continue
                # 断言捕获到 uint8 类型不支持的异常
                self.assertFalse(e, "Failed to catch unsupported uint8 type exception")  # noqa: F821

                # 使用默认的填充掩码类型进行编码
                test_train_bool = encoder(test, src_key_padding_mask=pad_mask)
                # 设置编码器为评估模式
                encoder.eval()

                # 预期会引发 long 类型不支持的异常
                ex = None
                try:
                    test_eval_uint8 = encoder(test, src_key_padding_mask=pad_mask.to(torch.int64))
                except AssertionError as e:
                    continue
                # 断言捕获到 long 类型不支持的异常
                self.assertFalse(e, "Failed to catch unsupported Long type exception")  # noqa: F821

                # 使用默认的填充掩码类型进行编码
                test_eval_bool = encoder(test, src_key_padding_mask=pad_mask)
                # 计算布尔类型填充掩码下的 L1 损失
                l1_bool = nn.L1Loss()(test_train_bool[:, 0:2, :], test_eval_bool[:, 0:2, :]).item()
                # 断言评估模式和训练模式下的差异小于指定阈值
                self.assertTrue(l1_bool < 1e-4, "Eval/Train difference in pad_mask BOOL")

    # 参数化测试方法的参数列表，用于自动生成多组测试用例
    @parametrize("attn_mask_dim", [2, 3, None])
    @parametrize("key_padding_mask_dim", [2, None])
    @parametrize("mask_dtype", [torch.bool, torch.float32])
    # 定义一个测试函数，用于测试 Multihead Attention 的快速路径和注意力掩码
    def test_multiheadattention_fastpath_attn_mask(self, device, attn_mask_dim, key_padding_mask_dim, mask_dtype):
        # 禁止梯度计算，因为这是一个测试函数
        with torch.no_grad():
            B = 2  # 批次大小
            L = 4  # 序列长度
            D = 8  # 模型维度
            H = 4  # 头数

            # 根据不同的维度情况，创建注意力掩码
            if attn_mask_dim == 2:
                attn_mask = make_tensor((L, L), dtype=mask_dtype, device=device)
            elif attn_mask_dim == 3:
                attn_mask = make_tensor((B * H, L, L), dtype=mask_dtype, device=device)
            elif attn_mask_dim is None:
                attn_mask = None

            # 根据不同的维度情况，创建填充掩码
            if key_padding_mask_dim == 2:
                key_padding_mask = make_tensor((B, L), dtype=mask_dtype, device=device)
            elif key_padding_mask_dim is None:
                key_padding_mask = None

            # 创建一个 Multihead Attention 模型
            mha = nn.MultiheadAttention(D, H, batch_first=True, device=device)
            X = torch.randn(B, L, D, device=device)

            # 将模型设置为训练模式，以禁用快速路径
            mha.train()
            out, _ = mha(X, X, X, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
            
            # 将模型设置为评估模式，以启用快速路径
            mha.eval()
            out_fp, _ = mha(X, X, X, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
            
            # 使用断言检查禁用和启用快速路径的输出是否相等
            self.assertEqual(out, out_fp)

    # 使用参数化装饰器的方式定义一个测试函数，测试 TransformerEncoderLayer 的源掩码
    @parametrize("nhead", [1, 4, 8])
    def test_transformerencoderlayer_src_mask(self, device, nhead):
        batch_size = 2  # 批次大小
        seqlen = 4  # 序列长度
        d_model = 8  # 模型维度
        dim_feedforward = 32  # 前馈网络维度

        # 创建 TransformerEncoderLayer 模型，并将其移动到指定设备上
        model = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True).to(device)
        
        # 创建输入数据源，并将其移动到指定设备上
        src = torch.rand(batch_size, seqlen, d_model).to(device)
        
        # 创建全零的源掩码张量，并将其转换为布尔类型，然后移动到指定设备上
        src_mask = torch.zeros(seqlen, seqlen).to(torch.bool).to(device)

        # 对模型进行一次前向传播，使用源掩码
        model(src, src_mask=src_mask)
        
        # 将模型设置为评估模式，然后进行一次无梯度的前向传播
        model.eval()
        with torch.no_grad():
            model(src, src_mask=src_mask)

    # 使用参数化装饰器的方式定义一个测试函数，测试带钩子的 TransformerEncoderLayer 的自注意力机制
    @parametrize("nhead", [3, 4])
    def test_transformerencoderlayer_no_fastpath_with_hooks(self, device, nhead):
        batch_size = 2  # 批次大小
        seqlen = 4  # 序列长度
        d_model = 12  # 模型维度

        # 创建 TransformerEncoderLayer 模型，并将其移动到指定设备上，同时设置为评估模式
        model = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model,
            batch_first=True).to(device).eval()
        
        # 创建输入数据源，并将其移动到指定设备上
        src = torch.rand(batch_size, seqlen, d_model).to(device)

        # 用于存储输出的缓存列表
        cache = []

        # 定义一个前向钩子函数，用于保存输出
        def hook(module, inputs, output):
            cache.append(output[0].detach())

        # 注册钩子到 self-attention 层，以获取输出
        handle = model.self_attn.register_forward_hook(hook)

        # 使用推理模式进行前向传播
        with torch.inference_mode():
            model(src)

        # 使用断言检查输出缓存列表的长度是否为1，即只有一个输出
        assert len(cache) == 1, f"Expected 1 output, got {len(cache)}"

        # 移除注册的钩子
        handle.remove()

    # 使用参数化装饰器的方式定义一个测试函数，参数化项为是否使用 TorchScript
    @parametrize("use_torchscript", [False])
    @parametrize("enable_nested_tensor", [True, False])
    @parametrize("use_autocast", [True, False])
    @parametrize("d_model", [12, 256])
    @parametrize("with_no_grad", [True, False])
    @parametrize("training", [True, False])
    @parametrize("enable_nested_tensor", [False])
    
    
    # 参数化装饰器，为测试方法提供多组参数进行参数化测试
    def test_transformerencoder_square_input(self, with_no_grad, training, enable_nested_tensor, device):
        """
        Test for edge cases when input of shape (batch size, sequence length, embedding dimension) has
        batch size == sequence length
        """
        # 创建 TransformerEncoder 模型对象，指定层参数
        model = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=4, nhead=2, dim_feedforward=16, dropout=0.0, batch_first=True),
            num_layers=2,
            enable_nested_tensor=enable_nested_tensor
        ).to(device)
    
        with torch.no_grad():
            # 设置模型参数的常数权重
            for idx, p in enumerate(model.parameters()):
                x = p.data
                sz = x.view(-1).size(0)
                shape = x.shape
                x = torch.cos(torch.arange(0, sz).float().view(shape))
                p.data.copy_(x)
    
        if training:
            # 如果处于训练模式，则设置模型为训练状态
            model = model.train()
        else:
            # 否则设置模型为评估状态
            model = model.eval()
    
        # 创建输入张量 x，形状为 (2, 2, 4)，转换为指定设备的默认数据类型
        x = torch.arange(0, 16).reshape(2, 2, 4).to(torch.get_default_dtype()).to(device)
        # 创建源掩码张量 src_mask，指定数据类型为布尔型，转换为指定设备
        src_mask = torch.Tensor([[0, 1], [0, 0]]).to(torch.bool).to(device)
    
        if with_no_grad:
            # 如果 with_no_grad 为真，则使用 torch.no_grad() 上下文管理器
            cm = torch.no_grad()
        else:
            # 否则使用 contextlib.nullcontext() 上下文管理器
            cm = contextlib.nullcontext()
    
        with cm:
            # 在上下文管理器中使用模型进行前向传播计算，传入输入张量 x 和源掩码 src_mask
            result = model(x, mask=src_mask)
    
        # 创建参考输出张量 ref_output，与计算结果进行形状和数值的断言比较
        ref_output = torch.Tensor([[[2.420306205749512, 0.017629241570830, -0.607857942581177, -0.085519507527351],
                                    [2.420306205749512, 0.017629241570830, -0.607857942581177, -0.085519507527351]],
                                   [[2.419836044311523, 0.017548924311996, -0.608187675476074, -0.085347734391689],
                                    [2.419836044311523, 0.017548924311996, -0.608187675476074, -0.085347734391689]]]
                                  ).to(device)
        # 断言计算结果的形状与参考输出的形状相等
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        # 使用 torch.testing.assert_close 进行数值的断言比较，设定相对误差和绝对误差的阈值
        torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)
    
    @parametrize("batch_first", [True, False])
    @parametrize("training", [True, False])
    @parametrize("enable_nested_tensor", [True, False])
    @unittest.skipIf(sys.version_info < (3, 11), "not supported on pre-3.11 Python")
    def test_encoder_padding_and_src_mask_bool(self):
        # 创建一个 Transformer 编码器层对象
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=16,                      # 模型维度为 16
            nhead=2,                         # 多头注意力机制的头数为 2
            dim_feedforward=32,              # 前馈神经网络的隐藏层维度为 32
            dropout=0.1,                     # Dropout 概率为 0.1
            activation='relu',               # 激活函数为 ReLU
            batch_first=True,                # 输入张量的第一个维度为 batch_size
        )
        # 对编码器输出进行 Layer Normalization
        encoder_norm = nn.LayerNorm(16)
        # 创建 Transformer 编码器对象，使用上面定义的编码器层和 Layer Normalization
        encoder = nn.TransformerEncoder(
            encoder_layer, 2, encoder_norm   # 2 表示堆叠 2 层编码器层
        )

        # 创建输入张量，大小为 (2, 3, 16)，表示 batch_size 为 2，序列长度为 3，特征维度为 16
        inputs = torch.randn(2, 3, 16)

        # 创建源序列掩码，是一个上三角形式的 bool 张量
        src_mask = torch.ones(3, 3, dtype=torch.bool).triu_(diagonal=1)
        # 创建填充掩码，是一个 bool 张量，用来指示哪些位置是填充的
        input_seq_len = torch.tensor([3, 2])
        padding_mask = (
            torch.arange(3)[None, :].cpu() >= input_seq_len[:, None]
        )

        # 使用 TorchDynamo 进行断言
        with (self.assertNoLogs(None) if not TEST_WITH_TORCHDYNAMO else contextlib.nullcontext()):
            # 调用编码器，传入输入张量、源序列掩码和填充掩码
            encoder(
                inputs,
                mask=src_mask,
                src_key_padding_mask=padding_mask,
            )

    @unittest.skipIf(sys.version_info < (3, 11), "not supported on pre-3.11 Python")
    def test_decoder_padding_and_src_mask_bool(self):

        def transformer_decoder(inputs, input_seq_len, memory):
            # 创建一个 Transformer 解码器层对象
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=16,                  # 模型维度为 16
                nhead=2,                     # 多头注意力机制的头数为 2
                dim_feedforward=32,          # 前馈神经网络的隐藏层维度为 32
                dropout=0.1,                 # Dropout 概率为 0.1
                activation='relu',           # 激活函数为 ReLU
                batch_first=True,            # 输入张量的第一个维度为 batch_size
            )
            # 对解码器输出进行 Layer Normalization
            decoder_norm = nn.LayerNorm(16)
            # 创建 Transformer 解码器对象，使用上面定义的解码器层和 Layer Normalization
            decoder = nn.TransformerDecoder(
                decoder_layer, 2, decoder_norm   # 2 表示堆叠 2 层解码器层
            )

            # 创建目标序列掩码，是一个上三角形式的 bool 张量
            src_mask = torch.ones(
                inputs.shape[1], inputs.shape[1], dtype=torch.bool
            ).triu_(diagonal=1)
            # 创建填充掩码，是一个 bool 张量，用来指示哪些位置是填充的
            padding_mask = (
                torch.arange(inputs.shape[1])[None, :].cpu()
                >= input_seq_len[:, None]
            )

            # 调用解码器，传入输入张量、记忆张量、目标序列掩码和填充掩码
            return decoder(
                inputs,
                memory,
                tgt_mask=src_mask,
                tgt_key_padding_mask=padding_mask,
                memory_key_padding_mask=padding_mask,
            )

        # 创建输入张量，大小为 (2, 3, 16)，表示 batch_size 为 2，序列长度为 3，特征维度为 16
        inputs = torch.randn(2, 3, 16)
        # 创建记忆张量，大小与输入张量相同
        memory = torch.randn(2, 3, 16)
        # 创建输入序列长度张量，大小为 (2,)，表示两个序列的实际长度
        input_seq_len = torch.tensor([3, 2])

        # 使用 TorchDynamo 进行断言
        with self.assertNoLogs(None):
            # 调用 transformer_decoder 函数，传入输入张量、输入序列长度张量和记忆张量
            transformer_decoder(inputs, input_seq_len, memory)

    def test_encoder_is_causal(self):

        d_model = 3
        # 创建一个 Transformer 编码器层对象，只有一个头，堆叠 6 层，batch_first=True
        layer = torch.nn.TransformerEncoderLayer(d_model, 1, 6, batch_first=True)
        layer.eval()
        # 创建输入张量，大小为 (1, 5, 3)，表示 batch_size 为 1，序列长度为 5，特征维度为 3
        x = torch.randn(1, 5, d_model)
        # 使用未屏蔽的输入调用编码器层，得到未屏蔽的输出
        unmasked_output = layer(x)
        # 生成一个方形的未来掩码
        mask = torch.nn.Transformer.generate_square_subsequent_mask(x.size(1))
        # 使用未屏蔽的输入和未来掩码调用编码器层，设置 is_causal=True
        is_causal_output = layer(x, src_mask=mask, is_causal=True)
        # 使用未屏蔽的输入和未来掩码调用编码器层，得到屏蔽的输出
        masked_output = layer(x, src_mask=mask)

        # 断言屏蔽的输出等于设置 is_causal=True 的输出
        self.assertEqual(masked_output, is_causal_output)

    @onlyCUDA
    @parametrize("nb_heads", [1, 8])
    @parametrize("bias", [True, False])
    # 定义一个测试函数，用于测试带有指定参数的多头注意力机制
    def test_mha_native_args(self, nb_heads, bias):

        # 定义批处理大小（Batch size）、序列长度（Sequence length）和特征维度（Feature dimension）
        B, L, F = 8, 100, 128
        # 是否将批次数据放在第一维度
        batch_first = True
        # 是否使用快速路径
        fast_path = True
        # 根据偏置值确定是否使用填充遮罩
        use_pad_mask = (bias % 2) == 1

        # 创建一个多头注意力机制对象，并将其移动到 CUDA 设备上
        mha = nn.MultiheadAttention(
            embed_dim=F,
            num_heads=nb_heads,
            batch_first=batch_first,
            bias=bias
        ).cuda()
        # 设置为评估模式
        mha.eval()

        # 根据快速路径条件选择上下文管理器
        ctx = torch.no_grad if fast_path else contextlib.nullcontext
        # 执行上下文管理器中的操作
        with ctx():
            # 创建一个随机张量作为查询、键和值
            x = torch.randn(B, L, F).cuda()
            # 如果不是以批次为第一维，则转置张量
            if not batch_first:
                x = x.transpose(0, 1)

            # 如果需要使用填充遮罩，则创建一个填充遮罩张量
            pad_mask = None
            if use_pad_mask:
                pad_mask = torch.zeros((B, L), dtype=torch.bool).cuda()

            # 调用多头注意力机制，传入查询、键、值和填充遮罩
            mha(query=x, key=x, value=x, key_padding_mask=pad_mask)

    # 定义一个测试函数，用于测试使用嵌套张量的输入和遮罩处理
    def test_kpm_mask_trailing_column_with_nested_tensor(self, device):
        # 创建一个 Transformer 编码器层对象
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=512,
            activation='gelu',
            norm_first=False,
            batch_first=False,
        )
        # 创建一个 Transformer 编码器对象，使用嵌套张量并移动到指定设备
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3, enable_nested_tensor=True).to(device)

        # 创建一个随机输入张量，并将其移动到指定设备
        x = torch.randn(10, 6, 256).to(device)
        # 创建一个遮罩张量，用于指定哪些位置的输入需要被忽略
        mask = torch.ones(6, 10)
        # 在遮罩张量中标记一列（第一列）为需要屏蔽的
        mask[0, :] = 0  # 这里我屏蔽了5列而不是一列
        # 将遮罩张量转换为布尔类型，并移动到指定设备
        mask = mask.bool().to(device)
        # 使用 Transformer 编码器处理输入张量和遮罩张量
        out = transformer_encoder(src=x, src_key_padding_mask=mask)
        # 断言输出张量的第二个维度长度为6
        self.assertEqual(out.shape[1], 6)

    # 在仅 CUDA 环境下执行的测试函数，用于测试使用嵌套张量的输入
    @onlyCUDA
    def test_with_nested_tensor_input(self, device):
        # 创建一个 Transformer 编码器层对象
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=512,
            activation='gelu',
            norm_first=False,
            batch_first=True,
        )
        # 创建一个 Transformer 编码器对象，使用嵌套张量并移动到指定设备
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3, enable_nested_tensor=True).to(device)

        # 将 Transformer 编码器设置为评估模式，并使用无梯度计算上下文
        transformer_encoder.eval()
        with torch.no_grad():
            # 创建一个随机输入张量，并将其移动到指定设备
            x = torch.randn(6, 10, 256).to(device)
            # 创建一个遮罩张量，用于指定哪些位置的输入需要被忽略
            mask = torch.ones(6, 10)
            # 在遮罩张量中标记多列（每行标记5列）为需要屏蔽的
            mask[0, 0:] = 0  # 这里我屏蔽了5列而不是一列
            mask[2, 2:] = 0  # 这里我屏蔽了5列而不是一列
            mask[4, 4:] = 0  # 这里我屏蔽了5列而不是一列
            mask[5, 8:] = 0  # 这里我屏蔽了5列而不是一列
            # 将遮罩张量转换为布尔类型，并移动到指定设备
            mask = mask.bool().to(device)
            # 使用 _nested_tensor_from_mask 方法根据遮罩张量创建嵌套张量
            x = torch._nested_tensor_from_mask(x, mask.logical_not(), mask_check=False)
            # 使用 Transformer 编码器处理输入张量和遮罩张量
            out = transformer_encoder(src=x, src_key_padding_mask=None)

        # 断言输出张量是否为嵌套张量
        self.assertEqual(out.is_nested, True)
    # 测试自定义的 TransformerEncoderLayer 的子类化
    def test_script_encoder_subclass(self, device):
        # 定义一个简单的自定义层，继承自 nn.TransformerEncoderLayer
        class MyCustomLayer(nn.TransformerEncoderLayer):
            pass

        # 创建 TransformerEncoder 对象，使用自定义层作为参数
        encoder = nn.TransformerEncoder(
            MyCustomLayer(d_model=256, nhead=8), num_layers=6
        ).to(device=device)
        # 对 encoder 进行 TorchScript 脚本化
        torch.jit.script(encoder)

    # 从 test_transformerencoderlayer_src_mask 而来，测试 torchscripted transformerencoderlayer 子类
    def test_transformerencoderlayer_subclass(self, device):
        # 定义一个简单的自定义层，继承自 nn.TransformerEncoderLayer
        class MyCustomLayer(nn.TransformerEncoderLayer):
            pass

        # 定义所需的参数
        nhead = 4
        batch_size = 2
        seqlen = 4
        d_model = 8
        dim_feedforward = 32

        # 创建自定义层的实例
        model = MyCustomLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True).to(device)
        # 对模型进行 TorchScript 脚本化
        script_model = torch.jit.script(model)

        # 创建输入数据和掩码
        src = torch.rand(batch_size, seqlen, d_model).to(device)  # bs, seqlen, d_model
        src_mask = torch.zeros(seqlen, seqlen).to(torch.bool).to(device)

        # 设置随机种子，计算原始模型和脚本化模型的结果，并进行断言比较
        torch.manual_seed(42)
        result = model(src, src_mask=src_mask)
        torch.manual_seed(42)
        scripted_result = script_model(src, src_mask=src_mask)
        self.assertEqual(result, scripted_result)

        # 切换模型为评估模式，并再次进行 TorchScript 脚本化
        model.eval()
        script_model = torch.jit.script(model)

        # 使用 torch.no_grad() 上下文，计算模型和脚本化模型的结果，并进行断言比较
        with torch.no_grad():
            result = model(src, src_mask=src_mask)
            scripted_result = script_model(src, src_mask=src_mask)
            self.assertEqual(result, scripted_result)


    def test_transformerencoderlayer_subclass_model(self, device):
        # 定义一个简单的自定义层，继承自 nn.TransformerEncoderLayer
        class MyCustomLayer(nn.TransformerEncoderLayer):
            pass

        # 定义所需的参数
        nhead = 4
        batch_size = 2
        seqlen = 4
        d_model = 8
        dim_feedforward = 32

        # 创建自定义层的实例
        layer = MyCustomLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True)
        # 创建 TransformerEncoder 模型，使用自定义层作为参数
        model = nn.TransformerEncoder(
            layer, num_layers=6
        ).to(device=device)
        # 对模型进行 TorchScript 脚本化
        script_model = torch.jit.script(model)

        # 创建输入数据和掩码
        src = torch.rand(batch_size, seqlen, d_model).to(device)  # bs, seqlen, d_model
        src_mask = torch.zeros(seqlen, seqlen).to(torch.bool).to(device)

        # 设置随机种子，计算原始模型和脚本化模型的结果，并进行断言比较
        torch.manual_seed(42)
        result = model(src, mask=src_mask)
        torch.manual_seed(42)
        scripted_result = script_model(src, mask=src_mask)
        self.assertEqual(result, scripted_result)

        # 切换模型为评估模式，并再次进行 TorchScript 脚本化
        model.eval()
        script_model = torch.jit.script(model)

        # 使用 torch.no_grad() 上下文，计算模型和脚本化模型的结果，并进行断言比较
        with torch.no_grad():
            result = model(src, mask=src_mask)
            scripted_result = script_model(src, mask=src_mask)
            self.assertEqual(result, scripted_result)


    @onlyCUDA
    @unittest.skipIf(not TEST_FAIRSEQ, "Fairseq not found")
    # 定义一个测试函数，测试只包含解码层的情况
    def test_decoder_only_layer(self):
        # 默认的填充索引为0
        DEFAULT_PADDING_IDX = 0

        # 定义 FairseqDecoder 类，继承自 torch.nn.Module
        class FairseqDecoder(torch.nn.Module):
            # 初始化函数，设置解码器的各种参数
            def __init__(
                self,
                embed_dim,  # 嵌入维度
                attention_heads,  # 注意力头数
                ffn_embed_dim,  # 前馈网络嵌入维度
                num_layers,  # 层数
                embedding_layer,  # 嵌入层，必须有 padding_idx 字段
                dropout=0,  # dropout 概率，默认为0
                normalize_before=False,  # 归一化处理是否在前，默认为 False
                torch_encoder=None,  # 可以映射权重的 torch 编码器
                activation="relu",  # 激活函数，默认为 relu
            ):
                super().__init__()

                # 创建一个 TransformerConfig 对象 cfg
                cfg = fairseq_transformer.TransformerConfig()
                cfg.decoder.embed_dim = embed_dim  # 解码器嵌入维度
                cfg.decoder.output_dim = embed_dim  # 解码器输出维度
                cfg.decoder.attention_heads = attention_heads  # 解码器注意力头数
                cfg.decoder.ffn_embed_dim = ffn_embed_dim  # 解码器前馈网络嵌入维度
                cfg.dropout = dropout  # dropout 概率
                cfg.decoder.normalize_before = normalize_before  # 是否前归一化
                cfg.decoder.layers = num_layers  # 解码器层数
                cfg.no_token_positional_embeddings = True  # 不使用令牌位置嵌入
                cfg.no_scale_embedding = True  # 不进行嵌入缩放
                cfg.activation_fn = activation  # 激活函数设置为指定的 activation

                dictionary = {}  # 创建一个空字典，待验证用途

                # 使用 Fairseq 的 TransformerDecoder 构造解码器 self.decoder
                self.decoder = fairseq_transformer.TransformerDecoder(
                    cfg,
                    dictionary,
                    embedding_layer,
                    no_encoder_attn=True,  # 不使用编码器注意力
                    output_projection=None,  # 输出投影设置为 None
                )

                # 如果传入了 torch_encoder，将其转换为 Fairseq 的解码器
                if torch_encoder is not None:
                    self.decoder = torch_to_fairseq(torch_encoder, self.decoder)  # noqa: F821
                # 将解码器设为评估模式，放置在 CUDA 上，并使用半精度浮点数
                self.decoder = self.decoder.eval().cuda().half()

            # 前向传播函数定义，接受 tokens 和其他参数
            def forward(
                self,
                tokens,  # 输入的 tokens
                src_lengths=None,  # 源长度，可选参数
                with_triangle_mask=False,  # 是否使用三角形遮罩
                incremental_state=None,  # 增量状态，可选参数
            ):
                # 调用 self.decoder 进行前向传播
                return self.decoder(
                    prev_output_tokens=tokens,  # 前一个输出的 tokens
                    encoder_out=None,  # 编码器输出为 None
                    incremental_state=incremental_state,  # 增量状态
                    features_only=True,  # 仅返回特征
                    full_context_alignment=not with_triangle_mask,  # 是否完整上下文对齐
                    alignment_layer=None,  # 对齐层设置为 None
                    alignment_heads=None,  # 对齐头数设置为 None
                    src_lengths=src_lengths,  # 源长度
                    return_all_hiddens=False,  # 不返回所有隐藏状态
                )[0]  # 返回第一个元素作为结果

    # 使用 parametrize 装饰器对输入维度、注意力掩码维度和是否因果进行参数化测试
    @parametrize("input_dim,attn_mask_dim,is_causal",
                 [(3, None, False), (3, 2, False), (3, 2, True), (3, 3, False), (3, 3, True),
                  (4, None, False), (4, 2, False), (4, 2, True), (4, 4, False), (4, 4, True)],
                 # 生成测试名称的函数
                 name_fn=lambda input_dim, attn_dim, is_causal: (
                     f"{input_dim}D_input_dim_" + (
                         f"{attn_dim}D_{'causal_' if is_causal else ''}attn_mask"
                         if attn_dim is not None else "no_attn_mask")))
    @parametrize("dropout_p", [0.0, 0.2, 0.5])
    # 参数化测试，使用不同的 dropout_p 值进行多次测试
    @sdpa_kernel(backends=[SDPBackend.MATH])
    # 使用指定的 SDPA 内核（例如数学库）进行设置
    @unittest.skipIf(TEST_WITH_CROSSREF, 'Fastpath not available with crossref')
    # 如果 TEST_WITH_CROSSREF 为真，则跳过测试，因为快速路径在 crossref 模式下不可用
    @torch.no_grad()
    # 在测试期间关闭梯度计算，加快运行速度
    # Test failing MHA when bias was NoneType
    def test_bias_is_none(self):
        # 创建随机张量 x，形状为 (1, 5, 10)
        x = torch.rand((1, 5, 10))
        # 创建一个 Multihead Attention 模型，设定输入特征数为 10，头数为 1，无偏置，批处理为第一维
        model = torch.nn.modules.activation.MultiheadAttention(10, 1, bias=False, batch_first=True)
        # 设置模型为评估模式
        model.eval()
        # 调用模型，传入相同的张量 x 三次
        model(x, x, x)
        # completes without error

    def test_transformer_bias_is_none(self, device):
        # 设置批处理大小为 2，序列长度为 3，模型维度为 8，注意头数为 4
        batch_size = 2
        seqlen = 3
        d_model = 8
        nhead = 4

        # 创建一个 TransformerEncoderLayer，设定模型维度、注意头数、无偏置，批处理为第一维，使用给定的设备
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead, bias=False, batch_first=True, device=device)
        # 设置模型为评估模式
        encoder_layer.eval()
        # 创建输入张量 x，形状为 (batch_size, seqlen, d_model)，使用给定的设备
        x = torch.randn(batch_size, seqlen, d_model, device=device)
        # 调用 encoder_layer 处理输入张量 x
        encoder_layer(x)

        # 使用断言检查是否会出现 UserWarning，警告信息为 "encoder_layer.self_attn was passed bias=False"
        with self.assertWarnsRegex(UserWarning, "encoder_layer.self_attn was passed bias=False"):
            # 创建一个 TransformerEncoder，包含一个 encoder_layer，层数为 1，设定为评估模式
            encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1).eval()
            # 调用 encoder 处理输入张量 x
            encoder(x)

        # 使用断言检查是否会出现 UserWarning，警告信息为 "self_attn was passed bias=False"
        with self.assertWarnsRegex(UserWarning, "self_attn was passed bias=False"):
            # 创建一个 Transformer 模型，设定模型维度、注意头数、无偏置，批处理为第一维，使用给定的设备
            transformer = torch.nn.Transformer(
                d_model=d_model, nhead=nhead, bias=False, batch_first=True, device=device
            ).eval()
            # 调用 transformer 处理两个输入张量 x
            transformer(x, x)
    # 定义一个测试函数，用于测试带有 is_causal 参数的 Transformer 模型训练过程
    def test_train_with_is_causal(self, device):
        # 设置输入数据的大小和维度
        S, L, E, H = 1, 2, 2, 1
        # 创建一个 Transformer 编码器层
        layer = nn.TransformerEncoderLayer(
            d_model=2,  # 模型的输入和输出的特征维度
            dim_feedforward=4,  # 前馈网络中间层的维度
            nhead=H,  # 多头注意力机制的头数
            batch_first=True,  # 输入数据的第一个维度是否为 batch size
            activation="gelu",  # 激活函数类型
            dropout=0,  # Dropout 概率
        )
        # 定义均方误差损失函数
        criterion = nn.MSELoss()
        # 创建一个 Transformer 编码器
        encoder = nn.TransformerEncoder(layer, 2).to(device)
        # 定义优化器
        optimizer = optim.SGD(encoder.parameters(), lr=0.1, momentum=0.9)
        # 设置模型为训练模式
        encoder.train()

        encoder.train()
        # 清空优化器梯度
        optimizer.zero_grad()
        # 生成随机输入数据并将其转移到指定设备上
        inputs = torch.randn(S, L, E).to(device)
        # 生成一个方形的下三角蒙版用于自注意力机制
        mask = torch.nn.Transformer.generate_square_subsequent_mask(
            inputs.size(1), device=device
        )

        # 使用 Transformer 模型进行前向传播，设置 is_causal 参数为 True
        outputs = encoder(inputs, mask=mask, is_causal=True)

        # 计算损失值
        loss = criterion(outputs[:, 0:2, :], inputs[:, 0:2, :])
        # 反向传播计算梯度
        loss.backward()
        # 优化器执行单步优化
        optimizer.step()

        # 使用 is_causal 参数进行推断
        t_qvk = torch.randn((S, L, E), device=device, dtype=torch.float32)
        # 创建一个多头注意力机制
        mha = nn.MultiheadAttention(E, H).to(device)
        # 生成一个方形的蒙版用于自注意力机制
        mask = torch.nn.Transformer.generate_square_subsequent_mask(
            S, device=device
        )

        # 使用多头注意力机制进行前向传播，设置 is_causal 参数为 True
        attn_out, _ = mha(t_qvk, t_qvk, t_qvk, attn_mask=mask, is_causal=True)

        # 当只设置 is_causal 参数而不提供蒙版时，预期会出现运行时错误
        attn_mask = torch.randint(0, 2, size=(L, L), device=device, dtype=torch.bool)
        with self.assertRaises(RuntimeError):
            _ = mha(t_qvk, t_qvk, t_qvk, is_causal=True)

        # 通过设置一个因果蒙版来模拟 is_causal 参数为 True 的情况
        causal_mask = torch.triu(
            torch.ones(L, L, device=inputs.device) * float('-inf'), diagonal=1
        ).to(torch.bool)

        # 使用 MagicMock 创建一个模拟的多头注意力层，并将其替换到编码器的指定位置
        mock_layer = MagicMock(torch.nn.MultiheadAttention(E, H), return_value=inputs)
        encoder.layers[1] = mock_layer
        # 使用因果蒙版进行前向传播
        outputs = encoder(inputs, mask=causal_mask)
        # 验证模拟层是否被调用，其中 is_causal 参数被设置为 True
        mock_layer.assert_called_with(ANY, src_mask=ANY, is_causal=True, src_key_padding_mask=ANY)

        # 检查预期的数值结果与所有核心之间的 is_causal 参数
        self.is_causal_kernels([SDPBackend.MATH], device)
    def is_causal_kernels(self, kernels, device):
        # 定义一个内部函数，返回一个在指定设备上全为1的张量
        def ones_tensor(*shape):
            return torch.ones(shape, device=device, dtype=torch.float32).to(device)
        
        # 设置变量 S, L, E, H 的值分别为 1, 2, 4, 1
        S, L, E, H = 1, 2, 4, 1
        
        # 创建一个全为1的张量 qkv，形状为 (S, L, E)
        qkv = ones_tensor(S, L, E)
        
        # 创建一个多头注意力模型对象 mha，设定输入维度 E 和头数 H，并移动到指定设备
        mha = nn.MultiheadAttention(E, H).to(device)
        
        # 设置 mha 的输入投影权重为全1张量，并将其转换为可训练参数
        mha.in_proj_weight = Parameter(torch.ones((E * 3, E), device=device))
        
        # 设置 mha 的输出投影权重为全1张量，并将其转换为可训练参数
        mha.out_proj.weight = Parameter(torch.ones((E, E), device=device))
        
        # 创建一个预期的张量 expected，形状为 (S, L, E)，并移动到指定设备
        expected = torch.ones(size=(S, L, E)).to(device) * 16
        
        # 生成一个用于自注意力的掩码 mask，形状为 (L, L)，并移动到指定设备
        mask = torch.nn.Transformer.generate_square_subsequent_mask(
            qkv.size(1), device=device
        )
        
        # 遍历传入的 kernels 列表
        for kernel in kernels:
            # 使用指定的 SDPA 内核执行上下文管理
            with sdpa_kernel(backends=[kernel]):
                # 使用 mha 模型执行自注意力计算，返回实际的张量 actual 和权重
                actual, _ = mha(qkv, qkv, qkv, attn_mask=mask, need_weights=False, is_causal=True)
                
                # 断言实际输出与预期输出相等
                self.assertTrue(torch.equal(actual, expected))
                
                # 如果内核不是 MATH，则执行以下代码块
                if kernel != SDPBackend.MATH:
                    # 预期引发 RuntimeError 异常，且异常信息包含 "No available kernel"
                    with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                        # 创建新的输入张量 qkv_f 和 mha_f 模型对象
                        qkv_f, mha_f = ones_tensor(S, L, 2), nn.MultiheadAttention(2, H).to(device)
                        
                        # 生成一个新的掩码 mask，形状为 (L, L)，并移动到指定设备
                        mask = torch.nn.Transformer.generate_square_subsequent_mask(
                            qkv_f.size(1), device=device
                        )
                        
                        # 使用 mha_f 模型对象执行自注意力计算
                        _ = mha_f(qkv_f, qkv_f, qkv_f, attn_mask=mask, need_weights=False, is_causal=True)
                        
                        # 等待 CUDA 操作完成
                        torch.cuda.synchronize()

    @skipIfRocm  # 跳过测试，如果 ROCm 平台不支持 EFFICIENT_ATTENTION
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Platform does not supposrt fused SDPA or pre-SM80 hardware"
    )
    # 测试函数，验证在 GPU 上执行 is_causal_kernels 方法
    def test_is_causal_gpu(self):
        device = 'cuda'
        self.is_causal_kernels([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION], device)

    # 测试函数，验证脚本化的多头注意力模型在投影权重为 None 时的行为
    def test_script_mha_in_proj_weight_none(self):
        mha = torch.nn.MultiheadAttention(
            embed_dim=128, num_heads=8, kdim=256, vdim=256
        ).eval()

        # 对 mha 模型进行脚本化
        torch.jit.script(mha)

    @unittest.skipIf(TEST_WITH_CROSSREF, 'Fastpath not available with crossref')
    @torch.no_grad()
class TestSDPAFailureModes(NNTestCase):
    """ Used to test the failure modes of scaled_dot_product_attention
    """
    _do_cuda_memory_leak_check = True  # 设置CUDA内存泄漏检查标志
    _do_cuda_non_default_stream = True  # 设置非默认CUDA流标志

    @onlyCUDA
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION or not isSM8XDevice,
        "Does not support fused SDPA or not SM86+ hardware",
    )
    @parametrize("head_dim", [193, 204, 256])
    @parametrize("dropout_p", [0.0, 0.2])
    def test_flash_backward_failure_sm86plus(self, device, head_dim: int, dropout_p: float):
        dtype = torch.float16
        make_tensor = partial(torch.rand, device=device, dtype=dtype)
        # 检查在SM86/89硬件上是否需要梯度并且head_dim大于192的约束条件，
        # 见pytorch/aten/src/ATen/native/transformers/cuda/sdp_utils.h中的check_requires_grad_and_head_dim_gt192_constraints_on_sm86_89
        size = (2, 2, 4, head_dim)
        q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, None, 0.0, False)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            # 由于输入不需要梯度，不应该失败
            flash_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, None, 0.0, False)

            self.assertEqual(math_ref, flash_ref, atol=1e-3, rtol=1e-3)

            # 由于输入需要梯度，应该失败
            q = make_tensor(size, requires_grad=True)
            k = make_tensor(size, requires_grad=True)
            v = make_tensor(size, requires_grad=True)
            if 192 < head_dim <= 224 or (head_dim > 224 and dropout_p != 0.0):
                self.assertRaises(
                    RuntimeError,
                    lambda: torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, None, dropout_p, False
                    ),
                )
            else:
                flash_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, None, dropout_p, False)

    @onlyCUDA
    def test_dispatch_fails_no_backend(self, device):
        dtype = torch.float16
        with sdpa_kernel(backends=[SDPBackend.ERROR]):
            size = (2, 3, 4)
            q = torch.randn(size, device=device, dtype=dtype)
            k = torch.randn(size, device=device, dtype=dtype)
            v = torch.randn(size, device=device, dtype=dtype)
            self.assertRaisesRegex(RuntimeError, "No viable backend for scaled_dot_product_attention was found.",
                                   lambda: torch._fused_sdp_choice(q, k, v))
            self.assertRaisesRegex(RuntimeError, "No viable backend for scaled_dot_product_attention was found.",
                                   lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Does not support fused scaled dot product attention")
    @parametrize(
        "kernel",
        PLATFORM_SPECIFIC_SDPA,
    )


    # 参数化测试，对参数 kernel 使用 PLATFORM_SPECIFIC_SDPA 中的每个值进行测试
    def test_invalid_fused_inputs_dim_3(self, device, kernel: SDPBackend):
        # 进入使用特定 kernel 的测试函数
        with sdpa_kernel(backends=[kernel]):
            # 设置输入张量的尺寸，此处维度不是 4
            size = (2, 3, 8)
            # 设置张量的数据类型为 torch.float16
            dtype = torch.float16
            # 创建随机张量 q、k、v，指定设备和数据类型
            q = torch.randn(size, device=device, dtype=dtype)
            k = torch.randn(size, device=device, dtype=dtype)
            v = torch.randn(size, device=device, dtype=dtype)
            # 断言捕获 UserWarning，提示 query、key、value 需要是 4 维的警告信息
            with self.assertWarnsRegex(UserWarning, "Both fused kernels requires query, key and value to be 4 dimensional"):
                # 断言捕获 RuntimeError，调用 scaled_dot_product_attention 函数
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))


    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Does not support fused scaled dot product attention")
    @parametrize(
        "kernel",
        PLATFORM_SPECIFIC_SDPA,
    )


    # 仅在 CUDA 下运行的测试，如果不支持融合的缩放点积注意力则跳过
    def test_invalid_fused_inputs_broadcast(self, device, kernel: SDPBackend):
        # 进入使用特定 kernel 的测试函数
        with sdpa_kernel(backends=[kernel]):
            # Fused Kernels 不支持稠密输入的广播
            dtype = torch.float16
            size = (2, 4, 3, 8)
            size_broadcast = (1, 4, 3, 8)
            # 创建随机张量 q、k、v，指定设备和数据类型
            q = torch.randn(size_broadcast, device=device, dtype=dtype)
            k = torch.randn(size, device=device, dtype=dtype)
            v = torch.randn(size, device=device, dtype=dtype)
            # 断言捕获 RuntimeError，调用 scaled_dot_product_attention 函数
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, None, 0.0, False))


    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Does not support fused scaled dot product attention")
    @parametrize("kernel", PLATFORM_SPECIFIC_SDPA)


    # 仅在 CUDA 下运行的测试，如果不支持融合的缩放点积注意力则跳过
    def test_invalid_sequence_lengths(self, device, kernel: SDPBackend):
        # 进入使用特定 kernel 的测试函数
        with sdpa_kernel(backends=[kernel]):
            # 传入 q、k、v 的长度为 0 的序列将导致错误
            dtype = torch.float16
            make_tensor = partial(torch.rand, device=device, dtype=dtype)
            size = SdpaShape(2, 2, 0, 8)
            # 创建随机张量 q、k、v，指定设备和数据类型
            q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
            # 断言捕获 UserWarning，提示不支持零长度的序列长度警告信息
            with self.assertWarnsRegex(UserWarning, "Both fused kernels do not support zero seq_len_q or seq_len_kv."):
                # 断言捕获 RuntimeError，调用 scaled_dot_product_attention 函数
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))
    def test_invalid_last_dim_stride(self, device, kernel: SDPBackend):
        with sdpa_kernel(backends=[kernel]):
            # Passing in a q,k,v with last dim stride not equal to 1 will error
            # 设置数据类型为 torch.float16
            dtype = torch.float16
            # 创建一个生成随机张量的函数，指定设备和数据类型
            make_tensor = partial(torch.rand, device=device, dtype=dtype)
            # 定义张量的大小
            size = SdpaShape(2, 2, 8, 8)
            # 生成三个张量 q, k, v
            q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
            # 修改张量 q 的步长，使最后一个维度的步长不等于 1，预期引发错误
            q.as_strided_(size, [2, 2, 2, 2])
            # 断言捕获 UserWarning，验证警告信息是否包含指定文本
            with self.assertWarnsRegex(UserWarning, "Both fused kernels require the last dimension of the input to have stride 1."):
                # 断言捕获 RuntimeError 异常，验证函数调用是否引发异常
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not flash_attention fused scaled dot product attention")
    @parametrize("kernel", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    def test_invalid_fused_inputs_head_dim(self, device, kernel: SDPBackend):
        with sdpa_kernel(backends=[kernel]):
            # The embed dim per head is not divisible by 8 for flash attention
            # 设置数据类型为 torch.float16
            dtype = torch.float16
            # 创建一个生成随机张量的函数，指定设备和数据类型
            make_tensor = partial(torch.rand, device=device, dtype=dtype)
            # 根据不同的内核类型设置张量的大小
            size = SdpaShape(2, 2, 3, 9) if kernel == SDPBackend.EFFICIENT_ATTENTION else SdpaShape(2, 2, 3, 257)
            # 生成三个张量 q, k, v
            q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
            # 断言捕获 RuntimeError 异常，验证函数调用是否引发异常
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Does not support fused scaled dot product attention")
    @parametrize(
        "kernel",
        PLATFORM_SPECIFIC_SDPA,
    )
    def test_invalid_fused_inputs_invalid_dtype(self, device, kernel: SDPBackend):
        with sdpa_kernel(backends=[kernel]):
            # Invalid dtype for both Flash Attention and Mem Efficient Attention
            # 定义张量的大小
            size = SdpaShape(2, 2, 3, 16)
            # 创建一个生成随机张量的函数，指定设备和数据类型为 torch.float64（无效的数据类型）
            make_tensor = partial(torch.rand, device=device, dtype=torch.float64)
            # 生成三个张量 q, k, v
            q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
            # 断言捕获 RuntimeError 异常，验证函数调用是否引发异常
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention")
    @parametrize("kernel", [SDPBackend.FLASH_ATTENTION])
    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support fused SDPA or pre-SM80 hardware")
    def test_invalid_fused_inputs_attn_mask_present(self, device, kernel: SDPBackend):
        # 使用 `sdpa_kernel` 上下文管理器，设置当前的 SDP 内核为指定的 `kernel`
        with sdpa_kernel(backends=[kernel]):
            # 为不支持的 SDP 参数设置失败条件
            size = SdpaShape(2, 2, 3, 16)
            # 创建指定设备和数据类型的随机张量生成器
            make_tensor = partial(torch.rand, size, device=device, dtype=torch.float16)
            # 生成 q, k, v 张量
            q, k, v = make_tensor(), make_tensor(), make_tensor()
            # 创建非空的注意力掩码
            mask = torch.ones((2, 2, 3, 3), device=device, dtype=q.dtype)
            # 断言运行时错误，期望调用 scaled_dot_product_attention 失败
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, mask, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support fused SDPA or pre-SM80 hardware")
    def test_unaligned_tensors(self, device):
        # 根据架构依赖性指定 SM80OrLater，设置数据类型为 torch.float16
        dtype = torch.float16
        size = SdpaShape(2, 2, 8, 5)
        # 创建指定设备和数据类型的随机张量生成器
        make_tensor = partial(torch.rand, size, device=device, dtype=dtype)
        # 生成 q, k, v 张量
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        # 使用 `sdpa_kernel` 上下文管理器，设置当前的 SDP 内核为 EFFICIENT_ATTENTION
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            # 断言运行时错误，期望调用 scaled_dot_product_attention 失败
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support fused SDPA or pre-SM80 hardware")
    def test_flash_fail_fp32(self, device):
        # 设置数据类型为 torch.float
        dtype = torch.float
        size = SdpaShape(16, 16, 32, 32)
        # 创建指定设备和数据类型的随机张量生成器
        make_tensor = partial(torch.rand, size, device=device, dtype=dtype)
        # 生成 q, k, v 张量
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        # 使用 `sdpa_kernel` 上下文管理器，设置当前的 SDP 内核为 FLASH_ATTENTION
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            # 断言运行时错误，期望调用 scaled_dot_product_attention 失败，并产生 UserWarning
            with self.assertWarnsRegex(UserWarning, "Expected query, key and value to all be of dtype: {Half, BFloat16}"):
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support SDPA or pre-SM80 hardware")
    def test_flash_autocast_fp32_float16(self, device):
        # 设置数据类型为 torch.float
        dtype = torch.float
        size = SdpaShape(16, 16, 32, 32)
        # 创建指定设备和数据类型的随机张量生成器
        make_tensor = partial(torch.rand, size, device=device, dtype=dtype)
        # 生成 q, k, v 张量
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        # 使用 torch.autocast 设置自动混合精度计算环境为 cuda 的 torch.float16
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # 使用 `sdpa_kernel` 上下文管理器，设置当前的 SDP 内核为 FLASH_ATTENTION
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                # 调用 scaled_dot_product_attention，自动使用 torch.float16 进行计算
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False)

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support SDPA or pre-SM80 hardware")
    # 测试使用自动转换为 fp32 和 bfloat16 的函数
    def test_flash_autocast_fp32_bfloat16(self, device):
        # 指定张量的数据类型为 float32
        dtype = torch.float
        # 定义张量的大小
        size = SdpaShape(16, 16, 32, 32)
        # 创建张量的部分函数，使用指定的设备和数据类型
        make_tensor = partial(torch.rand, size, device=device, dtype=dtype)
        # 创建查询、键、值张量
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        # 使用自动转换进行上下文管理，设备类型为 'cuda'，数据类型为 torch.bfloat16
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # 使用 sdpa_kernel 上下文管理，指定后端为 FLASH_ATTENTION
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                # 调用 scaled_dot_product_attention 函数进行注意力计算
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False)

    # 注意：不要根据平台截断列表。这些测试应始终引发错误。
    @parametrize("kernel", [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    def test_invalid_inputs_different_datatypes(self, device, kernel: SDPBackend):
        # 使用 sdpa_kernel 上下文管理，指定使用给定的 kernel
        with sdpa_kernel(backends=[kernel]):
            # 不同数据类型的测试
            shape = (1, 4, 8, 16)
            # 创建查询张量，数据类型为 torch.float32
            query = torch.randn(shape, dtype=torch.float32, device=device)
            # 创建键和值张量，数据类型为 torch.float16，与查询张量不同
            key = torch.randn(shape, dtype=torch.float16, device=device)
            value = torch.randn(shape, dtype=torch.float16, device=device)
            # 断言应该抛出 RuntimeError 异常，因为数据类型不匹配
            self.assertRaises(RuntimeError, lambda: F.scaled_dot_product_attention(query, key, value))

    @onlyCUDA
    @parametrize("kernel", [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    def test_invalid_inputs_different_devices(self, device, kernel: SDPBackend):
        # 不同设备的测试
        shape = (1, 4, 8, 16)
        # 创建查询张量，数据类型为 torch.float32，设备为指定的 device
        query = torch.randn(shape, dtype=torch.float32, device=device)
        # 创建键和值张量，数据类型为 torch.float16，但设备为 'cpu'
        key = torch.randn(shape, dtype=torch.float16, device='cpu')
        value = torch.randn(shape, dtype=torch.float16, device='cpu')
        # 断言应该抛出 RuntimeError 异常，因为设备不匹配
        self.assertRaises(RuntimeError, lambda: F.scaled_dot_product_attention(query, key, value))

    @parametrize("kernel", [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    def test_invalid_inputs_1_dimensional_inputs(self, device, kernel: SDPBackend):
        # 使用 sdpa_kernel 上下文管理，指定使用给定的 kernel
        with sdpa_kernel(backends=[kernel]):
            # 1 维输入的测试
            shape = (1, 4)
            # 创建查询张量，数据类型为 torch.float16
            query = torch.randn(4, dtype=torch.float16, device=device)
            # 创建键和值张量，数据类型为 torch.float16，形状为 shape，设备为指定的 device
            key = torch.randn(shape, dtype=torch.float16, device=device)
            value = torch.randn(shape, dtype=torch.float16, device=device)
            # 断言应该抛出 RuntimeError 异常，因为输入张量不是二维的
            self.assertRaises(RuntimeError, lambda: F.scaled_dot_product_attention(query, key, value))

    @onlyCUDA
    @skipIfRocm  # Missing EFFICIENT_ATTENTION
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    # 测试具有嵌套广播错误情况的融合内核
    def test_fused_kernels_nested_broadcasting_error_cases(self, device):
        # 生成部分函数，用于创建随机的嵌套张量，类型为 "nested"，在指定设备上，数据类型为 torch.float32
        rand_nested_tensor = partial(rand_sdpa_tensor, type="nested", device=device, dtype=torch.float32)
        # 定义批次大小、头数、头维度
        batch, num_heads, head_dim = 32, 8, 64
        # 随机生成查询序列长度列表，每个元素在区间 [1, 32) 中
        seq_lens_q = torch.randint(low=1, high=32, size=(batch,)).tolist()
        # 随机生成值序列长度列表，每个元素在区间 [1, 32) 中
        seq_lens_v = torch.randint(low=1, high=32, size=(batch,)).tolist()

        # 创建查询形状对象
        q_shape = SdpaShape(batch, num_heads, seq_lens_q, head_dim)
        # 创建键形状对象，用于广播操作
        k_shape = SdpaShape(1, num_heads, 1, head_dim)
        # 创建值形状对象
        v_shape = SdpaShape(batch, num_heads, seq_lens_v, head_dim)

        # 生成随机嵌套张量作为查询，同时进行维度转置
        query = rand_nested_tensor(q_shape).transpose(1, 2)
        # 生成随机嵌套张量作为键，同时进行维度转置
        key = rand_nested_tensor(k_shape).transpose(1, 2)
        # 生成随机嵌套张量作为值，同时进行维度转置
        value = rand_nested_tensor(v_shape).transpose(1, 2)

        # 使用 SDP 后端的内核
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            # 断言运行时错误，匹配消息中包含 "No available kernel"
            with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                # 调用 scaled_dot_product_attention 函数进行注意力计算
                torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

    # 仅在 CUDA 下执行测试
    @onlyCUDA
    # 如果平台不支持 Flash Attention，则跳过测试
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Fused SDPA was not built for this system")
    # 测试嵌套张量在填充头维度时的失败情况
    def test_nested_fails_on_padding_head_dim(self, device):
        # 数据类型为 torch.bfloat16
        dtype = torch.bfloat16
        # 序列长度列表
        seq_len_list = [2, 4, 5, 6, 7]
        # 创建 SdpaShape 对象，指定批次大小、头数、每个序列长度列表、头维度
        shape = SdpaShape(5, 8, seq_len_list, 57)
        # 部分函数，生成随机张量，形状为 shape，类型为 "nested"，在指定设备上，数据类型为 dtype
        make_tensor = partial(rand_sdpa_tensor, shape=shape, type="nested", device=device, dtype=dtype)
        # 生成查询、键、值张量
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        
        # 使用 Flash Attention 后端的内核
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            # 断言警告消息中包含 "For NestedTensor inputs, Flash attention requires"
            with self.assertWarnsRegex(UserWarning, "For NestedTensor inputs, Flash attention requires"):
                # 断言运行时错误，期望抛出异常
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))

    # 仅在 CUDA 下执行测试
    @onlyCUDA
    # 如果平台不支持融合 SDPA 或当前设备的计算能力不足 SM80，则跳过测试
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION or not isLessThanSM80Device,
                     "Current platform does not support fused SDPA or is an SM80+ device.")
    # 测试内存效率失败情况，当数据类型为 torch.bfloat16 且设备的计算能力小于 SM80 时
    def test_mem_efficient_fail_bfloat16_less_than_sm80(self, device):
        # 数据类型为 torch.bfloat16
        dtype = torch.bfloat16
        # 定义张量大小为 SdpaShape(16, 16, 32, 32)
        size = SdpaShape(16, 16, 32, 32)
        # 部分函数，生成随机张量，形状为 size，类型为 torch.rand，指定设备上数据类型为 dtype
        make_tensor = partial(torch.rand, size, device=device, dtype=dtype)
        # 生成查询、键、值张量
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        # 使用 EFFICIENT_ATTENTION 后端的内核
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            # 断言警告消息中包含 "Expected query, key and value to all be of dtype: {Half, Float}"
            with self.assertWarnsRegex(UserWarning, "Expected query, key and value to all be of dtype: {Half, Float}"):
                # 断言运行时错误，期望抛出异常
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))

    # 仅在 CUDA 下执行测试
    @onlyCUDA
    # 如果平台不支持 Flash Attention，则跳过测试
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention")
    # 在 CUDA 设备上测试带有大量 bf16 NaN 值的闪存注意力机制
    def test_flash_atteention_large_bf16_nan_values(self, device):
        # 创建一个填充满 133120.0 的 torch.bfloat16 张量作为查询
        query = torch.full((1, 1, 1, 64), 133120.0, dtype=torch.bfloat16, device="cuda")
        # 创建一个填充满 133120.0 的 torch.bfloat16 张量作为键
        key = torch.full((1, 1, 1, 64), 133120.0, dtype=torch.bfloat16, device="cuda")
        # 创建一个填充满 133120.0 的 torch.bfloat16 张量作为值
        value = torch.full((1, 1, 1, 64), 133120.0, dtype=torch.bfloat16, device="cuda")

        # 使用 SDPABackend.FLASH_ATTENTION 上下文环境，执行 scaled_dot_product_attention 函数
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = torch.nn.functional.scaled_dot_product_attention(query, key, value)

        # 断言输出张量中不包含 NaN 值，否则输出指定的错误信息
        self.assertFalse(torch.isnan(out).any(), "Output should not contain NaNs!")

    # 仅在 CUDA 上运行的测试，跳过不支持融合注意力的平台
    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Fused SDPA was not built for this system")
    # 参数化测试，根据平台是否支持 FLASH_ATTENTION 决定使用哪种融合内核
    @parametrize("fused_kernel", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION] if
                 PLATFORM_SUPPORTS_FLASH_ATTENTION else [SDPBackend.EFFICIENT_ATTENTION])
    def test_fused_kernels_seq_len_0_inputs(self, device, fused_kernel):
        # 部分随机生成符合 SDPA 张量的嵌套张量，类型为 "nested"，在指定设备上使用 torch.float16 类型
        rand_nested_tensor = partial(rand_sdpa_tensor, type="nested", device=device, dtype=torch.float16)
        batch, num_heads, head_dim = 32, 16, 64
        # 随机生成序列长度，范围为 [1, 32]，大小为 batch
        seq_lens = torch.randint(low=1, high=32, size=(batch,))
        # 确保部分序列长度为 0
        num_zeros = 10
        indices = torch.randint(low=0, high=batch, size=(num_zeros,))
        seq_lens.scatter_(0, indices, 0)

        # 创建 SdpaShape 对象，描述 SDPA 张量的形状
        shape = SdpaShape(batch, num_heads, seq_lens.tolist(), head_dim)
        # 生成随机嵌套张量作为查询、键、值张量
        query = rand_nested_tensor(shape)
        key = rand_nested_tensor(shape)
        value = rand_nested_tensor(shape)

        # 转置查询、键、值张量的维度 1 和 2
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # 使用指定的融合内核上下文环境，测试 scaled_dot_product_attention 函数
        with sdpa_kernel(backends=[fused_kernel]):
            # 断言执行 scaled_dot_product_attention 函数时抛出 RuntimeError 异常，并包含指定的错误信息
            with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

    # 仅在 CUDA 上运行的测试，如果不支持 FLASH_ATTENTION，则跳过该测试
    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Fused SDPA was not built for this system")
    # 定义一个测试方法，用于测试融合内核在嵌套广播场景下的梯度跟踪失败情况，接受设备参数
    def test_fused_kernels_nested_broadcasting_requires_grad_failure(self, device):
        # 部分应用 rand_sdpa_tensor 函数，生成类型为“nested”的随机张量，设备为指定设备，数据类型为 torch.float16，需要梯度
        rand_nested_tensor = partial(rand_sdpa_tensor, type="nested", device=device, dtype=torch.float16, requires_grad=True)
        # 设置批次大小、头数、头维度和值维度
        batch, num_heads, head_dim, head_dim_v = 32, 16, 64, 64
        # 随机生成序列长度列表，列表长度为批次大小，取值范围在 [1, 32) 之间
        seq_lens = torch.randint(low=1, high=32, size=(batch,)).tolist()
        # 创建 SdpaShape 对象，定义查询形状
        q_shape = SdpaShape(1, num_heads, 1, head_dim)
        # 创建 SdpaShape 对象，定义键形状
        k_shape = SdpaShape(batch, num_heads, seq_lens, head_dim)
        # 创建 SdpaShape 对象，定义值形状
        v_shape = SdpaShape(batch, 1, seq_lens, head_dim_v)

        # 创建密集查询张量，随机初始化，设备为指定设备，数据类型为 torch.float16，需要梯度
        query = torch.randn(q_shape, device=device, dtype=torch.float16, requires_grad=True)
        # 生成随机嵌套张量作为键，使用 rand_nested_tensor 函数
        key = rand_nested_tensor(k_shape)
        # 生成随机嵌套张量作为值，使用 rand_nested_tensor 函数
        value = rand_nested_tensor(v_shape)

        # 对查询张量进行维度转置，交换维度 1 和 2
        query = query.transpose(1, 2)
        # 对键张量进行维度转置，交换维度 1 和 2
        key = key.transpose(1, 2)
        # 对值张量进行维度转置，交换维度 1 和 2
        value = value.transpose(1, 2)

        # 使用 sdpa_kernel 上下文管理器，设置后端为 FLASH_ATTENTION
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            # 使用 assertWarnsRegex 断言捕获 UserWarning，检查是否出现指定警告信息
            with self.assertWarnsRegex(UserWarning, "Both fused kernels do not support training with broadcasted NT inputs"):
                # 使用 assertRaisesRegex 断言捕获 RuntimeError，检查是否出现指定错误信息
                with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                    # 调用 scaled_dot_product_attention 函数进行注意力计算，传入查询、键、值张量以及其他参数
                    out = torch.nn.functional.scaled_dot_product_attention(
                        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

    # 仅在 CUDA 环境下执行的测试方法，用于测试 FLASH_ATTENTION 在非方形因果注意力场景下的失败情况，接受设备参数
    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention")
    def test_flash_attention_fail_with_non_square_causal_attention(self, device):
        # 设置数据类型为 torch.bfloat16
        dtype = torch.bfloat16
        # 创建 SdpaShape 对象，定义查询形状
        q_shape = SdpaShape(1, 1, 8, 16)
        # 创建 SdpaShape 对象，定义键值形状
        kv_shape = SdpaShape(1, 1, 12, 16)
        # 部分应用 torch.rand 函数，生成指定形状的随机张量，设备为指定设备，数据类型为指定数据类型
        make_q = partial(torch.rand, q_shape, device=device, dtype=dtype)
        make_kv = partial(torch.rand, kv_shape, device=device, dtype=dtype)
        # 创建查询张量 q、键张量 k 和值张量 v，使用 make_q 和 make_kv 函数生成
        q, k, v = make_q(), make_kv(), make_kv()
        # 设置警告字符串，提示 FLASH_ATTENTION 不支持非方形因果注意力场景
        warning_str = "Flash attention does not support the is_causal flag when seqlen_q != seqlen_k."
        # 使用 sdpa_kernel 上下文管理器，设置后端为 FLASH_ATTENTION
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            # 使用 assertWarnsRegex 断言捕获 UserWarning，检查是否出现指定警告信息
            with self.assertWarnsRegex(UserWarning, warning_str):
                # 使用 assertRaises 断言捕获 RuntimeError，检查是否出现运行时错误
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, is_causal=True))
# 根据设备、头维度、是否使用dropout、是否因果关系确定块大小
def _get_block_size_n(device, head_dim, is_dropout, is_causal):
    # 断言头维度不超过256，与CUDA内核中的块大小匹配
    assert head_dim <= 256
    # 获取CUDA设备的主要和次要版本号
    major, minor = torch.cuda.get_device_capability(device)
    # 判断是否为SM8x架构，排除SM80（A100）
    is_sm8x = major == 8 and minor > 0
    # 判断是否为SM80架构（A100）
    is_sm80 = major == 8 and minor == 0
    # 判断是否为SM90架构
    is_sm90 = major == 9 and minor == 0
    if head_dim <= 32:
        return 128
    if head_dim <= 64:
        # 如果不使用dropout，则块大小为128，否则为64
        return 128 if not is_dropout else 64
    elif head_dim <= 96:
        return 64
    elif head_dim <= 128:
        # 如果是SM8x架构且不使用dropout且是因果关系，则块大小为64，否则为32
        if is_sm8x:
            return 64 if (not is_dropout and is_causal) else 32
        else:
            return 64 if not is_dropout else 32
    elif head_dim <= 160:
        # 如果是SM8x架构，则块大小为64，否则为32
        if is_sm8x:
            return 64
        else:
            return 32
    elif head_dim <= 192:
        return 64
    elif head_dim <= 224:
        return 64
    elif head_dim <= 256:
        return 64


# 对输入张量的最后一个维度进行填充，使其能够被指定对齐大小整除
def pad_last_dim(input_tensor, alignment_size, slice: bool = False):
    # 获取张量的最后一个维度大小
    last_dim_size = input_tensor.size(-1)
    if (last_dim_size % alignment_size == 0):
        return input_tensor, last_dim_size
    # 计算需要填充的数量
    pad_count = alignment_size - (last_dim_size % alignment_size)
    # 使用F.pad函数对张量进行填充
    padded_tensor = F.pad(input_tensor, (0, pad_count))
    if slice:
        # 如果slice为True，则返回切片后的张量和最后一个维度大小
        return padded_tensor[..., :last_dim_size], last_dim_size
    # 否则返回填充后的张量和最后一个维度大小
    return padded_tensor, last_dim_size
    @parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16, torch.half])
    # 参数化测试，对 dtype 进行多组参数化测试，包括不同的 torch 数据类型
    def test_fused_sdp_choice_cpu(self, device, type: str, dropout: float, dtype: torch.dtype):
        # 测试在 CPU 和 nestedtensor CPU 下返回 MATH 后端
        make_tensor = partial(rand_sdpa_tensor, type=type, device=device, dtype=dtype)
        # 使用部分函数应用，生成随机张量的函数 make_tensor
        size = SdpaShape(2, 8, 128, 64)
        # 定义张量的形状 size
        q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
        # 创建 q, k, v 三个张量，通过 make_tensor 函数生成
        if type == "nested" \
                or dropout > 0.0 \
                or dtype not in [torch.float32, torch.float64, torch.bfloat16, torch.float16]:
            # 如果 type 是 "nested"，或者 dropout 大于 0.0，或者 dtype 不在指定的数据类型列表中
            assert torch._fused_sdp_choice(q, k, v, dropout_p=dropout) == SDPBackend.MATH.value
            # 断言调用 torch._fused_sdp_choice 函数后返回的值等于 SDPBackend.MATH.value
        else:
            assert torch._fused_sdp_choice(q, k, v, dropout_p=dropout) == SDPBackend.FLASH_ATTENTION.value
            # 否则断言调用 torch._fused_sdp_choice 函数后返回的值等于 SDPBackend.FLASH_ATTENTION.value

    @onlyCPU
    # 标记该测试仅在 CPU 上运行
    @parametrize("fused_kernel", [SDPBackend.FLASH_ATTENTION])
    # 参数化测试，对 fused_kernel 进行测试，参数为 SDPBackend.FLASH_ATTENTION
    @parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16, torch.float16])
    # 参数化测试，对 dtype 进行多组参数化测试，包括不同的 torch 数据类型
    @parametrize("batch_size", [2, 12])
    # 参数化测试，对 batch_size 进行测试，参数为 2 和 12
    @parametrize("seq_len", [267, 1030])
    # 参数化测试，对 seq_len 进行测试，参数为 267 和 1030
    @parametrize("n_head", [1, 3])
    # 参数化测试，对 n_head 进行测试，参数为 1 和 3
    @parametrize("head_dim", [8, 16])
    # 参数化测试，对 head_dim 进行测试，参数为 8 和 16
    @parametrize("causal", [True, False])
    # 参数化测试，对 causal 进行测试，参数为 True 和 False
    @parametrize("train", [True, False])
    # 参数化测试，对 train 进行测试，参数为 True 和 False
    def test_scaled_dot_product_fused_attention_vs_math_cpu(
        self,
        device,
        fused_kernel,
        dtype,
        batch_size,
        seq_len,
        n_head,
        head_dim,
        causal,
        train,
    ):
        # 默认的绝对误差和相对误差
        atol = 1e-5
        rtol = 5e-6
        # 如果数据类型是 torch.bfloat16，则调整误差阈值
        if dtype is torch.bfloat16:
            atol = 5e-2
            rtol = 5e-2
        # 如果数据类型是 torch.float16，则调整误差阈值
        if dtype is torch.float16:
            atol = 1e-2
            rtol = 1e-2

        # 计算每个头的总维度
        n_embd = n_head * head_dim
        # 创建一个随机的自注意张量，根据指定的形状和参数
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=dtype, packed=True, requires_grad=False)
        shape = SdpaShape(batch_size, n_head, seq_len, head_dim)
        # 创建两个相同形状的张量 x 和 x2
        x = make_tensor(shape)
        x2 = x.clone()

        # 如果是训练模式，则需要对 x 和 x2 启用梯度追踪
        if train:
            x.requires_grad_(True)
            x2.requires_grad_(True)

        # 将 x 和 x2 分割为 q, k, v 三个部分，每个部分的最后一维为 n_embd
        q, k, v = x.split(n_embd, dim=2)
        q2, k2, v2 = x2.split(n_embd, dim=2)

        # 如果数据类型是 torch.bfloat16 或 torch.float16，则将 q2, k2, v2 转换为 float 类型
        if dtype in [torch.bfloat16, torch.float16]:
            q2 = q2.float()
            k2 = k2.float()
            v2 = v2.float()

        # 转置 k, q, v, k2, q2, v2 的维度，使得维度顺序变为 (batch_size, n_head, seq_len, head_dim)
        k = k.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        q = q.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        k2 = k2.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        q2 = q2.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        v2 = v2.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)

        # 使用 sdpa_kernel 上下文管理器，指定后端为 fused_kernel，计算实际的注意力值
        with sdpa_kernel(backends=[fused_kernel]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal)
        # 使用 sdpa_kernel 上下文管理器，指定后端为 SDPBackend.MATH，计算数学参考的注意力值
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                q2, k2, v2, attn_mask=None, dropout_p=0.0, is_causal=causal)

        # 如果数据类型是 torch.bfloat16 或 torch.float16，则将 math_ref 转换为相应的数据类型
        if dtype in [torch.bfloat16, torch.float16]:
            math_ref = math_ref.to(dtype)

        # 断言实际值与数学参考值在给定的误差范围内相等
        self.assertEqual(actual, math_ref, atol=atol, rtol=rtol)

        # 如果是训练模式，则计算梯度并进行梯度断言
        if train:
            actual.sum().backward()
            math_ref.sum().backward()

            # 分别获取 x 和 x2 的梯度，并将其分割为 q, k, v 三个部分的梯度
            grad_x, grad_x2 = x.grad, x2.grad
            grad_q_actual, grad_k_actual, grad_v_actual = grad_x.split(n_embd, dim=2)
            grad_q_ref, grad_k_ref, grad_v_ref = grad_x2.split(n_embd, dim=2)

            # 断言实际梯度与数学参考梯度在给定的误差范围内相等
            self.assertEqual(grad_q_actual, grad_q_ref, atol=atol, rtol=rtol)
            self.assertEqual(grad_k_actual, grad_k_ref, atol=atol, rtol=rtol)
            self.assertEqual(grad_v_actual, grad_v_ref, atol=atol, rtol=rtol)
    # 定义一个测试方法，用于验证融合注意力机制与数学计算的效果，针对 CPU 运行
    def test_scaled_dot_product_fused_attention_mask_vs_math_cpu(
        self,
        device,
        fused_kernel,
        dtype,
        batch_size,
        q_seq_len,
        kv_seq_len,
        n_head,
        head_dim,
        mask_dim,
        bool_mask,
        train,
    ):
        # 参数化装饰器，指定不同的内核类型进行参数化测试
        @parametrize("kernel", [SDPBackend.MATH])
        # 定义测试方法，验证带有负缩放值的数学计算的缩放点积注意力机制
        def test_scaled_dot_product_attention_math_with_negative_scale(self, device, kernel: SDPBackend):
            # 引用函数，计算输入张量的缩放点积注意力机制的参考结果
            def ref(x):
                # 计算输入张量 x 与其转置的矩阵乘积
                v1 = torch.matmul(x, x.transpose(-1, -2))
                # 将乘积结果除以负数缩放值 -0.0001
                v2 = v1 / -0.0001
                # 对除法结果进行 softmax 操作，沿着最后一个维度进行计算
                v3 = v2.softmax(dim=-1)
                # 计算 softmax 结果与原始输入张量 x 的矩阵乘积
                v4 = torch.matmul(v3, x)
                return v4

            # 创建一个随机张量作为输入 x，维度为 [1, 3, 64, 64]，在指定设备上
            x = torch.randn(1, 3, 64, 64, device=device)
            # 计算参考结果，使用定义的 ref 函数
            ref_result = ref(x)
            # 使用指定的数学内核进行缩放点积注意力机制计算
            with sdpa_kernel(backends=[kernel]):
                sdp_math = torch.nn.functional.scaled_dot_product_attention(x, x, x, scale=-1.0 / 0.0001)
            # 断言参考结果与数学计算结果相等
            self.assertEqual(ref_result, sdp_math)
# NNTestCase 类的子类，用于测试 scaled_dot_product_attention 的 CUDA 特定功能
class TestSDPACudaOnly(NNTestCase):

    """ 用于测试 scaled_dot_product_attention 的 CUDA 特定功能
    Quarks:
        在测试此函数时有些技巧性。其运行行为取决于您正在测试的 CUDA 架构。
        请参阅文件顶部的 `PLATFORM_SUPPORTS_FUSED_ATTENTION`。
        Summary:
            Math: 总是支持
            FlashAttention: 仅在 sm80 或更新的硬件上支持
            MemEfficientAttention: 仅在 sm50 或更新的硬件上支持
    """

    # 是否进行 CUDA 内存泄漏检查
    _do_cuda_memory_leak_check = True
    # 是否使用非默认 CUDA 流进行测试
    _do_cuda_non_default_stream = True

    # TODO 用于测试得分，例如在 ALIBI 测试中我们现在不需要这个
    def normalize_flash_attn_S(
        self,
        attn_unnorm,
        q,
        k,
        v,
        query_padding_mask=None,
        key_padding_mask=None,
        attn_bias=None,
        is_dropout=False,
        causal=False,
        window_size=(-1, -1),  # -1 表示无限窗口大小
        scale=None,
    ):
        """
        Arguments:
            q: (batch_size, seqlen_q, nheads, head_dim)
            k, v: (batch_size, seqlen_k, nheads, head_dim)
            key_padding_mask: (batch_size, seqlen_q)
            attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        Output:
            softmax_lse: (batch_size, nheads, seqlen_q)
            softmax_max: (batch_size, nheads, seqlen_q)
        """
        # 转置 q, k, v 张量的维度，使得注意力计算更高效
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # 如果启用因果注意力，调整窗口大小的维度
        if causal:
            window_size = (window_size[0], 0)
        # 将 q, k, v 张量转换为 float 类型，以进行注意力分数的计算
        q, k, v = q.float(), k.float(), v.float()
        # 获取 q 张量的维度信息
        _, seqlen_q, _, head_dim = q.shape
        # 获取 k 张量的序列长度信息
        seqlen_k = k.shape[1]
        # 获取批次大小信息
        b = q.shape[0]
        # 导入计算缩放因子的函数，并计算缩放值
        from torch.nn.attention.bias import _calculate_scale
        scale = _calculate_scale(head_dim, scale)
        # 计算注意力分数，使用缩放因子对 q 和 k 进行加权乘积
        scores = torch.matmul(q.transpose(1, 2) * scale, k.permute(0, 2, 3, 1))
        # 如果存在键掩码，则将注意力分数中相应位置的分数替换为负无穷大
        if key_padding_mask is not None:
            scores.masked_fill_(~key_padding_mask.view(b, 1, 1, -1), float("-inf"))
        # 如果窗口大小大于等于 0，则构造局部注意力掩码
        if window_size[0] >= 0 or window_size[1] >= 0:
            local_mask = self.construct_local_mask(
                seqlen_q,
                seqlen_k,
                window_size,
                query_padding_mask,
                key_padding_mask,
                q.device,
            )
            scores.masked_fill_(local_mask, float("-inf"))
        # 如果存在注意力偏置，则将其添加到注意力分数中
        if attn_bias is not None:
            scores = scores + attn_bias.to(dtype=scores.dtype)
        # 获取分块大小，用于分块处理注意力分数
        block_size_n = _get_block_size_n(scores.device, head_dim, is_dropout, causal)
        scores_block = scores.split(block_size_n, dim=-1)
        # 计算每个块的对数求和指数（logsumexp）作为每个头部的注意力加权均值
        lse_block = torch.stack([torch.logsumexp(s, dim=-1) for s in scores_block], dim=-1)
        lse = torch.logsumexp(lse_block, dim=-1)
        # 如果 lse 中有 -inf 值（即所有 scores 值都为 -inf），将其替换为 inf，以避免在后续计算中产生 NaN
        lse[lse == float("-inf")] = float("inf")
        # 计算每个块的最大值作为每个头部的最大注意力值
        scores_max_block = torch.stack([torch.amax(s, dim=-1) for s in scores_block], dim=-1)
        # 计算每个块的累积最大值作为每个头部的累积最大值
        cummax_block = torch.cummax(scores_max_block.flip(-1), dim=-1).values.flip(-1).unbind(dim=-1)
        # 按块拆分未归一化的注意力权重
        attn_unnorm_block = attn_unnorm.split(block_size_n, dim=-1)
        # 根据 softmax 函数计算归一化的注意力权重
        attn_norm = torch.cat(
            [
                a * (torch.exp(m - lse)).unsqueeze(-1)
                for a, m in zip(attn_unnorm_block, cummax_block)
            ],
            dim=-1,
        )
        # 如果存在查询掩码，则将注意力权重中相应位置的值替换为 0.0
        if query_padding_mask is not None:
            attn_norm.masked_fill_(~query_padding_mask.view(b, 1, -1, 1), 0.0)
            # 另一种掩码替换方式，根据需要使用
            # attn_norm.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
        # 将注意力权重转换为与未归一化注意力相同的数据类型，并返回结果
        return attn_norm.to(dtype=attn_unnorm.dtype)
    # 构建局部注意力掩码，用于生成局部区域的掩码矩阵
    def construct_local_mask(self, seqlen_q, seqlen_k, window_size, query_padding_mask, key_padding_mask, device):
        # 创建行索引，表示查询序列的位置
        row_idx = torch.arange(seqlen_q, device=device, dtype=torch.long).view(-1, 1)
        # 创建列索引，表示键序列的位置
        col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
        # 计算键序列的有效长度，若没有键序列掩码则为总长度，否则为每个序列的非掩码数量
        sk = (
            seqlen_k
            if key_padding_mask is None
            else key_padding_mask.sum(-1).view(-1, 1, 1, 1)
            # else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        # 计算查询序列的有效长度，若没有查询序列掩码则为总长度，否则为每个序列的非掩码数量
        sq = (
            seqlen_q
            if query_padding_mask is None
            else query_padding_mask.sum(-1).view(-1, 1, 1, 1)
            # else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        # 如果窗口大小的第一个元素小于0，则返回大于窗口右边界的条件
        if window_size[0] < 0:
            return col_idx > row_idx + sk - sq + window_size[1]
        else:
            # 如果没有键序列掩码，则使用全序列长度作为有效长度
            sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
            # 返回在窗口范围内的条件
            return torch.logical_or(
                col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
                col_idx < row_idx + sk - sq - window_size[0],
            )

    # 将闪回注意力矩阵 S 转换为 softmax 格式，用于计算注意力分布
    def convert_flash_attn_S_to_softmax(
        self,
        S,
        seqlen_q,
        seqlen_k,
        query_padding_mask,
        key_padding_mask,
        causal=False,
        window_size=(-1, -1),  # -1 表示无限窗口大小
    ):
    ):
        """FlashAttention stores the S matrix in a different way.
        Arguments:
            S: (batch_size, nheads, seqlen_q, seqlen_k)
            query_padding_mask: (batch_size, seqlen_q)
            key_padding_mask: (batch_size, seqlen_k)
        """
        如果 TEST_WITH_ROCM 为真，直接返回 S 矩阵
        if TEST_WITH_ROCM:
            return S
        获取 batch_size
        b = S.shape[0]

        如果 causal 为真，将 window_size 第二个维度设为 0
        if causal:
            window_size = (window_size[0], 0)
        获取 S 矩阵的 seqlen_q 和 seqlen_k 维度大小
        seqlen_q_rounded, seqlen_k_rounded = S.shape[-2:]
        将 S_converted 初始化为 S 矩阵
        S_converted = S
        如果 window_size 的任一维度大于等于 0
        或
        window_size 的任一维度大于等于 0
        生成本地掩码
        local_mask = self.construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            S.device,
        )
        在 local_mask 上进行填充操作，确保其形状与 S_converted 相同，并用 True 填充
        local_mask = F.pad(
            local_mask,
            (0, seqlen_k_rounded - seqlen_k, 0, seqlen_q_rounded - seqlen_q),
            value=True,
        )
        使用 local_mask 对 S_converted 进行填充操作，将本地掩码为 True 的位置置为 0.0

        需要将不在 attention_mask 中的部分置零，以防 S 初始时包含随机值且部分值未被覆盖。
        获取原始的 seqlen_q 维度大小
        seqlen_q_og = (
            query_padding_mask.shape[-1] if query_padding_mask is not None else seqlen_q_rounded
        )
        如果 query_padding_mask 不为空
        对 query_padding_mask 进行填充操作，使其维度大小与 seqlen_q_rounded 相同
        query_padding_mask = F.pad(query_padding_mask, (0, seqlen_q_rounded - seqlen_q_og))
        使用 query_padding_mask 对 S_converted 进行填充操作，将 query_padding_mask 为 False 的位置置为 0.0

        获取原始的 seqlen_k 维度大小
        seqlen_k_og = key_padding_mask.shape[-1] if key_padding_mask is not None else seqlen_k
        如果 key_padding_mask 不为空
        对 key_padding_mask 进行填充操作，使其维度大小与 seqlen_k_rounded 相同
        key_padding_mask = F.pad(key_padding_mask, (0, seqlen_k_rounded - seqlen_k_og))
        使用 key_padding_mask 对 S_converted 进行填充操作，将 key_padding_mask 为 False 的位置置为 0.0

        最终对 S_converted 进行两次填充操作，确保其 seqlen_q 和 seqlen_k 维度大小与原始大小相同
        S_converted = F.pad(S_converted, (0, 0, 0, seqlen_q_og - seqlen_q_rounded))
        S_converted = F.pad(S_converted, (0, seqlen_k_og - seqlen_k_rounded))
        返回处理后的 S_converted 矩阵，截取有效部分维度返回
        return S_converted[:, :, :seqlen_q, :seqlen_k]

    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    @parametrize("mask_dim", [1, 2, 3, 4])
    # 定义测试方法，用于测试多种注意力掩码变体的内存效率
    def test_mem_efficient_attetntion_mask_variants(self, device, mask_dim: List[int]):
        # 定义数据类型为 torch.float16
        dtype = torch.float16
        # 创建部分函数，用于生成指定设备和数据类型的随机张量，需要梯度
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=True)
        # 定义批量大小、头数和头维度
        batch, num_heads, head_dim = 8, 8, 64
        # 定义查询序列长度和键值对序列长度
        seq_len_q, seq_len_kv = 64, 32
        # 创建查询张量
        query = make_tensor(SdpaShape(batch, num_heads, seq_len_q, head_dim))
        # 定义键值对形状
        kv_shape = SdpaShape(batch, num_heads, seq_len_kv, head_dim)
        # 创建键和值张量
        key, value = make_tensor(kv_shape), make_tensor(kv_shape)

        # 根据不同的掩码维度生成不同形状的随机掩码张量
        if mask_dim == 1:
            mask = torch.randn((seq_len_kv,), device=device, dtype=dtype)
        elif mask_dim == 2:
            mask = torch.randn((seq_len_q, seq_len_kv), device=device, dtype=dtype)
        elif mask_dim == 3:
            mask = torch.randn((num_heads, seq_len_q, seq_len_kv), device=device, dtype=dtype)
        elif mask_dim == 4:
            mask = torch.randn((batch, num_heads, seq_len_q, seq_len_kv), device=device, dtype=dtype)
        
        # 使用有效注意力核心进行计算
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            # 调用 scaled_dot_product_attention 函数进行注意力计算
            out = F.scaled_dot_product_attention(query, key, value, mask)
        # 计算输出张量的和，并进行反向传播
        out.sum().backward()

    # 跳过测试，如果当前平台不支持内存效率注意力机制
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    # 参数化测试，测试内存效率注意力的填充掩码
    @parametrize("dtype", [torch.float, torch.float16])
    def test_mem_eff_attention_pad_mask(self, device, dtype):
        # 创建部分函数，用于生成指定设备和数据类型的随机张量，需要梯度
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=True)
        # 定义批量大小、头数和头维度
        batch, num_heads, head_dim = 8, 8, 64
        # 定义查询序列长度和键值对序列长度
        seq_len_q, seq_len_kv = 64, 15
        # 创建查询张量
        query = make_tensor(SdpaShape(batch, num_heads, seq_len_q, head_dim))
        # 定义键值对形状
        kv_shape = SdpaShape(batch, num_heads, seq_len_kv, head_dim)
        # 创建键和值张量
        key, value = make_tensor(kv_shape), make_tensor(kv_shape)
        # 创建指定形状的随机掩码张量
        mask = torch.randn((batch, num_heads, seq_len_q, seq_len_kv), device=device, dtype=dtype)
        
        # 使用有效注意力核心进行计算
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            # 调用 scaled_dot_product_attention 函数进行注意力计算
            out = F.scaled_dot_product_attention(query, key, value, mask)
        # 计算输出张量的和，并进行反向传播
        out.sum().backward()
    # 定义一个测试函数，用于测试内存有效的注意力机制中的非连续掩码
    def test_mem_eff_attention_non_contiguous_mask(self, device, dtype):
        # 创建一个部分应用函数，用于生成具有指定设备、数据类型和梯度的随机张量
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=True)
        # 定义批量大小、头数和头维度
        batch, num_heads, head_dim = 8, 8, 64
        # 定义查询序列长度和键值对应的序列长度
        seq_len_q, seq_len_kv = 64, 16
        # 生成查询张量
        query = make_tensor(SdpaShape(batch, num_heads, seq_len_q, head_dim))
        # 定义键值对张量的形状
        kv_shape = SdpaShape(batch, num_heads, seq_len_kv, head_dim)
        # 生成键张量和值张量
        key, value = make_tensor(kv_shape), make_tensor(kv_shape)
        # 生成随机掩码张量，并调整其形状以匹配注意力计算要求
        mask = torch.randn((batch, num_heads, seq_len_q, seq_len_kv), device=device, dtype=dtype)
        mask = torch.as_strided(mask, (batch, num_heads, seq_len_q, seq_len_kv), (0, 0, 0, 1))
        # 使用内存有效的注意力机制核心函数，计算注意力值
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            out = F.scaled_dot_product_attention(query, key, value, mask)
        # 对输出进行求和，并进行反向传播
        out.sum().backward()

    # 如果当前平台不支持内存有效的注意力机制，则跳过测试
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    # 参数化测试函数，用于测试长序列掩码在内存有效的注意力机制下的表现
    @parametrize("dtype", [torch.float, torch.float16])
    def test_mem_eff_attention_long_sequence_mask(self, device, dtype):
        # 如果当前 CUDA 设备的总内存小于80GB，则跳过测试
        if torch.cuda.get_device_properties('cuda').total_memory < 80 * 2**30:
            unittest.skip("This test requires substantial GPU memory.")
            return
        # 创建一个部分应用函数，用于生成具有指定设备、数据类型和梯度的随机张量
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=True)
        # 定义批量大小、头数和头维度
        batch, num_heads, head_dim = 1, 32, 64
        # 定义查询序列长度和键值对应的序列长度（较大的值）
        seq_len_q, seq_len_kv = 8192, 8192
        # 生成查询张量
        query = make_tensor(SdpaShape(batch, num_heads, seq_len_q, head_dim))
        # 定义键值对张量的形状
        kv_shape = SdpaShape(batch, num_heads, seq_len_kv, head_dim)
        # 生成键张量和值张量
        key, value = make_tensor(kv_shape), make_tensor(kv_shape)
        # 生成随机掩码张量
        mask = torch.randn((batch, num_heads, seq_len_q, seq_len_kv), device=device, dtype=dtype)
        # 使用内存有效的注意力机制核心函数，计算注意力值
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            out = F.scaled_dot_product_attention(query, key, value, mask)
        # 对输出进行求和，并进行反向传播
        out.sum().backward()

    # 如果当前平台不支持内存有效的注意力机制，则跳过测试
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    def test_mem_eff_attention_non_contig_mask_bug(self, device):
        # Without the fix this produces `AssertionError: assert 0.07352933287620544 < 1e-07`
        # Shapes taken from repro
        # 定义查询张量的大小和步长
        query_size = (3, 16, 1, 128)
        query_strides = (2304, 128, 2048, 1)
        # 定义键张量的大小和步长
        key_size = (3, 16, 14, 128)
        key_strides = (3584, 0, 256, 1)
        # 定义值张量的大小和步长
        value_size = (3, 16, 14, 128)
        value_strides = (3584, 0, 256, 1)
        # 定义注意力掩码张量的大小和步长
        attention_mask_size = (3, 1, 1, 14)
        attn_mask_strides = (14, 14, 14, 1)

        # 计算每个张量所需的元素数
        query_num_elements = max(size * stride for size, stride in zip(query_size, query_strides))
        key_num_elements = max(size * stride for size, stride in zip(key_size, key_strides))
        value_num_elements = max(size * stride for size, stride in zip(value_size, value_strides))
        attention_mask_num_elements = max(size * stride for size, stride in zip(attention_mask_size, attn_mask_strides))

        # 使用指定的大小和步长创建张量
        query = torch.randn(query_num_elements, device=device).as_strided(query_size, query_strides)
        key = torch.randn(key_num_elements, device=device).as_strided(key_size, key_strides)
        value = torch.randn(value_num_elements, device=device).as_strided(value_size, value_strides)
        bias = torch.randn(attention_mask_num_elements, device=device).as_strided(attention_mask_size, attn_mask_strides)

        # 使用 sdpa_kernel 函数进行缩放点产品注意力计算
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            out = F.scaled_dot_product_attention(query, key, value, bias)
            out_contig = F.scaled_dot_product_attention(query, key, value, bias.contiguous())

        # 计算两种方式计算结果之间的最大差异
        max_diff = (out - out_contig).abs().mean()
        self.assertTrue(max_diff.item() < 1e-7)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Fused SDPA was not built for this system")
    def test_singelton_head_dim_stride_ne_1(self, device):
        # 创建查询、键和值张量，并进行形状变换
        query = torch.tensor([[[[1, 2]]]], dtype=torch.float16, device=device)
        query = query.transpose(-1, -2)
        key = torch.tensor([[[[1]]]], dtype=torch.float16, device=device)
        value = torch.tensor([[[[1]]]], dtype=torch.float16, device=device)

        # 使用指定参数调用 torch.backends.cuda.sdp_kernel 函数
        with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=False):
            scaled_dot_product_attention(query, key, value)

    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    @parametrize("type", ["dense", "nested"])
    @parametrize("is_contiguous", [True, False])
    # 定义测试方法，用于测试融合内核的缩放点积注意力函数，支持装置、类型和是否连续参数
    def test_scaled_dot_product_attention_fused_kernels_packed(self, device, type: str, is_contiguous: bool):
        # 如果在 ROCM 平台上测试，并且类型为 'nested'，则跳过测试并显示相应信息
        if TEST_WITH_ROCM and type == 'nested':
            self.skipTest("ROCM does not support efficient attention on nested tensors, for now")
        
        # 创建部分随机生成的张量，使用指定的类型、设备、数据类型（float16）、以及 packed 标志
        make_tensor = partial(rand_sdpa_tensor, type=type, device=device, dtype=torch.float16, packed=True)

        # 定义批次大小、序列长度、注意力头数以及头部维度
        batch_size, seq_len, num_heads, head_dim = 32, 64, 16, 64
        # 创建表示 SDPA 形状的对象
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)

        # 生成 packed 张量
        qkv = make_tensor(shape)
        # 将 qkv 张量分成 query、key、value 三部分
        query, key, value = qkv.chunk(3, dim=-1)

        # 将 query、key、value 张量重新形状以便进行矩阵乘法计算，并转置以匹配乘法操作
        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        # 如果 is_contiguous 为 True，则需要确保 query、key、value 张量是连续的
        if is_contiguous:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        # 使用 EFFICIENT_ATTENTION 内核执行融合缩放点积注意力
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        
        # 使用 MATH 内核执行融合缩放点积注意力的参考结果
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                query.contiguous(), key.contiguous(), value.contiguous(),
                attn_mask=None, dropout_p=0.0, is_causal=False)

        # 断言实际结果与参考结果在指定的容差范围内一致
        self.assertEqual(actual.contiguous(), math_ref.contiguous(), atol=2e-3, rtol=1e-2)
    # 定义一个测试方法，用于测试内存高效梯度与数学实现之间的对比
    def test_sdp_mem_efficient_grad_against_math(self, device, contiguous_inputs: bool, is_causal: bool):
        # 设置测试数据的维度参数
        batch_size, seq_len, num_heads, head_dim = 4, 4, 2, 16
        # 创建生成张量的部分函数，并使用部分参数设定
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device,
                              dtype=torch.float64, requires_grad=True, packed=True)

        # 创建包含查询、键、值的张量并进行处理
        qkv = make_tensor(SdpaShape(batch_size, num_heads, seq_len, head_dim))
        # 克隆并转换为 float32 数据类型的查询、键、值张量
        qkv_lp = qkv.detach().clone().to(torch.float32).requires_grad_()

        # 将查询、键、值张量分成三个部分
        query, key, value = qkv.chunk(3, dim=-1)
        query_lp, key_lp, value_lp = qkv_lp.chunk(3, dim=-1)

        # 将每个部分的张量视图重塑为 batch_size x num_heads x seq_len x head_dim，并转置
        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        # 对低精度的查询、键、值张量进行相同的重塑和转置操作
        query_lp = query_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key_lp = key_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value_lp = value_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        # 如果需要连续输入，则使用 contiguous 方法使查询、键、值张量连续
        if contiguous_inputs:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

            query_lp = query_lp.contiguous()
            key_lp = key_lp.contiguous()
            value_lp = value_lp.contiguous()

        # 使用数学实现的 SDP 内核计算注意力
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            out = torch.nn.functional.scaled_dot_product_attention(query, key, value, None, 0.0, is_causal)

        # 使用高效实现的 SDP 内核计算低精度注意力
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            out_lp = torch.nn.functional.scaled_dot_product_attention(
                query_lp, key_lp, value_lp, None, 0.0, is_causal)

        # 创建与 out 相同形状的随机张量，并将其转换为 float32 数据类型
        rand_upward = torch.rand_like(out)
        rand_upward_lp = rand_upward.to(torch.float32)

        # 分别对高精度和低精度的输出进行反向传播
        out.backward(rand_upward)
        out_lp.backward(rand_upward_lp)

        # 断言高精度和低精度计算的梯度是否一致
        self.assertEqual(qkv.grad, qkv_lp.grad.to(torch.float64), atol=1e-5, rtol=1e-5)

    # 根据平台支持情况跳过测试，如果平台不支持闪电注意力
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Flash Attention was not built for this system")
    @parametrize("contiguous_inputs", [True, False])
    @parametrize("is_causal", [True, False])
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    # 测试函数：对 SDP（Scaled Dot-Product）注意力的梯度与数学库的计算结果进行比较
    def test_sdp_flash_attention_grad_against_math(self, device, contiguous_inputs: bool, is_causal: bool, dtype: torch.dtype):
        # 设置测试数据的维度参数
        batch_size, seq_len, num_heads, head_dim = 4, 4, 2, 16
        # 定义生成张量的辅助函数，并设置张量的类型、设备、需要梯度，以及是否打包
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device,
                              dtype=torch.float64, requires_grad=True, packed=True)

        # 生成原始的 QKV 张量，并进行数据类型转换和梯度追踪
        qkv = make_tensor(SdpaShape(batch_size, num_heads, seq_len, head_dim))
        qkv_lp = qkv.detach().clone().to(dtype).requires_grad_()

        # 将 QKV 张量分解为查询（query）、键（key）和值（value）部分
        query, key, value = qkv.chunk(3, dim=-1)
        query_lp, key_lp, value_lp = qkv_lp.chunk(3, dim=-1)

        # 将每个部分重新组织为适当的形状，并转置以适应注意力机制的需求
        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        query_lp = query_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key_lp = key_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value_lp = value_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        # 如果需要连续的输入，在这里进行张量的连续化处理
        if contiguous_inputs:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

            query_lp = query_lp.contiguous()
            key_lp = key_lp.contiguous()
            value_lp = value_lp.contiguous()

        # 使用数学库进行 SDP 注意力的计算
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            out = torch.nn.functional.scaled_dot_product_attention(query, key, value, None, 0.0, is_causal)

        # 使用 Flash Attention 进行 SDP 注意力的计算
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            out_lp = torch.nn.functional.scaled_dot_product_attention(
                query_lp, key_lp, value_lp, None, 0.0, is_causal)

        # 创建与 out 相同形状的随机张量，并将其转换为指定的数据类型
        rand_upward = torch.rand_like(out)
        rand_upward_lp = rand_upward.to(dtype)

        # 计算 out 的梯度
        out.backward(rand_upward)
        # 计算 out_lp 的梯度
        out_lp.backward(rand_upward_lp)

        # 强制转换梯度为 torch.float64 并比较
        # 由于在 fp16 上进行计算，必须提高容差
        atol = 7e-4 if dtype == torch.float16 else 7e-3
        rtol = 7e-4 if dtype == torch.float16 else 7e-3
        # 如果使用 ROCm，进一步调整容差
        if TEST_WITH_ROCM:
            atol = 9e-4 if dtype == torch.float16 else 9e-3
        # 断言梯度计算结果的一致性
        self.assertEqual(qkv.grad, qkv_lp.grad.to(torch.float64), atol=atol, rtol=rtol)

    # 如果是 ROCm 环境，则跳过测试
    @skipIfRocm  # Missing nested and EFFICIENT_ATTENTION
    # 如果平台不支持融合的 SDPA，则跳过测试
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Platform does not support fused SDPA")
    # 参数化测试类型，包括 dense 和 nested
    @parametrize("type", ["dense", "nested"])
    # 测试融合的自注意力机制选择函数
    def test_fused_sdp_choice(self, device, type: str):
        # 定义张量的维度参数
        batch_size, seq_len, num_heads, head_dim = 2, 128, 8, 64
        # 创建用于生成随机自注意力张量的部分函数
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)
        make_tensor = partial(rand_sdpa_tensor, device=device, dtype=torch.float16, packed=True, requires_grad=True)

        # 生成自注意力张量并分解成查询、键、值
        qkv = make_tensor(shape, type=type)
        query, key, value = qkv.chunk(3, dim=-1)

        # 调整查询、键、值的形状以适应自注意力的操作顺序
        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        # 根据平台支持情况选择适当的自注意力实现，并进行断言验证
        if PLATFORM_SUPPORTS_FLASH_ATTENTION:
            assert torch._fused_sdp_choice(query, key, value) == SDPBackend.FLASH_ATTENTION.value
        else:
            assert torch._fused_sdp_choice(query, key, value) == SDPBackend.EFFICIENT_ATTENTION.value

        # 将数据类型更改为 float32，以确保选择高效的自注意力实现
        make_tensor = partial(rand_sdpa_tensor, device=device, dtype=torch.float32, packed=True)

        # 重新生成自注意力张量并再次分解
        qkv = make_tensor(shape, type=type)
        query, key, value = qkv.chunk(3, dim=-1)

        # 再次调整查询、键、值的形状以适应自注意力的操作顺序
        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        # 断言验证高效的自注意力实现是否被选择
        assert torch._fused_sdp_choice(query, key, value) == SDPBackend.EFFICIENT_ATTENTION.value

    @skipIfRocm  # 缺少 triton.float32（"triton" 前缀用于定位被跳过的单元测试），以及确定性算法
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Platform does not support fused SDPA")
    @parametrize("warn_only", [True, False])
    # 测试包含确定性的自注意力选择函数
    def test_sdp_choice_with_determinism(self, device, warn_only):
        # 定义张量的维度参数
        batch_size, seq_len, num_heads, head_dim = 1, 64, 8, 64
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=torch.float32, packed=False)
        query, key, value = make_tensor(shape), make_tensor(shape), make_tensor(shape)

        # 使用确定性算法上下文管理器设置算法为确定性，并在支持的后端中执行自注意力
        with use_deterministic_algorithims(True, warn_only=warn_only):
            with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                assert torch._fused_sdp_choice(query, key, value) == SDPBackend.EFFICIENT_ATTENTION.value

    @skipIfRocm  # 缺少确定性算法
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Fused SDPA was not built for this system")
    @parametrize("fused_kernel", PLATFORM_SPECIFIC_SDPA)
    @parametrize("warn_only", [True, False])
    # 定义测试方法，用于检验反向传播中融合内核是否会触发非确定性警告
    def test_fused_backwards_throws_determinism_warning(self, device, warn_only, fused_kernel):
        # 设置张量的维度参数
        batch_size, seq_len, num_heads, head_dim = 1, 64, 8, 64
        # 创建 SDPA 形状对象
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)
        # 部分应用随机 SDPA 张量生成函数，指定类型、设备、数据类型、是否打包、是否需要梯度
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=torch.float16, packed=False, requires_grad=True)
        # 生成查询、键、值张量
        query, key, value = make_tensor(shape), make_tensor(shape), make_tensor(shape)

        # 根据融合内核类型选择内核名称
        kernel_name = "Memory Efficient attention" if fused_kernel == SDPBackend.EFFICIENT_ATTENTION else "Flash Attention"
        # 根据 warn_only 参数决定是否检测警告
        warning_context = (
            # 如果 warn_only 为 True，则断言会发出 UserWarning 并匹配特定消息
            self.assertWarnsRegex(
                UserWarning,
                f"{kernel_name} defaults to a non-deterministic algorithm.",
            )
            # 否则提供一个空的上下文
            if warn_only
            else contextlib.nullcontext()
        )
        # 设置使用确定性算法上下文，并根据 warn_only 决定是否发出警告
        with use_deterministic_algorithims(True, warn_only=warn_only):
            # 使用融合内核执行 SDPA 算法
            with sdpa_kernel(backends=[fused_kernel]):
                # 在警告上下文中执行
                with warning_context:
                    # 执行缩放点乘注意力计算，并对结果求和后执行反向传播
                    torch.nn.functional.scaled_dot_product_attention(query, key, value).sum().backward()

    # 标记为跳过此测试，注释说明测试在 CI/CD 环境中表现不确定
    @unittest.skip("This test is not behaving deterministaclly non-deterministaclly on CI/CD")
    # 如果平台不支持融合 SDPA，则跳过测试
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Platform does not support fused SDPA")
    def test_mem_eff_backwards_determinism(self, device):
        # 设置数据类型为 float32
        dtype = torch.float32
        # 定义批大小、序列长度、注意力头数和头部维度
        batch_size, seq_len, n_heads, head_dim = 1, 1024, 8, 64
        # 创建随机查询张量，设备为指定设备，需要梯度计算
        query = torch.rand(batch_size, n_heads, seq_len, head_dim,
                           device=device, dtype=dtype, requires_grad=True)
        # 创建随机键张量，设备为指定设备，需要梯度计算
        key = torch.rand(batch_size, n_heads, seq_len, head_dim, device=device,
                         dtype=dtype, requires_grad=True)
        # 创建随机值张量，设备为指定设备，需要梯度计算
        value = torch.rand(batch_size, n_heads, seq_len, head_dim,
                           device=device, dtype=dtype, requires_grad=True)

        # 使用效率注意力的 SDPA 内核
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            # 运行一次以建立基准
            out = F.scaled_dot_product_attention(query, key, value)
            # 创建与输出相同形状的随机梯度
            upward_grad = torch.rand_like(out)
            # 计算反向传播
            out.backward(upward_grad)
            # 保存初始的查询梯度
            intial_query_grad = query.grad

            # 使用相同的上行梯度再次运行操作，并检查反向传播是否不确定
            diff_anwser_once = False
            for _ in range(100):
                query.grad = None
                out = F.scaled_dot_product_attention(query, key, value)
                out.backward(upward_grad)
                if not torch.equal(intial_query_grad, query.grad):
                    diff_anwser_once = True
                    break
            # 断言检查是否至少有一次反向传播不确定
            self.assertTrue(diff_anwser_once)

        # 使用确定性算法进行测试
        with use_deterministic_algorithims(True, warn_only=False):
            query.grad = None
            out = F.scaled_dot_product_attention(query, key, value)
            upward_grad = torch.rand_like(out)
            out.backward(upward_grad)
            intial_query_grad = query.grad

            # 使用相同的上行梯度再次运行操作，并检查反向传播是否确定
            diff_anwser_once = False
            for _ in range(100):
                query.grad = None
                out = F.scaled_dot_product_attention(query, key, value)
                out.backward(upward_grad)
                if not torch.equal(intial_query_grad, query.grad):
                    diff_anwser_once = True
                    break
            # 断言检查是否所有反向传播都是确定的
            self.assertFalse(diff_anwser_once)

    # 在 H100 上验证通过
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Does not support SDPA")
    @unittest.skipIf(IS_JETSON, "causing sigkill on Jetson")
    @parametrize("batch_size", [1, 8])
    @parametrize("seq_len_q", [4, 8, 64, 128, 256, 512, 1024, 2048] if MEM_EFF_CAPABILITY_MATCHES_SM80
                 else [4, 8, 64, 128, 256, 512])
    @parametrize("seq_len_k", [4, 8, 64, 128, 256, 512, 1024, 2048] if MEM_EFF_CAPABILITY_MATCHES_SM80
                 else [4, 8, 64, 128, 256, 512])
    @parametrize("head_dim", [8, 16, 32, 64, 72, 96, 128] if MEM_EFF_CAPABILITY_MATCHES_SM80
                 else [8, 16, 32, 64])
    @parametrize("is_causal", [False, True])
    # 参数化装饰器，用于测试 is_causal 参数为 False 和 True 时的情况
    
    @parametrize("dropout_p", [0.0, 0.22])
    # 参数化装饰器，用于测试 dropout_p 参数为 0.0 和 0.22 时的情况
    
    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32] if MEM_EFF_CAPABILITY_MATCHES_SM80
                 else [torch.float16, torch.float32])
    # 参数化装饰器，根据 MEM_EFF_CAPABILITY_MATCHES_SM80 条件参数化 dtype 参数，可能包括 torch.float16, torch.bfloat16, torch.float32
    
    @parametrize("scale", [None, "l1"])
    # 参数化装饰器，用于测试 scale 参数为 None 和 "l1" 时的情况
    
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Does not support SDPA")
    # 跳过装饰器，在平台不支持 MEM_EFF_ATTENTION 时跳过测试，原因是不支持 SDPA
    
    @unittest.skipIf(IS_JETSON, "causing sigkill on Jetson")
    # 跳过装饰器，在运行环境为 Jetson 时跳过测试，原因是可能导致 sigkill
    
    @parametrize("batch_size", [1, 8])
    # 参数化装饰器，用于测试 batch_size 参数为 1 和 8 时的情况
    
    @parametrize("seq_len_q", [4, 8, 64, 128, 256, 312, 512, 1024, 2048] if MEM_EFF_CAPABILITY_MATCHES_SM80
                 else [4, 8, 64, 128, 152, 256, 512])
    # 参数化装饰器，根据 MEM_EFF_CAPABILITY_MATCHES_SM80 条件参数化 seq_len_q 参数，可能包括指定的序列长度列表
    
    @parametrize("seq_len_k", [4, 8, 64, 65, 128, 256, 408, 512, 1024, 2048] if MEM_EFF_CAPABILITY_MATCHES_SM80
                 else [4, 8, 37, 64, 128, 256, 512])
    # 参数化装饰器，根据 MEM_EFF_CAPABILITY_MATCHES_SM80 条件参数化 seq_len_k 参数，可能包括指定的序列长度列表
    
    @parametrize("head_dim", [8, 16, 32, 64, 72, 96, 128] if MEM_EFF_CAPABILITY_MATCHES_SM80
                 else [8, 16, 32, 64])
    # 参数化装饰器，根据 MEM_EFF_CAPABILITY_MATCHES_SM80 条件参数化 head_dim 参数，可能包括指定的头维度列表
    
    @parametrize("is_causal", [False])
    # 参数化装饰器，用于测试 is_causal 参数为 False 时的情况
    
    @parametrize("dropout_p", [0.0, 0.22])
    # 参数化装饰器，用于测试 dropout_p 参数为 0.0 和 0.22 时的情况
    
    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32] if MEM_EFF_CAPABILITY_MATCHES_SM80
                 else [torch.float16, torch.float32])
    # 参数化装饰器，根据 MEM_EFF_CAPABILITY_MATCHES_SM80 条件参数化 dtype 参数，可能包括 torch.float16, torch.bfloat16, torch.float32
    
    @parametrize("scale", [None, "l1"])
    # 参数化装饰器，用于测试 scale 参数为 None 和 "l1" 时的情况
    
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support SDPA or pre-SM80 hardware")
    # 跳过装饰器，在平台不支持 FLASH_ATTENTION 时跳过测试，原因是不支持 SDPA 或者不支持 SM80 硬件之前的版本
    
    @unittest.skipIf(IS_JETSON, "causing sigkill on Jetson")
    # 跳过装饰器，在运行环境为 Jetson 时跳过测试，原因是可能导致 sigkill
    
    @parametrize("batch_size", [1, 8])
    # 参数化装饰器，用于测试 batch_size 参数为 1 和 8 时的情况
    
    @parametrize("seq_len_q", [4, 8, 64, 143, 256, 512, 1024, 2048])
    # 参数化装饰器，用于测试 seq_len_q 参数为指定的序列长度列表时的情况
    
    @parametrize("seq_len_k", [4, 8, 64, 128, 256, 587, 1024, 2048])
    # 参数化装饰器，用于测试 seq_len_k 参数为指定的序列长度列表时的情况
    
    @parametrize("head_dim", [8, 16, 21, 32, 64, 72, 96, 128, 160, 192, 203, 256])
    # 参数化装饰器，用于测试 head_dim 参数为指定的头维度列表时的情况
    
    @parametrize("is_causal", [True, False])
    # 参数化装饰器，用于测试 is_causal 参数为 True 和 False 时的情况
    
    @parametrize("dropout_p", [0.0, 0.22, 0.48])
    # 参数化装饰器，用于测试 dropout_p 参数为 0.0, 0.22 和 0.48 时的情况
    
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    # 参数化装饰器，用于测试 dtype 参数为 torch.float16 和 torch.bfloat16 时的情况
    
    @parametrize("scale", [None, "l1"])
    # 参数化装饰器，用于测试 scale 参数为 None 和 "l1" 时的情况
    
    @skipIfRocm  # FIXME: "capturing stream has unjoined work"
    # 跳过装饰器，在运行环境为 ROCm 时跳过测试，原因是存在未加入的工作流
    
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support SDPA or pre-SM80 hardware")
    # 跳过装饰器，在平台不支持 FLASH_ATTENTION 时跳过测试，原因是不支持 SDPA 或者不支持 SM80 硬件之前的版本
    
    @parametrize("batch_size", [1, 8])
    # 参数化装饰器，用于测试 batch_size 参数为 1 和 8 时的情况
    
    @parametrize("seq_len_q", [256, 512, 1024])
    # 参数化装饰器，用于测试 seq_len_q 参数为指定的序列长度列表时的情况
    
    @parametrize("seq_len_k", [256, 512, 1024])
    # 参数化装饰器，用于测试 seq_len_k 参数为指定的序列长度列表时的情况
    
    @parametrize("head_dim", [32, 64])
    # 参数化装饰器，用于测试 head_dim 参数为指定的头维度列表时的情况
    
    @parametrize("is_causal", [True, False])
    # 参数化装饰器，用于测试 is_causal 参数为 True 和 False 时的情况
    
    @parametrize("dropout_p", [0.0, 0.22])
    # 参数化装饰器，用于测试 dropout_p 参数为 0.0 和 0.22 时的情况
    
    @parametrize("dtype", [torch.float16,])
    # 参数化装饰器，用于测试 dtype 参数为 torch.float16 时的情况
    
    @parametrize("scale", [None, "l1"])
    # 参数化装饰器，用于测试 scale 参数为 None 和 "l1" 时的情况
    
    @parametrize("fused_kernel", PLATFORM_SPECIFIC_SDPA)
    # 参数化装饰器，根据平台特定的 SDPA 内核参数化 fused_kernel 参数
    
    @skipIfRocm  # Nested Tensor
    # 跳过装饰器，在运行环境为 ROCm 时跳过测试，原因是存在嵌套张量
    
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Fused SDPA was not built for this system")
    # 跳过装饰器
    # 定义测试方法，用于测试在序列长度为1的输入情况下的融合内核
    def test_fused_kernels_seq_len_1_inputs(self, device, fused_kernel):
        # 定义一个部分应用函数，生成类型为"nested"的随机SDPA张量
        rand_nested_tensor = partial(rand_sdpa_tensor, type="nested", device=device, dtype=torch.float16)
        
        # 设置批次大小、头数和头维度
        batch, num_heads, head_dim = 32, 16, 64
        
        # 生成随机序列长度，范围在1到32之间
        seq_lens = torch.randint(low=1, high=32, size=(batch,))
        
        # 确保一些序列长度为1
        num_ones = 10
        # 随机选择要设为1的序列长度的索引
        indices = torch.randint(low=0, high=batch, size=(num_ones,))
        # 将选定的索引处的序列长度设置为1
        seq_lens.scatter_(0, indices, 1)
        
        # 创建SdpaShape对象，用于描述SDPA的形状，包括批次、头数、序列长度列表和头维度
        shape = SdpaShape(batch, num_heads, seq_lens.tolist(), head_dim)
        
        # 生成随机的query、key和value张量，类型为"nested"
        query = rand_nested_tensor(shape)
        key = rand_nested_tensor(shape)
        value = rand_nested_tensor(shape)
        
        # 转置query、key和value张量的第1和第2维度
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # 使用指定的融合内核计算缩放点产品注意力
        with sdpa_kernel(backends=[fused_kernel]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        
        # 使用数学参考方法计算缩放点产品注意力，此时将张量转换为torch.float32
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                query.contiguous().to(torch.float32),
                key.contiguous().to(torch.float32),
                value.contiguous().to(torch.float32),
                attn_mask=None, dropout_p=0.0, is_causal=False)
        
        # 断言实际计算结果与数学参考计算结果非常接近
        self.assertEqual(actual.contiguous(), math_ref.contiguous().to(torch.float16), atol=1e-3, rtol=1e-2)


    # 在ROCm平台上跳过测试（因为涉及嵌套张量）
    @skipIfRocm  
    # 在不支持融合注意力的平台上跳过测试
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Fused SDPA was not built for this system")
    # 参数化测试，测试不同的内核选项
    @parametrize("kernel", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION] if
                 PLATFORM_SUPPORTS_FLASH_ATTENTION else [SDPBackend.EFFICIENT_ATTENTION])
    # 参数化测试，测试是否扩展query批次
    @parametrize("expand_q_batch", [True, False])
    # 参数化测试，测试是否扩展key批次
    @parametrize("expand_k_batch", [True, False])
    # 参数化测试，测试是否扩展value批次
    @parametrize("expand_v_batch", [True, False])
    # 参数化测试，测试是否扩展query头数
    @parametrize("expand_q_num_heads", [True, False])
    # 参数化测试，测试是否扩展key头数
    @parametrize("expand_k_num_heads", [True, False])
    # 参数化测试，测试是否扩展value头数
    @parametrize("expand_v_num_heads", [True, False])
    # 定义测试方法，测试嵌套广播的融合内核
    def test_fused_kernels_nested_broadcasting(
        self,
        device,
        kernel,
        expand_q_batch,
        expand_k_batch,
        expand_v_batch,
        expand_q_num_heads,
        expand_k_num_heads,
        expand_v_num_heads,
    # 在ROCm平台上跳过测试（因为涉及嵌套张量）
    @skipIfRocm  
    # 在不支持内存效率融合注意力的平台上跳过测试
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    # 测试使用嵌套张量进行的核心融合操作，涉及查询密集操作
    def test_fused_kernels_nested_broadcasting_query_dense(self, device):
        # 定义一个部分应用的函数，生成类型为"nested"的随机自注意力张量
        rand_nested_tensor = partial(rand_sdpa_tensor, type="nested", device=device, dtype=torch.float32)
        
        # 定义批次大小、头数、头维度和值维度
        batch, num_heads, head_dim, head_dim_v = 32, 16, 64, 96
        
        # 随机生成序列长度列表，长度为batch个随机整数
        seq_lens = torch.randint(low=1, high=32, size=(batch,)).tolist()
        
        # 定义查询张量的形状
        q_shape = (1, 1, num_heads, head_dim)
        
        # 定义键和值的形状，使用SdpaShape类来定义
        k_shape = SdpaShape(batch, num_heads, seq_lens, head_dim)
        v_shape = SdpaShape(batch, 1, seq_lens, head_dim_v)

        # 创建一个密集的查询张量
        query = torch.randn(q_shape, device=device, dtype=torch.float32)
        
        # 生成随机嵌套张量作为键和值
        key = rand_nested_tensor(k_shape)
        value = rand_nested_tensor(v_shape)

        # 将查询张量从 (1, 1, num_heads, head_dim) 转变为 (batch, 1, num_heads, head_dim)
        query_expanded = torch.nested.nested_tensor([query.squeeze(0) for _ in range(batch)]).transpose(1, 2)
        
        # 将值张量从 (batch, seq_lens, 1, head_dim) 转变为 (batch, seq_lens, num_heads, head_dim)
        value_expanded = torch.nested.nested_tensor(
            [t.expand(-1, num_heads, head_dim_v) for t in value.unbind()]).transpose(1, 2)

        # 调整查询、键和值张量的维度顺序
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # 使用 sdpa_kernel 上下文管理器，使用 EFFICIENT_ATTENTION 后端进行注意力计算
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        
        # 使用 sdpa_kernel 上下文管理器，使用 MATH 后端进行注意力计算
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            # 作为数学参考，使用扩展后的查询、键和值进行注意力计算
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                query_expanded.contiguous(), key.contiguous(), value_expanded.contiguous(),
                attn_mask=None, dropout_p=0.0, is_causal=False)

        # 断言实际计算结果与数学参考结果的近似性，使用给定的绝对误差和相对误差容差
        self.assertEqual(actual.contiguous(), math_ref.contiguous(), atol=1e-3, rtol=1e-2)
class TestAttnBias(NNTestCase):
    
    # 定义测试函数，测试注意力偏置功能
    def run_test(
        self,
        device,
        make_q,
        make_kv,
        attn_bias=None,
        forw_tolerances: Optional[Tolerances] = None,
        grad_tolerances: Optional[Tolerances] = None,
        backend=None,
    ):
        # 如果指定了后端，则重置 torch._dynamo
        if backend is not None:
            torch._dynamo.reset()

        # 生成 query、key、value
        query, key, value = make_q(), make_kv(), make_kv()
        # 复制 query、key、value 作为原型
        query_prototype, key_prototype, value_prototype = query_key_value_clones(query, key, value)

        # 若存在 attn_bias，则实例化其在指定设备上的版本
        realized = attn_bias._materialize(device) if attn_bias is not None else None
        # 使用 scaled_dot_product_attention 进行注意力计算，不使用 dropout，不考虑因果关系
        pytorch_output = scaled_dot_product_attention(
            query, key, value, attn_mask=realized, dropout_p=0.0, is_causal=False
        )

        # 根据指定的 backend 编译 scaled_dot_product_attention 函数或直接使用
        sdpa_op = (
            torch.compile(scaled_dot_product_attention, backend=backend)
            if backend is not None
            else scaled_dot_product_attention
        )
        # 使用 sdpa_op 进行注意力计算，传入原型 query、key、value，使用 attn_bias，不使用 dropout，不考虑因果关系
        sdpa_output = sdpa_op(
            query_prototype,
            key_prototype,
            value_prototype,
            attn_mask=attn_bias,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
        )

        # 生成与 pytorch_output 同样形状的随机张量 dOut
        dOut = torch.randn_like(pytorch_output)
        # 计算 pytorch_output 的梯度
        pytorch_output.backward(dOut)
        # 计算 sdpa_output 的梯度
        sdpa_output.backward(dOut)

        # 若未指定 forw_tolerances，默认使用 Tolerances(atol=None, rtol=None)
        if forw_tolerances is None:
            forw_tolerances = Tolerances(atol=None, rtol=None)
        # 若未指定 grad_tolerances，默认使用 Tolerances(atol=None, rtol=None)
        if grad_tolerances is None:
            grad_tolerances = Tolerances(atol=None, rtol=None)

        # 使用 torch.testing.assert_close 检查 pytorch_output 与 sdpa_output 的值接近程度
        torch.testing.assert_close(pytorch_output, sdpa_output, rtol=forw_tolerances.rtol, atol=forw_tolerances.atol)
        # 使用 torch.testing.assert_close 检查 query 的梯度与 query_prototype 的梯度的值接近程度
        torch.testing.assert_close(query.grad, query_prototype.grad, rtol=grad_tolerances.rtol, atol=grad_tolerances.atol)
        # 使用 torch.testing.assert_close 检查 key 的梯度与 key_prototype 的梯度的值接近程度
        torch.testing.assert_close(key.grad, key_prototype.grad, rtol=grad_tolerances.rtol, atol=grad_tolerances.atol)
        # 使用 torch.testing.assert_close 检查 value 的梯度与 value_prototype 的梯度的值接近程度
        torch.testing.assert_close(value.grad, value_prototype.grad, rtol=grad_tolerances.rtol, atol=grad_tolerances.atol)

    # 跳过在 Rocm 平台上的测试，因为目前不支持第二个变体
    @skipIfRocm
    # 参数化 causal_variant，包括 CausalVariant.UPPER_LEFT 和 CausalVariant.LOWER_RIGHT 两种变体
    @parametrize("causal_variant", [CausalVariant.UPPER_LEFT, CausalVariant.LOWER_RIGHT])
    # 参数化 shape，包括多个元组形状的变体
    @parametrize(
        "shape",
        [(16, 16, 128, 128, 16), (16, 16, 128, 256, 32), (16, 16, 256, 128, 32), (1, 1, 23, 56, 15)],
    )
    # 定义测试函数，用于验证不同因果变体在给定设备上的行为
    def test_causal_variants(self, device, causal_variant: CausalVariant, shape: List[Tuple[int]]):
        # 创建一个部分应用了参数的随机张量生成函数，设备为指定设备，数据类型为半精度浮点数，需要梯度计算
        make_tensor = partial(
            torch.rand, device=device, dtype=torch.float16, requires_grad=True
        )

        # 解构形状参数，分别表示批大小、头数、查询序列长度、键值序列长度、头维度
        bsz, num_heads, seq_len_q, seq_len_kv, head_dim = shape
        # 创建部分应用了形状参数的查询张量生成函数
        make_q_tensor = partial(make_tensor, SdpaShape(bsz, num_heads, seq_len_q, head_dim))
        # 创建部分应用了形状参数的键值张量生成函数
        make_kv_tensor = partial(make_tensor, SdpaShape(bsz, num_heads, seq_len_kv, head_dim))
        
        # 如果因果变体为 LOWER_RIGHT 并且查询序列长度大于键值序列长度，跳过测试
        if causal_variant == CausalVariant.LOWER_RIGHT and seq_len_q > seq_len_kv:
            self.skipTest(
                "Lower right causal mask will produce NaNs in the output when seq_len_q > seq_len_kv!"
            )

        # 前向传播容忍度
        forw_tol = Tolerances(1e-3, 1e-3)
        # 梯度容忍度
        grad_tol = Tolerances(5e-3, 5e-3)

        # 根据因果变体选择注意力偏置张量
        if causal_variant == CausalVariant.UPPER_LEFT:
            attn_bias = causal_upper_left(seq_len_q, seq_len_kv)
        else:
            attn_bias = causal_lower_right(seq_len_q, seq_len_kv)

        # 运行测试函数，验证模型在给定参数下的前向传播和梯度计算
        self.run_test(device, make_q_tensor, make_kv_tensor, attn_bias, forw_tol, grad_tol, backend=None)

    # 标记为不适用于 ROCm 环境的测试函数装饰器
    @skipIfRocm  # CausalVariant
    # 参数化测试，测试不同的因果变体
    @parametrize("causal_variant", [CausalVariant.UPPER_LEFT, CausalVariant.LOWER_RIGHT])
    # 参数化测试，测试不同的形状参数组合
    @parametrize(
        "shape",
        [(16, 16, 128, 128, 16), (16, 16, 128, 256, 32), (16, 16, 256, 128, 32), (1, 1, 23, 56, 15)],
    )
    # 在 Windows 环境下跳过测试，因为不支持 torch.compile
    @unittest.skipIf(IS_WINDOWS, "torch.compile is not supported on windows")
    # 标记为不适用于 Torch Dynamo 环境的测试函数装饰器
    @skipIfTorchDynamo("This function already calls torch.compile.")
    # 编译因果变体测试函数
    def test_causal_variants_compile(self, device, causal_variant: CausalVariant, shape: List[Tuple[int]]):
        # 创建一个计数器对象，用于跟踪编译调用次数，指定后端为 "aot_eager"
        cnts = CompileCounterWithBackend("aot_eager")
        # 创建一个部分应用了参数的随机张量生成函数，设备为指定设备，数据类型为半精度浮点数，需要梯度计算
        make_tensor = partial(
            torch.rand, device=device, dtype=torch.float16, requires_grad=True
        )

        # 解构形状参数，分别表示批大小、头数、查询序列长度、键值序列长度、头维度
        bsz, num_heads, seq_len_q, seq_len_kv, head_dim = shape
        # 创建部分应用了形状参数的查询张量生成函数
        make_q_tensor = partial(make_tensor, SdpaShape(bsz, num_heads, seq_len_q, head_dim))
        # 创建部分应用了形状参数的键值张量生成函数
        make_kv_tensor = partial(make_tensor, SdpaShape(bsz, num_heads, seq_len_kv, head_dim))
        
        # 如果因果变体为 LOWER_RIGHT 并且查询序列长度大于键值序列长度，跳过测试
        if causal_variant == CausalVariant.LOWER_RIGHT and seq_len_q > seq_len_kv:
            self.skipTest(
                "Lower right causal mask will produce NaNs in the output when seq_len_q > seq_len_kv!"
            )

        # 前向传播容忍度
        forw_tol = Tolerances(1e-3, 1e-3)
        # 梯度容忍度
        grad_tol = Tolerances(5e-3, 5e-3)

        # 根据因果变体选择注意力偏置张量
        if causal_variant == CausalVariant.UPPER_LEFT:
            attn_bias = causal_upper_left(seq_len_q, seq_len_kv)
        else:
            attn_bias = causal_lower_right(seq_len_q, seq_len_kv)

        # 运行测试函数，验证模型在给定参数下的前向传播和梯度计算，使用计数器后端
        self.run_test(device, make_q_tensor, make_kv_tensor, attn_bias, forw_tol, grad_tol, backend=cnts)
        # 断言编译的图应该只有 1 个框架
        self.assertEqual(cnts.frame_count, 1, "Compiled graph should have 1 frame!")

    # 参数化测试，测试不同的形状参数组合
    @parametrize("shape", [(16, 16, 128, 128, 16), (16, 16, 128, 256, 32), (16, 16, 256, 128, 32), (1, 1, 23, 56, 15)])
    # 定义一个测试方法，用于验证是否调用函数正确的设备和形状参数
    def test_is_causal_equals_upper_left(self, device, shape: List[Tuple[int]]):
        # 创建一个部分应用函数，用于生成指定设备上的随机浮点张量，支持梯度计算
        make_tensor = partial(
            torch.rand, device=device, dtype=torch.float16, requires_grad=True
        )

        # 解构形状参数
        bsz, num_heads, seq_len_q, seq_len_kv, head_dim = shape

        # 创建部分应用函数，生成指定形状的查询张量和键值对张量
        make_q_tensor = partial(make_tensor, SdpaShape(bsz, num_heads, seq_len_q, head_dim))
        make_kv_tensor = partial(make_tensor, SdpaShape(bsz, num_heads, seq_len_kv, head_dim))

        # 定义前向传播容差
        forw_tol = Tolerances(1e-3, 1e-3)
        # 定义梯度容差
        grad_tol = Tolerances(5e-3, 5e-3)

        # 生成查询、键、值张量
        query = make_q_tensor()
        key = make_kv_tensor()
        value = make_kv_tensor()

        # 创建因果上三角形式的注意力偏置张量
        attn_bias = causal_upper_left(seq_len_q, seq_len_kv)

        # 调用注意力计算函数，使用上述偏置进行加权计算
        out_attn_bias = scaled_dot_product_attention(query, key, value, attn_mask=attn_bias, dropout_p=0.0)
        # 调用注意力计算函数，使用因果性参数进行加权计算
        out_is_causal = scaled_dot_product_attention(query, key, value, is_causal=True, dropout_p=0.0)

        # 断言两种计算结果相等，根据指定容差
        torch.testing.assert_close(out_attn_bias, out_is_causal, rtol=forw_tol.rtol, atol=forw_tol.atol)

    # 定义测试方法，验证因果性和掩码同时使用是否引发异常
    def test_is_causal_and_mask_fails(self, device):
        # 创建一个部分应用函数，用于生成指定设备上的随机浮点张量，支持梯度计算
        make_tensor = partial(
            torch.rand, device=device, dtype=torch.float16, requires_grad=True
        )
        
        # 创建部分应用函数，生成指定形状的查询张量和键值对张量
        make_q_tensor = partial(make_tensor, SdpaShape(16, 16, 128, 16))
        make_kv_tensor = partial(make_tensor, SdpaShape(16, 16, 128, 16))

        # 生成查询、键、值张量
        query = make_q_tensor()
        key = make_kv_tensor()
        value = make_kv_tensor()

        # 创建因果上三角形式的注意力偏置张量
        attn_bias = causal_upper_left(128, 128)

        # 使用断言检查函数是否会引发预期的异常消息
        with self.assertRaisesRegex(ValueError, "CausalBias should not be used with causal=True"):
            scaled_dot_product_attention(query, key, value, attn_mask=attn_bias, is_causal=True, dropout_p=0.0)
class TestSDPAPrivateUse1Only(NNTestCase):
    # NNTestCase 的子类，用于测试特定的 SDPA 私有使用1场景

    @classmethod
    def setUpClass(cls):
        # 在整个测试类开始前执行的准备工作

        # 清除构建路径
        remove_build_path()

        # 加载自定义设备扩展模块 custom_device_extension
        cls.module = torch.utils.cpp_extension.load(
            name="custom_device_extension",
            sources=[
                "cpp_extensions/open_registration_extension.cpp",
            ],
            extra_include_paths=["cpp_extensions"],
            extra_cflags=["-g"],
            verbose=True,
        )

        # 注册 torch.foo 模块和 foo 设备到 torch
        torch.utils.rename_privateuse1_backend("foo")

        # 为私有使用1场景生成方法
        torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)

        # 注册假设的 foo 设备模块
        torch._register_device_module("foo", generate_faked_module())

    @skipIfTorchDynamo()
    def test_fused_sdp_choice_privateuseone(self):
        # 测试融合的 SDP 选择私有使用1场景

        # 定义张量的形状和类型
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = partial(torch.rand, device="cpu", dtype=torch.float16)
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)
        
        # 生成随机张量 q_cpu, k_cpu, v_cpu
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        
        # 将张量转移到私有使用1场景下
        q_privateuse1 = q_cpu.to("foo")
        k_privateuse1 = k_cpu.to("foo")
        v_privateuse1 = v_cpu.to("foo")
        
        # 断言私有使用1场景下的融合 SDP 选择函数的输出
        assert torch._fused_sdp_choice(q_privateuse1, k_privateuse1, v_privateuse1) == SDPBackend.OVERRIDEABLE.value

    def test_scaled_dot_product_fused_attention_overrideable(self):
        # 测试可覆盖的缩放点积融合注意力机制

        # 定义张量的形状和类型
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = partial(torch.rand, device="cpu", dtype=torch.float16)
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)
        
        # 生成随机张量 q_cpu, k_cpu, v_cpu
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        
        # 将张量转移到私有使用1场景下
        q_privateuse1 = q_cpu.to("foo")
        k_privateuse1 = k_cpu.to("foo")
        v_privateuse1 = v_cpu.to("foo")
        
        # 调用 torch.nn.functional 中的缩放点积注意力机制函数
        actual = torch.nn.functional.scaled_dot_product_attention(
            q_privateuse1, k_privateuse1, v_privateuse1, attn_mask=None, dropout_p=0.0)
    def test_scaled_dot_product_fused_attention_overrideable_backward(self):
        # 定义测试方法：测试可覆盖后向传播的缩放点积融合注意力机制

        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        # 设置批量大小、序列长度、注意头数和每个头维度

        make_tensor = partial(torch.rand, device="cpu", dtype=torch.float16, requires_grad=True)
        # 创建一个部分函数，用于生成具有指定设备、数据类型和梯度需求的随机张量
        shape = (batch_size, num_heads, seq_len, head_dim)
        # 定义张量形状

        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        # 生成查询、键、值张量并分配给 CPU 设备
        attn_mask = make_tensor((batch_size, num_heads, seq_len, seq_len))
        # 生成注意力掩码张量

        q_privateuse1 = q_cpu.to("foo")
        k_privateuse1 = k_cpu.to("foo")
        v_privateuse1 = v_cpu.to("foo")
        attn_mask_privateuse1 = attn_mask.to("foo")
        # 将查询、键、值和注意力掩码张量分配给名为 "foo" 的私有用途设备

        output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask = \
            torch.ops.aten._scaled_dot_product_fused_attention_overrideable(
                q_privateuse1, k_privateuse1, v_privateuse1, attn_bias=attn_mask_privateuse1)
        # 调用底层 C++ 实现的缩放点积融合注意力机制函数，计算输出及相关中间值

        rand_upward = torch.rand(shape, device="cpu", dtype=torch.float16, requires_grad=False)
        rand_upward_privateuse1 = rand_upward.to("foo")
        # 生成随机梯度张量并将其分配给名为 "foo" 的私有用途设备

        grad_input_mask = [True, True, True, True]
        # 定义输入梯度掩码，指定需要计算梯度的张量

        grad_q, grad_k, grad_v, grad_attn_mask = torch.ops.aten._scaled_dot_product_fused_attention_overrideable_backward(
            rand_upward_privateuse1, q_privateuse1, k_privateuse1, v_privateuse1, attn_mask_privateuse1,
            grad_input_mask, output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, dropout_p=0.0,
            is_causal=False, philox_seed=philox_seed, philox_offset=philox_offset)
        # 调用底层 C++ 实现的缩放点积融合注意力机制的后向传播函数，计算梯度
# 如果 NOTEST_CPU 为真，则设定设备类型为 ("cuda", )
if NOTEST_CPU:
    device_types = ("cuda", )
# 否则，设定设备类型为 ("cpu", "cuda")
else:
    device_types = ("cpu", "cuda")

# 实例化 TestTransformers 类的设备类型相关测试，并将其注册到全局命名空间中，仅适用于指定的设备类型
instantiate_device_type_tests(TestTransformers, globals(), only_for=device_types)

# 实例化 TestSDPAFailureModes 类的设备类型相关测试，并将其注册到全局命名空间中，仅适用于指定的设备类型
instantiate_device_type_tests(TestSDPAFailureModes, globals(), only_for=device_types)

# 实例化 TestSDPA 类的设备类型相关测试，并将其注册到全局命名空间中，仅适用于指定的设备类型
instantiate_device_type_tests(TestSDPA, globals(), only_for=device_types)

# 实例化 TestSDPACudaOnly 类的设备类型相关测试，并将其注册到全局命名空间中，仅适用于 "cuda" 设备类型
instantiate_device_type_tests(TestSDPACudaOnly, globals(), only_for=("cuda"))

# 实例化 TestAttnBias 类的设备类型相关测试，并将其注册到全局命名空间中，仅适用于指定的设备类型
instantiate_device_type_tests(TestAttnBias, globals(), only_for=device_types)

# 如果当前脚本被直接执行（而非被导入为模块），则执行测试运行
if __name__ == '__main__':
    run_tests()
```