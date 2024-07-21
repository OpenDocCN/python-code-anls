# `.\pytorch\test\inductor\test_flex_attention.py`

```py
# Owner(s): ["module: inductor"]
# flake8: noqa: B950

# 导入必要的模块和类
import functools  # 导入 functools 模块，用于创建偏函数
from collections import namedtuple  # 导入 namedtuple 类，用于创建命名元组
from typing import Callable, Optional  # 导入类型提示

from unittest import expectedFailure, skip, skipUnless  # 导入单元测试相关函数和装饰器
from unittest.mock import patch  # 导入 mock 模块中的 patch 函数

import torch  # 导入 PyTorch 模块

# 导入 PyTorch 内部测试和辅助模块
from torch._dynamo.testing import CompileCounterWithBackend, normalize_gm
from torch._higher_order_ops.flex_attention import flex_attention as flex_attention_hop
from torch._inductor import metrics  # 导入 Inductor 模块中的 metrics
from torch._inductor.test_case import TestCase as InductorTestCase  # 导入 Inductor 模块中的 TestCase 类
from torch._inductor.utils import run_and_get_code  # 导入 Inductor 模块中的 run_and_get_code 函数
from torch.nn.attention._flex_attention import (
    _causal,
    _compose,
    _create_block_sparse_mask,
    _create_empty_block_sparse_mask,
    _flex_attention,
    _generate_alibi_bias,
    _identity,
    _rel_bias,
    _rel_causal,
)  # 导入灵活注意力机制相关函数

from torch.testing import FileCheck  # 导入文件检查模块
from torch.testing._internal import common_utils  # 导入内部测试的通用工具函数
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_BF16  # 导入 CUDA 相关支持
from torch.utils._triton import has_triton  # 导入 Triton 相关函数

# 如果 Triton 不可用，则跳过测试
supported_platform = skipUnless(
    torch.cuda.is_available()  # CUDA 可用
    and torch.version.hip is None  # 不是 HIP 环境
    and has_triton()  # Triton 可用
    and torch.cuda.get_device_capability() >= (8, 0),  # CUDA 设备支持的版本大于等于 8.0
    "Requires CUDA and Triton",
)

# 定义容差元组，用于测试的数值容差
Tolerances = namedtuple("Tolerances", ["atol", "rtol"])
torch.set_float32_matmul_precision("high")

# 使用 torch.ops.aten.index 简化索引操作
index = torch.ops.aten.index


# 创建注意力机制的函数，返回一个偏函数
def create_attention(score_mod, block_sparse_mask):
    return functools.partial(
        _flex_attention, score_mod=score_mod, block_sparse_mask=block_sparse_mask
    )


# 根据 score_mod 创建块稀疏掩码的函数
def create_block_sparse_mask_from_score_mod(score_mod, query, key, value):
    Q_LEN = query.shape[-2]
    KV_LEN = key.shape[-2]
    if score_mod == _causal:
        # 如果 score_mod 是 _causal，则创建块稀疏掩码
        return _create_block_sparse_mask(
            torch.tril(
                torch.ones(Q_LEN, KV_LEN, dtype=torch.bool, device=query.device)
            ),
            128,
            128,
        )
    else:
        return None


# 定义测试用的数据类型列表，根据平台是否支持 BF16 来决定是否包含 torch.bfloat16
test_dtypes = (
    [torch.float16, torch.bfloat16, torch.float32]
    if PLATFORM_SUPPORTS_BF16
    else [torch.float16, torch.float32]
)

# 快速测试的数据类型列表，仅包含 torch.float16
test_dtypes_fast = [torch.float16]


# --------- 用于测试的有用 score_mod 函数 ---------
# 将 score 中的不满足条件的位置置为 -inf 的函数
def _inverse_causal(score, b, h, m, n):
    return torch.where(m <= n, score, float("-inf"))


# 将 score 值乘以 2 的函数
def _times_two(score, b, h, m, n):
    """Joint graph needed for correctness"""
    return score * 2


# 将 score 值平方的函数
def _squared(score, b, h, m, n):
    """Joint graph needed for correctness"""
    return score * score


# 创建头偏移的函数，返回一个 score_mod 函数
def _head_offset(dtype: torch.dtype):
    """Captured Buffer"""
    head_offset = torch.rand(H, device="cuda", dtype=dtype)

    def score_mod(score, b, h, m, n):
        return score * head_offset[h]

    return score_mod


# 对 score 进行三角函数计算的函数
def _trig(score, b, h, m, n):
    """Joint graph needed for correctness"""
    return torch.sin(torch.cos(score)) + torch.tan(b)


# 用于测试的另一个三角函数计算的函数
def _trig2(score, b, h, m, n):
    """Branching joint graph"""
    cos_score = torch.cos(score)
    sin_score = torch.sin(score)
    # 计算余弦分数和正弦分数的乘积加上正切函数的结果
    z = cos_score * sin_score + torch.tan(b)
    # 返回计算结果 z
    return z
# 定义一个包含多个测试分数修改函数的列表
test_score_mods = [
    _identity,              # 函数：恒等函数
    _times_two,             # 函数：乘以二
    _squared,               # 函数：平方
    _causal,                # 函数：因果相关
    _inverse_causal,        # 函数：反因果相关
    _rel_bias,              # 函数：相关偏差
    _rel_causal,            # 函数：相关因果
    _generate_alibi_bias(8),# 函数：生成假设偏差（参数为8）
]

# 定义一个捕获缓冲区映射的字典
captured_buffers_map = {
    "_head_offset": _head_offset,   # 键为"_head_offset"，值为_head_offset函数
}

# 定义常量 B, H, S, D 分别代表 4, 8, 2048, 64
B = 4   # 批处理大小
H = 8   # 头数
S = 2048    # 序列长度
D = 64      # 维度

def query_key_value_clones(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dtype: torch.dtype = None,
):
    """克隆查询、键和值张量，并将它们移动到指定的数据类型。"""
    if dtype is None:
        dtype = query.dtype
    query_ref = query.clone().detach().to(dtype).requires_grad_(query.requires_grad)
    key_ref = key.clone().detach().to(dtype).requires_grad_(key.requires_grad)
    value_ref = value.clone().detach().to(dtype).requires_grad_(value.requires_grad)
    return query_ref, key_ref, value_ref

class TestFlexAttention(InductorTestCase):
    def _check_equal(
        self,
        golden_out: torch.Tensor,
        ref_out: torch.Tensor,
        compiled_out: torch.Tensor,
        fudge_factor: float,
        tensor_name: Optional[str] = None,
    ):
        """检查编译输出与参考输出的误差是否在可接受范围内。"""
        compiled_error = (golden_out - compiled_out).abs().mean()
        ref_error = (golden_out - ref_out).abs().mean()
        if torch.isnan(compiled_error).any() and not torch.isnan(ref_error).any():
            self.assertTrue(False, "Output/Grad with NaN")
        if compiled_error > ref_error * fudge_factor:
            name = tensor_name if tensor_name is not None else ""
            msg = f"{name} 编译错误 {compiled_error} 超过参考错误 {ref_error} 的 {fudge_factor} 倍."
            self.assertTrue(False, msg)

    def _check_out_and_grad(
        self,
        golden_out: torch.Tensor,
        ref_out: torch.Tensor,
        compiled_out: torch.Tensor,
        q_gold: torch.Tensor,
        q_ref: torch.Tensor,
        q: torch.Tensor,
        k_gold: torch.Tensor,
        k_ref: torch.Tensor,
        k: torch.Tensor,
        v_gold: torch.Tensor,
        v_ref: torch.Tensor,
        v: torch.Tensor,
    ):
        # 略
    ):
        # 获取参考输出的数据类型
        dtype = ref_out.dtype
        # 在没有梯度的情况下进行操作
        with torch.no_grad():
            # 注意，由于在线softmax的存在，我们的精度似乎比float32计算要低
            if dtype == torch.float32:
                fudge_factor = 10.0
            else:
                fudge_factor = 1.1

            # 检查输出
            self._check_equal(golden_out, ref_out, compiled_out, fudge_factor, "Out")

            # 检查梯度
            q_fudge_factor = 2.5 * fudge_factor
            self._check_equal(
                q_gold.grad, q_ref.grad, q.grad, q_fudge_factor, "Grad_Query"
            )
            k_fudge_factor = 4 * fudge_factor
            self._check_equal(
                k_gold.grad, k_ref.grad, k.grad, k_fudge_factor, "Grad_Key"
            )
            v_fudge_factor = 4 * fudge_factor
            self._check_equal(
                v_gold.grad, v_ref.grad, v.grad, v_fudge_factor, "Grad_Value"
            )

    def run_test(
        self,
        score_mod: Callable,
        dtype: torch.dtype = torch.float16,
        Q_B: int = B,
        Q_H: int = H,
        Q_S: int = S,
        Q_D: int = D,
        KV_B: int = B,
        KV_H: int = H,
        KV_S: int = S,
        KV_D: int = D,
    ):
        # 在GPU上创建具有梯度的随机张量
        q = torch.randn(
            (Q_B, Q_H, Q_S, Q_D), dtype=dtype, device="cuda", requires_grad=True
        )
        k = torch.randn(
            (KV_B, KV_H, KV_S, KV_D), dtype=dtype, device="cuda", requires_grad=True
        )
        v = torch.randn(
            (KV_B, KV_H, KV_S, KV_D), dtype=dtype, device="cuda", requires_grad=True
        )
        # 复制查询、键和值张量用于引用
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        # 使用torch.float64复制查询、键和值张量用于黄金（高精度）参考
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)
        # 创建块稀疏掩码
        block_sparse_mask = create_block_sparse_mask_from_score_mod(score_mod, q, k, v)
        # 创建部分自注意力
        sdpa_partial = create_attention(score_mod, block_sparse_mask)
        # 编译自注意力
        compiled_sdpa = torch.compile(sdpa_partial)
        # 计算黄金（高精度）输出
        golden_out = sdpa_partial(q_gold, k_gold, v_gold)
        # 计算参考输出
        ref_out = sdpa_partial(q_ref, k_ref, v_ref)
        # 计算编译后输出
        compiled_out = compiled_sdpa(q, k, v)

        # 创建后向传播梯度张量
        backward_grad = torch.randn((Q_B, Q_H, Q_S, Q_D), dtype=dtype, device="cuda")

        # 计算黄金（高精度）输出的反向传播
        golden_out.backward(backward_grad.to(torch.float64))
        # 计算参考输出的反向传播
        ref_out.backward(backward_grad)
        # 计算编译后输出的反向传播
        compiled_out.backward(backward_grad)

        # 检查黄金（高精度）输出和梯度
        self._check_out_and_grad(
            golden_out,
            ref_out,
            compiled_out,
            q_gold,
            q_ref,
            q,
            k_gold,
            k_ref,
            k,
            v_gold,
            v_ref,
            v,
        )

    def run_dynamic_test(
        self,
        score_mod: Callable,
        dtype: torch.dtype = torch.float16,
        B: int = B,
        H: int = H,
        S: int = S,
        D: int = D,
        ):
            sdpa_partial = create_attention(score_mod)
            # 创建注意力函数，使用给定的分数模块

            # 第一个急切批次，形状为 (B, H, S, D)
            q1 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
            k1 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
            v1 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
            # 克隆查询、键、值张量
            q1_ref, k1_ref, v1_ref = query_key_value_clones(q1, k1, v1)
            q1_gold, k1_gold, v1_gold = query_key_value_clones(q1, k1, v1, torch.float64)
            # 使用部分注意力函数计算输出
            ref_out1 = sdpa_partial(q1_ref, k1_ref, v1_ref)
            golden_out1 = sdpa_partial(q1_gold, k1_gold, v1_gold)

            backward_grad1 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")

            # 对 golden_out1 和 ref_out1 进行反向传播
            golden_out1.backward(backward_grad1.to(torch.float64))
            ref_out1.backward(backward_grad1)

            # 第二个急切批次，形状为 (B * 2, H, S / 2, D)
            B = int(B * 2)
            S = int(S / 2)
            q2 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
            k2 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
            v2 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
            # 克隆查询、键、值张量
            q2_ref, k2_ref, v2_ref = query_key_value_clones(q2, k2, v2)
            q2_gold, k2_gold, v2_gold = query_key_value_clones(q2, k2, v2, torch.float64)
            # 使用部分注意力函数计算输出
            ref_out2 = sdpa_partial(q2_ref, k2_ref, v2_ref)
            golden_out2 = sdpa_partial(q2_gold, k2_gold, v2_gold)

            backward_grad2 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")

            # 对 golden_out2 和 ref_out2 进行反向传播
            golden_out2.backward(backward_grad2.to(torch.float64))
            ref_out2.backward(backward_grad2)

            # 需要清除 Dynamo 计数器，因为灵活注意力急切模式也使用 Dynamo 追踪。
            # 我们检查 dynamo 计数器中 frames 的 ok 值，确保没有重新编译。
            torch._dynamo.reset()
            # 使用动态形状编译第一个批次
            compiled_sdpa = torch.compile(sdpa_partial, dynamic=True)
            compiled_out1 = compiled_sdpa(q1, k1, v1)
            compiled_out1.backward(backward_grad1)

            self._check_out_and_grad(
                golden_out1,
                ref_out1,
                compiled_out1,
                q1_gold,
                q1_ref,
                q1,
                k1_gold,
                k1_ref,
                k1,
                v1_gold,
                v1_ref,
                v1,
            )
            # 断言 dynamo 计数器中 frames 的 ok 值为 1
            self.assertEqual(torch._dynamo.utils.counters["frames"]["ok"], 1)

            # 没有重新编译，使用编译后的动态形状版本
            compiled_out2 = compiled_sdpa(q2, k2, v2)
            compiled_out2.backward(backward_grad2)
            self._check_out_and_grad(
                golden_out2,
                ref_out2,
                compiled_out2,
                q2_gold,
                q2_ref,
                q2,
                k2_gold,
                k2_ref,
                k2,
                v2_gold,
                v2_ref,
                v2,
            )
            # 断言 dynamo 计数器中 frames 的 ok 值为 1
            self.assertEqual(torch._dynamo.utils.counters["frames"]["ok"], 1)
    # 定义一个方法用于运行自动化的动态测试
    def run_automatic_dynamic_test(
        # 参数 score_mod 是一个可调用对象，用于修改分数
        score_mod: Callable,
        # 参数 dtype 指定张量的数据类型，默认为 torch.float16
        dtype: torch.dtype = torch.float16,
        # 参数 B 指定 Batch 大小，默认使用之前定义的 B
        B: int = B,
        # 参数 H 指定张量的高度，默认使用之前定义的 H
        H: int = H,
        # 参数 S 指定张量的宽度，默认使用之前定义的 S
        S: int = S,
        # 参数 D 指定张量的深度，默认使用之前定义的 D
        D: int = D,
        sdpa_partial = create_attention(score_mod)
        # 使用给定的注意力模型和评分模式创建一个部分的自注意力对象

        # The first eager batch, shape (B, H, S, D)
        q1 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")
        k1 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")
        v1 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")
        # 生成第一个急切批次的查询(q1)，键(k1)和值(v1)张量，形状为(B, H, S, D)

        golden_out1 = sdpa_partial(
            q1.to(torch.float64), k1.to(torch.float64), v1.to(torch.float64)
        )
        ref_out1 = sdpa_partial(q1, k1, v1)
        # 使用部分自注意力对象计算第一个批次的输出golden_out1和参考输出ref_out1

        # The second eager batch, shape (B * 2, H, S / 2, D)
        B = int(B * 2)
        S = int(S / 2)
        q2 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")
        k2 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")
        v2 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")
        # 调整参数后生成第二个急切批次的查询(q2)，键(k2)和值(v2)张量，形状为(B * 2, H, S / 2, D)

        golden_out2 = sdpa_partial(
            q2.to(torch.float64), k2.to(torch.float64), v2.to(torch.float64)
        )
        ref_out2 = sdpa_partial(q2, k2, v2)
        # 使用部分自注意力对象计算第二个批次的输出golden_out2和参考输出ref_out2

        # The third eager batch, shape (B * 4, H, S / 4, D)
        B = int(B * 2)
        S = int(S / 2)
        q3 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")
        k3 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")
        v3 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")
        # 再次调整参数生成第三个急切批次的查询(q3)，键(k3)和值(v3)张量，形状为(B * 4, H, S / 4, D)

        golden_out3 = sdpa_partial(
            q3.to(torch.float64), k3.to(torch.float64), v3.to(torch.float64)
        )
        ref_out3 = sdpa_partial(q3, k3, v3)
        # 使用部分自注意力对象计算第三个批次的输出golden_out3和参考输出ref_out3

        # 需要清除 dynamo 计数器，因为弹性注意力急切模式也使用 dynamo 追踪。
        # 我们检查 dynamo 计数器["frames"]["ok"] 来确保：
        # 1. 第一个批次使用静态形状编译
        # 2. 第二个批次使用动态形状编译
        # 3. 第三个批次不重新编译
        torch._dynamo.reset()
        # 清除 dynamo 计数器的状态，以准备下一轮计数

        # 注意，似乎我们的计算比 float32 更不准确，这可能是由于在线 softmax 导致的
        if dtype == torch.float32:
            fudge_factor = 10.0
        else:
            fudge_factor = 1.1
        # 根据张量数据类型选择合适的 fudge_factor，以用于比较浮点数输出的精度

        # The first batch.
        compiled_sdpa = torch.compile(sdpa_partial)
        compiled_out1 = compiled_sdpa(q1, k1, v1)
        self._check_equal(golden_out1, ref_out1, compiled_out1, fudge_factor)
        self.assertEqual(torch._dynamo.utils.counters["frames"]["ok"], 1)
        # 编译部分自注意力对象并计算第一个批次的编译输出，然后检查输出的相等性及计数器状态

        # The second batch (automatic dynamic).
        compiled_out2 = compiled_sdpa(q2, k2, v2)
        self._check_equal(golden_out2, ref_out2, compiled_out2, fudge_factor)
        self.assertEqual(torch._dynamo.utils.counters["frames"]["ok"], 2)
        # 计算第二个批次的编译输出，并检查输出的相等性及计数器状态

        # The third batch (no re-compilation).
        compiled_out3 = compiled_sdpa(q3, k3, v3)
        self._check_equal(golden_out3, ref_out3, compiled_out3, fudge_factor)
        self.assertEqual(torch._dynamo.utils.counters["frames"]["ok"], 2)
        # 计算第三个批次的编译输出，并检查输出的相等性及计数器状态
    # 测试特定的内置评分修正函数与指定数据类型的运行情况
    def test_builtin_score_mods(self, dtype: torch.dtype, score_mod: Callable):
        self.run_test(score_mod, dtype)

    # 以下测试预期会失败，因为还不支持动态形状下的块稀疏性
    @expectedFailure
    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    @common_utils.parametrize("score_mod", test_score_mods)
    # 测试特定的内置评分修正函数与动态形状下的数据类型的运行情况
    def test_builtin_score_mods_dynamic(self, dtype: torch.dtype, score_mod: Callable):
        self.run_dynamic_test(score_mod, dtype)

    # 以下测试预期会失败，因为还不支持动态形状下的块稀疏性
    @expectedFailure
    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    @common_utils.parametrize("score_mod", test_score_mods)
    # 测试特定的内置评分修正函数与自动动态形状下的数据类型的运行情况
    def test_builtin_score_mods_automatic_dynamic(
        self, dtype: torch.dtype, score_mod: Callable
    ):
        self.run_automatic_dynamic_test(score_mod, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    @common_utils.parametrize("score_mod", test_score_mods)
    # 测试特定的内置评分修正函数，对于不同序列长度的情况下的运行情况
    def test_builtin_score_mods_different_seqlen(
        self, dtype: torch.dtype, score_mod: Callable
    ):
        self.run_test(
            score_mod,
            dtype,
            B,
            H,
            S // 2,  # Q序列长度与K/V序列长度不同
            D,
            B,
            H,
            S,
            D,
        )

    # 测试输入步长的不同情况
    test_input_strides = [
        ((H * S * D, S * D, D, 1), 997),  # 偏移
        ((H * D, D, B * H * D, 1), 499),  # 转置维度
        (
            (S * (D + 1), B * S * (D + 1), (D + 1), 1),
            293,
        ),  # 在一个维度上增加缓冲区
        (
            (1, D, (B + 1) * (H + 1) * D, 1),
            97,
        ),  # 在多个维度上增加缓冲区 + 共享维度
    ]

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    @common_utils.parametrize(
        "q_s", test_input_strides[:-2]
    )
    @common_utils.parametrize("k_s", test_input_strides)
    @common_utils.parametrize("v_s", test_input_strides)
    # 修复查询广播布局问题的待办事项
    # 测试函数，用于验证支持不同步长输入的注意力机制
    def test_strided_inputs(self, dtype: torch.dtype, q_s, k_s, v_s):
        # 创建具有指定形状和数据类型的随机张量，存储在 GPU 上
        q1 = torch.randn((B * H * S * D * 2), dtype=dtype, device="cuda")
        k1 = torch.randn((B * H * S * D * 2), dtype=dtype, device="cuda")
        v1 = torch.randn((B * H * S * D * 2), dtype=dtype, device="cuda")

        # 定义输入张量的形状，根据输入的步长设置
        q_shape = (B, H, S // 2, D)
        k_shape = (B, H, S, D)
        v_shape = (B, H, S, D)

        q_strides, q_offset = q_s
        # 计算每个维度上的最大偏移量，确保不超过张量的总长度
        q_max = [x * (y - 1) for x, y in zip(q_strides, q_shape)]
        assert sum(q_max) + q_offset < B * H * S * D * 2
        assert q_strides[-1] == 1
        # 使用给定的步长和偏移创建 q 张量
        q = torch.as_strided(q1, q_shape, q_strides, q_offset)

        k_strides, k_offset = k_s
        k_max = [x * (y - 1) for x, y in zip(k_strides, k_shape)]
        assert sum(k_max) + k_offset < B * H * S * D * 2
        assert k_strides[-1] == 1
        # 使用给定的步长和偏移创建 k 张量
        k = torch.as_strided(k1, k_shape, k_strides, k_offset)

        v_strides, v_offset = v_s
        v_max = [x * (y - 1) for x, y in zip(v_strides, v_shape)]
        assert sum(v_max) + v_offset < B * H * S * D * 2
        assert v_strides[-1] == 1
        # 使用给定的步长和偏移创建 v 张量
        v = torch.as_strided(v1, v_shape, v_strides, v_offset)

        # 创建块稀疏掩码，用于注意力机制
        block_mask = _create_empty_block_sparse_mask(q, k, v)
        # 创建部分注意力机制，并编译成可执行函数
        sdpa_partial = create_attention(
            score_mod=_generate_alibi_bias(8), block_sparse_mask=block_mask
        )
        compiled_sdpa = torch.compile(sdpa_partial)
        # 计算标准输出和编译输出的注意力值
        ref_out = sdpa_partial(q, k, v)
        compiled_out = compiled_sdpa(q, k, v)

        # 设置容差值，用于检验标准输出和编译输出的近似程度
        tolerance = Tolerances(atol=2e-1, rtol=2e-1)
        # 使用容差检验标准输出和编译输出的近似程度
        torch.testing.assert_close(
            ref_out, compiled_out, atol=tolerance.atol, rtol=tolerance.rtol
        )

    # 声明一个支持的平台测试函数，验证块稀疏掩码创建是否已编译
    @supported_platform
    def test_create_block_sparse_mask_is_compiled(self):
        # 创建一个生成指定形状、数据类型和设备的随机张量的部分函数
        make_tensor = functools.partial(
            torch.randn,
            (B, H, S, D),
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        )
        # 生成 q, k, v 张量
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        # 定义一个编译函数，用于创建块稀疏掩码并进行注意力计算
        @torch.compile
        def func(q, k, v):
            # 创建块稀疏掩码，限制只能在对角线以下
            block_sparse_mask = _create_block_sparse_mask(
                torch.tril(
                    torch.ones(
                        q.shape[-2], k.shape[-2], dtype=torch.bool, device=q.device
                    )
                ),
                128,
                128,
            )

            # 使用块稀疏掩码进行灵活的注意力计算
            out = _flex_attention(
                q,
                k,
                v,
                _causal,
                block_sparse_mask,
            )
            return out

        # 运行并获取函数代码及其相关信息
        _, code = run_and_get_code(func, q, k, v)
        # 确保 _create_block_sparse_mask 已编译并生成 3 个内核，
        # flex_attention 生成 1 个内核。
        FileCheck().check_count(".run(", 4, True).run(code[0])
    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    def test_block_sparse_mask_is_reused(self):
        # 定义一个局部函数 make_tensor，用于生成指定形状的张量，设定在 CUDA 设备上，并要求梯度计算
        make_tensor = functools.partial(
            torch.randn,
            (B, H, S, D),
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        )
        # 生成张量 q, k, v，均为随机张量在 CUDA 设备上，并要求梯度计算
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        # 创建 k2 和 v2 张量，分别在 k 和 v 的基础上加 1
        k2 = k + 1
        v2 = v + 1

        @torch.compile
        def func(q, k, v, k2, v2):
            # 创建块稀疏掩码，用于灵活注意力机制，基于给定的下三角张量掩码
            block_sparse_mask = _create_block_sparse_mask(
                torch.tril(
                    torch.ones(
                        q.shape[-2], k.shape[-2], dtype=torch.bool, device=q.device
                    )
                ),
                128,
                128,
            )

            # 第一个注意力调用，使用灵活注意力函数和 block_sparse_mask
            q = _flex_attention(
                q,
                k,
                v,
                _causal,
                block_sparse_mask,
            )
            # 第二个注意力调用，使用灵活注意力函数和相同的 block_sparse_mask，但是针对 k2 和 v2
            out = _flex_attention(
                q,
                k2,
                v2,
                _causal,
                block_sparse_mask,
            )
            return out

        # 运行并获取 func 函数的代码，检查生成的 CUDA 核心数
        _, code = run_and_get_code(func, q, k, v, k2, v2)
        # 确保 _create_block_sparse_mask 被编译，并生成了 3 个核心，2 个 _flex_attention 调用生成了 2 个核心
        FileCheck().check_count(".run(", 5, True).run(code[0])
    # 定义一个测试函数，测试序列掩码功能，接受一个数据类型参数 dtype
    def test_seq_masking(self, dtype):
        # 创建一个大小为 S 的全零张量，布尔类型，存储在 GPU 上
        seq_idx = torch.zeros(S, device="cuda", dtype=torch.bool)
        # 将序列索引的一半之后的元素设为 True，表示序列掩码的一部分
        seq_idx[S // 2 :] = 1

        # 定义一个序列掩码修改函数，根据序列索引过滤注意力分数
        def seq_mask_mod(score, b, h, q, kv):
            return torch.where(seq_idx[q] == seq_idx[kv], score, float("-inf"))

        # 运行测试，调用 self.run_test 方法，传入序列掩码修改函数和数据类型参数 dtype
        self.run_test(seq_mask_mod, dtype)

    # 标记为支持的平台，并使用指定的数据类型参数进行参数化测试
    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    # 测试从偏置加载的仅序列相关信息的函数
    def test_load_from_bias_seq_only(self, dtype):
        # 在 GPU 上生成一个大小为 SxS 的随机张量，使用指定的数据类型 dtype
        bias = torch.randn(S, S, device="cuda", dtype=dtype)

        # 定义一个偏置修改函数，根据查询和键的位置加上对应的偏置
        def bias_mod(score, b, h, q, kv):
            return score + bias[q, kv]

        # 运行测试，调用 self.run_test 方法，传入偏置修改函数和数据类型参数 dtype
        self.run_test(bias_mod, dtype)

    # 标记为支持的平台，并使用指定的数据类型参数进行参数化测试
    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    # 测试从偏置加载的批次序列相关信息的函数
    def test_load_from_bias_seq_batch(self, dtype):
        # 在 GPU 上生成一个大小为 BxSxS 的随机张量，使用指定的数据类型 dtype
        bias = torch.randn(B, S, S, device="cuda", dtype=dtype)

        # 定义一个偏置修改函数，根据批次、查询和键的位置加上对应的偏置
        def bias_mod(score, b, h, q, kv):
            return score + bias[b, q, kv]

        # 运行测试，调用 self.run_test 方法，传入偏置修改函数和数据类型参数 dtype
        self.run_test(bias_mod, dtype)

    # 标记为支持的平台，并使用指定的数据类型参数进行参数化测试
    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    # 测试从偏置加载的批次、头部和序列相关信息的函数
    def test_load_from_bias_head_seq_batch(self, dtype):
        # 在 GPU 上生成一个大小为 BxHxSxS 的随机张量，使用指定的数据类型 dtype
        bias = torch.randn(B, H, S, S, device="cuda", dtype=dtype)

        # 定义一个偏置修改函数，根据批次、头部、查询和键的位置加上对应的偏置
        def bias_mod(score, b, h, q, kv):
            return score + bias[b, h, q, kv]

        # 运行测试，调用 self.run_test 方法，传入偏置修改函数和数据类型参数 dtype
        self.run_test(bias_mod, dtype)

    # 标记为支持的平台，并使用指定的数据类型参数进行参数化测试
    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    # 测试从相对偏置加载的函数
    def test_load_rel_bias(self, dtype):
        # 在 GPU 上生成一个大小为 2*S 的随机张量，用于存储相对偏置，使用指定的数据类型 dtype
        rel_bias = torch.randn(2 * S, device="cuda", dtype=dtype)

        # 定义一个相对偏置修改函数，根据查询和键的位置加上对应的相对偏置
        def bias_mod(score, b, h, q, kv):
            return score + rel_bias[(q - kv) + S]

        # 运行测试，调用 self.run_test 方法，传入相对偏置修改函数和数据类型参数 dtype
        self.run_test(bias_mod, dtype)

    # 标记为支持的平台，并使用指定的数据类型参数进行参数化测试
    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    # 测试双向相关、因果依赖的函数
    def test_dependent_causal_bidirectional(self, dtype):
        # 在 GPU 上生成一个大小为 B 的随机整数张量，表示每个批次中的双向数量
        num_bidirectional = torch.randint(0, S, (B,), device="cuda", dtype=torch.int32)

        # 定义一个偏置修改函数，根据因果性和双向性判断对注意力分数的修改
        def bias_mod(score, b, h, q, kv):
            causal_attention = q >= kv
            cur_num_bidirectional = num_bidirectional[b]
            bidirectional_attention_on_video = (q <= cur_num_bidirectional) & (
                kv <= cur_num_bidirectional
            )
            return torch.where(
                bidirectional_attention_on_video | causal_attention,
                score,
                -float("inf"),
            )

        # 运行测试，调用 self.run_test 方法，传入偏置修改函数和数据类型参数 dtype
        self.run_test(bias_mod, dtype)

    # 标记为支持的平台，并使用指定的数据类型参数进行参数化测试
    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    # 测试二维注意力掩码函数
    def test_natten_2d(self, dtype):
        # 设置 H 的值为 32，计算 W 的值为 S//H，WINDOW 的值为 3，并断言 HxW 等于 S
        H = 32
        W = S // H
        WINDOW = 3
        assert W * H == S

        # 定义一个获取二维坐标的函数，根据索引 idx 计算其在二维坐标系中的位置
        def get_x_y(idx):
            # 这里应该是整数除法，但我们没有很好的支持这种操作
            return idx / W, idx % W

        # 定义一个二维注意力掩码函数，根据二维坐标的距离进行掩码
        def natten_mask(score, b, h, q, kv):
            q_x, q_y = get_x_y(q)
            kv_x, kv_y = get_x_y(kv)
            return torch.where(
                ((q_x - kv_x).abs() <= WINDOW) | ((q_y - kv_y).abs() <= WINDOW),
                score,
                float("-inf"),
            )

        # 运行测试，调用 self.run_test 方法，传入二维注意力掩码函数和数据类型参数 dtype
        self.run_test(natten_mask, dtype)
    # 声明一个测试函数，用于测试子图在分解方面的表现，支持多种数据类型
    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_subgraph_respect_decompostion(self, dtype):
        # 导入所需的模块和函数
        from torch._decomp import core_aten_decompositions
        from torch.fx.experimental.proxy_tensor import make_fx

        # 定义一个修改分数的函数，用于灵活的注意力计算
        def score_mod_func(score, b, h, q, kv):
            return score - q // (1 + kv)

        # 创建一个生成张量的部分函数，生成形状为(2, 2, 128, 4)的随机张量
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 4),
            device="cuda",
            dtype=torch.float64,
            requires_grad=True,
        )
        # 使用部分函数创建查询、键和值张量
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        # 定义一个灵活注意力机制的部分函数，使用自定义的分数修改函数
        flex_attention = functools.partial(_flex_attention, score_mod=score_mod_func)

        # 使用make_fx函数创建一个灵活注意力机制的函数图，传入空的分解表
        gm = make_fx(flex_attention, decomposition_table={})(query, key, value)

        # 断言生成的SDPA分数代码与预期的代码匹配
        self.assertExpectedInline(
            gm.sdpa_score0.code.strip(),
            """\
# 定义一个方法 `forward`，接收五个参数 `arg0_1`, `arg1_1`, `arg2_1`, `arg3_1`, `arg4_1`
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1):
    # 调用 PyTorch 的 ATen 操作 `add`，将 `arg4_1` 加上 1
    add = torch.ops.aten.add.Tensor(arg4_1, 1);  arg4_1 = None
    # 调用 PyTorch 的 ATen 操作 `floor_divide`，对 `arg3_1` 进行除法运算，使用先前计算的 `add` 作为除数
    floor_divide = torch.ops.aten.floor_divide.default(arg3_1, add);  arg3_1 = add = None
    # 调用 PyTorch 的 ATen 操作 `sub`，计算 `arg0_1` 减去先前计算的 `floor_divide`
    sub = torch.ops.aten.sub.Tensor(arg0_1, floor_divide);  arg0_1 = floor_divide = None
    # 返回结果 `sub`
    return sub""",
        )

# 将 `floor_div` 分解以供 `core_aten_decompositions` 使用
gm = make_fx(flex_attention, decomposition_table=core_aten_decompositions())(
    query, key, value
)
self.assertExpectedInline(
    gm.sdpa_score0.code.strip(),
    """\
# 重新定义方法 `forward`，接收五个参数 `arg0_1`, `arg1_1`, `arg2_1`, `arg3_1`, `arg4_1`
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1):
    # 调用 PyTorch 的 ATen 操作 `add`，将 `arg4_1` 加上 1
    add = torch.ops.aten.add.Tensor(arg4_1, 1);  arg4_1 = None
    # 调用 PyTorch 的 ATen 操作 `div`，对 `arg3_1` 进行除法运算，使用先前计算的 `add` 作为除数，同时指定舍入模式为 'floor'
    div = torch.ops.aten.div.Tensor_mode(arg3_1, add, rounding_mode = 'floor');  arg3_1 = add = None
    # 调用 PyTorch 的 ATen 操作 `sub`，计算 `arg0_1` 减去先前计算的 `div`
    sub = torch.ops.aten.sub.Tensor(arg0_1, div);  arg0_1 = div = None
    # 返回结果 `sub`
    return sub""",
)

@supported_platform
@common_utils.parametrize("dtype", test_dtypes_fast)
# 定义测试方法 `test_silu_on_score`，接收一个参数 `dtype`
def test_silu_on_score(self, dtype):
    # 定义内部函数 `silu_score`，接收参数 `score`, `b`, `h`, `q`, `kv`，并返回使用 torch.nn.functional.silu 处理后的 `score`
    def silu_score(score, b, h, q, kv):
        return torch.nn.functional.silu(score)

    # 运行测试 `silu_score`，使用给定的 `dtype`
    self.run_test(silu_score, dtype)

@supported_platform
@common_utils.parametrize("dtype", test_dtypes_fast)
# 定义测试方法 `test_padded_dense_causal`，接收一个参数 `dtype`
def test_padded_dense_causal(self, dtype):
    # 创建序列长度 `seq_len`，使用 torch.arange 在 CUDA 设备上生成 `B` 个整数序列，数据类型为 `torch.int32`
    seq_len = torch.arange(B, device="cuda", dtype=torch.int32) + 1

    # 定义内部函数 `create_padded_dense_wrapper`，接收一个原始得分模块 `orig_score_mod`，返回一个新的得分模块 `njt_score_mod`
    def create_padded_dense_wrapper(orig_score_mod):
        # 定义内部函数 `njt_score_mod`，接收参数 `qk`, `b`, `h`, `q`, `kv`，根据条件返回处理后的得分模块
        def njt_score_mod(qk, b, h, q, kv):
            return torch.where(
                # 使用 torch.where 根据条件判断 `qk` 是否小于等于 `seq_len[b]`，是则返回 `orig_score_mod` 处理后的结果，否则返回负无穷
                qk <= seq_len[b], orig_score_mod(qk, b, h, q, kv), -float("inf")
            )

        return njt_score_mod

    # 创建使用 `_causal` 的填充密集层得分模块 `causal_njt`
    causal_njt = create_padded_dense_wrapper(_causal)

    # 运行测试 `causal_njt`，使用给定的 `dtype`
    self.run_test(causal_njt, dtype)

@supported_platform
@common_utils.parametrize("dtype", test_dtypes_fast)
# 定义测试方法 `test_captured_scale`，接收一个参数 `dtype`
def test_captured_scale(self, dtype):
    # 创建标量 `scale`，在 CUDA 设备上生成一个值为 1 的张量，数据类型为 `torch.int32`
    scale = torch.ones((), device="cuda", dtype=torch.int32)

    # 定义内部函数 `score_mod_scale`，接收参数 `qk`, `b`, `h`, `q`, `kv`，返回 `qk` 加上 `scale` 后的结果
    def score_mod_scale(qk, b, h, q, kv):
        return qk + scale

    # 运行测试 `score_mod_scale`，使用给定的 `dtype`
    self.run_test(score_mod_scale, dtype)

@supported_platform
# 定义测试方法 `test_recompile_changed_score_mod`，接收一个参数 `dtype`
def test_recompile_changed_score_mod(self, dtype):
    # 创建标量 `scale`，在 CUDA 设备上生成一个值为 1 的张量，数据类型为 `torch.int32`
    scale = torch.ones((), device="cuda", dtype=torch.int32)
    # 定义一个布尔值常量 `ADD`，初始值为 True
    ADD = True

    # 定义内部函数 `score_mod_scale`，接收参数 `qk`, `b`, `h`, `q`, `kv`，根据 `ADD` 的值返回不同的计算结果
    def score_mod_scale(qk, b, h, q, kv):
        if ADD:
            return qk + scale
        else:
            return qk * scale

    # 运行测试 `score_mod_scale`，使用给定的 `dtype`
    self.run_test(score_mod_scale, dtype)
    # 修改 `ADD` 的值为 False
    ADD = False
    # 再次运行测试 `score_mod_scale`，使用给定的 `dtype`
    self.run_test(score_mod_scale, dtype)

@supported_platform
@expectedFailure  # 如果我们捕获了一个张量，那么我们可以对其进行缩减，这是不允许的
@common_utils.parametrize("dtype", test_dtypes_fast)
    def test_captured_reduction(self, dtype):
        # 在 CUDA 设备上生成一个形状为 (B, 8) 的随机张量
        scale = torch.randn((B, 8), device="cuda")

        # 定义一个函数 score_mod_scale，接受 qk, b, h, q, kv 作为参数，返回 qk 加上 scale[b] 在最后一个维度上的和
        def score_mod_scale(qk, b, h, q, kv):
            return qk + scale[b].sum(dim=-1)

        # 运行测试函数 run_test，传入 score_mod_scale 函数和指定的数据类型 dtype
        self.run_test(score_mod_scale, dtype)

    @supported_platform
    def test_multiple_score_mod_calls(self):
        # 在 CUDA 设备上生成一个形状为 (1, 8, 1024, 64) 的随机查询张量
        query = torch.randn((1, 8, 1024, 64), dtype=torch.float32, device="cuda")
        # 生成包含两个形状相同的随机键张量的列表
        keys = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device="cuda")
            for _ in range(2)
        ]
        # 生成包含两个形状相同的随机值张量的列表
        values = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device="cuda")
            for _ in range(2)
        ]

        # 定义一个函数 scoremod_1，接受 qk, b, h, q, kv 作为参数，返回 qk 加上 (q - kv)
        def scoremod_1(qk, b, h, q, kv):
            return qk + (q - kv)

        # 定义一个函数 scoremod_2，接受 qk, b, h, q, kv 作为参数，返回根据条件 q >= kv 来决定的值
        def scoremod_2(qk, b, h, q, kv):
            return torch.where(q >= kv, qk, -float("inf"))

        # 定义函数 f，接受 q, k1, k2, v1, v2 作为参数，调用 _flex_attention 函数两次，并应用不同的 score_mod 函数
        def f(q, k1, k2, v1, v2):
            q2 = _flex_attention(q, k1, v1, score_mod=scoremod_1)
            return _flex_attention(q2, k2, v2, score_mod=scoremod_2)

        # 调用函数 f，并传入查询张量和键值张量列表，将结果存储在 out 变量中
        out = f(query, *keys, *values)
        # 使用 torch.compile 编译函数 f，并再次传入相同的参数，将结果存储在 out2 变量中
        out2 = torch.compile(f)(query, *keys, *values)
        # 设置容差对象 tolerance，用于检验 out 和 out2 的接近程度
        tolerance = Tolerances(atol=2e-1, rtol=2e-1)
        # 使用 torch.testing.assert_close 断言 out 和 out2 在给定的容差范围内接近
        torch.testing.assert_close(out, out2, atol=tolerance.atol, rtol=tolerance.rtol)

    @supported_platform
    def test_multiple_score_mod_calls2(self):
        # 在 CUDA 设备上生成一个形状为 (1, 8, 1024, 64) 的随机查询张量
        query = torch.randn((1, 8, 1024, 64), dtype=torch.float32, device="cuda")
        # 生成包含三个形状相同的随机键张量的列表
        keys = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device="cuda")
            for _ in range(3)
        ]
        # 生成包含三个形状相同的随机值张量的列表
        values = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device="cuda")
            for _ in range(3)
        ]

        # 定义一个函数 scoremod_1，接受 qk, b, h, q, kv 作为参数，返回 qk 加上 (q - kv)
        def scoremod_1(qk, b, h, q, kv):
            return qk + (q - kv)

        # 定义一个函数 scoremod_2，接受 qk, b, h, q, kv 作为参数，返回根据条件 q >= kv 来决定的值
        def scoremod_2(qk, b, h, q, kv):
            return torch.where(q >= kv, qk, -float("inf"))

        # 使用 functools.partial 创建一个局部函数 attention1，固定 score_mod 参数为 scoremod_1
        attention1 = functools.partial(_flex_attention, score_mod=scoremod_1)

        # 定义函数 f，接受 q, k1, k2, k3, v1, v2, v3 作为参数，调用 attention1 函数一次，然后调用 _flex_attention 函数两次，并应用不同的 score_mod 函数
        def f(q, k1, k2, k3, v1, v2, v3):
            q2 = attention1(q, k1, v1)
            q3 = _flex_attention(q2, k2, v2, score_mod=scoremod_2)
            return _flex_attention(q3, k3, v3, score_mod=scoremod_1)

        # 调用函数 f，并传入查询张量和键值张量列表，将结果存储在 out 变量中
        out = f(query, *keys, *values)
        # 使用 torch.compile 编译函数 f，并再次传入相同的参数，将结果存储在 out2 变量中
        out2 = torch.compile(f)(query, *keys, *values)
        # 使用 self.assertTrue 断言 out 和 out2 的绝对值平均值小于 1e-2
        self.assertTrue((out - out2).abs().mean() < 1e-2)
    # 定义测试函数，验证输入是否被正确处理
    def test_inputs_are_realized(self):
        # 定义内部函数f，接收三个参数q, k, v，生成一个随机的CUDA张量x，并对其进行操作
        def f(q, k, v):
            x = torch.randn(1024, device="cuda")
            x = x * 2

            # 定义内部函数func，接收参数qk, b, h, q, kv，返回qk加上x[q]的结果
            def func(qk, b, h, q, kv):
                return qk + x[q]

            # 调用_flex_attention函数，传入参数q.sin(), k, v和自定义的score_mod函数func，并返回cosine值
            return _flex_attention(q.sin(), k, v, score_mod=func).cos()

        # 生成三个随机张量q, k, v，并传入函数f进行计算，得到参考结果ref
        q, k, v = (
            torch.randn(1, 8, 1024, 64, device="cuda", requires_grad=True)
            for _ in range(3)
        )
        ref = f(q, k, v)

        # 使用torch.compile对函数f进行编译，得到编译后的输出out
        out = torch.compile(f)(q, k, v)

        # 断言参考结果ref和编译输出out的平均绝对差小于1e-2
        self.assertTrue((ref - out).abs().mean() < 1e-2)

        # 随机生成与q相同大小的梯度张量gradOut
        gradOut = torch.randn_like(q)

        # 计算参考结果ref和编译输出out关于(q, k, v)的梯度
        ref_grads = torch.autograd.grad(ref, (q, k, v), gradOut)
        out_grads = torch.autograd.grad(out, (q, k, v), gradOut)

        # 逐一断言参考结果和编译输出的梯度的平均绝对差小于1e-2
        for ref, out in zip(ref_grads, out_grads):
            self.assertTrue((ref - out).abs().mean() < 1e-2)

    # 标记测试函数在支持的平台上运行
    @supported_platform
    def test_epilogue_fused(self):
        # 使用torch.compile装饰器编译函数f，接收参数q, k, v
        @torch.compile
        def f(q, k, v):
            # 调用_flex_attention函数，传入参数q, k, v，计算输出并取cosine值
            out = _flex_attention(q, k, v)
            return out.cos()

        # 生成三个随机张量q, k, v，并传入函数f进行计算
        q, k, v = (torch.randn(1, 8, 1024, 64, device="cuda") for _ in range(3))

        # 重置metrics
        metrics.reset()

        # 调用f函数，计算输出
        f(q, k, v)

        # 计算访问的字节数，假定每个元素占据torch.float32的大小
        accessed_bytes = 1 * 8 * 1024 * 64 * torch.float32.itemsize

        # 计算访问次数，包括q, k, v的读取和一个输出
        num_accesses = 4

        # TODO: Get rid of this fudge factor
        # 目前需要这个调整因子，因为：
        # 1. 出于某种原因，我们不必要地材料化了注意力的输出（这与突变有关）
        # 2. 我们还写了多余的logsumexp
        num_accesses += 2

        # 断言metrics.num_bytes_accessed小于计算得到的访问字节数乘以访问次数
        self.assertLess(metrics.num_bytes_accessed, accessed_bytes * num_accesses)

    # 标记测试函数在支持的平台上运行，并跳过该测试的原因是Triton的bug
    @supported_platform
    @skip("Triton bug ")  # https://github.com/pytorch/pytorch/issues/124571
    @common_utils.parametrize("dtype", test_dtypes)
    def test_njt_causal(self, dtype):
        # 生成offsets张量，设备为cuda，数据类型为torch.int32
        offsets = torch.tensor(
            [0, 1024, 1024 + 512, S], device="cuda", dtype=torch.int32
        )

        # 生成seq_idx张量，全零，设备为cuda，数据类型为torch.int32，长度为S
        seq_idx = torch.zeros(S, device="cuda", dtype=torch.int32)

        # 遍历offsets的索引，更新seq_idx的部分元素
        for idx in range(len(offsets) - 1):
            seq_idx[offsets[idx] : offsets[idx + 1]] = idx

        # 定义内部函数create_njt_wrapper，接收三个参数orig_score_mod, offsets, seq_idx
        def create_njt_wrapper(orig_score_mod, offsets, seq_idx):
            # 定义内部函数njt_score_mod，接收五个参数qk, b, h, q, kv
            def njt_score_mod(qk, b, h, q, kv):
                # 计算q_nested和kv_nested，使用offsets和seq_idx对q和kv进行调整
                q_nested = q - offsets[seq_idx[q]]
                kv_nested = kv - offsets[seq_idx[kv]]
                # 返回orig_score_mod的结果，传入调整后的q_nested和kv_nested
                return orig_score_mod(qk, b, h, q_nested, kv_nested)

            return njt_score_mod

        # 使用create_njt_wrapper函数创建causal_njt函数，传入_causal, offsets, seq_idx
        causal_njt = create_njt_wrapper(_causal, offsets, seq_idx)

        # 运行测试，传入causal_njt函数和dtype参数
        self.run_test(causal_njt, dtype)
    @supported_platform
    @patch.object(torch._inductor.config, "max_autotune", True)
    # 标记为在支持的平台上运行，并且模拟修改torch._inductor.config中max_autotune的值为True
    def test_max_autotune(self):
        # 定义一个函数score_mod，用于修改得分，参数为score, b, h, m, n，返回值是score乘以2
        def score_mod(score, b, h, m, n):
            return score * 2

        # 运行self对象的run_test方法，传入score_mod函数作为参数
        self.run_test(score_mod)

    @supported_platform
    @skip("TODO: Figure out why this is erroring")
    @patch.object(torch._inductor.config, "max_autotune", True)
    # 标记为在支持的平台上运行，同时忽略此测试的执行（因为注释中提到还需解决错误）
    def test_max_autotune_with_captured(self):
        # 创建使用cuda设备的头比例、批比例和令牌比例的张量
        head_scale = torch.randn(H, device="cuda")
        batch_scale = torch.randn(B, device="cuda")
        tok_scale = torch.randn(S, device="cuda")

        # 定义一个bias_mod函数，用于修改得分，参数为score, batch, head, token_q, token_kv
        def bias_mod(score, batch, head, token_q, token_kv):
            # 分别对score加上token_q对应的令牌比例、batch对应的批比例和head对应的头比例
            score = score + tok_scale[token_q]
            score = score + batch_scale[batch]
            score = score + head_scale[head]
            return score

        # 运行self对象的run_test方法，传入bias_mod函数作为参数
        self.run_test(bias_mod)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    @common_utils.parametrize("score_mod", [_identity, _causal])
    # 标记为在支持的平台上运行，使用common_utils.parametrize对dtype和score_mod进行参数化
    # 定义一个测试方法，用于验证 logsumexp 的正确性
    def test_logsumexp_correctness(self, dtype, score_mod):
        # 创建一个生成张量的部分函数，生成指定大小和设备的随机张量，需要梯度
        make_tensor = functools.partial(
            torch.randn,
            (B, H, S, D),
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
        # 生成三个张量 q, k, v
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        # 创建一个稀疏块掩码
        block_mask = _create_empty_block_sparse_mask(q, k, v)

        # 定义一个编译的 Torch 函数 sdpa_hop，使用 flex_attention_hop 进行计算
        @torch.compile
        def sdpa_hop(q, k, v, score_mod, block_mask):
            return flex_attention_hop(
                q,
                k,
                v,
                score_mod,
                block_mask.kv_num_blocks,
                block_mask.kv_indices,
                block_mask.q_num_blocks,
                block_mask.q_indices,
                block_mask.KV_BLOCK_SIZE,
                block_mask.Q_BLOCK_SIZE,
            )

        # 定义另一个编译的 Torch 函数 eager_sdpa_hop，主要入口点是 FlexAttention，不返回 LSE。
        # 此外，它确保使用 aot-eager 后端编译这个 hop。
        @torch.compile(backend="aot_eager")
        def eager_sdpa_hop(q, k, v, score_mod, block_mask):
            """The main entrypoint for FlexAttention doesnt return LSE.
            Besides dropping LSE it also ensures that the hop is compiled with aot-eager
            backend. We need to replicate this.
            """
            return flex_attention_hop(
                q,
                k,
                v,
                score_mod,
                block_mask.kv_num_blocks,
                block_mask.kv_indices,
                block_mask.q_num_blocks,
                block_mask.q_indices,
                block_mask.KV_BLOCK_SIZE,
                block_mask.Q_BLOCK_SIZE,
            )

        # 调用 eager_sdpa_hop 获取参考输出 ref_out 和参考 LSE ref_lse
        ref_out, ref_lse = eager_sdpa_hop(
            q.to(torch.float64),
            k.to(torch.float64),
            v.to(torch.float64),
            score_mod,
            block_mask,
        )
        # 调用 sdpa_hop 获取编译输出 compiled_out 和编译 LSE compiled_lse
        compiled_out, compiled_lse = sdpa_hop(q, k, v, score_mod, block_mask)

        # 比较编译版本和参考版本的 LSE
        # 编译版本使用基数转换技巧更高效地计算 LSE
        # 这意味着参考版本计算 LSE 的基数是 e，而编译版本的基数是 2。我们使用基数转换公式进行比较
        # log_2(x_compiled) = log_e(x_ref) * log_2(e) 其中
        # x_ref      = sum(_i e^(scores[i]))
        # x_compiled = sum(_i 2^(log2(e) * scores[i]))

        # 确保 ref_lse 和 compiled_lse 的数据类型为 torch.float64 和 torch.float32
        self.assertTrue(ref_lse.dtype == torch.float64)
        self.assertTrue(compiled_lse.dtype == torch.float32)
        # 计算 ref_lse 的改变基数为 2 的值
        ref_lse = ref_lse * torch.log2(torch.tensor(torch.e))

        # 设置容差
        tolerance = Tolerances(atol=2e-2, rtol=2e-2)
        # 使用容差进行张量值的比较
        torch.testing.assert_close(
            ref_out.to(dtype=torch.float32),
            compiled_out.to(dtype=torch.float32),
            atol=tolerance.atol,
            rtol=tolerance.rtol,
        )
        torch.testing.assert_close(
            ref_lse.to(dtype=torch.float32),
            compiled_lse.to(dtype=torch.float32),
            atol=tolerance.atol,
            rtol=tolerance.rtol,
        )

    @supported_platform
    # 定义一个测试函数，用于测试仅返回 logsumexp 的情况
    def test_logsumexp_only_return(self):
        # 创建一个部分应用函数，用于生成指定形状的随机张量
        make_tensor = functools.partial(
            torch.randn,
            (B, H, S, D),  # 张量形状为 B x H x S x D
            dtype=torch.float32,  # 张量数据类型为 float32
            device="cuda",  # 张量存储设备为 CUDA GPU
            requires_grad=True,  # 张量需要计算梯度
        )
        # 创建三个随机张量 q, k, v
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        # 使用 q, k, v 创建一个空的块稀疏掩码
        block_mask = _create_empty_block_sparse_mask(q, k, v)

        # 定义一个编译 Torch 函数的装饰器
        @torch.compile
        def func(q, k, v, score_mod, block_mask):
            # 调用灵活注意力函数，获取输出和 logsumexp 值
            _, lse = flex_attention_hop(
                q,
                k,
                v,
                score_mod,
                block_mask.kv_num_blocks,
                block_mask.kv_indices,
                block_mask.q_num_blocks,
                block_mask.q_indices,
                block_mask.KV_BLOCK_SIZE,
                block_mask.Q_BLOCK_SIZE,
            )
            # 将 logsumexp 值乘以 2
            lse_2 = lse * 2
            # 返回乘以 2 后的 logsumexp 值
            return lse_2

        # 运行并获取 func 函数的代码和结果
        _, code = run_and_get_code(func, q, k, v, _identity, block_mask)
        # 确保生成了两个内核
        FileCheck().check_count(".run(", 2, True).run(code[0])

    # 定义一个测试函数，用于测试 logsumexp 没有被融合的情况
    @supported_platform
    def test_logsumexp_is_not_fused(self):
        # 创建一个部分应用函数，用于生成指定形状的随机张量
        make_tensor = functools.partial(
            torch.randn,
            (B, H, S, D),  # 张量形状为 B x H x S x D
            dtype=torch.float32,  # 张量数据类型为 float32
            device="cuda",  # 张量存储设备为 CUDA GPU
            requires_grad=True,  # 张量需要计算梯度
        )
        # 创建三个随机张量 q, k, v
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        # 使用 q, k, v 创建一个空的块稀疏掩码
        block_mask = _create_empty_block_sparse_mask(q, k, v)

        # 定义一个编译 Torch 函数的装饰器
        @torch.compile
        def func(q, k, v, score_mod, block_mask):
            # 调用灵活注意力函数，获取输出和 logsumexp 值
            out, lse = flex_attention_hop(
                q,
                k,
                v,
                score_mod,
                block_mask.kv_num_blocks,
                block_mask.kv_indices,
                block_mask.q_num_blocks,
                block_mask.q_indices,
                block_mask.KV_BLOCK_SIZE,
                block_mask.Q_BLOCK_SIZE,
            )
            # 将 logsumexp 值乘以 2
            lse_2 = lse * 2
            # 返回输出和乘以 2 后的 logsumexp 值
            return out, lse_2

        # 运行并获取 func 函数的代码和结果
        _, code = run_and_get_code(func, q, k, v, _identity, block_mask)
        # 确保生成了两个内核
        FileCheck().check_count(".run(", 2, True).run(code[0])

    # 定义一个测试函数，用于测试 AOT (Ahead-of-Time) 和 Eager 模式下的梯度检查
    @supported_platform
    @common_utils.parametrize(
        "score_mod", [_identity, _causal, _times_two, _squared, _trig, _trig2]
    )
    def test_aot_eager_gradcheck(self, score_mod):
        # 创建一个部分应用函数，用于生成指定形状的随机张量
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 8, 4),  # 张量形状为 2 x 2 x 8 x 4
            device="cuda",  # 张量存储设备为 CUDA GPU
            dtype=torch.float64,  # 张量数据类型为 float64
            requires_grad=True,  # 张量需要计算梯度
        )
        # 创建三个随机张量 query, key, value
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        # 编译 _flex_attention 函数，使用 AOT Eager 后端并生成完整图
        func = torch.compile(_flex_attention, backend="aot_eager", fullgraph=True)

        # 断言在 func 函数上进行梯度检查是正确的
        self.assertTrue(
            torch.autograd.gradcheck(
                func, (query, key, value, score_mod), raise_exception=True
            )
        )

    # 定义一个测试函数，用于测试是否支持 AOT Eager 平台，同时参数化 score_mod_name 和 mode
    @supported_platform
    @common_utils.parametrize("score_mod_name", ["_head_offset"])
    @common_utils.parametrize("mode", ["eager", "aot_eager"])
    # 定义测试函数，用于测试指定得分模块和模式的梯度检查
    def test_captured_score_mod_aot_eager_gradcheck(
        self, score_mod_name: str, mode: str
    ):
        # 创建一个部分应用了随机正态分布生成函数的函数，生成CUDA上的张量，数据类型为float64，并且需要梯度
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 8, 4),
            device="cuda",
            dtype=torch.float64,
            requires_grad=True,
        )
        # 生成查询(query)、关键字(key)、值(value)张量
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        # 编译灵活注意力函数(_flex_attention)，使用指定的后端(mode)，并全图编译
        func = torch.compile(_flex_attention, backend=mode, fullgraph=True)
        
        # 根据给定的得分模块名称从captured_buffers_map中获取得分模块，并使用float64类型
        score_mod = captured_buffers_map[score_mod_name](torch.float64)

        # 断言梯度检查结果为True，即检查函数func关于(query, key, value, score_mod)的梯度是否正确
        self.assertTrue(
            torch.autograd.gradcheck(
                func, (query, key, value, score_mod), raise_exception=True
            )
        )

    # 声明一个装饰器，表明该测试函数支持的平台
    @supported_platform
    def test_fw_bw_graph_correctness(self):
        # 创建一个带有后端计数器的编译计数器对象，后端为"aot_eager"
        cnt = CompileCounterWithBackend("aot_eager")
        
        # 创建一个部分应用了随机正态分布生成函数的函数，生成CUDA上的张量，数据类型为float64，并且需要梯度
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 8, 4),
            device="cuda",
            dtype=torch.float64,
            requires_grad=True,
        )
        # 生成查询(query)、关键字(key)、值(value)张量
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        # 编译灵活注意力函数(_flex_attention)，使用cnt作为后端，并全图编译
        func = torch.compile(_flex_attention, backend=cnt, fullgraph=True)
        
        # 调用编译后的函数func，传入(query, key, value, _squared)，并获取输出
        out = func(query, key, value, _squared)
        
        # 对输出进行求和并进行反向传播
        out.sum().backward()
        
        # 断言编译计数器的帧计数为1
        self.assertEqual(cnt.frame_count, 1)
        
        # 断言编译计数器记录的图数量为1
        self.assertEqual(len(cnt.graphs), 1)
        
        # 获取编译后的图形，并对其进行规范化处理，使其更易读
        graph = cnt.graphs[0]
        norm_graph = normalize_gm(graph.print_readable(print_output=False))

        # 断言规范化后的图形与预期的内联字符串(norm_graph)相等
        self.assertExpectedInline(
            norm_graph,
            """\
# 定义一个名为 GraphModule 的类，继承自 torch.nn.Module
class GraphModule(torch.nn.Module):
    # 定义类的前向传播方法，接受三个输入参数
    def forward(self, L_args_0_: "f64[2, 2, 8, 4]", L_args_1_: "f64[2, 2, 8, 4]", L_args_2_: "f64[2, 2, 8, 4]"):
        # 将输入参数赋值给局部变量
        l_args_0_ = L_args_0_
        l_args_1_ = L_args_1_
        l_args_2_ = L_args_2_

        # 创建一个在 CUDA 设备上的整数类型的张量 ones，形状为 [1, 1, 1]
        ones: "i32[1, 1, 1]" = torch.ones([1, 1, 1], dtype=torch.int32, device=device(type='cuda', index=0))

        # 创建一个在 CUDA 设备上的整数类型的张量 zeros，形状为 [1, 1, 1, 1]
        zeros: "i32[1, 1, 1, 1]" = torch.zeros([1, 1, 1, 1], dtype=torch.int32, device=device(type='cuda', index=0))

        # 创建另一个在 CUDA 设备上的整数类型的张量 ones_1，形状为 [1, 1, 1]
        ones_1: "i32[1, 1, 1]" = torch.ones([1, 1, 1], dtype=torch.int32, device=device(type='cuda', index=0))

        # 创建另一个在 CUDA 设备上的整数类型的张量 zeros_1，形状为 [1, 1, 1, 1]
        zeros_1: "i32[1, 1, 1, 1]" = torch.zeros([1, 1, 1, 1], dtype=torch.int32, device=device(type='cuda', index=0))

        # 创建一个在 CUDA 设备上的空的浮点类型张量 new_empty，标记为需要梯度计算
        new_empty: "f64[]" = l_args_0_.new_empty([], requires_grad=True)

        # 创建一个在 CUDA 设备上的空的整数类型张量 new_empty_1
        new_empty_1: "i32[]" = l_args_0_.new_empty([], dtype=torch.int32)

        # 创建另外三个在 CUDA 设备上的空的整数类型张量 new_empty_2, new_empty_3, new_empty_4
        new_empty_2: "i32[]" = l_args_0_.new_empty([], dtype=torch.int32)
        new_empty_3: "i32[]" = l_args_0_.new_empty([], dtype=torch.int32)
        new_empty_4: "i32[]" = l_args_0_.new_empty([], dtype=torch.int32)

        # 获取当前对象的 flex_attention_0 属性
        flex_attention_0 = self.flex_attention_0
        
        # 调用 torch.ops.higher_order.flex_attention 方法进行灵活的注意力计算
        # 并将计算结果赋给 flex_attention，同时将所有输入变量置为 None
        flex_attention = torch.ops.higher_order.flex_attention(l_args_0_, l_args_1_, l_args_2_, flex_attention_0, ones, zeros, ones_1, zeros_1, 8, 8);
        l_args_0_ = l_args_1_ = l_args_2_ = flex_attention_0 = ones = zeros = ones_1 = zeros_1 = None
        
        # 从 flex_attention 的输出中获取第一个元素，赋给 out，并将 flex_attention 置为 None
        out: "f64[2, 2, 8, 4]" = flex_attention[0];
        flex_attention = None
        
        # 返回一个包含 out 的元组
        return (out,)
    
    # 嵌套定义 GraphModule 类，继承自 torch.nn.Module
    class GraphModule(torch.nn.Module):
        # 定义类的前向传播方法，接受五个输入参数
        def forward(self, new_empty: "f64[]", new_empty_1: "i32[]", new_empty_2: "i32[]", new_empty_3: "i32[]", new_empty_4: "i32[]"):
            # 计算 new_empty 张量的平方，并将结果赋给 mul，同时将 new_empty 置为 None
            mul: "f64[]" = new_empty * new_empty;
            new_empty = None
            
            # 返回计算结果 mul
            return mul
    # 定义一个方法 `forward`，接受多个输入参数，包括 primals 和 tangents 的张量，以及其它默认值和获取项
    def forward(self, primals_1: "f64[2, 2, 8, 4]", primals_2: "f64[2, 2, 8, 4]", primals_3: "f64[2, 2, 8, 4]", full_default: "i32[1, 1, 1]", full_default_1: "i32[1, 1, 1, 1]", getitem: "f64[2, 2, 8, 4]", getitem_1: "f32[2, 2, 8]", tangents_1: "f64[2, 2, 8, 4]"):
        # 将当前对象中的 `fw_graph` 赋值给局部变量 `fw_graph`
        fw_graph = self.fw_graph
        # 将当前对象中的 `joint_graph` 赋值给局部变量 `joint_graph`
        joint_graph = self.joint_graph
        # 调用 torch 的自定义操作 `flex_attention_backward` 进行高阶求导，返回多个结果
        flex_attention_backward = torch.ops.higher_order.flex_attention_backward(primals_1, primals_2, primals_3, getitem, getitem_1, tangents_1, fw_graph, joint_graph, full_default, full_default_1, full_default, full_default_1, 8, 8)
        # 释放各个输入参数的引用，以便内存管理
        primals_1 = primals_2 = primals_3 = getitem = getitem_1 = tangents_1 = fw_graph = joint_graph = full_default = full_default_1 = None
        # 从 `flex_attention_backward` 结果中获取特定的张量数据，分别为 `getitem_2`, `getitem_3`, `getitem_4`
        getitem_2: "f64[2, 2, 8, 4]" = flex_attention_backward[0]
        getitem_3: "f64[2, 2, 8, 4]" = flex_attention_backward[1]
        getitem_4: "f64[2, 2, 8, 4]" = flex_attention_backward[2]
        # 释放 `flex_attention_backward` 的引用，以便内存管理
        flex_attention_backward = None
        # 返回一个包含 `getitem_2`, `getitem_3`, `getitem_4` 的列表作为结果
        return [getitem_2, getitem_3, getitem_4]

    # 定义一个匿名类 `<lambda>` 继承自 `torch.nn.Module`
    class <lambda>(torch.nn.Module):
        # 定义匿名类 `<lambda>` 的前向计算方法 `forward`，接受一组输入参数
        def forward(self, arg0_1: "f64[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            # 使用 torch 的 `mul` 操作对 `arg0_1` 进行乘法运算，结果保存在 `mul` 中
            mul: "f64[]" = torch.ops.aten.mul.Tensor(arg0_1, arg0_1)
            # 释放 `arg0_1` 的引用，以便内存管理
            arg0_1 = None
            # 返回 `mul` 作为计算结果
            return mul

    # 定义一个匿名类 `<lambda>` 继承自 `torch.nn.Module`
    class <lambda>(torch.nn.Module):
        # 定义匿名类 `<lambda>` 的前向计算方法 `forward`，接受一组输入参数
        def forward(self, arg0_1: "f64[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]", arg5_1: "f64[]"):
            # 使用 torch 的 `mul` 操作对 `arg0_1` 进行乘法运算，结果保存在 `mul` 中
            mul: "f64[]" = torch.ops.aten.mul.Tensor(arg0_1, arg0_1)
            # 使用 torch 的 `mul` 操作对 `arg5_1` 和 `arg0_1` 进行乘法运算，结果保存在 `mul_1` 中
            mul_1: "f64[]" = torch.ops.aten.mul.Tensor(arg5_1, arg0_1)
            # 使用 torch 的 `mul` 操作对 `arg5_1` 和 `arg0_1` 进行乘法运算，结果保存在 `mul_2` 中
            mul_2: "f64[]" = torch.ops.aten.mul.Tensor(arg5_1, arg0_1)
            # 释放 `arg5_1` 和 `arg0_1` 的引用，以便内存管理
            arg5_1 = arg0_1 = None
            # 使用 torch 的 `add` 操作对 `mul_2` 和 `mul_1` 进行加法运算，结果保存在 `add` 中
            add: "f64[]" = torch.ops.aten.add.Tensor(mul_2, mul_1)
            # 释放 `mul_2` 和 `mul_1` 的引用，以便内存管理
            mul_2 = mul_1 = None
            # 返回包含 `add` 和多个 `None` 的列表作为计算结果
            return [add, None, None, None, None]
"""
这部分代码片段主要是进行测试相关的操作。

common_utils.instantiate_parametrized_tests(TestFlexAttention)
# 使用 common_utils 模块中的函数 instantiate_parametrized_tests 来实例化参数化测试类 TestFlexAttention。

if __name__ == "__main__":
    # 如果当前脚本作为主程序运行（而不是被导入其他模块）
    
    from torch._inductor.test_case import run_tests
    # 从 torch._inductor.test_case 模块中导入 run_tests 函数，用于运行测试用例。
    
    run_tests()
    # 执行测试用例。
```