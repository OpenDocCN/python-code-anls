# `.\pytorch\test\inductor\test_fused_attention.py`

```py
# Owner(s): ["module: inductor"]
# 导入必要的模块和库
import functools
import itertools
import math

import torch
import torch._inductor.config  # 导入 torch._inductor.config 模块
import torch.utils.checkpoint  # 导入 torch.utils.checkpoint 模块
from torch._dynamo.debug_utils import aot_graph_input_parser  # 导入 aot_graph_input_parser 函数
from torch._dynamo.utils import counters  # 导入 counters 模块
from torch._inductor.test_case import run_tests, TestCase  # 导入 run_tests 和 TestCase 类
from torch._inductor.utils import run_and_get_code  # 导入 run_and_get_code 函数
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FUSED_ATTENTION,  # 导入 PLATFORM_SUPPORTS_FUSED_ATTENTION 常量
    SM80OrLater,  # 导入 SM80OrLater 常量
)
from torch.testing._internal.common_utils import IS_LINUX, skipIfRocm  # 导入 IS_LINUX 和 skipIfRocm 常量
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA  # 导入 HAS_CPU 和 HAS_CUDA 常量


def checkpoint_wrapper(fn):
    # 定义装饰器函数 checkpoint_wrapper，用于实现函数的 checkpoint 化
    def inner(*args):
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=True)
    return inner


class TestSDPAPatternRewriterTemplate(TestCase):
    use_static_shapes = True  # 设置类属性 use_static_shapes 为 True

    def _clone_inputs(self, inputs):
        # 定义内部函数 _clone_inputs 用于克隆输入列表中的张量对象
        def clone(x):
            if not isinstance(x, torch.Tensor):
                return x
            return x.clone()  # 克隆张量 x

        return [clone(x) for x in inputs]  # 返回输入列表中每个元素的克隆副本列表

    def _check_common(
        self,
        dot_prod_attention,
        args1=None,
        contains=True,
        atol=1e-5,
        has_fuse_pattern=True,
        has_dropout=False,
        check_train=True,
        override_check_equal=False,
        dtype=torch.float,
        rtol=1.3e-6,
    ):
        # 如果 args1 为 None，则创建一个默认的 tensor_shape 并生成包含三个随机张量的 args1 列表
        if args1 is None:
            tensor_shape = (4, 2, 16, 32)
            args1 = [
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
            ]
        else:
            # 如果 args1 不为 None，则将其转换为列表形式
            args1 = list(args1)
        
        # 使用 _clone_inputs 方法克隆 args1 列表生成 args2
        args2 = self._clone_inputs(args1)

        # 根据 check_train 的值选择是否进行训练循环
        for training in [False, True] if check_train else [False]:
            # 遍历 args1 和 args2 中的张量，如果是浮点数类型的 Tensor，则根据 training 设置 requires_grad
            for x in itertools.chain(args1[:], args2[:]):
                if isinstance(x, torch.Tensor) and x.is_floating_point():
                    x.requires_grad = training

            # 如果不使用静态形状，则标记 args2 中的张量为动态张量
            if not self.use_static_shapes:
                torch._dynamo.mark_dynamic(args2[0], 0)
                torch._dynamo.mark_dynamic(args2[1], 0)
                torch._dynamo.mark_dynamic(args2[2], 0)

            # 如果具有 dropout，则在 dropout_arg 中添加 training 参数；设置随机种子；调用 dot_prod_attention 函数计算结果 result1
            dropout_arg = [training] if has_dropout else []
            torch.manual_seed(1234)
            result1 = dot_prod_attention(*(args1 + dropout_arg))

            # 清空 counters 计数器；设置随机种子；编译并运行 dot_prod_attention 函数，获取结果 result2 和源代码 source_code
            counters.clear()
            torch.manual_seed(1234)
            result2, source_code = run_and_get_code(
                torch.compile(dot_prod_attention, fullgraph=True),
                *(args2 + dropout_arg),
            )
            source_code = "\n".join(source_code)
            
            # 如果具有融合模式，断言 counters 中 "inductor" 下的 "fuse_attention" 大于等于 1
            if has_fuse_pattern:
                self.assertGreaterEqual(counters["inductor"]["fuse_attention"], 1)
            
            # 如果包含指定模式，断言 "aten._scaled_dot_product" 在源代码 source_code 中
            if contains:
                # many of the patterns get re-expanded in dispatcher
                self.assertIn(
                    "aten._scaled_dot_product",
                    source_code,
                )

            # 如果没有 dropout 或者 override_check_equal 为 True，则断言 result1 等于 result2，使用指定的 atol 和 rtol 进行比较
            if not has_dropout or override_check_equal:
                self.assertEqual(result1, result2, atol=atol, rtol=1.3e-6)

            # 如果处于训练状态，计算 result1 和 result2 的梯度，并断言它们相等；使用指定的 atol 和 rtol 进行比较
            if training:
                result1.sum().backward()
                result2.sum().backward()
                for arg1, arg2 in zip(args1, args2):
                    if (
                        isinstance(arg1, torch.Tensor)
                        and arg1.is_floating_point()
                        and (not has_dropout or override_check_equal)
                    ):
                        self.assertEqual(arg1.grad, arg2.grad, atol=atol, rtol=rtol)
    def _test_sdpa_rewriter_1(self):
        # 定义 dot-product 注意力函数，输入张量形状为 (batch_size, n_head, seq_len, embed_dim)
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
            # 计算注意力权重：query 和 key 的点积，除以 sqrt(key 的最后一维的大小)，然后进行 softmax 操作，最后与 value 相乘
            return (
                torch.matmul(query, key.transpose(-2, -1))
                .div(math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .matmul(value)
            )

        # 遍历数据类型列表 [torch.float, torch.half]
        for dtype in [torch.float, torch.half]:
            atol = 0.001
            # 根据数据类型设置相对误差容限
            rtol = 1.3e-6 if dtype == torch.float else 0.7
            # 如果在 CPU 上且数据类型是 torch.half，则调整绝对误差容限和相对误差容限
            if self.device == "cpu" and dtype == torch.half:
                atol = 2e-3
                rtol = 1e-2
            # 调用通用测试函数 _check_common，传入 dot_prod_attention 函数和相应的参数
            self._check_common(dot_prod_attention, dtype=dtype, atol=atol, rtol=rtol)
            # 使用检查点包装器对 dot_prod_attention 函数进行测试
            self._check_common(
                checkpoint_wrapper(dot_prod_attention),
                dtype=dtype,
                atol=atol,
                rtol=rtol,
            )

    @skipIfRocm
    @torch._inductor.config.patch("freezing", True)
    def _test_sdpa_rewriter_1_freezing(self):
        # 定义 dot-product 注意力函数，输入张量形状为 (batch_size, n_head, seq_len, embed_dim)
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
            # 计算注意力权重：query 和 key 的点积，除以 sqrt(key 的最后一维的大小)，然后进行 softmax 操作，最后与 value 相乘
            return (
                torch.matmul(query, key.transpose(-2, -1))
                .div(math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .matmul(value)
            )

        # 遍历数据类型列表 [torch.float, torch.half]
        for dtype in [torch.float, torch.half]:
            atol = 0.001
            # 根据数据类型设置相对误差容限
            rtol = 1.3e-6 if dtype == torch.float else 0.7
            # 如果在 CPU 上且数据类型是 torch.half，则调整绝对误差容限和相对误差容限
            if self.device == "cpu" and dtype == torch.half:
                atol = 2e-3
                rtol = 1e-2
            # 在没有梯度的情况下，调用通用测试函数 _check_common，传入 dot_prod_attention 函数和相应的参数
            with torch.no_grad():
                self._check_common(
                    dot_prod_attention,
                    dtype=dtype,
                    atol=atol,
                    rtol=rtol,
                    check_train=False,
                )

    @skipIfRocm
    def _test_pattern_fails_with_reuse(self):
        """
        This test checks that the replacement is not done
        when an intermediate result is being used / returned downstream
        """

        @torch.compile(fullgraph=True)
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            # 计算注意力权重
            attn_weights = (
                torch.matmul(query, key.transpose(-2, -1))  # 执行矩阵乘法
                .div(math.sqrt(key.shape[-1]))  # 归一化处理
                .softmax(dim=-1)  # 应用 softmax 函数
            )
            return attn_weights.matmul(value), attn_weights  # 返回加权后的值和注意力权重

        tensor_shape = (2, 4, 8, 16)
        args = [
            torch.randn(tensor_shape, device=self.device),  # 生成随机张量作为输入
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
        ]
        _, (source_code,) = run_and_get_code(dot_prod_attention, *args)  # 运行函数获取源代码
        self.assertNotIn("aten._scaled_dot_product_efficient_attention", source_code)  # 检查源代码中是否包含特定字符串

    @skipIfRocm
    def _test_sdpa_rewriter_2(self):
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            return (
                torch.matmul(query, key.transpose(-2, -1))  # 执行矩阵乘法
                .mul(1.0 / math.sqrt(key.shape[-1]))  # 归一化处理
                .softmax(dim=-1)  # 应用 softmax 函数
                .matmul(value)  # 返回加权后的值
            )

        self._check_common(dot_prod_attention)  # 调用公共函数检查 dot_prod_attention 的行为
        self._check_common(checkpoint_wrapper(dot_prod_attention))  # 调用公共函数检查经过 checkpoint 包装后的 dot_prod_attention 的行为

    @skipIfRocm  # 条件为真时跳过测试
    def _test_sdpa_rewriter_3(self):
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, training: bool
        ) -> torch.Tensor:
            return torch.nn.functional.dropout(
                torch.matmul(query, key.transpose(-2, -1)).div(3.0).softmax(dim=-1),  # 执行矩阵乘法、归一化处理和 softmax，然后应用 dropout
                p=0.4,  # dropout 概率
                training=training,  # 是否在训练模式下应用 dropout
                inplace=False,
            ).matmul(value)  # 返回加权后的值

        self._check_common(dot_prod_attention, contains=False, has_dropout=True)  # 调用公共函数检查 dot_prod_attention 的行为，确保没有特定字符串，并且存在 dropout
        self._check_common(
            checkpoint_wrapper(dot_prod_attention), contains=False, has_dropout=True
        )  # 调用公共函数检查经过 checkpoint 包装后的 dot_prod_attention 的行为，确保没有特定字符串，并且存在 dropout

    @skipIfRocm  # 条件为真时跳过测试
    def _test_sdpa_rewriter_4(self):
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            training: bool,
        ) -> torch.Tensor:
            return torch.nn.functional.dropout(
                torch.matmul(query, key.transpose(-2, -1)).mul(0.4).softmax(dim=-1),  # 执行矩阵乘法、缩放、softmax，然后应用 dropout
                p=0.2,  # dropout 概率
                inplace=False,
                training=training,  # 是否在训练模式下应用 dropout
            ).matmul(value)  # 返回加权后的值

        self._check_common(dot_prod_attention, contains=False, has_dropout=True)  # 调用公共函数检查 dot_prod_attention 的行为，确保没有特定字符串，并且存在 dropout
        self._check_common(
            checkpoint_wrapper(dot_prod_attention), contains=False, has_dropout=True
        )  # 调用公共函数检查经过 checkpoint 包装后的 dot_prod_attention 的行为，确保没有特定字符串，并且存在 dropout
    def _test_sdpa_rewriter_5(self):
        # 定义模式函数 sfdp_pattern_5_v1，计算注意力权重并返回加权后的值
        def sfdp_pattern_5_v1(query, key, value):
            # 创建全为 True 的注意力掩码矩阵，对角线及以下为 True，其余为 False
            attn_mask = torch.ones(
                query.size(-2), key.size(-2), dtype=torch.bool, device=query.device
            ).tril(diagonal=0)
            # 将掩码中的 False 值填充为负无穷大
            attn_mask = attn_mask.masked_fill(
                torch.logical_not(attn_mask), -float("inf")
            )
            # 计算注意力权重，对查询和键的点积除以缩放系数后加上注意力掩码，进行 softmax 处理
            attn_weight = torch.softmax(
                (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask,
                dim=-1,
            )
            return attn_weight @ value

        # 定义模式函数 sfdp_pattern_5_v2，参考 GitHub 上的问题修正版本
        def sfdp_pattern_5_v2(query, key, value):
            # 创建全为 False 的注意力掩码矩阵
            attn_mask = torch.zeros(
                query.size(-2), key.size(-2), dtype=torch.bool, device=query.device
            ).bool()
            # 计算注意力权重，对查询和键的点积除以缩放系数后加上注意力掩码，进行 softmax 处理
            attn_weight = torch.softmax(
                (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask,
                dim=-1,
            )
            return attn_weight @ value

        # 使用 self._check_common 方法测试 sfdp_pattern_5_v1，不应包含 dropout 操作
        self._check_common(sfdp_pattern_5_v1, contains=False)
        # 使用 self._check_common 方法测试经过检查点封装后的 sfdp_pattern_5_v1，不应包含 dropout 操作
        self._check_common(checkpoint_wrapper(sfdp_pattern_5_v1), contains=False)
        # 使用 self._check_common 方法测试 sfdp_pattern_5_v2，不应包含 dropout 操作
        self._check_common(sfdp_pattern_5_v2, contains=False)
        # 使用 self._check_common 方法测试经过检查点封装后的 sfdp_pattern_5_v2，不应包含 dropout 操作
        self._check_common(checkpoint_wrapper(sfdp_pattern_5_v2), contains=False)

    @skipIfRocm
    def _test_sdpa_rewriter_6(self):
        # 定义模式函数 sfdp_pattern_6，计算注意力权重并返回加权后的值，包含 dropout 操作
        def sfdp_pattern_6(query, key, value, training):
            # 创建全为 True 的注意力掩码矩阵，对角线及以下为 True，其余为 False
            attn_mask = torch.ones(
                query.size(-2), key.size(-2), dtype=torch.bool, device=query.device
            ).tril(diagonal=0)
            # 将掩码中的 False 值填充为负无穷大
            attn_mask = attn_mask.masked_fill(
                torch.logical_not(attn_mask), -float("inf")
            )
            # 计算注意力权重，对查询和键的点积除以缩放系数后加上注意力掩码，进行 softmax 处理
            attn_weight = torch.softmax(
                (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask,
                dim=-1,
            )
            # 在训练模式下对注意力权重应用 dropout 操作
            attn_weight = torch.nn.functional.dropout(attn_weight, 0.5, training)
            return attn_weight @ value

        # 使用 self._check_common 方法测试 sfdp_pattern_6，不应包含 dropout 操作但应包含 dropout 相关标志
        self._check_common(sfdp_pattern_6, contains=False, has_dropout=True)
        # 使用 self._check_common 方法测试经过检查点封装后的 sfdp_pattern_6，不应包含 dropout 操作但应包含 dropout 相关标志
        self._check_common(
            checkpoint_wrapper(sfdp_pattern_6), contains=False, has_dropout=True
        )

    @skipIfRocm
    def _test_sdpa_rewriter_7(self):
        # 定义函数 sfdp_pattern_7，对输入进行特定维度置换，计算注意力权重并应用 dropout
        def sfdp_pattern_7(query, key, value, training):
            # 对 query 进行维度置换
            q = query.permute(0, 2, 1, 3)
            # 对 key 进行维度置换
            k = key.permute(0, 2, 1, 3)
            # 对 value 进行维度置换
            v = value.permute(0, 2, 1, 3)
            # 计算 scaled dot-product attention 的分数
            div = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
            # 将分数转换为 float32 类型
            div = div.to(torch.float32)
            # 计算 softmax 归一化的注意力权重
            attn_weight = torch.softmax(div, dim=-1)
            # 应用 dropout，丢弃概率设为极小值，根据训练状态决定是否执行 dropout
            attn_weight = torch.dropout(attn_weight, 0.00000000001, training)
            # 将注意力权重转换为 float16 类型
            attn_weight = attn_weight.to(torch.float16)
            # 返回注意力权重加权的 value
            return attn_weight @ v

        # 创建输入参数元组
        args = (
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
        )
        # 调用测试函数 _check_common，检查 sfdp_pattern_7 函数的输出
        self._check_common(
            sfdp_pattern_7,
            args,
            contains=SM80OrLater,  # 检查是否包含 SM80 或更新版本的硬件特性
            has_dropout=True,      # 检查是否包含 dropout 操作
            override_check_equal=True,  # 覆盖默认的相等性检查行为
            atol=2e-3,             # 绝对误差的容忍度设置为 0.002
        )

        # 创建另一组输入参数元组，设备为 CUDA
        args = (
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
        )
        # 调用测试函数 _check_common，检查经过 checkpoint_wrapper 处理后的 sfdp_pattern_7 函数的输出
        self._check_common(
            checkpoint_wrapper(sfdp_pattern_7),
            args,
            contains=SM80OrLater,
            has_dropout=True,
            override_check_equal=True,
            atol=2e-3,
        )

    @skipIfRocm
    def _test_sdpa_rewriter_8(self):
        # 定义函数 sfdp_pattern_8，对输入进行特定维度置换，计算注意力权重
        def sfdp_pattern_8(query, key, value):
            # 对 query 进行维度置换
            q = query.permute(0, 2, 1, 3)
            # 对 key 进行维度置换
            k = key.permute(0, 2, 1, 3)
            # 对 value 进行维度置换
            v = value.permute(0, 2, 1, 3)
            # 计算 scaled dot-product attention 的分数
            div = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
            # 将分数转换为 float32 类型
            div = div.to(torch.float32)
            # 计算 softmax 归一化的注意力权重
            attn_weight = torch.softmax(div, dim=-1)
            # 将注意力权重转换为 float16 类型
            attn_weight = attn_weight.to(torch.float16)
            # 返回注意力权重加权的 value
            return attn_weight @ v

        # 创建输入参数元组
        args = (
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
        )
        # 调用测试函数 _check_common，检查 sfdp_pattern_8 函数的输出
        self._check_common(sfdp_pattern_8, args, atol=2e-3)

        # 创建另一组输入参数元组，设备为 CUDA
        args = (
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
        )
        # 调用测试函数 _check_common，检查经过 checkpoint_wrapper 处理后的 sfdp_pattern_8 函数的输出
        self._check_common(
            checkpoint_wrapper(sfdp_pattern_8),
            args,
            atol=2e-3,
        )
    def _test_sdpa_rewriter_9(self):
        # 定义一个函数，实现 sfdp_pattern_9 模式，用于处理查询、键、值和训练标志
        def sfdp_pattern_9(query, key, value, training):
            # 对查询张量进行维度置换，将第一个维度和第三个维度交换位置
            q = query.permute(0, 2, 1, 3)
            # 对键张量进行维度置换，将第一个维度和第三个维度交换位置
            k = key.permute(0, 2, 1, 3)
            # 对值张量进行维度置换，将第一个维度和第三个维度交换位置
            v = value.permute(0, 2, 1, 3)
            # 将查询张量除以其最后一个维度的平方根，用于缩放
            q = q / math.sqrt(q.size(-1))
            # 计算注意力权重，通过查询张量与键张量的转置矩阵相乘得到分数
            div = q @ k.transpose(-2, -1)
            # 转换分数张量的数据类型为 float32
            div = div.to(torch.float32)
            # 对分数进行 softmax 操作，得到归一化的注意力权重
            attn_weight = torch.softmax(div, dim=-1)
            # 非常低的 dropout 概率，用于测试通过
            attn_weight = torch.dropout(attn_weight, 0.00000000001, training)
            # 将注意力权重的数据类型转换为 float16
            attn_weight = attn_weight.to(torch.float16)
            # 返回加权后的值张量
            return attn_weight @ v

        # 定义输入参数
        args = (
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
        )
        # 执行公共检查方法，验证 sfdp_pattern_9 函数的行为
        self._check_common(
            sfdp_pattern_9,
            args,
            contains=SM80OrLater,  # 包含 SM80 或更高的硬件架构
            has_dropout=True,       # 函数包含 dropout 操作
            override_check_equal=True,  # 覆盖检查相等的行为
            atol=2e-3,              # 允许的误差范围为 2e-3
        )

        # 重新定义输入参数，指定在 CUDA 设备上进行测试
        args = (
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
        )
        # 执行公共检查方法，验证经过 checkpoint_wrapper 封装后的 sfdp_pattern_9 函数在 CUDA 上的行为
        self._check_common(
            checkpoint_wrapper(sfdp_pattern_9),
            args,
            contains=SM80OrLater,  # 包含 SM80 或更高的硬件架构
            has_dropout=True,       # 函数包含 dropout 操作
            override_check_equal=True,  # 覆盖检查相等的行为
            atol=2e-3,              # 允许的误差范围为 2e-3
        )

    @skipIfRocm
    def _test_sdpa_rewriter_10(self):
        # 定义一个函数，实现 sfdp_pattern_10 模式，用于处理查询、键、值
        def sfdp_pattern_10(query, key, value):
            # 对查询张量进行维度置换，将第一个维度和第三个维度交换位置
            q = query.permute(0, 2, 1, 3)
            # 对键张量进行维度置换，将第一个维度和第三个维度交换位置
            k = key.permute(0, 2, 1, 3)
            # 对值张量进行维度置换，将第一个维度和第三个维度交换位置
            v = value.permute(0, 2, 1, 3)
            # 将查询张量除以其最后一个维度的平方根，用于缩放
            q = q / math.sqrt(q.size(-1))
            # 计算注意力权重，通过查询张量与键张量的转置矩阵相乘得到分数
            div = q @ k.transpose(-2, -1)
            # 转换分数张量的数据类型为 float32
            div = div.to(torch.float32)
            # 对分数进行 softmax 操作，得到归一化的注意力权重
            attn_weight = torch.softmax(div, dim=-1)
            # 将注意力权重的数据类型转换为 float16
            attn_weight = attn_weight.to(torch.float16)
            # 返回加权后的值张量
            return attn_weight @ v

        # 定义输入参数
        args = (
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
        )
        # 执行公共检查方法，验证 sfdp_pattern_10 函数的行为
        self._check_common(sfdp_pattern_10, args, atol=2e-3)

        # 重新定义输入参数，指定在 CUDA 设备上进行测试
        args = (
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
        )
        # 执行公共检查方法，验证经过 checkpoint_wrapper 封装后的 sfdp_pattern_10 函数在 CUDA 上的行为
        self._check_common(
            checkpoint_wrapper(sfdp_pattern_10),
            args,
            atol=2e-3,              # 允许的误差范围为 2e-3
        )
    def _test_pattern_fails_with_tensor_factor(self):
        # https://github.com/pytorch/pytorch/issues/99124
        # 定义一个模型类，用于测试张量因子相关的失败模式
        class Model(torch.nn.Module):
            def __init__(self, is_inv_factor):
                super().__init__()
                self.is_inv_factor = is_inv_factor

            def forward(self, query, key, value, scale_factor) -> torch.Tensor:
                # 将 scale_factor 分离出来以稳定梯度
                scale_factor = scale_factor.detach()
                y = torch.matmul(query, key.transpose(-2, -1))
                if self.is_inv_factor:
                    y = y.div(scale_factor)  # 如果是逆因子模式，进行除法操作
                else:
                    y = y.mul(scale_factor)  # 否则进行乘法操作
                return y.softmax(dim=-1).matmul(value)  # 返回 softmax 处理后的乘积结果

        tensor_shape = (2, 4, 4, 4)
        # 遍历两种 is_inv_factor 的情况
        for is_inv_factor in [True, False]:
            args = [
                torch.randn(tensor_shape, device=self.device),
                torch.randn(tensor_shape, device=self.device),
                torch.randn(tensor_shape, device=self.device),
                torch.randn((4, 1, 1), device=self.device),
            ]
            model = Model(is_inv_factor).eval()
            # 在测试中验证模型行为的公共方法
            self._check_common(
                model, args1=args, contains=False, atol=1e-3, has_fuse_pattern=False
            )

    def _test_pattern_fails_with_unsupported_mask(self):
        if not self.use_static_shapes:
            self.skipTest("Causes shape specialization. TODO: investigate")
        
        # https://github.com/pytorch/pytorch/issues/100315
        # 定义一个模型类，用于测试不支持的掩码（mask）模式的失败情况
        class Model(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()

            def forward(self, query, key, value, attn_mask) -> torch.Tensor:
                attn_weight = torch.softmax(
                    query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
                    + attn_mask,
                    dim=-1,
                )
                return attn_weight @ value

        tensor_shape = (2, 4, 4, 4)

        upsupported_masks = [
            torch.randn((2, 4, 4, 4), device=self.device).to(dtype=torch.int),
            2.0,
        ]
        # 遍历不支持的掩码（mask）列表
        for atte_mask in upsupported_masks:
            args = [
                torch.randn(tensor_shape, device=self.device),
                torch.randn(tensor_shape, device=self.device),
                torch.randn(tensor_shape, device=self.device),
                atte_mask,
            ]
            model = Model().eval()
            # 在测试中验证模型行为的公共方法
            self._check_common(
                model, args1=args, contains=False, atol=1e-4, has_fuse_pattern=False
            )

    @skipIfRocm
    def _test_sdpa_rewriter_11(self):
        # 定义了一个函数 dot_prod_attention，实现了点积注意力机制
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            # 调整输入张量的维度顺序，以便进行矩阵乘法
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            # 计算注意力分数，并应用 softmax 函数，然后乘以 value 张量
            return (
                torch.matmul(q, k.transpose(-2, -1))
                .div(math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .matmul(v)
            )

        # 调用 _check_common 函数，对 dot_prod_attention 进行通用性检查
        self._check_common(dot_prod_attention)

    def _test_sdpa_rewriter_12(self):
        # 定义了一个函数 dot_prod_attention，实现了带有 dropout 的点积注意力机制
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            training: bool,
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            # 计算注意力分数，应用 softmax 和 dropout，然后乘以 value 张量
            return torch.nn.functional.dropout(
                torch.matmul(q, k.transpose(-2, -1))
                .div(math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .matmul(v),
                p=0.4,  # dropout 概率设为 0.4
                training=training,
                inplace=False,
            )

        # 调用 _check_common 函数，对 dot_prod_attention 进行通用性检查，检查是否包含 dropout
        self._check_common(dot_prod_attention, contains=False, has_dropout=True)

    @skipIfRocm
    def _test_sdpa_prev_13(self):
        # 定义了一个函数 dot_prod_attention，实现了点积注意力机制
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
            # 计算注意力分数，并应用 softmax 函数，然后乘以 value 张量，并克隆结果
            return (
                torch.matmul(query, key.transpose(-2, -1))
                .div(math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .clone()
                .matmul(value)
            )

        # 调用 _check_common 函数，对 dot_prod_attention 进行通用性检查，不检查训练模式
        self._check_common(dot_prod_attention, check_train=False)
        # 调用 checkpoint_wrapper 函数包装 dot_prod_attention，并进行通用性检查，不检查训练模式
        self._check_common(checkpoint_wrapper(dot_prod_attention), check_train=False)

    @skipIfRocm
    def _test_sdpa_prev_14(self):
        # 定义了一个函数 dot_prod_attention，实现了点积注意力机制
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            # 计算注意力分数，并应用 softmax 函数，乘以 value 张量，并克隆结果
            return (
                torch.matmul(query, key.transpose(-2, -1))
                .mul(1.0 / math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .clone()
                .matmul(value)
            )

        # 调用 _check_common 函数，对 dot_prod_attention 进行通用性检查，不检查训练模式
        self._check_common(dot_prod_attention, check_train=False)
        # 调用 checkpoint_wrapper 函数包装 dot_prod_attention，并进行通用性检查，不检查训练模式
        self._check_common(checkpoint_wrapper(dot_prod_attention), check_train=False)
    def _test_sdpa_prev_15(self):
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            # 转置 query, key, value 张量的轴，使得形状变为 (batch_size, n_head, seq_len, embed_dim)
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            # 计算注意力权重并进行缩放
            return (
                torch.matmul(q, k.transpose(-2, -1))  # 执行点积操作
                .div(math.sqrt(key.shape[-1]))  # 缩放操作，除以 key 张量的最后一个维度的平方根
                .softmax(dim=-1)  # 在最后一个维度上进行 softmax 操作，得到注意力权重
                .clone()  # 克隆结果张量，以确保在计算后不影响原始张量
                .matmul(v)  # 将注意力权重与 value 张量相乘得到最终输出
            )

        self._check_common(dot_prod_attention, check_train=False)

    @skipIfRocm
    def _test_sdpa_rewriter_13(self, dtype):
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            training: bool,
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            # 计算注意力权重，使用 BMM 进行批量矩阵乘法
            attn_weight = torch.bmm(query, key.transpose(1, 2)).softmax(dim=-1)
            # 在训练时进行 dropout 操作，防止过拟合
            attn_weight = torch.nn.functional.dropout(
                attn_weight, p=0.5, training=training
            )
            # 将注意力权重与 value 张量相乘得到最终输出
            return torch.bmm(attn_weight, value)

        tensor_shape = (4, 8, 16)
        args = [
            torch.randn(tensor_shape, device=self.device, dtype=dtype),
            torch.randn(tensor_shape, device=self.device, dtype=dtype),
            torch.randn(tensor_shape, device=self.device, dtype=dtype),
        ]

        self._check_common(
            dot_prod_attention,
            check_train=False,
            args1=args,
            has_dropout=True,
            override_check_equal=True,
            atol=1e-2,
            rtol=1e-2,
        )

    @skipIfRocm
    def _test_sdpa_rewriter_14(self):
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            # 创建一个下三角矩阵作为注意力掩码
            attn_mask = torch.ones(
                query.size(1), key.size(1), dtype=torch.bool, device=query.device
            ).tril(diagonal=0)
            # 使用掩码填充不需要关注的位置，将其设置为负无穷
            attn_mask = attn_mask.masked_fill(
                torch.logical_not(attn_mask), -float("inf")
            )
            # 调整 query, key, value 张量的轴，使得形状变为 (batch_size, n_head, seq_len, embed_dim)
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            # 计算加权后的注意力权重，并与 value 张量相乘得到最终输出
            return (
                (torch.matmul(q, k.transpose(-2, -1)).div(3.0) + attn_mask)
                .softmax(dim=-1)  # 在最后一个维度上进行 softmax 操作，得到最终的加权结果
                .matmul(v)  # 将加权后的注意力权重与 value 张量相乘得到最终输出
            )

        self._check_common(dot_prod_attention)
    def _test_sdpa_rewriter_15(self):
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            # 转置 query, key, value 张量的第二和第三个维度
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            # 获取 batch size 大小
            bs = q.size(0)
            # 获取 key 的长度
            k_len = k.size(-2)
            # 创建一个下三角形的注意力遮罩张量，用于 self-attention
            attn_mask = torch.ones(
                bs, k_len, dtype=torch.bool, device=query.device
            ).tril(diagonal=0)
            # 计算点积注意力分数
            scores = torch.matmul(q, k.transpose(-2, -1)) / 3.0
            # 根据注意力遮罩，将不需要的位置的分数置为负无穷
            attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
            scores = scores.masked_fill(attn_mask, -float("inf"))
            # 计算注意力权重，进行 softmax 操作
            weights = torch.nn.functional.softmax(scores, dim=-1)
            # 计算加权后的值
            return torch.matmul(weights, v)

        # 调用通用的测试函数，检查 dot_prod_attention 函数
        self._check_common(dot_prod_attention, check_train=False)

    @skipIfRocm
    def _test_sdpa_rewriter_16(self):
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, training
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            # 创建一个下三角形的注意力遮罩张量，用于 self-attention
            attn_mask = torch.ones(
                query.size(1), key.size(1), dtype=torch.bool, device=query.device
            ).tril(diagonal=0)
            attn_mask = attn_mask.masked_fill(
                torch.logical_not(attn_mask), -float("inf")
            )
            # 转置 query, key, value 张量的各个维度
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            # 计算点积注意力分数，并加上注意力遮罩
            return torch.nn.functional.dropout(
                (torch.matmul(q, k.transpose(-2, -1)).div(3.0) + attn_mask).softmax(
                    dim=-1
                ),
                p=0.4,
                training=training,
                inplace=False,
            ).matmul(v)

        # 调用通用的测试函数，检查 dot_prod_attention 函数的特定条件
        self._check_common(dot_prod_attention, contains=False, has_dropout=True)

        # 也检查 batch_size=1 的情况，因为图结构略有不同
        tensor_shape = (1, 2, 16, 32)
        args = [
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
        ]
        self._check_common(
            dot_prod_attention, args1=args, contains=False, has_dropout=True
        )

    @skipIfRocm
    def _test_sdpa_rewriter_16_fp32_mask(self):
        # 定义一个函数 dot_prod_attention，实现点积注意力机制
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, training
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            # 创建一个随机的下三角矩阵作为注意力遮罩，形状与 query 和 key 的长度相同
            attn_mask = torch.randn(
                query.size(1), key.size(1), dtype=torch.float, device=query.device
            ).tril(diagonal=0)
            # 将输入张量的维度顺序重新排列，将序列长度维度移到第二个位置
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            # 计算注意力分数，并加上注意力遮罩，然后进行 softmax 归一化
            return torch.nn.functional.dropout(
                (torch.matmul(q, k.transpose(-2, -1)).div(3.0) + attn_mask).softmax(
                    dim=-1
                ),
                p=0.4,
                training=training,
                inplace=False,
            ).matmul(v)

        # 调用通用检查方法 _check_common，验证 dot_prod_attention 函数
        self._check_common(dot_prod_attention, contains=False, has_dropout=True)

        # 另外一个测试，确保 batch_size=1 时图表现出略微不同的特性
        tensor_shape = (1, 2, 16, 32)
        args = [
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
        ]
        self._check_common(
            dot_prod_attention, args1=args, contains=False, has_dropout=True
        )

    @skipIfRocm
    def _test_sdpa_rewriter_17(self):
        # 定义一个函数 dot_prod_attention，实现点积注意力机制
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, training
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            # 将输入张量的维度顺序重新排列，将序列长度维度移到第二个位置
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            # 获取 batch_size 和 key 的长度
            bs = q.size(0)
            k_len = k.size(-2)
            # 创建一个下三角矩阵的布尔类型张量作为注意力遮罩
            attn_mask = torch.ones(
                bs, k_len, dtype=torch.bool, device=query.device
            ).tril(diagonal=0)
            # 计算注意力分数，并使用注意力遮罩填充
            scores = torch.matmul(q, k.transpose(-2, -1)) / 3.0
            attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
            scores = scores.masked_fill(attn_mask, -float("inf"))
            # 计算注意力权重并进行 dropout
            weights = torch.nn.functional.softmax(scores, dim=-1)
            weights = torch.nn.functional.dropout(
                weights,
                p=0.4,
                training=training,
                inplace=False,
            )
            # 返回加权后的值向量
            return torch.matmul(weights, v)

        # 调用通用检查方法 _check_common，验证 dot_prod_attention 函数
        self._check_common(dot_prod_attention, check_train=False, has_dropout=True)

    @skipIfRocm
    def _test_sdpa_rewriter_18(self):
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            causal_mask: torch.Tensor,
        ) -> torch.Tensor:
            # 将 query、key、value 张量的维度重新排列，以适应注意力计算的要求
            query = query.permute([0, 2, 1, 3])
            key = key.permute([0, 2, 1, 3])
            value = value.permute([0, 2, 1, 3])
            # 计算注意力权重
            attn_weights = torch.matmul(query, key.permute(0, 1, 3, 2))
            # 计算缩放因子的倒数，用于缩放注意力权重
            inv_scale = torch.full(
                (), math.sqrt(value.size(-1)), dtype=query.dtype, device=query.device
            )
            attn_weights = attn_weights.div(inv_scale)
            # 根据因果掩码设置注意力权重
            causal_mask_value = torch.full(
                (), torch.finfo(query.dtype).min, dtype=query.dtype, device=query.device
            )
            attn_weights = torch.where(causal_mask, attn_weights, causal_mask_value)
            # 计算加权后的值，并返回注意力权重、key 和 value 张量的转置
            return (
                (
                    torch.nn.functional.dropout(
                        attn_weights.softmax(dim=-1), 0.0
                    ).matmul(value)
                ),
                key.permute([0, 2, 1, 3]),
                value.permute([0, 2, 1, 3]),
            )

        # 定义张量的形状
        tensor_shape = (4, 2, 16, 32)
        # 生成一个因果掩码张量
        causal_mask = torch.ones(2, 2, dtype=torch.bool, device=self.device).tril(
            diagonal=0
        )
        # 定义参数列表
        args = [
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
            causal_mask,
        ]
        # 调用测试函数，检查 dot_prod_attention 函数的输出
        self._check_common(
            dot_prod_attention,
            args1=args,
            contains=False,
            has_dropout=False,
            check_train=False,
        )

        # 再次调用测试函数，检查 batch_size=1 时的情况
        tensor_shape = (1, 2, 16, 32)
        args = [
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
            causal_mask,
        ]
        self._check_common(
            dot_prod_attention,
            args1=args,
            contains=False,
            has_dropout=False,
            check_train=False,
        )

    @skipIfRocm
    # 定义一个函数 dot_prod_attention，实现点积注意力机制
    def _test_sdpa_rewriter_19(self):
        # 内部函数 dot_prod_attention 实现了点积注意力机制
        def dot_prod_attention(
            query: torch.Tensor,        # 查询张量
            key: torch.Tensor,          # 键张量
            value: torch.Tensor,        # 值张量
            causal_mask: torch.Tensor,  # 因果掩码张量
            attn_mask: torch.Tensor,    # 注意力掩码张量
            training,                   # 是否处于训练模式的布尔值
        ) -> torch.Tensor:             # 返回值为 torch 张量
            # 计算注意力权重矩阵，query 与 key 的乘积
            attn_weights = torch.matmul(query, key.permute(0, 1, 3, 2))
            # 计算缩放因子的倒数
            inv_scale = torch.full(
                (),
                math.sqrt(value.size(-1)),
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )
            # 对注意力权重进行缩放
            attn_weights = attn_weights.div(inv_scale)
            # 生成一个与 query 张量相同类型和设备的因果掩码值
            causal_mask_value = torch.full(
                (), torch.finfo(query.dtype).min, dtype=query.dtype, device=query.device
            )
            # 根据因果掩码选择性地应用掩码值
            attn_weights = torch.where(causal_mask, attn_weights, causal_mask_value)
            # 将注意力掩码添加到注意力权重中
            attn_weights = attn_weights + attn_mask
            # 对注意力权重进行 softmax 归一化，并转换为与值张量相同类型的数据类型
            attn_weights = attn_weights.softmax(dim=-1).type(value.dtype)
            # 对注意力权重进行 40% 的随机 dropout 操作
            return torch.nn.functional.dropout(
                attn_weights,
                p=0.4,
                training=training,
                inplace=False,
            ).matmul(value)  # 将 dropout 后的注意力权重与值张量相乘，得到最终的输出张量

        tensor_shape = (4, 2, 16, 32)  # 定义张量的形状
        causal_mask = torch.ones(16, 16, dtype=torch.bool, device=self.device).tril(
            diagonal=0
        )  # 生成一个下三角形式的因果掩码张量
        attn_mask = torch.randn((16, 16), dtype=torch.float, device=self.device)  # 生成随机的注意力掩码张量
        args = [
            torch.randn(tensor_shape, device=self.device),  # 生成随机的查询张量
            torch.randn(tensor_shape, device=self.device),  # 生成随机的键张量
            torch.randn(tensor_shape, device=self.device),  # 生成随机的值张量
            causal_mask,    # 因果掩码张量
            attn_mask,      # 注意力掩码张量
        ]
        # 调用 self._check_common 方法，验证 dot_prod_attention 函数的常见情况
        self._check_common(
            dot_prod_attention,
            args1=args,
            contains=False,
            has_dropout=True,
            check_train=False,
        )
# 如果 CUDA 可用并且平台支持融合注意力机制，则定义一个基于 CUDA 的动态测试类。
if HAS_CUDA and PLATFORM_SUPPORTS_FUSED_ATTENTION:

    class SDPAPatternRewriterCudaDynamicTests(SDPAPatternRewriterCudaTests):
        # 设置使用动态形状
        use_static_shapes = False


# 如果 CPU 可用，则定义一个基于 CPU 的测试类。
if HAS_CPU:

    class SDPAPatternRewriterCpuTests(TestSDPAPatternRewriterTemplate):
        # 指定设备为 CPU
        device = "cpu"
        # 定义 CPU 下的各个测试方法
        test_sdpa_rewriter_1_cpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_1
        test_pattern_fails_with_reuse_cpu = (
            TestSDPAPatternRewriterTemplate._test_pattern_fails_with_reuse
        )
        test_sdpa_rewriter_2_cpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_2
        test_sdpa_rewriter_5_cpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_5
        test_pattern_fails_with_tensor_factor_cpu = (
            TestSDPAPatternRewriterTemplate._test_pattern_fails_with_tensor_factor
        )
        test_pattern_fails_with_unsupported_mask_cpu = (
            TestSDPAPatternRewriterTemplate._test_pattern_fails_with_unsupported_mask
        )
        test_sdpa_rewriter_11_cpu = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_11
        )
        test_sdpa_rewriter_12_cpu = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_12
        )
        test_sdpa_prev_13_cpu = TestSDPAPatternRewriterTemplate._test_sdpa_prev_13
        test_sdpa_prev_14_cpu = TestSDPAPatternRewriterTemplate._test_sdpa_prev_14
        test_sdpa_prev_15_cpu = TestSDPAPatternRewriterTemplate._test_sdpa_prev_15
        # 部分方法使用偏函数定义
        test_sdpa_rewriter_13_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_13, dtype=torch.float32
        )
        test_sdpa_rewriter_14_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_14
        )
        test_sdpa_rewriter_15_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_15
        )
        test_sdpa_rewriter_16_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_16
        )
        test_sdpa_rewriter_16_fp32_mask_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_16_fp32_mask
        )
        test_sdpa_rewriter_17_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_17
        )
        test_sdpa_rewriter_18_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_18
        )
        test_sdpa_rewriter_19_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_19
        )

    # 定义一个基于 CPU 的动态测试类，继承自基础 CPU 测试类。
    class SDPAPatternRewriterCpuDynamicTests(SDPAPatternRewriterCpuTests):
        # 设置使用动态形状
        use_static_shapes = False


# 如果当前脚本作为主程序运行，并且运行环境是 Linux，则执行测试函数。
if __name__ == "__main__":
    if IS_LINUX:
        run_tests()
```