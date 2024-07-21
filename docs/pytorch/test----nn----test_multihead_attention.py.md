# `.\pytorch\test\nn\test_multihead_attention.py`

```py
# 导入所需模块和库
import contextlib  # 上下文管理模块，用于管理上下文中的资源
import random  # 随机数生成模块
import unittest  # 单元测试框架
import unittest.mock as mock  # 单元测试模拟对象模块

import torch  # PyTorch深度学习库
import torch.nn as nn  # PyTorch神经网络模块

from torch.nn import MultiheadAttention  # 多头注意力机制
from torch.testing._internal.common_device_type import (  # 导入设备类型相关测试工具
    dtypes,
    instantiate_device_type_tests,
    onlyCUDAAndPRIVATEUSE1,
)
from torch.testing._internal.common_nn import NNTestCase  # 导入神经网络通用测试类
from torch.testing._internal.common_utils import (  # 导入通用测试工具函数
    instantiate_parametrized_tests,
    parametrize as parametrize_test,
    run_tests,
    TEST_NUMPY,
    TEST_WITH_CROSSREF,
)

if TEST_NUMPY:
    import numpy as np  # 如果开启了numpy测试，导入numpy库


# 警告信息：如果在此文件中新增顶层测试用例，必须同时更新 test/run_test.py，
# 否则该测试用例将不会在持续集成系统中运行。

class TestMultiheadAttentionNN(NNTestCase):
    _do_cuda_memory_leak_check = True  # 检查CUDA内存泄漏
    _do_cuda_non_default_stream = True  # 使用非默认CUDA流

    @unittest.skipIf(not TEST_NUMPY, "numpy not found")
    @parametrize_test("average_attn_weights", [True, False])
    def test_multihead_attn_3d_attn_mask(self):
        embed_dim = 8  # 嵌入维度
        num_heads = 4  # 注意力头数
        batch_size = 8  # 批量大小
        src_len = 3  # 源序列长度
        tgt_len = 2  # 目标序列长度

        query = torch.rand(batch_size, tgt_len, embed_dim)  # 随机生成查询张量 [N, T, D]
        key = torch.rand(batch_size, src_len, embed_dim)  # 随机生成键张量 [N, S, D]
        value = key  # 值张量与键张量相同 [N, S, D]

        # 创建注意力掩码，随机整数张量，并进行填充
        attn_mask = torch.randint(0, 2, (batch_size, tgt_len, src_len)).float()  # [N, T, S]
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf")).masked_fill(
            attn_mask == 1, 0.0
        )

        mta_model = torch.nn.MultiheadAttention(embed_dim, num_heads)  # 创建多头注意力模型对象

        # 生成3D结果，扩展注意力掩码张量
        attn_mask_3d = torch.repeat_interleave(attn_mask, num_heads, dim=0)  # [N * H, T, S]
        output_3d = mta_model(
            query.transpose(0, 1),
            key.transpose(0, 1),
            value.transpose(0, 1),
            attn_mask=attn_mask_3d,
        )[0]
        output_3d = output_3d.transpose(0, 1)  # 调整输出维度顺序 [N, T, D]

        for i in range(0, batch_size):
            # 对单个样本进行2D测试
            output_2d = mta_model(
                query[i].unsqueeze(0).transpose(0, 1),
                key[i].unsqueeze(0).transpose(0, 1),
                value[i].unsqueeze(0).transpose(0, 1),
                attn_mask=attn_mask[i],
            )[0]

            # 断言输出2D张量形状与3D结果的一致性
            self.assertEqual(output_3d[i].unsqueeze(0).transpose(0, 1), output_2d)

    def test_multihead_attn_no_bias(self):
        embed_dim = 8  # 嵌入维度
        num_heads = 4  # 注意力头数
        mha = torch.nn.MultiheadAttention(embed_dim, num_heads, bias=False)  # 创建无偏置的多头注意力对象

        # 验证无偏置应用于输入和输出投影层
        self.assertIsNone(mha.in_proj_bias)  # 输入投影偏置为空
        self.assertIsNone(mha.out_proj.bias)  # 输出投影偏置为空
    # 定义一个测试函数，测试多头注意力机制的无效形状情况
    def test_multihead_attn_invalid_shape(self):
        # 创建一个包含 4 个头和 4 维特征的多头注意力对象
        mha = torch.nn.MultiheadAttention(4, 4)
        # 调用实现无效形状测试的具体实现方法
        self._test_multihead_attn_invalid_shape_impl(mha)
        # 给测试一个机会来触发快速路径。（目前情况下不会触发，但未来可能的情况下可能会更少限制。）
        with torch.no_grad():
            # 在评估模式下调用无效形状测试的具体实现方法
            self._test_multihead_attn_invalid_shape_impl(mha.eval())

    # 在无梯度计算的上下文中定义一个测试函数，测试多头注意力机制中 NestedTensor 的使用情况
    @torch.no_grad()
    def test_multihead_attn_nested_tensor_outside_fast_path(self):
        # 创建一个包含 4 个头、4 维特征且批量优先的多头注意力对象，并设置为评估模式
        mha = torch.nn.MultiheadAttention(4, 4, batch_first=True).eval()
        # 创建一个 NestedTensor，包含一个形状为 (4, 4) 的随机张量
        nt = torch.nested.nested_tensor([torch.randn(4, 4)])
        # 检查是否存在 torch_function 来处理其中的一个或多个对象，以测试 torch_function 的回退情况
        has_torch_func = torch.overrides.has_torch_function(
            (
                nt,
                mha.in_proj_weight,
                mha.in_proj_bias,
                mha.out_proj.weight,
                mha.out_proj.bias,
            )
        )
        # 根据是否存在 torch_function 选择不同的错误消息
        if has_torch_func:
            msg = "MultiheadAttention does not support NestedTensor.*argument has_torch_function"
        else:
            msg = (
                "MultiheadAttention does not support NestedTensor outside of its fast path.*grad is "
                + "enabled and.*or biases requires_grad"
            )
        # 断言在错误消息匹配的情况下抛出 AssertionError 异常
        with self.assertRaisesRegex(AssertionError, msg):
            mha(nt, nt, nt)

        # 如果存在 torch_function，直接返回，因为它们都将以相同的消息失败
        if has_torch_func:
            return

        # 在无梯度计算的上下文中继续进行测试
        with torch.no_grad():
            mha(nt, nt, nt)
        # 在推断模式下进行测试
        with torch.inference_mode():
            mha(nt, nt, nt)
        # 创建一个包含形状为 (4, 4, requires_grad=False) 的随机张量 NestedTensor
        nt = torch.nested.nested_tensor([torch.randn(4, 4, requires_grad=False)])
        # 设置 nt 的 requires_grad 属性为 False
        nt.requires_grad = False
        # 断言在错误消息匹配的情况下抛出 AssertionError 异常
        with self.assertRaisesRegex(AssertionError, msg):
            mha(nt, nt, nt)
        # 将 mha 的相关权重和偏置的 requires_grad 属性设置为 False
        mha.in_proj_weight.requires_grad = False
        mha.in_proj_bias.requires_grad = False
        mha.out_proj.weight.requires_grad = False
        mha.out_proj.bias.requires_grad = False
        # 调用 mha 的 NestedTensor 输入来测试
        mha(nt, nt, nt)
    class TestMultiheadAttentionNNDeviceType(NNTestCase):
        @torch.no_grad()
        @unittest.skipIf(
            TEST_WITH_CROSSREF,
            "CrossRef turns on TorchFunctionMode, and so disables fastpath.",
        )
        # 定义测试函数，验证多头自注意力机制的快速路径模拟
        def test_multihead_self_attn_two_masks_fast_path_mock(self, device):
            """
            Multihead self-attention should take fast path when both attention mask (mask type 0)
            and key padding mask (mask type 1) are provided at the same time on CPU and CUDA and PrivateUse1
            """
            device = device.rstrip(":0123456789")
            # 如果设备不是在 ["cpu", "cuda", torch._C._get_privateuse1_backend_name()] 中，则跳过测试
            if device not in ["cpu", "cuda", torch._C._get_privateuse1_backend_name()]:
                self.skipTest("Fastpath only runs on CPU and CUDA and PrivateUse1.")

            with torch.autocast(device_type=device, enabled=False):
                embed_dim = 16
                num_heads = 8
                batch_size = 8
                src_len = 5

                query = value = key = torch.rand(batch_size, src_len, embed_dim).to(device)
                # 创建两种不同类型的掩码
                attn_mask = torch.randint(0, 2, (src_len, src_len)).bool().to(device)
                key_padding_mask = (
                    torch.randint(0, 2, (batch_size, src_len)).bool().to(device)
                )

                with mock.patch(
                    "torch._native_multi_head_attention",
                    new=mock.MagicMock(return_value=(torch.Tensor(), torch.Tensor())),
                ) as fastpath_mock:
                    # 在快速路径上计算注意力
                    mta_model = torch.nn.MultiheadAttention(
                        embed_dim, num_heads, batch_first=True, device=device
                    ).eval()
                    mta_model.training = False
                    mta_model(
                        query,
                        key,
                        value,
                        attn_mask=attn_mask,
                        key_padding_mask=key_padding_mask,
                    )
                    # 如果模拟被调用，表示采用了快速路径
                    self.assertTrue(fastpath_mock.called)

        @onlyCUDAAndPRIVATEUSE1
        @dtypes(torch.half, torch.float, torch.double)
        # 测试不同数据类型下的多头注意力机制
        def test_multihead_attention_dtype(self, device, dtype):
            embed_dim = 128
            num_heads = 8
            sl = 10
            bs = 8
            model = nn.MultiheadAttention(embed_dim, num_heads).to(device).to(dtype)
            q = torch.randn(sl, bs, embed_dim, device=device, dtype=dtype)
            k = torch.randn(sl, bs, embed_dim, device=device, dtype=dtype)
            v = torch.randn(sl, bs, embed_dim, device=device, dtype=dtype)
            out = model(q, k, v)
            # 断言输出与输入的形状相同
            self.assertEqual(q.size(), out[0].size())
            # 断言输出的数据类型与预期的数据类型相同

        @onlyCUDAAndPRIVATEUSE1
        @dtypes(torch.half, torch.float, torch.double)
    # 定义一个测试方法，用于测试多头注意力机制的数据类型和 batch_first 参数设为 True
    def test_multihead_attention_dtype_batch_first(self, device, dtype):
        # 设定嵌入维度和注意力头数
        embed_dim = 128
        num_heads = 8
        # 序列长度和批量大小
        sl = 10
        bs = 8
        # 当 batch_first=True 时，如果模型调用 .eval() 进入推理模式，
        # 可能会触发本地快速路径。测试两种路径。
        for training in (True, False):
            # 创建多头注意力模型，指定嵌入维度、注意力头数和 batch_first=True
            model = (
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
                .to(device)  # 将模型移到指定设备
                .to(dtype)    # 设置模型的数据类型
            )
            # 如果不是训练模式，则切换到评估模式，并使用 torch.no_grad() 上下文管理器
            if not training:
                model = model.eval()
                cm = torch.no_grad()
            else:
                cm = contextlib.nullcontext()
            with cm:
                # 生成随机的查询、键和值张量，指定设备和数据类型
                q = torch.randn(bs, sl, embed_dim, device=device, dtype=dtype)
                k = torch.randn(bs, sl, embed_dim, device=device, dtype=dtype)
                v = torch.randn(bs, sl, embed_dim, device=device, dtype=dtype)
                # 调用多头注意力模型进行计算，禁止返回注意力权重
                out = model(q, k, v, need_weights=False)
                # 断言输出张量的大小与查询张量相同
                self.assertEqual(q.size(), out[0].size())
                # 断言输出张量的数据类型与指定的数据类型相同
                self.assertEqual(dtype, out[0].dtype)

    # 用于测试多头注意力模型快速路径中查询和偏置具有不同数据类型的情况
    @dtypes(torch.double)
    @torch.no_grad()
    def test_multihead_attn_fast_path_query_and_bias_have_different_dtypes(
        self, device, dtype
    ):
        # 创建多头注意力模型，设置嵌入维度、注意力头数、batch_first=True，并指定设备和数据类型
        mha = torch.nn.MultiheadAttention(
            4, 4, batch_first=True, dtype=dtype, device=device
        ).eval()
        # 将输入投影的偏置参数转换为半精度，然后移到指定设备
        mha.in_proj_bias = torch.nn.Parameter(
            mha.in_proj_bias.to(torch.half).to(device)
        )
        # 生成随机的查询张量，指定数据类型和设备，并调用多头注意力模型
        query = torch.randn(4, 4, 4, dtype=dtype, device=device)
        mha(query, query, query)

    # 用于测试多头注意力模型快速路径的小规模测试
    @dtypes(torch.double)
    @torch.no_grad()
    def test_multihead_attn_fast_path_small_test(self, device, dtype):
        # 创建多头注意力模型，设置嵌入维度、注意力头数、batch_first=True，并指定设备和数据类型
        mha = torch.nn.MultiheadAttention(
            4, 4, batch_first=True, dtype=dtype, device=device
        ).eval()
        # 生成随机的查询张量，指定数据类型和设备，并调用多头注意力模型
        query = torch.randn(4, 4, 4, dtype=dtype, device=device)
        mha(query, query, query)

    # 用于测试多头注意力模型中投影的偏置参数为 None 的情况
    @dtypes(torch.double)
    @torch.no_grad()
    def test_multihead_attn_in_proj_bias_none(self, device, dtype):
        # 创建多头注意力模型，设置输入维度、注意力头数、关闭偏置参数，指定设备和数据类型
        mha = torch.nn.MultiheadAttention(2, 2, bias=False, dtype=dtype, device=device)
        # 生成随机的查询张量，指定数据类型和设备，并调用多头注意力模型
        query = torch.rand(2, 2, 2, dtype=dtype, device=device)
        mha(query, query, query)

    # 用于测试多头注意力模型中投影的权重参数为 None 的情况
    @dtypes(torch.double)
    @torch.no_grad()
    def test_multihead_attn_in_proj_weight_none(self, device, dtype):
        # 设置 kdim == vdim == 2，这意味着 vdim != embed_dim
        # 将导致使用每个输入的项目权重，从而强制 self.in_proj_weight = None
        # 创建多头注意力模型，设置嵌入维度、注意力头数、指定维度参数，设备和数据类型
        mha = torch.nn.MultiheadAttention(
            4, 4, vdim=2, kdim=2, dtype=dtype, device=device
        )
        # 生成随机的查询和键张量，指定数据类型和设备，并调用多头注意力模型
        query = torch.rand(4, 4, 4, dtype=dtype, device=device)
        key = torch.rand(4, 4, 2, dtype=dtype, device=device)
        mha(query, key, key)
# 实例化设备类型测试，使用 TestMultiheadAttentionNNDeviceType 类，并将其注册到全局命名空间中
instantiate_device_type_tests(TestMultiheadAttentionNNDeviceType, globals())

# 实例化参数化测试，使用 TestMultiheadAttentionNN 类
instantiate_parametrized_tests(TestMultiheadAttentionNN)

# 如果脚本被直接执行（而不是被导入到其他脚本中），则运行测试
if __name__ == "__main__":
    run_tests()
```