# `.\pytorch\test\distributed\_tensor\test_attention.py`

```
# 导入单元测试模块
import unittest

# 导入 PyTorch 相关模块
import torch
from torch import nn
from torch.distributed._tensor import DeviceMesh, distribute_tensor, Shard
from torch.distributed._tensor.debug import CommDebugMode
from torch.distributed._tensor.experimental.attention import (
    _CausalBehavior,
    _is_causal_behavior,
    _scaled_dot_product_chunk_flash_attention,
    _scaled_dot_product_ring_efficient_attention,
    _scaled_dot_product_ring_flash_attention,
    attention_context_parallel,
    AttentionContextParallel,
)
from torch.distributed.tensor.parallel import parallelize_module
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_FUSED_ATTENTION,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
    TEST_CUDA,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfRocm,
    TEST_WITH_ROCM,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    ModelArgs,
    Transformer,
    with_comms,
)

# 获取 C10D 的函数接口
c10d_functional = torch.ops.c10d_functional


class RingAttentionTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    # 如果 GPU 小于两个则跳过测试
    @skip_if_lt_x_gpu(2)
    # 如果在 ROCm 平台上运行，则跳过测试
    @skipIfRocm  # Missing _c10d_functional_autograd::all_to_all_single
    # 如果平台不支持闪存注意力机制，则跳过测试
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention"
    )
    # 使用通信层进行测试
    @with_comms
    # 参数化测试，测试是否是因果行为
    @parametrize("is_causal", [True, False])
    # 如果 GPU 小于两个则跳过测试
    @skip_if_lt_x_gpu(2)
    # 如果平台不支持闪存注意力机制，则跳过测试
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention"
    )
    # 使用通信层进行测试
    @with_comms
    # 使用 SDPA 内核进行测试，选择闪存注意力机制作为后端
    @sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])
    # 参数化测试，测试是否是因果行为
    @parametrize("is_causal", [True, False])
    def test_ring_attention_native_transformer(self, is_causal: bool) -> None:
        # 创建一个设备网格，用于分布式计算，包括设备类型和全局设备索引
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, self.world_size),
        )
        # 定义张量的数据类型为 bfloat16
        dtype = torch.bfloat16
        # 定义批量大小
        bs = 8
        # 序列长度
        ntokens = 8
        # 模型维度
        dim = 32
        # 注意力头数
        nheads = 8
        # 编码器层数
        num_layers = 2

        # 创建 Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nheads,
            dim_feedforward=dim,
            batch_first=True,
        ).to(dtype)
        # 在设备网格上并行化编码器层的自注意力机制
        encoder_layer = parallelize_module(
            module=encoder_layer,
            device_mesh=device_mesh,
            parallelize_plan={
                "self_attn": AttentionContextParallel(),
            },
        )
        # 创建 Transformer 编码器模型
        model = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 将模型移动到指定设备类型并设置数据类型
        model = model.to(self.device_type).to(dtype)

        # 根据是否因果生成掩码
        mask = (
            nn.Transformer.generate_square_subsequent_mask(
                ntokens, device=self.device_type, dtype=dtype
            )
            if is_causal
            else None
        )
        # 创建随机输入序列张量
        seq = torch.rand((bs, ntokens, dim), device=self.device_type, dtype=dtype)

        # 进入通信调试模式
        with CommDebugMode() as comm_mode:
            # 使用模型进行序列处理，传入掩码和因果性标志
            out = model(seq, mask=mask, is_causal=is_causal)
        # 断言通信模式返回的通信计数字典
        self.assertDictEqual(
            comm_mode.get_comm_counts(),
            {
                c10d_functional.all_to_all_single: (self.world_size - 1) * num_layers,
            },
        )

        # 再次进入通信调试模式
        with CommDebugMode() as comm_mode:
            # 计算输出张量的和并执行反向传播
            out.sum().backward()
        # 断言通信模式返回的通信计数字典
        self.assertDictEqual(
            comm_mode.get_comm_counts(),
            {
                c10d_functional.all_to_all_single: (self.world_size - 1)
                * 2
                * num_layers,
            },
        )
    # 定义测试函数 `test_ring_attention_custom_transformer`，用于测试自定义的 Transformer 模型的环形注意力功能
    def test_ring_attention_custom_transformer(self) -> None:
        # 创建设备网格对象 `device_mesh`，包含所有设备的索引
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, self.world_size),
        )
        # 定义数据类型 `dtype` 为 torch.bfloat16
        dtype = torch.bfloat16
        # 定义批量大小 `bs` 为 2
        bs = 2
        # 创建模型参数对象 `args`
        args = ModelArgs()

        # 创建 Transformer 模型 `model`，转换为指定的数据类型 `dtype`，并移到指定设备 `self.device_type`
        model = Transformer(args).to(dtype).to(self.device_type)

        # 并行化模型的指定部分，根据设备网格 `device_mesh` 和并行化计划
        model = parallelize_module(
            module=model,
            device_mesh=device_mesh,
            parallelize_plan={
                f"layers.{i}.attention": AttentionContextParallel()
                for i in range(args.n_layers)
            },
        )

        # 生成随机整数序列 `seq`，形状为 (bs, args.max_seq_len)，使用指定设备 `self.device_type`
        seq = torch.randint(
            args.vocab_size, (bs, args.max_seq_len), device=self.device_type
        )

        # 启用通信调试模式 `CommDebugMode`
        with CommDebugMode() as comm_mode:
            # 输入序列 `seq` 到模型 `model` 中进行推理
            out = model(seq)
        # 断言通信统计结果与期望结果相等
        self.assertDictEqual(
            comm_mode.get_comm_counts(),
            {
                c10d_functional.all_to_all_single: (self.world_size - 1)
                * args.n_layers,
            },
        )

        # 启用通信调试模式 `CommDebugMode`
        with CommDebugMode() as comm_mode:
            # 对输出 `out` 进行求和并反向传播
            out.sum().backward()
        # 断言通信统计结果与期望结果相等
        self.assertDictEqual(
            comm_mode.get_comm_counts(),
            {
                c10d_functional.all_to_all_single: (self.world_size - 1)
                * 2
                * args.n_layers,
            },
        )

    # 根据 GPU 数量跳过测试，要求至少有 2 个 GPU
    @skip_if_lt_x_gpu(2)
    # 根据平台是否支持融合注意力功能来跳过测试
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_ATTENTION,
        "Does not support flash nor efficient attention",
    )
    # 在 CUDA 平台上（非 ROCM），如果不支持 Flash Attention，则跳过测试
    @unittest.skipIf(
        TEST_CUDA and not TEST_WITH_ROCM and not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "Does not support flash attention",
    )  # On CUDA (not ROCM) platform, the UT is skipped if no FA support (even if ME may get supported)
    # 使用通信的装饰器 `with_comms`
    @with_comms
    # 参数化测试函数 `attention_fn`，根据平台支持选择不同的注意力函数进行测试
    @parametrize(
        "attention_fn",
        [
            _scaled_dot_product_ring_flash_attention
            if PLATFORM_SUPPORTS_FLASH_ATTENTION
            else None,
            _scaled_dot_product_ring_efficient_attention
            if PLATFORM_SUPPORTS_MEM_EFF_ATTENTION
            else None,
            # _scaled_dot_product_ring_cudnn_attention, # TODO: not built by default
        ],
    )
    # 定义测试函数，用于编译和运行给定的注意力函数
    def test_ring_attention_compile(self, attention_fn: object) -> None:
        # 如果 attention_fn 为空，则跳过测试并输出不支持当前平台的信息
        if attention_fn is None:
            self.skipTest("Unsupported on current platform")
        
        # 创建设备网格对象，指定设备类型和全局设备索引
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, self.world_size),
        )
        
        # 定义张量数据类型为 bfloat16
        dtype = torch.bfloat16
        # 定义批量大小
        bs = 8
        # 定义查询 tokens 的数量
        query_tokens = 8
        # 定义上下文 tokens 的数量
        context_tokens = 24
        # 定义特征维度
        dim = 32
        # 定义注意力头数
        nheads = 8
        
        # 创建随机查询张量，指定形状、设备和数据类型，并设置为需要梯度
        query = torch.rand(
            (bs, nheads, self.world_size * query_tokens, dim),
            device=self.device_type,
            dtype=dtype,
            requires_grad=True,
        )
        
        # 创建随机键张量，指定形状、设备和数据类型
        key = torch.rand(
            (bs, nheads, self.world_size * context_tokens, dim),
            device=self.device_type,
            dtype=dtype,
        )
        
        # 创建随机值张量，指定形状、设备和数据类型
        value = torch.rand(
            (bs, nheads, self.world_size * context_tokens, dim),
            device=self.device_type,
            dtype=dtype,
        )

        # 定义查询张量的分布位置
        query_placement = [Shard(2)]
        # 将查询张量在设备网格上分布
        dquery = distribute_tensor(query, device_mesh, query_placement)
        # 断言查询张量的形状符合预期
        self.assertEqual(query.shape, (bs, nheads, self.world_size * query_tokens, dim))

        # 定义上下文张量的分布位置
        context_placement = [Shard(2)]
        # 将键张量在设备网格上分布
        dkey = distribute_tensor(key, device_mesh, context_placement)
        # 将值张量在设备网格上分布
        dvalue = distribute_tensor(value, device_mesh, context_placement)

        # 编译注意力函数以进行加速计算，使用全图模式和 eager 模式
        compiled = torch.compile(attention_fn, fullgraph=True, backend="aot_eager")

        # 调用编译后的注意力函数进行计算，并获取输出、最大局部序列误差以及其他参数
        out, lse, *args = compiled(
            device_mesh.get_group(),
            dquery.to_local(),
            dkey.to_local(),
            dvalue.to_local(),
        )
        
        # 断言输出张量的形状符合预期
        self.assertEqual(out.shape, (bs, nheads, query_tokens, dim))
        # 断言最大局部序列误差的类型为 torch.Tensor
        self.assertIsInstance(lse, torch.Tensor)

        # 使用分块计算的函数计算加权和，并断言与编译后的输出张量部分一致
        (
            out_chunk,
            *others,
        ) = _scaled_dot_product_chunk_flash_attention(
            query,
            key,
            value,
            size=self.world_size,
            is_causal=False,
        )
        self.assertEqual(
            out,
            out_chunk[
                :, :, self.rank * query_tokens : (self.rank + 1) * query_tokens, :
            ],
        )

        # 对输出张量的所有元素求和并进行反向传播
        out.sum().backward()
# 实例化带参数的测试类 RingAttentionTest，并注册其测试用例
instantiate_parametrized_tests(RingAttentionTest)

# 如果当前脚本作为主程序执行，则运行测试函数
if __name__ == "__main__":
    run_tests()
```