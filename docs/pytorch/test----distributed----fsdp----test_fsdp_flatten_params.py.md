# `.\pytorch\test\distributed\fsdp\test_fsdp_flatten_params.py`

```py
# Owner(s): ["oncall: distributed"]  # 指定该文件的所有者是分布式团队负责
import sys  # 导入系统模块

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
from torch import distributed as dist  # 导入PyTorch分布式模块
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # 导入全分片数据并行模块
from torch.distributed.fsdp._flat_param import (  # 导入参数扁平化相关模块
    FlatParamHandle,
    FlatParamShardMetadata,
    HandleShardingStrategy,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入用于测试的分布式通用模块
from torch.testing._internal.common_fsdp import FSDPTest  # 导入FSDP测试模块
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN  # 导入测试运行函数和测试调试标志

if not dist.is_available():  # 如果分布式环境不可用
    print("Distributed not available, skipping tests", file=sys.stderr)  # 打印信息并输出到标准错误流
    sys.exit(0)  # 退出程序

if TEST_WITH_DEV_DBG_ASAN:  # 如果处于开发调试ASAN模式
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",  # 打印跳过信息
        file=sys.stderr,
    )
    sys.exit(0)  # 退出程序

class TestFlattenParams(FSDPTest):
    """Tests parameter flattening and shard metadata logic."""
    
    @property
    def world_size(self) -> int:
        # 限制世界大小为1，因为这些单元测试仅对扁平化逻辑进行测试，或直接检查分片子程序，无需多个进程
        return 1

    def _get_default_config(self):
        # 返回默认配置字典
        return {
            "device": torch.device("cuda"),  # 使用CUDA设备
            "sharding_strategy": HandleShardingStrategy.FULL_SHARD,  # 参数分片策略为全分片
            "offload_params": False,  # 不开启参数卸载
            "mp_param_dtype": None,  # 多进程参数数据类型为空
            "mp_reduce_dtype": None,  # 多进程减少数据类型为空
            "keep_low_precision_grads": False,  # 不保留低精度梯度
            "process_group": self.process_group,  # 使用测试中的进程组
            "use_orig_params": False,  # 不使用原始参数
            "fsdp_extension": None,  # FSDP扩展为空
        }

    def _get_transformer(self, seed=0):
        torch.manual_seed(seed)  # 设置随机种子以保证结果可复现
        module = torch.nn.Transformer(  # 创建一个Transformer模型
            d_model=32,  # 模型的输入维度
            num_encoder_layers=2,  # 编码器层数
            num_decoder_layers=2,  # 解码器层数
            dim_feedforward=128,  # 前馈神经网络的隐藏层大小
            dropout=0.1,  # Dropout概率
        )
        module.register_buffer("dummy_buffer", torch.tensor(1.0))  # 注册一个缓冲区作为示例

        def get_input(device, dtype):
            torch.manual_seed(1)  # 设置随机种子以保证结果可复现
            src = torch.rand(20, 8, 32).to(device=device, dtype=dtype)  # 生成源数据张量
            tgt = torch.rand(10, 8, 32).to(device=device, dtype=dtype)  # 生成目标数据张量
            return (src, tgt)

        module.get_input = get_input  # 将输入数据生成函数赋给模型的get_input属性
        return module  # 返回创建的Transformer模型

    def _get_shared_params_transformer(self, seed=0):
        module = self._get_transformer(seed=seed)  # 获取指定随机种子的Transformer模型
        # 共享前馈神经网络
        for enc_layer, dec_layer in zip(module.encoder.layers, module.decoder.layers):
            dec_layer.linear1.weight = enc_layer.linear1.weight  # 共享线性层1权重
            dec_layer.linear2.weight = enc_layer.linear2.weight  # 共享线性层2权重
        return module  # 返回共享参数后的Transformer模型

    @skip_if_lt_x_gpu(1)  # 如果GPU数量少于1，跳过测试
    def test_partial_flattening(self):
        """Tests flattening some submodules but not others."""
        self.run_subtests(
            {"half": [False, True]},  # 运行子测试，参数是不同的配置
            self._test_partial_flattening,  # 调用具体的子测试方法
        )
    def _test_partial_flattening(self, half: bool):
        # 获取 Transformer 模型
        module = self._get_transformer()
        # 如果 half 为 True，则将模型转换为半精度
        if half:
            module = module.half()
        # 计算模型参数的总数量
        numel = sum(p.numel() for p in module.parameters())

        # 获取第二个编码器层和第一个解码器层的参数列表
        encoder_1_params = list(module.encoder.layers[1].parameters())
        decoder_0_params = list(module.decoder.layers[0].parameters())
        # 将这些参数列表合并成一个需要展平的参数列表
        params_to_flatten = encoder_1_params + decoder_0_params
        # 记录每个层的参数数量
        num_params = [len(encoder_1_params), len(decoder_0_params)]
        # 计算需要展平的参数总数量
        numel_to_flatten = sum(p.numel() for p in params_to_flatten)
        # 将第二个编码器层和第一个解码器层转换为 FSDP 类型
        module.encoder.layers[1] = FSDP(module.encoder.layers[1])
        module.decoder.layers[0] = FSDP(module.decoder.layers[0])
        # 获取展平后的参数列表
        flat_params = [
            module.encoder.layers[1]._flat_param,
            module.decoder.layers[0]._flat_param,
        ]

        # 断言展平后的参数总数量与预期展平数量相等
        self.assertEqual(sum(fp.numel() for fp in flat_params), numel_to_flatten)
        # 断言模型参数的总数量与最初的总数量相等
        self.assertEqual(sum(p.numel() for p in module.parameters()), numel)

        # 检查展平后的参数是否被替换为单个 `FlatParameter`
        self.assertEqual(len(list(module.encoder.layers[1].parameters())), 1)
        self.assertEqual(len(list(module.decoder.layers[0].parameters())), 1)

        # 检查未展平的参数是否保持不变
        self.assertEqual(
            len(list(module.encoder.layers[0].parameters())), num_params[0]
        )
        self.assertEqual(
            len(list(module.decoder.layers[1].parameters())), num_params[1]
        )

        # 检查调用 `module.to()` 是否影响到 `FlatParameter`
        orig_dtype = params_to_flatten[0].dtype
        new_dtype = torch.float32 if orig_dtype == torch.float16 else torch.float16
        for flat_param in flat_params:
            self.assertEqual(flat_param.dtype, orig_dtype)
        self.assertTrue(
            all(p.dtype == orig_dtype for p in module.encoder.layers[0].parameters())
        )
        # 将模型转换为新的数据类型
        module = module.to(dtype=new_dtype)
        # 检查 `FlatParameter` 的数据类型是否变为新的数据类型
        for flat_param in flat_params:
            self.assertEqual(flat_param.dtype, new_dtype)
        self.assertTrue(
            all(p.dtype == new_dtype for p in module.encoder.layers[0].parameters())
        )

    def test_flatten_nothing(self):
        """
        Tests that constructing a ``FlatParamHandle`` with no parameters
        raises an error.
        """
        # 运行子测试，测试构造空参数列表的 `FlatParamHandle` 是否会抛出错误
        self.run_subtests(
            {"half": [False, True]},
            self._test_flatten_nothing,
        )

    def _test_flatten_nothing(self, half: bool):
        # 获取 Transformer 模型
        module = self._get_transformer()
        # 如果 half 为 True，则将模型转换为半精度
        if half:
            module = module.half()
        # 使用断言检查构造空参数列表的 `FlatParamHandle` 是否会抛出 ValueError 错误
        with self.assertRaisesRegex(
            ValueError,
            "Cannot construct a FlatParamHandle with an empty parameter list",
        ):
            FlatParamHandle(
                [],
                module,
                **self._get_default_config(),
            )

    @skip_if_lt_x_gpu(1)
    def test_empty_module(self):
        """
        Tests flattening an empty module (i.e. one without any parameters).
        """
        # 获取一个没有参数的模块实例
        module = self._get_empty_module()
        # 创建一个随机输入数据
        in_data = torch.rand(1)
        # 获取参考输出
        ref_out = module(in_data)
        # 将模块应用于 FSDP，即分布式数据并行
        fsdp_module = FSDP(module)
        # 断言：检查 FSDP 模块的参数列表长度为 0
        self.assertEqual(len(list(fsdp_module.parameters())), 0)
        # 断言：检查 FSDP 模块的扁平化参数为 None
        self.assertIsNone(fsdp_module._flat_param)
        # 对输入数据应用 FSDP 模块
        fsdp_out = fsdp_module(in_data)
        # 断言：检查 FSDP 模块的输出与参考输出相等
        self.assertEqual(ref_out, fsdp_out)

    def _get_empty_module(self):
        """Returns a module with no parameters."""
        # 设定随机种子，保持结果可复现
        torch.manual_seed(0)

        class EmptyModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

            def get_input(self, device, dtype):
                # 设定随机种子，保持结果可复现
                torch.manual_seed(1)
                # 生成随机数据并移动到指定设备和数据类型
                return torch.rand(1).to(device=device, dtype=dtype)

        return EmptyModule()

    def test_numel_without_shared_params(self):
        """
        Tests that numel is preserved after flattening when there are no shared
        parameters in the module.
        """
        # 运行子测试：验证在模块中没有共享参数时，扁平化后 numel 保持不变
        self.run_subtests(
            {"half": [False, True]},
            self._test_numel_without_shared_params,
        )

    def _test_numel_without_shared_params(self, half: bool):
        # 获取变压器模块实例
        module = self._get_transformer()
        # 如果需要半精度计算，将模块转换为半精度
        if half:
            module = module.half()
        # 执行测试 numel 方法
        self._test_numel(module)

    def test_numel_with_shared_params(self):
        """
        Tests that numel is preserved after flattening when there are shared
        parameters in the module.
        """
        # 运行子测试：验证在模块中有共享参数时，扁平化后 numel 保持不变
        self.run_subtests(
            {"half": [False, True]},
            self._test_numel_with_shared_params,
        )

    def _test_numel_with_shared_params(self, half: bool):
        # 获取具有共享参数的变压器模块实例
        module = self._get_shared_params_transformer()
        # 如果需要半精度计算，将模块转换为半精度
        if half:
            module = module.half()
        # 执行测试 numel 方法
        self._test_numel(module)

    def _test_numel(self, module):
        # 计算模块所有参数的 numel 总和作为参考值
        ref_numel = sum(p.numel() for p in module.parameters())
        # 获取要扁平化的参数列表
        params_to_flatten = list(module.parameters())
        # 创建 FlatParamHandle 对象来处理扁平化参数
        flat_param_handle = FlatParamHandle(
            params_to_flatten,
            module,
            **self._get_default_config(),
        )
        # 断言：检查扁平化后的参数 numel 与参考值相等
        self.assertEqual(ref_numel, flat_param_handle.flat_param.numel())

    @skip_if_lt_x_gpu(1)
    def test_output_without_shared_params(self):
        """
        Tests a forward pass after flattening when there are no shared
        parameters in the module.
        """
        # 运行子测试：验证在模块中没有共享参数时，扁平化后的前向传播输出正确
        self.run_subtests(
            {"half": [False, True]},
            self._test_output_without_shared_params,
        )

    def _test_output_without_shared_params(self, half: bool):
        # 获取变压器模块实例
        module = self._get_transformer()
        # 如果需要半精度计算，将模块转换为半精度
        if half:
            module = module.half()
        # 执行测试前向传播输出方法
        self._test_output(module)

    @skip_if_lt_x_gpu(1)
    def test_output_with_shared_params(self):
        """
        Tests a forward pass after flattening when there are shared parameters
        in the module.
        """
        # 运行子测试，传入不同的参数组合进行测试
        self.run_subtests(
            {"half": [False, True]},
            self._test_output_with_shared_params,
        )

    def _test_output_with_shared_params(self, half: bool):
        # 获取包含共享参数的转换器模块
        module = self._get_shared_params_transformer()
        if half:
            # 如果 half 为 True，则将模块转换为半精度
            module = module.half()
        # 执行输出测试
        self._test_output(module)

    def _test_output(self, module: nn.Module):
        # 将模块移到指定的计算资源（例如 GPU）
        module = module.to(self.rank)
        # 获取模块的参考输出
        ref_output = self._get_output(module)
        # 使用 FSDP 封装模块
        fsdp_module = FSDP(module)
        # 获取 FSDP 封装后的输出
        fsdp_output = self._get_output(fsdp_module)
        # 断言参考输出与 FSDP 输出相等
        self.assertEqual(ref_output, fsdp_output)

    def _get_output(self, module):
        # 获取模块参数的设备（device）
        device = next(module.parameters()).device
        # 获取模块参数的数据类型（dtype）
        dtype = next(module.parameters()).dtype
        # 获取模块的输入数据
        input = module.get_input(device, dtype)
        # 执行模块的前向传播，并返回输出
        return module(*input)

    @skip_if_lt_x_gpu(1)
    def test_pnorm_after_step_with_shared_params(self):
        """
        Tests for parameter Frobenius norm parity after an optimizer step when
        there are shared parameters in the module. If the parameter sharing is
        handled incorrectly, then an optimizer step should reveal that.
        """
        # 运行子测试，传入不同的参数组合进行测试
        self.run_subtests(
            {"half": [False, True]},
            self._test_pnorm_after_step_with_shared_params,
        )

    def _test_pnorm_after_step_with_shared_params(self, half: bool):
        # 获取包含共享参数的转换器模块，并将其移到指定的计算资源
        module = self._get_shared_params_transformer().to(self.rank)
        if half:
            # 如果 half 为 True，则将模块转换为半精度
            module = module.half()
        # 获取参考输出后的 Frobenius 范数
        ref_pnorm_after_step = self._get_pnorm_after_step(module)
        # 重新创建模块，准备进行 FSDP 封装
        module = self._get_shared_params_transformer().to(self.rank)
        if half:
            # 如果 half 为 True，则将模块转换为半精度
            module = module.half()
        # 使用 FSDP 封装模块
        fsdp_module = FSDP(module)
        # 获取 FSDP 封装后的 Frobenius 范数
        fsdp_pnorm_after_step = self._get_pnorm_after_step(fsdp_module)
        # 断言参考输出后的 Frobenius 范数与 FSDP 输出后的 Frobenius 范数相等
        self.assertEqual(ref_pnorm_after_step, fsdp_pnorm_after_step)

    def _get_pnorm_after_step(self, module):
        # 使用 SGD 优化器对模块的参数进行优化
        optim = torch.optim.SGD(module.parameters(), lr=0.01)
        # 计算损失
        loss = self._get_output(module).sum()
        # 反向传播
        loss.backward()
        # 执行优化器步骤
        optim.step()
        # 计算并返回参数的 Frobenius 范数
        return torch.norm(torch.stack([p.detach().norm() for p in module.parameters()]))
    def test_flat_param_shard_metadata_aligned_full_precision(self):
        """
        Tests that ``FlatParameter`` shard metadata are computed as expected
        with alignment padding and parameter full precision.
        """
        # 创建一个包含三个线性层的神经网络模型，无偏置
        module = torch.nn.Sequential(
            torch.nn.Linear(3, 7, bias=False),  # 0.weight
            torch.nn.Linear(7, 5, bias=False),  # 1.weight
            torch.nn.Linear(5, 5, bias=False),  # 2.weight
        )
        # 获取模型中的所有参数并加入到列表中
        params_to_flatten = list(module.parameters())
        # 获取默认配置参数字典
        handle_kwargs = self._get_default_config()
        # 设置使用原始参数标志位为True
        handle_kwargs["use_orig_params"] = True
        # 创建FlatParamHandle对象，用于处理扁平化参数
        handle = FlatParamHandle(params_to_flatten, module, **handle_kwargs)
        
        # 对于32位全精度，FSDP在每个原始参数后面填充到3个元素，以实现0 mod 4 numel（即0 mod 16字节）。
        # 因此，未分片的`FlatParameter`布局如下：
        #   21 + (3) + 35 + (1) + 25
        # 其中 (x) 表示 x 个填充元素。总共有85个元素。

        # `FlatParamShardMetadata`不包括对齐填充，但会考虑它们
        self._test_flat_param_shard_metadata(
            handle,
            # 模拟2个rank中的rank 0
            start=0,
            end=42,
            expected=FlatParamShardMetadata(
                param_names=["0.weight", "1.weight"],
                param_shapes=[(7, 3), (5, 7)],
                param_numels=[21, 35],
                # 21 + (3) + 19 = 43
                param_offsets=[(0, 20), (0, 18)],
            ),
        )
        self._test_flat_param_shard_metadata(
            handle,
            # 模拟2个rank中的rank 1
            start=43,
            end=85,
            expected=FlatParamShardMetadata(
                param_names=["1.weight", "2.weight"],
                param_shapes=[(5, 7), (5, 5)],
                param_numels=[35, 25],
                # 16 + (1) + 25 = 42
                param_offsets=[(19, 34), (0, 24)],
            ),
        )
    def test_flat_param_shard_metadata_aligned_mixed_precision(self):
        """
        Tests that ``FlatParameter`` shard metadata are computed as expected
        with alignment padding and parameter mixed precision.
        """
        # 创建一个包含多层线性模块的神经网络模型
        module = torch.nn.Sequential(
            torch.nn.Linear(2, 5, bias=False),  # 0.weight
            torch.nn.Linear(5, 5, bias=False),  # 1.weight
            torch.nn.Linear(5, 3, bias=False),  # 2.weight
        )
        # 获取模型中所有参数，并存入列表
        params_to_flatten = list(module.parameters())
        # 获取默认配置参数
        handle_kwargs = self._get_default_config()
        # 设置使用原始参数标志为True
        handle_kwargs["use_orig_params"] = True
        # 设置混合精度参数数据类型为torch.float16
        handle_kwargs["mp_param_dtype"] = torch.float16
        # 创建FlatParamHandle对象，用于管理扁平化后的参数
        handle = FlatParamHandle(params_to_flatten, module, **handle_kwargs)
        
        # 对于16位混合精度，FSDP在每个原始参数后填充至多7个元素，以实现0模8的元素数（即0模16字节）。
        # 因此，未分片的FlatParameter布局如下：
        #   10 + (6) + 25 + (7) + 15
        # 这里的 (x) 表示x个填充元素。总共63个元素。

        # FlatParamShardMetadata不包括对齐填充，但考虑它们的存在
        self._test_flat_param_shard_metadata(
            handle,
            # 模拟2个rank中的rank 0
            start=0,
            end=31,
            expected=FlatParamShardMetadata(
                param_names=["0.weight", "1.weight"],
                param_shapes=[(5, 2), (5, 5)],
                param_numels=[10, 25],
                # 10 + (6) + 16 = 32
                param_offsets=[(0, 9), (0, 15)],
            ),
        )
        self._test_flat_param_shard_metadata(
            handle,
            # 模拟2个rank中的rank 1
            start=32,
            end=63,
            expected=FlatParamShardMetadata(
                param_names=["1.weight", "2.weight"],
                param_shapes=[(5, 5), (3, 5)],
                param_numels=[25, 15],
                # 9 + (7) + 15 = 31
                param_offsets=[(16, 24), (0, 14)],
            ),
        )

    def _test_flat_param_shard_metadata(
        self,
        handle: FlatParamHandle,
        start: int,
        end: int,
        expected: FlatParamShardMetadata,
        ):
        """
        Helper function to test the computation of FlatParameter shard metadata.
        """
    ):
        """
        Tests the subroutine ``_get_shard_metadata()`` that computes shard
        metadata based on start and end indices in the unsharded flat
        parameter, where both indices are inclusive.

        We manually set the relevant attributes on the flat parameter to be
        able to check the effect of ``_get_shard_metadata()`` via
        ``shard_metadata()`` since normally the attributes are set in
        ``_init_shard_metadata()`` with the start and end indices fixed based
        on rank and world size.
        """
        # 获取测试用例中的 flat 参数
        flat_param = handle.flat_param
        # 调用 _get_shard_metadata() 方法计算分片元数据，并设置到 flat 参数的 _shard_param_infos 属性中
        flat_param._shard_param_infos = handle._get_shard_metadata(start, end)
        # 调用 shard_metadata() 方法获取计算后的分片元数据
        shard_metadata = handle.shard_metadata()
        # 使用断言检查计算后的分片元数据是否与期望的结果 expected 相等
        self.assertEqual(
            shard_metadata,
            expected,
            msg=f"{handle.shard_metadata()}, {expected}",
        )
# 如果当前脚本被作为主程序执行，则运行测试函数
if __name__ == "__main__":
    run_tests()
```