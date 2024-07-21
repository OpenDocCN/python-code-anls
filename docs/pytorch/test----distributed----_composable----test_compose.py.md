# `.\pytorch\test\distributed\_composable\test_compose.py`

```
# Owner(s): ["oncall: distributed"]

# 导入必要的库和模块
import copy  # 导入深拷贝功能
import sys   # 导入系统相关功能

from typing import Dict  # 导入类型提示模块

import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入PyTorch分布式模块
import torch.nn as nn  # 导入PyTorch神经网络模块
from torch.distributed._composable import checkpoint, fully_shard, replicate  # 导入分布式相关功能
from torch.distributed._shard.sharded_tensor import ShardedTensor  # 导入分片张量模块
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType  # 导入完全分片数据并行模块
from torch.distributed.fsdp.api import MixedPrecision, ShardingStrategy  # 导入混合精度和分片策略模块
from torch.distributed.fsdp.wrap import ModuleWrapPolicy  # 导入模块包装策略模块
from torch.testing._internal.common_dist_composable import (  # 导入测试相关模块
    CompositeModel,
    CompositeParamModel,
    UnitModule,
)
from torch.testing._internal.common_distributed import (  # 导入分布式测试相关模块
    SaveForwardInputsModel,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_fsdp import FSDPTest  # 导入完全分片数据并行测试模块
from torch.testing._internal.common_utils import (  # 导入通用测试工具模块
    instantiate_parametrized_tests,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)

# 如果分布式不可用，则跳过测试
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 如果使用开发ASAN，则跳过测试，因为torch + multiprocessing spawn存在已知问题
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 定义测试类 TestFSDPCheckpoint，继承自 FSDPTest 类
class TestFSDPCheckpoint(FSDPTest):
    
    @property
    def world_size(self) -> int:
        return 2  # 返回世界大小为 2

    # 定义 _test_parity 方法，用于测试基本模型和测试模型的梯度一致性
    # TODO: 暂时为了兼容性定义 `use_same_inputs_across_ranks`，因为某些测试模型配置没有简单的基础模型进行比较。
    # 在这些情况下，我们使用跨等级相同的输入，以便平均梯度等于本地梯度，以检查平等性。这意味着梯度的归约是未检查的。
    def _test_parity(
        self,
        base_model: nn.Module,
        test_model: nn.Module,
        inp_size: torch.Size,
        inp_device: torch.device,
        grad_to_none: bool,
        use_same_inputs_across_ranks: bool,
    ):
        LR = 0.01
        base_optim = torch.optim.Adam(base_model.parameters(), lr=LR)  # 基础模型优化器
        test_optim = torch.optim.Adam(test_model.parameters(), lr=LR)  # 测试模型优化器

        for _ in range(5):
            if use_same_inputs_across_ranks:
                torch.manual_seed(0)  # 如果跨等级使用相同的输入，设置随机种子为0
            x = torch.randn(inp_size, device=inp_device)  # 生成指定大小和设备的随机张量 x
            test_loss = test_model(x).sum()  # 计算测试模型的损失和
            base_loss = base_model(x).sum()  # 计算基础模型的损失和

            self.assertEqual(test_loss, base_loss)  # 断言测试损失和基础损失相等

            test_loss.backward()  # 反向传播测试损失
            test_optim.step()  # 测试优化器执行一步优化
            test_optim.zero_grad(set_to_none=grad_to_none)  # 清空测试优化器的梯度

            base_loss.backward()  # 反向传播基础损失
            base_optim.step()  # 基础优化器执行一步优化
            base_optim.zero_grad(set_to_none=grad_to_none)  # 清空基础优化器的梯度

    @skip_if_lt_x_gpu(2)  # 如果 GPU 数量小于 2，跳过测试
    # 测试函数：测试在同一子模块中进行包装
    def test_wrap_same_submodule(self):
        # 创建一个 CUDA 设备上的 UnitModule 模型实例
        model = UnitModule(device=torch.device("cuda"))

        # 深度复制原始模型，以备后续比较
        base_model = copy.deepcopy(model)

        # 再次深度复制模型用于测试
        test_model = copy.deepcopy(model)

        # 对测试模型的 seq 属性进行检查点和完全分片操作
        test_model.seq = checkpoint(test_model.seq)
        test_model.seq = fully_shard(
            test_model.seq,
            policy=ModuleWrapPolicy({nn.Linear}),
        )

        # 运行子测试，验证模型行为的一致性
        self.run_subtests(
            {
                "base_model": [base_model],
                "test_model": [test_model],
                "inp_size": [torch.Size((2, 100))],
                "inp_device": [torch.device("cuda")],
                "grad_to_none": [True, False],
                "use_same_inputs_across_ranks": [True],
            },
            self._test_parity,
        )

    # 测试函数：测试检查点和完全分片在复合模型子模块中的应用
    def _test_checkpoint_fsdp_submodules(self):
        # 创建一个 CUDA 设备上的 CompositeModel 模型实例
        model = CompositeModel(device=torch.device("cuda"))

        # 深度复制原始模型，以备后续比较
        base_model = copy.deepcopy(model)

        # 再次深度复制模型用于测试
        test_model = copy.deepcopy(model)

        # 对模型的 u1 和 u2 子模块进行完全分片操作
        test_model.u1 = fully_shard(test_model.u1, policy=None)
        test_model.u2 = fully_shard(test_model.u2)

        # 对 u1 和 u2 模块的 seq 属性进行检查点操作
        test_model.u1.seq = checkpoint(test_model.u1.seq)
        test_model.u2.seq = checkpoint(test_model.u2.seq)

        # 运行子测试，验证模型行为的一致性
        self.run_subtests(
            {
                "base_model": [base_model],
                "test_model": [test_model],
                "inp_size": [torch.Size((2, 100))],
                "inp_device": [torch.device("cuda")],
                "grad_to_none": [True, False],
                "use_same_inputs_across_ranks": [True],
            },
            self._test_parity,
        )

    # 如果 GPU 数量小于 2 则跳过测试：测试检查点在复合模型子模块中的应用
    @skip_if_lt_x_gpu(2)
    def test_checkpoint_fsdp_submodules_non_reentrant(self):
        # 调用内部函数进行测试
        self._test_checkpoint_fsdp_submodules()

    # 如果 GPU 数量小于 2 则跳过测试：测试检查点和完全分片在前向输入中的应用
    @skip_if_lt_x_gpu(2)
    def test_checkpoint_fully_shard_cast_forward_inputs(self):
        # 运行子测试，验证模型行为的一致性
        self.run_subtests(
            {
                "checkpoint_strict_submodule": [False, True],
            },
            self._test_checkpoint_fully_shard_cast_forward_inputs,
        )

    # 测试函数：测试检查点和完全分片在前向输入中的应用
    def _test_checkpoint_fully_shard_cast_forward_inputs(
        self, checkpoint_strict_submodule: bool
    ):
        # 创建空字典以存储前向输入
        forward_inputs: Dict[nn.Module, torch.Tensor] = {}
        # 创建混合精度对象，使用 float16 参数类型，并将前向输入强制类型转换为 float16
        fp16_mp = MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True)
        # 创建混合精度对象，使用 float32 参数类型，并将前向输入强制类型转换为 float32
        fp32_mp = MixedPrecision(param_dtype=torch.float32, cast_forward_inputs=True)

        # 在 GPU 上创建 SaveForwardInputsModel 模型对象，初始化时不强制类型转换前向输入
        model = SaveForwardInputsModel(
            forward_inputs=forward_inputs, cast_forward_inputs=False
        ).cuda()
        # 在 GPU 上创建一个形状为 (2, 100) 的全零张量
        x = torch.zeros(2, 100, device="cuda")

        # 对 model.c2 层进行混合精度分片，使用 fp16_mp 对象进行操作
        fully_shard(model.c2, mixed_precision=fp16_mp)
        # 如果需要严格检查子模块，则对 model.c2.l 进行检查点操作，否则对整个 model.c2 进行检查点操作
        if checkpoint_strict_submodule:
            checkpoint(model.c2.l)
        else:
            checkpoint(model.c2)
        # 对整个模型 model 进行混合精度分片，使用 fp32_mp 对象进行操作
        fully_shard(model, mixed_precision=fp32_mp)

        # 计算模型在输入 x 上的输出，计算损失并反向传播
        loss = model(x).sum()
        loss.backward()

        # 断言前向输入经过重新计算后的数据类型为 float32
        self.assertEqual(forward_inputs[model].dtype, torch.float32)
        # 断言模型子模块 model.c1 的前向输入经过重新计算后的数据类型为 float32
        self.assertEqual(forward_inputs[model.c1].dtype, torch.float32)
        # 检查重新计算的前向输入 model.c2 的数据类型是否保持为 float16
        # 特别地，检查重新计算的前向输入是否保持正确的数据类型
        self.assertEqual(forward_inputs[model.c2].dtype, torch.float16)

    @skip_if_lt_x_gpu(2)
    def test_fully_shard_replicate_correct_replicate_params(self):
        # 创建 CompositeParamModel 模型对象，指定在 CUDA 设备上进行操作
        model = CompositeParamModel(device=torch.device("cuda"))
        # 在 UnitModule 内的线性层进行混合精度分片，策略为 ModuleWrapPolicy({nn.Linear})
        fully_shard(model.u1, policy=ModuleWrapPolicy({nn.Linear}))
        fully_shard(model.u2, policy=ModuleWrapPolicy({nn.Linear}))
        # 复制其余部分的参数
        replicate(model)
        # 运行前向传播和反向传播，以初始化分布式数据并行（DDP）
        inp = torch.randn(2, 100, device="cuda")
        model(inp).sum().backward()
        # 确保复制的参数名称符合预期，即模型的直接参数和模型的非 UnitModule 子模块的参数被复制
        param_names = replicate.state(model)._param_names
        replicated_modules = [
            (name, mod)
            for (name, mod) in model.named_children()
            if mod not in [model.u1, model.u2]
        ]
        replicated_param_names = [
            f"{module_name}.{n}"
            for module_name, mod in replicated_modules
            for n, _ in mod.named_parameters()
        ]
        replicated_param_names.extend(
            [n for n, _ in model.named_parameters(recurse=False)]
        )
        # 断言复制的参数名称集合与预期的复制参数名称集合相匹配
        self.assertEqual(set(param_names), set(replicated_param_names))

    @skip_if_lt_x_gpu(2)
    # 定义一个测试方法，用于验证带有参数的 FSDP 子模块的检查点功能
    def test_checkpoint_fsdp_submodules_with_param(self):
        # 创建一个带有 CUDA 设备的复合参数模型
        model = CompositeParamModel(device=torch.device("cuda"))

        # 复制基础模型作为基准模型
        base_model = copy.deepcopy(model)

        # 复制测试模型并对其子模块的序列进行检查点操作
        test_model = copy.deepcopy(model)
        test_model.u1.seq = checkpoint(test_model.u1.seq)
        test_model.u2.seq = checkpoint(test_model.u2.seq)

        # 对测试模型进行完全分片操作
        test_model = fully_shard(test_model)

        # 运行子测试，验证各项参数与 `_test_parity` 方法的一致性
        self.run_subtests(
            {
                "base_model": [base_model],
                "test_model": [test_model],
                "inp_size": [torch.Size((2, 100))],
                "inp_device": [torch.device("cuda")],
                "grad_to_none": [True, False],
                "use_same_inputs_across_ranks": [True],
            },
            self._test_parity,
        )

    # 如果 GPU 数量少于 2，则跳过执行当前测试方法
    @skip_if_lt_x_gpu(2)
    def test_checkpoint_fsdp_submodules_with_param_no_shard(self):
        # 创建一个带有 CUDA 设备的复合参数模型
        model = CompositeParamModel(device=torch.device("cuda"))

        # 复制基础模型作为基准模型
        base_model = copy.deepcopy(model)

        # 复制测试模型并对其子模块的序列进行检查点操作
        test_model = copy.deepcopy(model)
        test_model.u1.seq = checkpoint(test_model.u1.seq)
        test_model.u2.seq = checkpoint(test_model.u2.seq)

        # 对测试模型进行完全分片操作，但使用 `ShardingStrategy.NO_SHARD` 策略
        test_model = fully_shard(test_model, strategy=ShardingStrategy.NO_SHARD)

        # 运行子测试，验证各项参数与 `_test_parity` 方法的一致性
        self.run_subtests(
            {
                "base_model": [base_model],
                "test_model": [test_model],
                "inp_size": [torch.Size((2, 100))],
                "inp_device": [torch.device("cuda")],
                "grad_to_none": [True, False],
                "use_same_inputs_across_ranks": [True],
            },
            self._test_parity,
        )

    # 如果 GPU 数量少于 2，则跳过执行当前测试方法
    @skip_if_lt_x_gpu(2)
    def test_composable_fsdp_replicate(self):
        # 验证 API 如何组合使用，例如如果在同一模块上同时应用 `fully_shard` 和 `replicate` 应引发异常
        model = CompositeModel(device=torch.device("cuda"))

        # 对模型的第一层进行完全分片
        fully_shard(model.l1)

        # 使用断言验证在同一层上应用 `replicate` 是否会引发 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Cannot apply .*replicate"):
            replicate(model.l1)

        # 对模型的第二层进行复制操作，不应引发异常
        replicate(model.l2)

    # 如果 GPU 数量少于 2，则跳过执行当前测试方法
    @skip_if_lt_x_gpu(2)
    def test_fully_shard_replicate_composability(self):
        """
        测试 `fully_shard` 和 `replicate` 的组合性。为节省单元测试时间，我们在子测试中运行不同的配置。
        """
        # 运行子测试，验证各种配置参数与 `_test_replicate_in_fully_shard` 方法的一致性
        self.run_subtests(
            {
                "config": [
                    "1fm,1r",
                    "1r,1fm",
                    "1r,1fa",
                    "1r1fm,1fm",
                    "1r1fa,1fm",
                    "1fm1fm,1r1r,1fm",
                ]
            },
            self._test_replicate_in_fully_shard,
        )
    # 定义测试方法，用于在完全分片中复制模型
    def _test_replicate_in_fully_shard(self, config: str):
        """
        To interpret the config, each comma delineates a level in the module
        tree ordered bottom-up; 'r' means ``replicate``; 'f' means
        ``fully_shard``; 'a' means auto wrap; and 'm' means manual wrap.
        解释配置字符串，每个逗号分隔的部分表示模块树的一个级别，从底向上；
        'r' 表示复制模块；'f' 表示完全分片；'a' 表示自动包装；'m' 表示手动包装。
        """
        
        # 设置种子以确保所有排名初始化相同的模型
        torch.manual_seed(0)
        
        # 根据配置不同执行不同的模型复制和完全分片操作
        if config == "1fm,1r":
            base_model = CompositeModel(device=torch.device("cuda"))
            test_model = copy.deepcopy(base_model)
            fully_shard(test_model.l1)
            replicate(test_model)
        elif config == "1r,1fm":
            base_model = CompositeParamModel(torch.device("cuda"))
            test_model = copy.deepcopy(base_model)
            replicate(test_model.u1)
            fully_shard(test_model)
        elif config == "1r,1fa":
            base_model = CompositeParamModel(torch.device("cuda"))
            test_model = copy.deepcopy(base_model)
            replicate(test_model.u1)
            fully_shard(test_model, policy=ModuleWrapPolicy({UnitModule}))
        elif config == "1r1fm,1fm":
            base_model = CompositeParamModel(torch.device("cuda"))
            test_model = copy.deepcopy(base_model)
            replicate(test_model.u1)
            fully_shard(test_model.u2)
            fully_shard(test_model)
        elif config == "1r1fa,1fm":
            base_model = CompositeParamModel(torch.device("cuda"))
            test_model = copy.deepcopy(base_model)
            replicate(test_model.u1)
            fully_shard(test_model.u2, policy=ModuleWrapPolicy({UnitModule}))
            fully_shard(test_model)
        elif config == "1fm1fm,1r1r,1fm":
            base_model = CompositeParamModel(torch.device("cuda"))
            test_model = copy.deepcopy(base_model)
            fully_shard(test_model.u1.seq)
            fully_shard(test_model.u2.seq)
            replicate(test_model.u1)
            replicate(test_model.u2)
            fully_shard(test_model)
        else:
            raise ValueError(f"Unknown config: {config}")
        
        # 将基础模型复制以保证与测试模型应用相同的数据并行策略
        replicate(base_model)
        
        # 设置种子以确保不同排名获得不同的输入数据
        torch.manual_seed(self.rank + 1)
        
        # 执行测试以验证基础模型和测试模型的结果一致性
        self._test_parity(
            base_model,
            test_model,
            torch.Size((2, 100)),
            torch.device("cuda"),
            True,
            False,
        )

    @skip_if_lt_x_gpu(2)
    # 定义测试函数，用于测试模型对象的状态字典处理功能
    def test_state_dict_fsdp_submodules(self):
        # 创建一个在 CUDA 设备上的复合模型对象
        model = CompositeModel(device=torch.device("cuda"))

        # 定义全分片参数和无分片参数字典
        full_shard_args = {"strategy": ShardingStrategy.FULL_SHARD}
        no_shard_args = {"strategy": ShardingStrategy.NO_SHARD}

        # 对模型的两个组件 u1 和 u2 应用不同的分片策略
        model.u1 = fully_shard(model.u1, **full_shard_args)
        model.u2 = fully_shard(model.u2, **no_shard_args)

        # 设置模型的状态字典类型为分片状态字典
        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
        )

        # 获取模型当前的状态字典
        state_dict = model.state_dict()
        # 遍历状态字典的每个项
        for fqn, tensor in state_dict.items():
            # 检查每个项的完全限定名称是否包含 "u1"
            if "u1" in fqn:
                # 断言当前项的值类型为 ShardedTensor
                self.assertIsInstance(tensor, ShardedTensor)
            elif "u2" in fqn:
                # 断言当前项的值类型为 torch.Tensor
                self.assertIsInstance(tensor, torch.Tensor)

        # 确保能正确获取模型的状态字典类型设置
        _ = FSDP.get_state_dict_type(model)
# 使用参数化测试实例化 TestFSDPCheckpoint 类的测试
instantiate_parametrized_tests(TestFSDPCheckpoint)

# 如果当前脚本作为主程序运行，执行测试函数
if __name__ == "__main__":
    run_tests()
```