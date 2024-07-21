# `.\pytorch\test\distributed\fsdp\test_wrap.py`

```py
# 导入必要的模块和库

import functools  # 提供了创建偏函数的功能
import itertools  # 提供了用于迭代操作的函数
import os  # 提供了与操作系统交互的功能
import tempfile  # 提供了创建临时文件和目录的功能
import unittest  # 提供了编写和运行单元测试的框架
from enum import auto, Enum  # auto用于自动分配枚举值，Enum用于定义枚举类
from typing import Callable, Union  # 用于类型提示，指定函数参数和返回值的类型

import torch  # PyTorch深度学习框架
import torch.nn as nn  # 提供了神经网络层的模块
import torch.nn.functional as F  # 提供了神经网络中的各种函数，如激活函数、损失函数等
from torch.distributed.fsdp._wrap_utils import _validate_frozen_params  # 导入私有函数_validate_frozen_params
from torch.distributed.fsdp.fully_sharded_data_parallel import (  # 导入FSDP模块的相关类和函数
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (  # 导入wrap模块中的相关函数和类
    _or_policy,
    _Policy,
    _wrap_module_cls_individually,
    always_wrap_policy,
    CustomPolicy,
    enable_wrap,
    ModuleWrapPolicy,
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
    wrap,
)
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer  # 导入Transformer模型的层
from torch.nn.modules.batchnorm import _BatchNorm  # 导入私有模块_BatchNorm
from torch.testing._internal.common_cuda import TEST_MULTIGPU  # 导入测试多GPU相关的常量
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入跳过测试条件不满足时的函数
from torch.testing._internal.common_fsdp import (  # 导入FSDP相关的测试用例和模块
    _maybe_cuda,
    CUDAInitMode,
    DummyProcessGroup,
    FSDPInitMode,
    FSDPTest,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (  # 导入一般实用函数和常量
    FILE_SCHEMA,
    find_free_port,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_CUDA,
    TestCase,
)

# 定义一个简单的神经网络模型
class BatchNormNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(10, 10, bias=False)  # 创建一个全连接层，输入输出维度为10，无偏置
        self.bn1 = nn.BatchNorm1d(10)  # 创建一个一维批归一化层，输入维度为10
        self.bn2 = nn.BatchNorm2d(10)  # 创建一个二维批归一化层，输入通道数为10
        self.bn3 = nn.BatchNorm3d(10)  # 创建一个三维批归一化层，输入通道数为10
        self.sync_bn = nn.SyncBatchNorm(10)  # 创建一个同步批归一化层，输入通道数为10

# 定义一个LoRA解码器模型
class LoraModel(nn.Module):
    """This is a toy LoRA decoder model."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, 32)  # 创建一个嵌入层，词汇表大小为100，嵌入维度为32
        self.layers = nn.ModuleList([LoraDecoder() for _ in range(4)])  # 创建包含4个LoraDecoder层的模块列表
        self.norm = nn.LayerNorm(32)  # 创建一个层归一化层，输入特征维度为32
        self.embed_tokens.weight.requires_grad_(False)  # 设置嵌入层的权重不可训练
        self.norm.weight.requires_grad_(False)  # 设置层归一化层的权重不可训练
        self.norm.bias.requires_grad_(False)  # 设置层归一化层的偏置不可训练

# 定义一个LoRA解码器模型
class LoraDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = LoraAttention()  # 创建一个LoraAttention对象
        self.mlp = LoraMLP()  # 创建一个LoraMLP对象
        self.inp_layernorm = nn.LayerNorm(32)  # 创建一个输入层归一化层，输入特征维度为32
        self.post_attn_layernorm = nn.LayerNorm(32)  # 创建一个经过注意力层后的层归一化层，输入特征维度为32
        self.inp_layernorm.weight.requires_grad_(False)  # 设置输入层归一化层的权重不可训练
        self.inp_layernorm.bias.requires_grad_(False)  # 设置输入层归一化层的偏置不可训练
        self.post_attn_layernorm.weight.requires_grad_(False)  # 设置经过注意力层后的层归一化层的权重不可训练
        self.post_attn_layernorm.bias.requires_grad_(False)  # 设置经过注意力层后的层归一化层的偏置不可训练

# 定义一个LoRA注意力层模型
class LoraAttention(nn.Module):
    # 定义神经网络模型的初始化方法，继承父类的初始化方法
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，输入维度为32，输出维度为32，不使用偏置
        self.q_proj = nn.Linear(32, 32, bias=False)
        # 创建一个线性层，输入维度为32，输出维度为8，不使用偏置
        self.lora_A = nn.Linear(32, 8, bias=False)
        # 创建一个线性层，输入维度为8，输出维度为32，不使用偏置
        self.lora_B = nn.Linear(8, 32, bias=False)
        # 创建一个线性层，输入维度为32，输出维度为32，不使用偏置
        self.k_proj = nn.Linear(32, 32, bias=False)
        # 创建一个线性层，输入维度为32，输出维度为32，不使用偏置
        self.v_proj = nn.Linear(32, 32, bias=False)
        # 创建一个线性层，输入维度为32，输出维度为32，不使用偏置
        self.o_proj = nn.Linear(32, 32, bias=False)
        # 设置q_proj权重不需要计算梯度
        self.q_proj.weight.requires_grad_(False)
        # 设置k_proj权重不需要计算梯度
        self.k_proj.weight.requires_grad_(False)
        # 设置v_proj权重不需要计算梯度
        self.v_proj.weight.requires_grad_(False)
        # 设置o_proj权重不需要计算梯度
        self.o_proj.weight.requires_grad_(False)
class LoraMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个线性层，输入大小为32，输出大小为128，无偏置
        self.proj1 = nn.Linear(32, 128, bias=False)
        # 定义另一个线性层，输入大小为128，输出大小为32，无偏置
        self.proj2 = nn.Linear(128, 32, bias=False)
        # 将proj1和proj2的权重设为不可训练
        self.proj1.weight.requires_grad_(False)
        self.proj2.weight.requires_grad_(False)


class WrapMethod(Enum):
    FSDP_CTOR = auto()
    # FSDP_CTOR 是推荐的方式，但保留 WRAP_API 以应对可能存在的使用情况，随着时间修复并逐步改为支持 FSDP_CTOR。


class TestFSDPWrap(FSDPTest):
    """
    测试包装FSDP的主要API，即将 auto_wrap_policy 传递给 FSDP 构造函数。
    """

    def setUp(self) -> None:
        super().setUp()

    class NestedSequentialModel:
        @staticmethod
        def get_model(cuda=True):
            # 创建一个包含多个线性层的序列模型
            sequential = nn.Sequential(
                nn.Linear(5, 5),
                nn.Linear(5, 5),
                nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 5)),
            )
            if cuda:
                # 如果需要在GPU上运行，将模型移动到CUDA设备上
                sequential = sequential.cuda()
            return sequential

        @staticmethod
        def verify_model_all_wrapped(cls, model):
            # 确保所有模块都已被 FSDP 包装
            cls.assertTrue(isinstance(model, FSDP))
            cls.assertTrue(isinstance(model.module[0], FSDP))
            cls.assertTrue(isinstance(model.module[1], FSDP))
            cls.assertTrue(isinstance(model.module[2], FSDP))
            cls.assertTrue(isinstance(model.module[2].module[0], FSDP))
            cls.assertTrue(isinstance(model.module[2].module[1], FSDP))

        @staticmethod
        def verify_model(cls, model):
            # 确保部分模块被 FSDP 包装，部分未被包装
            cls.assertTrue(isinstance(model, FSDP))
            cls.assertTrue(isinstance(model.module[0], nn.Linear))
            cls.assertTrue(isinstance(model.module[1], nn.Linear))
            cls.assertTrue(isinstance(model.module[2], FSDP))
            cls.assertTrue(isinstance(model.module[2].module[0], nn.Linear))
            cls.assertTrue(isinstance(model.module[2].module[1], nn.Linear))

    def _get_linear(self, fin, fout):
        # 返回一个线性层，输入大小为fin，输出大小为fout，无偏置
        return nn.Linear(fin, fout, bias=False)

    def _get_already_wrapped_fsdp(
        self, cuda_init_mode=CUDAInitMode.CUDA_BEFORE, nested=False
    ) -> FSDP:
        fn_self = self

        class MyModel(nn.Module):
            def __init__(self, nested):
                super().__init__()
                # TODO: test the various init modes.
                move_to_cuda = cuda_init_mode == CUDAInitMode.CUDA_BEFORE
                # if nested=True, the FSDP module will be nested one layer deep
                # and we should pick that up.
                if nested:
                    # 设置第一个线性层为一个包含两个子模块的序列
                    self.lin1 = nn.Sequential(
                        _maybe_cuda(fn_self._get_linear(1, 1), move_to_cuda),  # 获取并可能移动到 CUDA 上的线性层
                        FSDP(_maybe_cuda(fn_self._get_linear(1, 1), move_to_cuda)),  # 将线性层包装为 FSDP 模块
                    )
                else:
                    # 将第一个线性层直接包装为 FSDP 模块
                    self.lin1 = FSDP(
                        _maybe_cuda(fn_self._get_linear(1, 1), move_to_cuda)
                    )
                # 将后续两个线性层都包装为 FSDP 模块
                self.lin2 = FSDP(_maybe_cuda(fn_self._get_linear(1, 1), move_to_cuda))
                self.lin3 = FSDP(_maybe_cuda(fn_self._get_linear(1, 1), move_to_cuda))

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                # 模型的前向传播，按顺序应用三个线性层
                return self.lin3(self.lin2(self.lin1(input)))

        # 创建 MyModel 实例并返回
        model = MyModel(nested=nested)
        return model

    @skip_if_lt_x_gpu(2)
    @parametrize("nested", [True, False])
    @parametrize("cuda_init_mode", [CUDAInitMode.CUDA_AFTER, CUDAInitMode.CUDA_BEFORE])
    def test_error_already_wrapped(self, nested, cuda_init_mode):
        """
        Test that an error is raised if we attempt to wrap when submodules are
        already FSDP.
        """
        # 获取已经被 FSDP 包装过的模块
        wrapped_fsdp = self._get_already_wrapped_fsdp(
            nested=nested, cuda_init_mode=cuda_init_mode
        )
        # 根据初始化模式，决定是否将 wrapped_fsdp 移动到 CUDA 设备上
        if cuda_init_mode == CUDAInitMode.CUDA_AFTER:
            wrapped_fsdp = wrapped_fsdp.cuda()

        # 根据嵌套情况确定被包装模块的名称
        wrapped_module_name = "lin1.1" if nested else "lin1"
        # 断言捕获的异常信息包含特定的错误信息
        with self.assertRaisesRegex(
            ValueError,
            "FSDP auto wrapping requires modules to not already have FSDP "
            f"applied but found {wrapped_module_name} in",
        ):
            # 尝试对已经被包装的模块再次应用 FSDP 包装，期望抛出异常
            FSDP(wrapped_fsdp, auto_wrap_policy=size_based_auto_wrap_policy)

    @skip_if_lt_x_gpu(2)
    @parametrize("use_or_policy", [True, False])
    # 定义一个测试方法，用于验证是否逐个包装批归一化层
    def test_wrap_batchnorm_individually(self, use_or_policy):
        # 定义一个永不包装策略的函数
        def never_wrap_policy(*args, **kwargs):
            return False
        
        # 部分应用函数，用于逐个包装指定模块类
        wrap_batchnorm_individually = functools.partial(
            _wrap_module_cls_individually,
            module_classes=[
                _BatchNorm,
            ],
        )
        
        # 根据参数选择性地构建策略函数
        policy = (
            functools.partial(
                _or_policy, policies=[never_wrap_policy, wrap_batchnorm_individually]
            )
            if use_or_policy
            else wrap_batchnorm_individually
        )
        
        # 创建一个 BatchNormNet 模型实例
        model = BatchNormNet()
        
        # 使用 FSDP 自动包装策略创建 FSDP 对象
        fsdp = FSDP(model, auto_wrap_policy=policy)
        
        # 断言批归一化层是否被正确包装
        # 遍历各层，确保每个批归一化层都是 FSDP 类型
        for layer in [fsdp.bn1, fsdp.bn2, fsdp.bn3, fsdp.sync_bn]:
            self.assertTrue(isinstance(layer, FSDP))
        
        # 断言线性层未被包装
        self.assertFalse(isinstance(fsdp.lin, FSDP))

    # 跳过少于两个 GPU 的情况下执行的测试方法
    @skip_if_lt_x_gpu(2)
    def test_bn_always_wrapped_individually(self):
        """
        通过使用 _or_policy 和 _wrap_module_cls_individually，
        即使其他策略导致包含批归一化单元的模块被包装，也确保内部的批归一化单元仍然被逐个包装。
        """

        # 定义一个包含批归一化网络的自定义模块类
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn_container = BatchNormNet()

        # 判断是否需要包装批归一化容器的函数
        def wrap_bn_container(module, recurse, *args, **kwargs):
            if recurse:
                return True
            return isinstance(module, BatchNormNet)

        # 部分应用函数，用于逐个包装指定模块类
        wrap_batchnorm_individually = functools.partial(
            _wrap_module_cls_individually,
            module_classes=[
                _BatchNorm,
            ],
        )

        # 构建策略函数，同时考虑包装批归一化容器和逐个包装批归一化单元
        my_policy = functools.partial(
            _or_policy, policies=[wrap_bn_container, wrap_batchnorm_individually]
        )
        
        # 创建 MyModule 实例
        mod = MyModule()
        
        # 使用 FSDP 自动包装策略创建 FSDP 对象
        fsdp = FSDP(mod, auto_wrap_policy=my_policy)

        # 验证包装结果应为 FSDP(FSDP(BatchNormNet(FSDP(BN))))
        # 而非 FSDP(FSDP(BatchNormNet(BN)))（在后者中，内部 BN 未逐个包装）

        # 遍历批归一化容器中的各批归一化单元，确保每个都是 FSDP 类型
        for bn in [
            fsdp.bn_container.bn1,
            fsdp.bn_container.bn2,
            fsdp.bn_container.bn3,
            fsdp.bn_container.sync_bn,
        ]:
            self.assertTrue(isinstance(bn, FSDP))

        # 如果只包装了 BN 容器，则各个批归一化单元不会被包装
        mod = MyModule()
        fsdp = FSDP(mod, auto_wrap_policy=wrap_bn_container)
        
        # 断言批归一化容器是 FSDP 类型
        self.assertTrue(isinstance(mod.bn_container, FSDP))
        
        # 遍历批归一化容器中的各批归一化单元，确保它们不是 FSDP 类型
        for bn in [
            fsdp.bn_container.bn1,
            fsdp.bn_container.bn2,
            fsdp.bn_container.bn3,
            fsdp.bn_container.sync_bn,
        ]:
            self.assertFalse(isinstance(bn, FSDP))

    # 跳过少于两个 GPU 的情况下执行的测试方法，并使用参数化测试
    @skip_if_lt_x_gpu(2)
    @parametrize(
        "cpu_offload",
        [CPUOffload(offload_params=False), CPUOffload(offload_params=True)],
    )
    # 使用 @parametrize 装饰器为测试函数提供多组参数进行参数化测试
    @parametrize(
        "backward_prefetch",
        [BackwardPrefetch.BACKWARD_POST, BackwardPrefetch.BACKWARD_PRE],
    )
    # 使用 @parametrize 装饰器为测试函数提供多组参数进行参数化测试
    @parametrize("forward_prefetch", [False, True])
    # 使用 @parametrize 装饰器为测试函数提供多组参数进行参数化测试
    @parametrize("cuda_init_mode", [CUDAInitMode.CUDA_AFTER, CUDAInitMode.CUDA_BEFORE])
    # 测试主函数，接受多个参数进行测试
    def test_main_wrap_api(
        self,
        cpu_offload: CPUOffload,
        backward_prefetch: BackwardPrefetch,
        forward_prefetch: bool,
        cuda_init_mode: CUDAInitMode,
    ):
        # 如果 CUDA 初始化模式为 CUDA_AFTER 并且存在 CPU offload 参数，则预期它们不会同时工作，直接返回
        if cuda_init_mode == CUDAInitMode.CUDA_AFTER and cpu_offload.offload_params:
            # they don't work together, expected
            return

        # 确定是否在 CUDA 初始化之前移动模型到 CUDA
        move_to_cuda = cuda_init_mode == CUDAInitMode.CUDA_BEFORE

        # 嵌套定义一个简单的神经网络模块
        class Nested(nn.Module):
            def __init__(self):
                super().__init__()
                # 使用 _maybe_cuda 函数根据 move_to_cuda 变量决定是否将 Linear 层移动到 CUDA
                self.nested_lin = _maybe_cuda(nn.Linear(1, 1, bias=False), move_to_cuda)

            def forward(self, input):
                return self.nested_lin(input)

        # 定义一个包含嵌套模块的复杂神经网络模型
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                # 使用 _maybe_cuda 函数根据 move_to_cuda 变量决定是否将 Linear 层移动到 CUDA
                self.lin1 = _maybe_cuda(nn.Linear(1, 1, bias=False), move_to_cuda)
                # 使用 _maybe_cuda 函数根据 move_to_cuda 变量决定是否将 Linear 层移动到 CUDA
                self.lin2 = _maybe_cuda(nn.Linear(1, 1, bias=False), move_to_cuda)
                # 使用 _maybe_cuda 函数根据 move_to_cuda 变量决定是否将 Linear 层移动到 CUDA
                self.lin3 = _maybe_cuda(nn.Linear(1, 1, bias=False), move_to_cuda)
                # 创建 Nested 类的实例作为模型的一个模块
                self.lin4 = Nested()

            def forward(self, input):
                # 模型的前向传播，依次调用各个线性层和嵌套模块
                return self.lin4(self.lin3(self.lin2(self.lin1(input))))

        # 创建 MyModel 类的实例
        model = MyModel()
        # 使用 FSDP 对模型进行包装，配置自动包装策略和其他参数
        wrapped_model = FSDP(
            model,
            auto_wrap_policy=functools.partial(
                size_based_auto_wrap_policy,
                min_num_params=0,  # wrap all modules
            ),
            cpu_offload=cpu_offload,
            backward_prefetch=backward_prefetch,
            forward_prefetch=forward_prefetch,
        )
        # 如果 CUDA 初始化模式为 CUDA_AFTER，则将包装后的模型移动到 CUDA 上
        if cuda_init_mode == CUDAInitMode.CUDA_AFTER:
            wrapped_model = wrapped_model.cuda()

        # 按照 FSDP 图的顺序列出模型的各个模块
        modules_in_fsdp_graph_order = [
            wrapped_model.module.lin1,
            wrapped_model.module.lin2,
            wrapped_model.module.lin3,
            wrapped_model.module.lin4.module.nested_lin,
            wrapped_model.module.lin4,
            wrapped_model,
        ]

        # 对列出的每个模块进行断言，确保它们都是 FSDP 类的实例，并检查 CPU offload、后向预取和前向预取参数
        for module in modules_in_fsdp_graph_order:
            self.assertTrue(isinstance(module, FSDP))
            self._check_cpu_offload(module, cpu_offload)
            self._check_backward_prefetch(module, backward_prefetch)
            self._check_forward_prefetch(module, forward_prefetch)

        # 多次运行模型以进行健全性检查
        optim = torch.optim.SGD(wrapped_model.parameters(), lr=1e-2, momentum=0.9)
        inp = torch.ones(1).cuda()
        for _ in range(6):
            optim.zero_grad()
            loss = wrapped_model(inp).sum()
            loss.backward()
            optim.step()
class TestAutoWrap(TestCase):
    # 设置测试环境
    def setUp(self) -> None:
        super().setUp()
        # 对所有的测试用例，使用一个虚拟的进程组
        self.process_group = DummyProcessGroup(rank=0, size=1)

    # 根据条件跳过测试（如果不满足多GPU条件则跳过）
    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    @parametrize("wrap_method", [WrapMethod.FSDP_CTOR, WrapMethod.WRAP_API])
    def test_wrap(self, wrap_method):
        # 如果 wrap_method 是 WrapMethod.WRAP_API
        if wrap_method == WrapMethod.WRAP_API:
            # 在上下文中启用包装器 FSDP，使用指定的进程组
            with enable_wrap(wrapper_cls=FSDP, process_group=self.process_group):
                # 包装 nn.Linear(5, 5) 层
                layer = wrap(nn.Linear(5, 5))
        else:
            # 否则，确保 wrap_method 是 WrapMethod.FSDP_CTOR
            assert wrap_method == WrapMethod.FSDP_CTOR
            # 使用 FSDP 包装 nn.Linear(5, 5) 层，并设置自动包装策略
            layer = FSDP(
                nn.Linear(5, 5),
                process_group=self.process_group,
                auto_wrap_policy=functools.partial(
                    size_based_auto_wrap_policy, min_num_params=1
                ),
            )
        # 断言 layer 类型为 FSDP
        self.assertTrue(isinstance(layer, FSDP))
        # 断言 layer 的 rank 等于当前进程组的 rank
        self.assertEqual(layer.rank, self.process_group.rank())
        # 断言 layer 的 world_size 等于当前进程组的 size

        self.assertEqual(layer.world_size, self.process_group.size())

    # 根据条件跳过测试（如果不满足多GPU条件则跳过）
    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    def test_wrap_disabled_outside_context(self):
        # 使用测试环境中的进程组
        pg = self.process_group

        # 定义一个包含 wrap 操作的简单模型类
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                # 在构造函数中包装 nn.Linear(5, 5) 层，并使用指定的进程组
                self.lin = wrap(nn.Linear(5, 5), process_group=pg)

        # 创建 MyModel 实例
        model = MyModel()
        # 在上下文中启用包装器 FSDP，包装整个模型
        with enable_wrap(wrapper_cls=FSDP, process_group=pg):
            model = wrap(model)

        # 断言 model 类型为 FSDP
        self.assertTrue(isinstance(model, FSDP))
        # 断言 model.lin 不是 FSDP 类型
        self.assertFalse(isinstance(model.lin, FSDP))
        # 断言 model.lin 是 nn.Linear 类型
        self.assertTrue(isinstance(model.lin, nn.Linear))

    # 根据条件跳过测试（如果不满足多GPU条件则跳过）
    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    def test_wrap_override_defaults(self):
        # 创建一个新的虚拟进程组
        new_process_group = DummyProcessGroup(rank=0, size=2)
        # 在上下文中启用包装器 FSDP，使用新的进程组包装 nn.Linear(5, 5) 层
        with enable_wrap(wrapper_cls=FSDP, process_group=self.process_group):
            layer = wrap(nn.Linear(5, 5), process_group=new_process_group)
        # 断言 layer 类型为 FSDP
        self.assertTrue(isinstance(layer, FSDP))
        # 断言 layer 的 process_group 属性是新创建的进程组
        self.assertTrue(layer.process_group is new_process_group)
        # 断言 layer 的 rank 等于 0
        self.assertEqual(layer.rank, 0)
        # 断言 layer 的 world_size 等于 2
        self.assertEqual(layer.world_size, 2)

    # 根据条件跳过测试（如果不满足CUDA测试条件则跳过）
    @unittest.skipIf(not TEST_CUDA, "Test Requires CUDA")
    def test_always_wrap(self):
        """
        Test to ensure that if `always_wrap_policy` is
        passed into FSDP, all submodules are wrapped.
        """
        # 获取一个使用 CUDA 的嵌套序列模型
        seq = TestFSDPWrap.NestedSequentialModel.get_model(cuda=True)
        # 使用 FSDP 包装 seq 模型，并设置进程组和自动包装策略
        model = FSDP(
            seq, process_group=self.process_group, auto_wrap_policy=always_wrap_policy
        )
        # 验证所有子模块都被正确包装
        TestFSDPWrap.NestedSequentialModel.verify_model_all_wrapped(self, model)

    # 根据条件跳过测试（如果不满足多GPU条件则跳过）
    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    # 测试 transformer_auto_wrap_policy 函数
    def test_transformer_auto_wrap_policy(self):
        """Tests the ``transformer_auto_wrap_policy``."""
        # 创建一个部分应用的函数对象 auto_wrap_policy，使用 transformer_auto_wrap_policy 函数和指定的 transformer_layer_cls 参数
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerEncoderLayer, TransformerDecoderLayer},
        )
        # 调用 _test_transformer_wrapping 方法，传入 auto_wrap_policy 作为参数进行测试
        self._test_transformer_wrapping(auto_wrap_policy)

    # 如果没有多 GPU 测试跳过该测试
    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    # 测试 ModuleWrapPolicy 类
    def test_module_wrap_policy(self):
        """Tests the ``ModuleWrapPolicy``."""
        # 创建 ModuleWrapPolicy 对象 auto_wrap_policy，使用 TransformerEncoderLayer 和 TransformerDecoderLayer 类
        auto_wrap_policy = ModuleWrapPolicy(
            {TransformerEncoderLayer, TransformerDecoderLayer}
        )
        # 调用 _test_transformer_wrapping 方法，传入 auto_wrap_policy 作为参数进行测试
        self._test_transformer_wrapping(auto_wrap_policy)

    # 如果没有多 GPU 测试跳过该测试
    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    # 测试 ModuleWrapPolicy 类作为 Callable 的情况
    def test_module_wrap_policy_callable(self):
        """Tests the ``ModuleWrapPolicy`` as a ``Callable``."""
        # 创建 ModuleWrapPolicy 对象 auto_wrap_policy，使用 TransformerEncoderLayer 和 TransformerDecoderLayer 类
        auto_wrap_policy = ModuleWrapPolicy(
            {TransformerEncoderLayer, TransformerDecoderLayer}
        )
        # 创建一个部分应用的函数对象 callable_policy，使用 _or_policy 函数和 auto_wrap_policy 作为参数
        callable_policy = functools.partial(_or_policy, policies=[auto_wrap_policy])
        # 调用 _test_transformer_wrapping 方法，传入 callable_policy 作为参数进行测试
        self._test_transformer_wrapping(callable_policy)

    # 测试自定义策略
    def _test_transformer_wrapping(self, auto_wrap_policy: Union[Callable, _Policy]):
        # 设置 fsdp_kwargs 字典，其中 auto_wrap_policy 作为一个键
        fsdp_kwargs = {"auto_wrap_policy": auto_wrap_policy}
        # 使用 TransformerWithSharedParams 类初始化 fsdp_model，传入 process_group、FSDPInitMode.RECURSIVE、CUDAInitMode.CUDA_BEFORE 和 fsdp_kwargs
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_BEFORE,
            fsdp_kwargs,
        )
        # 获取 fsdp_model 中的所有模块
        modules = list(fsdp_model.modules())
        # 获取 fsdp_model 中 transformer 模块的 encoder 和 decoder 层
        encoder_layers = set(fsdp_model.module.transformer.encoder.layers)
        decoder_layers = set(fsdp_model.module.transformer.decoder.layers)
        # 遍历所有模块
        for module in modules:
            # 如果模块是 fsdp_model 本身、在 encoder_layers 中或在 decoder_layers 中，则断言模块是 FSDP 类型
            if (
                module is fsdp_model
                or module in encoder_layers
                or module in decoder_layers
            ):
                self.assertTrue(isinstance(module, FSDP))
            # 否则断言模块不是 FSDP 类型
            else:
                self.assertFalse(isinstance(module, FSDP))

    # 如果没有多 GPU 测试跳过该测试
    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    # 测试自定义策略的方法
    def test_custom_policy(self):
        """
        Tests ``CustomPolicy`` with both a lambda function that uses uniform
        kwargs (so only returns ``False`` or ``True``) and a lambda function
        that uses non-uniform kwargs (so returns a dict to override the root
        kwargs).
        """
        # 遍历 use_uniform_kwargs 取值为 False 和 True 的情况
        for use_uniform_kwargs in [False, True]:
            # 调用 _test_custom_policy 方法，传入 use_uniform_kwargs 作为参数进行测试
            self._test_custom_policy(use_uniform_kwargs)

    # 如果没有多 GPU 测试跳过该测试
    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    def test_auto_wrap_api(self):
        """
        Test to ensure with auto wrap, we wrap child modules correctly based on the min_num_params.
        ``nn.Linear(5, 5)`` does not exceed the bucket size, but combined they do.
        """
        # 获取嵌套序列模型（没有使用CUDA加速）
        sequential = TestFSDPWrap.NestedSequentialModel.get_model(cuda=False)
        # 部分函数应用，使用自定义的基于大小的自动包装策略，设置最小参数数量为40
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=40
        )
        # 创建包装后的FSDP模型，使用指定的过程组和自动包装策略
        model = FSDP(
            sequential,
            process_group=self.process_group,
            auto_wrap_policy=my_auto_wrap_policy,
        )

        # 验证模型是否正确
        TestFSDPWrap.NestedSequentialModel.verify_model(self, model)

    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    def test_auto_wrap_preset_exclude_wrap(self):
        """
        Test to ensure excluded modules are not wrapped, regardless if the total param size is greater than the
        min_num_params. the size_based_auto_wrap_policy excludes wrapping for {nn.ModuleList, nn.ModuleDict}
        """
        # 创建包含两个线性层的模块列表
        sequential = nn.ModuleList([nn.Linear(5, 5), nn.Linear(5, 5)])
        # 部分函数应用，使用自定义的基于大小的自动包装策略，设置最小参数数量为40
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=40
        )

        # 创建包装后的FSDP模型，使用指定的过程组和自动包装策略
        model = FSDP(
            sequential,
            process_group=self.process_group,
            auto_wrap_policy=my_auto_wrap_policy,
        )

        # 断言模型及其子模块类型是否符合预期
        self.assertTrue(isinstance(model, FSDP))
        self.assertTrue(isinstance(model[0], nn.Linear))
        self.assertTrue(isinstance(model[1], nn.Linear))

    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    def test_auto_wrap_preset_exclude_wrap_include_children(self):
        """
        Test to ensure excluded modules are not wrapped, but children are if param size is greater than
        min_num_params
        """
        # 创建包含一个线性层的模块列表
        sequential = nn.ModuleList([nn.Linear(10, 10)])
        # 部分函数应用，使用自定义的基于大小的自动包装策略，设置最小参数数量为40
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=40
        )
        
        # 创建包装后的FSDP模型，使用指定的过程组和自动包装策略
        model = FSDP(
            sequential,
            process_group=self.process_group,
            auto_wrap_policy=my_auto_wrap_policy,
        )

        # 断言模型及其子模块类型是否符合预期
        self.assertTrue(isinstance(model, FSDP))
        self.assertTrue(isinstance(model[0], FSDP))
    def test_auto_wrap_preset_force_leaf(self):
        """
        测试确保强制叶子模块不被包装，并且子模块不被包装。
        size_based_auto_wrap_policy 强制 {nn.MultiheadAttention} 类型的叶子模块不被包装。
        """
        # 创建一个包含线性层和多头注意力层的序列模型
        sequential = nn.Sequential(nn.Linear(10, 10), nn.MultiheadAttention(100, 1))
        # 定义自定义的自动包装策略，基于模型参数数量
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=40
        )
        # 将序列模型用FSDP封装
        model = FSDP(
            sequential,
            process_group=self.process_group,
            auto_wrap_policy=my_auto_wrap_policy,
        )
        # 断言第一个子模块已经被FSDP包装
        self.assertTrue(isinstance(model.module[0], FSDP))
        # 断言多头注意力层的子模块没有被包装
        self.assertTrue(isinstance(model.module[1], nn.MultiheadAttention))
        # 断言多头注意力层的输出投影层是线性层
        self.assertTrue(isinstance(model.module[1].out_proj, nn.Linear))

    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    def test_auto_wrap_preset_force_leaf_custom(self):
        """
        测试确保强制叶子模块不被包装。
        """
        # 定义自定义的自动包装策略，基于模型参数数量，并强制包含线性模块作为叶子模块
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=40,
            force_leaf_modules=size_based_auto_wrap_policy.FORCE_LEAF_MODULES.union(
                {nn.Linear}
            ),
        )
        # 创建一个包含线性层和模块列表的序列模型
        sequential = nn.Sequential(
            nn.Linear(10, 10), nn.ModuleList([nn.Linear(10, 10)])
        )
        # 将序列模型用FSDP封装
        model = FSDP(
            sequential,
            process_group=self.process_group,
            auto_wrap_policy=my_auto_wrap_policy,
        )
        # 断言整个模型被FSDP包装，因为没有内部模块被包装
        self.assertTrue(isinstance(model, FSDP))
        # 断言第一个子模块是线性层
        self.assertTrue(isinstance(model.module[0], nn.Linear))
        # 断言第二个子模块是模块列表
        self.assertTrue(isinstance(model.module[1], nn.ModuleList))

    @unittest.skipIf(not TEST_CUDA, "Test Requires CUDA")
    @parametrize("cuda_init_mode", [CUDAInitMode.CUDA_BEFORE, CUDAInitMode.CUDA_AFTER])
    @parametrize(
        "cpu_offload",
        [CPUOffload(offload_params=False), CPUOffload(offload_params=True)],
    )
    @parametrize("use_device_id", [True, False])
    # 定义自动包装功能的烟雾测试方法，用于测试不同的初始化模式和设备配置
    def test_auto_wrap_smoke_test(self, cuda_init_mode, cpu_offload, use_device_id):
        # 如果存在 CPU 卸载参数且 CUDA 初始化模式为 CUDA_AFTER，则返回，因为它们不兼容
        if cpu_offload.offload_params and cuda_init_mode == CUDAInitMode.CUDA_AFTER:
            return

        # 设置当前设备为 CUDA 设备
        device = torch.device("cuda")
        torch.cuda.set_device(0)
        # 如果使用设备 ID，则获取当前 CUDA 设备作为设备 ID
        device_id = (
            torch.device("cuda", torch.cuda.current_device()) if use_device_id else None
        )

        # 随机选择一个端口以避免冲突
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())

        # 创建临时文件以作为初始化方法的一部分
        file_name = tempfile.NamedTemporaryFile(delete=False).name
        # 使用 NCCL 后端初始化分布式进程组
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"{FILE_SCHEMA}_{file_name}",
            rank=0,
            world_size=1,
        )

        # NOTE: 我们在使用 FSDP 后才将模型移动到 CUDA 上，以模拟真实场景
        # 其中整个模型无法加载到 GPU，但它们的分片可以。
        cuda_after_init = cuda_init_mode == CUDAInitMode.CUDA_AFTER
        try:
            # 获取嵌套序列模型
            sequential = TestFSDPWrap.NestedSequentialModel.get_model(
                cuda=(not cuda_after_init)
            )
            # 定义基于大小的自动包装策略
            my_auto_wrap_policy = functools.partial(
                size_based_auto_wrap_policy, min_num_params=40
            )
            # 使用 FSDP 封装模型
            model = FSDP(
                sequential,
                cpu_offload=cpu_offload,
                auto_wrap_policy=my_auto_wrap_policy,
                device_id=device_id,
            )
            # 验证模型是否正确
            TestFSDPWrap.NestedSequentialModel.verify_model(self, model)
            # 如果初始化后需要将模型移动到 CUDA
            if cuda_after_init:
                model = model.cuda()
            # 创建输入张量并将其移动到指定设备
            input = torch.rand((1, 5), dtype=torch.float).to(device)
            # 使用模型进行推理
            output = model(input)
            # 计算损失
            loss = F.mse_loss(input, output)
            # 反向传播损失
            loss.backward()
        finally:
            # 销毁分布式进程组
            torch.distributed.destroy_process_group()

        try:
            # 移除临时文件
            os.remove(file_name)
        except FileNotFoundError:
            pass
    # 定义一个测试方法，用于测试在忽略指定模块的情况下进行自动包装
    def test_auto_wrap_with_ignored_modules(self, wrap_method: WrapMethod):
        # 获取一个嵌套的序列模型，不使用 CUDA
        sequential = TestFSDPWrap.NestedSequentialModel.get_model(cuda=False)
        # 指定要忽略的模块列表
        ignored_modules = [sequential[1], sequential[2][0]]
        # 定义一个自定义的自动包装策略，基于模型参数数量
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=40,
        )
        # 配置 FSDP 所需的参数，包括进程组和自动包装策略
        fsdp_kwargs = {
            "process_group": self.process_group,
            "auto_wrap_policy": my_auto_wrap_policy,
            "ignored_modules": ignored_modules,
        }
        # 根据包装方法选择相应的处理方式
        if wrap_method == WrapMethod.FSDP_CTOR:
            # 使用 FSDP 类直接对模型进行包装
            model = FSDP(sequential, **fsdp_kwargs)
        elif wrap_method == WrapMethod.WRAP_API:
            # 使用 enable_wrap 上下文管理器对模型进行包装
            with enable_wrap(wrapper_cls=FSDP, **fsdp_kwargs):
                model = wrap(sequential)
        else:
            # 如果选择了未支持的包装方法，则断言失败
            assert 0, f"Unsupported wrap method: {wrap_method}"
        # 断言：所有未被忽略的模块应该被 FSDP 包装
        self.assertTrue(isinstance(model, FSDP))
        self.assertTrue(isinstance(model.module[0], nn.Linear))
        self.assertTrue(isinstance(model.module[1], nn.Linear))
        self.assertTrue(isinstance(model.module[2], nn.Sequential))
        self.assertTrue(isinstance(model.module[2][0], nn.Linear))
        self.assertTrue(isinstance(model.module[2][1], nn.Linear))
    def test_frozen_params(self):
        """
        Tests that mixing frozen/non-frozen parameters in an FSDP instance
        raises for ``use_orig_params=False`` and warns for ``True``.
        """
        # 定义被测试的模块类列表
        module_classes = (LoraAttention, LoraMLP, LoraDecoder)
        # 创建模块包装策略对象
        module_wrap_policy = ModuleWrapPolicy(module_classes)

        # 定义检查模块是否符合均匀封装条件的lambda函数
        def lambda_fn_uniform(module: nn.Module):
            return isinstance(module, module_classes)

        # 定义检查模块是否符合非均匀封装条件的lambda函数
        def lambda_fn_nonuniform(module: nn.Module):
            if isinstance(module, LoraAttention):
                return {"sharding_strategy": ShardingStrategy.SHARD_GRAD_OP}
            elif isinstance(module, module_classes):
                return True
            return False

        # 创建基于lambda函数的自定义封装策略对象
        lambda_wrap_policy_uniform = CustomPolicy(lambda_fn_uniform)
        lambda_wrap_policy_nonuniform = CustomPolicy(lambda_fn_nonuniform)

        # 遍历参数组合进行测试
        for use_orig_params, policy in itertools.product(
            [True, False],
            [
                module_wrap_policy,
                lambda_wrap_policy_uniform,
                lambda_wrap_policy_nonuniform,
            ],
        ):
            # 调用内部方法进行具体的参数冻结测试
            self._test_frozen_params(use_orig_params, policy)

    def _test_frozen_params(self, use_orig_params: bool, policy: _Policy):
        # 创建LoraModel模型并放置在GPU上
        model = LoraModel().cuda()
        # 定义异常消息字符串的起始部分
        msg = "layers.0.attn has both parameters with requires_grad=True and False. "
        
        if use_orig_params:
            # 在异常消息末尾追加警告信息，因为use_orig_params为True
            msg += "We do not recommend wrapping such modules"
            # 断言会发出UserWarning并匹配消息字符串
            ctx = self.assertWarnsRegex(UserWarning, msg)
        else:
            # 在异常消息末尾追加错误信息，因为use_orig_params为False
            msg += "FSDP does not support wrapping such modules when use_orig_params=False."
            # 断言会抛出ValueError并匹配消息字符串
            ctx = self.assertRaisesRegex(ValueError, msg)
        
        with ctx:
            # 创建FSDP对象，并传入模型、处理组、封装策略和参数冻结标志
            FSDP(
                model,
                process_group=self.process_group,
                auto_wrap_policy=policy,
                use_orig_params=use_orig_params,
            )
class TestWrapUtils(TestCase):
    # 定义一个测试类 TestWrapUtils，继承自 TestCase

    def test_validate_frozen_params(self):
        """Tests the method ``_validate_frozen_params()``."""
        # 定义测试方法 test_validate_frozen_params，用于测试 _validate_frozen_params 方法
        for use_orig_params in [True, False]:
            # 遍历布尔值 True 和 False，分别作为参数调用 _test_validate_frozen_params 方法
            self._test_validate_frozen_params(use_orig_params)

instantiate_parametrized_tests(TestFSDPWrap)
# 使用参数化测试工具实例化 TestFSDPWrap 类的测试

instantiate_parametrized_tests(TestAutoWrap)
# 使用参数化测试工具实例化 TestAutoWrap 类的测试

if __name__ == "__main__":
    # 如果该脚本作为主程序运行
    run_tests()
    # 运行所有的测试用例
```