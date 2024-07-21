# `.\pytorch\test\distributed\fsdp\test_fsdp_use_orig_params.py`

```py
# Owner(s): ["oncall: distributed"]

import copy  # 导入 copy 模块，用于对象的深拷贝操作
import functools  # 导入 functools 模块，用于高阶函数操作
import itertools  # 导入 itertools 模块，用于创建和操作迭代器的函数
import os  # 导入 os 模块，提供了丰富的方法用于处理文件和目录
import sys  # 导入 sys 模块，提供了对解释器相关的操作
import unittest  # 导入 unittest 模块，用于编写和运行单元测试
from typing import Any, Dict, List, Optional, Tuple, Type  # 导入类型提示相关的工具

import torch  # 导入 PyTorch 深度学习库
import torch.nn as nn  # 导入 PyTorch 神经网络模块
from torch import distributed as dist  # 导入 PyTorch 分布式训练相关的模块
from torch.distributed.fsdp import (  # 导入 FSDP 模块的子模块和类
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp._common_utils import clean_tensor_name  # 导入 FSDP 模块中的工具函数
from torch.distributed.fsdp._flat_param import (  # 导入 FSDP 模块中的常量和变量
    _FSDP_SKIP_WRITEBACK_CHECK,
    _FSDP_USE_FULL_PREC_IN_EVAL,
)
from torch.distributed.fsdp._init_utils import NO_RESHARD_AFTER_FORWARD_STRATEGIES  # 导入 FSDP 模块中的常量
from torch.distributed.fsdp.wrap import always_wrap_policy, ModuleWrapPolicy  # 导入 FSDP 模块中的策略和类
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer  # 导入 PyTorch 中的特定层
from torch.nn.parallel.distributed import DistributedDataParallel as DDP  # 导入 PyTorch 分布式训练模块
from torch.testing._internal.common_cuda import TEST_CUDA  # 导入 PyTorch 测试相关的 CUDA 模块
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入 PyTorch 分布式训练测试工具
from torch.testing._internal.common_fsdp import (  # 导入 PyTorch FSDP 测试相关的模块和类
    CUDAInitMode,
    FSDPInitMode,
    FSDPTest,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (  # 导入 PyTorch 测试相关的常用工具函数
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)
from torch.utils._triton import has_triton  # 导入 Triton 相关的工具函数

# 如果分布式训练不可用，则打印错误信息并退出
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 如果处于开发调试模式下，避免使用 dev-asan 因为 torch + multiprocessing spawn 存在已知问题
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestFSDPUseOrigParamsMultipleParamGroups(FSDPTest):
    """Tests multiple parameter groups."""

    @property
    def world_size(self) -> int:
        return 2

    def _get_param_groups(self, model: nn.Module) -> List[Dict[str, Any]]:
        """
        Constructs separate parameter groups for weights, biases, and other
        parameters.
        """
        param_groups = [
            {"params": [], "weight_decay": 0.1, "lr": 1e-2},  # 第一个参数组，包含权重衰减和学习率
            {"params": [], "weight_decay": 0.01, "lr": 1e-3},  # 第二个参数组，包含权重衰减和学习率
            {"params": []},  # 第三个参数组，无权重衰减和学习率设置
        ]
        # 遍历模型的所有参数，根据名称将参数分组到对应的参数组中
        for param_name, param in model.named_parameters():
            if "weight" in param_name:
                param_groups[0]["params"].append(param)  # 将权重参数添加到第一个参数组
            elif "bias" in param_name:
                param_groups[1]["params"].append(param)  # 将偏置参数添加到第二个参数组
            else:
                param_groups[2]["params"].append(param)  # 将其他参数添加到第三个参数组
        return param_groups

    def _get_optim(
        self,
        model: nn.Module,
        optim_class: Type[torch.optim.Optimizer],
        multi_tensor: bool,
    ) -> torch.optim.Optimizer:
        """
        Constructs an Adam optimizer with three parameter groups, one for
        weights, one for biases, and one for everything else, each with
        different weight decay and learning rates.
        """
        # 获取模型的参数组
        param_groups = self._get_param_groups(model)
        # 使用给定的优化器类构建 Adam 优化器，设置学习率为 5e-3，并且使用多元张量
        return optim_class(param_groups, lr=5e-3, foreach=multi_tensor)

    def _get_ddp_transformer(self, find_unused_params: bool) -> DDP:
        """Returns a transformer with shared parameters wrapped with DDP."""
        # 初始化带有共享参数的 Transformer 模型
        model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
            deterministic=True,
        )
        # 使用 DistributedDataParallel (DDP) 将模型包装起来
        ddp_model = DDP(
            model,
            device_ids=[self.rank],
            find_unused_parameters=find_unused_params,
        )
        return ddp_model

    def _get_fsdp_transformer_and_optim(
        self,
        cuda_init_mode: CUDAInitMode,
        init_optim_before_wrap: bool,
        optim_class: Type[torch.optim.Optimizer],
        multi_tensor: bool,
        sharding_strategy: ShardingStrategy,
        backward_prefetch: Optional[BackwardPrefetch],
        cpu_offload: CPUOffload,
    ) -> Tuple[FSDP, torch.optim.Optimizer]:
        """
        Returns a transformer with shared parameters wrapped with FSDP and a
        corresponding optimizer.
        """
        # 定义 FSDP 初始化参数
        fsdp_kwargs = {
            "auto_wrap_policy": ModuleWrapPolicy(
                {
                    TransformerEncoderLayer,
                    TransformerDecoderLayer,
                }
            ),
            "use_orig_params": True,
            "sharding_strategy": sharding_strategy,
            "backward_prefetch": backward_prefetch,
            "cpu_offload": cpu_offload,
        }
        # 初始化带有共享参数的 Transformer 模型
        model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            cuda_init_mode,
            deterministic=True,
        )
        if init_optim_before_wrap:
            # 如果在包装前初始化优化器，则先获取优化器并包装模型
            fsdp_optim = self._get_optim(model, optim_class, multi_tensor)
            fsdp_model = FSDP(model, self.process_group, **fsdp_kwargs)
        else:
            # 否则先包装模型，然后获取优化器
            fsdp_model = FSDP(model, self.process_group, **fsdp_kwargs)
            fsdp_optim = self._get_optim(fsdp_model, optim_class, multi_tensor)
        # 如果 CUDA 初始化模式为 CUDA_AFTER 并且未开启 CPU 降载，则将模型移动到 GPU
        if (
            cuda_init_mode == CUDAInitMode.CUDA_AFTER
            and not fsdp_model.cpu_offload.offload_params
        ):
            fsdp_model = fsdp_model.cuda()
        return fsdp_model, fsdp_optim
    def _check_train_parity(
        self,
        ddp_model: DDP,
        ddp_optim: torch.optim.Optimizer,
        fsdp_model: FSDP,
        fsdp_optim: torch.optim.Optimizer,
        set_to_none: bool,
        num_iters: int = 10,
    ):
        """Checks training parity between DDP and FSDP."""
        device = torch.device("cuda")  # 设备选择为CUDA加速器
        for i in range(num_iters):  # 循环执行指定次数的迭代
            iter_losses = []  # 存储每个迭代的损失值
            for model, optim in ((ddp_model, ddp_optim), (fsdp_model, fsdp_optim)):
                module = model.module  # 获取模型的模块部分
                # 测试两种不同的`zero_grad()`时机
                if i % 2 == 0:
                    optim.zero_grad(set_to_none=set_to_none)  # 在前向传播前执行梯度清零操作
                inp = module.get_input(device)  # 获取输入数据并将其移到指定设备上
                output = model(*inp)  # 模型前向传播
                loss = module.get_loss(inp, output).to(device)  # 计算损失并将其移到指定设备上
                iter_losses.append(loss)  # 将损失值添加到迭代损失列表中
                if i % 2 == 1:
                    optim.zero_grad(set_to_none=set_to_none)  # 在反向传播前执行梯度清零操作
                module.run_backward(loss)  # 执行反向传播
                # 如果需要，将 DDP 模型的参数移到CPU执行以匹配 FSDP
                if model is ddp_model and fsdp_model.cpu_offload.offload_params:
                    model.to(torch.device("cpu"))
                optim.step()  # 执行优化器的更新步骤
                if model is ddp_model and fsdp_model.cpu_offload.offload_params:
                    model.to(device)  # 将模型参数移到指定设备上
            torch.testing.assert_close(iter_losses[0], iter_losses[1])  # 断言两个迭代的损失值接近
            iter_losses.clear()  # 清空迭代损失列表
        self._check_ddp_fsdp_param_parity(ddp_model, fsdp_model)  # 检查 DDP 和 FSDP 参数的一致性

    def _check_ddp_fsdp_param_parity(self, ddp_model: DDP, fsdp_model: FSDP):
        with FSDP.summon_full_params(fsdp_model):  # 使用 FSDP 提取完整参数
            for (n1, p1), (n2, p2) in zip(
                ddp_model.module.named_parameters(), fsdp_model.named_parameters()
            ):
                # 允许 FSDP 的前缀存在
                self.assertEqual(n1, clean_tensor_name(n2))  # 断言参数名称一致
                torch.testing.assert_close(p1, p2)  # 断言参数值接近

    def _get_sharding_strategy_from_str(
        self, sharding_strategy_str: str
    ) -> ShardingStrategy:
        if sharding_strategy_str == "no_shard":
            sharding_strategy = ShardingStrategy.NO_SHARD  # 选择不分片策略
        elif sharding_strategy_str == "shard_grad_op":
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP  # 选择梯度操作分片策略
        elif sharding_strategy_str == "full_shard":
            sharding_strategy = ShardingStrategy.FULL_SHARD  # 选择完全分片策略
        else:
            raise ValueError(f"Invalid string: {sharding_strategy_str}")  # 抛出值错误异常
        return sharding_strategy  # 返回选择的分片策略

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)  # 如果GPU数小于2，则跳过该测试
    # 定义测试函数 test_fsdp_compile，用于测试 FSDP 编译功能
    def test_fsdp_compile(self):
        # 运行子测试，测试不同的分片策略和是否跳过 FSDP 保护
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.NO_SHARD,
                ],
                "skip_fsdp_guards": [True, False],
            },
            self._test_fsdp_compile,
        )

    # 定义内部测试函数 _test_fsdp_compile，接受分片策略和是否跳过 FSDP 保护作为参数
    def _test_fsdp_compile(
        self, sharding_strategy: ShardingStrategy, skip_fsdp_guards: bool
    ):
        # 设置全局的 skip_fsdp_guards 参数
        torch._dynamo.config.skip_fsdp_guards = skip_fsdp_guards
        # 准备 FSDP 的相关参数
        fsdp_kwargs = {
            "auto_wrap_policy": ModuleWrapPolicy(
                {
                    TransformerEncoderLayer,
                    TransformerDecoderLayer,
                }
            ),
            "use_orig_params": True,
            "sharding_strategy": sharding_strategy,
            "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
            "cpu_offload": CPUOffload(False),
        }
        # 初始化基础模型 base_model，使用特定的初始化模式和参数
        base_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
            deterministic=True,
        )
        # 使用深拷贝创建参考模型 ref_model，并配置为使用 FSDP
        ref_model = FSDP(copy.deepcopy(base_model), self.process_group, **fsdp_kwargs)
        # 使用 Adam 优化器初始化参考模型的优化器 ref_optim
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        # 使用深拷贝创建模型 model，并配置为使用 FSDP
        model = FSDP(copy.deepcopy(base_model), self.process_group, **fsdp_kwargs)
        # 对模型进行编译，返回编译后的模型
        model = torch.compile(model)
        # 使用 Adam 优化器初始化编译后模型的优化器 optim
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        # 进行 10 轮训练
        for i in range(10):
            losses = []
            # 获取参考模型的输入 inp，在 CUDA 设备上
            inp = ref_model.get_input(torch.device("cuda"))
            # 对参考模型和编译后模型分别进行优化
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad()
                # 计算模型输出的损失值
                loss = _model(*inp).sum()
                losses.append(loss)
                # 反向传播
                loss.backward()
                # 更新优化器
                _optim.step()
            # 断言两个模型的损失值相等
            self.assertEqual(losses[0], losses[1])

    # 根据 GPU 数量决定是否跳过测试
    @skip_if_lt_x_gpu(2)
    # 参数化测试，测试不同的分片策略字符串表示
    @parametrize(
        "sharding_strategy_str",
        ["no_shard", "shard_grad_op", "full_shard"],
    )
    def test_diff_hyperparams(self, sharding_strategy_str: str):
        """
        Tests FSDP parity with DDP when using multiple parameter groups with
        different hyperparameter settings.
        """
        # 将输入的 sharding_strategy_str 字符串转换为对应的分片策略对象
        sharding_strategy = self._get_sharding_strategy_from_str(sharding_strategy_str)
        
        # 运行子测试，测试不同的超参数组合
        self.run_subtests(
            {
                "cuda_init_mode": [
                    CUDAInitMode.CUDA_BEFORE,
                    CUDAInitMode.CUDA_AFTER,
                ],
                "init_optim_before_wrap": [False, True],
                "optim_class": [torch.optim.AdamW],
                "multi_tensor": [False, True],
                "set_to_none": [False, True],
                "backward_prefetch": [
                    None,
                    BackwardPrefetch.BACKWARD_PRE,
                    BackwardPrefetch.BACKWARD_POST,
                ],
                "skip_writeback_check": [False, True],
            },
            self._test_diff_hyperparams,  # 传递测试函数的引用作为回调函数
            cpu_offload=CPUOffload(offload_params=False),  # 设置 CPU offload 参数对象
            sharding_strategy=sharding_strategy,  # 设置分片策略对象
        )

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "sharding_strategy_str",
        ["no_shard", "shard_grad_op", "full_shard"],
    )
    def test_diff_hyperparams_cpu_offload(self, sharding_strategy_str: str):
        """
        Tests FSDP parity with DDP when using multiple parameter groups with
        different hyperparameter settings with CPU offloading enabled. This is
        separate from :meth:`test_diff_hyperparams` because CPU offloading has
        some issues with subtesting for some specific subtesting configs (e.g.,
        with ``offload_params=False`` followed by ``True`` but not vice versa).
        """
        # 将输入的 sharding_strategy_str 字符串转换为对应的分片策略对象
        sharding_strategy = self._get_sharding_strategy_from_str(sharding_strategy_str)
        
        # 针对不同的 skip_writeback_check 值进行循环测试
        for skip_writeback_check in (False, True):
            # 调用 _test_diff_hyperparams 方法进行测试，传递不同的参数组合
            self._test_diff_hyperparams(
                cuda_init_mode=CUDAInitMode.CUDA_BEFORE,
                init_optim_before_wrap=False,
                optim_class=torch.optim.Adam,
                multi_tensor=False,
                set_to_none=False,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                cpu_offload=CPUOffload(offload_params=True),  # 开启 CPU offload
                sharding_strategy=sharding_strategy,  # 设置分片策略对象
                skip_writeback_check=skip_writeback_check,
            )

    def _test_diff_hyperparams(
        self,
        cuda_init_mode: CUDAInitMode,
        init_optim_before_wrap: bool,
        optim_class: Type[torch.optim.Optimizer],
        multi_tensor: bool,
        set_to_none: bool,
        backward_prefetch: Optional[BackwardPrefetch],
        cpu_offload: CPUOffload,
        sharding_strategy: ShardingStrategy,
        skip_writeback_check: bool,
    ):
        """
        Args:
            init_optim_before_wrap (bool): If ``True``, initializes the
                FSDP optimizer before wrapping the model with FSDP; otherwise,
                initializes the FSDP optimizer after wrapping the model with
                FSDP. We permit both forms of initialization to give users
                flexibility.
        """
        # 如果初始化模式为 CUDA_AFTER 且需要 CPU offload，则不支持该模式
        if cuda_init_mode == CUDAInitMode.CUDA_AFTER and cpu_offload.offload_params:
            return  # not supported

        # 如果设置了 skip_writeback_check 标志，设置环境变量 _FSDP_SKIP_WRITEBACK_CHECK 为 "1"
        if skip_writeback_check:
            os.environ[_FSDP_SKIP_WRITEBACK_CHECK] = "1"

        # 获取使用 DDP 转换后的模型
        ddp_model = self._get_ddp_transformer(find_unused_params=False)

        # 使用给定的优化器类和 multi_tensor 标志获取 DDP 模型的优化器
        ddp_optim = self._get_optim(ddp_model, optim_class, multi_tensor)

        # 获取使用 FSDP 转换后的模型和优化器
        fsdp_model, fsdp_optim = self._get_fsdp_transformer_and_optim(
            cuda_init_mode=cuda_init_mode,
            init_optim_before_wrap=init_optim_before_wrap,
            optim_class=optim_class,
            multi_tensor=multi_tensor,
            sharding_strategy=sharding_strategy,
            backward_prefetch=backward_prefetch,
            cpu_offload=cpu_offload,
        )

        # 检查训练过程中 DDP 模型和 FSDP 模型的参数是否一致
        self._check_train_parity(
            ddp_model, ddp_optim, fsdp_model, fsdp_optim, set_to_none
        )

    @skip_if_lt_x_gpu(2)
    def test_diff_trainability(self):
        """
        Tests FSDP parity with DDP when using multiple parameter groups and
        freezing the parameters in one parameter group.
        """
        # 运行多个子测试，以测试不同训练性能
        self.run_subtests(
            {
                "multi_tensor": [False, True],
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.NO_SHARD,
                ],
            },
            self._test_diff_trainability,
        )

    def _test_diff_trainability(
        self,
        multi_tensor: bool,
        sharding_strategy: ShardingStrategy,
    ):
        # 使用 Adam 优化器类
        optim_class = torch.optim.Adam
        
        # 获取使用 DDP 转换后的模型，确保找到未使用的参数
        ddp_model = self._get_ddp_transformer(find_unused_params=True)
        
        # 使用给定的优化器类和 multi_tensor 标志获取 DDP 模型的优化器
        ddp_optim = self._get_optim(ddp_model, optim_class, multi_tensor)
        
        # 获取使用 FSDP 转换后的模型和优化器，初始化模式为 CUDA_BEFORE
        fsdp_model, fsdp_optim = self._get_fsdp_transformer_and_optim(
            cuda_init_mode=CUDAInitMode.CUDA_BEFORE,
            init_optim_before_wrap=False,
            optim_class=optim_class,
            multi_tensor=multi_tensor,
            sharding_strategy=sharding_strategy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            cpu_offload=None,
        )

        # 冻结所有偏置项参数（它们恰好在同一参数组中）
        for param_name, param in ddp_model.named_parameters():
            if "bias" in param_name:
                param.requires_grad_(False)
        
        # 冻结 FSDP 模型中所有偏置项参数
        for param_name, param in fsdp_model.named_parameters():
            if "bias" in param_name:
                param.requires_grad_(False)
        
        # 检查训练过程中 DDP 模型和 FSDP 模型的参数是否一致
        self._check_train_parity(ddp_model, ddp_optim, fsdp_model, fsdp_optim, False)
    @skip_if_lt_x_gpu(2)
    # 使用装饰器 @skip_if_lt_x_gpu(2)，如果当前 GPU 数量少于 2，则跳过这个测试
    def test_multiple_optimizers(self):
        """
        Tests using two optimizers where only one sets gradients to ``None``.
        """
        # 运行子测试，传入参数包括分片策略和测试函数
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                ]
            },
            self._test_multiple_optimizers,  # 传入测试函数的引用
        )
    # 定义测试类 TestFSDPUseOrigParamsUnshardReshard，继承自 FSDPTest，测试未分片/重分片流程
    class TestFSDPUseOrigParamsUnshardReshard(FSDPTest):
        """Tests the unshard/reshard flow."""

        # 定义属性 world_size，返回整数值 2
        @property
        def world_size(self) -> int:
            return 2

        # 定义方法 _get_fsdp_models_and_optims，返回两对 (FSDP 模型, 优化器)，分别对应 use_orig_params=False 和 True
        def _get_fsdp_models_and_optims(
            self,
            sharding_strategy: ShardingStrategy,
            cpu_offload: CPUOffload,
        ) -> Tuple[FSDP, torch.optim.Optimizer, FSDP, torch.optim.Optimizer]:
            """
            Returns a pair of (FSDP model, optimizer) for ``use_orig_params=False``
            and ``True``, respectively.
            """
            # 设置学习率 LR
            LR = 1e-2
            # 构建 FSDP 参数字典
            fsdp_kwargs = {
                "sharding_strategy": sharding_strategy,
                "cpu_offload": cpu_offload,
                "use_orig_params": False,
            }
            # 初始化使用共享参数的 TransformerWithSharedParams 模型
            fsdp_model = TransformerWithSharedParams.init(
                self.process_group,
                FSDPInitMode.RECURSIVE,
                CUDAInitMode.CUDA_BEFORE,
                fsdp_kwargs=fsdp_kwargs,
                deterministic=True,
            )
            # 使用 Adam 优化器初始化模型参数
            optim = torch.optim.Adam(fsdp_model.parameters(), foreach=False, lr=LR)
            # 切换 use_orig_params 为 True
            fsdp_kwargs["use_orig_params"] = True
            # 初始化使用原始参数的 TransformerWithSharedParams 模型
            fsdp_model_orig_params = TransformerWithSharedParams.init(
                self.process_group,
                FSDPInitMode.RECURSIVE,
                CUDAInitMode.CUDA_BEFORE,
                fsdp_kwargs=fsdp_kwargs,
                deterministic=True,
            )
            # 使用 Adam 优化器初始化原始参数模型
            optim_orig_params = torch.optim.Adam(
                fsdp_model_orig_params.parameters(), foreach=False, lr=LR
            )
            # 返回模型和优化器
            return fsdp_model, optim, fsdp_model_orig_params, optim_orig_params

        # 定义方法 _check_fsdp_parameter_parity，检查两个 FSDP 实例的模型参数是否相同
        def _check_fsdp_parameter_parity(self, fsdp1: FSDP, fsdp2: FSDP) -> None:
            """Checks that two FSDP instances have the same model parameters."""
            # 使用 FSDP.summon_full_params 比较两个 FSDP 实例的全部参数
            with FSDP.summon_full_params(fsdp1), FSDP.summon_full_params(fsdp2):
                # 遍历并比较两个 FSDP 实例的命名参数
                for (n1, p1), (n2, p2) in zip(
                    fsdp1.named_parameters(),
                    fsdp2.named_parameters(),
                ):
                    self.assertEqual(n1, n2)  # 断言参数名称相同
                    torch.testing.assert_close(p1, p2)  # 使用 Torch 的测试工具检查参数值的接近程度

        # 定义方法 _get_fsdp_parity_subtest_config，返回分片测试的配置字典
        def _get_fsdp_parity_subtest_config(self):
            return {
                "sharding_strategy": [
                    ShardingStrategy.NO_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.FULL_SHARD,
                ],
            }

        # 使用装饰器 skip_if_lt_x_gpu(2) 标记测试方法 test_multiple_forward，测试多次前向传播
        @skip_if_lt_x_gpu(2)
        @parametrize("offload_params", [False, True])
        def test_multiple_forward(self, offload_params: bool):
            """
            Tests that ``use_orig_params=True`` has parity with ``False`` when
            running multiple forward passes before a backward pass.
            """
            # 根据 offload_params 构建 CPUOffload 对象
            cpu_offload = CPUOffload(offload_params=offload_params)
            # 运行子测试，测试多次前向传播
            self.run_subtests(
                self._get_fsdp_parity_subtest_config(),
                self._test_multiple_forward,
                cpu_offload=cpu_offload,
            )

        # 使用装饰器 skip_if_lt_x_gpu(2) 标记测试方法 _test_multiple_forward
        @skip_if_lt_x_gpu(2)
        def _test_multiple_forward(
            self,
            sharding_strategy: ShardingStrategy,
            cpu_offload: CPUOffload,
        ):
            """
            Tests that ``use_orig_params=True`` has parity with ``False`` when
            running multiple forward passes before a backward pass.
            """
            # 调用 run_subtests 方法，测试多次前向传播
            cpu_offload = CPUOffload(offload_params=offload_params)
            self.run_subtests(
                self._get_fsdp_parity_subtest_config(),
                self._test_multiple_forward,
                cpu_offload=cpu_offload,
            )
    ):
        (
            fsdp_model,
            optim,
            fsdp_model_orig_params,
            optim_orig_params,
        ) = self._get_fsdp_models_and_optims(sharding_strategy, cpu_offload)
        # 获取分布式模型和优化器，包括模型和原始参数，以及优化器和原始参数
        device = torch.device("cuda")
        # 将设备设置为 CUDA
        for _ in range(3):
            # 循环三次，每次获取模型的输入数据
            inp1 = fsdp_model.get_input(device)
            _inp2 = fsdp_model.get_input(device)
            inp2 = tuple(
                t + torch.ones_like(t) for t in _inp2
            )  # 使其与 `inp1` 不同
            # 创建另一个输入，与 `inp1` 不同
            # 对于这些损失列表：第0个元素是基线；第1个元素是测试
            losses1 = []
            losses2 = []
            losses = []
            for _model, _optim in (fsdp_model, optim), (
                fsdp_model_orig_params,
                optim_orig_params,
            ):
                # 循环两次，分别使用分布式模型和优化器，以及原始模型参数和优化器参数
                _optim.zero_grad()
                # 将优化器的梯度归零
                loss1 = _model(*inp1)
                losses1.append(loss1)
                # 计算第一个输入的损失并添加到损失列表1
                loss2 = _model(*inp2)
                losses2.append(loss2)
                # 计算第二个输入的损失并添加到损失列表2
                loss = (loss1 + loss2).sum()
                losses.append(loss)
                # 计算总损失并添加到总损失列表
                _model.run_backward(loss)
                # 执行反向传播
                _optim.step()
                # 执行优化器的步骤
            self.assertEqual(losses1[0], losses1[1])
            # 断言损失列表1中的两个值相等
            self.assertEqual(losses2[0], losses2[1])
            # 断言损失列表2中的两个值相等
            self.assertEqual(losses[0], losses[1])
            # 断言总损失列表中的两个值相等
        self._check_fsdp_parameter_parity(fsdp_model, fsdp_model_orig_params)
        # 检查分布式模型和原始模型参数的参数平等性

    @skip_if_lt_x_gpu(2)
    @parametrize("offload_params", [False, True])
    def test_summon_between_two_forwards(self, offload_params: bool):
        """
        Tests that ``use_orig_params=True`` has parity with ``False`` when
        running a forward pass, :meth:`summon_full_params()`, and another
        forward pass before a backward pass.
        """
        cpu_offload = CPUOffload(offload_params=offload_params)
        # 根据 offload_params 参数创建 CPUOffload 对象
        self.run_subtests(
            self._get_fsdp_parity_subtest_config(),
            self._test_summon_between_two_forwards,
            cpu_offload=cpu_offload,
        )
        # 运行子测试，检查两次前向传播之间的平等性

    def _test_summon_between_two_forwards(
        self,
        sharding_strategy: ShardingStrategy,
        cpu_offload: CPUOffload,
        # 定义私有方法 _test_summon_between_two_forwards，接受分片策略和 CPUOffload 对象
        (
            fsdp_model,  # 从 _get_fsdp_models_and_optims 方法中获取经过 FSDP 处理后的模型
            optim,  # 从 _get_fsdp_models_and_optims 方法中获取优化器
            fsdp_model_orig_params,  # 从 _get_fsdp_models_and_optims 方法中获取未经处理的原始模型
            optim_orig_params,  # 从 _get_fsdp_models_and_optims 方法中获取原始模型的优化器
        ) = self._get_fsdp_models_and_optims(sharding_strategy, cpu_offload)

        device = torch.device("cuda")  # 设置设备为 CUDA 加速器

        for _ in range(3):  # 循环三次
            optim.zero_grad()  # 清除 FSDP 处理后模型的梯度
            optim_orig_params.zero_grad()  # 清除原始模型的梯度

            inp1 = fsdp_model.get_input(device)  # 获取输入数据，并将其放在 CUDA 设备上
            loss1 = fsdp_model(*inp1)  # 计算 FSDP 处理后模型的损失
            loss_orig_params1 = fsdp_model_orig_params(*inp1)  # 计算原始模型的损失
            self.assertEqual(loss1, loss_orig_params1)  # 断言两个损失值应该相等

            # 调用 `summon_full_params()` 方法
            self._check_fsdp_parameter_parity(fsdp_model, fsdp_model_orig_params)

            inp2 = fsdp_model.get_input(device)  # 再次获取输入数据，并将其放在 CUDA 设备上
            loss2 = fsdp_model(*inp2)  # 计算第二次 FSDP 处理后模型的损失
            loss_orig_params2 = fsdp_model_orig_params(*inp2)  # 计算第二次原始模型的损失
            self.assertEqual(loss2, loss_orig_params2)  # 断言第二次损失值应该相等

            loss = (loss1 + loss2).sum()  # 计算总损失
            loss_orig_params = (loss_orig_params1 + loss_orig_params2).sum()  # 计算原始模型的总损失
            fsdp_model.run_backward(loss)  # 在 FSDP 处理后模型上执行反向传播
            fsdp_model_orig_params.run_backward(loss_orig_params)  # 在原始模型上执行反向传播
            optim.step()  # 更新 FSDP 处理后模型的参数
            optim_orig_params.step()  # 更新原始模型的参数

        self._check_fsdp_parameter_parity(fsdp_model, fsdp_model_orig_params)  # 最后再次调用参数一致性检查方法
class TestFSDPUseOrigParamsParamAccess(FSDPTest):
    """Tests original parameter access."""

    @property
    def world_size(self):
        # 强制设定世界大小为2，因为测试中硬编码了FSDP分片策略以检查分片参数的一致性
        return 2

    @skip_if_lt_x_gpu(2)
    def test_access_params_after_forward(self):
        """
        Tests that accessing the original parameters after the forward but
        before the backward. Notably, this is not supported when
        ``use_orig_params=False``. However, for ``True``, FSDP exposes the
        (flattened) sharded original parameters, making it possible.
        """
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.NO_SHARD,
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                ],
            },
            self._test_access_params_after_forward,
        )

    def _test_access_params_after_forward(
        self,
        sharding_strategy: ShardingStrategy,
    ):
        # 实际测试函数，检验在前向传播后但在反向传播前访问原始参数的行为
        # 注意，当 ``use_orig_params=False`` 时，不支持此操作。但是对于 ``True``，
        # FSDP暴露了（扁平化的）分片原始参数，使这种访问成为可能。

class TestFSDPUseOrigParamsWriteback(FSDPTest):
    """Tests parameter and gradient writeback."""

    class Model(nn.Module):
        def __init__(self, device: torch.device):
            super().__init__()
            torch.manual_seed(42)
            self.lin1 = nn.Linear(5, 5, bias=True, device=device)
            self.lin2 = nn.Linear(5, 7, bias=True, device=device)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            z = self.lin1(x)
            z = nn.functional.relu(z)
            z = self.lin2(z)
            return z

        def get_input(self, device: torch.device) -> Tuple[torch.Tensor, ...]:
            return (torch.randn((2, 5)).to(device),)

        def get_loss(self, inp, out):
            return out.sum()

    @property
    def world_size(self):
        # 强制设定世界大小为2，因为测试中硬编码了FSDP分片策略
        return 2

    def _check_param_parity(self, ddp_model: DDP, fsdp_model: FSDP):
        with FSDP.summon_full_params(fsdp_model):
            for (n1, p1), (n2, p2) in zip(
                ddp_model.module.named_parameters(),
                fsdp_model.named_parameters(),
            ):
                self.assertEqual(n1, n2)
                torch.testing.assert_close(p1, p2)

    @skip_if_lt_x_gpu(2)
    def test_param_writeback(self):
        """Tests that changes to the original parameters are written back."""
        self.run_subtests(
            {
                "change_first_weight": [True, False],  # 第一个权重 vs 第二个权重
                "change_data": [True, False],  # 改变 `.data` vs 变量本身
            },
            self._test_param_writeback,
        )

    def _test_param_writeback(
        self,
        change_first_weight: bool,
        change_data: bool,
    ):
        # 实际测试函数，测试原始参数的更改是否被写回
    def _test_param_writeback(self, change_first_weight: bool, change_data: bool):
        def transform_param(param: nn.Parameter) -> nn.Parameter:
            return nn.Parameter(torch.ones_like(param) * 2)

        # 检查写回是否正确传播

        # 使用 DDP 模型来进行分布式数据并行训练，传入 CUDA 设备
        ddp_model = DDP(
            TestFSDPUseOrigParamsWriteback.Model(torch.device("cuda")),
            device_ids=[self.rank],
        )
        
        # 使用 FSDP 模型来进行混合精度和分布式数据并行训练，保留原始参数
        fsdp_model = FSDP(
            TestFSDPUseOrigParamsWriteback.Model(torch.device("cuda")),
            use_orig_params=True,
        )
        
        ddp = ddp_model.module  # 为简洁起见使用模块
        fsdp = fsdp_model.module
        
        if change_first_weight:
            if change_data:
                # 如果更改数据，将 DDP 模型的第一个权重参数设置为转换后的参数
                ddp.lin1.weight.data = transform_param(ddp.lin1.weight)
                # 如果更改数据，将 FSDP 模型的第一个权重参数设置为转换后的参数
                fsdp.lin1.weight.data = transform_param(fsdp.lin1.weight)
            else:
                # 否则将 DDP 模型的第一个权重参数整体设置为转换后的参数
                ddp.lin1.weight = transform_param(ddp.lin1.weight)
                # 否则将 FSDP 模型的第一个权重参数整体设置为转换后的参数
                fsdp.lin1.weight = transform_param(fsdp.lin1.weight)
        else:
            if change_data:
                # 如果更改数据，将 DDP 模型的第二个权重参数数据设置为转换后的参数
                ddp.lin2.weight.data = transform_param(ddp.lin2.weight)
                # 如果更改数据，将 FSDP 模型的第二个权重参数数据设置为转换后的参数
                fsdp.lin2.weight.data = transform_param(fsdp.lin2.weight)
            else:
                # 否则将 DDP 模型的第二个权重参数整体设置为转换后的参数
                ddp.lin2.weight = transform_param(ddp.lin2.weight)
                # 否则将 FSDP 模型的第二个权重参数整体设置为转换后的参数
                fsdp.lin2.weight = transform_param(fsdp.lin2.weight)
        
        # 检查参数的一致性
        self._check_param_parity(ddp_model, fsdp_model)  # 触发写回操作

    @skip_if_lt_x_gpu(2)
    def test_grad_writeback(self):
        """
        Tests that changes to the original parameters' gradients are written
        back.
        """
        # 运行子测试来检查梯度是否正确写回
        self.run_subtests(
            {
                "change_first_weight_grad": [False, True],
                "change_data": [False, True],  # 改变 `.data` 还是变量本身
                "set_to_none": [False, True],
            },
            self._test_grad_writeback,
        )

    def _test_grad_writeback(
        self,
        change_first_weight_grad: bool,
        change_data: bool,
        set_to_none: bool,
    ):
        if change_data and set_to_none:
            return  # 如果设置了change_data并且set_to_none，则返回，表示操作未定义

        def transform_grad(param: nn.Parameter) -> nn.Parameter:
            return None if set_to_none else torch.ones_like(param) * 2
            # 如果set_to_none为真，则返回None；否则返回与param形状相同的全为2的张量

        ddp_model = DDP(
            TestFSDPUseOrigParamsWriteback.Model(torch.device("cuda")),
            device_ids=[self.rank],
        )
        # 使用分布式数据并行（DDP）对模型进行封装，使用CUDA设备进行计算，device_ids指定设备ID列表

        fsdp_model = FSDP(
            TestFSDPUseOrigParamsWriteback.Model(torch.device("cuda")),
            use_orig_params=True,
        )
        # 使用混合精度分布式（FSDP）对模型进行封装，使用CUDA设备进行计算，使用原始参数（use_orig_params=True）

        LR = 1e-2  # 学习率设定为0.01

        # TODO: 如果我们添加了`summon_full_params(with_grads=True)`，则替换以下操作。
        # 目前，我们使用优化器步骤作为检查梯度写回的代理。
        # 使用Adam优化器对DDP和FSDP模型的参数进行优化，学习率为LR

        ddp_optim = torch.optim.Adam(ddp_model.parameters(), lr=LR)
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters(), lr=LR)

        # 生成初始梯度
        inp = fsdp_model.get_input(torch.device("cuda"))
        ddp_out = ddp_model(*inp)
        fsdp_out = fsdp_model(*inp)
        ddp_out.sum().backward()
        fsdp_out.sum().backward()

        # 通过原始参数修改梯度
        ddp = ddp_model.module  # 为简洁起见，使用ddp_model的模块
        fsdp = fsdp_model.module
        if change_first_weight_grad:
            if change_data:
                ddp.lin1.weight.grad.data = transform_grad(ddp.lin1.weight)
                # 如果change_data为真，则将ddp.lin1.weight的梯度数据转换为新的梯度
                if fsdp.lin1.weight.grad is not None:
                    fsdp.lin1.weight.grad.data = transform_grad(fsdp.lin1.weight)
                    # 如果fsdp.lin1.weight的梯度不为None，则将其梯度数据转换为新的梯度
            else:
                ddp.lin1.weight.grad = transform_grad(ddp.lin1.weight)
                # 否则，直接将ddp.lin1.weight的梯度替换为新的梯度
                fsdp.lin1.weight.grad = transform_grad(fsdp.lin1.weight)
                # 直接将fsdp.lin1.weight的梯度替换为新的梯度
        else:
            if change_data:
                ddp.lin2.weight.grad.data = transform_grad(ddp.lin2.weight)
                # 如果change_data为真，则将ddp.lin2.weight的梯度数据转换为新的梯度
                if fsdp.lin2.weight.grad is not None:
                    fsdp.lin2.weight.grad.data = transform_grad(fsdp.lin2.weight)
                    # 如果fsdp.lin2.weight的梯度不为None，则将其梯度数据转换为新的梯度
            else:
                ddp.lin2.weight.grad = transform_grad(ddp.lin2.weight)
                # 否则，直接将ddp.lin2.weight的梯度替换为新的梯度
                fsdp.lin2.weight.grad = transform_grad(fsdp.lin2.weight)
                # 直接将fsdp.lin2.weight的梯度替换为新的梯度

        ddp_optim.step()  # 执行一步优化器更新参数
        fsdp_optim.step()  # 执行一步优化器更新参数
        self._check_param_parity(ddp_model, fsdp_model)  # 触发参数写回的检查操作

        # 故意不清零梯度以检查写回操作
        inp = fsdp_model.get_input(torch.device("cuda"))
        ddp_out = ddp_model(*inp)
        fsdp_out = fsdp_model(*inp)
        ddp_out.sum().backward()
        fsdp_out.sum().backward()
        ddp_optim.step()
        fsdp_optim.step()
        self._check_param_parity(ddp_model, fsdp_model)  # 触发参数写回的检查操作

    @skip_if_lt_x_gpu(2)
    def test_writeback_shape_mismatch(self):
        # 创建 FSDP 模型，使用原始参数写回功能
        fsdp_model = FSDP(
            TestFSDPUseOrigParamsWriteback.Model(torch.device("cuda")),
            use_orig_params=True,
        )
        # 获取模型的 module 属性，简化后续代码
        fsdp = fsdp_model.module  # for brevity
        # 断言当前进程的排名是 0 或 1，期望的世界大小是 2，否则抛出异常
        assert self.rank in (0, 1), f"Expects world size of 2 but got {self.world_size}"
        # 使用断言检查 RuntimeError 异常，并验证其错误消息包含 "Cannot writeback"
        with self.assertRaisesRegex(RuntimeError, "Cannot writeback"):
            # 根据进程的排名，修改梯度，以使写回时发生形状不匹配的情况
            if self.rank == 0:
                # 修改 `lin1.weight.grad`，因为它存在于排名为 0 的进程上
                lin1_weight_shape = list(fsdp.lin1.weight.shape)
                for dim_index in range(len(lin1_weight_shape)):
                    lin1_weight_shape[dim_index] += 1
                # 创建新的参数，强制形状不匹配
                fsdp.lin1.weight = nn.Parameter(
                    torch.randn(
                        torch.Size(lin1_weight_shape), device=fsdp.lin1.weight.device
                    )
                )
                fsdp.lin1.weight.grad = torch.randn(
                    torch.Size(lin1_weight_shape), device=fsdp.lin1.weight.device
                )
            elif self.rank == 1:
                # 修改 `lin2.weight.grad`，因为它部分存在于排名为 1 的进程上
                lin2_weight_shape = list(fsdp.lin2.weight.shape)
                for dim_index in range(len(lin2_weight_shape)):
                    lin2_weight_shape[dim_index] += 1
                # 创建新的参数，强制形状不匹配
                fsdp.lin2.weight = nn.Parameter(
                    torch.randn(
                        torch.Size(lin2_weight_shape), device=fsdp.lin2.weight.device
                    )
                )
                fsdp.lin2.weight.grad = torch.randn(
                    torch.Size(lin2_weight_shape), device=fsdp.lin2.weight.device
                )
            # 触发完整参数的召唤，触发写回操作
            with FSDP.summon_full_params(fsdp_model):
                ...
    def test_writeback_between_fwd_and_bwd_for_no_reshard_raises(self):
        # 定义测试函数，测试在不重新分片情况下写回时触发异常的情况

        fsdp_kwargs = {
            "sharding_strategy": ShardingStrategy.SHARD_GRAD_OP,
            "auto_wrap_policy": ModuleWrapPolicy({nn.Linear}),
            "use_orig_params": True,
        }
        # 使用 functools.partial 创建 FSDP 包装函数，并传入参数 fsdp_kwargs
        fsdp_wrapper = functools.partial(FSDP, **fsdp_kwargs)

        # 创建 FSDP 模型对象，使用自定义的测试模型在 CUDA 设备上
        fsdp_model = fsdp_wrapper(
            TestFSDPUseOrigParamsWriteback.Model(torch.device("cuda"))
        )

        # 获取模型输入数据，使用 CUDA 设备
        inp = fsdp_model.get_input(torch.device("cuda"))

        # 计算模型输出的损失值
        loss = fsdp_model(*inp).sum()

        # 复制权重数据并设置给同名参数
        fsdp_model.lin1.weight.data = fsdp_model.lin1.weight.clone()

        # 断言消息，指示 FSDP 不支持在前向和反向传播之间更改参数
        assert_msg = (
            "FSDP does not support changing the parameters between forward and backward"
        )

        # 使用断言验证损失反向传播时是否抛出预期的 AssertionError 异常
        with self.assertRaisesRegex(AssertionError, assert_msg):
            loss.backward()

        # 重新创建 FSDP 模型对象，使用相同的测试模型和 CUDA 设备
        fsdp_model = fsdp_wrapper(
            TestFSDPUseOrigParamsWriteback.Model(torch.device("cuda"))
        )

        # 获取模型输入数据，使用 CUDA 设备
        inp = fsdp_model.get_input(torch.device("cuda"))

        # 计算模型输出的损失值
        loss = fsdp_model(*inp).sum()

        # 将参数变量本身更改为新的 nn.Parameter 对象
        fsdp_model.lin1._fsdp_wrapped_module.weight = nn.Parameter(
            fsdp_model.lin1.weight.clone()
        )

        # 再次使用断言验证损失反向传播时是否抛出预期的 AssertionError 异常
        with self.assertRaisesRegex(AssertionError, assert_msg):
            loss.backward()

    @skip_if_lt_x_gpu(2)
    def test_no_reshard_and_mixed_precision(self):
        """
        Tests that writeback does not falsely get triggered for a few
        configurations (exercising the sharded view skipping logic):
        - Train forward -> full-precision unshard -> train forward
        - Train forward -> eval forward
        - Train forward/backward -> eval forward -> model checkpoint
        """
        # 此测试函数验证在几种配置下写回不会错误触发：
        # - 训练前向 -> 全精度解分片 -> 训练前向
        # - 训练前向 -> 评估前向
        # - 训练前向/反向 -> 评估前向 -> 模型检查点
        self.run_subtests(
            # 测试不同配置下是否会全精度评估
            {"use_full_prec_in_eval": [False, True]},
            self._test_no_reshard_and_mixed_precision,
        )
    # 定义一个测试函数，用于测试不使用重新分片和混合精度的情况
    def _test_no_reshard_and_mixed_precision(self, use_full_prec_in_eval: bool):
        # 如果在评估中使用全精度，则设置环境变量 `_FSDP_USE_FULL_PREC_IN_EVAL` 为 "1"
        if use_full_prec_in_eval:
            os.environ[_FSDP_USE_FULL_PREC_IN_EVAL] = "1"
        
        # 定义 FSDP 配置参数
        fsdp_kwargs = {
            "sharding_strategy": ShardingStrategy.SHARD_GRAD_OP,
            "auto_wrap_policy": ModuleWrapPolicy({nn.Linear}),
            "mixed_precision": MixedPrecision(param_dtype=torch.float16),
            "use_orig_params": True,
        }

        # 创建 FSDP 模型对象，指定 CUDA 设备，使用给定的 fsdp_kwargs 进行配置
        fsdp_model = FSDP(
            TestFSDPUseOrigParamsWriteback.Model(torch.device("cuda")), **fsdp_kwargs
        )

        # 获取 FSDP 模型的输入，并将其放在 CUDA 设备上
        inp = fsdp_model.get_input(torch.device("cuda"))

        # 在完整参数情况下进行训练前向传播 -> 不分片的全精度 -> 训练前向传播
        fsdp_model(*inp)
        with FSDP.summon_full_params(fsdp_model):
            ...

        # 设置模型为训练模式，并进行训练前向传播
        fsdp_model.train()
        fsdp_model(*inp)

        # 设置模型为评估模式，并进行评估前向传播
        fsdp_model.eval()
        fsdp_model(*inp)

        # 设置模型为训练模式，并进行训练前向传播和反向传播 -> 设置模型为评估模式 -> 加载模型状态并进行前向传播
        fsdp_model.train()
        fsdp_model(*inp).sum().backward()
        fsdp_model.eval()
        fsdp_model(*inp)
        with FSDP.state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT):
            # 获取模型状态字典并加载
            sd = fsdp_model.state_dict()
            fsdp_model.load_state_dict(sd)

        # 再次进行训练前向传播和反向传播
        fsdp_model(*inp).sum().backward()
class TestFSDPUseOrigParamsFQNs(FSDPTest):
    @skip_if_lt_x_gpu(2)
    def test_named_parameters_in_forward(self):
        """
        Tests that calling ``named_parameters()`` during forward returns FQNs
        and ``Tensor`` s corresponding to the original parameters.
        """
        param_shapes = [None, None]  # 初始化参数形状列表，用于保存参数的形状信息
        assert_equal_fn = self.assertEqual  # 设置断言函数为当前测试类中的 assertEqual 方法

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(5, 5)  # 创建一个包含线性层的模型，输入和输出维度均为5

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                nonlocal param_shapes  # 声明 param_shapes 变量为非局部变量，使得闭包可以访问外部的 param_shapes
                # Allow for FSDP prefixes
                param_names = [
                    clean_tensor_name(tup[0]) for tup in self.named_parameters()
                ]  # 获取模型中所有参数的名称，并通过 clean_tensor_name 函数处理，得到干净的参数名列表
                params = [tup[1] for tup in self.named_parameters()]  # 获取模型中所有参数的值，并保存到 params 列表中
                assert (
                    param_shapes[0] is not None and param_shapes[1] is not None
                ), "`param_sizes` should be set"  # 断言 param_shapes 中的参数形状信息不为 None
                assert_equal_fn(
                    param_names,
                    [
                        "lin.weight",
                        "lin.bias",
                    ],
                )  # 使用 assert_equal_fn 断言 param_names 应该包含 "lin.weight" 和 "lin.bias"
                assert_equal_fn(params[0].shape, param_shapes[0])  # 断言 params[0] 的形状与 param_shapes[0] 相同
                assert_equal_fn(params[1].shape, param_shapes[1])  # 断言 params[1] 的形状与 param_shapes[1] 相同
                return self.lin(x)  # 返回线性层的前向传播结果

        model = Model().cuda()  # 创建一个 Model 类的实例，并将其移动到 GPU 上
        # Save the *unsharded* original parameter shapes and check the shapes
        # match in the forward pass
        param_shapes[0] = model.lin.weight.shape  # 保存线性层权重参数的形状到 param_shapes[0]
        param_shapes[1] = model.lin.bias.shape  # 保存线性层偏置参数的形状到 param_shapes[1]
        fsdp_model = FSDP(model, use_orig_params=True)  # 使用 FSDP 对象封装模型，使用原始参数
        inp = torch.randn((2, 5), device=torch.device("cuda"))  # 创建一个随机输入张量，放置在 GPU 上
        fsdp_model(inp)  # 对输入进行模型前向传播计算


class TestFSDPUseOrigParamsNoSync(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2  # 返回并设置测试的世界大小为 2

    @skip_if_lt_x_gpu(2)
    def test_no_sync_correctness(self):
        """
        Tests a basic ``no_sync()`` setup by comparing ``use_orig_params=True``
        against ``use_orig_params=False``.
        """
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.NO_SHARD,
                ],
            },
            self._test_no_sync_correctness,
        )  # 运行子测试，比较使用 use_orig_params=True 和 use_orig_params=False 时的基本 no_sync() 设置

    @skip_if_lt_x_gpu(2)
    def test_no_sync_mixed_precision(self):
        """
        Tests that dtypes are as expected when using ``no_sync()`` with
        ``use_orig_params=True`` and parameter mixed precision.
        """
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.NO_SHARD,
                ]
            },
            self._test_no_sync_mixed_precision,
        )  # 测试在使用 use_orig_params=True 和参数混合精度时，dtype 是否与预期一致
    # 定义一个测试方法 `_test_no_sync_mixed_precision`，接受一个分片策略参数 `sharding_strategy`
    def _test_no_sync_mixed_precision(self, sharding_strategy: ShardingStrategy):
        # 创建一个具有3个输入和3个输出的线性模型，使用 CUDA 设备
        model = nn.Linear(3, 3, device="cuda")
        # 创建混合精度对象 `MixedPrecision`，指定参数精度为 float16，减少精度为 float32
        mixed_precision = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float32,
        )
        # 定义 FSDP 配置参数
        fsdp_kwargs = {
            "sharding_strategy": sharding_strategy,  # 分片策略
            "mixed_precision": mixed_precision,       # 混合精度对象
            "use_orig_params": True,                  # 使用原始参数
        }
        # 使用 FSDP 封装模型，传入模型和 FSDP 配置参数
        fsdp_model = FSDP(model, **fsdp_kwargs)
        # 生成一个在 CUDA 设备上的随机张量作为输入
        inp = torch.randn((2, 3), device="cuda")
        # 使用 `no_sync()` 上下文管理器
        with fsdp_model.no_sync():
            # 在每个 `no_sync()` 后向传播中，检查梯度是否为低精度参数类型（FP16）
            fsdp_model(inp).sum().backward()
            for param in fsdp_model.parameters():
                if param.grad is not None:
                    self.assertEqual(param.grad.dtype, torch.float16)
            # 再次执行后向传播并检查梯度精度
            fsdp_model(inp).sum().backward()
            for param in fsdp_model.parameters():
                if param.grad is not None:
                    self.assertEqual(param.grad.dtype, torch.float16)
        # 在 `no_sync()` 外部执行后向传播，检查梯度是否转换为完整精度以准备优化器步骤
        fsdp_model(inp).sum().backward()
        for param in fsdp_model.parameters():
            if param.grad is not None:
                self.assertEqual(param.grad.dtype, torch.float32)
class TestFSDPUseOrigParamsInit(FSDPTest):
    @skip_if_lt_x_gpu(2)
    def test_non_uniform_requires_grad(self):
        # 创建一个包含两个线性层的序列模型，设备为 CUDA
        model = nn.Sequential(
            nn.Linear(3, 3, device="cuda"),
            nn.Linear(3, 3, device="cuda"),
        )
        # 仅冻结偏置项，并将权重和偏置项展平到同一个 `FlatParameter` 中，以测试非均匀的 `requires_grad`
        model[0].bias.requires_grad = False
        model[1].bias.requires_grad = False
        # 使用原始参数初始化 FSDP 模型
        fsdp_model = FSDP(model, use_orig_params=True)
        # 断言第一个线性层的权重需要梯度
        self.assertTrue(fsdp_model[0].weight.requires_grad)
        # 断言第一个线性层的偏置不需要梯度
        self.assertFalse(fsdp_model[0].bias.requires_grad)
        # 断言第二个线性层的权重需要梯度
        self.assertTrue(fsdp_model[1].weight.requires_grad)
        # 断言第二个线性层的偏置不需要梯度
        self.assertFalse(fsdp_model[1].bias.requires_grad)


# 触发堆栈破坏的大小定义为足够大
NUM_SIZE0_TENSORS = 1000


class TestMultiTensorApply(TestCase):
    def test_multi_tensor_apply_size0_tensors_cpu(self):
        # 创建包含空张量的列表，设备为 CPU
        size0_tensors = [torch.empty(0, device="cpu") for _ in range(NUM_SIZE0_TENSORS)]
        # 检查这不会导致段错误
        torch._foreach_mul_(size0_tensors, 0.1)

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_multi_tensor_apply_size0_tensors_cuda(self):
        # 创建包含空张量的列表，设备为 CUDA
        size0_tensors = [
            torch.empty(0, device="cuda") for _ in range(NUM_SIZE0_TENSORS)
        ]
        # 检查这不会导致段错误
        torch._foreach_mul_(size0_tensors, 0.1)


# 实例化参数化测试类的测试用例
instantiate_parametrized_tests(TestFSDPUseOrigParamsMultipleParamGroups)
instantiate_parametrized_tests(TestFSDPUseOrigParamsUnshardReshard)
instantiate_parametrized_tests(TestFSDPUseOrigParamsParamAccess)
instantiate_parametrized_tests(TestFSDPUseOrigParamsFQNs)
instantiate_parametrized_tests(TestFSDPUseOrigParamsNoSync)

if __name__ == "__main__":
    # 运行测试
    run_tests()
```