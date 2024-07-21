# `.\pytorch\test\distributed\checkpoint\test_state_dict.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入必要的库和模块
import copy                      # 复制对象
import functools                 # 提供高阶函数和操作函数的功能
import sys                       # 提供对Python解释器的访问和控制
from itertools import chain      # 提供操作迭代器和生成器的函数
from typing import Callable, Tuple, Type, Union  # 提供类型提示

import torch                    # PyTorch库
import torch.distributed as dist  # PyTorch分布式通信模块
import torch.nn as nn           # PyTorch神经网络模块
from torch.distributed._composable import fully_shard, replicate  # 导入两个分布式相关的函数

# 将fully_shard命名为FSDP2，以避免与当前测试中使用的原始fully_shard重名
# TODO: 删除旧的composable fully_shard，以免必须将新的fully_shard命名为FSDP2
from torch.distributed._composable.fsdp import (
    fully_shard as FSDP2,
    fully_shard as fsdp_fully_shard,
)
from torch.distributed._shard.sharded_tensor import ShardedTensor  # 分布式张量相关模块
from torch.distributed._tensor import DTensor, init_device_mesh  # 分布式张量和设备初始化相关函数
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,  # 应用激活检查点的函数
)
from torch.distributed.checkpoint import state_dict as ptd_state_dict  # 分布式检查点相关模块
from torch.distributed.checkpoint.state_dict import (
    _patch_model_state_dict,  # 模型状态字典的补丁函数
    _patch_optimizer_state_dict,  # 优化器状态字典的补丁函数
    get_model_state_dict,  # 获取模型状态字典
    get_optimizer_state_dict,  # 获取优化器状态字典
    get_state_dict,  # 获取状态字典
    set_model_state_dict,  # 设置模型状态字典
    set_optimizer_state_dict,  # 设置优化器状态字典
    StateDictOptions,  # 状态字典选项
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,  # 全分片数据并行模块
    ShardingStrategy,  # 分片策略
    StateDictType,  # 状态字典类型
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy  # 模块包装策略
from torch.distributed.optim import _apply_optimizer_in_backward  # 在反向传播中应用优化器相关函数
from torch.nn.parallel import DistributedDataParallel as DDP  # 分布式数据并行模块
from torch.optim import Optimizer  # 优化器基类
from torch.testing._internal.common_dist_composable import (
    CompositeParamModel,  # 复合参数模型
    UnitModule,  # 单元模块
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 如果GPU数量小于x，则跳过
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN  # 运行测试和开发调试相关
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,  # 分布式张量测试基类
    MultiProcessTestCase,  # 多进程测试用例
    with_comms,  # 用于通信的装饰器
)
from torch.testing._internal.distributed.common_state_dict import VerifyStateDictMixin  # 状态字典验证混合类
from torch.utils._pytree import tree_all, tree_all_only  # pytree相关函数

# 如果分布式不可用，则输出消息并退出测试
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 如果测试开启了dev-asan，打印消息并退出，因为torch + multiprocessing spawn存在已知问题
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestStateDict(DTensorTestBase, VerifyStateDictMixin):
    """Tests state_dict and load_state_dict"""

    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    # 测试保存和加载模型状态的方法
    def _test_save_load(
        self,
        init_model_optim: Callable,
        test_frozen: bool = False,
    ):
        ...

    # 测试FullyShardedDataParallel（FSDP）的功能
    def _test_fsdp(
        self,
        *,
        use_orig_params: bool,
        use_composable: bool,
        use_dtensor: bool,
        wrapping: Tuple[nn.Module] = (),
        compile_model: bool = False,
        optimizer_class: Type[Optimizer],
    ) -> None:
        # 如果不使用原始参数并且使用可组合模式，则直接返回，不执行下面的代码
        if not use_orig_params and use_composable:
            return

        # TODO: 在设备网格的可组合API侧更改完成之后，移除此返回语句
        # 如果使用可组合模式并且使用了dtensor，则直接返回，不执行下面的代码
        if use_composable and use_dtensor:
            return

        # 初始化模型和优化器
        def init_model_optim():
            # 如果使用了dtensor，则初始化设备网格为cuda，设备数量为self.world_size
            if use_dtensor:
                device_mesh = init_device_mesh("cuda", (self.world_size,))

            # 创建原始模型和优化器，将模型放置在cuda设备上
            orig_model = CompositeParamModel(device=torch.device("cuda"))
            orig_optim = optimizer_class(orig_model.parameters(), lr=1e-3)
            copy_optim = optimizer_class(orig_model.parameters(), lr=1e-3)

            # 如果有包装策略，则使用给定的策略，否则使用默认的UnitModule
            if wrapping:
                strategy = set(wrapping)
            else:
                strategy = {UnitModule}

            # 如果使用可组合模式，则对原始模型进行全面分片
            if use_composable:
                dist_model = fully_shard(
                    copy.deepcopy(orig_model), policy=ModuleWrapPolicy(strategy)
                )
            else:
                # 如果使用dtensor，则使用FSDP进行模型分布，并使用给定的设备网格
                if use_dtensor:
                    device_mesh = init_device_mesh("cuda", (self.world_size,))
                    dist_model = FSDP(
                        copy.deepcopy(orig_model),
                        auto_wrap_policy=ModuleWrapPolicy(strategy),
                        use_orig_params=use_orig_params,
                        device_mesh=device_mesh,
                    )
                else:
                    # 否则，使用FSDP进行模型分布，使用给定的包装策略和是否使用原始参数
                    dist_model = FSDP(
                        copy.deepcopy(orig_model),
                        auto_wrap_policy=ModuleWrapPolicy(strategy),
                        use_orig_params=use_orig_params,
                    )

            # 如果需要编译模型，则对分布模型进行编译
            if compile_model:
                dist_model = torch.compile(dist_model)
            # 使用分布模型初始化优化器
            dist_optim = optimizer_class(dist_model.parameters(), lr=1e-3)
            return orig_model, orig_optim, copy_optim, dist_model, dist_optim

        # 执行测试方法_test_save_load，传入初始化模型和优化器的函数init_model_optim
        self._test_save_load(init_model_optim)

    # 使用通信装饰器with_comms装饰的测试方法test_fsdp
    @with_comms
    # 如果GPU数少于2个，则跳过测试
    @skip_if_lt_x_gpu(2)
    def test_fsdp(self) -> None:
        # 运行子测试，根据不同的参数组合进行测试
        self.run_subtests(
            {
                "use_orig_params": [True, False],
                "use_composable": [True, False],
                "use_dtensor": [True, False],
                "wrapping": [tuple(), (nn.Linear, UnitModule)],
                "optimizer_class": [torch.optim.Adam, torch.optim.AdamW],
            },
            # 测试方法为_test_fsdp
            self._test_fsdp,
        )

    # 使用通信装饰器with_comms装饰的测试方法test_compiled_fsdp
    @with_comms
    # 如果GPU数少于2个，则跳过测试
    @skip_if_lt_x_gpu(2)
    def test_compiled_fsdp(self) -> None:
        # 运行子测试，根据不同的参数组合进行测试
        self.run_subtests(
            {
                "use_orig_params": [True],
                "use_composable": [False],
                "use_dtensor": [False],
                "wrapping": [tuple()],
                "optimizer_class": [torch.optim.Adam, torch.optim.AdamW],
            },
            # 测试方法为_test_fsdp
            self._test_fsdp,
        )

    # 测试方法_test_fsdp2的定义
    def _test_fsdp2(
        self,
        *,
        reshard_after_forward: Union[bool, int],
        optimizer_class: Type[Optimizer],
        compile_model: bool,
        foreach: bool = True,
   `
    ):
        # 定义初始化模型和优化器的内部函数
        def init_model_optim():
            # 创建一个包含 CUDA 设备的 CompositeParamModel 实例
            orig_model = CompositeParamModel(device=torch.device("cuda"))
            # 初始化原始优化器，使用指定的优化器类，学习率为 1e-3，使用 foreach 参数
            orig_optim = optimizer_class(
                orig_model.parameters(), lr=1e-3, foreach=foreach
            )
            # 初始化复制优化器，使用指定的优化器类，学习率为 1e-3，使用 foreach 参数
            copy_optim = optimizer_class(
                orig_model.parameters(), lr=1e-3, foreach=foreach
            )

            # 使用 deepcopy 克隆原始模型，初始化 FSDP2 分布式模型
            dist_model = FSDP2(
                copy.deepcopy(orig_model),
                reshard_after_forward=reshard_after_forward,
            )

            # 如果编译模型标志为真，则编译分布式模型
            if compile_model:
                dist_model = torch.compile(dist_model)
            # 初始化分布式优化器，使用指定的优化器类，学习率为 1e-3，使用 foreach 参数
            dist_optim = optimizer_class(
                dist_model.parameters(), lr=1e-3, foreach=foreach
            )

            # 返回原始模型、原始优化器、复制优化器、分布式模型和分布式优化器
            return orig_model, orig_optim, copy_optim, dist_model, dist_optim

        # 调用测试保存和加载的方法，传入初始化模型和优化器的函数
        self._test_save_load(init_model_optim)

    # 使用分布式通信装饰器和 GPU 数量检查装饰器
    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_fsdp2(self) -> None:
        # 运行子测试，传入多个配置选项
        self.run_subtests(
            {
                "reshard_after_forward": [True, False],  # 是否在前向计算后重分区
                "optimizer_class": [torch.optim.Adam, torch.optim.AdamW],  # 优化器类列表
                "compile_model": [True, False],  # 是否编译模型
            },
            self._test_fsdp2,  # 测试方法
        )

    # 定义测试 DDP 的内部函数
    def _test_ddp(self, use_composable: bool, optimizer_class: Type[Optimizer]) -> None:
        def init_model_optim():
            # 创建一个包含 CUDA 设备的 CompositeParamModel 实例
            orig_model = CompositeParamModel(device=torch.device("cuda"))
            # 初始化原始优化器，使用指定的优化器类，学习率为 1e-3
            orig_optim = optimizer_class(orig_model.parameters(), lr=1e-3)
            # 初始化复制优化器，使用指定的优化器类，学习率为 1e-3
            copy_optim = optimizer_class(orig_model.parameters(), lr=1e-3)
            # 如果使用可组合模型，则使用 replicate 函数复制模型
            if use_composable:
                dist_model = replicate(copy.deepcopy(orig_model))
            else:
                # 否则，使用 DDP 初始化分布式数据并行模型
                dist_model = DDP(copy.deepcopy(orig_model))
            # 初始化分布式优化器，使用指定的优化器类，学习率为 1e-3
            dist_optim = optimizer_class(dist_model.parameters(), lr=1e-3)
            # 返回原始模型、原始优化器、复制优化器、分布式模型和分布式优化器
            return orig_model, orig_optim, copy_optim, dist_model, dist_optim

        # 调用测试保存和加载的方法，传入初始化模型和优化器的函数
        self._test_save_load(init_model_optim)

    # 使用分布式通信装饰器和 GPU 数量检查装饰器
    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_ddp(self) -> None:
        # 运行子测试，传入多个配置选项
        self.run_subtests(
            {
                "use_composable": [True, False],  # 是否使用可组合模型
                "optimizer_class": [torch.optim.Adam, torch.optim.AdamW],  # 优化器类列表
            },
            self._test_ddp,  # 测试方法
        )

    # 定义测试 FSDP 和 DDP 的内部函数，包含多个参数
    def _test_fsdp_ddp(
        self,
        use_composable: bool,
        optimizer_class: Type[Optimizer],
        optim_in_backward: bool = False,
        test_frozen: bool = False,
    ) -> None:
        # 定义一个初始化模型和优化器的函数
        def init_model_optim():
            # 创建一个基础的复合参数模型对象，并指定使用 CUDA 设备
            orig_model = CompositeParamModel(device=torch.device("cuda"))
            # 如果测试冻结状态，冻结部分模型参数
            if test_frozen:
                for param in chain(
                    orig_model.u1.parameters(), orig_model.u2.parameters()
                ):
                    param.requires_grad = False
            # 使用给定的优化器类初始化原始模型的优化器
            orig_optim = optimizer_class(orig_model.parameters(), lr=1e-3)
            # 创建一个原始模型参数的复制对象，并使用相同的优化器类初始化其优化器
            copy_optim = optimizer_class(orig_model.parameters(), lr=1e-3)
            # 使用深拷贝创建一个分布式模型对象
            dist_model = copy.deepcopy(orig_model)
            # 如果启用可组合性，则复制模型的层和策略
            if use_composable:
                replicate(dist_model.l)
                fully_shard(dist_model, policy=ModuleWrapPolicy({UnitModule}))
            else:
                # 否则，将模型的某一层封装为分布式数据并更新模型对象
                dist_model.l = DDP(dist_model.l)
                dist_model = FSDP(
                    copy.deepcopy(orig_model),
                    auto_wrap_policy=ModuleWrapPolicy({UnitModule}),
                    use_orig_params=optim_in_backward,
                    ignored_modules=[dist_model.l],
                )
            # 如果优化器应用在反向传播过程中
            if optim_in_backward:
                # 对分布式模型的参数应用优化器设置
                _apply_optimizer_in_backward(
                    optimizer_class, dist_model.parameters(), {"lr": 1e-3}
                )
                # 获取所有参数的反向传播优化器
                dist_optim = [
                    p._in_backward_optimizers[0] for p in dist_model.parameters()
                ]
            else:
                # 否则，初始化分布式模型的优化器
                dist_optim = optimizer_class(dist_model.parameters(), lr=1e-3)
            # 返回初始化后的对象：原始模型、原始模型优化器、复制模型优化器、分布式模型、分布式模型优化器
            return orig_model, orig_optim, copy_optim, dist_model, dist_optim

        # 调用测试函数 _test_save_load，并传入初始化模型和优化器的函数作为参数
        self._test_save_load(init_model_optim, test_frozen)

    # 通过 with_comms 和 skip_if_lt_x_gpu 装饰器，定义一个测试函数 test_fsdp_ddp
    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_fsdp_ddp(self) -> None:
        # 运行子测试，传入参数字典和测试函数 _test_fsdp_ddp
        self.run_subtests(
            {
                "use_composable": [True, False],
                "optimizer_class": [torch.optim.Adam, torch.optim.AdamW],
            },
            self._test_fsdp_ddp,
        )

    # 通过 with_comms 和 skip_if_lt_x_gpu 装饰器，定义一个测试函数 test_frozen_parameters
    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_frozen_parameters(self) -> None:
        # 运行子测试，传入参数字典和测试函数 _test_fsdp_ddp
        self.run_subtests(
            {
                "use_composable": [True],
                "optimizer_class": [torch.optim.Adam, torch.optim.AdamW],
                "test_frozen": [True],
            },
            self._test_fsdp_ddp,
        )

    # TODO: enable use_dtensor once 2D device_mesh support is fully landed.
    """
    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_use_dtensor(self) -> None:
        self._test_fsdp_ddp(use_composable=False, use_dtensor=True)
    """

    # TODO: enable the test after FSDP + apply_optimizer_in_backward works.
    # Disable this test as it is broken after
    # https://github.com/pytorch/pytorch/pull/108298.
    """
    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_apply_optimizer_in_backward(self) -> None:
        # 运行子测试，传入参数字典和测试函数 _test_fsdp_ddp，设置优化器应用在反向传播过程中
        self.run_subtests(
            {"use_composable": [True, False]},
            self._test_fsdp_ddp,
            optim_in_backward=True,
        )
    """
    # 定义一个测试方法，用于测试单个 GPU 的情况下的优化器行为
    def _test_single_gpu(self, optimizer_class: Type[Optimizer]) -> None:
        # 定义内部方法，用于初始化模型和优化器
        def init_model_optim():
            # 创建一个在 CUDA 设备上的 CompositeParamModel 模型实例
            orig_model = CompositeParamModel(device=torch.device("cuda"))
            # 使用给定的优化器类和学习率初始化原始模型的优化器
            orig_optim = optimizer_class(orig_model.parameters(), lr=1e-3)
            # 使用相同的优化器类和学习率初始化另一个原始模型的优化器
            copy_optim = optimizer_class(orig_model.parameters(), lr=1e-3)
            # 深拷贝原始模型，创建模型的副本
            model_copy = copy.deepcopy(orig_model)
            # 使用相同的优化器类和学习率初始化模型副本的优化器
            optim_copy = optimizer_class(model_copy.parameters(), lr=1e-3)
            # 返回初始化后的原始模型、两个不同优化器的原始副本、以及副本模型和优化器
            return orig_model, orig_optim, copy_optim, model_copy, optim_copy

        # 调用测试保存和加载方法，传入模型和优化器初始化函数
        self._test_save_load(init_model_optim)

    # 使用带有通信装饰器的方法，仅在至少有一块 GPU 的情况下运行
    @with_comms
    @skip_if_lt_x_gpu(1)
    def test_single_gpu(self) -> None:
        # 运行子测试，测试不同优化器类的单 GPU 行为
        self.run_subtests(
            {"optimizer_class": [torch.optim.Adam, torch.optim.AdamW]},
            self._test_single_gpu,
        )

    # 使用带有通信装饰器的方法，仅在至少有一块 GPU 的情况下运行
    @with_comms
    @skip_if_lt_x_gpu(1)
    def test_strict(self) -> None:
        # 创建一个在 CUDA 设备上的 CompositeParamModel 模型实例
        model = CompositeParamModel(device=torch.device("cuda"))

        # 获取模型的状态字典
        model_state_dict = get_model_state_dict(model)
        # 获取状态字典的第一个键
        key = next(iter(model_state_dict.keys()))
        # 向模型状态字典添加一个名为 "abc" 的张量
        model_state_dict["abc"] = torch.zeros(10)
        # 使用断言检测运行时错误，确保模型状态字典中不应该有 "abc" 这个键
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            set_model_state_dict(model, model_state_dict=model_state_dict)
        # 从状态字典中移除之前添加的 "abc" 键
        model_state_dict.pop(key)
        # 使用非严格模式设置模型的状态字典，并获取不兼容的键集合
        incompatible_keys = set_model_state_dict(
            model,
            model_state_dict=model_state_dict,
            options=StateDictOptions(strict=False),
        )
        # 使用断言检测确保缺失的键是之前移除的键
        self.assertEqual(incompatible_keys.missing_keys, [key])
        # 使用断言检测确保意外的键包含 "abc"
        self.assertEqual(incompatible_keys.unexpected_keys, ["abc"])
        # 再次从状态字典中移除 "abc" 键
        model_state_dict.pop("abc")
        # 使用断言检测运行时错误，确保模型状态字典中应该有一个缺失的键
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            set_model_state_dict(model, model_state_dict=model_state_dict)

    # 定义一个测试方法，用于测试 CPU 降级和完整状态字典
    def _test_cpu_offload_full_state_dict(
        self, optimizer_class: Type[Optimizer]
    ) -> None:
        # 创建一个包含多个参数模型的组合模型，并使用 CUDA 设备
        orig_model = CompositeParamModel(device=torch.device("cuda"))
        # 使用 CUDA 设备初始化设备网格
        device_mesh = init_device_mesh("cuda", (self.world_size,))
        # 使用 FSDP 封装深拷贝后的原始模型，设置自动封装策略和设备网格
        dist_model = FSDP(
            copy.deepcopy(orig_model),
            auto_wrap_policy=ModuleWrapPolicy({UnitModule}),
            use_orig_params=True,
            device_mesh=device_mesh,
        )

        # 使用给定的优化器类别初始化分布式优化器
        dist_optim = optimizer_class(dist_model.parameters(), lr=1e-3)

        # 获取分布式模型和优化器的状态字典
        mst, ost = get_state_dict(
            dist_model,
            dist_optim,
            options=StateDictOptions(cpu_offload=True),
        )

        # 设置 CPU 设备
        cpu_device = torch.device("cpu")

        # 定义函数检查张量是否在 CPU 上
        def is_cpu(v):
            if isinstance(v, DTensor):
                return v.device == cpu_device
            elif isinstance(v, ShardedTensor):
                shards = v.local_shards()
                if not shards:
                    return True
                return shards[0].tensor.device == cpu_device
            else:
                return v.device == cpu_device

        # 断言所有主状态字典中的张量为 CPU 张量
        self.assertTrue(
            tree_all_only((torch.Tensor, DTensor, ShardedTensor), is_cpu, mst)
        )
        # 断言所有优化状态字典中的张量为 CPU 张量
        self.assertTrue(
            tree_all_only((torch.Tensor, DTensor, ShardedTensor), is_cpu, ost)
        )

        # 获取完整状态字典（不包括特定类型的张量）
        mst, ost = get_state_dict(
            dist_model, dist_optim, options=StateDictOptions(full_state_dict=True)
        )

        # 断言主状态字典中不存在 DTensor 或 ShardedTensor 类型的张量
        self.assertTrue(
            tree_all(lambda v: not isinstance(v, (DTensor, ShardedTensor)), mst)
        )
        # 断言优化状态字典中不存在 DTensor 或 ShardedTensor 类型的张量
        self.assertTrue(
            tree_all(lambda v: not isinstance(v, (DTensor, ShardedTensor)), ost)
        )

        # 获取包括 CPU 转移的完整状态字典
        mst, ost = get_state_dict(
            dist_model,
            dist_optim,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

        # 如果是主进程，断言所有主状态字典中的张量为 CPU 张量
        if self.rank == 0:
            self.assertTrue(
                tree_all_only((torch.Tensor, DTensor, ShardedTensor), is_cpu, mst)
            )
            # 如果是主进程，断言所有优化状态字典中的张量为 CPU 张量
            self.assertTrue(
                tree_all_only((torch.Tensor, DTensor, ShardedTensor), is_cpu, ost)
            )
        else:
            # 如果不是主进程，期望状态字典为空
            self.assertEqual(mst, {})
            self.assertEqual(ost, {})

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_cpu_offload_full_state_dict(self) -> None:
        # 运行子测试，测试 CPU 转移的完整状态字典
        self.run_subtests(
            {"optimizer_class": [torch.optim.Adam, torch.optim.AdamW]},
            self._test_cpu_offload_full_state_dict,
        )

    @with_comms
    @skip_if_lt_x_gpu(1)
    def test_activation_ckpt_fqns_ddp(self) -> None:
        """Tests that activation checkpointing prefixes are removed from module names"""
        # 创建包含多个参数模型的组合模型，并使用 CUDA 设备
        model = CompositeParamModel(device=torch.device("cuda"))
        # 获取模型的原始状态字典键
        original_keys = get_model_state_dict(model).keys()

        # 应用激活检查点，并将模型放置在 DDP 中
        apply_activation_checkpointing(model)
        model = DDP(model)
        # 获取修改后的模型状态字典键
        new_keys = get_model_state_dict(model).keys()

        # 断言原始状态字典键与新状态字典键相同
        self.assertEqual(original_keys, new_keys)

    @with_comms
    @skip_if_lt_x_gpu(1)
    # 定义一个测试方法，用于测试激活检查点前缀是否从模块名称中移除
    def test_activation_ckpt_fqns_fsdp1(self) -> None:
        # 运行子测试，测试不同的参数组合
        self.run_subtests(
            {"use_orig_params": [True, False]},  # 参数字典包含两种布尔值组合
            self._test_activation_ckpt_fqns_fsdp1,  # 调用具体的测试方法进行测试
        )

    # 实际的测试方法，测试激活检查点前缀是否从模块名称中移除
    def _test_activation_ckpt_fqns_fsdp1(self, use_orig_params: bool) -> None:
        """Tests that activation checkpointing prefixes are removed from module names"""
        # 创建一个使用 CUDA 设备的复合参数模型
        model = CompositeParamModel(device=torch.device("cuda"))
        # 获取模型的原始状态字典的键集合
        original_keys = get_model_state_dict(model).keys()

        # 应用激活检查点功能到模型
        apply_activation_checkpointing(model)
        # 使用 FSDP 对模型进行包装，指定是否使用原始参数
        model = FSDP(model, use_orig_params=use_orig_params)
        # 获取经过修改后模型的状态字典的键集合
        new_keys = get_model_state_dict(model).keys()

        # 断言原始状态字典的键集合与修改后模型的状态字典的键集合相等
        self.assertEqual(original_keys, new_keys)

    # 标记为需要通信的测试方法，并要求至少有一块 GPU
    @with_comms
    @skip_if_lt_x_gpu(1)
    def test_extra_state(self) -> None:
        # 创建一个使用 CUDA 设备的复合参数模型
        model = CompositeParamModel(device=torch.device("cuda"))

        # 定义获取额外状态的方法
        def get_extra_state(self):
            return "MyState"

        # 定义设置额外状态的方法
        def set_extra_state(self, state):
            return

        # 将获取和设置额外状态的方法分配给模型的单元模块
        UnitModule.get_extra_state = get_extra_state
        UnitModule.set_extra_state = set_extra_state

        # 使用深度复制创建一个 DDP 模型
        ddp_model = DDP(copy.deepcopy(model))
        # 设置 DDP 模型的状态字典为当前模型的状态字典
        set_model_state_dict(ddp_model, get_model_state_dict(ddp_model))
        # 断言模型的状态字典中的指定键的值为"MyState"
        self.assertEqual(model.state_dict()["u1._extra_state"], "MyState")
        # 断言模型的状态字典与 DDP 模型的状态字典相等
        self.assertEqual(model.state_dict(), get_model_state_dict(ddp_model))

    # 标记为需要通信的测试方法，并要求至少有一块 GPU
    @with_comms
    @skip_if_lt_x_gpu(1)
    def test_non_persistent_buffers(self) -> None:
        # 创建一个使用 CUDA 设备的复合参数模型
        model = CompositeParamModel(device=torch.device("cuda"))
        # 注册一个非持久化缓冲区到模型中
        model.register_buffer(
            "dont_save_me", torch.rand(100, device="cuda"), persistent=False
        )
        # 使用深度复制创建一个 DDP 模型
        ddp_model = DDP(copy.deepcopy(model))
        # 设置 DDP 模型的状态字典为当前模型的状态字典
        set_model_state_dict(ddp_model, get_model_state_dict(ddp_model))
        # 断言模型的状态字典与 DDP 模型的状态字典相等
        self.assertEqual(model.state_dict(), get_model_state_dict(ddp_model))
    # 定义一个测试函数，用于测试从排名0广播的情况
    def _test_broadcast_from_rank0(self, wrapper) -> None:
        # 创建一个在 CUDA 设备上的复合参数模型
        model = CompositeParamModel(device=torch.device("cuda"))
        # 使用 Adam 优化器对模型参数进行优化
        optim = torch.optim.Adam(model.parameters())
        # 使用包装器对模型进行深拷贝，并在此基础上创建 FSDP 模型
        fsdp_model = wrapper(copy.deepcopy(model))
        # 使用 Adam 优化器对 FSDP 模型的参数进行优化
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters())

        # 创建一个在 CUDA 设备上的随机张量作为批量输入
        batch = torch.rand(8, 100, device="cuda")
        # 对原始模型进行前向传播、反向传播和优化
        model(batch).sum().backward()
        optim.step()
        # 获取原始模型和优化器的状态字典
        states, optim_states = get_state_dict(model, optim)

        # 对 FSDP 模型进行前向传播、反向传播和优化
        fsdp_model(batch).sum().backward()
        fsdp_optim.step()

        # 定义一个检查函数，用于比较两个模型和优化器的状态字典是否相等
        def check(equal):
            # 获取 FSDP 模型的状态字典
            fsdp_states = get_model_state_dict(
                fsdp_model,
                options=StateDictOptions(full_state_dict=True),
            )
            # 获取 FSDP 模型优化器的状态字典
            fsdp_optim_states = get_optimizer_state_dict(
                fsdp_model,
                fsdp_optim,
                options=StateDictOptions(full_state_dict=True),
            )
            # 如果 equal 为 True，则断言原始模型和 FSDP 模型的状态字典相等
            if equal:
                self.assertEqual(states, fsdp_states)
                self.assertEqual(optim_states, fsdp_optim_states)
            # 如果 equal 为 False，则断言原始模型和 FSDP 模型的状态字典不相等
            else:
                self.assertNotEqual(states, fsdp_states)
                self.assertNotEqual(optim_states, fsdp_optim_states)

        # 首先进行 equal=True 的检查
        check(equal=True)
        # 再次对 FSDP 模型进行前向传播、反向传播和优化
        fsdp_model(batch).sum().backward()
        fsdp_optim.step()
        # 进行 equal=False 的检查
        check(equal=False)

        # 如果当前进程的排名大于0，则创建空的加载状态字典和优化器状态字典
        if dist.get_rank() > 0:
            load_states = {}
            load_states2 = {}
            load_optim_states = {}
        # 否则，深拷贝原始模型和优化器的状态字典
        else:
            load_states = copy.deepcopy(states)
            load_states2 = copy.deepcopy(states)
            load_optim_states = copy.deepcopy(optim_states)

        # 将加载状态字典应用于 FSDP 模型
        set_model_state_dict(
            fsdp_model,
            model_state_dict=load_states,
            options=StateDictOptions(broadcast_from_rank0=True, full_state_dict=True),
        )
        # 将加载优化器状态字典应用于 FSDP 模型的优化器
        set_optimizer_state_dict(
            fsdp_model,
            fsdp_optim,
            optim_state_dict=load_optim_states,
            options=StateDictOptions(broadcast_from_rank0=True, full_state_dict=True),
        )

        # 进行 equal=True 的检查
        check(equal=True)
        # 验证 strict 标志
        load_states = load_states2
        # 如果加载状态字典非空，则移除其中的一个键
        if load_states:
            key = next(iter(load_states.keys()))
            load_states.pop(key)
        # 断言设置模型状态字典时抛出 RuntimeError 异常，异常消息包含 "Missing key"
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            set_model_state_dict(
                fsdp_model,
                model_state_dict=load_states,
                options=StateDictOptions(
                    broadcast_from_rank0=True, full_state_dict=True
                ),
            )

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_broadcast_from_rank0(self) -> None:
        # 使用 CUDA 初始化设备网格，指定网格大小为当前世界大小
        device_mesh = init_device_mesh("cuda", (self.world_size,))
        # 运行子测试，传入一组包含两个部分的字典作为参数
        self.run_subtests(
            {
                "wrapper": [
                    # 使用 FSDP2 的部分应用，传入设备网格作为参数
                    functools.partial(FSDP2, mesh=device_mesh),
                    # 使用 FSDP 的部分应用，传入设备网格作为参数
                    functools.partial(FSDP, device_mesh=device_mesh),
                ]
            },
            # 指定测试方法为 _test_broadcast_from_rank0
            self._test_broadcast_from_rank0,
        )

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_broadcast_from_rank0_hsdp(self) -> None:
        # 使用 CUDA 初始化设备网格，指定网格大小为 (2, self.world_size // 2)
        device_mesh = init_device_mesh("cuda", (2, self.world_size // 2))
        # 运行子测试，传入包含单个部分的字典作为参数
        self.run_subtests(
            {
                "wrapper": [
                    # 使用 FSDP 的部分应用，传入设备网格和分片策略作为参数
                    functools.partial(
                        FSDP,
                        device_mesh=device_mesh,
                        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                    ),
                ]
            },
            # 指定测试方法为 _test_broadcast_from_rank0
            self._test_broadcast_from_rank0,
        )

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_fsdp_root_not_initialized(self) -> None:
        # 此测试验证 FSDP 根部未初始化，但我们仍应能够无错误地获取状态字典，
        # 因为 fsdp_model.state_dict() 会触发 FSDP 初始化。
        # 使用 CUDA 初始化设备网格，指定网格大小为当前世界大小
        device_mesh = init_device_mesh("cuda", (self.world_size,))
        # 创建一个包含复合参数模型的模型对象，并深度复制到 fsdp_model
        model = CompositeParamModel(device=torch.device("cuda"))
        fsdp_model = FSDP(copy.deepcopy(model), device_mesh=device_mesh)
        # 使用 Adam 优化器初始化 fsdp_model 的参数
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters())
        # 获取 fsdp_model 的模型状态字典
        get_model_state_dict(fsdp_model)
        # 获取 fsdp_model 和 fsdp_optim 的优化器状态字典
        get_optimizer_state_dict(fsdp_model, fsdp_optim)

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_optim_state_dict_param_matching(self) -> None:
        # This test verifies parameters between optim and optim_state_dict
        # "initial_lr" is added to optim_state_dict, but not to the new optim
        # We test whether "initial_lr" appears in optim after
        # set_optimizer_state_dict.

        # 设定运行设备为 CUDA
        device = "cuda"
        # 设置随机种子为 0
        torch.manual_seed(0)
        # 创建包含两个线性层的序列模型，每个层的设备为 CUDA，无偏置
        model = nn.Sequential(
            *[nn.Linear(4, 4, device=device, bias=False) for _ in range(2)]
        )
        # 对每一层进行分片处理
        for layer in model:
            fully_shard(layer)
        # 对整个模型进行分片处理
        fully_shard(model)
        # 使用 Adam 优化器，学习率为 0.01
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        # 使用 LambdaLR 学习率调度器，每个 epoch 学习率按指数衰减
        torch.optim.lr_scheduler.LambdaLR(
            optim, lr_lambda=[lambda epoch: 0.95**epoch]
        )
        # 获取模型及优化器的状态字典
        opt_state_dict = ptd_state_dict.get_optimizer_state_dict(
            model,
            optim,
            options=ptd_state_dict.StateDictOptions(
                full_state_dict=True, cpu_offload=True
            ),
        )
        # 如果进程的排名为 0，验证优化器状态字典中是否包含 "initial_lr"
        if dist.get_rank() == 0:
            self.assertTrue("initial_lr" in opt_state_dict["param_groups"][0])

        # 重新创建 Adam 优化器，学习率为 0.01
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        # 验证新创建的优化器中不包含 "initial_lr"
        self.assertTrue("initial_lr" not in optim.param_groups[0])

        # 设置模型及优化器状态字典
        ptd_state_dict.set_optimizer_state_dict(
            model,
            optim,
            optim_state_dict=opt_state_dict,
            options=ptd_state_dict.StateDictOptions(
                broadcast_from_rank0=True, full_state_dict=True
            ),
        )
        # 如果进程的排名为 0，验证优化器中是否包含 "initial_lr"
        if dist.get_rank() == 0:
            self.assertTrue("initial_lr" in optim.param_groups[0])

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_optim_state_dict_tensor_matching(self) -> None:
        # 设定运行设备为 CUDA
        device = "cuda"
        # 设置随机种子为 0
        torch.manual_seed(0)
        # 创建包含两个线性层的序列模型，每个层的设备为 CUDA，无偏置
        model = nn.Sequential(
            *[nn.Linear(4, 4, device=device, bias=False) for _ in range(2)]
        )
        # 对每一层进行分片处理
        for layer in model:
            fsdp_fully_shard(layer)
        # 对整个模型进行分片处理
        fsdp_fully_shard(model)
        # 使用 Adam 优化器，学习率为 0.01
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        # 创建输入数据张量
        x = torch.randn((4, 4), device=device)
        # 前向传播、反向传播、优化器更新梯度
        model(x).sum().backward()
        optim.step()
        optim.zero_grad()
        # 验证优化器状态中的 exp_avg 是否为 DTensor 类型
        self.assertIsInstance(
            list(optim.state.values())[0]["exp_avg"], DTensor  # noqa: RUF015
        )
        # 获取模型及优化器状态字典
        opt_state_dict = ptd_state_dict.get_optimizer_state_dict(
            model,
            optim,
            options=ptd_state_dict.StateDictOptions(full_state_dict=True),
        )
        # 重新创建 Adam 优化器，学习率为 0.01
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        # 设置模型及优化器状态字典
        ptd_state_dict.set_optimizer_state_dict(
            model,
            optim,
            optim_state_dict=opt_state_dict,
            options=ptd_state_dict.StateDictOptions(full_state_dict=True),
        )
        # 验证优化器状态中的 exp_avg 是否为 DTensor 类型
        self.assertIsInstance(
            list(optim.state.values())[0]["exp_avg"], DTensor  # noqa: RUF015
        )

    @with_comms
    @skip_if_lt_x_gpu(2)
    # 定义一个测试方法，用于测试扁平化 OSD（Optimizer State Dictionary）
    def test_flattened_osd(self) -> None:
        # 初始化设备网格，使用 CUDA 设备和给定的世界大小
        device_mesh = init_device_mesh("cuda", (self.world_size,))
        # 创建一个复合参数模型，使用 CUDA 设备
        model = CompositeParamModel(device=torch.device("cuda"))
        # 使用深拷贝创建 FSDP2 模型，并指定设备网格
        fsdp_model = FSDP2(copy.deepcopy(model), mesh=device_mesh)
        # 创建一个 AdamW 优化器，优化 FSDP2 模型的参数
        fsdp_optim = torch.optim.AdamW(fsdp_model.parameters())
        # 创建一个随机张量作为批处理数据，放在 CUDA 设备上
        batch = torch.rand(8, 100, device="cuda")
        # 对模型输出的总和进行反向传播
        fsdp_model(batch).sum().backward()
        # 在优化器上执行一步优化
        fsdp_optim.step()
        # 清空优化器的梯度
        fsdp_optim.zero_grad()
        # 获取优化器状态字典的第一个 OSD（Optimizer State Dictionary）
        osd1 = get_optimizer_state_dict(fsdp_model, fsdp_optim)
        # 获取优化器状态字典的第二个 OSD，通过设置选项来扁平化 OSD
        osd2 = get_optimizer_state_dict(
            fsdp_model,
            fsdp_optim,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        # 创建另一个 AdamW 优化器，用于比较状态字典是否相同
        fsdp_optim2 = torch.optim.AdamW(fsdp_model.parameters())
        # 将第二个 OSD 应用于另一个优化器
        set_optimizer_state_dict(
            fsdp_model, optimizers=fsdp_optim2, optim_state_dict=osd2
        )
        # 断言两个优化器的状态字典是否相等
        self.assertEqual(fsdp_optim.state_dict(), fsdp_optim2.state_dict())
        # 将第一个 OSD 应用于另一个优化器
        set_optimizer_state_dict(
            fsdp_model, optimizers=fsdp_optim2, optim_state_dict=osd1
        )
        # 再次断言两个优化器的状态字典是否相等
        self.assertEqual(fsdp_optim.state_dict(), fsdp_optim2.state_dict())

    @with_comms
    @skip_if_lt_x_gpu(1)
    # 定义一个测试方法，用于测试部分弃用功能
    def test_deprecate_partial(self) -> None:
        # 创建一个 CompositeParamModel 模型实例，使用 CUDA 设备
        model = CompositeParamModel(device=torch.device("cuda"))

        # 获取模型的状态字典（浅拷贝）
        model_state_dict1 = get_model_state_dict(model)
        model_state_dict1 = copy.deepcopy(model_state_dict1)
        
        # 使用断言检查警告信息，验证特定情况下获取模型状态字典的部分功能已弃用
        with self.assertWarnsRegex(
            FutureWarning,
            "Getting submodules only model/optim state_dict is deprecated",
        ):
            # 获取模型状态字典，只包括模块 model.l 的子模块
            model_state_dict2 = get_model_state_dict(model, submodules={model.l})
        model_state_dict2 = copy.deepcopy(model_state_dict2)
        
        # 再次使用断言检查警告信息，验证另一种情况下获取模型状态字典的部分功能已弃用
        with self.assertWarnsRegex(
            FutureWarning,
            "Getting submodules only model/optim state_dict is deprecated",
        ):
            # 获取模型状态字典，只包括模块 model.l 的子模块，并设置选项关闭子模块前缀
            model_state_dict3 = get_model_state_dict(
                model,
                submodules={model.l},
                options=StateDictOptions(keep_submodule_prefixes=False),
            )
        model_state_dict3 = copy.deepcopy(model_state_dict3)
        
        # 使用断言验证模型状态字典的长度是否符合预期
        self.assertEqual(len(model_state_dict2), 2)
        self.assertEqual(len(model_state_dict3), 2)
        
        # 遍历模型状态字典的键
        for key in model_state_dict3.keys():
            # 构建完整的键名，包括模块前缀
            full_fqn = f"l.{key}"
            # 获取不同模型状态字典中相同键名对应的值
            value1 = model_state_dict1[full_fqn]
            value2 = model_state_dict2[full_fqn]
            value3 = model_state_dict3[key]
            # 使用断言验证这些值是否相等
            self.assertEqual(value1, value2)
            self.assertEqual(value2, value3)

        # 创建一个全零状态字典，与模型状态字典的键值对形式相同但值为全零张量
        zeros_state_dict = {
            k: torch.zeros_like(v) for k, v in model_state_dict1.items()
        }
        
        # 将模型加载为全零状态
        model.load_state_dict(zeros_state_dict)
        
        # 设置模型状态字典，使用非严格模式
        set_model_state_dict(
            model,
            model_state_dict=model_state_dict2,
            options=StateDictOptions(strict=False),
        )
        
        # 使用断言验证模型某一层的权重和偏置是否与预期值相等
        self.assertEqual(model.l.weight, model_state_dict1["l.weight"])
        self.assertEqual(model.l.bias, model_state_dict1["l.bias"])

        # 再次将模型加载为全零状态
        model.load_state_dict(zeros_state_dict)
        
        # 使用断言检查警告信息，验证模型状态字典的传递已被弃用
        with self.assertWarnsRegex(FutureWarning, "Passing model_state_dict as a "):
            # 设置模型状态字典，传递的是模块 model.l 与其对应状态字典的映射
            set_model_state_dict(
                model,
                model_state_dict={model.l: model_state_dict3},
                options=StateDictOptions(strict=False),
            )
        
        # 使用断言验证模型某一层的权重和偏置是否与预期值相等
        self.assertEqual(model.l.weight, model_state_dict1["l.weight"])
        self.assertEqual(model.l.bias, model_state_dict1["l.bias"])

    # 装饰器，用于处理通信相关的设置
    @with_comms
    # 如果系统中 GPU 数量小于 1，则跳过当前测试
    @skip_if_lt_x_gpu(1)
    # 定义测试方法，用于测试已弃用的 FSDP API
    def test_deprecate_fsdp_api(self) -> None:
        # 初始化设备网格，指定为 CUDA，且大小为当前世界大小
        device_mesh = init_device_mesh("cuda", (self.world_size,))
        # 创建复合参数模型，并指定设备为 CUDA
        model = CompositeParamModel(device=torch.device("cuda"))
        # 深拷贝原始模型，然后使用设备网格初始化 FSDP 模型
        fsdp_model = FSDP(copy.deepcopy(model), device_mesh=device_mesh)
        
        # 断言将会发出未来警告，并检查警告消息中是否包含特定文本
        with self.assertWarnsRegex(
            FutureWarning,
            r"FSDP.state_dict_type\(\) and FSDP.set_state_dict_type\(\) are being deprecated",
        ):
            # 使用 FULL_STATE_DICT 状态类型来获取 FSDP 模型的状态字典
            with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT):
                fsdp_model.state_dict()

        # 断言将会抛出断言错误，并检查错误消息中是否包含特定文本
        with self.assertRaisesRegex(AssertionError, "FutureWarning not triggered"):
            # 断言将会发出未来警告，并检查警告消息中是否包含特定文本
            with self.assertWarnsRegex(
                FutureWarning,
                r"FSDP.state_dict_type\(\) and FSDP.set_state_dict_type\(\) are being deprecated",
            ):
                # 获取原始模型的状态字典
                get_model_state_dict(model)

    # 带有通信装饰器的测试方法，仅在 GPU 数量大于等于 2 时执行
    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_shared_weight(self):
        # 定义绑定嵌入模型类
        class TiedEmbeddingModel(nn.Module):
            def __init__(self, vocab_size, embedding_dim):
                super().__init__()
                # 定义嵌入层，输入词汇大小和嵌入维度
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                # 定义线性层作为解码器，并将其权重绑定到嵌入层的权重上
                self.decoder = nn.Linear(embedding_dim, vocab_size)
                self.decoder.weight = self.embedding.weight  # Tying weights

            def forward(self, input):
                # 将输入乘以 10 并转换为整数类型
                input = (input * 10).to(torch.int)
                # 对输入进行嵌入操作
                embedded = self.embedding(input)
                # 使用解码器进行解码操作
                output = self.decoder(embedded)
                return output

        # 初始化模型和优化器方法
        def init_model_optim():
            # 初始化设备网格，指定为 CUDA，且大小为当前世界大小
            device_mesh = init_device_mesh("cuda", (self.world_size,))
            # 创建原始绑定嵌入模型，并将其移动到 CUDA 设备
            orig_model = TiedEmbeddingModel(10000, 300).to(torch.device("cuda"))
            # 使用 AdamW 优化器初始化原始模型的参数
            orig_optim = torch.optim.AdamW(orig_model.parameters(), lr=1e-3)
            # 深拷贝原始模型，然后创建另一个 AdamW 优化器
            copy_optim = torch.optim.AdamW(orig_model.parameters(), lr=1e-3)
            # 使用设备网格初始化 FSDP 模型，并使用 AdamW 优化器初始化其参数
            dist_model = FSDP(copy.deepcopy(orig_model), device_mesh=device_mesh)
            dist_optim = torch.optim.AdamW(dist_model.parameters(), lr=1e-3)
            return orig_model, orig_optim, copy_optim, dist_model, dist_optim

        # 执行保存加载测试方法
        self._test_save_load(init_model_optim)
class TestNoComm(MultiProcessTestCase):
    # 定义一个测试类 TestNoComm，继承自 MultiProcessTestCase
    def setUp(self) -> None:
        # 设置测试方法的准备工作，调用父类的 setUp 方法
        super().setUp()
        # 启动多进程测试环境
        self._spawn_processes()

    @skip_if_lt_x_gpu(1)
    # 如果 GPU 数量小于 1，则跳过该测试方法
    def test_no_dist(self) -> None:
        # 定义一个测试方法 test_no_dist，无返回值
        # 创建一个 CompositeParamModel 模型，使用 CUDA 设备
        model = CompositeParamModel(device=torch.device("cuda"))
        # 使用 AdamW 优化器，学习率为 1e-3
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # 断言分布式环境未初始化
        self.assertFalse(dist.is_initialized())
        # 获取模型的完整状态字典，且在 CPU 上进行数据转移
        msd = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        # 检查状态字典中所有参数都不在 CUDA 设备上
        for v in msd.values():
            self.assertFalse(v.is_cuda)
        # 断言模型的当前状态字典与获取的状态字典相等
        self.assertEqual(model.state_dict(), msd)
        # 将模型状态字典设置回模型
        set_model_state_dict(model, model.state_dict())
        # 获取优化器的完整状态字典，且在 CPU 上进行数据转移
        osd = get_optimizer_state_dict(
            model,
            optim,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        # 将优化器的状态字典设置回模型的优化器
        set_optimizer_state_dict(model, optim, osd)
        # 再次设置优化器的状态字典
        set_optimizer_state_dict(model, optim, optim.state_dict())


if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行测试
    run_tests()
```