# `.\pytorch\test\distributed\_composable\fully_shard\test_fully_shard_init.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入必要的模块和库
import copy  # 导入深拷贝函数
import sys  # 导入系统相关功能
from typing import Optional  # 导入类型提示

import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入分布式模块
import torch.distributed.fsdp._traversal_utils as traversal_utils  # 导入FSDP相关的遍历工具
import torch.nn as nn  # 导入神经网络模块
from torch.distributed._composable import fully_shard  # 导入完全分片函数
from torch.distributed.fsdp import BackwardPrefetch, FullyShardedDataParallel as FSDP  # 导入FSDP相关模块
from torch.distributed.fsdp._common_utils import _is_fsdp_flattened, clean_tensor_name  # 导入FSDP通用工具函数
from torch.distributed.fsdp.wrap import _Policy, CustomPolicy, ModuleWrapPolicy  # 导入FSDP相关的封装策略
from torch.testing._internal.common_dist_composable import (  # 导入测试相关模块
    CompositeParamModel,
    FakeSequential,
    NestedSequentialModel,
    UnitModule,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入分布式测试辅助函数
from torch.testing._internal.common_fsdp import FSDPTest  # 导入FSDP测试基类
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN  # 导入测试运行相关函数

# 如果分布式不可用，输出提示信息并退出
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 如果开启了dev-asan测试，输出相关信息并退出，因为torch + multiprocessing spawn存在已知问题
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 测试类，继承自FSDPTest基类，用于测试完全分片初始化
class TestInitialization(FSDPTest):
    """Tests ``fully_shard`` initialization."""

    @property
    def world_size(self) -> int:
        return 2  # 返回世界大小为2

    @skip_if_lt_x_gpu(2)  # 如果GPU少于2个，跳过测试
    def test_policy(self):
        """Tests passing a ``policy`` for pseudo-auto-wrapping."""

        # 定义一个lambda函数，用于检查模块是否是Sequential或FakeSequential类型
        def lambda_fn(module: nn.Module):
            if isinstance(module, nn.Sequential):
                return True
            elif isinstance(module, FakeSequential):
                return {"backward_prefetch": BackwardPrefetch.BACKWARD_POST}
            return False

        # 运行子测试，测试不同的策略
        self.run_subtests(
            {
                "policy": [
                    None,
                    ModuleWrapPolicy({UnitModule}),  # 使用UnitModule进行封装策略
                    ModuleWrapPolicy({nn.Sequential}),  # 使用nn.Sequential进行封装策略
                    CustomPolicy(lambda_fn),  # 使用自定义的lambda函数作为封装策略
                ],
            },
            self._test_policy,  # 调用_test_policy函数进行测试
        )

    # 测试具体的封装策略函数
    def _test_policy(self, policy: Optional[_Policy]):
        # 检查是否使用了嵌套的Sequential模型
        use_nested_sequential_model = "Sequential" in getattr(
            policy, "_module_classes_str", ""
        )
        # 根据使用的模型类型选择本地模型
        local_model = (
            NestedSequentialModel(torch.device("cuda"))  # 使用嵌套Sequential模型
            if use_nested_sequential_model
            else CompositeParamModel(torch.device("cuda"))  # 使用参数组合模型
        )
        # 使用FSDP进行模型封装
        fsdp_wrapped_model = FSDP(
            copy.deepcopy(local_model),  # 深度拷贝本地模型
            auto_wrap_policy=policy,  # 使用指定的封装策略
            use_orig_params=True,  # 使用原始参数
        )
        # 对比完全分片函数的构造结果
        composable_module = copy.deepcopy(local_model)
        fully_shard(
            composable_module,
            policy=policy,  # 使用指定的封装策略
        )
        self._test_fully_shard_construction(
            local_model,
            fsdp_wrapped_model,
            composable_module,
        )

    @skip_if_lt_x_gpu(2)  # 如果GPU少于2个，跳过测试
    # 定义测试手动应用“fully_shard”的方法
    def test_manual_fully_shard(self):
        """Tests manually applying ``fully_shard``."""
        # 创建本地模型对象
        local_model = CompositeParamModel(torch.device("cuda"))
        # 深度复制本地模型对象
        fsdp_wrapped_model = copy.deepcopy(local_model)
        # 使用FSDP包装模型对象的u2属性
        fsdp_wrapped_model.u2 = FSDP(fsdp_wrapped_model.u2, use_orig_params=True)
        # 使用FSDP包装整个模型对象
        fsdp_wrapped_model = FSDP(fsdp_wrapped_model, use_orig_params=True)
        # 深度复制本地模型对象
        composable_module = copy.deepcopy(local_model)
        # 对composable_module的u2属性应用fully_shard
        fully_shard(composable_module.u2)
        # 对整个composable_module应用fully_shard
        fully_shard(composable_module)
        # 调用测试fully_shard构造的方法，传入参数为本地模型、FSDP包装模型、composable_module
        self._test_fully_shard_construction(
            local_model,
            fsdp_wrapped_model,
            composable_module,
        )

    # 定义测试fully_shard构造的方法
    def _test_fully_shard_construction(
        self,
        local_model: nn.Module,
        fsdp_wrapped_model: FSDP,
        composable_module: nn.Module,
    ):
        # 检查可组合模块是否具有与本地模型相同的名称，以及与 FSDP 封装模型相同的分片参数
        for (
            (local_name, _),
            (composable_name, composable_param),
            (_, fsdp_wrapped_param),
        ) in zip(
            local_model.named_parameters(),  # 获取本地模型的命名参数
            composable_module.named_parameters(),  # 获取可组合模块的命名参数
            fsdp_wrapped_model.named_parameters(),  # 获取 FSDP 封装模型的命名参数
        ):
            self.assertEqual(local_name, composable_name)  # 检查本地模型参数名是否与可组合模块参数名相同
            self.assertEqual(fsdp_wrapped_param, composable_param)  # 检查 FSDP 封装模型参数是否与可组合模块参数相同

        # 检查可组合模块是否具有与 FSDP 封装模型相同的 `FlatParameter` 结构
        composable_handles = traversal_utils._get_fsdp_handles(composable_module)  # 获取可组合模块的 FSDP 句柄
        fsdp_wrapped_handles = traversal_utils._get_fsdp_handles(fsdp_wrapped_model)  # 获取 FSDP 封装模型的 FSDP 句柄
        self.assertEqual(len(composable_handles), len(fsdp_wrapped_handles))  # 检查两个模型的 FSDP 句柄数量是否相同
        for composable_handle, fsdp_wrapped_handle in zip(
            composable_handles, fsdp_wrapped_handles
        ):
            self.assertEqual(
                composable_handle.flat_param.shape, fsdp_wrapped_handle.flat_param.shape
            )  # 检查 FSDP 句柄的平坦参数形状是否相同
            self.assertEqual(
                composable_handle.flat_param._fqns,
                fsdp_wrapped_handle.flat_param._fqns,
            )  # 检查 FSDP 句柄的平坦参数全限定名是否相同

        # 检查可组合模块是否没有添加任何包装类
        local_module_classes = set()
        composable_module_classes = set()
        local_module_classes.update(
            type(submodule) for submodule in local_model.modules()
        )  # 收集本地模型中所有子模块的类型
        composable_module_classes.update(
            type(submodule) for submodule in composable_module.modules()
        )  # 收集可组合模块中所有子模块的类型
        self.assertEqual(local_module_classes, composable_module_classes)  # 检查本地模型和可组合模块的子模块类型集合是否相同

        # 检查可组合模块是否具有与 FSDP 封装模型相同的 FSDP 状态及其属性
        wrapper_states = traversal_utils._get_fsdp_states(fsdp_wrapped_model)  # 获取 FSDP 封装模型的 FSDP 状态
        composable_states = traversal_utils._get_fsdp_states(composable_module)  # 获取可组合模块的 FSDP 状态
        self.assertEqual(len(wrapper_states), len(composable_states))  # 检查两个模型的 FSDP 状态数量是否相同
        for wrapper_state, composable_state in zip(wrapper_states, composable_states):
            self.assertEqual(
                wrapper_state.sharding_strategy, composable_state.sharding_strategy
            )  # 检查 FSDP 状态的分片策略是否相同
            self.assertEqual(
                wrapper_state.backward_prefetch, composable_state.backward_prefetch
            )  # 检查 FSDP 状态的后向预取是否相同

    @skip_if_lt_x_gpu(2)
    def test_device_id(self):
        """Tests passing a ``device_id``."""
        # 定义 CPU 设备
        cpu_device = torch.device("cpu")
        # 创建一个 CompositeParamModel 实例，指定设备为 CPU
        composable_module = CompositeParamModel(device=cpu_device)
        # 遍历模型的参数，确保它们被初始化在 CPU 上
        for param in composable_module.parameters():
            assert (
                param.device == cpu_device
            ), "Expects module to be initialized on CPU for this unit test"
        # 对 composable_module 进行完全分片操作
        fully_shard(
            composable_module,
            policy=ModuleWrapPolicy({UnitModule}),
            device_id=self.rank,
        )
        # 再次遍历模型的参数，确保它们被正确地初始化在指定的 CUDA 设备上
        for param in composable_module.parameters():
            self.assertEqual(param.device, torch.device("cuda", self.rank))

    @skip_if_lt_x_gpu(2)
    def test_sync_module_states(self):
        """Tests passing ``sync_module_states=True``."""
        # 创建一个本地的 CompositeParamModel 实例，指定设备为 CUDA
        local_model = CompositeParamModel(device=torch.device("cuda"))
        # 使用深度复制创建一个 composable_module 的副本
        composable_module = copy.deepcopy(local_model)
        # 如果当前 rank 不是 0，则将 composable_module 的参数置零
        if self.rank != 0:
            for param in composable_module.parameters():
                with torch.no_grad():
                    param.zero_()
        # 定义模块包装策略
        policy = ModuleWrapPolicy({UnitModule})
        # 使用 FSDP 封装本地模型，确保使用原始参数
        fsdp_wrapped_model = FSDP(
            copy.deepcopy(local_model),
            auto_wrap_policy=policy,
            use_orig_params=True,
        )
        # 对 composable_module 进行完全分片操作，并同步模块状态
        fully_shard(
            composable_module,
            policy=policy,
            sync_module_states=True,
        )
        # 遍历 composable_module 和 fsdp_wrapped_model 的参数，确保它们相等
        for composable_param, fsdp_wrapped_param in zip(
            composable_module.parameters(),
            fsdp_wrapped_model.parameters(),
        ):
            self.assertEqual(composable_param, fsdp_wrapped_param)

    @skip_if_lt_x_gpu(2)
    # 定义一个测试方法，用于测试元设备模块的实例化
    def test_materialize_meta_module(self):
        """Tests materializing a meta-device module."""

        def _param_init_fn(module: nn.Module):
            """
            This is an example ``param_init_fn`` for composable FSDP.

            TODO: This function is not satisfactory because:
            (1) This requires guarding with ``_is_fsdp_flattened()``. This
            guard is needed to avoid re-initializing parameters for nested
            cases since some initialization methods strictly require non-1D
            shape (e.g. ``kaiming_uniform_()``), while FSDP replaces the
            original parameters with their 1D shards.
            (2) This requires module-by-module traversal and manual ``setattr``
            usage as opposed to first calling ``module.to_empty()`` and then
            initializing each parameter after. The latter will override the
            initialization of already-initialized nested parameters. In other
            words, this parameter initialization function must strictly modify
            only the parameters on meta device.
            """
            # 设置随机数种子
            torch.manual_seed(0)
            # 遍历模块内的子模块
            for submodule in module.modules():
                # 遍历每个子模块的参数
                for param_name, param in submodule.named_parameters(recurse=False):
                    # 检查参数是否非FSDP扁平化且为元参数
                    if not _is_fsdp_flattened(param) and param.is_meta:
                        # 创建一个与参数形状相同的空参数张量，放置在CUDA设备上
                        materialized_param = nn.Parameter(
                            torch.empty_like(param, device=torch.device("cuda"))
                        )
                        # 使用均匀分布初始化参数
                        nn.init.uniform_(materialized_param)
                        # 将初始化后的参数设置回子模块中
                        setattr(submodule, param_name, materialized_param)

        # 创建一个使用元设备的复合参数模型
        composable_module = CompositeParamModel(device=torch.device("meta"))
        # 创建另一个使用元设备的复合参数模型
        meta_model = CompositeParamModel(device=torch.device("meta"))
        # 使用FSDP封装模型，自动封装策略为ModuleWrapPolicy({UnitModule})，并使用指定的参数初始化函数
        fsdp_wrapped_model = FSDP(
            meta_model,
            auto_wrap_policy=ModuleWrapPolicy({UnitModule}),
            param_init_fn=_param_init_fn,
            use_orig_params=True,
        )
        # 对composable_module进行完全分片，使用指定的封装策略和参数初始化函数
        fully_shard(
            composable_module,
            policy=ModuleWrapPolicy({UnitModule}),
            param_init_fn=_param_init_fn,
        )
        # 遍历composable_module和fsdp_wrapped_model的命名参数，并进行比较
        for (
            (composable_param_name, composable_param),
            (fsdp_wrapped_param_name, fsdp_wrapped_param),
        ) in zip(
            composable_module.named_parameters(),
            fsdp_wrapped_model.named_parameters(),
        ):
            # 使用断言确保命名参数一致
            self.assertEqual(
                composable_param_name, clean_tensor_name(fsdp_wrapped_param_name)
            )
            # 使用断言确保参数在CUDA设备上
            self.assertEqual(
                composable_param.device,
                torch.device("cuda", torch.cuda.current_device()),
            )
            # 使用断言确保参数值一致
            self.assertEqual(composable_param, fsdp_wrapped_param)

    # 跳过如果GPU数量小于2
    @skip_if_lt_x_gpu(2)
    # 定义一个测试方法，用于测试嵌套应用完全分片功能是否共享预期的数据结构状态。
    def test_nested_fully_shard_shared_state(self):
        """
        Tests that nested applications of ``fully_shard`` share the expected
        data structure state.
        """
        # 运行子测试，传入不同的策略参数组合和测试方法 `_test_nested_fully_shard_shared_state`
        self.run_subtests(
            {"use_policy": [False, True]},
            self._test_nested_fully_shard_shared_state,
        )

    # 定义具体的测试方法 `_test_nested_fully_shard_shared_state`，接受一个布尔类型的 `use_policy` 参数
    def _test_nested_fully_shard_shared_state(self, use_policy: bool):
        # 指定使用 CUDA 设备
        device = torch.device("cuda")
        # 创建一个 CompositeParamModel 实例，传入设备参数
        composable_module = CompositeParamModel(device=device)
        
        # 根据 `use_policy` 参数选择是否应用策略 `ModuleWrapPolicy({UnitModule})` 
        if use_policy:
            fully_shard(composable_module, policy=ModuleWrapPolicy({UnitModule}))
        else:
            # 否则，分别对 composable_module 的 u1、u2 和整体应用完全分片
            fully_shard(composable_module.u1)
            fully_shard(composable_module.u2)
            fully_shard(composable_module)

        # 运行一次前向传播，触发懒初始化
        inp = torch.randn((2, 100), device=device)
        composable_module(inp)

        # 检查所有应用了 `fully_shard` 的模块是否共享相同的数据结构状态
        # 对给定名称的数据结构进行检查，不需要检查所有名称来验证共享是否生效。
        # 注意：此检查仅需要数据结构状态被共享。即使共享 FSDP 状态对象本身已足够但并非必须。
        data_structure_names = [
            "_exec_order_data",
            "_free_event_queue",
            "_pre_unshard_stream",
            "_unshard_stream",
            "_post_backward_stream",
            "_default_stream",
        ]
        for data_structure_name in data_structure_names:
            all_structures = set()
            # 收集所有模块中指定数据结构名对应对象的 ID
            all_structures.update(
                id(getattr(fully_shard.state(module), data_structure_name))
                for module in (
                    composable_module.u1,
                    composable_module.u2,
                    composable_module,
                )
            )
            # 断言所有收集到的结构对象 ID 都相同，即数据结构状态被共享
            self.assertEqual(len(all_structures), 1)
# 如果这个脚本被直接执行（而不是被作为模块导入），则执行以下操作
if __name__ == "__main__":
    # 调用名为 run_tests() 的函数来运行测试
    run_tests()
```