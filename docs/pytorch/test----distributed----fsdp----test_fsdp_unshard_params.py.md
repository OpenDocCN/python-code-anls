# `.\pytorch\test\distributed\fsdp\test_fsdp_unshard_params.py`

```py
# 导入必要的库和模块
import contextlib  # 提供了上下文管理工具的模块
import itertools  # 提供了创建和操作迭代器的函数的模块
import math  # 提供了数学运算函数的模块
import sys  # 提供了与 Python 解释器进行交互的功能
from typing import Any, Dict, List, Optional, Union  # 引入类型提示相关的模块

import torch  # 引入 PyTorch 深度学习库
import torch.distributed.fsdp._traversal_utils as traversal_utils  # 引入特定分布式模块
import torch.nn as nn  # 引入神经网络模块
from torch import distributed as dist  # 引入分布式计算模块
from torch.distributed.fsdp import (  # 引入 FSDP 相关模块
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp._common_utils import clean_tensor_name  # 引入清理张量名称的工具函数
from torch.distributed.fsdp._flat_param import FlatParameter  # 引入扁平化参数类
from torch.distributed.fsdp.fully_sharded_data_parallel import FLAT_PARAM  # 引入 FSDP 的扁平化参数常量
from torch.distributed.fsdp.wrap import ModuleWrapPolicy  # 引入模块包装策略类
from torch.nn.parallel.distributed import DistributedDataParallel as DDP  # 引入分布式数据并行类 DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 引入测试辅助函数
from torch.testing._internal.common_fsdp import (  # 引入 FSDP 测试相关的模块和类
    CUDAInitMode,
    FSDPInitMode,
    FSDPTest,
    NestedWrappedModule,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN  # 引入测试运行和配置相关的工具函数

# 检查分布式环境是否可用，若不可用则输出提示信息并退出
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 如果配置为使用开发模式的 AddressSanitizer（ASAN），则输出相关提示信息并退出
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestUnshardParamsBase(FSDPTest):
    """
    This contains any methods common to both the sharded and non-sharded cases.
    """

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)  # 返回基于 CUDA 的设备对象，使用当前进程的排名

    def _test_unshard_params_writeback(
        self,
        writeback: bool,
        check_outer: bool,
        **fsdp_kwargs: Dict[str, Any],
        # 定义一个方法，测试参数的写回功能
    ):
        # 创建一个包含两个线性层的神经网络模型，每层输入输出维度为5，无偏置项，使用指定的设备
        model = nn.Sequential(
            nn.Linear(5, 5, bias=False, device=self.device),
            nn.Linear(5, 3, bias=False, device=self.device),
        )
        # 在第一个线性层上应用FSDP（Fully Sharded Data Parallel）优化，使用指定的FSDP参数
        model[0] = FSDP(model[0], **fsdp_kwargs)
        # 将整个模型应用FSDP优化，使用指定的FSDP参数
        model = FSDP(model, **fsdp_kwargs)
        # 检查模型是否使用了分片策略
        uses_sharded_strategy = model.sharding_strategy != ShardingStrategy.NO_SHARD
        # 获取模型的CPU数据卸载的参数
        offloading_params = model.cpu_offload.offload_params

        # 假设使用深度优先遍历模型的`.parameters()`方法
        # 选择要检查的参数，根据check_outer决定是外部模型参数还是第一个线性层的参数
        outer_param: Union[FlatParameter, nn.Parameter] = next(model.parameters())
        inner_param: Union[FlatParameter, nn.Parameter] = next(model[0].parameters())
        param_to_check = outer_param if check_outer else inner_param

        # 使用torch.no_grad()上下文管理器，将param_to_check的所有元素置零，然后加上self.rank + 2
        with torch.no_grad():
            param_to_check.zero_()
            param_to_check += self.rank + 2

        # 使用FSDP.summon_full_params(model, writeback=writeback)上下文管理器，将模型的所有参数置零
        with FSDP.summon_full_params(model, writeback=writeback), torch.no_grad():
            for param in model.parameters():
                param.zero_()

        # 检查分片参数（sharded parameter）的第一个单例元素是否被正确地置零
        param_elem_to_check = param_to_check[0]
        if param_elem_to_check.numel() > 1:
            # 如果param_elem_to_check不是单元素，而是原始参数（use_orig_params=True）且没有分片策略（NO_SHARD），则再次访问
            param_elem_to_check = param_elem_to_check[0]

        # 根据条件检查param_elem_to_check的值是否符合预期（与self.rank + 2相等或者为0）
        if writeback or (not uses_sharded_strategy and not offloading_params):
            self.assertEqual(param_elem_to_check, 0)
        else:
            self.assertEqual(param_elem_to_check, self.rank + 2)

        # 如果开启了参数CPU卸载，确保模型的所有参数都被移到CPU设备上
        if offloading_params:
            cpu_device = torch.device("cpu")
            for param in model.parameters():
                self.assertEqual(param.device, cpu_device)

    # 返回测试用例的参数组合字典
    def _get_test_unshard_params_writeback_config(self) -> Dict[str, List[Any]]:
        return {
            "writeback": [True, False],
            "check_outer": [True, False],
            "mixed_precision": [MixedPrecision(param_dtype=torch.float16), None],
            "cpu_offload": [
                CPUOffload(offload_params=False),
                CPUOffload(offload_params=True),
            ],
            "use_orig_params": [True, False],
        }

    # 测试参数数据的方法，用于测试参数未分片的情况
    def _test_unshard_params_param_data(
        self,
        rank0_only: bool,
        offload_to_cpu: bool,
        cpu_offload: CPUOffload,
        mixed_precision: Optional[MixedPrecision],
        use_orig_params: bool,
        ):
            # 初始化一个本地模型，使用指定的参数进行初始化
            local_model = NestedWrappedModule.init(
                self.process_group,
                FSDPInitMode.NO_FSDP,
                CUDAInitMode.CUDA_BEFORE,
                fsdp_kwargs={},
                deterministic=True,
            )
            # 应用 FSDP，确保根模块没有应用 FSDP，但存在多个 FSDP 根子模块（后面会证明）
            fsdp_model = NestedWrappedModule.init(
                self.process_group,
                FSDPInitMode.RECURSIVE,
                CUDAInitMode.CUDA_BEFORE,
                fsdp_kwargs={
                    "cpu_offload": cpu_offload,
                    "mixed_precision": mixed_precision,
                    "use_orig_params": use_orig_params,
                },
                deterministic=True,
            )
            self.assertFalse(isinstance(fsdp_model, FSDP))

            # 因为获取以下名称非常复杂，所以硬编码它们
            non_fsdp_managed_param_names = {
                "module.0.weight",
                "module.0.bias",
                "module.3.weight",
                "module.3.bias",
            }

            # 使用 FSDP.summon_full_params 上下文管理器，处理完整的参数
            with FSDP.summon_full_params(
                fsdp_model,
                rank0_only=rank0_only,
                writeback=not rank0_only,
                offload_to_cpu=offload_to_cpu,
            ):
                # 如果不仅仅是 rank0 或者当前进程是 rank0
                if not rank0_only or self.rank == 0:
                    # 遍历本地模型和 FSDP 模型的参数
                    for p1, (n2, p2) in zip(
                        local_model.parameters(), fsdp_model.named_parameters()
                    ):
                        # 断言两个参数的形状相同
                        self.assertEqual(p1.shape, p2.shape)
                        # 如果启用了 offload_to_cpu，并且参数名称不在非 FSDP 管理的参数名列表中
                        if (
                            offload_to_cpu
                            and clean_tensor_name(n2) not in non_fsdp_managed_param_names
                        ):
                            # 断言参数 p2 在 CPU 设备上
                            self.assertEqual(torch.device("cpu"), p2.device)
                        else:
                            # 否则断言两个参数在相同的设备上
                            self.assertEqual(p1.device, p2.device)
                        # 断言两个参数的数据类型相同，即使 FSDP 使用混合精度
                        self.assertEqual(
                            p1.dtype, p2.dtype
                        )
                        # 断言两个参数的值相等
                        self.assertEqual(p1, p2)
                        # 断言 p2 是 nn.Parameter 类型
                        self.assertTrue(isinstance(p2, nn.Parameter))
                else:
                    # 否则，对于每个 FlatParameter，检查其是否具有分片大小，作为重新分片的代理
                    for handle in traversal_utils._get_fsdp_handles(fsdp_model):
                        if handle.uses_sharded_strategy:
                            # 断言 FlatParameter 的形状与其分片大小相等
                            self.assertEqual(
                                handle.flat_param.shape, handle.flat_param._sharded_size
                            )
                        else:
                            # 否则断言 FlatParameter 的形状与其未填充未分片大小相等
                            self.assertEqual(
                                handle.flat_param.shape,
                                handle.flat_param._unpadded_unsharded_size,
                            )

            # 证明在惰性初始化后，FSDP 根的数量
            num_fsdp_roots = 0
            for fsdp_state in traversal_utils._get_fsdp_states(fsdp_model):
                num_fsdp_roots += fsdp_state._is_root
            # 断言 FSDP 根的数量大于 1
            self.assertGreater(num_fsdp_roots, 1)
    # 定义一个私有方法 _get_test_unshard_params_param_data_config，返回一个字典，键为字符串，值为列表，每个列表包含多种不同类型的对象
    def _get_test_unshard_params_param_data_config(self) -> Dict[str, List[Any]]:
        # 返回包含以下键值对的字典：
        return {
            "rank0_only": [False, True],  # 键 "rank0_only" 对应的值是包含布尔值 False 和 True 的列表
            "offload_to_cpu": [False, True],  # 键 "offload_to_cpu" 对应的值是包含布尔值 False 和 True 的列表
            "cpu_offload": [  # 键 "cpu_offload" 对应的值是包含两个 CPUOffload 对象的列表
                CPUOffload(offload_params=False),  # CPUOffload 对象的 offload_params 属性为 False
                CPUOffload(offload_params=True),   # CPUOffload 对象的 offload_params 属性为 True
            ],
            "mixed_precision": [  # 键 "mixed_precision" 对应的值是包含两个 MixedPrecision 对象的列表
                MixedPrecision(param_dtype=torch.float16),  # MixedPrecision 对象指定了参数类型为 torch.float16
                None,  # 第二个元素是 None，表示未指定具体的 MixedPrecision 对象
            ],
            "use_orig_params": [True, False],  # 键 "use_orig_params" 对应的值是包含布尔值 True 和 False 的列表
        }
class TestUnshardParams(TestUnshardParamsBase):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_unshard_params_writeback(self):
        """Tests the ``writeback`` argument (using default for all others)."""
        # 执行子测试，使用默认设置测试 ``writeback`` 参数
        self.run_subtests(
            self._get_test_unshard_params_writeback_config(),
            self._test_unshard_params_writeback,
        )

    @skip_if_lt_x_gpu(2)
    def test_unshard_params_param_data(self):
        """
        Tests that parameters are exposed correctly for ``recurse=True`` and
        all other argument configs for a non-FSDP root module.
        """
        # 执行子测试，验证参数在 ``recurse=True`` 和其它配置下是否正确暴露
        self.run_subtests(
            self._get_test_unshard_params_param_data_config(),
            self._test_unshard_params_param_data,
        )

    @skip_if_lt_x_gpu(2)
    def test_unshard_singleton_param_writeback(self):
        """
        Tests ``writeback=True`` for a singleton parameter, which includes
        testing that writing to padding does not persist.
        NOTE: This method depends on FSDP internals.
        """
        # 创建一个 FSDP 模型，并测试单一参数的 ``writeback=True``，
        # 包括确保写入填充值不会持久化
        model = FSDP(nn.Linear(1, 1, bias=False, device=self.device))
        flat_param = model._handle.flat_param
        self.assertEqual(1, flat_param.numel())
        # 向 *sharded* 的 `FlatParameter` 写入已知值
        with torch.no_grad():
            # 对于非零等级，此写入是对填充的
            flat_param[0] = self.rank + 2
        # 使用 ``summon_full_params()`` 启用完整参数，确保写入单一参数
        with FSDP.summon_full_params(model, writeback=True):
            self.assertEqual(1, flat_param.numel())
            with torch.no_grad():
                flat_param.zero_()
        # 注意：这里检查填充写入是否持久化，对正确性来说并非严格要求
        if self.rank == 0:  # 没有写入填充
            self.assertEqual(0, flat_param[0])
        else:  # 写入了填充
            self.assertEqual(self.rank + 2, flat_param[0])

    @skip_if_lt_x_gpu(2)
    def test_unshard_params_respects_reshard(self):
        """
        Tests that unsharding parameters respects the expected reshard behavior
        between forward and backward as well as after backward.

        For mixed precision, we should *not* respect the reshard behavior
        because the ``summon_full_params()`` forces full precision, which uses
        a different all-gather tensor than the one already in memory and will
        not persist any modifications correctly.
        """
        # 执行子测试，验证参数解散在前向和后向传播以及后向后是否符合预期的重新分配行为
        # 对于混合精度，我们不应该尊重重新分配行为，因为 ``summon_full_params()``
        # 强制使用全精度，这使用了一个不同的全收集张量，不会正确保留任何修改
        self.run_subtests(
            {
                "rank0_only": [False, True],
                "offload_to_cpu": [False, True],
                "mixed_precision": [MixedPrecision(param_dtype=torch.float16), None],
                "use_orig_params": [False, True],
            },
            self._test_unshard_params_respects_reshard,
        )
    # 定义测试函数 `_test_unshard_params_respects_reshard`，用于测试参数是否符合重新分片的要求
    def _test_unshard_params_respects_reshard(
        self,
        # 是否仅在 rank0 执行测试
        rank0_only: bool,
        # 是否将数据迁移到 CPU 进行测试
        offload_to_cpu: bool,
        # 混合精度设置，可以是 MixedPrecision 类型或 None
        mixed_precision: Optional[MixedPrecision],
        # 是否使用原始参数进行测试
        use_orig_params: bool,
    ):
        """NOTE: This method depends on FSDP internals."""
        # 定义FSDP模型的关键字参数字典
        fsdp_kwargs = {
            "mixed_precision": mixed_precision,
            "use_orig_params": use_orig_params,
        }
        # 创建包含两个线性层的FSDP模型，每层具有5个输入和5个输出，无偏置，设备为self.device
        model = FSDP(
            nn.Sequential(
                FSDP(nn.Linear(5, 5, bias=False, device=self.device), **fsdp_kwargs),
                nn.Linear(5, 3, bias=False, device=self.device),
            ),
            **fsdp_kwargs,
        )
        # 获取外层参数的扁平化表示
        outer_flat_param = model._handle.flat_param
        # 获取内层参数的扁平化表示
        inner_flat_param = model.module[0]._handle.flat_param
        # NOTE: This assumes uniform sharding with padding across ranks.
        # 计算外层参数的未分片大小，假设在所有排名上具有均匀分片和填充
        expected_outer_flat_param_unsharded_numel = (
            outer_flat_param.numel() * self.world_size
        )

        def _get_unsharded_storage_size(flat_param: FlatParameter):
            return flat_param._full_param_padded.storage().size()

        # Validate the expected behavior: the root does not reshard after
        # forward; the non-root reshards after forward; and both reshard after
        # backward
        # 执行模型前向传播，并验证预期行为：根据排名，根节点不在前向传播后进行重新分片，非根节点在前向传播后重新分片，所有节点在反向传播后重新分片
        output = model(torch.zeros(5, device=self.device))
        self.assertEqual(
            expected_outer_flat_param_unsharded_numel,
            _get_unsharded_storage_size(outer_flat_param),
        )
        self.assertEqual(0, _get_unsharded_storage_size(inner_flat_param))
        output.sum().backward()
        self.assertEqual(0, _get_unsharded_storage_size(outer_flat_param))
        self.assertEqual(0, _get_unsharded_storage_size(inner_flat_param))

        # Check that with parameter unsharding in between forward and backward
        # as well as after backward, the reshard behavior matches
        # 再次执行模型前向传播，并检查前向和反向传播之间以及反向传播后参数未分片的情况，验证重新分片行为是否一致
        output = model(torch.zeros(5, device=self.device))
        with FSDP.summon_full_params(
            model,
            rank0_only=rank0_only,
            writeback=not rank0_only,
            offload_to_cpu=offload_to_cpu,
        ):
            pass
        if mixed_precision is not None:
            # After forcing full precision, we must invalidate the existing
            # unsharded low-precision flat parameter since it will not persist
            # changes from the `summon_full_params()` context, so we cannot
            # respect the reshard behavior
            # 在强制使用全精度后，必须使现有的未分片低精度扁平参数无效，因为它不会保留`summon_full_params()`上下文中的更改，所以不能保证重新分片行为
            expected_outer_flat_param_unsharded_numel = 0
        self.assertEqual(
            expected_outer_flat_param_unsharded_numel,
            _get_unsharded_storage_size(outer_flat_param),
        )
        self.assertEqual(0, _get_unsharded_storage_size(inner_flat_param))
        output.sum().backward()
        with FSDP.summon_full_params(
            model,
            rank0_only=rank0_only,
            writeback=not rank0_only,
            offload_to_cpu=offload_to_cpu,
        ):
            pass
        self.assertEqual(0, _get_unsharded_storage_size(outer_flat_param))
        self.assertEqual(0, _get_unsharded_storage_size(inner_flat_param))

    @skip_if_lt_x_gpu(2)
    def test_unshard_params_recurse(self):
        """Tests the ``recurse`` argument (using default for all others)."""
        # 运行子测试，测试不同参数组合下的 ``_test_unshard_params_recurse`` 方法
        self.run_subtests(
            {
                "recurse": [False, True],
                "unshard_outer": [False, True],
                "mixed_precision": [MixedPrecision(param_dtype=torch.float16), None],
                "use_orig_params": [False, True],
            },
            self._test_unshard_params_recurse,
        )

    def _test_unshard_params_recurse(
        self,
        recurse: bool,
        unshard_outer: bool,
        mixed_precision: Optional[MixedPrecision],
        use_orig_params: bool,
    ):
        """NOTE: This method depends on FSDP internals."""
        # 根据给定参数设置 FSDP 的关键字参数
        fsdp_kwargs = {
            "mixed_precision": mixed_precision,
            "use_orig_params": use_orig_params,
        }
        # 创建 FSDP 模型，包含线性层序列
        model = FSDP(
            nn.Sequential(
                FSDP(nn.Linear(5, 5, bias=False, device=self.device), **fsdp_kwargs),
                nn.Linear(5, 3, bias=False, device=self.device),
            ),
            **fsdp_kwargs,
        )
        # 计算未分片参数的元素个数
        unsharded_inner_numel = 5 * 5
        unsharded_outer_numel = 5 * 3
        if use_orig_params:
            # 考虑未分片填充：因为每个 `FlatParameter` 只有一个原始参数，
            # 我们只需要为了世界大小的整除性进行填充，而不是地址对齐
            if unsharded_inner_numel % self.world_size:
                unsharded_inner_numel += self.world_size - (
                    unsharded_inner_numel % self.world_size
                )
            if unsharded_outer_numel % self.world_size:
                unsharded_outer_numel += self.world_size - (
                    unsharded_outer_numel % self.world_size
                )
        # 四舍五入分片后的元素个数以考虑填充
        sharded_inner_numel = int(math.ceil(unsharded_inner_numel / self.world_size))
        sharded_outer_numel = int(math.ceil(unsharded_outer_numel / self.world_size))
        # 获取内部和外部的扁平参数
        inner_flat_param = model.module[0]._handle.flat_param
        outer_flat_param = model._handle.flat_param
        # 断言分片后的元素个数
        self.assertEqual(sharded_inner_numel, inner_flat_param.numel())
        self.assertEqual(sharded_outer_numel, outer_flat_param.numel())
        # 根据条件设定期望的外部元素个数和内部元素个数
        expected_outer_numel = (
            unsharded_outer_numel if unshard_outer else sharded_outer_numel
        )
        expected_inner_numel = (
            unsharded_inner_numel
            if recurse or not unshard_outer
            else sharded_inner_numel
        )
        # 根据是否在外部分片条件下选择模块进行分片
        module_to_unshard = model if unshard_outer else model[0]
        # 使用 FSDP.summon_full_params 激活完整参数，根据递归参数决定
        with FSDP.summon_full_params(module_to_unshard, recurse=recurse):
            self.assertEqual(expected_outer_numel, outer_flat_param.numel())
            self.assertEqual(expected_inner_numel, inner_flat_param.numel())

    @skip_if_lt_x_gpu(2)
    def test_named_parameters_and_buffers(self):
        """
        Tests that ``named_parameters()`` and ``named_buffers()`` for a
        top-level FSDP-wrapped model matches their behavior for the equivalent
        non-wrapped module.
        """
        # 运行子测试，测试带有给定参数前缀和递归选项的`_test_named_parameters_and_buffers`方法
        self.run_subtests(
            {"prefix": ["", "test_prefix"], "recurse": [False, True]},
            self._test_named_parameters_and_buffers,
        )

    def _test_named_parameters_and_buffers(self, prefix: str, recurse: bool):
        # 初始化模型为带有嵌套包装的模块，使用给定的初始化模式和CUDA初始化模式
        model = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
            deterministic=True,
        )
        # 在模型中注册缓冲区
        model.register_buffer("buffer", torch.ones(1))
        
        # 使用FSDP包装顶层模型，因为在非FSDP根模块上调用`named_parameters()`和`named_buffers()`会包含FSDP前缀
        fsdp_model = FSDP(
            NestedWrappedModule.init(
                self.process_group,
                FSDPInitMode.NO_FSDP,
                CUDAInitMode.CUDA_BEFORE,
                deterministic=True,
            ),
            self.process_group,
        )
        # 在FSDP模型中注册缓冲区
        fsdp_model.register_buffer("buffer", torch.ones(1))
        
        # 使用FSDP全参数召唤上下文
        with FSDP.summon_full_params(fsdp_model):
            for call in ["named_parameters", "named_buffers"]:
                # 使用itertools.zip_longest逐一比较FSDP模型和非FSDP模型的命名参数和缓冲区
                for (n1, p1), (n2, p2) in itertools.zip_longest(
                    getattr(fsdp_model, call)(prefix=prefix, recurse=recurse),
                    getattr(model, call)(prefix=prefix, recurse=recurse),
                ):
                    # 断言FSDP模型和非FSDP模型的命名参数和缓冲区应该一致
                    self.assertEqual(n1, n2)
                    self.assertEqual(p1, p2)

    @skip_if_lt_x_gpu(2)
    def test_with_grads_core(self):
        """
        Tests the core usage of ``with_grads=True`` by comparing against DDP as
        the unsharded equivalent.
        """
        # 运行子测试，测试`_test_with_grads_core`方法，验证`with_grads=True`的核心用法并与DDP作为未分片等效进行比较
        self.run_subtests(
            {
                "writeback": [False, True],
                "offload_to_cpu": [False, True],
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.NO_SHARD,
                ],
                "use_orig_params": [True],
            },
            self._test_with_grads_core,
        )

    def _test_with_grads_core(
        self,
        writeback: bool,
        offload_to_cpu: bool,
        sharding_strategy: ShardingStrategy,
        use_orig_params: bool,
    ):
        # 此处省略了方法体，根据上下文和函数签名可推测该方法是测试`with_grads=True`的核心用法的实现
    # 定义测试函数，检查当所有排名的 FlatParameter 的梯度为 None 时，每个原始参数也看到 None 梯度。
    def test_with_grads_none_grads(self):
        """
        Tests that if all ranks' ``FlatParameter`` has ``None`` gradient, then
        each original parameter sees ``None`` gradient as well.
        """
        # 运行子测试，传入不同的分片策略
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.NO_SHARD,
                ]
            },
            self._test_with_grads_none_grads,  # 调用 _test_with_grads_none_grads 进行具体测试
        )

    # 具体测试函数，根据给定的分片策略初始化 FSDP 模型并进行断言验证
    def _test_with_grads_none_grads(self, sharding_strategy: ShardingStrategy):
        # 初始化带有共享参数的 Transformer 模型
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_BEFORE,
            deterministic=True,
            fsdp_kwargs={
                "use_orig_params": True,
                "sharding_strategy": sharding_strategy,
            },
        )
        # 遍历 FSDP 模型的所有 FSDP 模块
        for fsdp_module in FSDP.fsdp_modules(fsdp_model):
            # 如果模块有 _handle 属性
            if fsdp_module._handle:
                # 断言该模块的 flat_param 的梯度为 None
                assert fsdp_module._handle.flat_param.grad is None
        # 使用 FSDP.summon_full_params 函数召唤完整参数，并验证每个参数的梯度都为 None
        with FSDP.summon_full_params(fsdp_model, with_grads=True):
            for param in fsdp_model.parameters():
                self.assertTrue(param.grad is None)

    # 如果 GPU 数量小于 2，则跳过该测试
    @skip_if_lt_x_gpu(2)
    def test_unshard_submodule(self):
        # 创建包含两个子序列的序贯模型，并将其移到 GPU 上
        model = nn.Sequential(
            nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16)),
            nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16)),
        ).cuda()
        # 使用 FSDP 包装模型，并指定自动包装策略为 ModuleWrapPolicy((nn.Sequential,))
        model = FSDP(model, auto_wrap_policy=ModuleWrapPolicy((nn.Sequential,)))
        # 使用 FSDP.summon_full_params 函数召唤第一个子模块的完整参数
        with FSDP.summon_full_params(model[0]):
            # 检查召唤的模块不包含 flat parameter
            for param_name, param in model[0].named_parameters():
                self.assertFalse(FLAT_PARAM in param_name)
            # 断言模块的参数数量大于 1
            self.assertGreater(len(list(model[0].parameters())), 1)
# 定义 TestUnshardParamsNoShard 类，继承自 TestUnshardParamsBase 类
class TestUnshardParamsNoShard(TestUnshardParamsBase):

    # 返回整数值 1，表示世界大小为 1
    @property
    def world_size(self) -> int:
        return 1

    # 使用装饰器 skip_if_lt_x_gpu(1) 标记，条件为 GPU 数量小于 1 时跳过测试
    def test_unshard_params_writeback_no_shard(self):
        """Tests the ``writeback`` argument (using default for all others)."""
        # 运行子测试，使用 _get_test_unshard_params_writeback_config() 和 _test_unshard_params_writeback
        self.run_subtests(
            self._get_test_unshard_params_writeback_config(),
            self._test_unshard_params_writeback,
        )

    # 使用装饰器 skip_if_lt_x_gpu(1) 标记，条件为 GPU 数量小于 1 时跳过测试
    def test_unshard_params_param_data_no_shard(self):
        """
        Tests that parameters are exposed correctly for ``recurse=True`` and
        all other argument configs for a non-FSDP root module.
        """
        # 获取测试参数配置
        config = self._get_test_unshard_params_param_data_config()
        # 设置 offload_to_cpu=False，因为 `NO_SHARD` 情况下不支持 offload_to_cpu=True
        config["offload_to_cpu"] = [False]
        # 运行子测试，使用 config 和 _test_unshard_params_param_data
        self.run_subtests(
            config,
            self._test_unshard_params_param_data,
        )


# 定义 TestUnshardParamsErrors 类，继承自 TestUnshardParamsBase 类
class TestUnshardParamsErrors(TestUnshardParamsBase):

    # 返回整数值 2，表示世界大小为 2
    @property
    def world_size(self) -> int:
        return 2

    # 使用装饰器 skip_if_lt_x_gpu(2) 标记，条件为 GPU 数量小于 2 时跳过测试
    def test_unshard_params_from_forward_raises(self):
        # 定义 MyModule 类，继承自 nn.Module
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Parameter(torch.zeros(5))

            # 定义 forward 方法
            def forward(self, fsdp_module):
                # 使用 fsdp_module.summon_full_params(fsdp_module) 进行参数管理
                with fsdp_module.summon_full_params(fsdp_module):
                    pass

        # 创建 FSDP 包装后的 MyModule 模型，并移动到指定 GPU（self.rank）
        model = FSDP(MyModule()).cuda(self.rank)
        # 使用 assertRaisesRegex 断言捕获 AssertionError，验证在 forward/backward 期间无法手动取消分片参数
        with self.assertRaisesRegex(
            AssertionError, "Cannot manually unshard parameters during forward/backward"
        ):
            model(model)

    # 使用装饰器 skip_if_lt_x_gpu(2) 标记，条件为 GPU 数量小于 2 时跳过测试
    def test_unshard_params_from_backward_raises(self):
        # 创建 FSDP 包装后的 nn.Linear 模型，设备为 self.device
        model = FSDP(nn.Linear(2, 1, device=self.device))
        # 对模型进行前向传播
        output = model(torch.ones(2, device=self.device))

        # 定义无效的 backward hook 函数
        def invalid_backward_hook(*args, **kwargs):
            # 使用 FSDP.summon_full_params(model) 进行参数管理
            with FSDP.summon_full_params(model):
                pass

        # 确认输出需要梯度计算
        self.assertTrue(output.requires_grad)
        # 注册无效的 backward hook 函数
        output.register_hook(invalid_backward_hook)
        # 使用 assertRaisesRegex 断言捕获 AssertionError，验证在 forward/backward 期间无法手动取消分片参数
        with self.assertRaisesRegex(
            AssertionError, "Cannot manually unshard parameters during forward/backward"
        ):
            output.backward()

    # 使用装饰器 skip_if_lt_x_gpu(2) 标记，条件为 GPU 数量小于 2 时跳过测试
    def test_rank0_only_with_writeback_raises(self):
        # 初始化 NestedWrappedModule，使用给定的 self.process_group、FSDPInitMode.RECURSIVE 和 CUDAInitMode.CUDA_BEFORE
        nested_wrapped_module = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_BEFORE,
        )
        # 使用 assertRaisesRegex 断言捕获 NotImplementedError，验证 rank0_only=True 且 writeback=True 情况下的错误
        with self.assertRaisesRegex(NotImplementedError, "is not supported"):
            with FSDP.summon_full_params(
                nested_wrapped_module, rank0_only=True, writeback=True
            ):
                pass

    # 使用装饰器 skip_if_lt_x_gpu(2) 标记，条件为 GPU 数量小于 2 时跳过测试
    # 定义测试方法，验证在没有分片的情况下将计算转移到 CPU 时是否引发异常
    def test_offload_to_cpu_no_shard_raises(self):
        # 初始化一个嵌套封装模块，使用指定的初始化和 CUDA 配置，不使用分片策略
        nested_wrapped_module = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_BEFORE,
            {"sharding_strategy": ShardingStrategy.NO_SHARD},
        )
        # 使用断言验证在上下文中引发 NotImplementedError 异常，并确保异常消息包含特定文本
        with self.assertRaisesRegex(NotImplementedError, "is not supported"):
            # 在 FSDP 全部参数转换上下文中，仅在 rank0 执行并写回参数
            with FSDP.summon_full_params(
                nested_wrapped_module, rank0_only=True, writeback=True
            ):
                # 仅为了语法完整性而存在的占位符
                pass
# 如果这个脚本作为主程序运行（而不是被导入为模块），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```