# `.\pytorch\test\distributed\_composable\fsdp\test_fully_shard_comm.py`

```
# Owner(s): ["oncall: distributed"]

# 导入必要的模块和库
import copy  # 导入深拷贝函数
import functools  # 导入函数工具库，用于高阶函数操作
import itertools  # 导入迭代工具库，用于生成迭代器的函数
import unittest  # 导入单元测试框架
from typing import Callable, List, Optional, Tuple, Union  # 引入类型提示

import torch  # 导入PyTorch深度学习框架
import torch.distributed as dist  # 导入PyTorch分布式训练模块
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.nn.functional as F  # 导入PyTorch函数式接口

# 导入分布式训练中的可组合模块
from torch.distributed._composable import checkpoint, replicate
from torch.distributed._composable.fsdp import (
    FSDPModule,
    fully_shard,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
# 导入分布式训练中的收集操作相关函数
from torch.distributed._composable.fsdp._fsdp_collectives import (
    _div_if_needed,
    _get_gradient_divide_factors,
    foreach_all_gather,
    foreach_all_gather_copy_out,
    foreach_reduce,
)
# 导入分布式训练中的共享定义和状态相关模块
from torch.distributed._composable.fsdp._fsdp_common import FSDPMeshInfo, TrainingState
# 导入分布式训练中的初始化和网格信息相关函数
from torch.distributed._composable.fsdp._fsdp_init import (
    _get_post_forward_mesh_info,
    _init_default_fully_shard_mesh,
)
# 导入分布式训练中的参数共享状态相关类
from torch.distributed._composable.fsdp._fsdp_param import ShardedState
# 导入分布式训练中的参数组相关类
from torch.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup
# 导入分布式Tensor相关类
from torch.distributed._tensor import DTensor
# 导入通信调试模式相关类
from torch.distributed._tensor.debug.comm_mode import CommDebugMode
# 导入实验性复制相关模块
from torch.distributed._tensor.experimental import implicit_replication
# 导入设备网格初始化函数
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
# 导入CUDA测试相关标记
from torch.testing._internal.common_cuda import TEST_CUDA
# 导入分布式测试相关函数
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
# 导入FSDP测试相关类和函数
from torch.testing._internal.common_fsdp import (
    DoubleLinear,
    FSDPTest,
    FSDPTestMultiThread,
    MLP,
    patch_post_backward,
    patch_reshard,
    patch_unshard,
)
# 导入通用测试工具函数
from torch.testing._internal.common_utils import run_tests
# 导入分布式Tensor公共数据相关类
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    TransformerBlock,
)

# 设置C10D操作接口
c10d_ops = torch.ops.c10d

# 用于记录FSDP事件的类型，如取消共享或后向传播
EventType = Tuple[str, str, TrainingState]


class TestFullyShardCollectiveOps(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 128  # 返回测试的全局大小为128

    @property
    def device(self) -> torch.device:
        return torch.device("cuda:0")  # 返回使用的设备为cuda:0

    def _get_param_sizes(self) -> List[torch.Size]:
        # 返回一组参数大小列表，用于测试
        # 对于全局大小为128，fp32全收集和减少散点测试需要约0.22 GB
        return [
            torch.Size([17, 257]),
            torch.Size([17]),
            torch.Size([64, 312]),
            torch.Size([64]),
            torch.Size([64, 64]),
            torch.Size([512, 64]),
            torch.Size([256]),
            torch.Size([64, 297]),
        ]
    # 初始化参数函数，接受参数尺寸列表并返回参数列表
    def _init_params(self, param_sizes: List[torch.Size]) -> List[nn.Parameter]:
        # 设定随机种子为42
        torch.manual_seed(42)
        # 使用列表推导式创建原始参数列表，每个参数为一个随机生成的张量参数
        orig_params = [
            nn.Parameter(torch.randn(size, device=self.device)) for size in param_sizes
        ]
        # 由于种子是进程级别的，而不是线程级别的，因此需要广播以确保跨进程相同的原始参数
        for orig_param in orig_params:
            dist.broadcast(orig_param, src=0)
        # 返回原始参数列表
        return orig_params
    
    # 初始化FSDP参数组函数，接受参数列表、是否在每次前向传播后重新分片的标志
    def _init_fsdp_param_group(
        self, params: List[nn.Parameter], reshard_after_forward: Union[bool, int]
    ):
        # 创建参数列表的副本，每个参数是原始参数的分离和克隆
        module = nn.ParameterList([param.detach().clone() for param in params])
        # 创建默认的FSDP网格信息
        mesh_info = FSDPMeshInfo(_init_default_fully_shard_mesh(), shard_mesh_dim=0)
        # 获取前向传播后的网格信息
        post_forward_mesh_info = _get_post_forward_mesh_info(
            reshard_after_forward, mesh_info
        )
        # 创建FSDP参数组对象，包含模块参数、模块、网格信息、前向传播后网格信息、设备、混合精度策略、卸载策略
        fsdp_param_group = FSDPParamGroup(
            list(module.parameters()),
            module,
            mesh_info,
            post_forward_mesh_info,
            self.device,
            MixedPrecisionPolicy(),
            OffloadPolicy(),
        )
        # 执行延迟初始化
        fsdp_param_group.lazy_init()
        # 返回FSDP参数组对象
        return fsdp_param_group
    
    # 标记为单元测试，如果不支持CUDA，则跳过测试
    @unittest.skipIf(not TEST_CUDA, "no cuda")
    # 测试函数，测试所有gather的fp32情况
    def test_all_gather_fp32(self):
        # 获取参数尺寸列表
        param_sizes = self._get_param_sizes()
        # 获取默认CUDA流
        default_stream = torch.cuda.current_stream()
        # 创建两个CUDA流对象
        stream1, stream2 = torch.cuda.Stream(), torch.cuda.Stream()
        # 使用product函数生成异步操作、流和前向传播后是否重新分片的所有可能组合
        for async_op, streams, reshard_after_forward in itertools.product(
            (False, True),
            ((default_stream, default_stream), (stream1, stream2)),
            (True, 8),
        ):
            all_gather_copy_in_stream, all_gather_stream = streams
            # 在非异步操作或所有gather流为默认流时，仅测试前向传播后重新分片作为整数的情况，以节省测试时间
            if type(reshard_after_forward) is int and (
                async_op or all_gather_stream is default_stream
            ):
                continue
            # 调用内部函数，测试所有gather的功能
            self._test_all_gather(
                param_sizes,
                reshard_after_forward=reshard_after_forward,
                async_op=async_op,
                all_gather_copy_in_stream=all_gather_copy_in_stream,
                all_gather_stream=all_gather_stream,
            )
    
    # 内部测试函数，测试所有gather的功能
    def _test_all_gather(
        self,
        param_sizes: List[torch.Size],
        reshard_after_forward: Union[bool, int],
        async_op: bool,
        all_gather_copy_in_stream: torch.cuda.Stream,
        all_gather_stream: torch.cuda.Stream,
        ):
            # 定义一个函数 all_gather，用于处理参数组和进程组，执行参数的全局聚集操作
            def all_gather(fsdp_param_group: FSDPParamGroup, group: dist.ProcessGroup):
                # 执行 foreach_all_gather 函数，获取所有全局聚集的结果
                all_gather_result = foreach_all_gather(
                    fsdp_param_group.fsdp_params,
                    group,
                    async_op=async_op,
                    all_gather_copy_in_stream=all_gather_copy_in_stream,
                    all_gather_stream=all_gather_stream,
                    device=self.device,
                )
                # 将聚集结果复制回各个参数
                foreach_all_gather_copy_out(all_gather_result, fsdp_params, group)
                # 将参数状态转换为未分片状态，以注册未分片参数
                for fsdp_param in fsdp_param_group.fsdp_params:
                    fsdp_param.init_unsharded_param()
                fsdp_param_group._to_unsharded()

            # 定义一个函数 check_all_gathered_params，用于检查全局聚集后的参数正确性
            def check_all_gathered_params(
                orig_params: List[nn.Parameter], module: nn.Module
            ):
                # 检查每个参数是否为 torch.Tensor 类型
                for orig_param, param in zip(orig_params, module.parameters()):
                    self.assertIsInstance(param, torch.Tensor)
                    # 检查每个参数是否为 nn.Parameter 类型
                    self.assertIsInstance(param, nn.Parameter)
                    # 检查每个参数是否等于其对应原始参数的数据类型
                    self.assertEqual(param, orig_param.to(param.dtype))

            # 初始化参考参数并构建 FSDP 参数组
            orig_params = self._init_params(param_sizes)
            fsdp_param_group = self._init_fsdp_param_group(
                orig_params, reshard_after_forward
            )
            fsdp_params = fsdp_param_group.fsdp_params
            module = fsdp_param_group.module

            # 对参数分片情况进行基本检查
            for orig_param, param in zip(orig_params, module.parameters()):
                # 检查每个参数是否为 DTensor 类型
                self.assertTrue(isinstance(param, DTensor))
                # 检查每个参数的完整张量是否与原始参数相同
                self.assertEqual(param.full_tensor(), orig_param)

            # 运行 foreach 全局聚集操作（包括复制输入和输出）
            all_gather(fsdp_param_group, fsdp_param_group.mesh_info.shard_process_group)

            # 检查全局聚集的参数正确性
            check_all_gathered_params(orig_params, module)

            # 如果 reshard_after_forward 不是整数，则返回
            # 进行后向传播全局聚集前的测试仿真
            if type(reshard_after_forward) is not int:
                return
            fsdp_param_group._to_sharded_post_forward()
            # 运行全局聚集操作，用于后向传播后的分片
            all_gather(
                fsdp_param_group,
                fsdp_param_group.post_forward_mesh_info.shard_process_group,
            )
            # 再次检查全局聚集后的参数正确性
            check_all_gathered_params(orig_params, module)

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_reduce_scatter_fp32(self):
        # 获取参数大小
        param_sizes = self._get_param_sizes()
        # 获取默认 CUDA 流
        default_stream = torch.cuda.current_stream()
        # 创建新的 CUDA 流
        stream = torch.cuda.Stream()
        # 对每个 reduce_scatter 流执行测试
        for reduce_scatter_stream in (default_stream, stream):
            self._test_reduce_scatter(
                param_sizes,
                reduce_scatter_stream=reduce_scatter_stream,
                reduce_scatter_dtype=torch.float32,
            )

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    # 定义测试方法，用于测试 FP16 的 reduce_scatter 功能
    def test_reduce_scatter_fp16(self):
        # 获取参数大小列表
        param_sizes = self._get_param_sizes()
        # 获取当前 CUDA 默认流
        default_stream = torch.cuda.current_stream()
        # 创建新的 CUDA 流
        stream = torch.cuda.Stream()
        # 遍历默认流和新流两种情况
        for reduce_scatter_stream in (default_stream, stream):
            # 调用 _test_reduce_scatter 方法进行测试
            self._test_reduce_scatter(
                param_sizes,
                reduce_scatter_stream=reduce_scatter_stream,
                reduce_scatter_dtype=torch.float16,
            )

    # 定义私有方法，用于测试 reduce_scatter 功能
    def _test_reduce_scatter(
        self,
        param_sizes: List[torch.Size],
        reduce_scatter_stream: torch.cuda.Stream,
        reduce_scatter_dtype: torch.dtype,
    ):
        # 初始化原始参数并构建 FSDP 组
        orig_params = self._init_params(param_sizes)
        fsdp_param_group = self._init_fsdp_param_group(orig_params, True)
        fsdp_params = fsdp_param_group.fsdp_params
        # 懒初始化通信上下文
        fsdp_param_group.comm_ctx.lazy_init()

        # 运行一次 unshard 操作以初始化元数据
        fsdp_param_group.unshard()
        fsdp_param_group.wait_for_unshard()
        fsdp_param_group.reshard()

        # 运行 foreach reduce-scatter 操作（包括拷贝输入和视图输出）
        torch.manual_seed(42)
        # 创建未分片的梯度列表
        unsharded_grads = [torch.ones_like(param) * self.rank for param in orig_params]
        # 获取分片进程组
        group = fsdp_param_group.mesh_info.shard_process_group
        # 断言分片进程组的大小等于世界大小
        self.assertEqual(group.size(), self.world_size)
        # 创建 CUDA 流
        all_reduce_stream = torch.cuda.Stream()
        # 调用 foreach_reduce 函数执行 reduce-scatter 操作
        (
            reduce_scatter_input,
            reduce_scatter_event,
            post_reduce_event,
            _,
        ) = foreach_reduce(
            fsdp_params,
            unsharded_grads,
            group,
            reduce_scatter_stream,
            orig_dtype=orig_params[0].dtype,
            reduce_dtype=reduce_scatter_dtype,
            device=self.device,
            all_reduce_group=None,
            all_reduce_stream=all_reduce_stream,
            all_reduce_grads=True,
            partial_reduce_output=None,
        )
        # 等待事件完成
        torch.cuda.current_stream().wait_event(post_reduce_event)

        # 检查 reduce-scatter 的正确性
        # 获取梯度除法因子
        predivide_factor, postdivide_factor = _get_gradient_divide_factors(
            group, None, reduce_scatter_dtype
        )
        # 克隆减少后的梯度
        reduced_grads = [grad.detach().clone() for grad in unsharded_grads]
        # 对减少后的梯度进行处理
        for grad in reduced_grads:
            _div_if_needed(grad, predivide_factor)
            dist.all_reduce(
                grad,
                group=group,
                op=dist.ReduceOp.AVG if predivide_factor is None else dist.ReduceOp.SUM,
            )
            _div_if_needed(grad, postdivide_factor)
        # 检查每个 FSDP 参数和减少后的梯度
        for fsdp_param, reduced_grad in zip(fsdp_params, reduced_grads):
            sharded_grad = fsdp_param.sharded_param.grad
            # 断言分片梯度的类型为 DTensor
            self.assertIsInstance(sharded_grad, DTensor)
            # 断言分片梯度的完整张量与减少后的梯度相等
            self.assertEqual(sharded_grad.full_tensor(), reduced_grad)
class TestFullyShardCommunication(FSDPTest):
    # 定义一个测试类 TestFullyShardCommunication，继承自 FSDPTest

    @property
    def world_size(self) -> int:
        # 定义一个属性方法 world_size，返回当前 CUDA 设备数量和 4 的较小值
        return min(4, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    def test_fully_shard_communication_count(self):
        """
        Tests that FSDP issues the expected number of all-gathers and
        reduce-scatters during forward and backward.
        """
        # 测试 FSDP 在前向和后向传播期间执行预期数量的 all-gathers 和 reduce-scatters
        self.run_subtests(
            {"reshard_after_forward": [True, False, 2]},
            self._test_communication_count,
        )

    def _test_communication_count(
        self,
        reshard_after_forward: Union[bool, int],
    ):
        # 定义一个测试方法 _test_communication_count，接受参数 reshard_after_forward
        torch.manual_seed(42)
        # 设置随机种子为 42
        model_args = ModelArgs()
        # 创建模型参数对象 model_args
        model = Transformer(model_args)
        # 创建 Transformer 模型对象 model
        fully_shard_fn = functools.partial(
            fully_shard, reshard_after_forward=reshard_after_forward
        )
        # 创建部分函数 fully_shard_fn，使用 fully_shard 函数和 reshard_after_forward 参数
        num_blocks = 0
        # 初始化块数为 0
        for module in model.modules():
            # 遍历模型的所有模块
            if isinstance(module, TransformerBlock):
                # 如果模块是 TransformerBlock 类型
                fully_shard_fn(module)
                # 对该模块执行 fully_shard_fn 函数
                num_blocks += 1
                # 块数加一
        fully_shard_fn(model)
        # 对整个模型 model 执行 fully_shard_fn 函数
        # We construct `num_blocks` plus 1 FSDP states/communication groups
        # 我们构造 `num_blocks` 加 1 个 FSDP 状态/通信组

        torch.manual_seed(42 + self.rank)
        # 设置随机种子为 42 加上当前进程的 rank
        inp = torch.randint(0, model_args.vocab_size, (2, 16), device="cuda")
        # 生成一个大小为 (2, 16)、值在 [0, model_args.vocab_size) 范围内的整数张量 inp，放置在 CUDA 设备上
        with CommDebugMode() as fwd_comm_mode:
            # 使用 CommDebugMode 上下文管理器，命名为 fwd_comm_mode
            loss = model(inp)
            # 将输入 inp 输入模型并计算损失
        fwd_comm_counts = fwd_comm_mode.get_comm_counts()
        # 获取前向传播时通信计数
        self.assertEqual(len(fwd_comm_counts), 1)
        # 断言前向传播通信计数的长度为 1
        self.assertEqual(fwd_comm_counts[c10d_ops._allgather_base_], num_blocks + 1)
        # 断言前向传播中的 allgather 操作计数等于 num_blocks + 1
        with CommDebugMode() as bwd_comm_mode:
            # 使用 CommDebugMode 上下文管理器，命名为 bwd_comm_mode
            loss.sum().backward()
            # 计算损失的梯度并反向传播
        bwd_comm_counts = bwd_comm_mode.get_comm_counts()
        # 获取后向传播时通信计数
        if reshard_after_forward is False:
            # 如果 reshard_after_forward 为 False
            self.assertEqual(len(bwd_comm_counts), 1)
            # 断言后向传播通信计数的长度为 1
        else:
            # 否则
            # The root always does not reshard after forward
            # 根节点在前向传播后始终不进行重分片
            self.assertEqual(len(bwd_comm_counts), 2)
            # 断言后向传播通信计数的长度为 2
            self.assertEqual(bwd_comm_counts[c10d_ops._allgather_base_], num_blocks)
            # 断言后向传播中的 allgather 操作计数等于 num_blocks
        self.assertEqual(
            bwd_comm_counts[c10d_ops._reduce_scatter_base_], num_blocks + 1
        )
        # 断言后向传播中的 reduce-scatter 操作计数等于 num_blocks + 1

    @skip_if_lt_x_gpu(2)
    # 定义测试函数，用于测试在设置 ``reshard_after_forward=False`` 的 FSDP 模块上手动调用 ``reshard`` 功能
    def test_manual_reshard_with_reshard_after_forward_false(self):
        """
        Tests that we can manually call ``reshard`` on FSDP modules that were
        initialized with ``reshard_after_forward=False`` and still run unshard.
        """
        # 设定随机种子为 42
        torch.manual_seed(42)
        # 创建模型参数对象
        model_args = ModelArgs()
        # 创建 Transformer 模型
        model = Transformer(model_args)
        # 遍历模型中的每个模块
        for module in model.modules():
            # 如果模块是 TransformerBlock 类型
            if isinstance(module, TransformerBlock):
                # 对该模块执行完全分片，但不在每次前向传播后重新分片
                fully_shard(module, reshard_after_forward=False)
        # 对整个模型执行完全分片，但不在每次前向传播后重新分片
        model = fully_shard(model, reshard_after_forward=False)
        # 计算模型中的 FSDP 模块数量
        num_fsdp_modules = sum(
            isinstance(module, FSDPModule) for module in model.modules()
        )

        # 设置随机种子为 42 加上当前进程的排名
        torch.manual_seed(42 + self.rank)
        # 在 GPU 上生成输入张量，形状为 (2, 16)，数值从 0 到 model_args.vocab_size 随机选择
        inp = torch.randint(0, model_args.vocab_size, (2, 16), device="cuda")
        # 进入通信调试模式
        with CommDebugMode() as fwd_comm_mode:
            # 使用模型进行前向传播，计算损失
            loss = model(inp)
        # 获取前向通信计数
        fwd_comm_counts = fwd_comm_mode.get_comm_counts()
        # 断言仅有一次通信事件
        self.assertEqual(len(fwd_comm_counts), 1)
        # 断言此通信事件是由 num_fsdp_modules 数量的 FSDP 模块进行的全收集操作
        self.assertEqual(fwd_comm_counts[c10d_ops._allgather_base_], num_fsdp_modules)

        # 遍历模型中的每个模块
        for module in model.modules():
            # 如果模块是 FSDPModule 类型
            if isinstance(module, FSDPModule):
                # 执行模块的重新分片操作
                module.reshard()

        # 进入通信调试模式
        with CommDebugMode() as bwd_comm_mode:
            # 对损失进行求和并执行反向传播
            loss.sum().backward()
        # 获取反向传播中的通信计数
        bwd_comm_counts = bwd_comm_mode.get_comm_counts()
        # 断言有两次通信事件
        self.assertEqual(len(bwd_comm_counts), 2)
        # 断言这两次通信事件都是由 num_fsdp_modules 数量的 FSDP 模块执行的全收集和归约分散操作
        self.assertEqual(bwd_comm_counts[c10d_ops._allgather_base_], num_fsdp_modules)
        self.assertEqual(
            bwd_comm_counts[c10d_ops._reduce_scatter_base_], num_fsdp_modules
        )
# 定义 TestFullyShardPrefetch 类，继承自 FSDPTest 类
class TestFullyShardPrefetch(FSDPTest):

    # 定义 world_size 属性，返回最小值为 4 和当前 CUDA 设备数量的较小者
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    # 如果 GPU 数量小于 2，跳过测试
    @skip_if_lt_x_gpu(2)
    def test_fully_shard_backward_prefetch(self):
        # 激活检查点应不影响预期的 FSDP 事件

        # 运行子测试，设置 "reshard_after_forward" 和 "checkpoint_impl" 参数，调用 _test_backward_prefetch_forward_backward 方法
        self.run_subtests(
            {
                "reshard_after_forward": [True, False, 2],
                "checkpoint_impl": [None, "utils", "composable"],
            },
            self._test_backward_prefetch_forward_backward,
        )

        # 再次运行子测试，设置 "reshard_after_forward" 和 "checkpoint_impl" 参数，调用 _test_backward_prefetch_multi_forward 方法
        self.run_subtests(
            {
                "reshard_after_forward": [True, False, 2],
                "checkpoint_impl": [None, "utils", "composable"],
            },
            self._test_backward_prefetch_multi_forward,
        )

        # 调用 _test_backward_prefetch_unused_in_backward 方法，设置参数为 True
        self._test_backward_prefetch_unused_in_backward(True)

    # 定义 _test_backward_prefetch_forward_backward 方法，参数包括 reshard_after_forward 和 checkpoint_impl
    def _test_backward_prefetch_forward_backward(
        self, reshard_after_forward: Union[bool, int], checkpoint_impl: Optional[str]
    ):
        # 设置层数为3
        n_layers = 3
        # 初始化 transformer 模型、优化器和输入
        model, optim, inp = self._init_transformer(
            n_layers, reshard_after_forward, checkpoint_impl
        )
        # 创建事件列表，用于记录事件
        events: List[EventType] = []
        # 获取未分片记录的参数组
        unshard_with_record = self._get_unshard_with_record(
            FSDPParamGroup.unshard, events
        )
        # 获取后向传播记录的参数组
        post_backward_with_record = self._get_post_backward_with_record(
            FSDPParamGroup.post_backward, events
        )
        # 检查正常情况下的顺序，即1次前向传播，1次后向传播，1次优化器步骤
        with patch_unshard(unshard_with_record), patch_post_backward(
            post_backward_with_record
        ):
            # 迭代3次
            for iter_idx in range(3):
                # 计算模型输出的损失
                loss = model(inp)
                # 期望的事件顺序
                expected_events = [
                    ("unshard", "", TrainingState.FORWARD),  # 根节点
                    ("unshard", "layers.0", TrainingState.FORWARD),
                    ("unshard", "layers.1", TrainingState.FORWARD),
                    ("unshard", "layers.2", TrainingState.FORWARD),
                ]
                # 断言事件列表与期望事件列表相等
                self.assertEqual(events, expected_events)
                # 清空事件列表
                events.clear()
                # 计算损失的梯度
                loss.sum().backward()
                # 期望的事件顺序（包括后向传播的事件）
                expected_events = [
                    # 根节点在前向传播后不会再进行分片，因此后向传播中没有未分片事件
                    ("unshard", "layers.2", TrainingState.PRE_BACKWARD),
                    # 显式的后向传播预取将未分片提前一个模块
                    ("unshard", "layers.1", TrainingState.PRE_BACKWARD),
                    ("post_backward", "layers.2", TrainingState.POST_BACKWARD),
                    ("unshard", "layers.0", TrainingState.PRE_BACKWARD),
                    ("post_backward", "layers.1", TrainingState.POST_BACKWARD),
                    ("post_backward", "layers.0", TrainingState.POST_BACKWARD),
                    ("post_backward", "", TrainingState.POST_BACKWARD),
                ]
                # 如果在前向传播后不进行分片，则不需要后向传播的未分片事件
                if reshard_after_forward is False:
                    expected_events = [e for e in expected_events if e[0] != "unshard"]
                # 断言事件列表与期望事件列表相等
                self.assertEqual(events, expected_events)
                # 清空事件列表
                events.clear()
                # 执行优化器的一步优化
                optim.step()
                # 每隔一次迭代将梯度置零（设置为None）
                optim.zero_grad(set_to_none=(iter_idx % 2 == 0))

    def _test_backward_prefetch_multi_forward(
        self, reshard_after_forward: Union[bool, int], checkpoint_impl: Optional[str]
    ):
        # 这是一个测试函数，用于多次前向传播后的后向传播预取
        pass

    def _test_backward_prefetch_unused_in_backward(
        self, reshard_after_forward: Union[bool, int]
    ):
        # 这是一个测试函数，用于在后向传播中未使用的后向传播预取
        pass
    ):
        """
        Test a model with a linear module then a split into two linear modules,
        where we run backward through one path first before the other, meaning
        that (1) only one linear of the two split is used per backward and (2)
        the initial shared linear is used in both backwards.
        """
        dim = 8
        model = nn.Sequential(nn.Linear(dim, dim), DoubleLinear(dim))
        fully_shard(model[0], reshard_after_forward=reshard_after_forward)
        fully_shard(model[1].lin1, reshard_after_forward=reshard_after_forward)
        fully_shard(model[1].lin2, reshard_after_forward=reshard_after_forward)
        fully_shard(model, reshard_after_forward=reshard_after_forward)
        inp = torch.randn((4, dim), device="cuda")
        events: List[EventType] = []
        unshard_with_record = self._get_unshard_with_record(
            FSDPParamGroup.unshard, events
        )
        post_backward_with_record = self._get_post_backward_with_record(
            FSDPParamGroup.post_backward, events
        )
        with patch_unshard(unshard_with_record), patch_post_backward(
            post_backward_with_record
        ):
            # Forward pass through the model
            loss1, loss2 = model(inp)
            expected_events = [
                # Root has no parameters, so it does not have an unshard
                ("unshard", "0", TrainingState.FORWARD),
                ("unshard", "1.lin1", TrainingState.FORWARD),
                ("unshard", "1.lin2", TrainingState.FORWARD),
            ]
            # Verify expected events during forward pass
            self.assertEqual(events, expected_events)
            events.clear()

            # Prepare for backward pass with unsharding
            model.set_is_last_backward(False)
            # Backward pass through the second linear module
            loss2.sum().backward(retain_graph=True)
            expected_events = [
                ("unshard", "1.lin2", TrainingState.PRE_BACKWARD),
                # NOTE: This `1.lin1` unshard is a mistargeted prefetch.
                ("unshard", "1.lin1", TrainingState.PRE_BACKWARD),
                ("post_backward", "1.lin2", TrainingState.POST_BACKWARD),
                ("unshard", "0", TrainingState.PRE_BACKWARD),
                ("post_backward", "0", TrainingState.POST_BACKWARD),
            ]
            # Verify expected events during backward pass for loss2
            self.assertEqual(events, expected_events)
            events.clear()

            # Prepare for backward pass with unsharding
            model.set_is_last_backward(True)
            # Backward pass through the first linear module
            loss1.sum().backward()
            expected_events = [
                # NOTE: `1.lin1` is already unsharded from the mistargeted
                # prefetch in the first backward.
                # Prefetch `0`
                ("unshard", "0", TrainingState.PRE_BACKWARD),
                ("post_backward", "1.lin1", TrainingState.POST_BACKWARD),
                ("post_backward", "0", TrainingState.POST_BACKWARD),
            ]
            # Verify expected events during backward pass for loss1
            self.assertEqual(events, expected_events)
            events.clear()
    # 初始化一个 Transformer 模型，并根据参数配置是否使用检查点激活
    def _init_transformer(
        self,
        n_layers: int,
        reshard_after_forward: Union[bool, int],
        checkpoint_impl: Optional[str],
    ):
        # 创建模型参数对象
        model_args = ModelArgs(
            n_layers=n_layers, checkpoint_activations=(checkpoint_impl == "utils")
        )
        # 根据模型参数创建 Transformer 模型
        model = Transformer(model_args)
        
        # 遍历模型的各个模块
        for module in model.modules():
            # 如果是 TransformerBlock 模块
            if isinstance(module, TransformerBlock):
                # 如果配置为使用可组合检查点实现，则对该模块进行检查点
                if checkpoint_impl == "composable":
                    checkpoint(module)
                # 根据参数对模块进行全面分片操作
                fully_shard(module, reshard_after_forward=reshard_after_forward)
        
        # 对整个模型进行全面分片操作
        fully_shard(model, reshard_after_forward=reshard_after_forward)
        
        # 使用 Adam 优化器，学习率设为 0.01，优化模型的参数
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        
        # 在 CUDA 设备上生成随机整数输入张量，用于模型输入
        inp = torch.randint(
            0, model_args.vocab_size, (2, model_args.max_seq_len), device="cuda"
        )
        
        # 返回初始化好的模型、优化器和输入张量
        return model, optim, inp

    # 返回一个带有记录功能的 unshard 函数
    def _get_unshard_with_record(
        self, orig_unshard: Callable, events: List[EventType]
    ) -> Callable:
        def unshard_with_record(self, *args, **kwargs):
            nonlocal events
            # 如果未执行全局收集操作并且模块状态不是 UNSHARDED，则记录 unshard 事件
            if (
                self._all_gather_result is None
                and self._sharded_state != ShardedState.UNSHARDED
            ):  # skip no-ops
                events.append(("unshard", self._module_fqn, self._training_state))
            return orig_unshard(self, *args, **kwargs)

        return unshard_with_record

    # 返回一个带有记录功能的 reshard 函数
    def _get_reshard_with_record(
        self, orig_reshard: Callable, events: List[EventType]
    ) -> Callable:
        def reshard_with_record(self, *args, **kwargs):
            nonlocal events
            # 如果训练状态为 FORWARD 并且不是在 FORWARD 后重新分片，则跳过无操作
            if (
                self._training_state == TrainingState.FORWARD
                and not self._reshard_after_forward
            ):  # skip no-ops
                return
            # 记录 reshard 事件，包括模块的完全限定名和当前训练状态
            events.append(("reshard", self._module_fqn, self._training_state))
            return orig_reshard(self, *args, **kwargs)

        return reshard_with_record

    # 返回一个带有记录功能的 post_backward 函数
    def _get_post_backward_with_record(
        self, orig_post_backward: Callable, events: List[EventType]
    ) -> Callable:
        def post_backward_with_record(self, *args, **kwargs):
            nonlocal events
            # 调用原始的 post_backward 函数，并获取其返回值
            ret = orig_post_backward(self, *args, **kwargs)
            # 在运行 post_backward 后，使用训练状态来检查状态是否按预期转换为 POST_BACKWARD
            events.append(("post_backward", self._module_fqn, self._training_state))
            return ret

        return post_backward_with_record
class TestFullyShardUnshardMultiProcess(FSDPTest):
    @property
    def world_size(self) -> int:
        # 返回 GPU 数量和 2 中的较小值作为进程数
        return min(torch.cuda.device_count(), 2)

    @skip_if_lt_x_gpu(2)
class TestFullyShardUnshardMultiThread(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        # 设置进程数为 2
        return 2

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_unshard_no_param_group(self):
        # 检查对于没有参数组/未管理参数的模块，是否可以调用 `unshard()` 而不出错
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        for lin in model:
            fully_shard(lin)  # 对每一层进行分片操作
        fully_shard(model)  # 对整个模型进行分片操作
        handle = model.unshard(async_op=True)  # 异步操作，返回处理句柄
        handle.wait()  # 等待处理完成

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_unshard_without_lazy_init(self):
        torch.manual_seed(42)
        model = MLP(4)
        # 将模型参数广播到所有设备上
        for param in model.parameters():
            dist.broadcast(param, src=0)
        ref_model = copy.deepcopy(model)
        fully_shard(model)  # 对模型进行分片操作
        model.unshard()  # 执行非延迟初始化的反分片操作
        # 检查反分片后的参数与参考模型的参数是否相等
        for ref_param, param in zip(ref_model.parameters(), model.parameters()):
            self.assertEqual(ref_param, param)


if __name__ == "__main__":
    run_tests()
```