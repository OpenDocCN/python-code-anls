# `.\pytorch\test\distributed\_spmd\test_tracing.py`

```
# Owner(s): ["oncall: distributed"]

from copy import deepcopy  # 导入深拷贝函数
from functools import wraps  # 导入装饰器函数
from typing import Any, List, Type  # 引入类型提示相关的模块

import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入分布式通信模块
import torch.distributed._functional_collectives as funcol  # 导入分布式函数式集合操作
import torch.fx as fx  # 导入PyTorch FX库
import torch.nn as nn  # 导入神经网络模块
from torch.distributed._spmd.api import compile, COMPILED_OBJECT_KEY, Override  # 导入分布式SPMD相关的模块和API
from torch.distributed._spmd.comm_tensor import CommTensor  # 导入分布式通信张量模块
from torch.distributed._tensor import DeviceMesh  # 导入分布式张量网格模块
from torch.distributed._tensor._op_schema import OpSchema, OutputSharding  # 导入张量操作模式和输出分片模块
from torch.distributed._tensor.ops.utils import register_prop_rule  # 导入张量操作的属性规则注册函数
from torch.distributed._tensor.placement_types import DTensorSpec  # 导入张量放置类型规范模块
from torch.distributed.distributed_c10d import get_global_rank, get_world_size  # 导入全局排名和全局世界大小模块
from torch.fx.experimental.proxy_tensor import make_fx  # 导入FX代理张量生成函数
from torch.nn import functional as F  # 导入神经网络的函数模块
from torch.nn.parallel import DistributedDataParallel as DDP  # 导入分布式数据并行模块作为DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入测试时用于跳过GPU数量不足的装饰器函数
from torch.testing._internal.common_utils import run_tests  # noqa: TCH001 导入用于运行测试的函数
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms as base_with_comms,
)


def with_comms(func):
    @base_with_comms
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # make sure we set different random seeds for each rank
        # otherwise we dont need DDP / SPMD
        # (we would have the same parameters and inputs everywhere)
        torch.manual_seed(self.rank)  # 设置每个排名的不同随机种子
        return func(self, *args, **kwargs)

    return wrapper


class TraceDeviceMeshTestBase:
    def _test_tracing_all_reduce_nd(self, mesh_tensor):
        mesh = DeviceMesh(self.device_type, mesh_tensor)  # 创建设备网格对象
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank  # 创建本地张量，元素值为当前排名

        # 检查所有维度组
        dim_to_subgroups = mesh.get_all_groups()  # 获取所有维度的子组
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)  # 获取维度组的世界大小
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]  # 获取全局排名列表

            def fn(tensor: torch.Tensor):
                tensor = funcol.all_reduce(tensor, "sum", group=(mesh, dim))  # 对张量执行全局归约操作
                # multiply with 1 to trigger wait on read during tracing.
                return tensor * 1  # 乘以1来触发跟踪期间的读取等待

            # 使用 local_tensor + 1 来进行跟踪，以确保不仅仅重播记录的张量值
            traced_fn = make_fx(fn)(local_tensor + 1)

            # 执行跟踪后的 DeviceMesh 通信
            reduced_tensor = traced_fn(local_tensor.clone())
            res_num = sum(global_ranks)
            self.assertEqual(reduced_tensor, torch.ones(3, 3) * res_num)  # 断言检查归约后的张量是否符合预期
    def _test_broadcast_nd(self, mesh_tensor):
        # 创建一个 DeviceMesh 对象，用给定的设备类型和网格张量初始化
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        # 获取维度到子组的映射
        dim_to_subgroups = mesh.get_group()
        # 遍历所有维度组
        for dim, dim_group in enumerate(dim_to_subgroups):
            # 获取当前维度组的大小
            dim_group_size = get_world_size(dim_group)
            # 获取当前维度组的全局排名列表
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]

            # 定义一个函数 fn，用于广播张量
            def fn(tensor: torch.Tensor):
                # 克隆输入张量并封装成 CommTensor 对象
                received_tensor = CommTensor(tensor.clone())
                # 在网格上进行广播操作
                mesh.broadcast(received_tensor, mesh_dim=dim)
                # 乘以1以触发在追踪期间的读取等待
                return received_tensor * 1

            # 创建一个本地张量，用于测试，形状为 (3, 3)，设备类型与当前对象一致
            local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank
            # 在追踪期间使用 local_tensor + 1，以确保不仅简单重放记录的张量值
            traced_fn = make_fx(fn)(local_tensor + 1)

            # 执行追踪后的 DeviceMesh 通信操作
            received_tensor = traced_fn(local_tensor)
            # 获取预期结果值的全局排名
            res_num = global_ranks[0]
            # 断言接收到的张量与预期的全是1的张量乘以全局排名相等
            self.assertEqual(received_tensor, torch.ones(3, 3) * res_num)

    def _test_scatter_nd(self, mesh_tensor):
        # 创建一个 DeviceMesh 对象，用给定的设备类型和网格张量初始化
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        # 获取维度到子组的映射
        dim_to_subgroups = mesh.get_group()
        # 遍历所有维度组
        for dim, dim_group in enumerate(dim_to_subgroups):
            # 获取当前维度组的大小
            dim_group_size = get_world_size(dim_group)
            # 获取当前维度组的全局排名列表
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            # 根据全局排名创建分散的张量列表
            scattered_tensors = [
                torch.ones(3, 3, device=self.device_type) * global_rank
                for global_rank in global_ranks
            ]

            # 定义一个函数 fn，用于执行 scatter 操作
            def fn(to_receive: torch.Tensor, to_scatter: List[torch.Tensor]):
                # 将输入的张量列表转换成 CommTensor 对象列表
                to_scatter = [CommTensor(t) for t in to_scatter]
                # 将接收张量转换成 CommTensor 对象
                to_receive = CommTensor(to_receive)
                # 在网格上执行 scatter 操作
                mesh.scatter(to_receive, to_scatter, mesh_dim=dim)
                # 乘以1以触发在追踪期间的读取等待
                return to_receive * 1

            # 创建一个空张量用于接收 scatter 操作的结果，形状与 scattered_tensors[dim] 相同
            to_receive = torch.empty_like(scattered_tensors[mesh.get_coordinate()[dim]])
            # 在追踪期间使用 scattered_tensors 加 1，以确保不仅简单重放记录的张量值
            traced_fn = make_fx(fn)(to_receive, [t + 1 for t in scattered_tensors])

            # 执行追踪后的 scatter 操作
            received_tensor = traced_fn(to_receive, scattered_tensors)
            # 断言接收到的张量与全是1的张量乘以 self.rank 相等
            self.assertEqual(received_tensor, torch.ones(3, 3) * self.rank)
    # 定义一个测试函数，用于测试在给定的 mesh tensor 上进行 all_gather_nd 操作
    def _test_all_gather_nd(self, mesh_tensor):
        # 创建一个 DeviceMesh 对象，使用给定的设备类型和 mesh_tensor
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        
        # 创建一个本地的 tensor，每个进程有自己的 tensor，全体聚合后得到一个大的 tensor
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank

        # 获取维度到子组的映射字典
        dim_to_subgroups = mesh.get_group()
        
        # 遍历维度和对应的子组
        for dim, dim_group in enumerate(dim_to_subgroups):
            # 获取当前子组的大小（进程数）
            dim_group_size = get_world_size(dim_group)
            
            # 获取全局 ranks 列表，表示当前子组内各个进程的全局 rank
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]

            # 定义一个接受 tensor 作为输入的函数 fn
            def fn(tensor: torch.Tensor):
                # 调用 funcol.all_gather_tensor 函数，对 tensor 在 gather_dim=0 的维度上进行全局聚合
                # group 参数指定使用的 mesh 和当前维度 dim_group
                big_tensor = funcol.all_gather_tensor(
                    tensor, gather_dim=0, group=(mesh, dim)
                )
                # 将聚合后的大 tensor 切分成列表，每个元素代表一个全局聚合后的 tensor
                return list(torch.chunk(big_tensor, dim_group_size))

            # 使用 local_tensor + 1 进行追踪，以确保我们不仅仅是重放记录的 tensor 值
            traced_fn = make_fx(fn)(local_tensor + 1)
            
            # 调用追踪后的函数 traced_fn，传入 local_tensor，得到全局聚合后的列表 gathered_list
            gathered_list = traced_fn(local_tensor)

            # 断言 gathered_list 的长度应该等于当前子组的大小
            self.assertEqual(len(gathered_list), dim_group_size)
            
            # 遍历 gathered_list 中的每个元素 gathered_tensor 和对应的 global_ranks 进行比较
            for idx, gathered_tensor in enumerate(gathered_list):
                self.assertEqual(gathered_tensor, torch.ones(3, 3) * global_ranks[idx])
class TraceDeviceMesh3DTest(DTensorTestBase, TraceDeviceMeshTestBase):
    # 定义一个测试类，继承自DTensorTestBase和TraceDeviceMeshTestBase
    @property
    def world_size(self):
        # 返回世界大小为8
        return 8

    @with_comms
    def test_tracing_all_reduce_nd(self):
        # 调用测试方法 _test_tracing_all_reduce_nd，传入一个形状为 (2, 2, 2) 的张量进行测试
        self._test_tracing_all_reduce_nd(torch.arange(8).reshape(2, 2, 2))

    @with_comms
    def test_broadcast_nd(self):
        # 调用测试方法 _test_broadcast_nd，传入一个形状为 (2, 2, 2) 的张量进行测试
        self._test_broadcast_nd(torch.arange(8).reshape(2, 2, 2))

    @with_comms
    def test_scatter_nd(self):
        # 调用测试方法 _test_scatter_nd，传入一个形状为 (2, 2, 2) 的张量进行测试
        self._test_scatter_nd(torch.arange(8).reshape(2, 2, 2))

    @with_comms
    def test_all_gather_nd(self):
        # 调用测试方法 _test_all_gather_nd，传入一个形状为 (2, 2, 2) 的张量进行测试
        self._test_all_gather_nd(torch.arange(8).reshape(2, 2, 2))


class TraceDeviceMesh2DTest(DTensorTestBase, TraceDeviceMeshTestBase):
    # 定义一个测试类，继承自DTensorTestBase和TraceDeviceMeshTestBase
    @property
    def world_size(self):
        # 返回世界大小为4
        return 4

    @with_comms
    def test_tracing_all_reduce_nd(self):
        # 调用测试方法 _test_tracing_all_reduce_nd，传入一个形状为 (2, 2) 的张量进行测试
        self._test_tracing_all_reduce_nd(torch.arange(4).reshape(2, 2))

    @with_comms
    def test_broadcast_nd(self):
        # 调用测试方法 _test_broadcast_nd，传入一个形状为 (2, 2) 的张量进行测试
        self._test_broadcast_nd(torch.arange(4).reshape(2, 2))

    @with_comms
    def test_scatter_nd(self):
        # 调用测试方法 _test_scatter_nd，传入一个形状为 (2, 2) 的张量进行测试
        self._test_scatter_nd(torch.arange(4).reshape(2, 2))

    @with_comms
    def test_all_gather_nd(self):
        # 调用测试方法 _test_all_gather_nd，传入一个形状为 (2, 2) 的张量进行测试
        self._test_all_gather_nd(torch.arange(4).reshape(2, 2))


class DataDependentModule(nn.Module):
    # 定义一个继承自nn.Module的数据相关模块类
    def __init__(self, world_size):
        super().__init__()
        self.world_size = world_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 抛出运行时错误，提示此实现不应被执行，用于解释如何规避数据相关的用户定义模块
        raise RuntimeError(
            "This eager implementation shouldn't be executed."
            "This implementation is just an example of how to get around "
            "data-dependant user-defined modules. "
        )
        shape = x.shape
        x = x.view(-1)
        positive = x[x >= 0]
        negative = x[x < 0]

        in_sizes = torch.tensor([positive.numel(), negative.numel()], dtype=torch.int32)
        out_sizes = torch.empty_like(in_sizes)
        dist.all_to_all_single(
            out_sizes,
            in_sizes,
            output_split_sizes=[1, 1],
            input_split_sizes=[1, 1],
        )

        xs = [positive, negative]
        ys = [torch.Tensor(out_sizes[i].item()) for i in range(out_sizes.numel())]
        dist.all_to_all(ys, xs)

        # some dummy compute
        for y in ys:
            y.add_(1)

        dist.all_to_all(xs, ys)

        return torch.cat(xs).reshape(shape)


class DummyModel(nn.Module):
    # 定义一个简单的模型类，继承自nn.Module
    def __init__(self, world_size):
        super().__init__()
        self.l1 = nn.Linear(10, 10)
        self.ddm = DataDependentModule(world_size)
        self.l2 = nn.Linear(10, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        assert len(x.size()) == 2

        return self.relu(self.l2(self.ddm(self.l1(x))))


def ddm(x: torch.Tensor) -> torch.Tensor:
    # 定义一个函数 ddm，接受一个张量并返回相同的张量
    return x


def ddm_backward(grad: torch.Tensor) -> torch.Tensor:
    # 定义一个函数 ddm_backward，接受一个梯度张量并返回相同的梯度张量
    return grad


dummy_lib = torch.library.Library("dummy", "DEF")  # noqa: TOR901
dummy_lib.define("ddm(Tensor x) -> Tensor")
dummy_lib.impl("ddm", ddm, "CompositeExplicitAutograd")
dummy_lib.define("ddm_backward(Tensor x) -> Tensor")
dummy_lib.impl("ddm_backward", ddm_backward, "CompositeExplicitAutograd")

# 定义了一个函数映射规则，接受一个名为 op_schema 的 OpSchema 对象作为参数，并返回一个 OutputSharding 对象
def _identity_prop_rule(op_schema: OpSchema) -> OutputSharding:
    # 从 op_schema 中获取参数列表，并确保第一个参数 x 是 DTensorSpec 类型
    (x,) = op_schema.args_schema
    assert isinstance(x, DTensorSpec), f"expecting DTensorSpec but got {x}"
    
    # 返回一个具有与输入 x 相同 mesh 和 placements 的 OutputSharding 对象
    return OutputSharding(output_spec=DTensorSpec(x.mesh, x.placements))

# 注册 torch.ops.dummy.ddm.default 的属性规则，接受一个 OpSchema 对象作为参数，并返回一个 OutputSharding 对象
@register_prop_rule(torch.ops.dummy.ddm.default)
def _prop_ddm(op_schema: OpSchema) -> OutputSharding:
    # 调用 _identity_prop_rule 函数，返回处理后的 OutputSharding 对象
    return _identity_prop_rule(op_schema)

# 注册 torch.ops.dummy.ddm_backward.default 的属性规则，接受一个 OpSchema 对象作为参数，并返回一个 OutputSharding 对象
@register_prop_rule(torch.ops.dummy.ddm_backward.default)
def _prop_ddm_backward(op_schema: OpSchema) -> OutputSharding:
    # 调用 _identity_prop_rule 函数，返回处理后的 OutputSharding 对象
    return _identity_prop_rule(op_schema)

# 定义一个继承自 torch.autograd.Function 的 DDMFunction 类
class DDMFunction(torch.autograd.Function):
    @staticmethod
    # 前向传播函数，接受一个上下文 ctx 和一个 torch.Tensor 类型的输入 x，返回一个 torch.Tensor 类型的输出
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        # 调用 torch.ops.dummy.ddm 函数，返回其结果
        return torch.ops.dummy.ddm(x)

    @staticmethod
    # 反向传播函数，接受一个上下文 ctx 和一个 torch.Tensor 类型的梯度 grad_x，返回一个 torch.Tensor 类型的梯度
    def backward(ctx: Any, grad_x: torch.Tensor) -> torch.Tensor:
        # 调用 torch.ops.dummy.ddm_backward 函数，返回其结果
        return torch.ops.dummy.ddm_backward(grad_x)

# 定义一个继承自 nn.Module 的 DummyDDM 类
class DummyDDM(nn.Module):
    def __init__(self):
        super().__init__()

    # 前向传播方法，接受一个输入 x，返回 DDMFunction 类的 apply 方法处理后的结果
    def forward(self, x):
        return DDMFunction.apply(x)

# 定义一个继承自 DTensorTestBase 的 TraceTrainStepTest 类
class TraceTrainStepTest(DTensorTestBase):
    @property
    # 返回当前测试的全局大小为 2
    def world_size(self):
        return 2

    # 如果 GPU 数量小于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    # 使用通信上下文进行测试
    @with_comms
    # 测试训练步骤的简单实现
    def test_train_step_simple(self):
        @compile()
        # 编译函数 train_step，接受模型 mod 和输入 inp 作为参数，执行模型反向传播并返回参数的梯度
        def train_step(mod, inp):
            mod(inp).sum().backward()
            return [p.grad for p in mod.parameters()]

        inp = torch.randn(2, 10).cuda(self.rank)
        # FIXME(@mrshenli): 一旦 dist.compile 能够同步模块参数，应该移除手动设置种子。
        # 手动设置随机种子为 0，以确保模块参数是同步的
        torch.manual_seed(0)
        mod = nn.Linear(10, 10).cuda(self.rank)

        # 使用深度复制方式创建 DDP 模型，并指定设备 ID 为当前 rank
        ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])
        ddp_inp = deepcopy(inp)

        # 执行模型训练步骤，获取模型参数的梯度
        grads = train_step(mod, inp)
        # 对 DDP 模型进行前向传播并计算损失的梯度
        ddp_mod(ddp_inp).sum().backward()

        # 遍历模型参数的梯度 grads 和 DDP 模型的参数 p2，并进行断言比较
        for g1, p2 in zip(grads, ddp_mod.parameters()):
            # FIXME(@mrshenli): DDP 默认会将梯度除以全局大小。我们应该匹配这种行为吗？
            # 断言 g1 除以全局大小等于 p2 的梯度
            self.assertEqual(g1 / self.world_size, p2.grad)
    # 测试优化器的功能，包括模型、分布式模型、优化器和分布式优化器，以及输入数据和训练步骤
    def _test_optimizer(self, mod, ddp_mod, opt, ddp_opt, inp, train_step):
        # 深拷贝输入数据以用于分布式训练
        ddp_inp = deepcopy(inp)

        # 计算模型对输入数据的输出，计算损失并进行反向传播
        mod(inp).sum().backward()
        # 执行优化器的一步参数更新
        opt.step()
        # 清空优化器的梯度缓存
        opt.zero_grad()

        # 由于一些测试设置了需要梯度的缓冲区，需要对它们进行特殊处理
        # 这里遍历模型的缓冲区并将它们的梯度设置为None
        for buf in mod.buffers():
            buf.grad = None

        # 对分布式模型进行类似的操作：计算输出、反向传播、优化器步骤和梯度清空
        ddp_mod(ddp_inp).sum().backward()
        ddp_opt.step()
        ddp_opt.zero_grad()
        for buf in ddp_mod.buffers():
            buf.grad = None

        # 测试参数的一致性
        train_step(mod, opt, inp)

        # 对分布式模型再次执行反向传播
        ddp_mod(ddp_inp).sum().backward()
        # 由于torch.distributed.compile尚未默认按world size分割梯度，这里进行手动调整
        with torch.no_grad():
            for p in ddp_mod.parameters():
                p.grad *= self.world_size
        # 执行分布式优化器的一步参数更新
        ddp_opt.step()

        # 检查模型和分布式模型的参数是否一致
        for p1, p2 in zip(mod.parameters(), ddp_mod.parameters()):
            self.assertEqual(p1, p2)

    # 跳过如果GPU数量小于2的情况，用于装饰测试函数
    @skip_if_lt_x_gpu(2)
    # 设置通信环境，用于装饰测试函数
    @with_comms
    def test_sgd(self):
        # 定义训练步骤的编译版本
        @compile()
        def train_step(mod, opt, inp):
            # 计算模型对输入数据的输出，计算损失并进行反向传播
            mod(inp).sum().backward()
            # 执行优化器的一步参数更新
            opt.step()

        # FIXME(@mrshenli): 当dist.compile可以同步模块参数时，移除手动设置种子
        # 手动设置随机种子
        torch.manual_seed(1)
        # 创建一个在GPU上运行的线性模型，包括偏置项
        mod = nn.Linear(10, 10, bias=True).cuda(self.rank)
        # 使用SGD优化器，并针对模型的参数进行优化
        opt = torch.optim.SGD(mod.parameters(), lr=0.01, foreach=True)
        # 创建一个形状为(2, 10)的随机输入数据，并在GPU上运行
        inp = torch.randn(2, 10).cuda(self.rank)

        # 使用分布式数据并行(DDP)来复制模型，并指定设备ID
        ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])
        # 使用SGD优化器，并针对分布式模型的参数进行优化
        ddp_opt = torch.optim.SGD(ddp_mod.parameters(), lr=0.01, foreach=True)
        
        # 调用上面定义的测试优化器函数，测试模型和分布式模型的行为
        self._test_optimizer(mod, ddp_mod, opt, ddp_opt, inp, train_step)
    # 定义名为 _test_adam 的方法，用于测试 Adam 优化器的功能
    def _test_adam(self, *, foreach: bool, fused: bool):
        # 定义内部类 AssertOverride，继承自 Override 类
        class AssertOverride(Override):
            def __init__(self, outer):
                self.outer = outer

            # 重写 replacement 方法，返回原始子模块
            def replacement(
                self, fqn: str, orig_submodule: torch.nn.Module
            ) -> torch.nn.Module:
                return orig_submodule

            # 重写 transform 方法，处理图形模块和平坦状态列表
            def transform(
                self,
                gm: fx.GraphModule,
                flat_state: List[torch.Tensor],
            ) -> fx.Graph:
                # 断言检查去重是否成功，确保只有一个 allreduce 操作
                self.outer.assertEqual(
                    len(
                        [
                            n
                            for n in gm.graph.nodes
                            if n.target == torch.ops.c10d_functional.all_reduce.default
                        ]
                    ),
                    1,
                )

                return gm

        # 使用 compile 装饰器，应用 AssertOverride 类作为模块覆盖
        @compile(module_override=[AssertOverride(self)])
        def train_step(mod, opt, inp):
            # 计算模型输出的总和，并进行反向传播
            mod(inp).sum().backward()
            # 执行优化器的优化步骤
            opt.step()

        # FIXME(@mrshenli): 一旦 dist.compile 能够同步模块参数，移除手动设置的随机种子
        # 设置随机种子为 0
        torch.manual_seed(0)
        # FIXME(@mrshenli): 梯度缺失偏置项
        # 创建一个包含单个线性层的序列模型，并将其部署到 self.rank 指定的 GPU 上
        mod = nn.Sequential(nn.Linear(10, 10, bias=False)).cuda(self.rank)
        # 创建 Adam 优化器，配置学习率为 0.01，根据 foreach 和 fused 参数设置优化器行为
        opt = torch.optim.Adam(
            mod.parameters(),
            lr=0.01,
            foreach=foreach,
            fused=fused,
            capturable=True,
        )
        # 创建输入张量，形状为 (2, 10)，并将其部署到 self.rank 指定的 GPU 上
        inp = torch.randn(2, 10).cuda(self.rank)

        # 使用深拷贝创建 DDP 分布式数据并行模型，设备 ID 为 [self.rank]
        ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])
        # 创建 DDP 模型的 Adam 优化器，配置学习率为 0.01，根据 foreach 和 fused 参数设置优化器行为
        ddp_opt = torch.optim.Adam(
            ddp_mod.parameters(), lr=0.01, foreach=foreach, fused=fused
        )
        
        # 调用测试优化器方法，传入原始模型、DDP 模型、优化器及其参数、输入数据和训练步骤函数
        self._test_optimizer(mod, ddp_mod, opt, ddp_opt, inp, train_step)

    # 装饰器，如果 GPU 数量少于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    # 装饰器，启用通信设置
    @with_comms
    # 测试 Adam 优化器的 foreach 模式
    def test_adam_foreach(self):
        self._test_adam(foreach=True, fused=False)

    # 装饰器，如果 GPU 数量少于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    # 装饰器，启用通信设置
    @with_comms
    # 测试 Adam 优化器的 fused 模式
    def test_adam_fused(self):
        self._test_adam(foreach=False, fused=True)
    def _test_train_step_override(self):
        # 初始化空列表，用于存储需要转换的目标节点
        transform_targets = []

        # 定义一个继承自Override的内部类DDMOverride
        class DDMOverride(Override):
            # 替换方法，根据条件返回DummyDDM或原始子模块
            def replacement(
                self, fqn: str, orig_submodule: torch.nn.Module
            ) -> torch.nn.Module:
                return (
                    DummyDDM()
                    if isinstance(orig_submodule, DataDependentModule)
                    else orig_submodule
                )

            # 转换方法，修改图中特定节点的行为
            def transform(
                self,
                gm: fx.GraphModule,
                flat_state: List[torch.Tensor],
            ) -> fx.Graph:
                nonlocal transform_targets
                # 遍历图中的节点
                for node in gm.graph.nodes:
                    # 如果节点的目标在指定的列表中
                    if node.target in [
                        torch.ops.dummy.ddm.default,
                        torch.ops.dummy.ddm_backward.default,
                    ]:
                        # 将节点目标添加到transform_targets列表中
                        transform_targets.append(node.target)
                        # N.B.: this is not a complete subgraph representing
                        # original logic, as we are testing the ability to
                        # modify graph after DTensor expansion.
                        # 在节点之前插入新节点，使用torch.add函数
                        with gm.graph.inserting_before(node):
                            new_node = gm.graph.call_function(torch.add, args=node.args)
                        # 替换所有使用此节点的地方为新节点
                        node.replace_all_uses_with(new_node)

                # 对图进行静态检查
                gm.graph.lint()
                # 消除死代码
                gm.graph.eliminate_dead_code()

                # 返回修改后的图模块
                return gm

        # 使用compile装饰器，传入DDMOverride类实例化对象作为参数
        @compile(module_override=[DDMOverride()])
        def train_step(mod, opt, inp):
            # 执行模型前向传播和反向传播
            mod(inp).sum().backward()
            # 执行优化器步骤
            opt.step()

        # 创建DummyModel模型实例，分配到指定GPU上
        mod = DummyModel(self.world_size).cuda(self.rank)
        # 使用SGD优化器，设置学习率为0.01
        opt = torch.optim.SGD(mod.parameters(), lr=0.01, foreach=False)
        # FIXME: symbolic tracing treats bs=1 as constant, have to use bs > 1.
        # 创建输入张量，分配到指定GPU上
        inp = torch.randn(4, 10).cuda(self.rank)
        # 调用train_step函数执行训练步骤
        train_step(mod, opt, inp)

        # 检查转换是否被正确调用
        self.assertEqual(
            transform_targets,
            [torch.ops.dummy.ddm.default, torch.ops.dummy.ddm_backward.default],
        )

    # 如果GPU数小于2，则跳过该测试用例
    @skip_if_lt_x_gpu(2)
    # 使用通信装饰器
    @with_comms
    def test_module_override(self):
        # 调用内部测试方法_test_train_step_override
        self._test_train_step_override()

    # 如果GPU数小于2，则跳过该测试用例
    @skip_if_lt_x_gpu(2)
    # 使用通信装饰器
    @with_comms
    def test_module_multi_fqn_override(self):
        transform_targets = []  # 初始化一个空列表，用于存储被转换目标的标识符

        class DDMOverride(Override):
            def replacement(
                self, fqn: str, orig_submodule: torch.nn.Module
            ) -> torch.nn.Module:
                # 如果原始子模块是 DataDependentModule 类型，则替换为 DummyDDM，否则保持不变
                return (
                    DummyDDM()
                    if isinstance(orig_submodule, DataDependentModule)
                    else orig_submodule
                )

            def transform(
                self,
                gm: fx.GraphModule,
                flat_state: List[torch.Tensor],
            ) -> fx.Graph:
                nonlocal transform_targets  # 使用外部的 transform_targets 列表
                for node in gm.graph.nodes:  # 遍历计算图中的每个节点
                    if node.target in [
                        torch.ops.dummy.ddm.default,
                        torch.ops.dummy.ddm_backward.default,
                    ]:
                        transform_targets.append(node.target)  # 将目标节点的标识符添加到列表中
                        # 注意：这不是一个完整的子图，用来表示原始逻辑，我们在扩展 DTensor 后测试修改图形的能力。
                        with gm.graph.inserting_before(node):
                            new_node = gm.graph.call_function(torch.add, args=node.args)
                        node.replace_all_uses_with(new_node)  # 用新节点替换所有使用该节点的地方

                gm.graph.eliminate_dead_code()  # 清除无效代码

                return gm  # 返回修改后的图形模块对象

        class MultiDDM(nn.Module):
            def __init__(self, world_size):
                super().__init__()
                self.l1 = nn.Linear(10, 10)
                self.ddm1 = DataDependentModule(world_size)  # 创建一个 DataDependentModule 实例
                self.l2 = nn.Linear(10, 10)
                self.ddm2 = DataDependentModule(world_size)  # 创建另一个 DataDependentModule 实例
                self.relu = nn.ReLU()

            def forward(self, x):
                assert len(x.size()) == 2  # 断言输入张量 x 的维度为 2

                # 返回模块的前向传播结果，依次经过 l1、ddm1、l2、ddm2 和 relu 层
                return self.relu(self.ddm2(self.l2(self.ddm1(self.l1(x)))))


        @compile(module_override=[DDMOverride()])
        def train_step(mod, opt, inp):
            mod(inp).sum().backward()  # 计算模型输出的和并进行反向传播
            opt.step()  # 执行优化器的参数更新

        mod = MultiDDM(self.world_size).cuda(self.rank)  # 创建 MultiDDM 模型实例，并移至 GPU
        opt = torch.optim.SGD(mod.parameters(), lr=0.01, foreach=False)  # 创建 SGD 优化器实例
        inp = torch.randn(4, 10).cuda(self.rank)  # 创建输入张量，并移至 GPU
        train_step(mod, opt, inp)  # 执行训练步骤

        # 检查是否确实调用了变换目标
        self.assertEqual(
            transform_targets,
            [
                torch.ops.dummy.ddm.default,
                torch.ops.dummy.ddm.default,
                torch.ops.dummy.ddm_backward.default,
                torch.ops.dummy.ddm_backward.default,
            ],
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_gm_cache_and_transformation(self):
        # 定义一个用于图优化的类
        class GraphOptimization:
            def __init__(self):
                self.call_count = 0

            # 用于对图模块进行优化的方法，每次调用计数加一
            def __call__(self, gm: fx.GraphModule) -> fx.GraphModule:
                self.call_count += 1
                return gm

        # 创建图优化对象实例
        graph_optimization = GraphOptimization()

        # 使用装饰器将图优化对象应用到训练步骤函数上
        @compile(gm_transformation=graph_optimization)
        def train_step(mod, opt, inp):
            # 模型前向传播，计算损失，反向传播
            mod(inp).sum().backward()
            # 优化器执行一步参数更新
            opt.step()

        # 获取当前进程的分布式计算排名
        rank = torch.distributed.get_rank()
        # 设置随机种子
        torch.manual_seed(0)
        # 在指定的 GPU 设备上创建一个线性模型
        mod = nn.Linear(10, 10, bias=False).cuda(rank)
        # 使用 Adam 优化器，并捕获参数状态
        opt = torch.optim.Adam(
            mod.parameters(), lr=0.01, foreach=False, capturable=True
        )
        # 创建输入张量，并移到指定的 GPU 设备上
        inp = torch.randn(2, 10).cuda(rank)

        # 实例化一次优化器状态
        mod(inp).sum().backward()
        opt.step()
        opt.zero_grad()

        # 执行训练步骤
        train_step(mod, opt, inp)
        # 断言图优化对象的调用次数为1
        self.assertEqual(graph_optimization.call_count, 1)
        # 获取已编译的训练步骤函数中的图模块对象
        gm = train_step.__dict__[COMPILED_OBJECT_KEY].gm
        # 再次执行训练步骤
        train_step(mod, opt, inp)
        # 断言两个图模块对象的内存地址相同
        self.assertEqual(id(gm), id(train_step.__dict__[COMPILED_OBJECT_KEY].gm))
        # 断言图优化对象的调用次数仍然为1
        self.assertEqual(graph_optimization.call_count, 1)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_buffer(self):
        # 定义一个包含缓冲区的神经网络模型
        class BufferModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)
                # 注册一个固定值为全1的缓冲区
                self.register_buffer("dummy_buffer", torch.ones(10, 10))

            def forward(self, x):
                # 注意：在前向传播中设置 requires_grad=True，因为深拷贝不适用于 requires_grad=True 的缓冲区
                self.dummy_buffer.requires_grad = True
                return torch.matmul(self.fc(x), self.dummy_buffer)

        # 定义一个自定义的优化器类
        class AssertOptimizer(torch.optim.Optimizer):
            def __init__(self, params, lr):
                super().__init__(params, dict(lr=lr))

            def step(self):
                # 断言参数组的长度为2
                assert len(self.param_groups[0]["params"]) == 2
                with torch.no_grad():
                    for p in self.param_groups[0]["params"]:
                        p += p.grad

        # 使用装饰器编译训练步骤函数
        @compile()
        def train_step(mod, opt, inp):
            mod(inp).sum().backward()
            opt.step()

        # 设置随机种子
        torch.manual_seed(0)
        # 在指定的 GPU 设备上创建 BufferModule 的实例
        mod = BufferModule().cuda(self.rank)
        # 创建输入张量，并移到指定的 GPU 设备上
        inp = torch.randn(2, 10).cuda(self.rank)
        # 使用自定义的断言优化器初始化优化器实例
        opt = AssertOptimizer(mod.parameters(), lr=0.01)

        # 使用分布式数据并行在多个 GPU 上复制模型和优化器
        ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])
        ddp_opt = AssertOptimizer(ddp_mod.parameters(), lr=0.01)

        # 调用辅助方法测试优化器
        self._test_optimizer(mod, ddp_mod, opt, ddp_opt, inp, train_step)
        # 断言模型的缓冲区与分布式数据并行模型的缓冲区相同
        self.assertEqual(mod.dummy_buffer, ddp_mod.module.dummy_buffer)
    def test_expand_dimension(self):
        # 定义一个测试方法，用于测试维度扩展功能

        @compile()
        # 使用装饰器编译以下函数
        def train_step(mod, opt, inp):
            # 训练步骤函数：对模型输出求和并进行反向传播
            mod(inp).sum().backward()
            opt.step()

        # 在 GPU 上创建一个具有偏置的线性模型
        mod = nn.Linear(10, 10, bias=True).cuda(self.rank)
        # 使用 SGD 优化器来优化模型参数
        opt = torch.optim.SGD(mod.parameters(), lr=0.01, foreach=True)
        # 创建一个指定形状的输入张量，并放置在指定的 GPU 设备上
        inp = torch.randn(2, 10).cuda(self.rank)
        # 执行训练步骤
        train_step(mod, opt, inp)

        # 遍历编译后函数的图中的所有节点
        for node in train_step._compiled_obj.gm.graph.nodes:
            # 如果节点的目标是 torch.ops.aten.expand.default
            if node.target == torch.ops.aten.expand.default:
                # 断言：反向传播扩展梯度操作的第二个参数应该与局部批次大小匹配
                # 而不是全局批次大小。
                self.assertEqual(node.args[1], [2, 10])
class CoverageTest(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    # 定义测试训练步骤的方法，其中使用了分布式数据并行（DDP）
    def _test_train_step(self, train_step, mod, *args):
        # 使用深拷贝创建分布式数据并行模型，指定设备ID为当前rank
        ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])

        # 使用随机梯度下降优化器初始化普通模型和分布式数据并行模型的优化器
        opt = torch.optim.SGD(mod.parameters(), lr=0.01, foreach=True)
        ddp_opt = torch.optim.SGD(ddp_mod.parameters(), lr=0.01, foreach=True)

        # 使用深拷贝复制参数列表作为分布式数据并行模型的参数列表
        ddp_args = deepcopy(args)

        # 在普通模型上执行训练步骤，反向传播梯度，执行优化步骤并清空梯度
        mod(*args).sum().backward()
        opt.step()
        opt.zero_grad()

        # 在分布式数据并行模型上执行训练步骤，反向传播梯度，执行优化步骤并清空梯度
        ddp_mod(*ddp_args).sum().backward()
        ddp_opt.step()
        ddp_opt.zero_grad()

        # 测试参数的一致性
        train_step(mod, opt, *args)

        # 在分布式数据并行模型上再次执行反向传播，修正梯度
        ddp_mod(*ddp_args).sum().backward()
        
        # FIXME(@mrshenli): DDP 默认会将梯度除以 world size，但 torch.distributed.compile 目前尚未执行该操作。
        # 使用 torch.no_grad() 块遍历分布式数据并行模型的参数，并调整梯度以补偿除法操作
        with torch.no_grad():
            for p in ddp_mod.parameters():
                p.grad *= self.world_size
        ddp_opt.step()

        # 检查普通模型参数与分布式数据并行模型参数是否相等
        for p1, p2 in zip(mod.parameters(), ddp_mod.parameters()):
            self.assertEqual(p1, p2)

    # 装饰器，条件为 GPU 数量小于 2 时跳过测试
    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_log_softmax(self):
        torch.manual_seed(0)

        # 使用 @compile() 装饰的训练步骤，执行模型的前向传播、反向传播和优化步骤
        @compile()
        def train_step(mod, opt, inp):
            mod(inp).sum().backward()
            opt.step()

        # 创建包含线性层和对数 softmax 的序列模型，将其部署到当前 rank 所指定的 GPU 上
        mod = nn.Sequential(
            nn.Linear(10, 10),
            nn.LogSoftmax(dim=1),
        ).cuda(self.rank)
        
        # 生成随机输入数据，部署到当前 rank 所指定的 GPU 上
        inp = torch.randn(2, 10).cuda(self.rank)
        
        # 执行测试训练步骤
        self._test_train_step(train_step, mod, inp)

    # 装饰器，条件为 GPU 数量小于 2 时跳过测试
    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_nll_loss(self):
        # 包含损失函数的模块类定义
        class ModuleWithLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.mod = nn.Sequential(
                    nn.Linear(10, 10),
                    nn.LogSoftmax(dim=1),
                )
                self.lss = nn.NLLLoss()

            def forward(self, x, tgt):
                return self.lss(self.mod(x), tgt)

        torch.manual_seed(0)
        # 使用包含损失函数的模块类创建模型，并部署到当前 rank 所指定的 GPU 上
        mod = ModuleWithLoss().cuda(self.rank)

        # 使用 @compile() 装饰的训练步骤，执行模型的前向传播、反向传播和优化步骤
        @compile()
        def train_step(mod, opt, inp, tgt):
            mod(inp, tgt).backward()
            opt.step()

        # 生成随机输入数据和目标标签，部署到当前 rank 所指定的 GPU 上
        inp = torch.randn(2, 10).to(self.rank)
        tgt = torch.empty(2, dtype=torch.long).random_(0, 10).to(self.rank)

        # 执行测试训练步骤
        self._test_train_step(train_step, mod, inp, tgt)

    # 装饰器，条件为 GPU 数量小于 2 时跳过测试
    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_replicated_embedding(self):
        N, D, B = 10, 8, 2  # 设置 Embedding 的参数：N 是词汇表大小，D 是嵌入维度，B 是批量大小

        class EmbeddingModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(N, D)  # 创建一个 NxD 的嵌入层
                self.norm = nn.LayerNorm(D, elementwise_affine=False)  # 使用 Layer Normalization 对 D 维向量进行标准化
                self.fc = nn.Linear(D, D)  # 创建一个从 D 维到 D 维的线性变换层
                self.softmax = nn.Softmax(dim=1)  # 在第一个维度上进行 Softmax 计算
                self.lss = nn.NLLLoss()  # 使用负对数似然损失函数

            def forward(self, ids, tgt):
                return self.lss(self.softmax(self.fc(self.norm(self.emb(ids)))), tgt)
                # 前向传播：嵌入层 -> Layer Norm -> 线性变换 -> Softmax -> NLLLoss

        torch.manual_seed(0)  # 设置随机种子
        mod = EmbeddingModule().cuda(self.rank)  # 创建模型实例并将其放到指定的 GPU 上

        @compile()  # 编译为可执行的计算图
        def train_step(mod, opt, ids, tgt):
            mod(ids, tgt).sum().backward()  # 计算损失并反向传播
            opt.step()  # 执行优化步骤

        ids = torch.randint(0, N, (B,)).cuda(self.rank)  # 生成随机的词索引，放到指定的 GPU 上
        tgt = torch.empty(B, dtype=torch.long).random_(0, D).to(self.rank)  # 生成随机的目标标签，放到指定的 GPU 上

        self._test_train_step(train_step, mod, ids, tgt)  # 执行训练步骤的测试

    @skip_if_lt_x_gpu(2)  # 如果 GPU 数量小于 2 则跳过该测试
    @with_comms  # 启用通信功能
    def test_pos_embedding(self):
        N, D, B, Block = 10, 8, 2, 20  # 设置位置嵌入的参数：N 是词汇表大小，D 是嵌入维度，B 是批量大小，Block 是块大小

        class EmbeddingModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.wte = nn.Embedding(N, D)  # 创建一个 NxD 的嵌入层（词嵌入）
                self.wpe = nn.Embedding(Block, D)  # 创建一个 BlockxD 的嵌入层（位置嵌入）
                self.norm = nn.LayerNorm(D, elementwise_affine=False)  # 使用 Layer Normalization 对 D 维向量进行标准化
                self.fc = nn.Linear(D, D)  # 创建一个从 D 维到 D 维的线性变换层

            def forward(self, ids, tgt):
                _, t = ids.size()  # 获取输入 ids 的维度信息
                wte = self.wte(ids)  # 计算词嵌入
                wpe = self.wpe(
                    torch.arange(0, t, dtype=torch.long, device=ids.device).unsqueeze(0)
                )  # 计算位置嵌入
                emb = wpe + wte  # 将词嵌入和位置嵌入相加
                norm = self.norm(emb)  # 对结果进行 Layer Norm
                fc = self.fc(norm)  # 线性变换
                log = F.softmax(fc, dim=-1)  # Softmax
                return F.cross_entropy(log.view(-1, log.size(-1)), tgt.view(-1))
                # 计算交叉熵损失

        torch.manual_seed(0)  # 设置随机种子
        mod = EmbeddingModule().cuda(self.rank)  # 创建模型实例并将其放到指定的 GPU 上

        @compile()  # 编译为可执行的计算图
        def train_step(mod, opt, ids, tgt):
            mod(ids, tgt).sum().backward()  # 计算损失并反向传播
            opt.step()  # 执行优化步骤

        ids = torch.randint(0, N, (B, Block)).cuda(self.rank)  # 生成随机的词索引，放到指定的 GPU 上
        tgt = torch.empty((B, Block), dtype=torch.long).random_(0, D).to(self.rank)  # 生成随机的目标标签，放到指定的 GPU 上

        self._test_train_step(train_step, mod, ids, tgt)  # 执行训练步骤的测试

    def _test_op_with_train_step(self, Model: Type[nn.Module]):
        torch.manual_seed(0)  # 设置随机种子
        mod = Model().cuda(self.rank)  # 创建模型实例并将其放到指定的 GPU 上

        @compile()  # 编译为可执行的计算图
        def train_step(mod, opt, inp):
            mod(inp).sum().backward()  # 计算损失并反向传播
            opt.step()  # 执行优化步骤

        inp = torch.randn(2, 10).cuda(self.rank)  # 生成随机的输入数据，放到指定的 GPU 上
        self._test_train_step(train_step, mod, inp)  # 执行训练步骤的测试

    @skip_if_lt_x_gpu(2)  # 如果 GPU 数量小于 2 则跳过该测试
    @with_comms  # 启用通信功能
    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_factory_full(self):
        # 定义一个测试函数，测试完整的模型功能
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个全连接层，输入和输出都是10维
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                # 创建一个与输入 x 形状相同的全为7的张量 y
                y = torch.full(x.shape, 7, device=x.device)
                # 返回全为7的张量 y 加上输入 x 经过全连接层 fc 后的结果
                return y + self.fc(x)

        # 调用 _test_op_with_train_step 函数测试模型 Model 的训练步骤
        self._test_op_with_train_step(Model)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_factory_arange(self):
        # 定义一个测试函数，测试通过 arange 创建的模型功能
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个全连接层，输入和输出都是10维
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                # 创建一个从0开始到 x 中元素总数的张量 y，形状与 x 相同
                y = torch.arange(x.numel(), device=x.device).view(x.shape)
                # 创建一个从0到 x 中元素总数的张量 z，形状与 x 相同
                z = torch.arange(0, x.numel(), device=x.device).view(x.shape)
                # 返回输入 x 经过全连接层 fc 后的结果，加上张量 y 和 z 的和
                return self.fc(x) + y + z

        # 调用 _test_op_with_train_step 函数测试模型 Model 的训练步骤
        self._test_op_with_train_step(Model)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_sym_numel(self):
        # 定义一个测试函数，测试获取权重元素个数的模型功能
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个全连接层，输入和输出都是10维
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                # 获取全连接层权重的元素个数 y
                y = self.fc.weight.numel()
                # 返回输入 x 经过全连接层 fc 后的结果，加上权重元素个数 y
                return self.fc(x) + y

        # 调用 _test_op_with_train_step 函数测试模型 Model 的训练步骤
        self._test_op_with_train_step(Model)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_sym_stride(self):
        # 定义一个测试函数，测试获取权重步长的模型功能
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个全连接层，输入和输出都是10维
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                # 获取全连接层权重的步长 y
                y = self.fc.weight.stride(0)
                # 返回输入 x 经过全连接层 fc 后的结果，加上权重步长 y
                return self.fc(x) + y

        # 调用 _test_op_with_train_step 函数测试模型 Model 的训练步骤
        self._test_op_with_train_step(Model)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_scalar(self):
        # 定义一个测试函数，测试标量张量创建的模型功能
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个全连接层，输入和输出都是10维
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                # FIXME: torch.tensor(x.numel()) is captured as a tensor constant
                # 创建一个标量张量 y，值为7，与输入 x 的数据类型和设备一致
                y = torch.ops.aten.scalar_tensor.default(
                    7, dtype=x.dtype, device=x.device
                )
                # 返回输入 x 经过全连接层 fc 后的结果，加上标量张量 y
                return self.fc(x) + y

        # 调用 _test_op_with_train_step 函数测试模型 Model 的训练步骤
        self._test_op_with_train_step(Model)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_stack(self):
        # 定义一个测试函数，测试堆叠张量的模型功能
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个全连接层，输入和输出都是10维
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                # 返回一个张量列表，包含输入 x 和经过全连接层 fc 后的结果，沿着第二维堆叠
                return torch.stack([x, self.fc(x)], dim=1)

        # 调用 _test_op_with_train_step 函数测试模型 Model 的训练步骤
        self._test_op_with_train_step(Model)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_arithmetic_ops_on_symint(self):
        # 定义一个测试函数，测试对称整数运算的模型功能
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个全连接层，输入和输出都是10维
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                # 返回输入 x 经过全连接层 fc 后的结果，加上 x 的元素个数乘以 x 中总元素数，减去 x 的元素个数整除2的结果
                return self.fc(x) + x.shape[0] * x.numel() - x.shape[0] // 2

        # 调用 _test_op_with_train_step 函数测试模型 Model 的训练步骤
        self._test_op_with_train_step(Model)
    # 定义一个测试方法，用于测试模型的切片操作
    def test_slice(self):
        # 定义一个简单的神经网络模型类
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义模型的全连接层，输入和输出均为大小为10的向量
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                # 模型的前向传播，返回全连接层输出的第一列数据
                return self.fc(x)[:, :1]

        # 调用测试方法，使用 Model 类进行测试
        self._test_op_with_train_step(Model)

    # 如果 GPU 数量小于 2，则跳过执行以下测试方法
    @skip_if_lt_x_gpu(2)
    # 使用通信上下文执行测试方法
    @with_comms
    def test_bulk_cat(self):
        # 定义一个模型类，用于测试 torch.cat 的批量连接操作
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义模型的全连接层，输入和输出均为大小为10的向量
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                # 模型的前向传播，使用列表推导式生成100个全连接层的输出，并在维度1上连接
                return torch.cat([self.fc(x) for _ in range(100)], dim=1)

        # 调用测试方法，使用 Model 类进行测试
        self._test_op_with_train_step(Model)
# 如果当前脚本是被直接运行的（而不是被导入到其他模块中执行），则执行以下代码块
if __name__ == "__main__":
    # 由于条件表达式为 False，因此此代码块内部的代码不会执行
    if False:
        # 运行测试函数（假设此处有一个名为 run_tests 的函数用于执行测试）
        run_tests()
```