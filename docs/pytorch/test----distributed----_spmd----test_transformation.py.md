# `.\pytorch\test\distributed\_spmd\test_transformation.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入单元测试框架
import unittest
# 导入深拷贝函数
from copy import deepcopy
# 导入装饰器函数
from functools import wraps
# 导入 MagicMock 类
from unittest.mock import MagicMock

# 导入 PyTorch 相关模块
import torch
import torch.nn as nn
# 导入分布式 SPMD 模块相关 API
from torch.distributed._spmd.api import compile
# 导入图模块转换类
from torch.distributed._spmd.gm_transformation import GraphModuleTransformation
# 导入图优化相关函数
from torch.distributed._spmd.graph_optimization import (
    _optimized_func,
    comm_fusion_with_concat,
    find_all_descendants,
    get_all_fused_optimizer_blocks,
    graph_optimization_pass,
    iter_move_grads_and_optimizers,
    remove_copy_from_optimizer,
    schedule_comm_wait,
    split_fused_optimizer,
)
# 导入图工具函数
from torch.distributed._spmd.graph_utils import find_node
# 导入迭代图模块类
from torch.distributed._spmd.iter_graph_module import IterGraphModule
# 导入分布式数据并行类
from torch.nn.parallel import DistributedDataParallel as DDP
# 导入 GPU 数量检测函数
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
# 导入 ROCm 平台检测装饰器
from torch.testing._internal.common_utils import run_tests, skipIfRocm
# 导入 Triton 加速库检测函数
from torch.utils._triton import has_triton

# 自定义装饰器函数，设置不同的随机种子以确保每个进程使用不同的随机数
def with_comms(func):
    @base_with_comms
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # 设置随机种子为当前进程的排名（self.rank）
        torch.manual_seed(self.rank)
        return func(self, *args, **kwargs)

    return wrapper


# 定义一个简单的神经网络模型类
class DummyModel(nn.Module):
    def __init__(self, layers: int, dim: int):
        super().__init__()
        modules = []
        # 构建指定层数和维度的神经网络模型
        for _ in range(layers):
            modules.extend([nn.Linear(dim, dim), nn.ReLU()])
        self.mod = nn.Sequential(*modules)

    def forward(self, x):
        return self.mod(x)


# 继承自 DTensorTestBase 类的图优化测试类
class GraphPassWrapperTest(DTensorTestBase):
    @property
    def world_size(self):
        return 1  # 返回当前测试的全局进程数量为 1
    def test_order(self):
        # 定义图优化的装饰器函数 my_pass1，无先决条件，无应用后条件
        @graph_optimization_pass(
            prerequisites=[],
            apply_after=[],
        )
        def my_pass1(gm) -> None:
            return

        # 定义图优化的装饰器函数 my_pass2，先决条件为 my_pass1，无应用后条件
        @graph_optimization_pass(
            prerequisites=[my_pass1],
            apply_after=[],
        )
        def my_pass2(gm) -> None:
            return

        # 定义图优化的装饰器函数 my_pass3，无先决条件，应用后条件为 my_pass1
        @graph_optimization_pass(
            prerequisites=[],
            apply_after=[my_pass1],
        )
        def my_pass3(gm) -> None:
            return

        # 创建一个 MagicMock 对象 gm，模拟 IterGraphModule
        gm = MagicMock(spec=IterGraphModule)
        
        # 调用 my_pass1 和 my_pass3，不应引发错误
        my_pass1(gm)
        my_pass3(gm)
        
        # 调用 my_pass2 会引发 AssertionError，因为它的先决条件未满足
        with self.assertRaisesRegex(AssertionError, "are the prerequisites of"):
            my_pass2(gm)
        
        # 再次调用 my_pass3 会引发 AssertionError，因为它必须在 my_pass1 后应用
        with self.assertRaisesRegex(AssertionError, "must be applied after"):
            my_pass3(gm)
            my_pass1(gm)
# 定义一个继承自DTensorTestBase的TransformationTest类，用于测试模型转换功能
class TransformationTest(DTensorTestBase):

    # 定义world_size属性，返回值为2，表示测试时使用的世界大小
    @property
    def world_size(self):
        return 2

    # 定义_init方法，用于初始化测试所需的模型、优化器等
    def _init(self, batch_size, layers, dim, foreach: bool = True, fused: bool = False):
        # 设置随机种子为0
        torch.manual_seed(0)
        # 创建一个DummyModel模型实例，使用CUDA加速
        model = DummyModel(layers, dim).cuda()
        # 使用DDP将模型进行分布式数据并行处理
        ddp_model = DDP(deepcopy(model), device_ids=[self.rank])
        # 创建Adam优化器，用于优化模型的参数
        optim = torch.optim.Adam(
            model.parameters(), lr=0.01, foreach=foreach, fused=fused, capturable=True
        )
        # 使用DDP模型的参数创建Adam优化器
        ddp_optim = torch.optim.Adam(
            ddp_model.parameters(),
            lr=0.01,
            foreach=foreach,
            fused=fused,
            capturable=True,
        )
        # 创建一个随机输入数据batch，使用CUDA加速
        batch = torch.randn(batch_size, dim).cuda()

        # 计算模型对输入batch的输出
        out = model(batch)
        # 对模型输出的结果进行反向传播
        out.sum().backward()
        # 使用优化器更新模型参数
        optim.step()
        # 清空优化器中的梯度
        optim.zero_grad()

        # 对DDP模型进行相同的操作
        ddp_out = ddp_model(batch)
        ddp_out.sum().backward()
        ddp_optim.step()
        ddp_optim.zero_grad()

        # 断言DDP模型的输出与非分布式模型的输出一致
        self.assertEqual(ddp_out, out)
        # 断言DDP模型的参数列表与非分布式模型的参数列表一致
        self.assertEqual(list(ddp_model.parameters()), list(model.parameters()))
        # 返回非分布式模型、优化器、DDP模型及其优化器
        return model, optim, ddp_model, ddp_optim

    # 定义_test_train_step方法，用于测试训练步骤
    def _test_train_step(
        self, train_step, num_iters, batch_size, layers, dim, use_fused_optimizer=False
    ):
        # 定义_ddp_train_step内部函数，用于分布式数据并行训练步骤
        def _ddp_train_step(model, optim, batch):
            # 计算模型对输入batch的输出，并进行反向传播
            model(batch).sum().backward()
            # 将每个参数的梯度乘以世界大小
            with torch.no_grad():
                for p in model.parameters():
                    p.grad *= self.world_size
            # 使用优化器更新模型参数
            optim.step()
            # 清空优化器中的梯度
            optim.zero_grad()

        # 初始化非分布式模型、优化器、DDP模型及其优化器
        model, optim, ddp_model, ddp_optim = self._init(
            batch_size,
            layers,
            dim,
            foreach=(not use_fused_optimizer),
            fused=use_fused_optimizer,
        )
        # 迭代执行训练步骤
        for i in range(num_iters):
            # 创建一个随机输入数据batch，使用CUDA加速
            batch = torch.randn(batch_size, dim).cuda()
            # 如果是最后一次迭代，传入last_train_step参数为True
            kwargs = {} if i < num_iters - 1 else {"last_train_step": True}
            # 执行训练步骤
            out = train_step(model, optim, batch, **kwargs)
            # 执行DDP模型的训练步骤
            ddp_out = _ddp_train_step(ddp_model, ddp_optim, batch)
        # 断言DDP模型的参数列表与非分布式模型的参数列表一致
        self.assertEqual(list(ddp_model.parameters()), list(model.parameters()))

    # 定义test_basic_transformation测试方法，用于测试基本转换功能
    @skip_if_lt_x_gpu(2)  # 如果GPU数量小于2，跳过测试
    @with_comms  # 使用通信上下文
    def test_basic_transformation(self):
        batch_size = 100
        layers = 10
        dim = 100
        num_iters = 5

        # 定义train_step函数，使用图模块转换进行编译
        @compile(gm_transformation=GraphModuleTransformation())
        def train_step(model, optim, batch):
            # 计算模型对输入batch的输出，并进行反向传播
            model(batch).sum().backward()
            # 使用优化器更新模型参数
            optim.step()
            # 清空优化器中的梯度
            optim.zero_grad()

        # 执行测试训练步骤
        self._test_train_step(train_step, num_iters, batch_size, layers, dim)

    # 跳过测试条件：如果没有安装Triton或GPU架构较旧，则跳过测试
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)  # 如果GPU数量小于2，跳过测试
    @skipIfRocm  # 如果是Rocm环境，跳过测试
    @with_comms  # 使用通信上下文
    def test_inductor(self):
        batch_size = 100
        # 定义神经网络训练时的层级数量，过多层会导致编译超时。
        layers = 2
        # 定义输入数据的维度。
        dim = 100
        # 定义训练迭代的次数。
        num_iters = 5

        @compile(
            gm_transformation=GraphModuleTransformation(
                enable_inductor=True, dump_graphs=True
            )
        )
        def train_step(model, optim, batch):
            # 计算模型输出的和，然后反向传播梯度。
            model(batch).sum().backward()
            # 执行优化器的一步参数更新。
            optim.step()
            # 清空梯度信息。
            optim.zero_grad()

        # 调用内部函数 _test_train_step 进行训练步骤测试。
        self._test_train_step(
            train_step, num_iters, batch_size, layers, dim, use_fused_optimizer=True
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_graph_optimization_with_foreach(self):
        batch_size = 100
        layers = 2
        dim = 4096
        num_iters = 5

        @compile(
            gm_transformation=GraphModuleTransformation(
                enable_graph_optimization=True,
                dump_graphs=False,
            )
        )
        def train_step(model, optim, batch):
            model(batch).sum().backward()
            optim.step()
            optim.zero_grad()

        # 调用内部函数 _test_train_step 进行训练步骤测试。
        self._test_train_step(train_step, num_iters, batch_size, layers, dim)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_graph_optimization_with_fused(self):
        batch_size = 100
        layers = 2
        dim = 4096
        num_iters = 5

        @compile(
            gm_transformation=GraphModuleTransformation(
                enable_graph_optimization=True,
                dump_graphs=False,
            )
        )
        def train_step(model, optim, batch):
            model(batch).sum().backward()
            optim.step()
            optim.zero_grad()

        # 调用内部函数 _test_train_step 进行训练步骤测试，使用融合优化器。
        self._test_train_step(
            train_step, num_iters, batch_size, layers, dim, use_fused_optimizer=True
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_split_fused_optimizer(self):
        batch_size = 100
        layers = 2
        dim = 4096
        num_iters = 5

        def my_transformation(gm):
            # 将模型转换为迭代图模型。
            gm = IterGraphModule(gm)
            # 从优化器中移除复制操作。
            remove_copy_from_optimizer(gm)
            # 获取所有融合优化器块中的梯度信息。
            opt_block = get_all_fused_optimizer_blocks(gm, "_fused_adam")[0]
            gradients = {
                opt_block.optim.optim_node.args[1][1],
                opt_block.optim.optim_node.args[1][2],
            }
            # 将融合优化器拆分成两部分。
            split_fused_optimizer(gm, opt_block, gradients)
            # 消除无用的代码。
            gm.graph.eliminate_dead_code()
            # 重新编译模型。
            gm.recompile()
            # 断言融合优化器块的数量为2。
            self.assertEqual(len(get_all_fused_optimizer_blocks(gm, "_fused_adam")), 2)
            # 完成模型的设置。
            gm.finalize_setup()
            return gm

        @compile(gm_transformation=my_transformation)
        def train_step(model, optim, batch):
            model(batch).sum().backward()
            optim.step()
            optim.zero_grad()

        # 调用内部函数 _test_train_step 进行训练步骤测试，使用融合优化器。
        self._test_train_step(
            train_step, num_iters, batch_size, layers, dim, use_fused_optimizer=True
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    # 定义一个测试方法，用于测试迭代移动块和优化器
    def test_iter_move_blocks_and_optimizers(self):
        # 设置批处理大小
        batch_size = 100
        # 设置层数
        layers = 5
        # 设置维度
        dim = 4096
        # 设置迭代次数
        num_iters = 5

        # 定义一个变换函数，接受参数 gm（图模块）
        def my_transformation(gm):
            # 将图模块 gm 包装为 IterGraphModule 对象
            gm = IterGraphModule(gm)
            # 执行通信融合与连接操作
            comm_fusion_with_concat(gm, 100)
            # 调度通信等待
            schedule_comm_wait(gm)
            # 从优化器中移除复制操作
            remove_copy_from_optimizer(gm)
            # 执行迭代移动梯度和优化器操作，参数为 "all_reduce_default_1" 和 "relu"
            iter_move_grads_and_optimizers(gm, "all_reduce_default_1", "relu")
            # 完成最终设置
            gm.finalize_setup()
            # 返回修改后的图模块 gm
            return gm

        # 使用编译装饰器，传入 gm_transformation=my_transformation 参数
        @compile(gm_transformation=my_transformation)
        # 定义训练步骤函数 train_step，接受模型、优化器和批次数据作为参数
        def train_step(model, optim, batch):
            # 对模型进行前向传播计算，求和后进行反向传播
            model(batch).sum().backward()
            # 执行优化器的一步参数更新
            optim.step()
            # 清空优化器的梯度信息
            optim.zero_grad()

        # 调用内部方法 _test_train_step，传入 train_step 函数以及其他训练参数
        self._test_train_step(
            train_step, num_iters, batch_size, layers, dim, use_fused_optimizer=True
        )

    # 如果 GPU 数量少于 2，则跳过该测试
    @skip_if_lt_x_gpu(2)
    # 添加通信环境装饰器
    @with_comms
    # 定义测试方法，用于测试查找所有后代节点功能
    def test_find_all_descendants(self):
        # 设置批处理大小
        batch_size = 100
        # 设置层数
        layers = 5
        # 设置维度
        dim = 4096
        # 设置迭代次数
        num_iters = 2

        # 定义一个变换函数，接受参数 gm（图模块）
        def my_transformation(gm):
            # 将图模块 gm 包装为 IterGraphModule 对象
            gm = IterGraphModule(gm)
            # 查找图中第一个名为 "all_reduce" 的节点
            node1 = find_node(gm.graph, lambda n: n.name == "all_reduce")[0]
            # 查找图中第一个名为 "_foreach_add" 的节点
            node2 = find_node(gm.graph, lambda n: n.name == "_foreach_add")[0]
            # 查找所有指定节点的后代节点
            nodes_to_move = find_all_descendants(gm, [node1, node2])
            # 查找图中第一个名为 "relu" 的节点
            stop_node = find_node(gm.graph, lambda n: n.name == "relu")[0]
            # 将指定节点移动到下一次迭代之前，并返回修改后的图模块 gm
            gm.graph.move_to_next_iter_before(nodes_to_move, stop_node)
            return gm

        # 使用编译装饰器，传入 gm_transformation=my_transformation 参数
        @compile(gm_transformation=my_transformation)
        # 定义训练步骤函数 train_step，接受模型、优化器和批次数据作为参数
        def train_step(model, optim, batch):
            # 对模型进行前向传播计算，求和后进行反向传播
            model(batch).sum().backward()
            # 执行优化器的一步参数更新
            optim.step()
            # 清空优化器的梯度信息
            optim.zero_grad()

        # 调用内部方法 _test_train_step，传入 train_step 函数以及其他训练参数
        self._test_train_step(
            train_step, num_iters, batch_size, layers, dim, use_fused_optimizer=True
        )
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 检查条件，这里条件为 False，所以不会运行 run_tests() 函数
    if False:
        # 如果条件为真，则运行测试函数
        run_tests()
```