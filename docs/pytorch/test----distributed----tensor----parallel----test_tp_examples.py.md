# `.\pytorch\test\distributed\tensor\parallel\test_tp_examples.py`

```py
# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

# 导入所需的模块和库
import itertools
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.distributed._tensor.debug import CommDebugMode
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    loss_parallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.distributed.tensor.parallel.input_reshard import input_reshard
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule,
    ModelArgs,
    NUM_DEVICES,
    skip_unless_torch_gpu,
    Transformer,
    with_comms,
)

# 引用 C10D 的函数式接口
c10d_functional = torch.ops.c10d_functional

# 定义测试类，继承自 DTensorTestBase
class DistTensorParallelExampleTest(DTensorTestBase):

    # 辅助方法：检查模型参数是否相同
    def _check_module(self, m1, m2, check_grad=False):
        named_parameters = dict(m1.named_parameters())
        for name, param_m2 in m2.named_parameters():
            self.assertTrue(name in named_parameters)
            param_m1 = named_parameters[name]
            if check_grad:
                param_m2 = param_m2.grad
                param_m1 = param_m1.grad
            # 如果参数是 DTensor 类型，则重新分发到本地设备进行比较
            if isinstance(param_m2, DTensor):
                replicate = [Replicate()]
                param_m2 = param_m2.redistribute(
                    device_mesh=param_m2.device_mesh, placements=replicate
                ).to_local()
            self.assertEqual(param_m2, param_m1)

    # 测试 MLP 模型的推断过程
    def _test_mlp_inference(self, device_mesh):
        inp_size = [8, 10]
        # 确保随机种子相同以便结果可复现
        torch.manual_seed(0)
        inp = torch.rand(*inp_size, device=self.device_type)
        model = MLPModule(self.device_type)
        model_tp = deepcopy(model)

        # 确保模型初始化方式相同
        self._check_module(model, model_tp)

        # 使用指定的设备网格和并行计划对模型进行分片化处理
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        model_tp = parallelize_module(model_tp, device_mesh, parallelize_plan)

        # 执行推断
        output = model(inp)
        output_tp = model_tp(inp)
        self.assertEqual(output, output_tp)

    # 使用通信装饰器执行测试，并对参数进行参数化
    @with_comms
    @parametrize("is_seq_parallel", [True, False])
    # TODO: 需要重新审视 input_reshard API 失败的多GPU测试原因。
    # @parametrize("recompute_activation", [True, False])
    @parametrize("recompute_activation", [False])
    def test_mlp_training(self, is_seq_parallel, recompute_activation):
        # 调用内部方法 _test_mlp_training_e2e 进行 MLP 训练端到端测试
        self._test_mlp_training_e2e(
            is_seq_parallel=is_seq_parallel, recompute_activation=recompute_activation
        )

    @with_comms
    def test_mlp_inference(self):
        # 创建设备网格对象，用于推理测试
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, NUM_DEVICES),
        )
        # 进入推理模式
        with torch.inference_mode():
            # 调用内部方法 _test_mlp_inference 进行 MLP 推理测试
            self._test_mlp_inference(device_mesh)

    @with_comms
    @skip_unless_torch_gpu
    @parametrize("is_seq_parallel", [True, False])
    @with_comms
    def test_weight_tying(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化嵌入层和全连接层的权重，分别使用不同的随机种子
                torch.manual_seed(1)
                self.embedding = torch.nn.Embedding(16, 8)
                torch.manual_seed(2)
                self.fc = torch.nn.Linear(8, 16)

            def forward(self, x):
                # 前向传播函数，返回全连接层对嵌入层的输出
                return self.fc(self.embedding(x))

        # 创建 TestModule 模型实例，并移动到指定设备类型
        model = TestModule().to(self.device_type)
        
        # 并行化计划，指定嵌入层和全连接层的并行化策略
        parallelize_plan = {
            "embedding": ColwiseParallel(),
            "fc": RowwiseParallel(),
        }
        # 创建设备网格对象，涵盖指定数量的设备
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 并行化模型
        parallelize_module(model, device_mesh, parallelize_plan)

        input_size = [5]
        # 使用指定的种子生成输入数据
        torch.manual_seed(0)
        inp = torch.randint(16, input_size, device=self.device_type)

        # 检验权重未绑定的情况
        self.assertNotEqual(
            model.embedding.weight.to_local(), model.fc.weight.to_local()
        )
        # 进行模型推理
        output = model(inp)
        # 计算输出的梯度
        output.sum().backward()
        # 检验权重梯度未绑定的情况
        self.assertNotEqual(
            model.embedding.weight.grad.to_local(), model.fc.weight.grad.to_local()
        )
        # 清除模型梯度
        model.zero_grad()

        # 绑定权重后的情况
        model.fc.weight = model.embedding.weight

        # 断言嵌入层和全连接层权重相等
        self.assertEqual(model.embedding.weight, model.fc.weight)
        # 断言嵌入层和全连接层权重对象相同
        self.assertEqual(id(model.embedding.weight), id(model.fc.weight))
        # 再次进行模型推理
        output = model(inp)
        # 计算输出的梯度
        output.sum().backward()
        # 断言嵌入层和全连接层权重梯度相等
        self.assertEqual(model.embedding.weight.grad, model.fc.weight.grad)
        # 断言嵌入层和全连接层权重梯度对象相同

    @with_comms
    # 定义一个测试方法，用于测试并行计算中的损失函数
    def test_loss_parallel(self):
        # 构建设备网格，用于分布式计算
        device_mesh = self.build_device_mesh()
        # 创建通信调试模式对象
        comm_mode = CommDebugMode()

        # 设置通道大小和通道维度
        channel_size, channel_dim = 16, 1
        # 设置测试参数集合
        test_setup = [
            (2, (8, channel_size), (8,)),  # 调用 aten.nll_loss_forward
            (3, (8, channel_size, 12), (8, 12)),  # 调用 aten.nll_loss2d_forward
        ]
        # 随机生成权重矩阵
        weight = torch.rand(channel_size, device=self.device_type)

        # 遍历测试参数集合
        for input_ndim, input_size, target_size in test_setup:
            # 随机生成输入张量 x，并设置其在指定设备上，并且需要梯度信息
            x = torch.rand(*input_size, device=self.device_type, requires_grad=True)
            # 随机生成目标张量 target
            target = torch.randint(channel_size, target_size, device=self.device_type)

            # 设置分片维度和减少方式
            shard_dims = list(range(input_ndim))
            reductions = ["none", "mean", "sum"]
            # 遍历分片维度和减少方式的组合
            for shard_dim, reduction in itertools.product(shard_dims, reductions):
                # 将输入张量 x 在设备网格上分布，并通过指定分片维度进行分片
                dist_x = distribute_tensor(x, device_mesh, [Shard(shard_dim)])
                # 计算损失函数 y，并传入权重和减少方式
                y = F.cross_entropy(x, target, weight, reduction=reduction)
                # 进入损失函数并行环境
                with loss_parallel():
                    # 如果分片维度等于通道维度
                    if shard_dim == channel_dim:
                        # 进入通信调试模式
                        with comm_mode:
                            # 计算分布式环境下的损失函数 dist_y
                            dist_y = F.cross_entropy(
                                dist_x, target, weight, reduction=reduction
                            )
                            # 断言通信调试模式的总数为3
                            self.assertEqual(comm_mode.get_total_counts(), 3)
                            # 断言调用通信函数的次数为3
                            self.assertEqual(
                                comm_mode.get_comm_counts()[c10d_functional.all_reduce],
                                3,
                            )
                            # 断言 dist_y 的部署是复制的
                            self.assertTrue(dist_y.placements[0].is_replicate())
                            # 断言 dist_y 转为本地后与 y 相等
                            self.assertEqual(dist_y.to_local(), y)

                        # 再次进入通信调试模式
                        with comm_mode:
                            # 如果减少方式为 "none"，则计算损失函数的和并进行反向传播
                            if reduction == "none":
                                y.sum().backward()
                                dist_y.sum().backward()
                            else:
                                y.backward()
                                dist_y.backward()
                            # 断言通信调试模式的总数为0
                            self.assertEqual(comm_mode.get_total_counts(), 0)
                            # 断言 dist_x 梯度的部署是分片维度
                            self.assertTrue(
                                dist_x.grad.placements[0].is_shard(shard_dim)
                            )
                            # 断言 dist_x 的梯度与 x 的梯度相等
                            self.assertEqual(dist_x.grad.full_tensor(), x.grad)
                        # 清零 x 的梯度
                        x.grad.zero_()
                    else:
                        # 如果分片维度不等于通道维度，断言抛出 ValueError 异常
                        with self.assertRaisesRegex(
                            ValueError,
                            "loss_parallel",
                        ):
                            dist_y = F.cross_entropy(
                                dist_x, target, reduction=reduction
                            )
# 实例化带参数的测试用例，并传入 DistTensorParallelExampleTest 类
instantiate_parametrized_tests(DistTensorParallelExampleTest)

# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 运行测试
    run_tests()
```