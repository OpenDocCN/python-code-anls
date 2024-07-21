# `.\pytorch\test\distributed\tensor\parallel\test_parallelize_api.py`

```
# 导入必要的模块和类
from collections import OrderedDict  # 导入有序字典模块
from copy import deepcopy  # 导入深拷贝模块

import torch  # 导入PyTorch库
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard  # 导入分布式张量相关类
from torch.distributed.tensor.parallel.api import parallelize_module  # 导入模块并行化API
from torch.distributed.tensor.parallel.style import (
    ColwiseParallel,  # 导入按列并行风格
    PrepareModuleInput,  # 导入模块输入预处理
    PrepareModuleOutput,  # 导入模块输出预处理
    RowwiseParallel,  # 导入按行并行风格
)
from torch.testing._internal.common_utils import run_tests  # 导入测试运行函数
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,  # 导入分布式张量测试基类
    MLPModule,  # 导入多层感知机模块类
    MLPStacked,  # 导入多层感知机堆叠类
    with_comms,  # 导入通信装饰器
)


class DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x  # 简单的前向传播，返回输入


class TensorParallelAPITests(DTensorTestBase):
    @property
    def world_size(self):
        # 计算并返回GPU数量，作为世界大小的属性
        gpu_num = torch.cuda.device_count()
        return gpu_num if gpu_num % 2 == 0 and gpu_num > 4 else 4

    def _compare_params(
        self,
        local_module,
        dist_module,
        rank0_only,
        skip_rowwise_bias=False,
        compare_grad=False,
    ):
        # 比较本地模块和分布式模块的参数
        replicate = [Replicate()]  # 创建复制策略列表
        for name, param in local_module.named_parameters():
            dist_param = dist_module.get_parameter(name)  # 获取分布式模块的参数
            param = param.grad if compare_grad else param  # 如果需要比较梯度，则使用参数的梯度
            dist_param = dist_param.grad if compare_grad else dist_param  # 如果需要比较梯度，则使用分布式参数的梯度
            if (
                (not rank0_only)
                or (self.rank == 0)
                or (
                    name not in ["net2.bias"]
                    and not skip_rowwise_bias
                    or name not in ["bias", "net2.bias"]
                )
            ):
                # 如果不仅限于rank0，或当前rank是0，或者名称不在指定列表中
                self.assertEqual(
                    param,
                    dist_param.redistribute(
                        device_mesh=dist_param.device_mesh, placements=replicate
                    ).to_local(),
                    f"{name} not equal between dist and non-dist",  # 断言参数相等性
                )

    def _compare_module(
        self, local_module, dist_module, inp_size, rank0_only=True, rowwise=False
    ):
        LR = 0.25  # 设置用于测试的学习率
        local_optim = torch.optim.SGD(local_module.parameters(), lr=LR)  # 创建本地模型的随机梯度下降优化器
        dist_optim = torch.optim.SGD(dist_module.parameters(), lr=LR)   # 创建分布式模型的随机梯度下降优化器
        torch.manual_seed(0)  # 设置随机种子为0
        inp = torch.rand(*inp_size, device=self.device_type)  # 生成指定大小的随机输入张量，并放置在指定的设备上
        self._compare_params(local_module, dist_module, rank0_only)  # 比较本地模型和分布式模型的参数

        # 检查前向传播的正确性
        local_output = local_module(inp)  # 在本地模型上进行前向传播
        inp = inp.chunk(self.world_size, dim=-1)[self.rank] if rowwise else inp  # 如果按行分割，则根据进程排列方式调整输入张量
        dist_output = dist_module(inp)  # 在分布式模型上进行前向传播
        dist_output = (
            dist_output.redistribute(dist_output.device_mesh, [Replicate()]).to_local()
            if isinstance(dist_output, DTensor)
            else dist_output
        )  # 如果输出是 DTensor 类型，则重新分配设备并转换为本地设备
        self.assertEqual(local_output, dist_output)  # 断言本地输出和分布式输出是否相等

        local_output.sum().backward()  # 对本地模型的输出求和并进行反向传播
        dist_output.sum().backward()   # 对分布式模型的输出求和并进行反向传播

        # 检查反向传播并确保梯度相同
        self._compare_params(local_module, dist_module, rank0_only, rowwise, True)  # 比较本地模型和分布式模型的参数，包括梯度

        local_optim.step()  # 在本地模型上执行优化步骤
        dist_optim.step()   # 在分布式模型上执行优化步骤
        self._compare_params(local_module, dist_module, rank0_only, rowwise)  # 再次比较本地模型和分布式模型的参数
    def test_linear_row_wise_parallel(self):
        # 测试行并行化
        # 定义输入尺寸
        inp_size = [9, 16]
        # 创建行并行化对象
        rowwise = RowwiseParallel()

        # 设置随机种子
        torch.manual_seed(5)
        # 创建线性模型，输入维度为16，输出维度为10，设备为self.device_type
        model = torch.nn.Linear(16, 10, device=self.device_type)
        # 深拷贝模型用于后续操作
        model_tp = deepcopy(model)

        # 并行化模型 model_tp
        # 创建设备网格对象，包含多个设备
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 使用行并行化策略并行化模型
        model_tp = parallelize_module(model_tp, device_mesh, rowwise)

        # 让每个进程生成唯一的本地输入数据
        torch.manual_seed(self.rank)
        # 比较模块 model 和 model_tp 的效果，使用行并行化
        self._compare_module(model, model_tp, inp_size, rowwise=True)

    @with_comms
    def test_linear_col_wise_parallel(self):
        # 测试列并行化
        # 定义输入尺寸
        inp_size = [8, 10]
        # 创建列并行化对象，输出布局为Replicate
        colwise = ColwiseParallel(output_layouts=Replicate())

        # 设置随机种子
        torch.manual_seed(5)
        # 创建线性模型，输入维度为10，输出维度为16，设备为self.device_type
        model = torch.nn.Linear(10, 16, device=self.device_type)
        # 深拷贝模型用于后续操作
        model_tp = deepcopy(model)

        # 并行化模型 model_tp
        # 创建设备网格对象，包含多个设备
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 使用列并行化策略并行化模型
        model_tp = parallelize_module(model_tp, device_mesh, colwise)

        # 比较模块 model 和 model_tp 的效果，不使用行并行化
        self._compare_module(model, model_tp, inp_size)

    @with_comms
    def test_prepare_module_input(self):
        # 测试模块输入准备
        # 创建虚拟模块
        module = DummyModule()
        # 创建设备网格对象，包含多个设备
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 并行化模块 module，使用输入布局为Shard(0)，目标输入布局为Replicate
        parallelize_module(
            module,
            device_mesh,
            PrepareModuleInput(
                input_layouts=Shard(0), desired_input_layouts=Replicate()
            ),
        )
        # 创建随机输入张量
        inp = torch.rand(5, 7, device=self.device_type)
        # 将输入张量输入模块 module，然后重新分发到设备网格，并转换为本地张量
        output = module(inp).redistribute(device_mesh, [Shard(0)]).to_local()
        # 断言输入和输出张量相等
        self.assertEqual(inp, output)

    @with_comms
    def test_prepare_module_output(self):
        # 测试模块输出准备
        # 创建虚拟模块
        module = DummyModule()
        # 创建设备网格对象，包含多个设备
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 并行化模块 module，使用输出布局为Replicate，目标输出布局为Shard(0)
        parallelize_module(
            module,
            device_mesh,
            PrepareModuleOutput(
                output_layouts=Replicate(), desired_output_layouts=Shard(0)
            ),
        )
        # 设置随机种子
        torch.manual_seed(15)
        # 创建随机输入张量
        inp = torch.rand(16, 7, device=self.device_type)
        # 从本地张量创建分布式张量，使用Replicate布局，不进行运行时检查
        dtensor = DTensor.from_local(inp, device_mesh, [Replicate()], run_check=False)
        # 将分布式张量输入模块 module
        output = module(dtensor)
        # 将输出重新分发到设备网格，使用Shard(0)布局，并转换为本地张量
        inp = dtensor.redistribute(device_mesh, [Shard(0)]).to_local()
        # 断言输入和输出张量相等
        self.assertEqual(inp, output)

    @with_comms
    def test_parallelize_module_with_star(self):
        # 使用通配符并行化模块测试
        # 定义输入尺寸
        inp_size = [12, 10]
        # 创建MLP模块对象，设备为self.device_type
        model = MLPModule(self.device_type)
        # 创建设备网格对象，包含多个设备
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # 深拷贝模型用于后续操作
        model_tp = deepcopy(model)
        # 使用通配符并行化模型 model_tp，列并行化，输出布局为Replicate
        model_tp = parallelize_module(
            model_tp,
            device_mesh,
            {
                "net*": ColwiseParallel(output_layouts=Replicate()),
            },
        )
        # 比较模块 model 和 model_tp 的效果，不仅仅在rank0上执行
        self._compare_module(model, model_tp, inp_size, rank0_only=False)
    # 定义一个测试方法，用于测试并行化模块与带问号的模块名称
    def test_parallelize_module_with_question(self):
        # 输入数据大小
        inp_size = [12, 10]
        # 创建一个 MLP 模块对象
        model = MLPModule(self.device_type)
        # 创建设备网格对象
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # 深拷贝模型对象
        model_tp = deepcopy(model)
        # 并行化模块处理，使用 ColwiseParallel 输出布局为 Replicate
        model_tp = parallelize_module(
            model_tp,
            device_mesh,
            {
                "net?": ColwiseParallel(output_layouts=Replicate()),
            },
        )
        # 比较处理前后的模块
        self._compare_module(model, model_tp, inp_size, rank0_only=False)

    # 使用 @with_comms 装饰器定义的测试方法，测试并行化模块与数字的模块名称
    @with_comms
    def test_parallelize_module_with_digit(self):
        # 输入数据大小
        inp_size = [12, 10]
        # 创建一个 MLP 模块对象
        model = MLPModule(self.device_type)
        # 创建设备网格对象
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # 深拷贝模型对象
        model_tp = deepcopy(model)
        # 并行化模块处理，使用 ColwiseParallel 输出布局为 Replicate
        model_tp = parallelize_module(
            model_tp,
            device_mesh,
            {
                "net[1-2]": ColwiseParallel(output_layouts=Replicate()),
            },
        )
        # 比较处理前后的模块
        self._compare_module(model, model_tp, inp_size, rank0_only=False)

    # 使用 @with_comms 装饰器定义的测试方法，测试多个通配符的并行化模块
    def test_parallelize_module_multi_wildcard(self):
        # 输入数据大小
        inp_size = [12, 10]
        # 创建一个具有多层的 MLPStacked 模块对象
        model = MLPStacked(self.device_type, n_layers=2)
        # 创建设备网格对象
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # 深拷贝模型对象
        model_tp = deepcopy(model)
        # 并行化模块处理，使用 ColwiseParallel 输出布局为默认设置，和 RowwiseParallel
        model_tp = parallelize_module(
            model_tp,
            device_mesh,
            {
                "layers.*.net[1]": ColwiseParallel(),
                "layers.*.net[2]": RowwiseParallel(),
            },
        )
        # 比较处理前后的模块
        self._compare_module(model, model_tp, inp_size, rank0_only=False)
# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 执行 run_tests() 函数，用于运行测试
    run_tests()
```