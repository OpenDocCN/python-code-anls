# `.\pytorch\test\distributed\_tensor\experimental\test_tp_transform.py`

```py
# 导入必要的模块和类
from collections import defaultdict  # 导入默认字典模块
from typing import Dict  # 导入字典类型提示

import torch  # 导入PyTorch库
from torch.distributed._tensor.experimental.tp_transform import (
    tensor_parallel_transformation,  # 导入张量并行转换函数
)
from torch.distributed.tensor.parallel.style import (
    ColwiseParallel,  # 导入列并行风格类
    ParallelStyle,  # 导入并行风格基类
    RowwiseParallel,  # 导入行并行风格类
)
from torch.testing._internal.common_utils import run_tests  # 导入运行测试的函数
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,  # 导入分布式张量测试基类
    with_comms,  # 导入包含通信的装饰器函数
)


class MLPListModule(torch.nn.Module):
    """
    A dummy model with list of MLPs.
    """

    def __init__(self, num_mlps=3, bias=True):
        super().__init__()
        self.mlps = torch.nn.ModuleList()  # 初始化空的模块列表
        for _ in range(num_mlps):
            # 向模块列表中添加多层感知机模型
            self.mlps.append(
                torch.nn.Sequential(
                    torch.nn.Linear(6, 18),  # 添加线性层，输入大小为6，输出大小为18
                    torch.nn.ReLU(),  # ReLU激活函数
                    torch.nn.Linear(18, 6, bias=bias),  # 添加线性层，输入大小为18，输出大小为6，可选择是否有偏置
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.chunk(x, 2, dim=1)[0]  # 按列将张量x分块，取第一块
        for mlp in self.mlps:
            x = mlp(x)  # 应用模块列表中的每个多层感知机模型到x上
        return x + torch.ones_like(x)  # 返回x加上与x相同形状的全为1的张量


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 5)  # 添加线性层，输入大小为3，输出大小为5
        self.bn = torch.nn.BatchNorm1d(5)  # 添加批标准化层，参数大小为5

    def forward(self, x):
        return self.bn(self.fc(x))  # 返回批标准化层应用于线性层的结果


class TensorParallelTest(DTensorTestBase):
    def setUp(self) -> None:
        super().setUp()  # 调用父类的setUp方法

    def assert_has_c10d_ops(
        self, gm: torch.fx.GraphModule, expected_ops_count: Dict[str, int]
    ) -> None:
        actual_ops_count: Dict[str, int] = defaultdict(int)  # 初始化实际操作计数的默认字典
        for node in gm.graph.nodes:
            if node.op == "call_function":  # 如果节点的操作为"call_function"
                if "c10d_functional" in str(node.target):  # 如果目标函数名中包含"c10d_functional"
                    actual_ops_count[str(node.target)] += 1  # 增加目标函数名对应的计数
        self.assertDictEqual(expected_ops_count, actual_ops_count)  # 断言期望的操作计数与实际计数一致

    @with_comms
    def test_tp_transform_with_uncovered_op(self):
        # 创建一个虚拟的 DummyModel，并将其移动到指定的设备类型
        model = DummyModel().to(device=self.device_type)
        # 创建一个包含随机数据的输入张量，确保不需要梯度计算，并移动到指定设备类型
        inputs = (torch.randn(7, 3, requires_grad=False).to(device=self.device_type),)
        # 在没有梯度计算的上下文中，对模型进行推理
        with torch.no_grad():
            # 调用模型进行推理，生成输出结果
            res = model(*inputs)
            # 导出模型及其输入，执行分解运算
            exported_program = torch.export.export(
                model,
                inputs,
            ).run_decompositions()
        # 对导出的程序进行张量并行转换，使用指定的并行策略和设备信息
        tp_exported_program = tensor_parallel_transformation(
            exported_program,
            self.rank,
            self.world_size,
            self.device_type,
            {"fc": ColwiseParallel},
        )
        # 获取转换后的模型
        tp_model = tp_exported_program.module()
        # 再次在没有梯度计算的上下文中，对转换后的模型进行推理
        with torch.no_grad():
            tp_res = tp_model(*inputs)
        # 断言原始模型和转换后模型的输出结果相等
        self.assertEqual(res, tp_res)
        # 断言在图模块中插入了适用于分布式共享的全局聚合操作
        self.assert_has_c10d_ops(
            tp_exported_program.graph_module,
            {
                "_c10d_functional.all_gather_into_tensor.default": 1,
                "_c10d_functional.wait_tensor.default": 1,
            },
        )

    @with_comms
    def test_tp_transform_e2e(self):
        # 设置随机种子为0
        torch.manual_seed(0)
        # 创建一个 MLPListModule 模型实例，并将其移动到指定的设备类型
        model = MLPListModule(2).to(device=self.device_type)
        # 创建一个包含随机数据的输入张量，并移动到指定设备类型
        inputs = (torch.randn((10, 12)).to(device=self.device_type),)
        # 定义并行策略的字典，指定每个子模块的并行方式
        parallel_strategies: Dict[str, ParallelStyle] = {
            "mlps.0.0": ColwiseParallel,
            "mlps.0.2": RowwiseParallel,
            "mlps.1.0": ColwiseParallel,
            "mlps.1.2": RowwiseParallel,
        }

        # 进入推理模式，不跟踪梯度
        with torch.inference_mode():
            # 调用模型进行推理，生成输出结果
            res = model(*inputs)
            # 导出模型及其输入，执行分解运算
            exported_program = torch.export.export(
                model,
                inputs,
            ).run_decompositions()
        # 对导出的程序进行张量并行转换，使用指定的并行策略和设备信息
        tp_exported_program = tensor_parallel_transformation(
            exported_program,
            self.rank,
            self.world_size,
            self.device_type,
            parallel_strategies,
        )
        # 获取转换后的模型
        tp_model = tp_exported_program.module()
        # 再次进入推理模式，不跟踪梯度
        with torch.inference_mode():
            # 调用转换后的模型进行推理，生成输出结果
            tp_res = tp_model(*inputs)
        # 断言原始模型和转换后模型的输出结果相等
        self.assertEqual(res, tp_res)
        # 断言在图模块中插入了适用于每个 MLP 结束时的全局归约操作
        self.assert_has_c10d_ops(
            tp_exported_program.graph_module,
            {
                "_c10d_functional.all_reduce.default": 2,
                "_c10d_functional.wait_tensor.default": 2,
            },
        )
    # 定义一个名为 test_tp_transform_no_bias 的测试函数，测试无偏置情况下的张量并行转换
    def test_tp_transform_no_bias(self):
        # 设置随机种子为0，以便结果可重复
        torch.manual_seed(0)
        # 创建一个没有偏置的MLPListModule模型实例，并将其移到指定的设备上
        model = MLPListModule(1, bias=False).to(device=self.device_type)
        # 构造模型输入数据，大小为(10, 12)，并移动到指定设备上
        inputs = (torch.randn((10, 12)).to(device=self.device_type),)
        # 定义并初始化一个并行策略字典，指定每个子模块使用的并行方式
        parallel_strategies: Dict[str, ParallelStyle] = {
            "mlps.0.0": ColwiseParallel,
            "mlps.0.2": RowwiseParallel,
        }

        # 进入推断模式，不进行梯度计算
        with torch.inference_mode():
            # 对模型进行前向推断，获取输出结果 res
            res = model(*inputs)
            # 导出模型及其输入，并运行分解操作，得到导出的程序对象 exported_program
            exported_program = torch.export.export(
                model,
                inputs,
            ).run_decompositions()
        
        # 使用张量并行转换函数对导出的程序进行张量并行转换
        tp_exported_program = tensor_parallel_transformation(
            exported_program,
            self.rank,
            self.world_size,
            self.device_type,
            parallel_strategies,
        )
        
        # 获取转换后的模型实例
        tp_model = tp_exported_program.module()
        
        # 再次进入推断模式
        with torch.inference_mode():
            # 对转换后的模型进行前向推断，获取输出结果 tp_res
            tp_res = tp_model(*inputs)
        
        # 断言原始模型输出结果与转换后模型输出结果相等
        self.assertEqual(res, tp_res)
        
        # 断言导出的程序中包含指定的C10D操作
        self.assert_has_c10d_ops(
            tp_exported_program.graph_module,
            {
                "_c10d_functional.all_reduce.default": 1,
                "_c10d_functional.wait_tensor.default": 1,
            },
        )
# 如果这个模块被直接执行（而不是被导入到其它模块中），则运行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```