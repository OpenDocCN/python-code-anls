# `.\pytorch\test\distributed\tensor\parallel\test_tp_style.py`

```
# 从 copy 模块中导入 deepcopy 函数
from copy import deepcopy

# 导入 PyTorch 的核心模块
import torch
import torch.nn as nn

# 导入与分布式张量相关的模块和类
from torch.distributed._tensor import (
    distribute_tensor,
    DTensor,
    init_device_mesh,
    Replicate,
    Shard,
)
# 导入调试模式相关的类和函数
from torch.distributed._tensor.debug import CommDebugMode
# 导入分布式张量的部署类型相关模块
from torch.distributed._tensor.placement_types import _Partial
# 导入张量并行化的模块
from torch.distributed.tensor.parallel import parallelize_module
# 导入张量并行化的风格模块
from torch.distributed.tensor.parallel.style import (
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
)
# 导入用于测试的实用函数
from torch.testing._internal.common_utils import run_tests
# 导入分布式张量测试相关的基类和辅助函数
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    NUM_DEVICES,
    RMSNormPython,
    with_comms,
)

# 导入 C10D 功能操作符的接口
c10d_functional = torch.ops.c10d_functional

# 定义测试类，继承自 DTensorTestBase
class TensorParallelStyleTest(DTensorTestBase):

    # 定义属性方法，返回设备数量
    @property
    def world_size(self):
        return NUM_DEVICES

    # 使用装饰器，执行带有通信的测试函数
    @with_comms
    def test_colwise_parallel_style(self):
        # 初始化设备网格，并指定设备类型和设备数量
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # 创建通信调试模式的实例
        comm_mode = CommDebugMode()
        # 在指定设备上创建随机张量，要求梯度计算
        tensor = torch.rand(8, 16, device=self.device_type, requires_grad=True)
        # 创建在指定设备上的线性模型
        model = nn.Linear(16, 16, device=self.device_type)

        # 使用列并行的默认风格
        default_col_parallel = ColwiseParallel()
        # 并行化模型，指定设备网格和并行风格
        colwise_mod = parallelize_module(deepcopy(model), mesh, default_col_parallel)
        # 进入通信调试模式
        with comm_mode:
            # 对并行化后的模型进行前向传播
            out = colwise_mod(tensor)
            # 确保输出在最后一个维度上被分片
            self.assertEqual(out.shape, (8, 16 // self.world_size))
            # 确保前向传播过程中没有通信发生
            self.assertEqual(comm_mode.get_total_counts(), 0)

            # 对输出进行求和并执行反向传播
            out.sum().backward()
            # 在反向传播中执行全局归约操作
            self.assertEqual(comm_mode.get_comm_counts()[c10d_functional.all_reduce], 1)
            self.assertEqual(comm_mode.get_total_counts(), 1)

        # 使用分片的列并行风格
        sharded_col_parallel = ColwiseParallel(input_layouts=Shard(0))
        # 再次并行化模型，使用分片的列并行风格
        colwise_mod = parallelize_module(deepcopy(model), mesh, sharded_col_parallel)
        # 进入通信调试模式
        with comm_mode:
            # 对并行化后的模型进行前向传播
            out = colwise_mod(tensor)
            # 确保输出在最后一个维度上被分片
            self.assertEqual(out.shape, (8 * self.world_size, 16 // self.world_size))
            # 确保前向传播过程中执行了全局聚集操作
            self.assertEqual(
                comm_mode.get_comm_counts()[c10d_functional.all_gather_into_tensor], 1
            )
            self.assertEqual(comm_mode.get_total_counts(), 1)

            # 对输出进行求和并执行反向传播
            out.sum().backward()
            # 在反向传播中执行全局减少分散操作
            self.assertEqual(
                comm_mode.get_comm_counts()[c10d_functional.reduce_scatter_tensor], 1
            )
            self.assertEqual(comm_mode.get_total_counts(), 2)

    @with_comms
    def test_colwise_parallel_embedding(self):
        # 初始化设备网格，设备类型和世界大小决定网格的初始化
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # 使用调试通信模式
        comm_mode = CommDebugMode()

        # 创建一个张量并在指定设备上进行初始化
        tensor = torch.arange(8, device=self.device_type).reshape(4, 2)

        # 创建一个包含16个词嵌入向量的Embedding模型
        model = nn.Embedding(16, 16, device=self.device_type)

        # 创建默认的按列并行处理对象
        default_col_parallel = ColwiseParallel()

        # 使用并行化模块将深拷贝后的模型与设备网格和列并行对象进行并行化处理
        colwise_mod = parallelize_module(deepcopy(model), mesh, default_col_parallel)

        # 进入通信模式
        with comm_mode:
            # 将张量输入并行化后的模型，生成输出
            out = colwise_mod(tensor)
            # 确保输出在最后一个维度上进行分片
            self.assertEqual(out.shape, (4, 2, 16 // self.world_size))
            # 确保前向传播中没有通信发生
            self.assertEqual(comm_mode.get_total_counts(), 0)

            # 对输出结果进行求和并进行反向传播
            out.sum().backward()
            # 后向传播中没有通信发生
            self.assertEqual(comm_mode.get_total_counts(), 0)

    @with_comms
    def test_rowwise_parallel_style(self):
        # 初始化设备网格，设备类型和世界大小决定网格的初始化
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # 使用调试通信模式
        comm_mode = CommDebugMode()

        # 创建一个在指定设备上带有梯度的随机张量
        tensor = torch.rand(
            8, 16 // self.world_size, device=self.device_type, requires_grad=True
        )

        # 创建一个线性层模型
        model = nn.Linear(16, 16, device=self.device_type)

        # 创建默认的按行并行处理对象
        default_row_parallel = RowwiseParallel()

        # 使用并行化模块将深拷贝后的模型与设备网格和行并行对象进行并行化处理
        rowwise_mod = parallelize_module(deepcopy(model), mesh, default_row_parallel)

        # 进入通信模式
        with comm_mode:
            # 将张量输入并行化后的模型，生成输出
            out = rowwise_mod(tensor)
            # 确保输出结果被复制
            self.assertEqual(out.shape, (8, 16))
            # 前向传播中进行全局归约通信
            self.assertEqual(comm_mode.get_comm_counts()[c10d_functional.all_reduce], 1)
            self.assertEqual(comm_mode.get_total_counts(), 1)

            # 对输出结果进行求和并进行反向传播
            out.sum().backward()
            # 后向传播中没有通信发生
            self.assertEqual(comm_mode.get_total_counts(), 1)

        # 创建具有输出布局Shard(0)的按行并行处理对象
        sharded_row_parallel = RowwiseParallel(output_layouts=Shard(0))

        # 使用并行化模块将深拷贝后的模型与设备网格和具有Shard(0)输出布局的行并行对象进行并行化处理
        rowwise_mod = parallelize_module(deepcopy(model), mesh, sharded_row_parallel)

        # 进入通信模式
        with comm_mode:
            # 将张量输入并行化后的模型，生成输出
            out = rowwise_mod(tensor)
            # 确保输出结果被复制
            self.assertEqual(out.shape, (8 // self.world_size, 16))
            # 前向传播中进行降低散射通信
            self.assertEqual(
                comm_mode.get_comm_counts()[c10d_functional.reduce_scatter_tensor], 1
            )
            self.assertEqual(comm_mode.get_total_counts(), 1)

            # 对输出结果进行求和并进行反向传播
            out.sum().backward()
            # 后向传播中进行全局聚集通信
            self.assertEqual(
                comm_mode.get_comm_counts()[c10d_functional.all_gather_into_tensor], 1
            )
            self.assertEqual(comm_mode.get_total_counts(), 2)

    @with_comms
    def test_rowwise_parallel_embedding(self):
        # 初始化设备网格，根据设备类型和世界大小创建网格
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # 使用调试通信模式
        comm_mode = CommDebugMode()

        # 创建一个张量并在指定设备上初始化
        tensor = torch.arange(8, device=self.device_type).reshape(4, 2)

        # 创建一个指定设备上的嵌入模型
        model = nn.Embedding(16, 16, device=self.device_type)

        # 将模型并行化，使用行并行，输入布局为复制
        rowwise_mod = parallelize_module(
            deepcopy(model), mesh, RowwiseParallel(input_layouts=Replicate())
        )

        # 进入通信模式
        with comm_mode:
            # 在模型上进行前向传播
            out = rowwise_mod(tensor)
            # 确保输出张量形状符合预期
            self.assertEqual(out.shape, (4, 2, 16))
            # 确保在前向传播中发生了全reduce通信
            self.assertEqual(comm_mode.get_total_counts(), 1)
            self.assertEqual(comm_mode.get_comm_counts()[c10d_functional.all_reduce], 1)

            # 对输出进行反向传播
            out.sum().backward()
            # 后向传播中没有通信
            self.assertEqual(comm_mode.get_total_counts(), 1)

        # 创建具有分片行并行输出布局的行并行对象
        sharded_row_parallel = RowwiseParallel(
            input_layouts=Replicate(), output_layouts=Shard(1)
        )

        # 将模型并行化，使用分片行并行
        rowwise_mod = parallelize_module(deepcopy(model), mesh, sharded_row_parallel)

        # 创建输入索引张量
        inp_indices = torch.arange(8, device=self.device_type)
        # 进入通信模式
        with comm_mode:
            # 在模型上进行前向传播
            out = rowwise_mod(inp_indices)
            # 确保输出张量形状符合预期
            self.assertEqual(out.shape, (8, 16 // self.world_size))
            # 确保在前向传播中进行了reduce scatter通信
            self.assertEqual(comm_mode.get_total_counts(), 1)
            self.assertEqual(
                comm_mode.get_comm_counts()[c10d_functional.reduce_scatter_tensor], 1
            )
            # 对输出进行反向传播
            out.sum().backward()
            # 后向传播中进行了allgather通信
            self.assertEqual(comm_mode.get_total_counts(), 2)
            self.assertEqual(
                comm_mode.get_comm_counts()[c10d_functional.all_gather_into_tensor], 1
            )

    @with_comms
    def test_prepare_module_input(self):
        # 初始化设备网格，根据设备类型和世界大小创建网格
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # 创建一个指定设备上的全1张量
        tensor = torch.ones(2, 16, device=self.device_type)
        # 创建预期的全1张量，形状是原来的两倍世界大小
        expected_tensor = torch.ones(2 * self.world_size, 16, device=self.device_type)

        # 准备模块输入，输入布局为分片0，期望输入布局为复制
        prepare_inp_style = PrepareModuleInput(
            input_layouts=Shard(0), desired_input_layouts=Replicate()
        )

        # 创建一个身份模型
        model = nn.Identity()
        # 将模型并行化，使用准备模块输入样式
        allgather_mod = parallelize_module(model, mesh, prepare_inp_style)

        # 在模型上进行前向传播并获取完整张量
        output = allgather_mod(tensor).full_tensor()
        # 确保输出张量与预期张量相等
        self.assertEqual(output, expected_tensor)
    def test_prepare_module_input_multiple_inputs(self):
        # 初始化设备网格，根据给定的设备类型和世界大小
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # 定义一个测试用的神经网络模块类
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个线性层，输入维度为8，输出维度为8
                self.linear = torch.nn.Linear(8, 8)

            def forward(self, x, y):
                # 模块的前向传播，返回线性层输入x的输出加上y
                return self.linear(x) + y

        # 如果输入布局和期望输入布局的长度不同，抛出断言错误
        test_mod = TestModule().to(self.device_type)
        with self.assertRaisesRegex(
            AssertionError,
            "input_layouts and desired_input_layouts should have same length!",
        ):
            # 准备模块输入对象，抛出维度不匹配的错误
            prepare_inps_dimension_mismatch = PrepareModuleInput(
                input_layouts=Shard(0), desired_input_layouts=(Replicate(), None)
            )
        
        # 如果模块的输入和输入布局的长度不同，抛出断言错误
        prepare_inps_short_dimension = PrepareModuleInput(
            input_layouts=Shard(0), desired_input_layouts=Replicate()
        )

        # 并行化线性层模块
        parallelize_module(test_mod.linear, mesh, ColwiseParallel())
        # 并行化整个测试模块，抛出值错误，如果模块的输入和输入布局的长度不同
        parallelize_module(test_mod, mesh, prepare_inps_short_dimension)
        with self.assertRaisesRegex(
            ValueError, "module inputs and input_layouts should have same length!"
        ):
            output = test_mod(
                torch.randn(2, 8, device=self.device_type),
                torch.ones(
                    self.world_size * 2, 8 // self.world_size, device=self.device_type
                ),
            )

        # 重新初始化测试模块
        test_mod = TestModule().to(self.device_type)
        # 准备模块输入对象，输入布局包含Shard(0)和None，期望输入布局包含Replicate()和None
        prepare_inps = PrepareModuleInput(
            input_layouts=(Shard(0), None), desired_input_layouts=(Replicate(), None)
        )

        # 并行化线性层模块
        parallelize_module(test_mod.linear, mesh, ColwiseParallel())
        # 并行化整个测试模块，使用准备好的模块输入对象
        parallelize_module(test_mod, mesh, prepare_inps)
        # 对测试模块进行前向传播
        output = test_mod(
            torch.randn(2, 8, device=self.device_type),
            torch.ones(
                self.world_size * 2, 8 // self.world_size, device=self.device_type
            ),
        )
        # 断言输出的形状是否符合预期
        self.assertEqual(output.shape, (self.world_size * 2, 8 // self.world_size))

    @with_comms
    # 定义一个测试方法，用于准备模块关键字参数输入
    def test_prepare_module_kwargs_input(self):
        # 初始化设备网格，指定设备类型和世界大小
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # 定义一个测试用的模块，继承自torch.nn.Module
        class TestKwargModule(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 添加一个线性层
                self.linear = torch.nn.Linear(8, 8)

            # 前向传播方法，接收参数x，以及关键字参数y和z（y为必须关键字参数，z有默认值2）
            def forward(self, x, *, y, z=2):
                # 返回线性层的输出加上y和z
                return self.linear(x) + y + z

        # 创建一个TestKwargModule实例，并将其移到指定设备上
        test_mod = TestKwargModule().to(self.device_type)

        # 准备模块输入：定义输入关键字参数的布局和期望的输入关键字参数的布局
        prepare_inps_simple = PrepareModuleInput(
            input_kwarg_layouts={"y": Shard(0)},
            desired_input_kwarg_layouts={"y": Replicate()},
        )

        # 并行化模块中的线性层
        parallelize_module(
            test_mod.linear, mesh, ColwiseParallel(use_local_output=False)
        )

        # 并行化整个模块
        parallelize_module(test_mod, mesh, prepare_inps_simple)

        # 创建通信调试模式的实例
        comm_mode = CommDebugMode()

        # 进入通信调试模式
        with comm_mode:
            # 执行模块，传入输入数据和关键字参数y
            output = test_mod(
                torch.randn(1 * self.world_size, 8, device=self.device_type),
                y=torch.ones(1, 8, device=self.device_type),
            )

        # 断言总通信次数为1
        self.assertEqual(comm_mode.get_total_counts(), 1)
        # 断言输出的形状为(1 * self.world_size, 8)

        # 定义一个只有关键字参数的测试模块
        class TestKwargOnlyModule(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 添加一个线性层
                self.linear = torch.nn.Linear(8, 8)

            # 前向传播方法，只接收关键字参数x、y（有默认值2）、z（默认为None）
            def forward(self, *, x, y=2, z=None):
                # 返回线性层的输出加上y和z
                return self.linear(x) + y + z

        # 创建一个TestKwargOnlyModule实例，并将其移到指定设备上
        test_kwonly_mod = TestKwargOnlyModule().to(self.device_type)

        # 准备模块输入：定义输入关键字参数x和z的布局，以及期望的输入关键字参数的布局
        prepare_inps_simple = PrepareModuleInput(
            input_kwarg_layouts={"x": Shard(0), "z": Shard(0)},
            desired_input_kwarg_layouts={"x": Replicate(), "z": Replicate()},
        )

        # 并行化模块中的线性层
        parallelize_module(
            test_kwonly_mod.linear, mesh, ColwiseParallel(use_local_output=False)
        )

        # 并行化整个模块
        parallelize_module(test_kwonly_mod, mesh, prepare_inps_simple)

        # 进入通信调试模式
        with comm_mode:
            # 执行模块，传入关键字参数x和z
            output = test_kwonly_mod(
                x=torch.randn(1, 8, device=self.device_type),
                z=torch.ones(1, 8, device=self.device_type),
            )

        # 断言总通信次数为2
        self.assertEqual(comm_mode.get_total_counts(), 2)
        # 断言输出的形状为(1 * self.world_size, 8)

        # 测试当x为DTensor的情况
        x_dt = DTensor.from_local(
            torch.randn(1, 8, device=self.device_type), mesh, [Shard(0)]
        )

        # 进入通信调试模式
        with comm_mode:
            # 执行模块，传入DTensor类型的关键字参数x和普通的关键字参数z
            output = test_kwonly_mod(
                x=x_dt, z=torch.ones(1, 8, device=self.device_type)
            )

        # 断言总通信次数为2
        self.assertEqual(comm_mode.get_total_counts(), 2)
        # 断言输出的形状为(1 * self.world_size, 8)
    # 定义一个测试方法，用于测试准备模块输出功能
    def test_prepare_module_output(self):
        # 初始化设备网格，根据设备类型和世界大小进行初始化
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # 创建一个在指定设备上的张量，全为1，形状为(8, 16)
        tensor = torch.ones(8, 16, device=self.device_type)
        
        # 期望的张量，其行数为8除以世界大小后的结果，列数为16，设备类型同样为指定设备类型
        expected_tensor = torch.ones(8 // self.world_size, 16, device=self.device_type)
        
        # 创建一个准备模块输出的样式对象，使用Replicate作为输出布局，Shard(0)作为期望输出布局
        prepare_out_style = PrepareModuleOutput(
            output_layouts=Replicate(), desired_output_layouts=Shard(0)
        )

        # 创建一个简单的神经网络模型，使用nn.Identity作为身份映射模型
        model = nn.Identity()
        
        # 使用并行化方法parallelize_module将模型在设备网格上进行并行化处理，使用准备输出样式
        chunk_mod = parallelize_module(model, mesh, prepare_out_style)
        
        # 将输入张量tensor输入到并行化后的模块chunk_mod中，获取输出
        output = chunk_mod(tensor)
        
        # 使用断言验证输出是否与期望的张量expected_tensor相等
        self.assertEqual(output, expected_tensor)

    # 该方法使用了修饰器@with_comms，用于与通信相关的操作
# 如果当前脚本作为主程序运行（而非被导入为模块），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```