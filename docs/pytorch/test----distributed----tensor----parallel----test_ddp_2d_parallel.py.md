# `.\pytorch\test\distributed\tensor\parallel\test_ddp_2d_parallel.py`

```
# Owner(s): ["oncall: distributed"]

# 导入PyTorch相关模块
import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, DTensor, init_device_mesh, Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.distributed.tensor.parallel.ddp import _pre_dp_module_transform

# 导入分布式数据并行模块
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu

# 导入测试相关工具
from torch.testing._internal.common_utils import run_tests

# 导入分布式测试相关模块
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule,
    with_comms,
)

# Tensor-Parallel 的度
TP_DEGREE = 2
# 学习率
LR = 3e-5

# 初始化模型函数，设备类型为参数，默认模型并行大小为TP_DEGREE
def init_model(device_type, model_parallel_size=TP_DEGREE):
    # 设置随机种子
    torch.manual_seed(0)
    # 创建MLP模型实例
    model = MLPModule(device_type)
    # 设置随机种子
    torch.manual_seed(0)
    # 创建另一个MLP模型实例
    twod_model = MLPModule(device_type)
    # 使用分布式数据并行封装模型
    model = DDP(model)

    # 获取当前世界大小
    world_size = dist.get_world_size()
    # 创建2D设备网格，设备类型为参数，网格形状为[world_size // model_parallel_size, model_parallel_size]
    twod_mesh = DeviceMesh(
        device_type=device_type,
        mesh=torch.arange(0, world_size).view(-1, model_parallel_size),
    )
    # 初始化设备网格，设备类型为参数，网格形状为(world_size // model_parallel_size, model_parallel_size)，维度名称为("dp", "tp")
    mesh_2d = init_device_mesh(
        device_type,
        (world_size // model_parallel_size, model_parallel_size),
        mesh_dim_names=("dp", "tp"),
    )

    # 获取dp维度的处理组
    dp_pg = mesh_2d.get_group(mesh_dim=0)

    # 并行化计划，将"net1"使用列并行，"net2"使用行并行
    parallelize_plan = {
        "net1": ColwiseParallel(),
        "net2": RowwiseParallel(),
    }
    # 将twod_model并行化，使用mesh_2d的"tp"维度和parallelize_plan
    twod_model = parallelize_module(twod_model, mesh_2d["tp"], parallelize_plan)
    # 在数据并行模型变换前预处理twod_model
    _pre_dp_module_transform(twod_model)
    # TODO: 在使用梯度作为桶视图和静态图时添加测试的功能。
    # 使用dp_pg作为处理组，将twod_model包装在DDP中
    twod_model = DDP(twod_model, process_group=dp_pg)
    # 返回模型、二维模型和dp处理组
    return model, twod_model, dp_pg


# 测试2D并行集成类，继承自DTensorTestBase
class Test2dParallelIntegration(DTensorTestBase):
    # 检查模块方法，比较两个模型m1和m2的参数是否一致，支持梯度检查
    def _check_module(self, m1, m2, check_grad=False):
        # 获取m1的命名参数字典
        named_parameters = dict(m1.named_parameters())
        # 遍历m2的命名参数
        for name, param_m2 in m2.named_parameters():
            # 如果参数名不在命名参数字典中，则打印参数名和命名参数字典的键
            if name not in named_parameters:
                print(name, named_parameters.keys())
            # 断言参数名存在于命名参数字典中
            self.assertTrue(name in named_parameters)
            # 获取m1中对应参数名的参数
            param_m1 = named_parameters[name]
            # 如果需要检查梯度
            if check_grad:
                # 获取m2参数的梯度
                param_m2 = param_m2.grad
                # 获取m1参数的梯度
                param_m1 = param_m1.grad
            # 如果param_m2是DTensor类型
            if isinstance(param_m2, DTensor):
                # 创建Replicate实例
                replicate = [Replicate()]
                # 重新分配param_m2的设备网格，使用参数中的device_mesh和placements，转换为本地模式
                param_m2 = param_m2.redistribute(
                    device_mesh=param_m2.device_mesh, placements=replicate
                ).to_local()
            # 断言param_m2等于param_m1
            self.assertEqual(param_m2, param_m1)

    # 使用通信装饰器
    @with_comms
    # 如果GPU数小于4，则跳过测试
    @skip_if_lt_x_gpu(4)
    # 定义测试函数，用于测试二维分布式数据并行功能
    def test_2d_ddp_integration_functionality(self) -> None:
        # 初始化模型、二维模型和数据并行组
        model, twod_model, dp_pg = init_model(self.device_type)
        # 使用 Adam 优化器优化模型参数
        optim = torch.optim.Adam(model.parameters(), lr=LR)
        twod_optim = torch.optim.Adam(twod_model.parameters(), lr=LR)

        # 创建输入数据
        # 获取当前进程在数据并行组中的排名作为随机种子
        input_seed = dist.get_rank(dp_pg)
        torch.manual_seed(input_seed + 1)
        # 在指定设备上生成随机输入张量
        input = torch.rand(4, 10, device=self.device_type)

        # 对模型进行前向传播计算
        output = model(input)
        twod_output = twod_model(input)
        # 断言两个输出张量是否相等
        self.assertEqual(output, twod_output)

        # 对输出张量的所有元素进行求和，并执行反向传播
        output.sum().backward()
        twod_output.sum().backward()
        # 检查模型参数的梯度
        self._check_module(model, twod_model, check_grad=True)

        # 使用优化器执行一步参数更新
        optim.step()
        twod_optim.step()
        # 再次检查模型参数
        self._check_module(model, twod_model)

        # 使用新的随机种子重新生成输入数据
        torch.manual_seed(input_seed + 1004)
        input = torch.rand(16, 10, device=self.device_type)

        # 对新输入数据再次进行前向传播计算
        output = model(input)
        twod_output = twod_model(input)
        # 断言两个输出张量是否相等
        self.assertEqual(output, twod_output)

        # TODO: Add save/load of 2D verification.
        # TODO: 添加二维验证模型的保存和加载功能的实现。
# 如果当前脚本作为主程序运行（而不是被导入为模块），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```