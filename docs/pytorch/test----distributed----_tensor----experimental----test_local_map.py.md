# `.\pytorch\test\distributed\_tensor\experimental\test_local_map.py`

```
# 导入PyTorch相关库和模块
import torch
import torch.distributed._functional_collectives as funcol
# 导入各种相关的分布式张量和测试工具
from torch.distributed._tensor import (
    distribute_tensor,
    DTensor,
    init_device_mesh,
    Replicate,
    Shard,
)
from torch.distributed._tensor.debug import CommDebugMode
from torch.distributed._tensor.experimental import local_map
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

# 获取C++ functional collectives函数接口
funcol_py = torch.ops.c10d_functional

# 定义一个函数，用于在所有设备上进行all-gather操作并检查相等性
def equal_allgather_forward(device_mesh, X, Y):
    # 创建一个张量，用于检查X和Y在当前设备上是否相等
    eq = torch.tensor([torch.equal(X, Y)], device=X.device)
    # 使用functional collectives进行all-gather操作，将结果汇总到设备网格上的所有设备
    eq_gather = funcol.all_gather_tensor(eq, 0, device_mesh)
    return torch.all(eq_gather).item()

# 定义一个函数，执行矩阵乘法并在所有设备上进行all-gather操作
def mm_all_gather_forward(device_mesh, A, B):
    # 在本地设备上执行矩阵乘法
    local_mm_result = torch.mm(A, B)
    # 使用functional collectives进行all-gather操作，将结果汇总到设备网格上的所有设备，并等待操作完成
    return funcol.all_gather_tensor(local_mm_result, 0, device_mesh).wait()

# 定义一个函数，执行矩阵乘法，这里不需要设备网格，因为不进行集体操作
def mm_forward(A, B):
    return torch.mm(A, B)

# 定义一个函数，执行矩阵乘法并在所有设备上进行all-reduce操作
def mm_allreduce_forward(device_mesh, A, B):
    # 在本地设备上执行矩阵乘法
    partial_sum_tensor = torch.mm(A, B)
    # 使用functional collectives进行all-reduce操作，将结果汇总到设备网格上的所有设备，并等待操作完成
    return funcol.all_reduce(partial_sum_tensor, "sum", device_mesh).wait()

# 定义一个函数，执行张量乘法，这里不需要设备网格，因为不进行集体操作
def mul_forward(X, scalar):
    return torch.mul(X, scalar)

# 定义一个测试类，继承自DTensorTestBase，用于测试local_map功能
class TestLocalMap(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    # 使用修饰器with_comms进行简单的正确性检查
    @with_comms
    def test_local_map_correctness(self):
        # 初始化设备网格，设备类型和网格形状
        device_mesh = init_device_mesh(
            device_type=self.device_type, mesh_shape=(self.world_size,)
        )
        # 使用调试通信模式
        comm_mode = CommDebugMode()

        # Y = W @ X
        # 创建随机张量 W 和 X，并进行矩阵乘法操作
        W = torch.randn(12, 8, device=self.device_type, requires_grad=False)
        X = torch.randn(8, 16, device=self.device_type, requires_grad=False)
        Y = torch.mm(W, X)

        # row-wise 和 col-wise 的分片位置定义在一维网格上
        row_wise = [Shard(0)]  # 行分片放置在一维网格上
        col_wise = [Shard(1)]  # 列分片放置在一维网格上
        replicate = [Replicate()]  # 复制到所有设备

        # 使用 distribute_tensor 函数分发张量 W 和 X 到设备网格上
        # W 在列方向分片，X 在行方向分片
        W_dt = distribute_tensor(
            W, device_mesh, col_wise
        )  # 将 W 张量在列方向分片
        X_dt = distribute_tensor(
            X, device_mesh, row_wise
        )  # 将 X 张量在行方向分片

        # 使用 local_map 函数对 mm_allreduce_forward 函数进行包装
        # local_mm_allreduce_forward 用于处理 DTensor 和 Tensor 转换
        local_mm_allreduce_forward = local_map(
            mm_allreduce_forward,
            out_placements=replicate,
            in_placements=(None, col_wise, row_wise),
            device_mesh=device_mesh,
        )

        # 在 comm_mode 下调用 local_mm_allreduce_forward 函数
        Y_dt = local_mm_allreduce_forward(device_mesh, W_dt, X_dt)

        # 验证通信模式的总计数是否为 1
        self.assertEqual(comm_mode.get_total_counts(), 1)
        # 检查输出的分片位置是否为 Replicate
        for placement in Y_dt.placements:
            self.assertTrue(placement.is_replicate())
        # 检查输出值是否正确
        self.assertEqual(Y_dt.to_local(), Y)
    def test_local_map_out_placements(self):
        # 测试 1: 使用 `out_placements` 将输出包装成 DTensor
        device_mesh = init_device_mesh(
            device_type=self.device_type, mesh_shape=(self.world_size,)
        )
        comm_mode = CommDebugMode()

        # 创建随机张量 X 和 Y
        X = torch.randn(8, 8, device=self.device_type, requires_grad=False)
        Y = torch.randn(8, 8, device=self.device_type, requires_grad=False)
        row_wise = [Shard(0)]
        
        # 将张量 X 和 Y 在设备网格上分发
        X_dt = distribute_tensor(X, device_mesh, row_wise)
        Y_dt = distribute_tensor(Y, device_mesh, row_wise)
        
        # 使用 local_map 函数将 equal_allgather_forward 函数应用于分布式张量
        local_equal_allgather_forward = local_map(
            equal_allgather_forward,
            out_placements=None,
        )
        
        # 启用通信调试模式
        with comm_mode:
            equal_dt = local_equal_allgather_forward(device_mesh, X_dt, Y_dt)  # 返回一个布尔值
        
        # 断言通信模式的调用次数为1
        self.assertEqual(comm_mode.get_total_counts(), 1)
        # 断言 equal_dt 为假
        self.assertTrue(not equal_dt)
        # 断言 X 和 Y 不相等
        self.assertTrue(not (X.equal(Y)))

        # 测试 2: 如果没有 DTensor 参数，直接返回 out
        # DDP 中的矩阵乘法
        replicate = [Replicate()]
        X = torch.randn(
            4 // self.world_size, 4, device=self.device_type, requires_grad=False
        )
        W = torch.randn(4, 4, device=self.device_type, requires_grad=False)
        
        # 使用 local_map 函数将 mm_all_gather_forward 函数应用于分布式张量
        local_mm_all_gather_forward = local_map(
            mm_all_gather_forward,
            out_placements=row_wise,
            in_placements=(None, row_wise, replicate),
        )
        
        # 启用通信调试模式
        with comm_mode:
            Y = local_mm_all_gather_forward(device_mesh, X, W)

        # 断言通信模式的调用次数为1
        self.assertEqual(comm_mode.get_total_counts(), 1)
        # 断言 comm_mode 中 all_gather_into_tensor 的通信次数为1
        self.assertEqual(
            comm_mode.get_comm_counts()[funcol_py.all_gather_into_tensor], 1
        )
        
        # 执行 funcol.all_gather_tensor 操作以获得 X 的复制
        X_replicate = funcol.all_gather_tensor(X, 0, device_mesh).wait()
        # 计算 Y_replicate
        Y_replicate = torch.mm(X_replicate, W)
        # 断言 Y 与 Y_replicate 相等
        self.assertEqual(Y, Y_replicate)  # Y 是一个 torch.Tensor

    # 检查 `in_placements` 处理
    @with_comms
    # 检查 `redistribute_inputs` 处理
    @with_comms
    def test_local_map_redistribute(self):
        # 初始化设备网格，根据设备类型和世界大小确定网格形状
        device_mesh = init_device_mesh(
            device_type=self.device_type, mesh_shape=(self.world_size,)
        )
        # 调试通信模式
        comm_mode = CommDebugMode()

        # Y = W @ X
        # 生成随机张量 W 和 X，并在指定设备上计算，不需要梯度
        W = torch.randn(12, 8, device=self.device_type, requires_grad=False)
        X = torch.randn(8, 16, device=self.device_type, requires_grad=False)
        Y = torch.mm(W, X)

        # 定义行列分片以及复制策略
        row_wise = [Shard(0)]  # 在一维网格上行分片放置
        col_wise = [Shard(1)]  # 在一维网格上列分片放置
        replicate = [Replicate()]  # 复制数据到所有设备

        # 对 W 和 X 张量进行分布式张量分发
        W_dt = distribute_tensor(
            W, device_mesh, row_wise
        )  # 将 W 张量按行分片并分布到设备网格上
        X_dt = distribute_tensor(
            X, device_mesh, col_wise
        )  # 将 X 张量按列分片并分布到设备网格上

        # 测试1：允许输入重分布
        # 使用 local_map 函数对 mm_allreduce_forward 函数进行本地映射，设置输入重分布为真
        local_mm_allreduce_forward = local_map(
            mm_allreduce_forward,
            out_placements=replicate,
            in_placements=(None, col_wise, row_wise),
            device_mesh=device_mesh,
            redistribute_inputs=True,
        )
        with comm_mode:
            # 在调试通信模式下执行 local_mm_allreduce_forward 函数
            Y_dt = local_mm_allreduce_forward(device_mesh, W_dt, X_dt)

        # 断言：期望通信次数为输入重分布2次加上输出1次，总共3次
        self.assertEqual(comm_mode.get_total_counts(), 3)
        # 检查 Y_dt 的放置策略是否为复制到所有设备
        for placement in Y_dt.placements:
            self.assertTrue(placement.is_replicate())
        # 断言：Y_dt 在本地上的值等于预期的 Y 值
        self.assertEqual(Y_dt.to_local(), Y)

        # 测试2：不允许输入重分布
        # 使用 local_map 函数对 mm_allreduce_forward 函数进行本地映射，设置输入重分布为假
        local_mm_allreduce_forward = local_map(
            mm_allreduce_forward,
            out_placements=replicate,
            in_placements=(None, col_wise, row_wise),
            device_mesh=device_mesh,
            redistribute_inputs=False,
        )
        # 使用断言检查是否引发 ValueError，并包含特定的错误消息
        with self.assertRaisesRegex(ValueError, "set redistribute_inputs=True"):
            Y_dt = local_mm_allreduce_forward(device_mesh, W_dt, X_dt)
# 如果当前脚本被直接执行（而非被导入到其他脚本中），则执行下面的代码块
if __name__ == "__main__":
    # 调用名为 run_tests 的函数，用于执行测试或者其他任务
    run_tests()
```