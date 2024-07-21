# `.\pytorch\test\distributed\test_symmetric_memory.py`

```py
# Owner(s): ["module: c10d"]

# 导入必要的模块和库
import torch

import torch.distributed as dist
from torch._C._distributed_c10d import _SymmetricMemory
from torch.distributed._symmetric_memory import (
    _fused_all_gather_matmul_fallback,
    _fused_matmul_reduce_scatter_fallback,
    enable_symm_mem_for_group,
    restride_A_for_fused_matmul_reduce_scatter,
    restride_A_shard_for_fused_all_gather_matmul,
)

# 导入测试相关的模块和类
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    skipIfRocm,
)

# 检查是否需要 CUDA Peer-to-Peer 访问能力
def requires_cuda_p2p_access():
    cuda_p2p_access_available = (
        torch.cuda.is_available() and torch.cuda.device_count() >= 2
    )
    num_devices = torch.cuda.device_count()
    for i in range(num_devices - 1):
        for j in range(i + 1, num_devices):
            if not torch.cuda.can_device_access_peer(i, j):
                cuda_p2p_access_available = False
                break
        if not cuda_p2p_access_available:
            break

    # 在 Sandcastle 环境下跳过测试，如果 CUDA Peer-to-Peer 访问不可用
    return skip_but_pass_in_sandcastle_if(
        not cuda_p2p_access_available,
        "cuda p2p access is not available",
    )

# 实例化参数化测试，并要求 CUDA Peer-to-Peer 访问能力
@instantiate_parametrized_tests
@requires_cuda_p2p_access()
class SymmetricMemoryTest(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return 2

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.rank}")

    # 初始化进程方法
    def _init_process(self):
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        enable_symm_mem_for_group(dist.group.WORLD.group_name)

    # 验证对称内存操作的方法
    def _verify_symmetric_memory(self, symm_mem):
        self.assertEqual(symm_mem.world_size, 2)

        # 获取缓冲区并填充数据
        buf = symm_mem.get_buffer(0, (64, 64), torch.float32)
        if symm_mem.rank == 0:
            symm_mem.wait_signal(src_rank=1)
            self.assertTrue(buf.eq(42).all())
        else:
            buf.fill_(42)
            symm_mem.put_signal(dst_rank=0)

        symm_mem.barrier()

        # 进行数据同步和验证
        if symm_mem.rank == 0:
            symm_mem.barrier()
            self.assertTrue(buf.eq(43).all())
        else:
            buf.fill_(43)
            symm_mem.barrier()

        symm_mem.barrier()

    # 在 ROCm 平台下跳过测试
    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    # 定义测试函数，用于检测 CUDA NVLink 连通性
    def test_cuda_nvlink_connectivity_detection(self) -> None:
        # 导入必要的模块和函数
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _detect_dma_connectivity
        
        # 调用 _detect_dma_connectivity 函数，检测 CUDA 设备的 NVLink 连通性
        connectivity = _detect_dma_connectivity(DeviceType.CUDA, "nvlink")
        
        # 断言连接类型为 CUDA
        self.assertEqual(connectivity.device_type, DeviceType.CUDA)
        # 断言连接方式为 nvlink
        self.assertEqual(connectivity.connection_type, "nvlink")
        # 断言连接矩阵的行数与 CUDA 设备数相等
        self.assertEqual(len(connectivity.matrix), torch.cuda.device_count())
        
        # 遍历连接矩阵的每一行
        for row in connectivity.matrix:
            # 断言每一行的列数与 CUDA 设备数相等
            self.assertEqual(len(row), torch.cuda.device_count())

    # 使用装饰器 skipIfRocm 跳过在 ROCm 环境下运行的测试，并要求至少有两个 GPU
    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    # 测试空的分块 p2p 通信
    def test_empty_strided_p2p(self) -> None:
        # 初始化测试过程
        self._init_process()

        # 定义张量的形状、步幅、数据类型、设备以及分组名称
        shape = (64, 64)
        stride = (64, 1)
        dtype = torch.float32
        device = self.device
        group_name = "0"
        alloc_args = (shape, stride, dtype, device, group_name)

        # 创建一个空张量并验证会抛出 RuntimeError 异常
        t = torch.empty(shape, dtype=dtype, device=device)
        with self.assertRaises(RuntimeError):
            _SymmetricMemory.rendezvous(t)

        # 调用 _SymmetricMemory.empty_strided_p2p 函数创建分块 p2p 张量
        t = _SymmetricMemory.empty_strided_p2p(*alloc_args)
        # 进行分块内存的聚会操作，得到 symm_mem 对象
        symm_mem = _SymmetricMemory.rendezvous(t)

        # 删除临时张量对象 t
        del t
        # 验证分块内存 symm_mem 的正确性
        self._verify_symmetric_memory(symm_mem)
        # 销毁进程组
        dist.destroy_process_group()

    # 使用装饰器 skipIfRocm 跳过在 ROCm 环境下运行的测试，并要求至少有两个 GPU
    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    # 测试持久化的空分块 p2p 通信
    def test_empty_strided_p2p_persistent(self) -> None:
        # 初始化测试过程
        self._init_process()

        # 定义张量的形状、步幅、数据类型、设备、分组名称以及分配 ID
        shape = (64, 64)
        stride = (64, 1)
        dtype = torch.float32
        device = self.device
        alloc_id = 42  # Persistent allocation
        group_name = "0"
        alloc_args = (shape, stride, dtype, device, group_name, alloc_id)

        # 调用 _SymmetricMemory.empty_strided_p2p 函数创建持久化的分块 p2p 张量
        t = _SymmetricMemory.empty_strided_p2p(*alloc_args)
        # 获取张量的数据指针
        data_ptr = t.data_ptr()

        # 验证如果具有相同的 alloc_id 且存在活跃分块时，持久化分块的创建会抛出 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            _SymmetricMemory.empty_strided_p2p(*alloc_args)

        # 验证如果具有相同的 alloc_id 但没有活跃分块时，持久化分块的创建会成功，并且返回的张量具有相同的数据指针
        del t
        t = _SymmetricMemory.empty_strided_p2p(*alloc_args)
        self.assertEqual(t.data_ptr(), data_ptr)

        # 验证在聚会之前调用 _SymmetricMemory.get_symmetric_memory 会抛出 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            _SymmetricMemory.get_symmetric_memory(t)

        # 进行分块内存的聚会操作，得到 symm_mem_0 和 symm_mem_1
        symm_mem_0 = _SymmetricMemory.rendezvous(t)
        symm_mem_1 = _SymmetricMemory.get_symmetric_memory(t)
        # 断言 symm_mem_0 和 symm_mem_1 对象的 ID 相同
        self.assertEqual(id(symm_mem_0), id(symm_mem_1))

        # 验证分块内存 symm_mem_0 的正确性
        self._verify_symmetric_memory(symm_mem_0)
        # 销毁进程组
        dist.destroy_process_group()

    # 使用装饰器 skipIfRocm 跳过在 ROCm 环境下运行的测试，并要求至少有两个 GPU
    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    # 参数化测试，gather_dim 可以是 0 或 1
    @parametrize("gather_dim", [0, 1])
    # 定义一个测试方法，用于测试融合的全收集矩阵乘法
    def test_fused_all_gather_matmul(self, gather_dim: int) -> None:
        # 初始化当前进程组的环境
        self._init_process()

        # 定义矩阵维度和大小的常量
        B = 8    # 批处理大小
        M = 64   # 矩阵行数
        N = 16   # 矩阵列数
        K = 32   # 矩阵维度
        group = dist.group.WORLD   # 获取世界级别的分布式组
        rank = self.rank           # 当前进程的排名
        world_size = self.world_size   # 获取世界中的进程数

        # 设置随机种子
        torch.manual_seed(42 + rank)

        # 在 CUDA 设备上生成随机数据矩阵 A_shard
        A_shard = torch.rand(B, M // self.world_size, K, device="cuda")

        # 生成多个随机数据矩阵 Bs
        Bs = [torch.rand(K, N, device="cuda") for _ in range(3)]

        # 调用后备的融合全收集矩阵乘法函数
        ag_output_0, mm_outputs_0 = _fused_all_gather_matmul_fallback(
            A_shard, Bs, gather_dim=gather_dim, group_name=group.group_name
        )

        # 调用优化的融合全收集矩阵乘法函数
        ag_output_1, mm_outputs_1 = torch.ops.symm_mem.fused_all_gather_matmul(
            A_shard, Bs, gather_dim=gather_dim, group_name=group.group_name
        )

        # 断言两种实现方法的输出结果是否一致
        assert torch.allclose(ag_output_0, ag_output_1)
        assert ag_output_0.stride() == ag_output_1.stride()

        # 检查多个矩阵乘法结果是否一致
        for mm_output_0, mm_output_1 in zip(mm_outputs_0, mm_outputs_1):
            assert torch.allclose(mm_output_0, mm_output_1)
            assert mm_output_0.stride(), mm_output_1.stride()

        # 销毁进程组
        dist.destroy_process_group()

    # 跳过 ROCm 平台的测试装饰器
    @skipIfRocm
    # 跳过 GPU 数量小于 2 的测试装饰器
    @skip_if_lt_x_gpu(2)
    # 参数化测试方法，用于测试融合矩阵乘法的减少分散
    @parametrize("scatter_dim", [0, 1])
    def test_fused_matmul_reduce_scatter(self, scatter_dim: int) -> None:
        # 初始化当前进程组的环境
        self._init_process()

        # 定义矩阵维度和大小的常量
        B = 8    # 批处理大小
        M = 64   # 矩阵行数
        N = 16   # 矩阵列数
        K = 32   # 矩阵维度
        group = dist.group.WORLD   # 获取世界级别的分布式组
        rank = self.rank           # 当前进程的排名
        world_size = self.world_size   # 获取世界中的进程数

        # 设置随机种子
        torch.manual_seed(42 + rank)

        # 在 CUDA 设备上生成随机数据矩阵 A 和 B
        A = torch.rand(B, M, K, device="cuda")
        B = torch.rand(K, N, device="cuda")

        # 调用后备的融合矩阵乘法减少分散函数
        output_0 = _fused_matmul_reduce_scatter_fallback(
            A, B, "avg", scatter_dim=scatter_dim, group_name=group.group_name
        )

        # 调用优化的融合矩阵乘法减少分散函数
        output_1 = torch.ops.symm_mem.fused_matmul_reduce_scatter(
            A, B, "avg", scatter_dim=scatter_dim, group_name=group.group_name
        )

        # 断言两种实现方法的输出结果是否一致
        assert torch.allclose(output_0, output_1)
        assert output_0.stride() == output_1.stride()

        # 销毁进程组
        dist.destroy_process_group()

    # 参数化测试方法，用于测试优化布局
    @parametrize("dim", [0, 1, 2])
    def test_optimal_layout(self, dim: int) -> None:
        # 生成一个随机张量 t
        t = torch.rand(8, 64, 32, 16)

        # 对于融合全收集矩阵乘法，重新排列张量 t 的布局
        x = restride_A_shard_for_fused_all_gather_matmul(t, dim)
        self.assertTrue(x.movedim(dim, 0).is_contiguous())   # 断言重新排列后的张量是否是连续的
        self.assertTrue(torch.allclose(x, t))   # 断言重新排列后的张量与原始张量 t 是否一致

        # 对于融合矩阵乘法减少分散，重新排列张量 t 的布局
        x = restride_A_for_fused_matmul_reduce_scatter(t, dim)
        self.assertTrue(x.movedim(dim, 0).is_contiguous())   # 断言重新排列后的张量是否是连续的
        self.assertTrue(torch.allclose(x, t))   # 断言重新排列后的张量与原始张量 t 是否一致
# 如果当前脚本被直接运行而不是被导入作为模块，则执行 run_tests 函数
if __name__ == "__main__":
    run_tests()
```