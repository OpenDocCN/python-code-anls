# `.\pytorch\test\distributed\fsdp\test_shard_utils.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入 PyTorch 库
import torch

# 导入分布式相关模块
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.fsdp._shard_utils import (
    _create_chunk_dtensor,
    _create_chunk_sharded_tensor,
)
# 导入测试相关模块
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)


# 定义测试类，继承自 FSDPTest
class TestShardUtilsDistributed(FSDPTest):
    
    # 定义 world_size 属性，返回值为 2
    @property
    def world_size(self):
        return 2

    # 创建张量的方法，保持结果的确定性
    def _create_tensor(self, *size):
        torch.manual_seed(0)  # 使用种子保持随机数生成的确定性
        return torch.rand(*size).cuda()  # 生成指定大小的随机张量并将其放到 GPU 上

    # 装饰器，如果 GPU 数量小于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    def test_create_chunk_sharded_tensor(self):
        # 遍历不同的张量大小
        for size in ((1,), (1, 6), (12,), (12, 6), (25,), (25, 6)):
            tensor = self._create_tensor(*size)  # 创建指定大小的张量

            # 创建分块分布式张量
            sharded_tensor = _create_chunk_sharded_tensor(
                tensor,
                self.rank,  # 当前进程的排名
                self.world_size,  # 总进程数
                torch.cuda.device_count(),  # 当前系统的 GPU 数量
                _get_default_group(),  # 获取默认的分组
            )
            # 如果当前进程的排名为 0，则创建一个空的输出张量
            output = torch.empty(*size).cuda() if self.rank == 0 else None
            # 将分块分布式张量聚集到输出张量中
            sharded_tensor.gather(0, output)
            # 如果当前进程的排名为 0，则断言原始张量与输出张量相等
            if self.rank == 0:
                self.assertEqual(tensor, output)


# 测试分块分布式张量的相关类，继承自 DTensorTestBase
class TestShardUtilsDistributedDTensor(DTensorTestBase):
    
    # 定义 world_size 属性，返回值为 2
    @property
    def world_size(self):
        return 2

    # 创建张量的方法，保持结果的确定性
    def _create_tensor(self, *size):
        torch.manual_seed(0)  # 使用种子保持随机数生成的确定性
        return torch.rand(*size).cuda()  # 生成指定大小的随机张量并将其放到 GPU 上

    # 装饰器，执行通信相关操作
    @with_comms
    # 装饰器，如果 GPU 数量小于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    def test_create_chunk_dtensor(self):
        device_mesh = self.build_device_mesh()  # 构建设备网格

        # 遍历不同的张量大小
        for size in ((1,), (1, 6), (12,), (12, 6), (25,), (25, 6)):
            tensor = self._create_tensor(*size)  # 创建指定大小的张量
            tensor_chunks = torch.chunk(tensor, self.world_size, dim=0)  # 在指定维度上分块张量

            # 创建分块数据张量
            dtensor = _create_chunk_dtensor(tensor, self.rank, device_mesh)
            local_tensor = dtensor.to_local()  # 将分块数据张量转换为本地张量

            # 如果本地张量元素数不为 0，则断言本地张量与对应的分块张量相等
            if local_tensor.numel() != 0:
                self.assertEqual(local_tensor, tensor_chunks[self.rank])
            else:
                # 否则断言当前排名是否大于等于张量块数
                self.assertEqual(self.rank >= len(tensor_chunks), True)


# 如果当前脚本作为主程序运行，则执行测试
if __name__ == "__main__":
    run_tests()
```