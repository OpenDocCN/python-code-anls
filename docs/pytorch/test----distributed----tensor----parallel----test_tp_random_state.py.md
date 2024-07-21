# `.\pytorch\test\distributed\tensor\parallel\test_tp_random_state.py`

```py
# Owner(s): ["oncall: distributed"]
# 导入PyTorch库
import torch
# 导入分布式函数集合模块
import torch.distributed._functional_collectives as funcol
# 导入随机张量生成模块
import torch.distributed._tensor.random as random

# 从torch.distributed._tensor中导入初始化设备网格和复制类
from torch.distributed._tensor import init_device_mesh, Replicate
# 从torch.distributed.tensor.parallel.api中导入并行化模块
from torch.distributed.tensor.parallel.api import parallelize_module
# 从torch.distributed.tensor.parallel.style中导入ColwiseParallel风格
from torch.distributed.tensor.parallel.style import ColwiseParallel
# 从torch.testing._internal.common_distributed中导入跳过条件测试装饰器
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
# 从torch.testing._internal.common_utils中导入运行测试函数
from torch.testing._internal.common_utils import run_tests
# 从torch.testing._internal.distributed._tensor.common_dtensor中导入测试基类和MLP模块
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule,
    with_comms,
)

# 定义测试类TensorParallelRandomStateTests，继承自DTensorTestBase
class TensorParallelRandomStateTests(DTensorTestBase):
    # 定义方法get_tensor_slice，用于获取大张量的切片
    def get_tensor_slice(self, idx, n, large_tensor):
        # 获取大张量的形状
        shape = large_tensor.shape
        # 断言大张量的第一维度能够被n整除
        assert shape[0] % n == 0
        # 计算本地切片的形状
        local_shape = [shape[0] // n, shape[1]]

        # 计算切片的索引
        slice_idx = [
            slice(idx * local_shape[0], (idx + 1) * local_shape[0]),
            slice(local_shape[1]),
        ]
        # 返回大张量的切片
        return large_tensor[slice_idx]

    # 定义方法check_gathered_tensors，用于检查收集到的张量
    def check_gathered_tensors(self, self_rank, size, gathered_tensors, assertFunc):
        # 遍历所有进程
        for other_rank in range(size):
            # 如果当前进程号不等于其他进程号
            if self_rank != other_rank:
                # 断言当前进程的切片与其他进程的切片是否满足指定的断言函数
                assertFunc(
                    self.get_tensor_slice(self_rank, size, gathered_tensors),
                    self.get_tensor_slice(other_rank, size, gathered_tensors),
                )

    # 使用@with_comms装饰器确保测试在通信环境下运行
    @with_comms
    # 使用@skip_if_lt_x_gpu(4)装饰器跳过GPU数小于4的情况
    @skip_if_lt_x_gpu(4)
    # 如果当前脚本作为主程序运行，则执行测试
    if __name__ == "__main__":
        run_tests()
```