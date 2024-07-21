# `.\pytorch\test\distributed\_composable\fsdp\test_fully_shard_memory.py`

```py
# Owner(s): ["oncall: distributed"]  # 指定代码所有者为分布式团队

import functools  # 导入 functools 模块

import torch  # 导入 PyTorch 库

from torch.distributed._composable.fsdp import (  # 导入 FSDP 相关模块
    CPUOffloadPolicy,
    fully_shard,
    OffloadPolicy,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入跳过测试函数
from torch.testing._internal.common_fsdp import FSDPTest  # 导入 FSDP 测试基类
from torch.testing._internal.common_utils import run_tests  # 导入运行测试的函数
from torch.testing._internal.distributed._tensor.common_dtensor import (  # 导入测试用的数据结构和函数
    ModelArgs,
    Transformer,
    TransformerBlock,
)


class TestFullyShardMemory(FSDPTest):  # 定义测试类，继承自 FSDPTest

    @property
    def world_size(self) -> int:  # 定义属性方法，返回 CUDA 设备数量与 2 的最小值
        return min(2, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)  # 装饰器，如果 GPU 数量小于 2 则跳过测试
    def test_fully_shard_training_memory(self):  # 定义测试方法，测试全分片训练内存占用
        self.run_subtests(  # 运行子测试
            {
                "reshard_after_forward": [True, False],  # 参数：前向传播后重新分片
                "use_cpu_offload": [True, False],  # 参数：使用 CPU 卸载
                "run_optim_in_backward": [True, False],  # 参数：在反向传播时运行优化器
            },
            self._test_fully_shard_training_memory,  # 调用具体测试方法
        )

    def _test_fully_shard_training_memory(
        self,
        reshard_after_forward: bool,
        use_cpu_offload: bool,
        run_optim_in_backward: bool,
    ):  # 具体测试方法定义，接受多个布尔类型参数
        # 这里未提供完整方法定义，需要补充完整的测试逻辑

    def _get_peak_active_memory_mb(self) -> int:  # 获取 CUDA 设备的峰值活跃内存（MB）
        mem_stats = torch.cuda.memory_stats()
        return round(mem_stats["active_bytes.all.peak"] / 1e6)

    def _get_curr_active_memory_mb(self) -> int:  # 获取 CUDA 设备当前活跃内存（MB）
        mem_stats = torch.cuda.memory_stats()
        return round(mem_stats["active_bytes.all.current"] / 1e6)

    def _register_optim_in_backward(  # 在反向传播时注册优化器
        self, model: torch.nn.Module, **optim_kwargs
    ) -> None:
        param_to_optim = {}  # 创建空字典，用于存储参数与优化器的映射关系
        for param in model.parameters():  # 遍历模型的参数
            param_to_optim[param] = torch.optim.AdamW([param], **optim_kwargs)  # 使用 AdamW 优化器进行优化

        def optim_hook(param: torch.nn.Parameter) -> None:  # 定义优化钩子函数
            param_to_optim[param].step()  # 执行优化步骤
            param_to_optim[param].zero_grad()  # 清空梯度

        for param in model.parameters():  # 再次遍历模型的参数
            param.register_post_accumulate_grad_hook(optim_hook)  # 注册优化钩子函数

if __name__ == "__main__":  # 如果当前脚本作为主程序运行
    run_tests()  # 运行测试
```