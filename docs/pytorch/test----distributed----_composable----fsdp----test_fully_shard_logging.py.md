# `.\pytorch\test\distributed\_composable\fsdp\test_fully_shard_logging.py`

```py
# Owner(s): ["module: fsdp"]
# 导入 functools 模块，用于部分应用测试跳过装饰器的函数
import functools
# 导入操作系统相关功能的 os 模块
import os
# 导入 unittest.mock 模块，用于模拟 unittest 测试
import unittest.mock

# 导入 torch 分布式功能模块
import torch.distributed as dist
# 导入 torch._dynamo.test_case 模块下的 run_tests 函数
from torch._dynamo.test_case import run_tests
# 导入 torch.testing._internal.common_distributed 模块下的 skip_if_lt_x_gpu 装饰器
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
# 导入 torch.testing._internal.inductor_utils 模块下的 HAS_CUDA 变量
from torch.testing._internal.inductor_utils import HAS_CUDA
# 导入 torch.testing._internal.logging_utils 模块下的 LoggingTestCase 类
from torch.testing._internal.logging_utils import LoggingTestCase

# 定义 requires_cuda 装饰器，如果没有 CUDA 则跳过测试
requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")
# 定义 requires_distributed 装饰器，如果没有分布式支持则跳过测试
requires_distributed = functools.partial(
    unittest.skipIf, not dist.is_available(), "requires distributed"
)

# 使用 skip_if_lt_x_gpu 装饰器，要求至少有 2 个 GPU 才执行测试
@skip_if_lt_x_gpu(2)
# LoggingTests 类，继承自 LoggingTestCase 类
class LoggingTests(LoggingTestCase):
    # 使用 requires_distributed 装饰器，要求有分布式支持才执行该测试方法
    @requires_distributed()
    # test_fsdp_logging 方法，测试 FSDP 的日志功能
    def test_fsdp_logging(self):
        # 复制当前环境变量字典
        env = dict(os.environ)
        # 设置 TORCH_LOGS 环境变量为 "fsdp"
        env["TORCH_LOGS"] = "fsdp"
        # 设置 RANK 环境变量为 "0"
        env["RANK"] = "0"
        # 设置 WORLD_SIZE 环境变量为 "1"
        env["WORLD_SIZE"] = "1"
        # 设置 MASTER_PORT 环境变量为 "34715"
        env["MASTER_PORT"] = "34715"
        # 设置 MASTER_ADDR 环境变量为 "localhost"
        env["MASTER_ADDR"] = "localhost"
        # 运行给定代码块，并捕获标准输出和标准错误
        stdout, stderr = self.run_process_no_exception(
            """\
import logging
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
# 获取名为 "torch.distributed._composable.fsdp" 的 logger 对象
logger = logging.getLogger("torch.distributed._composable.fsdp")
# 设置 logger 的日志级别为 DEBUG
logger.setLevel(logging.DEBUG)
# 指定设备为 "cuda"
device = "cuda"
# 设置随机种子为 0
torch.manual_seed(0)
# 创建包含两个线性层的神经网络模型
model = nn.Sequential(*[nn.Linear(4, 4, device=device, bias=False) for _ in range(2)])
# 对模型的每一层进行分片
for layer in model:
    fully_shard(layer)
# 对整个模型进行分片
fully_shard(model)
# 在指定设备上生成一个随机输入张量
x = torch.randn((4, 4), device=device)
# 对模型进行前向传播，计算输出并求和，然后反向传播
model(x).sum().backward()
""",
            env=env,
        )
        # 断言特定字符串在标准错误输出中，用于验证日志记录功能的正确性
        self.assertIn("FSDP::root_pre_forward", stderr.decode("utf-8"))
        self.assertIn("FSDP::pre_forward (0)", stderr.decode("utf-8"))
        self.assertIn("FSDP::pre_forward (1)", stderr.decode("utf-8"))
        self.assertIn("FSDP::post_forward (0)", stderr.decode("utf-8"))
        self.assertIn("FSDP::post_forward (1)", stderr.decode("utf-8"))
        self.assertIn("FSDP::pre_backward (0)", stderr.decode("utf-8"))
        self.assertIn("FSDP::pre_backward (1)", stderr.decode("utf-8"))
        self.assertIn("FSDP::post_backward (0)", stderr.decode("utf-8"))
        self.assertIn("FSDP::post_backward (1)", stderr.decode("utf-8"))
        self.assertIn("FSDP::root_post_backward", stderr.decode("utf-8"))

# 如果该脚本作为主程序执行，则运行测试
if __name__ == "__main__":
    run_tests()
```