# `.\pytorch\test\distributed\_tools\test_memory_tracker.py`

```py
# Owner(s): ["oncall: distributed"]
# 导入所需的模块和库
import os
import unittest
import torch
import torch.nn as nn

# 导入内部测试工具和类
from torch.distributed._tools import MemoryTracker
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TestCase

# 定义测试类 TestMemoryTracker，继承自 TestCase
class TestMemoryTracker(TestCase):
    
    # 标记为跳过测试，如果没有 CUDA 支持的话
    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_local_model(self):
        """
        Minimal test case to check the memory tracker can collect the expected
        memory stats at operator level, as well as can print the summary result
        without crash.
        """
        
        # 使用固定的随机种子来初始化随机数生成器
        torch.manual_seed(0)
        
        # 创建一个包含多个模块的模型，将其部署在 CUDA 设备上
        model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=False),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            ),
            nn.Flatten(start_dim=1),
            nn.Sequential(nn.Linear(64, 2), nn.ReLU(inplace=True)),
        ).cuda()

        # 启动内存追踪器，开始监控模型的内存使用情况
        tracker = MemoryTracker()
        tracker.start_monitor(model)

        # 创建输入张量 x，并将其放置在 CUDA 设备上
        x = torch.randn(size=(2, 3, 224, 224), device=torch.device("cuda"))
        
        # 创建目标张量 target，并将其放置在 CUDA 设备上
        target = torch.LongTensor([0, 1]).cuda()
        
        # 定义损失函数为交叉熵损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 执行模型的前向传播和反向传播
        criterion(model(x), target).backward()

        # 断言内存追踪器的钩子数大于 0
        self.assertTrue(len(tracker._hooks) > 0)

        # 停止内存追踪
        tracker.stop()

        # 断言内存追踪器的钩子数等于 0
        self.assertTrue(len(tracker._hooks) == 0)

        # 保存内存统计信息到文件
        path = "memory.trace"
        tracker.save_stats(path)
        
        # 从文件中加载内存统计信息
        tracker.load(path)
        
        # 打印内存使用情况的总结信息
        tracker.summary()
        
        # 如果存在保存的路径文件，则删除它
        if os.path.exists(path):
            os.remove(path)

        # 进行额外的断言以验证内存追踪器的状态
        self.assertTrue(tracker._op_index > 0)
        self.assertTrue(len(tracker._operator_names) > 0)
        self.assertEqual(len(tracker.memories_allocated), tracker._op_index)
        self.assertEqual(len(tracker.memories_active), tracker._op_index)
        self.assertEqual(len(tracker.memories_reserved), tracker._op_index)
        self.assertTrue(len(tracker._markers) == 2)
        self.assertTrue(tracker._cur_module_name != "")
        self.assertTrue(hasattr(tracker, "_num_cuda_retries"))


if __name__ == "__main__":
    # 运行测试
    run_tests()
```