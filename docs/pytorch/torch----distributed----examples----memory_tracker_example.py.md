# `.\pytorch\torch\distributed\examples\memory_tracker_example.py`

```
# mypy: allow-untyped-defs
# 引入torchvision库
import torchvision

# 引入torch库
import torch
# 从torch.distributed._tools中引入MemoryTracker类
from torch.distributed._tools import MemoryTracker


# 定义函数run_one_model，接受一个torch.nn.Module类型的net和一个torch.Tensor类型的input作为参数
def run_one_model(net: torch.nn.Module, input: torch.Tensor):
    # 将网络模型移动到GPU上运行
    net.cuda()
    # 将输入数据也移动到GPU上
    input = input.cuda()

    # 创建内存追踪器对象
    mem_tracker = MemoryTracker()
    # 在训练迭代开始前启动内存监控
    mem_tracker.start_monitor(net)

    # 执行一次训练迭代
    net.zero_grad(True)
    # 计算损失
    loss = net(input)
    # 如果损失是字典类型，则取其中的"out"键对应的值
    if isinstance(loss, dict):
        loss = loss["out"]
    # 计算损失的总和并反向传播
    loss.sum().backward()
    # 清除梯度
    net.zero_grad(set_to_none=True)

    # 在训练迭代结束后停止内存监控
    mem_tracker.stop()
    # 打印内存统计摘要
    mem_tracker.summary()
    # 绘制操作级别的内存轨迹图
    mem_tracker.show_traces()


# 运行一个基于ResNet34的模型，并传入一个形状为(32, 3, 224, 224)的随机数据张量到GPU上
run_one_model(torchvision.models.resnet34(), torch.rand(32, 3, 224, 224, device="cuda"))
```