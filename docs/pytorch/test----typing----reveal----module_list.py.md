# `.\pytorch\test\typing\reveal\module_list.py`

```py
import torch


# 定义一个继承自 torch.nn.Module 的类 FooModule
class FooModule(torch.nn.Module):
    pass


# 定义一个继承自 torch.nn.Module 的类 BarModule
class BarModule(torch.nn.Module):
    pass


# 创建一个 ModuleList 对象 ml，其中包含两个 Module 类的实例对象 FooModule 和 BarModule
ml: torch.nn.ModuleList = torch.nn.ModuleList([FooModule(), BarModule()])
# 检查 ml 中第一个元素（索引为 0）的子模块列表是否为空，禁止特定警告 B015
ml[0].children() == []  # noqa: B015
# 揭示 ml 的类型为 {ModuleList}
reveal_type(ml)  # E: {ModuleList}
```