# `.\pytorch\torch\distributed\pipelining\_unflatten.py`

```
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

# 导入必要的类型 Dict
from typing import Dict

# 导入 torch 库
import torch
# 导入 torch.fx 模块中的 _ModuleFrame 类
from torch.export.unflatten import _ModuleFrame


# 定义一个函数 _outline_submodules，接受一个 torch.fx.Graph 类型的参数 orig_graph
def _outline_submodules(orig_graph: torch.fx.Graph):
    # 创建一个空的 GraphModule 对象，用于保存被概述的模块
    new_module = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
    # 创建两个空字典，用于记录已经处理过的节点和模块
    seen_nodes: Dict[str, torch.fx.Node] = {}
    seen_modules: Dict[int, torch.nn.Module] = {}
    # 实例化 _ModuleFrame 对象，并调用其 run_outer 方法来进行模块的概述
    _ModuleFrame(
        orig_graph,               # 原始的图结构
        tuple(orig_graph.nodes),  # 图中所有节点的元组
        seen_nodes,               # 记录已见过的节点的字典
        seen_modules,             # 记录已见过的模块的字典
        None,                     # 父模块的名称
        [""],                     # 当前模块的路径
        "",                       # 当前模块的名称
        {},                       # 额外的关键字参数
        module=new_module,        # 要存储概述结果的 GraphModule 对象
    ).run_outer()
    # 对新生成的模块进行静态检查
    new_module.graph.lint()
    # 重新编译新生成的模块
    new_module.recompile()
    # 返回被概述后的新模块
    return new_module
```