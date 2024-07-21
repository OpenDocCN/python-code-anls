# `.\pytorch\torch\fx\passes\graph_transform_observer.py`

```py
# mypy: allow-untyped-defs
# 导入所需的模块和类
import os  # 导入操作系统模块
from typing import Optional  # 导入类型提示中的可选类型

from torch.fx._compatibility import compatibility  # 导入 Torch 中的兼容性装饰器
from torch.fx.graph_module import GraphModule  # 导入 Torch 中的图模块

from .graph_drawer import FxGraphDrawer  # 导入自定义的图形绘制模块

__all__ = ["GraphTransformObserver"]  # 模块中公开的接口列表

@compatibility(is_backward_compatible=False)
class GraphTransformObserver:
    __pass_count = 0  # 类属性，记录转换过程的计数

    def __init__(self, gm: GraphModule, passname: str, log_url: Optional[str] = None):
        # 初始化方法，接收图模块实例、转换过程名称和可选的日志 URL
        self.log_url = log_url  # 初始化日志 URL
        if self.log_url is None:  # 如果日志 URL 为空，则不进行日志记录
            return
        
        GraphTransformObserver.__pass_count += 1  # 记录转换过程计数
        self.gm = gm  # 初始化图模块
        self.passname = passname  # 初始化转换过程名称

        # 创建输入图形的绘制对象
        self.input_dot_graph = FxGraphDrawer(
            self.gm,
            self.passname,
            ignore_getattr=True,
            ignore_parameters_and_buffers=True,
        ).get_dot_graph()

    @classmethod
    def get_current_pass_count(cls):
        # 类方法，返回当前的转换过程计数
        return cls.__pass_count

    def __enter__(self):
        # 进入上下文管理器时调用的方法
        if self.log_url is None or self.gm is None:
            return self

        # 初始化被擦除和被创建节点的集合
        self.erased_nodes = set()
        self.created_nodes = set()

        # 注册创建节点和擦除节点的钩子函数
        self.gm._register_create_node_hook(self.on_node_creation)
        self.gm._register_erase_node_hook(self.on_node_erase)

        return self

    def __exit__(self, type, value, tb):
        # 退出上下文管理器时调用的方法
        if self.log_url is None or self.gm is None:
            return

        # 取消注册创建节点和擦除节点的钩子函数
        self.gm._unregister_create_node_hook(self.on_node_creation)
        self.gm._unregister_erase_node_hook(self.on_node_erase)

        # 如果有创建或擦除的节点，更新输入和输出图形的节点颜色，并写入到文件
        if len(self.created_nodes) > 0 or len(self.erased_nodes) > 0:
            # 更新输入图形的节点颜色并写入到文件
            for e in self.input_dot_graph.get_node_list():
                if e.get_name() in self.erased_nodes:
                    e.obj_dict["attributes"]["fillcolor"] = "yellow"
                else:
                    e.obj_dict["attributes"]["fillcolor"] = "grey"
            self.input_dot_graph.write(
                os.path.join(
                    self.log_url,
                    f"pass_{GraphTransformObserver.__pass_count}_{self.passname}_input_graph.dot",
                )
            )

            # 更新输出图形的节点颜色并写入到文件
            output_dot_graph = FxGraphDrawer(
                self.gm,
                self.passname,
                ignore_getattr=True,
                ignore_parameters_and_buffers=True,
            ).get_dot_graph()
            for e in output_dot_graph.get_node_list():
                if e.get_name() in self.created_nodes:
                    e.obj_dict["attributes"]["fillcolor"] = "yellow"
                else:
                    e.obj_dict["attributes"]["fillcolor"] = "grey"
            output_dot_graph.write(
                os.path.join(
                    self.log_url,
                    f"pass_{GraphTransformObserver.__pass_count}_{self.passname}_output_graph.dot",
                )
            )

    def on_node_creation(self, node):
        # 当节点被创建时调用的回调函数
        self.created_nodes.add(node.name)

    def on_node_erase(self, node):
        # 当节点被擦除时调用的回调函数
        self.erased_nodes.add(node.name)
```