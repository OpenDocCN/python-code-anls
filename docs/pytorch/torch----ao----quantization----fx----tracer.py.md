# `.\pytorch\torch\ao\quantization\fx\tracer.py`

```
import torch  # 导入PyTorch库
from torch.fx._symbolic_trace import Tracer  # 导入Tracer类
from torch.fx.proxy import Scope  # 导入Scope类
from torch.ao.nn.intrinsic import _FusedModule  # 导入_FusedModule类
from typing import List, Callable  # 导入类型提示List和Callable

__all__ = [
    "QuantizationTracer",
]

class ScopeContextManager(torch.fx.proxy.ScopeContextManager):
    def __init__(
        self,
        scope: Scope,
        current_module: torch.nn.Module,
        current_module_path: str
    ):
        super().__init__(scope, Scope(current_module_path, type(current_module)))

class QuantizationTracer(Tracer):
    def __init__(
        self, skipped_module_names: List[str], skipped_module_classes: List[Callable]
    ):
        super().__init__()  # 调用父类的初始化方法
        self.skipped_module_names = skipped_module_names  # 设置跳过的模块名列表
        self.skipped_module_classes = skipped_module_classes  # 设置跳过的模块类列表
        # NB: initialized the module_type of top level module to None
        # we are assuming people won't configure the model with the type of top level
        # module here, since people can use "" for global config
        # We can change this if there is a use case that configures
        # qconfig using top level module type
        self.scope = Scope("", None)  # 初始化一个空的Scope对象，顶层模块类型设置为None
        self.record_stack_traces = True  # 启用堆栈跟踪记录

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        return (
            (
                (m.__module__.startswith("torch.nn") or m.__module__.startswith("torch.ao.nn"))
                and not isinstance(m, torch.nn.Sequential)
            )  # 检查模块是否在torch.nn或torch.ao.nn命名空间中，并且不是torch.nn.Sequential的实例
            or module_qualified_name in self.skipped_module_names  # 检查模块名是否在跳过的模块名列表中
            or type(m) in self.skipped_module_classes  # 检查模块类型是否在跳过的模块类列表中
            or isinstance(m, _FusedModule)  # 检查模块是否是_FusedModule的实例
        )
```