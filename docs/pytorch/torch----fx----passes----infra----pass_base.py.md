# `.\pytorch\torch\fx\passes\infra\pass_base.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和类
import abc  # 导入抽象基类模块
from collections import namedtuple  # 导入命名元组模块
from typing import Optional  # 导入类型提示中的可选类型

from torch.fx.graph_module import GraphModule  # 导入图模块类
from torch.fx._compatibility import compatibility  # 导入兼容性函数


__all__ = ['PassResult', 'PassBase']  # 导出模块中的 PassResult 和 PassBase 类

@compatibility(is_backward_compatible=False)
# 定义 PassResult 类，表示一个 pass 的结果，包含修改后的图模块和修改标志
class PassResult(namedtuple("PassResult", ["graph_module", "modified"])):
    """
    Result of a pass:
        graph_module: The modified graph module
        modified: A flag for if the pass has modified the graph module
    """
    def __new__(cls, graph_module, modified):
        return super().__new__(cls, graph_module, modified)

@compatibility(is_backward_compatible=False)
# 定义 PassBase 类，作为实现 passes 的基础接口
class PassBase(abc.ABC):
    """
    Base interface for implementing passes.

    It is required to implement the `call` function so that we can directly
    pass instances of the Pass directly to the PassManager and call them as a
    function.

    We can directly pass an instance of a class implementing this interface into
    the PassManager's `passes` attribute.
    """

    def __call__(self, graph_module: GraphModule) -> Optional[PassResult]:
        """
        Runs the precondition check, the pass itself, and the postcondition check.
        """
        # 调用 requires 函数，进行前置条件检查
        self.requires(graph_module)
        # 调用 call 函数，运行 pass 本身，获取结果
        res = self.call(graph_module)
        # 调用 ensures 函数，进行后置条件检查
        self.ensures(graph_module)
        # 返回 pass 运行的结果
        return res

    @abc.abstractmethod
    def call(self, graph_module: GraphModule) -> Optional[PassResult]:
        """
        The pass that is run through the given graph module. To implement a
        pass, it is required to implement this function.

        Args:
            graph_module: The graph module we will run a pass on
        """
        pass

    def requires(self, graph_module: GraphModule) -> None:  # noqa: B027
        """
        This function will be called before the pass is run and will check that
        the given graph module contains the preconditions needed to run the
        pass. It is not required to implement this function.

        Args:
            graph_module: The graph module we will run checks on
        """
        pass

    def ensures(self, graph_module: GraphModule) -> None:  # noqa: B027
        """
        This function will be called after the pass is run and will check that
        the given graph module contains the postconditions needed to run the
        pass. It is not required to implement this function.

        Args:
            graph_module: The graph module we will run checks on
        """
        pass
```