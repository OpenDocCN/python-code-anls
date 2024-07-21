# `.\pytorch\torch\fx\passes\infra\pass_manager.py`

```
# 声明类型检查允许未标记的函数定义
# 导入检查源码的模块
# 导入日志记录模块
# 导入队列数据结构模块
# 导入装饰器函数模块
# 导入类型提示相关模块
import inspect
import logging
from queue import Queue
from functools import wraps
from typing import Callable, Dict, List

# 导入神经网络模块中的神经网络类
import torch.nn as nn
# 导入 Torch 的图模块中的图模块类
from torch.fx.graph_module import GraphModule
# 导入 Torch 的兼容性模块
from torch.fx._compatibility import compatibility
# 导入 Torch 的基础通行证结果模块
from torch.fx.passes.infra.pass_base import PassResult

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)
# 设置日志记录器的日志级别为警告
logger.setLevel(logging.WARNING)

# 声明模块中公开的函数和类名
__all__ = ['pass_result_wrapper', 'this_before_that_pass_constraint', 'PassManager']

# 根据兼容性注解定义装饰器函数
@compatibility(is_backward_compatible=False)
def pass_result_wrapper(fn: Callable) -> Callable:
    """
    Wrapper for passes which currently do not return a PassResult.
    This wrapper makes them return a PassResult containing the modified object
    and True for the "modified" flag.

    Args:
        fn (Callable[Module, Any])

    Returns:
        wrapped_fn (Callable[Module, PassResult])
    """
    if fn is None:
        return None

    # 包装函数以返回适用的 PassResult 对象
    @wraps(fn)
    def wrapped_fn(gm):
        res = fn(gm)
        if res is None:
            return PassResult(gm, True)
        if isinstance(res, PassResult):
            return res
        elif isinstance(res, nn.Module):
            return PassResult(res, True)

    # 如果函数不是普通函数，则将函数名设为相应类型的名称
    if not inspect.isfunction(fn):
        wrapped_fn.__name__ = type(fn).__name__

    return wrapped_fn

# 验证通过调度约束的函数
def _validate_pass_schedule_constraint(
    constraint: Callable[[Callable, Callable], bool], passes: List[Callable]
) -> None:
    for i, a in enumerate(passes):
        for j, b in enumerate(passes[i + 1 :]):
            if constraint(a, b):
                continue
            # 抛出运行时错误，显示违反的调度约束
            raise RuntimeError(
                f"pass schedule constraint violated. Expected {a} before {b}"
                f" but found {a} at index {i} and {b} at index{j} in pass"
                f" list."
            )

# 对传递的函数列表进行拓扑排序的函数
def _topological_sort_passes(
    passes: List[Callable], constraints: List[Callable]
) -> List[Callable]:
    """
    Args
        passes: Passes that we are ordering
        constraints: Constraints applied on these passes

    Returns
        A sorted list of callables and a boolean of if a circular dependency
        existed
    """
    if len(constraints) == 0:
        return passes

    # 构建一个图，将节点映射到使用它们的节点列表
    graph: Dict[Callable, List[Callable]] = {p : [] for p in passes}
    # 构建一个入度映射表
    indegree_map: Dict[Callable, int] = dict.fromkeys(passes, 0)
    # 候选节点队列
    candidates: Queue = Queue()
    
    # 对每对节点应用约束，构建图和入度映射
    for a in passes:
        for b in passes:
            if a == b:
                continue

            for constraint in constraints:
                if not constraint(a, b):
                    graph[b].append(a)
                    indegree_map[a] += 1

        if indegree_map[a] == 0:
            candidates.put(a)

    # 记录已访问节点
    visited: Dict[Callable, bool] = dict.fromkeys(passes, False)
    # 排序后的节点列表
    sorted_passes: List[Callable] = []
    # 当候选节点队列不为空时，持续执行以下操作
    while not candidates.empty():
        # 从候选节点队列中获取一个节点
        p = candidates.get()
        # 将该节点加入已排序的通过列表中
        sorted_passes.append(p)
        # 标记该节点为已访问过
        visited[p] = True

        # 遍历当前节点的后继节点
        for n in graph[p]:
            # 如果后继节点尚未访问过
            if not visited[n]:
                # 减少后继节点的入度
                indegree_map[n] -= 1
                # 如果后继节点的入度减少至零，则将其加入候选节点队列
                if indegree_map[n] == 0:
                    candidates.put(n)

    # 检查是否存在未访问的节点（即图中的循环依赖）
    cycle_passes = list(filter(lambda p: indegree_map[p] != 0, indegree_map.keys()))
    # 如果存在未访问的节点，则抛出运行时错误
    if len(cycle_passes) != 0:
        error = f"Circular dependency detected within the following passes: {cycle_passes}"
        raise RuntimeError(error)

    # 返回拓扑排序后的通过列表
    return sorted_passes
# 定义一个装饰器函数，用于设置兼容性，指定是否向后兼容
@compatibility(is_backward_compatible=False)
# 定义一个函数，用于指定某个 pass 必须在另一个 pass 之前执行的约束条件
def this_before_that_pass_constraint(this: Callable, that: Callable) -> Callable:
    """
    Defines a partial order ('depends on' function) where `this` must occur
    before `that`.

    For example, the following pass list and constraint list would be invalid.
    ```
    passes = [pass_b, pass_a]

    constraints = [
        this_before_that_pass_constraint(pass_a, pass_b)
    ]
    ```

    Args:
        this (Callable): pass which should occur first
        that (Callable): pass which should occur later

    Returns:
        depends_on (Callable[[Object, Object], bool]
    """

    # 内部定义一个函数 depends_on，用于检查两个 pass 的执行顺序是否符合约束
    def depends_on(a: Callable, b: Callable):
        if a == that and b == this:
            return False
        return True

    # 返回内部定义的 depends_on 函数对象
    return depends_on


# 定义一个兼容性设置类，用于管理 pass 的执行顺序和约束条件
@compatibility(is_backward_compatible=False)
class PassManager:
    """
    Construct a PassManager.

    Collects passes and constraints. This defines the pass schedule, manages
    pass constraints and pass execution.

    Args:
        passes (Optional[List[Callable]]): List of passes. A pass is a
            callable which modifies an object and returns a PassResult
        constraint (Optional[List[Callable]]): List of constraints. A
            constraint is a callable which takes two passes (A, B) and returns
            True if A depends on B and False otherwise. See implementation of
            `this_before_that_pass_constraint` for example.
        steps (int): Max number of times we run the passes (default = 1).
        run_checks_after_each_pass (bool): Whether to run checks and linting
            after each pass
        suppress_check_failures (bool): Whether to raise errors when running
            checks
    """

    # 类变量：存储 pass 函数列表和约束条件列表，以及验证状态和执行步数设置
    passes: List[Callable[[nn.Module], PassResult]]
    constraints: List[Callable[[Callable, Callable], bool]]
    _validated: bool = False
    steps: int = 1

    # 初始化方法，接受传入的 passes、constraints、steps 以及其他设置
    def __init__(
        self,
        passes=None,
        constraints=None,
        steps=None,
        run_checks_after_each_pass: bool = False,
        suppress_check_failures: bool = False,
    ):
        # 初始化 passes 和 constraints 列表，若未提供则为空列表
        self.passes = passes or []
        self.constraints = constraints or []

        # 如果提供了 steps 参数，则设置类的步数属性为该值
        if steps:
            self.steps = steps

        # 设置是否在每次 pass 执行后运行检查和 linting 的标志
        self.run_checks_after_each_pass = run_checks_after_each_pass
        # 设置是否在运行检查时抑制失败的标志
        self.suppress_check_failures = suppress_check_failures

    # 方法：向 passes 列表中添加一个 pass 函数
    def add_pass(self, _pass: Callable):
        """
        Adds a pass into the current list of passes.
        """
        self.passes.append(_pass)
        # 标记验证状态为未验证
        self._validated = False

    # 方法：向 constraints 列表中添加一个约束条件函数
    def add_constraint(self, constraint: Callable):
        """
        Adds a constraint into the current list of constraints.
        """
        self.constraints.append(constraint)
        # 标记验证状态为未验证
        self._validated = False
    def validate_constraints(self):
        """
        Validates that current pass schedule defined by `self.passes` is valid
        according to all constraints in `self.constraints`
        """
        # 如果已经验证过，直接返回
        if self._validated:
            return
        # 遍历每个约束条件，验证当前的通行计划是否满足约束条件
        for constraint in self.constraints:
            _validate_pass_schedule_constraint(constraint, self.passes)
        # 标记为已验证
        self._validated = True

    def solve_constraints(self):
        """
        Finds a valid traversal order based on the given constraints and orders
        the passes based on this order.
        
        If a circular dependency exists between the constraints and steps = 1,
        then we will raise an error because if steps != 1 this means that we
        will re-run the passes, allowing for circular dependencies.
        """
        # 根据给定的约束条件对 passes 进行拓扑排序，以找到有效的遍历顺序
        self.passes = _topological_sort_passes(self.passes, self.constraints)
        # 标记为已验证
        self._validated = True

    def add_checks(self, check: Callable) -> None:
        """
        Adds a function which takes runs various checks on a given graph module.
        This function is run before and after each pass if the
        `run_checks_after_each_pass` flag is enabled.
        """
        # 检查添加的检查函数的参数数量是否为1
        sig = inspect.signature(check)
        if len(list(sig.parameters.values())) != 1:
            raise TypeError("PassManager check function should only take in one variable, a module")
        # 将传入的检查函数作为属性添加到当前对象中
        setattr(self, "check", check)  # noqa: B010

    def check(self, module: nn.Module) -> None:
        # Placeholder function for a check that operates on a given nn.Module
        pass
    # 定义一个方法，用于对给定的 nn.Module 执行一系列 passes，并返回 PassResult
    def __call__(self, module: nn.Module) -> PassResult:
        """
        Runs a list of passes in the order based on `self.passes` on the given
        graph module. Each time a pass is run, checks and linting will be run on
        the graph module if `run_checks_after_each_pass` is set.

        If the module is a graph module, we will run the list of passes until
        the graph stops changing, or until `steps` number of times.
        """
        
        # 如果尚未进行验证，则解决约束条件
        if not self._validated:
            self.solve_constraints()

        # 检查图的不变性
        self.check(module)

        # 运行 passes 的集合，重复执行 `steps` 次，或者直到图不再改变为止
        overall_modified = False
        for _ in range(self.steps):
            modified = False

            # 在图模块上运行 passes 的集合
            for i, fn in enumerate(self.passes):
                fn_name = fn.__name__ if inspect.isfunction(fn) else type(fn).__name__
                logger.debug("Running pass '%s'", fn_name)

                try:
                    # 调用 passes 中的函数，并获取返回结果
                    res = fn(module)

                    # 检查返回结果的类型是否为 PassResult，或者是否具有 graph_module 属性
                    if not isinstance(res, PassResult) and not hasattr(
                        res, "graph_module"
                    ):
                        raise TypeError(
                            f"The result of the pass {fn_name} should be type PassResult."
                            + "Please wrap it with pass_result_wrapper()"
                        )
                    
                    # 更新 module 为 pass 后的返回结果的 graph_module
                    module = res.graph_module
                    modified = modified or res.modified

                    # 如果 module 是 GraphModule 类型，则重新编译模块
                    if isinstance(module, GraphModule):
                        logger.debug("Graph after pass '%s': %s", fn_name, module.graph)
                        module.recompile()

                    # 如果设置了每个 pass 后运行检查，则再次检查图的不变性
                    if self.run_checks_after_each_pass:
                        self.check(module)

                except Exception as e:
                    # 在运行 pass 中发生异常时，记录异常信息
                    prev_pass_names = [
                        p.__name__ if inspect.isfunction(p) else type(p).__name__
                        for p in self.passes[:i]
                    ]
                    msg = f"An error occurred when running the '{fn_name}' pass after the following passes: {prev_pass_names}"
                    raise Exception(msg) from e  # noqa: TRY002

            # 如果图不再改变，则停止运行 passes
            overall_modified = overall_modified or modified
            if not modified:
                break

        # 返回 PassResult，包含最终的 module 和整体是否修改的标志
        return PassResult(module, overall_modified)
```