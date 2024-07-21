# `.\pytorch\torch\fx\passes\pass_manager.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和函数装饰器
from functools import wraps
from inspect import unwrap
from typing import Callable, List, Optional
import logging

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 定义在模块中公开的函数和类列表
__all__ = [
    "PassManager",
    "inplace_wrapper",
    "log_hook",
    "loop_pass",
    "this_before_that_pass_constraint",
    "these_before_those_pass_constraint",
]

# 用于修改对象并返回非对象本身的值的可调用函数装饰器
def inplace_wrapper(fn: Callable) -> Callable:
    """
    Convenience wrapper for passes which modify an object inplace. This
    wrapper makes them return the modified object instead.

    Args:
        fn (Callable[Object, Any])

    Returns:
        wrapped_fn (Callable[Object, Object])
    """

    @wraps(fn)
    def wrapped_fn(gm):
        # 执行传入的函数，并将结果保存到变量中
        val = fn(gm)
        # 返回修改后的对象本身
        return gm

    return wrapped_fn

# 日志记录装饰器，用于记录可调用对象的输出
def log_hook(fn: Callable, level=logging.INFO) -> Callable:
    """
    Logs callable output.

    This is useful for logging output of passes. Note inplace_wrapper replaces
    the pass output with the modified object. If we want to log the original
    output, apply this wrapper before inplace_wrapper.


    ```
    def my_pass(d: Dict) -> bool:
        changed = False
        if 'foo' in d:
            d['foo'] = 'bar'
            changed = True
        return changed

    pm = PassManager(
        passes=[
            inplace_wrapper(log_hook(my_pass))
        ]
    )
    ```py

    Args:
        fn (Callable[Type1, Type2])
        level: logging level (e.g. logging.INFO)

    Returns:
        wrapped_fn (Callable[Type1, Type2])
    """
    @wraps(fn)
    def wrapped_fn(gm):
        # 执行传入的函数，并将结果保存到变量中
        val = fn(gm)
        # 记录日志，包括传入函数的名称和返回值
        logger.log(level, "Ran pass %s\t Return value: %s", fn, val)
        # 返回传入函数的返回值
        return val

    return wrapped_fn

# 循环应用传入的基础函数的便捷装饰器
def loop_pass(base_pass: Callable, n_iter: Optional[int] = None, predicate: Optional[Callable] = None):
    """
    Convenience wrapper for passes which need to be applied multiple times.

    Exactly one of `n_iter`or `predicate` must be specified.

    Args:
        base_pass (Callable[Object, Object]): pass to be applied in loop
        n_iter (int, optional): number of times to loop pass
        predicate (Callable[Object, bool], optional):

    """
    # 确保只能指定 `n_iter` 或 `predicate` 中的一个
    assert (n_iter is not None) ^ (
        predicate is not None
    ), "Exactly one of `n_iter`or `predicate` must be specified."

    @wraps(base_pass)
    def new_pass(source):
        # 初始化输出为源对象本身
        output = source
        # 根据传入的参数循环应用基础传递函数
        if n_iter is not None and n_iter > 0:
            for _ in range(n_iter):
                output = base_pass(output)
        elif predicate is not None:
            while predicate(output):
                output = base_pass(output)
        else:
            # 如果参数不正确，则抛出运行时错误
            raise RuntimeError(
                f"loop_pass must be given positive int n_iter (given "
                f"{n_iter}) xor predicate (given {predicate})"
            )
        # 返回最终的输出结果
        return output

    return new_pass

# Pass Schedule Constraints:
#
# 实现为“依赖于”运算符。如果列表根据此比较运算符具有有效的部分顺序，则约束得到满足。
def _validate_pass_schedule_constraint(
    constraint: Callable[[Callable, Callable], bool], passes: List[Callable]
):
    # 遍历传入的 passes 列表
    for i, a in enumerate(passes):
        # 在 passes 中找到当前 pass a 的索引 i，并遍历其后的所有 pass b
        for j, b in enumerate(passes[i + 1 :]):
            # 判断是否满足约束条件
            if constraint(a, b):
                continue
            # 若约束条件不满足，则抛出 RuntimeError 异常
            raise RuntimeError(
                f"pass schedule constraint violated. Expected {a} before {b}"
                f" but found {a} at index {i} and {b} at index{j} in pass"
                f" list."
            )


def this_before_that_pass_constraint(this: Callable, that: Callable):
    """
    定义一个部分顺序（“依赖于”函数），其中 `this` 必须在 `that` 之前发生。
    """

    def depends_on(a: Callable, b: Callable):
        # 如果 a 是 that 并且 b 是 this，则不满足条件，返回 False
        if a == that and b == this:
            return False
        return True

    return depends_on


def these_before_those_pass_constraint(these: Callable, those: Callable):
    """
    定义一个部分顺序（“依赖于”函数），其中 `these` 必须在 `those` 之前发生。
    输入在比较前会被“解包”。

    例如，以下 pass 列表和约束列表将是无效的。
    ```
    passes = [
        loop_pass(pass_b, 3),
        loop_pass(pass_a, 5),
    ]

    constraints = [
        these_before_those_pass_constraint(pass_a, pass_b)
    ]
    ```py

    Args:
        these (Callable): 应该先发生的 pass
        those (Callable): 应该稍后发生的 pass

    Returns:
        depends_on (Callable[[Object, Object], bool]): 依赖函数
    """

    def depends_on(a: Callable, b: Callable):
        # 如果 a 经过解包等于 those 并且 b 经过解包等于 these，则不满足条件，返回 False
        if unwrap(a) == those and unwrap(b) == these:
            return False
        return True

    return depends_on


class PassManager:
    """
    构建一个 PassManager。

    收集 passes 和 constraints。这定义了 pass 调度，管理 pass 约束和 pass 执行。

    Args:
        passes (Optional[List[Callable]]): pass 列表。一个 pass 是一个可调用对象，它修改一个对象并返回修改后的对象。
        constraint (Optional[List[Callable]]): 约束列表。一个约束是一个可调用对象，它接受两个 pass（A、B），如果 A 依赖于 B 则返回 True，否则返回 False。查看 `this_before_that_pass_constraint` 的实现示例。

    Attributes:
        passes (List[Callable]): pass 列表
        constraints (List[Callable]): 约束列表
        _validated (bool): 表示 pass 列表是否已验证过的标志，默认为 False
    """

    passes: List[Callable]
    constraints: List[Callable]
    _validated: bool = False

    def __init__(
        self,
        passes=None,
        constraints=None,
    ):
        # 初始化 PassManager 实例
        self.passes = passes or []
        self.constraints = constraints or []

    @classmethod
    def build_from_passlist(cls, passes):
        # 从 pass 列表构建 PassManager 实例
        pm = PassManager(passes)
        # TODO（alexbeloi）：添加约束管理/验证
        return pm

    def add_pass(self, _pass: Callable):
        # 向 pass 列表中添加一个 pass
        self.passes.append(_pass)
        # 设置 validated 标志为 False，表示 pass 列表已更改，需要重新验证
        self._validated = False
    # 将给定的约束条件添加到约束列表中，并标记为未验证状态
    def add_constraint(self, constraint):
        self.constraints.append(constraint)
        self._validated = False

    # 从当前的传递列表中移除指定的传递函数
    def remove_pass(self, _passes: List[str]):
        if _passes is None:
            return
        # 用于存储剩余传递函数的列表
        passes_left = []
        for ps in self.passes:
            # 如果传递函数的名称不在指定的列表中，则保留该传递函数
            if ps.__name__ not in _passes:
                passes_left.append(ps)
        self.passes = passes_left
        # 标记为未验证状态
        self._validated = False

    # 将传递列表中指定的传递函数替换为新的传递函数
    def replace_pass(self, _target, _replacement):
        passes_left = []
        for ps in self.passes:
            # 如果传递函数的名称与目标函数的名称匹配，则使用替换函数
            if ps.__name__ == _target.__name__:
                passes_left.append(_replacement)
            else:
                passes_left.append(ps)
        self.passes = passes_left
        # 标记为未验证状态
        self._validated = False

    # 验证当前的传递计划是否符合所有约束条件
    def validate(self):
        """
        Validates that current pass schedule defined by `self.passes` is valid
        according to all constraints in `self.constraints`
        """
        if self._validated:
            return
        # 对每个约束条件调用验证函数，验证传递计划是否合法
        for constraint in self.constraints:
            _validate_pass_schedule_constraint(constraint, self.passes)
        # 标记为已验证状态
        self._validated = True

    # 执行传递计划，将给定的源数据依次传递给每个传递函数进行处理
    def __call__(self, source):
        self.validate()
        out = source
        for _pass in self.passes:
            out = _pass(out)
        return out
```