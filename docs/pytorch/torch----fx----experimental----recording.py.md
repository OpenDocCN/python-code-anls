# `.\pytorch\torch\fx\experimental\recording.py`

```py
# mypy: allow-untyped-defs
# 引入 functools 模块，提供高阶函数和操作工具
import functools
# 引入 inspect 模块，提供用于检查源码的函数
import inspect
# 引入 itertools 模块，提供创建和操作迭代器的函数
import itertools
# 引入 logging 模块，用于记录日志
import logging
# 从 dataclasses 模块中引入 dataclass 装饰器，用于创建数据类
from dataclasses import dataclass
# 从 typing 模块中引入 Any, Callable, Dict, List, Optional, Tuple, Union 类型
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 引入 torch 库
import torch
# 从 torch.utils._pytree 模块中引入 pytree
import torch.utils._pytree as pytree

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)

# __all__ 列表，定义模块对外导出的符号名
__all__ = [
    "ShapeEnvEvent",                # 形状环境事件
    "record_shapeenv_event",        # 记录形状环境事件
    "replay_shape_env_events",      # 重播形状环境事件
    "FakeTensorMeta",               # 假张量元信息
    "shape_env_check_state_equal",  # 形状环境状态检查相等性
    "NotEqualError",                # 不相等错误
]

# [Note: Recording ShapeEnv Events]
# =================================
#
# What is a ShapeEnv event?
# -------------------------
# We consider a ShapeEnv event every function call (ShapeEnv method or
# independent function) that modifies the state of the ShapeEnv instance.
# Such calls are recorded alongside their positional and keyword arguments,
# so that it may be replayed over a different ShapeEnv instance.
#
# See [Note: ShapeEnv State Equality] for what is considered the state
# of a ShapeEnv instance.
#
# What is it for?
# ---------------
# ShapeEnv events recording is used for reconstructing the ShapeEnv in an
# arbitrary state in time.
#
# Being able to arbitrarily replay events like so is useful, mainly for
# translation validation bisection. i.e. if a ValidationException has been
# raised, find the earliest point in time where the translation validation
# fails.
#
# Besides that, it also allows us to inspect the given instance and,
# for example, check the guards that would actually be issued at that point.
#
# What kind of arguments can be stored in an event?
# -------------------------------------------------
# There's no specific rule for what cannot be used as an argument.
# That said, pay special attention to the following cases:
#
#   1. Tensor inputs: there are some tests that check whether the inputs
#      were garbage collected after execution. These will fail if there's
#      an event that is holding a reference to those inputs.
#
#   2. ShapeEnv arguments: if there is an argument of ShapeEnv type, that
#      will be automatically replaced by the new given ShapeEnv instance.
#
#   3. SymTypes arguments: they also hold references to ShapeEnv. So,
#      whenever we see them, we create a new instance, replacing the
#      ShapeEnv reference.
#
#   4. FX nodes: specifically, FX nodes from the FX graph for symbolic
#      shapes. That argument must be replaced when replaying the event at
#      ShapeEnvEvent.run, since it has to reference a node from the given
#      instance, and not from the recorded instance.


# Event class for reconstructing ShapeEnv at arbitrary time.
#
# Represents a method call that mutates ShapeEnv in a way that affects the
# issued guards, when ShapeEnv.produce_guards is called.
@dataclass
class ShapeEnvEvent:
    # ShapeEnv method.
    f: Callable

    # Arguments and keyword arguments called with.
    args: Optional[List[Any]] = None
    kwargs: Optional[Dict[str, Any]] = None

    # List of tracked_fakes at the time the method was called.
    tracked_fakes: Optional[List[Any]] = None
    # 跟踪的虚拟对象列表，初始值为 None

    # 被捕获事件的名称
    # 用于特殊处理特定方法
    name: Optional[str] = None

    # 返回使用 shape_env 作为 self 的重放结果字符串表示
    def __str__(self) -> str:
        # 如果有指定名称，则使用该名称；否则使用 self.f.__name__
        name = self.name if self.name is not None else self.f.__name__
        return f"event: {name} ({self.args}, {self.kwargs})"

    # 判断当前事件是否是 "_create_fx_call_function"
    def is_create_fx_call_function(self) -> bool:
        return self.name == "_create_fx_call_function"

    # 判断当前事件是否是 "evaluate_expr"
    def is_evaluate_expr(self) -> bool:
        return self.name == "evaluate_expr"

    # 判断当前事件是否是 "defer_runtime_assert"
    def is_defer_runtime_assert(self) -> bool:
        return self.name == "defer_runtime_assert"
# Extracts a ShapeEnv instance inside args and kwargs.
# Specifically, it looks for:
#   1. ShapeEnv arguments
#   2. SymInt, SymFloat, or SymBool arguments
# If we find more than one object of any of the above types, we
# also check that the ShapeEnv instance is the same for all of them.
def _extract_shape_env_and_assert_equal(args, kwargs):
    from torch.fx.experimental.symbolic_shapes import is_symbolic, ShapeEnv, SymTypes

    # Function to assert that a new ShapeEnv instance is equal to the old one
    def assert_equal(old: Optional[ShapeEnv], new: ShapeEnv) -> ShapeEnv:
        if old is not None:
            assert old is new, "call with different ShapeEnv"
        return new

    shape_env = None
    # Iterates through all arguments and keyword arguments
    for val in itertools.chain(args, kwargs.values()):
        # Checks if the value is an instance of ShapeEnv
        if isinstance(val, ShapeEnv):
            shape_env = assert_equal(shape_env, val)
        # Checks if the value is an instance of SymTypes and is symbolic
        if isinstance(val, SymTypes) and is_symbolic(val):
            # Retrieves the ShapeEnv associated with the symbolic object
            shape_env = assert_equal(shape_env, val.node.shape_env)

    return shape_env


# Decorator for recording the given function as a replayable event.
#
# This decorator should be used at every function that mutates the state of
# ShapeEnv in some way that affects the resulting issued guards (i.e. when
# ShapeEnv.produce_guards is called).
#
# save_tracked_fakes: saves a snapshot of the TrackedFake list.
# This is used when calling ShapeEnv.produce_guards at arbitrary points in time.
#
# When to save the list of TrackedFake?
# =====================================
# We should save the list of TrackedFake whenever the translation validation
# bisection may actually stop and call the produce_guards method at the moment
# right after the recorded function was played. In other words, since the
# bisection bisects through torch._assert calls, we should save in all methods
# that add a torch._assert call to the symbolic shapes FX graph.
#
# At the moment, there are 2 methods that save the list:
#   - ShapeEnv.evaluate_expr
#   - ShapeEnv.defer_runtime_assert
def record_shapeenv_event(*, save_tracked_fakes: bool = False) -> Callable:
    # Implementation of the decorator function that records ShapeEnv events
    pass
    # 定义一个装饰器函数 `decorator`，接受一个可调用对象作为参数，并返回一个装饰后的可调用对象
    def decorator(fn: Callable) -> Callable:
        # 断言被装饰的对象确实是可调用的
        assert callable(fn)
        # 使用 inspect 模块获取被装饰函数的参数规范，并断言第一个参数为 "self"
        args = inspect.getfullargspec(fn).args
        assert args and args[0] == "self", (
            "record_shapeenv_event should only wrap methods on ShapeEnv; refactor your "
            "code so that it calls into a method on ShapeEnv"
        )
        # 获取被装饰函数的名称
        name = fn.__name__

        # 定义装饰后的函数 wrapper，用于包裹原始函数 fn
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # 导入 ShapeEnv 类
            from torch.fx.experimental.symbolic_shapes import ShapeEnv

            # 断言第一个参数是 ShapeEnv 的实例
            assert isinstance(args[0], ShapeEnv)

            try:
                # 如果 ShapeEnv 实例正在记录事件
                if args[0].is_recording:  # type: ignore[has-type]
                    # 如果 ShapeEnv 已经在记录事件，直接调用被装饰的函数
                    return fn(*args, **kwargs)

                # 提取 ShapeEnv 实例，并确保所有参数和关键字参数引用的是相同的 ShapeEnv 实例
                self = _extract_shape_env_and_assert_equal(args, kwargs)

                # 如果没有任何 ShapeEnv 实例存在于参数中，则不记录事件，直接调用原始函数
                if self is None:
                    return fn(*args, **kwargs)

                # 否则，开始记录事件并调用函数
                with self._recording():
                    # 如果需要保存 tracked_fakes 的快照，则获取当前 tracked_fakes 的快照
                    tracked_fakes = (
                        self._snapshot_tracked_fakes() if save_tracked_fakes else None
                    )
                    # 创建 ShapeEnvEvent 对象来记录事件
                    event = ShapeEnvEvent(
                        fn, list(args), kwargs, tracked_fakes, name=fn.__name__
                    )
                    # 将事件添加到 ShapeEnv 实例的事件列表中
                    self.events.append(event)
                    try:
                        # 运行事件在当前 ShapeEnv 上的效果
                        return event.run(self)
                    except Exception:
                        # 如果运行事件发生异常，则从事件记录中移除该事件并抛出异常
                        self.events.pop()
                        raise

            except Exception:
                # 如果任何异常发生，记录错误日志
                log.error(  # noqa: G201
                    "failed while running %s(*%s, **%s)",
                    name,
                    args[1:],  # 记录参数列表，省略第一个参数 self
                    kwargs,
                    exc_info=log.isEnabledFor(logging.INFO),  # 如果 INFO 级别启用，则记录异常信息
                )
                raise

        return wrapper
    return decorator
# Replays the ShapeEnvEvents list.
# It assumes the first event is the constructor call.
#
# fn: transforms an old FX node into one corresponding to the newly created ShapeEnv.
def replay_shape_env_events(events):
    from torch.fx.experimental.symbolic_shapes import ShapeEnv  # 导入符号形状模块中的ShapeEnv类

    constructor_event = events[0]  # 获取事件列表中的第一个事件，假设是构造函数调用
    assert constructor_event.f == ShapeEnv  # 断言第一个事件的函数为ShapeEnv类的构造函数

    # Constructs the new ShapeEnv.
    shape_env = constructor_event.run()  # 运行构造函数事件，创建新的ShapeEnv对象

    for event in events[1:]:
        try:
            # Actually replays each event.
            # We need to call create_mapping_fn every time, since the node list might
            # change after each event is replayed.
            event.run(shape_env)  # 逐个重放剩余的事件，并传入当前的ShapeEnv对象
        except Exception as e:
            log.error("failed when running event: %s", event)  # 记录错误日志，指出运行失败的事件
            raise

    return shape_env  # 返回最终的ShapeEnv对象


# FakeTensor metadata.
# This is to be used in place of FakeTensor placeholders when calling
# ShapeEnv.produce_guards.
@dataclass
class FakeTensorMeta:
    tensor_size: Tuple[Union[int, torch.SymInt], ...]  # 张量尺寸，包含int或torch.SymInt类型的元组
    tensor_stride: Tuple[Union[int, torch.SymInt], ...]  # 张量步长，包含int或torch.SymInt类型的元组
    tensor_storage_offset: Union[int, torch.SymInt]  # 张量存储偏移量，可以是int或torch.SymInt类型
    is_nested: bool  # 是否为嵌套张量

    def size(self) -> Tuple[Union[int, torch.SymInt], ...]:
        return self.tensor_size  # 返回张量尺寸信息

    def stride(self) -> Tuple[Union[int, torch.SymInt], ...]:
        return self.tensor_stride  # 返回张量步长信息

    def storage_offset(self) -> Union[int, torch.SymInt]:
        return self.tensor_storage_offset  # 返回张量存储偏移量信息

    def dim(self) -> int:
        return len(self.tensor_size)  # 返回张量的维度数目

    @staticmethod
    def from_fake(fake) -> "FakeTensorMeta":
        return FakeTensorMeta(
            fake.size(), fake.stride(), fake.storage_offset(), fake.is_nested
        )  # 从给定的fake对象创建FakeTensorMeta实例的静态方法


# [Note: ShapeEnv State Equality]
# ===============================
#
# What is considered ShapeEnv state?
# ----------------------------------
# We consider to be the state of a ShapeEnv instance everything that
# is not in the inline tuple inside remove_nonstate_variables function.
# That is: the fields within ShapeEnv that modify the flow of execution
# of the program.
#
# So, for example: the replacements field might influence on how an
# expression is simplified. That, in turn, may result in a guard being
# statically known (i.e. not added).
#
# On the other hand, var_to_stack serves only changes what is printed
# in the screen, i.e. used only for debugging purposes. Therefore, we
# should not consider it when comparing states.
#
# What to do on NotEqualError?
# ----------------------------
# Here are a few possible causes for getting a NotEqualError raised:
#
#   1. New field that does not belong in the ShapeEnv state.
#      For example: log field of type ShapeEnvLoggerAdapter. Different
#      ShapeEnv instances will always have different ShapeEnvLoggerAdapter
#      instances, i.e. equality comparison would fail.
#      Solution: add it to the inlined tuple inside remove_nonstate_variables
#      function inside check_equal method.
#
# Checks whether the state of two ShapeEnv are equal w.r.t. the guards
# returned by ShapeEnv.produce_guards.
def shape_env_check_state_equal(env1, env2, non_state_variable_names, map_value):
    # Collect and remove variables that don't necessarily represent the state
    # of a ShapeEnv. Note: we copy the dictionary so that we don't modify the
    # instance itself.
    env1_vars = vars(env1).copy()  # 获取env1的所有实例变量，并创建其副本
    env2_vars = vars(env2).copy()  # 获取env2的所有实例变量，并创建其副本

    for v in non_state_variable_names:
        if v in env1_vars:
            env1_vars.pop(v)  # 移除env1_vars中的非状态变量
        if v in env2_vars:
            env2_vars.pop(v)  # 移除env2_vars中的非状态变量

    # Function for transforming the mismatched values into string.
    # Needed, since dict and set entries order might not be the same every time.
    def value_to_str(value: Any) -> str:
        if isinstance(value, dict):
            return (
                "{"  # 将字典转换为字符串表示，按键排序以保持一致性
                + ", ".join(f"{k}: {value[k]}" for k in sorted(value.keys(), key=str))
                + "}"
            )
        if isinstance(value, set):
            return "{" + ", ".join(f"{v}" for v in sorted(value)) + "}"  # 将集合转换为字符串表示，按值排序
        return str(value)  # 返回其他类型的值的字符串表示

    # Compares env1_vars with env2_vars.
    # Here, we allow the value of each field to be mapped, so that we appropriately
    # compare the two values.
    def compare_vars(
        map_value: Callable[[str, Any], Any]
    ):  # 定义一个函数用于比较env1_vars和env2_vars的值，允许通过map_value函数映射每个字段的值
    ) -> List[Tuple[str, str, str]]:
        # 将 env1_vars 和 env2_vars 转换为集合
        env1_set, env2_set = set(env1_vars), set(env2_vars)

        # 首先比较两个变量集合的键集合是否相同
        if env1_set != env2_set:
            # 如果不相同，抛出 NotEqualError 异常，附带错误信息
            raise NotEqualError(
                "field set mismatch:",
                [
                    (
                        "found unique fields:",
                        str(sorted(env1_set - env2_set)),  # 找到的 env1_vars 独有的字段
                        str(sorted(env2_set - env1_set)),  # 找到的 env2_vars 独有的字段
                    ),
                ],
            )

        # 然后，对键进行排序，并比较每个键的映射值
        sorted_keys = list(env1_set)
        sorted_keys.sort()

        # 创建一个列表，包含每个键及其在两个变量中的映射值
        mapped_dict = [
            (k, map_value(k, env1_vars[k]), map_value(k, env2_vars[k]))
            for k in sorted_keys
        ]

        # 返回一个列表，包含不匹配的字段及其对应的映射值
        return [
            (f"{k}: values don't match.", value_to_str(val1), value_to_str(val2))
            for k, val1, val2 in mapped_dict
            if val1 != val2
        ]

    # 调用 compare_vars 函数，获取不匹配的字段列表
    errors = compare_vars(map_value)

    # 如果存在不匹配的字段，抛出 NotEqualError 异常，附带错误信息
    if len(errors) > 0:
        raise NotEqualError("field values don't match:", errors)
class NotEqualError(Exception):
    # 自定义异常类，用于表示不相等的错误情况
    def __init__(
        self,
        msg: str,
        mismatched: List[Tuple[str, str, str]],
    ) -> None:
        # 格式化详细信息，展示每个不匹配的情况
        details = "\n".join(
            [
                "\n".join(
                    [
                        f"==> {inner_msg}",  # 显示具体内部消息
                        f"  >  Left: {str1}",  # 显示左侧字符串
                        f"  > Right: {str2}",  # 显示右侧字符串
                    ]
                )
                for inner_msg, str1, str2 in mismatched  # 遍历所有不匹配的情况
            ]
        )

        super().__init__(
            f"""\
ShapeEnv not equal: {msg}

{details}
"""
        )
```