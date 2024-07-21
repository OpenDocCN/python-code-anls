# `.\pytorch\torch\utils\data\datapipes\_decorator.py`

```py
# mypy: allow-untyped-defs
# 导入检查模块
import inspect
# 导入装饰器相关函数
from functools import wraps
# 导入类型提示相关模块
from typing import Any, Callable, get_type_hints, Optional, Type, Union

# 导入内部定义的类型和类
from torch.utils.data.datapipes._typing import _DataPipeMeta
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe


######################################################
# Functional API
######################################################

# 定义函数式数据管道装饰器类
class functional_datapipe:
    name: str

    def __init__(self, name: str, enable_df_api_tracing=False) -> None:
        """
        Define a functional datapipe.

        Args:
            enable_df_api_tracing - if set, any returned DataPipe would accept
            DataFrames API in tracing mode.
        """
        self.name = name
        self.enable_df_api_tracing = enable_df_api_tracing

    def __call__(self, cls):
        # 如果被装饰的类是 IterDataPipe 的子类
        if issubclass(cls, IterDataPipe):
            # 如果 cls 是类型对象
            if isinstance(cls, Type):  # type: ignore[arg-type]
                # 如果 cls 不是 _DataPipeMeta 的实例，抛出类型错误
                if not isinstance(cls, _DataPipeMeta):
                    raise TypeError(
                        "`functional_datapipe` can only decorate IterDataPipe"
                    )
            # 如果 cls 是对象实例
            else:
                # 如果 cls 不是 non_deterministic 的实例，并且不是 non_deterministic 的实例方法
                if not isinstance(cls, non_deterministic) and not (
                    hasattr(cls, "__self__")
                    and isinstance(cls.__self__, non_deterministic)
                ):
                    raise TypeError(
                        "`functional_datapipe` can only decorate IterDataPipe"
                    )
            # 注册函数式数据管道到 IterDataPipe 类
            IterDataPipe.register_datapipe_as_function(
                self.name, cls, enable_df_api_tracing=self.enable_df_api_tracing
            )
        # 如果被装饰的类是 MapDataPipe 的子类
        elif issubclass(cls, MapDataPipe):
            # 注册函数式数据管道到 MapDataPipe 类
            MapDataPipe.register_datapipe_as_function(self.name, cls)

        # 返回被装饰的类
        return cls


######################################################
# Determinism
######################################################

# 全局变量，用于标记数据管道的确定性
_determinism: bool = False


# 确保数据管道的确定性装饰器类
class guaranteed_datapipes_determinism:
    prev: bool

    def __init__(self) -> None:
        global _determinism
        # 保存当前确定性标记的状态，并设置为 True
        self.prev = _determinism
        _determinism = True

    def __enter__(self) -> None:
        # 进入方法，暂时不做处理
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _determinism
        # 退出方法，恢复之前保存的确定性标记状态
        _determinism = self.prev


# 非确定性数据管道标记类
class non_deterministic:
    cls: Optional[Type[IterDataPipe]] = None
    # TODO: Lambda for picking
    # 非确定性函数的定义
    deterministic_fn: Callable[[], bool]
    def __init__(self, arg: Union[Type[IterDataPipe], Callable[[], bool]]) -> None:
        # 1. 如果装饰器没有参数
        if isinstance(arg, Type):  # type: ignore[arg-type]
            # 如果参数是一个类而不是函数
            if not issubclass(arg, IterDataPipe):  # type: ignore[arg-type]
                # 抛出类型错误，只有 `IterDataPipe` 类可以使用 `non_deterministic` 装饰器
                raise TypeError(
                    "Only `IterDataPipe` can be decorated with `non_deterministic`"
                    f", but {arg.__name__} is found"
                )
            # 将类赋值给 self.cls
            self.cls = arg  # type: ignore[assignment]
        # 2. 如果装饰器带有一个函数参数
        #    这个类应该根据不同的输入表现不同。使用这个函数来验证每个实例的确定性。
        #    当函数返回 True 时，实例是非确定性的。否则，实例是确定性的 DataPipe。
        elif isinstance(arg, Callable):  # type:ignore[arg-type]
            # 将函数赋值给 self.deterministic_fn
            self.deterministic_fn = arg  # type: ignore[assignment, misc]
        else:
            # 抛出类型错误，无法使用 `non_deterministic` 装饰器装饰指定的参数
            raise TypeError(f"{arg} can not be decorated by non_deterministic")

    def __call__(self, *args, **kwargs):
        global _determinism
        # 装饰 IterDataPipe
        if self.cls is not None:
            if _determinism:
                # 如果设置了 'guaranteed_datapipes_determinism'，但是 self.cls 是非确定性的，则抛出类型错误
                raise TypeError(
                    f"{self.cls.__name__} is non-deterministic, but you set 'guaranteed_datapipes_determinism'. "
                    "You can turn off determinism for this DataPipe if that is acceptable "
                    "for your application"
                )
            # 返回通过 self.cls 创建的实例
            return self.cls(*args, **kwargs)  # type: ignore[call-arg]

        # 装饰具有函数参数的情况
        if not (
            isinstance(args[0], type)
            and issubclass(args[0], IterDataPipe)  # type: ignore[arg-type]
        ):
            # 抛出类型错误，只有 `IterDataPipe` 类可以被装饰，但是找到了 {args[0].__name__}
            raise TypeError(
                f"Only `IterDataPipe` can be decorated, but {args[0].__name__} is found"
            )
        # 将第一个参数类赋值给 self.cls
        self.cls = args[0]
        # 返回 deterministic_wrapper_fn 函数
        return self.deterministic_wrapper_fn

    def deterministic_wrapper_fn(self, *args, **kwargs) -> IterDataPipe:
        # 调用 deterministic_fn 函数来获取结果
        res = self.deterministic_fn(*args, **kwargs)  # type: ignore[call-arg, misc]
        # 如果结果不是布尔类型，则抛出类型错误
        if not isinstance(res, bool):
            raise TypeError(
                "deterministic_fn of `non_deterministic` decorator is required "
                f"to return a boolean value, but {type(res)} is found"
            )
        global _determinism
        # 如果设置了 `_determinism` 并且 res 为 True，则抛出类型错误
        if _determinism and res:
            raise TypeError(
                f"{self.cls.__name__} is non-deterministic with the inputs, but you set "
                "'guaranteed_datapipes_determinism'. You can turn off determinism "
                "for this DataPipe if that is acceptable for your application"
            )
        # 返回通过 self.cls 创建的实例
        return self.cls(*args, **kwargs)  # type: ignore[call-arg, misc]
######################################################
# Type validation
######################################################

# Validate each argument of DataPipe with hint as a subtype of the hint.
def argument_validation(f):
    # 获取函数 f 的参数签名
    signature = inspect.signature(f)
    # 获取函数 f 的类型提示信息
    hints = get_type_hints(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        # 绑定参数
        bound = signature.bind(*args, **kwargs)
        # 遍历绑定的参数及其对应的值
        for argument_name, value in bound.arguments.items():
            # 检查参数名是否在类型提示中，并且其类型是 _DataPipeMeta 的子类
            if argument_name in hints and isinstance(
                hints[argument_name], _DataPipeMeta
            ):
                hint = hints[argument_name]
                # 如果值不是 IterDataPipe 类型，则引发 TypeError
                if not isinstance(value, IterDataPipe):
                    raise TypeError(
                        f"Expected argument '{argument_name}' as a IterDataPipe, but found {type(value)}"
                    )
                # 如果值的类型不是提示的类型的子类型，则引发 TypeError
                if not value.type.issubtype(hint.type):
                    raise TypeError(
                        f"Expected type of argument '{argument_name}' as a subtype of "
                        f"hint {hint.type}, but found {value.type}"
                    )

        return f(*args, **kwargs)

    return wrapper


# Default value is True
# 运行时验证开关，默认为 True
_runtime_validation_enabled: bool = True


class runtime_validation_disabled:
    prev: bool

    def __init__(self) -> None:
        global _runtime_validation_enabled
        # 保存当前运行时验证的状态，并关闭运行时验证
        self.prev = _runtime_validation_enabled
        _runtime_validation_enabled = False

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _runtime_validation_enabled
        # 恢复之前保存的运行时验证状态
        _runtime_validation_enabled = self.prev


# Runtime checking
# 运行时验证装饰器，验证输出数据是否是返回类型提示的子类型
def runtime_validation(f):
    # TODO:
    # Can be extended to validate '__getitem__' and nonblocking
    # 如果函数名不是 "__iter__"，则无法使用 'runtime_validation' 装饰器
    if f.__name__ != "__iter__":
        raise TypeError(
            f"Can not decorate function {f.__name__} with 'runtime_validation'"
        )

    @wraps(f)
    def wrapper(self):
        global _runtime_validation_enabled
        # 如果运行时验证被禁用，则直接返回函数的生成器
        if not _runtime_validation_enabled:
            yield from f(self)
        else:
            # 否则，获取函数的生成器，并逐个验证生成的数据项
            it = f(self)
            for d in it:
                # 如果生成的数据项不是指定类型的实例的子类型，则引发 RuntimeError
                if not self.type.issubtype_of_instance(d):
                    raise RuntimeError(
                        f"Expected an instance as subtype of {self.type}, but found {d}({type(d)})"
                    )
                yield d

    return wrapper
```