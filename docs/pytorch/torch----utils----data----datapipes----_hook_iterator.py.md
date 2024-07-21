# `.\pytorch\torch\utils\data\datapipes\_hook_iterator.py`

```
# mypy: allow-untyped-defs
# 导入 functools 模块，用于支持函数式编程工具
import functools
# 导入 inspect 模块，用于对象的类型检查和获取对象信息
import inspect
# 导入 Enum 类型，用于定义枚举类型
from enum import Enum

# 导入 PyTorch 库
import torch


class _SnapshotState(Enum):
    r"""
    这些是 IterDataPipes 可能处于的与快照相关的状态。

    `NotStarted` - 允许恢复快照并创建一个带重置的迭代器
    `Restored` - 不能再次恢复，允许创建一个不重置数据管道的迭代器
    `Iterating` - 可以恢复，如果创建新的迭代器将重置数据管道
    """

    NotStarted = 0
    Restored = 1
    Iterating = 2


def _simplify_obj_name(obj) -> str:
    """简化对象的显示字符串，用于在 DataPipe 错误消息中渲染目的。"""
    if inspect.isfunction(obj):
        return obj.__name__
    else:
        return repr(obj)


def _strip_datapipe_from_name(name: str) -> str:
    # 从名称中移除 "IterDataPipe" 和 "MapDataPipe"
    return name.replace("IterDataPipe", "").replace("MapDataPipe", "")


def _generate_input_args_string(obj):
    """生成对象的输入参数字符串。"""
    signature = inspect.signature(obj.__class__)
    input_param_names = set(signature.parameters.keys())
    result = []
    for name, value in inspect.getmembers(obj):
        if name in input_param_names:
            result.append((name, _simplify_obj_name(value)))
    return ", ".join([f"{name}={value}" for name, value in result])


def _generate_iterdatapipe_msg(datapipe, simplify_dp_name: bool = False):
    # 生成 IterDataPipe 的消息字符串，包括其输入参数
    output_string = (
        f"{datapipe.__class__.__name__}({_generate_input_args_string(datapipe)})"
    )
    if simplify_dp_name:
        output_string = _strip_datapipe_from_name(output_string)
    return output_string


def _gen_invalid_iterdatapipe_msg(datapipe):
    # 生成无效 IterDataPipe 的消息字符串，提示用户可能存在多个对同一 IterDataPipe 的引用
    return (
        "This iterator has been invalidated because another iterator has been created "
        f"from the same IterDataPipe: {_generate_iterdatapipe_msg(datapipe)}\n"
        "This may be caused multiple references to the same IterDataPipe. We recommend "
        "using `.fork()` if that is necessary."
    )


_feedback_msg = (
    "\nFor feedback regarding this single iterator per IterDataPipe constraint, feel free "
    "to comment on this issue: https://github.com/pytorch/data/issues/45."
)


def _check_iterator_valid(datapipe, iterator_id, next_method_exists=False) -> None:
    r"""
    给定一个 DataPipe 实例和一个迭代器 ID，检查这些 ID 是否匹配，如果不匹配，则引发异常。

    对于 ChildDataPipe，还将比较 ID 是否与 main_datapipe 中存储的 ID 匹配。
    """
    # 如果 `next_method_exists` 为真，则进入此条件分支
    if next_method_exists:
        # 这种情况下，`IterDataPipe` 同时具有 `__iter__` 和 `__next__` 方法。
        # `_valid_iterator_id` 应该是未设置（`None`），或者被至多一个迭代器设置为 `0`。
        # 否则，表示存在多个迭代器。
        if datapipe._valid_iterator_id is not None and datapipe._valid_iterator_id != 0:
            extra_msg = "\nNote that this exception is raised inside your IterDataPipe's a `__next__` method"
            raise RuntimeError(
                _gen_invalid_iterdatapipe_msg(datapipe) + extra_msg + _feedback_msg
            )
    # 如果 `next_method_exists` 不为真，检查是否是子数据管道 `_is_child_datapipe` 为真
    elif (
        hasattr(datapipe, "_is_child_datapipe") and datapipe._is_child_datapipe is True
    ):
        # 如果子数据管道具有 `_check_valid_iterator_id` 方法
        if hasattr(datapipe, "_check_valid_iterator_id"):
            # 如果 `_check_valid_iterator_id` 方法返回假，则抛出运行时错误
            if not datapipe._check_valid_iterator_id(iterator_id):
                raise RuntimeError(
                    "This iterator has been invalidated, because a new iterator has been created "
                    f"from one of the ChildDataPipes of "
                    f"{_generate_iterdatapipe_msg(datapipe.main_datapipe)}."
                    + _feedback_msg
                )
        else:
            # 如果子数据管道没有 `_check_valid_iterator_id` 方法，则抛出运行时错误
            raise RuntimeError(
                "ChildDataPipe must have method `_check_valid_iterator_id`."
            )
    # 如果以上条件均不满足，检查 `_valid_iterator_id` 是否等于当前迭代器的 `iterator_id`
    elif datapipe._valid_iterator_id != iterator_id:
        # 如果不等，则抛出运行时错误
        raise RuntimeError(_gen_invalid_iterdatapipe_msg(datapipe) + _feedback_msg)
# 给定一个 DataPipe，更新其有效迭代器 ID 并重置 DataPipe
def _set_datapipe_valid_iterator_id(datapipe):
    # 检查是否为子 DataPipe，并且已经设置了_is_child_datapipe属性为True
    if hasattr(datapipe, "_is_child_datapipe") and datapipe._is_child_datapipe is True:
        # 如果是子 DataPipe，检查是否有 _set_main_datapipe_valid_iterator_id 方法，并调用它
        if hasattr(datapipe, "_set_main_datapipe_valid_iterator_id"):
            datapipe._set_main_datapipe_valid_iterator_id()  # 在适当时候内部调用reset()
        else:
            # 如果没有 _set_main_datapipe_valid_iterator_id 方法，抛出运行时错误
            raise RuntimeError(
                "ChildDataPipe must have method `_set_main_datapipe_valid_iterator_id`."
            )
    else:
        # 如果不是子 DataPipe，检查当前的有效迭代器 ID 是否为None
        if datapipe._valid_iterator_id is None:
            # 如果为None，则将有效迭代器 ID 设置为0
            datapipe._valid_iterator_id = 0
        else:
            # 如果不为None，则将有效迭代器 ID 增加1
            datapipe._valid_iterator_id += 1
        # 调用 DataPipe 的 reset 方法
        datapipe.reset()
    # 返回更新后的有效迭代器 ID
    return datapipe._valid_iterator_id


# 定义一个钩子函数，应用于所有 `_DataPipeMeta` 元类的 `__iter__` 方法
def hook_iterator(namespace):
    r"""
    This is done for the purpose of profiling and checking if an iterator is still valid.
    """
    
    # 定义一个函数，用于记录迭代器上下文的性能信息
    def profiler_record_fn_context(datapipe):
        # 如果 datapipe 没有 _profile_name 属性，为其生成迭代器数据管道消息
        if not hasattr(datapipe, "_profile_name"):
            datapipe._profile_name = _generate_iterdatapipe_msg(
                datapipe, simplify_dp_name=True
            )
        # 返回使用 torch.autograd.profiler.record_function 记录的 datapipe._profile_name
        return torch.autograd.profiler.record_function(datapipe._profile_name)
    # 定义一个名为 IteratorDecorator 的类，用于包装迭代器并修改其 `__next__` 方法
    class IteratorDecorator:
        r"""
        Wrap the iterator and modifying its `__next__` method.

        This decorator is applied to DataPipes of which `__iter__` method is NOT a generator function.
        Those `__iter__` method commonly returns `self` but not necessarily.
        """

        def __init__(self, iterator, datapipe, iterator_id, has_next_method):
            # 初始化方法，接收迭代器、数据管道、迭代器ID、是否有 `__next__` 方法的标志
            self.iterator = iterator
            self.datapipe = datapipe
            self.iterator_id = iterator_id
            self._profiler_enabled = torch.autograd._profiler_enabled()
            # 检查 `__iter__` 方法是否返回 `self`，以及数据管道是否有 `__next__` 方法
            self.self_and_has_next_method = (
                self.iterator is self.datapipe and has_next_method
            )

        def __iter__(self):
            # 返回自身，使得该类实例可以迭代
            return self

        def _get_next(self):
            """Return next with logic related to iterator validity, profiler, and incrementation of samples yielded."""
            # 检查迭代器的有效性，例如是否超出范围等
            _check_iterator_valid(self.datapipe, self.iterator_id)
            # 获取迭代器的下一个元素
            result = next(self.iterator)
            # 如果 `__iter__` 没有返回 `self` 或数据管道没有 `__next__` 方法，则增加已产生样本数的计数
            if not self.self_and_has_next_method:
                self.datapipe._number_of_samples_yielded += 1
            return result

        def __next__(self):
            # 实现迭代器的 `__next__` 方法
            # TODO: Add try-except to in-place reduce traceback from the Exception
            # See: https://github.com/pytorch/data/issues/284
            if self._profiler_enabled:
                # 如果启用了分析器，则在上下文中记录执行情况
                with profiler_record_fn_context(self.datapipe):
                    return self._get_next()
            else:
                # 否则直接调用 `_get_next()` 方法获取下一个元素
                # Decided against using `contextlib.nullcontext` for performance reasons
                return self._get_next()

        def __getattr__(self, name):
            # 如果访问的属性不存在于当前类中，则委托给迭代器对象处理
            return getattr(self.iterator, name)

    # 从命名空间中获取名为 `__iter__` 的函数或变量，并赋值给 `func`
    func = namespace["__iter__"]

    # ``__iter__`` 方法属于 IterDataPipe，是一个生成器函数
    # 检查函数是否为生成器函数
    if inspect.isgeneratorfunction(func):

        @functools.wraps(func)
        # 定义装饰器函数，用于生成器函数的包装
        def wrap_generator(*args, **kwargs):
            # 调用原始生成器函数，获取生成器对象
            gen = func(*args, **kwargs)
            # 获取数据管道对象
            datapipe = args[0]
            # 如果数据管道标记了快速迭代器
            if datapipe._fast_forward_iterator:
                # 从数据管道获取快速迭代器并清空标记
                it = datapipe._fast_forward_iterator
                datapipe._fast_forward_iterator = None
                # 设置数据管道的快照状态为迭代中
                datapipe._snapshot_state = _SnapshotState.Iterating
                # 循环迭代快速迭代器，直到遇到 StopIteration 异常
                while True:
                    try:
                        # 返回快速迭代器的下一个元素
                        yield next(it)
                    except StopIteration:
                        return
            # 为数据管道设置有效迭代器 ID，与每个创建的迭代器相关联
            iterator_id = _set_datapipe_valid_iterator_id(
                datapipe
            )  # 此 ID 与每个创建的迭代器相关联
            # 检查是否启用了分析器
            _profiler_enabled = torch.autograd._profiler_enabled()
            try:
                # 如果启用了分析器，使用 profiler_record_fn_context 上下文记录函数执行
                if _profiler_enabled:
                    with profiler_record_fn_context(datapipe):
                        response = gen.send(None)
                else:
                    # 否则直接发送 None 给生成器
                    response = gen.send(None)

                # 循环处理生成器的响应
                while True:
                    # 增加数据管道中已产出样本的计数
                    datapipe._number_of_samples_yielded += 1
                    # 接收下一个请求并传递给生成器，同时返回生成器的响应
                    request = yield response
                    # 如果启用了分析器，使用 profiler_record_fn_context 上下文记录函数执行
                    if _profiler_enabled:
                        with profiler_record_fn_context(datapipe):
                            _check_iterator_valid(datapipe, iterator_id)
                            response = gen.send(request)
                    else:
                        # 否则直接发送请求给生成器
                        _check_iterator_valid(datapipe, iterator_id)
                        response = gen.send(request)
            except StopIteration as e:
                # 捕获 StopIteration 异常并结束迭代
                return
            except Exception as e:
                # 捕获其它异常情况
                # TODO: 简化回溯消息以跳过 `response = gen.send(None)` 的部分
                #       参见 https://github.com/pytorch/data/issues/284
                # 获取相关数据管道和异常信息
                datapipe = args[0]
                msg = "thrown by __iter__ of"
                single_iterator_msg = "single iterator per IterDataPipe constraint"
                # 如果异常参数有长度属性
                if hasattr(e.args, "__len__"):
                    # 构建完整的异常消息
                    full_msg = f"{msg} {datapipe.__class__.__name__}({_generate_input_args_string(datapipe)})"
                    # 如果异常消息为空或者不是字符串
                    if len(e.args) == 0 or not isinstance(
                        e.args[0], str
                    ):  # 如果异常消息不存在
                        e.args = (f"\nThis exception is {full_msg}",)
                    # 如果异常消息不包含相关信息
                    elif msg not in e.args[0] and single_iterator_msg not in e.args[0]:
                        e.args = (
                            e.args[0] + f"\nThis exception is {full_msg}",
                        ) + e.args[1:]
                # 抛出异常
                raise

        # 将装饰后的生成器函数添加到 namespace 中的 "__iter__" 键下
        namespace["__iter__"] = wrap_generator
    else:  # ``__iter__`` of IterDataPipe is NOT a generator function
        # IterDataPipe 是一个既有 ``__iter__`` 又有 ``__next__`` 方法的迭代器
        # 并且 ``__iter__`` 方法可能返回 `self` 或者其他对象
        if "__next__" in namespace:  # 如果存在 `__next__` 方法，对其进行包装
            next_func = namespace["__next__"]

            @functools.wraps(next_func)
            def wrap_next(*args, **kwargs):
                datapipe = args[0]
                # 如果启用了 PyTorch 的分析器，则在上下文中记录调用信息
                if torch.autograd._profiler_enabled():
                    with profiler_record_fn_context(datapipe):
                        result = next_func(*args, **kwargs)
                else:
                    result = next_func(*args, **kwargs)
                # 增加迭代器已产生样本数量的计数
                datapipe._number_of_samples_yielded += 1
                return result

            namespace["__next__"] = wrap_next

            # 注意，如果 `__next__` 和 `__iter__` 执行完全不相关的操作，可能会引起问题，
            # 但是用户违反了迭代器协议。
            # 可能的问题包括：
            # 1. 有效迭代器 ID 可能不会正确更新或检查
            # 2. 产生的样本数量计数会错误

        # 无论是否存在 `__next__` 方法，都需要对 `__iter__` 进行包装以跟踪有效迭代器的数量
        @functools.wraps(func)
        def wrap_iter(*args, **kwargs):
            iter_ret = func(*args, **kwargs)
            datapipe = args[0]
            datapipe._snapshot_state = _SnapshotState.Iterating
            # 如果启用了快速前进迭代器，直接返回它
            if datapipe._fast_forward_iterator:
                iter_ret = datapipe._fast_forward_iterator
                datapipe._fast_forward_iterator = None
                return iter_ret
            # 设置数据管道的有效迭代器 ID
            iterator_id = _set_datapipe_valid_iterator_id(
                datapipe
            )  # 此 ID 与每个创建的迭代器相关联
            # 返回迭代器装饰器对象
            return IteratorDecorator(
                iter_ret, datapipe, iterator_id, "__next__" in namespace
            )

        namespace["__iter__"] = wrap_iter
```