# `.\pytorch\torch\utils\data\datapipes\datapipe.py`

```
# 导入 functools 模块，用于高阶函数操作
import functools
# 导入 pickle 模块，用于序列化和反序列化 Python 对象
import pickle
# 从 typing 模块导入类型相关的工具
from typing import Callable, Dict, Generic, Iterator, Optional, TypeVar

# 从 torch.utils._import_utils 模块导入 import_dill 函数
from torch.utils._import_utils import import_dill
# 从 torch.utils.data.datapipes._hook_iterator 模块导入 _SnapshotState 类
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
# 从 torch.utils.data.datapipes._typing 模块导入 _DataPipeMeta 和 _IterDataPipeMeta 类
from torch.utils.data.datapipes._typing import _DataPipeMeta, _IterDataPipeMeta
# 从 torch.utils.data.datapipes.utils.common 模块导入多个辅助函数
from torch.utils.data.datapipes.utils.common import (
    _deprecation_warning,
    _iter_deprecated_functional_names,
    _map_deprecated_functional_names,
)
# 从 torch.utils.data.dataset 模块导入 Dataset 和 IterableDataset 类
from torch.utils.data.dataset import Dataset, IterableDataset

# 导入 dill 库，并调用 import_dill 函数
dill = import_dill()
# 检查是否成功导入了 dill 库
HAS_DILL = dill is not None

# 定义公开的符号列表，包含可以从当前模块导入的名称
__all__ = [
    "DataChunk",
    "DFIterDataPipe",
    "IterDataPipe",
    "MapDataPipe",
]

# 定义一个类型变量 T，用于类型提示
T = TypeVar("T")
# 定义一个协变的类型变量 T_co，用于类型提示
T_co = TypeVar("T_co", covariant=True)

# 列出不可追踪的 DataFrame DataPipe 名称列表
UNTRACABLE_DATAFRAME_PIPES = [
    "batch",  # 由于返回 DataChunks
    "groupby",  # 由于返回 DataChunks
    "_dataframes_as_tuples",  # 由于解包 DF
    "trace_as_dataframe",  # 由于用于标记 DF 以进行追踪
]


class IterDataPipe(IterableDataset[T_co], metaclass=_IterDataPipeMeta):
    r"""
    Iterable-style DataPipe.

    All DataPipes that represent an iterable of data samples should subclass this.
    This style of DataPipes is particularly useful when data come from a stream, or
    when the number of samples is too large to fit them all in memory. ``IterDataPipe`` is lazily initialized and its
    elements are computed only when ``next()`` is called on the iterator of an ``IterDataPipe``.

    All subclasses should overwrite :meth:`__iter__`, which would return an
    iterator of samples in this DataPipe. Calling ``__iter__`` of an ``IterDataPipe`` automatically invokes its
    method ``reset()``, which by default performs no operation. When writing a custom ``IterDataPipe``, users should
    override ``reset()`` if necessary. The common usages include resetting buffers, pointers,
    and various state variables within the custom ``IterDataPipe``.

    Note:
        Only `one` iterator can be valid for each ``IterDataPipe`` at a time,
        and the creation a second iterator will invalidate the first one. This constraint is necessary because
        some ``IterDataPipe`` have internal buffers, whose states can become invalid if there are multiple iterators.
        The code example below presents details on how this constraint looks in practice.
        If you have any feedback related to this constraint, please see `GitHub IterDataPipe Single Iterator Issue`_.

    These DataPipes can be invoked in two ways, using the class constructor or applying their
    functional form onto an existing ``IterDataPipe`` (recommended, available to most but not all DataPipes).
    You can chain multiple `IterDataPipe` together to form a pipeline that will perform multiple
    operations in succession.

    .. _GitHub IterDataPipe Single Iterator Issue:
        https://github.com/pytorch/data/issues/45
    # 定义一个字典，用于存储各种函数的名称及其对应的可调用对象
    functions: Dict[str, Callable] = {}
    # 用于存储异常信息处理的钩子函数，当数据流处理时可以用来捕获和处理异常
    reduce_ex_hook: Optional[Callable] = None
    # 获取状态的钩子函数，用于获取数据流当前的状态信息
    getstate_hook: Optional[Callable] = None
    # 转换为字符串的钩子函数，将数据流对象转换为字符串表示形式时使用
    str_hook: Optional[Callable] = None
    # 对象的表示形式的钩子函数，用于生成数据流对象的表示形式
    repr_hook: Optional[Callable] = None
    # 当前有效的迭代器 ID，用于标识数据流的迭代器的唯一性
    _valid_iterator_id: Optional[int] = None
    # 已经产生的样本数量，记录数据流已经产生的样本数量，用于统计和管理
    _number_of_samples_yielded: int = 0
    # 快照状态，用于记录数据流的快照状态，可以是未开始、进行中或已完成
    _snapshot_state: _SnapshotState = _SnapshotState.NotStarted
    # 快速前进迭代器，用于在数据流上进行快速迭代，提高迭代效率
    _fast_forward_iterator: Optional[Iterator] = None

    # 实现迭代器接口，使对象可以通过迭代器进行遍历
    def __iter__(self) -> Iterator[T_co]:
        return self
    def __getattr__(self, attribute_name):
        # 检查属性名是否在函数字典中
        if attribute_name in IterDataPipe.functions:
            # 如果属性名是已弃用的函数名之一，则发出警告
            if attribute_name in _iter_deprecated_functional_names:
                kwargs = _iter_deprecated_functional_names[attribute_name]
                _deprecation_warning(**kwargs)
            # 获取函数对象并创建其部分应用
            f = IterDataPipe.functions[attribute_name]
            function = functools.partial(f, self)
            # 更新函数包装器，以保留原始函数的文档字符串
            functools.update_wrapper(wrapper=function, wrapped=f, assigned=("__doc__",))
            return function
        else:
            # 如果属性名不在函数字典中，则引发属性错误
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{attribute_name}"
            )

    @classmethod
    def register_function(cls, function_name, function):
        # 注册一个新的函数到类的函数字典中
        cls.functions[function_name] = function

    @classmethod
    def register_datapipe_as_function(
        cls, function_name, cls_to_register, enable_df_api_tracing=False
    ):
        # 如果函数名已存在于函数字典中，则抛出异常
        if function_name in cls.functions:
            raise Exception(
                f"Unable to add DataPipe function name {function_name} as it is already taken"
            )

        # 定义一个类函数，用于创建并返回一个新的数据管道对象
        def class_function(cls, enable_df_api_tracing, source_dp, *args, **kwargs):
            result_pipe = cls(source_dp, *args, **kwargs)
            # 如果返回的管道对象是 IterDataPipe 类的实例
            if isinstance(result_pipe, IterDataPipe):
                # 如果启用了数据框架 API 跟踪或者源数据管道是 DFIterDataPipe 类的实例
                if enable_df_api_tracing or isinstance(source_dp, DFIterDataPipe):
                    # 如果函数名不在不可跟踪的数据框架管道列表中，则跟踪为数据框架
                    if function_name not in UNTRACABLE_DATAFRAME_PIPES:
                        result_pipe = result_pipe.trace_as_dataframe()

            return result_pipe

        # 创建部分应用函数
        function = functools.partial(
            class_function, cls_to_register, enable_df_api_tracing
        )
        # 更新函数包装器，以保留原始类的文档字符串
        functools.update_wrapper(
            wrapper=function, wrapped=cls_to_register, assigned=("__doc__",)
        )
        # 将函数添加到类的函数字典中
        cls.functions[function_name] = function

    def __getstate__(self):
        """
        Serialize `lambda` functions when `dill` is available.

        If this doesn't cover your custom DataPipe's use case, consider writing custom methods for
        `__getstate__` and `__setstate__`, or use `pickle.dumps` for serialization.
        """
        # 返回对象的当前状态字典
        state = self.__dict__
        # 如果定义了 getstate_hook，调用它并返回结果
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        # 否则直接返回对象的状态字典
        return state

    def __reduce_ex__(self, *args, **kwargs):
        # 如果定义了 reduce_ex_hook，尝试调用它来实现对象的高级序列化
        if IterDataPipe.reduce_ex_hook is not None:
            try:
                return IterDataPipe.reduce_ex_hook(self)
            # 如果 reduce_ex_hook 抛出 NotImplementedError，则继续使用默认的序列化方法
            except NotImplementedError:
                pass
        # 调用父类的默认序列化方法
        return super().__reduce_ex__(*args, **kwargs)

    @classmethod
    def set_getstate_hook(cls, hook_fn):
        # 如果已定义 getstate_hook，并且尝试设置一个新的钩子函数，则引发运行时错误
        if IterDataPipe.getstate_hook is not None and hook_fn is not None:
            raise RuntimeError("Attempt to override existing getstate_hook")
        # 设置新的 getstate_hook
        IterDataPipe.getstate_hook = hook_fn

    @classmethod
    # 设置静态方法，用于设置 reduce_ex_hook，防止覆盖已存在的钩子函数
    def set_reduce_ex_hook(cls, hook_fn):
        # 如果 reduce_ex_hook 已经存在并且 hook_fn 不为 None，则抛出运行时错误
        if IterDataPipe.reduce_ex_hook is not None and hook_fn is not None:
            raise RuntimeError("Attempt to override existing reduce_ex_hook")
        # 设置 IterDataPipe 的 reduce_ex_hook 为指定的 hook_fn
        IterDataPipe.reduce_ex_hook = hook_fn

    # 返回对象的字符串表示形式
    def __repr__(self):
        # 如果定义了 repr_hook，则使用该钩子函数返回对象的字符串表示形式
        if self.repr_hook is not None:
            return self.repr_hook(self)
        # 否则返回对象的类名，而不是默认的对象内存地址信息
        return str(self.__class__.__qualname__)

    # 返回对象的简要字符串表示形式
    def __str__(self):
        # 如果定义了 str_hook，则使用该钩子函数返回对象的简要字符串表示形式
        if self.str_hook is not None:
            return self.str_hook(self)
        # 否则返回对象的类名，而不是默认的对象内存地址信息
        return str(self.__class__.__qualname__)

    # 返回对象的属性列表，用于交互式环境的自动补全（例如 Jupyter notebook）
    def __dir__(self):
        # 返回父类的属性列表和当前对象的 functions 字典的键组成的列表
        return list(super().__dir__()) + list(self.functions.keys())

    # 重置 IterDataPipe 对象到初始状态的方法
    def reset(self) -> None:
        r"""
        Reset the `IterDataPipe` to the initial state.

        By default, no-op. For subclasses of `IterDataPipe`, depending on their functionalities,
        they may want to override this method with implementations that
        may clear the buffers and reset pointers of the DataPipe.
        The `reset` method is always called when `__iter__` is called as part of `hook_iterator`.
        """
        pass
    # 定义一个数据管道类，用于处理映射式数据
    class DFIterDataPipe(IterDataPipe):
        # 返回True，表示当前数据管道类是DFPipe类型
        def _is_dfpipe(self):
            return True


    # MapDataPipe类继承自Dataset类，并使用_DataPipeMeta元类
    class MapDataPipe(Dataset[T_co], metaclass=_DataPipeMeta):
        """
        Map-style DataPipe.

        All datasets that represent a map from keys to data samples should subclass this.
        Subclasses should overwrite :meth:`__getitem__`, supporting fetching a
        data sample for a given, unique key. Subclasses can also optionally overwrite
        :meth:`__len__`, which is expected to return the size of the dataset by many
        :class:`~torch.utils.data.Sampler` implementations and the default options
        of :class:`~torch.utils.data.DataLoader`.

        These DataPipes can be invoked in two ways, using the class constructor or applying their
        functional form onto an existing `MapDataPipe` (recommend, available to most but not all DataPipes).

        Note:
            :class:`~torch.utils.data.DataLoader` by default constructs an index
            sampler that yields integral indices. To make it work with a map-style
            DataPipe with non-integral indices/keys, a custom sampler must be provided.

        Example:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.map import SequenceWrapper, Mapper
            >>> dp = SequenceWrapper(range(10))
            >>> map_dp_1 = dp.map(lambda x: x + 1)  # Using functional form (recommended)
            >>> list(map_dp_1)
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            >>> map_dp_2 = Mapper(dp, lambda x: x + 1)  # Using class constructor
            >>> list(map_dp_2)
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            >>> batch_dp = map_dp_1.batch(batch_size=2)
            >>> list(batch_dp)
            [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        """

        # 类级别的字典，存储各种注册的函数
        functions: Dict[str, Callable] = {}
        # 可选的钩子函数，用于扩展功能
        reduce_ex_hook: Optional[Callable] = None
        getstate_hook: Optional[Callable] = None
        str_hook: Optional[Callable] = None
        repr_hook: Optional[Callable] = None

        # 获取属性的特殊方法，用于动态调用注册的函数
        def __getattr__(self, attribute_name):
            if attribute_name in MapDataPipe.functions:
                # 如果属性名在已注册的函数字典中
                if attribute_name in _map_deprecated_functional_names:
                    kwargs = _map_deprecated_functional_names[attribute_name]
                    _deprecation_warning(**kwargs)
                # 获取对应的函数并创建偏函数，绑定到当前实例
                f = MapDataPipe.functions[attribute_name]
                function = functools.partial(f, self)
                # 更新偏函数的文档字符串和包装信息
                functools.update_wrapper(wrapper=function, wrapped=f, assigned=("__doc__",))
                return function
            else:
                # 若属性名不在注册的函数字典中，则抛出异常
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{attribute_name}"
                )

        # 类方法，用于注册新的函数到functions字典中
        @classmethod
        def register_function(cls, function_name, function):
            cls.functions[function_name] = function

        @classmethod
    def register_datapipe_as_function(cls, function_name, cls_to_register):
        if function_name in cls.functions:
            raise Exception(  # 抛出异常，如果函数名已存在于类的函数字典中
                f"Unable to add DataPipe function name {function_name} as it is already taken"
            )

        def class_function(cls, source_dp, *args, **kwargs):
            # 创建一个新的类函数，用于将源数据管道和其他参数传递给指定的类，返回结果管道
            result_pipe = cls(source_dp, *args, **kwargs)
            return result_pipe

        function = functools.partial(class_function, cls_to_register)
        # 更新函数的包装器，使其保留被包装函数的文档字符串
        functools.update_wrapper(
            wrapper=function, wrapped=cls_to_register, assigned=("__doc__",)
        )
        # 将新函数注册到类的函数字典中
        cls.functions[function_name] = function

    def __getstate__(self):
        """
        序列化 `lambda` 函数（当 `dill` 可用时）。

        如果这不符合您的自定义 DataPipe 的用例，请考虑编写 `__getstate__` 和 `__setstate__` 的自定义方法，
        或使用 `pickle.dumps` 进行序列化。
        """
        state = self.__dict__
        if MapDataPipe.getstate_hook is not None:
            return MapDataPipe.getstate_hook(state)
        return state

    def __reduce_ex__(self, *args, **kwargs):
        if MapDataPipe.reduce_ex_hook is not None:
            try:
                return MapDataPipe.reduce_ex_hook(self)
            except NotImplementedError:
                pass
        # 如果没有 reduce_ex_hook 或者 hook_fn 抛出 NotImplementedError，则调用超类的默认实现
        return super().__reduce_ex__(*args, **kwargs)

    @classmethod
    def set_getstate_hook(cls, hook_fn):
        # 设置自定义的 getstate_hook，用于序列化时的钩子函数
        if MapDataPipe.getstate_hook is not None and hook_fn is not None:
            raise RuntimeError("Attempt to override existing getstate_hook")
        MapDataPipe.getstate_hook = hook_fn

    @classmethod
    def set_reduce_ex_hook(cls, hook_fn):
        # 设置自定义的 reduce_ex_hook，用于高级序列化时的钩子函数
        if MapDataPipe.reduce_ex_hook is not None and hook_fn is not None:
            raise RuntimeError("Attempt to override existing reduce_ex_hook")
        MapDataPipe.reduce_ex_hook = hook_fn

    def __repr__(self):
        if self.repr_hook is not None:
            return self.repr_hook(self)
        # 如果设置了自定义的 repr_hook，则返回其返回值；否则返回类的限定名
        return str(self.__class__.__qualname__)

    def __str__(self):
        if self.str_hook is not None:
            return self.str_hook(self)
        # 如果设置了自定义的 str_hook，则返回其返回值；否则返回类的限定名
        return str(self.__class__.__qualname__)

    def __dir__(self):
        # 用于在 REPL（如 Jupyter 笔记本）中实现自动补全功能，返回类及其函数字典的键
        return list(super().__dir__()) + list(self.functions.keys())
# 定义一个包装器类 `_DataPipeSerializationWrapper`，用于序列化数据管道对象
class _DataPipeSerializationWrapper:
    def __init__(self, datapipe):
        self._datapipe = datapipe  # 初始化时保存数据管道对象的引用

    # 定义序列化方法，返回数据管道对象的序列化状态
    def __getstate__(self):
        use_dill = False  # 默认不使用 dill 序列化
        try:
            value = pickle.dumps(self._datapipe)  # 尝试使用 pickle 序列化数据管道对象
        except Exception:
            if HAS_DILL:  # 如果支持 dill 序列化，则使用 dill 序列化
                value = dill.dumps(self._datapipe)
                use_dill = True
            else:
                raise  # 抛出异常
        return (value, use_dill)  # 返回序列化后的数据及是否使用了 dill

    # 定义反序列化方法，根据状态恢复数据管道对象
    def __setstate__(self, state):
        value, use_dill = state
        if use_dill:
            self._datapipe = dill.loads(value)  # 使用 dill 还原数据管道对象
        else:
            self._datapipe = pickle.loads(value)  # 使用 pickle 还原数据管道对象

    # 定义长度方法，返回数据管道对象的长度
    def __len__(self):
        try:
            return len(self._datapipe)
        except Exception as e:
            raise TypeError(
                f"{type(self).__name__} instance doesn't have valid length"
            ) from e  # 抛出类型错误异常，说明对象没有有效长度


# 定义 `_IterDataPipeSerializationWrapper` 类，继承自 `_DataPipeSerializationWrapper` 和 `IterDataPipe` 类
class _IterDataPipeSerializationWrapper(_DataPipeSerializationWrapper, IterDataPipe):
    def __init__(self, datapipe: IterDataPipe[T_co]):
        super().__init__(datapipe)
        self._datapipe_iter: Optional[Iterator[T_co]] = None  # 初始化迭代器为 None

    # 实现迭代器接口，返回迭代器对象自身
    def __iter__(self) -> "_IterDataPipeSerializationWrapper":
        self._datapipe_iter = iter(self._datapipe)  # 使用数据管道对象创建迭代器
        return self

    # 实现迭代器的下一个方法，返回迭代器的下一个元素
    def __next__(self) -> T_co:  # type: ignore[type-var]
        assert self._datapipe_iter is not None  # 断言迭代器不为空
        return next(self._datapipe_iter)


# 定义 `_MapDataPipeSerializationWrapper` 类，继承自 `_DataPipeSerializationWrapper` 和 `MapDataPipe` 类
class _MapDataPipeSerializationWrapper(_DataPipeSerializationWrapper, MapDataPipe):
    # 实现索引访问方法，返回数据管道中指定索引的元素
    def __getitem__(self, idx):
        return self._datapipe[idx]


# 定义 `DataChunk` 类，继承自 `list`，支持泛型 `T`
class DataChunk(list, Generic[T]):
    def __init__(self, items):
        super().__init__(items)  # 调用父类构造方法初始化列表
        self.items = items  # 保存传入的项目列表

    # 将列表转换为字符串表示，带有缩进
    def as_str(self, indent=""):
        res = indent + "[" + ", ".join(str(i) for i in iter(self)) + "]"  # 将列表元素转换为字符串，并加上缩进
        return res

    # 实现迭代器接口，返回迭代器对象
    def __iter__(self) -> Iterator[T]:
        yield from super().__iter__()  # 使用父类的迭代器方法进行迭代

    # 返回原始迭代器，用于类型提示，但不具体实现类型
    def raw_iterator(self) -> T:  # type: ignore[misc]
        yield from self.items  # 返回项目列表的迭代器
```