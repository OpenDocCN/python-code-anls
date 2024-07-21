# `.\pytorch\torch\utils\data\datapipes\iter\callable.py`

```
# 设置类型提示允许未被类型化的函数定义
mypy: allow-untyped-defs

# 导入 functools 库，用于支持函数式编程中的偏函数应用
import functools

# 导入 namedtuple 类型，用于创建命名元组
from collections import namedtuple

# 导入类型提示相关的模块和类
from typing import Any, Callable, Dict, Iterator, List, Optional, Sized, TypeVar, Union

# 导入 torch.utils.data._utils.collate 模块中的 default_collate 函数
from torch.utils.data._utils.collate import default_collate

# 导入 torch.utils.data.datapipes._decorator 模块中的 functional_datapipe 装饰器
from torch.utils.data.datapipes._decorator import functional_datapipe

# 导入 torch.utils.data.datapipes.dataframe 模块中的 dataframe_wrapper 别名 df_wrapper
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper

# 导入 torch.utils.data.datapipes.datapipe 模块中的 IterDataPipe 类
from torch.utils.data.datapipes.datapipe import IterDataPipe

# 导入 torch.utils.data.datapipes.utils.common 模块中的函数和方法
from torch.utils.data.datapipes.utils.common import (
    _check_unpickable_fn,  # 校验不可 pickle 的函数
    validate_input_col,    # 验证输入列
)

# 声明 __all__ 变量，列出当前模块公开的类和函数
__all__ = [
    "CollatorIterDataPipe",  # 数据整理迭代器数据管道
    "MapperIterDataPipe",    # 映射器迭代器数据管道
]

# TypeVar 类型变量，用于协变泛型
T_co = TypeVar("T_co", covariant=True)

# 使用 functional_datapipe 装饰器标记 MapperIterDataPipe 类为“map”函数的函数式数据管道
@functional_datapipe("map")
class MapperIterDataPipe(IterDataPipe[T_co]):
    """
    对源数据管道中的每个项应用函数（函数名为“map”）。

    函数可以是任何常规的 Python 函数或偏函数对象。不建议使用 Lambda 函数，因为它们不支持 pickle。

    Args:
        datapipe: 源可迭代数据管道
        fn: 应用于每个项的函数
        input_col: 应用“fn”的数据索引或索引，例如：

            - 默认为“None”，直接应用“fn”到数据。
            - 用于列表/元组的整数（s）。
            - 用于字典的键（s）。

        output_col: 存放“fn”结果的数据索引。“output_col”只能在“input_col”不为“None”时指定

            - 默认为“None”，替换“input_col”指定的索引；对于具有多个索引的“input_col”，使用最左边的索引，其余索引将被移除。
            - 用于列表/元组的整数。“-1”表示在末尾追加结果。
            - 用于字典的键。接受新的键。

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper, Mapper
        >>> def add_one(x):
        ...     return x + 1
        >>> dp = IterableWrapper(range(10))
        >>> map_dp_1 = dp.map(add_one)  # 优先使用函数式形式调用
        >>> list(map_dp_1)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> # 不建议使用 Lambda 函数，因为它们无法与“pickle”序列化
        >>> # 使用 functools.partial 或显式定义函数替代
        >>> map_dp_2 = Mapper(dp, lambda x: x + 1)
        >>> list(map_dp_2)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    """

    # 类属性：源数据管道和应用的函数
    datapipe: IterDataPipe
    fn: Callable

    # 初始化方法，接受源数据管道、应用函数以及可选的输入和输出索引
    def __init__(
        self,
        datapipe: IterDataPipe,
        fn: Callable,
        input_col=None,
        output_col=None,
    ) -> None:
        super().__init__()
        self.datapipe = datapipe

        _check_unpickable_fn(fn)
        self.fn = fn  # type: ignore[assignment]

        self.input_col = input_col
        if input_col is None and output_col is not None:
            raise ValueError("`output_col` must be None when `input_col` is None.")
        if isinstance(output_col, (list, tuple)):
            if len(output_col) > 1:
                raise ValueError("`output_col` must be a single-element list or tuple")
            output_col = output_col[0]
        self.output_col = output_col
        validate_input_col(fn, input_col)

    # 应用函数到数据的方法
    def _apply_fn(self, data):
        if self.input_col is None and self.output_col is None:
            # 若无输入列和输出列，则直接应用函数到数据
            return self.fn(data)

        if self.input_col is None:
            # 若只有输出列而没有输入列，则应用函数到数据
            res = self.fn(data)
        elif isinstance(self.input_col, (list, tuple)):
            # 若输入列是列表或元组，则将对应列的数据作为参数应用函数
            args = tuple(data[col] for col in self.input_col)
            res = self.fn(*args)
        else:
            # 否则，按照输入列索引应用函数到数据
            res = self.fn(data[self.input_col])

        # 若数据是元组，则将其复制为列表以便进行就地修改，因为元组是不可变的
        if isinstance(data, tuple):
            t_flag = True
            data = list(data)
        else:
            t_flag = False

        if self.output_col is None:
            if isinstance(self.input_col, (list, tuple)):
                # 如果没有指定输出列，根据输入列设置数据的值，并删除多余的列
                data[self.input_col[0]] = res
                for idx in sorted(self.input_col[1:], reverse=True):
                    del data[idx]
            else:
                # 否则，根据输出列索引设置数据的值
                data[self.input_col] = res
        else:
            # 如果指定了输出列
            if self.output_col == -1:
                # 若输出列为-1，则将结果追加到数据末尾
                data.append(res)
            else:
                # 否则，根据输出列索引设置数据的值
                data[self.output_col] = res

        # 如果之前将元组转换为列表，则将其转换回元组
        return tuple(data) if t_flag else data

    # 实现迭代器接口
    def __iter__(self) -> Iterator[T_co]:
        for data in self.datapipe:
            # 对数据流中的每个数据应用函数并返回结果
            yield self._apply_fn(data)

    # 实现获取长度的方法
    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized):
            # 如果数据管道具有长度属性，则返回其长度
            return len(self.datapipe)
        # 否则，抛出类型错误异常
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
# 辅助函数，用于协助整理数据批次
def _collate_helper(conversion, item):
    # TODO(VitalyFedyunin): 验证 item 是否是任何一种批次数据类型
    if len(item.items) > 1:
        # TODO(VitalyFedyunin): 将所有批次的 DataFrame 压缩成一个
        raise RuntimeError("Only supports one DataFrame per batch")
    df = item[0]
    # 从数据框架获取列名
    columns_name = df_wrapper.get_columns(df)
    tuple_names: List = []
    tuple_values: List = []

    # 检查 conversion 中的键是否在列名中存在，否则抛出错误
    for name in conversion.keys():
        if name not in columns_name:
            raise RuntimeError("Conversion keys missmatch")

    # 遍历列名，如果列名在 conversion 中存在，则检查其值是否可调用，否则抛出错误
    for name in columns_name:
        if name in conversion:
            if not callable(conversion[name]):
                raise RuntimeError(
                    "Collate (DF)DataPipe requires callable as dict values"
                )
            collation_fn = conversion[name]
        else:
            # TODO(VitalyFedyunin): 在 df_wrapper 中添加默认整理函数
            try:
                import torcharrow.pytorch as tap  # type: ignore[import]

                collation_fn = tap.rec.Default()
            except Exception as e:
                raise RuntimeError(
                    "unable to import default collation function from the TorchArrow"
                ) from e

        tuple_names.append(str(name))
        # 使用整理函数处理对应列的数据，并将结果添加到 tuple_values 中
        value = collation_fn(df[name])
        tuple_values.append(value)

    # TODO(VitalyFedyunin): 可以在这里动态提取 tuple_values 的类型
    # TODO(VitalyFedyunin): 确保 tuple_names 不为空，而不是忽略 mypy 错误
    # 创建命名元组类，用于存储整理后的结果
    tpl_cls = namedtuple("CollateResult", tuple_names)  # type: ignore[misc]
    tuple = tpl_cls(*tuple_values)
    return tuple


# 数据管道的映射迭代器数据管道，用于批次数据的整理
@functional_datapipe("collate")
class CollatorIterDataPipe(MapperIterDataPipe):
    r"""
    通过自定义整理函数（功能名称为 ``collate`` ），将数据管道中的样本整理成张量。

    默认情况下，使用 :func:`torch.utils.data.default_collate`。

    .. note::
        编写自定义整理函数时，可以导入 :func:`torch.utils.data.default_collate` 获取默认行为，
        以及 `functools.partial` 用于指定任何额外的参数。

    Args:
        datapipe: 要整理的可迭代数据管道
        collate_fn: 自定义整理函数，用于收集和组合数据或数据批次。
            默认函数根据数据类型整理成张量。
    # 定义一个名为 `__init__` 的构造函数，用于初始化一个数据管道对象
    def __init__(
        self,
        datapipe: IterDataPipe,  # 参数 datapipe，类型为 IterDataPipe，表示输入的数据管道对象
        conversion: Optional[  # 参数 conversion，可选类型，可以是以下两种之一：
            Union[  # Union 表示可以是多种类型中的一种
                Callable[..., Any],  # 可调用对象，接受任意参数并返回任意类型的对象
                Dict[Union[str, Any], Union[Callable, Any]],  # 字典，键可以是字符串或任意类型，值可以是可调用对象或任意类型的对象
            ]
        ] = default_collate,  # 默认值为 default_collate，用于数据整理的默认函数
        collate_fn: Optional[Callable] = None,  # 参数 collate_fn，可选的可调用对象，用于数据整理
    ) -> None:
        # TODO(VitalyFedyunin): Replace `Callable[..., Any]` with `Callable[[IColumn], Any]`
        # TODO(VitalyFedyunin): Replace with `Dict[Union[str, IColumn], Union[Callable, Enum]]`
        
        # 如果 collate_fn 不为空
        if collate_fn is not None:
            # 调用父类的构造函数，传递 datapipe 和 collate_fn 作为参数
            super().__init__(datapipe, fn=collate_fn)
        else:
            # 如果 conversion 是可调用对象
            if callable(conversion):
                # 调用父类的构造函数，传递 datapipe 和 conversion 作为参数
                super().__init__(datapipe, fn=conversion)
            else:
                # 否则，conversion 应为字典时，使用 functools.partial 创建一个局部函数 collate_fn，
                # 该函数调用 _collate_helper，并传递 conversion 作为参数
                collate_fn = functools.partial(_collate_helper, conversion)
                # 调用父类的构造函数，传递 datapipe 和 collate_fn 作为参数
                super().__init__(datapipe, fn=collate_fn)
```