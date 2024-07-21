# `.\pytorch\torch\utils\data\datapipes\iter\selecting.py`

```py
# mypy: allow-untyped-defs
# 引入需要的类型和函数
from typing import Callable, Iterator, Tuple, TypeVar

# 从 Torch 数据模块中导入数据管道相关的装饰器和工具类
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import (
    _check_unpickable_fn,
    StreamWrapper,
    validate_input_col,
)

# 将本模块中的类和函数列入导出列表
__all__ = ["FilterIterDataPipe"]

# 定义类型变量
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)

# 使用装饰器标记为“filter”功能的数据管道类
@functional_datapipe("filter")
class FilterIterDataPipe(IterDataPipe[T_co]):
    r"""
    根据输入的 filter_fn 函数（功能名称为“filter”）从源数据管道中过滤元素。

    Args:
        datapipe: 要进行过滤的可迭代数据管道
        filter_fn: 将元素映射到布尔值的自定义函数
        input_col: 应用于 filter_fn 的数据的索引或索引，例如：

            - 默认为 None，直接将 filter_fn 应用于数据。
            - 对于列表/元组，使用整数索引。
            - 对于字典，使用键。

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> def is_even(n):
        ...     return n % 2 == 0
        >>> dp = IterableWrapper(range(5))
        >>> filter_dp = dp.filter(filter_fn=is_even)
        >>> list(filter_dp)
        [0, 2, 4]
    """

    # 类型注解，datapipe 是一个 IterDataPipe 类型的对象，T_co 是协变的类型变量
    datapipe: IterDataPipe[T_co]
    # filter_fn 是一个可调用对象（函数）
    filter_fn: Callable

    def __init__(
        self,
        datapipe: IterDataPipe[T_co],
        filter_fn: Callable,
        input_col=None,
    ) -> None:
        # 调用父类的构造函数
        super().__init__()
        # 初始化 datapipe 属性
        self.datapipe = datapipe

        # 检查 filter_fn 函数是否可序列化（即不可通过 pickle 序列化）
        _check_unpickable_fn(filter_fn)
        # 将 filter_fn 赋值给对象的 filter_fn 属性（类型标记为忽略赋值错误）
        self.filter_fn = filter_fn  # type: ignore[assignment]

        # 初始化 input_col 属性，并验证其合法性
        self.input_col = input_col
        validate_input_col(filter_fn, input_col)

    def _apply_filter_fn(self, data) -> bool:
        # 根据 input_col 属性应用 filter_fn 函数到数据
        if self.input_col is None:
            return self.filter_fn(data)
        elif isinstance(self.input_col, (list, tuple)):
            args = tuple(data[col] for col in self.input_col)
            return self.filter_fn(*args)
        else:
            return self.filter_fn(data[self.input_col])

    def __iter__(self) -> Iterator[T_co]:
        # 实现迭代器接口，遍历 datapipe 中的每个数据元素
        for data in self.datapipe:
            # 调用 _apply_filter_fn 方法检查是否符合过滤条件
            condition, filtered = self._apply_filter_fn(data)
            # 如果符合条件，则生成过滤后的数据
            if condition:
                yield filtered
            else:
                # 否则关闭数据中的流对象（例如文件流）
                StreamWrapper.close_streams(data)
    def _returnIfTrue(self, data: T) -> Tuple[bool, T]:
        # 调用对象的过滤函数，获取条件
        condition = self._apply_filter_fn(data)

        # 检查条件是否为DataFrame的列
        if df_wrapper.is_column(condition):
            # 如果是DataFrame的列，准备存储结果的列表
            result = []
            # 遍历DataFrame中符合条件的行
            for idx, mask in enumerate(df_wrapper.iterate(condition)):
                if mask:
                    # 如果行符合条件，将其加入结果列表
                    result.append(df_wrapper.get_item(data, idx))
            # 如果结果列表不为空，将所有符合条件的行连接起来返回
            if len(result):
                return True, df_wrapper.concat(result)
            else:
                return False, None  # type: ignore[return-value]

        # 如果条件不是布尔类型，则抛出数值错误异常
        if not isinstance(condition, bool):
            raise ValueError(
                "Boolean output is required for `filter_fn` of FilterIterDataPipe, got",
                type(condition),
            )

        # 返回条件及原始数据
        return condition, data
```