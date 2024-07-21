# `.\pytorch\torch\utils\data\datapipes\map\combining.py`

```
# mypy: allow-untyped-defs
# 引入必要的类型引用
from typing import Sized, Tuple, TypeVar

# 从torch.utils.data.datapipes._decorator中引入functional_datapipe装饰器
from torch.utils.data.datapipes._decorator import functional_datapipe
# 从torch.utils.data.datapipes.datapipe中引入MapDataPipe类
from torch.utils.data.datapipes.datapipe import MapDataPipe

# 所有公开的类和函数名
__all__ = ["ConcaterMapDataPipe", "ZipperMapDataPipe"]

# 定义一个协变类型变量T_co
T_co = TypeVar("T_co", covariant=True)

# 用functional_datapipe装饰器标记类为“concat”，表明它是一个功能性datapipe
@functional_datapipe("concat")
# 继承MapDataPipe类
class ConcaterMapDataPipe(MapDataPipe):
    r"""
    Concatenate multiple Map DataPipes (functional name: ``concat``).

    The new index of is the cumulative sum of source DataPipes.
    For example, if there are 2 source DataPipes both with length 5,
    index 0 to 4 of the resulting `ConcatMapDataPipe` would refer to
    elements of the first DataPipe, and 5 to 9 would refer to elements
    of the second DataPipe.

    Args:
        datapipes: Map DataPipes being concatenated

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp1 = SequenceWrapper(range(3))
        >>> dp2 = SequenceWrapper(range(3))
        >>> concat_dp = dp1.concat(dp2)
        >>> list(concat_dp)
        [0, 1, 2, 0, 1, 2]
    """

    # 定义一个元组，包含多个MapDataPipe类型的datapipes参数
    datapipes: Tuple[MapDataPipe]

    # 构造函数，接收多个MapDataPipe类型的datapipes参数
    def __init__(self, *datapipes: MapDataPipe):
        # 如果datapipes长度为0，抛出数值错误异常
        if len(datapipes) == 0:
            raise ValueError("Expected at least one DataPipe, but got nothing")
        # 如果不是所有的输入都是MapDataPipe类型，抛出类型错误异常
        if not all(isinstance(dp, MapDataPipe) for dp in datapipes):
            raise TypeError("Expected all inputs to be `MapDataPipe`")
        # 如果不是所有的输入都是可计数的，抛出类型错误异常
        if not all(isinstance(dp, Sized) for dp in datapipes):
            raise TypeError("Expected all inputs to be `Sized`")
        # 将传入的datapipes参数赋值给对象的datapipes属性
        self.datapipes = datapipes  # type: ignore[assignment]

    # 索引运算符重载，获取指定索引位置的元素
    def __getitem__(self, index) -> T_co:  # type: ignore[type-var]
        offset = 0
        # 遍历datapipes中的每一个DataPipe
        for dp in self.datapipes:
            # 如果索引减去偏移量小于当前DataPipe的长度
            if index - offset < len(dp):
                # 返回当前DataPipe中对应索引位置的元素
                return dp[index - offset]
            else:
                # 更新偏移量为当前DataPipe的长度
                offset += len(dp)
        # 如果索引超出范围，则抛出索引错误异常
        raise IndexError(f"Index {index} is out of range.")

    # 返回ConcatMapDataPipe对象的总长度
    def __len__(self) -> int:
        return sum(len(dp) for dp in self.datapipes)


# 用functional_datapipe装饰器标记类为“zip”，表明它是一个功能性datapipe
@functional_datapipe("zip")
# 继承MapDataPipe类，并指定元组类型为Tuple[T_co, ...]
class ZipperMapDataPipe(MapDataPipe[Tuple[T_co, ...]]):
    r"""
    Aggregates elements into a tuple from each of the input DataPipes (functional name: ``zip``).

    This MataPipe is out of bound as soon as the shortest input DataPipe is exhausted.

    Args:
        *datapipes: Map DataPipes being aggregated

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp1 = SequenceWrapper(range(3))
        >>> dp2 = SequenceWrapper(range(10, 13))
        >>> zip_dp = dp1.zip(dp2)
        >>> list(zip_dp)
        [(0, 10), (1, 11), (2, 12)]
    """

    # 定义一个元组，包含多个MapDataPipe[T_co]类型的datapipes参数
    datapipes: Tuple[MapDataPipe[T_co], ...]
    # 初始化方法，接受多个参数作为 MapDataPipe 实例
    def __init__(self, *datapipes: MapDataPipe[T_co]) -> None:
        # 如果没有传入任何参数，抛出值错误异常
        if len(datapipes) == 0:
            raise ValueError("Expected at least one DataPipe, but got nothing")
        # 检查所有传入参数是否都是 MapDataPipe 的实例，如果不是则抛出类型错误异常
        if not all(isinstance(dp, MapDataPipe) for dp in datapipes):
            raise TypeError("Expected all inputs to be `MapDataPipe`")
        # 检查所有传入参数是否都实现了 Sized 接口，如果没有则抛出类型错误异常
        if not all(isinstance(dp, Sized) for dp in datapipes):
            raise TypeError("Expected all inputs to be `Sized`")
        # 将传入的 datapipes 赋值给实例变量 self.datapipes
        self.datapipes = datapipes

    # 获取索引位置的元组数据的方法
    def __getitem__(self, index) -> Tuple[T_co, ...]:
        # 用于存放结果的空列表
        res = []
        # 遍历每一个 datapipes 实例
        for dp in self.datapipes:
            try:
                # 尝试从当前 datapipes 实例中获取指定索引位置的数据，并添加到结果列表中
                res.append(dp[index])
            except IndexError as e:
                # 如果索引超出范围，抛出新的 IndexError 异常，并指明具体的错误信息
                raise IndexError(
                    f"Index {index} is out of range for one of the input MapDataPipes {dp}."
                ) from e
        # 将结果列表转换为元组并返回
        return tuple(res)

    # 返回所有 datapipes 中最小的长度作为整体长度的方法
    def __len__(self) -> int:
        # 返回所有 datapipes 中长度的最小值
        return min(len(dp) for dp in self.datapipes)
```