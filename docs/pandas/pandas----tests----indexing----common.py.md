# `D:\src\scipysrc\pandas\pandas\tests\indexing\common.py`

```
# 导入必要的模块和类型提示
"""common utilities"""

from __future__ import annotations  # 导入未来的类型注解支持

from typing import (
    Any,        # 引入Any类型，表示任意类型
    Literal,    # 引入Literal类型，用于指定字面值类型
)


def _mklbl(prefix: str, n: int):
    # 创建以指定前缀和数量为基础的标签列表
    return [f"{prefix}{i}" for i in range(n)]


def check_indexing_smoketest_or_raises(
    obj,                # 要检查的对象
    method: Literal["iloc", "loc"],  # 使用的索引方法，只能是'iloc'或'loc'
    key: Any,           # 索引键，可以是任意类型
    axes: Literal[0, 1] | None = None,  # 指定的轴，可以是0、1或None
    fails=None,         # 可能抛出的异常类型，如果是这些异常类型，则不抛出异常
) -> None:
    if axes is None:
        axes_list = [0, 1]  # 如果未指定轴，则默认为0和1
    else:
        assert axes in [0, 1]  # 确保指定的轴在0或1中
        axes_list = [axes]

    for ax in axes_list:
        if ax < obj.ndim:  # 如果指定的轴小于对象的维度数
            # 创建一个元组访问器
            new_axes = [slice(None)] * obj.ndim  # 创建全切片的列表
            new_axes[ax] = key  # 将指定轴的切片替换为指定的键
            axified = tuple(new_axes)  # 转换为元组
            try:
                getattr(obj, method).__getitem__(axified)  # 尝试获取对象的指定方法的元素
            except (IndexError, TypeError, KeyError) as detail:
                # 如果出现索引错误、类型错误或键错误
                # 如果在fails列表中，则不抛出异常，直接返回
                if fails is not None:
                    if isinstance(detail, fails):
                        return
                raise  # 否则抛出异常
```