# `D:\src\scipysrc\pandas\pandas\core\arrays\_arrow_string_mixins.py`

```
from __future__ import annotations  # 允许在类型提示中使用字符串形式的类型名称

from typing import (  # 引入类型提示模块
    TYPE_CHECKING,  # 用于在类型检查时避免循环导入
    Literal,  # 用于指定字符串字面值类型
)

import numpy as np  # 引入 NumPy 库

from pandas.compat import pa_version_under10p1  # 从 pandas 兼容模块导入版本检查函数

if not pa_version_under10p1:  # 如果 pandas 版本在 10.1 以下
    import pyarrow as pa  # 导入 pyarrow 库
    import pyarrow.compute as pc  # 导入 pyarrow 的计算模块

if TYPE_CHECKING:  # 如果是类型检查阶段
    from pandas._typing import Self  # 导入 Self 类型提示

class ArrowStringArrayMixin:  # 定义 ArrowStringArrayMixin 类
    _pa_array = None  # 初始化类成员变量 _pa_array

    def __init__(self, *args, **kwargs) -> None:  # 构造函数，但抛出未实现错误
        raise NotImplementedError

    def _str_pad(  # 定义字符串填充函数
        self,
        width: int,  # 填充后的宽度
        side: Literal["left", "right", "both"] = "left",  # 填充的位置
        fillchar: str = " ",  # 填充字符
    ) -> Self:  # 返回类型为 Self
        if side == "left":  # 如果是左填充
            pa_pad = pc.utf8_lpad  # 使用 utf8_lpad 函数
        elif side == "right":  # 如果是右填充
            pa_pad = pc.utf8_rpad  # 使用 utf8_rpad 函数
        elif side == "both":  # 如果是两侧填充
            pa_pad = pc.utf8_center  # 使用 utf8_center 函数
        else:  # 否则抛出值错误
            raise ValueError(
                f"Invalid side: {side}. Side must be one of 'left', 'right', 'both'"
            )
        return type(self)(pa_pad(self._pa_array, width=width, padding=fillchar))  # 返回填充后的结果对象

    def _str_get(self, i: int) -> Self:  # 定义字符串索引函数
        lengths = pc.utf8_length(self._pa_array)  # 计算字符串数组的长度
        if i >= 0:  # 如果索引为正数
            out_of_bounds = pc.greater_equal(i, lengths)  # 检查是否超出范围
            start = i  # 开始索引位置
            stop = i + 1  # 结束索引位置
            step = 1  # 步长
        else:  # 如果索引为负数
            out_of_bounds = pc.greater(-i, lengths)  # 检查是否超出范围
            start = i  # 开始索引位置
            stop = i - 1  # 结束索引位置
            step = -1  # 负数索引的步长
        not_out_of_bounds = pc.invert(out_of_bounds.fill_null(True))  # 对超出范围的值取反
        selected = pc.utf8_slice_codeunits(  # 切片选取 UTF-8 编码单元
            self._pa_array, start=start, stop=stop, step=step
        )
        null_value = pa.scalar(  # 创建空标量对象
            None,
            type=self._pa_array.type,  # 指定对象类型
        )
        result = pc.if_else(not_out_of_bounds, selected, null_value)  # 根据条件选择切片结果或空值
        return type(self)(result)  # 返回切片后的结果对象

    def _str_slice_replace(  # 定义字符串切片替换函数
        self, start: int | None = None, stop: int | None = None, repl: str | None = None
    ) -> Self:  # 返回类型为 Self
        if repl is None:  # 如果替换字符串为空
            repl = ""  # 使用空字符串进行替换
        if start is None:  # 如果起始位置为空
            start = 0  # 默认从开头开始替换
        if stop is None:  # 如果结束位置为空
            stop = np.iinfo(np.int64).max  # 设置为 int64 的最大值
        return type(self)(pc.utf8_replace_slice(self._pa_array, start, stop, repl))  # 返回替换后的结果对象

    def _str_capitalize(self) -> Self:  # 定义字符串首字母大写函数
        return type(self)(pc.utf8_capitalize(self._pa_array))  # 返回首字母大写后的结果对象

    def _str_title(self) -> Self:  # 定义字符串标题化函数
        return type(self)(pc.utf8_title(self._pa_array))  # 返回标题化后的结果对象

    def _str_swapcase(self) -> Self:  # 定义字符串大小写互换函数
        return type(self)(pc.utf8_swapcase(self._pa_array))  # 返回大小写互换后的结果对象

    def _str_removesuffix(self, suffix: str):  # 定义删除后缀函数
        ends_with = pc.ends_with(self._pa_array, pattern=suffix)  # 检查是否以指定后缀结尾
        removed = pc.utf8_slice_codeunits(self._pa_array, 0, stop=-len(suffix))  # 删除指定后缀
        result = pc.if_else(ends_with, removed, self._pa_array)  # 根据条件选择删除后缀或保持原样
        return type(self)(result)  # 返回删除后缀后的结果对象
```