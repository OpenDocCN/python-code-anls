# `D:\src\scipysrc\pandas\pandas\core\strings\object_array.py`

```
from __future__ import annotations
# 导入将来版本的类型注解支持

import functools
# 导入 functools 模块，用于高阶函数操作
import re
# 导入 re 模块，用于正则表达式操作
import textwrap
# 导入 textwrap 模块，用于文本包装和填充
from typing import (
    TYPE_CHECKING,
    Literal,
    cast,
)
# 导入类型提示相关的功能：TYPE_CHECKING、Literal、cast

import unicodedata
# 导入 unicodedata 模块，用于 Unicode 数据库操作

import numpy as np
# 导入 NumPy 库，使用别名 np

from pandas._libs import lib
# 从 pandas._libs 导入 lib 模块

import pandas._libs.missing as libmissing
# 从 pandas._libs 导入 missing 子模块，使用别名 libmissing

import pandas._libs.ops as libops
# 从 pandas._libs 导入 ops 子模块，使用别名 libops

from pandas.core.dtypes.missing import isna
# 从 pandas.core.dtypes.missing 导入 isna 函数

from pandas.core.strings.base import BaseStringArrayMethods
# 从 pandas.core.strings.base 导入 BaseStringArrayMethods 类

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Sequence,
    )
    # 如果处于类型检查环境，导入 collections.abc 中的 Callable 和 Sequence

    from pandas._typing import (
        NpDtype,
        Scalar,
    )
    # 从 pandas._typing 导入 NpDtype 和 Scalar 类型

class ObjectStringArrayMixin(BaseStringArrayMethods):
    """
    String Methods operating on object-dtype ndarrays.
    """
    # 继承自 BaseStringArrayMethods，提供在 object-dtype ndarrays 上操作字符串的方法

    _str_na_value = np.nan
    # 设置类属性 _str_na_value 为 NaN，用于表示缺失值

    def __len__(self) -> int:
        # 返回对象的长度，用于类型提示，必须被子类实现
        raise NotImplementedError

    def _str_map(
        self, f, na_value=None, dtype: NpDtype | None = None, convert: bool = True
        # 定义 _str_map 方法，接受函数 f，缺失值 na_value，数据类型 dtype，转换标志 convert
    ):
        """
        Map a callable over valid elements of the array.

        Parameters
        ----------
        f : Callable
            A function to call on each non-NA element.
        na_value : Scalar, optional
            The value to set for NA values. Might also be used for the
            fill value if the callable `f` raises an exception.
            This defaults to ``self._str_na_value`` which is ``np.nan``
            for object-dtype and Categorical and ``pd.NA`` for StringArray.
        dtype : Dtype, optional
            The dtype of the result array.
        convert : bool, default True
            Whether to call `maybe_convert_objects` on the resulting ndarray
        """
        if dtype is None:
            # 如果未指定数据类型，则默认为对象类型
            dtype = np.dtype("object")
        if na_value is None:
            # 如果未指定缺失值的替代值，则使用对象的默认缺失值
            na_value = self._str_na_value

        if not len(self):
            # 如果数组长度为0，则返回一个空的 numpy 数组
            return np.array([], dtype=dtype)

        arr = np.asarray(self, dtype=object)
        # 创建一个布尔掩码，标识缺失值
        mask = isna(arr)
        # 是否需要转换结果数组类型
        map_convert = convert and not np.all(mask)
        try:
            # 尝试使用底层库函数进行映射和掩码推断
            result = lib.map_infer_mask(
                arr, f, mask.view(np.uint8), convert=map_convert
            )
        except (TypeError, AttributeError) as err:
            # 捕获可能的类型或属性错误异常
            # 如果函数 `f` 调用参数数量不正确，则重新引发异常
            p_err = (
                r"((takes)|(missing)) (?(2)from \d+ to )?\d+ "
                r"(?(3)required )positional arguments?"
            )

            if len(err.args) >= 1 and re.search(p_err, err.args[0]):
                # 如果匹配到参数数量错误的正则表达式，则重新引发异常
                raise err

            def g(x):
                # 在去除对象数据类型 .str 访问器后，可以移除此类回退行为
                try:
                    return f(x)
                except (TypeError, AttributeError):
                    return na_value

            return self._str_map(g, na_value=na_value, dtype=dtype)
        if not isinstance(result, np.ndarray):
            # 如果结果不是 numpy 数组，则直接返回结果
            return result
        if na_value is not np.nan:
            # 如果缺失值不是 np.nan，则用 na_value 替换结果中的缺失值
            np.putmask(result, mask, na_value)
            if convert and result.dtype == object:
                # 如果需要转换并且结果类型是对象类型，则尝试转换对象
                result = lib.maybe_convert_objects(result)
        return result

    def _str_count(self, pat, flags: int = 0):
        # 编译正则表达式
        regex = re.compile(pat, flags=flags)
        # 定义计数函数
        f = lambda x: len(regex.findall(x))
        # 返回调用 _str_map 方法后的结果，返回值类型为 int64
        return self._str_map(f, dtype="int64")

    def _str_pad(
        self,
        width: int,
        side: Literal["left", "right", "both"] = "left",
        fillchar: str = " ",
    ):
        # 根据指定的对齐方式创建匿名函数
        if side == "left":
            f = lambda x: x.rjust(width, fillchar)
        elif side == "right":
            f = lambda x: x.ljust(width, fillchar)
        elif side == "both":
            f = lambda x: x.center(width, fillchar)
        else:  # pragma: no cover
            # 如果指定的对齐方式无效，则抛出数值错误异常
            raise ValueError("Invalid side")
        # 调用对象内部的字符串映射方法，并使用创建的函数进行操作
        return self._str_map(f)

    def _str_contains(
        self, pat, case: bool = True, flags: int = 0, na=np.nan, regex: bool = True
    ):
        # 如果使用正则表达式，则根据大小写选项设置正则表达式的标志
        if regex:
            if not case:
                flags |= re.IGNORECASE

            # 编译正则表达式模式
            pat = re.compile(pat, flags=flags)

            # 创建匿名函数，检查字符串中是否存在匹配项
            f = lambda x: pat.search(x) is not None
        else:
            # 如果不使用正则表达式，则直接进行字符串包含检查
            if case:
                f = lambda x: pat in x
            else:
                # 将模式和字符串均转换为大写，再进行包含检查
                upper_pat = pat.upper()
                f = lambda x: upper_pat in x.upper()
        # 调用对象内部的字符串映射方法，并使用创建的函数进行操作
        return self._str_map(f, na, dtype=np.dtype("bool"))

    def _str_startswith(self, pat, na=None):
        # 创建匿名函数，检查字符串是否以指定前缀开头
        f = lambda x: x.startswith(pat)
        # 调用对象内部的字符串映射方法，并使用创建的函数进行操作
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def _str_endswith(self, pat, na=None):
        # 创建匿名函数，检查字符串是否以指定后缀结尾
        f = lambda x: x.endswith(pat)
        # 调用对象内部的字符串映射方法，并使用创建的函数进行操作
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def _str_replace(
        self,
        pat: str | re.Pattern,
        repl: str | Callable,
        n: int = -1,
        case: bool = True,
        flags: int = 0,
        regex: bool = True,
    ):
        # 如果不区分大小写，则添加忽略大小写的标志
        if case is False:
            flags |= re.IGNORECASE

        # 如果使用正则表达式、有标志位或替换函数，则处理模式
        if regex or flags or callable(repl):
            # 如果模式不是正则表达式对象，则根据参数情况编译或转义成正则表达式
            if not isinstance(pat, re.Pattern):
                if regex is False:
                    pat = re.escape(pat)
                pat = re.compile(pat, flags=flags)

            # 如果替换次数小于0，则置为0
            n = n if n >= 0 else 0
            # 创建匿名函数，进行字符串替换操作
            f = lambda x: pat.sub(repl=repl, string=x, count=n)
        else:
            # 创建匿名函数，进行普通字符串替换操作
            f = lambda x: x.replace(pat, repl, n)

        # 调用对象内部的字符串映射方法，并使用创建的函数进行操作
        return self._str_map(f, dtype=str)
    # 定义一个方法 `_str_repeat`，接受一个整数或整数序列作为参数 `repeats`
    def _str_repeat(self, repeats: int | Sequence[int]):
        # 如果 `repeats` 是整数，将其强制转换为整数类型 `rint`
        if lib.is_integer(repeats):
            rint = cast(int, repeats)

            # 定义一个函数 `scalar_rep`，根据参数 `x` 返回重复 `rint` 次的字节或字符串
            def scalar_rep(x):
                try:
                    return bytes.__mul__(x, rint)  # 尝试用字节乘法重复 `x` `rint` 次
                except TypeError:
                    return str.__mul__(x, rint)    # 如果失败，尝试用字符串乘法重复 `x` `rint` 次

            # 调用 `_str_map` 方法，对每个元素应用 `scalar_rep` 函数，并返回结果，数据类型为字符串
            return self._str_map(scalar_rep, dtype=str)
        else:
            # 如果 `repeats` 不是整数，则导入必要的模块
            from pandas.core.arrays.string_ import BaseStringArray

            # 定义一个函数 `rep`，根据参数 `x` 和 `r` 返回重复 `r` 次的字节或字符串
            def rep(x, r):
                if x is libmissing.NA:
                    return x  # 如果 `x` 是缺失值，直接返回 `x`
                try:
                    return bytes.__mul__(x, r)  # 尝试用字节乘法重复 `x` `r` 次
                except TypeError:
                    return str.__mul__(x, r)    # 如果失败，尝试用字符串乘法重复 `x` `r` 次

            # 使用 `libops.vec_binop` 方法，对数组和重复序列执行 `rep` 函数，返回结果
            result = libops.vec_binop(
                np.asarray(self),               # 将当前对象转换为 NumPy 数组
                np.asarray(repeats, dtype=object),  # 将重复序列转换为 NumPy 数组，数据类型为对象
                rep,                            # 使用 `rep` 函数进行操作
            )
            # 如果当前对象不是 `BaseStringArray` 类型，则直接返回结果
            if not isinstance(self, BaseStringArray):
                return result
            # 否则，通过 `_from_sequence` 方法创建相同类型的对象，使用 `result` 数组数据
            # 数据类型与当前对象相同
            return type(self)._from_sequence(result, dtype=self.dtype)

    # 定义一个方法 `_str_match`，用于在字符串数组中查找匹配正则表达式 `pat` 的元素
    def _str_match(
        self, pat: str, case: bool = True, flags: int = 0, na: Scalar | None = None
    ):
        # 如果 `case` 参数为 False，则设置 `flags` 添加 `re.IGNORECASE` 标志
        if not case:
            flags |= re.IGNORECASE

        # 使用给定的 `flags` 创建正则表达式对象 `regex`，用于匹配 `pat`
        regex = re.compile(pat, flags=flags)

        # 定义一个匿名函数 `f`，对每个元素应用正则表达式 `regex` 的 `match` 方法
        f = lambda x: regex.match(x) is not None
        # 调用 `_str_map` 方法，对每个元素应用函数 `f`，返回布尔类型的结果
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    # 定义一个方法 `_str_fullmatch`，用于在字符串数组中查找完全匹配正则表达式 `pat` 的元素
    def _str_fullmatch(
        self,
        pat: str | re.Pattern,
        case: bool = True,
        flags: int = 0,
        na: Scalar | None = None,
    ):
        # 如果 `case` 参数为 False，则设置 `flags` 添加 `re.IGNORECASE` 标志
        if not case:
            flags |= re.IGNORECASE

        # 使用给定的 `flags` 创建正则表达式对象 `regex`，用于完全匹配 `pat`
        regex = re.compile(pat, flags=flags)

        # 定义一个匿名函数 `f`，对每个元素应用正则表达式 `regex` 的 `fullmatch` 方法
        f = lambda x: regex.fullmatch(x) is not None
        # 调用 `_str_map` 方法，对每个元素应用函数 `f`，返回布尔类型的结果
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    # 定义一个方法 `_str_encode`，用于将字符串数组中的每个元素按指定编码和错误处理方式进行编码
    def _str_encode(self, encoding, errors: str = "strict"):
        # 定义一个匿名函数 `f`，对每个元素使用指定的编码和错误处理方式进行编码
        f = lambda x: x.encode(encoding, errors=errors)
        # 调用 `_str_map` 方法，对每个元素应用函数 `f`，返回结果，数据类型为对象
        return self._str_map(f, dtype=object)

    # 定义一个方法 `_str_find`，用于在字符串数组中查找子字符串 `sub` 的首次出现位置
    def _str_find(self, sub, start: int = 0, end=None):
        # 调用 `_str_find_` 方法，指定 `side` 为 "left"，返回查找结果
        return self._str_find_(sub, start, end, side="left")

    # 定义一个方法 `_str_rfind`，用于在字符串数组中查找子字符串 `sub` 的最后一次出现位置
    def _str_rfind(self, sub, start: int = 0, end=None):
        # 调用 `_str_find_` 方法，指定 `side` 为 "right"，返回查找结果
        return self._str_find_(sub, start, end, side="right")

    # 定义一个方法 `_str_find_`，用于在字符串数组中查找子字符串 `sub` 的位置
    # `side` 参数指定查找方向，"left" 表示从左到右，"right" 表示从右到左
    def _str_find_(self, sub, start, end, side):
        # 根据 `side` 参数确定使用的查找方法
        if side == "left":
            method = "find"
        elif side == "right":
            method = "rfind"
        else:  # 如果 `side` 不是 "left" 或 "right"，抛出异常
            raise ValueError("Invalid side")

        # 根据 `end` 参数选择相应的查找函数 `method`，并定义匿名函数 `f`
        if end is None:
            f = lambda x: getattr(x, method)(sub, start)  # 对每个元素调用对应的查找方法
        else:
            f = lambda x: getattr(x, method)(sub, start, end)  # 对每个元素调用对应的查找方法

        # 调用 `_str_map` 方法，对每个元素应用函数 `f`，返回结果，数据类型为 "int64"
        return self._str_map(f, dtype="int64")

    # 定义一个方法 `_str_findall`，用于在字符串数组中查找匹配正则表达式 `pat` 的所有子字符串
    def _str_findall(self, pat, flags: int = 0):
        # 使用给定的 `flags` 创建正则表达式对象 `regex`
        regex = re.compile(pat, flags=flags)

        # 调用 `_str_map` 方法，对每个元素应用正则表达式 `regex` 的 `findall` 方法
        # 返回结果，数据类型为 "object"
        return self._str_map(regex.findall, dtype="object")
    # 返回一个函数，该函数从输入的对象中获取指定索引的元素（字典或列表）
    def _str_get(self, i):
        def f(x):
            if isinstance(x, dict):
                return x.get(i)
            elif len(x) > i >= -len(x):
                return x[i]
            return self._str_na_value

        return self._str_map(f)

    # 返回一个函数，该函数从输入的字符串中查找指定子字符串的索引位置
    def _str_index(self, sub, start: int = 0, end=None):
        if end:
            f = lambda x: x.index(sub, start, end)
        else:
            f = lambda x: x.index(sub, start)
        return self._str_map(f, dtype="int64")

    # 返回一个函数，该函数从输入的字符串中从右向左查找指定子字符串的索引位置
    def _str_rindex(self, sub, start: int = 0, end=None):
        if end:
            f = lambda x: x.rindex(sub, start, end)
        else:
            f = lambda x: x.rindex(sub, start)
        return self._str_map(f, dtype="int64")

    # 返回一个函数，该函数将输入的字符串列表用指定分隔符连接起来
    def _str_join(self, sep: str):
        return self._str_map(sep.join)

    # 返回一个函数，该函数将输入的字符串根据指定分隔符分割，并返回分割结果的元组
    def _str_partition(self, sep: str, expand):
        result = self._str_map(lambda x: x.partition(sep), dtype="object")
        return result

    # 返回一个函数，该函数将输入的字符串从右向左根据指定分隔符分割，并返回分割结果的元组
    def _str_rpartition(self, sep: str, expand):
        return self._str_map(lambda x: x.rpartition(sep), dtype="object")

    # 返回一个函数，该函数计算输入字符串的长度
    def _str_len(self):
        return self._str_map(len, dtype="int64")

    # 返回一个函数，该函数对输入的字符串进行切片操作，返回切片后的子串
    def _str_slice(self, start=None, stop=None, step=None):
        obj = slice(start, stop, step)
        return self._str_map(lambda x: x[obj])

    # 返回一个函数，该函数对输入的字符串进行切片替换操作，返回替换后的字符串
    def _str_slice_replace(self, start=None, stop=None, repl=None):
        if repl is None:
            repl = ""

        def f(x):
            if x[start:stop] == "":
                local_stop = start
            else:
                local_stop = stop
            y = ""
            if start is not None:
                y += x[:start]
            y += repl
            if stop is not None:
                y += x[local_stop:]
            return y

        return self._str_map(f)

    # 返回一个函数，该函数根据指定的模式对输入的字符串进行分割操作，返回分割后的列表
    def _str_split(
        self,
        pat: str | re.Pattern | None = None,
        n=-1,
        expand: bool = False,
        regex: bool | None = None,
    ):
        if pat is None:
            if n is None or n == 0:
                n = -1
            f = lambda x: x.split(pat, n)
        else:
            new_pat: str | re.Pattern
            if regex is True or isinstance(pat, re.Pattern):
                new_pat = re.compile(pat)
            elif regex is False:
                new_pat = pat
            # regex is None so link to old behavior #43563
            else:
                if len(pat) == 1:
                    new_pat = pat
                else:
                    new_pat = re.compile(pat)

            if isinstance(new_pat, re.Pattern):
                if n is None or n == -1:
                    n = 0
                f = lambda x: new_pat.split(x, maxsplit=n)
            else:
                if n is None or n == 0:
                    n = -1
                f = lambda x: x.split(pat, n)
        return self._str_map(f, dtype=object)
    # 将字符串按指定模式和次数从右向左拆分
    def _str_rsplit(self, pat=None, n=-1):
        # 如果未指定拆分次数或次数为0，则设为-1，表示完全拆分
        if n is None or n == 0:
            n = -1
        # 定义匿名函数f，使用rsplit方法拆分字符串
        f = lambda x: x.rsplit(pat, n)
        # 调用_str_map方法，将函数f应用到每个元素上，返回处理后的结果
        return self._str_map(f, dtype="object")

    # 使用给定的转换表table对字符串进行转换
    def _str_translate(self, table):
        # 使用lambda表达式将translate方法应用到每个元素上，并返回结果
        return self._str_map(lambda x: x.translate(table))

    # 将字符串按指定宽度进行换行处理，并返回处理后的结果
    def _str_wrap(self, width: int, **kwargs):
        # 初始化TextWrapper对象tw，设置宽度和其他参数
        tw = textwrap.TextWrapper(**kwargs, width=width)
        # 使用lambda表达式将wrap方法应用到每个元素上，将处理后的结果以换行符连接起来返回
        return self._str_map(lambda s: "\n".join(tw.wrap(s)))

    # 将字符串按指定分隔符sep进行分割，并生成相应的虚拟变量（哑变量）
    def _str_get_dummies(self, sep: str = "|"):
        from pandas import Series

        # 将对象转换为Series，并用空值填充缺失值
        arr = Series(self).fillna("")
        try:
            # 尝试将sep与arr中的每个元素连接
            arr = sep + arr + sep
        except (TypeError, NotImplementedError):
            # 如果连接失败，则将arr转换为字符串类型后再试一次
            arr = sep + arr.astype(str) + sep

        # 初始化空集合tags，用于存储分割后的标签
        tags: set[str] = set()
        # 使用Series对象的str属性将arr按sep分割，并更新tags集合
        for ts in Series(arr, copy=False).str.split(sep):
            tags.update(ts)
        # 对tags集合进行排序，并去除空字符串
        tags2 = sorted(tags - {""})

        # 初始化一个空的二维数组dummies，用于存储哑变量结果
        dummies = np.empty((len(arr), len(tags2)), dtype=np.int64)

        # 定义内部函数_isin，用于检查元素是否存在于测试元素中
        def _isin(test_elements: str, element: str) -> bool:
            return element in test_elements

        # 遍历tags2列表，并根据每个标签生成对应的哑变量列
        for i, t in enumerate(tags2):
            pat = sep + t + sep
            # 使用map_infer函数将_isin函数应用到arr的元素上，将结果存储在dummies的第i列中
            dummies[:, i] = lib.map_infer(
                arr.to_numpy(), functools.partial(_isin, element=pat)
            )
        # 返回生成的哑变量数组和标签列表
        return dummies, tags2

    # 将字符串转换为大写形式，并返回处理后的结果
    def _str_upper(self):
        return self._str_map(lambda x: x.upper())

    # 检查字符串是否由字母和数字组成，并返回布尔值结果
    def _str_isalnum(self):
        return self._str_map(str.isalnum, dtype="bool")

    # 检查字符串是否全为字母，并返回布尔值结果
    def _str_isalpha(self):
        return self._str_map(str.isalpha, dtype="bool")

    # 检查字符串是否全为十进制数字，并返回布尔值结果
    def _str_isdecimal(self):
        return self._str_map(str.isdecimal, dtype="bool")

    # 检查字符串是否全为数字，并返回布尔值结果
    def _str_isdigit(self):
        return self._str_map(str.isdigit, dtype="bool")

    # 检查字符串是否全为小写，并返回布尔值结果
    def _str_islower(self):
        return self._str_map(str.islower, dtype="bool")

    # 检查字符串是否全为数字字符，并返回布尔值结果
    def _str_isnumeric(self):
        return self._str_map(str.isnumeric, dtype="bool")

    # 检查字符串是否全为空白字符，并返回布尔值结果
    def _str_isspace(self):
        return self._str_map(str.isspace, dtype="bool")

    # 检查字符串是否符合标题格式，并返回布尔值结果
    def _str_istitle(self):
        return self._str_map(str.istitle, dtype="bool")

    # 检查字符串是否全为大写，并返回布尔值结果
    def _str_isupper(self):
        return self._str_map(str.isupper, dtype="bool")

    # 将字符串首字母大写，其他字符小写，并返回处理后的结果
    def _str_capitalize(self):
        return self._str_map(str.capitalize)

    # 将字符串全部转换为小写，并返回处理后的结果
    def _str_casefold(self):
        return self._str_map(str.casefold)

    # 将字符串转换为标题格式，并返回处理后的结果
    def _str_title(self):
        return self._str_map(str.title)

    # 将字符串中的大小写互换，并返回处理后的结果
    def _str_swapcase(self):
        return self._str_map(str.swapcase)

    # 将字符串转换为小写，并返回处理后的结果
    def _str_lower(self):
        return self._str_map(str.lower)

    # 根据给定的Unicode规范形式对字符串进行规范化处理，并返回处理后的结果
    def _str_normalize(self, form):
        # 定义匿名函数f，使用unicodedata.normalize函数进行规范化处理
        f = lambda x: unicodedata.normalize(form, x)
        # 调用_str_map方法，将函数f应用到每个元素上，返回处理后的结果
        return self._str_map(f)

    # 去除字符串两端指定的字符（默认为空白字符），并返回处理后的结果
    def _str_strip(self, to_strip=None):
        # 使用lambda表达式将strip方法应用到每个元素上，并返回处理后的结果
        return self._str_map(lambda x: x.strip(to_strip))

    # 去除字符串左端指定的字符（默认为空白字符），并返回处理后的结果
    def _str_lstrip(self, to_strip=None):
        # 使用lambda表达式将lstrip方法应用到每个元素上，并返回处理后的结果
        return self._str_map(lambda x: x.lstrip(to_strip))

    # 去除字符串右端指定的字符（默认为空白字符），并返回处理后的结果
    def _str_rstrip(self, to_strip=None):
        # 使用lambda表达式将rstrip方法应用到每个元素上，并返回处理后的结果
        return self._str_map(lambda x: x.rstrip(to_strip))
    # 定义一个方法用于移除字符串列表中各元素的前缀
    def _str_removeprefix(self, prefix: str):
        # 调用 self._str_map 方法，传入 lambda 函数作为参数，lambda 函数移除各元素的前缀
        return self._str_map(lambda x: x.removeprefix(prefix))

    # 定义一个方法用于移除字符串列表中各元素的后缀
    def _str_removesuffix(self, suffix: str):
        # 调用 self._str_map 方法，传入 lambda 函数作为参数，lambda 函数移除各元素的后缀
        return self._str_map(lambda x: x.removesuffix(suffix))

    # 定义一个方法用于从字符串列表中提取匹配正则表达式的部分
    def _str_extract(self, pat: str, flags: int = 0, expand: bool = True):
        # 编译给定的正则表达式模式，生成 regex 对象
        regex = re.compile(pat, flags=flags)
        # 获取缺失值标记
        na_value = self._str_na_value

        # 如果不需要展开结果
        if not expand:
            # 定义函数 g，用于在每个字符串上执行正则匹配，并返回第一个捕获组的值或缺失值标记
            def g(x):
                m = regex.search(x)
                return m.groups()[0] if m else na_value

            # 调用 self._str_map 方法，传入函数 g，对每个元素执行 g 函数，并关闭类型转换
            return self._str_map(g, convert=False)

        # 如果需要展开结果
        empty_row = [na_value] * regex.groups

        # 定义函数 f，用于在每个字符串上执行正则匹配，并返回所有捕获组的值或缺失值标记构成的列表
        def f(x):
            # 如果 x 不是字符串，则返回一个由缺失值标记构成的空行列表
            if not isinstance(x, str):
                return empty_row
            m = regex.search(x)
            if m:
                # 如果找到匹配，则返回捕获组的值，否则返回空行列表
                return [na_value if item is None else item for item in m.groups()]
            else:
                return empty_row

        # 对 self 对象的每个元素调用 f 函数，并返回结果列表
        return [f(val) for val in np.asarray(self)]
```