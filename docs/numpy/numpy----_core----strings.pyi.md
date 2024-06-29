# `D:\src\scipysrc\numpy\numpy\_core\strings.pyi`

```py
# 引入类型提示所需的模块和类型定义
from typing import Any, overload

# 引入 NumPy 模块
import numpy as np
# 引入 NumPy 的部分类型定义
from numpy._typing import (
    NDArray,
    _ArrayLikeStr_co as U_co,
    _ArrayLikeBytes_co as S_co,
    _ArrayLikeInt_co as i_co,
    _ArrayLikeBool_co as b_co,
)

# equal 函数的重载定义，比较两个数组或字符串是否相等，返回布尔数组
@overload
def equal(x1: U_co, x2: U_co) -> NDArray[np.bool]: ...
@overload
def equal(x1: S_co, x2: S_co) -> NDArray[np.bool]: ...

# not_equal 函数的重载定义，比较两个数组或字符串是否不相等，返回布尔数组
@overload
def not_equal(x1: U_co, x2: U_co) -> NDArray[np.bool]: ...
@overload
def not_equal(x1: S_co, x2: S_co) -> NDArray[np.bool]: ...

# greater_equal 函数的重载定义，比较两个数组或字符串是否大于或等于，返回布尔数组
@overload
def greater_equal(x1: U_co, x2: U_co) -> NDArray[np.bool]: ...
@overload
def greater_equal(x1: S_co, x2: S_co) -> NDArray[np.bool]: ...

# less_equal 函数的重载定义，比较两个数组或字符串是否小于或等于，返回布尔数组
@overload
def less_equal(x1: U_co, x2: U_co) -> NDArray[np.bool]: ...
@overload
def less_equal(x1: S_co, x2: S_co) -> NDArray[np.bool]: ...

# greater 函数的重载定义，比较两个数组或字符串是否大于，返回布尔数组
@overload
def greater(x1: U_co, x2: U_co) -> NDArray[np.bool]: ...
@overload
def greater(x1: S_co, x2: S_co) -> NDArray[np.bool]: ...

# less 函数的重载定义，比较两个数组或字符串是否小于，返回布尔数组
@overload
def less(x1: U_co, x2: U_co) -> NDArray[np.bool]: ...
@overload
def less(x1: S_co, x2: S_co) -> NDArray[np.bool]: ...

# add 函数的重载定义，对两个数组或字符串进行加法操作，返回新的数组或字符串
@overload
def add(x1: U_co, x2: U_co) -> NDArray[np.str_]: ...
@overload
def add(x1: S_co, x2: S_co) -> NDArray[np.bytes_]: ...

# multiply 函数的重载定义，对数组或字符串进行乘法操作，返回新的数组或字符串
@overload
def multiply(a: U_co, i: i_co) -> NDArray[np.str_]: ...
@overload
def multiply(a: S_co, i: i_co) -> NDArray[np.bytes_]: ...

# mod 函数的重载定义，对数组或字符串进行取模操作，返回新的数组或字符串
@overload
def mod(a: U_co, value: Any) -> NDArray[np.str_]: ...
@overload
def mod(a: S_co, value: Any) -> NDArray[np.bytes_]: ...

# isalpha 函数的定义，判断数组或字符串中的元素是否都为字母，返回布尔数组
def isalpha(x: U_co | S_co) -> NDArray[np.bool]: ...

# isalnum 函数的定义，判断数组或字符串中的元素是否都为字母或数字，返回布尔数组
def isalnum(a: U_co | S_co) -> NDArray[np.bool]: ...

# isdigit 函数的定义，判断数组或字符串中的元素是否都为数字，返回布尔数组
def isdigit(x: U_co | S_co) -> NDArray[np.bool]: ...

# isspace 函数的定义，判断数组或字符串中的元素是否都为空格，返回布尔数组
def isspace(x: U_co | S_co) -> NDArray[np.bool]: ...

# isdecimal 函数的定义，判断数组或字符串中的元素是否都为十进制数字，返回布尔数组
def isdecimal(x: U_co) -> NDArray[np.bool]: ...

# isnumeric 函数的定义，判断数组或字符串中的元素是否都为数字，包括数字字符、Unicode数字，返回布尔数组
def isnumeric(x: U_co) -> NDArray[np.bool]: ...

# islower 函数的定义，判断数组或字符串中的元素是否都为小写字母，返回布尔数组
def islower(a: U_co | S_co) -> NDArray[np.bool]: ...

# istitle 函数的定义，判断数组或字符串中的元素是否符合标题格式，返回布尔数组
def istitle(a: U_co | S_co) -> NDArray[np.bool]: ...

# isupper 函数的定义，判断数组或字符串中的元素是否都为大写字母，返回布尔数组
def isupper(a: U_co | S_co) -> NDArray[np.bool]: ...

# str_len 函数的定义，返回数组或字符串中每个元素的长度，作为整数数组
def str_len(x: U_co | S_co) -> NDArray[np.int_]: ...

# find 函数的重载定义，查找子字符串在数组或字符串中的位置，返回整数数组
@overload
def find(
    a: U_co,
    sub: U_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.int_]: ...
@overload
def find(
    a: S_co,
    sub: S_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.int_]: ...

# rfind 函数的重载定义，从右侧开始查找子字符串在数组或字符串中的位置，返回整数数组
@overload
def rfind(
    a: U_co,
    sub: U_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.int_]: ...
@overload
def rfind(
    a: S_co,
    sub: S_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.int_]: ...

# index 函数的重载定义，查找子字符串在数组或字符串中的位置，返回整数数组
@overload
def index(
    a: U_co,
    sub: U_co,
    start: i_co = ...,
    end: None | i_co = ...,
) -> NDArray[np.int_]: ...
@overload
def index(
    a: S_co,
    sub: S_co,
    start: i_co = ...,
    end: None | i_co = ...,
) -> NDArray[np.int_]: ...

# rindex 函数的重载定义，从右侧开始查找子字符串在数组或字符串中的位置，返回整数数组
@overload
def rindex(
    a: U_co,
    sub: U_co,
    start: i_co = ...,
    end: None | i_co = ...,
) -> NDArray[np.int_]: ...
@overload
def rindex(
    a: S_co,
    sub: S_co,
    start: i_co = ...,
    end: None | i_co = ...,
) -> NDArray[np.int_]: ...

# count 函数的重载定义，统计子字符串在数组或字符串中出现的次数，返回整数数组
@overload
def count(
    a: U_co,
    # a 是变量名，表示一个类型为 U_co 的变量
    sub: U_co,
    # sub 是变量名，表示一个类型为 U_co 的变量
    start: i_co = ...,
    # start 是变量名，表示一个类型为 i_co 的变量，并且初始化为 ...
    end: i_co | None = ...,
    # end 是变量名，表示一个类型为 i_co 或者 None 的变量，并且初始化为 ...
# 函数签名，指定返回类型为 numpy 中的整数数组
) -> NDArray[np.int_]: ...

# 函数签名，计算字符串或字节串中子串出现的次数
@overload
def count(
    a: S_co,
    sub: S_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.int_]: ...

# 函数签名，检查字符串或字节串是否以指定前缀开头，返回布尔值数组
@overload
def startswith(
    a: U_co,
    prefix: U_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.bool]: ...

# 函数签名，检查字符串或字节串是否以指定后缀结尾，返回布尔值数组
@overload
def endswith(
    a: U_co,
    suffix: U_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.bool]: ...

# 函数签名，根据指定编码对字符串或字节串解码，返回字符串数组
def decode(
    a: S_co,
    encoding: None | str = ...,
    errors: None | str = ...,
) -> NDArray[np.str_]: ...

# 函数签名，根据指定编码对字符串或字节串编码，返回字节串数组
def encode(
    a: U_co,
    encoding: None | str = ...,
    errors: None | str = ...,
) -> NDArray[np.bytes_]: ...

# 函数签名，将字符串或字节串中的制表符扩展为空格，返回字符串数组
@overload
def expandtabs(a: U_co, tabsize: i_co = ...) -> NDArray[np.str_]: ...
@overload
def expandtabs(a: S_co, tabsize: i_co = ...) -> NDArray[np.bytes_]: ...

# 函数签名，将字符串或字节串居中对齐，返回字符串数组
@overload
def center(a: U_co, width: i_co, fillchar: U_co = ...) -> NDArray[np.str_]: ...
@overload
def center(a: S_co, width: i_co, fillchar: S_co = ...) -> NDArray[np.bytes_]: ...

# 函数签名，将字符串或字节串左对齐，返回字符串数组
@overload
def ljust(a: U_co, width: i_co, fillchar: U_co = ...) -> NDArray[np.str_]: ...
@overload
def ljust(a: S_co, width: i_co, fillchar: S_co = ...) -> NDArray[np.bytes_]: ...

# 函数签名，将字符串或字节串右对齐，返回字符串数组
@overload
def rjust(
    a: U_co,
    width: i_co,
    fillchar: U_co = ...,
) -> NDArray[np.str_]: ...
@overload
def rjust(
    a: S_co,
    width: i_co,
    fillchar: S_co = ...,
) -> NDArray[np.bytes_]: ...

# 函数签名，从字符串或字节串左侧移除空白字符，返回字符串数组
@overload
def lstrip(a: U_co, chars: None | U_co = ...) -> NDArray[np.str_]: ...
@overload
def lstrip(a: S_co, chars: None | S_co = ...) -> NDArray[np.bytes_]: ...

# 函数签名，从字符串或字节串右侧移除空白字符，返回字符串数组
@overload
def rstrip(a: U_co, char: None | U_co = ...) -> NDArray[np.str_]: ...
@overload
def rstrip(a: S_co, char: None | S_co = ...) -> NDArray[np.bytes_]: ...

# 函数签名，从字符串或字节串两侧移除空白字符，返回字符串数组
@overload
def strip(a: U_co, chars: None | U_co = ...) -> NDArray[np.str_]: ...
@overload
def strip(a: S_co, chars: None | S_co = ...) -> NDArray[np.bytes_]: ...

# 函数签名，将字符串或字节串填充到指定长度，左侧用 '0' 填充，返回字符串数组
@overload
def zfill(a: U_co, width: i_co) -> NDArray[np.str_]: ...
@overload
def zfill(a: S_co, width: i_co) -> NDArray[np.bytes_]: ...

# 函数签名，将字符串或字节串转换为大写，返回字符串数组
@overload
def upper(a: U_co) -> NDArray[np.str_]: ...
@overload
def upper(a: S_co) -> NDArray[np.bytes_]: ...

# 函数签名，将字符串或字节串转换为小写，返回字符串数组
@overload
def lower(a: U_co) -> NDArray[np.str_]: ...
@overload
def lower(a: S_co) -> NDArray[np.bytes_]: ...

# 函数签名，将字符串或字节串中的大小写互换，返回字符串数组
@overload
def swapcase(a: U_co) -> NDArray[np.str_]: ...
@overload
def swapcase(a: S_co) -> NDArray[np.bytes_]: ...

# 函数签名，将字符串或字节串首字母大写，返回字符串数组
@overload
def capitalize(a: U_co) -> NDArray[np.str_]: ...
@overload
def capitalize(a: S_co) -> NDArray[np.bytes_]: ...

# 函数签名，将字符串或字节串的每个单词首字母大写，返回字符串数组
@overload
def title(a: U_co) -> NDArray[np.str_]: ...
@overload
def title(a: S_co) -> NDArray[np.bytes_]: ...

# 函数签名，替换字符串或字节串中的旧子串为新子串，可指定替换次数，返回字符串数组
def replace(
    a: U_co,
    old: U_co,
    new: U_co,
    count: i_co = ...,
) -> NDArray[np.str_]: ...
# 定义一个函数签名，声明将返回一个字节字符串数组，接受参数 a 作为不可变字符串、old 作为不可变字符串、new 作为不可变字符串，
# count 作为可选的整数，默认为省略号
def replace(
    a: S_co,
    old: S_co,
    new: S_co,
    count: i_co = ...,
) -> NDArray[np.bytes_]: ...

# 定义了一个函数签名的重载，接受一个分隔符 U_co 和一个序列 U_co，返回一个字符串数组
@overload
def join(sep: U_co, seq: U_co) -> NDArray[np.str_]: ...
# 定义了一个函数签名的重载，接受一个分隔符 S_co 和一个序列 S_co，返回一个字节字符串数组
@overload
def join(sep: S_co, seq: S_co) -> NDArray[np.bytes_]: ...

# 定义了一个函数签名的重载，接受一个字符串或对象 a、可选的分隔符 None 或对象 U_co，默认为省略号的最大分割数和整数类型的最大分割数，
# 返回一个对象数组
@overload
def split(
    a: U_co,
    sep: None | U_co = ...,
    maxsplit: None | i_co = ...,
) -> NDArray[np.object_]: ...
# 定义了一个函数签名的重载，接受一个字节字符串或对象 a、可选的分隔符 None 或字节字符串对象 S_co，默认为省略号的最大分割数和整数类型的最大分割数，
# 返回一个对象数组
@overload
def split(
    a: S_co,
    sep: None | S_co = ...,
    maxsplit: None | i_co = ...,
) -> NDArray[np.object_]: ...

# 定义了一个函数签名的重载，接受一个字符串或对象 a、可选的分隔符 None 或对象 U_co，默认为省略号的最大分割数和整数类型的最大分割数，
# 从右向左分割字符串，返回一个对象数组
@overload
def rsplit(
    a: U_co,
    sep: None | U_co = ...,
    maxsplit: None | i_co = ...,
) -> NDArray[np.object_]: ...
# 定义了一个函数签名的重载，接受一个字节字符串或对象 a、可选的分隔符 None 或字节字符串对象 S_co，默认为省略号的最大分割数和整数类型的最大分割数，
# 从右向左分割字节字符串，返回一个对象数组
@overload
def rsplit(
    a: S_co,
    sep: None | S_co = ...,
    maxsplit: None | i_co = ...,
) -> NDArray[np.object_]: ...

# 定义了一个函数签名的重载，接受一个字符串或对象 a、可选的保留结束符 keepends，默认为 None 或布尔值对象，返回一个对象数组
@overload
def splitlines(a: U_co, keepends: None | b_co = ...) -> NDArray[np.object_]: ...
# 定义了一个函数签名的重载，接受一个字节字符串或对象 a、可选的保留结束符 keepends，默认为 None 或布尔值对象，返回一个对象数组
@overload
def splitlines(a: S_co, keepends: None | b_co = ...) -> NDArray[np.object_]: ...

# 定义了一个函数签名的重载，接受一个字符串或对象 a、分隔符 sep，返回一个字符串数组
@overload
def partition(a: U_co, sep: U_co) -> NDArray[np.str_]: ...
# 定义了一个函数签名的重载，接受一个字节字符串或对象 a、分隔符 sep，返回一个字节字符串数组
@overload
def partition(a: S_co, sep: S_co) -> NDArray[np.bytes_]: ...

# 定义了一个函数签名的重载，接受一个字符串或对象 a、分隔符 sep，从右向左分隔字符串，返回一个字符串数组
@overload
def rpartition(a: U_co, sep: U_co) -> NDArray[np.str_]: ...
# 定义了一个函数签名的重载，接受一个字节字符串或对象 a、分隔符 sep，从右向左分隔字节字符串，返回一个字节字符串数组
@overload
def rpartition(a: S_co, sep: S_co) -> NDArray[np.bytes_]: ...

# 定义了一个函数签名的重载，接受一个字符串或对象 a、表格 table、可选的删除字符 deletechars，默认为 None 或对象 U_co，返回一个字符串数组
@overload
def translate(
    a: U_co,
    table: U_co,
    deletechars: None | U_co = ...,
) -> NDArray[np.str_]: ...
# 定义了一个函数签名的重载，接受一个字节字符串或对象 a、表格 table、可选的删除字符 deletechars，默认为 None 或字节字符串对象 S_co，返回一个字节字符串数组
@overload
def translate(
    a: S_co,
    table: S_co,
    deletechars: None | S_co = ...,
) -> NDArray[np.bytes_]: ...
```