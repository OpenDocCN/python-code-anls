# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\char.pyi`

```py
import sys
from typing import Any

import numpy as np
import numpy.typing as npt

# 如果 Python 版本大于等于 3.11，使用标准库 typing 中的 assert_type 函数
if sys.version_info >= (3, 11):
    from typing import assert_type
# 否则，使用 typing_extensions 中的 assert_type 函数
else:
    from typing_extensions import assert_type

# 定义一个字符串类型的 NumPy 数组
AR_U: npt.NDArray[np.str_]
# 定义一个字节字符串类型的 NumPy 数组
AR_S: npt.NDArray[np.bytes_]

# 使用 assert_type 函数确保 np.char.equal 返回的结果是一个布尔型 NumPy 数组
assert_type(np.char.equal(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.char.equal(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.char.not_equal(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.char.not_equal(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.char.greater_equal(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.char.greater_equal(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.char.less_equal(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.char.less_equal(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.char.greater(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.char.greater(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.char.less(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.char.less(AR_S, AR_S), npt.NDArray[np.bool])

# 对字符串数组进行乘法操作，确保返回的结果是字符串类型的 NumPy 数组
assert_type(np.char.multiply(AR_U, 5), npt.NDArray[np.str_])
assert_type(np.char.multiply(AR_S, [5, 4, 3]), npt.NDArray[np.bytes_])

# 使用字符串格式化操作，确保返回的结果是字符串类型的 NumPy 数组
assert_type(np.char.mod(AR_U, "test"), npt.NDArray[np.str_])
assert_type(np.char.mod(AR_S, "test"), npt.NDArray[np.bytes_])

# 将字符串数组的每个元素首字母大写，确保返回的结果是字符串类型的 NumPy 数组
assert_type(np.char.capitalize(AR_U), npt.NDArray[np.str_])
assert_type(np.char.capitalize(AR_S), npt.NDArray[np.bytes_])

# 将字符串数组的每个元素居中对齐，确保返回的结果是字符串类型的 NumPy 数组
assert_type(np.char.center(AR_U, 5), npt.NDArray[np.str_])
assert_type(np.char.center(AR_S, [2, 3, 4], b"a"), npt.NDArray[np.bytes_])

# 将字符串数组编码为字节字符串类型的 NumPy 数组
assert_type(np.char.encode(AR_U), npt.NDArray[np.bytes_])
# 将字节字符串数组解码为字符串类型的 NumPy 数组
assert_type(np.char.decode(AR_S), npt.NDArray[np.str_])

# 将字符串数组中的制表符扩展为空格，确保返回的结果是字符串类型的 NumPy 数组
assert_type(np.char.expandtabs(AR_U), npt.NDArray[np.str_])
assert_type(np.char.expandtabs(AR_S, tabsize=4), npt.NDArray[np.bytes_])

# 使用指定的分隔符连接字符串数组的每个元素，确保返回的结果是字符串类型的 NumPy 数组
assert_type(np.char.join(AR_U, "_"), npt.NDArray[np.str_])
assert_type(np.char.join(AR_S, [b"_", b""]), npt.NDArray[np.bytes_])

# 将字符串数组的每个元素左对齐或右对齐到指定的宽度，确保返回的结果是字符串类型的 NumPy 数组
assert_type(np.char.ljust(AR_U, 5), npt.NDArray[np.str_])
assert_type(np.char.ljust(AR_S, [4, 3, 1], fillchar=[b"a", b"b", b"c"]), npt.NDArray[np.bytes_])
assert_type(np.char.rjust(AR_U, 5), npt.NDArray[np.str_])
assert_type(np.char.rjust(AR_S, [4, 3, 1], fillchar=[b"a", b"b", b"c"]), npt.NDArray[np.bytes_])

# 移除字符串数组每个元素的开头或结尾的空白字符或指定字符，确保返回的结果是字符串类型的 NumPy 数组
assert_type(np.char.lstrip(AR_U), npt.NDArray[np.str_])
assert_type(np.char.lstrip(AR_S, chars=b"_"), npt.NDArray[np.bytes_])
assert_type(np.char.rstrip(AR_U), npt.NDArray[np.str_])
assert_type(np.char.rstrip(AR_S, chars=b"_"), npt.NDArray[np.bytes_])
assert_type(np.char.strip(AR_U), npt.NDArray[np.str_])
assert_type(np.char.strip(AR_S, chars=b"_"), npt.NDArray[np.bytes_])

# 使用指定的分隔符分割字符串数组的每个元素，确保返回的结果是字符串类型的 NumPy 数组
assert_type(np.char.partition(AR_U, "\n"), npt.NDArray[np.str_])
assert_type(np.char.partition(AR_S, [b"a", b"b", b"c"]), npt.NDArray[np.bytes_])
assert_type(np.char.rpartition(AR_U, "\n"), npt.NDArray[np.str_])
assert_type(np.char.rpartition(AR_S, [b"a", b"b", b"c"]), npt.NDArray[np.bytes_])

# 使用指定的字符串替换字符串数组中的指定子字符串，确保返回的结果是字符串类型的 NumPy 数组
assert_type(np.char.replace(AR_U, "_", "-"), npt.NDArray[np.str_])
# 使用 np.char.replace 函数替换 AR_S 数组中的特定字节序列，将 b"_" 替换为 b"a"，b"" 替换为 b"b"
assert_type(np.char.replace(AR_S, [b"_", b""], [b"a", b"b"]), npt.NDArray[np.bytes_])

# 使用 np.char.split 函数将 AR_U 数组按照 "_" 分割，并返回分割后的对象数组
assert_type(np.char.split(AR_U, "_"), npt.NDArray[np.object_])

# 使用 np.char.split 函数将 AR_S 数组按照 maxsplit 指定的分割次数进行分割，并返回对象数组
assert_type(np.char.split(AR_S, maxsplit=[1, 2, 3]), npt.NDArray[np.object_])

# 使用 np.char.rsplit 函数将 AR_U 数组按照 "_" 分割，并返回从右侧开始的对象数组
assert_type(np.char.rsplit(AR_U, "_"), npt.NDArray[np.object_])

# 使用 np.char.rsplit 函数将 AR_S 数组按照 maxsplit 指定的分割次数从右侧开始分割，并返回对象数组
assert_type(np.char.rsplit(AR_S, maxsplit=[1, 2, 3]), npt.NDArray[np.object_])

# 使用 np.char.splitlines 函数按照行分割 AR_U 数组，并返回对象数组
assert_type(np.char.splitlines(AR_U), npt.NDArray[np.object_])

# 使用 np.char.splitlines 函数按照行分割 AR_S 数组，根据 keepends 参数保留行尾符号，并返回对象数组
assert_type(np.char.splitlines(AR_S, keepends=[True, True, False]), npt.NDArray[np.object_])

# 使用 np.char.swapcase 函数将 AR_U 数组中的字母大小写互换，并返回字符串数组
assert_type(np.char.swapcase(AR_U), npt.NDArray[np.str_])

# 使用 np.char.swapcase 函数将 AR_S 数组中的字母大小写互换，并返回字节字符串数组
assert_type(np.char.swapcase(AR_S), npt.NDArray[np.bytes_])

# 使用 np.char.title 函数将 AR_U 数组中的每个单词的首字母大写，并返回字符串数组
assert_type(np.char.title(AR_U), npt.NDArray[np.str_])

# 使用 np.char.title 函数将 AR_S 数组中的每个单词的首字母大写，并返回字节字符串数组
assert_type(np.char.title(AR_S), npt.NDArray[np.bytes_])

# 使用 np.char.upper 函数将 AR_U 数组中的字母转换为大写，并返回字符串数组
assert_type(np.char.upper(AR_U), npt.NDArray[np.str_])

# 使用 np.char.upper 函数将 AR_S 数组中的字母转换为大写，并返回字节字符串数组
assert_type(np.char.upper(AR_S), npt.NDArray[np.bytes_])

# 使用 np.char.zfill 函数将 AR_U 数组中的字符串填充为指定长度，并返回字符串数组
assert_type(np.char.zfill(AR_U, 5), npt.NDArray[np.str_])

# 使用 np.char.zfill 函数将 AR_S 数组中的字节字符串填充为指定长度，并返回字节字符串数组
assert_type(np.char.zfill(AR_S, [2, 3, 4]), npt.NDArray[np.bytes_])

# 使用 np.char.count 函数计算 AR_U 数组中字符 "a" 出现的次数，并返回整数数组
assert_type(np.char.count(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_])

# 使用 np.char.count 函数计算 AR_S 数组中特定字节序列出现的次数，并返回整数数组
assert_type(np.char.count(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])

# 使用 np.char.endswith 函数检查 AR_U 数组中是否以指定字符结尾，并返回布尔数组
assert_type(np.char.endswith(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.bool])

# 使用 np.char.endswith 函数检查 AR_S 数组中是否以指定字节序列结尾，并返回布尔数组
assert_type(np.char.endswith(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.bool])

# 使用 np.char.startswith 函数检查 AR_U 数组中是否以指定字符开头，并返回布尔数组
assert_type(np.char.startswith(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.bool])

# 使用 np.char.startswith 函数检查 AR_S 数组中是否以指定字节序列开头，并返回布尔数组
assert_type(np.char.startswith(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.bool])

# 使用 np.char.find 函数在 AR_U 数组中查找指定字符并返回其位置，未找到返回 -1，并返回整数数组
assert_type(np.char.find(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_])

# 使用 np.char.find 函数在 AR_S 数组中查找指定字节序列并返回其位置，未找到返回 -1，并返回整数数组
assert_type(np.char.find(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])

# 使用 np.char.rfind 函数在 AR_U 数组中从右侧开始查找指定字符并返回其位置，未找到返回 -1，并返回整数数组
assert_type(np.char.rfind(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_])

# 使用 np.char.rfind 函数在 AR_S 数组中从右侧开始查找指定字节序列并返回其位置，未找到返回 -1，并返回整数数组
assert_type(np.char.rfind(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])

# 使用 np.char.index 函数在 AR_U 数组中查找指定字符并返回其位置，未找到会引发 ValueError，并返回整数数组
assert_type(np.char.index(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_])

# 使用 np.char.index 函数在 AR_S 数组中查找指定字节序列并返回其位置，未找到会引发 ValueError，并返回整数数组
assert_type(np.char.index(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])

# 使用 np.char.rindex 函数在 AR_U 数组中从右侧开始查找指定字符并返回其位置，未找到会引发 ValueError，并返回整数数组
assert_type(np.char.rindex(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_])

# 使用 np.char.rindex 函数在 AR_S 数组中从右侧开始查找指定字节序列并返回其位置，未找到会引发 ValueError，并返回整数数组
assert_type(np.char.rindex(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])

# 使用 np.char.isalpha 函数检查 AR_U 数组中的每个字符串是否都是字母，并返回布尔数组
assert_type(np.char.isalpha(AR_U), npt.NDArray[np.bool])

# 使用 np.char.isalpha 函数检查 AR_S 数组中的每个字节序列是否都是字母，并返回布尔数组
assert_type(np.char.isalpha(AR_S), npt.NDArray[np.bool])

# 使用 np.char.isalnum 函数检查 AR_U 数组中的每个字符串是否都是字母或数字，并返回布尔数组
assert_type(np.char.isalnum(AR_U), npt.NDArray[np.bool])

# 使用 np.char.isalnum 函数检查 AR_S 数组中的每个字节序列是否都是字母或数字，并返回布尔数组
assert_type(np.char.isalnum(AR_S), npt.NDArray[np.bool])

# 使用 np.char.isdecimal 函数检查 AR_U 数组中的每个字符串是否都是十进制数字，并返回布尔数组
assert_type(np.char.isdecimal(AR_U), npt.NDArray[np.bool])

# 使用 np.char.isdigit 函数检查 AR_U 数组中的每个字符串是否都
# 确保 AR_U 中的每个字符都是大写，并返回一个布尔类型的 NumPy 数组
assert_type(np.char.isupper(AR_U), npt.NDArray[np.bool])
# 确保 AR_S 中的每个字符都是大写，并返回一个布尔类型的 NumPy 数组
assert_type(np.char.isupper(AR_S), npt.NDArray[np.bool])

# 计算 AR_U 中每个字符串的长度，并返回一个整数类型的 NumPy 数组
assert_type(np.char.str_len(AR_U), npt.NDArray[np.int_])
# 计算 AR_S 中每个字符串的长度，并返回一个整数类型的 NumPy 数组
assert_type(np.char.str_len(AR_S), npt.NDArray[np.int_])

# 将 AR_U 转换为字符类型的 NumPy 数组
assert_type(np.char.array(AR_U), np.char.chararray[Any, np.dtype[np.str_]])
# 将 AR_S 转换为字节类型的 NumPy 数组，按照内存布局 "K" 排列
assert_type(np.char.array(AR_S, order="K"), np.char.chararray[Any, np.dtype[np.bytes_]])
# 将字符串 "bob" 转换为字符类型的 NumPy 数组，复制原始数据
assert_type(np.char.array("bob", copy=True), np.char.chararray[Any, np.dtype[np.str_]])
# 将字节串 b"bob" 转换为字符类型的 NumPy 数组，每个元素占据 5 字节
assert_type(np.char.array(b"bob", itemsize=5), np.char.chararray[Any, np.dtype[np.bytes_]])
# 将整数 1 转换为字节类型的 NumPy 数组，unicode=False 表示不启用 Unicode 编码
assert_type(np.char.array(1, unicode=False), np.char.chararray[Any, np.dtype[np.bytes_]])
# 将整数 1 转换为字符类型的 NumPy 数组，unicode=True 表示启用 Unicode 编码
assert_type(np.char.array(1, unicode=True), np.char.chararray[Any, np.dtype[np.str_]])

# 将 AR_U 转换为字符类型的 NumPy 数组，并返回一个新的数组对象
assert_type(np.char.asarray(AR_U), np.char.chararray[Any, np.dtype[np.str_]])
# 将 AR_S 转换为字节类型的 NumPy 数组，按照内存布局 "K" 排列，并返回一个新的数组对象
assert_type(np.char.asarray(AR_S, order="K"), np.char.chararray[Any, np.dtype[np.bytes_]])
# 将字符串 "bob" 转换为字符类型的 NumPy 数组，并返回一个新的数组对象
assert_type(np.char.asarray("bob"), np.char.chararray[Any, np.dtype[np.str_]])
# 将字节串 b"bob" 转换为字符类型的 NumPy 数组，每个元素占据 5 字节，并返回一个新的数组对象
assert_type(np.char.asarray(b"bob", itemsize=5), np.char.chararray[Any, np.dtype[np.bytes_]])
# 将整数 1 转换为字节类型的 NumPy 数组，unicode=False 表示不启用 Unicode 编码，并返回一个新的数组对象
assert_type(np.char.asarray(1, unicode=False), np.char.chararray[Any, np.dtype[np.bytes_]])
# 将整数 1 转换为字符类型的 NumPy 数组，unicode=True 表示启用 Unicode 编码，并返回一个新的数组对象
assert_type(np.char.asarray(1, unicode=True), np.char.chararray[Any, np.dtype[np.str_]])
```