# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\char.pyi`

```py
import numpy as np  # 导入 NumPy 库，通常用 np 作为别名

AR_U: npt.NDArray[np.str_]  # 定义 AR_U 变量，类型为 NumPy 字符串数组
AR_S: npt.NDArray[np.bytes_]  # 定义 AR_S 变量，类型为 NumPy 字节串数组

np.char.equal(AR_U, AR_S)  # 使用 NumPy 字符串函数比较 AR_U 和 AR_S 中的元素是否相等，返回布尔数组  # E: incompatible type

np.char.not_equal(AR_U, AR_S)  # 使用 NumPy 字符串函数比较 AR_U 和 AR_S 中的元素是否不相等，返回布尔数组  # E: incompatible type

np.char.greater_equal(AR_U, AR_S)  # 使用 NumPy 字符串函数比较 AR_U 是否大于等于 AR_S 中的元素，返回布尔数组  # E: incompatible type

np.char.less_equal(AR_U, AR_S)  # 使用 NumPy 字符串函数比较 AR_U 是否小于等于 AR_S 中的元素，返回布尔数组  # E: incompatible type

np.char.greater(AR_U, AR_S)  # 使用 NumPy 字符串函数比较 AR_U 是否大于 AR_S 中的元素，返回布尔数组  # E: incompatible type

np.char.less(AR_U, AR_S)  # 使用 NumPy 字符串函数比较 AR_U 是否小于 AR_S 中的元素，返回布尔数组  # E: incompatible type

np.char.encode(AR_S)  # 对 AR_S 中的每个元素进行编码为字节串的操作，返回一个新的数组  # E: incompatible type
np.char.decode(AR_U)  # 对 AR_U 中的每个元素进行解码为字符串的操作，返回一个新的数组  # E: incompatible type

np.char.join(AR_U, b"_")  # 使用指定的分隔符 b"_" 将 AR_U 中的每个字符串数组元素连接成一个单独的字符串  # E: incompatible type
np.char.join(AR_S, "_")  # 使用指定的分隔符 "_" 将 AR_S 中的每个字节串数组元素连接成一个单独的字节串  # E: incompatible type

np.char.ljust(AR_U, 5, fillchar=b"a")  # 对 AR_U 中的每个字符串进行左对齐操作，不足部分使用 fillchar=b"a" 填充到长度 5  # E: incompatible type
np.char.ljust(AR_S, 5, fillchar="a")  # 对 AR_S 中的每个字节串进行左对齐操作，不足部分使用 fillchar="a" 填充到长度 5  # E: incompatible type
np.char.rjust(AR_U, 5, fillchar=b"a")  # 对 AR_U 中的每个字符串进行右对齐操作，不足部分使用 fillchar=b"a" 填充到长度 5  # E: incompatible type
np.char.rjust(AR_S, 5, fillchar="a")  # 对 AR_S 中的每个字节串进行右对齐操作，不足部分使用 fillchar="a" 填充到长度 5  # E: incompatible type

np.char.lstrip(AR_U, chars=b"a")  # 对 AR_U 中的每个字符串进行左侧去除指定字符 b"a" 的操作  # E: incompatible type
np.char.lstrip(AR_S, chars="a")  # 对 AR_S 中的每个字节串进行左侧去除指定字符 "a" 的操作  # E: incompatible type
np.char.strip(AR_U, chars=b"a")  # 对 AR_U 中的每个字符串进行两侧去除指定字符 b"a" 的操作  # E: incompatible type
np.char.strip(AR_S, chars="a")  # 对 AR_S 中的每个字节串进行两侧去除指定字符 "a" 的操作  # E: incompatible type
np.char.rstrip(AR_U, chars=b"a")  # 对 AR_U 中的每个字符串进行右侧去除指定字符 b"a" 的操作  # E: incompatible type
np.char.rstrip(AR_S, chars="a")  # 对 AR_S 中的每个字节串进行右侧去除指定字符 "a" 的操作  # E: incompatible type

np.char.partition(AR_U, b"a")  # 使用分隔符 b"a" 对 AR_U 中的每个字符串进行分割，返回分割后的数组  # E: incompatible type
np.char.partition(AR_S, "a")  # 使用分隔符 "a" 对 AR_S 中的每个字节串进行分割，返回分割后的数组  # E: incompatible type
np.char.rpartition(AR_U, b"a")  # 使用分隔符 b"a" 对 AR_U 中的每个字符串进行从右侧开始分割，返回分割后的数组  # E: incompatible type
np.char.rpartition(AR_S, "a")  # 使用分隔符 "a" 对 AR_S 中的每个字节串进行从右侧开始分割，返回分割后的数组  # E: incompatible type

np.char.replace(AR_U, b"_", b"-")  # 将 AR_U 中的每个字符串中的 b"_" 替换为 b"-"，返回替换后的数组  # E: incompatible type
np.char.replace(AR_S, "_", "-")  # 将 AR_S 中的每个字节串中的 "_" 替换为 "-"，返回替换后的数组  # E: incompatible type

np.char.split(AR_U, b"_")  # 使用分隔符 b"_" 对 AR_U 中的每个字符串进行分割，返回分割后的数组  # E: incompatible type
np.char.split(AR_S, "_")  # 使用分隔符 "_" 对 AR_S 中的每个字节串进行分割，返回分割后的数组  # E: incompatible type
np.char.rsplit(AR_U, b"_")  # 使用分隔符 b"_" 对 AR_U 中的每个字符串进行从右侧开始分割，返回分割后的数组  # E: incompatible type
np.char.rsplit(AR_S, "_")  # 使用分隔符 "_" 对 AR_S 中的每个字节串进行从右侧开始分割，返回分割后的数组  # E: incompatible type

np.char.count(AR_U, b"a", start=[1, 2, 3])  # 计算 AR_U 中的每个字符串中字符 b"a" 的出现次数，从指定位置 [1, 2, 3] 开始计数  # E: incompatible type
np.char.count(AR_S, "a", end=9)  # 计算 AR_S 中的每个字节串中字符 "a" 的出现次数，直到指定位置 9 结束计数  # E: incompatible type

np.char.endswith(AR_U, b"a", start=[1, 2, 3])  # 检查 AR_U 中的每个字符串是否以字符 b"a" 结尾，从指定位置 [1, 2, 3] 开始检查  # E: incompatible type
np.char.endswith(AR_S, "a", end=9)  # 检查 AR_S 中的每个字节串是否以字符 "a" 结尾，直到指定位置 9 结束检查  # E: incompatible type
np.char.startswith(AR_U, b"a", start=[1, 2, 3])  # 检查 AR_U 中的每个字符串是否以字符 b"a" 开头，从指定位置 [1, 2, 3] 开始检查  # E: incompatible type
np.char.startswith(AR_S, "a", end=9)  # 检查 AR_S 中的每个字节串是否以字符 "a" 开头，直到指定位置 9 结束检查  # E: incompatible type

np.char.find(AR_U, b"a", start=[1, 2, 3])  # 在 AR_U 中的每个字符串中查找字符 b"a"，从指定位置 [1, 2, 3] 开始查找，返回找到的第一个位置索引  # E: incompatible type
np.char.find(AR_S, "a", end=9)  # 在 AR_S 中的每个字节串中查找字符 "a"，直到指定位置 9 结束查找，
```