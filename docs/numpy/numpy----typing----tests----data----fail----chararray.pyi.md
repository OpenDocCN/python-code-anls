# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\chararray.pyi`

```py
import numpy as np
from typing import Any

# 定义一个类型为 np.char.chararray 的变量 AR_U，元素类型为 np.str_
AR_U: np.char.chararray[Any, np.dtype[np.str_]]
# 定义一个类型为 np.char.chararray 的变量 AR_S，元素类型为 np.bytes_
AR_S: np.char.chararray[Any, np.dtype[np.bytes_]]

# 对 AR_S 进行编码操作，但使用了无效的 self 参数
AR_S.encode()  # E: Invalid self argument
# 对 AR_U 进行解码操作，但使用了无效的 self 参数
AR_U.decode()  # E: Invalid self argument

# 尝试将 AR_U 中的每个元素用 b"_" 连接，但类型不兼容
AR_U.join(b"_")  # E: incompatible type
# 尝试将 AR_S 中的每个元素用 "_" 连接，但类型不兼容
AR_S.join("_")  # E: incompatible type

# 在 AR_U 中的每个元素右侧填充字符 b"a"，长度为 5，但类型不兼容
AR_U.ljust(5, fillchar=b"a")  # E: incompatible type
# 在 AR_S 中的每个元素右侧填充字符 "a"，长度为 5，但类型不兼容
AR_S.ljust(5, fillchar="a")  # E: incompatible type
# 在 AR_U 中的每个元素左侧填充字符 b"a"，长度为 5，但类型不兼容
AR_U.rjust(5, fillchar=b"a")  # E: incompatible type
# 在 AR_S 中的每个元素左侧填充字符 "a"，长度为 5，但类型不兼容
AR_S.rjust(5, fillchar="a")  # E: incompatible type

# 移除 AR_U 中的每个元素左侧的字符 b"a"，但类型不兼容
AR_U.lstrip(chars=b"a")  # E: incompatible type
# 移除 AR_S 中的每个元素左侧的字符 "a"，但类型不兼容
AR_S.lstrip(chars="a")  # E: incompatible type
# 移除 AR_U 中的每个元素两侧的字符 b"a"，但类型不兼容
AR_U.strip(chars=b"a")  # E: incompatible type
# 移除 AR_S 中的每个元素两侧的字符 "a"，但类型不兼容
AR_S.strip(chars="a")  # E: incompatible type
# 移除 AR_U 中的每个元素右侧的字符 b"a"，但类型不兼容
AR_U.rstrip(chars=b"a")  # E: incompatible type
# 移除 AR_S 中的每个元素右侧的字符 "a"，但类型不兼容
AR_S.rstrip(chars="a")  # E: incompatible type

# 在 AR_U 中的每个元素中寻找第一个出现的 b"a"，但类型不兼容
AR_U.partition(b"a")  # E: incompatible type
# 在 AR_S 中的每个元素中寻找第一个出现的 "a"，但类型不兼容
AR_S.partition("a")  # E: incompatible type
# 在 AR_U 中的每个元素中寻找最后一个出现的 b"a"，但类型不兼容
AR_U.rpartition(b"a")  # E: incompatible type
# 在 AR_S 中的每个元素中寻找最后一个出现的 "a"，但类型不兼容
AR_S.rpartition("a")  # E: incompatible type

# 将 AR_U 中的每个元素中的 b"_" 替换为 b"-"，但类型不兼容
AR_U.replace(b"_", b"-")  # E: incompatible type
# 将 AR_S 中的每个元素中的 "_" 替换为 "-"，但类型不兼容
AR_S.replace("_", "-")  # E: incompatible type

# 使用 b"_" 将 AR_U 中的每个元素分割，但类型不兼容
AR_U.split(b"_")  # E: incompatible type
# 使用 "_" 将 AR_S 中的每个元素分割，但类型不兼容
AR_S.split("_")  # E: incompatible type
# 尝试使用整数 1 将 AR_S 中的每个元素分割，但类型不兼容
AR_S.split(1)  # E: incompatible type
# 使用 b"_" 将 AR_U 中的每个元素从右侧开始分割，但类型不兼容
AR_U.rsplit(b"_")  # E: incompatible type
# 使用 "_" 将 AR_S 中的每个元素从右侧开始分割，但类型不兼容
AR_S.rsplit("_")  # E: incompatible type

# 统计 AR_U 中每个元素中出现的 b"a" 的数量，但类型不兼容
AR_U.count(b"a", start=[1, 2, 3])  # E: incompatible type
# 统计 AR_S 中每个元素中出现的 "a" 的数量，但类型不兼容
AR_S.count("a", end=9)  # E: incompatible type

# 检查 AR_U 中每个元素是否以 b"a" 结尾，但类型不兼容
AR_U.endswith(b"a", start=[1, 2, 3])  # E: incompatible type
# 检查 AR_S 中每个元素是否以 "a" 结尾，但类型不兼容
AR_S.endswith("a", end=9)  # E: incompatible type
# 检查 AR_U 中每个元素是否以 b"a" 开头，但类型不兼容
AR_U.startswith(b"a", start=[1, 2, 3])  # E: incompatible type
# 检查 AR_S 中每个元素是否以 "a" 开头，但类型不兼容
AR_S.startswith("a", end=9)  # E: incompatible type

# 查找 AR_U 中每个元素中第一次出现 b"a" 的位置，但类型不兼容
AR_U.find(b"a", start=[1, 2, 3])  # E: incompatible type
# 查找 AR_S 中每个元素中第一次出现 "a" 的位置，但类型不兼容
AR_S.find("a", end=9)  # E: incompatible type
# 查找 AR_U 中每个元素中最后一次出现 b"a" 的位置，但类型不兼容
AR_U.rfind(b"a", start=[1, 2, 3])  # E: incompatible type
# 查找 AR_S 中每个元素中最后一次出现 "a" 的位置，但类型不兼容
AR_S.rfind("a", end=9)  # E: incompatible type

# 查找 AR_U 是否等于 AR_S，但操作数类型不支持
AR_U == AR_S  # E: Unsupported operand types
# 查找 AR_U 是否不等于 AR_S，但操作数类型不支持
AR_U != AR_S  # E: Unsupported operand types
# 查找 AR_U 是否大于等于 AR_S，但操作数类型不支持
AR_U >= AR_S  # E: Unsupported operand types
# 查找 AR_U 是否小于等于 AR_S，但操作数类型不支持
AR_U <= AR_S  # E: Unsupported operand types
# 查找 AR_U 是否大于 AR_S，但操作数类型不支持
AR_U > AR_S  # E: Unsupported operand types
# 查找 AR_U 是否小于 AR_S，但操作数类型不支持
AR_U < AR_S  # E: Unsupported operand types
```