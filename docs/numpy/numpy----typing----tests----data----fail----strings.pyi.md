# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\strings.pyi`

```py
import numpy as np  # 导入 NumPy 库，通常用 np 作为别名

AR_U: npt.NDArray[np.str_]  # 定义一个类型为字符串的 NumPy 数组 AR_U
AR_S: npt.NDArray[np.bytes_]  # 定义一个类型为字节字符串的 NumPy 数组 AR_S

np.strings.equal(AR_U, AR_S)  # 调用 NumPy 中的字符串比较函数，比较 AR_U 和 AR_S 是否相等，返回布尔值，可能引发类型不匹配错误

np.strings.not_equal(AR_U, AR_S)  # 调用 NumPy 中的字符串比较函数，比较 AR_U 和 AR_S 是否不相等，返回布尔值，可能引发类型不匹配错误

np.strings.greater_equal(AR_U, AR_S)  # 调用 NumPy 中的字符串比较函数，比较 AR_U 是否大于等于 AR_S，返回布尔值，可能引发类型不匹配错误

np.strings.less_equal(AR_U, AR_S)  # 调用 NumPy 中的字符串比较函数，比较 AR_U 是否小于等于 AR_S，返回布尔值，可能引发类型不匹配错误

np.strings.greater(AR_U, AR_S)  # 调用 NumPy 中的字符串比较函数，比较 AR_U 是否大于 AR_S，返回布尔值，可能引发类型不匹配错误

np.strings.less(AR_U, AR_S)  # 调用 NumPy 中的字符串比较函数，比较 AR_U 是否小于 AR_S，返回布尔值，可能引发类型不匹配错误

np.strings.encode(AR_S)  # 调用 NumPy 中的字符串编码函数，尝试将 AR_S 中的元素编码为字节串，可能引发类型不匹配错误
np.strings.decode(AR_U)  # 调用 NumPy 中的字符串解码函数，尝试将 AR_U 中的元素解码为字符串，可能引发类型不匹配错误

np.strings.join(AR_U, b"_")  # 调用 NumPy 中的字符串连接函数，使用 b"_" 将 AR_U 中的元素连接起来，可能引发类型不匹配错误
np.strings.join(AR_S, "_")  # 调用 NumPy 中的字符串连接函数，使用 "_" 将 AR_S 中的元素连接起来，可能引发类型不匹配错误

np.strings.ljust(AR_U, 5, fillchar=b"a")  # 调用 NumPy 中的字符串左对齐函数，使用 b"a" 将 AR_U 中的元素左对齐填充至长度 5，可能引发类型不匹配错误
np.strings.ljust(AR_S, 5, fillchar="a")  # 调用 NumPy 中的字符串左对齐函数，使用 "a" 将 AR_S 中的元素左对齐填充至长度 5，可能引发类型不匹配错误
np.strings.rjust(AR_U, 5, fillchar=b"a")  # 调用 NumPy 中的字符串右对齐函数，使用 b"a" 将 AR_U 中的元素右对齐填充至长度 5，可能引发类型不匹配错误
np.strings.rjust(AR_S, 5, fillchar="a")  # 调用 NumPy 中的字符串右对齐函数，使用 "a" 将 AR_S 中的元素右对齐填充至长度 5，可能引发类型不匹配错误

np.strings.lstrip(AR_U, b"a")  # 调用 NumPy 中的字符串左移除函数，移除 AR_U 中开头的 b"a"，可能引发类型不匹配错误
np.strings.lstrip(AR_S, "a")  # 调用 NumPy 中的字符串左移除函数，移除 AR_S 中开头的 "a"，可能引发类型不匹配错误
np.strings.strip(AR_U, b"a")  # 调用 NumPy 中的字符串移除函数，移除 AR_U 中首尾的 b"a"，可能引发类型不匹配错误
np.strings.strip(AR_S, "a")  # 调用 NumPy 中的字符串移除函数，移除 AR_S 中首尾的 "a"，可能引发类型不匹配错误
np.strings.rstrip(AR_U, b"a")  # 调用 NumPy 中的字符串右移除函数，移除 AR_U 中结尾的 b"a"，可能引发类型不匹配错误
np.strings.rstrip(AR_S, "a")  # 调用 NumPy 中的字符串右移除函数，移除 AR_S 中结尾的 "a"，可能引发类型不匹配错误

np.strings.partition(AR_U, b"a")  # 调用 NumPy 中的字符串分割函数，使用 b"a" 分割 AR_U，可能引发类型不匹配错误
np.strings.partition(AR_S, "a")  # 调用 NumPy 中的字符串分割函数，使用 "a" 分割 AR_S，可能引发类型不匹配错误
np.strings.rpartition(AR_U, b"a")  # 调用 NumPy 中的字符串反向分割函数，使用 b"a" 反向分割 AR_U，可能引发类型不匹配错误
np.strings.rpartition(AR_S, "a")  # 调用 NumPy 中的字符串反向分割函数，使用 "a" 反向分割 AR_S，可能引发类型不匹配错误

np.strings.split(AR_U, b"_")  # 调用 NumPy 中的字符串分割函数，使用 b"_" 分割 AR_U，可能引发类型不匹配错误
np.strings.split(AR_S, "_")  # 调用 NumPy 中的字符串分割函数，使用 "_" 分割 AR_S，可能引发类型不匹配错误
np.strings.rsplit(AR_U, b"_")  # 调用 NumPy 中的字符串反向分割函数，使用 b"_" 反向分割 AR_U，可能引发类型不匹配错误
np.strings.rsplit(AR_S, "_")  # 调用 NumPy 中的字符串反向分割函数，使用 "_" 反向分割 AR_S，可能引发类型不匹配错误

np.strings.count(AR_U, b"a", [1, 2, 3], [1, 2, 3])  # 调用 NumPy 中的字符串计数函数，计算 AR_U 中包含 b"a" 的个数，可能引发类型不匹配错误
np.strings.count(AR_S, "a", 0, 9)  # 调用 NumPy 中的字符串计数函数，计算 AR_S 中包含 "a" 的个数，可能引发类型不匹配错误

np.strings.endswith(AR_U, b"a", [1, 2, 3], [1, 2, 3])  # 调用 NumPy 中的字符串结束匹配函数，判断 AR_U 是否以 b"a" 结束，可能引发类型不匹配错误
np.strings.endswith(AR_S, "a", 0, 9)  # 调用 NumPy 中的字符串结束匹配函数，判断 AR_S 是否以 "a" 结束，可能引发类型不匹配错误
np.strings.startswith(AR_U, b"a", [1, 2, 3], [1, 2, 3])  # 调用 NumPy 中的字符串开始匹配函数，判断 AR_U 是否以 b"a" 开始，可能引发类型不匹配错误
np.strings.startswith(AR_S, "a", 0, 9)  # 调用 NumPy 中的字符串开始匹配函数，判断 AR_S 是否以 "a" 开始，可能引发类型不匹配错误

np.strings.find(AR_U, b"a", [1, 2, 3], [1, 2, 3])  # 调用 NumPy 中的字符串查找函数，查找 AR_U 中 b"a" 的位置，可能引发类型不匹配错误
np.strings.find(AR_S, "a", 0, 9)  # 调用 NumPy 中的字符串查找函数，查找 AR_S 中 "a" 的位置，可能引发类型不匹配错误
np.strings.rfind(AR_U, b
```