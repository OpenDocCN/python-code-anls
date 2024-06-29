# `.\numpy\numpy\testing\print_coercion_tables.py`

```py
# 指定 Python 解释器为当前环境的 Python 3
#!/usr/bin/env python3
"""Prints type-coercion tables for the built-in NumPy types

"""
# 导入 NumPy 库并简写为 np
import numpy as np
# 从 NumPy 库中导入 obj2sctype 函数
from numpy._core.numerictypes import obj2sctype
# 导入 namedtuple 类
from collections import namedtuple

# 定义一个通用对象，可以进行加法操作但不做其他操作
class GenericObject:
    def __init__(self, v):
        self.v = v

    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self

    # 设置 dtype 属性为对象类型 'O'
    dtype = np.dtype('O')

# 定义函数 print_cancast_table，打印 NumPy 类型间的类型转换表
def print_cancast_table(ntypes):
    # 打印表头，显示类型字符
    print('X', end=' ')
    for char in ntypes:
        print(char, end=' ')
    print()
    # 打印每行类型及其与其他类型间的转换关系
    for row in ntypes:
        print(row, end=' ')
        for col in ntypes:
            # 根据 np.can_cast 函数返回值选择合适的标记符号
            if np.can_cast(row, col, "equiv"):
                cast = "#"
            elif np.can_cast(row, col, "safe"):
                cast = "="
            elif np.can_cast(row, col, "same_kind"):
                cast = "~"
            elif np.can_cast(row, col, "unsafe"):
                cast = "."
            else:
                cast = " "
            print(cast, end=' ')
        print()

# 定义函数 print_coercion_table，打印 NumPy 类型间的类型强制转换表
def print_coercion_table(ntypes, inputfirstvalue, inputsecondvalue, firstarray, use_promote_types=False):
    # 打印表头，显示类型字符
    print('+', end=' ')
    for char in ntypes:
        print(char, end=' ')
    print()
    # 打印每行类型及其与其他类型间的强制转换关系
    for row in ntypes:
        if row == 'O':
            rowtype = GenericObject
        else:
            rowtype = obj2sctype(row)

        print(row, end=' ')
        for col in ntypes:
            if col == 'O':
                coltype = GenericObject
            else:
                coltype = obj2sctype(col)
            try:
                # 根据输入值创建 NumPy 数组或对象，计算强制转换结果
                if firstarray:
                    rowvalue = np.array([rowtype(inputfirstvalue)], dtype=rowtype)
                else:
                    rowvalue = rowtype(inputfirstvalue)
                colvalue = coltype(inputsecondvalue)
                if use_promote_types:
                    char = np.promote_types(rowvalue.dtype, colvalue.dtype).char
                else:
                    value = np.add(rowvalue, colvalue)
                    if isinstance(value, np.ndarray):
                        char = value.dtype.char
                    else:
                        char = np.dtype(type(value)).char
            except ValueError:
                char = '!'
            except OverflowError:
                char = '@'
            except TypeError:
                char = '#'
            print(char, end=' ')
        print()

# 定义函数 print_new_cast_table，打印新的类型转换表
def print_new_cast_table(*, can_cast=True, legacy=False, flags=False):
    """Prints new casts, the values given are default "can-cast" values, not
    actual ones.
    """
    # 导入私有模块函数 get_all_cast_information
    from numpy._core._multiarray_tests import get_all_cast_information

    # 设置新的类型转换标志
    cast_table = {
        -1: " ",
        0: "#",  # No cast (classify as equivalent here)
        1: "#",  # equivalent casting
        2: "=",  # safe casting
        3: "~",  # same-kind casting
        4: ".",  # unsafe casting
    }
    # 定义一个字典，将整数映射到对应的特定字符表示
    flags_table = {
        0 : "▗", 7: "█",
        1: "▚", 2: "▐", 4: "▄",
                3: "▜", 5: "▙",
                        6: "▟",
    }

    # 创建一个命名元组类型cast_info，包含can_cast、legacy、flags三个字段
    cast_info = namedtuple("cast_info", ["can_cast", "legacy", "flags"])
    # 定义一个用于表示没有转换信息的特殊命名元组实例
    no_cast_info = cast_info(" ", " ", " ")

    # 获取所有转换信息的列表
    casts = get_all_cast_information()
    # 创建一个空字典用于存储转换信息的表格
    table = {}
    # 创建一个空集合，用于收集所有出现过的数据类型
    dtypes = set()
    
    # 遍历每个转换信息
    for cast in casts:
        # 将转换源和目标数据类型添加到数据类型集合中
        dtypes.add(cast["from"])
        dtypes.add(cast["to"])

        # 如果转换源数据类型尚未在表格中，则创建一个空字典用于存储转换目标和转换信息
        if cast["from"] not in table:
            table[cast["from"]] = {}
        to_dict = table[cast["from"]]

        # 根据转换信息设置can_cast、legacy和flags字段
        can_cast = cast_table[cast["casting"]]
        legacy = "L" if cast["legacy"] else "."
        flags = 0
        if cast["requires_pyapi"]:
            flags |= 1
        if cast["supports_unaligned"]:
            flags |= 2
        if cast["no_floatingpoint_errors"]:
            flags |= 4

        # 将整数表示的flags映射为对应的字符
        flags = flags_table[flags]
        # 将转换目标和对应的cast_info存入表格中
        to_dict[cast["to"]] = cast_info(can_cast=can_cast, legacy=legacy, flags=flags)

    # 下面开始处理数据类型排序的函数和打印表格的函数
    # 注释以下函数
    types = np.typecodes["All"]
    def sorter(x):
        dtype = np.dtype(x.type)
        try:
            indx = types.index(dtype.char)
        except ValueError:
            indx = np.inf
        return (indx, dtype.char)

    # 对数据类型集合进行排序，使用sorter函数进行排序依据
    dtypes = sorted(dtypes, key=sorter)

    # 定义一个打印表格的函数，field参数指定打印哪个字段的信息
    def print_table(field="can_cast"):
        print('X', end=' ')
        for dt in dtypes:
            print(np.dtype(dt.type).char, end=' ')
        print()
        for from_dt in dtypes:
            print(np.dtype(from_dt.type).char, end=' ')
            row = table.get(from_dt, {})
            for to_dt in dtypes:
                print(getattr(row.get(to_dt, no_cast_info), field), end=' ')
            print()

    # 判断是否存在can_cast，如果存在则打印can_cast表格
    if can_cast:
        print()
        print("Casting: # is equivalent, = is safe, ~ is same-kind, and . is unsafe")
        print()
        print_table("can_cast")

    # 判断是否存在legacy，如果存在则打印legacy表格
    if legacy:
        print()
        print("L denotes a legacy cast . a non-legacy one.")
        print()
        print_table("legacy")

    # 判断是否存在flags，如果存在则打印flags相关信息和对应的表格
    if flags:
        print()
        print(f"{flags_table[0]}: no flags, {flags_table[1]}: PyAPI, "
              f"{flags_table[2]}: supports unaligned, {flags_table[4]}: no-float-errors")
        print()
        print_table("flags")
# 如果当前脚本作为主程序运行，则执行以下代码块
if __name__ == '__main__':
    # 打印提示信息："can cast"
    print("can cast")
    # 调用函数 print_cancast_table，打印 numpy 所有数据类型的转换表格
    print_cancast_table(np.typecodes['All'])
    # 打印空行
    print()
    # 打印提示信息："In these tables, ValueError is '!', OverflowError is '@', TypeError is '#'"
    print("In these tables, ValueError is '!', OverflowError is '@', TypeError is '#'")
    # 打印空行
    print()
    # 打印提示信息："scalar + scalar"
    print("scalar + scalar")
    # 调用函数 print_coercion_table，打印将 numpy 所有数据类型与自身相加时的转换表格
    print_coercion_table(np.typecodes['All'], 0, 0, False)
    # 打印空行
    print()
    # 打印提示信息："scalar + neg scalar"
    print("scalar + neg scalar")
    # 调用函数 print_coercion_table，打印将 numpy 所有数据类型与负数自身相加时的转换表格
    print_coercion_table(np.typecodes['All'], 0, -1, False)
    # 打印空行
    print()
    # 打印提示信息："array + scalar"
    print("array + scalar")
    # 调用函数 print_coercion_table，打印将 numpy 所有数据类型与数组相加时的转换表格
    print_coercion_table(np.typecodes['All'], 0, 0, True)
    # 打印空行
    print()
    # 打印提示信息："array + neg scalar"
    print("array + neg scalar")
    # 调用函数 print_coercion_table，打印将 numpy 所有数据类型与负数数组相加时的转换表格
    print_coercion_table(np.typecodes['All'], 0, -1, True)
    # 打印空行
    print()
    # 打印提示信息："promote_types"
    print("promote_types")
    # 调用函数 print_coercion_table，打印 numpy 所有数据类型的提升表格
    print_coercion_table(np.typecodes['All'], 0, 0, False, True)
    # 打印提示信息："New casting type promotion:"
    print("New casting type promotion:")
    # 调用函数 print_new_cast_table，打印新的类型转换提升表格
    print_new_cast_table(can_cast=True, legacy=True, flags=True)
```