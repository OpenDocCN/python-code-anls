# `.\numpy\numpy\typing\tests\data\pass\arrayprint.py`

```
# 导入NumPy库，简称为np
import numpy as np

# 创建一个包含0到9的一维数组
AR = np.arange(10)

# 设置数组为只读模式，即不允许修改数组的值
AR.setflags(write=False)

# 使用默认的打印选项打印NumPy数组
with np.printoptions():
    # 设置打印选项，包括小数点后的精度为1，阈值为2，显示前后各3个元素，每行打印4个字符
    np.set_printoptions(
        precision=1,
        threshold=2,
        edgeitems=3,
        linewidth=4,
        suppress=False,
        nanstr="Bob",
        infstr="Bill",
        formatter={},
        sign="+",
        floatmode="unique",
    )
    
    # 获取当前的打印选项
    np.get_printoptions()
    
    # 将NumPy数组AR转换为字符串形式
    str(AR)

    # 将NumPy数组AR转换为指定格式的字符串
    np.array2string(
        AR,
        max_line_width=5,  # 每行最大宽度为5个字符
        precision=2,       # 小数点后精度为2
        suppress_small=True,  # 抑制小的浮点数形式
        separator=";",     # 数组元素的分隔符为分号
        prefix="test",     # 输出字符串的前缀为'test'
        threshold=5,       # 打印数组的阈值为5
        floatmode="fixed",  # 使用固定小数点模式
        suffix="?",        # 输出字符串的后缀为'?'
        legacy="1.13",     # 遗留模式为1.13版本
    )
    
    # 格式化一个浮点数为科学计数法字符串
    np.format_float_scientific(1, precision=5)
    
    # 格式化一个浮点数为定点形式的字符串
    np.format_float_positional(1, trim="k")
    
    # 返回NumPy数组AR的表达式形式的字符串
    np.array_repr(AR)
    
    # 返回NumPy数组AR的字符串形式
    np.array_str(AR)
```