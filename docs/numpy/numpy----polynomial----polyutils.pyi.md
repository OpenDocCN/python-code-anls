# `.\numpy\numpy\polynomial\polyutils.pyi`

```py
# 定义全局变量 __all__，表示在使用 from module import * 时应该导入的符号列表
__all__: list[str]

# 定义函数 trimseq，用于修剪序列，返回一个修剪后的序列
def trimseq(seq):
    ...

# 定义函数 as_series，将列表转换为序列（可能是 pandas.Series），可选地修剪元素
def as_series(alist, trim=...):
    ...

# 定义函数 trimcoef，用于修剪系数，返回修剪后的系数
def trimcoef(c, tol=...):
    ...

# 定义函数 getdomain，获取变量 x 的定义域
def getdomain(x):
    ...

# 定义函数 mapparms，将旧参数映射到新参数，返回映射后的结果
def mapparms(old, new):
    ...

# 定义函数 mapdomain，将变量 x 从旧域映射到新域，返回映射后的结果
def mapdomain(x, old, new):
    ...

# 定义函数 format_float，格式化浮点数 x 的字符串表示，可选地加括号
def format_float(x, parens=...):
    ...
```