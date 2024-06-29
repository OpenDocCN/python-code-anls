# `D:\src\scipysrc\numpy\numpy\char\__init__.pyi`

```py
# 从 numpy._core.defchararray 模块中导入一系列字符串操作函数，用于操作字符数组

from numpy._core.defchararray import (
    equal as equal,                     # 别名 equal，用于比较两个字符数组是否相等
    not_equal as not_equal,             # 别名 not_equal，用于比较两个字符数组是否不相等
    greater_equal as greater_equal,     # 别名 greater_equal，用于比较字符数组左侧是否大于等于右侧
    less_equal as less_equal,           # 别名 less_equal，用于比较字符数组左侧是否小于等于右侧
    greater as greater,                 # 别名 greater，用于比较字符数组左侧是否大于右侧
    less as less,                       # 别名 less，用于比较字符数组左侧是否小于右侧
    str_len as str_len,                 # 别名 str_len，返回字符数组中每个字符串的长度
    add as add,                         # 别名 add，用于字符数组的逐元素相加
    multiply as multiply,               # 别名 multiply，用于字符数组的逐元素相乘
    mod as mod,                         # 别名 mod，用于字符数组的逐元素取模运算
    capitalize as capitalize,           # 别名 capitalize，将字符数组中每个字符串的首字母大写
    center as center,                   # 别名 center，将字符数组中每个字符串居中对齐
    count as count,                     # 别名 count，统计字符数组中每个字符串出现的次数
    decode as decode,                   # 别名 decode，解码字符数组中的每个字符串
    encode as encode,                   # 别名 encode，编码字符数组中的每个字符串
    endswith as endswith,               # 别名 endswith，判断字符数组中每个字符串是否以指定后缀结尾
    expandtabs as expandtabs,           # 别名 expandtabs，将字符数组中每个字符串的制表符扩展为空格
    find as find,                       # 别名 find，查找字符数组中每个字符串第一次出现的位置
    index as index,                     # 别名 index，查找字符数组中每个字符串第一次出现的位置
    isalnum as isalnum,                 # 别名 isalnum，判断字符数组中每个字符串是否由字母和数字组成
    isalpha as isalpha,                 # 别名 isalpha，判断字符数组中每个字符串是否只包含字母
    isdigit as isdigit,                 # 别名 isdigit，判断字符数组中每个字符串是否只包含数字
    islower as islower,                 # 别名 islower，判断字符数组中每个字符串是否全为小写字母
    isspace as isspace,                 # 别名 isspace，判断字符数组中每个字符串是否只包含空白字符
    istitle as istitle,                 # 别名 istitle，判断字符数组中每个字符串是否遵循标题化规则
    isupper as isupper,                 # 别名 isupper，判断字符数组中每个字符串是否全为大写字母
    join as join,                       # 别名 join，将字符数组中每个字符串用指定分隔符连接成一个字符串
    ljust as ljust,                     # 别名 ljust，将字符数组中每个字符串左对齐，并用空格填充至指定长度
    lower as lower,                     # 别名 lower，将字符数组中每个字符串转换为小写
    lstrip as lstrip,                   # 别名 lstrip，将字符数组中每个字符串左侧的空白字符删除
    partition as partition,             # 别名 partition，将字符数组中每个字符串按照第一次出现的分隔符分为三部分
    replace as replace,                 # 别名 replace，将字符数组中每个字符串的指定子串替换为新的子串
    rfind as rfind,                     # 别名 rfind，查找字符数组中每个字符串最后一次出现的位置
    rindex as rindex,                   # 别名 rindex，查找字符数组中每个字符串最后一次出现的位置
    rjust as rjust,                     # 别名 rjust，将字符数组中每个字符串右对齐，并用空格填充至指定长度
    rpartition as rpartition,           # 别名 rpartition，将字符数组中每个字符串按照最后一次出现的分隔符分为三部分
    rsplit as rsplit,                   # 别名 rsplit，将字符数组中每个字符串按照指定分隔符从右向左分割
    rstrip as rstrip,                   # 别名 rstrip，将字符数组中每个字符串右侧的空白字符删除
    split as split,                     # 别名 split，将字符数组中每个字符串按照指定分隔符分割
    splitlines as splitlines,           # 别名 splitlines，将字符数组中每个字符串按照行分割
    startswith as startswith,           # 别名 startswith，判断字符数组中每个字符串是否以指定前缀开头
    strip as strip,                     # 别名 strip，将字符数组中每个字符串两侧的空白字符删除
    swapcase as swapcase,               # 别名 swapcase，将字符数组中每个字符串的大小写互换
    title as title,                     # 别名 title，将字符数组中每个字符串转换为标题化形式
    translate as translate,             # 别名 translate，根据字符映射表对字符数组中每个字符串进行转换
    upper as upper,                     # 别名 upper，将字符数组中每个字符串转换为大写
    zfill as zfill,                     # 别名 zfill，将字符数组中每个字符串左侧填充零到指定宽度
    isnumeric as isnumeric,             # 别名 isnumeric，判断字符数组中每个字符串是否只包含数字字符
    isdecimal as isdecimal,             # 别名 isdecimal，判断字符数组中每个字符串是否只包含十进制数字字符
    array as array,                     # 别名 array，创建一个字符数组对象
    asarray as asarray,                 # 别名 asarray，将输入转换为字符数组对象
    compare_chararrays as compare_chararrays,  # 别名 compare_chararrays，比较两个字符数组对象
    chararray as chararray              # 别名 chararray，字符数组对象类型
)

__all__: list[str]                      # 定义变量 __all__，包含在 from ... import * 语句中应导入的公共名称列表
```