# `.\numpy\numpy\strings\__init__.pyi`

```py
# 导入 numpy._core.strings 模块中的字符串操作函数，并重命名为易于理解的名称
from numpy._core.strings import (
    equal as equal,                     # 导入字符串相等比较函数，并重命名为 equal
    not_equal as not_equal,             # 导入字符串不相等比较函数，并重命名为 not_equal
    greater_equal as greater_equal,     # 导入字符串大于等于比较函数，并重命名为 greater_equal
    less_equal as less_equal,           # 导入字符串小于等于比较函数，并重命名为 less_equal
    greater as greater,                 # 导入字符串大于比较函数，并重命名为 greater
    less as less,                       # 导入字符串小于比较函数，并重命名为 less
    add as add,                         # 导入字符串连接函数，并重命名为 add
    multiply as multiply,               # 导入字符串乘法函数，并重命名为 multiply
    mod as mod,                         # 导入字符串取模函数，并重命名为 mod
    isalpha as isalpha,                 # 导入判断字符串是否只包含字母函数，并重命名为 isalpha
    isalnum as isalnum,                 # 导入判断字符串是否只包含字母和数字函数，并重命名为 isalnum
    isdigit as isdigit,                 # 导入判断字符串是否只包含数字函数，并重命名为 isdigit
    isspace as isspace,                 # 导入判断字符串是否只包含空格函数，并重命名为 isspace
    isnumeric as isnumeric,             # 导入判断字符串是否只包含数字函数，并重命名为 isnumeric
    isdecimal as isdecimal,             # 导入判断字符串是否只包含十进制数字函数，并重命名为 isdecimal
    islower as islower,                 # 导入判断字符串是否全部小写函数，并重命名为 islower
    isupper as isupper,                 # 导入判断字符串是否全部大写函数，并重命名为 isupper
    istitle as istitle,                 # 导入判断字符串是否标题化（每个单词首字母大写）函数，并重命名为 istitle
    str_len as str_len,                 # 导入获取字符串长度函数，并重命名为 str_len
    find as find,                       # 导入在字符串中查找子字符串函数，并重命名为 find
    rfind as rfind,                     # 导入从右侧开始在字符串中查找子字符串函数，并重命名为 rfind
    index as index,                     # 导入返回子字符串第一次出现的索引函数，并重命名为 index
    rindex as rindex,                   # 导入返回子字符串最后一次出现的索引函数，并重命名为 rindex
    count as count,                     # 导入计算子字符串出现次数函数，并重命名为 count
    startswith as startswith,           # 导入判断字符串是否以指定子字符串开头函数，并重命名为 startswith
    endswith as endswith,               # 导入判断字符串是否以指定子字符串结尾函数，并重命名为 endswith
    decode as decode,                   # 导入解码字符串函数，并重命名为 decode
    encode as encode,                   # 导入编码字符串函数，并重命名为 encode
    expandtabs as expandtabs,           # 导入将字符串中的制表符扩展为空格函数，并重命名为 expandtabs
    center as center,                   # 导入使字符串居中对齐函数，并重命名为 center
    ljust as ljust,                     # 导入使字符串左对齐函数，并重命名为 ljust
    rjust as rjust,                     # 导入使字符串右对齐函数，并重命名为 rjust
    lstrip as lstrip,                   # 导入去除字符串左侧空白字符函数，并重命名为 lstrip
    rstrip as rstrip,                   # 导入去除字符串右侧空白字符函数，并重命名为 rstrip
    strip as strip,                     # 导入去除字符串两侧空白字符函数，并重命名为 strip
    zfill as zfill,                     # 导入字符串右侧补零函数，并重命名为 zfill
    upper as upper,                     # 导入将字符串转换为大写函数，并重命名为 upper
    lower as lower,                     # 导入将字符串转换为小写函数，并重命名为 lower
    swapcase as swapcase,               # 导入交换字符串大小写函数，并重命名为 swapcase
    capitalize as capitalize,           # 导入将字符串首字母大写函数，并重命名为 capitalize
    title as title,                     # 导入将字符串标题化函数，并重命名为 title
    replace as replace,                 # 导入替换字符串中子字符串函数，并重命名为 replace
    join as join,                       # 导入连接可迭代对象中的字符串函数，并重命名为 join
    split as split,                     # 导入按指定分隔符拆分字符串函数，并重命名为 split
    rsplit as rsplit,                   # 导入从右侧开始按指定分隔符拆分字符串函数，并重命名为 rsplit
    splitlines as splitlines,           # 导入按行拆分字符串函数，并重命名为 splitlines
    partition as partition,             # 导入将字符串分割为三部分函数，并重命名为 partition
    rpartition as rpartition,           # 导入将字符串从右侧开始分割为三部分函数，并重命名为 rpartition
    translate as translate              # 导入根据指定映射替换字符串中的字符函数，并重命名为 translate
)

__all__: list[str]                      # 定义 __all__ 变量，指定在使用 from module import * 时导入的符号列表为 str 类型
```