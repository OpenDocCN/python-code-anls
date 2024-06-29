# `D:\src\scipysrc\numpy\numpy\_core\_type_aliases.pyi`

```
# 导入 numpy 库中的 generic 类型
from numpy import generic

# 定义一个类型为 dict 的变量 sctypeDict，键类型为 int 或 str，值类型为 numpy 的 generic 类型
sctypeDict: dict[int | str, type[generic]]
```