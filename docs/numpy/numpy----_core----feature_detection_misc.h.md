# `.\numpy\numpy\_core\feature_detection_misc.h`

```py
# 包含标准库 <stddef.h>，用于提供 size_t 类型的定义
#include <stddef.h>

# 声明函数 backtrace，返回类型为 int，接受两个参数：指向指针数组的指针和整数类型的参数
int backtrace(void **, int);

# 声明函数 madvise，返回类型为 int，接受三个参数：指向内存区域的指针、区域大小的 size_t 类型参数和一个整数类型参数
int madvise(void *, size_t, int);
```