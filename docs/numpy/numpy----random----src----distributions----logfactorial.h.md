# `.\numpy\numpy\random\src\distributions\logfactorial.h`

```py
#ifndef LOGFACTORIAL_H
#define LOGFACTORIAL_H

# 如果 LOGFACTORIAL_H 宏未定义，则定义它，这样可以避免头文件的多重包含问题


#include <stdint.h>

# 包含标准整数类型头文件，以便使用 int64_t 这种整数类型


double logfactorial(int64_t k);

# 声明一个函数 logfactorial，接受一个 int64_t 类型的整数参数 k，返回一个 double 类型的浮点数


#endif

# 结束条件编译指令，确保当 LOGFACTORIAL_H 宏已定义时，头文件内容正常结束
```