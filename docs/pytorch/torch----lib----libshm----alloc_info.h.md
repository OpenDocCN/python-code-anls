# `.\pytorch\torch\lib\libshm\alloc_info.h`

```
#pragma once
// 使用 pragma once 防止多次包含同一个头文件

#include <unistd.h>
// 包含标准 C 库中的 unistd.h 头文件

struct AllocInfo {
  pid_t pid;             // 定义结构体 AllocInfo，包含一个 pid_t 类型的成员变量 pid
  char free;             // 定义一个 char 类型的成员变量 free
  char filename[60];     // 定义一个 char 类型的数组成员变量 filename，长度为 60
};
// 定义一个结构体 AllocInfo，用于存储进程 ID、空闲状态和文件名信息
```