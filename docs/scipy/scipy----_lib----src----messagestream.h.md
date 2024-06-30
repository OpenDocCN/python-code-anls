# `D:\src\scipysrc\scipy\scipy\_lib\src\messagestream.h`

```
#ifndef MESSAGESTREAM_H_
#define MESSAGESTREAM_H_

#include <stdio.h>  // 包含标准输入输出库的头文件

#include "messagestream_config.h"  // 包含消息流配置文件的头文件

#if HAVE_OPEN_MEMSTREAM
// 如果支持 open_memstream 函数，则定义该函数，返回一个可写入内存的文件指针
FILE *messagestream_open_memstream(char **ptr, size_t *sizeloc)
{
    return open_memstream(ptr, sizeloc);
}
#else
// 如果不支持 open_memstream 函数，则定义该函数返回 NULL
FILE *messagestream_open_memstream(char **ptr, size_t *sizeloc)
{
    return NULL;
}
#endif

#endif /* MESSAGESTREAM_H_ */
```