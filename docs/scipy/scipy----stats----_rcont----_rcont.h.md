# `D:\src\scipysrc\scipy\scipy\stats\_rcont\_rcont.h`

```
#ifndef RCONT_H
#define RCONT_H

// 如果未定义 RCONT_H 宏，则开始定义该宏，用于防止头文件的多重包含


#include <stdint.h>

// 包含标准整数类型头文件，其中定义了各种整数类型，如 int64_t


#include "distributions.h"

// 包含自定义的 distributions.h 头文件，用于引入与概率分布相关的函数或数据结构声明


typedef int64_t tab_t;

// 定义 tab_t 为 int64_t 类型的别名，用于表示表格或计数器中的整数值


void rcont1_init(tab_t *work, int nc, const tab_t *c);

// 声明函数 rcont1_init，接受指向 tab_t 类型的指针 work，整数 nc，以及指向常量 tab_t 类型的指针 c 作为参数


void rcont1(tab_t *table, int nr, const tab_t *r, int nc, const tab_t *c,
            const tab_t ntot, tab_t *work, bitgen_t *rstate);

// 声明函数 rcont1，接受指向 tab_t 类型的指针 table，整数 nr，指向常量 tab_t 类型的指针 r，整数 nc，指向常量 tab_t 类型的指针 c，常量 tab_t 类型的 ntot，指向 tab_t 类型的指针 work，以及 bitgen_t 类型的指针 rstate 作为参数


void rcont2(tab_t *table, int nr, const tab_t *r, int nc, const tab_t *c,
            const tab_t ntot, bitgen_t *rstate);

// 声明函数 rcont2，接受指向 tab_t 类型的指针 table，整数 nr，指向常量 tab_t 类型的指针 r，整数 nc，指向常量 tab_t 类型的指针 c，常量 tab_t 类型的 ntot，以及 bitgen_t 类型的指针 rstate 作为参数


#endif

// 结束条件编译指令，确保在头文件结尾处 RCONT_H 宏的定义被关闭
```