# `.\pytorch\c10\util\static_tracepoint_elfx86.h`

```py
#pragma once
// 定义默认的操作数探测约束
#ifndef TORCH_SDT_ARG_CONSTRAINT
#define TORCH_SDT_ARG_CONSTRAINT      "nor"
#endif

// 定义探测时的空指令
#define TORCH_SDT_NOP                 nop

// 定义注记段的名称和类型
#define TORCH_SDT_NOTE_NAME           "stapsdt"
#define TORCH_SDT_NOTE_TYPE           3

// 信号量变量放置在此段中
#define TORCH_SDT_SEMAPHORE_SECTION   ".probes"

// 根据平台定义地址的大小
#ifdef __LP64__
#define TORCH_SDT_ASM_ADDR            .8byte
#else
#define TORCH_SDT_ASM_ADDR            .4byte
#endif

// 汇编器辅助宏定义
#define TORCH_SDT_S(x)                #x
#define TORCH_SDT_ASM_1(x)            TORCH_SDT_S(x) "\n"
#define TORCH_SDT_ASM_2(a, b)         TORCH_SDT_S(a) "," TORCH_SDT_S(b) "\n"
#define TORCH_SDT_ASM_3(a, b, c)      TORCH_SDT_S(a) "," TORCH_SDT_S(b) ","    \
                                      TORCH_SDT_S(c) "\n"
#define TORCH_SDT_ASM_STRING(x)       TORCH_SDT_ASM_1(.asciz TORCH_SDT_S(x))

// 判断参数是否是数组指针的辅助函数
#define TORCH_SDT_IS_ARRAY_POINTER(x)  ((__builtin_classify_type(x) == 14) ||  \
                                        (__builtin_classify_type(x) == 5))
// 获取参数的大小
#define TORCH_SDT_ARGSIZE(x)  (TORCH_SDT_IS_ARRAY_POINTER(x)                   \
                               ? sizeof(void*)                                 \
                               : sizeof(x))

// 定义探测参数的操作数格式
#define TORCH_SDT_ARG(n, x)                                                    \
  [TORCH_SDT_S##n] "n"                ((size_t)TORCH_SDT_ARGSIZE(x)),          \
  [TORCH_SDT_A##n] TORCH_SDT_ARG_CONSTRAINT (x)

// 定义添加操作数的模板
#define TORCH_SDT_OPERANDS_0()        [__sdt_dummy] "g" (0)
#define TORCH_SDT_OPERANDS_1(_1)      TORCH_SDT_ARG(1, _1)
#define TORCH_SDT_OPERANDS_2(_1, _2)                                           \
  TORCH_SDT_OPERANDS_1(_1), TORCH_SDT_ARG(2, _2)
#define TORCH_SDT_OPERANDS_3(_1, _2, _3)                                       \
  TORCH_SDT_OPERANDS_2(_1, _2), TORCH_SDT_ARG(3, _3)
#define TORCH_SDT_OPERANDS_4(_1, _2, _3, _4)                                   \
  TORCH_SDT_OPERANDS_3(_1, _2, _3), TORCH_SDT_ARG(4, _4)
#define TORCH_SDT_OPERANDS_5(_1, _2, _3, _4, _5)                               \
  TORCH_SDT_OPERANDS_4(_1, _2, _3, _4), TORCH_SDT_ARG(5, _5)
#define TORCH_SDT_OPERANDS_6(_1, _2, _3, _4, _5, _6)                           \
  TORCH_SDT_OPERANDS_5(_1, _2, _3, _4, _5), TORCH_SDT_ARG(6, _6)
#define TORCH_SDT_OPERANDS_7(_1, _2, _3, _4, _5, _6, _7)                       \
  TORCH_SDT_OPERANDS_6(_1, _2, _3, _4, _5, _6), TORCH_SDT_ARG(7, _7)
// 定义宏，接受 8 个操作数，将其转发到 TORCH_SDT_OPERANDS_7 宏，并添加第 8 个操作数作为 TORCH_SDT_ARG(8, _8)
#define TORCH_SDT_OPERANDS_8(_1, _2, _3, _4, _5, _6, _7, _8)                   \
  TORCH_SDT_OPERANDS_7(_1, _2, _3, _4, _5, _6, _7), TORCH_SDT_ARG(8, _8)
// 定义宏，接受 9 个操作数，将其转发到 TORCH_SDT_OPERANDS_8 宏，并添加第 9 个操作数作为 TORCH_SDT_ARG(9, _9)
#define TORCH_SDT_OPERANDS_9(_1, _2, _3, _4, _5, _6, _7, _8, _9)               \
  TORCH_SDT_OPERANDS_8(_1, _2, _3, _4, _5, _6, _7, _8), TORCH_SDT_ARG(9, _9)

// 定义字符串模板，用于从操作数中引用注释部分的参数
#define TORCH_SDT_ARGFMT(no)        %n[TORCH_SDT_S##no]@%[TORCH_SDT_A##no]
// 没有参数时的模板
#define TORCH_SDT_ARG_TEMPLATE_0    /*No arguments*/
// 1 个参数时的模板
#define TORCH_SDT_ARG_TEMPLATE_1    TORCH_SDT_ARGFMT(1)
// 2 个参数时的模板
#define TORCH_SDT_ARG_TEMPLATE_2    TORCH_SDT_ARG_TEMPLATE_1 TORCH_SDT_ARGFMT(2)
// 3 个参数时的模板
#define TORCH_SDT_ARG_TEMPLATE_3    TORCH_SDT_ARG_TEMPLATE_2 TORCH_SDT_ARGFMT(3)
// 4 个参数时的模板
#define TORCH_SDT_ARG_TEMPLATE_4    TORCH_SDT_ARG_TEMPLATE_3 TORCH_SDT_ARGFMT(4)
// 5 个参数时的模板
#define TORCH_SDT_ARG_TEMPLATE_5    TORCH_SDT_ARG_TEMPLATE_4 TORCH_SDT_ARGFMT(5)
// 6 个参数时的模板
#define TORCH_SDT_ARG_TEMPLATE_6    TORCH_SDT_ARG_TEMPLATE_5 TORCH_SDT_ARGFMT(6)
// 7 个参数时的模板
#define TORCH_SDT_ARG_TEMPLATE_7    TORCH_SDT_ARG_TEMPLATE_6 TORCH_SDT_ARGFMT(7)
// 8 个参数时的模板
#define TORCH_SDT_ARG_TEMPLATE_8    TORCH_SDT_ARG_TEMPLATE_7 TORCH_SDT_ARGFMT(8)
// 9 个参数时的模板
#define TORCH_SDT_ARG_TEMPLATE_9    TORCH_SDT_ARG_TEMPLATE_8 TORCH_SDT_ARGFMT(9)

// 定义信号量的名称，格式为 torch_sdt_semaphore_provider_name
#define TORCH_SDT_SEMAPHORE(provider, name)                                    \
  torch_sdt_semaphore_##provider##_##name

// 定义信号量，在 C 语言环境下声明一个未初始化的 volatile unsigned short 类型的全局变量
#define TORCH_SDT_DEFINE_SEMAPHORE(name)                                       \
  extern "C" {                                                                 \
    volatile unsigned short TORCH_SDT_SEMAPHORE(pytorch, name)                 \
    __attribute__((section(TORCH_SDT_SEMAPHORE_SECTION), used)) = 0;           \
  }

// 声明信号量，声明一个外部的 volatile unsigned short 类型的全局变量
#define TORCH_SDT_DECLARE_SEMAPHORE(name)                                      \
  extern "C" volatile unsigned short TORCH_SDT_SEMAPHORE(pytorch, name)

// 定义没有信号量的注释，使用 TORCH_SDT_ASM_1 宏，将地址 0 作为参数
#define TORCH_SDT_SEMAPHORE_NOTE_0(provider, name)                             \
  TORCH_SDT_ASM_1(     TORCH_SDT_ASM_ADDR 0) /*No Semaphore*/                  \

// 定义有信号量的注释，使用 TORCH_SDT_ASM_1 宏，将指定信号量的地址作为参数
#define TORCH_SDT_SEMAPHORE_NOTE_1(provider, name)                             \
  TORCH_SDT_ASM_1(TORCH_SDT_ASM_ADDR TORCH_SDT_SEMAPHORE(provider, name))
#define TORCH_SDT_NOTE_CONTENT(provider, name, has_semaphore, arg_template)    \
  TORCH_SDT_ASM_1(990: TORCH_SDT_NOP)                                          \
  // 在汇编代码中插入 NOP 操作，占位符
  TORCH_SDT_ASM_3(     .pushsection .note.stapsdt,"","note")                   \
  // 将汇编代码段推入 .note.stapsdt 符号表中
  TORCH_SDT_ASM_1(     .balign 4)                                              \
  // 对齐到 4 字节边界
  TORCH_SDT_ASM_3(     .4byte 992f-991f, 994f-993f, TORCH_SDT_NOTE_TYPE)       \
  // 添加 4 字节数据：(992f-991f), (994f-993f), TORCH_SDT_NOTE_TYPE
  TORCH_SDT_ASM_1(991: .asciz TORCH_SDT_NOTE_NAME)                             \
  // 定义标签 991，用于字符串 TORCH_SDT_NOTE_NAME
  TORCH_SDT_ASM_1(992: .balign 4)                                              \
  // 对齐到 4 字节边界
  TORCH_SDT_ASM_1(993: TORCH_SDT_ASM_ADDR 990b)                                \
  // 输出 990b 的地址作为一个地址值
  TORCH_SDT_ASM_1(     TORCH_SDT_ASM_ADDR 0) /*Reserved for Base Address*/     \
  // 保留基地址的空间，用于后续指定基地址
  TORCH_SDT_SEMAPHORE_NOTE_##has_semaphore(provider, name)                     \
  // 根据 has_semaphore 宏展开，处理信号量相关的注释内容
  TORCH_SDT_ASM_STRING(provider)                                               \
  // 将 provider 转换为字符串并插入汇编
  TORCH_SDT_ASM_STRING(name)                                                   \
  // 将 name 转换为字符串并插入汇编
  TORCH_SDT_ASM_STRING(arg_template)                                           \
  // 将 arg_template 转换为字符串并插入汇编
  TORCH_SDT_ASM_1(994: .balign 4)                                              \
  // 对齐到 4 字节边界
  TORCH_SDT_ASM_1(     .popsection)
  // 弹出 .note.stapsdt 符号表中的汇编代码段

// Main probe Macro.
#define TORCH_SDT_PROBE(provider, name, has_semaphore, n, arglist)             \
    __asm__ __volatile__ (                                                     \
      TORCH_SDT_NOTE_CONTENT(                                                  \
        provider, name, has_semaphore, TORCH_SDT_ARG_TEMPLATE_##n)             \
      :: TORCH_SDT_OPERANDS_##n arglist                                        \
    )                                                                          \
    // 使用内联汇编语法嵌入 TORCH_SDT_NOTE_CONTENT 生成的汇编代码，传入参数列表

// Helper Macros to handle variadic arguments.
#define TORCH_SDT_NARG_(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, N, ...) N
// 定义宏 TORCH_SDT_NARG_，用于计算参数个数 N
#define TORCH_SDT_NARG(...)                                                    \
  TORCH_SDT_NARG_(__VA_ARGS__, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
// 定义宏 TORCH_SDT_NARG，使用 TORCH_SDT_NARG_ 计算变参的个数

#define TORCH_SDT_PROBE_N(provider, name, has_semaphore, N, ...)               \
  TORCH_SDT_PROBE(provider, name, has_semaphore, N, (__VA_ARGS__))
// 定义宏 TORCH_SDT_PROBE_N，调用 TORCH_SDT_PROBE 宏并传递参数列表
```