# `.\numpy\numpy\_core\src\umath\fast_loop_macros.h`

```py
/**
 * 宏定义，用于构建快速的ufunc内部循环。
 *
 * 这些宏期望能够访问典型ufunc循环的参数，
 *
 *     char **args
 *     npy_intp const *dimensions
 *     npy_intp const *steps
 */
#ifndef _NPY_UMATH_FAST_LOOP_MACROS_H_
#define _NPY_UMATH_FAST_LOOP_MACROS_H_

#include <assert.h>

#include "simd/simd.h"

/*
 * numpy支持的最大SIMD向量大小（字节）
 * 目前是一个非常大的值，因为它仅用于内存重叠检查
 */
#if NPY_SIMD > 0
    // 足够用于编译器展开
    #define AUTOVEC_OVERLAP_SIZE NPY_SIMD_WIDTH*4
#else
    #define AUTOVEC_OVERLAP_SIZE 1024
#endif

/*
 * MAX_STEP_SIZE用于确定是否需要使用ufunc的SIMD版本。
 * 非常大的步长可能比使用标量处理还要慢。选择2097152（= 2MB）的值是基于两方面的考虑：
 * 1）典型的Linux内核页面大小为4KB，但有时也可能是2MB，与步长大小这么大相比，
 *    可能会导致16个不同页面上的所有加载/存储散射指令变慢。
 * 2）它还满足MAX_STEP_SIZE*16/esize < NPY_MAX_INT32的条件，这使得我们可以使用i32版本的gather/scatter
 *    （而不是i64版本），因为步长大于NPY_MAX_INT32*esize/16将需要使用i64gather/scatter。
 *    esize = 元素大小 = 浮点数/双精度数的4/8字节。
 */
#define MAX_STEP_SIZE 2097152

/**
 * 计算两个指针之间的绝对偏移量。
 *
 * @param a 指针a
 * @param b 指针b
 * @return 两个指针之间的绝对偏移量
 */
static inline npy_uintp
abs_ptrdiff(char *a, char *b)
{
    return (a > b) ? (a - b) : (b - a);
}

/**
 * 简单的未优化循环宏，以并行方式迭代ufunc参数。
 * @{
 */

/** (<ignored>) -> (op1) */
#define OUTPUT_LOOP\
    char *op1 = args[1];\
    npy_intp os1 = steps[1];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, op1 += os1)

/** (ip1) -> (op1) */
#define UNARY_LOOP\
    char *ip1 = args[0], *op1 = args[1];\
    npy_intp is1 = steps[0], os1 = steps[1];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += is1, op1 += os1)

/** (ip1) -> (op1, op2) */
#define UNARY_LOOP_TWO_OUT\
    char *ip1 = args[0], *op1 = args[1], *op2 = args[2];\
    npy_intp is1 = steps[0], os1 = steps[1], os2 = steps[2];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += is1, op1 += os1, op2 += os2)

#define BINARY_DEFS\
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];\
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];\
    npy_intp n = dimensions[0];\
    npy_intp i;\

#define BINARY_LOOP_SLIDING\
    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1)

/** (ip1, ip2) -> (op1) */
#define BINARY_LOOP\
    BINARY_DEFS\
    BINARY_LOOP_SLIDING

/** (ip1, ip2) -> (op1, op2) */
#define BINARY_LOOP_TWO_OUT\
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2], *op2 = args[3];\
    // 定义并初始化四个变量，分别表示输入步长和输出步长
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2], os2 = steps[3];
    // 获取维度数组的第一个元素，表示循环的次数
    npy_intp n = dimensions[0];
    // 定义循环变量 i，并进行循环操作
    npy_intp i;
    // 循环迭代 n 次，每次迭代更新输入指针 ip1 和 ip2，输出指针 op1 和 op2
    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1, op2 += os2)
/**
 * 定义一个三元循环宏，接受四个参数，分别是输入指针和步长，以及输出指针和步长。
 * 在循环中，使用指定的步长迭代输入指针，直到迭代完成所有给定的维度。
 */
#define TERNARY_LOOP\
    char *ip1 = args[0], *ip2 = args[1], *ip3 = args[2], *op1 = args[3];\
    npy_intp is1 = steps[0], is2 = steps[1], is3 = steps[2], os1 = steps[3];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, ip3 += is3, op1 += os1)

/** @} */

/**
 * 检查一元循环是否输入和输出是连续的。
 * 这里要求输入和输出的步长应该分别等于输入类型和输出类型的大小。
 */
#define IS_UNARY_CONT(tin, tout) (steps[0] == sizeof(tin) && \
                                  steps[1] == sizeof(tout))

/**
 * 检查输出是否是连续的。
 * 这里要求输出的步长应该等于输出类型的大小。
 */
#define IS_OUTPUT_CONT(tout) (steps[1] == sizeof(tout))

/*
 * 确保维度是非零的，使用断言来检查，以便后续代码可以忽略访问无效内存的问题。
 */
#define IS_BINARY_REDUCE (assert(dimensions[0] != 0), \
        (args[0] == args[2])\
        && (steps[0] == steps[2])\
        && (steps[0] == 0))

/**
 * 检查二元约简操作是否输入是连续的。
 * 这里要求第一个输入的步长应该等于第一个输入类型的大小。
 */
#define IS_BINARY_REDUCE_INPUT_CONT(tin) (assert(dimensions[0] != 0), \
         steps[1] == sizeof(tin))

/**
 * 检查二元循环是否输入和输出是连续的。
 * 这里要求第一个和第二个输入的步长应该等于输入类型的大小，输出的步长应该等于输出类型的大小。
 */
#define IS_BINARY_CONT(tin, tout) (steps[0] == sizeof(tin) && \
                                   steps[1] == sizeof(tin) && \
                                   steps[2] == sizeof(tout))

/**
 * 检查二元循环是否第一个输入是标量，而第二个输入和输出是连续的。
 * 这里要求第一个输入的步长为0，第二个输入的步长应该等于输入类型的大小，输出的步长应该等于输出类型的大小。
 */
#define IS_BINARY_CONT_S1(tin, tout) (steps[0] == 0 && \
                                   steps[1] == sizeof(tin) && \
                                   steps[2] == sizeof(tout))

/**
 * 检查二元循环是否第二个输入是标量，而第一个输入和输出是连续的。
 * 这里要求第一个输入的步长应该等于输入类型的大小，第二个输入的步长为0，输出的步长应该等于输出类型的大小。
 */
#define IS_BINARY_CONT_S2(tin, tout) (steps[0] == sizeof(tin) && \
                                   steps[1] == 0 && \
                                   steps[2] == sizeof(tout))

/*
 * 带有连续性特化的循环宏。
 * op 应该是处理输入类型为 `tin` 的 `in`，并将结果存储在 `tout *out` 中的代码。
 * 结合 NPY_GCC_OPT_3 可以允许自动向量化，应仅在值得避免代码膨胀时使用。
 */
#define BASE_UNARY_LOOP(tin, tout, op) \
    UNARY_LOOP { \
        const tin in = *(tin *)ip1; \
        tout *out = (tout *)op1; \
        op; \
    }

/**
 * 快速的一元循环宏。
 * 如果输入和输出是连续的，条件允许编译器优化通用宏。
 * 如果输入和输出是相同的，使用 BASE_UNARY_LOOP 处理。
 * 否则，使用 BASE_UNARY_LOOP 处理。
 */
#define UNARY_LOOP_FAST(tin, tout, op)          \
    do { \
        if (IS_UNARY_CONT(tin, tout)) { \
            if (args[0] == args[1]) { \
                BASE_UNARY_LOOP(tin, tout, op) \
            } \
            else { \
                BASE_UNARY_LOOP(tin, tout, op) \
            } \
        } \
        else { \
            BASE_UNARY_LOOP(tin, tout, op) \
        } \
    } \
    while (0)

/*
 * 带有连续性特化的循环宏。
 * op 应该是处理输入类型为 `tin` 的 `in1` 和 `in2`，并将结果存储在 `tout *out` 中的代码。
 * 结合 NPY_GCC_OPT_3 可以允许自动向量化，应仅在值得避免代码膨胀时使用。
 */
#define BASE_BINARY_LOOP(tin, tout, op) \
    BINARY_LOOP { \
        # 宏定义 BINARY_LOOP 的开始，这是一个多行宏
        const tin in1 = *(tin *)ip1; \
        # 定义并初始化变量 in1，类型为 tin，从指针 ip1 解引用得到值
        const tin in2 = *(tin *)ip2; \
        # 定义并初始化变量 in2，类型为 tin，从指针 ip2 解引用得到值
        tout *out = (tout *)op1; \
        # 定义输出指针 out，类型为 tout，指向 op1 所指向的地址
        op; \
        # 调用宏中的 op 操作，这里假设它会使用 in1、in2 和 out 进行某种运算
    }
/*
 * 定义了一个宏 `IVDEP_LOOP`，根据 GCC 编译器版本选择性地插入 GCC ivdep 声明，用于向编译器提示向量化优化信息。
 * 在 GCC 6 及以上版本中，使用 `_Pragma("GCC ivdep")` 实现向量化优化。
 * 在旧版本的 GCC 中，定义为空。
 */
#if __GNUC__ >= 6
#define IVDEP_LOOP _Pragma("GCC ivdep")
#else
#define IVDEP_LOOP
#endif

/*
 * 定义了一个宏 `BASE_BINARY_LOOP_INP`，用于执行基本的二进制操作循环，支持向量化优化。
 * 宏接受输入参数 `tin`（输入类型）、`tout`（输出类型）、`op`（操作），其中 `BINARY_DEFS` 是一个未定义的宏。
 * 使用 `IVDEP_LOOP` 声明以尝试向量化优化循环。
 * 循环遍历 `n` 次，依次处理 `ip1`、`ip2` 指针，以及 `op1` 指针，每次迭代更新指针位置。
 * 从 `ip1`、`ip2` 中读取 `tin` 类型的数据，并将结果存储为 `tout` 类型的数据。
 */
#define BASE_BINARY_LOOP_INP(tin, tout, op) \
    BINARY_DEFS\
    IVDEP_LOOP \
    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1) { \
        const tin in1 = *(tin *)ip1; \
        const tin in2 = *(tin *)ip2; \
        tout *out = (tout *)op1; \
        op; \
    }

/*
 * 定义了一个宏 `BASE_BINARY_LOOP_S`，用于执行基本的二进制操作循环，不支持向量化优化。
 * 宏接受输入参数 `tin`（输入类型）、`tout`（输出类型）、`cin`（常量输入）、`vin`（变量输入）、`op`（操作）。
 * 首先读取 `cin` 常量输入，然后执行循环处理 `vin` 变量输入。
 * 每次迭代更新 `op1` 指针，并从 `vinp` 中读取 `tin` 类型数据，最终将结果存储为 `tout` 类型数据。
 */
#define BASE_BINARY_LOOP_S(tin, tout, cin, cinp, vin, vinp, op) \
    const tin cin = *(tin *)cinp; \
    BINARY_LOOP { \
        const tin vin = *(tin *)vinp; \
        tout *out = (tout *)op1; \
        op; \
    }

/*
 * 定义了一个宏 `BASE_BINARY_LOOP_S_INP`，用于执行基本的二进制操作循环，不支持向量化优化。
 * 与 `BASE_BINARY_LOOP_S` 类似，不同之处在于输出的位置由 `vinp` 决定。
 * 每次迭代更新 `vinp` 指针，并从 `vinp` 中读取 `tin` 类型数据，最终将结果存储在 `vinp` 指向的位置。
 */
#define BASE_BINARY_LOOP_S_INP(tin, tout, cin, cinp, vin, vinp, op) \
    const tin cin = *(tin *)cinp; \
    BINARY_LOOP { \
        const tin vin = *(tin *)vinp; \
        tout *out = (tout *)vinp; \
        op; \
    }

/*
 * 定义了一个宏 `BINARY_LOOP_FAST`，用于根据输入的数据类型 `tin` 和 `tout` 执行二进制操作循环。
 * 宏接受输入参数 `tin`（输入类型）、`tout`（输出类型）、`op`（操作）。
 * 根据输入类型的连续性，选择最优的循环方式：向量化、标量操作或混合操作。
 * 如果输入类型 `tin` 和 `tout` 是连续的，则尝试使用 `BASE_BINARY_LOOP_INP` 进行向量化操作。
 * 如果 `tin` 和 `tout` 中只有一个是连续的，则使用 `BASE_BINARY_LOOP_S_INP` 或 `BASE_BINARY_LOOP_S` 进行操作。
 * 否则，使用普通的 `BASE_BINARY_LOOP` 进行操作。
 */
#define BINARY_LOOP_FAST(tin, tout, op)         \
    do { \
        /* condition allows compiler to optimize the generic macro */ \
        if (IS_BINARY_CONT(tin, tout)) { \
            if (abs_ptrdiff(args[2], args[0]) == 0 && \
                    abs_ptrdiff(args[2], args[1]) >= AUTOVEC_OVERLAP_SIZE) { \
                BASE_BINARY_LOOP_INP(tin, tout, op) \
            } \
            else if (abs_ptrdiff(args[2], args[1]) == 0 && \
                         abs_ptrdiff(args[2], args[0]) >= AUTOVEC_OVERLAP_SIZE) { \
                BASE_BINARY_LOOP_INP(tin, tout, op) \
            } \
            else { \
                BASE_BINARY_LOOP(tin, tout, op) \
            } \
        } \
        else if (IS_BINARY_CONT_S1(tin, tout)) { \
            if (abs_ptrdiff(args[2], args[1]) == 0) { \
                BASE_BINARY_LOOP_S_INP(tin, tout, in1, args[0], in2, ip2, op) \
            } \
            else { \
                BASE_BINARY_LOOP_S(tin, tout, in1, args[0], in2, ip2, op) \
            } \
        } \
        else if (IS_BINARY_CONT_S2(tin, tout)) { \
            if (abs_ptrdiff(args[2], args[0]) == 0) { \
                BASE_BINARY_LOOP_S_INP(tin, tout, in2, args[1], in1, ip1, op) \
            } \
            else { \
                BASE_BINARY_LOOP_S(tin, tout, in2, args[1], in1, ip1, op) \
            }\
        } \
        else { \
            BASE_BINARY_LOOP(tin, tout, op) \
        } \
    } \
    while (0)

/*
 * 定义了一个宏 `BINARY_REDUCE_LOOP_INNER`，用于执行二进制操作的内部循环。
 * 宏设置了 `ip2` 指向 `args[1]`，`is2` 为 `steps[1]`，`n` 为 `dimensions[0]`。
 * 循环处理 `n` 次，每次更新 `ip2` 指针位置。
 */
#define BINARY_REDUCE_LOOP_INNER\
    char *ip2 = args[1]; \
    npy_intp is2 = steps[1]; \
    npy_intp n = dimensions[0]; \
    npy_intp i; \
    for(i = 0; i < n; i++, ip2 += is2)

/*
 * 定义了一个宏 `BINARY_REDUCE_LOOP`，用于执行二进制操作的循环。
 * 宏设置了 `iop1` 指向 `args[0]`，并读取 `args[0]` 中的数据作为 `io1`。
 * 使用 `BINARY_REDUCE_LOOP_INNER` 执行二进制操作的内部循环。
 * 循环处理 `dimensions[0]` 次，每次更新 `iop1` 指针位置。
 */
#define BINARY_REDUCE_LOOP(TYPE)\
    char *iop1 = args[0]; \
    TYPE io1 = *(TYPE *)iop1; \
    BINARY_REDUCE_LOOP_INNER
/*
 * 定义一个基础的二元约简循环宏。
 * 宏接受两个参数：TYPE 是数据类型，op 是在循环内部执行的操作。
 * 在内部，从指针 ip2 中读取 TYPE 类型的数据，然后执行操作 op。
 */
#define BASE_BINARY_REDUCE_LOOP(TYPE, op) \
    BINARY_REDUCE_LOOP_INNER { \
        const TYPE in2 = *(TYPE *)ip2; \
        op; \
    }

/*
 * 定义一个快速版本的二元约简循环宏。
 * 宏接受两个参数：TYPE 是数据类型，op 是在循环内部执行的操作。
 * 根据 IS_BINARY_REDUCE_INPUT_CONT(TYPE) 的条件，选择性地调用 BASE_BINARY_REDUCE_LOOP 宏。
 */
#define BINARY_REDUCE_LOOP_FAST_INNER(TYPE, op)\
    /* 条件允许编译器优化通用宏 */ \
    if(IS_BINARY_REDUCE_INPUT_CONT(TYPE)) { \
        BASE_BINARY_REDUCE_LOOP(TYPE, op) \
    } \
    else { \
        BASE_BINARY_REDUCE_LOOP(TYPE, op) \
    }

/*
 * 定义一个快速版本的二元约简循环宏。
 * 宏接受两个参数：TYPE 是数据类型，op 是在循环内部执行的操作。
 * 使用 args 数组中的第一个元素作为指向数据的指针，执行循环内部操作，
 * 并在循环结束后将计算结果写回 args[0] 指向的位置。
 */
#define BINARY_REDUCE_LOOP_FAST(TYPE, op)\
    do { \
        char *iop1 = args[0]; \
        TYPE io1 = *(TYPE *)iop1; \
        BINARY_REDUCE_LOOP_FAST_INNER(TYPE, op); \
        *((TYPE *)iop1) = io1; \
    } \
    while (0)

/*
 * 检查步长是否为元素大小，并且输入和输出地址是否相等或者在寄存器内不重叠。
 * 通过检查 steps 数组元素与 esize 是否相等来保证 steps >= 0。
 */
#define IS_BINARY_STRIDE_ONE(esize, vsize) \
    ((steps[0] == esize) && \
     (steps[1] == esize) && \
     (steps[2] == esize) && \
     (abs_ptrdiff(args[2], args[0]) >= vsize) && \
     (abs_ptrdiff(args[2], args[1]) >= vsize))

/*
 * 检查是否可以对一元操作进行阻塞。
 * 检查条件包括：步长是否与元素大小相等、输入和输出地址是否对齐以及地址之间是否有足够的空间。
 */
#define IS_BLOCKABLE_UNARY(esize, vsize) \
    (steps[0] == (esize) && steps[0] == steps[1] && \
     (npy_is_aligned(args[0], esize) && npy_is_aligned(args[1], esize)) && \
     ((abs_ptrdiff(args[1], args[0]) >= (vsize)) || \
      ((abs_ptrdiff(args[1], args[0]) == 0))))

/*
 * 避免对于非常大的步长使用 SIMD 操作的宏定义。
 * 主要原因包括：
 * 1) 支持大步长需要使用 i64gather/scatter_ps 指令，性能不佳。
 * 2) 当加载/存储操作跨越页面边界时，使用 gather 和 scatter 指令会变慢。
 * 因此，只依赖于 i32gather/scatter_ps 指令，确保索引 < INT_MAX 以避免溢出。
 * 同时要求输入和输出数组在内存中不重叠。
 */
#define IS_BINARY_SMALL_STEPS_AND_NOMEMOVERLAP \
    ((labs(steps[0]) < MAX_STEP_SIZE)  && \
     (labs(steps[1]) < MAX_STEP_SIZE)  && \
     (labs(steps[2]) < MAX_STEP_SIZE)  && \
     (nomemoverlap(args[0], steps[0] * dimensions[0], args[2], steps[2] * dimensions[0])) && \
     (nomemoverlap(args[1], steps[1] * dimensions[0], args[2], steps[2] * dimensions[0])))

/*
 * 检查是否可以对两个输出的一元操作进行阻塞。
 * 条件包括：步长是否小于 MAX_STEP_SIZE、输入和输出地址是否在内存中不重叠。
 */
#define IS_UNARY_TWO_OUT_SMALL_STEPS_AND_NOMEMOVERLAP \
    ((labs(steps[0]) < MAX_STEP_SIZE)  && \
     (labs(steps[1]) < MAX_STEP_SIZE)  && \
     (labs(steps[2]) < MAX_STEP_SIZE)  && \
     (nomemoverlap(args[0], steps[0] * dimensions[0], args[2], steps[2] * dimensions[0])) && \
     (nomemoverlap(args[0], steps[0] * dimensions[0], args[1], steps[1] * dimensions[0])))
/*
 * 宏定义：检查是否可以对一元操作进行输出阻塞
 * 1) 第一个步长应该是输入元素大小的倍数，并且步长应小于最大步长以提高性能
 * 2) 输入和输出数组在内存中不应该有重叠
 */
#define IS_OUTPUT_BLOCKABLE_UNARY(esizein, esizeout, vsize) \
    ((steps[0] & (esizein-1)) == 0 && \
     steps[1] == (esizeout) && llabs(steps[0]) < MAX_STEP_SIZE && \
     (nomemoverlap(args[1], steps[1] * dimensions[0], args[0], steps[0] * dimensions[0])))

/*
 * 宏定义：检查是否可以对归约操作进行阻塞
 * 1) 第二个步长应该等于元素大小，并且输入和输出指针之间的距离应大于等于向量大小
 * 2) 输入和输出指针应该按元素大小对齐
 */
#define IS_BLOCKABLE_REDUCE(esize, vsize) \
    (steps[1] == (esize) && abs_ptrdiff(args[1], args[0]) >= (vsize) && \
     npy_is_aligned(args[1], (esize)) && \
     npy_is_aligned(args[0], (esize)))

/*
 * 宏定义：检查是否可以对二元操作进行阻塞
 * 1) 三个步长应该相等且等于元素大小，并且输入指针应该按元素大小对齐
 * 2) 输入指针之间的距离应大于等于向量大小或者为零
 */
#define IS_BLOCKABLE_BINARY(esize, vsize) \
    (steps[0] == steps[1] && steps[1] == steps[2] && steps[2] == (esize) && \
     npy_is_aligned(args[2], (esize)) && npy_is_aligned(args[1], (esize)) && \
     npy_is_aligned(args[0], (esize)) && \
     (abs_ptrdiff(args[2], args[0]) >= (vsize) || \
      abs_ptrdiff(args[2], args[0]) == 0) && \
     (abs_ptrdiff(args[2], args[1]) >= (vsize) || \
      abs_ptrdiff(args[2], args[1]) >= 0))

/*
 * 宏定义：检查是否可以对带有标量的二元操作进行阻塞（第一个标量）
 * 1) 第一个步长应为零，第二个和第三个步长应相等且等于元素大小，并且输入指针应按元素大小对齐
 * 2) 第一个标量指针与第二个标量指针之间的距离应大于等于向量大小，并且第一个标量指针与第三个指针之间的距离应大于等于元素大小
 */
#define IS_BLOCKABLE_BINARY_SCALAR1(esize, vsize) \
    (steps[0] == 0 && steps[1] == steps[2] && steps[2] == (esize) && \
     npy_is_aligned(args[2], (esize)) && npy_is_aligned(args[1], (esize)) && \
     ((abs_ptrdiff(args[2], args[1]) >= (vsize)) || \
      (abs_ptrdiff(args[2], args[1]) == 0)) && \
     abs_ptrdiff(args[2], args[0]) >= (esize))

/*
 * 宏定义：检查是否可以对带有标量的二元操作进行阻塞（第二个标量）
 * 1) 第二个步长应为零，第一个和第三个步长应相等且等于元素大小，并且输入指针应按元素大小对齐
 * 2) 第二个标量指针与第一个标量指针之间的距离应大于等于向量大小，并且第二个标量指针与第三个指针之间的距离应大于等于元素大小
 */
#define IS_BLOCKABLE_BINARY_SCALAR2(esize, vsize) \
    (steps[1] == 0 && steps[0] == steps[2] && steps[2] == (esize) && \
     npy_is_aligned(args[2], (esize)) && npy_is_aligned(args[0], (esize)) && \
     ((abs_ptrdiff(args[2], args[0]) >= (vsize)) || \
      (abs_ptrdiff(args[2], args[0]) == 0)) && \
     abs_ptrdiff(args[2], args[1]) >= (esize))

#undef abs_ptrdiff

/*
 * 宏定义：将变量按指定对齐方式对齐
 * 1) 使用 npy_aligned_block_offset 函数计算对齐偏移量 peel
 * 2) 循环从 peel 开始，直到满足对齐条件结束
 */
#define LOOP_BLOCK_ALIGN_VAR(var, type, alignment)\
    npy_intp i, peel = npy_aligned_block_offset(var, sizeof(type),\
                                                alignment, n);\
    for(i = 0; i < peel; i++)

/*
 * 宏定义：按块循环处理数组
 * 1) 循环处理被块分隔的数据块，每次增加一个块的大小
 */
#define LOOP_BLOCKED(type, vsize)\
    for(; i < npy_blocked_end(peel, sizeof(type), vsize, n);\
            i += (vsize / sizeof(type)))

/*
 * 宏定义：处理剩余的未被块处理的部分
 * 1) 处理剩余的不满足块处理条件的数据部分
 */
#define LOOP_BLOCKED_END\
    for (; i < n; i++)

#endif /* _NPY_UMATH_FAST_LOOP_MACROS_H_ */
```