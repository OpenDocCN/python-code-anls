# `.\numpy\numpy\_core\src\multiarray\dragon4.c`

```py
/*
 * 版权所有（c）2014 Ryan Juckett
 *
 * 根据以下条件，免费授予任何获得本软件及相关文档副本的人：
 * 无限制地处理本软件的权限，包括但不限于使用、复制、修改、合并、
 * 发布、分发、再许可和/或销售本软件的副本；以及
 * 允许使用本软件的人员，但需满足以下条件：
 *
 * 上述版权声明和本许可声明应包含在
 * 所有或实质部分的本软件副本中。
 *
 * 本软件按“原样”提供，无任何形式的明示或默示保证，
 * 包括但不限于适销性保证、特定用途的适用性和非侵权性保证。
 * 在任何情况下，作者或版权持有人均无法承担
 * 由于使用本软件或与其使用相关的其他操作而产生的任何索赔、损害或其他
 * 责任。
 */

/*
 * 本文件包含了 Ryan Juckett 的 Dragon4 实现的修改版本，
 * 从 https://www.ryanjuckett.com 获得，
 * 该版本已从 C++ 移植到 C，并具有特定于在 numpy 中打印浮点数的修改。
 *
 * Ryan Juckett 的原始代码使用 Zlib 许可证；他允许 numpy
 * 将其包含在 MIT 许可证下。
 */

#include "dragon4.h"
#include <numpy/npy_common.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <assert.h>

#if 0
#define DEBUG_ASSERT(stmnt) assert(stmnt)
#else
#define DEBUG_ASSERT(stmnt) do {} while(0)
#endif

/*
 * 返回一个 64 位无符号整数，其中低 n 位被置为 1。
 */
static inline npy_uint64
bitmask_u64(npy_uint32 n)
{
    return ~(~((npy_uint64)0) << n);
}

/*
 * 返回一个 32 位无符号整数，其中低 n 位被置为 1。
 */
static inline npy_uint32
bitmask_u32(npy_uint32 n)
{
    return ~(~((npy_uint32)0) << n);
}

/*
 * 获取 32 位无符号整数的以 2 为底的对数。
 * 参考：https://graphics.stanford.edu/~seander/bithacks.html#IntegerLogLookup
 */
static npy_uint32
LogBase2_32(npy_uint32 val)
{
    static const npy_uint8 logTable[256] =
    {
        0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
    };
    
    npy_uint32 temp;
    
    // 将 val 的高位向右移动 24 位，并且判断结果是否非零
    temp = val >> 24;
    if (temp) {
        // 如果 temp 非零，返回对应的 logTable 值加上 24
        return 24 + logTable[temp];
    }
    
    // 将 val 的高位向右移动 16 位，并且判断结果是否非零
    temp = val >> 16;
    if (temp) {
        // 如果 temp 非零，返回对应的 logTable 值加上 16
        return 16 + logTable[temp];
    }
    
    // 将 val 的高位向右移动 8 位，并且判断结果是否非零
    temp = val >> 8;
    if (temp) {
        // 如果 temp 非零，返回对应的 logTable 值加上 8
        return 8 + logTable[temp];
    }
    
    // 如果以上条件都不满足，则直接返回 logTable[val]
    return logTable[val];
}

static npy_uint32
LogBase2_64(npy_uint64 val)
{
    npy_uint64 temp;

    // 取输入值的高 32 位
    temp = val >> 32;
    // 如果高 32 位不为零，则返回 32 加上低 32 位的 LogBase2_32 值
    if (temp) {
        return 32 + LogBase2_32((npy_uint32)temp);
    }

    // 否则，返回低 32 位的 LogBase2_32 值
    return LogBase2_32((npy_uint32)val);
}

#if defined(HAVE_LDOUBLE_IEEE_QUAD_LE) || defined(HAVE_LDOUBLE_IEEE_QUAD_BE)
static npy_uint32
LogBase2_128(npy_uint64 hi, npy_uint64 lo)
{
    // 如果高位不为零，返回 64 加上高 64 位的 LogBase2_64 值
    if (hi) {
        return 64 + LogBase2_64(hi);
    }

    // 否则，返回低 64 位的 LogBase2_64 值
    return LogBase2_64(lo);
}
#endif /* HAVE_LDOUBLE_IEEE_QUAD_LE */

/*
 * Maximum number of 32 bit blocks needed in high precision arithmetic to print
 * out 128 bit IEEE floating point values. 1023 chosen to be large enough for
 * 128 bit floats, and BigInt is exactly 4kb (nice for page/cache?)
 */
// 定义在高精度算术中打印 128 位 IEEE 浮点值所需的最大 32 位块数
#define c_BigInt_MaxBlocks  1023

/*
 * This structure stores a high precision unsigned integer. It uses a buffer of
 * 32 bit integer blocks along with a length. The lowest bits of the integer
 * are stored at the start of the buffer and the length is set to the minimum
 * value that contains the integer. Thus, there are never any zero blocks at
 * the end of the buffer.
 */
// 定义存储高精度无符号整数的结构体 BigInt
typedef struct BigInt {
    npy_uint32 length;  // 整数的长度
    npy_uint32 blocks[c_BigInt_MaxBlocks];  // 32 位整数块的数组
} BigInt;

/*
 * Dummy implementation of a memory manager for BigInts. Currently, only
 * supports a single call to Dragon4, but that is OK because Dragon4
 * does not release the GIL.
 *
 * We try to raise an error anyway if dragon4 re-enters, and this code serves
 * as a placeholder if we want to make it re-entrant in the future.
 *
 * Each call to dragon4 uses 7 BigInts.
 */
// 用于 BigInt 的内存管理器的虚拟实现。当前仅支持 Dragon4 的单次调用。
// Dragon4 不释放 GIL，因此这是可以的。
// 如果 dragon4 重入，我们尝试引发错误，这段代码在未来可能变为可重入的占位符。
// 每次 dragon4 调用使用 7 个 BigInt。
#define BIGINT_DRAGON4_GROUPSIZE 7
typedef struct {
    BigInt bigints[BIGINT_DRAGON4_GROUPSIZE];  // 用于 Dragon4 的 BigInt 数组
    char repr[16384];  // 字符串表示形式的缓冲区
} Dragon4_Scratch;

static int _bigint_static_in_use = 0;  // 静态 BigInt 使用标志
static Dragon4_Scratch _bigint_static;  // 静态 Dragon4_Scratch 实例

static Dragon4_Scratch*
get_dragon4_bigint_scratch(void) {
    // 这个测试和设置不是线程安全的，但由于有 GIL，不会有问题
    if (_bigint_static_in_use) {
        PyErr_SetString(PyExc_RuntimeError,
            "numpy float printing code is not re-entrant. "
            "Ping the devs to fix it.");
        return NULL;
    }
    _bigint_static_in_use = 1;

    // 在这个虚拟实现中，我们只返回静态分配
    return &_bigint_static;
}

static void
free_dragon4_bigint_scratch(Dragon4_Scratch *mem){
    _bigint_static_in_use = 0;
}

/* Copy integer */
static void
BigInt_Copy(BigInt *dst, const BigInt *src)
{
    npy_uint32 length = src->length;  // 获取源 BigInt 的长度
    npy_uint32 * dstp = dst->blocks;  // 目标 BigInt 的块指针
    const npy_uint32 *srcp;  // 源 BigInt 的块指针
    // 循环复制源 BigInt 的每个块到目标 BigInt
    for (srcp = src->blocks; srcp != src->blocks + length; ++dstp, ++srcp) {
        *dstp = *srcp;
    }
    dst->length = length;  // 设置目标 BigInt 的长度
}

/* Basic type accessors */
static void
BigInt_Set_uint64(BigInt *i, npy_uint64 val)
{
    // 如果输入值大于 32 位掩码，则分别设置低 32 位和高 32 位块，并设置长度为 2
    if (val > bitmask_u64(32)) {
        i->blocks[0] = val & bitmask_u64(32);
        i->blocks[1] = (val >> 32) & bitmask_u64(32);
        i->length = 2;
    }
}
    else if (val != 0) {
        # 如果 val 不等于 0，则进入这个条件分支
        i->blocks[0] = val & bitmask_u64(32);
        # 将 val 和一个 32 位的掩码按位与，将结果存入 i 对象的 blocks 数组的第一个元素中
        i->length = 1;
        # 设置 i 对象的 length 属性为 1
    }
    else {
        # 如果 val 等于 0，则进入这个条件分支
        i->length = 0;
        # 设置 i 对象的 length 属性为 0
    }
}

#if (defined(HAVE_LDOUBLE_IBM_DOUBLE_DOUBLE_LE) || \
     defined(HAVE_LDOUBLE_IBM_DOUBLE_DOUBLE_BE) || \
     defined(HAVE_LDOUBLE_IEEE_QUAD_LE) || \
     defined(HAVE_LDOUBLE_IEEE_QUAD_BE))
static void
BigInt_Set_2x_uint64(BigInt *i, npy_uint64 hi, npy_uint64 lo)
{
    // 根据 hi 和 lo 的值设置 BigInt 的长度
    if (hi > bitmask_u64(32)) {
        i->length = 4;
    }
    else if (hi != 0) {
        i->length = 3;
    }
    else if (lo > bitmask_u64(32)) {
        i->length = 2;
    }
    else if (lo != 0) {
        i->length = 1;
    }
    else {
        i->length = 0;
    }

    /* Note deliberate fallthrough in this switch */
    // 根据 BigInt 的长度设置对应的 blocks 数组值
    switch (i->length) {
        case 4:
            i->blocks[3] = (hi >> 32) & bitmask_u64(32);
        case 3:
            i->blocks[2] = hi & bitmask_u64(32);
        case 2:
            i->blocks[1] = (lo >> 32) & bitmask_u64(32);
        case 1:
            i->blocks[0] = lo & bitmask_u64(32);
    }
}
#endif /* DOUBLE_DOUBLE and QUAD */

static void
BigInt_Set_uint32(BigInt *i, npy_uint32 val)
{
    // 设置 BigInt 为一个 32 位无符号整数
    if (val != 0) {
        i->blocks[0] = val;
        i->length = 1;
    }
    else {
        i->length = 0;
    }
}

/*
 * Returns 1 if the value is zero
 */
static int
BigInt_IsZero(const BigInt *i)
{
    // 判断 BigInt 是否为零
    return i->length == 0;
}

/*
 * Returns 1 if the value is even
 */
static int
BigInt_IsEven(const BigInt *i)
{
    // 判断 BigInt 是否为偶数
    return (i->length == 0) || ((i->blocks[0] % 2) == 0);
}

/*
 * Returns 0 if (lhs = rhs), negative if (lhs < rhs), positive if (lhs > rhs)
 */
static npy_int32
BigInt_Compare(const BigInt *lhs, const BigInt *rhs)
{
    int i;

    // 比较两个 BigInt 的大小
    /* A bigger length implies a bigger number. */
    npy_int32 lengthDiff = lhs->length - rhs->length;
    if (lengthDiff != 0) {
        return lengthDiff;
    }

    /* Compare blocks one by one from high to low. */
    // 逐个比较 blocks 数组中的值，从高位到低位
    for (i = lhs->length - 1; i >= 0; --i) {
        if (lhs->blocks[i] == rhs->blocks[i]) {
            continue;
        }
        else if (lhs->blocks[i] > rhs->blocks[i]) {
            return 1;
        }
        else {
            return -1;
        }
    }

    /* no blocks differed */
    // 没有发现不同的 blocks，说明两个数相等
    return 0;
}

/* result = lhs + rhs */
static void
BigInt_Add(BigInt *result, const BigInt *lhs, const BigInt *rhs)
{
    /* determine which operand has the smaller length */
    // 确定哪个操作数长度较小
    const BigInt *large, *small;
    npy_uint64 carry = 0;
    const npy_uint32 *largeCur, *smallCur, *largeEnd, *smallEnd;
    npy_uint32 *resultCur;

    if (lhs->length < rhs->length) {
        small = lhs;
        large = rhs;
    }
    else {
        small = rhs;
        large = lhs;
    }

    /* The output will be at least as long as the largest input */
    // 结果的长度至少与较大输入的长度一样长
    result->length = large->length;

    /* Add each block and add carry the overflow to the next block */
    // 每个块相加，并将溢出的 carry 添加到下一个块中
    largeCur  = large->blocks;
    largeEnd  = largeCur + large->length;
    smallCur  = small->blocks;
    smallEnd  = smallCur + small->length;
    resultCur = result->blocks;
}
    # 当小操作数未遍历完时继续执行循环
    while (smallCur != smallEnd) {
        # 计算当前位置的和，并考虑前一个位置的进位
        npy_uint64 sum = carry + (npy_uint64)(*largeCur) +
                                 (npy_uint64)(*smallCur);
        carry = sum >> 32;  # 更新进位，即将高 32 位的内容移到低 32 位
        *resultCur = sum & bitmask_u64(32);  # 将低 32 位存入结果数组中
        ++largeCur;  # 移动到大操作数的下一个位置
        ++smallCur;  # 移动到小操作数的下一个位置
        ++resultCur;  # 移动到结果数组的下一个位置
    }

    # 处理仅在大操作数中存在的块，将进位加到这些块上
    while (largeCur != largeEnd) {
        npy_uint64 sum = carry + (npy_uint64)(*largeCur);
        carry = sum >> 32;  # 更新进位
        (*resultCur) = sum & bitmask_u64(32);  # 将低 32 位存入结果数组中
        ++largeCur;  # 移动到大操作数的下一个位置
        ++resultCur;  # 移动到结果数组的下一个位置
    }

    # 如果还有进位，追加一个新的块到结果数组中
    if (carry != 0) {
        DEBUG_ASSERT(carry == 1);  # 断言进位值为1
        DEBUG_ASSERT((npy_uint32)(resultCur - result->blocks) ==
               large->length && (large->length < c_BigInt_MaxBlocks));  # 断言结果长度正确
        *resultCur = 1;  # 将进位值写入结果数组
        result->length = large->length + 1;  # 更新结果的长度
    }
    else {
        result->length = large->length;  # 结果长度为大操作数的长度
    }
/*
 * result = lhs * rhs
 */
static void
BigInt_Multiply(BigInt *result, const BigInt *lhs, const BigInt *rhs)
{
    const BigInt *large;    // 指向较大的 BigInt 结构体
    const BigInt *small;    // 指向较小的 BigInt 结构体
    npy_uint32 maxResultLen;    // 结果数组的最大长度
    npy_uint32 *cur, *end, *resultStart;    // 当前操作的指针和结束指针
    const npy_uint32 *smallCur;    // 指向小数位数组的当前指针

    DEBUG_ASSERT(result != lhs && result != rhs);    // 断言结果不等于乘数和被乘数

    /* determine which operand has the smaller length */
    if (lhs->length < rhs->length) {
        small = lhs;    // 左操作数较短
        large = rhs;    // 右操作数较长
    }
    else {
        small = rhs;    // 右操作数较短
        large = lhs;    // 左操作数较长
    }

    /* set the maximum possible result length */
    maxResultLen = large->length + small->length;    // 计算结果数组的最大长度
    DEBUG_ASSERT(maxResultLen <= c_BigInt_MaxBlocks);    // 断言结果长度不超过最大允许长度

    /* clear the result data */
    for (cur = result->blocks, end = cur + maxResultLen; cur != end; ++cur) {
        *cur = 0;    // 将结果数组清零
    }

    /* perform standard long multiplication for each small block */
    resultStart = result->blocks;    // 结果数组的起始位置
    for (smallCur = small->blocks;
            smallCur != small->blocks + small->length;
            ++smallCur, ++resultStart) {
        /*
         * if non-zero, multiply against all the large blocks and add into the
         * result
         */
        const npy_uint32 multiplier = *smallCur;    // 当前小数位的值作为乘数
        if (multiplier != 0) {    // 如果乘数不为零
            const npy_uint32 *largeCur = large->blocks;    // 大数位的起始位置
            npy_uint32 *resultCur = resultStart;    // 结果数组的当前位置
            npy_uint64 carry = 0;    // 进位初始化为零
            do {
                npy_uint64 product = (*resultCur) +
                                     (*largeCur)*(npy_uint64)multiplier + carry;    // 计算乘法并加上进位
                carry = product >> 32;    // 计算新的进位
                *resultCur = product & bitmask_u64(32);    // 将低32位作为新的结果
                ++largeCur;    // 移动到下一个大数位
                ++resultCur;    // 移动到下一个结果位置
            } while(largeCur != large->blocks + large->length);

            DEBUG_ASSERT(resultCur < result->blocks + maxResultLen);    // 断言结果位置不超过最大允许长度
            *resultCur = (npy_uint32)(carry & bitmask_u64(32));    // 将最终进位写入结果数组
        }
    }

    /* check if the terminating block has no set bits */
    if (maxResultLen > 0 && result->blocks[maxResultLen - 1] == 0) {
        result->length = maxResultLen - 1;    // 如果最高位为零，则结果长度减一
    }
    else {
        result->length = maxResultLen;    // 否则结果长度为计算得到的最大长度
    }
}

/* result = lhs * rhs */
static void
BigInt_Multiply_int(BigInt *result, const BigInt *lhs, npy_uint32 rhs)
{
    /* perform long multiplication */
    npy_uint32 carry = 0;    // 进位初始化为零
    npy_uint32 *resultCur = result->blocks;    // 结果数组的当前位置
    const npy_uint32 *pLhsCur = lhs->blocks;    // 左操作数的当前位置
    const npy_uint32 *pLhsEnd = lhs->blocks + lhs->length;    // 左操作数的结束位置
    for (; pLhsCur != pLhsEnd; ++pLhsCur, ++resultCur) {
        npy_uint64 product = (npy_uint64)(*pLhsCur) * rhs + carry;    // 计算乘法并加上进位
        *resultCur = (npy_uint32)(product & bitmask_u64(32));    // 将低32位作为新的结果
        carry = product >> 32;    // 计算新的进位
    }

    /* if there is a remaining carry, grow the array */
    if (carry != 0) {    // 如果还有进位
        /* grow the array */
        DEBUG_ASSERT(lhs->length + 1 <= c_BigInt_MaxBlocks);    // 断言结果长度加一不超过最大允许长度
        *resultCur = (npy_uint32)carry;    // 将最终进位写入结果数组
        result->length = lhs->length + 1;    // 结果长度加一
    }
    else {
        result->length = lhs->length;    // 结果长度保持不变
    }
}
/* result = in * 2 */
static void
BigInt_Multiply2(BigInt *result, const BigInt *in)
{
    /* shift all the blocks by one */
    npy_uint32 carry = 0;

    npy_uint32 *resultCur = result->blocks;   /* 指向结果 BigInt 结构体中的数据块 */
    const npy_uint32 *pLhsCur = in->blocks;   /* 指向输入 BigInt 结构体中的数据块 */
    const npy_uint32 *pLhsEnd = in->blocks + in->length;   /* 指向输入 BigInt 结构体数据块末尾的下一个位置 */
    for ( ; pLhsCur != pLhsEnd; ++pLhsCur, ++resultCur) {
        npy_uint32 cur = *pLhsCur;   /* 当前输入数据块的值 */
        *resultCur = (cur << 1) | carry;   /* 将当前输入数据块的值左移一位，并加上进位，存入结果数据块中 */
        carry = cur >> 31;   /* 计算当前输入数据块的最高位作为下一次的进位 */
    }

    if (carry != 0) {
        /* grow the array */
        DEBUG_ASSERT(in->length + 1 <= c_BigInt_MaxBlocks);   /* 断言，确保结果数组不会超过最大块数 */
        *resultCur = carry;   /* 如果最高位有进位，则将进位加入到结果数组的最后一块 */
        result->length = in->length + 1;   /* 更新结果 BigInt 结构体的长度 */
    }
    else {
        result->length = in->length;   /* 没有进位，则结果长度与输入相同 */
    }
}

/* result = result * 2 */
static void
BigInt_Multiply2_inplace(BigInt *result)
{
    /* shift all the blocks by one */
    npy_uint32 carry = 0;

    npy_uint32 *cur = result->blocks;   /* 指向结果 BigInt 结构体中的数据块 */
    npy_uint32 *end = result->blocks + result->length;   /* 指向结果 BigInt 结构体数据块末尾的下一个位置 */
    for ( ; cur != end; ++cur) {
        npy_uint32 tmpcur = *cur;   /* 当前结果数据块的值 */
        *cur = (tmpcur << 1) | carry;   /* 将当前结果数据块的值左移一位，并加上进位，存回结果数据块中 */
        carry = tmpcur >> 31;   /* 计算当前结果数据块的最高位作为下一次的进位 */
    }

    if (carry != 0) {
        /* grow the array */
        DEBUG_ASSERT(result->length + 1 <= c_BigInt_MaxBlocks);   /* 断言，确保结果数组不会超过最大块数 */
        *cur = carry;   /* 如果最高位有进位，则将进位加入到结果数组的最后一块 */
        ++result->length;   /* 更新结果 BigInt 结构体的长度 */
    }
}

/* result = result * 10 */
static void
BigInt_Multiply10(BigInt *result)
{
    /* multiply all the blocks */
    npy_uint64 carry = 0;

    npy_uint32 *cur = result->blocks;   /* 指向结果 BigInt 结构体中的数据块 */
    npy_uint32 *end = result->blocks + result->length;   /* 指向结果 BigInt 结构体数据块末尾的下一个位置 */
    for ( ; cur != end; ++cur) {
        npy_uint64 product = (npy_uint64)(*cur) * 10ull + carry;   /* 计算当前结果数据块乘以 10 后的结果，并加上进位 */
        (*cur) = (npy_uint32)(product & bitmask_u64(32));   /* 将结果截断为 32 位，并存回当前结果数据块中 */
        carry = product >> 32;   /* 计算除去 32 位后的进位 */
    }

    if (carry != 0) {
        /* grow the array */
        DEBUG_ASSERT(result->length + 1 <= c_BigInt_MaxBlocks);   /* 断言，确保结果数组不会超过最大块数 */
        *cur = (npy_uint32)carry;   /* 如果还有进位，则将进位加入到结果数组的最后一块 */
        ++result->length;   /* 更新结果 BigInt 结构体的长度 */
    }
}

static npy_uint32 g_PowerOf10_U32[] =
{
    1,          /* 10 ^ 0 */
    10,         /* 10 ^ 1 */
    100,        /* 10 ^ 2 */
    1000,       /* 10 ^ 3 */
    10000,      /* 10 ^ 4 */
    100000,     /* 10 ^ 5 */
    1000000,    /* 10 ^ 6 */
    10000000,   /* 10 ^ 7 */
};

/*
 * Note: This has a lot of wasted space in the big integer structures of the
 *       early table entries. It wouldn't be terribly hard to make the multiply
 *       function work on integer pointers with an array length instead of
 *       the BigInt struct which would allow us to store a minimal amount of
 *       data here.
 */
static BigInt g_PowerOf10_Big[] =
{
    /* 10 ^ 8 */
    { 1, { 100000000 } },   /* 以 BigInt 结构体表示的 10^8 */

    /* 10 ^ 16 */
    { 2, { 0x6fc10000, 0x002386f2 } },   /* 以 BigInt 结构体表示的 10^16 */

    /* 10 ^ 32 */
    { 4, { 0x00000000, 0x85acef81, 0x2d6d415b, 0x000004ee, } },   /* 以 BigInt 结构体表示的 10^32 */

    /* 10 ^ 64 */
    { 7, { 0x00000000, 0x00000000, 0xbf6a1f01, 0x6e38ed64, 0xdaa797ed,
           0xe93ff9f4, 0x00184f03, } },   /* 以 BigInt 结构体表示的 10^64 */
};
    { 14, { 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x2e953e01,
            0x03df9909, 0x0f1538fd, 0x2374e42f, 0xd3cff5ec, 0xc404dc08,
            0xbccdb0da, 0xa6337f19, 0xe91f2603, 0x0000024e, } },
    /* 10 ^ 256 */
    { 27, { 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x982e7c01, 0xbed3875b,
            0xd8d99f72, 0x12152f87, 0x6bde50c6, 0xcf4a6e70, 0xd595d80f,
            0x26b2716e, 0xadc666b0, 0x1d153624, 0x3c42d35a, 0x63ff540e,
            0xcc5573c0, 0x65f9ef17, 0x55bc28f2, 0x80dcc7f7, 0xf46eeddc,
            0x5fdcefce, 0x000553f7, } },
    /* 10 ^ 512 */
    { 54, { 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0xfc6cf801, 0x77f27267, 0x8f9546dc, 0x5d96976f,
            0xb83a8a97, 0xc31e1ad9, 0x46c40513, 0x94e65747, 0xc88976c1,
            0x4475b579, 0x28f8733b, 0xaa1da1bf, 0x703ed321, 0x1e25cfea,
            0xb21a2f22, 0xbc51fb2e, 0x96e14f5d, 0xbfa3edac, 0x329c57ae,
            0xe7fc7153, 0xc3fc0695, 0x85a91924, 0xf95f635e, 0xb2908ee0,
            0x93abade4, 0x1366732a, 0x9449775c, 0x69be5b0e, 0x7343afac,
            0xb099bc81, 0x45a71d46, 0xa2699748, 0x8cb07303, 0x8a0b1f13,
            0x8cab8a97, 0xc1d238d9, 0x633415d4, 0x0000001c, } },
    /* 10 ^ 1024 */


注释：
这段代码定义了三个数据结构，每个结构包含一个整数和一个数组。每个整数注释描述了它所代表的幂次方，而数组则包含了十进制数 10 的该幂次方的大整数表示。
    { 107, { 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x2919f001, 0xf55b2b72, 0x6e7c215b,
             0x1ec29f86, 0x991c4e87, 0x15c51a88, 0x140ac535, 0x4c7d1e1a,
             0xcc2cd819, 0x0ed1440e, 0x896634ee, 0x7de16cfb, 0x1e43f61f,
             0x9fce837d, 0x231d2b9c, 0x233e55c7, 0x65dc60d7, 0xf451218b,
             0x1c5cd134, 0xc9635986, 0x922bbb9f, 0xa7e89431, 0x9f9f2a07,
             0x62be695a, 0x8e1042c4, 0x045b7a74, 0x1abe1de3, 0x8ad822a5,
             0xba34c411, 0xd814b505, 0xbf3fdeb3, 0x8fc51a16, 0xb1b896bc,
             0xf56deeec, 0x31fb6bfd, 0xb6f4654b, 0x101a3616, 0x6b7595fb,
             0xdc1a47fe, 0x80d98089, 0x80bda5a5, 0x9a202882, 0x31eb0f66,
             0xfc8f1f90, 0x976a3310, 0xe26a7b7e, 0xdf68368a, 0x3ce3a0b8,
             0x8e4262ce, 0x75a351a2, 0x6cb0b6c9, 0x44597583, 0x31b5653f,
             0xc356e38a, 0x35faaba6, 0x0190fba0, 0x9fc4ed52, 0x88bc491b,
             0x1640114a, 0x005b8041, 0xf4f3235e, 0x1e8d4649, 0x36a8de06,
             0x73c55349, 0xa7e6bd2a, 0xc1a6970c, 0x47187094, 0xd2db49ef,
             0x926c3f5b, 0xae6209d4, 0x2d433949, 0x34f4a3c6, 0xd4305d94,
             0xd9d61a05, 0x00000325, } },
    /* 10 ^ 2048 */
    /* 10 ^ 4096 */
    
    
    注释：
};

/* result = 10^exponent */
static void
BigInt_Pow10(BigInt *result, npy_uint32 exponent, BigInt *temp)
{
    /* use two temporary values to reduce large integer copy operations */
    BigInt *curTemp = result;  // 当前临时变量指向结果
    BigInt *pNextTemp = temp;  // 下一个临时变量指向传入的临时变量
    npy_uint32 smallExponent;  // 小指数值
    npy_uint32 tableIdx = 0;   // 表索引

    /* make sure the exponent is within the bounds of the lookup table data */
    DEBUG_ASSERT(exponent < 8192);  // 断言，确保指数在查找表数据范围内

    /*
     * initialize the result by looking up a 32-bit power of 10 corresponding to
     * the first 3 bits
     */
    smallExponent = exponent & bitmask_u32(3);  // 取指数的低3位作为小指数
    BigInt_Set_uint32(curTemp, g_PowerOf10_U32[smallExponent]);  // 使用32位整数表中的值初始化结果

    /* remove the low bits that we used for the 32-bit lookup table */
    exponent >>= 3;  // 右移3位，去除已经使用过的低3位

    /* while there are remaining bits in the exponent to be processed */
    while (exponent != 0) {
        /* if the current bit is set, multiply by this power of 10 */
        if (exponent & 1) {
            BigInt *pSwap;

            /* multiply into the next temporary */
            BigInt_Multiply(pNextTemp, curTemp, &g_PowerOf10_Big[tableIdx]);

            /* swap to the next temporary */
            pSwap = curTemp;
            curTemp = pNextTemp;
            pNextTemp = pSwap;
        }

        /* advance to the next bit */
        ++tableIdx;
        exponent >>= 1;  // 右移一位，处理下一位指数
    }

    /* output the result */
    if (curTemp != result) {
        BigInt_Copy(result, curTemp);  // 如果当前临时变量不是结果，则拷贝到结果
    }
}

/* in = in * 10^exponent */
static void
BigInt_MultiplyPow10(BigInt *in, npy_uint32 exponent, BigInt *temp)
{
    /* use two temporary values to reduce large integer copy operations */
    BigInt *curTemp, *pNextTemp;
    npy_uint32 smallExponent;
    npy_uint32 tableIdx = 0;

    /* make sure the exponent is within the bounds of the lookup table data */
    DEBUG_ASSERT(exponent < 8192);  // 断言，确保指数在查找表数据范围内

    /*
     * initialize the result by looking up a 32-bit power of 10 corresponding to
     * the first 3 bits
     */
    smallExponent = exponent & bitmask_u32(3);  // 取指数的低3位作为小指数
    if (smallExponent != 0) {
        BigInt_Multiply_int(temp, in, g_PowerOf10_U32[smallExponent]);  // 使用32位整数表中的值乘以输入值，并存入临时变量
        curTemp = temp;
        pNextTemp = in;
    }
    else {
        curTemp = in;
        pNextTemp = temp;
    }

    /* remove the low bits that we used for the 32-bit lookup table */
    exponent >>= 3;  // 右移3位，去除已经使用过的低3位

    /* while there are remaining bits in the exponent to be processed */
    while (exponent != 0) {
        /* if the current bit is set, multiply by this power of 10 */
        if (exponent & 1) {
            BigInt *pSwap;

            /* multiply into the next temporary */
            BigInt_Multiply(pNextTemp, curTemp, &g_PowerOf10_Big[tableIdx]);

            /* swap to the next temporary */
            pSwap = curTemp;
            curTemp = pNextTemp;
            pNextTemp = pSwap;
        }

        /* advance to the next bit */
        ++tableIdx;
        exponent >>= 1;  // 右移一位，处理下一位指数
    }

    /* output the result */
    if (curTemp != in){
        BigInt_Copy(in, curTemp);  // 如果当前临时变量不是输入值，则拷贝到输入值
    }
}
/*
 * result = 2^exponent
 * Computes the power of 2 raised to the exponent for a given BigInt.
 */
static inline void
BigInt_Pow2(BigInt *result, npy_uint32 exponent)
{
    npy_uint32 bitIdx;
    npy_uint32 blockIdx = exponent / 32; // Determine the block index based on exponent
    npy_uint32 i;

    DEBUG_ASSERT(blockIdx < c_BigInt_MaxBlocks); // Assert that blockIdx is within valid range

    // Initialize blocks up to blockIdx to 0
    for (i = 0; i <= blockIdx; ++i) {
        result->blocks[i] = 0;
    }

    result->length = blockIdx + 1; // Set the length of the result BigInt

    bitIdx = (exponent % 32); // Determine the bit index within the last block
    result->blocks[blockIdx] |= ((npy_uint32)1 << bitIdx); // Set the corresponding bit in the last block
}

/*
 * This function will divide two large numbers under the assumption that the
 * result is within the range [0,10) and the input numbers have been shifted
 * to satisfy:
 * - The highest block of the divisor is greater than or equal to 8 such that
 *   there is enough precision to make an accurate first guess at the quotient.
 * - The highest block of the divisor is less than the maximum value on an
 *   unsigned 32-bit integer such that we can safely increment without overflow.
 * - The dividend does not contain more blocks than the divisor such that we
 *   can estimate the quotient by dividing the equivalently placed high blocks.
 *
 * quotient  = floor(dividend / divisor)
 * remainder = dividend - quotient*divisor
 *
 * dividend is updated to be the remainder and the quotient is returned.
 */
static npy_uint32
BigInt_DivideWithRemainder_MaxQuotient9(BigInt *dividend, const BigInt *divisor)
{
    npy_uint32 length, quotient;
    const npy_uint32 *finalDivisorBlock;
    npy_uint32 *finalDividendBlock;

    /*
     * Check that the divisor has been correctly shifted into range and that it
     * is not smaller than the dividend in length.
     */
    DEBUG_ASSERT(!divisor->length == 0 &&
                divisor->blocks[divisor->length-1] >= 8 &&
                divisor->blocks[divisor->length-1] < bitmask_u64(32) &&
                dividend->length <= divisor->length);

    /*
     * If the dividend is smaller than the divisor, the quotient is zero and the
     * divisor is already the remainder.
     */
    length = divisor->length;
    if (dividend->length < divisor->length) {
        return 0; // Return zero if dividend is smaller than divisor
    }

    finalDivisorBlock = divisor->blocks + length - 1;
    finalDividendBlock = dividend->blocks + length - 1;

    /*
     * Compute an estimated quotient based on the high block value. This will
     * either match the actual quotient or undershoot by one.
     */
    quotient = *finalDividendBlock / (*finalDivisorBlock + 1);
    DEBUG_ASSERT(quotient <= 9); // Assert that the estimated quotient is within [0, 9]

    /* Divide out the estimated quotient */
}
    if (quotient != 0) {
        /*
         * 如果商不为零，则执行以下操作：
         * dividend = dividend - divisor * quotient
         */
        const npy_uint32 *divisorCur = divisor->blocks;
        npy_uint32 *dividendCur = dividend->blocks;

        npy_uint64 borrow = 0;
        npy_uint64 carry = 0;
        do {
            npy_uint64 difference, product;

            product = (npy_uint64)*divisorCur * (npy_uint64)quotient + carry;
            carry = product >> 32;

            difference = (npy_uint64)*dividendCur
                       - (product & bitmask_u64(32)) - borrow;
            borrow = (difference >> 32) & 1;

            *dividendCur = difference & bitmask_u64(32);

            ++divisorCur;
            ++dividendCur;
        } while(divisorCur <= finalDivisorBlock);

        /*
         * 从 dividend 中移除所有前导的零块
         */
        while (length > 0 && dividend->blocks[length - 1] == 0) {
            --length;
        }

        dividend->length = length;
    }

    /*
     * 如果 dividend 仍然大于等于 divisor，则说明估算的商有误。
     * 此时需要增加商的值并再次减去一个 divisor 从 dividend 中。
     */
    if (BigInt_Compare(dividend, divisor) >= 0) {
        /*
         * dividend = dividend - divisor
         */
        const npy_uint32 *divisorCur = divisor->blocks;
        npy_uint32 *dividendCur = dividend->blocks;
        npy_uint64 borrow = 0;

        ++quotient;

        do {
            npy_uint64 difference = (npy_uint64)*dividendCur
                                  - (npy_uint64)*divisorCur - borrow;
            borrow = (difference >> 32) & 1;

            *dividendCur = difference & bitmask_u64(32);

            ++divisorCur;
            ++dividendCur;
        } while(divisorCur <= finalDivisorBlock);

        /*
         * 从 dividend 中移除所有前导的零块
         */
        while (length > 0 && dividend->blocks[length - 1] == 0) {
            --length;
        }

        dividend->length = length;
    }

    // 返回最终的商
    return quotient;
/* 左移大整数对象中的位，结果存储在 result 中 */
static void
BigInt_ShiftLeft(BigInt *result, npy_uint32 shift)
{
    /* 计算需要移动的块数和位数 */
    npy_uint32 shiftBlocks = shift / 32;
    npy_uint32 shiftBits = shift % 32;

    /* 从高到低处理块，以便可以安全地原地处理 */
    const npy_uint32 *pInBlocks = result->blocks;
    npy_int32 inLength = result->length;
    npy_uint32 *pInCur, *pOutCur;

    /* 断言确保移动后块的数量不超过最大限制 */
    DEBUG_ASSERT(inLength + shiftBlocks < c_BigInt_MaxBlocks);
    DEBUG_ASSERT(shift != 0);

    /* 检查移动是否按块对齐 */
    if (shiftBits == 0) {
        npy_uint32 i;

        /* 从高到低复制块 */
        for (pInCur = result->blocks + result->length,
                     pOutCur = pInCur + shiftBlocks;
             pInCur >= pInBlocks;
             --pInCur, --pOutCur) {
            *pOutCur = *pInCur;
        }

        /* 将剩余低块清零 */
        for (i = 0; i < shiftBlocks; ++i) {
            result->blocks[i] = 0;
        }

        result->length += shiftBlocks;
    }
    /* 否则需要移动部分块 */
    else {
        npy_uint32 i;
        npy_int32 inBlockIdx = inLength - 1;
        npy_uint32 outBlockIdx = inLength + shiftBlocks;

        /* 输出初始块 */
        const npy_uint32 lowBitsShift = (32 - shiftBits);
        npy_uint32 highBits = 0;
        npy_uint32 block = result->blocks[inBlockIdx];
        npy_uint32 lowBits = block >> lowBitsShift;

        /* 设置长度以容纳移动后的块 */
        DEBUG_ASSERT(outBlockIdx < c_BigInt_MaxBlocks);
        result->length = outBlockIdx + 1;

        while (inBlockIdx > 0) {
            result->blocks[outBlockIdx] = highBits | lowBits;
            highBits = block << shiftBits;

            --inBlockIdx;
            --outBlockIdx;

            block = result->blocks[inBlockIdx];
            lowBits = block >> lowBitsShift;
        }

        /* 输出最终块 */
        DEBUG_ASSERT(outBlockIdx == shiftBlocks + 1);
        result->blocks[outBlockIdx] = highBits | lowBits;
        result->blocks[outBlockIdx - 1] = block << shiftBits;

        /* 将剩余低块清零 */
        for (i = 0; i < shiftBlocks; ++i) {
            result->blocks[i] = 0;
        }

        /* 检查最后一个块是否没有设置位 */
        if (result->blocks[result->length - 1] == 0) {
            --result->length;
        }
    }
}


/* Dragon4 算法的实现，生成十进制数字字符串 */
static npy_uint32
Dragon4(BigInt *bigints, const npy_int32 exponent,
        const npy_uint32 mantissaBit, const npy_bool hasUnequalMargins,
        const DigitMode digitMode, const CutoffMode cutoffMode,
        npy_int32 cutoff_max, npy_int32 cutoff_min, char *pOutBuffer,
        npy_uint32 bufferSize, npy_int32 *pOutExponent)
{
    char *curDigit = pOutBuffer;
    /*
     * We compute values in integer format by rescaling as
     *   mantissa = scaledValue / scale
     *   marginLow = scaledMarginLow / scale
     *   marginHigh = scaledMarginHigh / scale
     * Here, marginLow and marginHigh represent 1/2 of the distance to the next
     * floating point value above/below the mantissa.
     *
     * scaledMarginHigh will point to scaledMarginLow in the case they must be
     * equal to each other, otherwise it will point to optionalMarginHigh.
     */

    // 定义各个 BigInt 变量指向 bigints 数组中的相应位置
    BigInt *mantissa = &bigints[0];  /* the only initialized bigint */
    BigInt *scale = &bigints[1];
    BigInt *scaledValue = &bigints[2];
    BigInt *scaledMarginLow = &bigints[3];
    BigInt *scaledMarginHigh;
    BigInt *optionalMarginHigh = &bigints[4];

    BigInt *temp1 = &bigints[5];
    BigInt *temp2 = &bigints[6];

    // 定义常量 log10_2 表示对数10以2为底的值
    const npy_float64 log10_2 = 0.30102999566398119521373889472449;
    npy_int32 digitExponent, hiBlock;
    npy_int32 cutoff_max_Exponent, cutoff_min_Exponent;
    npy_uint32 outputDigit;    /* current digit being output */
    npy_uint32 outputLen;
    npy_bool isEven = BigInt_IsEven(mantissa);  // 判断 mantissa 是否为偶数
    npy_int32 cmp;

    // values used to determine how to round
    // 用于确定四舍五入方式的值
    npy_bool low, high, roundDown;

    DEBUG_ASSERT(bufferSize > 0);  // 断言确保 bufferSize 大于0

    // 如果 mantissa 为零，则结果为零，无需进一步计算
    if (BigInt_IsZero(mantissa)) {
        *curDigit = '0';        // 当前数字为 '0'
        *pOutExponent = 0;      // 输出指数为 0
        return 1;               // 返回结果字符长度为1
    }

    // 将 mantissa 复制到 scaledValue 中，作为起始计算值
    BigInt_Copy(scaledValue, mantissa);
    if (hasUnequalMargins) {
        /* 如果边距不相等 */

        /* 如果指数大于0，表示没有小数部分 */
        if (exponent > 0) {
            /*
             * 1) 将输入值通过扩展乘以尾数和指数来展开。这表示输入值的整数表示方式。
             * 2) 应用额外的缩放因子2，以简化后续与边距值的比较。
             * 3) 将边距值设置为最低尾数位的缩放因子。
             */

            /* scaledValue      = 2 * 2 * mantissa*2^exponent */
            BigInt_ShiftLeft(scaledValue, exponent + 2);
            /* scale            = 2 * 2 * 1 */
            BigInt_Set_uint32(scale,  4);
            /* scaledMarginLow  = 2 * 2^(exponent-1) */
            BigInt_Pow2(scaledMarginLow, exponent);
            /* scaledMarginHigh = 2 * 2 * 2^(exponent-1) */
            BigInt_Pow2(optionalMarginHigh, exponent + 1);
        }
        /* else we have a fractional exponent */
        else {
            /*
             * 为了将尾数数据作为整数进行跟踪，我们将其存储为具有较大缩放的形式。
             */

            /* scaledValue      = 2 * 2 * mantissa */
            BigInt_ShiftLeft(scaledValue, 2);
            /* scale            = 2 * 2 * 2^(-exponent) */
            BigInt_Pow2(scale, -exponent + 2);
            /* scaledMarginLow  = 2 * 2^(-1) */
            BigInt_Set_uint32(scaledMarginLow, 1);
            /* scaledMarginHigh = 2 * 2 * 2^(-1) */
            BigInt_Set_uint32(optionalMarginHigh, 2);
        }

        /* 高和低边距不同 */
        scaledMarginHigh = optionalMarginHigh;
    }
    else {
        /* 如果边距相等 */

        /* 如果指数大于0，表示没有小数部分 */
        if (exponent > 0) {
            /* scaledValue     = 2 * mantissa*2^exponent */
            BigInt_ShiftLeft(scaledValue, exponent + 1);
            /* scale           = 2 * 1 */
            BigInt_Set_uint32(scale, 2);
            /* scaledMarginLow = 2 * 2^(exponent-1) */
            BigInt_Pow2(scaledMarginLow, exponent);
        }
        /* else we have a fractional exponent */
        else {
            /*
             * 为了将尾数数据作为整数进行跟踪，我们将其存储为具有较大缩放的形式。
             */

            /* scaledValue     = 2 * mantissa */
            BigInt_ShiftLeft(scaledValue, 1);
            /* scale           = 2 * 2^(-exponent) */
            BigInt_Pow2(scale, -exponent + 1);
            /* scaledMarginLow = 2 * 2^(-1) */
            BigInt_Set_uint32(scaledMarginLow, 1);
        }

        /* 高和低边距相等 */
        scaledMarginHigh = scaledMarginLow;
    }
    /*
     * 根据 Burger 和 Dybvig 的论文优化计算 digitExponent 的估算值，确保正确或者略低一位。
     * 参考论文链接：https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.72.4656
     * 我们额外减去 0.69 是为了增加估算失败的频率，这样可以在代码中执行更快速的分支。
     * 选择 0.69 的原因是因为 0.69 + log10(2) 比 1 小一个合理的 epsilon，可以考虑到任何浮点数误差。
     *
     * 我们希望将 digitExponent 设置为 floor(log10(v)) + 1
     *  v = mantissa*2^exponent
     *  log2(v) = log2(mantissa) + exponent;
     *  log10(v) = log2(v) * log10(2)
     *  floor(log2(v)) = mantissaBit + exponent;
     *  log10(v) - log10(2) < (mantissaBit + exponent) * log10(2) <= log10(v)
     *  log10(v) < (mantissaBit + exponent) * log10(2) + log10(2)
     *                                                 <= log10(v) + log10(2)
     *  floor(log10(v)) < ceil((mantissaBit + exponent) * log10(2))
     *                                                 <= floor(log10(v)) + 1
     *
     *  注意：这个计算假设 npy_float64 是 IEEE-binary64 浮点数。如果情况不是这样，可能需要更新此行代码。
     */
    digitExponent = (npy_int32)(
       ceil((npy_float64)((npy_int32)mantissaBit + exponent) * log10_2 - 0.69));

    /*
     * 如果 digitExponent 小于分数截断的最小期望位数，将其调整到合法范围内，以便后续进行四舍五入。
     * 注意，虽然 digitExponent 仍然是一个估算值，但这是安全的，因为它只会增加数字。
     * 这将修正 digitExponent 至准确值或将其夹到准确值之上。
     */
    if (cutoff_max >= 0 && cutoffMode == CutoffMode_FractionLength &&
            digitExponent <= -cutoff_max) {
        digitExponent = -cutoff_max + 1;
    }


    /* 将值除以 10^digitExponent。 */
    if (digitExponent > 0) {
        /* 正指数意味着除法，因此我们需要乘以相应的倍数。 */
        BigInt_MultiplyPow10(scale, digitExponent, temp1);
    }
    else if (digitExponent < 0) {
        /*
         * 负指数意味着乘法，因此我们需要对 scaledValue、scaledMarginLow 和 scaledMarginHigh 进行乘法运算。
         */
        BigInt *temp=temp1, *pow10=temp2;
        BigInt_Pow10(pow10, -digitExponent, temp);

        BigInt_Multiply(temp, scaledValue, pow10);
        BigInt_Copy(scaledValue, temp);

        BigInt_Multiply(temp, scaledMarginLow, pow10);
        BigInt_Copy(scaledMarginLow, temp);

        if (scaledMarginHigh != scaledMarginLow) {
            BigInt_Multiply2(scaledMarginHigh, scaledMarginLow);
        }
    }
    /* 如果 (value >= 1)，表示我们对 digitExponent 的估计太低了 */
    if (BigInt_Compare(scaledValue, scale) >= 0) {
        /*
         * 指数估计错误。
         * 增加指数并且不执行第一个循环迭代所需的预乘操作。
         */
        digitExponent = digitExponent + 1;
    }
    else {
        /*
         * 指数估计正确。
         * 将 scaledValue 和 scaledMarginLow 分别乘以输出基数，为第一个循环迭代做准备。
         */
        BigInt_Multiply10(scaledValue);
        BigInt_Multiply10(scaledMarginLow);
        if (scaledMarginHigh != scaledMarginLow) {
            BigInt_Multiply2(scaledMarginHigh, scaledMarginLow);
        }
    }

    /*
     * 计算 cutoff_max 指数（要打印的最后一个数字的指数）。默认为输出缓冲区的最大大小。
     */
    cutoff_max_Exponent = digitExponent - bufferSize;
    if (cutoff_max >= 0) {
        npy_int32 desiredCutoffExponent;

        if (cutoffMode == CutoffMode_TotalLength) {
            desiredCutoffExponent = digitExponent - cutoff_max;
            if (desiredCutoffExponent > cutoff_max_Exponent) {
                cutoff_max_Exponent = desiredCutoffExponent;
            }
        }
        /* 否则是 CutoffMode_FractionLength。打印小数点后的 cutoff_max 位数字或直到达到缓冲区大小 */
        else {
            desiredCutoffExponent = -cutoff_max;
            if (desiredCutoffExponent > cutoff_max_Exponent) {
                cutoff_max_Exponent = desiredCutoffExponent;
            }
        }
    }
    /* 同样计算 cutoff_min 指数。 */
    cutoff_min_Exponent = digitExponent;
    if (cutoff_min >= 0) {
        npy_int32 desiredCutoffExponent;

        if (cutoffMode == CutoffMode_TotalLength) {
            desiredCutoffExponent = digitExponent - cutoff_min;
            if (desiredCutoffExponent < cutoff_min_Exponent) {
                cutoff_min_Exponent = desiredCutoffExponent;
            }
        }
        else {
            desiredCutoffExponent = -cutoff_min;
            if (desiredCutoffExponent < cutoff_min_Exponent) {
                cutoff_min_Exponent = desiredCutoffExponent;
            }
        }
    }

    /* 输出将要打印的第一个数字的指数 */
    *pOutExponent = digitExponent - 1;

    /*
     * 为了调用 BigInt_DivideWithRemainder_MaxQuotient9() 做准备，
     * 我们需要扩展我们的值，以使分母的最高块大于或等于8。
     * 我们还需要保证在每个循环迭代之后，分子的长度永远不会大于分母的长度。
     * 这要求分母的最高块小于或等于 429496729，即可以乘以10而不会溢出到新的块。
     */
    DEBUG_ASSERT(scale->length > 0);
    // 获取最高块的值
    hiBlock = scale->blocks[scale->length - 1];
    // 检查最高块是否在有效范围内，若不在则执行以下操作
    if (hiBlock < 8 || hiBlock > 429496729) {
        npy_uint32 hiBlockLog2, shift;

        /*
         * 对所有值执行位移，将分母的最高块移动到范围[8, 429496729]内。
         * 我们更有可能在 BigInt_DivideWithRemainder_MaxQuotient9() 中使用较高的分母值进行准确的商估算，
         * 因此我们将分母移位，使最高位在最高块的第27位。
         * 这是安全的，因为 (2^28 - 1) = 268435455 小于 429496729。
         * 这意味着所有最高位在第27位的值都在有效范围内。
         */
        hiBlockLog2 = LogBase2_32(hiBlock);
        DEBUG_ASSERT(hiBlockLog2 < 3 || hiBlockLog2 > 27);
        shift = (32 + 27 - hiBlockLog2) % 32;

        BigInt_ShiftLeft(scale, shift);
        BigInt_ShiftLeft(scaledValue, shift);
        BigInt_ShiftLeft(scaledMarginLow, shift);
        if (scaledMarginHigh != scaledMarginLow) {
            BigInt_Multiply2(scaledMarginHigh, scaledMarginLow);
        }
    }

    if (digitMode == DigitMode_Unique) {
        /*
         * 对于唯一截断模式，我们将尝试打印直到达到可以唯一区分此值与其邻居的精度级别。
         * 如果输出缓冲区空间不足，我们会提前终止。
         */
        for (;;) {
            BigInt *scaledValueHigh = temp1;

            digitExponent = digitExponent - 1;

            /* 除以比例以提取数字 */
            outputDigit =
                BigInt_DivideWithRemainder_MaxQuotient9(scaledValue, scale);
            DEBUG_ASSERT(outputDigit < 10);

            /* 更新值的高端 */
            BigInt_Add(scaledValueHigh, scaledValue, scaledMarginHigh);

            /*
             * 如果我们距离相邻值足够远（并且我们已经打印了至少请求的最小数字），
             * 或者已达到截断数字，则停止循环。
             */
            cmp = BigInt_Compare(scaledValue, scaledMarginLow);
            low = isEven ? (cmp <= 0) : (cmp < 0);
            cmp = BigInt_Compare(scaledValueHigh, scale);
            high = isEven ? (cmp >= 0) : (cmp > 0);
            if (((low | high) & (digitExponent <= cutoff_min_Exponent)) |
                    (digitExponent == cutoff_max_Exponent)) {
                break;
            }

            /* 存储输出数字 */
            *curDigit = (char)('0' + outputDigit);
            ++curDigit;

            /* 将较大值乘以输出基数 */
            BigInt_Multiply10(scaledValue);
            BigInt_Multiply10(scaledMarginLow);
            if (scaledMarginHigh != scaledMarginLow) {
                BigInt_Multiply2(scaledMarginHigh, scaledMarginLow);
            }
        }
    }
    else {
        /*
         * For exact digit mode, we will try to print until we
         * have exhausted all precision (i.e. all remaining digits are zeros) or
         * until we reach the desired cutoff digit.
         */
        // 设置低位和高位标志为假，用于控制舍入
        low = NPY_FALSE;
        high = NPY_FALSE;

        // 无限循环，直到满足退出条件
        for (;;) {
            // 降低数字指数
            digitExponent = digitExponent - 1;

            /* divide out the scale to extract the digit */
            // 使用 BigInt_DivideWithRemainder_MaxQuotient9 函数从 scaledValue 中除去比例以提取数字
            outputDigit = BigInt_DivideWithRemainder_MaxQuotient9(scaledValue, scale);
            DEBUG_ASSERT(outputDigit < 10);

            // 如果 scaledValue 的长度为零，或者达到最大截断数字指数，则退出循环
            if ((scaledValue->length == 0) |
                    (digitExponent == cutoff_max_Exponent)) {
                break;
            }

            // 存储输出的数字
            *curDigit = (char)('0' + outputDigit);
            ++curDigit;

            // 将 scaledValue 乘以 10，准备下一个数字
            BigInt_Multiply10(scaledValue);
        }
    }

    /* default to rounding down the final digit if value got too close to 0 */
    // 如果低位为真，则默认向下舍入最后一个数字，如果值接近于零
    roundDown = low;

    /* if it is legal to round up and down */
    // 如果可以同时进行向上和向下舍入
    if (low == high) {
        npy_int32 compare;

        /*
         * round to the closest digit by comparing value with 0.5. To do this we
         * need to convert the inequality to large integer values.
         *  compare( value, 0.5 )
         *  compare( scale * value, scale * 0.5 )
         *  compare( 2 * scale * value, scale )
         */
        // 将 scaledValue 乘以 2，执行 BigInt_Multiply2_inplace 操作
        BigInt_Multiply2_inplace(scaledValue);
        // 比较 scaledValue 和 scale 的大小关系
        compare = BigInt_Compare(scaledValue, scale);
        // 根据比较结果决定是否向下舍入
        roundDown = compare < 0;

        /*
         * if we are directly in the middle, round towards the even digit (i.e.
         * IEEE rounding rules)
         */
        // 如果处于中间位置，按照偶数舍入规则处理
        if (compare == 0) {
            roundDown = (outputDigit & 1) == 0;
        }
    }

    /* print the rounded digit */
    // 打印舍入后的数字
    if (roundDown) {
        *curDigit = (char)('0' + outputDigit);
        ++curDigit;
    }
    else {
        /* handle rounding up */
        // 处理向上舍入的情况
        if (outputDigit == 9) {
            /* find the first non-nine prior digit */
            // 寻找第一个不为九的前一个数字
            for (;;) {
                /* if we are at the first digit */
                // 如果已经到达第一个数字
                if (curDigit == pOutBuffer) {
                    /* output 1 at the next highest exponent */
                    // 在更高的指数位置输出 1
                    *curDigit = '1';
                    ++curDigit;
                    *pOutExponent += 1;
                    break;
                }

                --curDigit;
                // 如果当前数字不是 '9'，则将其加一
                if (*curDigit != '9') {
                    *curDigit += 1;
                    ++curDigit;
                    break;
                }
            }
        }
        else {
            /* values in the range [0,8] can perform a simple round up */
            // 在范围 [0,8] 内的值可以进行简单的向上舍入
            *curDigit = (char)('0' + outputDigit + 1);
            ++curDigit;
        }
    }

    /* return the number of digits output */
    // 返回输出的数字长度
    outputLen = (npy_uint32)(curDigit - pOutBuffer);
    DEBUG_ASSERT(outputLen <= bufferSize);
    return outputLen;
/*
 * 结构体 Dragon4_Options 是用于方便传递 Dragon4 选项的结构体。
 *
 *   scientific - 控制是否使用科学计数法
 *   digit_mode - 决定使用唯一或固定小数输出
 *   cutoff_mode - 'precision' 是指所有数字，还是小数点后的数字
 *   precision - 当为负数时，打印足够唯一数字所需的位数；当为正数时，指定最大有效数字数
 *   sign - 是否始终显示符号
 *   trim_mode - 如何处理尾随的 0 和 '.'。参见 TrimMode 注释。
 *   digits_left - 小数点左边的填充字符数。-1 表示不填充
 *   digits_right - 小数点右边的填充字符数。-1 表示不填充。
 *                  填充会添加空格，直到小数点两边各有指定数量的字符。应用于 trim_mode 字符移除后。
 *                  如果 digits_right 是正数且小数点被移除，小数点将被空格字符替换。
 *   exp_digits - 仅影响科学计数法输出。如果为正数，用 0 填充指数直到达到这么多位数。如果为负数，只使用足够的位数。
 */
typedef struct Dragon4_Options {
    npy_bool scientific;
    DigitMode digit_mode;
    CutoffMode cutoff_mode;
    npy_int32 precision;
    npy_int32 min_digits;
    npy_bool sign;
    TrimMode trim_mode;
    npy_int32 digits_left;
    npy_int32 digits_right;
    npy_int32 exp_digits;
} Dragon4_Options;

/*
 * 输出正数的定点表示法：ddddd.dddd
 * 输出总是以 NUL 结尾，并返回输出长度（不包括 NUL）。
 *
 * 参数：
 *    buffer - 输出缓冲区
 *    bufferSize - 可以打印到缓冲区的最大字符数
 *    mantissa - 数值的尾数
 *    exponent - 数值在二进制中的指数
 *    signbit - 符号位的值。应为 '+', '-' 或 ''
 *    mantissaBit - 最高设置的尾数位索引
 *    hasUnequalMargins - 高边界是否是低边界的两倍
 *
 * 更多参数详见 Dragon4_Options 的描述。
 */
static npy_uint32
    /* 定义函数 FormatPositional，用于格式化并生成一个浮点数的字符串表示 */
    FormatPositional(char *buffer, npy_uint32 bufferSize, BigInt *mantissa,
                     npy_int32 exponent, char signbit, npy_uint32 mantissaBit,
                     npy_bool hasUnequalMargins, DigitMode digit_mode,
                     CutoffMode cutoff_mode, npy_int32 precision,
                     npy_int32 min_digits, TrimMode trim_mode,
                     npy_int32 digits_left, npy_int32 digits_right)
    {
        /* 声明局部变量 */
        npy_int32 printExponent;
        npy_int32 numDigits, numWholeDigits=0, has_sign=0;
        npy_int32 add_digits;

        /* 计算可以存储的最大字符串长度，并初始化位置指针 */
        npy_int32 maxPrintLen = (npy_int32)bufferSize - 1, pos = 0;

        /* 断言缓冲区大小大于零，确保缓冲区有效 */
        DEBUG_ASSERT(bufferSize > 0);

        /* 根据数字模式检查精度是否非负 */
        if (digit_mode != DigitMode_Unique) {
            DEBUG_ASSERT(precision >= 0);
        }

        /* 如果符号为正，则在缓冲区的开头添加 '+' */
        if (signbit == '+' && pos < maxPrintLen) {
            buffer[pos++] = '+';
            has_sign = 1;
        }
        /* 如果符号为负，则在缓冲区的开头添加 '-' */
        else if (signbit == '-' && pos < maxPrintLen) {
            buffer[pos++] = '-';
            has_sign = 1;
        }

        /* 调用 Dragon4 函数生成数字字符串，返回生成的数字字符数量 */
        numDigits = Dragon4(mantissa, exponent, mantissaBit, hasUnequalMargins,
                            digit_mode, cutoff_mode, precision, min_digits,
                            buffer + has_sign, maxPrintLen - has_sign,
                            &printExponent);

        /* 断言生成的数字字符数量大于零，并且不超过缓冲区大小 */
        DEBUG_ASSERT(numDigits > 0);
        DEBUG_ASSERT(numDigits <= bufferSize);

        /* 如果打印指数大于等于零，表示有整数部分 */
        if (printExponent >= 0) {
            /* 将整数部分留在缓冲区的开头 */
            numWholeDigits = printExponent + 1;

            /* 如果生成的数字字符数量小于等于整数部分的数量 */
            if (numDigits <= numWholeDigits) {
                npy_int32 count = numWholeDigits - numDigits;
                pos += numDigits;

                /* 避免缓冲区溢出 */
                if (pos + count > maxPrintLen) {
                    count = maxPrintLen - pos;
                }

                /* 添加末尾的零直到小数点位置 */
                numDigits += count;
                for (; count > 0; count--) {
                    buffer[pos++] = '0';
                }
            }
            /* 如果生成的数字字符数量大于整数部分数量 */
            else if (numDigits > numWholeDigits) {
                npy_int32 maxFractionDigits;

                /* 计算小数部分的数量，并限制在缓冲区可容纳的范围内 */
                numFractionDigits = numDigits - numWholeDigits;
                maxFractionDigits = maxPrintLen - numWholeDigits - 1 - pos;
                if (numFractionDigits > maxFractionDigits) {
                    numFractionDigits = maxFractionDigits;
                }

                /* 移动小数部分至正确的位置，插入小数点 */
                memmove(buffer + pos + numWholeDigits + 1,
                        buffer + pos + numWholeDigits, numFractionDigits);
                pos += numWholeDigits;
                buffer[pos] = '.';
                numDigits = numWholeDigits + 1 + numFractionDigits;
                pos += 1 + numFractionDigits;
            }
        }
    else {
        /* 将小数部分移出，以便为前导零腾出空间 */
        npy_int32 numFractionZeros = 0;
        if (pos + 2 < maxPrintLen) {
            npy_int32 maxFractionZeros, digitsStartIdx, maxFractionDigits, i;

            maxFractionZeros = maxPrintLen - 2 - pos;
            numFractionZeros = -(printExponent + 1);
            if (numFractionZeros > maxFractionZeros) {
                numFractionZeros = maxFractionZeros;
            }

            digitsStartIdx = 2 + numFractionZeros;

            /*
             * 将有效数字向右移动，以便有足够空间放置前导零
             */
            numFractionDigits = numDigits;
            maxFractionDigits = maxPrintLen - digitsStartIdx - pos;
            if (numFractionDigits > maxFractionDigits) {
                numFractionDigits = maxFractionDigits;
            }

            memmove(buffer + pos + digitsStartIdx, buffer + pos,
                    numFractionDigits);

            /* 插入前导零 */
            for (i = 2; i < digitsStartIdx; ++i) {
                buffer[pos + i] = '0';
            }

            /* 更新计数 */
            numFractionDigits += numFractionZeros;
            numDigits = numFractionDigits;
        }

        /* 添加小数点 */
        if (pos + 1 < maxPrintLen) {
            buffer[pos+1] = '.';
        }

        /* 添加初始零 */
        if (pos < maxPrintLen) {
            buffer[pos] = '0';
            numDigits += 1;
        }
        numWholeDigits = 1;
        pos += 2 + numFractionDigits;
    }

    /* 总是添加小数点，除非是 DprZeros 模式 */
    if (trim_mode != TrimMode_DptZeros && numFractionDigits == 0 &&
            pos < maxPrintLen) {
        buffer[pos++] = '.';
    }

    add_digits = digit_mode == DigitMode_Unique ? min_digits : precision;
    desiredFractionalDigits = add_digits < 0 ? 0 : add_digits;
    if (cutoff_mode == CutoffMode_TotalLength) {
        desiredFractionalDigits = add_digits - numWholeDigits;
    }

    if (trim_mode == TrimMode_LeaveOneZero) {
        /* 如果没有打印任何小数位，则添加一个尾随的 0 */
        if (numFractionDigits == 0 && pos < maxPrintLen) {
            buffer[pos++] = '0';
            numFractionDigits++;
        }
    }
    else if (trim_mode == TrimMode_None &&
             desiredFractionalDigits > numFractionDigits &&
             pos < maxPrintLen) {
        /* 添加尾随的零，直到达到 add_digits 的长度 */
        /* 计算所需的尾随零的数量 */
        npy_int32 count = desiredFractionalDigits - numFractionDigits;
        if (pos + count > maxPrintLen) {
            count = maxPrintLen - pos;
        }
        numFractionDigits += count;

        for ( ; count > 0; count--) {
            buffer[pos++] = '0';
        }
    }
    /* 否则，对于 trim_mode Zeros 或 DptZeros，无需再添加任何内容 */
}
    /*
     * 当进行四舍五入时，可能会产生末尾的零。根据修剪模式决定是否删除这些零。
     */
    if (trim_mode != TrimMode_None && numFractionDigits > 0) {
        // 循环移除末尾的零直到不是零为止
        while (buffer[pos-1] == '0') {
            pos--;
            numFractionDigits--;
        }
        // 如果末尾是小数点
        if (buffer[pos-1] == '.') {
            /* 在 TrimMode_LeaveOneZero 模式下，添加末尾的零 */
            if (trim_mode == TrimMode_LeaveOneZero){
                buffer[pos++] = '0';
                numFractionDigits++;
            }
            /* 在 TrimMode_DptZeros 模式下，移除末尾的小数点 */
            else if (trim_mode == TrimMode_DptZeros) {
                    pos--;
            }
        }
    }

    /* 添加右侧的空白填充 */
    if (digits_right >= numFractionDigits) {
        npy_int32 count = digits_right - numFractionDigits;

        /* 在 TrimMode_DptZeros 模式下，如果小数位数为零，且未到达最大打印长度，则添加一个空格代替小数点 */
        if (trim_mode == TrimMode_DptZeros && numFractionDigits == 0
                && pos < maxPrintLen) {
            buffer[pos++] = ' ';
        }

        // 如果要添加的空格数超过了最大打印长度与当前位置的差值，则调整为最大可添加的空格数
        if (pos + count > maxPrintLen) {
            count = maxPrintLen - pos;
        }

        // 添加右侧的空白填充
        for ( ; count > 0; count--) {
            buffer[pos++] = ' ';
        }
    }
    
    /* 添加左侧的空白填充 */
    if (digits_left > numWholeDigits + has_sign) {
        npy_int32 shift = digits_left - (numWholeDigits + has_sign);
        npy_int32 count = pos;

        // 如果要移动的字符数超过了最大打印长度与当前位置的差值，则调整为最大可移动的字符数
        if (count + shift > maxPrintLen) {
            count = maxPrintLen - shift;
        }

        // 将 buffer 中的字符向右移动 shift 个位置
        if (count > 0) {
            memmove(buffer + shift, buffer, count);
        }
        // 更新当前位置
        pos = shift + count;
        // 在左侧填充空白
        for ( ; shift > 0; shift--) {
            buffer[shift - 1] = ' ';
        }
    }

    /* 终止缓冲区 */
    DEBUG_ASSERT(pos <= maxPrintLen);
    buffer[pos] = '\0';

    // 返回有效字符数
    return pos;
/*
 * Outputs the positive number with scientific notation: d.dddde[sign]ddd
 * The output is always NUL terminated and the output length (not including the
 * NUL) is returned.
 *
 * Arguments:
 *    buffer - buffer to output into
 *    bufferSize - maximum characters that can be printed to buffer
 *    mantissa - value significand
 *    exponent - value exponent in base 2
 *    signbit - value of the sign position. Should be '+', '-' or ''
 *    mantissaBit - index of the highest set mantissa bit
 *    hasUnequalMargins - is the high margin twice as large as the low margin
 *
 * See Dragon4_Options for description of remaining arguments.
 */
static npy_uint32
FormatScientific (char *buffer, npy_uint32 bufferSize, BigInt *mantissa,
                  npy_int32 exponent, char signbit, npy_uint32 mantissaBit,
                  npy_bool hasUnequalMargins, DigitMode digit_mode,
                  npy_int32 precision, npy_int32 min_digits, TrimMode trim_mode,
                  npy_int32 digits_left, npy_int32 exp_digits)
{
    npy_int32 printExponent;
    npy_int32 numDigits;
    char *pCurOut;
    npy_int32 numFractionDigits;
    npy_int32 leftchars;
    npy_int32 add_digits;

    // 如果不是唯一数字模式，确保精度大于等于零
    if (digit_mode != DigitMode_Unique) {
        DEBUG_ASSERT(precision >= 0);
    }

    // 确保缓冲区大小大于零
    DEBUG_ASSERT(bufferSize > 0);

    pCurOut = buffer;

    /* add any whitespace padding to left side */
    // 添加任何左侧的空白填充
    leftchars = 1 + (signbit == '-' || signbit == '+');
    if (digits_left > leftchars) {
        int i;
        for (i = 0; i < digits_left - leftchars && bufferSize > 1; i++) {
            *pCurOut = ' ';
            pCurOut++;
            --bufferSize;
        }
    }

    // 添加正号或负号
    if (signbit == '+' && bufferSize > 1) {
        *pCurOut = '+';
        pCurOut++;
        --bufferSize;
    }
    else if (signbit == '-'  && bufferSize > 1) {
        *pCurOut = '-';
        pCurOut++;
        --bufferSize;
    }

    // 使用 Dragon4 算法生成科学计数法数字，并获取生成的数字长度
    numDigits = Dragon4(mantissa, exponent, mantissaBit, hasUnequalMargins,
                        digit_mode, CutoffMode_TotalLength,
                        precision < 0 ? -1 : precision + 1,
                        min_digits < 0 ? -1 : min_digits + 1,
                        pCurOut, bufferSize, &printExponent);

    // 确保生成的数字长度大于零，并不超过缓冲区大小
    DEBUG_ASSERT(numDigits > 0);
    DEBUG_ASSERT(numDigits <= bufferSize);

    /* keep the whole number as the first digit */
    // 将整数部分作为第一个数字保留
    if (bufferSize > 1) {
        pCurOut += 1;
        bufferSize -= 1;
    }

    /* insert the decimal point prior to the fractional number */
    // 在小数部分之前插入小数点
    numFractionDigits = numDigits - 1;
    if (numFractionDigits > 0 && bufferSize > 1) {
        npy_int32 maxFractionDigits = (npy_int32)bufferSize - 2;

        if (numFractionDigits > maxFractionDigits) {
            numFractionDigits =  maxFractionDigits;
        }

        // 将数字向后移动一个位置，插入小数点
        memmove(pCurOut + 1, pCurOut, numFractionDigits);
        pCurOut[0] = '.';
        pCurOut += (1 + numFractionDigits);
        bufferSize -= (1 + numFractionDigits);
    }
    /* 如果 trim_mode 不是 TrimMode_DptZeros 并且 numFractionDigits 等于 0，并且 bufferSize 大于 1，则添加小数点 */
    if (trim_mode != TrimMode_DptZeros && numFractionDigits == 0 &&
            bufferSize > 1) {
        *pCurOut = '.';
        ++pCurOut;
        --bufferSize;
    }

    /* 根据 digit_mode 和 precision 确定要添加的数字位数 */
    add_digits = digit_mode == DigitMode_Unique ? min_digits : precision;
    add_digits = add_digits < 0 ? 0 : add_digits;
    
    /* 如果 trim_mode 是 TrimMode_LeaveOneZero */
    if (trim_mode == TrimMode_LeaveOneZero) {
        /* 如果没有打印任何小数位，则添加一个 '0' */
        if (numFractionDigits == 0 && bufferSize > 1) {
            *pCurOut = '0';
            ++pCurOut;
            --bufferSize;
            ++numFractionDigits;
        }
    }
    /* 如果 trim_mode 是 TrimMode_None */
    else if (trim_mode == TrimMode_None) {
        /* 添加尾部的零，直到达到 add_digits 指定的长度 */
        if (add_digits > (npy_int32)numFractionDigits) {
            char *pEnd;
            /* 计算需要添加的尾部零的个数 */
            npy_int32 numZeros = (add_digits - numFractionDigits);

            /* 如果需要添加的零超过了 bufferSize 减去1，则限制为 bufferSize 减去1 */
            if (numZeros > (npy_int32)bufferSize - 1) {
                numZeros = (npy_int32)bufferSize - 1;
            }

            /* 在输出缓冲区的尾部添加零 */
            for (pEnd = pCurOut + numZeros; pCurOut < pEnd; ++pCurOut) {
                *pCurOut = '0';
                ++numFractionDigits;
            }
        }
    }
    /* 如果 trim_mode 是 TrimMode_Zeros 或 TrimMode_DptZeros，则不需要添加更多内容 */

    /*
     * 当进行四舍五入时，可能会有多余的尾部零。根据 trim 设置进行移除。
     */
    if (trim_mode != TrimMode_None && numFractionDigits > 0) {
        /* 向前移动 pCurOut，直到遇到非 '0' 字符 */
        --pCurOut;
        while (*pCurOut == '0') {
            --pCurOut;
            ++bufferSize;
            --numFractionDigits;
        }
        /* 如果 trim_mode 是 TrimMode_LeaveOneZero，并且最后一个字符是小数点，则添加一个 '0' */
        if (trim_mode == TrimMode_LeaveOneZero && *pCurOut == '.') {
            ++pCurOut;
            *pCurOut = '0';
            --bufferSize;
            ++numFractionDigits;
        }
        ++pCurOut;
    }

    /* 将指数打印到本地缓冲区，并复制到输出缓冲区 */
    // 如果缓冲区大小大于1，则执行以下操作
    if (bufferSize > 1) {
        // 定义指数缓冲区和数字数组
        char exponentBuffer[7];
        npy_int32 digits[5];
        npy_int32 i, exp_size, count;

        // 如果指数数字大于5，则将其限制为5
        if (exp_digits > 5) {
            exp_digits = 5;
        }
        // 如果指数数字小于0，则将其设置为2
        if (exp_digits < 0) {
            exp_digits = 2;
        }

        // 设置指数缓冲区的第一个字符为'e'
        exponentBuffer[0] = 'e';
        // 根据打印指数的正负性设置第二个字符
        if (printExponent >= 0) {
            exponentBuffer[1] = '+';
        }
        else {
            exponentBuffer[1] = '-';
            printExponent = -printExponent;
        }

        // 调试断言，验证打印指数小于100000
        DEBUG_ASSERT(printExponent < 100000);

        /* 获取指数的各个数字 */
        for (i = 0; i < 5; i++) {
            digits[i] = printExponent % 10;
            printExponent /= 10;
        }
        /* 回溯以去除前导零 */
        for (i = 5; i > exp_digits && digits[i-1] == 0; i--) {
        }
        // 计算有效的指数大小
        exp_size = i;
        /* 将剩余的数字写入临时缓冲区 */
        for (i = exp_size; i > 0; i--) {
            exponentBuffer[2 + (exp_size-i)] = (char)('0' + digits[i-1]);
        }

        /* 将指数缓冲区复制到输出中 */
        count = exp_size + 2;
        // 如果复制长度超过缓冲区大小减一，则截断
        if (count > (npy_int32)bufferSize - 1) {
            count = (npy_int32)bufferSize - 1;
        }
        // 将指数缓冲区内容复制到当前输出位置
        memcpy(pCurOut, exponentBuffer, count);
        // 更新当前输出位置指针
        pCurOut += count;
        // 更新剩余缓冲区大小
        bufferSize -= count;
    }

    // 调试断言，确保缓冲区大小仍大于0
    DEBUG_ASSERT(bufferSize > 0);
    // 将字符串结束符写入当前输出位置
    pCurOut[0] = '\0';

    // 返回填充后的输出长度
    return pCurOut - buffer;
/*
 * 打印给定宽度的十六进制值。
 * 输出的字符串总是以NUL结尾，并返回字符串长度（不包括NUL）。
 */
static npy_uint32
PrintHex(char * buffer, npy_uint32 bufferSize, npy_uint64 value,
         npy_uint32 width)
{
    const char digits[] = "0123456789abcdef";  // 十六进制数字字符集
    char *pCurOut;

    DEBUG_ASSERT(bufferSize > 0);  // 断言确保缓冲区大小大于0

    npy_uint32 maxPrintLen = bufferSize-1;
    if (width > maxPrintLen) {
        width = maxPrintLen;  // 限制打印宽度不超过缓冲区大小
    }

    pCurOut = buffer;
    while (width > 0) {
        --width;

        npy_uint8 digit = (npy_uint8)((value >> 4ull*(npy_uint64)width) & 0xF);  // 获取当前位的十六进制数字
        *pCurOut = digits[digit];  // 将当前位的十六进制数字写入输出缓冲区

        ++pCurOut;
    }

    *pCurOut = '\0';  // 结束字符串
    return pCurOut - buffer;  // 返回字符串长度（不包括NUL）
}

/*
 * 打印特殊情况下的无穷大和NaN值。
 * 输出的字符串总是以NUL结尾，并返回字符串长度（不包括NUL）。
 */
static npy_uint32
PrintInfNan(char *buffer, npy_uint32 bufferSize, npy_uint64 mantissa,
            npy_uint32 mantissaHexWidth, char signbit)
{
    npy_uint32 maxPrintLen = bufferSize-1;
    npy_uint32 pos = 0;

    DEBUG_ASSERT(bufferSize > 0);  // 断言确保缓冲区大小大于0

    /* 检查是否为无穷大 */
    if (mantissa == 0) {
        npy_uint32 printLen;

        /* 只为正负无穷值打印符号（尽管NaN可以有设置符号） */
        if (signbit == '+') {
            if (pos < maxPrintLen-1) {
                buffer[pos++] = '+';  // 如果缓冲区允许，添加正号
            }
        }
        else if (signbit == '-') {
            if (pos < maxPrintLen-1) {
                buffer[pos++] = '-';  // 如果缓冲区允许，添加负号
            }
        }

        /* 复制字符串并确保缓冲区以NUL结尾 */
        printLen = (3 < maxPrintLen - pos) ? 3 : maxPrintLen - pos;
        memcpy(buffer + pos, "inf", printLen);  // 将字符串 "inf" 复制到缓冲区
        buffer[pos + printLen] = '\0';  // 结束字符串
        return pos + printLen;  // 返回字符串长度（不包括NUL）
    }
    else {
        /* 复制字符串并确保缓冲区以NUL结尾 */
        npy_uint32 printLen = (3 < maxPrintLen - pos) ? 3 : maxPrintLen - pos;
        memcpy(buffer + pos, "nan", printLen);  // 将字符串 "nan" 复制到缓冲区
        buffer[pos + printLen] = '\0';  // 结束字符串

        /*
         * 对于numpy，我们忽略NaN的异常尾数值，但保留此代码以防以后更改我们的想法。
         *
         * // 追加十六进制值
         * if (maxPrintLen > 3) {
         *     printLen += PrintHex(buffer+3, bufferSize-3, mantissa,
         *                          mantissaHexWidth);
         * }
         */

        return pos + printLen;  // 返回字符串长度（不包括NUL）
    }
}
/*
 * The functions below format a floating-point numbers stored in particular
 * formats,  as a decimal string.  The output string is always NUL terminated
 * and the string length (not including the NUL) is returned.
 *
 * For 16, 32 and 64 bit floats we assume they are the IEEE 754 type.
 * For 128 bit floats we account for different definitions.
 *
 * Arguments are:
 *   buffer - buffer to output into
 *   bufferSize - maximum characters that can be printed to buffer
 *   value - value to print
 *   opt - Dragon4 options, see above
 */

/*
 * Helper function that takes Dragon4 parameters and options and
 * calls Dragon4.
 */
static npy_uint32
Format_floatbits(char *buffer, npy_uint32 bufferSize, BigInt *mantissa,
                 npy_int32 exponent, char signbit, npy_uint32 mantissaBit,
                 npy_bool hasUnequalMargins, Dragon4_Options *opt)
{
    /* format the value */
    if (opt->scientific) {
        // 调用科学计数法格式化函数 FormatScientific 来格式化浮点数值
        return FormatScientific(buffer, bufferSize, mantissa, exponent,
                                signbit, mantissaBit, hasUnequalMargins,
                                opt->digit_mode, opt->precision,
                                opt->min_digits, opt->trim_mode,
                                opt->digits_left, opt->exp_digits);
    }
    else {
        // 调用定点表示格式化函数 FormatPositional 来格式化浮点数值
        return FormatPositional(buffer, bufferSize, mantissa, exponent,
                                signbit, mantissaBit, hasUnequalMargins,
                                opt->digit_mode, opt->cutoff_mode,
                                opt->precision, opt->min_digits, opt->trim_mode,
                                opt->digits_left, opt->digits_right);
    }
}

/*
 * IEEE binary16 floating-point format
 *
 * sign:      1 bit
 * exponent:  5 bits
 * mantissa: 10 bits
 */
static npy_uint32
Dragon4_PrintFloat_IEEE_binary16(
        Dragon4_Scratch *scratch, npy_half *value, Dragon4_Options *opt)
{
    char *buffer = scratch->repr;   // 将输出缓冲区指向 scratch 结构体中的 repr 字段
    const npy_uint32 bufferSize = sizeof(scratch->repr);  // 获取输出缓冲区的最大容量
    BigInt *bigints = scratch->bigints;  // 获取 scratch 结构体中的大整数数组

    npy_uint16 val = *value;  // 获取传入的 16 位浮点数值
    npy_uint32 floatExponent, floatMantissa, floatSign;

    npy_uint32 mantissa;
    npy_int32 exponent;
    npy_uint32 mantissaBit;
    npy_bool hasUnequalMargins;
    char signbit = '\0';

    /* deconstruct the floating point value */
    floatMantissa = val & bitmask_u32(10);  // 从浮点数值中获取尾数部分
    floatExponent = (val >> 10) & bitmask_u32(5);  // 从浮点数值中获取指数部分
    floatSign = val >> 15;  // 从浮点数值中获取符号位

    /* output the sign */
    if (floatSign != 0) {
        signbit = '-';  // 如果符号位为1，则浮点数为负数
    }
    else if (opt->sign) {
        signbit = '+';  // 如果符号位为0且选项中需要显示符号，则为正数
    }

    /* if this is a special value */
    if (floatExponent == bitmask_u32(5)) {
        // 如果指数部分全为1，则该浮点数为特殊值（如无穷大或 NaN），调用 PrintInfNan 来处理
        return PrintInfNan(buffer, bufferSize, floatMantissa, 3, signbit);
    }
    /* else this is a number */

    /* factor the value into its parts */
}
    if (floatExponent != 0) {
        /*
         * 如果指数部分不为零，则为规格化浮点数
         * 浮点数的计算公式为:
         *  value = (1 + mantissa/2^10) * 2 ^ (exponent-15)
         * 我们通过将指数部分提取出一个2^10，将整数公式转换为:
         *  value = (1 + mantissa/2^10) * 2^10 * 2 ^ (exponent-15-10)
         *  value = (2^10 + mantissa) * 2 ^ (exponent-15-10)
         * 因为尾数前面有一个隐含的1，所以我们有10位精度。
         *   m = (2^10 + mantissa)
         *   e = (exponent-15-10)
         */
        mantissa            = (1UL << 10) | floatMantissa;
        exponent            = floatExponent - 15 - 10;
        mantissaBit         = 10;
        hasUnequalMargins   = (floatExponent != 1) && (floatMantissa == 0);
    }
    else {
        /*
         * 如果指数部分为零，则为非规格化浮点数
         * 浮点数的计算公式为:
         *  value = (mantissa/2^10) * 2 ^ (1-15)
         * 我们通过将指数部分提取出一个2^23，将整数公式转换为:
         *  value = (mantissa/2^10) * 2^10 * 2 ^ (1-15-10)
         *  value = mantissa * 2 ^ (1-15-10)
         * 我们有最多10位精度。
         *   m = (mantissa)
         *   e = (1-15-10)
         */
        mantissa           = floatMantissa;
        exponent           = 1 - 15 - 10;
        mantissaBit        = LogBase2_32(mantissa);
        hasUnequalMargins  = NPY_FALSE;
    }

    // 将尾数转换为大整数形式，并存储在bigints数组的第一个位置
    BigInt_Set_uint32(&bigints[0], mantissa);

    // 调用Format_floatbits函数，将浮点数各部分以字符串形式格式化到buffer中
    return Format_floatbits(buffer, bufferSize, bigints, exponent,
                            signbit, mantissaBit, hasUnequalMargins, opt);
/*
 * IEEE binary32 floating-point format
 *
 * sign:      1 bit
 * exponent:  8 bits
 * mantissa: 23 bits
 */

static npy_uint32
Dragon4_PrintFloat_IEEE_binary32(
        Dragon4_Scratch *scratch, npy_float32 *value,
        Dragon4_Options *opt)
{
    char *buffer = scratch->repr;  // 将 scratch 结构中的 repr 字段赋给 buffer，用于存储结果字符串
    const npy_uint32 bufferSize = sizeof(scratch->repr);  // 计算 repr 字段的字节大小并赋给 bufferSize
    BigInt *bigints = scratch->bigints;  // 将 scratch 结构中的 bigints 字段赋给 bigints，用于存储大整数对象数组

    union
    {
        npy_float32 floatingPoint;
        npy_uint32 integer;
    } floatUnion;  // 定义联合体 floatUnion，用于将浮点数和整数视为同一内存空间

    npy_uint32 floatExponent, floatMantissa, floatSign;  // 定义浮点数的指数、尾数和符号部分的变量

    npy_uint32 mantissa;  // 定义用于存储尾数的变量
    npy_int32 exponent;   // 定义用于存储指数的变量
    npy_uint32 mantissaBit;  // 定义用于存储尾数位数的变量
    npy_bool hasUnequalMargins;  // 定义用于标识是否具有不相等边界的布尔变量
    char signbit = '\0';  // 定义符号位的字符变量，默认为空字符

    /* deconstruct the floating point value */
    floatUnion.floatingPoint = *value;  // 将传入的浮点数值解构到 floatUnion 中
    floatMantissa = floatUnion.integer & bitmask_u32(23);  // 获取浮点数的尾数部分
    floatExponent = (floatUnion.integer >> 23) & bitmask_u32(8);  // 获取浮点数的指数部分
    floatSign = floatUnion.integer >> 31;  // 获取浮点数的符号部分

    /* output the sign */
    if (floatSign != 0) {
        signbit = '-';  // 如果浮点数为负数，则符号位为负号
    }
    else if (opt->sign) {
        signbit = '+';  // 如果浮点数为正数且 opt 参数要求显示符号，则符号位为正号
    }

    /* if this is a special value */
    if (floatExponent == bitmask_u32(8)) {
        return PrintInfNan(buffer, bufferSize, floatMantissa, 6, signbit);  // 如果浮点数是特殊值（如无穷大或NaN），则调用打印函数并返回
    }
    /* else this is a number */

    /* factor the value into its parts */
    if (floatExponent != 0) {
        /*
         * normalized
         * The floating point equation is:
         *  value = (1 + mantissa/2^23) * 2 ^ (exponent-127)
         * We convert the integer equation by factoring a 2^23 out of the
         * exponent
         *  value = (1 + mantissa/2^23) * 2^23 * 2 ^ (exponent-127-23)
         *  value = (2^23 + mantissa) * 2 ^ (exponent-127-23)
         * Because of the implied 1 in front of the mantissa we have 24 bits of
         * precision.
         *   m = (2^23 + mantissa)
         *   e = (exponent-127-23)
         */
        mantissa            = (1UL << 23) | floatMantissa;  // 计算规格化数的尾数部分
        exponent            = floatExponent - 127 - 23;  // 计算规格化数的指数部分
        mantissaBit         = 23;  // 尾数位数为 23
        hasUnequalMargins   = (floatExponent != 1) && (floatMantissa == 0);  // 判断是否具有不相等的边界
    }
    else {
        /*
         * denormalized
         * The floating point equation is:
         *  value = (mantissa/2^23) * 2 ^ (1-127)
         * We convert the integer equation by factoring a 2^23 out of the
         * exponent
         *  value = (mantissa/2^23) * 2^23 * 2 ^ (1-127-23)
         *  value = mantissa * 2 ^ (1-127-23)
         * We have up to 23 bits of precision.
         *   m = (mantissa)
         *   e = (1-127-23)
         */
        mantissa           = floatMantissa;  // 计算非规格化数的尾数部分
        exponent           = 1 - 127 - 23;  // 计算非规格化数的指数部分
        mantissaBit        = LogBase2_32(mantissa);  // 计算非规格化数的尾数位数
        hasUnequalMargins  = NPY_FALSE;  // 非规格化数不存在不相等的边界
    }

    BigInt_Set_uint32(&bigints[0], mantissa);  // 将尾数存入大整数对象数组中的第一个位置
    return Format_floatbits(buffer, bufferSize, bigints, exponent,
                           signbit, mantissaBit, hasUnequalMargins, opt);  // 调用格式化函数，生成浮点数的字符串表示并返回
}
/*
 * IEEE binary64 floating-point format
 *
 * sign:      1 bit              // 符号位：1位
 * exponent: 11 bits            // 指数部分：11位
 * mantissa: 52 bits            // 尾数部分：52位
 */
static npy_uint32
Dragon4_PrintFloat_IEEE_binary64(
        Dragon4_Scratch *scratch, npy_float64 *value, Dragon4_Options *opt)
{
    char *buffer = scratch->repr;                // 输出缓冲区
    const npy_uint32 bufferSize = sizeof(scratch->repr);  // 缓冲区大小
    BigInt *bigints = scratch->bigints;          // 大整数数组引用

    union
    {
        npy_float64 floatingPoint;
        npy_uint64 integer;
    } floatUnion;
    npy_uint32 floatExponent, floatSign;
    npy_uint64 floatMantissa;

    npy_uint64 mantissa;
    npy_int32 exponent;
    npy_uint32 mantissaBit;
    npy_bool hasUnequalMargins;
    char signbit = '\0';


    /* 分解浮点数值 */
    floatUnion.floatingPoint = *value;           // 将浮点数值存入联合体
    floatMantissa = floatUnion.integer & bitmask_u64(52);  // 提取尾数部分
    floatExponent = (floatUnion.integer >> 52) & bitmask_u32(11);  // 提取指数部分
    floatSign = floatUnion.integer >> 63;        // 提取符号位

    /* 输出符号位 */
    if (floatSign != 0) {
        signbit = '-';
    }
    else if (opt->sign) {
        signbit = '+';
    }

    /* 如果是特殊值 */
    if (floatExponent == bitmask_u32(11)) {
        return PrintInfNan(buffer, bufferSize, floatMantissa, 13, signbit);  // 输出无穷大或NaN
    }
    /* 否则是一个数字 */

    /* 将值分解为其部分 */
    if (floatExponent != 0) {
        /*
         * 正常值
         * 浮点数方程为：
         *   value = (1 + mantissa/2^52) * 2 ^ (exponent-1023)
         * 我们通过将2^52从指数中分解出来，将整数方程转换为：
         *   value = (1 + mantissa/2^52) * 2^52 * 2 ^ (exponent-1023-52)
         *   value = (2^52 + mantissa) * 2 ^ (exponent-1023-52)
         * 因为尾数前面隐含了1，我们有53位精度。
         *   m = (2^52 + mantissa)
         *   e = (exponent-1023+1-53)
         */
        mantissa            = (1ull << 52) | floatMantissa;  // 计算正常情况下的尾数
        exponent            = floatExponent - 1023 - 52;     // 计算正常情况下的指数
        mantissaBit         = 52;                            // 尾数的位数
        hasUnequalMargins   = (floatExponent != 1) && (floatMantissa == 0);  // 检查是否存在不等的边界
    }
    else {
        /*
         * 亚正常值
         * 浮点数方程为：
         *   value = (mantissa/2^52) * 2 ^ (1-1023)
         * 我们通过将2^52从指数中分解出来，将整数方程转换为：
         *   value = (mantissa/2^52) * 2^52 * 2 ^ (1-1023-52)
         *   value = mantissa * 2 ^ (1-1023-52)
         * 我们有高达52位的精度。
         *   m = (mantissa)
         *   e = (1-1023-52)
         */
        mantissa            = floatMantissa;                  // 计算亚正常情况下的尾数
        exponent            = 1 - 1023 - 52;                  // 计算亚正常情况下的指数
        mantissaBit         = LogBase2_64(mantissa);          // 尾数的位数
        hasUnequalMargins   = NPY_FALSE;                      // 检查是否存在不等的边界
    }

    BigInt_Set_uint64(&bigints[0], mantissa);                 // 设置大整数数组中的尾数值
    return Format_floatbits(buffer, bufferSize, bigints, exponent,
                            signbit, mantissaBit, hasUnequalMargins, opt);  // 格式化输出浮点数位
}
/*
 * 由于不同系统可能有不同类型的 long double，并且可能没有用于传递值的 128 字节格式，
 * 因此在这里我们创建自己的 128 位存储类型以方便操作。
 */
typedef struct FloatVal128 {
    npy_uint64 hi, lo;  // 高位和低位分别存储 64 位无符号整数
} FloatVal128;

#if defined(HAVE_LDOUBLE_INTEL_EXTENDED_10_BYTES_LE) || \
    defined(HAVE_LDOUBLE_INTEL_EXTENDED_12_BYTES_LE) || \
    defined(HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE) || \
    defined(HAVE_LDOUBLE_MOTOROLA_EXTENDED_12_BYTES_BE)
/*
 * Intel 的 80 位 IEEE 扩展精度浮点格式
 *
 * 使用此格式的 "long double" 存储为 96 或 128 位，但相当于带有高位零填充的 80 位类型。
 * Dragon4_PrintFloat_Intel_extended 函数期望用户使用 128 位的 FloatVal128 传入值，
 * 以支持 80、96 或 128 位存储格式，并且是端序无关的。
 *
 * sign:      1 bit,  second u64
 * exponent: 15 bits, second u64
 * intbit     1 bit,  first u64
 * mantissa: 63 bits, first u64
 */
static npy_uint32
Dragon4_PrintFloat_Intel_extended(
    Dragon4_Scratch *scratch, FloatVal128 value, Dragon4_Options *opt)
{
    char *buffer = scratch->repr;  // 字符串缓冲区
    const npy_uint32 bufferSize = sizeof(scratch->repr);  // 缓冲区大小
    BigInt *bigints = scratch->bigints;  // 大整数数组

    npy_uint32 floatExponent, floatSign;  // 浮点数的指数和符号
    npy_uint64 floatMantissa;  // 浮点数的尾数

    npy_uint64 mantissa;  // 尾数
    npy_int32 exponent;  // 指数
    npy_uint32 mantissaBit;  // 尾数位
    npy_bool hasUnequalMargins;  // 不等间距标志
    char signbit = '\0';  // 符号位初始化为空字符

    /* 拆解浮点数值（忽略 intbit） */
    floatMantissa = value.lo & bitmask_u64(63);  // 提取低位 63 位作为浮点数的尾数
    floatExponent = value.hi & bitmask_u32(15);  // 提取高位 15 位作为浮点数的指数
    floatSign = (value.hi >> 15) & 0x1;  // 提取符号位，右移 15 位并掩码出最低位

    /* 输出符号位 */
    if (floatSign != 0) {
        signbit = '-';  // 若符号位不为 0，则设为负号
    }
    else if (opt->sign) {
        signbit = '+';  // 否则如果选项中要求显示符号，则设为正号
    }

    /* 如果这是一个特殊值 */
    if (floatExponent == bitmask_u32(15)) {
        /*
         * 注意：技术上还有其他特殊的扩展值，如果 intbit 为 0，例如伪无穷大、伪 NaN、Quiet NaN。
         * 我们忽略所有这些，因为它们在现代处理器上不会生成。我们将 Quiet NaN 视为普通 NaN。
         */
        return PrintInfNan(buffer, bufferSize, floatMantissa, 16, signbit);  // 打印无穷大或 NaN
    }
    /* 否则，这是一个数字 */

    /* 分解值为其部分 */
    // （接下来的代码行未包含在示例中，因此不在注释范围内）
    if (floatExponent != 0) {
        /*
         * normal
         * 浮点数的计算公式为：
         *   value = (1 + mantissa/2^63) * 2 ^ (exponent-16383)
         * 我们通过将指数中的 2^63 提取出来，转换为整数计算：
         *   value = (1 + mantissa/2^63) * 2^63 * 2 ^ (exponent-16383-63)
         *   value = (2^63 + mantissa) * 2 ^ (exponent-16383-63)
         * 因为尾数前有一个隐含的 1，所以我们有64位的精度。
         *   m = (2^63 + mantissa)
         *   e = (exponent-16383+1-64)
         */
        mantissa            = (1ull << 63) | floatMantissa;
        exponent            = floatExponent - 16383 - 63;
        mantissaBit         = 63;
        hasUnequalMargins   = (floatExponent != 1) && (floatMantissa == 0);
    }
    else {
        /*
         * subnormal
         * 浮点数的计算公式为：
         *   value = (mantissa/2^63) * 2 ^ (1-16383)
         * 我们通过将指数中的 2^63 提取出来，转换为整数计算：
         *   value = (mantissa/2^63) * 2^52 * 2 ^ (1-16383-63)
         *   value = mantissa * 2 ^ (1-16383-63)
         * 我们有最多63位的精度。
         *   m = (mantissa)
         *   e = (1-16383-63)
         */
        mantissa            = floatMantissa;
        exponent            = 1 - 16383 - 63;
        mantissaBit         = LogBase2_64(mantissa);  // 计算64位整数的对数
        hasUnequalMargins   = NPY_FALSE;  // 设定为假
    }

    BigInt_Set_uint64(&bigints[0], mantissa);  // 设置大整数的64位无符号整数
    return Format_floatbits(buffer, bufferSize, bigints, exponent,
                            signbit, mantissaBit, hasUnequalMargins, opt);  // 格式化浮点位，并返回格式化后的字符串
#ifdef NPY_FLOAT128

typedef union FloatUnion128
{
    npy_float128 floatingPoint;  // 定义一个联合体，支持 npy_float128 类型的浮点数
    struct {
        npy_uint64 a;  // 联合体中的整数部分 a，占用 64 位
        npy_uint64 b;  // 联合体中的整数部分 b，占用 64 位
    } integer;
} FloatUnion128;

#ifdef HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE
/* Intel's 80-bit IEEE extended precision format, 128-bit storage */
static npy_uint32
Dragon4_PrintFloat_Intel_extended128(
    Dragon4_Scratch *scratch, npy_float128 *value, Dragon4_Options *opt)
{
    FloatVal128 val128;  // 定义一个 FloatVal128 结构体变量
    FloatUnion128 buf128;  // 定义一个 FloatUnion128 联合体变量，用于存储 npy_float128 类型的浮点数的整数部分

    buf128.floatingPoint = *value;  // 将传入的 npy_float128 浮点数值存入联合体变量 buf128 中
    /* Intel is little-endian */
    val128.lo = buf128.integer.a;  // 将 buf128 的整数部分 a 赋值给 val128 的低位
    val128.hi = buf128.integer.b;  // 将 buf128 的整数部分 b 赋值给 val128 的高位

    // 调用 Dragon4_PrintFloat_Intel_extended 函数处理 Intel 扩展格式的浮点数输出
    return Dragon4_PrintFloat_Intel_extended(scratch, val128, opt);
}
#endif /* HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE */
    # 调用 Dragon4_PrintFloat_Intel_extended 函数，并返回其结果
    return Dragon4_PrintFloat_Intel_extended(scratch, val128, opt);
#endif /* HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE */

#if defined(HAVE_LDOUBLE_IEEE_QUAD_LE) || defined(HAVE_LDOUBLE_IEEE_QUAD_BE)
/*
 * IEEE binary128 floating-point format
 *
 * sign:       1 bit
 * exponent:  15 bits
 * mantissa: 112 bits
 *
 * Currently binary128 format exists on only a few CPUs, such as on the POWER9
 * arch or aarch64. Because of this, this code has not been extensively tested.
 * I am not sure if the arch also supports uint128, and C does not seem to
 * support int128 literals. So we use uint64 to do manipulation.
 */
static npy_uint32
Dragon4_PrintFloat_IEEE_binary128(
    Dragon4_Scratch *scratch, FloatVal128 val128, Dragon4_Options *opt)
{
    char *buffer = scratch->repr;           // 缓冲区指针，用于存储浮点数的字符串表示
    const npy_uint32 bufferSize = sizeof(scratch->repr);  // 缓冲区的大小
    BigInt *bigints = scratch->bigints;     // 大整数数组的指针

    npy_uint32 floatExponent, floatSign;    // 浮点数的指数和符号位

    npy_uint64 mantissa_hi, mantissa_lo;    // 浮点数的高位和低位有效位
    npy_int32 exponent;                     // 浮点数的指数部分
    npy_uint32 mantissaBit;                 // 浮点数有效位数
    npy_bool hasUnequalMargins;             // 浮点数是否有不相等的边界
    char signbit = '\0';                    // 符号位字符，默认为空字符

    mantissa_hi = val128.hi & bitmask_u64(48);  // 获取浮点数高 64 位中的有效位
    mantissa_lo = val128.lo;                    // 获取浮点数低 64 位中的有效位
    floatExponent = (val128.hi >> 48) & bitmask_u32(15);  // 获取浮点数的指数部分
    floatSign = val128.hi >> 63;                       // 获取浮点数的符号位

    /* output the sign */
    if (floatSign != 0) {
        signbit = '-';                  // 如果符号位为1，表示负数
    }
    else if (opt->sign) {
        signbit = '+';                  // 如果符号位为0，且选项中需要显示符号，则为正数
    }

    /* if this is a special value */
    if (floatExponent == bitmask_u32(15)) {
        npy_uint64 mantissa_zero = mantissa_hi == 0 && mantissa_lo == 0;
        return PrintInfNan(buffer, bufferSize, !mantissa_zero, 16, signbit);
        // 如果浮点数是特殊值（如无穷大或NaN），则调用打印函数并返回
    }
    /* else this is a number */

    /* factor the value into its parts */
    if (floatExponent != 0) {
        /*
         * normal
         * The floating point equation is:
         *  value = (1 + mantissa/2^112) * 2 ^ (exponent-16383)
         * We convert the integer equation by factoring a 2^112 out of the
         * exponent
         *  value = (1 + mantissa/2^112) * 2^112 * 2 ^ (exponent-16383-112)
         *  value = (2^112 + mantissa) * 2 ^ (exponent-16383-112)
         * Because of the implied 1 in front of the mantissa we have 112 bits of
         * precision.
         *   m = (2^112 + mantissa)
         *   e = (exponent-16383+1-112)
         *
         *   Adding 2^112 to the mantissa is the same as adding 2^48 to the hi
         *   64 bit part.
         */
        mantissa_hi = (1ull << 48) | mantissa_hi;  // 将2^48加到高64位中，构造浮点数的有效位
        /* mantissa_lo is unchanged */
        exponent = floatExponent - 16383 - 112;   // 计算浮点数的指数部分
        mantissaBit = 112;                        // 浮点数的有效位数为112位
        hasUnequalMargins = (floatExponent != 1) && (mantissa_hi == 0 &&
                                                      mantissa_lo == 0);  // 判断是否有不相等的边界
    }
    else {
        /*
         * subnormal
         * 浮点数为次正规化情况
         * 浮点数计算公式为：
         *  value = (mantissa/2^112) * 2 ^ (1-16383)
         * 我们通过将指数中的 2^112 提取出来，转换为整数计算公式：
         *  value = (mantissa/2^112) * 2^112 * 2 ^ (1-16383-112)
         *  value = mantissa * 2 ^ (1-16383-112)
         * 我们有高达112位的精度。
         *   m = (mantissa)
         *   e = (1-16383-112)
         */
        // 计算指数
        exponent            = 1 - 16383 - 112;
        // 计算mantissa的位数
        hasUnequalMargins   = NPY_FALSE;
    }

    // 设置BigInt结构体，存
#if defined(HAVE_LDOUBLE_IBM_DOUBLE_DOUBLE_LE) || defined(HAVE_LDOUBLE_IBM_DOUBLE_DOUBLE_BE)
/*
 * IBM extended precision 128-bit floating-point format, aka IBM double-double
 *
 * IBM的双倍精度类型是一对IEEE二进制64位值，将它们相加得到总值。指数排列使得较低的双精度大约比高位双精度小2^52倍，最接近的float64值就是简单地使用上位双精度，此时对成对的值视为“规范化”（不要与“正常”和“次正常”的二进制64位值混淆）。我们假设这些值是规范化的。你可以通过构造非规范化值来看到glibc在ppc上的printf会产生奇怪的行为：
 *
 *     >>> from numpy._core._multiarray_tests import format_float_OSprintf_g
 *     >>> x = np.array([0.3,0.3], dtype='f8').view('f16')[0]
 *     >>> format_float_OSprintf_g(x, 2)
 *     0.30
 *     >>> format_float_OSprintf_g(2*x, 2)
 *     1.20
 *
 * 如果我们不假设规范化，x应该打印为0.6。
 *
 * 对于规范化值，gcc假设总的尾数不超过106位（53+53），因此当左移其指数导致第二个双精度超过106位时，我们可以丢弃一些位数，这种情况有时会发生。（对此曾有过争论，参见 https://gcc.gnu.org/bugzilla/show_bug.cgi?format=multiple&id=70117, https://sourceware.org/bugzilla/show_bug.cgi?id=22752 ）
 *
 * 注意：此函数适用于IBM双倍精度，它是一对IEEE二进制64位浮点数，如在ppc64系统上。这*不是*十六进制的IBM双倍精度类型，后者是一对IBM十六进制64位浮点数。
 *
 * 参见：
 * https://gcc.gnu.org/wiki/Ieee128PowerPCA
 * https://www.ibm.com/support/knowledgecenter/en/ssw_aix_71/com.ibm.aix.genprogc/128bit_long_double_floating-point_datatype.htm
 */
static npy_uint32
Dragon4_PrintFloat_IBM_double_double(
    Dragon4_Scratch *scratch, npy_float128 *value, Dragon4_Options *opt)
{
    // 获取 scratch 结构体中的字符串缓冲区指针
    char *buffer = scratch->repr;
    // 获取字符串缓冲区大小
    const npy_uint32 bufferSize = sizeof(scratch->repr);
    // 获取 scratch 结构体中的大整数数组指针
    BigInt *bigints = scratch->bigints;

    // 定义用于存储浮点数值的结构体和联合体
    FloatVal128 val128;
    FloatUnion128 buf128;

    // 定义用于存储浮点数的指数和尾数的变量
    npy_uint32 floatExponent1, floatExponent2;
    npy_uint64 floatMantissa1, floatMantissa2;
    npy_uint32 floatSign1, floatSign2;

    // 定义用于存储转换后的尾数和指数
    npy_uint64 mantissa1, mantissa2;
    npy_int32 exponent1, exponent2;
    int shift;
    npy_uint32 mantissaBit;
    npy_bool hasUnequalMargins;
    char signbit = '\0';

    /* The high part always comes before the low part, regardless of the
     * endianness of the system. */
    // 将浮点数值分解为高位和低位，不受系统字节顺序影响
    buf128.floatingPoint = *value;
    val128.hi = buf128.integer.a;
    val128.lo = buf128.integer.b;

    /* deconstruct the floating point values */
    // 解析浮点数值，获取尾数、指数和符号位
    floatMantissa1 = val128.hi & bitmask_u64(52);
    floatExponent1 = (val128.hi >> 52) & bitmask_u32(11);
    floatSign1 = (val128.hi >> 63) != 0;

    floatMantissa2 = val128.lo & bitmask_u64(52);
    floatExponent2 = (val128.lo >> 52) & bitmask_u32(11);
    floatSign2 = (val128.lo >> 63) != 0;

    /* output the sign using 1st float's sign */
    // 根据第一个浮点数的符号位确定输出的符号
    if (floatSign1) {
        signbit = '-';
    }
    else if (opt->sign) {
        signbit = '+';
    }

    /* we only need to look at the first float for inf/nan */
    // 如果第一个浮点数的指数字段全为1，则表示它是特殊值（无穷大或NaN）
    if (floatExponent1 == bitmask_u32(11)) {
        // 调用打印无穷大或NaN的函数，并返回结果
        return PrintInfNan(buffer, bufferSize, floatMantissa1, 13, signbit);
    }

    /* else this is a number */

    /* Factor the 1st value into its parts, see binary64 for comments. */
    // 将第一个浮点数值分解为多个部分，参考binary64的注释
    if (floatExponent1 == 0) {
        /*
         * If the first number is a subnormal value, the 2nd has to be 0 for
         * the float128 to be normalized, so we can ignore it. In this case
         * the float128 only has the precision of a single binary64 value.
         */
        // 如果第一个浮点数是次正规值（subnormal），则第二个浮点数的尾数必须为0，
        // 以使得float128规格化，此时float128的精度仅等同于单个binary64值
        mantissa1            = floatMantissa1;
        exponent1            = 1 - 1023 - 52;
        mantissaBit          = LogBase2_64(mantissa1);
        hasUnequalMargins    = NPY_FALSE;

        // 将尾数存入大整数结构体的第一个元素
        BigInt_Set_uint64(&bigints[0], mantissa1);
    }
    }

    // 调用格式化浮点数位的函数，并返回结果
    return Format_floatbits(buffer, bufferSize, bigints, exponent1,
                            signbit, mantissaBit, hasUnequalMargins, opt);
}

#endif /* HAVE_LDOUBLE_IBM_DOUBLE_DOUBLE_LE | HAVE_LDOUBLE_IBM_DOUBLE_DOUBLE_BE */

#endif /* NPY_FLOAT128 */


/*
 * Here we define two Dragon4 entry functions for each type. One of them
 * accepts the args in a Dragon4_Options struct for convenience, the
 * other enumerates only the necessary parameters.
 *
 * Use a very large string buffer in case anyone tries to output a large number.
 * 16384 should be enough to exactly print the integer part of any float128,
 * which goes up to about 10^4932. The Dragon4_scratch struct provides a string
 * buffer of this size.
 */
// 定义两个Dragon4入口函数，一个使用Dragon4_Options结构体中的参数，另一个列举必要的参数

#define make_dragon4_typefuncs_inner(Type, npy_type, format) \
\
PyObject *\
PyObject *Dragon4_Positional_##Type##_opt(npy_type *val, Dragon4_Options *opt)\
{\
    PyObject *ret;\
    // 获取 Dragon4_Scratch 结构的指针，用于 Dragon4 算法的临时存储
    Dragon4_Scratch *scratch = get_dragon4_bigint_scratch();\
    // 如果获取 Dragon4_Scratch 失败，则返回空指针
    if (scratch == NULL) {\
        return NULL;\
    }\
    // 调用 Dragon4_PrintFloat_##format 函数打印浮点数到字符串，若出错则释放 scratch 并返回空指针
    if (Dragon4_PrintFloat_##format(scratch, val, opt) < 0) {\
        free_dragon4_bigint_scratch(scratch);\
        return NULL;\
    }\
    // 从 scratch->repr 创建 Python Unicode 对象
    ret = PyUnicode_FromString(scratch->repr);\
    // 释放 Dragon4_Scratch 结构所占用的内存
    free_dragon4_bigint_scratch(scratch);\
    // 返回 Python Unicode 对象
    return ret;\
}\
\
PyObject *\
Dragon4_Positional_##Type(npy_type *val, DigitMode digit_mode,\
                   CutoffMode cutoff_mode, int precision, int min_digits, \
                   int sign, TrimMode trim, int pad_left, int pad_right)\
{\
    Dragon4_Options opt;\
    \
    // 初始化 Dragon4_Options 结构体的各个成员变量
    opt.scientific = 0;\
    opt.digit_mode = digit_mode;\
    opt.cutoff_mode = cutoff_mode;\
    opt.precision = precision;\
    opt.min_digits = min_digits;\
    opt.sign = sign;\
    opt.trim_mode = trim;\
    opt.digits_left = pad_left;\
    opt.digits_right = pad_right;\
    opt.exp_digits = -1;\
\
    // 调用 Dragon4_Positional_##Type##_opt 函数，返回其结果
    return Dragon4_Positional_##Type##_opt(val, &opt);\
}\
\
PyObject *\
Dragon4_Scientific_##Type##_opt(npy_type *val, Dragon4_Options *opt)\
{\
    PyObject *ret;\
    // 获取 Dragon4_Scratch 结构的指针，用于 Dragon4 算法的临时存储
    Dragon4_Scratch *scratch = get_dragon4_bigint_scratch();\
    // 如果获取 Dragon4_Scratch 失败，则返回空指针
    if (scratch == NULL) {\
        return NULL;\
    }\
    // 调用 Dragon4_PrintFloat_##format 函数打印浮点数到字符串，若出错则释放 scratch 并返回空指针
    if (Dragon4_PrintFloat_##format(scratch, val, opt) < 0) {\
        free_dragon4_bigint_scratch(scratch);\
        return NULL;\
    }\
    // 从 scratch->repr 创建 Python Unicode 对象
    ret = PyUnicode_FromString(scratch->repr);\
    // 释放 Dragon4_Scratch 结构所占用的内存
    free_dragon4_bigint_scratch(scratch);\
    // 返回 Python Unicode 对象
    return ret;\
}\
\
PyObject *\
Dragon4_Scientific_##Type(npy_type *val, DigitMode digit_mode, int precision,\
                   int min_digits, int sign, TrimMode trim, int pad_left, \
                   int exp_digits)\
{\
    Dragon4_Options opt;\
\
    // 初始化 Dragon4_Options 结构体的各个成员变量
    opt.scientific = 1;\
    opt.digit_mode = digit_mode;\
    opt.cutoff_mode = CutoffMode_TotalLength;\
    opt.precision = precision;\
    opt.min_digits = min_digits;\
    opt.sign = sign;\
    opt.trim_mode = trim;\
    opt.digits_left = pad_left;\
    opt.digits_right = -1;\
    opt.exp_digits = exp_digits;\
\
    // 调用 Dragon4_Scientific_##Type##_opt 函数，返回其结果
    return Dragon4_Scientific_##Type##_opt(val, &opt);\
}
    # 设置输出选项的精度
    opt.precision = precision;
    # 设置输出选项的最小数字位数
    opt.min_digits = min_digits;
    # 设置输出选项的符号显示
    opt.sign = sign;
    # 设置输出选项的修剪模式
    opt.trim_mode = trim;
    # 设置输出选项的左侧填充位数
    opt.digits_left = pad_left;
    # 设置输出选项的右侧填充位数
    opt.digits_right = pad_right;
    # 设置输出选项的指数位数为默认值 -1
    opt.exp_digits = -1;

    # 检查 obj 是否是半精度浮点数标量
    if (PyArray_IsScalar(obj, Half)) {
        # 获取半精度浮点数的值 x
        npy_half x = PyArrayScalar_VAL(obj, Half);
        # 调用 Dragon4 算法计算半精度浮点数的字符串表示，并返回结果
        return Dragon4_Positional_Half_opt(&x, &opt);
    }
    # 检查 obj 是否是单精度浮点数标量
    else if (PyArray_IsScalar(obj, Float)) {
        # 获取单精度浮点数的值 x
        npy_float x = PyArrayScalar_VAL(obj, Float);
        # 调用 Dragon4 算法计算单精度浮点数的字符串表示，并返回结果
        return Dragon4_Positional_Float_opt(&x, &opt);
    }
    # 检查 obj 是否是双精度浮点数标量
    else if (PyArray_IsScalar(obj, Double)) {
        # 获取双精度浮点数的值 x
        npy_double x = PyArrayScalar_VAL(obj, Double);
        # 调用 Dragon4 算法计算双精度浮点数的字符串表示，并返回结果
        return Dragon4_Positional_Double_opt(&x, &opt);
    }
    # 检查 obj 是否是长双精度浮点数标量
    else if (PyArray_IsScalar(obj, LongDouble)) {
        # 获取长双精度浮点数的值 x
        npy_longdouble x = PyArrayScalar_VAL(obj, LongDouble);
        # 调用 Dragon4 算法计算长双精度浮点数的字符串表示，并返回结果
        return Dragon4_Positional_LongDouble_opt(&x, &opt);
    }

    # 将 obj 转换为 Python 浮点数，并将其值赋给 val
    val = PyFloat_AsDouble(obj);
    # 如果在转换过程中出现错误，则返回 NULL
    if (PyErr_Occurred()) {
        return NULL;
    }
    # 调用 Dragon4 算法计算双精度浮点数的字符串表示，并返回结果
    return Dragon4_Positional_Double_opt(&val, &opt);
PyObject *
Dragon4_Scientific(PyObject *obj, DigitMode digit_mode, int precision,
                   int min_digits, int sign, TrimMode trim, int pad_left,
                   int exp_digits)
{
    npy_double val; // 声明一个双精度浮点数变量 val
    Dragon4_Options opt; // 声明 Dragon4_Options 结构体变量 opt

    opt.scientific = 1; // 设置 Dragon4_Options 结构体中 scientific 字段为 1，表示科学计数法
    opt.digit_mode = digit_mode; // 设置 Dragon4_Options 结构体中 digit_mode 字段为传入的 digit_mode 参数
    opt.cutoff_mode = CutoffMode_TotalLength; // 设置 Dragon4_Options 结构体中 cutoff_mode 字段为 CutoffMode_TotalLength
    opt.precision = precision; // 设置 Dragon4_Options 结构体中 precision 字段为传入的 precision 参数
    opt.min_digits = min_digits; // 设置 Dragon4_Options 结构体中 min_digits 字段为传入的 min_digits 参数
    opt.sign = sign; // 设置 Dragon4_Options 结构体中 sign 字段为传入的 sign 参数
    opt.trim_mode = trim; // 设置 Dragon4_Options 结构体中 trim_mode 字段为传入的 trim 参数
    opt.digits_left = pad_left; // 设置 Dragon4_Options 结构体中 digits_left 字段为传入的 pad_left 参数
    opt.digits_right = -1; // 设置 Dragon4_Options 结构体中 digits_right 字段为 -1
    opt.exp_digits = exp_digits; // 设置 Dragon4_Options 结构体中 exp_digits 字段为传入的 exp_digits 参数

    // 检查 obj 是否为 Half、Float、Double 或 LongDouble 类型的标量，并调用相应的 Dragon4_Scientific_*_opt 函数
    if (PyArray_IsScalar(obj, Half)) {
        npy_half x = PyArrayScalar_VAL(obj, Half); // 从 PyArray 标量中提取 npy_half 类型的值 x
        return Dragon4_Scientific_Half_opt(&x, &opt); // 调用 Dragon4_Scientific_Half_opt 函数处理 x 和 opt
    }
    else if (PyArray_IsScalar(obj, Float)) {
        npy_float x = PyArrayScalar_VAL(obj, Float); // 从 PyArray 标量中提取 npy_float 类型的值 x
        return Dragon4_Scientific_Float_opt(&x, &opt); // 调用 Dragon4_Scientific_Float_opt 函数处理 x 和 opt
    }
    else if (PyArray_IsScalar(obj, Double)) {
        npy_double x = PyArrayScalar_VAL(obj, Double); // 从 PyArray 标量中提取 npy_double 类型的值 x
        return Dragon4_Scientific_Double_opt(&x, &opt); // 调用 Dragon4_Scientific_Double_opt 函数处理 x 和 opt
    }
    else if (PyArray_IsScalar(obj, LongDouble)) {
        npy_longdouble x = PyArrayScalar_VAL(obj, LongDouble); // 从 PyArray 标量中提取 npy_longdouble 类型的值 x
        return Dragon4_Scientific_LongDouble_opt(&x, &opt); // 调用 Dragon4_Scientific_LongDouble_opt 函数处理 x 和 opt
    }

    val = PyFloat_AsDouble(obj); // 将 Python 对象 obj 转换为双精度浮点数，并赋值给 val
    if (PyErr_Occurred()) { // 检查是否发生了 Python 异常
        return NULL; // 若有异常发生，返回空指针
    }
    return Dragon4_Scientific_Double_opt(&val, &opt); // 调用 Dragon4_Scientific_Double_opt 函数处理 val 和 opt，并返回结果
}

#undef DEBUG_ASSERT
```