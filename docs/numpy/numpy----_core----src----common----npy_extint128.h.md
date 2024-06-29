# `.\numpy\numpy\_core\src\common\npy_extint128.h`

```py
/* 
   定义一个条件编译保护，防止多次包含本头文件
*/
#ifndef NUMPY_CORE_SRC_COMMON_NPY_EXTINT128_H_
#define NUMPY_CORE_SRC_COMMON_NPY_EXTINT128_H_

/* 
   定义一个结构体 npy_extint128_t，用于表示一个128位扩展整数，
   包括符号位和两个64位的无符号整数部分
*/
typedef struct {
    signed char sign;   // 符号位，1表示正数，-1表示负数
    npy_uint64 lo, hi;  // 低位和高位的64位无符号整数部分
} npy_extint128_t;

/* 
   定义一个静态内联函数，实现安全的64位整数加法，同时检查是否溢出
*/
static inline npy_int64
safe_add(npy_int64 a, npy_int64 b, char *overflow_flag)
{
    if (a > 0 && b > NPY_MAX_INT64 - a) {
        *overflow_flag = 1;  // 若加法溢出，则设置溢出标志
    }
    else if (a < 0 && b < NPY_MIN_INT64 - a) {
        *overflow_flag = 1;  // 若加法溢出，则设置溢出标志
    }
    return a + b;  // 返回加法结果
}

/* 
   定义一个静态内联函数，实现安全的64位整数减法，同时检查是否溢出
*/
static inline npy_int64
safe_sub(npy_int64 a, npy_int64 b, char *overflow_flag)
{
    if (a >= 0 && b < a - NPY_MAX_INT64) {
        *overflow_flag = 1;  // 若减法溢出，则设置溢出标志
    }
    else if (a < 0 && b > a - NPY_MIN_INT64) {
        *overflow_flag = 1;  // 若减法溢出，则设置溢出标志
    }
    return a - b;  // 返回减法结果
}

/* 
   定义一个静态内联函数，实现安全的64位整数乘法，同时检查是否溢出
*/
static inline npy_int64
safe_mul(npy_int64 a, npy_int64 b, char *overflow_flag)
{
    if (a > 0) {
        if (b > NPY_MAX_INT64 / a || b < NPY_MIN_INT64 / a) {
            *overflow_flag = 1;  // 若乘法溢出，则设置溢出标志
        }
    }
    else if (a < 0) {
        if (b > 0 && a < NPY_MIN_INT64 / b) {
            *overflow_flag = 1;  // 若乘法溢出，则设置溢出标志
        }
        else if (b < 0 && a < NPY_MAX_INT64 / b) {
            *overflow_flag = 1;  // 若乘法溢出，则设置溢出标志
        }
    }
    return a * b;  // 返回乘法结果
}

/* 
   定义一个静态内联函数，将64位有符号整数转换为128位扩展整数
*/
static inline npy_extint128_t
to_128(npy_int64 x)
{
    npy_extint128_t result;
    result.sign = (x >= 0 ? 1 : -1);  // 确定符号位
    if (x >= 0) {
        result.lo = x;  // 若为正数，直接赋值给低位部分
    }
    else {
        result.lo = (npy_uint64)(-(x + 1)) + 1;  // 若为负数，转换为补码形式
    }
    result.hi = 0;  // 高位部分清零
    return result;  // 返回128位扩展整数结果
}

/* 
   定义一个静态内联函数，将128位扩展整数转换为64位有符号整数
*/
static inline npy_int64
to_64(npy_extint128_t x, char *overflow)
{
    if (x.hi != 0 ||
        (x.sign > 0 && x.lo > NPY_MAX_INT64) ||
        (x.sign < 0 && x.lo != 0 && x.lo - 1 > -(NPY_MIN_INT64 + 1))) {
        *overflow = 1;  // 若转换溢出，则设置溢出标志
    }
    return x.lo * x.sign;  // 返回64位整数结果
}

/* 
   定义一个静态内联函数，实现两个64位整数的乘法，返回128位扩展整数结果
*/
static inline npy_extint128_t
mul_64_64(npy_int64 a, npy_int64 b)
{
    npy_extint128_t x, y, z;
    npy_uint64 x1, x2, y1, y2, r1, r2, prev;

    x = to_128(a);  // 将64位整数a转换为128位扩展整数
    y = to_128(b);  // 将64位整数b转换为128位扩展整数

    x1 = x.lo & 0xffffffff;  // 获取x的低32位
    x2 = x.lo >> 32;         // 获取x的高32位

    y1 = y.lo & 0xffffffff;  // 获取y的低32位
    y2 = y.lo >> 32;         // 获取y的高32位

    r1 = x1 * y2;            // 计算低32位乘积
    r2 = x2 * y1;            // 计算高32位乘积

    z.sign = x.sign * y.sign;  // 计算结果的符号位
    z.hi = x2 * y2 + (r1 >> 32) + (r2 >> 32);  // 计算结果的高64位
    z.lo = x1 * y1;           // 计算结果的低64位

    /* 进行加法运算，处理可能的进位 */
    prev = z.lo;
    z.lo += (r1 << 32);
    if (z.lo < prev) {
        ++z.hi;  // 若低位相加溢出，则高位加1
    }

    prev = z.lo;
    z.lo += (r2 << 32);
    if (z.lo < prev) {
        ++z.hi;  // 若低位相加溢出，则高位加1
    }

    return z;  // 返回128位扩展整数结果
}

/* 
   定义一个静态内联函数，实现两个128位扩展整数的加法，返回128位扩展整数结果
*/
static inline npy_extint128_t
add_128(npy_extint128_t x, npy_extint128_t y, char *overflow)
{
    npy_extint128_t z;

    if (x.sign == y.sign) {
        z.sign = x.sign;  // 若符号相同，则结果符号不变
        z.hi = x.hi + y.hi;  // 直接相加高位部分
        if (z.hi < x.hi) {
            *overflow = 1;  // 若高位相加溢出，则设置溢出标志
        }
        z.lo = x.lo + y.lo;  // 直接相加低位部分
        if (z.lo < x.lo) {
            if (z.hi == NPY_MAX_UINT64) {
                *overflow = 1;  // 若低位相加溢出，并且高位已经最大，则设置溢出标志
            }
            ++z.hi;  // 若低位相加溢出，则高位加1
        }
    }
    # 如果 x 的高位大于 y 的高位，或者当高位相等时 x 的低位大于等于 y 的低位，则执行以下代码块
    else if (x.hi > y.hi || (x.hi == y.hi && x.lo >= y.lo)) {
        # 结果 z 继承 x 的符号位
        z.sign = x.sign;
        # 计算 z 的高位，为 x 的高位减去 y 的高位
        z.hi = x.hi - y.hi;
        # 计算 z 的低位，为 x 的低位
        z.lo = x.lo;
        # 从 z 的低位中减去 y 的低位
        z.lo -= y.lo;
        # 如果 z 的低位结果大于 x 的低位，需要借位
        if (z.lo > x.lo) {
            # 减少 z 的高位
            --z.hi;
        }
    }
    # 如果上述条件不成立，则执行以下代码块
    else {
        # 结果 z 继承 y 的符号位
        z.sign = y.sign;
        # 计算 z 的高位，为 y 的高位减去 x 的高位
        z.hi = y.hi - x.hi;
        # 计算 z 的低位，为 y 的低位
        z.lo = y.lo;
        # 从 z 的低位中减去 x 的低位
        z.lo -= x.lo;
        # 如果 z 的低位结果大于 y 的低位，需要借位
        if (z.lo > y.lo) {
            # 减少 z 的高位
            --z.hi;
        }
    }

    # 返回计算结果 z
    return z;
/* 长整数取负操作 */
static inline npy_extint128_t
neg_128(npy_extint128_t x)
{
    // 复制输入的参数x到新变量z
    npy_extint128_t z = x;
    // 改变z的符号位
    z.sign *= -1;
    // 返回符号改变后的z
    return z;
}


/* 长整数减法操作 */
static inline npy_extint128_t
sub_128(npy_extint128_t x, npy_extint128_t y, char *overflow)
{
    // 调用add_128函数实现x - y操作，并返回结果
    return add_128(x, neg_128(y), overflow);
}


/* 长整数左移操作 */
static inline npy_extint128_t
shl_128(npy_extint128_t v)
{
    // 将输入的参数v赋值给新变量z
    npy_extint128_t z;
    z = v;
    // 高64位左移1位
    z.hi <<= 1;
    // 将低64位最高位移到高64位的最低位
    z.hi |= (z.lo & (((npy_uint64)1) << 63)) >> 63;
    // 低64位左移1位
    z.lo <<= 1;
    // 返回左移后的结果z
    return z;
}


/* 长整数右移操作 */
static inline npy_extint128_t
shr_128(npy_extint128_t v)
{
    // 将输入的参数v赋值给新变量z
    npy_extint128_t z;
    z = v;
    // 低64位右移1位
    z.lo >>= 1;
    // 将高64位的最低位移到低64位的最高位
    z.lo |= (z.hi & 0x1) << 63;
    // 高64位右移1位
    z.hi >>= 1;
    // 返回右移后的结果z
    return z;
}

/* 长整数比较操作，返回大于的判断结果 */
static inline int
gt_128(npy_extint128_t a, npy_extint128_t b)
{
    // 如果a和b的符号都为正数
    if (a.sign > 0 && b.sign > 0) {
        // 比较a和b的高64位和低64位
        return (a.hi > b.hi) || (a.hi == b.hi && a.lo > b.lo);
    }
    // 如果a和b的符号都为负数
    else if (a.sign < 0 && b.sign < 0) {
        // 比较a和b的高64位和低64位
        return (a.hi < b.hi) || (a.hi == b.hi && a.lo < b.lo);
    }
    // 如果a为正数，b为负数
    else if (a.sign > 0 && b.sign < 0) {
        // 判断a和b是否不同时为零
        return a.hi != 0 || a.lo != 0 || b.hi != 0 || b.lo != 0;
    }
    // 其他情况返回0
    else {
        return 0;
    }
}


/* 长整数除法操作 */
static inline npy_extint128_t
divmod_128_64(npy_extint128_t x, npy_int64 b, npy_int64 *mod)
{
    // 声明变量remainder, pointer, result, divisor，并初始化overflow为0
    npy_extint128_t remainder, pointer, result, divisor;
    char overflow = 0;

    // 断言b大于0
    assert(b > 0);

    // 如果b小于等于1或者x的高64位为0
    if (b <= 1 || x.hi == 0) {
        // 设置result的符号和低高64位
        result.sign = x.sign;
        result.lo = x.lo / b;
        result.hi = x.hi / b;
        // 计算余数并赋值给mod
        *mod = x.sign * (x.lo % b);
        // 返回计算结果result
        return result;
    }

    // 长除法计算，不是最有效率的选择
    // 将x赋值给remainder，并将其符号置为正数
    remainder = x;
    remainder.sign = 1;

    // 设置divisor为正数，低64位为b，高64位为0
    divisor.sign = 1;
    divisor.hi = 0;
    divisor.lo = b;

    // 设置result为正数，低高64位为0
    result.sign = 1;
    result.lo = 0;
    result.hi = 0;

    // 设置pointer为正数，低64位为1，高64位为0
    pointer.sign = 1;
    pointer.lo = 1;
    pointer.hi = 0;

    // 当divisor的高64位的最低位为0且remainder大于divisor时
    while ((divisor.hi & (((npy_uint64)1) << 63)) == 0 &&
           gt_128(remainder, divisor)) {
        // divisor和pointer分别左移一位
        divisor = shl_128(divisor);
        pointer = shl_128(pointer);
    }

    // 当pointer的低64位或高64位不为0时
    while (pointer.lo || pointer.hi) {
        // 如果remainder不小于divisor
        if (!gt_128(divisor, remainder)) {
            // remainder减去divisor，并将结果加到result中
            remainder = sub_128(remainder, divisor, &overflow);
            result = add_128(result, pointer, &overflow);
        }
        // divisor右移一位，pointer右移一位
        divisor = shr_128(divisor);
        pointer = shr_128(pointer);
    }

    // 修正结果的符号并返回，不会溢出
    result.sign = x.sign;
    // 将remainder的低64位乘以x的符号并赋值给mod
    *mod = x.sign * remainder.lo;

    // 返回计算结果result
    return result;
}


/* 除法并向下取整（正数除数；不会溢出） */
static inline npy_extint128_t
floordiv_128_64(npy_extint128_t a, npy_int64 b)
{
    // 声明变量result, remainder，并初始化overflow为0
    npy_extint128_t result;
    npy_int64 remainder;
    char overflow = 0;
    // 断言b大于0
    assert(b > 0);
    // 调用divmod_128_64函数计算a除以b的商和余数，结果赋值给result
    result = divmod_128_64(a, b, &remainder);
    // 如果a为负数且余数不为0，将result减去1
    if (a.sign < 0 && remainder != 0) {
        result = sub_128(result, to_128(1), &overflow);
    }
    // 返回计算结果result
    return result;
}


/* 除法并向上取整（正数除数；不会溢出） */
static inline npy_extint128_t
ceildiv_128_64(npy_extint128_t a, npy_int64 b)
{
    // 声明变量result, remainder，并初始化overflow为0
    npy_extint128_t result;
    npy_int64 remainder;
    // 断言b大于0
    assert(b > 0);
    // 调用divmod_128_64函数计算a除以b的商和余数，结果赋值给result
    result = divmod_128_64(a, b, &remainder);
    // 如果余数不为0，将result加上1
    if (remainder != 0) {
        result = add_128(result, to_128(1), NULL);
    }
    // 返回计算结果result
    return result;
}
    # 声明一个字符型变量 overflow，并初始化为 0
    char overflow = 0;
    # 断言 b 大于 0，确保除数 b 是一个正数
    assert(b > 0);
    # 调用函数 divmod_128_64 对 a 和 b 进行128位和64位的除法运算，并将余数存入 remainder 指向的地址中，返回商 result
    result = divmod_128_64(a, b, &remainder);
    # 如果 a 的符号为正，并且余数 remainder 不等于 0，则执行下面的条件
    if (a.sign > 0 && remainder != 0) {
        # 调用函数 add_128 对 result 和一个表示整数1的128位数相加，将溢出标志存入 overflow 指向的地址中，并更新 result
        result = add_128(result, to_128(1), &overflow);
    }
    # 返回最终的 result 结果
    return result;
}
#endif  /* NUMPY_CORE_SRC_COMMON_NPY_EXTINT128_H_ */


// 结束if指令的条件编译块，关闭条件编译的头文件保护宏
}
#endif  /* NUMPY_CORE_SRC_COMMON_NPY_EXTINT128_H_ */
```