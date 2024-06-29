# `.\numpy\numpy\_core\src\npysort\radixsort.cpp`

```
/*
 *****************************************************************************
 **                            INTEGER SORTS                                **
 *****************************************************************************
 */

// 引入相关头文件和库

#include "npy_sort.h"
#include "npysort_common.h"

// 引入 NumPy 相关头文件
#include "../common/numpy_tag.h"

// 引入 C 标准库头文件
#include <cstdlib>

// 引入类型特性头文件，用于类型判断
#include <type_traits>

// 定义宏，避免使用已弃用的 NumPy API
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

// 基于模板的 KEY_OF 函数，用于提取排序键值
template <class T, class UT>
UT
KEY_OF(UT x)
{
    // 如果 T 是浮点型，则禁用浮点运算
    // 在 macOS 上，双精度和单精度浮点数的排序测试成功，但在 Windows/Linux 上失败。
    // 基本排序测试成功，但依赖排序失败。可能与浮点数的规范化或多个 NaN 表示有关，尚不清楚。
    if (std::is_floating_point<T>::value) {
        // 对于浮点数，如果符号位设置了，则反转键值；否则反转符号位。
        return ((x) ^ (-((x) >> (sizeof(T) * 8 - 1)) |
                       ((UT)1 << (sizeof(T) * 8 - 1))));
    }
    else if (std::is_signed<T>::value) {
        // 对于有符号整数，翻转符号位，使负数排在正数之前。
        return ((x) ^ ((UT)1 << (sizeof(UT) * 8 - 1)));
    }
    else {
        return x;
    }
}

// 提取整数类型 T 的第 l 个字节
template <class T>
static inline npy_ubyte
nth_byte(T key, npy_intp l)
{
    return (key >> (l << 3)) & 0xFF;
}

// 基数排序的核心函数，排序整数类型为 UT 的数组
template <class T, class UT>
static UT *
radixsort0(UT *start, UT *aux, npy_intp num)
{
    // 计数数组，用于存储每个字节的计数
    npy_intp cnt[sizeof(UT)][1 << 8] = {{0}};

    // 获取第一个元素的键值
    UT key0 = KEY_OF<T>(start[0]);

    // 计算每个元素的键值，并在计数数组中增加对应位置的计数
    for (npy_intp i = 0; i < num; i++) {
        UT k = KEY_OF<T>(start[i]);

        // 对每个字节进行计数
        for (size_t l = 0; l < sizeof(UT); l++) {
            cnt[l][nth_byte(k, l)]++;
        }
    }

    // 记录需要排序的字节列
    size_t ncols = 0;
    npy_ubyte cols[sizeof(UT)];
    for (size_t l = 0; l < sizeof(UT); l++) {
        if (cnt[l][nth_byte(key0, l)] != num) {
            cols[ncols++] = l;
        }
    }

    // 对每个需要排序的字节列进行排序
    for (size_t l = 0; l < ncols; l++) {
        npy_intp a = 0;
        // 将计数数组转换为每个值的起始索引
        for (npy_intp i = 0; i < 256; i++) {
            npy_intp b = cnt[cols[l]][i];
            cnt[cols[l]][i] = a;
            a += b;
        }
    }

    // 使用辅助数组进行排序
    for (size_t l = 0; l < ncols; l++) {
        UT *temp;
        for (npy_intp i = 0; i < num; i++) {
            UT k = KEY_OF<T>(start[i]);
            npy_intp dst = cnt[cols[l]][nth_byte(k, cols[l])]++;
            aux[dst] = start[i];
        }

        // 交换排序结果和原始数组
        temp = aux;
        aux = start;
        start = temp;
    }

    return start;
}

// 基数排序的接口函数，排序整数类型为 UT 的数组
template <class T, class UT>
static int
radixsort_(UT *start, npy_intp num)
{
    // 如果数组元素小于 2，无需排序
    if (num < 2) {
        return 0;
    }

    // 检查数组是否已经有序
    npy_bool all_sorted = 1;
    UT k1 = KEY_OF<T>(start[0]);
    for (npy_intp i = 1; i < num; i++) {
        UT k2 = KEY_OF<T>(start[i]);
        if (k1 > k2) {
            all_sorted = 0;
            break;
        }
        k1 = k2;
    }

    // 如果数组已经有序，无需排序
    if (all_sorted) {
        return 0;
    }

    // 继续进行基数排序
    // （后续代码未提供完整，无法提供更多注释）
    # 分配内存以存储 num 个 UT 类型的对象，并将指针赋给 aux
    UT *aux = (UT *)malloc(num * sizeof(UT));
    
    # 检查内存分配是否成功，如果失败则返回内存不足错误码
    if (aux == nullptr) {
        return -NPY_ENOMEM;
    }
    
    # 对给定的数组 start 进行基数排序，使用 aux 作为辅助数组，排序长度为 num
    UT *sorted = radixsort0<T>(start, aux, num);
    
    # 如果排序后的数组 sorted 与原始数组 start 不同，将排序结果拷贝回 start
    if (sorted != start) {
        memcpy(start, sorted, num * sizeof(UT));
    }
    
    # 释放先前分配的辅助数组 aux 的内存
    free(aux);
    
    # 返回成功状态码
    return 0;
}

template <class T>
static int
radixsort(void *start, npy_intp num)
{
    // 使用std::make_unsigned<T>::type定义无符号整数类型UT
    using UT = typename std::make_unsigned<T>::type;
    // 调用具体的模板函数radixsort_<T>进行排序
    return radixsort_<T>((UT *)start, num);
}

template <class T, class UT>
static npy_intp *
aradixsort0(UT *start, npy_intp *aux, npy_intp *tosort, npy_intp num)
{
    // 定义存储计数的数组cnt，初始化为0
    npy_intp cnt[sizeof(UT)][1 << 8] = {{0}};
    // 获取起始元素的关键字值作为比较标准
    UT key0 = KEY_OF<T>(start[0]);

    // 遍历整个数组，统计每个字节的出现次数
    for (npy_intp i = 0; i < num; i++) {
        UT k = KEY_OF<T>(start[i]);
        // 按字节分别统计每个字节的出现次数
        for (size_t l = 0; l < sizeof(UT); l++) {
            cnt[l][nth_byte(k, l)]++;
        }
    }

    // 统计不同字节的数量，存储在cols数组中
    size_t ncols = 0;
    npy_ubyte cols[sizeof(UT)];
    for (size_t l = 0; l < sizeof(UT); l++) {
        // 如果某字节的出现次数不等于num，则将其加入到cols数组中
        if (cnt[l][nth_byte(key0, l)] != num) {
            cols[ncols++] = l;
        }
    }

    // 根据统计信息，计算每个字节的累积出现次数
    for (size_t l = 0; l < ncols; l++) {
        npy_intp a = 0;
        for (npy_intp i = 0; i < 256; i++) {
            npy_intp b = cnt[cols[l]][i];
            cnt[cols[l]][i] = a;
            a += b;
        }
    }

    // 根据字节信息对数组进行排序
    for (size_t l = 0; l < ncols; l++) {
        npy_intp *temp;
        for (npy_intp i = 0; i < num; i++) {
            UT k = KEY_OF<T>(start[tosort[i]]);
            // 计算排序后的位置，并将其存储在aux数组中
            npy_intp dst = cnt[cols[l]][nth_byte(k, cols[l])]++;
            aux[dst] = tosort[i];
        }

        // 交换aux和tosort数组的指针
        temp = aux;
        aux = tosort;
        tosort = temp;
    }

    // 返回排序后的tosort数组
    return tosort;
}

template <class T, class UT>
static int
aradixsort_(UT *start, npy_intp *tosort, npy_intp num)
{
    npy_intp *sorted;
    npy_intp *aux;
    UT k1, k2;
    npy_bool all_sorted = 1;

    // 如果数组长度小于2，无需排序，直接返回
    if (num < 2) {
        return 0;
    }

    // 获取第一个元素的关键字值作为初始比较标准
    k1 = KEY_OF<T>(start[tosort[0]]);
    // 遍历数组，检查是否已经有序
    for (npy_intp i = 1; i < num; i++) {
        k2 = KEY_OF<T>(start[tosort[i]]);
        if (k1 > k2) {
            all_sorted = 0;
            break;
        }
        k1 = k2;
    }

    // 如果已经有序，则无需进行排序，直接返回
    if (all_sorted) {
        return 0;
    }

    // 分配辅助数组aux的空间
    aux = (npy_intp *)malloc(num * sizeof(npy_intp));
    if (aux == NULL) {
        // 内存分配失败，返回错误码
        return -NPY_ENOMEM;
    }

    // 调用aradixsort0函数进行实际的排序操作
    sorted = aradixsort0<T>(start, aux, tosort, num);
    // 如果排序后的结果不是初始的tosort数组，则将结果拷贝回tosort数组
    if (sorted != tosort) {
        memcpy(tosort, sorted, num * sizeof(npy_intp));
    }

    // 释放辅助数组aux的内存空间
    free(aux);
    // 返回排序操作的结果
    return 0;
}

template <class T>
static int
aradixsort(void *start, npy_intp *tosort, npy_intp num)
{
    // 使用std::make_unsigned<T>::type定义无符号整数类型UT
    using UT = typename std::make_unsigned<T>::type;
    // 调用aradixsort_函数进行排序
    return aradixsort_<T>((UT *)start, tosort, num);
}

extern "C" {
NPY_NO_EXPORT int
radixsort_bool(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    // 调用radixsort函数对bool类型的数组进行排序
    return radixsort<npy_bool>(vec, cnt);
}
NPY_NO_EXPORT int
radixsort_byte(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    // 调用radixsort函数对byte类型的数组进行排序
    return radixsort<npy_byte>(vec, cnt);
}
NPY_NO_EXPORT int
radixsort_ubyte(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    // 调用radixsort函数对ubyte类型的数组进行排序
    return radixsort<npy_ubyte>(vec, cnt);
}
NPY_NO_EXPORT int
radixsort_short(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    // 调用radixsort函数对short类型的数组进行排序
    return radixsort<npy_short>(vec, cnt);
}
NPY_NO_EXPORT int
radixsort_ushort(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    // 调用radixsort函数对ushort类型的数组进行排序
    return radixsort<npy_ushort>(vec, cnt);
}
NPY_NO_EXPORT int
# 对整数类型进行基数排序的函数，接受一个指针参数和一个整数参数作为输入
radixsort_int(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    # 调用模板函数 radixsort，使用 npy_int 类型进行排序，并返回结果
    return radixsort<npy_int>(vec, cnt);
}

# 对无符号整数类型进行基数排序的函数，接受一个指针参数和一个整数参数作为输入
NPY_NO_EXPORT int
radixsort_uint(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    # 调用模板函数 radixsort，使用 npy_uint 类型进行排序，并返回结果
    return radixsort<npy_uint>(vec, cnt);
}

# 对长整型进行基数排序的函数，接受一个指针参数和一个整数参数作为输入
NPY_NO_EXPORT int
radixsort_long(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    # 调用模板函数 radixsort，使用 npy_long 类型进行排序，并返回结果
    return radixsort<npy_long>(vec, cnt);
}

# 对无符号长整型进行基数排序的函数，接受一个指针参数和一个整数参数作为输入
NPY_NO_EXPORT int
radixsort_ulong(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    # 调用模板函数 radixsort，使用 npy_ulong 类型进行排序，并返回结果
    return radixsort<npy_ulong>(vec, cnt);
}

# 对长长整型进行基数排序的函数，接受一个指针参数和一个整数参数作为输入
NPY_NO_EXPORT int
radixsort_longlong(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    # 调用模板函数 radixsort，使用 npy_longlong 类型进行排序，并返回结果
    return radixsort<npy_longlong>(vec, cnt);
}

# 对无符号长长整型进行基数排序的函数，接受一个指针参数和一个整数参数作为输入
NPY_NO_EXPORT int
radixsort_ulonglong(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    # 调用模板函数 radixsort，使用 npy_ulonglong 类型进行排序，并返回结果
    return radixsort<npy_ulonglong>(vec, cnt);
}

# 对布尔型数组进行基数排序的函数，接受一个指针参数和两个整数参数作为输入
NPY_NO_EXPORT int
aradixsort_bool(void *vec, npy_intp *ind, npy_intp cnt, void *NPY_UNUSED(null))
{
    # 调用模板函数 aradixsort，使用 npy_bool 类型进行排序，并返回结果
    return aradixsort<npy_bool>(vec, ind, cnt);
}

# 对字节类型数组进行基数排序的函数，接受一个指针参数和两个整数参数作为输入
NPY_NO_EXPORT int
aradixsort_byte(void *vec, npy_intp *ind, npy_intp cnt, void *NPY_UNUSED(null))
{
    # 调用模板函数 aradixsort，使用 npy_byte 类型进行排序，并返回结果
    return aradixsort<npy_byte>(vec, ind, cnt);
}

# 对无符号字节类型数组进行基数排序的函数，接受一个指针参数和两个整数参数作为输入
NPY_NO_EXPORT int
aradixsort_ubyte(void *vec, npy_intp *ind, npy_intp cnt,
                 void *NPY_UNUSED(null))
{
    # 调用模板函数 aradixsort，使用 npy_ubyte 类型进行排序，并返回结果
    return aradixsort<npy_ubyte>(vec, ind, cnt);
}

# 对短整型数组进行基数排序的函数，接受一个指针参数和两个整数参数作为输入
NPY_NO_EXPORT int
aradixsort_short(void *vec, npy_intp *ind, npy_intp cnt,
                 void *NPY_UNUSED(null))
{
    # 调用模板函数 aradixsort，使用 npy_short 类型进行排序，并返回结果
    return aradixsort<npy_short>(vec, ind, cnt);
}

# 对无符号短整型数组进行基数排序的函数，接受一个指针参数和两个整数参数作为输入
NPY_NO_EXPORT int
aradixsort_ushort(void *vec, npy_intp *ind, npy_intp cnt,
                  void *NPY_UNUSED(null))
{
    # 调用模板函数 aradixsort，使用 npy_ushort 类型进行排序，并返回结果
    return aradixsort<npy_ushort>(vec, ind, cnt);
}

# 对整数类型数组进行基数排序的函数，接受一个指针参数和两个整数参数作为输入
NPY_NO_EXPORT int
aradixsort_int(void *vec, npy_intp *ind, npy_intp cnt, void *NPY_UNUSED(null))
{
    # 调用模板函数 aradixsort，使用 npy_int 类型进行排序，并返回结果
    return aradixsort<npy_int>(vec, ind, cnt);
}

# 对无符号整数类型数组进行基数排序的函数，接受一个指针参数和两个整数参数作为输入
NPY_NO_EXPORT int
aradixsort_uint(void *vec, npy_intp *ind, npy_intp cnt, void *NPY_UNUSED(null))
{
    # 调用模板函数 aradixsort，使用 npy_uint 类型进行排序，并返回结果
    return aradixsort<npy_uint>(vec, ind, cnt);
}

# 对长整型数组进行基数排序的函数，接受一个指针参数和两个整数参数作为输入
NPY_NO_EXPORT int
aradixsort_long(void *vec, npy_intp *ind, npy_intp cnt, void *NPY_UNUSED(null))
{
    # 调用模板函数 aradixsort，使用 npy_long 类型进行排序，并返回结果
    return aradixsort<npy_long>(vec, ind, cnt);
}

# 对无符号长整型数组进行基数排序的函数，接受一个指针参数和两个整数参数作为输入
NPY_NO_EXPORT int
aradixsort_ulong(void *vec, npy_intp *ind, npy_intp cnt,
                 void *NPY_UNUSED(null))
{
    # 调用模板函数 aradixsort，使用 npy_ulong 类型进行排序，并返回结果
    return aradixsort<npy_ulong>(vec, ind, cnt);
}

# 对长长整型数组进行基数排序的函数，接受一个指针参数和两个整数参数作为输入
NPY_NO_EXPORT int
aradixsort_longlong(void *vec, npy_intp *ind, npy_intp cnt,
                    void *NPY_UNUSED(null))
{
    # 调用模板函数 aradixsort，使用 npy_longlong 类型进行排序，并返回结果
    return aradixsort<npy_longlong>(vec, ind, cnt);
}

# 对无符号长长整型数组进行基数排序的函数，接受一个指针参数和两个整数参数作为输入
NPY_NO_EXPORT int
aradixsort_ulonglong(void *vec, npy_intp *ind, npy_intp cnt,
                     void *NPY_UNUSED(null))
{
    # 调用模板函数 aradixsort，使用 npy_ulonglong 类型进行排序，并返回结果
    return aradixsort<npy_ulonglong>(vec, ind, cnt);
}
```