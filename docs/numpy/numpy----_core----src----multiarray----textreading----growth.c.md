# `.\numpy\numpy\_core\src\multiarray\textreading\growth.c`

```
/*
 * 宏定义，用于设置 NumPy 不使用过时的 API 版本，使用当前的 API 版本
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

/*
 * 宏定义，指定当前文件是多维数组模块的一部分
 */
#define _MULTIARRAYMODULE

/*
 * 包含 NumPy 头文件，用于获取 ndarray 类型相关的定义
 */
#include "numpy/ndarraytypes.h"

/*
 * 包含模板共享的头文件 templ_common.h
 */
#include "templ_common.h"

/*
 * 辅助函数：根据输入的大小信息和最小增长值来计算新的大小。
 * 当前方案是通过一定比例的增长（25%）来进行动态扩展，并且限制在 2**20 个元素以内，
 * 因为这使得我们处于大页大小的范围内（通常已经足够）。
 *
 * 进一步将新的大小乘以每个元素的大小 itemsize，并确保所有的结果适合于 npy_intp 类型。
 * 如果发生溢出或者结果无法适应，则返回 -1。
 * 调用者需要确保输入的 size 是 ssize_t 类型且不为负数。
 */
NPY_NO_EXPORT npy_intp
grow_size_and_multiply(npy_intp *size, npy_intp min_grow, npy_intp itemsize) {
    /* min_grow 必须是二的幂：*/
    assert((min_grow & (min_grow - 1)) == 0);

    // 将 size 转换为无符号整数类型
    npy_uintp new_size = (npy_uintp)*size;

    // 计算增长量，初始增长为当前大小的四分之一
    npy_intp growth = *size >> 2;

    // 如果增长量小于等于 min_grow，则使用 min_grow
    if (growth <= min_grow) {
        new_size += min_grow;
    }
    else {
        // 如果增长量大于 1 << 20，则限制在 1 << 20 范围内
        if (growth > 1 << 20) {
            growth = 1 << 20;
        }

        // 计算新的大小，保证是 min_grow 的倍数，并且不超过 NPY_MAX_INTP
        new_size += growth + min_grow - 1;
        new_size &= ~min_grow;

        if (new_size > NPY_MAX_INTP) {
            // 如果超过了 npy_intp 的最大值，则返回 -1 表示溢出
            return -1;
        }
    }

    // 将计算后的新大小赋值给 size
    *size = (npy_intp)new_size;

    // 计算最终的分配大小，即 new_size 乘以 itemsize
    npy_intp alloc_size;
    if (npy_mul_sizes_with_overflow(&alloc_size, (npy_intp)new_size, itemsize)) {
        // 如果乘法溢出，则返回 -1
        return -1;
    }

    // 返回最终的分配大小
    return alloc_size;
}
```