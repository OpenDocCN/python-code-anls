# `D:\src\scipysrc\numpy\numpy\_core\src\npysort\heapsort.hpp`

```
#ifndef NUMPY_SRC_COMMON_NPYSORT_HEAPSORT_HPP
#define NUMPY_SRC_COMMON_NPYSORT_HEAPSORT_HPP

#include "common.hpp"

namespace np::sort {

template <typename T>
inline bool LessThan(const T &a, const T &b)
{
    // 比较函数，根据类型 T 的不同，进行不同的比较操作
    if constexpr (std::is_floating_point_v<T>) {
        // 如果 T 是浮点数类型，则考虑 NaN 的情况进行比较
        return a < b || (b != b && a == a);
    }
    else if constexpr(std::is_same_v<T, Half>) {
        // 如果 T 是 Half 类型，则调用其自定义的 Less 方法进行比较
        bool a_nn = !a.IsNaN();
        return b.IsNaN() ? a_nn : a_nn && a.Less(b);
    }
    else {
        // 对于其他类型 T，直接进行常规的小于比较
        return a < b;
    }
}

// NUMERIC SORTS
template <typename T>
inline void Heap(T *start, SSize n)
{
    SSize i, j, l;
    // 堆排序需要将数组的索引偏移一位，以符合堆的索引规则
    T tmp, *a = start - 1;

    // 自上而下建堆过程
    for (l = n >> 1; l > 0; --l) {
        tmp = a[l];
        for (i = l, j = l << 1; j <= n;) {
            if (j < n && LessThan(a[j], a[j + 1])) {
                j += 1;
            }
            if (LessThan(tmp, a[j])) {
                a[i] = a[j];
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        a[i] = tmp;
    }

    // 堆排序的下沉调整过程
    for (; n > 1;) {
        tmp = a[n];
        a[n] = a[1];
        n -= 1;
        for (i = 1, j = 2; j <= n;) {
            if (j < n && LessThan(a[j], a[j + 1])) {
                j++;
            }
            if (LessThan(tmp, a[j])) {
                a[i] = a[j];
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        a[i] = tmp;
    }
}
} // namespace np::sort
#endif
```