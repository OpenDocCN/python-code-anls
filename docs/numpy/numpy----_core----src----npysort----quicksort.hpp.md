# `.\numpy\numpy\_core\src\npysort\quicksort.hpp`

```py
#ifndef NUMPY_SRC_COMMON_NPYSORT_QUICKSORT_HPP
#define NUMPY_SRC_COMMON_NPYSORT_QUICKSORT_HPP

#include "heapsort.hpp"
#include "common.hpp"

namespace np::sort {

// 栈的大小限制，以存储最大分区，上限为log2(n)个空间
constexpr size_t kQuickStack = sizeof(intptr_t) * 8 * 2;
// 当数组长度小于等于该值时，使用插入排序
constexpr ptrdiff_t kQuickSmall = 15;

// 数值排序
template <typename T>
inline void Quick(T *start, SSize num)
{
    T vp;              // 中值
    T *pl = start;     // 左指针
    T *pr = pl + num - 1; // 右指针
    T *stack[kQuickStack]; // 用于存储分区的栈
    T **sptr = stack;  // 栈指针
    T *pm, *pi, *pj, *pk; // 中间指针和临时指针
    int depth[kQuickStack]; // 存储深度的数组
    int *psdepth = depth;   // 深度指针
    int cdepth = BitScanReverse(static_cast<std::make_unsigned_t<SSize>>(num)) * 2; // 初始深度估算

    for (;;) {
        if (NPY_UNLIKELY(cdepth < 0)) {
            // 如果深度小于0，切换到堆排序
            Heap(pl, pr - pl + 1);
            goto stack_pop;
        }
        while ((pr - pl) > kQuickSmall) {
            // 快速排序分区
            pm = pl + ((pr - pl) >> 1); // 中间元素
            if (LessThan(*pm, *pl)) {
                std::swap(*pm, *pl);
            }
            if (LessThan(*pr, *pm)) {
                std::swap(*pr, *pm);
            }
            if (LessThan(*pm, *pl)) {
                std::swap(*pm, *pl);
            }
            vp = *pm; // 中值更新
            pi = pl;
            pj = pr - 1;
            std::swap(*pm, *pj);
            for (;;) {
                do {
                    ++pi;
                } while (LessThan(*pi, vp));
                do {
                    --pj;
                } while (LessThan(vp, *pj));
                if (pi >= pj) {
                    break;
                }
                std::swap(*pi, *pj);
            }
            pk = pr - 1;
            std::swap(*pi, *pk);
            // 将较大的分区压入栈中
            if (pi - pl < pr - pi) {
                *sptr++ = pi + 1;
                *sptr++ = pr;
                pr = pi - 1;
            }
            else {
                *sptr++ = pl;
                *sptr++ = pi - 1;
                pl = pi + 1;
            }
            *psdepth++ = --cdepth; // 减小深度
        }

        /* 插入排序 */
        for (pi = pl + 1; pi <= pr; ++pi) {
            vp = *pi;
            pj = pi;
            pk = pi - 1;
            while (pj > pl && LessThan(vp, *pk)) {
                *pj-- = *pk--;
            }
            *pj = vp;
        }

    stack_pop:
        if (sptr == stack) {
            break;
        }
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }
}
} // np::sort
#endif // NUMPY_SRC_COMMON_NPYSORT_QUICK_HPP
```