# `.\numpy\numpy\_core\src\npysort\quicksort.cpp`

```py
/* -*- c -*- */

/*
 * The purpose of this module is to add faster sort functions
 * that are type-specific.  This is done by altering the
 * function table for the builtin descriptors.
 *
 * These sorting functions are copied almost directly from numarray
 * with a few modifications (complex comparisons compare the imaginary
 * part if the real parts are equal, for example), and the names
 * are changed.
 *
 * The original sorting code is due to Charles R. Harris who wrote
 * it for numarray.
 */

/*
 * Quick sort is usually the fastest, but the worst case scenario is O(N^2) so
 * the code switches to the O(NlogN) worst case heapsort if not enough progress
 * is made on the large side of the two quicksort partitions. This improves the
 * worst case while still retaining the speed of quicksort for the common case.
 * This is variant known as introsort.
 *
 *
 * def introsort(lower, higher, recursion_limit=log2(higher - lower + 1) * 2):
 *   # sort remainder with heapsort if we are not making enough progress
 *   # we arbitrarily choose 2 * log(n) as the cutoff point
 *   if recursion_limit < 0:
 *       heapsort(lower, higher)
 *       return
 *
 *   if lower < higher:
 *      pivot_pos = partition(lower, higher)
 *      # recurse into smaller first and leave larger on stack
 *      # this limits the required stack space
 *      if (pivot_pos - lower > higher - pivot_pos):
 *          quicksort(pivot_pos + 1, higher, recursion_limit - 1)
 *          quicksort(lower, pivot_pos, recursion_limit - 1)
 *      else:
 *          quicksort(lower, pivot_pos, recursion_limit - 1)
 *          quicksort(pivot_pos + 1, higher, recursion_limit - 1)
 *
 *
 * the below code implements this converted to an iteration and as an
 * additional minor optimization skips the recursion depth checking on the
 * smaller partition as it is always less than half of the remaining data and
 * will thus terminate fast enough
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "npy_cpu_features.h"
#include "npy_sort.h"
#include "npysort_common.h"
#include "npysort_heapsort.h"
#include "numpy_tag.h"
#include "x86_simd_qsort.hpp"
#include "highway_qsort.hpp"

#include <cstdlib>
#include <utility>

#define NOT_USED NPY_UNUSED(unused)

/*
 * pushing largest partition has upper bound of log2(n) space
 * we store two pointers each time
 */
#define PYA_QS_STACK (NPY_BITSOF_INTP * 2)
#define SMALL_QUICKSORT 15
#define SMALL_MERGESORT 20
#define SMALL_STRING 16

// Disable AVX512 sorting on CYGWIN until we can figure
// out why it has test failures
template<typename T>
inline bool quicksort_dispatch(T *start, npy_intp num)
{
#if !defined(__CYGWIN__)
    using TF = typename np::meta::FixedWidth<T>::Type;
    // Function pointer declaration for dispatching the sorting algorithm based on type
    void (*dispfunc)(TF*, intptr_t) = nullptr;

    // Determine the appropriate sorting function based on CPU features and type
    dispfunc = np::sort_dispatch<TF>;
    // Call the selected sorting function with the provided start pointer and number of elements
    dispfunc(reinterpret_cast<TF*>(start), num);
    return true;
#else
    // Return false if AVX512 sorting is disabled on CYGWIN
    return false;
#endif
}
    // 检查模板参数 T 的大小是否与 uint16_t 相同
    if (sizeof(T) == sizeof(uint16_t)) {
        // 如果未禁用优化
        #ifndef NPY_DISABLE_OPTIMIZATION
            // 如果定义了 NPY_CPU_AMD64 或 NPY_CPU_X86，表示支持 x86 32位和64位架构
            #if defined(NPY_CPU_AMD64) || defined(NPY_CPU_X86)
                // 根据架构包含相应的头文件，选择对应的 SIMD 快速排序实现
                #include "x86_simd_qsort_16bit.dispatch.h"
                // 通过 CPU 分发调用 SIMD 快速排序模板函数 QSort
                NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::qsort_simd::template QSort, <TF>);
            #else
                // 使用 Highway 库的 SIMD 快速排序实现
                #include "highway_qsort_16bit.dispatch.h"
                // 通过 CPU 分发调用 SIMD 快速排序模板函数 QSort
                NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::highway::qsort_simd::template QSort, <TF>);
            #endif
        #endif
    }
    // 如果 T 的大小与 uint32_t 或 uint64_t 相同
    else if (sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t)) {
        // 如果未禁用优化
        #ifndef NPY_DISABLE_OPTIMIZATION
            // 如果定义了 NPY_CPU_AMD64 或 NPY_CPU_X86，表示支持 x86 32位和64位架构
            #if defined(NPY_CPU_AMD64) || defined(NPY_CPU_X86)
                // 根据架构包含相应的头文件，选择对应的 SIMD 快速排序实现
                #include "x86_simd_qsort.dispatch.h"
                // 通过 CPU 分发调用 SIMD 快速排序模板函数 QSort
                NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::qsort_simd::template QSort, <TF>);
            #else
                // 使用 Highway 库的 SIMD 快速排序实现
                #include "highway_qsort.dispatch.h"
                // 通过 CPU 分发调用 SIMD 快速排序模板函数 QSort
                NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::highway::qsort_simd::template QSort, <TF>);
            #endif
        #endif
    }
    // 如果 dispfunc 不为空指针（即成功选择了 SIMD 快速排序实现）
    if (dispfunc) {
        // 转换起始地址为 TF* 类型，并调用快速排序函数，参数是元素个数
        (*dispfunc)(reinterpret_cast<TF*>(start), static_cast<intptr_t>(num));
        // 返回排序成功
        return true;
    }
    #endif // __CYGWIN__
    // 确保参数未使用的警告不会出现
    (void)start; (void)num; // to avoid unused arg warn
    // 返回 false，表示未执行任何操作
    return false;
}

template<typename T>
inline bool aquicksort_dispatch(T *start, npy_intp* arg, npy_intp num)
{
#if !defined(__CYGWIN__)
    // 定义变量 TF 为 T 类型的固定宽度
    using TF = typename np::meta::FixedWidth<T>::Type;
    // 定义指向排序分发函数的指针，并初始化为 nullptr
    void (*dispfunc)(TF*, npy_intp*, npy_intp) = nullptr;
    // 在非禁用优化的情况下，包含特定的 SIMD 排序分发头文件
    #ifndef NPY_DISABLE_OPTIMIZATION
        #include "x86_simd_argsort.dispatch.h"
    #endif
    // 使用 CPU 分发宏调用排序 SIMD 函数模板 ArgQSort
    NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::qsort_simd::template ArgQSort, <TF>);
    // 如果 dispfunc 不为空，则调用对应的排序函数，返回 true
    if (dispfunc) {
        (*dispfunc)(reinterpret_cast<TF*>(start), arg, num);
        return true;
    }
#endif // __CYGWIN__
    // 确保参数未使用的警告不会出现
    (void)start; (void)arg; (void)num; // to avoid unused arg warn
    // 返回 false，表示未执行任何操作
    return false;
}

/*
 *****************************************************************************
 **                            NUMERIC SORTS                                **
 *****************************************************************************
 */

template <typename Tag, typename type>
static int
quicksort_(type *start, npy_intp num)
{
    // 定义变量 vp 为排序元素类型
    type vp;
    // 定义指向排序起始位置的指针 pl
    type *pl = start;
    // 定义指向排序结束位置的指针 pr
    type *pr = pl + num - 1;
    // 定义排序堆栈数组，用于存储排序分割区间
    type *stack[PYA_QS_STACK];
    // 定义排序堆栈指针，指向当前堆栈位置
    type **sptr = stack;
    // 定义堆栈深度数组，记录每个分割区间的深度
    int depth[PYA_QS_STACK];
    // 定义堆栈深度指针，指向当前深度位置
    int *psdepth = depth;
    // 计算排序元素数目的最高有效位，乘以 2 作为初始深度
    int cdepth = npy_get_msb(num) * 2;

    // 主排序循环
    for (;;) {
        // 如果当前深度小于零，则调用堆排序函数，并跳转至堆栈弹出处理
        if (NPY_UNLIKELY(cdepth < 0)) {
            heapsort_<Tag>(pl, pr - pl + 1);
            goto stack_pop;
        }
        // 当排序区间大于设定的小区间阈值时，执行快速排序分割
        while ((pr - pl) > SMALL_QUICKSORT) {
            /* quicksort partition */
            // 计算中间元素位置 pm
            pm = pl + ((pr - pl) >> 1);
            // 如果中间元素小于起始元素，交换它们
            if (Tag::less(*pm, *pl)) {
                std::swap(*pm, *pl);
            }
            // 如果结束元素小于中间元素，交换它们
            if (Tag::less(*pr, *pm)) {
                std::swap(*pr, *pm);
            }
            // 再次检查并交换中间元素和起始元素
            if (Tag::less(*pm, *pl)) {
                std::swap(*pm, *pl);
            }
            // 将中间元素值赋给 vp
            vp = *pm;
            // 初始化分割指针 pi 和 pj
            pi = pl;
            pj = pr - 1;
            // 交换中间元素和结束元素
            std::swap(*pm, *pj);
            // 快速排序的主循环
            for (;;) {
                // 找到第一个大于等于 vp 的元素 pi
                do {
                    ++pi;
                } while (Tag::less(*pi, vp));
                // 找到第一个小于等于 vp 的元素 pj
                do {
                    --pj;
                } while (Tag::less(vp, *pj));
                // 如果 pi >= pj，则退出循环
                if (pi >= pj) {
                    break;
                }
                // 交换 pi 和 pj 的元素
                std::swap(*pi, *pj);
            }
            // 重新交换 pi 和结束元素的位置
            pk = pr - 1;
            std::swap(*pi, *pk);
            // 将较大的分区推入堆栈
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
            // 记录当前分割区间的深度
            *psdepth++ = --cdepth;
        }

        // 插入排序处理小区间
        for (pi = pl + 1; pi <= pr; ++pi) {
            vp = *pi;
            pj = pi;
            pk = pi - 1;
            // 将大于当前元素的元素后移
            while (pj > pl && Tag::less(vp, *pk)) {
                *pj-- = *pk--;
            }
            // 插入当前元素到正确位置
            *pj = vp;
        }
    # 定义函数 stack_pop，用于从堆栈中弹出元素
    stack_pop:
        # 如果堆栈指针 sptr 指向堆栈的起始位置，则跳出循环
        if (sptr == stack) {
            break;
        }
        # 弹出堆栈顶部的两个元素，分别赋值给 pr 和 pl
        pr = *(--sptr);
        pl = *(--sptr);
        # 弹出保存当前深度的堆栈元素，赋值给 cdepth
        cdepth = *(--psdepth);
    }

    # 函数执行完成，返回 0 表示正常退出
    return 0;
// 静态函数 aquicksort_，实现快速排序算法
template <typename Tag, typename type>
static int
aquicksort_(type *vv, npy_intp *tosort, npy_intp num)
{
    // 初始化指针和变量
    type *v = vv;                    // 指向待排序数组的指针
    type vp;                         // 存储中间值
    npy_intp *pl = tosort;           // 左指针，指向数组的起始位置
    npy_intp *pr = tosort + num - 1; // 右指针，指向数组的末尾位置
    npy_intp *stack[PYA_QS_STACK];   // 存储分区位置的栈
    npy_intp **sptr = stack;         // 栈指针，指向栈顶
    npy_intp *pm, *pi, *pj, *pk, vi; // 用于快速排序的临时变量
    int depth[PYA_QS_STACK];         // 存储递归深度的栈
    int *psdepth = depth;            // 深度指针，指向栈顶
    int cdepth = npy_get_msb(num) * 2; // 计算递归深度的上限

    // 进入无限循环，实现快速排序
    for (;;) {
        // 当递归深度超过上限时，切换到堆排序
        if (NPY_UNLIKELY(cdepth < 0)) {
            aheapsort_<Tag>(vv, pl, pr - pl + 1);  // 调用堆排序函数
            goto stack_pop;  // 跳转到出栈操作
        }
        // 当数组长度大于阈值时，执行快速排序的分区操作
        while ((pr - pl) > SMALL_QUICKSORT) {
            /* quicksort partition */  // 快速排序的分区步骤
            pm = pl + ((pr - pl) >> 1);  // 计算中间位置
            if (Tag::less(v[*pm], v[*pl])) {  // 比较并交换元素
                std::swap(*pm, *pl);
            }
            if (Tag::less(v[*pr], v[*pm])) {  // 比较并交换元素
                std::swap(*pr, *pm);
            }
            if (Tag::less(v[*pm], v[*pl])) {  // 比较并交换元素
                std::swap(*pm, *pl);
            }
            vp = v[*pm];  // 保存中间值
            pi = pl;      // 初始化左指针
            pj = pr - 1;  // 初始化右指针
            std::swap(*pm, *pj);  // 交换中间值与最右边的值
            // 执行分区操作
            for (;;) {
                do {
                    ++pi;
                } while (Tag::less(v[*pi], vp));  // 找到大于中间值的元素
                do {
                    --pj;
                } while (Tag::less(vp, v[*pj]));  // 找到小于中间值的元素
                if (pi >= pj) {
                    break;
                }
                std::swap(*pi, *pj);  // 交换位置
            }
            pk = pr - 1;  // 保存右侧分区的位置
            std::swap(*pi, *pk);  // 将中间值放回正确位置
            /* push largest partition on stack */  // 将较大的分区压入栈中
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
            *psdepth++ = --cdepth;  // 保存当前递归深度
        }

        // 插入排序，处理小数组
        for (pi = pl + 1; pi <= pr; ++pi) {
            vi = *pi;
            vp = v[vi];
            pj = pi;
            pk = pi - 1;
            while (pj > pl && Tag::less(vp, v[*pk])) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    stack_pop:
        // 栈为空时跳出循环
        if (sptr == stack) {
            break;
        }
        // 出栈操作，恢复分区和递归深度
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }

    return 0;  // 返回排序完成标志
}

/*
 *****************************************************************************
 **                             STRING SORTS                                **
 *****************************************************************************
 */

// 静态函数 string_quicksort_，实现字符串数组的快速排序
template <typename Tag, typename type>
static int
string_quicksort_(type *start, npy_intp num, void *varr)
{
    PyArrayObject *arr = (PyArrayObject *)varr;  // 转换为 Python 数组对象
    const size_t len = PyArray_ITEMSIZE(arr) / sizeof(type);  // 计算元素长度
    type *vp;
    type *pl = start;  // 左指针，指向数组的起始位置
    type *pr = pl + (num - 1) * len;  // 右指针，指向数组的末尾位置
    type *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pk;  // 分区操作使用的栈和指针
    int depth[PYA_QS_STACK];  // 存储递归深度的栈
    int *psdepth = depth;  // 深度指针，指向栈顶
    int cdepth = npy_get_msb(num) * 2;  // 计算递归深度的上限
    /* 如果待排序数组长度为零，直接返回，不需要排序 */
    if (len == 0) {
        return 0;
    }

    /* 为存储元素的临时内存分配空间 */
    vp = (type *)malloc(PyArray_ITEMSIZE(arr));
    if (vp == NULL) {
        return -NPY_ENOMEM;  /* 内存分配失败 */
    }

    for (;;) {
        /* 如果深度小于零，使用堆排序 */
        if (NPY_UNLIKELY(cdepth < 0)) {
            string_heapsort_<Tag>(pl, (pr - pl) / len + 1, varr);
            goto stack_pop;  /* 跳转到堆栈弹出处理 */
        }
        while ((size_t)(pr - pl) > SMALL_QUICKSORT * len) {
            /* 快速排序分区 */
            pm = pl + (((pr - pl) / len) >> 1) * len;
            if (Tag::less(pm, pl, len)) {
                Tag::swap(pm, pl, len);  /* 交换元素 */
            }
            if (Tag::less(pr, pm, len)) {
                Tag::swap(pr, pm, len);  /* 交换元素 */
            }
            if (Tag::less(pm, pl, len)) {
                Tag::swap(pm, pl, len);  /* 交换元素 */
            }
            Tag::copy(vp, pm, len);  /* 复制元素 */
            pi = pl;
            pj = pr - len;
            Tag::swap(pm, pj, len);  /* 交换元素 */
            for (;;) {
                do {
                    pi += len;
                } while (Tag::less(pi, vp, len));  /* 找到大于等于vp的元素位置 */
                do {
                    pj -= len;
                } while (Tag::less(vp, pj, len));  /* 找到小于等于vp的元素位置 */
                if (pi >= pj) {
                    break;  /* 分区完成 */
                }
                Tag::swap(pi, pj, len);  /* 交换元素 */
            }
            pk = pr - len;
            Tag::swap(pi, pk, len);  /* 交换元素 */
            /* 将较大的分区压入堆栈 */
            if (pi - pl < pr - pi) {
                *sptr++ = pi + len;
                *sptr++ = pr;
                pr = pi - len;
            }
            else {
                *sptr++ = pl;
                *sptr++ = pi - len;
                pl = pi + len;
            }
            *psdepth++ = --cdepth;  /* 压入当前深度 */
        }

        /* 插入排序 */
        for (pi = pl + len; pi <= pr; pi += len) {
            Tag::copy(vp, pi, len);  /* 复制元素 */
            pj = pi;
            pk = pi - len;
            while (pj > pl && Tag::less(vp, pk, len)) {
                Tag::copy(pj, pk, len);  /* 复制元素 */
                pj -= len;
                pk -= len;
            }
            Tag::copy(pj, vp, len);  /* 复制元素 */
        }
    stack_pop:
        if (sptr == stack) {
            break;  /* 堆栈空，排序完成 */
        }
        pr = *(--sptr);  /* 弹出堆栈，恢复分区 */
        pl = *(--sptr);  /* 弹出堆栈，恢复分区 */
        cdepth = *(--psdepth);  /* 弹出堆栈，恢复深度 */
    }

    free(vp);  /* 释放临时内存空间 */
    return 0;  /* 排序成功 */
// 模板函数实现，用于快速排序字符串类型数组
template <typename Tag, typename type>
static int
string_aquicksort_(type *vv, npy_intp *tosort, npy_intp num, void *varr)
{
    // 初始化变量和指针
    type *v = vv;  // 指向排序数组的指针
    PyArrayObject *arr = (PyArrayObject *)varr;  // 将varr转换为PyArrayObject类型的指针
    size_t len = PyArray_ITEMSIZE(arr) / sizeof(type);  // 计算每个元素的字节大小并转换为元素个数
    type *vp;  // 指向中间值的指针
    npy_intp *pl = tosort;  // 左指针，指向要排序的数组的起始位置
    npy_intp *pr = tosort + num - 1;  // 右指针，指向要排序的数组的末尾位置
    npy_intp *stack[PYA_QS_STACK];  // 定义堆栈数组，存放分区的边界指针
    npy_intp **sptr = stack;  // 堆栈指针，指向堆栈顶部
    npy_intp *pm, *pi, *pj, *pk, vi;  // 定义用于分区和排序的辅助指针和变量
    int depth[PYA_QS_STACK];  // 存放每个分区的深度
    int *psdepth = depth;  // 深度指针，指向深度数组的顶部
    int cdepth = npy_get_msb(num) * 2;  // 初始化递归深度的估计值

    /* Items that have zero size don't make sense to sort */
    // 如果元素大小为0，无需排序，直接返回
    if (len == 0) {
        return 0;
    }

    // 主循环，进行快速排序或插入排序
    for (;;) {
        // 当深度估计小于0时，切换到堆排序
        if (NPY_UNLIKELY(cdepth < 0)) {
            // 调用堆排序函数，完成剩余元素的排序
            string_aheapsort_<Tag>(vv, pl, pr - pl + 1, varr);
            // 跳转到堆栈弹出的代码段
            goto stack_pop;
        }
        // 当剩余元素数量大于指定阈值时，使用快速排序分区
        while ((pr - pl) > SMALL_QUICKSORT) {
            /* quicksort partition */
            // 计算中间值的位置
            pm = pl + ((pr - pl) >> 1);
            // 根据排序标签进行交换操作，以保证左中右的顺序
            if (Tag::less(v + (*pm) * len, v + (*pl) * len, len)) {
                std::swap(*pm, *pl);
            }
            if (Tag::less(v + (*pr) * len, v + (*pm) * len, len)) {
                std::swap(*pr, *pm);
            }
            if (Tag::less(v + (*pm) * len, v + (*pl) * len, len)) {
                std::swap(*pm, *pl);
            }
            // 获取中间值的指针
            vp = v + (*pm) * len;
            pi = pl;
            pj = pr - 1;
            std::swap(*pm, *pj);
            // 进行分区操作，将小于和大于中间值的元素移动到两侧
            for (;;) {
                do {
                    ++pi;
                } while (Tag::less(v + (*pi) * len, vp, len));
                do {
                    --pj;
                } while (Tag::less(vp, v + (*pj) * len, len));
                if (pi >= pj) {
                    break;
                }
                std::swap(*pi, *pj);
            }
            pk = pr - 1;
            std::swap(*pi, *pk);
            // 将较大的分区边界压入堆栈
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
            // 记录当前分区的深度
            *psdepth++ = --cdepth;
        }

        // 插入排序，对小于阈值的分区进行排序
        for (pi = pl + 1; pi <= pr; ++pi) {
            vi = *pi;
            vp = v + vi * len;
            pj = pi;
            pk = pi - 1;
            // 使用标签进行比较和交换，完成插入排序
            while (pj > pl && Tag::less(vp, v + (*pk) * len, len)) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    stack_pop:
        // 检查堆栈是否为空，若为空则排序完成
        if (sptr == stack) {
            break;
        }
        // 弹出堆栈中保存的下一个分区边界
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }

    return 0;  // 返回排序完成的标志
}
    // 将 varr 转换为 PyArrayObject 指针类型的变量 arr
    PyArrayObject *arr = (PyArrayObject *)varr;
    // 获取 arr 元素的大小
    npy_intp elsize = PyArray_ITEMSIZE(arr);
    // 获取 arr 元素的比较函数指针 cmp
    PyArray_CompareFunc *cmp = PyDataType_GetArrFuncs(PyArray_DESCR(arr))->compare;
    // 指向起始位置的字符指针 pl
    char *vp;
    char *pl = (char *)start;
    // 指向结束位置的字符指针 pr
    char *pr = pl + (num - 1) * elsize;
    // 用于存放递归调用时的栈
    char *stack[PYA_QS_STACK];
    // 栈指针 sptr，指向栈顶
    char **sptr = stack;
    // 用于快速排序的中间指针 pm, pi, pj, pk
    char *pm, *pi, *pj, *pk;
    // 用于记录递归深度的数组 depth
    int depth[PYA_QS_STACK];
    // 深度指针 psdepth，指向 depth 的栈顶
    int *psdepth = depth;
    // 计算递归深度 cdepth，为 num 的最高有效位左移一位乘以 2
    int cdepth = npy_get_msb(num) * 2;

    /* Items that have zero size don't make sense to sort */
    // 如果元素大小为 0，则无需排序，直接返回 0
    if (elsize == 0) {
        return 0;
    }

    // 分配 elsize 大小的内存空间给 vp
    vp = (char *)malloc(elsize);
    // 如果分配内存失败，返回内存错误码 -NPY_ENOMEM
    if (vp == NULL) {
        return -NPY_ENOMEM;
    }

    // 无限循环，直到排序完成或栈为空
    for (;;) {
        // 如果递归深度 cdepth 小于 0，使用堆排序进行排序
        if (NPY_UNLIKELY(cdepth < 0)) {
            npy_heapsort(pl, (pr - pl) / elsize + 1, varr);
            // 跳转至 stack_pop 进行栈弹出操作
            goto stack_pop;
        }
        // 当待排序元素数量大于 SMALL_QUICKSORT 时，使用快速排序
        while (pr - pl > SMALL_QUICKSORT * elsize) {
            /* quicksort partition */
            // 计算快速排序的中间点 pm
            pm = pl + (((pr - pl) / elsize) >> 1) * elsize;
            // 三值取中，将 pm 与 pl 比较并交换
            if (cmp(pm, pl, arr) < 0) {
                GENERIC_SWAP(pm, pl, elsize);
            }
            // 将 pr 与 pm 比较并交换
            if (cmp(pr, pm, arr) < 0) {
                GENERIC_SWAP(pr, pm, elsize);
            }
            // 再次将 pm 与 pl 比较并交换
            if (cmp(pm, pl, arr) < 0) {
                GENERIC_SWAP(pm, pl, elsize);
            }
            // 将 pm 复制到 vp
            GENERIC_COPY(vp, pm, elsize);
            // 初始化指针 pi 和 pj
            pi = pl;
            pj = pr - elsize;
            // 将 pm 与 pj 交换
            GENERIC_SWAP(pm, pj, elsize);
            /*
             * 通用比较可能存在错误，不依赖哨兵保持指针不超出边界。
             */
            // 循环直到 pi >= pj
            for (;;) {
                // 从左向右找到大于等于 vp 的元素
                do {
                    pi += elsize;
                } while (cmp(pi, vp, arr) < 0 && pi < pj);
                // 从右向左找到小于等于 vp 的元素
                do {
                    pj -= elsize;
                } while (cmp(vp, pj, arr) < 0 && pi < pj);
                // 如果 pi >= pj，则退出循环
                if (pi >= pj) {
                    break;
                }
                // 交换 pi 和 pj 处的元素
                GENERIC_SWAP(pi, pj, elsize);
            }
            // 将 pi 与 pr-el 处的元素交换
            pk = pr - elsize;
            GENERIC_SWAP(pi, pk, elsize);
            // 将较大的分区压入栈中
            if (pi - pl < pr - pi) {
                *sptr++ = pi + elsize;
                *sptr++ = pr;
                pr = pi - elsize;
            }
            else {
                *sptr++ = pl;
                *sptr++ = pi - elsize;
                pl = pi + elsize;
            }
            // 减小递归深度并存入 depth 数组
            *psdepth++ = --cdepth;
        }

        // 插入排序
        for (pi = pl + elsize; pi <= pr; pi += elsize) {
            // 将 pi 处元素复制到 vp
            GENERIC_COPY(vp, pi, elsize);
            // 从右向左比较并移动元素，直到找到合适位置插入 vp
            pj = pi;
            pk = pi - elsize;
            while (pj > pl && cmp(vp, pk, arr) < 0) {
                GENERIC_COPY(pj, pk, elsize);
                pj -= elsize;
                pk -= elsize;
            }
            // 将 vp 复制到 pj 处
            GENERIC_COPY(pj, vp, elsize);
        }

    stack_pop:
        // 如果栈为空，跳出循环
        if (sptr == stack) {
            break;
        }
        // 弹出栈顶的 pl 和 pr
        pr = *(--sptr);
        pl = *(--sptr);
        // 弹出深度栈顶的 cdepth
        cdepth = *(--psdepth);
    }

    // 释放 vp 所占用的内存空间
    free(vp);
    // 排序完成，返回 0
    return 0;
# 定义了一个非导出的整型函数 `npy_aquicksort`，接受四个参数：指向排序对象的指针 `vv`，
# 指向要排序的索引数组的指针 `tosort`，排序的元素数量 `num`，以及指向 `PyArrayObject` 结构体的指针 `varr`
NPY_NO_EXPORT int
npy_aquicksort(void *vv, npy_intp *tosort, npy_intp num, void *varr)
{
    # 将 `vv` 转换为 `char` 类型指针 `v`
    char *v = (char *)vv;
    # 将 `varr` 转换为 `PyArrayObject` 类型指针 `arr`
    PyArrayObject *arr = (PyArrayObject *)varr;
    # 获取数组元素的大小 `elsize`
    npy_intp elsize = PyArray_ITEMSIZE(arr);
    # 获取用于比较的函数指针 `cmp`，这里使用 `arr` 的描述符获取比较函数
    PyArray_CompareFunc *cmp = PyDataType_GetArrFuncs(PyArray_DESCR(arr))->compare;
    # 定义指向首尾的指针 `vp`，左边界 `pl`，右边界 `pr`，初始指向排序索引数组的头尾
    char *vp;
    npy_intp *pl = tosort;
    npy_intp *pr = tosort + num - 1;
    # 定义存储栈的数组 `stack`，栈指针 `sptr` 指向 `stack`
    npy_intp *stack[PYA_QS_STACK];
    npy_intp **sptr = stack;
    # 定义深度存储数组 `depth`，深度指针 `psdepth` 指向 `depth`
    int depth[PYA_QS_STACK];
    int *psdepth = depth;
    # 计算初始深度 `cdepth`，用于快速排序过程中的深度控制
    int cdepth = npy_get_msb(num) * 2;

    /* Items that have zero size don't make sense to sort */
    # 如果元素大小 `elsize` 为 0，则无需排序，直接返回 0
    if (elsize == 0) {
        return 0;
    }

    # 进入无限循环，用于快速排序
    for (;;) {
        # 如果深度 `cdepth` 小于 0，使用堆排序对当前区间排序后跳转到 `stack_pop` 标签处
        if (NPY_UNLIKELY(cdepth < 0)) {
            npy_aheapsort(vv, pl, pr - pl + 1, varr);
            goto stack_pop;
        }
        # 当待排序区间大于 `SMALL_QUICKSORT` 时，使用快速排序算法分割区间
        while ((pr - pl) > SMALL_QUICKSORT) {
            /* quicksort partition */
            # 计算中间位置 `pm`，并根据比较函数 `cmp` 对三个位置进行排序
            pm = pl + ((pr - pl) >> 1);
            if (cmp(v + (*pm) * elsize, v + (*pl) * elsize, arr) < 0) {
                INTP_SWAP(*pm, *pl);
            }
            if (cmp(v + (*pr) * elsize, v + (*pm) * elsize, arr) < 0) {
                INTP_SWAP(*pr, *pm);
            }
            if (cmp(v + (*pm) * elsize, v + (*pl) * elsize, arr) < 0) {
                INTP_SWAP(*pm, *pl);
            }
            # 获取中间位置 `pm` 对应的元素值，并定义两个指针 `pi` 和 `pj`
            vp = v + (*pm) * elsize;
            pi = pl;
            pj = pr - 1;
            INTP_SWAP(*pm, *pj);
            # 进行双指针分割排序
            for (;;) {
                do {
                    ++pi;
                } while (cmp(v + (*pi) * elsize, vp, arr) < 0 && pi < pj);
                do {
                    --pj;
                } while (cmp(vp, v + (*pj) * elsize, arr) < 0 && pi < pj);
                if (pi >= pj) {
                    break;
                }
                INTP_SWAP(*pi, *pj);
            }
            pk = pr - 1;
            INTP_SWAP(*pi, *pk);
            /* push largest partition on stack */
            # 将较大的分区推入栈中
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
            *psdepth++ = --cdepth;
        }

        /* insertion sort */
        # 对于较小的区间使用插入排序
        for (pi = pl + 1; pi <= pr; ++pi) {
            vi = *pi;
            vp = v + vi * elsize;
            pj = pi;
            pk = pi - 1;
            while (pj > pl && cmp(vp, v + (*pk) * elsize, arr) < 0) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    stack_pop:
        # 如果栈为空则跳出循环，否则从栈中取出新的区间进行排序
        if (sptr == stack) {
            break;
        }
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }

    # 完成排序，返回 0
    return 0;
}
NPY_NO_EXPORT int
quicksort_byte(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    // 调用模板化的快速排序函数，排序字节数组
    return quicksort_<npy::byte_tag>((npy_byte *)start, n);
}

NPY_NO_EXPORT int
quicksort_ubyte(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    // 调用模板化的快速排序函数，排序无符号字节数组
    return quicksort_<npy::ubyte_tag>((npy_ubyte *)start, n);
}

NPY_NO_EXPORT int
quicksort_short(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    // 如果可以调用专门的快速排序分发函数，则返回0，否则使用通用的短整数快速排序
    if (quicksort_dispatch((npy_short *)start, n)) {
        return 0;
    }
    return quicksort_<npy::short_tag>((npy_short *)start, n);
}

NPY_NO_EXPORT int
quicksort_ushort(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    // 如果可以调用专门的快速排序分发函数，则返回0，否则使用通用的无符号短整数快速排序
    if (quicksort_dispatch((npy_ushort *)start, n)) {
        return 0;
    }
    return quicksort_<npy::ushort_tag>((npy_ushort *)start, n);
}

NPY_NO_EXPORT int
quicksort_int(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    // 如果可以调用专门的快速排序分发函数，则返回0，否则使用通用的整数快速排序
    if (quicksort_dispatch((npy_int *)start, n)) {
        return 0;
    }
    return quicksort_<npy::int_tag>((npy_int *)start, n);
}

NPY_NO_EXPORT int
quicksort_uint(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    // 如果可以调用专门的快速排序分发函数，则返回0，否则使用通用的无符号整数快速排序
    if (quicksort_dispatch((npy_uint *)start, n)) {
        return 0;
    }
    return quicksort_<npy::uint_tag>((npy_uint *)start, n);
}

NPY_NO_EXPORT int
quicksort_long(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    // 如果可以调用专门的快速排序分发函数，则返回0，否则使用通用的长整数快速排序
    if (quicksort_dispatch((npy_long *)start, n)) {
        return 0;
    }
    return quicksort_<npy::long_tag>((npy_long *)start, n);
}

NPY_NO_EXPORT int
quicksort_ulong(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    // 如果可以调用专门的快速排序分发函数，则返回0，否则使用通用的无符号长整数快速排序
    if (quicksort_dispatch((npy_ulong *)start, n)) {
        return 0;
    }
    return quicksort_<npy::ulong_tag>((npy_ulong *)start, n);
}

NPY_NO_EXPORT int
quicksort_longlong(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    // 如果可以调用专门的快速排序分发函数，则返回0，否则使用通用的长长整数快速排序
    if (quicksort_dispatch((npy_longlong *)start, n)) {
        return 0;
    }
    return quicksort_<npy::longlong_tag>((npy_longlong *)start, n);
}

NPY_NO_EXPORT int
quicksort_ulonglong(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    // 如果可以调用专门的快速排序分发函数，则返回0，否则使用通用的无符号长长整数快速排序
    if (quicksort_dispatch((npy_ulonglong *)start, n)) {
        return 0;
    }
    return quicksort_<npy::ulonglong_tag>((npy_ulonglong *)start, n);
}

NPY_NO_EXPORT int
quicksort_half(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    // 如果可以调用专门的快速排序分发函数，则返回0，否则使用通用的半精度浮点数快速排序
    if (quicksort_dispatch((np::Half *)start, n)) {
        return 0;
    }
    return quicksort_<npy::half_tag>((npy_half *)start, n);
}

NPY_NO_EXPORT int
quicksort_float(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    // 如果可以调用专门的快速排序分发函数，则返回0，否则使用通用的单精度浮点数快速排序
    if (quicksort_dispatch((npy_float *)start, n)) {
        return 0;
    }
    return quicksort_<npy::float_tag>((npy_float *)start, n);
}

NPY_NO_EXPORT int
quicksort_double(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    // 如果可以调用专门的快速排序分发函数，则返回0，否则使用通用的双精度浮点数快速排序
    if (quicksort_dispatch((npy_double *)start, n)) {
        return 0;
    }
    return quicksort_<npy::double_tag>((npy_double *)start, n);
}

NPY_NO_EXPORT int
quicksort_longdouble(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    // 使用通用的长双精度浮点数快速排序
    return quicksort_<npy::longdouble_tag>((npy_longdouble *)start, n);
}
// 使用特定的复数类型（cfloat），调用模板化的快速排序函数，并返回排序后的结果
quicksort_cfloat(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::cfloat_tag>((npy_cfloat *)start, n);
}

// 使用特定的双精度复数类型（cdouble），调用模板化的快速排序函数，并返回排序后的结果
NPY_NO_EXPORT int
quicksort_cdouble(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::cdouble_tag>((npy_cdouble *)start, n);
}

// 使用特定的长双精度复数类型（clongdouble），调用模板化的快速排序函数，并返回排序后的结果
NPY_NO_EXPORT int
quicksort_clongdouble(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::clongdouble_tag>((npy_clongdouble *)start, n);
}

// 使用特定的日期时间类型（datetime），调用模板化的快速排序函数，并返回排序后的结果
NPY_NO_EXPORT int
quicksort_datetime(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::datetime_tag>((npy_datetime *)start, n);
}

// 使用特定的时间间隔类型（timedelta），调用模板化的快速排序函数，并返回排序后的结果
NPY_NO_EXPORT int
quicksort_timedelta(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::timedelta_tag>((npy_timedelta *)start, n);
}

// 使用特定的布尔类型（bool），调用模板化的快速排序函数，并返回排序后的结果
NPY_NO_EXPORT int
aquicksort_bool(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::bool_tag>((npy_bool *)vv, tosort, n);
}

// 使用特定的字节类型（byte），调用模板化的快速排序函数，并返回排序后的结果
NPY_NO_EXPORT int
aquicksort_byte(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::byte_tag>((npy_byte *)vv, tosort, n);
}

// 使用特定的无符号字节类型（ubyte），调用模板化的快速排序函数，并返回排序后的结果
NPY_NO_EXPORT int
aquicksort_ubyte(void *vv, npy_intp *tosort, npy_intp n,
                 void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::ubyte_tag>((npy_ubyte *)vv, tosort, n);
}

// 使用特定的短整数类型（short），调用模板化的快速排序函数，并返回排序后的结果
NPY_NO_EXPORT int
aquicksort_short(void *vv, npy_intp *tosort, npy_intp n,
                 void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::short_tag>((npy_short *)vv, tosort, n);
}

// 使用特定的无符号短整数类型（ushort），调用模板化的快速排序函数，并返回排序后的结果
NPY_NO_EXPORT int
aquicksort_ushort(void *vv, npy_intp *tosort, npy_intp n,
                  void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::ushort_tag>((npy_ushort *)vv, tosort, n);
}

// 使用特定的整数类型（int），调用模板化的快速排序函数，并返回排序后的结果
NPY_NO_EXPORT int
aquicksort_int(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    // 如果排序分派函数返回 true，则直接返回 0
    if (aquicksort_dispatch((npy_int *)vv, tosort, n)) {
        return 0;
    }
    return aquicksort_<npy::int_tag>((npy_int *)vv, tosort, n);
}

// 使用特定的无符号整数类型（uint），调用模板化的快速排序函数，并返回排序后的结果
NPY_NO_EXPORT int
aquicksort_uint(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    // 如果排序分派函数返回 true，则直接返回 0
    if (aquicksort_dispatch((npy_uint *)vv, tosort, n)) {
        return 0;
    }
    return aquicksort_<npy::uint_tag>((npy_uint *)vv, tosort, n);
}

// 使用特定的长整数类型（long），调用模板化的快速排序函数，并返回排序后的结果
NPY_NO_EXPORT int
aquicksort_long(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    // 如果排序分派函数返回 true，则直接返回 0
    if (aquicksort_dispatch((npy_long *)vv, tosort, n)) {
        return 0;
    }
    return aquicksort_<npy::long_tag>((npy_long *)vv, tosort, n);
}

// 使用特定的无符号长整数类型（ulong），调用模板化的快速排序函数，并返回排序后的结果
NPY_NO_EXPORT int
aquicksort_ulong(void *vv, npy_intp *tosort, npy_intp n,
                 void *NPY_UNUSED(varr))
{
    // 如果排序分派函数返回 true，则直接返回 0
    if (aquicksort_dispatch((npy_ulong *)vv, tosort, n)) {
        return 0;
    }
    return aquicksort_<npy::ulong_tag>((npy_ulong *)vv, tosort, n);
}

// 使用特定的长长整数类型（longlong），调用模板化的快速排序函数，并返回排序后的结果
NPY_NO_EXPORT int
aquicksort_longlong(void *vv, npy_intp *tosort, npy_intp n,
                    void *NPY_UNUSED(varr))
{
    // 如果排序分派函数返回 true，则直接返回 0
    if (aquicksort_dispatch((npy_longlong *)vv, tosort, n)) {
        return 0;
    }
    return aquicksort_<npy::longlong_tag>((npy_longlong *)vv, tosort, n);
}
# 定义一个不导出的整型函数 aquicksort_ulonglong，用于快速排序无符号长长整型数组
NPY_NO_EXPORT int
aquicksort_ulonglong(void *vv, npy_intp *tosort, npy_intp n,
                     void *NPY_UNUSED(varr))
{
    # 如果 aquicksort_dispatch 返回 true，表示成功使用快速排序函数进行排序
    if (aquicksort_dispatch((npy_ulonglong *)vv, tosort, n)) {
        # 返回 0 表示排序成功
        return 0;
    }
    # 否则调用 aquicksort_<npy::ulonglong_tag>，使用特定标签的快速排序函数进行排序
    return aquicksort_<npy::ulonglong_tag>((npy_ulonglong *)vv, tosort, n);
}

# 定义一个不导出的整型函数 aquicksort_half，用于快速排序半精度浮点数数组
NPY_NO_EXPORT int
aquicksort_half(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    # 直接调用 aquicksort_<npy::half_tag>，使用特定标签的快速排序函数进行排序
    return aquicksort_<npy::half_tag>((npy_half *)vv, tosort, n);
}

# 定义一个不导出的整型函数 aquicksort_float，用于快速排序单精度浮点数数组
NPY_NO_EXPORT int
aquicksort_float(void *vv, npy_intp *tosort, npy_intp n,
                 void *NPY_UNUSED(varr))
{
    # 如果 aquicksort_dispatch 返回 true，表示成功使用快速排序函数进行排序
    if (aquicksort_dispatch((npy_float *)vv, tosort, n)) {
        # 返回 0 表示排序成功
        return 0;
    }
    # 否则调用 aquicksort_<npy::float_tag>，使用特定标签的快速排序函数进行排序
    return aquicksort_<npy::float_tag>((npy_float *)vv, tosort, n);
}

# 定义一个不导出的整型函数 aquicksort_double，用于快速排序双精度浮点数数组
NPY_NO_EXPORT int
aquicksort_double(void *vv, npy_intp *tosort, npy_intp n,
                  void *NPY_UNUSED(varr))
{
    # 如果 aquicksort_dispatch 返回 true，表示成功使用快速排序函数进行排序
    if (aquicksort_dispatch((npy_double *)vv, tosort, n)) {
        # 返回 0 表示排序成功
        return 0;
    }
    # 否则调用 aquicksort_<npy::double_tag>，使用特定标签的快速排序函数进行排序
    return aquicksort_<npy::double_tag>((npy_double *)vv, tosort, n);
}

# 定义一个不导出的整型函数 aquicksort_longdouble，用于快速排序长双精度浮点数数组
NPY_NO_EXPORT int
aquicksort_longdouble(void *vv, npy_intp *tosort, npy_intp n,
                      void *NPY_UNUSED(varr))
{
    # 直接调用 aquicksort_<npy::longdouble_tag>，使用特定标签的快速排序函数进行排序
    return aquicksort_<npy::longdouble_tag>((npy_longdouble *)vv, tosort, n);
}

# 定义一个不导出的整型函数 aquicksort_cfloat，用于快速排序复数单精度浮点数数组
NPY_NO_EXPORT int
aquicksort_cfloat(void *vv, npy_intp *tosort, npy_intp n,
                  void *NPY_UNUSED(varr))
{
    # 直接调用 aquicksort_<npy::cfloat_tag>，使用特定标签的快速排序函数进行排序
    return aquicksort_<npy::cfloat_tag>((npy_cfloat *)vv, tosort, n);
}

# 定义一个不导出的整型函数 aquicksort_cdouble，用于快速排序复数双精度浮点数数组
NPY_NO_EXPORT int
aquicksort_cdouble(void *vv, npy_intp *tosort, npy_intp n,
                   void *NPY_UNUSED(varr))
{
    # 直接调用 aquicksort_<npy::cdouble_tag>，使用特定标签的快速排序函数进行排序
    return aquicksort_<npy::cdouble_tag>((npy_cdouble *)vv, tosort, n);
}

# 定义一个不导出的整型函数 aquicksort_clongdouble，用于快速排序复数长双精度浮点数数组
NPY_NO_EXPORT int
aquicksort_clongdouble(void *vv, npy_intp *tosort, npy_intp n,
                       void *NPY_UNUSED(varr))
{
    # 直接调用 aquicksort_<npy::clongdouble_tag>，使用特定标签的快速排序函数进行排序
    return aquicksort_<npy::clongdouble_tag>((npy_clongdouble *)vv, tosort, n);
}

# 定义一个不导出的整型函数 aquicksort_datetime，用于快速排序日期时间数组
NPY_NO_EXPORT int
aquicksort_datetime(void *vv, npy_intp *tosort, npy_intp n,
                    void *NPY_UNUSED(varr))
{
    # 直接调用 aquicksort_<npy::datetime_tag>，使用特定标签的快速排序函数进行排序
    return aquicksort_<npy::datetime_tag>((npy_datetime *)vv, tosort, n);
}

# 定义一个不导出的整型函数 aquicksort_timedelta，用于快速排序时间间隔数组
NPY_NO_EXPORT int
aquicksort_timedelta(void *vv, npy_intp *tosort, npy_intp n,
                     void *NPY_UNUSED(varr))
{
    # 直接调用 aquicksort_<npy::timedelta_tag>，使用特定标签的快速排序函数进行排序
    return aquicksort_<npy::timedelta_tag>((npy_timedelta *)vv, tosort, n);
}

# 定义一个不导出的整型函数 quicksort_string，用于快速排序字符串数组
NPY_NO_EXPORT int
quicksort_string(void *start, npy_intp n, void *varr)
{
    # 直接调用 string_quicksort_<npy::string_tag>，使用特定标签的快速排序函数进行排序
    return string_quicksort_<npy::string_tag>((npy_char *)start, n, varr);
}

# 定义一个不导出的整型函数 quicksort_unicode，用于快速排序 Unicode 字符串数组
NPY_NO_EXPORT int
quicksort_unicode(void *start, npy_intp n, void *varr)
{
    # 直接调用 string_quicksort_<npy::unicode_tag>，使用特定标签的快速排序函数进行排序
    return string_quicksort_<npy::unicode_tag>((npy_ucs4 *)start, n, varr);
}

# 定义一个不导出的整型函数 aquicksort_string，用于快速排序字符串数组
NPY_NO_EXPORT int
aquicksort_string(void *vv, npy_intp *tosort, npy_intp n, void *varr)
{
    # 直接调用 string_aquicksort_<npy::string_tag>，使用特定标签的快速排序函数进行排序
    return string_aquicksort_<npy::string_tag>((npy_char *)vv, tosort, n, varr);
}

# 定义一个不导出的整型函数 aquicksort_unicode，用于快速排序 Unicode 字符串数组
NPY_NO_EXPORT int
aquicksort_unicode(void *vv, npy_intp *tosort, npy_intp n, void *varr)
{
    # 直接调用 string_aquicksort_<npy::unicode_tag>，使用特定标签的快速排序函数进行排序
    return string_aquicksort_<npy::unicode_tag>((npy_ucs4 *)vv, tosort, n, varr);
}
```