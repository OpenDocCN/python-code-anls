# `.\numpy\numpy\_core\src\npysort\timsort.cpp`

```
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
 * Quick sort is usually the fastest, but the worst case scenario can
 * be slower than the merge and heap sorts.  The merge sort requires
 * extra memory and so for large arrays may not be useful.
 *
 * The merge sort is *stable*, meaning that equal components
 * are unmoved from their entry versions, so it can be used to
 * implement lexicographic sorting on multiple keys.
 *
 * The heap sort is included for completeness.
 */

/* For details of Timsort, refer to
 * https://github.com/python/cpython/blob/3.7/Objects/listsort.txt
 */

/* Define to prevent using deprecated APIs */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

/* Include necessary headers */
#include "npy_sort.h"         /* Header for numpy sorting */
#include "npysort_common.h"   /* Header for common sorting utilities */
#include "numpy_tag.h"        /* Header for numpy type tags */

/* Standard library includes */
#include <cstdlib>            /* General utilities library */
#include <utility>            /* Utility components */

/* Stack size for the Timsort algorithm */
#define TIMSORT_STACK_SIZE 128

/* Function to compute the minimum run length for Timsort */
static npy_intp
compute_min_run(npy_intp num)
{
    npy_intp r = 0;

    /* Calculate the minimum run length based on the number */
    while (64 < num) {
        r |= num & 1;
        num >>= 1;
    }

    return num + r;
}

/* Structure to represent a run in Timsort */
typedef struct {
    npy_intp s; /* start pointer */
    npy_intp l; /* length */
} run;

/* Structure for a buffer used in argsort */
typedef struct {
    npy_intp *pw;  /* pointer to the buffer */
    npy_intp size; /* size of the buffer */
} buffer_intp;

/* Function to resize the buffer used in argsort */
static inline int
resize_buffer_intp(buffer_intp *buffer, npy_intp new_size)
{
    if (new_size <= buffer->size) {
        return 0;
    }

    /* Allocate memory for the resized buffer */
    npy_intp *new_pw = (npy_intp *)realloc(buffer->pw, new_size * sizeof(npy_intp));

    /* Update the buffer size */
    buffer->size = new_size;

    /* Handle memory allocation failure */
    if (NPY_UNLIKELY(new_pw == NULL)) {
        return -NPY_ENOMEM;
    }
    else {
        buffer->pw = new_pw;
        return 0;
    }
}

/*
 *****************************************************************************
 **                            NUMERIC SORTS                                **
 *****************************************************************************
 */

/* Template structure for buffer used in numeric sorts */
template <typename Tag>
struct buffer_ {
    typename Tag::type *pw; /* pointer to the buffer */
    npy_intp size;          /* size of the buffer */
};

/* Template function to resize the buffer used in numeric sorts */
template <typename Tag>
static inline int
resize_buffer_(buffer_<Tag> *buffer, npy_intp new_size)
{
    using type = typename Tag::type;
    if (new_size <= buffer->size) {
        return 0;
    }

    /* Allocate memory for the resized buffer */
    type *new_pw = (type *)realloc(buffer->pw, new_size * sizeof(type));

    /* Update the buffer size */
    buffer->size = new_size;

    /* Handle memory allocation failure */
    if (NPY_UNLIKELY(new_pw == NULL)) {
        return -NPY_ENOMEM;
    }
    else {
        buffer->pw = new_pw;
        return 0;
    }
}

/* Function to count runs in a numeric array for Timsort */
template <typename Tag, typename type>
static npy_intp
count_run_(type *arr, npy_intp l, npy_intp num, npy_intp minrun)
{
    npy_intp sz;  // 定义变量 sz，用于存储子序列的长度
    type vc, *pl, *pi, *pj, *pr;  // 定义变量 vc 以及指针 pl, pi, pj, pr

    if (NPY_UNLIKELY(num - l == 1)) {  // 如果 num - l 等于 1，表示子序列长度为 2，直接返回 1
        return 1;
    }

    pl = arr + l;  // 指针 pl 指向数组 arr 的第 l 个元素

    /* (not strictly) ascending sequence */
    if (!Tag::less(*(pl + 1), *pl)) {  // 如果 pl 和 pl+1 之间不是严格递增关系
        for (pi = pl + 1; pi < arr + num - 1 && !Tag::less(*(pi + 1), *pi);
             ++pi) {
        }  // 找到递增子序列的结束位置，pi 指向最后一个递增元素的后一个位置
    }
    else { /* (strictly) descending sequence */
        for (pi = pl + 1; pi < arr + num - 1 && Tag::less(*(pi + 1), *pi);
             ++pi) {
        }  // 找到递减子序列的结束位置，pi 指向最后一个递减元素的后一个位置

        for (pj = pl, pr = pi; pj < pr; ++pj, --pr) {  // 将递减子序列反转，pj 和 pr 分别从两端向中间移动
            std::swap(*pj, *pr);  // 交换 pj 和 pr 指向的元素
        }
    }

    ++pi;  // 将 pi 向后移动一位
    sz = pi - pl;  // 计算子序列的长度

    if (sz < minrun) {  // 如果子序列长度小于指定的最小运行长度
        if (l + minrun < num) {
            sz = minrun;  // 如果 l 加上最小运行长度小于 num，则将 sz 设为 minrun
        }
        else {
            sz = num - l;  // 否则将 sz 设为从 l 到 num 的长度
        }

        pr = pl + sz;  // pr 指向子序列的结束位置

        /* insertion sort */
        for (; pi < pr; ++pi) {  // 对子序列进行插入排序
            vc = *pi;  // 将当前元素赋值给 vc
            pj = pi;  // pj 指向当前元素的位置

            while (pl < pj && Tag::less(vc, *(pj - 1))) {  // 在已排序部分找到合适的位置插入 vc
                *pj = *(pj - 1);  // 元素右移
                --pj;
            }

            *pj = vc;  // 插入 vc 到合适位置
        }
    }

    return sz;  // 返回子序列的长度
/* 
 * 当数组左侧部分（p1）较小时，将p1复制到缓冲区并从左向右合并
 */
template <typename Tag, typename type>
static void
merge_left_(type *p1, npy_intp l1, type *p2, npy_intp l2, type *p3)
{
    // 计算p2数组的结束位置
    type *end = p2 + l2;
    // 将p1的内容复制到p3缓冲区
    memcpy(p3, p1, sizeof(type) * l1);
    // 将p2的第一个元素复制回p1，因为调用者将忽略第一个元素
    *p1++ = *p2++;

    while (p1 < p2 && p2 < end) {
        // 如果p2当前位置的元素比p3当前位置的元素小，将p2当前元素复制到p1，否则将p3当前元素复制到p1
        if (Tag::less(*p2, *p3)) {
            *p1++ = *p2++;
        }
        else {
            *p1++ = *p3++;
        }
    }

    // 处理剩余的元素，将p3缓冲区中剩余的元素复制回p1
    if (p1 != p2) {
        memcpy(p1, p3, sizeof(type) * (p2 - p1));
    }
}

/* 
 * 当数组右侧部分（p2）较小时，将p2复制到缓冲区并从右向左合并
 */
template <typename Tag, typename type>
static void
merge_right_(type *p1, npy_intp l1, type *p2, npy_intp l2, type *p3)
{
    npy_intp ofs;
    // 记录p1的起始位置
    type *start = p1 - 1;
    // 将p2的内容复制到p3缓冲区
    memcpy(p3, p2, sizeof(type) * l2);
    // 更新p1、p2和p3的位置
    p1 += l1 - 1;
    p2 += l2 - 1;
    p3 += l2 - 1;
    // 将p1的最后一个元素复制回p2，因为调用者将忽略最后一个元素
    *p2-- = *p1--;

    while (p1 < p2 && start < p1) {
        // 如果p3当前位置的元素比p1当前位置的元素小，将p1当前元素复制到p2，否则将p3当前元素复制到p2
        if (Tag::less(*p3, *p1)) {
            *p2-- = *p1--;
        }
        else {
            *p2-- = *p3--;
        }
    }

    // 处理剩余的元素，将p3缓冲区中剩余的元素复制回p1
    if (p1 != p2) {
        ofs = p2 - start;
        memcpy(start + 1, p3 - ofs + 1, sizeof(type) * ofs);
    }
}

/* 
 * 注意：gallop函数的命名约定与CPython不同。
 * 在这里，gallop_right表示从左向右gallop，
 * 而在CPython中，gallop_right表示gallop并找到相等元素中的最右元素。
 */
template <typename Tag, typename type>
static npy_intp
gallop_right_(const type *arr, const npy_intp size, const type key)
{
    npy_intp last_ofs, ofs, m;

    // 如果key比数组中第一个元素还小，直接返回0
    if (Tag::less(key, arr[0])) {
        return 0;
    }

    // 初始化gallop搜索过程的offsets
    last_ofs = 0;
    ofs = 1;

    for (;;) {
        // 如果ofs超出数组范围，或者小于0，则设为数组大小（arr[ofs]永远不会被访问）
        if (size <= ofs || ofs < 0) {
            ofs = size;
            break;
        }

        // 如果key小于等于arr[ofs]，退出循环；否则更新last_ofs并计算下一个ofs
        if (Tag::less(key, arr[ofs])) {
            break;
        }
        else {
            last_ofs = ofs;
            ofs = (ofs << 1) + 1; // ofs = 1, 3, 7, 15...
        }
    }

    // 使用二分搜索找到key的插入位置
    while (last_ofs + 1 < ofs) {
        m = last_ofs + ((ofs - last_ofs) >> 1);
        if (Tag::less(key, arr[m])) {
            ofs = m;
        }
        else {
            last_ofs = m;
        }
    }

    // 返回key的插入位置
    return ofs;
}

/* 
 * 注意：gallop函数的命名约定与CPython不同。
 * 在这里，gallop_left表示从右向左gallop，
 * 而在CPython中，gallop_left表示gallop并找到相等元素中的最左元素。
 */
template <typename Tag, typename type>
static npy_intp
gallop_left_(const type *arr, const npy_intp size, const type key)
{
    npy_intp last_ofs, ofs, l, m, r;

    // 如果key比数组中最后一个元素还大，直接返回数组大小
    if (Tag::less(arr[size - 1], key)) {
        return size;
    }

    // 初始化gallop搜索过程的offsets
    last_ofs = 0;
    ofs = 1;

    // 迭代执行gallop搜索
    for (;;) {
        if (size <= ofs || ofs < 0) {
            ofs = size; // arr[ofs] is never accessed
            break;
        }

        // 如果key小于等于arr[ofs]，退出循环；否则更新last_ofs并计算下一个ofs
        if (Tag::less(key, arr[ofs])) {
            break;
        }
        else {
            last_ofs = ofs;
            ofs = (ofs << 1) + 1; // ofs = 1, 3, 7, 15...
        }
    }

    // 使用二分搜索找到key的插入位置
    while (last_ofs + 1 < ofs) {
        m = last_ofs + ((ofs - last_ofs) >> 1);
        if (Tag::less(key, arr[m])) {
            ofs = m;
        }
        else {
            last_ofs = m;
        }
    }

    // 返回key的插入位置
    return ofs;
}
    # 进入一个无限循环，直到条件满足后跳出循环
    for (;;) {
        # 如果 ofs 大于等于 size 或者 ofs 小于 0，则将 ofs 设为 size 并跳出循环
        if (size <= ofs || ofs < 0) {
            ofs = size;
            break;
        }

        # 如果 arr[size-ofs-1] 小于 key，则跳出循环
        if (Tag::less(arr[size - ofs - 1], key)) {
            break;
        }
        else {
            # 否则更新 last_ofs，将 ofs 更新为 (ofs << 1) + 1
            last_ofs = ofs;
            ofs = (ofs << 1) + 1;
        }
    }

    /* 现在 arr[size-ofs-1] < key <= arr[size-last_ofs-1] */
    # 设定 l 和 r 的初始值，分别是 size-ofs-1 和 size-last_ofs-1
    l = size - ofs - 1;
    r = size - last_ofs - 1;

    # 进入二分查找循环，直到 l+1 >= r
    while (l + 1 < r) {
        # 取中间值 m，确保不会溢出
        m = l + ((r - l) >> 1);

        # 如果 arr[m] 小于 key，则更新 l
        if (Tag::less(arr[m], key)) {
            l = m;
        }
        else {
            # 否则更新 r
            r = m;
        }
    }

    /* 现在 arr[r-1] < key <= arr[r] */
    # 返回 r，即为查找到的位置
    return r;
// 尝试合并栈顶的三个子数组，使得合并后的子数组满足堆栈性质
template <typename Tag, typename type>
static int
try_collapse_(type *arr, run *stack, npy_intp *stack_ptr, buffer_<Tag> *buffer)
{
    int ret;
    npy_intp A, B, C, top;
    // 获取当前堆栈顶部索引
    top = *stack_ptr;

    // 只要堆栈中子数组个数大于1，就进行循环合并操作
    while (1 < top) {
        // 取出栈顶的三个子数组的长度信息
        B = stack[top - 2].l;
        C = stack[top - 1].l;

        // 检查是否可以合并前两个子数组
        if ((2 < top && stack[top - 3].l <= B + C) ||
            (3 < top && stack[top - 4].l <= stack[top - 3].l + B)) {
            // 取出第一个子数组的长度
            A = stack[top - 3].l;

            // 根据子数组长度大小进行合并操作
            if (A <= C) {
                // 调用合并函数进行合并，并更新堆栈信息
                ret = merge_at_<Tag>(arr, stack, top - 3, buffer);

                // 如果合并失败则返回错误代码
                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }

                // 更新合并后的子数组长度和堆栈状态
                stack[top - 3].l += B;
                stack[top - 2] = stack[top - 1];
                --top;
            }
            else {
                // 同上，根据不同情况调用不同位置的合并函数
                ret = merge_at_<Tag>(arr, stack, top - 2, buffer);

                // 如果合并失败则返回错误代码
                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }

                // 更新合并后的子数组长度和堆栈状态
                stack[top - 2].l += C;
                --top;
            }
        }
        else if (1 < top && B <= C) {
            // 合并前两个子数组，更新堆栈状态
            ret = merge_at_<Tag>(arr, stack, top - 2, buffer);

            // 如果合并失败则返回错误代码
            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            // 更新合并后的子数组长度和堆栈状态
            stack[top - 2].l += C;
            --top;
        }
        else {
            // 如果无法继续合并则退出循环
            break;
        }
    }

    // 更新堆栈顶部指针位置
    *stack_ptr = top;
    // 返回操作成功
    return 0;
}
    # 当堆栈中剩余元素数量大于2时，执行循环
    while (2 < top):
        # 如果堆栈倒数第三个元素的长度小于等于倒数第一个元素的长度
        if (stack[top - 3].l <= stack[top - 1].l):
            # 调用特定标签的 merge_at_ 函数，合并 arr 中的数据，结果存入 buffer 中
            ret = merge_at_<Tag>(arr, stack, top - 3, buffer);

            # 如果合并操作返回值小于 0，直接返回该值
            if (NPY_UNLIKELY(ret < 0)):
                return ret;

            # 将堆栈倒数第三个元素的长度增加堆栈倒数第二个元素的长度
            stack[top - 3].l += stack[top - 2].l;
            # 将堆栈倒数第二个元素更新为堆栈倒数第一个元素
            stack[top - 2] = stack[top - 1];
            # 减少堆栈元素数量
            --top;
        else:
            # 否则，调用特定标签的 merge_at_ 函数，合并 arr 中的数据，结果存入 buffer 中
            ret = merge_at_<Tag>(arr, stack, top - 2, buffer);

            # 如果合并操作返回值小于 0，直接返回该值
            if (NPY_UNLIKELY(ret < 0)):
                return ret;

            # 将堆栈倒数第二个元素的长度增加堆栈倒数第一个元素的长度
            stack[top - 2].l += stack[top - 1].l;
            # 减少堆栈元素数量
            --top;

    # 如果堆栈中剩余元素数量大于 1
    if (1 < top):
        # 调用特定标签的 merge_at_ 函数，合并 arr 中的数据，结果存入 buffer 中
        ret = merge_at_<Tag>(arr, stack, top - 2, buffer);

        # 如果合并操作返回值小于 0，直接返回该值
        if (NPY_UNLIKELY(ret < 0)):
            return ret;

    # 循环结束，返回 0 表示成功
    return 0;
}

/* 结束 timsort_ 函数 */

template <typename Tag>
static int
timsort_(void *start, npy_intp num)
{
    using type = typename Tag::type;
    int ret;
    npy_intp l, n, stack_ptr, minrun;
    buffer_<Tag> buffer;
    run stack[TIMSORT_STACK_SIZE];
    buffer.pw = NULL;
    buffer.size = 0;
    stack_ptr = 0;
    minrun = compute_min_run(num);

    for (l = 0; l < num;) {
        n = count_run_<Tag>((type *)start, l, num, minrun);
        stack[stack_ptr].s = l;
        stack[stack_ptr].l = n;
        ++stack_ptr;
        ret = try_collapse_<Tag>((type *)start, stack, &stack_ptr, &buffer);

        if (NPY_UNLIKELY(ret < 0)) {
            goto cleanup;
        }

        l += n;
    }

    ret = force_collapse_<Tag>((type *)start, stack, &stack_ptr, &buffer);

    if (NPY_UNLIKELY(ret < 0)) {
        goto cleanup;
    }

    ret = 0;

cleanup:

    free(buffer.pw);

    return ret;
}

/* argsort */

/* 结束 timsort_ 函数 */

template <typename Tag, typename type>
static npy_intp
acount_run_(type *arr, npy_intp *tosort, npy_intp l, npy_intp num,
            npy_intp minrun)
{
    npy_intp sz;
    type vc;
    npy_intp vi;
    npy_intp *pl, *pi, *pj, *pr;

    if (NPY_UNLIKELY(num - l == 1)) {
        return 1;
    }

    pl = tosort + l;

    /* (not strictly) ascending sequence */
    if (!Tag::less(arr[*(pl + 1)], arr[*pl])) {
        for (pi = pl + 1;
             pi < tosort + num - 1 && !Tag::less(arr[*(pi + 1)], arr[*pi]);
             ++pi) {
        }
    }
    else { /* (strictly) descending sequence */
        for (pi = pl + 1;
             pi < tosort + num - 1 && Tag::less(arr[*(pi + 1)], arr[*pi]);
             ++pi) {
        }

        for (pj = pl, pr = pi; pj < pr; ++pj, --pr) {
            std::swap(*pj, *pr);
        }
    }

    ++pi;
    sz = pi - pl;

    if (sz < minrun) {
        if (l + minrun < num) {
            sz = minrun;
        }
        else {
            sz = num - l;
        }

        pr = pl + sz;

        /* insertion sort */
        for (; pi < pr; ++pi) {
            vi = *pi;
            vc = arr[*pi];
            pj = pi;

            while (pl < pj && Tag::less(vc, arr[*(pj - 1)])) {
                *pj = *(pj - 1);
                --pj;
            }

            *pj = vi;
        }
    }

    return sz;
}

/* 结束 acount_run_ 函数 */

template <typename Tag, typename type>
static npy_intp
agallop_right_(const type *arr, const npy_intp *tosort, const npy_intp size,
               const type key)
{
    npy_intp last_ofs, ofs, m;

    if (Tag::less(key, arr[tosort[0]])) {
        return 0;
    }

    last_ofs = 0;
    ofs = 1;

    for (;;) {
        if (size <= ofs || ofs < 0) {
            ofs = size; /* arr[ofs] is never accessed */
            break;
        }

        if (Tag::less(key, arr[tosort[ofs]])) {
            break;
        }
        else {
            last_ofs = ofs;
            /* ofs = 1, 3, 7, 15... */
            ofs = (ofs << 1) + 1;
        }
    }

    /* now that arr[tosort[last_ofs]] <= key < arr[tosort[ofs]] */



注释：
    # 当上一个偏移量加一小于当前偏移量时，执行循环
    while (last_ofs + 1 < ofs) {
        # 使用二分法计算中间值 m
        m = last_ofs + ((ofs - last_ofs) >> 1);

        # 检查 key 是否小于 arr[tosort[m]] 所对应的值
        if (Tag::less(key, arr[tosort[m]])) {
            # 若是，则将 ofs 缩小为 m，继续在前半部分查找
            ofs = m;
        }
        else {
            # 否则，将 last_ofs 移动到 m，继续在后半部分查找
            last_ofs = m;
        }
    }

    /* 现在 arr[tosort[ofs-1]] <= key < arr[tosort[ofs]] */
    # 返回找到的偏移量 ofs，保证 arr[tosort[ofs-1]] 小于等于 key，arr[tosort[ofs]] 大于 key
    return ofs;
    /* 
     * 计算第一个位置（left）和最后一个位置（right），使得 Tag::less(arr[tosort[left]], key) 为真
     * 并且 Tag::less(arr[tosort[right]], key) 为假
     */
    npy_intp last_ofs, ofs, l, m, r;

    // 如果数组最后一个元素小于 key，则返回数组大小
    if (Tag::less(arr[tosort[size - 1]], key)) {
        return size;
    }

    // 初始偏移量设定
    last_ofs = 0;
    ofs = 1;

    // 二分查找最大的 ofs，使得 arr[tosort[size-ofs-1]] < key
    for (;;) {
        if (size <= ofs || ofs < 0) {
            ofs = size;
            break;
        }

        if (Tag::less(arr[tosort[size - ofs - 1]], key)) {
            break;
        }
        else {
            last_ofs = ofs;
            ofs = (ofs << 1) + 1;
        }
    }

    /* 
     * 现在 arr[tosort[size-ofs-1]] < key <= arr[tosort[size-last_ofs-1]]
     */
    l = size - ofs - 1;
    r = size - last_ofs - 1;

    // 二分查找使得 arr[tosort[r-1]] < key <= arr[tosort[r]]
    while (l + 1 < r) {
        m = l + ((r - l) >> 1);

        if (Tag::less(arr[tosort[m]], key)) {
            l = m;
        }
        else {
            r = m;
        }
    }

    /* 
     * 现在 arr[tosort[r-1]] < key <= arr[tosort[r]]
     */
    return r;
}
    # 如果第二个长度 l2 小于第一个长度 l1，则执行以下操作
    if (l2 < l1) {
        # 调整缓冲区大小为 l2，并将结果赋给 ret
        ret = resize_buffer_intp(buffer, l2);
    
        # 如果 ret 小于 0（不常见的情况），则直接返回 ret
        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }
    
        # 调用特定标签的右侧合并函数 amerge_right_<Tag>，将数组 arr 中从 p1 开始长度为 l1 的部分，
        # 与从 p2 开始长度为 l2 的部分进行合并，结果存储在 buffer->pw 中
        amerge_right_<Tag>(arr, p1, l1, p2, l2, buffer->pw);
    }
    # 如果第二个长度 l2 不小于第一个长度 l1，则执行以下操作
    else {
        # 调整缓冲区大小为 l1，并将结果赋给 ret
        ret = resize_buffer_intp(buffer, l1);
    
        # 如果 ret 小于 0（不常见的情况），则直接返回 ret
        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }
    
        # 调用特定标签的左侧合并函数 amerge_left_<Tag>，将数组 arr 中从 p1 开始长度为 l1 的部分，
        # 与从 p2 开始长度为 l2 的部分进行合并，结果存储在 buffer->pw 中
        amerge_left_<Tag>(arr, p1, l1, p2, l2, buffer->pw);
    }
    
    # 返回值为 0，表示操作成功完成
    return 0;
template <typename Tag, typename type>
static int
atry_collapse_(type *arr, npy_intp *tosort, run *stack, npy_intp *stack_ptr,
               buffer_intp *buffer)
{
    int ret;
    npy_intp A, B, C, top;
    top = *stack_ptr;

    while (1 < top) {
        B = stack[top - 2].l;  // 获取栈顶倒数第二个元素的长度 B
        C = stack[top - 1].l;  // 获取栈顶倒数第一个元素的长度 C

        // 判断是否可以进行合并
        if ((2 < top && stack[top - 3].l <= B + C) ||
            (3 < top && stack[top - 4].l <= stack[top - 3].l + B)) {
            A = stack[top - 3].l;  // 获取栈顶倒数第三个元素的长度 A

            if (A <= C) {
                // 调用合并函数，并更新栈顶元素及栈大小
                ret = amerge_at_<Tag>(arr, tosort, stack, top - 3, buffer);

                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }

                stack[top - 3].l += B;  // 更新栈顶倒数第三个元素的长度
                stack[top - 2] = stack[top - 1];  // 更新栈顶倒数第二个元素
                --top;  // 减小栈大小
            }
            else {
                // 调用合并函数，并更新栈顶元素及栈大小
                ret = amerge_at_<Tag>(arr, tosort, stack, top - 2, buffer);

                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }

                stack[top - 2].l += C;  // 更新栈顶倒数第二个元素的长度
                --top;  // 减小栈大小
            }
        }
        else if (1 < top && B <= C) {
            // 调用合并函数，并更新栈顶元素及栈大小
            ret = amerge_at_<Tag>(arr, tosort, stack, top - 2, buffer);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 2].l += C;  // 更新栈顶倒数第二个元素的长度
            --top;  // 减小栈大小
        }
        else {
            break;  // 跳出循环
        }
    }

    *stack_ptr = top;  // 更新传入的栈指针值
    return 0;  // 返回成功
}

template <typename Tag, typename type>
static int
aforce_collapse_(type *arr, npy_intp *tosort, run *stack, npy_intp *stack_ptr,
                 buffer_intp *buffer)
{
    int ret;
    npy_intp top = *stack_ptr;

    while (2 < top) {
        // 判断是否可以进行强制合并
        if (stack[top - 3].l <= stack[top - 1].l) {
            // 调用合并函数，并更新栈顶元素及栈大小
            ret = amerge_at_<Tag>(arr, tosort, stack, top - 3, buffer);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 3].l += stack[top - 2].l;  // 更新栈顶倒数第三个元素的长度
            stack[top - 2] = stack[top - 1];  // 更新栈顶倒数第二个元素
            --top;  // 减小栈大小
        }
        else {
            // 调用合并函数，并更新栈顶元素及栈大小
            ret = amerge_at_<Tag>(arr, tosort, stack, top - 2, buffer);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 2].l += stack[top - 1].l;  // 更新栈顶倒数第二个元素的长度
            --top;  // 减小栈大小
        }
    }

    if (1 < top) {
        // 如果还有剩余元素，调用合并函数，并更新栈顶元素及栈大小
        ret = amerge_at_<Tag>(arr, tosort, stack, top - 2, buffer);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }
    }

    return 0;  // 返回成功
}

template <typename Tag>
static int
atimsort_(void *v, npy_intp *tosort, npy_intp num)
{
    using type = typename Tag::type;
    int ret;
    npy_intp l, n, stack_ptr, minrun;
    buffer_intp buffer;
    run stack[TIMSORT_STACK_SIZE];
    buffer.pw = NULL;
    buffer.size = 0;
    stack_ptr = 0;
    minrun = compute_min_run(num);  // 计算最小运行长度

    // 初始化堆栈和缓冲区
    # 使用变量 l 进行循环迭代，直到 l 小于 num 为止
    for (l = 0; l < num;) {
        # 调用 acount_run_<Tag> 函数，计算运行长度编码，返回值存入 n 中
        n = acount_run_<Tag>((type *)v, tosort, l, num, minrun);
        # 将当前 l 和计算得到的 n 压入栈中，用于后续处理
        stack[stack_ptr].s = l;
        stack[stack_ptr].l = n;
        # 栈指针加一，准备处理下一个元素
        ++stack_ptr;
        # 调用 atry_collapse_<Tag> 函数，尝试将部分排序块合并，结果存入 ret 中
        ret = atry_collapse_<Tag>((type *)v, tosort, stack, &stack_ptr,
                                  &buffer);

        # 如果 ret 小于 0，表示操作失败，跳转到 cleanup 处理清理工作
        if (NPY_UNLIKELY(ret < 0)) {
            goto cleanup;
        }

        # 更新 l 的值，准备处理下一段排序块
        l += n;
    }

    # 强制合并剩余的排序块，结果存入 ret 中
    ret = aforce_collapse_<Tag>((type *)v, tosort, stack, &stack_ptr, &buffer);

    # 如果 ret 小于 0，表示操作失败，跳转到 cleanup 处理清理工作
    if (NPY_UNLIKELY(ret < 0)) {
        goto cleanup;
    }

    # 将 ret 置为 0，表示整个过程执行成功
    ret = 0;
cleanup:
    # 如果缓冲区中的指针不为空，则释放其内存空间
    if (buffer.pw != NULL) {
        free(buffer.pw);
    }
    
    # 返回函数的结果
    return ret;
}

/* For string sorts and generic sort, element comparisons are very expensive,
 * and the time cost of insertion sort (involves N**2 comparison) clearly
 * hurts. Implementing binary insertion sort and probably gallop mode during
 * merging process can hopefully boost the performance. Here as a temporary
 * workaround we use shorter run length to reduce the cost of insertion sort.
 */

static npy_intp
compute_min_run_short(npy_intp num)
{
    npy_intp r = 0;
    
    # 当 num 大于 16 时，不断右移 num 并设置 r 为 num 的最低位
    while (16 < num) {
        r |= num & 1;
        num >>= 1;
    }
    
    # 返回计算得到的 num 加上 r 的结果
    return num + r;
}

/*
 *****************************************************************************
 **                             STRING SORTS                                **
 *****************************************************************************
 */

template <typename Tag>
struct string_buffer_ {
    typename Tag::type *pw;  // 指向字符串缓冲区的指针
    npy_intp size;           // 缓冲区的大小
    size_t len;              // 字符串元素的长度
};

template <typename Tag>
static inline int
resize_buffer_(string_buffer_<Tag> *buffer, npy_intp new_size)
{
    using type = typename Tag::type;
    
    # 如果请求的新大小小于等于当前缓冲区的大小，则直接返回
    if (new_size <= buffer->size) {
        return 0;
    }
    
    // 重新分配内存以扩展缓冲区大小
    type *new_pw = (type *)realloc(buffer->pw, sizeof(type) * new_size * buffer->len);
    buffer->size = new_size;
    
    # 如果内存分配失败，则返回内存不足错误
    if (NPY_UNLIKELY(new_pw == NULL)) {
        return -NPY_ENOMEM;
    }
    else {
        buffer->pw = new_pw;
        return 0;
    }
}

template <typename Tag, typename type>
static npy_intp
count_run_(type *arr, npy_intp l, npy_intp num, npy_intp minrun, type *vp,
           size_t len)
{
    npy_intp sz;
    type *pl, *pi, *pj, *pr;
    
    # 如果 l 和 num 之间的距离为 1，返回 1
    if (NPY_UNLIKELY(num - l == 1)) {
        return 1;
    }
    
    pl = arr + l * len;
    
    /* (not strictly) ascending sequence */
    // 如果不是严格升序序列，则向后查找升序序列
    if (!Tag::less(pl + len, pl, len)) {
        for (pi = pl + len;
             pi < arr + (num - 1) * len && !Tag::less(pi + len, pi, len);
             pi += len) {
        }
    }
    else { /* (strictly) descending sequence */
        // 如果是严格降序序列，则向后查找降序序列，并将其反转为升序
        for (pi = pl + len;
             pi < arr + (num - 1) * len && Tag::less(pi + len, pi, len);
             pi += len) {
        }

        for (pj = pl, pr = pi; pj < pr; pj += len, pr -= len) {
            Tag::swap(pj, pr, len);
        }
    }
    
    pi += len;
    sz = (pi - pl) / len;
    
    # 如果当前序列长度小于指定的最小运行长度，则进行插入排序
    if (sz < minrun) {
        if (l + minrun < num) {
            sz = minrun;
        }
        else {
            sz = num - l;
        }
        
        pr = pl + sz * len;
        
        // 插入排序
        for (; pi < pr; pi += len) {
            Tag::copy(vp, pi, len);
            pj = pi;
            
            while (pl < pj && Tag::less(vp, pj - len, len)) {
                Tag::copy(pj, pj - len, len);
                pj -= len;
            }
            
            Tag::copy(pj, vp, len);
        }
    }
    
    // 返回当前运行的长度
    return sz;
}
// 在排序数组 arr 中，使用 Gallop Right 算法找到第一个大于给定 key 的位置索引
gallop_right_(const typename Tag::type *arr, const npy_intp size,
              const typename Tag::type *key, size_t len)
{
    npy_intp last_ofs, ofs, m;

    // 如果 key 小于 arr 的第一个元素，直接返回 0
    if (Tag::less(key, arr, len)) {
        return 0;
    }

    // 初始化 gallop right 的搜索参数
    last_ofs = 0;
    ofs = 1;

    // 开始 gallop right 搜索
    for (;;) {
        // 如果 ofs 超出数组范围或小于 0，将 ofs 调整为 size，表示越界
        if (size <= ofs || ofs < 0) {
            ofs = size; /* arr[ofs] is never accessed */
            break;
        }

        // 比较 key 与 arr[ofs*len] 处的元素，若 key 小于该元素则退出循环
        if (Tag::less(key, arr + ofs * len, len)) {
            break;
        }
        else {
            // 更新 last_ofs，并按照 1, 3, 7, 15... 的规律增加 ofs
            last_ofs = ofs;
            ofs = (ofs << 1) + 1; // ofs = 1, 3, 7, 15...
        }
    }

    // 现在 arr[last_ofs*len] <= key < arr[ofs*len]

    // 使用二分法细化搜索区间
    while (last_ofs + 1 < ofs) {
        m = last_ofs + ((ofs - last_ofs) >> 1);

        // 比较 key 与 arr[m*len] 处的元素，调整搜索区间
        if (Tag::less(key, arr + m * len, len)) {
            ofs = m;
        }
        else {
            last_ofs = m;
        }
    }

    // 现在 arr[(ofs-1)*len] <= key < arr[ofs*len]

    // 返回 gallop right 的最终结果，即第一个大于 key 的位置索引
    return ofs;
}

// 在排序数组 arr 中，使用 Gallop Left 算法找到第一个大于等于给定 key 的位置索引
template <typename Tag>
static npy_intp
gallop_left_(const typename Tag::type *arr, const npy_intp size,
             const typename Tag::type *key, size_t len)
{
    npy_intp last_ofs, ofs, l, m, r;

    // 如果 arr 的最后一个元素小于 key，则直接返回 size，表示 key 大于数组所有元素
    if (Tag::less(arr + (size - 1) * len, key, len)) {
        return size;
    }

    // 初始化 gallop left 的搜索参数
    last_ofs = 0;
    ofs = 1;

    // 开始 gallop left 搜索
    for (;;) {
        // 如果 ofs 超出数组范围或小于 0，将 ofs 调整为 size，表示越界
        if (size <= ofs || ofs < 0) {
            ofs = size;
            break;
        }

        // 比较 arr[(size-ofs-1)*len] 处的元素与 key，若 arr 小于 key 则退出循环
        if (Tag::less(arr + (size - ofs - 1) * len, key, len)) {
            break;
        }
        else {
            // 更新 last_ofs，并按照 1, 3, 7, 15... 的规律增加 ofs
            last_ofs = ofs;
            ofs = (ofs << 1) + 1; // ofs = 1, 3, 7, 15...
        }
    }

    // 现在 arr[(size-ofs-1)*len] < key <= arr[(size-last_ofs-1)*len]

    // 使用二分法细化搜索区间
    l = size - ofs - 1;
    r = size - last_ofs - 1;

    while (l + 1 < r) {
        m = l + ((r - l) >> 1);

        // 比较 arr[m*len] 处的元素与 key，调整搜索区间
        if (Tag::less(arr + m * len, key, len)) {
            l = m;
        }
        else {
            r = m;
        }
    }

    // 现在 arr[(r-1)*len] < key <= arr[r*len]

    // 返回 gallop left 的最终结果，即第一个大于等于 key 的位置索引
    return r;
}

// 将排序数组 p1 和 p2 合并到数组 p3 中，以保持排序状态
template <typename Tag>
static void
merge_left_(typename Tag::type *p1, npy_intp l1, typename Tag::type *p2,
            npy_intp l2, typename Tag::type *p3, size_t len)
{
    using type = typename Tag::type;
    type *end = p2 + l2 * len;

    // 将 p1 的内容复制到 p3 中
    memcpy(p3, p1, sizeof(type) * l1 * len);

    // 将 p2 的第一个元素复制到 p1，并调整指针
    Tag::copy(p1, p2, len);
    p1 += len;
    p2 += len;

    // 开始合并排序数组 p1 和 p2 到 p3
    while (p1 < p2 && p2 < end) {
        if (Tag::less(p2, p3, len)) {
            // 如果 p2 的元素小于 p3 的元素，将 p2 的元素复制到 p1
            Tag::copy(p1, p2, len);
            p1 += len;
            p2 += len;
        }
        else {
            // 否则将 p3 的元素复制到 p1，并调整指针
            Tag::copy(p1, p3, len);
            p1 += len;
            p3 += len;
        }
    }

    // 如果 p1 没有完全复制完成，将 p3 剩余的元素复制到 p1
    if (p1 != p2) {
        memcpy(p1, p3, sizeof(type) * (p2 - p1));
    }
}

// 将排序数组 p1 和 p2 合并到数组 p3 中，以保持排序状态（逆向合并）
template <typename Tag, typename type>
static void
merge_right_(type *p1, npy_intp l1, type *p2, npy_intp l2, type *p3,
             size_t len)
{
    npy_intp ofs;
    type *start = p1 - len;

    // 将 p2 的内容复制到 p3 中
    memcpy(p3, p2, sizeof(type) * l2 * len);

    // 将 p1 的最后一个元素指针调整到合适位置
    p1 += (l1 - 1) * len;

    // 开始逆向合并排序数组 p1 和 p2 到 p3
    // 更新指针 p2，使其指向第一个元素之前的位置
    p2 += (l2 - 1) * len;
    // 更新指针 p3，使其指向第一个元素之前的位置
    p3 += (l2 - 1) * len;
    /* 第一个元素必须在 p1 中，否则会在调用方中被跳过 */
    Tag::copy(p2, p1, len);
    // 将指针 p2 和 p1 向前移动一个元素的长度
    p2 -= len;
    p1 -= len;

    // 当 p1 小于 p2 且 start 小于 p1 时，执行循环
    while (p1 < p2 && start < p1) {
        // 如果 p3 所指向的元素小于 p1 所指向的元素
        if (Tag::less(p3, p1, len)) {
            // 将 p1 所指向的元素复制到 p2
            Tag::copy(p2, p1, len);
            // 更新指针 p2 和 p1，使其向前移动一个元素的长度
            p2 -= len;
            p1 -= len;
        }
        // 否则
        else {
            // 将 p3 所指向的元素复制到 p2
            Tag::copy(p2, p3, len);
            // 更新指针 p2 和 p3，使其向前移动一个元素的长度
            p2 -= len;
            p3 -= len;
        }
    }

    // 如果 p1 不等于 p2
    if (p1 != p2) {
        // 计算偏移量 ofs
        ofs = p2 - start;
        // 将 p3 - ofs + len 处的数据复制到 start + len 处
        memcpy(start + len, p3 - ofs + len, sizeof(type) * ofs);
    }
// 定义静态函数模板 `merge_at_`，合并两个有序子数组的算法
template <typename Tag, typename type>
static int
merge_at_(type *arr, const run *stack, const npy_intp at,
          string_buffer_<Tag> *buffer, size_t len)
{
    int ret;
    npy_intp s1, l1, s2, l2, k;
    type *p1, *p2;
    
    // 获取第一个子数组的起始位置和长度
    s1 = stack[at].s;
    l1 = stack[at].l;
    
    // 获取第二个子数组的起始位置和长度
    s2 = stack[at + 1].s;
    l2 = stack[at + 1].l;
    
    // 将第二个子数组的第一个元素复制到缓冲区
    Tag::copy(buffer->pw, arr + s2 * len, len);
    
    // 在第一个子数组中寻找第二个子数组的起始元素应插入的位置
    k = gallop_right_<Tag>(arr + s1 * len, l1, buffer->pw, len);

    if (l1 == k) {
        // 如果找到的位置已经是有序的，直接返回
        return 0;
    }

    // 更新第一个子数组的起始位置和长度
    p1 = arr + (s1 + k) * len;
    l1 -= k;
    
    // 更新第二个子数组的起始位置
    p2 = arr + s2 * len;
    
    // 将第二个子数组的最后一个元素复制到缓冲区
    Tag::copy(buffer->pw, arr + (s2 - 1) * len, len);
    
    // 在第二个子数组中寻找第一个子数组的最后一个元素应插入的位置
    l2 = gallop_left_<Tag>(arr + s2 * len, l2, buffer->pw, len);

    if (l2 < l1) {
        // 如果第二个子数组的插入点在第一个子数组之前，调整缓冲区大小并执行右侧合并
        ret = resize_buffer_<Tag>(buffer, l2);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }

        merge_right_<Tag>(p1, l1, p2, l2, buffer->pw, len);
    }
    else {
        // 否则，调整缓冲区大小并执行左侧合并
        ret = resize_buffer_<Tag>(buffer, l1);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }

        merge_left_<Tag>(p1, l1, p2, l2, buffer->pw, len);
    }

    // 返回成功标志
    return 0;
}

// 定义静态函数模板 `try_collapse_`，尝试合并栈中相邻且可合并的三个子数组
template <typename Tag, typename type>
static int
try_collapse_(type *arr, run *stack, npy_intp *stack_ptr,
              string_buffer_<Tag> *buffer, size_t len)
{
    int ret;
    npy_intp A, B, C, top;
    
    // 获取当前栈顶索引
    top = *stack_ptr;

    while (1 < top) {
        // 获取栈顶三个子数组的长度
        B = stack[top - 2].l;
        C = stack[top - 1].l;

        // 检查是否可以合并前两个子数组
        if ((2 < top && stack[top - 3].l <= B + C) ||
            (3 < top && stack[top - 4].l <= stack[top - 3].l + B)) {
            A = stack[top - 3].l;

            // 根据长度比较结果选择合并的方式，并更新栈信息
            if (A <= C) {
                ret = merge_at_<Tag>(arr, stack, top - 3, buffer, len);

                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }

                stack[top - 3].l += B;
                stack[top - 2] = stack[top - 1];
                --top;
            }
            else {
                ret = merge_at_<Tag>(arr, stack, top - 2, buffer, len);

                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }

                stack[top - 2].l += C;
                --top;
            }
        }
        else if (1 < top && B <= C) {
            // 只合并前两个子数组，并更新栈信息
            ret = merge_at_<Tag>(arr, stack, top - 2, buffer, len);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 2].l += C;
            --top;
        }
        else {
            // 如果无法继续合并，则退出循环
            break;
        }
    }

    // 更新栈顶索引并返回成功标志
    *stack_ptr = top;
    return 0;
}

// 定义静态函数模板 `force_collapse_`，强制合并栈中所有子数组
template <typename Tag, typename type>
static int
force_collapse_(type *arr, run *stack, npy_intp *stack_ptr,
                string_buffer_<Tag> *buffer, size_t len)
{
    int ret;
    npy_intp top = *stack_ptr;

    // 循环合并直到栈中只剩一个子数组
    while (1 < top) {
        // 合并前两个子数组，并更新栈信息
        ret = merge_at_<Tag>(arr, stack, top - 2, buffer, len);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }

        stack[top - 2].l += stack[top - 1].l;
        --top;
    }

    // 更新栈顶索引并返回成功标志
    *stack_ptr = top;
    return 0;
}
    # 当堆栈中的元素个数大于2时，执行循环
    while (2 < top) {
        # 如果堆栈倒数第三个元素的长度小于等于倒数第一个元素的长度
        if (stack[top - 3].l <= stack[top - 1].l) {
            # 调用 merge_at_<Tag> 函数进行合并操作，返回合并结果
            ret = merge_at_<Tag>(arr, stack, top - 3, buffer, len);

            # 如果合并操作返回小于0的值（异常情况）
            if (NPY_UNLIKELY(ret < 0)) {
                # 返回异常值
                return ret;
            }

            # 更新堆栈中倒数第三个元素的长度为原长度加上倒数第二个元素的长度
            stack[top - 3].l += stack[top - 2].l;
            # 将堆栈中倒数第二个元素替换为倒数第一个元素
            stack[top - 2] = stack[top - 1];
            # 减少堆栈元素个数
            --top;
        }
        else {
            # 否则，调用 merge_at_<Tag> 函数进行合并操作，返回合并结果
            ret = merge_at_<Tag>(arr, stack, top - 2, buffer, len);

            # 如果合并操作返回小于0的值（异常情况）
            if (NPY_UNLIKELY(ret < 0)) {
                # 返回异常值
                return ret;
            }

            # 更新堆栈中倒数第二个元素的长度为原长度加上倒数第一个元素的长度
            stack[top - 2].l += stack[top - 1].l;
            # 减少堆栈元素个数
            --top;
        }
    }

    # 处理堆栈中剩余的元素（如果堆栈中有大于1个元素）
    if (1 < top) {
        # 调用 merge_at_<Tag> 函数进行合并操作，返回合并结果
        ret = merge_at_<Tag>(arr, stack, top - 2, buffer, len);

        # 如果合并操作返回小于0的值（异常情况）
        if (NPY_UNLIKELY(ret < 0)) {
            # 返回异常值
            return ret;
        }
    }

    # 操作成功，返回0
    return 0;
}

/* 
   template <typename Tag>
   NPY_NO_EXPORT int
   string_timsort_(void *start, npy_intp num, void *varr)
   {
       using type = typename Tag::type;
       PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(varr);
       size_t elsize = PyArray_ITEMSIZE(arr);
       size_t len = elsize / sizeof(type);
       int ret;
       npy_intp l, n, stack_ptr, minrun;
       run stack[TIMSORT_STACK_SIZE];
       string_buffer_<Tag> buffer;

       // Items that have zero size don't make sense to sort
       if (len == 0) {
           return 0;
       }

       buffer.pw = NULL;
       buffer.size = 0;
       buffer.len = len;
       stack_ptr = 0;
       minrun = compute_min_run_short(num);
       // used for insertion sort and gallop key
       ret = resize_buffer_<Tag>(&buffer, 1);

       if (NPY_UNLIKELY(ret < 0)) {
           goto cleanup;
       }

       for (l = 0; l < num;) {
           n = count_run_<Tag>((type *)start, l, num, minrun, buffer.pw, len);
           // both s and l are scaled by len
           stack[stack_ptr].s = l;
           stack[stack_ptr].l = n;
           ++stack_ptr;
           ret = try_collapse_<Tag>((type *)start, stack, &stack_ptr, &buffer,
                                    len);

           if (NPY_UNLIKELY(ret < 0)) {
               goto cleanup;
           }

           l += n;
       }

       ret = force_collapse_<Tag>((type *)start, stack, &stack_ptr, &buffer, len);

       if (NPY_UNLIKELY(ret < 0)) {
           goto cleanup;
       }

       ret = 0;

   cleanup:
       if (buffer.pw != NULL) {
           free(buffer.pw);
       }
       return ret;
   }

   // argsort

   template <typename Tag, typename type>
   static npy_intp
   acount_run_(type *arr, npy_intp *tosort, npy_intp l, npy_intp num,
               npy_intp minrun, size_t len)
   {
       npy_intp sz;
       npy_intp vi;
       npy_intp *pl, *pi, *pj, *pr;

       if (NPY_UNLIKELY(num - l == 1)) {
           return 1;
       }

       pl = tosort + l;

       // (not strictly) ascending sequence
       if (!Tag::less(arr + (*(pl + 1)) * len, arr + (*pl) * len, len)) {
           for (pi = pl + 1;
                pi < tosort + num - 1 &&
                !Tag::less(arr + (*(pi + 1)) * len, arr + (*pi) * len, len);
                ++pi) {
           }
       }
       else { // (strictly) descending sequence
           for (pi = pl + 1;
                pi < tosort + num - 1 &&
                Tag::less(arr + (*(pi + 1)) * len, arr + (*pi) * len, len);
                ++pi) {
           }

           for (pj = pl, pr = pi; pj < pr; ++pj, --pr) {
               std::swap(*pj, *pr);
           }
       }

       ++pi;
       sz = pi - pl;

       if (sz < minrun) {
           if (l + minrun < num) {
               sz = minrun;
           }
           else {
               sz = num - l;
           }

           pr = pl + sz;

           // insertion sort
           for (; pi < pr; ++pi) {
               vi = *pi;
               pj = pi;

               while (pl < pj &&
                      Tag::less(arr + vi * len, arr + (*(pj - 1)) * len, len)) {
                   *pj = *(pj - 1);
                   --pj;
               }

               *pj = vi;
           }
       }

       return sz;
   }
# 左侧二分搜索函数，用于找到数组中第一个大于等于给定键值的位置索引
template <typename Tag, typename type>
static npy_intp
agallop_left_(const type *arr, const npy_intp *tosort, const npy_intp size,
              const type *key, size_t len)
{
    npy_intp last_ofs, ofs, l, m, r;

    // 如果数组中最后一个元素对应的键值小于给定键值，则直接返回数组大小
    if (Tag::less(arr + tosort[size - 1] * len, key, len)) {
        return size;
    }

    // 初始化二分搜索的起始偏移量
    last_ofs = 0;
    ofs = 1;

    for (;;) {
        // 若偏移量超出数组大小或者小于0，则将偏移量设置为数组大小，并结束循环
        if (size <= ofs || ofs < 0) {
            ofs = size;
            break;
        }

        // 如果指定位置对应的键值小于给定键值，则退出循环；否则更新 last_ofs 并加倍偏移量
        if (Tag::less(arr + tosort[size - ofs - 1] * len, key, len)) {
            break;
        }
        else {
            last_ofs = ofs;
            ofs = (ofs << 1) + 1;
        }
    }

    /* 现在 arr[tosort[size-ofs-1]*len] < key <= arr[tosort[size-last_ofs-1]*len] */
    // 确定左侧边界 l 和右侧边界 r
    l = size - ofs - 1;
    r = size - last_ofs - 1;

    while (l + 1 < r) {
        // 计算中间位置 m
        m = l + ((r - l) >> 1);

        // 根据键值比较调整边界 l 或 r
        if (Tag::less(arr + tosort[m] * len, key, len)) {
            l = m;
        }
        else {
            r = m;
        }
    }

    /* 现在 arr[tosort[r-1]*len] < key <= arr[tosort[r]*len] */
    return r;
}

# 右侧二分搜索函数，用于找到数组中第一个大于给定键值的位置索引
template <typename Tag, typename type>
static npy_intp
agallop_right_(const type *arr, const npy_intp *tosort, const npy_intp size,
               const type *key, size_t len)
{
    npy_intp last_ofs, ofs, m;

    // 如果给定键值小于数组中第一个元素的键值，则直接返回0
    if (Tag::less(key, arr + tosort[0] * len, len)) {
        return 0;
    }

    // 初始化二分搜索的起始偏移量
    last_ofs = 0;
    ofs = 1;

    for (;;) {
        // 若偏移量超出数组大小或者小于0，则将偏移量设置为数组大小，并结束循环
        if (size <= ofs || ofs < 0) {
            ofs = size; /* arr[ofs] is never accessed */
            break;
        }

        // 如果给定键值小于等于指定位置对应的键值，则退出循环；否则更新 last_ofs 并加倍偏移量
        if (Tag::less(key, arr + tosort[ofs] * len, len)) {
            break;
        }
        else {
            last_ofs = ofs;
            ofs = (ofs << 1) + 1;
        }
    }

    /* 现在 arr[tosort[last_ofs]*len] <= key < arr[tosort[ofs]*len] */
    while (last_ofs + 1 < ofs) {
        // 计算中间位置 m
        m = last_ofs + ((ofs - last_ofs) >> 1);

        // 根据键值比较调整边界 ofs 或 last_ofs
        if (Tag::less(key, arr + tosort[m] * len, len)) {
            ofs = m;
        }
        else {
            last_ofs = m;
        }
    }

    /* 现在 arr[tosort[ofs-1]*len] <= key < arr[tosort[ofs]*len] */
    return ofs;
}

# 合并数组左侧部分的辅助函数
template <typename Tag, typename type>
static void
amerge_left_(type *arr, npy_intp *p1, npy_intp l1, npy_intp *p2, npy_intp l2,
             npy_intp *p3, size_t len)
{
    npy_intp *end = p2 + l2;
    // 复制 p1 的内容到 p3
    memcpy(p3, p1, sizeof(npy_intp) * l1);
    /* first element must be in p2 otherwise skipped in the caller */
    // 将 p2 的第一个元素复制到 p1
    *p1++ = *p2++;

    while (p1 < p2 && p2 < end) {
        // 比较 p2 和 p3 指向的元素，将较小者复制到 p1
        if (Tag::less(arr + (*p2) * len, arr + (*p3) * len, len)) {
            *p1++ = *p2++;
        }
        else {
            *p1++ = *p3++;
        }
    }

    // 将剩余的 p3 元素复制到 p1
    if (p1 != p2) {
        memcpy(p1, p3, sizeof(npy_intp) * (p2 - p1));
    }
}

# 合并数组右侧部分的辅助函数
template <typename Tag, typename type>
static void
amerge_right_(type *arr, npy_intp *p1, npy_intp l1, npy_intp *p2, npy_intp l2,
              npy_intp *p3, size_t len)
{
    npy_intp ofs;
    npy_intp *start = p1 - 1;
    # 将 p2 指针指向的数据复制到 p3 指针指向的位置，复制长度为 sizeof(npy_intp) * l2
    memcpy(p3, p2, sizeof(npy_intp) * l2);
    
    # p1 指针向前移动 l1 - 1 个元素位置
    p1 += l1 - 1;
    
    # p2 指针向前移动 l2 - 1 个元素位置
    p2 += l2 - 1;
    
    # p3 指针向前移动 l2 - 1 个元素位置
    p3 += l2 - 1;
    
    # 将 p1 指针当前位置的数据复制到 p2 指针当前位置，并且将 p1 指针和 p2 指针都向前移动一个位置
    /* first element must be in p1 otherwise skipped in the caller */
    *p2-- = *p1--;

    # 当 p1 指针小于 p2 指针且 start 小于 p1 指针时，执行循环
    while (p1 < p2 && start < p1) {
        # 如果 arr + (*p3) * len 指向的位置的值小于 arr + (*p1) * len 指向的位置的值，执行以下操作
        if (Tag::less(arr + (*p3) * len, arr + (*p1) * len, len)) {
            # 将 p1 指针当前位置的数据复制到 p2 指针当前位置，并且将 p1 指针和 p2 指针都向前移动一个位置
            *p2-- = *p1--;
        }
        else {
            # 将 p3 指针当前位置的数据复制到 p2 指针当前位置，并且将 p2 指针向前移动一个位置
            *p2-- = *p3--;
        }
    }

    # 如果 p1 指针不等于 p2 指针，执行以下操作
    if (p1 != p2) {
        # 计算偏移量，即 p2 指针减去 start 指针的距离
        ofs = p2 - start;
        # 将 p3 指针的数据（从 p3 - ofs + 1 处开始，复制长度为 sizeof(npy_intp) * ofs）复制到 start + 1 处
        memcpy(start + 1, p3 - ofs + 1, sizeof(npy_intp) * ofs);
    }
template <typename Tag, typename type>
static int
amerge_at_(type *arr, npy_intp *tosort, const run *stack, const npy_intp at,
           buffer_intp *buffer, size_t len)
{
    int ret;
    npy_intp s1, l1, s2, l2, k;
    npy_intp *p1, *p2;
    // 获取当前运行栈元素的起始索引和长度
    s1 = stack[at].s;
    l1 = stack[at].l;
    s2 = stack[at + 1].s;
    l2 = stack[at + 1].l;
    /* tosort[s2] belongs to tosort[s1+k] */
    // 使用二分查找找到合适的插入点 k，使得 tosort[s2] 在 tosort[s1+k] 之间
    k = agallop_right_<Tag>(arr, tosort + s1, l1, arr + tosort[s2] * len, len);

    if (l1 == k) {
        /* already sorted */
        // 如果 l1 == k，说明已经有序，无需合并
        return 0;
    }

    p1 = tosort + s1 + k;
    l1 -= k;
    p2 = tosort + s2;
    /* tosort[s2-1] belongs to tosort[s2+l2] */
    // 使用二分查找找到合适的插入点 l2，使得 tosort[s2-1] 在 tosort[s2+l2] 之间
    l2 = agallop_left_<Tag>(arr, tosort + s2, l2, arr + tosort[s2 - 1] * len,
                            len);

    if (l2 < l1) {
        // 如果 l2 < l1，则合并右侧的 run
        ret = resize_buffer_intp(buffer, l2);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }

        amerge_right_<Tag>(arr, p1, l1, p2, l2, buffer->pw, len);
    }
    else {
        // 否则，合并左侧的 run
        ret = resize_buffer_intp(buffer, l1);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }

        amerge_left_<Tag>(arr, p1, l1, p2, l2, buffer->pw, len);
    }

    return 0;
}

template <typename Tag, typename type>
static int
atry_collapse_(type *arr, npy_intp *tosort, run *stack, npy_intp *stack_ptr,
               buffer_intp *buffer, size_t len)
{
    int ret;
    npy_intp A, B, C, top;
    top = *stack_ptr;

    while (1 < top) {
        B = stack[top - 2].l;
        C = stack[top - 1].l;

        if ((2 < top && stack[top - 3].l <= B + C) ||
            (3 < top && stack[top - 4].l <= stack[top - 3].l + B)) {
            A = stack[top - 3].l;

            if (A <= C) {
                // 尝试合并第 top-3 和 top-2 两个 run
                ret = amerge_at_<Tag>(arr, tosort, stack, top - 3, buffer,
                                      len);

                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }

                stack[top - 3].l += B;
                stack[top - 2] = stack[top - 1];
                --top;
            }
            else {
                // 尝试合并第 top-2 和 top-1 两个 run
                ret = amerge_at_<Tag>(arr, tosort, stack, top - 2, buffer,
                                      len);

                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }

                stack[top - 2].l += C;
                --top;
            }
        }
        else if (1 < top && B <= C) {
            // 尝试合并第 top-2 和 top-1 两个 run
            ret = amerge_at_<Tag>(arr, tosort, stack, top - 2, buffer, len);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 2].l += C;
            --top;
        }
        else {
            break;
        }
    }

    *stack_ptr = top;
    return 0;
}

template <typename Tag, typename type>
static int
aforce_collapse_(type *arr, npy_intp *tosort, run *stack, npy_intp *stack_ptr,
                 buffer_intp *buffer, size_t len)
{
    int ret;
    npy_intp top = *stack_ptr;
    // 尝试强制合并剩余的 run，直到栈中只剩一个 run
    while (1 < top) {
        ret = amerge_at_<Tag>(arr, tosort, stack, top - 2, buffer, len);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }

        stack[top - 2].l += stack[top - 1].l;
        --top;
    }

    *stack_ptr = top;
    return 0;
}
    # 当栈中元素个数大于2时，进行以下循环
    while (2 < top) {
        # 如果栈顶第三个元素的长度小于等于栈顶第一个元素的长度
        if (stack[top - 3].l <= stack[top - 1].l) {
            # 调用特定模板函数 amerge_at_<Tag> 进行合并操作，并返回结果
            ret = amerge_at_<Tag>(arr, tosort, stack, top - 3, buffer, len);

            # 如果返回值小于0，直接返回该值
            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            # 更新栈顶第三个元素的长度，加上栈顶第二个元素的长度
            stack[top - 3].l += stack[top - 2].l;
            # 栈顶第二个元素更新为栈顶第一个元素
            stack[top - 2] = stack[top - 1];
            # 栈顶元素个数减一
            --top;
        }
        else {
            # 否则，调用特定模板函数 amerge_at_<Tag> 进行合并操作，并返回结果
            ret = amerge_at_<Tag>(arr, tosort, stack, top - 2, buffer, len);

            # 如果返回值小于0，直接返回该值
            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            # 更新栈顶第二个元素的长度，加上栈顶第一个元素的长度
            stack[top - 2].l += stack[top - 1].l;
            # 栈顶元素个数减一
            --top;
        }
    }

    # 如果栈中还剩余超过1个元素
    if (1 < top) {
        # 调用特定模板函数 amerge_at_<Tag> 进行合并操作，并返回结果
        ret = amerge_at_<Tag>(arr, tosort, stack, top - 2, buffer, len);

        # 如果返回值小于0，直接返回该值
        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }
    }

    # 如果顺利完成合并操作，返回0
    return 0;
    // 对于具有零大小的项目，排序是无意义的
    if (len == 0) {
        return 0;
    }

    // 初始化缓冲区和堆栈指针，计算最小运行长度
    buffer.pw = NULL;
    buffer.size = 0;
    stack_ptr = 0;
    minrun = compute_min_run_short(num);

    // 进行迭代，划分运行并尝试合并
    for (l = 0; l < num;) {
        // 计算当前运行的长度
        n = acount_run_<Tag>((type *)start, tosort, l, num, minrun, len);

        // 将当前运行的起始位置和长度压入堆栈
        stack[stack_ptr].s = l;
        stack[stack_ptr].l = n;
        ++stack_ptr;

        // 尝试将运行堆栈合并为较大的运行
        ret = atry_collapse_<Tag>((type *)start, tosort, stack, &stack_ptr,
                                  &buffer, len);

        // 如果合并失败，进行清理并返回
        if (NPY_UNLIKELY(ret < 0)) {
            goto cleanup;
        }

        l += n;
    }

    // 强制合并堆栈中剩余的运行
    ret = aforce_collapse_<Tag>((type *)start, tosort, stack, &stack_ptr,
                                &buffer, len);

    // 如果强制合并失败，进行清理并返回
    if (NPY_UNLIKELY(ret < 0)) {
        goto cleanup;
    }

    // 清理阶段，释放缓冲区内存
cleanup:
    if (buffer.pw != NULL) {
        free(buffer.pw);
    }
    return ret;
}



/*
 *****************************************************************************
 **                             GENERIC SORT                                **
 *****************************************************************************
 */

typedef struct {
    char *pw;
    npy_intp size;
    size_t len;
} buffer_char;

static inline int
resize_buffer_char(buffer_char *buffer, npy_intp new_size)
{
    // 如果新尺寸小于等于当前尺寸，无需重新分配内存
    if (new_size <= buffer->size) {
        return 0;
    }

    // 重新分配缓冲区内存，并更新缓冲区的大小
    char *new_pw = (char *)realloc(buffer->pw, sizeof(char) * new_size * buffer->len);
    buffer->size = new_size;

    // 如果内存分配失败，返回错误码
    if (NPY_UNLIKELY(new_pw == NULL)) {
        return -NPY_ENOMEM;
    }
    else {
        // 更新缓冲区指针为新分配的内存
        buffer->pw = new_pw;
        return 0;
    }
}

static npy_intp
npy_count_run(char *arr, npy_intp l, npy_intp num, npy_intp minrun, char *vp,
              size_t len, PyArray_CompareFunc *cmp, PyArrayObject *py_arr)
{
    npy_intp sz;
    char *pl, *pi, *pj, *pr;

    // 单个元素的情况，直接返回长度为1
    if (NPY_UNLIKELY(num - l == 1)) {
        return 1;
    }

    // 计算当前运行的起始位置
    pl = arr + l * len;

    // 检查序列是否是（严格）升序
    if (cmp(pl, pl + len, py_arr) <= 0) {
        // 找到升序序列的结束位置
        for (pi = pl + len;
             pi < arr + (num - 1) * len && cmp(pi, pi + len, py_arr) <= 0;
             pi += len) {
        }
    }
    else { // 序列是（严格）降序
        // 找到降序序列的结束位置
        for (pi = pl + len;
             pi < arr + (num - 1) * len && cmp(pi + len, pi, py_arr) < 0;
             pi += len) {
        }

        // 对降序部分进行反转
        for (pj = pl, pr = pi; pj < pr; pj += len, pr -= len) {
            GENERIC_SWAP(pj, pr, len);
        }
    }

    // 返回当前运行的长度
    pi += len;
    sz = (pi - pl) / len;
    // 如果当前子数组长度小于最小运行长度minrun，则根据情况设置sz的值
    if (sz < minrun) {
        // 如果剩余元素数量大于最小运行长度minrun，则将sz设置为minrun
        if (l + minrun < num) {
            sz = minrun;
        }
        // 否则，将sz设置为剩余元素的数量
        else {
            sz = num - l;
        }

        // 计算插入排序的结束位置pr
        pr = pl + sz * len;

        /* 插入排序 */
        for (; pi < pr; pi += len) {
            // 将当前元素复制到vp中
            GENERIC_COPY(vp, pi, len);
            // 将pj设置为当前位置pi
            pj = pi;

            // 在已排序的子数组中寻找正确位置插入当前元素vp
            while (pl < pj && cmp(vp, pj - len, py_arr) < 0) {
                // 将元素向右移动一个位置
                GENERIC_COPY(pj, pj - len, len);
                pj -= len;
            }

            // 将当前元素vp插入到正确位置
            GENERIC_COPY(pj, vp, len);
        }
    }

    // 返回当前子数组的长度sz
    return sz;
}

static npy_intp
npy_gallop_right(const char *arr, const npy_intp size, const char *key,
                 size_t len, PyArray_CompareFunc *cmp, PyArrayObject *py_arr)
{
    npy_intp last_ofs, ofs, m;

    // 如果 key 比数组 arr 的第一个元素还小，则直接返回 0
    if (cmp(key, arr, py_arr) < 0) {
        return 0;
    }

    // 初始设置 gallop 搜索的起始点
    last_ofs = 0;
    ofs = 1;

    for (;;) {
        // 当前的 ofs 已经超过数组大小或者小于 0，则将其设置为 size，不再访问 arr[ofs]
        if (size <= ofs || ofs < 0) {
            ofs = size; /* arr[ofs] is never accessed */
            break;
        }

        // 如果 key 小于 arr[ofs*len]，则退出循环
        if (cmp(key, arr + ofs * len, py_arr) < 0) {
            break;
        }
        else {
            // 否则更新 last_ofs，并将 ofs 更新为类似于二进制指数增长的值
            last_ofs = ofs;
            /* ofs = 1, 3, 7, 15... */
            ofs = (ofs << 1) + 1;
        }
    }

    // 现在 arr[last_ofs*len] <= key < arr[ofs*len]
    while (last_ofs + 1 < ofs) {
        m = last_ofs + ((ofs - last_ofs) >> 1);

        // 继续二分搜索，直到找到合适的位置
        if (cmp(key, arr + m * len, py_arr) < 0) {
            ofs = m;
        }
        else {
            last_ofs = m;
        }
    }

    // 现在 arr[(ofs-1)*len] <= key < arr[ofs*len]
    return ofs;
}

static npy_intp
npy_gallop_left(const char *arr, const npy_intp size, const char *key,
                size_t len, PyArray_CompareFunc *cmp, PyArrayObject *py_arr)
{
    npy_intp last_ofs, ofs, l, m, r;

    // 如果 key 比数组 arr 的最后一个元素还大，则直接返回 size
    if (cmp(arr + (size - 1) * len, key, py_arr) < 0) {
        return size;
    }

    // 初始设置 gallop 搜索的起始点
    last_ofs = 0;
    ofs = 1;

    for (;;) {
        // 当前的 ofs 已经超过数组大小或者小于 0，则将其设置为 size
        if (size <= ofs || ofs < 0) {
            ofs = size;
            break;
        }

        // 如果 key 小于 arr[(size - ofs - 1) * len]，则退出循环
        if (cmp(arr + (size - ofs - 1) * len, key, py_arr) < 0) {
            break;
        }
        else {
            // 否则更新 last_ofs，并将 ofs 更新为类似于二进制指数增长的值
            last_ofs = ofs;
            ofs = (ofs << 1) + 1;
        }
    }

    // 现在 arr[(size-ofs-1)*len] < key <= arr[(size-last_ofs-1)*len]
    l = size - ofs - 1;
    r = size - last_ofs - 1;

    while (l + 1 < r) {
        m = l + ((r - l) >> 1);

        // 继续二分搜索，直到找到合适的位置
        if (cmp(arr + m * len, key, py_arr) < 0) {
            l = m;
        }
        else {
            r = m;
        }
    }

    // 现在 arr[(r-1)*len] < key <= arr[r*len]
    return r;
}

static void
npy_merge_left(char *p1, npy_intp l1, char *p2, npy_intp l2, char *p3,
               size_t len, PyArray_CompareFunc *cmp, PyArrayObject *py_arr)
{
    char *end = p2 + l2 * len;

    // 将 p1 的内容复制到 p3
    memcpy(p3, p1, sizeof(char) * l1 * len);
    /* first element must be in p2 otherwise skipped in the caller */
    // 将 p2 的第一个元素复制到 p1
    GENERIC_COPY(p1, p2, len);
    p1 += len;
    p2 += len;

    while (p1 < p2 && p2 < end) {
        // 如果 p2 小于 p3，则将 p2 的内容复制到 p1，并移动指针
        if (cmp(p2, p3, py_arr) < 0) {
            GENERIC_COPY(p1, p2, len);
            p1 += len;
            p2 += len;
        }
        else {
            // 否则将 p3 的内容复制到 p1，并移动指针
            GENERIC_COPY(p1, p3, len);
            p1 += len;
            p3 += len;
        }
    }

    // 如果 p1 还未走完，将 p3 的剩余内容复制到 p1
    if (p1 != p2) {
        memcpy(p1, p3, sizeof(char) * (p2 - p1));
    }
}

static void
npy_merge_right(char *p1, npy_intp l1, char *p2, npy_intp l2, char *p3,
                size_t len, PyArray_CompareFunc *cmp, PyArrayObject *py_arr)
{
    npy_intp ofs;
    char *start = p1 - len;
    // 将 p2 指向的内存区域复制到 p3 指向的内存区域，复制长度为 sizeof(char) * l2 * len
    memcpy(p3, p2, sizeof(char) * l2 * len);
    // 将 p1 向前移动 (l1 - 1) * len 个字节
    p1 += (l1 - 1) * len;
    // 将 p2 向前移动 (l2 - 1) * len 个字节
    p2 += (l2 - 1) * len;
    // 将 p3 向前移动 (l2 - 1) * len 个字节
    p3 += (l2 - 1) * len;
    // 如果 p1 指向的元素不在 p1 内，则在调用方跳过该元素
    GENERIC_COPY(p2, p1, len);
    // 将 p2 向前移动 len 个字节
    p2 -= len;
    // 将 p1 向前移动 len 个字节
    p1 -= len;

    // 当 p1 小于 p2 并且 start 小于 p1 时执行循环
    while (p1 < p2 && start < p1) {
        // 如果 p3 指向的元素小于 p1 指向的元素（通过 cmp 函数比较），则执行以下操作
        if (cmp(p3, p1, py_arr) < 0) {
            // 将 p1 指向的内存区域复制到 p2 指向的内存区域，复制长度为 len
            GENERIC_COPY(p2, p1, len);
            // 将 p2 向前移动 len 个字节
            p2 -= len;
            // 将 p1 向前移动 len 个字节
            p1 -= len;
        }
        // 否则执行以下操作
        else {
            // 将 p3 指向的内存区域复制到 p2 指向的内存区域，复制长度为 len
            GENERIC_COPY(p2, p3, len);
            // 将 p2 向前移动 len 个字节
            p2 -= len;
            // 将 p3 向前移动 len 个字节
            p3 -= len;
        }
    }

    // 如果 p1 不等于 p2
    if (p1 != p2) {
        // 计算偏移量 ofs，偏移量为 p2 减去 start 的结果
        ofs = p2 - start;
        // 将 p3 向前移动 ofs + len 个字节的内存区域复制到 start + len 指向的内存区域，复制长度为 sizeof(char) * ofs
        memcpy(start + len, p3 - ofs + len, sizeof(char) * ofs);
    }
static int
npy_merge_at(char *arr, const run *stack, const npy_intp at,
             buffer_char *buffer, size_t len, PyArray_CompareFunc *cmp,
             PyArrayObject *py_arr)
{
    int ret;
    npy_intp s1, l1, s2, l2, k;
    char *p1, *p2;

    // 获取当前运行堆栈位置 `at` 的起始位置 `s1` 和长度 `l1`
    s1 = stack[at].s;
    l1 = stack[at].l;
    // 获取下一个运行堆栈位置 `at+1` 的起始位置 `s2` 和长度 `l2`
    s2 = stack[at + 1].s;
    l2 = stack[at + 1].l;

    /* arr[s2] belongs to arr[s1+k] */
    // 将 `arr[s2]` 复制到缓冲区 `buffer->pw`
    GENERIC_COPY(buffer->pw, arr + s2 * len, len);
    // 在 `arr[s1]` 中查找 `buffer->pw` 的插入点 `k`
    k = npy_gallop_right(arr + s1 * len, l1, buffer->pw, len, cmp, py_arr);

    if (l1 == k) {
        /* already sorted */
        // 如果 `l1` 等于 `k`，则说明已经是有序的，直接返回 0
        return 0;
    }

    // 更新指针 `p1` 指向 `arr[s1+k]`
    p1 = arr + (s1 + k) * len;
    // 更新 `l1` 为未处理部分的长度
    l1 -= k;
    // 更新指针 `p2` 指向 `arr[s2]`
    p2 = arr + s2 * len;

    /* arr[s2-1] belongs to arr[s2+l2] */
    // 将 `arr[s2-1]` 复制到缓冲区 `buffer->pw`
    GENERIC_COPY(buffer->pw, arr + (s2 - 1) * len, len);
    // 在 `arr[s2]` 中查找 `buffer->pw` 的插入点 `l2`
    l2 = npy_gallop_left(arr + s2 * len, l2, buffer->pw, len, cmp, py_arr);

    if (l2 < l1) {
        // 如果 `l2` 小于 `l1`，则合并右侧部分
        ret = resize_buffer_char(buffer, l2);
        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }
        // 调用合并函数 `npy_merge_right` 合并右侧部分
        npy_merge_right(p1, l1, p2, l2, buffer->pw, len, cmp, py_arr);
    } else {
        // 否则，合并左侧部分
        ret = resize_buffer_char(buffer, l1);
        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }
        // 调用合并函数 `npy_merge_left` 合并左侧部分
        npy_merge_left(p1, l1, p2, l2, buffer->pw, len, cmp, py_arr);
    }

    // 返回操作成功的标志
    return 0;
}

static int
npy_try_collapse(char *arr, run *stack, npy_intp *stack_ptr,
                 buffer_char *buffer, size_t len, PyArray_CompareFunc *cmp,
                 PyArrayObject *py_arr)
{
    int ret;
    npy_intp A, B, C, top;
    top = *stack_ptr;

    // 在堆栈顶部 `top` 大于 1 的情况下，循环尝试合并
    while (1 < top) {
        B = stack[top - 2].l;
        C = stack[top - 1].l;

        // 检查是否可以合并当前堆栈顶部的三个运行段
        if ((2 < top && stack[top - 3].l <= B + C) ||
            (3 < top && stack[top - 4].l <= stack[top - 3].l + B)) {
            A = stack[top - 3].l;

            if (A <= C) {
                // 如果 A 小于等于 C，调用 `npy_merge_at` 合并堆栈顶部的前两个运行段
                ret = npy_merge_at(arr, stack, top - 3, buffer, len, cmp,
                                   py_arr);
                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }
                // 更新合并后的运行段信息并减少堆栈顶部指针 `top`
                stack[top - 3].l += B;
                stack[top - 2] = stack[top - 1];
                --top;
            } else {
                // 否则，调用 `npy_merge_at` 合并堆栈顶部的后两个运行段
                ret = npy_merge_at(arr, stack, top - 2, buffer, len, cmp,
                                   py_arr);
                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }
                // 更新合并后的运行段信息并减少堆栈顶部指针 `top`
                stack[top - 2].l += C;
                --top;
            }
        } else if (1 < top && B <= C) {
            // 如果无法合并三个运行段，但可以合并前两个运行段，调用 `npy_merge_at` 合并前两个运行段
            ret = npy_merge_at(arr, stack, top - 2, buffer, len, cmp, py_arr);
            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }
            // 更新合并后的运行段信息并减少堆栈顶部指针 `top`
            stack[top - 2].l += C;
            --top;
        } else {
            // 如果无法合并任何运行段，则退出循环
            break;
        }
    }

    // 更新堆栈指针 `stack_ptr` 并返回操作成功的标志
    *stack_ptr = top;
    return 0;
}

static int
npy_force_collapse(char *arr, run *stack, npy_intp *stack_ptr,
                   buffer_char *buffer, size_t len, PyArray_CompareFunc *cmp,
                   PyArrayObject *py_arr)
{
    int ret;  // 定义整型变量 ret，用于存储函数返回值或错误码
    npy_intp top = *stack_ptr;  // 初始化 top 变量为栈顶指针所指向的值

    while (2 < top) {  // 进入循环，条件为栈顶指针指向的值大于 2
        if (stack[top - 3].l <= stack[top - 1].l) {  // 检查栈中倒数第三个元素的长度是否小于等于倒数第一个元素的长度
            ret = npy_merge_at(arr, stack, top - 3, buffer, len, cmp, py_arr);  // 调用函数 npy_merge_at 进行合并操作

            if (NPY_UNLIKELY(ret < 0)) {  // 检查返回值是否小于 0，表示异常情况
                return ret;  // 返回 ret，函数结束
            }

            stack[top - 3].l += stack[top - 2].l;  // 更新栈中倒数第三个元素的长度
            stack[top - 2] = stack[top - 1];  // 将栈中倒数第二个元素更新为栈中倒数第一个元素
            --top;  // 栈顶指针减一
        }
        else {  // 如果栈中倒数第三个元素的长度大于倒数第一个元素的长度
            ret = npy_merge_at(arr, stack, top - 2, buffer, len, cmp, py_arr);  // 调用函数 npy_merge_at 进行合并操作

            if (NPY_UNLIKELY(ret < 0)) {  // 检查返回值是否小于 0，表示异常情况
                return ret;  // 返回 ret，函数结束
            }

            stack[top - 2].l += stack[top - 1].l;  // 更新栈中倒数第二个元素的长度
            --top;  // 栈顶指针减一
        }
    }

    if (1 < top) {  // 如果栈顶指针指向的值大于 1
        ret = npy_merge_at(arr, stack, top - 2, buffer, len, cmp, py_arr);  // 调用函数 npy_merge_at 进行合并操作

        if (NPY_UNLIKELY(ret < 0)) {  // 检查返回值是否小于 0，表示异常情况
            return ret;  // 返回 ret，函数结束
        }
    }

    return 0;  // 返回 0，表示排序完成
}

NPY_NO_EXPORT int
npy_timsort(void *start, npy_intp num, void *varr)
{
    PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(varr);  // 将 varr 转换为 PyArrayObject 类型指针 arr
    size_t len = PyArray_ITEMSIZE(arr);  // 获取数组元素的大小
    PyArray_CompareFunc *cmp = PyDataType_GetArrFuncs(PyArray_DESCR(arr))->compare;  // 获取数组元素比较函数
    int ret;  // 定义整型变量 ret，用于存储函数返回值或错误码
    npy_intp l, n, stack_ptr, minrun;  // 定义整型变量 l, n, stack_ptr, minrun
    run stack[TIMSORT_STACK_SIZE];  // 定义结构体数组 stack，用于存储运行数据
    buffer_char buffer;  // 定义结构体 buffer_char，用于存储缓冲区相关信息

    /* Items that have zero size don't make sense to sort */
    if (len == 0) {  // 如果数组元素的大小为 0
        return 0;  // 直接返回 0，不需要排序
    }

    buffer.pw = NULL;  // 初始化缓冲区指针为 NULL
    buffer.size = 0;  // 初始化缓冲区大小为 0
    buffer.len = len;  // 设置缓冲区元素大小为数组元素的大小
    stack_ptr = 0;  // 初始化栈顶指针为 0
    minrun = compute_min_run_short(num);  // 计算最小运行长度

    /* used for insertion sort and gallop key */
    ret = resize_buffer_char(&buffer, len);  // 调整缓冲区大小

    if (NPY_UNLIKELY(ret < 0)) {  // 检查返回值是否小于 0，表示异常情况
        goto cleanup;  // 跳转到 cleanup 标签，执行清理操作
    }

    for (l = 0; l < num;) {  // 进入循环，从 l=0 开始迭代到 num
        n = npy_count_run((char *)start, l, num, minrun, buffer.pw, len, cmp,
                          arr);  // 计算运行的长度

        /* both s and l are scaled by len */
        stack[stack_ptr].s = l;  // 将当前运行的起始索引存入栈中
        stack[stack_ptr].l = n;  // 将当前运行的长度存入栈中
        ++stack_ptr;  // 栈顶指针加一
        ret = npy_try_collapse((char *)start, stack, &stack_ptr, &buffer, len,
                               cmp, arr);  // 尝试进行折叠操作

        if (NPY_UNLIKELY(ret < 0)) {  // 检查返回值是否小于 0，表示异常情况
            goto cleanup;  // 跳转到 cleanup 标签，执行清理操作
        }

        l += n;  // 更新 l，指向下一个运行的起始位置
    }

    ret = npy_force_collapse((char *)start, stack, &stack_ptr, &buffer, len,
                             cmp, arr);  // 强制折叠剩余的运行数据

    if (NPY_UNLIKELY(ret < 0)) {  // 检查返回值是否小于 0，表示异常情况
        goto cleanup;  // 跳转到 cleanup 标签，执行清理操作
    }

    ret = 0;  // 设置返回值为 0，表示排序完成

cleanup:
    if (buffer.pw != NULL) {  // 如果缓冲区指针不为 NULL
        free(buffer.pw);  // 释放缓冲区内存
    }
    return ret;  // 返回 ret，函数结束
}

/* argsort */

static npy_intp
npy_acount_run(char *arr, npy_intp *tosort, npy_intp l, npy_intp num,
               npy_intp minrun, size_t len, PyArray_CompareFunc *cmp,
               PyArrayObject *py_arr)
{
    npy_intp sz;  // 定义整型变量 sz，用于存储数组大小
    npy_intp vi;  // 定义整型变量 vi，用于存储索引值
    npy_intp *pl, *pi, *pj, *pr;  // 定义整型指针 pl, pi, pj, pr，用于存储数组指针

    if (NPY_UNLIKELY(num - l == 1)) {  // 如果待排序数组的长度为 1
        return 1;  // 返回 1，表示运行长度为 1
    }

    pl = tosort + l;  // 设置指针 pl，指向待排序数组的起始位置

    /* (not strictly) ascending sequence */
    // 检查当前子数组的顺序（升序或者非严格降序）
    if (cmp(arr + (*pl) * len, arr + (*(pl + 1)) * len, py_arr) <= 0) {
        // 如果是升序或非严格降序，则继续查找非递增点
        for (pi = pl + 1;
             pi < tosort + num - 1 &&
             cmp(arr + (*pi) * len, arr + (*(pi + 1)) * len, py_arr) <= 0;
             ++pi) {
        }
    }
    else { /* (strictly) descending sequence */
        // 如果是严格降序，则找到下一个非降序点
        for (pi = pl + 1;
             pi < tosort + num - 1 &&
             cmp(arr + (*(pi + 1)) * len, arr + (*pi) * len, py_arr) < 0;
             ++pi) {
        }

        // 将找到的降序序列反转，使其变为升序
        for (pj = pl, pr = pi; pj < pr; ++pj, --pr) {
            std::swap(*pj, *pr);
        }
    }

    // pi 移动到下一个位置，计算当前子数组的长度
    ++pi;
    sz = pi - pl;

    // 如果当前子数组长度小于最小运行长度，则执行插入排序
    if (sz < minrun) {
        if (l + minrun < num) {
            sz = minrun;
        }
        else {
            sz = num - l;
        }

        pr = pl + sz;

        /* 插入排序 */
        for (; pi < pr; ++pi) {
            vi = *pi;
            pj = pi;

            // 在已排序部分找到合适位置插入当前元素
            while (pl < pj &&
                   cmp(arr + vi * len, arr + (*(pj - 1)) * len, py_arr) < 0) {
                *pj = *(pj - 1);
                --pj;
            }

            *pj = vi;
        }
    }

    // 返回当前子数组的长度
    return sz;
# 在有序数组中执行左侧插值搜索，返回第一个大于给定键值的索引位置
static npy_intp
npy_agallop_left(const char *arr, const npy_intp *tosort, const npy_intp size,
                 const char *key, size_t len, PyArray_CompareFunc *cmp,
                 PyArrayObject *py_arr)
{
    npy_intp last_ofs, ofs, l, m, r;

    # 如果给定键值大于数组中最后一个元素对应的键值，则直接返回数组大小
    if (cmp(arr + tosort[size - 1] * len, key, py_arr) < 0) {
        return size;
    }

    last_ofs = 0;
    ofs = 1;

    for (;;) {
        # 检查索引是否超出数组大小或小于零，若是，则将 ofs 置为数组大小并跳出循环
        if (size <= ofs || ofs < 0) {
            ofs = size;
            break;
        }

        # 比较给定键值与当前索引位置对应的键值，若小于则跳出循环，否则更新 last_ofs 并按指数级增加 ofs
        if (cmp(arr + tosort[size - ofs - 1] * len, key, py_arr) < 0) {
            break;
        }
        else {
            last_ofs = ofs;
            ofs = (ofs << 1) + 1;
        }
    }

    # 现在 arr[tosort[size-ofs-1]*len] < key <= arr[tosort[size-last_ofs-1]*len]
    l = size - ofs - 1;
    r = size - last_ofs - 1;

    while (l + 1 < r) {
        m = l + ((r - l) >> 1);

        # 二分查找，比较给定键值与中间索引位置对应的键值，更新区间边界 l 或 r
        if (cmp(arr + tosort[m] * len, key, py_arr) < 0) {
            l = m;
        }
        else {
            r = m;
        }
    }

    # 现在 arr[tosort[r-1]*len] < key <= arr[tosort[r]*len]
    return r;
}

# 在有序数组中执行右侧插值搜索，返回第一个大于等于给定键值的索引位置
static npy_intp
npy_agallop_right(const char *arr, const npy_intp *tosort, const npy_intp size,
                  const char *key, size_t len, PyArray_CompareFunc *cmp,
                  PyArrayObject *py_arr)
{
    npy_intp last_ofs, ofs, m;

    # 如果给定键值小于数组中第一个元素对应的键值，则直接返回 0
    if (cmp(key, arr + tosort[0] * len, py_arr) < 0) {
        return 0;
    }

    last_ofs = 0;
    ofs = 1;

    for (;;) {
        # 检查索引是否超出数组大小或小于零，若是，则将 ofs 置为数组大小并跳出循环
        if (size <= ofs || ofs < 0) {
            ofs = size;  # arr[ofs] is never accessed
            break;
        }

        # 比较给定键值与当前索引位置对应的键值，若小于则跳出循环，否则更新 last_ofs 并按指数级增加 ofs
        if (cmp(key, arr + tosort[ofs] * len, py_arr) < 0) {
            break;
        }
        else {
            last_ofs = ofs;
            ofs = (ofs << 1) + 1;  # ofs = 1, 3, 7, 15...
        }
    }

    # 现在 arr[tosort[last_ofs]*len] <= key < arr[tosort[ofs]*len]
    while (last_ofs + 1 < ofs) {
        m = last_ofs + ((ofs - last_ofs) >> 1);

        # 二分查找，比较给定键值与中间索引位置对应的键值，更新区间边界 ofs 或 last_ofs
        if (cmp(key, arr + tosort[m] * len, py_arr) < 0) {
            ofs = m;
        }
        else {
            last_ofs = m;
        }
    }

    # 现在 arr[tosort[ofs-1]*len] <= key < arr[tosort[ofs]*len]
    return ofs;
}

# 将两个有序数组合并到第三个数组中，保持有序性
static void
npy_amerge_left(char *arr, npy_intp *p1, npy_intp l1, npy_intp *p2,
                npy_intp l2, npy_intp *p3, size_t len,
                PyArray_CompareFunc *cmp, PyArrayObject *py_arr)
{
    npy_intp *end = p2 + l2;
    # 复制第一个有序数组的内容到第三个数组
    memcpy(p3, p1, sizeof(npy_intp) * l1);
    # 将第一个有序数组与第二个有序数组进行合并，结果存放在第一个数组中
    # 注意：调用者中的第一个元素必须来自第二个数组，否则将被忽略
    *p1++ = *p2++;

    while (p1 < p2 && p2 < end) {
        # 比较第二个数组与第三个数组当前位置对应的键值，将较小者复制到第一个数组中
        if (cmp(arr + (*p2) * len, arr + (*p3) * len, py_arr) < 0) {
            *p1++ = *p2++;
        }
        else {
            *p1++ = *p3++;
        }
    }

    # 若第一个数组没有遍历完，将剩余部分复制到第三个数组中
    if (p1 != p2) {
        memcpy(p1, p3, sizeof(npy_intp) * (p2 - p1));
    }
}

# 接下来的函数未提供代码，因此无需进一步注释
# 合并右侧数组片段到左侧数组片段，利用的比较函数为cmp，数组元素长度为len，结果存储在p3中
npy_amerge_right(char *arr, npy_intp *p1, npy_intp l1, npy_intp *p2,
                 npy_intp l2, npy_intp *p3, size_t len,
                 PyArray_CompareFunc *cmp, PyArrayObject *py_arr)
{
    npy_intp ofs;
    npy_intp *start = p1 - 1;  # 将p1的前一个位置赋值给start，用于后续计算ofs
    memcpy(p3, p2, sizeof(npy_intp) * l2);  # 将p2指向的长度为l2的整数数组复制到p3指向的位置

    p1 += l1 - 1;  # 将p1移动到其末尾元素处
    p2 += l2 - 1;  # 将p2移动到其末尾元素处
    p3 += l2 - 1;  # 将p3移动到其末尾元素处
    /* first element must be in p1 otherwise skipped in the caller */
    *p2-- = *p1--;  # 将p1的末尾元素赋值给p2，并将p1和p2向前移动一位

    while (p1 < p2 && start < p1) {
        if (cmp(arr + (*p3) * len, arr + (*p1) * len, py_arr) < 0) {
            *p2-- = *p1--;  # 如果cmp比较结果小于0，则将p1的元素复制给p2，并将p1和p2向前移动一位
        }
        else {
            *p2-- = *p3--;  # 否则将p3的元素复制给p2，并将p3和p2向前移动一位
        }
    }

    if (p1 != p2) {
        ofs = p2 - start;  # 计算p2相对于start的偏移量
        memcpy(start + 1, p3 - ofs + 1, sizeof(npy_intp) * ofs);  # 将p3中的ofs个元素复制到start的后面
    }
}

static int
npy_amerge_at(char *arr, npy_intp *tosort, const run *stack, const npy_intp at,
              buffer_intp *buffer, size_t len, PyArray_CompareFunc *cmp,
              PyArrayObject *py_arr)
{
    int ret;
    npy_intp s1, l1, s2, l2, k;
    npy_intp *p1, *p2;
    s1 = stack[at].s;  # 设置s1为stack[at]的起始位置
    l1 = stack[at].l;  # 设置l1为stack[at]的长度
    s2 = stack[at + 1].s;  # 设置s2为stack[at+1]的起始位置
    l2 = stack[at + 1].l;  # 设置l2为stack[at+1]的长度
    /* tosort[s2] belongs to tosort[s1+k] */
    k = npy_agallop_right(arr, tosort + s1, l1, arr + tosort[s2] * len, len,
                          cmp, py_arr);  # 查找满足agallop_right条件的k值

    if (l1 == k) {
        /* already sorted */
        return 0;  # 如果l1等于k，则已经排序完成，直接返回0
    }

    p1 = tosort + s1 + k;  # 设置p1为tosort的起始位置加上k
    l1 -= k;  # 减去k得到新的l1
    p2 = tosort + s2;  # 设置p2为tosort的起始位置加上s2
    /* tosort[s2-1] belongs to tosort[s2+l2] */
    l2 = npy_agallop_left(arr, tosort + s2, l2, arr + tosort[s2 - 1] * len,
                          len, cmp, py_arr);  # 查找满足agallop_left条件的l2值

    if (l2 < l1) {
        ret = resize_buffer_intp(buffer, l2);  # 调整buffer的大小为l2

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;  # 如果调整失败，返回错误码
        }

        npy_amerge_right(arr, p1, l1, p2, l2, buffer->pw, len, cmp, py_arr);  # 调用npy_amerge_right函数合并右侧数组片段到左侧数组片段
    }
    else {
        ret = resize_buffer_intp(buffer, l1);  # 调整buffer的大小为l1

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;  # 如果调整失败，返回错误码
        }

        npy_amerge_left(arr, p1, l1, p2, l2, buffer->pw, len, cmp, py_arr);  # 调用npy_amerge_left函数合并左侧数组片段到右侧数组片段
    }

    return 0;  # 返回0表示成功
}

static int
npy_atry_collapse(char *arr, npy_intp *tosort, run *stack, npy_intp *stack_ptr,
                  buffer_intp *buffer, size_t len, PyArray_CompareFunc *cmp,
                  PyArrayObject *py_arr)
{
    int ret;
    npy_intp A, B, C, top;
    top = *stack_ptr;
    # 当栈中元素个数大于1时，执行以下循环
    while (1 < top) {
        # 取出栈顶前两个元素的长度赋值给 B 和 C
        B = stack[top - 2].l;
        C = stack[top - 1].l;

        # 如果满足以下条件之一：
        # 1. 栈顶元素个数大于2且第三个元素长度小于等于 B + C
        # 2. 栈顶元素个数大于3且第四个元素长度小于等于第三个元素长度加上 B
        if ((2 < top && stack[top - 3].l <= B + C) ||
            (3 < top && stack[top - 4].l <= stack[top - 3].l + B)) {
            # 将第三个元素长度赋值给 A
            A = stack[top - 3].l;

            # 如果 A 小于等于 C，执行以下操作
            if (A <= C) {
                # 调用 npy_amerge_at 函数进行合并操作
                ret = npy_amerge_at(arr, tosort, stack, top - 3, buffer, len,
                                    cmp, py_arr);

                # 如果返回值小于0，直接返回 ret
                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }

                # 更新栈顶第三个元素的长度为 B
                stack[top - 3].l += B;
                # 将栈顶第二个元素更新为栈顶第一个元素
                stack[top - 2] = stack[top - 1];
                # 栈顶元素个数减一
                --top;
            }
            else {
                # 否则执行以下操作
                # 调用 npy_amerge_at 函数进行合并操作
                ret = npy_amerge_at(arr, tosort, stack, top - 2, buffer, len,
                                    cmp, py_arr);

                # 如果返回值小于0，直接返回 ret
                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }

                # 更新栈顶第二个元素的长度为 C
                stack[top - 2].l += C;
                # 栈顶元素个数减一
                --top;
            }
        }
        # 如果满足条件 B <= C
        elif (1 < top && B <= C) {
            # 调用 npy_amerge_at 函数进行合并操作
            ret = npy_amerge_at(arr, tosort, stack, top - 2, buffer, len, cmp,
                                py_arr);

            # 如果返回值小于0，直接返回 ret
            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            # 更新栈顶第二个元素的长度为 C
            stack[top - 2].l += C;
            # 栈顶元素个数减一
            --top;
        }
        else {
            # 如果不满足上述条件，跳出循环
            break;
        }
    }

    # 将栈顶元素个数 top 更新到 stack_ptr 指向的位置
    *stack_ptr = top;
    # 返回 0 表示成功执行
    return 0;
static int
npy_aforce_collapse(char *arr, npy_intp *tosort, run *stack,
                    npy_intp *stack_ptr, buffer_intp *buffer, size_t len,
                    PyArray_CompareFunc *cmp, PyArrayObject *py_arr)
{
    int ret;
    npy_intp top = *stack_ptr;

    // 当栈中元素数量大于2时，执行循环
    while (2 < top) {
        // 如果栈顶前两个 run 的长度满足非递减序列条件
        if (stack[top - 3].l <= stack[top - 1].l) {
            // 调用函数尝试合并 run
            ret = npy_amerge_at(arr, tosort, stack, top - 3, buffer, len, cmp,
                                py_arr);

            // 如果合并失败，返回错误码
            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            // 更新栈顶第一个 run 的长度为前两个 run 的长度之和
            stack[top - 3].l += stack[top - 2].l;
            // 移除栈顶第二个 run
            stack[top - 2] = stack[top - 1];
            // 栈顶指针减一
            --top;
        }
        else {
            // 如果栈顶前两个 run 的长度不满足非递减序列条件
            // 调用函数尝试合并 run
            ret = npy_amerge_at(arr, tosort, stack, top - 2, buffer, len, cmp,
                                py_arr);

            // 如果合并失败，返回错误码
            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            // 更新栈顶第一个 run 的长度为栈顶第一个和第二个 run 的长度之和
            stack[top - 2].l += stack[top - 1].l;
            // 栈顶指针减一
            --top;
        }
    }

    // 如果栈中仍有多余一个 run
    if (1 < top) {
        // 调用函数尝试合并 run
        ret = npy_amerge_at(arr, tosort, stack, top - 2, buffer, len, cmp,
                            py_arr);

        // 如果合并失败，返回错误码
        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }
    }

    // 成功完成排序，返回0
    return 0;
}

NPY_NO_EXPORT int
npy_atimsort(void *start, npy_intp *tosort, npy_intp num, void *varr)
{
    // 将 void* 类型的 varr 转换为 PyArrayObject*
    PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(varr);
    // 获取数组元素的大小
    size_t len = PyArray_ITEMSIZE(arr);
    // 获取比较函数指针
    PyArray_CompareFunc *cmp = PyDataType_GetArrFuncs(PyArray_DESCR(arr))->compare;
    int ret;
    npy_intp l, n, stack_ptr, minrun;
    run stack[TIMSORT_STACK_SIZE];
    buffer_intp buffer;

    // 如果数组元素大小为0，直接返回
    if (len == 0) {
        return 0;
    }

    // 初始化缓冲区
    buffer.pw = NULL;
    buffer.size = 0;
    // 初始化栈指针和最小运行长度
    stack_ptr = 0;
    minrun = compute_min_run_short(num);

    // 进行排序的主循环
    for (l = 0; l < num;) {
        // 计算当前运行的长度
        n = npy_acount_run((char *)start, tosort, l, num, minrun, len, cmp,
                           arr);
        // 将当前运行的起始索引和长度存入栈中
        stack[stack_ptr].s = l;
        stack[stack_ptr].l = n;
        // 栈指针加一
        ++stack_ptr;
        // 尝试折叠栈中的运行
        ret = npy_atry_collapse((char *)start, tosort, stack, &stack_ptr,
                                &buffer, len, cmp, arr);

        // 如果折叠失败，跳转到清理代码段
        if (NPY_UNLIKELY(ret < 0)) {
            goto cleanup;
        }

        // 更新当前处理的元素索引
        l += n;
    }

    // 强制折叠栈中的运行
    ret = npy_aforce_collapse((char *)start, tosort, stack, &stack_ptr,
                              &buffer, len, cmp, arr);

    // 如果强制折叠失败，跳转到清理代码段
    if (NPY_UNLIKELY(ret < 0)) {
        goto cleanup;
    }

    // 完成排序，返回0
    ret = 0;

cleanup:
    // 清理缓冲区内存
    if (buffer.pw != NULL) {
        free(buffer.pw);
    }
    return ret;
}

/***************************************
 * C > C++ dispatch
 ***************************************/

// 对布尔类型进行 timsort 排序
NPY_NO_EXPORT int
timsort_bool(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return timsort_<npy::bool_tag>(start, num);
}

// 对字节类型进行 timsort 排序
NPY_NO_EXPORT int
timsort_byte(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return timsort_<npy::byte_tag>(start, num);
}
# 使用模板函数 timsort_ 进行无符号字节排序
timsort_ubyte(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用 timsort_ 函数，使用无符号字节标签对指定范围进行排序
    return timsort_<npy::ubyte_tag>(start, num);
}

# 使用模板函数 timsort_ 进行短整型排序
timsort_short(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用 timsort_ 函数，使用短整型标签对指定范围进行排序
    return timsort_<npy::short_tag>(start, num);
}

# 使用模板函数 timsort_ 进行无符号短整型排序
timsort_ushort(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用 timsort_ 函数，使用无符号短整型标签对指定范围进行排序
    return timsort_<npy::ushort_tag>(start, num);
}

# 使用模板函数 timsort_ 进行整型排序
timsort_int(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用 timsort_ 函数，使用整型标签对指定范围进行排序
    return timsort_<npy::int_tag>(start, num);
}

# 使用模板函数 timsort_ 进行无符号整型排序
timsort_uint(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用 timsort_ 函数，使用无符号整型标签对指定范围进行排序
    return timsort_<npy::uint_tag>(start, num);
}

# 使用模板函数 timsort_ 进行长整型排序
timsort_long(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用 timsort_ 函数，使用长整型标签对指定范围进行排序
    return timsort_<npy::long_tag>(start, num);
}

# 使用模板函数 timsort_ 进行无符号长整型排序
timsort_ulong(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用 timsort_ 函数，使用无符号长整型标签对指定范围进行排序
    return timsort_<npy::ulong_tag>(start, num);
}

# 使用模板函数 timsort_ 进行长长整型排序
timsort_longlong(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用 timsort_ 函数，使用长长整型标签对指定范围进行排序
    return timsort_<npy::longlong_tag>(start, num);
}

# 使用模板函数 timsort_ 进行无符号长长整型排序
timsort_ulonglong(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用 timsort_ 函数，使用无符号长长整型标签对指定范围进行排序
    return timsort_<npy::ulonglong_tag>(start, num);
}

# 使用模板函数 timsort_ 进行半精度浮点数排序
timsort_half(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用 timsort_ 函数，使用半精度浮点数标签对指定范围进行排序
    return timsort_<npy::half_tag>(start, num);
}

# 使用模板函数 timsort_ 进行单精度浮点数排序
timsort_float(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用 timsort_ 函数，使用单精度浮点数标签对指定范围进行排序
    return timsort_<npy::float_tag>(start, num);
}

# 使用模板函数 timsort_ 进行双精度浮点数排序
timsort_double(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用 timsort_ 函数，使用双精度浮点数标签对指定范围进行排序
    return timsort_<npy::double_tag>(start, num);
}

# 使用模板函数 timsort_ 进行长双精度浮点数排序
timsort_longdouble(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用 timsort_ 函数，使用长双精度浮点数标签对指定范围进行排序
    return timsort_<npy::longdouble_tag>(start, num);
}

# 使用模板函数 timsort_ 进行复数浮点数排序
timsort_cfloat(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用 timsort_ 函数，使用复数浮点数标签对指定范围进行排序
    return timsort_<npy::cfloat_tag>(start, num);
}

# 使用模板函数 timsort_ 进行双精度复数浮点数排序
timsort_cdouble(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用 timsort_ 函数，使用双精度复数浮点数标签对指定范围进行排序
    return timsort_<npy::cdouble_tag>(start, num);
}

# 使用模板函数 timsort_ 进行长双精度复数浮点数排序
timsort_clongdouble(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用 timsort_ 函数，使用长双精度复数浮点数标签对指定范围进行排序
    return timsort_<npy::clongdouble_tag>(start, num);
}

# 使用模板函数 timsort_ 进行日期时间排序
timsort_datetime(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用 timsort_ 函数，使用日期时间标签对指定范围进行排序
    return timsort_<npy::datetime_tag>(start, num);
}

# 使用模板函数 timsort_ 进行时间间隔排序
timsort_timedelta(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用 timsort_ 函数，使用时间间隔标签对指定范围进行排序
    return timsort_<npy::timedelta_tag>(start, num);
}

# 使用字符串模板函数 string_timsort_ 进行字符串排序
timsort_string(void *start, npy_intp num, void *varr)
{
    // 调用 string_timsort_ 函数，使用字符串标签对指定范围进行排序
    return string_timsort_<npy::string_tag>(start, num, varr);
}

# 使用字符串模板函数 string_timsort_ 进行Unicode字符串排序
timsort_unicode(void *start, npy_intp num, void *varr)
{
    // 调用 string_timsort_ 函数，使用Unicode字符串标签对指定范围进行排序
    return string_timsort_<npy::unicode_tag>(start, num, varr);
}

# 使用模板函数 atimsort_ 进行布尔值数组排序
atimsort_bool(void *v, npy_intp *tosort, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用 atimsort_ 函数，使用布尔值标签对指定数组进行排序
    return atimsort_<npy::bool_tag>(v, tosort, num);
}
// 使用模板函数 atimsort_ 来对字节数组进行排序，使用字节标签 npy::byte_tag
atimsort_byte(void *v, npy_intp *tosort, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用模板函数 atimsort_，传入字节标签 npy::byte_tag，排序给定的数组
    return atimsort_<npy::byte_tag>(v, tosort, num);
}

// 使用模板函数 atimsort_ 来对无符号字节数组进行排序，使用无符号字节标签 npy::ubyte_tag
NPY_NO_EXPORT int
atimsort_ubyte(void *v, npy_intp *tosort, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用模板函数 atimsort_，传入无符号字节标签 npy::ubyte_tag，排序给定的数组
    return atimsort_<npy::ubyte_tag>(v, tosort, num);
}

// 使用模板函数 atimsort_ 来对短整型数组进行排序，使用短整型标签 npy::short_tag
NPY_NO_EXPORT int
atimsort_short(void *v, npy_intp *tosort, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用模板函数 atimsort_，传入短整型标签 npy::short_tag，排序给定的数组
    return atimsort_<npy::short_tag>(v, tosort, num);
}

// 使用模板函数 atimsort_ 来对无符号短整型数组进行排序，使用无符号短整型标签 npy::ushort_tag
NPY_NO_EXPORT int
atimsort_ushort(void *v, npy_intp *tosort, npy_intp num,
                void *NPY_UNUSED(varr))
{
    // 调用模板函数 atimsort_，传入无符号短整型标签 npy::ushort_tag，排序给定的数组
    return atimsort_<npy::ushort_tag>(v, tosort, num);
}

// 使用模板函数 atimsort_ 来对整型数组进行排序，使用整型标签 npy::int_tag
NPY_NO_EXPORT int
atimsort_int(void *v, npy_intp *tosort, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用模板函数 atimsort_，传入整型标签 npy::int_tag，排序给定的数组
    return atimsort_<npy::int_tag>(v, tosort, num);
}

// 使用模板函数 atimsort_ 来对无符号整型数组进行排序，使用无符号整型标签 npy::uint_tag
NPY_NO_EXPORT int
atimsort_uint(void *v, npy_intp *tosort, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用模板函数 atimsort_，传入无符号整型标签 npy::uint_tag，排序给定的数组
    return atimsort_<npy::uint_tag>(v, tosort, num);
}

// 使用模板函数 atimsort_ 来对长整型数组进行排序，使用长整型标签 npy::long_tag
NPY_NO_EXPORT int
atimsort_long(void *v, npy_intp *tosort, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用模板函数 atimsort_，传入长整型标签 npy::long_tag，排序给定的数组
    return atimsort_<npy::long_tag>(v, tosort, num);
}

// 使用模板函数 atimsort_ 来对无符号长整型数组进行排序，使用无符号长整型标签 npy::ulong_tag
NPY_NO_EXPORT int
atimsort_ulong(void *v, npy_intp *tosort, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用模板函数 atimsort_，传入无符号长整型标签 npy::ulong_tag，排序给定的数组
    return atimsort_<npy::ulong_tag>(v, tosort, num);
}

// 使用模板函数 atimsort_ 来对长长整型数组进行排序，使用长长整型标签 npy::longlong_tag
NPY_NO_EXPORT int
atimsort_longlong(void *v, npy_intp *tosort, npy_intp num,
                  void *NPY_UNUSED(varr))
{
    // 调用模板函数 atimsort_，传入长长整型标签 npy::longlong_tag，排序给定的数组
    return atimsort_<npy::longlong_tag>(v, tosort, num);
}

// 使用模板函数 atimsort_ 来对无符号长长整型数组进行排序，使用无符号长长整型标签 npy::ulonglong_tag
NPY_NO_EXPORT int
atimsort_ulonglong(void *v, npy_intp *tosort, npy_intp num,
                   void *NPY_UNUSED(varr))
{
    // 调用模板函数 atimsort_，传入无符号长长整型标签 npy::ulonglong_tag，排序给定的数组
    return atimsort_<npy::ulonglong_tag>(v, tosort, num);
}

// 使用模板函数 atimsort_ 来对半精度浮点数数组进行排序，使用半精度浮点数标签 npy::half_tag
NPY_NO_EXPORT int
atimsort_half(void *v, npy_intp *tosort, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用模板函数 atimsort_，传入半精度浮点数标签 npy::half_tag，排序给定的数组
    return atimsort_<npy::half_tag>(v, tosort, num);
}

// 使用模板函数 atimsort_ 来对单精度浮点数数组进行排序，使用单精度浮点数标签 npy::float_tag
NPY_NO_EXPORT int
atimsort_float(void *v, npy_intp *tosort, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用模板函数 atimsort_，传入单精度浮点数标签 npy::float_tag，排序给定的数组
    return atimsort_<npy::float_tag>(v, tosort, num);
}

// 使用模板函数 atimsort_ 来对双精度浮点数数组进行排序，使用双精度浮点数标签 npy::double_tag
NPY_NO_EXPORT int
atimsort_double(void *v, npy_intp *tosort, npy_intp num,
                void *NPY_UNUSED(varr))
{
    // 调用模板函数 atimsort_，传入双精度浮点数标签 npy::double_tag，排序给定的数组
    return atimsort_<npy::double_tag>(v, tosort, num);
}

// 使用模板函数 atimsort_ 来对长双精度浮点数数组进行排序，使用长双精度浮点数标签 npy::longdouble_tag
NPY_NO_EXPORT int
atimsort_longdouble(void *v, npy_intp *tosort, npy_intp num,
                    void *NPY_UNUSED(varr))
{
    // 调用模板函数 atimsort_，传入长双精度浮点数标签 npy::longdouble_tag，排序给定的数组
    return atimsort_<npy::longdouble_tag>(v, tosort, num);
}

// 使用模板函数 atimsort_ 来对复数浮点数数组进行排序，使用复数浮点数标签 npy::cfloat_tag
NPY_NO_EXPORT int
atimsort_cfloat(void *v, npy_intp *tosort, npy_intp num,
                void *NPY_UNUSED(varr))
{
    // 调用模板函数 atimsort_，传入复数浮点数标签 npy::cfloat_tag，排序给定的数组
    return atimsort_<npy::cfloat_tag>(v, tosort, num);
}

// 使用模板函数 atimsort_ 来对双精度复数浮点数数组进行排序，使用双精度复数浮点数标签 npy::cdouble_tag
NPY_NO_EXPORT int
atimsort_cdouble(void *v, npy_intp *tosort, npy_intp num,
# 使用 NPY_NO_EXPORT 宏定义一个返回整数的函数，接受一个指向 void 类型的指针 v，
# 一个指向 npy_intp 类型的数组 tosort，以及一个 npy_intp 类型的整数 num，
# varr 参数未使用
NPY_NO_EXPORT int
atimsort_timedelta(void *v, npy_intp *tosort, npy_intp num,
                   void *NPY_UNUSED(varr))
{
    # 调用 atimsort_ 函数，使用 npy::timedelta_tag 标签，对给定的 v, tosort, num 进行排序处理，并返回结果
    return atimsort_<npy::timedelta_tag>(v, tosort, num);
}

# 使用 NPY_NO_EXPORT 宏定义一个返回整数的函数，接受一个指向 void 类型的指针 v，
# 一个指向 npy_intp 类型的数组 tosort，以及一个 npy_intp 类型的整数 num，
# varr 参数将被使用
NPY_NO_EXPORT int
atimsort_string(void *v, npy_intp *tosort, npy_intp num, void *varr)
{
    # 调用 string_atimsort_ 函数，使用 npy::string_tag 标签，对给定的 v, tosort, num, varr 进行排序处理，并返回结果
    return string_atimsort_<npy::string_tag>(v, tosort, num, varr);
}

# 使用 NPY_NO_EXPORT 宏定义一个返回整数的函数，接受一个指向 void 类型的指针 v，
# 一个指向 npy_intp 类型的数组 tosort，以及一个 npy_intp 类型的整数 num，
# varr 参数将被使用
NPY_NO_EXPORT int
atimsort_unicode(void *v, npy_intp *tosort, npy_intp num, void *varr)
{
    # 调用 string_atimsort_ 函数，使用 npy::unicode_tag 标签，对给定的 v, tosort, num, varr 进行排序处理，并返回结果
    return string_atimsort_<npy::unicode_tag>(v, tosort, num, varr);
}
```