# `.\numpy\numpy\_core\src\npysort\selection.cpp`

```py
/*
 *
 * The code is loosely based on the quickselect from
 * Nicolas Devillard - 1998 public domain
 * http://ndevilla.free.fr/median/median/
 *
 * Quick select with median of 3 pivot is usually the fastest,
 * but the worst case scenario can be quadratic complexity,
 * e.g. np.roll(np.arange(x), x / 2)
 * To avoid this if it recurses too much it falls back to the
 * worst case linear median of median of group 5 pivot strategy.
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "numpy/npy_math.h"

#include "npy_partition.h"
#include "npy_sort.h"
#include "npysort_common.h"
#include "numpy_tag.h"

#include <array>
#include <cstdlib>
#include <utility>
#include "x86_simd_qsort.hpp"

#define NOT_USED NPY_UNUSED(unused)
#define DISABLE_HIGHWAY_OPTIMIZATION (defined(__arm__) || defined(__aarch64__))

template<typename T>
inline bool quickselect_dispatch(T* v, npy_intp num, npy_intp kth)
{
#ifndef __CYGWIN__
    /*
     * Only defined for int16_t, uint16_t, float16, int32_t, uint32_t, float32,
     * int64_t, uint64_t, double
     */
    if constexpr (
        (std::is_integral_v<T> || std::is_floating_point_v<T> || std::is_same_v<T, np::Half>) &&
        (sizeof(T) == sizeof(uint16_t) || sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t))) {
        using TF = typename np::meta::FixedWidth<T>::Type;
        void (*dispfunc)(TF*, npy_intp, npy_intp) = nullptr;
        if constexpr (sizeof(T) == sizeof(uint16_t)) {
            #ifndef NPY_DISABLE_OPTIMIZATION
                #include "x86_simd_qsort_16bit.dispatch.h"
            #endif
            NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::qsort_simd::template QSelect, <TF>);
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t)) {
            #ifndef NPY_DISABLE_OPTIMIZATION
                #include "x86_simd_qsort.dispatch.h"
            #endif
            NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::qsort_simd::template QSelect, <TF>);
        }
        if (dispfunc) {
            (*dispfunc)(reinterpret_cast<TF*>(v), num, kth);
            return true;
        }
    }
#endif
    (void)v; (void)num; (void)kth; // to avoid unused arg warn
    return false;
}

template<typename T>
inline bool argquickselect_dispatch(T* v, npy_intp* arg, npy_intp num, npy_intp kth)
{
#ifndef __CYGWIN__
    /*
     * Only defined for int32_t, uint32_t, float32, int64_t, uint64_t, double
     */
    // 如果 T 是整数或浮点数类型，并且其大小为 uint32_t 或 uint64_t 的大小
    if constexpr (
        (std::is_integral_v<T> || std::is_floating_point_v<T>) &&
        (sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t))) {
        
        // 使用 FixedWidth<T>::Type 定义 TF 类型
        using TF = typename np::meta::FixedWidth<T>::Type;

        // 如果未禁用优化，则包含 x86 SIMD 排序的调度头文件
        #ifndef NPY_DISABLE_OPTIMIZATION
            #include "x86_simd_argsort.dispatch.h"
        #endif
        
        // 定义函数指针 dispfunc，用于接收 SIMD 排序算法中的选择函数
        void (*dispfunc)(TF*, npy_intp*, npy_intp, npy_intp) = nullptr;
        
        // 使用 NPY_CPU_DISPATCH_CALL_XB 宏调用 SIMD 排序的 ArgQSelect 函数
        NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::qsort_simd::template ArgQSelect, <TF>);
        
        // 如果 dispfunc 非空，则调用相应的 SIMD 排序函数处理输入数据
        if (dispfunc) {
            (*dispfunc)(reinterpret_cast<TF*>(v), arg, num, kth);
            // 处理完成，返回 true 表示成功
            return true;
        }
    }
#endif
    (void)v; (void)arg; (void)num; (void)kth; // to avoid unused arg warn
    // 忽略未使用的参数以避免警告
    return false;
}

template <typename Tag, bool arg, typename type>
NPY_NO_EXPORT int
introselect_(type *v, npy_intp *tosort, npy_intp num, npy_intp kth, npy_intp *pivots, npy_intp *npiv);
/*
 *****************************************************************************
 **                            NUMERIC SORTS                                **
 *****************************************************************************
 */
// 数值排序

static inline void
store_pivot(npy_intp pivot, npy_intp kth, npy_intp *pivots, npy_intp *npiv)
{
    if (pivots == NULL) {
        return;
    }
    // 如果 pivots 为空，则直接返回

    /*
     * If pivot is the requested kth store it, overwriting other pivots if
     * required. This must be done so iterative partition can work without
     * manually shifting lower data offset by kth each time
     */
    // 如果 pivot 等于请求的 kth 值，则存储它，如果需要可以覆盖其他的 pivot
    // 这样做是为了使迭代分区可以在每次操作时不需要手动移动低数据偏移量

    if (pivot == kth && *npiv == NPY_MAX_PIVOT_STACK) {
        pivots[*npiv - 1] = pivot;
    }
    // 如果 pivot 等于 kth，并且当前已存储的 pivots 达到了最大允许值 NPY_MAX_PIVOT_STACK，则将其存储在最后一个位置

    /*
     * we only need pivots larger than current kth, larger pivots are not
     * useful as partitions on smaller kth would reorder the stored pivots
     */
    // 我们只需要大于当前 kth 的 pivots，因为更大的 pivots 在较小的 kth 上进行分区时会重新排列已存储的 pivots

    else if (pivot >= kth && *npiv < NPY_MAX_PIVOT_STACK) {
        pivots[*npiv] = pivot;
        (*npiv) += 1;
    }
    // 否则，如果 pivot 大于等于 kth 并且当前存储的 pivots 小于最大允许值 NPY_MAX_PIVOT_STACK，则存储 pivot，并增加 npiv 的计数

}

template <typename type, bool arg>
struct Sortee {
    type *v;
    Sortee(type *v, npy_intp *) : v(v) {}
    type &operator()(npy_intp i) const { return v[i]; }
};

template <bool arg>
struct Idx {
    Idx(npy_intp *) {}
    npy_intp operator()(npy_intp i) const { return i; }
};

template <typename type>
struct Sortee<type, true> {
    npy_intp *tosort;
    Sortee(type *, npy_intp *tosort) : tosort(tosort) {}
    npy_intp &operator()(npy_intp i) const { return tosort[i]; }
};

template <>
struct Idx<true> {
    npy_intp *tosort;
    Idx(npy_intp *tosort) : tosort(tosort) {}
    npy_intp operator()(npy_intp i) const { return tosort[i]; }
};

template <class T>
static constexpr bool
inexact()
{
    return !std::is_integral<T>::value;
}
// 检查类型 T 是否为浮点类型（不精确类型）

/*
 * median of 3 pivot strategy
 * gets min and median and moves median to low and min to low + 1
 * for efficient partitioning, see unguarded_partition
 */
// 三元中值选取策略
// 获取最小值和中值，并将中值移动到 low，最小值移动到 low + 1
// 用于高效的分区，参见 unguarded_partition

template <typename Tag, bool arg, typename type>
static inline void
median3_swap_(type *v, npy_intp *tosort, npy_intp low, npy_intp mid,
              npy_intp high)
{
    Idx<arg> idx(tosort);
    Sortee<type, arg> sortee(v, tosort);

    if (Tag::less(v[idx(high)], v[idx(mid)])) {
        std::swap(sortee(high), sortee(mid));
    }
    // 如果 v[high] < v[mid]，则交换 high 和 mid 的值

    if (Tag::less(v[idx(high)], v[idx(low)])) {
        std::swap(sortee(high), sortee(low));
    }
    // 如果 v[high] < v[low]，则交换 high 和 low 的值

    /* move pivot to low */
    // 将 pivot 移动到 low 的位置
    if (Tag::less(v[idx(low)], v[idx(mid)])) {
        std::swap(sortee(low), sortee(mid));
    }
    // 如果 v[low] < v[mid]，则交换 low 和 mid 的值

    /* move 3-lowest element to low + 1 */
    // 将最小的三个元素移动到 low + 1 的位置
    std::swap(sortee(mid), sortee(low + 1));
}

/* select index of median of five elements */
// 选择五个元素的中值的索引位置

template <typename Tag, bool arg, typename type>
static npy_intp
median5_(type *v, npy_intp *tosort)
{
    // 使用 tosort 创建索引对象 idx
    Idx<arg> idx(tosort);
    // 使用 tosort 和 v 创建 Sortee 对象 sortee
    Sortee<type, arg> sortee(v, tosort);

    /* 可以优化，因为我们只需要索引（不需要交换） */
    // 检查 v[idx(1)] 是否小于 v[idx(0)]，如果是，则交换 sortee(1) 和 sortee(0)
    if (Tag::less(v[idx(1)], v[idx(0)])) {
        std::swap(sortee(1), sortee(0));
    }
    // 检查 v[idx(4)] 是否小于 v[idx(3)]，如果是，则交换 sortee(4) 和 sortee(3)
    if (Tag::less(v[idx(4)], v[idx(3)])) {
        std::swap(sortee(4), sortee(3));
    }
    // 检查 v[idx(3)] 是否小于 v[idx(0)]，如果是，则交换 sortee(3) 和 sortee(0)
    if (Tag::less(v[idx(3)], v[idx(0)])) {
        std::swap(sortee(3), sortee(0));
    }
    // 检查 v[idx(4)] 是否小于 v[idx(1)]，如果是，则交换 sortee(4) 和 sortee(1)
    if (Tag::less(v[idx(4)], v[idx(1)])) {
        std::swap(sortee(4), sortee(1));
    }
    // 检查 v[idx(2)] 是否小于 v[idx(1)]，如果是，则交换 sortee(2) 和 sortee(1)
    if (Tag::less(v[idx(2)], v[idx(1)])) {
        std::swap(sortee(2), sortee(1));
    }
    // 检查 v[idx(3)] 是否小于 v[idx(2)]，如果是，进一步检查 v[idx(3)] 是否小于 v[idx(1)]，返回对应索引
    if (Tag::less(v[idx(3)], v[idx(2)])) {
        if (Tag::less(v[idx(3)], v[idx(1)])) {
            return 1;
        }
        else {
            return 3;
        }
    }
    else {
        /* v[1] and v[2] swapped into order above */
        // 如果 v[1] 和 v[2] 在上面的步骤中已经交换顺序，则返回索引 2
        return 2;
    }
/*
 * partition and return the index were the pivot belongs
 * the data must have following property to avoid bound checks:
 *                  ll ... hh
 * lower-than-pivot [x x x x] larger-than-pivot
 */
template <typename Tag, bool arg, typename type>
static inline void
unguarded_partition_(type *v, npy_intp *tosort, const type pivot, npy_intp *ll,
                     npy_intp *hh)
{
    // 使用模板参数 Tag 实例化 Idx 和 Sortee 类，分别初始化索引和排序对象
    Idx<arg> idx(tosort);
    Sortee<type, arg> sortee(v, tosort);

    // 无限循环，直到 ll 和 hh 指针相遇
    for (;;) {
        // 向右移动 ll 指针，直到找到第一个大于等于 pivot 的元素
        do {
            (*ll)++;
        } while (Tag::less(v[idx(*ll)], pivot));
        // 向左移动 hh 指针，直到找到第一个小于等于 pivot 的元素
        do {
            (*hh)--;
        } while (Tag::less(pivot, v[idx(*hh)]));

        // 如果 hh 指针在 ll 指针左侧，则退出循环
        if (*hh < *ll) {
            break;
        }

        // 交换 ll 和 hh 指针所指的元素
        std::swap(sortee(*ll), sortee(*hh));
    }
}

/*
 * select median of median of blocks of 5
 * if used as partition pivot it splits the range into at least 30%/70%
 * allowing linear time worstcase quickselect
 */
template <typename Tag, bool arg, typename type>
static npy_intp
median_of_median5_(type *v, npy_intp *tosort, const npy_intp num,
                   npy_intp *pivots, npy_intp *npiv)
{
    // 使用模板参数 Tag 实例化 Idx 和 Sortee 类，分别初始化索引和排序对象
    Idx<arg> idx(tosort);
    Sortee<type, arg> sortee(v, tosort);

    npy_intp i, subleft;
    npy_intp right = num - 1;
    npy_intp nmed = (right + 1) / 5;

    // 将数据分成大小为 5 的块，对每个块使用 median5_ 函数找到中位数，并与第 i 个元素交换
    for (i = 0, subleft = 0; i < nmed; i++, subleft += 5) {
        npy_intp m = median5_<Tag, arg>(v + (arg ? 0 : subleft),
                                        tosort + (arg ? subleft : 0));
        std::swap(sortee(subleft + m), sortee(i));
    }

    // 如果块的数量大于 2，则使用 introselect_ 函数对中位数块进行选择
    if (nmed > 2) {
        introselect_<Tag, arg>(v, tosort, nmed, nmed / 2, pivots, npiv);
    }
    // 返回中位数块的中位数索引
    return nmed / 2;
}

/*
 * N^2 selection, fast only for very small kth
 * useful for close multiple partitions
 * (e.g. even element median, interpolating percentile)
 */
template <typename Tag, bool arg, typename type>
static int
dumb_select_(type *v, npy_intp *tosort, npy_intp num, npy_intp kth)
{
    // 使用模板参数 Tag 实例化 Idx 和 Sortee 类，分别初始化索引和排序对象
    Idx<arg> idx(tosort);
    Sortee<type, arg> sortee(v, tosort);

    npy_intp i;
    // 选择排序算法，选择最小的 kth+1 个元素，并进行交换排序
    for (i = 0; i <= kth; i++) {
        npy_intp minidx = i;
        type minval = v[idx(i)];
        npy_intp k;
        for (k = i + 1; k < num; k++) {
            if (Tag::less(v[idx(k)], minval)) {
                minidx = k;
                minval = v[idx(k)];
            }
        }
        std::swap(sortee(i), sortee(minidx));
    }

    return 0;
}

/*
 * iterative median of 3 quickselect with cutoff to median-of-medians-of5
 * receives stack of already computed pivots in v to minimize the
 * partition size were kth is searched in
 *
 * area that needs partitioning in [...]
 * kth 0:  [8  7  6  5  4  3  2  1  0] -> med3 partitions elements [4, 2, 0]
 *          0  1  2  3  4  8  7  5  6  -> pop requested kth -> stack [4, 2]
 * kth 3:   0  1  2 [3] 4  8  7  5  6  -> stack [4]
 * kth 5:   0  1  2  3  4 [8  7  5  6] -> stack [6]
 * kth 8:   0  1  2  3  4  5  6 [8  7] -> stack []
 *
 */
template <typename Tag, bool arg, typename type>
NPY_NO_EXPORT int
    // 定义一个索引对象 idx，用于对 tosort 数组进行索引操作
    Idx<arg> idx(tosort);
    // 定义一个排序对象 sortee，用于对 v 数组按照 tosort 数组的顺序进行排序
    Sortee<type, arg> sortee(v, tosort);

    // 初始化低位和高位的索引
    npy_intp low = 0;
    npy_intp high = num - 1;
    // 深度限制的初始值
    int depth_limit;

    // 如果 npiv 为 NULL，则将 pivots 设置为 NULL
    if (npiv == NULL) {
        pivots = NULL;
    }

    // 循环直到 pivots 不为 NULL 且 *npiv 大于 0
    while (pivots != NULL && *npiv > 0) {
        // 如果 pivots[*npiv - 1] 大于 kth，则将 high 设置为 pivots[*npiv - 1] - 1，并退出循环
        if (pivots[*npiv - 1] > kth) {
            /* pivot larger than kth set it as upper bound */
            high = pivots[*npiv - 1] - 1;
            break;
        }
        // 如果 pivots[*npiv - 1] 等于 kth，则表示 kth 已经在之前的迭代中找到，直接返回 0
        else if (pivots[*npiv - 1] == kth) {
            /* kth was already found in a previous iteration -> done */
            return 0;
        }

        // 否则，将 low 设置为 pivots[*npiv - 1] + 1
        low = pivots[*npiv - 1] + 1;

        /* pop from stack */
        // 从堆栈中弹出一个元素
        *npiv -= 1;
    }

    /*
     * 对于非 NULL 的 pivots 且 *npiv 大于 0 的情况下，使用一个更快的 O(n*kth) 算法，
     * 例如用于插值百分位数的情况
     */
    if (kth - low < 3) {
        // 调用 dumb_select_ 函数进行选择操作，选择范围为 v + (arg ? 0 : low) 到 v + high，数量为 high - low + 1，要找的第 kth - low 个元素
        dumb_select_<Tag, arg>(v + (arg ? 0 : low), tosort + (arg ? low : 0),
                               high - low + 1, kth - low);
        // 将 kth 和 kth 作为枢轴存储到 pivots 中，并更新 *npiv
        store_pivot(kth, kth, pivots, npiv);
        return 0;
    }

    // 如果类型为 inexact<type>() 并且 kth 等于 num - 1
    else if (inexact<type>() && kth == num - 1) {
        /* useful to check if NaN present via partition(d, (x, -1)) */
        // 初始化 maxidx 为 low，maxval 为 v[idx(low)]
        npy_intp k;
        npy_intp maxidx = low;
        type maxval = v[idx(low)];
        // 遍历从 low + 1 到 num 的所有元素，找到最大的元素及其索引
        for (k = low + 1; k < num; k++) {
            if (!Tag::less(v[idx(k)], maxval)) {
                maxidx = k;
                maxval = v[idx(k)];
            }
        }
        // 将 kth 和 maxidx 处的元素进行交换
        std::swap(sortee(kth), sortee(maxidx));
        return 0;
    }

    // 计算深度限制，为 num 的最高位的两倍
    depth_limit = npy_get_msb(num) * 2;

    // 确保至少有三个元素
    # 用于二分查找，直到 low 和 high 相差小于 1 为止
    for (; low + 1 < high;) {
        # 初始化 ll 和 hh
        npy_intp ll = low + 1;
        npy_intp hh = high;

        """
         * 如果使用中位数3法没有足够的进展，
         * 则回退到中位数5法的枢纽以应对线性最坏情况，
         * 对于小尺寸的 med3 需要无守护分区
         """
        if (depth_limit > 0 || hh - ll < 5) {
            const npy_intp mid = low + (high - low) / 2;
            # 中位数3枢纽策略，为了有效分区而进行交换
            median3_swap_<Tag, arg>(v, tosort, low, mid, high);
        }
        else {
            npy_intp mid;
            # FIXME: 始终使用枢纽来优化这种迭代分区
            mid = ll + median_of_median5_<Tag, arg>(v + (arg ? 0 : ll),
                                                    tosort + (arg ? ll : 0),
                                                    hh - ll, NULL, NULL);
            std::swap(sortee(mid), sortee(low));
            # 适应比 med3 枢纽更大的分区
            ll--;
            hh++;
        }

        # 深度限制减少一层
        depth_limit--;

        """
         * 找到放置枢纽的位置（在低位）：
         * 先前的交换消除了边界检查的需求
         * 枢纽 3-最低 [x x x] 3-最高
         """
        unguarded_partition_<Tag, arg>(v, tosort, v[idx(low)], &ll, &hh);

        # 将枢纽移动到相应的位置
        std::swap(sortee(low), sortee(hh));

        # 存储 kth 枢纽
        if (hh != kth) {
            store_pivot(hh, kth, pivots, npiv);
        }

        if (hh >= kth) {
            high = hh - 1;
        }
        if (hh <= kth) {
            low = ll;
        }
    }

    # 两个元素时的处理
    if (high == low + 1) {
        if (Tag::less(v[idx(high)], v[idx(low)]) {
            std::swap(sortee(high), sortee(low));
        }
    }
    # 存储 kth 枢纽
    store_pivot(kth, kth, pivots, npiv);

    # 返回结果
    return 0;
/*
 *****************************************************************************
 **                             GENERATOR                                   **
 *****************************************************************************
 */

template <typename Tag>
static int
introselect_noarg(void *v, npy_intp num, npy_intp kth, npy_intp *pivots,
                  npy_intp *npiv, npy_intp nkth, void *)
{
    // 使用标签 Tag 推断出元素类型 T，如果 kth 为 1 并且通过 quickselect_dispatch 可以完成快速选择，则返回 0
    using T = typename std::conditional<std::is_same_v<Tag, npy::half_tag>, np::Half, typename Tag::type>::type;
    if ((nkth == 1) && (quickselect_dispatch((T *)v, num, kth))) {
        return 0;
    }
    // 否则调用 introselect_ 函数进行选择排序
    return introselect_<Tag, false>((typename Tag::type *)v, nullptr, num, kth,
                                    pivots, npiv);
}

template <typename Tag>
static int
introselect_arg(void *v, npy_intp *tosort, npy_intp num, npy_intp kth,
                npy_intp *pivots, npy_intp *npiv, npy_intp nkth, void *)
{
    // 使用标签 Tag 推断出元素类型 T，如果 kth 为 1 并且通过 argquickselect_dispatch 可以完成快速选择，则返回 0
    using T = typename Tag::type;
    if ((nkth == 1) && (argquickselect_dispatch((T *)v, tosort, num, kth))) {
        return 0;
    }
    // 否则调用 introselect_ 函数进行带排序索引的选择排序
    return introselect_<Tag, true>((typename Tag::type *)v, tosort, num, kth,
                                   pivots, npiv);
}

struct arg_map {
    int typenum; // 数据类型编号
    PyArray_PartitionFunc *part[NPY_NSELECTS]; // 非带索引和带索引的分区函数指针数组
    PyArray_ArgPartitionFunc *argpart[NPY_NSELECTS]; // 带排序索引的非带索引和带索引的分区函数指针数组
};

template <class... Tags>
static constexpr std::array<arg_map, sizeof...(Tags)>
make_partition_map(npy::taglist<Tags...>)
{
    // 创建一个包含所有标签类型的分区映射数组
    return std::array<arg_map, sizeof...(Tags)>{
            arg_map{Tags::type_value, {&introselect_noarg<Tags>},
                {&introselect_arg<Tags>}}...};
}

struct partition_t {
    using taglist =
            npy::taglist<npy::bool_tag, npy::byte_tag, npy::ubyte_tag,
                         npy::short_tag, npy::ushort_tag, npy::int_tag,
                         npy::uint_tag, npy::long_tag, npy::ulong_tag,
                         npy::longlong_tag, npy::ulonglong_tag, npy::half_tag,
                         npy::float_tag, npy::double_tag, npy::longdouble_tag,
                         npy::cfloat_tag, npy::cdouble_tag,
                         npy::clongdouble_tag>;

    // 创建静态常量数组 map，存储所有数据类型的分区映射
    static constexpr std::array<arg_map, taglist::size> map =
            make_partition_map(taglist());
};

// 获取给定数据类型和选择类型的非带索引分区函数指针
static inline PyArray_PartitionFunc *
_get_partition_func(int type, NPY_SELECTKIND which)
{
    npy_intp i;
    npy_intp ntypes = partition_t::map.size();

    if ((int)which < 0 || (int)which >= NPY_NSELECTS) {
        return NULL; // 如果选择类型不在合法范围内，返回空指针
    }
    // 遍历分区映射数组，根据数据类型找到对应的非带索引分区函数指针
    for (i = 0; i < ntypes; i++) {
        if (type == partition_t::map[i].typenum) {
            return partition_t::map[i].part[which];
        }
    }
    return NULL; // 找不到对应的分区函数，返回空指针
}

// 获取给定数据类型和选择类型的带索引分区函数指针
static inline PyArray_ArgPartitionFunc *
_get_argpartition_func(int type, NPY_SELECTKIND which)
{
    npy_intp i;
    npy_intp ntypes = partition_t::map.size();

    // 如果选择类型不在合法范围内，返回空指针
    if ((int)which < 0 || (int)which >= NPY_NSELECTS) {
        return NULL;
    }
    // 遍历分区映射数组，根据数据类型找到对应的带索引分区函数指针
    for (i = 0; i < ntypes; i++) {
        if (type == partition_t::map[i].typenum) {
            return partition_t::map[i].argpart[which];
        }
    }
    return NULL; // 找不到对应的分区函数，返回空指针
}
    // 遍历从 0 到 ntypes 的整数序列
    for (i = 0; i < ntypes; i++) {
        // 检查当前 type 是否等于 partition_t::map[i] 的 typenum
        if (type == partition_t::map[i].typenum) {
            // 如果匹配，则返回 partition_t::map[i] 中指定 which 索引位置的 argpart
            return partition_t::map[i].argpart[which];
        }
    }
    // 如果未找到匹配项，返回空指针 NULL
    return NULL;
}

/*
 *****************************************************************************
 **                            C INTERFACE                                  **
 *****************************************************************************
 */

# 定义 C 语言接口，以下是 C 函数的声明

extern "C" {
// 返回一个 PyArray_PartitionFunc 指针，调用 _get_partition_func 函数
NPY_NO_EXPORT PyArray_PartitionFunc *
get_partition_func(int type, NPY_SELECTKIND which)
{
    return _get_partition_func(type, which);
}

// 返回一个 PyArray_ArgPartitionFunc 指针，调用 _get_argpartition_func 函数
NPY_NO_EXPORT PyArray_ArgPartitionFunc *
get_argpartition_func(int type, NPY_SELECTKIND which)
{
    return _get_argpartition_func(type, which);
}
}
```