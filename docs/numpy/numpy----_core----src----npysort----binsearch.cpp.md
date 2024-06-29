# `.\numpy\numpy\_core\src\npysort\binsearch.cpp`

```py
/* -*- c -*- */

// 禁用已弃用的 NumPy API
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

// 引入 NumPy 的数组类型和通用头文件
#include "numpy/ndarraytypes.h"
#include "numpy/npy_common.h"

// 引入排序和二分搜索相关的自定义头文件
#include "npy_binsearch.h"
#include "npy_sort.h"
#include "numpy_tag.h"

// 引入标准库头文件
#include <array>
#include <functional>  // 用于 std::less 和 std::less_equal

// binsearch 函数的搜索变体的枚举器
enum arg_t
{
    noarg,  // 无参数
    arg     // 有参数
};
enum side_t
{
    left,   // 左侧
    right   // 右侧
};

// 将枚举器映射到比较器的模板
template <class Tag, side_t side>
struct side_to_cmp;

// 左侧搜索比较器的特化
template <class Tag>
struct side_to_cmp<Tag, left> {
    static constexpr auto value = Tag::less;  // 使用 Tag 的 less 函数对象
};

// 右侧搜索比较器的特化
template <class Tag>
struct side_to_cmp<Tag, right> {
    static constexpr auto value = Tag::less_equal;  // 使用 Tag 的 less_equal 函数对象
};

// 将搜索方向映射到通用比较器的模板
template <side_t side>
struct side_to_generic_cmp;

// 左侧搜索通用比较器的特化
template <>
struct side_to_generic_cmp<left> {
    using type = std::less<int>;  // 使用 std::less 进行整数比较
};

// 右侧搜索通用比较器的特化
template <>
struct side_to_generic_cmp<right> {
    using type = std::less_equal<int>;  // 使用 std::less_equal 进行整数比较
};

/*
 *****************************************************************************
 **                            NUMERIC SEARCHES                             **
 *****************************************************************************
 */

// 泛型二分搜索函数的实现
template <class Tag, side_t side>
static void
binsearch(const char *arr, const char *key, char *ret, npy_intp arr_len,
          npy_intp key_len, npy_intp arr_str, npy_intp key_str,
          npy_intp ret_str, PyArrayObject *)
{
    using T = typename Tag::type;  // 使用 Tag 的类型 T
    auto cmp = side_to_cmp<Tag, side>::value;  // 获取比较器函数对象
    npy_intp min_idx = 0;
    npy_intp max_idx = arr_len;
    T last_key_val;

    if (key_len == 0) {
        return;  // 如果 key 长度为 0，则直接返回
    }
    last_key_val = *(const T *)key;  // 获取第一个 key 的值

    for (; key_len > 0; key_len--, key += key_str, ret += ret_str) {
        const T key_val = *(const T *)key;  // 获取当前 key 的值
        /*
         * 根据前一个 key 值更新索引，当 key 是有序的时候能显著提升搜索效率，
         * 但对于纯随机的情况略微降低速度。
         */
        if (cmp(last_key_val, key_val)) {
            max_idx = arr_len;  // 如果上一个 key 小于当前 key，则更新最大索引
        }
        else {
            min_idx = 0;  // 否则重置最小索引
            max_idx = (max_idx < arr_len) ? (max_idx + 1) : arr_len;  // 更新最大索引
        }

        last_key_val = key_val;  // 更新上一个 key 的值为当前 key

        while (min_idx < max_idx) {
            const npy_intp mid_idx = min_idx + ((max_idx - min_idx) >> 1);  // 计算中间索引
            const T mid_val = *(const T *)(arr + mid_idx * arr_str);  // 获取中间值
            if (cmp(mid_val, key_val)) {
                min_idx = mid_idx + 1;  // 更新最小索引
            }
            else {
                max_idx = mid_idx;  // 更新最大索引
            }
        }
        *(npy_intp *)ret = min_idx;  // 将结果存入 ret 中
    }
}

// 带参数的泛型二分搜索函数的实现
template <class Tag, side_t side>
static int
argbinsearch(const char *arr, const char *key, const char *sort, char *ret,
             npy_intp arr_len, npy_intp key_len, npy_intp arr_str,
             npy_intp key_str, npy_intp sort_str, npy_intp ret_str,
             PyArrayObject *)
{
    using T = typename Tag::type;  // 使用 Tag 的类型 T
    auto cmp = side_to_cmp<Tag, side>::value;  // 获取比较器函数对象
    npy_intp min_idx = 0;  // 初始化最小索引为0，表示搜索范围的起始位置
    npy_intp max_idx = arr_len;  // 初始化最大索引为数组长度，表示搜索范围的结束位置
    T last_key_val;  // 声明变量用于存储上一个比较的关键值

    if (key_len == 0) {
        return 0;  // 如果关键字长度为0，直接返回0
    }
    last_key_val = *(const T *)key;  // 获取关键字的第一个元素作为初始的上一个关键值

    for (; key_len > 0; key_len--, key += key_str, ret += ret_str) {
        const T key_val = *(const T *)key;  // 获取当前关键字的值

        /*
         * 根据前一个关键字值更新索引范围，
         * 当关键字按顺序排列时，这样做可以显著提升搜索效率，
         * 但是对于完全随机的关键字会略微减慢速度。
         */
        if (cmp(last_key_val, key_val)) {
            max_idx = arr_len;  // 如果当前关键字与上一个不同，则更新最大索引为数组长度
        }
        else {
            min_idx = 0;  // 如果当前关键字与上一个相同，则重置最小索引为0
            max_idx = (max_idx < arr_len) ? (max_idx + 1) : arr_len;  // 并增加最大索引，但不超过数组长度
        }

        last_key_val = key_val;  // 更新上一个关键值为当前关键值

        while (min_idx < max_idx) {
            const npy_intp mid_idx = min_idx + ((max_idx - min_idx) >> 1);  // 计算中间索引位置
            const npy_intp sort_idx = *(npy_intp *)(sort + mid_idx * sort_str);  // 获取排序后的索引值
            T mid_val;

            if (sort_idx < 0 || sort_idx >= arr_len) {
                return -1;  // 如果排序索引超出有效范围，返回错误标志
            }

            mid_val = *(const T *)(arr + sort_idx * arr_str);  // 获取数组中排序后位置的值

            if (cmp(mid_val, key_val)) {
                min_idx = mid_idx + 1;  // 如果中间值小于关键值，更新最小索引为中间索引加一
            }
            else {
                max_idx = mid_idx;  // 否则更新最大索引为中间索引
            }
        }
        *(npy_intp *)ret = min_idx;  // 将最小索引值写入返回值中
    }
    return 0;  // 返回成功标志
/*
 * 使用模板函数实现二分查找算法，查找指定键在排序数组中的位置
 * 参数说明：
 * - arr: 输入数组的起始地址
 * - key: 要查找的键的起始地址
 * - ret: 返回的位置的地址
 * - arr_len: 数组的长度
 * - key_len: 键的长度（假设是键的长度，可能是用来迭代的计数器）
 * - arr_str: 数组中每个元素的步长（假设是元素之间的偏移量）
 * - key_str: 键中每个字符的步长（假设是字符之间的偏移量）
 * - ret_str: 返回位置指针每次迭代时的步长
 * - cmp: 用于比较元素的对象
 */
template <side_t side>
static void
npy_binsearch(const char *arr, const char *key, char *ret, npy_intp arr_len,
              npy_intp key_len, npy_intp arr_str, npy_intp key_str,
              npy_intp ret_str, PyArrayObject *cmp)
{
    // 使用模板元编程，根据边界方向选择相应的比较器类型
    using Cmp = typename side_to_generic_cmp<side>::type;
    // 获取用于比较元素的函数指针
    PyArray_CompareFunc *compare = PyDataType_GetArrFuncs(PyArray_DESCR(cmp))->compare;
    // 初始化二分查找的起始和结束位置
    npy_intp min_idx = 0;
    npy_intp max_idx = arr_len;
    // 记录上一个键的位置
    const char *last_key = key;

    // 循环遍历每个键
    for (; key_len > 0; key_len--, key += key_str, ret += ret_str) {
        /*
         * 当前键与上一个键进行比较，根据比较结果更新搜索范围的下界和上界
         * 当键有序时，根据上一个键的比较结果更新搜索范围，优化查找速度；
         * 当键是随机的时，可能会轻微减慢速度。
         */
        if (Cmp{}(compare(last_key, key, cmp), 0)) {
            max_idx = arr_len;
        }
        else {
            min_idx = 0;
            max_idx = (max_idx < arr_len) ? (max_idx + 1) : arr_len;
        }

        // 更新上一个键的位置
        last_key = key;

        // 二分查找算法
        while (min_idx < max_idx) {
            const npy_intp mid_idx = min_idx + ((max_idx - min_idx) >> 1);
            const char *arr_ptr = arr + mid_idx * arr_str;

            // 根据比较结果更新搜索范围
            if (Cmp{}(compare(arr_ptr, key, cmp), 0)) {
                min_idx = mid_idx + 1;
            }
            else {
                max_idx = mid_idx;
            }
        }
        // 将当前位置保存到返回位置的指针中
        *(npy_intp *)ret = min_idx;
    }
}

/*
 * 使用模板函数实现二分查找算法，查找指定键在排序数组中的位置
 * 参数说明：
 * - arr: 输入数组的起始地址
 * - key: 要查找的键的起始地址
 * - sort: 排序数组的起始地址（假设是排序数组，可能用于比较）
 * - ret: 返回的位置的地址
 * - arr_len: 数组的长度
 * - key_len: 键的长度（假设是键的长度，可能是用来迭代的计数器）
 * - arr_str: 数组中每个元素的步长（假设是元素之间的偏移量）
 * - key_str: 键中每个字符的步长（假设是字符之间的偏移量）
 * - sort_str: 排序数组中每个元素的步长（假设是元素之间的偏移量）
 * - ret_str: 返回位置指针每次迭代时的步长
 * - cmp: 用于比较元素的对象
 * 返回值：
 * - 返回找到的位置的索引，或者是否找到（假设是成功返回索引，失败返回-1）
 */
template <side_t side>
static int
npy_argbinsearch(const char *arr, const char *key, const char *sort, char *ret,
                 npy_intp arr_len, npy_intp key_len, npy_intp arr_str,
                 npy_intp key_str, npy_intp sort_str, npy_intp ret_str,
                 PyArrayObject *cmp)
{
    // 使用模板元编程，根据边界方向选择相应的比较器类型
    using Cmp = typename side_to_generic_cmp<side>::type;
    // 获取用于比较元素的函数指针
    PyArray_CompareFunc *compare = PyDataType_GetArrFuncs(PyArray_DESCR(cmp))->compare;
    // 初始化二分查找的起始和结束位置
    npy_intp min_idx = 0;
    npy_intp max_idx = arr_len;
    // 记录上一个键的位置
    const char *last_key = key;

    // 循环遍历每个键
    for (; key_len > 0; key_len--, key += key_str, ret += ret_str) {
        /*
         * 当前键与上一个键进行比较，根据比较结果更新搜索范围的下界和上界
         * 当键有序时，根据上一个键的比较结果更新搜索范围，优化查找速度；
         * 当键是随机的时，可能会轻微减慢速度。
         */
        if (Cmp{}(compare(last_key, key, cmp), 0)) {
            max_idx = arr_len;
        }
        else {
            min_idx = 0;
            max_idx = (max_idx < arr_len) ? (max_idx + 1) : arr_len;
        }

        // 更新上一个键的位置
        last_key = key;

        // 二分查找算法
        while (min_idx < max_idx) {
            const npy_intp mid_idx = min_idx + ((max_idx - min_idx) >> 1);
            const char *arr_ptr = arr + mid_idx * arr_str;

            // 根据比较结果更新搜索范围
            if (Cmp{}(compare(arr_ptr, key, cmp), 0)) {
                min_idx = mid_idx + 1;
            }
            else {
                max_idx = mid_idx;
            }
        }
        // 将当前位置保存到返回位置的指针中
        *(npy_intp *)ret = min_idx;
    }

    // 返回找到的位置的索引，或者是否找到（假设是成功返回索引，失败返回-1）
    return (min_idx < arr_len && compare(arr + min_idx * arr_str, key, cmp) == 0) ? min_idx : -1;
}
    for (; key_len > 0; key_len--, key += key_str, ret += ret_str) {
        /*
         * 根据前一个键的比较结果更新索引，当键有序时，可以显著提升搜索速度，
         * 但对于完全随机的键略微降低速度。
         */
        
        // 根据最后一个键与当前键的比较结果，更新最大索引
        if (Cmp{}(compare(last_key, key, cmp), 0)) {
            max_idx = arr_len;
        }
        else {
            min_idx = 0;
            max_idx = (max_idx < arr_len) ? (max_idx + 1) : arr_len;
        }

        // 更新最后一个键为当前键
        last_key = key;

        // 二分查找在排序数组中定位键的位置
        while (min_idx < max_idx) {
            // 计算中间索引
            const npy_intp mid_idx = min_idx + ((max_idx - min_idx) >> 1);
            // 获取排序索引
            const npy_intp sort_idx = *(npy_intp *)(sort + mid_idx * sort_str);
            const char *arr_ptr;

            // 检查排序索引是否有效
            if (sort_idx < 0 || sort_idx >= arr_len) {
                return -1;
            }

            // 获取数组中的指针
            arr_ptr = arr + sort_idx * arr_str;

            // 比较数组中的值和当前键的大小
            if (Cmp{}(compare(arr_ptr, key, cmp), 0)) {
                // 更新最小索引以查找更大的键
                min_idx = mid_idx + 1;
            }
            else {
                // 更新最大索引以查找更小的键
                max_idx = mid_idx;
            }
        }
        
        // 将搜索结果写入返回数组中
        *(npy_intp *)ret = min_idx;
    }
    // 返回成功标志
    return 0;
/*
 *****************************************************************************
 **                             GENERATOR                                   **
 *****************************************************************************
 */

// 模板声明：二分搜索基类
template <arg_t arg>
struct binsearch_base;

// 特化模板：针对具有参数的情况（arg != noarg）
template <>
struct binsearch_base<arg> {
    // 定义函数指针类型
    using function_type = PyArray_ArgBinSearchFunc *;
    
    // 值类型结构体定义
    struct value_type {
        int typenum;  // 类型编号
        function_type binsearch[NPY_NSEARCHSIDES];  // 二分搜索函数指针数组
    };
    
    // 生成二分搜索映射表的静态方法
    template <class... Tags>
    static constexpr std::array<value_type, sizeof...(Tags)>
    make_binsearch_map(npy::taglist<Tags...>)
    {
        // 返回由标签列表生成的值类型数组
        return std::array<value_type, sizeof...(Tags)>{
                value_type{Tags::type_value,
                           {(function_type)&argbinsearch<Tags, left>,
                            (function_type)argbinsearch<Tags, right>}}...};
    }
    
    // 静态成员：无参数的二分搜索函数指针数组
    static constexpr std::array<function_type, 2> npy_map = {
            (function_type)&npy_argbinsearch<left>,
            (function_type)&npy_argbinsearch<right>};
};

// 初始化静态成员：无参数的二分搜索函数指针数组
constexpr std::array<binsearch_base<arg>::function_type, 2>
        binsearch_base<arg>::npy_map;

// 特化模板：针对无参数的情况（arg == noarg）
template <>
struct binsearch_base<noarg> {
    // 定义函数指针类型
    using function_type = PyArray_BinSearchFunc *;
    
    // 值类型结构体定义
    struct value_type {
        int typenum;  // 类型编号
        function_type binsearch[NPY_NSEARCHSIDES];  // 二分搜索函数指针数组
    };
    
    // 生成二分搜索映射表的静态方法
    template <class... Tags>
    static constexpr std::array<value_type, sizeof...(Tags)>
    make_binsearch_map(npy::taglist<Tags...>)
    {
        // 返回由标签列表生成的值类型数组
        return std::array<value_type, sizeof...(Tags)>{
                value_type{Tags::type_value,
                           {(function_type)&binsearch<Tags, left>,
                            (function_type)binsearch<Tags, right>}}...};
    }
    
    // 静态成员：有参数的二分搜索函数指针数组
    static constexpr std::array<function_type, 2> npy_map = {
            (function_type)&npy_binsearch<left>,
            (function_type)&npy_binsearch<right>};
};

// 初始化静态成员：有参数的二分搜索函数指针数组
constexpr std::array<binsearch_base<noarg>::function_type, 2>
        binsearch_base<noarg>::npy_map;

// 处理所有二分搜索变体的生成
template <arg_t arg>
struct binsearch_t : binsearch_base<arg> {
    // 继承基类的 make_binsearch_map 方法
    using binsearch_base<arg>::make_binsearch_map;
    // 定义值类型为基类的值类型
    using value_type = typename binsearch_base<arg>::value_type;

    // 定义标签列表类型
    using taglist = npy::taglist<
            /* If adding new types, make sure to keep them ordered by type num
             */
            npy::bool_tag, npy::byte_tag, npy::ubyte_tag, npy::short_tag,
            npy::ushort_tag, npy::int_tag, npy::uint_tag, npy::long_tag,
            npy::ulong_tag, npy::longlong_tag, npy::ulonglong_tag,
            npy::float_tag, npy::double_tag, npy::longdouble_tag, 
            npy::cfloat_tag, npy::cdouble_tag, npy::clongdouble_tag, 
            npy::datetime_tag, npy::timedelta_tag, npy::half_tag>;

    // 静态成员：标签列表映射表
    static constexpr std::array<value_type, taglist::size> map =
            make_binsearch_map(taglist());
};

// 模板声明：二分搜索变体
template <arg_t arg>
/*
 * 用于存储二进制搜索函数的映射表，具体类型和大小由模板参数arg决定
 */
constexpr std::array<typename binsearch_t<arg>::value_type,
                     binsearch_t<arg>::taglist::size>
        binsearch_t<arg>::map;

/*
 * 获取特定类型arg的二进制搜索函数，根据dtype和side参数返回相应的函数指针
 */
template <arg_t arg>
static inline typename binsearch_t<arg>::function_type
_get_binsearch_func(PyArray_Descr *dtype, NPY_SEARCHSIDE side)
{
    // 使用binsearch_t<arg>简化类型名称
    using binsearch = binsearch_t<arg>;
    // 获取映射表的大小
    npy_intp nfuncs = binsearch::map.size();
    // 初始化搜索范围的下界和上界
    npy_intp min_idx = 0;
    npy_intp max_idx = nfuncs;
    // 获取dtype的类型编号
    int type = dtype->type_num;

    // 检查side参数是否有效，若无效则返回NULL
    if ((int)side >= (int)NPY_NSEARCHSIDES) {
        return NULL;
    }

    /*
     * 使用二分搜索算法查找符合dtype类型的二进制搜索函数
     */
    while (min_idx < max_idx) {
        // 计算中间索引
        npy_intp mid_idx = min_idx + ((max_idx - min_idx) >> 1);

        // 根据类型编号比较映射表中的元素，调整搜索范围
        if (binsearch::map[mid_idx].typenum < type) {
            min_idx = mid_idx + 1;
        }
        else {
            max_idx = mid_idx;
        }
    }

    // 如果找到对应dtype类型的函数，则返回该函数指针
    if (min_idx < nfuncs && binsearch::map[min_idx].typenum == type) {
        return binsearch::map[min_idx].binsearch[side];
    }

    // 若未找到对应dtype类型的函数，且dtype具有比较功能，则返回通用的二进制搜索函数
    if (PyDataType_GetArrFuncs(dtype)->compare) {
        return binsearch::npy_map[side];
    }

    // 若以上条件都不满足，则返回NULL
    return NULL;
}

/*
 *****************************************************************************
 **                            C INTERFACE                                  **
 *****************************************************************************
 */

extern "C" {
    // 返回无参数版本（noarg）的二进制搜索函数
    NPY_NO_EXPORT PyArray_BinSearchFunc *
    get_binsearch_func(PyArray_Descr *dtype, NPY_SEARCHSIDE side)
    {
        return _get_binsearch_func<noarg>(dtype, side);
    }

    // 返回带参数版本（arg）的二进制搜索函数
    NPY_NO_EXPORT PyArray_ArgBinSearchFunc *
    get_argbinsearch_func(PyArray_Descr *dtype, NPY_SEARCHSIDE side)
    {
        return _get_binsearch_func<arg>(dtype, side);
    }
}
```