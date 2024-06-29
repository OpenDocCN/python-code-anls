# `.\numpy\numpy\_core\src\npysort\highway_qsort.dispatch.cpp`

```
#include "highway_qsort.hpp"
#include "hwy/contrib/sort/vqsort-inl.h"

// 定义宏，指定只使用静态版本的 VQSort
#define VQSORT_ONLY_STATIC 1

// 定义模板函数 DISPATCH_VQSORT，用于生成特定类型的 QSort 函数
#define DISPATCH_VQSORT(TYPE) \
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(TYPE *arr, intptr_t size) \
{ \
    // 调用 HWY 命名空间中的 VQSortStatic 函数，对数组进行升序排序
    hwy::HWY_NAMESPACE::VQSortStatic(arr, size, hwy::SortAscending()); \
} \

namespace np { namespace highway { namespace qsort_simd {

    // 实例化 DISPATCH_VQSORT 模板函数，针对不同的数据类型
    DISPATCH_VQSORT(int32_t)
    DISPATCH_VQSORT(uint32_t)
    DISPATCH_VQSORT(int64_t)
    DISPATCH_VQSORT(uint64_t)
    DISPATCH_VQSORT(double)
    DISPATCH_VQSORT(float)

} } } // np::highway::qsort_simd
```