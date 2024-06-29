# `.\numpy\numpy\_core\src\npysort\highway_qsort_16bit.dispatch.cpp`

```
// 包含自定义的排序头文件 "highway_qsort.hpp"
#include "highway_qsort.hpp"
// 定义宏 VQSORT_ONLY_STATIC 为 1，用于特定的静态排序
#define VQSORT_ONLY_STATIC 1
// 包含高速公路库中的排序实现头文件 "hwy/contrib/sort/vqsort-inl.h"
#include "hwy/contrib/sort/vqsort-inl.h"

// 包含自定义的快速排序头文件 "quicksort.hpp"
#include "quicksort.hpp"

// 命名空间声明：np -> highway -> qsort_simd
namespace np { namespace highway { namespace qsort_simd {

// 模板特化：处理半精度浮点数数组的 QSort 函数
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(Half *arr, intptr_t size)
{
    // 如果支持半精度浮点数操作
#if HWY_HAVE_FLOAT16
    // 调用高速公路库中的静态向量化排序，对半精度浮点数数组 arr 进行升序排序
    hwy::HWY_NAMESPACE::VQSortStatic(reinterpret_cast<hwy::float16_t*>(arr), size, hwy::SortAscending());
// 如果不支持半精度浮点数操作
#else
    // 调用自定义的快速排序，对半精度浮点数数组 arr 进行排序
    sort::Quick(arr, size);
#endif
}

// 模板特化：处理无符号 16 位整数数组的 QSort 函数
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(uint16_t *arr, intptr_t size)
{
    // 调用高速公路库中的静态向量化排序，对无符号 16 位整数数组 arr 进行升序排序
    hwy::HWY_NAMESPACE::VQSortStatic(arr, size, hwy::SortAscending());
}

// 模板特化：处理有符号 16 位整数数组的 QSort 函数
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(int16_t *arr, intptr_t size)
{
    // 调用高速公路库中的静态向量化排序，对有符号 16 位整数数组 arr 进行升序排序
    hwy::HWY_NAMESPACE::VQSortStatic(arr, size, hwy::SortAscending());
}

} } } // np::highway::qsort_simd
```