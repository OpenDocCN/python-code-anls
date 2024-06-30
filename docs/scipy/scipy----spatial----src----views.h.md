# `D:\src\scipysrc\scipy\scipy\spatial\src\views.h`

```
#pragma once

// `#pragma once` 指令确保头文件只被编译一次，防止多重包含问题


#include <vector>
#include <array>
#include <cstdint>

// 包含标准库头文件 `<vector>`、`<array>` 和 `<cstdint>`，用于定义 `std::vector`、`std::array` 和整数类型的标准库支持。


struct ArrayDescriptor {
    ArrayDescriptor(intptr_t ndim):
        ndim(ndim), shape(ndim, 1), strides(ndim, 0) {
    }

    intptr_t ndim;
    intptr_t element_size;
    std::vector<intptr_t> shape, strides;
};

// 定义结构体 `ArrayDescriptor`，描述一个多维数组的结构信息。
// - `intptr_t ndim`：数组的维度。
// - `intptr_t element_size`：数组元素的大小（尚未初始化）。
// - `std::vector<intptr_t> shape`：存储数组各维度的大小。
// - `std::vector<intptr_t> strides`：存储数组各维度的步进大小。


template <typename T>
struct StridedView2D {
    std::array<intptr_t, 2> shape;
    std::array<intptr_t, 2> strides;
    T* data;

    T& operator()(intptr_t i, intptr_t j) {
        return data[i * strides[0] + j * strides[1]];
    }
};

// 定义模板结构体 `StridedView2D`，表示一个二维步进视图。
// - `std::array<intptr_t, 2> shape`：保存视图的形状（行数和列数）。
// - `std::array<intptr_t, 2> strides`：保存视图的步进大小（行步进和列步进）。
// - `T* data`：指向视图数据的指针。
// - `T& operator()(intptr_t i, intptr_t j)`：重载函数调用运算符，根据步进和索引计算并返回视图中指定位置 `(i, j)` 的元素引用。
```