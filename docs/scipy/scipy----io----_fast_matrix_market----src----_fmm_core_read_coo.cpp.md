# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\src\_fmm_core_read_coo.cpp`

```
// 版权声明和许可证信息，指明代码版权和使用许可
// Copyright (C) 2023 Adam Lugowski. All rights reserved.
// Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
// SPDX-License-Identifier: BSD-2-Clause

#include "_fmm_core.hpp"

/**
 * 将 Matrix Market 的数据主体读入三元组中。
 */
template <typename IT, typename VT>
void read_body_coo(read_cursor& cursor, py::array_t<IT>& row, py::array_t<IT>& col, py::array_t<VT>& data) {
    // 检查输入数组的大小是否与矩阵的非零元素数相等，若不相等则抛出异常
    if (row.size() != cursor.header.nnz || col.size() != cursor.header.nnz || data.size() != cursor.header.nnz) {
        throw std::invalid_argument("NumPy Array sizes need to equal matrix nnz");
    }
    // 获取可变的未检查数组视图，用于后续的数据写入
    auto row_unchecked = row.mutable_unchecked();
    auto col_unchecked = col.mutable_unchecked();
    auto data_unchecked = data.mutable_unchecked();
    // 创建三元组解析处理器对象，用于处理三元组数据
    auto handler = fmm::triplet_calling_parse_handler<IT, VT, decltype(row_unchecked), decltype(data_unchecked)>(
        row_unchecked, col_unchecked, data_unchecked);

    // 如果矩阵是坐标格式，则调用 mmread() 方法处理数据，禁用处理数组矩阵的代码路径以减小库的最终大小和编译时间
#ifdef FMM_SCIPY_PRUNE
    fmm::read_matrix_market_body<decltype(handler), fmm::compile_coordinate_only>(cursor.stream(), cursor.header, handler, 1, cursor.options);
#else
    fmm::read_matrix_market_body<decltype(handler), fmm::compile_all>(cursor.stream(), cursor.header, handler, 1, cursor.options);
#endif
    // 关闭游标对象
    cursor.close();
}

// 初始化 COO 格式的读取函数，将其绑定到 Python 模块
void init_read_coo(py::module_ &m) {
    // 绑定不同数据类型的 COO 格式读取函数到 Python 接口
    m.def("read_body_coo", &read_body_coo<int32_t, int64_t>);
    m.def("read_body_coo", &read_body_coo<int32_t, uint64_t>);
    m.def("read_body_coo", &read_body_coo<int32_t, double>);
    m.def("read_body_coo", &read_body_coo<int32_t, std::complex<double>>);

    m.def("read_body_coo", &read_body_coo<int64_t, int64_t>);
    m.def("read_body_coo", &read_body_coo<int64_t, uint64_t>);
    m.def("read_body_coo", &read_body_coo<int64_t, double>);
    m.def("read_body_coo", &read_body_coo<int64_t, std::complex<double>>);

    // 根据预编译标识符 FMM_SCIPY_PRUNE 的状态，绑定更多的 COO 格式读取函数
#ifndef FMM_SCIPY_PRUNE
    m.def("read_body_coo", &read_body_coo<int32_t, float>);
    m.def("read_body_coo", &read_body_coo<int32_t, long double>);
    m.def("read_body_coo", &read_body_coo<int32_t, std::complex<float>>);
    m.def("read_body_coo", &read_body_coo<int32_t, std::complex<long double>>);

    m.def("read_body_coo", &read_body_coo<int64_t, float>);
    m.def("read_body_coo", &read_body_coo<int64_t, long double>);
    m.def("read_body_coo", &read_body_coo<int64_t, std::complex<float>>);
    m.def("read_body_coo", &read_body_coo<int64_t, std::complex<long double>>);
#endif
}
```