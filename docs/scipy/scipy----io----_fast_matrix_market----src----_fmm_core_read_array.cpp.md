# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\src\_fmm_core_read_array.cpp`

```
/**
 * 2023年Adam Lugowski版权所有。保留所有权利。
 * 使用此源代码受LICENSE.txt文件中的BSD 2-Clause许可管理。
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "_fmm_core.hpp"

/**
 * 读取 Matrix Market 数据体到一个 numpy 数组中。
 *
 * @param cursor 由 open_read() 打开的游标。
 * @param array NumPy 数组。假定其大小正确，并已清零。
 */
template <typename T>
void read_body_array(read_cursor& cursor, py::array_t<T>& array) {
    // 设置选项以通用化对称性
    cursor.options.generalize_symmetry = true;
    // 获取可变的未检查的数组视图
    auto unchecked = array.mutable_unchecked();
    // 创建解析处理程序，将数据添加到 dense_2d_call_adding_parse_handler 中
    auto handler = fmm::dense_2d_call_adding_parse_handler<decltype(unchecked), int64_t, T>(unchecked);

    // 如果矩阵是数组，则只有 mmread() 会调用此方法。
    // 为了减少最终库大小和编译时间，在此禁用读取坐标矩阵的代码路径。
#ifdef FMM_SCIPY_PRUNE
    fmm::read_matrix_market_body<decltype(handler), fmm::compile_array_only>(cursor.stream(), cursor.header, handler, 1, cursor.options);
#else
    fmm::read_matrix_market_body<decltype(handler), fmm::compile_all>(cursor.stream(), cursor.header, handler, 1, cursor.options);
#endif
    // 关闭游标
    cursor.close();
}

/**
 * 在 Python 模块中初始化 read_body_array 函数的不同类型版本。
 *
 * @param m Python 模块对象的引用。
 */
void init_read_array(py::module_ &m) {
    // 将不同类型的 read_body_array 函数注册到 Python 模块中
    m.def("read_body_array", &read_body_array<int64_t>);
    m.def("read_body_array", &read_body_array<uint64_t>);
    m.def("read_body_array", &read_body_array<double>);
    m.def("read_body_array", &read_body_array<std::complex<double>>);

    // 如果未启用 FMM_SCIPY_PRUNE 宏，注册更多的 read_body_array 函数
#ifndef FMM_SCIPY_PRUNE
    m.def("read_body_array", &read_body_array<float>);
    m.def("read_body_array", &read_body_array<long double>);
    m.def("read_body_array", &read_body_array<std::complex<float>>);
    m.def("read_body_array", &read_body_array<std::complex<long double>>);
#endif
}
```