# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\src\_fmm_core_write_coo_32.cpp`

```
// 版权声明和许可信息，指明此代码的版权和使用条款
// Copyright (C) 2023 Adam Lugowski. All rights reserved.
// Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
// SPDX-License-Identifier: BSD-2-Clause

// 包含头文件 "_fmm_core.hpp"，该文件可能包含了一些核心功能的定义和声明
#include "_fmm_core.hpp"

// 初始化函数，用于将 COO 格式的数据写入文件
void init_write_coo_32(py::module_ &m) {
    // 定义并注册模板函数 "write_body_coo" 的具体实例化，其中 int32_t 是第一个模板参数
    // 第二个模板参数依次为不同的数据类型，包括 int32_t、uint32_t、int64_t、uint64_t、float、double、long double、
    // std::complex<float>、std::complex<double>、std::complex<long double>
    m.def("write_body_coo", &write_body_coo<int32_t, int32_t>);
    m.def("write_body_coo", &write_body_coo<int32_t, uint32_t>);
    m.def("write_body_coo", &write_body_coo<int32_t, int64_t>);
    m.def("write_body_coo", &write_body_coo<int32_t, uint64_t>);
    m.def("write_body_coo", &write_body_coo<int32_t, float>);
    m.def("write_body_coo", &write_body_coo<int32_t, double>);
    m.def("write_body_coo", &write_body_coo<int32_t, long double>);
    m.def("write_body_coo", &write_body_coo<int32_t, std::complex<float>>);
    m.def("write_body_coo", &write_body_coo<int32_t, std::complex<double>>);
    m.def("write_body_coo", &write_body_coo<int32_t, std::complex<long double>>);
}
```