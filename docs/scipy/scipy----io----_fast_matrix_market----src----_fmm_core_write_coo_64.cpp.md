# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\src\_fmm_core_write_coo_64.cpp`

```
// 定义函数 init_write_coo_64，接受一个名为 m 的 py::module_ 对象作为参数
void init_write_coo_64(py::module_ &m) {
    // 将模板函数 write_body_coo 实例化为不同参数类型的函数，并绑定到 Python 模块 m 上
    m.def("write_body_coo", &write_body_coo<int64_t, int32_t>);
    m.def("write_body_coo", &write_body_coo<int64_t, uint32_t>);
    m.def("write_body_coo", &write_body_coo<int64_t, int64_t>);
    m.def("write_body_coo", &write_body_coo<int64_t, uint64_t>);
    m.def("write_body_coo", &write_body_coo<int64_t, float>);
    m.def("write_body_coo", &write_body_coo<int64_t, double>);
    m.def("write_body_coo", &write_body_coo<int64_t, long double>);
    m.def("write_body_coo", &write_body_coo<int64_t, std::complex<float>>);
    m.def("write_body_coo", &write_body_coo<int64_t, std::complex<double>>);
    m.def("write_body_coo", &write_body_coo<int64_t, std::complex<long double>>);
}
```