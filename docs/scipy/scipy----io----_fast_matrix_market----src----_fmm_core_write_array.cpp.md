# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\src\_fmm_core_write_array.cpp`

```
/**
 * Write numpy array to MatrixMarket file
 */
// 定义模板函数，用于将 numpy 数组写入 MatrixMarket 文件
template <typename T>
void write_body_array(write_cursor& cursor, py::array_t<T>& array) {
    // 检查数组维度是否为 2
    if (array.ndim() != 2) {
        throw std::invalid_argument("Only 2D arrays supported.");
    }

    // 设置行数和列数为数组的形状
    cursor.header.nrows = array.shape(0);
    cursor.header.ncols = array.shape(1);

    // 设置头部信息：对象类型为 matrix，字段类型为数组中元素的数据类型，格式为数组
    cursor.header.object = fmm::matrix;
    cursor.header.field = fmm::get_field_type((const T*)nullptr);
    cursor.header.format = fmm::array;

    // 调用函数写入头部信息到流
    fmm::write_header(cursor.stream(), cursor.header, cursor.options);

    // 获取未检查的数组数据视图
    auto unchecked = array.unchecked();

    // 创建行格式化器
    fmm::line_formatter<int64_t, T> lf(cursor.header, cursor.options);

    // 创建二维密集调用格式化器
    auto formatter = fmm::dense_2d_call_formatter<decltype(lf), decltype(unchecked), int64_t>(
        lf, unchecked, cursor.header.nrows, cursor.header.ncols);

    // 调用函数写入数组主体部分到流
    fmm::write_body(cursor.stream(), formatter, cursor.options);

    // 关闭写入游标
    cursor.close();
}

// 初始化 Python 模块，定义多个类型的 write_body_array 函数
void init_write_array(py::module_ &m) {
    m.def("write_body_array", &write_body_array<int32_t>);
    m.def("write_body_array", &write_body_array<uint32_t>);
    m.def("write_body_array", &write_body_array<int64_t>);
    m.def("write_body_array", &write_body_array<uint64_t>);
    m.def("write_body_array", &write_body_array<float>);
    m.def("write_body_array", &write_body_array<double>);
    m.def("write_body_array", &write_body_array<long double>);
    m.def("write_body_array", &write_body_array<std::complex<float>>);
    m.def("write_body_array", &write_body_array<std::complex<double>>);
    m.def("write_body_array", &write_body_array<std::complex<long double>>);
}
```