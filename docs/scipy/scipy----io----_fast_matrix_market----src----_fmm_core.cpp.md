# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\src\_fmm_core.cpp`

```
// 版权信息声明，声明该源代码的版权为2022-2023年Adam Lugowski所有，受LICENSE.txt文件中的BSD 2-Clause许可证控制
// SPDX-License-Identifier: BSD-2-Clause

// 引入_fmm_core.hpp头文件，可能包含了fast_matrix_market命名空间和相关函数和结构的定义
#include "_fmm_core.hpp"

// 引入所需的标准库头文件
#include <fast_matrix_market/types.hpp>
#include <cstdint>

// fast_matrix_market命名空间定义
namespace fast_matrix_market {

    // 根据输入的type指针获取字段类型，这个函数用于从uint32_t类型指针获取unsigned_integer类型
    field_type get_field_type([[maybe_unused]] const uint32_t* type) {
        return unsigned_integer;
    }

    // 根据输入的type指针获取字段类型，这个函数用于从uint64_t类型指针获取unsigned_integer类型
    field_type get_field_type([[maybe_unused]] const uint64_t* type) {
        return unsigned_integer;
    }

} // namespace fast_matrix_market

// 引入fast_matrix_market库的头文件
#include <fast_matrix_market/fast_matrix_market.hpp>

////////////////////////////////////////////////
//// Header methods
////////////////////////////////////////////////

// 获取矩阵市场头部的形状（行数和列数），并以tuple形式返回
std::tuple<int64_t, int64_t> get_header_shape(const fmm::matrix_market_header& header) {
    return std::make_tuple(header.nrows, header.ncols);
}

// 设置矩阵市场头部的形状（行数和列数）
void set_header_shape(fmm::matrix_market_header& header, const std::tuple<int64_t, int64_t>& shape) {
    header.nrows = std::get<0>(shape);
    header.ncols = std::get<1>(shape);
}

// 获取矩阵市场头部的对象类型的字符串表示
std::string get_header_object(const fmm::matrix_market_header& header) {
    return fmm::object_map.at(header.object);
}

// 获取矩阵市场头部的格式类型的字符串表示
std::string get_header_format(const fmm::matrix_market_header& header) {
    return fmm::format_map.at(header.format);
}

// 获取矩阵市场头部的字段类型的字符串表示
std::string get_header_field(const fmm::matrix_market_header& header) {
    return fmm::field_map.at(header.field);
}

// 获取矩阵市场头部的对称性类型的字符串表示
std::string get_header_symmetry(const fmm::matrix_market_header& header) {
    return fmm::symmetry_map.at(header.symmetry);
}

// 设置矩阵市场头部的对象类型
void set_header_object(fmm::matrix_market_header& header, const std::string& value) {
    header.object = fmm::parse_enum<fmm::object_type>(value, fmm::object_map);
}

// 设置矩阵市场头部的格式类型
void set_header_format(fmm::matrix_market_header& header, const std::string& value) {
    header.format = fmm::parse_enum<fmm::format_type>(value, fmm::format_map);
}

// 设置矩阵市场头部的字段类型
void set_header_field(fmm::matrix_market_header& header, const std::string& value) {
    header.field = fmm::parse_enum<fmm::field_type>(value, fmm::field_map);
}

// 设置矩阵市场头部的对称性类型
void set_header_symmetry(fmm::matrix_market_header& header, const std::string& value) {
    header.symmetry = fmm::parse_enum<fmm::symmetry_type>(value, fmm::symmetry_map);
}

// 创建并返回一个矩阵市场头部，包括形状、非零元素数目、注释、对象类型、格式、字段类型和对称性类型
fmm::matrix_market_header create_header(const std::tuple<int64_t, int64_t>& shape, int64_t nnz,
                                        const std::string& comment,
                                        const std::string& object, const std::string& format,
                                        const std::string& field, const std::string& symmetry) {
    // 创建一个新的矩阵市场头部对象
    fmm::matrix_market_header header{};
    // 设置矩阵市场头部的形状
    set_header_shape(header, shape);
    // 设置矩阵市场头部的非零元素数目
    header.nnz = nnz;
    // 设置矩阵市场头部的注释
    header.comment = comment;
    // 设置矩阵市场头部的对象类型
    set_header_object(header, object);
    // 设置矩阵市场头部的格式类型
    set_header_format(header, format);
    // 设置矩阵市场头部的字段类型
    set_header_field(header, field);
    // 设置矩阵市场头部的对称性类型
    set_header_symmetry(header, symmetry);
    // 返回创建的矩阵市场头部对象
    return header;
}
#ifndef FMM_SCIPY_PRUNE
// 将 C++ 中的矩阵市场头部信息转换为 Python 字典
py::dict header_to_dict(fmm::matrix_market_header& header) {
    py::dict dict;
    dict["shape"] = py::make_tuple(header.nrows, header.ncols);  // 将行数和列数作为元组存入字典
    dict["nnz"] = header.nnz;  // 将非零元素个数存入字典
    dict["comment"] = header.comment;  // 将注释存入字典
    dict["object"] = get_header_object(header);  // 调用函数获取对象类型并存入字典
    dict["format"] = get_header_format(header);  // 调用函数获取格式类型并存入字典
    dict["field"] = get_header_field(header);    // 调用函数获取数据类型并存入字典
    dict["symmetry"] = get_header_symmetry(header);  // 调用函数获取对称性信息并存入字典

    return dict;  // 返回构建好的字典
}

// 将 C++ 中的矩阵市场头部信息转换为字符串表示形式
std::string header_repr(const fmm::matrix_market_header& header) {
    std::ostringstream oss;
    oss << "header(";  // 开始构建字符串
    oss << "shape=(" << header.nrows << ", " << header.ncols << "), ";  // 添加形状信息
    oss << "nnz=" << header.nnz << ", ";  // 添加非零元素个数信息
    oss << "comment=\"" << header.comment << "\", ";  // 添加注释信息
    oss << "object=\"" << get_header_object(header) << "\", ";  // 添加对象类型信息
    oss << "format=\"" << get_header_format(header) << "\", ";  // 添加格式类型信息
    oss << "field=\"" << get_header_field(header) << "\", ";    // 添加数据类型信息
    oss << "symmetry=\"" << get_header_symmetry(header) << "\"";  // 添加对称性信息
    oss << ")";  // 结束构建字符串
    return oss.str();  // 返回构建好的字符串
}
#endif

////////////////////////////////////////////////
//// Read cursor - open files/streams for reading
////////////////////////////////////////////////

// 打开读取游标，用于打开文件/流进行读取操作
void open_read_rest(read_cursor& cursor) {
    // This is done later in Python to match SciPy behavior
    cursor.options.generalize_symmetry = false;  // 设置选项，Python 中稍后处理以匹配 SciPy 的行为

    // read header
    fmm::read_header(cursor.stream(), cursor.header);  // 读取头部信息
}

// 打开读取文件，返回读取游标
read_cursor open_read_file(const std::string& filename, int num_threads) {
    read_cursor cursor(filename);  // 创建读取游标，指定文件名
    // 设置选项
    cursor.options.num_threads = num_threads;  // 设置线程数选项
    cursor.options.float_out_of_range_behavior = fmm::BestMatch;  // 设置浮点数超出范围时的行为为最佳匹配

    open_read_rest(cursor);  // 执行打开读取游标的其余操作
    return cursor;  // 返回读取游标
}

// 打开读取流，返回读取游标
read_cursor open_read_stream(std::shared_ptr<pystream::istream>& external, int num_threads) {
    read_cursor cursor(external);  // 创建读取游标，使用外部流
    // 设置选项
    cursor.options.num_threads = num_threads;  // 设置线程数选项
    cursor.options.float_out_of_range_behavior = fmm::BestMatch;  // 设置浮点数超出范围时的行为为最佳匹配

    open_read_rest(cursor);  // 执行打开读取游标的其余操作
    return cursor;  // 返回读取游标
}


////////////////////////////////////////////////
//// Write cursor - open files/streams writing reading
////////////////////////////////////////////////

// 打开写入文件，返回写入游标
write_cursor open_write_file(const std::string& filename, const fmm::matrix_market_header& header,
                             int num_threads, int precision) {
    write_cursor cursor(filename);  // 创建写入游标，指定文件名
    // 设置选项
    cursor.options.num_threads = num_threads;  // 设置线程数选项
    cursor.options.precision = precision;  // 设置精度选项
    cursor.options.always_comment = true; // scipy.io._mmio 总是写入注释行，即使注释为空。

    cursor.header = header;  // 设置头部信息
    return cursor;  // 返回写入游标
}

// 打开写入流，返回写入游标
write_cursor open_write_stream(std::shared_ptr<pystream::ostream>& stream, fmm::matrix_market_header& header,
                               int num_threads, int precision) {
    write_cursor cursor(stream);  // 创建写入游标，使用流
    // 设置选项
    cursor.options.num_threads = num_threads;  // 设置线程数选项
    cursor.options.precision = precision;  // 设置精度选项
    # 设置游标对象的精度选项为给定的精度值
    cursor.options.precision = precision;
    # 设置游标对象的总是注释选项为真，这样即使注释为空，scipy.io._mmio 也会写入一个注释行。
    cursor.options.always_comment = true; // scipy.io._mmio always writes a comment line, even if comment is empty.
    # 将游标对象的头部信息设置为给定的头部内容
    cursor.header = header;
    # 返回更新后的游标对象
    return cursor;
////////////////////////////////////////////////
//// pybind11 module definition
//// Define the _fmm_core module here, it is used by __init__.py
////////////////////////////////////////////////

// 使用 pybind11 定义 C++ 扩展模块 _fmm_core，用于 Python 的初始化文件 __init__.py 中调用

PYBIND11_MODULE(_fmm_core, m) {
    // 设置模块的文档字符串
    m.doc() = R"pbdoc(
        fast_matrix_market
    )pbdoc";

    // 异常转换注册函数，用于将 C++ 异常映射到相应的 Python 异常类型
    py::register_local_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const fmm::out_of_range &e) {
            PyErr_SetString(PyExc_OverflowError, e.what());
        } catch (const fmm::support_not_selected& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        } catch (const fmm::fmm_error &e) {
            // 其他所有异常映射为 ValueError
            PyErr_SetString(PyExc_ValueError, e.what());
        }
    });

    // 定义 matrix_market_header 类的 Python 绑定
    py::class_<fmm::matrix_market_header>(m, "header", py::module_local())
#ifndef FMM_SCIPY_PRUNE
    .def(py::init<>())  // 默认构造函数
    .def(py::init<int64_t, int64_t>())  // 初始化给定 nrows 和 ncols 的构造函数
    .def(py::init([](std::tuple<int64_t, int64_t> shape) { return fmm::matrix_market_header{std::get<0>(shape), std::get<1>(shape)}; }))  // 接受形状元组的构造函数
#endif
    .def(py::init(&create_header), py::arg("shape")=std::make_tuple(0, 0), "nnz"_a=0, "comment"_a=std::string(), "object"_a="matrix", "format"_a="coordinate", "field"_a="real", "symmetry"_a="general")  // 使用 create_header 函数的构造函数
    .def_readwrite("nrows", &fmm::matrix_market_header::nrows)  // nrows 属性的读写绑定
    .def_readwrite("ncols", &fmm::matrix_market_header::ncols)  // ncols 属性的读写绑定
    .def_property("shape", &get_header_shape, &set_header_shape)  // shape 属性的读写绑定
    .def_readwrite("nnz", &fmm::matrix_market_header::nnz)  // nnz 属性的读写绑定
    .def_readwrite("comment", &fmm::matrix_market_header::comment)  // comment 属性的读写绑定
    .def_property("object", &get_header_object, &set_header_object)  // object 属性的读写绑定
    .def_property("format", &get_header_format, &set_header_format)  // format 属性的读写绑定
    .def_property("field", &get_header_field, &set_header_field)  // field 属性的读写绑定
    .def_property("symmetry", &get_header_symmetry, &set_header_symmetry)  // symmetry 属性的读写绑定
#ifndef FMM_SCIPY_PRUNE
    .def("to_dict", &header_to_dict, R"pbdoc(
        Return the values in the header as a dict.
    )pbdoc")  // 将头部信息转换为字典的方法绑定
    .def("__repr__", [](const fmm::matrix_market_header& header) { return header_repr(header); })  // 自定义头部对象的字符串表示形式
#endif
    ;

#ifndef FMM_SCIPY_PRUNE
    // 定义写入操作的函数 write_header_only
    m.def("write_header_only", &write_header_only);
#endif

    ///////////////////////////////
    // Read methods
    // 定义 _read_cursor 类的 Python 绑定
    py::class_<read_cursor>(m, "_read_cursor", py::module_local())
    .def_readonly("header", &read_cursor::header)  // header 属性的只读绑定
    .def("close", &read_cursor::close);  // close 方法的绑定

    // 定义 open_read_file 函数的 Python 绑定
    m.def("open_read_file", &open_read_file);
    // 定义 open_read_stream 函数的 Python 绑定
    m.def("open_read_stream", &open_read_stream);

    // 初始化 read_array 模块
    init_read_array(m);
    // 初始化 read_coo 模块
    init_read_coo(m);

    ///////////////////////////////
    // Write methods
    // 定义 _write_cursor 类的 Python 绑定
    py::class_<write_cursor>(m, "_write_cursor", py::module_local())
#ifndef FMM_SCIPY_PRUNE
    // 定义一个名为 "header" 的读写属性，它与 write_cursor 类中的 header 成员变量相连
    .def_readwrite("header", &write_cursor::header)
#endif
    ;

// 如果定义了 `FMM_SCIPY_PRUNE` 宏，则跳过当前语句；否则，添加一个空语句，用于代码结构。


m.def("open_write_file", &open_write_file);
m.def("open_write_stream", &open_write_stream);

// 将 `open_write_file` 和 `open_write_stream` 函数绑定到 Python 模块 `m` 中，以供 Python 脚本调用。


init_write_array(m);
init_write_coo_32(m);
init_write_coo_64(m);

// 初始化 `m` 模块中的数组写入相关函数。


#ifndef FMM_SCIPY_PRUNE
init_write_csc_32(m);
init_write_csc_64(m);
#endif

// 如果未定义 `FMM_SCIPY_PRUNE` 宏，则初始化 `m` 模块中的 `csc` 格式写入相关函数。


// Module version
#ifdef VERSION_INFO
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
m.attr("__version__") = "dev";
#endif

// 根据是否定义了 `VERSION_INFO` 宏，设置 Python 模块 `m` 的 `__version__` 属性。
```