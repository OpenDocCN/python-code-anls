# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\fast_matrix_market\include\fast_matrix_market\app\array.hpp`

```
// 版权声明，指定代码版权归 Adam Lugowski 所有，遵循 BSD 2-Clause 许可证，详见 LICENSE.txt 文件
// SPDX-License-Identifier: BSD-2-Clause

#pragma once

#include "../fast_matrix_market.hpp"  // 包含 fast_matrix_market 库头文件

namespace fast_matrix_market {

// MSVC 默认情况下未正确定义 __cplusplus；如果 _MSVC_LANG 存在，则使用它来判断 C++ 标准版本
#if __cplusplus >= 202002L || (defined(_MSVC_LANG) && _MSVC_LANG >= 202002L)
    // 如果支持，使用 C++20 概念（concepts）提升程序员的代码清晰度和错误信息
    // 如果不支持 C++20，以下内容展示 fast_matrix_market 对每个模板类型的预期支持

    /**
     * 可以将矩阵市场文件读取到任何可以调整大小并进行迭代的向量类型中。
     * 典型的例子是 std::vector<>。
     */
    template <typename VEC>
    concept array_read_vector = requires(VEC v) {
        v.empty();
        v.resize(1);  // 假设新元素是默认构造的
        v.begin();
    };

    /**
     * 可以从任何能够报告其大小（用于错误检查）并可以进行迭代的向量类型写入矩阵市场文件。
     * 典型的例子是 std::vector<>。
     */
    template <typename VEC>
    concept array_write_vector = requires(VEC v) {
        v.size();
        v.begin();
    };
#else
    // 如果概念（concepts）不可用，保持所有功能仍然可用
#define array_read_vector typename
#define array_write_vector typename
#endif

    /**
     * 从矩阵市场文件中读取到数组中。
     */
    template <array_read_vector VEC>
    void read_matrix_market_array(std::istream &instream,
                                  matrix_market_header& header,
                                  VEC& values,
                                  storage_order order = row_major,
                                  const read_options& options = {}) {
        read_header(instream, header);  // 读取矩阵市场文件头部信息

        if (!values.empty()) {
            values.resize(0);  // 如果 values 不为空，先清空
        }
        values.resize(header.nrows * header.ncols);  // 调整 values 大小以容纳矩阵元素

        auto handler = dense_adding_parse_handler(values.begin(), order, header.nrows, header.ncols);
        read_matrix_market_body(instream, header, handler, 1, options);  // 读取矩阵市场文件主体内容
    }

    /**
     * 方便的方法，如果用户只关心矩阵维度而不关心头部信息。
     */
    template <array_read_vector VEC, typename DIM>
    void read_matrix_market_array(std::istream &instream,
                                  DIM& nrows, DIM& ncols,
                                  VEC& values,
                                  storage_order order = row_major,
                                  const read_options& options = {}) {
        matrix_market_header header;
        read_matrix_market_array(instream, header, values, order, options);  // 调用上面的函数读取矩阵市场文件

        nrows = header.nrows;  // 将矩阵行数赋给 nrows
        ncols = header.ncols;  // 将矩阵列数赋给 ncols
    }
    /**
     * Convenience method that omits the header requirement if the user only cares about the values
     * (e.g. loading a 1D vector, where the std::vector length already includes the length).
     */
    template <array_read_vector VEC>
    void read_matrix_market_array(std::istream &instream,
                                  VEC& values,
                                  storage_order order = row_major,
                                  const read_options& options = {}) {
        // 声明并初始化一个 matrix_market_header 对象
        matrix_market_header header;
        // 调用另一个重载的 read_matrix_market_array 函数，传递头部对象和其他参数
        read_matrix_market_array(instream, header, values, order, options);
    }
    
    /**
     * Write an array to a Matrix Market file.
     */
    template <array_write_vector VEC>
    void write_matrix_market_array(std::ostream &os,
                                   matrix_market_header header,
                                   const VEC& values,
                                   storage_order order = row_major,
                                   const write_options& options = {}) {
        // 定义 values 迭代器的 value_type
        using VT = typename std::iterator_traits<decltype(values.begin())>::value_type;
    
        // 检查数组长度与矩阵维度是否匹配
        if (header.nrows * header.ncols != (int64_t)values.size()) {
            throw invalid_argument("Array length does not match matrix dimensions.");
        }
    
        // 设置头部的非零元素个数为 values 的大小
        header.nnz = values.size();
    
        // 设置头部对象为矩阵
        header.object = matrix;
        // 如果选项中需要填充头部的字段类型，则获取字段类型
        if (options.fill_header_field_type) {
            header.field = get_field_type((const VT *) nullptr);
        }
        // 设置头部格式为数组
        header.format = array;
        // 设置头部对称性为一般对称
        header.symmetry = general;
    
        // 写入头部信息到输出流 os
        write_header(os, header, options);
    
        // 创建线格式化对象 lf，用于处理行格式化
        line_formatter<int64_t, VT> lf(header, options);
        // 创建数组格式化对象，将数组 values 的内容写入到输出流 os 中
        auto formatter = array_formatter(lf, values.begin(), order, header.nrows, header.ncols);
        write_body(os, formatter, options);
    }
#if __cplusplus < 202002L
// 如果 C++ 版本低于 C++20，执行以下清理操作

// 取消定义 array_read_vector 宏，用于清理可能在前文定义的宏
#undef array_read_vector
// 取消定义 array_write_vector 宏，用于清理可能在前文定义的宏
#undef array_write_vector
#endif
}
// 结束 if 条件判断并结束代码块
```