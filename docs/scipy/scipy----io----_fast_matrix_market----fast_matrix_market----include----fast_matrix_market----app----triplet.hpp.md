# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\fast_matrix_market\include\fast_matrix_market\app\triplet.hpp`

```
/**
 * 版权声明和许可声明，指明此代码的版权和使用许可。
 * Copyright (C) 2022 Adam Lugowski. All rights reserved.
 * Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "../fast_matrix_market.hpp"

namespace fast_matrix_market {

#if __cplusplus >= 202002L || (defined(_MSVC_LANG) && _MSVC_LANG >= 202002L)
    // 如果可用，使用 C++20 概念提高程序员的代码可读性。
    // 这展示了 fast_matrix_market 期望每种模板类型支持的功能。

    /**
     * 可以将矩阵市场文件读入任何可以调整大小并进行迭代的向量类型中。
     * 经典示例是 std::vector<>。
     */
    template <typename VEC>
    concept triplet_read_vector = requires(VEC v) {
        v.resize(1);    // 能够调整大小
        v.begin();      // 能够迭代
    };

    /**
     * 可以从任何可以迭代的向量类型中写入矩阵市场文件。
     * 经典示例是 std::vector<>。
     */
    template <typename VEC>
    concept triplet_write_vector = requires(VEC v) {
        v.cbegin();     // 能够以常量方式迭代开始
        v.cend();       // 能够以常量方式迭代结束
    };
#else
    // 如果概念不可用，仍然使一切正常工作
#define triplet_read_vector typename
#define triplet_write_vector typename
#endif

    /**
     * 泛化三元组的对称性。
     *
     * 不重复对角线元素。
     */
    template <typename IVEC, typename VVEC>
    void generalize_symmetry_triplet(IVEC& rows, IVEC& cols, VVEC& values, const symmetry_type& symmetry) {
        if (symmetry == general) {
            return; // 如果对称性是一般的，直接返回，不做处理
        }

        std::size_t num_diagonal_elements = 0;

        // 统计对角线元素的数量（这些不会被复制）
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (rows[i] == cols[i]) {
                ++num_diagonal_elements;
            }
        }

        // 调整向量的大小
        auto orig_size = rows.size();
        auto new_size = 2 * orig_size - num_diagonal_elements;
        rows.resize(new_size);
        cols.resize(new_size);
        values.resize(new_size);

        // 填充新值
        auto row_iter = rows.begin() + orig_size;
        auto col_iter = cols.begin() + orig_size;
        auto val_iter = values.begin() + orig_size;
        for (std::size_t i = 0; i < orig_size; ++i) {
            if (rows[i] == cols[i]) {
                continue; // 跳过对角线元素
            }

            *row_iter = cols[i];
            *col_iter = rows[i];
            *val_iter = get_symmetric_value<typename VVEC::value_type>(values[i], symmetry);

            ++row_iter; ++col_iter; ++val_iter;
        }
    }

    // 模板定义的一部分，用于指定输入向量类型的约束条件
    template <triplet_read_vector IVEC, triplet_read_vector VVEC, typename T>
    /**
     * 从输入流中读取 Matrix Market 文件的主体部分，转换为三元组（行、列、值的向量）。
     * 如果选项中启用了对称性泛化且应用泛化，则设置 app_generalize 为 true，
     * 并且关闭对称性泛化选项。
     */
    void read_matrix_market_body_triplet(std::istream &instream,
                                         const matrix_market_header& header,
                                         IVEC& rows, IVEC& cols, VVEC& values,
                                         T pattern_value,
                                         read_options options = {}) {
        bool app_generalize = false;
        if (options.generalize_symmetry && options.generalize_symmetry_app) {
            app_generalize = true;
            options.generalize_symmetry = false;
        }

        // 获取存储非零元素个数并分配空间
        auto nnz = get_storage_nnz(header, options);
        rows.resize(nnz);
        cols.resize(nnz);
        values.resize(nnz);

        // 创建处理器对象，用于解析三元组数据
        auto handler = triplet_parse_handler(rows.begin(), cols.begin(), values.begin());
        // 读取 Matrix Market 文件的主体部分
        read_matrix_market_body(instream, header, handler, pattern_value, options);

        // 如果需要应用泛化对称性，调用泛化函数
        if (app_generalize) {
            generalize_symmetry_triplet(rows, cols, values, header.symmetry);
        }
    }

    /**
     * 从输入流中读取 Matrix Market 文件的头部信息，并调用函数读取其主体部分。
     */
    template <triplet_read_vector IVEC, triplet_read_vector VVEC>
    void read_matrix_market_triplet(std::istream &instream,
                                    matrix_market_header& header,
                                    IVEC& rows, IVEC& cols, VVEC& values,
                                    const read_options& options = {}) {
        // 读取 Matrix Market 文件的头部信息
        read_header(instream, header);

        // 推断值类型并调用具体的读取函数，将文件内容解析为三元组格式
        using VT = typename std::iterator_traits<decltype(values.begin())>::value_type;
        read_matrix_market_body_triplet(instream, header, rows, cols, values, pattern_default_value((const VT*)nullptr), options);
    }

    /**
     * 当用户只关心 Matrix Market 文件的维度信息时，提供的便捷方法，省略头部信息的读取。
     */
    template <triplet_read_vector IVEC, triplet_read_vector VVEC, typename DIM>
    void read_matrix_market_triplet(std::istream &instream,
                                    DIM& nrows, DIM& ncols,
                                    IVEC& rows, IVEC& cols, VVEC& values,
                                    const read_options& options = {}) {
        // 创建 Matrix Market 文件头部对象并调用带头部的读取函数
        matrix_market_header header;
        read_matrix_market_triplet(instream, header, rows, cols, values, options);
        // 将读取的行数和列数赋值给输出参数
        nrows = header.nrows;
        ncols = header.ncols;
    }
    void write_matrix_market_triplet(std::ostream &os,
                                     matrix_market_header header,
                                     const IVEC& rows,
                                     const IVEC& cols,
                                     const VVEC& values,
                                     const write_options& options = {}) {
        // 确定行和值的迭代器类型
        using IT = typename std::iterator_traits<decltype(rows.begin())>::value_type;
        using VT = typename std::iterator_traits<decltype(values.begin())>::value_type;
    
        // 设置头部的非零元素数目为行数
        header.nnz = rows.size();
    
        // 设置头部对象为矩阵
        header.object = matrix;
        // 如果非零元素数大于0且值迭代器范围为空，则设置字段为模式
        if (header.nnz > 0 && (values.cbegin() == values.cend())) {
            header.field = pattern;
        } else if (header.field != pattern && options.fill_header_field_type) {
            // 如果字段不为模式且选项启用填充头部字段类型，则根据值的类型获取字段类型
            header.field = get_field_type((const VT *) nullptr);
        }
        // 设置格式为坐标格式
        header.format = coordinate;
    
        // 写入头部信息到输出流
        write_header(os, header, options);
    
        // 创建行格式化器
        line_formatter<IT, VT> lf(header, options);
        // 创建三元组格式化器，根据字段类型确定值迭代器的结束位置
        auto formatter = triplet_formatter(lf,
                                           rows.cbegin(), rows.cend(),
                                           cols.cbegin(), cols.cend(),
                                           values.cbegin(), header.field == pattern ? values.cbegin() : values.cend());
        // 写入主体部分到输出流
        write_body(os, formatter, options);
    }
    
    /**
     * Write CSC/CSR to a Matrix Market file.
     */
    template <triplet_write_vector IVEC, triplet_write_vector VVEC>
    void write_matrix_market_csc(std::ostream &os,
                                 matrix_market_header header,
                                 const IVEC& indptr,
                                 const IVEC& indices,
                                 const VVEC& values,
                                 bool is_csr,
                                 const write_options& options = {}) {
        // 确定指针和值的迭代器类型
        using IT = typename std::iterator_traits<decltype(indptr.begin())>::value_type;
        using VT = typename std::iterator_traits<decltype(values.begin())>::value_type;
    
        // 设置头部的非零元素数目为索引数
        header.nnz = indices.size();
    
        // 设置头部对象为矩阵
        header.object = matrix;
        // 如果非零元素数大于0且值迭代器范围为空，则设置字段为模式
        if (header.nnz > 0 && (values.cbegin() == values.cend())) {
            header.field = pattern;
        } else if (header.field != pattern && options.fill_header_field_type) {
            // 如果字段不为模式且选项启用填充头部字段类型，则根据值的类型获取字段类型
            header.field = get_field_type((const VT *) nullptr);
        }
        // 设置格式为坐标格式
        header.format = coordinate;
    
        // 写入头部信息到输出流
        write_header(os, header, options);
    
        // 创建行格式化器
        line_formatter<IT, VT> lf(header, options);
        // 创建CSC格式化器，根据字段类型确定值迭代器的结束位置
        auto formatter = csc_formatter(lf,
                                       indptr.cbegin(), indptr.cend() - 1,
                                       indices.cbegin(), indices.cend(),
                                       values.cbegin(), header.field == pattern ? values.cbegin() : values.cend(),
                                       is_csr);
        // 写入主体部分到输出流
        write_body(os, formatter, options);
    }
#if __cplusplus < 202002L
    // 如果 C++ 标准版本低于 C++20

    // 清理我们自己的定义
#undef triplet_read_vector
#undef triplet_write_vector
#endif
}
```