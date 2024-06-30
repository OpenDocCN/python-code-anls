# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\fast_matrix_market\include\fast_matrix_market\app\doublet.hpp`

```
// 版权声明，指明代码归 Adam Lugowski 所有，受 BSD 2-Clause 许可证保护，许可文件位于 LICENSE.txt。
// SPDX-License-Identifier: BSD-2-Clause

// 预处理指令，确保此头文件只被包含一次
#pragma once

// 包含上级目录下的 fast_matrix_market.hpp 文件
#include "../fast_matrix_market.hpp"

// fast_matrix_market 命名空间
namespace fast_matrix_market {

    // 如果编译器支持 C++20 概念或者是 MSVC 且支持 C++20，则定义以下概念
#if __cplusplus >= 202002L || (defined(_MSVC_LANG) && _MSVC_LANG >= 202002L)
    // 如果可用，使用 C++20 概念提升程序员的清晰度。
    // 这展示了 fast_matrix_market 期望每个模板类型支持的功能。

    /**
     * 可以将 Matrix Market 文件读取到任何可以调整大小和迭代的向量类型中。
     * 典型示例是 std::vector<>。
     */
    template <typename VEC>
    concept doublet_read_vector = requires(VEC v) {
        v.resize(1);   // 能够调整大小
        v.begin();     // 能够获取起始迭代器
    };

    /**
     * 可以从任何可以迭代的向量类型写入 Matrix Market 文件。
     * 典型示例是 std::vector<>。
     */
    template <typename VEC>
    concept doublet_write_vector = requires(VEC v) {
        v.cbegin();    // 能够获取常量起始迭代器
        v.cend();      // 能够获取常量结束迭代器
    };
#else
    // 如果概念不可用，则定义宏替代
#define doublet_read_vector typename
#define doublet_write_vector typename
#endif

    /**
     * 将 Matrix Market 向量文件读取到双重稀疏向量中（即索引向量和值向量）。
     *
     * 任何类似向量的 Matrix Market 文件都可以工作：
     *  - 对象=向量文件，无论是密集还是稀疏的
     *  - 对象=矩阵文件，只要 nrows=1 或 ncols=1
     */
    template <doublet_read_vector IVEC, doublet_read_vector VVEC>
    void read_matrix_market_doublet(std::istream &instream,
                                    matrix_market_header& header,
                                    IVEC& indices, VVEC& values,
                                    const read_options& options = {}) {
        // 获取值向量的值类型
        using VT = typename std::iterator_traits<decltype(values.begin())>::value_type;

        // 读取头部信息
        read_header(instream, header);

        // 调整索引和值向量的大小以容纳数据
        indices.resize(header.nnz);
        values.resize(get_storage_nnz(header, options));

        // 创建双重解析处理器，并读取 Matrix Market 数据主体
        auto handler = doublet_parse_handler(indices.begin(), values.begin());
        read_matrix_market_body(instream, header, handler, pattern_default_value((const VT*)nullptr), options);
    }

    /**
     * 如果用户只关心维度，方便的方法，省略了需要头部信息的要求。
     */
    template <doublet_read_vector IVEC, doublet_read_vector VVEC, typename DIM>
    void read_matrix_market_doublet(std::istream &instream,
                                    DIM& length,
                                    IVEC& indices, VVEC& values,
                                    const read_options& options = {}) {
        matrix_market_header header;
        read_matrix_market_doublet(instream, header, indices, values, options);
        length = header.vector_length;
    }

    /**
     * 将双重向量写入 Matrix Market 文件。
     */
    template <doublet_write_vector IVEC, doublet_write_vector VVEC>
    // 将稀疏矩阵以 Matrix Market 格式写入输出流
    void write_matrix_market_doublet(std::ostream &os,
                                     // Matrix Market 文件头信息
                                     matrix_market_header header,
                                     // 索引向量，存储非零元素的行和列索引
                                     const IVEC& indices,
                                     // 值向量，存储非零元素的值
                                     const VVEC& values,
                                     // 写入选项，默认为空选项
                                     const write_options& options = {}) {
        // 定义索引向量的值类型
        using IT = typename std::iterator_traits<decltype(indices.begin())>::value_type;
        // 定义值向量的值类型
        using VT = typename std::iterator_traits<decltype(values.begin())>::value_type;
    
        // 更新文件头中的非零元素数量
        header.nnz = indices.size();
    
        // 设置文件头中的对象类型为向量
        header.object = vector;
    
        // 如果非零元素数量大于0且值向量为空，则将文件头中的数据类型设置为模式（只有结构信息）
        if (header.nnz > 0 && (values.cbegin() == values.cend())) {
            header.field = pattern;
        } 
        // 如果文件头中的数据类型不是模式且选项中指定了填充文件头数据类型的选项，则根据值向量的类型推断数据类型
        else if (header.field != pattern && options.fill_header_field_type) {
            header.field = get_field_type((const VT *) nullptr);
        }
        
        // 设置文件头中的格式为坐标格式
        header.format = coordinate;
    
        // 写入文件头信息到输出流
        write_header(os, header, options);
    
        // 使用向量行格式化器创建格式化器
        vector_line_formatter<IT, VT> lf(header, options);
        // 创建三元组格式化器，处理行索引、列索引和值
        auto formatter = triplet_formatter(lf,
                                           indices.cbegin(), indices.cend(),
                                           indices.cbegin(), indices.cend(),
                                           // 如果数据类型为模式，则值向量只使用起始迭代器
                                           values.cbegin(), header.field == pattern ? values.cbegin() : values.cend());
        // 写入主体内容到输出流
        write_body(os, formatter, options);
    }
#if __cplusplus < 202002L
    // 如果 C++ 版本低于 C++20，执行以下清理操作

    // 取消定义 doublet_read_vector 宏
#undef doublet_read_vector
    // 取消定义 doublet_write_vector 宏
#undef doublet_write_vector
#endif
// 结束条件编译指令
}
```