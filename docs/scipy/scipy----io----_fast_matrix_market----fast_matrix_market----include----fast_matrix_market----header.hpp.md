# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\fast_matrix_market\include\fast_matrix_market\header.hpp`

```
// 版权声明和许可信息
// 版权所有 (C) 2022 Adam Lugowski。保留所有权利。
// 使用此源代码受 LICENSE.txt 文件中 BSD 2-Clause 许可证的约束。
// SPDX-License-Identifier: BSD-2-Clause

#pragma once

#include <fast_matrix_market/fast_matrix_market.hpp>  // 包含第三方库 fast_matrix_market

#include "chunking.hpp"  // 包含自定义头文件 chunking.hpp

namespace fast_matrix_market {

    /**
     * Matrix Market 文件的标头以此字符串开头。
     */
    const std::string kMatrixMarketBanner = "%%MatrixMarket";

    /**
     * 无效的标头，但某些软件包会发出这种形式而不是双 %% 版本。
     */
    const std::string kMatrixMarketBanner2 = "%MatrixMarket";

    /**
     * 从字符串 s 解析出 ENUM 类型的枚举值，使用映射表 mp 进行匹配。
     * 将 s 转换为小写以进行不区分大小写的匹配。
     */
    template <typename ENUM>
    ENUM parse_enum(const std::string& s, std::map<ENUM, const std::string> mp) {
        std::string lower(s);
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c){ return std::tolower(c); });

        // 遍历映射表 mp，查找匹配的键值对应的枚举值
        for (const auto& [key, value] : mp) {
            if (value == lower) {
                return key;
            }
        }

        // 构造异常信息，指示无效的输入值
        std::string acceptable;
        std::string delim;
        for (const auto& [key, value] : mp) {
            acceptable += delim + std::string(value);
            delim = ", ";
        }
        throw invalid_argument(std::string("Invalid value. Must be one of: ") + acceptable);
    }

    /**
     * 检查字符串 line 是否全为空格。
     * 如果字符串为空，则返回 true。
     */
    inline bool is_line_all_spaces(const std::string& line) {
        if (line.empty()) {
            return true;
        }

        auto end = std::cend(line);

        // 如果行末尾有换行符，则忽略换行符
        if (line[line.size()-1] == '\n') {
            --end;
        }

        // 调用 is_all_spaces 函数检查从行首到有效末尾（排除可能的换行符）是否全为空格
        return is_all_spaces(std::cbegin(line), end);
    }

    /**
     * 去除字符串 line 的末尾 '\r' 字符。
     * 如果字符串不为空且末尾字符是 '\r'，则将其移除。
     */
    inline void strip_trailing_cr(std::string &line) {
        if (!line.empty() && line[line.size() - 1] == '\r') {
            line.resize(line.size() - 1);
        }
    }

    /**
     * 计算需要存储的非零元素数量。
     * 对于一般的矩阵，这将与 header.nnz 相同，但如果 MatrixMarket 文件具有对称性，并选择了 generalize symmetry，
     * 则此函数将计算所需的总数。
     */
    // 根据 Matrix Market 头部信息和读取选项计算存储的非零元素个数
    inline int64_t get_storage_nnz(const matrix_market_header& header, const read_options options) {
        // 如果对象是向量，则直接返回非零元素个数
        if (header.object == vector) {
            return header.nnz;
        }

        // 如果格式是 coordinate
        if (header.format == coordinate) {
            // 如果对称性不是 general 且选项允许泛化对称性，则返回 2 倍的非零元素个数，否则返回原始非零元素个数
            if (header.symmetry != general && options.generalize_symmetry) {
                return 2 * header.nnz;
            } else {
                return header.nnz;
            }
        } else {
            // 计算对角线元素个数和非对角线元素个数
            auto diag_count = header.nrows;
            auto off_diag_count = header.nrows * header.ncols - diag_count;
            auto off_diag_half = off_diag_count / 2;

            // 如果选项允许泛化对称性
            if (options.generalize_symmetry) {
                // 根据对称性类型返回对应的非零元素个数
                switch (header.symmetry) {
                    case skew_symmetric:
                        // 对于 skew-symmetric，对角线上的元素必须为零，返回非对角线元素个数
                        return off_diag_count;
                    default:
                        return header.nnz;
                }
            } else {
                // 如果不允许泛化对称性，则根据对称性类型返回对应的非零元素个数
                switch (header.symmetry) {
                    case symmetric:
                        // 对称矩阵情况，返回非对角线元素一半加上对角线元素个数
                        return off_diag_half + diag_count;
                    case skew_symmetric:
                        // skew-symmetric 情况，返回非对角线元素一半
                        return off_diag_half;
                    case hermitian:
                        // Hermitian 情况，返回非对角线元素一半加上对角线元素个数
                        return off_diag_half + diag_count;
                    case general:
                        // general 情况，返回原始非零元素个数
                        return header.nnz;
                }
            }
        }
        // 如果无法匹配任何情况，则抛出异常
        throw fmm_error("Unknown configuration for get_storage_nnz().");
    }

    /**
     * 解析 Matrix Market 头部的注释行。
     * @param header Matrix Market 头部信息
     * @param line 要解析的行
     * @return 如果行是空行，则返回 true；否则返回 false
     */
    inline bool read_comment(matrix_market_header& header, const std::string& line) {
        // 空行在文件中任意位置都是允许的，并且应该被忽略
        if (is_line_all_spaces(line)) {
            return true;
        }

        // 跳过行首的空白字符
        unsigned int pos = 0;
        while ((pos+1) < line.size() && std::isblank(line[pos])) {
            ++pos;
        }

        // 如果行不以 '%' 开头，则返回 false
        if (line[pos] != '%') {
            return false;
        }

        // 跳过 '%' 符号
        ++pos;

        // 将注释行保存到头部信息中
        header.comment += line.substr(pos) + "\n";
        return true;
    }

    /**
     * 解析一个枚举值，但会生成适合头部解析的错误消息。
     */
    template <typename ENUM>
    ENUM parse_header_enum(const std::string& s, std::map<ENUM, const std::string> mp, int64_t line_num) {
        // 将输入字符串转换为小写以进行不区分大小写的匹配
        std::string lower(s);
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c){ return std::tolower(c); });

        // 遍历枚举映射，寻找匹配项
        for (const auto& [key, value] : mp) {
            if (value == lower) {
                return key;
            }
        }
        // 如果没有找到匹配项，则抛出异常
        throw invalid_mm(std::string("Invalid MatrixMarket header element: ") + s, line_num);
    }
    /**
     * Writes a Matrix Market header to the specified output stream.
     * @param os Output stream to write to
     * @param header Structure containing the Matrix Market header information to write
     * @param options Optional write options controlling the output format
     * @return True if the header was successfully written
     */
    inline bool write_header(std::ostream& os, const matrix_market_header& header, const write_options options = {}) {
        // Write the banner
        os << kMatrixMarketBanner << kSpace;        // Write the Matrix Market banner
        os << object_map.at(header.object) << kSpace;  // Write the object type (matrix or vector)
        os << format_map.at(header.format) << kSpace;  // Write the data format (coordinate, array, etc.)
        os << field_map.at(header.field) << kSpace;    // Write the field type (real, complex, etc.)
        os << symmetry_map.at(header.symmetry) << kNewline;  // Write the symmetry type (general, symmetric, skew-symmetric, etc.)
    
        // Write the comment
        if (!header.comment.empty()) {               // Check if a comment exists in the header
            std::string write_comment = replace_all(header.comment, "\n", "\n%");  // Replace newlines with '%'
            os << "%" << write_comment << kNewline;  // Write the comment prefixed with '%'
        } else if (options.always_comment) {         // If no comment exists but always_comment option is set
            os << "%" << kNewline;                   // Write an empty comment line prefixed with '%'
        }
    
        // Write dimension line
        if (header.object == vector) {               // If the object type is a vector
            os << header.vector_length;              // Write the length of the vector
            if (header.format == coordinate) {       // If format is coordinate, also write nnz (number of non-zeros)
                os << kSpace << header.nnz;
            }
        } else {                                     // If the object type is a matrix
            os << header.nrows << kSpace << header.ncols;  // Write number of rows and columns
            if (header.format == coordinate) {       // If format is coordinate, also write nnz
                os << kSpace << header.nnz;
            }
        }
        os << kNewline;                              // Write newline at the end
    
        return true;                                 // Return true indicating successful write
    }
}



# 这是一个代码块的结束符号，结束了一个代码块的定义或循环体等。
```