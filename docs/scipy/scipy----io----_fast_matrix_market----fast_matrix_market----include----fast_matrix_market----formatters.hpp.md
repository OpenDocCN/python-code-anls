# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\fast_matrix_market\include\fast_matrix_market\formatters.hpp`

```
// 版权声明及许可证信息
// 2022年Adam Lugowski版权所有。本源代码使用BSD 2条款许可证管理，详见LICENSE.txt文件。
// SPDX-License-Identifier: BSD-2-Clause

// 预处理命令，确保本文件仅被编译一次
#pragma once

// 包含C++标准库中的头文件
#include <algorithm>
#include <utility>

// 包含自定义头文件"fast_matrix_market.hpp"
#include "fast_matrix_market.hpp"

// fast_matrix_market命名空间
namespace fast_matrix_market {

    /**
     * 格式化单行数据（矩阵版本）的类模板。
     */
    template <typename IT, typename VT>
    class line_formatter {
    public:
        // 构造函数，接收matrix_market_header和write_options对象的引用
        line_formatter(const matrix_market_header &header, const write_options &options) : header(header),
                                                                                           options(options) {}

        // 根据header中的格式要求，格式化坐标矩阵行数据
        std::string coord_matrix(const IT& row, const IT& col, const VT& val) {
            // 如果格式为array，则调用array_matrix方法处理
            if (header.format == array) {
                return array_matrix(row, col, val);
            }

            // 否则，按照坐标格式输出行数据
            std::string line{};
            line += int_to_string(row + 1);   // 将行号加1转换为字符串并添加到行数据中
            line += kSpace;                   // 添加一个空格
            line += int_to_string(col + 1);   // 将列号加1转换为字符串并添加到行数据中

            // 如果header中的字段不是pattern，则添加值的字符串表示
            if (header.field != pattern) {
                line += kSpace;  // 添加一个空格
                line += value_to_string(val, options.precision);  // 根据精度将值转换为字符串并添加到行数据中
            }
            line += kNewline;  // 添加换行符

            return line;  // 返回格式化好的行数据
        }

        // 根据header中的格式要求，格式化坐标矩阵模式行数据
        std::string coord_matrix_pattern(const IT& row, const IT& col) {
            // 直接输出行号和列号，形成行数据
            std::string line{};
            line += int_to_string(row + 1);   // 将行号加1转换为字符串并添加到行数据中
            line += kSpace;                   // 添加一个空格
            line += int_to_string(col + 1);   // 将列号加1转换为字符串并添加到行数据中
            line += kNewline;                 // 添加换行符

            return line;  // 返回格式化好的行数据
        }

        // 根据header中的格式要求，格式化array格式的矩阵行数据
        std::string array_matrix(const IT& row, const IT& col, const VT& val) {
            // 如果矩阵格式为非general对称矩阵
            if (header.symmetry != general) {
                // 如果当前行小于列，跳过上三角部分，返回空字符串
                if (row < col) {
                    // 省略上三角部分
                    return {};
                }
                // 如果是skew_symmetric对称矩阵且当前行等于列，跳过对角线部分，返回空字符串
                if (header.symmetry == skew_symmetric && row == col) {
                    // 省略skew-symmetric对角线部分
                    return {};
                }
            }

            // 根据指定的精度将值转换为字符串，并添加换行符，返回结果
            std::string ret = value_to_string(val, options.precision);
            ret += kNewline;
            return ret;  // 返回格式化好的行数据
        }
    protected:
        const matrix_market_header& header;  // 引用传递的matrix_market_header对象
        const write_options& options;        // 引用传递的write_options对象
    };

    /**
     * 格式化单行数据（向量版本）的类模板。
     */
    template <typename IT, typename VT>
    class vector_line_formatter {
        // 省略...
    /**
     * 用于格式化行、列、值向量的类。
     *
     * 当值的范围为空（即 val_begin == val_end）时，可以完全省略写入值。对于模式矩阵非常有用。
     *
     * @tparam LF 格式化器类型
     * @tparam A_ITER 行迭代器类型
     * @tparam B_ITER 列迭代器类型
     * @tparam C_ITER 必须是有效的迭代器，但如果 begin == end 则不写入值。
     *                （对于模式矩阵非常有用）
     */
    template<typename LF, typename A_ITER, typename B_ITER, typename C_ITER>
    class triplet_formatter {
    /**
     * 构造函数，初始化三元组格式化器。
     * @param lf 行格式化器
     * @param row_begin 行迭代器的起始位置
     * @param row_end 行迭代器的结束位置
     * @param col_begin 列迭代器的起始位置
     * @param col_end 列迭代器的结束位置
     * @param val_begin 值迭代器的起始位置
     * @param val_end 值迭代器的结束位置
     * @throws invalid_argument 如果行、列和值的范围长度不一致，则抛出异常
     */
    explicit triplet_formatter(LF lf,
                               const A_ITER row_begin, const A_ITER row_end,
                               const B_ITER col_begin, const B_ITER col_end,
                               const C_ITER val_begin, const C_ITER val_end) :
                               line_formatter(lf),
                               row_iter(row_begin), row_end(row_end),
                               col_iter(col_begin),
                               val_iter(val_begin), val_end(val_end) {
        if (row_end - row_begin != col_end - col_begin ||
                (row_end - row_begin != val_end - val_begin && val_end != val_begin)) {
            throw invalid_argument("Row, column, and value ranges must have equal length.");
        }
    }

    /**
     * 检查是否还有下一个元素。
     * @return 如果仍有元素返回 true，否则返回 false
     */
    [[nodiscard]] bool has_next() const {
        return row_iter != row_end;
    }

    /**
     * 内部类，表示三元组的一个块。
     */
    class chunk {
    public:
        /**
         * 构造函数，初始化块。
         * @param lf 行格式化器
         * @param row_begin 行迭代器的起始位置
         * @param row_end 行迭代器的结束位置
         * @param col_begin 列迭代器的起始位置
         * @param val_begin 值迭代器的起始位置
         * @param val_end 值迭代器的结束位置
         */
        explicit chunk(LF lf,
                       const A_ITER row_begin, const A_ITER row_end,
                       const B_ITER col_begin,
                       const C_ITER val_begin, const C_ITER val_end) :
                line_formatter(lf),
                row_iter(row_begin), row_end(row_end),
                col_iter(col_begin),
                val_iter(val_begin), val_end(val_end) {}

        /**
         * 生成并返回块的字符串表示。
         * @return 块的字符串表示
         */
        std::string operator()() {
            std::string chunk;
            chunk.reserve((row_end - row_iter)*25); // 预留足够空间以容纳所有可能的输出字符

            for (; row_iter != row_end; ++row_iter, ++col_iter) {
                if (val_iter != val_end) {
                    chunk += line_formatter.coord_matrix(*row_iter, *col_iter, *val_iter);
                    ++val_iter;
                } else {
                    chunk += line_formatter.coord_matrix_pattern(*row_iter, *col_iter);
                }
            }

            return chunk;
        }

        LF line_formatter; // 行格式化器
        A_ITER row_iter, row_end; // 行迭代器的当前位置和结束位置
        B_ITER col_iter; // 列迭代器的当前位置
        C_ITER val_iter, val_end; // 值迭代器的当前位置和结束位置
    };

    /**
     * 返回下一个数据块。
     * @param options 写入选项
     * @return 下一个数据块
     */
    chunk next_chunk(const write_options& options) {
        auto chunk_size = std::min(options.chunk_size_values, (int64_t)(row_end - row_iter));
        A_ITER row_chunk_end = row_iter + chunk_size;
        B_ITER col_chunk_end = col_iter + chunk_size;
        C_ITER val_chunk_end = (val_iter != val_end) ? val_iter + chunk_size : val_end;

        chunk c(line_formatter,
                row_iter, row_chunk_end,
                col_iter,
                val_iter, val_chunk_end);

        row_iter = row_chunk_end;
        col_iter = col_chunk_end;
        val_iter = val_chunk_end;

        return c;
    }

protected:
    LF line_formatter; // 行格式化器
    A_ITER row_iter, row_end; // 行迭代器的当前位置和结束位置
    B_ITER col_iter; // 列迭代器的当前位置
    C_ITER val_iter, val_end; // 值迭代器的当前位置和结束位置
};
    template<typename LF, typename PTR_ITER, typename IND_ITER, typename VAL_ITER>
    class csc_formatter {
    protected:
        // 行格式化器
        LF line_formatter;
        // 指针迭代器的开始、当前位置和结束位置
        PTR_ITER ptr_begin, ptr_iter, ptr_end;
        // 索引迭代器的开始位置
        IND_ITER ind_begin;
        // 值迭代器的开始和结束位置
        VAL_ITER val_begin, val_end;
        // 是否转置标志
        bool transpose;
        // 每列的非零元素平均数
        double nnz_per_column;
    };
    
    /**
     * Format dense arrays.
     */
    template<typename LF, typename VT_ITER>
    class array_formatter {
    public:
        // 构造函数，初始化行格式化器、值迭代器、存储顺序、行数和列数
        explicit array_formatter(LF lf, const VT_ITER& values, storage_order order, int64_t nrows, int64_t ncols) :
                line_formatter(lf), values(values), order(order), nrows(nrows), ncols(ncols) {}
    
        // 检查是否还有下一个列
        [[nodiscard]] bool has_next() const {
            return cur_col != ncols;
        }
    
        // 内部块类，用于迭代处理每个块
        class chunk {
        public:
            // 块的构造函数，初始化行格式化器、值迭代器、存储顺序、行数、列数和当前列索引
            explicit chunk(LF lf, const VT_ITER& values, storage_order order, int64_t nrows, int64_t ncols, int64_t cur_col) :
                    line_formatter(lf), values(values), order(order), nrows(nrows), ncols(ncols), cur_col(cur_col) {}
    
            // 生成并返回格式化后的字符串块
            std::string operator()() {
                std::string c;
                c.reserve(ncols * 15); // 预分配字符串容量
    
                for (int64_t row = 0; row < nrows; ++row) {
                    int64_t offset;
                    if (order == row_major) {
                        offset = row * ncols + cur_col; // 根据存储顺序计算偏移量
                    } else {
                        offset = cur_col * nrows + row; // 根据存储顺序计算偏移量
                    }
    
                    // 将行、列和值传递给行格式化器，生成格式化后的字符串，并添加到结果中
                    c += line_formatter.array_matrix(row, cur_col, *(values + offset));
                }
    
                return c;
            }
    
            LF line_formatter;
            const VT_ITER values;
            storage_order order;
            int64_t nrows, ncols;
            int64_t cur_col;
        };
    
        // 返回下一个块对象
        chunk next_chunk([[maybe_unused]] const write_options& options) {
            return chunk(line_formatter, values, order, nrows, ncols, cur_col++);
        }
    
    protected:
        LF line_formatter;
        const VT_ITER values;
        storage_order order;
        int64_t nrows, ncols;
        int64_t cur_col = 0;
    };
    
    /**
     * Formats any structure that has:
     * operator(row, col) - returns the value at (row, col)
     *
     * Includes Eigen Dense Matrix/Vector and NumPy arrays.
     */
    template<typename LF, typename DenseType, typename DIM>
    class dense_2d_call_formatter {
    // 定义一个公有类 `dense_2d_call_formatter`
    public:
        // 显式构造函数，初始化对象的行格式化器 `line_formatter`，密集矩阵 `mat`，行数 `nrows`，列数 `ncols`
        explicit dense_2d_call_formatter(LF lf, const DenseType& mat, DIM nrows, DIM ncols) :
        line_formatter(lf), mat(mat), nrows(nrows), ncols(ncols) {}

        // 声明一个不可忽略的布尔值方法 `has_next()`，判断是否还有下一列未处理
        [[nodiscard]] bool has_next() const {
            return col_iter < ncols;  // 返回当前列迭代器是否小于总列数 `ncols`
        }

        // 定义一个内部类 `chunk`
        class chunk {
        public:
            // 显式构造函数，初始化对象的行格式化器 `line_formatter`，密集矩阵 `mat`，行数 `nrows`，起始列迭代器 `col_iter`，结束列迭代器 `col_end`
            explicit chunk(LF lf, const DenseType& mat, DIM nrows, DIM col_iter, DIM col_end) :
                    line_formatter(lf), mat(mat), nrows(nrows), col_iter(col_iter), col_end(col_end) {}

            // 重载调用运算符 `operator()`
            std::string operator()() {
                std::string chunk;  // 定义一个空字符串 `chunk`
                chunk.reserve((col_end - col_iter) * nrows * 15);  // 预留足够的空间以容纳列片段的数据

                // 迭代处理分配的列范围
                for (; col_iter != col_end; ++col_iter) {
                    // 遍历每行数据
                    for (DIM row = 0; row < nrows; ++row)
                    {
                        // 将行格式化后的数组矩阵元素添加到 `chunk` 中
                        chunk += line_formatter.array_matrix(row, col_iter, mat(row, col_iter));
                    }
                }

                return chunk;  // 返回组合好的 `chunk`
            }

            LF line_formatter;       // 行格式化器对象
            const DenseType& mat;    // 密集矩阵对象的常量引用
            DIM nrows;               // 行数
            DIM col_iter, col_end;   // 起始列迭代器和结束列迭代器
        };

        // 定义一个公有方法 `next_chunk`
        chunk next_chunk(const write_options& options) {
            auto num_columns = (DIM)((double)options.chunk_size_values / nrows) + 1;  // 计算可处理的列数
            num_columns = std::min(num_columns, ncols - col_iter);  // 取较小值，确保不超出列范围

            DIM col_end = col_iter + num_columns;  // 计算本次处理的结束列迭代器
            chunk c(line_formatter, mat, nrows, col_iter, col_end);  // 创建 `chunk` 对象 `c`
            col_iter = col_end;  // 更新当前列迭代器为结束列迭代器

            return c;  // 返回 `chunk` 对象
        }

    protected:
        LF line_formatter;       // 行格式化器对象
        const DenseType& mat;    // 密集矩阵对象的常量引用
        DIM nrows, ncols;        // 行数和列数
        DIM col_iter = 0;        // 当前列迭代器，默认为 0
    };
}



# 这行代码仅仅是一个右括号 '}'，没有上下文和作用域，因此它本身没有实际意义，可能是代码中的错误或遗漏。
```