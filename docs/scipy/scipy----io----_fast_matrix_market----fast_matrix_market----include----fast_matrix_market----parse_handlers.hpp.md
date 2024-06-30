# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\fast_matrix_market\include\fast_matrix_market\parse_handlers.hpp`

```
// 版权声明和许可证信息，指定此文件的使用权限
// 在 LICENSE.txt 文件中可以找到 BSD 2-clause 许可证
// SPDX-License-Identifier: BSD-2-Clause
#pragma once

// 包含标准库头文件
#include <algorithm>    // 包含算法相关的函数和类模板
#include <charconv>     // 提供字符转换功能的头文件
#include <complex>      // 复数类模板和相关操作的头文件
#include <functional>   // 函数对象、函数指针及函数调用封装的头文件
#include <iterator>     // 迭代器相关操作的头文件
#include <type_traits>  // 提供类型特性检查功能的头文件

// 包含自定义头文件 "fast_matrix_market.hpp"
#include "fast_matrix_market.hpp"

// fast_matrix_market 命名空间
namespace fast_matrix_market {

    /**
     * This parse handler supports parallelism.
     * 此解析处理器支持并行处理。
     */
    constexpr int kParallelOk = 1;

    /**
     * Writing to the same (row, column) position a second time will affect the previous value.
     * This means that if there is a possibility of duplicate writes from different threads then this
     * parse handler is unsafe. Example: coordinate file parsed into a dense array. Coordinate files may have dupes.
     * 第二次写入相同的 (行，列) 位置会影响先前的值。
     * 这意味着如果存在不同线程可能写入重复数据的情况，此解析处理器就不安全。
     * 例如：将坐标文件解析成稠密数组。坐标文件可能会有重复数据。
     */
    constexpr int kDense = 2;

    /**
     * This parse handler can handle a variable number of elements. If this flag is not set then the memory
     * has already been allocated and must be filled. If this flag is set, then potentially fewer elements can be
     * written without loss of correctness.
     * 此解析处理器可以处理可变数量的元素。如果未设置此标志，则内存已分配并且必须填充。
     * 如果设置了此标志，则可以写入较少的元素而不会丢失正确性。
     * 这对于在广义对称性时不需要复制主对角线上的元素很有用。
     */
    constexpr int kAppending = 4;

    /**
     * Tuple handler. A single vector of (row, column, value) tuples.
     * 元组处理器。一个 (行，列，值) 元组的单一向量。
     */
    template<typename IT, typename VT, typename ITER>
    class tuple_parse_handler {
    public:
        using coordinate_type = IT;        // 定义坐标类型
        using value_type = VT;             // 定义值类型
        static constexpr int flags = kParallelOk;  // 使用并行处理

        using TUPLE = typename std::iterator_traits<ITER>::value_type;  // 迭代器值类型的别名

        // 构造函数，初始化迭代器
        explicit tuple_parse_handler(const ITER& iter) : begin_iter(iter), iter(iter) {}

        // 处理 (row, col, value) 元组的方法
        void handle(const coordinate_type row, const coordinate_type col, const value_type value) {
            *iter = TUPLE(row, col, value); // 将元组写入迭代器指向的位置
            ++iter;                        // 迭代器向前移动
        }

        // 获取偏移后的元组处理器
        tuple_parse_handler<IT, VT, ITER> get_chunk_handler(int64_t offset_from_begin) {
            return tuple_parse_handler(begin_iter + offset_from_begin);  // 返回新的元组处理器
        }

    protected:
        ITER begin_iter;  // 起始迭代器
        ITER iter;        // 当前迭代器
    };

    /**
     * Triplet handler. Separate row, column, value iterators.
     * 三元组处理器。分离的行、列、值迭代器。
     */
    template<typename IT_ITER, typename VT_ITER>
    class triplet_parse_handler {
        // 省略类模板的定义，留待后续注释
    /**
     * 公共接口和成员类型声明
     */
    public:
        using coordinate_type = typename std::iterator_traits<IT_ITER>::value_type;
        using value_type = typename std::iterator_traits<VT_ITER>::value_type;
        static constexpr int flags = kParallelOk;

        /**
         * 构造函数，初始化三个迭代器
         */
        explicit triplet_parse_handler(const IT_ITER& rows,
                                       const IT_ITER& cols,
                                       const VT_ITER& values) : begin_rows(rows), begin_cols(cols), begin_values(values),
                                                                rows(rows), cols(cols), values(values) {}

        /**
         * 处理函数，用给定的行、列和值填充迭代器指向的位置
         */
        void handle(const coordinate_type row, const coordinate_type col, const value_type value) {
            *rows = row;
            *cols = col;
            *values = value;

            ++rows;
            ++cols;
            ++values;
        }

        /**
         * 处理函数，处理模式矩阵中的行和列
         * @param row 行坐标
         * @param col 列坐标
         * @param pat 模式占位符类型，此处未使用
         */
        void handle(const coordinate_type row, const coordinate_type col, [[maybe_unused]] const pattern_placeholder_type& pat) {
            *rows = row;
            *cols = col;

            ++rows;
            ++cols;
            // values 迭代器不递增，因为此处理函数不处理值
        }

        /**
         * 返回偏移后的新的三元组处理器对象
         * @param offset_from_begin 从起始位置的偏移量
         * @return 偏移后的新的三元组处理器对象
         */
        triplet_parse_handler<IT_ITER, VT_ITER> get_chunk_handler(int64_t offset_from_begin) {
            return triplet_parse_handler(begin_rows + offset_from_begin,
                                         begin_cols + offset_from_begin,
                                         begin_values + offset_from_begin);
        }

    protected:
        IT_ITER begin_rows; ///< 起始行迭代器
        IT_ITER begin_cols; ///< 起始列迭代器
        VT_ITER begin_values; ///< 起始值迭代器

        IT_ITER rows; ///< 当前行迭代器
        IT_ITER cols; ///< 当前列迭代器
        VT_ITER values; ///< 当前值迭代器
    };

    /**
     * 用于处理模式矩阵的三元组处理器。仅支持行和列向量。
     */
    template<typename IT_ITER>
    class triplet_pattern_parse_handler {
    public:
        using coordinate_type = typename std::iterator_traits<IT_ITER>::value_type;
        using value_type = pattern_placeholder_type;
        static constexpr int flags = kParallelOk;

        /**
         * 构造函数，初始化行和列迭代器
         */
        explicit triplet_pattern_parse_handler(const IT_ITER& rows,
                                               const IT_ITER& cols) : begin_rows(rows), begin_cols(cols),
                                                                      rows(rows), cols(cols) {}

        /**
         * 处理函数，用给定的行和列填充迭代器指向的位置
         * @param row 行坐标
         * @param col 列坐标
         * @param ignored 模式占位符类型，此处未使用
         */
        void handle(const coordinate_type row, const coordinate_type col, [[maybe_unused]] const value_type ignored) {
            *rows = row;
            *cols = col;

            ++rows;
            ++cols;
            // values 迭代器不递增，因为此处理函数不处理值
        }

        /**
         * 返回偏移后的新的模式三元组处理器对象
         * @param offset_from_begin 从起始位置的偏移量
         * @return 偏移后的新的模式三元组处理器对象
         */
        triplet_pattern_parse_handler<IT_ITER> get_chunk_handler(int64_t offset_from_begin) {
            return triplet_pattern_parse_handler(begin_rows + offset_from_begin,
                                                 begin_cols + offset_from_begin);
        }
    protected:
        IT_ITER begin_rows; ///< 起始行迭代器
        IT_ITER begin_cols; ///< 起始列迭代器

        IT_ITER rows; ///< 当前行迭代器
        IT_ITER cols; ///< 当前列迭代器
    };
    /**
     * 三元组解析处理器类
     */
    class triplet_calling_parse_handler {
    public:
        // 使用 IT 类型作为坐标类型
        using coordinate_type = IT;
        // 使用 VT 类型作为值类型
        using value_type = VT;
        // 表示这个处理器支持并行处理
        static constexpr int flags = kParallelOk;

        /**
         * 构造函数，初始化三元组解析处理器
         * @param rows 行数组引用
         * @param cols 列数组引用
         * @param values 值数组引用
         * @param offset 偏移量，默认为 0
         */
        explicit triplet_calling_parse_handler(IT_ARR& rows,
                                               IT_ARR& cols,
                                               VT_ARR& values,
                                               int64_t offset = 0) : rows(rows), cols(cols), values(values), offset(offset) {}

        /**
         * 处理方法，将给定的行、列、值存储到对应数组中，并增加偏移量
         * @param row 行坐标
         * @param col 列坐标
         * @param value 值
         */
        void handle(const coordinate_type row, const coordinate_type col, const value_type value) {
            rows(offset) = row;
            cols(offset) = col;
            values(offset) = value;

            ++offset;
        }

        /**
         * 获取新的三元组解析处理器，用于处理从指定偏移量开始的数据
         * @param offset_from_begin 从起始位置的偏移量
         * @return 新的三元组解析处理器
         */
        triplet_calling_parse_handler<IT, VT, IT_ARR, VT_ARR> get_chunk_handler(int64_t offset_from_begin) {
            return triplet_calling_parse_handler(rows, cols, values, offset_from_begin);
        }

    protected:
        IT_ARR& rows;        // 行数组引用
        IT_ARR& cols;        // 列数组引用
        VT_ARR& values;      // 值数组引用

        int64_t offset;      // 偏移量
    };

    /**
     * 双元处理器，用于处理 (索引, 值) 的稀疏向量
     */
    template<typename IT_ITER, typename VT_ITER>
    class doublet_parse_handler {
    public:
        // 使用迭代器的值类型作为坐标类型和值类型
        using coordinate_type = typename std::iterator_traits<IT_ITER>::value_type;
        using value_type = typename std::iterator_traits<VT_ITER>::value_type;
        // 表示这个处理器支持并行处理
        static constexpr int flags = kParallelOk;

        /**
         * 构造函数，初始化双元解析处理器
         * @param index 索引迭代器
         * @param values 值迭代器
         */
        explicit doublet_parse_handler(const IT_ITER& index,
                                       const VT_ITER& values) : begin_index(index), begin_values(values),
                                                                index(index), values(values) {}

        /**
         * 处理方法，将给定的行、列、值存储到对应迭代器中，并增加迭代器
         * @param row 行坐标
         * @param col 列坐标
         * @param value 值
         */
        void handle(const coordinate_type row, const coordinate_type col, const value_type value) {
            *index = std::max(row, col);
            *values = value;

            ++index;
            ++values;
        }

        /**
         * 获取新的双元解析处理器，用于处理从指定偏移量开始的数据
         * @param offset_from_begin 从起始位置的偏移量
         * @return 新的双元解析处理器
         */
        doublet_parse_handler<IT_ITER, VT_ITER> get_chunk_handler(int64_t offset_from_begin) {
            return doublet_parse_handler(begin_index + offset_from_begin,
                                         begin_values + offset_from_begin);
        }
    protected:
        IT_ITER begin_index;   // 起始索引迭代器
        VT_ITER begin_values;  // 起始值迭代器

        IT_ITER index;         // 当前索引迭代器
        VT_ITER values;        // 当前值迭代器
    };

    /**
     * 适用于任何支持 `mat(row, column) += value` 操作的类型
     */
    template<typename MAT, typename IT, typename VT>
    class dense_2d_call_adding_parse_handler {
    /**
     * 二维稠密数组处理器（按行主序）。
     */
    template <typename MAT, typename IT, typename VT>
    class dense_2d_call_adding_parse_handler {
    public:
        // 使用 IT 类型作为坐标类型
        using coordinate_type = IT;
        // 使用 VT 类型作为数值类型
        using value_type = VT;
        // 设置处理器的标志位，支持并行处理和稠密数组
        static constexpr int flags = kParallelOk | kDense;

        // 构造函数，初始化 MAT 对象的引用
        explicit dense_2d_call_adding_parse_handler(MAT &mat) : mat(mat) {}

        // 处理函数，用于处理给定坐标位置 (row, col) 处的值增加操作
        void handle(const coordinate_type row, const coordinate_type col, const value_type value) {
            // 使用 std::plus<value_type> 对象执行值增加操作
            mat(row, col) += std::plus<value_type>()(mat(row, col), value);
        }

        // 获取处理器的分块处理器，此处的 offset_from_begin 参数未使用
        dense_2d_call_adding_parse_handler<MAT, IT, VT> get_chunk_handler([[maybe_unused]] int64_t offset_from_begin) {
            return *this;
        }

    protected:
        // MAT 对象的引用
        MAT &mat;
    };

    /**
     * 稠密数组处理器（按行主序）。
     */
    template <typename VT_ITER>
    class dense_adding_parse_handler {
    public:
        // 使用 int64_t 作为坐标类型
        using coordinate_type = int64_t;
        // 使用 VT_ITER 迭代器的值类型作为数值类型
        using value_type = typename std::iterator_traits<VT_ITER>::value_type;
        // 设置处理器的标志位，支持并行处理和稠密数组
        static constexpr int flags = kParallelOk | kDense;

        // 构造函数，初始化 values 迭代器、存储顺序、行数和列数
        explicit dense_adding_parse_handler(const VT_ITER& values, storage_order order, int64_t nrows, int64_t ncols) :
        values(values), order(order), nrows(nrows), ncols(ncols) {}

        // 处理函数，用于处理给定坐标位置 (row, col) 处的值增加操作
        void handle(const coordinate_type row, const coordinate_type col, const value_type value) {
            int64_t offset;
            // 根据存储顺序计算偏移量
            if (order == row_major) {
                offset = row * ncols + col;
            } else {
                offset = col * nrows + row;
            }
            // 使用 std::plus<value_type> 对象执行值增加操作
            values[offset] = std::plus<value_type>()(values[offset], value);
        }

        // 获取处理器的分块处理器，此处的 offset_from_begin 参数未使用
        dense_adding_parse_handler<VT_ITER> get_chunk_handler([[maybe_unused]] int64_t offset_from_begin) {
            return *this;
        }

    protected:
        // VT_ITER 迭代器，用于存储数据
        VT_ITER values;
        // 存储顺序
        storage_order order;
        // 行数
        int64_t nrows;
        // 列数
        int64_t ncols;
    };
}
```