# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\fast_matrix_market\include\fast_matrix_market\types.hpp`

```
// 版权声明和许可信息
// 该源代码的使用受 BSD 2-Clause 许可证的约束，详见 LICENSE.txt 文件
// SPDX-License-Identifier: BSD-2-Clause

#pragma once

#include <complex>    // 包含复数支持
#include <map>        // 包含用于映射的标准库
#include <cstdint>    // 包含用于固定大小整数类型的标准库
#include <string>     // 包含字符串支持的标准库

namespace fast_matrix_market {

    // 定义对象类型枚举
    enum object_type {matrix, vector};
    // 对象类型到字符串的映射
    const std::map<object_type, const std::string> object_map = {
            {matrix, "matrix"},
            {vector, "vector"},
    };

    // 定义格式类型枚举
    enum format_type {array, coordinate};
    // 格式类型到字符串的映射
    const std::map<format_type, const std::string> format_map = {
            {array, "array"},
            {coordinate, "coordinate"},
    };

    // 定义字段类型枚举
    enum field_type {real, double_, complex, integer, pattern, unsigned_integer};
    // 字段类型到字符串的映射
    const std::map<field_type, const std::string> field_map = {
            {real, "real"},
            {double_, "double"},  // 非标准
            {complex, "complex"},
            {integer, "integer"},
            {pattern, "pattern"},
            {unsigned_integer, "unsigned-integer"}, // 仅限于 SciPy
    };

    // 定义对称性类型枚举
    enum symmetry_type {general, symmetric, skew_symmetric, hermitian};
    // 对称性类型到字符串的映射
    const std::map<symmetry_type, const std::string> symmetry_map = {
            {general, "general"},
            {symmetric, "symmetric"},
            {skew_symmetric, "skew-symmetric"},
            {hermitian, "hermitian"},
    };

    /**
     * Matrix Market 头部
     */
    struct matrix_market_header {
        matrix_market_header() = default;   // 默认构造函数
        explicit matrix_market_header(int64_t vector_length) : object(vector), vector_length(vector_length) {}
        matrix_market_header(int64_t nrows, int64_t ncols) : nrows(nrows), ncols(ncols) {}

        object_type object = matrix;        // 对象类型，默认为矩阵
        format_type format = coordinate;    // 格式类型，默认为坐标格式
        field_type field = real;            // 字段类型，默认为实数
        symmetry_type symmetry = general;   // 对称性类型，默认为一般性

        // 矩阵维度
        int64_t nrows = 0;                  // 行数，默认为0
        int64_t ncols = 0;                  // 列数，默认为0

        // 向量维度
        int64_t vector_length = 0;          // 向量长度，默认为0

        // 稀疏对象的非零元素个数
        int64_t nnz = 0;                    // 非零元素个数，默认为0

        // 文件头部的注释
        std::string comment;                // 文件头的注释内容

        // 头部占据的行数。这个值会被 read_header() 函数填充。
        int64_t header_line_count = 1;      // 头部所占行数，默认为1
    };

    // 定义存储顺序枚举
    enum storage_order {row_major = 1, col_major = 2};

    // 定义超出范围行为枚举
    enum out_of_range_behavior {BestMatch = 1, ThrowOutOfRange = 2};

}  // namespace fast_matrix_market


这段代码是一个 C++ 的命名空间，定义了用于矩阵市场文件格式的各种枚举和数据结构。每个枚举都有一个对应的映射表，将枚举值映射到字符串，便于理解和处理不同的对象、格式、类型和对称性。结构体 `matrix_market_header` 描述了矩阵市场文件的头部信息，包括对象类型、格式、字段类型、对称性等，以及与矩阵和向量相关的维度信息。
    // 定义了一个结构体 read_options，用于存储读取选项的设置

    /**
     * Chunk size for the parsing step, in bytes.
     * 解析步骤的块大小，单位为字节。
     */
    int64_t chunk_size_bytes = 2 << 20;

    /**
     * If true then any symmetries other than general are expanded out.
     * For any symmetries other than general, only entries in the lower triangular portion need be supplied.
     * symmetric: for (row, column, value), also generate (column, row, value) except if row==column
     * skew-symmetric: for (row, column, value), also generate (column, row, -value) except if row==column
     * hermitian: for (row, column, value), also generate (column, row, complex_conjugate(value)) except if row==column
     * 如果为 true，则展开除了一般对称性外的所有对称性。
     * 对于除了一般对称性之外的任何对称性，只需提供下三角部分的条目。
     * 对称矩阵：对于 (行, 列, 值)，还会生成 (列, 行, 值)，但如果行==列则不生成。
     * 反对称矩阵：对于 (行, 列, 值)，还会生成 (列, 行, -值)，但如果行==列则不生成。
     * Hermite 矩阵：对于 (行, 列, 值)，还会生成 (列, 行, 复共轭(值))，但如果行==列则不生成。
     */
    bool generalize_symmetry = true;

    /**
     * If true, perform symmetry generalization in the application binding as a post-processing step.
     * If supported by the binding this method can avoid extra diagonal elements.
     * If false or unsupported, diagonals are handled according to `generalize_coordinate_diagnonal_values`.
     * 如果为 true，在应用绑定中作为后处理步骤执行对称性泛化。
     * 如果绑定支持，此方法可以避免额外的对角元素。
     * 如果为 false 或不支持，则根据 `generalize_coordinate_diagnonal_values` 处理对角线元素。
     */
    bool generalize_symmetry_app = true;

    /**
     * Generalize Symmetry:
     * How to handle a value on the diagonal of a symmetric coordinate matrix.
     *  - DuplicateElement: Duplicate the diagonal element
     *  - ExtraZeroElement: emit a zero along with the diagonal element. The zero will appear first.
     *
     *  The extra cannot simply be omitted because the handlers work by setting already-allocated memory. This
     *  is necessary for efficient parallelization.
     *
     *  This value is ignored if the parse handler has the kAppending flag set. In that case only a single
     *  diagonal element is emitted.
     * 对称性泛化：
     * 如何处理对称坐标矩阵对角线上的值。
     *  - DuplicateElement: 复制对角线元素
     *  - ExtraZeroElement: 在对角线元素前面加上一个零。零将首先出现。
     *
     *  额外的零不能简单地省略，因为处理程序通过设置已分配的内存来工作。这对于有效的并行化是必要的。
     *
     *  如果解析处理程序设置了 kAppending 标志，则此值将被忽略。在这种情况下，只会发出单个对角线元素。
     */
    enum {ExtraZeroElement, DuplicateElement} generalize_coordinate_diagnonal_values = ExtraZeroElement;

    /**
     * Whether parallel implementation is allowed.
     * 是否允许并行实现。
     */
    bool parallel_ok = true;

    /**
     * Number of threads to use. 0 means std::thread::hardware_concurrency().
     * 要使用的线程数。0 表示使用 std::thread::hardware_concurrency()。
     */
    int num_threads = 0;

    /**
     * How to handle floating-point values that do not fit into their declared type.
     * For example, parsing 1e9999 will
     *  - BestMatch: return Infinity
     *  - ThrowOutOfRange: throw out_of_range exception
     * 如何处理不适合其声明类型的浮点值。
     * 例如，解析 1e9999 将
     *  - BestMatch: 返回 Infinity
     *  - ThrowOutOfRange: 抛出 out_of_range 异常
     */
    out_of_range_behavior float_out_of_range_behavior = BestMatch;
    # 定义结构体 write_options，用于控制写入操作的选项

    struct write_options {
        # 定义默认的 chunk 大小为 2^12（即 4096）
        int64_t chunk_size_values = 2 << 12;

        /**
         * 是否允许并行实现。
         */
        bool parallel_ok = true;

        /**
         * 使用的线程数。0 表示使用 std::thread::hardware_concurrency() 决定的线程数。
         */
        int num_threads = 0;

        /**
         * 浮点数格式化精度。
         * 占位符。当前未使用，因为支持多种浮点数渲染后端。
         */
        int precision = -1;

        /**
         * 是否总是写入注释行，即使注释为空。
         */
        bool always_comment = false;

        /**
         * 是否基于提供的数据结构确定头字段类型。
         *
         * 如果为 true，则使用 `get_field_type()` 设置头字段。唯一的例外是
         * 如果字段 == pattern，则不做更改。
         *
         * 可能设置为 false 的原因：
         *  - 使用自定义类型，例如 std::string，其中 `get_field_type()` 可能返回错误的类型
         *  - 将整数结构写入为实数
         */
        bool fill_header_field_type = true;
    };

    # 检查类型 T 是否为复数类型的模板结构
    template<class T> struct is_complex : std::false_type {};

    # 特化模板，检查类型 T 是否为 std::complex<T> 的复数类型
    template<class T> struct is_complex<std::complex<T>> : std::true_type {};

    # 检查类型 T 是否能够读取复数的模板结构
    template<class T> struct can_read_complex : is_complex<T> {};
}



# 这行代码表示一个代码块的结束
```