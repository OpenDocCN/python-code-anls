# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\fast_matrix_market\include\fast_matrix_market\read_body.hpp`

```
// 版权声明和许可信息
// 2022 年 Adam Lugowski 版权所有。此源代码的使用受 LICENSE.txt 文件中的 BSD 2-Clause 许可证约束。
// SPDX 许可证标识符：BSD-2-Clause

// 预处理指令，确保头文件只被包含一次
#pragma once

// 包含 C++ 标准库的头文件
#include <functional>

// 包含自定义的头文件
#include "fast_matrix_market.hpp"
#include "chunking.hpp"

// fast_matrix_market 命名空间的声明
namespace fast_matrix_market {

    // line_counts 结构体，用于跟踪文件行数和元素数
    struct line_counts {
        int64_t file_line = 0;    // 文件行数
        int64_t element_num = 0;  // 元素数
    };

    // compile_format 枚举类型，用于编译格式选项
    enum compile_format {compile_array_only = 1, compile_coordinate_only = 2, compile_all = 3};

    /**
     * pattern_parse_adapter 类模板，用于处理模式矩阵的适配器包装器。
     * 它将一个固定值转发给处理程序，例如将 1.0 写入双精度矩阵，避免使用零值。
     */
    template<typename FWD_HANDLER>
    class pattern_parse_adapter {
    public:
        using coordinate_type = typename FWD_HANDLER::coordinate_type;
        using value_type = typename FWD_HANDLER::value_type;
        static constexpr int flags = FWD_HANDLER::flags;

        // 构造函数，初始化成员变量 handler 和 fwd_value
        explicit pattern_parse_adapter(const FWD_HANDLER &handler, typename FWD_HANDLER::value_type fwd_value) : handler(
                handler), fwd_value(fwd_value) {}

        // 处理函数，将固定值转发给 handler 处理
        void handle(const coordinate_type row, const coordinate_type col, [[maybe_unused]] const pattern_placeholder_type ignored) {
            handler.handle(row, col, fwd_value);
        }

        // 处理函数，将值转发给 handler 处理
        void handle(const coordinate_type row, const coordinate_type col, const value_type val) {
            handler.handle(row, col, val);
        }

        // 获取分块处理器的函数
        pattern_parse_adapter<FWD_HANDLER> get_chunk_handler(int64_t offset_from_start) {
            return pattern_parse_adapter<FWD_HANDLER>(handler.get_chunk_handler(offset_from_start), fwd_value);
        }

    protected:
        FWD_HANDLER handler;                // 原始处理器对象
        typename FWD_HANDLER::value_type fwd_value;  // 转发的固定值
    };

    // 如果未定义 FMM_SCIPY_PRUNE，则定义下列内容
#ifndef FMM_SCIPY_PRUNE
    /**
     * complex_parse_adapter 类模板，用于将实数/整数文件读取为 std::complex 矩阵，
     * 并将所有虚部分设置为零的适配器包装器。
     *
     * @deprecated 此适配器现在在内部循环中执行适应性，因此不再需要。
     */
    template<typename COMPLEX_HANDLER>
    class [[maybe_unused]] complex_parse_adapter {
    // 声明公共部分
    public:
        // 使用 COMPLEX_HANDLER 的坐标类型作为 coordinate_type
        using coordinate_type = typename COMPLEX_HANDLER::coordinate_type;
        // 使用 COMPLEX_HANDLER 的值类型作为 complex_type
        using complex_type = typename COMPLEX_HANDLER::value_type;
        // complex_type 的值类型作为 value_type
        using value_type = typename complex_type::value_type;
        // 从 COMPLEX_HANDLER 中获取 flags，并设置为静态常量
        static constexpr int flags = COMPLEX_HANDLER::flags;

        // 显式构造函数，接受 COMPLEX_HANDLER 的引用作为参数
        explicit complex_parse_adapter(const COMPLEX_HANDLER &handler) : handler(handler) {}

        // 处理给定行列坐标及模式占位符的处理函数
        void handle(const coordinate_type row, const coordinate_type col, const pattern_placeholder_type& pat) {
            // 调用 handler 对象的 handle 方法，传递行列坐标及模式占位符
            handler.handle(row, col, pat);
        }

        // 处理给定行列坐标及实部值的处理函数
        void handle(const coordinate_type row, const coordinate_type col, const value_type real) {
            // 调用 handler 对象的 handle 方法，传递行列坐标及复数对象，虚部设为 0
            handler.handle(row, col, complex_type(real, 0));
        }

        // 获取从起始处偏移量为 offset_from_start 的 chunk 处理器
        complex_parse_adapter<COMPLEX_HANDLER> get_chunk_handler(int64_t offset_from_start) {
            // 调用 handler 对象的 get_chunk_handler 方法，传递偏移量，并返回新的适配器对象
            return complex_parse_adapter(handler.get_chunk_handler(offset_from_start));
        }

    // 受保护部分
    protected:
        // 内部使用的 COMPLEX_HANDLER 对象
        COMPLEX_HANDLER handler;
    };
#endif

这是一个预处理器指令，用于结束一个条件编译区块。


    ///////////////////////////////////////////////////////////////////
    // Limit bool parallelism
    // vector<bool> is specialized to use a bitfield-like scheme. This means
    // that different elements can share the same bytes, making
    // writes to this container require locking.
    // Instead, disable parallelism for bools.

限制布尔类型的并行性。`vector<bool>` 被特化为使用位域方案。这意味着不同的元素可以共享相同的字节，因此对这种容器的写操作需要加锁。为了避免这种情况，禁用布尔类型的并行处理。


    template <typename T, typename std::enable_if<std::is_same<T, bool>::value, int>::type = 0>
    bool limit_parallelism_for_value_type(bool) {
        return false;
    }

模板函数，当模板参数 `T` 是布尔类型时，限制并行性的函数。始终返回 `false`，表示不支持并行处理。


    template <typename T, typename std::enable_if<!std::is_same<T, bool>::value, int>::type = 0>
    bool limit_parallelism_for_value_type(bool parallelism_selected) {
        return parallelism_selected;
    }

模板函数，当模板参数 `T` 不是布尔类型时，根据 `parallelism_selected` 的值返回结果。这样允许对非布尔类型进行并行处理的选择。


    ///////////////////////////////////////////////////////////////////
    // Chunks
    ///////////////////////////////////////////////////////////////////

表示接下来是关于“Chunks（块）”的部分的开始。


    template <typename RET, typename T>
    RET get_symmetric_value(const T& v, const symmetry_type& symmetry) {
        switch (symmetry) {
            case symmetric:
                return v;
            case skew_symmetric:
                if constexpr (std::is_unsigned_v<T>) {
                    throw invalid_argument("Cannot load skew-symmetric matrix into unsigned value type.");
                } else {
                    return negate(v);
                }
            case hermitian:
                return complex_conjugate(v);
            case general:
                return v;
        }
        return v;
    }

模板函数 `get_symmetric_value`，根据对称性类型 `symmetry` 返回相应的值。处理对称矩阵的不同类型：对称、斜对称、厄米特、一般情况。


    template<typename HANDLER, typename IT, typename VT>

模板的开始，声明了三个模板参数 `HANDLER`、`IT`、`VT`，但未完全定义。
    /**
     * 处理通用的对称性坐标转换，根据不同的对称性类型调用处理器进行处理。
     */
    void generalize_symmetry_coordinate(HANDLER& handler,
                                        const matrix_market_header &header,
                                        const read_options &options,
                                        const IT& row,
                                        const IT& col,
                                        const VT& value) {
        // 如果行列索引不相同
        if (col != row) {
            // 根据矩阵头部的对称性类型进行处理
            switch (header.symmetry) {
                case symmetric:
                    // 对称矩阵：处理 (col, row, value)
                    handler.handle(col, row, value);
                    break;
                case skew_symmetric:
                    // 反对称矩阵：如果值类型不是无符号数，则处理 (col, row, -value)
                    if constexpr (!std::is_unsigned_v<typename HANDLER::value_type>) {
                        handler.handle(col, row, negate(value));
                    } else {
                        // 否则抛出异常，无法加载反对称矩阵到无符号值类型
                        throw invalid_argument("Cannot load skew-symmetric matrix into unsigned value type.");
                    }
                    break;
                case hermitian:
                    // Hermite 矩阵：处理 (col, row, value 的复共轭)
                    handler.handle(col, row, complex_conjugate(value));
                    break;
                case general:
                    break;
            }
        } else {
            // 如果行列索引相同
            if (!test_flag(HANDLER::flags, kAppending)) {
                // 根据选项处理对角线值
                switch (options.generalize_coordinate_diagnonal_values) {
                    case read_options::ExtraZeroElement:
                        // 添加额外的零元素
                        handler.handle(row, col, get_zero<typename HANDLER::value_type>());
                        break;
                    case read_options::DuplicateElement:
                        // 复制元素值
                        handler.handle(row, col, value);
                        break;
                }
            }
        }
    }
    
    /**
     * 处理通用的对称性数组转换，根据不同的对称性类型调用处理器进行处理。
     */
    template<typename HANDLER, typename IT, typename VT>
    void generalize_symmetry_array(HANDLER& handler,
                                   const matrix_market_header &header,
                                   const IT& row,
                                   const IT& col,
                                   const VT& value) {
        // 根据矩阵头部的对称性类型进行处理
        switch (header.symmetry) {
            case symmetric:
                // 对称矩阵：处理 (col, row, value)
                handler.handle(col, row, value);
                break;
            case skew_symmetric:
                // 反对称矩阵：如果值类型不是无符号数，则处理 (col, row, -value)
                if constexpr (!std::is_unsigned_v<typename HANDLER::value_type>) {
                    handler.handle(col, row, negate(value));
                } else {
                    // 否则抛出异常，无法加载反对称矩阵到无符号值类型
                    throw invalid_argument("Cannot load skew-symmetric matrix into unsigned value type.");
                }
                break;
            case hermitian:
                // Hermite 矩阵：处理 (col, row, value 的复共轭)
                handler.handle(col, row, complex_conjugate(value));
                break;
            case general:
                break;
        }
    }
    
    /**
     * 读取一个值，将实数矩阵值适配到复杂数据结构。
     */
    template <typename value_type>
    // 根据模板参数判断数值类型是否为复数类型，如果是复数则执行条件语句块
    void read_real_or_complex(value_type& value,
                              // 传入参数，表示当前读取位置的指针，指向数据末尾的指针，以及数据的头部信息和读取选项
                              const char*& pos,
                              const char* end,
                              const matrix_market_header &header,
                              const read_options &options) {
        // 如果 value_type 是复数类型
        if constexpr (is_complex<value_type>::value) {
            // 如果数据文件头部标识为复数类型
            if (header.field == complex) {
                // 调用 read_value 函数读取复数值并更新读取位置指针 pos
                pos = read_value(pos, end, value, options);
            } else {
                // 否则，如果头部标识为实数类型，声明一个实数变量 real
                typename value_type::value_type real;
                // 调用 read_value 函数读取实数值并更新读取位置指针 pos
                pos = read_value(pos, end, real, options);
                // 将读取得到的实数部分赋值给复数 value 的实部，虚部设为 0
                value.real(real);
                value.imag(0);
            }
        } else {
            // 如果 value_type 不是复数类型，直接调用 read_value 函数读取值并更新读取位置指针 pos
            pos = read_value(pos, end, value, options);
        }
    }

    // 声明一个模板函数，参数为 HANDLER 类型
    template<typename HANDLER>
    line_counts read_chunk_matrix_coordinate(const std::string &chunk, const matrix_market_header &header,
                                             line_counts line, HANDLER &handler, const read_options &options) {
        // 将 chunk 字符串转换为 C 风格字符串，并初始化指针 pos 指向其开头
        const char *pos = chunk.c_str();
        // 计算 chunk 的结束位置
        const char *end = pos + chunk.size();

        // 循环处理 chunk 中的每一行内容，直到 pos 到达 chunk 的结尾
        while (pos != end) {
            try {
                // 定义行号、列号和数值的变量
                typename HANDLER::coordinate_type row, col;
                typename HANDLER::value_type value;

                // 跳过空格和换行符，更新 pos 和行号
                pos = skip_spaces_and_newlines(pos, line.file_line);
                // 如果 pos 到达 chunk 的结尾，则退出循环，表示空行
                if (pos == end) {
                    // empty line
                    break;
                }
                // 如果读取的元素数量超过了文件头中的非零元素数量，则抛出异常
                if (line.element_num >= header.nnz) {
                    throw invalid_mm("Too many lines in file (file too long)");
                }

                // 读取行号并更新 pos
                pos = read_int(pos, end, row);
                // 跳过行号后的空格
                pos = skip_spaces(pos);
                // 读取列号并更新 pos
                pos = read_int(pos, end, col);
                // 如果不是模式文件，则跳过列号后的空格并读取实数或复数值
                if (header.field != pattern) {
                    pos = skip_spaces(pos);
                    read_real_or_complex(value, pos, end, header, options);
                }
                // 跳到下一行的起始位置
                pos = bump_to_next_line(pos, end);

                // 验证行号是否在有效范围内（从 1 开始到 nrows）
                if (row <= 0 || static_cast<int64_t>(row) > header.nrows) {
                    throw invalid_mm("Row index out of bounds");
                }
                // 验证列号是否在有效范围内（从 1 开始到 ncols）
                if (col <= 0 || static_cast<int64_t>(col) > header.ncols) {
                    throw invalid_mm("Column index out of bounds");
                }

                // 转换 Matrix Market 的一维坐标为 C 风格的零基坐标
                row = row - 1;
                col = col - 1;

                // 处理对称性的通用化
                // 如果不是通用对称性且选项开启了通用对称性处理，则调用 generalize_symmetry_coordinate 函数
                if (header.symmetry != general && options.generalize_symmetry) {
                    if (header.field != pattern) {
                        generalize_symmetry_coordinate(handler, header, options, row, col, value);
                    } else {
                        generalize_symmetry_coordinate(handler, header, options, row, col, pattern_placeholder_type());
                    }
                }

                // 如果不是模式文件，则调用 handler 的 handle 方法处理行号、列号和数值
                if (header.field != pattern) {
                    handler.handle(row, col, value);
                } else {
                    handler.handle(row, col, pattern_placeholder_type());
                }

                // 更新文件行数和元素数量
                ++line.file_line;
                ++line.element_num;
            } catch (invalid_mm& inv) {
                // 捕获并处理 invalid_mm 异常，将当前行号添加到异常信息中并重新抛出
                inv.prepend_line_number(line.file_line + 1);
                throw;
            }
        }
        // 返回处理后的 line 对象
        return line;
    }
#ifndef FMM_NO_VECTOR
    // 如果未定义 FMM_NO_VECTOR 宏，则编译以下代码块

    template<typename HANDLER>
    // 模板函数，参数为 HANDLER 类型
    line_counts read_chunk_vector_coordinate(const std::string &chunk, const matrix_market_header &header,
                                             line_counts line, HANDLER &handler, const read_options &options) {
        // 从 chunk 字符串中读取向量坐标数据块，处理到 HANDLER 中，并返回处理后的行数计数器 line_counts

        const char *pos = chunk.c_str();
        // 设置 pos 指向 chunk 字符串的首地址
        const char *end = pos + chunk.size();
        // 设置 end 指向 chunk 字符串的末尾地址

        while (pos != end) {
            // 循环处理 pos 指向的字符，直到 pos 到达 end

            try {
                // 尝试执行以下代码块，捕获异常并处理
                typename HANDLER::coordinate_type row;
                // 定义行号变量，根据 HANDLER 类型确定
                typename HANDLER::value_type value;
                // 定义数值类型变量，根据 HANDLER 类型确定

                pos = skip_spaces_and_newlines(pos, line.file_line);
                // 跳过 pos 指向的空白字符和换行符，更新 pos 的位置和行号计数

                if (pos == end) {
                    // 如果 pos 已经到达 chunk 字符串末尾
                    // 空行
                    break;
                }

                if (line.element_num >= header.nnz) {
                    // 如果元素数量超过了 header 中的非零元素总数 nnz
                    throw invalid_mm("Too many lines in file (file too long)");
                    // 抛出异常，文件行数过多（文件太长）
                }

                pos = read_int(pos, end, row);
                // 从 pos 开始读取整数，并更新 pos 和行号计数

                if (header.field != pattern) {
                    // 如果 header 中的字段不是 pattern
                    pos = skip_spaces(pos);
                    // 跳过 pos 指向的空白字符
                    read_real_or_complex(value, pos, end, header, options);
                    // 从 pos 开始读取实数或复数，并更新 pos
                }

                pos = bump_to_next_line(pos, end);
                // 跳过 pos 指向的换行符，更新 pos 的位置

                // 验证行号是否合法
                if (row <= 0 || static_cast<int64_t>(row) > header.vector_length) {
                    // 如果行号小于等于 0 或者大于 header 中的向量长度
                    throw invalid_mm("Vector index out of bounds");
                    // 抛出异常，向量索引超出范围
                }

                // Matrix Market 格式中索引从 1 开始，将 row 转换为零基索引
                row = row - 1;

                if (header.field != pattern) {
                    // 如果 header 中的字段不是 pattern
                    handler.handle(row, 0, value);
                    // 处理向量坐标数据到 HANDLER 中
                } else {
                    handler.handle(row, 0, pattern_placeholder_type());
                    // 处理模式占位符数据到 HANDLER 中
                }

                ++line.file_line;
                // 增加行号计数
                ++line.element_num;
                // 增加元素数量计数
            } catch (invalid_mm& inv) {
                // 捕获 invalid_mm 异常
                inv.prepend_line_number(line.file_line + 1);
                // 在异常消息前加上行号
                throw;
                // 抛出异常
            }
        }
        return line;
        // 返回处理后的行数计数器
    }
#endif

    template<typename HANDLER>
    // 模板函数，参数为 HANDLER 类型
    line_counts read_chunk_array(const std::string &chunk, const matrix_market_header &header, line_counts line,
                                 HANDLER &handler, const read_options &options,
                                 typename HANDLER::coordinate_type &row,
                                 typename HANDLER::coordinate_type &col) {
        // 从 chunk 字符串中读取数组数据块，处理到 HANDLER 中，并返回处理后的行数计数器 line_counts

        const char *pos = chunk.c_str();
        // 设置 pos 指向 chunk 字符串的首地址
        const char *end = pos + chunk.size();
        // 设置 end 指向 chunk 字符串的末尾地址

        if (header.symmetry == skew_symmetric) {
            // 如果 header 中的对称性是 skew_symmetric（斜对称）
            if (row == 0 && col == 0 && header.nrows > 0) {
                // 如果行号和列号都为 0，并且 header 中的行数大于 0
                // 斜对称矩阵的对角线元素为零
//                if (test_flag(HANDLER::flags, kDense)) {
//                    handler.handle(row, col, get_zero<typename HANDLER::value_type>());
`
//                }
                // 重置行数为1，因为每个新的列要从第一行开始读取
                row = 1;
            }
        }

        // 在未到达文件结尾之前循环处理文件内容
        while (pos != end) {
            try {
                typename HANDLER::value_type value;

                // 跳过空格和换行符，更新读取位置
                pos = skip_spaces_and_newlines(pos, line.file_line);
                if (pos == end) {
                    // 如果是空行则跳出循环
                    break;
                }
                if (static_cast<int64_t>(col) >= header.ncols) {
                    // 如果列数超过了文件头中的列数，抛出异常
                    throw invalid_mm("Too many values in array (file too long)");
                }

                // 读取实数或复数的值
                read_real_or_complex(value, pos, end, header, options);
                // 更新位置到下一行的起始位置
                pos = bump_to_next_line(pos, end);

                // 调用处理程序处理当前值
                handler.handle(row, col, value);

                // 如果行数不等于列数且允许对称化，则进行对称化处理
                if (row != col && options.generalize_symmetry) {
                    generalize_symmetry_array(handler, header, row, col, value);
                }

                // 矩阵市场格式是列优先，向下移动一列
                ++row;
                if (static_cast<int64_t>(row) == header.nrows) {
                    ++col;
                    if (header.symmetry == general) {
                        row = 0;
                    } else {
                        row = col;
                        if (header.symmetry == skew_symmetric) {
                            // 偏对称矩阵对角线为零
//                            if (test_flag(HANDLER::flags, kDense)) {
//                                handler.handle(row, col, get_zero<typename HANDLER::value_type>());
//                            }
                            // 如果当前行数小于行数-1，则向下移动一行
                            if (static_cast<int64_t>(row) < header.nrows-1) {
                                ++row;
                            }
                        }
                    }
                }

                // 更新文件行数和元素数
                ++line.file_line;
                ++line.element_num;
            } catch (invalid_mm& inv) {
                // 捕获异常并加上文件中的行数信息，然后重新抛出异常
                inv.prepend_line_number(line.file_line + 1);
                throw;
            }
        }
        // 返回处理过的行数统计信息
        return line;
    }

    ////////////////////////////////////////////////
    // 读取矩阵市场文件主体
    // 从文件获取数据块，逐块读取
    ///////////////////////////////////////////////
}

#include "read_body_threads.hpp"

namespace fast_matrix_market {

    template <typename HANDLER>
    line_counts read_coordinate_body_sequential(std::istream& instream, const matrix_market_header& header,
                                                HANDLER& handler, const read_options& options = {}) {
        // 初始化行数统计信息，包含文件头行数和读取行数
        line_counts lc{header.header_line_count, 0};

        // 循环读取文件的数据块
        while (instream.good()) {
            // 获取下一个数据块
            std::string chunk = get_next_chunk(instream, options);

            // 解析数据块
            if (header.object == matrix) {
                // 如果是矩阵对象，则调用适当的函数处理数据块
                lc = read_chunk_matrix_coordinate(chunk, header, lc, handler, options);
            } else {
#ifdef FMM_NO_VECTOR
                // 如果不支持向量矩阵市场文件，则抛出异常
                throw no_vector_support("Vector Matrix Market files not supported.");
    // 如果未定义 FMM_NO_VECTOR，则执行以下代码块
    else {
        // 从输入流中读取顺序数组的主体内容，并处理每个数据块
        read_array_body_sequential(instream, header, handler, options);
    }
#endif
#endif
// 如果定义了 #endif，则表示当前代码段属于某个条件编译块的结束

        // 输入的合法性检查
        // 如果对象是向量，但对称性不是 general，则抛出异常
        if (header.object == vector && header.symmetry != general) {
            throw invalid_mm("Vectors cannot have symmetry.");
        }

        // 如果格式是 array，并且字段是 pattern，则抛出异常
        if (header.format == array && header.field == pattern) {
            throw invalid_mm("Array matrices may not be pattern.");
        }

        // 声明 line_counts 对象 lc
        line_counts lc;
        // 检查是否可以并行处理，条件为选项允许并行、线程数不为 1，并且处理器标志允许并行
        bool threads = options.parallel_ok && options.num_threads != 1 && test_flag(HANDLER::flags, kParallelOk);

        // 根据值类型限制并行处理的能力
        threads = limit_parallelism_for_value_type<typename HANDLER::value_type>(threads);

        // 如果对称性不是 general 并且格式是 array，则禁用并行处理
        if (header.symmetry != general && header.format == array) {
            // 并行数组加载器不处理对称性
            threads = false;
        }

        // 如果格式是 coordinate 并且处理器标志指定为 kDense，则存在潜在的竞态条件
        if (header.format == coordinate && test_flag(HANDLER::flags, kDense)) {
            // 如果文件包含重复项，则禁用并行处理
            threads = false;
        }

        // 根据并行处理标志选择相应的读取方式
        if (threads) {
            // 使用多线程读取数据
            lc = read_body_threads<HANDLER, FORMAT>(instream, header, handler, options);
        } else {
            // 不使用多线程读取数据
            if (header.format == coordinate) {
                // 如果格式是 coordinate
                if constexpr ((FORMAT & compile_coordinate_only) == compile_coordinate_only) {
                    // 如果 FORMAT 包含 compile_coordinate_only，顺序读取坐标数据
                    lc = read_coordinate_body_sequential(instream, header, handler, options);
                } else {
                    // 否则抛出不支持的异常
                    throw support_not_selected("Matrix is coordinate but reading coordinate files not enabled for this method.");
                }
            } else {
                // 如果格式不是 coordinate
                if constexpr ((FORMAT & compile_array_only) == compile_array_only) {
                    // 如果 FORMAT 包含 compile_array_only，顺序读取数组数据
                    lc = read_array_body_sequential(instream, header, handler, options);
                } else {
                    // 否则抛出不支持的异常
                    throw support_not_selected("Matrix is array but reading array files not enabled for this method.");
                }
            }
        }

        // 验证文件是否截断
        if (lc.element_num < header.nnz) {
            // 如果实际读取元素数量小于 header 中声明的非零元素数量 nnz
            if (!(header.symmetry != general && header.format == array)) {
                // 除非对称性不是 general 并且格式是 array，否则抛出截断文件异常
                throw invalid_mm(std::string("Truncated file. Expected another ") +
                                 std::to_string(header.nnz - lc.element_num) + " lines.");
            }
        }
    }

#ifndef FMM_SCIPY_PRUNE
    /**
     * Read the body by adapting real files to complex HANDLER.
     *
     * @deprecated Use read_matrix_market_body_no_adapters() directly. It now handles the adaptation that this method does.
     */
    // 模板函数声明，用于读取实际文件到复数类型 HANDLER
    template <typename HANDLER, typename std::enable_if<is_complex<typename HANDLER::value_type>::value, int>::type = 0>
    [[deprecated]] [[maybe_unused]]
    void read_matrix_market_body_no_pattern(std::istream& instream, const matrix_market_header& header,
                                            HANDLER& handler, const read_options& options = {}) {
        // 如果文件字段为复数，调用适用于复数的处理函数
        if (header.field == complex) {
            read_matrix_market_body_no_adapters(instream, header, handler, options);
        } else {
            // 处理程序期望 std::complex 值，但文件只有整数/实数
            // 提供适配器
            auto fwd_handler = complex_parse_adapter<HANDLER>(handler);
            // 调用适用于复数的处理函数，使用适配器处理实数/整数数据
            read_matrix_market_body_no_adapters(instream, header, fwd_handler, options);
        }
    }
    
    /**
     * Read the body by adapting real files to complex HANDLER.
     *
     * @deprecated 直接使用 read_matrix_market_body_no_adapters()。现在它处理了此方法执行的适配。
     */
    template <typename HANDLER, typename std::enable_if<!is_complex<typename HANDLER::value_type>::value, int>::type = 0>
    [[deprecated]] [[maybe_unused]]
    void read_matrix_market_body_no_pattern(std::istream& instream, const matrix_market_header& header,
                                            HANDLER& handler, const read_options& options = {}) {
        // 如果文件字段不是复数，调用不需要适配的处理函数
        if (header.field != complex) {
            read_matrix_market_body_no_adapters(instream, header, handler, options);
        } else {
            // 文件是复数，但值不是复数
            throw complex_incompatible("Matrix Market file has complex fields but passed data structure cannot handle complex values.");
        }
    }
#endif

/**
 * Main body reader entry point.
 *
 * This function serves as the main entry point for reading the body of a Matrix Market file.
 * It automatically handles adaptations based on the file's characteristics:
 *  - If the file is a pattern file, each element will be substituted with 'pattern_value'
 *  - If 'HANDLER' expects std::complex values but the file is not complex, imag=0 is provided for each value.
 */
template <typename HANDLER, compile_format FORMAT = compile_all>
void read_matrix_market_body(std::istream& instream, const matrix_market_header& header,
                             HANDLER& handler,
                             typename HANDLER::value_type pattern_value,
                             const read_options& options = {}) {
    // Check if the file is complex but the handler cannot handle complex values
    if (header.field == complex && !can_read_complex<typename HANDLER::value_type>::value) {
        // Throw an exception indicating incompatibility
        throw complex_incompatible("Matrix Market file has complex fields but passed data structure cannot handle complex values.");
    }

    // Adapt the handler to handle pattern values if necessary
    auto fwd_handler = pattern_parse_adapter<HANDLER>(handler, pattern_value);

    // Call the function to read the Matrix Market file body with the adapted handler
    read_matrix_market_body_no_adapters<decltype(fwd_handler), FORMAT>(instream, header, fwd_handler, options);
}
```