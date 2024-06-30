# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\fast_matrix_market\include\fast_matrix_market\field_conv.hpp`

```
// 版权声明和许可证信息，指明代码归Adam Lugowski所有，受BSD 2-Clause许可证管理
// SPDX-License-Identifier: BSD-2-Clause
#pragma once

#include <charconv>         // 提供了用于数字转换的标准库函数
#include <cmath>            // 提供了数学函数，如数学运算和特殊函数
#include <cstring>          // 提供了 C 字符串操作函数
#include <complex>          // 提供了复数操作支持
#include <limits>           // 提供了数字类型的极值信息
#include <iomanip>          // 提供了格式化输入输出的库函数和参数设置
#include <type_traits>      // 提供了模板元编程的支持，用于类型特性的查询和转换

#ifdef FMM_USE_FAST_FLOAT
#include <fast_float/fast_float.h>  // 使用了快速浮点数解析的第三方库
#endif

#ifdef FMM_USE_DRAGONBOX
#include <dragonbox/dragonbox_to_chars.h>  // 使用了 Dragonbox 算法的第三方库，用于浮点数转字符
#endif

#ifdef FMM_USE_RYU
#include <ryu/ryu.h>         // 使用了 Ryu 算法的第三方库，用于快速浮点数转字符
#endif

#include "fast_matrix_market.hpp"  // 引入自定义的头文件，可能包含了特定的矩阵市场处理逻辑

namespace fast_matrix_market {
    ///////////////////////////////////////////
    // 空白字符管理
    ///////////////////////////////////////////

    inline const char* skip_spaces(const char* pos) {
        return pos + std::strspn(pos, " \t\r");  // 跳过 pos 指向的字符串中的空白字符
    }

    inline const char* skip_spaces_and_newlines(const char* pos, int64_t& line_num) {
        pos = skip_spaces(pos);  // 跳过空白字符
        while (*pos == '\n') {   // 处理换行符
            ++line_num;          // 增加行号计数
            ++pos;               // 指针移动到下一个字符
            pos = skip_spaces(pos);  // 继续跳过空白字符
        }
        return pos;
    }

    inline const char* bump_to_next_line(const char* pos, const char* end) {
        if (pos == end) {
            return pos;  // 如果已经到达末尾，直接返回当前位置
        }

        pos = std::strchr(pos, '\n');  // 在字符串中查找下一个换行符的位置

        if (pos != end) {
            ++pos;  // 移动到下一行的开始位置
        }
        return pos;
    }

    ///////////////////////////////////////////
    // 整数 / 浮点数字段解析器
    ///////////////////////////////////////////

#ifdef FMM_FROM_CHARS_INT_SUPPORTED
    /**
     * 使用 std::from_chars 解析整数
     */
    template <typename IT>
    const char* read_int_from_chars(const char* pos, const char* end, IT& out) {
        std::from_chars_result result = std::from_chars(pos, end, out);  // 调用 std::from_chars 进行解析
        if (result.ec != std::errc()) {  // 如果解析结果出错
            if (result.ec == std::errc::result_out_of_range) {
                throw out_of_range("Integer out of range.");  // 抛出范围溢出异常
            } else {
                throw invalid_mm("Invalid integer value.");  // 抛出无效整数值异常
            }
        }
        return result.ptr;  // 返回解析后的指针位置
    }
#endif

    // 兼容旧标准的整数解析器，使用 std::strtoll 进行解析
    inline const char* read_int_fallback(const char* pos, [[maybe_unused]] const char* end, long long& out) {
        errno = 0;  // 清除错误号

        char* value_end;
        out = std::strtoll(pos, &value_end, 10);  // 使用 std::strtoll 解析整数
        if (errno != 0 || pos == value_end) {  // 如果解析出错或者解析后位置未移动
            if (errno == ERANGE) {
                throw out_of_range("Integer out of range.");  // 抛出范围溢出异常
            } else {
                throw invalid_mm("Invalid integer value.");  // 抛出无效整数值异常
            }
        }

        return value_end;  // 返回解析后的位置
    }
}
    // 设置 errno 为 0，准备读取整数
    inline const char* read_int_fallback(const char* pos, [[maybe_unused]] const char* end, unsigned long long& out) {
        errno = 0;

        // 初始化变量 value_end，用于记录解析整数后的位置
        char *value_end;
        // 使用 std::strtoull 函数解析字符串 pos 开始的整数，并将结果保存到 out 中
        out = std::strtoull(pos, &value_end, 10);
        // 检查是否发生错误或者未解析出有效整数
        if (errno != 0 || pos == value_end) {
            // 如果发生范围错误，抛出 out_of_range 异常
            if (errno == ERANGE) {
              throw out_of_range("Integer out of range.");
            } else {
                // 否则抛出 invalid_mm 异常，表示整数值无效
                throw invalid_mm("Invalid integer value.");
            }
        }
        // 返回解析结束位置 value_end
        return value_end;
    }

    /**
     * 使用 C 标准库函数解析整数。
     *
     * 这是一种兼容性的后备方案。
     */
    template <typename T>
    // 根据模板参数 T 解析整数
    const char* read_int_fallback(const char* pos, const char* end, T& out) {
        // 定义一个 long long 类型的变量 i64，用于存储解析后的整数值
        long long i64;
        // 调用上面的函数 read_int_fallback 解析整数，并获取解析结束位置
        const char* ret = read_int_fallback(pos, end, i64);

        // 如果 T 类型占用的字节数小于 long long 类型，则进行进一步检查
        if (sizeof(T) < sizeof(long long)) {
            // 如果解析得到的 long long 值超出了 T 类型的表示范围，则抛出范围错误异常
            if (i64 > (long long) std::numeric_limits<T>::max() ||
                i64 < (long long) std::numeric_limits<T>::min()) {
                throw out_of_range(std::string("Integer out of range."));
            }
        }
        // 将解析得到的 long long 值转换为 T 类型，并赋值给 out
        out = static_cast<T>(i64);
        // 返回解析结束位置 ret
        return ret;
    }

    /**
     * 使用最佳可用方法解析整数
     */
    template <typename IT>
    // 根据模板参数 IT 解析整数
    const char* read_int(const char* pos, const char* end, IT& out) {
#ifdef FMM_FROM_CHARS_INT_SUPPORTED
        // 如果支持从字符中解析整数，则调用对应函数
        return read_int_from_chars(pos, end, out);
#else
        // 否则，调用整数解析的后备函数
        return read_int_fallback(pos, end, out);
#endif
    }

#ifdef FMM_USE_FAST_FLOAT
    /**
     * 使用 fast_float::from_chars 解析 float 或 double
     */
    template <typename FT>
    const char* read_float_fast_float(const char* pos, const char* end, FT& out, out_of_range_behavior oorb) {
        // 调用 fast_float::from_chars 进行解析
        fast_float::from_chars_result result = fast_float::from_chars(pos, end, out, fast_float::chars_format::general);

        // 检查解析结果的错误码
        if (result.ec != std::errc()) {
            // 如果解析结果指示数值超出范围
            if (result.ec == std::errc::result_out_of_range) {
                // 根据指定的超出范围行为处理异常
                if (oorb == ThrowOutOfRange) {
                    throw out_of_range("Floating-point value out of range.");
                }
            } else {
                // 否则抛出无效数值异常
                throw invalid_mm("Invalid floating-point value.");
            }
        }
        // 返回解析后的指针位置
        return result.ptr;
    }
#endif


#ifdef FMM_FROM_CHARS_DOUBLE_SUPPORTED
    /**
     * 使用 std::from_chars 解析 float 或 double
     */
    template <typename FT>
    const char* read_float_from_chars(const char* pos, const char* end, FT& out, out_of_range_behavior oorb) {
        // 调用 std::from_chars 进行解析
        std::from_chars_result result = std::from_chars(pos, end, out);

        // 检查解析结果的错误码
        if (result.ec != std::errc()) {
            // 如果解析结果指示数值超出范围
            if (result.ec == std::errc::result_out_of_range) {
                // 根据指定的超出范围行为处理异常
                if (oorb == ThrowOutOfRange) {
                    throw out_of_range("Floating-point overflow");
                } else {
                    // 如果不抛出异常，则回退到 strtod 函数进行解析
                    out = static_cast<FT>(std::strtod(pos, nullptr));
                }
            } else {
                // 否则抛出无效数值异常
                throw invalid_mm("Invalid floating-point value.");
            }
        }
        // 返回解析后的指针位置
        return result.ptr;
    }
#endif

    /**
     * 使用 strtod() 解析 double。这是一个兼容性后备方案。
     */
    inline const char* read_float_fallback(const char* pos, [[maybe_unused]] const char* end, double& out, out_of_range_behavior oorb = ThrowOutOfRange) {
        // 清空 errno
        errno = 0;

        // 调用 strtod 函数解析 double
        char* value_end;
        out = std::strtod(pos, &value_end);

        // 检查解析过程中是否出现错误
        if (errno != 0 || pos == value_end) {
            // 如果 errno 表示数值超出范围
            if (errno == ERANGE) {
                // 根据指定的超出范围行为处理异常
                if (oorb == ThrowOutOfRange) {
                    throw out_of_range("Floating-point value out of range.");
                }
            } else {
                // 否则抛出无效数值异常
                throw invalid_mm("Invalid floating-point value.");
            }
        }
        // 返回解析后的指针位置
        return value_end;
    }

    /**
     * 使用 strtof() 解析 float。这是一个兼容性后备方案。
     */
    // 从输入位置 pos 开始尝试解析一个浮点数，并将解析结果存入 out 中
    inline const char* read_float_fallback(const char* pos, [[maybe_unused]] const char* end, float& out, out_of_range_behavior oorb) {
        // 设置错误号为 0，以便检测解析过程中的错误
        errno = 0;

        // 定义变量 value_end 用于存储解析成功后的结束位置
        char* value_end;

        // 使用 std::strtof 函数尝试将 pos 处的字符串转换为浮点数，并将结果存入 out
        out = std::strtof(pos, &value_end);

        // 如果 errno 不为 0，或者 pos 和 value_end 指向相同位置，则表示解析出现错误
        if (errno != 0 || pos == value_end) {
            // 如果错误类型为 ERANGE（表示浮点数超出范围），并且 oorb 参数为 ThrowOutOfRange，则抛出异常
            if (errno == ERANGE) {
                if (oorb == ThrowOutOfRange) {
                    throw out_of_range("Floating-point value out of range.");
                }
            } else {
                // 否则，抛出无效浮点数值异常
                throw invalid_mm("Invalid floating-point value.");
            }
        }

        // 返回解析成功后的结束位置 value_end
        return value_end;
    }

    // 模板函数，用于解析不同类型的浮点数
    template <typename FT>
    const char* read_float(const char* pos, const char* end, FT& out, out_of_range_behavior oorb) {
        // constexpr 变量，用于判断是否支持快速浮点数解析
        constexpr bool have_fast_float =
#ifdef FMM_USE_FAST_FLOAT
        true;  // 如果定义了 FMM_USE_FAST_FLOAT 宏，则此处是真值
#else
        false;  // 如果未定义 FMM_USE_FAST_FLOAT 宏，则此处是假值
#endif

        // 如果 have_fast_float 为 true，且 FT 是 float 或 double 类型之一，则调用 read_float_fast_float 函数
        if constexpr (have_fast_float && (std::is_same_v<FT, float> || std::is_same_v<FT, double>)) {
            return read_float_fast_float(pos, end, out, oorb);
        } else {
#if defined(FMM_FROM_CHARS_DOUBLE_SUPPORTED)
            // 调用 read_float_from_chars 函数处理浮点数解析（双精度）
            return read_float_from_chars(pos, end, out, oorb);
#else
            // 调用 read_float_fallback 函数处理浮点数解析（回退）
            return read_float_fallback(pos, end, out, oorb);
#endif
        }
    }

#ifdef FMM_FROM_CHARS_LONG_DOUBLE_SUPPORTED
    /**
     * 使用 std::from_chars 解析 long double 类型的浮点数
     */
    inline const char* read_float_from_chars(const char* pos, const char* end, long double& out, out_of_range_behavior oorb) {
        std::from_chars_result result = std::from_chars(pos, end, out);
        if (result.ec != std::errc()) {
            if (result.ec == std::errc::result_out_of_range) {
                if (oorb == ThrowOutOfRange) {
                    throw out_of_range("Floating-point value out of range.");
                } else {
                    // std::from_chars 在下溢/上溢时不返回最佳匹配，因此回退到 strtold
                    out = std::strtold(pos, nullptr);
                }
            } else {
                throw invalid_mm("Invalid floating-point value.");
            }
        }
        return result.ptr;
    }
#endif

    /**
     * 使用 std::strtold 解析 long double 类型的浮点数
     *
     * fast_float 不支持 long double 类型
     */
    inline const char* read_float_fallback(const char* pos, [[maybe_unused]] const char* end, long double& out, out_of_range_behavior oorb) {
        errno = 0;

        char* value_end;
        out = std::strtold(pos, &value_end);
        if (errno != 0 || pos == value_end) {
            if (errno == ERANGE) {
                if (oorb == ThrowOutOfRange) {
                    throw out_of_range("Floating-point value out of range.");
                }
            } else {
                throw invalid_mm("Invalid floating-point value.");
            }
        }
        return value_end;
    }

    /**
     * 解析浮点数，根据宏定义选择具体的解析方式
     */
    inline const char* read_float(const char* pos, [[maybe_unused]] const char* end, long double& out, out_of_range_behavior oorb) {
#ifdef FMM_FROM_CHARS_LONG_DOUBLE_SUPPORTED
        return read_float_from_chars(pos, end, out, oorb);
#else
        return read_float_fallback(pos, end, out, oorb);
#endif
    }

    //////////////////////////////////////
    // 读取值。根据请求的类型，这些将评估为相应的字段解析器
    //////////////////////////////////////

    /**
     * 模式值是空操作
     */
    inline const char* read_value(const char* pos, [[maybe_unused]] const char* end, [[maybe_unused]] pattern_placeholder_type& out, [[maybe_unused]] const read_options& options = {}) {
        return pos;
    }

    template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
    // 从字符数组中读取一个值并转换为指定类型 T，更新 pos 指针并返回新的 pos 指针
    const char* read_value(const char* pos, const char* end, T& out, [[maybe_unused]] const read_options& options = {}) {
        return read_int(pos, end, out);
    }

    // 从字符数组中读取一个布尔值，并根据解析的浮点数设置 out 的值，更新 pos 指针并返回新的 pos 指针
    inline const char* read_value(const char* pos, const char* end, bool& out, const read_options& options = {}) {
        double parsed;
        // 调用 read_float 函数从 pos 开始读取浮点数，更新 parsed，根据 parsed 的值设置 out
        auto ret = read_float(pos, end, parsed, options.float_out_of_range_behavior);
        out = (parsed != 0);
        return ret;
    }

    // 从字符数组中读取一个浮点数值，并更新 pos 指针，返回新的 pos 指针
    template <typename T, typename std::enable_if<std::is_floating_point_v<T>, int>::type = 0>
    const char* read_value(const char* pos, const char* end, T& out, const read_options& options = {}) {
        return read_float(pos, end, out, options.float_out_of_range_behavior);
    }

    // 从字符数组中读取一个复数值 COMPLEX，并更新 pos 指针，返回新的 pos 指针
    template <typename COMPLEX, typename std::enable_if<is_complex<COMPLEX>::value, int>::type = 0>
    const char* read_value(const char* pos, const char* end, COMPLEX& out, const read_options& options = {}) {
        typename COMPLEX::value_type real, imaginary;
        // 从 pos 开始读取实部 real，更新 pos 指针
        pos = read_float(pos, end, real, options.float_out_of_range_behavior);
        // 跳过空格
        pos = skip_spaces(pos);
        // 从 pos 开始读取虚部 imaginary，更新 pos 指针
        pos = read_float(pos, end, imaginary, options.float_out_of_range_behavior);

        // 将读取到的实部和虚部设置到复数对象 out 中
        out.real(real);
        out.imag(imaginary);

        return pos;
    }

    // 计算复数类型 T 的共轭复数
    template <typename T, typename std::enable_if<is_complex<T>::value, int>::type = 0>
    T complex_conjugate(const T& value) {
        return T(value.real(), -value.imag());
    }

    // 对非复数类型 T，返回其本身
    template <typename T, typename std::enable_if<!is_complex<T>::value, int>::type = 0>
    T complex_conjugate(const T& value) {
        return value;
    }
#ifdef FMM_TO_CHARS_INT_SUPPORTED
    /**
     * Convert integral types to string.
     * std::to_string and std::to_chars has similar performance, however std::to_string is locale dependent and
     * therefore will cause thread serialization.
     */
    // 如果定义了 FMM_TO_CHARS_INT_SUPPORTED 宏，则使用此函数将整数类型转换为字符串
    template <typename T>
    std::string int_to_string(const T& value) {
        // 创建一个长度为 20 的字符串，填充空格，用于存储转换后的字符串
        std::string ret(20, ' ');
        // 使用 std::to_chars 尝试将 value 转换为字符串，并将结果存储在 ret 中
        std::to_chars_result result = std::to_chars(ret.data(), ret.data() + ret.size(), value);
        // 如果转换成功，缩减字符串长度到实际需要的长度并返回
        if (result.ec == std::errc()) {
            ret.resize(result.ptr - ret.data());
            return ret;
        } else {
            // 转换失败时，回退到 std::to_string 进行转换并返回结果
            return std::to_string(value);
        }
    }
#else
    /**
     * Convert integral types to string. This is the fallback due to locale dependence (and hence thread serialization).
     */
    // 如果未定义 FMM_TO_CHARS_INT_SUPPORTED 宏，则使用此函数将整数类型转换为字符串
    template <typename T>
    std::string int_to_string(const T& value) {
        // 直接使用 std::to_string 进行转换并返回结果
        return std::to_string(value);
    }
#endif

    // 以下两个函数对特定类型的参数进行字符串转换，但是在此上下文中未被使用
    inline std::string value_to_string([[maybe_unused]] const pattern_placeholder_type& value, [[maybe_unused]] int precision) {
        return {};
    }

    inline std::string value_to_string(const bool & value, [[maybe_unused]] int precision) {
        // 将布尔值转换为 "1" 或 "0" 的字符串表示并返回
        return value ? "1" : "0";
    }

    template <typename T, typename std::enable_if<std::is_integral_v<T>, int>::type = 0>
    std::string value_to_string(const T& value, [[maybe_unused]] int precision) {
        // 对整数类型的参数使用 int_to_string 函数进行转换并返回结果
        return int_to_string(value);
    }

    /**
     * stdlib fallback
     */
    // 通用的值转换为字符串的函数，根据不同情况采用不同的策略进行转换
    template <typename T>
    std::string value_to_string_fallback(const T& value, int precision) {
        if (precision < 0) {
            // 当精度小于零时，选择最短表示法
            if constexpr (std::is_floating_point_v<T>
                            && !std::is_same_v<T, float>
                            && !std::is_same_v<T, double>
                            && !std::is_same_v<T, long double>) {
                // 对于特定的浮点数类型，使用 std::ostringstream 进行转换并返回字符串
                std::ostringstream oss;
                oss << value;
                return oss.str();
            } else {
                // 对于其他类型，使用 std::to_string 进行转换并返回字符串
                return std::to_string(value);
            }
        } else {
            // 当精度大于等于零时，设置输出精度并使用 std::ostringstream 进行转换并返回字符串
            std::ostringstream oss;
            oss << std::setprecision(precision) << value;
            return oss.str();
        }
    }

// Sometimes Dragonbox and Ryu may render '1' as '1E0'
// This controls whether to truncate those suffixes.
// 有时 Dragonbox 和 Ryu 可能会将 '1' 渲染为 '1E0'，这个宏控制是否截断这些后缀
#ifndef FMM_DROP_ENDING_E0
#define FMM_DROP_ENDING_E0 1
#endif

// Same as above, but for context where precision is specified
// 类似上面的宏定义，但适用于指定精度的上下文
#ifndef FMM_DROP_ENDING_E0_PRECISION
#define FMM_DROP_ENDING_E0_PRECISION 0
#endif

#ifdef FMM_USE_DRAGONBOX

    // 如果定义了 FMM_USE_DRAGONBOX 宏，则使用此函数将浮点数类型转换为字符串
    inline std::string value_to_string_dragonbox(const float& value) {
        // 创建一个足够大的缓冲区用于 Dragonbox 的输出
        std::string buffer(jkj::dragonbox::max_output_string_length<jkj::dragonbox::ieee754_binary32> + 1, ' ');

        // 使用 Dragonbox 的 to_chars 函数将浮点数值转换为字符串，并返回转换后的结果
        char *end_ptr = jkj::dragonbox::to_chars(value, buffer.data());
        buffer.resize(end_ptr - buffer.data());
        return buffer;
#if FMM_DROP_ENDING_E0
        // 如果定义了 FMM_DROP_ENDING_E0 宏
        if (ends_with(buffer, "E0")) {
            // 如果字符串 buffer 以 "E0" 结尾
            buffer.resize(buffer.size() - 2);
            // 则删除末尾的两个字符
        }
#endif
        // 返回处理后的字符串 buffer
        return buffer;
    }

    inline std::string value_to_string_dragonbox(const double& value) {
        // 创建一个字符串 buffer，其长度为 dragonbox 库支持的最大输出长度加一，并用空格填充
        std::string buffer(jkj::dragonbox::max_output_string_length<jkj::dragonbox::ieee754_binary64> + 1, ' ');

        // 将双精度浮点数 value 转换成字符串，并将结果存储到 buffer 中
        char *end_ptr = jkj::dragonbox::to_chars(value, buffer.data());
        // 调整 buffer 的大小，使其适合实际使用的字符数
        buffer.resize(end_ptr - buffer.data());

#if FMM_DROP_ENDING_E0
        // 如果定义了 FMM_DROP_ENDING_E0 宏
        if (ends_with(buffer, "E0")) {
            // 如果字符串 buffer 以 "E0" 结尾
            buffer.resize(buffer.size() - 2);
            // 则删除末尾的两个字符
        }
#endif
        // 返回处理后的字符串 buffer
        return buffer;
    }
#endif

#ifdef FMM_USE_RYU
    inline std::string value_to_string_ryu(const float& value, int precision) {
        // 创建一个长度为 16 的字符串 ret，并用空格填充
        std::string ret(16, ' ');

        if (precision < 0) {
            // 如果精度小于零，选择最短表示法
            auto len = f2s_buffered_n(value, ret.data());
            // 将浮点数 value 转换成字符串并存储到 ret 中
            ret.resize(len);

#if FMM_DROP_ENDING_E0
            // 如果定义了 FMM_DROP_ENDING_E0 宏
            if (ends_with(ret, "E0")) {
                // 如果字符串 ret 以 "E0" 结尾
                ret.resize(ret.size() - 2);
                // 则删除末尾的两个字符
            }
#endif
        } else {
            // 显示指定精度
            if (precision > 0) {
                // 调整精度，将其解释为有效数字的位数
                --precision;
            }
            // 将单精度浮点数 value 转换成字符串并存储到 ret 中
            auto len = d2exp_buffered_n(static_cast<double>(value), precision, ret.data());
            // 调整 ret 的大小，使其适合实际使用的字符数
            ret.resize(len);

#if FMM_DROP_ENDING_E0_PRECISION
            // 如果定义了 FMM_DROP_ENDING_E0_PRECISION 宏
            if (ends_with(ret, "e+00")) {
                // 如果字符串 ret 以 "e+00" 结尾
                ret.resize(ret.size() - 4);
                // 则删除末尾的四个字符
            }
#endif
        }

        // 返回处理后的字符串 ret
        return ret;
    }

    inline std::string value_to_string_ryu(const double& value, int precision) {
        // 创建一个长度为 26 的字符串 ret，并用空格填充
        std::string ret(26, ' ');

        if (precision < 0) {
            // 如果精度小于零，选择最短表示法
            auto len = d2s_buffered_n(value, ret.data());
            // 将双精度浮点数 value 转换成字符串并存储到 ret 中
            ret.resize(len);

#if FMM_DROP_ENDING_E0
            // 如果定义了 FMM_DROP_ENDING_E0 宏
            if (ends_with(ret, "E0")) {
                // 如果字符串 ret 以 "E0" 结尾
                ret.resize(ret.size() - 2);
                // 则删除末尾的两个字符
            }
#endif
        } else {
            // 显示指定精度
            if (precision > 0) {
                // 调整精度，将其解释为有效数字的位数
                --precision;
            }
            // 将双精度浮点数 value 转换成字符串并存储到 ret 中
            auto len = d2exp_buffered_n(value, precision, ret.data());
            // 调整 ret 的大小，使其适合实际使用的字符数
            ret.resize(len);

#if FMM_DROP_ENDING_E0_PRECISION
            // 如果定义了 FMM_DROP_ENDING_E0_PRECISION 宏
            if (ends_with(ret, "e+00")) {
                // 如果字符串 ret 以 "e+00" 结尾
                ret.resize(ret.size() - 4);
                // 则删除末尾的四个字符
            }
#endif
        }

        // 返回处理后的字符串 ret
        return ret;
    }
#endif

#ifdef FMM_TO_CHARS_DOUBLE_SUPPORTED
    // 对于浮点数类型 T，仅当 T 是浮点数时才启用该模板
    template <typename T, typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
    // 将值转换为字符串，结果作为字符序列返回
    inline std::string value_to_string_to_chars(const T& value, int precision) {
        // 创建一个初始长度为100的字符串，填充空格
        std::string ret(100, ' ');
        // 存储 std::to_chars 的结果
        std::to_chars_result result{};
        
        // 如果 precision 小于 0，使用最短的表示方式
        if (precision < 0) {
            // 调用 std::to_chars 将 value 转换为字符串
            result = std::to_chars(ret.data(), ret.data() + ret.size(), value);
        } else {
            // 使用显式的精度参数调用 std::to_chars，格式为一般格式
            result = std::to_chars(ret.data(), ret.data() + ret.size(), value, std::chars_format::general, precision);
        }
        
        // 检查转换结果是否成功
        if (result.ec == std::errc()) {
            // 调整字符串大小以匹配实际转换结果的长度
            ret.resize(result.ptr - ret.data());
            // 返回转换后的字符串
            return ret;
        } else {
            // 如果转换失败，调用备用的字符串转换方法
            return value_to_string_fallback(value, precision);
        }
    }
#ifdef FMM_TO_CHARS_LONG_DOUBLE_SUPPORTED
    // 将 long double 类型的值转换为字符串
    inline std::string value_to_string_to_chars(const long double& value, int precision) {
        // 创建一个长度为 50 的字符串，用空格填充
        std::string ret(50, ' ');
        // 存储 std::to_chars 的调用结果
        std::to_chars_result result{};
        if (precision < 0) {
            // 使用最短的表示形式转换 long double 值为字符串
            result = std::to_chars(ret.data(), ret.data() + ret.size(), value);
        } else {
            // 使用指定精度转换 long double 值为字符串
            result = std::to_chars(ret.data(), ret.data() + ret.size(), value, std::chars_format::general, precision);
        }
        // 检查转换是否成功
        if (result.ec == std::errc()) {
            // 调整字符串大小，使其匹配转换结果的实际长度，并返回
            ret.resize(result.ptr - ret.data());
            return ret;
        } else {
            // 转换失败时，使用备用方法处理
            return value_to_string_fallback(value, precision);
        }
    }
#endif

    /**
     * 将 long double 类型的值转换为字符串。
     *
     * 首选顺序：to_chars，备选。
     * 注意：在某些平台上，Ryu 的 generic_128 可能能够执行此操作，但不是所有平台都可靠。
     * 参见 https://github.com/ulfjack/ryu/issues/215
     */
    inline std::string value_to_string(const long double& value, int precision) {
#if defined(FMM_TO_CHARS_LONG_DOUBLE_SUPPORTED)
        // 如果支持 to_chars 转换，使用该方法
        return value_to_string_to_chars(value, precision);
#else
        // 否则使用备选方法处理
        return value_to_string_fallback(value, precision);
#endif
    }


    /**
     * 将浮点数类型的值转换为字符串。
     *
     * 首选顺序：Dragonbox（快速但不支持精度控制）、to_chars、Ryu、备选。
     *
     * Dragonbox 和 Ryu 只支持 float 和 double 类型。
     */
    template <typename T, typename std::enable_if<std::is_floating_point_v<T> && !std::is_same_v<T, long double>, int>::type = 0>
    std::string value_to_string(const T& value, int precision) {
#ifdef FMM_USE_DRAGONBOX
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            if (precision < 0) {
                // 使用 Dragonbox 提供的最短表示形式转换浮点数值为字符串
                return value_to_string_dragonbox(value);
            }
        }
#endif

#ifdef FMM_TO_CHARS_DOUBLE_SUPPORTED
        // 使用 to_chars 方法将浮点数值转换为字符串
        return value_to_string_to_chars(value, precision);
#else
        // 定义是否使用 Ryu 的标志
        constexpr bool have_ryu =
#ifdef FMM_USE_RYU
            true;
#else
            false;
#endif

        if constexpr (have_ryu && (std::is_same_v<T, float> || std::is_same_v<T, double>)) {
            // 如果使用 Ryu，并且值为 float 或 double 类型，则使用 Ryu 方法转换为字符串
            return value_to_string_ryu(value, precision);
        } else {
            // 其他情况下使用备选方法处理
            return value_to_string_fallback(value, precision);
        }
#endif
    }

    template <typename COMPLEX, typename std::enable_if<is_complex<COMPLEX>::value, int>::type = 0>
    inline std::string value_to_string(const COMPLEX& value, int precision) {
        // 对复数类型进行处理，将实部和虚部转换为字符串并连接起来
        return value_to_string(value.real(), precision) + " " + value_to_string(value.imag(), precision);
    }

    /**
     * 捕获所有类型的转换需求
     */
    template <typename T, typename std::enable_if<!std::is_integral_v<T> && !std::is_floating_point_v<T> && !is_complex<T>::value, int>::type = 0>
    // 定义函数 `value_to_string`，接收模板类型 T 的值 `value` 和精度 `precision`，返回其字符串表示
    std::string value_to_string(const T& value, int precision) {
        // 调用 `value_to_string_fallback` 函数，将 `value` 和 `precision` 作为参数传递，返回结果
        return value_to_string_fallback(value, precision);
    }
ultimate
```