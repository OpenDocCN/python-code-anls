# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\fast_matrix_market\include\fast_matrix_market\fast_matrix_market.hpp`

```
// 版权声明和许可声明，指明此代码的版权归Adam Lugowski所有，遵循BSD 2-Clause许可
// SPDX-License-Identifier: BSD-2-Clause
// 声明如果 FAST_MATRIX_MARKET_H 未定义，则定义它，以避免头文件重复包含
#ifndef FAST_MATRIX_MARKET_H
#define FAST_MATRIX_MARKET_H

// 确保此头文件只被编译一次
#pragma once

// 包含所需的标准库头文件
#include <algorithm>    // 包含用于算法操作的标准库
#include <future>       // 包含用于并行任务的标准库
#include <iostream>     // 包含用于标准输入输出的标准库
#include <map>          // 包含用于映射容器的标准库
#include <string>       // 包含用于字符串操作的标准库
#include <sstream>      // 包含用于字符串流操作的标准库
#include <utility>      // 包含用于实用工具的标准库
#include <vector>       // 包含用于向量容器的标准库

// 包含自定义类型头文件
#include "types.hpp"

// 支持 std::string 作为用户自定义类型
#include "app/user_type_string.hpp"

// fast_matrix_market 命名空间，用于封装所有相关的类和常量
namespace fast_matrix_market {

    // 版本宏定义
    // 与 python/pyproject.toml 中的版本号保持同步
#define FAST_MATRIX_MARKET_VERSION_MAJOR 1
#define FAST_MATRIX_MARKET_VERSION_MINOR 7
#define FAST_MATRIX_MARKET_VERSION_PATCH 4

    // constexpr 字符串视图常量定义
    constexpr std::string_view kSpace = " ";    // 空格字符常量
    constexpr std::string_view kNewline = "\n"; // 换行字符常量

    /**
     * fmm_error 类：自定义异常类，继承自 std::exception
     */
    class fmm_error : public std::exception {
    public:
        explicit fmm_error(std::string msg): msg(std::move(msg)) {} // 构造函数，接收错误信息字符串

        // 重写 what() 函数，返回异常信息字符串的 C 风格字符指针
        [[nodiscard]] const char* what() const noexcept override {
            return msg.c_str();
        }
    protected:
        std::string msg;    // 存储异常信息的字符串
    };

    /**
     * invalid_mm 类：自定义异常类，继承自 fmm_error，表示提供的流不表示 Matrix Market 文件
     */
    class invalid_mm : public fmm_error {
    public:
        explicit invalid_mm(std::string msg): fmm_error(std::move(msg)) {} // 构造函数，接收错误信息字符串
        explicit invalid_mm(std::string msg, int64_t line_num) : fmm_error(std::move(msg)) {
            prepend_line_number(line_num);  // 构造函数，接收错误信息字符串和行号，增加行号信息
        }

        // 增加行号到异常信息前缀的函数
        void prepend_line_number(int64_t line_num) {
            msg = std::string("Line ") + std::to_string(line_num) + ": " + msg;
        }
    };

    /**
     * out_of_range 类：自定义异常类，继承自 invalid_mm，表示遇到不适合提供类型的值
     */
    class out_of_range : public invalid_mm {
    public:
        explicit out_of_range(std::string msg): invalid_mm(std::move(msg)) {} // 构造函数，接收错误信息字符串
    };

    /**
     * invalid_argument 类：自定义异常类，继承自 fmm_error，表示传入的参数无效
     */
    class invalid_argument : public fmm_error {
    public:
        explicit invalid_argument(std::string msg): fmm_error(std::move(msg)) {} // 构造函数，接收错误信息字符串
    };

    /**
     * complex_incompatible 类：自定义异常类，继承自 invalid_argument，表示 Matrix Market 文件具有复杂字段，但数据结构无法处理复杂值
     */
    class complex_incompatible : public invalid_argument {
    public:
        explicit complex_incompatible(std::string msg): invalid_argument(std::move(msg)) {} // 构造函数，接收错误信息字符串
    };

    /**
     * support_not_selected 类：自定义异常类，继承自 invalid_argument，表示 Matrix Market 文件需要已禁用的功能
     */
    class support_not_selected : public invalid_argument {
    public:
        explicit support_not_selected(std::string msg): invalid_argument(std::move(msg)) {} // 构造函数，接收错误信息字符串
    };

    /**
     * no_vector_support 类：自定义异常类，继承自 support_not_selected，表示 Matrix Market 文件是 'vector' 类型，但此构建中禁用了向量支持
     */
    class no_vector_support : public support_not_selected {
    public:
        explicit no_vector_support(std::string msg): support_not_selected(std::move(msg)) {} // 构造函数，接收错误信息字符串
    };
    /**
     * A value type to use for pattern matrices. Pattern Matrix Market files do not write a value column, only the
     * coordinates. Setting this as the value type signals the parser to not attempt to read a column that isn't there.
     */
    struct pattern_placeholder_type {};
    
    /**
     * Negation of a pattern_placeholder_type needed to support symmetry generalization.
     * Skew-symmetric symmetry negates values.
     */
    inline pattern_placeholder_type operator-(const pattern_placeholder_type& o) { return o; }
    
    /**
     * MSVC does not like std::negate<bool>
     */
    inline bool negate(const bool o) {
        return !o;
    }
    
    inline bool negate(const std::vector<bool>::reference o) {
        return !o;
    }
    
    template <typename T>
    T negate(const T& o) {
        return std::negate<T>()(o);
    }
    
    template <typename T>
    T pattern_default_value([[maybe_unused]] const T* type) {
        return 1;
    }
    
    /**
     * Zero generator for generalize symmetry with ExtraZeroElement.
     */
    template <typename T>
    T get_zero() {
        return {};
    }
    
    /**
     * Determine if a std::future is ready to return a result, i.e. finished computing.
     * @return true if the future is ready.
     */
    template<typename R>
    bool is_ready(std::future<R> const& f)
    {
        return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }
    
    /**
     * @param flags flags bitwise ORed together
     * @param flag flag bit to test for
     * @return true if the flag bit is set in flags, false otherwise
     */
    inline bool test_flag(int flags, int flag) {
        return (flags & flag) == flag;
    }
    
    inline bool starts_with(const std::string &str, const std::string& prefix) {
        if (prefix.size() > str.size()) {
            return false;
        }
        return std::equal(prefix.begin(), prefix.end(), str.begin());
    }
    
    inline bool ends_with(const std::string &str, const std::string& suffix) {
        if (suffix.size() > str.size()) {
            return false;
        }
        return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
    }
    
    /**
     * Trim the whitespace from both ends of a string. Returns a copy.
     */
    inline std::string trim(std::string s) {
        // ltrim
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
    
        // rtrim
        s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), s.end());
    
        return s;
    }
    
    /**
     * Replace all instances of `from` with `to` in `str`.
     */
    // 定义一个函数，用于替换字符串中所有的指定子串
    inline std::string replace_all(const std::string& str, const std::string& from, const std::string& to) {
        // 创建一个字符串副本，初始值为参数中的原始字符串
        std::string ret(str);

        // 如果要查找替换的原始字符串为空，则直接返回副本
        if (from.empty())
            return ret;

        // 定义起始搜索位置为字符串的开头
        std::string::size_type start_pos = 0;
        // 循环查找并替换所有出现的原始字符串
        while ((start_pos = ret.find(from, start_pos)) != std::string::npos) {
            // 使用目标字符串替换原始字符串的位置
            ret.replace(start_pos, from.length(), to);
            // 更新起始搜索位置，以避免无限循环
            start_pos += to.length();
        }

        // 返回替换完成的字符串副本
        return ret;
    }
}



#include "field_conv.hpp"



#include "header.hpp"



#include "parse_handlers.hpp"



#include "formatters.hpp"



#include "read_body.hpp"



#include "write_body.hpp"



#include "app/array.hpp"



#include "app/doublet.hpp"



#include "app/triplet.hpp"



#endif


注释：

}

这行代码结束了一个C++头文件的定义部分，关闭了之前使用`#ifndef`开始的条件编译指令。

#include "field_conv.hpp"

包含了名为`field_conv.hpp`的头文件，用于引入其中定义的函数、变量或类到当前源文件中。

#include "header.hpp"

引入了名为`header.hpp`的头文件，通常包含一些全局定义或者整个项目的基本配置信息。

#include "parse_handlers.hpp"

包含了名为`parse_handlers.hpp`的头文件，可能包含处理解析逻辑的函数或类。

#include "formatters.hpp"

引入了名为`formatters.hpp`的头文件，其中可能定义了一些格式化输出或处理函数。

#include "read_body.hpp"

包含了名为`read_body.hpp`的头文件，可能包含了读取数据体相关的函数或类。

#include "write_body.hpp"

引入了名为`write_body.hpp`的头文件，可能定义了写入数据体相关的函数或类。

#include "app/array.hpp"

包含了名为`array.hpp`的头文件，可能定义了与数组相关的一些功能或类。

#include "app/doublet.hpp"

引入了名为`doublet.hpp`的头文件，可能定义了与双元组相关的一些功能或类。

#include "app/triplet.hpp"

包含了名为`triplet.hpp`的头文件，可能定义了与三元组相关的一些功能或类。

#endif

结束了条件编译指令的区块，与之前的`#ifndef`对应，标志着头文件的结束。
```