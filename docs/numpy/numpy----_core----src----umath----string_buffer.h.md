# `.\numpy\numpy\_core\src\umath\string_buffer.h`

```py
#ifndef _NPY_CORE_SRC_UMATH_STRING_BUFFER_H_
#define _NPY_CORE_SRC_UMATH_STRING_BUFFER_H_

#include <Python.h>
#include <cstddef>
#include <wchar.h>

// 定义不使用过时 API 的标志
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
// 定义多维数组模块
#define _MULTIARRAYMODULE
// 定义数学模块
#define _UMATHMODULE

#include "numpy/ndarraytypes.h"
#include "stringdtype/utf8_utils.h"
#include "string_fastsearch.h"
#include "gil_utils.h"

// 定义宏，检查索引是否溢出
#define CHECK_OVERFLOW(index) if (buf + (index) >= after) return 0
// 定义宏，提取一个整数的最高有效位
#define MSB(val) ((val) >> 7 & 1)

// 枚举类型，表示字符编码方式
enum class ENCODING {
    ASCII, UTF32, UTF8
};

// 枚举类型，表示实现的一元函数
enum class IMPLEMENTED_UNARY_FUNCTIONS {
    ISALPHA,
    ISDECIMAL,
    ISDIGIT,
    ISSPACE,
    ISALNUM,
    ISLOWER,
    ISUPPER,
    ISTITLE,
    ISNUMERIC,
    STR_LEN,
};

// 模板函数，根据编码类型返回字符
template <ENCODING enc>
inline npy_ucs4
getchar(const unsigned char *buf, int *bytes);

// 特化模板函数，处理 ASCII 编码类型的字符
template <>
inline npy_ucs4
getchar<ENCODING::ASCII>(const unsigned char *buf, int *bytes)
{
    *bytes = 1;  // ASCII 字符占用一个字节
    return (npy_ucs4) *buf;  // 返回字符的 Unicode 码点
}

// 特化模板函数，处理 UTF32 编码类型的字符
template <>
inline npy_ucs4
getchar<ENCODING::UTF32>(const unsigned char *buf, int *bytes)
{
    *bytes = 4;  // UTF32 字符占用四个字节
    return *(npy_ucs4 *)buf;  // 返回 UTF32 编码字符的 Unicode 码点
}

// 特化模板函数，处理 UTF8 编码类型的字符
template <>
inline npy_ucs4
getchar<ENCODING::UTF8>(const unsigned char *buf, int *bytes)
{
    Py_UCS4 codepoint;
    *bytes = utf8_char_to_ucs4_code(buf, &codepoint);  // 将 UTF8 字符转换为 Unicode 码点
    return (npy_ucs4)codepoint;  // 返回 Unicode 码点
}

// 模板函数，根据编码类型判断字符是否为字母
template<ENCODING enc>
inline bool
codepoint_isalpha(npy_ucs4 code);

// 特化模板函数，处理 ASCII 编码类型的字母判断
template<>
inline bool
codepoint_isalpha<ENCODING::ASCII>(npy_ucs4 code)
{
    return NumPyOS_ascii_isalpha(code);  // 使用 NumPy 的 ASCII 字母判断函数
}

// 特化模板函数，处理 UTF32 编码类型的字母判断
template<>
inline bool
codepoint_isalpha<ENCODING::UTF32>(npy_ucs4 code)
{
    return Py_UNICODE_ISALPHA(code);  // 使用 Python 内部的 UTF32 字母判断函数
}

// 特化模板函数，处理 UTF8 编码类型的字母判断
template<>
inline bool
codepoint_isalpha<ENCODING::UTF8>(npy_ucs4 code)
{
    return Py_UNICODE_ISALPHA(code);  // 使用 Python 内部的 UTF8 字母判断函数
}

// 模板函数，根据编码类型判断字符是否为数字
template<ENCODING enc>
inline bool
codepoint_isdigit(npy_ucs4 code);

// 特化模板函数，处理 ASCII 编码类型的数字判断
template<>
inline bool
codepoint_isdigit<ENCODING::ASCII>(npy_ucs4 code)
{
    return NumPyOS_ascii_isdigit(code);  // 使用 NumPy 的 ASCII 数字判断函数
}

// 特化模板函数，处理 UTF32 编码类型的数字判断
template<>
inline bool
codepoint_isdigit<ENCODING::UTF32>(npy_ucs4 code)
{
    return Py_UNICODE_ISDIGIT(code);  // 使用 Python 内部的 UTF32 数字判断函数
}

// 特化模板函数，处理 UTF8 编码类型的数字判断
template<>
inline bool
codepoint_isdigit<ENCODING::UTF8>(npy_ucs4 code)
{
    return Py_UNICODE_ISDIGIT(code);  // 使用 Python 内部的 UTF8 数字判断函数
}

// 模板函数，根据编码类型判断字符是否为空白字符
template<ENCODING enc>
inline bool
codepoint_isspace(npy_ucs4 code);

// 特化模板函数，处理 ASCII 编码类型的空白字符判断
template<>
inline bool
codepoint_isspace<ENCODING::ASCII>(npy_ucs4 code)
{
    return NumPyOS_ascii_isspace(code);  // 使用 NumPy 的 ASCII 空白字符判断函数
}

// 特化模板函数，处理 UTF32 编码类型的空白字符判断
template<>
inline bool
codepoint_isspace<ENCODING::UTF32>(npy_ucs4 code)
{
    return Py_UNICODE_ISSPACE(code);  // 使用 Python 内部的 UTF32 空白字符判断函数
}

// 特化模板函数，处理 UTF8 编码类型的空白字符判断
template<>
inline bool
codepoint_isspace<ENCODING::UTF8>(npy_ucs4 code)
{
    return Py_UNICODE_ISSPACE(code);  // 使用 Python 内部的 UTF8 空白字符判断函数
}

// 模板函数，根据编码类型判断字符是否为字母或数字
template<ENCODING enc>
inline bool
codepoint_isalnum(npy_ucs4 code);

// 特化模板函数，处理 ASCII 编码类型的字母或数字判断
template<>
inline bool
codepoint_isalnum<ENCODING::ASCII>(npy_ucs4 code)
{
    return NumPyOS_ascii_isalnum(code);  // 使用 NumPy 的 ASCII 字母或数字判断函数
}

// 特化模板函数，处理 UTF32 编码类型的字母或数字判断
template<>
inline bool
codepoint_isalnum<ENCODING::UTF32>(npy_ucs4 code)
{
    return Py_UNICODE_ISALNUM(code);  // 使用 Python 内部的 UTF32 字母或数字判断函数
}

// 特化模板函数，处理 UTF8 编码类型的字母或数字判断
template<>
inline bool
codepoint_isalnum<ENCODING::UTF8>(npy_ucs4 code)
{
    return Py_UNICODE_ISALNUM(code);  // 使用 Python 内部的 UTF8 字母或数字判断函数
}

// 继续添加更多的模板函数实现...
// 声明模板特化，用于判断给定编码下的 Unicode 代码点是否为小写字母
template<>
inline bool
codepoint_islower<ENCODING::ASCII>(npy_ucs4 code)
{
    return NumPyOS_ascii_islower(code);
}

// 声明模板特化，用于判断给定编码下的 Unicode 代码点是否为小写字母
template<>
inline bool
codepoint_islower<ENCODING::UTF32>(npy_ucs4 code)
{
    return Py_UNICODE_ISLOWER(code);
}

// 声明模板特化，用于判断给定编码下的 Unicode 代码点是否为小写字母
template<>
inline bool
codepoint_islower<ENCODING::UTF8>(npy_ucs4 code)
{
    return Py_UNICODE_ISLOWER(code);
}

// 声明模板特化，用于判断给定编码下的 Unicode 代码点是否为大写字母
template<ENCODING enc>
inline bool
codepoint_isupper(npy_ucs4 code);

// 模板特化实现，用于判断 ASCII 编码下的 Unicode 代码点是否为大写字母
template<>
inline bool
codepoint_isupper<ENCODING::ASCII>(npy_ucs4 code)
{
    return NumPyOS_ascii_isupper(code);
}

// 模板特化实现，用于判断 UTF32 编码下的 Unicode 代码点是否为大写字母
template<>
inline bool
codepoint_isupper<ENCODING::UTF32>(npy_ucs4 code)
{
    return Py_UNICODE_ISUPPER(code);
}

// 模板特化实现，用于判断 UTF8 编码下的 Unicode 代码点是否为大写字母
template<>
inline bool
codepoint_isupper<ENCODING::UTF8>(npy_ucs4 code)
{
    return Py_UNICODE_ISUPPER(code);
}

// 声明模板特化，用于判断给定编码下的 Unicode 代码点是否为标题字母
template<ENCODING enc>
inline bool
codepoint_istitle(npy_ucs4);

// 模板特化实现，ASCII 编码下的 Unicode 代码点不支持标题字母的定义，始终返回 false
template<>
inline bool
codepoint_istitle<ENCODING::ASCII>(npy_ucs4 code)
{
    return false;
}

// 模板特化实现，UTF32 编码下的 Unicode 代码点是否为标题字母
template<>
inline bool
codepoint_istitle<ENCODING::UTF32>(npy_ucs4 code)
{
    return Py_UNICODE_ISTITLE(code);
}

// 模板特化实现，UTF8 编码下的 Unicode 代码点是否为标题字母
template<>
inline bool
codepoint_istitle<ENCODING::UTF8>(npy_ucs4 code)
{
    return Py_UNICODE_ISTITLE(code);
}

// 判断给定的 Unicode 代码点是否为数字
inline bool
codepoint_isnumeric(npy_ucs4 code)
{
    return Py_UNICODE_ISNUMERIC(code);
}

// 判断给定的 Unicode 代码点是否为十进制数字
inline bool
codepoint_isdecimal(npy_ucs4 code)
{
    return Py_UNICODE_ISDECIMAL(code);
}
    {
        // 开始一个 switch 结构，根据 enc 的值进行不同的操作
        switch (enc) {
        case ENCODING::ASCII:
            // 如果编码是 ASCII，将 buf 减去 rhs
            buf -= rhs;
            break;
        case ENCODING::UTF32:
            // 如果编码是 UTF32，将 buf 减去 rhs 乘以每个字符的字节数（sizeof(npy_ucs4)）
            buf -= rhs * sizeof(npy_ucs4);
            break;
        case ENCODING::UTF8:
            // 如果编码是 UTF8，查找 buf 中前一个 UTF8 字符的位置，并将 buf 设置为该位置
            buf = (char *) find_previous_utf8_character((unsigned char *)buf, (size_t) rhs);
        }
        // 返回当前对象的引用
        return *this;
    }

    inline Buffer<enc>&
    operator++()
    {
        // 前缀递增运算符重载，调用后缀递增运算符重载，然后返回当前对象的引用
        *this += 1;
        return *this;
    }

    inline Buffer<enc>
    operator++(int)
    {
        // 后缀递增运算符重载，保存当前对象的副本，调用前缀递增运算符重载，然后返回保存的副本
        Buffer<enc> old = *this;
        operator++();
        return old;
    }

    inline Buffer<enc>&
    operator--()
    {
        // 前缀递减运算符重载，调用后缀递减运算符重载，然后返回当前对象的引用
        *this -= 1;
        return *this;
    }

    inline Buffer<enc>
    operator--(int)
    {
        // 后缀递减运算符重载，保存当前对象的副本，调用前缀递减运算符重载，然后返回保存的副本
        Buffer<enc> old = *this;
        operator--();
        return old;
    }

    inline npy_ucs4
    operator*()
    {
        // 解引用运算符重载，获取当前位置 buf 的字符并返回
        int bytes;
        return getchar<enc>((unsigned char *) buf, &bytes);
    }

    inline int
    buffer_memcmp(Buffer<enc> other, size_t len)
    {
        // 比较当前对象 buf 和另一个对象 other 的前 len 字节或字符
        if (len == 0) {
            return 0;
        }
        switch (enc) {
            case ENCODING::ASCII:
            case ENCODING::UTF8:
                // 对于 ASCII 和 UTF8，按字节比较 len 长度
                return memcmp(buf, other.buf, len);
            case ENCODING::UTF32:
                // 对于 UTF32，按每个字符占用的字节数进行比较
                return memcmp(buf, other.buf, len * sizeof(npy_ucs4));
        }
    }

    inline void
    buffer_memcpy(Buffer<enc> out, size_t n_chars)
    {
        // 将当前对象 buf 的前 n_chars 个字节或字符复制到目标对象 out 的 buf 中
        if (n_chars == 0) {
            return;
        }
        switch (enc) {
            case ENCODING::ASCII:
            case ENCODING::UTF8:
                // 对于 ASCII 和 UTF8，按字节复制 n_chars 长度
                memcpy(out.buf, buf, n_chars);
                break;
            case ENCODING::UTF32:
                // 对于 UTF32，按每个字符占用的字节数进行复制
                memcpy(out.buf, buf, n_chars * sizeof(npy_ucs4));
                break;
        }
    }

    inline npy_intp
    buffer_memset(npy_ucs4 fill_char, size_t n_chars)
    {
        // 将当前对象 buf 的前 n_chars 个字节或字符设置为 fill_char 所代表的值
        if (n_chars == 0) {
            return 0;
        }
        switch (enc) {
            case ENCODING::ASCII:
                // 对于 ASCII，使用 memset 将 buf 的每个字节设置为 fill_char
                memset(this->buf, fill_char, n_chars);
                return n_chars;
            case ENCODING::UTF32:
            {
                // 对于 UTF32，将 buf 的每个字符设置为 fill_char，每个字符占用 sizeof(npy_ucs4) 字节
                char *tmp = this->buf;
                for (size_t i = 0; i < n_chars; i++) {
                    *(npy_ucs4 *)tmp = fill_char;
                    tmp += sizeof(npy_ucs4);
                }
                return n_chars;
            }
            case ENCODING::UTF8:
            {
                // 对于 UTF8，将 fill_char 转换为 UTF8 编码后，复制 n_chars 个字符到 buf 中
                char utf8_c[4] = {0};
                char *tmp = this->buf;
                size_t num_bytes = ucs4_code_to_utf8_char(fill_char, utf8_c);
                for (size_t i = 0; i < n_chars; i++) {
                    memcpy(tmp, utf8_c, num_bytes);
                    tmp += num_bytes;
                }
                return num_bytes * n_chars;
            }
        }
    }

    inline void
    buffer_fill_with_zeros_after_index(size_t start_index)
    {
        // 将从 start_index 开始之后的缓冲区填充为零
        Buffer<enc> offset = *this + start_index;
        for (char *tmp = offset.buf; tmp < after; tmp++) {
            *tmp = 0;
        }
    }

    inline void
    advance_chars_or_bytes(size_t n) {
        // 根据编码类型前进 n 个字符或字节
        switch (enc) {
            case ENCODING::ASCII:
            case ENCODING::UTF32:
                *this += n;
                break;
            case ENCODING::UTF8:
                this->buf += n;
                break;
        }
    }

    inline size_t
    num_bytes_next_character(void) {
        // 返回下一个字符所占的字节数
        switch (enc) {
            case ENCODING::ASCII:
                return 1;
            case ENCODING::UTF32:
                return 4;
            case ENCODING::UTF8:
                return num_bytes_for_utf8_character((unsigned char *)(*this).buf);
        }
    }

    template<IMPLEMENTED_UNARY_FUNCTIONS f>
    inline bool
    unary_loop()
    {
        // 对每个代码点执行特定的一元操作，直到结束或出现失败
        size_t len = num_codepoints();
        if (len == 0) {
            return false;
        }

        Buffer<enc> tmp = *this;

        for (size_t i=0; i<len; i++) {
            bool result;

            call_buffer_member_function<f, enc, bool> cbmf;

            // 调用特定的一元操作函数
            result = cbmf(tmp);

            if (!result) {
                return false;
            }
            tmp++;
        }
        return true;
    }

    inline bool
    isalpha()
    {
        // 检查是否所有字符都是字母
        return unary_loop<IMPLEMENTED_UNARY_FUNCTIONS::ISALPHA>();
    }

    inline bool
    first_character_isspace()
    {
        // 检查首字符是否为空白字符
        switch (enc) {
            case ENCODING::ASCII:
                return NumPyOS_ascii_isspace(**this);
            case ENCODING::UTF32:
            case ENCODING::UTF8:
                return Py_UNICODE_ISSPACE(**this);
        }
    }

    inline bool
    isspace()
    {
        // 检查是否所有字符都是空白字符
        return unary_loop<IMPLEMENTED_UNARY_FUNCTIONS::ISSPACE>();
    }

    inline bool
    isdigit()
    {
        // 检查是否所有字符都是数字
        return unary_loop<IMPLEMENTED_UNARY_FUNCTIONS::ISDIGIT>();
    }

    inline bool
    isalnum()
    {
        // 检查是否所有字符都是字母或数字
        return unary_loop<IMPLEMENTED_UNARY_FUNCTIONS::ISALNUM>();
    }

    inline bool
    islower()
    {
        // 检查是否所有字符都是小写字母
        size_t len = num_codepoints();
        if (len == 0) {
            return false;
        }

        Buffer<enc> tmp = *this;
        bool cased = 0;
        for (size_t i = 0; i < len; i++) {
            if (codepoint_isupper<enc>(*tmp) || codepoint_istitle<enc>(*tmp)) {
                return false;
            }
            else if (!cased && codepoint_islower<enc>(*tmp)) {
                cased = true;
            }
            tmp++;
        }
        return cased;
    }

    inline bool
    isupper()
    {
        // 获取此缓冲区中的字符数
        size_t len = num_codepoints();
        // 如果字符数为零，返回 false
        if (len == 0) {
            return false;
        }
    
        // 创建缓冲区的临时副本
        Buffer<enc> tmp = *this;
        // 指示是否至少有一个字符为大写
        bool cased = 0;
        // 遍历缓冲区中的每个字符
        for (size_t i = 0; i < len; i++) {
            // 如果字符是小写或标题格式，则返回 false
            if (codepoint_islower<enc>(*tmp) || codepoint_istitle<enc>(*tmp)) {
                return false;
            }
            // 如果之前未遇到大写字符且当前字符是大写，则设置 cased 为 true
            else if (!cased && codepoint_isupper<enc>(*tmp)) {
                cased = true;
            }
            // 移动临时副本到下一个字符位置
            tmp++;
        }
        // 返回是否存在大写字符的标志
        return cased;
    }
    
    inline bool
    istitle()
    {
        // 获取此缓冲区中的字符数
        size_t len = num_codepoints();
        // 如果字符数为零，返回 false
        if (len == 0) {
            return false;
        }
    
        // 创建缓冲区的临时副本
        Buffer<enc> tmp = *this;
        // 指示是否存在标题格式的字符
        bool cased = false;
        // 指示前一个字符是否为标题格式或大写
        bool previous_is_cased = false;
        // 遍历缓冲区中的每个字符
        for (size_t i = 0; i < len; i++) {
            // 如果字符是大写或标题格式
            if (codepoint_isupper<enc>(*tmp) || codepoint_istitle<enc>(*tmp)) {
                // 如果之前已经有字符是标题格式或大写，则返回 false
                if (previous_is_cased) {
                    return false;
                }
                // 设置前一个字符为标题格式或大写
                previous_is_cased = true;
                // 设置 cased 为 true
                cased = true;
            }
            // 如果字符是小写
            else if (codepoint_islower<enc>(*tmp)) {
                // 如果之前没有遇到标题格式或大写字符，则返回 false
                if (!previous_is_cased) {
                    return false;
                }
                // 设置 cased 为 true
                cased = true;
            }
            // 如果字符既不是大写也不是小写
            else {
                // 设置前一个字符不是标题格式或大写
                previous_is_cased = false;
            }
            // 移动临时副本到下一个字符位置
            tmp++;
        }
        // 返回是否存在标题格式的字符的标志
        return cased;
    }
    
    inline bool
    isnumeric()
    {
        // 调用 unary_loop 函数检查是否存在数值字符
        return unary_loop<IMPLEMENTED_UNARY_FUNCTIONS::ISNUMERIC>();
    }
    
    inline bool
    isdecimal()
    {
        // 调用 unary_loop 函数检查是否存在十进制字符
        return unary_loop<IMPLEMENTED_UNARY_FUNCTIONS::ISDECIMAL>();
    }
    
    inline Buffer<enc>
    rstrip()
    {
        // 创建缓冲区的临时副本
        Buffer<enc> tmp(after, 0);
        // 将 tmp 移动到末尾字符的前一个位置
        tmp--;
        // 移除尾部的空白字符或者 '\0'
        while (tmp >= *this && (*tmp == '\0' || NumPyOS_ascii_isspace(*tmp))) {
            tmp--;
        }
        // 将 after 指向 tmp 的缓冲区
        after = tmp.buf;
        // 返回原始缓冲区
        return *this;
    }
    
    inline int
    strcmp(Buffer<enc> other, bool rstrip)
    {
        // 如果 rstrip 为 true，则对当前对象和 other 对象执行 rstrip 操作
        Buffer tmp1 = rstrip ? this->rstrip() : *this;
        Buffer tmp2 = rstrip ? other.rstrip() : other;
    
        // 比较两个缓冲区中的字符内容
        while (tmp1.buf < tmp1.after && tmp2.buf < tmp2.after) {
            // 如果 tmp1 中的字符小于 tmp2 中的字符，返回 -1
            if (*tmp1 < *tmp2) {
                return -1;
            }
            // 如果 tmp1 中的字符大于 tmp2 中的字符，返回 1
            if (*tmp1 > *tmp2) {
                return 1;
            }
            // 移动 tmp1 和 tmp2 到下一个字符位置
            tmp1++;
            tmp2++;
        }
        // 如果 tmp1 还有剩余字符且不为 '\0'，返回 1
        while (tmp1.buf < tmp1.after) {
            if (*tmp1) {
                return 1;
            }
            tmp1++;
        }
        // 如果 tmp2 还有剩余字符且不为 '\0'，返回 -1
        while (tmp2.buf < tmp2.after) {
            if (*tmp2) {
                return -1;
            }
            tmp2++;
        }
        // 如果两个缓冲区内容相同，返回 0
        return 0;
    }
    
    inline int
    strcmp(Buffer<enc> other)
    {
        // 调用带 rstrip 参数为 false 的 strcmp 函数
        return strcmp(other, false);
    }
// 结构模板，用于调用缓冲区成员函数的特定操作
template <IMPLEMENTED_UNARY_FUNCTIONS f, ENCODING enc, typename T>
struct call_buffer_member_function {
    // 重载函数调用操作符，根据给定的操作选择并调用对应的代码点检查函数
    T operator()(Buffer<enc> buf) {
        switch (f) {
            case IMPLEMENTED_UNARY_FUNCTIONS::ISALPHA:
                return codepoint_isalpha<enc>(*buf);
            case IMPLEMENTED_UNARY_FUNCTIONS::ISDIGIT:
                return codepoint_isdigit<enc>(*buf);
            case IMPLEMENTED_UNARY_FUNCTIONS::ISSPACE:
                return codepoint_isspace<enc>(*buf);
            case IMPLEMENTED_UNARY_FUNCTIONS::ISALNUM:
                return codepoint_isalnum<enc>(*buf);
            case IMPLEMENTED_UNARY_FUNCTIONS::ISNUMERIC:
                return codepoint_isnumeric(*buf);
            case IMPLEMENTED_UNARY_FUNCTIONS::ISDECIMAL:
                return codepoint_isdecimal(*buf);
        }
    }
};

// 重载加法操作符，用于将缓冲区和一个整数相加
template <ENCODING enc>
inline Buffer<enc>
operator+(Buffer<enc> lhs, npy_int64 rhs)
{
    switch (enc) {
        case ENCODING::ASCII:
            // 返回一个新的缓冲区，偏移 rhs 个字节后
            return Buffer<enc>(lhs.buf + rhs, lhs.after - lhs.buf - rhs);
        case ENCODING::UTF32:
            // 返回一个新的 UTF32 编码的缓冲区，偏移 rhs 个字符后
            return Buffer<enc>(lhs.buf + rhs * sizeof(npy_ucs4),
                          lhs.after - lhs.buf - rhs * sizeof(npy_ucs4));
        case ENCODING::UTF8:
            char* buf = lhs.buf;
            // 迭代偏移 rhs 次，找到新位置的 UTF8 字符的起始位置
            for (int i=0; i<rhs; i++) {
                buf += num_bytes_for_utf8_character((unsigned char *)buf);
            }
            // 返回一个新的 UTF8 编码的缓冲区，偏移后的长度
            return Buffer<enc>(buf, (npy_int64)(lhs.after - buf));
    }
}

// 重载减法操作符，用于将缓冲区和一个整数相减
template <ENCODING enc>
inline std::ptrdiff_t
operator-(Buffer<enc> lhs, Buffer<enc> rhs)
{
    switch (enc) {
    case ENCODING::ASCII:
    case ENCODING::UTF8:
        // 返回两个缓冲区指针之间的偏移量
        // 对于 UTF8 字符串，除非比较的是同一字符串中的两个位置，否则结果可能无意义
        return lhs.buf - rhs.buf;
    case ENCODING::UTF32:
        // 返回两个 UTF32 编码的缓冲区指针之间的偏移量，以字符数表示
        return (lhs.buf - rhs.buf) / (std::ptrdiff_t) sizeof(npy_ucs4);
    }
}

// 重载减法操作符，用于将缓冲区和一个整数相减
template <ENCODING enc>
inline Buffer<enc>
operator-(Buffer<enc> lhs, npy_int64 rhs)
{
    switch (enc) {
        case ENCODING::ASCII:
            // 返回一个新的缓冲区，偏移 rhs 个字节前
            return Buffer<enc>(lhs.buf - rhs, lhs.after - lhs.buf + rhs);
        case ENCODING::UTF32:
            // 返回一个新的 UTF32 编码的缓冲区，偏移 rhs 个字符前
            return Buffer<enc>(lhs.buf - rhs * sizeof(npy_ucs4),
                          lhs.after - lhs.buf + rhs * sizeof(npy_ucs4));
        case ENCODING::UTF8:
            char* buf = lhs.buf;
            // 找到从当前位置往前偏移 rhs 个字符后的 UTF8 字符的起始位置
            buf = (char *)find_previous_utf8_character((unsigned char *)buf, rhs);
            // 返回一个新的 UTF8 编码的缓冲区，偏移后的长度
            return Buffer<enc>(buf, (npy_int64)(lhs.after - buf));
    }
}

// 重载相等比较操作符，用于比较两个缓冲区是否相等
template <ENCODING enc>
inline bool
operator==(Buffer<enc> lhs, Buffer<enc> rhs)
{
    return lhs.buf == rhs.buf;
}

// 重载不等比较操作符，用于比较两个缓冲区是否不相等
template <ENCODING enc>
inline bool
operator!=(Buffer<enc> lhs, Buffer<enc> rhs)
{
    return !(rhs == lhs);
}

// 重载小于比较操作符，用于比较两个缓冲区的大小关系
template <ENCODING enc>
inline bool
operator<(Buffer<enc> lhs, Buffer<enc> rhs)
{
    return lhs.buf < rhs.buf;
}

// 重载大于比较操作符，用于比较两个缓冲区的大小关系
template <ENCODING enc>
inline bool
operator>(Buffer<enc> lhs, Buffer<enc> rhs)
{
    return rhs < lhs;
}

// 未完成的模板函数声明
template <ENCODING enc>
inline bool
/*
 * Operator <= for Buffer objects.
 *
 * This function provides the less than or equal to comparison
 * for Buffer objects. It delegates to the negation of the
 * greater than operator for the implementation.
 */
operator<=(Buffer<enc> lhs, Buffer<enc> rhs)
{
    return !(lhs > rhs);
}

/*
 * Operator >= for Buffer objects.
 *
 * This function provides the greater than or equal to comparison
 * for Buffer objects. It delegates to the negation of the
 * less than operator for the implementation.
 */
template <ENCODING enc>
inline bool
operator>=(Buffer<enc> lhs, Buffer<enc> rhs)
{
    return !(lhs < rhs);
}

/*
 * Helper function to adjust start/end slice values.
 *
 * This function adjusts start and end slice indices to ensure
 * they are within valid bounds for a given length. It modifies
 * the start and end values accordingly based on Python's slicing
 * rules to handle negative indices and bounds exceeding length.
 */
static inline void
adjust_offsets(npy_int64 *start, npy_int64 *end, size_t len)
{
    if (*end > static_cast<npy_int64>(len)) {
        *end = len;
    }
    else if (*end < 0) {
        *end += len;
        if (*end < 0) {
            *end = 0;
        }
    }

    if (*start < 0) {
        *start += len;
        if (*start < 0) {
            *start = 0;
        }
    }
}

/*
 * Find function for strings in a given encoding.
 *
 * This function searches for buf2 within buf1 using start and end
 * indices. It handles different encodings (UTF8, ASCII, UTF32) and
 * adjusts the search indices using adjust_offsets() for compatibility
 * with Python's str functions. It returns the index of the first occurrence
 * of buf2 in buf1[start:end], or -1 if not found.
 */
template <ENCODING enc>
static inline npy_intp
string_find(Buffer<enc> buf1, Buffer<enc> buf2, npy_int64 start, npy_int64 end)
{
    size_t len1 = buf1.num_codepoints();
    size_t len2 = buf2.num_codepoints();

    // Adjust start and end indices to fit within the length of buf1
    adjust_offsets(&start, &end, len1);

    // If buf2 is longer than the slice [start:end], return -1
    if (end - start < static_cast<npy_int64>(len2)) {
        return (npy_intp) -1;
    }

    // If buf2 is empty, return the start index
    if (len2 == 0) {
        return (npy_intp) start;
    }

    char *start_loc = NULL;
    char *end_loc = NULL;

    // Depending on encoding, set start_loc and end_loc appropriately
    if (enc == ENCODING::UTF8) {
        find_start_end_locs(buf1.buf, (buf1.after - buf1.buf), start, end,
                            &start_loc, &end_loc);
    }
    else {
        start_loc = (buf1 + start).buf;
        end_loc = (buf1 + end).buf;
    }

    // Handle the case of searching for a single character
    if (len2 == 1) {
        npy_intp result;
        switch (enc) {
            case ENCODING::UTF8:
            {
                // Handle UTF8 characters that span multiple bytes
                if (num_bytes_for_utf8_character((const unsigned char *)buf2.buf) > 1) {
                    goto multibyte_search;
                }
                // fall through to the ASCII case because this is a one-byte character
            }
            case ENCODING::ASCII:
            {
                char ch = *buf2;
                CheckedIndexer<char> ind(start_loc, end_loc - start_loc);
                result = (npy_intp) findchar(ind, end_loc - start_loc, ch);
                // Adjust result index for UTF8 encoding if necessary
                if (enc == ENCODING::UTF8 && result > 0) {
                    result = utf8_character_index(
                            start_loc, start_loc - buf1.buf, start, result,
                            buf1.after - start_loc);
                }
                break;
            }
            case ENCODING::UTF32:
            {
                npy_ucs4 ch = *buf2;
                CheckedIndexer<npy_ucs4> ind((npy_ucs4 *)(buf1 + start).buf, end-start);
                result = (npy_intp) findchar(ind, end - start, ch);
                break;
            }
        }
        // Return -1 if character not found, otherwise return adjusted index
        if (result == -1) {
            return (npy_intp) -1;
        }
        else {
            return result + (npy_intp) start;
        }
    }

  multibyte_search:

    npy_intp pos;
    // 根据不同的编码格式进行搜索
    switch(enc) {
        case ENCODING::UTF8:
            // 如果是UTF-8编码，使用快速搜索算法在buf2中查找子串位置
            pos = fastsearch(start_loc, end_loc - start_loc, buf2.buf, buf2.after - buf2.buf, -1, FAST_SEARCH);
            // pos 是字节索引，需要转换为字符索引
            if (pos > 0) {
                pos = utf8_character_index(start_loc, start_loc - buf1.buf, start, pos, buf1.after - start_loc);
            }
            break;
        case ENCODING::ASCII:
            // 如果是ASCII编码，使用快速搜索算法在buf2中查找子串位置
            pos = fastsearch(start_loc, end - start, buf2.buf, len2, -1, FAST_SEARCH);
            break;
        case ENCODING::UTF32:
            // 如果是UTF-32编码，使用快速搜索算法在buf2中查找子串位置（类型转换为npy_ucs4*）
            pos = fastsearch((npy_ucs4 *)start_loc, end - start,
                             (npy_ucs4 *)buf2.buf, len2, -1, FAST_SEARCH);
            break;
    }

    // 如果找到了匹配的位置，转换为全局索引
    if (pos >= 0) {
        pos += start;
    }
    // 返回搜索到的位置
    return pos;
/* 
   定义了一个模板函数 string_index，用于在 buf1 中查找 buf2 的子串，
   并返回其在 buf1 中的索引位置。如果未找到子串，则抛出异常并返回 -2。
*/
template <ENCODING enc>
static inline npy_intp
string_index(Buffer<enc> buf1, Buffer<enc> buf2, npy_int64 start, npy_int64 end)
{
    // 调用 string_find 函数在 buf1 中查找 buf2 的位置
    npy_intp pos = string_find(buf1, buf2, start, end);
    // 如果找不到子串，则抛出 ValueError 异常并返回 -2
    if (pos == -1) {
        npy_gil_error(PyExc_ValueError, "substring not found");
        return -2;
    }
    // 返回找到的子串在 buf1 中的索引位置
    return pos;
}

/* 
   定义了一个模板函数 string_rfind，用于在 buf1 中逆向查找 buf2 的子串，
   并返回其在 buf1 中的索引位置。如果未找到子串或条件不符合，则返回 -1。
*/
template <ENCODING enc>
static inline npy_intp
string_rfind(Buffer<enc> buf1, Buffer<enc> buf2, npy_int64 start, npy_int64 end)
{
    // 获取 buf1 和 buf2 的字符数
    size_t len1 = buf1.num_codepoints();
    size_t len2 = buf2.num_codepoints();

    // 调整查找范围的起始和结束位置，确保不超出 buf1 的长度范围
    adjust_offsets(&start, &end, len1);
    
    // 如果查找范围内的长度小于 buf2 的长度，则直接返回 -1
    if (end - start < static_cast<npy_int64>(len2)) {
        return (npy_intp) -1;
    }

    // 如果 buf2 的长度为 0，则返回查找的结束位置 end
    if (len2 == 0) {
        return (npy_intp) end;
    }

    // 初始化起始和结束位置的指针
    char *start_loc = NULL;
    char *end_loc = NULL;
    
    // 根据编码类型选择不同的查找函数和位置计算方式
    if (enc == ENCODING::UTF8) {
        // 如果是 UTF-8 编码，则调用 find_start_end_locs 函数获取起始和结束位置的指针
        find_start_end_locs(buf1.buf, (buf1.after - buf1.buf), start, end,
                            &start_loc, &end_loc);
    }
    else {
        // 其他编码方式则直接计算起始和结束位置的指针
        start_loc = (buf1 + start).buf;
        end_loc = (buf1 + end).buf;
    }

    // 如果 buf2 的长度为 1，则根据不同的编码类型选择不同的查找方式
    if (len2 == 1) {
        npy_intp result;
        switch (enc) {
            case ENCODING::UTF8:
            {
                // 如果是 UTF-8 编码并且 buf2 是多字节字符，则跳转到 multibyte_search 分支
                if (num_bytes_for_utf8_character((const unsigned char *)buf2.buf) > 1) {
                    goto multibyte_search;
                }
                // 单字节字符则继续 ASCII 查找
            }
            // ASCII 编码的情况
            case ENCODING::ASCII:
            {
                // 获取 buf2 的第一个字符
                char ch = *buf2;
                // 初始化 CheckedIndexer 对象，用于安全地访问字符数组
                CheckedIndexer<char> ind(start_loc, end_loc - start_loc);
                // 在 [start_loc, end_loc) 范围内逆向查找字符 ch，并返回索引位置
                result = (npy_intp) rfindchar(ind, end_loc - start_loc, ch);
                // 如果是 UTF-8 编码并且找到了字符，则调整结果索引为字符的实际位置
                if (enc == ENCODING::UTF8 && result > 0) {
                    result = utf8_character_index(
                            start_loc, start_loc - buf1.buf, start, result,
                            buf1.after - start_loc);
                }
                break;
            }
            // UTF-32 编码的情况
            case ENCODING::UTF32:
            {
                // 获取 buf2 的第一个字符
                npy_ucs4 ch = *buf2;
                // 初始化 CheckedIndexer 对象，用于安全地访问 UCS-4 编码字符数组
                CheckedIndexer<npy_ucs4> ind((npy_ucs4 *)(buf1 + start).buf, end - start);
                // 在 [start, end) 范围内逆向查找字符 ch，并返回索引位置
                result = (npy_intp) rfindchar(ind, end - start, ch);
                break;
            }
        }
        // 如果未找到字符，则返回 -1；否则返回字符的实际位置加上起始位置 start
        if (result == -1) {
            return (npy_intp) -1;
        }
        else {
            return result + (npy_intp) start;
        }
    }

  multibyte_search:

    npy_intp pos;
    switch (enc) {
        case ENCODING::UTF8:
            // 在 buf2 中快速搜索 start_loc 到 end_loc 之间的内容
            pos = fastsearch(start_loc, end_loc - start_loc, buf2.buf, buf2.after - buf2.buf, -1, FAST_RSEARCH);
            // pos 是字节索引，需要转换为字符索引
            if (pos > 0) {
                // 根据 UTF-8 编码，计算字符索引
                pos = utf8_character_index(start_loc, start_loc - buf1.buf, start, pos, buf1.after - start_loc);
            }
            break;
        case ENCODING::ASCII:
            // 在 buf2 中快速搜索 start 到 end 之间的内容
            pos = (npy_intp) fastsearch(start_loc, end - start, buf2.buf, len2, -1, FAST_RSEARCH);
            break;
        case ENCODING::UTF32:
            // 在 buf2 中快速搜索 start 到 end 之间的 UTF-32 编码内容
            pos = (npy_intp) fastsearch((npy_ucs4 *)start_loc, end - start,
                                        (npy_ucs4 *)buf2.buf, len2, -1, FAST_RSEARCH);
            break;
    }
    // 如果找到位置 pos，则将其转换为全局字符索引
    if (pos >= 0) {
        pos += start;
    }
    // 返回最终的位置索引
    return pos;
/*
 * string_rindex: Searches for the last occurrence of buf2 in buf1 within the range [start, end).
 * Returns the index of the occurrence or -2 if not found (with an exception raised).
 */
template <ENCODING enc>
static inline npy_intp
string_rindex(Buffer<enc> buf1, Buffer<enc> buf2, npy_int64 start, npy_int64 end)
{
    // Find the last occurrence of buf2 in buf1
    npy_intp pos = string_rfind(buf1, buf2, start, end);
    // Handle case where substring is not found
    if (pos == -1) {
        npy_gil_error(PyExc_ValueError, "substring not found");
        return -2;
    }
    // Return the index of the last occurrence
    return pos;
}

/*
 * string_count: Counts occurrences of buf2 in buf1 within the range [start, end).
 * Returns the count of occurrences.
 */
template <ENCODING enc>
static inline npy_intp
string_count(Buffer<enc> buf1, Buffer<enc> buf2, npy_int64 start, npy_int64 end)
{
    size_t len1 = buf1.num_codepoints();
    size_t len2 = buf2.num_codepoints();

    // Adjust start and end positions based on buffer lengths
    adjust_offsets(&start, &end, len1);
    
    // If the search range is invalid or empty, return 0 occurrences
    if (end < start || end - start < static_cast<npy_int64>(len2)) {
        return (npy_intp) 0;
    }

    // Handle case where buf2 is empty
    if (len2 == 0) {
        return (end - start) < PY_SSIZE_T_MAX ? end - start + 1 : PY_SSIZE_T_MAX;
    }

    char *start_loc = NULL;
    char *end_loc = NULL;

    // Determine start and end positions based on encoding type
    if (enc == ENCODING::UTF8) {
        find_start_end_locs(buf1.buf, (buf1.after - buf1.buf), start, end,
                            &start_loc, &end_loc);
    }
    else {
        start_loc = (buf1 + start).buf;
        end_loc = (buf1 + end).buf;
    }

    npy_intp count;
    // Perform fast search based on encoding type
    switch (enc) {
        case ENCODING::UTF8:
            count = fastsearch(start_loc, end_loc - start_loc, buf2.buf,
                               buf2.after - buf2.buf, PY_SSIZE_T_MAX,
                               FAST_COUNT);
            break;
        case ENCODING::ASCII:
            count = (npy_intp) fastsearch(start_loc, end - start, buf2.buf, len2,
                                          PY_SSIZE_T_MAX, FAST_COUNT);
            break;
        case ENCODING::UTF32:
            count = (npy_intp) fastsearch((npy_ucs4 *)start_loc, end - start,
                                          (npy_ucs4 *)buf2.buf, len2,
                                          PY_SSIZE_T_MAX, FAST_COUNT);
            break;
    }

    // Return 0 if an error occurred during search
    if (count < 0) {
        return 0;
    }
    // Return the count of occurrences found
    return count;
}

/*
 * tailmatch: Checks if buf2 matches the end of buf1 within the specified range [start, end).
 * Returns true (1) if there's a match, otherwise false (0).
 */
template <ENCODING enc>
inline npy_bool
tailmatch(Buffer<enc> buf1, Buffer<enc> buf2, npy_int64 start, npy_int64 end,
          STARTPOSITION direction)
{
    size_t len1 = buf1.num_codepoints();
    size_t len2 = buf2.num_codepoints();

    // Adjust start and end positions based on buffer lengths
    adjust_offsets(&start, &end, len1);
    
    // Calculate the position to start checking for tail match
    end -= len2;
    if (end < start) {
        return 0;
    }

    // Handle case where buf2 is empty, always return true
    if (len2 == 0) {
        return 1;
    }

    size_t offset;
    size_t end_sub = len2 - 1;

    // Determine offset based on direction of search (front or back)
    if (direction == STARTPOSITION::BACK) {
        offset = end;
    }
    else {
        offset = start;
    }

    size_t size2 = len2;

    // Adjust size2 for UTF8 encoding
    if (enc == ENCODING::UTF8) {
        size2 = (buf2.after - buf2.buf);
    }

    // Create buffer slices for comparison
    Buffer start_buf = (buf1 + offset);
    Buffer end_buf = start_buf + end_sub;

    // Check if the buffers match at the start and end positions
    if (*start_buf == *buf2 && *end_buf == *(buf2 + end_sub)) {
        return !start_buf.buffer_memcmp(buf2, size2);
    }

    // Default return indicating no match

    // Return false indicating no match
    return 0;
}
    return 0;
}

// 枚举类型，表示字符串修剪的类型，可以是左修剪、右修剪或者两者结合
enum class STRIPTYPE {
    LEFTSTRIP, RIGHTSTRIP, BOTHSTRIP
};

// 模板函数，用于去除字符串两端的空白字符
template <ENCODING enc>
static inline size_t
string_lrstrip_whitespace(Buffer<enc> buf, Buffer<enc> out, STRIPTYPE striptype)
{
    // 获取字符串的字符数
    size_t len = buf.num_codepoints();
    
    // 如果字符串长度为0
    if (len == 0) {
        // 如果编码不是UTF-8，则将输出缓冲区填充为零
        if (enc != ENCODING::UTF8) {
            out.buffer_fill_with_zeros_after_index(0);
        }
        return 0;  // 返回修剪后的长度
    }

    size_t i = 0;

    // 计算输入缓冲区的字节数
    size_t num_bytes = (buf.after - buf.buf);
    Buffer traverse_buf = Buffer<enc>(buf.buf, num_bytes);

    // 如果不是右修剪类型，则从左端开始移除空白字符
    if (striptype != STRIPTYPE::RIGHTSTRIP) {
        while (i < len) {
            if (!traverse_buf.first_character_isspace()) {
                break;
            }
            num_bytes -= traverse_buf.num_bytes_next_character();
            traverse_buf++;
            i++;
        }
    }

    npy_intp j = len - 1;  // j表示从右端开始的索引，如果整个字符串都被修剪，则可能变成负数
    if (enc == ENCODING::UTF8) {
        traverse_buf = Buffer<enc>(buf.after, 0) - 1;  // UTF-8编码时，从尾部开始处理
    }
    else {
        traverse_buf = buf + j;  // 其他编码，从末尾字符开始处理
    }

    // 如果不是左修剪类型，则从右端开始移除空白字符
    if (striptype != STRIPTYPE::LEFTSTRIP) {
        while (j >= static_cast<npy_intp>(i)) {
            if (*traverse_buf != 0 && !traverse_buf.first_character_isspace()) {
                break;
            }
            num_bytes -= traverse_buf.num_bytes_next_character();
            traverse_buf--;
            j--;
        }
    }

    Buffer offset_buf = buf + i;  // 根据i偏移得到修剪后的起始位置
    if (enc == ENCODING::UTF8) {
        offset_buf.buffer_memcpy(out, num_bytes);  // 将修剪后的字符串复制到输出缓冲区
        return num_bytes;  // 返回修剪后的长度
    }
    offset_buf.buffer_memcpy(out, j - i + 1);  // 将修剪后的字符串复制到输出缓冲区
    out.buffer_fill_with_zeros_after_index(j - i + 1);  // 将剩余部分填充为零
    return j - i + 1;  // 返回修剪后的长度
}

// 模板函数，用于根据指定的字符集合修剪字符串的两端字符
template <ENCODING enc>
static inline size_t
string_lrstrip_chars(Buffer<enc> buf1, Buffer<enc> buf2, Buffer<enc> out, STRIPTYPE striptype)
{
    // 获取第一个缓冲区的字符数
    size_t len1 = buf1.num_codepoints();
    // 如果第一个缓冲区长度为0
    if (len1 == 0) {
        // 如果编码是UTF-8，则将整个第一个缓冲区复制到输出缓冲区
        if (enc == ENCODING::UTF8) {
            buf1.buffer_memcpy(out, (buf1.after - buf1.buf));
            return buf1.after - buf1.buf;
        }
        // 否则，将第一个缓冲区的长度复制到输出缓冲区，并填充剩余部分为零
        buf1.buffer_memcpy(out, len1);
        out.buffer_fill_with_zeros_after_index(len1);
        return len1;
    }

    size_t i = 0;

    // 获取第一个缓冲区的字节数
    size_t num_bytes = (buf1.after - buf1.buf);
    Buffer traverse_buf = Buffer<enc>(buf1.buf, num_bytes);

    // 确定字符串开始修剪的位置
    if (striptype != STRIPTYPE::RIGHTSTRIP) {
        while (i < len1) {
            // 如果第一个缓冲区的第一个字符不是指定的字符集合中的空白字符，则停止修剪
            if (!traverse_buf.first_character_isspace()) {
                break;
            }
            num_bytes -= traverse_buf.num_bytes_next_character();
            traverse_buf++;
            i++;
        }
    }

    // 获取第二个缓冲区的字符数
    size_t len2 = buf2.num_codepoints();
    // 如果第二个缓冲区长度为0
    if (len2 == 0) {
        // 如果编码是UTF-8，则将整个第一个缓冲区复制到输出缓冲区
        if (enc == ENCODING::UTF8) {
            buf1.buffer_memcpy(out, (buf1.after - buf1.buf));
            return buf1.after - buf1.buf;
        }
        // 否则，将第一个缓冲区的长度复制到输出缓冲区，并填充剩余部分为零
        buf1.buffer_memcpy(out, len1);
        out.buffer_fill_with_zeros_after_index(len1);
        return len1;
    }

    Buffer offset_buf = buf1 + i;  // 根据i偏移得到修剪后的起始位置
    if (enc == ENCODING::UTF8) {
        offset_buf.buffer_memcpy(out, num_bytes);  // 将修剪后的字符串复制到输出缓冲区
        return num_bytes;  // 返回修剪后的长度
    }
    offset_buf.buffer_memcpy(out, len1);  // 将修剪后的字符串复制到输出缓冲区
    out.buffer_fill_with_zeros_after_index(len1);  // 将剩余部分填充为零
    return len1;  // 返回修剪后的长度
}
    # 如果不是右侧去除空格类型，则执行以下代码块
    if (striptype != STRIPTYPE::RIGHTSTRIP) {
        # 循环直到 i 小于 len1
        while (i < len1) {
            # 声明一个 Py_ssize_t 类型的变量 res
            Py_ssize_t res;
            # 根据编码类型进行不同处理
            switch (enc) {
                # 对于 ASCII 和 UTF8 编码
                case ENCODING::ASCII:
                case ENCODING::UTF8:
                {
                    # 使用 CheckedIndexer<char> 对象 ind，索引 buf2.buf 的前 len2 个元素
                    CheckedIndexer<char> ind(buf2.buf, len2);
                    # 调用 findchar<char> 函数，在 ind 中查找 traverse_buf 指向的字符
                    res = findchar<char>(ind, len2, *traverse_buf);
                    break;
                }
                # 对于 UTF32 编码
                case ENCODING::UTF32:
                {
                    # 使用 CheckedIndexer<npy_ucs4> 对象 ind，索引 buf2.buf 的前 len2 个元素
                    CheckedIndexer<npy_ucs4> ind((npy_ucs4 *)buf2.buf, len2);
                    # 调用 findchar<npy_ucs4> 函数，在 ind 中查找 traverse_buf 指向的字符
                    res = findchar<npy_ucs4>(ind, len2, *traverse_buf);
                    break;
                }
            }
            # 如果未找到目标字符，则跳出循环
            if (res < 0) {
                break;
            }
            # 更新 num_bytes，减去 traverse_buf 下一个字符的字节数
            num_bytes -= traverse_buf.num_bytes_next_character();
            # traverse_buf 向后移动一个位置
            traverse_buf++;
            # i 自增
            i++;
        }
    }

    # 初始化 j 为 len1 减 1
    npy_intp j = len1 - 1;
    # 如果编码类型是 UTF8
    if (enc == ENCODING::UTF8) {
        # 将 traverse_buf 指向 Buffer<enc>(buf1.after, 0) 的前一个位置
        traverse_buf = Buffer<enc>(buf1.after, 0) - 1;
    }
    # 否则
    else {
        # 将 traverse_buf 指向 buf1 的末尾
        traverse_buf = buf1 + j;
    }

    # 如果不是左侧去除空格类型，则执行以下代码块
    if (striptype != STRIPTYPE::LEFTSTRIP) {
        # 循环直到 j 大于等于 i
        while (j >= static_cast<npy_intp>(i)) {
            # 声明一个 Py_ssize_t 类型的变量 res
            Py_ssize_t res;
            # 根据编码类型进行不同处理
            switch (enc) {
                # 对于 ASCII 和 UTF8 编码
                case ENCODING::ASCII:
                case ENCODING::UTF8:
                {
                    # 使用 CheckedIndexer<char> 对象 ind，索引 buf2.buf 的前 len2 个元素
                    CheckedIndexer<char> ind(buf2.buf, len2);
                    # 调用 findchar<char> 函数，在 ind 中查找 traverse_buf 指向的字符
                    res = findchar<char>(ind, len2, *traverse_buf);
                    break;
                }
                # 对于 UTF32 编码
                case ENCODING::UTF32:
                {
                    # 使用 CheckedIndexer<npy_ucs4> 对象 ind，索引 buf2.buf 的前 len2 个元素
                    CheckedIndexer<npy_ucs4> ind((npy_ucs4 *)buf2.buf, len2);
                    # 调用 findchar<npy_ucs4> 函数，在 ind 中查找 traverse_buf 指向的字符
                    res = findchar<npy_ucs4>(ind, len2, *traverse_buf);
                    break;
                }
            }
            # 如果未找到目标字符，则跳出循环
            if (res < 0) {
                break;
            }
            # 更新 num_bytes，减去 traverse_buf 下一个字符的字节数
            num_bytes -= traverse_buf.num_bytes_next_character();
            # j 自减
            j--;
            # 如果 j 大于 0，则 traverse_buf 向前移动一个位置
            if (j > 0) {
                traverse_buf--;
            }
        }
    }

    # 将 offset_buf 初始化为 buf1 + i
    Buffer offset_buf = buf1 + i;
    # 如果编码类型是 UTF8
    if (enc == ENCODING::UTF8) {
        # 将 offset_buf 的数据拷贝到 out 中，并返回 num_bytes
        offset_buf.buffer_memcpy(out, num_bytes);
        return num_bytes;
    }
    # 否则
    offset_buf.buffer_memcpy(out, j - i + 1);
    # 将 out 中索引为 j - i + 1 之后的位置填充为零
    out.buffer_fill_with_zeros_after_index(j - i + 1);
    # 返回 j - i + 1
    return j - i + 1;
// 如果要替换的字符串为空，则直接返回索引 0
template <typename char_type>
static inline npy_intp
findslice_for_replace(CheckedIndexer<char_type> buf1, npy_intp len1,
                      CheckedIndexer<char_type> buf2, npy_intp len2)
{
    if (len2 == 0) {
        return 0;
    }
    // 如果要替换的字符串长度为 1，则在 buf1 中查找该字符的位置并返回索引
    if (len2 == 1) {
        return (npy_intp) findchar(buf1, len1, *buf2);
    }
    // 使用快速搜索算法在 buf1 中查找 buf2，并返回第一个匹配的位置索引
    return (npy_intp) fastsearch(buf1.buffer, len1, buf2.buffer, len2, -1, FAST_SEARCH);
}

// 根据不同的编码方式进行字符串替换操作
template <ENCODING enc>
static inline size_t
string_replace(Buffer<enc> buf1, Buffer<enc> buf2, Buffer<enc> buf3, npy_int64 count,
               Buffer<enc> out)
{
    size_t len1 = buf1.num_codepoints();  // 获取 buf1 的字符数
    size_t len2 = buf2.num_codepoints();  // 获取 buf2 的字符数
    size_t len3 = buf3.num_codepoints();  // 获取 buf3 的字符数
    char *start;
    size_t length = len1;
    // 根据编码类型确定起始位置 start
    if (enc == ENCODING::UTF8) {
        start = buf1.after;  // UTF8 编码的起始位置为 buf1 的末尾后面
        length = 0;  // 针对 UTF8 编码，长度设为 0
    }
    else if (enc == ENCODING::UTF32) {
        start = buf1.buf + sizeof(npy_ucs4) * len1;  // UTF32 编码的起始位置为 buf1 末尾加上 len1 个 UCS4 字节长度
    }
    else {
        start = buf1.buf + len1;  // 其他编码的起始位置为 buf1 末尾
    }

    Buffer<enc> end1(start, length);  // 根据起始位置和长度创建 Buffer 对象 end1
    size_t span2, span3;

    switch(enc) {
        case ENCODING::ASCII:
        case ENCODING::UTF32:
        {
            span2 = len2;  // 对于 ASCII 和 UTF32 编码，span2 设为 len2
            span3 = len3;  // 对于 ASCII 和 UTF32 编码，span3 设为 len3
            break;
        }
        case ENCODING::UTF8:
        {
            span2 = buf2.after - buf2.buf;  // 对于 UTF8 编码，span2 是 buf2 的长度
            span3 = buf3.after - buf3.buf;  // 对于 UTF8 编码，span3 是 buf3 的长度
            break;
        }
    }

    size_t ret = 0;

    // 只有在替换次数 count 大于 0 且满足一些条件时才尝试进行替换操作
    if (count <= 0                      // 没有需要替换的内容
        || len1 < len2                  // 输入太小无法匹配
        || (len2 <= 0 && len3 <= 0)     // 匹配和替换字符串均为空
        || (len2 == len3 && buf2.strcmp(buf3) == 0)) {  // 匹配和替换字符串相同
        goto copy_rest;  // 跳转到 copy_rest 标签处，执行剩余复制操作
    }
    // 如果需要替换的字符串长度大于 0，则执行以下逻辑
    if (len2 > 0) {
        // 循环处理每个时间戳
        for (npy_int64 time = 0; time < count; time++) {
            npy_intp pos;  // 定义位置变量

            // 根据编码类型选择合适的索引器
            switch (enc) {
                case ENCODING::ASCII:
                case ENCODING::UTF8:
                {
                    // 使用 CheckedIndexer 处理 buf1 和 buf2，获取替换位置
                    CheckedIndexer<char> ind1(buf1.buf, end1 - buf1);
                    CheckedIndexer<char> ind2(buf2.buf, span2);
                    pos = findslice_for_replace(ind1, end1 - buf1, ind2, span2);
                    break;
                }
                case ENCODING::UTF32:
                {
                    // 使用 CheckedIndexer 处理 buf1 和 buf2，获取替换位置
                    CheckedIndexer<npy_ucs4> ind1((npy_ucs4 *)buf1.buf, end1 - buf1);
                    CheckedIndexer<npy_ucs4> ind2((npy_ucs4 *)buf2.buf, span2);
                    pos = findslice_for_replace(ind1, end1 - buf1, ind2, span2);
                    break;
                }
            }

            // 如果未找到替换位置，跳出循环
            if (pos < 0) {
                break;
            }

            // 将 buf1 的数据拷贝到 out，更新相关变量
            buf1.buffer_memcpy(out, pos);
            ret += pos;
            out.advance_chars_or_bytes(pos);
            buf1.advance_chars_or_bytes(pos);

            // 将 buf3 的数据拷贝到 out，更新相关变量
            buf3.buffer_memcpy(out, span3);
            ret += span3;
            out.advance_chars_or_bytes(span3);
            buf1.advance_chars_or_bytes(span2);
        }
    }
    else {  // 如果匹配字符串为空，则执行交替处理
        // 循环处理每个时间戳
        while (count > 0) {
            // 将 buf3 的数据拷贝到 out，更新相关变量
            buf3.buffer_memcpy(out, span3);
            ret += span3;
            out.advance_chars_or_bytes(span3);

            // 减少计数，如果小于等于 0，则跳出循环
            if (--count <= 0) {
                break;
            }

            // 根据编码类型选择处理方式
            switch (enc) {
                case ENCODING::ASCII:
                case ENCODING::UTF32:
                    // 将 buf1 的数据拷贝到 out，更新相关变量
                    buf1.buffer_memcpy(out, 1);
                    ret += 1;
                    break;
                case ENCODING::UTF8:
                    // 获取 UTF-8 字符的字节数量，并将 buf1 的数据拷贝到 out，更新相关变量
                    size_t n_bytes = buf1.num_bytes_next_character();
                    buf1.buffer_memcpy(out, n_bytes);
                    ret += n_bytes;
                    break;
            }

            // 更新 buf1 和 out 的位置
            buf1 += 1;
            out += 1;
        }
    }
copy_rest:
    // 使用 buf1 的 buffer_memcpy 方法将 buf1 的内容复制到 out 中，复制长度为 end1 - buf1
    buf1.buffer_memcpy(out, end1 - buf1);
    // 将复制的字节数累加到 ret 中
    ret += end1 - buf1;
    // 如果编码为 UTF8，则直接返回当前 ret 的值
    if (enc == ENCODING::UTF8) {
        return ret;
    }
    // 否则，在复制结束后，使用 out 的 buffer_fill_with_zeros_after_index 方法将 end1 - buf1 之后的字节填充为零
    out.buffer_fill_with_zeros_after_index(end1 - buf1);
    // 返回 ret 的值作为函数结果
    return ret;
}



template <ENCODING enc>
static inline npy_intp
string_expandtabs_length(Buffer<enc> buf, npy_int64 tabsize)
{
    // 计算 buf 中的代码点数目
    size_t len = buf.num_codepoints();

    npy_intp new_len = 0, line_pos = 0;

    // 复制一份 buf 到 tmp 中
    Buffer<enc> tmp = buf;
    // 遍历 buf 中的每个代码点
    for (size_t i = 0; i < len; i++) {
        // 获取当前代码点的值
        npy_ucs4 ch = *tmp;
        // 如果当前代码点是制表符 '\t'
        if (ch == '\t') {
            // 如果制表符大小 tabsize 大于 0
            if (tabsize > 0) {
                // 计算需要增加的长度，使得 line_pos 对齐到 tabsize 的倍数
                npy_intp incr = tabsize - (line_pos % tabsize);
                line_pos += incr;
                new_len += incr;
            }
        }
        else {
            // 如果当前代码点不是制表符
            line_pos += 1;
            // 获取当前代码点在缓冲区中所占的字节数
            size_t n_bytes = tmp.num_bytes_next_character();
            new_len += n_bytes;
            // 如果当前代码点是换行符 '\n' 或回车符 '\r'，将行位置重置为 0
            if (ch == '\n' || ch == '\r') {
                line_pos = 0;
            }
        }
        // 检查 new_len 是否超过 INT_MAX 或为负数，如果是则抛出溢出错误并返回 -1
        if (new_len > INT_MAX  || new_len < 0) {
            npy_gil_error(PyExc_OverflowError, "new string is too long");
            return -1;
        }
        // 指向下一个代码点
        tmp++;
    }
    // 返回计算得到的 new_len 作为函数结果
    return new_len;
}



template <ENCODING enc>
static inline npy_intp
string_expandtabs(Buffer<enc> buf, npy_int64 tabsize, Buffer<enc> out)
{
    // 计算 buf 中的代码点数目
    size_t len = buf.num_codepoints();

    npy_intp new_len = 0, line_pos = 0;

    // 复制一份 buf 到 tmp 中
    Buffer<enc> tmp = buf;
    // 遍历 buf 中的每个代码点
    for (size_t i = 0; i < len; i++) {
        // 获取当前代码点的值
        npy_ucs4 ch = *tmp;
        // 如果当前代码点是制表符 '\t'
        if (ch == '\t') {
            // 如果制表符大小 tabsize 大于 0
            if (tabsize > 0) {
                // 计算需要增加的长度，使得 line_pos 对齐到 tabsize 的倍数
                npy_intp incr = tabsize - (line_pos % tabsize);
                line_pos += incr;
                // 使用 out 的 buffer_memset 方法将 incr 个空格填充到输出缓冲区中
                new_len += out.buffer_memset((npy_ucs4) ' ', incr);
                // 移动 out 指针，向后移动 incr 个位置
                out += incr;
            }
        }
        else {
            // 如果当前代码点不是制表符
            line_pos++;
            // 使用 out 的 buffer_memset 方法将当前代码点 ch 写入输出缓冲区
            new_len += out.buffer_memset(ch, 1);
            // 移动 out 指针，向后移动一个位置
            out++;
            // 如果当前代码点是换行符 '\n' 或回车符 '\r'，将行位置重置为 0
            if (ch == '\n' || ch == '\r') {
                line_pos = 0;
            }
        }
        // 指向下一个代码点
        tmp++;
    }
    // 返回计算得到的 new_len 作为函数结果
    return new_len;
}



enum class JUSTPOSITION {
    CENTER, LEFT, RIGHT
};

template <ENCODING enc>
static inline npy_intp
string_pad(Buffer<enc> buf, npy_int64 width, npy_ucs4 fill, JUSTPOSITION pos, Buffer<enc> out)
{
    // 计算最终的字符串宽度，如果 width 小于等于 0，则设为 0
    size_t finalwidth = width > 0 ? width : 0;
    // 如果 finalwidth 超过了 PY_SSIZE_T_MAX，抛出溢出错误并返回 -1
    if (finalwidth > PY_SSIZE_T_MAX) {
        npy_gil_error(PyExc_OverflowError, "padded string is too long");
        return -1;
    }

    // 计算 buf 中的代码点数目和字节长度
    size_t len_codepoints = buf.num_codepoints();
    size_t len_bytes = buf.after - buf.buf;

    size_t len;
    // 根据编码类型确定长度的计算方式
    if (enc == ENCODING::UTF8) {
        len = len_bytes;
    }
    else {
        len = len_codepoints;
    }

    // 如果 buf 的代码点数目大于等于最终宽度，直接将 buf 复制到 out 中并返回长度
    if (len_codepoints >= finalwidth) {
        buf.buffer_memcpy(out, len);
        return (npy_intp) len;
    }

    size_t left, right;
    // 根据对齐方式 pos 计算左右填充量
    if (pos == JUSTPOSITION::CENTER) {
        size_t pad = finalwidth - len_codepoints;
        left = pad / 2 + (pad & finalwidth & 1);
        right = pad - left;
    }
    else if (pos == JUSTPOSITION::LEFT) {
        left = 0;
        right = finalwidth - len_codepoints;
    }



    // 在此处继续填写剩余的代码...
}
    # 否则分支：计算左侧填充空间和右侧填充空间
    else:
        left = finalwidth - len_codepoints;  # 计算左侧填充空间
        right = 0;  # 右侧填充空间设为0

    # 断言：确保左侧填充空间和右侧填充空间非负
    assert(left >= 0 or right >= 0);
    # 断言：确保左侧填充后的长度和右侧填充后的长度不会超出最大允许长度
    assert(left <= PY_SSIZE_T_MAX - len and right <= PY_SSIZE_T_MAX - (left + len));

    # 如果存在左侧填充空间大于0
    if (left > 0):
        # 将填充字符（fill）写入左侧填充空间，然后向前移动输出位置（out）
        out.advance_chars_or_bytes(out.buffer_memset(fill, left));

    # 将数据拷贝到输出缓冲区
    buf.buffer_memcpy(out, len);
    # 更新输出位置，增加已处理的代码点数
    out += len_codepoints;

    # 如果存在右侧填充空间大于0
    if (right > 0):
        # 将填充字符（fill）写入右侧填充空间，然后向前移动输出位置（out）
        out.advance_chars_or_bytes(out.buffer_memset(fill, right));

    # 返回最终输出的宽度
    return finalwidth;
// 结束当前的 C++ 函数定义，适用于模板函数和静态内联函数
}

// 定义一个静态内联函数 string_zfill，模板参数为编码类型 enc
template <ENCODING enc>
static inline npy_intp
string_zfill(Buffer<enc> buf, npy_int64 width, Buffer<enc> out)
{
    // 计算最终的宽度，如果指定宽度大于0，则使用指定宽度，否则为0
    size_t finalwidth = width > 0 ? width : 0;

    // 填充字符设为 '0'
    npy_ucs4 fill = '0';
    // 调用 string_pad 函数对 buf 进行填充，返回填充后的长度
    npy_intp new_len = string_pad(buf, width, fill, JUSTPOSITION::RIGHT, out);
    // 如果填充失败，返回 -1
    if (new_len == -1) {
        return -1;
    }

    // 计算偏移量，将 out 指针偏移 finalwidth - buf.num_codepoints() 个位置
    size_t offset = finalwidth - buf.num_codepoints();
    Buffer<enc> tmp = out + offset;

    // 取出 tmp 指针所指位置的字符
    npy_ucs4 c = *tmp;
    // 如果字符为 '+' 或 '-'，则将 tmp 指针位置的字符改为 fill
    if (c == '+' || c == '-') {
        tmp.buffer_memset(fill, 1);  // 将 tmp 指针位置的字符改为 fill
        out.buffer_memset(c, 1);     // 将 out 指针位置的字符改为 c
    }

    // 返回新长度 new_len
    return new_len;
}

// 定义一个静态内联函数 string_partition，模板参数为编码类型 enc
template <ENCODING enc>
static inline void
string_partition(Buffer<enc> buf1, Buffer<enc> buf2, npy_int64 idx,
                 Buffer<enc> out1, Buffer<enc> out2, Buffer<enc> out3,
                 npy_intp *final_len1, npy_intp *final_len2, npy_intp *final_len3,
                 STARTPOSITION pos)
{
    // 断言编码类型不为 UTF8，如果为 UTF8，则触发错误
    assert(enc != ENCODING::UTF8);

    // 计算 buf1 和 buf2 的字符数
    size_t len1 = buf1.num_codepoints();
    size_t len2 = buf2.num_codepoints();

    // 如果 buf2 长度为 0，则抛出异常并设置长度返回值为 -1
    if (len2 == 0) {
        npy_gil_error(PyExc_ValueError, "empty separator");
        *final_len1 = *final_len2 = *final_len3 = -1;
        return;
    }

    // 如果 idx 小于 0
    if (idx < 0) {
        // 根据 pos 的值进行处理
        if (pos == STARTPOSITION::FRONT) {
            buf1.buffer_memcpy(out1, len1);  // 将 buf1 复制到 out1
            *final_len1 = len1;  // 设置 out1 的最终长度
            *final_len2 = *final_len3 = 0;  // 设置 out2 和 out3 的最终长度为 0
        }
        else {
            buf1.buffer_memcpy(out3, len1);  // 将 buf1 复制到 out3
            *final_len1 = *final_len2 = 0;  // 设置 out1 和 out2 的最终长度为 0
            *final_len3 = len1;  // 设置 out3 的最终长度
        }
        return;
    }

    // 将 buf1 的前 idx 个字符复制到 out1
    buf1.buffer_memcpy(out1, idx);
    *final_len1 = idx;  // 设置 out1 的最终长度为 idx
    // 将 buf2 复制到 out2
    buf2.buffer_memcpy(out2, len2);
    *final_len2 = len2;  // 设置 out2 的最终长度为 len2
    // 将 buf1 的 idx+len2 到末尾的字符复制到 out3
    (buf1 + idx + len2).buffer_memcpy(out3, len1 - idx - len2);
    *final_len3 = len1 - idx - len2;  // 设置 out3 的最终长度
}
```