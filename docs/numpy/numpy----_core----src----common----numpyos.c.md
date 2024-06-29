# `.\numpy\numpy\_core\src\common\numpyos.c`

```
/*
 * 定义宏，确保使用最新的 NumPy API 版本
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
/*
 * 定义宏，标识多维数组模块
 */
#define _MULTIARRAYMODULE

/*
 * 清除 Python.h 中的 PY_SSIZE_T，确保仅使用最新的大小类型 API
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

/*
 * 引入 NumPy 的数组对象头文件
 */
#include "numpy/arrayobject.h"
/*
 * 引入 NumPy 的数学函数头文件
 */
#include "numpy/npy_math.h"

/*
 * 引入 NumPy 的配置文件
 */
#include "npy_config.h"

/*
 * 如果支持 strtold_l 并且未定义 _GNU_SOURCE，则定义 _GNU_SOURCE
 */
#if defined(HAVE_STRTOLD_L) && !defined(_GNU_SOURCE)
# define _GNU_SOURCE
#endif

/*
 * 引入本地化设置头文件
 */
#include <locale.h>
/*
 * 引入标准输入输出头文件
 */
#include <stdio.h>

/*
 * 如果支持 strtold_l，则引入标准库头文件
 */
#ifdef HAVE_STRTOLD_L
#include <stdlib.h>
/*
 * 如果支持 X/Open locale，则引入 xlocale.h
 * 注意：xlocale 在 glibc 2.26 中被移除
 */
#ifdef HAVE_XLOCALE_H
#include <xlocale.h>  // xlocale 在 glibc 2.26 中被移除，请参考 gh-8367
#endif
#endif

/*
 * 从 C99 标准第 7.19.6 节：指数始终至少包含两位数字，
 * 可以包含足够表示指数所需的额外位数。
 */

/*
 * 定义最小指数位数为 2
 */
#define MIN_EXPONENT_DIGITS 2

/*
 * 确保任何存在的指数至少具有 MIN_EXPONENT_DIGITS 指定的长度。
 * 参数 buffer 是输入输出缓冲区，buf_size 是缓冲区大小。
 */
static void
ensure_minimum_exponent_length(char* buffer, size_t buf_size)
{
    /*
     * 在 buffer 中查找 'e' 或 'E' 字符，并返回该字符的指针位置。
     */
    char *p = strpbrk(buffer, "eE");
    // 检查指针 p 是否有效，并且下一个字符是 '-' 或者 '+'
    if (p && (*(p + 1) == '-' || *(p + 1) == '+')) {
        // 设置指针 start 指向 p 的后两个位置
        char *start = p + 2;
        // 指数数字的总数
        int exponent_digit_cnt = 0;
        // 领先的零的数量
        int leading_zero_cnt = 0;
        // 是否处于领先零状态的标志
        int in_leading_zeros = 1;
        // 有效数字的数量
        int significant_digit_cnt;

        /* 跳过指数和符号 */
        p += 2;

        /* 找到指数的结束位置，并跟踪前导零 */
        while (*p && isdigit(Py_CHARMASK(*p))) {
            if (in_leading_zeros && *p == '0') {
                ++leading_zero_cnt;
            }
            if (*p != '0') {
                in_leading_zeros = 0;
            }
            ++p;
            ++exponent_digit_cnt;
        }

        significant_digit_cnt = exponent_digit_cnt - leading_zero_cnt;

        // 如果指数数字的数量等于 MIN_EXPONENT_DIGITS
        if (exponent_digit_cnt == MIN_EXPONENT_DIGITS) {
            /*
             * 如果有 2 个确切的数字，我们完成了，
             * 不管它们包含什么
             */
        }
        else if (exponent_digit_cnt > MIN_EXPONENT_DIGITS) {
            int extra_zeros_cnt;

            /*
             * 指数中有超过 2 个数字。看看我们是否可以删除一些前导零
             */
            if (significant_digit_cnt < MIN_EXPONENT_DIGITS) {
                significant_digit_cnt = MIN_EXPONENT_DIGITS;
            }
            extra_zeros_cnt = exponent_digit_cnt - significant_digit_cnt;

            /*
             * 从指数的前端删除 extra_zeros_cnt 个字符
             */
            assert(extra_zeros_cnt >= 0);

            /*
             * 将 significant_digit_cnt 加一，以复制
             * 结尾的 0 字节，从而设置长度
             */
            memmove(start, start + extra_zeros_cnt, significant_digit_cnt + 1);
        }
        else {
            /*
             * 如果数字少于 2 个，则添加零
             * 直到达到 2 个，如果有足够的空间
             */
            int zeros = MIN_EXPONENT_DIGITS - exponent_digit_cnt;
            if (start + zeros + exponent_digit_cnt + 1 < buffer + buf_size) {
                memmove(start + zeros, start, exponent_digit_cnt + 1);
                memset(start, '0', zeros);
            }
        }
    }
/*
 * Ensure that buffer has a decimal point in it.  The decimal point
 * will not be in the current locale, it will always be '.'
 */
static void
ensure_decimal_point(char* buffer, size_t buf_size)
{
    int insert_count = 0;
    char* chars_to_insert;

    /* search for the first non-digit character */
    char *p = buffer;
    if (*p == '-' || *p == '+')
        /*
         * Skip leading sign, if present.  I think this could only
         * ever be '-', but it can't hurt to check for both.
         */
        ++p;
    while (*p && isdigit(Py_CHARMASK(*p))) {
        ++p;
    }
    if (*p == '.') {
        if (isdigit(Py_CHARMASK(*(p+1)))) {
            /*
             * Nothing to do, we already have a decimal
             * point and a digit after it.
             */
        }
        else {
            /*
             * We have a decimal point, but no following
             * digit.  Insert a zero after the decimal.
             */
            ++p;
            chars_to_insert = "0";
            insert_count = 1;
        }
    }
    else {
        chars_to_insert = ".0";
        insert_count = 2;
    }
    if (insert_count) {
        size_t buf_len = strlen(buffer);
        if (buf_len + insert_count + 1 >= buf_size) {
            /*
             * If there is not enough room in the buffer
             * for the additional text, just skip it.  It's
             * not worth generating an error over.
             */
        }
        else {
            memmove(p + insert_count, p, buffer + strlen(buffer) - p + 1);
            memcpy(p, chars_to_insert, insert_count);
        }
    }
}

/* see FORMATBUFLEN in unicodeobject.c */
#define FLOAT_FORMATBUFLEN 120

/*
 * Given a string that may have a decimal point in the current
 * locale, change it back to a dot.  Since the string cannot get
 * longer, no need for a maximum buffer size parameter.
 */
static void
change_decimal_from_locale_to_dot(char* buffer)
{
    struct lconv *locale_data = localeconv();
    const char *decimal_point = locale_data->decimal_point;

    if (decimal_point[0] != '.' || decimal_point[1] != 0) {
        size_t decimal_point_len = strlen(decimal_point);

        if (*buffer == '+' || *buffer == '-') {
            buffer++;
        }
        while (isdigit(Py_CHARMASK(*buffer))) {
            buffer++;
        }
        if (strncmp(buffer, decimal_point, decimal_point_len) == 0) {
            *buffer = '.';
            buffer++;
            if (decimal_point_len > 1) {
                /* buffer needs to get smaller */
                size_t rest_len = strlen(buffer + (decimal_point_len - 1));
                memmove(buffer, buffer + (decimal_point_len - 1), rest_len);
                buffer[rest_len] = 0;
            }
        }
    }
}

/*
 * Check that the format string is a valid one for NumPyOS_ascii_format*
 */
static int
check_ascii_format(const char *format)
{
    char format_char;
    size_t format_len = strlen(format);
    /* 获取格式字符串的最后一个字符作为格式字符 */
    format_char = format[format_len - 1];

    /* 如果格式字符串的第一个字符不是 '%'，则返回错误 */
    if (format[0] != '%') {
        return -1;
    }

    /*
     * 我不确定为什么这个测试在这里。它确保格式字符串的第一个字符后面
     * 没有单引号、小写的 'l' 或百分号。这与大约 10 行前的注释掉的测试
     * 是相反的。
     */
    if (strpbrk(format + 1, "'l%")) {
        return -1;
    }

    /*
     * 同样让人好奇的是，这个函数接受像 "%xg" 这样对于浮点数无效的
     * 格式字符串。总体来说，这个函数的接口设计并不是很好，但由于它
     * 是一个公共 API，改变它很困难。
     */
    if (!(format_char == 'e' || format_char == 'E'
          || format_char == 'f' || format_char == 'F'
          || format_char == 'g' || format_char == 'G')) {
        return -1;
    }

    /* 如果所有的格式验证都通过，则返回成功 */
    return 0;
/*
 * Fix the generated string: make sure the decimal is ., that exponent has a
 * minimal number of digits, and that it has a decimal + one digit after that
 * decimal if decimal argument != 0 (Same effect that 'Z' format in
 * PyOS_ascii_formatd)
 */
static char*
fix_ascii_format(char* buf, size_t buflen, int decimal)
{
    /*
     * Get the current locale, and find the decimal point string.
     * Convert that string back to a dot.
     */
    change_decimal_from_locale_to_dot(buf);

    /*
     * If an exponent exists, ensure that the exponent is at least
     * MIN_EXPONENT_DIGITS digits, providing the buffer is large enough
     * for the extra zeros.  Also, if there are more than
     * MIN_EXPONENT_DIGITS, remove as many zeros as possible until we get
     * back to MIN_EXPONENT_DIGITS
     */
    ensure_minimum_exponent_length(buf, buflen);

    // 如果 decimal 参数不为 0，确保在 buf 中有小数点及至少一位小数
    if (decimal != 0) {
        ensure_decimal_point(buf, buflen);
    }

    // 返回修正后的 buf 字符串
    return buf;
}

/*
 * NumPyOS_ascii_format*:
 *      - buffer: A buffer to place the resulting string in
 *      - buf_size: The length of the buffer.
 *      - format: The printf()-style format to use for the code to use for
 *      converting.
 *      - value: The value to convert
 *      - decimal: if != 0, always has a decimal, and at leasat one digit after
 *      the decimal. This has the same effect as passing 'Z' in the original
 *      PyOS_ascii_formatd
 *
 * This is similar to PyOS_ascii_formatd in python > 2.6, except that it does
 * not handle 'n', and handles nan / inf.
 *
 * Converts a #gdouble to a string, using the '.' as decimal point. To format
 * the number you pass in a printf()-style format string. Allowed conversion
 * specifiers are 'e', 'E', 'f', 'F', 'g', 'G'.
 *
 * Return value: The pointer to the buffer with the converted string.
 */
#define ASCII_FORMAT(type, suffix, print_type)                          \
    NPY_NO_EXPORT char*                                                 \
    NumPyOS_ascii_format ## suffix(char *buffer, size_t buf_size,       \
                                   const char *format,                  \
                                   type val, int decimal)               \
    {                                                                   \
        // 检查浮点数是否有限，即不是NaN或无穷大
        if (npy_isfinite(val)) {                                        
            // 检查是否要求输出ASCII格式，如果是，则返回空指针
            if (check_ascii_format(format)) {                           
                return NULL;                                            
            }                                                           
            // 将浮点数按指定格式输出到缓冲区中
            PyOS_snprintf(buffer, buf_size, format, (print_type)val);   
            // 修正ASCII格式，确保输出正确格式的数据
            return fix_ascii_format(buffer, buf_size, decimal);         
        }                                                               
        // 如果浮点数是NaN
        else if (npy_isnan(val)){                                       
            // 如果缓冲区大小不足以容纳"nan"，则返回空指针
            if (buf_size < 4) {                                         
                return NULL;                                            
            }                                                           
            // 将"nan"复制到缓冲区
            strcpy(buffer, "nan");                                      
        }                                                               
        // 如果浮点数是无穷大
        else {                                                          
            // 如果浮点数为负无穷且缓冲区大小不足以容纳"-inf"，则返回空指针
            if (npy_signbit(val)) {                                     
                if (buf_size < 5) {                                     
                    return NULL;                                        
                }                                                       
                // 将"-inf"复制到缓冲区
                strcpy(buffer, "-inf");                                 
            }                                                           
            // 如果浮点数为正无穷且缓冲区大小不足以容纳"inf"，则返回空指针
            else {                                                      
                if (buf_size < 4) {                                     
                    return NULL;                                        
                }                                                       
                // 将"inf"复制到缓冲区
                strcpy(buffer, "inf");                                  
            }                                                           
        }                                                               
        // 返回填充好数据的缓冲区
        return buffer;                                                  
    }
/*
 * ASCII_FORMAT(float, f, float)
 *
 * Macro definition for formatting float types with ASCII representation.
 */
ASCII_FORMAT(float, f, float)

/*
 * ASCII_FORMAT(double, d, double)
 *
 * Macro definition for formatting double types with ASCII representation.
 */
ASCII_FORMAT(double, d, double)

#ifndef FORCE_NO_LONG_DOUBLE_FORMATTING
/*
 * ASCII_FORMAT(long double, l, long double)
 *
 * Macro definition for formatting long double types with ASCII representation,
 * unless FORCE_NO_LONG_DOUBLE_FORMATTING is defined.
 */
ASCII_FORMAT(long double, l, long double)
#else
/*
 * ASCII_FORMAT(long double, l, double)
 *
 * Macro definition for formatting long double types with ASCII representation,
 * when FORCE_NO_LONG_DOUBLE_FORMATTING is defined.
 */
ASCII_FORMAT(long double, l, double)
#endif

/*
 * NumPyOS_ascii_isspace:
 *
 * Function to determine if the provided character is a whitespace character
 * in the ASCII character set.
 */
NPY_NO_EXPORT int
NumPyOS_ascii_isspace(int c)
{
    return c == ' ' || c == '\f' || c == '\n' || c == '\r' || c == '\t'
                    || c == '\v';
}

/*
 * NumPyOS_ascii_isalpha:
 *
 * Function to determine if the provided character is an alphabetic character
 * (a-z or A-Z) in the ASCII character set.
 */
NPY_NO_EXPORT int
NumPyOS_ascii_isalpha(char c)
{
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

/*
 * NumPyOS_ascii_isdigit:
 *
 * Function to determine if the provided character is a digit (0-9)
 * in the ASCII character set.
 */
NPY_NO_EXPORT int
NumPyOS_ascii_isdigit(char c)
{
    return (c >= '0' && c <= '9');
}

/*
 * NumPyOS_ascii_isalnum:
 *
 * Function to determine if the provided character is either a digit or
 * an alphabetic character (a-z, A-Z, or 0-9) in the ASCII character set.
 */
NPY_NO_EXPORT int
NumPyOS_ascii_isalnum(char c)
{
    return NumPyOS_ascii_isdigit(c) || NumPyOS_ascii_isalpha(c);
}

/*
 * NumPyOS_ascii_islower:
 *
 * Function to determine if the provided character is a lowercase alphabetic
 * character (a-z) in the ASCII character set.
 */
NPY_NO_EXPORT int
NumPyOS_ascii_islower(char c)
{
    return c >= 'a' && c <= 'z';
}

/*
 * NumPyOS_ascii_isupper:
 *
 * Function to determine if the provided character is an uppercase alphabetic
 * character (A-Z) in the ASCII character set.
 */
NPY_NO_EXPORT int
NumPyOS_ascii_isupper(char c)
{
    return c >= 'A' && c <= 'Z';
}

/*
 * NumPyOS_ascii_tolower:
 *
 * Function to convert the provided character to its lowercase equivalent
 * in the ASCII character set.
 */
NPY_NO_EXPORT int
NumPyOS_ascii_tolower(int c)
{
    if (c >= 'A' && c <= 'Z') {
        return c + ('a'-'A');
    }
    return c;
}

/*
 * NumPyOS_ascii_strncasecmp:
 *
 * Function to compare two strings case-insensitively up to a specified length,
 * considering ASCII character set rules.
 */
static int
NumPyOS_ascii_strncasecmp(const char* s1, const char* s2, size_t len)
{
    while (len > 0 && *s1 != '\0' && *s2 != '\0') {
        int diff = NumPyOS_ascii_tolower(*s1) - NumPyOS_ascii_tolower(*s2);
        if (diff != 0) {
            return diff;
        }
        ++s1;
        ++s2;
        --len;
    }
    if (len > 0) {
        return *s1 - *s2;
    }
    return 0;
}

/*
 * NumPyOS_ascii_strtod_plain:
 *
 * Function similar to PyOS_ascii_strtod, without enhanced features,
 * compatible with Python versions >= 2.7.
 */
static double
NumPyOS_ascii_strtod_plain(const char *s, char** endptr)
{
    double result;
    NPY_ALLOW_C_API_DEF;
    NPY_ALLOW_C_API;
    result = PyOS_string_to_double(s, endptr, NULL);
    if (PyErr_Occurred()) {
        if (endptr) {
            *endptr = (char*)s;
        }
        PyErr_Clear();
    }
    NPY_DISABLE_C_API;
    return result;
}

/*
 * NumPyOS_ascii_strtod:
 *
 * Function to convert a string to a double, addressing bugs in PyOS_ascii_strtod.
 */
NPY_NO_EXPORT double
NumPyOS_ascii_strtod(const char *s, char** endptr)
{
    const char *p;
    double result;

    while (NumPyOS_ascii_isspace(*s)) {
        ++s;
    }

    /*
     * Recognize POSIX inf/nan representations on all platforms.
     */
    p = s;
    result = 1.0;
    if (*p == '-') {
        result = -1.0;
        ++p;
    }
    else if (*p == '+') {
        ++p;
    }
    # 如果字符串以 "nan" 开头，表示遇到 NaN（Not a Number）情况
    if (NumPyOS_ascii_strncasecmp(p, "nan", 3) == 0) {
        # 将指针 p 向后移动 3 个字符，跳过 "nan"
        p += 3;
        # 如果接下来是 '(', 继续处理括号内的内容
        if (*p == '(') {
            ++p;
            # 处理括号内的字母数字和下划线，直到结束括号 ')'
            while (NumPyOS_ascii_isalnum(*p) || *p == '_') {
                ++p;
            }
            if (*p == ')') {
                ++p;
            }
        }
        # 如果有 endptr 参数，将指针 p 赋值给它
        if (endptr != NULL) {
            *endptr = (char*)p;
        }
        # 返回 NaN 常量
        return NPY_NAN;
    }
    # 如果字符串以 "inf" 开头，表示遇到正无穷（infinity）情况
    else if (NumPyOS_ascii_strncasecmp(p, "inf", 3) == 0) {
        # 将指针 p 向后移动 3 个字符，跳过 "inf"
        p += 3;
        # 如果接下来是 "inity", 继续处理完整的 "infinity"
        if (NumPyOS_ascii_strncasecmp(p, "inity", 5) == 0) {
            p += 5;
        }
        # 如果有 endptr 参数，将指针 p 赋值给它
        if (endptr != NULL) {
            *endptr = (char*)p;
        }
        # 返回正无穷常量
        return result * NPY_INFINITY;
    }
    /* End of ##1 */

    # 如果不是 NaN 或正无穷，则调用 NumPy 的字符串转换为 double 函数继续处理
    return NumPyOS_ascii_strtod_plain(s, endptr);
}



NPY_NO_EXPORT long double
NumPyOS_ascii_strtold(const char *s, char** endptr)
{
    const char *p;
    long double result;
#ifdef HAVE_STRTOLD_L
    locale_t clocale;
#endif

    // 跳过输入字符串开头的空白字符
    while (NumPyOS_ascii_isspace(*s)) {
        ++s;
    }

    /*
     * ##1
     *
     * Recognize POSIX inf/nan representations on all platforms.
     * 识别所有平台上的 POSIX inf/nan 表示形式。
     */
    p = s;
    result = 1.0;
    if (*p == '-') {
        result = -1.0;
        ++p;
    }
    else if (*p == '+') {
        ++p;
    }
    // 检查是否为 NaN
    if (NumPyOS_ascii_strncasecmp(p, "nan", 3) == 0) {
        p += 3;
        // 处理 NaN 可选的附加信息
        if (*p == '(') {
            ++p;
            // 跳过可能存在的附加标识符
            while (NumPyOS_ascii_isalnum(*p) || *p == '_') {
                ++p;
            }
            if (*p == ')') {
                ++p;
            }
        }
        // 如果有 endptr，将其设置为指向字符串结尾的指针
        if (endptr != NULL) {
            *endptr = (char*)p;
        }
        // 返回 NaN 常量
        return NPY_NAN;
    }
    // 检查是否为 Infinity
    else if (NumPyOS_ascii_strncasecmp(p, "inf", 3) == 0) {
        p += 3;
        // 检查是否为 "infinity" 的后缀
        if (NumPyOS_ascii_strncasecmp(p, "inity", 5) == 0) {
            p += 5;
        }
        // 如果有 endptr，将其设置为指向字符串结尾的指针
        if (endptr != NULL) {
            *endptr = (char*)p;
        }
        // 返回 ±Infinity
        return result * NPY_INFINITY;
    }
    /* End of ##1 */

#ifdef HAVE_STRTOLD_L
    // 使用本地化设置解析长双精度浮点数
    clocale = newlocale(LC_ALL_MASK, "C", NULL);
    if (clocale) {
        errno = 0;
        result = strtold_l(s, endptr, clocale);
        freelocale(clocale);
    }
    else {
        // 如果没有本地化支持，直接将 endptr 设置为 s，并返回 0
        if (endptr != NULL) {
            *endptr = (char*)s;
        }
        result = 0;
    }
    return result;
#else
    // 没有本地化支持时，使用标准 strtod 解析浮点数
    return NumPyOS_ascii_strtod(s, endptr);
#endif
}

/*
 * read_numberlike_string:
 *      * fp: FILE pointer
 *      * value: Place to store the value read
 *
 * Read what looks like valid numeric input and store it in a buffer
 * for later parsing as a number.
 *
 * Similarly to fscanf, this function always consumes leading whitespace,
 * and any text that could be the leading part in valid input.
 *
 * Return value: similar to fscanf.
 *      * 0 if no number read,
 *      * 1 if a number read,
 *      * EOF if end-of-file met before reading anything.
 */
static int
read_numberlike_string(FILE *fp, char *buffer, size_t buflen)
{

    char *endp;
    char *p;
    int c;
    int ok;

    /*
     * Fill buffer with the leftmost matching part in regexp
     *
     *     \s*[+-]? ( [0-9]*\.[0-9]+([eE][+-]?[0-9]+)
     *              | nan  (  \([:alphanum:_]*\) )?
     *              | inf(inity)?
     *              )
     *
     * case-insensitively.
     *
     * The "do { ... } while (0)" wrapping in macros ensures that they behave
     * properly eg. in "if ... else" structures.
     */
    
    // 宏定义：跳转到 buffer_filled 标签处
#define END_MATCH()                                                         \
        goto buffer_filled


注：以上是对给定代码的详细注释，按照要求进行了逐行解释。
    /* 定义宏：获取下一个字符，并将其放入缓冲区，处理 EOF 和缓冲区溢出 */
#define NEXT_CHAR()                                                         \
        do {                                                                \
            if (c == EOF || endp >= buffer + buflen - 1)            \
                END_MATCH();                                                \
            *endp++ = (char)c;                                              \
            c = getc(fp);                                                   \
        } while (0)

    /* 定义宏：匹配不区分大小写的字符串 */
#define MATCH_ALPHA_STRING_NOCASE(string)                                   \
        do {                                                                \
            for (p=(string); *p!='\0' && (c==*p || c+('a'-'A')==*p); ++p)   \
                NEXT_CHAR();                                                \
            if (*p != '\0') END_MATCH();                                    \
        } while (0)

    /* 定义宏：匹配零个或一个条件 */
#define MATCH_ONE_OR_NONE(condition)                                        \
        do { if (condition) NEXT_CHAR(); } while (0)

    /* 定义宏：匹配一个或多个条件 */
#define MATCH_ONE_OR_MORE(condition)                                        \
        do {                                                                \
            ok = 0;                                                         \
            while (condition) { NEXT_CHAR(); ok = 1; }                      \
            if (!ok) END_MATCH();                                           \
        } while (0)

    /* 定义宏：匹配零个或多个条件 */
#define MATCH_ZERO_OR_MORE(condition)                                       \
        while (condition) { NEXT_CHAR(); }

    /* 1. 模拟 fscanf 处理 EOF */
    c = getc(fp);
    if (c == EOF) {
        return EOF;
    }

    /* 2. 无条件消耗前导空白 */
    while (NumPyOS_ascii_isspace(c)) {
        c = getc(fp);
    }

    /* 3. 开始读取匹配输入到缓冲区 */
    endp = buffer;

    /* 4.1 符号部分（可选） */
    MATCH_ONE_OR_NONE(c == '+' || c == '-');

    /* 4.2 nan, inf, infinity；不区分大小写 */
    if (c == 'n' || c == 'N') {
        NEXT_CHAR();
        MATCH_ALPHA_STRING_NOCASE("an");

        /* 接受 nan([:alphanum:_]*)，类似于 strtod */
        if (c == '(') {
            NEXT_CHAR();
            MATCH_ZERO_OR_MORE(NumPyOS_ascii_isalnum(c) || c == '_');
            if (c == ')') {
                NEXT_CHAR();
            }
        }
        END_MATCH();
    }
    else if (c == 'i' || c == 'I') {
        NEXT_CHAR();
        MATCH_ALPHA_STRING_NOCASE("nfinity");
        END_MATCH();
    }

    /* 4.3 小数部分 */
    MATCH_ZERO_OR_MORE(NumPyOS_ascii_isdigit(c));

    if (c == '.') {
        NEXT_CHAR();
        MATCH_ONE_OR_MORE(NumPyOS_ascii_isdigit(c));
    }

    /* 4.4 指数部分 */
    if (c == 'e' || c == 'E') {
        NEXT_CHAR();
        MATCH_ONE_OR_NONE(c == '+' || c == '-');
        MATCH_ONE_OR_MORE(NumPyOS_ascii_isdigit(c));
    }

    END_MATCH();

buffer_filled:

    ungetc(c, fp);
    *endp = '\0';

    /* 返回1表示有内容被读取，否则返回0 */
    # 如果 buffer 等于 endp，则返回 0；否则返回 1。
    return (buffer == endp) ? 0 : 1;
/*
 * NumPyOS_ascii_ftolf:
 *      * fp: FILE pointer
 *      * value: Place to store the value read
 *
 * Similar to PyOS_ascii_strtod, except that it reads input from a file.
 *
 * Similarly to fscanf, this function always consumes leading whitespace,
 * and any text that could be the leading part in valid input.
 *
 * Return value: similar to fscanf.
 *      * 0 if no number read,
 *      * 1 if a number read,
 *      * EOF if end-of-file met before reading anything.
 */
NPY_NO_EXPORT int
NumPyOS_ascii_ftolf(FILE *fp, double *value)
{
    // 缓冲区用于存储从文件中读取的数字字符串
    char buffer[FLOAT_FORMATBUFLEN + 1];
    char *p;
    int r;

    // 从文件中读取类似数字的字符串
    r = read_numberlike_string(fp, buffer, FLOAT_FORMATBUFLEN+1);

    // 如果成功读取数字字符串
    if (r != EOF && r != 0) {
        // 使用 NumPyOS_ascii_strtod 将字符串转换为双精度浮点数
        *value = NumPyOS_ascii_strtod(buffer, &p);
        // 如果转换后的指针与缓冲区指针相同，表明没有有效数字
        r = (p == buffer) ? 0 : 1;
    }
    // 返回读取操作的结果
    return r;
}

/*
 * NumPyOS_ascii_ftoLf:
 *      * fp: FILE pointer
 *      * value: Place to store the value read
 *
 * Similar to PyOS_ascii_strtod, except that it reads input from a file.
 *
 * Similarly to fscanf, this function always consumes leading whitespace,
 * and any text that could be the leading part in valid input.
 *
 * Return value: similar to fscanf.
 *      * 0 if no number read,
 *      * 1 if a number read,
 *      * EOF if end-of-file met before reading anything.
 */
NPY_NO_EXPORT int
NumPyOS_ascii_ftoLf(FILE *fp, long double *value)
{
    // 缓冲区用于存储从文件中读取的数字字符串
    char buffer[FLOAT_FORMATBUFLEN + 1];
    char *p;
    int r;

    // 从文件中读取类似数字的字符串
    r = read_numberlike_string(fp, buffer, FLOAT_FORMATBUFLEN+1);

    // 如果成功读取数字字符串
    if (r != EOF && r != 0) {
        // 使用 NumPyOS_ascii_strtold 将字符串转换为长双精度浮点数
        *value = NumPyOS_ascii_strtold(buffer, &p);
        // 如果转换后的指针与缓冲区指针相同，表明没有有效数字
        r = (p == buffer) ? 0 : 1;
    }
    // 返回读取操作的结果
    return r;
}

/*
 * NumPyOS_strtoll:
 *      * str: 要转换为长长整型的字符串
 *      * endptr: 如果提供，将返回第一个非数字字符的指针
 *      * base: 数字的进制，如 10 表示十进制
 *
 * 使用标准函数 strtoll 将字符串转换为长长整型数。
 */
NPY_NO_EXPORT npy_longlong
NumPyOS_strtoll(const char *str, char **endptr, int base)
{
    return strtoll(str, endptr, base);
}

/*
 * NumPyOS_strtoull:
 *      * str: 要转换为无符号长长整型的字符串
 *      * endptr: 如果提供，将返回第一个非数字字符的指针
 *      * base: 数字的进制，如 10 表示十进制
 *
 * 使用标准函数 strtoull 将字符串转换为无符号长长整型数。
 */
NPY_NO_EXPORT npy_ulonglong
NumPyOS_strtoull(const char *str, char **endptr, int base)
{
    return strtoull(str, endptr, base);
}

#ifdef _MSC_VER

#include <stdlib.h>

#if _MSC_VER >= 1900
/* npy3k_compat.h uses this function in the _Py_BEGIN/END_SUPPRESS_IPH
 * macros. It does not need to be defined when building using MSVC
 * earlier than 14.0 (_MSC_VER == 1900).
 */

// Microsoft Visual C++ 1900 及更高版本的静默无效参数处理器
static void __cdecl _silent_invalid_parameter_handler(
    wchar_t const* expression,
    wchar_t const* function,
    wchar_t const* file,
    unsigned int line,
    uintptr_t pReserved) { }

_invalid_parameter_handler _Py_silent_invalid_parameter_handler = _silent_invalid_parameter_handler;

#endif

#endif
```