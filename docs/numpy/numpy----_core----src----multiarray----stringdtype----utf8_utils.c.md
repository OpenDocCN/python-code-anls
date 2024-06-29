# `.\numpy\numpy\_core\src\multiarray\stringdtype\utf8_utils.c`

```
// 导入 Python.h 头文件，包含 Python C API
#include <Python.h>
// 定义 NPY_NO_DEPRECATED_API 为 NPY_API_VERSION，避免使用废弃的 API
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
// 定义 _MULTIARRAYMODULE 宏，用于多维数组模块
#define _MULTIARRAYMODULE
// 定义 _UMATHMODULE 宏，用于通用数学函数模块

// 包含 numpy/ndarraytypes.h 头文件，定义了 ndarray 的数据类型和宏
#include "numpy/ndarraytypes.h"
// 包含 utf8_utils.h 头文件，提供 UTF-8 编码的工具函数

// 给定 UTF-8 字节 *c*，将 *code* 设置为对应的 unicode 码点，返回字符所占字节数。
// 不进行验证或错误检查：假定 *c* 是有效的 UTF-8 编码
NPY_NO_EXPORT size_t
utf8_char_to_ucs4_code(const unsigned char *c, Py_UCS4 *code)
{
    // 如果第一个字节的最高位为 0，则为 7 位 ASCII 码
    if (c[0] <= 0x7F) {
        // 0zzzzzzz -> 0zzzzzzz
        *code = (Py_UCS4)(c[0]);
        return 1;
    }
    // 如果第一个字节的最高两位为 110，则为 2 字节 UTF-8 序列
    else if (c[0] <= 0xDF) {
        // 110yyyyy 10zzzzzz -> 00000yyy yyzzzzzz
        *code = (Py_UCS4)(((c[0] << 6) + c[1]) - ((0xC0 << 6) + 0x80));
        return 2;
    }
    // 如果第一个字节的最高三位为 1110，则为 3 字节 UTF-8 序列
    else if (c[0] <= 0xEF) {
        // 1110xxxx 10yyyyyy 10zzzzzz -> xxxxyyyy yyzzzzzz
        *code = (Py_UCS4)(((c[0] << 12) + (c[1] << 6) + c[2]) -
                          ((0xE0 << 12) + (0x80 << 6) + 0x80));
        return 3;
    }
    // 如果第一个字节的最高四位为 11110，则为 4 字节 UTF-8 序列
    else {
        // 11110www 10xxxxxx 10yyyyyy 10zzzzzz -> 000wwwxx xxxxyyyy yyzzzzzz
        *code = (Py_UCS4)(((c[0] << 18) + (c[1] << 12) + (c[2] << 6) + c[3]) -
                          ((0xF0 << 18) + (0x80 << 12) + (0x80 << 6) + 0x80));
        return 4;
    }
}

// 从 *c* 开始向前查找有效 UTF-8 字符的起始位置，*nchar* 是要查找的字符数
NPY_NO_EXPORT const unsigned char*
find_previous_utf8_character(const unsigned char *c, size_t nchar)
{
    while (nchar > 0) {
        do
        {
            // 假设 UTF-8 格式正确，不检查是否超出字符串开始位置
            c--;
        // UTF-8 字符的第一个字节要么最高位为 0，要么最高两位都为 1
        } while ((*c & 0xC0) == 0x80);
        nchar--;
    }
    return c;
}

// 返回 UTF-8 字符 *c* 所需的字节数
NPY_NO_EXPORT int
num_bytes_for_utf8_character(const unsigned char *c) {
    if (c[0] <= 0x7F) {
        return 1;
    }
    else if (c[0] <= 0xDF) {
        return 2;
    }
    else if (c[0] <= 0xEF) {
        return 3;
    }
    return 4;
}

// 返回给定 unicode 码点 *code* 在 UTF-8 编码下所需的字节数
NPY_NO_EXPORT int
num_utf8_bytes_for_codepoint(uint32_t code)
{
    if (code <= 0x7F) {
        return 1;
    }
    else if (code <= 0x07FF) {
        return 2;
    }
    else if (code <= 0xFFFF) {
        if ((code >= 0xD800) && (code <= 0xDFFF)) {
            // 代理对在 UCS4 中无效
            return -1;
        }
        return 3;
        }
    else if (code <= 0x10FFFF) {
        return 4;
    }
    else {
        // 码点超出有效的 Unicode 范围
        return -1;
    }
}

// 查找存储由 *codepoints* 表示的 UTF-8 编码字符串所需的字节数。
// *codepoints* 数组的长度为 *max_length*，但可能以空码点填充。
// *num_codepoints* 是不包括尾部空码点的码点数。
// 成功时返回 0，当发现无效码点时返回 -1。
NPY_NO_EXPORT int
// 计算给定 UCS4 编码点数组的 UTF-8 编码所需的字节数和非空编码点数量
utf8_size(const Py_UCS4 *codepoints, long max_length, size_t *num_codepoints,
          size_t *utf8_bytes)
{
    // 初始化 UCS4 编码点数组的长度为最大长度
    size_t ucs4len = max_length;

    // 去除末尾的空编码点，更新 ucs4len 为非空编码点的数量
    while (ucs4len > 0 && codepoints[ucs4len - 1] == 0) {
        ucs4len--;
    }
    // ucs4len 现在是非尾部空编码点的数量。

    // 初始化 UTF-8 编码所需的总字节数
    size_t num_bytes = 0;

    // 遍历 UCS4 编码点数组，计算每个编码点的 UTF-8 编码字节数，并累加到 num_bytes 中
    for (size_t i = 0; i < ucs4len; i++) {
        Py_UCS4 code = codepoints[i];
        // 调用函数获取当前编码点的 UTF-8 编码字节数
        int codepoint_bytes = num_utf8_bytes_for_codepoint((uint32_t)code);
        // 如果编码点无效，返回错误标志 -1
        if (codepoint_bytes == -1) {
            return -1;
        }
        // 累加当前编码点的 UTF-8 编码字节数到 num_bytes
        num_bytes += codepoint_bytes;
    }

    // 将计算得到的非空编码点数量和 UTF-8 编码总字节数分别存入传入的指针中
    *num_codepoints = ucs4len;
    *utf8_bytes = num_bytes;

    // 返回成功标志 0
    return 0;
}

// 将 UCS4 编码点 *code* 转换为 UTF-8 字符数组 *c*，假设 *c* 是一个以 0 填充的 4 字节数组，
// *code* 是一个有效的编码点且不进行任何错误检查！返回 UTF-8 字符的字节数。
NPY_NO_EXPORT size_t
ucs4_code_to_utf8_char(Py_UCS4 code, char *c)
{
    // 如果编码点在 ASCII 范围内，直接转换为单字节 UTF-8 字符
    if (code <= 0x7F) {
        // 0zzzzzzz -> 0zzzzzzz
        c[0] = (char)code;
        return 1;
    }
    // 如果编码点在 0x80 到 0x07FF 范围内，转换为两字节 UTF-8 字符
    else if (code <= 0x07FF) {
        // 00000yyy yyzzzzzz -> 110yyyyy 10zzzzzz
        c[0] = (0xC0 | (code >> 6));
        c[1] = (0x80 | (code & 0x3F));
        return 2;
    }
    // 如果编码点在 0x0800 到 0xFFFF 范围内，转换为三字节 UTF-8 字符
    else if (code <= 0xFFFF) {
        // xxxxyyyy yyzzzzzz -> 110yyyyy 10zzzzzz
        c[0] = (0xe0 | (code >> 12));
        c[1] = (0x80 | ((code >> 6) & 0x3f));
        c[2] = (0x80 | (code & 0x3f));
        return 3;
    }
    // 对于其他编码点，转换为四字节 UTF-8 字符
    else {
        // 00wwwxx xxxxyyyy yyzzzzzz -> 11110www 10xxxxxx 10yyyyyy 10zzzzzz
        c[0] = (0xf0 | (code >> 18));
        c[1] = (0x80 | ((code >> 12) & 0x3f));
        c[2] = (0x80 | ((code >> 6) & 0x3f));
        c[3] = (0x80 | (code & 0x3f));
        return 4;
    }
}

/*******************************************************************************/
// 以下内容是 Bjoern Hoerhmann 的 DFA UTF-8 验证器的拷贝
// 许可证：MIT
// 版权所有 (c) 2008-2009 Bjoern Hoehrmann <bjoern@hoehrmann.de>
//
// 在遵守以下条件的情况下，特此免费授权给获取本软件及相关文档文件（以下简称“软件”）
// 的任何人，可以无限制地处理本软件，包括但不限于使用、复制、修改、合并、出版、
// 发行、再授权和/或销售本软件的副本，以及允许接收本软件的人这样做，前提是：
//
// 上述版权声明和本许可声明应包含在所有副本或实质性部分的本软件中。

// 本软件按“原样”提供，不提供任何形式的明示或暗示保证，包括但不限于对适销性、特定
// 目的的适用性和非侵权性的保证。在任何情况下，作者或版权持有人均不承担任何索赔、
// 损害赔偿或其他责任。
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// See http://bjoern.hoehrmann.de/utf-8/decoder/dfa/ for details.
//
// in principle could use something like simdutf to accelerate this

// 定义 UTF-8 解析器的状态常量
#define UTF8_ACCEPT 0
#define UTF8_REJECT 1

// UTF-8 解析器的状态转换表
static const uint8_t utf8d[] = {
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 00..1f
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 20..3f
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 40..5f
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 60..7f
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9, // 80..9f
  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7, // a0..bf
  8,8,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, // c0..df
  0xa,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x4,0x3,0x3, // e0..ef
  0xb,0x6,0x6,0x6,0x5,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8, // f0..ff
  0x0,0x1,0x2,0x3,0x5,0x8,0x7,0x1,0x1,0x1,0x4,0x6,0x1,0x1,0x1,0x1, // s0..s0
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,1, // s1..s2
  1,2,1,1,1,1,1,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1, // s3..s4
  1,2,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,3,1,1,1,1,1,1, // s5..s6
  1,3,1,1,1,1,1,3,1,3,1,1,1,1,1,1,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // s7..s8
};

// UTF-8 解码函数，根据当前状态和输入字节解析出一个 Unicode 编码点
static uint32_t inline
utf8_decode(uint32_t* state, uint32_t* codep, uint32_t byte) {
    uint32_t type = utf8d[byte];

    *codep = (*state != UTF8_ACCEPT) ?
            (byte & 0x3fu) | (*codep << 6) :
            (0xff >> type) & (byte);

    *state = utf8d[256 + *state*16 + type];
    return *state;
}

/*******************************************************************************/

// 计算存储 UTF-8 编码的 UTF-32 编码字符串所需的字节大小
// **s** 指向的 UTF-32 编码字符串的最大长度为 **max_bytes**。
NPY_NO_EXPORT Py_ssize_t
utf8_buffer_size(const uint8_t *s, size_t max_bytes)
{
    uint32_t codepoint;
    uint32_t state = 0;
    size_t num_bytes = 0;
    Py_ssize_t encoded_size_in_bytes = 0;

    // 忽略末尾的空字节
    while (max_bytes > 0 && s[max_bytes - 1] == 0) {
        max_bytes--;
    }

    if (max_bytes == 0) {
        return 0;
    }

    // 遍历输入的 UTF-32 编码字符串
    for (; num_bytes < max_bytes; ++s)
    {
        // 解析每个字节并更新解析状态和 Unicode 编码点
        utf8_decode(&state, &codepoint, *s);
        if (state == UTF8_REJECT)
        {
            return -1;  // 解析失败，返回错误码
        }
        else if(state == UTF8_ACCEPT)
        {
            // 当解析状态为接受状态时，计算该 Unicode 编码点的 UTF-8 编码字节数
            encoded_size_in_bytes += num_utf8_bytes_for_codepoint(codepoint);
        }
        num_bytes += 1;
    }

    // 如果最终状态不是接受状态，则表示输入不是一个有效的 UTF-8 字符串
    if (state != UTF8_ACCEPT) {
        return -1;
    }
    return encoded_size_in_bytes;  // 返回计算出的 UTF-8 编码字节数
}
// 计算以 UTF-8 编码的字节串中的代码点数量
num_codepoints_for_utf8_bytes(const unsigned char *s, size_t *num_codepoints, size_t max_bytes)
{
    uint32_t codepoint;  // 存储解码后的代码点
    uint32_t state = 0;  // UTF-8 解码状态
    size_t num_bytes = 0;  // 已处理的字节数
    *num_codepoints = 0;  // 初始化代码点数量为0

    // 忽略末尾的空字节
    while (max_bytes > 0 && s[max_bytes - 1] == 0) {
        max_bytes--;
    }

    if (max_bytes == 0) {
        return UTF8_ACCEPT;  // 如果最大字节数为0，直接返回 UTF8_ACCEPT
    }

    // 遍历输入的字节序列
    for (; num_bytes < max_bytes; ++s)
    {
        utf8_decode(&state, &codepoint, *s);  // 解码一个 UTF-8 字符
        if (state == UTF8_REJECT)
        {
            return state;  // 如果解码失败，返回 UTF8_REJECT
        }
        else if(state == UTF8_ACCEPT)
        {
            *num_codepoints += 1;  // 如果成功解码一个代码点，增加代码点计数
        }
        num_bytes += 1;  // 增加已处理的字节数
    }

    return state != UTF8_ACCEPT;  // 返回是否最终处于接受状态
}

// 查找指定起始和结束索引对应的位置
NPY_NO_EXPORT void
find_start_end_locs(char* buf, size_t buffer_size, npy_int64 start_index, npy_int64 end_index,
                    char **start_loc, char **end_loc) {
    size_t bytes_consumed = 0;  // 已消耗的字节数
    size_t num_codepoints = 0;  // 当前已处理的代码点数量
    if (num_codepoints == (size_t) start_index) {
        *start_loc = buf;  // 如果当前代码点数量等于起始索引，设置起始位置
    }
    if (num_codepoints == (size_t) end_index) {
        *end_loc = buf;  // 如果当前代码点数量等于结束索引，设置结束位置
    }
    // 遍历缓冲区直到达到结束索引或超过缓冲区大小
    while (bytes_consumed < buffer_size && num_codepoints < (size_t) end_index) {
        size_t num_bytes = num_bytes_for_utf8_character((const unsigned char*)buf);  // 获取当前 UTF-8 字符所占字节数
        num_codepoints += 1;  // 增加已处理的代码点数量
        bytes_consumed += num_bytes;  // 增加已消耗的字节数
        buf += num_bytes;  // 移动缓冲区指针到下一个字符的起始位置
        if (num_codepoints == (size_t) start_index) {
            *start_loc = buf;  // 如果当前代码点数量等于起始索引，设置起始位置
        }
        if (num_codepoints == (size_t) end_index) {
            *end_loc = buf;  // 如果当前代码点数量等于结束索引，设置结束位置
        }
    }
    assert(start_loc != NULL);  // 断言起始位置非空
    assert(end_loc != NULL);  // 断言结束位置非空
}

// 计算从给定起始位置到搜索字节偏移量处的 UTF-8 字符索引
NPY_NO_EXPORT size_t
utf8_character_index(
        const char* start_loc, size_t start_byte_offset, size_t start_index,
        size_t search_byte_offset, size_t buffer_size)
{
    size_t bytes_consumed = 0;  // 已消耗的字节数
    size_t cur_index = start_index;  // 当前代码点索引，初始为起始索引
    while (bytes_consumed < buffer_size && bytes_consumed < search_byte_offset) {
        size_t num_bytes = num_bytes_for_utf8_character((const unsigned char*)start_loc);  // 获取当前 UTF-8 字符所占字节数
        cur_index += 1;  // 增加当前代码点索引
        bytes_consumed += num_bytes;  // 增加已消耗的字节数
        start_loc += num_bytes;  // 移动起始位置到下一个字符的起始位置
    }
    return cur_index - start_index;  // 返回起始索引到搜索偏移量处的代码点数量
}
```