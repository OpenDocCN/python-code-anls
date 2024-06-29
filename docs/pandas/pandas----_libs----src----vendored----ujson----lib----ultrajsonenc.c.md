# `D:\src\scipysrc\pandas\pandas\_libs\src\vendored\ujson\lib\ultrajsonenc.c`

```
/*
Copyright (c) 2011-2013, ESN Social Software AB and Jonas Tarnstrom
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the ESN Social Software AB nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL ESN SOCIAL SOFTWARE AB OR JONAS TARNSTROM BE
LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Portions of code from MODP_ASCII - Ascii transformations (upper/lower, etc)
https://github.com/client9/stringencoders
Copyright (c) 2007  Nick Galbreath -- nickg [at] modp [dot] com. All rights
reserved.

Numeric decoder derived from TCL library
https://www.opensource.apple.com/source/tcl/tcl-14/tcl/license.terms
 * Copyright (c) 1988-1993 The Regents of the University of California.
 * Copyright (c) 1994 Sun Microsystems, Inc.
*/

// Licence at LICENSES/ULTRAJSON_LICENSE

#include "pandas/portable.h"
#include "pandas/vendored/ujson/lib/ultrajson.h"
#include <locale.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

/*
Worst cases being:

Control characters (ASCII < 32)
0x00 (1 byte) input => \u0000 output (6 bytes)
1 * 6 => 6 (6 bytes required)

or UTF-16 surrogate pairs
4 bytes input in UTF-8 => \uXXXX\uYYYY (12 bytes).

4 * 6 => 24 bytes (12 bytes required)

The extra 2 bytes are for the quotes around the string

*/
#define RESERVE_STRING(_len) (2 + ((_len) * 6))


注释：

/*
版权声明部分，声明对代码的版权和许可条件。

引入必要的头文件和定义：
- pandas/portable.h：Pandas项目的一个头文件
- pandas/vendored/ujson/lib/ultrajson.h：UltraJSON库的头文件
- locale.h：处理本地化相关的函数声明
- math.h：数学函数声明
- stdint.h：定义了各种整数类型
- stdio.h：标准输入输出函数声明
- stdlib.h：标准库函数声明
- string.h：字符串处理函数声明

定义TRUE和FALSE宏，如果未定义的话：

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

宏定义RESERVE_STRING(_len)，用于计算字符串的预留空间，考虑了最坏情况下的字符转换情况：
- 控制字符 (ASCII < 32)，1字节输入转换成\u0000输出需要6字节。
- UTF-16代理对，4字节UTF-8输入需要12字节输出。

计算公式为：
#define RESERVE_STRING(_len) (2 + ((_len) * 6))
其中_len为输入字符串的长度。
*/
static const double g_pow10[] = {1,            // 数组，存储10的幂的值，用于快速计算10的幂
                                 10,           // 10^1
                                 100,          // 10^2
                                 1000,         // 10^3
                                 10000,        // 10^4
                                 100000,       // 10^5
                                 1000000,      // 10^6
                                 10000000,     // 10^7
                                 100000000,    // 10^8
                                 1000000000,   // 10^9
                                 10000000000,  // 10^10
                                 100000000000, // 10^11
                                 1000000000000,// 10^12
                                 10000000000000,// 10^13
                                 100000000000000,// 10^14
                                 1000000000000000}; // 10^15
static const char g_hexChars[] = "0123456789abcdef"; // 存储十六进制字符的数组，用于十六进制转换
static const char g_escapeChars[] = "0123456789\\b\\t\\n\\f\\r\\\"\\\\\\/"; // 存储转义字符的数组，用于JSON转义处理

/*
FIXME: 虽然当前工作正常，但这些是魔法值，可能只有作者能够理解。
需要进行整理和更多文档说明
*/

/*
用于将所有ASCII字符输出为\uXXXX的表格，转义所有大于127的字符
*/
static const JSUINT8 g_asciiOutputTable[256] = {
    /* 0x00 */ 0,
    30,
    30,
    30,
    30,
    30,
    30,
    30,
    10,
    12,
    14,
    30,
    16,
    18,
    30,
    30,
    /* 0x10 */ 30,
    30,
    30,
    30,
    30,
    30,
    30,
    30,
    30,
    30,
    30,
    30,
    30,
    30,
    30,
    30,
    /* 0x20 */ 1,
    1,
    20,
    1,
    1,
    1,
    29,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    24,
    /* 0x30 */ 1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    29,
    1,
    29,
    1,
    /* 0x40 */ 1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    /* 0x50 */ 1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    22,
    1,
    1,
    1,
    /* 0x60 */ 1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    /* 0x70 */ 1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    /* 0x80 */ 1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    /* 0x90 */ 1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    /* 0xa0 */ 1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    /* 0xb0 */ 1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    /* 0xc0 */ 2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    /* 0xd0 */ 2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    /* 0xe0 */ 3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    /* 0xf0 */ 3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3};
    /* 下面是一个整数数组，包含了16个整数元素 */
    {
        4,  // 第一个元素为4
        4,  // 第二个元素为4
        4,  // 第三个元素为4
        4,  // 第四个元素为4
        4,  // 第五个元素为4
        4,  // 第六个元素为4
        4,  // 第七个元素为4
        4,  // 第八个元素为4
        5,  // 第九个元素为5
        5,  // 第十个元素为5
        5,  // 第十一个元素为5
        5,  // 第十二个元素为5
        6,  // 第十三个元素为6
        6,  // 第十四个元素为6
        1,  // 第十五个元素为1
        1   // 最后一个元素为1
    };
/*
FIXME: 跨多次编码器调用时跟踪其大小，并尝试进行估算
这样我们每次调用时就不会遇到内存不足的问题
*/
static void SetError(JSOBJ obj, JSONObjectEncoder *enc, const char *message) {
  // 将错误消息设置为给定的消息字符串
  enc->errorMsg = message;
  // 设置发生错误的对象
  enc->errorObj = obj;
}

/*
FIXME: 跨多次编码器调用时跟踪其大小，并尝试进行估算
这样我们每次调用时就不会遇到内存不足的问题
*/
void Buffer_Realloc(JSONObjectEncoder *enc, size_t cbNeeded) {
  // 获取当前缓冲区大小
  size_t curSize = enc->end - enc->start;
  // 初始新的缓冲区大小为当前大小的两倍
  size_t newSize = curSize * 2;
  // 计算当前偏移量
  size_t offset = enc->offset - enc->start;

  // 如果新的缓冲区大小小于当前大小加上所需大小，则进行倍增直到满足条件
  while (newSize < curSize + cbNeeded) {
    newSize *= 2;
  }

  // 如果已经使用堆内存
  if (enc->heap) {
    // 重新分配堆内存，新大小为 newSize 字节
    enc->start = (char *)enc->realloc(enc->start, newSize);
    // 如果重新分配失败，则设置错误并返回
    if (!enc->start) {
      SetError(NULL, enc, "Could not reserve memory block");
      return;
    }
  } else {
    // 否则，保存旧的起始地址
    char *oldStart = enc->start;
    // 设置堆内存标志
    enc->heap = 1;
    // 分配新的堆内存，新大小为 newSize 字节
    enc->start = (char *)enc->malloc(newSize);
    // 如果分配失败，则设置错误并返回
    if (!enc->start) {
      SetError(NULL, enc, "Could not reserve memory block");
      return;
    }
    // 将旧数据复制到新分配的内存中
    memcpy(enc->start, oldStart, offset);
  }
  // 更新偏移量为新的起始地址加上之前的偏移
  enc->offset = enc->start + offset;
  // 更新缓冲区的末尾位置为新的起始地址加上新的大小
  enc->end = enc->start + newSize;
}

INLINE_PREFIX void FASTCALL_MSVC
Buffer_AppendShortHexUnchecked(char *outputOffset, unsigned short value) {
  // 将无效的短十六进制值附加到输出偏移位置
  *(outputOffset++) = g_hexChars[(value & 0xf000) >> 12];
  *(outputOffset++) = g_hexChars[(value & 0x0f00) >> 8];
  *(outputOffset++) = g_hexChars[(value & 0x00f0) >> 4];
  *(outputOffset++) = g_hexChars[(value & 0x000f) >> 0];
}

int Buffer_EscapeStringUnvalidated(JSONObjectEncoder *enc, const char *io,
                                   const char *end) {
  // 将编码器的偏移地址转换为字符指针
  char *of = (char *)enc->offset;

  // 循环处理输入字符流
  for (;;) {
    switch (*io) {
    case 0x00: {
      // 如果是空字符
      if (io < end) {
        // 转义为 Unicode 码点 \u0000
        *(of++) = '\\';
        *(of++) = 'u';
        *(of++) = '0';
        *(of++) = '0';
        *(of++) = '0';
        *(of++) = '0';
        break;
      } else {
        // 更新偏移量并返回真值表示结束
        enc->offset += (of - enc->offset);
        return TRUE;
      }
    }
    case '\"':
      // 转义双引号
      (*of++) = '\\';
      (*of++) = '\"';
      break;
    case '\\':
      // 转义反斜线
      (*of++) = '\\';
      (*of++) = '\\';
      break;
    case '/':
      // 转义斜线
      (*of++) = '\\';
      (*of++) = '/';
      break;
    case '\b':
      // 转义退格
      (*of++) = '\\';
      (*of++) = 'b';
      break;
    case '\f':
      // 转义换页符
      (*of++) = '\\';
      (*of++) = 'f';
      break;
    case '\n':
      // 转义换行符
      (*of++) = '\\';
      (*of++) = 'n';
      break;
    case '\r':
      // 转义回车符
      (*of++) = '\\';
      (*of++) = 'r';
      break;
    case '\t':
      // 转义制表符
      (*of++) = '\\';
      (*of++) = 't';
      break;
    case 0x26: // '/'
    case 0x3c: // '<'
    case 0x3e: // '>'
      // 如果需要转义 HTML 特殊字符
      if (enc->encodeHTMLChars) {
        // 通过 \u00XX 转义
        PD_FALLTHROUGH;
      } else {
        // 否则不转义，直接输出原字符
        (*of++) = (*io);
        break;
      }
    // 以下情况都输出原字符
    case 0x01: case 0x02: case 0x03: case 0x04: case 0x05:
    case 0x06: case 0x07: case 0x0b: case 0x0e: case 0x0f:
    case 0x10: case 0x11: case 0x12: case 0x13: case 0x14:
    case 0x15: case 0x16: case 0x17:
      (*of++) = (*io);
      break;
    }
    // 指针后移，处理下一个字符
    io++;
  }
}
    case 0x18:
    case 0x19:
    case 0x1a:
    case 0x1b:
    case 0x1c:
    case 0x1d:
    case 0x1e:
    case 0x1f: {
      // 如果当前字符为Unicode控制字符（0x18到0x1f之间），则转义为\u00xx格式
      *(of++) = '\\';  // 在输出流中添加反斜杠 '\'
      *(of++) = 'u';   // 添加 'u' 表示Unicode转义序列的开始
      *(of++) = '0';   // 添加 '0'，这是Unicode转义序列的固定部分
      *(of++) = '0';   // 添加 '0'，这是Unicode转义序列的固定部分
      // 将当前字符的高4位转换为十六进制字符添加到输出流中
      *(of++) = g_hexChars[(unsigned char)(((*io) & 0xf0) >> 4)];
      // 将当前字符的低4位转换为十六进制字符添加到输出流中
      *(of++) = g_hexChars[(unsigned char)((*io) & 0x0f)];
      break;  // 转义完成，跳出当前情况的处理
    }
    default:
      // 对于非Unicode控制字符，直接将当前字符添加到输出流中
      (*of++) = (*io);
      break;
    }
    io++;  // 处理下一个输入字符
}

int Buffer_EscapeStringValidated(JSOBJ obj, JSONObjectEncoder *enc,
                                 const char *io, const char *end) {
  // 定义变量：字符流偏移位置
  char *of = (char *)enc->offset;

  // 进入无限循环，处理字符流直到结束
  for (;;) {
    // 获取当前字符的 UTF-8 编码长度
    JSUINT8 utflen = g_asciiOutputTable[(unsigned char)*io];

    // 根据 UTF-8 编码长度进行不同情况处理
    switch (utflen) {
    // 如果是 0 表示无效的 UTF-8 字符
    case 0: {
      if (io < end) {
        // 添加转义字符 "\u0000" 到输出流
        *(of++) = '\\';
        *(of++) = 'u';
        *(of++) = '0';
        *(of++) = '0';
        *(of++) = '0';
        *(of++) = '0';
        io++;
        continue;
      } else {
        // 更新字符流偏移位置，并返回成功
        enc->offset += (of - enc->offset);
        return TRUE;
      }
    }

    // 如果是 1 表示单字节 UTF-8 字符
    case 1: {
      // 直接复制字符到输出流
      *(of++) = (*io++);
      continue;
    }

    // 如果是 2 表示双字节 UTF-8 字符
    case 2: {
      JSUTF32 in;
      JSUTF16 in16;

      // 检查是否有足够的字节来完整读取 UTF-8 序列
      if (end - io < 1) {
        enc->offset += (of - enc->offset);
        // 设置错误信息并返回失败
        SetError(obj, enc, "Unterminated UTF-8 sequence when encoding string");
        return FALSE;
      }

      // 复制两个字节作为 UTF-16 编码
      memcpy(&in16, io, sizeof(JSUTF16));
      in = (JSUTF32)in16;

      // 根据系统端序处理 UTF-16 转 UTF-32
#ifdef __LITTLE_ENDIAN__
      ucs = ((in & 0x1f) << 6) | ((in >> 8) & 0x3f);
#else
      ucs = ((in & 0x1f00) >> 2) | (in & 0x3f);
#endif

      // 检查是否出现了过长的 UTF-8 序列
      if (ucs < 0x80) {
        enc->offset += (of - enc->offset);
        // 设置错误信息并返回失败
        SetError(obj, enc,
                 "Overlong 2 byte UTF-8 sequence detected when "
                 "encoding string");
        return FALSE;
      }

      io += 2;
      break;
    }

    // 如果是 3 表示三字节 UTF-8 字符
    case 3: {
      JSUTF32 in;
      JSUTF16 in16;
      JSUINT8 in8;

      // 检查是否有足够的字节来完整读取 UTF-8 序列
      if (end - io < 2) {
        enc->offset += (of - enc->offset);
        // 设置错误信息并返回失败
        SetError(obj, enc, "Unterminated UTF-8 sequence when encoding string");
        return FALSE;
      }

      // 复制两个字节作为 UTF-16 编码，一个字节作为低位字节
      memcpy(&in16, io, sizeof(JSUTF16));
      memcpy(&in8, io + 2, sizeof(JSUINT8));

      // 根据系统端序处理 UTF-16 转 UTF-32
#ifdef __LITTLE_ENDIAN__
      in = (JSUTF32)in16;
      in |= in8 << 16;
      ucs =
          ((in & 0x0f) << 12) | ((in & 0x3f00) >> 2) | ((in & 0x3f0000) >> 16);
#else
      in = in16 << 8;
      in |= in8;
      ucs = ((in & 0x0f0000) >> 4) | ((in & 0x3f00) >> 2) | (in & 0x3f);
#endif

      // 检查是否出现了过长的 UTF-8 序列
      if (ucs < 0x800) {
        enc->offset += (of - enc->offset);
        // 设置错误信息并返回失败
        SetError(obj, enc,
                 "Overlong 3 byte UTF-8 sequence detected when "
                 "encoding string");
        return FALSE;
      }

      io += 3;
      break;
    }
    // 如果是 4 表示四字节 UTF-8 字符
    case 4: {
      JSUTF32 in;

      // 检查是否有足够的字节来完整读取 UTF-8 序列
      if (end - io < 3) {
        enc->offset += (of - enc->offset);
        // 设置错误信息并返回失败
        SetError(obj, enc, "Unterminated UTF-8 sequence when encoding string");
        return FALSE;
      }

      // 复制四个字节作为 UTF-32 编码
      memcpy(&in, io, sizeof(JSUTF32));

      // 根据系统端序处理 UTF-32
#ifdef __LITTLE_ENDIAN__
      ucs = ((in & 0x07) << 18) | ((in & 0x3f00) << 4) |
            ((in & 0x3f0000) >> 10) | ((in & 0x3f000000) >> 24);
#else
      ucs = ((in & 0x07000000) >> 6) | ((in & 0x3f0000) >> 4) |
            ((in & 0x3f00) >> 2) | (in & 0x3f);
#endif

      io += 4;
      break;
    }
#endif
      // 如果 UCS 小于 0x10000
      if (ucs < 0x10000) {
        // 更新偏移量并设置错误信息，指示检测到的超长 4 字节 UTF-8 序列
        enc->offset += (of - enc->offset);
        SetError(obj, enc,
                 "Overlong 4 byte UTF-8 sequence detected when "
                 "encoding string");
        return FALSE;
      }

      // 增加输入输出指针 4 个字节的长度
      io += 4;
      break;
    }

    case 5:
    case 6: {
      // 更新偏移量并设置错误信息，指示不支持的 UTF-8 序列长度
      enc->offset += (of - enc->offset);
      SetError(obj, enc,
               "Unsupported UTF-8 sequence length when encoding string");
      return FALSE;
    }

    case 29: {
      // 如果 encodeHTMLChars 开启，跳转到下面的 case 30 处理
      if (enc->encodeHTMLChars) {
        PD_FALLTHROUGH;  // 转到 \u00XX case 30 处
      } else {
        // 与 case 1 相同的处理
        *(of++) = (*io++);
        continue;
      }
    }

    case 30: {
      // 对 \uXXXX 进行编码
      *(of++) = '\\';
      *(of++) = 'u';
      *(of++) = '0';
      *(of++) = '0';
      *(of++) = g_hexChars[(unsigned char)(((*io) & 0xf0) >> 4)];
      *(of++) = g_hexChars[(unsigned char)((*io) & 0x0f)];
      io++;
      continue;
    }
    case 10:
    case 12:
    case 14:
    case 16:
    case 18:
    case 20:
    case 22:
    case 24: {
      // 处理特定的转义字符
      *(of++) = *((char *)(g_escapeChars + utflen + 0));
      *(of++) = *((char *)(g_escapeChars + utflen + 1));
      io++;
      continue;
    }
    // 这种情况不会发生，只是为了兼容性而存在
    default: {
      ucs = 0;
      break;
    }
    }

    /*
    如果字符是长度大于 1 的 UTF8 序列，会进入这里 */
    if (ucs >= 0x10000) {
      ucs -= 0x10000;
      *(of++) = '\\';
      *(of++) = 'u';
      Buffer_AppendShortHexUnchecked(of, (unsigned short)(ucs >> 10) + 0xd800);
      of += 4;

      *(of++) = '\\';
      *(of++) = 'u';
      Buffer_AppendShortHexUnchecked(of,
                                     (unsigned short)(ucs & 0x3ff) + 0xdc00);
      of += 4;
    } else {
      *(of++) = '\\';
      *(of++) = 'u';
      Buffer_AppendShortHexUnchecked(of, (unsigned short)ucs);
      of += 4;
    }
  }
}

#define Buffer_Reserve(__enc, __len)                                           \
  // 如果剩余空间小于所需长度，重新分配缓冲区
  if ((size_t)((__enc)->end - (__enc)->offset) < (size_t)(__len)) {            \
    Buffer_Realloc((__enc), (__len));                                          \
  }

#define Buffer_AppendCharUnchecked(__enc, __chr) *((__enc)->offset++) = __chr;

INLINE_PREFIX void FASTCALL_MSVC strreverse(char *begin, char *end) {
  char aux;
  while (end > begin)
    aux = *end, *end-- = *begin, *begin++ = aux;
}

void Buffer_AppendIndentNewlineUnchecked(JSONObjectEncoder *enc) {
  // 如果缩进值大于 0，则追加一个换行符到缓冲区
  if (enc->indent > 0)
    Buffer_AppendCharUnchecked(enc, '\n');
}

// 此函数可以重构为只接受 enc 作为参数，但这是直接从 ujson 源码采用的方式
void Buffer_AppendIndentUnchecked(JSONObjectEncoder *enc, JSINT32 value) {
  int i;
  // 如果缩进值大于 0
  if (enc->indent > 0) {
    // 循环追加空格字符，以达到指定的缩进层数
    while (value-- > 0)
      for (i = 0; i < enc->indent; i++)
        Buffer_AppendCharUnchecked(enc, ' ');
  }
}
void Buffer_AppendIntUnchecked(JSONObjectEncoder *enc, JSINT32 value) {
  char *wstr;
  JSUINT32 uvalue = (value < 0) ? -value : value;
  wstr = enc->offset;

  // Conversion. Number is reversed.
  // 将整数转换为字符串，数字顺序颠倒
  do {
    *wstr++ = (char)(48 + (uvalue % 10));  // 取出最低位数字并转换为字符
  } while (uvalue /= 10);  // 移除已经处理的最低位

  if (value < 0)
    *wstr++ = '-';  // 如果原始数值为负数，添加负号

  // Reverse string
  strreverse(enc->offset, wstr - 1);  // 颠倒字符串顺序
  enc->offset += (wstr - (enc->offset));  // 更新偏移量
}

void Buffer_AppendLongUnchecked(JSONObjectEncoder *enc, JSINT64 value) {
  char *wstr;
  JSUINT64 uvalue;
  if (value == INT64_MIN) {
    uvalue = INT64_MAX + UINT64_C(1);
  } else {
    uvalue = (value < 0) ? -value : value;
  }

  wstr = enc->offset;
  // Conversion. Number is reversed.
  // 将长整数转换为字符串，数字顺序颠倒

  do {
    *wstr++ = (char)(48 + (uvalue % 10ULL));  // 取出最低位数字并转换为字符
  } while (uvalue /= 10ULL);  // 移除已经处理的最低位

  if (value < 0)
    *wstr++ = '-';  // 如果原始数值为负数，添加负号

  // Reverse string
  strreverse(enc->offset, wstr - 1);  // 颠倒字符串顺序
  enc->offset += (wstr - (enc->offset));  // 更新偏移量
}

int Buffer_AppendDoubleUnchecked(JSOBJ obj, JSONObjectEncoder *enc,
                                 double value) {
  /* if input is beyond the thresholds, revert to exponential */
  // 如果输入值超出阈值，采用指数形式
  const double thres_max = (double)1e16 - 1;
  const double thres_min = (double)1e-15;
  char precision_str[20];
  int count;
  double diff = 0.0;
  char *str = enc->offset;
  char *wstr = str;
  unsigned long long whole;
  double tmp;
  unsigned long long frac;
  int neg;
  double pow10;

  if (value == HUGE_VAL || value == -HUGE_VAL) {
    SetError(obj, enc, "Invalid Inf value when encoding double");
    return FALSE;
  }

  if (!(value == value)) {
    SetError(obj, enc, "Invalid Nan value when encoding double");
    return FALSE;
  }

  // we'll work in positive values and deal with the
  // negative sign issue later
  // 将负数转为正数进行处理，后续再处理负号问题
  neg = 0;
  if (value < 0) {
    neg = 1;
    value = -value;
  }

  /*
  for very large or small numbers switch back to native sprintf for
  exponentials.  anyone want to write code to replace this?
  */
  // 对于非常大或非常小的数值，采用指数形式
  if (value > thres_max || (value != 0.0 && fabs(value) < thres_min)) {
    precision_str[0] = '%';
    precision_str[1] = '.';
#if defined(_WIN32) && defined(_MSC_VER)
    sprintf_s(precision_str + 2, sizeof(precision_str) - 2, "%ug",
              enc->doublePrecision);
    enc->offset += sprintf_s(str, enc->end - enc->offset, precision_str,
                             neg ? -value : value);
#else
    snprintf(precision_str + 2, sizeof(precision_str) - 2, "%ug",
             enc->doublePrecision);
    enc->offset += snprintf(str, enc->end - enc->offset, precision_str,
                            neg ? -value : value);
#endif
    return TRUE;
  }

  pow10 = g_pow10[enc->doublePrecision];

  whole = (unsigned long long)value;
  tmp = (value - whole) * pow10;
  frac = (unsigned long long)(tmp);
  diff = tmp - frac;

  if (diff > 0.5) {
    ++frac;
  } else if (diff == 0.5 && ((frac == 0) || (frac & 1))) {
    /* if halfway, round up if odd, OR
    if last digit is 0.  That last part is strange */
    // 如果四舍五入的结果为一半时，如果是奇数就进位
    // 或者如果最后一位数字是0，这部分逻辑很奇怪
    ++frac;
  }

  // 处理溢出情况，例如：
  // 当 frac >= pow10 时，将 frac 置为 0，同时整数部分加一
  if (frac >= pow10) {
    frac = 0;
    ++whole;
  }

  // 如果 doublePrecision 为 0，进行如下处理
  if (enc->doublePrecision == 0) {
    diff = value - whole;

    // 如果小数部分 diff 大于 0.5，向上舍入，例如 1.6 -> 2
    if (diff > 0.5) {
      ++whole;
    } else if (diff == 0.5 && (whole & 1)) {
      // 如果 diff 等于 0.5 且整数部分为奇数，则向上舍入，例如 1.5 -> 2，但 2.5 -> 2
      ++whole;
    }

    // vvvvvvvvvvvvvvvvvvv  与 modp_dto2 的区别点
  } else if (frac) {
    count = enc->doublePrecision;

    // 处理小数部分，作为无符号数
    // 确保小数部分不为 0，同时可能存在前导零，需要移除
    while (!(frac % 10)) {
      --count;
      frac /= 10;
    }
    //^^^^^^^^^^^^^^^^^^^  与 modp_dto2 的区别点

    // 处理小数部分，作为无符号数
    do {
      --count;
      *wstr++ = (char)(48 + (frac % 10));
    } while (frac /= 10);
    // 添加额外的 0
    while (count-- > 0) {
      *wstr++ = '0';
    }
    // 添加小数点
    *wstr++ = '.';
  } else {
    *wstr++ = '0';
    *wstr++ = '.';
  }

  // 处理整数部分，包括符号处理，数字被反转
  do {
    *wstr++ = (char)(48 + (whole % 10));
  } while (whole /= 10);

  // 如果是负数，添加负号
  if (neg) {
    *wstr++ = '-';
  }
  // 反转字符串
  strreverse(str, wstr - 1);
  // 更新偏移量
  enc->offset += (wstr - (enc->offset));

  // 返回成功
  return TRUE;
}

/*
FIXME:
Handle integration functions returning NULL here */

/*
FIXME:
Perhaps implement recursion detection */

void encode(JSOBJ obj, JSONObjectEncoder *enc, const char *name,
            size_t cbName) {
  const char *value;
  char *objName;
  int count;
  JSOBJ iterObj;
  size_t szlen;
  JSONTypeContext tc;
  tc.encoder = enc;

  if (enc->level > enc->recursionMax) {
    // 如果递归层级超过设定的最大递归深度，设置错误信息并返回
    SetError(obj, enc, "Maximum recursion level reached");
    return;
  }

  /*
  This reservation must hold

  length of _name as encoded worst case +
  maxLength of double to string OR maxLength of JSLONG to string
  */
  
  // 为 JSON 字符串编码预留足够的缓冲区空间
  Buffer_Reserve(enc, 256 + RESERVE_STRING(cbName));
  if (enc->errorMsg) {
    return;
  }

  if (name) {
    // 如果有属性名，将其作为字符串加入到缓冲区中
    Buffer_AppendCharUnchecked(enc, '\"');

    if (enc->forceASCII) {
      // 如果强制 ASCII，对属性名进行转义并加入缓冲区
      if (!Buffer_EscapeStringValidated(obj, enc, name, name + cbName)) {
        return;
      }
    } else {
      // 否则，不验证转义的情况下直接加入缓冲区
      if (!Buffer_EscapeStringUnvalidated(enc, name, name + cbName)) {
        return;
      }
    }

    Buffer_AppendCharUnchecked(enc, '\"');

    Buffer_AppendCharUnchecked(enc, ':');
#ifndef JSON_NO_EXTRA_WHITESPACE
    Buffer_AppendCharUnchecked(enc, ' ');
#endif
  }

  // 初始化类型上下文
  enc->beginTypeContext(obj, &tc);

  switch (tc.type) {
  case JT_INVALID: {
    return;
  }

  case JT_ARRAY: {
    // 处理数组类型
    count = 0;
    enc->iterBegin(obj, &tc);

    Buffer_AppendCharUnchecked(enc, '[');
    Buffer_AppendIndentNewlineUnchecked(enc);

    while (enc->iterNext(obj, &tc)) {
      if (count > 0) {
        Buffer_AppendCharUnchecked(enc, ',');
#ifndef JSON_NO_EXTRA_WHITESPACE
        Buffer_AppendCharUnchecked(enc, ' ');
#endif
        Buffer_AppendIndentNewlineUnchecked(enc);
      }

      // 获取数组元素并进行递归编码
      iterObj = enc->iterGetValue(obj, &tc);

      enc->level++;
      Buffer_AppendIndentUnchecked(enc, enc->level);
      encode(iterObj, enc, NULL, 0);
      count++;
    }

    enc->iterEnd(obj, &tc);
    Buffer_AppendIndentNewlineUnchecked(enc);
    Buffer_AppendIndentUnchecked(enc, enc->level);
    Buffer_AppendCharUnchecked(enc, ']');
    break;
  }

  case JT_OBJECT: {
    // 处理对象类型
    count = 0;
    enc->iterBegin(obj, &tc);

    Buffer_AppendCharUnchecked(enc, '{');
    Buffer_AppendIndentNewlineUnchecked(enc);

    while (enc->iterNext(obj, &tc)) {
      if (count > 0) {
        Buffer_AppendCharUnchecked(enc, ',');
#ifndef JSON_NO_EXTRA_WHITESPACE
        Buffer_AppendCharUnchecked(enc, ' ');
#endif
        Buffer_AppendIndentNewlineUnchecked(enc);
      }

      // 获取对象属性名和属性值，进行递归编码
      iterObj = enc->iterGetValue(obj, &tc);
      objName = enc->iterGetName(obj, &tc, &szlen);

      enc->level++;
      Buffer_AppendIndentUnchecked(enc, enc->level);
      encode(iterObj, enc, objName, szlen);
      count++;
    }

    enc->iterEnd(obj, &tc);
    Buffer_AppendIndentNewlineUnchecked(enc);
    Buffer_AppendIndentUnchecked(enc, enc->level);
    Buffer_AppendCharUnchecked(enc, '}');
    break;
  }

  case JT_LONG: {
    // 处理长整型数据
    Buffer_AppendLongUnchecked(enc, enc->getLongValue(obj, &tc));
    break;
  }

  case JT_INT: {
    // 处理整型数据
    Buffer_AppendIntUnchecked(enc, enc->getIntValue(obj, &tc));
    break;
  }
  }
}
    Buffer_AppendIntUnchecked(enc, enc->getIntValue(obj, &tc));
    # 将整数值转换为字符串并追加到缓冲区中
    break;

  }

  case JT_TRUE: {
    # 向缓冲区依次追加字符 't', 'r', 'u', 'e'
    Buffer_AppendCharUnchecked(enc, 't');
    Buffer_AppendCharUnchecked(enc, 'r');
    Buffer_AppendCharUnchecked(enc, 'u');
    Buffer_AppendCharUnchecked(enc, 'e');
    break;
  }

  case JT_FALSE: {
    # 向缓冲区依次追加字符 'f', 'a', 'l', 's', 'e'
    Buffer_AppendCharUnchecked(enc, 'f');
    Buffer_AppendCharUnchecked(enc, 'a');
    Buffer_AppendCharUnchecked(enc, 'l');
    Buffer_AppendCharUnchecked(enc, 's');
    Buffer_AppendCharUnchecked(enc, 'e');
    break;
  }

  case JT_NULL: {
    # 向缓冲区依次追加字符 'n', 'u', 'l', 'l'
    Buffer_AppendCharUnchecked(enc, 'n');
    Buffer_AppendCharUnchecked(enc, 'u');
    Buffer_AppendCharUnchecked(enc, 'l');
    Buffer_AppendCharUnchecked(enc, 'l');
    break;
  }

  case JT_DOUBLE: {
    # 获取浮点数值并将其转换为字符串追加到缓冲区中
    if (!Buffer_AppendDoubleUnchecked(obj, enc,
                                      enc->getDoubleValue(obj, &tc))) {
      enc->endTypeContext(obj, &tc);
      enc->level--;
      return;
    }
    break;
  }

  case JT_UTF8: {
    # 获取 UTF-8 字符串值并将其转义后追加到缓冲区中
    value = enc->getStringValue(obj, &tc, &szlen);
    if (enc->errorMsg) {
      enc->endTypeContext(obj, &tc);
      return;
    }
    Buffer_Reserve(enc, RESERVE_STRING(szlen));
    Buffer_AppendCharUnchecked(enc, '\"');

    if (enc->forceASCII) {
      # 如果强制 ASCII 输出，验证并转义字符串追加到缓冲区中
      if (!Buffer_EscapeStringValidated(obj, enc, value, value + szlen)) {
        enc->endTypeContext(obj, &tc);
        enc->level--;
        return;
      }
    } else {
      # 否则，不验证直接转义字符串追加到缓冲区中
      if (!Buffer_EscapeStringUnvalidated(enc, value, value + szlen)) {
        enc->endTypeContext(obj, &tc);
        enc->level--;
        return;
      }
    }

    Buffer_AppendCharUnchecked(enc, '\"');
    break;
  }

  case JT_BIGNUM: {
    # 获取大整数字符串值并追加到缓冲区中
    value = enc->getBigNumStringValue(obj, &tc, &szlen);

    Buffer_Reserve(enc, RESERVE_STRING(szlen));
    if (enc->errorMsg) {
      enc->endTypeContext(obj, &tc);
      return;
    }

    if (enc->forceASCII) {
      # 如果强制 ASCII 输出，验证并转义字符串追加到缓冲区中
      if (!Buffer_EscapeStringValidated(obj, enc, value, value + szlen)) {
        enc->endTypeContext(obj, &tc);
        enc->level--;
        return;
      }
    } else {
      # 否则，不验证直接转义字符串追加到缓冲区中
      if (!Buffer_EscapeStringUnvalidated(enc, value, value + szlen)) {
        enc->endTypeContext(obj, &tc);
        enc->level--;
        return;
      }
    }

    break;
  }
  }

  # 结束当前对象的类型上下文
  enc->endTypeContext(obj, &tc);
  # 减少嵌套层级
  enc->level--;
}

// 编码 JSON 对象为字符串
char *JSON_EncodeObject(JSOBJ obj, JSONObjectEncoder *enc, char *_buffer,
                        size_t _cbBuffer) {
  // 为未指定的内存分配函数设置默认值
  enc->malloc = enc->malloc ? enc->malloc : malloc;
  enc->free = enc->free ? enc->free : free;
  enc->realloc = enc->realloc ? enc->realloc : realloc;
  enc->errorMsg = NULL;  // 清空错误消息
  enc->errorObj = NULL;  // 清空错误对象
  enc->level = 0;  // 设置初始递归深度为 0

  // 如果递归最大深度小于 1，则设置为默认深度
  if (enc->recursionMax < 1) {
    enc->recursionMax = JSON_MAX_RECURSION_DEPTH;
  }

  // 如果双精度小数的精度设置不在合理范围内，则设置为默认值
  if (enc->doublePrecision < 0 ||
      enc->doublePrecision > JSON_DOUBLE_MAX_DECIMALS) {
    enc->doublePrecision = JSON_DOUBLE_MAX_DECIMALS;
  }

  // 如果未提供缓冲区，则分配默认大小的缓冲区
  if (_buffer == NULL) {
    _cbBuffer = 32768;
    enc->start = (char *)enc->malloc(_cbBuffer);
    if (!enc->start) {
      // 分配失败，设置错误信息并返回 NULL
      SetError(obj, enc, "Could not reserve memory block");
      return NULL;
    }
    enc->heap = 1;  // 标记为堆上分配的内存
  } else {
    enc->start = _buffer;  // 使用提供的缓冲区
    enc->heap = 0;  // 标记为非堆上分配的内存
  }

  // 设置结束位置和偏移量
  enc->end = enc->start + _cbBuffer;
  enc->offset = enc->start;

  // 获取当前的数字格式化地区设置
  char *locale = setlocale(LC_NUMERIC, NULL);
  if (!locale) {
    // 获取地区设置失败，设置错误信息并返回 NULL
    SetError(NULL, enc, "setlocale call failed");
    return NULL;
  }

  // 如果当前地区设置不是 "C"，则暂时将其设置为 "C" 进行编码
  if (strcmp(locale, "C")) {
    size_t len = strlen(locale) + 1;
    char *saved_locale = malloc(len);
    if (saved_locale == NULL) {
      // 分配失败，设置错误信息并返回 NULL
      SetError(NULL, enc, "Could not reserve memory block");
      return NULL;
    }
    memcpy(saved_locale, locale, len);
    setlocale(LC_NUMERIC, "C");
    encode(obj, enc, NULL, 0);  // 执行编码操作
    setlocale(LC_NUMERIC, saved_locale);  // 恢复之前的地区设置
    free(saved_locale);  // 释放保存地区设置的内存
  } else {
    encode(obj, enc, NULL, 0);  // 执行编码操作
  }

  Buffer_Reserve(enc, 1);  // 确保缓冲区能够容纳额外的一个字符
  if (enc->errorMsg) {
    return NULL;  // 如果出现错误，返回 NULL
  }
  Buffer_AppendCharUnchecked(enc, '\0');  // 在字符串末尾添加 NULL 终止符

  return enc->start;  // 返回编码后的字符串起始位置
}
```