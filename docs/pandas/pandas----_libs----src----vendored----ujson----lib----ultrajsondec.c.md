# `D:\src\scipysrc\pandas\pandas\_libs\src\vendored\ujson\lib\ultrajsondec.c`

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

// 包含 ultrajson 库的头文件
#include "pandas/vendored/ujson/lib/ultrajson.h"
// 包含系统标准库头文件
#include <errno.h>
#include <limits.h>
#include <locale.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

// 如果 TRUE 和 FALSE 未定义，则定义为 1 和 0
#ifndef TRUE
#define TRUE 1
#define FALSE 0
#endif
// 如果 NULL 未定义，则定义为 0
#ifndef NULL
#define NULL 0
#endif

// 定义解析状态结构体
struct DecoderState {
  char *start;         // 解析起始位置
  char *end;           // 解析结束位置
  wchar_t *escStart;   // 转义字符起始位置
  wchar_t *escEnd;     // 转义字符结束位置
  int escHeap;         // 转义字符堆栈标志
  int lastType;        // 上一个解析类型
  JSUINT32 objDepth;   // 对象深度
  void *prv;           // 私有数据指针
  JSONObjectDecoder *dec; // JSON 对象解析器指针
};

// 定义快速调用 MSVC 下的解析函数原型
JSOBJ FASTCALL_MSVC decode_any(struct DecoderState *ds);
typedef JSOBJ (*PFN_DECODER)(struct DecoderState *ds);

// 设置解析错误信息并返回 NULL
static JSOBJ SetError(struct DecoderState *ds, int offset,
                      const char *message) {
  ds->dec->errorOffset = ds->start + offset; // 记录错误偏移位置
  ds->dec->errorStr = (char *)message;       // 设置错误信息字符串
  return NULL;                               // 返回 NULL 表示错误
}
double createDouble(double intNeg, double intValue, double frcValue,
                    int frcDecimalCount) {
  // 定义用于计算浮点数的10的负幂次方的数组
  static const double g_pow10[] = {1.0,
                                   0.1,
                                   0.01,
                                   0.001,
                                   0.0001,
                                   0.00001,
                                   0.000001,
                                   0.0000001,
                                   0.00000001,
                                   0.000000001,
                                   0.0000000001,
                                   0.00000000001,
                                   0.000000000001,
                                   0.0000000000001,
                                   0.00000000000001,
                                   0.000000000000001};
  // 根据整数部分、小数部分和小数点位数，计算并返回浮点数
  return (intValue + (frcValue * g_pow10[frcDecimalCount])) * intNeg;
}

JSOBJ FASTCALL_MSVC decodePreciseFloat(struct DecoderState *ds) {
  char *end;
  double value;
  errno = 0;

  // 将字符串形式的数字转换为双精度浮点数
  value = strtod(ds->start, &end);

  // 如果转换时发生范围错误，返回错误信息
  if (errno == ERANGE) {
    return SetError(ds, -1, "Range error when decoding numeric as double");
  }

  ds->start = end;
  return ds->dec->newDouble(ds->prv, value);
}

JSOBJ FASTCALL_MSVC decode_numeric(struct DecoderState *ds) {
  int intNeg = 1;
  JSUINT64 intValue;
  JSUINT64 prevIntValue;
  int chr;
  int decimalCount = 0;
  double frcValue = 0.0;
  double expNeg;
  double expValue;
  char *offset = ds->start;

  JSUINT64 overflowLimit = LLONG_MAX;

  // 如果起始字符为'I'，跳转到解析无穷大部分
  if (*(offset) == 'I') {
    goto DECODE_INF;
  } else if (*(offset) == 'N') {
    // 如果起始字符为'N'，跳转到解析非数值部分
    goto DECODE_NAN;
  } else if (*(offset) == '-') {
    // 如果起始字符为负号，跳过负号并设置负数标志
    offset++;
    intNeg = -1;
    overflowLimit = LLONG_MIN;
    if (*(offset) == 'I') {
      // 如果负号后直接跟'I'，跳转到解析无穷大部分
      goto DECODE_INF;
    }
  }

  // 扫描整数部分
  intValue = 0;

  while (1) {
    chr = (int)(unsigned char)*(offset);

    switch (chr) {
    // 处理数字字符，计算整数部分的值
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9': {
      prevIntValue = intValue;
      intValue = intValue * 10ULL + (JSLONG)(chr - 48);

      // 检查整数部分是否溢出
      if (intNeg == 1 && prevIntValue > intValue) {
        return SetError(ds, -1, "Value is too big!");
      } else if (intNeg == -1 && intValue > overflowLimit) {
        return SetError(ds, -1,
                        overflowLimit == LLONG_MAX ? "Value is too big!"
                                                   : "Value is too small");
      }

      offset++;
      break;
    }
    // 遇到小数点，跳转到解析小数部分
    case '.': {
      offset++;
      goto DECODE_FRACTION;
      break;
    }
    // 遇到指数符号，跳转到解析指数部分
    case 'e':
    case 'E': {
      offset++;
      goto DECODE_EXPONENT;
      break;
    }

    // 其他情况跳出整数部分扫描循环
    default: {
      goto BREAK_INT_LOOP;
      break;
    }
    }
  }

BREAK_INT_LOOP:

  ds->lastType = JT_INT;
  ds->start = offset;

  // 检查整数部分的符号位和溢出情况
  if (intNeg == 1 && (intValue & 0x8000000000000000ULL) != 0)
    return ds->dec->newUnsignedLong(ds->prv, intValue);

# 返回一个新的无符号长整型值，使用给定的私有数据和整数值。


  else if ((intValue >> 31))

# 如果整数值的符号位为1（即负数），则执行以下操作。


    return ds->dec->newLong(ds->prv, (JSINT64)(intValue * (JSINT64)intNeg));

# 返回一个新的长整型值，使用给定的私有数据和整数值乘以负数因子后的结果。


  else

# 如果整数值为非负数，则执行以下操作。


    return ds->dec->newInt(ds->prv, (JSINT32)(intValue * intNeg));

# 返回一个新的整型值，使用给定的私有数据和整数值乘以负数因子后的结果。
DECODE_FRACTION:

  // 如果精确浮点标志被设置，调用精确浮点解析函数
  if (ds->dec->preciseFloat) {
    return decodePreciseFloat(ds);
  }

  // 初始化小数部分值
  frcValue = 0.0;

  // 循环扫描小数部分
  for (;;) {
    chr = (int)(unsigned char)*(offset);

    switch (chr) {
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9': {
      // 如果小数位数未达到最大限制，则累加小数值
      if (decimalCount < JSON_DOUBLE_MAX_DECIMALS) {
        frcValue = frcValue * 10.0 + (double)(chr - 48);
        decimalCount++;
      }
      offset++;
      break;
    }
    case 'e':
    case 'E': {
      // 遇到指数符号，跳转到解析指数部分
      offset++;
      goto DECODE_EXPONENT;
      break;
    }
    default: {
      // 遇到其他字符，跳出小数部分解析循环
      goto BREAK_FRC_LOOP;
    }
    }
  }

BREAK_FRC_LOOP:
  // FIXME: 在此处检查算术溢出
  // 设置解析状态为双精度浮点数
  ds->lastType = JT_DOUBLE;
  // 记录解析开始位置
  ds->start = offset;
  // 返回新创建的双精度浮点数值
  return ds->dec->newDouble(
      ds->prv,
      createDouble((double)intNeg, (double)intValue, frcValue, decimalCount));

DECODE_EXPONENT:
  // 如果精确浮点标志被设置，调用精确浮点解析函数
  if (ds->dec->preciseFloat) {
    return decodePreciseFloat(ds);
  }

  // 初始化指数部分的符号
  expNeg = 1.0;

  // 检查指数部分的符号
  if (*(offset) == '-') {
    expNeg = -1.0;
    offset++;
  } else if (*(offset) == '+') {
    expNeg = +1.0;
    offset++;
  }

  // 初始化指数值
  expValue = 0.0;

  // 循环扫描指数部分
  for (;;) {
    chr = (int)(unsigned char)*(offset);

    switch (chr) {
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9': {
      // 累加指数值
      expValue = expValue * 10.0 + (double)(chr - 48);
      offset++;
      break;
    }
    default: {
      // 遇到其他字符，跳出指数部分解析循环
      goto BREAK_EXP_LOOP;
    }
    }
  }

DECODE_NAN:
  // 跳过 'NaN' 字符串
  offset++;
  if (*(offset++) != 'a')
    goto SET_NAN_ERROR;
  if (*(offset++) != 'N')
    goto SET_NAN_ERROR;

  // 设置解析状态为 null 类型
  ds->lastType = JT_NULL;
  // 记录解析开始位置
  ds->start = offset;
  // 返回新创建的 null 值
  return ds->dec->newNull(ds->prv);

SET_NAN_ERROR:
  // 解析 'NaN' 时发现意外字符，设置错误状态
  return SetError(ds, -1, "Unexpected character found when decoding 'NaN'");

DECODE_INF:
  // 跳过 'Infinity' 字符串
  offset++;
  if (*(offset++) != 'n')
    goto SET_INF_ERROR;
  if (*(offset++) != 'f')
    goto SET_INF_ERROR;
  if (*(offset++) != 'i')
    goto SET_INF_ERROR;
  if (*(offset++) != 'n')
    goto SET_INF_ERROR;
  if (*(offset++) != 'i')
    goto SET_INF_ERROR;
  if (*(offset++) != 't')
    goto SET_INF_ERROR;
  if (*(offset++) != 'y')
    goto SET_INF_ERROR;

  // 记录解析开始位置
  ds->start = offset;

  // 根据正负号设置解析状态为正无穷或负无穷
  if (intNeg == 1) {
    ds->lastType = JT_POS_INF;
    return ds->dec->newPosInf(ds->prv);
  } else {
    ds->lastType = JT_NEG_INF;
    return ds->dec->newNegInf(ds->prv);
  }

SET_INF_ERROR:
  // 根据正负号设置错误消息，解析 'Infinity' 时发现意外字符
  if (intNeg == 1) {
    const char *msg = "Unexpected character found when decoding 'Infinity'";
    return SetError(ds, -1, msg);
  } else {
    const char *msg = "Unexpected character found when decoding '-Infinity'";
    return SetError(ds, -1, msg);
  }

BREAK_EXP_LOOP:
  // FIXME: 在此处检查算术溢出
  // 设置解析状态为双精度浮点数
  ds->lastType = JT_DOUBLE;
  // 记录解析开始位置
  ds->start = offset;
  // 返回新创建的双精度浮点数值，乘以指数部分的值
  return ds->dec->newDouble(
      ds->prv,
      createDouble((double)intNeg, (double)intValue, frcValue, decimalCount) *
          pow(10.0, expValue * expNeg));
}
JSOBJ FASTCALL_MSVC decode_true(struct DecoderState *ds) {
  // 从起始位置开始解析，offset 指向数据流的第一个字节
  char *offset = ds->start;
  // 移动到下一个字节，跳过 't' 的位置
  offset++;

  // 检查下一个字符是否为 'r'
  if (*(offset++) != 'r')
    goto SETERROR;
  // 检查下一个字符是否为 'u'
  if (*(offset++) != 'u')
    goto SETERROR;
  // 检查下一个字符是否为 'e'
  if (*(offset++) != 'e')
    goto SETERROR;

  // 标记上一个解析的类型为 true
  ds->lastType = JT_TRUE;
  // 更新起始解析位置到当前 offset
  ds->start = offset;
  // 调用解析器的 newTrue 方法并返回结果
  return ds->dec->newTrue(ds->prv);

SETERROR:
  // 如果出现不匹配的字符，则设置错误信息并返回错误
  return SetError(ds, -1, "Unexpected character found when decoding 'true'");
}

JSOBJ FASTCALL_MSVC decode_false(struct DecoderState *ds) {
  // 从起始位置开始解析，offset 指向数据流的第一个字节
  char *offset = ds->start;
  // 移动到下一个字节，跳过 'f' 的位置
  offset++;

  // 检查下一个字符是否为 'a'
  if (*(offset++) != 'a')
    goto SETERROR;
  // 检查下一个字符是否为 'l'
  if (*(offset++) != 'l')
    goto SETERROR;
  // 检查下一个字符是否为 's'
  if (*(offset++) != 's')
    goto SETERROR;
  // 检查下一个字符是否为 'e'
  if (*(offset++) != 'e')
    goto SETERROR;

  // 标记上一个解析的类型为 false
  ds->lastType = JT_FALSE;
  // 更新起始解析位置到当前 offset
  ds->start = offset;
  // 调用解析器的 newFalse 方法并返回结果
  return ds->dec->newFalse(ds->prv);

SETERROR:
  // 如果出现不匹配的字符，则设置错误信息并返回错误
  return SetError(ds, -1, "Unexpected character found when decoding 'false'");
}

JSOBJ FASTCALL_MSVC decode_null(struct DecoderState *ds) {
  // 从起始位置开始解析，offset 指向数据流的第一个字节
  char *offset = ds->start;
  // 移动到下一个字节，跳过 'n' 的位置
  offset++;

  // 检查下一个字符是否为 'u'
  if (*(offset++) != 'u')
    goto SETERROR;
  // 检查下一个字符是否为 'l'
  if (*(offset++) != 'l')
    goto SETERROR;
  // 检查下一个字符是否为 'l'
  if (*(offset++) != 'l')
    goto SETERROR;

  // 标记上一个解析的类型为 null
  ds->lastType = JT_NULL;
  // 更新起始解析位置到当前 offset
  ds->start = offset;
  // 调用解析器的 newNull 方法并返回结果
  return ds->dec->newNull(ds->prv);

SETERROR:
  // 如果出现不匹配的字符，则设置错误信息并返回错误
  return SetError(ds, -1, "Unexpected character found when decoding 'null'");
}

void FASTCALL_MSVC SkipWhitespace(struct DecoderState *ds) {
  // 定义偏移量指针
  char *offset;

  // 从起始位置开始遍历数据流，直到遇到非空白字符或者到达数据流末尾
  for (offset = ds->start; (ds->end - offset) > 0; offset++) {
    switch (*offset) {
    // 对于空格、制表符、回车和换行符，跳过不处理
    case ' ':
    case '\t':
    case '\r':
    case '\n':
      break;

    // 对于其他字符，更新起始解析位置到当前 offset，并返回
    default:
      ds->start = offset;
      return;
    }
  }

  // 如果遍历到达数据流末尾，则将起始解析位置设置为数据流末尾
  if (offset == ds->end) {
    ds->start = ds->end;
  }
}

enum DECODESTRINGSTATE {
  DS_ISNULL = 0x32, // 表示状态为 null 的标识符
  DS_ISQUOTE,       // 表示状态为引号的标识符
  DS_ISESCAPE,      // 表示状态为转义字符的标识符
  DS_UTFLENERROR,   // 表示 UTF-8 长度错误的标识符
};

// 解析器使用的查找表，用于判断字符的状态
static const JSUINT8 g_decoderLookup[256] = {
    /* 0x00 */ DS_ISNULL,
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
    /* 0x10 */ 1,
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
    /* 0x20 */ 1,
    1,
    DS_ISQUOTE,
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
    1,
    1,
    1,
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
    DS_ISESCAPE,
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
    // 省略部分查找表内容，因为未提供完整数据
};
    /* 0xa0 */ 1,
    /* 0xa1 */ 1,
    /* 0xa2 */ 1,
    /* 0xa3 */ 1,
    /* 0xa4 */ 1,
    /* 0xa5 */ 1,
    /* 0xa6 */ 1,
    /* 0xa7 */ 1,
    /* 0xa8 */ 1,
    /* 0xa9 */ 1,
    /* 0xaa */ 1,
    /* 0xab */ 1,
    /* 0xac */ 1,
    /* 0xad */ 1,
    /* 0xae */ 1,
    /* 0xaf */ 1,
    /* 0xb0 */ 1,
    /* 0xb1 */ 1,
    /* 0xb2 */ 1,
    /* 0xb3 */ 1,
    /* 0xb4 */ 1,
    /* 0xb5 */ 1,
    /* 0xb6 */ 1,
    /* 0xb7 */ 1,
    /* 0xb8 */ 1,
    /* 0xb9 */ 1,
    /* 0xba */ 1,
    /* 0xbb */ 1,
    /* 0xbc */ 1,
    /* 0xbd */ 1,
    /* 0xbe */ 1,
    /* 0xbf */ 1,
    /* 0xc0 */ 2,
    /* 0xc1 */ 2,
    /* 0xc2 */ 2,
    /* 0xc3 */ 2,
    /* 0xc4 */ 2,
    /* 0xc5 */ 2,
    /* 0xc6 */ 2,
    /* 0xc7 */ 2,
    /* 0xc8 */ 2,
    /* 0xc9 */ 2,
    /* 0xca */ 2,
    /* 0xcb */ 2,
    /* 0xcc */ 2,
    /* 0xcd */ 2,
    /* 0xce */ 2,
    /* 0xcf */ 2,
    /* 0xd0 */ 2,
    /* 0xd1 */ 2,
    /* 0xd2 */ 2,
    /* 0xd3 */ 2,
    /* 0xd4 */ 2,
    /* 0xd5 */ 2,
    /* 0xd6 */ 2,
    /* 0xd7 */ 2,
    /* 0xd8 */ 2,
    /* 0xd9 */ 2,
    /* 0xda */ 2,
    /* 0xdb */ 2,
    /* 0xdc */ 2,
    /* 0xdd */ 2,
    /* 0xde */ 2,
    /* 0xdf */ 2,
    /* 0xe0 */ 3,
    /* 0xe1 */ 3,
    /* 0xe2 */ 3,
    /* 0xe3 */ 3,
    /* 0xe4 */ 3,
    /* 0xe5 */ 3,
    /* 0xe6 */ 3,
    /* 0xe7 */ 3,
    /* 0xe8 */ 3,
    /* 0xe9 */ 3,
    /* 0xea */ 3,
    /* 0xeb */ 3,
    /* 0xec */ 3,
    /* 0xed */ 3,
    /* 0xee */ 3,
    /* 0xef */ 3,
    /* 0xf0 */ 4,
    /* 0xf1 */ 4,
    /* 0xf2 */ 4,
    /* 0xf3 */ 4,
    /* 0xf4 */ 4,
    /* 0xf5 */ 4,
    /* 0xf6 */ 4,
    /* 0xf7 */ 4,
    /* 0xf8 */ DS_UTFLENERROR,
    /* 0xf9 */ DS_UTFLENERROR,
    /* 0xfa */ DS_UTFLENERROR,
    /* 0xfb */ DS_UTFLENERROR,
    /* 0xfc */ DS_UTFLENERROR,
    /* 0xfd */ DS_UTFLENERROR,
    /* 0xfe */ DS_UTFLENERROR,
    /* 0xff */ DS_UTFLENERROR,
};

JSOBJ FASTCALL_MSVC decode_string(struct DecoderState *ds) {
  // 初始化一个数组，用于存储UTF-16编码的字符序列
  JSUTF16 sur[2] = {0};
  int iSur = 0;
  int index;
  wchar_t *escOffset;
  wchar_t *escStart;
  // 计算转义字符的长度
  size_t escLen = (ds->escEnd - ds->escStart);
  JSUINT8 *inputOffset;
  JSUINT8 oct;
  JSUTF32 ucs;
  // 将上一个解码类型设为无效
  ds->lastType = JT_INVALID;
  // 移动解码器状态开始指针到下一个位置
  ds->start++;

  // 如果剩余的解码字符串长度大于转义字符的长度
  if ((size_t)(ds->end - ds->start) > escLen) {
    size_t newSize = (ds->end - ds->start);

    // 如果允许使用堆内存
    if (ds->escHeap) {
      // 检查是否能够分配足够的内存空间
      if (newSize > (SIZE_MAX / sizeof(wchar_t))) {
        // 返回内存分配错误
        return SetError(ds, -1, "Could not reserve memory block");
      }
      // 尝试重新分配已有的转义字符内存块
      escStart = (wchar_t *)ds->dec->realloc(ds->escStart, newSize * sizeof(wchar_t));
      if (!escStart) {
        // 如果重新分配失败，释放原有内存并返回错误
        ds->dec->free(ds->escStart);
        return SetError(ds, -1, "Could not reserve memory block");
      }
      ds->escStart = escStart;
    } else {
      wchar_t *oldStart = ds->escStart;
      // 检查是否能够分配足够的内存空间
      if (newSize > (SIZE_MAX / sizeof(wchar_t))) {
        // 返回内存分配错误
        return SetError(ds, -1, "Could not reserve memory block");
      }
      // 在堆上分配新的转义字符内存块，并复制原有数据
      ds->escStart = (wchar_t *)ds->dec->malloc(newSize * sizeof(wchar_t));
      if (!ds->escStart) {
        // 如果分配失败，返回内存分配错误
        return SetError(ds, -1, "Could not reserve memory block");
      }
      ds->escHeap = 1;
      memcpy(ds->escStart, oldStart, escLen * sizeof(wchar_t));
    }

    // 更新转义字符的结束指针
    ds->escEnd = ds->escStart + newSize;
  }

  // 设置转义字符的当前偏移量为起始位置
  escOffset = ds->escStart;
  // 设置输入数据的偏移量为解码器状态的起始位置
  inputOffset = (JSUINT8 *)ds->start;

  // 开始循环处理输入数据
  for (;;) {
    // 根据当前输入字符的解码类型进行处理
    switch (g_decoderLookup[(JSUINT8)(*inputOffset)]) {
    case DS_ISNULL: {
      // 发现未匹配的单引号或双引号错误时，返回解码错误
      return SetError(ds, -1, "Unmatched ''\"' when when decoding 'string'");
    }
    case DS_ISQUOTE: {
      // 将最后的解码类型设为UTF-8，并返回新创建的字符串对象
      ds->lastType = JT_UTF8;
      inputOffset++;
      ds->start += ((char *)inputOffset - (ds->start));
      return ds->dec->newString(ds->prv, ds->escStart, escOffset);
    }
    case DS_UTFLENERROR: {
      // 发现无效的UTF-8序列长度时，返回解码错误
      return SetError(ds, -1, "Invalid UTF-8 sequence length when decoding 'string'");
    }
    # 处理字符串中的转义字符情况
    case DS_ISESCAPE:
      # 移动到下一个字符位置
      inputOffset++;
      switch (*inputOffset) {
      case '\\':
        # 将 '\\' 转义为 '\\'
        *(escOffset++) = L'\\';
        inputOffset++;
        continue;
      case '\"':
        # 将 '\"' 转义为 '\"'
        *(escOffset++) = L'\"';
        inputOffset++;
        continue;
      case '/':
        # 将 '/' 转义为 '/'
        *(escOffset++) = L'/';
        inputOffset++;
        continue;
      case 'b':
        # 将 '\b' 转义为退格字符
        *(escOffset++) = L'\b';
        inputOffset++;
        continue;
      case 'f':
        # 将 '\f' 转义为换页符
        *(escOffset++) = L'\f';
        inputOffset++;
        continue;
      case 'n':
        # 将 '\n' 转义为换行符
        *(escOffset++) = L'\n';
        inputOffset++;
        continue;
      case 'r':
        # 将 '\r' 转义为回车符
        *(escOffset++) = L'\r';
        inputOffset++;
        continue;
      case 't':
        # 将 '\t' 转义为制表符
        *(escOffset++) = L'\t';
        inputOffset++;
        continue;

      case 'u': {
        int index;
        # 处理Unicode转义序列
        inputOffset++;

        for (index = 0; index < 4; index++) {
          switch (*inputOffset) {
          case '\0':
            # Unicode转义序列未终止时报错
            return SetError(ds, -1,
                            "Unterminated unicode "
                            "escape sequence when "
                            "decoding 'string'");
          default:
            # 在Unicode转义序列中遇到意外字符时报错
            return SetError(ds, -1,
                            "Unexpected character in "
                            "unicode escape sequence "
                            "when decoding 'string'");

          case '0':
          case '1':
          case '2':
          case '3':
          case '4':
          case '5':
          case '6':
          case '7':
          case '8':
          case '9':
            # 解析Unicode十六进制数值
            sur[iSur] = (sur[iSur] << 4) + (JSUTF16)(*inputOffset - '0');
            break;

          case 'a':
          case 'b':
          case 'c':
          case 'd':
          case 'e':
          case 'f':
            # 解析Unicode十六进制数值（小写字母）
            sur[iSur] = (sur[iSur] << 4) + 10 + (JSUTF16)(*inputOffset - 'a');
            break;

          case 'A':
          case 'B':
          case 'C':
          case 'D':
          case 'E':
          case 'F':
            # 解析Unicode十六进制数值（大写字母）
            sur[iSur] = (sur[iSur] << 4) + 10 + (JSUTF16)(*inputOffset - 'A');
            break;
          }

          inputOffset++;
        }

        if (iSur == 0) {
          if ((sur[iSur] & 0xfc00) == 0xd800) {
            // 首字符为代理对的高位，继续解析
            iSur++;
            break;
          }
          (*escOffset++) = (wchar_t)sur[iSur];
          iSur = 0;
        } else {
          // 解码代理对
          if ((sur[1] & 0xfc00) != 0xdc00) {
            return SetError(ds, -1,
                            "Unpaired high surrogate when "
                            "decoding 'string'");
          }
#if WCHAR_MAX == 0xffff
          // 如果 wchar_t 的最大值为 0xffff，表示使用 UTF-16 编码
          (*escOffset++) = (wchar_t)sur[0];
          (*escOffset++) = (wchar_t)sur[1];
#else
          // 否则，计算 UTF-32 编码值
          (*escOffset++) = (wchar_t)0x10000 +
                           (((sur[0] - 0xd800) << 10) | (sur[1] - 0xdc00));
#endif
          // 重置 iSur 以结束当前的 UTF-16 编码序列处理
          iSur = 0;
        }
        break;
      }

      case '\0':
        // 遇到字符串中的未终止的转义序列时报错
        return SetError(ds, -1,
                        "Unterminated escape sequence when "
                        "decoding 'string'");
      default:
        // 遇到不识别的转义序列时报错
        return SetError(ds, -1,
                        "Unrecognized escape sequence when "
                        "decoding 'string'");
      }
      break;

    case 1: {
      // 处理单字节的 UTF-8 字符
      *(escOffset++) = (wchar_t)(*inputOffset++);
      break;
    }

    case 2: {
      // 处理双字节的 UTF-8 字符
      ucs = (*inputOffset++) & 0x1f;
      ucs <<= 6;
      if (((*inputOffset) & 0x80) != 0x80) {
        // 检测到无效的 UTF-8 序列
        return SetError(ds, -1,
                        "Invalid octet in UTF-8 sequence when "
                        "decoding 'string'");
      }
      ucs |= (*inputOffset++) & 0x3f;
      if (ucs < 0x80)
        // 检测到过长的 2 字节 UTF-8 序列
        return SetError(ds, -1,
                        "Overlong 2 byte UTF-8 sequence detected "
                        "when decoding 'string'");
      *(escOffset++) = (wchar_t)ucs;
      break;
    }

    case 3: {
      // 处理三字节的 UTF-8 字符
      JSUTF32 ucs = 0;
      ucs |= (*inputOffset++) & 0x0f;

      for (index = 0; index < 2; index++) {
        ucs <<= 6;
        oct = (*inputOffset++);

        if ((oct & 0x80) != 0x80) {
          // 检测到无效的 UTF-8 序列
          return SetError(ds, -1,
                          "Invalid octet in UTF-8 sequence when "
                          "decoding 'string'");
        }

        ucs |= oct & 0x3f;
      }

      if (ucs < 0x800)
        // 检测到过长的 3 字节 UTF-8 序列
        return SetError(ds, -1,
                        "Overlong 3 byte UTF-8 sequence detected "
                        "when encoding string");
      *(escOffset++) = (wchar_t)ucs;
      break;
    }

    case 4: {
      // 处理四字节的 UTF-8 字符
      JSUTF32 ucs = 0;
      ucs |= (*inputOffset++) & 0x07;

      for (index = 0; index < 3; index++) {
        ucs <<= 6;
        oct = (*inputOffset++);

        if ((oct & 0x80) != 0x80) {
          // 检测到无效的 UTF-8 序列
          return SetError(ds, -1,
                          "Invalid octet in UTF-8 sequence when "
                          "decoding 'string'");
        }

        ucs |= oct & 0x3f;
      }

      if (ucs < 0x10000)
        // 检测到过长的 4 字节 UTF-8 序列
        return SetError(ds, -1,
                        "Overlong 4 byte UTF-8 sequence detected "
                        "when decoding 'string'");

#if WCHAR_MAX == 0xffff
      if (ucs >= 0x10000) {
        // 在 UTF-16 编码下处理超出 BMP 范围的字符
        ucs -= 0x10000;
        *(escOffset++) = (wchar_t)(ucs >> 10) + 0xd800;
        *(escOffset++) = (wchar_t)(ucs & 0x3ff) + 0xdc00;
      } else {
        *(escOffset++) = (wchar_t)ucs;
      }
#else
      // 在 UTF-32 编码下直接写入字符
      *(escOffset++) = (wchar_t)ucs;
#endif
      break;
    }
    }
  }
}

JSOBJ FASTCALL_MSVC decode_array(struct DecoderState *ds) {
  // 解码 JSON 数组，递增对象深度计数
  JSOBJ itemValue;
  JSOBJ newObj;
  int len;
  ds->objDepth++;
  if (ds->objDepth > JSON_MAX_OBJECT_DEPTH) {
    return SetError(ds, -1, "Reached object decoding depth limit");



// 返回错误：已达到对象解码深度限制



  }

  newObj = ds->dec->newArray(ds->prv, ds->dec);
  len = 0;



  // 创建新的数组对象
  newObj = ds->dec->newArray(ds->prv, ds->dec);
  // 初始化数组长度为0
  len = 0;



  ds->lastType = JT_INVALID;
  ds->start++;



  // 设置最后解析的类型为无效
  ds->lastType = JT_INVALID;
  // 指针向前移动一步，跳过当前位置的字符
  ds->start++;



  for (;;) {
    SkipWhitespace(ds);

    if ((*ds->start) == ']') {
      ds->objDepth--;
      if (len == 0) {
        ds->start++;
        return ds->dec->endArray(ds->prv, newObj);
      }

      ds->dec->releaseObject(ds->prv, newObj, ds->dec);
      return SetError(
          ds, -1, "Unexpected character found when decoding array value (1)");
    }

    itemValue = decode_any(ds);

    if (itemValue == NULL) {
      ds->dec->releaseObject(ds->prv, newObj, ds->dec);
      return NULL;
    }

    if (!ds->dec->arrayAddItem(ds->prv, newObj, itemValue)) {
      ds->dec->releaseObject(ds->prv, newObj, ds->dec);
      return NULL;
    }

    SkipWhitespace(ds);

    switch (*(ds->start++)) {
    case ']': {
      ds->objDepth--;
      return ds->dec->endArray(ds->prv, newObj);
    }
    case ',':
      break;

    default:
      ds->dec->releaseObject(ds->prv, newObj, ds->dec);
      return SetError(
          ds, -1, "Unexpected character found when decoding array value (2)");
    }

    len++;
  }



  // 无限循环，解析数组中的各个元素
  for (;;) {
    // 跳过空白字符
    SkipWhitespace(ds);

    // 如果当前字符为']'，表示数组结束
    if ((*ds->start) == ']') {
      // 减少对象深度计数
      ds->objDepth--;
      // 如果数组长度为0，直接跳过当前字符并返回结束数组解析
      if (len == 0) {
        ds->start++;
        return ds->dec->endArray(ds->prv, newObj);
      }

      // 释放数组对象并返回错误，指示在解码数组值时发现意外字符(1)
      ds->dec->releaseObject(ds->prv, newObj, ds->dec);
      return SetError(
          ds, -1, "Unexpected character found when decoding array value (1)");
    }

    // 解码当前数组元素的值
    itemValue = decode_any(ds);

    // 如果解析值为空，释放数组对象并返回空值
    if (itemValue == NULL) {
      ds->dec->releaseObject(ds->prv, newObj, ds->dec);
      return NULL;
    }

    // 将解析的数组元素值添加到数组中
    if (!ds->dec->arrayAddItem(ds->prv, newObj, itemValue)) {
      ds->dec->releaseObject(ds->prv, newObj, ds->dec);
      return NULL;
    }

    // 跳过空白字符
    SkipWhitespace(ds);

    // 根据当前字符的不同情况处理
    switch (*(ds->start++)) {
    // 如果当前字符为']'，表示数组结束
    case ']': {
      ds->objDepth--;
      // 返回结束数组解析
      return ds->dec->endArray(ds->prv, newObj);
    }
    // 如果当前字符为','，继续处理下一个数组元素
    case ',':
      break;

    // 其他情况，释放数组对象并返回错误，指示在解码数组值时发现意外字符(2)
    default:
      ds->dec->releaseObject(ds->prv, newObj, ds->dec);
      return SetError(
          ds, -1, "Unexpected character found when decoding array value (2)");
    }

    // 增加数组长度计数
    len++;
  }
}

JSOBJ FASTCALL_MSVC decode_object(struct DecoderState *ds) {
  JSOBJ itemName;
  JSOBJ itemValue;
  JSOBJ newObj;

  ds->objDepth++;  // 增加对象解码深度计数器
  if (ds->objDepth > JSON_MAX_OBJECT_DEPTH) {  // 如果对象解码深度超过最大限制
    return SetError(ds, -1, "Reached object decoding depth limit");  // 返回解码错误
  }

  newObj = ds->dec->newObject(ds->prv, ds->dec);  // 创建新的对象

  ds->start++;  // 跳过当前字符，准备解析对象内容

  for (;;) {  // 进入对象内容解析循环
    SkipWhitespace(ds);  // 跳过空白字符

    if ((*ds->start) == '}') {  // 如果当前字符为 '}'，表示对象结束
      ds->objDepth--;  // 减少对象解码深度计数器
      ds->start++;  // 跳过 '}'
      return ds->dec->endObject(ds->prv, newObj);  // 调用结束对象解码的处理函数，并返回结果
    }

    ds->lastType = JT_INVALID;  // 设置上一个解析的类型为无效

    itemName = decode_any(ds);  // 解析对象的键名

    if (itemName == NULL) {  // 如果键名解析失败
      ds->dec->releaseObject(ds->prv, newObj, ds->dec);  // 释放新对象
      return NULL;  // 返回空指针，表示解析错误
    }

    if (ds->lastType != JT_UTF8) {  // 如果键名类型不是 UTF-8 字符串
      ds->dec->releaseObject(ds->prv, newObj, ds->dec);  // 释放新对象
      ds->dec->releaseObject(ds->prv, itemName, ds->dec);  // 释放键名对象
      return SetError(ds, -1, "Key name of object must be 'string' when decoding 'object'");  // 返回键名类型错误
    }

    SkipWhitespace(ds);  // 跳过空白字符

    if (*(ds->start++) != ':') {  // 如果当前字符不是 ':'
      ds->dec->releaseObject(ds->prv, newObj, ds->dec);  // 释放新对象
      ds->dec->releaseObject(ds->prv, itemName, ds->dec);  // 释放键名对象
      return SetError(ds, -1, "No ':' found when decoding object value");  // 返回没有找到 ':' 的错误
    }

    SkipWhitespace(ds);  // 跳过空白字符

    itemValue = decode_any(ds);  // 解析对象的值

    if (itemValue == NULL) {  // 如果值解析失败
      ds->dec->releaseObject(ds->prv, newObj, ds->dec);  // 释放新对象
      ds->dec->releaseObject(ds->prv, itemName, ds->dec);  // 释放键名对象
      return NULL;  // 返回空指针，表示解析错误
    }

    if (!ds->dec->objectAddKey(ds->prv, newObj, itemName, itemValue)) {  // 向对象中添加键值对失败
      ds->dec->releaseObject(ds->prv, newObj, ds->dec);  // 释放新对象
      ds->dec->releaseObject(ds->prv, itemName, ds->dec);  // 释放键名对象
      ds->dec->releaseObject(ds->prv, itemValue, ds->dec);  // 释放值对象
      return NULL;  // 返回空指针，表示解析错误
    }

    SkipWhitespace(ds);  // 跳过空白字符

    switch (*(ds->start++)) {  // 检查下一个字符
    case '}': {  // 如果是 '}'
      ds->objDepth--;  // 减少对象解码深度计数器
      return ds->dec->endObject(ds->prv, newObj);  // 调用结束对象解码的处理函数，并返回结果
    }
    case ',':
      break;  // 继续下一个键值对
    default:
      ds->dec->releaseObject(ds->prv, newObj, ds->dec);  // 释放新对象
      return SetError(ds, -1, "Unexpected character found when decoding object value");  // 返回解析到意外字符的错误
    }
  }
}

JSOBJ FASTCALL_MSVC decode_any(struct DecoderState *ds) {
  for (;;) {  // 进入解析任意类型数据的循环
    switch (*ds->start) {  // 根据当前字符类型进行不同解析
    case '\"':
      return decode_string(ds);  // 解析字符串
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
    case 'I':
    case 'N':
    case '-':
      return decode_numeric(ds);  // 解析数值
    case '[':
      return decode_array(ds);  // 解析数组
    case '{':
      return decode_object(ds);  // 解析对象
    case 't':
      return decode_true(ds);  // 解析 true
    case 'f':
      return decode_false(ds);  // 解析 false
    case 'n':
      return decode_null(ds);  // 解析 null
    case ' ':
    case '\t':
    case '\r':
    case '\n':
      // White space
      ds->start++;  // 跳过空白字符
      break;
    default:
      return SetError(ds, -1, "Expected object or value");  // 返回期望对象或值的错误
    }
  }
}
/*
FIXME: 根据 cbBuffer 的大小调整 escBuffer 的大小，以避免每次 Unicode 转义时遇到限制
*/
JSOBJ JSON_DecodeObject(JSONObjectDecoder *dec, const char *buffer,
                        size_t cbBuffer) {
  // 声明本地变量
  char *locale;
  // 定义解码器状态结构体实例
  struct DecoderState ds;
  // 定义用于存储转义字符的缓冲区
  wchar_t escBuffer[(JSON_MAX_STACK_BUFFER_SIZE / sizeof(wchar_t))];
  // 定义返回的 JSON 对象
  JSOBJ ret;

  // 初始化解码器状态结构体
  ds.start = (char *)buffer;
  ds.end = ds.start + cbBuffer;

  // 设置转义字符缓冲区的起始和结束位置
  ds.escStart = escBuffer;
  ds.escEnd = ds.escStart + (JSON_MAX_STACK_BUFFER_SIZE / sizeof(wchar_t));
  ds.escHeap = 0;
  ds.prv = dec->prv;
  ds.dec = dec;
  ds.dec->errorStr = NULL;
  ds.dec->errorOffset = NULL;
  ds.objDepth = 0;

  // 复制解码器到解码器状态结构体
  ds.dec = dec;

  // 获取当前的数字格式本地化设置
  locale = setlocale(LC_NUMERIC, NULL);
  if (!locale) {
    // 若获取失败则返回错误
    return SetError(&ds, -1, "setlocale call failed");
  }

  // 检查当前本地化设置是否为 "C"
  if (strcmp(locale, "C")) {
    // 如果不是 "C" 则保存当前本地化设置
    size_t len = strlen(locale) + 1;
    char *saved_locale = malloc(len);
    if (saved_locale == NULL) {
      // 分配失败则返回错误
      return SetError(&ds, -1, "Could not reserve memory block");
    }
    memcpy(saved_locale, locale, len);
    // 设置本地化为 "C"
    setlocale(LC_NUMERIC, "C");
    // 解码任意数据类型
    ret = decode_any(&ds);
    // 恢复之前保存的本地化设置
    setlocale(LC_NUMERIC, saved_locale);
    free(saved_locale);
  } else {
    // 如果已经是 "C" 则直接解码任意数据类型
    ret = decode_any(&ds);
  }

  // 如果转义字符缓冲区在堆上分配，则释放其内存
  if (ds.escHeap) {
    dec->free(ds.escStart);
  }

  // 跳过空白字符
  SkipWhitespace(&ds);

  // 如果解析后仍有剩余数据且有返回对象，则释放对象并返回错误
  if (ds.start != ds.end && ret) {
    dec->releaseObject(ds.prv, ret, ds.dec);
    return SetError(&ds, -1, "Trailing data");
  }

  // 返回解码后的 JSON 对象
  return ret;
}
```