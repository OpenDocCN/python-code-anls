# `D:\src\scipysrc\pandas\pandas\_libs\include\pandas\vendored\ujson\lib\ultrajson.h`

```
/*
版权声明，指定此代码的版权和使用条件
Copyright (c) 2011-2013, ESN Social Software AB and Jonas Tarnstrom
保留所有权利。

在源代码和二进制形式的再发布和使用，无论是否修改，必须遵守以下条件：
    * 必须保留上述版权声明、条件列表和以下免责声明。
    * 在提供的文档和/或其他材料中，必须在二进制形式的再发布中重现上述版权声明、条件列表和以下免责声明。
    * 未经特定的书面许可，不得使用 ESN Social Software AB 或其贡献者的名称来认可或推广从此软件衍生的产品。

此软件按 "原样" 提供，不提供任何明示或暗示的保证，包括但不限于对适销性和特定用途适用性的暗示保证。在任何情况下，ESN Social Software AB 或 Jonas Tarnstrom 均不对任何直接、间接、偶然、特殊、惩罚性或后果性损害负责，包括但不限于采购替代商品或服务、使用损失、数据或利润损失，或业务中断，无论是在合同、严格责任或侵权（包括疏忽或其他方式）的任何理论下造成的，即使已被告知可能性。

从 MODP_ASCII 获得部分代码 - ASCII 转换（大写/小写等）
https://github.com/client9/stringencoders
版权所有 (c) 2007 Nick Galbreath -- nickg [at] modp [dot] com. 保留所有权利。

从 TCL 库中获取的数值解码器
https://www.opensource.apple.com/source/tcl/tcl-14/tcl/license.terms
版权所有 (c) 1988-1993 加利福尼亚大学理事会。
版权所有 (c) 1994 Sun Microsystems, Inc.
*/
#ifndef JSON_MAX_STACK_BUFFER_SIZE
// 如果未定义 JSON_MAX_STACK_BUFFER_SIZE 宏，则定义为 131072
#define JSON_MAX_STACK_BUFFER_SIZE 131072
#endif

#ifdef _WIN32
// 如果是在 Windows 平台下编译

// 定义 64 位有符号和无符号整数类型
typedef __int64 JSINT64;
typedef unsigned __int64 JSUINT64;

// 定义 32 位有符号和无符号整数类型，以及无符号 8 位、16 位整数类型
typedef __int32 JSINT32;
typedef unsigned __int32 JSUINT32;
typedef unsigned __int8 JSUINT8;
typedef unsigned __int16 JSUTF16;
typedef unsigned __int32 JSUTF32;

// 定义 64 位长整型
typedef __int64 JSLONG;

// 定义导出函数的标记
#define EXPORTFUNCTION __declspec(dllexport)

// 定义 MSVC 下的快速调用约定
#define FASTCALL_MSVC __fastcall

// 定义静态内联函数的前缀
#define INLINE_PREFIX static __inline

#else
// 如果不是在 Windows 平台下编译

// 包含 C99 标准的头文件，定义标准整数类型
#include <stdint.h>
typedef int64_t JSINT64;
typedef uint64_t JSUINT64;

typedef int32_t JSINT32;
typedef uint32_t JSUINT32;

// 定义 MSVC 下的快速调用约定为空
#define FASTCALL_MSVC

// 定义静态内联函数的前缀为 static inline
#define INLINE_PREFIX static inline

// 定义 8 位和 16 位无符号整数类型，以及 32 位 Unicode 字符类型
typedef uint8_t JSUINT8;
typedef uint16_t JSUTF16;
typedef uint32_t JSUTF32;

// 定义 64 位长整型
typedef int64_t JSLONG;

// 定义空的导出函数标记
#define EXPORTFUNCTION
#endif

// 如果未定义 __LITTLE_ENDIAN__ 和 __BIG_ENDIAN__ 宏
#if !(defined(__LITTLE_ENDIAN__) || defined(__BIG_ENDIAN__))

// 根据编译器提供的字节序信息判断
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
// 如果编译器字节序为小端序，则定义 __LITTLE_ENDIAN__
#define __LITTLE_ENDIAN__
#else

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
// 如果编译器字节序为大端序，则定义 __BIG_ENDIAN__
#define __BIG_ENDIAN__
#endif

#endif

#endif

// 如果既未定义 __LITTLE_ENDIAN__ 也未定义 __BIG_ENDIAN__
#if !defined(__LITTLE_ENDIAN__) && !defined(__BIG_ENDIAN__)
// 报错，因为不支持的字节序
#error "Endianness not supported"
#endif

// 枚举定义不同 JSON 数据类型
enum JSTYPES {
  JT_NULL,    // NULL
  JT_TRUE,    // boolean true
  JT_FALSE,   // boolean false
  JT_INT,     // (JSINT32 (signed 32-bit))
  JT_LONG,    // (JSINT64 (signed 64-bit))
  JT_DOUBLE,  // (double)
  JT_BIGNUM,  // integer larger than sys.maxsize
  JT_UTF8,    // (char 8-bit)
  JT_ARRAY,   // Array structure
  JT_OBJECT,  // Key/Value structure
  JT_INVALID, // Internal, do not return nor expect
  JT_POS_INF, // Positive infinity
  JT_NEG_INF, // Negative infinity
};

// 定义 JSON 对象和迭代器的类型
typedef void *JSOBJ;
typedef void *JSITER;

// 定义 JSON 类型上下文结构体，包含类型、编码器和私有数据
typedef struct __JSONTypeContext {
  int type;
  void *encoder;
  void *prv;
} JSONTypeContext;

/*
Function pointer declarations, suitable for implementing UltraJSON */
// 定义函数指针类型，用于实现 UltraJSON 的迭代和内存管理函数
typedef void (*JSPFN_ITERBEGIN)(JSOBJ obj, JSONTypeContext *tc);
typedef int (*JSPFN_ITERNEXT)(JSOBJ obj, JSONTypeContext *tc);
typedef void (*JSPFN_ITEREND)(JSOBJ obj, JSONTypeContext *tc);
typedef JSOBJ (*JSPFN_ITERGETVALUE)(JSOBJ obj, JSONTypeContext *tc);
typedef char *(*JSPFN_ITERGETNAME)(JSOBJ obj, JSONTypeContext *tc,
                                   size_t *outLen);
typedef void *(*JSPFN_MALLOC)(size_t size);
typedef void (*JSPFN_FREE)(void *pptr);
typedef void *(*JSPFN_REALLOC)(void *base, size_t size);
/*
定义一个结构体 JSONObjectEncoder，用于将对象编码为 JSON 格式

结构体成员说明：
*/

typedef struct __JSONObjectEncoder {
  void (*beginTypeContext)(JSOBJ obj, JSONTypeContext *tc);
  void (*endTypeContext)(JSOBJ obj, JSONTypeContext *tc);
  const char *(*getStringValue)(JSOBJ obj, JSONTypeContext *tc,
                                size_t *_outLen);
  JSINT64 (*getLongValue)(JSOBJ obj, JSONTypeContext *tc);
  JSINT32 (*getIntValue)(JSOBJ obj, JSONTypeContext *tc);
  double (*getDoubleValue)(JSOBJ obj, JSONTypeContext *tc);
  const char *(*getBigNumStringValue)(JSOBJ obj, JSONTypeContext *tc,
                                      size_t *_outLen);

  /*
  开始迭代可迭代对象（JS_ARRAY 或 JS_OBJECT）
  实现者应该在 ti->prv 中设置迭代状态
  */
  JSPFN_ITERBEGIN iterBegin;

  /*
  获取迭代中的下一个对象。返回 0 表示迭代结束，返回 1 表示还有更多项。
  实现者负责维护迭代状态，使用 ti->prv 字段
  */
  JSPFN_ITERNEXT iterNext;

  /*
  结束可迭代对象的迭代。
  可以在这里释放 ti->prv 中存储的任何迭代状态
  */
  JSPFN_ITEREND iterEnd;

  /*
  返回迭代器的值对象引用
  调用者负责返回字符串的生命周期管理。使用 iterNext/iterEnd 和 ti->prv 来跟踪当前对象
  */
  JSPFN_ITERGETVALUE iterGetValue;

  /*
  返回迭代器的名称
  调用者负责返回字符串的生命周期管理。使用 iterNext/iterEnd 和 ti->prv 来跟踪当前对象
  */
  JSPFN_ITERGETNAME iterGetName;

  /*
  根据前一个 getValue 调用中设置 ti->release = 1 来释放值。
  ti->prv 数组应包含释放值所需的上下文
  */
  void (*releaseObject)(JSOBJ obj);

  /* 库函数
  如果设为 NULL，则使用标准库的 malloc、realloc、free */
  JSPFN_MALLOC malloc;
  JSPFN_REALLOC realloc;
  JSPFN_FREE free;

  /*
  最大递归深度配置，设为 0 使用默认值（参见 JSON_MAX_RECURSION_DEPTH）
  */
  int recursionMax;

  /*
  双精度浮点数的最大小数位数配置 (0-9)
  */
  int doublePrecision;

  /*
  如果为 true，则输出将使用 ASCII 编码，所有大于 127 的字符将被编码为 \uXXXX
  如果为 false，则输出将使用 UTF-8 或其他字符集的原始字符串
  */
  int forceASCII;

  /*
  如果为 true，则将 '<', '>', '&' 字符编码为 \u003c, \u003e, 和 \u0026，否则不进行特殊编码
  */
  int encodeHTMLChars;

  /*
  缩进空格数配置
  */
  int indent;

  /*
  如果发生错误，设置为错误消息
  */
  const char *errorMsg;
  JSOBJ errorObj;

  /* 缓冲区相关 */
  char *start;
  char *offset;
  char *end;
  int heap;
  int level;
} JSONObjectEncoder;

/*
将对象结构编码为 JSON 格式的函数

参数：
obj - 表示对象的匿名类型
enc - 用于查询 JSOBJ 类型的函数定义
*/
/*
buffer - 预分配的缓冲区，用于存储结果。如果为 NULL，则函数会分配自己的缓冲区。
cbBuffer - 缓冲区的长度（如果 buffer 为 NULL，则被忽略）

Returns:
作为以空字符结尾的字符串的编码后的 JSON 对象。

NOTE:
如果提供的缓冲区不足以容纳结果，函数将分配一个新的缓冲区。提供的缓冲区的生命周期仍由调用者管理。

如果返回值不等于指定的缓冲区，则调用者必须使用 JSONObjectEncoder.free 或 free() 方法释放内存。
*/
EXPORTFUNCTION char *JSON_EncodeObject(JSOBJ obj, JSONObjectEncoder *enc,
                                       char *buffer, size_t cbBuffer);

typedef struct __JSONObjectDecoder {
  JSOBJ (*newString)(void *prv, wchar_t *start, wchar_t *end);
  int (*objectAddKey)(void *prv, JSOBJ obj, JSOBJ name, JSOBJ value);
  int (*arrayAddItem)(void *prv, JSOBJ obj, JSOBJ value);
  JSOBJ (*newTrue)(void *prv);
  JSOBJ (*newFalse)(void *prv);
  JSOBJ (*newNull)(void *prv);
  JSOBJ (*newPosInf)(void *prv);
  JSOBJ (*newNegInf)(void *prv);
  JSOBJ (*newObject)(void *prv, void *decoder);
  JSOBJ (*endObject)(void *prv, JSOBJ obj);
  JSOBJ (*newArray)(void *prv, void *decoder);
  JSOBJ (*endArray)(void *prv, JSOBJ obj);
  JSOBJ (*newInt)(void *prv, JSINT32 value);
  JSOBJ (*newLong)(void *prv, JSINT64 value);
  JSOBJ (*newUnsignedLong)(void *prv, JSUINT64 value);
  JSOBJ (*newDouble)(void *prv, double value);
  void (*releaseObject)(void *prv, JSOBJ obj, void *decoder);
  JSPFN_MALLOC malloc;
  JSPFN_FREE free;
  JSPFN_REALLOC realloc;
  char *errorStr;
  char *errorOffset;
  int preciseFloat;
  void *prv;
} JSONObjectDecoder;

/*
解码 JSON 对象并返回其 JSOBJ 表示。
dec - JSON 对象解码器对象，包含解码操作的函数指针和配置信息。
buffer - JSON 对象的字符串表示，以 null 结尾。
cbBuffer - 字符串 buffer 的长度。

Returns:
解码后的 JSOBJ 对象。

*/
EXPORTFUNCTION JSOBJ JSON_DecodeObject(JSONObjectDecoder *dec,
                                       const char *buffer, size_t cbBuffer);

/*
编码 JSON 对象并将其写入指定的缓冲区。
obj - 要编码的 JSON 对象。
enc - JSON 对象编码器对象，包含编码操作的函数指针和配置信息。
buffer - 用于存储编码结果的缓冲区。
cbBuffer - 缓冲区的长度。

*/
EXPORTFUNCTION void encode(JSOBJ obj, JSONObjectEncoder *enc, const char *buffer, size_t cbBuffer);
```