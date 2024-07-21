# `.\pytorch\third_party\miniz-2.1.0\miniz.h`

```py
/*
#pragma once

/* Defines to completely disable specific portions of miniz.c: 
   If all macros here are defined the only functionality remaining will be CRC-32, adler-32, tinfl, and tdefl. */

/* Define MINIZ_NO_STDIO to disable all usage and any functions which rely on stdio for file I/O. */
/*#define MINIZ_NO_STDIO */

/* If MINIZ_NO_TIME is specified then the ZIP archive functions will not be able to get the current time, or */
/* get/set file times, and the C run-time funcs that get/set times won't be called. */
#define MINIZ_NO_TIME

/* Define MINIZ_NO_ARCHIVE_APIS to disable all ZIP archive API's. */
/*#define MINIZ_NO_ARCHIVE_APIS */

/* Define MINIZ_NO_ARCHIVE_WRITING_APIS to disable all writing related ZIP archive API's. */
/*#define MINIZ_NO_ARCHIVE_WRITING_APIS */

/* Define MINIZ_NO_ZLIB_APIS to remove all ZLIB-style compression/decompression API's. */
/*#define MINIZ_NO_ZLIB_APIS */

/* Define MINIZ_NO_ZLIB_COMPATIBLE_NAME to disable zlib names, to prevent conflicts against stock zlib. */
#define MINIZ_NO_ZLIB_COMPATIBLE_NAMES

/* Define MINIZ_NO_MALLOC to disable all calls to malloc, free, and realloc. 
   Note if MINIZ_NO_MALLOC is defined then the user must always provide custom user alloc/free/realloc
   callbacks to the zlib and archive API's, and a few stand-alone helper API's which don't provide custom user
   functions (such as tdefl_compress_mem_to_heap() and tinfl_decompress_mem_to_heap()) won't work. */
/*#define MINIZ_NO_MALLOC */

#if defined(__TINYC__) && (defined(__linux) || defined(__linux__))
/* TODO: Work around "error: include file 'sys\utime.h' when compiling with tcc on Linux */
#define MINIZ_NO_TIME
#endif

#include <stddef.h>

#if !defined(MINIZ_NO_TIME) && !defined(MINIZ_NO_ARCHIVE_APIS)
#include <time.h>
#endif

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__i386) || defined(__i486__) || defined(__i486__) || defined(i386) || defined(__ia64__) || defined(__x86_64__)
/* MINIZ_X86_OR_X64_CPU is only used to help set the below macros. */
#define MINIZ_X86_OR_X64_CPU 1
#else
#define MINIZ_X86_OR_X64_CPU 0
#endif

#if (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__) || MINIZ_X86_OR_X64_CPU
/* Set MINIZ_LITTLE_ENDIAN to 1 if the processor is little endian. */
#define MINIZ_LITTLE_ENDIAN 1
#else
#define MINIZ_LITTLE_ENDIAN 0
#endif

/* Set MINIZ_USE_UNALIGNED_LOADS_AND_STORES only if not set */
#if !defined(MINIZ_USE_UNALIGNED_LOADS_AND_STORES)
#if MINIZ_X86_OR_X64_CPU
/* Set MINIZ_USE_UNALIGNED_LOADS_AND_STORES to 1 on CPU's that permit efficient integer loads and stores from unaligned addresses. */
/* zdevito: ASAN doesn't like unligned loads and stores, and -O3 optimizes the unoptimized code pattern away anyawy */
#define MINIZ_USE_UNALIGNED_LOADS_AND_STORES 0
/* zdevito: ASAN doesn't like unligned loads and stores, and -O3 optimizes the unoptimized code pattern away anyawy */
#endif
#endif
*/
/*#define MINIZ_UNALIGNED_USE_MEMCPY*/
#else
#define MINIZ_USE_UNALIGNED_LOADS_AND_STORES 0
#endif
#endif

#if defined(_M_X64) || defined(_WIN64) || defined(__MINGW64__) || defined(_LP64) || defined(__LP64__) || defined(__ia64__) || defined(__x86_64__)
/* 如果编译目标是64位架构，则设置MINIZ_HAS_64BIT_REGISTERS为1，表示支持64位寄存器操作 */
#define MINIZ_HAS_64BIT_REGISTERS 1
#else
/* 如果编译目标不是64位架构，则设置MINIZ_HAS_64BIT_REGISTERS为0，表示不支持64位寄存器操作 */
#define MINIZ_HAS_64BIT_REGISTERS 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------- zlib-style API Definitions. */

/* 为了与zlib更兼容，miniz.c中使用unsigned long来定义某些参数/结构成员。注意：mz_ulong可以是32位或64位！ */
typedef unsigned long mz_ulong;

/* mz_free()内部使用MZ_FREE()宏（默认调用free()，除非修改了MZ_MALLOC宏）来释放从堆中分配的块。 */
void mz_free(void *p);

#define MZ_ADLER32_INIT (1)
/* 当ptr==NULL时，mz_adler32()返回用于计算adler-32校验的初始值。 */
mz_ulong mz_adler32(mz_ulong adler, const unsigned char *ptr, size_t buf_len);

#define MZ_CRC32_INIT (0)
/* 当ptr==NULL时，mz_crc32()返回用于计算CRC-32校验的初始值。 */
mz_ulong mz_crc32(mz_ulong crc, const unsigned char *ptr, size_t buf_len);

/* 压缩策略 */
enum
{
    MZ_DEFAULT_STRATEGY = 0,
    MZ_FILTERED = 1,
    MZ_HUFFMAN_ONLY = 2,
    MZ_RLE = 3,
    MZ_FIXED = 4
};

/* 压缩方法 */
#define MZ_DEFLATED 8

/* 堆分配回调函数。注意，mz_alloc_func参数类型有意与zlib的不同：items/size是size_t，而不是unsigned long。 */
typedef void *(*mz_alloc_func)(void *opaque, size_t items, size_t size);
typedef void (*mz_free_func)(void *opaque, void *address);
typedef void *(*mz_realloc_func)(void *opaque, void *address, size_t items, size_t size);

/* 压缩级别：0-9是标准的zlib风格级别，10是最佳压缩（不兼容zlib，并且可能非常慢），MZ_DEFAULT_COMPRESSION=MZ_DEFAULT_LEVEL。 */
enum
{
    MZ_NO_COMPRESSION = 0,
    MZ_BEST_SPEED = 1,
    MZ_BEST_COMPRESSION = 9,
    MZ_UBER_COMPRESSION = 10,
    MZ_DEFAULT_LEVEL = 6,
    MZ_DEFAULT_COMPRESSION = -1
};

#define MZ_VERSION "10.1.0"
#define MZ_VERNUM 0xA100
#define MZ_VER_MAJOR 10
#define MZ_VER_MINOR 1
#define MZ_VER_REVISION 0
#define MZ_VER_SUBREVISION 0

#ifndef MINIZ_NO_ZLIB_APIS

/* Flush值。对于典型用法，您只需要MZ_NO_FLUSH和MZ_FINISH。其他值用于高级用法（参考zlib文档）。 */
enum
{
    MZ_NO_FLUSH = 0,
    MZ_PARTIAL_FLUSH = 1,
    MZ_SYNC_FLUSH = 2,
    MZ_FULL_FLUSH = 3,
    MZ_FINISH = 4,
    MZ_BLOCK = 5
};

/* 返回状态码。MZ_PARAM_ERROR是非标准的。 */
enum
{
    MZ_OK = 0,
    MZ_STREAM_END = 1,
    MZ_NEED_DICT = 2,
    MZ_ERRNO = -1,
    MZ_STREAM_ERROR = -2,
    MZ_DATA_ERROR = -3,
    MZ_MEM_ERROR = -4,
    MZ_BUF_ERROR = -5,
    # 定义常量 MZ_VERSION_ERROR 并赋值为 -6，表示版本错误的常量值
    MZ_VERSION_ERROR = -6,
    
    # 定义常量 MZ_PARAM_ERROR 并赋值为 -10000，表示参数错误的常量值
    MZ_PARAM_ERROR = -10000
};

/* Window bits */
#define MZ_DEFAULT_WINDOW_BITS 15

struct mz_internal_state;

/* Compression/decompression stream struct. */
typedef struct mz_stream_s
{
    const unsigned char *next_in; /* pointer to next byte to read */
    unsigned int avail_in;        /* number of bytes available at next_in */
    mz_ulong total_in;            /* total number of bytes consumed so far */

    unsigned char *next_out; /* pointer to next byte to write */
    unsigned int avail_out;  /* number of bytes that can be written to next_out */
    mz_ulong total_out;      /* total number of bytes produced so far */

    char *msg;                       /* error msg (unused) */
    struct mz_internal_state *state; /* internal state, allocated by zalloc/zfree */

    mz_alloc_func zalloc; /* optional heap allocation function (defaults to malloc) */
    mz_free_func zfree;   /* optional heap free function (defaults to free) */
    void *opaque;         /* heap alloc function user pointer */

    int data_type;     /* data_type (unused) */
    mz_ulong adler;    /* adler32 of the source or uncompressed data */
    mz_ulong reserved; /* not used */
} mz_stream;

typedef mz_stream *mz_streamp;

/* Returns the version string of miniz.c. */
const char *mz_version(void);

/* mz_deflateInit() initializes a compressor with default options: */
/* Parameters: */
/*  pStream must point to an initialized mz_stream struct. */
/*  level must be between [MZ_NO_COMPRESSION, MZ_BEST_COMPRESSION]. */
/*  level 1 enables a specially optimized compression function that's been optimized purely for performance, not ratio. */
/*  (This special func. is currently only enabled when MINIZ_USE_UNALIGNED_LOADS_AND_STORES and MINIZ_LITTLE_ENDIAN are defined.) */
/* Return values: */
/*  MZ_OK on success. */
/*  MZ_STREAM_ERROR if the stream is bogus. */
/*  MZ_PARAM_ERROR if the input parameters are bogus. */
/*  MZ_MEM_ERROR on out of memory. */
int mz_deflateInit(mz_streamp pStream, int level);

/* mz_deflateInit2() is like mz_deflate(), except with more control: */
/* Additional parameters: */
/*   method must be MZ_DEFLATED */
/*   window_bits must be MZ_DEFAULT_WINDOW_BITS (to wrap the deflate stream with zlib header/adler-32 footer) or -MZ_DEFAULT_WINDOW_BITS (raw deflate/no header or footer) */
/*   mem_level must be between [1, 9] (it's checked but ignored by miniz.c) */
int mz_deflateInit2(mz_streamp pStream, int level, int method, int window_bits, int mem_level, int strategy);

/* Quickly resets a compressor without having to reallocate anything. Same as calling mz_deflateEnd() followed by mz_deflateInit()/mz_deflateInit2(). */
int mz_deflateReset(mz_streamp pStream);

/* mz_deflate() compresses the input to output, consuming as much of the input and producing as much output as possible. */
/* Parameters: */
/*   pStream is the stream to read from and write to. You must initialize/update the next_in, avail_in, next_out, and avail_out members. */
/*   flush may be MZ_NO_FLUSH, MZ_PARTIAL_FLUSH/MZ_SYNC_FLUSH, MZ_FULL_FLUSH, or MZ_FINISH. */
/* Return values: */
/*   MZ_OK on success (when flushing, or if more input is needed but not available, and/or there's more output to be written but the output buffer is full). */
/*   MZ_STREAM_END if all input has been consumed and all output bytes have been written. Don't call mz_deflate() on the stream anymore. */
/*   MZ_STREAM_ERROR if the stream is bogus. */
/*   MZ_PARAM_ERROR if one of the parameters is invalid. */
/*   MZ_BUF_ERROR if no forward progress is possible because the input and/or output buffers are empty. (Fill up the input buffer or free up some output space and try again.) */
int mz_deflate(mz_streamp pStream, int flush);

/* mz_deflateEnd() deinitializes a compressor: */
/* Return values: */
/*  MZ_OK on success. */
/*  MZ_STREAM_ERROR if the stream is bogus. */
int mz_deflateEnd(mz_streamp pStream);

/* mz_deflateBound() returns a (very) conservative upper bound on the amount of data that could be generated by deflate(), assuming flush is set to only MZ_NO_FLUSH or MZ_FINISH. */
mz_ulong mz_deflateBound(mz_streamp pStream, mz_ulong source_len);

/* Single-call compression functions mz_compress() and mz_compress2(): */
/* Returns MZ_OK on success, or one of the error codes from mz_deflate() on failure. */
int mz_compress(unsigned char *pDest, mz_ulong *pDest_len, const unsigned char *pSource, mz_ulong source_len);
int mz_compress2(unsigned char *pDest, mz_ulong *pDest_len, const unsigned char *pSource, mz_ulong source_len, int level);

/* mz_compressBound() returns a (very) conservative upper bound on the amount of data that could be generated by calling mz_compress(). */
mz_ulong mz_compressBound(mz_ulong source_len);

/* Initializes a decompressor. */
int mz_inflateInit(mz_streamp pStream);

/* mz_inflateInit2() is like mz_inflateInit() with an additional option that controls the window size and whether or not the stream has been wrapped with a zlib header/footer: */
/* window_bits must be MZ_DEFAULT_WINDOW_BITS (to parse zlib header/footer) or -MZ_DEFAULT_WINDOW_BITS (raw deflate). */
int mz_inflateInit2(mz_streamp pStream, int window_bits);

/* Quickly resets a compressor without having to reallocate anything. Same as calling mz_inflateEnd() followed by mz_inflateInit()/mz_inflateInit2(). */
int mz_inflateReset(mz_streamp pStream);

/* Decompresses the input stream to the output, consuming only as much of the input as needed, and writing as much to the output as possible. */
/* Parameters: */
/*   pStream is the stream to read from and write to. You must initialize/update the next_in, avail_in, next_out, and avail_out members. */
/*   flush may be MZ_NO_FLUSH, MZ_SYNC_FLUSH, or MZ_FINISH. */
/*   On the first call, if flush is MZ_FINISH it's assumed the input and output buffers are both sized large enough to decompress the entire stream in a single call (this is slightly faster). */
/* 
   函数 mz_inflate() 执行解压缩操作，根据参数 pStream 提供的数据流进行解压缩。
   参数 flush 控制解压缩的行为，具体取值含义请参考函数文档。
   返回值：
     MZ_OK 表示成功，可能需要更多输入或者输出缓冲区已满。
     MZ_STREAM_END 表示所有输入已消耗完毕并且所有输出数据已写入。对于 zlib 流，还验证了解压缩数据的 adler-32 校验和。
     MZ_STREAM_ERROR 表示数据流不正确。
     MZ_DATA_ERROR 表示解压缩流无效。
     MZ_PARAM_ERROR 表示参数无效。
     MZ_BUF_ERROR 表示因为输入缓冲区为空但需要更多输入来继续，或者输出缓冲区不足以容纳输出数据，需要再次调用 mz_inflate()。
*/
int mz_inflate(mz_streamp pStream, int flush);

/* 
   函数 mz_inflateEnd() 用于结束解压缩器的使用，释放相关资源。
*/
int mz_inflateEnd(mz_streamp pStream);

/* 
   函数 mz_uncompress() 实现单次调用的解压缩操作。
   参数 pDest 指向解压缩后数据存放的缓冲区。
   参数 pDest_len 指向输出缓冲区的长度，成功时返回解压后数据的实际长度。
   参数 pSource 指向待解压缩数据的起始地址。
   参数 source_len 待解压缩数据的长度。
   返回值：
     MZ_OK 表示成功。
     其他返回值表示解压缩失败，具体错误代码参考 mz_inflate() 的返回值。
*/
int mz_uncompress(unsigned char *pDest, mz_ulong *pDest_len, const unsigned char *pSource, mz_ulong source_len);

/* 
   函数 mz_error() 根据给定的错误码返回对应的错误描述字符串。
   如果错误码无效，则返回 NULL。
*/
const char *mz_error(int err);

/* 
   以下是一系列宏定义，将 zlib 兼容的名称重新定义为 miniz 的等效名称。
   这样可以在使用 miniz 的项目中直接替换掉 zlib 的部分功能。
*/
#ifndef MINIZ_NO_ZLIB_COMPATIBLE_NAMES
typedef unsigned char Byte;
typedef unsigned int uInt;
typedef mz_ulong uLong;
typedef Byte Bytef;
typedef uInt uIntf;
typedef char charf;
typedef int intf;
typedef void *voidpf;
typedef uLong uLongf;
typedef void *voidp;
typedef void *const voidpc;
#define Z_NULL 0
#define Z_NO_FLUSH MZ_NO_FLUSH
#define Z_PARTIAL_FLUSH MZ_PARTIAL_FLUSH
#define Z_SYNC_FLUSH MZ_SYNC_FLUSH
#define Z_FULL_FLUSH MZ_FULL_FLUSH
#define Z_FINISH MZ_FINISH
#define Z_BLOCK MZ_BLOCK
#define Z_OK MZ_OK
#define Z_STREAM_END MZ_STREAM_END
#define Z_NEED_DICT MZ_NEED_DICT
#define Z_ERRNO MZ_ERRNO
#define Z_STREAM_ERROR MZ_STREAM_ERROR
#define Z_DATA_ERROR MZ_DATA_ERROR
#define Z_MEM_ERROR MZ_MEM_ERROR
#define Z_BUF_ERROR MZ_BUF_ERROR
#define Z_VERSION_ERROR MZ_VERSION_ERROR
#define Z_PARAM_ERROR MZ_PARAM_ERROR
#define Z_NO_COMPRESSION MZ_NO_COMPRESSION
#define Z_BEST_SPEED MZ_BEST_SPEED
#define Z_BEST_COMPRESSION MZ_BEST_COMPRESSION
#define Z_DEFAULT_COMPRESSION MZ_DEFAULT_COMPRESSION
#define Z_DEFAULT_STRATEGY MZ_DEFAULT_STRATEGY
#define Z_FILTERED MZ_FILTERED
#define Z_HUFFMAN_ONLY MZ_HUFFMAN_ONLY
#define Z_RLE MZ_RLE
#define Z_FIXED MZ_FIXED
#endif
#define Z_DEFLATED MZ_DEFLATED
#define Z_DEFAULT_WINDOW_BITS MZ_DEFAULT_WINDOW_BITS
#define alloc_func mz_alloc_func
#define free_func mz_free_func
#define internal_state mz_internal_state
#define z_stream mz_stream
#define deflateInit mz_deflateInit
#define deflateInit2 mz_deflateInit2
#define deflateReset mz_deflateReset
#define deflate mz_deflate
#define deflateEnd mz_deflateEnd
#define deflateBound mz_deflateBound
#define compress mz_compress
#define compress2 mz_compress2
#define compressBound mz_compressBound
#define inflateInit mz_inflateInit
#define inflateInit2 mz_inflateInit2
#define inflateReset mz_inflateReset
#define inflate mz_inflate
#define inflateEnd mz_inflateEnd
#define uncompress mz_uncompress
#define crc32 mz_crc32
#define adler32 mz_adler32
#define MAX_WBITS 15
#define MAX_MEM_LEVEL 9
#define zError mz_error
#define ZLIB_VERSION MZ_VERSION
#define ZLIB_VERNUM MZ_VERNUM
#define ZLIB_VER_MAJOR MZ_VER_MAJOR
#define ZLIB_VER_MINOR MZ_VER_MINOR
#define ZLIB_VER_REVISION MZ_VER_REVISION
#define ZLIB_VER_SUBREVISION MZ_VER_SUBREVISION
#define zlibVersion mz_version
#define zlib_version mz_version()
#endif /* #ifndef MINIZ_NO_ZLIB_COMPATIBLE_NAMES */

#endif /* MINIZ_NO_ZLIB_APIS */

#ifdef __cplusplus
}
#endif
#pragma once
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ------------------- Types and macros */

// 定义无符号8位整数类型
typedef unsigned char mz_uint8;
// 定义有符号16位整数类型
typedef signed short mz_int16;
// 定义无符号16位整数类型
typedef unsigned short mz_uint16;
// 定义无符号32位整数类型
typedef unsigned int mz_uint32;
// 定义无符号整数类型
typedef unsigned int mz_uint;
// 定义64位有符号整数类型
typedef int64_t mz_int64;
// 定义64位无符号整数类型
typedef uint64_t mz_uint64;
// 定义布尔类型
typedef int mz_bool;

// 定义常量表示假
#define MZ_FALSE (0)
// 定义常量表示真
#define MZ_TRUE (1)

/* Works around MSVC's spammy "warning C4127: conditional expression is constant" message. */
// 在 MSVC 中解决 "warning C4127: conditional expression is constant" 的警告
#ifdef _MSC_VER
#define MZ_MACRO_END while (0, 0)
#else
#define MZ_MACRO_END while (0)
#endif

#ifdef MINIZ_NO_STDIO
// 如果禁用了标准 IO，则定义文件类型为 void 指针
#define MZ_FILE void *
#else
#include <stdio.h>
// 否则定义文件类型为标准文件类型 FILE
#define MZ_FILE FILE
#endif /* #ifdef MINIZ_NO_STDIO */

#ifdef MINIZ_NO_TIME
// 如果禁用了时间相关功能，则定义一个虚拟的时间结构体
typedef struct mz_dummy_time_t_tag
{
    int m_dummy;
} mz_dummy_time_t;
// 定义时间类型为虚拟时间结构体
#define MZ_TIME_T mz_dummy_time_t
#else
// 否则直接使用系统的 time_t 类型
#define MZ_TIME_T time_t
#endif

// 断言宏，用于验证条件 x 是否为真
#define MZ_ASSERT(x) assert(x)

#ifdef MINIZ_NO_MALLOC
// 如果禁用了动态内存分配，则定义一系列内存操作宏为无效操作
#define MZ_MALLOC(x) NULL
#define MZ_FREE(x) (void)x, ((void)0)
#define MZ_REALLOC(p, x) NULL
#else
// 否则定义一系列内存操作宏为对应的标准库函数
#define MZ_MALLOC(x) malloc(x)
#define MZ_FREE(x) free(x)
#define MZ_REALLOC(p, x) realloc(p, x)
#endif

// 定义求最大值和最小值的宏
#define MZ_MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MZ_MIN(a, b) (((a) < (b)) ? (a) : (b))
// 清空对象的宏，使用 memset 将对象的内存空间清零
#define MZ_CLEAR_OBJ(obj) memset(&(obj), 0, sizeof(obj))

#if MINIZ_USE_UNALIGNED_LOADS_AND_STORES && MINIZ_LITTLE_ENDIAN
// 如果支持非对齐读写并且是小端序系统，则定义读取小端16位和32位整数的宏
#define MZ_READ_LE16(p) *((const mz_uint16 *)(p))
#define MZ_READ_LE32(p) *((const mz_uint32 *)(p))
#else
// 否则定义读取小端16位整数的宏
#define MZ_READ_LE16(p) ((mz_uint32)(((const mz_uint8 *)(p))[0]) | ((mz_uint32)(((const mz_uint8 *)(p))[1]) << 8U))
#define MZ_READ_LE32(p) ((mz_uint32)(((const mz_uint8 *)(p))[0]) | ((mz_uint32)(((const mz_uint8 *)(p))[1]) << 8U) | ((mz_uint32)(((const mz_uint8 *)(p))[2]) << 16U) | ((mz_uint32)(((const mz_uint8 *)(p))[3]) << 24U))
#endif



#define MZ_READ_LE64(p) (((mz_uint64)MZ_READ_LE32(p)) | (((mz_uint64)MZ_READ_LE32((const mz_uint8 *)(p) + sizeof(mz_uint32))) << 32U))



#ifdef _MSC_VER
#define MZ_FORCEINLINE __forceinline
#elif defined(__GNUC__)
#define MZ_FORCEINLINE __inline__ __attribute__((__always_inline__))
#else
#define MZ_FORCEINLINE inline
#endif



#ifdef __cplusplus
extern "C" {
#endif



extern void *miniz_def_alloc_func(void *opaque, size_t items, size_t size);
extern void miniz_def_free_func(void *opaque, void *address);
extern void *miniz_def_realloc_func(void *opaque, void *address, size_t items, size_t size);



#define MZ_UINT16_MAX (0xFFFFU)
#define MZ_UINT32_MAX (0xFFFFFFFFU)



#ifdef __cplusplus
}
#endif
#pragma once



#ifdef __cplusplus
extern "C" {
#endif



/* ------------------- Low-level Compression API Definitions */

/* Set TDEFL_LESS_MEMORY to 1 to use less memory (compression will be slightly slower, and raw/dynamic blocks will be output more frequently). */
#define TDEFL_LESS_MEMORY 0



/* tdefl_init() compression flags logically OR'd together (low 12 bits contain the max. number of probes per dictionary search): */
/* TDEFL_DEFAULT_MAX_PROBES: The compressor defaults to 128 dictionary probes per dictionary search. 0=Huffman only, 1=Huffman+LZ (fastest/crap compression), 4095=Huffman+LZ (slowest/best compression). */
enum
{
    TDEFL_HUFFMAN_ONLY = 0,
    TDEFL_DEFAULT_MAX_PROBES = 128,
    TDEFL_MAX_PROBES_MASK = 0xFFF
};



/* TDEFL_WRITE_ZLIB_HEADER: If set, the compressor outputs a zlib header before the deflate data, and the Adler-32 of the source data at the end. Otherwise, you'll get raw deflate data. */
/* TDEFL_COMPUTE_ADLER32: Always compute the adler-32 of the input data (even when not writing zlib headers). */
/* TDEFL_GREEDY_PARSING_FLAG: Set to use faster greedy parsing, instead of more efficient lazy parsing. */
/* TDEFL_NONDETERMINISTIC_PARSING_FLAG: Enable to decrease the compressor's initialization time to the minimum, but the output may vary from run to run given the same input (depending on the contents of memory). */
/* TDEFL_RLE_MATCHES: Only look for RLE matches (matches with a distance of 1) */
/* TDEFL_FILTER_MATCHES: Discards matches <= 5 chars if enabled. */
/* TDEFL_FORCE_ALL_STATIC_BLOCKS: Disable usage of optimized Huffman tables. */
/* TDEFL_FORCE_ALL_RAW_BLOCKS: Only use raw (uncompressed) deflate blocks. */
/* The low 12 bits are reserved to control the max # of hash probes per dictionary lookup (see TDEFL_MAX_PROBES_MASK). */
enum
{
    TDEFL_WRITE_ZLIB_HEADER = 0x01000,
    TDEFL_COMPUTE_ADLER32 = 0x02000,
    TDEFL_GREEDY_PARSING_FLAG = 0x04000,
    TDEFL_NONDETERMINISTIC_PARSING_FLAG = 0x08000,
    TDEFL_RLE_MATCHES = 0x10000,
    TDEFL_FILTER_MATCHES = 0x20000,
    # 定义常量 TDEFL_FORCE_ALL_STATIC_BLOCKS，其值为 0x40000
    TDEFL_FORCE_ALL_STATIC_BLOCKS = 0x40000,
    # 定义常量 TDEFL_FORCE_ALL_RAW_BLOCKS，其值为 0x80000
    TDEFL_FORCE_ALL_RAW_BLOCKS = 0x80000
/* 结构体，用于接收用于写入缓冲区的函数指针和用户数据指针 */
/* Output stream interface. The compressor uses this interface to write compressed data. It'll typically be called TDEFL_OUT_BUF_SIZE at a time. */
typedef mz_bool (*tdefl_put_buf_func_ptr)(const void *pBuf, int len, void *pUser);

/* 将内存中的数据块压缩到通过malloc()分配的堆块中 */
/* tdefl_compress_mem_to_heap() compresses a block in memory to a heap block allocated via malloc(). */
/* On entry: */
/*  pSrc_buf, src_buf_len: Pointer and size of source block to compress. */
/*  flags: The max match finder probes (default is 128) logically OR'd against the above flags. Higher probes are slower but improve compression. */
/* On return: */
/*  Function returns a pointer to the compressed data, or NULL on failure. */
/*  *pOut_len will be set to the compressed data's size, which could be larger than src_buf_len on uncompressible data. */
/*  The caller must free() the returned block when it's no longer needed. */
void *tdefl_compress_mem_to_heap(const void *pSrc_buf, size_t src_buf_len, size_t *pOut_len, int flags);

/* 将内存中的数据块压缩到另一个内存块中 */
/* tdefl_compress_mem_to_mem() compresses a block in memory to another block in memory. */
/* Returns 0 on failure. */
size_t tdefl_compress_mem_to_mem(void *pOut_buf, size_t out_buf_len, const void *pSrc_buf, size_t src_buf_len, int flags);

/* 将图像压缩为内存中的压缩PNG文件 */
/* Compresses an image to a compressed PNG file in memory. */
/* On entry: */
/*  pImage, w, h, and num_chans describe the image to compress. num_chans may be 1, 2, 3, or 4. */
/*  The image pitch in bytes per scanline will be w*num_chans. The leftmost pixel on the top scanline is stored first in memory. */
/*  level may range from [0,10], use MZ_NO_COMPRESSION, MZ_BEST_SPEED, MZ_BEST_COMPRESSION, etc. or a decent default is MZ_DEFAULT_LEVEL */
/*  If flip is true, the image will be flipped on the Y axis (useful for OpenGL apps). */
/* On return: */
/*  Function returns a pointer to the compressed data, or NULL on failure. */
/*  *pLen_out will be set to the size of the PNG image file. */
/*  The caller must mz_free() the returned heap block (which will typically be larger than *pLen_out) when it's no longer needed. */
void *tdefl_write_image_to_png_file_in_memory_ex(const void *pImage, int w, int h, int num_chans, size_t *pLen_out, mz_uint level, mz_bool flip);
void *tdefl_write_image_to_png_file_in_memory(const void *pImage, int w, int h, int num_chans, size_t *pLen_out);

/* 将数据块压缩到输出流 */
/* tdefl_compress_mem_to_output() compresses a block to an output stream. The above helpers use this function internally. */
mz_bool tdefl_compress_mem_to_output(const void *pBuf, size_t buf_len, tdefl_put_buf_func_ptr pPut_buf_func, void *pPut_buf_user, int flags);

/* 定义一些压缩算法中需要用到的常量 */
/* Constants used in compression algorithms */
enum
{
    TDEFL_MAX_HUFF_TABLES = 3,
    TDEFL_MAX_HUFF_SYMBOLS_0 = 288,
    TDEFL_MAX_HUFF_SYMBOLS_1 = 32,
    TDEFL_MAX_HUFF_SYMBOLS_2 = 19,
    TDEFL_LZ_DICT_SIZE = 32768,
    TDEFL_LZ_DICT_SIZE_MASK = TDEFL_LZ_DICT_SIZE - 1,
    TDEFL_MIN_MATCH_LEN = 3,
    TDEFL_MAX_MATCH_LEN = 258
};
/* TDEFL_OUT_BUF_SIZE MUST be large enough to hold a single entire compressed output block (using static/fixed Huffman codes). */
#if TDEFL_LESS_MEMORY
enum
{
    // Size of the LZ code buffer for compression
    TDEFL_LZ_CODE_BUF_SIZE = 24 * 1024,
    // Size of the output buffer, calculated based on LZ code buffer size
    TDEFL_OUT_BUF_SIZE = (TDEFL_LZ_CODE_BUF_SIZE * 13) / 10,
    // Maximum number of Huffman symbols
    TDEFL_MAX_HUFF_SYMBOLS = 288,
    // Number of bits used for LZ hash
    TDEFL_LZ_HASH_BITS = 12,
    // Mask for hash size in level 1
    TDEFL_LEVEL1_HASH_SIZE_MASK = 4095,
    // Shift value for LZ hash
    TDEFL_LZ_HASH_SHIFT = (TDEFL_LZ_HASH_BITS + 2) / 3,
    // Size of LZ hash table
    TDEFL_LZ_HASH_SIZE = 1 << TDEFL_LZ_HASH_BITS
};
#else
enum
{
    // Size of the LZ code buffer for compression
    TDEFL_LZ_CODE_BUF_SIZE = 64 * 1024,
    // Size of the output buffer, calculated based on LZ code buffer size
    TDEFL_OUT_BUF_SIZE = (TDEFL_LZ_CODE_BUF_SIZE * 13) / 10,
    // Maximum number of Huffman symbols
    TDEFL_MAX_HUFF_SYMBOLS = 288,
    // Number of bits used for LZ hash
    TDEFL_LZ_HASH_BITS = 15,
    // Mask for hash size in level 1
    TDEFL_LEVEL1_HASH_SIZE_MASK = 4095,
    // Shift value for LZ hash
    TDEFL_LZ_HASH_SHIFT = (TDEFL_LZ_HASH_BITS + 2) / 3,
    // Size of LZ hash table
    TDEFL_LZ_HASH_SIZE = 1 << TDEFL_LZ_HASH_BITS
};
#endif

/* The low-level tdefl functions below may be used directly if the above helper functions aren't flexible enough. The low-level functions don't make any heap allocations, unlike the above helper functions. */
typedef enum {
    // Status indicating bad parameters
    TDEFL_STATUS_BAD_PARAM = -2,
    // Status indicating failure to write compressed data to buffer
    TDEFL_STATUS_PUT_BUF_FAILED = -1,
    // Status indicating successful operation
    TDEFL_STATUS_OKAY = 0,
    // Status indicating completion of compression
    TDEFL_STATUS_DONE = 1
} tdefl_status;

/* Must map to MZ_NO_FLUSH, MZ_SYNC_FLUSH, etc. enums */
typedef enum {
    // No flush mode
    TDEFL_NO_FLUSH = 0,
    // Sync flush mode
    TDEFL_SYNC_FLUSH = 2,
    // Full flush mode
    TDEFL_FULL_FLUSH = 3,
    // Finish compression
    TDEFL_FINISH = 4
} tdefl_flush;

/* tdefl's compression state structure. */
typedef struct
{
    // Pointer to user-defined function for putting compressed data
    tdefl_put_buf_func_ptr m_pPut_buf_func;
    // User-defined pointer passed to m_pPut_buf_func
    void *m_pPut_buf_user;
    // Flags for compressor behavior
    mz_uint m_flags, m_max_probes[2];
    // Flag for greedy parsing
    int m_greedy_parsing;
    // Adler32 checksum
    mz_uint m_adler32, m_lookahead_pos, m_lookahead_size, m_dict_size;
    // Pointers and sizes for internal LZ and output buffers
    mz_uint8 *m_pLZ_code_buf, *m_pLZ_flags, *m_pOutput_buf, *m_pOutput_buf_end;
    // Remaining flags in the buffer
    mz_uint m_num_flags_left, m_total_lz_bytes, m_lz_code_buf_dict_pos, m_bits_in, m_bit_buffer;
    // Saved match distances, lengths, and literals
    mz_uint m_saved_match_dist, m_saved_match_len, m_saved_lit, m_output_flush_ofs, m_output_flush_remaining, m_finished, m_block_index, m_wants_to_finish;
    // Previous return status
    tdefl_status m_prev_return_status;
    // Input buffer pointer
    const void *m_pIn_buf;
    // Output buffer pointer
    void *m_pOut_buf;
    // Sizes of input and output buffers
    size_t *m_pIn_buf_size, *m_pOut_buf_size;
    // Flush mode
    tdefl_flush m_flush;
    // Pointer to source buffer
    const mz_uint8 *m_pSrc;
    // Remaining size of source buffer
    size_t m_src_buf_left, m_out_buf_ofs;
    // LZ dictionary and hash tables
    mz_uint8 m_dict[TDEFL_LZ_DICT_SIZE + TDEFL_MAX_MATCH_LEN - 1];
    mz_uint16 m_huff_count[TDEFL_MAX_HUFF_TABLES][TDEFL_MAX_HUFF_SYMBOLS];
    mz_uint16 m_huff_codes[TDEFL_MAX_HUFF_TABLES][TDEFL_MAX_HUFF_SYMBOLS];
    mz_uint8 m_huff_code_sizes[TDEFL_MAX_HUFF_TABLES][TDEFL_MAX_HUFF_SYMBOLS];
    mz_uint8 m_lz_code_buf[TDEFL_LZ_CODE_BUF_SIZE];
    mz_uint16 m_next[TDEFL_LZ_DICT_SIZE];
    mz_uint16 m_hash[TDEFL_LZ_HASH_SIZE];
    mz_uint8 m_output_buf[TDEFL_OUT_BUF_SIZE];
} tdefl_compressor;

/* Initializes the compressor. */
/* There is no corresponding deinit() function because the tdefl API's do not dynamically allocate memory. */
/* pBut_buf_func: If NULL, output data will be supplied to the specified callback. In this case, the user should call the tdefl_compress_buffer() API for compression. */
/* If pBut_buf_func is NULL the user should always call the tdefl_compress() API. */
/* 如果 pBut_buf_func 为 NULL，则用户应始终调用 tdefl_compress() API。 */
tdefl_status tdefl_init(tdefl_compressor *d, tdefl_put_buf_func_ptr pPut_buf_func, void *pPut_buf_user, int flags);

/* Compresses a block of data, consuming as much of the specified input buffer as possible, and writing as much compressed data to the specified output buffer as possible. */
/* 压缩数据块，尽可能消耗指定的输入缓冲区，并尽可能多地将压缩数据写入指定的输出缓冲区。 */
tdefl_status tdefl_compress(tdefl_compressor *d, const void *pIn_buf, size_t *pIn_buf_size, void *pOut_buf, size_t *pOut_buf_size, tdefl_flush flush);

/* tdefl_compress_buffer() is only usable when the tdefl_init() is called with a non-NULL tdefl_put_buf_func_ptr. */
/* 当 tdefl_init() 被非空的 tdefl_put_buf_func_ptr 调用时，才能使用 tdefl_compress_buffer()。 */
/* tdefl_compress_buffer() 总是消耗整个输入缓冲区。 */
tdefl_status tdefl_compress_buffer(tdefl_compressor *d, const void *pIn_buf, size_t in_buf_size, tdefl_flush flush);

tdefl_status tdefl_get_prev_return_status(tdefl_compressor *d);
mz_uint32 tdefl_get_adler32(tdefl_compressor *d);

/* Create tdefl_compress() flags given zlib-style compression parameters. */
/* 根据 zlib 风格的压缩参数创建 tdefl_compress() 标志。 */
/* level 可以范围在 [0,10] 之间（其中 10 是最大压缩，但在某些文件上可能更慢） */
/* window_bits 可以是 -15（原始 deflate）或 15（zlib） */
/* strategy 可以是 MZ_DEFAULT_STRATEGY, MZ_FILTERED, MZ_HUFFMAN_ONLY, MZ_RLE 或 MZ_FIXED */
mz_uint tdefl_create_comp_flags_from_zip_params(int level, int window_bits, int strategy);

#ifndef MINIZ_NO_MALLOC
/* Allocate the tdefl_compressor structure in C so that */
/* non-C language bindings to tdefl_ API don't need to worry about */
/* structure size and allocation mechanism. */
/* 在 C 中分配 tdefl_compressor 结构，使得非 C 语言绑定到 tdefl_ API 不需要担心结构大小和分配机制。 */
tdefl_compressor *tdefl_compressor_alloc(void);
void tdefl_compressor_free(tdefl_compressor *pComp);
#endif

#ifdef __cplusplus
}
#endif
#pragma once

/* ------------------- Low-level Decompression API Definitions */

#ifdef __cplusplus
extern "C" {
#endif
/* Decompression flags used by tinfl_decompress(). */
/* tinfl_decompress() 使用的解压标志。 */
/* TINFL_FLAG_PARSE_ZLIB_HEADER: 如果设置，则输入具有有效的 zlib 头，并以 adler32 校验和结束（它是一个有效的 zlib 流）。否则，输入是原始 deflate 流。 */
/* TINFL_FLAG_HAS_MORE_INPUT: 如果设置，则提供的输入缓冲区之外还有更多的输入字节。如果清除，则输入缓冲区包含所有剩余的输入。 */
/* TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF: 如果设置，则输出缓冲区足够大以容纳整个解压缩流。如果清除，则输出缓冲区至少是字典的大小（通常为 32KB）。 */
/* TINFL_FLAG_COMPUTE_ADLER32: 强制计算解压缩字节的 adler-32 校验和。 */
enum
{
    TINFL_FLAG_PARSE_ZLIB_HEADER = 1,
    TINFL_FLAG_HAS_MORE_INPUT = 2,
    TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF = 4,
    TINFL_FLAG_COMPUTE_ADLER32 = 8
};

/* High level decompression functions: */
/* 高级解压缩函数： */
/* tinfl_decompress_mem_to_heap() 将内存中的块解压缩到通过 malloc() 分配的堆块中。 */
/* 进入时： */
/* tinfl_decompress_mem_to_heap()函数：将Deflate或zlib格式的源数据解压缩到堆内存中。*/
/* 参数说明：
   pSrc_buf: 指向要解压缩的Deflate或zlib源数据的指针。
   src_buf_len: 源数据的大小。
   pOut_len: 解压缩后数据的大小将被设置到这个指针指向的位置，可能比src_buf_len大，特别是在无法压缩的数据上。
   返回值：
   函数返回指向解压缩后数据的指针，失败时返回NULL。
   调用者在不再需要返回的数据时必须调用mz_free()来释放返回的内存块。 */
void *tinfl_decompress_mem_to_heap(const void *pSrc_buf, size_t src_buf_len, size_t *pOut_len, int flags);

/* tinfl_decompress_mem_to_mem()函数：将内存中的一个数据块解压缩到另一个内存块中。*/
/* 返回值：
   在成功时返回写入的字节数，失败时返回TINFL_DECOMPRESS_MEM_TO_MEM_FAILED。 */
#define TINFL_DECOMPRESS_MEM_TO_MEM_FAILED ((size_t)(-1))
size_t tinfl_decompress_mem_to_mem(void *pOut_buf, size_t out_buf_len, const void *pSrc_buf, size_t src_buf_len, int flags);

/* tinfl_decompress_mem_to_callback()函数：将内存中的一个数据块解压缩到一个内部32KB缓冲区中，并调用用户提供的回调函数来刷新缓冲区。*/
/* 返回值：
   成功时返回1，失败时返回0。 */
typedef int (*tinfl_put_buf_func_ptr)(const void *pBuf, int len, void *pUser);
int tinfl_decompress_mem_to_callback(const void *pIn_buf, size_t *pIn_buf_size, tinfl_put_buf_func_ptr pPut_buf_func, void *pPut_buf_user, int flags);

struct tinfl_decompressor_tag;
typedef struct tinfl_decompressor_tag tinfl_decompressor;

#ifndef MINIZ_NO_MALLOC
/* tinfl_decompressor_alloc()函数：在C中分配tinfl_decompressor结构体。*/
/* 非C语言绑定到tinfl_ API时，无需担心结构体大小和分配机制。 */
tinfl_decompressor *tinfl_decompressor_alloc(void);

/* tinfl_decompressor_free()函数：释放通过tinfl_decompressor_alloc()分配的内存。*/
void tinfl_decompressor_free(tinfl_decompressor *pDecomp);
#endif

/* TINFL_LZ_DICT_SIZE：LZ字典的最大大小。 */
#define TINFL_LZ_DICT_SIZE 32768

/* 返回状态枚举类型。 */
typedef enum {
    /* TINFL_STATUS_FAILED_CANNOT_MAKE_PROGRESS：解压器需要1个或多个输入字节来取得进展，但调用者指示没有更多可用的字节。 */
    /* 这可能表示压缩数据已损坏。如果再次调用解压器并提供更多字节，则会尝试继续处理输入，但这是一个不好的迹象（数据可能损坏或调用方式不正确）。 */
    /* 如果再次无输入调用，则会再次返回TINFL_STATUS_FAILED_CANNOT_MAKE_PROGRESS。 */
    TINFL_STATUS_FAILED_CANNOT_MAKE_PROGRESS = -4,

    /* TINFL_STATUS_BAD_PARAM：其中一个输入参数明显错误。（可以尝试再次调用，但如果出现此错误，则调用代码错误。） */
    TINFL_STATUS_BAD_PARAM = -3,

    /* TINFL_STATUS_ADLER32_MISMATCH：解压器已完成，但未能通过未压缩数据的adler32校验。 */
    /* 如果再次调用，将返回TINFL_STATUS_DONE。 */
    TINFL_STATUS_ADLER32_MISMATCH = -2,
    /* 这些标志表明解压器在某种情况下失败了（如错误的代码、损坏的输入等）。如果不通过 tinfl_init() 重置它，再次调用它将会继续返回相同的失败状态码。 */
    TINFL_STATUS_FAILED = -1,

    /* 任何小于 TINFL_STATUS_DONE 的状态码必须表示失败。 */

    /* 这个标志表明解压器已经返回了它可以返回的所有未压缩数据字节，已经消耗了它需要的每个字节，成功地到达了 deflate 流的末尾，并且如果启用了 zlib 头和 adler32 校验，则成功地检查了未压缩数据的 adler32。如果再次调用它，你将只会得到 TINFL_STATUS_DONE。 */
    TINFL_STATUS_DONE = 0,

    /* 这个标志表明解压器必须有更多的输入数据（甚至 1 个字节），才能继续前进，或者在下一次调用时需要清除 TINFL_FLAG_HAS_MORE_INPUT 标志，如果没有更多的源数据可用的话。如果源数据出现某种方式的损坏，解压器可能会继续要求输入以便继续进行，所以一定要正确设置 TINFL_FLAG_HAS_MORE_INPUT 标志。 */
    TINFL_STATUS_NEEDS_MORE_INPUT = 1,

    /* 这个标志表明解压器肯定有 1 个或更多字节的未压缩数据可用，但它无法将这些数据写入输出缓冲区。 */
    /* 注意，如果源压缩数据被损坏，解压器可能会向调用者返回大量未压缩数据。我假设您知道期望的未压缩数据量（精确或最坏情况），并在接收到过多数据后停止调用解压器并失败。在纯流式处理场景中，如果您不知道期望的字节数，可能就无法这样操作，因此可能需要添加一些代码来处理这种情况。 */
    TINFL_STATUS_HAS_MORE_OUTPUT = 2
/* tinfl_status 是一个枚举类型，用于表示解压状态。 */
} tinfl_status;

/* tinfl_init 宏定义，用于初始化解压器到初始状态。 */
#define tinfl_init(r)     \
    do                    \
    {                     \
        (r)->m_state = 0; \  // 将解压器状态置为0
    }                     \
    MZ_MACRO_END

/* tinfl_get_adler32 宏定义，用于获取 Adler-32 校验和。 */
#define tinfl_get_adler32(r) (r)->m_check_adler32  // 返回解压器中的 Adler-32 校验和

/* tinfl_decompress 函数声明，用于执行低级别的解压缩操作。 */
/* 此函数是解压缩所需的唯一函数。其他函数只是为了提高易用性而提供的高级别辅助功能。 */
/* 这是一个通用的 API，可以用作构建任何所需高级解压缩 API 的基本构件。在极限情况下，可以每输入或输出一个字节调用一次。 */
tinfl_status tinfl_decompress(tinfl_decompressor *r, const mz_uint8 *pIn_buf_next, size_t *pIn_buf_size, mz_uint8 *pOut_buf_start, mz_uint8 *pOut_buf_next, size_t *pOut_buf_size, const mz_uint32 decomp_flags);

/* 下面是内部/私有部分的定义和声明。 */

/* 枚举常量定义，用于指定 Huffman 表的最大数量和每个表的最大符号数量。 */
enum
{
    TINFL_MAX_HUFF_TABLES = 3,
    TINFL_MAX_HUFF_SYMBOLS_0 = 288,
    TINFL_MAX_HUFF_SYMBOLS_1 = 32,
    TINFL_MAX_HUFF_SYMBOLS_2 = 19,
    TINFL_FAST_LOOKUP_BITS = 10,
    TINFL_FAST_LOOKUP_SIZE = 1 << TINFL_FAST_LOOKUP_BITS
};

/* 定义了一个结构体 tinfl_huff_table，用于存储 Huffman 表的相关信息。 */
typedef struct
{
    mz_uint8 m_code_size[TINFL_MAX_HUFF_SYMBOLS_0];
    mz_int16 m_look_up[TINFL_FAST_LOOKUP_SIZE], m_tree[TINFL_MAX_HUFF_SYMBOLS_0 * 2];
} tinfl_huff_table;

/* 根据是否支持 64 位寄存器来定义使用 64 位位缓冲的条件编译宏和相关类型。 */
#if MINIZ_HAS_64BIT_REGISTERS
#define TINFL_USE_64BIT_BITBUF 1
#else
#define TINFL_USE_64BIT_BITBUF 0
#endif

#if TINFL_USE_64BIT_BITBUF
typedef mz_uint64 tinfl_bit_buf_t;  // 定义使用 64 位位缓冲时的位缓冲类型
#define TINFL_BITBUF_SIZE (64)
#else
typedef mz_uint32 tinfl_bit_buf_t;  // 定义使用 32 位位缓冲时的位缓冲类型
#define TINFL_BITBUF_SIZE (32)
#endif

/* tinfl_decompressor_tag 结构体定义，包含了解压器的各种状态和数据。 */
struct tinfl_decompressor_tag
{
    mz_uint32 m_state, m_num_bits, m_zhdr0, m_zhdr1, m_z_adler32, m_final, m_type, m_check_adler32, m_dist, m_counter, m_num_extra, m_table_sizes[TINFL_MAX_HUFF_TABLES];
    tinfl_bit_buf_t m_bit_buf;
    size_t m_dist_from_out_buf_start;
    tinfl_huff_table m_tables[TINFL_MAX_HUFF_TABLES];
    mz_uint8 m_raw_header[4], m_len_codes[TINFL_MAX_HUFF_SYMBOLS_0 + TINFL_MAX_HUFF_SYMBOLS_1 + 137];
};

#ifdef __cplusplus
}
#endif

#pragma once

/* ------------------- ZIP archive reading/writing */

#ifndef MINIZ_NO_ARCHIVE_APIS

#ifdef __cplusplus
extern "C" {
#endif

/* 枚举常量定义，用于 ZIP 存档读写操作。 */
enum
{
    /* 这些枚举常量可以根据需要减少以节省内存或堆栈空间 - 它们相对保守。 */
    MZ_ZIP_MAX_IO_BUF_SIZE = 64 * 1024,
    MZ_ZIP_MAX_ARCHIVE_FILENAME_SIZE = 512,
    MZ_ZIP_MAX_ARCHIVE_FILE_COMMENT_SIZE = 512
};

/* 定义了一个结构体，用于表示 ZIP 存档中的单个文件条目。 */
typedef struct
{
    /* 中央目录文件索引。 */
    mz_uint32 m_file_index;

    /* 该条目在存档中央目录中的字节偏移量。注意，我们目前仅支持最多 UINT_MAX 或更少字节的中央目录。 */
    mz_uint64 m_central_dir_ofs;

    /* 这些字段直接从 ZIP 的中央目录复制。 */
    mz_uint16 m_version_made_by;
    mz_uint16 m_version_needed;
    mz_uint16 m_bit_flag;
    mz_uint16 m_method;

#ifndef MINIZ_NO_TIME
    MZ_TIME_T m_time;  // 如果未定义 MINIZ_NO_TIME，则包含时间信息
#endif
    /* CRC-32 of uncompressed data. */
    mz_uint32 m_crc32;  // 未压缩数据的 CRC-32 校验值

    /* File's compressed size. */
    mz_uint64 m_comp_size;  // 文件的压缩大小

    /* File's uncompressed size. Note, I've seen some old archives where directory entries had 512 bytes for their uncompressed sizes, but when you try to unpack them you actually get 0 bytes. */
    mz_uint64 m_uncomp_size;  // 文件的未压缩大小。注意，某些旧的归档文件可能在目录项中标记为 512 字节的未压缩大小，但实际解包时可能得到 0 字节。

    /* Zip internal and external file attributes. */
    mz_uint16 m_internal_attr;  // ZIP 内部和外部文件属性
    mz_uint32 m_external_attr;  // ZIP 外部文件属性

    /* Entry's local header file offset in bytes. */
    mz_uint64 m_local_header_ofs;  // 条目在 ZIP 文件中的本地头偏移量（字节）

    /* Size of comment in bytes. */
    mz_uint32 m_comment_size;  // 注释的大小（字节）

    /* MZ_TRUE if the entry appears to be a directory. */
    mz_bool m_is_directory;  // 如果条目看起来是一个目录，则为 MZ_TRUE

    /* MZ_TRUE if the entry uses encryption/strong encryption (which miniz_zip doesn't support) */
    mz_bool m_is_encrypted;  // 如果条目使用加密/强加密（miniz_zip 不支持）

    /* MZ_TRUE if the file is not encrypted, a patch file, and if it uses a compression method we support. */
    mz_bool m_is_supported;  // 如果文件未加密且是一个补丁文件，并且使用我们支持的压缩方法，则为 MZ_TRUE

    /* Filename. If string ends in '/' it's a subdirectory entry. */
    /* Guaranteed to be zero terminated, may be truncated to fit. */
    char m_filename[MZ_ZIP_MAX_ARCHIVE_FILENAME_SIZE];  // 文件名。如果以 '/' 结尾，则是一个子目录条目。保证以零结尾，可能会被截断以适应大小。

    /* Comment field. */
    /* Guaranteed to be zero terminated, may be truncated to fit. */
    char m_comment[MZ_ZIP_MAX_ARCHIVE_FILE_COMMENT_SIZE];  // 注释字段。保证以零结尾，可能会被截断以适应大小。
typedef size_t (*mz_file_read_func)(void *pOpaque, mz_uint64 file_ofs, void *pBuf, size_t n);
typedef size_t (*mz_file_write_func)(void *pOpaque, mz_uint64 file_ofs, const void *pBuf, size_t n);
typedef mz_bool (*mz_file_needs_keepalive)(void *pOpaque);

struct mz_zip_internal_state_tag;
typedef struct mz_zip_internal_state_tag mz_zip_internal_state;
    // 总文件数目
    mz_uint32 m_total_files;
    // ZIP 操作模式
    mz_zip_mode m_zip_mode;
    // ZIP 类型
    mz_zip_type m_zip_type;
    // 最后的 ZIP 错误码
    mz_zip_error m_last_error;
    
    // 文件偏移对齐值
    mz_uint64 m_file_offset_alignment;
    
    // 内存分配函数指针
    mz_alloc_func m_pAlloc;
    // 内存释放函数指针
    mz_free_func m_pFree;
    // 内存重新分配函数指针
    mz_realloc_func m_pRealloc;
    // 内存分配的不透明指针
    void *m_pAlloc_opaque;
    
    // 文件读取函数指针
    mz_file_read_func m_pRead;
    // 文件写入函数指针
    mz_file_write_func m_pWrite;
    // 文件保持活动状态函数指针
    mz_file_needs_keepalive m_pNeeds_keepalive;
    // 文件 I/O 的不透明指针
    void *m_pIO_opaque;
    
    // ZIP 内部状态指针
    mz_zip_internal_state *m_pState;
/* 定义了一个名为 mz_zip_archive 的结构体 */
} mz_zip_archive;

/* 定义了一个名为 mz_zip_reader_extract_iter_state 的结构体 */
typedef struct
{
    mz_zip_archive *pZip;                   /* 指向 mz_zip_archive 结构体的指针 */
    mz_uint flags;                          /* 用于标志的无符号整数 */

    int status;                             /* 状态变量 */
#ifndef MINIZ_DISABLE_ZIP_READER_CRC32_CHECKS
    mz_uint file_crc32;                     /* 文件的 CRC32 校验和 */
#endif
    mz_uint64 read_buf_size, read_buf_ofs, read_buf_avail, comp_remaining, out_buf_ofs, cur_file_ofs;
                                            /* 多个无符号整数和无符号 64 位整数变量 */
    mz_zip_archive_file_stat file_stat;     /* 存储文件统计信息的结构体 */
    void *pRead_buf;                        /* 用于读取的缓冲区指针 */
    void *pWrite_buf;                       /* 用于写入的缓冲区指针 */

    size_t out_blk_remain;                  /* 剩余输出块大小 */

    tinfl_decompressor inflator;            /* tinfl 解压缩器结构体 */

} mz_zip_reader_extract_iter_state;

/* -------- ZIP reading */

/* 初始化 ZIP 归档的读取器 */
/* 这些函数读取并验证归档的中心目录 */
mz_bool mz_zip_reader_init(mz_zip_archive *pZip, mz_uint64 size, mz_uint flags);

mz_bool mz_zip_reader_init_mem(mz_zip_archive *pZip, const void *pMem, size_t size, mz_uint flags);

#ifndef MINIZ_NO_STDIO
/* 从磁盘文件中读取归档 */
/* file_start_ofs 是归档实际开始的文件偏移量，或者为 0 */
/* actual_archive_size 是归档的真实总大小，可能小于磁盘上文件的实际大小。如果为零，则整个文件被视为归档 */
mz_bool mz_zip_reader_init_file(mz_zip_archive *pZip, const char *pFilename, mz_uint32 flags);
mz_bool mz_zip_reader_init_file_v2(mz_zip_archive *pZip, const char *pFilename, mz_uint flags, mz_uint64 file_start_ofs, mz_uint64 archive_size);

/* 从已经打开的 FILE 中读取归档，从当前文件位置开始 */
/* 假设归档长度为 archive_size 字节。如果 archive_size < 0，则假定整个文件剩余部分包含归档 */
/* 在调用 mz_zip_reader_end() 时，不会关闭 FILE */
mz_bool mz_zip_reader_init_cfile(mz_zip_archive *pZip, MZ_FILE *pFile, mz_uint64 archive_size, mz_uint flags);
#endif

/* 结束归档读取，释放所有分配的内存，并在使用 mz_zip_reader_init_file() 时关闭输入归档文件 */
mz_bool mz_zip_reader_end(mz_zip_archive *pZip);

/* -------- ZIP reading or writing */

/* 将 mz_zip_archive 结构体清零 */
/* 重要提示：在传递结构体给任何 mz_zip 函数之前必须执行此操作 */
void mz_zip_zero_struct(mz_zip_archive *pZip);

/* 获取 mz_zip_archive 结构体的模式 */
mz_zip_mode mz_zip_get_mode(mz_zip_archive *pZip);

/* 获取 mz_zip_archive 结构体的类型 */
mz_zip_type mz_zip_get_type(mz_zip_archive *pZip);

/* 返回归档中的文件总数 */
mz_uint mz_zip_reader_get_num_files(mz_zip_archive *pZip);

/* 返回归档的总大小 */
mz_uint64 mz_zip_get_archive_size(mz_zip_archive *pZip);

/* 返回归档文件实际开始的偏移量 */
mz_uint64 mz_zip_get_archive_file_start_offset(mz_zip_archive *pZip);

/* 返回与 mz_zip_archive 关联的 MZ_FILE 指针 */
MZ_FILE *mz_zip_get_cfile(mz_zip_archive *pZip);

/* 读取归档数据 */
/* 从文件偏移量 file_ofs 处开始读取 n 字节的原始归档数据到 pBuf 中 */
size_t mz_zip_read_archive_data(mz_zip_archive *pZip, mz_uint64 file_ofs, void *pBuf, size_t n);

/* 所有 mz_zip 函数都会设置 mz_zip_archive 结构体中的 m_last_error 字段 */
/* 这些函数用于检索/操作 m_last_error 字段 */
/* 注意，m_last_error 功能不是线程安全的 */
/* Sets the last error for a given zip archive object. */
mz_zip_error mz_zip_set_last_error(mz_zip_archive *pZip, mz_zip_error err_num);

/* Peeks at the last error for a given zip archive object without clearing it. */
mz_zip_error mz_zip_peek_last_error(mz_zip_archive *pZip);

/* Clears the last error for a given zip archive object. */
mz_zip_error mz_zip_clear_last_error(mz_zip_archive *pZip);

/* Retrieves the last error code for a given zip archive object. */
mz_zip_error mz_zip_get_last_error(mz_zip_archive *pZip);

/* Retrieves a string description of a given zip error code. */
const char *mz_zip_get_error_string(mz_zip_error mz_err);

/* Checks if the specified file entry in the zip archive is a directory. */
/* Returns MZ_TRUE if it is a directory, otherwise MZ_FALSE. */
mz_bool mz_zip_reader_is_file_a_directory(mz_zip_archive *pZip, mz_uint file_index);

/* Checks if the specified file entry in the zip archive is encrypted. */
/* Returns MZ_TRUE if the file is encrypted, otherwise MZ_FALSE. */
mz_bool mz_zip_reader_is_file_encrypted(mz_zip_archive *pZip, mz_uint file_index);

/* Checks if the specified file entry in the zip archive is supported for extraction. */
/* Returns MZ_TRUE if the file's compression method is supported, not encrypted, and not a compressed patch file; otherwise MZ_FALSE. */
mz_bool mz_zip_reader_is_file_supported(mz_zip_archive *pZip, mz_uint file_index);

/* Retrieves the filename of the specified archive file entry. */
/* Returns the number of bytes written to pFilename. If filename_buf_size is 0, returns the number of bytes needed to fully store the filename. */
mz_uint mz_zip_reader_get_filename(mz_zip_archive *pZip, mz_uint file_index, char *pFilename, mz_uint filename_buf_size);

/* Attempts to locate a file in the archive's central directory. */
/* Valid flags: MZ_ZIP_FLAG_CASE_SENSITIVE, MZ_ZIP_FLAG_IGNORE_PATH */
/* Returns -1 if the file cannot be found, otherwise the index of the file. */
int mz_zip_reader_locate_file(mz_zip_archive *pZip, const char *pName, const char *pComment, mz_uint flags);
int mz_zip_reader_locate_file_v2(mz_zip_archive *pZip, const char *pName, const char *pComment, mz_uint flags, mz_uint32 *file_index);

/* Retrieves detailed information about a specific archive file entry. */
/* Returns MZ_TRUE if successful, filling out pStat with file information. */
mz_bool mz_zip_reader_file_stat(mz_zip_archive *pZip, mz_uint file_index, mz_zip_archive_file_stat *pStat);

/* Checks if the zip archive is in zip64 format. */
/* Returns MZ_TRUE if the archive is zip64, otherwise MZ_FALSE. */
mz_bool mz_zip_is_zip64(mz_zip_archive *pZip);

/* Retrieves the total size of the central directory in bytes for the zip archive. */
/* The maximum supported size is <= MZ_UINT32_MAX. */
size_t mz_zip_get_central_dir_size(mz_zip_archive *pZip);

/* Extracts a specified archive file entry into a memory buffer without allocating memory. */
/* Requires enough space on the stack to store the inflator's state (~34KB). */
mz_bool mz_zip_reader_extract_to_mem_no_alloc(mz_zip_archive *pZip, mz_uint file_index, void *pBuf, size_t buf_size, mz_uint flags, void *pUser_read_buf, size_t user_read_buf_size);

/* Extracts a specified archive file entry into a memory buffer. */
mz_bool mz_zip_reader_extract_to_mem(mz_zip_archive *pZip, mz_uint file_index, void *pBuf, size_t buf_size, mz_uint flags);

/* Extracts a specified archive file entry into a memory buffer without allocating memory. */
/* Requires enough space on the stack to store the inflator's state (~34KB). */
mz_bool mz_zip_reader_extract_file_to_mem_no_alloc(mz_zip_archive *pZip, const char *pFilename, void *pBuf, size_t buf_size, mz_uint flags, void *pUser_read_buf, size_t user_read_buf_size);
/* Extracts a specific file from a ZIP archive into a pre-allocated memory buffer. */
mz_bool mz_zip_reader_extract_file_to_mem(mz_zip_archive *pZip, const char *pFilename, void *pBuf, size_t buf_size, mz_uint flags);

/* Extracts a file from a ZIP archive to a dynamically allocated heap buffer. */
/* Memory allocation is managed by mz_zip_archive's alloc/realloc functions. */
/* Returns NULL and sets an error on failure. */
void *mz_zip_reader_extract_to_heap(mz_zip_archive *pZip, mz_uint file_index, size_t *pSize, mz_uint flags);

/* Extracts a specific file from a ZIP archive to a dynamically allocated heap buffer. */
/* Memory allocation is managed by mz_zip_archive's alloc/realloc functions. */
/* Returns NULL and sets an error on failure. */
void *mz_zip_reader_extract_file_to_heap(mz_zip_archive *pZip, const char *pFilename, size_t *pSize, mz_uint flags);

/* Extracts a specific file from a ZIP archive using a callback function to write the file's data. */
mz_bool mz_zip_reader_extract_to_callback(mz_zip_archive *pZip, mz_uint file_index, mz_file_write_func pCallback, void *pOpaque, mz_uint flags);

/* Extracts a specific file from a ZIP archive using a callback function to write the file's data. */
mz_bool mz_zip_reader_extract_file_to_callback(mz_zip_archive *pZip, const char *pFilename, mz_file_write_func pCallback, void *pOpaque, mz_uint flags);

/* Initializes an iterator for extracting a specific file from a ZIP archive. */
mz_zip_reader_extract_iter_state* mz_zip_reader_extract_iter_new(mz_zip_archive *pZip, mz_uint file_index, mz_uint flags);

/* Initializes an iterator for extracting a specific file from a ZIP archive. */
mz_zip_reader_extract_iter_state* mz_zip_reader_extract_file_iter_new(mz_zip_archive *pZip, const char *pFilename, mz_uint flags);

/* Reads data from an iterator for extracting a specific file from a ZIP archive. */
size_t mz_zip_reader_extract_iter_read(mz_zip_reader_extract_iter_state* pState, void* pvBuf, size_t buf_size);

/* Frees resources associated with an iterator for extracting a specific file from a ZIP archive. */
mz_bool mz_zip_reader_extract_iter_free(mz_zip_reader_extract_iter_state* pState);

#ifndef MINIZ_NO_STDIO
/* Extracts a specific file from a ZIP archive to a disk file, setting its last accessed and modified times. */
/* Only extracts files, not archive directory records. */
mz_bool mz_zip_reader_extract_to_file(mz_zip_archive *pZip, mz_uint file_index, const char *pDst_filename, mz_uint flags);

/* Extracts a specific file from a ZIP archive to a disk file, setting its last accessed and modified times. */
mz_bool mz_zip_reader_extract_file_to_file(mz_zip_archive *pZip, const char *pArchive_filename, const char *pDst_filename, mz_uint flags);

/* Extracts a specific file from a ZIP archive to a FILE stream, starting at the current position in the stream. */
mz_bool mz_zip_reader_extract_to_cfile(mz_zip_archive *pZip, mz_uint file_index, MZ_FILE *File, mz_uint flags);

/* Extracts a specific file from a ZIP archive to a FILE stream, starting at the current position in the stream. */
mz_bool mz_zip_reader_extract_file_to_cfile(mz_zip_archive *pZip, const char *pArchive_filename, MZ_FILE *pFile, mz_uint flags);
#endif

#if 0
/* TODO: Functionality related to streaming extraction from ZIP archives. */
    typedef void *mz_zip_streaming_extract_state_ptr;
    mz_zip_streaming_extract_state_ptr mz_zip_streaming_extract_begin(mz_zip_archive *pZip, mz_uint file_index, mz_uint flags);
    uint64_t mz_zip_streaming_extract_get_size(mz_zip_archive *pZip, mz_zip_streaming_extract_state_ptr pState);
    uint64_t mz_zip_streaming_extract_get_cur_ofs(mz_zip_archive *pZip, mz_zip_streaming_extract_state_ptr pState);
    mz_bool mz_zip_streaming_extract_seek(mz_zip_archive *pZip, mz_zip_streaming_extract_state_ptr pState, uint64_t new_ofs);
    size_t mz_zip_streaming_extract_read(mz_zip_archive *pZip, mz_zip_streaming_extract_state_ptr pState, void *pBuf, size_t buf_size);
#endif
    # 调用函数 mz_zip_streaming_extract_end，用于结束流式解压过程并清理相关状态
    mz_bool mz_zip_streaming_extract_end(mz_zip_archive *pZip, mz_zip_streaming_extract_state_ptr pState);
#endif

/* This directive ends a conditional compilation block started by #ifndef. It ensures that the code following it is included for compilation only if the preceding #ifndef condition was not true. */

/* This function validates the integrity of a specific file within a ZIP archive. It compares the archive's local headers, optional local zip64 extended information block, and optional descriptor against the data in the central directory. It also verifies the file can be successfully uncompressed unless the MZ_ZIP_FLAG_VALIDATE_HEADERS_ONLY flag is specified. */
mz_bool mz_zip_validate_file(mz_zip_archive *pZip, mz_uint file_index, mz_uint flags);

/* This function validates the entire contents of a ZIP archive by calling mz_zip_validate_file() for each file within the archive. */
mz_bool mz_zip_validate_archive(mz_zip_archive *pZip, mz_uint flags);

/* These functions provide utilities and helpers valid for both ZIP reading and writing operations. */

/* Validates a ZIP archive stored in memory. pMem points to the memory block containing the archive, size specifies the size of the memory block, flags provide additional validation options, and pErr is used to return any encountered errors. */
mz_bool mz_zip_validate_mem_archive(const void *pMem, size_t size, mz_uint flags, mz_zip_error *pErr);

/* Validates a ZIP archive stored in a file. pFilename is the path to the archive file, flags provide additional validation options, and pErr is used to return any encountered errors. */
mz_bool mz_zip_validate_file_archive(const char *pFilename, mz_uint flags, mz_zip_error *pErr);

/* This function terminates the use of a ZIP archive. It determines whether to call mz_zip_reader_end() or mz_zip_writer_end() based on the type of ZIP archive object passed as pZip. */

/* -------- ZIP writing */

#ifndef MINIZ_NO_ARCHIVE_WRITING_APIS

/* This directive starts a conditional compilation block for APIs related to writing ZIP archives. It ensures that the following code is included for compilation only if MINIZ_NO_ARCHIVE_WRITING_APIS is not defined. */

/* Initializes a ZIP archive writer. pZip is a pointer to the mz_zip_archive struct representing the archive, and existing_size specifies the expected size of the archive, if known beforehand. */
mz_bool mz_zip_writer_init(mz_zip_archive *pZip, mz_uint64 existing_size);

/* Initializes a ZIP archive writer with additional flags. In addition to the parameters of mz_zip_writer_init(), flags provide further initialization options. */
mz_bool mz_zip_writer_init_v2(mz_zip_archive *pZip, mz_uint64 existing_size, mz_uint flags);

/* Initializes a ZIP archive writer that allocates memory dynamically. size_to_reserve_at_beginning specifies the initial size to reserve for the archive, and initial_allocation_size specifies the initial size allocated for writing. */
mz_bool mz_zip_writer_init_heap(mz_zip_archive *pZip, size_t size_to_reserve_at_beginning, size_t initial_allocation_size);

/* Initializes a ZIP archive writer with additional flags for heap-based allocation. */
mz_bool mz_zip_writer_init_heap_v2(mz_zip_archive *pZip, size_t size_to_reserve_at_beginning, size_t initial_allocation_size, mz_uint flags);

#ifndef MINIZ_NO_STDIO
/* Initializes a ZIP archive writer that writes to a file specified by pFilename. size_to_reserve_at_beginning specifies the initial size to reserve for the archive within the file. */
mz_bool mz_zip_writer_init_file(mz_zip_archive *pZip, const char *pFilename, mz_uint64 size_to_reserve_at_beginning);

/* Initializes a ZIP archive writer with additional flags, writing to a file specified by pFilename. */
mz_bool mz_zip_writer_init_file_v2(mz_zip_archive *pZip, const char *pFilename, mz_uint64 size_to_reserve_at_beginning, mz_uint flags);

/* Initializes a ZIP archive writer that writes to a C standard library FILE object specified by pFile. */
mz_bool mz_zip_writer_init_cfile(mz_zip_archive *pZip, MZ_FILE *pFile, mz_uint flags);
#endif

/* Converts a ZIP archive reader object into a writer object, allowing in-place appends to an existing archive. */
/* For archives opened with mz_zip_reader_init_file, pFilename must be the filename of the archive to reopen for writing. If reopening fails, mz_zip_reader_end() is called. */
/* For archives opened with mz_zip_reader_init_mem, the memory block must be resizable using the default realloc callback. */
/* For archives opened with mz_zip_reader_init, the m_pWrite function provided by the mz_zip_archive must not be NULL. */
/* Caution: Modifying an archive in-place is risky; if the process halts or errors occur before finalizing the archive, the central directory may be corrupted. */
/* 
 * 从现有的 ZIP 读取器初始化 ZIP 写入器。
 * pZip: 指向要初始化的 mz_zip_archive 结构的指针
 * pFilename: ZIP 文件名
 * 返回值: 初始化成功返回非零值，否则返回零
 */
mz_bool mz_zip_writer_init_from_reader(mz_zip_archive *pZip, const char *pFilename);

/*
 * 从现有的 ZIP 读取器初始化 ZIP 写入器，支持更多选项。
 * pZip: 指向要初始化的 mz_zip_archive 结构的指针
 * pFilename: ZIP 文件名
 * flags: 标志位，用于指定初始化选项，例如压缩级别等
 * 返回值: 初始化成功返回非零值，否则返回零
 */
mz_bool mz_zip_writer_init_from_reader_v2(mz_zip_archive *pZip, const char *pFilename, mz_uint flags);

/*
 * 将内存缓冲区的内容添加到存档中。
 * pZip: 指向 mz_zip_archive 结构的指针
 * pArchive_name: 存档中的文件名
 * pBuf: 指向要添加的数据缓冲区的指针
 * buf_size: 缓冲区的大小（字节数）
 * level_and_flags: 压缩级别和标志位，可以与 MZ_BEST_SPEED、MZ_BEST_COMPRESSION 等逻辑 OR 运算，或者仅设置为 MZ_DEFAULT_COMPRESSION
 * 返回值: 添加成功返回非零值，否则返回零
 */
mz_bool mz_zip_writer_add_mem(mz_zip_archive *pZip, const char *pArchive_name, const void *pBuf, size_t buf_size, mz_uint level_and_flags);

/*
 * 类似 mz_zip_writer_add_mem()，允许指定文件注释字段，并可选择提供已压缩数据。
 * pZip: 指向 mz_zip_archive 结构的指针
 * pArchive_name: 存档中的文件名
 * pBuf: 指向要添加的数据缓冲区的指针
 * buf_size: 缓冲区的大小（字节数）
 * pComment: 指向文件注释数据的指针
 * comment_size: 注释数据的大小（字节数）
 * level_and_flags: 压缩级别和标志位
 * uncomp_size: 未压缩数据的大小（仅当设置了 MZ_ZIP_FLAG_COMPRESSED_DATA 标志时使用）
 * uncomp_crc32: 未压缩数据的 CRC32 校验码（仅当设置了 MZ_ZIP_FLAG_COMPRESSED_DATA 标志时使用）
 * 返回值: 添加成功返回非零值，否则返回零
 */
mz_bool mz_zip_writer_add_mem_ex(mz_zip_archive *pZip, const char *pArchive_name, const void *pBuf, size_t buf_size, const void *pComment, mz_uint16 comment_size, mz_uint level_and_flags,
                                 mz_uint64 uncomp_size, mz_uint32 uncomp_crc32);

/*
 * 类似 mz_zip_writer_add_mem_ex()，支持更多选项，如记录修改时间、用户额外数据等。
 * pZip: 指向 mz_zip_archive 结构的指针
 * pArchive_name: 存档中的文件名
 * pBuf: 指向要添加的数据缓冲区的指针
 * buf_size: 缓冲区的大小（字节数）
 * pComment: 指向文件注释数据的指针
 * comment_size: 注释数据的大小（字节数）
 * level_and_flags: 压缩级别和标志位
 * uncomp_size: 未压缩数据的大小（仅当设置了 MZ_ZIP_FLAG_COMPRESSED_DATA 标志时使用）
 * uncomp_crc32: 未压缩数据的 CRC32 校验码（仅当设置了 MZ_ZIP_FLAG_COMPRESSED_DATA 标志时使用）
 * last_modified: 指向文件最后修改时间的指针
 * user_extra_data_local: 指向本地文件头部的用户额外数据
 * user_extra_data_local_len: 本地文件头部用户额外数据的长度
 * user_extra_data_central: 指向中央目录记录的用户额外数据
 * user_extra_data_central_len: 中央目录记录用户额外数据的长度
 * 返回值: 添加成功返回非零值，否则返回零
 */
mz_bool mz_zip_writer_add_mem_ex_v2(mz_zip_archive *pZip, const char *pArchive_name, const void *pBuf, size_t buf_size, const void *pComment, mz_uint16 comment_size, mz_uint level_and_flags,
                                    mz_uint64 uncomp_size, mz_uint32 uncomp_crc32, MZ_TIME_T *last_modified, const char *user_extra_data_local, mz_uint user_extra_data_local_len,
                                    const char *user_extra_data_central, mz_uint user_extra_data_central_len);

/*
 * 将文件的内容添加到存档中，记录磁盘文件的修改时间。
 * pZip: 指向 mz_zip_archive 结构的指针
 * pArchive_name: 存档中的文件名
 * pSrc_filename: 要添加的磁盘文件名
 * pComment: 指向文件注释数据的指针
 * comment_size: 注释数据的大小（字节数）
 * level_and_flags: 压缩级别和标志位
 * 返回值: 添加成功返回非零值，否则返回零
 */
mz_bool mz_zip_writer_add_file(mz_zip_archive *pZip, const char *pArchive_name, const char *pSrc_filename, const void *pComment, mz_uint16 comment_size, mz_uint level_and_flags);

/*
 * 将磁盘文件的内容添加到存档中，记录磁盘文件的修改时间。
 * pZip: 指向 mz_zip_archive 结构的指针
 * pArchive_name: 存档中的文件名
 * read_callback: 文件数据读取回调函数
 * callback_opaque: 回调函数的透明数据指针
 * size_to_add: 要添加到存档的数据大小
 * pFile_time: 指向文件修改时间的指针
 * pComment: 指向文件注释数据的指针
 * comment_size: 注释数据的大小（字节数）
 * level_and_flags: 压缩级别和标志位
 * user_extra_data_local: 指向本地文件头部的用户额外数据
 * user_extra_data_local_len: 本地文件头部用户额外数据的长度
 * user_extra_data_central: 指向中央目录记录的用户额外数据
 * user_extra_data_central_len: 中央目录记录用户额外数据的长度
 * 返回值: 添加成功返回非零值，否则返回零
 */
mz_bool mz_zip_writer_add_read_buf_callback(mz_zip_archive *pZip, const char *pArchive_name, mz_file_read_func read_callback, void* callback_opaque, mz_uint64 size_to_add,
    const MZ_TIME_T *pFile_time, const void *pComment, mz_uint16 comment_size, mz_uint level_and_flags, const char *user_extra_data_local, mz_uint user_extra_data_local_len,
    const char *user_extra_data_central, mz_uint user_extra_data_central_len);
/* Like mz_zip_writer_add_file(), except the file data is read from the specified FILE stream. */
/* 使用给定的 FILE 流中的数据来添加文件到 ZIP 归档中。 */

mz_bool mz_zip_writer_add_cfile(mz_zip_archive *pZip, const char *pArchive_name, MZ_FILE *pSrc_file, mz_uint64 size_to_add,
                                const MZ_TIME_T *pFile_time, const void *pComment, mz_uint16 comment_size, mz_uint level_and_flags, const char *user_extra_data_local, mz_uint user_extra_data_local_len,
                                const char *user_extra_data_central, mz_uint user_extra_data_central_len);
#endif

/* Adds a file to an archive by fully cloning the data from another archive. */
/* 通过从另一个归档中完全克隆数据，将文件添加到归档中。 */
/* This function fully clones the source file's compressed data (no recompression), along with its full filename, extra data (it may add or modify the zip64 local header extra data field), and the optional descriptor following the compressed data. */
/* 此函数完全克隆源文件的压缩数据（无需重新压缩），包括其完整的文件名、额外数据（可能会添加或修改zip64本地头部的额外数据字段），以及紧随压缩数据后的可选描述符。 */

mz_bool mz_zip_writer_add_from_zip_reader(mz_zip_archive *pZip, mz_zip_archive *pSource_zip, mz_uint src_file_index);

/* Finalizes the archive by writing the central directory records followed by the end of central directory record. */
/* 通过写入中央目录记录，然后写入中央目录结束记录，完成归档的最终操作。 */
/* After an archive is finalized, the only valid call on the mz_zip_archive struct is mz_zip_writer_end(). */
/* 完成归档之后，mz_zip_archive 结构上唯一有效的调用是 mz_zip_writer_end()。 */
/* An archive must be manually finalized by calling this function for it to be valid. */
/* 必须通过调用此函数手动完成归档，使其有效。 */

mz_bool mz_zip_writer_finalize_archive(mz_zip_archive *pZip);

/* Finalizes a heap archive, returning a poiner to the heap block and its size. */
/* 完成堆归档，返回指向堆块及其大小的指针。 */
/* The heap block will be allocated using the mz_zip_archive's alloc/realloc callbacks. */
/* 堆块将使用 mz_zip_archive 的分配/重新分配回调函数进行分配。 */

mz_bool mz_zip_writer_finalize_heap_archive(mz_zip_archive *pZip, void **ppBuf, size_t *pSize);

/* Ends archive writing, freeing all allocations, and closing the output file if mz_zip_writer_init_file() was used. */
/* 结束归档写入，释放所有分配的资源，并关闭输出文件（如果使用了 mz_zip_writer_init_file()）。 */
/* Note for the archive to be valid, it *must* have been finalized before ending (this function will not do it for you). */
/* 注意：要使归档有效，必须在结束之前完成归档（此函数不会自动完成）。 */

mz_bool mz_zip_writer_end(mz_zip_archive *pZip);

/* -------- Misc. high-level helper functions: */

/* mz_zip_add_mem_to_archive_file_in_place() efficiently (but not atomically) appends a memory blob to a ZIP archive. */
/* mz_zip_add_mem_to_archive_file_in_place() 高效地（但不是原子性地）将内存块追加到 ZIP 归档中。 */
/* Note this is NOT a fully safe operation. If it crashes or dies in some way your archive can be left in a screwed up state (without a central directory). */
/* 注意：这不是一个完全安全的操作。如果出现崩溃或其他问题，你的归档可能会处于混乱状态（没有中央目录）。 */
/* level_and_flags - compression level (0-10, see MZ_BEST_SPEED, MZ_BEST_COMPRESSION, etc.) logically OR'd with zero or more mz_zip_flags, or just set to MZ_DEFAULT_COMPRESSION. */
/* level_and_flags - 压缩级别（0-10，参见 MZ_BEST_SPEED、MZ_BEST_COMPRESSION 等），与一个或多个 mz_zip_flags 逻辑“或”，或者仅设置为 MZ_DEFAULT_COMPRESSION。 */
/* TODO: Perhaps add an option to leave the existing central dir in place in case the add dies? We could then truncate the file (so the old central dir would be at the end) if something goes wrong. */
/* TODO: 或许可以添加一个选项，在添加操作失败时保留现有的中央目录？这样，如果出现问题，我们可以截断文件（使旧的中央目录位于末尾）。 */

mz_bool mz_zip_add_mem_to_archive_file_in_place(const char *pZip_filename, const char *pArchive_name, const void *pBuf, size_t buf_size, const void *pComment, mz_uint16 comment_size, mz_uint level_and_flags);
/* mz_zip_add_mem_to_archive_file_in_place_v2函数声明 */
/* 向指定ZIP文件中添加内存中的数据作为一个新的存档项 */
/* pZip_filename: ZIP文件名 */
/* pArchive_name: 新存档项的名称 */
/* pBuf: 指向数据缓冲区的指针 */
/* buf_size: 数据缓冲区的大小 */
/* pComment: 存档项的注释（可选） */
/* comment_size: 注释的大小 */
/* level_and_flags: 压缩级别和标志 */
/* pErr: 指向错误代码的指针 */
mz_bool mz_zip_add_mem_to_archive_file_in_place_v2(const char *pZip_filename, const char *pArchive_name,
                                                  const void *pBuf, size_t buf_size,
                                                  const void *pComment, mz_uint16 comment_size,
                                                  mz_uint level_and_flags, mz_zip_error *pErr);

/* mz_zip_extract_archive_file_to_heap函数声明 */
/* 从存档中提取单个文件到堆中的内存块 */
/* pZip_filename: ZIP文件名 */
/* pArchive_name: 要提取的存档项名称 */
/* pSize: 指向返回数据大小的指针 */
/* flags: 提取操作的标志 */
/* 返回NULL表示提取失败 */
void *mz_zip_extract_archive_file_to_heap(const char *pZip_filename, const char *pArchive_name,
                                          size_t *pSize, mz_uint flags);

/* mz_zip_extract_archive_file_to_heap_v2函数声明 */
/* 从存档中提取单个文件到堆中的内存块，带有注释参数 */
/* pZip_filename: ZIP文件名 */
/* pArchive_name: 要提取的存档项名称 */
/* pComment: 指定的存档项注释（可选） */
/* pSize: 指向返回数据大小的指针 */
/* flags: 提取操作的标志 */
/* pErr: 指向错误代码的指针 */
void *mz_zip_extract_archive_file_to_heap_v2(const char *pZip_filename, const char *pArchive_name,
                                             const char *pComment, size_t *pSize,
                                             mz_uint flags, mz_zip_error *pErr);
```