# `D:\src\scipysrc\pandas\pandas\_libs\include\pandas\parser\tokenizer.h`

```
/*
 *  包含版权声明和许可信息
 */

#pragma once

// 定义 PY_SSIZE_T_CLEAN 以避免和 Python 2 中的宏定义冲突
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// 定义几种错误码常量
#define ERROR_NO_DIGITS 1
#define ERROR_OVERFLOW 2
#define ERROR_INVALID_CHARS 3

// 引入标准整数类型
#include <stdint.h>

// 定义流的初始大小常量
#define STREAM_INIT_SIZE 32

// 定义几种返回状态常量
#define REACHED_EOF 1
#define CALLING_READ_FAILED 2

/*

  用于 pandas / NumPy 的 C 平面文件解析低级代码

 */

/*
 *  read_rows() 和 tokenize() 函数的通用错误类型集合
 */

// 若定义了 VERBOSE，则定义 TRACE 宏为打印输出
#if defined(VERBOSE)
#define TRACE(X) printf X;
#else
#define TRACE(X)
#endif // VERBOSE

// 内存不足的错误码
#define PARSER_OUT_OF_MEMORY -1

/*
 *  TODO: 可能希望将 count_rows() 和 read_rows() 耦合在一起，以避免一些文件 I/O 的重复
 */

// 定义解析器的状态枚举
typedef enum {
  START_RECORD,
  START_FIELD,
  ESCAPED_CHAR,
  IN_FIELD,
  IN_QUOTED_FIELD,
  ESCAPE_IN_QUOTED_FIELD,
  QUOTE_IN_QUOTED_FIELD,
  EAT_CRNL,
  EAT_CRNL_NOP,
  EAT_WHITESPACE,
  EAT_COMMENT,
  EAT_LINE_COMMENT,
  WHITESPACE_LINE,
  START_FIELD_IN_SKIP_LINE,
  IN_FIELD_IN_SKIP_LINE,
  IN_QUOTED_FIELD_IN_SKIP_LINE,
  QUOTE_IN_QUOTED_FIELD_IN_SKIP_LINE,
  FINISHED
} ParserState;

// 定义引号样式枚举
typedef enum {
  QUOTE_MINIMAL,
  QUOTE_ALL,
  QUOTE_NONNUMERIC,
  QUOTE_NONE
} QuoteStyle;

// 定义坏行处理方法枚举
typedef enum { ERROR, WARN, SKIP } BadLineHandleMethod;

// 定义用于 I/O 回调的函数指针类型
typedef char *(*io_callback)(void *src, size_t nbytes, size_t *bytes_read,
                             int *status, const char *encoding_errors);

// 定义用于资源清理的函数指针类型
typedef void (*io_cleanup)(void *src);
typedef struct parser_t {
  void *source;           // 数据源指针
  io_callback cb_io;      // 输入输出回调函数
  io_cleanup cb_cleanup;  // 清理回调函数

  int64_t chunksize;      // 每个块的字节数
  char *data;             // 要处理的数据指针
  int64_t datalen;        // 可用数据量
  int64_t datapos;        // 数据当前位置

  char *stream;           // 写出标记化数据的位置
  uint64_t stream_len;    // 流的当前长度
  uint64_t stream_cap;    // 流的容量

  char **words;           // 存储单词的矩阵（可能是不规则的）
  int64_t *word_starts;   // 单词在流中的起始位置
  uint64_t words_len;     // 单词数组当前长度
  uint64_t words_cap;     // 单词数组容量
  uint64_t max_words_cap; // 最大单词容量

  char *pword_start;      // 当前字段在流中的起始位置
  int64_t word_start;     // 当前字段的起始位置

  int64_t *line_start;    // 行的起始位置数组
  int64_t *line_fields;   // 每行的字段数
  uint64_t lines;         // 观察到的有效行数
  uint64_t file_lines;    // 文件中的总行数（包括无效或跳过的）
  uint64_t lines_cap;     // 行数组容量

  ParserState state;      // 解析器状态
  int doublequote;        // 双引号是否由两个双引号表示
  char delimiter;         // 字段分隔符
  int delim_whitespace;   // 是否通过消耗空格/制表符进行分隔
  char quotechar;         // 引用字符
  char escapechar;        // 转义字符
  char lineterminator;    // 行终止符
  int skipinitialspace;   // 忽略分隔符后面的空格
  int quoting;            // 引用风格

  char commentchar;       // 注释字符
  int allow_embedded_newline; // 是否允许嵌入式换行

  int usecols;            // 是否提供了 usecols

  Py_ssize_t expected_fields; // 预期字段数
  BadLineHandleMethod on_bad_lines; // 处理不良行的方法

  char decimal;           // 浮点数选项：小数点符号
  char sci;               // 科学计数法选项

  char thousands;         // 千分位分隔符（逗号、点）

  int header;             // 是否有标题行
  int64_t header_start;   // 标题行开始位置
  uint64_t header_end;    // 标题行结束位置

  void *skipset;          // 跳过集合
  PyObject *skipfunc;     // 跳过函数
  int64_t skip_first_N_rows; // 跳过的前 N 行
  int64_t skip_footer;    // 跳过的尾部行数

  double (*double_converter)(const char *, char **, char, char, char, int,
                             int *, int *); // 双精度转换器函数指针

  char *warn_msg;         // 警告消息
  char *error_msg;        // 错误消息

  int skip_empty_lines;   // 是否跳过空行
} parser_t;

typedef struct coliter_t {
  char **words;           // 单词数组
  int64_t *line_start;    // 行起始位置数组
  int64_t col;            // 当前列索引
} coliter_t;

void coliter_setup(coliter_t *self, parser_t *parser, int64_t i, int64_t start);

#define COLITER_NEXT(iter, word)                                               \
  do {                                                                         \
    const int64_t i = *iter.line_start++ + iter.col;                           \
    word = i >= *iter.line_start ? "" : iter.words[i];                         \
  } while (0)

parser_t *parser_new(void);         // 创建新的解析器对象
int parser_init(parser_t *self);    // 初始化解析器对象
int parser_consume_rows(parser_t *self, size_t nrows);  // 消费指定数量的行
int parser_trim_buffers(parser_t *self);                // 修剪缓冲区
int parser_add_skiprow(parser_t *self, int64_t row);    // 添加要跳过的行号
// 设置解析器跳过的行数，用于忽略文件的开头几行数据
void parser_set_skipfirstnrows(parser_t *self, int64_t nrows);

// 释放解析器占用的资源
void parser_free(parser_t *self);

// 删除解析器对象并释放其资源
void parser_del(parser_t *self);

// 设置解析器的默认选项
void parser_set_default_options(parser_t *self);

// 对数据进行分词，限定行数，使用指定的编码错误处理方式
int tokenize_nrows(parser_t *self, size_t nrows, const char *encoding_errors);

// 对所有数据进行分词，使用指定的编码错误处理方式
int tokenize_all_rows(parser_t *self, const char *encoding_errors);

// 初始化 uint_state 结构体
typedef struct uint_state {
  int seen_sint;  // 记录是否已经见过有符号整数
  int seen_uint;  // 记录是否已经见过无符号整数
  int seen_null;  // 记录是否已经见过空值
} uint_state;

// 初始化 uint_state 结构体实例
void uint_state_init(uint_state *self);

// 检查 uint_state 结构体中是否出现了 uint64 冲突
int uint64_conflict(uint_state *self);

// 将字符串转换为 uint64_t 类型的整数，支持最大值和错误处理
uint64_t str_to_uint64(uint_state *state, const char *p_item, int64_t int_max,
                       uint64_t uint_max, int *error, char tsep);

// 将字符串转换为 int64_t 类型的整数，支持最小值、最大值和错误处理
int64_t str_to_int64(const char *p_item, int64_t int_min, int64_t int_max,
                     int *error, char tsep);

// 将字符串转换为 double 类型的浮点数，支持自定义小数点、科学计数法、千位分隔符和错误处理
double xstrtod(const char *p, char **q, char decimal, char sci, char tsep,
               int skip_trailing, int *error, int *maybe_int);

// 精确的字符串转换为 double 类型的浮点数，支持自定义小数点、科学计数法、千位分隔符和错误处理
double precise_xstrtod(const char *p, char **q, char decimal, char sci,
                       char tsep, int skip_trailing, int *error,
                       int *maybe_int);

// 对字符串进行往返转换，转换为 double 类型的浮点数，支持自定义小数点、科学计数法、千位分隔符和错误处理
// GH-15140 - round_trip 需要并自行获取 GIL
double round_trip(const char *p, char **q, char decimal, char sci, char tsep,
                  int skip_trailing, int *error, int *maybe_int);

// 将字符串转换为布尔类型，支持将字符串表示的布尔值转换为 uint8_t 类型的值
int to_boolean(const char *item, uint8_t *val);
```