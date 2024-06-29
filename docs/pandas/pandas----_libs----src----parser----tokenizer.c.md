# `D:\src\scipysrc\pandas\pandas\_libs\src\parser\tokenizer.c`

```
/*

Copyright (c) 2012, Lambda Foundry, Inc., except where noted

Incorporates components of WarrenWeckesser/textreader, licensed under 3-clause
BSD

See LICENSE for the license

*/

/*

Low-level ascii-file processing for pandas. Combines some elements from
Python's built-in csv module and Warren Weckesser's textreader project on
GitHub. See Python Software Foundation License and BSD licenses for these.

*/

#include "pandas/parser/tokenizer.h"
#include "pandas/portable.h"

#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>

#include "pandas/portable.h"
#include "pandas/vendored/klib/khash.h" // for kh_int64_t, kh_destroy_int64

// 设置列迭代器的初始值
void coliter_setup(coliter_t *self, parser_t *parser, int64_t i,
                   int64_t start) {
  // 将单词列表指针设置为解析器的单词列表
  self->words = parser->words;
  // 设置当前列号
  self->col = i;
  // 设置当前行的起始位置
  self->line_start = parser->line_start + start;
}

// 如果指针不为空则释放其指向的内存
static void free_if_not_null(void **ptr) {
  TRACE(("free_if_not_null %p\n", *ptr))
  if (*ptr != NULL) {
    free(*ptr);
    *ptr = NULL;
  }
}

/*

  Parser / tokenizer

*/

// 动态增长缓冲区以适应所需的长度
static void *grow_buffer(void *buffer, uint64_t length, uint64_t *capacity,
                         int64_t space, int64_t elsize, int *error) {
  uint64_t cap = *capacity;
  void *newbuffer = buffer;

  // 如果缓冲区无法容纳所需的空间，进行动态扩展
  while ((length + space >= cap) && (newbuffer != NULL)) {
    cap = cap ? cap << 1 : 2;
    buffer = newbuffer;
    newbuffer = realloc(newbuffer, elsize * cap);
  }

  if (newbuffer == NULL) {
    // 如果realloc失败，保持*capacity不变，设置*error为errno，并返回最后一次成功的realloc缓冲区
    *error = errno;
    newbuffer = buffer;
  } else {
    // 如果realloc成功，更新*capacity为扩展后的大小，设置*error为0
    *capacity = cap;
    *error = 0;
  }
  return newbuffer;
}

// 设置解析器的默认选项
void parser_set_default_options(parser_t *self) {
  // 小数点默认为'.'
  self->decimal = '.';
  // 科学计数法字符默认为'E'
  self->sci = 'E';

  // 用于标记当前处理状态的初始化
  self->state = START_RECORD;

  // 分隔符默认为','
  self->delimiter = ','; // XXX
  // 是否使用空白作为分隔符的标志，默认为0（不使用）
  self->delim_whitespace = 0;

  // 是否启用双引号的标志，默认为0（不启用）
  self->doublequote = 0;
  // 引号字符，默认为双引号('"')
  self->quotechar = '"';
  // 转义字符，默认为0（不启用）
  self->escapechar = 0;

  // 行终止符，默认为'\0'（NUL->标准逻辑）
  self->lineterminator = '\0'; /* NUL->standard logic */

  // 是否跳过字段前的空格，默认为0（不跳过）
  self->skipinitialspace = 0;
  // 引号处理方式，默认为QUOTE_MINIMAL
  self->quoting = QUOTE_MINIMAL;
  // 允许嵌套的换行符，默认为1（允许）
  self->allow_embedded_newline = 1;

  // 期望的字段数，默认为-1
  self->expected_fields = -1;
  // 处理错误行的策略，默认为ERROR
  self->on_bad_lines = ERROR;

  // 注释字符，默认为'#'
  self->commentchar = '#';
  // 千位分隔符，默认为'\0'
  self->thousands = '\0';

  // 跳过集合，默认为空指针
  self->skipset = NULL;
  // 跳过函数，默认为空指针
  self->skipfunc = NULL;
  // 跳过的行数，默认为-1
  self->skip_first_N_rows = -1;
  // 跳过末尾行数，默认为0
  self->skip_footer = 0;
}

// 创建并返回新的解析器对象
parser_t *parser_new(void) { return (parser_t *)calloc(1, sizeof(parser_t)); }

// 清理解析器数据缓冲区
static void parser_clear_data_buffers(parser_t *self) {
  // 如果流缓冲区不为空，则释放其内存
  free_if_not_null((void *)&self->stream);
  // 如果单词列表不为空，则释放其内存
  free_if_not_null((void *)&self->words);
  // 如果单词起始位置列表不为空，则释放其内存
  free_if_not_null((void *)&self->word_starts);
  // 如果行起始位置列表不为空，则释放其内存
  free_if_not_null((void *)&self->line_start);
  // 如果行字段列表不为空，则释放其内存
  free_if_not_null((void *)&self->line_fields);
}
static void parser_cleanup(parser_t *self) {
  // 释放错误消息的内存，如果非空
  free_if_not_null((void *)&self->error_msg);
  // 释放警告消息的内存，如果非空
  free_if_not_null((void *)&self->warn_msg);

  // 如果 skipset 集合非空，销毁该集合
  if (self->skipset != NULL) {
    kh_destroy_int64((kh_int64_t *)self->skipset);
    self->skipset = NULL;
  }

  // 清空数据缓冲区
  parser_clear_data_buffers(self);

  // 如果定义了清理回调函数，则调用清理回调函数并将其置空
  if (self->cb_cleanup != NULL) {
    self->cb_cleanup(self->source);
    self->cb_cleanup = NULL;
  }
}

int parser_init(parser_t *self) {
  /*
    初始化数据缓冲区
  */

  // 初始化流指针为空
  self->stream = NULL;
  // 初始化单词指针为空
  self->words = NULL;
  // 初始化单词起始指针为空
  self->word_starts = NULL;
  // 初始化行起始指针为空
  self->line_start = NULL;
  // 初始化行字段指针为空
  self->line_fields = NULL;
  // 初始化错误消息指针为空
  self->error_msg = NULL;
  // 初始化警告消息指针为空
  self->warn_msg = NULL;

  // 分配流缓冲区的初始大小
  self->stream = malloc(STREAM_INIT_SIZE * sizeof(char));
  if (self->stream == NULL) {
    // 如果分配失败，则清理资源并返回内存不足错误
    parser_cleanup(self);
    return PARSER_OUT_OF_MEMORY;
  }
  // 设置流缓冲区的容量和当前长度
  self->stream_cap = STREAM_INIT_SIZE;
  self->stream_len = 0;

  // 设置单词指针数组的初始大小
  const int64_t sz = STREAM_INIT_SIZE / 10;
  self->words = malloc(sz * sizeof(char *));
  // 设置单词起始位置数组的初始大小
  self->word_starts = malloc(sz * sizeof(int64_t));
  // 设置单词数组的最大容量和当前长度
  self->max_words_cap = sz;
  self->words_cap = sz;
  self->words_len = 0;

  // 设置行起始位置数组的初始大小
  self->line_start = malloc(sz * sizeof(int64_t));
  // 设置行字段数组的初始大小
  self->line_fields = malloc(sz * sizeof(int64_t));
  // 设置行数组的容量和初始长度
  self->lines_cap = sz;
  self->lines = 0;
  // 设置文件行数的初始值为0
  self->file_lines = 0;

  // 如果任一内存分配失败，则清理资源并返回内存不足错误
  if (self->stream == NULL || self->words == NULL ||
      self->word_starts == NULL || self->line_start == NULL ||
      self->line_fields == NULL) {
    parser_cleanup(self);
    return PARSER_OUT_OF_MEMORY;
  }

  /* 缓冲区中的字节数量 */
  self->datalen = 0;
  // 数据位置设置为0
  self->datapos = 0;

  // 设置第一行的起始位置和字段数量为0
  self->line_start[0] = 0;
  self->line_fields[0] = 0;

  // 设置单词起始指针为流的起始位置，单词起始位置为0
  self->pword_start = self->stream;
  self->word_start = 0;

  // 设置解析器状态为开始记录
  self->state = START_RECORD;

  // 初始化错误消息和警告消息指针为空
  self->error_msg = NULL;
  self->warn_msg = NULL;

  // 设置注释字符为NUL
  self->commentchar = '\0';

  // 返回初始化成功的状态值0
  return 0;
}

void parser_free(parser_t *self) {
  // 对应 parser_init 的反向操作
  parser_cleanup(self);
}

void parser_del(parser_t *self) { 
  // 释放解析器对象的内存
  free(self); 
}

static int make_stream_space(parser_t *self, size_t nbytes) {
  // 检查流缓冲区是否能容纳指定数量的字节（包括空字符）

  /*
    流缓冲区
  */

  int status;
  // 保存原始流指针
  char *orig_ptr = (void *)self->stream;
  TRACE(("\n\nmake_stream_space: nbytes = %zu.  grow_buffer(self->stream...)\n",
         nbytes))
  // 调用 grow_buffer 函数来扩展流缓冲区
  self->stream =
      (char *)grow_buffer((void *)self->stream, self->stream_len,
                          &self->stream_cap, nbytes * 2, sizeof(char), &status);
  TRACE(("make_stream_space: self->stream=%p, self->stream_len = %zu, "
         "self->stream_cap=%zu, status=%zu\n",
         self->stream, self->stream_len, self->stream_cap, status))

  // 如果扩展失败，则返回内存不足错误
  if (status != 0) {
    return PARSER_OUT_OF_MEMORY;
  }

  // 如果 realloc 在移动缓冲区时设置了 errno？

  // 如果重新分配成功，更新流指针并返回成功状态
  if (self->stream != orig_ptr) {
    //realloc 在移动缓冲区时设置了 errno 吗？
    // 将 self->stream 的地址加上 self->word_start 的偏移量，赋值给 self->pword_start
    self->pword_start = self->stream + self->word_start;

    // 遍历 self->words_len 长度的数组
    for (uint64_t i = 0; i < self->words_len; ++i) {
      // 将 self->stream 的地址加上 self->word_starts[i] 的偏移量，赋值给 self->words[i]
      self->words[i] = self->stream + self->word_starts[i];
    }
  }

  /*
    WORD VECTORS
  */

  // 从 self 对象中获取 self->words_cap 的值
  const uint64_t words_cap = self->words_cap;

  /**
   * 如果我们正在以块方式读取数据，需要注意先前块中出现的最大单词数（self->max_words_cap），
   * 这样可以在读取后续块时正确分配内存。
   *
   * 否则，如果误判了单词数不足而进行了错误的内存分配，可能会导致缓冲区溢出。
   */
  // 计算 length 变量的值，以便根据需要分配内存空间
  const uint64_t length = self->words_len + nbytes < self->max_words_cap
                              ? self->max_words_cap - nbytes - 1
                              : self->words_len;

  // 调用 grow_buffer 函数，重新分配 self->words 数组的内存空间
  self->words =
      (char **)grow_buffer((void *)self->words, length, &self->words_cap,
                           nbytes, sizeof(char *), &status);
  TRACE(("make_stream_space: grow_buffer(self->self->words, %zu, %zu, %zu, "
         "%d)\n",
         self->words_len, self->words_cap, nbytes, status))
  // 如果分配内存失败，则返回 PARSER_OUT_OF_MEMORY 错误码
  if (status != 0) {
    return PARSER_OUT_OF_MEMORY;
  }

  // 如果 realloc 操作改变了 words_cap 的值
  if (words_cap != self->words_cap) {
    TRACE(("make_stream_space: cap != self->words_cap, nbytes = %d, "
           "self->words_cap=%d\n",
           nbytes, self->words_cap))
    // 重新分配 self->word_starts 数组的内存空间
    int64_t *newptr = (int64_t *)realloc(self->word_starts,
                                         sizeof(int64_t) * self->words_cap);
    // 如果 realloc 失败，则返回 PARSER_OUT_OF_MEMORY 错误码
    if (newptr == NULL) {
      return PARSER_OUT_OF_MEMORY;
    } else {
      self->word_starts = newptr;
    }
  }

  /*
    LINE VECTORS
  */
  // 从 self 对象中获取 self->lines_cap 的值
  const uint64_t lines_cap = self->lines_cap;
  // 调用 grow_buffer 函数，重新分配 self->line_start 数组的内存空间
  self->line_start = (int64_t *)grow_buffer((void *)self->line_start,
                                            self->lines + 1, &self->lines_cap,
                                            nbytes, sizeof(int64_t), &status);
  TRACE(
      ("make_stream_space: grow_buffer(self->line_start, %zu, %zu, %zu, %d)\n",
       self->lines + 1, self->lines_cap, nbytes, status))
  // 如果分配内存失败，则返回 PARSER_OUT_OF_MEMORY 错误码
  if (status != 0) {
    return PARSER_OUT_OF_MEMORY;
  }

  // 如果 realloc 操作改变了 lines_cap 的值
  if (lines_cap != self->lines_cap) {
    TRACE(("make_stream_space: cap != self->lines_cap, nbytes = %d\n", nbytes))
    // 重新分配 self->line_fields 数组的内存空间
    int64_t *newptr = (int64_t *)realloc(self->line_fields,
                                         sizeof(int64_t) * self->lines_cap);
    // 如果 realloc 失败，则返回 PARSER_OUT_OF_MEMORY 错误码
    if (newptr == NULL) {
      return PARSER_OUT_OF_MEMORY;
    } else {
      self->line_fields = newptr;
    }
  }

  // 成功完成内存重分配，返回 0 表示没有错误
  return 0;
static int push_char(parser_t *self, char c) {
  // 打印调试信息，显示当前字符、流长度和流容量
  TRACE(("push_char: self->stream[%zu] = %x, stream_cap=%zu\n",
         self->stream_len + 1, c, self->stream_cap))
  
  // 检查流长度是否超过流容量，如果超过则处理缓冲区溢出错误
  if (self->stream_len >= self->stream_cap) {
    // 打印调试信息，显示流长度和流容量，并输出错误消息
    TRACE(("push_char: ERROR!!! self->stream_len(%d) >= "
           "self->stream_cap(%d)\n",
           self->stream_len, self->stream_cap))
    const size_t bufsize = 100;
    // 分配错误消息的缓冲区并写入错误信息
    self->error_msg = malloc(bufsize);
    snprintf(self->error_msg, bufsize,
             "Buffer overflow caught - possible malformed input file.\n");
    // 返回内存溢出错误码
    return PARSER_OUT_OF_MEMORY;
  }
  
  // 将字符 c 添加到流中，并增加流长度
  self->stream[self->stream_len++] = c;
  
  // 返回操作成功的标志
  return 0;
}

static inline int end_field(parser_t *self) {
  // XXX cruft
  // 检查词数组长度是否超过容量，如果超过则处理缓冲区溢出错误
  if (self->words_len >= self->words_cap) {
    // 打印调试信息，显示词数组长度和容量，并输出错误消息
    TRACE(("end_field: ERROR!!! self->words_len(%zu) >= "
           "self->words_cap(%zu)\n",
           self->words_len, self->words_cap))
    const size_t bufsize = 100;
    // 分配错误消息的缓冲区并写入错误信息
    self->error_msg = malloc(bufsize);
    snprintf(self->error_msg, bufsize,
             "Buffer overflow caught - possible malformed input file.\n");
    // 返回内存溢出错误码
    return PARSER_OUT_OF_MEMORY;
  }

  // 在词流中加入空字符作为当前词的结束标志
  push_char(self, '\0');

  // 设置词指针和元数据
  self->words[self->words_len] = self->pword_start;

  // 打印调试信息，显示词之间的字符差异
  TRACE(("end_field: Char diff: %d\n", self->pword_start - self->words[0]));

  // 打印调试信息，显示词的起始位置、当前词的指针和词数量
  TRACE(("end_field: Saw word %s at: %d. Total: %d\n", self->pword_start,
         self->word_start, self->words_len + 1))

  // 设置词的起始位置和增加词数量
  self->word_starts[self->words_len] = self->word_start;
  self->words_len++;

  // 增加行的字段计数
  self->line_fields[self->lines]++;

  // 设置下一个词的起始指针和词的起始位置
  self->pword_start = self->stream + self->stream_len;
  self->word_start = self->stream_len;

  // 返回操作成功的标志
  return 0;
}

static void append_warning(parser_t *self, const char *msg) {
  // 计算警告消息的长度
  const int64_t length = strlen(msg);

  // 如果警告消息为空，分配缓冲区并拷贝消息内容
  if (self->warn_msg == NULL) {
    self->warn_msg = malloc(length + 1);
    snprintf(self->warn_msg, length + 1, "%s", msg);
  } else {
    // 如果警告消息已存在，重新分配更大的缓冲区并追加消息内容
    const int64_t ex_length = strlen(self->warn_msg);
    char *newptr = (char *)realloc(self->warn_msg, ex_length + length + 1);
    if (newptr != NULL) {
      self->warn_msg = newptr;
      snprintf(self->warn_msg + ex_length, length + 1, "%s", msg);
    }
  }
}

static int end_line(parser_t *self) {
  // 获取期望的字段数和当前行的字段数
  int64_t ex_fields = self->expected_fields;
  int64_t fields = self->line_fields[self->lines];

  // 打印调试信息，显示行结束，当前行的字段数
  TRACE(("end_line: Line end, nfields: %d\n", fields));

  // 打印调试信息，显示当前行数
  TRACE(("end_line: lines: %d\n", self->lines));

  // 如果不是第一行，根据上一行的字段数确定期望的字段数
  if (self->lines > 0) {
    if (self->expected_fields >= 0) {
      ex_fields = self->expected_fields;
    } else {
      ex_fields = self->line_fields[self->lines - 1];
    }
  }
  
  // 打印调试信息，显示期望的字段数
  TRACE(("end_line: ex_fields: %d\n", ex_fields));

  // 如果当前状态是跳过行中的字段处理状态，直接返回
  if (self->state == START_FIELD_IN_SKIP_LINE ||
      self->state == IN_FIELD_IN_SKIP_LINE ||
      self->state == IN_QUOTED_FIELD_IN_SKIP_LINE ||
      self->state == QUOTE_IN_QUOTED_FIELD_IN_SKIP_LINE) {
    // 打印调试信息，显示跳过当前行
    TRACE(("end_line: Skipping row %d\n", self->file_lines));
    // 返回操作成功的标志

    return 0;
  }

  // 返回期望的字段数与当前行的字段数之间的差值
  return ex_fields - fields;
}
    // 增加文件行数计数
    self->file_lines++;

    // 跳过这一行中的字段
    self->line_start[self->lines] += fields;

    // 重置字段计数
    self->line_fields[self->lines] = 0;
    return 0;
  }

  // 如果行数超过头部结束行数加一，并且字段数大于预期字段数，并且未使用列过滤器
  if (!(self->lines <= self->header_end + 1) && (fields > ex_fields) &&
      !(self->usecols)) {
    // 增加文件行数计数
    self->file_lines++;

    // 跳过这一行中的字段
    self->line_start[self->lines] += fields;

    // 重置字段计数
    self->line_fields[self->lines] = 0;

    // file_lines 现在是实际文件行号（从1开始）
    if (self->on_bad_lines == ERROR) {
      const size_t bufsize = 100;
      self->error_msg = malloc(bufsize);
      snprintf(self->error_msg, bufsize,
               "Expected %" PRId64 " fields in line %" PRIu64 ", saw %" PRId64
               "\n",
               ex_fields, self->file_lines, fields);

      TRACE(("Error at line %d, %d fields\n", self->file_lines, fields));

      return -1;
    } else {
      // 简单地跳过错误的行
      if (self->on_bad_lines == WARN) {
        // 传递错误消息
        const size_t bufsize = 100;
        char *msg = (char *)malloc(bufsize);
        snprintf(msg, bufsize,
                 "Skipping line %" PRIu64 ": expected %" PRId64
                 " fields, saw %" PRId64 "\n",
                 self->file_lines, ex_fields, fields);
        append_warning(self, msg);
        free(msg);
      }
    }
  } else {
    // 缺少尾随分隔符
    if ((self->lines >= self->header_end + 1) && fields < ex_fields) {
      // 在关闭字段时可能溢出缓冲区
      if (make_stream_space(self, ex_fields - fields) < 0) {
        const size_t bufsize = 100;
        self->error_msg = malloc(bufsize);
        snprintf(self->error_msg, bufsize, "out of memory");
        return -1;
      }

      // 当字段少于预期字段时，循环关闭字段
      while (fields < ex_fields) {
        end_field(self);
        fields++;
      }
    }

    // 增加文件行数和总行数
    self->file_lines++;
    self->lines++;

    // 良好的行，设置新的起始点
    if (self->lines >= self->lines_cap) {
      TRACE(("end_line: ERROR!!! self->lines(%zu) >= self->lines_cap(%zu)\n",
             self->lines, self->lines_cap))
      const size_t bufsize = 100;
      self->error_msg = malloc(bufsize);
      snprintf(self->error_msg, bufsize,
               "Buffer overflow caught - "
               "possible malformed input file.\n");
      return PARSER_OUT_OF_MEMORY;
    }
    self->line_start[self->lines] =
        (self->line_start[self->lines - 1] + fields);

    TRACE(("end_line: new line start: %d\n", self->line_start[self->lines]));

    // 新行起始点，字段数为0
    self->line_fields[self->lines] = 0;
  }

  // 结束行的跟踪信息
  TRACE(("end_line: Finished line, at %d\n", self->lines));

  return 0;
}

// 将指定行数的行号添加到解析器的跳过集合中
int parser_add_skiprow(parser_t *self, int64_t row) {
    khiter_t k;              // 哈希表迭代器
    kh_int64_t *set;         // 64位整数键的哈希表
    int ret = 0;             // 返回值

    if (self->skipset == NULL) {
        self->skipset = (void *)kh_init_int64();   // 初始化哈希表
    }

    set = (kh_int64_t *)self->skipset;   // 将哈希表转换为相应类型的指针

    k = kh_put_int64(set, row, &ret);    // 在哈希表中插入行号
    set->keys[k] = row;                  // 设置键的值为行号

    return 0;                            // 返回成功
}

// 设置解析器跳过的前n行
void parser_set_skipfirstnrows(parser_t *self, int64_t nrows) {
    // 因为self->file_lines是从0开始计数，所以从nrows中减去1
    if (nrows > 0) {
        self->skip_first_N_rows = nrows - 1;   // 设置跳过的行数
    }
}

// 缓冲区字节处理函数
static int parser_buffer_bytes(parser_t *self, size_t nbytes,
                               const char *encoding_errors) {
    int status;             // 状态值
    size_t bytes_read;      // 读取的字节数

    status = 0;             // 初始化状态为0
    self->datapos = 0;      // 数据位置重置为0
    self->data =            // 调用回调函数获取数据
        self->cb_io(self->source, nbytes, &bytes_read, &status, encoding_errors);
    TRACE(
        ("parser_buffer_bytes self->cb_io: nbytes=%zu, datalen: %d, status=%d\n",
         nbytes, bytes_read, status));
    self->datalen = bytes_read;   // 设置数据长度

    // 处理非EOF和数据为NULL的异常情况
    if (status != REACHED_EOF && self->data == NULL) {
        const size_t bufsize = 200;   // 错误消息缓冲区大小
        self->error_msg = malloc(bufsize);   // 分配错误消息内存

        if (status == CALLING_READ_FAILED) {
            snprintf(self->error_msg, bufsize,
                     "Calling read(nbytes) on source failed. "
                     "Try engine='python'.");   // 格式化错误消息
        } else {
            snprintf(self->error_msg, bufsize, "Unknown error in IO callback");   // 格式化未知错误消息
        }
        return -1;   // 返回错误码
    }

    TRACE(("datalen: %d\n", self->datalen));   // 跟踪数据长度

    return status;   // 返回状态
}

/*

  Tokenization macros and state machine code

*/

// 将字符推入流，并检查是否溢出缓冲区
#define PUSH_CHAR(c)                                                           \
  TRACE(("PUSH_CHAR: Pushing %c, slen= %d, stream_cap=%zu, stream_len=%zu\n",  \
         c, slen, self->stream_cap, self->stream_len))                         \
  if (slen >= self->stream_cap) {                                              \
    TRACE(("PUSH_CHAR: ERROR!!! slen(%d) >= stream_cap(%d)\n", slen,           \
           self->stream_cap))                                                  \
    const size_t bufsize = 100;                                                \
    self->error_msg = malloc(bufsize);                                         \
    snprintf(self->error_msg, bufsize,                                         \
             "Buffer overflow caught - possible malformed input file.\n");     \
    return PARSER_OUT_OF_MEMORY;                                               \
  }                                                                            \
  *stream++ = c;                                                               \
  slen++;

// 这是一个临时的解决方案，但目前可以使用

// 结束当前字段处理
#define END_FIELD()                                                            \
  self->stream_len = slen;                                                     \
  if (end_field(self) < 0) {                                                   \
    # 跳转到解析错误处理标签，用于处理解析过程中的错误情况
    goto parsingerror;                                                         \
  }                                                                            \
  # 将指针移到流的末尾位置
  stream = self->stream + self->stream_len;                                    \
  # 记录当前流的长度到变量 slen 中
  slen = self->stream_len;
// 定义宏 END_LINE_STATE(STATE)，用于处理结束行状态
#define END_LINE_STATE(STATE)                                                  \
  self->stream_len = slen;                                                     \  // 设置对象中的流长度为当前流长度
  if (end_line(self) < 0) {                                                    \  // 调用 end_line 函数结束当前行，若返回负值则跳转到 parsingerror 标签
    goto parsingerror;                                                         \  // 跳转到 parsingerror 处理错误
  }                                                                            \
  stream = self->stream + self->stream_len;                                    \  // 更新流指针，指向当前流末尾位置
  slen = self->stream_len;                                                     \  // 更新当前流长度
  self->state = STATE;                                                         \  // 设置对象的状态为指定的 STATE
  if (line_limit > 0 && self->lines == start_lines + line_limit) {             \  // 如果行数限制大于 0 并且当前行数达到限制
    goto linelimit;                                                            \  // 跳转到 linelimit 处理行数限制
  }

// 定义宏 END_LINE_AND_FIELD_STATE(STATE)，用于处理结束行和字段状态
#define END_LINE_AND_FIELD_STATE(STATE)                                        \
  self->stream_len = slen;                                                     \  // 设置对象中的流长度为当前流长度
  if (end_line(self) < 0) {                                                    \  // 调用 end_line 函数结束当前行，若返回负值则跳转到 parsingerror 标签
    goto parsingerror;                                                         \  // 跳转到 parsingerror 处理错误
  }                                                                            \
  if (end_field(self) < 0) {                                                   \  // 调用 end_field 函数结束当前字段，若返回负值则跳转到 parsingerror 标签
    goto parsingerror;                                                         \  // 跳转到 parsingerror 处理错误
  }                                                                            \
  stream = self->stream + self->stream_len;                                    \  // 更新流指针，指向当前流末尾位置
  slen = self->stream_len;                                                     \  // 更新当前流长度
  self->state = STATE;                                                         \  // 设置对象的状态为指定的 STATE
  if (line_limit > 0 && self->lines == start_lines + line_limit) {             \  // 如果行数限制大于 0 并且当前行数达到限制
    goto linelimit;                                                            \  // 跳转到 linelimit 处理行数限制
  }

// 定义宏 END_LINE()，用于简化调用 END_LINE_STATE 宏并设置状态为 START_RECORD
#define END_LINE() END_LINE_STATE(START_RECORD)

// 定义宏 IS_TERMINATOR(c)，检查字符 c 是否为行终止符
#define IS_TERMINATOR(c) (c == lineterminator)

// 定义宏 IS_QUOTE(c)，检查字符 c 是否为引号符且引号模式不为 QUOTE_NONE
#define IS_QUOTE(c) ((c == self->quotechar && self->quoting != QUOTE_NONE))

// 定义宏 IS_CARRIAGE(c)，检查字符 c 是否为换行符
#define IS_CARRIAGE(c) (c == carriage_symbol)

// 定义宏 IS_COMMENT_CHAR(c)，检查字符 c 是否为注释符
#define IS_COMMENT_CHAR(c) (c == comment_symbol)

// 定义宏 IS_ESCAPE_CHAR(c)，检查字符 c 是否为转义符
#define IS_ESCAPE_CHAR(c) (c == escape_symbol)

// 定义宏 IS_SKIPPABLE_SPACE(c)，检查字符 c 是否为空格且跳过初始空格设置为真
#define IS_SKIPPABLE_SPACE(c)                                                  \
  ((!self->delim_whitespace && c == ' ' && self->skipinitialspace))

// 定义宏 IS_DELIMITER(c)，检查字符 c 是否为分隔符
#define IS_DELIMITER(c)                                                        \
  ((!delim_whitespace && c == delimiter) || (delim_whitespace && isblank(c)))

// 定义宏 _TOKEN_CLEANUP()，清理令牌处理状态并更新数据位置
#define _TOKEN_CLEANUP()                                                       \
  self->stream_len = slen;                                                     \  // 设置对象中的流长度为当前流长度
  self->datapos = i;                                                           \  // 更新数据位置为 i
  TRACE(("_TOKEN_CLEANUP: datapos: %d, datalen: %d\n", self->datapos,          \  // 跟踪日志，记录数据位置和长度信息
         self->datalen));
// 宏定义，检查缓冲区中是否存在字节顺序标记（BOM）
#define CHECK_FOR_BOM()                                                        \
  // 如果缓冲区的前三个字节是 UTF-8 BOM（0xefbbbf），则跳过它们
  if (*buf == '\xef' && *(buf + 1) == '\xbb' && *(buf + 2) == '\xbf') {        \
    buf += 3;                                                                  \
    // 更新数据位置，跳过 BOM 的三个字节
    self->datapos += 3;                                                        \
  }

// 静态函数，用于判断是否跳过当前行数据的处理
static int skip_this_line(parser_t *self, int64_t rownum) {
  // 如果存在跳过行处理的回调函数
  if (self->skipfunc != NULL) {
    // 在 Python 中调用跳过行处理的回调函数，并获取返回结果
    PyGILState_STATE state = PyGILState_Ensure();
    PyObject *result = PyObject_CallFunction(self->skipfunc, "i", rownum);

    // 如果调用过程中发生错误，则在 Cython 层面处理捕获
    const int should_skip = result == NULL ? -1 : PyObject_IsTrue(result);

    Py_XDECREF(result);
    PyGILState_Release(state);

    return should_skip;
  } else if (self->skipset != NULL) {
    // 如果存在跳过行的集合，检查当前行是否在集合中
    return (kh_get_int64((kh_int64_t *)self->skipset, self->file_lines) !=
            ((kh_int64_t *)self->skipset)->n_buckets);
  } else {
    // 否则，根据设定的跳过前 N 行的数量来决定是否跳过当前行
    return (rownum <= self->skip_first_N_rows);
  }
}

// 静态函数，处理字节流的分词操作
static int tokenize_bytes(parser_t *self, size_t line_limit,
                          uint64_t start_lines) {
  // 从数据起始位置获取缓冲区
  char *buf = self->data + self->datapos;

  // 确定行终止符的值，如果未设定，则默认为换行符 '\n'
  const char lineterminator =
      (self->lineterminator == '\0') ? '\n' : self->lineterminator;

  // 确定是否使用分隔符代替空格分隔，默认为禁用
  const int delim_whitespace = self->delim_whitespace;
  // 获取设定的分隔符
  const char delimiter = self->delimiter;

  // 确定换行符的 ASCII 值，如果未设定，则使用 1000
  const int carriage_symbol = (self->lineterminator == '\0') ? '\r' : 1000;
  // 确定注释符的 ASCII 值，如果未设定，则使用 1000
  const int comment_symbol =
      (self->commentchar != '\0') ? self->commentchar : 1000;
  // 确定转义符的 ASCII 值，如果未设定，则使用 1000
  const int escape_symbol =
      (self->escapechar != '\0') ? self->escapechar : 1000;

  // 如果无法为流分配足够的空间，报告内存不足错误并返回 -1
  if (make_stream_space(self, self->datalen - self->datapos) < 0) {
    const size_t bufsize = 100;
    // 分配错误消息的缓冲区并填充错误信息
    self->error_msg = malloc(bufsize);
    snprintf(self->error_msg, bufsize, "out of memory");
    return -1;
  }

  // 在流的尾部追加当前数据块
  char *stream = self->stream + self->stream_len;
  // 获取当前流的长度
  uint64_t slen = self->stream_len;

  // 打印调试信息，输出当前缓冲区的内容
  TRACE(("%s\n", buf));

  // 如果处理的是第一行数据，检查是否存在 BOM，并跳过它
  if (self->file_lines == 0) {
    CHECK_FOR_BOM();
  }

  char c;
  int64_t i;
  for (i = self->datapos; i < self->datalen; ++i) {
    // 逐字符处理文件中的下一个字符
    c = *buf++;

    // 打印详细的调试信息，输出当前处理字符的状态
    TRACE(("tokenize_bytes - Iter: %d Char: 0x%x Line %d field_count %d, "
           "state %d\n",
           i, c, self->file_lines + 1, self->line_fields[self->lines],
           self->state));

    // 根据当前状态机的状态进行处理
    switch (self->state) {
    case START_FIELD_IN_SKIP_LINE:
      // 如果当前字符是终止符
      if (IS_TERMINATOR(c)) {
        // 结束当前行的处理
        END_LINE();
      } else if (IS_CARRIAGE(c)) {
        // 如果当前字符是回车符，增加行数并切换到处理 CRNL 的状态
        self->file_lines++;
        self->state = EAT_CRNL_NOP;
      } else if (IS_QUOTE(c)) {
        // 如果当前字符是引号，切换到处理跳过行中的带引号字段的状态
        self->state = IN_QUOTED_FIELD_IN_SKIP_LINE;
      } else if (IS_DELIMITER(c)) {
        // 如果当前字符是分隔符，不执行任何操作，继续开始新字段的处理
        // 因为跳过行处理不需要特别处理分隔符
      } else {
        // 否则，切换到处理跳过行中的非引号字段的状态
        self->state = IN_FIELD_IN_SKIP_LINE;
      }
      break;
    // 处于跳过行中的字段中状态
    case IN_FIELD_IN_SKIP_LINE:
      // 如果是终止符，则结束当前行
      if (IS_TERMINATOR(c)) {
        END_LINE();
      } else if (IS_CARRIAGE(c)) {
        // 如果是回车符，增加文件行数计数，并切换到吞噬回车换行的状态
        self->file_lines++;
        self->state = EAT_CRNL_NOP;
      } else if (IS_DELIMITER(c)) {
        // 如果是分隔符，则切换到开始跳过行中的字段状态
        self->state = START_FIELD_IN_SKIP_LINE;
      }
      break;

    // 处于跳过行中的引号包裹字段中状态
    case IN_QUOTED_FIELD_IN_SKIP_LINE:
      // 如果是引号，则根据是否双引号处理状态切换
      if (IS_QUOTE(c)) {
        if (self->doublequote) {
          self->state = QUOTE_IN_QUOTED_FIELD_IN_SKIP_LINE;
        } else {
          self->state = IN_FIELD_IN_SKIP_LINE;
        }
      }
      break;

    // 处于跳过行中的引号包裹字段中的引号状态
    case QUOTE_IN_QUOTED_FIELD_IN_SKIP_LINE:
      // 如果是引号，则切换回跳过行中的引号包裹字段状态；如果是终止符，则结束当前行；如果是回车符，增加文件行数计数并切换到吞噬回车换行的状态；如果是分隔符，则切换到开始跳过行中的字段状态；否则切换回跳过行中的字段状态
      if (IS_QUOTE(c)) {
        self->state = IN_QUOTED_FIELD_IN_SKIP_LINE;
      } else if (IS_TERMINATOR(c)) {
        END_LINE();
      } else if (IS_CARRIAGE(c)) {
        self->file_lines++;
        self->state = EAT_CRNL_NOP;
      } else if (IS_DELIMITER(c)) {
        self->state = START_FIELD_IN_SKIP_LINE;
      } else {
        self->state = IN_FIELD_IN_SKIP_LINE;
      }
      break;

    // 处于空白行状态
    case WHITESPACE_LINE:
      // 如果是终止符，则增加文件行数计数，切换到开始记录状态；如果是回车符，则增加文件行数计数，切换到吞噬回车换行的状态；如果不是以空白字符分隔字段，并且当前字符是空白字符但不是分隔符，则执行回溯操作；否则继续向后处理
      if (IS_TERMINATOR(c)) {
        self->file_lines++;
        self->state = START_RECORD;
        break;
      } else if (IS_CARRIAGE(c)) {
        self->file_lines++;
        self->state = EAT_CRNL_NOP;
        break;
      } else if (!self->delim_whitespace) {
        if (isblank(c) && c != self->delimiter) {
          // 如果是空白字符但不是分隔符，则进行回溯
        } else { // backtrack
          // 使用 i + 1 是因为 buf 已经增加，但 i 还未增加
          do {
            --buf;
            --i;
          } while (i + 1 > self->datapos && !IS_TERMINATOR(*buf));

          // 如果到达了换行符而不是开头
          if (IS_TERMINATOR(*buf)) {
            ++buf; // 将指针移动到换行符后的第一个字符
            ++i;
          }
          self->state = START_FIELD;
        }
        break;
      }
      // 如果是以空白字符分隔字段，则穿过继续执行

    // 吞噬空白字符状态
    case EAT_WHITESPACE:
      // 如果是终止符，则结束当前行，切换到开始记录状态；如果是回车符，则切换到吞噬回车换行状态；如果是注释字符，则切换到吞噬注释状态；如果不是空白字符，则切换到开始字段状态，并使用 PD_FALLTHROUGH 注释提示穿透到下一个状态；否则继续吞噬空白字符
      if (IS_TERMINATOR(c)) {
        END_LINE();
        self->state = START_RECORD;
        break;
      } else if (IS_CARRIAGE(c)) {
        self->state = EAT_CRNL;
        break;
      } else if (IS_COMMENT_CHAR(c)) {
        self->state = EAT_COMMENT;
        break;
      } else if (!isblank(c)) {
        self->state = START_FIELD;
        PD_FALLTHROUGH; // 穿透到后续状态
      } else {
        // 如果是空白字符，继续吞噬
        break;
      }
    case START_RECORD: {
      // 记录开始处
      const int should_skip = skip_this_line(self, self->file_lines);

      // 判断是否需要跳过此行
      if (should_skip == -1) {
        goto parsingerror;  // 如果需要跳过，跳转到解析错误处理
      } else if (should_skip) {
        // 如果需要跳过，并且当前字符是引号，则进入跳过行内引号字段状态
        if (IS_QUOTE(c)) {
          self->state = IN_QUOTED_FIELD_IN_SKIP_LINE;
        } else {
          // 否则进入跳过行内字段状态
          self->state = IN_FIELD_IN_SKIP_LINE;

          // 如果当前字符是终止符，则结束当前行
          if (IS_TERMINATOR(c)) {
            END_LINE();
          }
        }
        break;
      } else if (IS_TERMINATOR(c)) {
        // 如果当前字符是终止符（换行），根据设置决定是否增加文件行数
        if (self->skip_empty_lines) {
          self->file_lines++;
        } else {
          END_LINE();
        }
        break;
      } else if (IS_CARRIAGE(c)) {
        // 如果当前字符是回车符
        if (self->skip_empty_lines) {
          self->file_lines++;
          self->state = EAT_CRNL_NOP;  // 进入吃掉回车换行空操作状态
        } else {
          self->state = EAT_CRNL;  // 进入吃掉回车换行状态
        }
        break;
      } else if (IS_COMMENT_CHAR(c)) {
        // 如果当前字符是注释符，进入吃掉整行注释状态
        self->state = EAT_LINE_COMMENT;
        break;
      } else if (isblank(c)) {
        // 如果当前字符是空白字符
        if (self->delim_whitespace) {
          // 如果允许分隔空白字符，根据设置进入空白行状态或吃掉空白状态
          if (self->skip_empty_lines) {
            self->state = WHITESPACE_LINE;
          } else {
            self->state = EAT_WHITESPACE;
          }
          break;
        } else if (c != self->delimiter && self->skip_empty_lines) {
          // 如果当前字符不是分隔符且允许跳过空行，进入空白行状态
          self->state = WHITESPACE_LINE;
          break;
        }
      }

      // 普通字符 - 继续处理为开始字段
      // 切换状态为开始字段
      self->state = START_FIELD;
      PD_FALLTHROUGH;
    }
    case START_FIELD:
      // 期待字段
      if (IS_TERMINATOR(c)) {
        END_FIELD();  // 结束当前字段
        END_LINE();   // 结束当前行
      } else if (IS_CARRIAGE(c)) {
        END_FIELD();        // 结束当前字段
        self->state = EAT_CRNL;  // 进入吃掉回车换行状态
      } else if (IS_QUOTE(c)) {
        // 开始引号字段
        self->state = IN_QUOTED_FIELD;
      } else if (IS_ESCAPE_CHAR(c)) {
        // 可能是转义字符
        self->state = ESCAPED_CHAR;
      } else if (IS_SKIPPABLE_SPACE(c)) {
        // 忽略字段开头的空格
      } else if (IS_DELIMITER(c)) {
        if (self->delim_whitespace) {
          self->state = EAT_WHITESPACE;  // 吃掉空白状态
        } else {
          END_FIELD();  // 保存空字段
        }
      } else if (IS_COMMENT_CHAR(c)) {
        END_FIELD();        // 结束当前字段
        self->state = EAT_COMMENT;  // 进入吃掉注释状态
      } else {
        // 开始新的非引号字段
        PUSH_CHAR(c);  // 将字符压入栈
        self->state = IN_FIELD;  // 进入字段状态
      }
      break;

    case ESCAPED_CHAR:
      PUSH_CHAR(c);      // 压入字符
      self->state = IN_FIELD;  // 进入字段状态
      break;

    case EAT_LINE_COMMENT:
      if (IS_TERMINATOR(c)) {
        self->file_lines++;    // 增加文件行数
        self->state = START_RECORD;  // 返回开始记录状态
      } else if (IS_CARRIAGE(c)) {
        self->file_lines++;    // 增加文件行数
        self->state = EAT_CRNL_NOP;  // 进入吃掉回车换行空操作状态
      }
      break;
    case IN_FIELD:
      // 如果当前在未引用的字段中
      if (IS_TERMINATOR(c)) {
        END_FIELD();  // 结束当前字段
        END_LINE();   // 结束当前行
      } else if (IS_CARRIAGE(c)) {
        END_FIELD();  // 结束当前字段
        self->state = EAT_CRNL;  // 进入吃掉回车换行的状态
      } else if (IS_ESCAPE_CHAR(c)) {
        // 可能是转义字符，进入转义字符状态
        self->state = ESCAPED_CHAR;
      } else if (IS_DELIMITER(c)) {
        // 字段结束 - 但还未到行结束
        END_FIELD();  // 结束当前字段

        if (self->delim_whitespace) {
          self->state = EAT_WHITESPACE;  // 吃掉空白字符状态
        } else {
          self->state = START_FIELD;  // 开始新字段状态
        }
      } else if (IS_COMMENT_CHAR(c)) {
        END_FIELD();  // 结束当前字段
        self->state = EAT_COMMENT;  // 进入吃掉注释状态
      } else {
        // 普通字符 - 存入字段中
        PUSH_CHAR(c);  // 将字符压入当前字段
      }
      break;

    case IN_QUOTED_FIELD:
      // 如果当前在引用的字段中
      if (IS_ESCAPE_CHAR(c)) {
        // 可能是转义字符
        self->state = ESCAPE_IN_QUOTED_FIELD;  // 进入引号内的转义字符状态
      } else if (IS_QUOTE(c)) {
        if (self->doublequote) {
          // 双引号 - "" 表示一个引号
          self->state = QUOTE_IN_QUOTED_FIELD;  // 进入引号内的引号状态
        } else {
          // 引号部分字段结束
          self->state = IN_FIELD;  // 回到未引用字段状态
        }
      } else {
        // 普通字符 - 存入字段中
        PUSH_CHAR(c);  // 将字符压入当前字段
      }
      break;

    case ESCAPE_IN_QUOTED_FIELD:
      PUSH_CHAR(c);  // 将字符压入当前字段
      self->state = IN_QUOTED_FIELD;  // 回到引用字段状态
      break;

    case QUOTE_IN_QUOTED_FIELD:
      // 双引号 - 在引用字段内看到一个引号
      if (IS_QUOTE(c)) {
        // 将 "" 存为 "
        PUSH_CHAR(c);  // 将字符压入当前字段
        self->state = IN_QUOTED_FIELD;  // 回到引用字段状态
      } else if (IS_DELIMITER(c)) {
        // 字段结束 - 但还未到行结束
        END_FIELD();  // 结束当前字段

        if (self->delim_whitespace) {
          self->state = EAT_WHITESPACE;  // 吃掉空白字符状态
        } else {
          self->state = START_FIELD;  // 开始新字段状态
        }
      } else if (IS_TERMINATOR(c)) {
        END_FIELD();  // 结束当前字段
        END_LINE();   // 结束当前行
      } else if (IS_CARRIAGE(c)) {
        END_FIELD();  // 结束当前字段
        self->state = EAT_CRNL;  // 进入吃掉回车换行的状态
      } else {
        PUSH_CHAR(c);  // 将字符压入当前字段
        self->state = IN_FIELD;  // 回到未引用字段状态
      }
      break;

    case EAT_COMMENT:
      if (IS_TERMINATOR(c)) {
        END_LINE();  // 结束当前行
      } else if (IS_CARRIAGE(c)) {
        self->state = EAT_CRNL;  // 进入吃掉回车换行的状态
      }
      break;

    // 只会在非自定义行终止符时发生，
    // 因此直接检查 '\n'
    // 处理 EAT_CRNL 状态下的情况
    case EAT_CRNL:
      // 如果字符为 '\n'，表示结束一行
      if (c == '\n') {
        END_LINE();
      } else if (IS_DELIMITER(c)) {  // 如果是分隔符
        if (self->delim_whitespace) {  // 如果是空白字符分隔
          END_LINE_STATE(EAT_WHITESPACE);  // 结束当前行，进入吃掉空白字符状态
        } else {
          // 处理 \r 作为行结束的文件
          END_LINE_AND_FIELD_STATE(START_FIELD);  // 结束当前行和字段状态，进入开始字段状态
        }
      } else {
        if (self->delim_whitespace) {  // 如果是空白字符分隔
          /* XXX
           * 新记录的第一个字符--需要回退并
           * 重新读取
           * 以正确处理...
           */
          i--;  // 回退一个字符位置
          buf--; // 向前回退一个字符（HACK!）
          END_LINE_STATE(START_RECORD);  // 结束当前行，进入开始记录状态
        } else {
          // \r 作为行终止符
          // UGH. 我们实际上不想
          // 消耗这个标记。稍后修复
          self->stream_len = slen;  // 设置流的长度
          if (end_line(self) < 0) {  // 结束当前行的处理
            goto parsingerror;  // 转到解析错误处理标签
          }

          stream = self->stream + self->stream_len;  // 更新流位置
          slen = self->stream_len;  // 更新流长度
          self->state = START_RECORD;  // 进入开始记录状态

          --i;  // 回退一个字符位置
          buf--; // 向前回退一个字符（HACK!）
          if (line_limit > 0 && self->lines == start_lines + line_limit) {
            goto linelimit;  // 如果达到行数限制，跳转到行数限制标签
          }
        }
      }
      break;

    // 只有在非自定义行终止符时才会发生，因此直接检查 '\n'
    case EAT_CRNL_NOP: // 在忽略的注释行内部
      self->state = START_RECORD;  // 进入开始记录状态
      // \r 作为行终止符，需要重新解析这个字符
      if (c != '\n' && !IS_DELIMITER(c)) {
        --i;  // 回退一个字符位置
        --buf;  // 向前回退一个字符
      }
      break;
    default:
      break;
    }
  }

  _TOKEN_CLEANUP();  // 清理令牌

  TRACE(("Finished tokenizing input\n"))  // 跟踪输出，表示完成输入的令牌化

  return 0;  // 返回值为0，表示成功结束
parsingerror:
  i++;  // 增加计数器 i 的值
  _TOKEN_CLEANUP();  // 调用宏 _TOKEN_CLEANUP，用于清理解析器的状态

  return -1;  // 返回错误代码 -1，表示解析错误

linelimit:
  i++;  // 增加计数器 i 的值
  _TOKEN_CLEANUP();  // 调用宏 _TOKEN_CLEANUP，用于清理解析器的状态

  return 0;  // 返回成功代码 0，表示达到行限制

}

static int parser_handle_eof(parser_t *self) {
  const size_t bufsize = 100;  // 定义缓冲区大小为 100

  TRACE(("handling eof, datalen: %d, pstate: %d\n", self->datalen, self->state))
  // 打印跟踪消息，显示 datalen 和 pstate 的值

  if (self->datalen != 0)
    return -1;  // 如果数据长度不为零，返回错误代码 -1

  switch (self->state) {  // 根据解析器的状态进行分支处理
  case START_RECORD:
  case WHITESPACE_LINE:
  case EAT_CRNL_NOP:
  case EAT_LINE_COMMENT:
    return 0;  // 对于指定状态，返回成功代码 0

  case ESCAPE_IN_QUOTED_FIELD:
  case IN_QUOTED_FIELD:
    // 如果处于引号包围的字段中或在转义字符中，分配错误消息缓冲区并格式化错误消息
    self->error_msg = (char *)malloc(bufsize);
    snprintf(self->error_msg, bufsize,
             "EOF inside string starting at row %" PRIu64, self->file_lines);
    return -1;  // 返回错误代码 -1

  case ESCAPED_CHAR:
    // 如果处于转义字符后，分配错误消息缓冲区并格式化错误消息
    self->error_msg = (char *)malloc(bufsize);
    snprintf(self->error_msg, bufsize, "EOF following escape character");
    return -1;  // 返回错误代码 -1

  case IN_FIELD:
  case START_FIELD:
  case QUOTE_IN_QUOTED_FIELD:
    if (end_field(self) < 0)  // 如果结束字段操作失败，返回错误代码 -1
      return -1;
    break;  // 否则继续执行后续操作

  default:
    break;  // 其他状态不做处理
  }

  if (end_line(self) < 0)  // 结束行处理操作，如果失败，返回错误代码 -1；成功返回 0
    return -1;
  else
    return 0;  // 返回成功代码 0
}

int parser_consume_rows(parser_t *self, size_t nrows) {
  if (nrows > self->lines) {  // 如果要消费的行数大于解析器中的总行数，将行数设为解析器中的总行数
    nrows = self->lines;
  }

  /* do nothing */
  if (nrows == 0)  // 如果消费行数为 0，不做任何操作，直接返回成功代码 0
    return 0;

  /* cannot guarantee that nrows + 1 has been observed */
  const int64_t word_deletions =
      self->line_start[nrows - 1] + self->line_fields[nrows - 1];

  /* if word_deletions == 0 (i.e. this case) then char_count must
   * be 0 too, as no data needs to be skipped */
  const uint64_t char_count =
      word_deletions >= 1 ? (self->word_starts[word_deletions - 1] +
                             strlen(self->words[word_deletions - 1]) + 1)
                          : 0;

  TRACE(("parser_consume_rows: Deleting %d words, %d chars\n", word_deletions,
         char_count));
  // 打印跟踪消息，显示要删除的单词数和字符数

  /* move stream, only if something to move */
  if (char_count < self->stream_len) {
    memmove(self->stream, (self->stream + char_count),
            self->stream_len - char_count);
  }
  /* buffer counts */
  self->stream_len -= char_count;  // 更新流的长度

  /* move token metadata */
  // 注意：我们应该始终确保 words_len < word_deletions，因此此减法将保持适当的类型。
  int64_t offset;
  for (uint64_t i = 0; i < self->words_len - word_deletions; ++i) {
    offset = i + word_deletions;

    self->words[i] = self->words[offset] - char_count;  // 调整单词在流中的位置
    self->word_starts[i] = self->word_starts[offset] - char_count;  // 调整单词开始位置
  }
  self->words_len -= word_deletions;  // 更新单词数

  /* move current word pointer to stream */
  self->pword_start -= char_count;  // 更新当前单词指针在流中的位置
  self->word_start -= char_count;  // 更新单词开始位置

  /* move line metadata */
  // 注意：我们应该始终确保 self->lines - nrows + 1 >= 0，因此此减法将保持适当的类型。
  for (uint64_t i = 0; i < self->lines - nrows + 1; ++i) {
    offset = i + nrows;
    self->line_start[i] = self->line_start[offset] - word_deletions;  // 调整行的起始位置
    self->line_fields[i] = self->line_fields[offset];  // 更新行的字段数
  }
  self->lines -= nrows;  // 更新剩余行数

  return 0;  // 返回成功代码 0
}
int parser_trim_buffers(parser_t *self) {
  /*
    Free memory and trim buffers as necessary
   */

  /**
   * Before freeing up space and trimming buffers, we check
   * if the current word capacity exceeds the previously recorded
   * maximum. If so, update the maximum word capacity.
   *
   * This is crucial when parsing data in chunks, as it informs
   * subsequent parsing about the potential number of words.
   */
  if (self->words_cap > self->max_words_cap) {
    self->max_words_cap = self->words_cap;
  }

  /* Trim words and word_starts arrays */
  size_t new_cap = _next_pow2(self->words_len) + 1;
  if (new_cap < self->words_cap) {
    TRACE(("parser_trim_buffers: new_cap < self->words_cap\n"));
    self->words = realloc(self->words, new_cap * sizeof(char *));
    if (self->words == NULL) {
      return PARSER_OUT_OF_MEMORY;
    }
    self->word_starts = realloc(self->word_starts, new_cap * sizeof(int64_t));
    if (self->word_starts == NULL) {
      return PARSER_OUT_OF_MEMORY;
    }
    self->words_cap = new_cap;
  }

  /* Trim stream buffer */
  new_cap = _next_pow2(self->stream_len) + 1;
  TRACE(("parser_trim_buffers: new_cap = %zu, stream_cap = %zu, lines_cap = "
         "%zu\n",
         new_cap, self->stream_cap, self->lines_cap));
  if (new_cap < self->stream_cap) {
    TRACE(("parser_trim_buffers: new_cap < self->stream_cap, calling "
           "realloc\n"));
    void *newptr = realloc(self->stream, new_cap);
    if (newptr == NULL) {
      return PARSER_OUT_OF_MEMORY;
    } else {
      // Update pointers in the self->words array (char **) if `realloc`
      // moved the `self->stream` buffer. This block mirrors a similar
      // block in `make_stream_space`.
      if (self->stream != newptr) {
        self->pword_start = (char *)newptr + self->word_start;

        for (uint64_t i = 0; i < self->words_len; ++i) {
          self->words[i] = (char *)newptr + self->word_starts[i];
        }
      }

      self->stream = newptr;
      self->stream_cap = new_cap;
    }
  }

  /* Trim line_start and line_fields arrays */
  new_cap = _next_pow2(self->lines) + 1;
  if (new_cap < self->lines_cap) {
    TRACE(("parser_trim_buffers: new_cap < self->lines_cap\n"));
    void *newptr = realloc(self->line_start, new_cap * sizeof(int64_t));
    if (newptr == NULL) {
      return PARSER_OUT_OF_MEMORY;
    } else {
      self->line_start = newptr;
    }
    newptr = realloc(self->line_fields, new_cap * sizeof(int64_t));
    if (newptr == NULL) {
      return PARSER_OUT_OF_MEMORY;
    } else {
      self->line_fields = newptr;
      self->lines_cap = new_cap;
    }
  }

  return 0;
}
// 帮助函数：分词辅助函数
static int _tokenize_helper(parser_t *self, size_t nrows, int all,
                            const char *encoding_errors) {
  // 保存当前行数，作为起始行数
  const uint64_t start_lines = self->lines;

  // 如果状态已经是完成状态，则直接返回
  if (self->state == FINISHED) {
    return 0;
  }

  // 打印追踪信息，显示请求分词的行数、数据位置和数据长度
  TRACE(
      ("_tokenize_helper: Asked to tokenize %d rows, datapos=%d, datalen=%d\n",
       nrows, self->datapos, self->datalen));

  // 进入循环，处理数据分词
  while (1) {
    // 如果非全部处理并且行数超过指定行数，则退出循环
    if (!all && self->lines - start_lines >= nrows)
      break;

    // 如果数据位置达到数据长度，则调用 parser_buffer_bytes 函数读取更多数据
    if (self->datapos == self->datalen) {
      status = parser_buffer_bytes(self, self->chunksize, encoding_errors);

      // 如果读取到文件末尾，则处理文件末尾情况并结束
      if (status == REACHED_EOF) {
        // 关闭最后一行数据的处理
        status = parser_handle_eof(self);
        self->state = FINISHED;
        break;
      } else if (status != 0) {
        return status;
      }
    }

    // 打印追踪信息，显示尝试处理的字节数、数据长度和数据位置
    TRACE(("_tokenize_helper: Trying to process %d bytes, datalen=%d, "
           "datapos= %d\n",
           self->datalen - self->datapos, self->datalen, self->datapos));

    // 调用 tokenize_bytes 函数处理数据分词，返回状态
    status = tokenize_bytes(self, nrows, start_lines);

    // 如果 tokenize_bytes 返回负数状态，则打印错误信息并中断循环
    if (status < 0) {
      // XXX 表示有错误发生
      TRACE(("_tokenize_helper: Status %d returned from tokenize_bytes, "
             "breaking\n",
             status));
      status = -1;
      break;
    }
  }
  // 打印追踪信息，显示离开 tokenize_helper 函数
  TRACE(("leaving tokenize_helper\n"));
  return status;
}

// 函数：按行数进行分词处理
int tokenize_nrows(parser_t *self, size_t nrows, const char *encoding_errors) {
  // 调用 _tokenize_helper 函数进行行数限制的分词处理
  return _tokenize_helper(self, nrows, 0, encoding_errors);
}

// 函数：处理所有行数的分词
int tokenize_all_rows(parser_t *self, const char *encoding_errors) {
  // 调用 _tokenize_helper 函数进行所有行数的分词处理
  return _tokenize_helper(self, -1, 1, encoding_errors);
}

/*
 * 函数: to_boolean
 * --------------------
 *
 * 验证项目是否应该识别为布尔字段。
 *
 * item: 表示解析文本的 const char* 指针
 * val: 指向布尔表示的 uint8_t 指针
 *
 * 如果项目被确定为布尔值，则设置 val 的适当值并返回 0。
 * 非零退出状态意味着未能将项目推断为布尔值，并且保持 *val 的值不变。
 */
int to_boolean(const char *item, uint8_t *val) {
  // 如果 item 是 "TRUE"，则设置 *val 为 1，表示真
  if (strcasecmp(item, "TRUE") == 0) {
    *val = 1;
    return 0;
  } else if (strcasecmp(item, "FALSE") == 0) {  // 如果 item 是 "FALSE"，则设置 *val 为 0，表示假
    *val = 0;
    return 0;
  }

  // 如果不是 "TRUE" 或 "FALSE"，则返回 -1，表示无法识别为布尔值
  return -1;
}

// ---------------------------------------------------------------------------
// xstrtod 的实现

//
// strtod.c
//
// 将字符串转换为 double 类型
//
// 版权所有 (C) 2002 Michael Ringgaard. 保留所有权利。
//
// 允许在源代码和二进制形式下进行再发布和使用，无论是否修改，只要满足以下条件：
//
// 1. 源代码的再发布必须保留上述版权声明、此条件列表和以下免责声明。
// 2. 二进制形式的再发布必须在分发的文档和/或其他材料中重复上述版权声明、此条件列表和以下免责声明。
//
// 3. Neither the name of the project nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.
//
// -----------------------------------------------------------------------
// Modifications by Warren Weckesser, March 2011:
// * Rename strtod() to xstrtod().
// * Added decimal and sci arguments.
// * Skip trailing spaces.
// * Commented out the other functions.
// Modifications by Richard T Guy, August 2013:
// * Add tsep argument for thousands separator
//

// 定义一个函数 xstrtod，用于将字符串转换为 double 类型数字
double xstrtod(const char *str, char **endptr, char decimal, char sci,
               char tsep, int skip_trailing, int *error, int *maybe_int) {
  // 指针 p 指向输入字符串 str
  const char *p = str;
  // 如果 maybe_int 不为 NULL，将其设为 1
  if (maybe_int != NULL)
    *maybe_int = 1;
  
  // 跳过字符串开头的空白字符
  while (isspace_ascii(*p))
    p++;

  // 处理可选的符号部分
  int negative = 0;
  switch (*p) {
  case '-':
    negative = 1;
    PD_FALLTHROUGH; // 继续处理下一个情况
  case '+':
    p++;
    break;
  }

  int exponent = 0;
  int num_digits = 0;
  int num_decimals = 0;

  // 悲观但快速的估算，假设每个小数位需要 4 位存储空间
  // TODO: C23 中有 UINT64_WIDTH 宏可以在编译时使用
  const int max_int_decimal_digits = (sizeof(unsigned int) * 8) / 4;

  // 处理数字字符
  unsigned int i_number = 0;
  while (isdigit_ascii(*p) && num_digits <= max_int_decimal_digits) {
    i_number = i_number * 10 + (*p - '0');
    p++;
    num_digits++;

    // 处理千位分隔符
    p += (tsep != '\0' && *p == tsep);
  }
  double number = i_number;

  // 如果超过了整数部分的最大位数，继续处理作为浮点数
  if (num_digits > max_int_decimal_digits) {
    while (isdigit_ascii(*p)) {
      number = number * 10. + (*p - '0');
      p++;
      num_digits++;

      p += (tsep != '\0' && *p == tsep);
    }
  }

  // 处理小数部分
  if (*p == decimal) {
    if (maybe_int != NULL)
      *maybe_int = 0;
    p++;

    while (isdigit_ascii(*p)) {
      number = number * 10. + (*p - '0');
      p++;
      num_digits++;
      num_decimals++;
    }

    exponent -= num_decimals;
  }

  // 如果没有有效数字，则设置错误码为 ERANGE
  if (num_digits == 0) {
    *error = ERANGE;
  return 0.0;
  // 返回浮点数 0.0，表示函数结束并返回该值
}

// Correct for sign.
// 修正数值的符号。

if (negative)
  number = -number;
// 如果标志为负，将数值取反。

// Process an exponent string.
// 处理指数字符串。

if (toupper_ascii(*p) == toupper_ascii(sci)) {
  if (maybe_int != NULL)
    *maybe_int = 0;
  // 如果当前字符转换为大写后与sci相等，则处理指数部分。

  // Handle optional sign.
  // 处理可选的符号。

  negative = 0;
  switch (*++p) {
  case '-':
    negative = 1;
    PD_FALLTHROUGH; // Fall through to increment position.
  case '+':
    p++;
    break;
  }
  // 根据指数部分的正负处理。

  // Process string of digits.
  // 处理数字字符串。

  num_digits = 0;
  int n = 0;
  while (isdigit_ascii(*p)) {
    n = n * 10 + (*p - '0');
    num_digits++;
    p++;
  }
  // 读取并转换指数值。

  if (negative)
    exponent -= n;
  else
    exponent += n;
  // 根据符号将读取的数字转换为指数值。

  // If no digits, after the 'e'/'E', un-consume it
  // 如果没有读取到数字，则在'e'/'E'后回退一个字符。
  if (num_digits == 0)
    p--;
}

if (exponent < DBL_MIN_EXP || exponent > DBL_MAX_EXP) {
  *error = ERANGE;
  return HUGE_VAL;
}
// 检查指数是否超出双精度浮点数范围，如超出则返回错误码并返回特定值。

// Scale the result.
// 缩放结果。

double p10 = 10.;
int n = exponent;
if (n < 0)
  n = -n;
while (n) {
  if (n & 1) {
    if (exponent < 0)
      number /= p10;
    else
      number *= p10;
  }
  n >>= 1;
  p10 *= p10;
}
// 根据指数值对数值进行缩放。

if (number == HUGE_VAL) {
  *error = ERANGE;
}
// 如果数值为HUGE_VAL，则返回错误码。

if (skip_trailing) {
  // Skip trailing whitespace.
  // 跳过末尾的空白字符。

  while (isspace_ascii(*p))
    p++;
}
// 如果需要跳过末尾的空白字符，则继续移动指针直到非空白字符。

if (endptr)
  *endptr = (char *)p;
return number;
// 如果endptr非空，则将其指向当前指针位置。返回最终解析的数值。
}

// 函数：将字符串转换为双精度浮点数，支持自定义参数
double precise_xstrtod(const char *str, char **endptr, char decimal, char sci,
                       char tsep, int skip_trailing, int *error,
                       int *maybe_int) {
  // 指向字符串的指针
  const char *p = str;
  // 最大有效数字位数
  const int max_digits = 17;

  // 如果 maybe_int 不为空，设置为可能是整数
  if (maybe_int != NULL)
    *maybe_int = 1;

  // 缓存10的幂次方，静态存储在内存中
  static double e[] = {
      1.,    1e1,   1e2,   1e3,   1e4,   1e5,   1e6,   1e7,   1e8,   1e9,
      1e10,  1e11,  1e12,  1e13,  1e14,  1e15,  1e16,  1e17,  1e18,  1e19,
      1e20,  1e21,  1e22,  1e23,  1e24,  1e25,  1e26,  1e27,  1e28,  1e29,
      1e30,  1e31,  1e32,  1e33,  1e34,  1e35,  1e36,  1e37,  1e38,  1e39,
      1e40,  1e41,  1e42,  1e43,  1e44,  1e45,  1e46,  1e47,  1e48,  1e49,
      1e50,  1e51,  1e52,  1e53,  1e54,  1e55,  1e56,  1e57,  1e58,  1e59,
      1e60,  1e61,  1e62,  1e63,  1e64,  1e65,  1e66,  1e67,  1e68,  1e69,
      1e70,  1e71,  1e72,  1e73,  1e74,  1e75,  1e76,  1e77,  1e78,  1e79,
      1e80,  1e81,  1e82,  1e83,  1e84,  1e85,  1e86,  1e87,  1e88,  1e89,
      1e90,  1e91,  1e92,  1e93,  1e94,  1e95,  1e96,  1e97,  1e98,  1e99,
      1e100, 1e101, 1e102, 1e103, 1e104, 1e105, 1e106, 1e107, 1e108, 1e109,
      1e110, 1e111, 1e112, 1e113, 1e114, 1e115, 1e116, 1e117, 1e118, 1e119,
      1e120, 1e121, 1e122, 1e123, 1e124, 1e125, 1e126, 1e127, 1e128, 1e129,
      1e130, 1e131, 1e132, 1e133, 1e134, 1e135, 1e136, 1e137, 1e138, 1e139,
      1e140, 1e141, 1e142, 1e143, 1e144, 1e145, 1e146, 1e147, 1e148, 1e149,
      1e150, 1e151, 1e152, 1e153, 1e154, 1e155, 1e156, 1e157, 1e158, 1e159,
      1e160, 1e161, 1e162, 1e163, 1e164, 1e165, 1e166, 1e167, 1e168, 1e169,
      1e170, 1e171, 1e172, 1e173, 1e174, 1e175, 1e176, 1e177, 1e178, 1e179,
      1e180, 1e181, 1e182, 1e183, 1e184, 1e185, 1e186, 1e187, 1e188, 1e189,
      1e190, 1e191, 1e192, 1e193, 1e194, 1e195, 1e196, 1e197, 1e198, 1e199,
      1e200, 1e201, 1e202, 1e203, 1e204, 1e205, 1e206, 1e207, 1e208, 1e209,
      1e210, 1e211, 1e212, 1e213, 1e214, 1e215, 1e216, 1e217, 1e218, 1e219,
      1e220, 1e221, 1e222, 1e223, 1e224, 1e225, 1e226, 1e227, 1e228, 1e229,
      1e230, 1e231, 1e232, 1e233, 1e234, 1e235, 1e236, 1e237, 1e238, 1e239,
      1e240, 1e241, 1e242, 1e243, 1e244, 1e245, 1e246, 1e247, 1e248, 1e249,
      1e250, 1e251, 1e252, 1e253, 1e254, 1e255, 1e256, 1e257, 1e258, 1e259,
      1e260, 1e261, 1e262, 1e263, 1e264, 1e265, 1e266, 1e267, 1e268, 1e269,
      1e270, 1e271, 1e272, 1e273, 1e274, 1e275, 1e276, 1e277, 1e278, 1e279,
      1e280, 1e281, 1e282, 1e283, 1e284, 1e285, 1e286, 1e287, 1e288, 1e289,
      1e290, 1e291, 1e292, 1e293, 1e294, 1e295, 1e296, 1e297, 1e298, 1e299,
      1e300, 1e301, 1e302, 1e303, 1e304, 1e305, 1e306, 1e307, 1e308};

  // 跳过前导空格
  while (isspace_ascii(*p))
    p++;

  // 处理可选的符号
  int negative = 0;
  switch (*p) {
  case '-':
    negative = 1;
    PD_FALLTHROUGH; // 继续执行以增加位置
  case '+':
    p++;
    // 继续下一步处理
    ```
  break;
}

double number = 0.;
int exponent = 0;
int num_digits = 0;
int num_decimals = 0;

// 处理数字字符串部分。
while (isdigit_ascii(*p)) {
  // 如果数字位数未超过最大限制，则将字符转换为数字并累加到数值中。
  if (num_digits < max_digits) {
    number = number * 10. + (*p - '0');
    num_digits++;
  } else {
    // 超过最大位数限制时增加指数。
    ++exponent;
  }

  p++;
  // 处理千位分隔符（如果存在）后面的字符。
  p += (tsep != '\0' && *p == tsep);
}

// 处理小数部分。
if (*p == decimal) {
  // 如果 maybe_int 不为空，将其设为 0。
  if (maybe_int != NULL)
    *maybe_int = 0;
  p++;

  // 处理小数点后的数字。
  while (num_digits < max_digits && isdigit_ascii(*p)) {
    number = number * 10. + (*p - '0');
    p++;
    num_digits++;
    num_decimals++;
  }

  // 如果数字位数超过最大限制，则跳过额外的小数位数。
  if (num_digits >= max_digits)
    while (isdigit_ascii(*p))
      ++p;

  // 调整指数以考虑小数点位数。
  exponent -= num_decimals;
}

if (num_digits == 0) {
  // 如果没有读取到有效数字，设置错误码并返回 0.0。
  *error = ERANGE;
  return 0.0;
}

// 根据负号修正数值。
if (negative)
  number = -number;

// 处理指数部分。
if (toupper_ascii(*p) == toupper_ascii(sci)) {
  // 如果 maybe_int 不为空，将其设为 0。
  if (maybe_int != NULL)
    *maybe_int = 0;

  // 处理可选的符号。
  negative = 0;
  switch (*++p) {
  case '-':
    negative = 1;
    PD_FALLTHROUGH; // 继续增加位置。
  case '+':
    p++;
    break;
  }

  // 处理指数部分的数字。
  num_digits = 0;
  int n = 0;
  while (num_digits < max_digits && isdigit_ascii(*p)) {
    n = n * 10 + (*p - '0');
    num_digits++;
    p++;
  }

  // 根据符号修正指数值。
  if (negative)
    exponent -= n;
  else
    exponent += n;

  // 如果 'e'/'E' 后没有数字，则取消消耗它。
  if (num_digits == 0)
    p--;
}

if (exponent > 308) {
  // 如果指数超过上限，设置错误码并返回 HUGE_VAL。
  *error = ERANGE;
  return HUGE_VAL;
} else if (exponent > 0) {
  // 如果指数为正数，乘以对应的指数运算值。
  number *= e[exponent];
} else if (exponent < -308) { // 亚正常数
  if (exponent < -616) {      // 防止无效数组访问。
    number = 0.;
  } else {
    number /= e[-308 - exponent];
    number /= e[308];
  }

} else {
  // 如果指数为负数，除以对应的指数运算值。
  number /= e[-exponent];
}

// 如果数值为 HUGE_VAL 或 -HUGE_VAL，则设置错误码。
if (number == HUGE_VAL || number == -HUGE_VAL)
  *error = ERANGE;

if (skip_trailing) {
  // 跳过尾部的空白字符。
  while (isspace_ascii(*p))
    p++;
}

if (endptr)
  // 将结束指针指向当前位置。
  *endptr = (char *)p;
// 返回解析得到的数值。
return number;
/* 结构体初始化函数，初始化一个 uint_state 结构体实例 */
void uint_state_init(uint_state *self) {
    // 将结构体实例中的 seen_sint 字段置为 0
    self->seen_sint = 0;
    // 将结构体实例中的 seen_uint 字段置为 0
    self->seen_uint = 0;
    // 将结构体实例中的 seen_null 字段置为 0
    self->seen_null = 0;
}
int uint64_conflict(uint_state *self) {
  // 返回值是一个布尔表达式，检查是否见过无符号整数并且见过有符号整数或空值
  return self->seen_uint && (self->seen_sint || self->seen_null);
}

int64_t str_to_int64(const char *p_item, int64_t int_min, int64_t int_max,
                     int *error, char tsep) {
  const char *p = p_item;
  // 跳过前导空格。
  while (isspace_ascii(*p)) {
    ++p;
  }

  // 处理符号。
  const bool isneg = *p == '-' ? true : false;
  // 处理符号。
  if (isneg || (*p == '+')) {
    p++;
  }

  // 检查是否有第一个数字。
  if (!isdigit_ascii(*p)) {
    // 错误处理：没有数字。
    *error = ERROR_NO_DIGITS;
    return 0;
  }

  int64_t number = 0;
  if (isneg) {
    // 如果是负数，计算溢出前的最小值。
    int dig_pre_min = -(int_min % 10);
    int64_t pre_min = int_min / 10;

    // 处理数字。
    char d = *p;
    if (tsep != '\0') {
      while (1) {
        if (d == tsep) {
          d = *++p;
          continue;
        } else if (!isdigit_ascii(d)) {
          break;
        }
        // 检查是否溢出。
        if ((number > pre_min) ||
            ((number == pre_min) && (d - '0' <= dig_pre_min))) {
          number = number * 10 - (d - '0');
          d = *++p;
        } else {
          *error = ERROR_OVERFLOW;
          return 0;
        }
      }
    } else {
      while (isdigit_ascii(d)) {
        // 检查是否溢出。
        if ((number > pre_min) ||
            ((number == pre_min) && (d - '0' <= dig_pre_min))) {
          number = number * 10 - (d - '0');
          d = *++p;
        } else {
          *error = ERROR_OVERFLOW;
          return 0;
        }
      }
    }
  } else {
    // 如果是正数，计算溢出前的最大值。
    int64_t pre_max = int_max / 10;
    int dig_pre_max = int_max % 10;

    // 处理数字。
    char d = *p;
    if (tsep != '\0') {
      while (1) {
        if (d == tsep) {
          d = *++p;
          continue;
        } else if (!isdigit_ascii(d)) {
          break;
        }
        // 检查是否溢出。
        if ((number < pre_max) ||
            ((number == pre_max) && (d - '0' <= dig_pre_max))) {
          number = number * 10 + (d - '0');
          d = *++p;
        } else {
          *error = ERROR_OVERFLOW;
          return 0;
        }
      }
    } else {
      while (isdigit_ascii(d)) {
        // 检查是否溢出。
        if ((number < pre_max) ||
            ((number == pre_max) && (d - '0' <= dig_pre_max))) {
          number = number * 10 + (d - '0');
          d = *++p;
        } else {
          *error = ERROR_OVERFLOW;
          return 0;
        }
      }
    }
  }

  // 跳过尾随空格。
  while (isspace_ascii(*p)) {
    ++p;
  }

  // 检查是否使用了所有字符。
  if (*p) {
    *error = ERROR_INVALID_CHARS;
    return 0;
  }

  // 没有错误，设置错误码为0，并返回数字。
  *error = 0;
  return number;
}

uint64_t str_to_uint64(uint_state *state, const char *p_item, int64_t int_max,
                       uint64_t uint_max, int *error, char tsep) {
  const char *p = p_item;
  // 跳过前导空格。
  while (isspace_ascii(*p)) {
    ++p;
  }

  // 处理符号部分。
  if (*p == '-') {
    // 标记已见到有符号整数
    state->seen_sint = 1;
    // 清空错误标志并返回0
    *error = 0;
    return 0;
  } else if (*p == '+') {
    // 跳过正号直接处理数字部分
    p++;
  }

  // 检查第一个数字是否存在。
  if (!isdigit_ascii(*p)) {
    // 如果不是数字，设置错误标志并返回0
    *error = ERROR_NO_DIGITS;
    return 0;
  }

  // 如果数字小于预设最大值 pre_max，则可以继续处理至少一个数字字符而不会溢出。
  //
  // 处理数字部分。
  uint64_t number = 0;
  const uint64_t pre_max = uint_max / 10;
  const uint64_t dig_pre_max = uint_max % 10;
  char d = *p;
  if (tsep != '\0') {
    while (1) {
      if (d == tsep) {
        // 跳过千位分隔符并继续处理下一个字符
        d = *++p;
        continue;
      } else if (!isdigit_ascii(d)) {
        // 如果不是数字，则结束循环
        break;
      }
      if ((number < pre_max) ||
          ((number == pre_max) && ((uint64_t)(d - '0') <= dig_pre_max))) {
        // 将字符转换为数字并添加到当前数字中
        number = number * 10 + (d - '0');
        d = *++p;

      } else {
        // 数字超出范围，设置错误标志并返回0
        *error = ERROR_OVERFLOW;
        return 0;
      }
    }
  } else {
    while (isdigit_ascii(d)) {
      if ((number < pre_max) ||
          ((number == pre_max) && ((uint64_t)(d - '0') <= dig_pre_max))) {
        // 将字符转换为数字并添加到当前数字中
        number = number * 10 + (d - '0');
        d = *++p;

      } else {
        // 数字超出范围，设置错误标志并返回0
        *error = ERROR_OVERFLOW;
        return 0;
      }
    }
  }

  // 跳过尾部空格。
  while (isspace_ascii(*p)) {
    ++p;
  }

  // 检查是否已经使用了所有字符。
  if (*p) {
    // 如果还有剩余字符，设置错误标志并返回0
    *error = ERROR_INVALID_CHARS;
    return 0;
  }

  // 如果数字超过 int_max，则标记已见到无符号整数
  if (number > (uint64_t)int_max) {
    state->seen_uint = 1;
  }

  // 清空错误标志并返回解析得到的数字
  *error = 0;
  return number;
}



# 这是一个单独的右花括号 '}'，用于结束某个代码块或语句块。
```