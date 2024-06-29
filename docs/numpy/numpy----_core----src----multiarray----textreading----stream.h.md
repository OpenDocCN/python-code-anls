# `.\numpy\numpy\_core\src\multiarray\textreading\stream.h`

```
#ifndef NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_STREAM_H_
#define NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_STREAM_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * When getting the next line, we hope that the buffer provider can already
 * give some information about the newlines, because for Python iterables
 * we definitely expect to get line-by-line buffers.
 *
 * BUFFER_IS_FILEEND must be returned when the end of the file is reached and
 * must NOT be returned together with a valid (non-empty) buffer.
 */
// 定义常量：缓冲区可能包含换行符
#define BUFFER_MAY_CONTAIN_NEWLINE 0
// 定义常量：缓冲区是行结束
#define BUFFER_IS_LINEND 1
// 定义常量：缓冲区是文件结束
#define BUFFER_IS_FILEEND 2

/*
 * Base struct for streams.  We currently have two, a chunked reader for
 * filelikes and a line-by-line for any iterable.
 * As of writing, the chunked reader was only used for filelikes not already
 * opened.  That is to preserve the amount read in case of an error exactly.
 * If we drop this, we could read it more often (but not when `max_rows` is
 * used).
 *
 * The "streams" can extend this struct to store their own data (so it is
 * a very lightweight "object").
 */
// 定义流的基本结构体，用于处理不同类型的流
typedef struct _stream {
    // 函数指针，用于获取下一个缓冲区
    int (*stream_nextbuf)(void *sdata, char **start, char **end, int *kind);
    // 函数指针，用于关闭流
    int (*stream_close)(struct _stream *strm);
} stream;

// 宏定义：通过函数指针调用stream_nextbuf函数
#define stream_nextbuf(s, start, end, kind)  \
        ((s)->stream_nextbuf((s), start, end, kind))
// 宏定义：通过函数指针调用stream_close函数
#define stream_close(s)    ((s)->stream_close((s)))

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_STREAM_H_ */


这段代码是一个 C 语言的头文件，定义了用于流处理的基本结构体和相关的宏定义。
```