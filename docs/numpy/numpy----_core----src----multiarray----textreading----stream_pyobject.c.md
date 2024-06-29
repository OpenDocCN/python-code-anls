# `.\numpy\numpy\_core\src\multiarray\textreading\stream_pyobject.c`

```
/*
 * C side structures to provide capabilities to read Python file like objects
 * in chunks, or iterate through iterables with each result representing a
 * single line of a file.
 */

/*
 * Cleans the typedef before including Python.h to avoid potential conflicts
 * with legacy code.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

/*
 * Includes necessary headers for standard I/O operations and memory allocation.
 */
#include <stdio.h>
#include <stdlib.h>

/*
 * Ensures compatibility with the latest NumPy API.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"

/*
 * Includes custom header for stream operations related to text reading.
 */
#include "textreading/stream.h"

/*
 * Defines the chunk size for reading operations.
 */
#define READ_CHUNKSIZE 1 << 14

/*
 * Structure representing a file reading context in Python with chunked
 * operations.
 */
typedef struct {
    stream stream;          /* Custom stream structure */
    PyObject *file;         /* Python file object being read */
    PyObject *read;         /* `read` attribute of the file object */
    PyObject *chunksize;    /* Amount to read each time from `obj.read()` */
    PyObject *chunk;        /* Most recently read line from the file */
    const char *encoding;   /* Encoding compatible with PyUnicode_Encode */
} python_chunks_from_file;


/*
 * Helper function to process Python string-like objects, supporting both
 * byte objects and Unicode strings.
 *
 * NOTE: Steals a reference to `str` (although usually returns it unmodified).
 */
static inline PyObject *
process_stringlike(PyObject *str, const char *encoding)
{
    if (PyBytes_Check(str)) {
        PyObject *ustr;
        /* Converts byte object to Unicode using specified encoding */
        ustr = PyUnicode_FromEncodedObject(str, encoding, NULL);
        if (ustr == NULL) {
            return NULL;
        }
        Py_DECREF(str);
        return ustr;
    }
    else if (!PyUnicode_Check(str)) {
        PyErr_SetString(PyExc_TypeError,
                "non-string returned while reading data");
        Py_DECREF(str);
        return NULL;
    }
    return str;
}


/*
 * Retrieves buffer information from a Unicode string object, setting
 * start and end pointers and kind of Unicode data (1-byte, 2-byte, or 4-byte).
 */
static inline void
buffer_info_from_unicode(PyObject *str, char **start, char **end, int *kind)
{
    Py_ssize_t length = PyUnicode_GET_LENGTH(str);
    *kind = PyUnicode_KIND(str);

    /* Determines the type of Unicode data and sets appropriate pointers */
    if (*kind == PyUnicode_1BYTE_KIND) {
        *start = (char *)PyUnicode_1BYTE_DATA(str);
    }
    else if (*kind == PyUnicode_2BYTE_KIND) {
        *start = (char *)PyUnicode_2BYTE_DATA(str);
        length *= sizeof(Py_UCS2);
    }
    else if (*kind == PyUnicode_4BYTE_KIND) {
        *start = (char *)PyUnicode_4BYTE_DATA(str);
        length *= sizeof(Py_UCS4);
    }
    *end = *start + length;
}


/*
 * Retrieves the next buffer from the Python file object in chunks,
 * updating start and end pointers and kind of Unicode data.
 */
static int
fb_nextbuf(python_chunks_from_file *fb, char **start, char **end, int *kind)
{
    /* Release any previous chunk read */
    Py_XDECREF(fb->chunk);
    fb->chunk = NULL;

    /* Calls `read` method of the file object to fetch the next chunk */
    PyObject *chunk = PyObject_CallFunctionObjArgs(fb->read, fb->chunksize, NULL);
    if (chunk == NULL) {
        return -1;  /* Returns error if unable to read */
    }
    /* Processes the retrieved chunk as a string-like object */
    fb->chunk = process_stringlike(chunk, fb->encoding);
    if (fb->chunk == NULL) {
        return -1;  /* Returns error if processing fails */
    }
    /* Retrieves buffer information from the processed Unicode chunk */
    buffer_info_from_unicode(fb->chunk, start, end, kind);
    if (*start == *end) {
        return BUFFER_IS_FILEEND;  /* Indicates end of file */
    }
    return BUFFER_MAY_CONTAIN_NEWLINE;  /* Indicates more data available */
}


/*
 * Deletes the allocated resources associated with the file reading context.
 */
static int
fb_del(stream *strm)
{
    python_chunks_from_file *fb = (python_chunks_from_file *)strm;

    /* Releases Python objects and memory allocated for the context */
    Py_XDECREF(fb->file);
    Py_XDECREF(fb->read);
    Py_XDECREF(fb->chunksize);
    Py_XDECREF(fb->chunk);

    /* Frees the memory allocated for the stream structure */
    PyMem_FREE(strm);

    /* Returns success status */
    return 0;
}
    # 返回整数值 0，结束函数并返回调用者
    return 0;
/*
 * 从 Python 文件对象中创建流
 */
NPY_NO_EXPORT stream *
stream_python_file(PyObject *obj, const char *encoding)
{
    // 分配内存以存储 python_chunks_from_file 结构体
    python_chunks_from_file *fb;
    
    // 使用 PyMem_Calloc 分配内存，如果失败则抛出内存错误并返回空
    fb = (python_chunks_from_file *)PyMem_Calloc(1, sizeof(python_chunks_from_file));
    if (fb == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    
    // 设置流的下一个缓冲区函数为 fb_nextbuf
    fb->stream.stream_nextbuf = (void *)&fb_nextbuf;
    // 设置流的关闭函数为 fb_del
    fb->stream.stream_close = &fb_del;
    
    // 设置编码方式
    fb->encoding = encoding;
    // 增加 Python 文件对象的引用计数
    Py_INCREF(obj);
    // 将 Python 文件对象赋值给结构体中的 file 成员
    fb->file = obj;
    
    // 获取 Python 文件对象的 read 方法
    fb->read = PyObject_GetAttrString(obj, "read");
    if (fb->read == NULL) {
        // 如果获取失败，则跳转到失败标签进行清理并返回空
        goto fail;
    }
    // 设置 chunksize 为 READ_CHUNKSIZE 的 Python 长整型对象
    fb->chunksize = PyLong_FromLong(READ_CHUNKSIZE);
    if (fb->chunksize == NULL) {
        // 如果创建失败，则跳转到失败标签进行清理并返回空
        goto fail;
    }
    
    // 成功则返回文件流结构体的指针转换为流类型的指针
    return (stream *)fb;

fail:
    // 失败时清理结构体内存并返回空
    fb_del((stream *)fb);
    return NULL;
}

/*
 * 从 Python 可迭代对象中创建流，将每个项目解释为文件中的一行
 */
typedef struct {
    stream stream;
    // 正在读取的 Python 文件对象
    PyObject *iterator;
    
    // 最近获取的 Python str 对象行
    PyObject *line;
    
    // 与 Python 的 PyUnicode_Encode 兼容的编码（可以为空）
    const char *encoding;
} python_lines_from_iterator;

/*
 * 清理迭代器流
 */
static int
it_del(stream *strm)
{
    // 将流类型强制转换为 python_lines_from_iterator 类型
    python_lines_from_iterator *it = (python_lines_from_iterator *)strm;
    
    // 释放迭代器和行对象的引用
    Py_XDECREF(it->iterator);
    Py_XDECREF(it->line);
    
    // 释放流的内存
    PyMem_FREE(strm);
    return 0;
}

/*
 * 获取下一个缓冲区的数据
 */
static int
it_nextbuf(python_lines_from_iterator *it, char **start, char **end, int *kind)
{
    // 清理之前的行对象引用
    Py_XDECREF(it->line);
    it->line = NULL;
    
    // 从迭代器获取下一行对象
    PyObject *line = PyIter_Next(it->iterator);
    if (line == NULL) {
        // 如果获取失败且有错误发生，则返回 -1
        if (PyErr_Occurred()) {
            return -1;
        }
        // 否则设置开始和结束为 NULL，表示文件结束
        *start = NULL;
        *end = NULL;
        return BUFFER_IS_FILEEND;
    }
    // 处理行对象，并使用给定编码转换
    it->line = process_stringlike(line, it->encoding);
    if (it->line == NULL) {
        return -1;
    }
    
    // 将 Unicode 编码的字符串转换为缓冲区信息
    buffer_info_from_unicode(it->line, start, end, kind);
    return BUFFER_IS_LINEND;
}

/*
 * 从 Python 可迭代对象中创建流
 */
NPY_NO_EXPORT stream *
stream_python_iterable(PyObject *obj, const char *encoding)
{
    // 分配内存以存储 python_lines_from_iterator 结构体
    python_lines_from_iterator *it;
    
    // 检查对象是否为可迭代对象，否则设置类型错误并返回空
    if (!PyIter_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                "error reading from object, expected an iterable.");
        return NULL;
    }
    
    // 分配内存以存储迭代器流结构体
    it = (python_lines_from_iterator *)PyMem_Calloc(1, sizeof(*it));
    if (it == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    
    // 设置流的下一个缓冲区函数为 it_nextbuf
    it->stream.stream_nextbuf = (void *)&it_nextbuf;
    // 设置流的关闭函数为 it_del
    it->stream.stream_close = &it_del;
    
    // 设置编码方式
    it->encoding = encoding;
    // 增加 Python 可迭代对象的引用计数
    Py_INCREF(obj);
    // 将 Python 可迭代对象赋值给结构体中的 iterator 成员
    it->iterator = obj;
    
    // 成功则返回可迭代器流结构体的指针转换为流类型的指针
    return (stream *)it;
}
```