# `.\numpy\numpy\_core\src\multiarray\stringdtype\static_string.h`

```py
#ifndef NUMPY_CORE_SRC_MULTIARRAY_STATIC_STRING_H_
#define NUMPY_CORE_SRC_MULTIARRAY_STATIC_STRING_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>

// some types used by this header are defined in ndarraytypes.h

// 定义最大字符串长度，使用小字符串优化时需要保留一个字节用于标志
#define NPY_MAX_STRING_SIZE (((int64_t)1 << 8 * (sizeof(size_t) - 1)) - 1)

// 分配器函数的类型定义
typedef void *(*npy_string_malloc_func)(size_t size);
typedef void (*npy_string_free_func)(void *ptr);
typedef void *(*npy_string_realloc_func)(void *ptr, size_t size);

// 使用这些函数创建和销毁字符串分配器。通常用户不会直接使用这些函数，
// 而是使用已经附加到dtype实例的分配器
NPY_NO_EXPORT npy_string_allocator *
NpyString_new_allocator(npy_string_malloc_func m, npy_string_free_func f,
                        npy_string_realloc_func r);

// 释放内部缓冲区和分配器本身
NPY_NO_EXPORT void
NpyString_free_allocator(npy_string_allocator *allocator);

// 为*to_init*分配新的缓冲区，*to_init*在调用此函数之前必须设置为NULL，
// 将新分配的缓冲区填充为*init*中前*size*个条目的复制内容，*init*必须是有效并已初始化的。
// 在现有字符串上调用NpyString_free或将NPY_EMPTY_STRING的内容复制到*to_init*，
// 即可初始化它。不检查*to_init*是否为NULL或内部缓冲区是否为非NULL，
// 如果将此函数传递给未初始化的结构体指针、NULL指针或现有的堆分配字符串，
// 可能会导致未定义行为或内存泄漏。如果分配字符串会超过允许的最大字符串大小或耗尽可用内存，则返回-1。
// 成功时返回0。
NPY_NO_EXPORT int
NpyString_newsize(const char *init, size_t size,
                  npy_packed_static_string *to_init,
                  npy_string_allocator *allocator);

// 清空压缩字符串并释放任何堆分配的数据。对于区域分配的数据，检查数据是否在区域内，
// 如果不在则返回-1。成功时返回0。
NPY_NO_EXPORT int
NpyString_free(npy_packed_static_string *str, npy_string_allocator *allocator);

// 将*in*的内容复制到*out*中。为*out*分配一个新的字符串缓冲区，
// 如果*out*指向现有字符串，则必须在调用此函数之前调用NpyString_free。
// 如果malloc失败则返回-1。成功时返回0。
NPY_NO_EXPORT int
NpyString_dup(const npy_packed_static_string *in,
              npy_packed_static_string *out,
              npy_string_allocator *in_allocator,
              npy_string_allocator *out_allocator);

// 为*out*分配一个新的字符串缓冲区，以存储*size*字节的文本。
// 不执行任何初始化，调用者必须


注释：
// 初始化一个空字符串缓冲区。在调用该函数之前，调用 NpyString_free(*to_init*) 可以释放已有的字符串，
// 或者使用 NPY_EMPTY_STRING 初始化一个新字符串。不检查 *to_init* 是否已经初始化或内部缓冲区是否非空。
// 如果将 NULL 指针传递给该函数或传递一个已在堆上分配的字符串，可能会导致未定义的行为或内存泄漏。
// 返回值：成功返回 0。如果分配字符串会超过最大允许的字符串大小或耗尽可用内存，返回 -1。
NPY_NO_EXPORT int
NpyString_newemptysize(size_t size, npy_packed_static_string *out,
                       npy_string_allocator *allocator);

// 判断 *in* 是否为 null 字符串（例如 NA 对象）。
// 返回值：-1 表示 *in* 无法解包；1 表示 *in* 是 null 字符串；0 表示不是 null 字符串。
NPY_NO_EXPORT int
NpyString_isnull(const npy_packed_static_string *in);

// 比较两个字符串。其语义与使用 strcmp 比较以 null 结尾的 C 字符串 *s1* 和 *s2* 的内容相同。
NPY_NO_EXPORT int
NpyString_cmp(const npy_static_string *s1, const npy_static_string *s2);

// 返回打包字符串中字符串数据的大小。在只需要字符串大小而不需要确定是否为 null 或解包字符串的情况下很有用。
NPY_NO_EXPORT size_t
NpyString_size(const npy_packed_static_string *packed_string);

// 检查两个打包的字符串是否共享内存。
// 返回值：0 表示不共享内存；非零表示共享内存。
NPY_NO_EXPORT int
NpyString_share_memory(const npy_packed_static_string *s1, npy_string_allocator *a1,
                       const npy_packed_static_string *s2, npy_string_allocator *a2);

#ifdef __cplusplus
}
#endif

#endif /* NUMPY_CORE_SRC_MULTIARRAY_STATIC_STRING_H_ */
```