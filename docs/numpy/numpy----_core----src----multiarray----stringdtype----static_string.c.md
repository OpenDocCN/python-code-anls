# `.\numpy\numpy\_core\src\multiarray\stringdtype\static_string.c`

```py
/*
 * Static string API
 *
 * Strings can be stored in multiple ways. Initialization leaves them as
 * either initialized or missing, with initialization being inside an arena
 * except for short strings that fit inside the static string struct stored in
 * the array buffer (<=15 or 7 bytes, depending on architecture).
 *
 * If a string is replaced, it will be allocated on the heap if it cannot fit
 * inside the original short string or arena allocation. If a string is set
 * to missing, the information about the previous allocation is kept, so
 * replacement with a new string can use the possible previous arena
 * allocation. Note that after replacement with a short string, any arena
 * information is lost, so a later replacement with a longer one will always
 * be on the heap.
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

// work around Python 3.10 and earlier issue, see
// the commit message of 82fd2b8 for more details
// also needed for the allocator mutex
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdint.h>
#include <string.h>
#include <stdarg.h>

#include "numpy/ndarraytypes.h"
#include "numpy/npy_2_compat.h"
#include "numpy/arrayobject.h"
#include "static_string.h"
#include "dtypemeta.h"

#if NPY_BYTE_ORDER == NPY_LITTLE_ENDIAN

// the last and hence highest byte in vstring.size is reserved for flags
//
// SSSS SSSF

typedef struct _npy_static_vstring_t {
    size_t offset;
    size_t size_and_flags;
} _npy_static_vstring_t;

typedef struct _short_string_buffer {
    char buf[sizeof(_npy_static_vstring_t) - 1];
    unsigned char size_and_flags;
} _short_string_buffer;

#elif NPY_BYTE_ORDER == NPY_BIG_ENDIAN

// the first and hence highest byte in vstring.size is reserved for flags
//
// FSSS SSSS

typedef struct _npy_static_vstring_t {
    size_t size_and_flags;
    size_t offset;
} _npy_static_vstring_t;

typedef struct _short_string_buffer {
    unsigned char size_and_flags;
    char buf[sizeof(_npy_static_vstring_t) - 1];
} _short_string_buffer;

#endif

typedef union _npy_static_string_u {
    _npy_static_vstring_t vstring;
    _short_string_buffer direct_buffer;
} _npy_static_string_u;


// Flags defining whether a string exists and how it is stored.
// Inside the arena, long means not medium (i.e., >255 bytes), while
// outside, it means not short (i.e., >15 or 7, depending on arch).

// set for null strings representing missing data
#define NPY_STRING_MISSING 0x80        // 1000 0000
// set after an array entry is initialized for the first time
#define NPY_STRING_INITIALIZED 0x40    // 0100 0000
// The string data is managed by malloc/free on the heap
// Only set for strings that have been mutated to be longer
// than the original entry
#define NPY_STRING_OUTSIDE_ARENA 0x20  // 0010 0000
// A string that lives in the arena with a size longer
// than 255 bytes, so the size in the arena is stored in a size_t
#define NPY_STRING_LONG 0x10           // 0001 0000
// 定义一个掩码，用于提取字符串标志字节中未使用的最后四位
#define NPY_STRING_FLAG_MASK 0xF0      // 1111 0000
// 定义一个掩码，用于提取短字符串的大小，适合于4位整数
#define NPY_SHORT_STRING_SIZE_MASK 0x0F  // 0000 1111
// NPY_SHORT_STRING_MAX_SIZE根据系统架构确定，表示短字符串的最大长度
#define NPY_SHORT_STRING_MAX_SIZE \
    (sizeof(npy_static_string) - 1)      // 15 or 7 depending on arch
// NPY_MEDIUM_STRING_MAX_SIZE定义中等字符串的最大长度
#define NPY_MEDIUM_STRING_MAX_SIZE 0xFF  // 255

// 定义宏，用于从字符串结构中获取其标志位
#define STRING_FLAGS(string) \
    (((_npy_static_string_u *)string)->direct_buffer.size_and_flags & NPY_STRING_FLAG_MASK)
// 定义宏，用于从字符串结构中获取短字符串的大小
#define SHORT_STRING_SIZE(string) \
    (string->direct_buffer.size_and_flags & NPY_SHORT_STRING_SIZE_MASK)
// 定义掩码，用于提取高位字节
#define HIGH_BYTE_MASK ((size_t)0XFF << 8 * (sizeof(size_t) - 1))
// 定义宏，用于获取可变长度字符串的大小
#define VSTRING_SIZE(string) (string->vstring.size_and_flags & ~HIGH_BYTE_MASK)

// 空字符串结构的静态实例，用于表示未初始化的、中等大小的字符串
static const _npy_static_string_u empty_string_u = {
        .direct_buffer = {.size_and_flags = 0, .buf = {0}}};

// 函数：判断给定的静态字符串是否为短字符串
static int
is_short_string(const npy_packed_static_string *s)
{
    // 检查字符串的标志是否表明它是一个初始化并且在区域外的短字符串
    return STRING_FLAGS(s) == (NPY_STRING_INITIALIZED | NPY_STRING_OUTSIDE_ARENA);
}

// 结构体：定义字符串内存分配器的状态
typedef struct npy_string_arena {
    size_t cursor;  // 当前位置指针
    size_t size;    // 缓冲区大小
    char *buffer;   // 缓冲区指针
} npy_string_arena;

// 结构体：定义字符串分配器的具体实现
struct npy_string_allocator {
    npy_string_malloc_func malloc;          // 内存分配函数指针
    npy_string_free_func free;              // 内存释放函数指针
    npy_string_realloc_func realloc;        // 内存重新分配函数指针
    npy_string_arena arena;                 // 字符串区域
    PyThread_type_lock *allocator_lock;     // 分配器锁指针
};

// 函数：设置可变长度字符串的大小
static void
set_vstring_size(_npy_static_string_u *str, size_t size)
{
    unsigned char current_flags = str->direct_buffer.size_and_flags;
    str->vstring.size_and_flags = size;  // 设置可变长度字符串的大小
    str->direct_buffer.size_and_flags = current_flags;  // 保持原有标志位
}

// 函数：获取可变长度字符串的缓冲区指针
static char *
vstring_buffer(npy_string_arena *arena, _npy_static_string_u *string)
{
    // 如果字符串标志表明其在区域外，则直接返回偏移量作为缓冲区指针
    if (STRING_FLAGS(string) & NPY_STRING_OUTSIDE_ARENA) {
        return (char *)string->vstring.offset;
    }
    // 如果字符串在区域内但是缓冲区为空，则返回空指针
    if (arena->buffer == NULL) {
        return NULL;
    }
    // 否则，返回缓冲区起始地址加上偏移量作为缓冲区指针
    return (char *)((size_t)arena->buffer + string->vstring.offset);
}

// 宏：定义缓冲区扩展因子为1.25倍
#define ARENA_EXPAND_FACTOR 1.25

// 函数：在字符串区域中进行内存分配
static char *
arena_malloc(npy_string_arena *arena, npy_string_realloc_func r, size_t size)
{
    // 计算实际存储空间大小，需要额外一个size_t来存储分配的大小信息
    size_t string_storage_size;
    if (size <= NPY_MEDIUM_STRING_MAX_SIZE) {
        string_storage_size = size + sizeof(unsigned char);
    }
    else {
        string_storage_size = size + sizeof(size_t);
    }
    // 检查剩余的空间是否足够存储新数据，若不足则进行重新分配
    if ((arena->size - arena->cursor) <= string_storage_size) {
        // 重新分配缓冲区，确保有足够的空间
        // 初始猜测是将缓冲区的大小加倍
        size_t newsize;
        if (arena->size == 0) {
            newsize = string_storage_size;
        }
        else if (((ARENA_EXPAND_FACTOR * arena->size) - arena->cursor) >
                 string_storage_size) {
            newsize = ARENA_EXPAND_FACTOR * arena->size;
        }
        else {
            newsize = arena->size + string_storage_size;
        }
        if ((arena->cursor + size) >= newsize) {
            // 如果需要比扩展因子更多的额外空间，留出一些填充
            newsize = ARENA_EXPAND_FACTOR * (arena->cursor + size);
        }
        // 调用 realloc 函数重新分配内存，若传入的旧缓冲区为 NULL，则行为类似 malloc
        char *newbuf = r(arena->buffer, newsize);
        if (newbuf == NULL) {
            return NULL;
        }
        // 将新分配的部分清零，以防止出现未初始化的内存
        memset(newbuf + arena->cursor, 0, newsize - arena->cursor);
        arena->buffer = newbuf;
        arena->size = newsize;
    }
    // 根据字符串大小选择存储方式
    char *ret;
    if (size <= NPY_MEDIUM_STRING_MAX_SIZE) {
        // 对于较小的字符串，将字符串大小存储在缓冲区中，并返回字符串数据的指针
        unsigned char *size_loc =
                (unsigned char *)&arena->buffer[arena->cursor];
        *size_loc = size;
        ret = &arena->buffer[arena->cursor + sizeof(char)];
    }
    else {
        // 对于较大的字符串，直接将大小值存储在缓冲区中，并返回字符串数据的指针
        char *size_ptr = (char *)&arena->buffer[arena->cursor];
        memcpy(size_ptr, &size, sizeof(size_t));
        ret = &arena->buffer[arena->cursor + sizeof(size_t)];
    }
    // 更新游标位置，指向下一个可用空间的起始位置
    arena->cursor += string_storage_size;
    // 返回指向存储字符串数据的指针
    return ret;
static int
arena_free(npy_string_arena *arena, _npy_static_string_u *str)
{
    // 计算字符串大小
    size_t size = VSTRING_SIZE(str);
    // 断言字符串大小应大于 0
    assert (size > 0);
    // 断言 arena 不为空
    assert (!(arena->size == 0 && arena->cursor == 0 && arena->buffer == NULL));

    // 获取字符串的指针
    char *ptr = vstring_buffer(arena, str);
    // 如果指针为空，返回错误
    if (ptr == NULL) {
        return -1;
    }

    // 将 arena 的起始地址、ptr 和结束地址进行 uintptr_t 类型转换
    uintptr_t buf_start = (uintptr_t)arena->buffer;
    uintptr_t ptr_loc = (uintptr_t)ptr;
    uintptr_t end_loc = ptr_loc + size;
    uintptr_t buf_end = buf_start + arena->size;
    
    // 检查指针和地址范围是否有效
    if (ptr_loc < buf_start || ptr_loc > buf_end || end_loc > buf_end) {
        return -1;
    }

    // 使用 0 填充字符串内容
    memset(ptr, 0, size);

    // 返回成功
    return 0;
}

static npy_string_arena NEW_ARENA = {0, 0, NULL};

NPY_NO_EXPORT npy_string_allocator *
NpyString_new_allocator(npy_string_malloc_func m, npy_string_free_func f,
                        npy_string_realloc_func r)
{
    // 分配内存以存储 npy_string_allocator 结构
    npy_string_allocator *allocator = m(sizeof(npy_string_allocator));
    if (allocator == NULL) {
        return NULL;
    }
    // 分配线程锁
    PyThread_type_lock *allocator_lock = PyThread_allocate_lock();
    if (allocator_lock == NULL) {
        // 如果分配线程锁失败，释放已分配的 allocator 内存，设置错误信息并返回 NULL
        f(allocator);
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate thread lock");
        return NULL;
    }
    // 初始化 allocator 结构
    allocator->malloc = m;
    allocator->free = f;
    allocator->realloc = r;
    // 初始化 arena 为空的 NEW_ARENA
    allocator->arena = NEW_ARENA;
    // 分配的线程锁赋值给 allocator 的 allocator_lock
    allocator->allocator_lock = allocator_lock;

    // 返回初始化后的 allocator
    return allocator;
}

NPY_NO_EXPORT void
NpyString_free_allocator(npy_string_allocator *allocator)
{
    // 获取 allocator 的 free 函数指针
    npy_string_free_func f = allocator->free;

    // 如果 arena 的 buffer 不为空，释放其内存
    if (allocator->arena.buffer != NULL) {
        f(allocator->arena.buffer);
    }
    // 如果 allocator 的 allocator_lock 不为空，释放其内存
    if (allocator->allocator_lock != NULL) {
        PyThread_free_lock(allocator->allocator_lock);
    }

    // 释放 allocator 结构自身的内存
    f(allocator);
}

/*NUMPY_API
 * 获取与 *descr* 相关联的 allocator 的互斥锁。
 *
 * 必须对此函数返回的 allocator 调用一次 NpyString_release_allocator。
 *
 * 注意，获取 allocator 互斥锁期间，不应调用需要 GIL 的函数，否则可能导致死锁。
 */
NPY_NO_EXPORT npy_string_allocator *
NpyString_acquire_allocator(const PyArray_StringDTypeObject *descr)
{
    // 尝试获取 allocator 的互斥锁，如果不能立即获得，则等待获取
    if (!PyThread_acquire_lock(descr->allocator->allocator_lock, NOWAIT_LOCK)) {
        PyThread_acquire_lock(descr->allocator->allocator_lock, WAIT_LOCK);
    }
    // 返回 descr 所关联的 allocator
    return descr->allocator;
}
/*NUMPY_API
 * Simultaneously acquire the mutexes locking the allocators attached to
 * multiple descriptors.
 *
 * Writes a pointer to the associated allocator in the allocators array for
 * each StringDType descriptor in the array. If any of the descriptors are not
 * StringDType instances, write NULL to the allocators array for that entry.
 *
 * *n_descriptors* is the number of descriptors in the descrs array that
 * should be examined. Any descriptor after *n_descriptors* elements is
 * ignored. A buffer overflow will happen if the *descrs* array does not
 * contain n_descriptors elements.
 *
 * If pointers to the same descriptor are passed multiple times, only acquires
 * the allocator mutex once but sets identical allocator pointers appropriately.
 *
 * The allocator mutexes must be released after this function returns, see
 * NpyString_release_allocators.
 *
 * Note that functions requiring the GIL should not be called while the
 * allocator mutex is held, as doing so may cause deadlocks.
 */
NPY_NO_EXPORT void
NpyString_acquire_allocators(size_t n_descriptors,
                             PyArray_Descr *const descrs[],
                             npy_string_allocator *allocators[])
{
    for (size_t i=0; i<n_descriptors; i++) {
        // 检查描述符是否为 StringDType 类型
        if (NPY_DTYPE(descrs[i]) != &PyArray_StringDType) {
            // 若不是，则分配器指针置为 NULL
            allocators[i] = NULL;
            continue;
        }
        int allocators_match = 0;
        // 检查之前的描述符是否已经处理过相同的分配器
        for (size_t j=0; j<i; j++) {
            if (allocators[j] == NULL) {
                continue;
            }
            // 如果找到相同的分配器，则复用之前的分配器指针
            if (((PyArray_StringDTypeObject *)descrs[i])->allocator ==
                ((PyArray_StringDTypeObject *)descrs[j])->allocator)
            {
                allocators[i] = allocators[j];
                allocators_match = 1;
                break;
            }
        }
        // 如果未找到相同的分配器，则获取新的分配器，并更新分配器指针
        if (!allocators_match) {
            allocators[i] = NpyString_acquire_allocator(
                    (PyArray_StringDTypeObject *)descrs[i]);
        }
    }
}

/*NUMPY_API
 * Release the mutex locking an allocator. This must be called exactly once
 * after acquiring the allocator mutex and all operations requiring the
 * allocator are done.
 *
 * If you need to release multiple allocators, see
 * NpyString_release_allocators, which can correctly handle releasing the
 * allocator once when given several references to the same allocator.
 */
NPY_NO_EXPORT void
NpyString_release_allocator(npy_string_allocator *allocator)
{
    // 释放单个分配器的互斥锁
    PyThread_release_lock(allocator->allocator_lock);
}

/*NUMPY_API
 * Release the mutexes locking N allocators.
 *
 * *length* is the length of the allocators array. NULL entries are ignored.
 *
 * If pointers to the same allocator are passed multiple times, only releases
 * the allocator mutex once.
 */
NPY_NO_EXPORT void
NpyString_release_allocators(size_t length, npy_string_allocator *allocators[])
{
    # 遍历从 0 到 length-1 的索引 i
    for (size_t i=0; i<length; i++) {
        # 如果 allocators[i] 是 NULL，则跳过当前循环，继续下一个 i
        if (allocators[i] == NULL) {
            continue;
        }
        # 初始化 matches 变量为 0，用于标记是否存在与 allocators[i] 相同的先前分配器
        int matches = 0;
        # 再次遍历从 0 到 i-1 的索引 j
        for (size_t j=0; j<i; j++) {
            # 如果 allocators[i] 等于 allocators[j]，则设置 matches 为 1，并跳出循环
            if (allocators[i] == allocators[j]) {
                matches = 1;
                break;
            }
        }
        # 如果 matches 为 0，表示未找到与 allocators[i] 相同的先前分配器
        if (!matches) {
            # 调用 NpyString_release_allocator 函数释放 allocators[i] 分配的资源
            NpyString_release_allocator(allocators[i]);
        }
    }
// Helper for allocating strings that will live on the heap or in the arena
// buffer. Determines whether this is a newly allocated array and the string
// should be appended to an existing arena buffer, new data for an existing
// arena string that is being mutated, or new data for an existing short
// string that is being mutated
static char *
heap_or_arena_allocate(npy_string_allocator *allocator,
                       _npy_static_string_u *to_init_u, size_t size,
                       int *on_heap)
{
    // 获取直接缓冲区的大小和标志位
    unsigned char *flags = &to_init_u->direct_buffer.size_and_flags;
    if (!(*flags & NPY_STRING_OUTSIDE_ARENA)) {
        // 如果标志位中不包含 NPY_STRING_OUTSIDE_ARENA，则表示使用 Arena 分配或重新分配内存。

        npy_string_arena *arena = &allocator->arena;
        // 获取分配器中的 Arena 对象的引用

        if (arena == NULL) {
            return NULL;
        }
        // 如果 Arena 为空，则返回空指针

        if (*flags == 0) {
            // 如果标志位为 0，表示字符串尚未分配，因此需要添加到现有的 Arena 分配中

            char *ret = arena_malloc(arena, allocator->realloc, sizeof(char) * size);
            // 在 Arena 中分配大小为 sizeof(char) * size 的内存块

            if (size < NPY_MEDIUM_STRING_MAX_SIZE) {
                *flags = NPY_STRING_INITIALIZED;
            }
            else {
                *flags = NPY_STRING_INITIALIZED | NPY_STRING_LONG;
            }
            // 设置标志位，表示字符串已初始化，并根据大小设置是否为长字符串

            return ret;
        }

        // 字符串已经在 Arena 中分配，检查是否仍有空间。
        // 大小信息存储在分配的内存块的开头之前。

        char *buf = vstring_buffer(arena, to_init_u);
        // 获取 Arena 中要初始化的字符串的缓冲区

        if (buf == NULL) {
            return NULL;
        }
        // 如果缓冲区为空，则返回空指针

        size_t alloc_size;
        if (*flags & NPY_STRING_LONG) {
            // 对于长字符串，大小信息可能不是内存对齐的，因此使用 memcpy 复制大小信息

            size_t *size_loc = (size_t *)((uintptr_t)buf - sizeof(size_t));
            memcpy(&alloc_size, size_loc, sizeof(size_t));
        }
        else {
            // 中等长度的字符串大小存储在一个 char 中，因此可以直接访问

            alloc_size = (size_t) * (unsigned char *)(buf - 1);
        }

        if (size <= alloc_size) {
            // 如果请求的大小小于等于分配的大小，则表示有足够的空间，直接返回缓冲区

            return buf;
        }

        // 没有足够的空间，需要通过堆分配。
    }

    // 在堆上分配
    *on_heap = 1;
    // 设置标志位，表示字符串在堆上分配
    *flags = NPY_STRING_INITIALIZED | NPY_STRING_OUTSIDE_ARENA | NPY_STRING_LONG;
    // 设置标志位，表示字符串已初始化、在 Arena 外部分配、为长字符串
    return allocator->malloc(sizeof(char) * size);
    // 使用分配器在堆上分配大小为 sizeof(char) * size 的内存块
static int
heap_or_arena_deallocate(npy_string_allocator *allocator,
                         _npy_static_string_u *str_u)
{
    // 确保字符串不为空
    assert (VSTRING_SIZE(str_u) > 0); // should not get here with empty string.

    if (STRING_FLAGS(str_u) & NPY_STRING_OUTSIDE_ARENA) {
        // 如果字符串存储在堆上（而非内存池中），则需要使用 free() 释放
        // 对于堆上的字符串，偏移量是一个原始地址，所以这里的类型转换是安全的。
        allocator->free((char *)str_u->vstring.offset);
        str_u->vstring.offset = 0;
    }
    else {
        // 字符串存储在内存池中
        npy_string_arena *arena = &allocator->arena;
        if (arena == NULL) {
            return -1;
        }
        // 调用 arena_free 函数释放内存池中的字符串
        if (arena_free(arena, str_u) < 0) {
            return -1;
        }
    }
    return 0;
}

// 一个普通的空字符串只是一个大小为 0 的短字符串。但如果一个字符串已经在内存池中初始化，
// 我们只需将 vstring 的大小设置为 0，这样如果字符串再次被重置，我们仍然可以使用内存池。
NPY_NO_EXPORT int
NpyString_pack_empty(npy_packed_static_string *out)
{
    _npy_static_string_u *out_u = (_npy_static_string_u *)out;
    unsigned char *flags = &out_u->direct_buffer.size_and_flags;
    if (*flags & NPY_STRING_OUTSIDE_ARENA) {
        // 这也将短字符串的大小设置为 0。
        *flags = NPY_STRING_INITIALIZED | NPY_STRING_OUTSIDE_ARENA;
    }
    else {
        // 调用 set_vstring_size 函数将 vstring 的大小设置为 0
        set_vstring_size(out_u, 0);
    }
    return 0;
}

NPY_NO_EXPORT int
NpyString_newemptysize(size_t size, npy_packed_static_string *out,
                       npy_string_allocator *allocator)
{
    if (size == 0) {
        return NpyString_pack_empty(out);
    }
    if (size > NPY_MAX_STRING_SIZE) {
        return -1;
    }

    _npy_static_string_u *out_u = (_npy_static_string_u *)out;

    if (size > NPY_SHORT_STRING_MAX_SIZE) {
        // 大于 NPY_SHORT_STRING_MAX_SIZE 的字符串存储在堆上或内存池中
        int on_heap = 0;
        char *buf = heap_or_arena_allocate(allocator, out_u, size, &on_heap);

        if (buf == NULL) {
            return -1;
        }

        if (on_heap) {
            // 如果存储在堆上，则将偏移量设置为 buf 的地址
            out_u->vstring.offset = (size_t)buf;
        }
        else {
            // 如果存储在内存池中，则将偏移量设置为 buf 相对于内存池缓冲区的偏移量
            npy_string_arena *arena = &allocator->arena;
            if (arena == NULL) {
                return -1;
            }
            out_u->vstring.offset = (size_t)buf - (size_t)arena->buffer;
        }
        // 调用 set_vstring_size 函数设置 vstring 的大小为 size
        set_vstring_size(out_u, size);
    }
    else {
        // 大小不超过 CPU 架构的限制（7 或 15），直接设置短字符串的大小和标志
        out_u->direct_buffer.size_and_flags =
            NPY_STRING_INITIALIZED | NPY_STRING_OUTSIDE_ARENA | size;
    }

    return 0;
}
/*NUMPY_API
 * 创建一个新的 npy_packed_static_string，将指定的初始化字符串复制到其中
 *
 * 初始化一个长度为 size 的静态字符串结构，并将 init 指向的字符串复制到其中。
 * 如果初始化失败或 size 为 0，则返回 -1；否则返回 0。
*/
NpyString_newsize(const char *init, size_t size,
                  npy_packed_static_string *to_init,
                  npy_string_allocator *allocator)
{
    // 尝试创建指定大小的空字符串结构，如果失败则返回 -1
    if (NpyString_newemptysize(size, to_init, allocator) < 0) {
        return -1;
    }

    // 如果 size 为 0，则直接返回成功
    if (size == 0) {
        return 0;
    }

    // 转换 to_init 到 _npy_static_string_u 指针类型
    _npy_static_string_u *to_init_u = ((_npy_static_string_u *)to_init);

    // 缓冲区指针，根据字符串大小选择合适的缓冲区
    char *buf = NULL;
    if (size > NPY_SHORT_STRING_MAX_SIZE) {
        // 对于大于最大短字符串限制的字符串，从 allocator 的 arena 中获取缓冲区
        buf = vstring_buffer(&allocator->arena, to_init_u);
    }
    else {
        // 否则使用直接缓冲区
        buf = to_init_u->direct_buffer.buf;
    }

    // 将 init 指向的字符串复制到 buf 缓冲区中
    memcpy(buf, init, size);

    return 0;
}

NPY_NO_EXPORT int
NpyString_free(npy_packed_static_string *str, npy_string_allocator *allocator)
{
    // 将 str 转换为 _npy_static_string_u 指针类型
    _npy_static_string_u *str_u = (_npy_static_string_u *)str;
    // 获取直接缓冲区的标志位指针
    unsigned char *flags = &str_u->direct_buffer.size_and_flags;
    
    // 无条件地移除指示某些内容缺失的标志位
    // 对于这种情况，字符串应该已经被释放，但为了安全起见，我们仍然进行检查
    *flags &= ~NPY_STRING_MISSING;

    // 如果是短字符串，且字符串大小大于 0
    if (is_short_string(str)) {
        if (SHORT_STRING_SIZE(str_u) > 0) {
            // 清零缓冲区，并设置标志位表示初始化完成的空字符串
            memcpy(str_u, &empty_string_u, sizeof(_npy_static_string_u));
            *flags = NPY_STRING_OUTSIDE_ARENA | NPY_STRING_INITIALIZED;
        }
    }
    else { // 如果是长字符串
        if (VSTRING_SIZE(str_u) > 0) {
            // 释放字符串，并将大小设置为 0 表示已释放
            if (heap_or_arena_deallocate(allocator, str_u) < 0) {
                return -1;
            }
            set_vstring_size(str_u, 0);
        }
    }
    return 0;
}

/*NUMPY_API
 * 将空字符串打包到 npy_packed_static_string 中
 *
 * 将空字符串打包到 packed_string 中。成功返回 0，失败返回 -1。
*/
NPY_NO_EXPORT int
NpyString_pack_null(npy_string_allocator *allocator,
                    npy_packed_static_string *packed_string)
{
    // 将 packed_string 中的字符串释放
    _npy_static_string_u *str_u = (_npy_static_string_u *)packed_string;
    if (NpyString_free(packed_string, allocator) < 0) {
        return -1;
    }
    
    // 保留标志位，因为允许变异，所以需要关联原始分配的元数据
    str_u->direct_buffer.size_and_flags |= NPY_STRING_MISSING;
    return 0;
}

NPY_NO_EXPORT int
NpyString_dup(const npy_packed_static_string *in,
              npy_packed_static_string *out,
              npy_string_allocator *in_allocator,
              npy_string_allocator *out_allocator)
{
    // 如果输入字符串为 NULL，则将输出字符串打包为 NULL
    if (NpyString_isnull(in)) {
        return NpyString_pack_null(out_allocator, out);
    }
    
    // 获取输入字符串的大小
    size_t size = NpyString_size(in);
    
    // 如果大小为 0，则将输出字符串打包为空字符串
    if (size == 0) {
        return NpyString_pack_empty(out);
    }

    // 省略部分代码...
}
    // 如果输入字符串是短字符串，则直接将输入复制到输出，返回 0
    if (is_short_string(in)) {
        memcpy(out, in, sizeof(_npy_static_string_u));
        return 0;
    }
    // 将输入转换为 _npy_static_string_u 指针类型
    _npy_static_string_u *in_u = (_npy_static_string_u *)in;
    // 初始化输入缓冲为 NULL
    char *in_buf = NULL;
    // 获取输入分配器的字符串竞技场
    npy_string_arena *arena = &in_allocator->arena;
    // 用于标记是否使用了 malloc 进行内存分配
    int used_malloc = 0;
    // 如果输入和输出的分配器相同且输入不是短字符串
    if (in_allocator == out_allocator && !is_short_string(in)) {
        // 使用输入分配器分配指定大小的内存
        in_buf = in_allocator->malloc(size);
        // 将 arena 中的数据复制到输入缓冲区中
        memcpy(in_buf, vstring_buffer(arena, in_u), size);
        // 标记使用了 malloc 进行内存分配
        used_malloc = 1;
    }
    // 否则直接将 arena 中的数据复制到输入缓冲区中
    else {
        in_buf = vstring_buffer(arena, in_u);
    }
    // 调用 NpyString_newsize 函数将输入缓冲区内容转换为指定大小的输出内容，并返回结果
    int ret = NpyString_newsize(in_buf, size, out, out_allocator);
    // 如果之前使用了 malloc 分配内存，则使用输入分配器释放输入缓冲区内存
    if (used_malloc) {
        in_allocator->free(in_buf);
    }
    // 返回转换结果
    return ret;
}

# 检查静态字符串是否为 NULL
NPY_NO_EXPORT int
NpyString_isnull(const npy_packed_static_string *s)
{
    # 检查字符串的标志位是否包含 NPY_STRING_MISSING 标志
    return (STRING_FLAGS(s) & NPY_STRING_MISSING) == NPY_STRING_MISSING;
}

# 比较两个静态字符串的内容
NPY_NO_EXPORT int
NpyString_cmp(const npy_static_string *s1, const npy_static_string *s2)
{
    # 计算两个字符串的最小长度
    size_t minsize = s1->size < s2->size ? s1->size : s2->size;

    int cmp = 0;

    if (minsize != 0) {
        # 使用 strncmp 比较两个字符串的前 minsize 个字符
        cmp = strncmp(s1->buf, s2->buf, minsize);
    }

    if (cmp == 0) {
        # 如果前 minsize 个字符相同，则根据字符串长度进行比较
        if (s1->size > minsize) {
            return 1;  # s1 比 s2 长
        }
        if (s2->size > minsize) {
            return -1;  # s2 比 s1 长
        }
    }

    return cmp;  # 返回比较结果
}

# 返回静态字符串的大小
NPY_NO_EXPORT size_t
NpyString_size(const npy_packed_static_string *packed_string)
{
    # 如果字符串为空，返回大小为 0
    if (NpyString_isnull(packed_string)) {
        return 0;
    }

    _npy_static_string_u *string_u = (_npy_static_string_u *)packed_string;

    # 如果是短字符串，返回直接缓冲区大小
    if (is_short_string(packed_string)) {
        return string_u->direct_buffer.size_and_flags &
               NPY_SHORT_STRING_SIZE_MASK;
    }

    # 否则返回可变字符串的大小
    return VSTRING_SIZE(string_u);
}

/*NUMPY_API
 * Pack a string buffer into a npy_packed_static_string
 *
 * Copy and pack the first *size* entries of the buffer pointed to by *buf*
 * into the *packed_string*. Returns 0 on success and -1 on failure.
*/
# 将字符串缓冲区打包到 npy_packed_static_string 中
NPY_NO_EXPORT int
NpyString_pack(npy_string_allocator *allocator,
               npy_packed_static_string *packed_string, const char *buf,
               size_t size)
{
    # 如果已经有数据，则释放之前的数据
    if (NpyString_free(packed_string, allocator) < 0) {
        return -1;
    }
    # 调用 NpyString_newsize 来设置新的数据和大小
    return NpyString_newsize(buf, size, packed_string, allocator);
}

# 检查两个静态字符串是否共享内存
NPY_NO_EXPORT int
NpyString_share_memory(const npy_packed_static_string *s1, npy_string_allocator *a1,
                       const npy_packed_static_string *s2, npy_string_allocator *a2) {
    # 如果分配器不同或者任一字符串为空或者为短字符串，则返回 0
    if (a1 != a2 ||
        is_short_string(s1) || is_short_string(s2) ||
        NpyString_isnull(s1) || NpyString_isnull(s2)) {
        return 0;
    }

    _npy_static_string_u *s1_u = (_npy_static_string_u *)s1;
    _npy_static_string_u *s2_u = (_npy_static_string_u *)s2;

    # 比较两个字符串的缓冲区是否相同
    if (vstring_buffer(&a1->arena, s1_u) == vstring_buffer(&a2->arena, s2_u))
    {
        return 1;  # 共享内存
    }
    return 0;  # 不共享内存
}
```