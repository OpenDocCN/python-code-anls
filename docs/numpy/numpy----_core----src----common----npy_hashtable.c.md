# `.\numpy\numpy\_core\src\common\npy_hashtable.c`

```
/*
 * This functionality is designed specifically for the ufunc machinery to
 * dispatch based on multiple DTypes. Since this is designed to be used
 * as purely a cache, it currently does no reference counting.
 * Even though this is a cache, there is currently no maximum size. It may
 * make sense to limit the size, or count collisions: If too many collisions
 * occur, we could grow the cache, otherwise, just replace an old item that
 * was presumably not used for a long time.
 *
 * If a different part of NumPy requires a custom hashtable, the code should
 * be reused with care since specializing it more for the ufunc dispatching
 * case is likely desired.
 */

#include "templ_common.h"
#include "npy_hashtable.h"

#if SIZEOF_PY_UHASH_T > 4
#define _NpyHASH_XXPRIME_1 ((Py_uhash_t)11400714785074694791ULL)
#define _NpyHASH_XXPRIME_2 ((Py_uhash_t)14029467366897019727ULL)
#define _NpyHASH_XXPRIME_5 ((Py_uhash_t)2870177450012600261ULL)
#define _NpyHASH_XXROTATE(x) ((x << 31) | (x >> 33))  /* Rotate left 31 bits */
#else
#define _NpyHASH_XXPRIME_1 ((Py_uhash_t)2654435761UL)
#define _NpyHASH_XXPRIME_2 ((Py_uhash_t)2246822519UL)
#define _NpyHASH_XXPRIME_5 ((Py_uhash_t)374761393UL)
#define _NpyHASH_XXROTATE(x) ((x << 13) | (x >> 19))  /* Rotate left 13 bits */
#endif

#ifdef Py_GIL_DISABLED
// TODO: replace with PyMutex when it is public
#define LOCK_TABLE(tb)                                      \
    if (!PyThread_acquire_lock(tb->mutex, NOWAIT_LOCK)) {    \
        PyThread_acquire_lock(tb->mutex, WAIT_LOCK);         \
    }
#define UNLOCK_TABLE(tb) PyThread_release_lock(tb->mutex);
#define INITIALIZE_LOCK(tb)                     \
    tb->mutex = PyThread_allocate_lock();       \
    if (tb->mutex == NULL) {                    \
        PyErr_NoMemory();                       \
        PyMem_Free(res);                        \
        return NULL;                            \
    }
#define FREE_LOCK(tb)                           \
    if (tb->mutex != NULL) {                    \
        PyThread_free_lock(tb->mutex);          \
    }
#else
// the GIL serializes access to the table so no need
// for locking if it is enabled
#define LOCK_TABLE(tb)
#define UNLOCK_TABLE(tb)
#define INITIALIZE_LOCK(tb)
#define FREE_LOCK(tb)
#endif

/*
 * This hashing function is basically the Python tuple hash with the type
 * identity hash inlined. The tuple hash itself is a reduced version of xxHash.
 *
 * Users cannot control pointers, so we do not have to worry about DoS attacks?
 */
static inline Py_hash_t
identity_list_hash(PyObject *const *v, int len)
{
    Py_uhash_t acc = _NpyHASH_XXPRIME_5;
    for (int i = 0; i < len; i++) {
        /*
         * Lane is the single item hash, which for us is the rotated pointer.
         * Identical to the python type hash (pointers end with 0s normally).
         */
        // 从数组 v 中取出第 i 个元素，并转换为 size_t 类型
        size_t y = (size_t)v[i];
        // 根据 y 的值进行位操作，生成一个 Py_uhash_t 类型的 lane
        Py_uhash_t lane = (y >> 4) | (y << (8 * SIZEOF_VOID_P - 4));
        // 将 lane 乘以 _NpyHASH_XXPRIME_2 并加到 acc 上
        acc += lane * _NpyHASH_XXPRIME_2;
        // 使用 _NpyHASH_XXROTATE 函数对 acc 进行旋转操作
        acc = _NpyHASH_XXROTATE(acc);
        // 将 acc 乘以 _NpyHASH_XXPRIME_1
        acc *= _NpyHASH_XXPRIME_1;
    }
    // 返回最终的累加结果作为哈希值
    return acc;
}
#undef _NpyHASH_XXPRIME_1
#undef _NpyHASH_XXPRIME_2
#undef _NpyHASH_XXPRIME_5
#undef _NpyHASH_XXROTATE

/*
 * The following functions are internal utility functions for handling
 * hash table operations and memory management.
 */

static inline PyObject **
find_item(PyArrayIdentityHash const *tb, PyObject *const *key)
{
    // 计算给定键的哈希值
    Py_hash_t hash = identity_list_hash(key, tb->key_len);
    npy_uintp perturb = (npy_uintp)hash;
    npy_intp bucket;
    npy_intp mask = tb->size - 1 ;
    PyObject **item;

    // 计算哈希桶的索引位置
    bucket = (npy_intp)hash & mask;
    while (1) {
        // 获取当前桶的位置
        item = &(tb->buckets[bucket * (tb->key_len + 1)]);

        // 如果当前桶为空，则返回这个空桶
        if (item[0] == NULL) {
            /* The item is not in the cache; return the empty bucket */
            return item;
        }
        // 如果当前桶中的键与给定键相等，则返回这个桶
        if (memcmp(item+1, key, tb->key_len * sizeof(PyObject *)) == 0) {
            /* This is a match, so return the item/bucket */
            return item;
        }
        // 处理哈希冲突，按照 Python 的方式进行 perturb 操作
        perturb >>= 5;  /* Python uses the macro PERTURB_SHIFT == 5 */
        bucket = mask & (bucket * 5 + perturb + 1);
    }
}

/*
 * Allocate and initialize a new PyArrayIdentityHash object with a given key length.
 */
NPY_NO_EXPORT PyArrayIdentityHash *
PyArrayIdentityHash_New(int key_len)
{
    PyArrayIdentityHash *res = PyMem_Malloc(sizeof(PyArrayIdentityHash));
    if (res == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    assert(key_len > 0);
    res->key_len = key_len;
    res->size = 4;  /* Start with a size of 4 */
    res->nelem = 0;

    // 初始化锁（假设这里有一个初始化锁的宏或函数）
    INITIALIZE_LOCK(res);

    // 分配并初始化桶数组
    res->buckets = PyMem_Calloc(4 * (key_len + 1), sizeof(PyObject *));
    if (res->buckets == NULL) {
        PyErr_NoMemory();
        PyMem_Free(res);
        return NULL;
    }
    return res;
}

/*
 * Deallocate memory associated with a PyArrayIdentityHash object.
 */
NPY_NO_EXPORT void
PyArrayIdentityHash_Dealloc(PyArrayIdentityHash *tb)
{
    // 释放桶数组的内存
    PyMem_Free(tb->buckets);
    // 释放锁资源
    FREE_LOCK(tb);
    // 释放 PyArrayIdentityHash 结构体的内存
    PyMem_Free(tb);
}

/*
 * Resize the hash table if necessary based on its current usage.
 */
static int
_resize_if_necessary(PyArrayIdentityHash *tb)
{
    npy_intp new_size, prev_size = tb->size;
    PyObject **old_table = tb->buckets;
    assert(prev_size > 0);

    // 如果当前桶的使用超过了其容量的一半，则扩大容量为原来的两倍
    if ((tb->nelem + 1) * 2 > prev_size) {
        /* Double in size */
        new_size = prev_size * 2;
    }
    else {
        // 否则根据一定策略缩小容量，以避免频繁的扩展和收缩
        new_size = prev_size;
        while ((tb->nelem + 8) * 2 < new_size / 2) {
            /*
             * Should possibly be improved.  However, we assume that we
             * almost never shrink.  Still if we do, do not shrink as much
             * as possible to avoid growing right away.
             */
            new_size /= 2;
        }
        assert(new_size >= 4);
    }
    // 如果新的大小与之前的大小相同，则无需调整
    if (new_size == prev_size) {
        return 0;
    }

    npy_intp alloc_size;
    // 计算需要分配的新桶数组的大小
    if (npy_mul_sizes_with_overflow(&alloc_size, new_size, tb->key_len + 1)) {
        return -1;
    }
    // 分配新的桶数组内存空间
    tb->buckets = PyMem_Calloc(alloc_size, sizeof(PyObject *));
    if (tb->buckets == NULL) {
        // 如果分配失败，回滚操作，并报内存错误
        tb->buckets = old_table;
        PyErr_NoMemory();
        return -1;
    }

    // 更新哈希表的大小
    tb->size = new_size;


注释：
    // 遍历旧哈希表中的每一个槽位
    for (npy_intp i = 0; i < prev_size; i++) {
        // 获取当前槽位的指针，每个槽位存储了键值对的数组
        PyObject **item = &old_table[i * (tb->key_len + 1)];
        // 如果当前槽位不为空（即有键值对存在）
        if (item[0] != NULL) {
            // 在新哈希表中查找当前键的位置
            PyObject **tb_item = find_item(tb, item + 1);
            // 将旧哈希表中的键值对复制到新哈希表中对应位置
            tb_item[0] = item[0];  // 复制键
            memcpy(tb_item + 1, item + 1, tb->key_len * sizeof(PyObject *));  // 复制值
        }
    }
    // 释放旧哈希表的内存空间
    PyMem_Free(old_table);
    // 返回操作成功的标志
    return 0;
/**
 * Add an item to the identity cache. The storage location must not change
 * unless the cache is cleared.
 *
 * @param tb The mapping.
 * @param key The key, must be a C-array of pointers of the length
 *            corresponding to the mapping.
 * @param value Normally a Python object, no reference counting is done.
 *              Use NULL to clear an item. If the item does not exist, no
 *              action is performed for NULL.
 * @param replace If 1, allow replacements. If replace is 0 an error is raised
 *                if the stored value is different from the value to be cached.
 *                If the value to be cached is identical to the stored value,
 *                the value to be cached is ignored and no error is raised.
 * @returns 0 on success, -1 with a MemoryError or RuntimeError (if an item
 *          is added which is already in the cache and replace is 0). The
 *          caller should avoid the RuntimeError.
 */
NPY_NO_EXPORT int
PyArrayIdentityHash_SetItem(PyArrayIdentityHash *tb,
                            PyObject *const *key, PyObject *value, int replace)
{
    // 对表格进行加锁，确保线程安全
    LOCK_TABLE(tb);
    
    // 如果传入的 value 不为空，并且如果需要的话，进行必要时的表格调整（缩小）
    if (value != NULL && _resize_if_necessary(tb) < 0) {
        /* Shrink, only if a new value is added. */
        // 如果添加新值时需要收缩，则解锁表格并返回错误
        UNLOCK_TABLE(tb);
        return -1;
    }

    // 在表格中查找指定 key 对应的项目
    PyObject **tb_item = find_item(tb, key);

    // 如果 value 不为空
    if (value != NULL) {
        // 如果表格中已经存在相同 key 的项，并且不允许替换，则解锁表格并返回运行时错误
        if (tb_item[0] != NULL && tb_item[0] != value && !replace) {
            UNLOCK_TABLE(tb);
            PyErr_SetString(PyExc_RuntimeError,
                            "Identity cache already includes an item with this key.");
            return -1;
        }
        // 将 value 存储到表格中对应的位置
        tb_item[0] = value;
        // 将 key 复制到表格的后续位置
        memcpy(tb_item+1, key, tb->key_len * sizeof(PyObject *));
        // 增加表格中元素的数量计数
        tb->nelem += 1;
    }
    else {
        /* Clear the bucket -- just the value should be enough though. */
        // 清空存储桶，通常只清空值即可
        memset(tb_item, 0, (tb->key_len + 1) * sizeof(PyObject *));
    }

    // 解锁表格
    UNLOCK_TABLE(tb);
    return 0;
}

/**
 * Retrieve an item from the identity cache.
 *
 * @param tb The mapping.
 * @param key The key to look up, must be a C-array of pointers of the length
 *            corresponding to the mapping.
 * @returns The stored value associated with the key, or NULL if not found.
 */
NPY_NO_EXPORT PyObject *
PyArrayIdentityHash_GetItem(PyArrayIdentityHash const *tb, PyObject *const *key)
{
    // 对表格进行加锁，确保线程安全
    LOCK_TABLE(tb);
    // 查找并返回与指定 key 对应的存储值
    PyObject *res = find_item(tb, key)[0];
    // 解锁表格
    UNLOCK_TABLE(tb);
    return res;
}
```