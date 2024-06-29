# `D:\src\scipysrc\pandas\pandas\_libs\include\pandas\vendored\klib\khash.h`

```
// 定义宏，指定许可证位于 LICENSES/KLIB_LICENSE

/*
  一个示例：

#include "khash.h"
KHASH_MAP_INIT_INT(32, char)
int main() {
        int ret, is_missing;
        khiter_t k; // 声明一个哈希表迭代器类型
        khash_t(32) *h = kh_init(32); // 初始化一个哈希表 h，键为整数，值为字符
        k = kh_put(32, h, 5, &ret); // 将键 5 插入哈希表 h 中，并返回对应的迭代器 k，如果存在则返回已有的
        if (!ret) kh_del(32, h, k); // 如果插入不成功（已存在），则删除该键
        kh_value(h, k) = 10; // 设置键 5 对应的值为 10
        k = kh_get(32, h, 10); // 在哈希表 h 中查找键 10，返回对应的迭代器 k
        is_missing = (k == kh_end(h)); // 检查迭代器 k 是否指向哈希表的末尾，表示键 10 不存在
        k = kh_get(32, h, 5); // 再次查找键 5，返回对应的迭代器 k
        kh_del(32, h, k); // 删除键 5 及其对应的值
        for (k = kh_begin(h); k != kh_end(h); ++k) // 遍历哈希表 h 中所有存在的键值对
                if (kh_exist(h, k)) kh_value(h, k) = 1; // 如果键值对存在，则将其值设为 1
        kh_destroy(32, h); // 销毁哈希表 h
        return 0; // 返回主函数结束
}
*/

/*
  2011-09-16 (0.2.6):

        * 容量为 2 的幂。这似乎显著提高了简单键的速度。感谢 Zilong Tan 的建议。参考：

           - https://github.com/stefanocasazza/ULib
           - https://nothings.org/computer/judy/

        * 允许选择使用线性探测，通常对随机输入性能更好。双重散列仍然是默认的，因为对某些非随机输入更为健壮。

        * 添加了 Wang 的整数哈希函数（默认未使用）。此哈希函数对某些非随机输入更为健壮。

  2011-02-14 (0.2.5):

    * 允许声明全局函数。

  2009-09-26 (0.2.4):

    * 提升可移植性。

  2008-09-19 (0.2.3):

        * 修正了示例
        * 改进了接口

  2008-09-11 (0.2.2):

        * 在 kh_put() 中稍微提升了速度

  2008-09-10 (0.2.1):

        * 添加了 kh_clear()
        * 修正了编译错误

  2008-09-02 (0.2.0):

        * 更改为令牌串联，增加了灵活性。

  2008-08-31 (0.1.2):

        * 修复了 kh_get() 中的一个 bug，在此之前未经过测试。

  2008-08-31 (0.1.1):

        * 添加了析构函数
*/

#ifndef __AC_KHASH_H
#define __AC_KHASH_H

/*!
  @header

  通用哈希表库。
 */

#define AC_VERSION_KHASH_H "0.2.6" // 定义头文件版本号

#include <limits.h> // 包含整数限制定义
#include <stdlib.h> // 包含标准库函数定义
#include <string.h> // 包含字符串处理函数定义

// 内存分配器的钩子，使用 C 运行时默认的分配器
#ifndef KHASH_MALLOC
#define KHASH_MALLOC malloc
#endif

#ifndef KHASH_REALLOC
#define KHASH_REALLOC realloc
#endif

#ifndef KHASH_CALLOC
#define KHASH_CALLOC calloc
#endif

#ifndef KHASH_FREE
#define KHASH_FREE free
#endif

#if UINT_MAX == 0xffffffffu
typedef unsigned int khuint32_t;
typedef signed int khint32_t;
#elif ULONG_MAX == 0xffffffffu
typedef unsigned long khuint32_t;
typedef signed long khint32_t;
#endif

#if ULONG_MAX == ULLONG_MAX
typedef unsigned long khuint64_t;
typedef signed long khint64_t;
#else
typedef unsigned long long khuint64_t;
typedef signed long long khint64_t;
#endif

#if UINT_MAX == 0xffffu
typedef unsigned int khuint16_t;
typedef signed int khint16_t;
#elif USHRT_MAX == 0xffffu
typedef unsigned short khuint16_t;
typedef signed short khint16_t;
#endif

#if UCHAR_MAX == 0xffu
typedef unsigned char khuint8_t;
typedef signed char khint8_t;
#endif

typedef double khfloat64_t; // 定义双精度浮点类型 khfloat64_t

#endif // __AC_KHASH_H
// 定义一个单精度浮点类型 khfloat32_t
typedef float khfloat32_t;

// 定义无符号整数类型 khuint32_t，并将其别名为 khuint_t 和 khiter_t
typedef khuint32_t khuint_t;
typedef khuint_t khiter_t;

// 定义宏函数，用于判断指定索引 i 处的标志 flag 是否为空（即未使用）
#define __ac_isempty(flag, i) ((flag[i >> 5] >> (i & 0x1fU)) & 1)

// 定义宏函数，用于判断指定索引 i 处的标志 flag 是否为删除标记
#define __ac_isdel(flag, i) (0)

// 定义宏函数，用于判断指定索引 i 处的标志 flag 是否为空或为删除标记
#define __ac_iseither(flag, i) __ac_isempty(flag, i)

// 定义宏函数，用于将指定索引 i 处的标志 flag 的删除标记设置为 false
#define __ac_set_isdel_false(flag, i) (0)

// 定义宏函数，用于将指定索引 i 处的标志 flag 的空标记设置为 false
#define __ac_set_isempty_false(flag, i) (flag[i >> 5] &= ~(1ul << (i & 0x1fU)))

// 定义宏函数，用于将指定索引 i 处的标志 flag 的空标记设置为 true
#define __ac_set_isempty_true(flag, i) (flag[i >> 5] |= (1ul << (i & 0x1fU)))

// 定义宏函数，用于将指定索引 i 处的标志 flag 的空标记和删除标记均设置为 false
#define __ac_set_isboth_false(flag, i) __ac_set_isempty_false(flag, i)

// 定义宏函数，用于将指定索引 i 处的标志 flag 的删除标记设置为 true
#define __ac_set_isdel_true(flag, i) ((void)0)

// 定义静态内联函数 murmur2_32to32，实现 MurmurHash2 算法的 32 位版本
static inline khuint32_t murmur2_32to32(khuint32_t k) {
    const khuint32_t SEED = 0xc70f6907UL;
    const khuint32_t M_32 = 0x5bd1e995;
    const int R_32 = 24;
    
    khuint32_t h = SEED ^ 4; // 初始化哈希值为固定的种子值
    
    k *= M_32;
    k ^= k >> R_32;
    k *= M_32;
    
    h *= M_32;
    h ^= k;
    
    h ^= h >> 13;
    h *= M_32;
    h ^= h >> 15;
    return h;
}

// 定义静态内联函数 murmur2_32_32to32，实现 MurmurHash2 算法的 32 位版本，适用于两个 32 位整数
static inline khuint32_t murmur2_32_32to32(khuint32_t k1, khuint32_t k2) {
    const khuint32_t SEED = 0xc70f6907UL;
    const khuint32_t M_32 = 0x5bd1e995;
    const int R_32 = 24;
    
    khuint32_t h = SEED ^ 4; // 初始化哈希值为固定的种子值
    
    k1 *= M_32;
    k1 ^= k1 >> R_32;
    k1 *= M_32;
    
    h *= M_32;
    h ^= k1;
    
    k2 *= M_32;
    k2 ^= k2 >> R_32;
    k2 *= M_32;
    
    h *= M_32;
    h ^= k2;
    
    h ^= h >> 13;
    h *= M_32;
    h ^= h >> 15;
    return h;
}

// 定义静态内联函数 murmur2_64to32，将 64 位整数转换为 32 位哈希值，基于 murmur2_32_32to32 实现
static inline khuint32_t murmur2_64to32(khuint64_t k) {
    khuint32_t k1 = (khuint32_t)k;
    khuint32_t k2 = (khuint32_t)(k >> 32);
    
    return murmur2_32_32to32(k1, k2);
}

// 根据是否定义 KHASH_LINEAR 宏选择不同的宏 __ac_inc 的实现方式
#ifdef KHASH_LINEAR
#define __ac_inc(k, m) 1
#else
#define __ac_inc(k, m) (murmur2_32to32(k) | 1) & (m)
#endif

// 定义宏函数 __ac_fsize，根据给定的 m 计算哈希表的大小
#define __ac_fsize(m) ((m) < 32 ? 1 : (m) >> 5)

// 定义常量 __ac_HASH_UPPER，表示哈希表的上限负载因子
static const double __ac_HASH_UPPER = 0.77;
#define KHASH_DECLARE(name, khkey_t, khval_t)                                  \
  typedef struct {                                                             \
    khuint_t n_buckets, size, n_occupied, upper_bound;                         \
    khuint32_t *flags;                                                         \
    khkey_t *keys;                                                             \
    khval_t *vals;                                                             \
  } kh_##name##_t;                                                             \
  extern kh_##name##_t *kh_init_##name();                                      \
  extern void kh_destroy_##name(kh_##name##_t *h);                             \
  extern void kh_clear_##name(kh_##name##_t *h);                               \
  extern khuint_t kh_get_##name(const kh_##name##_t *h, khkey_t key);          \
  extern void kh_resize_##name(kh_##name##_t *h, khuint_t new_n_buckets);      \
  extern khuint_t kh_put_##name(kh_##name##_t *h, khkey_t key, int *ret);      \
  extern void kh_del_##name(kh_##name##_t *h, khuint_t x);



#define KHASH_INIT2(name, SCOPE, khkey_t, khval_t, kh_is_map, __hash_func,     \
                    __hash_equal)                                              \
  typedef struct {                                                             \
    khuint_t n_buckets, size, n_occupied, upper_bound;                         \
    khuint32_t *flags;                                                         \
    khkey_t *keys;                                                             \
    khval_t *vals;                                                             \
  } kh_##name##_t;                                                             \
  SCOPE kh_##name##_t *kh_init_##name(void) {                                  \
    // 分配内存并返回初始化的 kh_##name##_t 结构体指针
    return (kh_##name##_t *)KHASH_CALLOC(1, sizeof(kh_##name##_t));            \
  }                                                                            \
  SCOPE void kh_destroy_##name(kh_##name##_t *h) {                             \
    if (h) {                                                                   \
      // 释放 keys、flags 和 vals 数组内存
      KHASH_FREE(h->keys);                                                     \
      KHASH_FREE(h->flags);                                                    \
      KHASH_FREE(h->vals);                                                     \
      // 最后释放整个 kh_##name##_t 结构体内存
      KHASH_FREE(h);                                                           \
    }                                                                          \
  }                                                                            \
  SCOPE void kh_clear_##name(kh_##name##_t *h) {                               \
    if (h && h->flags) {                                                       \
      // 使用特定值填充 flags 数组
      memset(h->flags, 0xaa, __ac_fsize(h->n_buckets) * sizeof(khuint32_t));   \
      // 重置 size 和 n_occupied
      h->size = h->n_occupied = 0;                                             \
  }                                                                          \
  }                                                                            \
  SCOPE khuint_t kh_get_##name(const kh_##name##_t *h, khkey_t key) {          \
    // 如果哈希表的桶数大于零，则执行以下操作
    if (h->n_buckets) {                                                        \
      // 定义增量、哈希值、当前索引、上一个索引、掩码
      khuint_t inc, k, i, last, mask;                                          \
      mask = h->n_buckets - 1;                                                 \
      k = __hash_func(key);                                                    \
      i = k & mask;                                                            \
      inc = __ac_inc(k, mask);                                                 \
      last = i; /* inc==1 for linear probing */                                \
      // 线性探测查找，直到找到空槽或者找到匹配的键
      while (!__ac_isempty(h->flags, i) &&                                     \
             (__ac_isdel(h->flags, i) || !__hash_equal(h->keys[i], key))) {    \
        i = (i + inc) & mask;                                                  \
        if (i == last)                                                         \
          return h->n_buckets;                                                 \
      }                                                                        \
      // 如果找到空槽，则返回桶的数量；否则返回找到的索引
      return __ac_iseither(h->flags, i) ? h->n_buckets : i;                    \
    } else                                                                     \
      // 如果哈希表没有桶，则返回0
      return 0;                                                                \
  }                                                                            \
  SCOPE void kh_resize_##name(                                                 \
      kh_##name##_t *h,                                                        \
      khuint_t new_n_buckets) { /* This function uses 0.25*n_bucktes bytes of  \
                                   working space instead of                    \
                                   [sizeof(key_t+val_t)+.25]*n_buckets. */     \
    khuint32_t *new_flags = 0;                                                 \
    khuint_t j = 1;                                                            \
    {                                                                          \
      kroundup32(new_n_buckets);                                               \
      // 调整桶数目至最接近的2的幂
      if (new_n_buckets < 4)                                                   \
        // 如果新的桶数小于4，则设为4
        new_n_buckets = 4;                                                     \
      if (h->size >= (khuint_t)(new_n_buckets * __ac_HASH_UPPER + 0.5))        \
        j = 0; /* 请求的大小太小 */                                             \
      else {   /* 要改变的哈希表大小（收缩或扩展）；重新哈希 */                \
        // 为新的标志数组分配内存空间，全部初始化为0xff
        new_flags = (khuint32_t *)KHASH_MALLOC(__ac_fsize(new_n_buckets) *     \
                                               sizeof(khuint32_t));            \
        memset(new_flags, 0xff,                                                \
               __ac_fsize(new_n_buckets) * sizeof(khuint32_t));                \
        if (h->n_buckets < new_n_buckets) { /* 扩展 */                         \
          // 重新分配并扩展键数组的内存空间
          h->keys = (khkey_t *)KHASH_REALLOC(h->keys,                          \
                                             new_n_buckets * sizeof(khkey_t)); \
          // 如果是映射表，重新分配并扩展值数组的内存空间
          if (kh_is_map)                                                       \
            h->vals = (khval_t *)KHASH_REALLOC(h->vals, new_n_buckets *        \
                                                            sizeof(khval_t));  \
        } /* 否则收缩 */                                                      \
      }                                                                        \
    }                                                                          \
    }                                                                          \
  }                                                                            \
  SCOPE khuint_t kh_put_##name(kh_##name##_t *h, khkey_t key, int *ret) {      \
    khuint_t x;                                                                \
    // 如果已占用的桶数超过了上界，则更新哈希表
    if (h->n_occupied >= h->upper_bound) { /* update the hash table */         \
      // 如果桶数大于当前大小的两倍，则缩小哈希表
      if (h->n_buckets > (h->size << 1))                                       \
        kh_resize_##name(h, h->n_buckets - 1); /* 清除 "删除" 元素 */          \
      else                                                                     \
        kh_resize_##name(h, h->n_buckets + 1); /* 扩展哈希表 */                \
    } /* TODO: 实现自动收缩；resize() 已经支持收缩 */                           \
    {                                                                          \
      khuint_t inc, k, i, site, last, mask = h->n_buckets - 1;                 \
      x = site = h->n_buckets;                                                 \
      k = __hash_func(key);                                                    \
      i = k & mask;                                                            \
      if (__ac_isempty(h->flags, i))                                           \
        x = i; /* for speed up */                                              \
      else {                                                                   \
        inc = __ac_inc(k, mask);                                               \
        last = i;                                                              \
        while (!__ac_isempty(h->flags, i) &&                                   \
               (__ac_isdel(h->flags, i) || !__hash_equal(h->keys[i], key))) {  \
          if (__ac_isdel(h->flags, i))                                         \
            site = i;                                                          \
          i = (i + inc) & mask;                                                \
          if (i == last) {                                                     \
            x = site;                                                          \
            break;                                                             \
          }                                                                    \
        }                                                                      \
        if (x == h->n_buckets) {                                               \
          if (__ac_isempty(h->flags, i) && site != h->n_buckets)               \
            x = site;                                                          \
          else                                                                 \
            x = i;                                                             \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    if (__ac_isempty(h->flags, x)) { /* not present at all */                  \
      h->keys[x] = key;                                                        \
      __ac_set_isboth_false(h->flags, x);                                      \
      ++h->size;                                                               \
      ++h->n_occupied;                                                         \
      *ret = 1;                                                                \


注释：


      // 定义变量和初始化，其中 mask 是用于计算哈希桶位置的掩码
      khuint_t inc, k, i, site, last, mask = h->n_buckets - 1;
      // 设置 x 和 site 初始值为哈希桶数，准备记录最终的插入位置
      x = site = h->n_buckets;
      // 计算键的哈希值
      k = __hash_func(key);
      // 计算键在哈希表中的初始位置
      i = k & mask;
      // 如果初始位置为空，则直接将 x 设为 i，以加快查找速度
      if (__ac_isempty(h->flags, i))
        x = i; /* for speed up */
      else {
        // 否则，根据哈希值计算增量 inc
        inc = __ac_inc(k, mask);
        last = i;
        // 开始线性探测，直到找到空位或者找到键的位置
        while (!__ac_isempty(h->flags, i) &&
               (__ac_isdel(h->flags, i) || !__hash_equal(h->keys[i], key))) {
          // 如果遇到已删除的位置，记录下来
          if (__ac_isdel(h->flags, i))
            site = i;
          // 更新 i 的位置
          i = (i + inc) & mask;
          // 如果回到了起始位置 last，则跳出循环
          if (i == last) {
            x = site;
            break;
          }
        }
        // 如果 x 仍未更新，则找到了空位置或者是找到了对应的键位置
        if (x == h->n_buckets) {
          // 如果当前位置为空，并且 site 不是初始值，则将 x 设置为 site
          if (__ac_isempty(h->flags, i) && site != h->n_buckets)
            x = site;
          else
            x = i;
        }
      }
    }
    // 如果找到了空位置 x，则进行插入操作
    if (__ac_isempty(h->flags, x)) { /* not present at all */
      // 将键存储在找到的空位置中
      h->keys[x] = key;
      // 标记该位置不为空且未删除
      __ac_set_isboth_false(h->flags, x);
      // 哈希表大小加一
      ++h->size;
      // 哈希表中被占用的桶数加一
      ++h->n_occupied;
      // 返回插入成功
      *ret = 1;
    } else if (__ac_isdel(h->flags, x)) { /* 如果元素已被标记为删除 */
      h->keys[x] = key;                        /* 将新的键值存入哈希表 */
      __ac_set_isboth_false(h->flags, x);      /* 清除删除标记 */
      ++h->size;                               /* 哈希表大小增加 */
      *ret = 2;                                /* 返回值为2，表示替换已删除元素 */
    } else                                     /* 如果元素已存在且未被删除 */
      *ret = 0; /* 不要修改 h->keys[x] 的内容 */ /* 返回值为0，表示元素已存在且未被删除 */
    return x;                                  /* 返回操作后的索引值 */
  }                                            /* 结束 kh_put_##name 函数定义 */

  SCOPE void kh_del_##name(kh_##name##_t *h, khuint_t x) { /* 删除指定索引处的元素 */
    if (x != h->n_buckets && !__ac_iseither(h->flags, x)) {  /* 如果索引有效且元素未被删除 */
      __ac_set_isdel_true(h->flags, x);                      /* 标记元素为已删除 */
      --h->size;                                             /* 哈希表大小减少 */
    }                                                        /* 结束 if 语句块 */
  }                                                          /* 结束 kh_del_##name 函数定义 */
#define KHASH_INIT(name, khkey_t, khval_t, kh_is_map, __hash_func,             \
                   __hash_equal)                                               \
  KHASH_INIT2(name, static inline, khkey_t, khval_t, kh_is_map, __hash_func,   \
              __hash_equal)

/* --- BEGIN OF HASH FUNCTIONS --- */

/*! @function
  @abstract     Integer hash function
  @param  key   The integer [khuint32_t]
  @return       The hash value [khuint_t]
 */
#define kh_int_hash_func(key) (khuint32_t)(key)

/*! @function
  @abstract     Integer comparison function
 */
#define kh_int_hash_equal(a, b) ((a) == (b))

/*! @function
  @abstract     64-bit integer hash function
  @param  key   The integer [khuint64_t]
  @return       The hash value [khuint_t]
 */
static inline khuint_t kh_int64_hash_func(khuint64_t key) {
  return (khuint_t)((key) >> 33 ^ (key) ^ (key) << 11);
}

/*! @function
  @abstract     64-bit integer comparison function
 */
#define kh_int64_hash_equal(a, b) ((a) == (b))

/*! @function
  @abstract     const char* hash function
  @param  s     Pointer to a null terminated string
  @return       The hash value
 */
static inline khuint_t __ac_X31_hash_string(const char *s) {
  khuint_t h = *s;
  if (h)
    for (++s; *s; ++s)
      h = (h << 5) - h + *s;
  return h;
}

/*! @function
  @abstract     Another interface to const char* hash function
  @param  key   Pointer to a null terminated string [const char*]
  @return       The hash value [khuint_t]
 */
#define kh_str_hash_func(key) __ac_X31_hash_string(key)

/*! @function
  @abstract     Const char* comparison function
 */
#define kh_str_hash_equal(a, b) (strcmp(a, b) == 0)

static inline khuint_t __ac_Wang_hash(khuint_t key) {
  key += ~(key << 15);
  key ^= (key >> 10);
  key += (key << 3);
  key ^= (key >> 6);
  key += ~(key << 11);
  key ^= (key >> 16);
  return key;
}

/*! @function
  @abstract     32-bit integer hash function based on Wang's algorithm
  @param  key   The integer [khuint_t]
  @return       The hash value [khuint_t]
 */
#define kh_int_hash_func2(key) __ac_Wang_hash((khuint_t)key)

/* --- END OF HASH FUNCTIONS --- */

/* Other convenient macros... */

/*!
  @abstract Type of the hash table.
  @param  name  Name of the hash table [symbol]
 */
#define khash_t(name) kh_##name##_t

/*! @function
  @abstract     Initiate a hash table.
  @param  name  Name of the hash table [symbol]
  @return       Pointer to the hash table [khash_t(name)*]
 */
#define kh_init(name) kh_init_##name(void)

/*! @function
  @abstract     Destroy a hash table.
  @param  name  Name of the hash table [symbol]
  @param  h     Pointer to the hash table [khash_t(name)*]
 */
#define kh_destroy(name, h) kh_destroy_##name(h)

/*! @function
  @abstract     Reset a hash table without deallocating memory.
  @param  name  Name of the hash table [symbol]
  @param  h     Pointer to the hash table [khash_t(name)*]
 */
#define kh_clear(name, h) kh_clear_##name(h)

/*! @function
  @abstract     Resize a hash table.
  @param  name  Name of the hash table [symbol]
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  s     New size [khuint_t]
 */
/*! @function
  @abstract     Resize the hash table.
  @param  name  Name of the hash table [symbol]
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  s     New size for the hash table [size_t]
 */
#define kh_resize(name, h, s) kh_resize_##name(h, s)

/*! @function
  @abstract     Insert a key to the hash table.
  @param  name  Name of the hash table [symbol]
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  k     Key [type of keys]
  @param  r     Extra return code: 0 if the key is present in the hash table;
                1 if the bucket is empty (never used); 2 if the element in
                the bucket has been deleted [int*]
  @return       Iterator to the inserted element [khuint_t]
 */
#define kh_put(name, h, k, r) kh_put_##name(h, k, r)

/*! @function
  @abstract     Retrieve a key from the hash table.
  @param  name  Name of the hash table [symbol]
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  k     Key [type of keys]
  @return       Iterator to the found element, or kh_end(h) is the element is
                absent [khuint_t]
 */
#define kh_get(name, h, k) kh_get_##name(h, k)

/*! @function
  @abstract     Remove a key from the hash table.
  @param  name  Name of the hash table [symbol]
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  k     Iterator to the element to be deleted [khuint_t]
 */
#define kh_del(name, h, k) kh_del_##name(h, k)

/*! @function
  @abstract     Test whether a bucket contains data.
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  x     Iterator to the bucket [khuint_t]
  @return       1 if containing data; 0 otherwise [int]
 */
#define kh_exist(h, x) (!__ac_iseither((h)->flags, (x)))

/*! @function
  @abstract     Get key given an iterator
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  x     Iterator to the bucket [khuint_t]
  @return       Key [type of keys]
 */
#define kh_key(h, x) ((h)->keys[x])

/*! @function
  @abstract     Get value given an iterator
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  x     Iterator to the bucket [khuint_t]
  @return       Value [type of values]
  @discussion   For hash sets, calling this results in segfault.
 */
#define kh_val(h, x) ((h)->vals[x])

/*! @function
  @abstract     Alias of kh_val()
 */
#define kh_value(h, x) ((h)->vals[x])

/*! @function
  @abstract     Get the start iterator
  @param  h     Pointer to the hash table [khash_t(name)*]
  @return       The start iterator [khuint_t]
 */
#define kh_begin(h) (khuint_t)(0)

/*! @function
  @abstract     Get the end iterator
  @param  h     Pointer to the hash table [khash_t(name)*]
  @return       The end iterator [khuint_t]
 */
#define kh_end(h) ((h)->n_buckets)

/*! @function
  @abstract     Get the number of elements in the hash table
  @param  h     Pointer to the hash table [khash_t(name)*]
  @return       Number of elements in the hash table [khuint_t]
 */
#define kh_size(h) ((h)->size)
# 定义宏，在哈希表中获取桶的数量
#define kh_n_buckets(h) ((h)->n_buckets)

# 更方便的接口

# 实例化一个包含整数键的哈希集
#define KHASH_SET_INIT_INT(name) KHASH_INIT(name, khint32_t, char, 0, kh_int_hash_func, kh_int_hash_equal)

# 实例化一个包含整数键的哈希映射
#define KHASH_MAP_INIT_INT(name, khval_t) KHASH_INIT(name, khint32_t, khval_t, 1, kh_int_hash_func, kh_int_hash_equal)

# 实例化一个包含无符号 32 位整数键的哈希映射
#define KHASH_MAP_INIT_UINT(name, khval_t) KHASH_INIT(name, khuint32_t, khval_t, 1, kh_int_hash_func, kh_int_hash_equal)

# 实例化一个包含 64 位整数键的哈希集
#define KHASH_SET_INIT_UINT64(name) KHASH_INIT(name, khuint64_t, char, 0, kh_int64_hash_func, kh_int64_hash_equal)

# 实例化一个包含 64 位整数键的哈希集
#define KHASH_SET_INIT_INT64(name) KHASH_INIT(name, khint64_t, char, 0, kh_int64_hash_func, kh_int64_hash_equal)

# 实例化一个包含 64 位整数键的哈希映射
#define KHASH_MAP_INIT_UINT64(name, khval_t) KHASH_INIT(name, khuint64_t, khval_t, 1, kh_int64_hash_func, kh_int64_hash_equal)

# 实例化一个包含 64 位整数键的哈希映射
#define KHASH_MAP_INIT_INT64(name, khval_t) KHASH_INIT(name, khint64_t, khval_t, 1, kh_int64_hash_func, kh_int64_hash_equal)

# 实例化一个包含 16 位整数键的哈希映射
#define KHASH_MAP_INIT_INT16(name, khval_t) KHASH_INIT(name, khint16_t, khval_t, 1, kh_int_hash_func, kh_int_hash_equal)

# 实例化一个包含无符号 16 位整数键的哈希映射
#define KHASH_MAP_INIT_UINT16(name, khval_t) KHASH_INIT(name, khuint16_t, khval_t, 1, kh_int_hash_func, kh_int_hash_equal)

# 实例化一个包含 8 位整数键的哈希映射
#define KHASH_MAP_INIT_INT8(name, khval_t) KHASH_INIT(name, khint8_t, khval_t, 1, kh_int_hash_func, kh_int_hash_equal)
#define KHASH_MAP_INIT_UINT8(name, khval_t)                                    \
  KHASH_INIT(name, khuint8_t, khval_t, 1, kh_int_hash_func, kh_int_hash_equal)


/*! @macro
  @abstract     定义一个使用 uint8_t 键的哈希映射
  @param  name  哈希表的名称 [符号]
  @param  khval_t  值的类型 [类型]
 */



typedef const char *kh_cstr_t;
/*! @function
  @abstract     实例化一个包含 const char* 键的哈希映射
  @param  name  哈希表的名称 [符号]
 */
#define KHASH_SET_INIT_STR(name)                                               \
  KHASH_INIT(name, kh_cstr_t, char, 0, kh_str_hash_func, kh_str_hash_equal)


/*! @macro
  @abstract     定义一个使用 const char* 键的集合型哈希映射
  @param  name  哈希表的名称 [符号]
 */



/*! @function
  @abstract     实例化一个包含 const char* 键的哈希映射
  @param  name  哈希表的名称 [符号]
  @param  khval_t  值的类型 [类型]
 */
#define KHASH_MAP_INIT_STR(name, khval_t)                                      \
  KHASH_INIT(name, kh_cstr_t, khval_t, 1, kh_str_hash_func, kh_str_hash_equal)


/*! @macro
  @abstract     定义一个使用 const char* 键的通用哈希映射
  @param  name  哈希表的名称 [符号]
  @param  khval_t  值的类型 [类型]
 */



#define kh_exist_str(h, k) (kh_exist(h, k))
#define kh_exist_float64(h, k) (kh_exist(h, k))
#define kh_exist_uint64(h, k) (kh_exist(h, k))
#define kh_exist_int64(h, k) (kh_exist(h, k))
#define kh_exist_float32(h, k) (kh_exist(h, k))
#define kh_exist_int32(h, k) (kh_exist(h, k))
#define kh_exist_uint32(h, k) (kh_exist(h, k))
#define kh_exist_int16(h, k) (kh_exist(h, k))
#define kh_exist_uint16(h, k) (kh_exist(h, k))
#define kh_exist_int8(h, k) (kh_exist(h, k))
#define kh_exist_uint8(h, k) (kh_exist(h, k))


/*! @macro
  @abstract     检查特定类型的哈希表中是否存在给定的键
  @param  h     哈希表
  @param  k     键
 */



KHASH_MAP_INIT_STR(str, size_t)
KHASH_MAP_INIT_INT(int32, size_t)
KHASH_MAP_INIT_UINT(uint32, size_t)
KHASH_MAP_INIT_INT64(int64, size_t)
KHASH_MAP_INIT_UINT64(uint64, size_t)
KHASH_MAP_INIT_INT16(int16, size_t)
KHASH_MAP_INIT_UINT16(uint16, size_t)
KHASH_MAP_INIT_INT8(int8, size_t)
KHASH_MAP_INIT_UINT8(uint8, size_t)


// 实例化不同类型的哈希映射，每种类型的键和值类型不同
```