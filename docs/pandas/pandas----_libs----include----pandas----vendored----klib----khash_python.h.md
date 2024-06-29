# `D:\src\scipysrc\pandas\pandas\_libs\include\pandas\vendored\klib\khash_python.h`

```
// 注释：声明常量，表示零哈希和NaN哈希的值
#define ZERO_HASH 0
#define NAN_HASH 0

// 注释：定义一个静态内联函数，用于计算双精度浮点数的哈希值
static inline khuint32_t kh_float64_hash_func(double val) {
    // 注释：如果值为正零或负零，它们应具有相同的哈希值
    if (val == 0.0) {
        return ZERO_HASH;
    }
    // 注释：如果值为NaN（Not a Number），它们应具有相同的哈希值
    if (val != val) {
        // （这里没有提供下一部分代码的注释，因为在这个代码块里面）
        // The remaining part of the function would handle the hash value for NaN.
        // All NaN values should map to the same hash value defined by NAN_HASH.
        return NAN_HASH;
    }
    // 其他情况，采用murmur2-hash或其他哈希策略计算哈希值
    // 这里可能会包含具体的哈希计算逻辑
}


这段代码是用C语言编写的哈希函数，用于计算双精度浮点数的哈希值。注释中解释了处理零值和NaN值的特殊情况，并提到了使用哈希策略来处理其他情况。
    return NAN_HASH;
  }


    // 如果输入值不是数字，则返回预定义的 NAN_HASH 值
    return NAN_HASH;
  }



  khuint64_t as_int = asuint64(val);
  return murmur2_64to32(as_int);


  // 将输入值转换为 uint64_t 类型
  khuint64_t as_int = asuint64(val);
  // 使用 MurmurHash2 算法将 64 位整数哈希为 32 位整数并返回结果
  return murmur2_64to32(as_int);
}

// 定义静态内联函数，用于计算 float32 类型的哈希值
static inline khuint32_t kh_float32_hash_func(float val) {
    // 处理特殊情况：0.0 和 -0.0 应具有相同的哈希值
    if (val == 0.0f) {
        return ZERO_HASH;
    }
    // 处理特殊情况：所有的 NaN 应该具有相同的哈希值
    if (val != val) {
        return NAN_HASH;
    }
    // 将 float 类型的值转换为 uint32，并应用 murmur2_32to32 哈希算法
    khuint32_t as_int = asuint32(val);
    return murmur2_32to32(as_int);
}

// 定义宏，用于比较 float 类型的键值是否相等
#define kh_floats_hash_equal(a, b) ((a) == (b) || ((b) != (b) && (a) != (a)))

// 定义宏，初始化一个 float64 类型的哈希表
#define KHASH_MAP_INIT_FLOAT64(name, khval_t)                                  \
  KHASH_INIT(name, khfloat64_t, khval_t, 1, kh_float64_hash_func,              \
             kh_floats_hash_equal)

// 初始化一个 float64 类型的哈希表实例
KHASH_MAP_INIT_FLOAT64(float64, size_t)

// 定义宏，初始化一个 float32 类型的哈希表
#define KHASH_MAP_INIT_FLOAT32(name, khval_t)                                  \
  KHASH_INIT(name, khfloat32_t, khval_t, 1, kh_float32_hash_func,              \
             kh_floats_hash_equal)

// 初始化一个 float32 类型的哈希表实例
KHASH_MAP_INIT_FLOAT32(float32, size_t)

// 定义静态内联函数，用于计算 complex128 类型的哈希值
static inline khint32_t kh_complex128_hash_func(khcomplex128_t val) {
    // 将 complex128 类型的实部和虚部分别进行哈希计算，使用 XOR 操作符组合结果
    return kh_float64_hash_func(val.real) ^ kh_float64_hash_func(val.imag);
}

// 定义静态内联函数，用于计算 complex64 类型的哈希值
static inline khint32_t kh_complex64_hash_func(khcomplex64_t val) {
    // 将 complex64 类型的实部和虚部分别进行哈希计算，使用 XOR 操作符组合结果
    return kh_float32_hash_func(val.real) ^ kh_float32_hash_func(val.imag);
}

// 定义宏，用于比较 complex 类型的键值是否相等
#define kh_complex_hash_equal(a, b)                                            \
  (kh_floats_hash_equal(a.real, b.real) && kh_floats_hash_equal(a.imag, b.imag))

// 定义宏，初始化一个 complex64 类型的哈希表
#define KHASH_MAP_INIT_COMPLEX64(name, khval_t)                                \
  KHASH_INIT(name, khcomplex64_t, khval_t, 1, kh_complex64_hash_func,          \
             kh_complex_hash_equal)

// 初始化一个 complex64 类型的哈希表实例
KHASH_MAP_INIT_COMPLEX64(complex64, size_t)

// 定义宏，初始化一个 complex128 类型的哈希表
#define KHASH_MAP_INIT_COMPLEX128(name, khval_t)                               \
  KHASH_INIT(name, khcomplex128_t, khval_t, 1, kh_complex128_hash_func,        \
             kh_complex_hash_equal)

// 初始化一个 complex128 类型的哈希表实例
KHASH_MAP_INIT_COMPLEX128(complex128, size_t)

// 定义宏，检查 complex64 类型的哈希表中是否存在指定键
#define kh_exist_complex64(h, k) (kh_exist(h, k))

// 定义宏，检查 complex128 类型的哈希表中是否存在指定键
#define kh_exist_complex128(h, k) (kh_exist(h, k))

// 定义静态内联函数，用于比较两个 PyFloatObject 对象的值是否相等
static inline int floatobject_cmp(PyFloatObject *a, PyFloatObject *b) {
    // 比较两个 PyFloatObject 对象是否都是 NaN，或者它们的值相等
    return (isnan(PyFloat_AS_DOUBLE(a)) && isnan(PyFloat_AS_DOUBLE(b))) ||
           (PyFloat_AS_DOUBLE(a) == PyFloat_AS_DOUBLE(b));
}

// 定义静态内联函数，用于比较两个 PyComplexObject 对象的值是否相等
static inline int complexobject_cmp(PyComplexObject *a, PyComplexObject *b) {
    // 比较两个 PyComplexObject 对象的实部和虚部是否都是 NaN 或者相等
    return (isnan(a->cval.real) && isnan(b->cval.real) &&
            isnan(a->cval.imag) && isnan(b->cval.imag)) ||
           (isnan(a->cval.real) && isnan(b->cval.real) &&
            a->cval.imag == b->cval.imag) ||
           (a->cval.real == b->cval.real &&
            isnan(a->cval.imag) && isnan(b->cval.imag)) ||
           (a->cval.real == b->cval.real &&
            a->cval.imag == b->cval.imag);
}

// 声明静态内联函数，用于比较两个 PyObject 对象的值是否相等
static inline int pyobject_cmp(PyObject *a, PyObject *b);
// see GH 41836
// 定义一个静态内联函数，用于比较两个 PyTupleObject 对象
static inline int tupleobject_cmp(PyTupleObject *a, PyTupleObject *b) {
  Py_ssize_t i;

  // 检查两个元组对象的大小是否相同
  if (Py_SIZE(a) != Py_SIZE(b)) {
    return 0;
  }

  // 遍历元组对象中的每个元素进行比较
  for (i = 0; i < Py_SIZE(a); ++i) {
    // 调用 pyobject_cmp 函数比较元组中的每个元素
    if (!pyobject_cmp(PyTuple_GET_ITEM(a, i), PyTuple_GET_ITEM(b, i))) {
      return 0;
    }
  }
  // 如果所有元素都相等，则返回 1 表示相等
  return 1;
}

// 定义一个静态内联函数，用于比较两个 PyObject 对象
static inline int pyobject_cmp(PyObject *a, PyObject *b) {
  // 如果两个对象指针相同，则直接返回 1 表示相等
  if (a == b) {
    return 1;
  }

  // 检查两个对象的类型是否相同
  if (Py_TYPE(a) == Py_TYPE(b)) {
    // 对于一些内置类型的特殊处理，这些类型可能包含 NaN（不是数字），
    // 我们希望它们被视为相等，但通常的 PyObject_RichCompareBool 会返回 False
    if (PyFloat_CheckExact(a)) {
      // 如果是浮点数对象，则调用 floatobject_cmp 函数进行比较
      return floatobject_cmp((PyFloatObject *)a, (PyFloatObject *)b);
    }
    if (PyComplex_CheckExact(a)) {
      // 如果是复数对象，则调用 complexobject_cmp 函数进行比较
      return complexobject_cmp((PyComplexObject *)a, (PyComplexObject *)b);
    }
    if (PyTuple_CheckExact(a)) {
      // 如果是元组对象，则调用 tupleobject_cmp 函数进行比较
      return tupleobject_cmp((PyTupleObject *)a, (PyTupleObject *)b);
    }
    // 目前不支持 frozenset
  }

  // 使用 PyObject_RichCompareBool 函数比较对象，使用 Py_EQ 表示相等
  int result = PyObject_RichCompareBool(a, b, Py_EQ);
  // 如果比较出错（result < 0），则清除异常并返回 0
  if (result < 0) {
    PyErr_Clear();
    return 0;
  }
  // 返回比较结果
  return result;
}

// 定义一个静态内联函数，用于计算 Python 浮点数对象的哈希值
static inline Py_hash_t _Pandas_HashDouble(double val) {
  // Python 3.10 之后，NaN 不再具有哈希值 0
  if (isnan(val)) {
    return 0;
  }
  // 根据 Python 版本选择 _Py_HashDouble 函数的调用方式
#if PY_VERSION_HEX < 0x030A0000
  return _Py_HashDouble(val);
#else
  return _Py_HashDouble(NULL, val);
#endif
}

// 定义一个静态内联函数，用于计算 Python 浮点数对象的哈希值
static inline Py_hash_t floatobject_hash(PyFloatObject *key) {
  // 调用 _Pandas_HashDouble 函数计算浮点数对象的双精度值的哈希值
  return _Pandas_HashDouble(PyFloat_AS_DOUBLE(key));
}

// 定义一个常量，用于复数对象哈希算法中虚部的权重
#define _PandasHASH_IMAG 1000003UL

// 替换 _Py_HashDouble 函数为 _Pandas_HashDouble
// 定义一个静态内联函数，用于计算 Python 复数对象的哈希值
static inline Py_hash_t complexobject_hash(PyComplexObject *key) {
  // 调用 _Pandas_HashDouble 计算复数对象实部的哈希值
  Py_uhash_t realhash = (Py_uhash_t)_Pandas_HashDouble(key->cval.real);
  // 调用 _Pandas_HashDouble 计算复数对象虚部的哈希值
  Py_uhash_t imaghash = (Py_uhash_t)_Pandas_HashDouble(key->cval.imag);
  // 如果任何一个哈希值为 -1，则返回 -1
  if (realhash == (Py_uhash_t)-1 || imaghash == (Py_uhash_t)-1) {
    return -1;
  }
  // 计算复合哈希值
  Py_uhash_t combined = realhash + _PandasHASH_IMAG * imaghash;
  // 如果组合哈希值为 -1，则返回 -2
  if (combined == (Py_uhash_t)-1) {
    return -2;
  }
  // 返回合并后的哈希值
  return (Py_hash_t)combined;
}

// 声明一个函数，用于计算 Python 对象的哈希值
static inline khuint32_t kh_python_hash_func(PyObject *key);

// 声明一个注释，指出使用了 CPython 元组的原始哈希算法
// 我们可以使用任何哈希算法，这里使用的是 CPython 元组的原始算法
#if SIZEOF_PY_UHASH_T > 4
// 如果 Py_uhash_t 大于 4 字节，定义一些常量
#define _PandasHASH_XXPRIME_1 ((Py_uhash_t)11400714785074694791ULL)
#define _PandasHASH_XXPRIME_2 ((Py_uhash_t)14029467366897019727ULL)
#define _PandasHASH_XXPRIME_5 ((Py_uhash_t)2870177450012600261ULL)
// 定义一个宏，用于左移和右移位来实现循环移位
#define _PandasHASH_XXROTATE(x) ((x << 31) | (x >> 33))
#else
// 如果 Py_uhash_t 小于等于 4 字节，定义一些常量
#define _PandasHASH_XXPRIME_1 ((Py_uhash_t)2654435761UL)
#define _PandasHASH_XXPRIME_2 ((Py_uhash_t)2246822519UL)
#define _PandasHASH_XXPRIME_5 ((Py_uhash_t)374761393UL)
// 定义一个宏，用于左移和右移位来实现循环移位
#define _PandasHASH_XXROTATE(x) ((x << 13) | (x >> 19))
#endif
// 定义了一个静态内联函数 tupleobject_hash，用于计算 PyTupleObject 的哈希值
static inline Py_hash_t tupleobject_hash(PyTupleObject *key) {
  Py_ssize_t i, len = Py_SIZE(key);  // 获取元组的长度
  PyObject **item = key->ob_item;    // 获取元组的项数组指针

  Py_uhash_t acc = _PandasHASH_XXPRIME_5;  // 初始化累积哈希值为 _PandasHASH_XXPRIME_5
  for (i = 0; i < len; i++) {
    Py_uhash_t lane = kh_python_hash_func(item[i]);  // 计算元组项的哈希值
    if (lane == (Py_uhash_t)-1) {  // 如果哈希值为 -1，则返回错误
      return -1;
    }
    acc += lane * _PandasHASH_XXPRIME_2;  // 根据当前项的哈希值更新累积哈希值
    acc = _PandasHASH_XXROTATE(acc);      // 对累积哈希值进行旋转操作
    acc *= _PandasHASH_XXPRIME_1;         // 再次更新累积哈希值
  }

  /* 添加输入的长度，使用异或操作来保留哈希(()的历史值。 */
  acc += len ^ (_PandasHASH_XXPRIME_5 ^ 3527539UL);  // 添加长度并进行特殊处理

  if (acc == (Py_uhash_t)-1) {  // 如果累积哈希值为 -1，则返回一个特定的值
    return 1546275796;
  }
  return acc;  // 返回计算得到的哈希值
}

// 定义了一个静态内联函数 kh_python_hash_func，根据不同的 Python 对象计算哈希值
static inline khuint32_t kh_python_hash_func(PyObject *key) {
  Py_hash_t hash;  // 存储哈希值

  // 对于不同的 Python 对象类型，计算不同的哈希值
  if (PyFloat_CheckExact(key)) {
    hash = floatobject_hash((PyFloatObject *)key);  // 浮点数对象的哈希值
  } else if (PyComplex_CheckExact(key)) {
    hash = complexobject_hash((PyComplexObject *)key);  // 复数对象的哈希值
  } else if (PyTuple_CheckExact(key)) {
    hash = tupleobject_hash((PyTupleObject *)key);  // 元组对象的哈希值
  } else {
    hash = PyObject_Hash(key);  // 其他类型的 Python 对象的哈希值
  }

  if (hash == -1) {  // 如果哈希值为 -1，则清除异常并返回 0
    PyErr_Clear();
    return 0;
  }

  // 根据不同的构建设置返回 32 位或 64 位的哈希值
#if SIZEOF_PY_HASH_T == 4
  return hash;  // 返回 32 位的哈希值
#else
  khuint64_t as_uint = (khuint64_t)hash;  // 转换为 64 位整数
  return (as_uint >> 32) ^ as_uint;       // 返回 64 位哈希值的一部分
#endif
}

// 定义了一个宏 KHASH_MAP_INIT_PYOBJECT，用于初始化一个基于 PyObject * 的键的哈希表
#define KHASH_MAP_INIT_PYOBJECT(name, khval_t)                                 \
  KHASH_INIT(name, kh_pyobject_t, khval_t, 1, kh_python_hash_func,             \
             kh_python_hash_equal)

// 定义了一个宏 KHASH_SET_INIT_PYOBJECT，用于初始化一个基于 PyObject * 的键的集合
#define KHASH_SET_INIT_PYOBJECT(name)                                          \
  KHASH_INIT(name, kh_pyobject_t, char, 0, kh_python_hash_func,                \
             kh_python_hash_equal)

// 定义了一个宏 kh_exist_pymap，用于检查 pymap 哈希表中是否存在指定的键
#define kh_exist_pymap(h, k) (kh_exist(h, k))

// 定义了一个宏 kh_exist_pyset，用于检查 pyset 集合中是否存在指定的键
#define kh_exist_pyset(h, k) (kh_exist(h, k))

// 定义了一个哈希表类型 strbox，键为 PyObject *，值为 kh_pyobject_t
KHASH_MAP_INIT_STR(strbox, kh_pyobject_t)

// 定义了结构 kh_str_starts_t，包含一个字符串哈希表和一个整数数组
typedef struct {
  kh_str_t *table;  // 字符串哈希表
  int starts[256];   // 整数数组
} kh_str_starts_t;

// 定义了一个静态内联函数 kh_init_str_starts，用于初始化 kh_str_starts_t 结构
static inline p_kh_str_starts_t kh_init_str_starts(void) {
  kh_str_starts_t *result =
      (kh_str_starts_t *)KHASH_CALLOC(1, sizeof(kh_str_starts_t));  // 分配并初始化结构
  result->table = kh_init_str();  // 初始化字符串哈希表
  return result;  // 返回初始化后的结构指针
}
// 插入键值对到字符串开始标记的哈希表中，并返回插入结果
static inline khuint_t kh_put_str_starts_item(kh_str_starts_t *table, char *key,
                                              int *ret) {
    // 在底层的字符串哈希表中插入键值对，并获取插入结果
    khuint_t result = kh_put_str(table->table, key, ret);
    // 如果插入操作新建了一个条目
    if (*ret != 0) {
        // 标记以键的首字母为索引的起始字符存在于哈希表中
        table->starts[(unsigned char)key[0]] = 1;
    }
    // 返回插入结果
    return result;
}

// 查询给定键是否存在于字符串起始标记哈希表中
static inline khuint_t kh_get_str_starts_item(const kh_str_starts_t *table,
                                              const char *key) {
    // 获取键的首字母
    unsigned char ch = *key;
    // 如果首字母对应的标记为真
    if (table->starts[ch]) {
        // 如果键为空字符或者在底层字符串哈希表中找到了键
        if (ch == '\0' || kh_get_str(table->table, key) != table->table->n_buckets)
            return 1;
    }
    // 返回未找到的情况
    return 0;
}

// 销毁字符串起始标记哈希表及其内部的字符串哈希表
static inline void kh_destroy_str_starts(kh_str_starts_t *table) {
    // 销毁底层的字符串哈希表
    kh_destroy_str(table->table);
    // 释放哈希表结构体内存
    KHASH_FREE(table);
}

// 调整字符串起始标记哈希表的大小
static inline void kh_resize_str_starts(kh_str_starts_t *table, khuint_t val) {
    // 调整底层的字符串哈希表的大小
    kh_resize_str(table->table, val);
}

// 实用函数：根据元素数量计算所需的哈希表桶数目
static inline khuint_t kh_needed_n_buckets(khuint_t n_elements) {
    // 使用元素数量作为候选桶数
    khuint_t candidate = n_elements;
    // 将桶数向上取 2 的幂次方
    kroundup32(candidate);
    // 计算上限值
    khuint_t upper_bound = (khuint_t)(candidate * __ac_HASH_UPPER + 0.5);
    // 返回满足条件的桶数目
    return (upper_bound < n_elements) ? 2 * candidate : candidate;
}
```