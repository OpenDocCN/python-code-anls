# `.\numpy\numpy\_core\src\multiarray\stringdtype\dtype.h`

```
#ifndef _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_DTYPE_H_
#define _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_DTYPE_H_

#ifdef __cplusplus
extern "C" {
#endif

// static string library中未公开的内容，因此需要在这里定义
// 这样可以在描述符上定义 `elsize` 和 `alignment`
//
// 如果 `npy_packed_static_string` 的布局在将来发生变化，可能需要更新此处内容。
#define SIZEOF_NPY_PACKED_STATIC_STRING 2 * sizeof(size_t)
#define ALIGNOF_NPY_PACKED_STATIC_STRING _Alignof(size_t)

// 返回一个新的字符串类型实例
NPY_NO_EXPORT PyObject *
new_stringdtype_instance(PyObject *na_object, int coerce);

// 初始化字符串类型的相关内容
NPY_NO_EXPORT int
init_string_dtype(void);

// 假设调用者已经获取了两个描述符的分配器锁
NPY_NO_EXPORT int
_compare(void *a, void *b, PyArray_StringDTypeObject *descr_a,
         PyArray_StringDTypeObject *descr_b);

// 初始化字符串类型的 NA 对象
NPY_NO_EXPORT int
init_string_na_object(PyObject *mod);

// 设置字符串类型的项目
NPY_NO_EXPORT int
stringdtype_setitem(PyArray_StringDTypeObject *descr, PyObject *obj, char **dataptr);

// 在调用此函数之前，必须先获取两个分配器的锁
NPY_NO_EXPORT int
free_and_copy(npy_string_allocator *in_allocator,
              npy_string_allocator *out_allocator,
              const npy_packed_static_string *in,
              npy_packed_static_string *out, const char *location);

// 加载新的字符串
NPY_NO_EXPORT int
load_new_string(npy_packed_static_string *out, npy_static_string *out_ss,
                size_t num_bytes, npy_string_allocator *allocator,
                const char *err_context);

// 最终化字符串类型的描述符
NPY_NO_EXPORT PyArray_Descr *
stringdtype_finalize_descr(PyArray_Descr *dtype);

// 执行等于比较
NPY_NO_EXPORT int
_eq_comparison(int scoerce, int ocoerce, PyObject *sna, PyObject *ona);

// 检查两个 NA 对象是否兼容
NPY_NO_EXPORT int
stringdtype_compatible_na(PyObject *na1, PyObject *na2, PyObject **out_na);

#ifdef __cplusplus
}
#endif

#endif /* _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_DTYPE_H_ */
```