# `.\numpy\numpy\_core\src\multiarray\buffer.c`

```py
/*
 * 定义 NPY_NO_DEPRECATED_API，避免使用已废弃的 NumPy API 版本
 * 定义 _MULTIARRAYMODULE，暂未说明具体用途
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

/*
 * 清除 PY_SSIZE_T_CLEAN 宏定义，确保 Py_ssize_t 被正确声明
 */
#define PY_SSIZE_T_CLEAN

/*
 * 包含 Python.h 头文件，提供 Python C API 支持
 * 包含 structmember.h 头文件，用于定义结构体成员
 */
#include <Python.h>
#include <structmember.h>

/*
 * 包含 NumPy 相关头文件
 */
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

/*
 * 包含 NumPy 的配置文件 npy_config.h
 */
#include "npy_config.h"

/*
 * 包含 NumPy 的 Python 兼容性支持文件 npy_pycompat.h
 */
#include "npy_pycompat.h"

/*
 * 包含 NumPy 的缓冲区协议实现 npy_buffer.h
 * 包含 NumPy 的通用函数和宏定义 common.h
 * 包含 NumPy 的操作系统相关功能 numpyos.h
 * 包含 NumPy 的数组对象定义 arrayobject.h
 * 包含 NumPy 的标量类型定义 scalartypes.h
 * 包含 NumPy 的数据类型元数据定义 dtypemeta.h
 */

/*************************************************************************
 ****************   实现缓冲区协议 ****************************
 *************************************************************************/

/*************************************************************************
 * PEP 3118 缓冲区协议
 *
 * 实现 PEP 3118 有些复杂，因为需要考虑以下要求：
 *
 * - 不向 ndarray 或 descr 结构体添加新成员，以保持二进制兼容性。
 *   （同时，向其中添加项目实际上并不是很有用，因为可变性问题阻碍了数组与缓冲区视图之间的一对一关系。）
 *
 * - 不使用 bf_releasebuffer，因为这会阻止 PyArg_ParseTuple("s#", ... 的工作。
 *   违反这一点会导致在 Python 2.6 上引起多个向后兼容性问题。
 *
 * - 在数组原地重塑或其 dtype 被修改时，要正确处理。
 *
 * 下面采取的解决方案是手动跟踪为 Py_buffers 分配的内存。
 *************************************************************************/

/*
 * 格式字符串转换器
 *
 * 将 PyArray_Descr 转换为 PEP 3118 格式字符串。
 */

/* 快速字符串 'class' */
typedef struct {
    char *s;
    size_t allocated;
    size_t pos;
} _tmp_string_t;

#define INIT_SIZE   16

/*
 * 将字符 c 追加到字符串 s 中
 */
static int
_append_char(_tmp_string_t *s, char c)
{
    if (s->pos >= s->allocated) {
        char *p;
        size_t to_alloc = (s->allocated == 0) ? INIT_SIZE : (2 * s->allocated);

        p = PyObject_Realloc(s->s, to_alloc);
        if (p == NULL) {
            PyErr_SetString(PyExc_MemoryError, "memory allocation failed");
            return -1;
        }
        s->s = p;
        s->allocated = to_alloc;
    }
    s->s[s->pos] = c;
    ++s->pos;
    return 0;
}

/*
 * 将字符串 p 追加到字符串 s 中
 */
static int
_append_str(_tmp_string_t *s, char const *p)
{
    for (; *p != '\0'; p++) {
        if (_append_char(s, *p) < 0) {
            return -1;
        }
    }
    return 0;
}

/*
 * 向字符串 str 追加一个 PEP3118 格式的字段名 ":name:"
 */
static int
_append_field_name(_tmp_string_t *str, PyObject *name)
{
    int ret = -1;
    char *p;
    Py_ssize_t len;
    PyObject *tmp;
    /* FIXME: XXX -- should it use UTF-8 here? */
    tmp = PyUnicode_AsUTF8String(name);
    if (tmp == NULL || PyBytes_AsStringAndSize(tmp, &p, &len) < 0) {
        PyErr_Clear();
        PyErr_SetString(PyExc_ValueError, "invalid field name");
        goto fail;
    }
    if (_append_char(str, ':') < 0) {
        goto fail;
    }
    while (len > 0) {
        // 当还有字符需要处理时进入循环
        if (*p == ':') {
            // 如果当前字符为':'，则抛出值错误异常并提示不允许在缓冲字段名中使用':'
            PyErr_SetString(PyExc_ValueError,
                            "':' is not an allowed character in buffer "
                            "field names");
            // 跳转到失败标签，表示处理失败
            goto fail;
        }
        // 将当前字符追加到字符串对象中，如果追加失败则跳转到失败标签
        if (_append_char(str, *p) < 0) {
            goto fail;
        }
        // 移动到下一个字符
        ++p;
        // 减少待处理字符数
        --len;
    }
    // 在字符串对象末尾追加字符':'，如果失败则跳转到失败标签
    if (_append_char(str, ':') < 0) {
        goto fail;
    }
    // 设置返回值为0，表示成功处理完所有字符
    ret = 0;
fail:
    // 释放临时对象的 Python 引用计数
    Py_XDECREF(tmp);
    // 返回操作的结果
    return ret;
}

/*
 * 如果一个类型在给定数组的每个项中都是对齐的，
 * 并且描述符元素大小是对齐的倍数，
 * 并且数组数据按照对齐的粒度定位，则返回非零值。
 */
static inline int
_is_natively_aligned_at(PyArray_Descr *descr,
                        PyArrayObject *arr, Py_ssize_t offset)
{
    int k;

    if (NPY_LIKELY(descr == PyArray_DESCR(arr))) {
        /*
         * 如果描述符与数组的描述符相同，可以假设数组的对齐是正确的。
         */
        assert(offset == 0);
        if (PyArray_ISALIGNED(arr)) {
            assert(descr->elsize % descr->alignment == 0);
            return 1;
        }
        return 0;
    }

    if ((Py_ssize_t)(PyArray_DATA(arr)) % descr->alignment != 0) {
        return 0;
    }

    if (offset % descr->alignment != 0) {
        return 0;
    }

    if (descr->elsize % descr->alignment) {
        return 0;
    }

    for (k = 0; k < PyArray_NDIM(arr); ++k) {
        if (PyArray_DIM(arr, k) > 1) {
            if (PyArray_STRIDE(arr, k) % descr->alignment != 0) {
                return 0;
            }
        }
    }

    return 1;
}

/*
 * 根据描述符填充字符串 str，生成适合 PEP 3118 格式的字符串。
 * 对于结构化数据类型，递归调用自身。每次调用在偏移量上扩展 str，
 * 更新偏移量，并使用 descr->byteorder（以及可能的 obj 中的字节顺序）
 * 来确定字节顺序字符。
 *
 * 返回 0 表示成功，-1 表示失败
 */
static int
_buffer_format_string(PyArray_Descr *descr, _tmp_string_t *str,
                      PyObject* obj, Py_ssize_t *offset,
                      char *active_byteorder)
{
    int k;
    char _active_byteorder = '@'; // 默认的字节顺序字符
    Py_ssize_t _offset = 0; // 默认的偏移量

    if (active_byteorder == NULL) {
        active_byteorder = &_active_byteorder;
    }
    if (offset == NULL) {
        offset = &_offset;
    }
    # 检查描述符是否具有子数组
    if (PyDataType_HASSUBARRAY(descr)) {
        # 将描述符转换为旧版数组描述符类型
        _PyArray_LegacyDescr *ldescr = (_PyArray_LegacyDescr *)descr;

        # 定义变量：子项对象、子数组元组、总元素计数、维度大小、旧偏移量、字符缓冲区和返回状态
        PyObject *item, *subarray_tuple;
        Py_ssize_t total_count = 1;
        Py_ssize_t dim_size;
        Py_ssize_t old_offset;
        char buf[128];
        int ret;

        # 检查子数组形状是否为元组，若是则直接使用，否则构建一个包含形状的元组
        if (PyTuple_Check(ldescr->subarray->shape)) {
            subarray_tuple = ldescr->subarray->shape;
            Py_INCREF(subarray_tuple);
        }
        else {
            subarray_tuple = Py_BuildValue("(O)", ldescr->subarray->shape);
        }

        # 在字符串缓冲中追加左括号 '('
        if (_append_char(str, '(') < 0) {
            ret = -1;
            goto subarray_fail;
        }
        # 遍历子数组元组
        for (k = 0; k < PyTuple_GET_SIZE(subarray_tuple); ++k) {
            # 如果不是第一个元素，追加逗号 ','
            if (k > 0) {
                if (_append_char(str, ',') < 0) {
                    ret = -1;
                    goto subarray_fail;
                }
            }
            # 获取子数组元组中的项
            item = PyTuple_GET_ITEM(subarray_tuple, k);
            # 将项转换为 Py_ssize_t 类型的维度大小
            dim_size = PyNumber_AsSsize_t(item, NULL);

            # 将维度大小转换为字符串，存入 buf 中
            PyOS_snprintf(buf, sizeof(buf), "%ld", (long)dim_size);
            # 将 buf 中的内容追加到字符串缓冲 str 中
            if (_append_str(str, buf) < 0) {
                ret = -1;
                goto subarray_fail;
            }
            # 计算总元素个数
            total_count *= dim_size;
        }
        # 在字符串缓冲中追加右括号 ')'
        if (_append_char(str, ')') < 0) {
            ret = -1;
            goto subarray_fail;
        }

        # 保存旧偏移量，调用 _buffer_format_string 处理子数组的基础类型描述符
        old_offset = *offset;
        ret = _buffer_format_string(ldescr->subarray->base, str, obj, offset,
                                    active_byteorder);
        # 更新偏移量，考虑到子数组的总元素个数
        *offset = old_offset + (*offset - old_offset) * total_count;

    subarray_fail:
        # 减少子数组元组的引用计数，并返回操作状态
        Py_DECREF(subarray_tuple);
        return ret;
    }
    else if (PyDataType_HASFIELDS(descr)) {
        // 如果描述符包含字段信息
        _PyArray_LegacyDescr *ldescr = (_PyArray_LegacyDescr *)descr;
        // 将偏移量保存到基础偏移量中
        Py_ssize_t base_offset = *offset;

        // 向字符串中追加 'T{'，表示开始一个复合类型描述
        if (_append_str(str, "T{") < 0) return -1;
        // 遍历复合类型中的字段
        for (k = 0; k < PyTuple_GET_SIZE(ldescr->names); ++k) {
            PyObject *name, *item, *offset_obj;
            PyArray_Descr *child;
            Py_ssize_t new_offset;
            int ret;

            // 获取字段名
            name = PyTuple_GET_ITEM(ldescr->names, k);
            // 从字段字典中获取字段信息
            item = PyDict_GetItem(ldescr->fields, name);

            // 获取字段的数据类型描述符和偏移量对象
            child = (PyArray_Descr*)PyTuple_GetItem(item, 0);
            offset_obj = PyTuple_GetItem(item, 1);
            // 将偏移量对象转换为 Py_ssize_t 类型
            new_offset = PyLong_AsLong(offset_obj);
            // 检查是否转换出错
            if (error_converting(new_offset)) {
                return -1;
            }
            // 根据基础偏移量计算新的偏移量
            new_offset += base_offset;

            /* 手动插入填充 */
            // 如果当前偏移量大于新的偏移量，说明需要插入填充字节
            if (*offset > new_offset) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "dtypes with overlapping or out-of-order fields are not "
                    "representable as buffers. Consider reordering the fields."
                );
                return -1;
            }
            // 插入填充字节，直到当前偏移量达到新的偏移量
            while (*offset < new_offset) {
                if (_append_char(str, 'x') < 0) return -1;
                ++*offset;
            }

            /* 插入子项 */
            // 递归处理子项的格式字符串
            ret = _buffer_format_string(child, str, obj, offset,
                                  active_byteorder);
            // 检查处理子项是否出错
            if (ret < 0) {
                return -1;
            }

            /* 插入字段名 */
            // 向字符串中插入字段名
            if (_append_field_name(str, name) < 0) return -1;
        }
        // 向字符串中追加 '}'，表示复合类型描述结束
        if (_append_char(str, '}') < 0) return -1;
    }
    // 返回成功标志
    return 0;
/* 结构体定义：用于存储为提供缓冲区接口而需要的每个数组的额外数据 */
typedef struct _buffer_info_t_tag {
    char *format;               // 缓冲区数据的格式
    int ndim;                   // 数组的维度
    Py_ssize_t *strides;        // 数组的步长信息
    Py_ssize_t *shape;          // 数组的形状信息
    struct _buffer_info_t_tag *next;   // 指向下一个缓冲区信息结构体的指针
} _buffer_info_t;

/* 创建并返回一个新的缓冲区信息结构体 */
static _buffer_info_t*
_buffer_info_new(PyObject *obj, int flags)
{
    /* 缓冲区信息被缓存为 PyLongObjects，这使得它们在 valgrind 中看起来像是无法访问的丢失内存 */
    _buffer_info_t *info;       // 缓冲区信息结构体指针
    _tmp_string_t fmt = {NULL, 0, 0};   // 临时字符串结构体
    int k;                      // 循环变量
    PyArray_Descr *descr = NULL;    // 数组描述符指针
    int err = 0;                // 错误码

    /* 如果 obj 是一个标量对象且类型是 Void */
    if (PyArray_IsScalar(obj, Void)) {
        info = PyObject_Malloc(sizeof(_buffer_info_t));   // 分配缓冲区信息结构体的内存空间
        if (info == NULL) {
            PyErr_NoMemory();   // 内存分配失败，抛出内存错误异常
            goto fail;          // 跳转到失败处理标签
        }
        info->ndim = 0;         // 设置数组维度为 0
        info->shape = NULL;     // 形状信息为空
        info->strides = NULL;   // 步长信息为空

        /* 从标量对象创建数组描述符 */
        descr = PyArray_DescrFromScalar(obj);
        if (descr == NULL) {
            goto fail;          // 如果创建描述符失败，跳转到失败处理标签
        }
    }
    else {
        // 断言对象是否为 NumPy 数组
        assert(PyArray_Check(obj));
        // 将对象转换为 PyArrayObject 类型
        PyArrayObject * arr = (PyArrayObject *)obj;
    
        // 分配内存给 info 结构体，包括 shape 和 strides
        info = PyObject_Malloc(sizeof(_buffer_info_t) +
                               sizeof(Py_ssize_t) * PyArray_NDIM(arr) * 2);
        // 如果内存分配失败，设置内存错误并跳转到失败标签
        if (info == NULL) {
            PyErr_NoMemory();
            goto fail;
        }
        /* 填充 shape 和 strides */
        info->ndim = PyArray_NDIM(arr);
    
        // 如果数组维度为 0，shape 和 strides 都为 NULL
        if (info->ndim == 0) {
            info->shape = NULL;
            info->strides = NULL;
        }
        else {
            // 分配内存给 shape 和 strides，确保地址对齐
            info->shape = (npy_intp *)((char *)info + sizeof(_buffer_info_t));
            assert((size_t)info->shape % sizeof(npy_intp) == 0);
            info->strides = info->shape + PyArray_NDIM(arr);
    
            /*
             * 一些缓冲区使用者可能希望连续的缓冲区在维度为 1 时也有格式良好的 strides，
             * 但是我们在内部不保证这一点。因此，对于连续数组，需要重新计算 strides。
             */
            int f_contiguous = (flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS;
            if (PyArray_IS_C_CONTIGUOUS(arr) && !(
                    f_contiguous && PyArray_IS_F_CONTIGUOUS(arr))) {
                Py_ssize_t sd = PyArray_ITEMSIZE(arr);
                for (k = info->ndim-1; k >= 0; --k) {
                    info->shape[k] = PyArray_DIMS(arr)[k];
                    info->strides[k] = sd;
                    sd *= info->shape[k];
                }
            }
            else if (PyArray_IS_F_CONTIGUOUS(arr)) {
                Py_ssize_t sd = PyArray_ITEMSIZE(arr);
                for (k = 0; k < info->ndim; ++k) {
                    info->shape[k] = PyArray_DIMS(arr)[k];
                    info->strides[k] = sd;
                    sd *= info->shape[k];
                }
            }
            else {
                // 非连续数组，直接复制 PyArray 的维度和 strides
                for (k = 0; k < PyArray_NDIM(arr); ++k) {
                    info->shape[k] = PyArray_DIMS(arr)[k];
                    info->strides[k] = PyArray_STRIDES(arr)[k];
                }
            }
        }
        descr = PyArray_DESCR(arr);
        // 增加对象的引用计数
        Py_INCREF(descr);
    }
    
    /* 填充格式 */
    // 如果 flags 包含 PyBUF_FORMAT 标志位
    if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) {
        // 获取数组的格式字符串
        err = _buffer_format_string(descr, &fmt, obj, NULL, NULL);
        // 减少描述符的引用计数
        Py_DECREF(descr);
        // 如果获取格式字符串出错，跳转到失败标签
        if (err != 0) {
            goto fail;
        }
        // 如果追加字符失败，跳转到失败标签
        if (_append_char(&fmt, '\0') < 0) {
            goto fail;
        }
        // 将格式字符串赋值给 info 的 format 字段
        info->format = fmt.s;
    }
    else {
        // 减少描述符的引用计数
        Py_DECREF(descr);
        // 将 format 字段设置为 NULL
        info->format = NULL;
    }
    info->next = NULL;
    // 返回 info 结构体
    return info;
fail:
    # 释放 fmt.s 所占用的内存
    PyObject_Free(fmt.s);
    # 释放 info 所占用的内存
    PyObject_Free(info);
    # 返回 NULL 表示操作失败
    return NULL;
}

/* Compare two info structures */
static Py_ssize_t
_buffer_info_cmp(_buffer_info_t *a, _buffer_info_t *b)
{
    Py_ssize_t c;
    int k;

    // 比较两个 _buffer_info_t 结构体的 format 字段
    if (a->format != NULL && b->format != NULL) {
        c = strcmp(a->format, b->format);
        if (c != 0) return c;
    }
    // 比较两个 _buffer_info_t 结构体的 ndim 字段
    c = a->ndim - b->ndim;
    if (c != 0) return c;

    // 逐个比较两个 _buffer_info_t 结构体的 shape 和 strides 数组
    for (k = 0; k < a->ndim; ++k) {
        c = a->shape[k] - b->shape[k];
        if (c != 0) return c;
        c = a->strides[k] - b->strides[k];
        if (c != 0) return c;
    }

    // 结构体比较完成，返回相等
    return 0;
}


/*
 * Tag the buffer info pointer by adding 2 (unless it is NULL to simplify
 * object initialization).
 * The linked list of buffer-infos was appended to the array struct in
 * NumPy 1.20. Tagging the pointer gives us a chance to raise/print
 * a useful error message instead of crashing hard if a C-subclass uses
 * the same field.
 */
static inline void *
buffer_info_tag(void *buffer_info)
{
    // 如果 buffer_info 为 NULL，则直接返回 NULL
    if (buffer_info == NULL) {
        return buffer_info;
    }
    else {
        // 否则，在 buffer_info 地址上加 3 并返回，用于标记
        return (void *)((uintptr_t)buffer_info + 3);
    }
}


static inline int
_buffer_info_untag(
        void *tagged_buffer_info, _buffer_info_t **buffer_info, PyObject *obj)
{
    // 如果 tagged_buffer_info 为 NULL，则直接返回，无需解标记
    if (tagged_buffer_info == NULL) {
        *buffer_info = NULL;
        return 0;
    }
    // 检查 tagged_buffer_info 是否正确标记
    if (NPY_UNLIKELY(((uintptr_t)tagged_buffer_info & 0x7) != 3)) {
        PyErr_Format(PyExc_RuntimeError,
                "Object of type %S appears to be C subclassed NumPy array, "
                "void scalar, or allocated in a non-standard way."
                "NumPy reserves the right to change the size of these "
                "structures. Projects are required to take this into account "
                "by either recompiling against a specific NumPy version or "
                "padding the struct and enforcing a maximum NumPy version.",
                Py_TYPE(obj));
        return -1;
    }
    // 解标记 tagged_buffer_info 并返回正确的 _buffer_info_t 结构体指针
    *buffer_info = (void *)((uintptr_t)tagged_buffer_info - 3);
    return 0;
}


/*
 * NOTE: for backward compatibility (esp. with PyArg_ParseTuple("s#", ...))
 * we do *not* define bf_releasebuffer at all.
 *
 * Instead, any extra data allocated with the buffer is released only in
 * array_dealloc.
 *
 * Ensuring that the buffer stays in place is taken care by refcounting;
 * ndarrays do not reallocate if there are references to them, and a buffer
 * view holds one reference.
 *
 * This is stored in the array's _buffer_info slot (currently as a void *).
 */
static void
_buffer_info_free_untagged(void *_buffer_info)
{
    _buffer_info_t *next = _buffer_info;
    // 释放整个链表上的 _buffer_info_t 结构体及其包含的内存
    while (next != NULL) {
        _buffer_info_t *curr = next;
        next = curr->next;
        if (curr->format) {
            PyObject_Free(curr->format);
        }
        /* Shape is allocated as part of info */
        PyObject_Free(curr);
    }
}
/*
 * 检查指针是否已标记，并释放缓存列表。
 * （标记检查仅适用于由于结构大小在1.20中更改而进行的过渡）
 */
NPY_NO_EXPORT int
_buffer_info_free(void *buffer_info, PyObject *obj)
{
    _buffer_info_t *untagged_buffer_info;
    // 如果解除标记失败，则返回-1
    if (_buffer_info_untag(buffer_info, &untagged_buffer_info, obj) < 0) {
        return -1;
    }
    // 释放未标记的缓存信息
    _buffer_info_free_untagged(untagged_buffer_info);
    return 0;
}




/*
 * 获取缓冲区信息，返回传入的旧信息或添加保持（因此替换）旧信息的新缓冲区信息。
 */
static _buffer_info_t*
_buffer_get_info(void **buffer_info_cache_ptr, PyObject *obj, int flags)
{
    _buffer_info_t *info = NULL;
    _buffer_info_t *stored_info;  /* 当前存储的第一个缓冲区信息 */

    // 如果解除标记失败，则返回空指针
    if (_buffer_info_untag(*buffer_info_cache_ptr, &stored_info, obj) < 0) {
        return NULL;
    }
    _buffer_info_t *old_info = stored_info;

    /* 计算信息（在简单情况下可以跳过这一步骤将会更好） */
    // 创建新的缓冲区信息对象
    info = _buffer_info_new(obj, flags);
    if (info == NULL) {
        return NULL;
    }

    // 如果存在旧信息并且新旧信息不同，则进行比较
    if (old_info != NULL && _buffer_info_cmp(info, old_info) != 0) {
        _buffer_info_t *next_info = old_info->next;
        old_info = NULL;  /* 不能使用此旧信息，但可能使用下一个 */

        // 如果 info 的维度大于1且下一个信息不为空，则比较两者
        if (info->ndim > 1 && next_info != NULL) {
            /*
             * 有些数组是C和F连续的，如果它们有多于一个维度，
             * 缓冲区信息可能在两者之间不同，因为长度为1的维度的步幅可能会调整。
             * 如果我们导出这两个缓冲区，第一个存储的可能是另一个连续性的缓冲区，
             * 因此检查两者。
             * 这在所有其他情况下通常是非常不可能的，因为在所有其他情况下，
             * 第一个将与第一个匹配，除非数组元数据被就地修改（这是不鼓励的）。
             */
            if (_buffer_info_cmp(info, next_info) == 0) {
                old_info = next_info;
            }
        }
    }
    // 如果存在旧信息，则处理新旧信息的格式相等性
    if (old_info != NULL) {
        /*
         * 如果其中一个 info->format 未设置格式（意味着格式是任意的并且可以修改）。
         * 如果新信息有格式，但我们重用旧信息，则将所有权转移到旧信息。
         */
        if (old_info->format == NULL) {
            old_info->format = info->format;
            info->format = NULL;
        }
        // 释放新信息的未标记对象并返回旧信息
        _buffer_info_free_untagged(info);
        info = old_info;
    }
    else {
        /* 将新信息插入到链接的缓冲区信息列表的第一项中 */
        info->next = stored_info;
        *buffer_info_cache_ptr = buffer_info_tag(info);
    }

    return info;
}




/*
 * 为ndarray检索缓冲区
 */
static int
array_getbuffer(PyObject *obj, Py_buffer *view, int flags)
{
    PyArrayObject *self;
    _buffer_info_t *info = NULL;
    // 定义指向 _buffer_info_t 结构体的指针变量，并初始化为 NULL

    self = (PyArrayObject*)obj;
    // 将 obj 强制转换为 PyArrayObject 类型，并赋值给 self

    /* Check whether we can provide the wanted properties */
    // 检查是否可以提供所需的属性

    if ((flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS &&
            !PyArray_CHKFLAGS(self, NPY_ARRAY_C_CONTIGUOUS)) {
        // 如果请求的是 C 连续的缓冲区，并且数组不是 C 连续的
        PyErr_SetString(PyExc_ValueError, "ndarray is not C-contiguous");
        // 设置异常，指示数组不是 C 连续的
        goto fail;
        // 跳转到失败处理部分
    }
    if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS &&
            !PyArray_CHKFLAGS(self, NPY_ARRAY_F_CONTIGUOUS)) {
        // 如果请求的是 Fortran 连续的缓冲区，并且数组不是 Fortran 连续的
        PyErr_SetString(PyExc_ValueError, "ndarray is not Fortran contiguous");
        // 设置异常，指示数组不是 Fortran 连续的
        goto fail;
        // 跳转到失败处理部分
    }
    if ((flags & PyBUF_ANY_CONTIGUOUS) == PyBUF_ANY_CONTIGUOUS
            && !PyArray_ISONESEGMENT(self)) {
        // 如果请求的是任意连续的缓冲区，并且数组不是单一段的
        PyErr_SetString(PyExc_ValueError, "ndarray is not contiguous");
        // 设置异常，指示数组不是连续的
        goto fail;
        // 跳转到失败处理部分
    }
    if ((flags & PyBUF_STRIDES) != PyBUF_STRIDES &&
            !PyArray_CHKFLAGS(self, NPY_ARRAY_C_CONTIGUOUS)) {
        // 如果请求的是非步幅数组，但数组不是 C 连续的
        PyErr_SetString(PyExc_ValueError, "ndarray is not C-contiguous");
        // 设置异常，指示数组不是 C 连续的
        goto fail;
        // 跳转到失败处理部分
    }
    if ((flags & PyBUF_WRITEABLE) == PyBUF_WRITEABLE) {
        // 如果请求的缓冲区可写
        if (PyArray_FailUnlessWriteable(self, "buffer source array") < 0) {
            // 检查数组是否可写，如果不可写则设置异常
            goto fail;
            // 跳转到失败处理部分
        }
    }

    if (view == NULL) {
        // 如果传入的视图指针为空
        PyErr_SetString(PyExc_ValueError, "NULL view in getbuffer");
        // 设置异常，表示在获取缓冲区时传入了空视图
        goto fail;
        // 跳转到失败处理部分
    }

    /* Fill in information (and add it to _buffer_info if necessary) */
    // 填充信息（如果需要，将其添加到 _buffer_info 中）
    info = _buffer_get_info(
            &((PyArrayObject_fields *)self)->_buffer_info, obj, flags);
    // 调用 _buffer_get_info 函数获取缓冲区信息，并赋值给 info
    if (info == NULL) {
        // 如果获取信息失败
        goto fail;
        // 跳转到失败处理部分
    }

    view->buf = PyArray_DATA(self);
    // 设置视图的缓冲区指针为数组的数据指针
    view->suboffsets = NULL;
    // 设置子偏移为 NULL
    view->itemsize = PyArray_ITEMSIZE(self);
    // 设置视图的每个项的大小为数组的每个项的大小

    /*
     * If a read-only buffer is requested on a read-write array, we return a
     * read-write buffer as per buffer protocol.
     * We set a requested buffer to readonly also if the array will be readonly
     * after a deprecation. This jumps the deprecation, but avoiding the
     * warning is not convenient here. A warning is given if a writeable
     * buffer is requested since `PyArray_FailUnlessWriteable` is called above
     * (and clears the `NPY_ARRAY_WARN_ON_WRITE` flag).
     */
    // 如果请求在可读写数组上获取只读缓冲区，则根据缓冲区协议返回可读写的缓冲区。
    // 如果数组在弃用后将变为只读，我们也将请求的缓冲区设置为只读。
    // 这跳过了弃用，但这里避免警告并不方便。如果请求可写的缓冲区，则会发出警告，因为上面调用了 `PyArray_FailUnlessWriteable`（并清除了 `NPY_ARRAY_WARN_ON_WRITE` 标志）。

    view->readonly = (!PyArray_ISWRITEABLE(self) ||
                      PyArray_CHKFLAGS(self, NPY_ARRAY_WARN_ON_WRITE));
    // 设置视图的只读属性，如果数组不可写或设置了 `NPY_ARRAY_WARN_ON_WRITE` 标志，则设置为只读

    view->internal = NULL;
    // 设置视图的内部指针为 NULL

    view->len = PyArray_NBYTES(self);
    // 设置视图的长度为数组的字节大小

    if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) {
        view->format = info->format;
        // 如果请求获取格式化信息，则将视图的格式指针设置为 info 中的格式信息
    } else {
        view->format = NULL;
        // 否则将格式指针设置为 NULL
    }

    if ((flags & PyBUF_ND) == PyBUF_ND) {
        view->ndim = info->ndim;
        // 如果请求获取维度信息，则将视图的维度设置为 info 中的维度信息
        view->shape = info->shape;
        // 并将形状指针设置为 info 中的形状信息
    }
    else {
        view->ndim = 0;
        // 否则将维度设置为 0
        view->shape = NULL;
        // 并将形状指针设置为 NULL
    }

    if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
        view->strides = info->strides;
        // 如果请求获取步幅信息，则将视图的步幅指针设置为 info 中的步幅信息
    }
    else {
        view->strides = NULL;
        // 否则将步幅指针设置为 NULL
    }

    view->obj = (PyObject*)self;
    // 将视图的对象指针设置为数组对象的指针

    Py_INCREF(self);
    // 增加数组对象的引用计数
    return 0;
    // 返回 0，表示获取缓冲区成功

fail:
    // 失败处理部分的标签
/*
 * 返回错误代码 -1，表示操作失败
 */
fail:
    return -1;
}

/*
 * 从 void scalar 中获取缓冲区（可以包含任意复杂类型），
 * 定义在 buffer.c 中，因为它需要复杂格式构建逻辑。
 */
NPY_NO_EXPORT int
void_getbuffer(PyObject *self, Py_buffer *view, int flags)
{
    // 将 self 强制转换为 PyVoidScalarObject 类型
    PyVoidScalarObject *scalar = (PyVoidScalarObject *)self;

    // 如果传入的 flags 包含 PyBUF_WRITABLE 标志，设置错误信息并返回 -1
    if (flags & PyBUF_WRITABLE) {
        PyErr_SetString(PyExc_BufferError, "scalar buffer is readonly");
        return -1;
    }

    // 设置视图的维度为 0，形状为空，步长为空，子偏移为空
    view->ndim = 0;
    view->shape = NULL;
    view->strides = NULL;
    view->suboffsets = NULL;
    // 设置视图的长度和项大小为 scalar 对象描述符的元素大小
    view->len = scalar->descr->elsize;
    view->itemsize = scalar->descr->elsize;
    // 视图为只读
    view->readonly = 1;
    view->suboffsets = NULL;
    // 增加 self 的引用计数
    Py_INCREF(self);
    // 视图的对象指针指向 self
    view->obj = self;
    // 视图的缓冲区指针指向 scalar 的值
    view->buf = scalar->obval;

    // 如果 flags 不包含 PyBUF_FORMAT 标志，设置视图的格式为 NULL 并返回 0
    if (((flags & PyBUF_FORMAT) != PyBUF_FORMAT)) {
        /* It is unnecessary to find the correct format */
        view->format = NULL;
        return 0;
    }

    /*
     * 如果正在导出格式，我们需要使用 _buffer_get_info 函数
     * 来找到正确的格式。此格式也必须存储，因为理论上它可以改变
     * （实际上它不应该改变）。
     */
    // 使用 _buffer_get_info 函数获取格式信息，存储在 info 中
    _buffer_info_t *info = _buffer_get_info(&scalar->_buffer_info, self, flags);
    // 如果 info 为 NULL，释放 self 的引用计数并返回 -1
    if (info == NULL) {
        Py_DECREF(self);
        return -1;
    }
    // 设置视图的格式为 info 中的格式
    view->format = info->format;
    // 返回成功标志 0
    return 0;
}


/*************************************************************************/

// array_as_buffer 结构体，包含获取缓冲区的函数指针和释放缓冲区的函数指针
NPY_NO_EXPORT PyBufferProcs array_as_buffer = {
    (getbufferproc)array_getbuffer,
    (releasebufferproc)0,
};


/*************************************************************************
 * 将 PEP 3118 格式字符串转换为 PyArray_Descr 结构体
 */

// 快速版本的 _descriptor_from_pep3118_format 函数声明
static int
_descriptor_from_pep3118_format_fast(char const *s, PyObject **result);

// 根据字母、本地化标志和复杂性标志返回数据类型
static int
_pep3118_letter_to_type(char letter, int native, int is_complex);

// 将 PEP 3118 格式字符串转换为 PyArray_Descr 结构体的函数定义
NPY_NO_EXPORT PyArray_Descr*
_descriptor_from_pep3118_format(char const *s)
{
    char *buf, *p;
    int in_name = 0;
    int obtained;
    PyObject *descr;
    PyObject *str;
    PyObject *_numpy_internal;

    // 如果传入的 s 为 NULL，返回一个新的 NPY_BYTE 类型的 PyArray_Descr 结构体
    if (s == NULL) {
        return PyArray_DescrNewFromType(NPY_BYTE);
    }

    // 快速路径，尝试使用快速版本的 _descriptor_from_pep3118_format 函数
    obtained = _descriptor_from_pep3118_format_fast(s, &descr);
    if (obtained) {
        return (PyArray_Descr*)descr;
    }

    // 去除 s 中的空白字符，但保留字段名中的空格
    buf = malloc(strlen(s) + 1);
    if (buf == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    p = buf;
    while (*s != '\0') {
        if (*s == ':') {
            in_name = !in_name;
            *p = *s;
            p++;
        }
        else if (in_name || !NumPyOS_ascii_isspace(*s)) {
            *p = *s;
            p++;
        }
        s++;
    }
    *p = '\0';

    // 根据处理后的 buf 创建 PyUnicode 对象
    str = PyUnicode_FromStringAndSize(buf, strlen(buf));
    // 如果创建失败，释放 buf 的内存并返回 NULL
    if (str == NULL) {
        free(buf);
        return NULL;
    }

    // 导入 numpy._core._internal 模块
    _numpy_internal = PyImport_ImportModule("numpy._core._internal");
    # 如果 _numpy_internal 是 NULL，则说明未能获取到 numpy 内部对象，函数无法继续执行，返回 NULL
    if (_numpy_internal == NULL) {
        Py_DECREF(str);  # 减少字符串对象的引用计数，避免内存泄漏
        free(buf);       # 释放 buf 所占用的内存空间
        return NULL;     # 返回 NULL 表示函数执行失败
    }
    # 调用 numpy 内部对象的方法 "_dtype_from_pep3118"，将 str 作为参数传递
    descr = PyObject_CallMethod(
        _numpy_internal, "_dtype_from_pep3118", "O", str);
    Py_DECREF(str);       # 减少字符串对象的引用计数，避免内存泄漏
    Py_DECREF(_numpy_internal);  # 减少 numpy 内部对象的引用计数，避免内存泄漏
    # 如果调用返回 NULL，说明处理失败，需要设置错误信息并返回 NULL
    if (descr == NULL) {
        PyObject *exc, *val, *tb;  # 定义异常、值和 traceback 对象
        PyErr_Fetch(&exc, &val, &tb);  # 获取当前的错误信息
        // 设置错误消息，指出 buf 不是一个有效的 PEP 3118 缓冲格式字符串
        PyErr_Format(PyExc_ValueError,
                     "'%s' is not a valid PEP 3118 buffer format string", buf);
        // 将当前异常链入上一个异常的原因中
        npy_PyErr_ChainExceptionsCause(exc, val, tb);
        free(buf);  # 释放 buf 所占用的内存空间
        return NULL;  # 返回 NULL 表示函数执行失败
    }
    // 检查 descr 是否是一个有效的数组描述符对象，如果不是则抛出运行时错误
    if (!PyArray_DescrCheck(descr)) {
        // 设置运行时错误信息，指出 numpy._core._internal._dtype_from_pep3118 没有返回有效的数组描述符
        PyErr_Format(PyExc_RuntimeError,
                     "internal error: numpy._core._internal._dtype_from_pep3118 "
                     "did not return a valid dtype, got %s", buf);
        Py_DECREF(descr);  // 减少描述符对象的引用计数，避免内存泄漏
        free(buf);         // 释放 buf 所占用的内存空间
        return NULL;       // 返回 NULL 表示函数执行失败
    }
    free(buf);  // 释放 buf 所占用的内存空间
    return (PyArray_Descr*)descr;  // 返回有效的数组描述符对象的指针类型
/*
 * Fast path for parsing buffer strings corresponding to simple types.
 *
 * Currently, this deals only with single-element data types.
 */



static int
_descriptor_from_pep3118_format_fast(char const *s, PyObject **result)
{
    PyArray_Descr *descr;

    int is_standard_size = 0;
    char byte_order = '=';
    int is_complex = 0;

    int type_num = NPY_BYTE;
    int item_seen = 0;

    for (; *s != '\0'; ++s) {
        is_complex = 0;
        switch (*s) {
        case '@':
        case '^':
            /* ^ means no alignment; doesn't matter for a single element */
            byte_order = '=';
            is_standard_size = 0;
            break;
        case '<':
            byte_order = '<';
            is_standard_size = 1;
            break;
        case '>':
        case '!':
            byte_order = '>';
            is_standard_size = 1;
            break;
        case '=':
            byte_order = '=';
            is_standard_size = 1;
            break;
        case 'Z':
            is_complex = 1;
            ++s;
        default:
            if (item_seen) {
                /* Not a single-element data type */
                return 0;
            }
            type_num = _pep3118_letter_to_type(*s, !is_standard_size,
                                               is_complex);
            if (type_num < 0) {
                /* Something unknown */
                return 0;
            }
            item_seen = 1;
            break;
        }
    }

    if (!item_seen) {
        return 0;
    }

    descr = PyArray_DescrFromType(type_num);
    if (descr == NULL) {
        return 0;
    }
    if (byte_order == '=') {
        *result = (PyObject*)descr;
    }
    else {
        *result = (PyObject*)PyArray_DescrNewByteorder(descr, byte_order);
        Py_DECREF(descr);
        if (*result == NULL) {
            return 0;
        }
    }

    return 1;
}



static int
_pep3118_letter_to_type(char letter, int native, int is_complex)
{
    switch (letter)
    {
    case '?': return NPY_BOOL;
    case 'b': return NPY_BYTE;
    case 'B': return NPY_UBYTE;
    case 'h': return native ? NPY_SHORT : NPY_INT16;
    case 'H': return native ? NPY_USHORT : NPY_UINT16;
    case 'i': return native ? NPY_INT : NPY_INT32;
    case 'I': return native ? NPY_UINT : NPY_UINT32;
    case 'l': return native ? NPY_LONG : NPY_INT32;
    case 'L': return native ? NPY_ULONG : NPY_UINT32;
    case 'q': return native ? NPY_LONGLONG : NPY_INT64;
    case 'Q': return native ? NPY_ULONGLONG : NPY_UINT64;
    case 'n': return native ? NPY_INTP : -1;
    case 'N': return native ? NPY_UINTP : -1;
    case 'e': return NPY_HALF;
    case 'f': return is_complex ? NPY_CFLOAT : NPY_FLOAT;
    case 'd': return is_complex ? NPY_CDOUBLE : NPY_DOUBLE;
    case 'g': return native ? (is_complex ? NPY_CLONGDOUBLE : NPY_LONGDOUBLE) : -1;
    default:
        /* Other unhandled cases */
        return -1;
    }
    return -1;
}
```