# `.\numpy\numpy\_core\src\multiarray\dlpack.c`

```
/*
 * 定义宏，指定使用的 NumPy API 版本，避免使用已废弃的 API
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
/*
 * 定义宏，用于指示要包含多维数组模块
 */
#define _MULTIARRAYMODULE

/*
 * 清除 PY_SSIZE_T_CLEAN 宏定义，确保使用 Py_ssize_t 类型的 API
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>  // 包含 Python 核心头文件

#include "dlpack/dlpack.h"  // 包含 DLPack 头文件
#include "numpy/arrayobject.h"  // 包含 NumPy 数组对象头文件
#include "npy_argparse.h"  // 包含 NumPy 参数解析头文件
#include "npy_dlpack.h"  // 包含 NumPy DLPack 头文件
#include "npy_static_data.h"  // 包含 NumPy 静态数据头文件
#include "conversion_utils.h"  // 包含类型转换工具头文件


/*
 * 用于 NumPy 导出的 dlpack DLManagedTensor(Versioned) 的删除器函数
 */
static void
array_dlpack_deleter(DLManagedTensorVersioned *self)
{
    /*
     * 如果 Python 尚未初始化，则直接返回，避免操作导致错误
     */
    if (!Py_IsInitialized()) {
        return;
    }

    PyGILState_STATE state = PyGILState_Ensure();  // 获取全局解释器锁

    PyArrayObject *array = (PyArrayObject *)self->manager_ctx;
    /*
     * 释放 self 指向的内存块，包括其分配的 shape 和 strides 内存
     */
    PyMem_Free(self);
    /*
     * 释放 array 指向的 Python 对象，并且解除其引用计数
     */
    Py_XDECREF(array);

    PyGILState_Release(state);  // 释放全局解释器锁
}

/* TODO: Basically same as above until dlpack v0 is removed: */

/*
 * 用于未版本化的 dlpack DLManagedTensor 的删除器函数
 */
static void
array_dlpack_deleter_unversioned(DLManagedTensor *self)
{
    /*
     * 如果 Python 尚未初始化，则直接返回，避免操作导致错误
     */
    if (!Py_IsInitialized()) {
        return;
    }

    PyGILState_STATE state = PyGILState_Ensure();  // 获取全局解释器锁

    PyArrayObject *array = (PyArrayObject *)self->manager_ctx;
    /*
     * 释放 self 指向的内存块
     */
    PyMem_Free(self);
    /*
     * 释放 array 指向的 Python 对象，并且解除其引用计数
     */
    Py_XDECREF(array);

    PyGILState_Release(state);  // 释放全局解释器锁
}

/*
 * 用于 DLPack 封装的 DLManagedTensor(Versioned) 的胶囊删除器函数
 *
 * 这与 dlpack 规范完全一致
 */
static void
dlpack_capsule_deleter(PyObject *self)
{
    /*
     * 如果胶囊有效，则直接返回，避免错误操作
     */
    if (PyCapsule_IsValid(self, NPY_DLPACK_VERSIONED_USED_CAPSULE_NAME)) {
        return;
    }

    DLManagedTensorVersioned *managed =
        (DLManagedTensorVersioned *)PyCapsule_GetPointer(
            self, NPY_DLPACK_VERSIONED_CAPSULE_NAME);
    /*
     * 如果 managed 为 NULL，则输出不可解析错误信息
     */
    if (managed == NULL) {
        PyErr_WriteUnraisable(NULL);
        return;
    }
    /*
     * 如果 managed 指定了删除器函数，则调用该函数进行删除操作
     */
    if (managed->deleter) {
        managed->deleter(managed);
    }
}

/* TODO: Basically same as above until dlpack v0 is removed: */

/*
 * 用于未版本化的 DLPack 封装的 DLManagedTensor 的胶囊删除器函数
 */
static void
dlpack_capsule_deleter_unversioned(PyObject *self)
{
    /*
     * 如果胶囊有效，则直接返回，避免错误操作
     */
    if (PyCapsule_IsValid(self, NPY_DLPACK_USED_CAPSULE_NAME)) {
        return;
    }

    DLManagedTensor *managed =
        (DLManagedTensor *)PyCapsule_GetPointer(self, NPY_DLPACK_CAPSULE_NAME);
    /*
     * 如果 managed 为 NULL，则输出不可解析错误信息
     */
    if (managed == NULL) {
        PyErr_WriteUnraisable(NULL);
        return;
    }

    /*
     * 如果 managed 指定了删除器函数，则调用该函数进行删除操作
     */
    if (managed->deleter) {
        managed->deleter(managed);
    }
}

/*
 * 用于作为 `from_dlpack` 中的 `base` 使用的胶囊的删除器函数
 *
 * 这几乎与上面的函数完全相同，作为我们数组的内部基础使用，以便重命名原始胶囊
 */
static void
array_dlpack_internal_capsule_deleter(PyObject *self)
{
    # 从 Python Capsule 对象中获取 DLManagedTensorVersioned 结构体指针
    DLManagedTensorVersioned *managed =
        (DLManagedTensorVersioned *)PyCapsule_GetPointer(
            self, NPY_DLPACK_VERSIONED_INTERNAL_CAPSULE_NAME);
    
    # 如果获取的指针为空，说明未能成功获取 DLManagedTensorVersioned 结构体指针，输出未能捕获异常并返回
    if (managed == NULL) {
        PyErr_WriteUnraisable(NULL);
        return;
    }
    
    # 检查 managed 结构体中的 deleter 函数指针是否非空
    /*
     *  根据规范，如果调用方无法提供合理的析构函数，deleter 可以是 NULL。
     */
    if (managed->deleter) {
        # 调用 managed 结构体中的 deleter 函数，释放 managed 结构体及其管理的资源
        managed->deleter(managed);
        /* TODO: deleter 是否允许设置 Python 异常？ */
        # 断言当前没有 Python 异常发生
        assert(!PyErr_Occurred());
    }
}

/* TODO: Basically same as above until dlpack v0 is removed: */
// 定义一个静态函数，用于删除未版本化的 DLPack Capsule 对象
static void
array_dlpack_internal_capsule_deleter_unversioned(PyObject *self)
{
    // 从 PyCapsule 中获取 DLManagedTensor 指针
    DLManagedTensor *managed =
        (DLManagedTensor *)PyCapsule_GetPointer(
            self, NPY_DLPACK_INTERNAL_CAPSULE_NAME);
    // 如果获取失败，记录异常信息并返回
    if (managed == NULL) {
        PyErr_WriteUnraisable(NULL);
        return;
    }

    // 如果存在自定义的删除函数，则调用该函数删除 managed 指向的对象
    if (managed->deleter) {
        managed->deleter(managed);
        assert(!PyErr_Occurred());
    }
}


// This function cannot return NULL, but it can fail,
// So call PyErr_Occurred to check if it failed after
// calling it.
// 获取 PyArrayObject 对象所在的设备信息，返回 DLDevice 结构体
static DLDevice
array_get_dl_device(PyArrayObject *self) {
    // 初始化返回的 DLDevice 结构体，设备类型为 CPU，设备 ID 为 0
    DLDevice ret;
    ret.device_type = kDLCPU;
    ret.device_id = 0;
    // 获取 PyArrayObject 的基础对象
    PyObject *base = PyArray_BASE(self);

    // 遍历基础对象链，直到找到不是 PyArrayObject 类型的对象为止
    while (base != NULL && PyArray_Check(base)) {
        base = PyArray_BASE((PyArrayObject *)base);
    }

    // 如果基础对象是有效的 DLPack Capsule，获取其 DLManagedTensor，并返回其设备信息
    if (PyCapsule_IsValid(base, NPY_DLPACK_INTERNAL_CAPSULE_NAME)) {
        DLManagedTensor *managed = (DLManagedTensor *)PyCapsule_GetPointer(
                base, NPY_DLPACK_INTERNAL_CAPSULE_NAME);
        if (managed == NULL) {
            return ret;
        }
        return managed->dl_tensor.device;
    }
    // 如果基础对象是有效的版本化 DLPack Capsule，获取其 DLManagedTensorVersioned，并返回其设备信息
    else if (PyCapsule_IsValid(base, NPY_DLPACK_VERSIONED_INTERNAL_CAPSULE_NAME)) {
        DLManagedTensorVersioned *managed = (DLManagedTensorVersioned *)PyCapsule_GetPointer(
                base, NPY_DLPACK_VERSIONED_INTERNAL_CAPSULE_NAME);
        if (managed == NULL) {
            return ret;
        }
        return managed->dl_tensor.device;
    }
    // 如果不是 DLPack Capsule，则返回默认的 CPU 设备信息
    return ret;
}


/*
 * Fill the dl_tensor struct from the `self` array.
 * This struct could be versioned, but as of now is not.
 */
// 从 PyArrayObject 对象中填充 dl_tensor 结构体的信息
static int
fill_dl_tensor_information(
    DLTensor *dl_tensor, PyArrayObject *self, DLDevice *result_device)
{
    // 获取数组元素的字节大小、维度数、步长和形状信息
    npy_intp itemsize = PyArray_ITEMSIZE(self);
    int ndim = PyArray_NDIM(self);
    npy_intp *strides = PyArray_STRIDES(self);
    npy_intp *shape = PyArray_SHAPE(self);

    // 检查数组是否是 C 连续存储的，且大小不为 1，若步长不是 itemsize 的倍数，则抛出异常
    if (!PyArray_IS_C_CONTIGUOUS(self) && PyArray_SIZE(self) != 1) {
        for (int i = 0; i < ndim; ++i) {
            if (shape[i] != 1 && strides[i] % itemsize != 0) {
                PyErr_SetString(PyExc_BufferError,
                        "DLPack only supports strides which are a multiple of "
                        "itemsize.");
                return -1;
            }
        }
    }

    // 获取数组的数据类型描述对象，检查是否需要进行字节顺序调整，若需要则抛出异常
    PyArray_Descr *dtype = PyArray_DESCR(self);
    if (PyDataType_ISBYTESWAPPED(dtype)) {
        PyErr_SetString(PyExc_BufferError,
                "DLPack only supports native byte order.");
            return -1;
    }

    // 设置 dl_tensor 结构体的数据类型信息
    dl_tensor->dtype.bits = 8 * itemsize;
    dl_tensor->dtype.lanes = 1;

    // 如果数组的数据类型是布尔型，设置 dl_tensor 的数据类型为 kDLBool
    if (PyDataType_ISBOOL(dtype)) {
        dl_tensor->dtype.code = kDLBool;
    }
``
    else if (PyDataType_ISSIGNED(dtype)) {
        // 如果数据类型是有符号整数类型，则设置管理的数据类型为 kDLInt
        managed_dtype.code = kDLInt;
    }
    else if (PyDataType_ISUNSIGNED(dtype)) {
        // 如果数据类型是无符号整数类型，则设置管理的数据类型为 kDLUInt
        managed_dtype.code = kDLUInt;
    }
    else if (PyDataType_ISFLOAT(dtype)) {
        // 如果数据类型是浮点数类型
        // 我们不能确定 dtype 是 IEEE 标准或者有填充的类型
        if (itemsize > 8) {
            // 若数据类型字节大小大于 8，抛出异常并返回 -1
            PyErr_SetString(PyExc_BufferError,
                    "DLPack only supports IEEE floating point types "
                    "without padding (longdouble typically is not IEEE).");
            return -1;
        }
        // 设置管理的数据类型为 kDLFloat
        managed_dtype.code = kDLFloat;
    }
    else if (PyDataType_ISCOMPLEX(dtype)) {
        // 如果数据类型是复数类型
        // 我们不能确定 dtype 是 IEEE 标准或者有填充的类型
        if (itemsize > 16) {
            // 若数据类型字节大小大于 16，抛出异常并返回 -1
            PyErr_SetString(PyExc_BufferError,
                    "DLPack only supports IEEE floating point types "
                    "without padding (longdouble typically is not IEEE).");
            return -1;
        }
        // 设置管理的数据类型为 kDLComplex
        managed_dtype.code = kDLComplex;
    }
    else {
        // 若数据类型不是上述类型之一，抛出异常并返回 -1
        PyErr_SetString(PyExc_BufferError,
                "DLPack only supports signed/unsigned integers, float "
                "and complex dtypes.");
        return -1;
    }

    /*
     * 注意：`dlpack.h` 头文件建议/规范 `data` 必须是 256 字节对齐的。
     * 我们有意忽略这一点，因为 `__dlpack__` 规定 `byte_offset` 目前必须为 0，
     * 以兼容 pytorch：https://github.com/data-apis/array-api/issues/293#issuecomment-964111413
     *
     * 我们进一步假设即使没有 `byte_offset`，导出完全不对齐的数据也是可以接受的，
     * 因为标准没有明确拒绝这种情况。
     * 可能，从 2023 年开始，pytorch 将支持导入 `byte_offset != 0`，而 NumPy
     * 可能会选择在此后使用它。届时，NumPy 可能必须使用 `byte_offset` 以符合标准
     * （如头文件中所指定）！
     */
    // 将 PyArray 对象的数据指针赋值给 dl_tensor 的 data
    dl_tensor->data = PyArray_DATA(self);
    // 设置 dl_tensor 的 byte_offset 为 0
    dl_tensor->byte_offset = 0;
    // 将 result_device 的值赋给 dl_tensor 的 device
    dl_tensor->device = *result_device;
    // 将 managed_dtype 赋给 dl_tensor 的 dtype

    for (int i = 0; i < ndim; ++i) {
        // 将 shape 中的第 i 个元素赋给 dl_tensor 的 shape
        dl_tensor->shape[i] = shape[i];
        // 在 DLPack 中，strides 是以项为单位，而在 NumPy 中是以字节为单位
        // 将 strides 中的第 i 个元素除以 itemsize 后赋给 dl_tensor 的 strides
        dl_tensor->strides[i] = strides[i] / itemsize;
    }

    // 设置 dl_tensor 的 ndim 为 ndim
    dl_tensor->ndim = ndim;
    // 如果 PyArray 是 C 连续存储的
    // 则不需要传递 strides，将 dl_tensor 的 strides 设置为 NULL
    if (PyArray_IS_C_CONTIGUOUS(self)) {
        dl_tensor->strides = NULL;
    }
    // 再次将 dl_tensor 的 byte_offset 设置为 0

    // 返回成功标志 0
    return 0;
# 创建一个 Python C 扩展函数，用于生成 DLPack Capsule 对象，将 PyArrayObject 转换为 DLTensor
static PyObject *
create_dlpack_capsule(
        PyArrayObject *self, int versioned, DLDevice *result_device, int copied)
{
    # 获取 PyArrayObject 的维度数
    int ndim = PyArray_NDIM(self);

    /*
     * 在结构的末尾对齐形状和步幅，需要对它们进行对齐，偏移量给出包括结构大小的形状（和步幅）的偏移量。
     */
    size_t align = sizeof(int64_t);
    size_t struct_size = (
        versioned ? sizeof(DLManagedTensorVersioned) : sizeof(DLManagedTensor));

    size_t offset = (struct_size + align - 1) / align * align;
    # 分配内存以包含 DLTensor 结构及其后续的形状和步幅数据
    void *ptr = PyMem_Malloc(offset + (sizeof(int64_t) * ndim * 2));
    if (ptr == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    DLTensor *dl_tensor;
    PyCapsule_Destructor capsule_deleter;
    const char *capsule_name;

    if (versioned) {
        # 如果是带版本的 DLManagedTensor，则将指针转换为 DLManagedTensorVersioned 类型
        DLManagedTensorVersioned *managed = (DLManagedTensorVersioned *)ptr;
        capsule_name = NPY_DLPACK_VERSIONED_CAPSULE_NAME;
        capsule_deleter = (PyCapsule_Destructor)dlpack_capsule_deleter;
        managed->deleter = array_dlpack_deleter;
        managed->manager_ctx = self;

        dl_tensor = &managed->dl_tensor;

        /* 版本化的张量有额外的字段需要设置 */
        managed->version.major = 1;
        managed->version.minor = 0;

        managed->flags = 0;
        if (!PyArray_CHKFLAGS(self, NPY_ARRAY_WRITEABLE)) {
            managed->flags |= DLPACK_FLAG_BITMASK_READ_ONLY;
        }
        if (copied) {
            managed->flags |= DLPACK_FLAG_BITMASK_IS_COPIED;
        }
    }
    else {
        # 如果是不带版本的 DLManagedTensor，则将指针转换为 DLManagedTensor 类型
        DLManagedTensor *managed = (DLManagedTensor *)ptr;
        capsule_name = NPY_DLPACK_CAPSULE_NAME;
        capsule_deleter = (PyCapsule_Destructor)dlpack_capsule_deleter_unversioned;
        managed->deleter = array_dlpack_deleter_unversioned;
        managed->manager_ctx = self;

        dl_tensor = &managed->dl_tensor;
    }

    # 设置 DLTensor 的形状指针为偏移后的位置
    dl_tensor->shape = (int64_t *)((char *)ptr + offset);
    /* 注意：如果是 C 连续的话，步幅可能会在后续设置为 NULL */
    dl_tensor->strides = dl_tensor->shape + ndim;

    # 填充 DLTensor 的信息，包括形状、步幅和设备信息
    if (fill_dl_tensor_information(dl_tensor, self, result_device) < 0) {
        PyMem_Free(ptr);
        return NULL;
    }

    # 创建一个 Python Capsule 对象，封装指针、名称和析构函数
    PyObject *capsule = PyCapsule_New(ptr, capsule_name, capsule_deleter);
    if (capsule == NULL) {
        PyMem_Free(ptr);
        return NULL;
    }

    // Capsule 持有对 self 的引用
    Py_INCREF(self);

    return capsule;
}
    # 如果设备类型是 kDLCPU 并且设备 ID 是 0，则执行以下操作
    if (type == kDLCPU && id == 0) {
        # 将设备类型和设备 ID 分配给结果设备结构体
        result_device->device_type = type;
        result_device->device_id = id;
        # 返回成功的状态码
        return NPY_SUCCEED;
    }

    # 如果条件不满足，则设置一个异常，说明请求的设备不受支持
    PyErr_SetString(PyExc_ValueError, "unsupported device requested");
    # 返回失败的状态码
    return NPY_FAIL;
/* 结束当前函数，返回一个 PyObject 类型的指针 */
NPY_NO_EXPORT PyObject *
array_dlpack(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    /* 初始化变量 stream 和 max_version，均指向 Py_None */
    PyObject *stream = Py_None;
    PyObject *max_version = Py_None;
    /* 设置默认的数据复制模式为 NPY_COPY_IF_NEEDED */
    NPY_COPYMODE copy_mode = NPY_COPY_IF_NEEDED;
    /* 初始化 major_version 为 0 */
    long major_version = 0;
    /* 获取当前数组的设备信息，保存在 result_device 中 */
    DLDevice result_device = array_get_dl_device(self);
    /* 如果之前有异常发生，则直接返回 NULL */
    if (PyErr_Occurred()) {
        return NULL;
    }

    /* 准备解析参数，根据函数名、参数、关键字参数等解析 */
    NPY_PREPARE_ARGPARSER;
    /* 如果解析参数出错，则返回 NULL */
    if (npy_parse_arguments("__dlpack__", args, len_args, kwnames,
            "$stream", NULL, &stream,
            "$max_version", NULL, &max_version,
            "$dl_device", &device_converter, &result_device,
            "$copy", &PyArray_CopyConverter, &copy_mode,
            NULL, NULL, NULL)) {
        return NULL;
    }

    /* 如果 max_version 不是 Py_None，则检查其格式是否正确 */
    if (max_version != Py_None) {
        /* max_version 必须是一个含有两个元素的元组 */
        if (!PyTuple_Check(max_version) || PyTuple_GET_SIZE(max_version) != 2) {
            PyErr_SetString(PyExc_TypeError,
                    "max_version must be None or a tuple with two elements.");
            return NULL;
        }
        /* 将元组中的第一个元素转换为 long 类型的 major_version */
        major_version = PyLong_AsLong(PyTuple_GET_ITEM(max_version, 0));
        /* 如果转换出错，则返回 NULL */
        if (major_version == -1 && PyErr_Occurred()) {
            return NULL;
        }
    }

    /* 如果 stream 不是 Py_None，则抛出运行时异常 */
    if (stream != Py_None) {
        PyErr_SetString(PyExc_RuntimeError,
                "NumPy only supports stream=None.");
        return NULL;
    }

    /* 如果需要始终复制数组，则创建数组的副本 */
    if (copy_mode == NPY_COPY_ALWAYS) {
        /* TODO: 可能需要先检查导出数据类型的能力 */
        /* 创建当前数组的副本，保持原有顺序 */
        self = (PyArrayObject *)PyArray_NewCopy(self, NPY_KEEPORDER);
        /* 如果创建副本失败，则返回 NULL */
        if (self == NULL) {
            return NULL;
        }
    }
    else {
        /* 增加当前数组的引用计数 */
        Py_INCREF(self);
    }

    /* 如果 major_version 小于 1，并且当前数组是只读的，则抛出异常 */
    if (major_version < 1 && !(PyArray_FLAGS(self) & NPY_ARRAY_WRITEABLE)) {
        PyErr_SetString(PyExc_BufferError,
            "Cannot export readonly array since signalling readonly "
            "is unsupported by DLPack (supported by newer DLPack version).");
        /* 减少当前数组的引用计数，并返回 NULL */
        Py_DECREF(self);
        return NULL;
    }

    /*
     * TODO: DLPack 的带版本和不带版本的结构非常相似，但不兼容 ABI，
     * 因此此处调用的函数需要分支（似乎不值得使用模板）。
     *
     * 在 NumPy 2.1 中应该弃用版本 0 的支持，并可以移除这些分支。
     */
    /* 创建一个 DLPack 的封装 Capsule 对象，传入当前数组和相关参数 */
    PyObject *res = create_dlpack_capsule(
            self, major_version >= 1, &result_device,
            copy_mode == NPY_COPY_ALWAYS);
    /* 减少当前数组的引用计数 */
    Py_DECREF(self);

    /* 返回创建的结果对象 */
    return res;
}
    // 定义函数 `from_dlpack`，接受多个参数，返回一个 PyObject 指针
    from_dlpack(PyObject *NPY_UNUSED(self),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
    {
        // 初始化变量 obj, copy, device，都指向 Py_None
        PyObject *obj, *copy = Py_None, *device = Py_None;
        // 使用宏 NPY_PREPARE_ARGPARSER 准备参数解析
        NPY_PREPARE_ARGPARSER;
        // 调用 npy_parse_arguments 解析参数，若失败返回 NULL
        if (npy_parse_arguments("from_dlpack", args, len_args, kwnames,
            "obj", NULL, &obj,
            "$copy", NULL, &copy,
            "$device", NULL, &device,
            NULL, NULL, NULL) < 0) {
            return NULL;
        }

        /* 准备调用对象的 __dlpack__() 方法所需的参数 */
        // 静态变量 call_kwnames 用于指定参数名列表
        static PyObject *call_kwnames = NULL;
        // 静态变量 dl_cpu_device_tuple 用于表示 CPU 设备
        static PyObject *dl_cpu_device_tuple = NULL;
        // 静态变量 max_version 用于指定最大版本号
        static PyObject *max_version = NULL;

        // 初始化 call_kwnames，若失败返回 NULL
        if (call_kwnames == NULL) {
            call_kwnames = Py_BuildValue("(sss)", "dl_device", "copy", "max_version");
            if (call_kwnames == NULL) {
                return NULL;
            }
        }
        // 初始化 dl_cpu_device_tuple，若失败返回 NULL
        if (dl_cpu_device_tuple == NULL) {
            dl_cpu_device_tuple = Py_BuildValue("(i,i)", 1, 0);
            if (dl_cpu_device_tuple == NULL) {
                return NULL;
            }
        }
        // 初始化 max_version，若失败返回 NULL
        if (max_version == NULL) {
            max_version = Py_BuildValue("(i,i)", 1, 0);
            if (max_version == NULL) {
                return NULL;
            }
        }

        /* 
         * 为完整调用准备参数。始终传递 copy 并传递我们的 max_version。
         * `device` 始终传递为 `None`，但如果用户提供了设备，则用 (1, 0) 替换之。
         */
        // 创建参数数组 call_args
        PyObject *call_args[] = {obj, Py_None, copy, max_version};
        // 设置参数数量为 1，使用 PY_VECTORCALL_ARGUMENTS_OFFSET 偏移
        Py_ssize_t nargsf = 1 | PY_VECTORCALL_ARGUMENTS_OFFSET;

        /* 若 device 不是 Py_None，则替换为 (1, 0) */
        if (device != Py_None) {
            /* 检查 device 是否为 CPU */
            NPY_DEVICE device_request = NPY_DEVICE_CPU;
            if (!PyArray_DeviceConverterOptional(device, &device_request)) {
                return NULL;
            }
            // 断言设备请求为 NPY_DEVICE_CPU
            assert(device_request == NPY_DEVICE_CPU);
            // 将 call_args[1] 设置为 dl_cpu_device_tuple
            call_args[1] = dl_cpu_device_tuple;
        }

        // 调用 PyObject_VectorcallMethod 执行对象的 __dlpack__() 方法
        PyObject *capsule = PyObject_VectorcallMethod(
            npy_interned_str.__dlpack__, call_args, nargsf, call_kwnames);
        // 若调用失败，尝试不带任何参数（若 device 和 copy 均为 None）
        if (capsule == NULL) {
            if (PyErr_ExceptionMatches(PyExc_TypeError)
                && device == Py_None && copy == Py_None) {
                PyErr_Clear();
                capsule = PyObject_VectorcallMethod(
                    npy_interned_str.__dlpack__, call_args, nargsf, NULL);
            }
            if (capsule == NULL) {
                return NULL;
            }
        }

        // 定义变量 managed_ptr, dl_tensor, readonly, versioned，检查版本化的 Capsule
        void *managed_ptr;
        DLTensor dl_tensor;
        int readonly;
        int versioned = PyCapsule_IsValid(capsule, NPY_DLPACK_VERSIONED_CAPSULE_NAME);
    # 如果 versioned 标志为真，则从 PyCapsule 中获取指针并转换为 DLManagedTensorVersioned 结构体指针
    if (versioned) {
        managed_ptr = PyCapsule_GetPointer(capsule, NPY_DLPACK_VERSIONED_CAPSULE_NAME);
        DLManagedTensorVersioned *managed = (DLManagedTensorVersioned *)managed_ptr;
        
        # 如果获取的 managed 指针为空，则释放 Python 对象并返回空指针
        if (managed == NULL) {
            Py_DECREF(capsule);
            return NULL;
        }
        
        # 如果 managed 结构体中的 major 版本号大于 1，则设定错误信息并释放 Python 对象后返回空指针
        if (managed->version.major > 1) {
            PyErr_SetString(PyExc_BufferError,
                "from_dlpack(): the exported DLPack major version is too "
                "high to be imported by this version of NumPy.");
            Py_DECREF(capsule);
            return NULL;
        }
        
        # 获取 DLManagedTensorVersioned 结构体中的 DLTensor 指针及是否只读标志
        dl_tensor = managed->dl_tensor;
        readonly = (managed->flags & DLPACK_FLAG_BITMASK_READ_ONLY) != 0;
    }
    # 如果 versioned 标志为假，则从 PyCapsule 中获取指针并转换为 DLManagedTensor 结构体指针
    else {
        managed_ptr = PyCapsule_GetPointer(capsule, NPY_DLPACK_CAPSULE_NAME);
        DLManagedTensor *managed = (DLManagedTensor *)managed_ptr;
        
        # 如果获取的 managed 指针为空，则释放 Python 对象并返回空指针
        if (managed == NULL) {
            Py_DECREF(capsule);
            return NULL;
        }
        
        # 获取 DLManagedTensor 结构体中的 DLTensor 指针，并将只读标志设置为 0
        dl_tensor = managed->dl_tensor;
        readonly = 0;
    }

    # 获取 DLTensor 的维度数
    const int ndim = dl_tensor.ndim;
    
    # 如果维度数大于 NPY_MAXDIMS（定义的最大维度数），设定错误信息并释放 Python 对象后返回空指针
    if (ndim > NPY_MAXDIMS) {
        PyErr_SetString(PyExc_RuntimeError,
                "maxdims of DLPack tensor is higher than the supported "
                "maxdims.");
        Py_DECREF(capsule);
        return NULL;
    }

    # 获取 DLTensor 的设备类型
    DLDeviceType device_type = dl_tensor.device.device_type;
    
    # 如果设备类型不是支持的类型（CPU、CUDA 主机、ROCM 主机、CUDA 管理），设定错误信息并释放 Python 对象后返回空指针
    if (device_type != kDLCPU &&
            device_type != kDLCUDAHost &&
            device_type != kDLROCMHost &&
            device_type != kDLCUDAManaged) {
        PyErr_SetString(PyExc_RuntimeError,
                "Unsupported device in DLTensor.");
        Py_DECREF(capsule);
        return NULL;
    }

    # 如果 DLTensor 的数据类型的 lanes 不为 1，设定错误信息并释放 Python 对象后返回空指针
    if (dl_tensor.dtype.lanes != 1) {
        PyErr_SetString(PyExc_RuntimeError,
                "Unsupported lanes in DLTensor dtype.");
        Py_DECREF(capsule);
        return NULL;
    }

    # 初始化 typenum 变量为 -1，用于存储 NumPy 类型码
    int typenum = -1;
    
    # 获取 DLTensor 的数据类型的 bits 和 itemsize
    const uint8_t bits = dl_tensor.dtype.bits;
    const npy_intp itemsize = bits / 8;
    
    # 根据 DLTensor 的数据类型 code 执行不同的分支
    switch (dl_tensor.dtype.code) {
    case kDLBool:
        # 如果 bits 等于 8，则设置 typenum 为 NPY_BOOL
        if (bits == 8) {
            typenum = NPY_BOOL;
        }
        break;
    case kDLInt:
        # 根据 bits 的不同值设置 typenum 为对应的整数类型
        switch (bits)
        {
            case 8: typenum = NPY_INT8; break;
            case 16: typenum = NPY_INT16; break;
            case 32: typenum = NPY_INT32; break;
            case 64: typenum = NPY_INT64; break;
        }
        break;
    case kDLUInt:
        # 根据 bits 的不同值设置 typenum 为对应的无符号整数类型
        switch (bits)
        {
            case 8: typenum = NPY_UINT8; break;
            case 16: typenum = NPY_UINT16; break;
            case 32: typenum = NPY_UINT32; break;
            case 64: typenum = NPY_UINT64; break;
        }
        break;
    case kDLFloat:
        # 根据 bits 的不同值设置 typenum 为对应的浮点数类型
        switch (bits)
        {
            case 16: typenum = NPY_FLOAT16; break;
            case 32: typenum = NPY_FLOAT32; break;
            case 64: typenum = NPY_FLOAT64; break;
        }
        break;
    // 检查 DLTensor 的数据类型，并根据其位数设置 NumPy 的数据类型
    case kDLComplex:
        switch (bits)
        {
            // 对于复数，根据位数选择相应的 NumPy 数据类型
            case 64: typenum = NPY_COMPLEX64; break;  // 64位复数类型
            case 128: typenum = NPY_COMPLEX128; break;  // 128位复数类型
        }
        break;
    }

    // 如果未能识别有效的数据类型，抛出运行时错误并清理资源后返回空值
    if (typenum == -1) {
        PyErr_SetString(PyExc_RuntimeError,
                "Unsupported dtype in DLTensor.");
        Py_DECREF(capsule);
        return NULL;
    }

    // 初始化形状和步长数组
    npy_intp shape[NPY_MAXDIMS];
    npy_intp strides[NPY_MAXDIMS];

    // 遍历 DLTensor 的每个维度，将其形状信息复制到 NumPy 的形状数组中
    for (int i = 0; i < ndim; ++i) {
        shape[i] = dl_tensor.shape[i];
        // 如果 DLTensor 有步长信息，将其转换为字节单位并复制到 NumPy 的步长数组中
        if (dl_tensor.strides != NULL) {
            strides[i] = dl_tensor.strides[i] * itemsize;
        }
    }

    // 计算数据指针的位置，考虑 DLTensor 的偏移量
    char *data = (char *)dl_tensor.data + dl_tensor.byte_offset;

    // 根据 typenum 创建 NumPy 的数据描述符对象
    PyArray_Descr *descr = PyArray_DescrFromType(typenum);
    if (descr == NULL) {
        Py_DECREF(capsule);
        return NULL;
    }

    // 使用描述符、形状、步长等信息创建 NumPy 数组对象
    PyObject *ret = PyArray_NewFromDescr(&PyArray_Type, descr, ndim, shape,
            dl_tensor.strides != NULL ? strides : NULL, data, 0, NULL);
    if (ret == NULL) {
        Py_DECREF(capsule);
        return NULL;
    }

    // 如果数组是只读的，清除写入标志
    if (readonly) {
        PyArray_CLEARFLAGS((PyArrayObject *)ret, NPY_ARRAY_WRITEABLE);
    }

    // 根据 versioned 标志创建 PyCapsule 对象
    PyObject *new_capsule;
    if (versioned) {
        new_capsule = PyCapsule_New(managed_ptr,
            NPY_DLPACK_VERSIONED_INTERNAL_CAPSULE_NAME,
            (PyCapsule_Destructor)array_dlpack_internal_capsule_deleter);
    }
    else {
        new_capsule = PyCapsule_New(managed_ptr,
            NPY_DLPACK_INTERNAL_CAPSULE_NAME,
            (PyCapsule_Destructor)array_dlpack_internal_capsule_deleter_unversioned);
    }

    // 如果创建 PyCapsule 失败，清理之前分配的资源并返回空值
    if (new_capsule == NULL) {
        Py_DECREF(capsule);
        Py_DECREF(ret);
        return NULL;
    }

    // 将新创建的 PyCapsule 对象设置为 NumPy 数组对象的基础对象
    if (PyArray_SetBaseObject((PyArrayObject *)ret, new_capsule) < 0) {
        Py_DECREF(capsule);
        Py_DECREF(ret);
        return NULL;
    }

    // 根据 versioned 标志设置 PyCapsule 的名称
    const char *new_name = (
        versioned ? NPY_DLPACK_VERSIONED_USED_CAPSULE_NAME
                  : NPY_DLPACK_USED_CAPSULE_NAME);
    if (PyCapsule_SetName(capsule, new_name) < 0) {
        Py_DECREF(capsule);
        Py_DECREF(ret);
        return NULL;
    }

    // 清理原始的 PyCapsule 对象并返回创建的 NumPy 数组对象
    Py_DECREF(capsule);
    return ret;
}


注释：


# 这行代码结束了一个代码块的定义或循环。在大多数编程语言中，用大括号 } 表示代码块的结束。
# 在这里，它标志着一个函数或者一个条件语句的结束，具体取决于前面的代码逻辑。
# 通常，这行代码与一个函数定义的开头或者一个 if/else 结构的起始部分对应。
```