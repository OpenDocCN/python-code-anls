# `.\pytorch\functorch\csrc\dim\dim.cpp`

```
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/csrc/utils/python_compat.h>

// Many APIs have changed/don't exist anymore
#if IS_PYTHON_3_12_PLUS

#include "dim.h"

// Re-enable this some day
// 函数 Dim_init()：初始化函数，返回一个错误，指示不支持 Python 3.12 的运行时错误
PyObject* Dim_init() {
    PyErr_SetString(PyExc_RuntimeError, "First class dim doesn't work with python 3.12");
    return nullptr;
}

#else

#include "minpybind.h"
#include <frameobject.h>
#include <opcode.h>
#include <utility>
#include <new>
#include <iostream>
#include <vector>
//#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/Export.h>
#include <ATen/functorch/BatchedTensorImpl.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/ATen.h>
#include <memory>
#include "arena.h"
#include "dim.h"
#include "python_variable_simple.h"

#if IS_PYTHON_3_11_PLUS
#define Py_BUILD_CORE
#include "internal/pycore_opcode.h"
#undef Py_BUILD_CORE
#endif

// C++ API functions for objects to
// * construct the object, returning a ref-counted handle
// * The actual API, with methods that take/return C-typed values

// extend minpybind.h to include
// * typed handles so that -> can get to their raw API
// * object/handle distinction for the typed handles

// class Dim: ---------------
// 下面开始定义一系列全局变量和函数声明，用于与 Python 交互和处理 Tensor 数据

mpy::handle torch_Tensor___mul__;
mpy::handle _Tensor;
mpy::handle _Tensor_sum;
mpy::handle NamedTuple;
mpy::dict_view pointwise;
mpy::handle torch_Tensor_expand;
binaryfunc THPVariable_getitem;
objobjargproc THPVariable_setitem;
mpy::handle no_slice;
PyTypeObject* torch_Tensor;
mpy::handle torch_Tensor_copy_;
mpy::handle torch_Tensor_split;
bool pointwise_optimize = true;
PyTypeObject* DimType = nullptr;

// 函数声明
PyObject* Tensor_getitem(PyObject* self, PyObject* index);
int Tensor_setitem(PyObject* self, PyObject* index, PyObject* value);

namespace {
// 可能的全局变量初始化函数
void maybeInitializeGlobals() {
    // 依赖于 Python dim 库的全局变量初始化
    if (_Tensor.ptr()) {
        return;
    }
    auto dim = mpy::import("functorch.dim");
    _Tensor = dim.attr("_Tensor");
    pointwise = dim.attr("pointwise");
    _Tensor_sum = _Tensor.attr("sum");
    DimType = (PyTypeObject*)mpy::import("functorch.dim").attr("Dim").ptr();
}

// 替换映射操作的函数
void replaceMappingIfMatches(mpy::handle tp) {
    auto T = (PyTypeObject*)tp.ptr();
    bool recurse = false;
    // 如果映射对象支持这个特定的获取子项操作，替换为自定义函数 Tensor_getitem
    if (T->tp_as_mapping->mp_subscript == THPVariable_getitem) {
        T->tp_as_mapping->mp_subscript = Tensor_getitem;
        recurse = true;
    }
    // 如果映射对象支持这个特定的设置子项操作，替换为自定义函数 Tensor_setitem
    if (T->tp_as_mapping->mp_ass_subscript == THPVariable_setitem) {
        T->tp_as_mapping->mp_ass_subscript = Tensor_setitem;
        recurse = true;
    }
    # 如果递归标志为真，则执行以下代码块
    if (recurse) {
        # 调用对象的 `__subclasses__` 属性，并获取返回结果
        auto result = tp.attr("__subclasses__").call();
        # 将返回结果转换为列表视图
        mpy::list_view lv(result);
        # 遍历列表视图中的每一个元素
        for (auto i : lv.enumerate()) {
            # 对列表视图中的每个元素调用 replaceMappingIfMatches 函数
            replaceMappingIfMatches(lv[i]);
        }
    }
}

void initializeGlobals(Arena & A) {
    // 导入 torch 模块并获取 Tensor 类型对象
    auto torch = mpy::import("torch");
    // 设置全局变量 torch_Tensor 指向 Torch 的 Tensor 类型对象
    torch_Tensor = (PyTypeObject*) torch.attr("Tensor").ptr();
    // 获取并设置 torch_Tensor___mul__ 为 Torch Tensor 对象的乘法方法
    torch_Tensor___mul__ = torch.attr("Tensor").attr("__mul__");

    // 获取并设置 torch_Tensor_expand 为 Torch TensorBase 类的 expand 方法
    torch_Tensor_expand = torch.attr("_C").attr("TensorBase").attr("expand");
    // 获取并设置 torch_Tensor_split 为 Torch TensorBase 类的 split 方法
    torch_Tensor_split = torch.attr("_C").attr("TensorBase").attr("split");
    // 获取并设置 torch_Tensor_copy_ 为 Torch Tensor 类的 copy_ 方法
    torch_Tensor_copy_ = torch.attr("Tensor").attr("copy_");
    // 获取 py_TensorBase 对象并设置 TensorBase 为其对应的 PyTypeObject 指针
    auto py_TensorBase = torch.attr("_C").attr("TensorBase");
    auto TensorBase = (PyTypeObject*) py_TensorBase.ptr();
    // 获取并设置 THPVariable_getitem 为 TensorBase 的 tp_as_mapping 的 mp_subscript 方法
    THPVariable_getitem = TensorBase->tp_as_mapping->mp_subscript;
    // 获取并设置 THPVariable_setitem 为 TensorBase 的 tp_as_mapping 的 mp_ass_subscript 方法
    THPVariable_setitem = TensorBase->tp_as_mapping->mp_ass_subscript;
    // 导入 typing 模块并设置 NamedTuple 为其 NamedTuple 类型
    NamedTuple = mpy::import("typing").attr("NamedTuple");
    // 创建一个空的切片对象 no_slice
    no_slice = PySlice_New(NULL, NULL, NULL);

}

mpy::handle DimensionBindError_;
mpy::handle DimensionBindError() {
    // 如果 DimensionBindError_ 未初始化，则导入 functorch.dim 模块并设置 DimensionBindError_ 为 DimensionBindError 类型
    if(!DimensionBindError_.ptr()) {
        DimensionBindError_ = mpy::import("functorch.dim").attr("DimensionBindError");
    }
    // 返回 DimensionBindError_ 对象句柄
    return DimensionBindError_;
}

static int64_t n_dims_created = 65;

struct Dim : public mpy::base<Dim> {
    int64_t level_; // for stable comparisons in prototype
    mpy::object name_;
    Dim()
    : level_(n_dims_created++) {}
    void init(mpy::object name, int64_t s = -1) {
        // 初始化 Dim 对象的 name_ 成员变量为给定的 name 对象
        name_ = std::move(name);
        // 初始化 size_ 成员变量为给定的 s 值，默认为 -1
        size_ = s;
    }

    static bool check_exact(mpy::handle v) {
        // 检查给定的对象 v 是否为 DimType 类型
        return Py_TYPE(v.ptr()) == DimType;
    }

    int64_t size() const {
        // 如果 size_ 值为 -1，则抛出异常表示未绑定维度
        if (size_ == -1) {
            mpy::raise_error(PyExc_ValueError, "dimension %S is unbound", name_.ptr());
        }
        // 返回 size_ 的值
        return size_;
    }
    void set_size(int64_t v) {
        // 如果 size_ 值为 -1，则将其设置为 v
        if (size_ == -1) {
            size_ = v;
        } else if(size_ != v) {
            // 如果 size_ 不为 -1 且与 v 不相等，则抛出维度绑定错误异常
            mpy::raise_error(DimensionBindError(), "Dim '%R' previously bound to a dimension of size %lld cannot bind to a dimension of size %lld", this, this->size_, v);
        }
    }
    bool is_bound() const {
        // 返回 size_ 是否不等于 -1，即是否已绑定维度
        return size_ != -1;
    }
    static mpy::obj<Dim> create(mpy::object name, int64_t s = -1) {
        // 如果 DimType 未初始化，则调用 maybeInitializeGlobals() 进行初始化
        if (!DimType) {
            maybeInitializeGlobals();
        }
        // 创建并返回一个新的 Dim 对象，初始化其 name 和 size 成员变量
        auto r = Dim::alloc(DimType);
        r->init(std::move(name), s);
        return r;
    }
    static PyTypeObject Type;
    const at::Tensor& range() {
        // 如果 range_ 未定义，则使用 at::arange(size()) 创建一个新的 Tensor 对象
        if (!range_.defined()) {
            range_ = at::arange(size());
        }
        // 返回 range_ 对象的引用
        return range_;
    }
    const at::Tensor& batchtensor() {
        // 如果 batchtensor_ 未定义，则使用 at::functorch::addBatchDim(range(), 0, level_) 创建一个新的 Tensor 对象
        if (!batchtensor_.defined()) {
            batchtensor_ = at::functorch::addBatchDim(range(), 0, level_);
        }
        // 返回 batchtensor_ 对象的引用
        return batchtensor_;
    }
private:
    int64_t size_{-1};
    at::Tensor range_;
    at::Tensor batchtensor_;
};


struct DimEntry {
    // 联合体，包含一个负数表示该维度来自右手边的哪个维度，或一个指向一级维度的指针。
    // 负数不会设置其最高位，因此检查数字是否为负数告诉我们它不是一个维度。
    # 检查当前对象的数据是否小于 0，判断是否是位置参数
    bool is_positional() const {
        return data_ < 0;
    }

    # 检查当前对象的数据是否等于 0，判断是否为空值
    bool is_none() const {
        return data_ == 0;
    }

    # 返回当前对象的数据，作为位置信息
    int64_t position() const {
        return data_;
    }

    # 从当前对象的数据中提取指针类型 Dim*，并封装成 mpy::hdl<Dim> 类型返回
    mpy::hdl<Dim> dim() const {
        Dim* result;
        std::memcpy(&result, &data_, sizeof(Dim*));
        return mpy::hdl<Dim>(result);
    }

    # 默认构造函数，初始化数据为 0
    DimEntry()
    : data_(0) {}

    # 使用给定的整数值初始化对象，同时断言该值小于 0
    DimEntry(int64_t pos)
    : data_(pos) {
        AT_ASSERT(pos < 0);
    }

    # 使用给定的 mpy::hdl<Dim> 对象初始化当前对象的数据
    DimEntry(mpy::hdl<Dim> d) {
       std::memcpy(&data_, &d, sizeof(int64_t));
    }

    # 判断当前对象与另一个 DimEntry 对象是否相等，比较它们的数据成员
    bool operator==(const DimEntry& rhs) const {
        return data_ == rhs.data_;
    }
private:
    int64_t data_;  // 私有数据成员，存储 int64_t 类型的数据

};

std::ostream& operator<<(std::ostream& ss, DimEntry entry) {
    // 自定义输出流操作符重载函数，根据 DimEntry 对象的状态输出不同的信息到流 ss 中
    if (entry.is_none()) {
        ss << "None";  // 如果 entry 表示 None，则输出字符串 "None"
    } else if (entry.is_positional()) {
        ss << entry.position();  // 如果 entry 是位置参数，则输出其位置信息
    } else {
        ss << entry.dim();  // 否则输出其维度信息
    }
    return ss;  // 返回输出流对象
}

// Dim wrapper methods
DimEntry _wrap_dim(mpy::handle d, size_t N, bool keepdim) {
    // 封装 Dim 对象的方法，根据参数 d 和条件 keepdim 进行不同的封装处理
    if (Dim::check(d)) {
        if (keepdim) {
            mpy::raise_error(PyExc_ValueError, "cannot preserve first-class dimensions with keepdim=True");
            // 如果 keepdim 为真，抛出 ValueError 异常，表示不能保留一级维度
        }
        return Dim::unchecked_wrap(d);  // 将 d 封装成 DimEntry 对象并返回
    } else if (mpy::is_int(d)) {
        auto i = mpy::to_int(d);  // 如果 d 是整数，将其转换为 int
        while (i >= 0) {
            i -= N;  // 如果 i 大于等于 0，将其减去 N
        }
        return i;  // 返回处理后的整数作为 DimEntry 对象
    } else {
        return DimEntry();  // 否则返回默认构造的 DimEntry 对象
    }
}


int Dim_init(mpy::hdl<Dim> self, PyObject *args, PyObject *kwds) {
    PY_BEGIN  // 宏，开始 Python C API 的异常处理块
    static constexpr const char* kwlist[] = {"name", "size", nullptr};
    mpy::handle name;
    mpy::handle size = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", const_cast<char **>(kwlist), &name, &size)) {
        return -1;  // 解析参数失败，返回 -1 表示错误
    }
    self->init(mpy::object::borrow(name), (size.ptr() && !mpy::is_none(size)) ? mpy::to_int(size) : -1);
    // 调用 self 对象的 init 方法进行初始化，传入 name 和 size 参数
    return 0;  // 返回 0 表示成功
    PY_END(-1)  // 宏，结束 Python C API 的异常处理块，处理异常返回 -1
}

PyObject* Dim_repr(Dim* self) {
    PY_BEGIN  // 宏，开始 Python C API 的异常处理块
    mpy::object name = (self->name_.ptr()) ? self->name_ : mpy::unicode_from_string("<uninitialized dim>");
    // 获取 Dim 对象的名称属性，若未初始化则返回默认字符串 "<uninitialized dim>"
    return name.release();  // 返回名称对象，释放所有权
    PY_END(nullptr)  // 宏，结束 Python C API 的异常处理块，处理异常返回 nullptr
}


PyObject* Dim_getsize(Dim* self, void*) {
    PY_BEGIN  // 宏，开始 Python C API 的异常处理块
    return mpy::from_int(self->size()).release();  // 返回 Dim 对象的大小属性，并释放所有权
    PY_END(nullptr)  // 宏，结束 Python C API 的异常处理块，处理异常返回 nullptr
}

int Dim_setsize(Dim* self, PyObject* size, void*) {
    PY_BEGIN  // 宏，开始 Python C API 的异常处理块
    self->set_size(mpy::to_int(size));  // 设置 Dim 对象的大小属性为 size
    return 0;  // 返回 0 表示成功
    PY_END(-1)  // 宏，结束 Python C API 的异常处理块，处理异常返回 -1
}

PyObject* Dim_getis_bound(Dim* self, void*) {
    return PyBool_FromLong(self->is_bound());  // 返回 Dim 对象的 is_bound() 方法的布尔结果
}

PyObject* Dim_getlevel(Dim* self, void*) {
    return PyLong_FromLong(self->level_);  // 返回 Dim 对象的 level_ 属性的长整型值
}

PyObject* Dim_get_levels(Dim* self, void*) {
    mpy::tuple t(1);  // 创建包含一个元素的元组对象 t
    t.set(0, mpy::object::borrow(self->ptr()));  // 将 Dim 对象的指针属性设置为元组的第一个元素
    return t.release();  // 返回元组对象，并释放所有权
}

PyObject* Dim_get_has_device(Dim* self, void*) {
    Py_RETURN_FALSE;  // 返回 Python 中的 False 值
}

PyObject* Dim_get_tensor(Dim* self, void*) {
    return THPVariable_Wrap(self->range());  // 返回 Dim 对象的 range() 方法的包装对象
}

PyObject* Dim_get_batchtensor(Dim* self, void*) {
    return THPVariable_Wrap(self->batchtensor());  // 返回 Dim 对象的 batchtensor() 方法的包装对象
}


PyGetSetDef Dim_getsetters[] = {
    {"size", (getter) Dim_getsize, (setter) Dim_setsize,
     "Dimension size", NULL},  // size 属性的 getter 和 setter 方法定义
    {"is_bound", (getter) Dim_getis_bound, NULL, "is_bound", NULL},  // is_bound 属性的 getter 方法定义
    {"_level", (getter) Dim_getlevel, NULL, "_level", NULL},  // _level 属性的 getter 方法定义
    {"_levels", (getter) Dim_get_levels, NULL, "_levels", NULL},  // _levels 属性的 getter 方法定义
    {"_has_device", (getter) Dim_get_has_device, NULL, "_has_device", NULL},  // _has_device 属性的 getter 方法定义
    {"_tensor", (getter) Dim_get_tensor, NULL, "_tensor", NULL},  // _tensor 属性的 getter 方法定义
    {"_batchtensor", (getter) Dim_get_batchtensor, NULL, "_batchtensor", NULL},  // _batchtensor 属性的 getter 方法定义
    {"ndim", (getter) [](PyObject* self, void*) -> PyObject* { return mpy::from_int(1).release(); }, NULL, "ndim", NULL},
    // ndim 属性的 getter 方法定义，返回整数 1
    {NULL}  /* Sentinel */  // PyGetSetDef 数组的结尾标记
};
PyTypeObject Dim::Type = {
    PyVarObject_HEAD_INIT(NULL, 0)  // 初始化 PyVarObject 的头部，此处为基类指针和大小
    "_C.Dim",                       /* tp_name */  // 类型名称字符串
    sizeof(Dim),                    /* tp_basicsize */  // 类型基本大小
    0,                              /* tp_itemsize */  // 每个项目的大小（对变长对象有用）
    Dim::dealloc_stub,               /* tp_dealloc */  // 析构函数指针
    0,                              /* tp_vectorcall_offset */  // Vectorcall 调用偏移量
    0,                              /* tp_getattr */  // 获取属性函数
    0,                              /* tp_setattr */  // 设置属性函数
    0,                              /* tp_as_async */  // 异步处理方法表
    (reprfunc)Dim_repr,             /* tp_repr */  // repr 方法
    0,                              /* tp_as_number */  // 数值方法表
    0,                              /* tp_as_sequence */  // 序列方法表
    0,                              /* tp_as_mapping */  // 映射方法表
    0,                              /* tp_hash */  // 哈希函数
    0,                              /* tp_call */  // 调用对象函数
    0,                              /* tp_str */  // 转换为字符串函数
    0,                              /* tp_getattro */  // 获取属性对象函数
    0,                              /* tp_setattro */  // 设置属性对象函数
    0,                              /* tp_as_buffer */  // 缓冲区接口
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  /* tp_flags */  // 类型标志
    "Dim Object",                   /* tp_doc */  // 类型文档字符串
    0,                              /* tp_traverse */  // 遍历对象函数
    0,                              /* tp_clear */  // 清理对象函数
    0,                              /* tp_richcompare */  // 富比较函数
    0,                              /* tp_weaklistoffset */  // 弱引用列表偏移量
    0,                              /* tp_iter */  // 迭代器
    0,                              /* tp_iternext */  // 迭代器下一个元素
    0,                              /* tp_methods */  // 方法列表
    0,                              /* tp_members */  // 成员列表
    Dim_getsetters,                 /* tp_getset */  // 获取和设置器方法表
    0,                              /* tp_base */  // 基类指针
    0,                              /* tp_dict */  // 类字典
    0,                              /* tp_descr_get */  // 描述器获取函数
    0,                              /* tp_descr_set */  // 描述器设置函数
    0,                              /* tp_dictoffset */  // 字典偏移量
    (initproc)(void*)static_cast<int(*)(mpy::hdl<Dim>,PyObject*,PyObject*)>(Dim_init),  /* tp_init */  // 初始化函数
    0,                              /* tp_alloc */  // 分配函数
    Dim::new_stub,                  /* tp_new */  // 新建对象函数
};

// class DimList ------------

struct DimList : public mpy::base<DimList> {
    mpy::object name_;              // 名称对象
    std::vector<mpy::obj<Dim>> dims_;  // Dim 对象向量
    static PyTypeObject Type;       // 类型对象
    void init(mpy::object name) {   // 初始化函数，设置名称对象
        name_ = std::move(name);
    }
    void set_dims(std::vector<mpy::obj<Dim>> dims) {  // 设置 Dim 向量
        bound_ = true;              // 标记已绑定
        dims_ = std::move(dims);    // 移动赋值到 dims_
    }
    bool is_bound() {               // 是否已绑定
        return bound_;
    }
    void bind_len(int64_t size) {   // 绑定长度函数
        if (bound_) {               // 如果已绑定
            int64_t b_size = dims_.size();  // 获取当前大小
            if (b_size != size) {   // 若与给定大小不符
                mpy::raise_error(DimensionBindError(), "Dimlist has size %lld but it is being bound to size %d", b_size, size);  // 抛出绑定错误异常
            }
        } else {                    // 若未绑定
            bound_ = true;          // 标记已绑定
            dims_.resize(size);     // 调整向量大小为给定大小
            for (Py_ssize_t i = 0; i < size; ++i) {  // 循环初始化 Dim 对象
                dims_[i] = Dim::create(mpy::unicode_from_format("%S%i", name_.ptr(), (int)i));
            }
        }
    }
    # 返回当前维度列表的大小，即维度数量
    int64_t size() const {
        # 如果维度列表未绑定，则触发维度绑定错误并抛出异常
        if (!bound_) {
            mpy::raise_error(DimensionBindError(), "DimList not bound");
        }
        # 返回维度列表的大小
        return dims_.size();
    }
    
    # 设置维度列表的绑定状态
    void set_bound(bool b) {
        # 将绑定状态设置为给定的布尔值
        bound_ = b;
    }
private:
    bool bound_ = false;
};


static int DimList_init(DimList *self, PyObject *args, PyObject *kwds);

// 定义 DimList_repr 方法，用于返回 DimList 对象的字符串表示形式
static PyObject* DimList_repr(DimList* self) {
    PY_BEGIN
    // 如果 DimList 对象已绑定
    if (self->is_bound()) {
        // 获取 DimList 对象中维度列表的大小
        size_t size = self->dims_.size();
        // 创建一个 Python 元组对象 t，大小为 size
        mpy::tuple t(size);
        // 将 self->dims_ 中的每个元素复制到元组 t 中
        for(size_t i = 0; i < size; ++i) {
            t.set(i, self->dims_[i]);
        }
        // 返回元组 t 的字符串表示形式
        return mpy::repr(t).release();
    } else if(!mpy::is_none(self->name_)) {
        // 如果 DimList 对象未绑定且存在 name_ 属性，则返回 "*%S" 格式的字符串
        return mpy::unicode_from_format("*%S", self->name_.ptr()).release();
    } else {
        // 如果 DimList 对象未绑定且 name_ 属性为空，则返回 "<unbound_dimlist>" 字符串
        return mpy::unicode_from_string("<unbound_dimlist>").release();
    }
    PY_END(nullptr)
}

// 定义 DimList_bind 方法，用于绑定 DimList 对象
static PyObject* DimList_bind(DimList *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    // 定义 sizes 变量
    mpy::handle sizes;
    // 定义关键字参数的名称数组
    static const char * const _keywords[] = {"sizes", nullptr};
    // 定义参数解析器
    static _PyArg_Parser parser = {"O", _keywords, 0};
    // 使用 _PyArg_ParseStackAndKeywords 解析参数
    if (!_PyArg_ParseStackAndKeywords(args, nargs, kwnames, &parser, &sizes)) {
        return nullptr;
    }
    // 检查 sizes 是否为序列对象
    if (!mpy::is_sequence(sizes)) {
        mpy::raise_error(PyExc_ValueError, "expected a sequence");
    }
    // 将 sizes 转换为序列视图
    mpy::sequence_view seq = sizes;
    // 获取序列的大小
    auto size = seq.size();
    // 绑定 DimList 对象的长度
    self->bind_len(size);
    // 遍历 sizes 序列，设置 DimList 对象中每个维度的大小
    for (Py_ssize_t i = 0; i < size; ++i) {
        self->dims_[i]->set_size(mpy::to_int(seq[i]));
    }
    // 返回 None
    Py_RETURN_NONE;
    PY_END(nullptr)
}

// 定义 DimList_bind_len 方法，用于设置 DimList 对象的长度
static PyObject* DimList_bind_len(DimList *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    // 定义 size 变量
    int size;
    // 定义关键字参数的名称数组
    static const char * const _keywords[] = {"N", nullptr};
    // 定义参数解析器
    static _PyArg_Parser parser = {"i", _keywords, 0};
    // 使用 _PyArg_ParseStackAndKeywords 解析参数
    if (!_PyArg_ParseStackAndKeywords(args, nargs, kwnames, &parser, &size)) {
        return nullptr;
    }
    // 设置 DimList 对象的长度
    self->bind_len(size);
    // 返回 None
    Py_RETURN_NONE;
    PY_END(nullptr)
}

// 定义 DimList_methods 数组，包含 DimList 对象的方法信息
static PyMethodDef DimList_methods[] = {
    {"bind", (PyCFunction)(void*) DimList_bind, METH_FASTCALL | METH_KEYWORDS},
    {"bind_len", (PyCFunction)(void*) DimList_bind_len, METH_FASTCALL | METH_KEYWORDS},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

// 定义 DimList_len 方法，返回 DimList 对象的大小
static Py_ssize_t DimList_len(DimList* self) {
    PY_BEGIN
    return self->size();
    PY_END(-1)
}

// 定义 DimList_item 方法，返回 DimList 对象中指定索引的元素
static PyObject * DimList_item(DimList* self, Py_ssize_t idx) {
    PY_BEGIN
    // 如果 DimList 对象未绑定，则抛出 DimensionBindError 异常
    if (!self->is_bound()) {
        mpy::raise_error(DimensionBindError(), "DimList not bound");
    }
    // 如果索引 idx 超出范围，则抛出 IndexError 异常
    if (idx < 0 || (size_t) idx >= self->dims_.size()) {
        mpy::raise_error(PyExc_IndexError, "index out of bounds");
    }
    // 获取 DimList 对象中指定索引 idx 处的元素，并释放其所有权
    mpy::object r = self->dims_[idx];
    return r.release();
    PY_END(nullptr)
}

// 定义 DimList_seq 结构体，包含 DimList 对象作为序列的方法
PySequenceMethods DimList_seq {
    (lenfunc) DimList_len, //lenfunc sq_length;
    0, //binaryfunc sq_concat;
    0, //ssizeargfunc sq_repeat;
    (ssizeargfunc) DimList_item, //ssizeargfunc sq_item;
    0, //void *was_sq_slice;
    0, //ssizeobjargproc sq_ass_item;
    0, //void *was_sq_ass_slice;
    0, //objobjproc sq_contains;
    // 定义一个函数指针变量 sq_contains，类型为 objobjproc，初始值为 0

    0, //binaryfunc sq_inplace_concat;
    // 定义一个函数指针变量 sq_inplace_concat，类型为 binaryfunc，初始值为 0

    0, //ssizeargfunc sq_inplace_repeat;
    // 定义一个函数指针变量 sq_inplace_repeat，类型为 ssizeargfunc，初始值为 0
};

// 定义 DimList 对象的 is_bound 属性的 getter 函数
static PyObject* DimList_getis_bound(DimList* self, void*) {
    return PyBool_FromLong(self->is_bound());
}

// 定义 DimList 对象的属性列表，包含 is_bound 属性的 getter 函数
static PyGetSetDef DimList_getsetters[] = {
    {"is_bound", (getter) DimList_getis_bound, NULL, "is_bound", NULL},
    {NULL}  /* Sentinel */
};

// 定义 DimList 对象的 subscript 方法，用于支持下标访问
static PyObject* DimList_subscript(DimList* self, mpy::handle idx) {
    PY_BEGIN
    // 如果 idx 是整数，调用 DimList_item 获取对应元素
    if (mpy::is_int(idx)) {
        return DimList_item(self, mpy::to_int(idx));
    } else if (mpy::is_slice(idx)) { // 如果 idx 是切片对象
        // 如果 DimList 对象未绑定，则抛出 DimensionBindError 异常
        if (!self->is_bound()) {
            mpy::raise_error(DimensionBindError(), "DimList not bound");
        }
        // 根据切片对象创建 slice_view，并初始化结果 tuple
        mpy::slice_view s(idx, self->dims_.size());
        mpy::tuple r(s.slicelength);
        // 遍历切片范围，将对应元素添加到结果 tuple 中
        for (Py_ssize_t i = s.start, j = 0; i < s.stop; i += s.step) {
            r.set(j++,  self->dims_[i]);
        }
        return r.release(); // 返回结果 tuple
    } else {
        // 如果 idx 既不是整数也不是切片，抛出 ValueError 异常
        mpy::raise_error(PyExc_ValueError, "expected an int or a slice");
        return nullptr;
    }
    PY_END(nullptr)
}

// 定义 DimList 对象的 mapping 方法，用于支持映射协议
PyMappingMethods DimList_mapping = {
    0, //lenfunc mp_length;  // 不支持长度方法，设为 0
    (binaryfunc)(void*) DimList_subscript, //binaryfunc mp_subscript; 使用 DimList_subscript 作为下标访问方法
    0, //objobjargproc mp_ass_subscript;  // 不支持赋值下标，设为 0
};

// 定义 DimList 类型对象的结构体
PyTypeObject DimList::Type = {
    PyVarObject_HEAD_INIT(NULL, 0)  // 初始化 PyObject 头部信息
    "_C.DimList",               /* tp_name */  // 类型对象名称
    sizeof(DimList),               /* tp_basicsize */  // 类型对象的基本大小
    0,                              /* tp_itemsize */  // 不需要额外分配内存的大小
    DimList::dealloc_stub,      /* tp_dealloc */  // 对象销毁时调用的函数
    0,                              /* tp_vectorcall_offset */
    0,                              /* tp_getattr */  // 不支持 getattr 方法
    0,                              /* tp_setattr */  // 不支持 setattr 方法
    0,                              /* tp_as_async */  // 不支持异步协议
    (reprfunc)DimList_repr,           /* tp_repr */  // 返回对象的字符串表示方法
    0,                 /* tp_as_number */  // 不支持数值协议
    &DimList_seq,                 /* tp_as_sequence */  // 序列协议的方法
    &DimList_mapping,             /* tp_as_mapping */  // 映射协议的方法
    0,      /* tp_hash */  // 不支持哈希方法
    0,                              /* tp_call */  // 不支持调用协议
    0,                              /* tp_str */  // 不支持字符串表示方法
    0,                              /* tp_getattro */  // 不支持 getattr
    0,                              /* tp_setattro */  // 不支持 setattr
    0,                              /* tp_as_buffer */  // 不支持缓冲协议
    0,                              /* tp_flags */  // 标记位
    "DimList Object",                   /* tp_doc */  // 类型对象的文档字符串
    0,                              /* tp_traverse */  // 不支持遍历
    0,                              /* tp_clear */  // 不支持清理
    0,                              /* tp_richcompare */  // 不支持富比较
    0,                              /* tp_weaklistoffset */  // 弱引用偏移量
    0,                              /* tp_iter */  // 不支持迭代器
    0,                              /* tp_iternext */  // 不支持迭代
    DimList_methods,                /* tp_methods */  // 方法列表
    0,                              /* tp_members */  // 成员变量列表
    DimList_getsetters,             /* tp_getset */  // 属性列表
    0,                              /* tp_base */  // 基类，这里不涉及继承
    0,                              /* tp_dict */  // 字典
    0,                              /* tp_descr_get */  // 获取描述符
    0,                              /* tp_descr_set */  // 设置描述符
    0,                              /* tp_dictoffset */  // 字典偏移量
    (initproc) DimList_init,            /* tp_init */
    // 设置 DimList 对象的初始化函数为 DimList_init，用于对象的初始化过程
    0,                              /* tp_alloc */
    // 不分配额外空间给 DimList 对象，因此为 0
    DimList::new_stub,                      /* tp_new */
    // 设置 DimList 对象的构造函数为 new_stub，用于对象的创建
};

// 初始化 DimList 结构体的实例
static int DimList_init(DimList *self, PyObject *args, PyObject *kwds) {
    PY_BEGIN
    // 定义关键字参数列表
    static constexpr const char* kwlist[] = {"len_or_dims", "name", nullptr};
    // 初始化长度或维度对象为 nullptr
    mpy::handle len_or_dims = nullptr;
    // 初始化名称对象为 nullptr
    PyObject* name = nullptr;
    // 解析传入的参数元组和关键字参数
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &len_or_dims, &name)) {
        return -1;
    }
    // 调用 DimList 实例的初始化方法，传入名称对象（如果存在）
    self->init(mpy::object::borrow(name ? name : Py_None));
    // 如果传入了长度或维度对象
    if (len_or_dims.ptr()) {
        // 如果长度或维度对象是整数
        if(mpy::is_int(len_or_dims)) {
            // 绑定长度到 DimList 实例
            self->bind_len(mpy::to_int(len_or_dims));
        } else if (mpy::is_sequence(len_or_dims)) {  // 如果长度或维度对象是序列
            // 创建序列视图并获取其大小
            mpy::sequence_view s(len_or_dims);
            std::vector<mpy::obj<Dim>> dims;
            size_t size = s.size();
            dims.reserve(size);
            // 遍历序列视图中的元素
            for (size_t i = 0; i < size; ++i) {
                auto r = s[i];
                // 如果元素是整数，创建维度对象并添加到 dims 向量中
                if (mpy::is_int(r)) {
                    dims.emplace_back(Dim::create(mpy::unicode_from_format("%S%i", self->name_.ptr(), (int)i),  mpy::to_int(r)));
                } else {  // 否则，包装对象并添加到 dims 向量中
                    dims.emplace_back(Dim::wrap(r));
                }
            }
            // 设置 DimList 实例的维度
            self->set_dims(std::move(dims));
        } else {  // 若长度或维度对象类型错误，抛出异常
            PyErr_Format(PyExc_ValueError, "expected a length or a sequence of dimensions");
            return -1;
        }
        return 0;
    }
    return 0;
    PY_END(-1);
}

// Tensor -----------------------------

// TensorType 指向 Python 包装类型的指针
PyTypeObject* TensorType = nullptr; // the python wrapper type.

// 运行 Torch 函数的包装器，返回 Torch 张量对象
mpy::object run_torch_function(Arena &A, mpy::handle orig, mpy::vector_args args, bool is_pointwise);

// 匿名命名空间，定义 _add_batch_dims 函数
namespace{

// 添加批次维度到 Torch 张量对象
at::Tensor _add_batch_dims(Arena& A, at::Tensor t, Slice<DimEntry> levels_) {
    // 初始化级别切片对象
    auto levels = Slice<DimEntry>();
    levels.extend(A, levels_);
    // 进入无限循环
    while (true) {
        int64_t min_real_index = -1;
        int64_t min_index = -1;
        int64_t min_value = INT_MAX;
        int64_t i = 0;
        int64_t r = 0;
        // 遍历级别切片中的条目
        for (auto l : levels) {
            if (!l.is_none()) {
                // 如果条目不为空且非位置型，并且级别小于最小值，更新最小值索引
                if (!l.is_positional() && l.dim()->level_ < min_value) {
                    min_value = l.dim()->level_;
                    min_index = i;
                    min_real_index = r;
                }
                ++i;
            }
            ++r;
        }
        // 如果找不到最小索引，返回原始张量对象
        if (min_index == -1) {
            return t;
        }
        // 在指定索引和值处添加批次维度，并更新原始张量对象
        auto t2 = at::functorch::addBatchDim(std::move(t), min_index, min_value);
        t = std::move(t2);
        // 清除处理过的级别切片条目
        levels[min_real_index] = DimEntry();
    }
}



// 延迟操作符结构体
struct DelayedOperator {
    // 构造函数，接收 Python 对象和向量参数
    DelayedOperator(mpy::object o, mpy::vector_args a)
    : orig(std::move(o)), args(a) {
        auto all = a.size();
        // 该操作将超出调用生命周期，因此获取临时对象的所有权
        // 在向量参数中获取临时对象
        auto buf = new mpy::handle[all];
        memcpy(buf, args.args, sizeof(mpy::handle)*all);
        args.args = buf;
        // 逐一增加临时对象的引用计数
        for (auto i : args.enumerate_all()) {
            Py_INCREF(args.args[i].ptr());
        }
        // 增加关键字名称对象的引用计数
        Py_XINCREF(args.kwnames.ptr());
    }
    // 析构函数，用于释放对象实例化时分配的资源
    ~DelayedOperator() {
        // 遍历参数列表中的所有参数
        for (auto i : args.enumerate_all()) {
            // 递减 Python 对象的引用计数
            Py_DECREF(args[i].ptr());
        }
        // 如果参数中包含关键字参数
        if (args.has_keywords()) {
            // 释放关键字参数名列表对象的引用计数
            Py_XDECREF(args.kwnames.ptr());
        }
        // 释放动态分配的参数数组内存
        delete [] args.args;
    }
    // 原始对象
    mpy::object orig;
    // 向量化参数对象
    mpy::vector_args args;
// 结构体用于保存张量相关信息，包括未批量处理的张量、维度条目切片、是否有设备信息和批处理张量的引用
struct TensorInfo {
    // 未批量处理的张量引用
    TensorRef tensor;
    // 维度条目切片
    Slice<DimEntry> levels;
    // 是否有设备信息
    bool has_device;
    // 批处理张量的引用
    TensorRef batchedtensor;

// 释放 levels 的维度条目资源
void free_levels_dims(Slice<DimEntry> levels) {
    // 遍历 levels 中的每个条目
    for(auto e : levels) {
        // 如果条目不是位置参数
        if (!e.is_positional()) {
            // 释放条目的维度资源
            mpy::object::steal(e.dim());
        }
    }
}

// 捕获 levels 的维度条目所有权
void capture_levels(Slice<DimEntry> levels) {
    // 遍历 levels 中的每个条目
    for (auto l : levels) {
        // 如果条目不是位置参数
        if (!l.is_positional()) {
            // 借用条目的维度资源并释放
            mpy::object::borrow(l.dim()).release();
        }
    }
    // 设置 levels_ 的维度条目，并使用 free_levels_dims 函数释放
    levels_.set(levels, free_levels_dims);
}

// 从位置参数创建 Tensor 对象
static mpy::object from_positional(Arena & A, at::Tensor tensor, Slice<DimEntry> levels, bool has_device);

// 创建延迟 Tensor 对象
static mpy::obj<Tensor> create_delayed(mpy::object op, mpy::vector_args args, Slice<DimEntry> levels, bool has_device);

// 检查是否为确切的 Tensor 类型
static bool check_exact(mpy::handle v) {
   return Py_TYPE(v.ptr()) == TensorType;
}

// 创建 Tensor 对象
static mpy::obj<Tensor> create() {
    // 如果 TensorType 未初始化，则导入并赋值
    if (!TensorType) {
        TensorType = (PyTypeObject*) mpy::import("functorch.dim").attr("Tensor").ptr();
    }
    // 分配并返回 Tensor 对象
    return Tensor::alloc(TensorType);
}

// 获取批处理实现的 BatchedTensorImpl，如果不是批处理张量则返回空指针
// 此版本避免了不必要的引用计数操作
at::functorch::BatchedTensorImpl* maybeGetBatchedImpl(const at::Tensor& tensor) {
    // 检查是否为 BatchedTensor
    if (at::functorch::isBatchedTensor(tensor)) {
        // 返回 BatchedTensorImpl 指针
        return static_cast<at::functorch::BatchedTensorImpl*>(tensor.unsafeGetTensorImpl());
    }
    // 非 BatchedTensor 返回空指针
    return nullptr;
}

// 从 Python 对象创建未检查的 TensorRef 引用
TensorRef unchecked_tensor_from(mpy::handle p) {
    // 将 Python 对象转换为 THPVariable
    auto v = (THPVariable*) p.ptr();
    // 返回 TensorRef 引用
    return TensorRef(*v->cdata);
}

// 返回 levels 中的位置参数的维度数量
static int64_t ndim_of_levels(Slice<DimEntry> levels) {
    // 统计位置参数的数量
    int64_t r = 0;
    for (auto l : levels) {
        if (l.is_positional()) {
            ++r;
        }
    }
    return r;
}

};
    // 返回当前张量的维度数量，通过调用ndim_of_levels(levels)来实现
    int64_t ndim() const {
        return ndim_of_levels(levels);
    }
    
    // 将Tensor对象转换为布尔值，若tensor存在则返回true，否则返回false
    operator bool() const {
        return tensor;
    }

    // 静态方法：根据给定的Arena和handle创建TensorInfo对象
    static TensorInfo create(Arena& A, mpy::handle h, bool ensure_batched=true, bool ensure_present=true) {
        // 如果handle是Tensor对象的确切类型
        if (Tensor::check_exact(h)) {
            // 将handle转换为Tensor对象t，并使用其属性创建TensorInfo对象返回
            auto t = Tensor::unchecked_wrap(h);
            return TensorInfo {t->tensor(A), t->levels(), t->has_device(), ensure_batched ? t->batchtensor(A) : TensorRef()};
        } 
        // 如果handle是Dim对象的确切类型
        else if (Dim::check_exact(h)) {
            // 将handle转换为Dim对象d，并使用其属性创建TensorInfo对象返回
            auto d = Dim::unchecked_wrap(h);
            return TensorInfo {d->range(), Slice<DimEntry>(A, DimEntry(d)), false, ensure_batched ? d->batchtensor() : TensorRef()};
        } 
        // 如果handle是THPVariable对象的类型
        else if (THPVariable_Check(h.ptr())) {
            // 从handle中获取未检查的TensorRef对象t
            TensorRef t = unchecked_tensor_from(h);
            // 创建一个Slice<DimEntry>对象levels，用于存储从-t->dim()到0的范围
            Slice<DimEntry> levels;
            for (auto i : irange(-t->dim(), 0)) {
                levels.append(A, i);
            }
            // 使用t和levels创建TensorInfo对象返回
            return TensorInfo {t, levels, true, t};
        } 
        // 如果handle不是上述类型且ensure_present为true，则抛出值错误异常
        else {
            if (ensure_present) {
                mpy::raise_error(PyExc_ValueError, "expected a tensor object");
            }
            // 返回空的TensorInfo对象
            return TensorInfo {};
        }
    }
};

// 从位置参数创建 Tensor 对象的 Python C API 函数
static PyObject* py_Tensor_from_positional(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    // 创建 Arena 对象，用于内存管理
    Arena A;
    // 使用 PY_BEGIN 宏开始 Python C API 的异常处理机制
    PY_BEGIN
    // 定义参数解析的宏，参数类型为 (mpy::handle, tensor), (mpy::handle, py_levels), (int, has_device)
    #define ARGS(_) _(mpy::handle, tensor) _(mpy::handle, py_levels) _(int, has_device)
    // 解析参数并关联到宏定义中的类型
    MPY_PARSE_ARGS_KWNAMES("OOp", ARGS)
    // 取消宏定义 ARGS
    #undef ARGS

    // 检查 tensor 是否为 THPVariable 类型，若不是则抛出 ValueError 异常
    if (!THPVariable_Check(tensor.ptr())) {
        mpy::raise_error(PyExc_ValueError, "_tensor is not a Tensor?");
    }

    // 创建 DimEntry 的 Slice 对象 levels，用于存储维度信息
    Slice<DimEntry> levels;
    // 使用 mpy::sequence_view 迭代 py_levels，将其转换为 DimEntry 类型并添加到 levels 中
    for (auto i : mpy::sequence_view(py_levels).enumerate()) {
        mpy::object v = mpy::sequence_view(py_levels)[i];
        if (mpy::is_int(v)) {
            auto vi = mpy::to_int(v);
            levels.append(A, vi);
        } else {
            auto dim = Dim::wrap(std::move(v));
            mpy::hdl<Dim> hdim = dim;
            levels.append(A, hdim);
        }
    }

    // 调用 Tensor::from_positional 方法创建 Tensor 对象，并释放 Arena A，返回 PyObject*
    return Tensor::from_positional(A, THPVariable_Unpack(tensor.ptr()), levels, has_device != 0).release();
    // 使用 PY_END 结束异常处理块，并返回空指针
    PY_END(nullptr)
}
}

// 根据 Arena 和输入参数创建 Tensor 对象的静态方法
mpy::object Tensor::from_positional(Arena & A, at::Tensor tensor, Slice<DimEntry> levels, bool has_device) {
    // 记录已处理的维度数量
    size_t seen_dims = 0;
    // 记录最后一个处理的维度位置
    int last = 0;
    // 遍历 levels 的每个元素
    for (auto i : levels.enumerate()) {
        auto l = levels[i];
        // 如果 l 是位置参数
        if (l.is_positional()) {
            // 断言最后一个处理的维度是 0 或者是 l 的位置
            AT_ASSERT(last == 0 || last + 1 == l.position());
            last = l.position();
        } else {
            // 如果 l 不是位置参数，释放 l.dim()，并增加已处理维度数量
            mpy::object::borrow(l.dim()).release();
            ++seen_dims;
        }
    }
    // 断言最后一个处理的维度是 0 或者是 -1
    AT_ASSERT(last == 0 || last == -1);
    // 如果没有处理的非位置参数维度，返回封装后的 tensor 对象
    if (!seen_dims) {
        return mpy::object::steal(THPVariable_Wrap(std::move(tensor)));
    }

    // 创建 Tensor 对象 self，并设置相关属性
    mpy::obj<Tensor> self = Tensor::create();
    self->tensor_ = std::move(tensor);
    // 断言 self 的 tensor_ 维度与 levels 的大小相同
    AT_ASSERT(self->tensor_.dim() == levels.size());
    // 设置 self 的 levels_，并使用 free_levels_dims 释放 levels
    self->levels_.set(levels, free_levels_dims);
    self->has_device_ = has_device;
    // 返回 self 对象
    mpy::object r = std::move(self);
    return r;
}

// 根据操作符、参数、维度信息和设备信息创建延迟计算的 Tensor 对象
mpy::obj<Tensor> Tensor::create_delayed(mpy::object op, mpy::vector_args args, Slice<DimEntry> levels, bool has_device) {
    // 创建 Tensor 对象 self
    mpy::obj<Tensor> self = Tensor::create();
    // 捕获 levels 信息
    self->capture_levels(levels);
    self->has_device_ = has_device;
    // 创建延迟操作符对象，并赋值给 self->delayed_
    self->delayed_ = std::make_unique<DelayedOperator>(std::move(op), args);
    // 返回创建的 Tensor 对象 self
    return self;
}

// 将 Slice<mpy::handle> 转换为 mpy::list 对象
mpy::list slice_to_list(Slice<mpy::handle> h) {
    // 创建长度为 h.size() 的 mpy::list 对象 lst
    mpy::list lst(h.size());
    // 遍历 h 中的每个元素，将其 borrow 后放入 lst
    for (auto i : h.enumerate()) {
        lst.set(i, mpy::object::borrow(h[i]));
    }
    // 返回创建的 mpy::list 对象 lst
    return lst;
}

// 将 Slice<mpy::handle> 转换为 mpy::tuple 对象
mpy::tuple slice_to_tuple(Slice<mpy::handle> h) {
    // 创建长度为 h.size() 的 mpy::tuple 对象 lst
    mpy::tuple lst(h.size());
    // 遍历 h 中的每个元素，将其 borrow 后放入 lst
    for (auto i : h.enumerate()) {
        lst.set(i, mpy::object::borrow(h[i]));
    }
    // 返回创建的 mpy::tuple 对象 lst
    return lst;
}

// 定义枚举类型 UType，包含 U_ELEM, U_TUPLE_LIKE, U_DICT
enum UType {
    U_ELEM,
    U_TUPLE_LIKE,
    U_DICT,
};

// 定义结构体 Unflatten
struct Unflatten {
    # 定义函数调用运算符的重载，根据存储的类型执行不同的操作并返回结果对象
    mpy::object operator()(Slice<mpy::handle>& elements) {
        # 声明返回结果对象
        mpy::object r;
        # 根据类型进行不同的操作
        switch (type) {
            case U_ELEM: {
                # 如果类型为单个元素，从参数中借用第一个元素作为结果，并更新剩余元素列表
                r = mpy::object::borrow(elements[0]);
                elements = elements.slice(1);
            } break;
            case U_TUPLE_LIKE: {
                # 如果类型类似元组，创建一个与子元素数量相匹配的元组
                mpy::tuple tup(children.size());
                # 遍历子元素并递归调用它们的操作符重载，将结果设置到元组中
                for (auto i : children.enumerate()) {
                    tup.set(i, children[i](elements));
                }
                # 使用元组调用存储的对象，并将结果赋给 r
                r = obj.call(tup);
            } break;
            case U_DICT: {
                # 如果类型为字典，从存储的对象中获取一个新的 Python 字典对象
                r = mpy::object::checked_steal(PyDict_New());
                # 创建字典视图以便操作
                mpy::dict_view rv(r);
                mpy::dict_view d(obj);
                Py_ssize_t pos = 0;
                mpy::handle k, v;
                # 遍历存储对象的键值对，并递归调用子元素的操作符重载，将结果设置到新字典中
                for (int i = 0; d.next(&pos, &k, &v); ++i) {
                    rv.set(k, children[i](elements));
                }
            } break;
        }
        # 返回最终结果对象
        return r;
    }
    # 存储对象的类型
    UType type;
    # 存储对象的句柄
    mpy::handle obj;
    # 存储子元素的切片
    Slice<Unflatten> children;
};

// 定义函数 Unflatten，用于递归地展开嵌套结构
Unflatten tree_flatten(Arena& A, mpy::handle agg, Slice<mpy::handle>& flat_elements) {
    Slice<Unflatten> c; // 创建 Slice 以存储 Unflatten 结构
    UType utype; // 定义变量 utype，表示展开类型
    mpy::handle obj; // 定义变量 obj，表示当前处理的对象

    // 检查是否为列表视图
    if (mpy::list_view::check(agg)) {
        obj = agg.type(); // 获取对象类型
        utype = U_TUPLE_LIKE; // 设置展开类型为类似元组的结构
        mpy::list_view l(agg); // 创建列表视图对象 l
        // 遍历列表视图 l 中的元素
        for (auto i : l.enumerate()) {
            c.append(A, tree_flatten(A, l[i], flat_elements)); // 递归调用 tree_flatten，将结果追加到 c 中
        }
    } else if (mpy::tuple_view::check(agg)) {
        obj = agg.type(); // 获取对象类型
        utype = U_TUPLE_LIKE; // 设置展开类型为类似元组的结构
        // 包括命名元组的元组视图
        mpy::tuple_view l(agg); // 创建元组视图对象 l
        // 遍历元组视图 l 中的元素
        for (auto i : l.enumerate()) {
            c.append(A, tree_flatten(A, l[i], flat_elements)); // 递归调用 tree_flatten，将结果追加到 c 中
        }
    } else if (mpy::dict_view::check(agg)) {
        utype = U_DICT; // 设置展开类型为字典
        mpy::dict_view d(agg); // 创建字典视图对象 d
        obj = agg; // 设置对象为当前字典视图对象
        Py_ssize_t pos = 0;
        mpy::handle k, v;
        // 遍历字典视图 d 中的键值对
        while (d.next(&pos, &k, &v)) {
            c.append(A, tree_flatten(A, v, flat_elements)); // 递归调用 tree_flatten，将结果追加到 c 中
        }
    } else {
        utype = U_ELEM; // 设置展开类型为单个元素
        flat_elements.append(A, agg); // 将当前对象追加到 flat_elements 中
    }
    // 返回 Unflatten 结构，包含展开类型 utype、对象 obj 和子元素列表 c
    return Unflatten {utype, obj, c};
}

// 定义结构 UnflattenVectorArgs，用于处理向量参数展开
struct UnflattenVectorArgs {
    // 重载运算符 ()，用于处理 Arena、Slice 和 flat_elements，返回 mpy::vector_args 结构
    mpy::vector_args operator()(Arena& A, Slice<mpy::handle>& elements) {
        if (!had_nested) { // 如果没有嵌套结构
            auto args = elements.begin(); // 获取 elements 的起始位置
            elements = Slice<mpy::handle>(); // 清空 elements
            return mpy::vector_args(args, nargs, kwnames); // 返回 mpy::vector_args 结构
        }
        Slice<mpy::handle> args; // 定义 Slice 用于存储参数
        // 遍历 children 中的元素
        for (auto u : children) {
            args.append(A, A.autorelease(u(elements))); // 将处理结果追加到 args 中
        }
        return mpy::vector_args(args.begin(), nargs, kwnames); // 返回 mpy::vector_args 结构
    }
    Slice<Unflatten> children; // 子 Unflatten 结构的 Slice
    Py_ssize_t nargs; // 参数数量
    mpy::handle kwnames; // 关键字参数
    bool had_nested; // 是否存在嵌套结构的标志
};

// 函数 tree_flatten 的重载版本，处理 mpy::vector_args 类型的参数展开
UnflattenVectorArgs tree_flatten(Arena& A, mpy::vector_args args, Slice<mpy::handle>& flat_elements) {
    UnflattenVectorArgs r; // 创建 UnflattenVectorArgs 结构体对象 r
    r.kwnames = args.kwnames; // 复制关键字参数
    r.nargs = args.nargs; // 复制参数数量
    r.had_nested = false; // 设置没有嵌套结构的标志为 false
    auto N = args.size(); // 获取参数的数量 N
    // 遍历参数数组 args
    for(auto i : irange(N)) {
        auto typ = Py_TYPE(args[i].ptr()); // 获取参数的类型
        // 快速检查该对象是否为嵌套结构的成员
        bool is_element = !typ->tp_as_sequence ||  typ == torch_Tensor || typ == TensorType || typ == DimType;
        if (!is_element) {
            flat_elements.extend(A, args.args, args.args + i); // 将 args 中的部分参数扩展到 flat_elements 中
            // 遍历已处理的参数索引范围
            for (auto j : irange(i)) {
                (void)j; // 空语句，用于消除未使用参数的警告
                r.children.append(A, Unflatten {U_ELEM}); // 将 U_ELEM 类型的 Unflatten 结构追加到 children 中
            }
            // 遍历剩余的参数索引范围
            for (auto j : irange(i, N)) {
                // 对剩余参数递归调用 tree_flatten，并将结果追加到 children 中
                r.children.append(A, tree_flatten(A, args[j], flat_elements));
                // 如果当前处理的子结构不是 U_ELEM 类型，则设置 had_nested 标志为 true
                if (r.children.back().type != U_ELEM) {
                    r.had_nested = true; // 存在嵌套结构
                }
            }
            return r; // 返回 UnflattenVectorArgs 结构体对象 r
        }
    }
    flat_elements.extend(A, args.args, args.args + N); // 将所有参数扩展到 flat_elements 中
    return r; // 返回 UnflattenVectorArgs 结构体对象 r
}

// 定义结构 UnflattenArena，包含 Arena 对象和 Unflatten 结构
struct UnflattenArena {
    Arena A; // Arena 对象
    Unflatten unflatten; // Unflatten 结构
};

// Python C API 函数 py_unflatten 的实现
PyObject* py_unflatten(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN // 宏，标志 Python 解释器开始执行
    # 定义宏ARGS，展开为参数列表 (mpy::handle, ns)
    # 使用MPY_PARSE_ARGS_KWNAMES宏解析关键字参数，期望接收一个对象参数
    # 取消宏ARGS的定义
    MPY_PARSE_ARGS_KWNAMES("O", ARGS)
    # 创建一个序列视图sv，用于操作ns
    mpy::sequence_view sv(ns);
    // 由于我们还没有自动释放池...
    // 创建一个Arena对象A，用于内存管理
    Arena A;
    // 创建一个Slice对象slice，存储mpy::handle类型的数据
    Slice<mpy::handle> slice;
    // 将PyObject*指向PyTuple_Type的对象赋给mpy::handle类型的变量Tuple
    mpy::handle Tuple = (PyObject*) &PyTuple_Type;
    // 调用Tuple对象，传入ns作为参数，返回一个输入对象inputs
    auto inputs = Tuple.call(ns);
    // 创建一个tuple_view对象tv，用于操作inputs
    mpy::tuple_view tv(inputs);
    // 遍历tv中的元素，并将其添加到slice中
    for (auto i : tv.enumerate()) {
        slice.append(A, tv[i]);
    }
    // 从self对象中获取名为"arena"的PyCapsule，并将其转换为UnflattenArena指针类型AA
    auto AA = (UnflattenArena*) PyCapsule_GetPointer(self, "arena");
    // 调用AA指向的对象的unflatten方法，传入slice作为参数，返回一个指针r
    auto r = AA->unflatten(slice).release();
    // 断言r不为空指针
    AT_ASSERT(r != nullptr);
    // 返回r作为函数结果
    return r;
    // 结束Python解释器调用，返回空指针
    PY_END(nullptr)
}

// 定义一个名为py_unflatten_def的PyMethodDef结构体，指定函数名称和函数指针
PyMethodDef py_unflatten_def = {"unflatten", (PyCFunction)(void*) py_unflatten, METH_FASTCALL | METH_KEYWORDS};

// 释放UnflattenArena对象的内存
void free_unflatten_arena(PyObject * pc) {
    delete (UnflattenArena*) PyCapsule_GetPointer(pc, "arena");
}

// Python函数：将树形结构展平为元组并返回
PyObject* py_tree_flatten(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    // 定义宏ARGS，解析函数参数并指定类型和变量名
    #define ARGS(_) _(mpy::handle, tree)
    // 解析函数参数，并将其解析为宏ARGS定义的参数
    MPY_PARSE_ARGS_KWNAMES("O", ARGS)
    // 取出UnflattenArena对象并初始化
    auto A = new UnflattenArena;
    // 定义Slice<mpy::handle>类型的elements变量
    Slice<mpy::handle> elements;
    // 调用tree_flatten方法，将树形结构展平，并将结果赋值给A->unflatten
    A->unflatten = tree_flatten(A->A, tree, elements);
    // 创建PyCapsule对象，管理A指针，使用free_unflatten_arena释放资源
    auto cap = mpy::object::checked_steal(PyCapsule_New(A, "arena", free_unflatten_arena));
    // 创建PyCFunction对象，用于调用py_unflatten函数
    auto unflatten = mpy::object::checked_steal(PyCFunction_New(&py_unflatten_def, cap.release()));
    // 创建长度为2的元组对象r
    mpy::tuple r(2);
    // 将elements转换为Python列表，并设置为元组r的第一个元素
    r.set(0, slice_to_list(elements));
    // 将unflatten函数对象设置为元组r的第二个元素
    r.set(1, std::move(unflatten));
    // 返回元组r的资源所有权
    return r.release();
    PY_END(nullptr)
}

// 使用fn函数对agg中的元素进行映射，并返回处理后的结果
mpy::object tree_map(Arena& A, const std::function<mpy::handle(mpy::handle)>& fn, mpy::handle agg) {
    // 定义Slice<mpy::handle>类型的elements变量
    Slice<mpy::handle> elements;
    // 调用tree_flatten方法，将树形结构展平，并将结果赋值给unflatten
    auto unflatten = tree_flatten(A, agg, elements);
    // 遍历elements中的元素，并使用fn函数对每个元素进行映射
    for (auto i : elements.enumerate()) {
        elements[i] = fn(elements[i]);
    }
    // 返回unflatten方法对映射后的elements进行处理的结果
    return unflatten(elements);
}

// 前提条件：h必须是_Tensor的实例
// 返回Tensor对象h的维度数
int64_t _Tensor_ndim(mpy::handle h) {
    if (Tensor::check(h)) {
        // 如果h是Tensor对象，则计算其层级为位置的数量并返回
        int64_t r = 0;
        for (auto l : Tensor::unchecked_wrap(h)->levels()) {
            if (l.is_positional()) {
                ++r;
            }
        }
        return r;
    }
    // 如果h不是Tensor对象，则返回0
    // 这里提到Dim或者DelayedMulTensor
    return 0;
}

// 将TensorRef对象t转换为mpy::handle类型的对象，并返回
mpy::handle handle_from_tensor(Arena& A, TensorRef t) {
    // 如果tensor在Python中仍然存在，则快速返回其对应的Python对象
    std::optional<PyObject*> mb_obj =
        t->unsafeGetTensorImpl()->pyobj_slot()->check_pyobj(getPyInterpreter(), /*ignore_hermetic_tls=*/false);
    if (mb_obj.has_value() && !t->unsafeGetTensorImpl()->pyobj_slot()->owns_pyobj()) {
        return *mb_obj;
    }
    // 否则，将TensorRef对象包装为THPVariable对象，并返回其mpy::handle类型的对象
    return A.autorelease(mpy::object::checked_steal(THPVariable_Wrap(*t)));
}
}

// 构造函数：EnableAllLayers类用于初始化并启用所有层
struct EnableAllLayers {
    EnableAllLayers(Arena& A, Slice<DimEntry> levels) {
        std::vector<std::pair<int64_t, int64_t>> layers;
        layers.reserve(levels.size());
        // 遍历levels中的每个DimEntry对象
        for (auto l : levels) {
            // 如果该DimEntry对象不是位置层，则获取其维度并添加到levels_to_dim_中
            if (!l.is_positional()) {
                auto d = l.dim();
                levels_to_dim_.append(A, d);
            }
        }
        // 根据层级排序levels_to_dim_
        std::sort(levels_to_dim_.begin(), levels_to_dim_.end(), [](mpy::hdl<Dim> lhs, mpy::hdl<Dim> rhs) { return lhs->level_ < rhs->level_;});

        // 遍历排序后的levels_to_dim_
        for (auto i : levels_to_dim_.enumerate()) {
            // 获取当前维度的大小
            auto batch_size = levels_to_dim_[i]->size();
            // 初始化并推送动态层，返回层级level
            auto level = at::functorch::initAndPushDynamicLayer(at::functorch::TransformType::Vmap, batch_size, at::functorch::RandomnessType::Different);
            // 如果是第一个维度，记录其层级level
            if (i == 0) {
                levels_start_ = level;
            }
        }
    }
    // 构造函数，用于启用所有层级
    EnableAllLayers() {
        // 计算要移除的层级的索引
        auto to_remove = levels_start_ + levels_to_dim_.size() - 1;
        // 遍历 levels_to_dim_ 中的每个元素
        for (auto i : levels_to_dim_.enumerate()) {
            // 断言当前动态层的 layerId 符合预期，以确保正确性
            AT_ASSERT(at::functorch::popDynamicLayerAndDeleteMetadata().layerId() == to_remove - i);
        }
    }

    // 从批处理张量创建 mpy::obj<Tensor> 对象
    mpy::obj<Tensor> from_batched(Arena& A, at::Tensor batchedtensor, bool has_device) {
        // 定义一个存储维度信息的 levels 切片
        Slice<DimEntry> levels;
        // 从负数到 0 遍历批处理张量的维度，将每个维度信息添加到 levels 中
        for (auto i : irange(-batchedtensor.dim(), 0)) {
            levels.append(A, i);
        }
        // 定义一个 TensorRef 对象用于存储张量的引用
        TensorRef tensor;
        // 获取批处理张量的 BatchedTensorImpl 实例
        at::functorch::BatchedTensorImpl * impl = maybeGetBatchedImpl(batchedtensor);
        // 迭代访问 BatchedTensorImpl 实例链，直到找到非批处理张量为止
        while(true) {
            auto level = impl->level();
            // 断言当前层级在有效范围内，确保正确性
            AT_ASSERT(level >= levels_start_ && level < levels_start_ + levels_to_dim_.size());
            // 获取层级对应的 Dim 对象，并将其添加到 levels 中
            mpy::hdl<Dim> dim = levels_to_dim_[level - levels_start_].ptr();
            levels.insert(A, impl->bdim(), dim);
            // 获取下一个 BatchedTensorImpl 实例，如果为 nullptr 则结束循环
            at::functorch::BatchedTensorImpl * nimpl = maybeGetBatchedImpl(impl->value());
            if (!nimpl) {
                tensor = impl->value();
                break;
            }
            impl = nimpl;
        }

        // 创建一个新的 mpy::obj<Tensor> 对象 self
        mpy::obj<Tensor> self = Tensor::create();
        // 将张量所有权转移给 self 对象
        self->tensor_ = *tensor;
        // 移动批处理张量的所有权给 self 对象
        self->batchtensor_ = std::move(batchedtensor);
        // 设置是否具有设备信息
        self->has_device_ = has_device;
        // 捕获 levels 信息并保存到 self 对象中
        self->capture_levels(levels);
        // 返回创建的 self 对象
        return self;
    }

    // 原地更新批处理张量的层级信息
    void inplace_update_layers(TensorRef batchtensor, Slice<DimEntry> levels) {
        // XXX - 需要通过 functorch 补丁来设置 set_level
        auto impl = maybeGetBatchedImpl(*batchtensor);
        // 反向遍历 levels_to_dim_ 中的每个元素
        for (auto i : levels_to_dim_.reversed_enumerate()) {
            // 如果 impl 为 nullptr，则退出循环
            if (!impl) {
                break;
            }
            // 如果 levels 包含 levels_to_dim_[i]，则设置 impl 的层级信息
            if (levels.contains(levels_to_dim_[i])) {
                impl->_unsafe_set_level(levels_start_ + i);
                // 获取 impl 的下一个 BatchedTensorImpl 实例
                impl = maybeGetBatchedImpl(impl->value());
            }
        }
    }
private:
    // 起始级别索引，初始化为0
    int64_t levels_start_{};
    // 从级别到维度的切片
    Slice<mpy::hdl<Dim>> levels_to_dim_;
};

namespace {
    // 匹配级别的函数，调整张量的维度
    TensorRef _match_levels(Arena& A, TensorRef v, Slice<DimEntry> from_levels, Slice<DimEntry> to_levels, bool drop_levels=false) {
        // 如果来源级别与目标级别相同，直接返回输入张量
        if (from_levels == to_levels) {
            return v;
        }
        // 获取张量的尺寸和步长
        at::IntArrayRef sz = v->sizes();
        at::IntArrayRef sd = v->strides();
        // 断言：如果不丢弃级别，则来源级别的数量不大于目标级别的数量
        AT_ASSERT(drop_levels || from_levels.size() <= to_levels.size());
        // 新的尺寸和步长的切片
        Slice<int64_t> nsz;
        Slice<int64_t> nsd;
        // 遍历目标级别列表
        for (auto l : to_levels) {
            // 在来源级别中查找当前级别的索引
            auto oidx = from_levels.index(l);
            if (!oidx) {
                // 如果未找到当前级别，根据情况添加尺寸和步长
                nsz.append(A, l.is_positional() ? 1 : l.dim()->size());
                nsd.append(A, 0);
            } else {
                // 如果找到当前级别，使用来源张量对应的尺寸和步长
                auto idx = *oidx;
                nsz.append(A, sz[idx]);
                nsd.append(A, sd[idx]);
            }
        }
        // 根据新的尺寸和步长重新创建张量，并自动释放中间结果
        return A.autorelease(v->as_strided(at::IntArrayRef(nsz.begin(), nsz.end()), at::IntArrayRef(nsd.begin(), nsd.end()), v->storage_offset()));
    }
}

// 运行 Torch 函数的接口，处理张量相关操作
mpy::object run_torch_function(Arena &A, mpy::handle orig, mpy::vector_args args, bool is_pointwise) {
    // 如果不进行逐点优化，强制取消逐点操作
    if (!pointwise_optimize) {
        is_pointwise = false;
    }
    // 输出调试信息，显示当前运行的 Torch 函数类型
    // std::cout << "__torch_function__ " << ((is_pointwise) ? "pointwise" : "functorch") << " " << orig << "\n";

    // 扁平化参数和所有维度
    Slice<mpy::hdl<Dim>> all_dims;
    // 扁平化的参数列表
    Slice<mpy::handle> flat_args;
    // 重新组织参数列表
    auto unflatten_args = tree_flatten(A, args, flat_args);
    // 持有张量的设备
    TensorRef device_holding_tensor;

    // 张量信息和结果级别的切片
    Slice<TensorInfo> infos;
    Slice<DimEntry> result_levels;
    // 遍历扁平化后的参数列表
    for (auto f : flat_args) {
        // 创建张量信息对象，包括是否批量处理和是否具有设备
        infos.append(A, TensorInfo::create(A, f, !is_pointwise, false));
        // 如果创建成功
        if (infos.back()) {
            // 获取当前张量信息对象的引用
            TensorInfo& info = infos.back();
            // 断言：逐点运算或者批处理张量必须是有效的
            AT_ASSERT(is_pointwise || info.batchedtensor);
            // 如果当前设备未确定并且当前张量信息对象包含设备信息
            if (!device_holding_tensor && info.has_device) {
                // 设定当前设备
                device_holding_tensor = infos.back().tensor;
            }
            // 遍历当前张量信息对象中的级别
            for (auto l : info.levels) {
                // 如果结果级别中不包含当前级别，则添加到结果级别中
                if (!result_levels.contains(l)) {
                    result_levels.append(A, l);
                }
            }
        }
    }
    // 如果是逐点操作
    if (is_pointwise) {
        // 遍历扁平化后的参数列表，并获取索引
        for (auto i : flat_args.enumerate()) {
            // 检查参数信息是否存在
            if (infos[i]) {
                // 获取对应的张量引用
                TensorRef tensor = infos[i].tensor;
                // 如果存在持有张量的设备但当前参数没有指定设备，将张量移到指定设备上
                if (device_holding_tensor && !infos[i].has_device) {
                    tensor = A.autorelease(tensor->to(device_holding_tensor->device()));
                }
                // 调用 _match_levels 函数，获取匹配的级别
                auto ml = _match_levels(A, tensor, infos[i].levels, result_levels);
                // 将处理后的张量包装成句柄，更新到 flat_args 中
                flat_args[i] = handle_from_tensor(A, std::move(ml));
            }
        }

        // 将扁平化后的参数列表 flat_args 转换为切片 flat_it
        Slice<mpy::handle> flat_it = flat_args;
        // 将 flat_it 解压为向量参数 uargs
        mpy::vector_args uargs = unflatten_args(A, flat_it);

        // 调用原始函数对象的 call_vector 方法，获取结果对象 result
        mpy::object result = orig.call_vector(uargs);

        // 快速包装，用于一般情况下操作符返回张量的情况
        if (THPVariable_Check(result.ptr())) {
            // 将 THPVariable 类型的结果解包成 Tensor 对象并返回
            return Tensor::from_positional(A, THPVariable_Unpack(result.ptr()), result_levels, device_holding_tensor);
        }

        // 匿名函数 wrap，用于处理非张量类型的结果
        auto wrap = [&](mpy::handle h) {
            if (THPVariable_Check(h.ptr())) {
                // 将 THPVariable 类型的结果解包成 Tensor 对象并返回
                return A.autorelease(Tensor::from_positional(A, THPVariable_Unpack(h.ptr()), result_levels, device_holding_tensor));
            }
            // 直接返回处理后的句柄
            return h;
        };

        // 对结果对象进行 tree_map 处理，并返回最终结果
        return tree_map(A, wrap, result);
    } else {
        // 如果不是逐点操作，打开所有层级的保护
        EnableAllLayers guard(A, result_levels);
        // 遍历扁平化后的参数列表，并获取索引
        for (auto i : flat_args.enumerate()) {
            // 检查参数信息是否存在
            if (infos[i]) {
                // 获取对应的批处理张量引用
                TensorRef batched = infos[i].batchedtensor;
                // 如果存在持有张量的设备但当前参数没有指定设备，将批处理张量移到指定设备上
                if (device_holding_tensor && !infos[i].has_device) {
                    batched = A.autorelease(batched->to(device_holding_tensor->device()));
                }
                // 在保护层中就地更新批处理张量和级别
                guard.inplace_update_layers(batched, infos[i].levels);
                // 将处理后的批处理张量包装成句柄，更新到 flat_args 中
                flat_args[i] = handle_from_tensor(A, batched);
            }
        }

        // 将扁平化后的参数列表 flat_args 转换为切片 flat_it
        Slice<mpy::handle> flat_it = flat_args;
        // 将 flat_it 解压为向量参数 uargs
        mpy::vector_args uargs = unflatten_args(A, flat_it);
        // 断言扁平化后的参数列表 flat_it 应该为空
        AT_ASSERT(flat_it.size() == 0);

        // 调用原始函数对象的 call_vector 方法，获取结果对象 result
        mpy::object result = orig.call_vector(uargs);

        // 匿名函数 wrap，用于处理非张量类型的结果
        auto wrap = [&](mpy::handle h) {
            if (THPVariable_Check(h.ptr())) {
                // 将 THPVariable 类型的结果解包成 Tensor 对象并返回
                return A.autorelease(guard.from_batched(A, THPVariable_Unpack(h.ptr()), device_holding_tensor));
            }
            // 直接返回处理后的句柄
            return h;
        };

        // 如果结果对象是 THPVariable 类型，将其解包成 Tensor 对象并返回
        if (THPVariable_Check(result.ptr())) {
            return guard.from_batched(A, THPVariable_Unpack(result.ptr()), device_holding_tensor);
        }

        // 对结果对象进行 tree_map 处理，并返回最终结果
        return tree_map(A, wrap, result);
    }
}

namespace {

// 实现 Torch 函数的自定义处理
mpy::object __torch_function__(Arena &A, mpy::handle orig, mpy::vector_args args, bool is_pointwise) {
    // 检查原始函数是否为 torch.Tensor.__mul__
    if (orig == torch_Tensor___mul__) {
        // 断言参数个数为2且无关键字参数
        AT_ASSERT(args.nargs == 2 && !args.has_keywords());
        // 获取左右操作数
        auto lhs = args[0];
        auto rhs = args[1];
        // 如果左右操作数均为 _Tensor 类型且均为零维张量
        if (mpy::isinstance(lhs, _Tensor) && mpy::isinstance(rhs, _Tensor) && _Tensor_ndim(lhs) == 0 && _Tensor_ndim(rhs) == 0) {
            bool has_device = false;
            Slice<DimEntry> levels;
            // 遍历所有位置参数
            for (auto i : args.enumerate_positional()) {
                // 创建 TensorInfo 对象
                auto t = TensorInfo::create(A, args[i], false);
                // 如果操作数不是浮点数类型，返回标准的 Torch 函数处理结果
                if (!t.tensor->is_floating_point()) {
                    return run_torch_function(A, orig, args, is_pointwise);
                }
                // 更新是否具有设备信息
                has_device = has_device || t.has_device;
                // 将 TensorInfo 中的维度信息添加到 levels 中
                for (auto l : t.levels) {
                    if (!levels.contains(l)) {
                        levels.append(A, l);
                    }
                }
            }
            // 打印调试信息（已注释掉的语句）
            // std::cout << "__torch_function__ " << "delay" << " " << orig << "\n";
            // 创建并返回延迟创建的 Tensor 对象
            return Tensor::create_delayed(mpy::object::borrow(orig), args, levels, has_device);
        }
    }
    // 默认情况下运行 Torch 函数
    return run_torch_function(A, orig, args, is_pointwise);
}

// 将 Python 元组和关键字参数转换为 vector_args 结构
mpy::vector_args as_vector_args(Arena &A, mpy::handle args, mpy::handle kwargs) {
    // 解析位置参数
    auto pos_args = (mpy::handle*)&PyTuple_GET_ITEM(args.ptr(), 0);
    auto pos_n = PyTuple_GET_SIZE(args.ptr());
    // 如果不存在关键字参数，直接返回位置参数的 vector_args 结构
    if (!kwargs.ptr()) {
        return mpy::vector_args(pos_args, pos_n, nullptr);
    }
    Slice<mpy::handle> all_args;
    Slice<mpy::handle> kwnames;
    // 将所有位置参数加入 all_args 中
    all_args.extend(A, pos_args, pos_args + pos_n);
    // 遍历关键字参数，加入 all_args 和 kwnames
    mpy::dict_view dv(kwargs);
    Py_ssize_t pos = 0;
    mpy::handle key, value;
    while (dv.next(&pos, &key, &value)) {
        all_args.append(A, value);
        kwnames.append(A, key);
    }
    // 返回构造好的 vector_args 结构
    return mpy::vector_args(all_args.begin(), pos_n, A.autorelease(slice_to_tuple(kwnames)));
}

// Python 的 __torch_function__ 函数的封装器
PyObject* py___torch_function__(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    // 创建 Arena 对象
    Arena A;
    PY_BEGIN
    // 可能初始化全局变量
    maybeInitializeGlobals();
    // 断言参数个数为4或5
    AT_ASSERT(nargs == 4 || nargs == 5);
    // 将 Python 参数转换为 vector_args 结构
    auto va = as_vector_args(A, args[3], nargs == 5 ? args[4] : nullptr);
    // 判断是否为 pointwise 操作
    bool is_pointwise = pointwise.contains(args[1]);
    // 调用 __torch_function__ 函数并释放其返回值
    return __torch_function__(A, args[1], std::move(va), is_pointwise).release();
    PY_END(nullptr)
}

// 将 Slice<DimEntry> 结构转换为 Python 元组
mpy::object levels_to_tuple(Slice<DimEntry> slice) {
    // 创建 Python 元组对象
    mpy::tuple t(slice.size());
    // 将 Slice 中的数据转换为 Python 对象并放入元组中
    for (auto i : slice.enumerate()) {
        t.set(i, slice[i].is_positional() ? mpy::from_int(slice[i].position()) : mpy::object::borrow(slice[i].dim()));
    }
    // 移动构造并返回元组对象
    mpy::object r = std::move(t);
    return r;
}

// Tensor 的 ndim 属性获取函数
PyObject* Tensor_ndim(Tensor* self, void*) {
    // 初始化维度计数器
    Py_ssize_t i = 0;
    // 遍历self对象的levels()方法返回的所有元素，使用auto关键字推断类型
    for (auto l : self->levels()) {
        // 检查当前元素l是否是位置相关的
        if (l.is_positional()) {
            // 如果l是位置相关的，则增加i的值
            ++i;
        }
    }
    // 将整数i转换为mpy::from_int对象，并释放其所有权，返回结果
    return mpy::from_int(i).release();
}

PyGetSetDef Tensor_getsetters[] = {
   {"_has_device", (getter) [](PyObject* self, void*) -> PyObject* { return mpy::from_bool(((Tensor*)self)->has_device()).release(); }, NULL},
   {"_tensor", (getter) [](PyObject* self, void*) -> PyObject* {
       Arena A;
       return THPVariable_Wrap(((Tensor*)self)->tensor(A)); }, NULL},
   {"_batchtensor", (getter) [](PyObject* self, void*) -> PyObject* {
       Arena A;
       return THPVariable_Wrap(((Tensor*)self)->batchtensor(A)); }, NULL},
   {"_levels", (getter) [](PyObject* self, void*) -> PyObject* {
       PY_BEGIN
       return levels_to_tuple(((Tensor*)self)->levels()).release();
       PY_END(nullptr)
   }},
    {"ndim", (getter) Tensor_ndim, NULL, "ndim", NULL},
    {NULL}  /* Sentinel */
};

// PyGetSetDef 结构体数组，用于定义 Python 对象 Tensor 的属性

PyMethodDef Tensor_methods[] = {
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

// PyMethodDef 结构体数组，用于定义 Python 对象 Tensor 的方法

}

// 这里是一个代码块的结尾，但不太清楚这段代码应该属于哪个函数或类的定义

PyTypeObject Tensor::Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_C.Tensor",               /* tp_name */
    sizeof(Tensor),               /* tp_basicsize */
    0,                              /* tp_itemsize */
    Tensor::dealloc_stub,      /* tp_dealloc */
    0,                              /* tp_vectorcall_offset */
    0,                              /* tp_getattr */
    0,                              /* tp_setattr */
    0,                              /* tp_as_async */
    0,           /* tp_repr */
    0,                 /* tp_as_number */
    0,                 /* tp_as_sequence */
    0,             /* tp_as_mapping */
    0,      /* tp_hash */
    0,                              /* tp_call */
    0,                              /* tp_str */
    0,                              /* tp_getattro */
    0,                              /* tp_setattro */
    0,                              /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE , /* tp_flags */
    "Tensor Object",                   /* tp_doc */
    0,                              /* tp_traverse */
    0,                              /* tp_clear */
    0,  /* tp_richcompare */
    0,                              /* tp_weaklistoffset */
    0,                              /* tp_iter */
    0,                              /* tp_iternext */
    Tensor_methods,                /* tp_methods */
    0,                              /* tp_members */
    Tensor_getsetters,             /* tp_getset */
    0,                              /* tp_base */
    0,                              /* tp_dict */
    0,                              /* tp_descr_get */
    0,                              /* tp_descr_set */
    0,                              /* tp_dictoffset */
    0,            /* tp_init */
    0,                              /* tp_alloc */
    Tensor::new_stub,                      /* tp_new */
};

// PyTypeObject 结构体，定义了 Python 对象类型 Tensor 的具体属性和行为

// dim() --------------------

static bool relevant_op(_Py_CODEUNIT c) {


注释：
    # 根据输入的变量 c 的不同值进行不同的操作
    switch(c) {
        # 如果 c 的值是 STORE_NAME，STORE_GLOBAL，STORE_FAST，或 STORE_DEREF，则返回 true
        case STORE_NAME:
        case STORE_GLOBAL:
        case STORE_FAST:
        case STORE_DEREF:
            return true;
        # 如果 c 的值不是以上列出的任何一个，返回 false
        default:
            return false;
    }
}

// 创建一个维度对象的静态方法，接受名称和大小作为参数
static mpy::object create_dim(mpy::object name, mpy::handle size) {
    // 使用名称创建一个 Dim 对象
    auto d = Dim::create(std::move(name));
    // 如果大小参数不为空
    if (!mpy::is_none(size)) {
        // 将大小设置为整数形式
        d->set_size(mpy::to_int(size));
    }
    // 返回创建的维度对象
    return std::move(d);
}

// 创建一个维度列表对象的静态方法，接受名称和大小作为参数
static mpy::object create_dimlist(mpy::object name, mpy::handle size) {
    // 使用名称创建一个 DimList 对象
    auto d = DimList::create(std::move(name));
    // 如果大小参数不为空
    if (!mpy::is_none(size)) {
        // 如果大小是整数类型，则绑定列表长度
        if (mpy::is_int(size)) {
            d->bind_len(mpy::to_int(size));
        } else {
            // 如果大小是序列，则绑定列表长度并设置每个维度的大小
            mpy::sequence_view s(size);
            d->bind_len(s.size());
            for (auto i : irange(d->size())) {
                d->dims_[i]->set_size(mpy::to_int(s[i]));
            }
        }
    }
    // 返回创建的维度列表对象
    return std::move(d);
}

// Python包装器，为旧版本运行时提供新的反射原语
#if !(IS_PYTHON_3_11_PLUS)
#define _PyCode_CODE(CO) ((_Py_CODEUNIT*)PyBytes_AS_STRING((CO)->co_code))
#endif

// 匿名命名空间，用于实现私有结构或函数的封装
namespace {
    // Python 指令解码器结构体，用于解析 Python 字节码指令
    struct PyInstDecoder {
        // 构造函数，初始化 Python 字节码对象和偏移量
        PyInstDecoder(PyCodeObject* code_object, int lasti)
        : code_object_(code_object), code_(_PyCode_CODE(code_object)), offset_(lasti / sizeof(_Py_CODEUNIT))  {}

        // 移动到下一个指令位置
        void next() {
        #if IS_PYTHON_3_11_PLUS
            offset_ += _PyOpcode_Caches[opcode()];
        #endif
            offset_ += 1;
        }

        // 获取当前指令的操作码
        int opcode() {
            auto r = _Py_OPCODE(code_[offset_]);
        #if IS_PYTHON_3_11_PLUS
            r = _PyOpcode_Deopt[r];
        #endif
            return r;
        }

        // 获取当前指令的操作数
        int oparg() {
            return _Py_OPARG(code_[offset_]);
        }

        // 获取当前指令的名称对象
        mpy::object name() {
            mpy::object names;
            switch(opcode()) {
                // 根据不同操作码选择不同的名称集合
                case STORE_NAME:
                case STORE_GLOBAL:
                    names = mpy::object::borrow(code_object_->co_names);
                    break;
                case STORE_FAST:
                    names = mpy::object::steal(PyCode_GetVarnames(code_object_));
                    break;
                case STORE_DEREF:
                    names = mpy::object::steal(PyCode_GetCellvars(code_object_));
                    break;
                default:
                    return mpy::object();
            }
            // 返回操作数对应的名称对象
            return mpy::object::steal(PySequence_GetItem(names.ptr(), oparg()));
        }
    private:
        PyCodeObject* code_object_;   // Python 字节码对象
        _Py_CODEUNIT* code_;          // Python 字节码数组
        int offset_;                  // 当前解析位置的偏移量
    };

    // 模板函数，用于创建对象，接受名称和大小作为参数
    template<mpy::object (*create_object)(mpy::object, mpy::handle)>
    static PyObject* _dims(PyObject *self,
                          PyObject *const *args,
                          Py_ssize_t nargs,
                          PyObject *kwnames) {
        PY_BEGIN
        Py_ssize_t specified_ndims = -1;  // 指定的维度数初始化为 -1
        Py_ssize_t found_ndims = 0;       // 已找到的维度数初始化为 0
        Py_ssize_t sizes = -1;            // 大小初始化为 -1
        mpy::handle n = Py_None;          // Python None 对象的句柄
        mpy::handle py_sizes = Py_None;   // Python None 对象的句柄
    // 如果有位置参数或关键字参数存在
    if (nargs || kwnames) {
        // 使用 vector_args 类处理参数，其中包括位置参数和关键字参数
        mpy::vector_args va(args, nargs, kwnames);
        // 解析参数 "dims"，期望得到 "n" 和 "sizes" 参数的值，如果没有则使用默认值 0
        va.parse("dims", {"n", "sizes"}, {&n, &py_sizes}, 0);
        // 如果 py_sizes 不为 None，则获取其大小作为 sizes，并设置指定的维度数为 sizes
        if (!mpy::is_none(py_sizes)) {
            sizes = mpy::sequence_view(py_sizes).size();
            specified_ndims = sizes;
        }
        // 如果 n 不为 None，则将其转换为整数并设置指定的维度数为 n 的值
        if (!mpy::is_none(n)) {
            specified_ndims = mpy::to_int(n);
        }
    }

    // 获取当前线程状态
    PyThreadState* state = PyThreadState_GET();
    // 获取当前帧对象
    auto f = mpy::obj<PyFrameObject>::steal(PyThreadState_GetFrame(state));
    // 获取当前帧的代码对象
    auto c = mpy::obj<PyCodeObject>::steal(PyFrame_GetCode(f.ptr()));
    // 获取最后执行的指令索引
    auto lasti = PyFrame_GetLasti(f.ptr());
    // 创建指令解码器对象，解码当前帧的代码对象和最后执行的指令索引
    auto decoder = PyInstDecoder(c.ptr(), lasti);

    // 如果是 Python 3.11 及以上版本
    #if IS_PYTHON_3_11_PLUS
    // 当 Python 3.11 适应字节码时，lasti 指向预调用，而不是调用指令之后
    if (decoder.opcode() == PRECALL) {
        decoder.next();
    }
    #endif
    // 跳到下一个指令

    // 如果当前指令是与维度相关的操作
    if (relevant_op(decoder.opcode())) {
        found_ndims = 1;
    } else if (decoder.opcode() == UNPACK_SEQUENCE) {
        // 如果当前指令是 UNPACK_SEQUENCE，则设置找到的维度数为其操作数
        found_ndims = decoder.oparg();
        decoder.next();
    }

    // 如果指定的维度数为 -1
    if (specified_ndims == -1) {
        // 如果没有找到任何维度，抛出语法错误
        if (found_ndims == 0) {
            mpy::raise_error(PyExc_SyntaxError, "dims() must be assigned to a sequence of variable names or have argument n specified");
        }
        // 否则将指定的维度数设置为找到的维度数
        specified_ndims = found_ndims;
    }
    // 如果找到的维度数与指定的维度数不一致，则将 found_ndims 设为 0，避免选择错误的维度名
    if (found_ndims != specified_ndims) {
        found_ndims = 0; // 避免选择错误的维度名
    }

    // 创建生成对象的 lambda 函数，根据索引 i 创建一个对象
    auto genobject = [&](int i) -> mpy::object {
        mpy::object name;
        // 如果 i 小于找到的维度数，则获取解码器当前指令的名称作为维度名
        if (i < found_ndims) {
            name = decoder.name();
        }
        // 如果未找到名称，则创建一个默认格式的名称 "d%d"，并将 found_ndims 设为 0
        if (!name.ptr()) {
            name = mpy::unicode_from_format("d%d", i);
            found_ndims = 0; // 一旦找不到名称，就可以找到任何名称
        } else {
            decoder.next();
        }
        // 创建对象并返回
        return create_object(std::move(name), sizes != -1 ? mpy::sequence_view(py_sizes)[i] : mpy::handle(Py_None));
    };

    // 如果 sizes 不为 -1 且不等于指定的维度数，则抛出值错误异常
    if (sizes != -1 && sizes != specified_ndims) {
        mpy::raise_error(PyExc_ValueError, "expected %d sizes but found %d", int(specified_ndims), int(sizes));
    }

    // 如果指定的维度数为 1，则返回生成的第一个对象
    if (specified_ndims == 1) {
        return genobject(0).release();
    }

    // 否则创建一个指定维度数的元组对象，并用生成的对象填充
    mpy::tuple result(specified_ndims);
    for (int i = 0; i < specified_ndims; ++i) {
        result.set(i, genobject(i));
    }
    // 返回创建的结果元组对象
    return result.release();
    PY_END(nullptr)
}

// 结构体定义，用于表示点积的一部分
struct DotPart {
    // 维度条目的切片
    Slice<DimEntry> dims;
    // 总大小，默认为1
    size_t total_size = 1;
    
    // 向DotPart对象中追加维度条目
    void append(Arena& A, mpy::hdl<Dim> d) {
        // 计算总大小
        total_size *= d->size();
        // 向dims中添加维度条目
        dims.append(A, d);
    }
};

// 模板函数，将Slice<T>转换为at::ArrayRef<T>
template<typename T>
static at::ArrayRef<T> as_array_ref(Slice<T> t) {
    return at::ArrayRef<T>(t.begin(), t.end());
}

// 静态函数，为点积操作准备数据
static TensorRef dot_prepare(Arena& A, std::initializer_list<DotPart> parts, const TensorInfo& t) {
    // 新的级别切片
    Slice<DimEntry> new_levels;
    // 是否需要重塑
    bool needs_reshape = false;
    
    // 遍历所有的DotPart对象
    for (auto p : parts) {
        // 如果DotPart对象中维度条目数量不为1，则需要重塑
        if (p.dims.size() != 1) {
            needs_reshape = true;
        }
        // 向new_levels中扩展维度条目
        new_levels.extend(A, p.dims);
    }
    
    // 调用_match_levels函数，根据需要执行级别匹配
    auto r = _match_levels(A, t.tensor, t.levels, new_levels, true);
    // 如果不需要重塑，则直接返回r
    if (!needs_reshape) {
        return r;
    }
    
    // 视图切片
    Slice<int64_t> view;
    // 遍历所有的DotPart对象，向view中添加总大小
    for (auto p : parts) {
        view.append(A, p.total_size);
    }
    // 执行重塑操作，并返回结果
    return A.autorelease(r->reshape(at::IntArrayRef(view.begin(), view.end())));
}

// 静态函数，完成点积计算并返回mpy::object对象
static mpy::object dot_finish(Arena& A, std::initializer_list<DotPart> parts, at::Tensor r) {
    // 结果级别切片
    Slice<DimEntry> result_levels;
    // 是否需要重塑
    bool needs_reshape = false;
    
    // 遍历所有的DotPart对象
    for (auto p : parts) {
        // 如果DotPart对象中维度条目数量不为1，则需要重塑
        if (p.dims.size() != 1) {
            needs_reshape = true;
        }
        // 向result_levels中扩展维度条目
        result_levels.extend(A, p.dims);
    }
    
    // 如果需要重塑，则执行重塑操作
    if (needs_reshape) {
        // 新的大小切片
        Slice<int64_t> new_size;
        // 遍历所有的result_levels，向new_size中添加维度大小
        for (auto l : result_levels) {
            new_size.append(A, l.dim()->size());
        }
        // 执行重塑操作，并更新r
        r = r.reshape(at::IntArrayRef(new_size.begin(), new_size.end()));
    }
    
    // 使用Tensor::from_positional函数创建并返回mpy::object对象
    return Tensor::from_positional(A, std::move(r), result_levels, true);
}

// 静态函数，执行点积运算
static mpy::object dot(Arena& A, TensorInfo lhs, TensorInfo rhs, Slice<DimEntry> sum) {
    // 左右操作数的步长
    auto lhs_strides = lhs.tensor->strides();
    auto rhs_strides = rhs.tensor->strides();

    // 四个DotPart对象，分别代表不同的维度情况
    DotPart lro_dims;
    DotPart lo_dims;
    DotPart ro_dims;
    DotPart lr_dims;

    // 插入维度的Lambda函数
    auto insert_dim = [&] (mpy::hdl<Dim> d, at::optional<int> lhs_idx, at::optional<int> rhs_idx) {
        // 判断维度是否被减少
        bool reduced = sum.contains(d);
        // 获取左右操作数的步长
        int64_t lhs_stride = lhs_idx ? lhs_strides[*lhs_idx] : 0;
        int64_t rhs_stride = rhs_idx ? rhs_strides[*rhs_idx] : 0;
        
        // 如果维度被减少
        if (reduced) {
            // 将维度条目追加到lr_dims中
            lr_dims.append(A, d);
        } else {
            // 判断是否为lro情况
            if ((lhs_stride == 0) == (rhs_stride == 0)) {
                lro_dims.append(A, d);
            } else if (lhs_stride != 0) {
                // lo情况
                lo_dims.append(A, d);
            } else {
                // ro情况
                AT_ASSERT(rhs_stride != 0);
                ro_dims.append(A, d);
            }
        }
    };

    // 右操作数的标记数组，初始化为false
    auto rhs_seen = A.allocate<bool>(rhs.levels.size());
    std::fill(rhs_seen, rhs_seen + rhs.levels.size(), false);

    // 遍历左操作数的级别切片
    for (auto i : lhs.levels.enumerate()) {
        // 获取级别d
        auto d = lhs.levels[i];
        // 查找级别d在右操作数中的索引
        auto rhs_idx = rhs.levels.index(d);
        // 如果找到了，则将rhs_seen中对应位置设置为true
        if (rhs_idx) {
            rhs_seen[*rhs_idx] = true;
        }
        // 插入维度d
        insert_dim(d.dim(), i, rhs_idx);
    }
    // 遍历 rhs.levels 中的每个元素 i，并对其进行枚举
    for (auto i : rhs.levels.enumerate()) {
        // 如果 rhs_seen 中已经包含了当前元素 i，则跳过当前循环
        if (rhs_seen[i]) {
            continue;
        }
        // 从 rhs.levels 中获取元素 i 的值，并将其插入到当前对象中
        auto d = rhs.levels[i];
        insert_dim(d.dim(), at::nullopt, i);
    }

    // 如果 lr_dims.dims 的大小与 sum 的大小不相等，则进行以下操作
    if (lr_dims.dims.size() != sum.size()) {
        // 遍历 sum 中的每个元素 d
        for (auto & d : sum) {
            // 如果 lhs.levels 不包含 d 并且 rhs.levels 也不包含 d，则抛出 DimensionBindError 异常
            if (!lhs.levels.contains(d) && !rhs.levels.contains(d)) {
                mpy::raise_error(DimensionBindError(), "summing over non-existant dimension %S", d.dim().ptr());
            }
        }
    }

    // 输出 lhs.levels、rhs.levels 和 sum 的值到标准输出流
    // std::cout << lhs.levels << " " << rhs.levels << " " << sum << "\n";
    
    // 输出 lro_dims.dims、lo_dims.dims、ro_dims.dims 和 lr_dims.dims 的值到标准输出流
    // std::cout << lro_dims.dims << " " << lo_dims.dims << " " << ro_dims.dims << " " << lr_dims.dims << "\n";

    // 如果 lro_dims.dims 的大小不为 0，则执行以下操作；否则执行另一组操作
    if (lro_dims.dims.size() != 0) {
        // 使用 dot_prepare 函数准备 lhs 和 rhs，然后调用 dot_finish 函数完成操作并返回结果
        auto lhs_ = dot_prepare(A, {lro_dims, lo_dims, lr_dims}, lhs);
        auto rhs_ = dot_prepare(A, {lro_dims, lr_dims, ro_dims}, rhs);
        return dot_finish(A, {lro_dims, lo_dims, ro_dims}, at::bmm(*lhs_, *rhs_));
    } else {
        // 使用 dot_prepare 函数准备 lhs 和 rhs，然后调用 dot_finish 函数完成操作并返回结果
        auto lhs_ = dot_prepare(A, {lo_dims, lr_dims}, lhs);
        auto rhs_ = dot_prepare(A, {lr_dims, ro_dims}, rhs);
        return dot_finish(A, {lo_dims, ro_dims}, at::mm(*lhs_, *rhs_));
    }
}

static PyObject* test_c(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN

    // 创建一个 Arena 对象，用于分配内存
    Arena A;
    
    // 创建一个 Slice<int> 对象 s，包含初始元素 3, 4, 5
    Slice<int> s(A, 3, 4, 5);
    
    // 断言 s 的大小为 3，容量为 8
    AT_ASSERT(s.size() == 3 && s.capacity() == 8);
    
    // 断言 s 的元素符合预期值
    AT_ASSERT(s[0] == 3 && s[1] == 4 && s[2] == 5);
    
    // 在 s 中添加一个新元素 6
    s.append(A, 6);
    
    // 断言 s 的第四个元素为 6
    AT_ASSERT(s[3] == 6);
    
    // 使用循环向 s 中依次添加元素 0 到 9
    for(int i : irange(10)) {
        s.append(A, i);
    }
    
    // 断言 s 的部分状态：首元素为 3，末尾元素为 9，大小为 14，容量为 16
    AT_ASSERT(s[0] == 3 && s.back() == 9 && s.size() == 14 && s.capacity() == 16);

    // 创建另一个 Slice<int> 对象 s2，包含初始元素 -1, -2, -3
    Slice<int> s2(A, -1, -2, -3);
    
    // 断言 s2 的第二个元素为 -2，同时验证 s 的首元素为 3
    AT_ASSERT(s2[1] == -2 && s[0] == 3);

    // 创建一个切片对象 ss，从 s 中切出索引为 1 到 2 的部分
    auto ss = s.slice(1,2);
    
    // 断言 ss 的大小为 1，元素为 4，容量为 1
    AT_ASSERT(ss.size() == 1);
    AT_ASSERT(ss[0] == 4);
    AT_ASSERT(ss.capacity() == 1);
    
    // 在 ss 中添加一个新元素 -4
    ss.append(A, -4);
    
    // 断言 ss 的大小为 2，第二个元素为 -4
    AT_ASSERT(ss.size() == 2 && ss[1] == -4);
    
    // 修改 ss 的第一个元素为 3
    ss[0] = 3;
    
    // 断言 s 的第二个元素为 4
    AT_ASSERT(s[1] == 4);

    // 在 s 中索引为 1 到 4 的区间中插入 ss 的内容
    s.insert(A, s.slice(1, 4), ss);
    
    // 断言 s 的第二个元素为 3，第三个元素为 -4，第四个元素为 0
    AT_ASSERT(s[1] == 3  && s[2] == -4 && s[3] == 0);

    // 获取 s 的当前大小
    auto sz = s.size();
    
    // 在 s 中索引为 1 到 1 的区间中插入整数 4
    s.insert(A, s.slice(1, 1), 4);
    
    // 断言 s 的第二个元素为 4，且大小比之前增加了 1
    AT_ASSERT(s[1] == 4 && sz + 1 == s.size());

    // 创建一个 Slice<int> 对象 d，包含初始元素 0 到 4
    Slice<int> d(A, 0, 1, 2, 3, 4);

    // 创建一个 Slice<int> 对象 b，包含初始元素 0 到 4
    Slice<int> b(A, 0, 1, 2, 3, 4);
    
    // 在 b 中索引为 1 到 1 的区间中插入 d 的内容
    b.insert(A, b.slice(1,1), d);
    
    // 断言 b 的大小为 10，第二个元素为 0，第六个元素为 4，最后一个元素为 4
    AT_ASSERT(b.size() == 10);
    AT_ASSERT(b[1] == 0);
    AT_ASSERT(b[5] == 4);
    AT_ASSERT(b.back() == 4);

    // 返回 Python 中的 None 对象
    Py_RETURN_NONE;

    // 结束异常处理块
    PY_END(nullptr);
}


static PyObject* order(PyObject *_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    // 创建一个 Arena 对象，用于分配内存
    Arena A;
    
    // 开始异常处理块
    PY_BEGIN
    
    // 如果传入了关键字参数，则引发 TypeError 异常
    if (kwnames) {
        mpy::raise_error(PyExc_TypeError, "unexpected keyword arguments %S", kwnames);
    }
    
    // 断言参数个数大于 0
    AT_ASSERT(nargs-- > 0);
    
    // 创建两个 Slice<DimEntry> 对象 orig_levels 和 levels
    Slice<DimEntry> orig_levels;
    Slice<DimEntry> levels;
    
    // 创建一个 TensorRef 对象 data
    TensorRef data;
    
    // 获取 Python 对象 self，并进行类型检查
    mpy::handle self = args++[0];
    bool has_device;
    if (Tensor::check_exact(self)) {
        auto t = Tensor::unchecked_wrap(self);
        orig_levels = t->levels();
        data = t->tensor(A);
        has_device = t->has_device();
    } else {
       auto d = Dim::unchecked_wrap(self);
        orig_levels.append(A, d);
        data = d->range();
        has_device = false;
    }

    // 创建两个 Slice<DimEntry> 对象 flat_positional_dims 和 to_flatten
    Slice<DimEntry> flat_positional_dims;
    Slice<std::pair<int, int>> to_flatten;
    
    // 将 orig_levels 的内容扩展到 levels 中
    levels.extend(A, orig_levels);

    // 获取 levels 的原始维度数
    int orig_ndim = ndim_of_levels(levels);
    
    // 定义一个 lambda 函数 append，用于向 flat_positional_dims 添加新元素
    auto append = [&](DimEntry d) {
        auto midx = levels.index(d);
        if (!midx) {
            if (d.is_positional()) {
                // 如果 d 是位置维度，则引发 ValueError 异常
                mpy::raise_error(PyExc_ValueError, "tensor has %d positional dimensions, but %d specified, or it was specified twice", int(orig_ndim), int(d.position() + orig_ndim));
            } else {
                // 如果 d 不在 levels 中，则引发 ValueError 异常
                mpy::raise_error(PyExc_ValueError, "tensor of dimensions %R does not contain dim %R or it was specified twice", levels_to_tuple(orig_levels).ptr(), d.dim().ptr());
            }
        }
        // 将 levels 中的对应位置置空，并向 flat_positional_dims 添加 d
        levels[*midx] = DimEntry();
        flat_positional_dims.append(A, d);
    };

    // 初始化新位置维度数为 0
    int n_new_positional = 0;
    // 遍历参数列表 nargs
    for (auto i : irange(nargs)) {
        // 获取第 i 个参数并存储在 arg 中
        mpy::handle arg = args[i];
        // 使用 _wrap_dim 函数将参数包装成 DimEntry 对象，使用原始维度 orig_ndim，不强制创建新对象
        DimEntry entry = _wrap_dim(arg, orig_ndim, false);
        // 如果 entry 不是 None
        if (!entry.is_none()) {
            // 将 entry 添加到当前对象的尾部
            append(entry);
            // 增加新位置参数计数
            ++n_new_positional;
        } else if (DimList::check(arg)) {
            // 如果参数是 DimList 类型
            auto dl = DimList::unchecked_wrap(arg);
            // 遍历 DimList 中的每个 Dim 对象
            for (mpy::obj<Dim> & d : dl->dims_) {
                // 将 Dim 对象添加到当前对象的尾部
                append(mpy::hdl<Dim>(d));
                // 增加新位置参数计数
                ++n_new_positional;
            }
        } else {
            // 如果参数既不是 DimEntry 也不是 DimList，则增加新位置参数计数
            ++n_new_positional;
            // 如果参数不是序列类型，抛出错误
            if (!mpy::is_sequence(arg)) {
                mpy::raise_error(PyExc_ValueError, "expected a Dim, List[Dim], or Sequence[Dim]");
            }
            // 将参数视为序列并获取其视图
            mpy::sequence_view sq(arg);
            // 获取序列的大小
            auto N = sq.size();
            // 将需要展开的参数和其在 flat_positional_dims 中的位置对存储起来
            to_flatten.append(A, std::make_pair(flat_positional_dims.size(), N));
            // 遍历序列中的每个元素
            for (auto j : irange(N)) {
                // 使用 _wrap_dim 函数将序列中的每个元素包装成 DimEntry 对象
                DimEntry e = _wrap_dim(A.autorelease(sq[j]), orig_ndim, false);
                // 如果包装结果为 None，抛出错误
                if (e.is_none()) {
                    mpy::raise_error(PyExc_ValueError, "expected a Dim, or int");
                }
                // 将包装后的 DimEntry 对象添加到当前对象的尾部
                append(e);
            }
        }
    }

    // 初始化变量 ndim 为 0，用于计数非 None 的 l 对象
    int ndim = 0;
    // 初始化变量 insert_point 为 -1，用于标记新级别的插入点
    int insert_point = -1;
    // 初始化 Slice<DimEntry> 类型的新级别列表 new_levels
    Slice<DimEntry> new_levels;
    // 遍历 levels 中的每个 l 对象
    for (auto l : levels) {
        // 如果 l 是 None，则继续下一个循环
        if (l.is_none()) {
            continue;
        }
        // 如果 l 是位置参数
        if (l.is_positional()) {
            // 增加 ndim 计数
            ndim++;
            // 如果 insert_point 仍为 -1，则设置为 new_levels 的当前大小，用 flat_positional_dims 扩展 new_levels
            if (insert_point == -1) {
                insert_point = new_levels.size();
                new_levels.extend(A, flat_positional_dims);
            }
        }
        // 将 l 添加到 new_levels 的尾部
        new_levels.append(A, l);
    }
    // 如果 insert_point 仍为 -1，则设置为 new_levels 的当前大小，用 flat_positional_dims 扩展 new_levels
    if (insert_point == -1) {
        insert_point = new_levels.size();
        new_levels.extend(A, flat_positional_dims);
    }

    // 使用 _match_levels 函数匹配原始级别 orig_levels 和新级别 new_levels，返回匹配后的数据 Tensor
    at::Tensor ndata = *_match_levels(A, data, orig_levels, new_levels);
    // 如果需要展开的维度数量不为零
    if (to_flatten.size()) {
        // 定义一个切片视图
        Slice<int64_t> view;
        // 获取原始数据的维度大小
        auto sz = ndata.sizes();
        
        // 在插入点之前的原始维度
        for (auto i : irange(0, insert_point)) {
            view.append(A, sz[i]);
        }
        
        int i = 0;
        // 遍历需要展开的维度
        for (auto to_flat : to_flatten) {
            // 在插入点之后的原始维度
            for (; i < to_flat.first; ++i) {
                view.append(A, sz[insert_point + i]);
            }
            int64_t new_size = 1;
            int last = i + to_flat.second;
            // 计算新的维度大小
            for (; i < last; ++i) {
                new_size *= sz[insert_point + i];
            }
            view.append(A, new_size);
        }
        
        // 添加剩余的原始维度
        for (; i < flat_positional_dims.size(); ++i) {
            view.append(A, sz[insert_point + i]);
        }
        
        // 在新的位置维度之后的原始维度
        for (auto i : irange(insert_point + flat_positional_dims.size(), levels.size())) {
            view.append(A, sz[i]);
        }
        
        // 缩减维度数量，从新的层级中移除这些维度
        // 之后将重新编号它们
        auto n_to_remove = flat_positional_dims.size() - n_new_positional;
        new_levels.insert(A, new_levels.slice(insert_point, insert_point + n_to_remove), Slice<DimEntry>());
        
        // 重新调整数据维度
        ndata = std::move(ndata).reshape(at::IntArrayRef(view.begin(), view.end()));
    }

    // 重新编号位置维度
    int seen = 0;
    for (auto i : new_levels.reversed_enumerate()) {
        if (new_levels[i].is_positional() || (i >= insert_point && i < insert_point + n_new_positional)) {
            new_levels[i] = --seen;
        }
    }
    
    // 返回重新排列后的张量
    return Tensor::from_positional(A, std::move(ndata), new_levels, has_device).release();

    // 结束 Python 解释器的作用域
    PY_END(nullptr)
static PyObject* expand(PyObject *_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    // 创建 Arena 对象，用于内存管理
    Arena A;
    // 开始 Python C API 的错误处理块
    PY_BEGIN
    // 确保参数个数大于0
    AT_ASSERT(nargs-- > 0);
    // 创建 TensorInfo 对象，用于处理张量信息
    auto info = TensorInfo::create(A, args++[0], false);
    // 遍历参数列表（除第一个参数之外）
    for (auto i : irange(nargs)) {
        // 检查参数是否为有效的维度对象
        if (!Dim::check(args[i])) {
            // 可能初始化全局变量
            maybeInitializeGlobals();
            // 创建向量参数对象
            mpy::vector_args vargs(args - 1, nargs + 1, kwnames);
            // 如果最后一个参数是 THPVariable 类型，则调用对应的扩展函数并返回结果
            if (THPVariable_Check(args[-1])) {
                return torch_Tensor_expand.call_vector(vargs).release();
            } else {
                // 否则调用 __torch_function__ 函数处理扩展操作并返回结果
                return __torch_function__(A, torch_Tensor_expand, vargs, false).release();
            }
        }
    }
    // 获取数据张量的引用
    const at::Tensor& data = *info.tensor;
    // 获取张量的级别信息
    auto levels = info.levels;
    // 创建新级别的切片对象
    Slice<DimEntry> new_levels;
    // 创建新尺寸和步幅的切片对象
    Slice<int64_t> sz;
    Slice<int64_t> sd;
    // 遍历参数列表
    for (auto i : irange(nargs)) {
        // 包装未检查的维度对象
        auto d = Dim::unchecked_wrap(args[i]);
        // 检查维度是否已经存在于当前级别或新级别中
        if (levels.contains(d) || new_levels.contains(d)) {
            // 抛出维度绑定错误异常
            mpy::raise_error(DimensionBindError(), "expanding dimension %R already exists in tensor with dims", d.ptr());
        }
        // 添加维度到新级别中
        new_levels.append(A, d);
        // 添加维度大小到 sz 切片中
        sz.append(A, d->size());
        // 添加步幅到 sd 切片中
        sd.append(A, 0);
    }
    // 将新级别扩展到现有级别中
    new_levels.extend(A, levels);
    // 获取数据张量的原始尺寸和步幅
    at::IntArrayRef osz = data.sizes();
    at::IntArrayRef osd = data.strides();
    // 将原始尺寸添加到 sz 切片中
    sz.extend(A, osz.begin(), osz.end());
    // 将原始步幅添加到 sd 切片中
    sd.extend(A, osd.begin(), osd.end());
    // 根据新尺寸和步幅创建张量的视图
    at::Tensor ndata = data.as_strided(at::IntArrayRef(sz.begin(), sz.end()), at::IntArrayRef(sd.begin(), sd.end()), data.storage_offset());
    // 根据位置参数创建张量对象，并返回结果
    return Tensor::from_positional(A, std::move(ndata), new_levels, info.has_device).release();
    // 结束 Python C API 的错误处理块，返回空指针
    PY_END(nullptr)
}


static void _bind_dims_to_size(Arena & A, int64_t sz, int64_t sd,
                        Slice<mpy::hdl<Dim>> dims, Slice<int64_t>& nsz, Slice<int64_t>& nsd) {
    // 右侧乘积初始化为1
    int64_t rhs_prod = 1;
    // 枚举维度切片对象中的每个维度
    for (auto i : dims.enumerate()) {
        // 如果维度未绑定
        if (!dims[i]->is_bound()) {
            // 遍历当前维度之后的每个维度
            for (auto j : irange(i + 1, dims.size())) {
                // 如果当前维度和后续维度都未绑定，抛出维度绑定错误异常
                if (!dims[j]->is_bound()) {
                    mpy::raise_error(DimensionBindError(), "cannot infer the sizes of two dimensions at once %R and %R", dims[i].ptr(), dims[j].ptr());
                }
                // 计算右侧乘积
                rhs_prod *= dims[j]->size();
            }
            // 如果 sz 不能整除右侧乘积，抛出维度绑定错误异常
            if (sz % rhs_prod != 0) {
                // 创建维度元组对象
                mpy::tuple tup(dims.size());
                // 遍历维度切片对象中的每个维度
                for (auto j : dims.enumerate()) {
                    // 将维度的尺寸添加到元组中，如果维度已绑定则添加占位符 "?"
                    tup.set(j, dims[j]->is_bound() ? mpy::from_int(dims[j]->size()) : mpy::unicode_from_string("?"));
                }
                // 抛出维度绑定错误异常
                mpy::raise_error(DimensionBindError(), "inferred dimension does not evenly fit into larger dimension: %d vs %R", (int) sz, tup.ptr());
            }
            // 计算推断的维度大小并设置维度对象的大小
            int64_t inferred_size = sz / rhs_prod;
            dims[i]->set_size(inferred_size);
            // 更新右侧乘积为当前维度的大小
            rhs_prod = sz;
            // 跳出循环
            break;
        }
        // 更新右侧乘积为当前维度的大小
        rhs_prod *= dims[i]->size();
    }
}
    // 检查右手边的维度大小是否等于sz
    if (rhs_prod != sz) {
        // 创建一个包含dims.size()个元素的元组对象
        mpy::tuple tup(dims.size());
        // 遍历dims，并将每个维度封装为borrowed对象放入tup中
        for (auto j : dims.enumerate()) {
            tup.set(j, mpy::object::borrow(dims[j]));
        }
        // 抛出DimensionBindError异常，提供错误信息和维度信息tup
        mpy::raise_error(DimensionBindError(), "Dimension sizes to do not match (%d != %d) when matching dimension pack %R", (int) sz, (int) rhs_prod, tup.ptr());
    }
    // 为新数组的步幅分配内存空间，类型为int64_t，大小为dims.size()
    auto new_strides = A.allocate<int64_t>(dims.size());
    // 初始化前一个步幅为sd
    auto prev_stride = sd;
    // 从dims的末尾向前遍历
    for (auto i : dims.reversed_enumerate()) {
        // 设置当前维度的新步幅为prev_stride
        new_strides[i] = prev_stride;
        // 更新prev_stride为当前维度大小乘以前一个步幅
        prev_stride = dims[i]->size() * prev_stride;
    }
    // 遍历dims的每个维度
    for (auto i : dims.enumerate()) {
        // 将new_strides[i]添加到nsd中
        nsd.append(A, new_strides[i]);
        // 将dims[i]->size()添加到nsz中
        nsz.append(A, dims[i]->size());
    }
}

static bool has_dims(mpy::handle d) {
    // 检查给定对象是否是 Dim 类型或 Tensor 类型的确切实例
    return Dim::check_exact(d) || Tensor::check_exact(d);
}

struct IndexingInfo {
    bool can_call_original; // 如果为 true，则可以安全地直接调用 getitem 或 setitem，这些对象不需要特殊处理
    bool advanced_indexing; // 需要实际查找
    TensorRef self; // 指向张量的引用
    Slice<mpy::handle> flat_inputs; // 平坦输入的切片
    Slice<DimEntry> result_levels; // 结果级别的切片
    bool has_device; // 是否有设备
};
}

// 获取和设置平坦索引的信息
IndexingInfo getsetitem_flat(Arena& A, TensorInfo self_info, Slice<mpy::handle> input, Slice<DimEntry> keys, Slice<mpy::handle> values, bool has_dimpacks_or_none);

namespace {
// 将 Python 元组视图转换为 mpy::handle 的切片
Slice<mpy::handle> as_slice(mpy::tuple_view tv) {
    PyObject** begin = &PyTuple_GET_ITEM(tv.ptr(),0);
    return Slice<mpy::handle>((mpy::handle*)begin, (mpy::handle*)(begin + tv.size()));
}

// 将 Python 列表视图转换为 mpy::handle 的切片
Slice<mpy::handle> as_slice(mpy::list_view tv) {
    PyObject** begin = &PyList_GET_ITEM(tv.ptr(),0);
    return Slice<mpy::handle>((mpy::handle*)begin, (mpy::handle*)(begin + tv.size()));
}


bool maybe_dimpack(Slice<mpy::handle>& elements, mpy::handle s, bool check_first=true) {
    // 是否可以避免重新检查？
    if (mpy::list_view::check(s)) {
        mpy::list_view tv(s);
        // 如果不需要首次检查或者第一个元素是确切的 Dim 类型，则转换为元素切片
        if (!check_first || (tv.size() && Dim::check_exact(tv[0]))) {
            elements = as_slice(tv);
            return true;
        }
    }
    // 是否可以避免重新检查？
    if (mpy::tuple_view::check(s)) {
        mpy::tuple_view tv(s);
        // 如果不需要首次检查或者第一个元素是确切的 Dim 类型，则转换为元素切片
        if (!check_first || (tv.size() && Dim::check_exact(tv[0]))) {
            elements = as_slice(tv);
            return true;
        }
    }
    return false;
};

// 检查给定对象是否是 DimPack
bool is_dimpack(mpy::handle s) {
    Slice<mpy::handle> e;
    return maybe_dimpack(e, s);
}

// 调用 getitem 函数
mpy::object invoke_getitem(Arena& A, const IndexingInfo& iinfo) {
    at::Tensor rtensor;
    if (iinfo.advanced_indexing) {
        auto self_hdl = handle_from_tensor(A, iinfo.self);
        auto tup = slice_to_tuple(iinfo.flat_inputs);
        // std::cout << "calling original getindex " << self_hdl << " " << tup << "\n";
        auto pytensor = mpy::object::checked_steal(THPVariable_getitem(self_hdl.ptr(), tup.ptr()));
        rtensor = THPVariable_Unpack(pytensor.ptr());
    } else {
        // std::cout << "skipping original getindex\n";
        rtensor = *iinfo.self;
    }
    // std::cout << "returning (from_positional)\n";
    return Tensor::from_positional(A, std::move(rtensor), iinfo.result_levels, iinfo.has_device);
}

// 对象索引操作
mpy::object index(Arena& A, mpy::handle self, mpy::handle dims, mpy::handle indices) {
    maybeInitializeGlobals();
    Slice<mpy::handle> dims_list;
    Slice<mpy::handle> indices_list;
    // 允许将单个维度匹配到多个维度，需要先将所有内容标准化为列表在左手边和右手边的情况
    bool lhs_list = mpy::tuple_view::check(dims) || mpy::list_view::check(dims);
    bool rhs_list = mpy::tuple_view::check(indices) || mpy::list_view::check(indices);
    // 检查 lhs_list 和 rhs_list 是否都存在
    if (lhs_list && rhs_list) {
        // 创建 dimensions 和 indices 的视图对象
        mpy::sequence_view dv(dims);
        mpy::sequence_view ind(indices);
        // 获取 dimensions 和 indices 的大小
        Py_ssize_t N = dv.size();
        // 如果 dimensions 和 indices 的大小不相等，则抛出类型错误异常
        if (N != ind.size()) {
            mpy::raise_error(PyExc_TypeError, "dims (%d) and indices (%d) must have the same length", int(N), int(ind.size()));
        }
        // 遍历 dimensions 和 indices，分别添加到 dims_list 和 indices_list
        for (auto i : irange(N)) {
            dims_list.append(A, A.autorelease(dv[i]));
            indices_list.append(A, A.autorelease(ind[i]));
        }
    } else {
        // 如果 lhs_list 或 rhs_list 不存在，则直接将 dims 和 indices 添加到 dims_list 和 indices_list
        dims_list.append(A, dims);
        indices_list.append(A, indices);
    }

    // dims 正在索引的维度可以被组合成一个单一的索引空间，
    // 我们必须在可以对它们进行索引之前将它们展平为单个维度…
    // 创建 TensorInfo 对象以获取有关 self 张量的信息
    auto self_info = TensorInfo::create(A, self, false);
    // 获取张量的维度数
    auto ndim = self_info.ndim();
    // 初始化新的级别列表和要展平的维度列表
    Slice<DimEntry> new_levels;
    Slice<DimEntry> to_flatten;
    Slice<DimEntry> dims_list_flat;
    
    // 定义解析维度条目的 lambda 函数
    auto parse_dim_entry = [&](mpy::handle s) -> DimEntry {
        // 将输入转换为维度条目对象
        auto d = _wrap_dim(s, ndim, false);
        // 如果转换失败，则抛出类型错误异常
        if (d.is_none()) {
            mpy::raise_error(PyExc_TypeError, "expected a dimension specifyer but found %R", s.ptr());
        }
        return d;
    };
    
    // 定义维度不存在时引发异常的 lambda 函数
    auto dim_not_present = [&](DimEntry d) {
        // 如果维度是位置参数类型，则引发相关维度不在张量中的异常
        if (d.is_positional()) {
            mpy::raise_error(PyExc_TypeError, "dimension %d not in tensor of %d dimensions", d.position() + ndim , ndim);
        } else {
            // 否则，引发维度不存在的异常
            mpy::raise_error(PyExc_TypeError, "dimension %R not in tensor", d.dim()->ptr());
        }
    };

    // 遍历 dims_list 中的维度条目
    for (auto i : dims_list.enumerate()) {
        // 定义 m 作为处理维度打包的临时变量
        Slice<mpy::handle> m;
        // 如果可能将 dims_list[i] 解析为维度打包，则执行以下逻辑
        if (maybe_dimpack(m, dims_list[i], /*check_first=*/false)) {
            // 如果打包后的维度数量为0，则直接添加一个空的 DimEntry
            if (m.size() == 0) {
                dims_list_flat.append(A, DimEntry()); // 值只是被丢弃了
            }
            // 解析第一个维度条目，并将其添加到 dims_list_flat 中
            auto first = parse_dim_entry(m[0]);
            dims_list_flat.append(A, first);
            // 如果只有一个维度条目，则继续下一个循环
            if (m.size() == 1) {
                continue;
            }
            // 如果 to_flatten 列表为空，则扩展 new_levels 到 self_info.levels 的内容
            if (to_flatten.size() == 0) {
                new_levels.extend(A, self_info.levels);
            }
            // 处理剩余的维度条目，并将其添加到 rest 中
            Slice<DimEntry> rest;
            for (auto i : irange(1, m.size())) {
                auto d = parse_dim_entry(m[i]);
                // 如果 new_levels 中不存在该维度条目，则引发维度不存在异常
                if (!new_levels.remove(A, d)) {
                    dim_not_present(d);
                }
                rest.append(A, d);
            }

            // 查找第一个维度条目在 new_levels 中的索引位置
            auto first_idx = new_levels.index(first);
            // 如果未找到，则引发维度不存在异常
            if (!first_idx) {
                dim_not_present(first);
            }
            // 在 new_levels 中插入新的维度条目
            new_levels.insert(A, new_levels.slice(*first_idx + 1, *first_idx + 1), rest);
            // 将 rest 添加到 to_flatten 中
            to_flatten.extend(A, rest);
        } else {
            // 否则，直接将 dims_list[i] 解析为维度条目并添加到 dims_list_flat 中
            dims_list_flat.append(A, parse_dim_entry(dims_list[i]));
        }
    }
    # 检查要展开的张量是否包含元素
    if (to_flatten.size() > 0) {
        # 对输入张量进行重新排序以匹配新的级别顺序，并返回重新排序后的张量引用
        TensorRef rearranged = _match_levels(A, self_info.tensor, self_info.levels, new_levels);
        # 获取重新排序后张量的尺寸
        at::IntArrayRef sizes = rearranged->sizes();
        # 定义用于重新整形的新尺寸切片
        Slice<int64_t> new_sizes;
        # 定义用于重新整形的级别切片
        Slice<DimEntry> reshape_levels;
        # 遍历新级别列表及其索引
        for (auto i : new_levels.enumerate()) {
            # 如果要展开的级别包含在要展开的张量中
            if (to_flatten.contains(new_levels[i])) {
                # 将新尺寸的最后一个维度乘以当前维度的大小
                new_sizes.back() *= sizes[i];
            } else {
                # 否则将当前维度大小添加到新尺寸中
                new_sizes.append(A, sizes[i]);
                # 并将当前级别添加到重新整形的级别中
                reshape_levels.append(A, new_levels[i]);
            }
        }
        # 对重新排序后的张量进行重新整形，使用新尺寸
        self_info.tensor = A.autorelease(rearranged->reshape(at::IntArrayRef(new_sizes.begin(), new_sizes.end())));

        # 更新对象的级别信息为重新整形后的级别信息
        self_info.levels = reshape_levels;
        // 注意: 我们使用展平组中的第一个级别来代表操作的其余部分
        // 我们需要小心，不要依赖维度大小，因为它不匹配整个组的大小
    }
    # 检查索引列表中是否存在维度包
    bool has_dimpacks = false;
    for (auto idx : indices_list) {
        # 如果索引是元组视图或列表视图，则设置维度包标志为真
        if (mpy::tuple_view::check(idx) || mpy::list_view::check(idx)) {
            has_dimpacks = true;
            break;
        }
    }
    # 调用 getsetitem_flat 函数，获取扁平化处理后的索引信息
    IndexingInfo info = getsetitem_flat(A, self_info, Slice<mpy::handle>(), dims_list_flat, indices_list, has_dimpacks);
    # 调用 invoke_getitem 函数，根据获取的信息执行getitem操作，并返回结果
    return invoke_getitem(A, info);
// 结束当前代码块

Slice<mpy::handle> slice_from_sequence(Arena& A, mpy::handle value) {
    // 如果 value 是元组视图，返回作为切片的元组视图
    if (mpy::tuple_view::check(value)) {
        return as_slice(mpy::tuple_view(value));
    } else if (mpy::list_view::check(value)) {
        // 如果 value 是列表视图，返回作为切片的列表视图
        return as_slice(mpy::list_view(value));
    } else {
        // 否则将 value 视为序列视图
        mpy::sequence_view sv(value);
        Slice<mpy::handle> r;
        // 遍历序列视图中的元素并添加到结果切片中
        for (auto i : sv.enumerate()) {
            r.append(A, A.autorelease(sv[i]));
        }
        return r;
    }
}

bool extractIndices(Arena& A, mpy::handle index, Slice<mpy::handle>& indices) {
    // 如果 index 是元组视图，将其作为切片扩展到 indices 中
    if (mpy::tuple_view::check(index)) {
        indices.extend(A, as_slice(mpy::tuple_view(index)));
        return true;
    } else if (THPVariable_Check(index.ptr())) {
        // 如果 index 是 THPVariable 类型，将其作为单独的索引添加到 indices 中
        indices.append(A, index);
        return false;
    } else if (!mpy::is_sequence(index)) {
        // 如果 index 不是序列，将其作为单独的索引添加到 indices 中
        indices.append(A, index);
        return false;
    }
    // 将 index 视为序列视图
    // 复制 treatSequenceAsTuple 的修改版本以添加 Dim 和包装后的张量
    mpy::sequence_view sv(index);
    // 如果序列视图的大小大于等于 32，将其作为切片扩展到 indices 中
    if (sv.size() >= 32) {
        indices.extend(A, slice_from_sequence(A, index));
        return true;
    }
    // 遍历序列视图中的元素
    for (auto i : sv.enumerate()) {
        mpy::handle item;
        try {
            item = sv[i];
        } catch (mpy::exception_set & e) {
            PyErr_Clear();
            // 如果出现异常，将 index 作为单独的索引添加到 indices 中
            indices.append(A, index);
            return false;
        }
        // 检查元素是否是 THPVariable 类型、序列、切片、省略号、None 或具有维度信息
        if (THPVariable_Check(item.ptr()) || mpy::is_sequence(item) || PySlice_Check(item.ptr()) || item.ptr() == Py_Ellipsis || mpy::is_none(item) || has_dims(item)) {
            // 如果满足条件，将 item 视为切片扩展到 indices 中
            indices.extend(A, slice_from_sequence(A, index));
            return true;
        }
    }
    // 如果未满足条件，将 index 作为单独的索引添加到 indices 中
    indices.append(A, index);
    return false;
}

IndexingInfo getsetitem(Arena & A, mpy::handle self, mpy::handle index, bool tensors_have_dims) {
    // 是否可以调用原始的 getitem 取决于 tensors_have_dims 变量
    bool can_call_original_getitem = !tensors_have_dims;

    // 输入索引的切片
    Slice<mpy::handle> input;
    // 如果索引具有维度信息，则将其作为切片添加到 input 中
    if (has_dims(index)) {
        input.append(A, index);
    } else {
        // 否则尝试从索引中提取切片，is_sequence 表示是否提取成功
        bool is_sequence = extractIndices(A, index, input);
        // 如果可以调用原始的 getitem 并且索引不是序列，则返回可调用原始 getitem 的信息
        if (can_call_original_getitem && !is_sequence) {
            return { true };
        }
    }

    int64_t dims_indexed = 0;
    int64_t expanding_object = -1;
    DimList* unbound_dim_list = nullptr;
    auto check_expanding = [&](int64_t i) {
        // 检查是否已经存在扩展的对象，如果是，抛出维度绑定错误
        if (expanding_object != -1) {
            mpy::raise_error(DimensionBindError(), "at most one ... or unbound dimension list can exist in indexing list but found 2 at offsets %d and %d", (int) expanding_object, (int) i);
        }
        expanding_object = i;
    };
    Slice<int64_t> dimlists;

    // 计算已经索引的维度数量，以便计算 ... 的大小或扩展可能的未绑定维度列表。

    // 是否具有维度包或 None
    bool has_dimpacks_or_none = false;
    // 遍历输入的枚举元素，i 是索引
    for (auto i : input.enumerate()) {
        // 获取输入中的元素 s
        mpy::handle s = input[i];
        // 检查 s 是否为 Dim 或 Tensor 对象的确切类型
        if (Dim::check_exact(s) || Tensor::check_exact(s)) {
            // 如果是，则禁止调用原始的 getitem 方法，并增加 dims_indexed 计数
            can_call_original_getitem = false;
            ++dims_indexed;
        } else if (s.ptr() == Py_Ellipsis) {
            // 如果 s 是 Py_Ellipsis，调用 check_expanding 函数
            check_expanding(i);
        } else if (DimList::check(s)) {
            // 如果 s 是 DimList 类型
            // 禁止调用原始的 getitem 方法，并解包 DimList
            can_call_original_getitem = false;
            auto dl = DimList::unchecked_wrap(s);
            if (!dl->is_bound()) {
                // 如果 DimList 没有绑定，则调用 check_expanding 函数，并设置 unbound_dim_list
                check_expanding(i);
                unbound_dim_list = dl.ptr();
            } else {
                // 否则，增加 dims_indexed 计数
                dims_indexed += dl->dims_.size();
            }
            // 将索引 i 添加到 dimlists 中
            dimlists.append(A, i);
        } else if (mpy::is_none(s)) {
            // 如果 s 是 None 类型，设置 has_dimpacks_or_none 为 true
            has_dimpacks_or_none = true;
        } else if (is_dimpack(s)) {
            // 如果 s 是 dimpack 类型
            // 禁止调用原始的 getitem 方法，并增加 dims_indexed 计数，设置 has_dimpacks_or_none 为 true
            can_call_original_getitem = false;
            has_dimpacks_or_none = true;
            ++dims_indexed;
        } else {
            // 其他情况，增加 dims_indexed 计数
            ++dims_indexed;
        }
    }

    // 如果 can_call_original_getitem 为 true，返回一个包含 true 的集合
    if (can_call_original_getitem) {
        return {true};
    }

    // 创建 TensorInfo 对象 self_info，表示当前 self 的信息
    TensorInfo self_info = TensorInfo::create(A, self, false, true);
    // 获取 self_info 的维度数 ndim
    auto ndim = self_info.ndim();
    // 如果 dims_indexed 大于 ndim，抛出 ValueError 异常
    if (dims_indexed > ndim) {
        mpy::raise_error(PyExc_ValueError, "at least %d indices were supplied but the tensor only has %d dimensions", (int) dims_indexed, (int) ndim);
    }

    // 计算需要扩展的维度数 expanding_dims
    auto expanding_dims = ndim - dims_indexed;
    // 如果 expanding_object 不为 -1
    if (expanding_object != -1) {
        // 如果存在 unbound_dim_list，则绑定其长度为 expanding_dims
        if (unbound_dim_list) {
            unbound_dim_list->bind_len(expanding_dims);
        } else {
            // 否则，创建扩展维度的 Slice<mpy::handle> 对象 no_slices
            Slice<mpy::handle> no_slices;
            for (auto i : irange(expanding_dims)) {
                (void) i;
                no_slices.append(A, no_slice);
            }
            // 在 input 中的 expanding_object 处插入扩展的 slices
            input.insert(A, input.slice(expanding_object, expanding_object + 1), no_slices);
        }
    }

    // 遍历 dimlists，将其中存储的维度直接展开到输入 input 中
    for (int64_t i = dimlists.size() - 1; i >=0; --i) {
        auto idx = dimlists[i];
        // 如果没有 unbound_dim_list 且 expanding_object 不为 -1 且 idx 大于 expanding_object，则调整 idx
        if (!unbound_dim_list && expanding_object != -1 && idx > expanding_object) {
            idx += expanding_dims;
        }
        // 解包 input[idx] 的 DimList 对象 dl
        auto dl = DimList::unchecked_wrap(input[idx]);
        // 创建 Slice<mpy::handle> 对象 more_dims，存储 dl 中的维度
        Slice<mpy::handle> more_dims((mpy::handle*) &*dl->dims_.begin(), (mpy::handle*) &*dl->dims_.end());
        // 在 input 中的 idx 处插入 more_dims
        input.insert(A, input.slice(idx, idx + 1), more_dims);
    }

    // 调用 getsetitem_flat 函数，返回其结果
    return getsetitem_flat(A, self_info, input, Slice<DimEntry>(), Slice<mpy::handle>(), has_dimpacks_or_none);
    }
    // 结束对 getsetitem_flat 函数的实现

    // 返回 IndexingInfo 结构，表示对张量的扁平化索引操作结果
    IndexingInfo getsetitem_flat(Arena& A, TensorInfo self_info, Slice<mpy::handle> input, Slice<DimEntry> keys, Slice<mpy::handle> values, bool has_dimpacks_or_none) {
        // 在这一点上：
        // ..., DimList 已被消除
        // Dim, Tensor, Tuple[Dim,...], int, slice 仍然存在

        // 我们需要计算每个维度出现的次数。
        // A[i,j] 是一个简单的绑定操作，但 A[i, i+j] 或 A[i, i] 则需要高级索引。
        Slice<mpy::hdl<Dim>> seen_dims;
        Slice<int64_t> seen_dims_nuses;

        // 添加维度到 seen_dims 和 seen_dims_nuses 中的 Lambda 函数
        auto add_dim = [&](mpy::hdl<Dim> entry) {
            auto midx = seen_dims.index(entry);
            if (!midx) {
                seen_dims.append(A, entry);
                seen_dims_nuses.append(A, 1);
            } else {
                ++seen_dims_nuses[*midx];
            }
        };

        Slice<mpy::handle> input_it = input;

        Slice<mpy::handle> flat_inputs;
        Slice<TensorInfo> tensor_inputs;

        // 向 flat_inputs 中添加 mpy::handle 的 Lambda 函数
        auto append_flat_handle = [&](mpy::handle h) {
            flat_inputs.append(A, h);
            tensor_inputs.append(A, TensorInfo());
        };

        TensorRef device_holding_tensor;
        // 向 tensor_inputs 中添加 TensorInfo 的 Lambda 函数
        auto append_tensor_input = [&](TensorInfo ti) {
            flat_inputs.append(A, mpy::handle());
            tensor_inputs.append(A, ti);
            if (ti.has_device && !device_holding_tensor) {
                device_holding_tensor = ti.tensor;
            }
        };

        Slice<int64_t> nsz;
        Slice<int64_t> nsd;
        at::IntArrayRef sz = self_info.tensor->sizes();
        at::IntArrayRef sd = self_info.tensor->strides();

        // 向 nsz 和 nsd 中添加尺寸信息的 Lambda 函数
        auto append_size = [&](int i) {
            if (has_dimpacks_or_none) {
                nsz.append(A, sz[i]);
                nsd.append(A, sd[i]);
            }
        };

        // 解析输入中的 None 值的 Lambda 函数
        auto parse_nones = [&]() {
            while (input_it.size() && mpy::is_none(input_it[0])) {
                append_flat_handle(no_slice);
                nsz.append(A, 1);
                nsd.append(A, 0);
                input_it = input_it.slice(1);
            }
        };

        // std::cout << "self levels: " << self_info.levels << "\n";

        // 返回空的 IndexingInfo 结构，暂时表示函数没有返回值
        return IndexingInfo();
    }
    // 定义一个 lambda 函数 append_item，用于处理不同类型的索引项
    auto append_item = [&](int i, mpy::handle arg) {
        // 如果参数 arg 是确切的维度对象
        if (Dim::check_exact(arg)) {
            // 将 arg 包装成维度对象 d，设置其大小为 sz[i]
            auto d = Dim::unchecked_wrap(arg);
            d->set_size(sz[i]);
            // 将维度对象 d 添加到当前对象的维度列表中
            add_dim(d);
            // 将索引项 i 添加到索引列表中
            append_size(i);
            // 添加扁平化处理后的句柄 arg
            append_flat_handle(arg);
            return;
        }
        // 否则，根据参数 arg 创建张量信息对象 info
        auto info = TensorInfo::create(A, arg, false, false);
        if (info) {
            // 如果成功创建了张量信息对象
            append_size(i);
            // 添加张量输入到索引列表中
            append_tensor_input(info);
            // 遍历张量信息对象中的层级信息
            for (auto il : info.levels) {
                // 如果当前层级不是位置信息
                if (!il.is_positional()) {
                    // 将该层级的维度添加到当前对象的维度列表中
                    add_dim(il.dim());
                }
            }
            return;
        }

        // 如果可能存在维度包或为 None 的情况
        if (has_dimpacks_or_none) {
            Slice<mpy::handle> mp;
            // 尝试将参数 arg 转换为维度包
            if (maybe_dimpack(mp, arg)) {
                // 维度包情况
                // 创建一个维度包的切片 dim_pack
                Slice<mpy::hdl<Dim>> dim_pack;
                for (auto d : mp) {
                    // 将每个维度对象包装后添加到 dim_pack 中
                    dim_pack.append(A, Dim::wrap(d));
                    // 将维度对象添加到当前对象的维度列表中
                    add_dim(dim_pack.back());
                    // 添加扁平化处理后的句柄 dim_pack.back()
                    append_flat_handle(dim_pack.back());
                }
                // 将维度包与尺寸信息绑定到当前对象上
                _bind_dims_to_size(A, sz[i], sd[i], dim_pack, nsz, nsd);
                return;
            }
        }

        // 默认情况下，将索引项 i 添加到索引列表中
        append_size(i);
        // 添加扁平化处理后的句柄 arg
        append_flat_handle(arg);
    };

    // 将索引表达式与当前对象的维度配对
    // 当前对象可能具有一流维度，这些维度不参与索引
    for (auto i : self_info.levels.enumerate()) {
        // 获取当前层级 l
        auto l = self_info.levels[i];
        // 在 keys 中查找层级 l 对应的索引 idx
        auto idx = keys.index(l);
        if (idx) {
            // 如果找到了对应的索引，将值 values[*idx] 作为参数调用 append_item 处理
            append_item(i, values[*idx]);
        } else if (l.is_positional()) {
            // 如果层级 l 是位置信息
            // 解析 None 值
            parse_nones();
            if (!input_it.size()) {
                // 如果输入迭代器为空，隐式地使用 no_slice 索引剩余的维度
                append_flat_handle(no_slice);
                append_size(i);
            } else {
                // 否则，从输入迭代器中获取参数 arg
                mpy::handle arg = input_it[0];
                input_it = input_it.slice(1);
                // 调用 append_item 处理参数 arg
                append_item(i, arg);
            }
        } else {
            // 否则，将层级 l 的维度添加到当前对象的维度列表中
            add_dim(l.dim());
            // 添加扁平化处理后的句柄 l.dim()
            append_flat_handle(l.dim());
            // 将索引项 i 添加到索引列表中
            append_size(i);
        }
    }

    // 处理任何训练过程中的 None 值可能导致当前对象没有关联的维度
    parse_nones();

    // 如果存在维度包或 None 的情况，需要重新调整张量以折叠维度包并引入 None 维度
    if (has_dimpacks_or_none) {
        self_info.tensor = A.autorelease(self_info.tensor->as_strided(at::IntArrayRef(nsz.begin(), nsz.end()),at::IntArrayRef(nsd.begin(), nsd.end()), self_info.tensor->storage_offset()));
    }

    // 确定索引张量的形状及结果张量的形状
    Slice<DimEntry> result_levels;
    Slice<DimEntry> index_levels;
    int64_t tensor_insert_point = -1;
    bool requires_getindex = false;
    // 定义一个 lambda 函数 mark_tensor_index，用于标记张量索引点
    auto mark_tensor_index = [&] {
        // 如果 tensor_insert_point 为 -1，将其设为 result_levels 的当前大小
        if (tensor_insert_point == -1) {
            tensor_insert_point = result_levels.size();
        } else if (tensor_insert_point != result_levels.size()) {
            // 否则，将 tensor_insert_point 设为 0
            tensor_insert_point = 0;
        }
    };

    // 遍历 flat_inputs 中的每个元素 i
    for (auto i : flat_inputs.enumerate()) {
        // 获取 flat_inputs 中的第 i 个元素
        auto inp = flat_inputs[i];
        
        // 如果 tensor_inputs[i] 存在
        if(tensor_inputs[i]) {
            // 设置 requires_getindex 为 true
            requires_getindex = true;
            // 调用 mark_tensor_index 函数，标记张量索引点
            mark_tensor_index();
            
            // 遍历 tensor_inputs[i] 的 levels
            for (auto l : tensor_inputs[i].levels) {
                // 如果 index_levels 不包含 l，则将其追加到 index_levels 中
                if (!index_levels.contains(l)) {
                    index_levels.append(A, l);
                }
            }
        } else if (Dim::check_exact(inp)) {
            // 如果 inp 是精确的维度对象
            auto d = Dim::unchecked_wrap(inp);
            
            // 维度只使用一次则视为绑定操作
            if (1 == seen_dims_nuses[*seen_dims.index(d)]) {
                // 将 flat_inputs[i] 设置为 no_slice
                flat_inputs[i] = no_slice;
                // 将 d 添加到 result_levels 中
                result_levels.append(A, d);
            } else {
                // 否则，设置 requires_getindex 为 true
                requires_getindex = true;
                // 将 flat_inputs[i] 设置为 mpy::handle()
                flat_inputs[i] = mpy::handle();
                // 初始化 tensor_inputs[i] 的信息
                tensor_inputs[i] = TensorInfo {d->range(), Slice<DimEntry>(A, DimEntry(d)), false, TensorRef()};
                // 如果 index_levels 不包含 d，则将其追加到 index_levels 中
                if (!index_levels.contains(d)) {
                     index_levels.append(A, d);
                }
                // 标记张量索引点
                mark_tensor_index();
            }
         } else {
            // 如果 inp 的指针不等于 no_slice 的指针
            if (inp.ptr() != no_slice.ptr()) {
                // 设置 requires_getindex 为 true
                requires_getindex = true;
            }
            // 如果 inp 不是整数
            if (!mpy::is_int(inp)) {
                // 将 -1 添加到 result_levels 中
                result_levels.append(A, -1);
            }
         }
    }

    // 如果 tensor_insert_point 不为 -1，则将 index_levels 插入到 result_levels 中的 tensor_insert_point 处
    if (tensor_insert_point != -1) {
        result_levels.insert(A, result_levels.slice(tensor_insert_point, tensor_insert_point), index_levels);
    }

    // 输出 flat_inputs 的内容（注释掉的输出语句）
    // 输出 result_levels 的内容（注释掉的输出语句）
    // 输出 index_levels 的内容（注释掉的输出语句）

    // 如果 requires_getindex 为 true，则将 flat_inputs 调整为适合索引的形状
    if (requires_getindex) {
        // 遍历 flat_inputs 中的每个元素 i
        for (auto i : flat_inputs.enumerate()) {
            // 如果 tensor_inputs[i] 存在
            if (tensor_inputs[i]) {
                // 断言 flat_inputs[i] 的指针不为空
                AT_ASSERT(!flat_inputs[i].ptr());
                // 输出张量的信息（注释掉的输出语句）
                // 获取 tensor_inputs[i] 的张量 t
                TensorRef t = tensor_inputs[i].tensor;
                // 如果 tensor_inputs[i] 没有设备，并且有设备持有张量，则将 t 调整为设备上的张量
                if (!tensor_inputs[i].has_device && device_holding_tensor) {
                    t = A.autorelease(t->to(device_holding_tensor->device()));
                }
                // 将 flat_inputs[i] 设置为根据索引和维度匹配的处理后的张量
                flat_inputs[i] = handle_from_tensor(A, _match_levels(A, t, tensor_inputs[i].levels, index_levels));
            }
        }
    }
    // 初始化计数器，用于跟踪已经处理的位置参数数量
    auto seen_positionals = 0;
    // 遍历结果级别列表，采用逆序遍历方式并且带有索引
    for (auto i : result_levels.reversed_enumerate()) {
        // 检查当前级别是否为位置参数
        if (result_levels[i].is_positional()) {
            // 如果是位置参数，则将其替换为负数的位置参数索引（从-1开始递减）
            result_levels[i] = -(++seen_positionals);
        }
    }

    // 构造并返回 IndexingInfo 结构体，其中包含多个成员：
    // - 是否为索引操作
    // - 是否需要进行 getindex 操作
    // - self_info 的张量信息
    // - 扁平化后的输入
    // - 处理后的结果级别列表
    // - self_info 是否包含设备信息
    return IndexingInfo {false, requires_getindex, self_info.tensor, flat_inputs, result_levels, self_info.has_device};
// 定义了一个名为 __getitem__ 的函数，用于从对象中获取元素并返回 Python 对象
mpy::object __getitem__(Arena & A, mpy::handle self, mpy::handle index) {
    // 初始化全局变量（如果尚未初始化）
    maybeInitializeGlobals();
    // 获取设置项的信息，根据对象和索引，以及对象是否具有维度信息
    auto iinfo = getsetitem(A, self, index, has_dims(self));
    // 如果可以调用原始对象的获取元素方法，则返回获取的对象作为 Python 对象
    if (iinfo.can_call_original) {
        return mpy::object::checked_steal(THPVariable_getitem(self.ptr(), index.ptr()));
    }
    // 否则，调用自定义的获取元素方法，并返回结果
    return invoke_getitem(A, iinfo);
}

// 定义了一个名为 __setitem__ 的函数，用于将值设置到对象的指定索引位置
void __setitem__(Arena & A, mpy::handle self, mpy::handle index, mpy::handle rhs) {
    // 初始化全局变量（如果尚未初始化）
    maybeInitializeGlobals();
    // 获取设置项的信息，根据对象和索引，以及对象和右侧值是否具有维度信息
    auto iinfo = getsetitem(A, self, index, has_dims(self) || has_dims(rhs));
    // 如果可以调用原始对象的设置元素方法
    if (iinfo.can_call_original) {
        // 调用 THPVariable_setitem 函数设置元素，并检查是否成功，否则抛出异常
        if (-1 == THPVariable_setitem(self.ptr(), index.ptr(), rhs.ptr())) {
            throw mpy::exception_set();
        }
        return;
    }
    // 获取右侧值的张量信息
    auto rhs_info = TensorInfo::create(A, rhs, false, false);
    // 如果存在右侧值的张量信息
    if (rhs_info) { // otherwise rhs can be a scalar...
        // 遍历右侧值的级别
        for (auto l : rhs_info.levels) {
            // 如果结果级别不包含当前级别，则根据级别类型抛出维度绑定错误异常
            if (!iinfo.result_levels.contains(l)) {
                if (l.is_positional()) {
                    mpy::raise_error(DimensionBindError(), "rhs contains too many dimensions (%d) compared to indexed value (%d)", ndim_of_levels(iinfo.result_levels), rhs_info.ndim());
                } else {
                    auto tup = levels_to_tuple(iinfo.result_levels);
                    mpy::raise_error(DimensionBindError(), "rhs of setitem contains dimension %R which is not in the dimension on the left (%R)", l.dim().ptr(), tup.ptr());
                }
            }
        }
        // 将右侧值与结果级别匹配，并获取匹配后的处理后的张量
        auto rhs_matched = _match_levels(A, rhs_info.tensor, rhs_info.levels, iinfo.result_levels);
        rhs = handle_from_tensor(A, rhs_matched);
    }
    // 将对象转换为处理后的张量
    self = handle_from_tensor(A, iinfo.self);
    // 如果是高级索引
    if (iinfo.advanced_indexing) {
        // 将切片转换为元组，并调用 THPVariable_setitem 函数设置元素，并检查是否成功，否则抛出异常
        auto tup = slice_to_tuple(iinfo.flat_inputs);
        if (-1 == THPVariable_setitem(self.ptr(), tup.ptr(), rhs.ptr())) {
            throw mpy::exception_set();
        }
    } else {
        // 否则，调用 torch_Tensor_copy_ 函数将右侧值复制到对象
        torch_Tensor_copy_.call(self, rhs);
    }
}

// 定义了名为 Tensor_getitem 的函数，用于从张量对象中获取元素并返回 Python 对象
PyObject* Tensor_getitem(PyObject* self, PyObject* index) {
    // 创建 Arena 对象，用于内存分配
    Arena A;
    // 开始 Python 代码执行块
    PY_BEGIN
    // 调用 __getitem__ 函数从对象中获取元素，并释放返回的 Python 对象的所有权
    return __getitem__(A, self, index).release();
    // 结束 Python 代码执行块，并返回空指针（nullptr），表示执行正常结束
    PY_END(nullptr);
}

// 定义了名为 Tensor_setitem 的函数，用于将值设置到张量对象的指定索引位置
int Tensor_setitem(PyObject* self, PyObject* index, PyObject* value) {
    // 创建 Arena 对象，用于内存分配
    Arena A;
    // 开始 Python 代码执行块
    PY_BEGIN
    // 调用 __setitem__ 函数设置对象的元素
    __setitem__(A, self, index, value);
    // 返回 0，表示设置操作成功完成
    return 0;
    // 结束 Python 代码执行块，并返回 -1，表示设置操作失败
    PY_END(-1);
}

// 定义了匿名命名空间，包含 Python 的特殊方法 __getitem__ 的实现
namespace {
PyObject* py___getitem__(PyObject *_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    // 创建 Arena 对象，用于内存分配
    Arena A;
    // 开始 Python 代码执行块
    PY_BEGIN
    // 确保传入的参数个数为 2
    AT_ASSERT(nargs == 2);
    // 调用 __getitem__ 函数从对象中获取元素，并释放返回的 Python 对象的所有权
    return __getitem__(A, args[0], args[1]).release();
    // 结束 Python 代码执行块，并返回空指针（nullptr），表示执行正常结束
    PY_END(nullptr)
}

// 定义了匿名命名空间，包含 Python 的特殊方法 __setitem__ 的实现
PyObject* py___setitem__(PyObject *_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    // 创建 Arena 对象，用于内存分配
    Arena A;
    // 开始 Python 代码执行块
    PY_BEGIN
    // 确保传入的参数个数为 3
    AT_ASSERT(nargs == 3);
    // 调用 __setitem__ 函数设置对象的元素
    __setitem__(A, args[0], args[1], args[2]);
    // 返回 Python None 对象，表示执行成功但不返回任何值
    Py_RETURN_NONE;
    // 结束 Python 代码执行块，并返回空指针（nullptr），表示执行正常结束
    PY_END(nullptr)
}
}
// 创建 Arena 对象，用于管理内存分配
Arena A;
// 执行 Python C API 调用的起始宏，处理异常和内存管理
PY_BEGIN
// 解析 Python 函数的参数列表
mpy::vector_args va(args, nargs, kwnames);
// 定义并初始化用于存储 Python 对象的句柄
mpy::handle self, split_size_or_sections, dim;
// 解析函数参数并将其关联到相应的 Python 对象
va.parse("split", {"self", "split_size_or_sections", "dim"}, {&self, &split_size_or_sections, &dim}, 2);

// 检查 dim 是否为对象类型，并返回布尔值
bool dim_is_object = dim.ptr() && Dim::check_exact(dim);
// 定义并初始化存储 mpy::handle 对象的切片
Slice<mpy::handle> sizes;

// 初始化布尔标志，用于检查 split_size_or_sections 中的数据类型
bool all_dims = true;
bool all_ints = true;

// 如果 split_size_or_sections 不是整数，则将其视为序列视图处理
if (!mpy::is_int(split_size_or_sections)) {
    // 创建序列视图，遍历其中的每个元素
    mpy::sequence_view sv(split_size_or_sections);
    for (auto i : sv.enumerate()) {
        // 将 sv[i] 释放到 Arena A 中，并追加到 sizes 切片中
        sizes.append(A, A.autorelease(sv[i]));
        // 检查 sizes 的每个元素是否为精确的维度对象
        if (Dim::check_exact(sizes.back())) {
            all_ints = false;
        } else {
            all_dims = false;
        }
    }
}
    // 如果所有的分割大小都是整数
    if (all_ints) {
        // 如果维度是对象，抛出类型错误，要求分割大小必须是维度
        if (dim_is_object) {
            mpy::raise_error(PyExc_TypeError, "when dim is specified as a Dim object, split sizes must also be dimensions.");
        }
        // 调用原始的 split 函数（如果自身具有维度，则使用 Torch 函数进行分割）
        return torch_Tensor_split.call_vector(mpy::vector_args(args, nargs, kwnames)).release();
    }
    // 如果不是所有的分割大小都是维度
    if (!all_dims) {
        // 抛出类型错误，要求分割列表必须是整数或维度，不允许混合
        mpy::raise_error(PyExc_TypeError, "split list must be ints or dims but got a mix");
    }

    // 创建 self 的 TensorInfo 对象
    auto self_info = TensorInfo::create(A, self, false);
    // 获取 self 的维度数量
    auto ndim = self_info.ndim();
    // 如果维度不是对象并且 self 是零维张量，则抛出类型错误，要求至少是 1 维张量
    if (!dim_is_object && ndim == 0) {
        mpy::raise_error(PyExc_TypeError, "split expects at least a 1-dimension tensor");
    }
    // 将 dim 转换为 DimEntry，如果为空则设为负的维度数量
    DimEntry dim_l = dim.ptr() ? _wrap_dim(dim, ndim, false) : -ndim;

    // 查找 dim_l 在 self 的维度列表中的索引
    auto idx = self_info.levels.index(dim_l);
    // 如果索引不存在
    if (!idx) {
        // 如果 dim 为空，则设为 0
        if (!dim.ptr()) {
            dim = A.autorelease(mpy::from_int(0));
        }
        // 抛出类型错误，指明 self 不包含指定的维度
        mpy::raise_error(PyExc_TypeError, "tensor does not comtain dimension %R", dim.ptr());
    }

    // 初始化 indices 为 Slice<int64_t>
    Slice<int64_t> indices;
    // 初始化 total_size 为 0
    int64_t total_size = 0;
    // 初始化 unbound 为 Slice<int64_t>
    Slice<int64_t> unbound;
    // 遍历 sizes 中的每个元素
    for (auto i : sizes.enumerate()) {
        // 将 sizes[i] 包装为 Dim 对象
        auto d = Dim::unchecked_wrap(sizes[i]);
        // 如果 d 是有界的
        if (d->is_bound()) {
            // 将 d 的大小添加到 indices 中
            indices.append(A, d->size());
            // 累加 total_size
            total_size += indices.back();
        } else {
            // 将 0 添加到 indices 中
            indices.append(A, 0);
            // 将 i 添加到 unbound 中
            unbound.append(A, i);
        }
    }

    // 获取 self 的 tensor_size
    auto tensor_size = self_info.tensor->sizes()[*idx];

    // 如果存在未绑定的维度
    if (unbound.size()) {
        // 如果 total_size 大于 tensor_size，则抛出类型错误，说明目标维度的大小总和超过了源维度
        if (total_size > tensor_size) {
           mpy::raise_error(PyExc_TypeError, "sizes of target dimensions add up to more (%d) than source dim (%d)", int(total_size), int(tensor_size));
        }
        // 计算剩余大小
        auto remaining_size = tensor_size - total_size;
        // 计算每个未绑定维度的块大小
        auto chunk_size = (remaining_size + unbound.size() - 1) / unbound.size();
        // 遍历 unbound 中的每个索引 u
        for (auto u : unbound) {
            // 计算当前维度的大小 sz，为剩余大小和块大小的最小值
            auto sz = std::min(chunk_size, remaining_size);
            // 将 sizes[u] 包装为 Dim 对象，并设置其大小为 sz
            Dim::unchecked_wrap(sizes[u])->set_size(sz);
            // 更新 indices 中的第 u 个元素为 sz
            indices[u] = sz;
            // 减去已分配的大小
            remaining_size -= sz;
        }
    // 如果不存在未绑定的维度但 tensor_size 不等于 total_size
    } else if (tensor_size != total_size) {
        // 抛出类型错误，说明目标维度的大小总和与源维度不匹配
        mpy::raise_error(PyExc_TypeError, "sum of sizes of target dimensions (%d) do not match the than source dim (%d)", int(total_size), int(tensor_size));
    }

    // 使用 indices 分割 self_info.tensor，并得到结果张量数组 result_tensors
    auto result_tensors = self_info.tensor->split_with_sizes(at::IntArrayRef(indices.begin(), indices.end()), *idx);
    // 创建长度为 result_tensors.size() 的 mpy::tuple 对象 result
    mpy::tuple result(result_tensors.size());
    // 初始化 new_levels 为 Slice<DimEntry>，并扩展为 self_info.levels 的副本
    Slice<DimEntry> new_levels;
    new_levels.extend(A, self_info.levels);
    // 遍历 sizes 中的每个元素
    for (auto i : sizes.enumerate()) {
        // 将 sizes[i] 包装为 Dim 对象，并将其赋值给 new_levels[*idx]
        new_levels[*idx] = Dim::unchecked_wrap(sizes[i]);
        // 将 result_tensors[i] 转换为 Tensor，并使用 new_levels 创建 Tensor 对象，将其设置为 result 的第 i 个元素
        result.set(i, Tensor::from_positional(A, std::move(result_tensors[i]), new_levels, true));
    }

    // 返回 result 的所有权给调用者
    return result.release();

    // Python 扩展结束
    PY_END(nullptr)
}

// 将维度包装成 DimEntry 切片的辅助函数
Slice<DimEntry> _wrap_dims(Arena& A, mpy::handle d, size_t N, bool keepdim) {
    // 调用 _wrap_dim 函数将单个维度包装成 DimEntry 对象
    auto de = _wrap_dim(d, N, keepdim);
    // 创建 DimEntry 对象的切片 r
    Slice<DimEntry> r;
    // 如果 de 不为空
    if (!de.is_none()) {
        // 将 de 添加到切片 r 中
        r.append(A, de);
    } else {
        // 如果 de 为空，从序列视图 sq 中枚举每个元素 i
        mpy::sequence_view sq(d);
        for (auto i : sq.enumerate()) {
            // 将序列中的元素逐个包装成 DimEntry，并添加到切片 r 中
            r.append(A, _wrap_dim(A.autorelease(sq[i]), N, keepdim));
        }
    }
    // 返回包装后的 DimEntry 切片
    return r;
}

// 定义一个 WrappedOperator 结构体，继承自 mpy::base
struct WrappedOperator : public mpy::base<WrappedOperator> {
    // 成员变量
    mpy::object orig;
    PyMethodDef method_def;
    mpy::object name, doc;
    bool is_pointwise = false;
    int64_t dim_offset = 0;
    int64_t keepdim_offset = 1;
    std::string dim_name;
    bool single_dim = false;
    bool reduce = true;

    // 静态成员变量 Type，表示 Python 类型对象
    static PyTypeObject Type;

    // 初始化方法，接受原始对象、包装实现、和维度名称
    void init(mpy::object orig_, PyCFunction wrapper_implementation, std::string dim_name_="") {
        orig = std::move(orig_);
        method_def.ml_meth = wrapper_implementation;
        name = orig.attr("__name__");
        doc = orig.attr("__doc__");
        dim_name = std::move(dim_name_);
        // 如果原始对象有文档并且维度名称不为空，则创建相应的文档
        if (!mpy::is_none(doc) && !dim_name.empty()) {
            doc = mpy::unicode_from_format("%S\nArgument '%s' can be either an integer or a torchdim.Dim object.\n", doc.ptr(), dim_name.c_str());
        }
        // 设置方法定义的名称和文档
        method_def.ml_name = mpy::is_none(name) ? "" : PyUnicode_AsUTF8(name.ptr());
        method_def.ml_doc = mpy::is_none(doc) ? "" : PyUnicode_AsUTF8(doc.ptr());
        method_def.ml_flags = METH_FASTCALL | METH_KEYWORDS;
    }

    // 返回函数对象
    mpy::object function() {
        return mpy::object::checked_steal(PyCFunction_New(&method_def, ptr()));
    }
};
}

// 定义 WrappedOperator 类型对象的 Python 类型
PyTypeObject WrappedOperator::Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_C.WrappedOperator",               /* tp_name */
    sizeof(WrappedOperator),               /* tp_basicsize */
    0,                              /* tp_itemsize */
    WrappedOperator::dealloc_stub,      /* tp_dealloc */
    0,                              /* tp_vectorcall_offset */
    0,                              /* tp_getattr */
    0,                              /* tp_setattr */
    0,                              /* tp_as_async */
    0,           /* tp_repr */
    0,                 /* tp_as_number */
    0,                 /* tp_as_sequence */
    0,             /* tp_as_mapping */
    0,      /* tp_hash */
    0,                              /* tp_call */
    0,                              /* tp_str */
    0,                              /* tp_getattro */
    0,                              /* tp_setattro */
    0,                              /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    "Wrapped Object Holder",                   /* tp_doc */
    0,                              /* tp_traverse */
    0,                              /* tp_clear */
    0,  /* tp_richcompare */
    0,                              /* tp_weaklistoffset */
    0,                              /* tp_iter */
    0,                              /* tp_iternext */
    0,                /* tp_methods */
    0,                              /* tp_members */
    0,             /* tp_getset */
    0,                              /* tp_base */
    0,                              /* tp_dict */
    0,                              /* tp_descr_get */
    0,                              /* tp_descr_set */
    0,                              /* tp_dictoffset */
    0,            /* tp_init */
    0,                              /* tp_alloc */
    WrappedOperator::new_stub,                      /* tp_new */
// 匿名命名空间，用于定义本文件内部的局部函数和变量
namespace {
    // 定义函数 patched_dim_method，该函数为 Python C API 的方法封装提供了补丁功能
    PyObject* patched_dim_method(PyObject * self_,
                          PyObject *const *args,
                          Py_ssize_t nargs,
                          PyObject *kwnames) {
        // Arena A 是一个内存管理工具
        Arena A;
        // 使用 self_ 包装为 WrappedOperator 对象，这里不检查合法性
        auto self = WrappedOperator::unchecked_wrap(self_);
        PY_BEGIN  // 宏，Python C API 错误处理的开始标记

        // vector_args 对象 va 将 Python 参数数组 args 转换为 C++ vector
        mpy::vector_args va(args, nargs, kwnames);

        // _getarg 函数获取指定名称的参数值，offset_ 是偏移量
        auto _getarg = [&](const char* name, int64_t offset_) -> mpy::handle {
            auto offset = offset_ + 1; // 不包括 self 参数
            auto idx = va.index(name, offset);
            return idx == -1 ? mpy::handle() : va[idx];
        };

        // 初始化 patched_args 数组，并扩展以包含所有参数
        Slice<mpy::handle> patched_args;
        patched_args.extend(A, va.begin(), va.end());

        // _patcharg 函数用于修改指定参数的值
        auto _patcharg = [&](const char* name, int64_t offset_, mpy::handle value) {
            auto offset = offset_ + 1; // 不包括 self 参数
            auto idx = va.index(name, offset);
            if (idx == -1) {
                // 抛出异常，指示缺少指定的参数
                mpy::raise_error(PyExc_ValueError, "Missing argument %s", name);
            }
            patched_args[idx] = value;
        };

        // 获取名为 dim_name 的参数 dim，并对其进行处理
        auto dim = _getarg(self->dim_name.c_str(), self->dim_offset);
        if (!dim.ptr()) {
            // 如果 dim 不存在，创建 TensorInfo 对象 info，并更新所有层级
            auto info = TensorInfo::create(A, args[0], true);
            EnableAllLayers l(A, info.levels);
            l.inplace_update_layers(info.batchedtensor, info.levels);
            // 将 info.batchedtensor 转换为 Python 对象，调用原始函数 self->orig.call_vector
            patched_args[0] = handle_from_tensor(A, info.batchedtensor);
            auto r = self->orig.call_vector(patched_args.begin(), nargs, kwnames);
            // 从批处理张量中提取结果，返回新的 Python 对象
            return l.from_batched(A, THPVariable_Unpack(r.ptr()), info.has_device).release();
        }

        // 如果 dim 存在，创建 TensorInfo 对象 info，获取是否保持维度的标志 keepdim
        auto info = TensorInfo::create(A, args[0]);
        auto keepdim = false;
        if (self->reduce) {
            // 如果需要 reduce，获取 keepdim 参数值
            auto py_keepdim = _getarg("keepdim", self->keepdim_offset);
            if (py_keepdim.ptr()) {
                keepdim = mpy::to_bool(py_keepdim);
            }
        }

        // 获取 info 的维度数 ndim 和包装后的维度 dims
        auto ndim = info.ndim();
        auto dims = _wrap_dims(A, dim, ndim, keepdim);

        // 初始化 dim_indices，并标记已经使用过的层级
        Slice<int64_t> dim_indices;
        auto seen = A.allocate<bool>(info.levels.size());
        std::fill(seen, seen + info.levels.size(), false);

        // 遍历 dims 中的维度值，并将对应层级的索引添加到 dim_indices 中
        for (auto d : dims) {
            auto midx = info.levels.index(d);
            if (!midx) {
                // 如果维度值不在 info.levels 中，抛出异常
                auto tup = levels_to_tuple(info.levels);
                mpy::raise_error(PyExc_ValueError, "Tensor with dimensions %R does not contain one of %R\n", tup.ptr(), dim.ptr());
            }
            seen[*midx] = true;
            dim_indices.append(A, *midx);
        }

        // 初始化 new_levels 以存储新的层级信息
        Slice<DimEntry> new_levels;
        if (self->reduce && !keepdim) {
            // 如果需要 reduce 且不保持维度，过滤掉未使用的层级
            for (auto i : info.levels.enumerate()) {
                if (!seen[i]) {
                    new_levels.append(A, info.levels[i]);
                }
            }
        } else {
            // 否则，直接使用 info.levels
            new_levels = info.levels;
        }

        // 根据 dim_indices 的大小创建 py_indices 对象，将其转换为 Python 对象
        mpy::object py_indices;
        if (dim_indices.size() == 1) {
            py_indices = mpy::from_int(dim_indices[0]);
        } else {
            mpy::tuple tup(dim_indices.size());
            for (auto i : dim_indices.enumerate()) {
                tup.set(i, mpy::from_int(dim_indices[i]));
            }
            py_indices = std::move(tup);
        }

        // 将 py_indices 更新到参数列表中
        _patcharg(self->dim_name.c_str(), self->dim_offset, py_indices);
    # 将第一个参数替换为从张量 A 和信息对象中获取的处理结果
    patched_args[0] = handle_from_tensor(A, info.tensor);
    
    # 调用原始函数 self->orig.call_vector，使用修正后的参数列表 patched_args，以及参数个数 nargs 和关键字参数 kwnames
    auto r = self->orig.call_vector(patched_args.begin(), nargs, kwnames);
    
    # 定义一个 lambda 函数 wrap，用于处理返回值中的 mpy::handle 对象
    auto wrap = [&](mpy::handle h) {
        if (THPVariable_Check(h.ptr())) {
            # 如果 h 是 THPVariable 类型的对象，则从中解包数据并创建新的 Tensor 对象，考虑新的级别 new_levels 和设备信息 info.has_device，最后自动释放 A
            return A.autorelease(Tensor::from_positional(A, THPVariable_Unpack(h.ptr()), new_levels, info.has_device));
        }
        # 如果 h 不是 THPVariable 类型的对象，直接返回 h
        return h;
    };
    
    # 对 r 应用 tree_map 函数，使用 wrap 函数处理每个元素，并释放结果
    return tree_map(A, wrap, r).release();
    PY_END(nullptr)
}

PyObject* _wrap(PyObject * self_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    // 创建一个 Arena 对象，用于管理内存分配
    Arena A;
    // 开始 Python 解释器锁
    PY_BEGIN

    // 定义宏 ARGS，用于解析参数
    #define ARGS(_) _(mpy::handle, orig) _(mpy::handle, dim_offset) _(mpy::handle, keepdim_offset) \
                    _(mpy::handle, dim_name) _(mpy::handle, single_dim) _(mpy::handle, reduce)
    // 解析参数并获取关键字参数名
    MPY_PARSE_ARGS_KWNAMES("O|OOOOO", ARGS)

    // 将 dim_name 转换为字符串，如果为空则使用默认值 "dim"
    std::string dim_name_str;
    if (dim_name.ptr()) {
        dim_name_str = PyUnicode_AsUTF8(dim_name.ptr());
    } else {
        dim_name_str = "dim";
    }
    // 创建 WrappedOperator 对象 info，调用 WrappedOperator::create 方法
    auto info = WrappedOperator::create(mpy::object::borrow(orig), (PyCFunction)(void*) patched_dim_method, std::move(dim_name_str));
    // 如果 dim_offset 不为空，则设置 info->dim_offset
    if (dim_offset.ptr()) {
        info->dim_offset = mpy::to_int(dim_offset);
    }
    // 如果 keepdim_offset 不为空，则设置 info->keepdim_offset
    if (keepdim_offset.ptr()) {
        info->keepdim_offset = mpy::to_int(keepdim_offset);
    }

    // 如果 single_dim 不为空，则设置 info->single_dim
    if (single_dim.ptr()) {
        info->single_dim = mpy::to_bool(single_dim);
    }
    // 如果 reduce 不为空，则设置 info->reduce
    if (reduce.ptr()) {
        info->reduce = mpy::to_bool(reduce);
    }
    // 释放 Python 解释器锁并返回 info 对象的函数指针
    return info->function().release();
    // 取消宏定义 ARGS
    #undef ARGS

    // 结束 Python 解释器锁并返回空指针
    PY_END(nullptr)
}

PyObject* call_torch_function(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    // 开始 Python 解释器锁
    PY_BEGIN
    // 创建一个 Arena 对象，用于管理内存分配
    Arena A;
    // 初始化全局变量（如果未初始化）
    maybeInitializeGlobals();
    // 使用 self 创建 WrappedOperator 对象 info
    auto info = WrappedOperator::unchecked_wrap(self);
    // 调用 __torch_function__ 方法，并返回其结果的释放指针
    return __torch_function__(A, info->orig, mpy::vector_args(args, nargs, kwnames), info->is_pointwise).release();
    // 结束 Python 解释器锁并返回空指针
    PY_END(nullptr)
}

PyObject* _wrap_method(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    // 开始 Python 解释器锁
    PY_BEGIN
    // 断言参数个数为 2
    AT_ASSERT(nargs == 2);
    // XXX - 忽略 Python 函数包装，直接调用 torch 函数
    mpy::handle orig = args[0];
    // 如果 pointwise 为空，则从 functorch.dim 模块导入 pointwise 函数
    if (!pointwise.ptr()) {
        auto dim = mpy::import("functorch.dim");
        pointwise = dim.attr("pointwise");
    }
    // 创建 WrappedOperator 对象 info，调用 WrappedOperator::create 方法
    auto info = WrappedOperator::create(mpy::object::borrow(orig), (PyCFunction)(void*) call_torch_function);
    // 设置 info->is_pointwise 标志，表示是否为 pointwise 函数
    info->is_pointwise = pointwise.contains(orig);
    // 返回 info 对象的函数实例方法指针
    return PyInstanceMethod_New(info->function().release());
    // 结束 Python 解释器锁并返回空指针
    PY_END(nullptr);
}

PyObject* Tensor_sum(PyObject * self_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    // 创建一个 Arena 对象，用于管理内存分配
    Arena A;
    // 开始 Python 解释器锁
    PY_BEGIN
    // 初始化全局变量（如果未初始化）
    maybeInitializeGlobals();
    // 解析参数并创建 mpy::vector_args 对象 va
    mpy::vector_args va(args, nargs, kwnames);
    // 将 args[0] 包装成 Tensor 对象 self_
    auto self_ = Tensor::unchecked_wrap(args[0]);
    // 获取 self_ 的延迟对象 d
    auto d = self_->delayed();
    // 如果 d 为空
    if (!d) {
        // 调用 _Tensor_sum 对象的 call_vector 方法，并释放其结果的指针
        return _Tensor_sum.call_vector(va).release();
    }
    // 定义变量 self, dim, keepdim, dtype，并解析参数
    mpy::handle self, dim, keepdim, dtype;
    va.parse("sum", {"self", "dim", "keepdim", "dtype"}, {&self, &dim, &keepdim, &dtype}, 1, 1);
    // 检查条件：如果 dtype 是指针，或者 keepdim 是指针并且 mpy::to_bool(keepdim) 返回 true
    if (dtype.ptr() || (keepdim.ptr() && mpy::to_bool(keepdim))) {
        // 如果满足上述条件，输出调试信息并直接返回使用向量参数调用 _Tensor_sum 的结果
        // 注意：这里是一个早返回，意味着后续代码将不会执行
        // std::cout << "SKIPPING fusion because dtype or keepdim=True specified\n";
        return _Tensor_sum.call_vector(va).release();
    }
    // 获取当前对象的 levels
    auto levels = self_->levels();

    // 计算 levels 的维度数
    auto N = ndim_of_levels(levels);

    // 根据给定的 A、dim、N 和 false，获取包装后的维度信息
    auto reduced_dims = _wrap_dims(A, dim, N, false);

    // 使用 dot 函数计算 A 与 d->args[0] 和 d->args[1] 的点积，并使用给定的 reduced_dims
    // 注意：此处调用 dot 函数并返回其结果
    return dot(A, TensorInfo::create(A, d->args[0], false), TensorInfo::create(A, d->args[1], false), reduced_dims).release();
    // 返回 PY_END(nullptr)，结束当前函数的执行
    PY_END(nullptr)
}

PyObject* _parse_test(PyObject * self_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    // 开始 Python C API 的异常处理块
    PY_BEGIN
    // 可能初始化全局变量
    maybeInitializeGlobals();

    // 将参数转换为整数类型
    int required = mpy::to_int(args[0]);
    int kwonly = mpy::to_int(args[1]);

    // 解析向量参数
    mpy::vector_args va(args + 2, nargs - 2, kwnames);

    // 处理 a, b, c, d 四个参数
    mpy::handle a, b, c, d;
    va.parse("_parse_test", {"a", "b", "c", "d"}, {&a, &b, &c, &d}, required, kwonly);

    // 创建一个包含四个元素的元组
    mpy::tuple r(4);
    // 将 a, b, c, d 转换为 Python 对象并放入元组中
    r.set(0, mpy::object::borrow(a.ptr() ? a : Py_None));
    r.set(1, mpy::object::borrow(b.ptr() ? b : Py_None));
    r.set(2, mpy::object::borrow(c.ptr() ? c : Py_None));
    r.set(3, mpy::object::borrow(d.ptr() ? d : Py_None));

    // 释放元组对象并返回
    return r.release();

    // 结束 Python C API 的异常处理块，如果出现异常返回 nullptr
    PY_END(nullptr)
}

PyObject* _set_pointwise_optimize(PyObject * self_,
                                  PyObject *const *args,
                                  Py_ssize_t nargs,
                                  PyObject *kwnames) {
    // 开始 Python C API 的异常处理块
    PY_BEGIN
    // 定义一个 Python 对象的句柄
    mpy::handle value;
    // 解析向量参数
    mpy::vector_args va(args, nargs, kwnames);
    // 解析单个参数 value
    va.parse("_set_pointwise_optimization", {"value"}, {&value}, 1);
    // 将 value 转换为布尔值并设置全局变量 pointwise_optimize
    pointwise_optimize = mpy::to_bool(value);
    // 返回 None
    Py_RETURN_NONE;

    // 结束 Python C API 的异常处理块，如果出现异常返回 nullptr
    PY_END(nullptr)
}

PyObject* _patch_tensor_class(PyObject * self_,
                              PyObject *const *args,
                              Py_ssize_t nargs,
                              PyObject *kwnames) {
    // 开始 Python C API 的异常处理块
    PY_BEGIN

    // 导入 torch 模块并获取 TensorBase 对象
    auto torch = mpy::import("torch");
    auto py_TensorBase = torch.attr("_C").attr("TensorBase");
    // 如果匹配则替换映射
    replaceMappingIfMatches(py_TensorBase);

    // 返回 None
    Py_RETURN_NONE;

    // 结束 Python C API 的异常处理块，如果出现异常返回 nullptr
    PY_END(nullptr)
}
    {"stack", (PyCFunction)(void*) py_stack, METH_FASTCALL | METH_KEYWORDS},
    {"split", (PyCFunction)(void*) py_split, METH_FASTCALL | METH_KEYWORDS},
    {"expand", (PyCFunction)(void*) expand, METH_FASTCALL | METH_KEYWORDS},
    {"__getitem__", (PyCFunction)(void*) py___getitem__, METH_FASTCALL | METH_KEYWORDS},
    {"__setitem__", (PyCFunction)(void*) py___setitem__, METH_FASTCALL | METH_KEYWORDS},
    {"_wrap", (PyCFunction)(void*) _wrap, METH_FASTCALL | METH_KEYWORDS},
    {"Tensor_sum", (PyCFunction)(void*) Tensor_sum, METH_FASTCALL | METH_KEYWORDS},
    {"_parse_test", (PyCFunction)(void*) _parse_test, METH_FASTCALL | METH_KEYWORDS},
    {"_set_pointwise_optimize", (PyCFunction)(void*) _set_pointwise_optimize, METH_FASTCALL | METH_KEYWORDS},
    {"_patch_tensor_class", (PyCFunction)(void*) _patch_tensor_class, METH_FASTCALL | METH_KEYWORDS},
    {NULL, NULL, 0, NULL}        /* Sentinel */



# 定义一个静态映射表格，用于将字符串与对应的 C 函数指针及调用方式关联起来
# 每一项都是一个包含函数名、函数指针、调用方式的元组
# 最后一项的 NULL 表示映射结束的标志，用于循环遍历时的终止条件
};

// 定义 Python 模块的结构体
struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,   // 模块定义的头初始化
    "_C",   /* name of module */   // 模块的名称为 "_C"
    NULL,   /* module documentation, may be NULL */   // 模块的文档字符串，可以为 NULL
    -1,     /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    methods // 模块使用的方法数组
};
}

// 初始化 Python 模块
PyObject* Dim_init() {
    // 创建一个 Arena 对象
    Arena A;
    try {
        // 创建 Python 模块并获取模块对象
        mpy::object mod = mpy::object::checked_steal(PyModule_Create(&module_def));
        // 初始化 Dim 类在模块中的表示
        Dim::ready(mod, "Dim");
        // 初始化 DimList 类在模块中的表示
        DimList::ready(mod, "DimList");
        // 初始化 Tensor 类在模块中的表示
        Tensor::ready(mod, "Tensor");
        // 初始化 WrappedOperator 类在模块中的表示
        WrappedOperator::ready(mod, "_WrappedOperator");
        // 增加 PyInstanceMethod_Type 的引用计数
        Py_INCREF(&PyInstanceMethod_Type);
        // 将 PyInstanceMethod_Type 添加到模块对象中
        PyModule_AddObject(mod.ptr(), "_instancemethod", (PyObject *)&PyInstanceMethod_Type);

        // 初始化全局变量
        initializeGlobals(A);
        // 返回模块对象
        return mod.release();
    } catch(mpy::exception_set& err) {
        // 捕获 mpy::exception_set 异常，返回空指针
        return nullptr;
    }
}

#endif
```