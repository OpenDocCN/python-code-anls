# `.\pytorch\functorch\csrc\dim\minpybind.h`

```
// 声明并保护代码的版权信息和许可证信息
// BSD风格许可证，详见根目录下的LICENSE文件
#pragma once

// 定义宏，确保Python的ssize_t类型在包含Python.h之前被声明为clean状态
#define PY_SSIZE_T_CLEAN
#include <Python.h>     // 引入Python C API头文件
#include <utility>      // 引入C++标准库中的utility头文件
#include <ostream>      // 引入C++标准库中的ostream头文件
#include <memory>       // 引入C++标准库中的memory头文件

// 定义宏，简化异常处理代码块的书写
#define PY_BEGIN try {  
#define PY_END(v) } catch(mpy::exception_set & err) { return (v); }

// 根据Python版本选择合适的函数调用宏
#if PY_VERSION_HEX < 0x03080000
    #define PY_VECTORCALL _PyObject_FastCallKeywords
#else
    #define PY_VECTORCALL _PyObject_Vectorcall
#endif

// 定义一个结构体irange，表示一个整数范围
struct irange {
 public:
    // 构造函数，使用结束位置初始化，开始位置默认为0，步长默认为1
    irange(int64_t end)
    : irange(0, end, 1) {}
    // 构造函数，使用开始、结束和步长初始化
    irange(int64_t begin, int64_t end, int64_t step = 1)
    : begin_(begin), end_(end), step_(step) {}
    // 解引用操作符，返回当前范围的起始位置
    int64_t operator*() const {
        return begin_;
    }
    // 前缀递增操作符，增加当前范围的起始位置，并返回修改后的对象引用
    irange& operator++() {
        begin_ += step_;
        return *this;
    }
    // 不等操作符，比较两个irange对象的起始位置是否相同
    bool operator!=(const irange& other) {
        return begin_ != other.begin_;
    }
    // 返回当前范围的起始位置，用于范围迭代的起始
    irange begin() {
        return *this;
    }
    // 返回表示结束位置的irange对象，用于范围迭代的终止
    irange end() {
        return irange {end_, end_, step_};
    }
 private:
    int64_t begin_;     // 范围的起始位置
    int64_t end_;       // 范围的结束位置
    int64_t step_;      // 范围的步长
};

// 命名空间mpy，用于定义Python相关的结构和函数
namespace mpy {

// 空结构体，用作异常抛出的标记
struct exception_set {
};

// 前置声明Python对象和向量参数结构
struct object;
struct vector_args;

// Python对象的包装类，封装PyObject指针
struct handle {
    // 构造函数，使用给定的PyObject指针初始化对象
    handle(PyObject* ptr)
    : ptr_(ptr) {}
    // 默认构造函数，默认初始化对象
    handle() = default;

    // 返回对象持有的PyObject指针
    PyObject* ptr() const {
        return ptr_;
    }
    // 获取对象的属性，根据键值返回相应的Python对象
    object attr(const char* key);
    // 检查对象是否具有指定的属性
    bool hasattr(const char* key);
    // 返回对象的类型对象
    handle type() const {
        return (PyObject*) Py_TYPE(ptr());
    }

    // 调用对象的方法或函数，支持可变数量的参数
    template<typename... Args>
    object call(Args&&... args);
    // 调用对象作为函数，传入参数为mpy::handle对象
    object call_object(mpy::handle args);
    // 调用对象作为函数，传入参数为mpy::handle对象和关键字参数
    object call_object(mpy::handle args, mpy::handle kwargs);
    // 调用对象作为函数，传入参数为mpy::handle指针数组、参数个数和关键字参数名
    object call_vector(mpy::handle* begin, Py_ssize_t nargs, mpy::handle kwnames);
    // 调用对象作为函数，传入参数为vector_args对象
    object call_vector(vector_args args);
    // 判断两个handle对象是否指向同一个PyObject
    bool operator==(handle rhs) {
        return ptr_ == rhs.ptr_;
    }

    // 静态方法，检查PyObject指针是否有效，无效时抛出异常exception_set
    static handle checked(PyObject* ptr) {
        if (!ptr) {
            throw exception_set();
        }
        return ptr;
    }

protected:
    PyObject* ptr_ = nullptr;   // 指向Python对象的PyObject指针
};


// 模板类，封装Python对象的具体类型
template<typename T>
struct obj;

// 模板类，继承自handle类，封装Python对象的具体类型，并提供方便的指针访问
template<typename T>
struct hdl : public handle {
    // 返回对象的指针，转换为特定类型T的指针
    T* ptr() {
        return  (T*) handle::ptr();
    }
    // 重载箭头运算符，返回对象的指针，转换为特定类型T的指针
    T* operator->() {
        return ptr();
    }
    // 构造函数，使用特定类型T的指针初始化hdl对象
    hdl(T* ptr)
    : hdl((PyObject*) ptr) {}
    // 构造函数，使用obj<T>对象初始化hdl对象
    hdl(const obj<T>& o)
    : hdl(o.ptr()) {}
private:
    // 私有构造函数，使用基类handle对象初始化hdl对象
    hdl(handle h) : handle(h) {}
};

// Python对象的包装类，继承自handle类
struct object : public handle {
    // 默认构造函数，初始化为空对象
    object() = default;
    // 拷贝构造函数，使用另一个object对象初始化当前对象
    object(const object& other)
    : handle(other.ptr_) {
        Py_XINCREF(ptr_);
    }
    // 移动构造函数，使用另一个object对象初始化当前对象，并清空另一个对象的指针
    object(object&& other) noexcept
    : handle(other.ptr_) {
        other.ptr_ = nullptr;
    }
    // 拷贝赋值运算符，使用另一个object对象赋值给当前对象
    object& operator=(const object& other) {
        return *this = object(other);
    }
    // 移动赋值运算符，使用另一个object对象移动赋值给当前对象
    object& operator=(object&& other) noexcept {
        PyObject* tmp = ptr_;
        ptr_ = other.ptr_;
        other.ptr_ = tmp;
        return *this;
    }
    // 析构函数，释放对象持有的Python对象的引用计数
    ~object() {
        Py_XDECREF(ptr_);
    }

    // More member functions can follow here
};
    # 静态方法：从给定的 handle 对象 o 中窃取指针并返回一个新的 object 对象
    static object steal(handle o) {
        return object(o.ptr());
    }

    # 静态方法：检查给定的 handle 对象 o 是否为 nullptr，如果是则抛出异常
    # 否则从中窃取指针并返回一个新的 object 对象
    static object checked_steal(handle o) {
        if (!o.ptr()) {
            throw exception_set();
        }
        return steal(o);
    }

    # 静态方法：增加给定 handle 对象 o 的引用计数，并从中窃取指针返回一个新的 object 对象
    static object borrow(handle o) {
        Py_XINCREF(o.ptr());
        return steal(o);
    }

    # 释放当前对象持有的指针，并将其置为 nullptr，然后返回释放的指针
    PyObject* release() {
        auto tmp = ptr_;
        ptr_ = nullptr;
        return tmp;
    }
protected:
    // 声明一个显式构造函数，接受一个 PyObject 指针作为参数，初始化 handle 成员变量
    explicit object(PyObject* ptr)
    : handle(ptr) {}
};

template<typename T>
struct obj : public object {
    // 默认构造函数
    obj() = default;
    // 拷贝构造函数，从另一个 obj 对象构造，调用基类 object 的拷贝构造函数，并增加引用计数
    obj(const obj& other)
    : object(other.ptr_) {
        Py_XINCREF(ptr_);
    }
    // 移动构造函数，从另一个 obj 对象构造，调用基类 object 的移动构造函数
    obj(obj&& other) noexcept
    : object(other.ptr_) {
        other.ptr_ = nullptr;
    }
    // 拷贝赋值运算符重载，调用拷贝构造函数构造一个新对象并返回
    obj& operator=(const obj& other) {
        return *this = obj(other);
    }
    // 移动赋值运算符重载，交换两个对象的指针值
    obj& operator=(obj&& other) noexcept {
        PyObject* tmp = ptr_;
        ptr_ = other.ptr_;
        other.ptr_ = tmp;
        return *this;
    }
    // 静态方法，从 hdl<T> 对象中窃取指针创建一个新的 obj 对象
    static obj steal(hdl<T> o) {
        return obj(o.ptr());
    }
    // 静态方法，从 hdl<T> 对象中安全地窃取指针创建一个新的 obj 对象，若为空则抛出异常
    static obj checked_steal(hdl<T> o) {
        if (!o.ptr()) {
            throw exception_set();
        }
        return steal(o);
    }
    // 静态方法，从 hdl<T> 对象中借用指针创建一个新的 obj 对象，并增加引用计数
    static obj borrow(hdl<T> o) {
        Py_XINCREF(o.ptr());
        return steal(o);
    }
    // 返回类型为 T* 的指针
    T* ptr() const {
        return (T*) object::ptr();
    }
    // 重载箭头运算符，返回类型为 T* 的指针
    T* operator->() {
        return ptr();
    }
protected:
    // 显式构造函数，接受一个 T* 类型指针作为参数，调用基类 object 的构造函数
    explicit obj(T* ptr)
    : object((PyObject*)ptr) {}
};


static bool isinstance(handle h, handle c) {
    // 检查 handle h 是否是类型为 c 的实例
    return PyObject_IsInstance(h.ptr(), c.ptr());
}

[[ noreturn ]] inline void raise_error(handle exception, const char *format, ...) {
    // 抛出异常并格式化错误消息
    va_list args;
    va_start(args, format);
    PyErr_FormatV(exception.ptr(), format, args);
    va_end(args);
    throw exception_set();
}

template<typename T>
struct base {
    // Python 对象头结构体
    PyObject_HEAD
    // 返回指向当前对象的 PyObject 指针
    PyObject* ptr() const {
        return (PyObject*) this;
    }
    // 分配并初始化一个 T 类型对象的 obj<T> 对象
    static obj<T> alloc(PyTypeObject* type = nullptr) {
        if (!type) {
            type = &T::Type;
        }
        // 分配内存并构造对象
        auto self = (T*) type->tp_alloc(type, 0);
        if (!self) {
            throw mpy::exception_set();
        }
        new (self) T;
        return obj<T>::steal(self);
    }
    // 创建并初始化一个 T 类型对象的 obj<T> 对象，支持任意数量的参数
    template<typename ... Args>
    static obj<T> create(Args ... args) {
        auto self = alloc();
        self->init(std::forward<Args>(args)...);
        return self;
    }
    // 检查给定的 handle 是否是 T 类型对象的实例
    static bool check(handle v) {
        return isinstance(v, (PyObject*)&T::Type);
    }

    // 将 handle 转换为 hdl<T>，不进行类型检查
    static hdl<T> unchecked_wrap(handle self_) {
        return hdl<T>((T*)self_.ptr());
    }
    // 将 handle 转换为 hdl<T>，并进行类型检查，若不是则抛出异常
    static hdl<T> wrap(handle self_) {
        if (!check(self_)) {
            raise_error(PyExc_ValueError, "not an instance of %S", &T::Type);
        }
        return unchecked_wrap(self_);
    }

    // 将 object 转换为 obj<T>，不进行类型检查
    static obj<T> unchecked_wrap(object self_) {
        return obj<T>::steal(unchecked_wrap(self_.release()));
    }
    // 将 object 转换为 obj<T>，并进行类型检查，若不是则抛出异常
    static obj<T> wrap(object self_) {
        return obj<T>::steal(wrap(self_.release()));
    }

    // 创建一个类型为 T 的新对象，用于 Python 类型结构体的 tp_new 函数
    static PyObject* new_stub(PyTypeObject *type, PyObject *args, PyObject *kwds) {
        PY_BEGIN
        return (PyObject*) alloc(type).release();
        PY_END(nullptr)
    }
    // 释放类型为 T 的对象，用于 Python 类型结构体的 tp_dealloc 函数
    static void dealloc_stub(PyObject *self) {
        ((T*)self)->~T();
        Py_TYPE(self)->tp_free(self);
    }
    // 定义静态函数 ready，用于初始化 Python 类型并将其添加到指定的模块中
    static void ready(mpy::handle mod, const char* name) {
        // 准备 Python 类型 T::Type，如果失败则抛出异常
        if (PyType_Ready(&T::Type)) {
            throw exception_set();
        }
        // 将 Python 类型 T::Type 添加到指定的 Python 模块中，如果失败则抛出异常
        if(PyModule_AddObject(mod.ptr(), name, (PyObject*) &T::Type) < 0) {
            throw exception_set();
        }
    }
};

// 获取给定 key 对应的属性对象
inline object handle::attr(const char* key) {
    return object::checked_steal(PyObject_GetAttrString(ptr(), key));
}

// 检查对象是否具有给定 key 的属性
inline bool handle::hasattr(const char* key) {
    return PyObject_HasAttrString(ptr(), key);
}

// 导入指定模块并返回其对象表示
inline object import(const char* module) {
    return object::checked_steal(PyImport_ImportModule(module));
}

// 调用对象的方法，传入可变数量的参数
template<typename... Args>
inline object handle::call(Args&&... args) {
    return object::checked_steal(PyObject_CallFunctionObjArgs(ptr_, args.ptr()..., nullptr));
}

// 调用对象的方法，传入单个参数对象
inline object handle::call_object(mpy::handle args) {
    return object::checked_steal(PyObject_CallObject(ptr(), args.ptr()));
}

// 调用对象的方法，传入参数对象和关键字参数对象
inline object handle::call_object(mpy::handle args, mpy::handle kwargs) {
    return object::checked_steal(PyObject_Call(ptr(), args.ptr(), kwargs.ptr()));
}

// 调用对象的方法，传入参数对象数组和关键字参数名对象
inline object handle::call_vector(mpy::handle* begin, Py_ssize_t nargs, mpy::handle kwnames) {
    return object::checked_steal(PY_VECTORCALL(ptr(), (PyObject*const*) begin, nargs, kwnames.ptr()));
}

// tuple 结构的封装，设置指定位置的元素值
struct tuple : public object {
    void set(int i, object v) {
        PyTuple_SET_ITEM(ptr_, i, v.release());
    }
    tuple(int size)
    : object(checked_steal(PyTuple_New(size))) {}
};

// list 结构的封装，设置指定位置的元素值
struct list : public object {
    void set(int i, object v) {
        PyList_SET_ITEM(ptr_, i, v.release());
    }
    list(int size)
    : object(checked_steal(PyList_New(size))) {}
};

// 以下是命名空间内的函数和结构的实现

namespace {
// 根据格式化字符串和参数列表创建 Unicode 对象
mpy::object unicode_from_format(const char* format, ...) {
    va_list args;
    va_start(args, format);
    auto r = PyUnicode_FromFormatV(format, args);
    va_end(args);
    return mpy::object::checked_steal(r);
}

// 根据字符串创建 Unicode 对象
mpy::object unicode_from_string(const char * str) {
    return mpy::object::checked_steal(PyUnicode_FromString(str));
}

// 根据 Py_ssize_t 类型的整数创建对象
mpy::object from_int(Py_ssize_t s) {
    return mpy::object::checked_steal(PyLong_FromSsize_t(s));
}

// 根据布尔值创建对象
mpy::object from_bool(bool b) {
    return mpy::object::borrow(b ? Py_True : Py_False);
}

// 检查对象是否是序列类型
bool is_sequence(handle h) {
    return PySequence_Check(h.ptr());
}
}

// sequence_view 结构，用于操作序列对象的视图
struct sequence_view : public handle {
    sequence_view(handle h)
    : handle(h) {}

    // 获取序列的长度
    Py_ssize_t size() const {
        auto r = PySequence_Size(ptr());
        if (r == -1 && PyErr_Occurred()) {
            throw mpy::exception_set();
        }
        return r;
    }

    // 返回一个遍历序列的范围
    irange enumerate() const {
        return irange(size());
    }

    // 将给定对象封装为 sequence_view
    static sequence_view wrap(handle h) {
        if (!is_sequence(h)) {
            raise_error(PyExc_ValueError, "expected a sequence");
        }
        return sequence_view(h);
    }

    // 获取序列中指定位置的对象
    mpy::object operator[](Py_ssize_t i) const {
        return mpy::object::checked_steal(PySequence_GetItem(ptr(), i));
    }
};

// 以下是命名空间内的函数实现

namespace {
// 返回对象的字符串表示形式
mpy::object repr(handle h) {
    return mpy::object::checked_steal(PyObject_Repr(h.ptr()));
}

// 返回对象的字符串表示形式
mpy::object str(handle h) {
    return mpy::object::checked_steal(PyObject_Str(h.ptr()));
}

// 检查对象是否是整数类型
bool is_int(handle h) {
    return PyLong_Check(h.ptr());
}

// 检查对象是否为 None
bool is_none(handle h) {
    return h.ptr() == Py_None;
}

// 检查对象是否是布尔类型
bool is_bool(handle h) {
    # 调用 CPython C API 的 PyBool_Check 函数，检查 h.ptr() 返回的对象是否是布尔类型
    return PyBool_Check(h.ptr());
}

// 将 Python 对象转换为 Py_ssize_t 类型整数
Py_ssize_t to_int(handle h) {
    Py_ssize_t r = PyLong_AsSsize_t(h.ptr());
    // 检查转换是否失败并且是否设置了异常，如果是，则抛出异常
    if (r == -1 && PyErr_Occurred()) {
        throw mpy::exception_set();
    }
    return r;
}

// 将 Python 对象转换为 double 类型浮点数
double to_float(handle h) {
    double r = PyFloat_AsDouble(h.ptr());
    // 检查是否发生了异常，如果是，则抛出异常
    if (PyErr_Occurred()) {
        throw mpy::exception_set();
    }
    return r;
}

// 快速地将 Python 对象转换为布尔类型，不进行异常检查
bool to_bool_unsafe(handle h) {
    return h.ptr() == Py_True;
}

// 将 Python 对象转换为布尔类型，进行异常检查
bool to_bool(handle h) {
    return PyObject_IsTrue(h.ptr()) != 0;
}
}

// 表示 Python 的切片视图
struct slice_view {
    slice_view(handle h, Py_ssize_t size)  {
        // 解析 Python 的切片对象，如果失败则抛出异常
        if(PySlice_Unpack(h.ptr(), &start, &stop, &step) == -1) {
            throw mpy::exception_set();
        }
        // 调整切片的索引范围，以确保不超出边界
        slicelength = PySlice_AdjustIndices(size, &start, &stop, step);
    }
    Py_ssize_t start, stop, step, slicelength;
};

// 检查给定的 Python 对象是否为切片对象
static bool is_slice(handle h) {
    return PySlice_Check(h.ptr());
}

// 重载输出流操作符，将 Python 对象转换为 UTF-8 编码的字符串并输出
inline std::ostream& operator<<(std::ostream& ss, handle h) {
    ss << PyUnicode_AsUTF8(str(h).ptr());
    return ss;
}

// 表示 Python 的元组视图
struct tuple_view : public handle {
    tuple_view() = default;
    tuple_view(handle h) : handle(h) {}

    // 返回元组的大小
    Py_ssize_t size() const {
        return PyTuple_GET_SIZE(ptr());
    }

    // 返回一个范围对象，用于遍历元组的索引
    irange enumerate() const {
        return irange(size());
    }

    // 获取元组中指定索引位置的元素
    handle operator[](Py_ssize_t i) {
        return PyTuple_GET_ITEM(ptr(), i);
    }

    // 检查给定的 Python 对象是否为元组
    static bool check(handle h) {
        return PyTuple_Check(h.ptr());
    }
};

// 表示 Python 的列表视图
struct list_view : public handle {
    list_view() = default;
    list_view(handle h) : handle(h) {}

    // 返回列表的大小
    Py_ssize_t size() const {
        return PyList_GET_SIZE(ptr());
    }

    // 返回一个范围对象，用于遍历列表的索引
    irange enumerate() const {
        return irange(size());
    }

    // 获取列表中指定索引位置的元素
    handle operator[](Py_ssize_t i) {
        return PyList_GET_ITEM(ptr(), i);
    }

    // 检查给定的 Python 对象是否为列表
    static bool check(handle h) {
        return PyList_Check(h.ptr());
    }
};

// 表示 Python 的字典视图
struct dict_view : public handle {
    dict_view() = default;
    dict_view(handle h) : handle(h) {}

    // 返回字典的键视图对象
    object keys() const {
        return mpy::object::checked_steal(PyDict_Keys(ptr()));
    }

    // 返回字典的值视图对象
    object values() const {
        return mpy::object::checked_steal(PyDict_Values(ptr()));
    }

    // 返回字典的键值对视图对象
    object items() const {
        return mpy::object::checked_steal(PyDict_Items(ptr()));
    }

    // 检查字典中是否包含指定的键
    bool contains(handle k) const {
        return PyDict_Contains(ptr(), k.ptr());
    }

    // 获取字典中指定键对应的值
    handle operator[](handle k) {
        return mpy::handle::checked(PyDict_GetItem(ptr(), k.ptr()));
    }

    // 检查给定的 Python 对象是否为字典
    static bool check(handle h) {
        return PyDict_Check(h.ptr());
    }

    // 在字典中迭代下一个键值对，并返回是否还有更多元素
    bool next(Py_ssize_t* pos, mpy::handle* key, mpy::handle* value) {
        PyObject *k = nullptr, *v = nullptr;
        auto r = PyDict_Next(ptr(), pos, &k, &v);
        *key = k;
        *value = v;
        return r;
    }

    // 设置字典中指定键对应的值
    void set(handle k, handle v) {
        if (-1 == PyDict_SetItem(ptr(), k.ptr(), v.ptr())) {
            throw exception_set();
        }
    }
};


// 表示 Python 的关键字参数名称视图
struct kwnames_view : public handle {
    kwnames_view() = default;
    kwnames_view(handle h) : handle(h) {}
    # 返回元组对象的大小，即元素数量
    Py_ssize_t size() const {
        return PyTuple_GET_SIZE(ptr());
    }

    # 返回一个范围对象，用于枚举当前元组的索引范围
    irange enumerate() const {
        return irange(size());
    }

    # 获取元组中指定索引位置的元素，并将其转换为 UTF-8 格式的 C 字符串返回
    const char* operator[](Py_ssize_t i) const {
        PyObject* obj = PyTuple_GET_ITEM(ptr(), i);
        return PyUnicode_AsUTF8(obj);
    }

    # 检查给定的 Python 对象句柄是否为元组类型
    static bool check(handle h) {
        return PyTuple_Check(h.ptr());
    }
    // 结构体结束
};

// 定义一个内联函数 funcname，接受一个 mpy::handle 参数 func
inline mpy::object funcname(mpy::handle func) {
    // 如果 func 具有属性 "__name__"
    if (func.hasattr("__name__")) {
        // 返回 func 的 "__name__" 属性
        return func.attr("__name__");
    } else {
        // 否则返回 func 的字符串表示形式
        return mpy::str(func);
    }
}

// 定义一个结构体 vector_args
struct vector_args {
    // 构造函数接受 PyObject 类型的指针数组 a、数组长度 n、关键字参数 k
    vector_args(PyObject *const *a,
                      Py_ssize_t n,
                      PyObject *k)
    : vector_args((mpy::handle*)a, n, k) {}
    // 构造函数接受 mpy::handle 类型的指针数组 a、数组长度 n、mpy::handle 类型的关键字参数 k
    vector_args(mpy::handle* a,
                    Py_ssize_t n,
                    mpy::handle k)
    : args((mpy::handle*)a), nargs(n), kwnames(k) {}
    // 成员变量，指向 mpy::handle 类型的指针数组 args，参数个数 nargs，关键字参数 kwnames

    // 返回 args 数组的起始地址
    mpy::handle* begin() {
        return args;
    }
    // 返回 args 数组的末尾地址
    mpy::handle* end() {
        return args + size();
    }

    // 返回 args 数组中索引为 i 的元素
    mpy::handle operator[](int64_t i) const {
        return args[i];
    }
    // 检查是否有关键字参数
    bool has_keywords() const {
        return kwnames.ptr();
    }
    // 返回位置参数的范围
    irange enumerate_positional() {
        return irange(nargs);
    }
    // 返回所有参数的范围
    irange enumerate_all() {
        return irange(size());
    }
    // 返回参数的总数，包括关键字参数
    int64_t size() const {
        return nargs + (has_keywords() ? kwnames.size() : 0);
    }

    // 以下是一系列方法的声明，用于参数解析，未提供具体实现

    // 绑定一个测试函数以便测试，前两个参数为必需参数和仅关键字参数，然后返回解析的内容...

    // 提供写关键字参数的能力
    // 不提供必需参数
    // 不提供可选参数
    // 提供一个已经提供的位置参数的关键字参数
    // 将仅关键字参数作为位置参数提供
    // 以错误的顺序提供关键字参数
    // 仅提供关键字参数

    // 解析方法，接受一个 C 字符串 fname_cstr、名称列表 names、值列表 values、必需参数 required、可选参数 kwonly
    void parse(const char * fname_cstr, std::initializer_list<const char*> names, std::initializer_list<mpy::handle*> values, int required, int kwonly=0) {
        auto error = [&]() {
            // 一旦检测到错误，使用较慢的基础设施格式化和抛出错误消息

            // 必须泄漏这些内存，因为 Python 期望它们保持有效
            const char** names_buf = new const char*[names.size() + 1];
            std::copy(names.begin(), names.end(), &names_buf[0]);
            names_buf[names.size()] = nullptr;

            // 根据 Python 版本选择格式字符串的创建方式
#if PY_VERSION_HEX < 0x03080000
            char* format_str = new char[names.size() + 3];
            int i = 0;
            char* format_it = format_str;
            // 根据参数名称设置格式字符串，用 '|' 标记必需参数位置，用 '$' 标记仅关键字参数位置
            for (auto it = names.begin(); it != names.end(); ++it, ++i) {
                if (i == required) {
                    *format_it++ = '|';
                }
                if (i == (int)names.size() - kwonly) {
                    *format_it++ = '$';
                }
                *format_it++ = 'O';
            }
            *format_it++ = '\0';
            // 创建 _PyArg_Parser 对象，并进行参数解析
            _PyArg_Parser* _parser = new _PyArg_Parser{format_str, &names_buf[0], fname_cstr, 0};
            PyObject *dummy = NULL;
            _PyArg_ParseStackAndKeywords((PyObject*const*)args, nargs, kwnames.ptr(), _parser, &dummy, &dummy, &dummy, &dummy, &dummy);
#else
    // 如果未定义，则执行以下代码块
    _PyArg_Parser* _parser = new _PyArg_Parser{NULL, &names_buf[0], fname_cstr, 0};
    // 创建 _PyArg_Parser 对象，用于解析参数，初始化为空指针和其他参数
    std::unique_ptr<PyObject*[]> buf(new PyObject*[names.size()]);
    // 创建包含 PyObject* 数组的 unique_ptr，用于存储参数值
    _PyArg_UnpackKeywords((PyObject*const*)args, nargs, NULL, kwnames.ptr(), _parser, required, (Py_ssize_t)values.size() - kwonly, 0, &buf[0]);
#endif
    // 使用 _PyArg_UnpackKeywords 解析关键字参数，抛出异常集合
    throw exception_set();
};

auto values_it = values.begin();
auto names_it = names.begin();
auto npositional = values.size() - kwonly;

if (nargs > (Py_ssize_t)npositional) {
    // 如果传入的参数个数超过了可接受的位置参数个数
    error();
}
for (auto i : irange(nargs)) {
    *(*values_it++) = args[i];
    ++names_it;
}

if (!kwnames.ptr()) {
    if (nargs < required) {
        // 如果传入的位置参数个数少于要求的个数
        error();
    }
} else {
    int consumed = 0;
    for (auto i : irange(nargs, values.size())) {
        bool success = i >= required;
        const char* target_name = *(names_it++);
        for (auto j : kwnames.enumerate()) {
            if (!strcmp(target_name,kwnames[j])) {
                *(*values_it) = args[nargs + j];
                ++consumed;
                success = true;
                break;
            }
        }
        ++values_it;
        if (!success) {
            // 如果需要的参数未指定
            error();
        }
    }
    if (consumed != kwnames.size()) {
        // 如果未使用所有的关键字参数
        error();
    }
}
}
int index(const char* name, int pos) {
    if (pos < nargs) {
        return pos;
    }
    if (kwnames.ptr()) {
        for (auto j : kwnames.enumerate()) {
            if (!strcmp(name, kwnames[j])) {
                return nargs + j;
            }
        }
    }
    return -1;
}
};

inline object handle::call_vector(vector_args args) {
    return object::checked_steal(PY_VECTORCALL(ptr(), (PyObject*const*) args.args, args.nargs, args.kwnames.ptr()));
}
}
// 返回从 PyObject 到 PyObject 检查的 object
#define MPY_ARGS_NAME(typ, name) #name ,
#define MPY_ARGS_DECLARE(typ, name) typ name;
#define MPY_ARGS_POINTER(typ, name) &name ,
#define MPY_PARSE_ARGS_KWARGS(fmt, FORALL_ARGS) \
    // 定义关键字参数列表
    static char* kwlist[] = { FORALL_ARGS(MPY_ARGS_NAME) nullptr}; \
    FORALL_ARGS(MPY_ARGS_DECLARE) \
    // 解析参数和关键字
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, fmt, kwlist, FORALL_ARGS(MPY_ARGS_POINTER) nullptr)) { \
        throw mpy::exception_set(); \
    }

#define MPY_PARSE_ARGS_KWNAMES(fmt, FORALL_ARGS) \
    // 定义关键字名称列表
    static const char * const kwlist[] = { FORALL_ARGS(MPY_ARGS_NAME) nullptr}; \
    FORALL_ARGS(MPY_ARGS_DECLARE) \
    static _PyArg_Parser parser = {fmt, kwlist, 0}; \
    # 使用宏 `_PyArg_ParseStackAndKeywords` 解析 Python C API 中的函数参数，并根据解析结果执行相应的操作
    if (!_PyArg_ParseStackAndKeywords(args, nargs, kwnames, &parser, FORALL_ARGS(MPY_ARGS_POINTER) nullptr)) {
        # 如果参数解析失败，则抛出一个自定义的异常，用于表示解析失败的情况
        throw mpy::exception_set();
    }
```