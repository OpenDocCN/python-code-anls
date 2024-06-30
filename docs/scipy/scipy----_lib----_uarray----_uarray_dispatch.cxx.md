# `D:\src\scipysrc\scipy\scipy\_lib\_uarray\_uarray_dispatch.cxx`

```
#include <Python.h>
// 引入 Python C API 的头文件

#include "small_dynamic_array.h"
// 引入自定义的 small_dynamic_array 头文件

#include "vectorcall.h"
// 引入自定义的 vectorcall 头文件

#include <algorithm>
// 引入 STL 中的算法库，例如 std::swap

#include <cstddef>
// 引入标准库中的 cstddef，提供了一些与 C++ 标准库相关的宏和定义

#include <new>
// 引入标准库中的 new，提供了内存分配和释放操作符

#include <stdexcept>
// 引入标准异常库，提供了一些标准异常类

#include <string>
// 引入标准字符串库，提供了 string 类和一些字符串操作函数

#include <unordered_map>
// 引入标准无序映射库，提供了 unordered_map 类

#include <utility>
// 引入标准实用工具库，提供了一些通用的实用工具函数

#include <vector>
// 引入标准向量库，提供了 vector 类和一些向量操作函数

namespace {
// 匿名命名空间，用于限制定义的作用域，避免全局命名冲突

/** Handle to a python object that automatically DECREFs */
// 自动减少引用计数的 Python 对象句柄类

class py_ref {
  explicit py_ref(PyObject * object): obj_(object) {}

public:
  py_ref() noexcept: obj_(nullptr) {}
  // 默认构造函数，初始化为 nullptr

  py_ref(std::nullptr_t) noexcept: py_ref() {}
  // 接受 nullptr 的构造函数，调用默认构造函数

  py_ref(const py_ref & other) noexcept: obj_(other.obj_) { Py_XINCREF(obj_); }
  // 拷贝构造函数，复制对象并增加引用计数

  py_ref(py_ref && other) noexcept: obj_(other.obj_) { other.obj_ = nullptr; }
  // 移动构造函数，接受右值引用并置空原对象

  /** Construct from new reference (No INCREF) */
  // 从新引用构造（不增加引用计数）

  static py_ref steal(PyObject * object) { return py_ref(object); }
  // 使用给定的对象构造一个 py_ref 对象，不增加引用计数

  /** Construct from borrowed reference (and INCREF) */
  // 从借用的引用构造（增加引用计数）

  static py_ref ref(PyObject * object) {
    Py_XINCREF(object);
    return py_ref(object);
  }
  // 使用给定的对象构造一个 py_ref 对象，并增加引用计数

  ~py_ref() { Py_XDECREF(obj_); }
  // 析构函数，减少引用计数

  py_ref & operator=(const py_ref & other) noexcept {
    py_ref(other).swap(*this);
    return *this;
  }
  // 拷贝赋值运算符，交换内容

  py_ref & operator=(py_ref && other) noexcept {
    py_ref(std::move(other)).swap(*this);
    return *this;
  }
  // 移动赋值运算符，交换内容

  friend bool operator==(const py_ref & lhs, const py_ref & rhs) {
    return lhs.obj_ == rhs.obj_;
  }
  // 比较相等运算符，比较两个 py_ref 对象的指针是否相等

  friend bool operator==(PyObject * lhs, const py_ref & rhs) {
    return lhs == rhs.obj_;
  }
  // 比较相等运算符，比较 PyObject 指针和 py_ref 对象的指针是否相等

  friend bool operator==(const py_ref & lhs, PyObject * rhs) {
    return lhs.obj_ == rhs;
  }
  // 比较相等运算符，比较 py_ref 对象的指针和 PyObject 指针是否相等

  friend bool operator!=(const py_ref & lhs, const py_ref & rhs) {
    return lhs.obj_ != rhs.obj_;
  }
  // 比较不等运算符，比较两个 py_ref 对象的指针是否不相等

  friend bool operator!=(PyObject * lhs, const py_ref & rhs) {
    return lhs != rhs.obj_;
  }
  // 比较不等运算符，比较 PyObject 指针和 py_ref 对象的指针是否不相等

  friend bool operator!=(const py_ref & lhs, PyObject * rhs) {
    return lhs.obj_ != rhs;
  }
  // 比较不等运算符，比较 py_ref 对象的指针和 PyObject 指针是否不相等

  void swap(py_ref & other) noexcept { std::swap(other.obj_, obj_); }
  // 交换内容，用于实现赋值运算符

  explicit operator bool() const { return obj_ != nullptr; }
  // 显式的 bool 转换运算符，判断是否为空指针

  PyObject * get() const { return obj_; }
  // 获取内部保存的 PyObject 指针

  PyObject * release() {
    PyObject * t = obj_;
    obj_ = nullptr;
    return t;
  }
  // 释放对象的所有权，返回 PyObject 指针并置空 obj_

  void reset() { Py_CLEAR(obj_); }
  // 重置对象，减少引用计数并置空 obj_

private:
  PyObject * obj_;
  // 私有成员变量，保存 PyObject 指针
};

PyObject * py_get(const py_ref & ref) { return ref.get(); }
// 获取 py_ref 对象的 PyObject 指针

PyObject * py_get(PyObject * obj) { return obj; }
// 返回传入的 PyObject 指针

/** Make tuple from variadic set of PyObjects */
// 从可变参数的 PyObject 集合创建元组

template <typename... Ts>
py_ref py_make_tuple(const Ts &... args) {
  return py_ref::steal(PyTuple_Pack(sizeof...(args), py_get(args)...));
}
// 使用 PyTuple_Pack 创建包含所有参数的元组，并返回 py_ref 对象

py_ref py_bool(bool input) { return py_ref::ref(input ? Py_True : Py_False); }
// 创建一个布尔类型的 py_ref 对象

template <typename T, size_t N>
constexpr size_t array_size(const T (&array)[N]) {
  return N;
}
// 返回数组的大小，模板函数

struct backend_options {
  py_ref backend;
  bool coerce = false;
  bool only = false;

  bool operator==(const backend_options & other) const {
    return (
        backend == other.backend && coerce == other.coerce &&
        only == other.only);
  }
  // 比较相等运算符，比较两个 backend_options 结构体是否相等

  bool operator!=(const backend_options & other) const {
    return !(*this == other);
  }
  // 比较不等运算符，比较两个 backend_options 结构体是否不相等
};
// 后端选项结构体，包含一个 py_ref 对象和两个布尔值
// 定义全局后端选项结构体，包含全局选项和已注册后端列表
struct global_backends {
  backend_options global;  // 全局选项
  std::vector<py_ref> registered;  // 已注册的后端列表
  bool try_global_backend_last = false;  // 是否尝试将全局后端放置在最后
};

// 定义本地后端结构体，包含被跳过的后端列表和首选的后端列表
struct local_backends {
  std::vector<py_ref> skipped;  // 被跳过的后端列表
  std::vector<backend_options> preferred;  // 首选的后端列表
};

// 使用哈希映射实现全局状态类型，映射域名到全局后端结构体
using global_state_t = std::unordered_map<std::string, global_backends>;

// 使用哈希映射实现本地状态类型，映射域名到本地后端结构体
using local_state_t = std::unordered_map<std::string, local_backends>;

// 静态变量，用于引用未实现的后端错误
static py_ref BackendNotImplementedError;

// 全局域名映射，线程局部指针指向全局域名映射
thread_local global_state_t * current_global_state = &global_domain_map;

// 线程局部变量，用于存储线程本地域名映射
thread_local global_state_t thread_local_domain_map;
thread_local local_state_t local_domain_map;

/** Constant Python string identifiers

Using these with PyObject_GetAttr is faster than PyObject_GetAttrString which
has to create a new python string internally.
 */
struct {
  py_ref ua_convert;  // Python 字符串标识符 __ua_convert__
  py_ref ua_domain;   // Python 字符串标识符 __ua_domain__
  py_ref ua_function; // Python 字符串标识符 __ua_function__

  // 初始化函数，为每个标识符分配 Python 字符串对象
  bool init() {
    ua_convert = py_ref::steal(PyUnicode_InternFromString("__ua_convert__"));
    if (!ua_convert)
      return false;

    ua_domain = py_ref::steal(PyUnicode_InternFromString("__ua_domain__"));
    if (!ua_domain)
      return false;

    ua_function = py_ref::steal(PyUnicode_InternFromString("__ua_function__"));
    if (!ua_function)
      return false;

    return true;
  }

  // 清理函数，重置所有标识符对象
  void clear() {
    ua_convert.reset();
    ua_domain.reset();
    ua_function.reset();
  }
} identifiers;

// 验证域名对象是否为有效的 Python 字符串
bool domain_validate(PyObject * domain) {
  if (!PyUnicode_Check(domain)) {
    PyErr_SetString(PyExc_TypeError, "__ua_domain__ must be a string");
    return false;
  }

  auto size = PyUnicode_GetLength(domain);
  if (size == 0) {
    PyErr_SetString(PyExc_ValueError, "__ua_domain__ must be non-empty");
    return false;
  }

  return true;
}

// 将域名对象转换为 C++ 字符串
std::string domain_to_string(PyObject * domain) {
  if (!domain_validate(domain)) {
    return {};
  }

  Py_ssize_t size;
  const char * str = PyUnicode_AsUTF8AndSize(domain, &size);
  if (!str)
    return {};

  if (size == 0) {
    PyErr_SetString(PyExc_ValueError, "__ua_domain__ must be non-empty");
    return {};
  }

  return std::string(str, size);
}

// 获取后端对象所包含的域名数量
Py_ssize_t backend_get_num_domains(PyObject * backend) {
  auto domain =
      py_ref::steal(PyObject_GetAttr(backend, identifiers.ua_domain.get()));
  if (!domain)
    return -1;

  if (PyUnicode_Check(domain.get())) {
    return 1;
  }

  if (!PySequence_Check(domain.get())) {
    PyErr_SetString(
        PyExc_TypeError,
        "__ua_domain__ must be a string or sequence of strings");
    return -1;
  }

  return PySequence_Size(domain.get());
}

// 枚举类型，用于指示循环操作的返回状态
enum class LoopReturn { Continue, Break, Error };

// 对每个后端对象中的域名执行指定函数
template <typename Func>
LoopReturn backend_for_each_domain(PyObject * backend, Func f) {
  auto domain =
      py_ref::steal(PyObject_GetAttr(backend, identifiers.ua_domain.get()));
  if (!domain)
    return LoopReturn::Error;

  if (PyUnicode_Check(domain.get())) {
    return f(domain.get());
  }

  if (!PySequence_Check(domain.get())) {
    # 设置一个类型错误异常，并指定错误信息字符串
    PyErr_SetString(
        PyExc_TypeError,
        "__ua_domain__ must be a string or sequence of strings");
    # 返回循环返回类型为错误
    return LoopReturn::Error;
  }

  # 获取序列对象的大小
  auto size = PySequence_Size(domain.get());
  # 如果获取大小失败，返回循环返回类型为错误
  if (size < 0)
    return LoopReturn::Error;
  # 如果序列大小为0，设置数值错误异常并返回循环返回类型为错误
  if (size == 0) {
    PyErr_SetString(PyExc_ValueError, "__ua_domain__ lists must be non-empty");
    return LoopReturn::Error;
  }

  # 遍历序列对象中的每一项
  for (Py_ssize_t i = 0; i < size; ++i) {
    # 从序列中获取第i项，并创建一个Python对象的引用
    auto dom = py_ref::steal(PySequence_GetItem(domain.get(), i));
    # 如果获取失败，返回循环返回类型为错误
    if (!dom)
      return LoopReturn::Error;

    # 调用函数f，传入dom作为参数，并获取返回值
    auto res = f(dom.get());
    # 如果返回值不等于Continue，直接返回该返回值
    if (res != LoopReturn::Continue) {
      return res;
    }
  }
  # 循环正常结束，返回循环返回类型为Continue
  return LoopReturn::Continue;
}

// 根据给定的后端对象和处理函数，对每个域名执行处理，并返回处理结果
template <typename Func>
LoopReturn backend_for_each_domain_string(PyObject * backend, Func f) {
  // 使用 lambda 表达式实现对每个域名的处理
  return backend_for_each_domain(backend, [&](PyObject * domain) {
    // 将域名转换为字符串
    auto domain_string = domain_to_string(domain);
    // 如果域名字符串为空，则返回错误
    if (domain_string.empty()) {
      return LoopReturn::Error;
    }
    // 否则，调用处理函数处理域名字符串
    return f(domain_string);
  });
}

// 验证给定后端对象中的所有域名是否有效
bool backend_validate_ua_domain(PyObject * backend) {
  // 调用 backend_for_each_domain 函数对每个域名执行验证
  const auto res = backend_for_each_domain(backend, [&](PyObject * domain) {
    // 如果域名验证有效，则返回 LoopReturn::Continue；否则返回 LoopReturn::Error
    return domain_validate(domain) ? LoopReturn::Continue : LoopReturn::Error;
  });
  // 如果结果不是 Error，则返回 true，否则返回 false
  return (res != LoopReturn::Error);
}

// 后端状态结构体
struct BackendState {
  PyObject_HEAD
  global_state_t globals;  // 全局状态
  local_state_t locals;    // 局部状态
  bool use_thread_local_globals = true;  // 是否使用线程局部全局状态

  // 析构函数，释放 BackendState 对象
  static void dealloc(BackendState * self) {
    self->~BackendState();
    Py_TYPE(self)->tp_free(self);
  }

  // 创建新的 BackendState 对象
  static PyObject * new_(
      PyTypeObject * type, PyObject * args, PyObject * kwargs) {
    auto self = reinterpret_cast<BackendState *>(type->tp_alloc(type, 0));
    if (self == nullptr)
      return nullptr;

    // 在已分配的内存上构造 BackendState 对象
    self = new (self) BackendState;
    return reinterpret_cast<PyObject *>(self);
  }

  // 将 BackendState 对象序列化为 Python 对象
  static PyObject * pickle_(BackendState * self) {
    try {
      // 转换全局状态为 Python 对象
      py_ref py_global = BackendState::convert_py(self->globals);
      // 转换局部状态为 Python 对象
      py_ref py_locals = BackendState::convert_py(self->locals);
      // 转换是否使用线程局部全局状态为 Python 对象
      py_ref py_use_local_globals =
          BackendState::convert_py(self->use_thread_local_globals);

      // 将转换后的对象打包为元组并返回
      return py_make_tuple(py_global, py_locals, py_use_local_globals)
          .release();
    } catch (std::runtime_error &) {
      return nullptr;
    }
  }

  // 从 Python 对象反序列化创建 BackendState 对象
  static PyObject * unpickle_(PyObject * cls, PyObject * args) {
    try {
      PyObject *py_locals, *py_global;
      // 从参数中解析 Python 对象
      py_ref ref =
          py_ref::steal(Q_PyObject_Vectorcall(cls, nullptr, 0, nullptr));
      BackendState * output = reinterpret_cast<BackendState *>(ref.get());
      if (output == nullptr)
        return nullptr;

      int use_thread_local_globals;
      // 解析参数元组并获取局部状态、全局状态以及线程局部全局状态标志
      if (!PyArg_ParseTuple(
              args, "OOp", &py_global, &py_locals, &use_thread_local_globals))
        return nullptr;
      // 转换局部状态为 C++ 类型
      local_state_t locals = convert_local_state(py_locals);
      // 转换全局状态为 C++ 类型
      global_state_t globals = convert_global_state(py_global);

      // 将解析的值设置到输出的 BackendState 对象中
      output->locals = std::move(locals);
      output->globals = std::move(globals);
      output->use_thread_local_globals = use_thread_local_globals;

      // 返回已构造的 BackendState 对象
      return ref.release();
    } catch (std::invalid_argument &) {
      return nullptr;
    } catch (std::bad_alloc &) {
      PyErr_NoMemory();
      return nullptr;
    }
  }

  // 将 Python 迭代器中的对象转换为指定类型的向量
  template <typename T, typename Convertor>
  static std::vector<T> convert_iter(
      PyObject * input, Convertor item_convertor) {
    std::vector<T> output;
    // 获取 Python 迭代器对象
    py_ref iterator = py_ref::steal(PyObject_GetIter(input));
    if (!iterator)
      throw std::invalid_argument("");

    py_ref item;
    // 迭代获取每个对象，并转换为指定类型后加入到输出向量中
    while ((item = py_ref::steal(PyIter_Next(iterator.get())))) {
      output.push_back(item_convertor(item.get()));
    }
``
    if (PyErr_Occurred())
      throw std::invalid_argument("");

    return output;
  }



  template <
      typename K, typename V, typename KeyConvertor, typename ValueConvertor>
  static std::unordered_map<K, V> convert_dict(
      PyObject * input, KeyConvertor key_convertor,
      ValueConvertor value_convertor) {
    std::unordered_map<K, V> output;

    if (!PyDict_Check(input))
      throw std::invalid_argument("");

    PyObject *key, *value;
    Py_ssize_t pos = 0;

    // 遍历 Python 字典对象，将键值转换后存入 C++ 的无序映射中
    while (PyDict_Next(input, &pos, &key, &value)) {
      output[key_convertor(key)] = value_convertor(value);
    }

    if (PyErr_Occurred())
      throw std::invalid_argument("");

    return output;
  }



  static std::string convert_domain(PyObject * input) {
    // 调用 domain_to_string 函数将 Python 对象转换为字符串表示形式
    std::string output = domain_to_string(input);
    if (output.empty())
      throw std::invalid_argument("");

    return output;
  }



  static backend_options convert_backend_options(PyObject * input) {
    backend_options output;
    int coerce, only;
    PyObject * py_backend;
    
    // 解析输入参数为元组，并读取其中的对象和整数
    if (!PyArg_ParseTuple(input, "Opp", &py_backend, &coerce, &only))
      throw std::invalid_argument("");

    if (py_backend != Py_None) {
      // 如果 py_backend 不为 None，则使用 py_ref::ref 对象来管理其生命周期
      output.backend = py_ref::ref(py_backend);
    }
    output.coerce = coerce;
    output.only = only;

    return output;
  }



  static py_ref convert_backend(PyObject * input) { return py_ref::ref(input); }



  static local_backends convert_local_backends(PyObject * input) {
    PyObject *py_skipped, *py_preferred;
    
    // 解析输入参数为元组，并读取其中的两个对象
    if (!PyArg_ParseTuple(input, "OO", &py_skipped, &py_preferred))
      throw std::invalid_argument("");

    local_backends output;
    // 将 py_skipped 和 py_preferred 转换为相应的 C++ 结构体类型
    output.skipped =
        convert_iter<py_ref>(py_skipped, BackendState::convert_backend);
    output.preferred = convert_iter<backend_options>(
        py_preferred, BackendState::convert_backend_options);

    return output;
  }



  static global_backends convert_global_backends(PyObject * input) {
    PyObject *py_global, *py_registered;
    int try_global_backend_last;
    
    // 解析输入参数为元组，并读取其中的对象和整数
    if (!PyArg_ParseTuple(
            input, "OOp", &py_global, &py_registered, &try_global_backend_last))
      throw std::invalid_argument("");

    global_backends output;
    // 转换全局后端选项和注册后端的迭代对象为 C++ 结构体
    output.global = BackendState::convert_backend_options(py_global);
    output.registered =
        convert_iter<py_ref>(py_registered, BackendState::convert_backend);
    output.try_global_backend_last = try_global_backend_last;

    return output;
  }



  static global_state_t convert_global_state(PyObject * input) {
    // 调用 convert_dict 函数将 Python 字典转换为 global_state_t 类型
    return convert_dict<std::string, global_backends>(
        input, BackendState::convert_domain,
        BackendState::convert_global_backends);
  }



  static local_state_t convert_local_state(PyObject * input) {
  // 将输入的键值对映射转换为 Python 字典类型并返回
  return convert_dict<std::string, local_backends>(
      input, BackendState::convert_domain,
      BackendState::convert_local_backends);
}

// 将 Python 对象转换为相同类型的 Python 对象，直接返回输入对象
static py_ref convert_py(py_ref input) { return input; }

// 将布尔类型输入转换为相应的 Python 布尔对象并返回
static py_ref convert_py(bool input) { return py_bool(input); }

// 将 backend_options 结构体输入转换为包含三个元素的 Python 元组对象并返回
static py_ref convert_py(backend_options input) {
  if (!input.backend) {
    input.backend = py_ref::ref(Py_None);
  }
  py_ref output = py_make_tuple(
      input.backend, py_bool(input.coerce), py_bool(input.only));
  if (!output)
    throw std::runtime_error("");
  return output;
}

// 将 std::string 类型输入转换为 Python Unicode 字符串对象并返回
static py_ref convert_py(const std::string & input) {
  py_ref output =
      py_ref::steal(PyUnicode_FromStringAndSize(input.c_str(), input.size()));
  if (!output)
    throw std::runtime_error("");
  return output;
}

// 将 std::vector<T> 类型输入转换为 Python 列表对象并返回
template <typename T>
static py_ref convert_py(const std::vector<T> & input) {
  py_ref output = py_ref::steal(PyList_New(input.size()));

  if (!output)
    throw std::runtime_error("");

  for (size_t i = 0; i < input.size(); i++) {
    py_ref element = convert_py(input[i]);
    PyList_SET_ITEM(output.get(), i, element.release());
  }

  return output;
}

// 将 local_backends 结构体输入转换为包含两个元素的 Python 元组对象并返回
static py_ref convert_py(const local_backends & input) {
  py_ref py_skipped = BackendState::convert_py(input.skipped);
  py_ref py_preferred = BackendState::convert_py(input.preferred);
  py_ref output = py_make_tuple(py_skipped, py_preferred);

  if (!output)
    throw std::runtime_error("");

  return output;
}

// 将 global_backends 结构体输入转换为包含三个元素的 Python 元组对象并返回
static py_ref convert_py(const global_backends & input) {
  py_ref py_globals = BackendState::convert_py(input.global);
  py_ref py_registered = BackendState::convert_py(input.registered);
  py_ref output = py_make_tuple(
      py_globals, py_registered, py_bool(input.try_global_backend_last));

  if (!output)
    throw std::runtime_error("");

  return output;
}

// 将 std::unordered_map<K, V> 类型输入转换为 Python 字典对象并返回
template <typename K, typename V>
static py_ref convert_py(const std::unordered_map<K, V> & input) {
  py_ref output = py_ref::steal(PyDict_New());

  if (!output)
    throw std::runtime_error("");

  for (const auto & kv : input) {
    py_ref py_key = convert_py(kv.first);
    py_ref py_value = convert_py(kv.second);

    if (PyDict_SetItem(output.get(), py_key.get(), py_value.get()) < 0) {
      throw std::runtime_error("");
    }
  }

  return output;
}
/** Clean up global python references when the module is finalized. */
void globals_free(void * /* self */) {
  // 清空全局域映射中的内容
  global_domain_map.clear();
  // 重置 BackendNotImplementedError 对象
  BackendNotImplementedError.reset();
  // 清空 identifiers 容器
  identifiers.clear();
}

/** Allow GC to break reference cycles between the module and global backends.
 *
 * NOTE: local state and "thread local globals" can't be visited because we
 * can't access locals from another thread. However, since those are only
 * set through context managers they should always be unset before module
 * cleanup.
 */
int globals_traverse(PyObject * self, visitproc visit, void * arg) {
  // 遍历 global_domain_map 中的每个键值对
  for (const auto & kv : global_domain_map) {
    const auto & globals = kv.second;
    // 访问全局后端对象，并处理循环引用
    PyObject * backend = globals.global.backend.get();
    Py_VISIT(backend);
    // 遍历已注册的后端对象列表，并处理循环引用
    for (const auto & reg : globals.registered) {
      backend = reg.get();
      Py_VISIT(backend);
    }
  }
  return 0;
}

int globals_clear(PyObject * /* self */) {
  // 清空 global_domain_map 中的内容
  global_domain_map.clear();
  return 0;
}

PyObject * set_global_backend(PyObject * /* self */, PyObject * args) {
  PyObject * backend;
  int only = false, coerce = false, try_last = false;
  // 解析参数元组，获取后端对象和选项
  if (!PyArg_ParseTuple(args, "O|ppp", &backend, &coerce, &only, &try_last))
    return nullptr;

  // 验证后端对象是否有效
  if (!backend_validate_ua_domain(backend)) {
    return nullptr;
  }

  // 根据后端对象设置全局后端对象和选项
  const auto res =
      backend_for_each_domain_string(backend, [&](const std::string & domain) {
        backend_options options;
        options.backend = py_ref::ref(backend);
        options.coerce = coerce;
        options.only = only;

        auto & domain_globals = (*current_global_state)[domain];
        domain_globals.global = options;
        domain_globals.try_global_backend_last = try_last;
        return LoopReturn::Continue;
      });

  if (res == LoopReturn::Error)
    return nullptr;

  Py_RETURN_NONE;
}

PyObject * register_backend(PyObject * /* self */, PyObject * args) {
  PyObject * backend;
  // 解析参数元组，获取后端对象
  if (!PyArg_ParseTuple(args, "O", &backend))
    return nullptr;

  // 验证后端对象是否有效
  if (!backend_validate_ua_domain(backend)) {
    return nullptr;
  }

  // 将后端对象注册到各个域
  const auto ret =
      backend_for_each_domain_string(backend, [&](const std::string & domain) {
        (*current_global_state)[domain].registered.push_back(
            py_ref::ref(backend));
        return LoopReturn::Continue;
      });
  if (ret == LoopReturn::Error)
    return nullptr;

  Py_RETURN_NONE;
}

void clear_single(const std::string & domain, bool registered, bool global) {
  // 查找指定域的全局状态
  auto domain_globals = current_global_state->find(domain);
  if (domain_globals == current_global_state->end())
    return;

  // 清空注册的后端对象或者全局后端对象，或者二者都清空
  if (registered && global) {
    current_global_state->erase(domain_globals);
    return;
  }

  if (registered) {
    domain_globals->second.registered.clear();
  }

  if (global) {
    domain_globals->second.global.backend.reset();
    domain_globals->second.try_global_backend_last = false;
  }
}
PyObject * clear_backends(PyObject * /* self */, PyObject * args) {
  // domain 是指向 Python 对象的指针，默认为 nullptr
  PyObject * domain = nullptr;
  // registered 和 global 分别初始化为 true 和 false
  int registered = true, global = false;
  // 解析传入的 Python 元组参数，可以接受一个对象和两个可选的布尔参数
  if (!PyArg_ParseTuple(args, "O|pp", &domain, &registered, &global))
    // 解析失败时返回空指针
    return nullptr;

  // 如果 domain 是 None，并且 registered 和 global 都为真
  if (domain == Py_None && registered && global) {
    // 清除全局状态
    current_global_state->clear();
    // 返回 Python 中的 None
    Py_RETURN_NONE;
  }

  // 将 domain 转换为字符串形式
  auto domain_str = domain_to_string(domain);
  // 调用 clear_single 函数处理给定的域名字符串及其注册和全局标志
  clear_single(domain_str, registered, global);
  // 返回 Python 中的 None
  Py_RETURN_NONE;
}

/** Common functionality of set_backend and skip_backend */
template <typename T>
class context_helper {
public:
  // BackendLists 是一个小型动态数组，存储指向 vector<T> 的指针
  using BackendLists = SmallDynamicArray<std::vector<T> *>;
  // using BackendLists = std::vector<std::vector<T> *>;

private:
  // new_backend_ 存储当前上下文中的后端对象
  T new_backend_;
  // backend_lists_ 存储多个后端列表的指针
  BackendLists backend_lists_;

public:
  // 获取当前的后端对象
  const T & get_backend() const { return new_backend_; }

  // 默认构造函数
  context_helper() {}

  // 初始化函数，接受一个移动语义的 BackendLists 对象和一个新的后端对象
  bool init(BackendLists && backend_lists, T new_backend) {
    // 静态断言确保 BackendLists 可以进行无异常移动赋值
    static_assert(std::is_nothrow_move_assignable<BackendLists>::value, "");
    // 将传入的 backend_lists 移动赋值给 backend_lists_
    backend_lists_ = std::move(backend_lists);
    // 移动赋值新的后端对象给 new_backend_
    new_backend_ = std::move(new_backend);
    return true;
  }

  // 初始化函数，接受一个 vector<T> 的引用和一个新的后端对象
  bool init(std::vector<T> & backends, T new_backend) {
    try {
      // 使用 backends 构造一个只包含一个元素的 BackendLists 对象
      backend_lists_ = BackendLists(1, &backends);
    } catch (std::bad_alloc &) {
      // 内存分配失败时，设置 Python 的内存错误异常
      PyErr_NoMemory();
      return false;
    }
    // 移动赋值新的后端对象给 new_backend_
    new_backend_ = std::move(new_backend);
    return true;
  }

  // 进入上下文的函数
  bool enter() {
    auto first = backend_lists_.begin();
    auto last = backend_lists_.end();
    auto cur = first;
    try {
      // 遍历所有的后端列表，并在每个列表中添加新的后端对象
      for (; cur < last; ++cur) {
        (*cur)->push_back(new_backend_);
      }
    } catch (std::bad_alloc &) {
      // 如果内存分配失败，回滚之前已添加的后端对象
      for (; first < cur; ++first) {
        (*first)->pop_back();
      }
      // 设置 Python 的内存错误异常
      PyErr_NoMemory();
      return false;
    }
    return true;
  }

  // 退出上下文的函数
  bool exit() {
    bool success = true;

    // 遍历所有的后端列表
    for (auto * backends : backend_lists_) {
      // 如果列表为空，设置 Python 的系统退出异常并标记操作失败
      if (backends->empty()) {
        PyErr_SetString(
            PyExc_SystemExit, "__exit__ call has no matching __enter__");
        success = false;
        continue;
      }

      // 如果列表中最后一个元素不等于当前上下文中的后端对象，
      // 设置 Python 的运行时错误异常并标记操作失败
      if (backends->back() != new_backend_) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "Found invalid context state while in __exit__. "
            "__enter__ and __exit__ may be unmatched");
        success = false;
      }

      // 移除列表中的最后一个后端对象
      backends->pop_back();
    }

    return success;
  }
};


struct SetBackendContext {
  PyObject_HEAD

  // ctx_ 是一个 context_helper 对象，用于管理后端对象的上下文
  context_helper<backend_options> ctx_;

  // 释放函数，用于释放 SetBackendContext 对象的内存
  static void dealloc(SetBackendContext * self) {
    // 取消对象的垃圾回收跟踪
    PyObject_GC_UnTrack(self);
    // 调用对象的析构函数
    self->~SetBackendContext();
    // 释放对象的内存
    Py_TYPE(self)->tp_free(self);
  }

  // 创建新的 SetBackendContext 对象的函数
  static PyObject * new_(
      PyTypeObject * type, PyObject * args, PyObject * kwargs) {
    // 分配内存来存储新的 SetBackendContext 对象
    auto self = reinterpret_cast<SetBackendContext *>(type->tp_alloc(type, 0));
    if (self == nullptr)
      // 内存分配失败时返回空指针
      return nullptr;

    // 使用 placement new 在分配的内存上构造 SetBackendContext 对象
    self = new (self) SetBackendContext;
    // 将构造的对象转换为 PyObject* 并返回
    return reinterpret_cast<PyObject *>(self);
  }

  // 初始化函数，用于初始化 SetBackendContext 对象
  static int init(
      SetBackendContext * self, PyObject * args, PyObject * kwargs) {
    // 定义关键字列表，用于参数解析
    static const char * kwlist[] = {"backend", "coerce", "only", nullptr};
    // 指向传入参数的 Python 对象指针
    PyObject * backend = nullptr;
    // 是否强制类型转换的标志，默认为假
    int coerce = false;
    // 是否仅限制的标志，默认为假

    // 使用 PyArg_ParseTupleAndKeywords 函数解析传入的参数和关键字参数
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O|pp", (char **)kwlist, &backend, &coerce, &only))
      return -1;

    // 验证传入的 backend 参数是否为有效的域名格式
    if (!backend_validate_ua_domain(backend)) {
      return -1;
    }

    // 获取 backend 参数对应的域名数量
    auto num_domains = backend_get_num_domains(backend);
    // 如果获取失败，返回错误
    if (num_domains < 0) {
      return -1;
    }

    try {
      // 创建后端列表对象，其大小为 num_domains
      decltype(ctx_)::BackendLists backend_lists(num_domains);
      // 索引初始化为 0
      int idx = 0;

      // 针对每个域名调用 backend_for_each_domain_string 函数
      const auto ret = backend_for_each_domain_string(
          backend, [&](const std::string & domain) {
            // 将本地域名映射中的首选项指针存入 backend_lists 中相应位置
            backend_lists[idx] = &local_domain_map[domain].preferred;
            ++idx;
            return LoopReturn::Continue;
          });

      // 如果循环返回错误，返回 -1
      if (ret == LoopReturn::Error) {
        return -1;
      }

      // 创建后端选项对象
      backend_options opt;
      // 设置选项对象的 backend 成员为传入的 backend 参数的引用
      opt.backend = py_ref::ref(backend);
      // 设置选项对象的 coerce 成员为传入的 coerce 标志
      opt.coerce = coerce;
      // 设置选项对象的 only 成员为传入的 only 标志

      // 初始化 self 对象的 ctx_ 成员
      if (!self->ctx_.init(std::move(backend_lists), opt)) {
        return -1;
      }
    } catch (std::bad_alloc &) {
      // 内存分配失败，引发 Python 内存错误异常
      PyErr_NoMemory();
      return -1;
    }

    // 执行成功，返回 0
    return 0;
  }

  // 进入上下文管理器
  static PyObject * enter__(SetBackendContext * self, PyObject * /* args */) {
    // 调用 ctx_ 对象的 enter 方法
    if (!self->ctx_.enter())
      return nullptr;
    // 返回 None 对象
    Py_RETURN_NONE;
  }

  // 退出上下文管理器
  static PyObject * exit__(SetBackendContext * self, PyObject * /*args*/) {
    // 调用 ctx_ 对象的 exit 方法
    if (!self->ctx_.exit())
      return nullptr;
    // 返回 None 对象
    Py_RETURN_NONE;
  }

  // 遍历对象引用
  static int traverse(SetBackendContext * self, visitproc visit, void * arg) {
    // 访问并遍历 ctx_ 对象中的 backend 成员
    Py_VISIT(self->ctx_.get_backend().backend.get());
    // 返回 0 表示成功
    return 0;
  }

  // 对象序列化
  static PyObject * pickle_(SetBackendContext * self, PyObject * /*args*/) {
    // 获取 ctx_ 对象中的 backend 成员，并赋值给 opt
    const backend_options & opt = self->ctx_.get_backend();
    // 使用 py_make_tuple 函数创建元组对象，包含 opt 的成员值，并释放其所有权
    return py_make_tuple(opt.backend, py_bool(opt.coerce), py_bool(opt.only))
        .release();
  }
};

// 定义一个结构体 SkipBackendContext，继承自 PyObject_HEAD
struct SkipBackendContext {
  PyObject_HEAD

  // 使用 context_helper 包装的 py_ref 类型成员变量 ctx_
  context_helper<py_ref> ctx_;

  // 静态方法：释放 SkipBackendContext 实例
  static void dealloc(SkipBackendContext * self) {
    PyObject_GC_UnTrack(self);  // 停止追踪 Python 垃圾回收系统对 self 的引用
    self->~SkipBackendContext();  // 调用析构函数
    Py_TYPE(self)->tp_free(self);  // 释放内存
  }

  // 静态方法：创建 SkipBackendContext 实例
  static PyObject * new_(
      PyTypeObject * type, PyObject * args, PyObject * kwargs) {
    auto self = reinterpret_cast<SkipBackendContext *>(type->tp_alloc(type, 0));  // 分配内存
    if (self == nullptr)
      return nullptr;

    // 在分配的内存上使用 placement new 构造 SkipBackendContext 对象
    self = new (self) SkipBackendContext;
    return reinterpret_cast<PyObject *>(self);  // 返回 Python 对象指针
  }

  // 静态方法：初始化 SkipBackendContext 实例
  static int init(
      SkipBackendContext * self, PyObject * args, PyObject * kwargs) {
    static const char * kwlist[] = {"backend", nullptr};
    PyObject * backend;

    // 解析参数，期望传入一个对象类型的参数
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O", (char **)kwlist, &backend))
      return -1;

    // 验证 backend 是否有效的用户代理域名
    if (!backend_validate_ua_domain(backend)) {
      return -1;
    }

    // 获取 backend 包含的域名数量
    auto num_domains = backend_get_num_domains(backend);
    if (num_domains < 0) {
      return -1;
    }

    try {
      // 使用 decltype(ctx_)::BackendLists 类型定义 backend_lists
      decltype(ctx_)::BackendLists backend_lists(num_domains);
      int idx = 0;

      // 对 backend 中每个域名调用 lambda 表达式进行处理
      const auto ret = backend_for_each_domain_string(
          backend, [&](const std::string & domain) {
            // 将本地域名映射中的跳过对象地址存储到 backend_lists 中
            backend_lists[idx] = &local_domain_map[domain].skipped;
            ++idx;
            return LoopReturn::Continue;  // 继续遍历
          });

      // 处理过程中如果出现错误，则返回错误状态
      if (ret == LoopReturn::Error) {
        return -1;
      }

      // 初始化 ctx_ 成员变量，使用移动构造函数传递 backend_lists 和 backend
      if (!self->ctx_.init(std::move(backend_lists), py_ref::ref(backend))) {
        return -1;
      }
    } catch (std::bad_alloc &) {
      PyErr_NoMemory();  // 内存分配失败异常处理
      return -1;
    }

    return 0;  // 返回成功状态
  }

  // 进入上下文管理器
  static PyObject * enter__(SkipBackendContext * self, PyObject * /* args */) {
    if (!self->ctx_.enter())  // 调用 ctx_ 的 enter 方法
      return nullptr;
    Py_RETURN_NONE;  // 返回 Python None 对象
  }

  // 退出上下文管理器
  static PyObject * exit__(SkipBackendContext * self, PyObject * /*args*/) {
    if (!self->ctx_.exit())  // 调用 ctx_ 的 exit 方法
      return nullptr;
    Py_RETURN_NONE;  // 返回 Python None 对象
  }

  // 遍历 SkipBackendContext 对象
  static int traverse(SkipBackendContext * self, visitproc visit, void * arg) {
    Py_VISIT(self->ctx_.get_backend().get());  // 访问 ctx_ 成员变量中的 backend 对象
    return 0;  // 返回成功状态
  }

  // 将 SkipBackendContext 对象转为 pickle 对象
  static PyObject * pickle_(SkipBackendContext * self, PyObject * /*args*/) {
    return py_make_tuple(self->ctx_.get_backend()).release();  // 调用 py_make_tuple 函数返回 Pickle 对象
  }
};

// 获取指定域名对应的本地后端
const local_backends & get_local_backends(const std::string & domain_key) {
  static const local_backends null_local_backends;  // 定义一个静态的空 local_backends 对象
  auto itr = local_domain_map.find(domain_key);  // 查找域名在本地域名映射中的迭代器
  if (itr == local_domain_map.end()) {  // 如果找不到对应域名的映射，则返回空对象
    return null_local_backends;
  }
  return itr->second;  // 返回找到的本地后端映射
}

// 获取指定域名对应的全局后端
const global_backends & get_global_backends(const std::string & domain_key) {
  static const global_backends null_global_backends;  // 定义一个静态的空 global_backends 对象
  const auto & cur_globals = *current_global_state;  // 获取当前全局状态的引用
  auto itr = cur_globals.find(domain_key);  // 查找域名在当前全局状态中的迭代器
  if (itr == cur_globals.end()) {  // 如果找不到对应域名的映射，则返回空对象
    return null_global_backends;
  }
  return itr->second;  // 返回找到的全局后端映射
}

// 模板函数：遍历指定域名中的每个后端
template <typename Callback>
LoopReturn for_each_backend_in_domain(
    const std::string & domain_key, Callback call) {

参数 `domain_key` 是一个常量引用，表示域名或关键字。`Callback` 是一个回调函数类型，用于处理后续的操作。


  const local_backends & locals = get_local_backends(domain_key);

调用函数 `get_local_backends` 获取特定域名的本地后端信息，并将结果存储在常量引用 `locals` 中。


  auto & skip = locals.skipped;
  auto & pref = locals.preferred;

从 `locals` 中获取 `skipped` 和 `preferred` 的引用，这些是本地后端列表中跳过和首选的后端选项。


  auto should_skip = [&](PyObject * backend) -> int {
    bool success = true;
    auto it = std::find_if(skip.begin(), skip.end(), [&](const py_ref & be) {
      auto result = PyObject_RichCompareBool(be.get(), backend, Py_EQ);
      success = (result >= 0);
      return (result != 0);
    });

    if (!success) {
      return -1;
    }

    return (it != skip.end());
  };

定义 lambda 函数 `should_skip`，用于检查给定的 `backend` 是否在 `skip` 列表中。它遍历 `skip` 列表并使用 Python C API 的函数进行比较。


  LoopReturn ret = LoopReturn::Continue;

初始化 `ret`，用于迭代过程中的返回状态，初始为 `Continue` 表示继续循环。


  for (int i = pref.size() - 1; i >= 0; --i) {
    auto options = pref[i];
    int skip_current = should_skip(options.backend.get());
    if (skip_current < 0)
      return LoopReturn::Error;
    if (skip_current)
      continue;

    ret = call(options.backend.get(), options.coerce);
    if (ret != LoopReturn::Continue)
      return ret;

    if (options.only || options.coerce)
      return LoopReturn::Break;
  }

遍历 `preferred` 列表中的后端选项，按照顺序检查是否需要跳过。如果 `should_skip` 返回错误或者需要跳过，则根据情况返回 `Error` 或者继续下一个迭代。调用 `call` 处理后端操作，根据返回值决定继续还是中断迭代。


  auto & globals = get_global_backends(domain_key);

获取全局后端信息，存储在 `globals` 中。


  auto try_global_backend = [&] {
    auto & options = globals.global;
    if (!options.backend)
      return LoopReturn::Continue;

    int skip_current = should_skip(options.backend.get());
    if (skip_current < 0)
      return LoopReturn::Error;
    if (skip_current > 0)
      return LoopReturn::Continue;

    return call(options.backend.get(), options.coerce);
  };

定义 lambda 函数 `try_global_backend`，尝试处理全局后端。检查是否需要跳过后端，然后调用 `call` 处理操作。


  if (!globals.try_global_backend_last) {
    ret = try_global_backend();
    if (ret != LoopReturn::Continue)
      return ret;

    if (globals.global.only || globals.global.coerce)
      return LoopReturn::Break;
  }

如果不是最后尝试全局后端，调用 `try_global_backend` 处理操作，根据返回值决定是否继续或者中断迭代。


  for (size_t i = 0; i < globals.registered.size(); ++i) {
    py_ref backend = globals.registered[i];
    int skip_current = should_skip(backend.get());
    if (skip_current < 0)
      return LoopReturn::Error;
    if (skip_current)
      continue;

    ret = call(backend.get(), false);
    if (ret != LoopReturn::Continue)
      return ret;
  }

遍历全局注册的后端列表 `registered`，检查并处理未被跳过的后端。


  if (!globals.try_global_backend_last) {
    return ret;
  }
  return try_global_backend();

如果是最后一次尝试全局后端，返回之前的处理结果；否则调用 `try_global_backend` 处理操作并返回结果。


}

函数结束。
}

template <typename Callback>
LoopReturn for_each_backend(std::string domain, Callback call) {
  // 使用 do-while 循环遍历给定域名的后端
  do {
    // 调用 for_each_backend_in_domain 函数处理当前域名的后端
    auto ret = for_each_backend_in_domain(domain, call);
    // 如果返回值不是 LoopReturn::Continue，则直接返回该值
    if (ret != LoopReturn::Continue) {
      return ret;
    }

    // 查找域名中最后一个 '.' 的位置
    auto dot_pos = domain.rfind('.');
    // 如果找不到 '.'，则返回当前返回值
    if (dot_pos == std::string::npos) {
      return ret;
    }

    // 缩小域名字符串至最后一个 '.' 前的部分
    domain.resize(dot_pos);
  } while (!domain.empty());  // 如果域名非空，则继续循环

  // 循环结束时返回 LoopReturn::Continue，表示成功处理所有域名
  return LoopReturn::Continue;
}

struct py_func_args {
  py_ref args, kwargs;
};

struct Function {
  PyObject_HEAD
  py_ref extractor_, replacer_;  // 处理分发函数的函数对象
  std::string domain_key_;       // UTF8 编码的关联 __ua_domain__
  py_ref def_args_, def_kwargs_; // 默认参数
  py_ref def_impl_;              // 默认实现
  py_ref dict_;                  // __dict__

  // 调用函数的方法，接收参数和关键字参数
  PyObject * call(PyObject * args, PyObject * kwargs);

  // 替换分发函数中的可调用对象
  py_func_args replace_dispatchables(
      PyObject * backend, PyObject * args, PyObject * kwargs,
      PyObject * coerce);

  // 规范化函数的位置参数
  py_ref canonicalize_args(PyObject * args);
  // 规范化函数的关键字参数
  py_ref canonicalize_kwargs(PyObject * kwargs);

  // 释放函数对象的内存
  static void dealloc(Function * self) {
    PyObject_GC_UnTrack(self);  // 取消 Python 垃圾回收跟踪
    self->~Function();          // 调用函数对象的析构函数
    Py_TYPE(self)->tp_free(self);  // 释放函数对象内存
  }

  // 创建新的 Function 对象
  static PyObject * new_(
      PyTypeObject * type, PyObject * args, PyObject * kwargs) {
    auto self = reinterpret_cast<Function *>(type->tp_alloc(type, 0));  // 分配内存空间
    if (self == nullptr)
      return nullptr;

    // 在分配的内存空间上使用 placement new 构造函数创建 Function 对象
    self = new (self) Function;
    return reinterpret_cast<PyObject *>(self);  // 返回 Python 对象指针
  }

  // 初始化 Function 对象
  static int init(Function * self, PyObject * args, PyObject * /*kwargs*/) {
    PyObject *extractor, *replacer;
    PyObject * domain;
    PyObject *def_args, *def_kwargs;
    PyObject * def_impl;

    // 解析传入的参数元组
    if (!PyArg_ParseTuple(
            args, "OOO!O!O!O", &extractor, &replacer, &PyUnicode_Type, &domain,
            &PyTuple_Type, &def_args, &PyDict_Type, &def_kwargs, &def_impl)) {
      return -1;  // 解析失败则返回 -1
    }

    // 检查提取器和替换器是否可调用
    if (!PyCallable_Check(extractor) ||
        (replacer != Py_None && !PyCallable_Check(replacer))) {
      PyErr_SetString(
          PyExc_TypeError, "Argument extractor and replacer must be callable");
      return -1;  // 类型错误则返回 -1
    }

    // 检查默认实现是否为可调用对象或为 None
    if (def_impl != Py_None && !PyCallable_Check(def_impl)) {
      PyErr_SetString(
          PyExc_TypeError, "Default implementation must be Callable or None");
      return -1;  // 类型错误则返回 -1
    }

    // 将 domain 转换为 UTF8 编码的字符串并赋值给 domain_key_
    self->domain_key_ = domain_to_string(domain);
    if (PyErr_Occurred())
      return -1;  // 如果发生错误则返回 -1

    // 分别持有传入的各种参数和对象的引用
    self->extractor_ = py_ref::ref(extractor);
    self->replacer_ = py_ref::ref(replacer);
    self->def_args_ = py_ref::ref(def_args);
    self->def_kwargs_ = py_ref::ref(def_kwargs);
    self->def_impl_ = py_ref::ref(def_impl);

    // 成功初始化返回 0
    return 0;
  }
};


这些注释详细解释了每行代码的功能和作用，确保代码的每个细节都得到了适当的说明。
    return 0;
  }



# 返回整数 0，结束函数
static PyObject * repr(Function * self);



# 声明一个静态函数 repr，返回类型为 PyObject*，参数为 Function* 类型的指针 self
static PyObject * descr_get(PyObject * self, PyObject * obj, PyObject * type);



# 声明一个静态函数 descr_get，返回类型为 PyObject*，参数为三个 PyObject* 类型的指针 self, obj, type
static int traverse(Function * self, visitproc visit, void * arg);



# 声明一个静态函数 traverse，返回类型为 int，参数为 Function* 类型的指针 self，一个 visitproc 类型的 visit 函数指针，和一个 void* 类型的 arg 指针
static int clear(Function * self);



# 声明一个静态函数 clear，返回类型为 int，参数为 Function* 类型的指针 self
static PyObject * get_extractor(Function * self);



# 声明一个静态函数 get_extractor，返回类型为 PyObject*，参数为 Function* 类型的指针 self
static PyObject * get_replacer(Function * self);



# 声明一个静态函数 get_replacer，返回类型为 PyObject*，参数为 Function* 类型的指针 self
static PyObject * get_domain(Function * self);



# 声明一个静态函数 get_domain，返回类型为 PyObject*，参数为 Function* 类型的指针 self
static PyObject * get_default(Function * self);



# 声明一个静态函数 get_default，返回类型为 PyObject*，参数为 Function* 类型的指针 self
};

// 检查给定的 Python 对象是否等于默认值
bool is_default(PyObject * value, PyObject * def) {
  // TODO: 如果是内置类型，可以进行更丰富的比较吗？（如果廉价的话）
  return (value == def);
}

// 规范化函数参数列表，确保与默认参数一致
py_ref Function::canonicalize_args(PyObject * args) {
  const auto arg_size = PyTuple_GET_SIZE(args);  // 获取参数元组的大小
  const auto def_size = PyTuple_GET_SIZE(def_args_.get());  // 获取默认参数元组的大小

  if (arg_size > def_size)
    return py_ref::ref(args);  // 如果参数个数多于默认参数，直接返回参数的引用

  Py_ssize_t mismatch = 0;
  // 从最后一个参数开始向前遍历
  for (Py_ssize_t i = arg_size - 1; i >= 0; --i) {
    auto val = PyTuple_GET_ITEM(args, i);  // 获取第 i 个参数的值
    auto def = PyTuple_GET_ITEM(def_args_.get(), i);  // 获取第 i 个默认参数的值
    if (!is_default(val, def)) {
      mismatch = i + 1;  // 如果参数值与默认值不相同，记录不匹配的位置
      break;
    }
  }

  return py_ref::steal(PyTuple_GetSlice(args, 0, mismatch));  // 返回规范化后的参数元组
}

// 规范化关键字参数，删除与默认值相同的项
py_ref Function::canonicalize_kwargs(PyObject * kwargs) {
  if (kwargs == nullptr)
    return py_ref::steal(PyDict_New());  // 如果关键字参数为空，返回一个新的空字典

  PyObject *key, *def_value;
  Py_ssize_t pos = 0;
  // 遍历默认关键字参数字典
  while (PyDict_Next(def_kwargs_.get(), &pos, &key, &def_value)) {
    auto val = PyDict_GetItem(kwargs, key);  // 获取关键字参数中的值
    if (val && is_default(val, def_value)) {
      PyDict_DelItem(kwargs, key);  // 如果值与默认值相同，则从关键字参数中删除该项
    }
  }
  return py_ref::ref(kwargs);  // 返回规范化后的关键字参数字典
}

// 替换可调度对象，并返回新的参数和关键字参数
py_func_args Function::replace_dispatchables(
    PyObject * backend, PyObject * args, PyObject * kwargs, PyObject * coerce) {
  auto has_ua_convert = PyObject_HasAttr(backend, identifiers.ua_convert.get());
  if (!has_ua_convert) {
    return {py_ref::ref(args), py_ref::ref(kwargs)};  // 如果后端对象没有 ua_convert 属性，直接返回原始的参数和关键字参数
  }

  auto dispatchables =
      py_ref::steal(PyObject_Call(extractor_.get(), args, kwargs));  // 调用提取器函数获取可调度对象
  if (!dispatchables)
    return {};  // 如果调用失败，返回空结果

  PyObject * convert_args[] = {backend, dispatchables.get(), coerce};
  auto res = py_ref::steal(Q_PyObject_VectorcallMethod(
      identifiers.ua_convert.get(), convert_args,
      array_size(convert_args) | Q_PY_VECTORCALL_ARGUMENTS_OFFSET, nullptr));  // 调用 ua_convert 方法
  if (!res) {
    return {};  // 如果调用失败，返回空结果
  }

  if (res == Py_NotImplemented) {
    return {py_ref::ref(res), nullptr};  // 如果返回 Py_NotImplemented，返回其作为结果的元组
  }

  auto replaced_args = py_ref::steal(PySequence_Tuple(res.get()));  // 将返回结果转换为元组
  if (!replaced_args)
    return {};  // 如果转换失败，返回空结果

  PyObject * replacer_args[] = {nullptr, args, kwargs, replaced_args.get()};
  res = py_ref::steal(Q_PyObject_Vectorcall(
      replacer_.get(), &replacer_args[1],
      (array_size(replacer_args) - 1) | Q_PY_VECTORCALL_ARGUMENTS_OFFSET,
      nullptr));  // 调用 replacer 函数
  if (!res)
    return {};  // 如果调用失败，返回空结果

  if (!PyTuple_Check(res.get()) || PyTuple_Size(res.get()) != 2) {
    PyErr_SetString(
        PyExc_TypeError,
        "Argument replacer must return a 2-tuple (args, kwargs)");  // 如果返回值不是 2 元组，则设置类型错误异常
    return {};  // 返回空结果
  }

  auto new_args = py_ref::ref(PyTuple_GET_ITEM(res.get(), 0));  // 获取返回结果的第一个元素作为新的参数
  auto new_kwargs = py_ref::ref(PyTuple_GET_ITEM(res.get(), 1));  // 获取返回结果的第二个元素作为新的关键字参数

  new_kwargs = canonicalize_kwargs(new_kwargs.get());  // 规范化新的关键字参数

  if (!PyTuple_Check(new_args.get()) || !PyDict_Check(new_kwargs.get())) {
    PyErr_SetString(PyExc_ValueError, "Invalid return from argument_replacer");  // 如果返回的参数和关键字参数不是元组和字典类型，则设置值错误异常
    return {};  // 返回空结果
  }

  return {std::move(new_args), std::move(new_kwargs)};  // 返回新的参数和关键字参数
}

// 调用函数对象的 call 方法
PyObject * Function_call(Function * self, PyObject * args, PyObject * kwargs) {
  return self->call(args, kwargs);  // 返回调用结果
}
}

# 定义一个名为 py_errinf 的类
class py_errinf {
  # 类的私有成员变量，用于保存异常的类型、值和回溯信息
  py_ref type_, value_, traceback_;

public:
  # 静态方法，用于获取当前异常信息并返回一个 py_errinf 对象
  static py_errinf fetch() {
    # 定义三个 PyObject 指针，用于存储异常的类型、值和回溯信息
    PyObject *type, *value, *traceback;
    # 获取当前 Python 异常信息并存储到 type、value、traceback 指针中
    PyErr_Fetch(&type, &value, &traceback);
    # 创建一个 py_errinf 对象 err
    py_errinf err;
    # 调用 err 对象的 set 方法，将获取到的异常信息存储到 err 对象的成员变量中
    err.set(type, value, traceback);
    # 返回存储异常信息的 err 对象
    return err;
  }

  # 公有方法，用于获取异常的值
  py_ref get_exception() {
    # 调用 normalize 方法，确保异常信息已经标准化
    normalize();
    # 返回异常的值
    return value_;
  }

private:
  # 私有方法，用于设置异常信息
  void set(PyObject * type, PyObject * value, PyObject * traceback) {
    # 使用 py_ref::steal 方法将异常的类型、值和回溯信息转换为 py_ref 对象并保存到类的成员变量中
    type_ = py_ref::steal(type);
    value_ = py_ref::steal(value);
    traceback_ = py_ref::steal(traceback);
  }

  # 私有方法，用于标准化异常信息
  void normalize() {
    # 释放 type_、value_、traceback_ 对象并获取其原始指针
    auto type = type_.release();
    auto value = value_.release();
    auto traceback = value_.release();  # 此处应为 traceback_.release();
    # 标准化异常信息，并更新 type、value、traceback 指针
    PyErr_NormalizeException(&type, &value, &traceback);
    # 如果存在 traceback 信息，则将其设置为异常的回溯信息
    if (traceback) {
      PyException_SetTraceback(value, traceback);
    }
    # 调用 set 方法，更新类的成员变量
    set(type, value, traceback);
  }
};
PyObject * Function::call(PyObject * args_, PyObject * kwargs_) {
  // 规范化传入的参数和关键字参数
  auto args = canonicalize_args(args_);
  auto kwargs = canonicalize_kwargs(kwargs_);

  // 初始化结果变量和错误列表
  py_ref result;
  std::vector<std::pair<py_ref, py_errinf>> errors;

  // 调用每个后端处理函数，根据返回值决定后续操作
  auto ret =
      for_each_backend(domain_key_, [&, this](PyObject * backend, bool coerce) {
        // 替换可调度对象，并检查返回值
        auto new_args = replace_dispatchables(
            backend, args.get(), kwargs.get(), coerce ? Py_True : Py_False);
        if (new_args.args == Py_NotImplemented)
          return LoopReturn::Continue; // 继续循环
        if (new_args.args == nullptr)
          return LoopReturn::Error; // 返回错误

        // 准备参数列表并调用向量调用方法
        PyObject * args[] = {
            backend, reinterpret_cast<PyObject *>(this), new_args.args.get(),
            new_args.kwargs.get()};
        result = py_ref::steal(Q_PyObject_VectorcallMethod(
            identifiers.ua_function.get(), args,
            array_size(args) | Q_PY_VECTORCALL_ARGUMENTS_OFFSET, nullptr));

        // 如果抛出BackendNotImplemented异常，则返回NotImplemented
        if (!result &&
            PyErr_ExceptionMatches(BackendNotImplementedError.get())) {
          errors.push_back({py_ref::ref(backend), py_errinf::fetch()});
          result = py_ref::ref(Py_NotImplemented);
        }

        // 尝试使用默认实现调用当前后端
        if (result == Py_NotImplemented && def_impl_ != Py_None) {
          backend_options opt;
          opt.backend = py_ref::ref(backend);
          opt.coerce = coerce;
          opt.only = true;
          context_helper<backend_options> ctx;
          try {
            if (!ctx.init(
                    local_domain_map[domain_key_].preferred, std::move(opt)))
              return LoopReturn::Error;
          } catch (std::bad_alloc &) {
            PyErr_NoMemory();
            return LoopReturn::Error;
          }

          if (!ctx.enter())
            return LoopReturn::Error;

          // 调用默认实现函数
          result = py_ref::steal(PyObject_Call(
              def_impl_.get(), new_args.args.get(), new_args.kwargs.get()));

          // 如果抛出BackendNotImplemented异常，则返回NotImplemented
          if (PyErr_Occurred() &&
              PyErr_ExceptionMatches(BackendNotImplementedError.get())) {
            errors.push_back({py_ref::ref(backend), py_errinf::fetch()});
            result = py_ref::ref(Py_NotImplemented);
          }

          if (!ctx.exit())
            return LoopReturn::Error;
        }

        // 检查结果是否为空
        if (!result)
          return LoopReturn::Error;

        // 如果结果为NotImplemented，则继续循环
        if (result == Py_NotImplemented)
          return LoopReturn::Continue;

        // 如果成功调用后端，则退出循环
        return LoopReturn::Break; // Backend called successfully
      });

  // 如果有错误发生，返回空指针
  if (ret == LoopReturn::Error)
    return nullptr;

  // 如果有有效结果且不是NotImplemented，则释放结果
  if (result && result != Py_NotImplemented)
    return result.release();

  // 最后尝试直接调用默认实现函数
  // 仅当没有后端被标记为only或coerce时才调用
  if (ret == LoopReturn::Continue && def_impl_ != Py_None) {
    result =
        py_ref::steal(PyObject_Call(def_impl_.get(), args.get(), kwargs.get()));
    // 如果结果为空指针，则执行以下操作
    if (!result) {
      // 如果当前异常不匹配 BackendNotImplementedError 异常类型，则返回空指针
      if (!PyErr_ExceptionMatches(BackendNotImplementedError.get()))
        return nullptr;
    
      // 将 Py_None 包装成 Python 引用，获取异常信息并存入错误列表
      errors.push_back({py_ref::ref(Py_None), py_errinf::fetch()});
      // 将 Py_NotImplemented 包装成 Python 引用，并赋给 result
      result = py_ref::ref(Py_NotImplemented);
    } else if (result != Py_NotImplemented)
      // 如果 result 不等于 Py_NotImplemented，则释放 result 并返回其所有权
      return result.release();
    }
    
    
    // 所有后端和默认值都失败，构造异常元组
    auto exception_tuple = py_ref::steal(PyTuple_New(errors.size() + 1));
    PyTuple_SET_ITEM(
        exception_tuple.get(), 0,
        PyUnicode_FromString(
            "No selected backends had an implementation for this function."));
    for (size_t i = 0; i < errors.size(); ++i) {
      // 创建一个元组，其中包含错误的第一个元素和异常的第二个元素
      auto pair =
          py_make_tuple(errors[i].first, errors[i].second.get_exception());
      if (!pair)
        return nullptr;
    
      // 将创建的元组放入异常元组中的适当位置
      PyTuple_SET_ITEM(exception_tuple.get(), i + 1, pair.release());
    }
    // 设置 BackendNotImplementedError 异常对象为带有异常元组的当前异常
    PyErr_SetObject(BackendNotImplementedError.get(), exception_tuple.get());
    // 返回空指针
    return nullptr;
PyObject * Function::repr(Function * self) {
  // 如果存在 __name__ 属性，从字典中获取它
  if (self->dict_)
    if (auto name = PyDict_GetItemString(self->dict_.get(), "__name__"))
      // 根据 __name__ 属性创建一个格式化的 Unicode 字符串表示
      return PyUnicode_FromFormat("<uarray multimethod '%S'>", name);

  // 如果不存在 __name__ 属性，则返回默认的 Unicode 字符串表示
  return PyUnicode_FromString("<uarray multimethod>");
}


/** Implements the descriptor protocol to allow binding to class instances */
PyObject * Function::descr_get(
    PyObject * self, PyObject * obj, PyObject * type) {
  // 如果 obj 为 NULL，说明该方法被直接访问，增加对 self 的引用计数并返回 self
  if (!obj) {
    Py_INCREF(self);
    return self;
  }

  // 否则创建一个新的方法对象，绑定到 obj 实例上
  return PyMethod_New(self, obj);
}


/** Make members visible to the garbage collector */
int Function::traverse(Function * self, visitproc visit, void * arg) {
  // 逐个增加成员对象的引用计数，以便垃圾收集器访问
  Py_VISIT(self->extractor_.get());
  Py_VISIT(self->replacer_.get());
  Py_VISIT(self->def_args_.get());
  Py_VISIT(self->def_kwargs_.get());
  Py_VISIT(self->def_impl_.get());
  Py_VISIT(self->dict_.get());
  return 0;
}


/** Break reference cycles when being GCed */
int Function::clear(Function * self) {
  // 逐个释放成员对象的引用，以便断开可能存在的循环引用
  self->extractor_.reset();
  self->replacer_.reset();
  self->def_args_.reset();
  self->def_kwargs_.reset();
  self->def_impl_.reset();
  self->dict_.reset();
  return 0;
}

PyObject * Function::get_extractor(Function * self) {
  // 增加并返回提取器对象的引用
  Py_INCREF(self->extractor_.get());
  return self->extractor_.get();
}

PyObject * Function::get_replacer(Function * self) {
  // 增加并返回替换器对象的引用
  Py_INCREF(self->replacer_.get());
  return self->replacer_.get();
}

PyObject * Function::get_default(Function * self) {
  // 增加并返回默认实现对象的引用
  Py_INCREF(self->def_impl_.get());
  return self->def_impl_.get();
}

PyObject * Function::get_domain(Function * self) {
  // 返回域键的 Unicode 字符串表示
  return PyUnicode_FromStringAndSize(
      self->domain_key_.c_str(), self->domain_key_.size());
}


PyMethodDef BackendState_Methods[] = {
    // _pickle 方法，用于序列化对象
    {"_pickle", (PyCFunction)BackendState::pickle_, METH_NOARGS, nullptr},
    // _unpickle 方法，用于反序列化对象
    {"_unpickle", (PyCFunction)BackendState::unpickle_,
     METH_VARARGS | METH_CLASS, nullptr},
    // Sentinel，表示方法列表的结束
    {NULL} /* Sentinel */
};

PyTypeObject BackendStateType = {
    PyVarObject_HEAD_INIT(NULL, 0)     /* boilerplate */
    "uarray._BackendState",            /* tp_name */
    sizeof(BackendState),              /* tp_basicsize */
    0,                                 /* tp_itemsize */
    (destructor)BackendState::dealloc, /* tp_dealloc */
    0,                                 /* tp_print */
    0,                                 /* tp_getattr */
    0,                                 /* tp_setattr */
    0,                                 /* tp_reserved */
    0,                                 /* tp_repr */
    0,                                 /* tp_as_number */
    0,                                 /* tp_as_sequence */
    0,                                 /* tp_as_mapping */
    0,                                 /* tp_hash  */
    0,                                 /* tp_call */
    0,                                 /* tp_str */
    0,                                 /* tp_getattro */
    0                                  /* tp_setattro */
};
    0,                                 /* tp_as_buffer */
    # 缓冲区接口（buffer interface）未实现，设置为0表示不支持
    Py_TPFLAGS_DEFAULT,                /* tp_flags */
    # 类型对象的默认标志，通常为默认标志集合
    0,                                 /* tp_doc */
    # 文档字符串，此处未提供任何文档
    0,                                 /* tp_traverse */
    # 用于遍历对象的函数，此处未实现
    0,                                 /* tp_clear */
    # 清理函数，用于释放对象占用的资源
    0,                                 /* tp_richcompare */
    # 丰富比较（rich comparison）函数，未提供
    0,                                 /* tp_weaklistoffset */
    # 弱引用列表偏移量，用于支持对象的弱引用
    0,                                 /* tp_iter */
    # 迭代器协议方法，未实现
    0,                                 /* tp_iternext */
    # 迭代器的下一个方法，未实现
    BackendState_Methods,              /* tp_methods */
    # 指向类型对象的方法描述结构体
    0,                                 /* tp_members */
    # 成员描述，此处未定义
    0,                                 /* tp_getset */
    # 获取/设置描述符，未定义
    0,                                 /* tp_base */
    # 基类，此处未定义基类
    0,                                 /* tp_dict */
    # 类的字典属性，未定义
    0,                                 /* tp_descr_get */
    # 获取描述符，未实现
    0,                                 /* tp_descr_set */
    # 设置描述符，未实现
    0,                                 /* tp_dictoffset */
    # 类字典偏移量，未定义
    0,                                 /* tp_init */
    # 初始化函数，未实现
    0,                                 /* tp_alloc */
    # 分配函数，未实现
    BackendState::new_,                /* tp_new */
    # 类的构造函数，指向BackendState::new_方法
};

// 定义获取状态的函数，接受两个参数，但不使用它们
PyObject * get_state(PyObject * /* self */, PyObject * /* args */) {
  // 创建 Python 引用对象，用于包装调用 C 函数获取的对象
  py_ref ref = py_ref::steal(Q_PyObject_Vectorcall(
      reinterpret_cast<PyObject *>(&BackendStateType), nullptr, 0, nullptr));
  // 将 Python 对象转换为 C++ 结构体指针
  BackendState * output = reinterpret_cast<BackendState *>(ref.get());

  // 设置 output 结构体的本地变量为全局映射 local_domain_map
  output->locals = local_domain_map;
  // 确定是否使用线程局部全局变量，根据当前全局状态判断
  output->use_thread_local_globals =
      (current_global_state != &global_domain_map);
  // 设置 output 结构体的全局变量为当前全局状态的副本
  output->globals = *current_global_state;

  // 释放 Python 引用对象并返回其指向的 Python 对象
  return ref.release();
}

// 定义设置状态的函数，接受一个参数和一个可选的布尔值
PyObject * set_state(PyObject * /* self */, PyObject * args) {
  PyObject * arg;
  int reset_allowed = false;
  // 解析 Python 函数参数，获取第一个参数和可选的第二个参数
  if (!PyArg_ParseTuple(args, "O|p", &arg, &reset_allowed))
    return nullptr;

  // 检查参数是否是指定类型的对象
  if (!PyObject_IsInstance(
          arg, reinterpret_cast<PyObject *>(&BackendStateType))) {
    // 如果参数类型不匹配，抛出类型错误异常
    PyErr_SetString(
        PyExc_TypeError, "state must be a uarray._BackendState object.");
    return nullptr;
  }

  // 将 Python 对象转换为 C++ 结构体指针
  BackendState * state = reinterpret_cast<BackendState *>(arg);
  // 将 state 结构体的本地变量赋值给全局映射 local_domain_map
  local_domain_map = state->locals;
  // 确定是否使用线程局部全局变量，根据 reset_allowed 和 state 结构体的标志位
  bool use_thread_local_globals =
      (!reset_allowed) || state->use_thread_local_globals;
  // 根据 use_thread_local_globals 设置当前全局状态的指针
  current_global_state =
      use_thread_local_globals ? &thread_local_domain_map : &global_domain_map;

  // 如果使用线程局部全局变量，将 state 结构体的全局变量赋值给线程局部全局变量
  if (use_thread_local_globals)
    thread_local_domain_map = state->globals;
  else
    // 否则清空线程局部全局变量
    thread_local_domain_map.clear();

  // 返回 None 表示成功
  Py_RETURN_NONE;
}

// 定义确定后端的函数，接受三个参数
PyObject * determine_backend(PyObject * /*self*/, PyObject * args) {
  PyObject *domain_object, *dispatchables;
  int coerce;
  // 解析 Python 函数参数，获取域对象、分发对象和强制标志
  if (!PyArg_ParseTuple(
          args, "OOp:determine_backend", &domain_object, &dispatchables,
          &coerce))
    return nullptr;

  // 将域对象转换为字符串
  auto domain = domain_to_string(domain_object);
  if (domain.empty())
    return nullptr;

  // 转换分发对象为 Python 元组
  auto dispatchables_tuple = py_ref::steal(PySequence_Tuple(dispatchables));
  if (!dispatchables_tuple)
    return nullptr;

  // 创建 Python 引用对象，用于选定的后端
  py_ref selected_backend;
  // 对每个后端在指定域中执行操作，通过 lambda 表达式处理
  auto result = for_each_backend_in_domain(
      domain, [&](PyObject * backend, bool coerce_backend) {
        // 检查后端对象是否有 __ua_convert__ 属性
        auto has_ua_convert =
            PyObject_HasAttr(backend, identifiers.ua_convert.get());

        // 如果没有 __ua_convert__ 属性，跳过该后端
        if (!has_ua_convert) {
          return LoopReturn::Continue;
        }

        // 准备调用 __ua_convert__ 方法的参数
        PyObject * convert_args[] = {
            backend, dispatchables_tuple.get(),
            (coerce && coerce_backend) ? Py_True : Py_False};

        // 调用 __ua_convert__ 方法，并获取其结果
        auto res = py_ref::steal(Q_PyObject_VectorcallMethod(
            identifiers.ua_convert.get(), convert_args,
            array_size(convert_args) | Q_PY_VECTORCALL_ARGUMENTS_OFFSET,
            nullptr));
        if (!res) {
          return LoopReturn::Error;
        }

        // 如果 __ua_convert__ 返回 Py_NotImplemented，继续下一个后端
        if (res == Py_NotImplemented) {
          return LoopReturn::Continue;
        }

        // 如果 __ua_convert__ 成功，则选定该后端
        selected_backend = py_ref::ref(backend);
        return LoopReturn::Break;
      });

  // 如果循环未结束，返回空指针
  if (result != LoopReturn::Continue)
    return selected_backend.release();

返回从 `selected_backend` 中释放的指针对象。


  // All backends failed, raise an error
  PyErr_SetString(
      BackendNotImplementedError.get(),
      "No backends could accept input of this type.");
  return nullptr;

如果所有的后端都失败了，则设置一个错误信息并抛出 `BackendNotImplementedError` 异常，表明没有任何后端能够接受此类型的输入，然后返回空指针 `nullptr`。
}


// 在 Python 3.7 之前，getset 接收可变的 char * 参数
static char dict__[] = "__dict__"; // 静态字符数组，用于 "__dict__" 字符串
static char arg_extractor[] = "arg_extractor"; // 静态字符数组，用于 "arg_extractor" 字符串
static char arg_replacer[] = "arg_replacer"; // 静态字符数组，用于 "arg_replacer" 字符串
static char default_[] = "default"; // 静态字符数组，用于 "default" 字符串
static char domain[] = "domain"; // 静态字符数组，用于 "domain" 字符串
PyGetSetDef Function_getset[] = {
    {dict__, PyObject_GenericGetDict, PyObject_GenericSetDict}, // "__dict__" 的 getter 和 setter 定义
    {arg_extractor, (getter)Function::get_extractor, NULL}, // "arg_extractor" 的 getter 定义
    {arg_replacer, (getter)Function::get_replacer, NULL}, // "arg_replacer" 的 getter 定义
    {default_, (getter)Function::get_default, NULL}, // "default" 的 getter 定义
    {domain, (getter)Function::get_domain, NULL}, // "domain" 的 getter 定义
    {NULL} /* Sentinel */ // GetSet 结构体数组的终止标记
};

PyTypeObject FunctionType = {
    PyVarObject_HEAD_INIT(NULL, 0) /* boilerplate */ // 初始化 PyVarObject 的头部
    /* tp_name= */ "uarray._Function", // 类型对象的名称
    /* tp_basicsize= */ sizeof(Function), // 类型对象的基本大小
    /* tp_itemsize= */ 0, // 不适用于变长对象
    /* tp_dealloc= */ (destructor)Function::dealloc, // 析构函数
    /* tp_print= */ 0, // 不使用默认打印函数
    /* tp_getattr= */ 0, // 不使用默认 getattr 函数
    /* tp_setattr= */ 0, // 不使用默认 setattr 函数
    /* tp_reserved= */ 0, // 保留字段
    /* tp_repr= */ (reprfunc)Function::repr, // repr 函数
    /* tp_as_number= */ 0, // 不支持数值操作
    /* tp_as_sequence= */ 0, // 不支持序列操作
    /* tp_as_mapping= */ 0, // 不支持映射操作
    /* tp_hash= */ 0, // 不支持哈希操作
    /* tp_call= */ (ternaryfunc)Function_call, // 调用函数
    /* tp_str= */ 0, // 不使用默认 str 函数
    /* tp_getattro= */ PyObject_GenericGetAttr, // 默认的 getattr 函数
    /* tp_setattro= */ PyObject_GenericSetAttr, // 默认的 setattr 函数
    /* tp_as_buffer= */ 0, // 不支持缓冲区接口
    /* tp_flags= */
    (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Q_Py_TPFLAGS_METHOD_DESCRIPTOR), // 类型对象的标志
    /* tp_doc= */ 0, // 没有文档字符串
    /* tp_traverse= */ (traverseproc)Function::traverse, // 遍历函数
    /* tp_clear= */ (inquiry)Function::clear, // 清除函数
    /* tp_richcompare= */ 0, // 不支持富比较
    /* tp_weaklistoffset= */ 0, // 弱引用偏移量
    /* tp_iter= */ 0, // 不支持迭代器
    /* tp_iternext= */ 0, // 不支持迭代器
    /* tp_methods= */ 0, // 没有方法
    /* tp_members= */ 0, // 没有成员
    /* tp_getset= */ Function_getset, // getset 属性数组
    /* tp_base= */ 0, // 没有基类
    /* tp_dict= */ 0, // 不使用实例字典
    /* tp_descr_get= */ Function::descr_get, // 描述符的 getter
    /* tp_descr_set= */ 0, // 不使用描述符的 setter
    /* tp_dictoffset= */ offsetof(Function, dict_), // 字典偏移量
    /* tp_init= */ (initproc)Function::init, // 初始化函数
    /* tp_alloc= */ 0, // 不自定义内存分配
    /* tp_new= */ Function::new_, // 构造新对象的函数
};


PyMethodDef SetBackendContext_Methods[] = {
    {"__enter__", (PyCFunction)SetBackendContext::enter__, METH_NOARGS,
     nullptr}, // "__enter__" 方法定义
    {"__exit__", (PyCFunction)SetBackendContext::exit__, METH_VARARGS, nullptr}, // "__exit__" 方法定义
    {"_pickle", (PyCFunction)SetBackendContext::pickle_, METH_NOARGS, nullptr}, // "_pickle" 方法定义
    {NULL} /* Sentinel */ // 方法结构体数组的终止标记
};

PyTypeObject SetBackendContextType = {
    PyVarObject_HEAD_INIT(NULL, 0)             /* boilerplate */ // 初始化 PyVarObject 的头部
    "uarray._SetBackendContext",               /* tp_name */ // 类型对象的名称
    sizeof(SetBackendContext),                 /* tp_basicsize */ // 类型对象的基本大小
    0,                                         /* tp_itemsize */ // 不适用于变长对象
    (destructor)SetBackendContext::dealloc,    /* tp_dealloc */ // 析构函数
    0,                                         /* tp_print */ // 不使用默认打印函数
    0,                                         /* tp_getattr */ // 不使用默认 getattr 函数
    0,                                         /* tp_setattr */ // 不使用默认 setattr 函数
    0,                                         /* tp_reserved */ // 保留字段
    0,                                         /* tp_repr */ // 不使用默认 repr 函数
    0,                                         /* tp_as_number */
    0,                                         /* tp_as_sequence */
    0,                                         /* tp_as_mapping */
    0,                                         /* tp_hash  */
    0,                                         /* tp_call */
    0,                                         /* tp_str */
    0,                                         /* tp_getattro */
    0,                                         /* tp_setattro */
    0,                                         /* tp_as_buffer */
    (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC), /* tp_flags */
    0,                                         /* tp_doc */
    (traverseproc)SetBackendContext::traverse, /* tp_traverse */
    0,                                         /* tp_clear */
    0,                                         /* tp_richcompare */
    0,                                         /* tp_weaklistoffset */
    0,                                         /* tp_iter */
    0,                                         /* tp_iternext */
    SetBackendContext_Methods,                 /* tp_methods */
    0,                                         /* tp_members */
    0,                                         /* tp_getset */
    0,                                         /* tp_base */
    0,                                         /* tp_dict */
    0,                                         /* tp_descr_get */
    0,                                         /* tp_descr_set */
    0,                                         /* tp_dictoffset */
    (initproc)SetBackendContext::init,         /* tp_init */
    0,                                         /* tp_alloc */
    SetBackendContext::new_,                   /* tp_new */


注释：

    0,                                         /* tp_as_number */        // 指向数字协议的类型对象
    0,                                         /* tp_as_sequence */      // 指向序列协议的类型对象
    0,                                         /* tp_as_mapping */       // 指向映射协议的类型对象
    0,                                         /* tp_hash  */            // 哈希函数
    0,                                         /* tp_call */             // 调用对象的操作
    0,                                         /* tp_str */              // 对象的字符串表达形式
    0,                                         /* tp_getattro */         // 获取对象属性
    0,                                         /* tp_setattro */         // 设置对象属性
    0,                                         /* tp_as_buffer */        // 缓冲协议的类型对象
    (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC), /* tp_flags */            // 类型标志
    0,                                         /* tp_doc */              // 类型文档字符串
    (traverseproc)SetBackendContext::traverse, /* tp_traverse */         // 遍历对象的函数指针
    0,                                         /* tp_clear */            // 清理对象的函数指针
    0,                                         /* tp_richcompare */      // 富比较函数
    0,                                         /* tp_weaklistoffset */   // 弱引用列表偏移量
    0,                                         /* tp_iter */             // 迭代器生成函数
    0,                                         /* tp_iternext */         // 迭代器的下一个元素函数
    SetBackendContext_Methods,                 /* tp_methods */          // 类型的方法列表
    0,                                         /* tp_members */          // 成员变量列表
    0,                                         /* tp_getset */           // get/set 方法列表
    0,                                         /* tp_base */             // 基类指针
    0,                                         /* tp_dict */             // 类型字典
    0,                                         /* tp_descr_get */        // 获取描述符的函数
    0,                                         /* tp_descr_set */        // 设置描述符的函数
    0,                                         /* tp_dictoffset */       // 字典偏移量
    (initproc)SetBackendContext::init,         /* tp_init */             // 初始化函数指针
    0,                                         /* tp_alloc */            // 分配函数指针
    SetBackendContext::new_,                   /* tp_new */              // 新建对象函数指针
// SkipBackendContext_Methods 数组，包含了 SkipBackendContext 类的方法定义
PyMethodDef SkipBackendContext_Methods[] = {
    {"__enter__", (PyCFunction)SkipBackendContext::enter__, METH_NOARGS,
     nullptr},  // 定义 __enter__ 方法，无参数
    {"__exit__", (PyCFunction)SkipBackendContext::exit__, METH_VARARGS,
     nullptr},  // 定义 __exit__ 方法，接受参数
    {"_pickle", (PyCFunction)SkipBackendContext::pickle_, METH_NOARGS, nullptr},  // 定义 _pickle 方法，无参数
    {NULL} /* Sentinel */  // 方法列表结束标志
};

// SkipBackendContextType 结构体，定义了 SkipBackendContext 类型对象
PyTypeObject SkipBackendContextType = {
    PyVarObject_HEAD_INIT(NULL, 0)              /* boilerplate */
    "uarray._SkipBackendContext",               /* tp_name */  // 类型对象的名称
    sizeof(SkipBackendContext),                 /* tp_basicsize */  // 类型对象的基本大小
    0,                                          /* tp_itemsize */  // 元素大小，对于非变长对象为0
    (destructor)SkipBackendContext::dealloc,    /* tp_dealloc */  // 析构函数
    0,                                          /* tp_print */  // 打印函数
    0,                                          /* tp_getattr */  // 获取属性
    0,                                          /* tp_setattr */  // 设置属性
    0,                                          /* tp_reserved */
    0,                                          /* tp_repr */  // 表示函数
    0,                                          /* tp_as_number */  // 数值协议
    0,                                          /* tp_as_sequence */  // 序列协议
    0,                                          /* tp_as_mapping */  // 映射协议
    0,                                          /* tp_hash  */  // 哈希函数
    0,                                          /* tp_call */  // 调用函数
    0,                                          /* tp_str */  // 字符串转换函数
    0,                                          /* tp_getattro */  // 获取属性对象
    0,                                          /* tp_setattro */  // 设置属性对象
    0,                                          /* tp_as_buffer */  // 缓冲区协议
    (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC),  /* tp_flags */  // 类型标志
    0,                                          /* tp_doc */  // 文档字符串
    (traverseproc)SkipBackendContext::traverse, /* tp_traverse */  // 遍历对象
    0,                                          /* tp_clear */  // 清理函数
    0,                                          /* tp_richcompare */  // 对象比较函数
    0,                                          /* tp_weaklistoffset */  // 弱引用偏移量
    0,                                          /* tp_iter */  // 迭代器
    0,                                          /* tp_iternext */  // 迭代器的下一个元素
    SkipBackendContext_Methods,                 /* tp_methods */  // 类型对象的方法列表
    0,                                          /* tp_members */  // 成员列表
    0,                                          /* tp_getset */  // 获取/设置函数
    0,                                          /* tp_base */  // 基类
    0,                                          /* tp_dict */  // 字典
    0,                                          /* tp_descr_get */  // 获取描述器
    0,                                          /* tp_descr_set */  // 设置描述器
    0,                                          /* tp_dictoffset */  // 字典偏移量
    (initproc)SkipBackendContext::init,         /* tp_init */  // 初始化函数
    0,                                          /* tp_alloc */  // 分配函数
    SkipBackendContext::new_,                   /* tp_new */  // 新建对象函数
};

// method_defs 数组，包含了一些方法的定义
PyMethodDef method_defs[] = {
    {"set_global_backend", set_global_backend, METH_VARARGS, nullptr},  // 定义 set_global_backend 方法，接受参数
};
    {"register_backend", register_backend, METH_VARARGS, nullptr},
    // 注册后端函数，接受变长参数，无返回值，使用空指针
    {"clear_backends", clear_backends, METH_VARARGS, nullptr},
    // 清除所有后端函数的注册，接受变长参数，无返回值，使用空指针
    {"determine_backend", determine_backend, METH_VARARGS, nullptr},
    // 确定后端函数的具体实现，接受变长参数，无返回值，使用空指针
    {"get_state", get_state, METH_NOARGS, nullptr},
    // 获取当前状态的函数，不接受参数，使用空指针
    {"set_state", set_state, METH_VARARGS, nullptr},
    // 设置当前状态的函数，接受变长参数，使用空指针
    {NULL} /* Sentinel */
    // 用于标记函数列表结束的空项，后续没有更多函数定义
};

// 定义 Python 模块的结构体
PyModuleDef uarray_module = {
    PyModuleDef_HEAD_INIT,  // 使用宏初始化 Python 模块定义的头部
    /* m_name= */ "uarray._uarray",  // 模块名字设为 "uarray._uarray"
    /* m_doc= */ nullptr,  // 模块文档字符串为空
    /* m_size= */ -1,  // 模块状态大小设为 -1
    /* m_methods= */ method_defs,  // 模块方法定义为预先定义的方法数组
    /* m_slots= */ nullptr,  // 模块插槽为空
    /* m_traverse= */ globals_traverse,  // 全局变量遍历函数为 globals_traverse
    /* m_clear= */ globals_clear,  // 全局变量清除函数为 globals_clear
    /* m_free= */ globals_free};  // 全局变量释放函数为 globals_free

} // namespace

// Python 模块初始化函数，名为 PyInit__uarray
PyMODINIT_FUNC PyInit__uarray(void) {

  auto m = py_ref::steal(PyModule_Create(&uarray_module));  // 创建 Python 模块对象并引用
  if (!m)
    return nullptr;  // 如果创建失败，返回空指针

  if (PyType_Ready(&FunctionType) < 0)
    return nullptr;  // 如果初始化 FunctionType 类型失败，返回空指针
  Py_INCREF(&FunctionType);  // 增加 FunctionType 的引用计数
  PyModule_AddObject(m.get(), "_Function", (PyObject *)&FunctionType);  // 将 FunctionType 类型添加到模块中

  if (PyType_Ready(&SetBackendContextType) < 0)
    return nullptr;  // 如果初始化 SetBackendContextType 类型失败，返回空指针
  Py_INCREF(&SetBackendContextType);  // 增加 SetBackendContextType 的引用计数
  PyModule_AddObject(
      m.get(), "_SetBackendContext", (PyObject *)&SetBackendContextType);  // 将 SetBackendContextType 类型添加到模块中

  if (PyType_Ready(&SkipBackendContextType) < 0)
    return nullptr;  // 如果初始化 SkipBackendContextType 类型失败，返回空指针
  Py_INCREF(&SkipBackendContextType);  // 增加 SkipBackendContextType 的引用计数
  PyModule_AddObject(
      m.get(), "_SkipBackendContext", (PyObject *)&SkipBackendContextType);  // 将 SkipBackendContextType 类型添加到模块中

  if (PyType_Ready(&BackendStateType) < 0)
    return nullptr;  // 如果初始化 BackendStateType 类型失败，返回空指针
  Py_INCREF(&BackendStateType);  // 增加 BackendStateType 的引用计数
  PyModule_AddObject(m.get(), "_BackendState", (PyObject *)&BackendStateType);  // 将 BackendStateType 类型添加到模块中

  // 创建 BackendNotImplementedError 异常对象
  BackendNotImplementedError = py_ref::steal(PyErr_NewExceptionWithDoc(
      "uarray.BackendNotImplementedError",
      "An exception that is thrown when no compatible"
      " backend is found for a method.",
      PyExc_NotImplementedError, nullptr));
  if (!BackendNotImplementedError)
    return nullptr;  // 如果创建失败，返回空指针
  Py_INCREF(BackendNotImplementedError.get());  // 增加 BackendNotImplementedError 的引用计数
  PyModule_AddObject(
      m.get(), "BackendNotImplementedError", BackendNotImplementedError.get());  // 将 BackendNotImplementedError 添加到模块中

  // 初始化标识符集合 identifiers
  if (!identifiers.init())
    return nullptr;  // 如果初始化失败，返回空指针

  return m.release();  // 返回 Python 模块对象
}
```