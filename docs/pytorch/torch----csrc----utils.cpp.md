# `.\pytorch\torch\csrc\utils.cpp`

```py
// 引入必要的头文件
#include <fmt/core.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/invalid_arguments.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/python_symnode.h>
#include <torch/csrc/utils/python_tuples.h>

#include <torch/csrc/Export.h>

#include <algorithm>  // 标准库：算法
#include <cstdarg>    // 标准库：可变参数列表
#include <cstring>    // 标准库：C风格字符串操作
#include <iterator>   // 标准库：迭代器
#include <sstream>    // 标准库：字符串流
#include <string>     // 标准库：字符串
#include <unordered_map>  // 标准库：无序映射
#include <utility>    // 标准库：实用工具
#include <vector>     // 标准库：向量

// 检查给定对象是否可调用，如果是则返回1，并将结果存储在result中
int THPUtils_getCallable(PyObject* arg, PyObject** result) {
  if (!PyCallable_Check(arg))  // 检查对象是否可调用
    return 0;
  *result = arg;  // 将可调用对象存储在result中
  return 1;  // 返回成功标志
}

// 检查对象是否可以用作索引
bool THPUtils_checkIndex(PyObject* obj) {
  if (PyBool_Check(obj)) {  // 如果对象是布尔类型
    return false;  // 返回false，不可用作索引
  }
  if (THPUtils_checkLong(obj)) {  // 如果对象是长整型
    return true;  // 返回true，可用作索引
  }
  // 避免提前使用__index__，因为这会立即引发保护
  if (torch::is_symint(py::handle(obj))) {  // 如果对象是符号整数
    return true;  // 返回true，可用作索引
  }
  torch::jit::tracer::NoWarn no_warn_guard;  // 跟踪器：无警告保护
  auto index = THPObjectPtr(PyNumber_Index(obj));  // 尝试获取对象的索引
  if (!index) {  // 如果获取失败
    PyErr_Clear();  // 清除Python错误状态
    return false;  // 返回false，不可用作索引
  }
  return true;  // 返回true，可用作索引
}

// 解包长整型列表或元组
std::vector<int64_t> THPUtils_unpackLongs(PyObject* arg) {
  bool tuple = PyTuple_Check(arg);  // 检查是否为元组
  bool list = PyList_Check(arg);    // 检查是否为列表
  if (tuple || list) {  // 如果是元组或列表
    const auto nDim = tuple ? PyTuple_GET_SIZE(arg) : PyList_GET_SIZE(arg);  // 获取元素数量
    std::vector<int64_t> sizes(nDim);  // 创建存储长整型的向量
    for (int i = 0; i != nDim; ++i) {  // 遍历元组或列表
      PyObject* item =  // 获取元组或列表中的元素
          tuple ? PyTuple_GET_ITEM(arg, i) : PyList_GET_ITEM(arg, i);
      if (!THPUtils_checkLong(item)) {  // 检查元素是否为长整型
        std::ostringstream oss;  // 创建字符串流对象
        oss << "expected int at position " << i
            << ", but got: " << THPUtils_typename(item);  // 构造错误信息
        throw std::runtime_error(oss.str());  // 抛出运行时错误
      }
      sizes[i] = THPUtils_unpackLong(item);  // 解包并存储长整型值
    }
    return sizes;  // 返回长整型向量
  }
  throw std::runtime_error("Expected tuple or list");  // 如果不是元组或列表，抛出运行时错误
}

// 检查对象是否为整数元组
bool THPUtils_checkIntTuple(PyObject* arg) {
  if (!PyTuple_Check(arg)) {  // 如果不是元组
    return false;  // 返回false
  }
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(arg); ++i) {  // 遍历元组中的每个元素
    if (!THPUtils_checkLong(PyTuple_GET_ITEM(arg, i))) {  // 检查元素是否为长整型
      return false;  // 如果有一个元素不是长整型，返回false
    }
  }
  return true;  // 所有元素都是长整型，返回true
}

// 解包整数元组
std::vector<int> THPUtils_unpackIntTuple(PyObject* arg) {
  if (!THPUtils_checkIntTuple(arg)) {  // 检查是否为整数元组
    throw std::runtime_error("Couldn't unpack int tuple");  // 如果不是整数元组，抛出运行时错误
  }
  std::vector<int> values(PyTuple_GET_SIZE(arg));  // 创建存储整数的向量
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(arg); ++i) {  // 遍历元组中的每个元素
    values[i] = (int)THPUtils_unpackLong(PyTuple_GET_ITEM(arg, i));  // 解包并存储整数值
  }
  return values;  // 返回整数向量
}

// 设置Python运行时错误信息
void THPUtils_setError(const char* format, ...) {
  static const size_t ERROR_BUFFER_SIZE = 1000;  // 错误信息缓冲区大小
  char buffer[ERROR_BUFFER_SIZE];  // 创建错误信息缓冲区
  va_list fmt_args;  // 声明可变参数列表

  va_start(fmt_args, format);  // 开始可变参数列表
  vsnprintf(buffer, ERROR_BUFFER_SIZE, format, fmt_args);  // 格式化错误信息
  va_end(fmt_args);  // 结束可变参数列表
  PyErr_SetString(PyExc_RuntimeError, buffer);  // 设置Python运行时错误
}

// 添加Python方法定义
void THPUtils_addPyMethodDefs(
    std::vector<PyMethodDef>& vector,  // 传入的向量的引用，用于存储 PyMethodDef 结构体
    PyMethodDef* methods) {           // 指向 PyMethodDef 结构体数组的指针
  if (!vector.empty()) {              // 检查向量是否非空
    // 移除末尾的 nullptr 终止符
    vector.pop_back();
  }
  while (true) {                      // 无限循环，直至条件 break 触发退出
    vector.push_back(*methods);       // 将当前指针指向的 PyMethodDef 结构体添加到向量中
    if (!methods->ml_name) {          // 检查当前 PyMethodDef 结构体的 ml_name 是否为空
      break;                          // 如果为空，退出循环
    }
    methods++;                        // 指针向后移动，指向下一个 PyMethodDef 结构体
  }
}
}

// 返回给定对象的类名或类型名字符串
static const char* classOrTypename(PyObject* obj) {
  // 如果对象是类型对象，则返回其类型名
  if (PyType_Check(obj)) {
    return ((PyTypeObject*)obj)->tp_name;
  }
  // 否则返回对象的类型的类型名
  return Py_TYPE(obj)->tp_name;
}

// 调度无状态方法的实现
PyObject* THPUtils_dispatchStateless(
    PyObject* tensor,
    const char* name,
    PyObject* args,
    PyObject* kwargs) {
  // 获取张量对象的 stateless 方法集合
  THPObjectPtr methods(
      PyObject_GetAttrString(tensor, THP_STATELESS_ATTRIBUTE_NAME));
  // 如果方法集合获取失败，则返回类型错误异常
  if (!methods) {
    return PyErr_Format(
        PyExc_TypeError,
        "Type %s doesn't implement stateless methods",
        classOrTypename(tensor));
  }
  // 获取指定名称的 stateless 方法对象
  THPObjectPtr method(PyObject_GetAttrString(methods, name));
  // 如果方法对象获取失败，则返回类型错误异常
  if (!method) {
    return PyErr_Format(
        PyExc_TypeError,
        "Type %s doesn't implement stateless method %s",
        classOrTypename(tensor),
        name);
  }
  // 调用方法对象，传递给定的参数和关键字参数
  return PyObject_Call(method.get(), args, kwargs);
}

// 抛出无效参数类型错误
void THPUtils_invalidArguments(
    PyObject* given_args,
    PyObject* given_kwargs,
    const char* function_name,
    size_t num_options,
    ...) {
  // 创建选项字符串向量
  std::vector<std::string> option_strings;
  va_list option_list;
  va_start(option_list, num_options);
  // 使用可变参数列表生成选项字符串向量
  std::generate_n(
      std::back_inserter(option_strings), num_options, [&option_list] {
        return va_arg(option_list, const char*);
      });
  va_end(option_list);

  // 设置类型错误异常，使用 Torch 格式化无效参数
  PyErr_SetString(
      PyExc_TypeError,
      torch::format_invalid_args(
          given_args, given_kwargs, function_name, option_strings)
          .c_str());
}

// THPGenerator 类型的 THPPointer 的 free 方法特化实现
template <>
void THPPointer<THPGenerator>::free() {
  // 如果指针不为空，减少其引用计数
  if (ptr)
    Py_DECREF(ptr);
}

// 实例化 THPGenerator 类型的 THPPointer
template class THPPointer<THPGenerator>;

// 是否显示后向兼容广播警告标志
static bool backCompatBroadcastWarn = false;

// 设置后向兼容广播警告标志
void setBackCompatBroadcastWarn(bool warn) {
  backCompatBroadcastWarn = warn;
}

// 获取后向兼容广播警告标志
bool getBackCompatBroadcastWarn() {
  return backCompatBroadcastWarn;
}

// 是否显示后向兼容 keepdim 警告标志
static bool backCompatKeepdimWarn = false;

// 设置后向兼容 keepdim 警告标志
void setBackCompatKeepdimWarn(bool warn) {
  backCompatKeepdimWarn = warn;
}

// 获取后向兼容 keepdim 警告标志
bool getBackCompatKeepdimWarn() {
  return backCompatKeepdimWarn;
}

// 可能抛出后向兼容 keepdim 警告
bool maybeThrowBackCompatKeepdimWarn(char* func) {
  // 如果设置了后向兼容 keepdim 警告，则打印警告信息
  if (getBackCompatKeepdimWarn()) {
    std::ostringstream ss;
    ss << "backwards compatibility: call to \"" << func
       << "\" uses default value for keepdim which has changed default to False.  Consider passing as kwarg.";
    // 发出用户警告，打印警告信息
    PyErr_WarnEx(PyExc_UserWarning, ss.str().c_str(), 1);
  }
  return true;
}

// THPStorage 类型的 THPPointer 的 free 方法特化实现
template <>
void THPPointer<THPStorage>::free() {
  // 如果指针不为空，减少其引用计数
  if (ptr)
    Py_DECREF(ptr);
}

// 填充存储区域中的所有元素为指定值
void storage_fill(const at::Storage& self, uint8_t value) {
  // 创建与 self 相同设备和数据类型的选项
  auto options = c10::TensorOptions().device(self.device()).dtype(at::kByte);
  // 创建空张量并关联到 self
  auto self_t = at::empty({0}, options).set_(self);
  // 填充张量所有元素为指定值
  self_t.fill_(value);
}

// 设置存储区域中指定索引的元素为指定值
void storage_set(const at::Storage& self, ptrdiff_t idx, uint8_t value) {
  // 检查索引是否有效
  TORCH_CHECK(
      (idx >= 0) && (idx < static_cast<ptrdiff_t>(self.nbytes())),
      "out of bounds");
  // 创建与 self 相同设备和数据类型的选项
  auto options = c10::TensorOptions().device(self.device()).dtype(at::kByte);
  // 创建空张量并关联到 self
  auto self_t = at::empty({0}, options).set_(self);
  // 设置张量指定索引的元素为指定值
  self_t[idx].fill_(value);
}
// 获取存储器中指定索引处的值，并确保索引在有效范围内
uint8_t storage_get(const at::Storage& self, ptrdiff_t idx) {
  TORCH_CHECK(
      (idx >= 0) && (idx < static_cast<ptrdiff_t>(self.nbytes())),
      "out of bounds");
  // 创建一个新的空张量，使用与原张量相同的设备和数据类型选项，并将原始存储器数据填充到其中
  auto options = c10::TensorOptions().device(self.device()).dtype(at::kByte);
  auto self_t = at::empty({0}, options).set_(self);
  // 返回指定索引处的值，转换为 uint8_t 类型
  return self_t[idx].item<uint8_t>();
}

// 实例化模板类 THPPointer<THPStorage>
template class THPPointer<THPStorage>;

namespace torch::gdb {
/* ~~~ misc debugging utilities ~~~
 *
 * torch::gdb::* functions are NOT meant to be called by general pytorch code,
 * but only from within a gdb session. As such, utils.h does not contain any
 * declaration for those.
 */

// 用于 torch-tensor-repr gdb 命令的辅助函数
// 返回给定张量的人类可读表示。结果存储在 malloc() 分配的缓冲区中，调用者负责释放它。
// 由于计算张量的表示的代码当前在 Python 中编写，因此我们需要先将张量包装到 Python 对象中。
char* tensor_repr(at::Tensor tensor) {
  PyGILState_STATE gil = PyGILState_Ensure();
  PyObject* pytensor = nullptr;
  PyObject* repr = nullptr;
  Py_ssize_t bufsize = 0;
  const char* buf = nullptr;
  char* result = nullptr;

  pytensor = THPVariable_Wrap(std::move(tensor));
  if (!pytensor)
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    goto error;
  repr = PyObject_Repr(pytensor);
  if (!repr)
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    goto error;
  buf = PyUnicode_AsUTF8AndSize(repr, &bufsize);
  if (!buf)
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    goto error;
  // 分配内存以存储结果，并复制缓冲区内容到结果中
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  result = static_cast<char*>(malloc(bufsize + 1));
  if (!result) {
    fmt::print(stderr, "cannot allocate memory for the result\n");
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    goto error;
  }
  std::strncpy(result, buf, bufsize);
  result[bufsize] = '\0';
  Py_XDECREF(pytensor);
  Py_XDECREF(repr);
  PyGILState_Release(gil);
  return result;

error:
  fprintf(stderr, "torch::gdb::tensor_repr: unexpected error\n");
  if (PyErr_Occurred())
    PyErr_Print();
  Py_XDECREF(pytensor);
  Py_XDECREF(repr);
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  free(result);
  PyGILState_Release(gil);
  return nullptr;
}

// 将 IntArrayRef 转换为字符串表示
std::string int_array_ref_string(at::IntArrayRef sizes) {
  std::stringstream ss;
  ss << sizes;
  return ss.str();
}

// 将 DispatchKeySet 转换为字符串表示
std::string dispatch_keyset_string(c10::DispatchKeySet keyset) {
  std::stringstream ss;
  ss << keyset;
  return ss.str();
}

} // namespace torch::gdb

namespace pybind11::detail {

// pybind11 的类型转换器，用于加载 at::Tensor 对象
bool type_caster<at::Tensor>::load(handle src, bool) {
  PyObject* obj = src.ptr();
  if (THPVariable_Check(obj)) {
    // 如果是 THPVariable 类型，则解包为 at::Tensor 对象
    value = THPVariable_Unpack(obj);
    return true;
  }
  return false;
}
handle type_caster<at::Tensor>::cast(
    const at::Tensor& src,
    return_value_policy /* policy */,
    handle /* parent */) {
  // 将给定的 Tensor 包装为 Python 对象，并返回其对应的 handle
  return handle(THPVariable_Wrap(src));
}

bool type_caster<at::IntArrayRef>::load(handle src, bool) {
  PyObject* source = src.ptr();
  auto tuple = PyTuple_Check(source);
  if (tuple || PyList_Check(source)) {
    // NOLINTNEXTLINE(bugprone-branch-clone)
    // 检查 source 是 tuple 还是 list，计算其长度
    const auto size =
        tuple ? PyTuple_GET_SIZE(source) : PyList_GET_SIZE(source);
    // 调整 v_value 的大小以容纳元素数量为 size
    v_value.resize(size);
    // 遍历 tuple 或 list 中的元素
    for (const auto idx : c10::irange(size)) {
      PyObject* obj =
          tuple ? PyTuple_GET_ITEM(source, idx) : PyList_GET_ITEM(source, idx);
      // 如果元素是 THPVariable 对象，则解包并转换为 int64_t 类型存入 v_value
      if (THPVariable_Check(obj)) {
        v_value[idx] = THPVariable_Unpack(obj).item<int64_t>();
      // 如果元素是 Python 的长整型对象，则使用 THPUtils_unpackLong 解包并存入 v_value
      } else if (PyLong_Check(obj)) {
        // use THPUtils_unpackLong after it is safe to include
        // python_numbers.h
        v_value[idx] = THPUtils_unpackLong(obj);
      // 如果元素类型不符合预期，则返回 false
      } else {
        return false;
      }
    }
    // 将 v_value 赋值给 value，表示加载成功，并返回 true
    value = v_value;
    return true;
  }
  // 如果 source 不是 tuple 或 list，则直接返回 false
  return false;
}

handle type_caster<at::IntArrayRef>::cast(
    at::IntArrayRef src,
    return_value_policy /* policy */,
    handle /* parent */) {
  // 将 at::IntArrayRef 转换为 Python 对象，并返回其对应的 handle
  return handle(THPUtils_packInt64Array(src.size(), src.data()));
}

bool type_caster<at::SymIntArrayRef>::load(handle src, bool) {
  PyObject* source = src.ptr();

  auto tuple = PyTuple_Check(source);
  if (tuple || PyList_Check(source)) {
    // NOLINTNEXTLINE(bugprone-branch-clone)
    // 检查 source 是 tuple 还是 list，计算其长度
    const auto size =
        tuple ? PyTuple_GET_SIZE(source) : PyList_GET_SIZE(source);
    // 调整 v_value 的大小以容纳元素数量为 size
    v_value.resize(size);
    // 遍历 tuple 或 list 中的元素
    for (const auto idx : c10::irange(size)) {
      PyObject* obj =
          tuple ? PyTuple_GET_ITEM(source, idx) : PyList_GET_ITEM(source, idx);

      // 如果元素是 THPVariable 对象，则解包并转换为 int64_t 存入 v_value
      if (THPVariable_Check(obj)) {
        // TODO: this is for consistency with IntArrayRef but arguably
        // we shouldn't really allow this on pybind11 casters
        v_value[idx] = THPVariable_Unpack(obj).item<int64_t>();
      // 如果元素是 torch 的 SymInt 类型，则直接转换并存入 v_value
      } else if (torch::is_symint(py::handle(obj))) {
        v_value[idx] = py::handle(obj).cast<c10::SymInt>();
      // 如果元素是 Python 的长整型对象，则转换为 SymInt 类型并存入 v_value
      } else if (PyLong_Check(obj)) {
        v_value[idx] = c10::SymInt(THPUtils_unpackIndex(obj));
      // 如果元素类型不符合预期，则返回 false
      } else {
        return false;
      }
    }
    // 将 v_value 赋值给 value，表示加载成功，并返回 true
    value = v_value;
    return true;
  }
  // 如果 source 不是 tuple 或 list，则直接返回 false
  return false;
}

handle type_caster<at::SymIntArrayRef>::cast(
    at::SymIntArrayRef src,
    return_value_policy /* policy */,
    handle /* parent */) {
  // 将 at::SymIntArrayRef 转换为 Python 对象，并返回其对应的 handle
  py::list t(src.size());
  for (const auto i : c10::irange(src.size())) {
    t[i] = py::cast(src[i]);
  }
  return t.release();
}

bool type_caster<at::ArrayRef<c10::SymNode>>::load(handle src, bool) {
  // 抛出尚未实现的异常信息
  TORCH_INTERNAL_ASSERT(0, "NYI");
}

handle type_caster<at::ArrayRef<c10::SymNode>>::cast(
    at::ArrayRef<c10::SymNode> src,
    return_value_policy /* policy */,
    handle /* parent */) {
  // 将 at::ArrayRef<c10::SymNode> 转换为 Python 对象，并返回其对应的 handle
  py::list t(src.size());
  for (const auto i : c10::irange(src.size())) {
    // TODO: this is terrible but I don't know how to override when
    // ... (注释内容被截断，未完成)
    // 尝试将 src[i] 转换为 PythonSymNodeImpl 类型指针，如果转换成功，则 py_node 指向该对象
    auto* py_node = dynamic_cast<torch::impl::PythonSymNodeImpl*>(src[i].get());
    // 如果 py_node 不为空指针，则 src[i] 是 PythonSymNodeImpl 类型的对象
    if (py_node) {
      // 直接返回 Python 对象，通过 py_node 获取其内部的 Python 对象
      t[i] = py_node->getPyObj();
    } else {
      // 否则，将 src[i] 转换为 py::cast 返回的 Python 对象
      t[i] = py::cast(src[i]);
    }
  }
  // 释放 t 的所有权，返回其底层的原始指针
  return t.release();
}

} // namespace pybind11::detail
```