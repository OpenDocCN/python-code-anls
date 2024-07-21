# `.\pytorch\torch\csrc\Size.cpp`

```py
// 引入所需头文件：用于迭代、类型操作和绑定 Python 对象
#include <c10/util/irange.h>
#include <pybind11/pytypes.h>
#include <torch/csrc/Size.h>
#include <torch/csrc/utils/pybind.h>

// 引入必要的 Torch C++ 工具类和函数
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/python_tuples.h>
#include <string>

// 引入 Torch 自动求导和跟踪相关的头文件
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/jit/frontend/tracer.h>

// 定义一个结构体 THPSize，用于表示大小信息
struct THPSize {
  PyTupleObject tuple;
};

// 根据给定的 torch::autograd::Variable 对象创建一个 Python 对象
PyObject* THPSize_New(const torch::autograd::Variable& var) {
  // 如果没有进行 Torch 的追踪，则获取变量的大小信息并创建 THPSize 对象
  if (!torch::jit::tracer::isTracing()) {
    auto sizes = var.sizes();
    return THPSize_NewFromSizes(var.dim(), sizes.data());
  }
  
  // 如果正在进行 Torch 的追踪，则基于追踪信息创建 THPSize 对象
  auto self = THPObjectPtr(THPSizeType.tp_alloc(&THPSizeType, var.dim()));
  if (!self)
    throw python_error();

  // 遍历变量的每个维度，获取其大小并封装成 Python 对象
  for (const auto i : c10::irange(var.dim())) {
    PyObject* py_size_tensor =
        THPVariable_Wrap(torch::jit::tracer::getSizeOf(var, i));
    if (!py_size_tensor)
      throw python_error();
    PyTuple_SET_ITEM(self.get(), i, py_size_tensor);
  }

  return self.release();
}

// 根据给定的维度和大小数组创建 THPSize 对象
PyObject* THPSize_NewFromSizes(int64_t dim, const int64_t* sizes) {
  auto self = THPObjectPtr(THPSizeType.tp_alloc(&THPSizeType, dim));
  if (!self)
    throw python_error();
  THPUtils_packInt64Array(self, dim, sizes);
  return self.release();
}

// 根据给定的 at::Tensor 对象的符号大小信息创建 THPSize 对象
PyObject* THPSize_NewFromSymSizes(const at::Tensor& self_) {
  auto sym_sizes = self_.sym_sizes();

  // 根据符号大小的个数分配 Python 元组对象的空间
  auto ret = THPObjectPtr(THPSizeType.tp_alloc(
      &THPSizeType, static_cast<Py_ssize_t>(sym_sizes.size())));
  if (!ret)
    throw python_error();

  // 遍历每个符号大小，根据情况创建相应的 Python 对象
  for (auto i : c10::irange(sym_sizes.size())) {
    auto si = sym_sizes[i];
    if (si.is_symbolic()) {
      // 如果是符号值，则检查是否在 Torch 追踪模式下，抛出错误提示追踪符号整数不支持
      TORCH_CHECK(
          !torch::jit::tracer::isTracing(),
          "JIT Tracing of SymInts isn't supported");
      auto py_symint = py::cast(si).release().ptr();
      if (!py_symint)
        throw python_error();
      PyTuple_SET_ITEM(ret.get(), i, py_symint);
    } else {
      // 如果是整数值，根据追踪模式决定如何封装
      auto m = si.maybe_as_int();
      if (torch::jit::tracer::isTracing()) {
        PyObject* py_size_tensor = THPVariable_Wrap(
            torch::jit::tracer::getSizeOf(self_, static_cast<int64_t>(i)));
        if (!py_size_tensor)
          throw python_error();
        PyTuple_SET_ITEM(ret.get(), i, py_size_tensor);
      } else {
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        PyTuple_SET_ITEM(ret.get(), i, THPUtils_packInt64(*m));
      }
    }
  }
  return ret.release();
}

// 检查给定的 Python 对象是否为追踪的零维变量
static bool isTracedZeroDimVar(PyObject* item) {
  // 如果不是 THPVariable 类型，则返回 false
  if (!THPVariable_Check(item))
    return false;
  // 解包并获取 Torch 变量对象
  auto& var = THPVariable_Unpack(item);
  // 返回判断结果：是否为零维且被追踪
  return var.dim() == 0 && torch::jit::tracer::getValueTrace(var);
}

// 定义 Python 类型 THPSize 的构造函数
static PyObject* THPSize_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
      // 处理异常
      HANDLE_TH_ERRORS
      // 使用 PyTuple_Type.tp_new 创建一个新的 PyTuple 对象
      THPObjectPtr self(PyTuple_Type.tp_new(type, args, kwargs));
      // 如果成功创建了 PyTuple 对象
      if (self) {
        // 遍历 PyTuple 对象的每个元素
        for (Py_ssize_t i = 0; i < PyTuple_Size(self); ++i) {
          // 获取 PyTuple 对象中索引为 i 的元素
          PyObject* item = PyTuple_GET_ITEM(self.get(), i);
          // 检查元素是否为长整型
          if (THPUtils_checkLong(item)) {
            // 如果是长整型，继续处理下一个元素
            continue;
          }
          // 检查元素是否为 torch 的符号整型
          if (torch::is_symint(item)) {
            // 如果是符号整型，继续处理下一个元素
            continue;
          }
          // 如果正在追踪 torch 的运行，并且元素是被追踪的零维变量
          if (torch::jit::tracer::isTracing() && isTracedZeroDimVar(item)) {
            // 继续处理下一个元素
            continue;
          }
          // 尝试将元素转换为索引，支持处理零维张量和只包含一个元素的张量
          THPObjectPtr number(PyNumber_Index(item));
          // 如果成功转换，并且转换后的结果是长整型
          if (number && THPUtils_checkLong(number.get())) {
            // 增加结果的引用计数
            Py_INCREF(number.get());
            // 设置 PyTuple 对象中索引为 i 的元素为转换后的结果
            auto status = PyTuple_SetItem(self, i, number.get());
            // 如果设置失败，抛出异常
            if (status != 0) {
              throw python_error();
            }
            // 继续处理下一个元素
            continue;
          }
          // 如果元素不能转换为长整型，返回类型错误异常
          return PyErr_Format(
              PyExc_TypeError,
              "torch.Size() takes an iterable of 'int' (item %zd is '%s')",
              i,
              Py_TYPE(item)->tp_name);
        }
      }
      // 返回 PyTuple 对象的所有权
      return self.release();
      // 结束异常处理
      END_HANDLE_TH_ERRORS
    }
// 定义 THPSize_repr 函数，用于返回 THPSize 对象的字符串表示
static PyObject* THPSize_repr(THPSize* self) {
  HANDLE_TH_ERRORS
  // 初始化 repr 字符串，表示 torch.Size 对象
  std::string repr("torch.Size([");
  // 遍历 THPSize 对象中的每个元素
  for (Py_ssize_t i = 0; i < PyTuple_Size((PyObject*)self); ++i) {
    // 如果不是第一个元素，添加逗号和空格
    if (i != 0) {
      repr += ", ";
    }
    // 获取元组中的每个元素
    auto item = PyTuple_GET_ITEM(self, i);
    // 将元素包装为 Python 对象的句柄
    auto ih = py::handle(item);

    // 如果元素是符号整数，转换为字符串后添加到 repr 中；否则将其转换为长整型
    repr += torch::is_symint(ih)
        ? std::string(py::str(ih))
        : std::to_string(THPUtils_unpackLong(PyTuple_GET_ITEM(self, i)));
  }
  // 结束拼接 repr 字符串
  repr += "])";
  // 返回字符串表示，转换为 Python 对象
  return THPUtils_packString(repr);
  END_HANDLE_TH_ERRORS
}

// 匿名命名空间，用于定义与 PyTuple_Type 相关的函数指针
namespace {
// 获取 PyTuple_Type 序列类型的 sq_concat 函数指针
auto sq_concat = PyTuple_Type.tp_as_sequence->sq_concat;
// 获取 PyTuple_Type 序列类型的 sq_repeat 函数指针
auto sq_repeat = PyTuple_Type.tp_as_sequence->sq_repeat;
// 获取 PyTuple_Type 映射类型的 mp_subscript 函数指针
binaryfunc mp_subscript = PyTuple_Type.tp_as_mapping->mp_subscript;
} // namespace

// 定义 THPSize_as_sequence 结构体，实现序列操作方法
static PySequenceMethods THPSize_as_sequence = {
    nullptr, /* sq_length */
    // 包装 sq_concat 函数并调用，返回新的 Python 对象
    wrap_tuple_fn<decltype(&sq_concat), &sq_concat>,
    // 包装 sq_repeat 函数并调用，返回新的 Python 对象
    wrap_tuple_fn<decltype(&sq_repeat), &sq_repeat>,
    nullptr, /* sq_item */
    nullptr, /* sq_slice */
    nullptr, /* sq_ass_item */
    nullptr, /* sq_ass_slice */
    nullptr /* sq_contains */
};

// 定义 THPSize_as_mapping 结构体，实现映射操作方法
static PyMappingMethods THPSize_as_mapping = {
    nullptr, /* mp_length */
    // 包装 mp_subscript 函数并调用，返回新的 Python 对象
    wrap_tuple_fn<decltype(&mp_subscript), &mp_subscript>,
    nullptr};

// 定义 THPSize_numel 函数，返回 THPSize 对象的元素个数
static PyObject* THPSize_numel(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 将输入的 Python 对象转换为 THPSize 对象
  auto self = (THPSize*)_self;
  // 初始化元素个数为 1
  int64_t numel = 1;
  // 遍历 THPSize 对象中的每个元素
  for (Py_ssize_t i = 0; i < PyTuple_Size((PyObject*)self); ++i) {
    // 计算所有元素的乘积
    numel *= THPUtils_unpackLong(PyTuple_GET_ITEM(self, i));
  }
  // 将计算结果打包为 Python 整型对象并返回
  return THPUtils_packInt64(numel);
  END_HANDLE_TH_ERRORS
}

// 定义 THPSize_reduce 函数，用于序列化 THPSize 对象
static PyObject* THPSize_reduce(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 将输入的 Python 对象转换为 THPSize 对象
  auto self = (THPSize*)_self;
  // 创建一个长度为 2 的新元组对象 ret
  auto ret = THPObjectPtr{PyTuple_New(2)};
  if (!ret)
    throw python_error();

  // 在 ret 中设置第一个元素为 THPSizeType 类型对象的引用
  auto obj = (PyObject*)(&THPSizeType);
  Py_INCREF(&THPSizeType);
  PyTuple_SET_ITEM(ret.get(), 0, obj);

  // 创建一个新的元组对象 t，与 self 中相同长度
  THPObjectPtr t(PyTuple_New(PyTuple_Size((PyObject*)self)));
  if (!t)
    throw python_error();
  // 遍历 self 中的每个元素，逐个添加到 t 中
  for (Py_ssize_t i = 0; i < PyTuple_Size((PyObject*)self); ++i) {
    auto d = PyTuple_GET_ITEM(self, i);
    Py_INCREF(d);
    PyTuple_SET_ITEM(t.get(), i, d);
  }

  // 将 t 打包为元组，作为 ret 的第二个元素
  THPObjectPtr dims(Py_BuildValue("(O)", t.get()));
  if (!dims)
    throw python_error();
  PyTuple_SET_ITEM(ret.get(), 1, dims.release());

  // 返回序列化结果 ret
  return ret.release();
  END_HANDLE_TH_ERRORS
}

// 定义 THPSize_methods 数组，包含 THPSize 的所有方法定义
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
static PyMethodDef THPSize_methods[] = {
    {"numel", THPSize_numel, METH_NOARGS, nullptr},
    // 键为 "numel"，值为 THPSize_numel 函数指针，表示无参数方法
    {"__reduce__", THPSize_reduce, METH_NOARGS, nullptr},
    // 键为 "__reduce__"，值为 THPSize_reduce 函数指针，表示无参数方法
    {nullptr}};
    // 结束符，表示这是一个以 nullptr 结尾的 C 结构体数组
PyTypeObject THPSizeType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch.Size", /* tp_name */
    sizeof(THPSize), /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    (reprfunc)THPSize_repr, /* tp_repr */
    nullptr, /* tp_as_number */
    &THPSize_as_sequence, /* tp_as_sequence */
    &THPSize_as_mapping, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    THPSize_methods, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    &PyTuple_Type, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THPSize_pynew, /* tp_new */
};

void THPSize_init(PyObject* module) {
    // 准备 THPSizeType 类型对象，如果失败则抛出异常
    if (PyType_Ready(&THPSizeType) < 0) {
        throw python_error();
    }
    // 增加类型对象的引用计数，避免被意外释放
    Py_INCREF(&THPSizeType);
    // 将类型对象添加到给定的 Python 模块中，如果失败则抛出异常
    if (PyModule_AddObject(module, "Size", (PyObject*)&THPSizeType) < 0) {
        throw python_error();
    }
}
```