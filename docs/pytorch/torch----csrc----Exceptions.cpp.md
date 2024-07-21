# `.\pytorch\torch\csrc\Exceptions.cpp`

```py
# 引入所需的头文件和库
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/python_headers.h>

#include <array>
#include <cstdarg>
#include <exception>
#include <utility>

#include <fmt/format.h>
#include <torch/csrc/THP.h>

#include <c10/util/StringUtil.h>

# 定义一系列的异常对象，用于在 Python 中表示不同的异常类型
PyObject *THPException_FatalError, *THPException_LinAlgError,
    *THPException_OutOfMemoryError, *THPException_DistError,
    *THPException_DistBackendError, *THPException_DistNetworkError,
    *THPException_DistStoreError;

# 定义一个宏，用于在条件不满足时直接返回 false
#define ASSERT_TRUE(cond) \
  if (!(cond))            \
  return false

# 初始化异常对象，将它们添加到给定的 Python 模块中
bool THPException_init(PyObject* module) {
  // NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
  ASSERT_TRUE(
      THPException_FatalError =
          PyErr_NewException("torch.FatalError", nullptr, nullptr));
  
  // 将 torch.FatalError 异常对象添加到给定的 Python 模块中
  // NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
  ASSERT_TRUE(
      PyModule_AddObject(module, "FatalError", THPException_FatalError) == 0);

  // 设置异常类的文档字符串，因为在修改错误类的 tp_doc 时，_add_docstr 会抛出 malloc 错误
  // NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
  ASSERT_TRUE(
      THPException_LinAlgError = PyErr_NewExceptionWithDoc(
          "torch._C._LinAlgError",
          "Error raised by torch.linalg function when the cause of error is a numerical inconsistency in the data.\n \
For example, you can the torch.linalg.inv function will raise torch.linalg.LinAlgError when it finds that \
a matrix is not invertible.\n \
\n\
Example:\n \
>>> # xdoctest: +REQUIRES(env:TORCH_DOCKTEST_LAPACK)\n \
>>> matrix = torch.eye(3, 3)\n \
>>> matrix[-1, -1] = 0\n \
>>> matrix\n \
    tensor([[1., 0., 0.],\n \
            [0., 1., 0.],\n \
            [0., 0., 0.]])\n \
>>> torch.linalg.inv(matrix)\n \
Traceback (most recent call last):\n \
File \"<stdin>\", line 1, in <module>\n \
torch._C._LinAlgError: torch.linalg.inv: The diagonal element 3 is zero, the inversion\n \
",
          nullptr));

  // 返回初始化结果
  return true;
}
// NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
ASSERT_TRUE(
    THPException_OutOfMemoryError = PyErr_NewExceptionWithDoc(
        "torch.OutOfMemoryError",
        "Exception raised when device is out of memory",
        PyExc_RuntimeError,
        nullptr));
// 将 "torch.OutOfMemoryError" 定义为一个新的异常对象，并添加文档说明
PyTypeObject* type = (PyTypeObject*)THPException_OutOfMemoryError;
type->tp_name = "torch.OutOfMemoryError";
ASSERT_TRUE(
    PyModule_AddObject(
        module, "OutOfMemoryError", THPException_OutOfMemoryError) == 0);

// NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
ASSERT_TRUE(
    THPException_DistError = PyErr_NewExceptionWithDoc(
        "torch.distributed.DistError",
        "Exception raised when an error occurs in the distributed library",
        PyExc_RuntimeError,
        nullptr));
// 将 "torch.distributed.DistError" 定义为一个新的异常对象，并添加文档说明
ASSERT_TRUE(
    PyModule_AddObject(module, "_DistError", THPException_DistError) == 0);

// NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
ASSERT_TRUE(
    THPException_DistBackendError = PyErr_NewExceptionWithDoc(
        "torch.distributed.DistBackendError",
        "Exception raised when a backend error occurs in distributed",
        THPException_DistError,
        nullptr));
// 将 "torch.distributed.DistBackendError" 定义为一个新的异常对象，继承自 "torch.distributed.DistError"，并添加文档说明
ASSERT_TRUE(
    PyModule_AddObject(
        module, "_DistBackendError", THPException_DistBackendError) == 0);

// NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
ASSERT_TRUE(
    THPException_DistNetworkError = PyErr_NewExceptionWithDoc(
        "torch.distributed.DistNetworkError",
        "Exception raised when a network error occurs in distributed",
        THPException_DistError,
        nullptr));
// 将 "torch.distributed.DistNetworkError" 定义为一个新的异常对象，继承自 "torch.distributed.DistError"，并添加文档说明
ASSERT_TRUE(
    PyModule_AddObject(
        module, "_DistNetworkError", THPException_DistNetworkError) == 0);

// NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
ASSERT_TRUE(
    THPException_DistStoreError = PyErr_NewExceptionWithDoc(
        "torch.distributed.DistStoreError",
        "Exception raised when an error occurs in the distributed store",
        THPException_DistError,
        nullptr));
// 将 "torch.distributed.DistStoreError" 定义为一个新的异常对象，继承自 "torch.distributed.DistError"，并添加文档说明
ASSERT_TRUE(
    PyModule_AddObject(
        module, "_DistStoreError", THPException_DistStoreError) == 0);

// 返回 true 表示成功完成异常定义和添加
return true;
// 定义一个静态函数，格式化传入的格式化字符串和参数列表为一个字符串消息
static std::string formatMessage(const char* format, va_list fmt_args) {
  // 定义错误缓冲区的大小为 1024 字节
  static const size_t ERROR_BUF_SIZE = 1024;
  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  // 使用指定大小的错误缓冲区来格式化字符串和参数列表
  char error_buf[ERROR_BUF_SIZE];
  vsnprintf(error_buf, ERROR_BUF_SIZE, format, fmt_args);

  // 确保字符串以空字符结尾
  error_buf[sizeof(error_buf) / sizeof(*error_buf) - 1] = 0;

  // 返回格式化后的字符串作为 std::string 类型
  return std::string(error_buf);
}

// 将 C++ 异常指针转换为 Python 异常处理
void translate_exception_to_python(const std::exception_ptr& e_ptr) {
  try {
    // 断言异常指针有效，否则抛出错误消息
    TORCH_INTERNAL_ASSERT(
        e_ptr,
        "translate_exception_to_python "
        "called with invalid exception pointer");
    // 重新抛出异常指针对应的异常
    std::rethrow_exception(e_ptr);
  }
  // 捕获所有类型的异常并返回
  CATCH_ALL_ERRORS(return)
}

// 构造函数，通过格式化字符串和参数列表初始化错误消息
TypeError::TypeError(const char* format, ...) {
  va_list fmt_args{};
  va_start(fmt_args, format);
  // 使用 formatMessage 函数格式化字符串和参数列表为错误消息
  msg = formatMessage(format, fmt_args);
  va_end(fmt_args);
}

// 构造函数，通过格式化字符串和参数列表初始化错误消息
AttributeError::AttributeError(const char* format, ...) {
  va_list fmt_args{};
  va_start(fmt_args, format);
  // 使用 formatMessage 函数格式化字符串和参数列表为错误消息
  msg = formatMessage(format, fmt_args);
  va_end(fmt_args);
}

// 处理 Python 警告的内部处理器的处理函数，将警告添加到警告缓冲区
void PyWarningHandler::InternalHandler::process(const c10::Warning& warning) {
  warning_buffer_.push_back(warning);
}

// PyWarningHandler 类的构造函数，设置警告处理程序为内部处理器
PyWarningHandler::PyWarningHandler() noexcept(true)
    : prev_handler_(c10::WarningUtils::get_warning_handler()),
      in_exception_(false) {
  c10::WarningUtils::set_warning_handler(&internal_handler_);
}

// 根据警告类型映射到相应的 Python 警告类型对象
PyObject* map_warning_to_python_type(const c10::Warning& warning) {
  // 定义一个访问者结构体，根据不同的警告类型返回相应的 Python 警告类型对象
  struct Visitor {
    PyObject* operator()(const c10::UserWarning&) const {
      return PyExc_UserWarning;
    }
    PyObject* operator()(const c10::DeprecationWarning&) const {
      return PyExc_DeprecationWarning;
    }
  };
  // 使用 std::visit 调用访问者来获取警告类型对应的 Python 警告类型对象
  return std::visit(Visitor(), warning.type());
}

/// See NOTE [ Conversion Cpp Python Warning ] for noexcept justification
/// NOLINTNEXTLINE(bugprone-exception-escape)
// PyWarningHandler 类的析构函数，用于释放资源和恢复之前的警告处理程序
PyWarningHandler::~PyWarningHandler() noexcept(false) {
  // 恢复之前的警告处理程序
  c10::WarningUtils::set_warning_handler(prev_handler_);
  // 获取内部处理器的警告缓冲区的引用
  auto& warning_buffer = internal_handler_.warning_buffer_;

  // 如果警告缓冲区非空，则处理 Python 异常上下文
  if (!warning_buffer.empty()) {
    PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
    pybind11::gil_scoped_acquire gil;
    auto result = 0;
    if (in_exception_) {
      // 如果在异常上下文中，则获取当前 Python 错误状态
      PyErr_Fetch(&type, &value, &traceback);
    }
    // 遍历警告缓冲区中的每个警告
    for (const auto& warning : warning_buffer) {
      // 获取警告的源位置信息
      auto source_location = warning.source_location();
      // 获取警告消息
      auto msg = warning.msg();
      // 在原地处理错误消息
      processErrorMsgInplace(msg);
      
      // 检查源文件是否为nullptr，选择不同的警告触发方式
      if (source_location.file == nullptr) {
        // 如果源文件为空，则使用 PyErr_WarnEx 发出警告，
        // 该函数会忽略 Python 的警告过滤器，始终显示警告
        result =
            PyErr_WarnEx(map_warning_to_python_type(warning), msg.c_str(), 1);
      } else if (warning.verbatim()) {
        // 如果警告标记为verbatim
        // 设置警告的源位置信息
        // 注意：PyErr_WarnExplicit 将忽略 Python 的警告过滤器，
        // 总是显示警告。这与 PyErr_WarnEx 相反，后者尊重警告过滤器。
        result = PyErr_WarnExplicit(
            /*category=*/map_warning_to_python_type(warning),
            /*message=*/msg.c_str(),
            /*filename=*/source_location.file,
            /*lineno=*/static_cast<int>(source_location.line),
            /*module=*/nullptr,
            /*registry=*/nullptr);
      } else {
        // 如果不是verbatim，允许 Python 设置源位置，并将 C++ 的警告位置信息放入消息中
        auto buf = fmt::format(
            "{} (Triggered internally at {}:{}.)",
            msg,
            source_location.file,
            source_location.line);
        result =
            PyErr_WarnEx(map_warning_to_python_type(warning), buf.c_str(), 1);
      }
      
      // 检查 PyErr_WarnEx 或 PyErr_WarnExplicit 的结果
      if (result < 0) {
        if (in_exception_) {
          // 如果在异常处理中，调用 PyErr_Print 将回溯打印到 sys.stderr 并清除错误指示器
          PyErr_Print();
        } else {
          // 否则跳出循环
          break;
        }
      }
    }
    
    // 清空警告缓冲区
    warning_buffer.clear();
    
    // 如果结果小于0并且不在异常处理中，抛出 python_error 异常
    if ((result < 0) && (!in_exception_)) {
      /// 发出警告引发了错误，需要强制父函数返回错误码。
      throw python_error();
    }
    
    // 如果在异常处理中，恢复之前的异常状态
    if (in_exception_) {
      PyErr_Restore(type, value, traceback);
    }
}

} // namespace torch
```