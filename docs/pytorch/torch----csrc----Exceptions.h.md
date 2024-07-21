# `.\pytorch\torch\csrc\Exceptions.h`

```py
/// NOTE [ Conversion Cpp Python Warning ]
/// 警告处理程序无法立即设置 Python 警告，因为需要获取 GIL（可能死锁），
/// 如果警告引发 Python 错误，则需要干净地退出。因此，我们缓冲警告并在返回到 Python 时处理它们。
/// 以下两个 try/catch 块处理以下情况：
///   - 如果内部 try/catch 没有引发错误，则缓冲的警告将作为 Python 警告处理。
///     - 如果它们没有引发错误，则函数将继续处理原始的返回代码。
///     - 如果其中任何一个引发错误，则设置错误（PyErr_*），析构函数将引发 cpp 异常 python_error()，
///       外部 try/catch 将能够修改函数的返回值以反映错误。
///   - 如果内部 try/catch 引发了错误，则内部 try/catch 必须设置 Python 错误。
///     然后，缓冲的警告将作为 cpp 警告处理，因为我们无法预测 Python 警告是否会引发错误，
///     并且我们不能同时处理两个错误。
/// 此高级处理程序仅在当前线程中使用。如果使用其他线程，则警告将作为 cpp 警告处理。
#define HANDLE_TH_ERRORS                              \
  try {                                               \
    torch::PyWarningHandler __enforce_warning_buffer; \
    try {

// 只捕获 torch 特定的异常
#define _CATCH_GENERIC_ERROR(ErrorType, PythonErrorType, retstmnt) \
  catch (const c10::ErrorType& e) {                                \
    auto msg = torch::get_cpp_stacktraces_enabled()                \
        ? e.what()                                                 \
        : e.what_without_backtrace();                              \
    PyErr_SetString(PythonErrorType, torch::processErrorMsg(msg)); \
    retstmnt;                                                      \
  }

// 只捕获核心的 torch 异常
#define CATCH_CORE_ERRORS(retstmnt)                                           \
  catch (python_error & e) {                                                  \
    e.restore();                                                              \
    retstmnt;                                                                 \
  }                                                                           \
  catch (py::error_already_set & e) {                                         \
    e.restore();                                                              \
    retstmnt;                                                                 \
  }                                                                           \
  _CATCH_GENERIC_ERROR(IndexError, PyExc_IndexError, retstmnt)                \
  _CATCH_GENERIC_ERROR(ValueError, PyExc_ValueError, retstmnt)                \
  _CATCH_GENERIC_ERROR(TypeError, PyExc_TypeError, retstmnt)                  \
  _CATCH_GENERIC_ERROR(                                                       \
      NotImplementedError, PyExc_NotImplementedError, retstmnt)               \
  _CATCH_GENERIC_ERROR(LinAlgError, THPException_LinAlgError, retstmnt)       \
  _CATCH_GENERIC_ERROR(                                                       \
      OutOfMemoryError, THPException_OutOfMemoryError, retstmnt)              \
  _CATCH_GENERIC_ERROR(                                                       \
      DistBackendError, THPException_DistBackendError, retstmnt)              \
  _CATCH_GENERIC_ERROR(                                                       \
      DistNetworkError, THPException_DistNetworkError, retstmnt)              \
  _CATCH_GENERIC_ERROR(DistStoreError, THPException_DistStoreError, retstmnt) \
  _CATCH_GENERIC_ERROR(DistError, THPException_DistError, retstmnt)           \
  _CATCH_GENERIC_ERROR(Error, PyExc_RuntimeError, retstmnt)                   \
  catch (torch::PyTorchError & e) {                                           \
    auto msg = torch::processErrorMsg(e.what());                              \
    PyErr_SetString(e.python_type(), msg);                                    \
    retstmnt;                                                                 \
  }



// 捕获常见的 C++ 异常并重新抛出对应的 Python 异常
retstmnt;                                                                 \
}                                                                           \
catch (py::error_already_set & e) {                                         \
  // 如果 Python C++ 扩展中有未处理的 Python 异常，则恢复并重新抛出
  e.restore();                                                              \
  retstmnt;                                                                 \
}                                                                           \
// 捕获其他常见异常并抛出对应的 Python 异常
_CATCH_GENERIC_ERROR(IndexError, PyExc_IndexError, retstmnt)                \
_CATCH_GENERIC_ERROR(ValueError, PyExc_ValueError, retstmnt)                \
_CATCH_GENERIC_ERROR(TypeError, PyExc_TypeError, retstmnt)                  \
_CATCH_GENERIC_ERROR(                                                       \
    NotImplementedError, PyExc_NotImplementedError, retstmnt)               \
_CATCH_GENERIC_ERROR(LinAlgError, THPException_LinAlgError, retstmnt)       \
_CATCH_GENERIC_ERROR(                                                       \
    OutOfMemoryError, THPException_OutOfMemoryError, retstmnt)              \
_CATCH_GENERIC_ERROR(                                                       \
    DistBackendError, THPException_DistBackendError, retstmnt)              \
_CATCH_GENERIC_ERROR(                                                       \
    DistNetworkError, THPException_DistNetworkError, retstmnt)              \
_CATCH_GENERIC_ERROR(DistStoreError, THPException_DistStoreError, retstmnt) \
_CATCH_GENERIC_ERROR(DistError, THPException_DistError, retstmnt)           \
_CATCH_GENERIC_ERROR(Error, PyExc_RuntimeError, retstmnt)                   \
catch (torch::PyTorchError & e) {                                           \
  // 捕获 PyTorch 异常，设置异常信息并重新抛出
  auto msg = torch::processErrorMsg(e.what());                              \
  PyErr_SetString(e.python_type(), msg);                                    \
  retstmnt;                                                                 \
}
# 定义宏，用于捕获 Torch 调用的核心错误，并返回指定的语句
#define CATCH_TH_ERRORS(retstmnt) CATCH_CORE_ERRORS(retstmnt)

# 定义宏，用于捕获所有错误，包括标准异常，将其转换为 Python 运行时错误并设置错误消息，然后返回指定的语句
#define CATCH_ALL_ERRORS(retstmnt)               \
  CATCH_TH_ERRORS(retstmnt)                      \
  catch (const std::exception& e) {              \
    auto msg = torch::processErrorMsg(e.what()); \
    PyErr_SetString(PyExc_RuntimeError, msg);    \
    retstmnt;                                    \
  }

# 定义宏，结束 Torch 异常处理块，如果捕获到未知异常，则设置警告缓冲区以指示异常已经在处理中，并且重新抛出异常
#define END_HANDLE_TH_ERRORS_PYBIND                                 \
  }                                                                 \
  catch (...) {                                                     \
    __enforce_warning_buffer.set_in_exception();                    \
    throw;                                                          \
  }                                                                 \
  }                                                                 \
  catch (py::error_already_set & e) {                               \
    throw;                                                          \
  }                                                                 \
  catch (py::builtin_exception & e) {                               \
    throw;                                                          \
  }                                                                 \
  catch (torch::jit::JITException & e) {                            \
    throw;                                                          \
  }                                                                 \
  catch (const std::exception& e) {                                 \
    torch::translate_exception_to_python(std::current_exception()); \
    throw py::error_already_set();                                  \
  }

# 定义宏，结束 Torch 异常处理块，如果捕获到未知异常，则设置警告缓冲区以指示异常已经在处理中，并返回指定的返回值
#define END_HANDLE_TH_ERRORS_RET(retval)                            \
  }                                                                 \
  catch (...) {                                                     \
    __enforce_warning_buffer.set_in_exception();                    \
    throw;                                                          \
  }                                                                 \
  }                                                                 \
  catch (const std::exception& e) {                                 \
    torch::translate_exception_to_python(std::current_exception()); \
    return retval;                                                  \
  }

# 定义宏，结束 Torch 异常处理块，如果捕获到未知异常，则设置警告缓冲区以指示异常已经在处理中，并返回空指针
#define END_HANDLE_TH_ERRORS END_HANDLE_TH_ERRORS_RET(nullptr)

# 声明外部变量，用于指向 Torch 异常类型的 Python 对象，例如致命错误、线性代数错误等
extern PyObject *THPException_FatalError, *THPException_LinAlgError,
    *THPException_OutOfMemoryError, *THPException_DistError,
    *THPException_DistBackendError, *THPException_DistNetworkError,
    *THPException_DistStoreError;

# 抛出此异常意味着 Python 错误标志已经设置，并且应立即返回控制权到解释器。
struct python_error : public std::exception {
  // 默认构造函数
  python_error() = default;

  // 拷贝构造函数
  python_error(const python_error& other)
      : type(other.type),
        value(other.value),
        traceback(other.traceback),
        message(other.message) {
    // 获取全局解释器锁
    pybind11::gil_scoped_acquire gil;
    // 增加引用计数，避免空指针异常
    Py_XINCREF(type);
    Py_XINCREF(value);
    Py_XINCREF(traceback);
  }

  // 移动构造函数
  python_error(python_error&& other) noexcept
      : type(other.type),
        value(other.value),
        traceback(other.traceback),
        message(std::move(other.message)) {
    // 将原对象的指针置空，避免重复释放
    other.type = nullptr;
    other.value = nullptr;
    other.traceback = nullptr;
  }

  // 析构函数，释放异常相关的资源
  // NOLINTNEXTLINE(bugprone-exception-escape)
  ~python_error() override {
    if (type || value || traceback) {
      // 获取全局解释器锁
      pybind11::gil_scoped_acquire gil;
      // 释放异常相关的对象
      Py_XDECREF(type);
      Py_XDECREF(value);
      Py_XDECREF(traceback);
    }
  }

  // 返回异常消息
  const char* what() const noexcept override {
    return message.c_str();
  }

  // 构建异常消息
  void build_message() {
    // 获取全局解释器锁
    pybind11::gil_scoped_acquire gil;

    // 进入函数时，确保没有设置任何错误，因为 PyErr_Fetch 清除了错误指示器
    TORCH_INTERNAL_ASSERT(!PyErr_Occurred());

    // 设置默认消息
    message = "python_error";

    // 尝试从异常值中获取错误消息
    if (value != nullptr) {
      // 引用计数不应为零
      TORCH_INTERNAL_ASSERT(Py_REFCNT(value) > 0);

      // 转换异常值为字符串对象
      PyObject* pyStr = PyObject_Str(value);
      if (pyStr != nullptr) {
        // 将字符串对象编码为 utf-8 格式
        PyObject* encodedString =
            PyUnicode_AsEncodedString(pyStr, "utf-8", "strict");
        if (encodedString != nullptr) {
          char* bytes = PyBytes_AS_STRING(encodedString);
          if (bytes != nullptr) {
            // 设置异常消息
            message = std::string(bytes);
          }
          Py_XDECREF(encodedString);
        }
        Py_XDECREF(pyStr);
      }
    }

    // 清除任何错误，因为我们不希望将错误传播到构建错误消息的函数中
    PyErr_Clear();
  }

  // 将异常保存以便在不同线程上重新抛出
  inline void persist() {
    if (type)
      return; // 不要覆盖已有异常
    // PyErr_Fetch 覆盖指针
    pybind11::gil_scoped_acquire gil;
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);
    PyErr_Fetch(&type, &value, &traceback);
    build_message();
  }

  // 从此异常设置当前 Python 错误
  inline void restore() {
    if (!type)
      return;
    // PyErr_Restore 偷取引用
    pybind11::gil_scoped_acquire gil;
    Py_XINCREF(type);
    Py_XINCREF(value);
    Py_XINCREF(traceback);
    PyErr_Restore(type, value, traceback);
  }

  // 异常类型、值、回溯信息
  PyObject* type{nullptr};
  PyObject* value{nullptr};
  PyObject* traceback{nullptr};

  // what() 方法返回给用户的消息
  std::string message;
};
// 命名空间 torch 开始

namespace torch {

// 从 C++ 异常设置 Python 当前异常
TORCH_PYTHON_API void translate_exception_to_python(const std::exception_ptr&);

// 处理错误信息字符串，返回处理后的字符串
TORCH_PYTHON_API std::string processErrorMsg(std::string str);

// PyTorchError 的抽象基类，用于表示转换为特定 Python 类型的异常
struct PyTorchError : public std::exception {
  // 默认构造函数
  PyTorchError() = default;
  // 带有消息字符串的构造函数
  PyTorchError(std::string msg_) : msg(std::move(msg_)) {}
  // 虚函数，返回 Python 对象指针，子类需实现
  virtual PyObject* python_type() = 0;
  // 返回异常消息字符串
  const char* what() const noexcept override {
    return msg.c_str();
  }
  // 异常消息字符串
  std::string msg;
};

// 对于 gcc & clang，声明类似 printf 的函数，用于检查格式
#ifdef __GNUC__
#define TORCH_FORMAT_FUNC(FORMAT_INDEX, VA_ARGS_INDEX) \
  __attribute__((format(printf, FORMAT_INDEX, VA_ARGS_INDEX)))
#else
#define TORCH_FORMAT_FUNC(FORMAT_INDEX, VA_ARGS_INDEX)
#endif

// 表示转换为 Python TypeError 的异常
struct TypeError : public PyTorchError {
  // 使用基类的构造函数
  using PyTorchError::PyTorchError;
  // 带有格式化字符串的构造函数，用于格式检查
  TORCH_PYTHON_API TypeError(const char* format, ...) TORCH_FORMAT_FUNC(2, 3);
  // 返回 Python 类型对象指针
  PyObject* python_type() override {
    return PyExc_TypeError;
  }
};

// 表示转换为 Python AttributeError 的异常
struct AttributeError : public PyTorchError {
  // 使用基类的构造函数
  AttributeError(const char* format, ...) TORCH_FORMAT_FUNC(2, 3);
  // 返回 Python 类型对象指针
  PyObject* python_type() override {
    return PyExc_AttributeError;
  }
};

// ATen 的 Python 警告处理程序
struct PyWarningHandler {
  // 将实际处理程序移动到一个具有 noexcept 析构函数的单独类中
  // 否则，我们需要强制所有 WarningHandler 子类具有 noexcept(false) 的析构函数
  struct InternalHandler : at::WarningHandler {
    // 虚析构函数，默认实现
    ~InternalHandler() override = default;
    // 处理警告，重载基类函数
    void process(const c10::Warning& warning) override;

    // 警告缓冲区
    std::vector<c10::Warning> warning_buffer_;
  };

 public:
  /// 参见注释 [ Conversion Cpp Python Warning ]，用于 noexcept 语义的合理性
  TORCH_PYTHON_API PyWarningHandler() noexcept(true);
  // NOLINTNEXTLINE(bugprone-exception-escape)
  TORCH_PYTHON_API ~PyWarningHandler() noexcept(false);

  /** 如果发生异常，则调用此函数

   *  在析构函数中抛出异常前需要调用此函数，
   *  因为 std::uncaught_exception 在某些平台上存在问题，且跨动态库调用不可靠
   */
  void set_in_exception() {
    in_exception_ = true;
  }

 private:
  // 内部处理程序对象
  InternalHandler internal_handler_;
  // 前一个警告处理程序指针
  at::WarningHandler* prev_handler_;
  // 是否处于异常状态的标志
  bool in_exception_;
};

namespace detail {

// 空操作的 GIL 释放对象
struct noop_gil_scoped_release {
  // 用户定义的构造函数，避免在使用此类的地方出现未使用变量警告
  noop_gil_scoped_release() {}
};

// 条件性 GIL 释放对象，根据 release_gil 模板参数选择具体类型
template <bool release_gil>
using conditional_gil_scoped_release = std::conditional_t<
    release_gil,
    pybind11::gil_scoped_release,
    noop_gil_scoped_release>;

// 函数参数类型模板，从函数调用特性中提取第 i 个参数类型
template <typename Func, size_t i>
using Arg = typename invoke_traits<Func>::template arg<i>::type;
// 定义模板函数 wrap_pybind_function_impl_
template <typename Func, size_t... Is, bool release_gil>
auto wrap_pybind_function_impl_(
    // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    Func&& f,                               // 模板函数参数，转发引用
    std::index_sequence<Is...>,             // 模板参数，索引序列
    std::bool_constant<release_gil>) {      // 模板参数，是否释放 GIL 的标志

  namespace py = pybind11;                  // 引入 pybind11 命名空间

  // 返回一个 lambda 函数
  return [f = std::forward<Func>(f)](Arg<Func, Is>... args) {
    HANDLE_TH_ERRORS                       // 开始处理 Torch 异常
    conditional_gil_scoped_release<release_gil> no_gil;  // 根据 release_gil 条件选择是否释放 GIL
    return c10::guts::invoke(f, std::forward<Arg<Func, Is>>(args)...);  // 调用函数 f，并传递参数
    END_HANDLE_TH_ERRORS_PYBIND            // 结束 Torch 异常处理
  };
}
} // namespace detail                           // 结束 detail 命名空间

// 包装一个函数，添加 Torch 错误和警告处理
// 返回一个适合与 pybind11 注册的函数对象
template <typename Func>
auto wrap_pybind_function(Func&& f) {
  using traits = invoke_traits<Func>;        // 使用 invoke_traits 获取函数 f 的特性
  return torch::detail::wrap_pybind_function_impl_(
      std::forward<Func>(f),
      std::make_index_sequence<traits::arity>{},  // 创建参数索引序列
      std::false_type{});                    // 不释放 GIL
}

// 包装一个函数，添加 Torch 错误和警告处理，并释放 GIL
// 返回一个适合与 pybind11 注册的函数对象
template <typename Func>
auto wrap_pybind_function_no_gil(Func&& f) {
  using traits = invoke_traits<Func>;        // 使用 invoke_traits 获取函数 f 的特性
  return torch::detail::wrap_pybind_function_impl_(
      std::forward<Func>(f),
      std::make_index_sequence<traits::arity>{},  // 创建参数索引序列
      std::true_type{});                     // 释放 GIL
}

} // namespace torch                          // 结束 torch 命名空间
```