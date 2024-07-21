# `.\pytorch\c10\util\Backtrace.cpp`

```
#ifdef FBCODE_CAFFE2
// 在 fbcode 中使用更好的堆栈跟踪实现，见 https://github.com/pytorch/pytorch/issues/56399
// 当可用时，优先使用此实现
class GetBacktraceImpl {
 public:
  // 构造函数，初始化堆栈跟踪对象
  C10_ALWAYS_INLINE GetBacktraceImpl(
      size_t frames_to_skip,
      size_t /* maximum_number_of_frames */,
      bool /* skip_python_frames */)
      : st_(/*skipFrames=*/frames_to_skip) {}

  // 对堆栈进行符号化处理，并返回字符串表示
  std::string symbolize() const {
    return st_.toString();
  }

 private:
  facebook::process::StackTrace st_;
};

#elif SUPPORTS_BACKTRACE && defined(C10_ANDROID)

// Android 平台下的堆栈跟踪状态结构体
struct AndroidBacktraceState {
  std::vector<void*> buffer;  // 用于存储堆栈帧地址的缓冲区
};

// Android 平台下的回调函数，用于获取堆栈帧地址
_Unwind_Reason_Code android_unwind_callback(
    struct _Unwind_Context* context,
    void* arg) {
  AndroidBacktraceState* state = (AndroidBacktraceState*)arg;
  uintptr_t pc = _Unwind_GetIP(context);
  if (pc) {
    state->buffer.emplace_back(reinterpret_cast<void*>(pc));  // 将地址添加到缓冲区中
  }
  return _URC_NO_REASON;
}

// Android 平台下的堆栈跟踪实现类
class GetBacktraceImpl {
 public:
  // 构造函数，通过回调函数获取堆栈帧地址
  C10_ALWAYS_INLINE GetBacktraceImpl(
      size_t /* frames_to_skip */,
      size_t /* maximum_number_of_frames */,
      bool /* skip_python_frames */) {
    _Unwind_Backtrace(android_unwind_callback, &state_);
  }

  // 对获取的堆栈帧地址进行符号化处理，并返回字符串表示
  std::string symbolize() const {
    std::ostringstream os;
    int idx = 0;
    char* demangled = nullptr;
    size_t length = 0;

    // 遍历堆栈帧地址，获取符号信息并格式化输出
    for (const void* addr : state_.buffer) {
      const char* symbol = "";

      Dl_info info;
      if (dladdr(addr, &info) && info.dli_sname) {
        symbol = info.dli_sname;  // 获取符号名
      }

      int status = 0;
      // 尝试解析符号名，获取可读的函数名
      demangled = __cxxabiv1::__cxa_demangle(
          /*mangled_name*/ symbol,
          /*output_buffer*/ demangled,
          /*length*/ &length,
          /*status*/ &status);

      // 将格式化后的堆栈帧信息添加到输出流中
      os << " frame #" << idx++ << "\t"
         << ((demangled != NULL && status == 0) ? demangled : symbol) << "["
         << addr << "]\t" << std::endl;
    }
    free(demangled);  // 释放解码后的符号名内存
    return os.str();  // 返回完整的堆栈帧信息字符串
  }

 private:
  AndroidBacktraceState state_;  // Android 平台下的堆栈跟踪状态对象
};

#elif SUPPORTS_BACKTRACE // !defined(C10_ANDROID)
struct FrameInformation {
  /// If available, the demangled name of the function at this frame, else
  /// whatever (possibly mangled) name we got from `backtrace()`.
  std::string function_name;
  /// This is a number in hexadecimal form (e.g. "0xdead") representing the
  /// offset into the function's machine code at which the function's body
  /// starts, i.e. skipping the "prologue" that handles stack manipulation and
  /// other calling convention things.
  std::string offset_into_function;
  /// NOTE: In debugger parlance, the "object file" refers to the ELF file that
  /// the symbol originates from, i.e. either an executable or a library.
  std::string object_file;
};

bool is_python_frame(const FrameInformation& frame) {
  return frame.object_file == "python" || frame.object_file == "python3" ||
      (frame.object_file.find("libpython") != std::string::npos);
}

std::optional<FrameInformation> parse_frame_information(
    const std::string& frame_string) {
  FrameInformation frame;

  // This is the function name in the CXX ABI mangled format, e.g. something
  // like _Z1gv. Reference:
  // https://itanium-cxx-abi.github.io/cxx-abi/abi.html#mangling
  std::string mangled_function_name;

#if defined(__GLIBCXX__)
  // In GLIBCXX, `frame_string` follows the pattern
  // `<object-file>(<mangled-function-name>+<offset-into-function>)
  // [<return-address>]`

  // Locate the start of the function name within the frame string
  auto function_name_start = frame_string.find('(');
  if (function_name_start == std::string::npos) {
    return c10::nullopt;  // Return empty optional if '(' not found
  }
  function_name_start += 1;

  // Locate the start of the offset within the frame string
  auto offset_start = frame_string.find('+', function_name_start);
  if (offset_start == std::string::npos) {
    return c10::nullopt;  // Return empty optional if '+' not found
  }
  offset_start += 1;

  // Locate the end of the offset substring
  const auto offset_end = frame_string.find(')', offset_start);
  if (offset_end == std::string::npos) {
    return c10::nullopt;  // Return empty optional if ')' not found
  }

  // Extract and assign the object file name from the frame string
  frame.object_file = frame_string.substr(0, function_name_start - 1);

  // Extract and assign the offset into function from the frame string
  frame.offset_into_function =
      frame_string.substr(offset_start, offset_end - offset_start);

  // Extract the mangled function name from the frame string
  mangled_function_name = frame_string.substr(
      function_name_start, (offset_start - 1) - function_name_start);
#elif defined(_LIBCPP_VERSION)
  // In LIBCXX, The pattern is
  // `<frame number> <object-file> <return-address> <mangled-function-name> +
  // <offset-into-function>`
  std::string skip;
  std::istringstream input_stream(frame_string);
  // operator>>() does not fail -- if the input stream is corrupted, the
  // strings will simply be empty.
  input_stream >> skip >> frame.object_file >> skip >> mangled_function_name >>
      skip >> frame.offset_into_function;
#else
#warning Unknown standard library, backtraces may have incomplete debug information
  return c10::nullopt;  // Return empty optional for unknown library
#endif // defined(__GLIBCXX__)

// 如果函数名为空，说明系统级别函数没有足够的调试信息，显示为"<unknown function>"
// 仍然会有返回地址和其他信息。
if (mangled_function_name.empty()) {
  // 将函数名设置为"<unknown function>"
  frame.function_name = "<unknown function>";
  // 返回当前帧信息
  return frame;
}

// 解析函数名并设置到帧信息中
frame.function_name = demangle(mangled_function_name.c_str());
// 返回当前帧信息
return frame;
}

class GetBacktraceImpl {
public:
C10_ALWAYS_INLINE GetBacktraceImpl(
    size_t frames_to_skip,
    size_t maximum_number_of_frames,
    bool skip_python_frames)
    : skip_python_frames_(skip_python_frames),
      callstack_(frames_to_skip + maximum_number_of_frames, nullptr) {
  // 我们总是跳过这一帧（backtrace）
  frames_to_skip += 1;

  // backtrace() 给出当前调用栈中的返回地址列表。
  // 注意：根据 man (3) backtrace 的文档，它永远不会失败。
  auto number_of_frames = static_cast<size_t>(
      ::backtrace(callstack_.data(), static_cast<int>(callstack_.size())));

  // 跳过请求的帧数。
  frames_to_skip = std::min(frames_to_skip, number_of_frames);
  number_of_frames -= frames_to_skip;
  // 删除跳过的帧数
  callstack_.erase(
      callstack_.begin(),
      callstack_.begin() + static_cast<ssize_t>(frames_to_skip));
  // 调整堆栈帧数到实际数量
  callstack_.resize(number_of_frames);
}

// 返回符号化的堆栈信息字符串
std::string symbolize() const {
  // `backtrace_symbols` 使用从`backtrace()`获取的返回地址，并获取每个堆栈的字符串表示。
  // 不幸的是，它不返回单独的信息结构，而是一个串联的字符串，我们需要在解析字符串之后使用。
  // 注意：`backtrace_symbols` 返回的数组是由 malloc 分配的，必须手动释放，但数组内的字符串不需要。
  std::unique_ptr<char*, std::function<void(char**)>> raw_symbols(
      ::backtrace_symbols(
          callstack_.data(), static_cast<int>(callstack_.size())),
      /*deleter=*/free);
  // 将字符指针数组转换为字符串向量
  const std::vector<std::string> symbols(
      raw_symbols.get(), raw_symbols.get() + callstack_.size());

  // 用于存储堆栈字符串的输出流
  std::ostringstream stream;

  // 在第一次跳过 Python 帧后将开关切换为 true
  bool has_skipped_python_frames = false;
    // 对于每个调用栈帧编号，使用范围循环遍历调用栈大小
    for (const auto frame_number : c10::irange(callstack_.size())) {
      // 解析当前帧的信息，返回一个指向帧信息的指针
      const auto frame = parse_frame_information(symbols[frame_number]);

      // 如果设置了跳过 Python 帧并且当前帧存在且是 Python 帧，则跳过处理
      if (skip_python_frames_ && frame && is_python_frame(*frame)) {
        // 如果尚未跳过 Python 帧，输出省略 Python 帧信息的提示
        if (!has_skipped_python_frames) {
          stream << "<omitting python frames>\n";
          has_skipped_python_frames = true;
        }
        // 继续下一次循环处理下一个帧
        continue;
      }

      // 输出当前帧的编号信息
      // frame #<number>:
      stream << "frame #" << frame_number << ": ";

      // 如果帧信息存在
      if (frame) {
        // 输出函数名 + 偏移量 (返回地址在对象文件中的位置)
        stream << frame->function_name << " + " << frame->offset_into_function
               << " (" << callstack_[frame_number] << " in "
               << frame->object_file << ")\n";
      } else {
        // 处理无法解析帧信息的边缘情况，直接输出符号信息
        stream << symbols[frame_number] << "\n";
      }
    }

    // 返回调用栈信息的字符串表示形式
    return stream.str();
  }

 private:
  const bool skip_python_frames_;  // 是否跳过 Python 帧的标志
  std::vector<void*> callstack_;   // 存储调用栈帧的指针数组
// 结束上一个代码段的条件编译块
};

// 如果是在 MSC 编译器下，并且不支持回溯，则执行以下代码
#elif defined(_MSC_VER) // !SUPPORTS_BACKTRACE

// 定义最大模块名称长度为 256
const int max_name_len = 256;

// 根据地址获取模块的基本名称
std::string get_module_base_name(void* addr) {
  HMODULE h_module;
  char module[max_name_len];
  // 将 module 初始化为空字符串
  strcpy(module, "");
  // 获取包含指定地址的模块的句柄
  GetModuleHandleEx(
      GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
          GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
      (LPCTSTR)addr,
      &h_module);
  // 如果获取到模块句柄
  if (h_module != NULL) {
    // 获取模块文件名
    GetModuleFileNameA(h_module, module, max_name_len);
  }
  // 查找最后一个反斜杠的位置
  char* last_slash_pos = strrchr(module, '\\');
  // 如果找到最后一个反斜杠
  if (last_slash_pos) {
    // 获取模块的基本名称并返回
    std::string module_base_name(last_slash_pos + 1);
    return module_base_name;
  } else {
    // 否则直接返回模块名称
    std::string module_base_name(module);
    return module_base_name;
  }
}

// 定义符号助手类
class SymbolHelper {
 public:
  // 获取唯一实例的静态方法
  static SymbolHelper& getInstance() {
    static SymbolHelper instance;
    return instance;
  }
  // 初始化标志和进程句柄
  bool inited = false;
  HANDLE process;

 private:
  // 私有构造函数，初始化进程句柄和符号加载选项
  SymbolHelper() {
    process = GetCurrentProcess();
    DWORD flags = SymGetOptions();
    SymSetOptions(flags | SYMOPT_DEFERRED_LOADS);
    // 初始化符号信息
    inited = SymInitialize(process, NULL, TRUE);
  }
  // 析构函数，在对象生命周期结束时清理符号信息
  ~SymbolHelper() {
    if (inited) {
      SymCleanup(process);
    }
  }

 public:
  // 禁用拷贝构造函数和赋值运算符
  SymbolHelper(SymbolHelper const&) = delete;
  void operator=(SymbolHelper const&) = delete;
};

// 该类用于在 Windows 平台上通过 Windows API 实现回溯功能，使用 CaptureStackBackTrace、SymFromAddr 和 SymGetLineFromAddr64
// 实现堆栈回溯功能参考：
// https://stackoverflow.com/questions/5693192/win32-backtrace-from-c-code
// https://stackoverflow.com/questions/26398064/counterpart-to-glibcs-backtrace-and-backtrace-symbols-on-windows
// https://docs.microsoft.com/en-us/windows/win32/debug/capturestackbacktrace
// https://docs.microsoft.com/en-us/windows/win32/api/dbghelp/nf-dbghelp-symfromaddr
// https://docs.microsoft.com/en-us/windows/win32/api/dbghelp/nf-dbghelp-symgetlinefromaddr64
// TODO: Support skipping python frames
class GetBacktraceImpl {
 public:
  // 构造函数，初始化堆栈回溯的相关参数和数据结构
  C10_ALWAYS_INLINE GetBacktraceImpl(
      size_t frames_to_skip,
      size_t maximum_number_of_frames,
      bool /* skip_python_frames */)
      : back_trace_(new void*[maximum_number_of_frames]) {
    // 始终跳过这个帧（backtrace）本身
    frames_to_skip += 1;

    // 获取堆栈帧信息
    n_frame_ = CaptureStackBackTrace(
        static_cast<DWORD>(frames_to_skip),
        static_cast<DWORD>(maximum_number_of_frames),
        back_trace_.get(),
        NULL);
  }

  // 符号化堆栈信息并返回字符串表示
  std::string symbolize() const {
    DWORD64 displacement;
    DWORD disp;
    std::unique_ptr<IMAGEHLP_LINE64> line;

    char buffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)];
    PSYMBOL_INFO p_symbol = (PSYMBOL_INFO)buffer;

    bool with_symbol = false;
    bool with_line = false;

    // 用于保存堆栈信息的输出流
    std::ostringstream stream;

    // 如果必要，初始化符号助手
    SymbolHelper& sh = SymbolHelper::getInstance();
    for (USHORT i_frame = 0; i_frame < n_frame_; ++i_frame) {
      // 遍历堆栈帧数组，处理每一个帧

      // 获取符号的地址和名称
      if (sh.inited) {
        // 如果符号引擎已初始化，则设置符号信息结构体的大小和最大名称长度
        p_symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
        p_symbol->MaxNameLen = MAX_SYM_NAME;
        // 通过地址获取符号信息，填充p_symbol结构体
        with_symbol = SymFromAddr(
            sh.process, (ULONG64)back_trace_[i_frame], &displacement, p_symbol);
      }

      // 获取代码行号和模块信息
      if (sh.inited) {
        // 如果符号引擎已初始化，则创建IMAGEHLP_LINE64结构体，并设置其大小
        line.reset(new IMAGEHLP_LINE64());
        line->SizeOfStruct = sizeof(IMAGEHLP_LINE64);
        // 通过地址获取代码行号信息，填充line结构体
        with_line = SymGetLineFromAddr64(
            sh.process, (ULONG64)back_trace_[i_frame], &disp, line.get());
      }

      // 获取模块的基本名称
      std::string module = get_module_base_name(back_trace_[i_frame]);

      // Windows系统的打印格式为：
      // `<return-address> <symbol-address> <module-name>!<demangled-function-name> [<file-name> @ <line-number>]`
      stream << std::setfill('0') << std::setw(16) << std::uppercase << std::hex
             << back_trace_[i_frame] << std::dec;
      if (with_symbol) {
        // 如果成功获取到符号信息，则打印符号地址、模块名称和符号名称
        stream << std::setfill('0') << std::setw(16) << std::uppercase
               << std::hex << p_symbol->Address << std::dec << " " << module
               << "!" << p_symbol->Name;
      } else {
        // 如果未能获取到符号信息，则打印未知符号地址和名称
        stream << " <unknown symbol address> " << module << "!<unknown symbol>";
      }
      // 打印文件名和行号信息
      stream << " [";
      if (with_line) {
        stream << line->FileName << " @ " << line->LineNumber;
      } else {
        stream << "<unknown file> @ <unknown line number>";
      }
      stream << "]" << std::endl;
    }

    // 返回所有堆栈帧的打印信息
    return stream.str();
  }

 private:
  // 堆栈帧数组和帧数
  std::unique_ptr<void*[]> back_trace_;
  USHORT n_frame_;
} // 如果没有定义宏，则结束当前代码块

#else

class GetBacktraceImpl {
 public:
  C10_ALWAYS_INLINE GetBacktraceImpl(
      size_t /* frames_to_skip */,
      size_t /* maximum_number_of_frames */,
      bool /* skip_python_frames */) {}
      // GetBacktraceImpl 构造函数，接受三个参数但未使用

  std::string symbolize() const {
    return "(no backtrace available)";
    // 返回字符串 "(no backtrace available)"，表示没有回溯信息
  }
};

#endif

} // namespace 结束命名空间

std::string get_backtrace(
    size_t frames_to_skip,
    size_t maximum_number_of_frames,
    bool skip_python_frames) {
  return GetBacktraceImpl{
      frames_to_skip, maximum_number_of_frames, skip_python_frames}
      .symbolize();
      // 调用 GetBacktraceImpl 的构造函数并调用 symbolize() 方法返回结果
}

Backtrace get_lazy_backtrace(
    size_t frames_to_skip,
    size_t maximum_number_of_frames,
    bool skip_python_frames) {
  class LazyBacktrace : public OptimisticLazyValue<std::string> {
   public:
    LazyBacktrace(GetBacktraceImpl&& impl) : impl_(std::move(impl)) {}
    // LazyBacktrace 类的构造函数，接受一个 GetBacktraceImpl 的右值引用

   private:
    std::string compute() const override {
      return impl_.symbolize();
      // 调用 impl_ 的 symbolize() 方法返回结果
    }

    GetBacktraceImpl impl_;
    // 私有成员变量 impl_，存储了 GetBacktraceImpl 对象
  };

  return std::make_shared<LazyBacktrace>(GetBacktraceImpl{
      frames_to_skip, maximum_number_of_frames, skip_python_frames});
      // 返回一个指向 LazyBacktrace 对象的 shared_ptr，该对象接受一个 GetBacktraceImpl 的临时对象
}

} // namespace c10 结束命名空间
```