# `.\pytorch\torch\csrc\profiler\combined_traceback.cpp`

```py
// 引入 Torch 框架的性能分析模块相关头文件
#include <torch/csrc/profiler/combined_traceback.h>
#include <torch/csrc/utils/cpp_stacktraces.h>

// 定义 Torch 命名空间
namespace torch {

// 静态原子指针，用于支持 Python 捕获的跟踪信息
static std::atomic<CapturedTraceback::Python*> python_support_ = nullptr;

// 静态函数，收集捕获的堆栈跟踪信息，支持 Python、脚本、C++ 的捕获
std::shared_ptr<CapturedTraceback> CapturedTraceback::gather(
    bool python,
    bool script,
    bool cpp) {
  auto r = std::make_shared<CapturedTraceback>();

  // 如果需要 Python 的跟踪信息
  if (python) {
    auto p = python_support_.load();
    while (p && r->frames_.empty()) {
      r->frames_ = p->gather();
      r->python_ = p;
      p = p->next_;
    }
  }

  // 如果需要脚本的调用堆栈信息
  if (script) {
    r->script_frames_ = torch::jit::currentCallstack();
  }

  // 如果需要 C++ 的调用堆栈信息
  if (cpp) {
    r->cpp_frames_ = unwind::unwind();
  }

  return r;
}

// 遍历 Python 跟踪信息的帧，应用访问器函数 visit
int CapturedTraceback::traversePython(visitproc visit, void* arg) {
  TORCH_INTERNAL_ASSERT(python_);
  return python_->traverse(frames_, visit, arg);
}

// 清空 Python 跟踪信息的帧
int CapturedTraceback::clearPython() {
  TORCH_INTERNAL_ASSERT(python_);
  return python_->clear(frames_);
}

// CapturedTraceback 类的析构函数，释放 Python 跟踪信息的帧
CapturedTraceback::~CapturedTraceback() {
  if (!frames_.empty()) {
    TORCH_INTERNAL_ASSERT(python_);
    python_->release(frames_);
  }
}

// 自定义的 PyFrame 的哈希函数对象
struct PyFrameHash {
  std::size_t operator()(const CapturedTraceback::PyFrame& f) const {
    return std::hash<void*>()(f.code) ^ std::hash<int>()(f.lasti);
  }
};

// 自定义的 PyFrame 的相等性比较函数对象
struct PyFrameEq {
  std::size_t operator()(
      const CapturedTraceback::PyFrame& lhs,
      const CapturedTraceback::PyFrame& rhs) const {
    return lhs.code == rhs.code && lhs.lasti == rhs.lasti;
  }
};

// 对一组 CapturedTraceback 对象进行符号化处理，返回符号化的跟踪信息
SymbolizedTracebacks symbolize(
    const std::vector<CapturedTraceback*>& to_symbolize) {
  SymbolizedTracebacks r;

  // 存储 C++ 帧地址到偏移量的映射，用于去重和符号化请求
  std::unordered_map<void*, size_t> ip_to_frame_offset;
  
  // 存储 Python 帧到偏移量的映射，使用自定义的哈希和相等性比较函数
  std::unordered_map<CapturedTraceback::PyFrame, size_t, PyFrameHash, PyFrameEq> py_to_frame_offset;
  
  // 存储所有 C++ 帧的地址
  std::vector<void*> all_cpp_ips;

  // 去重和收集需要符号化的 C++ 帧
  for (const auto& e : to_symbolize) {
    for (void* f : e->cpp_frames_) {
      if (!ip_to_frame_offset.count(f)) {
        ip_to_frame_offset[f] = all_cpp_ips.size();
        all_cpp_ips.push_back(f);
      }
    }
  }

  // 如果存在需要符号化的 C++ 帧，调用 unwind::symbolize 进行符号化
  if (!all_cpp_ips.empty()) {
    r.all_frames = unwind::symbolize(all_cpp_ips, torch::get_symbolize_mode());
  }

  // 批量处理 Python 帧的符号化请求
  // 注意：需要确保在切换解释器前刷新请求
  CapturedTraceback::Python* cur_python = nullptr;
  std::vector<CapturedTraceback::PyFrame> cur_py_frames;
  size_t py_frames_size_ = 0;

  for (const auto& e : to_symbolize) {
    // 如果存在 Python 符号化信息
    if (e->python_) {
      // 如果当前 Python 对象不同于 e->python_ 并且当前 Python 堆栈不为空
      if (cur_python != e->python_ && !cur_py_frames.empty()) {
        // 如果当前有有效的 cur_python 对象，则将当前 Python 堆栈符号化并追加到结果中
        if (cur_python) {
          cur_python->appendSymbolized(cur_py_frames, r);
        }
        // 清空当前 Python 堆栈
        cur_py_frames.clear();
      }
      // 更新当前 Python 对象为 e->python_
      cur_python = e->python_;
      // 遍历当前异常对象 e 的所有帧
      for (const auto& f : e->frames_) {
        // 如果 py_to_frame_offset 中不存在当前帧 f，则将其添加到 py_to_frame_offset 并更新 py_frames_size_
        if (!py_to_frame_offset.count(f)) {
          py_to_frame_offset[f] = py_frames_size_++;
          // 将当前帧 f 添加到 cur_py_frames 中
          cur_py_frames.push_back(f);
        }
      }
    }
  }
  // 如果当前 Python 堆栈不为空
  if (!cur_py_frames.empty()) {
    // 如果当前有有效的 cur_python 对象，则将当前 Python 堆栈符号化并追加到结果中
    if (cur_python) {
      cur_python->appendSymbolized(cur_py_frames, r);
    }
    // 清空当前 Python 堆栈
    cur_py_frames.clear();
  }
  // 将 Python 堆栈的符号化结果迁移到 r.tracebacks 中
  std::vector<std::vector<uint64_t>> python_frame_fragments =
      std::move(r.tracebacks);
  // 清空 r.tracebacks
  r.tracebacks = {};

  // 遍历需要符号化的每个符号化请求 sc
  for (const auto& sc : to_symbolize) {
    // 为当前符号化请求添加一个新的 traceback
    r.tracebacks.emplace_back();
    // 初始化 Python 堆栈迭代器
    auto py_it = sc->frames_.begin();
    auto py_end = sc->frames_.end();

    // 判断是否已经追加了 JIT 帧
    bool jit_appended = false;

    // 定义 lambda 函数 append_python，用于将指定 Python 帧的片段添加到当前 traceback 中
    auto append_python = [&](const CapturedTraceback::PyFrame& f) {
      const auto& fragment =
          python_frame_fragments.at(py_to_frame_offset.at(f));
      r.tracebacks.back().insert(
          r.tracebacks.back().end(), fragment.begin(), fragment.end());
    };

    // 定义 lambda 函数 append_jit，用于追加 JIT 帧到当前 traceback 中
    auto append_jit = [&]() {
      if (jit_appended) {
        return;
      }
      jit_appended = true;
      // 遍历 script_frames_，并将其添加到 r.all_frames 中
      for (const auto& f : sc->script_frames_) {
        unwind::Frame frame;
        frame.funcname =
            f.filename; // 注意：torchscript 将函数名存储在 filename 字段中
        auto flc = f.range.file_line_col();
        if (flc) {
          size_t col = 0;
          std::tie(frame.filename, frame.lineno, col) = *flc;
        } else {
          frame.filename = "??";
          frame.lineno = 0;
        }
        r.tracebacks.back().push_back(r.all_frames.size());
        r.all_frames.emplace_back(std::move(frame));
      }
    };

    // 遍历 cpp_frames_，并根据不同的 C++ 帧类型执行不同的追加操作
    for (void* f : sc->cpp_frames_) {
      uint64_t cpp_frame = ip_to_frame_offset.at(f);
      const unwind::Frame& uf = r.all_frames.at(cpp_frame);
      // 如果当前 C++ 帧的函数名包含 "PyEval_EvalFrame"，则追加对应的 Python 帧
      if (uf.funcname.find("PyEval_EvalFrame") != std::string::npos) {
        if (py_it != py_end) {
          append_python(*py_it++);
        }
      // 如果当前 C++ 帧的函数名以 "torch::jit::InterpreterStateImpl::run" 开头，则追加 JIT 帧
      } else if (
          uf.funcname.rfind("torch::jit::InterpreterStateImpl::run", 0) !=
          std::string::npos) {
        append_jit();
      }
      // 将当前 C++ 帧的索引添加到当前 traceback 中
      r.tracebacks.back().push_back(cpp_frame);
    }

    // 如果在上述循环中未添加 JIT 帧，则补充添加
    append_jit();

    // 将剩余的 Python 帧追加到当前 traceback 中
    for (; py_it != py_end; ++py_it) {
      append_python(*py_it);
    }

    // 将所有用户定义的帧追加到当前 traceback 中
    for (const auto& f : sc->user_defined_frames_) {
      r.tracebacks.back().push_back(r.all_frames.size());
      r.all_frames.emplace_back(f);
    }
  }
  // 返回结果对象 r
  return r;
}

void CapturedTraceback::addPythonUnwinder(CapturedTraceback::Python* p) {
    // 加载当前的 Python 支持对象
    CapturedTraceback::Python* old_unwinder = python_support_.load();
    // 使用自旋锁，尝试将新的 Python 支持对象 p 添加到链表中
    do {
        p->next_ = old_unwinder;
    } while (!python_support_.compare_exchange_strong(old_unwinder, p));
}

} // namespace torch
```