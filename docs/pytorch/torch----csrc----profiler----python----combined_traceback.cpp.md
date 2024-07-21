# `.\pytorch\torch\csrc\profiler\python\combined_traceback.cpp`

```
#include <torch/csrc/profiler/python/combined_traceback.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pythoncapi_compat.h>

namespace py = pybind11;

namespace torch {

// Locking:
// We need to free PyCodeObjects when ~StackContext runs, but
// CUDACachingAllocator may hold its device lock when ~StackContext runs.

// Because the thread calling the allocator _may_ hold the GIL,
// attempting to lock the GIL in ~StackContext can deadlock:
// T0: GIL Lock -> Call Allocator    ->| Waiting Device Lock
// T1: Call Allocator -> Device Lock ->| Waiting GIL Lock
// Instead the destructor defers freeing stack frames by putting them in
// to_free_frames. We still need a lock to manage this vector, but
// we can ensure an overall lock ordering of GIL -> device_lock ->
// to_free_frames_mutex because ::gather is called outside of the device lock.

namespace {
static std::mutex to_free_frames_mutex;  // 用于管理 to_free_frames 向量的互斥锁
static std::vector<CapturedTraceback::PyFrame> to_free_frames;  // 存储待释放的 PyFrame 对象的向量

struct PythonTraceback : public CapturedTraceback::Python {
  std::vector<CapturedTraceback::PyFrame> gather() override {
    if (!Py_IsInitialized()) {
      return {};  // 如果 Python 未初始化，返回空向量
    }
    std::vector<CapturedTraceback::PyFrame> frames;  // 存储捕获的 Python 帧的向量
    py::gil_scoped_acquire acquire;  // 获取全局解释器锁（GIL）
    {
      std::lock_guard<std::mutex> lock(to_free_frames_mutex);  // 使用互斥锁锁定 to_free_frames 向量
      // 释放 to_free_frames 中所有帧的 PyCodeObject
      for (CapturedTraceback::PyFrame f : to_free_frames) {
        Py_XDECREF(f.code);
      }
      to_free_frames.clear();  // 清空 to_free_frames
    }
    PyFrameObject* f = PyEval_GetFrame();  // 获取当前 Python 帧
    Py_XINCREF(f);  // 增加当前帧的引用计数
    while (f) {
      // 将当前帧的 PyCodeObject 和 last_instruction 存入 frames 中
      frames.emplace_back(
          CapturedTraceback::PyFrame{PyFrame_GetCode(f), PyFrame_GetLasti(f)});
      auto f_back = PyFrame_GetBack(f);  // 获取当前帧的后继帧
      Py_XDECREF(f);  // 减少当前帧的引用计数
      f = f_back;  // 将当前帧指向后继帧
    }
    return frames;  // 返回捕获的 Python 帧的向量
  }

  void release(std::vector<CapturedTraceback::PyFrame>& frames) override {
    std::lock_guard<std::mutex> lock(to_free_frames_mutex);  // 使用互斥锁锁定 to_free_frames 向量
    // 将 frames 中的帧添加到 to_free_frames 向量末尾
    to_free_frames.insert(to_free_frames.end(), frames.begin(), frames.end());
  }

  using void_visitproc = int (*)(void* self, void* arg);
  int traverse(
      std::vector<CapturedTraceback::PyFrame>& frames,
      void_visitproc visit,
      void* arg) override {
    // 遍历 frames 中的所有帧，调用 Py_VISIT 对其进行访问
    for (auto& f : frames) {
      Py_VISIT(f.code);
    }
    return 0;
  }

  int clear(std::vector<CapturedTraceback::PyFrame>& frames) override {
    // 遍历 frames 中的所有帧，调用 Py_CLEAR 对其进行清除
    for (auto& f : frames) {
      Py_CLEAR(f.code);
    }
    return 0;
  }

  void appendSymbolized(
      const std::vector<CapturedTraceback::PyFrame>& to_symbolize,
      SymbolizedTracebacks& result) override {
    py::gil_scoped_acquire acquire;  // 获取全局解释器锁（GIL）
    py::str line_s = "line";  // 创建 Python 字符串对象 "line"
    py::str name_s = "name";  // 创建 Python 字符串对象 "name"
    py::str filename_s = "filename";  // 创建 Python 字符串对象 "filename"

    auto torch = py::module::import("torch");  // 导入名为 "torch" 的 Python 模块
    py::object stack_frames_for_code;  // 创建 Python 对象来保存代码的堆栈帧信息
    // 检查 torch 模块是否具有属性 "_inductor"
    if (py::hasattr(torch, "_inductor")) {
      // 获取 torch._inductor 对象
      py::object inductor = torch.attr("_inductor");
      // 检查 inductor 对象是否具有属性 "codecache"
      if (py::hasattr(inductor, "codecache")) {
        // 获取 inductor.codecache.PyCodeCache.stack_frames_for_code 对象
        stack_frames_for_code = inductor.attr("codecache")
                                    .attr("PyCodeCache")
                                    .attr("stack_frames_for_code");
      }
    }
    // 遍历要进行符号化的每个元素
    for (const auto& f : to_symbolize) {
      // 将 f.code 强制转换为 PyCodeObject 指针类型
      auto f_code = (PyCodeObject*)f.code;
      // 获取 f_code 对象的 co_filename 属性
      py::handle filename = f_code->co_filename;
      // 获取 f_code 对象的 co_name 属性
      py::handle funcname = f_code->co_name;
      // 使用 PyCode_Addr2Line 函数获取 f_code 对象和 f.lasti 参数对应的行号
      auto lineno = PyCode_Addr2Line(f_code, f.lasti);
      // 将一个新的 traceback 对象添加到 result.tracebacks 中
      result.tracebacks.emplace_back();
      // 将新添加的 traceback 对象的索引值添加到 result.tracebacks.back() 中
      result.tracebacks.back().push_back(result.all_frames.size());
      // 将一个新的 unwind::Frame 对象添加到 result.all_frames 中
      result.all_frames.emplace_back(unwind::Frame{
          // 将 filename 强制转换为 std::string 类型
          py::cast<std::string>(filename),
          // 将 funcname 强制转换为 std::string 类型
          py::cast<std::string>(funcname),
          // 将 lineno 强制转换为 uint64_t 类型
          (uint64_t)lineno});
      // 查找与 inductor 生成的代码相关联的所有额外帧
      // 如果 stack_frames_for_code 不为空指针
      if (stack_frames_for_code.ptr()) {
        // 调用 stack_frames_for_code 函数，传入 filename 和 lineno 作为参数
        py::object extra = stack_frames_for_code(filename, lineno);
        // 如果 extra 不为 None
        if (!extra.is_none()) {
          // 遍历 extra 中的每个 handle 对象
          for (py::handle h : extra) {
            // 将新添加的 traceback 对象的索引值添加到 result.tracebacks.back() 中
            result.tracebacks.back().push_back(result.all_frames.size());
            // 将一个新的 unwind::Frame 对象添加到 result.all_frames 中
            result.all_frames.emplace_back(unwind::Frame{
                // 将 h[filename_s] 强制转换为 std::string 类型
                py::cast<std::string>(h[filename_s]),
                // 将 h[name_s] 强制转换为 std::string 类型
                py::cast<std::string>(h[name_s]),
                // 将 h[line_s] 强制转换为 uint64_t 类型
                py::cast<uint64_t>(h[line_s])});
          }
        }
      }
    }
};

} // namespace

// 定义一个函数 py_symbolize，接受一个 CapturedTraceback 对象的 vector，并返回一个 py::object 的 vector
std::vector<py::object> py_symbolize(
    std::vector<CapturedTraceback*>& to_symbolize) {
  
  // 使用 unordered_map 缓存 CapturedTraceback 指针和它们在 unique_frames 中的索引
  std::unordered_map<CapturedTraceback*, uint64_t> cached_frames;
  // 存储不重复的 CapturedTraceback 对象
  std::vector<CapturedTraceback*> unique_frames;
  
  // 遍历输入的 to_symbolize 向量
  for (const auto& sc : to_symbolize) {
    // 检查当前 CapturedTraceback 是否已经在 cached_frames 中
    auto it = cached_frames.find(sc);
    // 如果未找到，则将其添加到 cached_frames 并添加到 unique_frames
    if (it == cached_frames.end()) {
      cached_frames.insert({sc, unique_frames.size()});
      unique_frames.push_back(sc);
    }
  }
  
  // 调用 symbolize 函数，将 unique_frames 转换为 s 结构体
  auto s = symbolize(unique_frames);

  // 定义 Python 字符串对象
  py::str line_s = "line";
  py::str name_s = "name";
  py::str filename_s = "filename";
  
  // 存储所有帧的 Python 字典对象
  std::vector<py::dict> all_frames;
  
  // 遍历 s 结构体中的所有帧
  for (const auto& f : s.all_frames) {
    py::dict d;
    // 将函数名、文件名和行号添加到 Python 字典对象 d 中
    d[name_s] = f.funcname;
    d[filename_s] = f.filename;
    d[line_s] = f.lineno;
    all_frames.emplace_back(std::move(d));
  }

  // 存储唯一帧的 Python 对象的向量
  std::vector<py::object> py_unique_frames;
  
  // 遍历 s 结构体中的所有 traceback
  for (const auto& t : s.tracebacks) {
    py::list l;
    // 遍历每个 traceback 中的帧索引，将对应的 Python 字典对象添加到列表 l 中
    for (const auto& e : t) {
      l.append(all_frames.at(e));
    }
    // 将列表 l 添加到 py_unique_frames 中
    py_unique_frames.push_back(std::move(l));
  }

  // 存储最终结果的 Python 对象的向量
  std::vector<py::object> result;
  result.reserve(to_symbolize.size());
  
  // 根据 cached_frames 中的索引，将结果添加到 result 中
  for (const auto& sc : to_symbolize) {
    result.push_back(py_unique_frames.at(cached_frames.at(sc)));
  }
  
  return result; // 返回最终结果
}

// 释放已经死亡的 CapturedTraceback 帧的函数
void freeDeadCapturedTracebackFrames() {
  std::lock_guard<std::mutex> lock(to_free_frames_mutex);
  // 遍历要释放的 CapturedTraceback 的 PyFrame 结构体，释放其 code 成员
  for (CapturedTraceback::PyFrame f : to_free_frames) {
    Py_XDECREF(f.code);
  }
  // 清空 to_free_frames 向量
  to_free_frames.clear();
}

// 安装捕获的 traceback 的 Python 解析器
void installCapturedTracebackPython() {
  // 添加 PythonTraceback 对象到 CapturedTraceback 的解析器中
  CapturedTraceback::addPythonUnwinder(new PythonTraceback());
}

} // namespace torch
```