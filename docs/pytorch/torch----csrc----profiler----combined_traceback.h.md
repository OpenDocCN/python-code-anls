# `.\pytorch\torch\csrc\profiler\combined_traceback.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <torch/csrc/jit/runtime/interpreter.h>
// 引入torch库中的interpreter头文件

#include <torch/csrc/profiler/unwind/unwind.h>
// 引入torch库中的profiler模块的unwind头文件

namespace torch {

// 声明torch命名空间

// struct that holds the result of symbolizing multiple tracebacks
// each traceback is a list of indices into all_frames
// (lots of Frames get duplicated across traces)
struct TORCH_API SymbolizedTracebacks {
  // 结构体定义，用于存储多个traceback符号化的结果
  // 每个traceback是一个对all_frames的索引列表
  // (很多帧在不同trace中会重复)
  std::vector<unwind::Frame> all_frames;
  // 存储在all_frames中的索引，用于在构造python对象时去重帧对象
  std::vector<std::vector<uint64_t>> tracebacks;
};

// 定义torch命名空间中的SymbolizedTracebacks结构体

struct TORCH_API CapturedTraceback : public c10::GatheredContext {
  // 定义CapturedTraceback结构体，继承自c10::GatheredContext类

  struct PyFrame {
    // 嵌套结构PyFrame，用于表示Python帧信息
    void* code; // PyCodeObject*, but python headers not present
    int lasti;
  };

  // 静态成员函数gather，返回shared_ptr<CapturedTraceback>，用于收集堆栈信息
  static std::shared_ptr<CapturedTraceback> gather(
      bool python,
      bool script,
      bool cpp);
  
  // 默认构造函数，默认拷贝构造函数和赋值运算符被禁用，移动构造函数和移动赋值运算符使用默认实现
  CapturedTraceback() = default;
  
  // 虚析构函数，用于释放资源
  ~CapturedTraceback() override;

  using visitproc = int (*)(void* self, void* arg);

  // 嵌套结构Python，用于处理Python堆栈信息
  struct Python {
    // 纯虚函数，由派生类实现，用于收集Python帧信息
    virtual std::vector<PyFrame> gather() = 0;
    
    // 纯虚函数，由派生类实现，用于释放Python帧信息
    virtual void release(std::vector<PyFrame>& frames) = 0;
    
    // 纯虚函数，由派生类实现，用于向符号化tracebacks中追加帧信息
    virtual void appendSymbolized(
        const std::vector<PyFrame>& to_symbolize,
        SymbolizedTracebacks& st) = 0;
    
    // 纯虚函数，由派生类实现，用于遍历Python帧信息
    virtual int traverse(
        std::vector<PyFrame>& frames,
        visitproc visit,
        void* arg) = 0;
    
    // 纯虚函数，由派生类实现，用于清除Python帧信息
    virtual int clear(std::vector<PyFrame>& frames) = 0;
    
    // 虚析构函数，默认实现
    virtual ~Python() = default;
    
    Python* next_ = nullptr; // 指向下一个Python对象的指针
  };
  
  // 静态成员函数addPythonUnwinder，向Python解释器注册Python堆栈记录功能
  static void addPythonUnwinder(Python* p);

  // 成员函数traversePython，遍历Python堆栈信息
  int traversePython(visitproc visit, void* arg);
  
  // 成员函数clearPython，清除Python堆栈信息
  int clearPython();

  // 成员函数recordUserDefinedFrame，记录用户定义的堆栈帧信息
  void recordUserDefinedFrame(const unwind::Frame& frame);

 private:
  std::vector<PyFrame> frames_; // 存储Python帧信息的向量
  std::vector<void*> cpp_frames_; // 存储C++帧信息的向量
  std::vector<jit::StackEntry> script_frames_; // 存储脚本帧信息的向量
  std::vector<unwind::Frame> user_defined_frames_; // 存储用户定义帧信息的向量
  
  friend TORCH_API SymbolizedTracebacks
  symbolize(const std::vector<CapturedTraceback*>& to_symbolize);
  
  // 非拥有引用，指向已注册的不朽Python对象的指针
  Python* python_ = nullptr;
};

// 声明symbolize函数，用于符号化给定的CapturedTraceback对象
TORCH_API SymbolizedTracebacks
symbolize(const std::vector<CapturedTraceback*>& to_symbolize);

} // namespace torch
```