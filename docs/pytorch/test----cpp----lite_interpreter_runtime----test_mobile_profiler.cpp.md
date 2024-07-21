# `.\pytorch\test\cpp\lite_interpreter_runtime\test_mobile_profiler.cpp`

```
// 包含 ATen 库的函数声明
#include <ATen/Functions.h>
// 包含 Google Test 的头文件，用于单元测试
#include <gtest/gtest.h>
// 包含 JIT 测试工具的头文件
#include <test/cpp/jit/test_utils.h>
// 包含 Torch 模块的 API 定义
#include <torch/csrc/jit/api/module.h>
// 包含 Torch 前端解析器的头文件
#include <torch/csrc/jit/frontend/resolver.h>
// 包含 Torch 移动端模型导入功能的头文件
#include <torch/csrc/jit/mobile/import.h>
// 包含 Torch 移动端模块定义的头文件
#include <torch/csrc/jit/mobile/module.h>
// 包含 Torch 移动端分析器边缘功能的头文件
#include <torch/csrc/jit/mobile/profiler_edge.h>
// 包含文件输入输出流的头文件
#include <fstream>

// 包含无序集合的标准库头文件
#include <unordered_set>

// 包含 Torch 分析器事件的头文件
#include <torch/csrc/profiler/events.h>

// 包含用于测试的轻量级解释器运行时资源的头文件
#include "test/cpp/lite_interpreter_runtime/resources.h"

// 如果定义了 EDGE_PROFILER_USE_KINETO，则进入命名空间 torch::jit::mobile
#ifdef EDGE_PROFILER_USE_KINETO
namespace torch {
namespace jit {
namespace mobile {

// 匿名命名空间，用于定义私有函数和变量，本例中定义了一个检查元数据的函数
namespace {
// 检查给定操作名、元数据名、元数据值在跟踪文件中是否存在的函数
bool checkMetaData(
    const std::string& op_name,
    const std::string& metadata_name,
    const std::string& metadata_val,
    std::ifstream& trace_file) {
  std::string line;
  // 逐行读取跟踪文件
  while (std::getline(trace_file, line)) {
    // 如果找到包含操作名的行
    if (line.find(op_name) != std::string::npos) {
      // 继续读取直到找到包含元数据名的行
      while (std::getline(trace_file, line)) {
        if (line.find(metadata_name) != std::string::npos) {
          // 如果找到元数据名，并且元数据值匹配或为空，则返回 true
          if (line.find(metadata_val) != std::string::npos ||
              !metadata_val.size()) {
            /* 如果找到正确的元数据值或者期望的元数据值为空字符串，则忽略该元数据值 */
            return true;
          }
        }
      }
    }
  }
  // 如果未找到符合条件的元数据值，则返回 false
  return false;
}
} // namespace

// 定义 MobileProfiler 测试类，测试移动端分析器功能
TEST(MobileProfiler, ModuleHierarchy) {
  // 获取要测试的模型文件的路径
  auto testModelFile = torch::testing::getResourcePath(
      "test/cpp/lite_interpreter_runtime/to_be_profiled_module.ptl");

  // 准备模型输入数据
  std::vector<IValue> inputs;
  inputs.emplace_back(at::rand({64, 64}));
  inputs.emplace_back(at::rand({64, 64}));

  // 指定跟踪文件的路径和名称
  std::string trace_file_name("/tmp/test_trace.trace");

  // 加载移动端模型，创建模块对象
  mobile::Module bc = _load_for_mobile(testModelFile.string());
  {
    // 创建 KinetoEdgeCPUProfiler 对象，用于对模块进行性能分析
    KinetoEdgeCPUProfiler profiler(
        bc,
        trace_file_name,
        false, // 不记录输入形状
        false, // 不进行内存分析
        true,  // 记录调用堆栈
        false, // 不记录浮点运算
        true,  // 记录模块层次结构
        {},    // 事件为空
        false  // 不调整 Vulkan 时间戳
    ); // 完成 KinetoEdgeCPUProfiler 对象的创建
  bc.forward(inputs);
  // 调用神经网络模型的前向传播函数

  } // End of profiler
  // 结束性能分析器的作用域

  std::ifstream trace_file(trace_file_name);
  // 打开一个文件输入流，用于读取跟踪文件

  std::string line;
  // 定义一个字符串变量，用于存储读取的每一行内容

  ASSERT_TRUE(trace_file.is_open());
  // 断言文件流已经成功打开

  trace_file.seekg(0, std::ios_base::beg);
  // 将文件指针移动到文件开头

  const std::string metadata_name("Module Hierarchy");
  // 定义一个常量字符串，表示元数据的名称为 "Module Hierarchy"

  ASSERT_TRUE(checkMetaData(
      "aten::sub",
      metadata_name,
      "top(C)::<unknown>.A0(A)::forward.aten::sub",
      trace_file));
  // 断言检查元数据信息，验证是否存在给定的函数调用关系

  trace_file.seekg(0, std::ios_base::beg);
  // 再次将文件指针移动到文件开头，准备读取下一个检查

  ASSERT_TRUE(checkMetaData(
      "aten::mul",
      metadata_name,
      "top(C)::<unknown>.A0(A)::forward.SELF(A)::forward_impl_.SELF(A)::my_new_method.aten::mul",
      trace_file));
  // 断言检查元数据信息，验证是否存在给定的函数调用关系

  trace_file.seekg(0, std::ios_base::beg);
  // 再次将文件指针移动到文件开头，准备读取下一个检查

  ASSERT_TRUE(checkMetaData(
      "aten::add",
      metadata_name,
      "top(C)::<unknown>.A0(A)::forward.SELF(A)::forward_impl_.aten::add",
      trace_file));
  // 断言检查元数据信息，验证是否存在给定的函数调用关系

  ASSERT_TRUE(checkMetaData(
      "aten::add",
      metadata_name,
      "top(C)::<unknown>.SELF(C)::call_b.B0(B)::forward.aten::add",
      trace_file));
  // 断言检查元数据信息，验证是否存在给定的函数调用关系

  ASSERT_TRUE(checkMetaData(
      "aten::add", metadata_name, "top(C)::<unknown>.aten::add", trace_file));
  // 断言检查元数据信息，验证是否存在给定的函数调用关系


这段代码是一个 C++ 程序，对一个跟踪文件中的特定元数据进行多次检查，并通过断言（ASSERT_TRUE）来验证这些元数据是否存在于文件中。
TEST(MobileProfiler, ProfilerEvent) {
  auto testModelFile = torch::testing::getResourcePath(
      "test/cpp/lite_interpreter_runtime/test_backend_for_profiling.ptl");

  // 准备输入数据
  std::vector<IValue> inputs;
  inputs.emplace_back(at::rand({64, 64}));
  inputs.emplace_back(at::rand({64, 64}));

  // 定义追踪文件的路径
  std::string trace_file_name("/tmp/test_trace_profiler_event.trace");

  // 获取当前支持的性能事件列表
  std::vector<std::string> events(
      torch::profiler::ProfilerPerfEvents.begin(),
      torch::profiler::ProfilerPerfEvents.end());

  // 加载移动端模型
  mobile::Module bc = _load_for_mobile(testModelFile.string());

  {
    // 在此创建性能分析器对象
    mobile::KinetoEdgeCPUProfiler profiler(
        bc,
        trace_file_name,
        false, // 不记录输入形状
        false, // 不分析内存
        true,  // 记录调用堆栈
        false, // 不记录浮点操作
        true); // 记录模块层次结构
    // 执行模型的前向传播
    bc.forward(inputs);
  } // 结束性能分析器作用域

  // 打开追踪文件并检查是否成功
  std::ifstream trace_file(trace_file_name);
  std::string line;
  ASSERT_TRUE(trace_file.is_open());
  trace_file.seekg(0, std::ios_base::beg);

  // 准备元数据名称并检查模块层次结构信息
  std::string metadata_name("Module Hierarchy");
  ASSERT_TRUE(checkMetaData(
      "aten::add", metadata_name, "top(m)::<unknown>.aten::add", trace_file));

  // 重新定位文件指针，准备检查后端信息
  trace_file.seekg(0, std::ios_base::beg);
  metadata_name = "Backend";
  ASSERT_TRUE(
      checkMetaData("aten::add", metadata_name, "test_backend", trace_file));
}
  try {
    // 使用 KinetoEdgeCPUProfiler 类创建性能分析器对象，配置如下选项：
    // - bc: 被分析的计算图
    // - trace_file_name: 跟踪文件名
    // - false: 不记录输入形状
    // - false: 不进行内存分析
    // - true: 记录调用栈
    // - false: 不记录浮点运算次数
    // - true: 记录模块层次结构
    // - events: 性能事件列表
    KinetoEdgeCPUProfiler profiler(
        bc,
        trace_file_name,
        false, // record input_shapes
        false, // profile memory
        true,  // record callstack
        false, // record flops
        true,  // record module hierarchy
        events); // performance events
    // 运行计算图的前向传播
    bc.forward(inputs);
  } catch (...) {
    // 如果发生异常，直接返回
    return;
  }
} // End of profiler

// 打开跟踪文件以读取
std::ifstream trace_file(trace_file_name);
std::string line;
// 确保成功打开跟踪文件
ASSERT_TRUE(trace_file.is_open());

// 遍历事件列表中的每个事件
for (auto& event : events) {
  // 将文件指针重新定位到文件开头
  trace_file.seekg(0, std::ios_base::beg);
  /*
   * Just checking if the event entry exists in the chrometrace.
   * Checking the value in a hardware independent matter is tricky.
   */
  // 断言确保在 chrometrace 中存在特定事件条目
  ASSERT_TRUE(checkMetaData("aten::__getitem__", event, "", trace_file));
}
}
} // namespace mobile
} // namespace jit
} // namespace torch
#endif


注释：


// 结束 mobile 命名空间的定义
}
// 结束 jit 命名空间的定义
} // namespace mobile
// 结束 torch 命名空间的定义
} // namespace jit
// 结束文件预处理器条件编译指令的区域
} // namespace torch
// 结束文件的条件编译指令区域
#endif
```