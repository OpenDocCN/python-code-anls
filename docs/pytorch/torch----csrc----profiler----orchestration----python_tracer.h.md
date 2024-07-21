# `.\pytorch\torch\csrc\profiler\orchestration\python_tracer.h`

```
#pragma once
// 预处理指令：确保头文件只被编译一次

#include <cstdint>
// 包含标准整数类型定义头文件

#include <memory>
// 包含智能指针类模板头文件

#include <utility>
// 包含实用程序组件的头文件

#include <vector>
// 包含向量容器类模板头文件

#include <c10/util/ApproximateClock.h>
// 包含C10库中的近似时钟实用工具头文件

#include <c10/util/strong_type.h>
// 包含C10库中的强类型定义头文件

#include <torch/csrc/profiler/kineto_shim.h>
// 包含Torch库中的Kineto性能分析相关头文件

#include <torch/csrc/profiler/util.h>
// 包含Torch库中的性能分析实用工具头文件

namespace torch {
namespace profiler {
namespace impl {

class RecordQueue;
// 声明类RecordQueue，用于存储记录队列

struct Result;
// 声明结构体Result，用于存储结果数据

namespace python_tracer {

using TraceKey = strong::type<
    uint64_t,
    struct TraceKey_,
    strong::regular,
    strong::hashable,
    strong::ostreamable>;
// 定义TraceKey别名，为强类型uint64_t，支持正规操作、哈希和流输出

struct CompressedEvent {
  TraceKey key_;
  uint64_t system_tid_{};
  kineto::DeviceAndResource kineto_info_{};
  c10::time_t enter_t_{};
};
// 定义结构体CompressedEvent，包含跟踪键、系统线程ID、Kineto信息和进入时间

/*
Libtorch不依赖Python（例如无法#include <Python.h>）；
然而，当我们从libtorch_python调用性能分析器时，
我们需要分析器能够处理从Python跟踪器收集的数据。（`PyEval_SetProfile`）

为了解决这个依赖问题，我们定义了一个虚基类和一个注册获取器的函数。
然后Python跟踪器实现这些函数，并通过从`torch/csrc/autograd/init.cpp`调用`registerTracer`来公开自身。

在libtorch中为假的Python依赖注册这种模式在PyTorch代码库中很常见。
*/
struct TORCH_API PythonTracerBase {
  static std::unique_ptr<PythonTracerBase> make(RecordQueue* queue);
  // 静态方法：创建PythonTracerBase的唯一指针，传入记录队列参数

  virtual ~PythonTracerBase() = default;
  // 虚析构函数：默认实现

  virtual void stop() = 0;
  // 纯虚函数：停止函数，需要子类实现

  virtual std::vector<std::shared_ptr<Result>> getEvents(
      std::function<c10::time_t(c10::approx_time_t)> time_converter,
      std::vector<CompressedEvent>& enters,
      c10::time_t end_time_ns) = 0;
  // 纯虚函数：获取事件函数，返回结果为结果指针的向量，需要子类实现
};

using MakeFn = std::unique_ptr<PythonTracerBase> (*)(RecordQueue*);
// 定义MakeFn别名，为指向PythonTracerBase创建函数的指针

TORCH_API void registerTracer(MakeFn make_tracer);
// 函数声明：注册跟踪器函数，接受创建函数的指针作为参数

} // namespace python_tracer
} // namespace impl
} // namespace profiler
} // namespace torch
```