# `.\pytorch\aten\src\ATen\cuda\CUDAGraph.h`

```py
#pragma once
// 使用 #pragma once 指令确保头文件只被编译一次

#include <ATen/Tensor.h>
// 包含 ATen 库中的 Tensor 头文件，用于张量操作

#include <c10/core/Device.h>
// 包含 c10 核心库中的 Device 头文件，用于设备相关操作

#include <c10/cuda/CUDAGraphsC10Utils.h>
// 包含 c10 CUDA 图形工具的头文件，用于 CUDA 图形相关的实用工具

#include <c10/cuda/CUDAStream.h>
// 包含 c10 CUDA 流的头文件，用于 CUDA 流的管理

#include <c10/util/flat_hash_map.h>
// 包含 c10 实用工具中的 flat_hash_map 头文件，用于平面哈希映射的实现

namespace at {
// 命名空间 at，包含了 ATen 库的所有内容

struct Generator;
// 声明 Generator 结构体

struct CUDAGeneratorImpl;
// 声明 CUDAGeneratorImpl 结构体

struct CUDAGeneratorState;
// 声明 CUDAGeneratorState 结构体

namespace cuda {
// 命名空间 cuda，包含了 ATen 中 CUDA 相关的内容

// 获取一个唯一的内存池 ID，可以作为 pool=... 参数传递给 CUDAGraph::capture_begin 函数
TORCH_CUDA_CPP_API MempoolId_t graph_pool_handle();
// 声明 graph_pool_handle 函数，返回类型为 MempoolId_t，用于获取内存池 ID
struct TORCH_CUDA_CPP_API CUDAGraph {
  // 默认构造函数，初始化 CUDA 图对象
  CUDAGraph();

  // 析构函数，清理 CUDA 图对象
  ~CUDAGraph();

  // 增加待处理事件查询计数
  static void inc_pending_event_queries();

  // 减少待处理事件查询计数
  static void dec_pending_event_queries();

  // 获取当前待处理事件查询计数
  static int num_pending_event_queries();

  // 注册 CUDA 图的生成器状态，使用智能指针
  // 参见“Note [Explicit Registration of Generators to the CUDA Graph]”
  void register_generator_state(c10::intrusive_ptr<at::CUDAGeneratorState> state);

  // 注册 CUDA 图的生成器状态，使用生成器对象
  void register_generator_state(const at::Generator& generator);

  // 开始捕获 CUDA 图的记录
  void capture_begin(
      MempoolId_t pool = {0, 0},  // 记录的内存池ID，默认为{0, 0}
      cudaStreamCaptureMode capture_mode = cudaStreamCaptureModeGlobal);  // 捕获模式，默认为全局捕获模式

  // 结束 CUDA 图的捕获记录
  void capture_end();

  // 回放 CUDA 图
  void replay();

  // 重置 CUDA 图对象
  void reset();

  // 返回当前 CUDA 图对象的内存池ID
  MempoolId_t pool();

  // 启用调试模式
  void enable_debug_mode();

  // 将调试信息输出到指定路径
  void debug_dump(const std::string& debug_path);

 protected:
  cudaGraph_t graph_ = nullptr;      // CUDA 图句柄，初始为nullptr
  cudaGraphExec_t graph_exec_ = nullptr;  // CUDA 图执行句柄，初始为nullptr

  static std::atomic<int> pending_event_queries;  // 静态原子整数，用于管理待处理事件查询计数

  // 内部状态以便 reset() 方法可以尽可能清理
  bool has_graph_ = false;   // 标志位，表示是否存在 CUDA 图
  bool has_graph_exec_ = false;  // 标志位，表示是否存在 CUDA 图执行实例

  CaptureId_t id_;  // 当前捕获实例的ID

  CaptureId_t capture_id_ = -1;  // CUDA 图捕获ID，初始化为-1

  MempoolId_t mempool_id_;  // 当前内存池ID

  at::cuda::CUDAStream capture_stream_;  // 捕获开始时的 CUDA 流对象

  // 管理由 CUDA 图管理的多个生成器状态及其增量
  ska::flat_hash_map<c10::intrusive_ptr<at::CUDAGeneratorState>, uint64_t>
      captured_generator_states_;

  int capture_dev_;  // 捕获操作所在的设备编号
};

} // namespace cuda
} // namespace at
```