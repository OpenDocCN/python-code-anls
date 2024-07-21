# `.\pytorch\torch\csrc\jit\codegen\cuda\interface.h`

```
#pragma once


// 使用 #pragma once 指令，确保头文件只被编译一次，避免重复包含

#include <c10/macros/Export.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/runtime/profiling_record.h>


/*
 * This file contains APIs for cuda fuser;
 *
 * We use an empty static struct to hold the function pointers, which are
 * registered separately. This is to support cpu-only compilation.
 * Registration is done in torch/csrc/jit/codegen/cuda/register_interface.cpp
 */


// 命名空间 torch::jit::fuser::cuda，包含 CUDA 融合相关的 API
namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// 获取 CUDA 融合保护模式的原子布尔值
TORCH_API std::atomic<bool>& getCudaFusionGuardMode();

// 获取是否启用单例融合
TORCH_API bool getSingletonFusion();
// 设置是否启用单例融合
TORCH_API bool setSingletonFusion(bool value);
// 获取是否启用水平融合
TORCH_API bool getHorizontalFusion();
// 设置是否启用水平融合
TORCH_API bool setHorizontalFusion(bool value);

// dummy struct to allow API registration
// 虚拟结构体，用于允许 API 的注册
struct CudaFuserInterface {
  // 编译融合节点对应的函数指针
  void (*fn_compile_n)(Node*) = nullptr;
  // 运行融合节点对应的函数指针
  void (*fn_run_n_s)(const Node*, Stack&) = nullptr;
  // 对图进行融合的函数指针
  void (*fn_fuse_graph)(std::shared_ptr<Graph>&) = nullptr;
  // 判断节点是否可以融合的函数指针
  bool (*fn_can_fuse_n)(const Node*) = nullptr;
  // 插入 CUDA 融合器的性能记录节点的函数指针
  void (*fn_insert_profile_inodes)(ProfilingRecord* pr) = nullptr;
  // 对节点进行性能分析的函数指针
  bool (*fn_profile_n)(const Node*) = nullptr;
  // 跳过节点的函数指针
  bool (*fn_skip_n)(const std::string&, bool flip) = nullptr;
};

// 获取 CUDA 融合接口的实例
TORCH_API CudaFuserInterface* getFuserInterface();

// 编译融合组节点
TORCH_API void compileFusionGroup(Node* fusion_node);
// 运行融合组节点
TORCH_API void runFusionGroup(const Node* fusion_node, Stack& stack);
// 对图进行融合
TORCH_API void fuseGraph(std::shared_ptr<Graph>&);
// 判断节点是否可以融合
TORCH_API bool canFuseNode(const Node* node);
// 插入 CUDA 融合器的性能记录节点
TORCH_API void InsertProfileNodesForCUDAFuser(ProfilingRecord* pr);
// 对节点进行性能分析
TORCH_API bool profileNode(const Node* node);

// 跳过节点的 API
TORCH_API bool skipNode(const std::string& symbol_str, bool flip = true);

// 判断是否启用 CUDA 融合
TORCH_API bool isEnabled();
// 设置是否启用 CUDA 融合
TORCH_API bool setEnabled(bool is_enabled);
// 判断是否可以启用 CUDA 融合
TORCH_API bool canBeEnabled();

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
```