# `.\pytorch\torch\csrc\jit\passes\pass_manager.cpp`

```
// 包含 Torch 的 JIT 模块中的 Pass Manager 头文件
#include <torch/csrc/jit/passes/pass_manager.h>

// 定义 torch 命名空间内的 jit 命名空间
namespace torch {
namespace jit {

// 初始化图形 Pass 的唯一标识符为 1
static GraphPassNameType graphPassID = 1;

// 返回自定义后处理 Pass 的静态全局变量引用
std::vector<GraphPassEntry>& getCustomPostPasses() {
  static std::vector<GraphPassEntry> passes;
  return passes;
}

// 返回自定义前处理 Pass 的静态全局变量引用
std::vector<GraphPassEntry>& getCustomPrePasses() {
  static std::vector<GraphPassEntry> passes;
  return passes;
}

// 注册后处理 Pass，并返回其唯一标识符
GraphPassNameType registerPostPass(GraphPass p) {
  // 获取后处理 Pass 列表的引用，添加新的 Pass 对象和其唯一标识符
  getCustomPostPasses().emplace_back(std::move(p), graphPassID);
  // 返回当前 Pass 的唯一标识符，并递增下一个 Pass 的唯一标识符
  return graphPassID++;
}

// 注册通用 Pass，并返回其唯一标识符
static GraphPassNameType registerPass(GraphPass p) {
  // 调用 registerPostPass 来注册新的 Pass
  return registerPostPass(std::move(p));
}

// 注册前处理 Pass，并返回其唯一标识符
GraphPassNameType registerPrePass(GraphPass p) {
  // 获取前处理 Pass 列表的引用，添加新的 Pass 对象和其唯一标识符
  getCustomPrePasses().emplace_back(std::move(p), graphPassID);
  // 返回当前 Pass 的唯一标识符，并递增下一个 Pass 的唯一标识符
  return graphPassID++;
}

// 清除特定后处理 Pass
void clearPostPass(GraphPassNameType pid) {
  auto& passes = getCustomPostPasses();
  auto it = passes.begin();
  // 查找要清除的 Pass
  for (; it != passes.end(); it++) {
    if (pid == (*it).second)
      break;
  }
  // 如果找到要清除的 Pass，则从列表中移除
  if (it != passes.end())
    passes.erase(it);
}

// 清除特定前处理 Pass
void clearPrePass(GraphPassNameType pid) {
  auto& passes = getCustomPrePasses();
  auto it = passes.begin();
  // 查找要清除的 Pass
  for (; it != passes.end(); it++) {
    if (pid == (*it).second)
      break;
  }
  // 如果找到要清除的 Pass，则从列表中移除
  if (it != passes.end())
    passes.erase(it);
}

// 清除所有后处理 Pass
void clearAllPostPasses() {
  auto& passes = getCustomPostPasses();
  // 清空所有后处理 Pass 列表
  passes.erase(passes.begin(), passes.end());
}

// 清除所有前处理 Pass
void clearAllPrePasses() {
  auto& passes = getCustomPrePasses();
  // 清空所有前处理 Pass 列表
  passes.erase(passes.begin(), passes.end());
}

// 用于兼容旧调用的注册后处理 Pass 构造函数
RegisterPostPass::RegisterPostPass(GraphPass p) {
  // 调用 registerPass 注册新的 Pass
  registerPass(std::move(p));
}

} // namespace jit
} // namespace torch
```