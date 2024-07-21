# `.\pytorch\torch\csrc\distributed\c10d\ParamCommsUtils.cpp`

```
// 包含 Torch 的分布式通信库参数通信工具的头文件
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>

// 声明 torch 命名空间
namespace torch {

// ParamCommsDebugInfo 类的构造函数定义
ParamCommsDebugInfo::ParamCommsDebugInfo(
    // 构造函数参数列表：
    // pgName 是包含两个字符串的元组，表示参数组的名称
    std::tuple<std::string, std::string> pgName,
    // rank 表示当前进程的排名
    int rank,
    // collName 是收集操作的名称（移动语义，即传递后源对象将无效）
    std::string&& collName,
    // inNelems 表示输入消息的元素数量
    int inNelems,
    // outNelems 表示输出消息的元素数量
    int outNelems,
    // dType 是数据类型（如整数、浮点数等）
    at::ScalarType dType,
    // inSplitSizes 是输入分片大小的向量
    std::vector<int64_t> inSplitSizes,
    // outSplitSizes 是输出分片大小的向量
    std::vector<int64_t> outSplitSizes,
    // globalRankStart 是全局排名的起始值
    int globalRankStart,
    // globalRankStride 是全局排名的步幅
    int globalRankStride,
    // worldSize 是通信世界中的进程总数
    int worldSize)
    // 构造函数初始化列表：
    : pgName_(std::move(pgName)),  // 初始化 pgName_
      rank_(rank),                // 初始化 rank_
      worldSize_(worldSize),      // 初始化 worldSize_
      collectiveName_(std::move(collName)),  // 初始化 collectiveName_
      inMessageNelems_(inNelems),             // 初始化 inMessageNelems_
      outMessageNelems_(outNelems),           // 初始化 outMessageNelems_
      dType_(dType),                          // 初始化 dType_
      inputSplitSizes_(std::move(inSplitSizes)),   // 初始化 inputSplitSizes_
      outputSplitSizes_(std::move(outSplitSizes)), // 初始化 outputSplitSizes_
      globalRankStart_(globalRankStart),           // 初始化 globalRankStart_
      globalRankStride_(globalRankStride) {        // 初始化 globalRankStride_

  // 如果 globalRankStride 大于 0，则计算 groupRanks_ 向量
  if (globalRankStride > 0) {
    for (int i = 0; i < worldSize; i++) {
      // 计算每个进程的全局排名并存储到 groupRanks_ 向量中
      groupRanks_.push_back(globalRankStart + i * globalRankStride);
    }
  }
}

} // namespace torch
```