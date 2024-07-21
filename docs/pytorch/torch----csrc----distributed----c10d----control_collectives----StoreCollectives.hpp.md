# `.\pytorch\torch\csrc\distributed\c10d\control_collectives\StoreCollectives.hpp`

```py
#pragma once
// 预处理指令：确保此头文件只包含一次

#include <c10/macros/Macros.h>
// 包含 c10 库中的宏定义

#include <c10/util/FbcodeMaps.h>
// 包含 c10 库中的 FbcodeMaps 实用工具

#include <torch/csrc/distributed/c10d/Store.hpp>
// 包含 Torch 分布式模块中的 Store 类定义

#include <torch/csrc/distributed/c10d/control_collectives/ControlCollectives.hpp>
// 包含 Torch 分布式模块中的控制集合控件 ControlCollectives 类定义

namespace c10d {

class TORCH_API StoreCollectives : public ControlCollectives {
// 定义 StoreCollectives 类，继承自 ControlCollectives 类
 public:
  explicit StoreCollectives(
      c10::intrusive_ptr<Store> store,
      int rank,
      int worldSize);
  // 构造函数：初始化 StoreCollectives 实例

  void barrier(
      const std::string& key,
      std::chrono::milliseconds timeout = 5min,
      bool block = true) override;
  // 实现 barrier 方法：执行分布式屏障操作

  void broadcastSend(
      const std::string& key,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) override;
  // 实现 broadcastSend 方法：发送广播数据到所有进程

  std::vector<uint8_t> broadcastRecv(
      const std::string& key,
      std::chrono::milliseconds timeout = 5min) override;
  // 实现 broadcastRecv 方法：接收广播数据

  void gatherSend(
      const std::string& key,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) override;
  // 实现 gatherSend 方法：发送数据以进行集合操作

  std::vector<std::vector<uint8_t>> gatherRecv(
      const std::string& key,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) override;
  // 实现 gatherRecv 方法：接收集合操作的数据

  std::vector<uint8_t> scatterSend(
      const std::string& key,
      const std::vector<std::vector<uint8_t>>& data,
      std::chrono::milliseconds timeout = 5min) override;
  // 实现 scatterSend 方法：发送数据以进行散射操作

  std::vector<uint8_t> scatterRecv(
      const std::string& key,
      std::chrono::milliseconds timeout = 5min) override;
  // 实现 scatterRecv 方法：接收散射操作的数据

  std::vector<std::vector<uint8_t>> allGather(
      const std::string& key,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) override;
  // 实现 allGather 方法：执行全局集合操作

  int64_t allSum(
      const std::string& key,
      int64_t data,
      std::chrono::milliseconds timeout = 5min) override;
  // 实现 allSum 方法：计算所有进程的总和

 private:
  void enforceUnique(const std::string& key);
  // 私有方法：确保键值唯一性的内部实现

 private:
  c10::intrusive_ptr<Store> store_;
  // 成员变量：存储指向分布式存储的智能指针

  int rank_;
  // 成员变量：当前进程的排名

  int worldSize_;
  // 成员变量：总进程数

  c10::FastSet<std::string> seenKeys_{};
  // 成员变量：用于跟踪已见过的键集合，使用 c10 提供的快速集合类型
};

} // namespace c10d
// 命名空间结束：c10d
```