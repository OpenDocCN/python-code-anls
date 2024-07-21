# `.\pytorch\torch\csrc\distributed\c10d\control_collectives\ControlCollectives.hpp`

```
#pragma once
// 防止头文件被多次包含

#include <ATen/core/ivalue.h>
// 包含 ATen 库的 IValue 类定义

#include <chrono>
// 包含时间库 chrono，用于处理时间相关的功能

#include <cstdint>
// 包含标准整数类型定义的头文件

#include <string>
// 包含处理字符串的标准库

#include <vector>
// 包含处理向量的标准库

#include <c10/macros/Macros.h>
// 包含 C10 库的宏定义

#include <torch/custom_class.h>
// 包含定义自定义类的 Torch 头文件

namespace c10d {
// c10d 命名空间，用于组织控制集合通信相关的类和函数

using namespace std::chrono_literals;
// 使用 C++ 标准库中的 chrono_literals 命名空间，用于支持时间字面量

class TORCH_API ControlCollectives : public torch::CustomClassHolder {
// 定义 ControlCollectives 类，继承自 torch::CustomClassHolder
 public:
  virtual void barrier(
      const std::string& key,
      std::chrono::milliseconds timeout = 5min,
      bool block = true) = 0;
  // 定义纯虚函数 barrier，用于实现屏障操作，传入 key 表示操作标识符，
  // timeout 表示超时时间，默认为 5 分钟，block 表示是否阻塞，默认为 true

  virtual void broadcastSend(
      const std::string& key,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) = 0;
  // 定义纯虚函数 broadcastSend，用于实现广播发送操作，
  // key 表示操作标识符，data 表示发送的数据，
  // timeout 表示超时时间，默认为 5 分钟

  virtual std::vector<uint8_t> broadcastRecv(
      const std::string& key,
      std::chrono::milliseconds timeout = 5min) = 0;
  // 定义纯虚函数 broadcastRecv，用于实现广播接收操作，
  // key 表示操作标识符，timeout 表示超时时间，默认为 5 分钟，
  // 返回接收到的数据向量

  virtual void gatherSend(
      const std::string& key,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) = 0;
  // 定义纯虚函数 gatherSend，用于实现聚集发送操作，
  // key 表示操作标识符，data 表示发送的数据，
  // timeout 表示超时时间，默认为 5 分钟

  virtual std::vector<std::vector<uint8_t>> gatherRecv(
      const std::string& key,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) = 0;
  // 定义纯虚函数 gatherRecv，用于实现聚集接收操作，
  // key 表示操作标识符，data 表示发送的数据，
  // timeout 表示超时时间，默认为 5 分钟，
  // 返回接收到的数据向量的向量

  virtual std::vector<uint8_t> scatterSend(
      const std::string& key,
      const std::vector<std::vector<uint8_t>>& data,
      std::chrono::milliseconds timeout = 5min) = 0;
  // 定义纯虚函数 scatterSend，用于实现分散发送操作，
  // key 表示操作标识符，data 表示发送的数据向量的向量，
  // timeout 表示超时时间，默认为 5 分钟，
  // 返回接收到的数据向量

  virtual std::vector<uint8_t> scatterRecv(
      const std::string& key,
      std::chrono::milliseconds timeout = 5min) = 0;
  // 定义纯虚函数 scatterRecv，用于实现分散接收操作，
  // key 表示操作标识符，timeout 表示超时时间，默认为 5 分钟，
  // 返回接收到的数据向量

  virtual std::vector<std::vector<uint8_t>> allGather(
      const std::string& key,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) = 0;
  // 定义纯虚函数 allGather，用于实现全聚集操作，
  // key 表示操作标识符，data 表示发送的数据，
  // timeout 表示超时时间，默认为 5 分钟，
  // 返回接收到的数据向量的向量

  virtual int64_t allSum(
      const std::string& key,
      int64_t data,
      std::chrono::milliseconds timeout = 5min) = 0;
  // 定义纯虚函数 allSum，用于实现全求和操作，
  // key 表示操作标识符，data 表示发送的数据，
  // timeout 表示超时时间，默认为 5 分钟，
  // 返回求和结果
};

} // namespace c10d
// 结束 c10d 命名空间声明
```