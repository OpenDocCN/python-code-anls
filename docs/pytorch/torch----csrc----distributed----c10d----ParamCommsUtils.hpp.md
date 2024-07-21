# `.\pytorch\torch\csrc\distributed\c10d\ParamCommsUtils.hpp`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <ATen/core/ivalue.h>
// 包含 ATen 库的 IValue 头文件

#include <ATen/record_function.h>
// 包含 ATen 库的 record_function 头文件

#include <c10/macros/Macros.h>
// 包含 c10 库的宏定义头文件

#include <c10/util/ThreadLocalDebugInfo.h>
// 包含 c10 库的 ThreadLocalDebugInfo 头文件

#include <string>
// 包含 C++ 标准库的 string 头文件

#include <vector>
// 包含 C++ 标准库的 vector 头文件

namespace torch {

class TORCH_API ParamCommsDebugInfo : public c10::DebugInfoBase {
 public:
  ParamCommsDebugInfo() = default;
  // 默认构造函数

  ParamCommsDebugInfo(
      std::tuple<std::string, std::string> pgName,
      int rank,
      std::string&& collName,
      int inNelems,
      int outNelems,
      at::ScalarType dType,
      std::vector<int64_t> inSplitSizes,
      std::vector<int64_t> outSplitSizes,
      int globalRankStart,
      int globalRankStride,
      int worldSize);
  // 构造函数声明，接受多个参数来初始化对象

  ~ParamCommsDebugInfo() override = default;
  // 虚析构函数，用于正确释放资源

  const std::string getProcessGroupName() const {
    return std::get<0>(pgName_);
  }
  // 返回进程组名的访问方法

  const std::string getProcessGroupDesc() const {
    return std::get<1>(pgName_);
  }
  // 返回进程组描述的访问方法

  int getRank() const {
    return rank_;
  }
  // 返回进程排名的访问方法

  int getWorldSize() const {
    return worldSize_;
  }
  // 返回全局进程数量的访问方法

  int getGlobalRankStart() const {
    return globalRankStart_;
  }
  // 返回全局进程起始排名的访问方法

  int getGlobalRankStride() const {
    return globalRankStride_;
  }
  // 返回全局进程排名步长的访问方法

  const std::string getCollectiveName() const {
    return collectiveName_;
  }
  // 返回集合通信名称的访问方法

  int getInMessageNelems() const {
    return inMessageNelems_;
  }
  // 返回输入消息元素数量的访问方法

  int getOutMessageNelems() const {
    return outMessageNelems_;
  }
  // 返回输出消息元素数量的访问方法

  at::ScalarType getDType() const {
    return dType_;
  }
  // 返回数据类型的访问方法

  const std::vector<int64_t>& getInputSplitSizes() const {
    return inputSplitSizes_;
  }
  // 返回输入分割尺寸的访问方法

  const std::vector<int64_t>& getOutputSplitSizes() const {
    return outputSplitSizes_;
  }
  // 返回输出分割尺寸的访问方法

  const std::vector<int64_t>& getGroupRanks() const {
    return groupRanks_;
  }
  // 返回组排名的访问方法

 private:
  std::tuple<std::string, std::string> pgName_; // <group_name, group_desc>
  // 成员变量，用于存储进程组名和组描述

  int rank_{};
  // 成员变量，用于存储进程排名，默认初始化为零

  int worldSize_{};
  // 成员变量，用于存储全局进程数量，默认初始化为零

  std::string collectiveName_;
  // 成员变量，用于存储集合通信名称

  int inMessageNelems_{};
  // 成员变量，用于存储输入消息元素数量，默认初始化为零

  int outMessageNelems_{};
  // 成员变量，用于存储输出消息元素数量，默认初始化为零

  at::ScalarType dType_ = at::kByte;
  // 成员变量，用于存储数据类型，默认初始化为字节类型

  std::vector<int64_t> inputSplitSizes_;
  // 成员变量，用于存储输入分割尺寸

  std::vector<int64_t> outputSplitSizes_;
  // 成员变量，用于存储输出分割尺寸

  int globalRankStart_{};
  // 成员变量，用于存储全局进程排名起始值，默认初始化为零

  int globalRankStride_{};
  // 成员变量，用于存储全局进程排名步长，默认初始化为零

  std::vector<int64_t> groupRanks_;
  // 成员变量，用于存储组排名
};

#define RECORD_PARAM_COMMS(                                                    \
    seq,                                                                       \
    pgName,                                                                    \
    rank,                                                                      \
    collName,                                                                  \
    inNelems,                                                                  \
    outNelems,                                                                 \
    dType,                                                                     \
    inSplitSizes,                                                              \
    outSplitSizes,                                                             \
    globalRankStart,                                                           \
    globalRankStride,                                                          \
    worldSize)                                                                 \
// 宏定义，用于记录参数通信信息，接受多个参数
    globalRankStride,                                                          \
    worldSize)                                                                 \
  // 创建一个共享指针，用于存储调试信息对象，这个对象用于跟踪参数通信的调试信息
  auto paramCommsInfo = std::make_shared<torch::ParamCommsDebugInfo>(          \
      pgName,                                                                  \
      rank,                                                                    \
      collName,                                                                \
      inNelems,                                                                \
      outNelems,                                                               \
      dType,                                                                   \
      inSplitSizes,                                                            \
      outSplitSizes,                                                           \
      globalRankStart,                                                         \
      globalRankStride,                                                        \
      worldSize);                                                              \
  // 设置调试信息守卫，指定调试信息的种类为 PARAM_COMMS_INFO，关联上述的调试信息对象
  c10::DebugInfoGuard g(c10::DebugInfoKind::PARAM_COMMS_INFO, paramCommsInfo); \
  // 创建初始化列表，包含参数通信函数所需的输入信息
  std::initializer_list<const c10::IValue> paramList = {                       \
      c10::IValue(seq),                                                        \
      pgName,                                                                  \
      rank,                                                                    \
      collName,                                                                \
      inSplitSizes,                                                            \
      outSplitSizes,                                                           \
      globalRankStart,                                                         \
      globalRankStride,                                                        \
      worldSize};                                                              \
  // 创建 ArrayRef 对象，用于封装初始化列表，作为参数传递给记录函数
  c10::ArrayRef<const c10::IValue> paramInputs(paramList);                     \
  // 记录函数调用，用于记录参数通信的函数调用，名称为 at::kParamCommsCallName，参数为 paramInputs
  RECORD_FUNCTION(at::kParamCommsCallName, paramInputs);
#define RECORD_PARAM_COMMS_DATA(                                               \
    seq,                                                                       \  // 定义宏 RECORD_PARAM_COMMS_DATA，接受多个参数
    pgName,                                                                    \  // 序列号
    InputTensors,                                                              \  // 程序组名
    OutputTensors,                                                             \  // 输入张量列表
    rank,                                                                      \  // 输出张量列表
    collName,                                                                  \  // 等级
    inNelems,                                                                  \  // 集合名
    outNelems,                                                                 \  // 输入元素数
    dType,                                                                     \  // 输出元素数
    inSplitSizes,                                                              \  // 数据类型
    outSplitSizes,                                                             \  // 输入分割大小
    globalRankStart,                                                           \  // 输出分割大小
    globalRankStride,                                                          \  // 全局等级起点
  // 定义一个共享指针，用于存储参数通信调试信息对象
  auto paramCommsInfo = std::make_shared<torch::ParamCommsDebugInfo>(
      pgName,                       // 进程组名称
      rank,                         // 进程的排名
      collName,                     // 通信集合名称
      inNelems,                     // 输入元素数量
      outNelems,                    // 输出元素数量
      dType,                        // 数据类型
      inSplitSizes,                 // 输入分割大小
      outSplitSizes,                // 输出分割大小
      globalRankStart,              // 全局排名起始值
      globalRankStride,             // 全局排名步长
      worldSize);                   // 总进程数
  // 使用调试信息保护器，设置调试信息类型为 PARAM_COMMS_INFO
  c10::DebugInfoGuard g(c10::DebugInfoKind::PARAM_COMMS_INFO, paramCommsInfo);
  // 初始化参数列表，包括输入张量、序列、进程组名称、排名、通信集合名称等
  std::initializer_list<const c10::IValue> paramList = {
      c10::IValue(InputTensors),    // 输入张量列表
      c10::IValue(seq),             // 序列
      pgName,                       // 进程组名称
      rank,                         // 进程的排名
      collName,                     // 通信集合名称
      inSplitSizes,                 // 输入分割大小
      outSplitSizes,                // 输出分割大小
      globalRankStart,              // 全局排名起始值
      globalRankStride,             // 全局排名步长
      worldSize                     // 总进程数
  };
  // 创建参数输入的 ArrayRef
  c10::ArrayRef<const c10::IValue> paramInputs(paramList);
  // 记录函数调用，包括函数名、参数输入、输出张量
  RECORD_FUNCTION_WITH_INPUTS_OUTPUTS(
      at::kParamCommsCallName,      // 参数通信调用名称
      paramInputs,                  // 参数输入列表
      std::vector<c10::IValue>(1, c10::IValue(OutputTensors))); // 输出张量
} // namespace torch
```