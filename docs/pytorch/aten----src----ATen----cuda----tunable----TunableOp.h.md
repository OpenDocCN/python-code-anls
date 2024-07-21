# `.\pytorch\aten\src\ATen\cuda\tunable\TunableOp.h`

```
// Original TunableOp is from onnxruntime.
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/framework/tunable.h
// https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/rocm/tunable
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Adapting TunableOp into PyTorch
// Copyright (c) Advanced Micro Devices, Inc.
//
#pragma once

#include <ATen/cuda/tunable/Tunable.h>  // 包含 ATen CUDA 可调优组件的头文件 Tunable.h
#include <ATen/cuda/Sleep.h>            // 包含 ATen CUDA 的 Sleep 功能的头文件
#include <c10/cuda/CUDACachingAllocator.h>  // 包含 C10 CUDA 内存缓存分配器的头文件

#ifndef _WIN32
#include <cxxabi.h>  // 如果不是在 Windows 平台，包含 cxxabi.h 头文件
#endif

#include <string>              // 包含 C++ 标准库中的 string 类
#include <type_traits>         // 包含 C++ 标准库中的类型特性支持
#include <unordered_map>       // 包含 C++ 标准库中的无序映射容器支持
#include <vector>              // 包含 C++ 标准库中的向量容器支持

namespace at::cuda::tunable {

template <typename ParamsT>
class Callable {
  public:
    Callable() = default;           // 默认构造函数
    Callable(Callable&&) = default; // 移动构造函数
    virtual ~Callable() = default;  // 虚拟析构函数
    virtual TuningStatus Call(const ParamsT*) {  // 虚拟函数 Call，返回调优状态
      return FAIL;                  // 默认返回失败状态
    }
    virtual TuningStatus IsSupported(const ParamsT* params) {  // 虚拟函数 IsSupported，检查参数是否支持
      return Call(params);          // 调用 Call 函数检查支持状态
    }
};

template <typename ParamsT, typename TimerT>
class TunableOp {
  public:
    TunableOp() = default;           // 默认构造函数
    TunableOp(TunableOp&&) = default; // 移动构造函数
    virtual ~TunableOp() = default;  // 虚拟析构函数

    TuningStatus operator()(const ParamsT* params) {  // 调用运算符重载函数，返回调优状态
      ResultEntry result = ResultEntry::Null();  // 初始化结果条目为空
      TuningContext* ctx = getTuningContext();  // 获取调优上下文
      if (ctx->IsTunableOpEnabled()) {  // 如果可调优操作已启用
        auto& mgr = ctx->GetTuningResultsManager();  // 获取调优结果管理器
        auto op_sig = Signature();  // 获取操作的签名
        auto params_sig = params->Signature();  // 获取参数的签名
        result = mgr.Lookup(op_sig, params_sig);  // 查询调优结果
        // 如果没有找到先前的调优结果，则在启用调优时进行调优
        if (result == ResultEntry::Null() && ctx->IsTuningEnabled()) {
          result = FindFastest(params);  // 查找最快的调优结果
          mgr.Add(op_sig, params_sig, result);  // 添加调优结果到管理器中
        }
      }
      else {
        result = ResultEntry::Default();  // 否则使用默认结果
      }
      if (result == ResultEntry::Null()) {  // 如果结果仍为空
        TUNABLE_LOG2("no result, using default");  // 记录日志，使用默认结果
        result = ResultEntry::Default();  // 使用默认结果
      }
      auto iter = ops_.find(result);  // 查找结果对应的操作迭代器
      TORCH_CHECK(iter != ops_.end());  // 断言确保迭代器有效
      return iter->second->Call(params);  // 调用操作的 Call 函数并返回结果
    }

    virtual std::string Signature() {  // 虚拟函数，返回操作的签名字符串
      // 根据 C++17 标准 https://wg21.link/n4659 第 15.7.4 节
      // > 如果 typeid 的操作数引用正在构造或销毁的对象，则 typeid 返回代表构造函数或析构函数类的 std::type_info 对象。
      // 因此延迟操作签名的生成。
      c10::call_once(signature_init_once_, [this]() { signature_ = CreateSignature(); });  // 仅调用一次创建操作签名
      return signature_;  // 返回操作签名
    }

  protected:
    void RegisterOp(const std::string& name, std::unique_ptr<Callable<ParamsT>> op) {
      this->op_names_.emplace_back(name);  // 注册操作名称到操作名列表
      this->ops_.emplace(name, std::move(op));  // 移动操作到操作映射中
    }

  private:
    // 执行预热操作以确保操作和参数在实际调用前已经准备好
    static void WarmUp(Callable<ParamsT> *op, const std::vector<ParamsT*> &param, size_t num_iter, size_t &offset) {
      // 获取调优上下文
      TuningContext* ctx = getTuningContext();
      // 检查是否启用指令缓存刷新
      bool do_flush = ctx->IsICacheFlushEnabled();
      // 循环执行预热操作
      for (size_t i = 0; i < num_iter; i++) {
        // 如果启用了指令缓存刷新，则刷新 CUDA 指令缓存
        if (do_flush) {
          at::cuda::flush_icache();
        }
        // 调用操作对象的 Call 方法，并检查返回状态是否为 OK
        TORCH_CHECK(op->Call(param[(i+offset++)%param.size()]) == OK);
      }
    }

    // 执行性能分析操作，返回每次迭代的平均持续时间
    static double Profile(Callable<ParamsT> *op, const std::vector<ParamsT*> &param, size_t num_iter, size_t &offset) {
      // 获取调优上下文
      TuningContext* ctx = getTuningContext();
      // 检查是否启用指令缓存刷新
      bool do_flush = ctx->IsICacheFlushEnabled();
      // 创建计时器对象并开始计时
      TimerT timer{};
      timer.Start();
      // 循环执行性能分析操作
      for (size_t i = 0; i < num_iter; i++) {
        // 如果启用了指令缓存刷新，则刷新 CUDA 指令缓存
        if (do_flush) {
          at::cuda::flush_icache();
        }
        // 调用操作对象的 Call 方法，并检查返回状态是否为 OK
        TORCH_CHECK(op->Call(param[(i+offset++)%param.size()]) == OK);
      }
      // 结束计时并返回每次迭代的平均持续时间
      timer.End();
      return timer.Duration() / num_iter;
    }
#ifndef _WIN32
      // 如果不是在 Windows 平台下
      const auto* name = typeid(*this).name();
      // 获取当前对象的类型名
      char buf[256];
      // 定义一个字符数组用于存放 demangle 后的类型名，长度为 256
      size_t buf_len = 256;
      // 设置 buf 的长度为 256
      abi::__cxa_demangle(name, buf, &buf_len, nullptr);
      // 对类型名进行 demangle，并将结果存放到 buf 中
      buf[255] = '\0';
      // 将 buf 的最后一个字符设置为 '\0'，以确保字符串结尾
      return buf;
      // 返回 demangle 后的类型名
#else
      // 如果是在 Windows 平台下
      return typeid(*this).name();
      // 直接返回当前对象的类型名
#endif
    }

    mutable c10::once_flag signature_init_once_;
    // 可变的 once_flag 对象，用于标记签名初始化的单次执行

    std::string signature_;
    // 用于存储对象的签名字符串

    std::unordered_map<std::string, std::unique_ptr<Callable<ParamsT>>> ops_;
    // 哈希映射，键为字符串，值为指向可调用对象的唯一指针，用于存储操作

    std::vector<std::string> op_names_;
    // 存储操作的名称的向量

};

struct OpParams {
  OpParams() {}
  // 默认构造函数

  virtual ~OpParams() = default;
  // 虚析构函数，用于多态释放资源

  virtual std::string Signature() const = 0;
  // 纯虚函数，用于返回参数对象的签名字符串
};

} // namespace at::cuda::tunable
// 结束 at::cuda::tunable 命名空间的声明
```