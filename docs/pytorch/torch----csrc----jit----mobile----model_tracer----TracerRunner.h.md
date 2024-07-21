# `.\pytorch\torch\csrc\jit\mobile\model_tracer\TracerRunner.h`

```
#pragma once
// 只要引用一次，这个头文件就会被包含在程序中

#include <set>
// 引入集合（set）模块，用于存储唯一元素的数据结构
#include <string>
// 引入字符串（string）模块，用于处理字符串数据
#include <vector>
// 引入向量（vector）模块，用于存储动态数组

#include <ATen/core/ivalue.h>
// 引入 ATen 库中的 IValue 类，用于表示 PyTorch 中的各种值
#include <torch/csrc/jit/mobile/model_tracer/BuildFeatureTracer.h>
// 引入 Torch 移动端模型追踪库中的 BuildFeatureTracer 类
#include <torch/csrc/jit/mobile/model_tracer/CustomClassTracer.h>
// 引入 Torch 移动端模型追踪库中的 CustomClassTracer 类
#include <torch/csrc/jit/mobile/model_tracer/KernelDTypeTracer.h>
// 引入 Torch 移动端模型追踪库中的 KernelDTypeTracer 类

namespace torch {
namespace jit {
namespace mobile {

const std::vector<std::string> always_included_traced_ops = {
    // 下列操作在设置部分被调用。
    "aten::resize_",
    // 调整尺寸的 ATen 操作
    "aten::slice.Tensor",
    // 对张量进行切片的 ATen 操作
};

struct TracerResult {
  std::set<std::string> root_ops;
  // 根操作的集合
  std::set<std::string> traced_operators;
  // 追踪操作的集合
  KernelDTypeTracer::kernel_tags_type called_kernel_tags;
  // 调用内核数据类型追踪器的内核标签类型
  CustomClassTracer::custom_classes_type loaded_classes;
  // 加载类的自定义类追踪器的自定义类类型
  BuildFeatureTracer::build_feature_type build_features;
  // 构建特性追踪器的构建特性类型
  std::set<std::string> enabled_backends;
  // 启用的后端集合
};

/**
 * Trace a single model and return the TracerResult.
 */
TracerResult trace_run(const std::string& input_module_path);
// 跟踪单个模型并返回 TracerResult 结构

/**
 * Trace multiple models and return the TracerResult.
 */
TracerResult trace_run(const std::vector<std::string>& input_module_paths);
// 跟踪多个模型并返回 TracerResult 结构

} // namespace mobile
} // namespace jit
} // namespace torch
```