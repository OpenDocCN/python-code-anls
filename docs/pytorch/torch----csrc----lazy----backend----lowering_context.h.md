# `.\pytorch\torch\csrc\lazy\backend\lowering_context.h`

```
#pragma once

#include <memory>  // 包含内存管理相关的头文件
#include <string>  // 包含字符串处理相关的头文件
#include <unordered_map>  // 包含无序映射容器相关的头文件
#include <utility>  // 包含实用程序功能的头文件
#include <vector>  // 包含向量容器相关的头文件

#include <torch/csrc/lazy/backend/backend_data.h>  // 包含 Torch 后端数据相关头文件
#include <torch/csrc/lazy/backend/backend_device.h>  // 包含 Torch 后端设备相关头文件
#include <torch/csrc/lazy/core/ir.h>  // 包含 Torch 惰性求值核心 IR 相关头文件
#include <torch/csrc/lazy/core/ir_util.h>  // 包含 Torch 惰性求值核心 IR 实用工具相关头文件

namespace torch {
namespace lazy {

class TORCH_API Computation {
 public:
  virtual int parameters_size() const = 0;  // 纯虚函数，返回参数的数量

  virtual const std::vector<Shape>& parameter_shapes() const = 0;  // 纯虚函数，返回参数的形状向量引用

  virtual const std::vector<std::string>& parameter_names() const = 0;  // 纯虚函数，返回参数的名称向量引用

  virtual const Shape& result_shape() const = 0;  // 纯虚函数，返回结果的形状引用

  virtual const std::string to_string() const = 0;  // 纯虚函数，返回对象描述的字符串

  virtual ~Computation() = default;  // 默认虚析构函数

  // 表示此计算是否在标记步骤内执行
  // 默认为 false，除非另有设定
  bool in_mark_step = false;
};

using ComputationPtr = std::shared_ptr<Computation>;  // Computation 的智能指针类型定义

// 保持代码生成状态的上下文
class TORCH_API LoweringContext {
 public:
  LoweringContext(const std::string& name, BackendDevice device);  // 构造函数，接受名称和后端设备参数

  LoweringContext(
      const std::string& name,
      BackendDevice device,
      c10::ArrayRef<const torch::lazy::Node*> post_order,
      Util::EmissionMap emit_status);  // 构造函数，接受名称、后端设备、节点数组引用和发射状态

  virtual ~LoweringContext() = default;  // 默认虚析构函数

  // 静态方法，创建 LoweringContext 对象，接受名称、后端设备、节点数组引用和发射状态
  static std::unique_ptr<LoweringContext> Create(
      const std::string& name,
      BackendDevice device,
      c10::ArrayRef<const torch::lazy::Node*> post_order,
      Util::EmissionMap emit_status);

  // 静态方法，创建 LoweringContext 对象，接受名称和后端设备参数
  static std::unique_ptr<LoweringContext> Create(
      const std::string& name,
      BackendDevice device);

  const BackendDevice& device() const;  // 返回后端设备对象的常引用

  // 返回存储与参数指令相关的所有张量的向量
  const std::vector<BackendDataPtr>& GetParametersData() const;

  // 添加新的输入/输出别名
  virtual void SetUpAlias(
      const std::vector<int64_t>& output_index,
      int64_t param_number,
      const std::vector<int64_t>& param_index,
      bool must_alias = false);  // 虚函数，默认实现为空操作

  // 检查参数形状是否与给定结果索引处的结果匹配
  virtual bool CheckResultShape(
      const BackendDataPtr& parameter_data,
      size_t result_idx);  // 虚函数，默认实现为空操作
};

}  // namespace lazy
}  // namespace torch
  // 返回 false
  return false;
}

  // 将给定的输出作为结果元组的一个组件添加，并返回其在元组中的位置。
  virtual size_t AddResult(const torch::lazy::Output& output) = 0;

  // 将给定的输出与指定索引和形状的输入参数关联起来。仅用于逐运算符执行，主要用于调试目的。
  virtual void AddParameter(
      const torch::lazy::Output& output,
      size_t index,
      const Shape& shape,
      const std::string& name) = 0;

  // 构建计算，捕获所有使用内嵌生成器（由 builder() API 返回）创建的操作。
  virtual ComputationPtr Build() = 0;

  // 返回已发出节点的数量
  size_t GetEmittedNodeCount() const {
    return emit_status_.size();
  }

protected:
  BackendDevice device_; // 后端设备
  std::vector<BackendDataPtr> parameters_; // 参数列表
  std::vector<size_t> parameter_sequence_; // 参数顺序
  Util::EmissionMap emit_status_; // 发射状态映射
};

} // namespace lazy
} // namespace torch
```