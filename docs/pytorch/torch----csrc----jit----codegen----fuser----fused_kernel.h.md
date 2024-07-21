# `.\pytorch\torch\csrc\jit\codegen\fuser\fused_kernel.h`

```py
#pragma once

#include <ATen/ATen.h>
#include <ATen/Utils.h>
#include <torch/csrc/jit/codegen/fuser/partition_desc.h>
#include <torch/csrc/jit/codegen/fuser/tensor_desc.h>

#include <cstdint>
#include <string>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {

// 结构体定义，表示一个融合内核
struct FusedKernel {
  AT_DISALLOW_COPY_AND_ASSIGN(FusedKernel);  // 禁止复制和赋值操作

  // 构造函数，初始化融合内核对象
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  FusedKernel(
      std::string name,  // 内核名称
      std::string code,  // 内核代码
      std::vector<TensorDesc> input_desc,   // 输入张量描述列表
      std::vector<TensorDesc> output_desc,  // 输出张量描述列表
      std::vector<PartitionDesc> chunk_desc,    // 分块描述列表
      std::vector<PartitionDesc> concat_desc,   // 连接描述列表
      bool has_random)  // 是否包含随机数
      : name_(std::move(name)),  // 初始化名称
        code_(std::move(code)),  // 初始化代码
        input_desc_(std::move(input_desc)),    // 初始化输入张量描述列表
        output_desc_(std::move(output_desc)),  // 初始化输出张量描述列表
        chunk_desc_(std::move(chunk_desc)),    // 初始化分块描述列表
        concat_desc_(std::move(concat_desc)),  // 初始化连接描述列表
        has_random_(has_random) {}  // 初始化随机数标志

  virtual ~FusedKernel() = default;  // 虚析构函数

  // launch_raw方法，纯虚函数，用于启动编译后的CUDA/CPU代码
  // arguments是指向编译后CUDA/CPU代码参数的指针列表
  // 适合直接传递给cuLaunchKernel作为内核参数
  // 目前，第一个参数是numel的指针（用于传递给CUDA代码），其余参数是指向编译后代码使用的TensorInfo<T>结构体的指针
  // launch_with_tensors方法负责将at::Tensor打包到这些参数中
  // CPU代码使用相同的约定，以便launch_with_tensors可以共享
  virtual void launch_raw(const uint32_t numel, std::vector<void*>& arguments)
      const = 0;
  
  // backend方法，纯虚函数，返回内核的后端类型
  virtual at::Backend backend() const = 0;

  // 获取器方法
  const std::string& name() const {  // 返回内核名称
    return name_;
  }
  const std::string& code() const {  // 返回内核代码
    return code_;
  }
  const std::vector<TensorDesc>& inputDesc() const {  // 返回输入张量描述列表
    return input_desc_;
  }
  const std::vector<TensorDesc>& outputDesc() const {  // 返回输出张量描述列表
    return output_desc_;
  }
  const std::vector<PartitionDesc>& chunkDesc() const {  // 返回分块描述列表
    return chunk_desc_;
  }
  const std::vector<PartitionDesc>& concatDesc() const {  // 返回连接描述列表
    return concat_desc_;
  }
  bool hasRandom() const {  // 返回是否包含随机数
    return has_random_;
  }

private:
  std::string name_;  // 内核名称
  std::string code_;  // 内核代码
  std::vector<TensorDesc> input_desc_;   // 输入张量描述列表
  std::vector<TensorDesc> output_desc_;  // 输出张量描述列表
  std::vector<PartitionDesc> chunk_desc_;    // 分块描述列表
  std::vector<PartitionDesc> concat_desc_;   // 连接描述列表
  bool has_random_;  // 是否包含随机数
};

} // namespace fuser
} // namespace jit
} // namespace torch
    // 返回成员变量 has_random_
      }
      
     protected:
      // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
      // 对象名称
      const std::string name_;
      // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
      // 对象代码
      const std::string code_;
      // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
      // 输入张量描述信息的向量
      const std::vector<TensorDesc> input_desc_;
      // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
      // 输出张量描述信息的向量
      const std::vector<TensorDesc> output_desc_;
    
      // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
      // 输入描述与是否分解为子张量（块）相关的描述信息向量，大小与 input_desc_ 相同
      const std::vector<PartitionDesc> chunk_desc_;
    
      // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
      // 输出描述与是否为融合组生成的许多子张量的串联相关的描述信息向量，大小与 output_desc_ 相同
      const std::vector<PartitionDesc> concat_desc_;
    
      // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
      // 表示对象是否包含随机性质
      const bool has_random_;
};

// 结束 fuser 命名空间定义

} // namespace fuser
// 结束 jit 命名空间定义

} // namespace jit
// 结束 torch 命名空间定义

} // namespace torch
```