# `.\pytorch\torch\csrc\api\include\torch\data\samplers\random.h`

```
#pragma once

# 预处理指令，确保此头文件只被编译一次


#include <torch/csrc/Export.h>
#include <torch/data/samplers/base.h>
#include <torch/types.h>

#include <cstddef>
#include <vector>

# 包含必要的头文件，用于 Torch 库和数据采样器


namespace torch {
namespace serialize {
class OutputArchive;
class InputArchive;
} // namespace serialize
} // namespace torch

# 命名空间定义：torch 序列化相关的类 OutputArchive 和 InputArchive


namespace torch {
namespace data {
namespace samplers {

# 命名空间定义：torch 数据模块中的采样器相关实现


/// A `Sampler` that returns random indices.
class TORCH_API RandomSampler : public Sampler<> {
 public:

# RandomSampler 类的声明，继承自 Sampler<>，用于返回随机索引的采样器


  /// Constructs a `RandomSampler` with a size and dtype for the stored indices.
  ///
  /// The constructor will eagerly allocate all required indices, which is the
  /// sequence `0 ... size - 1`. `index_dtype` is the data type of the stored
  /// indices. You can change it to influence memory usage.
  explicit RandomSampler(int64_t size, Dtype index_dtype = torch::kInt64);

# 构造函数声明：使用给定的大小和索引数据类型创建 RandomSampler 对象


  ~RandomSampler() override;

# 析构函数声明：清理 RandomSampler 对象所占用的资源


  /// Resets the `RandomSampler` to a new set of indices.
  void reset(optional<size_t> new_size = nullopt) override;

# 重置方法声明：将 RandomSampler 重置为新的索引集合


  /// Returns the next batch of indices.
  optional<std::vector<size_t>> next(size_t batch_size) override;

# 下一个批次索引的方法声明：返回下一个批次的索引集合


  /// Serializes the `RandomSampler` to the `archive`.
  void save(serialize::OutputArchive& archive) const override;

# 序列化方法声明：将 RandomSampler 序列化到指定的输出存档中


  /// Deserializes the `RandomSampler` from the `archive`.
  void load(serialize::InputArchive& archive) override;

# 反序列化方法声明：从输入存档中反序列化 RandomSampler 对象


  /// Returns the current index of the `RandomSampler`.
  size_t index() const noexcept;

# 获取当前索引方法声明：返回当前 RandomSampler 的索引值


 private:
  at::Tensor indices_;
  int64_t index_ = 0;
};

# 私有成员变量声明：存储索引的 Tensor 对象和当前索引值


} // namespace samplers
} // namespace data
} // namespace torch

# 命名空间结束：结束 torch 数据模块中的采样器相关实现声明
```