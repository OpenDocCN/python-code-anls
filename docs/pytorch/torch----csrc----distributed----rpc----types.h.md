# `.\pytorch\torch\csrc\distributed\rpc\types.h`

```
#pragma once

#include <ATen/core/ivalue.h>
#include <atomic>

namespace torch {
namespace distributed {
namespace rpc {

// 使用 int16_t 定义 worker_id_t 类型
using worker_id_t = int16_t;
// 使用 int64_t 定义 local_id_t 类型
using local_id_t = int64_t;

// 返回是否允许对 JIT RRef 进行 Pickle
bool getAllowJitRRefPickle();
// 允许对 JIT RRef 进行 Pickle
TORCH_API void enableJitRRefPickle();
// 禁止对 JIT RRef 进行 Pickle
TORCH_API void disableJitRRefPickle();

// 构造函数和析构函数，用于管理 JIT RRef Pickle 的开启和关闭
struct TORCH_API JitRRefPickleGuard {
  JitRRefPickleGuard();
  ~JitRRefPickleGuard();
};

// 表示全局唯一 ID 的结构体
struct TORCH_API GloballyUniqueId final {
  // 构造函数，初始化 createdOn 和 localId
  GloballyUniqueId(worker_id_t createdOn, local_id_t localId);
  // 默认复制构造函数，删除赋值操作符
  GloballyUniqueId(const GloballyUniqueId& other) = default;
  GloballyUniqueId& operator=(const GloballyUniqueId& other) = delete;

  // 比较操作符重载，判断两个 GloballyUniqueId 是否相等或不相等
  bool operator==(const GloballyUniqueId& other) const;
  bool operator!=(const GloballyUniqueId& other) const;

  // 将对象转换为 at::IValue 类型
  at::IValue toIValue() const;
  // 从 at::IValue 创建 GloballyUniqueId 对象
  static GloballyUniqueId fromIValue(const at::IValue&);

  // 哈希函数对象，用于将 GloballyUniqueId 映射到哈希值
  struct Hash {
    size_t operator()(const GloballyUniqueId& key) const {
      return (uint64_t(key.createdOn_) << kLocalIdBits) | key.localId_;
    }
  };

  // 定义 kLocalIdBits 常量为 48
  static constexpr int kLocalIdBits = 48;

  // 声明为常量成员变量，代表对象创建的 worker_id 和本地 ID
  const worker_id_t createdOn_;
  const local_id_t localId_;
};

// 重载 << 运算符，用于将 GloballyUniqueId 对象输出到流中
TORCH_API std::ostream& operator<<(
    std::ostream& os,
    const GloballyUniqueId& globalId);

// RRef 的全局唯一 ID，使用 GloballyUniqueId 类型
using RRefId = GloballyUniqueId;
// Fork 的全局唯一 ID，使用 GloballyUniqueId 类型
using ForkId = GloballyUniqueId;
// 用于性能分析的全局唯一 ID，使用 GloballyUniqueId 类型
using ProfilingId = GloballyUniqueId;

// 表示序列化的 Python 对象的结构体
struct TORCH_API SerializedPyObj final {
  // 构造函数，接受序列化的 payload 和张量列表 tensors
  SerializedPyObj(std::string&& payload, std::vector<at::Tensor>&& tensors)
      : payload_(std::move(payload)), tensors_(std::move(tensors)) {}

  // 将对象转换为 at::IValue 向量，并移动对象自身
  std::vector<at::IValue> toIValues() &&;
  // 从 at::IValue 向量创建 SerializedPyObj 对象
  static SerializedPyObj fromIValues(std::vector<at::IValue> value);

  // 存储序列化的 payload 字符串
  std::string payload_;
  // 存储序列化的张量列表
  std::vector<at::Tensor> tensors_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
```