# `.\pytorch\torch\csrc\jit\serialization\storage_context.h`

```
#pragma once
// 预处理指令，确保头文件仅被包含一次

#include <ATen/core/ivalue.h>
// 包含 ATen 库的 IValue 头文件

namespace torch::jit {

// 声明命名空间 torch::jit

// 用于 torch.package 和 TorchScript 序列化，协调模型之间共享存储。
// 也用于为存储创建确定性命名。
class TORCH_API SerializationStorageContext {
 public:
  explicit SerializationStorageContext() = default;
  // 显式默认构造函数

  SerializationStorageContext operator=(const SerializationStorageContext&) =
      delete;
  // 删除赋值运算符重载

  SerializationStorageContext(const SerializationStorageContext&) = delete;
  // 删除拷贝构造函数

  uint64_t getOrAddStorage(const c10::Storage& storage) {
    // 获取或添加存储到映射关系中
    if (!hasStorage(storage)) {
      uint64_t size = storage_id_map_.size();
      storage_id_map_[storage] = size;
    }
    return storage_id_map_[storage];
  }

  bool hasStorage(const c10::Storage& storage) {
    // 检查存储是否存在于映射关系中
    return storage_id_map_.find(storage) != storage_id_map_.end();
  }

  ~SerializationStorageContext() = default;
  // 默认析构函数

 private:
  // 内部类，用于存储的哈希函数对象
  class StorageSerializationHash {
   public:
    size_t operator()(const c10::Storage& storage) const {
      return std::hash<void*>()(
          reinterpret_cast<void*>(storage.unsafeGetStorageImpl()));
    }
  };

  // 内部类，用于存储的相等性比较对象
  class StorageSerializationEqual {
   public:
    bool operator()(const c10::Storage& lhs, const c10::Storage& rhs) const {
      return lhs.unsafeGetStorageImpl() == rhs.unsafeGetStorageImpl();
    }
  };

  std::unordered_map<
      c10::Storage,
      uint64_t,
      StorageSerializationHash,
      StorageSerializationEqual>
      storage_id_map_;
  // 存储到 ID 的映射关系
};

// 用于 torch.package 和 TorchScript 反序列化，协调模型之间共享存储。
class TORCH_API DeserializationStorageContext {
 public:
  explicit DeserializationStorageContext() = default;
  // 显式默认构造函数

  DeserializationStorageContext operator=(
      const DeserializationStorageContext&) = delete;
  // 删除赋值运算符重载

  DeserializationStorageContext(const DeserializationStorageContext&) = delete;
  // 删除拷贝构造函数

  void addStorage(std::string name, c10::Storage storage) {
    // 添加存储和其名称的映射关系
    TORCH_INTERNAL_ASSERT(!hasStorage(name));
    name_storage_map_.emplace(std::move(name), std::move(storage));
  }

  bool hasStorage(const std::string& name) {
    // 检查指定名称的存储是否存在
    return name_storage_map_.find(name) != name_storage_map_.end();
  }

  c10::Storage getStorage(const std::string& name) {
    // 获取指定名称的存储
    TORCH_INTERNAL_ASSERT(hasStorage(name));
    return name_storage_map_.find(name)->second;
  }
  ~DeserializationStorageContext() = default;
  // 默认析构函数

 private:
  std::unordered_map<std::string, c10::Storage> name_storage_map_;
  // 名称到存储的映射关系
};

} // namespace torch::jit
// 命名空间结束
```