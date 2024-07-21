# `.\pytorch\torch\csrc\distributed\c10d\FileStore.hpp`

```py
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <sys/types.h>
// 包含系统类型相关的头文件

#include <mutex>
// 包含互斥锁的头文件

#include <unordered_map>
// 包含无序映射的头文件

#include <torch/csrc/distributed/c10d/Store.hpp>
// 包含Torch分布式存储相关的头文件

namespace c10d {

class TORCH_API FileStore : public Store {
 public:
  explicit FileStore(std::string path, int numWorkers);
  // 构造函数，初始化FileStore对象

  ~FileStore() override;
  // 虚析构函数，用于资源释放

  void set(const std::string& key, const std::vector<uint8_t>& value) override;
  // 设置键值对

  std::vector<uint8_t> compareSet(
      const std::string& key,
      const std::vector<uint8_t>& expectedValue,
      const std::vector<uint8_t>& desiredValue) override;
  // 比较并设置键值对

  std::vector<uint8_t> get(const std::string& key) override;
  // 获取键对应的值

  int64_t add(const std::string& key, int64_t value) override;
  // 将指定键的值增加特定量

  int64_t getNumKeys() override;
  // 获取键的数量

  bool deleteKey(const std::string& key) override;
  // 删除指定键

  bool check(const std::vector<std::string>& keys) override;
  // 检查多个键是否存在

  void wait(const std::vector<std::string>& keys) override;
  // 等待多个键变为可用状态

  void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) override;
  // 在指定时间内等待多个键变为可用状态

  // Returns the path used by the FileStore.
  const std::string& getPath() const noexcept {
    return path_;
  }
  // 返回FileStore使用的路径

 protected:
  int64_t addHelper(const std::string& key, int64_t i);
  // 辅助函数，用于增加键对应的值

  std::string path_;
  // 存储FileStore使用的路径

  off_t pos_{0};
  // 文件偏移量初始化为0

  int numWorkers_;
  // 工作线程数量

  const std::string cleanupKey_;
  const std::string refCountKey_;
  const std::string regularPrefix_;
  const std::string deletePrefix_;
  // 存储不同类型的键前缀

  std::unordered_map<std::string, std::vector<uint8_t>> cache_;
  // 使用无序映射缓存键值对

  std::mutex activeFileOpLock_;
  // 互斥锁，用于保护并发操作
};

} // namespace c10d
// 结束c10d命名空间
```