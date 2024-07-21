# `.\pytorch\torch\csrc\distributed\c10d\PrefixStore.hpp`

```py
#pragma once

#include <torch/csrc/distributed/c10d/Store.hpp>

namespace c10d {

// PrefixStore 类继承自 Store 类，用于添加键前缀的存储操作
class TORCH_API PrefixStore : public Store {
 public:
  // 构造函数，初始化 PrefixStore 对象
  explicit PrefixStore(std::string prefix, c10::intrusive_ptr<Store> store);

  // 使用基类 Store 的 set 函数，设置指定键的值
  using Store::set;
  void set(const std::string& key, const std::vector<uint8_t>& value) override;

  // 使用基类 Store 的 compareSet 函数，比较并设置指定键的值
  using Store::compareSet;
  std::vector<uint8_t> compareSet(
      const std::string& key,
      const std::vector<uint8_t>& expectedValue,
      const std::vector<uint8_t>& desiredValue) override;

  // 使用基类 Store 的 get 函数，获取指定键的值
  std::vector<uint8_t> get(const std::string& key) override;

  // 使用基类 Store 的 add 函数，向指定键的值添加整数值
  int64_t add(const std::string& key, int64_t value) override;

  // 使用基类 Store 的 deleteKey 函数，删除指定键
  bool deleteKey(const std::string& key) override;

  // 使用基类 Store 的 getNumKeys 函数，获取键的数量
  int64_t getNumKeys() override;

  // 使用基类 Store 的 check 函数，检查指定键是否存在
  bool check(const std::vector<std::string>& keys) override;

  // 使用基类 Store 的 wait 函数，等待指定键的出现
  void wait(const std::vector<std::string>& keys) override;

  // 使用基类 Store 的 wait 函数，带超时等待指定键的出现
  void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) override;

  // 使用基类 Store 的 getTimeout 函数，获取超时时间
  const std::chrono::milliseconds& getTimeout() const noexcept override;

  // 使用基类 Store 的 setTimeout 函数，设置超时时间
  void setTimeout(const std::chrono::milliseconds& timeout) override;

  // 使用基类 Store 的 append 函数，向指定键的值追加数据
  void append(const std::string& key, const std::vector<uint8_t>& value)
      override;

  // 使用基类 Store 的 multiGet 函数，批量获取指定键的值
  std::vector<std::vector<uint8_t>> multiGet(
      const std::vector<std::string>& keys) override;

  // 使用基类 Store 的 multiSet 函数，批量设置指定键的值
  void multiSet(
      const std::vector<std::string>& keys,
      const std::vector<std::vector<uint8_t>>& values) override;

  // 返回 true，表示此存储支持 append、multiGet 和 multiSet 扩展 API
  bool hasExtendedApi() const override;

  // 获取底层存储对象的指针
  c10::intrusive_ptr<Store> getUnderlyingStore();

  // 递归获取被 PrefixStore 包装前的底层存储对象的指针
  c10::intrusive_ptr<Store> getUnderlyingNonPrefixStore();

 protected:
  std::string prefix_;  // 存储的键的前缀
  c10::intrusive_ptr<Store> store_;  // 内部使用的底层存储对象

  // 将前缀与键结合成完整的键名
  std::string joinKey(const std::string& key);

  // 将前缀与键列表结合成完整的键名列表
  std::vector<std::string> joinKeys(const std::vector<std::string>& keys);
};

} // namespace c10d
```