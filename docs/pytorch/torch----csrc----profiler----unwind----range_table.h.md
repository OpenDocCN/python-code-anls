# `.\pytorch\torch\csrc\profiler\unwind\range_table.h`

```
#pragma once
// 包含 unwind_error.h 文件，这是 Torch Profiler 的堆栈解析器相关错误处理头文件
#include <torch/csrc/profiler/unwind/unwind_error.h>
// 包含必要的标准库头文件
#include <algorithm>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

// torch::unwind 命名空间开始
namespace torch::unwind {

// 模板结构体 RangeTable 开始
template <typename T>
struct RangeTable {
  // 构造函数
  RangeTable() {
    // 确保 lower_bound[-1] 始终有效
    addresses_.push_back(0);
    payloads_.emplace_back(std::nullopt);
  }

  // 添加地址及其 payload 到表中
  void add(uint64_t address, unwind::optional<T> payload, bool sorted) {
    // 检查地址是否有序，如果不是则标记为未排序状态
    if (addresses_.back() > address) {
      UNWIND_CHECK(!sorted, "expected addresses to be sorted");
      sorted_ = false;
    }
    // 将地址和 payload 添加到相应的容器中
    addresses_.push_back(address);
    payloads_.emplace_back(std::move(payload));
  }

  // 查找给定地址的 payload
  unwind::optional<T> find(uint64_t address) {
    // 如果未排序，则执行排序
    maybeSort();
    // 使用 std::upper_bound 查找大于给定地址的第一个元素
    auto it = std::upper_bound(addresses_.begin(), addresses_.end(), address);
    // 返回地址对应的 payload
    return payloads_.at(it - addresses_.begin() - 1);
  }

  // 打印所有地址及其 payloads
  void dump() {
    for (size_t i = 0; i < addresses_.size(); i++) {
      // 使用 fmt::print 输出地址和对应的 payload 或 "END"（如果 payload 为空）
      fmt::print("{} {:x}: {}\n", i, addresses_[i], payloads_[i] ? "" : "END");
    }
  }

  // 返回表的大小
  size_t size() const {
    return addresses_.size();
  }

  // 返回最后一个地址
  uint64_t back() {
    // 如果未排序，则执行排序
    maybeSort();
    return addresses_.back();
  }

 private:
  // 如果未排序，则执行排序
  void maybeSort() {
    if (sorted_) {
      return;
    }
    // 创建索引向量并初始化
    std::vector<uint64_t> indices;
    indices.reserve(addresses_.size());
    for (size_t i = 0; i < addresses_.size(); i++) {
      indices.push_back(i);
    }
    // 使用自定义排序函数对索引进行排序，以保证地址有序，并按 payload 存在性进行次级排序
    std::sort(indices.begin(), indices.end(), [&](uint64_t a, uint64_t b) {
      return addresses_[a] < addresses_[b] ||
             (addresses_[a] == addresses_[b] &&
              bool(payloads_[a]) < bool(payloads_[b]));
    });
    // 根据排序后的索引重新组织地址和 payloads
    std::vector<uint64_t> addresses;
    std::vector<unwind::optional<T>> payloads;
    addresses.reserve(addresses_.size());
    payloads.reserve(addresses_.size());
    for (auto i : indices) {
      addresses.push_back(addresses_[i]);
      payloads.push_back(payloads_[i]);
    }
    // 更新 addresses_ 和 payloads_ 为排序后的结果
    addresses_ = std::move(addresses);
    payloads_ = std::move(payloads);
    sorted_ = true;
  }

  // 标志表是否已排序
  bool sorted_ = true;
  // 存储地址的容器
  std::vector<uint64_t> addresses_;
  // 存储 payloads 的容器
  std::vector<unwind::optional<T>> payloads_;
};

} // namespace torch::unwind
```