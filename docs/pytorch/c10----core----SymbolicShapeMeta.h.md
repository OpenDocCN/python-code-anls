# `.\pytorch\c10\core\SymbolicShapeMeta.h`

```py
#pragma once
// 包含必要的头文件以及声明 C10_API 宏

#include <c10/core/SymBool.h>
#include <c10/core/SymInt.h>
#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>
#include <c10/util/DimVector.h>

#include <atomic>  // 原子操作相关的头文件
#include <cstdint> // 固定大小整数类型的头文件
#include <mutex>   // 互斥锁的头文件
#include <utility> // 各种实用工具的头文件

namespace c10 {

class C10_API SymbolicShapeMeta {
 public:
  SymDimVector sizes_ = {0};          // 存储维度大小的符号化向量
  SymDimVector strides_ = {1};        // 存储步幅的符号化向量
  SymInt storage_offset_ = 0;         // 存储偏移量的符号化整数

  bool strides_valid_ = true;         // 标记步幅是否有效，例如对于稀疏张量可能无效

  SymbolicShapeMeta() = default;      // 默认构造函数
  SymbolicShapeMeta(const SymbolicShapeMeta& other);  // 拷贝构造函数

  // 刷新元素数量的状态
  void refresh_numel() {
    // 非 const 函数，不需要持有 mutables_ 锁
    available_.fetch_and(~numel_avail);
    numel_ = 1;
  }

  // 刷新连续性的状态
  void refresh_contiguous() {
    // 非 const 函数，不需要持有 mutables_ 锁
    available_.fetch_and(numel_avail);
    is_contiguous_ = false;
    is_channels_last_contiguous_ = false;
    is_channels_last_3d_contiguous_ = false;
    is_channels_last_ = false;
    is_channels_last_3d_ = false;
    is_non_overlapping_and_dense_ = false;
  }

  int64_t dim() const {
    return static_cast<int64_t>(sizes_.size());  // 返回维度的数量
  }

  // 访问派生量的访问器，首次访问时延迟计算

  bool has_numel() const {
    return available_.load() & numel_avail;  // 检查是否有元素数量可用
  }
  bool has_is_contiguous() const {
    return available_.load() & is_contiguous_avail;  // 检查是否有连续性信息可用
  }
  bool has_is_channels_last_contiguous() const {
    return available_.load() & is_channels_last_contiguous_avail;  // 检查是否有通道最后连续性信息可用
  }
  bool has_is_channels_last_3d_contiguous() const {
    return available_.load() & is_channels_last_3d_contiguous_avail;  // 检查是否有三维通道最后连续性信息可用
  }
  bool has_is_channels_last() const {
    return available_.load() & is_channels_last_avail;  // 检查是否有通道最后信息可用
  }
  bool has_is_channels_last_3d() const {
    return available_.load() & is_channels_last_3d_avail;  // 检查是否有三维通道最后信息可用
  }
  bool has_is_non_overlapping_and_dense() const {
    return available_.load() & is_non_overlapping_and_dense_avail;  // 检查是否有非重叠且密集信息可用
  }

  // 访问缓存的派生属性
  // 不要在持有 mutables_ 锁的情况下调用这些函数

  const SymInt& numel() const {
    if (C10_UNLIKELY(!has_numel())) {  // 如果没有元素数量信息可用，则初始化
      init_numel();
    }
    return numel_;  // 返回元素数量的符号化整数
  }

  const SymBool& is_contiguous() const {
    if (C10_UNLIKELY(!has_is_contiguous())) {  // 如果没有连续性信息可用，则初始化
      init_is_contiguous();
    }
    return is_contiguous_;  // 返回连续性状态的符号化布尔值
  }

  const SymBool& is_channels_last_contiguous() const {
    if (C10_UNLIKELY(!has_is_channels_last_contiguous())) {  // 如果没有通道最后连续性信息可用，则初始化
      init_is_channels_last_contiguous();
    }
    return is_channels_last_contiguous_;  // 返回通道最后连续性状态的符号化布尔值
  }

  const SymBool& is_channels_last_3d_contiguous() const {
    if (C10_UNLIKELY(!has_is_channels_last_3d_contiguous())) {  // 如果没有三维通道最后连续性信息可用，则初始化
      init_is_channels_last_3d_contiguous();
    }
    return is_channels_last_3d_contiguous_;  // 返回三维通道最后连续性状态的符号化布尔值
  }

  const SymBool& is_channels_last() const {
    if (C10_UNLIKELY(!has_is_channels_last())) {  // 如果没有通道最后信息可用，则初始化
      init_is_channels_last();
    }
    return is_channels_last_;  // 返回通道最后状态的符号化布尔值
  }

  const SymBool& is_channels_last_3d() const {
    // 如果没有三维通道最后信息可用，则初始化
    if (C10_UNLIKELY(!has_is_channels_last_3d())) {
      init_is_channels_last_3d();
    }
    return is_channels_last_3d_;  // 返回三维通道最后状态的符号化布尔值
  }
  
  // private:
    // 如果没有已经确定是 channels last 的信息，则初始化它
    if (C10_UNLIKELY(!has_is_channels_last_3d())) {
      init_is_channels_last_3d();
    }
    // 返回当前对象存储的 channels last 3d 的状态
    return is_channels_last_3d_;
  }

  // 返回当前对象存储的 non overlapping and dense 的状态
  const SymBool& is_non_overlapping_and_dense() const {
    // 如果没有已经确定是 non overlapping and dense 的信息，则初始化它
    if (C10_UNLIKELY(!has_is_non_overlapping_and_dense())) {
      init_is_non_overlapping_and_dense();
    }
    return is_non_overlapping_and_dense_;
  }

  // 假设以下函数执行的前提条件，以便可以跳过计算步骤
  // 注意：由于这些变量不是常量，不需要锁定 mutables_
  void assume_contiguous(SymBool val = true) {
    // 将 is_contiguous_ 设置为指定的值
    is_contiguous_ = std::move(val);
    // 将 is_contiguous_avail 标记位添加到 available_ 中
    available_.fetch_or(is_contiguous_avail);
  }
  void assume_channels_last_contiguous(SymBool val = true) {
    // 将 is_contiguous_ 设置为指定的值
    is_contiguous_ = std::move(val);
    // 将 is_channels_last_contiguous_avail 标记位添加到 available_ 中
    available_.fetch_or(is_channels_last_contiguous_avail);
  }
  void assume_channels_last_3d_contiguous(SymBool val = true) {
    // 将 is_channels_last_3d_contiguous_ 设置为指定的值
    is_channels_last_3d_contiguous_ = std::move(val);
    // 将 is_channels_last_3d_contiguous_avail 标记位添加到 available_ 中
    available_.fetch_or(is_channels_last_3d_contiguous_avail);
  }
  void assume_channels_last(SymBool val = true) {
    // 将 is_channels_last_ 设置为指定的值
    is_channels_last_ = std::move(val);
    // 将 is_channels_last_avail 标记位添加到 available_ 中
    available_.fetch_or(is_channels_last_avail);
  }
  void assume_channels_last_3d(SymBool val = true) {
    // 将 is_channels_last_3d_ 设置为指定的值
    is_channels_last_3d_ = std::move(val);
    // 将 is_channels_last_3d_avail 标记位添加到 available_ 中
    available_.fetch_or(is_channels_last_3d_avail);
  }
  void assume_non_overlapping_and_dense(SymBool val = true) {
    // 将 is_non_overlapping_and_dense_ 设置为指定的值
    is_non_overlapping_and_dense_ = std::move(val);

    // 将 is_non_overlapping_and_dense_ 设置为指定的值
    is_non_overlapping_and_dense_ = std::move(val);
    // 将 is_non_overlapping_and_dense_avail 标记位添加到 available_ 中
    available_.fetch_or(is_non_overlapping_and_dense_avail);
}
  // 设置 available_ 的指定位为 is_non_overlapping_and_dense_avail，使用原子操作确保线程安全
  available_.fetch_or(is_non_overlapping_and_dense_avail);
}

private:
SymBool compute_contiguous() const;
SymBool compute_channels_last_contiguous_2d() const;
SymBool compute_channels_last_contiguous_3d() const;
SymBool compute_strides_like_channels_last_2d() const;
SymBool compute_strides_like_channels_last_3d() const;
SymBool compute_non_overlapping_and_dense() const;

// 这些函数是对真正的 compute_ 函数的包装，可以利用其他连续性字段来进行短路优化。
// 对于 SymBool，需要单独实现，因为 SymBool 不支持短路优化。
// TODO: SymBool 的情况是否应避免短路优化？需要推理是否正确，以及简化表达是否更适合分析（也许不是！）

SymBool compute_channels_last_contiguous_3d_dim5() const;
SymBool compute_channels_last_2d_dim5() const;
SymBool compute_channels_last_3d_dim5() const;
SymBool compute_is_non_overlapping_and_dense_dim4() const;
SymBool compute_is_non_overlapping_and_dense_dim5() const;
SymBool compute_is_non_overlapping_and_dense_anydim() const;

void init_numel() const;
void init_is_contiguous() const;
void init_is_channels_last_contiguous() const;
void init_is_channels_last_3d_contiguous() const;
void init_is_channels_last() const;
void init_is_channels_last_3d() const;
void init_is_non_overlapping_and_dense() const;

// 注意：这些函数仅在 !has_foo() 的情况下设置值
void set_numel(SymInt val) const;
void set_is_contiguous(SymBool val) const;
void set_is_channels_last_contiguous(SymBool val) const;
void set_is_channels_last_3d_contiguous(SymBool val) const;
void set_is_channels_last(SymBool val) const;
void set_is_channels_last_3d(SymBool val) const;
void set_is_non_overlapping_and_dense(SymBool val) const;

// 惰性初始化变量，相应的 available_ 标志指示变量是否已初始化
mutable std::atomic<int> available_{0};
enum avail {
  numel_avail = 1 << 0,
  is_contiguous_avail = 1 << 1,
  is_channels_last_contiguous_avail = 1 << 2,
  is_channels_last_3d_contiguous_avail = 1 << 3,
  is_channels_last_avail = 1 << 4,
  is_channels_last_3d_avail = 1 << 5,
  is_non_overlapping_and_dense_avail = 1 << 6,
};

// 互斥锁，用于防止在常量访问器中初始化变量时出现竞态条件
mutable std::mutex mutables_;
mutable SymInt numel_ = 1;
mutable SymBool is_contiguous_{true};
mutable SymBool is_channels_last_contiguous_{false};
mutable SymBool is_channels_last_3d_contiguous_{false};
mutable SymBool is_channels_last_{false};
mutable SymBool is_channels_last_3d_{false};
mutable SymBool is_non_overlapping_and_dense_{true};
};

// 结束 c10 命名空间的定义
} // namespace c10
```