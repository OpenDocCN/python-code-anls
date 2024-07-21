# `.\pytorch\c10\core\impl\LocalDispatchKeySet.cpp`

```py
// 线程局部变量，用于保存本地分发键集的 POD 结构
// 注意 [TLS 初始化]
// 我们希望 raw_local_dispatch_key_set 被非零状态初始化，例如 BackendSelect 和 ADInplaceOrView 在包含的集合中。
// 但是某些 Windows 编译器（例如 ARVR 测试中使用的编译器）只允许 TLS 被零初始化。
// 为了保持默认状态的 raw TLS 存储为零的不变性，我们通过将 raw_local_dispatch_key_set.included_ 与 c10::default_included_set 进行异或来获取实际的包含键集。
// 这个逻辑封装在 PODLocalDispatchKeySet 结构体中。
thread_local PODLocalDispatchKeySet raw_local_dispatch_key_set;

// 如果定义了 _MSC_VER、C10_ANDROID 或 C10_IPHONE，则定义 tls_local_dispatch_key_set 函数
// 返回 raw_local_dispatch_key_set 的本地分发键集
#if defined(_MSC_VER) || defined(C10_ANDROID) || defined(C10_IPHONE)
LocalDispatchKeySet tls_local_dispatch_key_set() {
  return raw_local_dispatch_key_set;
}
#endif // defined(_MSC_VER) || defined(C10_ANDROID) || defined(C10_IPHONE)

// 强制设置 raw_local_dispatch_key_set 的本地分发键集
void _force_tls_local_dispatch_key_set(LocalDispatchKeySet key_set) {
  raw_local_dispatch_key_set.set_included(key_set.included_);
  raw_local_dispatch_key_set.set_excluded(key_set.excluded_);
}

// RAII 守卫类的构造函数，用于设置包含的分发键集
// 如果 include 集合中存在尚未包含的键，将它们添加到 raw_local_dispatch_key_set 的包含键集中
IncludeDispatchKeyGuard::IncludeDispatchKeyGuard(DispatchKeySet include)
    : tls_(&raw_local_dispatch_key_set), include_(include - tls_->included()) {
  if (!include_.empty()) {
    tls_->set_included(tls_->included() | include_);
  }
}

// RAII 守卫类的析构函数，用于恢复之前设置的包含键集
// 从 raw_local_dispatch_key_set 的包含键集中移除之前添加的键
IncludeDispatchKeyGuard::~IncludeDispatchKeyGuard() {
  if (!include_.empty()) {
    tls_->set_included(tls_->included() - include_);
  }
}

// RAII 守卫类的构造函数，用于设置排除的分发键集
// 如果 exclude 集合中存在尚未排除的键，将它们添加到 raw_local_dispatch_key_set 的排除键集中
ExcludeDispatchKeyGuard::ExcludeDispatchKeyGuard(DispatchKeySet exclude)
    : tls_(&raw_local_dispatch_key_set), exclude_(exclude - tls_->excluded()) {
  if (!exclude_.empty()) {
    tls_->set_excluded(tls_->excluded() | exclude_);
  }
}

// RAII 守卫类的析构函数，用于恢复之前设置的排除键集
// 从 raw_local_dispatch_key_set 的排除键集中移除之前添加的键
ExcludeDispatchKeyGuard::~ExcludeDispatchKeyGuard() {
  if (!exclude_.empty()) {
    tls_->set_excluded(tls_->excluded() - exclude_);
  }
}

// 非 RAII API
// 请优先使用 RAII API。有关详细信息，请参阅 LocalDispatchKeySet.h 中的声明。
# 检查给定的调度键是否在本地调度键集的排除集合中
bool tls_is_dispatch_key_excluded(DispatchKey x) {
  return raw_local_dispatch_key_set.excluded().has(x);
}

# 设置给定调度键是否在本地调度键集的排除集合中，并指定期望的状态
void tls_set_dispatch_key_excluded(DispatchKey x, bool desired_state) {
  auto* tls = &raw_local_dispatch_key_set;
  # 获取当前调度键在排除集合中的状态
  bool current_state = tls->excluded().has(x);
  # 如果期望状态与当前状态不同
  if (desired_state != current_state) {
    # 如果期望状态为真，则将调度键添加到排除集合中
    if (desired_state) {
      tls->set_excluded(tls->excluded().add(x));
    } else {  # 如果期望状态为假，则从排除集合中移除调度键
      tls->set_excluded(tls->excluded().remove(x));
    }
  }
}

# 检查给定的调度键是否在本地调度键集的包含集合中
bool tls_is_dispatch_key_included(DispatchKey x) {
  return raw_local_dispatch_key_set.included().has(x);
}

# 设置给定调度键是否在本地调度键集的包含集合中，并指定期望的状态
void tls_set_dispatch_key_included(DispatchKey x, bool desired_state) {
  auto* tls = &raw_local_dispatch_key_set;
  # 获取当前调度键在包含集合中的状态
  bool current_state = tls->included().has(x);
  # 如果期望状态与当前状态不同
  if (desired_state != current_state) {
    # 如果期望状态为真，则将调度键添加到包含集合中
    if (desired_state) {
      tls->set_included(tls->included().add(x));
    } else {  # 如果期望状态为假，则从包含集合中移除调度键
      tls->set_included(tls->included().remove(x));
    }
  }
}

# 检查本地调度键集的排除集合是否是给定调度键集的超集
bool tls_is_dispatch_keyset_excluded(DispatchKeySet ks) {
  return raw_local_dispatch_key_set.excluded().isSupersetOf(ks);
}

# 检查本地调度键集的包含集合是否是给定调度键集的超集
bool tls_is_dispatch_keyset_included(DispatchKeySet ks) {
  return raw_local_dispatch_key_set.included().isSupersetOf(ks);
}
```