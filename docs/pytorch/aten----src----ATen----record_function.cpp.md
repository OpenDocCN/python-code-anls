# `.\pytorch\aten\src\ATen\record_function.cpp`

```
// 包含 ATen 库的记录功能头文件
#include <ATen/record_function.h>
// 包含 ATen 库的调度器头文件
#include <ATen/core/dispatch/Dispatcher.h>
// 包含 C10 宏定义的头文件
#include <c10/macros/Macros.h>
// 包含 C10 工具的线程局部存储头文件
#include <c10/util/ThreadLocal.h>
// 包含 C10 工具的多态函数封装头文件
#include <c10/util/overloaded.h>

// 包含标准库头文件
#include <algorithm>
#include <cstdlib>
#include <random>

// ATen 命名空间
namespace at {

// 全局变量：参数通信调用名称
extern const std::string kParamCommsCallName = "record_param_comms";

// 匿名命名空间，用于生成唯一的回调句柄
namespace {

// 生成下一个唯一的回调句柄
CallbackHandle next_unique_callback_handle() {
  static std::atomic<uint64_t> unique_cb_id {1};
  return CallbackHandle(unique_cb_id++);
}

// 生成下一个唯一的记录函数句柄
RecordFunctionHandle next_unique_record_function_handle() {
  static std::atomic<uint64_t> unique_rf_id {1};
  return RecordFunctionHandle(unique_rf_id++);
}

// 默认节点 ID，初始为 -1
std::atomic<int64_t> defaultNodeId(-1);

// 枚举线程 ID 的原子变量，逻辑上标识线程 ID
std::atomic<uint64_t> next_thread_id_ {0};
// 线程局部变量：当前线程的线程 ID，默认为 0
thread_local uint64_t current_thread_id_ = 0;

// 常量：记录范围的数量
static constexpr size_t NumRecordScopes =
    static_cast<size_t>(RecordScope::NUM_SCOPES);

// 在记录函数回调列表中查找指定回调句柄的回调函数迭代器
RecordFunctionCallbacks::iterator findCallback(
    RecordFunctionCallbacks& entries,
    CallbackHandle handle) {
  auto match_handle = [handle](const auto& el) { return el.handle_ == handle; };
  return std::find_if(entries.begin(), entries.end(), match_handle);
}

// 从记录函数回调列表中提取指定回调句柄的回调函数
std::optional<RecordFunctionCallback> extractCallback(
    RecordFunctionCallbacks& entries,
    CallbackHandle handle) {
  auto it = findCallback(entries, handle);
  if (it == entries.end()) {
    return c10::nullopt;
  }
  auto out = it->callback_;
  entries.erase(it);
  return out;
}

} // end anonymous namespace

// ============================================================================
// == 回调管理器 ===============================================================
// ============================================================================
// 记录函数回调机制的高级思想基于以下观察：
// 回调函数集合的变更频率较低。但为了重用活动集，必须在活动集发生变更时进行无效化。
// 可能导致回调应该运行的三个事件：
//  1) 全局回调集合变更
//  2) 本地回调集合变更
//  3) 存在采样回调，并应在此迭代中运行
//
// 全局回调依赖线程局部复制和原子版本计数器来保持一致性。每当更改活动全局回调集（添加/删除/启用/禁用）时，
// `GlobalCallbackManager` 增加版本号并在持有互斥锁的同时更新全局状态。本地回调管理器捕获全局回调快照，
// 通过比较 `GlobalCallbackManager::version()`（简单的原子读取）与上次重建的版本来懒惰地重建。在绝大多数情况下，
// 它们匹配时可以重用现有快照。否则，它必须调用更昂贵的（并加锁的）过程。
// `GlobalCallbackManager::getSnapshot()`.
//
// Handling changes to the thread local callbacks is trivial; functions that
// change them can simply force a cache rebuild for that thread after the
// changes are made.
//
// Sampling is by far the most challenging to handle efficiently. In general,
// sampling callbacks are expected to have very low frequency (e.g., 1 per
// million). Random number generation is rather expensive, so flipping a coin on
// every call for every sampling callback is wasteful. We can significantly
// reduce this cost by noting that the number of failures of a Bernoulli random
// variable is a geometric distribution, and thus we can sample the geometric
// distribution to determine the next time a callback should run. This reduces
// the cost from a random sample to a simple integer decrement.
//
// We can further note that Bernoulli samples are independent (in contrast to,
// say, sampling without replacement). This means that we can generate a
// counter for each scope that a given callback supports and then decrement the
// counter corresponding to the RecordScope being called. Conceptually, this is
// analogous to flipping different coins with the same probability. By sharding
// on RecordScope, we can consolidate the decrement to a single shared counter
// and update individual counters during rebuild.

class GlobalCallbackManager {
 public:
  static GlobalCallbackManager& get(); // Singleton

 private:
  GlobalCallbackManager() = default;

 public:
  static constexpr size_t NoVersion = 0;
  using snapshot_t = std::pair<size_t, RecordFunctionCallbacks>;

  // Returns the current version number of the manager.
  // Does not require locking as it only reads an atomic variable.
  size_t version() const; // No

  // Retrieves a snapshot of the current state of global callbacks
  // along with the version number at the time of snapshot.
  // Requires locking to ensure snapshot consistency.
  snapshot_t getSnapshot() const; // Yes

  // Adds a callback to the global manager and returns a handle to it.
  // Requires locking to modify the internal state atomically.
  CallbackHandle addCallback(RecordFunctionCallback cb); // Yes

  // Enables or disables a callback identified by the provided handle.
  // Requires locking to ensure consistent modification of callback states.
  void setCallbackEnabled(CallbackHandle handle, bool enabled); // Yes

  // Removes a callback identified by the provided handle.
  // Requires locking to safely remove the callback.
  void removeCallback(CallbackHandle handle); // Yes

  // Clears all callbacks from the manager.
  // Requires locking to ensure safe clearing of all callbacks.
  void clearCallbacks(); // Yes

 private:
  std::atomic<size_t> version_{NoVersion + 1}; // Atomic version counter.
  RecordFunctionCallbacks global_callbacks_; // Source of truth for callbacks.
  mutable std::mutex update_mutex_; // Mutex for thread safety.
};

class CacheEntry {
 public:
  CacheEntry() = default;
  CacheEntry(std::mt19937* generator, RecordScope scope);

  // Retrieves the active callbacks associated with this cache entry.
  // Caller is expected to check the version and update if necessary.
  StepCallbacks getActiveCallbacks();

  // Retrieves the active callbacks unless the list is empty.
  // Caller is expected to check the version and update if necessary.
  std::optional<StepCallbacks> getActiveCallbacksUnlessEmpty();

  // Performs a full update of callbacks, typically during registration.
  // Updates the internal state with a new set of callbacks.
  void update(const std::vector<RecordFunctionCallback>& callbacks);

 private:
  struct CallbackAndCounter {
    RecordFunctionCallback callback_;

    // `-1` indicates that a callback is not sampled.
    // More details about the callback state and sampling mechanism can be found here.
    // This structure is part of managing how callbacks are sampled and executed.
    // The choice of `-1` is likely a convention indicating no sampling is currently happening.
    // This is a private structure used internally to manage callback states.
    // It encapsulates each callback along with its sampling state.
    // 在类中声明一个私有变量 tries_left_，类型为 int
    int tries_left_{-1};
  };

  // 声明一个内联函数 getActiveCallbacksImpl，无返回值，无参数
  C10_ALWAYS_INLINE void getActiveCallbacksImpl();

  // 声明一个函数 rebuildActiveCallbacks，无返回值，无参数
  void rebuildActiveCallbacks();

  // 声明一个私有变量 generator_，类型为指向 std::mt19937 的指针，默认为 nullptr
  // std::mt19937 很大，因此所有作用域共享同一个生成器
  std::mt19937* generator_{nullptr};

  // 声明一个 c10::SmallVector 类型的 callbacks_，存储 CallbackAndCounter 对象，初始容量为 kSoftLimitCallbacks
  // callbacks_ 包含等待运行的采样回调函数
  c10::SmallVector<CallbackAndCounter, kSoftLimitCallbacks> callbacks_;

  // 声明一个 scope_ 变量，类型为 RecordScope::FUNCTION，默认初始化为 RecordScope 的 FUNCTION 枚举成员
  RecordScope scope_{RecordScope::FUNCTION};

  // 声明一个 active_callbacks_ 变量，类型为 StepCallbacks
  StepCallbacks active_callbacks_;

  // 声明一个 sampling_countdown_ 变量，类型为 int，默认初始化为 0
  int sampling_countdown_{0};

  // 声明一个 steps_for_this_update_ 变量，类型为 int，默认初始化为 0
  int steps_for_this_update_{0};
};

// LocalCallbackManager 类定义开始

class LocalCallbackManager {
 public:
  static LocalCallbackManager& get(); // 返回单例对象的引用

 private:
  LocalCallbackManager(); // 私有构造函数，用于单例模式

 public:
  const RecordFunctionTLS& getTLS() const; // 获取注册的 TLS（线程本地存储）记录函数回调对象的常量引用
  StepCallbacks getActiveCallbacks(const RecordScope scope); // 获取指定作用域下的活跃回调函数列表
  std::optional<StepCallbacks> getActiveCallbacksUnlessEmpty(const RecordScope scope); // 获取指定作用域下的活跃回调函数列表，如果列表为空则返回空的 optional

  void setTLS(const RecordFunctionTLS& tls); // 设置 TLS（线程本地存储）记录函数回调对象
  void seed(uint32_t seed); // 设置随机数发生器的种子值
  CallbackHandle addCallback(RecordFunctionCallback callback); // 添加记录函数回调并返回其句柄
  bool setCallbackEnabled(CallbackHandle handle, bool enabled); // 设置指定回调函数的启用状态
  bool removeCallback(CallbackHandle handle); // 移除指定的回调函数
  void clearCallbacks(); // 清空所有回调函数

 private:
  void rebuildActiveCallbacksIfNeeded(); // 如果需要，重建活跃的回调函数列表

  void rebuild_all(const GlobalCallbackManager::snapshot_t& global_snapshot); // 重建所有全局回调函数的快照

  void rebuild_callback_scopes(
      const GlobalCallbackManager::snapshot_t& global_snapshot,
      const RecordFunctionCallback& callback); // 根据全局回调函数的快照和指定的回调函数，重建回调函数的作用域

  void rebuild_scope(
      const GlobalCallbackManager::snapshot_t& global_snapshot,
      const RecordScope scope); // 根据全局回调函数的快照和指定的作用域，重建作用域

  // 注册的回调函数 TLS（线程本地存储）对象，为单一来源
  RecordFunctionTLS registered_callbacks_;

  // 运行时的缓存
  size_t global_version_{GlobalCallbackManager::NoVersion}; // 全局版本号，默认为无版本
  std::array<CacheEntry, NumRecordScopes> active_callbacks_; // 活跃回调函数的缓存数组
  std::mt19937 generator_{}; // 随机数生成器
};

// ============================================================================
// == GlobalCallbackManager: Implementation ===================================
// ============================================================================

// 获取全局回调管理器的单例对象
GlobalCallbackManager& GlobalCallbackManager::get() {
  static GlobalCallbackManager manager;
  return manager;
}

// 返回当前版本号
size_t GlobalCallbackManager::version() const {
  return version_.load(std::memory_order_relaxed);
}

// 获取全局回调函数的快照
std::pair<size_t, RecordFunctionCallbacks> GlobalCallbackManager::getSnapshot() const {
  std::lock_guard<std::mutex> guard(update_mutex_);
  return {version_.load(std::memory_order_seq_cst), global_callbacks_};
}

// 添加全局回调函数并返回其句柄
CallbackHandle GlobalCallbackManager::addCallback(RecordFunctionCallback cb) {
  std::lock_guard<std::mutex> guard(update_mutex_);
  ++version_;
  auto handle = next_unique_callback_handle();
  global_callbacks_.emplace_back(cb, handle);
  return handle;
}

// 设置回调函数的启用状态
void GlobalCallbackManager::setCallbackEnabled(
    CallbackHandle handle,
    bool enabled) {
  std::lock_guard<std::mutex> guard(update_mutex_);
  auto it = findCallback(global_callbacks_, handle);
  if (it != global_callbacks_.end()) {
    if (it->enabled_ != enabled) {
      ++version_;
      it->enabled_ = enabled;
    }
  } else {
    LOG(WARNING) << "Requested callback is not found";
  }
}

// 移除指定的回调函数
void GlobalCallbackManager::removeCallback(CallbackHandle handle) {
  std::lock_guard<std::mutex> guard(update_mutex_);
  if (extractCallback(global_callbacks_, handle).has_value()) {
    ++version_;
  } else {
    LOG(WARNING) << "Requested callback is not found";
  }
}

// 清空所有回调函数
void GlobalCallbackManager::clearCallbacks() {
  std::lock_guard<std::mutex> guard(update_mutex_);
  ++version_;
  global_callbacks_.clear();
}
// ============================================================================
// == CacheEntry: Implementation ==============================================
// ============================================================================

// 构造函数，初始化 CacheEntry 对象
CacheEntry::CacheEntry(std::mt19937* generator, RecordScope scope)
    : generator_{generator}, scope_{scope} {
  // 重建活动回调函数集合
  rebuildActiveCallbacks();
}

// 更新活动回调函数集合
void CacheEntry::update(const std::vector<RecordFunctionCallback>& callbacks) {
  // 清空当前回调函数列表
  callbacks_.clear();
  // 预留空间以容纳新的回调函数列表
  callbacks_.reserve(callbacks.size());
  // 遍历传入的回调函数列表
  for (const auto& callback : callbacks) {
    // 获取回调函数的采样概率
    const auto p = callback.samplingProb();
    // 根据采样概率决定是否进行采样，并将结果存入回调函数列表中
    callbacks_.push_back({callback, p < 1.0 ? sampleTries(p) : -1});
  }

  // 重新构建活动回调函数集合
  rebuildActiveCallbacks();
}

// 获取活动回调函数集合的实现函数
void CacheEntry::getActiveCallbacksImpl() {
  // 如果 sampling_countdown_ 在函数开始时为零，表示出现错误
  TORCH_INTERNAL_ASSERT(sampling_countdown_ > 0, sampling_countdown_);

  // 如果 sampling_countdown_ 减为零，则重新设置活动回调函数集合
  if (C10_UNLIKELY(!(--sampling_countdown_))) {
    // 根据步数更新已采样的回调函数
    for (auto& i : callbacks_) {
      if (i.tries_left_ > 0) {
        // 确保剩余尝试次数不小于当前更新的步数
        TORCH_INTERNAL_ASSERT(i.tries_left_ >= steps_for_this_update_);
        i.tries_left_ -= steps_for_this_update_;
      }
    }

    // 决定运行哪些回调函数以及运行的时长
    rebuildActiveCallbacks();

    // 对已运行的回调函数重新进行采样
    for (auto& i : callbacks_) {
      if (!i.tries_left_) {
        i.tries_left_ = sampleTries(i.callback_.samplingProb());
      }
    }
  }
}

// 获取活动回调函数集合
StepCallbacks CacheEntry::getActiveCallbacks() {
  getActiveCallbacksImpl();
  return active_callbacks_;
}

// 在活动回调函数集合不为空时获取活动回调函数集合的可选实现
std::optional<StepCallbacks> CacheEntry::getActiveCallbacksUnlessEmpty() {
  getActiveCallbacksImpl();
  if (C10_LIKELY(active_callbacks_.empty())) {
    return c10::nullopt;
  }
  return active_callbacks_;
}

// 重新构建活动回调函数集合
void CacheEntry::rebuildActiveCallbacks() {
  // 可以在 CacheEntry 中存储线程 ID，但重建频率低，因此避免传递线程 ID
  const auto thread_id = RecordFunction::currentThreadId();
  // 初始化活动回调函数集合
  active_callbacks_ = StepCallbacks(thread_id, scope_);

  // 设置采样倒计时为整数上限
  sampling_countdown_ = std::numeric_limits<int>::max();
  // 遍历回调函数列表，根据其尝试次数决定是否进行采样
  for (const auto& i : callbacks_) {
    if (i.tries_left_ < 0) {
      // 如果回调函数不被采样，无条件加入活动回调函数集合
      active_callbacks_.callbacks_.push_back(
          {i.callback_.start(), i.callback_.end()});
    } else if (i.tries_left_ == 0) {
      // 如果回调函数被采样且已到达采样事件，加入活动回调函数集合并设置采样倒计时为 1
      active_callbacks_.callbacks_.push_back(
          {i.callback_.start(), i.callback_.end()});
      sampling_countdown_ = 1;
    } else {
      // 如果回调函数被抽样且未到抽样事件，设置`sampling_countdown_`以在回调函数应执行时重新构建。
      sampling_countdown_ = std::min(sampling_countdown_, i.tries_left_);
    }
    // 检查当前回调函数是否需要输入，并将结果更新到`active_callbacks_`中
    active_callbacks_.needs_inputs_ |= i.callback_.needsInputs();
    // 检查当前回调函数是否需要输出，并将结果更新到`active_callbacks_`中
    active_callbacks_.needs_outputs_ |= i.callback_.needsOutputs();
    // 检查当前回调函数是否需要标识符，并将结果更新到`active_callbacks_`中
    active_callbacks_.needs_ids_ |= i.callback_.needsIds();
  }
  // 将`sampling_countdown_`的值赋给`steps_for_this_update_`
  steps_for_this_update_ = sampling_countdown_;
}

// CacheEntry 类的成员函数 sampleTries，用于返回样本尝试次数
int CacheEntry::sampleTries(double p) const {
  // 断言生成器非空
  TORCH_INTERNAL_ASSERT(generator_ != nullptr);
  // 断言概率 p 在 (0.0, 1.0] 范围内
  TORCH_INTERNAL_ASSERT(p > 0.0 && p <= 1.0);

  // 几何分布返回失败次数，加一表示成功调用的次数
  return std::geometric_distribution<int>(p)(*generator_) + 1;
}

// ============================================================================
// == LocalCallbackManager: Implementation ====================================
// ============================================================================

// 返回 LocalCallbackManager 的静态实例
LocalCallbackManager& LocalCallbackManager::get() {
#if defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
  static c10::ThreadLocal<LocalCallbackManager> manager;
  return manager.get();
#else // defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
  static thread_local LocalCallbackManager manager;
  return manager;
#endif // defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
}

// LocalCallbackManager 的构造函数
LocalCallbackManager::LocalCallbackManager() {
  // 初始化 active_callbacks_ 数组
  for (auto i : c10::irange(NumRecordScopes)) {
    active_callbacks_[i] = CacheEntry(&generator_, static_cast<RecordScope>(i));
  }
  // 重建所有全局回调的快照
  rebuild_all(GlobalCallbackManager::get().getSnapshot());
}

// 返回注册的回调函数 TLS
const RecordFunctionTLS& LocalCallbackManager::getTLS() const {
  return registered_callbacks_;
}

// 如果需要，重新构建活跃回调函数
void LocalCallbackManager::rebuildActiveCallbacksIfNeeded() {
  // 获取全局版本号
  const auto global_version = GlobalCallbackManager::get().version();
  // 如果全局版本号不匹配当前版本号，重新构建所有回调
  if (C10_UNLIKELY(global_version != global_version_)) {
    rebuild_all(GlobalCallbackManager::get().getSnapshot());
  }
}

// 获取指定作用域的活跃回调函数
StepCallbacks LocalCallbackManager::getActiveCallbacks(
    const RecordScope scope) {
  // 如果需要，重新构建活跃回调函数
  rebuildActiveCallbacksIfNeeded();
  // 返回指定作用域的活跃回调函数
  return active_callbacks_[static_cast<size_t>(scope)].getActiveCallbacks();
}

// 获取指定作用域的非空活跃回调函数的可选项
std::optional<StepCallbacks> LocalCallbackManager::getActiveCallbacksUnlessEmpty(
    const RecordScope scope) {
  // 如果需要，重新构建活跃回调函数
  rebuildActiveCallbacksIfNeeded();
  // 返回指定作用域的非空活跃回调函数的可选项
  return active_callbacks_[static_cast<size_t>(scope)].getActiveCallbacksUnlessEmpty();
}

// 设置注册的回调函数 TLS
void LocalCallbackManager::setTLS(const RecordFunctionTLS& tls) {
  registered_callbacks_ = tls;
  // 重建所有全局回调的快照
  rebuild_all(GlobalCallbackManager::get().getSnapshot());
}

// 设置生成器的种子值
void LocalCallbackManager::seed(uint32_t seed) {
  generator_.seed(seed);
}

// 添加回调函数到排序好的 TLS 回调函数列表中
CallbackHandle LocalCallbackManager::addCallback(
    RecordFunctionCallback callback) {
  auto handle = next_unique_callback_handle();
  auto& callbacks = registered_callbacks_.sorted_tls_callbacks_;
  callbacks.emplace_back(callback, handle);
  // 根据全局回调的快照重新构建回调作用域
  rebuild_callback_scopes(
      GlobalCallbackManager::get().getSnapshot(), callbacks.back().callback_);
  return handle;
}

// 设置回调函数是否启用
bool LocalCallbackManager::setCallbackEnabled(
    CallbackHandle handle,
    bool enabled) {
  auto it = findCallback(registered_callbacks_.sorted_tls_callbacks_, handle);
  auto found = (it != registered_callbacks_.sorted_tls_callbacks_.end());
  // 如果找到回调函数并且启用状态与参数不同，更新状态
  if (found && it->enabled_ != enabled) {
    it->enabled_ = enabled;
    //
    // 使用全局回调管理器获取其快照，并根据迭代器指向的回调函数重新建立回调作用域
    rebuild_callback_scopes(
        GlobalCallbackManager::get().getSnapshot(), it->callback_);
  }
  // 返回找到的标志，指示是否找到匹配项
  return found;
}

// 从注册的回调中移除指定的回调函数
bool LocalCallbackManager::removeCallback(CallbackHandle handle) {
  // 获取已注册的回调列表的引用
  auto& callbacks = registered_callbacks_.sorted_tls_callbacks_;
  // 提取具有给定句柄的回调函数
  auto callback = extractCallback(callbacks, handle);
  // 如果成功提取到回调函数
  if (callback.has_value()) {
    // 重新构建所有回调作用域
    rebuild_callback_scopes(
        GlobalCallbackManager::get().getSnapshot(), *callback);
  }
  // 返回是否成功提取到回调函数
  return callback.has_value();
}

// 清空所有本地回调函数
void LocalCallbackManager::clearCallbacks() {
  // 清空已注册的回调列表
  registered_callbacks_.sorted_tls_callbacks_.clear();
  // 重建所有回调
  rebuild_all(GlobalCallbackManager::get().getSnapshot());
}

// 重建所有作用域
void LocalCallbackManager::rebuild_all(const GlobalCallbackManager::snapshot_t& global_snapshot) {
  // 更新全局版本号
  global_version_ = global_snapshot.first;
  // 遍历所有记录作用域，重新构建
  for (auto i : c10::irange(NumRecordScopes)) {
    rebuild_scope(global_snapshot, static_cast<RecordScope>(i));
  }
}

// 重新构建回调函数的作用域
void LocalCallbackManager::rebuild_callback_scopes(
    const GlobalCallbackManager::snapshot_t& global_snapshot,
    const RecordFunctionCallback& callback) {
  // 如果全局快照的版本号与当前全局版本号相同
  if (global_snapshot.first == global_version_) {
    // 仅重新构建与给定回调相关的作用域
    for (auto i : c10::irange(NumRecordScopes)) {
      if (callback.checkScope(static_cast<RecordScope>(i))) {
        rebuild_scope(global_snapshot, static_cast<RecordScope>(i));
      }
    }
  } else {
    // 否则，重新构建所有作用域
    rebuild_all(global_snapshot);
  }
}

// 重新构建指定作用域的回调函数
void LocalCallbackManager::rebuild_scope(
    const GlobalCallbackManager::snapshot_t& global_snapshot,
    const RecordScope scope) {
  // 存储符合条件的回调函数
  std::vector<RecordFunctionCallback> callbacks;
  // 如果启用 TLS 记录函数
  if (registered_callbacks_.tls_record_function_enabled_) {
    // 填充回调函数列表的 lambda 函数
    auto populate_callbacks =
        [&](const RecordFunctionCallbacks& raw_callbacks) {
          for (const auto& i : raw_callbacks) {
            // 如果回调函数已启用且满足作用域条件和采样概率大于0
            if (i.enabled_ && i.callback_.checkScope(scope) &&
                i.callback_.samplingProb() > 0) {
              callbacks.push_back(i.callback_);
            }
          }
        };
    // 填充全局快照和已排序的 TLS 回调列表的回调函数
    populate_callbacks(global_snapshot.second);
    populate_callbacks(registered_callbacks_.sorted_tls_callbacks_);
  }
  // 更新活跃回调函数列表中指定作用域的回调函数
  active_callbacks_[static_cast<size_t>(scope)].update(callbacks);
}

// ============================================================================
// == Callback execution ======================================================
// ============================================================================

// 记录尝试运行回调函数时的异常信息
void logTryRunCallbackError(const char* what, const char* name) {
  // 记录异常信息到日志中
  LOG(WARNING) << "Exception in RecordFunction callback: " << what
               << " , for the range " << name;
}

// 尝试运行回调函数（模板函数）
template <bool is_start>
C10_ALWAYS_INLINE bool tryRunCallback(
    const StepCallbacks::StartEndPair callback_ptrs,
    const RecordFunction& rf,
    std::unique_ptr<ObserverContext>& ctx) {
  try {
    // 如果是开始回调且回调函数存在，则调用开始回调函数
    if (is_start && callback_ptrs.start_) {
      ctx = callback_ptrs.start_(rf);
    }

    // 如果不是开始回调且结束回调函数存在，则调用结束回调函数
    if (!is_start && callback_ptrs.end_) {
      callback_ptrs.end_(rf, ctx.get());
    }

    // 返回执行成功
    return true;
  } catch (const std::exception& e) {
    // 捕获并记录异常信息到日志中
    logTryRunCallbackError(e.what(), rf.name());
    // 返回执行失败
    return false;
  } catch (...) {
    // 捕获并记录未知类型异常信息到日志中
    // 调用函数 logTryRunCallbackError，记录未知错误，传入字符串 "unknown" 和 rf.name() 的返回值
    logTryRunCallbackError("unknown", rf.name());
    // 返回布尔值 false，表示函数执行失败
    return false;
} // namespace

} // namespace

// RecordFunction 类的构造函数，根据给定的 RecordScope 调用另一个构造函数以获取 StepCallbacks
RecordFunction::RecordFunction(RecordScope scope)
    : RecordFunction(getStepCallbacks(scope)) {}

// RecordFunction 类的构造函数，接受移动语义的 StepCallbacks 对象，并初始化成员变量
RecordFunction::RecordFunction(StepCallbacks&& step_callbacks)
    : step_callbacks_{std::move(step_callbacks)} {
  // 根据 callbacks 的大小调整 ctx_ 的大小，以容纳回调函数需要的上下文信息
  ctx_.resize(step_callbacks_.callbacks_.size());
  // 如果需要唯一的记录函数句柄，设置句柄
  if (step_callbacks_.needs_ids_) {
    setHandle(next_unique_record_function_handle());
  }
}

// 执行所有起始回调函数
void RecordFunction::runStartCallbacks() {
  // 遍历所有注册的回调函数，尝试执行起始回调函数
  for (const auto i : c10::irange(step_callbacks_.callbacks_.size())) {
    tryRunCallback</*is_start=*/true>(
        step_callbacks_.callbacks_[i], *this, ctx_[i]);
  }
  // 标记已经调用了起始回调函数
  called_start_callbacks_ = true;
}

// 结束函数记录
void RecordFunction::end() {
  // 如果已经调用了起始回调函数
  if (called_start_callbacks_) {
    // 遍历所有注册的回调函数，尝试执行结束回调函数
    for (const auto i : c10::irange(step_callbacks_.callbacks_.size())) {
      tryRunCallback</*is_start=*/false>(
        step_callbacks_.callbacks_[i], *this, ctx_[i]);
    }
    // 清空回调函数列表
    step_callbacks_.callbacks_.clear();
  }
}

// 返回当前函数名称的 C 字符串表示
const char* RecordFunction::name() const {
  return std::visit(
      // 使用 std::visit 访问 fn_ 中的变体类型
      c10::overloaded(
          [](const std::string& name) { return name.c_str(); },
          [](const schema_ref_t schema) {
            return schema.get().name().c_str();
          }),
      fn_);
}

// 返回输入参数的数量
size_t RecordFunction::num_inputs() const {
  return std::visit(
      // 使用 std::visit 访问 fn_ 中的变体类型
      c10::overloaded(
          [&](const std::string&) { return inputs_.size(); },
          [](const schema_ref_t schema) {
            return schema.get().arguments().size();
          }),
      fn_);
}

// 返回输出参数的数量
size_t RecordFunction::num_outputs() const {
  return std::visit(
      // 使用 std::visit 访问 fn_ 中的变体类型
      c10::overloaded(
          [&](const std::string&) { return outputs_.size(); },
          [](const schema_ref_t schema) {
            return schema.get().returns().size();
          }),
      fn_);
}

// 返回操作符名称的可选对象
std::optional<OperatorName> RecordFunction::operator_name() const {
  return std::visit(
      // 使用 std::visit 访问 fn_ 中的变体类型
      c10::overloaded(
          [&](const std::string&) -> std::optional<OperatorName> {
            return c10::nullopt;
          },
          [](const schema_ref_t schema) -> std::optional<OperatorName> {
            return schema.get().operator_name();
          }),
      fn_);
}

// 返回操作符模式的可选对象
std::optional<c10::FunctionSchema> RecordFunction::operator_schema() const {
  return std::visit(
      // 使用 std::visit 访问 fn_ 中的变体类型
      c10::overloaded(
          [&](const std::string&) -> std::optional<c10::FunctionSchema> {
            return c10::nullopt;
          },
          [](const schema_ref_t schema) -> std::optional<c10::FunctionSchema> {
            return schema.get();
          }),
      fn_);
}

// 根据 RecordScope 获取活跃的步骤回调函数
StepCallbacks getStepCallbacks(RecordScope scope) {
  return LocalCallbackManager::get().getActiveCallbacks(scope);
}

// 根据 RecordScope 获取非空的步骤回调函数，返回可选对象
std::optional<StepCallbacks> getStepCallbacksUnlessEmpty(RecordScope scope) {
  return LocalCallbackManager::get().getActiveCallbacksUnlessEmpty(scope);
}

// 获取记录函数的线程局部存储对象
const RecordFunctionTLS& get_record_function_tls_() {
  return LocalCallbackManager::get().getTLS();
}

// 设置记录函数的线程局部存储对象
void set_record_function_tls_(const RecordFunctionTLS& tls) {
  LocalCallbackManager::get().setTLS(tls);
}

// 匿名命名空间的结束
namespace {
bool anyEnabled(const RecordFunctionCallbacks& callbacks) {
  // 检查回调函数列表中是否有任何一个回调函数处于启用状态
  return std::any_of(callbacks.begin(), callbacks.end(), [](const auto& cb) {
    return cb.enabled_;
  });
}
} // namespace

bool hasCallbacks() {
  // 检查是否存在任何类型的回调函数（全局或线程本地）
  return hasThreadLocalCallbacks() || hasGlobalCallbacks();
}

bool hasGlobalCallbacks() {
  // 检查全局回调管理器中是否有任何启用的回调函数
  return anyEnabled(GlobalCallbackManager::get().getSnapshot().second);
}

bool hasThreadLocalCallbacks() {
  // 检查线程本地回调管理器中是否有任何启用的回调函数
  return anyEnabled(get_record_function_tls_().sorted_tls_callbacks_);
}

CallbackHandle addThreadLocalCallback(
    RecordFunctionCallback cb) {
  // 向线程本地回调管理器中添加回调函数，并返回其句柄
  return LocalCallbackManager::get().addCallback(cb);
}

CallbackHandle addGlobalCallback(
    RecordFunctionCallback cb) {
  // 向全局回调管理器中添加回调函数，并返回其句柄
  return GlobalCallbackManager::get().addCallback(cb);
}

void removeCallback(CallbackHandle handle) {
  // 尝试从线程本地回调管理器中移除指定句柄的回调函数，
  // 如果失败，则从全局回调管理器中移除
  if (!LocalCallbackManager::get().removeCallback(handle)) {
    GlobalCallbackManager::get().removeCallback(handle);
  }
}

void disableCallback(CallbackHandle handle) {
  // 尝试禁用线程本地回调管理器中指定句柄的回调函数，
  // 如果失败，则禁用全局回调管理器中的对应回调函数
  if (!LocalCallbackManager::get().setCallbackEnabled(handle, false)) {
    GlobalCallbackManager::get().setCallbackEnabled(handle, false);
  }
}

void reenableCallback(CallbackHandle handle) {
  // 尝试重新启用线程本地回调管理器中指定句柄的回调函数，
  // 如果失败，则重新启用全局回调管理器中的对应回调函数
  if (!LocalCallbackManager::get().setCallbackEnabled(handle, true)) {
    GlobalCallbackManager::get().setCallbackEnabled(handle, true);
  }
}

void clearGlobalCallbacks() {
  // 清空全局回调管理器中的所有回调函数
  GlobalCallbackManager::get().clearCallbacks();
}

void clearThreadLocalCallbacks() {
  // 清空线程本地回调管理器中的所有回调函数
  LocalCallbackManager::get().clearCallbacks();
}

void clearCallbacks() {
  // 清空所有回调函数，包括全局和线程本地的
  clearGlobalCallbacks();
  clearThreadLocalCallbacks();
}

bool isRecordFunctionEnabled() {
  // 检查记录函数是否在线程本地被启用
  return LocalCallbackManager::get().getTLS().tls_record_function_enabled_;
}

void enableRecordFunction(bool enable) {
  // 启用或禁用线程本地记录函数
  auto tls = LocalCallbackManager::get().getTLS();
  if (tls.tls_record_function_enabled_ != enable) {
    tls.tls_record_function_enabled_ = enable;
    LocalCallbackManager::get().setTLS(tls);
  }
}

void set_record_function_seed_for_testing(uint32_t seed) {
  // 为测试目的设置线程本地记录函数的种子值
  LocalCallbackManager::get().seed(seed);
}

/* static */
uint64_t RecordFunction::currentThreadId() {
  if (!current_thread_id_) {
    // 每个线程只执行一次，为当前线程分配唯一的线程 ID
    current_thread_id_ = ++next_thread_id_;
  }
  return current_thread_id_;
}

void RecordFunction::before(const char* name, int64_t sequence_nr) {
  // 在函数执行前设置函数名和序列号，并根据特定条件设置是否为NCCL元信息函数
  fn_ = name;
  sequence_nr_ = sequence_nr;
  is_nccl_meta_ = (std::strcmp(name, kParamCommsCallName.c_str()) == 0);

#ifndef NDEBUG
    inputs_valid_ = true;
#endif
  // 执行函数开始时的回调函数
  runStartCallbacks();
  // 使输入无效化，用于确保重新计算
  invalidateInputs();
}

void RecordFunction::before(std::string name, int64_t sequence_nr) {
  // 在函数执行前设置函数名和序列号，并根据特定条件设置是否为NCCL元信息函数
  is_nccl_meta_ = (name == kParamCommsCallName);
  fn_ = std::move(name);
  sequence_nr_ = sequence_nr;

#ifndef NDEBUG
    inputs_valid_ = true;
#endif
  // 执行函数开始时的回调函数
  runStartCallbacks();
  // 使输入无效化，用于确保重新计算
  invalidateInputs();
}

void RecordFunction::before(
    RecordFunction::schema_ref_t schema,
    int64_t sequence_nr) {
  // 在函数执行前设置函数模式和序列号，并根据特定条件设置是否为NCCL元信息函数
  sequence_nr_ = sequence_nr;
  fn_ = schema;
  is_nccl_meta_ = (schema.get().name() == kParamCommsCallName);

#ifndef NDEBUG
    # 设置一个变量 `inputs_valid_` 的布尔值为真
    inputs_valid_ = true;
#endif
  // 运行结束回调函数
  runStartCallbacks();
  // 使输入无效
  invalidateInputs();
}

/* static */ void RecordFunction::setDefaultNodeId(int64_t newDefaultNodeId) {
  // 检查新的默认节点 ID 是否大于等于 0
  TORCH_CHECK(newDefaultNodeId >= 0, "setDefaultNodeId expects an id >= 0.");
  // 设置默认节点 ID
  defaultNodeId = newDefaultNodeId;
}

/* static */ int64_t RecordFunction::getDefaultNodeId() {
  // 返回默认节点 ID
  return defaultNodeId;
}

RecordFunction::~RecordFunction() {
  // 结束记录函数
  end();
}

void RecordFunction::_setAsync() {
  // 设置函数为异步模式
  is_async_ = true;
}

bool RecordFunction::isAsync() const {
  // 返回函数是否为异步模式
  return is_async_;
}

void RecordFunction::_setStaticRuntimeOutVariant() {
  // 如果记录函数处于活动状态，则设置为静态运行时输出变体
  if (isActive()) {
    is_static_runtime_out_variant_ = true;
  }
}

bool RecordFunction::isStaticRuntimeOutVariant() const {
  // 如果记录函数处于活动状态，则返回是否为静态运行时输出变体
  if (isActive()) {
    return is_static_runtime_out_variant_;
  }
  // 否则返回 false
  return false;
}
} // namespace at
```