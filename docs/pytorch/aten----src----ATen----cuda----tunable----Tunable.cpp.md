# `.\pytorch\aten\src\ATen\cuda\tunable\Tunable.cpp`

```
// Original TunableOp is from onnxruntime.
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/framework/tunable.h
// https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/rocm/tunable
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Adapting TunableOp into PyTorch
// Copyright (c) Advanced Micro Devices, Inc.

#include <cuda_runtime.h>  // CUDA runtime APIs

#include <ATen/cuda/CUDAContextLight.h>  // Lightweight CUDA context management
#include <ATen/cuda/tunable/Tunable.h>   // Tunable interfaces for CUDA
#include <c10/util/Exception.h>          // C10 exception utilities
#include <c10/util/StringUtil.h>         // C10 string utilities
#include <torch/version.h>               // PyTorch version macros

#ifndef _WIN32
#include <cxxabi.h>  // Demangling C++ symbols
#endif

#include <chrono>           // C++ time utilities
#include <fstream>          // File stream operations
#include <functional>       // C++ function utilities
#include <limits>           // Numeric limits for types
#include <memory>           // Memory management utilities
#include <mutex>            // Mutex primitives
#include <sstream>          // String stream operations
#include <string>           // String utilities
#include <thread>           // C++ thread support
#include <type_traits>      // Type traits
#include <unordered_map>    // Unordered map containers
#include <unordered_set>    // Unordered set containers
#include <utility>          // Utility components
#include <vector>           // Vector containers

namespace at::cuda::tunable {

namespace {

TuningContext tuning_context;  // Tuning context singleton

} // anonymous namespace

// Return a pointer to the global TuningContext instance
TuningContext* getTuningContext() {
  return &tuning_context;
}

// Define the output stream operator for ResultEntry objects
std::ostream& operator<<(std::ostream& stream, const ResultEntry& entry) {
  return stream << entry.key_ << "," << entry.time_;
}

// TuningResultsManager methods implementation

// Lookup the kernel map for a given operation signature
KernelMap TuningResultsManager::Lookup(const std::string& op_signature) {
  std::scoped_lock l{lock_};  // Scoped lock to synchronize access
  auto it = results_.find(op_signature);  // Find the operation signature in results map
  if (it == results_.cend()) {
    return {};  // Return an empty KernelMap if not found
  }
  return it->second;  // Return a copy of the KernelMap associated with the operation signature
}

// Lookup the best ResultEntry for a specific operation and parameter signature
ResultEntry TuningResultsManager::Lookup(const std::string& op_signature, const std::string& params_signature) {
  std::scoped_lock l{lock_};  // Scoped lock for thread safety
  auto kernel_map_it = results_.find(op_signature);  // Find the operation signature in results map
  if (kernel_map_it == results_.cend()) {
    TUNABLE_LOG3("missing op_signature, returning null ResultEntry");  // Log if operation signature is missing
    return ResultEntry::Null();  // Return a null ResultEntry if op_signature is not found
  }

  const auto& km = kernel_map_it->second;  // Get the KernelMap associated with op_signature
  auto it = km.find(params_signature);     // Find the parameter signature in the KernelMap
  if (it == km.cend()) {
    TUNABLE_LOG3("missing params_signature, returning null ResultEntry");  // Log if params_signature is missing
    return ResultEntry::Null();  // Return a null ResultEntry if params_signature is not found
  }
  return it->second;  // Return the ResultEntry associated with params_signature
}

// Add a new best ResultEntry for a specific operation and parameter signature
void TuningResultsManager::Add(const std::string& op_signature, const std::string& params_signature, ResultEntry best) {
  std::scoped_lock l{lock_};  // Scoped lock for thread safety

  auto it = results_.find(op_signature);  // Find the operation signature in results map
  if (it == results_.end()) {
    // If operation signature not found, create a new entry
    // This is the first time tuning results are added for this operation
    # 在 results_ 中插入一个新的元素，其键为 op_signature，值为一个空的字典
    it = results_.insert({op_signature, {}}).first;
  }

  # 调用 AddImpl 函数，将 op_signature、params_signature、best 和 it->second 作为参数传入
  AddImpl(op_signature, params_signature, best, it->second);
}

// 删除给定操作签名和参数签名的结果条目
void TuningResultsManager::Delete(const std::string& op_signature, const std::string& params_signature) {
  // 使用互斥锁锁定共享资源
  std::scoped_lock l{lock_};

  // 查找操作签名在结果集中的位置
  auto it = results_.find(op_signature);
  // 如果操作签名不存在于结果集中，则直接返回
  if (it == results_.end()) {
    return;
  }

  // 在操作签名的结果中查找参数签名的位置
  auto it2 = it->second.find(params_signature);
  // 如果参数签名不存在于操作签名的结果中，则直接返回
  if (it2 == it->second.end()) {
    return;
  }

  // 记录删除操作的日志
  TUNABLE_LOG2(op_signature, "(", params_signature, ")");
  // 删除参数签名的结果条目
  it->second.erase(it2);
}

// 执行不相交合并，将给定操作签名的内核映射与结果集进行合并
inline void TuningResultsManager::DisjointMergeImpl(
    const std::string& op_signature,
    const KernelMap& kernel_map,
    /*out*/ std::unordered_map<std::string, KernelMap>& results) {
  // 查找操作签名在结果集中的位置
  auto it = results.find(op_signature);
  // 如果操作签名不存在于结果集中，则执行添加操作并返回
  if (it == results.end()) {
    for (const auto& [param_sig, kernel_id] : kernel_map) {
      // 记录内核映射的日志
      TUNABLE_LOG2(op_signature, "(", param_sig, ") -> ", kernel_id);
    }
    // 将操作签名和其对应的内核映射添加到结果集中
    results[op_signature] = kernel_map;
    return;
  }

  // 对于每个参数签名和其对应的最佳内核，执行添加操作
  for (const auto& [params_signature, best] : kernel_map) {
    AddImpl(op_signature, params_signature, best, it->second);
  }
}

// 加载给定的结果集到当前结果管理器中
void TuningResultsManager::Load(const std::unordered_map<std::string, KernelMap>& results_to_load) {
  // 记录加载操作的日志
  TUNABLE_LOG1("Loading results");
  // 使用互斥锁锁定共享资源
  std::scoped_lock l{lock_};
  // 遍历每个操作签名及其对应的内核映射，执行不相交合并操作
  for (const auto& [op_signature, kernel_map] : results_to_load) {
    DisjointMergeImpl(op_signature, kernel_map, results_);
  }
}

// 导出当前结果集
ResultsMap TuningResultsManager::Dump() {
  // 使用互斥锁锁定共享资源
  std::scoped_lock l{lock_};
  // 返回当前结果集
  return results_;
}

// 执行不相交合并，将给定操作签名的内核映射与结果集进行合并
void TuningResultsManager::DisjointMerge(const std::string& op_signature, const KernelMap& kernel_map) {
  // 使用互斥锁锁定共享资源
  std::scoped_lock l{lock_};
  // 执行不相交合并操作
  DisjointMergeImpl(op_signature, kernel_map, results_);
}

// 获取当前结果集中条目的总数
size_t TuningResultsManager::GetSize() {
  size_t size = 0;
  // 使用互斥锁锁定共享资源
  std::scoped_lock l{lock_};
  // 遍历每个操作签名及其对应的内核映射，累加内核映射的大小
  for (const auto& [op_signature, kernel_map] : results_) {
    size += kernel_map.size();
  }
  // 返回结果集中条目的总数
  return size;
}

// TuningResultsValidator

// 初始化 TuningResultsValidator 对象
TuningResultsValidator::TuningResultsValidator() {
  // 注册验证器，关联 "PT_VERSION" 键和相应的获取和验证函数
  RegisterValidator(
      "PT_VERSION",
      [this]() { return GetPyTorchVersion(); },
      [this](auto&& k) { return ValidatePyTorchVersion(std::forward<decltype(k)>(k)); });
}

// 获取所有注册的验证器及其结果
std::unordered_map<std::string, std::string> TuningResultsValidator::GetAllValidators() const {
  std::unordered_map<std::string, std::string> ret;
  // 遍历每个验证器，获取验证结果并存储在返回的映射中
  for (const auto& [key, get_validate_func_pair] : validators_) {
    const GetFunc& getter = get_validate_func_pair.first;
    ret[key] = getter();
  }
  // 返回所有验证器的结果映射
  return ret;
}

// 检查必需键是否在验证函数和待检查映射中都存在
static bool CheckMandatoryKeys(
    const TuningResultsValidator::GetValidateFuncs& gv_funcs,
    const std::unordered_map<std::string, std::string>& to_check) {
  bool passed = true;
  // 遍历每个必需键
  for (const auto& k : TuningResultsValidator::mandatory_keys) {
    // 检查验证函数中是否包含该键
    if (gv_funcs.find(k) == gv_funcs.end()) {
      passed = false;
      // 记录日志，指出验证函数未注册该键
      TUNABLE_LOG1("key=\"", k, "\" is not registered for Get and Validate. ");
    }

    // 检查待检查映射中是否包含该键
    if (to_check.find(k) == to_check.end()) {
      passed = false;
      // 记录日志，指出待检查映射未提供该键
      TUNABLE_LOG1("key=\"", k, "\" is not provided for validation. ");
    }
  }
  // 返回是否所有必需键都存在于验证函数和待检查映射中
  return passed;
}

// 检查两个映射中的键是否匹配
    // 定义 Lambda 函数 get_keys，用于从 pair 类型中获取键值
    auto get_keys = [](const auto& it) -> std::string { return it.first; };
    // 创建空的字符串向量 required_keys 和 provided_keys
    std::vector<std::string> required_keys;
    std::vector<std::string> provided_keys;
    // 从 gv_funcs（一个映射函数的容器）中提取所有键，并存储到 required_keys 中
    std::transform(gv_funcs.cbegin(), gv_funcs.cend(), std::back_inserter(required_keys), get_keys);
    // 从 to_check（一个字符串到字符串的映射）中提取所有键，并存储到 provided_keys 中
    std::transform(to_check.cbegin(), to_check.cend(), std::back_inserter(provided_keys), get_keys);
    // 对 required_keys 和 provided_keys 进行升序排序
    std::sort(required_keys.begin(), required_keys.end());
    std::sort(provided_keys.begin(), provided_keys.end());

    // 创建一个无序集合 intersection，用于存储 required_keys 和 provided_keys 的交集
    std::unordered_set<std::string> intersection;
    // 找到 required_keys 和 provided_keys 的交集，并存储到 intersection 中
    std::set_intersection(required_keys.cbegin(), required_keys.cend(),
                          provided_keys.cbegin(), provided_keys.cend(),
                          std::inserter(intersection, intersection.end()));
    // 初始化 matched 为 true，用于跟踪匹配结果
    bool matched = true;
    // 如果交集大小不等于 required_keys 的大小，说明有未匹配项
    if (intersection.size() != required_keys.size()) {
        matched = false;
        // 遍历 required_keys，找到未在 intersection 中的键，并发出警告信息
        for (const auto& k : required_keys) {
            if (intersection.find(k) == intersection.end()) {
                TORCH_WARN("Unmatched validator: \"", k, "\" is required, but the tuning results does not provide it. ");
            }
        }
    }
    // 如果交集大小不等于 provided_keys 的大小，说明有未消费项
    if (intersection.size() != provided_keys.size()) {
        matched = false;
        // 遍历 provided_keys，找到未在 intersection 中的键，并发出警告信息
        for (const auto& k : provided_keys) {
            if (intersection.find(k) == intersection.end()) {
                TORCH_WARN("Unmatched validator: \"", k, "\" is provided, but pytorch is unable to consume it. ");
            }
        }
    }
    // 返回匹配结果
    return matched;
}

// TuningResultsValidator 类的 ValidateAll 方法实现
TuningStatus TuningResultsValidator::ValidateAll(
        const std::unordered_map<std::string, std::string>& to_validate) const {
  // 检查必需的键是否存在
  if (!CheckMandatoryKeys(validators_, to_validate)) {
    return FAIL;
  }
  // 检查键是否匹配
  if (!CheckKeysMatching(validators_, to_validate)) {
    return FAIL;
  }

  // 遍历要验证的键值对
  for (const auto& [key, value] : to_validate) {
    // 查找与当前键对应的验证函数
    const auto& it = validators_.find(key);
    if (it == validators_.cend()) {
      // 如果找不到对应的验证函数，则记录警告并返回失败
      TORCH_WARN("Failed to lookup validator using key ", key);
      // 输出可用的键列表
      for (const auto& [key2, val2] : validators_) {
        TORCH_WARN("available key ", key2);
      }
      return FAIL;
    }
    // 获取验证函数
    const ValidateFunc& validator = it->second.second;
    // 调用验证函数，检查值是否有效
    if (validator(value) != OK) {
      // 如果验证失败，则记录警告并返回失败
      TORCH_WARN("Failed validator: ", key);
      return FAIL;
    }
  }

  // 所有验证通过，返回成功
  return OK;
}

// TuningResultsValidator 类的 RegisterValidator 方法实现
void TuningResultsValidator::RegisterValidator(const std::string& key, const GetFunc& gf, const ValidateFunc& vf) {
  // 检查是否已经注册过相同的键
  if (validators_.find(key) != validators_.end()) {
    // 如果已注册，则记录警告信息
    TORCH_WARN("Attempting to re-register validator with key ", key);
  }
  else {
    // 否则注册新的验证函数
    validators_[key] = std::make_pair(gf, vf);
  }
}

// TuningResultsValidator 类的 GetPyTorchVersion 方法实现
std::string TuningResultsValidator::GetPyTorchVersion() const {
  // 返回当前 PyTorch 版本信息
  return TORCH_VERSION;
}

// TuningResultsValidator 类的 ValidatePyTorchVersion 方法实现
TuningStatus TuningResultsValidator::ValidatePyTorchVersion(const std::string& value) const {
  // 检查给定的值是否与当前 PyTorch 版本匹配
  if (value == GetPyTorchVersion()) {
    return OK;
  }
  // 不匹配则返回失败
  return FAIL;
}

// TuningContext 类的构造函数实现
TuningContext::TuningContext() :
    enable_{false},
    tuning_enable_{true},
    manager_initialized_{false},
    write_file_on_exit_{true},
    numerics_check_enable_{false},
    max_tuning_duration_ms_{30},
    max_tuning_iterations_{100},
    max_warmup_duration_ms_{0},
    max_warmup_iterations_{0},
    icache_flush_{true},
    rotating_buffer_size_{-1},
    filename_{},
    results_count_from_input_file_{0}
{
}

// TuningContext 类的析构函数实现
TuningContext::~TuningContext() {
  // 如果 TuningResultsManager 从未初始化过，则无需操作
  if (!manager_initialized_) {
    // 在 DDP 作业中，可能是由 Python 进程生成其他工作进程，但本身不执行任何计算
    return;
  }
  // 获取当前的文件名
  auto filename = GetFilename();
  // 如果启用了可调优操作，并且允许调优，并且文件名不为空，并且在退出时需要写文件
  if (IsTunableOpEnabled() && IsTuningEnabled() && !filename.empty() && write_file_on_exit_) {
    // 如果输入文件中的结果数量少于 TuningResultsManager 中的数量
    if (results_count_from_input_file_ < GetTuningResultsManager().GetSize()) {
      // 如果结果数量大于 0，则表示有额外的调优结果需要重新写入文件
      if (results_count_from_input_file_ > 0) {
        TUNABLE_LOG1("additional tuning results available, rewriting file ", filename);
      }
      else {
        TUNABLE_LOG1("writing file ", filename);
      }
      // 尝试写入文件
      if (!WriteFile(filename)) {
        TUNABLE_LOG1("failed to write file ", filename);
      }
    }
  }
}

// TuningContext 类的 EnableTunableOp 方法实现
void TuningContext::EnableTunableOp(bool value) {
  // 设置是否启用可调优操作
  enable_ = value;
  // 记录启用或禁用信息到日志
  if (value) {
    TUNABLE_LOG1("Enable TunableOp");
  }
  else {
    TUNABLE_LOG1("Disable TunableOp");
  }
}

// TuningContext 类的 IsTunableOpEnabled 方法实现
bool TuningContext::IsTunableOpEnabled() const {
  // 检查环境变量 PYTORCH_TUNABLEOP_ENABLED 是否为 "1"，来确定是否启用可调优操作
  static const char *env = std::getenv("PYTORCH_TUNABLEOP_ENABLED");
  if (env != nullptr && strcmp(env, "1") == 0) {
    return true;
  }
  # 返回 true 值，结束当前函数并返回该值
  return enable_;
}

// 设置是否启用调优功能
void TuningContext::EnableTuning(bool value) {
  // 将成员变量 tuning_enable_ 设置为传入的值
  tuning_enable_ = value;
  // 如果启用了调优功能，记录日志
  if (value) {
    TUNABLE_LOG1("Enable Tuning for TunableOp");
  }
  else {
    // 如果未启用调优功能，记录日志
    TUNABLE_LOG1("Disable Tuning for TunableOp");
  }
}

// 检查调优是否已启用
bool TuningContext::IsTuningEnabled() const {
  // 获取环境变量 PYTORCH_TUNABLEOP_TUNING 的值
  static const char *env = std::getenv("PYTORCH_TUNABLEOP_TUNING");
  // 如果环境变量存在且值为 "0"，则返回 false
  if (env != nullptr && strcmp(env, "0") == 0) {
    return false;
  }
  // 否则返回 tuning_enable_ 的值
  return tuning_enable_;
}

// 设置程序退出时是否写入文件
void TuningContext::WriteFileOnExit(bool value) {
  // 将成员变量 write_file_on_exit_ 设置为传入的值
  write_file_on_exit_ = value;
}

// 设置是否启用数值检查
void TuningContext::EnableNumericsCheck(bool value) {
  // 将成员变量 numerics_check_enable_ 设置为传入的值
  numerics_check_enable_ = value;
}

// 检查数值检查是否已启用
bool TuningContext::IsNumericsCheckEnabled() const {
  // 获取环境变量 PYTORCH_TUNABLEOP_NUMERICAL_CHECK 的值
  const char *env = getenv("PYTORCH_TUNABLEOP_NUMERICAL_CHECK");
  // 如果环境变量存在且值为 "1"，则返回 true
  if (env != nullptr && strcmp(env, "1") == 0) {
    return true;
  }
  // 否则返回 numerics_check_enable_ 的值
  return numerics_check_enable_;
}

// 设置最大调优持续时间（毫秒）
void TuningContext::SetMaxTuningDurationMs(int max_duration_ms) {
  // 将成员变量 max_tuning_duration_ms_ 设置为传入值，如果小于 0 则设置为 0
  max_tuning_duration_ms_ = max_duration_ms < 0 ? 0 : max_duration_ms;
}

// 获取最大调优持续时间（毫秒）
int TuningContext::GetMaxTuningDurationMs() const {
  // 获取环境变量 PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS 的值
  static const char *env = std::getenv("PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS");
  // 如果环境变量存在，将其转换为整数并返回，如果小于 0 则返回 0
  if (env != nullptr) {
    int val = atoi(env);
    return val < 0 ? 0 : val;
  }
  // 否则返回 max_tuning_duration_ms_ 的值
  return max_tuning_duration_ms_;
}

// 设置最大调优迭代次数
void TuningContext::SetMaxTuningIterations(int max_iter) {
  // 将成员变量 max_tuning_iterations_ 设置为传入值，如果小于 0 则设置为 0
  max_tuning_iterations_ = max_iter < 0 ? 0 : max_iter;
}

// 获取最大调优迭代次数
int TuningContext::GetMaxTuningIterations() const {
  // 获取环境变量 PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS 的值
  static const char *env = std::getenv("PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS");
  // 如果环境变量存在，将其转换为整数并返回，如果小于 0 则返回 0
  if (env != nullptr) {
    int val = atoi(env);
    return val < 0 ? 0 : val;
  }
  // 否则返回 max_tuning_iterations_ 的值
  return max_tuning_iterations_;
}

// 设置最大预热持续时间（毫秒）
void TuningContext::SetMaxWarmupDurationMs(int max_duration_ms) {
  // 将成员变量 max_warmup_duration_ms_ 设置为传入值，如果小于 0 则设置为 0
  max_warmup_duration_ms_ = max_duration_ms < 0 ? 0 : max_duration_ms;
}

// 获取最大预热持续时间（毫秒）
int TuningContext::GetMaxWarmupDurationMs() const {
  // 获取环境变量 PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS 的值
  static const char *env = std::getenv("PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS");
  // 如果环境变量存在，将其转换为整数并返回，如果小于 0 则返回 0
  if (env != nullptr) {
    int val = atoi(env);
    return val < 0 ? 0 : val;
  }
  // 否则返回 max_warmup_duration_ms_ 的值
  return max_warmup_duration_ms_;
}

// 设置最大预热迭代次数
void TuningContext::SetMaxWarmupIterations(int max_iter) {
  // 将成员变量 max_warmup_iterations_ 设置为传入值，如果小于 0 则设置为 0
  max_warmup_iterations_ = max_iter < 0 ? 0 : max_iter;
}

// 获取最大预热迭代次数
int TuningContext::GetMaxWarmupIterations() const {
  // 获取环境变量 PYTORCH_TUNABLEOP_MAX_WARMUP_ITERATIONS 的值
  static const char *env = std::getenv("PYTORCH_TUNABLEOP_MAX_WARMUP_ITERATIONS");
  // 如果环境变量存在，将其转换为整数并返回，如果小于 0 则返回 0
  if (env != nullptr) {
    int val = atoi(env);
    return val < 0 ? 0 : val;
  }
  // 否则返回 max_warmup_iterations_ 的值
  return max_warmup_iterations_;
}

// 设置是否启用 ICache 刷新
void TuningContext::EnableICacheFlush(bool value) {
  // 将成员变量 icache_flush_ 设置为传入的值
  icache_flush_ = value;
}

// 检查是否启用 ICache 刷新
bool TuningContext::IsICacheFlushEnabled() const {
  // 获取环境变量 PYTORCH_TUNABLEOP_ICACHE_FLUSH_ENABLED 的值
  static const char *env = std::getenv("PYTORCH_TUNABLEOP_ICACHE_FLUSH_ENABLED");
  // 如果环境变量存在且值为 "0"，则返回 false
  if (env != nullptr && strcmp(env, "0") == 0) {
    return false;
  }
  // 否则返回 icache_flush_ 的值
  return icache_flush_;
}

// 设置旋转缓冲区大小
void TuningContext::SetRotatingBufferSize(int size) {
  // 将成员变量 rotating_buffer_size_ 设置为传入值，如果小于 0 则设置为 0
  rotating_buffer_size_ = size < 0 ? 0 : size;
}

// 获取旋转缓冲区大小
int TuningContext::GetRotatingBufferSize() const {
  // 获取环境变量 PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE 的值
  static const char *env = std::getenv("PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE");
  // 如果环境变量存在，将其转换为整数并返回，否则返回 rotating_buffer_size_ 的值
  if (env != nullptr) {
    constexpr int MB = 1024 * 1024;
    int val = atoi(env);
    return val < 0 ? 0 : val;
  }
  return rotating_buffer_size_;
}
    return val < 0 ? 0 : val * MB;  // 如果值小于0，则返回0，否则返回以MB为单位的字节大小
  }
  else {
    if (rotating_buffer_size_ < 0) {
      // 如果旋转缓冲区大小为负数（默认情况），则查询当前设备的L2缓存大小
      int l2_cache_size = at::cuda::getCurrentDeviceProperties()->l2CacheSize;
      return l2_cache_size;
    }
    else {
      // 如果旋转缓冲区大小为非负数，则直接返回该大小
      return rotating_buffer_size_;
    }
  }
}

// 返回调优结果管理器的引用
TuningResultsManager& TuningContext::GetTuningResultsManager() {
  // 保证 manager_init_once_ 只被调用一次
  c10::call_once(manager_init_once_, [this]() {
    // 标记管理器已初始化
    manager_initialized_ = true;
    // 如果未设置文件名，使用默认或环境变量中的值
    if (GetFilename().empty()) {
      const char *env = std::getenv("PYTORCH_TUNABLEOP_FILENAME");
      std::string filename = (env == nullptr) ? "tunableop_results.csv" : env;
      // 设置文件名
      SetFilename(filename, true);
    }
    // 获取文件名
    auto filename = GetFilename();
    if (!filename.empty()) {
      // 读取文件内容
      ReadFile(filename);
      // 尝试打开文件以便写入，用于早期错误捕捉
      std::ofstream file(filename, std::ios::out | std::ios::app);
      if (!file.good()) {
        // 如果打开文件失败，发出警告
        TORCH_WARN("failed to open file '", filename, "' for writing; your tuning results will not be saved");
      }
    }
  });
  return manager_;
}

// 返回调优结果验证器的引用
TuningResultsValidator& TuningContext::GetTuningResultsValidator() {
  return validator_;
}

// 获取调优结果
TuningResults TuningContext::GetTuningResults() {
  TuningResults tr;
  // 获取所有验证器
  tr.validators = GetTuningResultsValidator().GetAllValidators();
  // 获取调优结果
  tr.results = GetTuningResultsManager().Dump();
  return tr;
}

// 加载调优结果
TuningStatus TuningContext::LoadTuningResults(const TuningResults& tr) {
  // 验证所有结果的有效性
  TORCH_CHECK(GetTuningResultsValidator().ValidateAll(tr.validators));
  // 加载调优结果到管理器中
  GetTuningResultsManager().Load(tr.results);
  return OK;
}

// 设置文件名
void TuningContext::SetFilename(const std::string& filename, bool insert_device_ordinal) {
  filename_ = filename;

  if (filename_.empty()) {
    return;
  }

  if (insert_device_ordinal) {
    // 根据设备序号区分文件名，避免多个设备同时写入同一文件
    std::string device = c10::str(int(c10::cuda::current_device()));

    // 文件名是否包含 %d 以插入设备序号？
    const std::string TOKEN("%d");
    std::size_t found = filename_.find(TOKEN);
    if (found != std::string::npos) {
      // 替换 %d 为设备序号
      filename_.replace(found, TOKEN.length(), device);
    }
    else {
      // 如果没有 %d，则在最后一个 '.' 前插入设备序号
      found = filename_.rfind(".");
      if (found != std::string::npos) {
        filename_.insert(found, device);
      }
      else {
        // 否则直接追加设备序号
        filename_.append(device);
      }
    }
  }
}

// 获取文件名
std::string TuningContext::GetFilename() const {
  return filename_;
}

// 读取文件内容
bool TuningContext::ReadFile(const std::string& filename_) {
  // 如果未指定具体文件名，则使用当前对象的文件名
  std::string filename = filename_.empty() ? GetFilename() : filename_;
  TUNABLE_LOG1("reading tuning results from ", filename);
  ResultsMap results;
  std::unordered_map<std::string, std::string> validators;
  std::string line;
  std::ifstream file(filename);
  if (!file) {
    // 如果文件无法打开，记录错误
    TUNABLE_LOG1("could not open ", filename, " for reading tuning results");
    return false;
  }
  while (std::getline(file, line)) {
    // 忽略空行
    if (line.empty()) {
      continue;
    }
    std::string part;
    std::vector<std::string> parts;
    // 将输入的字符串 line 转换为字符串流
    std::stringstream line_as_stream(line);
    // 使用逗号作为分隔符，逐个读取 line_as_stream 中的内容到 part 中，并添加到 parts 向量中
    while (std::getline(line_as_stream, part, ',')) {
      parts.push_back(part);
    }
    // 如果 parts 的第一个元素是 "Validator"，且 parts 的大小至少为 3
    if (parts[0] == "Validator" && parts.size() >= 3) {
      // 将 parts[1] 和 parts[2] 添加到 validators 映射中
      validators[parts[1]] = parts[2];
      // 记录日志，表示成功添加了一个验证器及其值
      TUNABLE_LOG1("Validator ", parts[1], "=", parts[2]);
    }
    // 如果 parts 的大小至少为 4
    else if (parts.size() >= 4) {
      // 根据 parts 中的元素创建 ResultEntry 对象，并将其添加到 results 中
      results[parts[0]].emplace(parts[1], ResultEntry(parts[2], atof(parts[3].c_str())));
    }
    // 如果 parts 的大小至少为 3
    else if (parts.size() >= 3) {
      // 使用 parts[2] 创建一个带有默认时间戳 0 的 ResultEntry 对象，并将其添加到 results 中
      // 文件中的时间戳是可选的
      results[parts[0]].emplace(parts[1], ResultEntry(parts[2], 0));
    }
    // 如果上述条件都不满足
    else {
      // 记录日志，表示无法解析当前行 line
      TUNABLE_LOG1("could not parse line: ", line);
    }
  }
  // 对验证器进行全面验证，如果验证通过
  if (GetTuningResultsValidator().ValidateAll(validators) != FAIL) {
    // 载入结果到 manager_ 中
    manager_.Load(results);
    // 记录从输入文件中读取的结果数量
    results_count_from_input_file_ = manager_.GetSize();
  }
  else {
    // 记录日志，表示结果验证失败
    TUNABLE_LOG1("results validator check failed");
    // 返回 false，表示处理失败
    return false;
  }
  // 返回 true，表示处理成功
  return true;
} // namespace at::cuda::tunable



// 关闭命名空间 at::cuda::tunable
}



bool TuningContext::WriteFile(const std::string& filename_) {
    // 使用给定的文件名，如果为空则获取默认文件名
    std::string filename = filename_.empty() ? GetFilename() : filename_;
    // 打开文件以进行写入操作，如果无法打开则记录错误并返回失败
    std::ofstream file(filename, std::ios::out | std::ios::trunc);
    if (!file.good()) {
        TUNABLE_LOG1("error opening tuning results file for writing ", filename);
        return false;
    }
    // 获取所有验证器，并将它们写入文件
    auto validators = GetTuningResultsValidator().GetAllValidators();
    for (const auto& [key, val] : validators) {
        file << "Validator," << key << "," << val << std::endl;
    }
    // 获取所有调优结果，并将其转储到文件
    auto results = GetTuningResultsManager().Dump();
    for (const auto& [op_sig, kernelmap] : results) {
        for (const auto& [param_sig, result] : kernelmap) {
            file << op_sig << "," << param_sig << "," << result << std::endl;
        }
    }
    // 关闭文件
    file.close();
    // 返回写入操作是否成功
    return true;
}



namespace at::cuda::tunable {



// 定义在命名空间 at::cuda::tunable 中的方法 WriteFile，用于写入调优结果到文件中
bool TuningContext::WriteFile(const std::string& filename_) {
    // 使用给定的文件名，如果为空则获取默认文件名
    std::string filename = filename_.empty() ? GetFilename() : filename_;
    // 打开文件以进行写入操作，如果无法打开则记录错误并返回失败
    std::ofstream file(filename, std::ios::out | std::ios::trunc);
    if (!file.good()) {
        TUNABLE_LOG1("error opening tuning results file for writing ", filename);
        return false;
    }
    // 获取所有验证器，并将它们写入文件
    auto validators = GetTuningResultsValidator().GetAllValidators();
    for (const auto& [key, val] : validators) {
        file << "Validator," << key << "," << val << std::endl;
    }
    // 获取所有调优结果，并将其转储到文件
    auto results = GetTuningResultsManager().Dump();
    for (const auto& [op_sig, kernelmap] : results) {
        for (const auto& [param_sig, result] : kernelmap) {
            file << op_sig << "," << param_sig << "," << result << std::endl;
        }
    }
    // 关闭文件
    file.close();
    // 返回写入操作是否成功
    return true;
}
```