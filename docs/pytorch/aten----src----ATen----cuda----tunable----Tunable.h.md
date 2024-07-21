# `.\pytorch\aten\src\ATen\cuda\tunable\Tunable.h`

```py
// Original TunableOp is from onnxruntime.
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/framework/tunable.h
// https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/rocm/tunable
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Adapting TunableOp into PyTorch
// Copyright (c) Advanced Micro Devices, Inc.
//
#pragma once

#include <c10/util/CallOnce.h>  // 包含 C10 库中的 CallOnce 功能

#include <fstream>              // 文件输入输出流
#include <functional>           // 函数对象库
#include <iostream>             // 标准输入输出流库
#include <memory>               // 智能指针库
#include <mutex>                // 互斥量库
#include <string>               // 字符串库
#include <type_traits>          // 类型特性库
#include <unordered_map>        // 无序映射库
#include <utility>              // 实用工具库
#include <vector>               // 向量库

namespace at::cuda::tunable {

namespace detail {

// 定义一个结构体 MaybeDelete，用于管理是否拥有指针所有权
struct MaybeDelete {
  bool owns_pointer;
  void operator()(std::ostream* os) const { if (owns_pointer) delete os; }
};

// 使用 unique_ptr 包装 ostream 指针，通过 MaybeDelete 确保安全地释放指针
using OstreamPtr = std::unique_ptr<std::ostream, MaybeDelete>;

// 根据文件名获取输出流指针
static OstreamPtr get_stream(std::string filename) {
  if (filename.compare("out") == 0) {
    return OstreamPtr { &std::cout, MaybeDelete {false} };  // 如果文件名是 "out"，返回标准输出流
  }
  else if (filename.compare("err") == 0) {
    return OstreamPtr { &std::cerr, MaybeDelete {false} };  // 如果文件名是 "err"，返回标准错误流
  }
  else {
    return OstreamPtr { new std::ofstream {filename.c_str()}, MaybeDelete {true} };  // 否则创建文件输出流
  }
}

}

// 定义静态函数 TunableLog，用于记录调试信息到指定的输出流
static void TunableLog(int level, const std::string& msg) {
  static const char *env_file = getenv("PYTORCH_TUNABLEOP_VERBOSE_FILENAME");  // 获取环境变量 PYTORCH_TUNABLEOP_VERBOSE_FILENAME
  static const char *env_verbose = getenv("PYTORCH_TUNABLEOP_VERBOSE");        // 获取环境变量 PYTORCH_TUNABLEOP_VERBOSE
  static int level_user = env_verbose ? atoi(env_verbose) : 0;                 // 将环境变量 PYTORCH_TUNABLEOP_VERBOSE 转换为整数
  static auto streamptr = detail::get_stream(env_file ? env_file : "err");     // 获取日志输出流，默认为标准错误流
  if (level_user >= level) {
    (*streamptr) << msg << std::endl;  // 如果用户指定的日志级别大于等于当前级别，将消息写入日志流
  }
}

// 定义宏 TUNABLE_LOGV，用于记录不同级别的详细调试信息
#define TUNABLE_LOGV(LEVEL, ...) TunableLog(LEVEL, c10::str(__VA_ARGS__))
#define TUNABLE_LOG1(...) TUNABLE_LOGV(1, __VA_ARGS__)
#define TUNABLE_LOG2(...) TUNABLE_LOGV(2, __VA_ARGS__)
#define TUNABLE_LOG3(...) TUNABLE_LOGV(3, __VA_ARGS__)

// 定义枚举类型 TuningStatus，表示调优状态
enum TORCH_CUDA_CPP_API TuningStatus {
  OK = 0,           // 调优成功
  FAIL = 1,         // 调优失败
  UNSUPPORTED = 2,  // 不支持的调优
};

// 定义类 ResultEntry，表示参数签名到内核 ID 的映射
class TORCH_CUDA_CPP_API ResultEntry {
public:
  explicit ResultEntry(const std::string& key, double time) : key_(key), time_(time) {}  // 构造函数，初始化参数签名和执行时间
  bool operator==(const ResultEntry& other) { return key_ == other.key_; }              // 比较运算符重载，判断是否相等
  bool operator!=(const ResultEntry& other) { return key_ != other.key_; }              // 比较运算符重载，判断是否不等
  operator std::string () { return key_; }                                               // 类型转换运算符重载，将对象转换为字符串
  std::string GetKey() const { return key_; }                                            // 获取参数签名的方法
  double GetTime() const { return time_; }                                                // 获取执行时间的方法
  friend std::ostream& operator<<(std::ostream& stream, const ResultEntry& entry);       // 友元函数声明，用于输出流中打印对象信息
  static ResultEntry Null() { return ResultEntry("Null", 0.0); }                          // 静态方法，返回空对象
  static ResultEntry Default() { return ResultEntry("Default", 0.0); }                    // 静态方法，返回默认对象

private:
  std::string key_;  // 参数签名
  double time_;      // 执行时间
};

// 定义类型别名 KernelMap 和 ResultsMap，分别表示参数签名到内核 ID 的映射和结果映射
typedef std::unordered_map<std::string, ResultEntry> KernelMap;
typedef std::unordered_map<std::string, KernelMap> ResultsMap;
struct TORCH_CUDA_CPP_API TuningResults {
  // 校验这些结果是否与库兼容
  std::unordered_map<std::string, std::string> validators;  // 使用哈希表存储校验器

  // 从可调用签名到调整结果的映射
  ResultsMap results;  // 调整结果的映射
};

class TORCH_CUDA_CPP_API TuningResultsManager {
  public:
    TuningResultsManager() = default;  // 默认构造函数
    ~TuningResultsManager() = default;  // 默认析构函数

    KernelMap Lookup(const std::string& op_signature);  // 查找操作签名对应的内核映射

    ResultEntry Lookup(const std::string& op_signature, const std::string& params_signature);  // 查找操作签名和参数签名对应的结果条目

    inline void AddImpl(const std::string& op_signature,
        const std::string& params_signature,
        ResultEntry best,
        KernelMap& kernel_map);  // 内联函数，向内核映射中添加操作签名和参数签名对应的最佳结果条目

    void Add(const std::string& op_signature,
        const std::string& params_signature,
        ResultEntry best);  // 向结果管理器中添加操作签名和参数签名对应的最佳结果

    void Delete(const std::string& op_signature, const std::string& params_signature);  // 删除操作签名和参数签名对应的条目

    inline void DisjointMergeImpl(
        const std::string& op_signature,
        const KernelMap& kernel_map,
        /*out*/ ResultsMap& results);  // 内联函数，执行操作签名对应的内核映射与结果映射的不相交合并

    void Load(const ResultsMap& results_to_load);  // 载入给定的结果映射

    ResultsMap Dump();  // 导出当前的结果映射

    void DisjointMerge(const std::string& op_signature, const KernelMap& kernel_map);  // 执行操作签名对应的内核映射与结果映射的不相交合并

    size_t GetSize();  // 获取结果映射的大小

  private:
    std::mutex lock_;  // 互斥量，用于保护结果映射的并发访问
    ResultsMap results_;  // 存储操作签名到结果的映射
};

class TORCH_CUDA_CPP_API TuningResultsValidator {
  public:
    using GetFunc = std::function<std::string()>;
    using ValidateFunc = std::function<TuningStatus(const std::string&)>;
    using GetValidateFuncs = std::unordered_map<std::string, std::pair<GetFunc, ValidateFunc>>;

    TuningResultsValidator();  // 默认构造函数
    ~TuningResultsValidator() = default;  // 默认析构函数

    std::unordered_map<std::string, std::string> GetAllValidators() const;  // 获取所有验证器

    TuningStatus ValidateAll(const std::unordered_map<std::string, std::string>& to_validate) const;  // 对所有给定的验证器进行验证

    void RegisterValidator(const std::string& key, const GetFunc& gf, const ValidateFunc& vf);  // 注册一个验证器函数

  protected:
    std::string GetPyTorchVersion() const;  // 获取 PyTorch 版本信息

    TuningStatus ValidatePyTorchVersion(const std::string& value) const;  // 验证 PyTorch 版本信息

  public:
    static constexpr const std::array mandatory_keys{"PT_VERSION"};  // 强制的键名数组

  private:
    GetValidateFuncs validators_;  // 存储验证器函数的哈希表
};

class TORCH_CUDA_CPP_API TuningContext {
  public:
    TuningContext();  // 默认构造函数
    ~TuningContext();  // 默认析构函数
    TuningContext(TuningContext &) = delete;  // 禁用拷贝构造函数
    TuningContext(TuningContext &&) = delete;  // 禁用移动构造函数
    TuningContext &operator=(TuningContext &) = delete;  // 禁用拷贝赋值运算符
    TuningContext &operator=(TuningContext &&) = delete;  // 禁用移动赋值运算符

    void EnableTunableOp(bool value);  // 启用或禁用可调整操作的标志位
    bool IsTunableOpEnabled() const;  // 检查是否启用了可调整操作

    void EnableTuning(bool value);  // 启用或禁用调整的标志位
    bool IsTuningEnabled() const;  // 检查是否启用了调整

    void EnableNumericsCheck(bool value);  // 启用或禁用数值检查的标志位
    bool IsNumericsCheckEnabled() const;  // 检查是否启用了数值检查

    void SetMaxTuningDurationMs(int max_duration_ms);  // 设置最大调整持续时间（毫秒）
    int GetMaxTuningDurationMs() const;  // 获取最大调整持续时间（毫秒）

    void SetMaxTuningIterations(int max_iter);  // 设置最大调整迭代次数
    int GetMaxTuningIterations() const;  // 获取最大调整迭代次数

    void SetMaxWarmupDurationMs(int max_duration_ms);  // 设置最大预热持续时间（毫秒）
    // 返回最大预热持续时间（毫秒）
    int GetMaxWarmupDurationMs() const;

    // 设置最大预热迭代次数
    void SetMaxWarmupIterations(int max_iter);
    // 获取最大预热迭代次数
    int GetMaxWarmupIterations() const;

    // 启用或禁用指令缓存刷新
    void EnableICacheFlush(bool value);
    // 检查指令缓存刷新是否已启用
    bool IsICacheFlushEnabled() const;

    // 设置旋转缓冲区大小
    void SetRotatingBufferSize(int size);
    // 获取旋转缓冲区大小
    int GetRotatingBufferSize() const;

    // 获取调优结果管理器的引用
    TuningResultsManager& GetTuningResultsManager();

    // 获取调优结果验证器的引用
    TuningResultsValidator& GetTuningResultsValidator();

    // 获取调优结果
    TuningResults GetTuningResults();

    // 载入调优结果并返回载入状态
    TuningStatus LoadTuningResults(const TuningResults& tr);

    // 设置文件名，可以选择是否插入设备序号
    void SetFilename(const std::string& filename, bool insert_device_ordinal=false);
    // 获取当前设置的文件名
    std::string GetFilename() const;

    // 在退出时设置是否写文件
    void WriteFileOnExit(bool value);

    // 读取指定文件，如果未指定文件名则返回false
    bool ReadFile(const std::string& filename={});
    // 写入指定文件，如果未指定文件名则返回false
    bool WriteFile(const std::string& filename={});

  private:
    // 启用状态
    bool enable_;
    // 是否启用调优
    bool tuning_enable_;
    // 管理器是否已初始化
    bool manager_initialized_;
    // 是否在退出时写文件
    bool write_file_on_exit_;
    // 是否启用数值检查
    bool numerics_check_enable_;
    // 最大调优持续时间（毫秒）
    int max_tuning_duration_ms_;
    // 最大调优迭代次数
    int max_tuning_iterations_;
    // 最大预热持续时间（毫秒）
    int max_warmup_duration_ms_;
    // 最大预热迭代次数
    int max_warmup_iterations_;
    // 是否启用指令缓存刷新
    bool icache_flush_;
    // 旋转缓冲区大小
    int rotating_buffer_size_;
    // 调优结果管理器
    mutable TuningResultsManager manager_;
    // 管理器初始化的标志
    mutable c10::once_flag manager_init_once_;
    // 调优结果验证器
    TuningResultsValidator validator_;
    // 当前设置的文件名
    std::string filename_;
    // 从输入文件中获取的结果计数
    size_t results_count_from_input_file_;
};

// 声明一个名为 getTuningContext 的函数，返回类型为 TuningContext 指针
TORCH_CUDA_CPP_API TuningContext* getTuningContext();

// 定义一个名为 ITimer 的抽象基类
class ITimer {
  public:
    // 构造函数声明，默认生成
    ITimer() = default;
    // 虚析构函数声明，默认生成
    virtual ~ITimer() = default;

    // 纯虚函数，用于启动计时
    virtual void Start() = 0;
    // 纯虚函数，用于结束计时
    virtual void End() = 0;

    // 纯虚函数，计算 Start() 和 End() 之间经过的时间，返回毫秒
    virtual float Duration() = 0;
};

// 命名空间结束标记，当前命名空间为 at::cuda::tunable
} // namespace at::cuda::tunable
```