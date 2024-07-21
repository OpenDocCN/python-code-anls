# `.\pytorch\torch\csrc\profiler\util.h`

```
#pragma once

#include <cstddef>  // C++ 标准库：提供 size_t 和 nullptr_t 定义
#include <cstdint>  // C++ 标准库：提供整数类型定义
#include <list>     // C++ 标准库：提供双向链表容器定义
#include <string>   // C++ 标准库：提供字符串类定义
#include <unordered_map>  // C++ 标准库：提供无序映射容器定义
#include <vector>   // C++ 标准库：提供动态数组容器定义

#include <ATen/record_function.h>  // PyTorch ATen 库：记录函数调用信息
#include <c10/macros/Macros.h>     // PyTorch C10 库：包含宏定义
#include <c10/util/Optional.h>     // PyTorch C10 库：提供可选类型工具
#include <c10/util/hash.h>         // PyTorch C10 库：提供哈希函数工具
#include <torch/csrc/Export.h>     // PyTorch 导出接口
#include <torch/csrc/jit/frontend/source_range.h>  // PyTorch JIT 前端：源代码范围定义

// TODO: replace with pytorch/rfcs#43 when it is ready.
// 定义软断言宏，用于条件检查，如果条件不满足则记录软断言信息并处理
#define SOFT_ASSERT(cond, ...)                         \
  [&]() -> bool {                                      \
    if (C10_UNLIKELY(!(cond))) {                       \
      torch::profiler::impl::logSoftAssert(            \
          __func__,                                    \
          __FILE__,                                    \
          static_cast<uint32_t>(__LINE__),             \
          #cond,                                       \
          ::c10::str(__VA_ARGS__));                    \
      if (torch::profiler::impl::softAssertRaises()) { \
        TORCH_INTERNAL_ASSERT(cond, __VA_ARGS__);      \
      } else {                                         \
        TORCH_WARN_ONCE(__VA_ARGS__);                  \
      }                                                \
      return false;                                    \
    }                                                  \
    return true;                                       \
  }()

namespace torch::profiler::impl {
// torch::profiler::impl 命名空间：实现细节

// 声明软断言是否抛出异常的函数
TORCH_API bool softAssertRaises();

// 设置软断言是否抛出异常的函数
TORCH_API void setSoftAssertRaises(std::optional<bool> value);

// 记录软断言的详细信息函数重载：C 字符串版本
TORCH_API void logSoftAssert(
    const char* func,
    const char* file,
    uint32_t line,
    const char* cond,
    const char* args);

// 记录软断言的详细信息函数重载：空字符串版本
TORCH_API inline void logSoftAssert(
    const char* func,
    const char* file,
    uint32_t line,
    const char* cond,
    ::c10::detail::CompileTimeEmptyString args) {
  logSoftAssert(func, file, line, cond, (const char*)args);
}

// 记录软断言的详细信息函数重载：字符串版本
TORCH_API void logSoftAssert(
    const char* func,
    const char* file,
    uint32_t line,
    const char* cond,
    const std::string& args);

using shape =
    std::variant<std::vector<int64_t>, std::vector<std::vector<int64_t>>>;
// shape 类型定义：支持 int64_t 的向量或向量的向量

constexpr int TENSOR_LIST_DISPLAY_LENGTH_LIMIT = 30;
// TENSOR_LIST_DISPLAY_LENGTH_LIMIT 常量定义：张量列表显示长度限制

// 根据名称、序列号和形状获取 NVTX 字符串
std::string getNvtxStr(
    const char* name,
    int64_t sequence_nr,
    const std::vector<std::vector<int64_t>>& shapes,
    at::RecordFunctionHandle op_id = 0,
    const std::list<std::pair<at::RecordFunctionHandle, int>>& input_op_ids =
        {});

// 文件、行号、函数名的结构体定义
struct TORCH_API FileLineFunc {
  std::string filename;
  size_t line;
  std::string funcname;
};

// 准备调用堆栈信息函数
TORCH_API std::vector<FileLineFunc> prepareCallstack(
    const std::vector<jit::StackEntry>& cs);

// 调用堆栈信息转字符串函数
TORCH_API std::vector<std::string> callstackStr(
    const std::vector<FileLineFunc>& cs);

// 将堆栈信息字符串化函数
TORCH_API std::string stacksToStr(
    const std::vector<std::string>& stacks,
    const char* delim);

// 获取输入尺寸函数
TORCH_API std::vector<std::vector<int64_t>> inputSizes(
    const at::RecordFunction& fn,
    const bool flatten_list_enabled = false);
// 将 variantShapesToStr 函数声明为 TORCH_API，返回一个描述多种形状的字符串
TORCH_API std::string variantShapesToStr(const std::vector<shape>& shapes);

// 将 shapesToStr 函数声明为 TORCH_API，返回一个描述多维形状的字符串
TORCH_API std::string shapesToStr(
    const std::vector<std::vector<int64_t>>& shapes);

// 将 strListToStr 函数声明为 TORCH_API，返回一个描述字符串列表的字符串
TORCH_API std::string strListToStr(const std::vector<std::string>& types);

// 将 inputOpIdsToStr 函数声明为 TORCH_API，返回一个描述输入操作 ID 对的字符串
TORCH_API std::string inputOpIdsToStr(
    const std::list<std::pair<at::RecordFunctionHandle, int>>& input_op_ids);

// 将 ivalueListToStr 函数声明为 TORCH_API，返回一个描述 IValue 列表的字符串
TORCH_API std::string ivalueListToStr(const std::vector<c10::IValue>& list);

// 将 inputTypes 函数声明为 TORCH_API，返回一个描述输入类型的字符串列表
TORCH_API std::vector<std::string> inputTypes(const at::RecordFunction& fn);

// 在命名空间 torch::profiler::impl 中定义 saveExtraArgs 函数，返回一个从记录函数到额外参数的映射
std::unordered_map<std::string, c10::IValue> TORCH_API
saveExtraArgs(const at::RecordFunction& fn);

// 在命名空间 torch::profiler::impl 中定义 saveNcclMeta 函数，返回一个从记录函数到 NCCL 元信息的映射
std::unordered_map<std::string, std::string> TORCH_API
saveNcclMeta(const at::RecordFunction& fn, bool truncate = true);

// 将 computeFlops 函数声明为 TORCH_API，计算给定操作名和额外参数的浮点操作数
uint64_t TORCH_API computeFlops(
    const std::string& op_name,
    const std::unordered_map<std::string, c10::IValue>& extra_args);

// 定义 shapeToStr 函数，返回一个描述给定形状的字符串
std::string shapeToStr(const std::vector<int64_t>& shape);

// 定义模板类 GlobalStateManager，在命名空间 torch::profiler::impl 中
template <typename T>
class TORCH_API GlobalStateManager {
 public:
  // 返回单例 GlobalStateManager 实例的引用
  static GlobalStateManager& singleton() {
    static GlobalStateManager singleton_;
    return singleton_;
  }

  // 将状态压入单例 GlobalStateManager 实例
  static void push(std::shared_ptr<T>&& state) {
    // 如果已经存在状态，输出警告信息到日志
    if (singleton().state_) {
      LOG(WARNING) << "GlobalStatePtr already exists!";
    } else {
      // 否则移动状态到单例实例
      singleton().state_ = std::move(state);
    }
  }

  // 获取单例 GlobalStateManager 实例的状态指针
  static auto* get() {
    return singleton().state_.get();
  }

  // 弹出单例 GlobalStateManager 实例的状态
  static std::shared_ptr<T> pop() {
    auto out = singleton().state_;
    singleton().state_.reset();
    return out;
  }

 private:
  // 默认构造函数，私有化以确保单例模式
  GlobalStateManager() = default;

  std::shared_ptr<T> state_;  // 单例状态指针
};

// 定义结构体 HashCombine，在命名空间 torch::profiler::impl 中
struct HashCombine {
  // 计算 pair 对象的哈希值
  template <typename T0, typename T1>
  size_t operator()(const std::pair<T0, T1>& i) {
    return c10::get_hash((*this)(i.first), (*this)(i.second));
  }

  // 计算 tuple 对象的哈希值
  template <typename... Args>
  size_t operator()(const std::tuple<Args...>& i) {
    return c10::get_hash(i);
  }

  // 计算单个对象的哈希值
  template <typename T>
  size_t operator()(const T& i) {
    return c10::get_hash(i);
  }
};

#ifdef USE_DISTRIBUTED
// 定义常量 kCommsName，表示通信集合名称
constexpr auto kCommsName = "Collective name";
constexpr auto kDtype = "dtype";  // 定义常量 kDtype，表示数据类型
constexpr auto kInMsgNelems = "In msg nelems";  // 定义常量 kInMsgNelems，表示输入消息长度
constexpr auto kOutMsgNelems = "Out msg nelems";  // 定义常量 kOutMsgNelems，表示输出消息长度
constexpr auto kInSplit = "In split size";  // 定义常量 kInSplit，表示输入分割大小
constexpr auto kOutSplit = "Out split size";  // 定义常量 kOutSplit，表示输出分割大小
constexpr auto kGlobalRankStart = "Global rank start";  // 定义常量 kGlobalRankStart，表示全局排名起始点
constexpr auto kGlobalRankStride = "Global rank stride";  // 定义常量 kGlobalRankStride，表示全局排名步幅
constexpr auto kGroupSize = "Group size";  // 定义常量 kGroupSize，表示组大小
constexpr auto kProcessGroupName = "Process Group Name";  // 定义常量 kProcessGroupName，表示进程组名称
constexpr auto kProcessGroupDesc = "Process Group Description";  // 定义常量 kProcessGroupDesc，表示进程组描述
constexpr auto kGroupRanks = "Process Group Ranks";  // 定义常量 kGroupRanks，表示进程组排名
constexpr auto kRank = "Rank";  // 定义常量 kRank，表示排名
#endif // USE_DISTRIBUTED
```