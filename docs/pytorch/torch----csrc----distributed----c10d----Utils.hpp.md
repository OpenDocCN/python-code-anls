# `.\pytorch\torch\csrc\distributed\c10d\Utils.hpp`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <ATen/ATen.h>
// 包含 ATen 库，用于张量操作

#include <c10/util/Exception.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
// 包含 C10 实用工具，如异常处理、累积函数、迭代范围

#include <torch/csrc/distributed/c10d/Types.hpp>
// 包含 Torch 分布式训练相关的类型定义

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SSIZE_T ssize_t;
#pragma comment(lib, "Ws2_32.lib")
// Windows 平台的头文件和声明，定义 ssize_t 类型
#else
#include <fcntl.h>
#include <netdb.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <unistd.h>
// 非 Windows 平台的头文件，包括文件控制、网络操作等
#endif

#include <sys/types.h>
// 包含 POSIX 系统调用相关的类型定义

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>
// C++ 标准库的头文件，包括整数类型、字符串、向量等

namespace c10d {

TORCH_API size_t getTensorsNumel(const std::vector<at::Tensor>& tensors);
// 声明一个函数用于计算张量列表中所有张量元素的总数

// Retrieve tensor shapes from a given tensor.
TORCH_API std::vector<at::Tensor> getTensorShapes(
    const std::vector<at::Tensor>& tensors);
// 声明一个函数用于从给定张量中获取张量形状

// Use -2 to represent unset state of env vars
#define C10D_ENV_NOT_SET -2
// 定义环境变量未设置状态的常量值为 -2

#define WARN_ENV_VAR_ONCE(deprecated_env, new_env)                        \
  TORCH_WARN_ONCE(                                                        \
      "Environment variable " + deprecated_env + " is deprecated; use " + \
      new_env + " instead");
// 定义一个宏，用于发出一次性警告，提示某环境变量已废弃，建议使用新的环境变量代替

// Turns at::IntArrayRef into "(1, 2, 3, 4)".
inline std::string toString(at::IntArrayRef l) {
  std::stringstream ss;
  ss << "(";
  for (const auto i : c10::irange(l.size())) {
    if (i > 0) {
      ss << ", ";
    }
    ss << l[i];
  }
  ss << ")";
  return ss.str();
}
// 定义一个内联函数，将 at::IntArrayRef 转换为字符串形式，表示为 "(1, 2, 3, 4)"

inline std::string toString(const c10::Layout& layout) {
  std::stringstream ss;
  ss << layout;
  return ss.str();
}
// 定义一个内联函数，将 c10::Layout 转换为字符串形式

inline void assertSameType(
    const at::DeprecatedTypeProperties& type,
    const std::vector<at::Tensor>& tensors) {
  for (const auto i : c10::irange(tensors.size())) {
    if (!tensors[i].options().type_equal(type.options())) {
      const std::string expected = type.toString();
      const std::string actual = tensors[i].toString();
      throw std::invalid_argument(
          // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
          "mixed types (" + expected + " and " + actual + ")");
    }
  }
}
// 定义一个内联函数，用于断言张量列表中所有张量的类型是否与给定类型相同

inline std::vector<std::string> split(
    char separator,
    const std::string& string) {
  std::vector<std::string> pieces;
  std::stringstream ss(string);
  std::string item;
  while (std::getline(ss, item, separator)) {
    pieces.push_back(std::move(item));
  }
  return pieces;
}
// 定义一个内联函数，用于将字符串按指定分隔符分割成子串列表

inline std::string getCvarString(
    const std::vector<std::string>& env,
    const char* def) {
  const char* ret = def;

  if (env.empty()) {
    TORCH_CHECK(false, "No environment variables passed");
    return ret;
  }

  /* parse environment variable in reverse order, so the early
   * versions of a variable get higher priority than the latter
   * versions of the same variable */
  for (ssize_t i = static_cast<ssize_t>(env.size()) - 1; i >= 0; i--) {
    const char* val = std::getenv(env[i].c_str());
    if (val == nullptr) {
      continue;
    } else if (i) {
      WARN_ENV_VAR_ONCE(env[i], env[0]);
    }

    ret = val;
  }

  return ret;
}
// 定义一个内联函数，根据优先级解析环境变量，返回优先级最高的环境变量值
inline int getCvarInt(const std::vector<std::string>& env, int def) {
  int ret = def; // 默认返回值为传入的默认值

  if (env.empty()) {
    TORCH_CHECK(false, "No environment variables passed"); // 如果环境变量为空，抛出错误并返回默认值
    return ret;
  }

  /* parse environment variable in reverse order, so the early
   * versions of a variable get higher priority than the latter
   * versions of the same variable */
  for (ssize_t i = static_cast<ssize_t>(env.size()) - 1; i >= 0; i--) {
    char* val = std::getenv(env[i].c_str()); // 获取环境变量的值
    if (val == nullptr) { // 如果值为空，继续下一个环境变量
      continue;
    } else if (i) {
      WARN_ENV_VAR_ONCE(env[i], env[0]); // 警告，暂未定义
    }

    try {
      ret = std::stoi(val); // 将环境变量的值转换为整数
    } catch (std::exception&) {
      TORCH_CHECK(false, "Invalid value for environment variable: " + env[i]); // 捕获转换异常，抛出错误
    }
  }

  return ret; // 返回最终解析得到的环境变量值
}

inline bool getCvarBool(const std::vector<std::string>& env, bool def) {
  bool ret = def; // 默认返回值为传入的默认值

  if (env.empty()) {
    TORCH_CHECK(false, "No environment variables passed"); // 如果环境变量为空，抛出错误并返回默认值
    return ret;
  }

  /* parse environment variable in reverse order, so the early
   * versions of a variable get higher priority than the latter
   * versions of the same variable */
  for (ssize_t i = static_cast<ssize_t>(env.size()) - 1; i >= 0; i--) {
    char* val_ = std::getenv(env[i].c_str()); // 获取环境变量的值
    if (val_ == nullptr) { // 如果值为空，继续下一个环境变量
      continue;
    } else if (i) {
      WARN_ENV_VAR_ONCE(env[i], env[0]); // 警告，暂未定义
    }

    std::string val = std::string(val_);
    for (auto& x : val) {
      // NOLINTNEXTLINE(*-narrowing-conversions)
      x = std::tolower(x); // 将环境变量值转换为小写
    }

    if (val == "y" || val == "yes" || val == "1" || val == "t" ||
        val == "true") {
      ret = true; // 如果值符合 true 的各种形式，则设为 true
    } else if (
        val == "n" || val == "no" || val == "0" || val == "f" ||
        val == "false") {
      ret = false; // 如果值符合 false 的各种形式，则设为 false
    } else {
      TORCH_CHECK(false, "Invalid value for environment variable: " + env[i]); // 否则抛出错误
      return ret;
    }
  }

  return ret; // 返回最终解析得到的环境变量布尔值
}

inline void assertSameSizes(
    const at::IntArrayRef& sizes,
    const std::vector<at::Tensor>& tensors) {
  for (const auto i : c10::irange(tensors.size())) {
    if (!tensors[i].sizes().equals(sizes)) {
      const auto expected = toString(sizes); // 将预期尺寸转换为字符串
      const auto actual = toString(tensors[i].sizes()); // 将实际尺寸转换为字符串
      throw std::invalid_argument(
          // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
          "mixed sizes (" + expected + " and " + actual + ")"); // 抛出包含尺寸信息的异常
    }
  }
}

inline void assertSameSizeAndType(const std::vector<at::Tensor>& tensors) {
  // Ensure we have at least one tensor
  if (tensors.empty()) {
    throw std::invalid_argument("argument is empty"); // 如果张量数组为空，抛出错误
  }

  // Ensure all tensors have identical type and shape
  auto options = tensors[0].options(); // 获取第一个张量的选项
  auto sizes = tensors[0].sizes(); // 获取第一个张量的尺寸
  for (const auto i : c10::irange(1, tensors.size())) {
    // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
    if (tensors[i].options() != options || !tensors[i].sizes().equals(sizes)) {
      throw std::invalid_argument(
          // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
          "mixed types and sizes (" + toString(options) + " and " +
              toString(tensors[i].options()) + ", " + toString(sizes) + " and " +
              toString(tensors[i].sizes()) + ")"); // 抛出包含类型和尺寸信息的异常
    }
  }
}
    // 检查当前张量与给定选项是否类型不匹配
    if (!tensors[i].options().type_equal(options)) {
      // 转换期望的选项和当前张量的选项为字符串
      const auto expected = toString(options);
      const auto actual = toString(tensors[i].options());
      // 抛出异常，指示参数包含混合类型
      throw std::invalid_argument(
          // 禁止下一行进行性能-低效的字符串连接（NOLINT）
          "argument contains mixed types (" + expected + " and " + actual +
          ")");
    }
    // 检查当前张量与给定尺寸是否尺寸不匹配
    if (!tensors[i].sizes().equals(sizes)) {
      // 转换期望的尺寸和当前张量的尺寸为字符串
      const auto expected = toString(sizes);
      const auto actual = toString(tensors[i].sizes());
      // 抛出异常，指示参数包含混合类型
      throw std::invalid_argument(
          // 禁止下一行进行性能-低效的字符串连接（NOLINT）
          "argument contains mixed types (" + expected + " and " + actual +
          ")");
    }
// 确保给定索引处的张量类型与期望类型匹配，否则调用给定的函数 fn 报告错误信息
inline void assertTypeMatch(
    const std::function<void(const std::string&)>& fn,
    const at::DeprecatedTypeProperties& type,
    const at::ArrayRef<at::Tensor> tensors,
    size_t index) {
  // 如果张量的类型与期望的类型不匹配
  if (!tensors[index].options().type_equal(type.options())) {
    // 调用函数 fn 报告错误信息，指明索引处的张量类型不符合预期
    fn("invalid tensor type at index " + std::to_string(index) + " (expected " +
       type.toString() + ", got " + tensors[index].toString() + ")");
  }
}

// 确保给定索引处的张量类型与期望选项匹配，否则调用给定的函数 fn 报告错误信息
inline void assertTypeMatch(
    const std::function<void(const std::string&)>& fn,
    const at::TensorOptions& options,
    const at::ArrayRef<at::Tensor> tensors,
    size_t index) {
  // 如果张量的类型选项与期望的选项不匹配
  if (!tensors[index].options().type_equal(options)) {
    // 调用函数 fn 报告错误信息，指明索引处的张量类型不符合预期
    fn("invalid tensor type at index " + std::to_string(index) + " (expected " +
       toString(options) + ", got " + toString(tensors[index].options()) + ")");
  }
}

// 确保给定索引处的张量尺寸与期望尺寸匹配，否则调用给定的函数 fn 报告错误信息
inline void assertSizesMatch(
    const std::function<void(const std::string&)>& fn,
    const at::IntArrayRef& sizes,
    const at::ArrayRef<at::Tensor> tensors,
    size_t index) {
  // 如果张量的尺寸与期望的尺寸不匹配
  if (tensors[index].sizes() != sizes) {
    // 调用函数 fn 报告错误信息，指明索引处的张量尺寸不符合预期
    fn("invalid tensor size at index " + std::to_string(index) + " (expected " +
       toString(sizes) + ", got " + toString(tensors[index].sizes()) + ")");
  }
}

// 确保给定索引处的张量布局与期望布局匹配，否则调用给定的函数 fn 报告错误信息
inline void assertLayoutMatch(
    const std::function<void(const std::string&)>& fn,
    const c10::Layout& expected,
    const at::ArrayRef<at::Tensor> tensors,
    size_t index) {
  // 获取张量的实际布局
  const auto& actual = tensors[index].layout();
  // 如果实际布局与期望布局不匹配
  if (actual != expected) {
    // 调用函数 fn 报告错误信息，指明索引处的张量布局不符合预期
    fn("invalid tensor layout at index " + std::to_string(index) +
       " (expected " + toString(expected) + ", got " + toString(actual) + ")");
  }
}

// 确保张量列表中所有张量的布局与第一个张量的布局匹配，否则调用给定的函数 fn 报告错误信息
inline void assertLayoutMatch(
    const std::function<void(const std::string&)>& fn,
    const at::ArrayRef<at::Tensor> tensors) {
  // 获取列表中第一个张量的布局
  const auto& layout = tensors[0].layout();
  // 遍历列表中的每个张量（除第一个张量外）
  for (const auto i : c10::irange(1, tensors.size())) {
    // 调用 assertLayoutMatch 函数确保每个张量的布局与第一个张量的布局匹配
    assertLayoutMatch(fn, layout, tensors, i);
  }
}

// 确保张量列表非空，否则调用给定的函数 fn 报告错误信息
inline void assertNonEmpty(
    const std::function<void(const std::string&)>& fn,
    const at::ArrayRef<at::Tensor> tensors) {
  // 如果张量列表为空
  if (tensors.empty()) {
    // 调用函数 fn 报告错误信息，指明需要非空的张量列表
    fn("requires non-empty tensor list");
  }
}

// 确保张量列表中仅有一个张量，否则调用给定的函数 fn 报告错误信息
inline void assertSingleElement(
    const std::function<void(const std::string&)>& fn,
    const at::ArrayRef<at::Tensor> tensors) {
  // 如果张量列表中不止一个张量
  if (tensors.size() != 1) {
    // 调用函数 fn 报告错误信息，指明需要仅有一个张量的列表
    fn("requires a single-element tensor list");
  }
}

// 确保输入张量列表中仅有一个张量，否则调用给定的函数 fn 报告错误信息
inline void assertSingleElementInput(
    const std::function<void(const std::string&)>& fn,
    const at::ArrayRef<at::Tensor> tensors) {
  // 如果输入张量列表中不止一个张量
  if (tensors.size() != 1) {
    // 调用函数 fn 报告错误信息，指明需要仅有一个输入张量的列表
    fn("requires a single-element input tensor list");
  }
}

// 确保输出张量列表中仅有一个张量，否则调用给定的函数 fn 报告错误信息
inline void assertSingleElementOutput(
    const std::function<void(const std::string&)>& fn,
    const at::ArrayRef<at::Tensor> tensors) {
  // 如果输出张量列表中不止一个张量
  if (tensors.size() != 1) {
    // 调用函数 fn 报告错误信息，指明需要仅有一个输出张量的列表
    fn("requires a single-element output tensor list");
  }
}

// 确保 rank 大小在 [0, size) 范围内，否则调用给定的函数 fn 报告错误信息
inline void assertRootRank(
    const std::function<void(const std::string&)>& fn,
    int64_t rank,
    int64_t size) {
  // 如果 rank 小于 0 或大于等于 size
  if (rank < 0 || rank >= size) {
    // 调用函数 fn 报告错误信息，指明 rank 超出有效范围
    fn("invalid root rank (expected between 0 and " + std::to_string(size) + ", got " + std::to_string(rank) + ")");
  }
}
    fn("invalid root rank: " + std::to_string(rank));


注释：


// 使用 rank 的值构造一个字符串，然后添加到 "invalid root rank: " 字符串后面，
// 最终传递给函数 fn 进行处理。
// 结束函数 assertRootTensor 的定义
inline void assertRootTensor(
    const std::function<void(const std::string&)>& fn, // 函数参数：接收一个函数对象和两个整数
    int64_t rank, // 根张量的秩
    int64_t size) { // 张量的总数
  if (rank < 0 || rank >= size) { // 如果根张量的秩不在有效范围内
    fn("invalid root tensor: " + std::to_string(rank)); // 调用传入的函数对象，报告无效根张量
  }
}

// 结束函数 assertDense 的定义
inline void assertDense(
    const std::function<void(const std::string&)>& fn, // 函数参数：接收一个函数对象和一个张量数组引用
    const at::ArrayRef<at::Tensor> tensors) { // 输入的张量数组
  const auto& layout = tensors[0].layout(); // 获取第一个张量的布局
  if (layout != at::kStrided) { // 如果布局不是步进布局
    fn("only supports dense tensors"); // 调用传入的函数对象，报告只支持密集张量
  }
}

// 结束函数 assertCPU 的定义
inline void assertCPU(
    const std::function<void(const std::string&)>& fn, // 函数参数：接收一个函数对象和一个张量数组引用
    const at::ArrayRef<at::Tensor> tensors) { // 输入的张量数组
  const auto& device = tensors[0].device(); // 获取第一个张量的设备
  if (device.type() != at::kCPU) { // 如果设备类型不是 CPU
    fn("only supports CPU tensors"); // 调用传入的函数对象，报告只支持 CPU 张量
  }
}

// 结束函数 assertSameDevice 的定义
inline void assertSameDevice(
    const std::function<void(const std::string&)>& fn, // 函数参数：接收一个函数对象和一个张量数组引用
    const at::ArrayRef<at::Tensor> tensors) { // 输入的张量数组
  if (tensors.size() < 2) { // 如果张量数组长度小于 2
    return; // 直接返回
  }
  const auto& device = tensors[0].device(); // 获取第一个张量的设备
  for (const auto i : c10::irange(1, tensors.size())) { // 遍历除第一个张量之外的所有张量
    if (tensors[i].device() != device) { // 如果有张量的设备与第一个张量的设备不同
      fn("tensors should be on the same device"); // 调用传入的函数对象，报告张量应在相同设备上
    }
  }
}

// 结束函数 assertTypeAndSizesMatch 的定义，重载版本1
inline void assertTypeAndSizesMatch(
    const std::function<void(const std::string&)>& fn, // 函数参数：接收一个函数对象和一个张量数组引用
    const at::ArrayRef<at::Tensor> tensors,
    const at::DeprecatedTypeProperties& type, // 张量类型的属性
    const at::IntArrayRef& sizes) { // 张量的大小
  for (const auto i : c10::irange(tensors.size())) { // 遍历所有张量
    assertTypeMatch(fn, type, tensors, i); // 检查张量类型是否匹配
    assertSizesMatch(fn, sizes, tensors, i); // 检查张量大小是否匹配
  }
}

// 结束函数 assertTypeAndSizesMatch 的定义，重载版本2
inline void assertTypeAndSizesMatch(
    const std::function<void(const std::string&)>& fn, // 函数参数：接收一个函数对象和一个张量数组引用
    const at::ArrayRef<at::Tensor> tensors,
    const at::TensorOptions& options, // 张量的选项
    const at::IntArrayRef& sizes) { // 张量的大小
  for (const auto i : c10::irange(tensors.size())) { // 遍历所有张量
    assertTypeMatch(fn, options, tensors, i); // 检查张量类型是否匹配
    assertSizesMatch(fn, sizes, tensors, i); // 检查张量大小是否匹配
  }
}

// 结束函数 assertTypeAndSizesMatch 的定义，重载版本3
inline void assertTypeAndSizesMatch(
    const std::function<void(const std::string&)>& fn, // 函数参数：接收一个函数对象和一个张量数组引用
    const at::ArrayRef<at::Tensor> tensors) { // 输入的张量数组
  const auto& options = tensors[0].options(); // 获取第一个张量的选项
  const auto sizes = tensors[0].sizes(); // 获取第一个张量的大小
  assertTypeAndSizesMatch(fn, tensors.slice(1), options, sizes); // 调用重载版本2，检查所有张量的类型和大小是否匹配
}

// 从 ATen/core/functional.h 复制的模板函数
template <typename F, typename T>
inline auto fmap(T& inputs, const F& fn)
    -> std::vector<decltype(fn(*inputs.begin()))> { // 函数模板：将函数 fn 映射到 inputs 上
  std::vector<decltype(fn(*inputs.begin()))> r; // 结果向量
  r.reserve(inputs.size()); // 预留空间
  for (auto& input : inputs) { // 遍历 inputs 中的每一个元素
    r.push_back(fn(input)); // 对每个元素应用 fn 函数，并将结果添加到 r 中
  }
  return r; // 返回结果向量
}

// 从 torch/csrc/utils/tensor_flatten.h 复制的函数
inline at::Tensor flattenDenseTensors(at::TensorList tensors) { // 将密集张量列表展平为一个张量
  static const auto flatten = [](const at::Tensor& t) { // 定义静态的展平函数
    return t.contiguous().view({-1}); // 返回连续的张量视图
  };
  if (tensors.size() == 1) { // 如果张量列表只有一个张量
    return flatten(tensors[0]); // 直接展平这个张量并返回
  }
  return at::cat(::c10d::fmap(tensors, flatten)); // 否则，将所有张量展平并连接起来
}

// 结束函数 newLikeFlat 的定义
inline at::Tensor newLikeFlat(
    std::vector<std::vector<at::Tensor>>& tensors, // 二维张量向量
    size_t deviceIdx) { // 设备索引
  if (tensors.empty() || tensors[0].empty()) { // 如果张量向量为空或第一个向量为空
    // 检查条件，如果为假，则抛出异常并显示消息
    TORCH_CHECK(false, "Received an empty list");

  }
  // 检查设备索引是否超出张量列表的范围，如果是，则抛出异常并显示消息
  if (deviceIdx >= tensors.size()) {
    TORCH_CHECK(false, "Invalid device index");
  }
  // 获取指定设备索引位置上第一个张量的引用
  auto& t = tensors[deviceIdx][0];
  // 获取该张量的设备类型
  auto device = t.device();
  // 遍历该设备索引位置上的所有张量，确保它们都在同一设备上，否则抛出异常并显示消息
  for (const auto i : c10::irange(1, tensors[deviceIdx].size())) {
    if (tensors[deviceIdx][i].device() != device) {
      TORCH_CHECK(false, "Expecting all tensors on the same device");
    }
  }
  // 使用at::DeviceGuard保护当前设备，防止后续操作改变设备
  at::DeviceGuard gpuGuard(device);
  // 创建存储张量大小和步长的向量
  std::vector<int64_t> sizes{static_cast<int64_t>(tensors[deviceIdx].size())};
  std::vector<int64_t> strides{static_cast<int64_t>(t.numel())};
  // 将张量t的大小和步长添加到sizes和strides向量中
  sizes.insert(sizes.end(), t.sizes().begin(), t.sizes().end());
  strides.insert(strides.end(), t.strides().begin(), t.strides().end());
  // 返回一个新的空张量，其大小和步长与给定的sizes和strides相匹配
  return at::empty_strided(
      sizes, strides, t.options().memory_format(c10::nullopt));
}

// inline 函数，创建一个和第一个张量相同类型和大小的空张量
inline at::Tensor newLikeFlat(std::vector<at::Tensor>& tensors) {
  // 检查张量列表是否为空，如果是，则抛出错误信息
  if (tensors.empty()) {
    TORCH_CHECK(false, "Received an empty list");
  }
  // 获取第一个张量的引用
  auto& t = tensors[0];
  // 切换到第一个张量所在的设备
  at::DeviceGuard gpuGuard(t.device());
  // 创建一个大小为 [张量列表长度, t的大小] 的空张量
  std::vector<int64_t> sizes{static_cast<int64_t>(tensors.size())};
  sizes.insert(sizes.end(), t.sizes().begin(), t.sizes().end());
  return at::empty(sizes, t.options());
}

// inline 函数，获取张量列表中每个张量的大小
inline std::vector<std::vector<int64_t>> getSizes(
    const std::vector<at::Tensor>& tensors) {
  // 创建一个大小为张量列表长度的二维向量，用于存放每个张量的大小
  std::vector<std::vector<int64_t>> sizes(tensors.size());
  // 遍历张量列表，获取每个张量的大小并存入 sizes 中
  for (const auto i : c10::irange(tensors.size())) {
    sizes[i] = tensors[i].sizes().vec();
  }
  return sizes;
}

// inline 函数，获取张量列表中每个张量所在的设备编号
inline std::vector<int> getDevices(const std::vector<at::Tensor>& tensors) {
  // 创建一个大小为张量列表长度的整型向量，初始值为 -1
  std::vector<int> devices(tensors.size(), -1);
  // 如果第一个张量在 CUDA 设备上，获取每个张量的设备索引
  if (tensors[0].device().is_cuda()) {
    for (const auto i : c10::irange(tensors.size())) {
      // NOLINTNEXTLINE(bugprone-signed-char-misuse)
      devices[i] = tensors[i].storage().device().index();
    }
  }
  return devices;
}

// 模板函数，获取指定类型的张量数据指针
template <typename T>
inline T* getDataPointer(const at::Tensor& tensor) {
  // 该方法目前仅在 ProcessGroupGloo 中使用，调用方必须确保输入张量是连续的
  // 如果张量不是从存储的开始位置开始，使用 data_ptr() 而不是 tensor.storage().data()
  // 注意：不使用 tensor.data<T>()，因为 tensor 不了解 gloo::TYPE
  return static_cast<T*>(tensor.data_ptr());
}

// 模板函数，获取指定类型的张量列表的数据指针
template <typename T>
std::vector<T*> getDataPointers(const std::vector<at::Tensor>& tensors) {
  // 创建一个大小为张量列表长度的指针向量，存放每个张量的数据指针
  std::vector<T*> ptrs(tensors.size());
  // 遍历张量列表，获取每个张量的数据指针并存入 ptrs 中
  for (const auto i : c10::irange(tensors.size())) {
    ptrs[i] = getDataPointer<T>(tensors[i]);
  }
  return ptrs;
}

// inline 函数，用于检查 alltoall 操作的分割大小是否合理
inline void checkSplitSizes(
    const std::vector<int64_t>& split_sizes,
    const at::Tensor& tensor,
    int group_size) {
  // 如果分割大小列表为空，检查张量的第 0 维度是否能被 group_size 整除
  if (split_sizes.empty()) {
    TORCH_CHECK(
        tensor.size(0) % group_size == 0,
        "Tensor's dim 0 does not divide equally across group size");
  } else {
    // 否则，检查分割大小列表的长度是否等于 group_size
    TORCH_CHECK(
        split_sizes.size() == static_cast<size_t>(group_size),
        "Number of tensor splits not equal to group size");
    // 检查分割大小列表的总和是否等于张量的第 0 维度大小
    const auto sum = c10::sum_integers(split_sizes);
    TORCH_CHECK(
        sum == tensor.size(0), "Split sizes doesn't match total dim 0 size");
  }
}

// 模板函数，计算 alltoall 操作的长度和偏移量，处理多维张量
template <typename T>
size_t computeLengthsAndOffsets(
    const std::vector<int64_t>& split_sizes,
    const at::Tensor& tensor,
    std::vector<T>* lengths,
    std::vector<T>* offsets) {
  // 初始化变量
  size_t group_size = lengths->size();
  bool equal_splits = false;
  size_t dim0_size = tensor.size(0);
  size_t row_size = (dim0_size ? tensor.numel() / dim0_size : 1);
  size_t split_size = 0;
  size_t offset = 0;

  // 如果分割大小列表为空
  if (split_sizes.empty()) {
    equal_splits = true;
    // 计算每个分组的大小，tensor.size(0) 是张量的第一个维度大小
    split_size = tensor.size(0) / group_size;
  }
  // 遍历每个分组
  for (const auto i : c10::irange(group_size)) {
    // 计算当前分组的长度，根据 equal_splits 确定是否使用相同的分割大小
    size_t length = row_size * (equal_splits ? split_size : split_sizes[i]);
    // 将当前分组的长度存入 lengths 中
    (*lengths)[i] = length;
    // 将当前分组的偏移量存入 offsets 中
    (*offsets)[i] = offset;
    // TODO: 看是否需要为偏移量添加溢出保护
    // 更新偏移量，为下一个分组做准备
    offset += length;
  }
  // 返回最终的偏移量
  return offset;
// 定义一个模板函数，计算张量组中每个张量的长度和偏移量，并填充给定的长度和偏移量向量
template <typename T>
size_t computeLengthsAndOffsets(
    const std::vector<at::Tensor>& tensors,  // 输入的张量向量
    std::vector<T>* lengths,                  // 输出的长度向量指针
    std::vector<T>* offsets) {                // 输出的偏移量向量指针
  size_t group_size = lengths->size();        // 获取长度向量的大小
  size_t offset = 0;                          // 初始化偏移量为0
  for (const auto i : c10::irange(group_size)) {  // 对于每个张量的索引
    size_t length = tensors[i].numel();       // 计算当前张量的元素个数
    (*lengths)[i] = length;                   // 将计算得到的长度存入长度向量
    (*offsets)[i] = offset;                   // 将当前偏移量存入偏移量向量
    offset += length;                         // 更新偏移量，累加当前张量的长度
  }
  return offset;                              // 返回总的偏移量
}

// 使用别名定义 RankType 和 SizeType 分别为 uint32_t 和 uint64_t
using RankType = uint32_t;
using SizeType = uint64_t;

// 如果操作系统为 Windows
#ifdef _WIN32
// 定义宏 SYSCHECK，用于检查表达式执行是否成功，并处理错误情况
#define SYSCHECK(expr, success_cond)                                      \
  while (true) {                                                          \
    auto __output = (expr);                                               \
    auto errno_local = WSAGetLastError();                                 \
    (void)__output;                                                       \
    if (!(success_cond)) {                                                \
      if (errno == EINTR) {                                               \
        continue;                                                         \
      } else if (                                                         \
          errno_local == WSAETIMEDOUT || errno_local == WSAEWOULDBLOCK) { \
        C10_THROW_ERROR(DistNetworkError, "Socket Timeout");              \
      } else {                                                            \
        C10_THROW_ERROR(DistNetworkError, std::strerror(errno_local));    \
      }                                                                   \
    } else {                                                              \
      break;                                                              \
    }                                                                     \
  }
#else
// 如果操作系统为非 Windows，定义宏 SYSCHECK，用于检查表达式执行是否成功，并处理错误情况
#define SYSCHECK(expr, success_cond)                             \
  while (true) {                                                 \
    auto __output = (expr);                                      \
    (void)__output;                                              \
    if (!(success_cond)) {                                       \
      if (errno == EINTR) {                                      \
        continue;                                                \
      } else {                                                   \
        C10_THROW_ERROR(DistNetworkError, std::strerror(errno)); \
      }                                                          \
    } else {                                                     \
      break;                                                     \
    }                                                            \
  }
#endif
    // 如果未达到成功条件，则根据不同的错误类型进行处理
    if (!(success_cond)) {
      // 如果错误是由于信号中断导致的，继续循环尝试
      if (errno == EINTR) {
        continue;
      }
      // 如果错误是超时错误或者是非阻塞操作导致的阻塞错误，抛出DistNetworkError异常并指明“Socket Timeout”
      else if (errno == EAGAIN || errno == EWOULDBLOCK) {
        C10_THROW_ERROR(DistNetworkError, "Socket Timeout");
      }
      // 对于其他类型的错误，抛出DistNetworkError异常并使用标准错误描述作为错误信息
      else {
        C10_THROW_ERROR(DistNetworkError, std::strerror(errno));
      }
    }
    // 如果成功达到条件，跳出循环
    else {
      break;
    }
#endif

// `#endif` 结束条件编译指令块

// 大多数函数通过返回 `-1` 表示错误。这是一个常见情况的辅助宏，用于 `SYSCHECK`。
// 因为在 MSVC 中，`SOCKET_ERROR = -1`，因此也可以使用 `SYSCHECK_ERR_RETURN_NEG1`。
#define SYSCHECK_ERR_RETURN_NEG1(expr) SYSCHECK(expr, __output != -1)

// 检查张量是否包含 NaN（Not a Number）
void checkForNan(const at::Tensor& tensor);

namespace tcputil {

// 发送和接收数据的模板函数
template <typename T>
void sendBytes(
    int socket,
    const T* buffer,
    size_t length,
    bool moreData = false) {
  size_t bytesToSend = sizeof(T) * length;
  if (bytesToSend == 0) {
    return;
  }

  auto currentBytes = reinterpret_cast<const char*>(buffer);

  int flags = 0;

#ifdef MSG_MORE
  // 如果有更多数据需要发送，则设置 `MSG_MORE` 标志
  if (moreData) {
    flags |= MSG_MORE;
  }
#endif

// 忽略 SIGPIPE 信号，因为 send() 的返回值总是检查错误
#ifdef MSG_NOSIGNAL
  flags |= MSG_NOSIGNAL;
#endif

  // 循环发送数据直到所有数据发送完毕
  while (bytesToSend > 0) {
    ssize_t bytesSent = 0;
    // 使用 `send()` 发送数据，并检查返回值是否为错误
    SYSCHECK_ERR_RETURN_NEG1(
        bytesSent = ::send(socket, currentBytes, bytesToSend, flags))
    // 如果 `send()` 返回 0，抛出连接重置的异常
    if (bytesSent == 0) {
      C10_THROW_ERROR(DistNetworkError, std::strerror(ECONNRESET));
    }

    bytesToSend -= bytesSent;
    currentBytes += bytesSent;
  }
}

// 接收数据的模板函数
template <typename T>
void recvBytes(int socket, T* buffer, size_t length) {
  size_t bytesToReceive = sizeof(T) * length;
  if (bytesToReceive == 0) {
    return;
  }

  auto currentBytes = reinterpret_cast<char*>(buffer);

  // 循环接收数据直到所有数据接收完毕
  while (bytesToReceive > 0) {
    ssize_t bytesReceived = 0;
    // 使用 `recv()` 接收数据，并检查返回值是否为错误
    SYSCHECK_ERR_RETURN_NEG1(
        bytesReceived = recv(socket, currentBytes, bytesToReceive, 0))
    // 如果 `recv()` 返回 0，抛出连接重置的异常
    if (bytesReceived == 0) {
      C10_THROW_ERROR(DistNetworkError, std::strerror(ECONNRESET));
    }

    bytesToReceive -= bytesReceived;
    currentBytes += bytesReceived;
  }
}

// 发送向量的长度和数据
template <typename T>
void sendVector(int socket, const std::vector<T>& vec, bool moreData = false) {
  SizeType size = vec.size();
  // 先发送向量的长度
  sendBytes<SizeType>(socket, &size, 1, true);
  // 再发送向量的数据
  sendBytes<T>(socket, vec.data(), size, moreData);
}

// 接收通过 `sendVector` 发送的向量数据
template <typename T>
std::vector<T> recvVector(int socket) {
  SizeType valueSize = 0;
  // 首先接收向量的长度
  recvBytes<SizeType>(socket, &valueSize, 1);
  // 然后根据长度接收向量的数据
  std::vector<T> value(valueSize);
  recvBytes<T>(socket, value.data(), value.size());
  return value;
}

// 方便发送 rvalue（右值）时使用的模板函数
template <typename T>
void sendValue(int socket, const T& value, bool moreData = false) {
  // 直接发送值
  sendBytes<T>(socket, &value, 1, moreData);
}

// 接收值的模板函数
template <typename T>
T recvValue(int socket) {
  T value;
  // 直接接收值
  recvBytes<T>(socket, &value, 1);
  return value;
}

// 发送字符串的长度和数据
inline void sendString(
    int socket,
    const std::string& str,
    bool moreData = false) {
  SizeType size = str.size();
  // 首先发送字符串的长度
  sendBytes<SizeType>(socket, &size, 1, true);
  // 然后发送字符串的数据
  sendBytes<char>(socket, str.data(), size, moreData);
}

// 接收通过 `sendString` 发送的字符串数据
// 接收字符串数据并返回为 std::string 对象
inline std::string recvString(int socket) {
  // 声明并初始化变量 valueSize，用于存储接收到的字符串长度
  SizeType valueSize = 0;
  // 调用 recvBytes 函数接收 SizeType 类型的数据，填充到 valueSize 变量中
  recvBytes<SizeType>(socket, &valueSize, 1);
  // 创建一个字符向量 value，大小为 valueSize，用于存储接收到的字符串内容
  std::vector<char> value(valueSize);
  // 调用 recvBytes 函数接收字符数据，填充到 value 向量中
  recvBytes<char>(socket, value.data(), value.size());
  // 根据接收到的字符数据创建并返回一个 std::string 对象
  return std::string(value.data(), value.size());
}

// 命名空间结束符，结束 tcputil 命名空间
} // namespace tcputil
// 命名空间结束符，结束 c10d 命名空间
} // namespace c10d
```