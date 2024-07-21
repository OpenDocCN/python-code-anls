# `.\pytorch\aten\src\ATen\test\rng_test.h`

```py
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件
#include <ATen/Generator.h>  // 引入 ATen 库中生成器相关的头文件
#include <ATen/Tensor.h>  // 引入 ATen 库中张量相关的头文件
#include <ATen/native/TensorIterator.h>  // 引入 ATen 库中张量迭代器相关的头文件
#include <torch/library.h>  // 引入 Torch 库的头文件
#include <c10/util/Optional.h>  // 引入 C10 库中的可选类型相关的头文件
#include <torch/all.h>  // 引入 Torch 库的所有头文件
#include <stdexcept>  // 引入标准异常处理相关的头文件

namespace {

constexpr auto int64_min_val = std::numeric_limits<int64_t>::lowest();  // 初始化 int64 类型的最小值
constexpr auto int64_max_val = std::numeric_limits<int64_t>::max();  // 初始化 int64 类型的最大值

template <typename T,
          typename std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
constexpr int64_t _min_val() {
  return int64_min_val;  // 返回浮点数类型 T 对应的最小值，使用 int64 表示
}

template <typename T,
          typename std::enable_if_t<std::is_integral_v<T>, int> = 0>
constexpr int64_t _min_val() {
  return static_cast<int64_t>(std::numeric_limits<T>::lowest());  // 返回整数类型 T 对应的最小值，转换为 int64
}

template <typename T,
          typename std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
constexpr int64_t _min_from() {
  return -(static_cast<int64_t>(1) << std::numeric_limits<T>::digits);  // 返回浮点数类型 T 对应的最小起始值
}

template <typename T,
          typename std::enable_if_t<std::is_integral_v<T>, int> = 0>
constexpr int64_t _min_from() {
  return _min_val<T>();  // 返回整数类型 T 对应的最小起始值
}

template <typename T,
          typename std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
constexpr int64_t _max_val() {
  return int64_max_val;  // 返回浮点数类型 T 对应的最大值，使用 int64 表示
}

template <typename T,
          typename std::enable_if_t<std::is_integral_v<T>, int> = 0>
constexpr int64_t _max_val() {
  return static_cast<int64_t>(std::numeric_limits<T>::max());  // 返回整数类型 T 对应的最大值，转换为 int64
}

template <typename T,
          typename std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
constexpr int64_t _max_to() {
  return static_cast<int64_t>(1) << std::numeric_limits<T>::digits;  // 返回浮点数类型 T 对应的最大结束值
}

template <typename T,
          typename std::enable_if_t<std::is_integral_v<T>, int> = 0>
constexpr int64_t _max_to() {
  return _max_val<T>();  // 返回整数类型 T 对应的最大结束值
}

template<typename RNG, c10::ScalarType S, typename T>
void test_random_from_to(const at::Device& device) {

  constexpr int64_t max_val = _max_val<T>();  // 获取类型 T 的最大值
  constexpr int64_t max_to = _max_to<T>();  // 获取类型 T 的最大结束值

  constexpr auto uint64_max_val = std::numeric_limits<uint64_t>::max();  // 获取 uint64_t 类型的最大值

  std::vector<int64_t> froms;  // 创建起始值的向量
  std::vector<::std::optional<int64_t>> tos;  // 创建可选结束值的向量
  if constexpr (::std::is_same_v<T, bool>) {  // 如果 T 类型为布尔型
    froms = {  // 设置布尔型起始值
      0L
    };
    tos = {  // 设置布尔型结束值
      1L,
      static_cast<::std::optional<int64_t>>(c10::nullopt)
    };
  } else if constexpr (::std::is_signed_v<T>) {  // 如果 T 类型为有符号整数
    constexpr int64_t min_from = _min_from<T>();  // 获取 T 类型的最小起始值
    froms = {  // 设置有符号整数起始值
      min_from,
      -42L,
      0L,
      42L
    };
    tos = {  // 设置有符号整数可选结束值
      ::std::optional<int64_t>(-42L),
      ::std::optional<int64_t>(0L),
      ::std::optional<int64_t>(42L),
      ::std::optional<int64_t>(max_to),
      static_cast<::std::optional<int64_t>>(c10::nullopt)
    };
  } else {  // 如果 T 类型为无符号整数
    froms = {  // 设置无符号整数起始值
      0L,
      42L
    };
    tos = {  // 设置无符号整数可选结束值
      ::std::optional<int64_t>(42L),
      ::std::optional<int64_t>(max_to),
      static_cast<::std::optional<int64_t>>(c10::nullopt)
    };
  }

  const std::vector<uint64_t> vals = {  // 创建无符号整数值的向量
    0L,
    42L,
    static_cast<uint64_t>(max_val),
  // 定义一个包含 max_val 的 static_cast 结果加 1 和 uint64_max_val 的数组
  static_cast<uint64_t>(max_val) + 1,
  uint64_max_val
};

// 初始化覆盖情况标志
bool full_64_bit_range_case_covered = false;
bool from_to_case_covered = false;
bool from_case_covered = false;

// 遍历 froms 数组中的每个 int64_t 元素
for (const int64_t from : froms) {
  // 遍历 tos 数组中的每个 optional<int64_t> 元素的引用
  for (const ::std::optional<int64_t> & to : tos) {
    // 如果 to 不存在或者 from 小于 *to
    if (!to.has_value() || from < *to) {
      // 遍历 vals 数组中的每个 uint64_t 元素
      for (const uint64_t val : vals) {
        // 使用 val 创建一个 RNG 生成器
        auto gen = at::make_generator<RNG>(val);

        // 创建一个形状为 {3, 3}、数据类型为 S、设备为指定 device 的空张量 actual
        auto actual = torch::empty({3, 3}, torch::TensorOptions().dtype(S).device(device));
        
        // 使用 gen 生成随机数填充 actual，范围为 [from, to)
        actual.random_(from, to, gen);

        // 初始化期望值 exp 和范围 range
        T exp;
        uint64_t range;
        
        // 如果 to 不存在且 from 等于 int64_min_val
        if (!to.has_value() && from == int64_min_val) {
          // exp 设置为 val 的 int64_t 类型，并标记全覆盖 64 位范围的情况已被覆盖
          exp = static_cast<int64_t>(val);
          full_64_bit_range_case_covered = true;
        } else {
          // 如果 to 存在
          if (to.has_value()) {
            // 计算 range 为 *to - from，并标记 from_to_case_covered 已被覆盖
            range = static_cast<uint64_t>(*to) - static_cast<uint64_t>(from);
            from_to_case_covered = true;
          } else {
            // 否则计算 range 为 max_to - from + 1，并标记 from_case_covered 已被覆盖
            range = static_cast<uint64_t>(max_to) - static_cast<uint64_t>(from) + 1;
            from_case_covered = true;
          }
          
          // 根据 range 的大小选择 exp 的计算方式
          if (range < (1ULL << 32)) {
            exp = static_cast<T>(static_cast<int64_t>((static_cast<uint32_t>(val) % range + from)));
          } else {
            exp = static_cast<T>(static_cast<int64_t>((val % range + from)));
          }
        }
        
        // 断言 from 小于等于 exp
        ASSERT_TRUE(from <= exp);
        
        // 如果 to 存在，断言 exp 小于 *to
        if (to.has_value()) {
          ASSERT_TRUE(static_cast<int64_t>(exp) < *to);
        }
        
        // 创建一个形状和 actual 相同，值为 exp 的张量 expected
        const auto expected = torch::full_like(actual, exp);
        
        // 如果 T 是 bool 类型，使用 torch::allclose 检查张量的相等性
        if constexpr (::std::is_same_v<T, bool>) {
          ASSERT_TRUE(torch::allclose(actual.toType(torch::kInt), expected.toType(torch::kInt)));
        } else {
          ASSERT_TRUE(torch::allclose(actual, expected));
        }
      }
    }
  }
}

// 如果 T 是 int64_t 类型，断言 full_64_bit_range_case_covered 已被覆盖
if constexpr (::std::is_same_v<T, int64_t>) {
  ASSERT_TRUE(full_64_bit_range_case_covered);
} else {
  (void)full_64_bit_range_case_covered;
}

// 断言 from_to_case_covered 和 from_case_covered 已被覆盖
ASSERT_TRUE(from_to_case_covered);
ASSERT_TRUE(from_case_covered);
}

// 结束函数模板的定义

template<typename RNG, c10::ScalarType S, typename T>
void test_random(const at::Device& device) {
  // 获取类型 T 的最大值
  const auto max_val = _max_val<T>();
  // 获取 uint64_t 类型的最大值
  const auto uint64_max_val = std::numeric_limits<uint64_t>::max();

  // 初始化 uint64_t 类型的向量 vals
  const std::vector<uint64_t> vals = {
    0L,  // 初始值为 0
    42L,  // 初始值为 42
    static_cast<uint64_t>(max_val),  // T 类型最大值的强制转换
    static_cast<uint64_t>(max_val) + 1,  // T 类型最大值加一的强制转换
    uint64_max_val  // uint64_t 类型的最大值
  };

  // 遍历 vals 中的每个值
  for (const uint64_t val : vals) {
    // 使用 val 创建 RNG 类型的随机数生成器 gen
    auto gen = at::make_generator<RNG>(val);

    // 创建一个形状为 {3, 3} 的空张量 actual，数据类型为 S，存储设备为 device
    auto actual = torch::empty({3, 3}, torch::TensorOptions().dtype(S).device(device));
    // 使用 gen 生成随机数填充 actual
    actual.random_(gen);

    uint64_t range;
    // 根据类型 T 的不同条件确定 range 的值
    if constexpr (::std::is_floating_point_v<T>) {
      // 如果 T 是浮点数类型，计算 range
      range = static_cast<uint64_t>((1ULL << ::std::numeric_limits<T>::digits) + 1);
    } else if constexpr (::std::is_same_v<T, bool>) {
      // 如果 T 是布尔类型，设置 range 为 2
      range = 2;
    } else {
      // 否则，根据类型 T 的最大值计算 range
      range = static_cast<uint64_t>(::std::numeric_limits<T>::max()) + 1;
    }
    T exp;
    // 根据类型 T 的不同条件确定 exp 的值
    if constexpr (::std::is_same_v<T, double> || ::std::is_same_v<T, int64_t>) {
      // 如果 T 是 double 或 int64_t 类型，计算 exp
      exp = val % range;
    } else {
      // 否则，将 val 转换为 uint32_t 类型后再计算 exp
      exp = static_cast<uint32_t>(val) % range;
    }

    // 断言 exp 的值应该在 [0, range) 范围内
    ASSERT_TRUE(0 <= static_cast<int64_t>(exp));
    ASSERT_TRUE(static_cast<uint64_t>(exp) < range);

    // 创建一个与 actual 相同形状的张量 expected，填充值为 exp
    const auto expected = torch::full_like(actual, exp);
    if constexpr (::std::is_same_v<T, bool>) {
      // 如果 T 是布尔类型，使用 allclose 检查 actual 和 expected 是否相近（转换为整数类型）
      ASSERT_TRUE(torch::allclose(actual.toType(torch::kInt), expected.toType(torch::kInt)));
    } else {
      // 否则，使用 allclose 检查 actual 和 expected 是否相近
      ASSERT_TRUE(torch::allclose(actual, expected));
    }
  }
}

// 结束函数模板的定义
```