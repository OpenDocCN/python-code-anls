# `.\pytorch\c10\test\core\DispatchKeySet_test.cpp`

```
#include <gtest/gtest.h>  // 包含 Google Test 的头文件

#include <cstddef>  // 包含标准库的头文件
#include <iterator>  // 包含迭代器相关的头文件
#include <unordered_set>  // 包含无序集合的头文件

#include <c10/core/DispatchKeySet.h>  // 包含 C10 的 DispatchKeySet 头文件
#include <c10/util/irange.h>  // 包含 C10 的 irange 头文件

using namespace c10;  // 使用 C10 命名空间

// 此测试不是为了详尽无遗地测试，而是更清楚地展示 DispatchKeySet 的语义。

TEST(DispatchKeySet, Empty) {  // 测试用例：DispatchKeySet 的空集合
  DispatchKeySet empty_set;  // 创建一个空的 DispatchKeySet 对象
  for (uint8_t i = 0;  // 对于 i 从 0 开始循环
       i <= static_cast<uint8_t>(DispatchKey::EndOfRuntimeBackendKeys);  // 当 i 小于等于 DispatchKey 的运行时后端键的结束值时继续循环
       i++) {
    auto tid = static_cast<DispatchKey>(i);  // 将 i 转换为 DispatchKey 类型
    if (tid == DispatchKey::Undefined)  // 如果 tid 是 Undefined，则跳过此次循环
      continue;
    ASSERT_FALSE(empty_set.has(tid));  // 断言：empty_set 中不包含 tid
  }
  ASSERT_TRUE(empty_set.empty());  // 断言：empty_set 是空的
  DispatchKeySet empty_set2;  // 创建另一个空的 DispatchKeySet 对象
  ASSERT_TRUE(empty_set == empty_set2);  // 断言：两个空的 DispatchKeySet 对象相等
}

// 此测试覆盖所有对应于单个后端位的键，例如 BackendComponent::CPUBit。
// 即使这些不是运行时键，我们仍允许直接将它们添加到键集中。

TEST(DispatchKeySet, SingletonBackendComponent) {  // 测试用例：单个后端组件的 DispatchKeySet
  for (const auto i : c10::irange(1, num_backends)) {  // 对于范围在 1 到 num_backends 的所有 i 进行循环
    auto tid = static_cast<DispatchKey>(i);  // 将 i 转换为 DispatchKey 类型
    DispatchKeySet sing(tid);  // 创建一个包含单个 tid 的 DispatchKeySet 对象
    ASSERT_EQ(sing, sing);  // 断言：sing 等于自身
    ASSERT_EQ(sing, DispatchKeySet().add(tid));  // 断言：sing 等于将 tid 添加到空的 DispatchKeySet 后得到的结果
    ASSERT_EQ(sing, sing.add(tid));  // 断言：sing 等于添加 tid 后的自身
    ASSERT_EQ(sing, sing | sing);  // 断言：sing 等于自身与自身的并集
    ASSERT_FALSE(sing.empty());  // 断言：sing 不是空的
    ASSERT_TRUE(sing.has(tid));  // 断言：sing 包含 tid
  }
}

// 此测试覆盖所有对应于单个功能位的键：
// - 运行时、非后端功能键，例如 DispatchKey::FuncTorchBatched
// - 运行时的“虚拟后端”键，例如 DispatchKey::FPGA
// - 非运行时、每个后端的功能键，例如 DispatchKey::Dense
// 即使它不是运行时键，我们仍允许直接将其添加到键集中。

TEST(DispatchKeySet, SingletonFunctionalityKeys) {  // 测试用例：单个功能键的 DispatchKeySet
  for (const auto i : c10::irange(1, num_functionality_keys)) {  // 对于范围在 1 到 num_functionality_keys 的所有 i 进行循环
    auto tid = static_cast<DispatchKey>(i);  // 将 i 转换为 DispatchKey 类型
    DispatchKeySet sing(tid);  // 创建一个包含单个 tid 的 DispatchKeySet 对象
    ASSERT_EQ(sing, sing);  // 断言：sing 等于自身
    ASSERT_EQ(sing, DispatchKeySet().add(tid));  // 断言：sing 等于将 tid 添加到空的 DispatchKeySet 后得到的结果
    ASSERT_EQ(sing, sing.add(tid));  // 断言：sing 等于添加 tid 后的自身
    ASSERT_EQ(sing, sing | sing);  // 断言：sing 等于自身与自身的并集
    ASSERT_FALSE(sing.empty());  // 断言：sing 不是空的
    ASSERT_TRUE(sing.has(tid));  // 断言：sing 包含 tid
    ASSERT_EQ(sing.remove(tid), DispatchKeySet());  // 断言：从 sing 中移除 tid 后得到空的 DispatchKeySet 对象
  }
}

// 此测试覆盖占用 DispatchKeySet 中多个位的每后端运行时键。
// 它们占用一个功能位 + 一个后端位，例如 CPU、CUDA、SparseCPU、SparseCUDA、AutogradCPU、AutogradCUDA

TEST(DispatchKeySet, SingletonPerBackendFunctionalityKeys) {  // 测试用例：每后端功能键的 DispatchKeySet
  for (uint8_t i = static_cast<uint8_t>(DispatchKey::StartOfDenseBackends);
       i <= static_cast<uint8_t>(DispatchKey::EndOfRuntimeBackendKeys);
       i++) {
    auto tid = static_cast<DispatchKey>(i);  // 将 i 转换为 DispatchKey 类型
    // 跳过这些因为它们不是真实的键。
    if (tid == DispatchKey::StartOfDenseBackends ||
        tid == DispatchKey::StartOfSparseBackends ||
        tid == DispatchKey::StartOfQuantizedBackends ||
        tid == DispatchKey::StartOfAutogradFunctionalityBackends) {
      continue;
    }
    DispatchKeySet sing(tid);  // 创建一个包含单个 tid 的 DispatchKeySet 对象
    // 断言 sing 等于 sing 自身，用于确认 sing 的自反性
    ASSERT_EQ(sing, sing);
    // 断言 sing 等于包含给定 tid 的 DispatchKeySet，用于确认 DispatchKeySet 的构造与添加操作
    ASSERT_EQ(sing, DispatchKeySet().add(tid));
    // 断言 sing 等于将给定 tid 添加到 sing 后得到的结果，用于确认 DispatchKeySet 的添加操作
    ASSERT_EQ(sing, sing.add(tid));
    // 断言 sing 等于 sing 与自身的按位或操作结果，用于确认 DispatchKeySet 的按位操作
    ASSERT_EQ(sing, sing | sing);
    // 断言 sing 不为空，用于确认 DispatchKeySet 的空检测方法
    ASSERT_FALSE(sing.empty());
    // 断言 sing 包含给定 tid，用于确认 DispatchKeySet 的成员检测方法
    ASSERT_TRUE(sing.has(tid));

    // 根据给定 tid 获取功能键和后端组件键
    auto functionality_key = toFunctionalityKey(tid);
    auto backend_key = toBackendComponent(tid);
    // 下面两个集合应当是等价的:
    // DispatchKeySet(DispatchKey::CPU)
    // DispatchKeySet({DispatchKey::Dense, BackendComponent::CPUBit})
    auto expected_ks =
        DispatchKeySet(functionality_key) | DispatchKeySet(backend_key);
    // 断言 sing 等于预期的 expected_ks，用于确认 DispatchKeySet 的集合并操作
    ASSERT_EQ(sing, expected_ks);
    // 下面两个集合应当是等价的:
    // DispatchKeySet(DispatchKey::CPU).remove(DispatchKey::Dense)
    // DispatchKeySet(BackendComponent::CPUBit)
    expected_ks = DispatchKeySet(toBackendComponent(tid));
    // 断言 sing 移除给定 tid 后等于预期的 expected_ks，用于确认 DispatchKeySet 的移除操作
    ASSERT_EQ(sing.remove(tid), expected_ks);
  }
TEST(DispatchKeySet, DoubletonPerBackend) {
  // 使用循环遍历从 StartOfDenseBackends 到 EndOfRuntimeBackendKeys 的整数值
  for (uint8_t i = static_cast<uint8_t>(DispatchKey::StartOfDenseBackends);
       i <= static_cast<uint8_t>(DispatchKey::EndOfRuntimeBackendKeys);
       i++) {
    // 循环体留空，无操作
  }
}

TEST(DispatchKeySet, Full) {
  // 创建包含所有 DispatchKey 的 DispatchKeySet 对象
  DispatchKeySet full(DispatchKeySet::FULL);
  // 使用 c10::irange 迭代从 1 到 num_functionality_keys 的值
  for (const auto i : c10::irange(1, num_functionality_keys)) {
    // 将整数 i 转换为 DispatchKey 类型
    auto tid = static_cast<DispatchKey>(i);
    // 断言 full 中包含 tid
    ASSERT_TRUE(full.has(tid));
  }
  // 断言 full 中不包含 DispatchKey::EndOfFunctionalityKeys
  ASSERT_FALSE(full.has(DispatchKey::EndOfFunctionalityKeys));
}

TEST(DispatchKeySet, IteratorBasicOps) {
  // 创建空的 DispatchKeySet 对象
  DispatchKeySet empty_set;
  // 创建包含所有 DispatchKey 的 DispatchKeySet 对象
  DispatchKeySet full_set(DispatchKeySet::FULL);
  // 创建包含 DispatchKey::CPU 的 DispatchKeySet 对象
  DispatchKeySet mutated_set = empty_set.add(DispatchKey::CPU);

  // 构造函数测试 + 比较操作
  ASSERT_EQ(*empty_set.begin(), DispatchKey::EndOfFunctionalityKeys);
  ASSERT_EQ(*empty_set.end(), DispatchKey::EndOfFunctionalityKeys);
  ASSERT_EQ(*mutated_set.begin(), DispatchKey::CPU);

  // 比较迭代器相等性
  ASSERT_TRUE(empty_set.begin() == empty_set.end());
  ASSERT_TRUE(full_set.begin() != full_set.end());

  // 递增操作测试
  ASSERT_TRUE(full_set.begin() == full_set.begin()++);
  ASSERT_TRUE(full_set.begin() != ++full_set.begin());
}

TEST(DispatchKeySet, getHighestPriorityBackendTypeId) {
  // 创建包含 DispatchKey::AutogradCPU 和 DispatchKey::CPU 的 DispatchKeySet 对象
  DispatchKeySet dense_cpu({DispatchKey::AutogradCPU, DispatchKey::CPU});
  // 断言 dense_cpu 中的最高优先级的 DispatchKey 是 DispatchKey::CPU
  ASSERT_EQ(DispatchKey::CPU, c10::highestPriorityBackendTypeId(dense_cpu));

  // 创建包含 DispatchKey::Functionalize 和 DispatchKey::SparseCUDA 的 DispatchKeySet 对象
  DispatchKeySet sparse_cuda(
      {DispatchKey::Functionalize, DispatchKey::SparseCUDA});
  // 断言 sparse_cuda 中的最高优先级的 DispatchKey 是 DispatchKey::SparseCUDA
  ASSERT_EQ(
      DispatchKey::SparseCUDA, c10::highestPriorityBackendTypeId(sparse_cuda));

  // 创建包含 DispatchKey::Functionalize 和 DispatchKey::SparseCsrCUDA 的 DispatchKeySet 对象
  DispatchKeySet sparse_compressed_cuda(
      {DispatchKey::Functionalize, DispatchKey::SparseCsrCUDA});
  // 断言 sparse_compressed_cuda 中的最高优先级的 DispatchKey 是 DispatchKey::SparseCsrCUDA
  ASSERT_EQ(
      DispatchKey::SparseCsrCUDA,
      c10::highestPriorityBackendTypeId(sparse_compressed_cuda));

  // 创建包含 DispatchKey::CUDA 和 DispatchKey::QuantizedCUDA 的 DispatchKeySet 对象
  DispatchKeySet quantized_cuda(
      {DispatchKey::CUDA, DispatchKey::QuantizedCUDA});
  // 断言 quantized_cuda 中的最高优先级的 DispatchKey 是 DispatchKey::QuantizedCUDA
  ASSERT_EQ(
      DispatchKey::QuantizedCUDA,
      c10::highestPriorityBackendTypeId(quantized_cuda));
}

TEST(DispatchKeySet, IteratorEmpty) {
  // 创建空的 DispatchKeySet 对象
  DispatchKeySet empty_set;
  // 初始化计数器 i 为 0
  uint8_t i = 0;

  // 使用迭代器遍历 empty_set
  for (auto it = empty_set.begin(); it != empty_set.end(); ++it) {
    // 每次迭代计数器 i 自增
    i++;
  }
  // 断言迭代器遍历时计数器 i 的值为 0
  ASSERT_EQ(i, 0);
}
TEST(DispatchKeySet, IteratorCrossProduct) {
  // 测试 DispatchKeySet 迭代器的交叉产品功能

  // 创建一个包含 {BackendComponent::CPUBit, BackendComponent::CUDABit} 的 DispatchKeySet 对象
  auto ks =
      DispatchKeySet({BackendComponent::CPUBit, BackendComponent::CUDABit}) |
      DispatchKeySet(
          {DispatchKey::Dense,
           DispatchKey::FPGA,
           DispatchKey::AutogradFunctionality});

  // 获取迭代器的起始位置
  auto iter = ks.begin();

  // 验证迭代器返回的值，首先是 CPU
  ASSERT_EQ(DispatchKey::CPU, *(iter++));
  // 然后是 CUDA
  ASSERT_EQ(DispatchKey::CUDA, *(iter++));
  // FPGA 没有后端比特，因此不包含在交叉产品中
  ASSERT_EQ(DispatchKey::FPGA, *(iter++));
  // 最后是 Autograd CPU
  ASSERT_EQ(DispatchKey::AutogradCPU, *(iter++));
  // 然后是 Autograd CUDA
  ASSERT_EQ(DispatchKey::AutogradCUDA, *(iter++));
}

TEST(DispatchKeySet, IteratorFull) {
  // 创建一个包含所有运行时条目的 DispatchKeySet 对象
  DispatchKeySet full_set(DispatchKeySet::FULL);
  // 计算 DispatchKeySet 的起始到结束之间的元素个数
  std::ptrdiff_t count = std::distance(full_set.begin(), full_set.end());

  // 总的运行时条目数包括一个 DispatchKey::Undefined 条目，
  // 在迭代 DispatchKeySet 时不包含在内
  ASSERT_EQ(count, std::ptrdiff_t{num_runtime_entries} - 1);
}

TEST(DispatchKeySet, FailAtEndIterator) {
  // 创建一个包含所有运行时条目的 DispatchKeySet 对象
  DispatchKeySet full_set(DispatchKeySet::FULL);
  // 获取其原始表示
  uint64_t raw_repr = full_set.raw_repr();

  // 创建一个迭代器，不会抛出异常
  DispatchKeySet::iterator(&raw_repr, num_backends + num_functionality_keys);
  
  // 通过创建一个超出范围的迭代器，期望抛出 c10::Error 异常
  EXPECT_THROW(
      DispatchKeySet::iterator(
          &raw_repr, num_backends + num_functionality_keys + 1),
      c10::Error);
}

TEST(DispatchKeySet, TestBackendComponentToString) {
  // 创建一个用于存放字符串的无序集合
  std::unordered_set<std::string> seen_strings;

  // 遍历所有 BackendComponent 枚举值
  for (int64_t i = 0;
       i <= static_cast<int64_t>(BackendComponent::EndOfBackendKeys);
       i++) {
    auto k = static_cast<BackendComponent>(i);
    auto res = std::string(toString(k));

    // 验证字符串不为 "UNKNOWN_BACKEND_BIT"
    ASSERT_FALSE(res == "UNKNOWN_BACKEND_BIT");
    // 验证字符串在集合中尚未存在
    ASSERT_FALSE(seen_strings.count(res) > 0);
    // 将字符串加入集合中
    seen_strings.insert(res);
  }
}

TEST(DispatchKeySet, TestEndOfRuntimeBackendKeysAccurate) {
  // 定义一个 DispatchKey 变量并初始化为 DispatchKey::Undefined
  DispatchKey k = DispatchKey::Undefined;

  // 根据宏展开设置 DispatchKey 的值
#define SETTER(fullname, prefix) k = DispatchKey::EndOf##fullname##Backends;
  C10_FORALL_FUNCTIONALITY_KEYS(SETTER)
#undef SETTER

  // 验证 k 的值是否等于 DispatchKey::EndOfRuntimeBackendKeys
  ASSERT_TRUE(k == DispatchKey::EndOfRuntimeBackendKeys);
}

TEST(DispatchKeySet, TestFunctionalityDispatchKeyToString) {
  // 创建一个用于存放字符串的无序集合
  std::unordered_set<std::string> seen_strings;

  // 遍历所有 DispatchKey 枚举值
  for (int i = 0; i <= static_cast<int>(DispatchKey::EndOfAliasKeys); i++) {
    auto k = static_cast<DispatchKey>(i);

    // 这些合成的键实际上不会被使用，不需要打印它们的字符串形式
    // 因此这里没有进行详细的验证和插入操作
    // 如果 k 是以下枚举值之一，跳过当前循环：
    // - EndOfFunctionalityKeys
    // - StartOfDenseBackends
    // - StartOfQuantizedBackends
    // - StartOfSparseBackends
    // - StartOfSparseCsrBackends
    // - StartOfNestedTensorBackends
    // - StartOfAutogradFunctionalityBackends
    if (k == DispatchKey::EndOfFunctionalityKeys ||
        k == DispatchKey::StartOfDenseBackends ||
        k == DispatchKey::StartOfQuantizedBackends ||
        k == DispatchKey::StartOfSparseBackends ||
        k == DispatchKey::StartOfSparseCsrBackends ||
        k == DispatchKey::StartOfNestedTensorBackends ||
        k == DispatchKey::StartOfAutogradFunctionalityBackends)
      continue;

    // 将枚举值 k 转换为其对应的字符串表示，并存储在 res 中
    auto res = std::string(toString(k));

    // 使用断言确保字符串 "Unknown" 不在 res 中出现，否则输出以下信息：
    // - 当前索引 i
    // - 之前的枚举值的字符串表示 (toString(static_cast<DispatchKey>(i - 1)))
    ASSERT_TRUE(res.find("Unknown") == std::string::npos)
        << i << " (before is " << toString(static_cast<DispatchKey>(i - 1))
        << ")";

    // 使用断言确保 seen_strings 集合中没有重复的字符串 res
    ASSERT_TRUE(seen_strings.count(res) == 0);

    // 将 res 插入 seen_strings 集合，表示已经处理过该字符串
    seen_strings.insert(res);
  }
}


注释：


# 这是一个代码块的结束标记，对应于某个控制流语句或函数的结束。
# 在这段代码中，它标志着一个代码块的结束，可能是一个条件语句、循环语句或函数定义的结束。
```