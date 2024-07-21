# `.\pytorch\aten\src\ATen\core\IListRef_test.cpp`

```
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif
#include <ATen/core/IListRef.h>
#include <gtest/gtest.h>
#include <algorithm>

using namespace c10;

// 定义一个静态函数，返回包含空张量的向量
static std::vector<at::Tensor> get_tensor_vector() {
  std::vector<at::Tensor> tensors;
  const size_t SIZE = 5;
  // 循环五次，向向量中添加空张量
  for (size_t i = 0; i < SIZE; i++) {
    tensors.emplace_back(at::empty({0}));
  }
  return tensors;
}

// 定义一个静态函数，返回包含可选空张量的向量
static std::vector<optional<at::Tensor>> get_boxed_opt_tensor_vector() {
  std::vector<optional<at::Tensor>> optional_tensors;
  const size_t SIZE = 5;
  // 循环十次，根据偶数索引添加空张量或空值到向量中
  for (size_t i = 0; i < SIZE * 2; i++) {
    auto opt_tensor = (i % 2 == 0) ? optional<at::Tensor>(at::empty({0})) : nullopt;
    optional_tensors.emplace_back(opt_tensor);
  }
  return optional_tensors;
}

// 定义一个静态函数，返回包含不可选空张量引用的向量
static std::vector<at::OptionalTensorRef> get_unboxed_opt_tensor_vector() {
  static std::vector<at::Tensor> tensors;
  std::vector<at::OptionalTensorRef> optional_tensors;
  constexpr size_t SIZE = 5;
  // 循环五次，向静态张量向量中添加空张量，并向向量中添加对这些张量的引用
  for (size_t i = 0; i < SIZE; i++) {
    tensors.push_back(at::empty({0}));
    optional_tensors.emplace_back(tensors[i]);
    optional_tensors.emplace_back();
  }
  return optional_tensors;
}

// 模板函数，比较两个张量列表是否相同
template <typename T>
void check_elements_same(at::ITensorListRef list, const T& thing, int use_count) {
  EXPECT_EQ(thing.size(), list.size());
  size_t i = 0;
  // 遍历列表中的每个张量，并比较其是否与给定的张量列表中的对应张量相同
  for (const auto& t : list) {
    const at::Tensor& other = thing[i];
    EXPECT_EQ(other.use_count(), use_count);
    EXPECT_TRUE(other.is_same(t));
    i++;
  }
}

// 测试用例，测试空张量列表的构造函数是否为None，并期望抛出错误
TEST(ITensorListRefTest, CtorEmpty_IsNone_Throws) {
  at::ITensorListRef list;
  EXPECT_TRUE(list.isNone());
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_THROW(list.size(), c10::Error);
}

// 测试用例，测试包含包装过的张量列表的构造函数是否为Boxed
TEST(ITensorListRefTest, CtorBoxed_IsBoxed) {
  auto vec = get_tensor_vector();
  List<at::Tensor> boxed(vec);
  at::ITensorListRef list(boxed);
  EXPECT_TRUE(list.isBoxed());
}

// 测试用例，测试包含未包装的张量列表的构造函数是否为Unboxed
TEST(ITensorListRefTest, CtorUnboxed_IsUnboxed) {
  auto vec = get_tensor_vector();
  at::ArrayRef<at::Tensor> unboxed(vec);
  at::ITensorListRef list(unboxed);
  EXPECT_TRUE(list.isUnboxed());
}

// 测试用例，测试包含间接未包装的张量列表的构造函数是否为Unboxed
TEST(ITensorListRefTest, CtorUnboxedIndirect_IsUnboxed) {
  auto vec = get_tensor_vector();
  auto check_is_unboxed = [](const at::ITensorListRef& list) {
    EXPECT_TRUE(list.isUnboxed());
  };
  // 检查多种初始化方式下的列表是否为Unboxed
  check_is_unboxed(at::ITensorListRef{vec[0]});
  check_is_unboxed(at::ITensorListRef{vec.data(), vec.size()});
  check_is_unboxed(at::ITensorListRef{vec.data(), vec.data() + vec.size()});
  check_is_unboxed(vec);
  check_is_unboxed({vec[0], vec[1], vec[2]});
}

// 测试用例，测试临时张量列表的构造函数是否为Unboxed
TEST(ITensorListRefTest, CtorTemp_IsUnboxed) {
  auto check_is_unboxed = [](const at::ITensorListRef& list) {
    EXPECT_TRUE(list.isUnboxed());
  };

  auto vec = get_tensor_vector();
  check_is_unboxed({vec[0], vec[1]});
}
TEST(ITensorListRefTest, Boxed_GetConstRefTensor) {
  // 获取张量向量
  auto vec = get_tensor_vector();
  // 将张量向量包装成常量引用的列表
  // 因为 List<Tensor>::operator[] 返回 ListElementReference 而非 Tensor，
  // 而 List<Tensor>::operator[] const 返回 const Tensor &
  const List<at::Tensor> boxed(vec);
  // 使用 boxed 构造 ITensorListRef 对象
  at::ITensorListRef list(boxed);
  // 断言 ITensorListRef 返回的元素是 const Tensor 引用
  static_assert(
      std::is_same_v<decltype(*list.begin()), const at::Tensor&>,
      "Accessing elements from List<Tensor> through a ITensorListRef should be const references.");
  // 检查 boxed[0] 是否与 list.begin() 指向同一张量
  EXPECT_TRUE(boxed[0].is_same(*list.begin()));
  // 检查 boxed[1] 是否与 ++list.begin() 指向同一张量
  EXPECT_TRUE(boxed[1].is_same(*(++list.begin())));
}

TEST(ITensorListRefTest, Unboxed_GetConstRefTensor) {
  // 获取张量向量
  auto vec = get_tensor_vector();
  // 使用张量向量构造 ITensorListRef 对象
  at::ITensorListRef list(vec);
  // 断言 ITensorListRef 返回的元素是 const Tensor 引用
  static_assert(
      std::is_same_v<decltype(*list.begin()), const at::Tensor&>,
      "Accessing elements from ArrayRef<Tensor> through a ITensorListRef should be const references.");
  // 检查 vec[0] 是否与 list.begin() 指向同一张量
  EXPECT_TRUE(vec[0].is_same(*list.begin()));
  // 检查 vec[1] 是否与 ++list.begin() 指向同一张量
  EXPECT_TRUE(vec[1].is_same(*(++list.begin())));
}

TEST(ITensorListRefTest, Boxed_Equal) {
  // 获取张量向量
  auto vec = get_tensor_vector();
  // 使用张量向量构造 boxed 列表
  List<at::Tensor> boxed(vec);
  // 检查 boxed 与 vec 的元素是否相同，使用计数为 2
  check_elements_same(boxed, vec, /* use_count= */ 2);
}

TEST(ITensorListRefTest, Unboxed_Equal) {
  // 获取张量向量
  auto vec = get_tensor_vector();
  // 检查 ArrayRef<Tensor> 与 vec 的元素是否相同，使用计数为 1
  check_elements_same(at::ArrayRef<at::Tensor>(vec), vec, /* use_count= */ 1);
}

TEST(ITensorListRefTest, UnboxedIndirect_Equal) {
  // 4 个引用计数位置：
  //   1. vec
  //   2. ITensorListRef 的 initializer_list
  //   3. std::vector 的 initializer_list
  //   4. 临时 std::vector
  auto vec = get_tensor_vector();
  // 使用隐式构造函数
  // 检查 vec[0] 与 std::vector<at::Tensor>{vec[0]} 的元素是否相同，使用计数为 3
  check_elements_same(vec[0], std::vector<at::Tensor>{vec[0]}, /* use_count= */ 3);
  // 检查 {vec.data(), vec.size()} 与 vec 的元素是否相同，使用计数为 1
  check_elements_same({vec.data(), vec.size()}, vec, /* use_count= */ 1);
  // 检查 {vec.data(), vec.data() + vec.size()} 与 vec 的元素是否相同，使用计数为 1
  check_elements_same({vec.data(), vec.data() + vec.size()}, vec, /* use_count= */ 1);
  // 使用向量构造函数
  // 检查 vec 与 vec 的元素是否相同，使用计数为 1
  check_elements_same(vec, vec, /* use_count= */ 1);
  // 使用初始化列表构造函数
  // 检查 {vec[0], vec[1], vec[2]} 与 std::vector<at::Tensor>{vec[0], vec[1], vec[2]} 的元素是否相同，使用计数为 4
  check_elements_same({vec[0], vec[1], vec[2]}, std::vector<at::Tensor>{vec[0], vec[1], vec[2]}, /* use_count= */ 4);
}

TEST(ITensorListRefTest, BoxedMaterialize_Equal) {
  // 获取张量向量
  auto vec = get_tensor_vector();
  // 使用张量向量构造 boxed 列表
  List<at::Tensor> boxed(vec);
  // 使用 boxed 构造 ITensorListRef 对象
  at::ITensorListRef list(boxed);
  // 将 list materialize 成标准张量向量
  auto materialized = list.materialize();
  // 检查 list 与 vec 的元素是否相同，使用计数为 2
  check_elements_same(list, vec, 2);
  // 检查 list 与 materialized 的元素是否相同，使用计数为 2
  check_elements_same(list, materialized, 2);
  // 检查 materialized 与 vec 的元素是否相同，使用计数为 2
  check_elements_same(materialized, vec, 2);
}

TEST(ITensorListRefTest, UnboxedMaterialize_Equal) {
  // 获取张量向量
  auto vec = get_tensor_vector();
  // 使用张量向量构造 ArrayRef<at::Tensor> 对象
  at::ArrayRef<at::Tensor> unboxed(vec);
  // 使用 unboxed 构造 ITensorListRef 对象
  at::ITensorListRef list(unboxed);
  // 将 list materialize 成标准张量向量
  auto materialized = list.materialize();
  // 检查 list 与 vec 的元素是否相同，使用计数为 1
  check_elements_same(list, vec, 1);
  // 检查 list 与 materialized 的元素是否相同，使用计数为 1
  check_elements_same(list, materialized, 1);
  // 检查 materialized 与 vec 的元素是否相同，使用计数为 1
  check_elements_same(materialized, vec, 1);
}
TEST(ITensorListRefIteratorTest, CtorEmpty_ThrowsError) {
  // 创建一个空的 ITensorListRefIterator 对象
  at::ITensorListRefIterator* it = new at::ITensorListRefIterator();
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  // 预期访问 **it 会抛出 c10::Error 异常
  EXPECT_THROW(**it, c10::Error);

  // 根据编译器和迭代器调试级别，预期删除指针 it 时可能抛出 c10::Error 异常
#if defined(_MSC_VER) && _ITERATOR_DEBUG_LEVEL == 2
  EXPECT_THROW({ delete it; }, c10::Error);
#else
  delete it;
#endif
}

TEST(ITensorListRefIteratorTest, Boxed_GetFirstElement) {
  // 获取张量向量
  auto vec = get_tensor_vector();
  // 使用 List<at::Tensor> 封装张量向量
  const List<at::Tensor> boxed(vec);
  // 创建 ITensorListRef 对象
  at::ITensorListRef list(boxed);
  // 验证 ITensorListRef 的开头元素与 boxed 的第一个元素相同
  EXPECT_TRUE(boxed[0].is_same(*list.begin()));
}

TEST(ITensorListRefIteratorTest, Unboxed_GetFirstElement) {
  // 获取张量向量
  auto vec = get_tensor_vector();
  // 创建 ITensorListRef 对象
  at::ITensorListRef list(vec);
  // 验证 ITensorListRef 的开头元素与 vec 的第一个元素相同
  EXPECT_TRUE(vec[0].is_same(*list.begin()));
}

TEST(ITensorListRefIteratorTest, Boxed_Equality) {
  // 获取张量向量
  auto vec = get_tensor_vector();
  // 使用 List<at::Tensor> 封装张量向量
  List<at::Tensor> boxed(vec);
  // 创建 ITensorListRef 对象
  at::ITensorListRef list(boxed);
  // 验证迭代器的相等性
  EXPECT_EQ(list.begin(), list.begin());
  EXPECT_NE(list.begin(), list.end());
  EXPECT_NE(list.end(), list.begin());
  EXPECT_EQ(list.end(), list.end());
}

TEST(ITensorListRefIteratorTest, Unboxed_Equality) {
  // 获取张量向量
  auto vec = get_tensor_vector();
  // 创建 ITensorListRef 对象
  at::ITensorListRef list(vec);
  // 验证迭代器的相等性
  EXPECT_EQ(list.begin(), list.begin());
  EXPECT_NE(list.begin(), list.end());
  EXPECT_NE(list.end(), list.begin());
  EXPECT_EQ(list.end(), list.end());
}

TEST(ITensorListRefIteratorTest, Boxed_Iterate) {
  // 获取张量向量
  auto vec = get_tensor_vector();
  // 使用 List<at::Tensor> 封装张量向量
  const List<at::Tensor> boxed(vec);
  // 创建 ITensorListRef 对象
  at::ITensorListRef list(boxed);
  size_t i = 0;
  // 遍历 ITensorListRef 对象
  for (const auto& t : list) {
    // 验证 boxed 的第 i 个元素与迭代器返回的 t 是否相同
    EXPECT_TRUE(boxed[i++].is_same(t));
  }
  EXPECT_EQ(i, list.size());
}

TEST(ITensorListRefIteratorTest, Unboxed_Iterate) {
  // 获取张量向量
  auto vec = get_tensor_vector();
  // 创建 ITensorListRef 对象
  at::ITensorListRef list(vec);
  size_t i = 0;
  // 遍历 ITensorListRef 对象
  for (const auto& t : list) {
    // 验证 vec 的第 i 个元素与迭代器返回的 t 是否相同
    EXPECT_TRUE(vec[i++].is_same(t));
  }
  EXPECT_EQ(i, list.size());
}

TEST(IOptTensorListRefTest, Boxed_Iterate) {
  // 获取可选张量向量
  auto vec = get_boxed_opt_tensor_vector();
  // 使用 List<optional<at::Tensor>> 封装可选张量向量
  const List<optional<at::Tensor>> boxed(vec);
  // 创建 IOptTensorListRef 对象
  at::IOptTensorListRef list(boxed);
  size_t i = 0;
  // 遍历 IOptTensorListRef 对象
  for (const auto t : list) {
    // 验证 boxed 的第 i 个元素是否与 t 的可选性一致
    EXPECT_EQ(boxed[i].has_value(), t.has_value());
    if (t.has_value()) {
      // 验证 boxed 的第 i 个元素与 t 的值是否相同
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      EXPECT_TRUE((*boxed[i]).is_same(*t));
    }
    i++;
  }
  EXPECT_EQ(i, list.size());
}

TEST(IOptTensorListRefTest, Unboxed_Iterate) {
  // 获取非封箱可选张量向量
  auto vec = get_unboxed_opt_tensor_vector();
  // 使用 at::ArrayRef<at::OptionalTensorRef> 封装非封箱可选张量向量
  at::ArrayRef<at::OptionalTensorRef> unboxed(vec);
  // 创建 IOptTensorListRef 对象
  at::IOptTensorListRef list(unboxed);
  size_t i = 0;
  // 遍历 IOptTensorListRef 对象
  for (const auto t : list) {
    // 验证 unboxed 的第 i 个元素是否与 t 的可选性一致
    EXPECT_EQ(unboxed[i].has_value(), t.has_value());
    if (t.has_value()) {
      // 验证 unboxed 的第 i 个元素与 t 的值是否相同
      EXPECT_TRUE((*unboxed[i]).is_same(*t));
    }
    i++;
  }
  EXPECT_EQ(i, list.size());
}
```