# `.\pytorch\test\cpp\api\tensor_indexing.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <torch/torch.h>  // 引入 PyTorch 的头文件

#include <test/cpp/api/support.h>  // 引入测试支持函数的头文件

using namespace torch::indexing;  // 使用 torch::indexing 命名空间
using namespace torch::test;  // 使用 torch::test 命名空间

TEST(TensorIndexingTest, Slice) {
  Slice slice(1, 2, 3);  // 创建一个切片对象 slice，起始为1，结束为2，步长为3
  ASSERT_EQ(slice.start(), 1);  // 断言切片的起始索引为1
  ASSERT_EQ(slice.stop(), 2);   // 断言切片的结束索引为2
  ASSERT_EQ(slice.step(), 3);   // 断言切片的步长为3

  ASSERT_EQ(c10::str(slice), "1:2:3");  // 断言切片对象的字符串表示为 "1:2:3"
}

TEST(TensorIndexingTest, TensorIndex) {
  {
    std::vector<TensorIndex> indices = {  // 创建存储 TensorIndex 的向量
        None,                             // 空索引对象
        "...",                            // 省略号索引对象
        Ellipsis,                         // 省略号索引对象
        0,                                // 整数索引对象
        true,                             // 布尔索引对象
        Slice(1, None, 2),                // 切片索引对象，起始为1，结束为无限大，步长为2
        torch::tensor({1, 2})};           // 张量索引对象，包含元素 1 和 2
    ASSERT_TRUE(indices[0].is_none());  // 断言第一个索引是空索引
    ASSERT_TRUE(indices[1].is_ellipsis());  // 断言第二个索引是省略号索引
    ASSERT_TRUE(indices[2].is_ellipsis());  // 断言第三个索引是省略号索引
    ASSERT_TRUE(indices[3].is_integer());   // 断言第四个索引是整数索引
    ASSERT_TRUE(indices[3].integer() == 0); // 断言第四个索引的整数值为0
    ASSERT_TRUE(indices[4].is_boolean());   // 断言第五个索引是布尔索引
    ASSERT_TRUE(indices[4].boolean() == true);  // 断言第五个索引的布尔值为 true
    ASSERT_TRUE(indices[5].is_slice());  // 断言第六个索引是切片索引
    ASSERT_TRUE(indices[5].slice().start() == 1);  // 断言切片的起始索引为1
    ASSERT_TRUE(indices[5].slice().stop() == INDEX_MAX);  // 断言切片的结束索引为无限大
    ASSERT_TRUE(indices[5].slice().step() == 2);  // 断言切片的步长为2
    ASSERT_TRUE(indices[6].is_tensor());  // 断言第七个索引是张量索引
    ASSERT_TRUE(torch::equal(indices[6].tensor(), torch::tensor({1, 2})));  // 断言张量索引对象包含元素 1 和 2
  }

  ASSERT_THROWS_WITH(
      TensorIndex(".."),  // 抛出异常，因为期望省略号索引，但传入的是 ".."
      "Expected \"...\" to represent an ellipsis index, but got \"..\"");

  {
    std::vector<TensorIndex> indices = {
        None, "...", Ellipsis, 0, true, Slice(1, None, 2)};  // 创建存储 TensorIndex 的向量
    ASSERT_EQ(
        c10::str(indices),  // 断言向量的字符串表示
        c10::str("(None, ..., ..., 0, true, 1:", INDEX_MAX, ":2)"));  // 预期的字符串表示
    ASSERT_EQ(c10::str(indices[0]), "None");  // 断言第一个索引的字符串表示为 "None"
    ASSERT_EQ(c10::str(indices[1]), "...");   // 断言第二个索引的字符串表示为 "..."
    ASSERT_EQ(c10::str(indices[2]), "...");   // 断言第三个索引的字符串表示为 "..."
    ASSERT_EQ(c10::str(indices[3]), "0");     // 断言第四个索引的字符串表示为 "0"
    ASSERT_EQ(c10::str(indices[4]), "true");  // 断言第五个索引的字符串表示为 "true"
    ASSERT_EQ(c10::str(indices[5]), c10::str("1:", INDEX_MAX, ":2"));


// 断言：验证第6个索引处的字符串表示与期望字符串相等
ASSERT_EQ(
    c10::str(indices[5]),
    c10::str("1:", INDEX_MAX, ":2"));



  }


// 函数体结束
}



  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice()})),
      c10::str("(0:", INDEX_MAX, ":1)"));


// 断言：验证包含一个空切片的 TensorIndex 向量的字符串表示与期望字符串相等
ASSERT_EQ(
    c10::str(std::vector<TensorIndex>({Slice()})),
    c10::str("(0:", INDEX_MAX, ":1)"));



  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(None, None)})),
      c10::str("(0:", INDEX_MAX, ":1)"));


// 断言：验证包含一个包含 None 的切片的 TensorIndex 向量的字符串表示与期望字符串相等
ASSERT_EQ(
    c10::str(std::vector<TensorIndex>({Slice(None, None)})),
    c10::str("(0:", INDEX_MAX, ":1)"));



  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(None, None, None)})),
      c10::str("(0:", INDEX_MAX, ":1)"));


// 断言：验证包含一个全局切片的 TensorIndex 向量的字符串表示与期望字符串相等
ASSERT_EQ(
    c10::str(std::vector<TensorIndex>({Slice(None, None, None)})),
    c10::str("(0:", INDEX_MAX, ":1)"));



  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(1, None)})),
      c10::str("(1:", INDEX_MAX, ":1)"));


// 断言：验证包含一个从索引1开始的切片的 TensorIndex 向量的字符串表示与期望字符串相等
ASSERT_EQ(
    c10::str(std::vector<TensorIndex>({Slice(1, None)})),
    c10::str("(1:", INDEX_MAX, ":1)"));



  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(1, None, None)})),
      c10::str("(1:", INDEX_MAX, ":1)"));


// 断言：验证包含一个从索引1开始、步长为1的切片的 TensorIndex 向量的字符串表示与期望字符串相等
ASSERT_EQ(
    c10::str(std::vector<TensorIndex>({Slice(1, None, None)})),
    c10::str("(1:", INDEX_MAX, ":1)"));



  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(None, 3)})),
      c10::str("(0:3:1)"));


// 断言：验证包含一个到索引3的切片的 TensorIndex 向量的字符串表示与期望字符串相等
ASSERT_EQ(
    c10::str(std::vector<TensorIndex>({Slice(None, 3)})),
    c10::str("(0:3:1)"));



  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(None, 3, None)})),
      c10::str("(0:3:1)"));


// 断言：验证包含一个到索引3、步长为1的切片的 TensorIndex 向量的字符串表示与期望字符串相等
ASSERT_EQ(
    c10::str(std::vector<TensorIndex>({Slice(None, 3, None)})),
    c10::str("(0:3:1)"));



  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(None, None, 2)})),
      c10::str("(0:", INDEX_MAX, ":2)"));


// 断言：验证包含一个全局切片、步长为2的 TensorIndex 向量的字符串表示与期望字符串相等
ASSERT_EQ(
    c10::str(std::vector<TensorIndex>({Slice(None, None, 2)})),
    c10::str("(0:", INDEX_MAX, ":2)"));



  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(None, None, -1)})),
      c10::str("(", INDEX_MAX, ":", INDEX_MIN, ":-1)"));


// 断言：验证包含一个全局切片、逆向步长为1的 TensorIndex 向量的字符串表示与期望字符串相等
ASSERT_EQ(
    c10::str(std::vector<TensorIndex>({Slice(None, None, -1)})),
    c10::str("(", INDEX_MAX, ":", INDEX_MIN, ":-1)"));



  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(1, 3)})), c10::str("(1:3:1)"));


// 断言：验证包含一个从索引1到索引3的切片的 TensorIndex 向量的字符串表示与期望字符串相等
ASSERT_EQ(
    c10::str(std::vector<TensorIndex>({Slice(1, 3)})),
    c10::str("(1:3:1)"));



  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(1, None, 2)})),
      c10::str("(1:", INDEX_MAX, ":2)"));


// 断言：验证包含一个从索引1开始、步长为2的切片的 TensorIndex 向量的字符串表示与期望字符串相等
ASSERT_EQ(
    c10::str(std::vector<TensorIndex>({Slice(1, None, 2)})),
    c10::str("(1:", INDEX_MAX, ":2)"));



  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(1, None, -1)})),
      c10::str("(1:", INDEX_MIN, ":-1)"));


// 断言：验证包含一个从索引1开始、逆向步长为1的切片的 TensorIndex 向量的字符串表示与期望字符串相等
ASSERT_EQ(
    c10::str(std::vector<TensorIndex>({Slice(1, None, -1)})),
    c10::str("(1:", INDEX_MIN, ":-1)"));



  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(None, 3, 2)})),
      c10::str("(0:3:2)"));


// 断言：验证包含一个到索引3、步长为2的切片的 TensorIndex 向量的字符串表示与期望字符串相等
ASSERT_EQ(
    c10::str(std::vector<TensorIndex>({Slice(None, 3, 2)})),
    c10::str("(0:3:2)"));



  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(None, 3, -1)})),
      c10::str("(", INDEX_MAX, ":3:-1)"));


// 断言：验证包含一个到索引3、逆向步长为1的切片的 TensorIndex 向量的字符串表示与期望字符串相等
ASSERT_EQ(
    c10::str(std::vector<TensorIndex>({Slice(None, 3, -1)})),
    c10::str("(", INDEX_MAX, ":3:-1)"));



  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(1, 3, 2)})),
      c10::str("(1:3:2)"));


// 断言：验证包含一个从索引1到索引3、步长为2的切片的 TensorIndex 向量的字符串表示与期望字符串相等
ASSERT_EQ(
    c10::str(std::vector<TensorIndex>({Slice(1, 3, 2)})),
    c10::str("(1:3:2)"));
}

// 定义测试用例 `TensorIndexingTest` 下的子测试 `TestNoIndices`
TEST(TensorIndexingTest, TestNoIndices) {
  // 创建一个大小为 (20, 20) 的随机张量 `tensor`
  torch::Tensor tensor = torch::randn({20, 20});
  // 创建一个大小为 (20, 20) 的随机张量 `value`
  torch::Tensor value = torch::randn({20, 20});
  // 创建一个空的 `indices` 向量
  std::vector<TensorIndex> indices;

  // 断言：对空索引列表调用 `tensor.index({})` 应抛出异常
  ASSERT_THROWS_WITH(
      tensor.index({}),
      "Passing an empty index list to Tensor::index() is not valid syntax");
  // 断言：对空索引列表调用 `tensor.index_put_({})` 应抛出异常
  ASSERT_THROWS_WITH(
      tensor.index_put_({}, 1),
      "Passing an empty index list to Tensor::index_put_() is not valid syntax");
  // 断言：对空索引列表调用 `tensor.index_put_({})` 应抛出异常
  ASSERT_THROWS_WITH(
      tensor.index_put_({}, value),
      "Passing an empty index list to Tensor::index_put_() is not valid syntax");

  // 断言：对空 `indices` 调用 `tensor.index(indices)` 应抛出异常
  ASSERT_THROWS_WITH(
      tensor.index(indices),
      "Passing an empty index list to Tensor::index() is not valid syntax");
  // 断言：对空 `indices` 调用 `tensor.index_put_(indices, 1)` 应抛出异常
  ASSERT_THROWS_WITH(
      tensor.index_put_(indices, 1),
      "Passing an empty index list to Tensor::index_put_() is not valid syntax");
  // 断言：对空 `indices` 调用 `tensor.index_put_(indices, value)` 应抛出异常
  ASSERT_THROWS_WITH(
      tensor.index_put_(indices, value),
      "Passing an empty index list to Tensor::index_put_() is not valid syntax");
}

// 定义测试用例 `TensorIndexingTest` 下的子测试 `TestAdvancedIndexingWithListOfTensor`
TEST(TensorIndexingTest, TestAdvancedIndexingWithListOfTensor) {
  {
    // 创建一个大小为 (20, 20) 的随机张量 `tensor`
    torch::Tensor tensor = torch::randn({20, 20});
    // 创建一个包含从 0 到 9 的长整型张量 `index`
    torch::Tensor index = torch::arange(10, torch::kLong).cpu();
    // 使用 `at::index` 进行高级索引，返回结果 `result`
    torch::Tensor result = at::index(tensor, {index});
    // 使用 `tensor.index({index})` 进行高级索引，返回结果 `result_with_init_list`
    torch::Tensor result_with_init_list = tensor.index({index});
    // 断言：两种高级索引方法返回的结果应相等
    ASSERT_TRUE(result.equal(result_with_init_list));
  }
  {
    // 创建一个大小为 (20, 20) 的随机张量 `tensor`
    torch::Tensor tensor = torch::randn({20, 20});
    // 创建一个包含从 0 到 9 的长整型张量 `index`
    torch::Tensor index = torch::arange(10, torch::kLong).cpu();
    // 使用 `at::index_put_` 进行高级索引赋值，返回结果 `result`
    torch::Tensor result = at::index_put_(tensor, {index}, torch::ones({20}));
    // 使用 `tensor.index_put_({index}, torch::ones({20}))` 进行高级索引赋值，返回结果 `result_with_init_list`
    torch::Tensor result_with_init_list =
        tensor.index_put_({index}, torch::ones({20}));
    // 断言：两种高级索引赋值方法返回的结果应相等
    ASSERT_TRUE(result.equal(result_with_init_list));
  }
  {
    // 创建一个大小为 (20, 20) 的随机张量 `tensor`
    torch::Tensor tensor = torch::randn({20, 20});
    // 创建一个包含从 0 到 9 的长整型张量 `index`
    torch::Tensor index = torch::arange(10, torch::kLong).cpu();
    // 使用 `at::index_put_` 进行高级索引赋值，返回结果 `result`
    torch::Tensor result =
        at::index_put_(tensor, {index}, torch::ones({1, 20}));
    // 使用 `tensor.index_put_({index}, torch::ones({1, 20}))` 进行高级索引赋值，返回结果 `result_with_init_list`
    torch::Tensor result_with_init_list =
        tensor.index_put_({index}, torch::ones({1, 20}));
    // 断言：两种高级索引赋值方法返回的结果应相等
    ASSERT_TRUE(result.equal(result_with_init_list));
  }
}

// 定义测试用例 `TensorIndexingTest` 下的子测试 `TestSingleInt`
TEST(TensorIndexingTest, TestSingleInt) {
  // 创建一个大小为 (5, 7, 3) 的随机张量 `v`
  auto v = torch::randn({5, 7, 3});
  // 断言：对 `v` 使用单个整数索引 `{4}` 后，结果的尺寸应为 (7, 3)
  ASSERT_EQ(v.index({4}).sizes(), torch::IntArrayRef({7, 3}));
}

// 定义测试用例 `TensorIndexingTest` 下的子测试 `TestMultipleInt`
TEST(TensorIndexingTest, TestMultipleInt) {
  // 创建一个大小为 (5, 7, 3) 的随机张量 `v`
  auto v = torch::randn({5, 7, 3});
  // 断言：对 `v` 使用单个整数索引 `{4}` 后，结果的尺寸应为 (7, 3)
  ASSERT_EQ(v.index({4}).sizes(), torch::IntArrayRef({7, 3}));
  // 断言：对 `v` 使用多个索引 `{4, Slice(), 1}` 后，结果的尺寸应为 (7)
  ASSERT_EQ(v.index({4, Slice(), 1}).sizes(), torch::IntArrayRef({7}));

  // 展示 `.index_put_` 的使用
  // 对 `v` 使用索引 `{4, 3, 1}` 进行赋值为 0
  v.index_put_({4, 3, 1}, 0);
  // 断言：对 `v` 使用索引 `{4, 3, 1}` 后的元素应为 0
  ASSERT_EQ(v.index({4, 3, 1}).item<double>(), 0);
}

// 定义测试用例 `TensorIndexingTest` 下的子测试 `TestNone`
TEST(TensorIndexingTest, TestNone) {
  // 创建一个大小为 (5, 7, 3) 的随机张量 `v`
  auto v = torch::randn({5, 7, 3});
  // 断言：对 `v` 使用索引 `{None}` 后，结果的尺寸应为 (1, 5, 7, 3)
  ASSERT_EQ(v.index({None}).sizes(), torch::IntArrayRef({1, 5, 7, 3}));
  // 断言：对 `v` 使用索引 `{Slice(), None}` 后，结果的尺寸应为 (5, 1, 7, 3)
  ASSERT_EQ(v.index({Slice(), None}).sizes(), torch::IntArrayRef({5, 1, 7, 3}));
  // 断言：对 `v` 使用索引 `{Slice(), None, None}` 后，结果的尺寸应为 (5, 1, 1, 7, 3)
  ASSERT_EQ(
      v.index({Slice(), None, None}).sizes(),
      torch::IntArrayRef({5, 1, 1, 7, 3}));
  // 断言：对 `v` 使用索引 `{"...", None}` 后，结果的尺寸应为 (5, 7, 3, 1)
  ASSERT_EQ(v.index({"...", None}).sizes(), torch::IntArrayRef({5, 7, 3, 1}));
}
TEST(TensorIndexingTest, TestStep) {
  // 创建一个张量 v，包含从 0 到 9 的连续整数
  auto v = torch::arange(10);
  // 使用索引切片选择所有元素，步长为 1，应该与原始张量相等
  assert_tensor_equal(v.index({Slice(None, None, 1)}), v);
  // 使用索引切片选择所有元素，步长为 2，应该得到 {0, 2, 4, 6, 8} 的张量
  assert_tensor_equal(
      v.index({Slice(None, None, 2)}), torch::tensor({0, 2, 4, 6, 8}));
  // 使用索引切片选择所有元素，步长为 3，应该得到 {0, 3, 6, 9} 的张量
  assert_tensor_equal(
      v.index({Slice(None, None, 3)}), torch::tensor({0, 3, 6, 9}));
  // 使用索引切片选择所有元素，步长为 11，应该只返回第一个元素 {0}
  assert_tensor_equal(v.index({Slice(None, None, 11)}), torch::tensor({0}));
  // 使用索引切片选择从索引 1 到 6 的元素，步长为 2，应该得到 {1, 3, 5} 的张量
  assert_tensor_equal(v.index({Slice(1, 6, 2)}), torch::tensor({1, 3, 5}));
}

TEST(TensorIndexingTest, TestStepAssignment) {
  // 创建一个 4x4 的零张量 v
  auto v = torch::zeros({4, 4});
  // 使用索引放置操作，对 v 的第一行的奇数列（从第 1 列开始，步长为 2）赋值为 {3., 4.}
  v.index_put_({0, Slice(1, None, 2)}, torch::tensor({3., 4.}));
  // 检查第一行的内容，应该为 {0., 3., 0., 4.}
  assert_tensor_equal(v.index({0}), torch::tensor({0., 3., 0., 4.}));
  // 检查从第二行到最后一行的内容的和，应该为零
  assert_tensor_equal(v.index({Slice(1, None)}).sum(), torch::tensor(0));
}

TEST(TensorIndexingTest, TestBoolIndices) {
  {
    // 创建一个大小为 5x7x3 的随机张量 v
    auto v = torch::randn({5, 7, 3});
    // 创建布尔索引张量，选择索引为 {0, 2, 3} 的轴 0 的内容
    auto boolIndices =
        torch::tensor({true, false, true, true, false}, torch::kBool);
    // 检查选择后的张量形状，应为 {3, 7, 3}
    ASSERT_EQ(v.index({boolIndices}).sizes(), torch::IntArrayRef({3, 7, 3}));
    // 检查选择后的内容，应该与索引 {0, 2, 3} 对应的内容堆叠得到的结果相等
    assert_tensor_equal(
        v.index({boolIndices}),
        torch::stack({v.index({0}), v.index({2}), v.index({3})}));
  }
  {
    // 创建一个大小为 3 的布尔张量 v
    auto v = torch::tensor({true, false, true}, torch::kBool);
    // 创建布尔索引张量和 uint8 索引张量
    auto boolIndices = torch::tensor({true, false, false}, torch::kBool);
    auto uint8Indices = torch::tensor({1, 0, 0}, torch::kUInt8);

    {
      // 捕获警告信息
      WarningCapture warnings;

      // 检查两种索引方式的结果张量形状应该相同
      ASSERT_EQ(
          v.index({boolIndices}).sizes(), v.index({uint8Indices}).sizes());
      // 检查两种索引方式的结果张量内容应该相同，都为 {true}
      assert_tensor_equal(
          v.index({boolIndices}), torch::tensor({true}, torch::kBool));

      // 检查警告信息中 torch.uint8 索引使用已经不推荐的次数
      ASSERT_EQ(
          count_substr_occurrences(
              warnings.str(),
              "indexing with dtype torch.uint8 is now deprecated"),
          2);
    }
  }
}

TEST(TensorIndexingTest, TestBoolIndicesAccumulate) {
  // 创建一个大小为 10 的布尔张量 mask 和一个大小为 10x10 的张量 y，全为 1
  auto mask = torch::zeros({10}, torch::kBool);
  auto y = torch::ones({10, 10});
  // 使用布尔索引 mask 对 y 进行索引放置操作，累加模式，将对应位置的 y 的值加到 y 上
  y.index_put_({mask}, {y.index({mask})}, /*accumulate=*/true);
  // 检查结果张量 y 应该全为 1
  assert_tensor_equal(y, torch::ones({10, 10}));
}

TEST(TensorIndexingTest, TestMultipleBoolIndices) {
  // 创建一个大小为 5x7x3 的随机张量 v
  auto v = torch::randn({5, 7, 3});
  // 创建两个布尔索引张量 mask1 和 mask2
  // 注意：这些会广播在一起，并且在第一维上进行转置
  auto mask1 = torch::tensor({1, 0, 1, 1, 0}, torch::kBool);
  auto mask2 = torch::tensor({1, 1, 1}, torch::kBool);
  // 检查选择后的张量形状，应为 {3, 7}
  ASSERT_EQ(
      v.index({mask1, Slice(), mask2}).sizes(), torch::IntArrayRef({3, 7}));
}

TEST(TensorIndexingTest, TestByteMask) {
  {
    // 创建一个大小为 5x7x3 的随机张量 v
    auto v = torch::randn({5, 7, 3});
    // 创建一个字节（uint8）索引张量 mask
    auto mask = torch::tensor({1, 0, 1, 1, 0}, torch::kByte);
    {
      // 捕获警告信息
      WarningCapture warnings;

      // 检查使用字节索引 mask 后的张量形状，应为 {3, 7, 3}
      ASSERT_EQ(v.index({mask}).sizes(), torch::IntArrayRef({3, 7, 3}));
      // 检查选择后的内容，应该与索引 {0, 2, 3} 对应的内容堆叠得到的结果相等
      assert_tensor_equal(v.index({mask}), torch::stack({v[0], v[2], v[3]}));

      // 检查警告信息中 torch.uint8 索引使用已经不推荐的次数
      ASSERT_EQ(
          count_substr_occurrences(
              warnings.str(),
              "indexing with dtype torch.uint8 is now deprecated"),
          2);
    }
  }
  {
    # 创建一个包含单个浮点数 1. 的 Torch 张量
    auto v = torch::tensor({1.});
    # 使用张量的索引功能，选择所有值为 0 的元素，然后断言其应与一个空的随机张量相等
    assert_tensor_equal(v.index({v == 0}), torch::randn({0}));
}

// 定义一个名为 `TestByteMaskAccumulate` 的测试用例，测试字节掩码累加操作
TEST(TensorIndexingTest, TestByteMaskAccumulate) {
  // 创建一个大小为 10 的零张量 `mask`，数据类型为 `torch::kUInt8`
  auto mask = torch::zeros({10}, torch::kUInt8);
  // 创建一个大小为 (10, 10) 的全一张量 `y`
  auto y = torch::ones({10, 10});
  {
    // 捕获警告信息的辅助对象
    WarningCapture warnings;

    // 使用字节掩码进行索引操作，累加原张量的部分数据
    y.index_put_({mask}, y.index({mask}), /*accumulate=*/true);
    // 断言张量 `y` 仍然全为 1
    assert_tensor_equal(y, torch::ones({10, 10}));

    // 断言警告信息中关于使用 `torch.uint8` 类型索引已废弃的次数为 2 次
    ASSERT_EQ(
        count_substr_occurrences(
            warnings.str(),
            "indexing with dtype torch.uint8 is now deprecated"),
        2);
  }
}

// 定义一个名为 `TestMultipleByteMask` 的测试用例，测试多个字节掩码索引操作
TEST(TensorIndexingTest, TestMultipleByteMask) {
  // 创建一个随机张量 `v`，大小为 (5, 7, 3)
  auto v = torch::randn({5, 7, 3});
  // 创建一个字节类型的张量 `mask1`，并赋初值
  auto mask1 = torch::tensor({1, 0, 1, 1, 0}, torch::kByte);
  // 创建另一个字节类型的张量 `mask2`，并赋初值
  auto mask2 = torch::tensor({1, 1, 1}, torch::kByte);
  {
    // 捕获警告信息的辅助对象
    WarningCapture warnings;

    // 断言通过字节掩码 `mask1` 和 `mask2` 索引后的张量大小为 (3, 7)
    ASSERT_EQ(
        v.index({mask1, Slice(), mask2}).sizes(), torch::IntArrayRef({3, 7}));

    // 断言警告信息中关于使用 `torch.uint8` 类型索引已废弃的次数为 2 次
    ASSERT_EQ(
        count_substr_occurrences(
            warnings.str(),
            "indexing with dtype torch.uint8 is now deprecated"),
        2);
  }
}

// 定义一个名为 `TestByteMask2d` 的测试用例，测试二维字节掩码索引操作
TEST(TensorIndexingTest, TestByteMask2d) {
  // 创建一个随机张量 `v`，大小为 (5, 7, 3)
  auto v = torch::randn({5, 7, 3});
  // 创建一个随机张量 `c`，大小为 (5, 7)
  auto c = torch::randn({5, 7});
  // 统计 `c` 大于 0 的元素个数，并转换为 `int64_t` 类型
  int64_t num_ones = (c > 0).sum().item().to<int64_t>();
  // 根据 `c` 大于 0 的条件进行索引操作，结果张量大小为 (num_ones, 3)
  auto r = v.index({c > 0});
  // 断言结果张量 `r` 的大小为 (num_ones, 3)
  ASSERT_EQ(r.sizes(), torch::IntArrayRef({num_ones, 3}));
}

// 定义一个名为 `TestIntIndices` 的测试用例，测试整数索引操作
TEST(TensorIndexingTest, TestIntIndices) {
  // 创建一个随机张量 `v`，大小为 (5, 7, 3)
  auto v = torch::randn({5, 7, 3});
  // 断言通过整数索引 `{0, 4, 2}` 后的张量大小为 (3, 7, 3)
  ASSERT_EQ(
      v.index({torch::tensor({0, 4, 2})}).sizes(),
      torch::IntArrayRef({3, 7, 3}));
  // 断言通过切片索引和整数索引 `{0, 4, 2}` 后的张量大小为 (5, 3, 3)
  ASSERT_EQ(
      v.index({Slice(), torch::tensor({0, 4, 2})}).sizes(),
      torch::IntArrayRef({5, 3, 3}));
  // 断言通过切片索引和整数索引 `{{0, 1}, {4, 3}}` 后的张量大小为 (5, 2, 2, 3)
  ASSERT_EQ(
      v.index({Slice(), torch::tensor({{0, 1}, {4, 3}})}).sizes(),
      torch::IntArrayRef({5, 2, 2, 3}));
}

// 定义一个名为 `TestIntIndices2d` 的测试用例，测试二维整数索引操作
TEST(TensorIndexingTest, TestIntIndices2d) {
  // 创建一个张量 `x`，根据 NumPy 的索引示例进行初始化
  auto x = torch::arange(0, 12, torch::kLong).view({4, 3});
  // 创建行索引和列索引张量 `rows` 和 `columns`
  auto rows = torch::tensor({{0, 0}, {3, 3}});
  auto columns = torch::tensor({{0, 2}, {0, 2}});
  // 断言通过二维整数索引后的张量大小为 (2, 2)
  assert_tensor_equal(
      x.index({rows, columns}), torch::tensor({{0, 2}, {9, 11}}));
}

// 定义一个名为 `TestIntIndicesBroadcast` 的测试用例，测试整数索引广播操作
TEST(TensorIndexingTest, TestIntIndicesBroadcast) {
  // 创建一个张量 `x`，根据 NumPy 的索引示例进行初始化
  auto x = torch::arange(0, 12, torch::kLong).view({4, 3});
  // 创建行索引和列索引张量 `rows` 和 `columns`
  auto rows = torch::tensor({0, 3});
  auto columns = torch::tensor({0, 2});
  // 进行索引操作并断言结果张量的值与预期一致
  auto result = x.index({rows.index({Slice(), None}), columns});
  assert_tensor_equal(result, torch::tensor({{0, 2}, {9, 11}}));
}

// 定义一个名为 `TestEmptyIndex` 的测试用例，测试空索引操作
TEST(TensorIndexingTest, TestEmptyIndex) {
  // 创建一个张量 `x`，大小为 (4, 3)
  auto x = torch::arange(0, 12).view({4, 3});
  // 创建一个空的长整型索引张量 `idx`
  auto idx = torch::tensor({}, torch::kLong);
  // 断言空索引后张量的元素个数为 0
  ASSERT_EQ(x.index({idx}).numel(), 0);

  // 创建张量 `y`，作为 `x` 的克隆
  auto y = x.clone();
  // 使用空索引对张量 `y` 进行赋值操作，预期没有影响
  y.index_put_({idx}, -1);
  // 断言经过空索引赋值后 `x` 和 `y` 的值相等
  assert_tensor_equal(x, y);

  // 创建一个布尔类型的全零掩码张量 `mask`
  auto mask = torch::zeros({4, 3}, torch::kBool);
  // 使用空掩码进行索引操作，预期没有影响
  y.index_put_({mask}, -1);
  // 再次断言经过空掩码赋值后 `x` 和 `y` 的值相等
  assert_tensor_equal(x, y);
}

// 定义一个名为 `TestEmptyNdimIndex` 的测试用例，测试空的多维索引操作
TEST(TensorIndexingTest, TestEmptyNdimIndex) {
  // 创建一个 CPU 设备对象
  torch::Device device(torch::kCPU);
  {
    // 在指定设备上创建一个大小为 (5
    {
        // 创建一个空的张量，形状为 {0, 2}，在指定的设备上
        assert_tensor_equal(
            torch::empty({0, 2}, device),
            x.index({torch::empty(
                {0, 2}, torch::TensorOptions(torch::kInt64).device(device))}));
      }
      {
        // 创建一个形状为 {2, 3, 4, 5} 的随机张量 x，并在指定设备上执行索引操作
        auto x = torch::randn({2, 3, 4, 5}, device);
        // 断言操作，比较指定形状 {2, 0, 6, 4, 5} 的空张量与 x 的索引结果是否相等
        assert_tensor_equal(
            torch::empty({2, 0, 6, 4, 5}, device),
            x.index(
                {Slice(),
                 torch::empty(
                     {0, 6}, torch::TensorOptions(torch::kInt64).device(device))}));
      }
      {
        // 创建一个形状为 {10, 0} 的空张量 x
        auto x = torch::empty({10, 0});
        // 断言操作，验证索引 {torch::tensor({1, 2})} 后的张量形状是否为 {2, 0}
        ASSERT_EQ(
            x.index({torch::tensor({1, 2})}).sizes(), torch::IntArrayRef({2, 0}));
        // 断言操作，验证索引 {torch::tensor({}, torch::kLong), torch::tensor({}, torch::kLong)} 后的张量形状是否为 {0}
        ASSERT_EQ(
            x.index(
                 {torch::tensor({}, torch::kLong), torch::tensor({}, torch::kLong)})
                .sizes(),
            torch::IntArrayRef({0}));
        // 预期索引操作抛出异常，报告维度尺寸为 0 的错误
        ASSERT_THROWS_WITH(
            x.index({Slice(), torch::tensor({0, 1})}), "for dimension with size 0");
      }
TEST(TensorIndexingTest, TestEmptyNdimIndex_CUDA) {
  // 定义使用 CUDA 设备
  torch::Device device(torch::kCUDA);
  {
    // 创建一个形状为 {5} 的随机张量 x，使用 CUDA 设备
    auto x = torch::randn({5}, device);
    // 断言空张量与 x 使用空索引（形状为 {0, 2}）后的结果相等
    assert_tensor_equal(
        torch::empty({0, 2}, device),
        x.index({torch::empty(
            {0, 2}, torch::TensorOptions(torch::kInt64).device(device))}));
  }
  {
    // 创建一个形状为 {2, 3, 4, 5} 的随机张量 x，使用 CUDA 设备
    auto x = torch::randn({2, 3, 4, 5}, device);
    // 断言空张量与 x 使用空索引（第一维全取，第二维取 {0, 6}）后的结果相等
    assert_tensor_equal(
        torch::empty({2, 0, 6, 4, 5}, device),
        x.index(
            {Slice(),
             torch::empty(
                 {0, 6}, torch::TensorOptions(torch::kInt64).device(device))}));
  }
}

TEST(TensorIndexingTest, TestEmptyNdimIndexBool) {
  // 定义使用 CPU 设备
  torch::Device device(torch::kCPU);
  // 创建一个形状为 {5} 的随机张量 x，使用 CPU 设备
  auto x = torch::randn({5}, device);
  // 使用空布尔索引（形状为 {0, 2}）时，断言抛出异常
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(
      x.index({torch::empty(
          {0, 2}, torch::TensorOptions(torch::kUInt8).device(device))}),
      c10::Error);
}

TEST(TensorIndexingTest, TestEmptyNdimIndexBool_CUDA) {
  // 定义使用 CUDA 设备
  torch::Device device(torch::kCUDA);
  // 创建一个形状为 {5} 的随机张量 x，使用 CUDA 设备
  auto x = torch::randn({5}, device);
  // 使用空布尔索引（形状为 {0, 2}）时，断言抛出异常
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(
      x.index({torch::empty(
          {0, 2}, torch::TensorOptions(torch::kUInt8).device(device))}),
      c10::Error);
}

TEST(TensorIndexingTest, TestEmptySlice) {
  // 定义使用 CPU 设备
  torch::Device device(torch::kCPU);
  // 创建一个形状为 {2, 3, 4, 5} 的随机张量 x，使用 CPU 设备
  auto x = torch::randn({2, 3, 4, 5}, device);
  // 使用切片索引，获取 y 张量
  auto y = x.index({Slice(), Slice(), Slice(), 1});
  // 使用切片索引，获取 z 张量
  auto z = y.index({Slice(), Slice(1, 1), Slice()});
  // 断言 z 张量的形状为 {2, 0, 4}
  ASSERT_EQ(z.sizes(), torch::IntArrayRef({2, 0, 4}));
  // 检查 z 张量的步幅与 NumPy 计算匹配，虽然这在技术上并非必需
  ASSERT_EQ(z.strides(), torch::IntArrayRef({60, 20, 5}));
  // 断言 z 张量是连续的
  ASSERT_TRUE(z.is_contiguous());
}

TEST(TensorIndexingTest, TestEmptySlice_CUDA) {
  // 定义使用 CUDA 设备
  torch::Device device(torch::kCUDA);
  // 创建一个形状为 {2, 3, 4, 5} 的随机张量 x，使用 CUDA 设备
  auto x = torch::randn({2, 3, 4, 5}, device);
  // 使用切片索引，获取 y 张量
  auto y = x.index({Slice(), Slice(), Slice(), 1});
  // 使用切片索引，获取 z 张量
  auto z = y.index({Slice(), Slice(1, 1), Slice()});
  // 断言 z 张量的形状为 {2, 0, 4}
  ASSERT_EQ(z.sizes(), torch::IntArrayRef({2, 0, 4}));
  // 检查 z 张量的步幅与 NumPy 计算匹配，虽然这在技术上并非必需
  ASSERT_EQ(z.strides(), torch::IntArrayRef({60, 20, 5}));
  // 断言 z 张量是连续的
  ASSERT_TRUE(z.is_contiguous());
}

TEST(TensorIndexingTest, TestIndexGetitemCopyBoolsSlices) {
  // 创建一个值为 1 的 UInt8 张量 true_tensor
  auto true_tensor = torch::tensor(1, torch::kUInt8);
  // 创建一个值为 0 的 UInt8 张量 false_tensor
  auto false_tensor = torch::tensor(0, torch::kUInt8);

  // 创建张量数组 tensors 包含随机张量和张量 3
  std::vector<torch::Tensor> tensors = {torch::randn({2, 3}), torch::tensor(3)};

  // 遍历 tensors 数组
  for (auto& a : tensors) {
    // 断言 a 与 a.index({true}) 的数据指针不相等
    ASSERT_NE(a.data_ptr(), a.index({true}).data_ptr());
    {
      // 创建形状为 {0} 的 sizes 数组，插入 a 的形状
      std::vector<int64_t> sizes = {0};
      sizes.insert(sizes.end(), a.sizes().begin(), a.sizes().end());
      // 断言空张量与 a 使用 false 索引后的结果相等
      assert_tensor_equal(torch::empty(sizes), a.index({false}));
    }
    // 断言 a 与 a.index({true_tensor}) 的数据指针不相等
    ASSERT_NE(a.data_ptr(), a.index({true_tensor}).data_ptr());
    {
      // 创建形状为 {0} 的 sizes 数组，插入 a 的形状
      std::vector<int64_t> sizes = {0};
      sizes.insert(sizes.end(), a.sizes().begin(), a.sizes().end());
      // 断言空张量与 a 使用 false_tensor 索引后的结果相等
      assert_tensor_equal(torch::empty(sizes), a.index({false_tensor}));
    }
    # 断言：验证张量 a 的数据指针与使用空索引的结果的数据指针是否相同
    ASSERT_EQ(a.data_ptr(), a.index({None}).data_ptr());
    
    # 断言：验证张量 a 的数据指针与使用字符串 "..." 作为索引的结果的数据指针是否相同
    ASSERT_EQ(a.data_ptr(), a.index({"..."}).data_ptr());
TEST(TensorIndexingTest, TestIndexSetitemBoolsSlices) {
  // 创建一个值为 1 的无符号 8 位整数张量
  auto true_tensor = torch::tensor(1, torch::kUInt8);
  // 创建一个值为 0 的无符号 8 位整数张量
  auto false_tensor = torch::tensor(0, torch::kUInt8);

  // 创建包含两个张量的向量
  std::vector<torch::Tensor> tensors = {torch::randn({2, 3}), torch::tensor(3)};

  // 对于每个张量执行以下操作
  for (auto& a : tensors) {
    // 创建一个与张量 a 大小相同且每个元素为 -1 的张量
    auto neg_ones = torch::ones_like(a) * -1;
    // 在前面添加两个维度，用于与 numpy 兼容，因为一些操作已经在大小前面添加了一个 1
    auto neg_ones_expanded = neg_ones.unsqueeze(0).unsqueeze(0);

    // 使用真值索引集合 true 对张量 a 进行索引赋值操作，将 neg_ones_expanded 赋给对应位置
    a.index_put_({true}, neg_ones_expanded);
    // 断言张量 a 是否与 neg_ones 相等
    assert_tensor_equal(a, neg_ones);

    // 使用假值索引集合 false 对张量 a 进行索引赋值操作，将值 5 赋给对应位置
    a.index_put_({false}, 5);
    // 断言张量 a 是否与 neg_ones 相等
    assert_tensor_equal(a, neg_ones);

    // 使用 true_tensor 索引集合对张量 a 进行索引赋值操作，将 neg_ones_expanded * 2 赋给对应位置
    a.index_put_({true_tensor}, neg_ones_expanded * 2);
    // 断言张量 a 是否与 neg_ones * 2 相等
    assert_tensor_equal(a, neg_ones * 2);

    // 使用 false_tensor 索引集合对张量 a 进行索引赋值操作，将值 5 赋给对应位置
    a.index_put_({false_tensor}, 5);
    // 断言张量 a 是否与 neg_ones * 2 相等
    assert_tensor_equal(a, neg_ones * 2);

    // 使用 None 索引集合对张量 a 进行索引赋值操作，将 neg_ones_expanded * 3 赋给整个张量
    a.index_put_({None}, neg_ones_expanded * 3);
    // 断言张量 a 是否与 neg_ones * 3 相等
    assert_tensor_equal(a, neg_ones * 3);

    // 使用 "..." 索引集合对张量 a 进行索引赋值操作，将 neg_ones_expanded * 4 赋给整个张量
    a.index_put_({"..."}, neg_ones_expanded * 4);
    // 断言张量 a 是否与 neg_ones * 4 相等
    assert_tensor_equal(a, neg_ones * 4);

    // 如果张量 a 的维度为 0，则抛出异常
    if (a.dim() == 0) {
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
      ASSERT_THROW(a.index_put_({Slice()}, neg_ones_expanded * 5), c10::Error);
    }
  }
}

TEST(TensorIndexingTest, TestIndexScalarWithBoolMask) {
  // 定义设备为 CPU
  torch::Device device(torch::kCPU);

  // 创建一个值为 1 的张量，并指定设备
  auto a = torch::tensor(1, device);
  // 创建一个值为 true 的无符号 8 位整数张量，并指定设备
  auto uintMask =
      torch::tensor(true, torch::TensorOptions(torch::kUInt8).device(device));
  // 创建一个值为 true 的布尔型张量，并指定设备
  auto boolMask =
      torch::tensor(true, torch::TensorOptions(torch::kBool).device(device));
  // 断言通过 uintMask 和 boolMask 索引得到的张量相等
  assert_tensor_equal(a.index({uintMask}), a.index({boolMask}));
  // 断言通过 uintMask 和 boolMask 索引得到的张量的数据类型相等
  ASSERT_EQ(a.index({uintMask}).dtype(), a.index({boolMask}).dtype());

  // 将张量 a 重新赋值为一个布尔型张量，并指定设备
  a = torch::tensor(true, torch::TensorOptions(torch::kBool).device(device));
  // 断言通过 uintMask 和 boolMask 索引得到的张量相等
  assert_tensor_equal(a.index({uintMask}), a.index({boolMask}));
  // 断言通过 uintMask 和 boolMask 索引得到的张量的数据类型相等
  ASSERT_EQ(a.index({uintMask}).dtype(), a.index({boolMask}).dtype());
}

TEST(TensorIndexingTest, TestIndexScalarWithBoolMask_CUDA) {
  // 定义设备为 CUDA
  torch::Device device(torch::kCUDA);

  // 创建一个值为 1 的张量，并指定设备
  auto a = torch::tensor(1, device);
  // 创建一个值为 true 的无符号 8 位整数张量，并指定设备
  auto uintMask =
      torch::tensor(true, torch::TensorOptions(torch::kUInt8).device(device));
  // 创建一个值为 true 的布尔型张量，并指定设备
  auto boolMask =
      torch::tensor(true, torch::TensorOptions(torch::kBool).device(device));
  // 断言通过 uintMask 和 boolMask 索引得到的张量相等
  assert_tensor_equal(a.index({uintMask}), a.index({boolMask}));
  // 断言通过 uintMask 和 boolMask 索引得到的张量的数据类型相等
  ASSERT_EQ(a.index({uintMask}).dtype(), a.index({boolMask}).dtype());

  // 将张量 a 重新赋值为一个布尔型张量，并指定设备
  a = torch::tensor(true, torch::TensorOptions(torch::kBool).device(device));
  // 断言通过 uintMask 和 boolMask 索引得到的张量相等
  assert_tensor_equal(a.index({uintMask}), a.index({boolMask}));
  // 断言通过 uintMask 和 boolMask 索引得到的张量的数据类型相等
  ASSERT_EQ(a.index({uintMask}).dtype(), a.index({boolMask}).dtype());
}
TEST(TensorIndexingTest, TestSetitemExpansionError) {
  auto true_tensor = torch::tensor(true);  // 创建一个包含单个布尔值的张量 true_tensor
  auto a = torch::randn({2, 3});  // 创建一个大小为 2x3 的随机张量 a

  // 检查前缀不全为1的情况下，扩展操作无效
  std::vector<int64_t> tensor_sizes{5, 1};
  tensor_sizes.insert(tensor_sizes.end(), a.sizes().begin(), a.sizes().end());
  auto a_expanded = a.expand(tensor_sizes);  // 使用 tensor_sizes 扩展张量 a

  // 断言异常抛出：当使用非法索引 true 进行 index_put_ 操作时，预期会抛出 c10::Error 异常
  ASSERT_THROW(a.index_put_({true}, a_expanded), c10::Error);
  ASSERT_THROW(a.index_put_({true_tensor}, a_expanded), c10::Error);
}

TEST(TensorIndexingTest, TestGetitemScalars) {
  auto zero = torch::tensor(0, torch::kInt64);  // 创建一个值为0的64位整数张量 zero
  auto one = torch::tensor(1, torch::kInt64);  // 创建一个值为1的64位整数张量 one

  auto a = torch::randn({2, 3});  // 创建一个大小为2x3的随机张量 a

  // 断言：使用标量索引操作后的张量相等性检查
  assert_tensor_equal(a.index({0}), a.index({zero}));
  assert_tensor_equal(a.index({0}).index({1}), a.index({zero}).index({one}));
  assert_tensor_equal(a.index({0, 1}), a.index({zero, one}));
  assert_tensor_equal(a.index({0, one}), a.index({zero, 1}));

  // 断言：标量索引操作应该是切片而不是复制
  ASSERT_EQ(a.index({0, 1}).data_ptr(), a.index({zero, one}).data_ptr());
  ASSERT_EQ(a.index({1}).data_ptr(), a.index({one.to(torch::kInt)}).data_ptr());
  ASSERT_EQ(a.index({1}).data_ptr(), a.index({one.to(torch::kShort)}).data_ptr());

  // 创建一个标量张量 r
  auto r = torch::randn({});
  ASSERT_THROW(r.index({Slice()}), c10::Error);  // 断言：使用 Slice() 进行标量索引会抛出异常
  ASSERT_THROW(r.index({zero}), c10::Error);  // 断言：使用标量索引 zero 会抛出异常
  assert_tensor_equal(r, r.index({"..."}));  // 断言：使用 "..." 进行索引操作后张量 r 保持不变
}

TEST(TensorIndexingTest, TestSetitemScalars) {
  auto zero = torch::tensor(0, torch::kInt64);  // 创建一个值为0的64位整数张量 zero

  auto a = torch::randn({2, 3});  // 创建一个大小为2x3的随机张量 a
  auto a_set_with_number = a.clone();  // 创建 a 的副本 a_set_with_number
  auto a_set_with_scalar = a.clone();  // 创建 a 的副本 a_set_with_scalar
  auto b = torch::randn({3});  // 创建一个大小为3的随机张量 b

  a_set_with_number.index_put_({0}, b);  // 使用张量 b 替换 a_set_with_number 中的索引 {0} 处的值
  a_set_with_scalar.index_put_({zero}, b);  // 使用张量 b 替换 a_set_with_scalar 中的索引 {zero} 处的值
  assert_tensor_equal(a_set_with_number, a_set_with_scalar);  // 断言：a_set_with_number 和 a_set_with_scalar 应该相等

  a.index_put_({1, zero}, 7.7);  // 使用值 7.7 替换 a 中索引 {1, 0} 处的值

  // 断言：检查索引 {1, 0} 处的值是否接近于 7.7
  ASSERT_TRUE(a.index({1, 0}).allclose(torch::tensor(7.7)));

  auto r = torch::randn({});  // 创建一个标量张量 r

  // 断言：使用 Slice() 进行标量索引 put 操作会抛出异常
  ASSERT_THROW(r.index_put_({Slice()}, 8.8), c10::Error);
  // 断言：使用标量索引 zero 进行 put 操作会抛出异常
  ASSERT_THROW(r.index_put_({zero}, 8.8), c10::Error);

  r.index_put_({"..."}, 9.9);  // 使用 "..." 进行索引 put 操作，设置张量 r 的所有元素为 9.9

  // 断言：检查张量 r 的所有元素是否接近于 9.9
  ASSERT_TRUE(r.allclose(torch::tensor(9.9)));
}
TEST(TensorIndexingTest, TestBasicAdvancedCombined) {
  // 从 NumPy 索引示例中创建张量 x，包含元素 0 到 11，reshape 成 4x3 的张量
  auto x = torch::arange(0, 12).to(torch::kLong).view({4, 3});
  // 检查索引 {Slice(1, 2), Slice(1, 3)} 和 {Slice(1, 2), {1, 2}} 返回的张量是否相等
  assert_tensor_equal(
      x.index({Slice(1, 2), Slice(1, 3)}),
      x.index({Slice(1, 2), torch::tensor({1, 2})}));
  // 检查索引 {Slice(1, 2), Slice(1, 3)} 返回的张量是否等于 {{4, 5}}
  assert_tensor_equal(
      x.index({Slice(1, 2), Slice(1, 3)}), torch::tensor({{4, 5}}));

  // 检查索引操作是否生成副本
  {
    auto unmodified = x.clone();
    // 对索引 {Slice(1, 2), {1, 2}} 的视图进行零填充操作
    x.index({Slice(1, 2), torch::tensor({1, 2})}).zero_();
    // 断言修改后的张量 x 是否等于未修改的张量 unmodified
    assert_tensor_equal(x, unmodified);
  }

  // 但是赋值操作应当修改原始张量
  {
    auto unmodified = x.clone();
    // 对索引 {Slice(1, 2), {1, 2}} 的视图进行赋值为 0 的操作
    x.index_put_({Slice(1, 2), torch::tensor({1, 2})}, 0);
    // 断言修改后的张量 x 是否不等于未修改的张量 unmodified
    assert_tensor_not_equal(x, unmodified);
  }
}

TEST(TensorIndexingTest, TestIntAssignment) {
  {
    // 创建张量 x 包含元素 0 到 3，reshape 成 2x2 的张量
    auto x = torch::arange(0, 4).to(torch::kLong).view({2, 2});
    // 对索引 {1} 的位置进行赋值为 5
    x.index_put_({1}, 5);
    // 断言张量 x 是否等于 {{0, 1}, {5, 5}}
    assert_tensor_equal(x, torch::tensor({{0, 1}, {5, 5}}));
  }

  {
    // 创建张量 x 包含元素 0 到 3，reshape 成 2x2 的张量
    auto x = torch::arange(0, 4).to(torch::kLong).view({2, 2});
    // 对索引 {1} 的位置进行赋值为 5 和 6
    x.index_put_({1}, torch::arange(5, 7).to(torch::kLong));
    // 断言张量 x 是否等于 {{0, 1}, {5, 6}}
    assert_tensor_equal(x, torch::tensor({{0, 1}, {5, 6}}));
  }
}

TEST(TensorIndexingTest, TestByteTensorAssignment) {
  // 创建张量 x 包含元素 0 到 15，reshape 成 4x4 的张量
  auto x = torch::arange(0., 16).to(torch::kFloat).view({4, 4});
  // 创建布尔张量 b，用于索引操作
  auto b = torch::tensor({true, false, true, false}, torch::kByte);
  // 创建值张量 value，用于赋值操作
  auto value = torch::tensor({3., 4., 5., 6.});

  {
    // 捕获警告信息
    WarningCapture warnings;

    // 使用布尔张量 b 进行索引操作并赋值为 value
    x.index_put_({b}, value);

    // 断言警告信息中包含 "indexing with dtype torch.uint8 is now deprecated" 的次数为 1
    ASSERT_EQ(
        count_substr_occurrences(
            warnings.str(),
            "indexing with dtype torch.uint8 is now deprecated"),
        1);
  }

  // 断言索引 {0} 返回的张量是否等于 value
  assert_tensor_equal(x.index({0}), value);
  // 断言索引 {1} 返回的张量是否等于 {4, 5, 6, 7}
  assert_tensor_equal(x.index({1}), torch::arange(4, 8).to(torch::kLong));
  // 断言索引 {2} 返回的张量是否等于 value
  assert_tensor_equal(x.index({2}), value);
  // 断言索引 {3} 返回的张量是否等于 {12, 13, 14, 15}
  assert_tensor_equal(x.index({3}), torch::arange(12, 16).to(torch::kLong));
}

TEST(TensorIndexingTest, TestVariableSlicing) {
  // 创建张量 x 包含元素 0 到 15，reshape 成 4x4 的张量
  auto x = torch::arange(0, 16).view({4, 4});
  // 创建索引张量 indices 包含 {0, 1}
  auto indices = torch::tensor({0, 1}, torch::kInt);
  // 获取 indices 的第一个和第二个元素作为整数 i 和 j
  int i = indices[0].item<int>();
  int j = indices[1].item<int>();
  // 断言索引 {Slice(i, j)} 返回的张量是否等于索引 {Slice(0, 1)} 返回的张量
  assert_tensor_equal(x.index({Slice(i, j)}), x.index({Slice(0, 1)}));
}

TEST(TensorIndexingTest, TestEllipsisTensor) {
  // 创建张量 x 包含元素 0 到 8，reshape 成 3x3 的张量
  auto x = torch::arange(0, 9).to(torch::kLong).view({3, 3});
  // 创建索引张量 idx 包含 {0, 2}
  auto idx = torch::tensor({0, 2});
  // 断言索引 {"...", idx} 返回的张量是否等于 {{0, 2}, {3, 5}, {6, 8}}
  assert_tensor_equal(
      x.index({"...", idx}), torch::tensor({{0, 2}, {3, 5}, {6, 8}}));
  // 断言索引 {idx, "..."} 返回的张量是否等于 {{0, 1, 2}, {6, 7, 8}}
  assert_tensor_equal(
      x.index({idx, "..."}), torch::tensor({{0, 1, 2}, {6, 7, 8}}));
}
TEST(TensorIndexingTest, TestOutOfBoundIndex) {
  // 创建一个大小为 (2, 5, 10) 的张量，并初始化为从 0 到 99 的整数序列
  auto x = torch::arange(0, 100).view({2, 5, 10});
  // 断言对超出边界的索引进行索引时会抛出异常，并验证异常消息
  ASSERT_THROWS_WITH(
      x.index({0, 5}), "index 5 is out of bounds for dimension 1 with size 5");
  // 同上，但这次是维度 0 的边界检查
  ASSERT_THROWS_WITH(
      x.index({4, 5}), "index 4 is out of bounds for dimension 0 with size 2");
  // 同上，但这次是维度 2 的边界检查
  ASSERT_THROWS_WITH(
      x.index({0, 1, 15}),
      "index 15 is out of bounds for dimension 2 with size 10");
  // 使用切片对象时，也要进行边界检查，这里检查维度 2 的边界
  ASSERT_THROWS_WITH(
      x.index({Slice(), Slice(), 12}),
      "index 12 is out of bounds for dimension 2 with size 10");
}

TEST(TensorIndexingTest, TestZeroDimIndex) {
  // 创建一个标量张量
  auto x = torch::tensor(10);

  // 使用 lambda 表达式执行张量索引操作，并打印结果
  auto runner = [&]() -> torch::Tensor {
    std::cout << x.index({0}) << std::endl;
    return x.index({0});
  };

  // 断言对于无效的索引（这里是零维度索引）会抛出异常，并验证异常消息
  ASSERT_THROWS_WITH(runner(), "invalid index");
}

// 下面的测试来自 NumPy 的 test_indexing.py，经过修改以与 libtorch 兼容。
// 使用以下 BSD 许可证：
//
// 版权所有（c）2005-2017 年，NumPy 开发者。
// 保留所有权利。
//
// 源代码的重新发布和使用，无论是否修改，均允许，前提是满足以下条件：
//
//     * 源代码的再发布必须保留上述版权声明、此条件列表和以下免责声明。
//
//     * 二进制形式的再发布必须在文档和/或其他提供的材料中再现上述版权声明、此条件列表和以下免责声明。
//
//     * 未经特定事先书面许可，不得使用 NumPy 开发者的名称或任何贡献者的名称来认可或推广从本软件派生的产品。
//
// 本软件由版权持有人和贡献者提供“按原样”提供，任何明示或暗示的担保，
// 包括但不限于对适销性和特定用途的适用性的暗示担保，都是没有的。
// 在任何情况下，无论是合同责任、严格责任还是侵权行为（包括疏忽或其他）
// 由使用本软件引起的任何直接、间接、偶然、特殊、惩罚性或后果性损害（包括但不限于
// 替代商品或服务的采购；使用、数据或利润的损失；或业务中断）均不承担责任，
// 即使事先被告知此类损害的可能性。

TEST(NumpyTests, TestNoneIndex) {
  // `None` 索引将为张量添加一个新的轴
  auto a = torch::tensor({1, 2, 3});
  // 断言添加 `None` 索引后的维度数目会增加
  ASSERT_EQ(a.index({None}).dim(), a.dim() + 1);
}
TEST(NumpyTests, TestEmptyFancyIndex) {
  // 创建一个包含元素 {1, 2, 3} 的张量 a
  auto a = torch::tensor({1, 2, 3});
  // 使用空列表索引返回一个空张量
  assert_tensor_equal(
      a.index({torch::tensor({}, torch::kLong)}), torch::tensor({}));

  // 创建一个空张量 b，并将其转换为 torch::kLong 类型
  auto b = torch::tensor({}).to(torch::kLong);
  // 使用空列表索引返回一个空张量
  assert_tensor_equal(
      a.index({torch::tensor({}, torch::kLong)}),
      torch::tensor({}, torch::kLong));

  // 将 b 转换为 torch::kFloat 类型
  b = torch::tensor({}).to(torch::kFloat);
  // 使用 b 进行索引操作，预期抛出异常
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(a.index({b}), c10::Error);
}

TEST(NumpyTests, TestEllipsisIndex) {
  // 创建一个二维张量 a
  auto a = torch::tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  // 使用 Ellipsis 索引操作不会返回原始张量 a
  ASSERT_FALSE(a.index({"..."}).is_same(a));
  // 使用 Ellipsis 索引返回原始张量 a
  assert_tensor_equal(a.index({"..."}), a);
  // 在 numpy <1.9 中，`a[...]` 等价于 `a`
  ASSERT_EQ(a.index({"..."}).data_ptr(), a.data_ptr());

  // 使用 Ellipsis 进行切片可以跳过任意数量的维度
  assert_tensor_equal(a.index({0, "..."}), a.index({0}));
  assert_tensor_equal(a.index({0, "..."}), a.index({0, Slice()}));
  assert_tensor_equal(a.index({"...", 0}), a.index({Slice(), 0}));

  // 在 NumPy 中，使用 Ellipsis 进行切片会得到一个零维数组。在 PyTorch 中，
  // 我们没有单独的零维数组和标量。
  assert_tensor_equal(a.index({0, "...", 1}), torch::tensor(2));

  // 在零维数组上使用 `Ellipsis` 进行赋值
  auto b = torch::tensor(1);
  b.index_put_({Ellipsis}, 2);
  ASSERT_EQ(b.item<int64_t>(), 2);
}

TEST(NumpyTests, TestSingleIntIndex) {
  // 单个整数索引选择一个行
  auto a = torch::tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});

  assert_tensor_equal(a.index({0}), torch::tensor({1, 2, 3}));
  assert_tensor_equal(a.index({-1}), torch::tensor({7, 8, 9}));

  // 索引超出边界会产生 IndexError
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(a.index({1 << 30}), c10::Error);
  // 注意：根据标准
  // (http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0543r0.html)，对于
  // 有符号整数，在表达式求值期间，如果结果在其类型的可表示值范围之外，行为未定义。
  // 因此，无法检查索引溢出的情况，因为它可能不会抛出异常。
}

TEST(NumpyTests, TestSingleBoolIndex) {
  // 单个布尔值索引
  auto a = torch::tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});

  assert_tensor_equal(a.index({true}), a.index({None}));
  assert_tensor_equal(a.index({false}), a.index({None}).index({Slice(0, 0)}));
}

TEST(NumpyTests, TestBooleanShapeMismatch) {
  // 创建一个全为 1 的张量 arr，形状为 {5, 4, 3}
  auto arr = torch::ones({5, 4, 3});

  // 创建一个包含 true 的张量 index
  auto index = torch::tensor({true});
  // 当索引与张量的形状不匹配时，抛出异常，错误信息包含 "mask"
  ASSERT_THROWS_WITH(arr.index({index}), "mask");

  // 创建一个包含 false 的张量 index
  index = torch::tensor({false, false, false, false, false, false});
  // 当索引与张量的形状不匹配时，抛出异常，错误信息包含 "mask"
  ASSERT_THROWS_WITH(arr.index({index}), "mask");

  {
    // 捕获警告
    WarningCapture warnings;

    // 创建一个形状为 {4, 4}，数据类型为 torch::kByte，值全为 0 的张量 index
    index = torch::empty({4, 4}, torch::kByte).zero_();
    # 断言语句，验证在 arr 数组中查找 {index} 元素时是否会抛出异常，并且异常消息包含 "mask"
    ASSERT_THROWS_WITH(arr.index({index}), "mask");

    # 断言语句，验证在 arr 数组中使用 {Slice(), index} 进行索引时是否会抛出异常，并且异常消息包含 "mask"
    ASSERT_THROWS_WITH(arr.index({Slice(), index}), "mask");

    # 断言语句，验证在 warnings 字符串中查找 "indexing with dtype torch.uint8 is now deprecated" 子串出现的次数是否等于 2
    ASSERT_EQ(
        count_substr_occurrences(
            warnings.str(),
            "indexing with dtype torch.uint8 is now deprecated"),
        2);
}

// 定义一个测试用例 NumpyTests.TestBooleanIndexingOnedim
TEST(NumpyTests, TestBooleanIndexingOnedim) {
  // 使用长度为一的布尔数组对二维数组进行索引
  auto a = torch::tensor({{0., 0., 0.}});
  auto b = torch::tensor({true});
  // 断言索引后的张量与原始张量相等
  assert_tensor_equal(a.index({b}), a);
  // 对布尔数组进行赋值操作
  a.index_put_({b}, 1.);
  // 断言修改后的张量与预期张量相等
  assert_tensor_equal(a, torch::tensor({{1., 1., 1.}}));
}

// 定义一个测试用例 NumpyTests.TestBooleanAssignmentValueMismatch
TEST(NumpyTests, TestBooleanAssignmentValueMismatch) {
  // 当值的形状无法广播到订阅时，布尔赋值应该失败（参见 gh-3458）
  auto a = torch::arange(0, 4);

  // 定义一个 lambda 函数 f，接受一个张量 a 和一个整数向量 v
  auto f = [](torch::Tensor a, std::vector<int64_t> v) -> void {
    // 使用布尔索引将张量 a 中大于 -1 的位置赋值为张量 v
    a.index_put_({a > -1}, torch::tensor(v));
  };

  // 断言调用 f(a, {}) 会抛出 shape mismatch 的异常
  ASSERT_THROWS_WITH(f(a, {}), "shape mismatch");
  // 断言调用 f(a, {1, 2, 3}) 会抛出 shape mismatch 的异常
  ASSERT_THROWS_WITH(f(a, {1, 2, 3}), "shape mismatch");
  // 断言调用 f(a.index({Slice(None, 1)}), {1, 2, 3}) 会抛出 shape mismatch 的异常
  ASSERT_THROWS_WITH(f(a.index({Slice(None, 1)}), {1, 2, 3}), "shape mismatch");
}

// 定义一个测试用例 NumpyTests.TestBooleanIndexingTwodim
TEST(NumpyTests, TestBooleanIndexingTwodim) {
  // 使用二维布尔数组对二维数组进行索引
  auto a = torch::tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  auto b = torch::tensor(
      {{true, false, true}, {false, true, false}, {true, false, true}});
  // 断言索引后的张量与预期张量相等
  assert_tensor_equal(a.index({b}), torch::tensor({1, 3, 5, 7, 9}));
  // 断言对子数组进行索引后的张量与预期张量相等
  assert_tensor_equal(a.index({b.index({1})}), torch::tensor({{4, 5, 6}}));
  // 断言对索引数组进行索引后的张量与预期张量相等
  assert_tensor_equal(a.index({b.index({0})}), a.index({b.index({2})}));

  // 使用布尔索引对数组进行赋值操作
  a.index_put_({b}, 0);
  // 断言修改后的张量与预期张量相等
  assert_tensor_equal(a, torch::tensor({{0, 2, 0}, {4, 0, 6}, {0, 8, 0}}));
}

// 定义一个测试用例 NumpyTests.TestBooleanIndexingWeirdness
TEST(NumpyTests, TestBooleanIndexingWeirdness) {
  // 测试一些奇怪的布尔索引行为
  auto a = torch::ones({2, 3, 4});
  // 断言对数组使用奇怪的布尔索引后的张量尺寸与预期尺寸相等
  ASSERT_EQ(
      a.index({false, true, "..."}).sizes(), torch::IntArrayRef({0, 2, 3, 4}));
  // 断言对数组使用奇怪的布尔索引后的张量与预期张量相等
  assert_tensor_equal(
      torch::ones({1, 2}),
      a.index(
          {true,
           torch::tensor({0, 1}),
           true,
           true,
           torch::tensor({1}),
           torch::tensor({{2}})}));
  // 使用 NOLINTNEXTLINE 来禁止某些检查，并断言对数组使用奇怪的布尔索引会抛出异常
  ASSERT_THROW(a.index({false, torch::tensor({0, 1}), "..."}), c10::Error);
}

// 定义一个测试用例 NumpyTests.TestBooleanIndexingWeirdnessTensors
TEST(NumpyTests, TestBooleanIndexingWeirdnessTensors) {
  // 测试一些奇怪的布尔索引行为
  auto false_tensor = torch::tensor(false);
  auto true_tensor = torch::tensor(true);
  auto a = torch::ones({2, 3, 4});
  // 断言对数组使用奇怪的布尔索引后的张量尺寸与预期尺寸相等
  ASSERT_EQ(
      a.index({false, true, "..."}).sizes(), torch::IntArrayRef({0, 2, 3, 4}));
  // 断言对数组使用奇怪的布尔索引后的张量与预期张量相等
  assert_tensor_equal(
      torch::ones({1, 2}),
      a.index(
          {true_tensor,
           torch::tensor({0, 1}),
           true_tensor,
           true_tensor,
           torch::tensor({1}),
           torch::tensor({{2}})}));
  // 使用 NOLINTNEXTLINE 来禁止某些检查，并断言对数组使用奇怪的布尔索引会抛出异常
  ASSERT_THROW(
      a.index({false_tensor, torch::tensor({0, 1}), "..."}), c10::Error);
}
TEST(NumpyTests, TestBooleanIndexingAlldims) {
  // 创建一个标量张量 true_tensor
  auto true_tensor = torch::tensor(true);
  // 创建一个形状为 [2, 3] 的全一张量 a
  auto a = torch::ones({2, 3});
  // 断言使用布尔值索引 {true, true} 后张量形状为 [1, 2, 3]
  ASSERT_EQ(a.index({true, true}).sizes(), torch::IntArrayRef({1, 2, 3}));
  // 断言使用张量 true_tensor 作为索引后张量形状为 [1, 2, 3]
  ASSERT_EQ(
      a.index({true_tensor, true_tensor}).sizes(),
      torch::IntArrayRef({1, 2, 3}));
}

TEST(NumpyTests, TestBooleanListIndexing) {
  // 使用布尔列表索引一个二维数组
  auto a = torch::tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  // 创建布尔张量 b，索引第一行
  auto b = torch::tensor({true, false, false});
  // 创建布尔张量 c，索引前两行
  auto c = torch::tensor({true, true, false});
  // 断言使用布尔张量 b 索引后结果为 [[1, 2, 3]]
  assert_tensor_equal(a.index({b}), torch::tensor({{1, 2, 3}}));
  // 断言使用布尔张量 b 两次索引后结果为 [1]
  assert_tensor_equal(a.index({b, b}), torch::tensor({1}));
  // 断言使用布尔张量 c 索引后结果为 [[1, 2, 3], [4, 5, 6]]
  assert_tensor_equal(a.index({c}), torch::tensor({{1, 2, 3}, {4, 5, 6}}));
  // 断言使用布尔张量 c 两次索引后结果为 [1, 5]
  assert_tensor_equal(a.index({c, c}), torch::tensor({1, 5}));
}

TEST(NumpyTests, TestEverythingReturnsViews) {
  // `...` 返回自身
  auto a = torch::tensor({5});

  ASSERT_FALSE(a.is_same(a.index({"..."})));
  ASSERT_FALSE(a.is_same(a.index({Slice()})));
}

TEST(NumpyTests, TestBroaderrorsIndexing) {
  // 创建一个全零的 5x5 张量 a
  auto a = torch::zeros({5, 5});
  // 使用不符合形状的索引引发异常
  ASSERT_THROW(
      a.index({torch::tensor({0, 1}), torch::tensor({0, 1, 2})}), c10::Error);
  // 使用不符合形状的索引引发异常
  ASSERT_THROW(
      a.index_put_({torch::tensor({0, 1}), torch::tensor({0, 1, 2})}, 0),
      c10::Error);
}

TEST(NumpyTests, TestTrivialFancyOutOfBounds) {
  // 创建一个全零的长度为 5 的张量 a
  auto a = torch::zeros({5});
  // 创建一个长度为 20 的长整型张量 ind，且元素全部为 1
  auto ind = torch::ones({20}, torch::kInt64);
  // 在 ind 的倒数第一个位置（即第 20 个位置）放置元素 10
  ind.index_put_({-1}, 10);
  // 使用超出范围的索引引发异常
  ASSERT_THROW(a.index({ind}), c10::Error);
  // 使用超出范围的索引引发异常
  ASSERT_THROW(a.index_put_({ind}, 0), c10::Error);
  // 重新创建长度为 20 的长整型张量 ind，且元素全部为 1
  ind = torch::ones({20}, torch::kInt64);
  // 在 ind 的第一个位置放置元素 11
  ind.index_put_({0}, 11);
  // 使用超出范围的索引引发异常
  ASSERT_THROW(a.index({ind}), c10::Error);
  // 使用超出范围的索引引发异常
  ASSERT_THROW(a.index_put_({ind}, 0), c10::Error);
}

TEST(NumpyTests, TestIndexIsLarger) {
  // 简单的高维索引广播示例
  auto a = torch::zeros({5, 5});
  // 使用嵌套张量作为索引，对指定位置赋值
  a.index_put_(
      {torch::tensor({{0}, {1}, {2}}), torch::tensor({0, 1, 2})},
      torch::tensor({2., 3., 4.}));

  // 断言切片索引结果是否全部为指定值
  ASSERT_TRUE(
      (a.index({Slice(None, 3), Slice(None, 3)}) == torch::tensor({2., 3., 4.}))
          .all()
          .item<bool>());
}

TEST(NumpyTests, TestBroadcastSubspace) {
  // 创建一个全零的 100x100 张量 a
  auto a = torch::zeros({100, 100});
  // 创建一个沿第一维度从 0 到 99 的浮点数张量 v
  auto v = torch::arange(0., 100).index({Slice(), None});
  // 创建一个从 99 到 0 的长整型张量 b
  auto b = torch::arange(99, -1, -1).to(torch::kLong);
  // 使用张量 b 作为索引，将张量 v 的值放入张量 a
  a.index_put_({b}, v);
  // 创建预期的张量 expected，形状与 a 相同，元素值为 b 的浮点表示
  auto expected = b.to(torch::kDouble).unsqueeze(1).expand({100, 100});
  // 断言张量 a 与预期结果 expected 相等
  assert_tensor_equal(a, expected);
}
```