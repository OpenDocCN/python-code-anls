# `.\pytorch\aten\src\ATen\test\undefined_tensor_test.cpp`

```py
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <string>

using namespace at;

TEST(TestUndefined, UndefinedTest) {
  // 设定随机种子为123
  manual_seed(123);

  // 主要测试未定义张量上的操作不会导致段错误，并给出合理的错误消息。
  // 创建一个未定义的张量 `und`
  Tensor und;
  // 创建一个包含单个元素1的浮点型张量 `ft`
  Tensor ft = ones({1}, CPU(kFloat));

  // 创建一个字符串流对象 `ss`，并将 `und` 的信息写入其中
  std::stringstream ss;
  ss << und << std::endl;
  // 断言 `und` 未定义
  ASSERT_FALSE(und.defined());
  // 断言 `und` 的类型为 "UndefinedType"
  ASSERT_EQ(std::string("UndefinedType"), und.toString());

  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言执行 `und.strides()` 会抛出异常
  ASSERT_ANY_THROW(und.strides());
  // 断言 `und` 的维度为1
  ASSERT_EQ(und.dim(), 1);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言执行 lambda 表达式创建的临时张量赋值操作会抛出异常
  ASSERT_ANY_THROW([]() { return Tensor(); }() = Scalar(5));
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言执行 `und.add(und)` 会抛出异常
  ASSERT_ANY_THROW(und.add(und));
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言执行 `und.add(ft)` 会抛出异常
  ASSERT_ANY_THROW(und.add(ft));
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言执行 `ft.add(und)` 会抛出异常
  ASSERT_ANY_THROW(ft.add(und));
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言执行 `und.add(5)` 会抛出异常
  ASSERT_ANY_THROW(und.add(5));
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言执行 `und.mm(und)` 会抛出异常
  ASSERT_ANY_THROW(und.mm(und));

  // public variable API
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言执行 `und.variable_data()` 会抛出异常
  ASSERT_ANY_THROW(und.variable_data());
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言执行 `und.tensor_data()` 会抛出异常
  ASSERT_ANY_THROW(und.tensor_data());
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言执行 `und.is_view()` 会抛出异常
  ASSERT_ANY_THROW(und.is_view());
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言执行 `und._base()` 会抛出异常
  ASSERT_ANY_THROW(und._base());
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言执行 `und.name()` 会抛出异常
  ASSERT_ANY_THROW(und.name());
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言执行 `und.grad_fn()` 会抛出异常
  ASSERT_ANY_THROW(und.grad_fn());
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言执行 `und.remove_hook(0)` 会抛出异常
  ASSERT_ANY_THROW(und.remove_hook(0));
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言执行 `und.register_hook()` 会抛出异常
  ASSERT_ANY_THROW(und.register_hook([](const Tensor& x) -> Tensor { return x; }));

  // copy_
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言执行 `und.copy_(und)` 会抛出异常
  ASSERT_ANY_THROW(und.copy_(und));
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言执行 `und.copy_(ft)` 会抛出异常
  ASSERT_ANY_THROW(und.copy_(ft));
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言执行 `ft.copy_(und)` 会抛出异常
  ASSERT_ANY_THROW(ft.copy_(und));

  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言执行 `und.toBackend(Backend::CPU)` 会抛出异常
  ASSERT_ANY_THROW(und.toBackend(Backend::CPU));
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言执行 `ft.toBackend(Backend::Undefined)` 会抛出异常
  ASSERT_ANY_THROW(ft.toBackend(Backend::Undefined));

  // 创建一个张量 `to_move` 并使用 `std::move` 将其移动到 `m` 中
  Tensor to_move = ones({1}, CPU(kFloat));
  // NOLINTNEXTLINE(bugprone-use-after-move)
  // 断言 `to_move` 不再被定义
  ASSERT_FALSE(to_move.defined());
  // 断言 `to_move` 使用未定义张量实现
  ASSERT_EQ(to_move.unsafeGetTensorImpl(), UndefinedTensorImpl::singleton());
}
```