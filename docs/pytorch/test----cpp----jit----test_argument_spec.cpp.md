# `.\pytorch\test\cpp\jit\test_argument_spec.cpp`

```
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/jit.h>

#include "test/cpp/jit/test_utils.h"

namespace torch {
namespace jit {

namespace {

at::Device device(const autograd::Variable& v) {
  // 返回给定变量的设备信息
  return v.device();
}

bool isEqual(at::IntArrayRef lhs, at::IntArrayRef rhs) {
  // 检查两个整数数组是否相等
  return lhs.size() == rhs.size() &&
      std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

bool isEqual(const CompleteArgumentInfo& ti, const autograd::Variable& v) {
  // 比较完整参数信息对象与变量的相等性
  if (!ti.defined())
    return ti.defined() == v.defined();
  return ti.device() == device(v) && ti.requires_grad() == v.requires_grad() &&
      ti.type() == v.scalar_type() && isEqual(ti.sizes(), v.sizes()) &&
      isEqual(ti.strides(), v.strides());
}

bool isEqual(const ArgumentInfo& ti, const autograd::Variable& v) {
  // 比较参数信息对象与变量的相等性
  if (!ti.defined())
    return ti.defined() == v.defined();
  return ti.device() == device(v) && ti.requires_grad() == v.requires_grad() &&
      ti.type() == v.scalar_type() && ti.dim() == v.dim();
}

autograd::Variable var(
    at::TensorOptions t,
    at::IntArrayRef sizes,
    bool requires_grad) {
  // 创建具有给定选项、大小和梯度要求的变量
  return autograd::make_variable(at::rand(sizes, t), requires_grad);
}
autograd::Variable undef() {
  // 创建一个未定义的变量
  return autograd::Variable();
}
} // namespace

TEST(ArgumentSpecTest, CompleteArgumentSpec_CUDA) {
  auto const CF = at::CPU(at::kFloat);
  auto const CD = at::CPU(at::kDouble);
  auto const GF = at::CUDA(at::kFloat);
  auto const GD = at::CUDA(at::kDouble);

  auto list = createStack(
      {var(CF, {1}, true),
       var(CD, {1, 2}, false),
       var(GF, {}, true),
       var(GD, {4, 5, 6}, false),
       undef()});

  // 确保存在非标准步长
  list[1].toTensor().transpose_(0, 1);

  // 创建具有不同后端值的相同列表
  auto list2 = createStack(
      {var(CF, {1}, true),
       var(CD, {1, 2}, false),
       var(GF, {}, true),
       var(GD, {4, 5, 6}, false),
       undef()});
  list2[1].toTensor().transpose_(0, 1);

  CompleteArgumentSpec a(true, list);
  CompleteArgumentSpec b(true, list);
  ASSERT_EQ(a.hashCode(), b.hashCode());

  ASSERT_EQ(a, b);
  CompleteArgumentSpec d(true, list2);
  ASSERT_EQ(d, a);
  ASSERT_EQ(d.hashCode(), a.hashCode());

  for (size_t i = 0; i < list.size(); ++i) {
    // 断言：验证 a 的第 i 个元素与 list[i] 转换为张量后是否相等
    ASSERT_TRUE(isEqual(a.at(i), list[i].toTensor()));
  }
  
  // 创建一个不需要梯度的 CompleteArgumentSpec 对象，并将其与 list 初始化
  CompleteArgumentSpec no_grad(/*with_grad=*/false, list);
  // 断言：验证 no_grad 与 a 不相等
  ASSERT_TRUE(no_grad != a);

  // 创建一个无序集合 spec 用于存储 CompleteArgumentSpec 对象
  std::unordered_set<CompleteArgumentSpec> spec;
  // 将 a 插入 spec 中，因为后续会使用 a，所以不进行移动操作
  spec.insert(a);
  // 断言：验证 spec 中包含 b
  ASSERT_TRUE(spec.count(b) > 0);
  // 断言：验证 spec 中不包含 no_grad
  ASSERT_EQ(spec.count(no_grad), 0);
  // 将 no_grad 移动到 spec 中
  spec.insert(std::move(no_grad));
  // 断言：验证 spec 中包含一个带有 true 标志和 list 的 CompleteArgumentSpec 对象
  ASSERT_EQ(spec.count(CompleteArgumentSpec(true, list)), 1);

  // 修改 list2 中索引为 1 的元素，并对其进行转置操作
  list2[1].toTensor().transpose_(0, 1);
  // 创建一个带有 true 标志和修改后的 list2 的 CompleteArgumentSpec 对象 c
  CompleteArgumentSpec c(true, list2);
  // 断言：验证 c 不等于 a
  ASSERT_FALSE(c == a);
  // 断言：验证 spec 中不包含 c
  ASSERT_EQ(spec.count(c), 0);

  // 创建一个栈 stack 包含多个元素：张量、整数和张量
  Stack stack = {var(CF, {1, 2}, true), 3, var(CF, {1, 2}, true)};
  // 创建一个带有 true 标志和 stack 的 CompleteArgumentSpec 对象 with_const
  CompleteArgumentSpec with_const(true, stack);
  // 断言：验证 with_const 中索引为 2 的元素的尺寸数量为 2
  ASSERT_EQ(with_const.at(2).sizes().size(), 2);
// 测试用例：ArgumentSpecTest，验证在 CUDA 环境下的基本功能
TEST(ArgumentSpecTest, Basic_CUDA) {
  // 定义不同设备和数据类型的引用
  auto& CF = at::CPU(at::kFloat);
  auto& CD = at::CPU(at::kDouble);
  auto& GF = at::CUDA(at::kFloat);
  auto& GD = at::CUDA(at::kDouble);

  // 编译并转换为图形函数的 JIT 代码
  auto graph = toGraphFunction(jit::compile(R"JIT(
   def fn(a, b, c, d, e):
      return a, b, c, d, e
   )JIT")
                                   ->get_function("fn"))
                   .graph();

  // 创建参数规范创建器
  ArgumentSpecCreator arg_spec_creator(*graph);

  // 创建第一个参数列表
  auto list = createStack(
      {var(CF, {1}, true),
       var(CD, {1, 2}, false),
       var(GF, {}, true),
       var(GD, {4, 5, 6}, false),
       undef()});

  // 确保第二个张量具有非标准步长
  list[1].toTensor().transpose_(0, 1);

  // 创建第二个参数列表，与第一个列表的内容相同但是具有不同的支持值
  auto list2 = createStack(
      {var(CF, {1}, true),
       var(CD, {1, 2}, false),
       var(GF, {}, true),
       var(GD, {4, 5, 6}, false),
       undef()});
  list2[1].toTensor().transpose_(0, 1);

  // 创建参数规范对象 a 和 b，并验证它们的哈希码相等
  ArgumentSpec a = arg_spec_creator.create(true, list);
  ArgumentSpec b = arg_spec_creator.create(true, list);
  ASSERT_EQ(a.hashCode(), b.hashCode());

  // 验证参数规范对象 a 和 b 相等
  ASSERT_EQ(a, b);

  // 创建参数规范对象 d，并验证其与参数规范对象 a 相等
  ArgumentSpec d = arg_spec_creator.create(true, list2);
  ASSERT_EQ(d, a);
  ASSERT_EQ(d.hashCode(), a.hashCode());

  // 遍历参数列表，确保非标准步长的张量转置
  for (size_t i = 0; i < list.size(); ++i) {
    // 断言：验证 a 的第 i 个张量与 list[i] 转换为张量后是否相等
    ASSERT_TRUE(isEqual(a.tensorAt(i), list[i].toTensor()));
  }

  // 创建一个不需要梯度的 ArgumentSpec 对象，用于测试，其中包含 list 中的内容
  ArgumentSpec no_grad = arg_spec_creator.create(/*with_grad=*/false, list);
  // 断言：验证 no_grad 与 a 不相等
  ASSERT_TRUE(no_grad != a);

  // 创建一个无序集合 spec，用于存储 ArgumentSpec 对象
  std::unordered_set<ArgumentSpec> spec;
  // 将 a 插入 spec 中，用于后面的测试
  spec.insert(a);
  // 断言：验证 spec 中包含 b
  ASSERT_TRUE(spec.count(b) > 0);
  // 断言：验证 spec 中不包含 no_grad
  ASSERT_EQ(spec.count(no_grad), 0);

  // 将 no_grad 插入 spec 中（移动语义）
  spec.insert(std::move(no_grad));
  // 断言：验证 spec 中包含使用 arg_spec_creator 创建的新对象（with_grad=true, list）
  ASSERT_EQ(spec.count(arg_spec_creator.create(true, list)), 1);

  // 修改 list2 中第二个元素的张量，进行转置操作
  list2[1].toTensor().transpose_(0, 1);
  // 创建一个 ArgumentSpec 对象 c，与 list2 相同，除了一个步长不同（用于之前不同，现在相同的测试）
  ArgumentSpec c = arg_spec_creator.create(
      true, list2);
  // 断言：验证 c 与 a 相等
  ASSERT_TRUE(c == a);
  // 断言：验证 spec 中包含 c 的计数为 1
  ASSERT_EQ(spec.count(c), 1);
}

} // namespace jit
} // namespace torch
```