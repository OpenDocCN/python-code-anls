# `.\pytorch\test\cpp\jit\test_jit_type.cpp`

```py
#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/testing/file_check.h>
#include "torch/csrc/jit/ir/ir.h"
#include "torch/csrc/jit/ir/irparser.h"

namespace torch {
namespace jit {

// 定义测试 JitTypeTest.IsComplete，验证类型的完整性
TEST(JitTypeTest, IsComplete) {
  // 创建一个包含符号形状的 TensorType 对象
  auto tt = c10::TensorType::create(
      at::kFloat,
      at::kCPU,
      c10::SymbolicShape(std::vector<std::optional<int64_t>>({1, 49})),
      std::vector<c10::Stride>(
          {c10::Stride{2, true, 1},
           c10::Stride{1, true, 1},
           c10::Stride{0, true, c10::nullopt}}),
      false);
  // 断言此类型不完整
  TORCH_INTERNAL_ASSERT(!tt->isComplete());
  // 断言此类型的 strides 属性不完整
  TORCH_INTERNAL_ASSERT(!tt->strides().isComplete());
}

// 定义测试 JitTypeTest.UnifyTypes，验证类型的统一性
TEST(JitTypeTest, UnifyTypes) {
  // 创建一个 Bool 类型的 TensorType 对象
  auto bool_tensor = TensorType::get()->withScalarType(at::kBool);
  // 创建一个可选的 Bool 类型 TensorType 对象
  auto opt_bool_tensor = OptionalType::create(bool_tensor);
  // 统一两种类型，并验证可选 Bool 类型 TensorType 对象是统一类型的子类型
  auto unified_opt_bool = unifyTypes(bool_tensor, opt_bool_tensor);
  TORCH_INTERNAL_ASSERT(opt_bool_tensor->isSubtypeOf(**unified_opt_bool));

  // 获取普通 TensorType 对象
  auto tensor = TensorType::get();
  // 断言普通 TensorType 对象不是可选 Bool 类型 TensorType 对象的子类型
  TORCH_INTERNAL_ASSERT(!tensor->isSubtypeOf(*opt_bool_tensor));
  // 统一两种类型，并验证统一成功
  auto unified = unifyTypes(opt_bool_tensor, tensor);
  TORCH_INTERNAL_ASSERT(unified);
  // 获取统一后的元素类型，并验证其是 TensorType 对象的子类型
  auto elem = (*unified)->expectRef<OptionalType>().getElementType();
  TORCH_INTERNAL_ASSERT(elem->isSubtypeOf(*TensorType::get()));

  // 创建一个包含可选 NoneType 和 IntType 的元组的可选类型
  auto opt_tuple_none_int = OptionalType::create(
      TupleType::create({NoneType::get(), IntType::get()}));
  // 创建一个包含 IntType 和 NoneType 的元组类型
  auto tuple_int_none = TupleType::create({IntType::get(), NoneType::get()});
  // 统一两种类型，并验证统一成功
  auto out = unifyTypes(opt_tuple_none_int, tuple_int_none);
  TORCH_INTERNAL_ASSERT(out);

  // 创建一个 FutureType 对象，包含 IntType 类型
  auto fut_1 = FutureType::create(IntType::get());
  // 创建一个 FutureType 对象，包含 NoneType 类型
  auto fut_2 = FutureType::create(NoneType::get());
  // 统一两种 FutureType 对象，并验证统一成功
  auto fut_out = unifyTypes(fut_1, fut_2);
  TORCH_INTERNAL_ASSERT(fut_out);
  // 断言统一后的类型是 FutureType 的子类型，其包含可选的 IntType 类型
  TORCH_INTERNAL_ASSERT((*fut_out)->isSubtypeOf(
      *FutureType::create(OptionalType::create(IntType::get()))));

  // 创建一个 DictType 对象，键为 IntType，值为 NoneType
  auto dict_1 = DictType::create(IntType::get(), NoneType::get());
  // 创建一个 DictType 对象，键和值都为 IntType
  auto dict_2 = DictType::create(IntType::get(), IntType::get());
  // 统一两种 DictType 对象，并验证统一失败
  auto dict_out = unifyTypes(dict_1, dict_2);
  TORCH_INTERNAL_ASSERT(!dict_out);
}

} // namespace jit
} // namespace torch
```