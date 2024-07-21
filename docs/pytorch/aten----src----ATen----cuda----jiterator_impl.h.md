# `.\pytorch\aten\src\ATen\cuda\jiterator_impl.h`

```
#pragma once
#include <ATen/jit_macros.h>

#if AT_USE_JITERATOR()

#include <ATen/native/TensorIterator.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/jit_utils.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/native/cuda/JitLoops.cuh>

#include <string>
#include <variant>
#include <vector>

namespace at::native {

// 定义了一个宏，用于展开8种情况的代码模板
#define AT_FOR_8_CASES(_)  \
  _(1)                      \
  _(2)                      \
  _(3)                      \
  _(4)                      \
  _(5)                      \
  _(6)                      \
  _(7)                      \
  _(8)

// 定义了一个宏，用于展开8种情况的代码模板，并在每个情况后添加逗号
#define AT_FOR_8_CASES_WITH_COMMA(_)  \
  _(1)     ,                           \
  _(2)     ,                           \
  _(3)     ,                           \
  _(4)     ,                           \
  _(5)     ,                           \
  _(6)     ,                           \
  _(7)     ,                           \
  _(8)

// 返回额外参数的类型名列表
c10::SmallVector<std::string> get_extra_args_typenames(const c10::SmallVector<at::Scalar>& extra_args) {
  c10::SmallVector<std::string> args_typenames(extra_args.size());
  for (const auto i : c10::irange(extra_args.size())) {
    args_typenames[i] = at::cuda::jit::typeName(extra_args[i].type());
  }
  return args_typenames;
}

// 检查给定类型和指针是否可以向量化
int can_vectorize_up_to(at::ScalarType type, char* pointer) {
  switch(type) {
#define DEFINE_CASE(ctype, scalartype)                                   \
    case ScalarType::scalartype : return memory::can_vectorize_up_to<ctype>(pointer);

    // 展开所有标量类型的向量化检查情况
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CASE)
#undef DEFINE_CASE

    default: TORCH_INTERNAL_ASSERT(false, "Unrecognized ScalarType: ", type);
  }
}

// 基于迭代器的jit版本，查看上述函数
// 参见注释 [Jiterator]，依赖于那里列出的假设
int jitted_can_vectorize_up_to(const TensorIteratorBase& iter) {
  const at::ScalarType common_dtype = iter.common_dtype();
  const at::ScalarType result_dtype = common_dtype;

  // 处理输出
  int result = can_vectorize_up_to(result_dtype, static_cast<char*>(iter.data_ptr(0)));

  // 结合输入
  for (auto i = 1; i < iter.ntensors(); ++i) {
    result = std::min<int>(result, can_vectorize_up_to(common_dtype, static_cast<char*>(iter.data_ptr(i))));
  }

  return result;
}

// 创建唯一的偏移计算器实例，根据输入是否为输入和数组大小N
template<bool IS_INPUT, int N>
static std::unique_ptr<OffsetCalculator<N>> make_unique_offset_calculator(
          const TensorIteratorBase& iter) {
  // 数组大小不能为0，当N == 0时发生
  constexpr int array_size = std::max<int>(N, 1);
  TORCH_INTERNAL_ASSERT(N == (IS_INPUT ? iter.ninputs() : iter.noutputs()));

  std::array<const int64_t*, array_size> strides;
  int64_t element_sizes[array_size];
  for (int i = 0; i < N; i++) {
    int index = IS_INPUT ? i + iter.noutputs() : i;
    strides[i] = iter.strides(index).data();
    element_sizes[i] = iter.element_size(index);
  }
  return std::make_unique<OffsetCalculator<N>>(iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}

} // namespace at::native

#endif // AT_USE_JITERATOR()
template <bool IS_INPUT>
struct OffsetCalculatorVariant {
    // 定义 OffsetCalculatorTypes 类型为 std::variant，包含 8 种不同的 OffsetCalculator 类型
#define DEFINE_CASE(index) std::unique_ptr<OffsetCalculator<index>>
    using OffsetCalculatorTypes = std::variant<
        AT_FOR_8_CASES_WITH_COMMA(DEFINE_CASE)
    >;
#undef DEFINE_CASE

    // 构造函数，根据 IS_INPUT 决定计算输入还是输出的 OffsetCalculator
    OffsetCalculatorVariant(const TensorIteratorBase& iter) {
        // 获取输入或输出的张量数量
        int num = IS_INPUT ? iter.ninputs() : iter.noutputs();

        switch(num) {
#define DEFINE_CASE(index)        \
            case index : v = make_unique_offset_calculator<IS_INPUT, index>(iter); break;

            // 使用宏展开生成不同的 case 语句，实例化对应的 OffsetCalculator
            AT_FOR_8_CASES(DEFINE_CASE)
#undef DEFINE_CASE
            default:
                // 若 num 不在预期范围内，则抛出错误信息
                TORCH_CHECK(false, "OffsetCalculatorVariant is not implemented for num_tensor = ", num);
        }
    }

    // 返回当前 OffsetCalculatorTypes 中存储的数据指针
    void* data_ptr() {
        return std::visit([](auto & v){ return static_cast<void*>(v.get()); }, v);
    }

private:
    // 存储 OffsetCalculatorTypes 变体对象
    OffsetCalculatorTypes v{};
};

struct ArrayVariant {
    // 定义 ArrayTypes 类型为 std::variant，包含 8 种不同的 Array 类型
#define DEFINE_CASE(index) at::detail::Array<char*, index>, at::detail::Array<char*, index+8>
    using ArrayTypes = std::variant<
        AT_FOR_8_CASES_WITH_COMMA(DEFINE_CASE)
    >;
#undef DEFINE_CASE

    // 构造函数，根据迭代器中张量的数量 ntensors 进行初始化
    ArrayVariant(const TensorIteratorBase& iter) {
        // 获取迭代器中张量的数量
        int ntensors = iter.ntensors();
        
        switch(ntensors) {
#define DEFINE_CASE(index)                                            \
            case index: array = at::detail::Array<char*, index>{}; break;   \
            case index+8: array = at::detail::Array<char*, index+8>{}; break;

            // 使用宏展开生成不同的 case 语句，初始化不同大小的 Array 类型
            AT_FOR_8_CASES(DEFINE_CASE)
#undef DEFINE_CASE

            default:
                // 若 ntensors 不在预期范围内，则抛出错误信息
                TORCH_CHECK(false, "ArrayVariant is not implemented for ntensors = ", ntensors);
        }

        // 将张量数据指针填充到数组中对应的位置
        std::visit([&](auto& a) {
            for (auto i = 0; i < ntensors; ++i) {
                a[i] = (char*)iter.data_ptr(i);
            }
        }, array);
    }

    // 返回当前 ArrayTypes 中存储的数据指针
    void* data_ptr() {
        return std::visit([](auto & a){ return static_cast<void*>(&a); }, array);
    }

private:
    // 存储 ArrayTypes 变体对象
    ArrayTypes array;
};

struct TrivialOffsetCalculatorVariant {
    // 定义 TrivialOffsetCalculatorTypes 类型为 std::variant，包含 8 种不同的 TrivialOffsetCalculator 类型
#define DEFINE_CASE(index) TrivialOffsetCalculator<index>
    using TrivialOffsetCalculatorTypes = std::variant<
        AT_FOR_8_CASES_WITH_COMMA(DEFINE_CASE)
    >;
#undef DEFINE_CASE

    // 构造函数，根据 num 参数选择实例化不同的 TrivialOffsetCalculator
    TrivialOffsetCalculatorVariant(int num) {
        switch(num) {
#define DEFINE_CASE(index)      \
            case index: v = TrivialOffsetCalculator<index>(); break;

            // 使用宏展开生成不同的 case 语句，实例化对应的 TrivialOffsetCalculator
            AT_FOR_8_CASES(DEFINE_CASE)
#undef DEFINE_CASE

            default:
                // 若 num 不在预期范围内，则抛出错误信息
                TORCH_CHECK(false, "TrivialOffsetCalculatorVariant is not implemented for num_tensors = ", num);
        }
    }

    // 返回当前 TrivialOffsetCalculatorTypes 中存储的数据指针
    void* data_ptr() {
        return std::visit([](auto & v){ return static_cast<void*>(&v); }, v);
    }

private:
    // 存储 TrivialOffsetCalculatorTypes 变体对象
    TrivialOffsetCalculatorTypes v{};
};

struct LoadWithCastVariant {
    // 定义 LoadWithCastPtr 类型为 std::variant，包含 8 种不同的 LoadWithCast 指针类型
#define DEFINE_CASE(index) std::unique_ptr<memory::LoadWithCast<index>>
    using LoadWithCastPtr = std::variant<
        AT_FOR_8_CASES_WITH_COMMA(DEFINE_CASE)
    >;
#undef DEFINE_CASE

    // 构造函数，根据迭代器中输入张量的数量 arity 进行初始化
    LoadWithCastVariant(const TensorIteratorBase& iter) {
        // 获取迭代器中输入张量的数量
        int arity = iter.ninputs();

        switch(arity) {
// 定义一个结构体 LoadWithCastVariant，在此处开始命名空间 at::native
struct LoadWithCastVariant {

  // 定义一个指向 LoadWithCast 模板类的智能指针
  using LoadWithCastPtr = std::variant<
    AT_FOR_8_CASES_WITH_COMMA(std::unique_ptr<memory::LoadWithCast<index>>)
  >;

  // 构造函数，根据迭代器 iter 中的输入数量选择合适的 LoadWithCast 模板类
  LoadWithCastVariant(const TensorIteratorBase& iter) {
    // 获取输入的数量
    int arity = iter.ninputs();
    switch (arity) {
      // 根据不同的输入数量选择不同的 LoadWithCast 模板类
#define DEFINE_CASE(index)      \
      case index: v = std::make_unique<memory::LoadWithCast<index>>(iter); break;
      AT_FOR_8_CASES(DEFINE_CASE)
#undef DEFINE_CASE

      // 默认情况下，抛出异常，显示未实现相应输入数量的 LoadWithCastVariant
      default:
        TORCH_CHECK(false, "LoadWithCastVariant is not implemented for ninputs = ", arity);
    }
  }

  // 返回当前 v 指向的对象的指针
  void* data_ptr() {
    return std::visit([](auto & v){ return static_cast<void*>(v.get()); }, v);
  }

private:
  // 存储选择的 LoadWithCast 模板类的智能指针
  LoadWithCastPtr v{};
};

// 定义一个结构体 StoreWithCastVariant，在此处开始命名空间 at::native
struct StoreWithCastVariant {

  // 定义一个指向 StoreWithCast 模板类的智能指针
  using StoreWithCastPtr = std::variant<
    AT_FOR_8_CASES_WITH_COMMA(std::unique_ptr<memory::StoreWithCast<index>>)
  >;

  // 构造函数，根据迭代器 iter 中的输出数量选择合适的 StoreWithCast 模板类
  StoreWithCastVariant(const TensorIteratorBase& iter) {
    // 获取输出的数量
    int num = iter.noutputs();
    switch(num) {
      // 根据不同的输出数量选择不同的 StoreWithCast 模板类
#define DEFINE_CASE(index)      \
      case index: v = std::make_unique<memory::StoreWithCast<index>>(iter); break;
      AT_FOR_8_CASES(DEFINE_CASE)
#undef DEFINE_CASE

      // 默认情况下，抛出异常，显示未实现相应输出数量的 StoreWithCastVariant
      default:
        TORCH_CHECK(false, "StoreWithCastVariant is not implemented for noutputs = ", num);
    }
  }

  // 返回当前 v 指向的对象的指针
  void* data_ptr() {
    return std::visit([](auto & v){ return static_cast<void*>(v.get()); }, v);
  }

private:
  // 存储选择的 StoreWithCast 模板类的智能指针
  StoreWithCastPtr v{};
};

} // namespace at::native

#endif // AT_USE_JITERATOR()
```