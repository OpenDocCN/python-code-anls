# `.\pytorch\aten\src\ATen\native\cpu\CopyKernel.cpp`

```
// 定义宏 TORCH_ASSERT_NO_OPERATORS

// 包含 ATen 库的分发头文件
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>

// 包含 ATen 库的复制操作、一元操作、张量迭代器等头文件
#include <ATen/native/Copy.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/TensorIterator.h>

// 包含 ATen 库 CPU 相关的复制内核、循环头文件
#include <ATen/native/cpu/CopyKernel.h>
#include <ATen/native/cpu/Loops.h>

// 包含 C10 实用工具的类型转换头文件
#include <c10/util/TypeCast.h>

// 包含 ATen 库 CPU 相关的数学函数头文件
#include <ATen/native/cpu/zmath.h>

// 包含 ATen 库内部的张量迭代器头文件
#include <ATen/TensorIteratorInternal.h>

// 包含 ATen 库的并行处理头文件
#include <ATen/Parallel.h>

// 包含 ATen 库 CPU 向量功能头文件
#include <ATen/cpu/vec/functional.h>

// ATen 库 native 命名空间下的 CPU_CAPABILITY 内联命名空间
namespace at::native {
inline namespace CPU_CAPABILITY {

// 匿名命名空间，定义了两个静态函数 reduced_input 和 reduced_output
namespace {
// 检查输入是否为缩减的浮点类型并且输出类型为 float
static bool reduced_input(ScalarType input_t, ScalarType output_t) {
  return !at::isFloat8Type(input_t) && at::isReducedFloatingType(input_t) &&
      output_t == kFloat;
}

// 检查输出是否为缩减的浮点类型并且输入类型为 float
static bool reduced_output(ScalarType input_t, ScalarType output_t) {
  return !at::isFloat8Type(output_t) && at::isReducedFloatingType(output_t) &&
      input_t == kFloat;
}
} // namespace

// 检查是否可以使用缩减的浮点类型进行复制操作，需满足条件并利用向量化优化
static bool reduced_float_type_copy(
    bool requires_conj,
    TensorIteratorBase& iter) {
  auto strides_out = iter.strides(0);
  auto strides_in = iter.strides(1);

  // 检查输入为 BFloat16/Half 数据类型且输出为 float 数据类型，或者输入为 float 数据类型且输出为 BFloat16/Half 数据类型，并且输入和输出需要连续的部分以利用向量化
  return (
      !requires_conj &&
      ((reduced_input(iter.dtype(1), iter.dtype(0)) &&
        sizeof(float) == strides_out[0] &&
        (static_cast<int64_t>(elementSize(iter.dtype(1))) == strides_in[0] ||
         strides_in[0] == 0)) ||
       (reduced_output(iter.dtype(1), iter.dtype(0)) &&
        static_cast<int64_t>(elementSize(iter.dtype(0))) == strides_out[0] &&
        (sizeof(float) == strides_in[0] || strides_in[0] == 0))));
}

// 执行缩减的浮点类型复制操作的内核函数
static void reduced_float_copy_kernel(TensorIteratorBase &iter, bool requires_neg) {
  auto strides_out = iter.strides(0);
  auto strides_in = iter.strides(1);
  auto shape = iter.shape();

  // 使用 SmallBuffer 存储 strides
  c10::SmallBuffer<int64_t, 8> strides(2 * std::max(iter.ndim(), 2));

  // 获取 strides 的函数
  auto get_strides = [](int64_t* strides, IntArrayRef strides_out, IntArrayRef strides_in, int64_t ndim) {
      for (const auto dim : c10::irange(ndim)) {
        for (const auto arg : c10::irange(2)) {
          *strides++ = arg == 0? strides_out[dim] : strides_in[dim];
        }
      }
      // 对于至少为 2 维的情况，始终有 2 维的 strides 以支持 2 维 for_each 循环
      if (ndim < 2) {
        std::fill_n(strides, (2 - ndim) * 2, 0);
      }
    };

  // 调用 get_strides 函数计算 strides
  get_strides(strides.data(), strides_out, strides_in, iter.ndim());

  // 如果输入为缩减的浮点类型，则执行对应操作
  if (reduced_input(iter.dtype(1), iter.dtype(0))) {
    // 省略部分内容，根据上下文应该是对输入进行操作的逻辑
  } else if (reduced_output(iter.dtype(1), iter.dtype(0))) {
    // 省略部分内容，根据上下文应该是对输出进行操作的逻辑
  }
}
    // 使用AT_DISPATCH_REDUCED_FLOATING_TYPES宏分发不同浮点类型的操作
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(0), "copy_kernel", [&]() {
      // 定义目标数据类型为scalar_t，源数据类型为float
      using dest_t = scalar_t;
      using source_t = float;
      // 定义目标数据类型和源数据类型的向量化类型
      using Vecd = Vectorized<dest_t>;
      using Vecs = Vectorized<source_t>;
      // 创建一个能够存放2个char*指针的小型缓冲区
      c10::SmallBuffer<char*, 2> ptrs(2);
      // 获取输出数据的指针
      dest_t* output_data = iter.tensor_base(0).data_ptr<dest_t>();
      // 获取输入数据的指针，并且将其类型转换为非const
      source_t* input_data = const_cast<source_t*>(iter.tensor_base(1).const_data_ptr<source_t>());

      // 将指针重新解释为char*类型，并存储在ptrs中
      ptrs[0] = reinterpret_cast<char*>(output_data);
      ptrs[1] = reinterpret_cast<char*>(input_data);

      // 定义一个默认的grain_size
      int64_t grain_size = at::internal::GRAIN_SIZE;

      // 定义循环操作的lambda函数，处理数据的复制
      auto loop = [strides_in, requires_neg](char** base, const int64_t* strides, int64_t size0, int64_t size1) {
        // 创建一个char*数组data，存储base的副本
        std::array<char*, 2> data;
        std::copy_n(base, 2, data.data());
        // 获取外部维度的步长信息
        const int64_t *outer_strides = &strides[2];

        // 遍历第一个维度的所有元素
        for (const auto it C10_UNUSED : c10::irange(size1)) {
          // 创建目标数据类型的向量化对象dst_s
          Vecd dst_s;
          // 如果目标数据类型的步长为0，直接加载源数据并存储到目标数据位置
          if (strides_in[0] == 0) {
            dst_s = Vecd(dest_t(*((source_t*)data[1])));
            // 如果需要取反，执行取反操作
            if (requires_neg) {
              dst_s = dst_s.neg();
            }
          }
          // 初始化i为0，处理向量化大小的数据
          int64_t i = 0;
          for (; i <= size0 - 2 * Vecs::size(); i += 2 * Vecs::size()) {
            // 如果目标数据类型的步长不为0，加载源数据向量并进行转换，存储到目标数据位置
            if (strides_in[0] != 0) {
              Vecs data_vec0 = Vecs::loadu(data[1] + i * sizeof(source_t));
              Vecs data_vec1 = Vecs::loadu(data[1] + (i + Vecs::size()) * sizeof(source_t));
              auto data_vec = convert_from_float<dest_t>(data_vec0, data_vec1);
              // 如果需要取反，执行取反操作
              if (requires_neg) {
                data_vec = data_vec.neg();
              }
              // 将处理后的数据存储到目标数据位置
              data_vec.store(data[0] + i * sizeof(dest_t));
            } else {
              // 否则直接将目标数据向量存储到目标数据位置
              dst_s.store(data[0] + i * sizeof(dest_t));
            }

          }
          // 处理剩余的数据，不满足向量化大小的部分
          if (i < size0) {
            if (strides_in[0] != 0) {
              Vecs data_vec0 = Vecs::loadu(data[1] + i * sizeof(source_t), ((size0 - i) > Vecs::size())?  Vecs::size() : (size0 - i));
              Vecs data_vec1 = Vecs::loadu(data[1] + (i + Vecs::size()) * sizeof(source_t), ((size0 - i) > Vecs::size())?  (size0 - i - Vecs::size()) : 0);
              auto data_vec = convert_from_float<dest_t>(data_vec0, data_vec1);
              // 如果需要取反，执行取反操作
              if (requires_neg) {
                data_vec = data_vec.neg();
              }
              // 将处理后的数据存储到目标数据位置，指定存储的大小
              data_vec.store(data[0] + i * sizeof(dest_t), size0 - i);
            } else {
              // 否则直接将目标数据向量存储到目标数据位置，指定存储的大小
              dst_s.store(data[0] + i * sizeof(dest_t), size0 - i);
            }
          }
          // 更新数据指针，移动到下一个数据块
          data[0] += outer_strides[0];
          data[1] += outer_strides[1];
        }

      };
      // 并行执行循环操作，使用parallel_for函数
      parallel_for(0, iter.numel(), grain_size, [&] (int64_t begin, int64_t end) {
        // 调用serial_for_each函数，执行循环操作
        at::internal::serial_for_each(shape, strides, ptrs.data(), 2, loop, {begin, end});
      });
    });
// 定义一个函数宏，用于在未定义 C10_MOBILE 的情况下分派到不同数据类型的处理函数
#define _AT_DISPATCH_ALL_TYPES(TYPE, NAME, ...)                                               \
        AT_DISPATCH_V2(TYPE, NAME, AT_WRAP(__VA_ARGS__),                                       \
            kComplexHalf, kHalf, kBool,              \
            kBFloat16, kFloat8_e5m2, kFloat8_e4m3fn, \
            kFloat8_e5m2fnuz, kFloat8_e4m3fnuz, AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES))
// 定义一个函数宏，用于在未定义 C10_MOBILE 的情况下分派到不同数据类型的处理函数，但不包括复杂类型
#define _AT_DISPATCH_ALL_TYPES_NO_CF(TYPE, NAME, ...)              \
        AT_DISPATCH_V2(TYPE, NAME, AT_WRAP(__VA_ARGS__),                    \
            kBool, kHalf, kBFloat16, kFloat8_e5m2, kFloat8_e4m3fn, \
            kFloat8_e5m2fnuz, kFloat8_e4m3fnuz, AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES))
#else
// 在定义了 C10_MOBILE 的情况下，使用更复杂的分派方式，包括复杂类型
#define _AT_DISPATCH_ALL_TYPES(TYPE, NAME, ...)                                               \
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(                                               \
            ScalarType::ComplexHalf, ScalarType::Half, ScalarType::Bool,ScalarType::BFloat16, \
            TYPE, NAME, __VA_ARGS__)
// 在定义了 C10_MOBILE 的情况下，使用更复杂的分派方式，但不包括复杂类型
#define _AT_DISPATCH_ALL_TYPES_NO_CF(TYPE, NAME, ...) \
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(       \
            kBool, kHalf, kBFloat16,                  \
            TYPE, NAME, __VA_ARGS__)
#endif

void direct_copy_kernel(TensorIteratorBase &iter) {
  // TODO: 我们实际上不需要每种数据类型分别实例化；我们只需要每种数据类型大小分别实例化。这样可能会在代码大小上节省一些空间
  // TODO: 不确定优化器能否将两层条件编译合并为单个跳转表。我们应该在这里只有一个跳转表；最好是手动写出分派语句，而不是使用 AT_DISPATCH
  // 获取迭代器中第一个元素的数据类型
  ScalarType dtype = iter.dtype(0);
  // 如果是量化整数类型
  if (isQIntType(dtype)) {
    // 调度到量化整数类型的处理函数
    AT_DISPATCH_QINT_TYPES(dtype, "copy_kernel", [&] {
      cpu_kernel_vec(
          iter,
          [=](scalar_t a) -> scalar_t { return a; },
          [=](Vectorized<scalar_t> a) -> Vectorized<scalar_t> { return a; });
    });
  // 如果是复数类型的半精度
  } else if (dtype == ScalarType::ComplexHalf) {
    // 调度到处理 c10::complex<at::Half> 类型的处理函数
    cpu_kernel(iter, [=](c10::complex<at::Half> a) -> c10::complex<at::Half> { return a; });
  // 如果是位类型
  } else if (isBitsType(dtype)) {
    // 调度到位类型的处理函数
    AT_DISPATCH_BIT_TYPES(dtype, "copy_kernel", [&] {
      cpu_kernel(
          iter,
          [=](scalar_t a) -> scalar_t { return a; });
    });
  // 对于其他所有类型
  } else {
    // 根据数据类型分派到处理函数，不包括复杂类型
    _AT_DISPATCH_ALL_TYPES_NO_CF(dtype, "copy_kernel", [&] {
      cpu_kernel_vec(
          iter,
          [=](scalar_t a) -> scalar_t { return a; },
          [=](Vectorized<scalar_t> a) -> Vectorized<scalar_t> { return a; });
    });
  }
}

static void neg_conj_kernel(TensorIteratorBase &iter) {
  // 执行 a = b.neg().conj_physical() 的融合操作
  AT_DISPATCH_COMPLEX_TYPES(iter.common_dtype(), "neg_conj_cpu", [&] {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return -conj_impl(a); },
        [=](Vectorized<scalar_t> a) -> Vectorized<scalar_t> { return a.neg().conj(); });



    // 使用 cpu_kernel_vec 函数，在迭代器 iter 上执行向量化操作
    // 第一个参数 iter 可能是迭代器对象或者迭代器范围

        // 定义 lambda 函数，对输入参数 a 执行共轭取反操作，并返回结果
        [=](scalar_t a) -> scalar_t { return -conj_impl(a); },

        // 定义 lambda 函数，对输入参数 a 执行向量化的取反和共轭操作，并返回结果
        [=](Vectorized<scalar_t> a) -> Vectorized<scalar_t> { return a.neg().conj(); }
    );
  });



  // 闭包结束标记，表明 lambda 函数定义结束
} // 结束 CPU_CAPABILITY 命名空间

static void copy_same_dtype(TensorIteratorBase &iter, bool requires_conj, bool requires_neg) {
  if (requires_neg) {
    // 如果需要进行负数处理，当前不会发生，因为目前没有获取到带有负位的复数张量
    if (requires_conj) {
      neg_conj_kernel(iter);
    } else {
      neg_kernel(iter);
    }
  } else {
    if (requires_conj) {
      conj_kernel(iter);
    } else {
      direct_copy_kernel(iter);
    }
  }
}

void copy_kernel(TensorIterator& iter, bool /*non_blocking*/) {
  ScalarType dtype = iter.dtype(0);
  // 检查是否需要共轭操作：如果是复数类型并且第一个张量的共轭状态与第二个不同
  const bool requires_conj = (
      isComplexType(dtype) && (iter.tensor_base(0).is_conj() != iter.tensor_base(1).is_conj()));
  // 检查是否需要取反操作：如果第一个张量为负数，与第二个张量不同
  const bool requires_neg = (iter.tensor_base(0).is_neg() != iter.tensor_base(1).is_neg());

  if (dtype == iter.dtype(1)) {
    // 如果数据类型相同，则执行相同数据类型的拷贝操作
    copy_same_dtype(iter, requires_conj, requires_neg);
  } else if (reduced_float_type_copy(requires_conj, iter)) {
    // 如果可以减少浮点类型拷贝，执行减少浮点类型的拷贝操作
    reduced_float_copy_kernel(iter, requires_neg);
  } else {
    // 否则，根据数据类型进行分发
    _AT_DISPATCH_ALL_TYPES(dtype, "copy_", [&] {
      using dest_t = scalar_t;
      _AT_DISPATCH_ALL_TYPES(iter.dtype(1), "copy_", [&] {
        if (iter.has_contiguous_first_dim()) {
          // 如果第一个维度是连续的，进行按元素操作
          TORCH_INTERNAL_ASSERT(iter.ninputs() == 1);
          TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

          iter.for_each([](char **data, const int64_t *strides, int64_t size) {
            auto src = reinterpret_cast<const scalar_t*>(data[1]);
            auto dst = reinterpret_cast<dest_t*>(data[0]);
            at::vec::convert(src, dst, size);
          });
        } else {
          // 否则，使用 CPU 内核执行操作
          cpu_kernel(iter, [](scalar_t x) -> dest_t {
            return c10::convert<dest_t>(x);
          });
        }
      });
    });

    // 如果需要共轭或取反操作，执行就地 "拷贝" 操作来完成缺失的取反或共轭操作
    if (requires_conj || requires_neg) {
      auto self = iter.tensor_base(0);
      auto iter = TensorIterator::unary_op(self, self);
      copy_same_dtype(iter, requires_conj, requires_neg);
    }
  }
}

} // 结束 at::native 命名空间

REGISTER_DISPATCH(copy_stub, &copy_kernel);
```