# `.\pytorch\aten\src\ATen\native\cpu\IndexKernel.cpp`

```py
// 定义宏以禁用Torch的运算符断言
#define TORCH_ASSERT_NO_OPERATORS
// 包含ATen库中的索引内核头文件
#include <ATen/native/IndexKernel.h>

// 包含标准库头文件
#include <cmath>
#include <iostream>

// 包含ATen库的其他必要头文件
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/Parallel.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/AtomicAddFloat.h>
#include <ATen/native/cpu/IndexKernelUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/irange.h>
#include <c10/core/Scalar.h>

// ATen命名空间
namespace at::native {

// 匿名命名空间，用于实现私有函数和数据
namespace {

// 使用向量化命名空间
using namespace vec;

// 索引内核函数，处理TensorIteratorBase对象
void index_kernel(TensorIteratorBase& iter, IntArrayRef index_size, IntArrayRef index_stride) {
  // 分发不同数据类型的索引处理函数
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(kComplexHalf, kHalf, kBool, kBFloat16,
    iter.dtype(), "index_cpu", [&] {
    // 调用CPU索引内核函数，根据偏移量复制数据
    cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [](char* dst, char* src, int64_t offset) {
      *(scalar_t*)dst = *(scalar_t*)(src + offset);
    });
  });
}

// 线性索引转偏移量计算器，根据TensorBase对象实现
// 与其GPU版本cuda::detail::IndexToOffset算法相同
// OffsetCalculator在列主序列中再次实现相同的算法
struct IndexToOffset {
  const IntArrayRef sizes;   // 引用尺寸数组
  const IntArrayRef strides; // 引用步长数组
  const int64_t ndim;        // 维度数
  explicit IndexToOffset(const TensorBase & tensor) :
      sizes(tensor.sizes()), strides(tensor.strides()), ndim(tensor.dim()) {
  }

  // 根据线性索引获取偏移量
  int64_t get(int64_t linear_index) const {
    int64_t offset = 0;
    for (int64_t i = ndim - 1; i > 0; i--) {
      offset += (linear_index % sizes[i]) * strides[i];
      linear_index /= sizes[i];
    }
    return offset + linear_index * strides[0];
  }
};

// CPU取值/置值内核函数模板
template <typename scalar_t, typename func_t>
void cpu_take_put_kernel(
    TensorIterator& iter,
    const TensorBase& indexed,
    bool is_indexed_data_mutated,
    const func_t& f,
    bool serial_execution=false) {
  // 此内核与`cpu_index_kernel`使用相同策略
  // 即使indexed_tensor是const，我们也通过data_ptr进行修改
  // 这有点不太规范，但否则需要向iter添加具有零步长的张量

  // 当启动并行版本时，设置一个相对较小的粒度，小于INTERNAL::GRAIN_SIZE
  // 以使所有可用的线程数获得更平衡的工作负载和更好的缓存位置
  // 此处的粒度由操作基准选择，以克服线程启动开销
  // 或许针对`put_`调整这个数字？此数字已针对`index_put`进行调整
  constexpr int parallel_grain_size = 3000;
  const bool is_contiguous = indexed.is_contiguous();
  const auto numel = indexed.numel();
  const auto offset_indexed = IndexToOffset(indexed);

  // 如果indexed数据发生变异，使用mutable的数据指针，否则使用const的数据指针
  auto* indexed_data = is_indexed_data_mutated ?
   indexed.data_ptr<scalar_t>()
   : const_cast<scalar_t*>(indexed.const_data_ptr<scalar_t>());
  
  // 循环操作函数，处理数据指针和步长
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    // 获取迭代数据的字节指针
    auto* iterated_data_bytes = data[0];
    // 获取索引数据的字节指针
    auto* index_data_bytes = data[1];
    // 对于每个索引值 elem 在 [0, n) 范围内
    for (const auto elem C10_UNUSED : c10::irange(n)) {
      // 将索引数据字节转换为 int64_t 类型的索引值 idx
      auto idx = *reinterpret_cast<int64_t*>(index_data_bytes);
      // 将迭代数据字节转换为 scalar_t 类型的迭代值 iterated
      auto& iterated = *reinterpret_cast<scalar_t*>(iterated_data_bytes);

      // 检查索引 idx 是否在有效范围内，报错信息包含索引值和元素总数 numel
      TORCH_CHECK_INDEX(idx >= -numel && idx < numel,
                        "out of range: tried to access index ",
                        idx, " on a tensor of ", numel, " elements.");
      // 如果索引 idx 小于 0，则转换为对应的正索引
      if (idx < 0) {
        idx += numel;
      }
      // 如果张量不是连续存储的，则使用偏移索引重新计算 idx
      if (!is_contiguous) {
        idx = offset_indexed.get(idx);
      }
      // 调用函数 f 处理迭代值 iterated、索引数据 indexed_data 和计算后的 idx
      f(iterated, indexed_data, idx);
      // 更新迭代数据字节指针，按照 strides[0] 移动
      iterated_data_bytes += strides[0];
      // 更新索引数据字节指针，按照 strides[1] 移动
      index_data_bytes += strides[1];
    }
  };
  // 如果是串行执行，则使用 iter.serial_for_each 执行循环任务
  if (serial_execution) {
    iter.serial_for_each(loop, {0, iter.numel()});
  } else {
    // 如果是并行执行，则使用 iter.for_each 执行循环任务，设定并行粒度 parallel_grain_size
    iter.for_each(loop, parallel_grain_size);
  }
}

// 将数据放入张量迭代器指向的位置，支持累加操作
void put_kernel(
  TensorIterator& iter,                  // 张量迭代器，用于迭代操作
  const TensorBase & self,               // 输入张量的基类引用
  const bool accumulate) {               // 是否执行累加操作的布尔值
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16,
    iter.dtype(), "take_put_cpu", [&] {  // 根据张量数据类型进行分派，lambda表达式引用外部变量
    // iter 虽然可以是const，但 for_each 没有 const 版本
    if (accumulate) {                    // 如果需要进行累加操作
      // 注意：这个确定性问题与 `index_put_kernel` 中的相同
      // 参见注释 [Enabling Deterministic Operations]
      // 并行的 CPU 累加内核是不确定的，因此如果启用了确定性算法，必须启用串行执行
      bool is_deterministic = at::globalContext().deterministicAlgorithms();
      bool use_parallel_for = (!is_deterministic) && (
        (iter.numel() >= internal::GRAIN_SIZE) && (at::get_num_threads() > 1));
      if (use_parallel_for && iter.dtype() == ScalarType::Float) {
        // 如果满足条件，并且数据类型为 Float，使用并行化的累加内核
        cpu_take_put_kernel<float>(iter, self, true,
            [](float& iterated, float* indexed, const int64_t idx) {
                cpu_atomic_add_float(indexed+idx, iterated);  // 原子化的 Float 加法操作
              });
      } else {
        // 否则，使用串行化的累加内核
        // TODO: 调查累加内核的并行化
        // 与非累加情况不同，这需要线程安全
        cpu_take_put_kernel<scalar_t>(iter, self, true,
            [](scalar_t& iterated, scalar_t* indexed, const int64_t idx) {
                indexed[idx] += iterated;  // 累加操作
              },
            /*serial_execution=*/true);
      }
    } else {
      // 如果不进行累加操作，直接使用非累加的内核
      cpu_take_put_kernel<scalar_t>(iter, self, true,
          [](scalar_t& iterated, scalar_t* indexed, const int64_t idx) {
              indexed[idx] = iterated;  // 赋值操作
            });
    }
  });
}

// 从输入张量中获取数据到张量迭代器指向的位置
void take_kernel(
  TensorIterator& iter,                  // 张量迭代器，用于迭代操作
  const TensorBase & input) {            // 输入张量的基类引用
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16,
    iter.dtype(), "take_cpu", [&] {       // 根据张量数据类型进行分派，lambda表达式引用外部变量
      // 使用非累加的内核，直接获取数据
      cpu_take_put_kernel<scalar_t>(iter, input, false,
          [](scalar_t& iterated, const scalar_t* indexed, const int64_t idx) {
              iterated = indexed[idx];    // 赋值操作
            });
    });
}

// 索引赋值操作的内核函数
void index_put_kernel(
  TensorIterator& iter,                  // 张量迭代器，用于迭代操作
  IntArrayRef index_size,                // 索引大小的数组引用
  IntArrayRef index_stride,              // 索引步长的数组引用
  bool accumulate) {                     // 是否执行累加操作的布尔值
  // 注意：仅当 accumulate 为 true 时支持重复索引
  AT_DISPATCH_V2(
    iter.dtype(),
    "index_put",
    AT_WRAP([&] {
      // See Note [Enabling Deterministic Operations]
      // 查看关于启用确定性操作的注释说明

      // 获取全局上下文中确定性算法的状态
      const bool is_deterministic = at::globalContext().deterministicAlgorithms();

      // 如果需要累加操作
      if (accumulate) {
        // 确定是否可以并行执行cpu_index_kernel，如果启用了确定性算法并且满足一定条件，则可以并行执行
        bool use_parallel_for = (!is_deterministic) && (
          (iter.numel() >= internal::GRAIN_SIZE) && (at::get_num_threads() > 1));
        
        // 如果可以并行执行且迭代器的数据类型为Float
        if (use_parallel_for && iter.dtype() == ScalarType::Float) {
          // 调用并行版本的cpu_index_kernel，用于累加操作，其中包含原子操作
          cpu_index_kernel<float>(iter, index_size, index_stride, [](char* dst, char* src, int64_t offset) {
            cpu_atomic_add_float((float*)(dst + offset), *(float*)src);
          });
        } else {
          // TODO: 调查累加操作内核的并行化。与非累加情况不同，这需要是线程安全的。
          // 调用cpu_index_kernel执行累加操作，强制使用串行执行
          cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [](char* dst, char* src, int64_t offset) {
            *(scalar_t*)(dst + offset) += *(scalar_t*)src;
          }, /*serial_execution=*/true);
        }
      } else {
        // 执行非累加操作的cpu_index_kernel，根据是否启用确定性算法决定是否使用串行执行
        cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [](char* dst, char* src, int64_t offset) {
          *(scalar_t*)(dst + offset) = *(scalar_t*)src;
        }, /*serial_execution=*/is_deterministic);
      }
    }),
    AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
    AT_EXPAND(AT_FLOAT8_TYPES),
    kComplexHalf,
    kHalf,
    kBool,
    kBFloat16);



// 执行AT_WRAP宏，其中包含了对不同类型和复杂类型的扩展处理
// 以及特定的类型常量 kComplexHalf, kHalf, kBool, kBFloat16 的处理
}

void index_fill_kernel(
  TensorIterator& iter,                    // 定义一个函数 index_fill_kernel，接收一个 TensorIterator 对象和三个整数参数
  int64_t dim,                             // 表示索引填充操作针对的维度
  int64_t self_dim_size,                    // 表示 self 张量在指定维度上的尺寸
  int64_t self_dim_stride,                  // 表示 self 张量在指定维度上的步长
  const Scalar& source) {                   // 表示用于填充的标量值 source
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16, kComplexHalf,
    iter.dtype(), "index_fill_cpu", [&] {   // 根据迭代器的数据类型调度对应的操作，lambda 表达式开始
    auto fill_val = source.to<scalar_t>();  // 将标量值 source 转换为当前数据类型 scalar_t，并赋值给 fill_val
    auto handle_nonzero_idx_stride = [&](char** data, const int64_t* strides, int64_t n) {  // 处理非零索引步长的情况的 lambda 函数
      auto* self_data_bytes = data[0];      // 获取 self 张量数据的字节流指针
      auto* index_data_bytes = data[1];     // 获取索引数据的字节流指针
      for (const auto elem C10_UNUSED : c10::irange(n)) {  // 迭代处理每个元素
        auto* self_data = reinterpret_cast<scalar_t*>(self_data_bytes);  // 将 self 数据字节流解释为 scalar_t 类型指针
        auto idx = *reinterpret_cast<int64_t*>(index_data_bytes);  // 解释索引数据字节流为 int64_t 类型
        TORCH_CHECK_INDEX(idx >= -self_dim_size && idx < self_dim_size,
                          "index ", idx, " is out of bounds for dimension ",
                          dim, " with size ", self_dim_size);  // 检查索引是否越界
        if (idx < 0) {                       // 如果索引为负数，转换为正数索引
          idx += self_dim_size;
        }

        self_data[idx * self_dim_stride] = fill_val;  // 在 self 张量中指定位置填充 fill_val

        self_data_bytes += strides[0];       // 更新 self 数据字节流指针
        index_data_bytes += strides[1];      // 更新索引数据字节流指针
      }
    };
    auto handle_zero_idx_stride = [&](char** data, const int64_t* strides, int64_t n) {  // 处理零索引步长的情况的 lambda 函数
      auto* self_data_bytes = data[0];      // 获取 self 张量数据的字节流指针
      auto* index_data_bytes = data[1];     // 获取索引数据的字节流指针
      auto idx = *reinterpret_cast<int64_t*>(index_data_bytes);  // 解释索引数据字节流为 int64_t 类型
      TORCH_CHECK_INDEX(idx >= -self_dim_size && idx < self_dim_size,
                        "index ", idx, " is out of bounds for dimension ",
                        dim, " with size ", self_dim_size);  // 检查索引是否越界
      if (idx < 0) {                       // 如果索引为负数，转换为正数索引
        idx += self_dim_size;
      }
      for (const auto elem C10_UNUSED: c10::irange(n)) {  // 迭代处理每个元素
        auto* self_data = reinterpret_cast<scalar_t*>(self_data_bytes);  // 将 self 数据字节流解释为 scalar_t 类型指针

        self_data[idx * self_dim_stride] = fill_val;  // 在 self 张量中指定位置填充 fill_val

        self_data_bytes += strides[0];       // 更新 self 数据字节流指针
      }
    };

    auto loop = [&](char** data, const int64_t* strides, int64_t n) {  // 定义循环处理函数
      auto idx_stride = strides[1];         // 获取索引步长
      if (idx_stride) {                     // 如果索引步长非零，调用 handle_nonzero_idx_stride 处理
        handle_nonzero_idx_stride(data, strides, n);
      }
      else {                                // 否则调用 handle_zero_idx_stride 处理
        handle_zero_idx_stride(data, strides, n);
      }
    };
    iter.for_each(loop);                    // 对迭代器中的每个元素应用循环处理函数
  });
}

void index_copy_kernel(
  TensorIterator& iter,                    // 定义一个函数 index_copy_kernel，接收一个 TensorIterator 对象和三个整数参数
  int64_t dim,                             // 表示索引复制操作针对的维度
  int64_t self_dim_size,                    // 表示 self 张量在指定维度上的尺寸
  int64_t self_dim_stride) {                // 表示 self 张量在指定维度上的步长
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16, kComplexHalf,
    iter.dtype(), "index_copy_cpu", [&] {   // 根据迭代器的数据类型调度对应的操作，lambda 表达式开始
    auto handle_nonzero_idx_stride = [&](char** data, const int64_t* strides, int64_t n) {
      // 获取指向自身数据的字节指针
      auto* self_data_bytes = data[0];
      // 获取指向索引数据的字节指针
      auto* index_data_bytes = data[1];
      // 获取指向源数据的字节指针
      auto* source_data_bytes = data[2];
      // 遍历范围为 n 的元素
      for (const auto elem C10_UNUSED : c10::irange(n)) {
        // 将自身数据字节指针转换为相应类型的指针
        auto* self_data = reinterpret_cast<scalar_t*>(self_data_bytes);
        // 获取索引值
        auto idx = *reinterpret_cast<int64_t*>(index_data_bytes);
        // 将源数据字节指针转换为相应类型的指针
        auto* source_data = reinterpret_cast<scalar_t*>(source_data_bytes);
        // 检查索引是否越界
        TORCH_CHECK_INDEX(idx >= 0 && idx < self_dim_size,
              "index_copy_(): index ", idx, " is out of bounds for dimension ",
              dim, " with size ", self_dim_size);

        // 执行索引复制操作
        self_data[idx * self_dim_stride] = *source_data;

        // 更新字节指针以指向下一个数据元素
        self_data_bytes += strides[0];
        index_data_bytes += strides[1];
        source_data_bytes += strides[2];
      }
    };
    
    auto handle_zero_idx_stride = [&](char** data, const int64_t* strides, int64_t n) {
      // 获取指向自身数据的字节指针
      auto* self_data_bytes = data[0];
      // 获取指向索引数据的字节指针
      auto* index_data_bytes = data[1];
      // 获取指向源数据的字节指针
      auto* source_data_bytes = data[2];
      // 获取索引值
      auto idx = *reinterpret_cast<int64_t*>(index_data_bytes);
      // 检查索引是否越界
      TORCH_CHECK_INDEX(idx >= 0 && idx < self_dim_size,
            "index_copy_(): index ", idx, " is out of bounds for dimension ",
            dim, " with size ", self_dim_size);
      // 遍历范围为 n 的元素
      for (const auto elem C10_UNUSED : c10::irange(n)) {
        // 将自身数据字节指针转换为相应类型的指针
        auto* self_data = reinterpret_cast<scalar_t*>(self_data_bytes);
        // 将源数据字节指针转换为相应类型的指针
        auto* source_data = reinterpret_cast<scalar_t*>(source_data_bytes);

        // 执行索引复制操作
        self_data[idx * self_dim_stride] = *source_data;

        // 更新字节指针以指向下一个数据元素
        self_data_bytes += strides[0];
        source_data_bytes += strides[2];
      }
    };

    auto loop = [&](char** data, const int64_t* strides, int64_t n) {
      // 获取索引步长
      auto idx_stride = strides[1];
      // 根据索引步长的情况选择处理函数
      if (idx_stride) {
        handle_nonzero_idx_stride(data, strides, n);
      }
      else {
        handle_zero_idx_stride(data, strides, n);
      }
    };
    // 获取当前是否为确定性算法的状态
    bool is_deterministic = at::globalContext().deterministicAlgorithms();
    // 根据确定性状态选择不同的迭代方法
    if (is_deterministic) {
      // 如果是确定性算法，使用串行执行方式
      iter.serial_for_each(loop, {0, iter.numel()});
    } else {
      // 如果不是确定性算法，使用普通迭代方式
      iter.for_each(loop);
    }
  });
  // 定义 lambda 函数 loop，用于处理每个元素
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    // 获取目标张量的指针和步长
    char* dst = data[0];
    // 获取源张量的指针和步长
    char* src = data[1];
    // 获取掩码张量的指针和步长
    char* mask = data[2];
    // 遍历迭代器中的每个元素
    for (const auto i : c10::irange(n)) {
      // 解析当前位置的掩码值
      auto mask_value = *reinterpret_cast<bool*>(mask + strides[2] * i);
      // 如果掩码值为真
      if (mask_value) {
        // 获取源张量中对应位置的数据，并存储到目标张量中
        *(scalar_t*)(dst + strides[0] * offset) = *(reinterpret_cast<scalar_t*>(src + strides[1] * i));
        // 增加偏移量，用于下一个有效位置
        offset++;
      }
    }
  };
  // 串行执行循环操作，覆盖整个迭代器的范围
  iter.serial_for_each(loop, {0, iter.numel()});
}


注释中的每一行都解释了代码中的具体操作，确保了读者能够理解每个步骤的目的和实现方式。
    for (const auto i : c10::irange(n)) {
      // 使用范围遍历器，遍历从0到n-1的整数
      mask_t mask_value = *(mask_t*)(mask + strides[2] * i);
      // 从mask数组中获取第i个元素的值，类型转换为mask_t，并存储在mask_value中
      if constexpr (!std::is_same<mask_t, bool>::value) {
        // 如果mask_t不是布尔类型，则进行条件编译
        TORCH_CHECK(mask_value == 0 || mask_value == 1, "Mask tensor can take 0 and 1 values only");
        // 检查mask_value是否为0或1，否则抛出错误信息"Mask tensor can take 0 and 1 values only"
      }
      if (mask_value) {
        // 如果mask_value为真值（非0），执行以下操作
        int64_t offset_bytes = offset * sizeof(scalar_t);
        // 计算偏移量的字节数，offset乘以scalar_t的大小
        f(dst, src + strides[1] * i, offset_bytes);
        // 调用函数f，传入参数dst，src加上strides[1]乘以i，以及offset_bytes
        offset++;
        // 增加offset的值
      }
    }
  };
  // 结束for循环后，定义了名为loop的lambda函数

  iter.serial_for_each(loop, {0, iter.numel()});
  // 使用iter对象的serial_for_each方法，以loop函数为参数，对0到iter.numel()-1的范围执行串行操作
}

void masked_select_serial_kernel(TensorIterator& iter, int64_t result_stride) {
  // 使用 TensorIterator 迭代器来执行掩码选择的串行内核操作
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Bool, ScalarType::BFloat16, ScalarType::Half,
    iter.dtype(), "masked_select", [&] {
      auto mask_dtype = iter.input_dtype(1);
      // 检查掩码的数据类型是否为布尔型
      if (mask_dtype == ScalarType::Bool) {
        // 如果掩码是布尔型，调用具体的 CPU 函数处理掩码选择操作
        cpu_masked_select_serial_kernel<scalar_t, bool>(iter, [result_stride](char* dst, char* src, int64_t offset) {
          *(scalar_t*)(dst + offset*result_stride) = *(scalar_t*)src;
        });
      } else {
        // 如果掩码不是布尔型，调用具体的 CPU 函数处理掩码选择操作（假设是无符号字符型）
        cpu_masked_select_serial_kernel<scalar_t, unsigned char>(iter, [result_stride](char* dst, char* src, int64_t offset) {
          *(scalar_t*)(dst + offset*result_stride) = *(scalar_t*)src;
        });
      }
    });
}

template <typename scalar_t, typename mask_t, typename func_t>
void cpu_masked_select_kernel(TensorIterator& iter, const func_t& f) {
  // 定义 CPU 下掩码选择的内核函数模板，接受一个函数对象 f 作为参数
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    char* dst = data[0];
    char* src = data[1];
    char* mask = data[2];
    char* mask_prefix_sum = data[3];
    // 遍历数据，根据掩码值进行选择操作
    for (const auto i : c10::irange(n)) {
      mask_t mask_value = *(mask_t*)(mask + strides[2] * i);
      // 如果 mask_t 不是布尔型，检查其取值是否为 0 或 1
      if constexpr (!std::is_same<mask_t, bool>::value) {
        TORCH_CHECK(mask_value == 0 || mask_value == 1, "Mask tensor can take 0 and 1 values only");
      }
      // 如果掩码值为真，则执行特定操作（根据 offset 计算偏移字节并调用函数对象 f 处理）
      if (mask_value) {
        int64_t offset = *(int64_t*)(mask_prefix_sum + strides[3] * i);
        int64_t offset_bytes = (offset - 1) * sizeof(scalar_t);
        f(dst, src + strides[1] * i, offset_bytes);
      }
    }
  };
  // 使用 iter.for_each 调用上述循环函数来执行掩码选择操作
  iter.for_each(loop);
}

void masked_select_kernel(TensorIterator& iter, int64_t result_stride) {
  // 使用 TensorIterator 迭代器来执行掩码选择的核心函数
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Bool, ScalarType::BFloat16, ScalarType::Half,
    iter.dtype(), "masked_select", [&] {
      auto mask_dtype = iter.input_dtype(1);
      // 检查掩码的数据类型是否为布尔型
      if (mask_dtype == ScalarType::Bool) {
        // 如果掩码是布尔型，调用具体的 CPU 函数处理掩码选择操作
        cpu_masked_select_kernel<scalar_t, bool>(iter, [result_stride](char* dst, char* src, int64_t offset) {
          *(scalar_t*)(dst + offset*result_stride) = *(scalar_t*)src;
        });
      } else {
        // 如果掩码不是布尔型，调用具体的 CPU 函数处理掩码选择操作（假设是无符号字符型）
        cpu_masked_select_kernel<scalar_t, unsigned char>(iter, [result_stride](char* dst, char* src, int64_t offset) {
          *(scalar_t*)(dst + offset*result_stride) = *(scalar_t*)src;
        });
      }
    });
}

template <typename scalar_t>
void cpu_hflip_vec(at::TensorIterator& iter) {
  // 定义 CPU 下的向量化水平翻转函数，接受 TensorIterator 迭代器为参数
  auto loop2d = [&](char** base, const int64_t *strides, int64_t size0, int64_t size1) {

    // Here ntensors is defined for output and 1 input. But tensor iterator has defined output, input
    // and restrided_input (see aten/src/ATen/native/TensorTransformations.cpp#L64-L66) but we use only
    // output and input.
    // 这里 ntensors 是为输出和一个输入定义的，但是张量迭代器定义了输出、输入和重定向输入（见 aten/src/ATen/native/TensorTransformations.cpp#L64-L66），但我们只使用了输出和输入。

    static constexpr int ntensors = 2;
    const int64_t *outer_strides = &strides[3];

    // 创建一个长度为 ntensors 的数据数组，用于存放迭代器基址的副本
    std::array<char*, ntensors> data_arr;
    std::copy_n(base, ntensors, data_arr.data());

    using Vec = Vectorized<scalar_t>;

    constexpr auto stride = sizeof(scalar_t);
    // 确保 stride 等于-strides[0] 且等于 strides[1]
    TORCH_INTERNAL_ASSERT(stride == -strides[0] && stride == strides[1]);

    // 对于 size1 范围内的每个 j，执行以下操作
    for (const auto j C10_UNUSED : c10::irange(size1)) {

      // 使用负步长对输出进行矢量化循环
      char** C10_RESTRICT data_ = data_arr.data();
      int64_t n = size0;

      // 将 data_arr 中的数据复制到 data 中
      char* C10_RESTRICT data[ntensors];
      for (const auto arg : c10::irange(ntensors)) {
        data[arg] = data_[arg];
      }

      int64_t i = 0;

      // data[0] 的非对齐预处理
      // 计算偏移量以确保对齐到 32 字节边界
      int64_t offset = (j * n + (n - i - Vec::size())) % 32;
      offset = (offset >= n) ? n : offset;
      for (; i < offset; i++) {
        // 计算输出指针位置并赋值
        scalar_t* out_ptr = (scalar_t*)(data[0] - i * stride);
        *out_ptr = *(scalar_t *)(data[1] + i * stride);
      }

      // 经验上发现，同时处理3个数据项比2个或4个更快
      for (; i <= n - 3 * Vec::size(); i += 3 * Vec::size()) {
        // 加载并翻转数据
        auto out1 = Vec::loadu(data[1] + i * stride);
        auto out2 = Vec::loadu(data[1] + (i + Vec::size()) * stride);
        auto out3 = Vec::loadu(data[1] + (i + 2 * Vec::size()) * stride);
        out1 = flip(out1);
        out2 = flip(out2);
        out3 = flip(out3);
        // 存储翻转后的数据到 data[0]
        out1.store(data[0] - (i + Vec::size() - 1) * stride);
        out2.store(data[0] - (i + 2 * Vec::size() - 1) * stride);
        out3.store(data[0] - (i + 3 * Vec::size() - 1) * stride);
      }

      // 处理剩余不足3个数据项的情况
      if (i < n) {
        for (; i < n; i++) {
          // 计算输出指针位置并赋值
          scalar_t* out_ptr = (scalar_t*)(data[0] - i * stride);
          *out_ptr = *(scalar_t *)(data[1] + i * stride);
        }
      }

      // 更新 data_arr 中每个数据的指针位置
      for (const auto arg : c10::irange(ntensors)) {
        data_arr[arg] += outer_strides[arg];
      }
    }
  };

  // 定义 grain_size 为 AT 内部的 GRAIN_SIZE
  int64_t grain_size = at::internal::GRAIN_SIZE;
  // 对迭代器 iter 执行 2D 循环，每次处理 grain_size 个元素
  iter.for_each(loop2d, grain_size);
  // 将输出结果转换成所需类型
  iter.cast_outputs();
}

void cpu_vflip_memcpy(at::TensorIterator& iter) {
    // 定义一个垂直翻转的特化版本，使用 memcpy 来加速运行时

    auto loop2d = [&](char** base, const int64_t *strides, int64_t size0, int64_t size1) {

        // 在这里，ntensors 被定义为输出和一个输入。但是张量迭代器定义了输出、输入和重构后的输入（参见 aten/src/ATen/native/TensorTransformations.cpp#L64-L66），但我们只使用输出和输入。
        static constexpr int ntensors = 2;
        const int64_t *outer_strides = &strides[3];

        // 创建一个包含 char* 类型的数组，用于保存 base 的副本
        std::array<char*, ntensors> data_arr;
        std::copy_n(base, ntensors, data_arr.data());

        // 断言确保第一个维度的步长与第二个相等
        TORCH_INTERNAL_ASSERT(strides[0] == strides[1]);
        const int64_t stride = strides[0];

        // 遍历第二个维度的大小
        for (const auto j C10_UNUSED : c10::irange(size1)) {

            // 获取当前数据数组的副本
            char** C10_RESTRICT data_ = data_arr.data();
            int64_t n = size0;

            // 创建一个受限制的数据数组，用于存储每个数据指针
            char* C10_RESTRICT data[ntensors];
            for (const auto arg : c10::irange(ntensors)) {
                data[arg] = data_[arg];
            }

            // 使用 memcpy 进行内存拷贝，从第一个数据指针拷贝到第二个数据指针，拷贝长度为 n * stride
            memcpy(data[0], data[1], n * stride);

            // 更新数据数组，以便下一次迭代
            for (const auto arg : c10::irange(data_arr.size())) {
                data_arr[arg] += outer_strides[arg];
            }
        }
    };

    // 获取用于循环的粒度大小
    int64_t grain_size = at::internal::GRAIN_SIZE;
    // 对迭代器应用 loop2d 函数
    iter.for_each(loop2d, grain_size);
    // 将输出数据转换为正确的类型
    iter.cast_outputs();
}

// 定义水平翻转掩码数组的大小
constexpr int64_t hflip_mask_size = 32;

// 生成水平翻转寄存器掩码的数组
std::array<char, hflip_mask_size> generate_vec_hflip_reg_mask(int64_t data_stride) {
    std::array<char, hflip_mask_size> mask;
    for (const auto k : c10::irange(hflip_mask_size / 2)) {
        // 计算掩码的值，根据数据步长进行调整
        int j = k / data_stride + 1;
        int v = (j * data_stride - 1) - (k % data_stride);
        v = std::min(v, (int) (hflip_mask_size / 2 - 1));
        mask[hflip_mask_size - 1 - k] = v;
        mask[hflip_mask_size / 2 - 1 - k] = v;
    }
    return mask;
}

// 执行基于通道末尾存储的向量化的 CPU 水平翻转操作
int64_t vectorized_cpu_hflip_channels_last(
    char * C10_RESTRICT *data, const int64_t data_size, const int64_t data_stride, const std::array<char, 32> & mdata) {

    int64_t i = 0;
#ifdef CPU_CAPABILITY_AVX2
// 检查是否支持 AVX2 指令集

  constexpr auto vec_size = 256 / 8;
  // 定义 AVX2 向量的大小为 256 bits，即 32 bytes

  if (data_size > vec_size) {
      // 如果数据大小大于向量大小

      // Example for num channels=3 and dtype=uint8
      // -> data_stride = 3
      // -> usable_vec_stride = 30
      // -> usable_vec_half_stride = 15
      // Data: (1 2 3) (4 5 6) (7 8 9) (10 11 12) (13 14 15) (16 17 18) (19 20 21) (22 23 24) (25 26 27) (28 29 30) (31 32 33)
      // load by 2 parts
      // R = [ (1 2 3) (4 5 6) (7 8 9) (10 11 12) (13 14 15) (16 | (16 17 18) (19 20 21) (22 23 24) (25 26 27) (28 29 30) (31 ]
      // flip(R) ->
      // R = [ 31 (28 29 30) (25 26 27) (22 23 24) (19 20 21) (16 17 18) | 16 (13 14 15) (10 11 12) (7 8 9) (4 5 6) (1 2 3) ]
      //
      // Write in 2 parts
      // Output pointer: output_ptr = data[0]                                                                                  v
      // - Init:
      //                (X X X)  (X X X)    (X X X)    (X X X)    (X X X)    (X X X)    (X X X)    (X X X)    (X X X) (X X X) (X X X)
      // 0) Move to initial position: output_ptr = data[0] + data_stride - vec_size / 2;
      //                                                                          v
      //                (X X X)  (X X X)    (X X X)    (X X X)    (X X X)    (X X X)    (X X X)    (X X X)    (X X X) (X X X) (X X X)
      // - In the loop:
      // 1) Write 1st block from output_ptr
      //                                                                            v
      //                                                                            |----> vec_size / 2 ---------------------------|
      // Output part 1: (X X X)  (X X X)    (X X X)    (X X X)    (X X X)     (X X 16)  (13 14 15) (10 11 12) (7 8 9) (4 5 6) (1 2 3)
      // 2) Write 2nd block from output_ptr - usable_vec_half_stride:
      //                                                                            v
      //                     |-----> vec_size / 2 ----------------------------------|
      // Output part 2: (X X 31) (28 29 30) (25 26 27) (22 23 24) (19 20 21) (16 17 18) (13 14 15) (10 11 12) (7 8 9) (4 5 6) (1 2 3)
      //
      // 3) Move to the next position: output_ptr -= usable_vec_stride
      //
      // - After the loop:
      // 4) Move to write position
      //                 v
      //                (X X 31) (28 29 30) (25 26 27) (22 23 24) (19 20 21) (16 17 18) (13 14 15) (10 11 12) (7 8 9) (4 5 6) (1 2 3)

    const __m256i mask = _mm256_loadu_si256((__m256i *) mdata.data());
    // 使用 AVX2 加载掩码数据

    const auto usable_vec_stride = 2 * (vec_size / 2 / data_stride) * data_stride;
    // 计算可用的向量步长，考虑数据步长和向量大小

    const auto usable_vec_half_stride = usable_vec_stride / 2;
    // 计算可用的一半向量步长

    auto output_ptr = data[0] + data_stride - vec_size / 2;
    // 设置输出指针的初始位置

    auto input_ptr = data[1];
    // 设置输入指针的初始位置
    // 循环遍历处理数据，每次增加步长 `usable_vec_stride` 直到 `data_size - vec_size`
    for (; i < data_size - vec_size; i += usable_vec_stride) {

      // 加载256位数据，分两次加载两个128位部分
      auto a0 = _mm_loadu_si128((__m128i *) (input_ptr + i));
      // 将128位数据扩展为256位数据
      auto b0 = _mm256_castsi128_si256(a0);
      // 加载下一个128位数据
      auto a1 = _mm_loadu_si128((__m128i *) (input_ptr + i + usable_vec_half_stride));
      // 将两个128位数据合并为一个256位数据
      auto data_vec = _mm256_inserti128_si256(b0, a1, 1);

      // 对合并后的256位数据进行按位反转操作，使用预定义的掩码 `mask`
      auto reversed_vec = _mm256_shuffle_epi8(data_vec, mask);

      // 将反转后的数据分成两部分写入输出
      auto rev_vec_h = _mm256_extracti128_si256(reversed_vec, 0);
      _mm_storeu_si128((__m128i *) (output_ptr - i), rev_vec_h); // 写入高128位部分到输出
      auto rev_vec_l = _mm256_extracti128_si256(reversed_vec, 1);
      _mm_storeu_si128((__m128i *) (output_ptr - i - usable_vec_half_stride), rev_vec_l); // 写入低128位部分到输出
    }

    // 调整数据数组中的元素
    data[0] -= i; // 减去循环中增加的步长 `i`
    data[1] += i; // 增加循环中增加的步长 `i`
  }
#endif
  // 直接返回当前循环计数器 i 的值
  return i;
}

void cpu_hflip_channels_last_vec(at::TensorIterator& iter) {

  // 获取输入张量的步长
  auto input_strides = iter.strides(1);
  const auto data_stride = input_strides[1];

  // 一次生成 AVX 掩码
  alignas(hflip_mask_size) auto mdata = generate_vec_hflip_reg_mask(data_stride);

  auto loop2d = [&](char** base, const int64_t *strides, int64_t size0, int64_t size1) {

    // 这里定义 ntensors 用于输出和一个输入。但是张量迭代器定义了输出、输入和重塑后的输入
    // 但我们仅使用输出和输入。
    static constexpr int ntensors = 2;
    const int64_t *outer_strides = &strides[3];
    const int64_t stride = strides[0];

    TORCH_INTERNAL_ASSERT(stride == strides[1]);

    auto c = -outer_strides[0];
    TORCH_INTERNAL_ASSERT(c == outer_strides[1]);

    // 创建数据数组，包含输出和输入的基址
    char* C10_RESTRICT data[ntensors] = {base[0], base[1]};
    const int64_t size = size0 * size1;

    int64_t i = 0;

    // 如果 c 在 2 到 16 之间，进行向量化水平翻转处理
    if (c >= 2 && c <= 16) {
      i = vectorized_cpu_hflip_channels_last(data, size * stride, c, mdata) / stride;
    }

    // 计算数据步长
    auto data_stride = size0 * stride;
    // 循环处理每个元素
    for (; i < size; i += size0) {

      // 使用 memcpy 复制数据
      memcpy(data[0], data[1], data_stride);

      // 向前移动数据指针
      for (const auto arg : c10::irange(ntensors)) {
        data[arg] += outer_strides[arg];
      }
    }

  };

  // 设置循环的粒度大小
  int64_t grain_size = at::internal::GRAIN_SIZE;
  // 对每个元素执行 loop2d 函数
  iter.for_each(loop2d, grain_size);
  // 将输出张量强制转换为正确类型
  iter.cast_outputs();
}

void flip_kernel(TensorIterator& iter, const bool quantized) {
  // 如果是量化类型的张量，使用 AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES 宏定义的 CPU 核心处理函数
  if (quantized) {
    AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(iter.dtype(), "flip_quantized_cpu",
        [&iter] { cpu_kernel(iter,
          [](scalar_t a, scalar_t /*dummy input*/) -> scalar_t {
            return a;
        });
    });
  } else {
    // 获取输出张量的步长
    auto output_strides = iter.strides(0);
    // 获取输入张量的步长
    auto input_strides = iter.strides(1);
    // 检查迭代器维度是否大于0，并且输出步长的第一个元素为负值，输入步长的第一个元素等于第二个元素的元素大小
    if (iter.ndim() > 0 && output_strides[0] == -iter.element_size(0) && input_strides[0] == iter.element_size(1)) {
      // 特殊情况：使用向量化进行水平翻转且输入是连续的
      // 上下文：水平翻转导致 strides[0] < 0，因此不满足 is_contiguous 条件，将采用非向量化的代码路径
      auto iter_dtype = iter.dtype();
      // 忽略半精度和 bfloat16，因为 cpu_hflip_vec 比 cpu_kernel_vec 更慢
      if (isIntegralType(iter_dtype, true) || iter_dtype == kDouble || iter_dtype == kFloat) {
        // 由于内部测试失败，将 AT_DISPATCH_ALL_TYPES_AND 替换为手动的 if/else：
        // - "dtype 'Float' not selected for kernel tag hflip_cpu"
        // - "dtype 'Long' not selected for kernel tag hflip_cpu"
        
        // 根据数据类型选择合适的函数进行水平翻转
        if (iter_dtype == kByte) {
          return cpu_hflip_vec<uint8_t>(iter);
        } else if (iter_dtype == kChar) {
          return cpu_hflip_vec<int8_t>(iter);
        } else if (iter_dtype == kInt) {
          return cpu_hflip_vec<int32_t>(iter);
        } else if (iter_dtype == kLong) {
          return cpu_hflip_vec<int64_t>(iter);
        } else if (iter_dtype == kShort) {
          return cpu_hflip_vec<int16_t>(iter);
        } else if (iter_dtype == kBool) {
          return cpu_hflip_vec<bool>(iter);
        } else if (iter_dtype == kFloat) {
          return cpu_hflip_vec<float>(iter);
        } else if (iter_dtype == kDouble) {
          return cpu_hflip_vec<double>(iter);
        }
      }
      // 其他数据类型（float16、bfloat16、复数）由 cpu_kernel_vec 处理（见下面的代码）
    } else if (iter.has_contiguous_first_dim()) {
      // 特殊情况：
      // a) 在 (N, C, H, W) 上进行通道最后的水平翻转，外部步长（= 数据类型大小 * C）在 [2, 16] 范围内
      // b) 在 (N, ..., M, C) 上对 dim=-2 进行翻转，外部步长（= 数据类型大小 * C）在 [2, 16] 范围内
      auto output_strides_2 = iter.strides(0);
      auto input_strides_2 = iter.strides(1);
      auto c = -output_strides_2[1];
      if (c >= 2 && c <= 16 &&
          c == input_strides_2[1] &&
          c == iter.element_size(0) * iter.shape()[0]  // 检查 dim=1 也是连续的
      ) {
        return cpu_hflip_channels_last_vec(iter);
      }
      // 特殊情况：使用 memcpy 进行垂直翻转（比通用的 cpu_kernel_vec 更快）
      return cpu_vflip_memcpy(iter);
    }

    // 使用 AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3 宏来处理所有数据类型（包括复数和特殊类型）
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kHalf, kBFloat16, iter.dtype(), "flip_cpu",
        [&iter] { cpu_kernel_vec(iter,
          [](scalar_t a, scalar_t /*dummy input*/) -> scalar_t {
            return a;
        },
          [](Vectorized<scalar_t> a, Vectorized<scalar_t> /*dummy input*/) -> Vectorized<scalar_t> {
            return a;
        });
    });
} // 结束匿名命名空间

} // 结束匿名命名空间

REGISTER_DISPATCH(index_stub, &index_kernel);
// 注册 index_stub 到 index_kernel 的调度器

REGISTER_DISPATCH(index_fill_stub, &index_fill_kernel);
// 注册 index_fill_stub 到 index_fill_kernel 的调度器

REGISTER_DISPATCH(index_copy_stub, &index_copy_kernel);
// 注册 index_copy_stub 到 index_copy_kernel 的调度器

REGISTER_DISPATCH(index_put_stub, &index_put_kernel);
// 注册 index_put_stub 到 index_put_kernel 的调度器

REGISTER_DISPATCH(put_stub, &put_kernel);
// 注册 put_stub 到 put_kernel 的调度器

REGISTER_DISPATCH(take_stub, &take_kernel);
// 注册 take_stub 到 take_kernel 的调度器

REGISTER_DISPATCH(masked_fill_stub, &masked_fill_kernel);
// 注册 masked_fill_stub 到 masked_fill_kernel 的调度器

REGISTER_DISPATCH(masked_select_serial_stub, &masked_select_serial_kernel);
// 注册 masked_select_serial_stub 到 masked_select_serial_kernel 的调度器

REGISTER_DISPATCH(masked_select_stub, &masked_select_kernel);
// 注册 masked_select_stub 到 masked_select_kernel 的调度器

REGISTER_DISPATCH(masked_scatter_stub, &masked_scatter_kernel);
// 注册 masked_scatter_stub 到 masked_scatter_kernel 的调度器

REGISTER_DISPATCH(flip_stub, &flip_kernel);
// 注册 flip_stub 到 flip_kernel 的调度器

} // 结束 at::native 命名空间
```