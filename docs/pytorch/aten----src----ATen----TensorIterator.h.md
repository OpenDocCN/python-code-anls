# `.\pytorch\aten\src\ATen\TensorIterator.h`

```py
// 防止头文件被多次包含，保证在单个编译单元中只包含一次
#pragma once

// 包含相关的 ATen 和 C10 头文件
#include <ATen/TensorMeta.h>
#include <ATen/core/Dimname.h>
#include <ATen/core/Range.h>
#include <ATen/core/TensorBase.h>
#include <c10/core/DynamicCast.h>
#include <c10/util/FunctionRef.h>
#include <c10/util/MaybeOwned.h>
#include <c10/util/SmallVector.h>
#include <c10/util/TypeCast.h>
#include <c10/util/irange.h>

#include <array>  // 包含标准库头文件 array
#include <bitset> // 包含标准库头文件 bitset

// 定义 at 命名空间，包含 Tensor 和相关类型
namespace at {
    class Tensor;
    class OptionalTensorRef;
    // 定义 NameVector 类型为 SmallVector，包含 Dimname 对象，静态大小为 kDimVectorStaticSize
    using NameVector = SmallVector<Dimname, kDimVectorStaticSize>;
} // namespace at

// TensorIterator 是用于元素逐个操作（如算术运算、比较、三角函数等）的辅助类。
// 它处理操作数的广播和类型转换。
//
// 受 NumPy 的数组迭代器 API（NpyIter）启发。
//
// Loops.h 和 Loops.cuh 文件提供了构建使用 TensorIterator 的内核函数。
//
// 示例：
//
//   auto iter = TensorIteratorConfig()
//     .add_output(output)
//     .add_input(input)
//     .build()
//
// [MyKernel.cpp / MyKernel.cu]
//   cpu_kernel(iter, [](float a, float b) {
//     return a + b;
//   });
//
//   gpu_kernel(iter, []GPU_LAMBDA(float a, float b) -> float {
//     return a + b;
//   });
//
// 注意 [构造顺序]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 在设置张量迭代器配置时，必须首先通过 TensorIteratorConfig::add_owned_output(at::Tensor) 添加输出张量。
// 添加所有输出后，可以通过 TensorIteratorConfig::add_owned_input(at::Tensor) 添加输入张量。
// 如果在添加输入之后再添加输出将引发异常。
//
// 注意 [通用 Dtype 计算]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 一些操作具有自然的“通用 dtype”或“计算 dtype”概念，在这些操作中，所有输入张量都转换为一个 dtype，
// 执行操作，然后将结果转换为所有输出张量的 dtype。
//
// 如果所有输入张量具有相同的 dtype，则 TensorIterator 推断出一个通用 dtype，
// 并且如果 promote_inputs_to_common_dtype_ 为 true，则使用其输入的类型提升规则计算它。
// 否则，尝试查询通用 dtype 将引发异常。
//
// 注意：在计算通用 dtype 时不考虑输出张量。

namespace at {

namespace internal {
// 此参数是启发式地选择的，用于确定最小化并行化工作的数量。
// 例如，在对数组求和时，认为长度小于 32768 的数组并行化效率低下。
// 没有并行算法（如 parallel_reduce）应将工作分割为小于 GRAIN_SIZE 的块。
constexpr int64_t GRAIN_SIZE = 32768;

// 用于非拥有 Tensor 的存储，而不需要包含 Tensor.h 头文件
class TORCH_API OpaqueOptionalTensorRef {
  // 使用 alignas 确保 data_ 成员与 TensorBase 对齐，使用 char 数组存储数据
  alignas(alignof(TensorBase)) std::array<char, sizeof(TensorBase)> data_{};

 public:
  // 默认构造函数
  OpaqueOptionalTensorRef();
  // 复制构造函数
  OpaqueOptionalTensorRef(const OpaqueOptionalTensorRef&) = default;
  // 复制赋值运算符
  OpaqueOptionalTensorRef& operator=(const OpaqueOptionalTensorRef&) = default;
  // 移动构造函数
  OpaqueOptionalTensorRef(OpaqueOptionalTensorRef&&) noexcept = default;
  // 移动赋值运算符
  OpaqueOptionalTensorRef& operator=(OpaqueOptionalTensorRef&&) noexcept =
      default;
  // 析构函数
  ~OpaqueOptionalTensorRef();

  // 获取可选的 OptionalTensorRef 指针
  OptionalTensorRef* get() {
    return reinterpret_cast<OptionalTensorRef*>(data_.data());
  }
  // 获取常量的 OptionalTensorRef 指针
  const OptionalTensorRef* get() const {
    return reinterpret_cast<const OptionalTensorRef*>(data_.data());
  }

  // 解引用操作符，返回 OptionalTensorRef 的引用
  OptionalTensorRef& operator*() {
    return *get();
  }
  // 常量版本的解引用操作符，返回常量 OptionalTensorRef 的引用
  const OptionalTensorRef& operator*() const {
    return *get();
  }
  // 成员访问操作符重载，返回 OptionalTensorRef 指针
  OptionalTensorRef* operator->() {
    return get();
  }
  // 常量版本的成员访问操作符重载，返回常量 OptionalTensorRef 指针
  const OptionalTensorRef* operator->() const {
    return get();
  }

  // 获取存储的 Tensor 对象的常量引用
  const Tensor& getTensor() const;
};
} // namespace internal

struct TORCH_API OperandInfo {
  // 定义 StrideVector 类型为 SmallVector<int64_t, 6>
  using StrideVector = SmallVector<int64_t, 6>;
  // 默认构造函数
  OperandInfo() = default;
  // 移动构造函数，使用 c10::MaybeOwned<TensorBase> 参数
  C10_ALWAYS_INLINE explicit OperandInfo(c10::MaybeOwned<TensorBase>&& t) {
    // 如果 TensorBase 对象有效，则初始化 device、target_dtype 和 current_dtype
    if (t->defined()) {
      device = t->device();
      target_dtype = t->scalar_type();
      current_dtype = target_dtype;
    }
    // 移动构造 TensorBase 对象
    tensor(std::move(t));
    // 验证操作信息的有效性
    validate();
  }

  // 复制构造函数
  C10_ALWAYS_INLINE OperandInfo(const OperandInfo&) = default;
  // 复制赋值运算符
  C10_ALWAYS_INLINE OperandInfo& operator=(const OperandInfo&) = default;
  // 移动构造函数
  C10_ALWAYS_INLINE OperandInfo(OperandInfo&&) noexcept = default;
  // 移动赋值运算符
  C10_ALWAYS_INLINE OperandInfo& operator=(OperandInfo&&) noexcept = default;
  // 析构函数
  C10_ALWAYS_INLINE ~OperandInfo() = default;

  /// 数据指针。如果迭代器被分割，这可能与 tensor->data_ptr() 不同。
  void* data = nullptr;

  /// 广播后的步幅。步幅以字节为单位，而不是元素个数。
  StrideVector stride_bytes;

  /// 操作数的目标设备和类型。对于输入，指定如有必要将输入转换为此类型。
  /// 对于输出，指定要分配的类型。target_dtype 和 device 初始值为 tensor 的 dtype 和 device，
  /// 但在类型提升期间 target_dtype 可能会与 tensor 的 dtype 不同，还可能会为未定义的 tensor 设置
  /// target_dtype 和 device，以便稍后正确构建 tensor。
  std::optional<Device> device = c10::nullopt;
  ScalarType target_dtype = ScalarType::Undefined;
  // 缓存 tensor 的 dtype，因为 scalar_type 是昂贵的操作。如果 tensor 的 dtype 更改（例如由于类型提升或在
  // allocate_outputs 中），此值也应更改。
  ScalarType current_dtype = ScalarType::Undefined;

  // 返回设备是否已定义
  bool is_device_defined() const {
    return device.has_value();
  }
  // 返回类型是否已定义
  bool is_type_defined() const {
  // 返回是否目标数据类型不是未定义类型
  return target_dtype != ScalarType::Undefined;
}

TensorOptions options() const {
  // 返回一个包含指定目标数据类型和设备的TensorOptions对象
  return TensorOptions(target_dtype).device(device);
}

bool is_output = false;

// will_resize仅适用于输出张量。
// 1) 函数调用（如torch.add(self, other)）：输出张量未定义，PyTorch使用TensorIterator中的公共形状和计算的步幅创建新张量；
// 2) 原地调用（如torch.add_(self, other)）：输出张量与输入张量相同，不能修改张量的大小和步幅；
// 3) 具有输出的操作调用（如torch.add(self, other, out=output)）：
//    输出张量已定义，但张量形状可能与公共形状不同。如果张量形状与公共形状不同，将使用TensorIterator中的公共形状和计算的步幅调整此输出张量的大小。否则，不能修改张量的大小和步幅。
bool will_resize = false;

bool is_read_write = false;

bool is_const = false;

void validate() {
  // 检查张量基础是否未定义或布局为kStrided，否则抛出异常
  TORCH_CHECK(
      !tensor_base_->defined() || tensor_base_->layout() == kStrided,
      "unsupported tensor layout: ",
      tensor_base_->layout());
}

/// 张量操作数。注意，由于维度重新排序和合并，步幅、数据指针和其他属性可能不同。
const Tensor& tensor() const {
  // 返回存储在tensor_storage_中的张量对象的引用
  return tensor_storage_.getTensor();
}
const TensorBase& tensor_base() const {
  // 返回tensor_base_指向的TensorBase对象的引用
  return *tensor_base_;
}
void tensor(c10::MaybeOwned<TensorBase>&& tensor);

// 在输出被修改时保存原始张量操作数（例如如果dtype被更改）
const Tensor& original_tensor() const {
  // 返回存储在original_tensor_storage_中的原始张量对象的引用
  return original_tensor_storage_.getTensor();
}
const TensorBase& original_tensor_base() const {
  // 返回original_tensor_base_指向的TensorBase对象的引用
  return *original_tensor_base_;
}

// 设置张量为新值，并将旧张量值存储在original_tensor中。一个操作数的生命周期内只能调用一次。
void exchange_tensor(c10::MaybeOwned<TensorBase>&& new_tensor);

// 将原始张量移回张量中，exchange_tensor必须在此之前调用过。
void restore_original_tensor();

private:
c10::MaybeOwned<TensorBase> tensor_base_;
c10::MaybeOwned<TensorBase> original_tensor_base_ =
    c10::MaybeOwned<TensorBase>::owned(std::in_place);

// 我们在头文件中明确存储TensorBase，以允许内联访问。
// 然而，由于TensorIterator API，有时我们需要一个真正的`const Tensor &`。
// 因此，我们还在这些`_storage_`变量中存储一个非拥有的`Tensor`对象。
internal::OpaqueOptionalTensorRef tensor_storage_;
internal::OpaqueOptionalTensorRef original_tensor_storage_;
  // 结构体 SplitUntil32Bit 的前向声明
  struct SplitUntil32Bit;

  // 枚举类型 FastSetupType，用于快速设置类型，基础类型为 uint8_t
  enum class FastSetupType : uint8_t {
    NONE,
    CONTIGUOUS,
    CHANNELS_LAST,
    NON_OVERLAPPING_DENSE
  };

  // 类 TensorIteratorConfig 的前向声明
  class TensorIteratorConfig;

  // 结构体 TensorIterator 的前向声明
  struct TensorIterator;

  // TensorIteratorBase 类，继承自 impl::MetaBase
  struct TORCH_API TensorIteratorBase : public impl::MetaBase {
    // 使用 std::bitset 定义 DimMask，长度为 64
    using DimMask = std::bitset<64>;
    // 使用 SmallVector 定义 PtrVector，存储 char* 指针，长度为 4
    using PtrVector = SmallVector<char*, 4>;
    // 使用 SmallVector 定义 StrideVector，存储 int64_t，长度为 6
    using StrideVector = SmallVector<int64_t, 6>;

    // 默认构造函数
    TensorIteratorBase();
    
    // 根据配置构建 TensorIterator
    void build(TensorIteratorConfig&);

    // 内部循环函数类型定义，处理最快移动的维度，实现基于 1-d strided tensors 的逐元素操作
    //
    // 参数：
    //   data: 每个操作数的数据指针数组（长度为 ntensors）
    //   strides: 每个操作数的步长（长度为 ntensors）
    //   size: 内部循环的大小
    //
    // size 通常与 shape[0] 相匹配，但由于内部循环的并行化可能会更小
    using loop2d_t = c10::function_ref<void(char** data, const int64_t* strides, int64_t size0, int64_t size1)>;

    // 子迭代器循环类型定义
    using loop_subiter_t = c10::function_ref<void(TensorIteratorBase& subiter)>;

    // 对每个缩减元素执行循环操作
    void foreach_reduced_elt(loop_subiter_t loop, bool parallelize = true);

    // 返回维度数
    int ndim() const {
      return static_cast<int>(shape_.size());
    }

    // 返回形状
    IntArrayRef shape() const {
      return shape_;
    }

    // 返回元素总数
    int64_t numel() const;

    // 返回操作数数量
    int ntensors() const {
      return static_cast<int>(operands_.size());
    }

    // 返回输出数目
    int noutputs() const {
      return num_outputs_;
    }

    // 返回输入数目
    int ninputs() const {
      return ntensors() - noutputs();
    }

    // 返回视图偏移量
    IntArrayRef view_offsets() const {
      return view_offsets_;
    }

    // 返回输出操作数的元素数量，对于非缩减操作等同于 numel()
    int64_t num_output_elements() const;

    // 返回缩减操作中的缩减维度数
    int num_reduce_dims() const;

    // 判断是否为一维迭代且无缓冲或类型转换
    bool is_trivial_1d() const;

    // 判断是否可缩减为一维且所有操作数都是连续的
    bool is_contiguous() const;

    // 判断指定维度是否已被缩减
    bool is_dim_reduced(int dim) const;

    // 访问指定操作数的步长数组
    IntArrayRef strides(int64_t arg) const {
      return operands_[arg].stride_bytes;
    }

    // 访问指定操作数的数据指针
    void* data_ptr(int64_t arg) const;

    // 返回指定操作数的数据类型
    ScalarType dtype(int64_t arg = 0) const {
      return operands_[arg].current_dtype;
    }

    // 返回公共数据类型
    ScalarType common_dtype() const {
      TORCH_INTERNAL_ASSERT(
          common_dtype_ != ScalarType::Undefined,
          "Queried for invalid common dtype!");
      return common_dtype_;
    }

    // 返回输入操作数的数据类型
    ScalarType input_dtype(int64_t arg = 0) const {
      return operands_[num_outputs_ + arg].current_dtype;
    }

    // 返回指定操作数的设备
    Device device(int64_t arg = 0) const {
      return operands_[arg].device.value();
    }

    // 返回指定操作数的设备类型
    c10::DeviceType device_type(int64_t arg = 0) const {
      return device(arg).type();
    }

    // 返回指定操作数的元素大小
    int64_t element_size(int64_t arg) const {
      return static_cast<int64_t>(elementSize(dtype(arg)));
    }

    // 判断指定操作数是否为标量
    bool is_scalar(int64_t arg) const;

    // 判断指定操作数是否为 CPU 标量
    bool is_cpu_scalar(int64_t arg) const;

    // 返回指定操作数的 TensorBase 引用
    const TensorBase& tensor_base(int64_t arg) const {
    // 返回操作数数组中第 arg 个操作数的基础张量对象
    return operands_[arg].tensor_base();
  }
  // 返回操作数数组中第 arg 个操作数的张量对象（常量版本）
  const Tensor& tensor(int64_t arg) const {
    return operands_[arg].tensor();
  }

  // 返回输出张量的基础张量对象（常量版本）
  const TensorBase& output_base(int64_t arg = 0) const {
    AT_ASSERT(arg < num_outputs_);
    return tensor_base(arg);
  }

  // 返回输出张量对象（常量版本）
  const Tensor& output(int64_t arg = 0) const {
    AT_ASSERT(arg < num_outputs_);
    return tensor(arg);
  }

  // 返回输入张量的基础张量对象（常量版本）
  const TensorBase& input_base(int64_t arg = 0) const {
    AT_ASSERT(arg >= 0 && arg < ntensors() - num_outputs_);
    return tensor_base(num_outputs_ + arg);
  }

  // 返回输入张量对象（常量版本）
  const Tensor& input(int64_t arg = 0) const {
    AT_ASSERT(arg >= 0 && arg < ntensors() - num_outputs_);
    return tensor(num_outputs_ + arg);
  }

  // 将临时输出复制回原始输出
  // 注意：仅在 CPU 上使用
  void cast_outputs();

  /// 从迭代器中移除指定的操作数
  void remove_operand(int64_t arg);

  /// 缩小迭代的维度范围
  void narrow(int dim, int64_t start, int64_t size);

  /// 将从 start_dim 开始的每个维度缩小为 size 为 1
  void select_all_keeping_dim(int start_dim, IntArrayRef starts);

  /// 替换指定操作数的数据指针
  /// 新的指针应该与原始数据具有相同的大小、步长和数据类型
  void unsafe_replace_operand(int64_t arg, void* data);

  /// 将这个 TensorIterator 拆分为两个迭代器
  /// 它们一起迭代整个操作，由 `with_32bit_indexing()` 使用
  std::unique_ptr<TensorIterator> split(int dim);

  /// 返回最大范围的维度：(size[dim]-1) * stride[dim]
  int get_dim_to_split() const;

  template <typename T>
  // 返回第 arg 个操作数的标量值
  T scalar_value(int64_t arg) {
    auto& op = operands_[arg];
    return c10::fetch_and_cast<T>(op.tensor_base().scalar_type(), op.data);
  }

  /// 如果定义了 original_tensor_base，则从中返回标量值
  /// 当 common_dtype 是 Half 时，将标量输入强制转换为 common_dtype 可能会溢出
  /// 如果标量已经是 Half 类型，则从 tensor_base 返回其标量值
  template <typename T>
  T original_scalar_value(int64_t arg) {
    auto& original_tensor_base = operands_[arg].original_tensor_base();
    if (original_tensor_base.defined()) {
      TORCH_INTERNAL_ASSERT(
          original_tensor_base.scalar_type() != common_dtype());
      return c10::fetch_and_cast<T>(
          original_tensor_base.scalar_type(),
          original_tensor_base.const_data_ptr());
    } else {
      return scalar_value<T>(arg);
    }
  }

 private:
  // 从 1D 循环对象中创建 2D 循环对象
  template <typename loop1d_t>
  auto loop_2d_from_1d(const loop1d_t& loop) {
    // 返回空，没有返回值
    return
        // 定义 lambda 函数，参数为 loop，ntensor 为调用 ntensors() 函数的结果
        [loop, ntensor = ntensors()](
            // base 为 char 指针数组，strides 为 int64_t 型指针，size0 和 size1 为 int64_t 型参数
            char** base, const int64_t* strides, int64_t size0, int64_t size1) {
          // 使用 base 和 ntensor 创建 PtrVector 对象 data
          PtrVector data(base, base + ntensor);
          // 获取 outer_strides 指向 strides 数组中第 ntensor 个元素的指针
          const int64_t* outer_strides = &strides[ntensor];
          // 遍历 size1 次数，其中 i 为迭代变量
          for (const auto i : c10::irange(size1)) {
            // 如果 i 大于 0，则执行以下代码块
            if (i > 0) {
              // 对于 data 中的每个元素执行操作，加上对应的 outer_strides 值
              for (const auto arg : c10::irange(ntensor)) {
                data[arg] += outer_strides[arg];
              }
            }
            // 调用 loop 函数，传入 data 的数据指针，strides，size0
            loop(data.data(), strides, size0);
          }
        };
    }
    
    public:
    // 模板函数，接受 loop1d_t 类型参数 loop 和 grain_size 参数，默认值为 at::internal::GRAIN_SIZE
    template <
        typename loop1d_t,
        // 如果 loop1d_t 可转换为 c10::function_ref<void(char**, const int64_t* strides, int64_t size)>，则启用此模板
        std::enable_if_t<
            std::is_convertible_v<
                loop1d_t,
                c10::function_ref<
                    void(char**, const int64_t* strides, int64_t size)>>,
            int> = 0>
    void for_each(loop1d_t loop, int64_t grain_size = at::internal::GRAIN_SIZE) {
      // 调用 for_each 函数，传入 loop_2d_from_1d(loop) 和 grain_size
      for_each(loop_2d_from_1d(loop), grain_size);
    }
    
    // 声明 for_each 函数，接受 loop2d_t 类型参数 loop 和 grain_size 参数
    void for_each(loop2d_t loop, int64_t grain_size = at::internal::GRAIN_SIZE);
    
    // 声明 parallel_reduce 函数，接受 loop2d_t 类型参数 loop
    void parallel_reduce(loop2d_t loop);
    
    // 模板函数，接受 loop1d_t 类型参数 loop 和 Range 类型参数 range
    template <
        typename loop1d_t,
        // 如果 loop1d_t 可转换为 c10::function_ref<void(char**, const int64_t* strides, int64_t size)>，则启用此模板
        std::enable_if_t<
            std::is_convertible_v<
                loop1d_t,
                c10::function_ref<
                    void(char**, const int64_t* strides, int64_t size)>>,
            int> = 0>
    void serial_for_each(loop1d_t loop, Range range) {
      // 调用 serial_for_each 函数，传入 loop_2d_from_1d(loop) 和 range
      serial_for_each(loop_2d_from_1d(loop), range);
    }
    
    // 声明 serial_for_each 函数，接受 loop2d_t 类型参数 loop 和 Range 类型参数 range
    void serial_for_each(loop2d_t loop, Range range) const;
    
    /// 创建一个与此迭代器形状相匹配的 Tensor 的 strides 数组。参数 `element_size` 指定 Tensor 数据类型的字节大小（例如 `float` 的字节数为 `4`）
    StrideVector compatible_stride(int64_t element_size) const;
    
    /// 反转由 reorder_dimensions 所做的重新排序操作。仅在调用 coalesce_dimensions() 之前可调用。
    DimVector invert_perm(IntArrayRef input) const;
    
    /// 重新应用与 reorder_dimensions 相同的重新排序操作。仅在调用 coalesce_dimensions() 之前可调用。
    DimVector apply_perm_and_mul(IntArrayRef input, int mul) const;
    
    /// 获取指定维度的 strides 数组，用于 CPU 迭代
    StrideVector get_dim_strides(int dim) const;
    /// 获取当前对象的 strides 数组
    StrideVector get_strides() const;
    /// 获取内部维度的 strides 数组，等同于 get_dim_strides(0)
    StrideVector get_inner_strides() const {
      return get_dim_strides(0);
    }
    /// 获取基本指针数组
    PtrVector get_base_ptrs() const;
    
    // 为高级 strides 操作（例如 torch.flip）提供的辅助函数
    void _unsafe_set_arg_strides(const int64_t arg, IntArrayRef strides) {
      operands_[arg].stride_bytes = strides;
    }
    void _unsafe_set_arg_data(const int64_t arg, void* data) {
      operands_[arg].data = data;
    }
    
    // 为自定义设备提供的辅助函数，自定义设备可以获取 OperandInfo 和 NameVector
    const OperandInfo& operand(int arg = 0) const {
      return operands_[arg];
    }
    OperandInfo& operand(int arg = 0) {
      return operands_[arg];
    }
    NameVector& get_dim_names() {
      return names_;
    }
    const NameVector& get_dim_names() const {
  // 返回成员变量 names_
  return names_;
}

/// true if the stride computation can use 32-bit arithmetic. Used by GPU
/// kernels
bool can_use_32bit_indexing() const;

/// An "iteratable" object that recursively splits this iterator into
/// sub-iterators that can use 32-bit indexing.
SplitUntil32Bit with_32bit_indexing() const;

/// If the kernel should accumulate into the output. Only relevant for CUDA
/// reductions.
bool should_accumulate() const {
  return accumulate_;
}

/// Whether this iterator produces the actual output,
/// as opposed to something that will be accumulated further. Only relevant
/// for CUDA reductions.
bool is_final_output() const {
  return final_output_;
}

bool has_contiguous_first_dim() const {
  // 如果维度数为0，直接返回true
  if (ndim() == 0) {
    return true;
  }

  // 获取张量的数量
  int num_tensors = ntensors();
  // 遍历每个张量
  for (const auto i : c10::irange(num_tensors)) {
    // 检查当前张量的第一个维度是否为其元素大小
    if (strides(i)[0] != element_size(i)) {
      return false;
    }
  }
  // 所有张量的第一个维度都相同，返回true
  return true;
}

void set_output_raw_strided(
    int64_t output_idx,
    IntArrayRef sizes,
    IntArrayRef strides,
    TensorOptions options,
    DimnameList names) override;
// 定义宏 TORCH_DISALLOW_TEMPORARIES_IMPL，用于禁止临时对象的方法重载
#define TORCH_DISALLOW_TEMPORARIES_IMPL(methodname, maybestatic)            \
  // 删除以下各种情况的重载：
  maybestatic void methodname(                                              \
      TensorBase&& out, const TensorBase& a, const TensorBase& b) = delete; \
  maybestatic void methodname(                                              \
      const TensorBase& out, TensorBase&& a, const TensorBase& b) = delete; \
  maybestatic void methodname(                                              \
      const TensorBase& out, const TensorBase& a, TensorBase&& b) = delete; \
  maybestatic void methodname(                                              \
      TensorBase&& out, TensorBase&& a, const TensorBase& b) = delete;      \
  maybestatic void methodname(                                              \
      TensorBase&& out, const TensorBase& a, TensorBase&& b) = delete;      \
  maybestatic void methodname(                                              \
      const TensorBase& out, TensorBase&& a, TensorBase&& b) = delete;      \
  maybestatic void methodname(                                              \
      TensorBase&& out, TensorBase&& a, TensorBase&& b) = delete;
// 定义宏 TORCH_DISALLOW_TEMPORARIES，用于禁止临时变量
#define TORCH_DISALLOW_TEMPORARIES(methodname) \
  TORCH_DISALLOW_TEMPORARIES_IMPL(methodname, )

// 声明一个函数 build_binary_float_op，用于构建二进制浮点运算的操作
void build_binary_float_op(
    const TensorBase& out,
    const TensorBase& a,
    const TensorBase& b);

// 声明一个函数 build_borrowing_binary_float_op，用于构建借用方式的二进制浮点运算的操作
void build_borrowing_binary_float_op(
    const TensorBase& out,
    const TensorBase& a,
    const TensorBase& b);

// 应用宏 TORCH_DISALLOW_TEMPORARIES 禁止 build_borrowing_binary_float_op 函数使用临时变量
TORCH_DISALLOW_TEMPORARIES(build_borrowing_binary_float_op)

// 声明一个函数 build_binary_op，用于构建二进制操作的操作
void build_binary_op(
    const TensorBase& out,
    const TensorBase& a,
    const TensorBase& b);

// 声明一个函数 build_borrowing_binary_op，用于构建借用方式的二进制操作的操作
void build_borrowing_binary_op(
    const TensorBase& out,
    const TensorBase& a,
    const TensorBase& b);

// 应用宏 TORCH_DISALLOW_TEMPORARIES 禁止 build_borrowing_binary_op 函数使用临时变量
TORCH_DISALLOW_TEMPORARIES(build_borrowing_binary_op)

// 声明一个函数 build_unary_float_op，用于构建一元浮点操作的操作
void build_unary_float_op(const TensorBase& out, const TensorBase& a);

// 声明一个函数 build_borrowing_unary_float_op，用于构建借用方式的一元浮点操作的操作
void build_borrowing_unary_float_op(
    const TensorBase& out,
    const TensorBase& a);

// 应用宏 TORCH_DISALLOW_TEMPORARIES 禁止 build_borrowing_unary_float_op 函数使用临时变量
TORCH_DISALLOW_TEMPORARIES(build_borrowing_unary_float_op)

// 声明一个函数 build_unary_op，用于构建一元操作的操作
void build_unary_op(const TensorBase& out, const TensorBase& a);

// 注释特殊情况：pow 函数需要借用输出，因为它是结构化内核，但参数可能是副本。
// 声明一个函数 build_output_borrowing_argument_owning_unary_op，用于构建借用方式的一元操作的操作
void build_output_borrowing_argument_owning_unary_op(
    const TensorBase& out,
    const TensorBase& a);

// 声明一个函数 build_borrowing_unary_op，用于构建借用方式的一元操作的操作
void build_borrowing_unary_op(const TensorBase& out, const TensorBase& a);

// 应用宏 TORCH_DISALLOW_TEMPORARIES 禁止 build_borrowing_unary_op 函数使用临时变量
TORCH_DISALLOW_TEMPORARIES(build_borrowing_unary_op)

// 声明一个函数 build_borrowing_unary_force_boolean_op，用于构建借用方式的强制布尔一元操作的操作
void build_borrowing_unary_force_boolean_op(
    const TensorBase& out,
    const TensorBase& a);

// 应用宏 TORCH_DISALLOW_TEMPORARIES 禁止 build_borrowing_unary_force_boolean_op 函数使用临时变量
TORCH_DISALLOW_TEMPORARIES(build_borrowing_unary_force_boolean_op)

// 声明一个函数 build_comparison_op，用于构建比较操作的操作
void build_comparison_op(
    const TensorBase& out,
    const TensorBase& a,
    const TensorBase& b);

// 声明一个函数 build_borrowing_comparison_op，用于构建借用方式的比较操作的操作
void build_borrowing_comparison_op(
    const TensorBase& out,
    const TensorBase& a,
    const TensorBase& b);

// 应用宏 TORCH_DISALLOW_TEMPORARIES 禁止 build_borrowing_comparison_op 函数使用临时变量
TORCH_DISALLOW_TEMPORARIES(build_borrowing_comparison_op)

// 注释特殊情况：对于比较操作，我们需要拥有第二个参数，因为它是比较运算符。
// 声明一个函数 build_borrowing_except_last_argument_comparison_op，用于构建借用方式的比较操作的操作
void build_borrowing_except_last_argument_comparison_op(
    const TensorBase& out,
    const TensorBase& a,
    const TensorBase& b);

// 声明一个函数 build_ternary_op，用于构建三元操作的操作
void build_ternary_op(
    const TensorBase& out,
    const TensorBase& a,
    const TensorBase& b,
    const TensorBase& c);

};

// 定义结构体 TensorIterator，继承自 TensorIteratorBase
struct TORCH_API TensorIterator final : public TensorIteratorBase {
  // 默认构造函数，调用基类构造函数
  TensorIterator() : TensorIteratorBase() {}

  // 允许通过切片构造 TensorIterator，保证其不具有任何字段
  // 调用基类构造函数以初始化
  TensorIterator(const TensorIteratorBase& iter) : TensorIteratorBase(iter) {}
#define TORCH_DISALLOW_TEMPORARIES(methodname) \
  TORCH_DISALLOW_TEMPORARIES_IMPL(methodname, static)

// 定义宏TORCH_DISALLOW_TEMPORARIES，它调用TORCH_DISALLOW_TEMPORARIES_IMPL宏，设定为静态方法

  static TensorIterator binary_float_op(
      TensorBase& out,
      const TensorBase& a,
      const TensorBase& b);
  // 声明静态方法binary_float_op，接受三个TensorBase参数并返回TensorIterator对象

  static TensorIterator binary_op(
      TensorBase& out,
      const TensorBase& a,
      const TensorBase& b);
  // 声明静态方法binary_op，接受三个TensorBase参数并返回TensorIterator对象

  static TensorIterator borrowing_binary_op(
      const TensorBase& out,
      const TensorBase& a,
      const TensorBase& b);
  // 声明静态方法borrowing_binary_op，接受三个常量TensorBase参数并返回TensorIterator对象

  TORCH_DISALLOW_TEMPORARIES(borrowing_binary_op)
  // 使用TORCH_DISALLOW_TEMPORARIES宏阻止使用borrowing_binary_op方法创建临时对象

  static TensorIterator comparison_op(
      TensorBase& out,
      const TensorBase& a,
      const TensorBase& b);
  // 声明静态方法comparison_op，接受三个TensorBase参数并返回TensorIterator对象

  static TensorIterator unary_op(TensorBase& out, const TensorBase& a);
  // 声明静态方法unary_op，接受两个TensorBase参数并返回TensorIterator对象

  static TensorIterator unary_float_op(TensorBase& out, const TensorBase& a);
  // 声明静态方法unary_float_op，接受两个TensorBase参数并返回TensorIterator对象

  static TensorIterator nullary_op(TensorBase& out);
  // 声明静态方法nullary_op，接受一个TensorBase参数并返回TensorIterator对象

  static TensorIterator borrowing_nullary_op(const TensorBase& out);
  // 声明静态方法borrowing_nullary_op，接受一个常量TensorBase参数并返回TensorIterator对象

  static TensorIterator borrowing_nullary_op(TensorBase&& out) = delete;
  // 声明删除版本的静态方法borrowing_nullary_op，接受一个右值引用TensorBase参数，不可用

  static TensorIterator reduce_op(TensorBase& out, const TensorBase& a);
  // 声明静态方法reduce_op，接受两个TensorBase参数并返回TensorIterator对象

  static TensorIterator reduce_op(
      TensorBase& out1,
      TensorBase& out2,
      const TensorBase& a);
  // 声明重载的静态方法reduce_op，接受三个TensorBase参数并返回TensorIterator对象

#undef TORCH_DISALLOW_TEMPORARIES
#undef TORCH_DISALLOW_TEMPORARIES_IMPL

  const Tensor& maybe_get_output(int64_t output_idx) override;
  // 重写maybe_get_output方法，返回const Tensor对象，根据输出索引获取输出

  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override;
  // 重写set_output_raw_strided方法，设置输出张量的原始步幅，尺寸，选项和维度名列表
};

class TORCH_API TensorIteratorConfig final {
 public:
  friend struct TensorIteratorBase;
  friend struct TensorIterator;

  TensorIteratorConfig() = default;
  // 默认构造函数

  C10_DISABLE_COPY_AND_ASSIGN(TensorIteratorConfig);
  // 禁用复制和赋值构造函数

  /// Construction
  // Stores input/output Tensors without incrementing the reference count.
  // Important: the outputs have to be added before the inputs.
  TensorIteratorConfig& add_output(const TensorBase& output) {
    return add_borrowed_output(output);
  }
  // 添加输出张量的配置，不增加引用计数，返回配置对象的引用

  TensorIteratorConfig& add_input(const TensorBase& input) {
    return add_borrowed_input(input);
  }
  // 添加输入张量的配置，不增加引用计数，返回配置对象的引用

  TensorIteratorConfig& add_const_input(const TensorBase& input) {
    // 添加常量输入张量的配置，不增加引用计数，返回配置对象的引用

    return add_borrowed_input(input);
  }
  // 添加常量输入张量的配置，不增加引用计数，返回配置对象的引用
  // 返回带有借用常量输入的结果
  return add_borrowed_const_input(input);
}

// 借用临时变量通常不会顺利进行。
TensorIteratorConfig& add_output(TensorBase&& output) = delete;
TensorIteratorConfig& add_input(TensorBase&& input) = delete;
TensorIteratorConfig& add_const_input(TensorBase&& input) = delete;

// 存储输入/输出张量并增加引用计数。
// 注意，几乎总是使用 add_{in,out}put 方法，例外情况（添加未命名临时变量）不会编译通过。
TensorIteratorConfig& add_owned_output(const TensorBase& output);
TensorIteratorConfig& add_owned_input(const TensorBase& input);
TensorIteratorConfig& add_owned_const_input(const TensorBase& input);

// 高级 API：存储输入/输出张量，但不增加引用计数。
// 调用者必须确保这些张量至少与 TensorIteratorConfig 及任何从该 TensorIteratorConfig 构建的 TensorIteratorBase 一样长。
// 重要提示：输出必须在输入之前添加。
TensorIteratorConfig& add_borrowed_output(const TensorBase& output);
TensorIteratorConfig& add_borrowed_input(const TensorBase& input);
TensorIteratorConfig& add_borrowed_const_input(const TensorBase& input);

// 借用临时变量通常不会顺利进行。
TensorIteratorConfig& add_borrowed_output(TensorBase&& output) = delete;
TensorIteratorConfig& add_borrowed_input(TensorBase&& input) = delete;
TensorIteratorConfig& add_borrowed_const_input(TensorBase&& input) = delete;

// 设置 check_mem_overlap_ 标志，默认为 true。
// 如果为 true，则检查输入与输出之间的部分重叠以及输出之间的内部重叠（例如广播视图）。
// 如果检测到不可接受的重叠，则引发错误。
// 如果要将现有操作符迁移到使用 TensorIterator，请考虑以前的实现是否检查了内存重叠。
// 如果没有，并且操作符是幂等的（例如 Tensor.fill_(0)），则检查内存重叠是 BC-breaking 的。
// 在这种情况下，请不要检查内存重叠。
TensorIteratorConfig& set_check_mem_overlap(bool check_mem_overlap) {
  check_mem_overlap_ = check_mem_overlap;
  return *this;
}

// 设置 check_all_same_dtype_ 标志，默认为 true。
// 如果为 true，则检查所有输入和定义的输出是否具有相同的数据类型。
// 如果将 promote_inputs_to_common_dtype_ 或 cast_common_dtype_to_outputs_ 中的任何一个设置为 true，将会将 check_all_same_dtype_ 设置为 false。
TensorIteratorConfig& check_all_same_dtype(const bool _check_all_same_dtype) {
  check_all_same_dtype_ = _check_all_same_dtype;
  // 返回当前对象的引用，用于支持链式调用
  return *this;
}

// 设置 check_all_same_device_ 标志，默认为 true
// 如果为 true，则所有操作数必须位于同一设备上，可能的例外是可以作为 CUDA 内核参数传递的 CPU 标量
TensorIteratorConfig& check_all_same_device(
    const bool _check_all_same_device) {
  check_all_same_device_ = _check_all_same_device;
  return *this;
}

// 设置 enforce_safe_casting_to_output_ 标志，默认为 false
// 如果为 true，则迭代器的“公共数据类型”必须可计算（参见[Common Dtype Computation]注释）
// 并且对于所有输出，canCast(公共数据类型, 输出数据类型) 必须为 true
TensorIteratorConfig& enforce_safe_casting_to_output(
    const bool _enforce_safe_casting_to_output) {
  enforce_safe_casting_to_output_ = _enforce_safe_casting_to_output;
  return *this;
}

// 设置 enforce_linear_iteration_ 标志，默认为 false
// 如果为 true，则迭代顺序与内存中的 C 连续张量的布局相同，即最后一维的迭代最快
// 此迭代顺序可能效率较低，甚至可能阻止向量化，只有在您的内核的正确性依赖于此时才使用
TensorIteratorConfig& enforce_linear_iteration(
    const bool _enforce_linear_iteration = true) {
  enforce_linear_iteration_ = _enforce_linear_iteration;
  return *this;
}

// 设置 promote_inputs_to_common_dtype_ 标志，默认为 false
// 如果为 true，则总是计算迭代器的“公共数据类型”（参见[Common Dtype Computation]注释）
// 并且在 CPU 上，将输入的临时副本以公共数据类型作为实际输入传递给操作
// 将此标志设置为 true 会将 check_all_same_dtype_ 设置为 false
TensorIteratorConfig& promote_inputs_to_common_dtype(
    const bool _promote_inputs_to_common_dtype) {
  promote_inputs_to_common_dtype_ = _promote_inputs_to_common_dtype;
  if (_promote_inputs_to_common_dtype) {
    check_all_same_dtype_ = false;
  }
  return *this;
}

// 设置 promote_integer_inputs_to_float_ 标志，默认为 false
// 注意：如果设置为 true，则 promote_inputs_to_common_dtype_ 也必须为 true
// 如果为 true，则如果迭代器的“公共数据类型”是整数类型（包括 bool），则将其更改为默认的浮点标量类型
TensorIteratorConfig& promote_integer_inputs_to_float(
    const bool _promote_integer_inputs_to_float) {
  promote_integer_inputs_to_float_ = _promote_integer_inputs_to_float;
  TORCH_INTERNAL_ASSERT(
      !promote_integer_inputs_to_float_ || promote_inputs_to_common_dtype_);
  return *this;
}

// 设置 is_reduction_ 标志
TensorIteratorConfig& is_reduction(const bool _is_reduction) {
  is_reduction_ = _is_reduction;
  return *this;
}

// 设置 allow_cpu_scalars_ 标志
TensorIteratorConfig& allow_cpu_scalars(const bool _allow_cpu_scalars) {
  allow_cpu_scalars_ = _allow_cpu_scalars;


这些注释为给定的每个函数定义了功能、默认行为以及参数的影响，使得每个函数的作用和预期行为变得清晰易懂。
    // 返回当前对象的引用，用于支持方法链调用
    return *this;
  }

  // 设置 cast_common_dtype_to_outputs_ 标志，默认为 false
  // 如果设置为 true，则迭代器的“公共数据类型”必须是可计算的
  // （见 [Common Dtype Computation] 注释），在 CPU 上，将临时复制的输出作为操作的实际输出传递。
  // 然后在操作执行后将这些临时输出复制到原始输出中（参见 cast_outputs() 方法）。
  // 将此标志设置为 true 会将 check_all_same_dtype_ 设置为 false。
  TensorIteratorConfig& cast_common_dtype_to_outputs(
      const bool _cast_common_dtype_to_outputs) {
    cast_common_dtype_to_outputs_ = _cast_common_dtype_to_outputs;
    if (_cast_common_dtype_to_outputs) {
      check_all_same_dtype_ = false;
    }
    return *this;
  }

  // 设置 resize_outputs_ 标志，用于控制是否调整输出大小，默认为 true
  TensorIteratorConfig& resize_outputs(bool resize_outputs) {
    resize_outputs_ = resize_outputs;
    return *this;
  }

  // 绕过输出的数据类型和设备的计算，并将其固定为此处指定的值
  TensorIteratorConfig& declare_static_dtype_and_device(
      ScalarType dtype,
      Device device);
  
  // 声明静态数据类型，绕过数据类型的计算
  TensorIteratorConfig& declare_static_dtype(ScalarType dtype);

  // 声明静态设备，绕过设备的计算
  TensorIteratorConfig& declare_static_device(Device device);

  // 声明静态形状，指定张量的静态形状
  TensorIteratorConfig& declare_static_shape(IntArrayRef shape);

  // 声明静态形状，并指定要压缩的维度
  TensorIteratorConfig& declare_static_shape(
      IntArrayRef shape,
      IntArrayRef squash_dims);

  // 构建并返回一个 TensorIterator 对象
  // 注意：如果此方法使用了 && 限定符（右值引用），会带来更好的性能，但需要更多的样板代码支持
  TensorIterator build() {
    TensorIterator iter;
    iter.build(*this);
    return iter;
  }

 private:
  // 检查张量在指定索引处是否为常量
  bool is_tensor_const(size_t idx);

  // 包含张量的列表
  SmallVector<c10::MaybeOwned<TensorBase>, 4> tensors_;
  int num_outputs_ = 0;  // 输出张量的数量
  int num_inputs_ = 0;   // 输入张量的数量

  // 静态形状、数据类型和设备的可选值
  std::optional<DimVector> static_shape_ = c10::nullopt;
  std::optional<ScalarType> static_dtype_ = c10::nullopt;
  std::optional<Device> static_device_ = c10::nullopt;

  // 控制内存重叠检查的标志，默认为 true
  bool check_mem_overlap_ = true;

  // 控制是否允许 CPU 标量，默认为 false
  bool allow_cpu_scalars_ = false;

  // 标志指示是否为归约操作，默认为 false
  bool is_reduction_ = false;

  // 控制是否调整输出大小的标志，默认为 true
  bool resize_outputs_ = true;

  // 控制是否检查所有张量的相同数据类型的标志，默认为 true
  bool check_all_same_dtype_ = true;

  // 控制是否检查所有张量的相同设备的标志，默认为 true
  bool check_all_same_device_ = true;

  // 控制是否强制安全类型转换到输出的标志，默认为 false
  bool enforce_safe_casting_to_output_ = false;

  // 控制是否执行线性迭代的标志，默认为 false
  bool enforce_linear_iteration_ = false;

  // 控制是否将输入张量提升到公共数据类型的标志，默认为 false
  bool promote_inputs_to_common_dtype_ = false;

  // 控制是否将整数输入提升为浮点数的标志，默认为 false
  bool promote_integer_inputs_to_float_ = false;

  // 控制是否将公共数据类型转换到输出的标志，默认为 false
  bool cast_common_dtype_to_outputs_ = false;

  // 包含常量张量索引的列表
  SmallVector<size_t, 4> const_tensor_indices_;
};

/// A container-like struct that acts as if it contains splits of a
/// TensorIterator that can use 32-bit indexing. Taken together the splits cover
/// the original TensorIterator.
struct TORCH_API SplitUntil32Bit {
  struct TORCH_API iterator {
    /// Default constructor for iterator.
    iterator() = default;
    
    /// Constructs an iterator from a TensorIteratorBase object.
    iterator(const TensorIteratorBase& iter);
    
    /// Move constructor for iterator.
    iterator(iterator&&) = default;

    // Guaranteed to be a TensorIterator proper!
    /// Dereference operator to access the underlying TensorIterator.
    TensorIterator& operator*() const;
    
    /// Prefix increment operator to move to the next split.
    iterator& operator++();
    
    /// Equality operator for iterators.
    bool operator==(const iterator& other) const {
      // two iterators are equal if they are the same object or they're both
      // empty
      return this == &other || (vec.empty() && other.vec.empty());
    }
    
    /// Inequality operator for iterators.
    // needed for C++11 range-based for loop
    bool operator!=(const iterator& other) const {
      return !(*this == other);
    }

    /// stack of TensorIterators to be split
    std::vector<std::unique_ptr<TensorIterator>> vec;
  };

  /// Constructor that initializes SplitUntil32Bit with a TensorIteratorBase object.
  SplitUntil32Bit(const TensorIteratorBase& iter) : iter(iter) {}

  /// Returns an iterator pointing to the beginning of the splits.
  iterator begin() const;
  
  /// Returns an iterator pointing to the end of the splits.
  iterator end() const;

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  /// Reference to the original TensorIteratorBase object.
  const TensorIteratorBase& iter;
};

} // namespace at
```