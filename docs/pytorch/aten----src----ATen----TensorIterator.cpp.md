# `.\pytorch\aten\src\ATen\TensorIterator.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 定义宏，仅包含方法操作符

#define TORCH_ASSERT_NO_OPERATORS
// 定义宏，不包含任何操作符

#include <ATen/TensorIterator.h>
// 包含 ATen 库中的 TensorIterator 头文件，用于处理张量迭代器的功能

#undef TORCH_ASSERT_NO_OPERATORS
// 取消先前定义的 TORCH_ASSERT_NO_OPERATORS 宏

#include <ATen/core/Tensor.h>
// 包含 ATen 库中的 Tensor 核心功能头文件

#include <ATen/ExpandUtils.h>
// 包含 ATen 库中的 ExpandUtils 头文件，用于张量扩展的工具函数

#include <ATen/Parallel.h>
// 包含 ATen 库中的 Parallel 头文件，用于并行计算支持

#include <ATen/native/TypeProperties.h>
// 包含 ATen 库中的 TypeProperties 头文件，定义了张量类型的属性和特性

#include <ATen/MemoryOverlap.h>
// 包含 ATen 库中的 MemoryOverlap 头文件，用于处理张量内存重叠检测

#include <ATen/native/Resize.h>
// 包含 ATen 库中的 Resize 头文件，提供了张量调整大小的功能

#include <ATen/NamedTensorUtils.h>
// 包含 ATen 库中的 NamedTensorUtils 头文件，提供了操作命名张量的工具函数

#include <ATen/TensorOperators.h>
// 包含 ATen 库中的 TensorOperators 头文件，定义了张量操作符

#include <ATen/TensorIteratorInternal.h>
// 包含 ATen 库中的 TensorIteratorInternal 头文件，定义了张量迭代器的内部实现细节

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#endif
// 根据是否定义了 AT_PER_OPERATOR_HEADERS 宏，选择包含不同的 ATen 操作函数头文件

#include <c10/util/irange.h>
// 包含 c10 库中的 irange 头文件，提供了对整数范围进行迭代的功能

#include <c10/util/SmallBuffer.h>
// 包含 c10 库中的 SmallBuffer 头文件，实现了小缓冲区的功能

#include <array>
// 包含标准库中的 array 头文件，提供了固定大小数组的支持

#include <algorithm>
// 包含标准库中的 algorithm 头文件，提供了各种算法的实现

#include <cmath>
// 包含标准库中的 cmath 头文件，提供了数学函数的定义

namespace at {
// 进入 ATen 命名空间

using DimMask = TensorIteratorBase::DimMask;
using PtrVector = TensorIteratorBase::PtrVector;
using loop2d_t = TensorIteratorBase::loop2d_t;
using StrideVector = TensorIteratorBase::StrideVector;
// 定义类型别名，引用 TensorIteratorBase 中的类型

namespace {

inline void get_base_ptrs(char** ptrs, ArrayRef<OperandInfo> operands) {
  // 定义内联函数，获取操作数的基础指针数组
  std::transform(operands.begin(), operands.end(), ptrs, [](const OperandInfo& op) {
    return static_cast<char*>(op.data);
  });
}

inline void get_strides(int64_t* strides, ArrayRef<OperandInfo> operands, int64_t ndim) {
  // 定义内联函数，获取操作数的步长数组
  for (const auto dim : c10::irange(ndim)) {
    for (const auto arg : c10::irange(operands.size())) {
      *strides++ = operands[arg].stride_bytes[dim];
    }
  }
  // 始终保证至少有 2 维步长以支持二维 for_each 循环
  if (ndim < 2) {
    auto ntensors = operands.size();
    std::fill_n(strides, (2 - ndim) * ntensors, 0);
  }
}

static OptionalTensorRef make_otr(const TensorBase &tensor) {
  // 定义静态函数，创建 OptionalTensorRef 对象
  if (tensor.defined()) {
    return OptionalTensorRef(tensor);
  } else {
    return OptionalTensorRef();
  }
}

}

namespace internal {

OpaqueOptionalTensorRef::OpaqueOptionalTensorRef() {
  // OpaqueOptionalTensorRef 类的默认构造函数实现
  static_assert(alignof(OptionalTensorRef) == alignof(TensorBase));
  static_assert(sizeof(OptionalTensorRef) == sizeof(TensorBase));
  new (data_.data()) OptionalTensorRef();
}

OpaqueOptionalTensorRef::~OpaqueOptionalTensorRef() {
  // OpaqueOptionalTensorRef 类的析构函数实现
  get()->~OptionalTensorRef();
}

const Tensor& OpaqueOptionalTensorRef::getTensor() const {
  // 返回 OpaqueOptionalTensorRef 对象内部的张量引用
  return get()->getTensorRef();
}

}

void OperandInfo::tensor(c10::MaybeOwned<TensorBase> &&tensor) {
  // OperandInfo 类的 tensor 方法实现，用于设置张量成员
  tensor_base_ = std::move(tensor);
  *tensor_storage_ = make_otr(*tensor_base_);
}

void OperandInfo::exchange_tensor(c10::MaybeOwned<TensorBase> &&new_tensor) {
  // OperandInfo 类的 exchange_tensor 方法实现，用于交换张量成员
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!original_tensor_base_->defined());
  original_tensor_base_ = std::exchange(tensor_base_, std::move(new_tensor));
  *original_tensor_storage_ = std::exchange(*tensor_storage_, make_otr(*tensor_base_));
}

void OperandInfo::restore_original_tensor() {
  // OperandInfo 类的 restore_original_tensor 方法实现，用于恢复原始张量成员
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(original_tensor_base_->defined());
  tensor_base_ = std::move(original_tensor_base_);
  *tensor_storage_ = std::exchange(*original_tensor_storage_, OptionalTensorRef{});
}

/// Construction
// 构造函数部分
// 向迭代器配置中添加一个拥有的输出张量
TensorIteratorConfig& TensorIteratorConfig::add_owned_output(const TensorBase& output) {
  // 断言确保没有添加输入之前先添加所有的输出
  TORCH_INTERNAL_ASSERT(
      num_inputs_ == 0,
      "Keep in mind that you have to add all outputs first before adding any input. "
      "For more details, see https://github.com/pytorch/pytorch/wiki/How-to-use-TensorIterator.");
  // 使用拥有的方式向张量列表中添加输出张量
  tensors_.push_back(c10::MaybeOwned<TensorBase>::owned(std::in_place, output));
  // 增加输出张量计数
  num_outputs_++;
  // 返回当前配置对象的引用
  return *this;
}

// 向迭代器配置中添加一个拥有的输入张量
TensorIteratorConfig& TensorIteratorConfig::add_owned_input(const TensorBase& input) {
  // 使用拥有的方式向张量列表中添加输入张量
  tensors_.push_back(c10::MaybeOwned<TensorBase>::owned(std::in_place, input));
  // 增加输入张量计数
  num_inputs_++;
  // 返回当前配置对象的引用
  return *this;
}

// 向迭代器配置中添加一个拥有的常量输入张量
TensorIteratorConfig& TensorIteratorConfig::add_owned_const_input(const TensorBase& input) {
  // 记录常量张量的索引位置
  const_tensor_indices_.push_back(tensors_.size());
  // 使用拥有的方式向张量列表中添加常量输入张量
  tensors_.push_back(c10::MaybeOwned<TensorBase>::owned(std::in_place, input));
  // 增加输入张量计数
  num_inputs_++;
  // 返回当前配置对象的引用
  return *this;
}

// 向迭代器配置中添加一个借用的输出张量
TensorIteratorConfig& TensorIteratorConfig::add_borrowed_output(const TensorBase& output) {
  // 断言确保没有添加输入之前先添加所有的输出
  TORCH_INTERNAL_ASSERT(
      num_inputs_ == 0,
      "Keep in mind that you have to add all outputs first before adding any input. "
      "For more details, see https://github.com/pytorch/pytorch/wiki/How-to-use-TensorIterator.");
  // 使用借用的方式向张量列表中添加输出张量
  tensors_.push_back(c10::MaybeOwned<TensorBase>::borrowed(output));
  // 增加输出张量计数
  num_outputs_++;
  // 返回当前配置对象的引用
  return *this;
}

// 向迭代器配置中添加一个借用的输入张量
TensorIteratorConfig& TensorIteratorConfig::add_borrowed_input(const TensorBase& input) {
  // 使用借用的方式向张量列表中添加输入张量
  tensors_.push_back(c10::MaybeOwned<TensorBase>::borrowed(input));
  // 增加输入张量计数
  num_inputs_++;
  // 返回当前配置对象的引用
  return *this;
}

// 向迭代器配置中添加一个借用的常量输入张量
TensorIteratorConfig& TensorIteratorConfig::add_borrowed_const_input(const TensorBase& input) {
  // 记录常量张量的索引位置
  const_tensor_indices_.push_back(tensors_.size());
  // 使用借用的方式向张量列表中添加常量输入张量
  tensors_.push_back(c10::MaybeOwned<TensorBase>::borrowed(input));
  // 增加输入张量计数
  num_inputs_++;
  // 返回当前配置对象的引用
  return *this;
}

// 声明静态的数据类型和设备类型
TensorIteratorConfig& TensorIteratorConfig::declare_static_dtype_and_device(ScalarType dtype, Device device) {
  // 检查是否已经设置了检查所有相同数据类型为假
  TORCH_CHECK(!check_all_same_dtype_, "check_all_same_dtype(false) must be called before declare_static_dtype(...)");
  // 设置静态的数据类型和设备类型
  static_dtype_ = dtype;
  static_device_ = device;
  // 返回当前配置对象的引用
  return *this;
}

// 声明静态的数据类型
TensorIteratorConfig& TensorIteratorConfig::declare_static_dtype(ScalarType dtype) {
  // 检查是否已经设置了检查所有相同数据类型为假
  TORCH_CHECK(!check_all_same_dtype_, "check_all_same_dtype(false) must be called before declare_static_dtype(...)");
  // 设置静态的数据类型
  static_dtype_ = dtype;
  // 返回当前配置对象的引用
  return *this;
}

// 声明静态的设备类型
TensorIteratorConfig& TensorIteratorConfig::declare_static_device(Device device) {
  // 设置静态的设备类型
  static_device_ = device;
  // 返回当前配置对象的引用
  return *this;
}

// 声明静态的张量形状
TensorIteratorConfig& TensorIteratorConfig::declare_static_shape(IntArrayRef shape) {
  // 警告：这将绕过TensorIterator中的所有形状检查
  // 调用此方法的核心应在调用add_owned_input或add_owned_output之前检查形状
  TORCH_CHECK(!resize_outputs_, "resize_outputs() must be called before declare_static_shape(...)")
  // 设置静态的张量形状
  static_shape_ = c10::make_optional(DimVector(shape));
  // 返回当前配置对象的引用
  return *this;
}
// 在声明静态形状时，根据提供的形状和压缩维度信息更新配置
TensorIteratorConfig& TensorIteratorConfig::declare_static_shape(IntArrayRef shape, IntArrayRef squash_dims) {
  // 调用重载方法，声明静态形状
  declare_static_shape(shape);
  // 如果静态形状为空，则直接返回当前配置对象
  if (static_shape_->empty()) return *this;
  // 遍历压缩维度列表，更新静态形状中相应维度的大小为1
  for (const auto& squash_dim : squash_dims) {
    // 检查压缩维度是否合法，即在静态形状的范围内
    TORCH_CHECK(squash_dim >= 0 && squash_dim < static_cast<int64_t>(static_shape_->size()),
                "squash_dim ", squash_dim, " must be in [0, ", static_shape_->size(), ").");
    // 更新静态形状中指定维度的大小为1
    (*static_shape_)[squash_dim] = 1;
  }
  // 返回更新后的配置对象
  return *this;
}

// 判断指定索引处的张量是否为常量
bool TensorIteratorConfig::is_tensor_const(size_t idx) {
  // 使用 STL 的 find 函数在常量张量索引列表中查找指定索引
  return std::find(const_tensor_indices_.begin(), const_tensor_indices_.end(), idx) != const_tensor_indices_.end();
}

// NOTE: [Computing output strides]
// 计算输出步长的算法说明
// 如果提供了正确大小的输出，尊重其步长并不做修改
// 否则，如果输出大小不正确或没有提供输出，
// 尝试通过对输入的步长进行排序来恢复应用的排列
// 优先考虑按添加顺序排列的输入和涉及非广播维度的排列
// 1. 我们从第一个输入开始循环
// 2. 对于所有输入，广播维度的步长被设置为0，并且0与任何值相等。如果正在比较的维度之一的步长为0，则移至下一个张量以确定是否需要交换这些维度。
// 3. 步长等于1的维度参与排序
// 4. 如果两个步长相等且都不为0，则通过查看相应的维度来尝试打破平局。如果在从末尾迭代时，对应于相同步长的维度是增加的，则维度被置换。
// 如果维度是非递增的，则移至下一个输入以打破平局。

// 对于平局打破规则4，我们可以直接进入下一个张量。 这将可能丢失第一个张量的正确排列，如果存在排列的平凡维度，但可能会改善第二个张量的遍历顺序。 我们选择前一个选项以更好地传播通道后布局。 例如，对于具有大小N1H1的张量。

// 这些规则导致直观行为，在大多数情况下恢复第一个参数（如果所有参数大小相同）或未广播的参数的排列，无论其位置如何。

// 作为一个额外的好处，它还导致输入和输出的遍历顺序合理行为良好 - 在内核中，输出线性遍历，并且由于它紧随输入布局，输入也线性遍历。

// 示例：
// 完整大小的张量 + 广播张量，其中0或1个非平凡维度=> 输出的步长与完整大小的输入的步长相同，无论顺序如何。
// 大小相同但步长不同的两个张量=> 输出步长与第一个参数相同。
//
// 对于具有相同步幅的内存密集型输入，我们还有快速路径（或者，单个内存密集型输入的情况），输出与输入具有相同的步幅。
// 与上述算法的唯一区别在于对于平凡（步幅为1）维度的步幅，在性能考虑下的模糊情况下，默认为连续的步幅。
// 例如：具有大小 NC11 和步幅 C1CC 的张量将生成步幅为 C111 的输出（注意只有在平凡维度的步幅中有差异，因此物理布局不受影响，但排列信息丢失）
// 一旦解决了性能问题，我们可能会在未来更改此行为

void TensorIteratorBase::reorder_dimensions() {
  // 根据步幅升序排序维度，并将减少的维度放在前面。注意这会颠倒 C 连续张量的顺序。
  // strides[0] 是最快移动的维度，而不是 strides[ndim - 1]。
  // 更详细的描述，请参见 NOTE: [Computing output strides] 和内联注释

  // 调整 perm_ 的大小以适应当前张量的维度
  perm_.resize(ndim());

  // 如果张量只有一个维度，直接设定 perm_ 为 [0]，然后返回
  if (ndim() == 1) {
    perm_[0] = 0;
    return;
  }

  // 初始化 perm_ 为 n-1, n-2, ..., 1, 0，即逆序排列维度
  std::iota(perm_.rbegin(), perm_.rend(), 0);

  // 如果需要强制线性迭代，则对维度进行排列
  if (enforce_linear_iteration_) {
    permute_dimensions(perm_);
    return;
  }

  // 定义一个 lambda 函数 should_swap，用于比较两个维度的顺序
  // 如果 dim0 应该在 dim1 之后返回 1，如果 dim0 应该在 dim1 之前返回 -1，如果比较模糊返回 0
  auto should_swap = [&](size_t dim0, size_t dim1) {
    // 对于每一个张量的索引，进行迭代
    for (const auto arg : c10::irange(ntensors())) {
      // 忽略未定义或尺寸不正确的张量
      if (operands_[arg].stride_bytes.empty() || operands_[arg].will_resize) {
        continue;
      }
      // 获取当前维度下的步长值
      int64_t stride0 = operands_[arg].stride_bytes[dim0];
      int64_t stride1 = operands_[arg].stride_bytes[dim1];
      // 如果是归约操作且当前张量是输出
      if (is_reduction_ && operands_[arg].is_output) {
        // 将归约的维度移至最前
        // 归约维度的步长由 review_reduce_result 设置为 0
        if ((stride0 == 0) != (stride1 == 0)) {
          // 如果 stride1 为 0，则返回 1；否则返回 -1
          return stride1 == 0 ? 1 : -1;
        }
      }
      // 如果其中一个维度的步长为 0，则跳过当前张量
      if (stride0 == 0 || stride1 == 0) {
        continue;
      // 如果步长相等，则使用维度本身来打破平局
      } else if (stride0 < stride1) {
        return -1;
      } else  if (stride0 > stride1) {
        return 1;
      } else { // 步长相等，使用维度本身来打破平局
        // 此时排除了步长为 0 的情况，保证 operand 的维度等于 shape_ 的维度
         auto t_dim0 = shape_[dim0];
         auto t_dim1 = shape_[dim1];
         // 如果 t_dim0 大于 t_dim1，则返回 1，否则继续下一个张量
         if (t_dim0 > t_dim1) {
             return 1;
         }
      }
    }
    // 若所有比较都未返回，则默认返回 0
    return 0;
  };

  // 使用插入排序，并支持不明确的比较结果
  for (const auto i : c10::irange(1, ndim())) {
    int dim1 = i;
    for (int dim0 = i - 1; dim0 >= 0; dim0--) {
      // 获取应当交换的维度索引
      int comparison = should_swap(perm_[dim0], perm_[dim1]);
      // 如果 comparison > 0，交换维度索引
      if (comparison > 0) {
        std::swap(perm_[dim0], perm_[dim1]);
        dim1 = dim0; // 更新 dim1 的值
      } else if (comparison < 0) {
        break; // 若 comparison < 0，跳出内层循环
      }
    }
  }

  // 执行维度和步长的重新排序
  permute_dimensions(perm_);
}

// 计算一个通用的数据类型，使用类型提升来决定
// 参见 [Common Dtype Computation] 注释
ScalarType TensorIteratorBase::compute_common_dtype() {
  // 初始化状态变量
  at::native::ResultTypeState state = {};
  
  // 遍历操作数列表
  for (const auto& op : operands_) {
    // 如果当前操作数是输出，则跳过
    if (op.is_output) {
      continue;
    }
    
    // 更新状态，根据操作数的张量类型
    state = at::native::update_result_type_state(op.tensor(), state);
  }

  // 计算通用数据类型
  common_dtype_ = at::native::result_type(state);
  // 断言通用数据类型不能是未定义的
  TORCH_INTERNAL_ASSERT(common_dtype_ != ScalarType::Undefined);

  // 返回计算得到的通用数据类型
  return common_dtype_;
}

// 返回操作信息对应的初始选项
static TensorOptions original_options(const OperandInfo& op) {
  if (op.original_tensor_base().defined()) {
    return op.original_tensor_base().options();
  } else {
    return op.options();
  }
}

// 实现以下标志的行为：
//   - check_all_same_dtype_
//   - check_all_same_device_
//   - enforce_safe_casting_to_output_
//   - promote_inputs_to_common_dtype_
//   - cast_common_dtype_to_outputs_
//
// 详细描述请参见 TensorIterator.h 文件
// 注意：更具体的行为检查（例如，第一个和第二个输入必须共享一个数据类型，
//   但第三个必须具有长整型数据类型）应直接在 TensorIterator 外部实现。
void TensorIteratorBase::compute_types(const TensorIteratorConfig& config) {
  // 回顾操作数（1/2）
  //   - 验证所有输入张量都已定义
  //   - 计算通用设备
  //   - 确定是否存在未定义的输出
  //   - 确定是否存在不同的数据类型，并尝试快速获取通用数据类型
  Device common_device = kCPU;
  common_dtype_ = ScalarType::Undefined;
  // 注意：尽管 output_dtype 看起来很通用，但只有在 check_all_same_dtype 为 true 时才会以非平凡的方式使用
  ScalarType output_dtype = ScalarType::Undefined;
  bool has_different_input_dtypes = false;
  bool has_different_output_dtypes = false;
  bool has_undefined_outputs = false;

  // 遍历操作数列表
  for (auto& op : operands_) {
    // 验证所有输入是否具有类型信息，并且如果输出缺少类型信息，则可以推断应分配其内存的设备。
    if (!op.is_type_defined()) {
      TORCH_INTERNAL_ASSERT(op.is_output, "Found type undefined input tensor!");

      // 如果配置中存在静态数据类型，则使用它作为目标数据类型
      if (config.static_dtype_.has_value()) {
        op.target_dtype = config.static_dtype_.value();
      } else {
        has_undefined_outputs = true;
      }

      // 如果配置中存在静态设备，则使用它作为目标设备
      if (config.static_device_.has_value()) {
        op.device = config.static_device_.value();
      } else {
        TORCH_INTERNAL_ASSERT(config.check_all_same_device_);
      }

      // 如果存在未定义的输出或设备没有值，则继续下一个操作数
      if (has_undefined_outputs || !op.device.has_value()) {
        continue;
      }
    }

    // 验证输入张量是否已定义
    if (!op.tensor_base().defined()) {
      TORCH_INTERNAL_ASSERT(op.is_output, "Found undefined input tensor!");
      continue;
    }

    // 断言目标数据类型与当前数据类型相同
    TORCH_INTERNAL_ASSERT(op.target_dtype == op.current_dtype);
    // 如果 common_device 是 kCPU，并且操作的基本张量不是 CPU 设备，则将第一个非 CPU 设备作为 common_device
    if (common_device == kCPU && !op.tensor_base().is_cpu()) {
      common_device = op.tensor_base().device();
    }

    if (!op.is_output) {
      // 判断是否存在不同的输入数据类型
      // 注意：common_dtype_ 被设置为首个定义的输入数据类型
      if (op.target_dtype != common_dtype_) {
        if (common_dtype_ == ScalarType::Undefined) {
          common_dtype_ = op.target_dtype;
        } else {
          has_different_input_dtypes = true;
        }
      }
    } else {  // op.is_output
      // 判断是否存在不同的输出数据类型
      // 注意：输出数据类型被设置为首个定义的输出数据类型
      if (op.target_dtype != output_dtype) {
        if (output_dtype == ScalarType::Undefined) {
          output_dtype = op.target_dtype;
        } else {
          has_different_output_dtypes = true;
        }
      }
    }
  }

  // 检查计算类型是否可计算或不需要
  TORCH_INTERNAL_ASSERT(!(has_different_input_dtypes && !config.promote_inputs_to_common_dtype_ &&
                        (has_undefined_outputs || config.enforce_safe_casting_to_output_ ||
                        config.cast_common_dtype_to_outputs_)));

  // 如果配置要求检查所有输入和定义的输出是否相同的数据类型
  if (config.check_all_same_dtype_ &&
      (has_different_input_dtypes || has_different_output_dtypes ||
      (common_dtype_ != output_dtype && output_dtype != ScalarType::Undefined))) {
    // 抛出信息丰富的错误消息
    for (auto& op : operands_) {
      if (!op.tensor_base().defined()) {
        continue;
      }

      // 检查每个操作的目标数据类型是否与 common_dtype_ 相符
      TORCH_CHECK(op.target_dtype == common_dtype_,
                  "Found dtype ", op.target_dtype, " but expected ", common_dtype_);
    }
  }

  // 如果没有未定义的输出，且不需要进行额外的工作，则提前结束
  if (!has_undefined_outputs && !config.check_all_same_device_ &&
      !config.promote_inputs_to_common_dtype_ && !config.cast_common_dtype_to_outputs_ &&
      !config.enforce_safe_casting_to_output_) {
    // 如果无法推断出 common_dtype_，则将其置为 Undefined
    common_dtype_ = has_different_input_dtypes ? ScalarType::Undefined : common_dtype_;
    return;
  }

  // 如果存在不同的输入数据类型或者所有操作都是标量，并且配置允许将输入提升为共同数据类型
  if ((has_different_input_dtypes || all_ops_are_scalars_) && config.promote_inputs_to_common_dtype_) {
    // 计算共同的数据类型
    common_dtype_ = compute_common_dtype();
  }

  // 如果配置要求将整数输入提升为默认的浮点标量类型
  if (config.promote_integer_inputs_to_float_ &&
      c10::isIntegralType(common_dtype_, /*includeBool=*/true)) {
    // 获取默认数据类型，并转换为对应的标量类型
    common_dtype_ = c10::typeMetaToScalarType(c10::get_default_dtype());
  }

  // 复查操作数 (2/2)
  //   - 为未定义的输出设置元数据
  //   - 如果需要，检查所有张量是否在相同设备上
  //   - 如果需要，检查通用数据类型是否能安全地转换为每个输出类型
  //   - 如果需要，为CPU操作创建临时变量
  common_device_ = common_device;
  int max_cpu_scalars_on_non_cpu = config.allow_cpu_scalars_ ? 1 : 0;
  int current_cpu_scalars_on_non_cpu = 0;
  for (auto& op : operands_) {
    bool is_type_defined = op.is_type_defined();
    bool is_device_defined = op.is_device_defined();

    // 如果类型未定义，设置为通用数据类型
    if (!is_type_defined) {
      op.target_dtype = common_dtype_;
    }
    // 如果设备未定义，设置为通用设备
    if (!is_device_defined) {
      op.device = common_device;
    }

    // 如果既未定义类型也未定义设备，则跳过
    if (!is_type_defined && !is_device_defined) {
      continue;
    }

    // 跳过未定义的张量
    if (!op.tensor_base().defined()) {
      continue;
    }

    // 如果需要检查所有张量是否在同一设备上
    if (config.check_all_same_device_) {
      // 处理在不支持的 CUDA 内核上的 CPU 标量
      if (!common_device.is_cpu() &&
          config.allow_cpu_scalars_ && !op.is_output && op.tensor_base().dim() == 0 &&
          op.tensor_base().is_cpu()) {
        TORCH_CHECK(current_cpu_scalars_on_non_cpu < max_cpu_scalars_on_non_cpu,
                    "Trying to pass too many CPU scalars to non-CPU kernel!");
        ++current_cpu_scalars_on_non_cpu;
      } else if (op.device.value() != common_device) {
        TORCH_CHECK(false,
                    "Expected all tensors to be on the same device, but "
                    "found at least two devices, ", common_device, " and ", op.device.value(), "!");
      }
    }

    // 如果需要安全转换类型检查，并且是输出张量且当前数据类型与通用数据类型不同
    if (config.enforce_safe_casting_to_output_ && op.is_output && op.current_dtype != common_dtype_) {
      TORCH_CHECK(canCast(common_dtype_, op.current_dtype),
                  "result type ", common_dtype_, " can't be cast to the "
                  "desired output type ", op.current_dtype);
    }

    // 如果需要并且请求了CPU操作，创建临时变量
    // TODO: 在可能的情况下重用临时变量（例如对于原位操作）
    // 如果当前设备是 CPU
    if (common_device == kCPU) {
      // 根据需要将输出转换为正确的数据类型，通过创建临时张量（如果需要）
      // 注意：在 is_meta_ 为真时跳过此步骤，因为如果不执行计算，则这里的临时分配是不必要的
      if (config.cast_common_dtype_to_outputs_ && op.is_output && op.current_dtype != common_dtype_ && !is_meta_) {
        // 断言确保 op 的基础张量已定义
        TORCH_INTERNAL_ASSERT(op.tensor_base().defined());
        // 标记 [Output original_tensor is set]
        // 注意：不要在这里使用 set_output，因为临时张量并不是真正的输出；
        // op.tensor 是真正的输出，并且它已经被预先提供给我们了。
        // TODO: cast_outputs 的逻辑需要由结构化内核的实现来处理。可能的处理方式是将推断出的数据类型传递给输出内核，
        // 然后在调用输出内核后执行转换（即在这里执行 cast_outputs），但是与现有的 TensorIterator 整合可能需要一些工作
        op.exchange_tensor(c10::MaybeOwned<TensorBase>::owned(
            at::empty_like(op.tensor(),
                           op.tensor_base().options().dtype(common_dtype_),
                           LEGACY_CONTIGUOUS_MEMORY_FORMAT)));
        // 如果 names_ 不为空，则传播名称信息到 op 的基础张量
        if (!names_.empty()) {
          namedinference::propagate_names(op.tensor_base(), names_);
        }
        // 更新 op 的当前数据类型和目标数据类型为 common_dtype_
        op.current_dtype = common_dtype_;
        op.target_dtype = common_dtype_;
      }

      // 通过创建临时张量将输入提升为正确的数据类型
      if (config.promote_inputs_to_common_dtype_ && !op.is_output && op.current_dtype != common_dtype_) {
        op.exchange_tensor(c10::MaybeOwned<TensorBase>::owned(op.tensor().to(common_dtype_)));
        // 更新 op 的当前数据类型和目标数据类型为 common_dtype_
        op.current_dtype = common_dtype_;
        op.target_dtype = common_dtype_;
      }
    }
  }


这段代码主要是针对特定的条件（common_device == kCPU）进行数据类型转换和处理操作。具体来说：
- 第一个条件块（if）处理输出张量的数据类型转换，如果需要将输出张量转换为与 common_dtype_ 相同的数据类型，会创建临时张量，并进行一些后续处理。
- 第二个条件块（if）处理输入张量的数据类型提升，如果需要将输入张量提升为与 common_dtype_ 相同的数据类型，同样会创建临时张量，并进行更新操作。
}

StrideVector TensorIteratorBase::compatible_stride(int64_t element_size) const {
  auto stride = StrideVector();  // 创建一个空的 StrideVector 对象
  int64_t next_stride = element_size;  // 设置初始步长为元素大小
  for (const auto dim : c10::irange(ndim())) {  // 遍历张量的所有维度
    stride.push_back(next_stride);  // 将当前步长添加到 StrideVector 中
    next_stride *= shape_[dim];  // 更新下一个步长为当前步长乘以当前维度大小
  }
  return stride;  // 返回计算得到的步长数组
}

DimVector TensorIteratorBase::invert_perm(IntArrayRef input) const {
  // 反转由 reorder_dimensions 引起的排列顺序。在调用 coalesce_dimensions 后不再有效。
  TORCH_INTERNAL_ASSERT(!has_coalesced_dimensions_);  // 断言未合并维度
  TORCH_INTERNAL_ASSERT(input.size()==perm_.size());  // 断言输入数组与排列数组大小相同
  auto res = DimVector(input.size());  // 创建与输入数组相同大小的 DimVector 对象
  for (const auto dim : c10::irange(ndim())) {  // 遍历张量的所有维度
    res[perm_[dim]] = input[dim];  // 将输入数组按排列顺序写入结果数组
  }
  return res;  // 返回重新排列后的数组
}

void TensorIteratorBase::allocate_or_resize_outputs() {
  for (const auto i : c10::irange(num_outputs_)) {  // 遍历所有输出张量
    auto& op = operands_[i];  // 获取当前操作数的引用
    if (!op.tensor_base().defined() || op.will_resize) {  // 如果操作数未定义或将调整大小
      TORCH_INTERNAL_ASSERT(op.is_type_defined(), "no type for operand", i);  // 断言操作数已定义类型
      auto element_size = elementSize(op.target_dtype);  // 获取操作数目标数据类型的元素大小
      op.stride_bytes = compatible_stride(static_cast<int64_t>(element_size));  // 计算操作数的字节步长
      // 检查排列是否只是倒序
      bool inverted = true;
      for (const auto j : c10::irange(ndim())) {  // 遍历张量的所有维度
        if (perm_[j] != ndim() - j - 1) {  // 如果排列不是倒序
          inverted = false;
          break;
        }
      }
      auto tensor_shape = invert_perm(shape_);  // 反转张量的形状排列
      if (inverted) {
        // 可以直接返回连续的输出
        // 这样做更快，因为它避免了分配大小为 0 的张量并重新调整大小和重新步进
        set_output_raw_strided(i, tensor_shape, {}, original_options(op), names_);  // 设置输出为原始步进
      } else {
        auto tensor_stride = invert_perm(op.stride_bytes);  // 反转操作数的字节步长排列
        for (const auto dim : c10::irange(ndim())) {  // 遍历张量的所有维度
          tensor_stride[dim] /= static_cast<int64_t>(element_size);  // 调整步长为元素大小的整数倍
        }
        set_output_raw_strided(i, tensor_shape, tensor_stride, original_options(op), names_);  // 设置输出为原始步进和调整后的步长
      }
      op.current_dtype = op.target_dtype;  // 更新操作数的当前数据类型
    } else if (op.tensor_base().defined()) {
      // 即使不调整大小，我们仍然需要告诉 set_output 有关输出，以便正确设置保护和传播名称
      set_output_raw_strided(i, op.tensor_base().sizes(), {}, original_options(op), names_);  // 设置输出为原始步进
    }
  }
}

void TensorIteratorBase::compute_names(const TensorIteratorConfig& config) {
  bool should_infer_names = std::any_of(
      operands_.begin(),
      operands_.end(),
      [](const OperandInfo& op) {
        return op.tensor_base().defined() && op.tensor_base().has_names();  // 判断操作数是否已定义且具有名称
      });
  if (!should_infer_names) {  // 如果不需要推断名称
    return;
  }

  for (auto& op : operands_) {  // 遍历所有操作数
    if (!op.tensor_base().defined()) continue;  // 如果操作数未定义，则继续下一次循环
    // 如果我们正在调整大小，不要包括输出张量，因为无论如何我们都会覆盖它们的名称。
    // （如果输出张量已
    // 如果配置要求调整输出大小并且当前操作是输出，则跳过此次循环
    if (config.resize_outputs_ && op.is_output) continue;
    // 如果 names_ 向量为空，执行名称推断
    if (names_.empty()) {
      // 将 op 的基础张量的名称赋给 names_
      names_ = op.tensor_base().names();
    } else {
      // 否则，使用 unify_from_right 函数将 names_ 向量与 op 的基础张量的名称向量进行右侧统一
      names_ = NameVector(unify_from_right(names_, op.tensor_base().names()));
    }
}

// 对张量迭代器进行维度合并操作
void TensorIteratorBase::coalesce_dimensions() {
  // 如果张量维度小于等于1，则无需进行合并
  if (ndim() <= 1) {
    return;
  }

  // 判断是否可以合并两个相邻的维度，条件为：任一维度大小为1或者 shape[n] * stride[n] == stride[n + 1]
  auto can_coalesce = [&](int dim0, int dim1) {
    auto shape0 = shape_[dim0];
    auto shape1 = shape_[dim1];
    if (shape0 == 1 || shape1 == 1) {
      return true;
    }
    // 遍历操作数，检查条件是否满足
    for (const auto i : c10::irange(ntensors())) {
      auto& stride = operands_[i].stride_bytes;
      if (shape0 * stride[dim0] != stride[dim1]) {
        return false;
      }
    }
    return true;
  };

  // 将 dim0 处的步长替换为 dim1 处的步长
  auto replace_stride = [&](int dim0, int dim1) {
    for (const auto i : c10::irange(ntensors())) {
      auto& stride = operands_[i].stride_bytes;
      stride[dim0] = stride[dim1];
    }
  };

  // 从第一个维度开始向后遍历，合并可合并的维度
  int prev_dim = 0;
  for (const auto dim : c10::irange(1, ndim())) {
    if (can_coalesce(prev_dim, dim)) {
      if (shape_[prev_dim] == 1) {
        replace_stride(prev_dim, dim);
      }
      shape_[prev_dim] *= shape_[dim];
    } else {
      prev_dim++;
      if (prev_dim != dim) {
        replace_stride(prev_dim, dim);
        shape_[prev_dim] = shape_[dim];
      }
    }
  }

  // 调整 shape_ 大小以适应合并后的维度数
  shape_.resize(prev_dim + 1);
  // 调整每个操作数的步长数组大小以匹配新的维度数
  for (const auto i : c10::irange(ntensors())) {
    operands_[i].stride_bytes.resize(ndim());
  }
  // 设置维度已合并的标志为真
  has_coalesced_dimensions_ = true;
}

// 计算张量迭代器涉及的元素总数
int64_t TensorIteratorBase::numel() const {
  int64_t numel = 1;
  for (int64_t size : shape_) {
    numel *= size;
  }
  return numel;
}

// 获取特定维度的步长数组
StrideVector TensorIteratorBase::get_dim_strides(int dim) const {
  auto dims = ndim();
  auto inner_strides = StrideVector();
  for (auto& op : operands_) {
    inner_strides.push_back(dims == 0 ? 0 : op.stride_bytes[dim]);
  }
  return inner_strides;
}

// 获取基础指针数组
SmallVector<char*, 4> TensorIteratorBase::get_base_ptrs() const {
  auto ptrs = SmallVector<char*, 4>(ntensors());
  at::get_base_ptrs(ptrs.data(), operands_);
  return ptrs;
}

// 判断特定维度是否为缩减维度
bool TensorIteratorBase::is_dim_reduced(int dim) const {
  for (auto& op : operands_) {
    if (op.is_output && op.stride_bytes[dim] == 0 && shape_[dim] > 1) {
      return true;
    }
  }
  return false;
}

// 对维度进行重新排列
void TensorIteratorBase::permute_dimensions(IntArrayRef perm) {
  // 断言排列的大小与当前维度数相同
  TORCH_INTERNAL_ASSERT(perm.size() == static_cast<unsigned>(ndim()));

  // 重新排列 shape_ 和 strides
  auto reorder = [perm](IntArrayRef data) {
    auto res = DimVector(data.size(), 0);
    for (const auto i : c10::irange(perm.size())) {
      res[i] = data[perm[i]];
    }
    return res;
  };

  shape_ = reorder(shape_);
  for (auto& op : operands_) {
    if (!op.stride_bytes.empty()) {
      op.stride_bytes = reorder(op.stride_bytes);
    }
  }
}

// 计算输出元素的总数
int64_t TensorIteratorBase::num_output_elements() const {
  int64_t elem = 1;
  for (const auto dim : c10::irange(ndim())) {
    if (operands_[0].stride_bytes[dim] != 0 || shape_[dim] == 0) {
      elem *= shape_[dim];
    }
  }
  return elem;
}
// 返回在迭代器操作数中维度为0的数量
int TensorIteratorBase::num_reduce_dims() const {
  int count = 0;  // 初始化维度为0的计数器
  // 遍历迭代器的维度范围
  for (const auto dim : c10::irange(ndim())) {
    // 如果操作数的第一个张量在当前维度上的步长为0，增加计数器
    if (operands_[0].stride_bytes[dim] == 0) {
      count++;
    }
  }
  return count;  // 返回维度为0的数量
}

// 针对每个元素执行循环操作，支持并行处理
void TensorIteratorBase::for_each(loop2d_t loop, int64_t grain_size) {
  int64_t numel = this->numel();  // 获取迭代器的元素总数
  if (numel == 0) {  // 如果元素总数为0，直接返回
    return;
  } else if (numel < grain_size || at::get_num_threads() == 1) {
    // 如果元素总数小于颗粒大小或者只有一个线程，串行执行循环操作
    return serial_for_each(loop, {0, numel});
  } else {
    // 否则，使用并行方式执行循环操作
    at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
      serial_for_each(loop, {begin, end});  // 并行循环中串行执行每个子范围
    });
  }
}

// 获取迭代器的步长向量
StrideVector TensorIteratorBase::get_strides() const {
  const auto dim = ndim();  // 获取迭代器的维度数
  StrideVector strides(static_cast<size_t>(std::max(dim, 2)) * ntensors());  // 创建步长向量
  at::get_strides(strides.data(), operands_, dim);  // 获取每个操作数的步长
  return strides;  // 返回步长向量
}

// 在给定范围内串行执行循环操作
void TensorIteratorBase::serial_for_each(loop2d_t loop, Range range) const {
  if (range.size() == 0) {  // 如果范围大小为0，直接返回
    return;
  }

  const auto ntensors = this->ntensors();  // 获取操作数的数量
  const auto ndim = this->ndim();  // 获取迭代器的维度数

  c10::SmallBuffer<char*, 4> ptrs(ntensors);  // 创建指针缓冲区
  c10::SmallBuffer<int64_t, 8> strides(ntensors * static_cast<size_t>(std::max(ndim, 2)));  // 创建步长缓冲区

  at::get_base_ptrs(ptrs.data(), operands_);  // 获取操作数的基础指针
  at::get_strides(strides.data(), operands_, ndim);  // 获取操作数的步长
  at::internal::serial_for_each(
      shape_, strides, ptrs.data(), ptrs.size(), loop, range);  // 调用内部串行循环操作
}

// 检查迭代器是否是1维的
bool TensorIteratorBase::is_trivial_1d() const {
  // TODO: 一旦支持类型转换，检查是否支持
  return ndim() == 1;  // 返回迭代器的维度是否为1
}

// 检查迭代器是否是连续的
bool TensorIteratorBase::is_contiguous() const {
  if (numel() == 1) {  // 如果元素总数为1，直接返回是连续的
    return true;
  }
  if (ndim() != 1) {  // 如果维度不为1，直接返回不是连续的
    return false;
  }
  return has_contiguous_first_dim();  // 否则检查第一个维度是否连续
}

// 检查操作数是否是标量
bool TensorIteratorBase::is_scalar(int64_t arg) const {
  const auto& stride = operands_[arg].stride_bytes;  // 获取指定操作数的步长
  for (const auto i : c10::irange(ndim())) {  // 遍历迭代器的维度范围
    // 如果步长不为0且对应维度的形状不为1，返回false
    if (stride[i] != 0 && shape_[i] != 1) {
      return false;
    }
  }
  return true;  // 否则返回true，表示是标量
}

// 检查操作数是否是CPU标量
bool TensorIteratorBase::is_cpu_scalar(int64_t arg) const {
  return is_scalar(arg) && device(arg).is_cpu();  // 返回操作数是否是标量并且设备类型是CPU
}

// 对输出进行类型转换和大小调整
void TensorIteratorBase::cast_outputs() {
  for (auto& op : operands_) {  // 遍历所有操作数
    // 如果是输出并且原始张量基础存在且当前数据类型与目标数据类型不同
    if (op.is_output && op.original_tensor_base().defined() &&
        op.original_tensor_base().scalar_type() != op.current_dtype) {
      const auto &original_tensor = op.original_tensor();  // 获取原始张量
      const auto &tensor = op.tensor();  // 获取当前张量
      // 如果原始张量大小与当前张量大小不同，调整原始张量的大小和步长
      if (original_tensor.sizes() != tensor.sizes()) {
        original_tensor.resize_as_(tensor).as_strided_(tensor.sizes(), tensor.strides());
      }
      original_tensor.copy_(tensor);  // 复制当前张量到原始张量
      op.restore_original_tensor();  // 恢复原始张量状态
    }
  }
}

// 获取指定操作数的数据指针
void* TensorIteratorBase::data_ptr(int64_t arg) const {
  return operands_[arg].data;  // 返回指定操作数的数据指针
}

// 移除指定位置的操作数
void TensorIteratorBase::remove_operand(int64_t arg) {
  operands_.erase(operands_.begin() + arg);  // 移除指定位置的操作数
}

// 替换指定位置的操作数的数据指针（不安全版本）
void TensorIteratorBase::unsafe_replace_operand(int64_t arg, void* data) {
  operands_[arg].data = data;  // 替换指定位置的操作数的数据指针
}
// 缩窄张量迭代器的指定维度，修改形状和视图偏移量，并根据新的起始位置重新计算操作数的数据指针
void TensorIteratorBase::narrow(int dim, int64_t start, int64_t size) {
  // 断言维度小于当前张量迭代器的维度数，并且尺寸大于等于1
  TORCH_INTERNAL_ASSERT(dim < ndim() && size >= 1);
  // 修改指定维度的形状为给定大小
  shape_[dim] = size;
  // 根据起始位置更新视图偏移量
  view_offsets_[dim] += start;
  // 更新每个操作数的数据指针，根据指定维度的步长和起始位置计算偏移量
  for (auto& op : operands_) {
    op.data = ((char*)op.data) + op.stride_bytes[dim] * start;
  }
  // 如果新的尺寸为1且不是归约操作，则进行维度合并优化
  if (size == 1 && !is_reduction_) {
    coalesce_dimensions();
  }
}

// 在保持指定起始维度的前提下，选择所有元素，并根据给定的索引重新计算操作数的数据指针
void TensorIteratorBase::select_all_keeping_dim(int start_dim, IntArrayRef indices) {
  // 断言起始维度小于等于当前张量迭代器的维度数
  TORCH_INTERNAL_ASSERT(start_dim <= ndim());
  // 遍历从起始维度到最后一个维度
  for (const auto i : c10::irange(start_dim, ndim())) {
    // 更新每个操作数的数据指针，根据当前维度的步长和对应索引计算偏移量
    for (auto& op : operands_) {
      op.data = ((char*)op.data) + op.stride_bytes[i] * indices[i - start_dim];
    }
    // 将当前维度的形状设置为1
    shape_[i] = 1;
  }
}

// 宏定义，用于配置二元浮点操作的张量迭代器
#define BINARY_FLOAT_OP_CONFIG()                \
  TensorIteratorConfig()                        \
    .set_check_mem_overlap(true)                \
    .allow_cpu_scalars(true)                    \
    .promote_inputs_to_common_dtype(true)       \
    .cast_common_dtype_to_outputs(true)         \
    .enforce_safe_casting_to_output(true)       \
    .promote_integer_inputs_to_float(true)

// 构建一个二元浮点操作的张量迭代器，将整数输入提升为浮点数
void TensorIteratorBase::build_binary_float_op(
    const TensorBase& out, const TensorBase& a, const TensorBase& b) {
  // 使用宏配置来构建张量迭代器，添加输出和输入张量
  build(BINARY_FLOAT_OP_CONFIG()
        .add_owned_output(out)
        .add_owned_const_input(a)
        .add_owned_const_input(b));
}

// 构建一个借用方式的二元浮点操作的张量迭代器
void TensorIteratorBase::build_borrowing_binary_float_op(
    const TensorBase& out, const TensorBase& a, const TensorBase& b) {
  // 使用宏配置来构建张量迭代器，添加输出和常量输入张量
  build(BINARY_FLOAT_OP_CONFIG()
        .add_output(out)
        .add_const_input(a)
        .add_const_input(b));
}

// 设置比较操作的配置，根据输出张量的定义情况特殊处理
static void set_up_comparison_op_config(TensorIteratorConfig& config, const TensorBase& out) {
  // 设置配置选项，检查内存重叠，允许CPU标量，将输入提升为公共数据类型
  config.set_check_mem_overlap(true);
  config.allow_cpu_scalars(true);
  config.promote_inputs_to_common_dtype(true);

  // 如果输出张量未定义（例如函数操作符'a == b'），将输出数据类型声明为布尔型
  if (!out.defined()) {
    config.declare_static_dtype(kBool);
  }

  // 特殊处理布尔型输出数据类型的情况
  // 当输出张量具有布尔类型时，不调用'cast_common_dtype_to_outputs'，以提升性能
  // 所有使用该TensorIterator的内核都需要特殊处理布尔型输出数据类型，并提供类型为(scalar_t, scalar_t -> bool)的lambda表达式
  if (out.defined() && out.scalar_type() != kBool) {
    config.cast_common_dtype_to_outputs(true);
  }
}

// 构建比较操作的张量迭代器
void TensorIteratorBase::build_comparison_op(
    const TensorBase& out, const TensorBase& a, const TensorBase& b) {
  // 创建一个名为 config 的 TensorIteratorConfig 对象，用于配置张量迭代器的参数
  TensorIteratorConfig config;
  // 调用函数 set_up_comparison_op_config，将输出张量 out 与比较运算相关的配置信息填入 config 中
  set_up_comparison_op_config(config, out);

  // 将输出张量 out 添加到 config 的所拥有的输出张量列表中
  config.add_owned_output(out);
  // 将输入张量 a 添加到 config 的所拥有的常量输入张量列表中
  config.add_owned_const_input(a);
  // 将输入张量 b 添加到 config 的所拥有的常量输入张量列表中
  config.add_owned_const_input(b);
  // 根据 config 中的配置信息构建张量迭代器
  build(config);
// 定义 TensorIteratorBase 类的 build_borrowing_comparison_op 方法，用于构建比较操作的迭代器
void TensorIteratorBase::build_borrowing_comparison_op(
    const TensorBase& out, const TensorBase& a, const TensorBase& b) {
  // 创建 TensorIteratorConfig 对象
  TensorIteratorConfig config;
  // 使用 out 张量设置比较操作的配置
  set_up_comparison_op_config(config, out);

  // 将 out 张量设置为借用的输出
  config.add_borrowed_output(out);
  // 将 a 张量设置为借用的常量输入
  config.add_borrowed_const_input(a);
  // 将 b 张量设置为借用的常量输入
  config.add_borrowed_const_input(b);
  // 使用配置对象构建迭代器
  build(config);
}

// 定义 TensorIteratorBase 类的 build_borrowing_except_last_argument_comparison_op 方法，用于构建比较操作迭代器
void TensorIteratorBase::build_borrowing_except_last_argument_comparison_op(
    const TensorBase& out, const TensorBase& a, const TensorBase& b) {
  // 创建 TensorIteratorConfig 对象
  TensorIteratorConfig config;
  // 使用 out 张量设置比较操作的配置
  set_up_comparison_op_config(config, out);

  // 将 out 张量设置为借用的输出
  config.add_borrowed_output(out);
  // 将 a 张量设置为借用的常量输入
  config.add_borrowed_const_input(a);
  // 将 b 张量设置为拥有的常量输入
  config.add_owned_const_input(b);
  // 使用配置对象构建迭代器
  build(config);
}

// 定义 TensorIteratorBase 类的 build_ternary_op 方法，用于构建三元操作的迭代器
void TensorIteratorBase::build_ternary_op(
    const TensorBase& out, const TensorBase& a,
    const TensorBase& b, const TensorBase& c) {
  // 使用 TensorIteratorConfig 的链式调用配置迭代器
  build(TensorIteratorConfig()
      .promote_inputs_to_common_dtype(true)  // 将输入提升为共同的数据类型
      .cast_common_dtype_to_outputs(true)    // 将共同的数据类型转换为输出类型
      .enforce_safe_casting_to_output(true)  // 强制安全地将类型转换为输出类型
      .add_owned_output(out)                // 添加拥有的输出张量
      .add_owned_const_input(a)             // 添加拥有的常量输入张量 a
      .add_owned_const_input(b)             // 添加拥有的常量输入张量 b
      .add_owned_const_input(c));           // 添加拥有的常量输入张量 c
}

// 定义宏 BINARY_OP_CONFIG，返回一个配置好的 TensorIteratorConfig 对象，用于构建二元操作的迭代器
#define BINARY_OP_CONFIG()                              \
  TensorIteratorConfig()                                \
    .set_check_mem_overlap(true)                        // 设置检查内存重叠
    .allow_cpu_scalars(true)                            // 允许 CPU 标量
    .promote_inputs_to_common_dtype(true)               // 将输入提升为共同的数据类型
    .cast_common_dtype_to_outputs(true)                 // 将共同的数据类型转换为输出类型
    .enforce_safe_casting_to_output(true)               // 强制安全地将类型转换为输出类型

// 定义 TensorIteratorBase 类的 build_binary_op 方法，用于构建二元操作的迭代器
void TensorIteratorBase::build_binary_op(const TensorBase& out, const TensorBase& a, const TensorBase& b) {
  // 使用宏 BINARY_OP_CONFIG 配置迭代器，并添加拥有的输出、拥有的常量输入 a 和 b
  build(BINARY_OP_CONFIG()
      .add_owned_output(out)
      .add_owned_const_input(a)
      .add_owned_const_input(b));
}

// 定义 TensorIteratorBase 类的 build_borrowing_binary_op 方法，用于构建借用的二元操作的迭代器
void TensorIteratorBase::build_borrowing_binary_op(
    const TensorBase& out, const TensorBase& a, const TensorBase& b) {
  // 使用宏 BINARY_OP_CONFIG 配置迭代器，并添加输出 out、常量输入 a 和 b
  build(BINARY_OP_CONFIG()
      .add_output(out)
      .add_const_input(a)
      .add_const_input(b));
}

// 定义宏 UNARY_FLOAT_OP_CONFIG，返回一个配置好的 TensorIteratorConfig 对象，用于构建浮点数的一元操作的迭代器
#define UNARY_FLOAT_OP_CONFIG()                                         \
  TensorIteratorConfig()                                                \
  .set_check_mem_overlap(true)                                          // 设置检查内存重叠
  .promote_inputs_to_common_dtype(true)                                 // 将输入提升为共同的数据类型
  .cast_common_dtype_to_outputs(true)                                   // 将共同的数据类型转换为输出类型
  .enforce_safe_casting_to_output(true)                                 // 强制安全地将类型转换为输出类型
  .promote_integer_inputs_to_float(true)                                // 将整数输入提升为浮点数

// 定义 TensorIteratorBase 类的 build_unary_float_op 方法，用于构建浮点数的一元操作的迭代器
void TensorIteratorBase::build_unary_float_op(const TensorBase& out, const TensorBase& a) {
  // 使用宏 UNARY_FLOAT_OP_CONFIG 配置迭代器，并添加拥有的输出和拥有的常量输入 a
  build(UNARY_FLOAT_OP_CONFIG()
      .add_owned_output(out)
      .add_owned_const_input(a));
}
// 构建基于一元浮点操作的张量迭代器，使用给定的输出张量和输入张量
void TensorIteratorBase::build_borrowing_unary_float_op(const TensorBase& out, const TensorBase& a) {
  // 使用 UNARY_FLOAT_OP_CONFIG 宏定义的配置构建迭代器
  build(UNARY_FLOAT_OP_CONFIG()
      // 添加输出张量
      .add_output(out)
      // 添加常量输入张量
      .add_const_input(a));
}

// 这不可能是一个函数，因为 TensorIteratorConfig 不可复制或可移动，所以无法从函数中返回它。
#define UNARY_OP_CONFIG()                                \
  // 创建一个新的 TensorIteratorConfig 对象，并设置以下配置项
  TensorIteratorConfig()                                 \
    // 检查内存重叠
    .set_check_mem_overlap(true)                         \
    // 将常见数据类型转换到输出张量的数据类型
    .cast_common_dtype_to_outputs(false)                 \
    // 强制安全转换到输出张量
    .enforce_safe_casting_to_output(false)               \
    // 检查所有张量的数据类型是否相同
    .check_all_same_dtype(true)

// 构建基于一元操作的张量迭代器，使用给定的输出张量和输入张量
void TensorIteratorBase::build_unary_op(const TensorBase& out, const TensorBase& a) {
  // 使用 UNARY_OP_CONFIG 宏定义的配置构建迭代器
  build(UNARY_OP_CONFIG()
      // 添加拥有的输出张量
      .add_owned_output(out)
      // 添加拥有的常量输入张量
      .add_owned_const_input(a));
}

// 构建基于一元操作的张量迭代器，使用给定的输出张量和输入张量
void TensorIteratorBase::build_borrowing_unary_op(const TensorBase& out, const TensorBase& a) {
  // 使用 UNARY_OP_CONFIG 宏定义的配置构建迭代器
  build(UNARY_OP_CONFIG()
      // 添加输出张量
      .add_output(out)
      // 添加常量输入张量
      .add_const_input(a));
}

// 构建输出张量借用输入张量但自身拥有的一元操作的张量迭代器
void TensorIteratorBase::build_output_borrowing_argument_owning_unary_op(const TensorBase& out, const TensorBase& a) {
  // 使用 UNARY_OP_CONFIG 宏定义的配置构建迭代器
  build(UNARY_OP_CONFIG()
      // 添加输出张量
      .add_output(out)
      // 添加拥有的常量输入张量
      .add_owned_const_input(a));
}

// 辅助函数，构建强制将输出张量升级为布尔类型的一元操作的张量迭代器
// 仅在输出张量必须是布尔类型时使用
void TensorIteratorBase::build_borrowing_unary_force_boolean_op(const TensorBase& out, const TensorBase& a) {
  // 创建新的 TensorIteratorConfig 对象，并设置以下配置项
  build(TensorIteratorConfig()
      // 检查内存重叠
      .set_check_mem_overlap(true)
      // 不检查所有张量的数据类型是否相同
      .check_all_same_dtype(false)
      // 声明静态数据类型为布尔类型
      .declare_static_dtype(at::kBool)
      // 声明静态设备与输入张量的设备相同
      .declare_static_device(a.device())
      // 添加输出张量
      .add_output(out)
      // 添加常量输入张量
      .add_const_input(a));
}

// 创建二元操作的张量迭代器，使用给定的输出张量和两个输入张量
TensorIterator TensorIterator::binary_op(TensorBase& out, const TensorBase& a, const TensorBase& b) {
  // 创建新的 TensorIterator 对象
  TensorIterator iter;
  // 使用给定的输出张量和输入张量构建二元操作的张量迭代器
  iter.build_binary_op(out, a, b);
  // 返回构建的迭代器对象
  return iter;
}

// 创建借用两个输入张量的二元操作的张量迭代器，使用给定的输出张量
TensorIterator TensorIterator::borrowing_binary_op(
    const TensorBase& out, const TensorBase& a, const TensorBase& b) {
  // 创建新的 TensorIterator 对象
  TensorIterator iter;
  // 使用给定的输出张量和输入张量构建借用输入张量的二元操作的张量迭代器
  iter.build_borrowing_binary_op(out, a, b);
  // 返回构建的迭代器对象
  return iter;
}

// 创建二元浮点操作的张量迭代器，使用给定的输出张量和两个输入张量
TensorIterator TensorIterator::binary_float_op(TensorBase& out, const TensorBase& a, const TensorBase& b) {
  // 创建新的 TensorIterator 对象
  TensorIterator iter;
  // 使用给定的输出张量和输入张量构建二元浮点操作的张量迭代器
  iter.build_binary_float_op(out, a, b);
  // 返回构建的迭代器对象
  return iter;
}

// 创建比较操作的张量迭代器，使用给定的输出张量和两个输入张量
TensorIterator TensorIterator::comparison_op(TensorBase& out, const TensorBase& a,
    const TensorBase& b) {
  // 创建新的 TensorIterator 对象
  TensorIterator iter;
  // 使用给定的输出张量和输入张量构建比较操作的张量迭代器
  iter.build_comparison_op(out, a, b);
  // 返回构建的迭代器对象
  return iter;
}

// 创建一元操作的张量迭代器，使用给定的输出张量和输入张量
TensorIterator TensorIterator::unary_op(TensorBase& out, const TensorBase& a) {
  // 创建新的 TensorIterator 对象
  TensorIterator iter;
  // 使用给定的输出张量和输入张量构建一元操作的张量迭代器
  iter.build_unary_op(out, a);
  // 返回构建的迭代器对象
  return iter;
}

// 创建一元浮点操作的张量迭代器，使用给定的输出张量和输入张量
TensorIterator TensorIterator::unary_float_op(TensorBase& out, const TensorBase& a) {
  // 创建新的 TensorIterator 对象
  TensorIterator iter;
  // 使用给定的输出张量和输入张量构建一元浮点操作的张量迭代器
  iter.build_unary_float_op(out, a);
  // 返回构建的迭代器对象
  return iter;
}

// 定义 NULLARY_OP_CONFIG 宏，创建只检查内存重叠的张量迭代器配置
#define NULLARY_OP_CONFIG()                                     \
  // 创建新的 TensorIteratorConfig 对象，并设置以下配置项
  TensorIteratorConfig()                                        \
    // 检查内存重叠
    .set_check_mem_overlap(true)
    .check_all_same_dtype(false)                                \
    # 设置参数：检查所有张量的数据类型是否相同，这里设置为 false 表示不进行检查
  /* FIXME: workaround for bug: https://github.com/pytorch/pytorch/issues/20342 */ \
    # 使用的是暂时的解决方案来解决 bug：https://github.com/pytorch/pytorch/issues/20342
    .resize_outputs(false)
    # 设置参数：是否调整输出的大小，这里设置为 false 表示不调整输出的大小
// 返回一个使用给定输出的空操作的 TensorIterator 对象
TensorIterator TensorIterator::nullary_op(TensorBase& out) {
  // 调用宏 NULLARY_OP_CONFIG() 返回一个配置对象，然后添加给定的输出，并构建 TensorIterator 对象
  return NULLARY_OP_CONFIG()
    .add_owned_output(out)
    .build();
}

// 返回一个使用给定输出的借用空操作的 TensorIterator 对象
TensorIterator TensorIterator::borrowing_nullary_op(const TensorBase& out) {
  // 调用宏 NULLARY_OP_CONFIG() 返回一个配置对象，然后添加给定的输出（以借用方式），并构建 TensorIterator 对象
  return NULLARY_OP_CONFIG()
    .add_output(out)
    .build();
}

// 返回一个使用给定输出和输入的缩减操作的 TensorIterator 对象
TensorIterator TensorIterator::reduce_op(TensorBase& out, const TensorBase& a) {
  // 内部断言，确保输出 out 已定义
  TORCH_INTERNAL_ASSERT(out.defined());
  // 创建 TensorIteratorConfig 对象，设置一系列配置选项，并构建 TensorIterator 对象
  return TensorIteratorConfig()
    .set_check_mem_overlap(false)  // 设置内存重叠检查为 false
    .add_owned_output(out)  // 添加一个拥有的输出 out
    .add_owned_const_input(a)  // 添加一个拥有的常量输入 a
    .resize_outputs(false)  // 设置不调整输出大小
    .is_reduction(true)  // 标记为缩减操作
    .promote_inputs_to_common_dtype(true)  // 提升输入到公共数据类型
    .build();
}

// 返回一个使用给定输出和输入的缩减操作的 TensorIterator 对象（带两个输出）
TensorIterator TensorIterator::reduce_op(TensorBase& out1, TensorBase& out2, const TensorBase& a) {
  // 内部断言，确保输出 out1 和 out2 已定义，并且输入 a 和输出在同一设备上，并具有相同的维度、大小和步长
  TORCH_INTERNAL_ASSERT(out1.defined());
  TORCH_INTERNAL_ASSERT(out2.defined());
  TORCH_CHECK(a.device() == out1.device() && out1.device() == out2.device(),
      "reduce_op(): expected input and both outputs to be on same device, but input is on ", a.device(),
      ", output1 is on ", out1.device(), " and output2 is on", out2.device());
  TORCH_CHECK(out1.dim() == out2.dim(), "reduce_op(): expected both outputs to have same number of dims, but output1 has ", out1.dim(),
      " and output2 has ", out2.dim());
  TORCH_CHECK(out1.sizes() == out2.sizes(), "reduce_op(): expected both outputs to have same sizes, but output1 has ", out1.sizes(),
      " and output2 has ", out2.sizes());
  TORCH_CHECK(out1.strides() == out2.strides(), "reduce_op(): expected both outputs to have same strides, but output1 has ", out1.strides(),
      " and output2 has ", out2.strides());
  // 创建 TensorIteratorConfig 对象，设置一系列配置选项，并构建 TensorIterator 对象
  return TensorIteratorConfig()
    .set_check_mem_overlap(false)  // 设置内存重叠检查为 false
    .add_owned_output(out1)  // 添加一个拥有的输出 out1
    .add_owned_output(out2)  // 添加一个拥有的输出 out2
    .add_owned_const_input(a)  // 添加一个拥有的常量输入 a
    .resize_outputs(false)  // 设置不调整输出大小
    .is_reduction(true)  // 标记为缩减操作
    .check_all_same_dtype(false)  // 设置不检查所有的输出是否具有相同的数据类型
    .build();
}

// 将操作数填充到 TensorIteratorConfig 对象中
void TensorIteratorBase::populate_operands(TensorIteratorConfig& config) {
  // 遍历配置对象中的张量列表
  for (const auto idx : c10::irange(config.tensors_.size())) {
    auto& tensor = config.tensors_[idx];
    // 如果任何一个参数是元张量，则整体计算是元计算（只计算输出信息），与多重分派语义一致
    if (tensor->is_meta()) {
      is_meta_ = true;
    }
    // 将张量移动到操作数列表中，并标记其是否是常量
    operands_.emplace_back(std::move(tensor));
    operands_[idx].is_const = config.is_tensor_const(idx);
  }
  // 设置操作数的数量为配置对象中的输出数量
  num_outputs_ = config.num_outputs_;
}

// 标记输出张量
void TensorIteratorBase::mark_outputs() {
  // 遍历输出的数量
  for (const auto i : c10::irange(num_outputs_)) {
    // 标记第 i 个操作数为输出
    operands_[i].is_output = true;
    const auto& output = tensor(i);
    // 如果输出未定义，则继续下一个循环
    if (!output.defined()) continue;

    // 检查输出是否也是输入的一部分
    # 遍历范围从 num_outputs_ 到 ntensors() 之间的索引，每次迭代的元素存储在 arg 中
    for (const auto arg : c10::irange(num_outputs_, ntensors())) {
      # 获取索引为 arg 的输入张量的引用
      const auto& input = tensor(arg);
      # 如果输出张量 output 与当前输入张量 input 相同
      if (output.is_same(input)) {
        # 将第 i 个操作数标记为读写操作
        operands_[i].is_read_write = true;
      }
    }
  }
void TensorIteratorBase::mark_resize_outputs(const TensorIteratorConfig& config) {
  // 检查是否存在静态形状，如果有则不需要调整输出大小，直接返回
  if (config.static_shape_.has_value()) {
    return;
  }
  // 遍历所有输出张量，检查是否需要调整其大小
  for (const auto i : c10::irange(num_outputs_)) {
    // 获取当前输出张量的引用
    const auto& output = tensor(i);
    // 如果当前输出张量未定义，标记其将会被调整大小
    if (!output.defined()) {
      operands_[i].will_resize = true;
    }
    // 如果当前输出张量已定义，并且其形状与推断的形状不匹配
    if (output.defined() && !output.sizes().equals(shape_)) {
      // 如果配置允许调整输出大小，并且当前操作数不是读写操作，则标记将调整大小
      if (config.resize_outputs_ && !operands_[i].is_read_write) {
        operands_[i].will_resize = true;
        continue;
      }
      // 对于缩减操作，输出大小与形状不匹配，抛出错误信息
      TORCH_CHECK(is_reduction_, "output with shape ", output.sizes(), " doesn't match the broadcast shape ",
                 shape_);
    }
  }
}

void TensorIteratorBase::compute_mem_overlaps(const TensorIteratorConfig& config) {
  // 如果配置中不需要检查内存重叠，则直接返回
  if (!config.check_mem_overlap_) {
    return;
  }
  // 遍历所有输出张量，检查它们之间的内存重叠情况
  for (const auto i : c10::irange(num_outputs_)) {
    // 获取当前输出张量的引用
    const auto& output = tensor_base(i);
    // 如果当前输出张量未定义，则跳过
    if (!output.defined()) continue;
    // 断言当前输出张量没有内部重叠
    assert_no_internal_overlap(output);
    // 进一步遍历所有输入张量，检查它们与当前输出张量之间的部分重叠
    for (const auto j : c10::irange(num_outputs_, ntensors())) {
      const auto& input = tensor_base(j);
      // 如果输入张量与当前输出张量不是同一个对象，则检查它们之间的部分重叠
      if (!input.is_same(output)) {
        assert_no_partial_overlap(output, input);
      }
    }
  }
}

void TensorIteratorBase::compute_shape(const TensorIteratorConfig& config) {
  // 如果配置中包含静态形状，则直接使用该静态形状，并返回
  if (config.static_shape_.has_value()) {
    shape_ = *config.static_shape_;
    return;
  }

  // 初始化标志和变量用于形状计算
  all_ops_same_shape_ = true;
  bool has_scalars = false;
  bool has_tensors = false;
  // 遍历所有操作数
  for (auto& op : operands_) {
    // 如果当前操作数的基础张量未定义，则跳过
    if (!op.tensor_base().defined()) continue;

    // 当调整输出大小时，不包括输出张量。这些形状不参与形状计算。
    // 这保留了旧行为，其中 torch.add(..., out=dst) 会调整目标张量的大小。
    // 如果输出张量同时也是输入，则稍后会在操作数中处理它。
    if (config.resize_outputs_ && op.is_output) continue;
    
    // 断言当前操作数的基础张量不包含符号大小和步幅
    TORCH_CHECK(!op.tensor_base().unsafeGetTensorImpl()->has_symbolic_sizes_strides(),
      "TensorIterator does not support symbolic shapes; please implement this operator in torch/_refs "
      "using the elementwise or reduction helpers (look at backtrace to find out what operator this is)");
    
    // 获取当前操作数的基础张量形状
    auto shape = op.tensor_base().sizes();
    // 如果形状为空，说明是标量
    if (shape.empty()) {
      has_scalars = true;
    } else {
      has_tensors = true;
    }
    // 如果同时存在标量和张量，则不是所有操作数具有相同的形状
    if (has_scalars && has_tensors) {
      all_ops_same_shape_ = false;
    }
    // 如果当前形状为空，直接使用当前形状
    if (shape_.empty()) {
      shape_ = shape;
    } else if (!shape.equals(shape_)) {
      // 如果当前形状与已记录形状不相同，则不是所有操作数具有相同的形状
      all_ops_same_shape_ = false;
      // 推断当前形状与已记录形状的大小向量
      shape_ = infer_size_dimvector(shape_, shape);
      // 确保形状是可推断的
    }
  }
}
    }
  }
  // 检查是否所有的操作数都是标量，当没有张量时为真
  all_ops_are_scalars_ = !has_tensors;
}

void TensorIteratorBase::compute_strides(const TensorIteratorConfig& config) {
  // 对于操作数列表中的每个操作数
  for (auto& op : operands_) {
    // 如果操作数的基础张量已定义且不会调整大小
    if (op.tensor_base().defined() && !op.will_resize) {
      // 获取原始形状，根据是否是静态形状选择 shape_ 或 op.tensor_base().sizes()
      IntArrayRef original_shape = config.static_shape_ ? shape_ : op.tensor_base().sizes();
      auto original_stride = op.tensor_base().strides(); // 获取原始步长
      auto element_size_in_bytes = op.tensor_base().element_size(); // 获取每个元素的字节大小
      auto offset = ndim() - original_shape.size(); // 计算偏移量
      if (offset > 0)
          op.stride_bytes.resize(ndim(), 0); // 如果偏移量大于0，调整步长数组大小并填充0
      else
          op.stride_bytes.resize(ndim()); // 否则只调整步长数组大小
      for (const auto i : c10::irange(original_shape.size())) {
        // 见注释：[计算输出步长]
        if (original_shape[i] == 1 && shape_[offset + i] !=1) {
          op.stride_bytes[offset + i] = 0; // 如果原始形状为1且当前形状不为1，则步长置为0
        } else {
          op.stride_bytes[offset + i] = original_stride[i] * element_size_in_bytes; // 计算步长
        }
      }
    }
  }
}

bool TensorIteratorBase::can_use_32bit_indexing() const {
  int64_t max_value = std::numeric_limits<int32_t>::max(); // 获取int32_t的最大值
  if (numel() > max_value) {
    return false; // 如果元素数量大于最大值，无法使用32位索引
  }
  for (auto& op : operands_) {
    int64_t max_offset = 1;
    for (const auto dim : c10::irange(ndim())) {
      max_offset += (shape_[dim] - 1) * op.stride_bytes[dim]; // 计算最大偏移量
    }
    if (max_offset > max_value) {
      return false; // 如果最大偏移量大于最大值，无法使用32位索引
    }
  }
  return true; // 可以使用32位索引
}

std::unique_ptr<TensorIterator> TensorIteratorBase::split(int dim) {
  TORCH_INTERNAL_ASSERT(dim >= 0 && dim < ndim() && shape()[dim] >= 2); // 断言维度有效且形状大于等于2
  auto copy = std::make_unique<TensorIterator>(*this); // 创建当前对象的副本

  bool overlaps = is_dim_reduced(dim); // 检查是否存在重叠的维度缩减
  auto copy_size = shape_[dim] / 2; // 计算副本大小
  auto this_size = shape_[dim] - copy_size; // 计算当前对象大小
  copy->narrow(dim, 0, copy_size); // 对副本进行窄化操作
  copy->final_output_ &= !overlaps; // 更新副本的最终输出状态
  this->narrow(dim, copy_size, this_size); // 对当前对象进行窄化操作
  this->accumulate_ |= overlaps; // 更新当前对象的累加状态

  return copy; // 返回副本
}


int TensorIteratorBase::get_dim_to_split() const {
  TORCH_INTERNAL_ASSERT(ndim() >= 1); // 断言至少有1个维度
  int64_t max_extent = -1; // 最大范围初始化为-1
  int dim_to_split = -1; // 待分割的维度初始化为-1
  for (int dim = ndim() - 1; dim >= 0; dim--) {
    const int64_t size = shape_[dim]; // 获取当前维度的大小
    if (size == 0) {
      continue; // 如果大小为0，则继续下一次循环
    }
    for (auto& op : operands_) {
      // std::abs 是为了处理一些特殊情况，其中我们支持负步长，见 at::flip 的 CUDA 后端
      const int64_t extent = (size - 1) * std::abs(op.stride_bytes[dim]); // 计算范围
      if (extent > max_extent) {
        max_extent = extent; // 更新最大范围
        dim_to_split = dim; // 更新待分割的维度
      }
    }
  }
  TORCH_INTERNAL_ASSERT(max_extent >= 0); // 断言最大范围非负
  return dim_to_split; // 返回待分割的维度
}

bool TensorIteratorBase::fast_set_up(const TensorIteratorConfig& config) {
  // 此函数尝试进行快速设置，以避免不必要地重新排序维度和跟踪输出步长
  // 如果可以进行快速设置，则返回true，否则返回false
  // TODO: 启用对约简操作的快速处理
  FastSetupType setup_type = compute_fast_setup_type(config); // 计算快速设置类型
  if (setup_type == FastSetupType::NONE) {
  // 返回 false 表示函数执行失败
  return false;
}

// 根据 setup_type 分配输出的内存空间，内存格式取决于 setup_type
switch (setup_type) {
  case FastSetupType::CONTIGUOUS:
    {
      // 对于每个输出，设置为连续内存格式
      for (const auto i : c10::irange(num_outputs_)) {
        auto& op = operands_[i];
        // 如果操作数对应的基本张量未定义，抛出断言错误
        if (!op.tensor_base().defined()) {
          TORCH_INTERNAL_ASSERT(op.is_type_defined(), "no type for operand", i);
        }
        // 设置输出为原始选项的连续内存格式
        set_output_raw_strided(i, shape_, {}, original_options(op).memory_format(MemoryFormat::Contiguous), names_);
      }
      break;
    }
  case FastSetupType::CHANNELS_LAST:
    {
      // 对于每个输出，设置为通道最后的内存格式
      for (const auto i : c10::irange(num_outputs_)) {
        auto& op = operands_[i];
        // 如果操作数对应的基本张量未定义，抛出断言错误
        if (!op.tensor_base().defined()) {
          TORCH_INTERNAL_ASSERT(op.is_type_defined(), "no type for operand", i);
        }
        // 设置输出为原始选项的通道最后内存格式
        set_output_raw_strided(i, shape_, {}, original_options(op).memory_format(MemoryFormat::ChannelsLast), names_);
      }
      break;
    }
  case FastSetupType::NON_OVERLAPPING_DENSE:
    {
      // 在操作数中找到从输入张量开始的定义张量的索引
      int i_defined; // NOLINT(cppcoreguidelines-init-variables)
      for (i_defined = ntensors() - 1; i_defined >= 0; --i_defined) {
        if (tensor(i_defined).defined()) break;
      }
      // 如果找不到定义的张量，抛出检查错误
      TORCH_CHECK(i_defined >= 0, "Can not find a defined tensor when fast allocating memory to outputs");
      // 对于每个输出，设置为与定义张量相同的步幅和原始选项
      for (const auto i : c10::irange(num_outputs_)) {
        auto& op = operands_[i];
        // 如果操作数对应的基本张量未定义，抛出断言错误
        if (!op.tensor_base().defined()) {
          TORCH_INTERNAL_ASSERT(op.is_type_defined(), "no type for operand", i);
        }
        // 设置输出为原始选项，使用与定义张量相同的步幅
        set_output_raw_strided(i, shape_, tensor_base(i_defined).strides(), original_options(op), names_);
      }
      break;
    }
  default:
    // 如果 setup_type 不被支持，抛出断言错误
    TORCH_INTERNAL_ASSERT(false, "Unsupported fast setup type", std::to_string((int)setup_type));
}

// 如果张量的维度大于 1，则表明存在合并的维度
if (ndim() > 1){
  has_coalesced_dimensions_ = true;
}

// 如果张量的维度大于或等于 1，则将形状第一个维度设置为元素数量，并且调整形状为只有一个维度
if (ndim() >= 1) {
  shape_[0] = numel();
  shape_.resize(1);
}

// 为每个操作数设置步幅字节数
for (auto& op : operands_ ) {
  auto element_size_in_bytes = op.tensor_base().element_size();
  op.stride_bytes.resize(ndim());
  // 如果张量的维度大于 0，则设置第一个维度的步幅为元素大小的字节数
  if (ndim()>0) {
    op.stride_bytes[0] = element_size_in_bytes;
  }
}

// 返回 true 表示函数执行成功
return true;
}

// 计算快速设置类型的函数，根据配置信息决定迭代器的快速设置类型
FastSetupType TensorIteratorBase::compute_fast_setup_type(const TensorIteratorConfig& config) {
  // 如果是归约操作或者操作数形状不全相同，则无法进行快速设置
  if (is_reduction_ || !all_ops_same_shape_) {
    return FastSetupType::NONE;
  }

  // 对于线性迭代，只有连续的张量可以被合并
  // 其它格式的快速设置需要改变迭代顺序
  if (enforce_linear_iteration_) {
    for (const auto& op : operands_) {
      if (op.tensor_base().defined() && !op.will_resize) {
        // 检查张量是否是连续的
        auto is_contiguous = op.tensor_base().is_contiguous(at::MemoryFormat::Contiguous);
        if (!is_contiguous) {
          return FastSetupType::NONE;
        }
      }
    }
    return FastSetupType::CONTIGUOUS;
  }

  // 初始化标志变量
  bool is_contiguous = true;
  bool is_channels_last = true;
  bool is_non_overlapping_and_dense = true;

  // 遍历所有操作数，检查张量的连续性、通道末尾内存格式以及非重叠稠密性
  for (const auto& op : operands_) {
    if (op.tensor_base().defined() && !op.will_resize) {
      // 检查张量是否是连续的
      is_contiguous &= op.tensor_base().is_contiguous(at::MemoryFormat::Contiguous);
      // 检查张量是否是通道末尾内存格式的
      is_channels_last &= op.tensor_base().is_contiguous(at::MemoryFormat::ChannelsLast);
      // 检查张量是否是非重叠稠密的
      is_non_overlapping_and_dense &= op.tensor_base().is_non_overlapping_and_dense();
    }
  }

  // 处理特定的情况，这里会对NC11这样的不明确情况视为连续
  if (is_contiguous) {
    return FastSetupType::CONTIGUOUS;
  }

  // 如果是通道末尾内存格式
  if (is_channels_last) {
    return FastSetupType::CHANNELS_LAST;
  }

  // 如果是非重叠稠密张量
  if (is_non_overlapping_and_dense) {
    int64_t prev = -1;
    // 快速设置仅允许当所有已定义张量具有相同形状和步幅时
    // 从后向前迭代，首先检查输入张量的步幅，然后是输出张量的步幅
    for (int64_t i = ntensors() - 1; i >= 0; --i) {
      const auto& op = operands_[i];
      if (op.tensor_base().defined() && !op.will_resize) {
        if (prev < 0) {
          prev = i;
          continue;
        }
        // 如果输入张量的步幅与前一个不同，则不支持快速设置
        if (!tensor_base(prev).strides().equals(op.tensor_base().strides())) {
          // [注意：快速设置中非连续张量的步幅检查]
          // 在这里阻止三种情况的快速设置：
          // 1. 输入张量具有不同的步幅。
          // 2. 输出张量不会被调整大小，并且具有不同的步幅。
          // 3. 输入张量具有相同的步幅，但输出张量与输入张量具有不同的步幅。
          //    在这种情况下，我们不允许重新步进输出张量，因为这与 numpy 不兼容。
          //    在 numpy 中，如果输出张量的形状与输入张量相同但步幅不同，则保留输出张量的步幅，
          //    我们在张量迭代器中也做同样的处理。
          return FastSetupType::NONE;
        }
      }
    }
    return FastSetupType::NON_OVERLAPPING_DENSE;
  }

  // 如果以上条件都不满足，则返回无法进行快速设置
  return FastSetupType::NONE;
}

// 默认构造函数定义
TensorIteratorBase::TensorIteratorBase() = default;
// 基于给定的配置构建张量迭代器
void TensorIteratorBase::build(TensorIteratorConfig& config) {
  // 将一些持久性配置字段填充
  is_reduction_ = config.is_reduction_;
  enforce_linear_iteration_ = config.enforce_linear_iteration_;

  // 根据配置填充操作数 operands_
  populate_operands(config);
  // 标记适当张量的 is_output 和 is_read_write 标志
  mark_outputs();
  // 检查输出张量不存在内部重叠并且不与输入共享内存
  compute_mem_overlaps(config);
  // 检查输入维度是否正确对齐，并计算输出名称
  compute_names(config);
  // 计算广播形状
  compute_shape(config);
  // 如果需要，标记输出张量用于调整大小
  mark_resize_outputs(config);
  // 计算结果的数据类型和设备
  compute_types(config);
  // 尝试快速设置输出张量，如果失败则回退到正常设置
  if (!fast_set_up(config)) {
    // 计算每个张量在广播后的步长
    compute_strides(config);
    // 重新排序维度以改善合并
    reorder_dimensions();
    // 如果未设置元数据，分配或调整输出张量的大小
    allocate_or_resize_outputs();
    // 在可能时合并相邻维度
    if (!is_meta_) coalesce_dimensions();
  }

  // 如果是元函数，直接返回
  if (is_meta_) return;

  // 检查所有操作数是否具有存储
  auto has_storage = true;
  for (auto& op : operands_) {
    has_storage &= op.tensor_base().has_storage();
  }

  // 检查是否是 PrivateUse1 设备且没有存储
  auto privateuse1_without_storage =
     common_device_.type() == DeviceType::PrivateUse1 &&
     !has_storage;

  // XLA 和 Lazy 张量没有存储，因此没有底层数据指针
  // 对于元设备，到此为止的内容不重要，可以提前退出
  // 将条件扩展到 MAIA 张量，因为 MAIA 张量也没有存储
  if (privateuse1_without_storage  ||
      common_device_.type() == DeviceType::MTIA ||
      common_device_.type() == DeviceType::XLA  ||
      common_device_.type() == DeviceType::IPU  ||
      common_device_.type() == DeviceType::Lazy ||
      common_device_.type() == DeviceType::MAIA  ||
      common_device_.type() == DeviceType::HPU) return;

  // 对每个操作数执行以下操作
  for (auto& op : operands_) {
    // 断言操作数的基础张量已定义
    TORCH_INTERNAL_ASSERT(op.tensor_base().defined());
    if (op.is_const) {
      // 如果是常量，将数据指针设为常量数据指针
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      op.data = const_cast<void*>(op.tensor_base().const_data_ptr());
    } else {
      // 否则，将数据指针设为可变数据指针
      op.data = op.tensor_base().mutable_data_ptr();
    }
  }

  // 将偏移量归零
  // 如果张量是标量，我们留出空间供它使用
  // 因此在缩减中的索引转换可以访问有效值的偏移量
  int64_t ndim_offsets = (ndim() ? ndim() : 1);
  view_offsets_ = DimVector(ndim_offsets, 0);
}

// 这是结构化内核实现的 set_output 函数。
// 它从不直接调用；而是 TensorIteratorBase 的子类将覆盖 set_output 以执行操作，
// 然后调用 TensorIteratorBase 的 set_output 来设置 TI 的元数据。
// 设置输出张量的原始分步信息，需要确保 maybe_get_output() 现在无条件返回一个有效的张量
void TensorIteratorBase::set_output_raw_strided(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options, DimnameList names) {
  // 获取输出张量在操作数列表中的引用
  auto& op = operands_[output_idx];
  // 断言输出索引在有效范围内
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(output_idx < num_outputs_);
  // 获取输出张量的引用
  const auto& t = maybe_get_output(output_idx);
  // 断言输出张量是定义好的
  TORCH_INTERNAL_ASSERT(t.defined());
  
  // 如果操作数的基础张量未定义，则设置为 maybe_get_output() 返回的张量
  if (!op.tensor_base().defined()) {
    op.tensor(c10::MaybeOwned<TensorBase>::borrowed(t));
    // 断言操作数的目标数据类型与张量的标量类型相同
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(op.target_dtype == t.scalar_type());
  } 
  // 如果操作数将被调整大小
  else if (op.will_resize) {
    // 省略部分代码
    // 检查原始张量是否已定义
    if (op.original_tensor_base().defined()) {
      // 这里的情况比较特殊。要理解为何会出现这种情况，首先看一下标记 [Output original_tensor is set]。
      // 这是唯一的地方，原始张量可能会被设置为输出操作数。基本上，当我们给定一个明确的输出张量，
      // 其数据类型与从输入操作数计算得到的公共数据类型不匹配时，我们进行了一种替换：
      // 我们用一个正确类型的临时张量替换（原本类型不正确的）输出张量，并将原始张量记忆在 original_tensor 中
      // （在 cast_outputs 时将写回）。

      // 现在，如果给定的输出张量恰好也是零大小（意味着我们将会调整其大小）呢？
      // 在上述调用点上，我们不一定(*) 知道正确的形状应该是什么，因此我们将临时张量设置为与原始张量相同的形状。
      // 在调用 set_output 时，我们才知道正确的大小，并且在结构化类的实现中负责调整 original_tensor 的大小。
      // 但我们仍然有这个尺寸不正确的临时输出张量，结构化子类对此一无所知，因此我们也有义务在这里调整其大小。

      // 这略微浪费了内存，因为以前 original_tensor 只在计算结束时才调整大小，而不是在开始时（如此处所示）。
      // 但是，峰值内存使用量是相同的，因为你需要实现原始张量和临时张量的复制。

      // (*) 实际上，技术上来说，我们可能确实知道形状应该是什么，因为我们在数据类型计算之前进行形状计算。
      // 所以在那个时间点上我们理论上可以弄清楚正确的形状，并直接在正确的大小上分配临时张量。

      // 但更好的解决方案是推迟临时张量的分配，直到 TensorIterator 构建器之后，等到我们实际想要执行计算。
      // 那样也会消除对 is_meta_ 测试的必要性。
      TORCH_INTERNAL_ASSERT(op.original_tensor_base().is_same(t));
      TORCH_INTERNAL_ASSERT(!op.tensor_base().is_same(t));

      // 获取操作的可选张量引用，并调整输出张量的大小为给定的 sizes
      OptionalTensorRef tensor(op.tensor());
      at::native::resize_output(*tensor, sizes);

      // 如果 strides 不为空，则将张量视为分步张量
      if (!strides.empty()) {
        // 确保 options.memory_format_opt() 没有值
        TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
        // 使用给定的 sizes 和 strides 对张量进行分步处理
        tensor->as_strided_(sizes, strides);
      } else if (options.memory_format_opt().has_value()) {
        // 否则，如果 options.memory_format_opt() 有值，则对张量进行空张量重排列
        tensor->unsafeGetTensorImpl()->empty_tensor_restride(*options.memory_format_opt());
      }
    }
  }
  // 在调试模式下，确保 op.tensor_base() 与 t 相同，或者 op.current_dtype 等于 op.tensor_base().scalar_type()
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      op.tensor_base().is_same(t) || op.current_dtype == op.tensor_base().scalar_type());
// 简单起见，始终更新缓存的当前数据类型。
op.current_dtype = op.tensor_base().scalar_type();



// 这是 set_output 的传统实现。在 TensorIterator 实例中，它直接从该文件中的各个调用点调用。
// 没有任何奇怪的操作。
void TensorIterator::set_output_raw_strided(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options, DimnameList names) {
  // 注意：这里有意没有调用超类的方法
  auto& op = operands_[output_idx];
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(output_idx < num_outputs_);
  if (!op.tensor_base().defined()) {
      // 如果操作数对应的基础张量未定义
      if (strides.empty()) {
        // 根据给定的大小和选项创建一个空张量
        op.tensor(c10::MaybeOwned<TensorBase>::owned(at::empty(sizes, options)));
      } else {
        // 根据给定的大小、步长和选项创建一个空步长张量
        op.tensor(c10::MaybeOwned<TensorBase>::owned(at::empty_strided(sizes, strides, options)));
      }
      // 更新当前数据类型为目标数据类型
      op.current_dtype = op.target_dtype;
  } else if (op.will_resize) {
      // 如果操作数将调整大小
      // 调整操作数对应的张量大小
      at::native::resize_output(op.tensor(), sizes);
      if (!strides.empty()) {
        // 如果步长不为空，断言选项的内存格式没有值
        TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
        // 将张量视图为步长张量
        op.tensor().as_strided_(sizes, strides);
      } else if (options.memory_format_opt().has_value()) {
        // 如果内存格式选项有值，通过不安全的方式重新调整张量
        op.tensor_base().unsafeGetTensorImpl()->empty_tensor_restride(*options.memory_format_opt());
      }
  }
  if (!names.empty()) {
    // 如果名称列表不为空，断言操作数对应的基础张量已定义
    TORCH_INTERNAL_ASSERT(op.tensor_base().defined());
    // 传播名称推断
    namedinference::propagate_names(op.tensor_base(), names);
  }
}



// 实际上没有被任何东西使用（TensorIterator 子类调用自己的 set_output 实现，知道所有输出的确切位置），
// 但是我们必须为 MetaBase 提供所有纯虚方法
const Tensor& TensorIterator::maybe_get_output(int64_t output_idx) {
  return output(output_idx);
}



// with_32bit_indexing 返回一个 SplitUntil32Bit 对象，用于支持 32 位索引的迭代器
SplitUntil32Bit TensorIteratorBase::with_32bit_indexing() const {
  return SplitUntil32Bit(*this);
}



/// SplitUntil32Bit. 递归地将一个迭代器分割成可以使用 32 位索引的子迭代器。
SplitUntil32Bit::iterator::iterator(const TensorIteratorBase& iter) {
  // 将一个新的 TensorIterator 添加到 vec 中
  vec.emplace_back(new TensorIterator(iter));
  // 添加一个空指针，因为 ++ 操作会弹出最后一个元素
  vec.emplace_back(nullptr);
  // 执行 ++ 操作以准备迭代
  ++(*this);
}

// SplitUntil32Bit::iterator 的前进操作
SplitUntil32Bit::iterator& SplitUntil32Bit::iterator::operator++() {
  // 弹出最后一个元素
  vec.pop_back();
  // 当 vec 不为空且最后一个元素不能使用 32 位索引时，继续循环
  while (!vec.empty() && !vec.back()->can_use_32bit_indexing()) {
    // 获取最后一个迭代器，并进行分割操作
    auto& iter = *vec.back();
    auto split_dim = iter.get_dim_to_split();
    vec.emplace_back(iter.split(split_dim));
  }
  return *this;
}

// 返回当前迭代器所指向的 TensorIterator 对象的引用
TensorIterator& SplitUntil32Bit::iterator::operator*() const {
  return *vec.back();
}

// 返回 SplitUntil32Bit 对象的开始迭代器
SplitUntil32Bit::iterator SplitUntil32Bit::begin() const {
  return SplitUntil32Bit::iterator(iter);
}

// 返回 SplitUntil32Bit 对象的结束迭代器
SplitUntil32Bit::iterator SplitUntil32Bit::end() const {
  return SplitUntil32Bit::iterator();
}



// DimCounter 构造函数，初始化 shape、range、values 和 offset
DimCounter::DimCounter(IntArrayRef shape, Range range)
  : shape(shape)
  , range(range)
  , values(shape.size())
  , offset(range.begin) {
  // 将 values 中所有元素填充为 0
  std::fill(values.begin(), values.end(), 0);
  // 如果 range 的起始值为 0
  if (range.begin == 0) {
    // 如果程序执行到这里，直接返回，不进行后续的操作
    return;
  }

  // 初始化线性偏移量为指定范围的起始位置
  int64_t linear_offset = range.begin;
  // 获取数组的维度数量
  auto ndim = values.size();
  // 遍历每一个维度
  for (const auto dim : c10::irange(ndim)) {
    // 获取当前维度的大小
    int64_t size = shape[dim];
    // 如果当前维度大小大于零
    if (size > 0) {
      // 计算当前维度上的索引值并存入对应位置的数组中
      values[dim] = linear_offset % size;
      // 更新线性偏移量，继续计算下一个维度的索引
      linear_offset /= size;
    }
  }
  // 断言线性偏移量应为零，即所有维度的索引计算完毕
  TORCH_INTERNAL_ASSERT(linear_offset == 0);
}

// 检查当前偏移量是否超过或等于范围的结束位置，确定是否完成
bool DimCounter::is_done() const {
    return offset >= range.end;
}

// 根据给定步长增加偏移量，并更新多维计数器的值
void DimCounter::increment(const std::array<int64_t, 2>& step) {
    // 根据步长更新偏移量
    offset += step[0] * step[1];
    // 获取多维计数器的维度数
    auto ndim = values.size();
    // 初始化溢出值为步长的第一个元素
    int64_t overflow = step[0];
    size_t i = 0;
    // 如果步长的第二个元素不为1，则进入条件分支
    if (step[1] != 1) {
        // 断言第一个维度的形状和值为0
        TORCH_INTERNAL_ASSERT(step[0] == shape[0] && values[0] == 0);
        i = 1;
        overflow = step[1];
    }
    // 遍历每个维度，处理溢出
    for (; i < ndim && overflow > 0; i++) {
        auto size = shape[i];
        auto prev = values[i];
        auto value = prev + overflow;
        // 如果值大于等于当前维度的大小，则发生溢出
        if (value >= size) {
            overflow = 1;
            value -= size;
            // 断言修正后的值仍小于当前维度的大小
            TORCH_INTERNAL_ASSERT(value < size);
        } else {
            overflow = 0;
        }
        // 更新多维计数器的当前维度值
        values[i] = static_cast<int64_t>(value);
    }
    // 最终确保溢出值只能是0或1
    TORCH_INTERNAL_ASSERT(overflow == 0 || overflow == 1);
}

// 计算在当前状态下最大的二维步长
std::array<int64_t, 2> DimCounter::max_2d_step() const {
    // 计算第一个步长，限制为当前维度的剩余空间和偏移量的剩余空间
    int64_t step0 = std::min(shape[0] - values[0], range.end - offset);
    int64_t step1 = 1;
    // 如果第一个步长等于第一个维度的大小且形状非空，则计算第二个步长
    if (step0 == shape[0] && !shape.empty()) {
        step1 = std::min(shape[1] - values[1], (range.end - offset) / shape[0]);
    }
    // 返回计算得到的最大二维步长
    return {step0, step1};
}

}  // namespace at
```