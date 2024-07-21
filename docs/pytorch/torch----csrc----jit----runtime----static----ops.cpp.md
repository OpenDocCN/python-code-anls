# `.\pytorch\torch\csrc\jit\runtime\static\ops.cpp`

```py
#include <torch/csrc/jit/runtime/static/ops.h>

#include <ATen/CPUFunctions.h>
#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorUtils.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/Fill.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/TensorConversions.h>
#include <ATen/native/cpu/SerialStackImpl.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qembeddingbag.h>
#include <ATen/native/quantized/cpu/qembeddingbag_prepack.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/ScalarType.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/symbolic_shape_runtime_fusion.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/csrc/jit/runtime/static/processed_node_wrapper.h>
#include <torch/csrc/jit/runtime/static/te_wrapper.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <iterator>
#include <mutex>
#include <unordered_map>

#include <ATen/CompositeExplicitAutogradFunctions.h>

// 定义一个布尔标志，控制是否启用静态运行时的快速数学优化
C10_DEFINE_bool(
    static_runtime_enable_fast_math,
    true,
    "If on, static runtime may use use optimizations that cause accuracy loss "
    "vs the jit interpreter");

namespace at::native {

// 静态函数，实现对张量的重复操作，结果存储在 result 中
static void repeat_out(
    at::Tensor& result,
    const Tensor& self,
    IntArrayRef repeats) {
  // 检查重复维度的数量不能少于张量自身的维度数量
  TORCH_CHECK(
      repeats.size() >= static_cast<size_t>(self.dim()),
      "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor");

  // 如果目标维度数大于源张量的维度数，则向张量添加新的前导维度
  int64_t num_new_dimensions = repeats.size() - self.dim();
  DimVector padded_size(num_new_dimensions, 1);
  padded_size.insert(
      padded_size.end(), self.sizes().begin(), self.sizes().end());
  DimVector target_size(repeats.size());
  bool zero_tensor = false;
  for (const auto idx : c10::irange(repeats.size())) {
    if (repeats[idx] == 0) {
      zero_tensor = true;
    }
    target_size[idx] = padded_size[idx] * repeats[idx];
  }

  // 如果存在任何重复维度为零，则返回一个空张量
  at::native::resize_(result, target_size, c10::nullopt);
  if (zero_tensor) {
    return;
  }

  // 使用 self 和 padded_size 创建一个扩展后的张量 xtensor
  Tensor xtensor = at::compositeexplicitautograd::expand(self, padded_size);
  // 创建一个指向结果的别名张量 urtensor
  Tensor urtensor = at::native::alias(result);
  // 遍历 xtensor 的每个维度
  for (const auto i : c10::irange(xtensor.dim())) {
    // 如果步长为 0，则无法展开，将步长至少设置为 1
    // (在这种情况下步长是多少并不重要，因为尺寸为 0)。
    urtensor = urtensor.unfold(
        i, xtensor.size(i), std::max<int64_t>(xtensor.size(i), 1));
  }

  // 将 xtensor 按 urtensor 的形状进行扩展，并复制到 urtensor
  at::native::copy_(urtensor, xtensor.expand_as(urtensor));
}

// copy version of view ops
// 将输入张量按照指定的形状重新调整大小，并将结果复制到输出张量中
at::Tensor& reshape_copy_out(
    at::Tensor& out,
    const at::Tensor& self,
    const at::DimVector& proposed_shape,
    bool infer_size) {
  // 根据需要推断出实际的形状
  const auto& shape = infer_size
      ? at::infer_size_dv(proposed_shape, self.numel())
      : proposed_shape;
  // 调整输出张量的大小为推断得到的形状
  at::native::resize_(out, shape, c10::nullopt);

  // 获取输入张量的连续版本
  auto self_contig = self.expect_contiguous();

  // 计算输入张量的字节数
  size_t nbytes = self.nbytes();
  // 如果输入张量字节数为零，直接返回输出张量
  if (nbytes == 0) {
    return out;
  }

  // 获取输入张量和输出张量的数据指针，并进行内存复制
  const void* self_data = self_contig->const_data_ptr();
  void* out_data = out.mutable_data_ptr();
  memcpy(out_data, self_data, nbytes);

  // 返回输出张量
  return out;
}

// static function for flattening and copying output
// 将输入张量展平并复制到输出张量中
static at::Tensor& flatten_copy_out(
    at::Tensor& out,
    const at::Tensor& self,
    int64_t start_dim,
    int64_t end_dim) {
  // 处理负数的起始和结束维度索引
  start_dim =
      start_dim < 0 ? c10::maybe_wrap_dim(start_dim, self.dim()) : start_dim;
  end_dim = end_dim < 0 ? c10::maybe_wrap_dim(end_dim, self.dim()) : end_dim;
  // 检查起始维度和结束维度的有效性
  TORCH_CHECK(
      start_dim <= end_dim,
      "flatten() has invalid args: start_dim cannot come after end_dim");

  // 如果输入张量是零维，则将其调整为一维并复制到输出张量中
  if (self.dim() == 0) {
    return reshape_copy_out(out, self, at::DimVector{1}, false);
  }

  // 如果起始维度等于结束维度，将整个形状作为一维进行处理并复制到输出张量中
  if (start_dim == end_dim) {
    auto shape = at::DimVector{self.sizes()};
    return reshape_copy_out(out, self, shape, false);
  }

  // 计算展平后切片的元素数目
  auto iter = self.sizes().data();
  auto slice_numel = std::accumulate(
      iter + start_dim,
      iter + end_dim + 1,
      static_cast<int64_t>(1),
      std::multiplies<int64_t>());

  // 构建展平后的形状
  at::DimVector shape;
  shape.reserve(self.dim() - end_dim + start_dim);
  for (const auto i : c10::irange(start_dim)) {
    shape.push_back(self.sizes()[i]);
  }
  shape.push_back(slice_numel);
  for (int64_t i = end_dim + 1; i < self.dim(); i++) {
    shape.push_back(self.sizes()[i]);
  }
  // 调用reshape_copy_out函数进行复制操作，并返回输出张量
  return reshape_copy_out(out, self, shape, false);
}

namespace {

// This is annoying and sily, but it's solving a real problem: the
// _MSC_VER version causes an ICE on our old clang5 builds. The
// non-_MSC_VER version is a syntax error according to MSVC. Use the
// appropriate version depending on if we're MSVC or not.

// 定义用于快速复制输出路径逻辑的宏，根据不同的编译环境使用不同版本的实现
#define TO_COPY_OUT_FAST_PATH_LOGIC(out, self, self_t)             \
  do {                                                             \
    // 获取输入张量的元素数目和数据指针
    const auto N = self.numel();                                   \
    const auto self_data = self.const_data_ptr<self_t>();          \
                                                                    \
    // 调用一个宏，展开为一个循环，处理所有类型（包括半精度、BFloat16、布尔类型）的张量
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(                        \
        kHalf,                                                     \
        kBFloat16,                                                 \
        kBool,                                                     \
        out.scalar_type(),                                         \
        "to_copy_out_inner_loop",                                  \
        [&]() {                                                    \
          // 获取输出张量的可变数据指针，类型为 scalar_t
          const auto out_data = out.mutable_data_ptr<scalar_t>();  \
          // 遍历索引范围为 N 的所有元素
          for (const auto idx : c10::irange(N)) {                  \
            /* NOLINTNEXTLINE(bugprone-signed-char-misuse) */      \
            // 将 self_data 中的数据转换为 scalar_t 类型后，赋值给 out_data 对应位置
            out_data[idx] = static_cast<scalar_t>(self_data[idx]); \
          }                                                        \
        });                                                        \
  } while (0)
#ifdef _MSC_VER
// 如果编译器是 Visual Studio，定义一个模板函数 to_copy_out_fast_path，用于快速路径的数据复制
template <typename T>
void to_copy_out_fast_path(Tensor& out, const Tensor& self) {
  // 调用宏 TO_COPY_OUT_FAST_PATH_LOGIC 处理数据复制的逻辑
  TO_COPY_OUT_FAST_PATH_LOGIC(out, self, T);
}

// 定义宏 TO_COPY_OUT_FAST_PATH_BODY，根据不同的编译环境使用不同的实现
#define TO_COPY_OUT_FAST_PATH_BODY(out, self) \
  to_copy_out_fast_path<scalar_t>(out, self)
#else
// 如果不是 Visual Studio 编译环境，使用默认的实现方式
#define TO_COPY_OUT_FAST_PATH_BODY(out, self) \
  using self_t = scalar_t;                    \
  TO_COPY_OUT_FAST_PATH_LOGIC(out, self, self_t)
#endif
} // namespace

// 实现函数 to_copy_out，用于将 self 张量的数据复制到 out 张量中
at::Tensor& to_copy_out(
    Tensor& out,
    const Tensor& self,
    bool non_blocking,
    bool copy_strides,
    std::optional<MemoryFormat> memory_format) {
  // 如果需要复制 strides
  if (copy_strides) {
    // 调用 resize_impl_cpu_ 函数重新调整 out 张量的大小和 strides
    at::native::resize_impl_cpu_(
        out.unsafeGetTensorImpl(), self.sizes(), self.strides());
  } else {
    // 否则调用 resize_ 函数调整 out 张量的大小，不考虑 strides
    at::native::resize_(out, self.sizes(), c10::nullopt);
  }
  // 定义 lambda 函数检查是否为不支持的数据类型
  auto is_unsupported_dtype = [](ScalarType t) {
    // 针对所有量化类型和复数类型，返回 true
#define TORCH_OPS_UNSUPPORTED_TYPE(_, type) \
  case k##type:                             \
    return true;
    switch (t) {
      // 对所有量化类型调用 TORCH_OPS_UNSUPPORTED_TYPE 宏
      AT_FORALL_QINT_TYPES(TORCH_OPS_UNSUPPORTED_TYPE)
      // 对所有复数类型调用 TORCH_OPS_UNSUPPORTED_TYPE 宏
      AT_FORALL_COMPLEX_TYPES(TORCH_OPS_UNSUPPORTED_TYPE)
      default:
        // 默认返回 false
        return false;
    }
#undef TORCH_OPS_UNSUPPORTED_TYPE
  };
  // 快速路径：可以直接复制数据吗？避免在 at::native::copy_ 中创建 TensorIterator，这相对较昂贵。
  if (self.is_contiguous() && !non_blocking &&
      // 用户请求复制的格式是不是连续的？
      (memory_format == c10::nullopt ||
       memory_format == c10::MemoryFormat::Preserve ||
       memory_format == c10::MemoryFormat::Contiguous) &&
      // CopyKernel.cpp 中会特殊处理此情况，所以我们不要干涉它。
      !self.is_neg() && !is_unsupported_dtype(self.dtype().toScalarType()) &&
      !is_unsupported_dtype(out.dtype().toScalarType()) &&
      !(
          // FBGEMM 优化可能会启用，不要干涉它。
          (self.dtype() == kFloat && out.dtype() == kHalf) ||
          (self.dtype() == kHalf && out.dtype() == kFloat))) {
    // 在所有类型和 kHalf、kBFloat16、kBool 类型上分派操作，调用 TO_COPY_OUT_FAST_PATH_BODY 处理快速路径
    AT_DISPATCH_ALL_TYPES_AND3(
        kHalf, kBFloat16, kBool, self.scalar_type(), "to_copy_out", [&]() {
          TO_COPY_OUT_FAST_PATH_BODY(out, self);
        });
    return out;
  }
  // 否则调用普通的复制函数 at::native::copy_
  at::native::copy_(out, self, non_blocking);
  return out;
}

// 实现函数 linear_out，用于执行线性操作并将结果保存到 output 张量中
static Tensor& linear_out(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias_opt) {
  // 检查 input 张量是否不是 MKL-DNN 张量
  TORCH_CHECK(!input.is_mkldnn());

  // 根据是否传入了 bias_opt，选择是否创建 bias 张量
  auto bias = bias_opt.has_value()
      ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
      : c10::MaybeOwned<Tensor>::owned(std::in_place);

  // 如果 input 张量的维度是 2 并且 bias 张量已定义，则使用融合的操作（addmm_out 函数）
  if (input.dim() == 2 && bias->defined()) {
    // 执行 addmm_out 函数，将结果保存到 output 张量中
    return at::cpu::addmm_out(output, *bias, input, weight.t());
  }
  // 否则调用普通的矩阵乘法操作 matmul_out 函数
  at::native::matmul_out(input, weight.t(), output);
  // 如果 bias 张量已定义，则执行加法操作
  if (bias->defined()) {
    at::cpu::add_(output, *bias);
  }
  // 返回结果张量 output
  return output;
}

// 实现函数 c2_argmin_out，用于计算输入张量 input 沿指定维度 dim 的最小值索引，并将结果保存到 output 张量中
static Tensor& c2_argmin_out(
    Tensor& output,
    const Tensor& input,
    const int64_t dim,
    // 定义函数 argmin_output，计算输入张量 input 沿指定维度 dim 的最小值索引，将结果输出到 output 张量中
    const bool keepdim) {
      // 获取输入张量的维度数
      const auto ndim = input.dim();
      // 计算有效的维度索引，确保不超出边界
      int64_t dim_ = maybe_wrap_dim(dim, ndim);
      TORCH_CHECK(dim_ >= 0 && dim_ < ndim);
    
      // 获取输入张量的各维度大小
      const auto in_dims = input.sizes();
    
      // 初始化输出张量的维度信息
      c10::SmallVector<int64_t, 5> out_dims;
      out_dims.reserve(ndim);
      int prev_size = 1;
      int next_size = 1;
      
      // 遍历维度 dim_ 之前的维度
      for (int i = 0; i < dim_; ++i) {
        out_dims.push_back(in_dims[i]);
        prev_size *= in_dims[i];
      }
      
      // 如果 keepdim 为真，则在输出维度中添加大小为 1 的维度
      if (keepdim) {
        out_dims.push_back(1);
      }
      
      // 遍历维度 dim_ 之后的维度
      for (auto i = dim_ + 1; i < ndim; ++i) {
        out_dims.push_back(in_dims[i]);
        next_size *= in_dims[i];
      }
      
      // 调整输出张量的尺寸为 out_dims
      at::native::resize_(output, out_dims, c10::nullopt);
    
      // 获取输入张量在维度 dim_ 上的大小
      const auto n = in_dims[dim_];
    
      // 如果 next_size 等于 1，则进行以下操作
      if (next_size == 1) {
        // 根据输入张量的数据类型执行以下操作
        AT_DISPATCH_ALL_TYPES_AND2(
            kHalf, kBFloat16, input.scalar_type(), "argmin_input", [&]() {
              // 获取输入张量和输出张量的指针
              const auto in_ptr = input.const_data_ptr<scalar_t>();
              const auto out_ptr = output.mutable_data_ptr<int64_t>();
              // 对于每个前缀大小 prev_size，执行以下操作
              for (int i = 0; i < prev_size; ++i) {
                // 在输入张量的第 i 个前缀区域中寻找最小值的索引
                auto v = std::min_element(
                    in_ptr + i * n,
                    in_ptr + (i + 1) * n,
                    [](scalar_t a, scalar_t b) {
                      // 如果 a 是 NaN，则按照 LessOrNan 语义认为 a 小于任何值
                      if (at::_isnan(a)) {
                        return true;
                      }
                      // 如果 a 不是 NaN 且 b 是 NaN，则认为 a 不小于 b
                      // 否则按照常规语义比较 a 和 b 的大小
                      return a < b;
                    });
                // 将最小值的索引写入输出张量中
                out_ptr[i] = std::distance(in_ptr + i * n, v);
              }
            });
      } else {
        // 如果 next_size 不等于 1，则执行以下操作
        AT_DISPATCH_ALL_TYPES_AND2(
            kHalf, kBFloat16, input.scalar_type(), "argmin_input", [&]() {
              // 定义用于比较的 LessOrNan 函数对象
              const auto less_or_nan = native::detail::LessOrNan<scalar_t>{};
    
              // 获取输入张量和输出张量的指针
              const auto in_ptr = input.const_data_ptr<scalar_t>();
              const auto out_ptr = output.mutable_data_ptr<int64_t>();
    
              // 将输出张量的内存清零
              std::memset(out_ptr, 0, prev_size * next_size * sizeof(int64_t));
    
              // 对于每个前缀大小 prev_size，执行以下操作
              for (int i = 0; i < prev_size; ++i) {
                // 获取当前输入张量的指针位置
                const scalar_t* cur_in_ptr = in_ptr + i * n * next_size + next_size;
                // 遍历输入张量中的每个元素
                for (int k = 1; k < n; ++k) {
                  // 遍历每个 next_size 大小的块
                  for (int j = 0; j < next_size; ++j) {
                    // 获取当前输出张量的指针位置
                    int64_t* cur_out_ptr = out_ptr + i * next_size + j;
                    // 使用 LessOrNan 函数对象比较当前输入值与历史最小值的大小
                    if (less_or_nan(
                            *cur_in_ptr,
                            in_ptr
                                [i * n * next_size + *cur_out_ptr * next_size + j],
                            *cur_out_ptr,
                            k)) {
                      // 如果当前输入值较小，则更新输出张量的对应位置为当前索引 k
                      *cur_out_ptr = k;
                    }
                    // 指针移动到下一个输入值
                    ++cur_in_ptr;
                  }
                }
              }
            });
      }
      // 返回计算结果的输出张量
      return output;
    }
} // 结束 namespace at::native

namespace torch::jit {

// 定义一个名为 SROperatorRegistry 的全局注册表，用于注册和查找单独运算符的函数对象
C10_DEFINE_REGISTRY(SROperatorRegistry, SROperatorFunctor);

// 检查给定的运算符名是否已注册
bool opIsRegistered(const c10::Symbol& op_name) {
  // 将运算符名转换为标准字符串格式
  const std::string name(op_name.toQualString());
  // 检查运算符名是否在注册表中
  return SROperatorRegistry()->Has(name);
}

// 静态函数，禁用不安全的数学运算符
static bool disableUnsafeMathOp(const char* op_name) {
  // 如果启用了快速数学运算，则返回 false
  if (FLAGS_static_runtime_enable_fast_math) {
    return false;
  }
  // 列出使用不保证位精确度的数学库或 NNC 的运算符
  static const c10::FastSet<std::string> fast_ops{
      "aten::add", "aten::tanh", "aten::sigmoid", "aten::logit"};
  // 检查给定的运算符名是否在快速运算符集合中
  return fast_ops.count(op_name) > 0;
}

// 根据节点获取对应的 out-of-place 操作对象
SROperator getOutOfPlaceOperation(Node* n) {
  // 获取节点的运算符名
  auto op_name = n->kind().toQualString();
  // 如果运算符已注册且不禁用该运算符的不安全性，则创建并返回该运算符的实例
  if (SROperatorRegistry()->Has(op_name) && !disableUnsafeMathOp(op_name)) {
    return SROperatorRegistry()->Create(op_name)->Generate(n);
  }
  // 否则返回空指针
  return nullptr;
}

// 检查节点是否具有可变参数
// 当节点的运算符是 prim::VarConcat 或 prim::VarStack 时返回 true
bool hasVarArgs(Node* n) {
  if (n->kind() == prim::VarConcat || n->kind() == prim::VarStack) {
    return true;
  }
  return false;
}

// 检查节点是否可以重用其输入和输出
bool canReuseInputsOutputs(
    Node* n,
    const c10::FastMap<Node*, bool>& node_has_out_variant) {
  // 查找节点是否已标记为具有输出变体
  auto it = node_has_out_variant.find(n);
  if (it != node_has_out_variant.end()) {
    return it->second;
  }
  // 如果节点没有输出变体，则尝试获取其 out-of-place 操作对象
  return getOutOfPlaceOperation(n) != nullptr;
}

// 检查节点的输入是否可以以 out-of-place 方式运行
static bool inputsCanRunOutOfPlace(
    Node* n,
    const c10::FastMap<Node*, bool>& node_has_out_variant) {
  // 遍历节点的每一个输入节点
  for (auto* input : n->inputs()) {
    // 如果输入节点的输入输出无法重用，则返回 false
    if (!canReuseInputsOutputs(input->node(), node_has_out_variant)) {
      return false;
    }
  }
  // 所有输入节点均能以 out-of-place 方式运行，返回 true
  return true;
}

// 检查节点的输出类型是否是可优化的容器类型
bool isOptimizableContainerType(
    Node* n,
    const c10::FastMap<Node*, bool>& node_has_out_variant) {
  // 获取节点的输出类型
  const auto& type = n->output()->type();
  bool is_supported_type = false;
  // 如果输出类型是列表类型，并且列表元素类型是 TensorType，则支持优化
  if (type->kind() == TypeKind::ListType) {
    const auto& list_type = type->expectRef<ListType>();
    is_supported_type =
        list_type.getElementType()->kind() == TypeKind::TensorType;
  } else if (type->kind() == TypeKind::TupleType) {
    // 如果输出类型是元组类型，则遍历元组的每个类型
    const auto& tuple_type = type->expectRef<TupleType>();
    auto types = tuple_type.containedTypes();
    // 检查元组中是否全部为 TensorType 类型
    is_supported_type = std::all_of(types.begin(), types.end(), [](const TypePtr& t) {
      return t->kind() == TypeKind::TensorType;
    });
  }
  // 返回是否支持优化的结果
  return is_supported_type;
}
    # 使用 std::find_if 在 types 容器中查找满足条件的元素
    const auto& iter =
        std::find_if(types.begin(), types.end(), [](const TypePtr& elem) {
          return elem->kind() == TypeKind::TensorType;
        });
    # 检查是否找到满足条件的元素
    is_supported_type = iter != types.end();
  }
  # 返回 is_supported_type 和 inputsCanRunOutOfPlace 函数的结果的逻辑与
  return is_supported_type && inputsCanRunOutOfPlace(n, node_has_out_variant);
}

static inline void listConstructSlowPath(
    const ListType& list_type,
    const size_t size,
    ProcessedNode* p_node) {
  // 创建一个空的列表，元素类型为 list_type 中指定的类型
  c10::List<IValue> vals(list_type.getElementType());
  // 预留空间以容纳 size 个元素
  vals.reserve(size);
  // 遍历 size 次，将 p_node 的输入作为列表的元素添加到 vals 中
  for (const auto i : c10::irange(size)) {
    vals.push_back(p_node->Input(i));
  }
  // 将 vals 赋值给 p_node 的输出
  p_node->Output(0) = vals;
}

bool sr_schema_check_kind(torch::jit::Node* node, c10::Symbol node_kind) {
  // 检查 node 的类型是否与 node_kind 匹配
  auto is_match = node->kind() == node_kind;
  // 如果不匹配，则记录并转储 node 的模式信息
  if (!is_match) {
    torch::jit::LogAndDumpSchema(node);
  }
  // 返回是否匹配的结果
  return is_match;
}

REGISTER_OPERATOR_FUNCTOR(
    prim::ListConstruct,
    prim_ListConstruct,
    [](Node* n) -> SROperator {
      // 检查节点 n 的类型是否为 prim::ListConstruct，如果不是返回空指针
      if (!sr_schema_check_kind(n, prim::ListConstruct)) {
        return nullptr;
      }
      // 检查是否可以优化容器类型的操作
      const bool can_optimize =
          isOptimizableContainerType(n, c10::FastMap<Node*, bool>());
      // 获取输出类型为 ListType 的期望引用
      const auto& type = n->output()->type()->expectRef<ListType>();
      // 获取输入的数量作为 size
      const size_t size = n->inputs().size();
      // 如果不能优化，则返回一个 lambda 函数，该函数执行慢速构造路径
      if (!can_optimize) {
        return [&type, size](ProcessedNode* p_node) {
          // 断言 p_node 的输入数量与 size 相等
          DCHECK(p_node->num_inputs() == size);
          // 执行慢速构造路径的函数调用
          listConstructSlowPath(type, size, p_node);
        };
      }
      // 如果可以优化，则返回一个 lambda 函数，该函数检查输出是否为空，如果为空则执行慢速构造路径
      return [&type, size](ProcessedNode* p_node) {
        DCHECK(p_node->num_inputs() == size);
        const auto& out_l = p_node->Output(0);
        if (!out_l.isNone()) {
          return;
        }
        listConstructSlowPath(type, size, p_node);
      };
    });

static inline void tupleConstructSlowPath(
    const size_t size,
    ProcessedNode* p_node) {
  // 准备输入数据
  switch (size) {
    // 根据元组大小选择不同的构造方式
    case 1:
      p_node->Output(0) = c10::ivalue::Tuple::create(p_node->Input(0));
      break;
    case 2:
      p_node->Output(0) =
          c10::ivalue::Tuple::create(p_node->Input(0), p_node->Input(1));
      break;
    case 3:
      p_node->Output(0) = c10::ivalue::Tuple::create(
          p_node->Input(0), p_node->Input(1), p_node->Input(2));
      break;
    default: {
      // 对于更大的元组，创建一个包含所有输入的值向量，然后创建一个元组
      std::vector<IValue> vals;
      vals.reserve(size);
      for (const auto i : c10::irange(size)) {
        vals.push_back(p_node->Input(i));
      }
      p_node->Output(0) = c10::ivalue::Tuple::create(std::move(vals));
      break;
    }
  }
}

REGISTER_OPERATOR_FUNCTOR(
    prim::TupleConstruct,
    prim_TupleConstruct,
    [](Node* n) -> SROperator {
      // 检查节点 n 的类型是否为 prim::TupleConstruct，如果不是返回空指针
      if (!sr_schema_check_kind(n, prim::TupleConstruct)) {
        return nullptr;
      }
      // 检查是否可以优化容器类型的操作
      const bool can_optimize =
          isOptimizableContainerType(n, c10::FastMap<Node*, bool>());
      // 获取输入的数量作为 size
      const size_t size = n->inputs().size();
      // 如果不能优化，则返回一个 lambda 函数，该函数执行慢速构造路径
      if (!can_optimize) {
        return [size](ProcessedNode* p_node) {
          DCHECK(p_node->num_inputs() == size);
          tupleConstructSlowPath(size, p_node);
        };
      }
      // 如果可以优化，则返回一个 lambda 函数，该函数检查输出是否为空，如果为空则执行慢速构造路径
      return [size](ProcessedNode* p_node) {
        DCHECK(p_node->num_inputs() == size);
        const auto& out_l = p_node->Output(0);
        if (!out_l.isNone()) {
          return;
        }
        tupleConstructSlowPath(size, p_node);
      };
    });
// 注册 ATen 操作符的函数，对应于 torch 中的 abs 操作
REGISTER_OPERATOR_FUNCTOR(aten::abs, aten_abs, [](Node* n) -> SROperator {
  // 检查节点 n 是否匹配给定的 aten::abs(Tensor self) -> Tensor 的模式
  if (!n->matches(torch::schema("aten::abs(Tensor self) -> Tensor"))) {
    // 如果不匹配，则记录并打印节点的模式信息，返回空指针
    LogAndDumpSchema(n);
    return nullptr;
  }
  // 匹配成功，返回一个 lambda 函数，处理节点
  return [](ProcessedNode* p_node) {
    // 获取输入节点的第一个张量
    const auto& in0_t = p_node->Input(0).toTensor();
    // 如果输出节点为 None，则计算 abs 的结果，并设置为输出节点的值
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::native::abs(in0_t);
      return;
    }
    // 如果输出节点已经有值，则重置为零长度，然后计算 abs 的结果到输出节点
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::native::abs_out(in0_t, out_t);
  };
});

// 注册 ATen 操作符的函数，对应于 torch 中的 mul 操作
REGISTER_OPERATOR_FUNCTOR(aten::mul, aten_mul, [](Node* n) -> SROperator {
  // 检查节点 n 是否匹配给定的 aten::mul.Tensor(Tensor self, Tensor other) -> Tensor 的模式
  if (!n->matches(torch::schema(
          "aten::mul.Tensor(Tensor self, Tensor other) -> Tensor"))) {
    // 如果不匹配，则记录并打印节点的模式信息，返回空指针
    LogAndDumpSchema(n);
    return nullptr;
  }

  // 匹配成功，返回一个 lambda 函数，处理节点
  return [](ProcessedNode* p_node) {
    // 获取输入节点的两个张量
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto& in1_t = p_node->Input(1).toTensor();
    // 如果输出节点为 None，则计算两个张量的乘积，并设置为输出节点的值
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::cpu::mul(in0_t, in1_t);
      return;
    }
    // 如果输出节点已经有值，则重置为零长度，然后计算乘积到输出节点
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::cpu::mul_out(out_t, in0_t, in1_t);
  };
});

// 注册 ATen 操作符的函数，对应于 torch 中的 addmm 操作
REGISTER_OPERATOR_FUNCTOR(aten::addmm, aten_addmm, [](Node* n) -> SROperator {
  // 检查节点 n 是否匹配给定的 aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor 的模式
  if (!n->matches(torch::schema(
          "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor"))) {
    // 如果不匹配，则记录并打印节点的模式信息，返回空指针
    LogAndDumpSchema(n);
    return nullptr;
  }
  // 匹配成功，返回一个 lambda 函数，处理节点
  return [](ProcessedNode* p_node) {
    // 获取输入节点的三个张量和两个标量参数
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto& in1_t = p_node->Input(1).toTensor();
    const auto& in2_t = p_node->Input(2).toTensor();
    const auto in3_s = p_node->Input(3).toScalar();
    const auto in4_s = p_node->Input(4).toScalar();
    // 如果输出节点为 None，则执行 addmm 操作，并设置为输出节点的值
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::cpu::addmm(in0_t, in1_t, in2_t, in3_s, in4_s);
      return;
    }
    // 如果输出节点已经有值，则重置为零长度，然后执行 addmm 操作到输出节点
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::cpu::addmm_out(out_t, in0_t, in1_t, in2_t, in3_s, in4_s);
  };
});

#ifdef FBCODE_CAFFE2
// 为 static_runtime::clamp_nan_to_num 注册 ATen 操作符的函数
REGISTER_OPERATOR_FUNCTOR(
    static_runtime::clamp_nan_to_num,
    static_runtime_clamp_nan_to_num,
    // 定义一个 lambda 函数，接受一个 Node* 参数，并返回一个 SROperator 对象
    [](Node* n) -> SROperator {
      // 检查节点 n 是否符合指定的静态运行时约束条件，如果不符合则返回空指针
      if (!sr_schema_check(
              n,
              "static_runtime::clamp_nan_to_num(Tensor input, Scalar? min, Scalar? max, float? nan, float? posinf, float? posinf) -> Tensor")) {
        return nullptr;
      }
      // 获取输入节点 n 的第一个和第二个输入值，并转换为可选的标量类型
      auto clamp_min_ival_opt = toIValue(n->input(1));
      auto clamp_max_ival_opt = toIValue(n->input(2));
      // 断言确保 clamp_min_ival_opt 和 clamp_max_ival_opt 都有值
      TORCH_CHECK(
          clamp_min_ival_opt.has_value() && clamp_max_ival_opt.has_value());
    
      // 将 clamp_min_ival_opt 和 clamp_max_ival_opt 转换为可选的 at::Scalar 类型，并断言有值
      auto clamp_min_opt = clamp_min_ival_opt->toOptional<at::Scalar>();
      auto clamp_max_opt = clamp_max_ival_opt->toOptional<at::Scalar>();
      TORCH_CHECK(clamp_min_opt.has_value() && clamp_max_opt.has_value());
    
      // 返回一个 lambda 函数，使用捕获的变量创建 clamp_nan_to_num 对象及其初始化参数
      return [te = createClampNanToNum(),
              clamp_min = clamp_min_opt->to<float>(),
              clamp_max =
                  clamp_max_opt->to<float>()](ProcessedNode* p_node) mutable {
        // 获取 p_node 的第一个输入，转换为 Tensor 引用
        const auto& in0_t = p_node->Input(0).toTensor();
        // 如果 p_node 的输出是空的，则使用输入张量创建一个空张量
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = create_empty_from(in0_t);
        }
        // 获取 p_node 的输出，并转换为 Tensor 引用
        auto& out_t = p_node->Output(0).toTensor();
        // 快速调整 out_t 的大小为零
        fastResizeToZero(out_t);
        // 获取 p_node 的第三个输入，并转换为可选的双精度浮点数
        auto in3_s = p_node->Input(3).toOptional<double>();
    
        // 如果 te 为空或者 te 无法处理 in0_t 的输入类型，则使用 CPU 端的 nan_to_num 函数进行计算
        if (!te || !te->checkInput<float>(in0_t)) {
          at::cpu::nan_to_num_out(
              out_t,
              at::cpu::clamp(in0_t, clamp_min, clamp_max),
              in3_s,
              c10::nullopt,
              c10::nullopt);
          return;
        }
        // 调整 out_t 的大小以匹配 in0_t 的尺寸
        at::native::resize_(out_t, in0_t.sizes(), c10::nullopt);
    
        // 计算输入张量的元素数量
        auto output_size = in0_t.numel();
    
        // 如果 in3_s 有值，则将其转换为 float 类型作为 nan 的值，否则设为 0.f
        auto nan = in3_s.has_value() ? static_cast<float>(*in3_s) : 0.f;
    
        // 调用 te 的 call 方法，传递所需的参数数组
        te->call(
            {out_t.data_ptr(),
             in0_t.data_ptr(),
             &clamp_min,
             &clamp_max,
             &nan,
             &output_size});
      };
    });
REGISTER_OPERATOR_FUNCTOR(aten::clamp, aten_clamp, [](Node* n) -> SROperator {
  // 检查节点是否匹配 clamp 函数的第一种模式
  if (n->matches(torch::schema(
          "aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor"))) {
    // 返回一个闭包函数，该函数实现 clamp 操作
    return [te = createClamp()](ProcessedNode* p_node) {
      // 获取输入张量
      const auto& in0_t = p_node->Input(0).toTensor();
      // 如果输出张量为空，创建一个与输入相同大小的空张量
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = create_empty_from(in0_t);
      }
      // 获取输出张量的引用
      auto& out_t = p_node->Output(0).toTensor();
      // 快速调整输出张量大小为零
      fastResizeToZero(out_t);
      // 获取输入的最小值和最大值，可能为空
      auto in1_s = p_node->Input(1).toOptional<at::Scalar>();
      auto in2_s = p_node->Input(2).toOptional<at::Scalar>();
      // 如果输入张量不符合特定类型要求，则调用 clamp_out 函数
      if (!te->checkInput<float>(in0_t)) {
        at::cpu::clamp_out(out_t, in0_t, in1_s, in2_s);
        return;
      }
      // 调整输出张量的大小与输入张量相同
      at::native::resize_(out_t, in0_t.sizes(), c10::nullopt);
      // 获取输出张量的元素数量
      auto output_size = in0_t.numel();
      // 获取最小值，如果未提供则为负无穷大
      auto min = in1_s.has_value() ? in1_s->toFloat()
                                   : -std::numeric_limits<float>::infinity();
      // 获取最大值，如果未提供则为正无穷大
      auto max = in2_s.has_value() ? in2_s->toFloat()
                                   : std::numeric_limits<float>::infinity();
      // 调用创建 clamp 操作的函数 te->call，并传入相应参数
      te->call({out_t.data_ptr(), in0_t.data_ptr(), &min, &max, &output_size});
    };
  }
  // 检查节点是否匹配 clamp 函数的第二种模式
  if (n->matches(
          "aten::clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor")) {
    // 返回一个闭包函数，该函数实现 clamp 操作
    return [](ProcessedNode* p_node) {
      // 获取输入张量
      const auto& in0_t = p_node->Input(0).toTensor();
      // 如果输出张量为空，创建一个与输入相同大小的空张量
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = create_empty_from(in0_t);
      }
      // 获取输出张量的引用
      auto& out_t = p_node->Output(0).toTensor();
      // 快速调整输出张量大小为零
      fastResizeToZero(out_t);
      // 获取输入的 min 和 max 张量，可能为空
      auto in1_t = p_node->Input(1).toOptional<at::Tensor>();
      auto in2_t = p_node->Input(2).toOptional<at::Tensor>();
      // 调用 clamp_out 函数进行 clamp 操作
      at::cpu::clamp_out(out_t, in0_t, in1_t, in2_t);
    };
  }
  // 如果节点不匹配任何模式，则记录并转储模式信息
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::bmm, aten_bmm, [](Node* n) -> SROperator {
  // 检查节点是否匹配 bmm 函数的模式
  if (!n->matches(
          torch::schema("aten::bmm(Tensor self, Tensor mat2) -> Tensor"))) {
    // 如果不匹配，则记录并转储模式信息
    LogAndDumpSchema(n);
    return nullptr;
  }
  // 返回一个闭包函数，该函数实现 bmm 操作
  return [](ProcessedNode* p_node) {
    // 获取输入张量
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto& in1_t = p_node->Input(1).toTensor();
    // 如果输出张量为空，创建一个与输入相同大小的空张量
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);
    }
    // 获取输出张量的引用
    auto& out_t = p_node->Output(0).toTensor();
    // 快速调整输出张量大小为零
    fastResizeToZero(out_t);
    // 调用 bmm_out 函数进行 bmm 操作
    at::cpu::bmm_out(out_t, in0_t, in1_t);
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::nan_to_num, aten_nan_to_num, [](Node* n) -> SROperator {
  // 检查节点是否匹配 nan_to_num 函数的模式
  if (!n->matches(torch::schema(
          "aten::nan_to_num(Tensor self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor"))) {
    // 如果不匹配，则记录并转储模式信息
    LogAndDumpSchema(n);
    return nullptr;
  }
  // 返回一个闭包函数，该函数实现 nan_to_num 操作
  return [](ProcessedNode* p_node) {
    // 获取输入张量
    const auto& in0_t = p_node->Input(0).toTensor();
    // 获取输入的 nan, posinf, neginf 值，可能为空
    const auto in1_d = p_node->Input(1).toOptional<double>();
    const auto in2_d = p_node->Input(2).toOptional<double>();
    const auto in3_d = p_node->Input(3).toOptional<double>();
    # 检查节点的第一个输出是否为空
    if (p_node->Output(0).isNone()) {
      # 如果为空，调用函数将输入张量和四个double类型的参数进行 NaN 替换操作，并将结果赋给节点的第一个输出
      p_node->Output(0) = at::native::nan_to_num(in0_t, in1_d, in2_d, in3_d);
      # 函数执行完毕，返回
      return;
    }
    # 如果节点的第一个输出不为空，获取其作为张量的引用
    auto& out_t = p_node->Output(0).toTensor();
    # 快速将输出张量的大小调整为零
    fastResizeToZero(out_t);
    # 使用指定的输入张量和四个double类型的参数，执行 NaN 替换操作，将结果写入输出张量中
    at::native::nan_to_num_out(in0_t, in1_d, in2_d, in3_d, out_t);
  };
});

namespace {

// 将结果张量按照指定维度重新调整大小，以容纳输入张量序列
void varStackSerialOut(
    at::Tensor& result,
    int64_t dim,
    const ProcessedNodeInputWrapper& inputs) {
  // 获取第一个输入张量的大小，并插入到指定维度位置
  auto result_sizes = inputs[0].sizes().vec();
  result_sizes.insert(result_sizes.begin() + dim, inputs.size());
  // 调整结果张量的大小
  at::native::resize_(result, result_sizes);

  // 根据结果张量的数据类型分发到相应的计算内核
  AT_DISPATCH_FLOATING_TYPES(
      result.scalar_type(), "varstack_serial_kernel", [&]() {
        // 调用具体的串行堆叠内核实现，处理结果张量和输入张量序列
        at::native::detail::
            stack_serial_kernel_impl<scalar_t, ProcessedNodeInputWrapper>(
                result, inputs, dim);
      });
}

// 在指定维度对输入张量序列进行unsqueeze操作，并返回结果向量
std::vector<at::Tensor> unsqueezeVarStackInputs(
    const ProcessedNodeInputWrapper& inputs,
    const int64_t dim) {
  std::vector<at::Tensor> result;
  result.reserve(inputs.size());
  for (const auto i : c10::irange(inputs.size())) {
    // 对每个输入张量在指定维度进行unsqueeze操作，并加入到结果向量中
    result.push_back(at::native::unsqueeze(inputs[i], dim));
  }
  return result;
}

// 将输入张量序列按照指定维度进行堆叠，并将结果写入指定的输出张量
void varstackNonserialOut(
    at::Tensor& result,
    const int64_t dim,
    const ProcessedNodeInputWrapper& inputs) {
  // 对输入张量序列在指定维度进行unsqueeze操作
  std::vector<at::Tensor> inputs_unsqueezed =
      unsqueezeVarStackInputs(inputs, dim);
  // 快速调整输出张量大小为零
  fastResizeToZero(result);
  // 在CPU上对unsqueeze后的输入张量序列进行cat操作，结果写入输出张量
  at::cpu::cat_outf(inputs_unsqueezed, dim, result);
}

// 将输入张量序列的第一个张量复制到输出张量中
void varStackFastOut(
    at::Tensor& out,
    int64_t dim,
    const ProcessedNodeInputWrapper& inputs) {
  // 断言输出张量是连续的
  DCHECK(out.is_contiguous());
  const auto num_inputs = static_cast<int64_t>(inputs.size());
  // 检查输入张量序列不为空
  TORCH_CHECK(num_inputs > 0, "stack expects a non-empty list of tensors");

  // 获取第一个输入张量的形状
  const auto first_tensor_shape = inputs[0].sizes();
  // 检查每个输入张量的形状是否与第一个张量相同
  for (const auto i : c10::irange(1, num_inputs)) {
    const auto shape = inputs[i].sizes();
    TORCH_CHECK(
        shape == first_tensor_shape,
        "Stack expects each tensor to be the same size, but got ",
        first_tensor_shape,
        " at position 0 and ",
        shape,
        " at position ",
        i);
  }

  // 根据堆叠维度选择输出张量的大小
  const std::array<int64_t, 2> output_size = (dim == 0 || dim == -2)
      ? std::array<int64_t, 2>{num_inputs, 1}
      : std::array<int64_t, 2>{1, num_inputs};

  // 调整输出张量的大小
  at::native::resize_(out, output_size, c10::nullopt);

  // 根据输出张量的数据类型分发到相应的计算内核
  AT_DISPATCH_ALL_TYPES(out.scalar_type(), "varStackFastOut", [&]() {
    auto* out_data = out.mutable_data_ptr<scalar_t>();
    // 将每个输入张量的第一个元素复制到输出张量中
    for (const auto i : c10::irange(num_inputs)) {
      auto& tensor = inputs[i];
      auto* input_ptr = tensor.const_data_ptr<scalar_t>();
      out_data[i] = *input_ptr;
    }
  });
}

// 检查输入张量序列是否都是标量
bool inputsAreScalars(const ProcessedNodeInputWrapper& inputs) {
  // 堆叠操作的所有输入张量应该具有相同的大小，因此只需检查第一个张量
  const auto& first_tensor = inputs[0];
  return first_tensor.sizes()[0] == 1 && first_tensor.dim() == 1;
}
void varStackOut(ProcessedNode& pnode, int64_t dim) {
  // 获取节点输入的数量
  const auto num_inputs = pnode.num_inputs();
  // 检查输入的张量数量是否大于1，否则抛出错误信息
  TORCH_CHECK(num_inputs > 1, "stack expects a non-empty list of tensors");
  // 根据输入的维度值和第一个输入张量的维度计算有效的堆叠维度
  dim = c10::maybe_wrap_dim(dim, pnode.Input(0).toTensor().dim() + 1);

  // 封装处理节点的输入
  auto inputs = ProcessedNodeInputWrapper(pnode);
  // 获取处理节点的输出张量的引用
  auto& output = pnode.Output(0).toTensor();

  // 如果输出张量是连续的并且所有输入都是标量，则调用快速堆叠函数并返回
  if (output.is_contiguous() && inputsAreScalars(inputs)) {
    varStackFastOut(output, dim, inputs);
    return;
  }

  // 检查是否可以使用串行堆叠函数进行操作
  bool can_use_serial = at::native::detail::CanUseNativeSerialStack<
      ProcessedNodeInputWrapper,
      /*skip_overlap_check*/ true>::call(output, inputs, dim);

  // 如果可以使用串行堆叠函数，则调用并返回
  if (can_use_serial) {
    varStackSerialOut(output, dim, inputs);
    return;
  }

  // 否则调用非串行堆叠函数
  varstackNonserialOut(output, dim, inputs);
}

} // namespace

// 为了满足 MSVC 预处理器的要求，将此部分拆分成一个函数
static SROperator aten_stack(Node* n) {
  // 检查节点是否匹配指定的 schema
  if (!n->matches(torch::schema(
          "aten::stack(Tensor[] tensors, int dim=0) -> Tensor"))) {
    // 记录并输出 schema 信息
    LogAndDumpSchema(n);
    return nullptr;
  }
  // 返回一个 lambda 函数，处理堆叠操作的具体逻辑
  return [](ProcessedNode* p_node) {
    // 获取输入张量列表
    const auto inputs = p_node->Input(0).toTensorVector();
    // 检查输入张量列表不能为空
    TORCH_CHECK(!inputs.empty(), "stack expects non-empty tensor list");
    // 获取堆叠的维度
    const auto dim = p_node->Input(1).toInt();
    // 如果输出张量为空，则创建一个空张量作为输出
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::native::_stack_cpu(inputs, dim);
      return;
    }
    // 否则获取输出张量的引用，并调整其大小为零
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    // 调用具体的 CPU 实现函数进行堆叠操作
    at::native::_stack_out_cpu(inputs, dim, out_t);
  };
}

// 注册操作函数符合
REGISTER_OPERATOR_FUNCTOR(aten::stack, aten_stack, aten_stack);

// 注册 VarStack 操作函数符合
REGISTER_OPERATOR_FUNCTOR(
    prim::VarStack,
    prim_VarStack,
    [](Node* n) -> SROperator {
      // 检查节点是否符合 VarStack 类型
      if (!sr_schema_check_kind(n, prim::VarStack)) {
        return nullptr;
      }
      // 返回一个 lambda 函数，处理 VarStack 操作的具体逻辑
      return [](ProcessedNode* p_node) {
        // 获取节点的输入数量
        const size_t num_inputs = p_node->num_inputs();
        // 获取堆叠的维度
        const auto dim = p_node->Input(num_inputs - 1).toInt();

        // 如果输出张量为空，则创建一个空张量作为输出
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = create_empty_from(p_node->Input(0).toTensor());
        }
        // 调用 varStackOut 函数进行堆叠操作
        varStackOut(*p_node, dim);
      };
    });

// 注册 leaky_relu 操作函数符合
REGISTER_OPERATOR_FUNCTOR(aten::leaky_relu, aten_leaky_relu, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 schema
  if (!n->matches(torch::schema(
          "aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor"))) {
    // 记录并输出 schema 信息
    LogAndDumpSchema(n);
    return nullptr;
  }
  // 返回一个 lambda 函数，处理 leaky_relu 操作的具体逻辑
  return [](ProcessedNode* p_node) {
    // 获取输入张量和斜率标量
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto in1_s = p_node->Input(1).toScalar();
    // 如果输出张量为空，则调用 CPU 实现的 leaky_relu 函数并返回结果
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::cpu::leaky_relu(in0_t, in1_s);
      return;
    }
    // 否则获取输出张量的引用，并调用 CPU 实现的 leaky_relu_out 函数
    auto& out_t = p_node->Output(0).toTensor();
    at::cpu::leaky_relu_out(out_t, in0_t, in1_s);
  };
});

// 注册 relu 操作函数符合
REGISTER_OPERATOR_FUNCTOR(aten::relu, aten_relu, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 schema
  if (!n->matches(torch::schema("aten::relu(Tensor self) -> Tensor"))) {
    // 记录并输出 schema 信息
    LogAndDumpSchema(n);
    return nullptr;
  }
  // 创建 relu 函数实例
  auto te = createRelu();
  // 返回一个 lambda 函数，处理 relu 操作的具体逻辑
  return [te](ProcessedNode* p_node) {
    // 实现省略，根据具体实现调用相关的 relu 函数
   `
    // 获取节点的第一个输入张量的常量引用
    const auto& in0_t = p_node->Input(0).toTensor();
    // 如果节点的第一个输出为空，则创建一个与第一个输入张量相同形状的空张量
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);
    }
    // 获取节点的第一个输出张量的引用
    auto& out_t = p_node->Output(0).toTensor();
    // 如果输入张量不是 float 类型，则进行处理
    if (!te->checkInput<float>(in0_t)) {
      // 将输出张量调整为零张量
      fastResizeToZero(out_t);
      // 对输出张量进行阈值操作，超过0的值保持不变，小于等于0的值设为0
      at::cpu::threshold_out(out_t, in0_t, 0, 0);
      // 退出函数
      return;
    }
    // 调整输出张量的尺寸与输入张量相同
    at::native::resize_(out_t, in0_t.sizes(), c10::nullopt);
    // 获取输入张量的元素数量
    int64_t nn = in0_t.numel();
    // 调用处理函数，传入输出张量数据指针、输入张量数据指针和元素数量
    te->call({out_t.data_ptr(), in0_t.data_ptr(), &nn});
  };
});

// 注册操作符的函数对象，处理 torch::aten::tanh 操作符
REGISTER_OPERATOR_FUNCTOR(aten::tanh, aten_tanh, [](Node* n) -> SROperator {
  // 如果节点 n 不匹配 aten::tanh 的定义，记录并输出其模式，返回空指针
  if (!n->matches(torch::schema("aten::tanh(Tensor self) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  // 创建 Tanh 操作符对象
  auto te = createTanh();
  // 返回一个 lambda 函数，该函数执行具体的操作
  return [te](ProcessedNode* p_node) {
    // 获取输入张量
    const auto& in0_t = p_node->Input(0).toTensor();
    // 如果输出张量为空，创建一个与输入张量相同大小的空张量
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);
    }
    // 获取输出张量的引用
    auto& out_t = p_node->Output(0).toTensor();
    // 如果输入张量类型不符合 float，快速调整输出张量大小为零，并使用 CPU 进行 tanh 运算
    if (!te->checkInput<float>(in0_t)) {
      fastResizeToZero(out_t);
      at::cpu::tanh_out(out_t, in0_t);
      return;
    }
    // 调整输出张量的大小与输入张量相同，不指定任何可选参数
    at::native::resize_(out_t, in0_t.sizes(), c10::nullopt);
    // 获取输入张量的元素数量
    int64_t nn = in0_t.numel();
    // 调用 Tanh 操作符的执行方法
    te->call({out_t.data_ptr(), in0_t.data_ptr(), &nn});
  };
});

// 注册操作符的函数对象，处理 prim::TensorExprDynamicGroup 操作符
REGISTER_OPERATOR_FUNCTOR(
    prim::TensorExprDynamicGroup,
    prim_TensorExprDynamicGroup,
    [](Node* n) -> SROperator {
      // 如果节点 n 不是 prim::TensorExprDynamicGroup 类型的，返回空指针
      if (!sr_schema_check_kind(n, prim::TensorExprDynamicGroup)) {
        return nullptr;
      }
      // 获取节点的子图属性，并创建相应的代码对象
      auto graph = n->g(attr::Subgraph);
      Code code(graph, "");
      // 返回一个 lambda 函数，该函数执行具体的操作
      return [code](ProcessedNode* p_node) {
        // 获取处理节点的输出数量
        auto num_outputs = p_node->num_outputs();
        // 创建一个栈来存储数据
        Stack stack;
        // 如果输出张量为空，分配足够的空间来存储输入
        if (p_node->Output(0).isNone()) {
          stack.reserve(p_node->num_inputs());
        } else {
          // 否则，分配足够的空间来存储输入和输出
          stack.reserve(p_node->num_inputs() + num_outputs);
          // 将输出放入栈中
          for (const auto& o : p_node->outputs()) {
            stack.emplace_back(o);
          }
        }
        // 将输入张量放入栈中
        for (auto i : c10::irange(p_node->num_inputs())) {
          stack.emplace_back(p_node->Input(i));
        }
        // 运行 TensorExprDynamicGroup 的代码
        runTensorExprDynamicGroup(code, stack);
        // 如果输出张量为空
        if (p_node->Output(0).isNone()) {
          // 确保栈中的输出数量与预期的数量相符
          TORCH_INTERNAL_ASSERT(
              stack.size() == num_outputs,
              "Unexpected # of outputs on stack after executing TensorExprDynamicGroup");
          // 将栈中的输出移动到处理节点的输出中
          for (auto i : c10::irange(num_outputs)) {
            p_node->Output(i) = std::move(stack[i]);
          }
        }
      };
    });

// 注册操作符的函数对象，处理 aten::sigmoid 操作符
REGISTER_OPERATOR_FUNCTOR(
    aten::sigmoid,
    aten_sigmoid,
    [](Node* n) -> SROperator {
      // 如果节点 n 不匹配 aten::sigmoid 的定义，记录并输出其模式，返回空指针
      if (!n->matches(torch::schema("aten::sigmoid(Tensor self) -> Tensor"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      // 创建 Sigmoid 操作符对象
      auto te = createSigmoid();
      // 返回一个 lambda 函数，该函数执行具体的操作
      return [te](ProcessedNode* p_node) {
        // 获取输入张量
        const auto& in0_t = p_node->Input(0).toTensor();
        // 如果输出张量为空，创建一个与输入张量相同大小的空张量
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = create_empty_from(in0_t);
        }
        // 获取输出张量的引用
        auto& out_t = p_node->Output(0).toTensor();
        // 如果输入张量类型不符合 float，快速调整输出张量大小为零，并使用 CPU 进行 sigmoid 运算
        if (!te->checkInput<float>(in0_t)) {
          fastResizeToZero(out_t);
          at::cpu::sigmoid_out(out_t, in0_t);
          return;
        }
        // 调整输出张量的大小与输入张量相同，不指定任何可选参数
        at::native::resize_(out_t, in0_t.sizes(), c10::nullopt);
        // 获取输入张量的元素数量
        int64_t nn = in0_t.numel();
        // 调用 Sigmoid 操作符的执行方法
        te->call({out_t.data_ptr(), in0_t.data_ptr(), &nn});
      };
    });
// 注册运算符functor函数aten::logit，对应的操作是aten_logit，lambda表达式返回SROperator
REGISTER_OPERATOR_FUNCTOR(aten::logit, aten_logit, [](Node* n) -> SROperator {
  // 检查节点n是否匹配给定的torch::schema，如果不匹配则记录并返回空指针
  if (!n->matches(torch::schema(
          "aten::logit(Tensor self, float? eps=None) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  // 初始化clamp值为无
  std::optional<float> clamp = c10::nullopt;
  // 如果第二个输入节点是常量节点
  if (n->inputs()[1]->node()->kind() == prim::Constant) {
    // 将第二个输入节点转换为IValue，再转换为可选的double值
    auto clamp_d = toIValue(n->inputs()[1])->toOptional<double>();
    // 如果转换成功，则将clamp设置为对应的float值，否则保持为无
    clamp = clamp_d
        ? c10::make_optional<float>(static_cast<float>(clamp_d.value()))
        : c10::nullopt;
  }
  // 根据clamp是否有值来创建te对象，或者设置为nullptr
  auto te = clamp ? createLogit() : nullptr;
  // 如果clamp有值，则设置clamp_value为其值，否则为0.0f
  float clamp_value = clamp ? *clamp : 0.0f;
  // 返回lambda函数，接收ProcessedNode指针p_node作为参数
  return [te, clamp_value](ProcessedNode* p_node) {
    // 获取输入节点0的Tensor引用
    const auto& in0_t = p_node->Input(0).toTensor();
    // 如果输出节点0为空，则创建一个与in0_t形状相同的空Tensor
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);
    }
    // 获取输出Tensor的引用
    auto& out_t = p_node->Output(0).toTensor();
    // 如果te为空或者te无法处理输入类型为float的情况
    if (!te || !te->checkInput<float>(in0_t)) {
      // 重新调整out_t为零大小
      fastResizeToZero(out_t);
      // 调用ATen的logit_out函数，处理输入in0_t和in1_d，结果保存到out_t中
      at::native::logit_out(in0_t, in1_d, out_t);
      return;
    }
    // 调整out_t的大小与in0_t相同
    at::native::resize_(out_t, in0_t.sizes(), c10::nullopt);
    // 获取in0_t的元素数量
    int64_t nn = in0_t.numel();
    // 设置c为clamp_value
    float c = clamp_value;
    // 调用te的call方法，传入参数为out_t.data_ptr()、in0_t.data_ptr()、元素数量nn、c
    te->call({out_t.data_ptr(), in0_t.data_ptr(), &nn, &c});
  };
});

// 注册运算符functor函数aten::clone，对应的操作是aten_clone，lambda表达式返回SROperator
REGISTER_OPERATOR_FUNCTOR(aten::clone, aten_clone, [](Node* n) -> SROperator {
  // 检查节点n是否匹配给定的torch::schema，如果不匹配则记录并返回空指针
  if (!n->matches(torch::schema(
          "aten::clone(Tensor self, *, MemoryFormat? memory_format=None) ->Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  // 返回lambda函数，接收ProcessedNode指针p_node作为参数
  return [](ProcessedNode* p_node) {
    // 获取输入节点0的Tensor引用
    const auto& src = p_node->Input(0).toTensor();
    // 获取输入节点1的可选MemoryFormat值，如果没有则使用Preserve
    const auto& optional_memory_format =
        p_node->Input(1).toOptional<c10::MemoryFormat>();
    auto memory_format =
        optional_memory_format.value_or(c10::MemoryFormat::Preserve);
    /*
      禁用stride = 0和内存格式不是preserve的情况下的out_variant克隆。执行动态分配，
      而不是为了简化实现而重用内存。我们可以原则上弄清楚strides的复制。
    */
    if ((at::has_internal_overlap(src.unsafeGetTensorImpl()) ==
         at::MemOverlap::Yes) ||
        (memory_format != c10::MemoryFormat::Preserve)) {
      // 如果源Tensor有内部重叠或者内存格式不是Preserve，则调用ATen的clone函数，结果保存到输出节点0中
      p_node->Output(0) = at::native::clone(src, memory_format);
      return;
    }
    // 如果输出节点0为空
    if (p_node->Output(0).isNone()) {
      if (src.is_non_overlapping_and_dense()) {
        // 复制所有的stride
        p_node->Output(0) =
            at::empty_strided(src.sizes(), src.strides(), src.options());
      } else {
        // 建议使用src的内存格式来创建空Tensor，并保存到输出节点0中
        memory_format = src.suggest_memory_format();
        p_node->Output(0) = create_empty_from(src, memory_format);
      }
    }
    // 获取输出Tensor的引用
    auto& out_t = p_node->Output(0).toTensor();
    // 调用ATen的resize_impl_cpu_函数，调整out_t的实现CPU内存，大小为src的sizes和strides
    at::native::resize_impl_cpu_(
        out_t.unsafeGetTensorImpl(), src.sizes(), src.strides());
    // 调用ATen的copy_函数，将src复制到out_t中
    at::native::copy_(out_t, src, false);
  };
});

// 注册运算符functor函数quantized::embedding_bag_byte_rowwise_offsets
// （此处代码片段省略，需要根据实际完整代码填写）
    quantized_embedding_bag_byte_rowwise_offsets,
    // 匿名函数，输入参数为节点指针 n，返回 SROperator 类型
    [](Node* n) -> SROperator {
      // 如果节点不符合特定的 Torch 脚本模式，则记录日志并返回空指针
      if (!n->matches(torch::schema(
              "quantized::embedding_bag_byte_rowwise_offsets(Tensor weight, Tensor indices, Tensor? offsets=None, bool scale_grad_by_freq=False, int mode=0, bool pruned_weights=False, Tensor? per_sample_weights=None, Tensor? compressed_indices_mapping=None, bool include_last_offset=False) -> Tensor"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      // 返回另一个匿名函数，输入为 ProcessedNode 指针，输出 SROperator 类型
      return [](ProcessedNode* p_node) {
        // 从输入节点中获取权重、索引和偏移量等参数
        const auto& weight = p_node->Input(0).toTensor();
        const auto& indices = p_node->Input(1).toTensor();
        const auto offsets = p_node->Input(2).toOptional<at::Tensor>();
        const auto pruned_weights = p_node->Input(5).toBool();
        const auto per_sample_weights =
            p_node->Input(6).toOptional<at::Tensor>();
        const auto compressed_indices_mapping =
            p_node->Input(7).toOptional<at::Tensor>();
        const auto include_last_offset = p_node->Input(8).toBool();
        // 如果输出张量为空，根据权重张量的类型创建一个空的输出张量
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = create_empty_from(weight, at::kFloat);
        }
        // 获取输出张量的引用
        auto& out_t = p_node->Output(0).toTensor();
        // 快速调整输出张量大小为零
        fastResizeToZero(out_t);
        // 调用 Torch 的嵌入计算函数，输出结果存储在 out_t 中
        return at::native::embedding_bag_byte_rowwise_offsets_out(
            out_t,
            weight,
            indices,
            offsets,
            false, // scale_grad_by_freq 参数未使用，设为 false
            0, // mode 参数未使用，设为 0
            pruned_weights,
            per_sample_weights,
            compressed_indices_mapping,
            include_last_offset);
      };
    });
// 注册量化版本的嵌入包操作，使用4比特行偏移
REGISTER_OPERATOR_FUNCTOR(
    quantized::embedding_bag_4bit_rowwise_offsets,
    embedding_bag_4bit_rowwise_offsets,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配特定的 Torch 脚本
      if (!n->matches(torch::schema(
              "quantized::embedding_bag_4bit_rowwise_offsets(Tensor weight, Tensor indices, Tensor? offsets=None, bool scale_grad_by_freq=False, int mode=0, bool pruned_weights=False, Tensor? per_sample_weights=None, Tensor? compressed_indices_mapping=None, bool include_last_offset=False) -> Tensor"))) {
        // 记录和转储不匹配的模式
        LogAndDumpSchema(n);
        return nullptr;
      }
      // 返回处理节点的 Lambda 函数
      return [](ProcessedNode* p_node) {
        // 获取输入张量和参数
        const auto& weight = p_node->Input(0).toTensor();
        const auto& indices = p_node->Input(1).toTensor();
        const auto offsets = p_node->Input(2).toOptional<at::Tensor>();
        const auto pruned_weights = p_node->Input(5).toBool();
        const auto per_sample_weights =
            p_node->Input(6).toOptional<at::Tensor>();
        const auto compressed_indices_mapping =
            p_node->Input(7).toOptional<at::Tensor>();
        const auto include_last_offset = p_node->Input(8).toBool();
        
        // 如果输出张量为空，则创建一个与 weight 类型相同的空张量
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = create_empty_from(weight, at::kFloat);
        }
        auto& out_t = p_node->Output(0).toTensor();
        
        // 快速调整输出张量大小为零
        fastResizeToZero(out_t);
        
        // 调用量化嵌入包操作的具体实现函数
        return at::native::embedding_bag_4bit_rowwise_offsets_out(
            out_t,
            weight,
            indices,
            offsets,
            false, // scale_grad_by_freq 参数未使用
            0,     // mode 参数未使用
            pruned_weights,
            per_sample_weights,
            compressed_indices_mapping,
            include_last_offset);
      };
    });

// 注册量化版本的预打包字节嵌入包操作
REGISTER_OPERATOR_FUNCTOR(
    quantized::embedding_bag_byte_prepack,
    embedding_bag_byte_prepack,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配特定的 Torch 脚本
      if (!n->matches(torch::schema(
              "quantized::embedding_bag_byte_prepack(Tensor weight) -> Tensor"))) {
        // 记录和转储不匹配的模式
        LogAndDumpSchema(n);
        return nullptr;
      }
      // 返回处理节点的 Lambda 函数
      return [](ProcessedNode* p_node) {
        // 获取输入权重张量
        const auto& weight = p_node->Input(0).toTensor();
        
        // 如果输出张量为空，则直接调用预打包字节嵌入包操作并返回
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = at::native::qembeddingbag_byte_prepack(weight);
          return;
        }
        
        // 否则，获取输出张量并快速调整大小为零
        auto& out_t = p_node->Output(0).toTensor();
        fastResizeToZero(out_t);
        
        // 调用预打包字节嵌入包操作的具体实现函数
        at::native::qembeddingbag_byte_prepack_out(out_t, weight);
      };
    });

// 注册 aten::narrow_copy 操作的操作函数，优先使用 out 变体
REGISTER_OPERATOR_FUNCTOR(aten::narrow_copy, aten_narrow_copy, [](Node* n) -> SROperator {
  // 检查节点是否匹配特定的 Torch 脚本
  if (!n->matches(torch::schema(
          "aten::narrow_copy(Tensor self, int dim, int start, int length) -> Tensor"))) {
    // 记录和转储不匹配的模式
    LogAndDumpSchema(n);
    return nullptr;
  }
  // 返回处理节点的 Lambda 函数
  return [](ProcessedNode* p_node) {
    // 获取输入张量和参数
    const auto& self = p_node->Input(0).toTensor(); // self
    const auto dim = p_node->Input(1).toInt();      // dim
    int64_t start = 0;
    
    // 如果输入参数 start 是标量，则转换为整数
    if (p_node->Input(2).isScalar()) {
      start = p_node->Input(2).toInt();
    } else {
      // 如果第二个输入是张量，则从中获取 int64_t 类型的值作为起始位置
      auto& t = p_node->Input(2).toTensor();
      start = t.item<int64_t>();
    }
    // 从第三个输入中获取整数值作为长度
    auto length = p_node->Input(3).toInt(); // length

    // 如果第一个输出是空的
    if (p_node->Output(0).isNone()) {
      // 在 CPU 上使用 narrow_copy_dense_cpu 函数进行密集复制，结果保存在输出中
      p_node->Output(0) =
          at::native::narrow_copy_dense_cpu(self, dim, start, length);
      return;
    }
    // 否则，获取第一个输出的张量引用
    auto& output = p_node->Output(0).toTensor();
    // 调整输出张量大小为零
    fastResizeToZero(output);
    // 在输出张量上使用 narrow_copy_dense_cpu_out 函数进行密集复制
    at::native::narrow_copy_dense_cpu_out(self, dim, start, length, output);
  };
});
// 注册操作符的函数对象，处理 torch 中的 aten::index 操作符
REGISTER_OPERATOR_FUNCTOR(aten::index, aten_index, [](Node* n) -> SROperator {
  // 检查节点 n 是否匹配特定的 torch 模式，如果不匹配则记录并返回空指针
  if (!n->matches(torch::schema(
          "aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  // 如果匹配，则返回处理函数
  return [](ProcessedNode* p_node) {
    // 获取输入节点的第一个张量，并转换为相应类型
    const auto& in0_t = p_node->Input(0).toTensor();
    // 将输入节点的第二个列表转换为张量的列表
    const auto in1_l =
        at::native::toListOfOptionalTensors(p_node->Input(1).toListRef());
    // 如果输出节点的第一个输出是空的，则进行索引操作并设置输出
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::cpu::index(in0_t, in1_l);
      return;
    }
    // 否则，获取输出张量并将其尺寸调整为零
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    // 执行索引操作，将结果存储在输出张量中
    at::cpu::index_out(out_t, in0_t, in1_l);
  };
});

// 注册操作符的函数对象，处理 torch 中的 aten::index_select 操作符
REGISTER_OPERATOR_FUNCTOR(
    aten::index_select,
    aten_index_select,
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配特定的 torch 模式，如果不匹配则记录并返回空指针
      if (!n->matches(torch::schema(
              "aten::index_select(Tensor self, int dim, Tensor index) -> Tensor"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      // 如果匹配，则返回处理函数
      return [](ProcessedNode* p_node) {
        // 获取输入节点的各个参数，并转换为相应类型
        const auto& self = p_node->Input(0).toTensor();
        const auto dim = p_node->Input(1).toInt();
        const auto& index = p_node->Input(2).toTensor();
        // 如果输出节点的第一个输出是空的，则进行索引选择操作并设置输出
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = at::native::index_select_cpu_(self, dim, index);
          return;
        }
        // 否则，获取输出张量并将其尺寸调整为零
        auto& out = p_node->Output(0).toTensor();
        fastResizeToZero(out);
        // 执行索引选择操作，将结果存储在输出张量中
        at::native::index_select_out_cpu_(self, dim, index, out);
      };
    });

// 注册操作符的函数对象，处理 torch 中的 aten::pow 操作符
REGISTER_OPERATOR_FUNCTOR(aten::pow, aten_pow, [](Node* n) -> SROperator {
  // 如果节点 n 匹配特定的 torch 模式
  if (n->matches(torch::schema(
          "aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor"))) {
    // 返回处理函数
    return [](ProcessedNode* p_node) {
      // 如果输出节点的第一个输出是空的，则创建一个与输入张量相同类型的空张量
      if (p_node->Output(0).isNone()) {
        const auto& in0_t = p_node->Input(0).toTensor();
        auto dtype =
            at::native::result_type(in0_t, p_node->Input(1).toTensor());
        p_node->Output(0) = create_empty_from(in0_t, dtype);
      }
      // 否则，获取输出张量并将其尺寸调整为零
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);
      // 执行指数运算，将结果存储在输出张量中
      at::cpu::pow_out(
          out_t, p_node->Input(0).toTensor(), p_node->Input(1).toTensor());
    };
  }
  // 如果节点 n 匹配另一种特定的 torch 模式
  if (n->matches(torch::schema(
          "aten::pow.Scalar(Scalar self, Tensor exponent) -> Tensor"))) {
    // 返回处理函数
    return [](ProcessedNode* p_node) {
      // 如果输出节点的第一个输出是空的，则创建一个与输入张量相同类型的空张量
      if (p_node->Output(0).isNone()) {
        const auto& in1_t = p_node->Input(1).toTensor();
        auto dtype =
            at::native::result_type(p_node->Input(0).toScalar(), in1_t);
        p_node->Output(0) = at::native::empty_like(
            in1_t,
            dtype,
            in1_t.options().layout_opt(),
            in1_t.options().device_opt(),
            in1_t.options().pinned_memory_opt(),
            at::MemoryFormat::Preserve);
      }
      // 否则，获取输出张量并将其尺寸调整为零
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);
      // 执行指数运算，将结果存储在输出张量中
      at::cpu::pow_out(
          out_t, p_node->Input(0).toScalar(), p_node->Input(1).toTensor());
      };
});
  };
}
// 检查节点 n 是否匹配指定的 Torch 脚本模式 "aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor"
if (n->matches(torch::schema(
        "aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor"))) {
  // 如果匹配成功，则返回一个 lambda 函数
  return [](ProcessedNode* p_node) {
    // 如果节点输出为空
    if (p_node->Output(0).isNone()) {
      // 获取输入节点的第一个张量和第二个标量
      const auto& in0_t = p_node->Input(0).toTensor();
      auto dtype =
          at::native::result_type(in0_t, p_node->Input(1).toScalar());
      // 根据输入张量的属性创建一个空张量，保持其布局、设备和内存格式
      p_node->Output(0) = at::native::empty_like(
          in0_t,
          dtype,
          in0_t.options().layout_opt(),
          in0_t.options().device_opt(),
          in0_t.options().pinned_memory_opt(),
          at::MemoryFormat::Preserve);
    }
    // 获取输出张量的引用
    auto& out_t = p_node->Output(0).toTensor();
    // 快速调整输出张量的大小为零
    fastResizeToZero(out_t);
    // 调用 ATen 库中的 pow_out 函数计算幂操作，结果存入输出张量中
    at::cpu::pow_out(
        out_t, p_node->Input(0).toTensor(), p_node->Input(1).toScalar());
  };
}
// 记录和转储节点 n 的模式信息
LogAndDumpSchema(n);
// 返回空指针
return nullptr;
// 结束匿名命名空间
});

// 匿名命名空间开始
namespace {

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
// 定义结构体 ToArgs，用于存储转换参数
struct ToArgs {
  std::optional<at::ScalarType> dtype;  // 可选的张量数据类型
  c10::Layout layout;                   // 张量布局
  bool know_to_will_alias = false;      // 标记是否知道会发生别名
  std::optional<c10::MemoryFormat> memory_format;  // 可选的内存格式
};

// 模板函数，根据模板参数提取转换参数 ToArgs
template <bool has_constant_non_tensor_dtype_and_flags, bool has_memory_format>
ToArgs extract_to_args(ProcessedNode* p_node) {
  ToArgs result;  // 创建 ToArgs 结构体对象 result
  if (!has_constant_non_tensor_dtype_and_flags && p_node->Input(1).isTensor()) {
    // 若不具有常量非张量数据类型和标志，并且第二个输入是张量
    const auto& other = p_node->Input(1).toTensor();
    result.dtype = other.scalar_type();  // 获取其他张量的数据类型
    result.layout = other.layout();      // 获取其他张量的布局
    TORCH_DCHECK_EQ(other.device().type(), c10::DeviceType::CPU);  // 断言：确保张量在 CPU 设备上
  } else {
    // 否则，使用第一个输入作为 self 张量
    const auto& self = p_node->Input(0).toTensor();
    result.dtype = p_node->Input(1).toOptional<at::ScalarType>();  // 获取第二个输入的可选数据类型
    result.layout = self.layout();  // 获取 self 张量的布局
    // 静态运行时只能处理 CPU 张量；此处不需要读取此信息。
    TORCH_DCHECK_EQ(self.device().type(), c10::DeviceType::CPU);  // 断言：确保 self 张量在 CPU 设备上
    // 判断是否知道会发生别名，依赖于是否具有常量非张量数据类型和标志，
    // 以及第二个输入的数据类型是否为空或与 self 张量的数据类型相同。
    result.know_to_will_alias = has_constant_non_tensor_dtype_and_flags &&
        (!result.dtype.has_value() ||
         result.dtype.value() == self.dtype().toScalarType());
  }
  if (has_memory_format) {
    // 若具有内存格式
    TORCH_DCHECK_EQ(p_node->num_inputs(), 5);  // 断言：确保节点输入数为 5
    result.memory_format = p_node->Input(4).toOptional<c10::MemoryFormat>();  // 获取第五个输入的可选内存格式
    // 判断是否知道会发生别名，依赖于之前的判断结果，
    // 以及内存格式是否为 Preserve 或者是否为空。
    result.know_to_will_alias = result.know_to_will_alias &&
        (result.memory_format.value_or(c10::MemoryFormat::Preserve) ==
         c10::MemoryFormat::Preserve);
  }

  return result;  // 返回处理后的 ToArgs 结构体对象
}

// 模板结构体，根据模板参数检查是否会发生别名
template <bool has_constant_non_tensor_dtype_and_flags, bool has_memory_format>
struct CheckToWillAlias {
  // 调用函数，接受处理后的节点 p_node、张量 self 和转换参数 to_args 作为参数
  static bool call(
      ProcessedNode* p_node,
      const at::Tensor& self,
      const ToArgs& to_args) {
    // 运算符 to_maybe_copy_out 应该已经检测到常量 true 的 `copy` 参数并使用 to_copy 替代。
    bool copy = false;
    if (has_constant_non_tensor_dtype_and_flags) {
      DCHECK(!p_node->Input(3).toBool());  // 断言：确保第四个输入不为 true
      copy = false;
    } else {
      copy = p_node->Input(3).toBool();  // 获取第四个输入的布尔值
    }
    // 返回是否不复制并且满足以下条件：
    // 知道会发生别名 或者 native::to_will_alias 函数判断不会发生别名
    return !copy &&
        (to_args.know_to_will_alias ||
         at::native::to_will_alias(
             self,
             to_args.dtype,
             to_args.layout,
             c10::Device{c10::DeviceType::CPU},
             copy,
             has_memory_format ? to_args.memory_format
                               : c10::MemoryFormat::Preserve));
  }
};

// 特化模板结构体，当具有常量非张量数据类型和标志，但没有内存格式时使用
template <>
struct CheckToWillAlias<true, false> {
  // 特殊情况！首先，没有内存格式需要检查。
  // 其次，我们知道布局和设备将与 self 张量匹配，因此只需要检查数据类型。
  static bool call(ProcessedNode* p_node, const at::Tensor& self) {
    DCHECK(!p_node->Input(3).toBool());  // 断言：确保第四个输入不为 true
    const auto dtype_opt = p_node->Input(1).toOptional<at::ScalarType>();  // 获取第二个输入的可选数据类型
    // 返回是否为空或者与 self 张量的数据类型相同
    return !dtype_opt.has_value() || *dtype_opt == self.dtype().toScalarType();
  }
};

// 强制内联以便在运行时不必在参数为空时进行分支判断。
// 定义模板函数 `to_copy_functor_impl`，根据模板参数决定函数行为
// 参数 `p_node`：处理过的节点指针，用于访问输入和输出
// 参数 `args`：指向转换参数的指针，用于确定输出张量的属性
template <bool has_constant_non_tensor_dtype_and_flags, bool has_memory_format>
C10_ALWAYS_INLINE void to_copy_functor_impl(
    ProcessedNode* p_node,
    const ToArgs* args) {
  
  // 获取输入节点的第一个张量 `self`
  const auto& self = p_node->Input(0).toTensor();

  // 忽略第三个输入 (copy)

  // 从第二个输入中获取 `non_blocking` 参数，并转换为布尔类型
  auto non_blocking = p_node->Input(2).toBool(); // non_blocking

  // 初始化 `copy_strides` 变量为 false，用于标记是否需要复制步长信息
  bool copy_strides = false;

  // 初始化 `memory_format` 为 Preserve（保留）的可选类型
  std::optional<c10::MemoryFormat> memory_format = c10::MemoryFormat::Preserve;

  // 初始化 `my_args` 为可选类型，用于存储提取的转换参数
  std::optional<ToArgs> my_args;

  // 如果未提供参数 `args`，则从 `p_node` 中提取参数并赋值给 `args`
  if (!args) {
    my_args = extract_to_args<
        has_constant_non_tensor_dtype_and_flags,
        has_memory_format>(p_node);
    args = &my_args.value();
  }

  // 如果需要处理内存格式
  if (has_memory_format) {
    // 获取参数 `args` 中的内存格式，若未指定则使用 Preserve（保留）
    memory_format = args->memory_format.value_or(c10::MemoryFormat::Preserve);
  }

  // 如果内存格式为 Preserve（保留）
  if (memory_format == c10::MemoryFormat::Preserve) {
    // 如果 `self` 是非重叠且密集的张量
    if (self.is_non_overlapping_and_dense()) {
      // 将内存格式设置为 nullptr，标记需要复制步长信息
      memory_format = c10::nullopt;
      copy_strides = true;
    } else {
      // 否则根据建议设置内存格式
      memory_format = self.suggest_memory_format();
    }
  }

  // 初始化 `need_to_allocate_output` 标志，用于确定是否需要重新分配输出张量
  bool need_to_allocate_output = true;

  // 如果输出节点已经是张量
  if (p_node->Output(0).isTensor()) {
    // 获取现有输出张量
    const auto& existing_output = p_node->Output(0).toTensor();

    // 如果没有常量非张量数据类型和标志，并且输出张量的属性与输入参数不匹配
    if ((!has_constant_non_tensor_dtype_and_flags &&
         (existing_output.dtype() != args->dtype ||
          existing_output.layout() != args->layout ||
          existing_output.device() != self.device())) ||
        // 或者如果需要处理内存格式，并且输出张量不是按指定格式连续存储
        (has_memory_format &&
         !existing_output.is_contiguous(
             memory_format.value_or(c10::MemoryFormat::Contiguous)))) {
      // 需要重新分配输出张量
      need_to_allocate_output = true;
    } else {
      // 否则不需要重新分配输出张量
      need_to_allocate_output = false;
    }
  }

  // 如果需要重新分配输出张量
  if (need_to_allocate_output) {
    // 使用 `empty_cpu` 函数创建一个空的 CPU 张量作为输出
    p_node->Output(0) = at::detail::empty_cpu(
        self.sizes(), // 使用输入张量的大小
        args->dtype, // 使用指定的数据类型
        args->layout, // 使用指定的布局
        self.device(), // 使用输入张量的设备
        c10::nullopt, // 不指定内存区域
        memory_format); // 使用指定的内存格式
  } else {
    // 否则如果需要处理内存格式
    if (has_memory_format) {
      // 从第五个输入中获取内存格式，并转换为可选的内存格式类型
      memory_format = p_node->Input(4).toOptional<c10::MemoryFormat>().value_or(
          c10::MemoryFormat::Preserve);
    } else {
      // 否则使用 Preserve（保留）作为内存格式
      memory_format = c10::MemoryFormat::Preserve;
    }
  }

  // 根据条件决定是否需要复制步长信息
  copy_strides = copy_strides ||
      (memory_format == c10::MemoryFormat::Preserve &&
       self.is_non_overlapping_and_dense());

  // 获取输出张量的引用
  auto& out_t = p_node->Output(0).toTensor();

  // 调整输出张量的大小为零
  fastResizeToZero(out_t);

  // 调用 `to_copy_out` 函数执行数据复制操作
  at::native::to_copy_out(
      out_t, // 输出张量
      self, // 输入张量
      non_blocking, // 是否非阻塞
      copy_strides, // 是否复制步长信息
      memory_format); // 使用的内存格式
}
void to_maybe_copy_out_functor(ProcessedNode* p_node) {
  // 针对每次迭代，希望避免每次都检查参数。然而，我们需要考虑在迭代之间自身的 dtype（和布局、内存格式等）可能会发生变化的情况。
  // 从 ProcessedNode 中提取 ToArgs 参数，用于处理是否具有非张量常量 dtype 和内存格式标志的情况
  ToArgs args = extract_to_args<
      has_constant_non_tensor_dtype_and_flags,
      has_memory_format>(p_node);
  // 获取输入的 Tensor 对象 self
  const auto& self = p_node->Input(0).toTensor();
  // 检查是否需要进行别名检查和拷贝操作
  if (CheckToWillAlias<
          has_constant_non_tensor_dtype_and_flags,
          has_memory_format>::call(p_node, self, args)) {
    // 如果不需要写入 Tensor 输出，则将输出置为 None
    p_node->Output(1) = false;
  } else {
    // 否则，将输出置为 true，并调用拷贝操作的实现函数
    p_node->Output(1) = true;
    to_copy_functor_impl<
        has_constant_non_tensor_dtype_and_flags,
        has_memory_format>(p_node, &args);
  }
}

// 在这种情况下，利用 CheckToWillAlias 不需要使用 args 参数
template <>
void to_maybe_copy_out_functor<true, false>(ProcessedNode* p_node) {
  // 获取输入的 Tensor 对象 self
  const auto& self = p_node->Input(0).toTensor();
  // 检查是否需要进行别名检查
  if (CheckToWillAlias<true, false>::call(p_node, self)) {
    // 如果不需要写入 Tensor 输出，则将输出置为 None
    p_node->Output(1) = false;
  } else {
    // 否则，将输出置为 true，并调用拷贝操作的实现函数
    p_node->Output(1) = true;
    // 从 ProcessedNode 中提取 ToArgs 参数，并调用拷贝操作的实现函数
    auto args = extract_to_args<true, false>(p_node);
    to_copy_functor_impl<true, false>(p_node, &args);
  }
}

// 检查节点是否具有非张量常量 dtype 和标志
bool node_has_constant_non_tensor_dtype_and_flags(Node* n) {
  // 获取节点的输入，检查是否具有常量类型和节点种类
  const auto* input1 = n->inputs()[1];
  return input1->type()->kind() != TypeKind::TensorType &&
      input1->node()->kind() == prim::Constant &&
      n->inputs()[2]->node()->kind() == prim::Constant &&
      n->inputs()[3]->node()->kind() == prim::Constant;
}

// 根据参数 has_constant_non_tensor_dtype_and_flags 和 has_memory_format 获取相应的拷贝操作函数
auto get_to_copy_functor(
    bool has_constant_non_tensor_dtype_and_flags,
    bool has_memory_format) {
  if (has_constant_non_tensor_dtype_and_flags) {
    if (has_memory_format) {
      return to_copy_functor<true, true>;
    } else {
      return to_copy_functor<true, false>;
    }
  } else {
    if (has_memory_format) {
      return to_copy_functor<false, true>;
    } else {
      return to_copy_functor<false, false>;
    }
  }
}

} // namespace

// 注册运算符的函数符，将 static_runtime::to_maybe_copy_out 映射到 aten_to_maybe_copy 函数
REGISTER_OPERATOR_FUNCTOR(
    static_runtime::to_maybe_copy_out,
    aten_to_maybe_copy,
    // 以下省略注册运算符的函数符的具体实现部分
    [](Node* n) -> SROperator {
      // 对于 adindexer/adfinder 模型支持4个或5个参数
      // 在这里保留 TORCH_CHECK 是因为没有备用方案可以回退
      if (!sr_schema_check(
              n,
              "static_runtime::to_maybe_copy_out.prim_dtype(Tensor self, int? dtype=None, bool non_blocking=False, bool copy=False) -> (Tensor, bool)",
              "static_runtime::to_maybe_copy_out.dtype(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> (Tensor, bool)",
              "static_runtime::to_maybe_copy_out.other(Tensor self, Tensor other, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> (Tensor, bool)")) {
        // 如果不符合预期的静态运行时模式，则返回空指针
        return nullptr;
      }
      // 检查输入节点的大小是否为4或5
      TORCH_CHECK(n->inputs().size() == 4 || n->inputs().size() == 5);
      const bool has_constant_non_tensor_dtype_and_flags =
          node_has_constant_non_tensor_dtype_and_flags(n);
      const bool has_memory_format = n->inputs().size() == 5;
    
      // 如果具有常量非张量数据类型和标志，则判断是否需要复制
      if (has_constant_non_tensor_dtype_and_flags) {
        const auto copyArg =
            torch::jit::constant_as<bool>(n->inputs()[3]->node()->output());
        DCHECK(copyArg.has_value());
        if (*copyArg) {
          // 如果需要复制，则返回相应的复制函数对象
          return get_to_copy_functor(
              has_constant_non_tensor_dtype_and_flags, has_memory_format);
        }
      }
      // 根据是否具有常量非张量数据类型和是否有内存格式，选择返回对应的函数对象
      if (has_constant_non_tensor_dtype_and_flags) {
        if (has_memory_format) {
          return to_maybe_copy_out_functor<true, true>;
        } else {
          return to_maybe_copy_out_functor<true, false>;
        }
      } else {
        if (has_memory_format) {
          return to_maybe_copy_out_functor<false, true>;
        } else {
          return to_maybe_copy_out_functor<false, false>;
        }
      }
    });
// out variant takes precedence over native
// 注册一个名为 static_runtime::to_copy 的操作符，实现为 static_runtime_to_copy，使用 lambda 表达式定义操作
REGISTER_OPERATOR_FUNCTOR(
    static_runtime::to_copy,
    static_runtime_to_copy,
    [](Node* n) -> SROperator {
      // 如果不符合指定的 schema，则返回 nullptr
      // 支持 4 或 5 个参数的 adindexer/adfinder 模型
      // 在此保留 TORCH_CHECK 是因为没有备用方案可供回退
      if (!sr_schema_check(
              n,
              "static_runtime::to_copy.prim_dtype(Tensor self, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor",
              "static_runtime::to_copy.dtype(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor",
              "static_runtime::to_copy.other(Tensor self, Tensor other, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor")) {
        return nullptr;
      }
      // 检查输入节点的参数数量是否为 4 或 5
      TORCH_CHECK(n->inputs().size() == 4 || n->inputs().size() == 5);
      // 检查节点是否包含常量非张量数据类型和标志
      const bool has_constant_non_tensor_dtype_and_flags =
          node_has_constant_non_tensor_dtype_and_flags(n);
      // 检查是否有内存格式参数
      const bool has_memory_format = n->inputs().size() == 5;
      // 返回根据条件获取的 to_copy 函数对象
      return get_to_copy_functor(
          has_constant_non_tensor_dtype_and_flags, has_memory_format);
    });

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
// 注册一个名为 static_runtime::dequantize_copy 的操作符，实现为 aten_dequantize_copy，使用 lambda 表达式定义操作
REGISTER_OPERATOR_FUNCTOR(
    static_runtime::dequantize_copy,
    aten_dequantize_copy,
    [](Node* n) -> SROperator {
      // 如果输入节点不匹配指定的 schema，则返回 nullptr
      if (!n->matches(torch::schema(
              "static_runtime::dequantize_copy.self(Tensor self) -> Tensor"))) {
        // 请实现对 aten::dequantize 使用张量列表的静态运行支持
        // 记录并转储 schema 信息
        LogAndDumpSchema(n);
        return nullptr;
      }
      // 返回一个 lambda 函数，实现处理节点的操作
      return [](ProcessedNode* p_node) {
        // 获取输入节点的第一个参数 self，作为 Tensor 对象
        const auto& self = p_node->Input(0).toTensor();
        // 如果节点的第一个输出为空，根据 self 的类型和推荐内存格式创建空的 Tensor 对象
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) =
              create_empty_from(self, at::kFloat, self.suggest_memory_format());
        }

        // 获取节点的第一个输出对象，并转换为 Tensor 对象
        auto& out_t = p_node->Output(0).toTensor();
        // 快速调整 Tensor 的大小为零
        fastResizeToZero(out_t);
        // 执行 at::native::dequantize_copy_out 操作，将 self 的数据拷贝到 out_t
        at::native::dequantize_copy_out(out_t, self);
      };
    });

// Out variants for view ops are registered to a separate registry because
// their outputs (views) can't participate in memory reuse.
// 注册一个名为 static_runtime::reshape_copy 的操作符，实现为 aten_reshape
// 专门用于视图操作的输出不参与内存重用，因此注册到独立的注册表中
REGISTER_OPERATOR_FUNCTOR(
    static_runtime::reshape_copy,
    aten_reshape,
    [](Node* n) -> SROperator {
      // Lambda函数：接收一个Node指针参数n，返回一个SROperator对象
      if (!sr_schema_check(
              n,
              "static_runtime::reshape_copy(Tensor self, int[] shape) -> Tensor")) {
        // 检查静态运行时模式，确保节点n满足指定的schema要求
        return nullptr;
      }
      // 确保输入节点n有两个输入
      TORCH_CHECK(n->inputs().size() == 2);
      // 返回一个lambda函数
      return [](ProcessedNode* p_node) {
        // Lambda函数：接收一个ProcessedNode指针p_node
        const auto& self = p_node->Input(0).toTensor(); // 获取输入节点p_node的第一个输入，命名为self，类型为Tensor
        const auto proposed_shape = p_node->Input(1).toDimVector(); // 获取输入节点p_node的第二个输入，作为提议的形状，类型为int数组
    
        if (p_node->Output(0).isNone()) {
          // 如果输出节点p_node的第一个输出为空
          p_node->Output(0) = create_empty_from(self); // 创建一个与self相同类型和形状的空Tensor，并赋给输出节点p_node的第一个输出
        }
        auto& out = p_node->Output(0).toTensor(); // 获取输出节点p_node的第一个输出，并将其作为Tensor引用命名为out
        at::native::reshape_copy_out(out, self, proposed_shape, true); // 调用PyTorch的reshape_copy_out函数，执行Tensor的形状重塑操作
      };
    });
    // Lambda函数的结尾标记
// 注册自定义操作符，将 static_runtime::flatten_copy 映射到 aten_flatten
REGISTER_OPERATOR_FUNCTOR(
    static_runtime::flatten_copy,
    aten_flatten,
    [](Node* n) -> SROperator {
      // 检查节点是否符合特定的运行时模式，如果不符合则返回空指针
      if (!sr_schema_check(
              n,
              "static_runtime::flatten_copy.using_ints(Tensor self, int start_dim=0, int end_dim=-1) -> Tensor")) {
        return nullptr;
      }
      // 检查输入节点的数量是否为3
      TORCH_CHECK(n->inputs().size() == 3);
      // 返回一个 lambda 函数，处理节点的具体逻辑
      return [](ProcessedNode* p_node) {
        // 从处理节点中获取输入的 Tensor self
        const auto& self = p_node->Input(0).toTensor();
        // 获取 start_dim 和 end_dim 参数作为整数
        const auto start_dim = p_node->Input(1).toInt();
        const auto end_dim = p_node->Input(2).toInt();

        // 如果输出节点为空，则创建一个与 self 相同形状的空 Tensor
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = create_empty_from(self);
        }
        // 获取输出节点的引用，准备进行 flatten 操作
        auto& out = p_node->Output(0).toTensor();
        // 调用 at::native::flatten_copy_out 函数执行 flatten 操作
        at::native::flatten_copy_out(out, self, start_dim, end_dim);
      };
    });

// 注册自定义操作符，将 aten::sum 映射到 aten_sum
REGISTER_OPERATOR_FUNCTOR(aten::sum, aten_sum, [](Node* n) -> SROperator {
  // 如果输入节点的数量既不是2也不是4，则返回空指针
  if (n->inputs().size() != 2 && n->inputs().size() != 4) {
    return nullptr;
  }
  // 如果节点匹配特定的 sum 模式，则返回一个 lambda 函数处理节点
  if (n->matches(torch::schema(
          "aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      // 获取输入节点的 Tensor self
      const at::Tensor& self = p_node->Input(0).toTensor();
      // 获取可选的 dtype 参数
      auto dtype = p_node->Input(1).toOptional<at::ScalarType>();
      // 定义空的维度向量和 keepdim 标志
      std::vector<int64_t> dim = {};
      bool keepdim = false;
      // 如果输出节点为空，则使用 at::cpu::sum 函数计算并赋值给输出节点
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::sum(self, dim, keepdim, dtype);
      } else {
        // 否则，获取输出节点的引用并调用 fastResizeToZero 函数
        auto& output = p_node->Output(0).toTensor();
        fastResizeToZero(output);
        // 使用 at::cpu::sum_out 函数计算并赋值给输出节点
        at::cpu::sum_out(output, self, dim, keepdim, dtype);
      }
    };
  }
  // 如果节点匹配另一种 sum 模式，则返回一个 lambda 函数处理节点
  if (n->matches(torch::schema(
          "aten::sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      // 获取输入节点的 Tensor self 和 dim 参数
      const at::Tensor& self = p_node->Input(0).toTensor();
      auto dim = p_node->Input(1).toDimVector();
      auto keepdim = p_node->Input(2).toBool();
      auto dtype = p_node->Input(3).toOptional<at::ScalarType>();
      // 如果输出节点为空，则使用 at::cpu::sum 函数计算并赋值给输出节点
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::sum(self, dim, keepdim, dtype);
      } else {
        // 否则，获取输出节点的引用并调用 fastResizeToZero 函数
        auto& output = p_node->Output(0).toTensor();
        fastResizeToZero(output);
        // 使用 at::cpu::sum_out 函数计算并赋值给输出节点
        at::cpu::sum_out(output, self, dim, keepdim, dtype);
      }
    };
  }
  // 如果以上两种模式都不匹配，则记录并转储节点的模式信息
  LogAndDumpSchema(n);
  return nullptr;
});

// 注册自定义操作符，将 aten::mean 映射到 aten_mean
REGISTER_OPERATOR_FUNCTOR(aten::mean, aten_mean, [](Node* n) -> SROperator {
  // 如果节点匹配特定的 mean 模式，则返回一个 lambda 函数处理节点
  if (n->matches(torch::schema(
          "aten::mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个输入作为 self 引用
      const auto& self = p_node->Input(0).toTensor();
      // 获取输入节点的第二个输入作为维度信息
      const auto dim = p_node->Input(1).toDimVector();
      // 获取输入节点的第三个输入作为 keepdim 标志
      const bool keepdim = p_node->Input(2).toBool();
      // 获取输入节点的第四个输入作为数据类型（可选），如果未提供则使用 self 的数据类型
      const auto dtype = p_node->Input(3).toOptional<at::ScalarType>();
      
      // 如果输出节点的第一个输出为空，则创建一个空的 Tensor 作为输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = create_empty_from(
            self, dtype.value_or(self.dtype().toScalarType()));
      }
      
      // 获取输出节点的第一个输出，并将其转换为 Tensor 引用
      auto& output = p_node->Output(0).toTensor();
      // 快速调整输出 Tensor 的大小为零
      fastResizeToZero(output);
      // 调用 ATen 库的函数计算输入 Tensor 的均值，并输出到 output 中
      at::cpu::mean_out(output, self, dim, keepdim, dtype);
    };
  }

  // 如果节点匹配 "aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor" 的模式
  if (n->matches(torch::schema(
          "aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个输入作为 self 引用
      const auto& self = p_node->Input(0).toTensor();
      // 获取输入节点的第二个输入作为数据类型（可选）
      const auto dtype = p_node->Input(1).toOptional<at::ScalarType>();
      
      // 如果输出节点的第一个输出为空，则创建一个空的 Tensor 作为输出
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = create_empty_from(
            self, dtype.value_or(self.dtype().toScalarType()));
      }
      
      // 获取输出节点的第一个输出，并将其转换为 Tensor 引用
      auto& output = p_node->Output(0).toTensor();
      // 快速调整输出 Tensor 的大小为零
      fastResizeToZero(output);
      // 调用 ATen 库的函数计算输入 Tensor 的均值，并输出到 output 中，不保持维度
      at::cpu::mean_out(output, self, /*dim=*/{}, /*keepdim=*/false, dtype);
    };
  }

  // 如果节点不匹配任何已知模式，则记录并转储其模式信息
  LogAndDumpSchema(n);
  // 返回空指针
  return nullptr;
REGISTER_OPERATOR_FUNCTOR(aten::repeat, aten_repeat, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 torch 模式，如果不匹配则记录并返回空指针
  if (!n->matches(torch::schema(
          "aten::repeat(Tensor self, int[] repeats) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  // 匹配成功时的处理逻辑
  return [](ProcessedNode* p_node) {
    // 获取输入节点的张量和重复次数
    const auto& self = p_node->Input(0).toTensor();
    const auto repeats = p_node->Input(1).toDimVector();

    // 如果输出节点未初始化，则直接赋值为 repeat 操作的结果
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::native::repeat(self, repeats);
      return;
    }
    // 否则，重用输出节点并调用 repeat_out 函数
    at::Tensor& output = p_node->Output(0).toTensor();
    at::native::repeat_out(output, self, repeats);
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::max, aten_max, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 torch 模式，返回 max 操作的 functor
  if (n->matches(torch::schema(
          "aten::max.other(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      // 获取输入节点的两个张量 self 和 other
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      // 如果输出节点未初始化，则直接赋值为 max 操作的结果
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::max(self, other);
        return;
      }
      // 否则，重用输出节点并调用 max_out 函数
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::max_out(self, other, out);
    };
  }

  // 检查节点是否匹配指定的 torch 模式，返回 max.dim 操作的 functor
  if (n->matches(torch::schema(
          "aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)"))) {
    return [](ProcessedNode* p_node) {
      // 获取输入节点的张量 self，维度 dim 和 keepdim 参数
      const auto& self = p_node->Input(0).toTensor();
      auto dim = p_node->Input(1).toInt();
      const auto keepdim = p_node->Input(2).toBool();

      // 如果第一个输出节点未初始化，则创建一个与 self 相同类型的空张量
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = create_empty_from(self);
      }

      // 如果第二个输出节点未初始化，则创建一个与 self 类型为 kLong 的空张量
      if (p_node->Output(1).isNone()) {
        p_node->Output(1) = create_empty_from(self, at::kLong);
      }

      // 获取输出节点的引用，并调用 max_out 函数进行计算
      auto& values = p_node->Output(0).toTensor();
      auto& indices = p_node->Output(1).toTensor();
      fastResizeToZero(values);
      fastResizeToZero(indices);
      at::cpu::max_out(values, indices, self, dim, keepdim);
    };
  }

  // 检查节点是否匹配指定的 torch 模式，返回 max 操作的 functor
  if (n->matches(torch::schema("aten::max(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      // 获取输入节点的张量 self
      const auto& self = p_node->Input(0).toTensor();
      // 如果输出节点未初始化，则创建一个与 self 相同类型的空张量
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = create_empty_from(self);
      }
      // 获取输出节点的引用，并调用 amax_out 函数进行计算
      auto& value = p_node->Output(0).toTensor();
      fastResizeToZero(value);
      at::cpu::amax_out(value, self);
    };
  }

  // 若以上所有模式都不匹配，则记录并返回空指针
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::sign, aten_sign, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 torch 模式，如果不匹配则记录并返回空指针
  if (!n->matches(torch::schema("aten::sign.Tensor(Tensor input) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  // 匹配成功时的处理逻辑
  return [](ProcessedNode* p_node) {
    // 获取输入节点的张量 input
    const auto& in0_t = p_node->Input(0).toTensor();
    // 如果输出节点未初始化，则直接赋值为 sign 操作的结果
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::cpu::sign(in0_t);
      return;
    }
    // 否则，重用输出节点并调用 sign_out 函数
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::cpu::sign_out(out_t, in0_t);
  };
});
// 注册 ATen 操作符 "aten::div" 的处理函数 aten_div
REGISTER_OPERATOR_FUNCTOR(aten::div, aten_div, [](Node* n) -> SROperator {
  // 检查节点 n 是否匹配 div 操作的任一 schema，若不匹配则记录日志并返回空指针
  if (!n->matches(torch::schema(
          "aten::div.Tensor(Tensor self, Tensor other) -> Tensor")) &&
      !n->matches(torch::schema(
          "aten::div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor")) &&
      !n->matches(torch::schema(
          "aten::div.Scalar(Tensor self, Scalar other) -> Tensor")) &&
      !n->matches(torch::schema(
          "aten::div.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor"))) {
    LogAndDumpSchema(n);  // 记录并输出当前节点的 schema 信息
    return nullptr;  // 返回空指针，表示未能处理该操作
  }

  // 返回一个 lambda 函数处理器，该函数接收 ProcessedNode 对象作为参数
  return [te = createDiv()](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();  // 获取输入的第一个 Tensor
    std::optional<c10::string_view> rounding_mode = c10::nullopt;  // 初始化舍入模式为无
    if (p_node->num_inputs() > 2) {
      rounding_mode = p_node->Input(2).toOptional<c10::string_view>();  // 获取第三个输入作为舍入模式
    }
    const auto& in1_t = p_node->Input(1).isTensor()
        ? p_node->Input(1).toTensor()  // 如果第二个输入是 Tensor，则直接获取
        : at::native::wrapped_scalar_tensor(p_node->Input(1).toScalar());  // 否则封装为标量 Tensor

    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);  // 如果输出为 None，则根据 in0_t 创建空输出
    }
    auto& out_t = p_node->Output(0).toTensor();  // 获取输出的 Tensor 引用

    // 检查输入 Tensor 的尺寸、数据类型、步幅等条件是否满足特定要求
    if (in0_t.sizes() == in1_t.sizes() &&
        in0_t.scalar_type() == in1_t.scalar_type() &&
        in0_t.strides() == in1_t.strides() && in0_t.is_contiguous() &&
        in0_t.scalar_type() == at::kFloat) {
      int64_t dim = in0_t.numel();  // 获取 Tensor 的元素数量
      int i_rounding_mode = 0;
      if (rounding_mode && !rounding_mode.value().empty()) {
        const char peek_rounding_mode = rounding_mode.value().at(0);
        if (peek_rounding_mode == 't') {
          // 如果舍入模式为 't'，则设定为截断
          i_rounding_mode = 1;
        } else if (peek_rounding_mode == 'f') {
          // 如果舍入模式为 'f'，则设定为向下取整
          i_rounding_mode = 2;
        }
      }
      at::native::resize_(out_t, in0_t.sizes());  // 调整输出 Tensor 的尺寸
      te->call(
          {out_t.data_ptr(),
           in0_t.data_ptr(),
           in1_t.data_ptr(),
           &i_rounding_mode,
           &dim});  // 调用 createDiv() 函数处理 Tensor 的除法操作
    } else {
      fastResizeToZero(out_t);  // 快速将输出 Tensor 调整为空
      at::cpu::div_out(out_t, in0_t, in1_t, rounding_mode);  // 执行 CPU 上的除法操作
    }
  };
});

// 注册 ATen 操作符 "aten::log" 的处理函数 aten_log
REGISTER_OPERATOR_FUNCTOR(aten::log, aten_log, [](Node* n) -> SROperator {
  // 检查节点 n 是否匹配 log 操作的特定 schema，若不匹配则记录日志并返回空指针
  if (!n->matches(torch::schema("aten::log.Tensor(Tensor input) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  // 返回一个 lambda 函数处理器，该函数接收 ProcessedNode 对象作为参数
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();  // 获取输入的 Tensor
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::cpu::log(in0_t);  // 如果输出为 None，则直接计算 log 并存入输出
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();  // 获取输出的 Tensor 引用
    fastResizeToZero(out_t);  // 快速将输出 Tensor 调整为空
    at::cpu::log_out(out_t, in0_t);  // 执行 log 操作并将结果存入输出 Tensor
  };
});

// 注册 ATen 操作符 "aten::sub" 的处理函数 aten_sub
REGISTER_OPERATOR_FUNCTOR(aten::sub, aten_sub, [](Node* n) -> SROperator {
  // 检查节点 n 是否匹配 sub 操作的特定 schema，若匹配则执行相应的操作
  if (n->matches(torch::schema(
          "aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"))) {
    // 如果节点匹配 "aten::sub(Tensor self, Tensor other, Scalar alpha=1) -> Tensor" 的模式
    if (n->matches(torch::schema(
            "aten::sub(Tensor self, Tensor other, Scalar alpha=1) -> Tensor"))) {
        // 返回一个 lambda 函数，该函数接受一个 ProcessedNode* 参数
        return [](ProcessedNode* p_node) {
          // 获取输入参数的第一个和第二个张量，并转换为引用
          const auto& in0_t = p_node->Input(0).toTensor();
          const auto& in1_t = p_node->Input(1).toTensor();
          // 获取第三个参数 alpha，并转换为标量
          const auto alpha = p_node->Input(2).toScalar();
          // 如果输出张量为空
          if (p_node->Output(0).isNone()) {
            // 在输出张量上执行 CPU 计算的减法操作
            p_node->Output(0) = at::cpu::sub(in0_t, in1_t, alpha);
            return;
          }
          // 否则，获取输出张量的引用
          auto& out_t = p_node->Output(0).toTensor();
          // 快速将输出张量大小调整为零
          fastResizeToZero(out_t);
          // 在输出张量上执行原位减法操作
          at::cpu::sub_out(out_t, in0_t, in1_t, alpha);
        };
      }
    
    // 如果节点匹配 "aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor" 的模式
    if (n->matches(torch::schema(
            "aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor"))) {
        // 返回一个 lambda 函数，该函数接受一个 ProcessedNode* 参数
        return [](ProcessedNode* p_node) {
          // 获取输入参数的第一个张量，并转换为引用
          const auto& in0_t = p_node->Input(0).toTensor();
          // 获取第二个参数作为标量，并封装为张量
          const auto& in1_t =
              at::native::wrapped_scalar_tensor(p_node->Input(1).toScalar());
          // 获取第三个参数 alpha，并转换为标量
          const auto alpha = p_node->Input(2).toScalar();
          // 如果输出张量为空
          if (p_node->Output(0).isNone()) {
            // 在输出张量上执行 CPU 计算的减法操作
            p_node->Output(0) = at::cpu::sub(in0_t, in1_t, alpha);
            return;
          }
          // 否则，获取输出张量的引用
          auto& out_t = p_node->Output(0).toTensor();
          // 快速将输出张量大小调整为零
          fastResizeToZero(out_t);
          // 在输出张量上执行原位减法操作
          at::cpu::sub_out(out_t, in0_t, in1_t, alpha);
        };
      }
    
    // 如果以上两个条件都不匹配，则记录并转储节点的模式信息
    LogAndDumpSchema(n);
    // 返回空指针
    return nullptr;
});

// TODO: support clamp_min.Tensor(Tensor self, Tensor min) -> Tensor
REGISTER_OPERATOR_FUNCTOR(
    aten::clamp_min,
    aten_clamp_min,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配给定的 schema，如果不匹配则记录并返回空指针
      if (!n->matches(torch::schema(
              "aten::clamp_min(Tensor self, Scalar min) -> Tensor"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      // 如果节点匹配，则返回 lambda 表达式处理节点的操作
      return [](ProcessedNode* p_node) {
        // 获取输入节点的第一个张量和第二个标量
        const auto& in0_t = p_node->Input(0).toTensor();
        const auto in1_s = p_node->Input(1).toScalar();
        // 如果输出节点为 None，则直接对第一个张量进行 clamp_min 操作并赋值给输出节点
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = at::cpu::clamp_min(in0_t, in1_s);
          return;
        }
        // 否则，获取输出张量的引用，并快速调整大小为零
        auto& out_t = p_node->Output(0).toTensor();
        fastResizeToZero(out_t);
        // 调用 clamp_min_out 函数将 clamp_min 操作的结果存入输出张量
        at::cpu::clamp_min_out(out_t, in0_t, in1_s);
      };
    });

// 注册 argmin 操作符的函数处理器
REGISTER_OPERATOR_FUNCTOR(aten::argmin, aten_argmin, [](Node* n) -> SROperator {
  // 检查节点是否匹配给定的 schema，如果不匹配则记录并返回空指针
  if (!n->matches(torch::schema(
          "aten::argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  // 如果节点匹配，则返回 lambda 表达式处理节点的操作
  return [](ProcessedNode* p_node) {
    // 获取输入节点的第一个张量，第二个参数（维度），以及第三个参数（是否保持维度）
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto dim = p_node->Input(1).toOptional<int64_t>();
    const auto keepdim = p_node->Input(2).toBool();
    // 如果输出节点为 None，则直接对第一个张量进行 argmin 操作并赋值给输出节点
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::cpu::argmin(in0_t, dim, keepdim);
      return;
    }
    // 否则，获取输出张量的引用，并快速调整大小为零
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    // 如果输入张量是连续的并且有指定维度，则调用 c2_argmin_out 函数进行 argmin 操作
    if (in0_t.is_contiguous() && dim.has_value()) {
      at::native::c2_argmin_out(out_t, in0_t, dim.value(), keepdim);
      return;
    }
    // 否则，调用 argmin_out 函数进行 argmin 操作
    at::cpu::argmin_out(out_t, in0_t, dim, keepdim);
  };
});

// 注册 softmax 操作符的函数处理器
REGISTER_OPERATOR_FUNCTOR(aten::softmax, aten_softmax, [](Node* n) -> SROperator {
  // 检查节点是否匹配给定的 schema，如果不匹配则记录并返回空指针
  if (!n->matches(torch::schema(
          "aten::softmax(Tensor self, int dim, ScalarType? dtype=None) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  // 如果节点匹配，则返回 lambda 表达式处理节点的操作
  return [](ProcessedNode* p_node) {
    // 获取输入节点的第一个张量，第二个参数（维度），以及第三个参数（数据类型）
    const auto& in_t = p_node->Input(0).toTensor();
    const auto& dim = p_node->Input(1).toInt();
    const auto& dtype = p_node->Input(2).toOptional<c10::ScalarType>();
    // 如果输出节点为 None，则直接对输入张量进行 softmax 操作并赋值给输出节点
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::native::softmax(in_t, dim, dtype);
      return;
    }
    // 否则，获取输出张量的引用，并快速调整大小为零
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    // 如果输入张量是半精度浮点数且目标数据类型是单精度浮点数，则将 half_to_float 设置为 true
    auto half_to_float = in_t.scalar_type() == at::ScalarType::Half &&
        dtype == at::ScalarType::Float;
    // 调用 _softmax_out 函数将 softmax 操作的结果存入输出张量
    at::cpu::_softmax_out(out_t, in_t, dim, half_to_float);
  };
});

namespace {

// 从可选的张量类型的 IValue 借用张量
c10::MaybeOwned<at::Tensor> borrow_from_optional_tensor_ivalue(
    const IValue& iv) {
  // 如果 IValue 是 None 类型，则返回一个空的张量
  if (iv.isNone()) {
    return c10::MaybeOwned<at::Tensor>::owned(std::in_place);
  }
  // 否则，从 IValue 中借用张量
  return c10::MaybeOwned<at::Tensor>::borrowed(iv.toTensor());
}

} // namespace
REGISTER_OPERATOR_FUNCTOR(aten::layer_norm, aten_layer_norm, [](Node* n) -> SROperator {
  // 检查节点是否符合指定的schema，如果不符合则返回nullptr
  if (!sr_schema_check(
          n,
          "aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor")) {
    return nullptr;
  }
  // 返回一个lambda表达式，处理节点数据
  return [](ProcessedNode* p_node) {
    // 忽略输入(5): `bool cudnn_enable=True`
    const auto& input = p_node->Input(0).toTensor();
    const auto normalized_shape = p_node->Input(1).toDimVector();
    float eps = p_node->Input(4).toDouble();

    // 从可选的Tensor输入中获取权重和偏置，并转换为常量引用
    c10::MaybeOwned<at::Tensor> weight_maybe_owned =
        borrow_from_optional_tensor_ivalue(p_node->Input(2));
    const at::Tensor& weight = *weight_maybe_owned;
    c10::MaybeOwned<at::Tensor> bias_maybe_owned =
        borrow_from_optional_tensor_ivalue(p_node->Input(3));
    const at::Tensor& bias = *bias_maybe_owned;

    // 检查和获取输入的M和N值
    auto M_N = at::native::_check_layer_norm_inputs(
        input, normalized_shape, weight, bias);
    auto M = M_N.first;
    auto N = M_N.second;
    auto X = input.expect_contiguous();
    auto gamma = weight.expect_contiguous();
    auto beta = bias.expect_contiguous();

    // 如果输出(0)为空，根据输入X的大小创建一个空的Tensor
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::native::empty_like(
          *X,
          c10::nullopt /* dtype */,
          c10::nullopt /* layout */,
          c10::nullopt /* device */,
          c10::nullopt /* pin_memory */,
          at::MemoryFormat::Contiguous);
    } else {
      // 否则，调整输出(0)的大小与输入X相同
      at::native::resize_(
          p_node->Output(0).toTensor(), X->sizes(), c10::nullopt);
    }
    // 获取输出Tensor的引用
    at::Tensor& output = p_node->Output(0).toTensor();
    // 调用CPU版本的layer_norm计算函数，输出结果存入output中
    at::native::layer_norm_cpu_out(output, *X, *gamma, *beta, eps, M, N);
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::norm, aten_norm, [](Node* n) -> SROperator {
  // 如果节点匹配指定的schema
  if (n->matches(torch::schema(
          "aten::norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor"))) {
    // 返回一个lambda表达式，处理节点数据
    return [](ProcessedNode* p_node) {
      // 获取输入Tensor
      const auto& in0_t = p_node->Input(0).toTensor();
      // 如果输出(0)为空，根据in0_t创建一个空的Tensor
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = create_empty_from(in0_t);
      }
      // 获取输出Tensor的引用
      auto& out_t = p_node->Output(0).toTensor();
      // 快速将输出Tensor大小调整为0
      fastResizeToZero(out_t);
      // 获取输入(1)的可选Scalar值
      const auto in1_s = p_node->Input(1).toOptional<at::Scalar>();
      // 调用CPU版本的norm_outf函数计算结果，存入out_t中
      at::cpu::norm_outf(
          in0_t,
          in1_s,
          c10::IntArrayRef{},  // 空的维度数组
          false,               // 不保持维度
          p_node->Input(2).toScalarType(),  // 获取输入(2)的标量类型
          out_t);
    };
  }
  // 如果节点匹配另一个schema
  if (n->matches(torch::schema(
          "aten::norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor"))) {
    // 返回一个 lambda 函数，该函数处理 ProcessedNode 指针参数 p_node
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个输入张量
      const auto& in0_t = p_node->Input(0).toTensor();

      // 如果输出节点的第一个输出为空，则创建一个与 in0_t 相同类型的空张量
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = create_empty_from(in0_t);
      }
      // 获取输出节点的第一个输出张量的引用
      auto& out_t = p_node->Output(0).toTensor();

      // 将 out_t 快速调整大小为零（即清空张量数据）
      fastResizeToZero(out_t);

      // 获取输入节点的第二个输入（可选的标量）转换为 Scalar 类型
      const auto in1_s = p_node->Input(1).toOptional<at::Scalar>();

      // 调用 at::cpu::norm_outf 函数计算张量 in0_t 的范数
      at::cpu::norm_outf(
          in0_t,
          in1_s,
          p_node->Input(2).toDimVector(), // dim 参数，转换为维度向量
          p_node->Input(3).toBool(),      // keepdim 参数，转换为布尔值
          out_t);                         // 输出结果保存在 out_t 中
    };
  }
  // 如果节点 n 符合指定的 Torch 模式
  if (n->matches(torch::schema(
          "aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor"))) {
    // 返回一个 lambda 函数，处理 ProcessedNode 指针参数 p_node
    return [](ProcessedNode* p_node) {
      // 获取输入节点的第一个输入张量
      const auto& in0_t = p_node->Input(0).toTensor();

      // 如果输出节点的第一个输出为空，则创建一个与 in0_t 相同类型的空张量
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = create_empty_from(in0_t);
      }
      // 获取输出节点的第一个输出张量的引用
      auto& out_t = p_node->Output(0).toTensor();

      // 将 out_t 快速调整大小为零（即清空张量数据）
      fastResizeToZero(out_t);

      // 获取输入节点的第二个输入（可选的标量）转换为 Scalar 类型
      const auto in1_s = p_node->Input(1).toOptional<at::Scalar>();

      // 调用 at::cpu::norm_outf 函数计算张量 in0_t 的范数
      at::cpu::norm_outf(
          in0_t,
          in1_s,
          p_node->Input(2).toDimVector(), // dim 参数，转换为维度向量
          p_node->Input(3).toBool(),      // keepdim 参数，转换为布尔值
          out_t);                         // 输出结果保存在 out_t 中
    };
  }
  // 如果节点 n 不符合上述模式，则记录并转储其模式信息
  LogAndDumpSchema(n);
  // 返回空指针
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::matmul, aten_matmul, [](Node* n) -> SROperator {
  // 检查节点是否匹配 matmul 的 Torch 模式，如果不匹配则记录日志并返回空指针
  if (!n->matches(
          torch::schema("aten::matmul(Tensor self, Tensor other) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  // 返回一个 lambda 表达式，执行 matmul 操作
  return [](ProcessedNode* p_node) {
    // 获取输入节点的第一个和第二个张量
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto& in1_t = p_node->Input(1).toTensor();

    // 如果输出节点为空，则直接调用 at::native::matmul 计算结果
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::native::matmul(in0_t, in1_t);
      return;
    }
    // 否则，获取输出张量并调用 fastResizeToZero 清空
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    // 调用 at::native::matmul_out 将结果写入输出张量
    at::native::matmul_out(in0_t, in1_t, out_t);
  };
});

REGISTER_OPERATOR_FUNCTOR(quantized::linear, quantized_linear, [](Node* n) -> SROperator {
  // 检查节点是否匹配 quantized::linear 的 Torch 模式，如果不匹配则记录日志并返回空指针
  if (!n->matches(torch::schema(
          "quantized::linear(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  // 尝试获取输入节点中的第二个张量作为 packed_weight
  const auto w = toIValue(n->inputs()[1]);
  c10::intrusive_ptr<LinearPackedParamsBase> packed_weight;
  if (w) {
    packed_weight = w->toCustomClass<LinearPackedParamsBase>();
  }
  // 返回一个 lambda 表达式，执行 quantized linear 操作
  return [packed_weight](ProcessedNode* p_node) {
    // 获取输入节点的第一个张量作为 input
    const auto& input = p_node->Input(0).toTensor();
    // 获取输入节点的第三个和第四个输入作为 scale 和 zero_point
    const auto output_scale = p_node->Input(2).toDouble();
    const auto output_zero_point = p_node->Input(3).toInt();

    // 如果输出节点为空，则创建一个空的量化张量
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::native::empty_affine_quantized(
          {0},
          c10::kQUInt8,
          c10::nullopt,
          c10::kCPU,
          false,
          output_scale,
          output_zero_point,
          c10::nullopt);
    }
    // 否则，获取输出张量并调用 fastResizeToZero 清空
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);

    // 如果 packed_weight 存在，则调用 packed_weight->apply_out 方法
    if (packed_weight) {
      packed_weight->apply_out(input, output_scale, output_zero_point, out_t);
    } else {
      // 否则，从输入节点中获取第二个张量作为临时 packed_weight_tmp，并调用其 apply_out 方法
      auto packed_weight_tmp =
          p_node->Input(1).toCustomClass<LinearPackedParamsBase>();
      packed_weight_tmp->apply_out(
          input, output_scale, output_zero_point, out_t);
    }
  };
});

REGISTER_OPERATOR_FUNCTOR(
    fb::quantized_linear,
    fb_quantized_linear,
    // 匿名函数定义，接受一个指向 Node 类型的指针参数 n，返回一个 SROperator 类型的对象
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配特定的 Torch 脚本模式，如果不匹配则记录日志并返回空指针
      if (!n->matches(torch::schema(
              "fb::quantized_linear(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase w_prepack, Tensor Y_scale_i, Tensor Y_zero_point_i) -> Tensor"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      // 获取节点输入的第二个参数，并转换为 IValue 对象
      const auto w = toIValue(n->inputs()[1]);
      // 声明一个指向 LinearPackedParamsBase 类型对象的智能指针
      c10::intrusive_ptr<LinearPackedParamsBase> packed_weight;
      // 如果 w 不为空，则将其转换为 LinearPackedParamsBase 类型
      if (w) {
        packed_weight = w->toCustomClass<LinearPackedParamsBase>();
      }
      // 返回一个 lambda 表达式，捕获 packed_weight 变量，处理节点 p_node
      return [packed_weight](ProcessedNode* p_node) {
        // 获取 p_node 的输入张量 input
        const auto& input = p_node->Input(0).toTensor();
        // 获取 p_node 的输入张量的缩放因子 output_scale，并转换为 float 类型
        const auto output_scale = p_node->Input(2).toTensor().item().toFloat();
        // 获取 p_node 的输入张量的零点 output_zero_point，并转换为 long 类型
        const auto output_zero_point =
            p_node->Input(3).toTensor().item().toLong();

        // 如果 p_node 的输出张量为空
        if (p_node->Output(0).isNone()) {
          // 创建一个空的仿射量化张量，类型为 kQUInt8
          p_node->Output(0) = at::native::empty_affine_quantized(
              {0},
              c10::kQUInt8,
              c10::nullopt,
              c10::kCPU,
              false,
              output_scale,
              output_zero_point,
              c10::nullopt);
        }
        // 获取 p_node 的输出张量 out_t 的引用
        auto& out_t = p_node->Output(0).toTensor();
        // 快速调整 out_t 的大小为零

        // 如果 packed_weight 不为空，则调用其 apply_out 方法
        if (packed_weight) {
          packed_weight->apply_out(
              input, output_scale, output_zero_point, out_t);
        } else {
          // 否则，权重可能在运行时进行量化，从 p_node 的输入获取临时 packed_weight_tmp
          auto packed_weight_tmp =
              p_node->Input(1).toCustomClass<LinearPackedParamsBase>();
          // 调用 packed_weight_tmp 的 apply_out 方法
          packed_weight_tmp->apply_out(
              input, output_scale, output_zero_point, out_t);
        }
      };
    });
namespace {

// 定义模板函数，根据是否有ReLU激活函数应用不同的动态输出处理
template <bool has_relu>
void apply_dynamic_out_functor(
    c10::intrusive_ptr<LinearPackedParamsBase> packed_weight,
    const at::Tensor& input,
    at::Tensor& out,
    bool reduce_range);

// 当没有ReLU时的特化实现
template <>
void apply_dynamic_out_functor<false>(
    c10::intrusive_ptr<LinearPackedParamsBase> packed_weight,
    const at::Tensor& input,
    at::Tensor& out,
    bool reduce_range) {
  // 调用线性层参数的动态输出函数
  packed_weight->apply_dynamic_out(input, out, reduce_range);
}

// 当有ReLU时的特化实现
template <>
void apply_dynamic_out_functor<true>(
    c10::intrusive_ptr<LinearPackedParamsBase> packed_weight,
    const at::Tensor& input,
    at::Tensor& out,
    bool reduce_range) {
  // 由于PackedLinearWeightFp16::apply_dynamic_impl不处理ReLU，这里显式执行ReLU
  packed_weight->apply_dynamic_out(input, out, reduce_range);
  // 对输出张量应用ReLU激活函数
  out.relu_();
}

// 定义量化动态FP16线性操作的实现函数模板
template <bool has_relu>
SROperator quantized_linear_dynamic_fp16_impl(Node* n) {
  // 获取权重参数
  const auto weight = toIValue(n->inputs()[1]);
  c10::intrusive_ptr<LinearPackedParamsBase> packed_weight;
  if (weight) {
    // 将权重转换为自定义类LinearPackedParamsBase的指针
    packed_weight = weight->toCustomClass<LinearPackedParamsBase>();
  }
  if (packed_weight) {
    // 如果存在打包的权重对象，返回一个lambda函数，用于处理节点
    return [packed_weight](ProcessedNode* p_node) {
      // 获取输入张量
      const auto& input = p_node->Input(0).toTensor();
      // 如果输出张量为空，则创建一个与输入类型相同的空张量
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = create_empty_from(input, at::kFloat);
      }
      auto& out_t = p_node->Output(0).toTensor();
      // 快速调整输出张量大小为0
      fastResizeToZero(out_t);
      // 应用动态输出处理函数，根据是否有ReLU
      apply_dynamic_out_functor<has_relu>(packed_weight, input, out_t, false);
    };
  } else {
    // 如果不存在打包的权重对象，返回一个lambda函数，用于处理节点
    return [](ProcessedNode* p_node) {
      // 获取输入张量
      const auto& input = p_node->Input(0).toTensor();
      // 如果输出张量为空，则创建一个与输入类型相同的空张量
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = create_empty_from(input, at::kFloat);
      }
      auto& out_t = p_node->Output(0).toTensor();
      // 快速调整输出张量大小为0
      fastResizeToZero(out_t);
      // 权重可能在运行时进行量化，应用动态输出处理函数，根据是否有ReLU
      auto packed_weight_tmp =
          p_node->Input(1).toCustomClass<LinearPackedParamsBase>();
      apply_dynamic_out_functor<has_relu>(
          packed_weight_tmp, input, out_t, false);
    };
  }
}

} // namespace

// 注册量化动态FP16线性操作符，返回操作符函数
REGISTER_OPERATOR_FUNCTOR(
    quantized::linear_dynamic_fp16,
    quantized_linear_dynamic_fp16,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配指定的模式
      if (!n->matches(torch::schema(
              "quantized::linear_dynamic_fp16(Tensor X, __torch__.torch.classes."
              "quantized.LinearPackedParamsBase W_prepack) -> Tensor Y"))) {
        // 记录并输出不匹配的模式信息
        LogAndDumpSchema(n);
        return nullptr;
      }
      // 返回不带ReLU的量化动态FP16线性操作函数
      return quantized_linear_dynamic_fp16_impl<false>(n);
    });

// 注册带ReLU的量化动态FP16线性操作符，返回操作符函数
REGISTER_OPERATOR_FUNCTOR(
    quantized::linear_relu_dynamic_fp16,
    quantized_linear_relu_dynamic_fp16,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配指定的模式
      if (!n->matches(torch::schema(
              "quantized::linear_relu_dynamic_fp16(Tensor X, __torch__.torch.classes."
              "quantized.LinearPackedParamsBase W_prepack) -> Tensor Y"))) {
        // 记录并输出不匹配的模式信息
        LogAndDumpSchema(n);
        return nullptr;
      }
      // 返回带ReLU的量化动态FP16线性操作函数
      return quantized_linear_dynamic_fp16_impl<true>(n);
    });
    [](Node* n) -> SROperator {
      // 匿名函数，接受一个指向 Node 类型的指针 n，返回一个 SROperator 类型的对象
      if (!n->matches(torch::schema(
              "quantized::linear_relu_dynamic_fp16(Tensor X, __torch__.torch.classes."
              "quantized.LinearPackedParamsBase W_prepack) -> Tensor Y"))) {
        // 如果 n 不匹配指定的 Torch schema，执行以下操作
        LogAndDumpSchema(n);  // 记录和转储不匹配的 schema 信息
        return nullptr;  // 返回空指针
      }
      // 如果匹配成功，则调用 quantized_linear_dynamic_fp16_impl<true> 处理 n，并返回结果
      return quantized_linear_dynamic_fp16_impl<true>(n);
    });
// 检查给定的 IValue 是否是一个 Tensor，并且其选项符合指定的 dtype 和 layout
static bool hasTensorWithOptions(
    const IValue& ivalue,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout) {
  // 如果 ivalue 不是 Tensor，则返回 false
  if (!ivalue.isTensor()) {
    return false;
  }
  // 获取 ivalue 对应的 Tensor 对象
  const auto& tensor = ivalue.toTensor();
  // 检查传入的 dtype 和 layout 是否与 tensor 的当前选项相匹配
  if (dtype == tensor.dtype().toScalarType() &&
      layout == tensor.options().layout_opt()) {
    return true;
  }
  // 若选项不匹配，则打印警告信息并返回 false
  VLOG(1) << "tensor exists, but tensor options were different";
  return false;
}

// 检查给定的 IValue 是否是一个 Tensor，并且其选项同时满足 dtype、layout 和 memory_format 的要求
static bool hasTensorWithOptions(
    const IValue& ivalue,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::MemoryFormat> memory_format) {
  // 调用前一个函数检查 dtype 和 layout 是否满足要求，并且检查 memory_format
  return hasTensorWithOptions(ivalue, dtype, layout) &&
      (memory_format == ivalue.toTensor().options().memory_format_opt());
}

// 注册自定义操作符 aten::full 的函数处理器
REGISTER_OPERATOR_FUNCTOR(aten::full, aten_full, [](Node* n) -> SROperator {
  // 检查节点 n 是否符合指定的 aten::full 的 schema，如果不符合则记录并返回空指针
  if (!n->matches(torch::schema(
          "aten::full(int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"))) {
    LogAndDumpSchema(n);  // 记录不匹配的 schema
    return nullptr;
  }
  // 返回一个 lambda 函数，处理给定的 ProcessedNode
  return [](ProcessedNode* p_node) {
    // 从输入节点中获取 size、fill_value、dtype 和 layout
    const auto& size = p_node->Input(0).toDimVector();
    const auto fill_value = p_node->Input(1).toScalar();
    const auto dtype = p_node->Input(2).toOptional<c10::ScalarType>();
    const auto layout = p_node->Input(3).toOptional<c10::Layout>();
    // 如果输出节点中的 Tensor 选项不符合指定的 dtype 和 layout，则重新创建 Tensor
    if (!hasTensorWithOptions(p_node->Output(0), dtype, layout)) {
      const auto device = p_node->Input(4).toOptional<c10::Device>();
      const auto pin_memory = p_node->Input(5).toOptional<bool>();
      p_node->Output(0) =
          at::native::full(size, fill_value, dtype, layout, device, pin_memory);
      return;
    }
    // 如果 Tensor 选项匹配，则直接使用输出节点中的 Tensor 创建 full Tensor
    p_node->Output(0) =
        at::native::full_out(size, fill_value, p_node->Output(0).toTensor());
  };
});

// 注册自定义操作符 aten::full_like 的函数处理器
REGISTER_OPERATOR_FUNCTOR(aten::full_like, aten_full_like, [](Node* n) -> SROperator {
  // 检查节点 n 是否符合指定的 aten::full_like 的 schema，如果不符合则记录并返回空指针
  if (!n->matches(torch::schema(
          "aten::full_like(Tensor self, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"))) {
    LogAndDumpSchema(n);  // 记录不匹配的 schema
    return nullptr;
  }
  // 返回一个 lambda 函数，处理给定的 ProcessedNode
  return [](ProcessedNode* p_node) {
    // 从输入节点中获取 fill_value 和 self Tensor
    const auto in1_s = p_node->Input(1).toScalar();
    const auto& in0_t = p_node->Input(0).toTensor();
    // 获取 dtype、layout 和 memory_format
    const auto dtype = p_node->Input(2).toOptional<c10::ScalarType>();
    const auto layout = p_node->Input(3).toOptional<c10::Layout>();
    // 如果输出节点中的 Tensor 选项不符合指定的 dtype、layout 和 memory_format，则重新创建 Tensor
    if (!hasTensorWithOptions(p_node->Output(0), dtype, layout)) {
      const auto device = p_node->Input(4).toOptional<c10::Device>();
      const auto pin_memory = p_node->Input(5).toOptional<bool>();
      const auto memory_format =
          p_node->Input(6).toOptional<c10::MemoryFormat>();
      p_node->Output(0) = at::native::empty_like(
          in0_t, dtype, layout, device, pin_memory, memory_format);
    }
    // 获取输出节点中的 Tensor 引用，并进行相应操作
    auto& out_t = p_node->Output(0).toTensor();
    // 使用指定的尺寸调整 out_t 张量，以匹配 in0_t 张量的尺寸，不进行内插
    at::native::resize_(out_t, in0_t.sizes(), c10::nullopt);
    // 使用 in1_s 中的值填充 out_t 张量
    at::native::fill_out(out_t, in1_s);
});

REGISTER_OPERATOR_FUNCTOR(aten::ones, aten_ones, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 torch schema，如果不匹配则记录日志并返回空指针
  if (!n->matches(torch::schema(
          "aten::ones(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"))) {
    LogAndDumpSchema(n);  // 记录并输出节点的 schema 信息
    return nullptr;       // 返回空指针，表示操作未注册
  }
  // 返回一个 lambda 函数，该函数处理给定节点的逻辑
  return [](ProcessedNode* p_node) {
    const auto size = p_node->Input(0).toDimVector();  // 获取输入的尺寸信息
    if (p_node->Output(0).isNone()) {
      const auto dtype = p_node->Input(1).toOptional<c10::ScalarType>();  // 获取可选的数据类型
      const auto layout = p_node->Input(2).toOptional<c10::Layout>();     // 获取可选的布局信息
      const auto device = p_node->Input(3).toOptional<c10::Device>();     // 获取可选的设备信息
      const auto pin_memory = p_node->Input(4).toOptional<bool>();        // 获取可选的内存固定信息
      // 在输出张量为空时，调用 ATen 函数创建全为 1 的张量，并赋值给输出节点
      p_node->Output(0) =
          at::native::ones(size, dtype, layout, device, pin_memory);
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();  // 获取输出张量的引用
    fastResizeToZero(out_t);                    // 快速调整输出张量的大小为零
    at::native::ones_out(size, out_t);          // 使用 ATen 函数填充输出张量为全为 1
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::ones_like, aten_ones_like, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 torch schema，如果不匹配则记录日志并返回空指针
  if (!n->matches(torch::schema(
          "aten::ones_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"))) {
    LogAndDumpSchema(n);  // 记录并输出节点的 schema 信息
    return nullptr;       // 返回空指针，表示操作未注册
  }
  // 返回一个 lambda 函数，该函数处理给定节点的逻辑
  return [](ProcessedNode* p_node) {
    const auto& self = p_node->Input(0).toTensor();  // 获取输入张量的引用
    const auto dtype = p_node->Input(1).toOptional<c10::ScalarType>();  // 获取可选的数据类型
    const auto layout = p_node->Input(2).toOptional<c10::Layout>();     // 获取可选的布局信息
    const auto device = p_node->Input(3).toOptional<c10::Device>();     // 获取可选的设备信息
    const auto pin_memory = p_node->Input(4).toOptional<bool>();        // 获取可选的内存固定信息
    const auto memory_format = p_node->Input(5).toOptional<c10::MemoryFormat>();  // 获取可选的内存格式信息
    // 如果输出张量不具有指定的选项，则调用 ATen 函数创建与输入张量相同形状的全为 1 的张量，并赋值给输出节点
    if (!hasTensorWithOptions(
            p_node->Output(0), dtype, layout, memory_format)) {
      p_node->Output(0) = at::native::ones_like(
          self, dtype, layout, device, pin_memory, memory_format);
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();  // 获取输出张量的引用
    fastResizeToZero(out_t);                    // 快速调整输出张量的大小为零
    at::native::ones_out(self.sizes(), out_t);  // 使用 ATen 函数填充输出张量为全为 1，形状与输入张量相同
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::zeros, aten_zeros, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的 torch schema，如果不匹配则记录日志并返回空指针
  if (!n->matches(torch::schema(
          "aten::zeros(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"))) {
    LogAndDumpSchema(n);  // 记录并输出节点的 schema 信息
    return nullptr;       // 返回空指针，表示操作未注册
  }
  // 返回一个 lambda 函数，该函数处理给定节点的逻辑
  return [](ProcessedNode* p_node) {
    const auto size = p_node->Input(0).toDimVector();  // 获取输入的尺寸信息
    const auto dtype = p_node->Input(1).toOptional<c10::ScalarType>();  // 获取可选的数据类型
    const auto layout = p_node->Input(2).toOptional<c10::Layout>();     // 获取可选的布局信息
    // 如果输出张量不具有指定的选项，则调用 ATen 函数创建全为 0 的张量，并赋值给输出节点
    if (!hasTensorWithOptions(p_node->Output(0), dtype, layout)) {
      p_node->Output(0) = at::compositeexplicitautograd::zeros(
          size, dtype, layout, c10::nullopt, c10::nullopt);
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();  // 获取输出张量的引用
    fastResizeToZero(out_t);                    // 快速调整输出张量的大小为零
    at::compositeexplicitautograd::zeros_out(out_t, size);  // 使用 ATen 函数填充输出张量为全为 0
  };
});
});

// 注册自定义操作符函数：aten::linear
REGISTER_OPERATOR_FUNCTOR(aten::linear, aten_linear, [](Node* n) -> SROperator {
  // 检查节点是否匹配给定的 Torch 脚本
  if (!n->matches(torch::schema(
          "aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor"))) {
    // 如果不匹配，则记录日志并转储节点的模式信息，然后返回空指针
    LogAndDumpSchema(n);
    return nullptr;
  }

  // 返回 lambda 函数，处理节点的逻辑
  return [](ProcessedNode* p_node) {
    // 获取节点输入的第一个、第二个和第三个张量
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto& in1_t = p_node->Input(1).toTensor();
    auto in2_t = p_node->Input(2).toOptional<at::Tensor>();

    // 如果节点的输出是空的
    if (p_node->Output(0).isNone()) {
      // 调用 Torch 的线性运算函数，将结果存入节点的输出中，并直接返回
      p_node->Output(0) = at::native::linear(in0_t, in1_t, in2_t);
      return;
    }

    // 获取节点的输出张量，并快速将其调整为零大小
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    // 调用 Torch 的线性运算函数，将结果存入节点的输出张量中
    at::native::linear_out(out_t, in0_t, in1_t, in2_t);
  };
});

// 注册自定义操作符函数：aten::linalg_norm
REGISTER_OPERATOR_FUNCTOR(aten::linalg_norm, aten_linalg_norm, [](Node* n) -> SROperator {
  // 检查节点是否匹配给定的 Torch 脚本
  if (n->matches(torch::schema(
          "aten::linalg_norm(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"))) {
    // 如果匹配，则返回 lambda 函数，处理节点的逻辑
    return [](ProcessedNode* p_node) {
      // 获取输入张量、维度、保持维度标志和数据类型
      const auto& input = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(2).toDimVector();
      const auto keepdim = p_node->Input(3).toBool();
      const auto dtype = p_node->Input(4).toOptional<c10::ScalarType>();

      // 如果节点的输出是空的
      if (p_node->Output(0).isNone()) {
        // 调用 Torch 的线性代数范数计算函数，将结果存入节点的输出中，并直接返回
        p_node->Output(0) = at::native::linalg_norm(
            input,
            p_node->Input(1).toOptional<at::Scalar>(),
            dim,
            keepdim,
            dtype);
        return;
      }

      // 获取节点的输出张量，并快速将其调整为零大小
      auto& output = p_node->Output(0).toTensor();
      fastResizeToZero(output);
      // 调用 Torch 的线性代数范数计算函数，将结果存入节点的输出张量中
      at::native::linalg_norm_out(
          input,
          p_node->Input(1).toOptional<at::Scalar>(),
          dim,
          keepdim,
          dtype,
          output);
    };
  }

  // 如果节点不匹配第一个 Torch 脚本，则检查是否匹配第二个 Torch 脚本
  if (n->matches(torch::schema(
          "aten::linalg_norm.ord_str(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"))) {
    // 如果匹配，则返回 lambda 函数，处理节点的逻辑
    return [](ProcessedNode* p_node) {
      // 获取输入张量、维度、保持维度标志和数据类型
      const auto& input = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(2).toDimVector();
      const auto keepdim = p_node->Input(3).toBool();
      const auto dtype = p_node->Input(4).toOptional<c10::ScalarType>();

      // 如果节点的输出是空的
      if (p_node->Output(0).isNone()) {
        // 调用 Torch 的线性代数范数计算函数，将结果存入节点的输出中，并直接返回
        p_node->Output(0) = at::native::linalg_norm(
            input, p_node->Input(1).toStringView(), dim, keepdim, dtype);
        return;
      }

      // 获取节点的输出张量，并快速将其调整为零大小
      auto& output = p_node->Output(0).toTensor();
      fastResizeToZero(output);
      // 调用 Torch 的线性代数范数计算函数，将结果存入节点的输出张量中
      at::native::linalg_norm_out(
          input, p_node->Input(1).toStringRef(), dim, keepdim, dtype, output);
    };
  }

  // 如果节点既不匹配第一个 Torch 脚本，也不匹配第二个 Torch 脚本，则记录日志并转储节点的模式信息，然后返回空指针
  LogAndDumpSchema(n);
  return nullptr;
});

// 注册自定义操作符函数：aten::cat
REGISTER_OPERATOR_FUNCTOR(aten::cat, aten_cat, [](Node* n) -> SROperator {
  // 检查节点是否匹配给定的 Torch 脚本
  if (!n->matches(
          torch::schema("aten::cat(Tensor[] tensors, int dim=0) -> Tensor"))) {
    // 如果不匹配，则记录日志并转储节点的模式信息，然后返回空指针
    LogAndDumpSchema(n);
    return nullptr;
  }

  // 返回 lambda 函数，处理节点的逻辑
  return [](ProcessedNode* p_node) {
    // 获取输入张量数组和维度
    const auto inputs = p_node->Input(0).toTensorVector();
    const auto dim = p_node->Input(1).toInt();

    // 进行 Torch 的张量拼接操作，并将结果存入节点的输出中
    p_node->Output(0) = at::cat(inputs, dim);
  };
});
    // 检查输入的张量列表是否非空，否则抛出错误信息
    TORCH_CHECK(!inputs.empty(), "concat expects non-empty tensor list");
    // 获取拼接维度，从节点的第二个输入中获取并转换为整数
    const auto dim = p_node->Input(1).toInt();
    // 如果节点的第一个输出不存在
    if (p_node->Output(0).isNone()) {
      // 将输入张量列表在指定维度上进行拼接，并将结果赋给节点的第一个输出
      p_node->Output(0) = at::cpu::cat(inputs, dim);
      // 函数返回
      return;
    }
    // 否则，获取节点的第一个输出并将其调整为大小为零
    auto& output = p_node->Output(0).toTensor();
    fastResizeToZero(output);
    // 在给定的输出张量上执行输入张量列表的拼接操作
    at::cpu::cat_outf(inputs, dim, output);
  };


这段代码看起来像是一个 C++ 或类似的语言中的函数或方法，它的作用是对输入的张量列表进行拼接操作，并将结果存储到节点的输出中。
// 注册操作符的函数处理器，处理 aten::cumsum 操作符
REGISTER_OPERATOR_FUNCTOR(aten::cumsum, aten_cumsum, [](Node* n) -> SROperator {
  // 检查节点 n 是否匹配指定的 torch::schema，如果不匹配则记录日志并返回空指针
  if (!n->matches(torch::schema(
          "aten::cumsum(Tensor self, int dim, ScalarType? dtype=None) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  // 返回一个 lambda 函数，处理节点的具体操作
  return [](ProcessedNode* p_node) {
    // 获取输入张量和维度参数
    const auto& input = p_node->Input(0).toTensor();
    const auto dim = p_node->Input(1).toInt();
    const auto dtype = p_node->Input(2).toOptional<c10::ScalarType>();
    // 如果输出张量为空，则直接计算累积和并赋给输出
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::cpu::cumsum(input, dim, dtype);
      return;
    }
    // 否则，获取输出张量并进行快速调整大小
    auto& output = p_node->Output(0).toTensor();
    fastResizeToZero(output);
    // 调用 cumsum_out 函数计算累积和并输出到指定张量
    at::cpu::cumsum_out(output, input, dim, dtype);
  };
});

// 注册操作符的函数处理器，处理 aten::nonzero 操作符
REGISTER_OPERATOR_FUNCTOR(
    aten::nonzero,
    aten_nonzero,
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配指定的 torch::schema，如果不匹配则记录日志并返回空指针
      if (!n->matches(torch::schema("aten::nonzero(Tensor self) -> Tensor"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      // 返回一个 lambda 函数，处理节点的具体操作
      return [](ProcessedNode* p_node) {
        // 获取输入张量
        const auto& input = p_node->Input(0).toTensor();
        // 如果输出张量为空，则调用非零值计算函数并赋给输出
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = at::native::nonzero_cpu(input);
          return;
        }
        // 否则，获取输出张量并进行快速调整大小
        auto& output = p_node->Output(0).toTensor();
        fastResizeToZero(output);
        // 调用 nonzero_out_cpu 函数计算非零值并输出到指定张量
        at::native::nonzero_out_cpu(input, output);
      };
    });

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
// 注册操作符的函数处理器，处理 prim::VarConcat 操作符
REGISTER_OPERATOR_FUNCTOR(
    prim::VarConcat,
    prim_VarConcat,
    [](Node* n) -> SROperator {
      // 检查节点 n 的类型是否为 prim::VarConcat，如果不是则返回空指针
      if (!sr_schema_check_kind(n, prim::VarConcat)) {
        return nullptr;
      }
      // 返回一个 lambda 函数，处理节点的具体操作
      return [](ProcessedNode* p_node) {
        // 获取输入张量的数量和数据
        const size_t num_inputs = p_node->num_inputs();
        std::vector<at::Tensor> inputs(num_inputs - 1);
        for (const auto i : c10::irange(num_inputs - 1)) {
          inputs[i] = p_node->Input(i).toTensor();
        }
        // 获取连接维度
        auto dim = p_node->Input(num_inputs - 1).toInt();
        // 如果输出张量为空，则调用 cat 函数进行张量拼接并赋给输出
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = at::cpu::cat(inputs, dim);
          return;
        }
        // 否则，获取输出张量并进行快速调整大小
        auto& out_t = p_node->Output(0).toTensor();
        fastResizeToZero(out_t);
        // 调用 cat_outf 函数进行张量拼接并输出到指定张量
        at::cpu::cat_outf(inputs, dim, out_t);
      };
    });

// 匿名命名空间，包含一个模板函数和其特化版本，用于处理无符号类型取绝对值的情况
namespace {
template <class T>
T abs_if_signed(T val) {
  return std::abs(val);
}

template <>
unsigned char abs_if_signed<unsigned char>(unsigned char val) {
  return val;
}
} // namespace

// 计算 f(x) = sign(x) * ln(|1 + x|) 的输出张量
void signed_log1p_out(at::Tensor& out, const at::Tensor& input) {
  // 调整输出张量大小以匹配输入张量
  at::native::resize_(out, input.sizes(), c10::nullopt);

  // 获取输入张量的连续数据和输出张量的连续数据
  const auto input_contig = input.expect_contiguous();
  auto output_contig = out.expect_contiguous();

  // 根据输入张量的数据类型进行分发处理
  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "signed_log1p_kernel", [&]() {
    const auto input_data = input_contig->const_data_ptr<scalar_t>();
    // 获取可变指向输出连续数据的指针
    auto output_data = output_contig->mutable_data_ptr<float>();
    // 获取输入张量中元素的总数
    const auto N = input.numel();

    // 遍历输入张量中的每一个元素
    for (const auto i : c10::irange(N)) {
      // 判断当前元素的正负性，并确定其符号
      const int sign = input_data[i] < 0 ? -1 : 1;
      // 计算输入元素的绝对值，并应用对应的符号，然后计算其 log(1 + x) 的值
      output_data[i] = std::log1p(abs_if_signed(input_data[i])) * sign;
    }
  });
} // 结束命名空间

// 注册静态运行时的 signed_log1p 操作符
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_OPERATOR_FUNCTOR(
    static_runtime::signed_log1p,
    static_runtime_signed_log1p,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配指定的 schema，如果不匹配则输出日志并返回空指针
      if (!n->matches(torch::schema(
              "static_runtime::signed_log1p(Tensor x) -> Tensor"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      // 创建 signed_log1p 操作符
      auto te = createSignedLog1p();
      return [te](ProcessedNode* p_node) {
        // 获取输入张量
        const auto& input = p_node->Input(0).toTensor();
        // 如果输出张量为空，则创建一个与输入张量相同大小的空张量
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = create_empty_from(input);
        }
        auto& out = p_node->Output(0).toTensor();
        // 如果操作符为空或者输入类型不匹配，则快速调整输出张量大小为零，并执行 signed_log1p_out 操作
        if (!te || !te->checkInput<float>(input)) {
          fastResizeToZero(out);
          signed_log1p_out(out, input);
          return;
        }
        // 调整输出张量大小为输入张量大小，并执行 te 操作
        at::native::resize_(out, input.sizes(), c10::nullopt);
        int64_t nn = input.numel();
        te->call({out.data_ptr(), input.data_ptr(), &nn});
      };
    });

// 注册 aten::remainder 操作符
REGISTER_OPERATOR_FUNCTOR(
    aten::remainder,
    aten_remainder,
    [](Node* n) -> SROperator {
      // 匹配第一种 schema
      if (n->matches(torch::schema(
              "aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) =
                at::cpu::remainder(self, p_node->Input(1).toTensor());
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::remainder_out(out, self, p_node->Input(1).toTensor());
        };
      }
      // 匹配第二种 schema
      if (n->matches(torch::schema(
              "aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) =
                at::native::remainder(self, p_node->Input(1).toScalar());
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::remainder_out(self, p_node->Input(1).toScalar(), out);
        };
      }

      // 未识别的重载
      LogAndDumpSchema(n);
      return nullptr;
    });

// 注册 aten::where 操作符
REGISTER_OPERATOR_FUNCTOR(aten::where, aten_where, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::where.self(Tensor condition, Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& cond = p_node->Input(0).toTensor();
      const auto& self = p_node->Input(1).toTensor();
      const auto& other = p_node->Input(2).toTensor();

      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = create_empty_from(self);
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::where_self_out(cond, self, other, out);
    };
  }


注释：


    // 这里是一个函数或代码块的结尾
    // 可能是一个 if 语句、for 循环、while 循环、或者函数的末尾
    // 在这种情况下，它结束了一个函数的定义或者一个代码块的执行范围
  }



  LogAndDumpSchema(n);
  return nullptr;


注释：


  // 调用 LogAndDumpSchema 函数，记录和输出参数 n 的模式信息
  LogAndDumpSchema(n);
  // 返回空指针 nullptr，结束当前函数的执行
  return nullptr;
});

// 注册用于处理 prim::NumToTensor 操作的函数符号，返回一个处理节点的操作函数
REGISTER_OPERATOR_FUNCTOR(
    prim::NumToTensor,
    prim_NumToTensor,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配特定的 Torch 脚本模式，将标量转换为张量
      if (n->matches(
              torch::schema("prim::NumToTensor.Scalar(Scalar s) -> Tensor")) ||
          n->matches(
              torch::schema("prim::NumToTensor.bool(bool a) -> Tensor"))) {
        // 返回处理节点的操作函数
        return [](ProcessedNode* pnode) {
          // 从输入中获取标量值
          const auto scalar = pnode->Input(0).toScalar();
          // 如果输出为 None，则创建一个标量转换为张量的结果
          if (pnode->Output(0).isNone()) {
            pnode->Output(0) = at::scalar_to_tensor(scalar);
            return;
          }
          // 否则，获取输出张量并使用标量填充
          auto& out = pnode->Output(0).toTensor();
          at::detail::scalar_fill(out, scalar);
        };
      }
      // 如果节点不匹配任何已知模式，则记录并转储节点的模式信息
      LogAndDumpSchema(n);
      return nullptr; // 返回空指针表示未注册处理函数
    });

// 注册用于处理 quantized::embedding_bag_byte_unpack 操作的函数符号，返回一个处理节点的操作函数
REGISTER_OPERATOR_FUNCTOR(
    quantized::embedding_bag_byte_unpack,
    quantized_embedding_bag_byte_unpack,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配 quantized::embedding_bag_byte_unpack 的特定 Torch 脚本模式
      if (!sr_schema_check(
              n,
              "quantized::embedding_bag_byte_unpack(Tensor weight) -> Tensor")) {
        return nullptr; // 如果不匹配，返回空指针
      }
      // 返回处理节点的操作函数
      return [](ProcessedNode* pnode) {
        // 获取输入张量 weight
        auto& weight = pnode->Input(0).toTensor();
        // 如果输出为 None，则创建一个空张量作为输出
        if (pnode->Output(0).isNone()) {
          pnode->Output(0) = at::empty(
              {},
              weight.options().dtype(at::kFloat),
              weight.suggest_memory_format());
        }
        // 否则，获取输出张量并调用特定的量化 embedding bag 解包函数
        auto& out = pnode->Output(0).toTensor();
        at::native::qembeddingbag_byte_unpack_out(out, weight);
      };
    });

} // namespace torch::jit
```