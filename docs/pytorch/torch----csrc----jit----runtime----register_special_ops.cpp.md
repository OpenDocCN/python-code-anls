# `.\pytorch\torch\csrc\jit\runtime\register_special_ops.cpp`

```
// 包含 ATen 和 Torch 的必要头文件
#include <ATen/Context.h>
#include <torch/library.h>

#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/jit_type.h>
#include <c10/core/DefaultDtype.h>
#include <c10/util/irange.h>
#include <torch/csrc/api/include/torch/utils.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>

#include <ATen/InitialTensorOptions.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/jit/frontend/error_report.h>

#include <regex>
#include <sstream>

// Torch 的 JIT 命名空间
namespace torch::jit {

// 匿名命名空间，用于定义局部函数和变量
namespace {

// 从函数签名获取别名分析类型 FROM_SCHEMA
c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

// 返回保守的别名分析类型 CONSERVATIVE
c10::AliasAnalysisKind aliasAnalysisConservative() {
  return c10::AliasAnalysisKind::CONSERVATIVE;
}

// 检查列表输入类型的函数，抛出异常如果不是 int、float 或 bool 类型
void checkListInputType(const c10::TypePtr& elem_type, bool empty_list) {
  if (!elem_type->isSubtypeOf(*NumberType::get()) &&
      !elem_type->isSubtypeOf(*BoolType::get())) {
    std::stringstream error;
    error << "Input must be of ints, floats, or bools, "
          << "got " << elem_type->repr_str();
    // 对于空列表 torch.tensor([])，进行特殊处理
    if (elem_type->isSubtypeOf(*TensorType::get())) {
      if (empty_list) {
        error << "\nEmpty lists default to List[Tensor]. Add a variable "
                 "annotation to the assignment to create an empty list "
                 "of another type (torch.jit.annotate(List[T, []]) where T "
                 "is the type of elements in the list for Python 2)";
      }
    }
    throw std::runtime_error(error.str());
  }
}

// 将 Tensor 强制转换为指定类型和设备的函数
at::Tensor castTensorTo(
    at::Tensor self,
    const IValue& dtype,
    const IValue& device) {
  // 获取要转换的标量类型，默认为 self 的类型
  at::ScalarType scalar_type =
      dtype.isNone() ? self.scalar_type() : dtype.toScalarType();
  // 获取要转换的设备，默认为 self 的设备
  c10::Device dev = device.isNone() ? self.device() : device.toDevice();
  // 如果标量类型或设备与 self 不同，则进行类型和设备转换
  if (scalar_type != self.scalar_type() || dev != self.device()) {
    self = self.to(dev, scalar_type);
  }
  return self;
}

// 计算序列的每个维度大小的函数
std::vector<int64_t> compute_sizes(const IValue& seq) {
  std::vector<int64_t> sizes;
  auto seq_recur = seq.toList();
  // 递归计算序列的大小
  while (true) {
    sizes.push_back(seq_recur.size());
    // 如果序列为空或者第一个元素不是列表，则停止递归
    if (seq_recur.empty() || !seq_recur.get(0).isList()) {
      break;
    }
    seq_recur = seq_recur.get(0).toList();
  }
  return sizes;
}

// 检查指定维度的序列大小是否符合预期的函数
void checkSequenceSize(int64_t n, int64_t dim, int64_t seq_size) {
  // 如果序列大小不等于预期长度 n，则抛出异常
  if (seq_size != n) {
    AT_ERROR(
        "Expected sequence of length ",
        n,
        " at dim ",
        dim,
        " (got ",
        seq_size,
        ")");
  }
}

// 存储最后一个维度数据的模板函数
template <typename DTYPE>
void storeLastDimension(
    char* data,
    const std::vector<int64_t>& sizes,
    const c10::ArrayRef<int64_t>& strides,
    int64_t dim,
    int elementSize,
    at::ArrayRef<IValue> obj) {
  auto n = sizes[dim];
  auto seq_size = obj.size();
  // 检查序列大小是否符合预期
  checkSequenceSize(n, dim, seq_size);
  // 将每个元素转换为指定类型的数据并存储在 data 中
  for (const auto i : c10::irange(n)) {
    *(DTYPE*)data = obj[i].to<DTYPE>();
    // 更新数据指针以存储下一个元素
    data += elementSize;
  }
}
    data += strides[dim] * elementSize;


# 增加 data 变量的偏移量，用于访问下一个元素的位置
data += strides[dim] * elementSize;
}

void storeLastDimensionFloat(
    char* data,
    const std::vector<int64_t>& sizes,
    const c10::ArrayRef<int64_t>& strides,
    int64_t dim,
    int elementSize,
    at::ArrayRef<IValue> obj) {
  auto n = sizes[dim];  // 获取当前维度的大小
  auto seq_size = obj.size();  // 获取传入对象的长度
  checkSequenceSize(n, dim, seq_size);  // 检查传入对象的长度是否与当前维度大小一致
  for (const auto i : c10::irange(n)) {  // 遍历当前维度的所有元素
    *(float*)data = static_cast<float>(obj[i].to<double>());  // 将对象中的双精度浮点数转换为单精度浮点数，并存储到数据中
    data += strides[dim] * elementSize;  // 根据当前维度的步长和元素大小更新数据指针
  }
}

void storeLastDimensionHalf(
    char* data,
    const std::vector<int64_t>& sizes,
    const c10::ArrayRef<int64_t>& strides,
    int64_t dim,
    int elementSize,
    at::ArrayRef<IValue> obj) {
  auto n = sizes[dim];  // 获取当前维度的大小
  auto seq_size = obj.size();  // 获取传入对象的长度
  checkSequenceSize(n, dim, seq_size);  // 检查传入对象的长度是否与当前维度大小一致
  for (const auto i : c10::irange(n)) {  // 遍历当前维度的所有元素
    *(at::Half*)data = at::convert<at::Half, double>(obj[i].to<double>());  // 将对象中的双精度浮点数转换为半精度浮点数，并存储到数据中
    data += strides[dim] * elementSize;  // 根据当前维度的步长和元素大小更新数据指针
  }
}

// reference python implementation recursive_store in tensor_new.cpp
void recursiveStore(
    char* data,
    const std::vector<int64_t>& sizes,
    const c10::ArrayRef<int64_t>& strides,
    int64_t dim,
    int tenElementSize,
    const IValue& obj) {
  auto ndim = sizes.size();  // 获取张量的维度数
  auto n = sizes[dim];  // 获取当前维度的大小
  auto seq = obj.toListRef();  // 获取对象的列表引用
  checkSequenceSize(n, dim, seq.size());  // 检查传入对象的长度是否与当前维度大小一致
  if (dim + 1 < static_cast<long>(ndim)) {  // 如果当前维度不是最后一个维度
    for (const auto i : c10::irange(n)) {  // 遍历当前维度的所有元素
      recursiveStore(data, sizes, strides, dim + 1, tenElementSize, seq[i]);  // 递归存储下一个维度的数据
      data += strides[dim] * tenElementSize;  // 根据当前维度的步长和元素大小更新数据指针
    }
  } else {  // 如果当前维度是最后一个维度
    if (obj.isIntList()) {  // 如果对象是整数列表
      storeLastDimension<int64_t>(
          data, sizes, strides, dim, tenElementSize, seq);  // 存储整数列表数据到张量中
    } else if (obj.isBoolList()) {  // 如果对象是布尔值列表
      storeLastDimension<bool>(
          data, sizes, strides, dim, tenElementSize, seq);  // 存储布尔值列表数据到张量中
    } else if (obj.isDoubleList()) {  // 如果对象是双精度浮点数列表
      if (tenElementSize ==
          static_cast<int>(elementSize(at::ScalarType::Double))) {  // 如果元素大小与双精度浮点数相同
        storeLastDimension<double>(
            data, sizes, strides, dim, tenElementSize, seq);  // 存储双精度浮点数列表数据到张量中
      } else if (
          tenElementSize ==
          static_cast<int>(elementSize(at::ScalarType::Float))) {  // 如果元素大小与单精度浮点数相同
        storeLastDimensionFloat(data, sizes, strides, dim, tenElementSize, seq);  // 存储单精度浮点数列表数据到张量中
      } else if (
          tenElementSize ==
          static_cast<int>(elementSize(at::ScalarType::Half))) {  // 如果元素大小与半精度浮点数相同
        storeLastDimensionHalf(data, sizes, strides, dim, tenElementSize, seq);  // 存储半精度浮点数列表数据到张量中
      } else {
        TORCH_INTERNAL_ASSERT(false);  // 抛出错误，不支持的元素类型
      }
    } else {
      TORCH_INTERNAL_ASSERT(false);  // 抛出错误，不支持的数据类型
    }
  }
}

template <bool if_set_requires_grad>
void createTensorFromList(Stack& stack) {
  // torch.tensor has a fourth requires_grad arg but torch.as_tensor not, so
  // we use the template arg to distinguish between these two cases
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  bool requires_grad;
  IValue data;
  IValue dtype;
  IValue device;
  if (if_set_requires_grad) {
    pop(stack, data, dtype, device, requires_grad);  // 弹出数据堆栈中的数据，用于创建张量
  } else {
    pop(stack, data, dtype, device);

# 弹出栈顶元素，将参数 data、dtype、device 传递给 pop 函数进行处理

  }
  auto elem_type = data.type();
  # 获取参数 data 的类型信息，并赋值给 elem_type

  while (elem_type->isSubtypeOf(AnyListType::get())) {
    # 当 elem_type 是 AnyListType 的子类型时循环执行以下操作
    elem_type = elem_type->containedType(0);
    # 获取 elem_type 的第一个包含类型，并赋值给 elem_type
  }

  auto sizes = compute_sizes(data);
  # 调用 compute_sizes 函数计算参数 data 的大小，并将结果赋值给 sizes

  checkListInputType(elem_type, sizes.size() == 1 && sizes[0] == 0);
  # 检查列表输入类型是否符合预期，其中列表大小为 1 且第一个元素大小为 0

  at::ScalarType initial_scalar_type = scalarTypeFromJitType(*elem_type);
  # 根据 elem_type 获取初始的标量类型，并赋值给 initial_scalar_type

  if (initial_scalar_type == at::ScalarType::Double) {
    # 如果初始的标量类型是 Double
    initial_scalar_type = typeMetaToScalarType(c10::get_default_dtype());
    # 将初始的标量类型转换为默认类型的标量类型
  }

  auto tensor =
      at::empty(sizes, at::initialTensorOptions().dtype(initial_scalar_type));
  # 创建一个指定大小和标量类型的空张量，并将其赋值给 tensor

  if (tensor.numel() != 0) {
    # 如果张量的元素数量不为 0
    recursiveStore(
        (char*)tensor.data_ptr(),
        sizes,
        tensor.strides(),
        0,
        tensor.element_size(),
        data);
    # 递归地将数据存储到张量中，通过参数 tensor、sizes、data 等进行处理
  }

  tensor = castTensorTo(tensor, dtype, device);
  # 将张量 tensor 转换为指定的 dtype 和 device

  auto default_type = at::typeMetaToScalarType(at::get_default_dtype());
  # 获取默认类型的标量类型，并赋值给 default_type

  if (dtype.isNone() && tensor.scalar_type() != default_type &&
      tensor.numel() == 0) {
    # 如果 dtype 为空，并且张量的标量类型不等于默认类型，且张量的元素数量为 0
    TORCH_WARN(
        "Creating a tensor from an empty ",
        elem_type->repr_str(),
        "list will create a tensor of default floating point type  (currently ",
        default_type,
        ") in python but a tensor of type ",
        elem_type->repr_str(),
        " in torchscript.\n",
        "Pass in a dtype argument to ensure consistent behavior");
    # 输出警告信息，说明从空列表创建张量将会导致类型不一致的问题，建议传入 dtype 参数以确保一致的行为
  }

  if (if_set_requires_grad) {
    # 如果设置了 requires_grad
    tensor.set_requires_grad(requires_grad);
    # 设置张量的 requires_grad 属性为 requires_grad
  }

  push(stack, std::move(tensor));
  # 将处理完毕的张量 tensor 推入栈中
}

RegisterOperators reg({
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA(
            "aten::split(Tensor(a -> *) self, int[] split_sizes, int dim=0) -> Tensor(a)[]"),
        [](Stack& stack) {
          RECORD_FUNCTION("split_with_sizes", last(stack, 3));  // 记录函数调用，用于自动微分

          auto result = at::split_with_sizes(
              (std::move(peek(stack, 0, 3))).toTensor(),  // 提取栈顶的第一个参数，并转换为 Tensor 类型
              (std::move(peek(stack, 1, 3))).toDimVector(),  // 提取栈顶的第二个参数，并转换为维度向量
              (std::move(peek(stack, 2, 3))).toInt());  // 提取栈顶的第三个参数，并转换为整数
          drop(stack, 3);  // 从栈中移除处理过的三个参数
          pack(stack, std::move(result));  // 将处理后的结果重新打包压入栈中
        },
        aliasAnalysisFromSchema()),  // 使用 schema 进行别名分析
#define DEFINE_TORCH_TENSOR_OP(operator_type, c_type, tensor_creation_op)       \
  OperatorGenerator(                                                            \
      TORCH_SELECTIVE_SCHEMA(                                                   \
          "aten::tensor." #operator_type "(" #operator_type                     \
          " t, *, ScalarType? dtype=None, Device? device=None"                  \
          ", bool requires_grad=False) -> Tensor"),                             \
      [](Stack& stack) {                                                        \
        c_type scalar_val;                                                      \
        IValue dtype;                                                           \
        IValue device;                                                          \
        bool requires_grad;                                                     \
        // 从栈中弹出参数：标量值、数据类型、设备类型、是否需要梯度
        pop(stack, scalar_val, dtype, device, requires_grad);                   \
        // 使用给定的 tensor_creation_op 创建张量对象
        auto tensor = tensor_creation_op;                                       \
        // 将张量转换为指定的数据类型和设备类型
        tensor = castTensorTo(tensor, dtype, device);                           \
        // 设置张量是否需要梯度
        tensor.set_requires_grad(requires_grad);                                \
        // 将处理后的张量推送回栈中
        push(stack, std::move(tensor));                                         \
      },                                                                        \
      aliasAnalysisFromSchema()),                                               \
      OperatorGenerator(                                                        \
          TORCH_SELECTIVE_SCHEMA(                                               \
              "aten::as_tensor." #operator_type "(" #operator_type              \
              " t, *, ScalarType? dtype=None, Device? device=None) -> Tensor"), \
          [](Stack& stack) {                                                    \
            c_type scalar_val;                                                  \
            IValue dtype;                                                       \
            IValue device;                                                      \
            // 从栈中弹出参数：标量值、数据类型、设备类型
            pop(stack, scalar_val, dtype, device);                              \
            // 使用给定的 tensor_creation_op 创建张量对象
            auto tensor = tensor_creation_op;                                   \
            // 将张量转换为指定的数据类型和设备类型
            tensor = castTensorTo(tensor, dtype, device);                       \
            // 将处理后的张量推送回栈中
            push(stack, std::move(tensor));                                     \
          },                                                                    \
          aliasAnalysisFromSchema()),
    // 定义一个名为 bool 的 Torch 张量操作，创建一个空的标量张量，并用给定的标量值填充
    DEFINE_TORCH_TENSOR_OP(
        bool,
        bool,
        at::empty({}, at::CPU(at::kBool).options()).fill_(scalar_val))
        
    // 定义一个名为 float 的 Torch 张量操作，根据标量值创建一个浮点数张量
    DEFINE_TORCH_TENSOR_OP(
        float,
        double,
        at::native::scalar_tensor(
            scalar_val,
            typeMetaToScalarType(c10::get_default_dtype()),
            c10::nullopt /* layout */,
            at::kCPU,
            c10::nullopt /* pin_memory*/))
            
    // 定义一个名为 int 的 Torch 张量操作，将标量值转换为整数张量
    DEFINE_TORCH_TENSOR_OP(
        int,
        int64_t,
        at::scalar_to_tensor(scalar_val))
        
    // 定义一个名为 complex 的 Torch 张量操作，根据标量值创建一个复数张量
    DEFINE_TORCH_TENSOR_OP(
        complex,
        c10::complex<double>,
        at::native::scalar_tensor(
            scalar_val,
            typeMetaToScalarType(c10::get_default_complex_dtype()),
            c10::nullopt /* layout */,
            at::kCPU,
            c10::nullopt /* pin_memory */))

    // 引用 Python 实现：tensor_new.cpp 中的 internal_new_from_data 函数
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA("aten::_infer_size(int[] a, int[] b) -> int[]"),
        [](Stack& stack) {
          auto a = pop(stack);
          auto b = pop(stack);
          // 推入栈顶一个张量，其形状由 at::infer_size 函数推断而来
          push(stack, at::infer_size(a.toDimVector(), b.toDimVector()));
        },
        aliasAnalysisFromSchema()),

    // 引用 Python 实现：torch::NoGradGuard 类在脚本中支持设置梯度模式时移除
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_no_grad_embedding_renorm_(Tensor weight, Tensor input, float max_norm, float norm_type) -> Tensor"),
        [](Stack& stack) {
          at::Tensor weight;
          at::Tensor input;
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          double max_norm;
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          double norm_type;
          pop(stack, weight, input, max_norm, norm_type);

          // TODO: 在脚本支持设置梯度模式时移除
          torch::NoGradGuard no_grad;

          // 对权重进行重新归一化处理，返回处理后的张量
          at::Tensor result =
              at::embedding_renorm_(weight, input, max_norm, norm_type);
          // 将结果张量推入栈顶
          push(stack, std::move(result));
        },
        aliasAnalysisFromSchema()),

    // 引用 Python 实现：tensor 函数用于从数据列表创建张量
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA(
            "aten::tensor(t[] data, *, ScalarType? dtype=None, Device? device=None, bool requires_grad=False) -> Tensor"),
        createTensorFromList<true>,
        aliasAnalysisFromSchema()),
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA(
            "aten::as_tensor(Tensor(a) data, *, ScalarType? dtype=None, Device? device=None) -> Tensor(a|b)"),
        [](Stack& stack) {
          // 弹出堆栈顶部的元素，转换为可选的设备类型和标量类型
          auto device = pop(stack).toOptional<c10::Device>();
          auto dtype = pop(stack).toOptional<at::ScalarType>();
          // 弹出堆栈顶部的张量数据
          at::Tensor data = pop(stack).toTensor();
          // 确定要使用的标量类型，默认为数据的当前标量类型
          at::ScalarType scalar_type =
              dtype ? dtype.value() : data.scalar_type();
          // 确定要使用的设备，默认为数据的当前设备
          c10::Device dev = device ? device.value() : data.device();
    
          // 如果指定的标量类型或设备与数据当前的标量类型或设备不同，则进行类型和设备的转换
          if (scalar_type != data.scalar_type() || dev != data.device()) {
            data = data.to(
                dev, scalar_type, /*non_blocking=*/false, /*copy=*/false);
          }
          // 将处理完的张量压入堆栈
          push(stack, std::move(data));
        },
        aliasAnalysisFromSchema()),
    
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA(
            "aten::as_tensor.list(t[] data, *, ScalarType? dtype=None, Device? device=None) -> Tensor"),
        createTensorFromList<false>,
        aliasAnalysisFromSchema()),
    
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_pack_sequence(Tensor output, Tensor batch_sizes, Tensor? sorted_indices, "
            "Tensor? unsorted_indices) -> (Tensor, Tensor, Tensor?, Tensor?)"),
        [](Stack& stack) {},
        aliasAnalysisFromSchema()),
    
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA("aten::_get_tracing_state() -> bool"),
        [](Stack& stack) { push(stack, false); },
        aliasAnalysisFromSchema()),
    
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA("aten::is_scripting() -> bool"),
        [](Stack& stack) { push(stack, true); },
        aliasAnalysisFromSchema()),
    
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA("aten::has_torch_function(...) -> bool"),
        [](Stack& stack) { push(stack, false); },
        aliasAnalysisFromSchema()),
    
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_no_grad_uniform_(Tensor(a!) tensor, float a, float b, Generator? generator=None) -> Tensor(a!)"),
        [](Stack& stack) {
          // TODO: remove when script supports setting grad mode
          // 进入无梯度模式保护块，确保操作不会影响梯度计算
          torch::NoGradGuard no_grad;
    
          at::Tensor tensor;
          // 弹出堆栈顶部的张量、浮点数 a 和 b
          double a;
          double b;
          std::optional<at::Generator> generator =
              pop(stack).toOptional<at::Generator>();
    
          pop(stack, tensor, a, b);
          // 对张量执行从区间 [a, b) 均匀分布的随机初始化，可指定生成器
          push(stack, tensor.uniform_(a, b, generator));
        },
        aliasAnalysisFromSchema()),
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_no_grad_normal_(Tensor(a!) tensor, float mean, float std, Generator? generator=None) -> Tensor(a!)"),
        [](Stack& stack) {
          // TODO: 当脚本支持设置梯度模式时移除此处代码
          // 进入无梯度模式
          torch::NoGradGuard no_grad;

          at::Tensor tensor;
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          double mean;
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          double std;
          // 从堆栈中弹出可选的生成器对象
          std::optional<at::Generator> generator =
              pop(stack).toOptional<at::Generator>();

          // 从堆栈中弹出张量、均值和标准差
          pop(stack, tensor, mean, std);
          // 将正态分布值填充到张量中
          push(stack, tensor.normal_(mean, std, generator));
        },
        aliasAnalysisFromSchema()),

    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_no_grad_fill_(Tensor(a!) tensor, float val) -> Tensor(a!)"),
        [](Stack& stack) {
          // TODO: 当脚本支持设置梯度模式时移除此处代码
          // 进入无梯度模式
          torch::NoGradGuard no_grad;

          at::Tensor tensor;
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          double val;
          // 从堆栈中弹出张量和填充值
          pop(stack, tensor, val);
          // 使用给定值填充张量
          push(stack, at::fill_(tensor, val));
        },
        aliasAnalysisFromSchema()),

    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_no_grad_zero_(Tensor(a!) tensor) -> Tensor(a!)"),
        [](Stack& stack) {
          // TODO: 当脚本支持设置梯度模式时移除此处代码
          // 进入无梯度模式
          torch::NoGradGuard no_grad;

          at::Tensor tensor;
          // 从堆栈中弹出张量
          pop(stack, tensor);
          // 将张量清零
          push(stack, at::zero_(tensor));
        },
        aliasAnalysisFromSchema()),

    Operator(
        "aten::is_grad_enabled() -> bool",
        [](Stack& stack) { push(stack, torch::GradMode::is_enabled()); },
        aliasAnalysisConservative()),

    Operator(
        "aten::set_grad_enabled(bool val) -> ()",
        [](Stack& stack) { torch::GradMode::set_enabled(pop(stack).toBool()); },
        aliasAnalysisConservative()),

    Operator(
        "aten::_get_cpu_capability() -> str",
        [](Stack& stack) { push(stack, at::get_cpu_capability()); },
        aliasAnalysisConservative()),
});
} // namespace torch::jit
} // namespace
```