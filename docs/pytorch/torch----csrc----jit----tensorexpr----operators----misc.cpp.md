# `.\pytorch\torch\csrc\jit\tensorexpr\operators\misc.cpp`

```
// 引入 Torch 的头文件，用于张量表达式计算
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/tensorexpr/operators/misc.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

// Torch 的命名空间，用于张量表达式计算的 JIT 模块
namespace torch {
namespace jit {
namespace tensorexpr {

// 标准化并检查索引，确保在列表大小范围内
int64_t normalizeAndCheckIndex(int64_t idx, int64_t list_size) {
  if (idx < 0) {
    // 处理负数索引，转换为正数索引
    idx = list_size + idx;
  }

  // 检查索引是否在有效范围内，否则抛出错误
  if (idx < 0 || idx >= list_size) {
    AT_ERROR("Invalid index ", idx, " for list_size", list_size);
  }
  return idx;
}

// 将布尔值表达式转换为整数，如果需要的话
ExprHandle boolToInteger(const ExprHandle& x) {
  return x.dtype().scalar_type() == ScalarType::Bool ? cast<int>(x) : x;
}

// 将表达式提升到指定的标量数据类型
ExprHandle promoteToDtype(ExprHandle e, ScalarType dt) {
  if (e.dtype().scalar_type() == dt) {
    return e;
  }

  // 根据目标数据类型进行类型转换
  switch (dt) {
#define TYPE_CASE(Type, Name) \
  case ScalarType::Name:      \
    e = cast<Type>(e);        \
    break;
    AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TYPE_CASE);
#undef TYPE_CASE
    case ScalarType::QUInt8:
      e = cast<c10::quint8>(e);
      break;
    case ScalarType::QInt8:
      e = cast<c10::qint8>(e);
      break;
    default:
      throw unsupported_dtype();
  }
  return e;
}

// 检查最高类型是否满足给定的类型约束
static bool checkTypes(const ScalarType highType, const int typeConstraints) {
  if (typeConstraints == kAllTypes) {
    return true;
  }

  // 根据高级数据类型检查是否满足特定类型约束
  if (c10::isIntegralType(highType, false)) {
    return (typeConstraints & kIntegralTypes) != 0;
  } else if (c10::isFloatingType(highType)) {
    return (typeConstraints & kFloatingPointTypes) != 0;
  } else if (highType == ScalarType::Bool) {
    return (typeConstraints & kBoolType) != 0;
  }

  // JIT 模块暂不支持复数和 qint 类型
  TORCH_INTERNAL_ASSERT(
      (typeConstraints & (kQintTypes | kComplexTypes)) == 0,
      buildErrorMessage(
          "Qint and Complex types are not supported in the fuser."));
  return false;
}

// 检查表达式是否为标量（常量或变量）
static bool isScalar(ExprHandle e) {
  auto n = e.node();
  return n->isConstant() || to<Var>(n);
}

// 将半精度浮点数表达式提升为单精度浮点数表达式
ExprHandle promoteHalfToFloat(const ExprHandle& e) {
  auto scalarType = static_cast<c10::ScalarType>(e.dtype().scalar_type());
  auto floatType = static_cast<c10::ScalarType>(tensorexpr::ScalarType::Float);
  if (c10::isFloatingType(scalarType) &&
      (c10::elementSize(scalarType) < c10::elementSize(floatType))) {
    return Cast::make(
        Dtype(tensorexpr::ScalarType::Float, e.dtype().lanes()), e);
  } else {
    return e;
  }
}

// 提升输入表达式的数据类型以满足类型约束
void promoteInputs(std::vector<ExprHandle>& inputs, const int typeConstraints) {
  if (inputs.empty()) {
    return;
  }

  // 找出输入表达式中的最高数据类型
  ScalarType highType = inputs[0].dtype().scalar_type();
  for (const auto& input : inputs) {
    auto inputType = input.dtype().scalar_type();

  // 迭代每个输入表达式，确定最高数据类型
  for (const auto& input : inputs) {
    auto inputType = input.dtype().scalar_type();
    // 如果输入是标量（单一值），则进行以下判断和处理
    if (isScalar(input)) {
      // 如果 highType 是整数类型且 inputType 是浮点数类型，则将 highType 设置为默认的标量类型
      if (isIntegralType(highType, false) && isFloatingType(inputType)) {
        highType = c10::get_default_dtype_as_scalartype();
      } else if (highType == c10::kBool) {
        // 如果 highType 是布尔类型，则将其设置为 inputType
        highType = inputType;
      }
    } else {
      // 如果输入不是标量，则使用 promoteTypes 函数提升 highType 和 inputType 的类型
      highType = promoteTypes(highType, inputType);
    }
  }

  // 检查 highType 是否符合给定的类型约束 typeConstraints
  if (!checkTypes(highType, typeConstraints)) {
    // 如果不符合类型约束，则抛出 unsupported_dtype 异常
    throw unsupported_dtype();
  }

  // 遍历 inputs 数组中的每个元素 e，并将其提升为指定的数据类型 highType
  for (ExprHandle& e : inputs) {
    e = promoteToDtype(e, highType);
  }
}

// 将整数类型提升为默认类型的表达式处理函数
ExprHandle promoteIntegerToDefaultType(const ExprHandle& e) {
  // 获取表达式的标量类型
  auto scalarType = static_cast<c10::ScalarType>(e.dtype().scalar_type());
  // 如果不是整数类型（包括布尔型），直接返回原表达式
  if (!c10::isIntegralType(scalarType, /*includeBool*/ true)) {
    return e;
  }

  // 获取默认的数据类型
  auto defaultType = c10::typeMetaToScalarType(c10::get_default_dtype());

  // 我们意图将整数类型提升为浮点数类型
  TORCH_INTERNAL_ASSERT(
      !c10::isIntegralType(defaultType, /*includeBool*/ true));

  // 创建类型转换节点，将表达式 e 转换为默认数据类型
  return Cast::make(
      Dtype(
          static_cast<tensorexpr::ScalarType>(defaultType), e.dtype().lanes()),
      e);
}

// 降低表达式的输出类型处理函数
ExprHandle demoteOutput(
    const ExprHandle& e,
    const std::optional<ScalarType> type) {
  // 如果未指定类型，直接返回原表达式
  if (!type.has_value()) {
    return e;
  }
  // 如果指定类型与表达式当前类型相同，直接返回原表达式
  if (*type == e.dtype().scalar_type()) {
    return e;
  }

  // 根据指定类型进行类型转换
  switch (*type) {
// NOLINTNEXTLINE
#define TYPE_CASE(Type, Name) \
  case ScalarType::Name:      \
    return cast<Type>(e);
    // 针对所有标量类型生成类型转换的 case 分支
    AT_FORALL_SCALAR_TYPES_AND2(Half, BFloat16, TYPE_CASE);
#undef TYPE_CASE
    // 如果指定类型是布尔型，进行布尔型的类型转换
    case ScalarType::Bool:
      return cast<bool>(e);
    // 不支持的数据类型抛出异常
    default:
      throw unsupported_dtype();
  }

  return e;
}

// 获取缓存对象信息的函数，返回一个包含维度和数据类型的 TensorInfo 结构体的可选值
std::optional<TensorInfo> getTensorInfo(BufHandle b) {
  std::vector<int64_t> dims;
  // 遍历缓存对象的维度
  for (auto dim : b.dims()) {
    // 获取维度的整数值，如果无法获取到整数值，返回空值
    auto val = intValue(dim.node());
    if (!val) {
      return c10::nullopt;
    }
    dims.push_back(*val);
  }
  // 返回包含维度和数据类型的 TensorInfo 结构体
  return TensorInfo{dims, static_cast<at::ScalarType>(b.dtype().scalar_type())};
}

// 对输入表达式进行范围限制的函数
ExprHandle clamp(
    const ExprHandle& cmin,
    const ExprHandle& cmax,
    const ExprHandle& input) {
  // 使用比较选择节点构造最小值的比较表达式
  auto mm = CompareSelect::make(input, cmin, cmin, input, kLT);
  // 使用比较选择节点构造最大值的比较表达式
  return CompareSelect::make(mm, cmax, cmax, mm, kGT);
}

// 判断表达式是否为 1 的静态函数
static bool isOne(ExprHandle e) {
  // 获取表达式的整数值
  auto const& n = intValue(e);
  // 如果无法获取到整数值，返回 false
  if (!n) {
    return false;
  }
  // 返回表达式是否为 1
  return *n == 1;
}

// 广播形状的实现函数，返回表达式向量和是否存在广播的布尔值
static std::pair<std::vector<ExprHandle>, bool> broadcastShapesImpl(
    const std::vector<ExprHandle>& a,
    const std::vector<ExprHandle>& b) {
  // 使用反向迭代器遍历两个向量的表达式
  auto at = a.rbegin();
  auto bt = b.rbegin();
  std::vector<ExprHandle> ret;
  bool hasBroadcast = false;
  // 当任一向量还有元素未处理时循环
  while (at != a.rend() || bt != b.rend()) {
    // 如果向量 a 已处理完，将向量 b 的元素推入结果中并标记广播
    if (at == a.rend()) {
      hasBroadcast = true;
      ret.push_back(*bt++);
      continue;
    }
    // 如果向量 b 已处理完，将向量 a 的元素推入结果中并标记广播
    if (bt == b.rend()) {
      hasBroadcast = true;
      ret.push_back(*at++);
      continue;
    }
    // 如果向量 a 和 b 的当前元素有一个为 1，选择非 1 的元素作为结果的维度
    ExprHandle dim = *at;
    if (isOne(*at)) {
      if (!isOne(*bt)) {
        dim = *bt;
        hasBroadcast = true;
      }
    }
    // 将选择的维度推入结果向量
    ret.push_back(dim);
    at++;
    bt++;
  }
  // 翻转结果向量并返回
  std::reverse(ret.begin(), ret.end());
  return {ret, hasBroadcast};
}

// 广播形状的实现函数，接受嵌套的表达式向量，并调用上述版本的实现函数处理
static std::pair<std::vector<ExprHandle>, bool> broadcastShapesImpl(
    std::vector<std::vector<ExprHandle>> shapes) {
  // 获取形状向量的大小
  size_t n = shapes.size();
  // 如果形状向量的大小为 1，直接调用单个向量版本的实现函数处理
  if (n == 1) {
    // 返回一个包含 shapes 第一个元素和布尔值 false 的集合
    return {shapes[0], false};
  }
  // 调用 broadcastShapesImpl 函数，处理 shapes 中倒数第二和最后一个元素的广播形状
  auto res1 = broadcastShapesImpl(shapes[n - 2], shapes[n - 1]);
  // 将第一个广播后的形状放入 shapes 数组中
  shapes[n - 2] = res1.first;
  // 移除 shapes 数组中的最后一个元素
  shapes.pop_back();
  // 调用 broadcastShapesImpl 函数，处理整个 shapes 数组的广播形状
  auto res2 = broadcastShapesImpl(shapes);
  // 返回一个包含第一个调用结果的第一个元素和两个调用结果的逻辑或的布尔值的集合
  return {res2.first, (res1.second || res2.second)};
}

// 广播多个形状的函数，返回广播后的形状向量
std::vector<ExprHandle> broadcastShapes(
    std::vector<std::vector<ExprHandle>> shapes) {
  return broadcastShapesImpl(shapes).first;
}

// 广播两个形状的函数重载，返回广播后的形状向量
std::vector<ExprHandle> broadcastShapes(
    const std::vector<ExprHandle>& a,
    const std::vector<ExprHandle>& b) {
  return broadcastShapesImpl(a, b).first;
}

// 根据参数值返回其形状向量
std::vector<ExprHandle> valueShape(const ArgValue& v) {
  if (auto b = std::get_if<tensorexpr::BufHandle>(&v)) {
    return b->dims();
  }
  return {};
}

// 根据参数值和轴向量返回张量或常数的表达式
ExprHandle tensorOrConstant(
    const ArgValue& v,
    const std::vector<ExprHandle>& axes) {
  if (auto b = std::get_if<BufHandle>(&v)) {
    return broadcast(*b, axes);
  }
  return constant(v);
}

// 根据参数值返回标量或常数的表达式
ExprHandle scalarOrConstant(const ArgValue& v) {
  if (auto vh = std::get_if<VarHandle>(&v)) {
    return *vh;
  }
  return constant(v);
}

// 根据缓冲区和轴向量返回广播加载的表达式
ExprHandle broadcast(BufHandle b, const std::vector<ExprHandle>& axes) {
  return b.load(computeIndicesToBroadcast(axes, b.dims()));
}

// 根据参数值返回常数的表达式
ExprHandle constant(const ArgValue& v) {
  if (auto s = std::get_if<tensorexpr::VarHandle>(&v)) {
    return *s;
  } else if (auto d = std::get_if<double>(&v)) {
    return DoubleImm::make(*d);
  } else if (auto i = std::get_if<int64_t>(&v)) {
    return LongImm::make(*i);
  } else if (auto b = std::get_if<bool>(&v)) {
    return BoolImm::make(*b);
  } else if (std::get_if<ArgNone>(&v)) {
    // 这只是一个占位符，防止抛出异常。None 的处理应该在特定操作的降低代码中正确处理。
    return IntImm::make(0);
  } else {
    throw unsupported_dtype("Trying to convert unsupported dtype to constant");
  }
}

// 计算用于广播的索引向量
std::vector<ExprHandle> computeIndicesToBroadcast(
    const std::vector<ExprHandle>& outputAxes,
    const std::vector<ExprHandle>& inputSizes) {
  if (outputAxes.size() < inputSizes.size()) {
    throw malformed_input("Cannot broadcast to a lower rank tensor");
  }
  std::vector<ExprHandle> bcast;
  auto axisIt = outputAxes.rbegin();
  auto sizeIt = inputSizes.rbegin();
  while (sizeIt != inputSizes.rend()) {
    auto const& size = intValue(*sizeIt);
    if (size && *size == 1) {
      bcast.emplace_back(LongImm::make(0));
    } else {
      bcast.emplace_back(*axisIt);
    }
    ++axisIt;
    ++sizeIt;
  }
  std::reverse(bcast.begin(), bcast.end());
  return bcast;
}

// 计算块的函数，返回计算块的张量
Tensor computeChunk(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device) {
  return Compute(
      "prim_constantchunk",
      outputShape,
      [inputs](const std::vector<VarHandle>& axes) {
        // 获取输入的第一个变量作为缓冲区句柄
        const auto& b = std::get<BufHandle>(inputs[0]);
        // 获取输入的第二个变量作为块索引
        int64_t chunkIdx = std::get<int64_t>(inputs[1]);
        // 获取输入的第三个变量作为维度
        int64_t dim = std::get<int64_t>(inputs[2]);
        // 获取输入的第四个变量作为块数量
        int64_t chunks = std::get<int64_t>(inputs[3]);
        // 将轴列表转换为表达式处理对象的索引列表
        std::vector<ExprHandle> indices(axes.begin(), axes.end());

        // 标准化并检查维度索引
        auto norm_dim = normalizeAndCheckIndex(dim, indices.size());
        // 获取缓冲区信息
        auto buf_info = getTensorInfo(b);
        // 计算步长
        size_t step = buf_info->dims[norm_dim] / chunks;

        // 创建新的索引列表
        std::vector<ExprHandle> new_indices;
        for (int64_t i = 0; i < static_cast<int64_t>(indices.size()); ++i) {
          // 如果当前索引是标准化的维度索引，则加入偏移量
          if (i == norm_dim) {
            new_indices.push_back(
                indices[i] + ExprHandle(immLike(indices[i], chunkIdx * step)));
          } else {
            new_indices.push_back(indices[i]);
          }
        }

        // 使用新的索引列表加载缓冲区数据并返回
        return b.load(new_indices);
      });
}

Tensor computeTranspose(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device) {
  auto A = std::get<BufHandle>(inputs[0]);
  // 如果输入张量 A 的维度小于等于1，则转置操作仅仅是复制
  if (A.ndim() <= 1) {
    // 返回一个 Compute 对象，表示转置操作，使用 outputShape 来定义形状
    return Compute(
        "aten_transpose", outputShape, [&](std::vector<VarHandle> axes) {
          TORCH_INTERNAL_ASSERT(
              axes.size() <= 1,
              buildErrorMessage("Invalid axes size in transpose"));
          // 加载张量 A 在指定轴 axes 上的数据
          return A.load(axes);
        });
  }
  // 一般情况下，转置操作会交换维度
  auto start_dim = at::maybe_wrap_dim(std::get<int64_t>(inputs[1]), A.ndim());
  auto to_dim = at::maybe_wrap_dim(std::get<int64_t>(inputs[2]), A.ndim());
  // 返回一个 Compute 对象，表示转置操作，使用 outputShape 来定义形状
  return Compute(
      "aten_transpose", outputShape, [&](std::vector<VarHandle> axes) {
        // 交换指定的维度 start_dim 和 to_dim
        std::swap(axes[start_dim], axes[to_dim]);
        // 加载张量 A 在调整后的轴 axes 上的数据
        return A.load(axes);
      });
}

Tensor computeExpand(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device) {
  auto A = std::get<BufHandle>(inputs[0]);
  // 返回一个 Compute 对象，表示扩展操作，使用 outputShape 来定义形状
  return Compute(
      "aten_expand", outputShape, [&](const std::vector<VarHandle>& axes) {
        // 将 VarHandle 类型的 axes 转换为 ExprHandle 类型的 indices
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        // 对张量 A 进行广播操作，返回广播后的结果
        return broadcast(A, indices);
      });
}

Tensor computeReshape(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device) {
  auto A = std::get<BufHandle>(inputs[0]);
  // 如果输入张量 A 的维度为0
  if (A.ndim() == 0) {
    // 返回一个 Compute 对象，用于执行 "aten_view" 操作，根据指定的 outputShape
    // 参数进行计算，并在匿名函数中定义操作逻辑
    return Compute(
        "aten_view", outputShape, [&](const std::vector<VarHandle>& axes) {
          // 创建一个空的索引列表
          std::vector<ExprHandle> empty_indices;
          // 调用 A 对象的 load 方法，加载数据，此时使用空的索引
          return A.load(empty_indices);
        });
  }
  // 如果不满足上述条件，则返回一个 Compute 对象，用于执行 "aten_reshape" 操作，
  // 根据指定的 outputShape 参数进行计算，并在匿名函数中定义操作逻辑
  return Compute(
      "aten_reshape", outputShape, [&](const std::vector<VarHandle>& axes) {
        // 创建一个新的轴列表
        std::vector<VarHandle> new_axes;
        // 断言确保 outputShape 的大小与 axes 的大小相等
        assert(outputShape.size() == axes.size());
        /*
        示例用于索引转换。假设有张量 A 和它的视图 B:
          A.size() = [6,2,3]
          B = A.view(2,1,9,1,2)

        在 TE IR 中，我们希望将 B 表示为以下循环嵌套：
          for (i1 in 0..2)
            for (i2 in 0..1)
              for (i3 in 0..9)
                for (i4 in 0..1)
                  for (i5 in 0..2)
                    idx = i5 + i4*2 + i3*2 + i2*18 + i1*18
                    B[i1,i2,i3,i4,i5] = A[idx/(3*2), (idx/3)%2, idx%3]
        */
        // 创建维度和索引列表，用于表示张量的维度和轴
        std::vector<ExprPtr> dims, indices;
        for (size_t idx = 0; idx < outputShape.size(); idx++) {
          dims.push_back(outputShape[idx].node());
          indices.push_back(axes[idx].node());
        }

        // 计算维度数量
        auto ndim = dims.size();
        // 创建步长列表
        std::vector<ExprPtr> strides(ndim);
        // 设置最后一个步长为 1
        strides[ndim - 1] = immLike(dims[ndim - 1], 1);
        // 生成其他步长
        for (size_t i = 1; i < ndim; i++) {
          strides[ndim - 1 - i] = alloc<Mul>(strides[ndim - i], dims[ndim - i]);
        }

        // 计算扁平化索引
        ExprHandle flat_idx = ExprHandle(flatten_index(dims, indices, strides));
        // 创建原始缓冲区索引列表
        std::vector<ExprHandle> orig_buf_indexes(A.ndim(), ExprHandle(0));
        // 初始化步长
        ExprHandle stride = ExprHandle(immLike(flat_idx, 1));
        for (size_t idx = 0; idx < A.ndim(); idx++) {
          size_t dim_idx = A.ndim() - idx - 1;
          // 对于第一个维度，不需要生成模除操作
          if (dim_idx > 0) {
            orig_buf_indexes[dim_idx] = flat_idx / stride % A.dim(dim_idx);
          } else {
            orig_buf_indexes[dim_idx] = flat_idx / stride;
          }
          // 更新步长
          stride = stride * A.dim(dim_idx);
        }
        // 返回 A 对象根据原始缓冲区索引列表加载的数据
        return A.load(orig_buf_indexes);
      });
}

Tensor computeFlatten(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device) {
  std::vector<int64_t> outputShapeVec;
  // 遍历输出形状表达式列表，提取每个表达式的长整型值，存入向量
  for (const auto dim : c10::irange(outputShape.size())) {
    outputShapeVec.push_back(outputShape[dim].AsNode<LongImm>()->value());
  }
  std::vector<ArgValue> reshapeInputs;
  reshapeInputs.push_back(inputs[0]);
  reshapeInputs.emplace_back(outputShapeVec);
  // 调用 computeReshape 函数，执行数据重塑操作
  return computeReshape(
      reshapeInputs, outputShape, outputStrides, outputType, device);
}

static std::pair<ScalarType, std::vector<BufHandle>> processCatList(
    const std::vector<BufHandle>& bufList) {
  if (bufList.empty()) {
    // 如果输入缓冲列表为空，抛出运行时异常
    throw std::runtime_error("Empty input list is passed to aten::cat");
  }
  std::vector<BufHandle> bufInputs;
  std::vector<BufHandle> nonEmptyInputs;
  for (auto buf : bufList) {
    bufInputs.push_back(buf);
    // 断言缓冲的维度不为空，否则抛出错误
    TORCH_INTERNAL_ASSERT(
        !buf.node()->dims().empty(), buildErrorMessage("Invalid buf rank"));
    // 忽略任何维度上为0的缓冲
    bool hasEmptyDims = false;
    for (const auto& dim : buf.dims()) {
      if (dim.AsNode<LongImm>() && immediateAs<int64_t>(dim) == 0ll) {
        hasEmptyDims = true;
        break;
      }
    }
    if (!hasEmptyDims) {
      nonEmptyInputs.push_back(buf);
    }
  }
  // 计算缓冲列表中的数据类型的最高类型
  ScalarType highType = bufInputs[0].dtype().scalar_type();
  for (const auto& input : bufInputs) {
    auto maybe_dtype = input.dtype().scalar_type();
    highType = promoteTypes(highType, maybe_dtype);
  }
  // 返回最高类型及非空缓冲列表
  return {highType, nonEmptyInputs};
}

static Tensor computeCatWoConditionals(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides) {
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  auto input_list = std::get<BufList>(inputs[0]);
  auto arg_dim = inputs[1];
  auto cat_info = processCatList(input_list);
  ScalarType high_type = cat_info.first;
  std::vector<BufHandle> non_empty_inputs = cat_info.second;

  // 现在我们为每个输入构建一个循环:
  //
  // for i
  //   for j
  //     for k
  //       output[i,j,k] = inp1[i,j,k]
  // for i
  //   for j
  //     for k
  //       output[i,j+l1,k] = inp2[i,j,k]
  // for i
  //   for j
  //     for k
  //       output[i,j+l2,k] = inp3[i,j,k]

  // 将输出形状表达式向量转换为表达式向量
  auto output_sizes_expr = ExprHandleVectorToExprVector(outputShape);
  auto output_strides_expr = ExprHandleVectorToExprVector(outputStrides);
  // 分配一个缓冲区用于输出张量，命名为 "aten_cat"
  auto output_buf = alloc<Buf>(
      "aten_cat",
      output_sizes_expr,
      ToDtype(high_type),
      nullptr,
      output_strides_expr);
  if (non_empty_inputs.empty()) {
    // 返回一个 Tensor 对象，其中包含输出缓冲区和一个空的 Block 对象
    return Tensor(
        output_buf, alloc<tensorexpr::Block>(std::vector<StmtPtr>({})));
  }

  // 从参数中获取拼接维度的值，并进行规范化和检查索引的有效性
  int64_t concat_dim = std::get<int64_t>(arg_dim);
  auto norm_concat_dim = normalizeAndCheckIndex(concat_dim, outputShape.size());

  // 定义一个 lambda 函数，根据缓冲区的特性返回循环顺序
  auto loop_order_fn = [&](const BufPtr& buf_) {
    std::vector<int32_t> loop_order;
    if (buf_->is_contiguous()) {
      // 如果缓冲区是连续的，按照逆序填充循环顺序数组
      for (int32_t i = buf_->ndim() - 1; i >= 0; i--) {
        loop_order.push_back(i);
      }
    } else if (buf_->is_contiguous(c10::MemoryFormat::ChannelsLast)) {
      // 如果缓冲区以 ChannelsLast 格式连续，设置固定的循环顺序
      loop_order = {1, 3, 2, 0};
    } else if (buf_->is_contiguous(c10::MemoryFormat::ChannelsLast3d)) {
      // 如果缓冲区以 ChannelsLast3d 格式连续，设置固定的循环顺序
      loop_order = {1, 4, 3, 2, 0};
    } else {
      // 对于其他情况，设置默认的循环顺序
      loop_order = {1, 2, 0};
    }

    return loop_order;
  };

  // 定义一个 lambda 函数，生成输入缓冲区的代码段
  auto gen_code_for_input = [&](const BufHandle& inp,
                                size_t inp_pos,
                                ExprPtr concat_dim_size,
                                const std::vector<ExprHandle>& dims) {
    // 创建循环变量和加载/存储索引的数组
    std::vector<VarPtr> for_vars(dims.size());
    std::vector<ExprPtr> load_indices(dims.size());
    std::vector<ExprPtr> store_indices(dims.size());
    for (int64_t i = 0; i < static_cast<int64_t>(dims.size()); ++i) {
      // 创建循环变量，并为加载和存储索引赋值
      for_vars[i] = alloc<Var>(
          "i" + std::to_string(inp_pos) + "_" + std::to_string(i),
          dims[i].dtype());
      load_indices[i] = for_vars[i];
      if (i == norm_concat_dim) {
        // 如果当前维度是拼接维度，则存储索引增加拼接维度的大小
        store_indices[i] = alloc<Add>(for_vars[i], concat_dim_size);
      } else {
        store_indices[i] = for_vars[i];
      }
    }
    auto inp_buf = inp.node();
    auto load_expr = alloc<Load>(inp_buf, load_indices);
    auto load_promoted = promoteToDtype(ExprHandle(load_expr), high_type);
    // 创建一个存储语句对象，将加载的表达式存储到输出缓冲区
    StmtPtr st = alloc<Store>(output_buf, store_indices, load_promoted.node());

    // 获取当前输入缓冲区的循环顺序
    auto loop_order = loop_order_fn(inp.node());
    // 根据循环顺序创建嵌套的 for 循环语句
    for (auto dim_index : loop_order) {
      st = alloc<For>(
          for_vars[dim_index],
          immLike(dims[dim_index], 0),
          dims[dim_index].node(),
          st);
    }

    return st;
  };

  // 初始化拼接维度的大小为 null
  ExprPtr concat_dim_size = nullptr;
  // 创建一个空的 Block 对象
  auto block = alloc<tensorexpr::Block>(std::vector<StmtPtr>({}));
  // 遍历非空输入列表，为每个输入生成代码段，并更新拼接维度的大小
  for (size_t i = 0; i < non_empty_inputs.size(); ++i) {
    auto input_dims =
        ExprVectorToExprHandleVector(non_empty_inputs[i].node()->dims());
    if (concat_dim_size == nullptr) {
      // 如果拼接维度的大小为 null，则初始化为第一个输入的拼接维度大小
      concat_dim_size = immLike(input_dims[norm_concat_dim], 0);
    }
    // 生成当前输入的代码段，并将其添加到 Block 对象中
    block->append_stmt(gen_code_for_input(
        non_empty_inputs[i], i, concat_dim_size, input_dims));
    // 更新拼接维度的大小为当前输入的拼接维度大小之和
    concat_dim_size =
        alloc<Add>(concat_dim_size, input_dims[norm_concat_dim].node());
  }
  // 返回一个 Tensor 对象，其中包含输出缓冲区和简化后的 Block 对象
  return Tensor(output_buf, IRSimplifier::simplify(block));
}
// 结束 computeCat 函数的定义

Tensor computeCat(
    const std::vector<ArgValue>& inputs,   // 接收输入参数列表
    const std::vector<ExprHandle>& outputShape,  // 输出张量形状
    const std::vector<ExprHandle>& outputStrides,  // 输出张量步长
    const std::optional<ScalarType>& outputType,  // 可选的输出数据类型
    at::Device device) {  // 设备类型参数

  if (device == at::kCPU && getCatWoConditionals()) {  // 检查设备类型和条件
    // 如果在 CPU 上且不使用条件拼接，则调用相应函数处理
    return computeCatWoConditionals(inputs, outputShape, outputStrides);
  }

  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  auto inputList = std::get<BufList>(inputs[0]);  // 获取输入列表的缓冲列表
  auto argDim = inputs[1];  // 获取维度参数
  auto catInfo = processCatList(inputList);  // 处理缓冲列表，获取拼接信息
  ScalarType highType = catInfo.first;  // 获取拼接后的高级数据类型
  std::vector<BufHandle> nonEmptyInputs = catInfo.second;  // 获取非空输入缓冲列表

  return Compute(
      "aten_cat",  // 创建计算对象，表示拼接操作
      outputShape,  // 输出张量形状
      outputStrides,  // 输出张量步长
      [&](const std::vector<VarHandle>& axes) {  // 创建 lambda 函数，处理轴向
        if (nonEmptyInputs.empty()) {  // 如果没有非空输入
          return ExprHandle(0);  // 返回零表达式
        }

        int64_t dim_ = std::get<int64_t>(argDim);  // 获取维度值
        auto dim = normalizeAndCheckIndex(dim_, axes.size());  // 标准化和检查索引

        // 促使输入类型升级。
        // 需要考虑所有输入，包括空的 - 它们也会影响最终的数据类型。

        // 现在我们知道最终的数据类型，知道哪些输入是非空的，
        // 并且知道至少有一个这样的输入。有了这些信息，我们构造一个张量表达式来执行拼接。
        // 我们在这里构建的表达式是一系列的 if-then-else，本质上表示：
        //
        //              inp1[i, j, k]         if 0   < i < l1,
        // out[i,j,k] = inp2[i, j-l1, k]      if l1 =< i < l1 + l2,
        //              ...
        //              inpN[i, j-l_N_1, k]   if l1+l2+...l_N_1  < i
        // 其中 l_i 是第 i 个输入的对应大小。
        std::vector<ExprHandle> newAxes(axes.begin(), axes.end());  // 复制轴向列表
        ExprHandle load = promoteToDtype(  // 升级为高级数据类型
            tensorOrConstant(nonEmptyInputs[0], newAxes), highType);
        auto offset = ExprHandle(nonEmptyInputs[0].node()->dim(dim));  // 获取偏移量
        newAxes[dim] = newAxes[dim] - offset;  // 调整轴向偏移量

        for (size_t ii = 1; ii < nonEmptyInputs.size(); ++ii) {  // 遍历非空输入
          auto input = nonEmptyInputs[ii];
          load = ifThenElse(  // 构建条件表达式
              CompareSelect::make(axes[dim], offset, kLT),  // 比较和选择
              load,
              promoteToDtype(tensorOrConstant(input, newAxes), highType));  // 升级为高级数据类型

          offset = offset + ExprHandle(input.node()->dim(dim));  // 更新偏移量
          newAxes[dim] = axes[dim] - offset;  // 调整轴向偏移量
        }

        return load;  // 返回加载结果
      });
}

Tensor computeEmbedding(
    const std::vector<ArgValue>& inputs,  // 接收输入参数列表
    const std::vector<ExprHandle>& outputShape,  // 输出张量形状
    const std::vector<ExprHandle>& outputStrides,  // 输出张量步长
    const std::optional<ScalarType>& outputType,  // 可选的输出数据类型
    at::Device device) {  // 设备类型参数

  Dtype dtype = kFloat;  // 设置默认数据类型为浮点型
  if (outputType) {  // 如果指定了输出数据类型
  // 使用变量 *outputType* 创建一个 Dtype 类型的对象 *dtype*
  dtype = Dtype(*outputType);

  // 创建一个名为 ResultBuf 的 BufHandle 对象，用于保存嵌入结果，名称为 "emb"，形状为 outputShape，数据类型为 dtype
  BufHandle ResultBuf("emb", outputShape, dtype);

  // 获取输入列表中的第一个元素，应当是一个 BufHandle 类型的对象，并将其赋给变量 w
  const BufHandle& w = std::get<BufHandle>(inputs[0]);

  // 获取输入列表中的第二个元素，应当是一个 BufHandle 类型的对象，并将其赋给变量 indices
  const BufHandle& indices = std::get<BufHandle>(inputs[1]);

  // 创建一个 ExternalCall 的 StmtPtr 对象 *s*，调用名为 "nnc_aten_embedding" 的外部函数，
  // 传递参数为 w 和 indices，无额外属性
  StmtPtr s =
      ExternalCall::make(ResultBuf, "nnc_aten_embedding", {w, indices}, {});

  // 返回一个 Tensor 对象，其底层数据为 ResultBuf.node()，关联的语句为 s
  return Tensor(ResultBuf.node(), s);
}

// 结束命名空间 "tensorexpr"
} // namespace tensorexpr

// 结束命名空间 "jit"
} // namespace jit

// 结束命名空间 "torch"
} // namespace torch
```