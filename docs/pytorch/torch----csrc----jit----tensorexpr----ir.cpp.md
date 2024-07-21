# `.\pytorch\torch\csrc\jit\tensorexpr\ir.cpp`

```py
// 引入头文件声明用于 Torch 的张量表达式和语句
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>

// 引入 C++ 标准库和 Torch 提供的实用工具
#include <c10/util/irange.h>

// 引入实用工具库中的标准库
#include <utility>

// 定义 Torch 的张量表达式命名空间
namespace torch::jit::tensorexpr {

// 静态函数，根据缓冲区和索引数据类型选择数据类型
static Dtype ChooseDtype(const Dtype& buffer_dtype, const Dtype& index_dtype) {
  return Dtype(buffer_dtype, index_dtype.lanes());
}

// 函数，返回给定索引表达式列表中第一个表达式的数据类型
static Dtype dtypeOfIndices(const std::vector<ExprPtr>& indices) {
  if (indices.empty()) {
    // 如果索引列表为空，返回一个标量整数类型
    return kInt;
  }
  return indices.at(0)->dtype();
}

// 函数，将给定索引列表中的所有表达式转换为整数类型（Int 或 Long）
static void castIndicesToInts(std::vector<ExprPtr>& indices) {
  // 默认将索引类型设置为 Int
  auto index_dtype = ScalarType::Int;
  for (auto& index : indices) {
    if (index->dtype().scalar_type() == ScalarType::Long) {
      // 如果有任何一个索引是 Long 类型，则将所有索引转换为 Long
      index_dtype = ScalarType::Long;
      break;
    }
  }

  // 对所有索引进行类型转换，确保与 index_dtype 匹配
  for (auto& index : indices) {
    const Dtype& dt = index->dtype();
    if (c10::isIntegralType(dt.scalar_type(), true) &&
        dt.scalar_type() != index_dtype) {
      index = alloc<Cast>(Dtype(index_dtype, dt.lanes()), index);
    }
  }
}

// Load 类的构造函数，用于加载数据
Load::Load(Dtype dtype, BufPtr buf, std::vector<ExprPtr> indices)
    : ExprNodeBase(dtype), buf_(std::move(buf)), indices_(std::move(indices)) {
  castIndicesToInts(indices_);
}

// Load 类的构造函数，根据给定的缓冲区和索引列表构造加载操作
Load::Load(BufPtr buf, const std::vector<ExprPtr>& indices)
    : Load(ChooseDtype(buf->dtype(), dtypeOfIndices(indices)), buf, indices) {}

// Load 类的静态工厂方法，用于创建加载操作表达式
ExprHandle Load::make(
    Dtype dtype,
    const BufHandle& buf,
    const std::vector<ExprHandle>& indices) {
  return ExprHandle(
      alloc<Load>(dtype, buf.node(), ExprHandleVectorToExprVector(indices)));
}

// Load 类的静态工厂方法，用于创建加载操作表达式（根据缓冲区和索引列表）
ExprHandle Load::make(
    const BufHandle& buf,
    const std::vector<ExprHandle>& indices) {
  return Load::make(buf.dtype(), buf, indices);
}

// Store 类的构造函数，用于存储数据
Store::Store(BufPtr buf, std::vector<ExprPtr> indices, ExprPtr value)
    : buf_(std::move(buf)),
      indices_(std::move(indices)),
      value_(std::move(value)) {
  castIndicesToInts(indices_);
}

// Store 类的静态工厂方法，用于创建存储操作
StorePtr Store::make(
    const BufHandle& buf,
    const std::vector<ExprHandle>& indices,
    const ExprHandle& value) {
  return alloc<Store>(
      buf.node(), ExprHandleVectorToExprVector(indices), value.node());
}

// BufHandle 类的方法，用于创建存储操作
StorePtr BufHandle::store(
    const std::vector<ExprHandle>& args,
    const ExprHandle& value) const {
  return Store::make(*this, args, value);
}

// 函数，用于展开索引
ExprPtr flatten_index(
    const std::vector<ExprPtr>& dims,
    const std::vector<ExprPtr>& indices,
    const std::vector<ExprPtr>& strides) {
  // 首先处理已经展开的索引情况
  if (indices.size() == 1) {
    return indices[0];
  }

  // 检查维度、索引和步长的数量是否一致
  size_t ndim = dims.size();
  if (ndim != indices.size()) {
    throw malformed_input("dimensions mismatch in flatten_index");
  }
  if (ndim != strides.size()) {
    throw malformed_input("strides mismatch in flatten_index");
  }
  if (ndim == 0) {
    // 如果维度为零，返回一个长整数常量0
    return alloc<LongImm>(0);
  }

  // 计算总索引，根据第一个索引的类型创建一个相似的表达式
  ExprPtr total_index = immLike(indices[0], 0);
  for (const auto i : c10::irange(ndim)) {
    # 计算累加索引的总和
    total_index = alloc<Add>(total_index, alloc<Mul>(indices[i], strides[i]));
    # 使用分配器（alloc）执行加法操作，将当前索引乘以对应步长后的结果与总索引相加
    
    
    
    # 返回最终累加的总索引
    return total_index;
}

# 定义一个名为 IntrinsicsDtype 的方法，用于确定操作的数据类型
Dtype Intrinsics::IntrinsicsDtype(IntrinsicsOp op_type, Dtype dt1) {
  # 如果操作类型是 kIsNan，则返回一个具有 Int 标量类型的 dt1 的克隆
  if (op_type == kIsNan) {
    return dt1.cloneWithScalarType(ScalarType::Int);
  }
  # TODO: 检查 op_type 并做出实际决定
  return dt1;
}

# 定义另一个名为 IntrinsicsDtype 的方法，用于确定操作的数据类型
Dtype Intrinsics::IntrinsicsDtype(IntrinsicsOp op_type, Dtype dt1, Dtype dt2) {
  # TODO: 检查 op_type 并做出实际决定
  return dt1;
}

# 定义另一个名为 IntrinsicsDtype 的方法，接受操作类型和参数表达式的向量，用于确定操作的数据类型
Dtype Intrinsics::IntrinsicsDtype(
    IntrinsicsOp op_type,
    const std::vector<ExprPtr>& params) {
  # TODO: 检查 op_type 并做出实际决定
  # 如果参数为空，则抛出异常 malformed_input
  if (params.empty()) {
    throw malformed_input("invalid params in Intrinsics");
  } else if (params.size() == 1) {
    # 如果参数只有一个，则返回第一个参数表达式的数据类型
    return IntrinsicsDtype(op_type, params[0]->dtype());
  } else if (params.size() == 2) {
    # 如果参数有两个，则返回两个参数表达式的数据类型
    return IntrinsicsDtype(op_type, params[0]->dtype(), params[1]->dtype());
  }
  # 默认返回第一个参数表达式的数据类型
  return params[0]->dtype();
}

# 定义一个名为 OpArgCount 的方法，用于返回特定操作类型的参数个数
int Intrinsics::OpArgCount(IntrinsicsOp op_type) {
  # 根据操作类型使用 switch 分支判断
  switch (op_type) {
    # 以下操作类型只有一个参数
    case kSin:
    case kCos:
    case kTan:
    case kAsin:
    case kAcos:
    case kAtan:
    case kSinh:
    case kCosh:
    case kTanh:
    case kSigmoid:
    case kExp:
    case kExpm1:
    case kAbs:
    case kLog:
    case kLog2:
    case kLog10:
    case kLog1p:
    case kErf:
    case kErfc:
    case kSqrt:
    case kRsqrt:
    case kCeil:
    case kFloor:
    case kRound:
    case kTrunc:
    case kFrac:
    case kLgamma:
    case kIsNan:
      return 1;
    # kRand 操作类型没有参数
    case kRand:
      return 0;
    # 以下操作类型有两个参数
    case kAtan2:
    case kFmod:
    case kPow:
    case kRemainder:
      return 2;
    # 默认情况下抛出运行时错误，显示无效的操作类型
    default:
      throw std::runtime_error("invalid op_type: " + std::to_string(op_type));
  }
}

# 定义一个名为 ExternalCall 的静态方法 make，用于创建外部调用对象
ExternalCallPtr ExternalCall::make(
    BufHandle buf,
    const std::string& func_name,
    const std::vector<BufHandle>& buf_args,
    const std::vector<ExprHandle>& args) {
  # 创建用于存储缓冲区节点的空间，并填充相应的缓冲区句柄
  std::vector<BufPtr> buf_arg_nodes;
  buf_arg_nodes.reserve(buf_args.size());
  for (const BufHandle& buf_arg : buf_args) {
    buf_arg_nodes.push_back(buf_arg.node());
  }
  # 使用分配器创建 ExternalCall 对象，返回其指针
  return alloc<ExternalCall>(
      buf.node(), func_name, buf_arg_nodes, ExprHandleVectorToExprVector(args));
}

# 定义一个名为 ExternalCallWithAlloc 的静态方法 make，用于创建带分配的外部调用对象
ExternalCallWithAllocPtr ExternalCallWithAlloc::make(
    const std::string& func_name,
    const std::vector<BufHandle>& buf_out_args,
    const std::vector<BufHandle>& buf_args,
    const std::vector<ExprHandle>& args) {
  # 创建用于存储输出缓冲区节点的空间，并填充相应的缓冲区句柄
  std::vector<BufPtr> buf_out_arg_nodes;
  buf_out_arg_nodes.reserve(buf_out_args.size());
  for (const BufHandle& buf_out_arg : buf_out_args) {
    buf_out_arg_nodes.push_back(buf_out_arg.node());
  }

  # 创建用于存储输入缓冲区节点的空间，并填充相应的缓冲区句柄
  std::vector<BufPtr> buf_arg_nodes;
  buf_arg_nodes.reserve(buf_args.size());
  for (const BufHandle& buf_arg : buf_args) {
    buf_arg_nodes.push_back(buf_arg.node());
  }
  # 使用分配器创建 ExternalCallWithAlloc 对象，返回其指针
  return alloc<ExternalCallWithAlloc>(
      func_name,
      buf_out_arg_nodes,
      buf_arg_nodes,
      ExprHandleVectorToExprVector(args));
}
// 创建一个 FreeExt 对象的静态方法，接受一个 BufHandle 类型的向量 bufs 作为参数
FreeExtPtr FreeExt::make(const std::vector<BufHandle>& bufs) {
  // 创建一个空的 BufPtr 类型向量 buf_nodes，并预留 bufs 大小的空间
  std::vector<BufPtr> buf_nodes;
  buf_nodes.reserve(bufs.size());
  // 遍历 bufs 向量中的每个 BufHandle 对象 buf
  for (const BufHandle& buf : bufs) {
    // 将 buf.node() 的结果添加到 buf_nodes 中
    buf_nodes.push_back(buf.node());
  }
  // 调用 alloc 函数创建并返回一个 FreeExtPtr 对象，传入 buf_nodes 作为参数
  return alloc<FreeExt>(buf_nodes);
}

// 将 ExprHandle 类型的向量 v 转换为 ExprPtr 类型的向量
std::vector<ExprPtr> ExprHandleVectorToExprVector(
    const std::vector<ExprHandle>& v) {
  // 创建一个大小与 v 相同的 ExprPtr 类型向量 result
  std::vector<ExprPtr> result(v.size());
  // 使用 c10::irange(v.size()) 遍历索引 i
  for (const auto i : c10::irange(v.size())) {
    // 将 v[i].node() 的结果赋值给 result[i]
    result[i] = v[i].node();
  }
  // 返回转换后的 ExprPtr 向量 result
  return result;
}

// 将 ExprPtr 类型的向量 v 转换为 ExprHandle 类型的向量
std::vector<ExprHandle> ExprVectorToExprHandleVector(
    const std::vector<ExprPtr>& v) {
  // 创建一个大小与 v 相同的 ExprHandle 类型向量 result
  std::vector<ExprHandle> result(v.size());
  // 使用 c10::irange(v.size()) 遍历索引 i
  for (const auto i : c10::irange(v.size())) {
    // 使用 v[i] 构造一个新的 ExprHandle 对象，并赋值给 result[i]
    result[i] = ExprHandle(v[i]);
  }
  // 返回转换后的 ExprHandle 向量 result
  return result;
}

// 将 VarHandle 类型的向量 v 转换为 VarPtr 类型的向量
std::vector<VarPtr> VarHandleVectorToVarVector(
    const std::vector<VarHandle>& v) {
  // 创建一个大小与 v 相同的 VarPtr 类型向量 result
  std::vector<VarPtr> result(v.size());
  // 使用 c10::irange(v.size()) 遍历索引 i
  for (const auto i : c10::irange(v.size())) {
    // 将 v[i].node() 的结果赋值给 result[i]
    result[i] = v[i].node();
  }
  // 返回转换后的 VarPtr 向量 result
  return result;
}

// 将 VarPtr 类型的向量 v 转换为 VarHandle 类型的向量
std::vector<VarHandle> VarVectorToVarHandleVector(
    const std::vector<VarPtr>& v) {
  // 创建一个大小与 v 相同的 VarHandle 类型向量 result
  std::vector<VarHandle> result(v.size());
  // 使用 c10::irange(v.size()) 遍历索引 i
  for (const auto i : c10::irange(v.size())) {
    // 使用 v[i] 构造一个新的 VarHandle 对象，并赋值给 result[i]
    result[i] = VarHandle(v[i]);
  }
  // 返回转换后的 VarHandle 向量 result
  return result;
}

// 检查 ExprPtr 类型的对象 e 是否为负值的立即数
bool immediateIsNegative(const ExprPtr& e) {
#define TYPE_CASE(Type, Name)                \
  if (Name##ImmPtr imm = to<Name##Imm>(e)) { \
    return imm->value() < 0;                 \
  }
  // 对所有标量类型（除 Bool、Half、BFloat16 外）进行循环处理，检查是否匹配立即数类型
  AT_FORALL_SCALAR_TYPES_AND2(Half, BFloat16, TYPE_CASE);
#undef TYPE_CASE
  // 如果没有匹配的类型，返回 false
  return false;
}

// 检查 ExprPtr 类型的对象 e 是否为正值的立即数
bool immediateIsPositive(const ExprPtr& e) {
#define TYPE_CASE(Type, Name)                \
  if (Name##ImmPtr imm = to<Name##Imm>(e)) { \
    return imm->value() > 0;                 \
  }
  // 对所有标量类型（除 Bool、Half、BFloat16 外）进行循环处理，检查是否匹配立即数类型
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TYPE_CASE);
#undef TYPE_CASE
  // 如果没有匹配的类型，返回 false
  return false;
}

// 检查 ExprPtr 类型的对象 e 是否为零值的立即数
bool immediateIsZero(const ExprPtr& e) {
#define TYPE_CASE(Type, Name)                \
  if (Name##ImmPtr imm = to<Name##Imm>(e)) { \
    return imm->value() == 0;                \
  }
  // 对所有标量类型（除 Bool、Half、BFloat16 外）进行循环处理，检查是否匹配立即数类型
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TYPE_CASE);
#undef TYPE_CASE
  // 如果没有匹配的类型，返回 false
  return false;
}

} // namespace torch::jit::tensorexpr
```