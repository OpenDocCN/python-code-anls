# `.\pytorch\torch\csrc\jit\codegen\fuser\codegen.cpp`

```py
// 引入 Torch JIT 中用于代码生成的相关头文件
#include <torch/csrc/jit/codegen/fuser/codegen.h>

// 引入 ATen 库中的相关组件
#include <ATen/ATen.h>
#include <ATen/code_template.h>
#include <c10/util/Exception.h>

// 引入 Torch JIT 中用于编译和接口的相关头文件
#include <torch/csrc/jit/codegen/fuser/compiler.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/codegen/fuser/tensor_info.h>
#include <torch/csrc/jit/ir/ir.h>

// 引入 CPU 和 CUDA 相关的资源字符串
#include <torch/csrc/jit/codegen/fuser/cpu/resource_strings.h>
#include <torch/csrc/jit/codegen/fuser/cuda/resource_strings.h>

// 引入 C++ 标准库头文件
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <tuple>
#include <vector>

// Torch JIT 命名空间
namespace torch {
namespace jit {
namespace fuser {

// 用于计算访问张量中值的偏移量的模板字符串
static auto dim_calc = at::jit::CodeTemplate(R"(
//printf("tensor ${tensor} sizes[${d}] = %d, strides[${d}] = %d\n", ${tensor}.sizes[${d}],${tensor}.strides[${d}]);
size_t ${tensor}_dimIndex${d} = ${tensor}_linearIndex ${mod_sizes};
${tensor}_offset += ${tensor}_dimIndex${d} ${times_stride};
)");

// 返回 Value 对应的名称，以其唯一标识符为后缀
static std::string valueName(const Value* n) {
  return "n" + std::to_string(n->unique());
}

// 返回整数类型的值的字符串表示
static std::string scalarValue(const int64_t v) {
  return std::to_string(v);
}

// 返回布尔类型的值的字符串表示
static std::string scalarValue(const bool v) {
  return std::to_string(v);
}

// 返回双精度浮点数类型的值的字符串表示，处理特殊值如 NAN 和 INFINITY
static std::string scalarValue(const double v) {
  std::ostringstream out;
  if (std::isnan(v)) {
    out << "NAN";
  } else if (std::isinf(v)) {
    if (v < 0) {
      out << "NEG_INFINITY";
    } else {
      out << "POS_INFINITY";
    }
  } else {
    out << std::setprecision(16) << v;
  }
  return out.str();
}

// 返回标量类型的字符串表示，针对半精度和 BFloat16 进行特殊处理
static const char* scalarTypeName(const at::ScalarType type) {
  if (type == at::ScalarType::Half) {
    return "half";
  }
  if (type == at::ScalarType::BFloat16) {
    return "__nv_bfloat16";
  }

  // 根据类型返回对应的 C++ 类型字符串
  switch (type) {
#define DEFINE_CASE(ctype, name) \
  case at::ScalarType::name:     \
    return #ctype;
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CASE)
#undef DEFINE_CASE
    default:
      throw std::runtime_error("unknown scalar type");
  }
}

// 返回计算标量类型的字符串表示，对于半精度和 BFloat16 返回 "float"
static const char* calcScalarTypeName(const at::ScalarType type) {
  if (type == at::ScalarType::Half) {
    return "float";
  }
  if (type == at::ScalarType::BFloat16) {
    return "float";
  }
  return scalarTypeName(type);
}

// 返回变量的类型字符串表示，根据 C10 类型进行区分
static std::string variableType(const c10::Type& t) {
  if (t.kind() == TypeKind::IntType) {
    return "int64_t";
  } else if (t.kind() == TypeKind::FloatType) {
    return "double";
  } else if (t.kind() == TypeKind::BoolType) {
    return "bool";
  } else if (auto scalar_type = t.expectRef<TensorType>().scalarType()) {
    return calcScalarTypeName(*scalar_type);
  }
  // 返回计算得到的标量类型名称，传入 calcScalarTypeName 函数所需的参数为 scalar_type 所指向的对象
  // 出现在 JIT 融合代码生成期间的类型分析过程中发生了错误
  throw std::runtime_error(
      "unknown scalar type during JIT fusion code generation");
  // 抛出运行时错误，指示在 JIT 融合代码生成期间遇到未知的标量类型
// 返回类型转换后的值的名称，根据输入类型 t 的种类和输出类型 outtype 以及变量名 vn 来确定
static std::string typeCastedValueName(
    const c10::Type& t,               // 输入类型 t 的引用
    const at::ScalarType outtype,     // 输出标量类型
    const std::string& vn) {          // 变量名 vn 的引用
  if (t.kind() == TypeKind::IntType || t.kind() == TypeKind::BoolType) {
    if (!isIntegralType(outtype, /*includeBool=*/false)) {
      // 如果 t 是整数类型或布尔类型，并且输出类型不是整数类型（包括布尔类型），则返回 vn 的类型转换字符串
      return std::string("((") + calcScalarTypeName(outtype) + ") " + vn + ")";
    }
    // 否则直接返回变量名 vn
    return vn;
  } else if (t.kind() == TypeKind::FloatType) {
    // 如果 t 是浮点类型，以保守的方式插入一个类型转换，以匹配输出类型
    // 注意：在我们的标量类型系统中，浮点类型和双精度没有区别，但张量标量类型中有区别。
    return std::string("((") + calcScalarTypeName(outtype) + ") " + vn + ")";
  } else if (t.kind() == TypeKind::NoneType) {
    // 对于可选参数（如内存格式）支持 None 值
    return vn;
  } else if (auto scalar_type = t.expectRef<TensorType>().scalarType()) {
    if (*scalar_type != outtype) {
      // 如果 t 是张量类型，且其标量类型不等于输出类型，则返回 vn 的类型转换字符串
      return std::string("((") + calcScalarTypeName(outtype) + ") " + vn + ")";
    }
    // 否则直接返回变量名 vn
    return vn;
  }
  // 在形状传播期间发生了类型分析错误
  throw std::runtime_error(
      "unknown scalar type during JIT fusion code generation");  // 抛出运行时错误，表示 JIT 融合代码生成过程中出现未知标量类型
}

// 编码特殊操作的右手边（RHS），例如 clamp 操作的情况
static std::string encodeSpecialRHS(const Node* n, at::jit::TemplateEnv& env) {
  // 对于 clamp 操作进行特殊处理，处理缺少的 min/max 输入的情况
  // 注意：下面的 min 和 max 在第一种情况中作为边界处理，以便在 min 或 max 是 NaN 时忽略，以及当输入是 NaN 时输出也是 NaN
  if (n->kind() == aten::clamp) {
    const auto min = n->input(1);
    const auto max = n->input(2);
    env.s("0", valueName(n->input(0)));  // 设置模板环境中的索引 0 的值为输入节点 n 的值名

    if (!min->node()->mustBeNone() && !max->node()->mustBeNone()) {
      env.s("1", valueName(min));  // 设置模板环境中的索引 1 的值为 min 节点的值名
      env.s("2", valueName(max));  // 设置模板环境中的索引 2 的值为 max 节点的值名
      // 返回 clamp 操作的字符串格式，用于生成代码，根据输入的 min/max 边界
      return format("(${0} < ${1} ? ${1} : (${0} > ${2}? ${2} : ${0}))", env);
    } else if (min->node()->mustBeNone()) {
      env.s("1", valueName(max));  // 设置模板环境中的索引 1 的值为 max 节点的值名
      // 返回 clamp 操作的字符串格式，当 min 为 None 时的处理方式
      return format("(${0} > ${1} ? ${1} : ${0})", env);
    } else if (max->node()->mustBeNone()) {
      env.s("1", valueName(min));  // 设置模板环境中的索引 1 的值为 min 节点的值名
      // 返回 clamp 操作的字符串格式，当 max 为 None 时的处理方式
      return format("(${0} < ${1} ? ${1} : ${0})", env);
    } else {
      // 如果 min 和 max 都为 None，则抛出运行时错误
      throw std::runtime_error(
          "At least one of 'min' or 'max' must not be None");
    }
  } else {
    // 不支持节点的编码右手边操作，抛出运行时错误
    throw std::runtime_error("Cannot encode RHS of the node, op not supported");
  }
}
struct RHSTemplate {
  // Common case: float and double dispatch are identical
  // 初始化 RHSTemplate 对象，float 和 double 的处理方式相同
  RHSTemplate(const char* for_float)
      : for_float(for_float), for_double(for_float) {}

  // 初始化 RHSTemplate 对象，分别指定 float 和 double 的处理方式
  RHSTemplate(const char* for_float, const char* for_double)
      : for_float(for_float), for_double(for_double) {}

  // 存储 float 处理方式的字符串
  const char* for_float;
  // 存储 double 处理方式的字符串
  const char* for_double;
};

// 写入"simple mappable"操作
return encodeSpecialRHS(n, env);
} else {
size_t i = 0;

auto outtype = n->output()->type()->expectRef<TensorType>().scalarType();
TORCH_INTERNAL_ASSERT(outtype);

for (auto in : n->inputs()) {
  // PyTorch 在应用操作符之前将（标量）参数类型转换为结果类型，例如 1.4-torch.tensor(3) = -2
  // 将输入值的类型转换为输出值的类型，并存储到环境中
  env.s(
      std::to_string(i),
      typeCastedValueName(*in->type(), *outtype, valueName(in)));
  // 仅用于比较操作符的未转换操作数
  env.s(std::to_string(i) + "_nocast", valueName(in));
  i++;
}

// 获取操作符对应的模板
const auto& templ = simple_map_ops.at(n->kind());
const char* str = nullptr;
if (*outtype == at::kFloat) {
  str = templ.for_float;
} else {
  str = templ.for_double;
}
AT_ASSERT(str);
// 格式化字符串并返回结果
return format(str, env);
}
}

static void emitIndexingFor(
std::ostream& out,
const std::string& tensor,
const int ndim,
const bool last_is_cont) {
at::jit::TemplateEnv env;
env.s("tensor", tensor);
out << format("IndexType ${tensor}_offset = 0;\n", env);
out << format("IndexType ${tensor}_linearIndex = linearIndex;\n", env);
for (int d = ndim - 1; d >= 0; --d) {
  env.d("d", d);
  env.s("mod_sizes", d > 0 ? format("% ${tensor}.sizes[${d}]", env) : "");
  env.s(
      "times_stride",
      (d < ndim - 1 || !last_is_cont)
          ? format("* ${tensor}.strides[${d}]", env)
          : "");
  out << dim_calc.format(env);
  if (d > 0) {
    out << format("${tensor}_linearIndex /= ${tensor}.sizes[${d}];\n", env);
  }
}
}

static void emitCheckFor(
std::ostream& out,
const std::string& tensor,
const int ndim,
const TensorDesc& desc) {
at::jit::TemplateEnv env;
env.s("tensor", tensor);
env.s("scalar_type", scalarTypeName(desc.scalar_type));

// 分配缓冲区以加载 4 个值
out << format("${scalar_type} ${tensor}_buf[4];\n", env);

// 检查最后一个维度是否连续
if (!desc.lastIsContiguous()) {
  out << "flag_vec4 = false;\n";
  return;
}

// 对于大于 4 字节的数据类型，禁用性能
if (at::elementSize(desc.scalar_type) > 4) {
  out << "flag_vec4 = false;\n";
  return;
}

// 最后一个维度大小是 4 的倍数，其他维度步长是 4 的倍数
for (int d = ndim - 1; d >= 0; --d) {
  env.d("d", d);
  if (d == ndim - 1) {
    // 最后一个维度的步长在编译时已经检查过
    out << format(
        "if(${tensor}.sizes[${d}] % 4 != 0) flag_vec4 = false;\n", env);
  } else {
    out << format(
        "if(${tensor}.strides[${d}] % 4 != 0) flag_vec4 = false;\n", env);
    }
  }



// 上述代码片段似乎是从一个较大的代码块中提取出来的，这里应当包含更多上下文才能正确理解其作用和用途。
// 通常情况下，这段代码可能是一个语句块的结尾，其中包含了函数定义、循环或条件语句的闭合。
// 在缺少上下文的情况下，很难准确地描述其功能和含义。



  // pointer aligned
  out << format(
      "if(((uint64_t) ${tensor}.data) % (4 * sizeof(${scalar_type})) != 0) flag_vec4 = false;\n",
      env);



// 将格式化后的条件检查语句输出到流对象 `out` 中，检查 `${tensor}` 对象的数据指针是否按照特定的对齐方式排列。
// 如果条件不满足，则将 `flag_vec4` 标志设置为 false。这通常用于检查数据的内存对齐性。
// ${tensor} 和 ${scalar_type} 是可能在运行时被实际值替换的占位符。
}

// TODO: 处理需要生成超过 2^32 元素张量的情况

// 生成内核函数，返回生成的内核代码字符串
std::string generateKernel(
    const std::string& name,  // 内核函数名称
    const Graph& graph,  // 图形对象，可能用于生成内核
    const std::vector<std::pair<const Value*, const std::optional<TensorDesc>>>& inputs,  // 输入值和可选的张量描述
    const std::vector<std::pair<const Value*, const TensorDesc>>& outputs,  // 输出值和张量描述
    const bool use_cuda) {  // 是否使用 CUDA
  at::jit::TemplateEnv env;  // 模板环境对象，用于替换模板中的变量

  env.s("kernelName", name);  // 替换模板中的 kernelName 变量为函数名称
  env.s(
      "IndexType",
      "unsigned int"); // 注意：为避免包含 cstdint，使用 unsigned int 而非 uint32_t

  std::stringstream tensorChecks;  // 字符串流，用于生成张量检查的代码
  std::stringstream body;  // 字符串流，用于生成主体代码
  std::stringstream body_vec4;  // 字符串流，用于生成向量化代码
  std::stringstream load;  // 字符串流，用于生成加载代码
  std::stringstream store;  // 字符串流，用于生成存储代码
  std::stringstream tensorOffsets;  // 字符串流，用于生成张量偏移量代码
  std::vector<std::string> formals;  // 存储形式参数字符串的向量
  std::vector<std::string> argument_loads;  // 存储参数加载字符串的向量

  // Lambda 函数，用于生成形式参数
  auto emitFormal = [&](const Value* n, const TensorDesc& desc) {
    env.d(
        "formal_index",
        formals.size() +
            1); // +1 是因为第一个参数是 linearIndex
    std::string tensor =
        "t" +
        std::to_string(
            formals.size());  // 不能使用 unique()，因为 Param 可能是输出
    const auto nDim = desc.nDim();
    emitCheckFor(tensorChecks, tensor, nDim, desc);  // 生成张量检查代码
    emitIndexingFor(tensorOffsets, tensor, nDim, desc.lastIsContiguous());  // 生成张量索引代码
    env.s("tensor", tensor);
    env.d("nDim", nDim);
    env.s("scalar_type", scalarTypeName(desc.scalar_type));
    formals.push_back(
        format("const TensorInfo<${scalar_type},${nDim}> ${tensor}", env));  // 添加到形式参数列表
    argument_loads.push_back(format(
        "*static_cast<TensorInfo<${scalar_type},${nDim}>*>(args[${formal_index}])",
        env));  // 添加到参数加载列表
  };

  // Lambda 函数，用于生成标量形式参数
  auto emitScalarFormal = [&](const Value* n) {
    env.d(
        "formal_index",
        formals.size() +
            1); // +1 是因为第一个参数是 linearIndex
    std::string scalar =
        "s" +
        std::to_string(
            formals.size());  // 不能使用 unique()，因为 Param 可能是输出
    env.s("scalar", scalar);
    env.s("scalar_type", variableType(*n->type()));
    formals.push_back(format("${scalar_type} ${scalar}", env));  // 添加到形式参数列表
    argument_loads.push_back(
        format("*static_cast<${scalar_type}*>(args[${formal_index}])", env));  // 添加到参数加载列表
  };

  // 生成输入参数
  for (const auto& input : inputs) {
    if (input.second.has_value()) {
      emitFormal(input.first, *input.second);  // 生成张量形式参数
    } else {
      emitScalarFormal(input.first);  // 生成标量形式参数
    }
  }

  // 生成输出参数
  for (const auto& output : outputs) {
    emitFormal(output.first, output.second);  // 生成张量形式参数
  }

  // 获取输入值
  bool has_half_tensor = false;  // 是否有半精度张量
  bool has_bfloat_tensor = false;  // 是否有 BFloat16 张量
  size_t formal_count = 0;  // 形式参数计数
  for (const auto& input : inputs) {
    auto p = input.first;
    env.s("node", valueName(p));  // 获取值的名称
    env.d("formal", formal_count++);

    // 获取和转换（如果需要）输入
    // 如果输入的第二个参数有值
    if (input.second.has_value()) {
      // 检查输入的第二个参数是否为半精度浮点数 (half)
      const auto is_half = input.second.has_value() &&
          ((*input.second).scalar_type == at::ScalarType::Half);
      // 检查输入的第二个参数是否为 BF16 类型
      const auto is_bfloat = input.second.has_value() &&
          ((*input.second).scalar_type == at::ScalarType::BFloat16);
      // 检查输入的第二个参数是否为布尔类型
      const auto is_bool = input.second.has_value() &&
          ((*input.second).scalar_type == at::ScalarType::Bool);
      
      // 如果是半精度浮点数 (half)
      if (is_half) {
        AT_ASSERT(use_cuda);  // 在 CUDA 上运行时进行断言
        // 生成对半精度数据访问的字符串表达式
        env.s(
            "access",
            format("__half2float(t${formal}.data[t${formal}_offset])", env));
        env.s("access_vec4", format("__half2float(t${formal}_buf[i])", env));
        has_half_tensor = true;  // 标记存在半精度张量
      } else if (is_bfloat) {
        AT_ASSERT(use_cuda);  // 在 CUDA 上运行时进行断言
        // 生成对 BF16 数据访问的字符串表达式
        env.s(
            "access",
            format(
                "__bfloat162float(t${formal}.data[t${formal}_offset])", env));
        env.s(
            "access_vec4", format("__bfloat162float(t${formal}_buf[i])", env));
        has_bfloat_tensor = true;  // 标记存在 BF16 张量
      } else if (use_cuda) {
        // 在 CUDA 上运行时
        // 对布尔类型没有 __ldg 重载
        if (is_bool) {
          env.s("access", format("t${formal}.data[t${formal}_offset]", env));
        } else {
          // 生成在 CUDA 上对数据进行 __ldg 访问的字符串表达式
          env.s(
              "access",
              format("__ldg(&t${formal}.data[t${formal}_offset])", env));
        }
        env.s("access_vec4", format("t${formal}_buf[i]", env));
      } else {
        // 在 CPU 上运行时，生成对数据的普通访问字符串表达式
        env.s("access", format("t${formal}.data[t${formal}_offset]", env));
        env.s("access_vec4", format("t${formal}_buf[i]", env));
      }
      // 计算并设置左操作数的数据类型字符串
      env.s("lhs_type", calcScalarTypeName(input.second->scalar_type));

      // 加载输入数据到向量化代码路径
      auto ele_size = at::elementSize((*input.second).scalar_type);
      // 根据数据类型大小选择合适的加载方式
      if (ele_size == 1) {
        // 加载一个字节大小的数据
        env.s(
            "load4",
            format(
                "*(reinterpret_cast<float*>(t${formal}_buf)) = *(reinterpret_cast<float*>(t${formal}.data + t${formal}_offset))",
                env));
      } else if (ele_size == 2) {
        // 加载两个字节大小的数据
        env.s(
            "load4",
            format(
                "*(reinterpret_cast<float2*>(t${formal}_buf)) = *(reinterpret_cast<float2*>(t${formal}.data + t${formal}_offset))",
                env));
      } else if (ele_size == 4) {
        // 加载四个字节大小的数据
        env.s(
            "load4",
            format(
                "*(reinterpret_cast<float4*>(t${formal}_buf)) = *(reinterpret_cast<float4*>(t${formal}.data + t${formal}_offset))",
                env));
      } else {
        // 对于其他大小的数据，使用循环加载
        env.s(
            "load4",
            format(
                "for(int i = 0; i<4; i++) t${formal}_buf[i] = t${formal}.data[t${formal}_offset + i]",
                env));
      }
      // 将加载的代码添加到 load 字符串流中
      load << format("${load4};\n", env);
  } else {
    // 设置 'access' 和 'access_vec4' 变量，用于获取表达式结果
    env.s("access", format("s${formal}", env));
    env.s("access_vec4", format("s${formal}", env));
    // 设置 'lhs_type' 变量，用于存储左值表达式的类型
    env.s("lhs_type", variableType(*input.first->type()));
  }
  // 将生成的代码追加到 'body' 和 'body_vec4' 中
  body << format("${lhs_type} ${node} = ${access};\n", env);
  body_vec4 << format("${lhs_type} ${node} = ${access_vec4};\n", env);
}

bool has_random = false;
// 为中间节点生成代码
// 注意：Concat 和 Chunk 节点会隐式生成
// 注意：仅支持 CUDA 内核的随机数生成
// 注意：常量 None 节点会被忽略，我们会在使用常量 None 节点的地方处理它
// 注意：不需要遍历引用，因为 n 是一个指针
for (const auto n : graph.nodes()) {
  static_assert(std::is_pointer<decltype(n)>::value, "n must be a pointer");
  // 对于 FusedConcat 节点，跳过处理
  if (n->kind() == prim::FusedConcat)
    continue;
  // 对于 ConstantChunk 节点，跳过处理
  if (n->kind() == prim::ConstantChunk)
    continue;
  // 如果节点必须为 None，则跳过处理
  if (n->mustBeNone())
    continue;
  // 对于 rand_like 节点，如果使用 CUDA，设置 has_random 标志为 true
  if (n->kind() == aten::rand_like) {
    AT_ASSERT(use_cuda);
    has_random = true;
  }
  // 对于 prim::Constant 节点，总是生成 double 类型的 RHS 表达式
  // 这将根据张量标量运算类型规则或数学函数规则后续进行窄化
  if (n->kind() == prim::Constant) {
    const auto val = toIValue(n->output()).value();
    std::string rhs;
    // 根据值的类型设置 RHS
    if (val.isDouble()) {
      rhs = scalarValue(val.toDouble());
    } else if (val.isBool()) {
      rhs = scalarValue(val.toBool());
    } else {
      AT_ASSERT(val.isInt());
      rhs = scalarValue(val.toInt());
    }
    // 设置 'node'、'rhs' 和 'lhs_type' 变量
    env.s("node", valueName(n->output()));
    env.s("rhs", rhs);
    env.s("lhs_type", variableType(*n->output()->type()));
  } else {
    // 对于其他类型的节点，设置 'node'、'rhs' 和 'lhs_type' 变量
    env.s("node", valueName(n->output()));
    env.s("rhs", encodeRHS(n));
    env.s("lhs_type", variableType(*n->output()->type()));
  }
  // 将生成的代码追加到 'body' 和 'body_vec4' 中
  body << format("${lhs_type} ${node} = ${rhs};\n", env);
  body_vec4 << format("${lhs_type} ${node} = ${rhs};\n", env);
}

// 为输出张量生成写入代码
for (const auto& output : outputs) {
  // 设置 'formal' 变量并格式化 'access' 和 'access_vec4' 变量
  env.d("formal", formal_count++);
  env.s("access", format("t${formal}.data[t${formal}_offset]", env));
  env.s("access_vec4", format("t${formal}_buf[i]", env));
  env.s("node", valueName(output.first));

  // 获取和转换（如果需要）输出
  // 注意：仅支持 CUDA 内核的 half 类型转换
  const auto is_half = (output.second.scalar_type == at::ScalarType::Half);
  const auto is_bfloat =
      (output.second.scalar_type == at::ScalarType::BFloat16);
  if (is_half) {
    AT_ASSERT(use_cuda);
    body << format("${access} = __float2half(${node});\n", env);
    body_vec4 << format("${access_vec4} = __float2half(${node});\n", env);
    has_half_tensor = true;
    } else if (is_bfloat) {
      // 如果是 bfloat16 类型，需要确保在 CUDA 环境下运行
      AT_ASSERT(use_cuda);
      // 将节点转换为 bfloat16 类型，并存储在代码体中
      body << format("${access} = __float2bfloat16(${node});\n", env);
      // 将节点转换为 bfloat16 类型的向量，并存储在向量化代码体中
      body_vec4 << format("${access_vec4} = __float2bfloat16(${node});\n", env);
      // 标记存在 bfloat16 张量
      has_bfloat_tensor = true;
    } else {
      // 否则直接将节点存储在代码体中
      body << format("${access} = ${node};\n", env);
      // 否则直接将节点存储在向量化代码体中
      body_vec4 << format("${access_vec4} = ${node};\n", env);
    }

    // 在向量化代码路径中存储输出
    auto ele_size = at::elementSize(output.second.scalar_type);
    if (ele_size == 1) {
      // 如果元素大小为 1 字节，使用 float* 进行存储
      env.s(
          "store4",
          format(
              "*(reinterpret_cast<float*>(t${formal}.data + t${formal}_offset)) = *(reinterpret_cast<float*>(t${formal}_buf))",
              env));
    } else if (ele_size == 2) {
      // 如果元素大小为 2 字节，使用 float2* 进行存储
      env.s(
          "store4",
          format(
              "*(reinterpret_cast<float2*>(t${formal}.data + t${formal}_offset)) = *(reinterpret_cast<float2*>(t${formal}_buf))",
              env));
    } else if (ele_size == 4) {
      // 如果元素大小为 4 字节，使用 float4* 进行存储
      env.s(
          "store4",
          format(
              "*(reinterpret_cast<float4*>(t${formal}.data + t${formal}_offset)) = *(reinterpret_cast<float4*>(t${formal}_buf))",
              env));
    } else {
      // 否则使用循环存储每个元素
      env.s(
          "store4",
          format(
              "for(int i = 0; i<4; i++) t${formal}.data[t${formal}_offset + i] = t${formal}_buf[i]",
              env));
    }
    // 将存储操作加入存储代码体中
    store << format("${store4};\n", env);
  }

  // 包含所需的头文件
  // 注意：CUDA 内核支持 half 类型和随机生成，CPU 内核不支持
  if (has_half_tensor) {
    env.s("HalfHeader", cuda::half_support_literal);
  } else {
    env.s("HalfHeader", "");
  }
  if (has_bfloat_tensor) {
    env.s("BFloat16Header", cuda::bfloat16_support_literal);
  } else {
    env.s("BFloat16Header", "");
  }

  // 如果需要随机数，添加随机数相关的头文件和初始化代码
  if (has_random) {
    env.s("RandHeader", cuda::rand_support_literal);
    env.s("RandParam", cuda::rand_param);
    env.s("RandInit", cuda::rand_init);
  } else {
    env.s("RandHeader", "");
    env.s("RandParam", "");
    env.s("RandInit", "");
  }

  // 结束 clang-format

  // 实例化 CUDA 或 CPU 特定的模板
  env.s("tensorOffsets", tensorOffsets.str());
  env.s("tensorChecks", tensorChecks.str());
  env.s("kernelBody", body.str());
  env.s("kernelBody_vec4", body_vec4.str());
  env.s("kernelLoad", load.str());
  env.s("kernelStore", store.str());
  env.v("formals", formals);
  env.v("argument_loads", argument_loads);
  std::string code_string;
  // 根据 use_cuda 变量选择 CUDA 或 CPU 的模板生成代码字符串
  if (use_cuda) {
    env.s("type_declarations", cuda::type_declarations_template.format(env));
    code_string = cuda::cuda_compilation_unit_template.format(env);
  } else {
    env.s("type_declarations", cpu::type_declarations_template.format(env));
    code_string = cpu::cpu_compilation_unit_template.format(env);
  }

  // 如果启用了调试模式，输出融合代码到标准错误流
  if (debugFuser()) {
    std::cerr << "fusion code:" << code_string << std::endl;
  }
  // 返回生成的代码字符串
  return code_string;
}

// 结束 torch 命名空间
} // namespace torch

// 结束 jit 命名空间
} // namespace jit

// 结束 fuser 命名空间
} // namespace fuser
```