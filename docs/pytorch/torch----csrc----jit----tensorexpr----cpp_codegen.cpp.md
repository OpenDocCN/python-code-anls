# `.\pytorch\torch\csrc\jit\tensorexpr\cpp_codegen.cpp`

```py
// 包含 C++ 标准库中的头文件
#include <algorithm>
#include <type_traits>
#include <vector>

// 包含用于 TensorExpr 的头文件
#include <torch/csrc/jit/tensorexpr/cpp_codegen.h>
#include <torch/csrc/jit/tensorexpr/cpp_intrinsics.h>
#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>
#include <torch/csrc/jit/tensorexpr/types.h>

// 声明 tensorexpr 命名空间
namespace torch::jit::tensorexpr {

// 用于重写变量名的访问者类，将 Graph IR 中的变量名转换为符合 C++ 命名规范的格式
// 例如，将变量名中的 '.' 替换为 '_'
class CppVarNameRewriter : public IRVisitor {
 public:
  // 访问变量的方法
  void visit(VarPtr v) override {
    constexpr char kDot = '.';
    constexpr char kUnderscore = '_';
    // 如果变量名不包含 '.', 则直接返回
    if (v->name_hint().find(kDot) == std::string::npos) {
      return;
    }
    // 获取变量名，并将其中的 '.' 替换为 '_'
    std::string name = v->name_hint();
    std::replace(name.begin(), name.end(), kDot, kUnderscore);
    // 更新变量的名称
    v->set_name_hint(std::move(name));
  }

  // 访问缓冲区对象的方法
  void visit(BufPtr v) override {
    // 调用其基础句柄的接受方法，继续进行访问
    v->base_handle()->accept(this);
  }
};

// 声明外部函数的字符串表示形式，以便后续生成 C++ 代码时使用
static std::string declareExternalFunction(const std::string& func_name) {
  return "void " + func_name +
      "("
      "int64_t bufs_num, "
      "void** buf_data, "
      "int64_t* buf_ranks, "
      "int64_t* buf_dims, "
      "int8_t* buf_dtypes, "
      "int64_t args_num, "
      "int64_t* extra_args);";
}

// CppPrinter 类的构造函数实现，继承自 IRPrinter，用于生成 C++ 代码
CppPrinter::CppPrinter(std::ostream* os) : IRPrinter(*os), lane_(0) {}

// CppPrinter 类的析构函数实现，默认
CppPrinter::~CppPrinter() = default;

// 打印 C++ 代码的起始部分，包含必要的头文件和宏定义
void CppPrinter::printPrologue() {
  os() << "#include <cassert>" << std::endl;
  os() << "#include <cmath>" << std::endl;
  os() << "#include <algorithm>" << std::endl;
  os() << "#include <type_traits>" << std::endl;
  os() << std::endl;

  os() << "#define POS_INFINITY INFINITY" << std::endl;
  os() << "#define NEG_INFINITY -INFINITY" << std::endl;
  os() << std::endl;

  os() << cpp_intrinsics_definition << std::endl;
  os() << std::endl;

  os() << "namespace torch {" << std::endl;
  os() << "namespace jit {" << std::endl;
  os() << "namespace tensorexpr {" << std::endl;
  // 遍历注册表中的所有外部函数，并生成其声明
  for (auto const& it : getNNCFunctionRegistry()) {
    os() << declareExternalFunction(it.first) << std::endl;
  }
  os() << "} // namespace tensorexpr" << std::endl;
  os() << "} // namespace jit" << std::endl;
  os() << "} // namespace torch" << std::endl;
  os() << std::endl;

  os() << "using namespace torch::jit::tensorexpr;" << std::endl;
  os() << std::endl;
}

// 模板函数，用于根据类型 T 生成对应的取模操作的 C++ 代码
template <typename T>
inline typename std::enable_if<!std::is_floating_point<T>::value, void>::type
visit_mod(std::ostream& os, const ExprPtr lhs, const ExprPtr rhs) {
  // 输出整数类型的取模运算
  os << *lhs << " % " << *rhs;
}

// 模板函数的重载版本，用于生成浮点数类型的取模操作的 C++ 代码
template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, void>::type
visit_mod(std::ostream& os, const ExprPtr lhs, const ExprPtr rhs) {
  // 输出浮点数类型的取模运算
  os << "std::fmod(" << *lhs << ", " << *rhs << ")";
}

// 另一个模板函数，根据类型生成不同类型的 C++ 代码
template <typename T>
inline typename std::enable_if<
    std::is_floating_point<T>::value || std::is_integral<T>::value,
    void>::type
void CppPrinter::visit(RampPtr v) {
  // 访问 RAMP 表达式，生成对应的 C++ 代码
  visit(alloc<Add>(v->base(), alloc<Mul>(alloc<IntImm>(lane_), v->stride())));
}

void CppPrinter::visit(BroadcastPtr v) {
  // 访问 BROADCAST 表达式，继续访问其值
  v->value()->accept(this);
}

void CppPrinter::visit(ModPtr v) {
  // 访问 MOD 表达式，分派 MOD 的打印操作
  dispatch_binary_op(os(), v.get());
}

void CppPrinter::visit(MaxPtr v) {
  // 访问 MAX 表达式，分派 MAX 的打印操作
  dispatch_binary_op(os(), v.get());
}

void CppPrinter::visit(MinPtr v) {
  // 访问 MIN 表达式，分派 MIN 的打印操作
  dispatch_binary_op(os(), v.get());
}

void CppPrinter::visit(CompareSelectPtr v) {
  // 访问 COMPARE_SELECT 表达式，生成对应的 C++ 三元操作符表达式
  os() << "((" << *v->lhs() << " " << IRPrinter::to_string(v->compare_select_op()) << " " << *v->rhs()
       << ") ? " << *v->ret_val1() << " : " << *v->ret_val2() << ")";
}

void CppPrinter::visit(IfThenElsePtr v) {
  // 访问 IF-THEN-ELSE 表达式，生成对应的 C++ 三元条件表达式
  os() << "((" << *v->condition() << ") ? " << *v->true_value() << " : " << *v->false_value() << ")";
}

void CppPrinter::visit(AllocatePtr v) {
  // 访问 ALLOCATE 表达式，计算分配的内存大小
  size_t size = v->dtype().byte_size();
  for (const auto& dim : v->dims()) {
    IntImmPtr d = to<IntImm>(dim);
    if (d) {
      size *= d->value();  // 如果维度是整数常量，计算总大小
    } else {
      throw std::runtime_error("Only IntImm dimensions are supported for now");  // 否则抛出错误
    }
  }
    }
  }



  // 结束内部作用域，对应前面的开放花括号
  emitIndent();
  // 调用emitIndent函数，输出当前的缩进空格
  os() << v->dtype().ToCppString() << "* " << (*v->buffer_var())
       << " = static_cast<" << v->dtype().ToCppString() << "*>(malloc(" << size
       << "));" << std::endl;
  // 向输出流os中写入语句：声明指针变量，分配内存并进行类型转换，打印到标准输出流
}

void CppPrinter::visit(FreePtr v) {
  // 输出空格和缩进
  emitIndent();
  // 输出 free 函数调用语句，释放指针变量所指向的内存
  os() << "free(" << *v->buffer_var() << ");" << std::endl;
}

void CppPrinter::visit(LoadPtr v) {
  // 输出空格和缩进
  auto flat_idx =
      // 计算多维数组的平坦索引，用于访问数组元素
      flatten_index(v->buf()->dims(), v->indices(), v->buf()->strides());
  // 输出数组加载表达式
  os() << *v->base_handle() << "[" << *flat_idx << "]";
}

void CppPrinter::visit(StorePtr v) {
  // 输出空格和缩进
  auto flat_idx =
      // 计算多维数组的平坦索引，用于访问数组元素
      flatten_index(v->buf()->dims(), v->indices(), v->buf()->strides());
  // 获取值的通道数
  const int lanes = v->value()->dtype().lanes();
  // 循环遍历通道数
  for (int lane = 0; lane < lanes; lane++) {
    // 设置当前通道号
    lane_ = lane;
    // 输出空格和缩进
    emitIndent();
    // 输出数组存储语句，将值存储到数组元素中
    os() << *v->base_handle() << "[" << *flat_idx << "] = " << *v->value()
         << ";" << std::endl;
  }
}

void CppPrinter::visit(CastPtr v) {
  // 输出静态类型转换表达式
  os() << "static_cast<" << v->dtype().ToCppString() << ">(" << *v->src_value()
       << ")";
}

void CppPrinter::visit(BitCastPtr v) {
  // 输出位级类型转换表达式
  os() << "std::bitcast<" << v->src_value()->dtype().ToCppString() << ", "
       << v->dtype().ToCppString() << ">(" << *v->src_value() << ")";
}

void CppPrinter::visit(IntrinsicsPtr v) {
  // 检查特定的内部函数调用类型
  if (v->op_type() == kRand || v->op_type() == kSigmoid) {
    // 抛出运行时异常，不支持 kRand 和 kSigmoid 操作
    throw std::runtime_error("kRand and kSigmoid are not supported");
  }

  // 输出内部函数调用表达式
  os() << "std::" << v->func_name() << "(";
  // 遍历内部函数的参数
  for (int i = 0; i < v->nparams(); i++) {
    if (i > 0) {
      os() << ", ";
    }
    // 输出参数表达式
    os() << *v->param(i);
  }
  os() << ")";
}

void CppPrinter::visit(ExternalCallPtr v) {
  // 生成的代码需要链接外部函数定义在 external_functions.cpp 文件中。

  // 获取全局的 NNVM 函数注册表
  auto& func_registry = getNNCFunctionRegistry();
  // 检查是否注册了该外部函数名
  if (!func_registry.count(v->func_name())) {
    // 如果未注册，抛出未实现的降级异常
    throw unimplemented_lowering(v);
  }

  // 准备外部调用的缓冲区参数
  std::vector<BufPtr> bufs(v->buf_args());
  bufs.insert(bufs.begin(), v->buf());
  // 用于每个缓冲区的输出 lambda 函数
  auto for_buf = [&](const std::function<void(const BufPtr)>& print_buf) {
    for (size_t i = 0; i < bufs.size(); i++) {
      if (i > 0) {
        os() << ", ";
      }
      // 输出缓冲区句柄
      print_buf(bufs[i]);
    }
  };

  // 输出空格和缩进，开始外部函数调用的代码块
  emitIndent();
  os() << "{" << std::endl;
  indent_++;

  // 输出空格和缩进，声明缓冲区指针数组
  emitIndent();
  os() << "void* buf_ptrs[]{";
  // 输出每个缓冲区的基础句柄
  for_buf([&](const BufPtr b) { os() << *b->base_handle(); });
  os() << "};" << std::endl;

  // 输出空格和缩进，声明缓冲区秩数组
  emitIndent();
  os() << "int64_t buf_ranks[]{";
  // 输出每个缓冲区的秩
  for_buf([&](const BufPtr b) { os() << b->ndim(); });
  os() << "};" << std::endl;

  // 输出空格和缩进，声明缓冲区维度数组
  emitIndent();
  os() << "int64_t buf_dims[]{";
  // 输出每个缓冲区的维度
  for_buf([&](const BufPtr buf) {
    for (size_t i = 0; i < buf->ndim(); i++) {
      if (i > 0) {
        os() << ", ";
      }
      // 输出维度值
      os() << *buf->dim(i);
    }
  });
  os() << "};" << std::endl;

  // 输出空格和缩进，声明缓冲区数据类型数组
  emitIndent();
  os() << "int8_t buf_dtypes[]{";
  // 输出每个缓冲区的数据类型
  for_buf([&](const BufPtr buf) {
    os() << static_cast<int>(buf->dtype().scalar_type());
  });
  os() << "};" << std::endl;

  // 输出空格和缩进，声明额外参数数组
  emitIndent();
  os() << "int64_t extra_args[]{";
  // 输出每个额外参数
  for (size_t i = 0; i < v->args().size(); i++) {
    if (i > 0) {
      os() << ", ";
    }
    // 输出额外参数的值
    os() << v->args()[i];
  }
  os() << "};" << std::endl;
}
  os() << *v->args()[i];

将 v 对象的第 i 个参数的内容输出到 os 流中。


  os() << "};" << std::endl;

在 os 流中输出 "};"，表示一个代码块的结束，并换行。


  emitIndent();

调用 emitIndent 函数，用于在输出前插入适当数量的缩进。


  os() << v->func_name() << "(" << std::endl;

在 os 流中输出 v 对象的函数名，并开始一个函数调用的输出。


  emitIndent();
  os() << "    " << bufs.size() << "," << std::endl;

输出一个整数值（bufs.size()）及其后的逗号，用于函数参数列表，并在换行前插入适当数量的缩进。


  emitIndent();
  os() << "    buf_ptrs," << std::endl;

输出 buf_ptrs 变量的名称作为函数参数，并在换行前插入适当数量的缩进。


  emitIndent();
  os() << "    buf_ranks," << std::endl;

输出 buf_ranks 变量的名称作为函数参数，并在换行前插入适当数量的缩进。


  emitIndent();
  os() << "    buf_dims," << std::endl;

输出 buf_dims 变量的名称作为函数参数，并在换行前插入适当数量的缩进。


  emitIndent();
  os() << "    buf_dtypes," << std::endl;

输出 buf_dtypes 变量的名称作为函数参数，并在换行前插入适当数量的缩进。


  emitIndent();
  os() << "    " << v->args().size() << "," << std::endl;

输出 v 对象的参数列表的大小，并在换行前插入适当数量的缩进。


  emitIndent();
  os() << "    extra_args);" << std::endl;

输出 extra_args 变量作为函数参数，并在换行前插入适当数量的缩进。


  indent_--;
  emitIndent();

减少当前缩进级别，并调用 emitIndent 函数来输出适当数量的缩进。


  os() << "}" << std::endl;

在 os 流中输出 "}"，表示一个函数的结束，并换行。
}

void CppPrinter::visit(LetPtr v) {
  // 如果变量的数据类型的 lanes 为 1
  if (v->var()->dtype().lanes() == 1) {
    // 输出当前缩进
    emitIndent();
    // 输出变量的 C++ 字符串表示和变量名，以及赋值语句
    os() << v->var()->dtype().ToCppString() << " " << *v->var() << " = "
         << *v->value() << ";" << std::endl;
  } else {
    // 将变量添加到 vector_vars_ 映射中
    vector_vars_[v->var()] = v->value();
  }
}

void CppPrinter::visit(VarPtr v) {
  // 如果变量的数据类型的 lanes 为 1
  if (v->dtype().lanes() == 1) {
    // 输出变量的唯一名称
    os() << name_manager()->get_unique_name(v);
  } else {
    // 输出变量在 vector_vars_ 映射中的值
    os() << *vector_vars_.at(v);
  }
}

CppCodeGen::CppCodeGen(
    StmtPtr stmt,
    const std::vector<BufferArg>& buffer_args,
    at::Device device,
    const std::string& kernel_func_name)
    : CodeGen(stmt, buffer_args, device, kernel_func_name) {
  // 初始化 CodeGen 基类
  init();
}

void CppCodeGen::init() {
  // 创建 CppPrinter 对象，使用 oss_ 输出流
  printer_ = std::make_unique<CppPrinter>(&oss_);
  // 创建 CppVarNameRewriter 对象
  var_name_rewriter_ = std::make_unique<CppVarNameRewriter>();

  // 应用变量名称重写器
  apply_visitor(var_name_rewriter_.get());

  // 打印函数的起始部分
  printer_->printPrologue();
  // 输出函数声明部分
  os() << "void " << kernel_func_name() << "(";
  const std::vector<BufferArg> buffer_args = this->buffer_args();
  for (size_t i = 0; i < buffer_args.size(); i++) {
    if (i > 0) {
      os() << ", ";
    }
    const BufferArg& buffer_arg = buffer_args[i];
    const VarPtr var = buffer_arg.var();
    Dtype dtype = buffer_arg.dtype();
    // 输出参数的 C++ 字符串表示和参数名
    os() << dtype.ToCppString() << (buffer_arg.isVar() ? " " : "* ") << *var;
  }
  os() << ")";
  // 接受代码树的访问并打印到 printer_ 的输出流中
  stmt()->accept(printer_.get());
  os() << std::endl;
}

CppCodeGen::~CppCodeGen() = default;

void CppCodeGen::call(const std::vector<CallArg>& args) {
  // TODO: 编译生成的 C++ 内核代码成为一个库，并在这里调用该库。
  os() << "int main() {}" << std::endl;
}

void CppCodeGen::call_raw(const std::vector<void*>& args) {
  // TODO: 编译生成的 C++ 内核代码成为一个库，并在这里调用该库。
  os() << "int main() {}" << std::endl;
}

RegisterCodeGen<CppCodeGen> cpp_codegen_reg("cpp_codegen");

} // namespace torch::jit::tensorexpr
```