# `.\pytorch\torch\csrc\jit\tensorexpr\codegen.h`

```py
#pragma once
// 包含 ATen 库的头文件
#include <ATen/ATen.h>
// 包含 TensorExpr 中的 IR 和 Tensor 头文件
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

// 引入 C++ 标准库中的实用工具
#include <utility>

// Torch 的命名空间：jit -> tensorexpr
namespace torch {
namespace jit {
namespace tensorexpr {

// 声明模板类 PaddedBuffer
template <typename T>
class PaddedBuffer;

// 定义 CodeGen 类，属于 Torch API
class TORCH_API CodeGen {
 public:
  // 嵌套类 BufferArg 和 CallArg 的前向声明
  class BufferArg;
  class CallArg;

  // 构造函数模板，接受 StmtPtr 和一系列参数
  template <typename... Ts>
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  CodeGen(StmtPtr stmt, Ts... ts)
      : stmt_(std::move(stmt)), buffer_args_({BufferArg(ts)...}) {}

  // 另一个构造函数，接受 StmtPtr、BufferArg 向量、设备类型和内核函数名
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  CodeGen(
      StmtPtr stmt,
      std::vector<BufferArg> buffer_args,
      at::Device device = at::kCPU,
      std::string kernel_func_name = "func");

  // 虚析构函数，默认
  virtual ~CodeGen() = default;

  // 返回成员变量 stmt_
  StmtPtr stmt() const {
    return stmt_;
  }

  // 设置成员变量 stmt_
  void set_stmt(StmtPtr s) {
    stmt_ = s;
  }

  // 应用 IRMutator 对象进行变换
  void apply_mutator(IRMutator* mutator) {
    stmt_ = stmt_->accept_mutator(mutator);
  }

  // 应用 IRVisitor 对象进行访问
  void apply_visitor(IRVisitor* visitor) {
    stmt_->accept(visitor);
  }

  // 返回 buffer_args_ 成员变量的引用
  std::vector<BufferArg>& buffer_args() {
    return buffer_args_;
  }

  // 返回 buffer_args_ 成员变量的常量引用
  const std::vector<BufferArg>& buffer_args() const {
    return buffer_args_;
  }

  // 返回设备类型
  at::Device device() {
    return device_;
  }

  // 返回生成的代码作为字符串的函数，需要子类实现
  virtual std::string getCodeText(const std::string& attr = "") {
    return ("");
  }

  // TODO: Figure out how to unify these call interfaces.

  /// 调用一个函数，参数是 CallArg 向量，这些向量能正确类型化参数。
  virtual void call(const std::vector<CallArg>& args) = 0;

  /// 通过假设生成的内核已知参数类型，快速调用一个函数。
  virtual void call_raw(const std::vector<void*>& args) = 0;

  /// 更快地调用一个函数，通过假设线程块数可以从 numel 推导出来的方式。
  virtual void call_with_numel(void** args, int64_t numel);

  // 创建一个空的张量，支持给定的 size、stride、dtype、layout、device 和 pin_memory 选项
  virtual at::Tensor empty_strided(
      c10::IntArrayRef size,
      c10::IntArrayRef stride,
      std::optional<c10::ScalarType> dtype_opt,
      std::optional<c10::Layout> layout_opt,
      std::optional<c10::Device> device_opt,
      std::optional<bool> pin_memory_opt) {
    return at::empty_strided(
        size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
  }

  // 返回内核函数名的引用
  const std::string& kernel_func_name() const {
    return kernel_func_name_;
  }

  // 分配中间缓冲区
  void allocIntermediateBufs();

 protected:
  // 将 BufferArg 和 CallArg 转换为指针的静态方法
  static void* argToPtr(const BufferArg& bufferArg, const CallArg& callArg);

 private:
  StmtPtr stmt_;                     // IR 语句的指针
  std::vector<BufferArg> buffer_args_;  // BufferArg 向量
  at::Device device_ = at::kCPU;      // 设备类型，默认为 CPU
  std::string kernel_func_name_ = "func";  // 内核函数名，默认为 "func"
};

// Torch 的命名空间：jit -> tensorexpr
} // namespace tensorexpr
} // namespace jit
} // namespace torch
class TORCH_API ExtCallMemoryReuse : public IRMutator {
  // 静态函数：创建外部调用函数名映射表
  static std::unordered_map<std::string, std::string> makeExtCallFuncNameMap();
  // 静态常量：外部调用函数名映射表
  static const std::unordered_map<std::string, std::string> extCallFuncNameMap_;

 public:
  // 构造函数：初始化对象，接收缓冲区参数
  explicit ExtCallMemoryReuse(
      const std::vector<CodeGen::BufferArg>& bufferArgs);
  // 虚析构函数：默认实现
  ~ExtCallMemoryReuse() override = default;
  // 重写的 mutate 方法：处理 ExternalCallPtr 类型的节点，返回 StmtPtr 类型
  StmtPtr mutate(ExternalCallPtr v) override;

 private:
  // 成员变量：缓冲区参数的无序集合
  std::unordered_set<BufPtr> bufferArgs_;
};

class CodeGen::BufferArg {
 public:
  // 构造函数：根据 Tensor 创建 BufferArg 对象
  BufferArg(const Tensor& tensor) : buf_(tensor.buf()) {}
  // 构造函数：根据 VarHandle 创建 BufferArg 对象
  BufferArg(const VarHandle& var) : var_(var.node()), isVar_(true) {}
  // 构造函数：根据 BufHandle 创建 BufferArg 对象
  BufferArg(const BufHandle& buf) : buf_(buf.node()) {}
  // 构造函数：根据 BufPtr 创建 BufferArg 对象
  BufferArg(BufPtr buf) : buf_(std::move(buf)) {}

  // 返回变量指针：如果是 Var 对象则返回 VarPtr，否则返回 Buf 对象的基础句柄
  VarPtr var() const {
    return isVar_ ? var_ : buf_->base_handle();
  }

  // 返回缓冲区指针：返回 BufPtr 对象
  BufPtr buf() const {
    return buf_;
  }

  // 是否是 Var 类型的标志：返回 isVar_ 标志
  bool isVar() const {
    return isVar_;
  }

  // 数据类型：返回变量或者缓冲区的数据类型
  Dtype dtype() const {
    return isVar_ ? var_->dtype() : buf_->dtype();
  }

 private:
  // 成员变量：VarPtr 对象，默认为 nullptr
  VarPtr var_ = nullptr;
  // 成员变量：BufPtr 对象，默认为 nullptr
  BufPtr buf_ = nullptr;
  // 成员变量：是否是 Var 类型的标志，默认为 false
  bool isVar_ = false;
};

class CodeGen::CallArg {
 public:
  // 模板构造函数：根据 PaddedBuffer 创建 CallArg 对象
  template <typename T>
  CallArg(const PaddedBuffer<T>& buffer);

  // 模板构造函数：根据 vector 创建 CallArg 对象
  template <typename T>
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,cppcoreguidelines-pro-type-const-cast)
  CallArg(const std::vector<T>& buffer)
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      : data_(const_cast<T*>(buffer.data())) {}

  // 指针构造函数：根据指针创建 CallArg 对象
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  CallArg(void* ptr) : data_(ptr) {}

#define ARG_TYPE_CTOR(Type, Name)      \
  // 构造函数：根据 Type 类型的值创建 CallArg 对象
  CallArg(Type v) {                    \
    memcpy(buffer_, &v, sizeof(Type)); \
    data_ = (void*)buffer_;            \
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, ARG_TYPE_CTOR);
#undef ARG_TYPE_CTOR

  // 返回数据指针：返回 CallArg 对象的数据指针
  void* data() const {
    return data_;
  }

  // 拷贝构造函数：根据 rhs 对象创建 CallArg 对象
  CallArg(const CallArg& rhs) {
    if (rhs.data_ == rhs.buffer_) {
      memcpy(this->buffer_, rhs.buffer_, sizeof(rhs.buffer_));
      this->data_ = (void*)(this->buffer_);
    } else {
      this->data_ = rhs.data_;
    }
  }

  // 拷贝赋值操作符：将 rhs 对象的值赋给当前对象
  CallArg& operator=(const CallArg& rhs) {
    if (rhs.data_ == rhs.buffer_) {
      memcpy(this->buffer_, rhs.buffer_, sizeof(rhs.buffer_));
      this->data_ = (void*)(this->buffer_);
    } else {
      this->data_ = rhs.data_;
    }
    return *this;
  }

#define ARG_PTR_DEFINE(Type, Name)                  \
  // 返回指针方法定义：返回指定类型的指针
  Type* Name##Ptr() const {                         \
    TORCH_INTERNAL_ASSERT(data_ == (void*)buffer_); \
    return (Type*)data_;                            \
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, ARG_PTR_DEFINE);
#undef`
#undef ARG_PTR_DEFINE

private:
void* data_;
// `data_` is a pointer used in `CallArg` to store a scalar value or, if needed, an 8-byte buffer for larger scalars.

char buffer_[8] = {0}; // Initialize an 8-byte buffer for storing larger scalars.

};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class RegisterCodeGenList {
public:
// Returns the singleton instance of `RegisterCodeGenList`
TORCH_API static RegisterCodeGenList& GetInstance() {
    static RegisterCodeGenList codegen_list;
    return codegen_list;
}

using StmtFactoryMethod = std::function<std::unique_ptr<CodeGen>(
    StmtPtr stmt,
    const std::vector<CodeGen::BufferArg>&,
    at::Device device,
    const std::string& kernel_func_name)>;

// Searches for a statement factory method by name
TORCH_API StmtFactoryMethod FindStmtFactoryMethod(const std::string& name);
RegisterCodeGenList(const RegisterCodeGenList&) = delete;
RegisterCodeGenList& operator=(const RegisterCodeGenList&) = delete;

private:
template <class CodeGenType>
friend class RegisterCodeGen;
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
RegisterCodeGenList() = default;
TORCH_API void AddStmtFactoryMethod(
    const std::string& name,
    const StmtFactoryMethod& stmt_factory_method);

std::unordered_map<std::string, StmtFactoryMethod> stmt_factory_methods_;
};

template <class CodeGenType>
class RegisterCodeGen {
public:
// Registers a code generation type with a specified name
explicit RegisterCodeGen(const std::string& name) {
    RegisterCodeGenList& codegen_list = RegisterCodeGenList::GetInstance();
    // Adds a statement factory method for the registered code generation type
    codegen_list.AddStmtFactoryMethod(
        name,
        [](StmtPtr stmt,
           const std::vector<CodeGen::BufferArg>& params,
           at::Device device,
           const std::string& kernel_func_name) {
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          // Constructs a new instance of `CodeGenType` for the given parameters
          std::unique_ptr<CodeGen> method(
              new CodeGenType(stmt, params, device, kernel_func_name));
          return method;
        });
}
};

// Creates a code generator instance based on the specified parameters
TORCH_API std::unique_ptr<CodeGen> CreateCodeGen(
    const std::string& name,
    StmtPtr stmt,
    const std::vector<CodeGen::BufferArg>& params,
    at::Device device = at::kCPU,
    const std::string& kernel_func_name = "func");

// Defines a class for expanding generic intrinsics during Intermediate Representation (IR) mutation
class TORCH_API GenericIntrinsicsExpander : public IRMutator {
protected:
// Overrides the mutation method for handling intrinsic functions
ExprPtr mutate(IntrinsicsPtr v) override;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
```