# `.\pytorch\aten\src\ATen\core\library.cpp`

```py
// 包含 Torch 库的头文件
#include <torch/library.h>

// 包含 ATen 核心调度器的头文件
#include <ATen/core/dispatch/Dispatcher.h>

// Torch 命名空间
namespace torch {

// 匿名命名空间，定义辅助函数和常量

// 根据文件和行号生成调试信息字符串，用于错误处理
// 如果定义了 STRIP_ERROR_MESSAGES 宏，则返回空字符串
std::string debugString(const char* file, uint32_t line) {
#ifdef STRIP_ERROR_MESSAGES
  return std::string();
#else
  return c10::str("registered at ", file, ":", line);
#endif
}

// 根据自定义的调试信息和文件/行号生成调试信息字符串
// 如果调试信息为空，则使用文件和行号生成调试信息
std::string debugString(std::string debug, const char* file, uint32_t line) {
#ifdef STRIP_ERROR_MESSAGES
  return std::string();
#else
  if (debug.empty()) {
    return debugString(file, line);
  } else {
    return debug;
  }
#endif
}

// 将 Library::Kind 枚举转换为字符串表示
const char* toString(Library::Kind kind) {
  switch (kind) {
    case Library::DEF:
      return "TORCH_LIBRARY";
    case Library::IMPL:
      return "TORCH_LIBRARY_IMPL";
    case Library::FRAGMENT:
      return "TORCH_LIBRARY_FRAGMENT";
  }
  return "(unknown)";
}

// 默认使用 CatchAll 调度键
constexpr auto CatchAll = c10::DispatchKey::CatchAll;
} // anonymous namespace

// 构造函数：CppFunction 对象的构造函数
CppFunction::CppFunction(c10::KernelFunction func, std::optional<c10::impl::CppSignature> cpp_signature, std::unique_ptr<c10::FunctionSchema> schema)
  : func_(std::move(func))
  , cpp_signature_(cpp_signature)
  , schema_(std::move(schema))
  , debug_()
  {}

// 析构函数：CppFunction 对象的析构函数
CppFunction::~CppFunction() = default;

// 重置注册器，清空所有注册信息
void Library::reset() {
  registrars_.clear();
}

// 宏定义：错误上下文信息
#define ERROR_CONTEXT "(Error occurred while processing ", toString(kind_), " block at ", file_, ":", line_, ")"

// 构造函数：Library 对象的构造函数，根据类型和命名空间注册库
Library::Library(Kind kind, std::string ns, std::optional<c10::DispatchKey> k, const char* file, uint32_t line)
  : kind_(kind)
  , ns_(ns == "_" ? c10::nullopt : c10::make_optional(std::move(ns)))
  , dispatch_key_(k.value_or(CatchAll) == CatchAll ? std::optional<c10::DispatchKey>() : k)
  , file_(file)
  , line_(line)
  {
    switch (kind_) {
      case DEF:
        // 只有 DEF 类型的库需要保证唯一性；片段不需要注册库
        registrars_.emplace_back(
          c10::Dispatcher::singleton().registerLibrary(
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
            *ns_, debugString(file_, line_)
          )
        );
        [[fallthrough]]; // 继续执行下一个 case
      case FRAGMENT:
        // 检查命名空间是否有值，不能使用通配符命名空间 _
        TORCH_CHECK(
          ns_.has_value(),
          toString(kind_), ": cannot define ", toString(kind_), " with the wildcard namespace _ "
          "(every ", toString(kind_), " defines operators for a distinct namespace!) "
          "Did you mean to use TORCH_LIBRARY_IMPL instead?  "
          ERROR_CONTEXT
        );
        // 确保没有指定调度键
        TORCH_INTERNAL_ASSERT(!dispatch_key_.has_value(), ERROR_CONTEXT);
        break;
      case IMPL:
        // IMPL 类型无需操作，一切正常
        break;
    }
  }

// 宏定义：DEF 类型库的前导字符串
#define DEF_PRELUDE "def(\"", schema.operator_name(), "\"): "
// 定义 Library 类的成员函数 _def，接受一个右值引用的 c10::FunctionSchema 对象，
// 一个指向 OperatorName 的指针 out_name，一个包含 at::Tag 的标签向量 tags，
// 以及一个 _RegisterOrVerify 枚举类型的参数 rv，并返回一个 Library 对象的引用
Library& Library::_def(c10::FunctionSchema&& schema, c10::OperatorName* out_name, const std::vector<at::Tag>& tags, _RegisterOrVerify rv) & {
  // 检查当前对象的 kind_ 属性是否为 DEF 或 FRAGMENT，否则抛出错误
  TORCH_CHECK(kind_ == DEF || kind_ == FRAGMENT,
    DEF_PRELUDE,
    "Cannot define an operator inside of a ", toString(kind_), " block.  "
    "All def()s should be placed in the (unique) TORCH_LIBRARY block for their namespace.  ",
    ERROR_CONTEXT
  );
  // 确保 ns_ 属性有值，否则抛出错误
  TORCH_INTERNAL_ASSERT(ns_.has_value(), ERROR_CONTEXT);
  // 确保 dispatch_key_ 属性没有值，否则抛出错误
  TORCH_INTERNAL_ASSERT(!dispatch_key_.has_value(), ERROR_CONTEXT);
  // 获取 schema 的命名空间，如果存在则检查是否与 ns_ 相同，否则设置命名空间为 ns_
  auto ns_opt = schema.getNamespace();
  if (ns_opt.has_value()) {
    // 如果指定了命名空间，则确保与当前对象的 ns_ 属性相匹配，否则抛出错误
    TORCH_CHECK(*ns_opt == *ns_,
      "Explicitly provided namespace (", *ns_opt, ") in schema string "
      "does not match namespace of enclosing ", toString(kind_), " block (", *ns_, ").  "
      "Move this definition to the (unique) TORCH_LIBRARY block corresponding to this namespace "
      "(and consider deleting the namespace from your schema string.)  ",
      ERROR_CONTEXT
    );
  } else {
    // 如果未指定命名空间，则尝试使用当前对象的 ns_ 属性设置 schema 的命名空间
    bool b = schema.setNamespaceIfNotSet(ns_->c_str());
    TORCH_INTERNAL_ASSERT(b, ERROR_CONTEXT);
  }
  // 如果 out_name 不为空指针，则将 schema 的操作符名称赋值给 out_name
  if (out_name) {
    *out_name = schema.operator_name(); // copy!
  }
  // 根据 rv 的值进行不同的操作
  switch (rv) {
    // 如果 rv 为 REGISTER，则注册 Python 模块或函数定义
    case _RegisterOrVerify::REGISTER:
      if (python_module_.has_value()) {
        // 如果 python_module_ 有值，则注册 Python 模块
        registrars_.emplace_back(
          c10::Dispatcher::singleton().registerPythonModule(
            schema.operator_name(),
            python_module_->first,
            python_module_->second)
        );
      }
      // 注册函数定义
      registrars_.emplace_back(
        c10::Dispatcher::singleton().registerDef(
          std::move(schema),
          debugString(file_, line_), // 获取当前文件和行号的调试信息
          tags // 使用给定的标签向量
        )
      );
      break;
    // 如果 rv 为 VERIFY，则等待函数定义的验证
    case _RegisterOrVerify::VERIFY:
      c10::Dispatcher::singleton().waitForDef(schema);
      break;
  }
  // 返回当前对象的引用
  return *this;
}
// 取消定义 DEF_PRELUDE 宏的定义
#undef DEF_PRELUDE

// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
// 定义 Library 类的另一个成员函数 _def，接受一个右值引用的 std::variant<c10::OperatorName, c10::FunctionSchema> 对象 name_or_schema，
// 一个右值引用的 CppFunction 对象 f，以及一个包含 at::Tag 的标签向量 tags，并返回一个 Library 对象的引用
Library& Library::_def(std::variant<c10::OperatorName, c10::FunctionSchema>&& name_or_schema, CppFunction&& f, const std::vector<at::Tag>& tags) & {
  // 创建一个 FunctionSchema 对象 schema，根据 name_or_schema 的类型来初始化
  c10::FunctionSchema schema = [&] {
    // 如果 name_or_schema 的类型是 c10::FunctionSchema，则返回其值
    if (std::holds_alternative<c10::FunctionSchema>(name_or_schema)){
      return std::get<c10::FunctionSchema>(std::move(name_or_schema));
    } else {
      // 如果 name_or_schema 推断为名称，则使用推断出的模式
      c10::OperatorName name = std::get<c10::OperatorName>(std::move(name_or_schema));
      // 检查函数的 schema 是否存在，如果不存在则报错
      TORCH_CHECK(f.schema_,
        "def(\"", name, "\"): "
        "未指定完整的 schema 字符串，并且无法推断出 schema。请显式提供 schema 字符串。",  // 错误信息
        ERROR_CONTEXT
      );
      // 使用函数的 schema 克隆出一个新的 schema，并设置别名分析为 CONSERVATIVE
      c10::FunctionSchema s = f.schema_->cloneWithName(std::move(name.name), std::move(name.overload_name));
      s.setAliasAnalysis(c10::AliasAnalysisKind::CONSERVATIVE);
      return s;  // 返回新的函数 schema
    }
  }();
  // 创建一个空的 OperatorName 对象，用于实现调用的命名空间名称
  c10::OperatorName name("", "");
  // 首先定义 schema...
  _def(std::move(schema), &name, tags);
  // 然后注册实现...
  auto dispatch_key = f.dispatch_key_.has_value() ? f.dispatch_key_ : dispatch_key_;
  registrars_.emplace_back(
    // 将函数实现注册到调度器中
    c10::Dispatcher::singleton().registerImpl(
      std::move(name),
      dispatch_key,
      std::move(f.func_),
      f.cpp_signature_,
      std::move(f.schema_),
      debugString(std::move(f.debug_), file_, line_)
    )
  );
  return *this;  // 返回当前对象的引用
}

#define IMPL_PRELUDE "impl(\"", name_str, "\", ...): "
// 定义了一个名为 IMPL_PRELUDE 的宏，用于构建一些特定格式的错误信息前缀

at::OperatorName Library::_parseNameForLib(const char* name_str) const {
  // 解析给定的操作符名称字符串为 OperatorName 对象
  auto name = torch::jit::parseName(name_str);
  // 获取操作符名称的命名空间（如果有）
  auto ns_opt = name.getNamespace();
  
  // 检查是否提供了命名空间，并确保与当前对象的命名空间一致
  if (ns_opt.has_value()) {
    // 查看注释：[注册代码中的冗余是可以接受的]
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    TORCH_CHECK(*ns_opt == *ns_,
      IMPL_PRELUDE,
      "显式提供的命名空间 (", *ns_opt, ") 在操作符名称中与封闭的 ", toString(kind_), " 块的命名空间 (", *ns_, ") 不匹配。"
      "将此定义移到相应命名空间对应的 ", toString(kind_), " 块中（考虑从模式字符串中删除命名空间）。",
      ERROR_CONTEXT
    );
  } else {
    // 如果未提供命名空间，则设置为当前对象的命名空间
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    bool b = name.setNamespaceIfNotSet(ns_->c_str());
    // 内部断言，确保设置命名空间成功
    TORCH_INTERNAL_ASSERT(b, ERROR_CONTEXT);
  }
  // 返回解析后的操作符名称
  return name;
}

// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
Library& Library::_impl(const char* name_str, CppFunction&& f, _RegisterOrVerify rv) & {
  // 解析操作符名称为 OperatorName 对象
  at::OperatorName name = _parseNameForLib(name_str);
  
  // 查看注释：[注册代码中的冗余是可以接受的]
  // 检查操作符的分发键是否与当前对象的分发键一致
  TORCH_CHECK(!(f.dispatch_key_.has_value() &&
                dispatch_key_.has_value() &&
                *f.dispatch_key_ != *dispatch_key_),
    IMPL_PRELUDE,
    "显式提供的分发键 (", *f.dispatch_key_, ") 与封闭的 ", toString(kind_), " 块的分发键 (", *dispatch_key_, ") 不一致。"
    "请为此分发键声明一个单独的 ", toString(kind_), " 块，并将您的 impl() 移动到那里。"
    ERROR_CONTEXT
  );
  
  // 确定最终要使用的分发键
  auto dispatch_key = f.dispatch_key_.has_value() ? f.dispatch_key_ : dispatch_key_;
  
  // 根据注册或验证方式执行不同操作
  switch (rv) {
    case _RegisterOrVerify::REGISTER:
      // 注册操作符实现
      registrars_.emplace_back(
        c10::Dispatcher::singleton().registerImpl(
          std::move(name),
          dispatch_key,
          std::move(f.func_),
          f.cpp_signature_,
          std::move(f.schema_),
          debugString(std::move(f.debug_), file_, line_)
        )
      );
      break;
    case _RegisterOrVerify::VERIFY:
      // 等待操作符实现
      c10::Dispatcher::singleton().waitForImpl(name, dispatch_key);
      break;
  }
  
  // 返回当前对象的引用
  return *this;
}

c10::OperatorName Library::_resolve(const char* name_str) const {
  // 解析操作符名称为 OperatorName 对象，并返回
  return _parseNameForLib(name_str);
}

#undef IMPL_PRELUDE

// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
Library& Library::_fallback(CppFunction&& f) & {
  // 检查当前对象的类型是否为 IMPL，因为只有在 IMPL 类型中才能定义操作符
  TORCH_CHECK(kind_ == IMPL,
    "fallback(...): 无法在 ", toString(kind_), " 块内部定义操作符。"
    "您是否打算在 TORCH_LIBRARY_IMPL 块内调用此函数？",
  ERROR_CONTEXT);
  // 获取函数对象的调度键，如果未指定则使用默认的调度键
  auto dispatch_key = f.dispatch_key_.has_value() ? f.dispatch_key_ : dispatch_key_;
  // 断言调度键已经存在，如果不存在则抛出异常
  TORCH_INTERNAL_ASSERT(dispatch_key.has_value(), ERROR_CONTEXT);
  // 检查命名空间是否未指定，如果指定了则抛出异常，因为不支持单独命名空间的回退函数
  TORCH_CHECK(!ns_.has_value(),
    "fallback(...): Fallback functions which apply to only a single namespace ",
    "(you specified ", *ns_, ") are not supported.  If you intended to apply ",
    "this fallback function globally, please define a separate block:\n\n",
    "    TORCH_LIBRARY_IMPL(_, ", *dispatch_key, ", m) { m.fallback(...); }\n\n",
    ERROR_CONTEXT);
  // 如果调度键为 DispatchKey::Undefined，这里会忽略它，因为 Undefined 不是运行时的键，不能向其注册任何内容。
  // 遍历运行时调度键集合中的每个调度键
  for (auto k : c10::getRuntimeDispatchKeySet(*dispatch_key)) {
    // 对于移动平台不使用所有的调度键，因此跳过未使用的键的回退函数注册
    auto idx = getDispatchTableIndexForDispatchKey(k);
    if (idx < 0) continue;
    // 向注册器列表中添加注册的回退函数
    registrars_.emplace_back(
      c10::Dispatcher::singleton().registerFallback(
        k,
        f.func_,
        debugString(f.debug_, file_, line_)
      )
    );
  }
  // 返回当前对象的引用
  return *this;
}
} // namespace torch
```