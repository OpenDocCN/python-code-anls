# `.\pytorch\torch\csrc\jit\serialization\import_source.cpp`

```
#include <torch/csrc/jit/serialization/import_source.h>

#include <ATen/core/ivalue_inl.h>
#include <ATen/core/qualified_name.h>
#include <torch/csrc/jit/frontend/parser.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/frontend/script_type_parser.h>
#include <torch/custom_class.h>

#include <regex>

namespace torch::jit {

// 表示操作值，继承自 SugaredValue 类
struct OpsValue : public SugaredValue {
  // 构造函数，初始化版本号
  OpsValue(size_t version) : version_(version) {}
  // 返回类型名称
  std::string kind() const override {
    return "ops";
  }
  // 获取属性的 SugaredValue
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field) override {
    // 返回内置模块 BuiltinModule 的实例
    return std::make_shared<BuiltinModule>(field, version_);
  }
  size_t version_;  // 操作值的版本号
};

// 表示嵌套命名空间，如 `foo.bar.Baz`
// 目前这些命名空间只能包含其他命名空间或 NamedTypes
struct TORCH_API ClassNamespaceValue : public SugaredValue {
  /**
   * @param  name  完全限定的路径，可以解析为命名空间或 NamedType
   * @param  si    搜索和加载类/函数的源导入器
   */
  explicit ClassNamespaceValue(
      c10::QualifiedName name,
      std::shared_ptr<SourceImporterImpl> si)
      : basename_(std::move(name)), si_(std::move(si)) {}

  // 获取属性的 SugaredValue
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& name) override;

  // 返回类型名称
  std::string kind() const override {
    return "Class Namespace";
  }

 private:
  c10::QualifiedName basename_;              // 基本名称
  std::shared_ptr<SourceImporterImpl> si_;   // 源导入器
};

// 此值将属性 CONSTANTS.c0 CONSTANTS.c1 映射到 'constants' 向量中的条目
// 这个表将以容器格式存储，并在恢复代码时提供给 import_method
struct ConstantTableValue : public SugaredValue {
  // 构造函数，初始化常量表
  explicit ConstantTableValue(const std::vector<at::IValue>* constants)
      : constants_(constants) {}

  // 返回类型名称
  std::string kind() const override {
    return "CONSTANTS";
  }

  // 获取属性的 SugaredValue
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field) override {
    const char* field_s = field.c_str();
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    char* end;
    int64_t offset = strtoll(field_s + 1, &end, 10);  // 解析常量索引
    if (field.size() < 2 || *end != 0)
      throw ErrorReport(loc) << "invalid constant specifier: " << field;
    if (offset < 0 || size_t(offset) >= constants_->size()) {
      throw ErrorReport(loc) << "constant index " << offset
                             << " is out of bounds (constant table has "
                             << constants_->size() << " entries)";
    }
    auto ivalue = constants_->at(offset);  // 获取常量值
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Value* value;

    // see [Constant Object Weak CompilationUnit Reference]
    // 检查 ivalue 是否为对象并且不是弱引用的编译参考
    if (ivalue.isObject() && !ivalue.toObject()->is_weak_compilation_ref()) {
      // 获取对象指针
      auto obj = ivalue.toObject();
      // 如果缓存中没有这个对象，则将其复制为弱引用的编译参考并加入缓存
      if (!non_holding_object_cache.count(obj)) {
        non_holding_object_cache[obj] = obj->copy_to_weak_compilation_ref();
      }
      // 将对象的常量值插入到图中的常量中
      value = m.graph()->insertConstant(non_holding_object_cache[obj], loc);
    } else {
      // 否则，将常量值插入到图中的常量中
      value = m.graph()->insertConstant(constants_->at(offset), loc);
    }

    // 在编译中特化张量类型会影响类型关系
    // 重新设置值的类型为未形状化的类型
    value->setType(unshapedType(value->type()));

    // 返回一个包含 value 的 SimpleValue 的共享指针
    return std::make_shared<SimpleValue>(value);
  }

 private:
  // 存储非持有对象的缓存，键为对象的指针，值为对象的指针
  std::unordered_map<
      c10::intrusive_ptr<at::ivalue::Object>,
      c10::intrusive_ptr<at::ivalue::Object>>
      non_holding_object_cache;
  // 指向常量的指针向量
  const std::vector<at::IValue>* constants_;
};

// SourceImporterImpl 类的构造函数，接收编译单元指针 cu，常量表指针 constant_table，源加载器 source_loader，和版本号 version 作为参数
SourceImporterImpl::SourceImporterImpl(
    std::shared_ptr<CompilationUnit> cu,
    const std::vector<at::IValue>* constant_table,
    SourceLoader source_loader,
    size_t version)
    : cu_(std::move(cu)),  // 初始化 cu_ 成员变量
      source_loader_(std::move(source_loader)),  // 初始化 source_loader_ 成员变量
      version_(version) {  // 初始化 version_ 成员变量

  // 初始化 env_ 成员变量，这是一个包含名称到值的映射
  env_ = {
      {"torch", std::make_shared<BuiltinModule>("aten", version)},
      {"ops", std::make_shared<OpsValue>(version)},
      // Constants present in the model. Used to resolve "CONSTANTS.n" to the
      // actual value
      {"CONSTANTS", std::make_shared<ConstantTableValue>(constant_table)},
      {"fork", SpecialFormValue::create(prim::fork)},
      {"awaitable", SpecialFormValue::create(prim::awaitable)},
      {"annotate", SpecialFormValue::create(prim::annotate)},
      {"unchecked_cast", SpecialFormValue::create(prim::unchecked_cast)},
      {"uninitialized", SpecialFormValue::create(prim::Uninitialized)},
  };
}

// 根据名称查找命名类型的方法
TypePtr SourceImporterImpl::findNamedType(const QualifiedName& name) {
  // 如果是自定义类，直接返回对应的 TypePtr
  if (auto custom_class = getCustomClass(name.qualifiedName())) {
    return custom_class;
  }
  // 如果需要解析源码，则解析相应的源码
  parseSourceIfNeeded(name.prefix());
  // 查找待定义列表中是否存在对应名称的类定义，如果是类定义，则导入并返回对应的 TypePtr
  auto it = to_be_defined_.find(name);
  if (it != to_be_defined_.end() && it->second->kind() == TK_CLASS_DEF) {
    ClassDef cd(std::move(it->second));
    to_be_defined_.erase(it);
    importNamedType(name.prefix(), cd);
  }
  // 返回编译单元 cu_ 中对应名称的 TypePtr
  return cu_->get_type(name);
}

// 根据名称查找函数的方法
Function* SourceImporterImpl::findFunction(const QualifiedName& name) {
  // 如果需要解析源码，则解析相应的源码
  parseSourceIfNeeded(name.prefix());
  // 查找待定义列表中是否存在对应名称的函数定义，如果是函数定义，则导入并返回对应的 Function 指针
  auto it = to_be_defined_.find(name);
  if (it != to_be_defined_.end() && it->second->kind() == TK_DEF) {
    Def d(it->second);
    to_be_defined_.erase(it);
    importFunction(name.prefix(), d);
  }
  // 返回编译单元 cu_ 中对应名称的 Function 指针
  return cu_->find_function(name);
}

// 如果需要的话，解析特定名称的源码
void SourceImporterImpl::parseSourceIfNeeded(const std::string& qualifier) {
  // 如果 qualifier 是空的，或者已经加载过对应源码，则直接返回
  if (qualifier.empty() || loaded_sources_.count(qualifier)) {
    return;
  }
  // 将 qualifier 加入已加载源码的集合中
  loaded_sources_.insert(qualifier);
  // 使用 source_loader_ 加载 qualifier 对应的源码
  std::shared_ptr<Source> src = source_loader_(qualifier);

  // 如果无法获取到源码，则直接返回
  if (!src) {
    return;
  }
  // 使用 Parser 解析源码
  Parser p(src);
  parsePossibleVersionNumber(p.lexer());

  auto& L = p.lexer();

  // 循环解析源码中的 import 语句和其他内容
  while (L.cur().kind != TK_EOF) {
    parseImports(L);
    auto tk = L.cur();
    auto kind = tk.kind;
    // 根据 kind 的值进行不同的操作分支
    switch (kind) {
      // 如果 kind 是 TK_CLASS_DEF 类型
      case TK_CLASS_DEF: {
        // 解析类定义并创建 ClassDef 对象
        auto parsed_treeref = ClassDef(p.parseClass());
        // 将解析得到的类定义对象存储到待定义列表中，使用限定名作为键
        to_be_defined_[QualifiedName(qualifier, parsed_treeref.name().name())] =
            parsed_treeref;
      } break;
      // 如果 kind 是 TK_DEF 类型
      case TK_DEF: {
        // 解析函数定义并创建 Def 对象，这里 is_method 参数为 false
        auto parsed_treeref = Def(p.parseFunction(/*is_method=*/false));
        // 将解析得到的函数定义对象存储到待定义列表中，使用限定名作为键
        to_be_defined_[QualifiedName(qualifier, parsed_treeref.name().name())] =
            parsed_treeref;
      } break;
      // 如果 kind 不是 TK_CLASS_DEF 或 TK_DEF，抛出异常
      default:
        throw ErrorReport(L.cur().range)
            << "Unexpected token in code import: " << kindToString(kind);
    }
}

void SourceImporterImpl::LEGACY_import_methods(
    const Module& mod,
    const std::shared_ptr<Source>& src) {
  auto self = SimpleSelf(mod.type());  // 创建一个 SimpleSelf 对象，使用模块类型
  c10::QualifiedName prefix = *mod.type()->name();  // 获取模块类型的限定名作为前缀
  Parser p(src);  // 使用给定的源文件创建解析器对象

  parsePossibleVersionNumber(p.lexer());  // 解析可能的版本号，从解析器的词法分析器获取信息
  parseImports(p.lexer());  // 解析导入语句，从解析器的词法分析器获取信息

  std::vector<Def> definitions;  // 创建空的函数定义向量
  std::vector<ResolverPtr> resolvers;  // 创建空的解析器指针向量
  while (p.lexer().cur().kind != TK_EOF) {  // 当当前词法单元不是文件结束符时循环
    auto def = Def(p.parseFunction(/*is_method=*/true));  // 解析一个方法函数并创建其定义
    definitions.emplace_back(def);  // 将解析得到的函数定义添加到定义向量中
    resolvers.emplace_back(shared_from_this());  // 将当前对象的共享指针添加到解析器指针向量中
  }
  cu_->define(
      prefix,
      /*properties=*/{},
      /*propResolvers=*/{},
      definitions,
      resolvers,
      &self);  // 使用解析得到的定义和解析器定义模块
}

std::shared_ptr<SugaredValue> SourceImporterImpl::resolveValue(
    const std::string& name,
    GraphFunction& m,
    const SourceRange& loc) {
  auto it = env_.find(name);  // 在环境中查找给定名称的值
  if (it != env_.end()) {  // 如果找到了
    return it->second;  // 返回环境中对应的值
  }
  auto graph = m.graph();  // 获取图形函数的图对象
  if (name == "inf") {  // 如果名称是 "inf"
    return std::make_shared<SimpleValue>(  // 返回表示正无穷大的简单值对象
        graph->insertConstant(std::numeric_limits<double>::infinity(), loc));
  }
  if (name == "nan") {  // 如果名称是 "nan"
    return std::make_shared<SimpleValue>(  // 返回表示 NaN 的简单值对象
        graph->insertConstant(std::numeric_limits<double>::quiet_NaN(), loc));
  }
  if (name == "infj") {  // 如果名称是 "infj"
    return std::make_shared<SimpleValue>(  // 返回表示复数正无穷大的简单值对象
        graph->insertConstant(
            c10::complex<double>(0, std::numeric_limits<double>::infinity()),
            loc));
  }
  if (name == "nanj") {  // 如果名称是 "nanj"
    return std::make_shared<SimpleValue>(  // 返回表示复数 NaN 的简单值对象
        graph->insertConstant(
            c10::complex<double>(0, std::numeric_limits<double>::quiet_NaN()),
            loc));
  }
  if (name == "__torch__") {  // 如果名称是 "__torch__"
    return std::make_shared<ClassNamespaceValue>(  // 返回表示 Torch 类命名空间的类命名空间值对象
        c10::QualifiedName(name), shared_from_this());
  }
  return nullptr;  // 如果名称未能解析，返回空指针
}

TypePtr SourceImporterImpl::resolveType(
    const std::string& name,
    const SourceRange& loc) {
  return findNamedType(QualifiedName(name));  // 返回给定名称的命名类型
}

void SourceImporterImpl::importFunction(
    const std::string& qualifier,
    const Def& def) {
  std::vector<Def> definitions{def};  // 创建包含给定函数定义的定义向量
  std::vector<ResolverPtr> resolvers{shared_from_this()};  // 创建包含当前对象共享指针的解析器指针向量
  cu_->define(
      qualifier,
      /*properties=*/{},
      /*propResolvers=*/{},
      definitions,
      resolvers,
      nullptr);  // 使用给定的限定符和定义导入函数
}

void SourceImporterImpl::importNamedType(
    const std::string& qualifier,
    const ClassDef& class_def) {
  const auto qualified_name =
      QualifiedName(QualifiedName(qualifier), class_def.name().name());  // 构造完全限定名
  if (!class_def.superclass().present()) {  // 如果没有超类
    return importClass(qualified_name, class_def, /*is_module=*/false);  // 导入普通类
  }
  const auto& superclass_name = Var(class_def.superclass().get()).name().name();  // 获取超类的名称
  if (superclass_name == "Module") {  // 如果超类是 "Module"
    importClass(qualified_name, class_def, /*is_module=*/true);  // 导入模块类
  } else if (superclass_name == "NamedTuple") {  // 如果超类是 "NamedTuple"
    // NamedTuple 有特殊规则（因为它们是 TupleTypes 而不是 ClassTypes）
    return importNamedTuple(qualified_name, class_def);  // 导入命名元组
  } else if (superclass_name == "Interface") {
    // 处理接口类的导入逻辑
    # 如果父类名称为 qualified_name，则定义一个接口类，不是模块
    cu_->define_interface(
        qualified_name, class_def, shared_from_this(), /*is_module=*/false);
  # 如果父类名称为 "ModuleInterface"，则定义一个接口类，是模块
  } else if (superclass_name == "ModuleInterface") {
    cu_->define_interface(
        qualified_name, class_def, shared_from_this(), /*is_module=*/true);
  # 如果父类名称为 "Enum"，则导入枚举类
  } else if (superclass_name == "Enum") {
    importEnum(qualified_name, class_def);
  # 如果父类名称不属于以上任何一种情况，抛出错误，表明 Torchscript 不支持类继承
  } else {
    throw ErrorReport(class_def.range())
        << "Torchscript does not support class inheritance.";
  }
// 结束 SourceImporterImpl 类的 attributeAssignmentSpecialHandlingHack 方法的实现

std::optional<Assign> SourceImporterImpl::
    attributeAssignmentSpecialHandlingHack(
        const QualifiedName& qualified_classname,
        const Assign& assign) {

  // 定义用于替换属性类型的描述结构体
  struct AttrTypeReplacementDescr {
    std::string attr_name;      // 属性名称
    std::string expected_type;  // 预期的类型
    std::string replacement_type;  // 替换的类型
  };

  // 静态映射，将模块解缠后的限定名称映射到替换描述结构体
  static std::unordered_map<std::string, AttrTypeReplacementDescr> replacements{
      {"__torch__.torch.ao.nn.quantized.modules.linear.LinearPackedParams",
       {"_packed_params",
        "Tensor",
        "__torch__.torch.classes.quantized.LinearPackedParamsBase"}},
      {"__torch__.torch.ao.nn.quantized.modules.linear.Linear",
       {"_packed_params",
        "Tensor",
        "__torch__.torch.classes.quantized.LinearPackedParamsBase"}},
      {"__torch__.torch.ao.nn.quantized.dynamic.modules.linear.Linear",
       {"_packed_params",
        "Tensor",
        "__torch__.torch.classes.quantized.LinearPackedParamsBase"}},
      {"__torch__.torch.ao.nn.quantized.modules.conv.Conv2d",
       {"_packed_params",
        "Tensor",
        "__torch__.torch.classes.quantized.Conv2dPackedParamsBase"}},
      {"__torch__.torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d",
       {"_packed_params",
        "Tensor",
        "__torch__.torch.classes.quantized.Conv2dPackedParamsBase"}},
      {"__torch__.torch.ao.nn.quantized.modules.conv.Conv3d",
       {"_packed_params",
        "Tensor",
        "__torch__.torch.classes.quantized.Conv3dPackedParamsBase"}},
      {"__torch__.torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU3d",
       {"_packed_params",
        "Tensor",
        "__torch__.torch.classes.quantized.Conv3dPackedParamsBase"}},
      // BC Stuff
      {"__torch__.torch.nn.quantized.modules.linear.LinearPackedParams",
       {"_packed_params",
        "Tensor",
        "__torch__.torch.classes.quantized.LinearPackedParamsBase"}},
      {"__torch__.torch.nn.quantized.modules.linear.Linear",
       {"_packed_params",
        "Tensor",
        "__torch__.torch.classes.quantized.LinearPackedParamsBase"}},
      {"__torch__.torch.nn.quantized.modules.conv.Conv2d",
       {"_packed_params",
        "Tensor",
        "__torch__.torch.classes.quantized.Conv2dPackedParamsBase"}},
      {"__torch__.torch.nn.quantized.modules.conv.Conv3d",
       {"_packed_params",
        "Tensor",
        "__torch__.torch.classes.quantized.Conv3dPackedParamsBase"}},
      {"__torch__.torch.nn.quantized.dynamic.modules.linear.Linear",
       {"_packed_params",
        "Tensor",
        "__torch__.torch.classes.quantized.LinearPackedParamsBase"}}};

  // 忽略 lint 检查 facebook-hte-StdRegexIsAwful，用于处理 mangle 的正则表达式
  static std::regex mangle_re("\\.___torch_mangle_\\d+");
  
  // 替换限定类名中的 mangle 部分，得到解缠后的类名
  auto demangled_classname =
      std::regex_replace(qualified_classname.qualifiedName(), mangle_re, "");

  // 检查 demangled_classname 是否在替换映射中
  if (replacements.count(demangled_classname)) {
    // 获取 assign 的左操作数
    auto lhs = Var(assign.lhs());
    // 检查是否赋值类型存在且为变量类型，若不符合条件则返回空值
    if (!assign.type().present() || assign.type().get().kind() != TK_VAR) {
      return c10::nullopt;
    }
    // 将赋值类型转换为变量类型
    auto type = Var(assign.type().get());

    // 获取替换信息中的属性名、期望类型和替换类型的引用
    auto& attr_name = replacements.at(demangled_classname).attr_name;
    auto& expected_type = replacements.at(demangled_classname).expected_type;
    auto& replacement_type =
        replacements.at(demangled_classname).replacement_type;
    
    // 检查左值名称是否等于目标属性名，检查类型名称是否等于期望类型
    if (lhs.name().name() == attr_name && type.name().name() == expected_type) {
      // 使用替换类型创建解析器对象
      Parser p(std::make_shared<Source>(replacement_type));
      // 解析替换类型的表达式
      auto typename_expr = p.parseExp();
      // 创建可能为空的表达式对象
      auto maybe_typename =
          Maybe<Expr>::create(typename_expr.range(), typename_expr);
      // 创建包含替换类型信息的赋值表达式对象
      return Assign::create(
          assign.range(), assign.lhs_list(), assign.rhs(), maybe_typename);
    }
  }
  // 若未找到匹配的替换条件，则返回空值
  return c10::nullopt;
  // 导入指定类到模块中的实现函数
  //
  // 对于旧版 TorchBind 类，进行兼容性处理：
  // 之前我们将 TorchBind 类序列化为具有方法的实际类，
  // 这些方法委托给 torch.ops.* 命名空间中的函数。
  // 现在我们改为直接依赖这些类存在于二进制文件中，
  // 并根据内存中的 ClassType 发出相关代码。
  //
  // TODO: 一旦生产模型中不再使用旧版 TorchBind 代码，删除此部分
  {
    // 检查是否为 TorchBind 类
    static QualifiedName torch_classes_qualname("__torch__.torch.classes");
    if (torch_classes_qualname.isPrefixOf(qualified_classname)) {
      return;  // 如果是 TorchBind 类，直接返回
    }
  }
  // 创建 ClassType 对象，表示指定类的类型信息
  auto class_type = ClassType::create(
      c10::QualifiedName(qualified_classname), cu_, is_module);

  // 初始化各种数据结构用于存储方法、钩子、属性等信息
  std::vector<Def> methods;
  std::vector<ResolverPtr> method_resolvers;
  std::map<std::string, Def> pre_hook_def_map;
  std::map<std::string, Def> hook_def_map;
  std::map<std::string, ResolverPtr> pre_hook_resolver_map;
  std::map<std::string, ResolverPtr> hook_resolver_map;
  std::vector<Assign> attributes;
  std::vector<Assign> constants;

  // 存储参数、缓冲区、预处理钩子、钩子的名称集合
  std::unordered_set<std::string> parameter_names;
  std::unordered_set<std::string> buffer_names;
  std::unordered_set<std::string> pre_hook_names;
  std::unordered_set<std::string> hook_names;

  // 用于跟踪预处理钩子和钩子的原始顺序，以防止多次调用
  std::vector<std::string> pre_hooks_order;
  std::vector<std::string> hooks_order;

  // 遍历类定义的主体，将语句分成属性和方法定义
  for (const auto& statement : class_def.body()) {
    // 待实现：处理每一个语句
  }

  // 使用 ScriptTypeParser 对象解析类型
  ScriptTypeParser type_parser(shared_from_this());

  // 遍历属性列表，将其添加到类类型中
  for (const auto& assign : attributes) {
    switch (assign.lhs().kind()) {
      case TK_VAR: {
        // 处理变量类型的属性
        const auto name = Var(assign.lhs()).name().name();
        TORCH_INTERNAL_ASSERT(name != "__parameters__");
        const auto type = assign.type().present()
            ? type_parser.parseTypeFromExpr(assign.type().get())
            : type_parser.parseTypeFromExpr(assign.rhs().get());
        const bool is_parameter = parameter_names.count(name);
        const bool is_buffer = buffer_names.count(name);
        // 将属性添加到类类型中
        class_type->addAttribute(name, type, is_parameter, is_buffer);
      } break;
      case TK_SUBSCRIPT: {
        // 处理下标类型的属性
        const auto name =
            StringLiteral(Subscript(assign.lhs()).subscript_exprs()[0]).text();
        const auto type = assign.type().present()
            ? type_parser.parseTypeFromExpr(assign.type().get())
            : type_parser.parseTypeFromExpr(assign.rhs().get());
        const bool is_parameter = parameter_names.count(name);
        const bool is_buffer = buffer_names.count(name);
        // 将属性添加到类类型中
        class_type->addAttribute(name, type, is_parameter, is_buffer);
      }
    }
  }
  }
}

// Populate class constants
// 通过常量列表设置类常量
for (const auto& assign : constants) {
  // 解析常量并添加到类类型中
  auto const_val = type_parser.parseClassConstant(assign);
  // 获取常量名
  const auto name = Var(assign.lhs()).name().name();
  // 将常量添加到类类型中
  class_type->addConstant(name, const_val);
}

// build pre hook and hook def/resolver pairs
// 构建前置钩子和钩子定义/解析器对
// 这些对在ir_emitter.cpp的CompilationUnit::define_hooks()中进行去重处理
// 此处的顺序是钩子的调用顺序
std::vector<Def> hooks;
std::vector<ResolverPtr> hook_resolvers;
for (const std::string& hook_name : hooks_order) {
  // 将钩子定义添加到hooks列表中
  hooks.emplace_back(hook_def_map.find(hook_name)->second);
  // 将钩子解析器添加到hook_resolvers列表中
  hook_resolvers.push_back(hook_resolver_map.find(hook_name)->second);
}
std::vector<Def> pre_hooks;
std::vector<ResolverPtr> pre_hook_resolvers;
for (const std::string& pre_hook_name : pre_hooks_order) {
  // 将前置钩子定义添加到pre_hooks列表中
  pre_hooks.emplace_back(pre_hook_def_map.find(pre_hook_name)->second);
  // 将前置钩子解析器添加到pre_hook_resolvers列表中
  pre_hook_resolvers.push_back(
      pre_hook_resolver_map.find(pre_hook_name)->second);
}

// 注册类类型到编译单元
cu_->register_type(class_type);
const auto self = SimpleSelf(class_type);
// TODO (this will include the version number later)
// 定义类的方法、方法解析器以及类属性
cu_->define(
    qualified_classname,
    /*properties=*/{},
    /*propResolvers=*/{},
    methods,
    method_resolvers,
    &self,
    /*shouldMangle=*/false,
    /*operator_set_version=*/version_);
// 定义钩子函数
cu_->define_hooks(
    qualified_classname,
    hooks,
    hook_resolvers,
    pre_hooks,
    pre_hook_resolvers,
    &self);
}

void SourceImporterImpl::importEnum(
    const QualifiedName& qualified_name,
    const ClassDef& enum_def) {
  std::vector<at::EnumNameValue> names_values;  // 用于存储枚举类的名称和值对的向量

  TypePtr value_type = nullptr;  // 枚举值的类型指针，初始设为nullptr
  auto set_or_check_type = [&value_type](  // 设置或检查枚举值的类型，并捕获value_type变量
                               const TypePtr& t, const SourceRange& loc) {
    if (!value_type) {
      value_type = t;  // 如果类型为空，将其设为当前类型t
    } else if (value_type != t) {
      throw ErrorReport(loc)
          << "Enum class with varying value types are not supported.";  // 报错：不支持具有不同值类型的枚举类
    }
  };

  for (const auto& statement : enum_def.body()) {  // 遍历枚举类定义的每个语句
    if (statement.kind() != TK_ASSIGN) {  // 如果语句类型不是赋值语句
      throw ErrorReport(statement.range())
          << "Unexpected statement in Enum class body: "
             "only enum attribute definitions are currently supported.";  // 报错：枚举类体中只支持枚举属性定义
    }

    const auto assign = Assign(statement);  // 将语句转换为赋值语句对象
    const auto name = Var(assign.lhs()).name().name();  // 获取左侧变量的名称

    IValue ivalue;  // 用于存储赋值的值
    auto rhs = assign.rhs().get();  // 获取赋值语句的右侧表达式
    switch (rhs.kind()) {  // 根据右侧表达式的类型进行不同的处理
      case TK_STRINGLITERAL:  // 如果是字符串字面量
        ivalue = IValue(StringLiteral(rhs).text());  // 将字符串字面量转换为IValue存储
        set_or_check_type(StringType::get(), statement.range());  // 设置或检查枚举值类型为字符串类型
        break;
      case TK_CONST: {  // 如果是常量
        auto numeric_const = Const(rhs);
        if (numeric_const.isFloatingPoint()) {  // 如果是浮点数
          ivalue = IValue(numeric_const.asFloatingPoint());  // 将浮点数转换为IValue存储
          set_or_check_type(FloatType::get(), statement.range());  // 设置或检查枚举值类型为浮点数类型
        } else if (numeric_const.isIntegral()) {  // 如果是整数
          ivalue = IValue(numeric_const.asIntegral());  // 将整数转换为IValue存储
          set_or_check_type(IntType::get(), statement.range());  // 设置或检查枚举值类型为整数类型
        }
        break;
      }
      default:
        throw ErrorReport(rhs.range())
            << "Unsupported enum value type: " << rhs.kind()
            << ". Only Integers, Floats and Strings are supported.";  // 报错：不支持的枚举值类型
    }

    names_values.emplace_back(name, ivalue);  // 将名称和值对加入到names_values向量中
  }

  if (!value_type) {
    throw ErrorReport(enum_def.range())
        << "No enum values defined for " << qualified_name.qualifiedName();  // 报错：未定义枚举值
  }

  auto enum_type = EnumType::create(
      qualified_name, std::move(value_type), std::move(names_values), cu_);  // 创建枚举类型对象
  cu_->register_type(enum_type);  // 将枚举类型对象注册到编译单元中
}

void SourceImporterImpl::importNamedTuple(
    const QualifiedName& qualified_name,
    const ClassDef& named_tuple_def) {
  ScriptTypeParser type_parser(shared_from_this());  // 创建类型解析器对象

  std::vector<std::string> field_names;  // 存储命名元组字段名称的向量
  std::vector<TypePtr> field_types;  // 存储命名元组字段类型的向量
  std::vector<IValue> field_defaults;  // 存储命名元组字段默认值的向量

  for (const auto& statement : named_tuple_def.body()) {  // 遍历命名元组定义体中的每个语句
    if (statement.kind() != TK_ASSIGN) {  // 如果语句类型不是赋值语句
      throw ErrorReport(statement.range())
          << "Unexpected statement in NamedTuple body: "
             "only attribute annotations are currently supported.";  // 报错：命名元组体中只支持属性注解
    }
    const auto assign = Assign(statement);  // 将语句转换为赋值语句对象

    auto name = Var(Assign(statement).lhs()).name().name();  // 获取左侧变量的名称

    std::optional<IValue> default_val;  // 可选的默认值
    // 检查是否存在赋值右侧的表达式
    if (assign.rhs().present()) {
      // 使用类型解析器评估默认值的表达式，并获取解析结果
      std::vector<IValue> parsed = type_parser.evaluateDefaults(
          assign.rhs().range(), {assign.rhs().get()}, {assign.type().get()});
      // 断言解析结果的大小为1
      TORCH_INTERNAL_ASSERT(parsed.size() == 1);
      // 将第一个解析出的默认值存储到变量 default_val 中
      default_val = parsed[0];
    }

    // 解析赋值语句中的类型表达式，获取类型对象
    auto type = type_parser.parseTypeFromExpr(assign.type().get());

    // 将字段名添加到字段名列表中
    field_names.emplace_back(std::move(name));
    // 将类型对象添加到字段类型列表中
    field_types.emplace_back(std::move(type));
    // 如果存在默认值，则将其添加到字段默认值列表中
    if (default_val) {
      field_defaults.emplace_back(std::move(*default_val));
    }
  }

  // 使用字段信息创建命名的元组类型对象
  auto tt = TupleType::createNamed(
      qualified_name, field_names, field_types, field_defaults);
  // 在当前编译单元中注册类型对象
  cu_->register_type(tt);
}

void SourceImporterImpl::parsePossibleVersionNumber(Lexer& L) {
  // 解析可能的版本号
  // 旧版本的序列化每个文件生成一个 op_version_set 字符串
  // 现在我们使用 PyTorchStreamReader 处理单一版本，不再使用 op_version_set
  // 以前我们检查 op_version_set 是否更新以支持向前兼容性，但现在它不存在了，因此我们可以丢弃它。
  if (L.cur().kind == TK_IDENT && L.cur().text() == "op_version_set") {
    auto range = L.cur().range; // 记录当前词法单元的范围
    L.next(); // 移动到下一个词法单元
    L.expect('='); // 确保下一个是 '='
    std::string version_text = L.expect(TK_NUMBER).text(); // 期望下一个词法单元是数字，并获取其文本
    L.expect(TK_NEWLINE); // 确保下一个是换行符
  }
}

// 解析导入语句
// 旧版本的序列化需要导入语句，并按照导入顺序逐个文件定义类。
// 问题在于在 Python 中，即使在单个类之间没有循环依赖的情况下，也可能构造文件之间的循环依赖。
// 新版本的加载器现在逐个类编译，因此不再需要遵循导入顺序。
// 未来的序列化可能会停止生成导入代码。
void SourceImporterImpl::parseImports(Lexer& L) {
  while (L.nextIf(TK_IMPORT)) { // 如果下一个词法单元是 IMPORT，则进入循环
    std::ostringstream s; // 创建一个字符串流
    while (L.cur().kind != TK_NEWLINE) { // 当当前词法单元不是换行符时
      s << L.cur().text(); // 将当前词法单元的文本添加到字符串流中
      L.next(); // 移动到下一个词法单元
    }
    L.expect(TK_NEWLINE); // 确保下一个是换行符
  }
}

std::shared_ptr<SugaredValue> ClassNamespaceValue::attr(
    const SourceRange& loc,
    GraphFunction& m,
    const std::string& name) {
  auto fullName = c10::QualifiedName(basename_, name); // 构造完整的名称
  // 可能是一个 ClassType 或 NamedTuple 构造函数
  if (auto serializable_type = si_->findNamedType(fullName)) { // 查找命名类型
    if (auto classType = serializable_type->cast<ClassType>()) { // 如果是 ClassType
      return std::make_shared<ClassValue>(classType); // 返回 ClassValue
    } else if (auto tupleType = serializable_type->cast<TupleType>()) { // 如果是 TupleType
      return std::make_shared<NamedTupleConstructor>(tupleType); // 返回 NamedTupleConstructor
    } else if (auto enumType = serializable_type->cast<EnumType>()) { // 如果是 EnumType
      return std::make_shared<SugaredEnumClass>(enumType); // 返回 SugaredEnumClass
    }
  }

  // 否则可能是一个自由函数
  if (auto fn = si_->findFunction(fullName)) { // 查找函数
    return std::make_shared<FunctionValue>(fn); // 返回 FunctionValue
  }

  // 如果以上情况都不是，则假定它是另一个命名空间
  return std::make_shared<ClassNamespaceValue>(std::move(fullName), si_); // 返回 ClassNamespaceValue
}

SourceImporter::SourceImporter(
    // 将导入的源文件关联到的编译单元
    std::shared_ptr<CompilationUnit> cu,
    const std::vector<IValue>* constant_table,
    SourceLoader loader,
    size_t version)
    : pImpl(std::make_shared<SourceImporterImpl>(
          std::move(cu),
          constant_table,
          std::move(loader),
          version)) {}

TypePtr SourceImporter::loadType(const QualifiedName& name) const {
  ScriptTypeParser type_parser(pImpl); // 使用解析器解析类型
  TypePtr t = type_parser.parseType(name.qualifiedName()); // 解析给定名称的类型
  return t; // 返回解析后的类型
}

void SourceImporter::LEGACY_import_methods(
    const Module& mod,
    const std::shared_ptr<Source>& src) {
  pImpl->LEGACY_import_methods(mod, src); // 调用实现类的 LEGACY_import_methods 方法
}
# 定义 SourceImporter 类的析构函数，使用默认方式进行析构
SourceImporter::~SourceImporter() = default;
# 结束 torch::jit 命名空间
} // namespace torch::jit
```