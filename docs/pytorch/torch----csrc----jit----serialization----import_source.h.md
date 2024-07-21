# `.\pytorch\torch\csrc\jit\serialization\import_source.h`

```py
#pragma once

#include <ATen/core/ivalue_inl.h>
#include <ATen/core/qualified_name.h>
#include <c10/util/Optional.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/parser.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/frontend/script_type_parser.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/custom_class.h>
#include <functional>
#include <memory>
#include <regex>
#include <string>
#include <vector>

// 命名空间 torch::jit 中定义了本模块的所有内容

namespace torch::jit {

// 使用 std::function 作为 SourceLoader 的别名，接受一个 std::string 参数并返回一个 std::shared_ptr<Source> 对象
using SourceLoader = std::function<std::shared_ptr<Source>(const std::string&)>;

// SourceImporterImpl 类，继承自 Resolver，并且可以通过 std::enable_shared_from_this 实现共享指针的安全共享
struct SourceImporterImpl : public Resolver,
                            std::enable_shared_from_this<SourceImporterImpl> {
  // 构造函数，初始化 SourceImporterImpl 对象
  SourceImporterImpl(
      std::shared_ptr<CompilationUnit> cu,
      const std::vector<at::IValue>* constant_table,
      SourceLoader source_loader,
      size_t version);
  
  // 根据给定的 QualifiedName 查找命名类型并返回其 TypePtr
  TypePtr findNamedType(const QualifiedName& name);
  
  // 根据给定的 QualifiedName 查找函数并返回其 Function 指针
  Function* findFunction(const QualifiedName& name);
  
  // 如果需要的话，解析给定限定符的源码
  void parseSourceIfNeeded(const std::string& qualifier);
  
  // 导入给定模块的方法，使用给定的源码指针 src
  void LEGACY_import_methods(
      const Module& mod,
      const std::shared_ptr<Source>& src);

  // 解析指定名称的值，并返回其对应的 SugaredValue 指针
  std::shared_ptr<SugaredValue> resolveValue(
      const std::string& name,
      GraphFunction& m,
      const SourceRange& loc) override;
  
  // 解析指定名称的类型，并返回其 TypePtr
  TypePtr resolveType(const std::string& name, const SourceRange& loc) override;

 private:
  // 导入给定限定符和函数定义的函数
  void importFunction(const std::string& qualifier, const Def& def);
  
  // 导入给定限定符和类定义的命名类型
  void importNamedType(const std::string& qualifier, const ClassDef& class_def);
  
  // 对于指定的限定类名和分配操作，进行属性分配的特殊处理
  std::optional<Assign> attributeAssignmentSpecialHandlingHack(
      const QualifiedName& qualified_classname,
      const Assign& assign);
  
  // 导入给定限定类名和类定义的类
  void importClass(
      const QualifiedName& qualified_classname,
      const ClassDef& class_def,
      bool is_module);
  
  // 导入给定限定名称和枚举定义的枚举
  void importEnum(
      const QualifiedName& qualified_name,
      const ClassDef& enum_def);
  
  // 导入给定限定名称和命名元组定义的命名元组
  void importNamedTuple(
      const QualifiedName& qualified_name,
      const ClassDef& named_tuple_def);

  // 解析可能的版本号，从 Lexer 中读取
  void parsePossibleVersionNumber(Lexer& L);

  // 解析导入声明，从 Lexer 中读取
  void parseImports(Lexer& L);

  // 编译单元的共享指针 cu_
  std::shared_ptr<CompilationUnit> cu_;
  
  // 环境变量的映射，从字符串到 SugaredValue 的共享指针
  std::unordered_map<std::string, std::shared_ptr<SugaredValue>> env_;
  
  // 源加载器，接受 std::string 参数并返回 std::shared_ptr<Source> 的函数对象
  SourceLoader source_loader_;
  
  // 版本号，使用 std::optional 进行包装，可能为空
  std::optional<size_t> version_ = c10::nullopt;
  
  // 已加载的源文件集合
  std::unordered_set<std::string> loaded_sources_;
  
  // 将要定义但尚未请求类型的命名类型和函数的映射
  std::unordered_map<QualifiedName, TreeRef> to_be_defined_;
};

// 给定序列化 TorchScript 源码的目录，
// 此类允许按名称加载单个命名类型的源码。
// 解决源文件之间的依赖关系，并在需要时解析源文件。
struct TORCH_API SourceImporter {
  // 构造函数，初始化 SourceImporter 对象
  SourceImporter(
      // 编译单元将拥有导入的源代码
      std::shared_ptr<CompilationUnit> cu,
      // 指向常量表的指针，用于加载常量表中的内容
      const std::vector<at::IValue>* constant_table,
      // 源码加载器，用于加载源码
      SourceLoader loader,
      // 版本号
      size_t version);

  // 根据 QualifiedName 加载类型，并返回 TypePtr
  TypePtr loadType(const QualifiedName& name) const;

  // 使用 SourceImporter 将定义在 `src` 中的方法添加到模块 `mod` 中，
  // 通过 loadType 解析任何类
  void LEGACY_import_methods(
      const Module& mod,
      const std::shared_ptr<Source>& src);

  // 析构函数，清理 SourceImporter 对象
  ~SourceImporter();

 private:
  // 指向具体实现的指针
  std::shared_ptr<SourceImporterImpl> pImpl;
};

// 命名空间声明，torch::jit
} // namespace torch::jit
```