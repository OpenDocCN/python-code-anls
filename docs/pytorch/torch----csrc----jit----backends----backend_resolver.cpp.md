# `.\pytorch\torch\csrc\jit\backends\backend_resolver.cpp`

```
// 包含头文件：用于 Torch 的 JIT 后端解析器
#include <torch/csrc/jit/backends/backend_resolver.h>
// 包含头文件：用于 Torch 的前端糖值处理
#include <torch/csrc/jit/frontend/sugared_value.h>
// 包含头文件：自定义 Torch 类
#include <torch/custom_class.h>

// Torch 命名空间
namespace torch {
// Torch JIT 命名空间
namespace jit {
// 匿名命名空间：用于定义 ClassNamespaceValue 结构体
namespace {
// ClassNamespaceValue 结构体：用于处理导入源码中的 ClassNamespaceValue，不包含 SourceImporterImpl 引用。
// 这有助于解析生成代码中 LoweredModule 的 __torch__.torch.classes.backends.{backend_name} 符号。
struct ClassNamespaceValue : public SugaredValue {
  explicit ClassNamespaceValue(c10::QualifiedName name)
      : basename_(std::move(name)) {}

  // attr 方法：获取属性值
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& name) override {
    // 创建完整的限定名称
    auto fullName = c10::QualifiedName(basename_, name);

    // 检查是否为自定义类
    if (auto custom_class = getCustomClass(fullName.qualifiedName())) {
      return std::make_shared<ClassValue>(custom_class);
    }

    // 如果不是自定义类，则假定它是另一个命名空间
    return std::make_shared<ClassNamespaceValue>(std::move(fullName));
  }

  // kind 方法：返回类的类型
  std::string kind() const override {
    return "Class Namespace";
  }

 private:
  c10::QualifiedName basename_;
};

// LoweredModuleResolver 结构体：用于解析 LoweredModule 中自定义后端类的解析器
struct LoweredModuleResolver : public Resolver {
  // resolveValue 方法：解析值
  std::shared_ptr<SugaredValue> resolveValue(
      const std::string& name,
      GraphFunction& m,
      const SourceRange& loc) override {
    // 如果名称为 "torch"，返回内置模块 "aten"
    if (name == "torch") {
      return std::make_shared<BuiltinModule>("aten");
    }
    // 如果名称为 "__torch__"，返回 ClassNamespaceValue 结构体
    else if (name == "__torch__") {
      return std::make_shared<ClassNamespaceValue>(c10::QualifiedName(name));
    }
    // 如果名称为 "Exception"，返回异常值 ExceptionValue
    else if (name == "Exception") {
      return std::make_shared<ExceptionValue>(name);
    }

    // 默认返回空指针
    return nullptr;
  }

  // resolveType 方法：解析类型
  TypePtr resolveType(const std::string& name, const SourceRange& loc)
      override {
    // 默认返回空指针
    return nullptr;
  }
};

} // namespace

// loweredModuleResolver 函数：返回 LoweredModuleResolver 的共享指针
std::shared_ptr<Resolver> loweredModuleResolver() {
  // 创建 LoweredModuleResolver 的共享指针 resolver
  std::shared_ptr<Resolver> resolver =
      std::make_shared<LoweredModuleResolver>();
  // 返回 resolver
  return resolver;
}

} // namespace jit
} // namespace torch
```