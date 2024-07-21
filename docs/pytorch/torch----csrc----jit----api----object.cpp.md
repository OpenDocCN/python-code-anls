# `.\pytorch\torch\csrc\jit\api\object.cpp`

```
// 引入 Torch JIT 模块中的必要头文件

#include <torch/csrc/jit/api/object.h>

// 引入 ATen 库中的 JIT 类型定义
#include <ATen/core/jit_type.h>

// 引入 Torch JIT 编译单元的定义
#include <torch/csrc/jit/api/compilation_unit.h>

// 引入 Torch JIT 前端的解析器接口
#include <torch/csrc/jit/frontend/resolver.h>

// 引入 Torch JIT 前端的糖语法值定义
#include <torch/csrc/jit/frontend/sugared_value.h>

// 定义 Torch JIT 的命名空间
namespace torch::jit {

// 实现 Object 类的构造函数，接受编译单元和类类型作为参数
Object::Object(
    std::shared_ptr<CompilationUnit> cu,
    const c10::ClassTypePtr& type)
    : Object(c10::ivalue::Object::create(
          c10::StrongTypePtr(std::move(cu), type),
          type->numAttributes())) {}

// 查找指定方法名称的方法，如果找到则返回其 Method 对象，否则返回空
std::optional<Method> Object::find_method(const std::string& basename) const {
  // 遍历当前类类型中定义的所有方法
  for (Function* fn : type()->methods()) {
    // 如果方法的名称与目标名称匹配，则返回对应的 Method 对象
    if (fn->name() == basename) {
      return Method(_ivalue(), fn);
    }
  }
  // 若未找到匹配的方法，则返回空 optional 对象
  return c10::nullopt;
}

// 定义对象的方法，将给定源代码和解析器注册到对象所属的编译单元中
void Object::define(const std::string& src, const ResolverPtr& resolver) {
  // 创建 SimpleSelf 对象，表示当前对象的上下文
  const auto self = SimpleSelf(type());
  
  // 将源代码和解析器注册到对象所属的编译单元中
  _ivalue()->compilation_unit()->define(
      *type()->name(), src, resolver ? resolver : nativeResolver(), &self);
}

// 创建当前对象的副本并返回
Object Object::copy() const {
  return Object(_ivalue()->copy());
}

// 创建当前对象的深层副本并返回
Object Object::deepcopy() const {
  return Object(_ivalue()->deepcopy());
}

} // namespace torch::jit
```