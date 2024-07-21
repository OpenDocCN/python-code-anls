# `.\pytorch\torch\csrc\jit\frontend\parser.h`

```py
#pragma once
// 预处理指令：指定头文件只包含一次

#include <torch/csrc/Export.h>
// 包含 Torch 导出相关的头文件

#include <torch/csrc/jit/frontend/tree.h>
// 包含 Torch 前端树相关的头文件

#include <torch/csrc/jit/frontend/tree_views.h>
// 包含 Torch 前端树视图相关的头文件

#include <memory>
// 包含 C++ 标准库中的内存管理相关功能

namespace torch {
namespace jit {

struct Decl;
// 声明一个结构体 Decl

struct ParserImpl;
// 声明一个结构体 ParserImpl

struct Lexer;

TORCH_API Decl mergeTypesFromTypeComment(
    const Decl& decl,
    const Decl& type_annotation_decl,
    bool is_method);
// 声明一个函数 mergeTypesFromTypeComment，用于合并类型注释信息

struct TORCH_API Parser {
  explicit Parser(const std::shared_ptr<Source>& src);
  // 显式构造函数，接受一个源代码的共享指针参数

  TreeRef parseFunction(bool is_method);
  // 解析函数，返回树节点引用，接受一个布尔值参数指示是否为方法

  TreeRef parseClass();
  // 解析类，返回树节点引用

  Decl parseTypeComment();
  // 解析类型注释，返回声明节点

  Expr parseExp();
  // 解析表达式，返回表达式节点

  Lexer& lexer();
  // 返回 Lexer 对象的引用

  ~Parser();
  // 析构函数，用于清理资源

 private:
  std::unique_ptr<ParserImpl> pImpl;
  // 私有成员变量，唯一指针，指向 ParserImpl 对象
};

} // namespace jit
} // namespace torch
```