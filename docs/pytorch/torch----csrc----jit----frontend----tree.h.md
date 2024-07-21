# `.\pytorch\torch\csrc\jit\frontend\tree.h`

```
#pragma once
// 预处理指令，确保本头文件只被编译一次

#include <functional>
// 引入函数对象相关的标准库头文件

#include <memory>
// 引入智能指针相关的标准库头文件

#include <unordered_map>
// 引入无序映射容器相关的标准库头文件

#include <vector>
// 引入动态数组容器相关的标准库头文件

#include <c10/util/SmallVector.h>
// 引入小型动态数组容器相关的头文件

#include <c10/util/intrusive_ptr.h>
// 引入侵入式指针相关的头文件

#include <torch/csrc/jit/frontend/lexer.h>
// 引入 Torch 前端词法分析器相关的头文件

namespace torch {
namespace jit {

// 声明命名空间 torch::jit

// Trees are used to represent all forms of TC IR, pre- and post-typechecking.
// Rather than have a full class hierarchy for all TC statements, trees are a
// slight variation of Lisp s-expressions. For instance, the expression a*b+1
// is represented as:
// (+ (* (ident a) (ident b)) (const 1))
// Atoms like 'a', 'b', and '1' are represented by subclasses of Tree which
// define stringValue(). Everything else is a Compound object, which has a
// 'kind' that is a token from lexer.h's TokenKind enum. Single-character
// operators like '+' are represented using the character itself (so, add.kind()
// would be '+'). Each Compound object also contains a list of subtrees and is
// associated with a SourceRange for error reporting.
// Memory management of trees is done using intrusive_ptr.

// 树结构用于表示所有类型检查前后的 TC IR。不同于为所有 TC 语句建立完整的类层次结构，
// 树结构是一种轻微变体的 Lisp s-表达式。例如，表达式 a*b+1 被表示为:
// (+ (* (ident a) (ident b)) (const 1))
// 'a'、'b' 和 '1' 这样的原子由 Tree 的子类表示，定义了 stringValue() 方法。
// 其他所有内容都是 Compound 对象，其 'kind' 属性是来自 lexer.h 的 TokenKind 枚举的一个标记。
// 单字符操作符如 '+' 则直接使用字符本身表示（因此，add.kind() 将是 '+'）。
// 每个 Compound 对象还包含一个子树列表，并关联一个 SourceRange 用于错误报告。
// 树结构的内存管理使用 intrusive_ptr 实现。

struct Tree;
// 声明结构体 Tree

using TreeRef = c10::intrusive_ptr<Tree>;
// 使用 intrusive_ptr 定义 TreeRef 类型别名，用于管理 Tree 的引用计数

using TreeList = at::SmallVector<TreeRef, 4>;
// 使用 SmallVector 定义 TreeList 类型别名，表示包含 TreeRef 的小型动态数组

struct Tree : c10::intrusive_ptr_target {
  // 定义结构体 Tree，继承自 intrusive_ptr_target

  Tree(int kind_) : kind_(kind_) {}
  // 构造函数，初始化 kind_ 成员变量

  int kind() const {
    return kind_;
  }
  // 返回 kind_ 成员变量的值，表示树节点的类型标记

  virtual bool isAtom() const {
    return true;
  }
  // 虚函数，返回 true，表示该节点是一个原子节点

  virtual const SourceRange& range() const {
    throw std::runtime_error("is an Atom");
  }
  // 虚函数，抛出运行时错误，表示在原子节点上调用了不支持的 range() 函数

  virtual const std::string& stringValue() const {
    throw std::runtime_error("stringValue can only be called on TK_STRING");
  }
  // 虚函数，抛出运行时错误，表示在非字符串类型的节点上调用了不支持的 stringValue() 函数

  virtual const TreeList& trees() const {
    static const TreeList empty_trees = {};
    return empty_trees;
  }
  // 虚函数，返回空的子树列表，表示该节点没有子节点

  const TreeRef& tree(size_t i) const {
    return trees().at(i);
  }
  // 返回第 i 个子节点的引用

  virtual TreeRef map(const std::function<TreeRef(TreeRef)>& fn) {
    (void)fn;
    c10::raw::intrusive_ptr::incref(this); // we are creating a new pointer
                                           // from a raw `this` pointer
                                           // so we need to bump the refcount
                                           // to account for this ownership
    return TreeRef::reclaim(this);
  }
  // 映射函数，接受一个函数对象，但实际上没有执行任何映射操作，仅增加引用计数后返回自身的引用

  template <typename... Args>
  void match(int k, Args&... args) const {
    matchD(k, "unknown", 0, args...);
  }
  // 模式匹配函数，根据给定的类型标记 k，匹配对应的子节点，并将结果存储到 args 中

  template <typename... Args>
  void matchD(int k, const char* filename, int lineno, Args&... args) const {
    std::initializer_list<TreeRef*> vars = {args...};
    matchNumSubtreesD(k, filename, lineno, vars.size(), true);
    size_t i = 0;
    for (TreeRef* v : vars) {
      *v = trees()[i++];
    }
  }
  // 模式匹配函数的详细实现，使用给定的类型标记 k、文件名 filename 和行号 lineno 进行匹配，
  // 并将结果存储到 args 中

  void matchNumSubtrees(int k, size_t expected_subtrees) {
    return matchNumSubtreesD(k, "unknown", 0, expected_subtrees, false);
  }
  // 匹配具有预期子节点数量的节点，对应的详细实现

  void matchNumSubtreesD(
      int k,
      const char* filename,
      int lineno,
      size_t expected_subtrees,
      bool allow_more) const {
    // 匹配具有预期子节点数量的节点，并支持更多子节点的情况，对应的详细实现



      // 匹配具有预期子节点数量的节点，并支持更多子节点的情况，对应的详细实现
    // 如果当前树的类型与期望的类型不匹配，则抛出运行时错误
    if (kind() != k) {
      // 创建一个字符串流对象，用于构建错误信息，包括文件名、行号、期望的类型和实际的类型
      std::stringstream ss;
      ss << filename << ":" << lineno << ": expecting kind '" << kindToString(k)
         << "' but found '" << kindToString(kind()) << "'\n";
      // 在源代码范围内突出显示错误信息
      range().highlight(ss);
      // 抛出运行时异常，并将错误信息作为异常的内容
      throw std::runtime_error(ss.str());
    }
    // 如果树的子树数量少于期望的数量，或者不允许更多子树且数量不匹配，则抛出运行时错误
    if (trees().size() < expected_subtrees ||
        (!allow_more && trees().size() != expected_subtrees)) {
      // 创建一个字符串流对象，用于构建错误信息，包括文件名、行号、期望的子树数量和实际的子树数量
      std::stringstream ss;
      ss << filename << ":" << lineno << ": expected at least "
         << expected_subtrees << " subtrees, but found only " << trees().size()
         << "\n";
      // 在源代码范围内突出显示错误信息
      range().highlight(ss);
      // 抛出运行时异常，并将错误信息作为异常的内容
      throw std::runtime_error(ss.str());
    }
  }
  // 虚析构函数，使用默认实现
  ~Tree() override = default;

 private:
  // 私有成员变量，表示树的类型
  int kind_;
};

// 继承自Tree的String结构体，表示字符串类型节点
struct String : public Tree {
  // 构造函数，接受一个std::string类型的参数并初始化value_
  String(std::string value) : Tree(TK_STRING), value_(std::move(value)) {}
  // 返回字符串值的引用
  const std::string& stringValue() const override {
    return value_;
  }
  // 创建String对象的静态方法，使用可变参数模板
  template <typename... Args>
  static TreeRef create(Args&&... args) {
    return c10::make_intrusive<String>(std::forward<Args>(args)...);
  }

 private:
  std::string value_; // 字符串值
};

// 将给定的范围c与TreeList中的其他范围合并后返回合并后的范围
static SourceRange mergeRanges(SourceRange c, const TreeList& others) {
  for (const auto& t : others) {
    // 如果当前节点是原子节点，则跳过
    if (t->isAtom())
      continue;
    // 计算合并后的起始和结束位置
    size_t s = std::min(c.start(), t->range().start());
    size_t e = std::max(c.end(), t->range().end());
    c = SourceRange(c.source(), s, e);
  }
  return c; // 返回合并后的范围
}

// 继承自Tree的Compound结构体，表示复合节点
struct Compound : public Tree {
  // 构造函数，接受节点种类kind和范围range
  Compound(int kind, SourceRange range)
      : Tree(kind), range_(std::move(range)) {}
  // 构造函数，接受节点种类kind、范围range_和树列表trees_
  Compound(int kind, const SourceRange& range_, TreeList&& trees_)
      : Tree(kind),
        range_(mergeRanges(range_, trees_)), // 调用mergeRanges合并范围
        trees_(std::move(trees_)) {}
  // 返回树列表trees_
  const TreeList& trees() const override {
    return trees_;
  }
  // 创建Compound对象的静态方法，接受节点种类kind、范围range_和树列表trees_
  static TreeRef create(
      int kind,
      const SourceRange& range_,
      TreeList&& trees_) {
    return c10::make_intrusive<Compound>(kind, range_, std::move(trees_));
  }
  // 判断是否为原子节点，复合节点返回false
  bool isAtom() const override {
    return false;
  }
  // 对树进行映射操作，接受一个函数fn，并返回映射后的树
  TreeRef map(const std::function<TreeRef(TreeRef)>& fn) override {
    TreeList ret;
    for (auto& t : trees()) {
      ret.push_back(fn(t));
    }
    return Compound::create(kind(), range(), std::move(ret));
  }

  // 返回节点的范围
  const SourceRange& range() const override {
    return range_;
  }

 private:
  SourceRange range_; // 节点的范围
  TreeList trees_;    // 树列表
};

// 树的美化打印类
struct pretty_tree {
  // 构造函数，接受一个TreeRef类型的树和一个列数col，默认为40
  pretty_tree(const TreeRef& tree, size_t col = 40) : tree(tree), col(col) {}
  const TreeRef& tree; // 树的引用
  size_t col;          // 列数
  std::unordered_map<TreeRef, std::string> flat_strings; // 平面字符串的哈希映射

  // 获取平面化字符串的函数
  const std::string& get_flat(const TreeRef& t) {
    auto it = flat_strings.find(t);
    if (it != flat_strings.end())
      return it->second;

    std::stringstream out;
    switch (t->kind()) {
      case TK_STRING:
        out << t->stringValue(); // 如果是字符串类型，输出字符串值
        break;
      default:
        out << "(" << kindToString(t->kind()); // 否则输出节点类型
        for (const auto& e : t->trees()) {
          out << " " << get_flat(e); // 递归获取子树的字符串表示
        }
        out << ")";
        break;
    }
    auto it_ = flat_strings.emplace(t, out.str());
    return it_.first->second;
  }

  // 打印树的函数，接受一个输出流out、树t和缩进量indent
  void print(std::ostream& out, const TreeRef& t, int indent) {
    const std::string& s = get_flat(t); // 获取树的平面化字符串表示
    if (indent + s.size() < col || t->isAtom()) {
      out << s; // 如果字符串长度不超过列数或者是原子节点，则直接输出
      return;
    }
    std::string k = kindToString(t->kind());
    out << "(" << k; // 否则输出节点类型
    for (const auto& e : t->trees()) {
      out << "\n" << std::string(indent + 2, ' '); // 输出换行和缩进
      print(out, e, indent + 2); // 递归打印子树
    }
    out << ")";
  }
};

// 重载输出流操作符<<，用于打印pretty_tree对象
static inline std::ostream& operator<<(std::ostream& out, pretty_tree t_) {
  t_.print(out, t_.tree, 0); // 调用pretty_tree的打印函数
  return out << std::endl; // 输出换行
}
# 定义一个内联函数，重载输出流操作符 <<，用于输出 TreeRef 对象到指定输出流
static inline std::ostream& operator<<(std::ostream& out, const TreeRef& t) {
  # 调用 pretty_tree 函数，将 TreeRef 对象 t 转换为美化的树形字符串，并输出到给定的输出流
  return out << pretty_tree(t);
}

# 命名空间 jit 结束
} // namespace jit

# 命名空间 torch 结束
} // namespace torch
```