# `.\pytorch\torch\csrc\jit\frontend\tree_views.h`

```py
// 一旦包含这个文件，就确保整个头文件只被编译一次
#pragma once

// 包含 Torch JIT 前端错误报告的头文件
#include <torch/csrc/jit/frontend/error_report.h>
// 包含 Torch JIT 前端字符串转换的头文件
#include <torch/csrc/jit/frontend/strtod.h>
// 包含 Torch JIT 前端树结构的头文件
#include <torch/csrc/jit/frontend/tree.h>

// 包含 C10 的复数实用工具
#include <c10/util/complex.h>

// 包含用于函数对象的头文件
#include <functional>
// 包含用于标准输入输出流的头文件
#include <iostream>
// 包含用于字符串处理的头文件
#include <string>
// 包含用于通用工具的头文件
#include <utility>

// Torch 命名空间开始
namespace torch {
// JIT 命名空间开始
namespace jit {

// clang-format off

// TreeView 提供了一个静态类型的方式来遍历树结构，
// 该树结构应该根据以下语法规则形成。
//
// 关于类型及其别名的一些注释：
// - List<T> 实际上是具有 TK_LIST 种类和子树作为元素的树
// - Maybe<T> 实际上是具有 TK_OPTION 种类且具有 0 或 1 个类型为 T 的子树的树
// - 内建类型有：Ident (TK_IDENT), String (TK_STRING)
//
// Param = Param(Maybe<Expr> type, Ident name)                          TK_PARAM
//
// Decl  = Decl(List<Param> params, Maybe<Expr> return_type)            TK_DECL
// Def   = Def(Ident name, Decl decl, List<Stmt> body)                  TK_DEF
// ClassDef = ClassDef(Ident name,                                      TK_CLASS_DEF
//                     Maybe<Expr> superclass,
//                     List<Stmt> body)
//
// Stmt  = If(Expr cond, List<Stmt> true_body, List<Stmt> false_body)   TK_IF
//       | For(List<Expr> targets, List<Expr> iters, List<Stmt> body)   TK_FOR
//       | While(Expr cond, List<Stmt> body)                            TK_WHILE
//       | Global(List<Ident> idents)                                   TK_GLOBAL
//       -- 注意：在 lhs 上允许的 Expr 类型仅限于 Var
//          或包含 Var 以及可选的 Starred 的元组
//       | Assign(Expr lhs, Maybe<Expr> rhs, Maybe<Expr> type)          TK_ASSIGN
//       | AugAssign(Expr lhs, AugAssignKind aug_op, Expr rhs)          TK_AUG_ASSIGN
//       | Return(List<Expr> values)                                    TK_RETURN
//       | ExprStmt(List<Expr> expr)                                    TK_EXPR_STMT
//       | Raise(Expr expr)                                             TK_RAISE
//       | Def                                                          TK_DEF
//       | With(List<WithItem> targets, List<Stmt> body)                TK_WITH
//
// Expr  = TernaryIf(Expr cond, Expr true_expr, Expr false_expr)        TK_IF_EXPR
//       | BinOp(Expr lhs, Expr rhs)
//       |     And                                                      TK_AND
//       |     Or                                                       TK_OR
//       |     Lt                                                       '<'
//       |     Gt                                                       '>'
//       |     Eq                                                       TK_EQ
//       |     Le                                                       TK_LE
//       |     Ge                                                       TK_GE
//       |     Ne                                                       TK_NE
//       |     Is                                                       TK_IS
//       |     IsNot                                                    TK_ISNOT
//       |     Add                                                      '+'
//       |     Sub                                                      '-'
//       |     Mul                                                      '*'
//       |     Div                                                      '/'
//       |     Mod                                                      '%'
//       |     MatMult                                                  '@'
//       |     Pow                                                      TK_POW
//       | UnaryOp(Expr expr)
//       |     Not                                                      TK_NOT
//       |     USub                                                     '-'
//       | Const(String value)                                          TK_CONST
//       -- NB: x.name(y) is desugared into name(x, y)
//       | Apply(Ident name, List<Expr> args, List<Attribute> kwargs)   TK_APPLY
//       | Select(Expr value, Ident selector)                           '.'
//       | Subscript(Expr value, List<Expr> subscript_exprs)            TK_SUBSCRIPT
//       | SliceExpr(Maybe<Expr> start, Maybe<Expr> end)                TK_SLICE_EXPR
//       | Var(Ident name)                                              TK_VAR
//       | ListLiteral(List<Expr> inputs)                               TK_LIST_LITERAL
//       | TupleLiteral(List<Expr> inputs)                              TK_TUPLE_LITERAL
//       | Starred(Expr expr)                                           TK_STARRED
//       | WithItem(Expr target, Maybe<Var> var)                        TK_WITH_ITEM
// -- NB: only allowed expressions are Const or List(Const)
//        (List as a value, not type constructor)
// Attribute = Attribute(Ident name, Expr value)                        TK_ATTRIBUTE
//
// AugAssignKind =
//            | Add()                                                   TK_PLUS_EQ
//            | Sub()                                                   TK_MINUS_EQ
//            | Mul()                                                   TK_TIMES_EQ
//            | Div()                                                   TK_DIV_EQ
//            | Mod()                                                   TK_MOD_EQ
//

// Each subclass of TreeView should provide:
// 1. Constructor that takes a TreeRef, and checks that it's of the right type.
// 2. Accessors that get underlying information out of the object. If they
//    return subtrees, they should wrap them in appropriate views too.
// 3. Static method 'create' that creates the underlying TreeRef object
//    for every TreeRef kind that has a TreeView, the parser always uses
//    (e.g.) Ident::create rather than Compound::Create, this means that
//    changes to the structure of Ident are always made right here rather
//    than both in the parser and in this code.
// XXX: 这些结构体应该没有字段，以防止通过值传递时切片
// clang-format on
// 树视图结构体，用于表示树的视图
struct TreeView {
  explicit TreeView(TreeRef tree) : tree_(std::move(tree)) {} // 构造函数，接受一个树对象并初始化
  TreeRef tree() const { // 返回当前树对象的引用
    return tree_;
  }
  const SourceRange& range() const { // 返回当前树对象的范围引用
    return tree_->range();
  }
  operator TreeRef() const { // 类型转换运算符，返回当前树对象
    return tree_;
  }
  const TreeRef& get() const { // 返回当前树对象的常量引用
    return tree_;
  }
  int kind() const { // 返回当前树对象的类型
    return tree_->kind();
  }
  void dump() const { // 打印当前树对象到标准输出流
    std::cout << tree_;
  }

 protected:
  const TreeRef& subtree(size_t i) const { // 返回当前树对象的第i个子树的引用
    return tree_->trees().at(i);
  }
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  TreeRef tree_; // 树对象的常量引用成员变量
};

// 泛型列表迭代器结构体，用于迭代树列表的元素
template <typename T>
struct ListIterator {
  ListIterator(TreeList::const_iterator it) : it(it) {} // 构造函数，接受树列表的常量迭代器并初始化
  bool operator!=(const ListIterator& rhs) const { // 不等运算符重载，比较迭代器是否不相等
    return it != rhs.it;
  }
  bool operator==(const ListIterator& rhs) const { // 等于运算符重载，比较迭代器是否相等
    return it == rhs.it;
  }
  T operator*() const { // 解引用运算符重载，返回迭代器当前位置的元素
    return T(*it);
  }
  ListIterator& operator+=(std::ptrdiff_t n) { // 递增运算符重载，将迭代器向前移动n步
    it += n;
    return *this;
  }
  ListIterator& operator++() { // 前置递增运算符重载，将迭代器向前移动一步
    ++it;
    return *this;
  }
  ListIterator& operator--() { // 前置递减运算符重载，将迭代器向后移动一步
    --it;
    return *this;
  }

 private:
  TreeList::const_iterator it; // 树列表的常量迭代器成员变量
};

// 泛型列表结构体，继承自树视图结构体
template <typename T>
struct List : public TreeView {
  using iterator = ListIterator<T>; // 使用ListIterator作为迭代器类型
  using const_iterator = ListIterator<T>; // 使用ListIterator作为常量迭代器类型

  List(const TreeRef& tree) : TreeView(tree) { // 构造函数，接受一个树对象并初始化基类
    tree->match(TK_LIST); // 匹配树对象的类型是否为TK_LIST
    // 迭代列表，临时实例化Ts以检查类型
    for (const T& elem : *this) {
      (void)elem; // 沉默未使用的警告
    }
  }
  iterator begin() const { // 返回列表的起始迭代器
    return iterator(tree_->trees().begin());
  }
  iterator end() const { // 返回列表的结束迭代器
    return iterator(tree_->trees().end());
  }
  bool empty() const { // 检查列表是否为空
    return tree_->trees().begin() == tree_->trees().end();
  }
  T operator[](size_t i) const { // 返回列表中索引为i的元素
    return T(subtree(i));
  }
  TreeRef map(const std::function<TreeRef(const T&)>& fn) { // 将函数应用于列表中的每个元素并返回新的树对象
    return tree_->map([&](TreeRef v) { return fn(T(v)); });
  }
  static List create(const SourceRange& range, const std::vector<T>& subtrees) { // 创建包含给定范围和子树的列表
    TreeList type_erased_sub{subtrees.begin(), subtrees.end()};
    return List(Compound::create(TK_LIST, range, std::move(type_erased_sub)));
  }
  static List unsafeCreate(const SourceRange& range, TreeList&& subtrees) { // 不安全地创建包含给定范围和子树的列表
    return List(Compound::create(TK_LIST, range, std::move(subtrees)));
  }
  size_t size() const { // 返回列表中元素的数量
    return tree_->trees().size();
  }
};

// 可能类型结构体，继承自树视图结构体
template <typename T>
struct Maybe : public TreeView {
  explicit Maybe(const TreeRef& tree) : TreeView(tree) { // 构造函数，接受一个树对象并初始化基类
    tree_->match(TK_OPTION); // 匹配树对象的类型是否为TK_OPTION
    if (tree_->trees().size() > 1)
      throw ErrorReport(tree) << "Maybe trees can have at most one subtree"; // 如果树对象有多于一个子树，则抛出异常
  }
  /* implicit */ Maybe(const T& tree) : TreeView(tree) {} // 隐式转换构造函数，接受一个T类型的对象并初始化基类
  bool present() const { // 检查是否存在子树
    return tree_->trees().size() > 0;
  }
  T get() const { // 返回子树对象
    return T(tree_->trees().at(0));
  }
  TreeRef map(const std::function<TreeRef(const T&)>& fn) { // 将函数应用于子树对象并返回新的树对象
    // 调用 tree_ 对象的 map 方法，对每个节点应用给定的函数 fn，并返回结果集合
    return tree_->map([&](TreeRef v) { return fn(T(v)); });
  }
  
  // 静态方法：根据给定的源代码范围创建一个空的 Maybe 对象
  static Maybe<T> create(const SourceRange& range) {
    return Maybe<T>(Compound::create(TK_OPTION, range, {}));
  }
  
  // 静态方法：根据给定的源代码范围和值创建一个包含值的 Maybe 对象
  static Maybe<T> create(const SourceRange& range, const T& value) {
    return Maybe<T>(Compound::create(TK_OPTION, range, {value}));
  }
};

// 树节点表示标识符的视图，继承自 TreeView
struct Ident : public TreeView {
  explicit Ident(const TreeRef& tree) : TreeView(tree) {
    // 确保树节点的类型为 TK_IDENT
    tree_->match(TK_IDENT);
  }
  // 返回标识符的名称
  const std::string& name() const {
    return subtree(0)->stringValue();
  }
  // 创建一个标识符节点
  static Ident create(const SourceRange& range, std::string name) {
    return Ident(
        // 创建一个复合节点，类型为 TK_IDENT，包含一个字符串节点作为子节点
        Compound::create(TK_IDENT, range, {String::create(std::move(name))}));
  }
};

////////////////////////////////////////////////////////////////////////////////
// Base types (production LHS)
////////////////////////////////////////////////////////////////////////////////

// 树节点表示语句的视图，继承自 TreeView
struct Stmt : public TreeView {
  explicit Stmt(const TreeRef& tree) : TreeView(tree) {
    // 根据树节点类型执行不同的操作
    switch (tree->kind()) {
      // 如果是以下这些类型之一，则通过
      case TK_IF:
      case TK_FOR:
      case TK_WHILE:
      case TK_GLOBAL:
      case TK_ASSIGN:
      case TK_AUG_ASSIGN:
      case TK_RETURN:
      case TK_EXPR_STMT:
      case TK_RAISE:
      case TK_ASSERT:
      case TK_PASS:
      case TK_BREAK:
      case TK_DELETE:
      case TK_CONTINUE:
      case TK_DEF:
      case TK_WITH:
        return;
      // 其他类型则抛出错误
      default:
        throw ErrorReport(tree)
            << kindToString(tree->kind()) << " is not a valid Stmt";
    }
  }
};

// 树节点表示表达式的视图，继承自 TreeView
struct Expr : public TreeView {
  explicit Expr(const TreeRef& tree) : TreeView(tree) {
    // 根据树节点类型执行不同的操作
    switch (tree->kind()) {
      // 如果是以下这些类型之一，则通过
      case TK_IF_EXPR:
      case TK_AND:
      case TK_OR:
      case '<':
      case '>':
      case TK_IS:
      case TK_ISNOT:
      case TK_EQ:
      case TK_LE:
      case TK_GE:
      case TK_NE:
      case '+':
      case '-':
      case TK_UNARY_MINUS:
      case '~':
      case '*':
      case TK_STARRED:
      case '/':
      case '%':
      case TK_NOT:
      case TK_CONST:
      case TK_STRINGLITERAL:
      case TK_TRUE:
      case TK_FALSE:
      case TK_NONE:
      case TK_NONE_TYPE:
      case TK_CAST:
      case TK_APPLY:
      case '.':
      case TK_SUBSCRIPT:
      case TK_SLICE_EXPR:
      case TK_VAR:
      case TK_LIST_LITERAL:
      case TK_TUPLE_LITERAL:
      case TK_DICT_LITERAL:
      case '@':
      case TK_POW:
      case TK_LSHIFT:
      case TK_RSHIFT:
      case TK_FLOOR_DIV:
      case '&':
      case '^':
      case '|':
      case TK_LIST_COMP:
      case TK_DICT_COMP:
      case TK_DOTS:
      case TK_IN:
      case TK_WITH_ITEM:
        return;
      // 其他类型则抛出错误
      default:
        throw ErrorReport(tree)
            << kindToString(tree->kind()) << " is not a valid Expr";
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
// Helper nodes (mostly for function arguments)
////////////////////////////////////////////////////////////////////////////////

// 树节点表示属性的视图，继承自 TreeView
struct Attribute : public TreeView {
  explicit Attribute(const TreeRef& tree) : TreeView(tree) {
    // 确保树节点的类型为 TK_ATTRIBUTE
    tree_->match(TK_ATTRIBUTE);
  }
  // 返回属性的名称
  Ident name() const {
    return Ident(subtree(0));
  }
  // 返回属性的值表达式
  Expr value() const {
    return Expr(subtree(1));
  }
  // 创建一个属性节点
  static Attribute create(
      const SourceRange& range,
      const Ident& name,
      const TreeRef& value) {
    // 创建一个 Attribute 对象，其中包含一个由 TK_ATTRIBUTE 标记的 Compound 对象，
    // 该 Compound 对象使用给定的范围、名称和值创建
    return Attribute(Compound::create(TK_ATTRIBUTE, range, {name, value}));
  }
};

// Param 结构体继承自 TreeView，表示函数参数
struct Param : public TreeView {
  // 构造函数，初始化时匹配 TK_PARAM 类型的树节点
  explicit Param(const TreeRef& tree) : TreeView(tree) {
    tree_->match(TK_PARAM);
  }
  
  // 静态工厂方法，创建 Param 对象
  static Param create(
      const SourceRange& range,
      const Ident& ident,
      const Maybe<Expr>& type,
      const Maybe<Expr>& def,
      bool kwarg_only) {
    // 根据 kwarg_only 的值创建布尔类型的树节点
    TreeRef kwarg_only_tree =
        Compound::create(kwarg_only ? TK_TRUE : TK_FALSE, range, {});
    // 使用 Compound::create 创建 TK_PARAM 类型的树节点，包含 ident、type、def 和 kwarg_only_tree
    return Param(Compound::create(
        TK_PARAM, range, {ident, type, def, std::move(kwarg_only_tree)}));
  }
  
  // 获取参数的标识符
  Ident ident() const {
    return Ident(subtree(0));
  }
  
  // 获取参数的类型表达式，可能为空
  Maybe<Expr> type() const {
    return Maybe<Expr>(subtree(1));
  }
  
  // 获取参数的默认值表达式，可能为空
  Maybe<Expr> defaultValue() const {
    return Maybe<Expr>(subtree(2));
  }
  
  // 判断参数是否仅限关键字参数
  bool kwarg_only() const {
    // 判断第三个子树的类型是否为 TK_TRUE
    return TK_TRUE == subtree(3)->kind();
  }
  
  // 返回具有指定类型的参数对象
  Param withType(const Maybe<Expr>& typ) const {
    return Param::create(range(), ident(), typ, defaultValue(), kwarg_only());
  }
};

////////////////////////////////////////////////////////////////////////////////
// Top level definitions
////////////////////////////////////////////////////////////////////////////////

// Decl 结构体继承自 TreeView，表示顶层声明
struct Decl : public TreeView {
  // 构造函数，初始化时匹配 TK_DECL 类型的树节点
  explicit Decl(const TreeRef& tree) : TreeView(tree) {
    tree->match(TK_DECL);
  }
  
  // 获取声明的参数列表
  List<Param> params() const {
    return List<Param>(subtree(0));
  }
  
  // 获取声明的返回类型表达式，可能为空
  Maybe<Expr> return_type() const {
    return Maybe<Expr>(subtree(1));
  }
  
  // 静态工厂方法，创建 Decl 对象
  static Decl create(
      const SourceRange& range,
      const List<Param>& params,
      const Maybe<Expr>& return_type) {
    // 使用 Compound::create 创建 TK_DECL 类型的树节点，包含 params 和 return_type
    return Decl(Compound::create(TK_DECL, range, {params, return_type}));
  }
};

// Def 结构体继承自 TreeView，表示函数定义
struct Def : public TreeView {
  // 构造函数，初始化时匹配 TK_DEF 类型的树节点
  explicit Def(const TreeRef& tree) : TreeView(tree) {
    tree->match(TK_DEF);
  }
  
  // 返回具有新名称的函数定义对象
  Def withName(std::string new_name) const {
    // 创建新的标识符对象
    auto new_ident = Ident::create(name().range(), std::move(new_name));
    // 调用静态方法 create 创建包含新标识符的函数定义对象
    return create(range(), new_ident, decl(), statements());
  }
  
  // 返回具有新声明的函数定义对象
  Def withDecl(const Decl& decl) const {
    // 调用静态方法 create 创建包含新声明的函数定义对象
    return create(range(), name(), decl, statements());
  }
  
  // 获取函数名标识符
  Ident name() const {
    return Ident(subtree(0));
  }
  
  // 获取函数声明
  Decl decl() const {
    return Decl(subtree(1));
  }
  
  // 获取函数体语句列表
  List<Stmt> statements() const {
    return List<Stmt>(subtree(2));
  }
  
  // 静态工厂方法，创建 Def 对象
  static Def create(
      const SourceRange& range,
      const Ident& name,
      const Decl& decl,
      const List<Stmt>& stmts) {
    // 使用 Compound::create 创建 TK_DEF 类型的树节点，包含 name、decl 和 stmts
    return Def(Compound::create(TK_DEF, range, {name, decl, stmts}));
  }
};

// Property 结构体继承自 TreeView，表示属性
// 包含一个名称、一个获取方法和一个可选的设置方法
struct Property : public TreeView {
  // 构造函数，初始化时匹配 TK_PROP 类型的树节点
  explicit Property(const TreeRef& tree) : TreeView(tree) {
    tree->match(TK_PROP);
  }
  
  // 获取属性名称标识符
  Ident name() const {
    return Ident(subtree(0));
  }
  
  // 获取属性的获取方法定义
  Def getter() const {
    return Def(subtree(1));
  }
  
  // 获取属性的设置方法定义，可能为空
  Maybe<Def> setter() const {
    return Maybe<Def>(subtree(2));
  }
  
  // 静态工厂方法，创建 Property 对象
  static Property create(
      const SourceRange& range,
      const Ident& name,
      const Def& getter,
      const Maybe<Def>& setter) {
    return Property(Compound::create(TK_PROP, range, {name, getter, setter}));


注释：


// 返回一个 Property 对象，该对象由 Compound 类的 create 方法创建而来，
// 参数分别为 TK_PROP（属性类型）、range（范围）、以及一个包含 name、getter 和 setter 的初始化列表。
};

// 结构体 Assign 的前向声明
struct Assign;

// ClassDef 类的定义，继承自 TreeView 类
struct ClassDef : public TreeView {
  // 构造函数，接受一个 TreeRef 类型的参数，用于初始化基类 TreeView
  explicit ClassDef(const TreeRef& tree) : TreeView(tree) {
    // 匹配并确认树节点类型为 TK_CLASS_DEF
    tree->match(TK_CLASS_DEF);
  }
  // 移动语义的构造函数，接受一个右值引用的 TreeRef 类型参数
  explicit ClassDef(TreeRef&& tree) : TreeView(std::move(tree)) {
    // 匹配并确认树节点类型为 TK_CLASS_DEF
    tree_->match(TK_CLASS_DEF);
  }
  // 创建一个具有新名称的 ClassDef 对象，返回新的对象
  ClassDef withName(std::string new_name) const {
    // 创建一个新的标识符，范围与当前名称相同，但名称为 new_name
    auto new_ident = Ident::create(name().range(), std::move(new_name));
    // 调用 create 静态方法，使用当前对象的范围、新标识符、超类、主体创建新的 ClassDef 对象
    return create(range(), new_ident, superclass(), body());
  }
  // 返回当前 ClassDef 对象的名称标识符
  Ident name() const {
    return Ident(subtree(0));
  }
  // 返回当前 ClassDef 对象的可能存在的超类表达式
  Maybe<Expr> superclass() const {
    return Maybe<Expr>(subtree(1));
  }
  // 返回当前 ClassDef 对象的主体语句列表
  List<Stmt> body() const {
    return List<Stmt>(subtree(2));
  }
  // 返回当前 ClassDef 对象的可能存在的属性列表
  Maybe<List<Property>> properties() const {
    return Maybe<List<Property>>(subtree(3));
  }
  // 返回当前 ClassDef 对象的可能存在的赋值语句列表
  Maybe<List<Assign>> assigns() const {
    return Maybe<List<Assign>>(subtree(4));
  }
  // 静态方法，创建一个新的 ClassDef 对象
  static ClassDef create(
      const SourceRange& range,
      const Ident& name,
      const Maybe<Expr>& superclass,
      const List<Stmt>& body) {
    // 使用 Compound::create 方法创建一个 TK_CLASS_DEF 类型的复合对象，包含名称、超类、主体等参数
    return ClassDef(Compound::create(
        TK_CLASS_DEF,
        range,
        {name,
         superclass,
         body,
         Maybe<List<Property>>::create(range),
         Maybe<List<Assign>>::create(range)}));
  }
  // 静态方法，创建一个新的 ClassDef 对象，包含额外的属性和赋值语句列表
  static ClassDef create(
      const SourceRange& range,
      const Ident& name,
      const Maybe<Expr>& superclass,
      const List<Stmt>& body,
      const List<Property>& properties,
      const List<Assign>& assigns);
};

// TORCH_API 的声明，返回未解析的类属性名称的字符串向量
TORCH_API std::vector<std::string> getUnresolvedClassAttributes(
    const ClassDef& def);

////////////////////////////////////////////////////////////////////////////////
// Statements
////////////////////////////////////////////////////////////////////////////////

// 表达式语句 If 的定义，继承自 Stmt 类
struct If : public Stmt {
  // 显式构造函数，接受一个 TreeRef 类型的参数
  explicit If(const TreeRef& tree) : Stmt(tree) {
    // 匹配并确认树节点类型为 TK_IF
    tree_->match(TK_IF);
  }
  // 返回当前 If 语句对象的条件表达式
  Expr cond() const {
    return Expr(subtree(0));
  }
  // 返回当前 If 语句对象的真实分支语句列表
  List<Stmt> trueBranch() const {
    return List<Stmt>(subtree(1));
  }
  // 返回当前 If 语句对象的虚假分支语句列表
  List<Stmt> falseBranch() const {
    return List<Stmt>(subtree(2));
  }
  // 返回具有新分支语句的新 If 对象
  If withNewBranches(
      const List<Stmt>& true_branch,
      const List<Stmt>& false_branch) const {
    // 调用 create 静态方法创建一个新的 If 对象，使用当前对象的范围、条件、真实分支、虚假分支
    return create(range(), cond(), true_branch, false_branch);
  }
  // 静态方法，创建一个新的 If 对象
  static If create(
      const SourceRange& range,
      const Expr& cond,
      const List<Stmt>& true_branch,
      const List<Stmt>& false_branch) {
    // 使用 Compound::create 方法创建一个 TK_IF 类型的复合对象，包含条件、真实分支、虚假分支
    return If(
        Compound::create(TK_IF, range, {cond, true_branch, false_branch}));
  }
};

// 表达式语句 While 的定义，继承自 Stmt 类
struct While : public Stmt {
  // 显式构造函数，接受一个 TreeRef 类型的参数
  explicit While(const TreeRef& tree) : Stmt(tree) {
    // 匹配并确认树节点类型为 TK_WHILE
    tree_->match(TK_WHILE);
  }
  // 返回当前 While 循环语句对象的条件表达式
  Expr cond() const {
    return Expr(subtree(0));
  }
  // 返回当前 While 循环语句对象的主体语句列表
  List<Stmt> body() const {
    return List<Stmt>(subtree(1));
  }
  // 静态方法，创建一个新的 While 循环对象
  static While create(
      const SourceRange& range,
      const Expr& cond,
      const List<Stmt>& body) {
    // 使用 Compound::create 方法创建一个 TK_WHILE 类型的复合对象，包含条件和主体
    return While(Compound::create(TK_WHILE, range, {cond, body}));
  }
};

// 表达式语句 For 的定义，继承自 Stmt 类
struct For : public Stmt {
  // 显式构造函数，接受一个 TreeRef 类型的参数
  explicit For(const TreeRef& tree) : Stmt(tree) {
    // 匹配并确认树节点类型为 TK_FOR
    tree->match(TK_FOR);
  }
  // 返回当前 For 循环语句对象的目标表达式列表
  List<Expr> targets() const {
    // 返回包含从索引0开始的子树的表达式列表
    return List<Expr>(subtree(0));
  }
  // 返回包含从索引1开始的子树的表达式列表
  List<Expr> itrs() const {
    return List<Expr>(subtree(1));
  }
  // 返回包含从索引2开始的子树的语句列表
  List<Stmt> body() const {
    return List<Stmt>(subtree(2));
  }
  // 创建并返回一个 For 对象，该对象包含给定的范围、目标表达式列表、迭代器表达式列表和语句列表
  static For create(
      const SourceRange& range,
      const List<Expr>& targets,
      const List<Expr>& itrs,
      const List<Stmt>& body) {
    return For(Compound::create(TK_FOR, range, {targets, itrs, body}));
  }
};

// TODO: supports only single comprehension for now
// 表示列表推导式的结构，继承自表达式类
struct ListComp : public Expr {
  // 构造函数，接受一个树节点引用作为参数
  explicit ListComp(const TreeRef& tree) : Expr(tree) {
    // 匹配树节点的类型为列表推导式
    tree->match(TK_LIST_COMP);
  }
  // 返回列表推导式的元素表达式
  Expr elt() const {
    return Expr(subtree(0));
  }
  // 返回列表推导式的目标表达式
  Expr target() const {
    return Expr(subtree(1));
  }
  // 返回列表推导式的迭代表达式
  Expr iter() const {
    return Expr(subtree(2));
  }
  // 静态方法，创建列表推导式对象
  // 参数包括范围、元素、目标、迭代表达式
  static ListComp create(
      const SourceRange& range,
      const Expr& elt,
      const Expr& target,
      const Expr& iter) {
    return ListComp(Compound::create(TK_LIST_COMP, range, {elt, target, iter}));
  }
};

// TODO: supports only single comprehension for now
// 表示字典推导式的结构，继承自表达式类
struct DictComp : public Expr {
  // 构造函数，接受一个树节点引用作为参数
  explicit DictComp(const TreeRef& tree) : Expr(tree) {
    // 匹配树节点的类型为字典推导式
    tree->match(TK_DICT_COMP);
  }
  // 返回字典推导式的键表达式
  Expr key() const {
    return Expr(subtree(0));
  }
  // 返回字典推导式的值表达式
  Expr value() const {
    return Expr(subtree(1));
  }
  // 返回字典推导式的目标表达式
  Expr target() const {
    return Expr(subtree(2));
  }
  // 返回字典推导式的迭代表达式
  Expr iter() const {
    return Expr(subtree(3));
  }
  // 静态方法，创建字典推导式对象
  // 参数包括范围、键、值、目标、迭代表达式
  static DictComp create(
      const SourceRange& range,
      const Expr& key,
      const Expr& value,
      const Expr& target,
      const Expr& iter) {
    return DictComp(
        Compound::create(TK_DICT_COMP, range, {key, value, target, iter}));
  }
};

// 表示全局声明的结构，继承自语句类
struct Global : public Stmt {
  // 构造函数，接受一个树节点引用作为参数
  explicit Global(const TreeRef& tree) : Stmt(tree) {
    // 匹配树节点的类型为全局声明
    tree_->match(TK_GLOBAL);
  }
  // 返回全局声明的名称列表
  List<Ident> names() {
    return List<Ident>(subtree(0));
  }
  // 静态方法，创建全局声明对象
  // 参数包括范围和名称列表
  static Global create(const SourceRange& range, const List<Ident>& names) {
    return Global(Compound::create(TK_GLOBAL, range, {names}));
  }
};

// 表示增强赋值操作符的类型，继承自树视图类
struct AugAssignKind : public TreeView {
  // 构造函数，接受一个树节点引用作为参数
  explicit AugAssignKind(const TreeRef& tree) : TreeView(tree) {
    // 根据树节点的类型判断是否是有效的增强赋值操作符
    switch (tree->kind()) {
      case '+':
      case '-':
      case '*':
      case '/':
      case '%':
      case '|':
      case '&':
      case '^':
      case TK_POW:
      case TK_LSHIFT:
      case TK_RSHIFT:
        return;
      default:
        throw ErrorReport(tree) << "is not a valid AugAssignKind";
    }
  }
};

// 表示增强赋值语句的结构，继承自语句类
struct AugAssign : public Stmt {
  // 构造函数，接受一个树节点引用作为参数
  explicit AugAssign(const TreeRef& tree) : Stmt(tree) {
    // 匹配树节点的类型为增强赋值语句
    tree_->match(TK_AUG_ASSIGN);
  }
  // 静态方法，创建增强赋值语句对象
  // 参数包括范围、左操作数、增强赋值操作符、右操作数
  static AugAssign create(
      const SourceRange& range,
      const Expr& lhs,
      const AugAssignKind& aug_op,
      const Expr& rhs) {
    return AugAssign(
        Compound::create(TK_AUG_ASSIGN, range, {lhs, aug_op, rhs}));
  }
  // 返回增强赋值语句的左操作数表达式
  Expr lhs() const {
    return Expr(subtree(0));
  }
  // 返回增强赋值语句的增强赋值操作符的类型
  int aug_op() const {
    return subtree(1)->kind();
  }
  // 返回增强赋值语句的右操作数表达式
  Expr rhs() const {
    return Expr(subtree(2));
  }
};

// 表示赋值语句的结构，继承自语句类
struct Assign : public Stmt {
  // 构造函数，接受一个树节点引用作为参数
  explicit Assign(const TreeRef& tree) : Stmt(tree) {
    // 匹配树节点的类型为赋值语句
    tree_->match(TK_ASSIGN);
  }
  // 静态方法，创建赋值语句对象
  // 参数包括范围、左操作数列表、右操作数（可能为空）、类型（可能为空）
  static Assign create(
      const SourceRange& range,
      const List<Expr>& lhs,
      const Maybe<Expr>& rhs,
      const Maybe<Expr>& type) {
    return Assign(Compound::create(TK_ASSIGN, range, {lhs, rhs, type}));
  }

  // 返回赋值语句的左操作数列表
  List<Expr> lhs_list() const {
    # 返回一个包含从树的根节点开始的子树的表达式列表
    return List<Expr>(subtree(0));
  }

  # 返回左手边表达式（lhs）
  Expr lhs() const {
    # 获取左手边列表的引用
    const auto& li = lhs_list();
    # 使用内部断言确保左手边列表大小为1
    TORCH_INTERNAL_ASSERT(li.size() == 1);
    # 返回左手边列表的第一个元素作为左手边表达式
    return *li.begin();
  }

  # 返回右手边表达式（rhs），可能为空
  Maybe<Expr> rhs() const {
    # 返回一个可能包含从树的第二个子树开始的表达式
    return Maybe<Expr>(subtree(1));
  }

  # 返回类型表达式，可能为空
  Maybe<Expr> type() const {
    # 返回一个可能包含从树的第三个子树开始的表达式
    return Maybe<Expr>(subtree(2));
  }
};

// 表示一个返回语句，继承自Stmt类
struct Return : public Stmt {
  explicit Return(const TreeRef& tree) : Stmt(tree) {
    // 匹配树节点，确认为TK_RETURN类型
    tree_->match(TK_RETURN);
  }
  
  // 返回表达式的方法
  Expr expr() const {
    return Expr(subtree(0));
  }
  
  // 创建一个Return对象的静态方法，传入源代码范围和表达式作为参数
  static Return create(const SourceRange& range, const Expr& value) {
    return Return(Compound::create(TK_RETURN, range, {value}));
  }
};

// 表示一个抛出异常的语句，继承自Stmt类
struct Raise : public Stmt {
  explicit Raise(const TreeRef& tree) : Stmt(tree) {
    // 匹配树节点，确认为TK_RAISE类型
    tree_->match(TK_RAISE);
  }
  
  // 返回抛出异常时的表达式
  Expr expr() const {
    return Expr(subtree(0));
  }
  
  // 创建一个Raise对象的静态方法，传入源代码范围和异常表达式作为参数
  static Raise create(const SourceRange& range, const Expr& expr) {
    return Raise(Compound::create(TK_RAISE, range, {expr}));
  }
};

// 表示一个断言语句，继承自Stmt类
struct Assert : public Stmt {
  explicit Assert(const TreeRef& tree) : Stmt(tree) {
    // 匹配树节点，确认为TK_ASSERT类型
    tree_->match(TK_ASSERT);
  }
  
  // 返回断言的测试表达式
  Expr test() const {
    return Expr(subtree(0));
  }
  
  // 返回可能存在的断言消息表达式
  Maybe<Expr> msg() const {
    return Maybe<Expr>(subtree(1));
  }
  
  // 创建一个Assert对象的静态方法，传入源代码范围、测试表达式和可选的消息表达式作为参数
  static Assert create(
      const SourceRange& range,
      const Expr& test,
      const Maybe<Expr>& msg) {
    return Assert(Compound::create(TK_ASSERT, range, {test, msg}));
  }
};

// 表示一个空语句（pass语句），继承自Stmt类
struct Pass : public Stmt {
  explicit Pass(const TreeRef& tree) : Stmt(tree) {
    // 匹配树节点，确认为TK_PASS类型
    tree_->match(TK_PASS);
  }
  
  // 创建一个Pass对象的静态方法，传入源代码范围作为参数
  static Pass create(const SourceRange& range) {
    return Pass(Compound::create(TK_PASS, range, {}));
  }
};

// 表示省略号（...）的表达式，继承自Expr类
struct Dots : public Expr {
  explicit Dots(const TreeRef& tree) : Expr(tree) {
    // 匹配树节点，确认为TK_DOTS类型
    tree_->match(TK_DOTS);
  }
  
  // 创建一个Dots对象的静态方法，传入源代码范围作为参数
  static Dots create(const SourceRange& range) {
    return Dots(Compound::create(TK_DOTS, range, {}));
  }
};

// 表示break语句，继承自Stmt类
struct Break : public Stmt {
  explicit Break(const TreeRef& tree) : Stmt(tree) {
    // 匹配树节点，确认为TK_BREAK类型
    tree_->match(TK_BREAK);
  }
  
  // 创建一个Break对象的静态方法，传入源代码范围作为参数
  static Break create(const SourceRange& range) {
    return Break(Compound::create(TK_BREAK, range, {}));
  }
};

// 表示continue语句，继承自Stmt类
struct Continue : public Stmt {
  explicit Continue(const TreeRef& tree) : Stmt(tree) {
    // 匹配树节点，确认为TK_CONTINUE类型
    tree_->match(TK_CONTINUE);
  }
  
  // 创建一个Continue对象的静态方法，传入源代码范围作为参数
  static Continue create(const SourceRange& range) {
    return Continue(Compound::create(TK_CONTINUE, range, {}));
  }
};

// 表示一个表达式语句，继承自Stmt类
struct ExprStmt : public Stmt {
  explicit ExprStmt(const TreeRef& tree) : Stmt(tree) {
    // 匹配树节点，确认为TK_EXPR_STMT类型
    tree_->match(TK_EXPR_STMT);
  }
  
  // 返回该表达式语句的表达式
  Expr expr() {
    return Expr(subtree(0));
  }
  
  // 创建一个ExprStmt对象的静态方法，传入源代码范围和表达式作为参数
  static ExprStmt create(const SourceRange& range, const Expr& list) {
    return ExprStmt(Compound::create(TK_EXPR_STMT, range, {list}));
  }
};
    switch (tree->kind()) {
      case TK_AND:
      case TK_OR:
      case '<':
      case '>':
      case TK_IS:
      case TK_ISNOT:
      case TK_EQ:
      case TK_LE:
      case TK_GE:
      case TK_NE:
      case '+':
      case '*':
      case '/':
      case '-':
      case '@':
      case TK_POW:
      case TK_LSHIFT:
      case TK_RSHIFT:
      case '%':
      case '&':
      case '^':
      case '|':
      case TK_FLOOR_DIV:
      case TK_IN:
        // 检查二元操作符的情况，确保有且仅有两个子树
        if (tree->trees().size() != 2)
          // 如果不符合预期，抛出错误报告
          throw ErrorReport(tree)
              << "BinOp expected 2 subtrees, found " << tree->trees().size();
        // 如果符合预期，直接返回
        return;
      default:
        // 如果不是有效的二元操作符，抛出错误报告
        throw ErrorReport(tree)
            << kindToString(tree->kind()) << " is not a valid BinOp";
    }
  }
  // 获取左子表达式
  Expr lhs() const {
    return Expr(subtree(0));
  }
  // 获取右子表达式
  Expr rhs() const {
    return Expr(subtree(1));
  }
  // 静态方法：创建一个二元操作符表达式
  static BinOp create(
      const SourceRange& range,
      int kind,
      const Expr& lhs,
      const Expr& rhs) {
    // 调用 Compound 类的静态方法创建复合对象，返回 BinOp 对象
    return BinOp(Compound::create(kind, range, {lhs, rhs}));
  }
};

// 表示一元操作符的结构体，继承自表达式类 Expr
struct UnaryOp : public Expr {
  // 构造函数，接受一个树形结构的参数 tree
  explicit UnaryOp(const TreeRef& tree) : Expr(tree) {
    // 根据树的种类进行不同的操作
    switch (tree->kind()) {
      // 对于负号、按位非、逻辑非操作符
      case TK_UNARY_MINUS:
      case '~':
      case TK_NOT:
        // 检查子树数量是否为1，否则抛出异常报告
        if (tree->trees().size() != 1)
          throw ErrorReport(tree)
              << "UnaryOp expected 1 subtree, found " << tree->trees().size();
        return;
      // 对于其他种类的操作符，抛出错误报告
      default:
        throw ErrorReport(tree)
            << kindToString(tree->kind()) << " is not a valid UnaryOp";
    }
  }
  // 静态方法，创建 UnaryOp 对象，接受源代码范围、操作符种类、表达式作为参数
  static UnaryOp create(const SourceRange& range, int kind, const Expr& expr) {
    // 调用 Compound 的静态方法创建复合对象，并返回 UnaryOp 对象
    return UnaryOp(Compound::create(kind, range, {expr}));
  }
};

// 表示常量表达式的结构体，继承自表达式类 Expr
struct Const : public Expr {
  // 构造函数，接受一个树形结构的参数 tree
  explicit Const(const TreeRef& tree) : Expr(tree) {
    // 匹配并确保树的种类为 TK_CONST，且子树数量为1
    tree_->matchNumSubtrees(TK_CONST, 1);
  }
  // 判断常量是否为浮点数
  bool isFloatingPoint() const {
    // 如果是复数则不是浮点数
    if (isComplex())
      return false;
    // 判断常量字符串是否为无穷大或包含浮点数表示字符
    bool is_inf = subtree(0)->stringValue() == "inf";
    return is_inf ||
        subtree(0)->stringValue().find_first_of(".eE") != std::string::npos;
  }
  // 判断常量是否为整数
  bool isIntegral() const {
    // 不是浮点数且不是复数则为整数
    return !isFloatingPoint() && !isComplex();
  }
  // 判断常量是否为复数
  bool isComplex() const {
    // 判断常量字符串是否包含虚数单位字符 'j'
    return subtree(0)->stringValue().find_first_of('j') != std::string::npos;
  }
  // 将常量转换为整数
  int64_t asIntegral() const {
    try {
      // 使用 std::stoll 将常量字符串转换为整数类型，处理范围溢出异常
      return std::stoll(subtree(0)->stringValue(), /*__idx=*/0, /*base=*/0);
    } catch (const std::out_of_range&) {
      // 报告整数常量超出范围的错误
      throw ErrorReport(range()) << "Integral constant out of range "
                                    "(must fit in a signed 64 bit integer)";
    }
  }
  // 将常量转换为浮点数
  double asFloatingPoint() const {
    // 调用 torch::jit::strtod_c 将常量字符串转换为双精度浮点数
    // 注意：Android 版本的 strtod_c 不接受 nullptr 作为参数
    char* dummy;
    return torch::jit::strtod_c(subtree(0)->stringValue().c_str(), &dummy);
  }
  // 将常量转换为复数
  c10::complex<double> asComplex() const {
    char* dummy;
    auto str = subtree(0)->stringValue();
    // 处理复数常量的转换，实部为0时只解析虚部
    auto imag =
        torch::jit::strtod_c(str.substr(0, str.size() - 1).c_str(), &dummy);
    return c10::complex<double>(0, imag);
  }
  // 获取常量的字符串表示
  const std::string& text() const {
    return subtree(0)->stringValue();
  }
  // 静态方法，创建常量表达式对象，接受源代码范围和常量值作为参数
  static Const create(const SourceRange& range, const std::string& value) {
    // 调用 Compound 的静态方法创建复合对象，并返回 Const 对象
    return Const(Compound::create(TK_CONST, range, {String::create(value)}));
  }
};

// 表示字符串字面量表达式的结构体，继承自表达式类 Expr
struct StringLiteral : public Expr {
  // 构造函数，接受一个树形结构的参数 tree
  explicit StringLiteral(const TreeRef& tree) : Expr(tree) {
    // 匹配并确保树的种类为 TK_STRINGLITERAL，且子树数量为1
    tree_->matchNumSubtrees(TK_STRINGLITERAL, 1);
  }
  // 获取字符串字面量的文本内容
  const std::string& text() const {
    return subtree(0)->stringValue();
  }
  // 静态方法，创建字符串字面量表达式对象，接受源代码范围和字符串值作为参数
  static StringLiteral create(
      const SourceRange& range,
      const std::string& value) {
    return StringLiteral(
        Compound::create(TK_STRINGLITERAL, range, {String::create(value)}));
  }



    创建一个 StringLiteral 对象，并返回该对象。
    这里使用了 Compound::create 方法来创建一个复合对象，类型为 TK_STRINGLITERAL，使用给定的范围 range 和包含一个 String::create(value) 的集合作为参数。
    返回的 StringLiteral 对象即为函数的返回值。
struct Apply : public Expr {
  // 构造函数，接受一个树节点作为参数，并将其传递给基类Expr的构造函数
  explicit Apply(const TreeRef& tree) : Expr(tree) {
    // 匹配树节点的类型是否为TK_APPLY
    tree_->match(TK_APPLY);
  }
  // 返回调用表达式的表达式
  Expr callee() const {
    return Expr(subtree(0));
  }
  // 返回应用表达式的输入参数列表
  List<Expr> inputs() const {
    return List<Expr>(subtree(1));
  }
  // 返回应用表达式的属性列表
  List<Attribute> attributes() const {
    return List<Attribute>(subtree(2));
  }
  // 创建一个新的Apply对象
  static Apply create(
      const SourceRange& range,
      const Expr& callee,
      const List<Expr>& inputs,
      const List<Attribute>& attributes) {
    // 使用Compound::create创建一个复合树节点，类型为TK_APPLY
    return Apply(
        Compound::create(TK_APPLY, range, {callee, inputs, attributes}));
  }
};

struct Select : public Expr {
  // 构造函数，接受一个树节点作为参数，并将其传递给基类Expr的构造函数
  explicit Select(const TreeRef& tree) : Expr(tree) {
    // 匹配树节点的类型是否为'.'
    tree_->match('.');
  }
  // 返回选择表达式的值
  Expr value() const {
    return Expr(subtree(0));
  }
  // 返回选择表达式的选择器
  Ident selector() const {
    return Ident(subtree(1));
  }
  // 创建一个新的Select对象
  static Select create(
      const SourceRange& range,
      const Expr& value,
      const Ident& selector) {
    // 使用Compound::create创建一个复合树节点，类型为'.'
    return Select(Compound::create('.', range, {value, selector}));
  }
};

struct SliceExpr : public Expr {
  // 构造函数，接受一个树节点作为参数，并将其传递给基类Expr的构造函数
  explicit SliceExpr(const TreeRef& tree) : Expr(tree) {
    // 匹配树节点的类型是否为TK_SLICE_EXPR
    tree_->match(TK_SLICE_EXPR);
  }
  // 返回切片表达式的起始位置
  Maybe<Expr> start() const {
    return Maybe<Expr>(subtree(0));
  }
  // 返回切片表达式的结束位置
  Maybe<Expr> end() const {
    return Maybe<Expr>(subtree(1));
  }
  // 返回切片表达式的步长
  Maybe<Expr> step() const {
    return Maybe<Expr>(subtree(2));
  }
  // 返回切片表达式的起始位置，如果不存在则返回指定的替代值
  Expr startOr(int64_t alternative) const {
    const auto startOption = start();
    return startOption.present() ? startOption.get() : createInt(alternative);
  }
  // 返回切片表达式的结束位置，如果不存在则返回指定的替代值
  Expr endOr(int64_t alternative) const {
    const auto endOption = end();
    return endOption.present() ? endOption.get() : createInt(alternative);
  }
  // 返回切片表达式的步长，如果不存在则返回指定的替代值
  Expr stepOr(int64_t alternative) const {
    const auto stepOption = step();
    return stepOption.present() ? stepOption.get() : createInt(alternative);
  }
  // 创建一个新的SliceExpr对象
  static SliceExpr create(
      const SourceRange& range,
      const Maybe<Expr>& start,
      const Maybe<Expr>& end,
      const Maybe<Expr>& step) {
    // 使用Compound::create创建一个复合树节点，类型为TK_SLICE_EXPR
    return SliceExpr(
        Compound::create(TK_SLICE_EXPR, range, {start, end, step}));
  }

 private:
  // 创建一个表示整数的表达式对象
  Expr createInt(int64_t value) const {
    return Expr(Const::create(range(), std::to_string(value)));
  }
};

struct Subscript : public Expr {
  // 构造函数，接受一个树节点作为参数，并将其传递给基类Expr的构造函数
  explicit Subscript(const TreeRef& tree) : Expr(tree) {
    // 匹配树节点的类型是否为TK_SUBSCRIPT
    tree_->match(TK_SUBSCRIPT);
  }
  // 返回下标表达式的值
  Expr value() const {
    return Expr(subtree(0));
  }
  // 返回下标表达式的子表达式列表
  List<Expr> subscript_exprs() const {
    return List<Expr>(subtree(1));
  }
  // 创建一个新的Subscript对象
  static Subscript create(
      const SourceRange& range,
      const Expr& value,
      const List<Expr>& subscript_exprs) {
    // 计算整个子表达式的范围
    auto whole_range = SourceRange(
        range.source(), range.start(), subscript_exprs.range().end() + 1);
    // 使用Compound::create创建一个复合树节点，类型为TK_SUBSCRIPT
    return Subscript(
        Compound::create(TK_SUBSCRIPT, whole_range, {value, subscript_exprs}));
  }
};

struct Var : public Expr {
  // 构造函数，接受一个树节点作为参数，并将其传递给基类Expr的构造函数
  explicit Var(const TreeRef& tree) : Expr(tree) {
    // 匹配树节点的类型是否为TK_VAR
    tree_->match(TK_VAR);
  };
  // 返回变量名标识符
  Ident name() const {
    // 返回树节点的标识符子节点
    // （假设在Expr类中，subtree(0)返回第一个子树节点）
    return Ident(subtree(0));
  }
};
    return Ident(subtree(0));
  }
  // 创建一个标识符对象，并返回
  static Var create(const SourceRange& range, const Ident& name) {
    // 使用提供的范围和标识符名称创建一个包含单个标识符的复合对象
    return Var(Compound::create(TK_VAR, range, {name}));
  }
};

// WithItem 表示在 WithStmt 中使用的项。
struct WithItem : public Expr {
  explicit WithItem(const TreeRef& tree) : Expr(tree) {
    tree_->match(TK_WITH_ITEM);  // 匹配并初始化 WithItem 对象
  }

  // 返回目标表达式
  Expr target() const {
    return Expr(subtree(0));
  }

  // 返回可能存在的变量
  Maybe<Var> var() const {
    return Maybe<Var>(subtree(1));
  }

  // 创建 WithItem 对象
  static WithItem create(
      const SourceRange& range,
      const Expr& target,
      const Maybe<Var>& var) {
    return WithItem(Compound::create(TK_WITH_ITEM, range, {target, var}));
  }
};

// With 表示一个 with 语句，包含一组 WithItem 和语句体。
struct With : public Stmt {
  explicit With(const TreeRef& tree) : Stmt(tree) {
    tree_->match(TK_WITH);  // 匹配并初始化 With 对象
  }

  // 返回 with 语句的目标列表
  List<WithItem> targets() const {
    return List<WithItem>(subtree(0));
  }

  // 返回 with 语句的主体语句列表
  List<Stmt> body() const {
    return List<Stmt>(subtree(1));
  }

  // 创建 With 对象
  static With create(
      const SourceRange& range,
      const List<WithItem>& targets,
      const List<Stmt>& body) {
    return With(Compound::create(TK_WITH, range, {targets, body}));
  }
};

// TernaryIf 表示三元条件表达式。
struct TernaryIf : public Expr {
  explicit TernaryIf(const TreeRef& tree) : Expr(tree) {
    tree_->matchNumSubtrees(TK_IF_EXPR, 3);  // 匹配并初始化 TernaryIf 对象
  };

  // 返回条件表达式
  Expr cond() const {
    return Expr(subtree(0));
  }

  // 返回条件为真时的表达式
  Expr true_expr() const {
    return Expr(subtree(1));
  }

  // 返回条件为假时的表达式
  Expr false_expr() const {
    return Expr(subtree(2));
  }

  // 创建 TernaryIf 对象
  static TernaryIf create(
      const SourceRange& range,
      const Expr& cond,
      const Expr& true_expr,
      const Expr& false_expr) {
    return TernaryIf(
        Compound::create(TK_IF_EXPR, range, {cond, true_expr, false_expr}));
  };
};

// ListLiteral 表示列表字面量。
struct ListLiteral : public Expr {
  explicit ListLiteral(const TreeRef& tree) : Expr(tree) {
    tree_->match(TK_LIST_LITERAL);  // 匹配并初始化 ListLiteral 对象
  }

  // 返回列表字面量的元素列表
  List<Expr> inputs() const {
    return subtree(0);
  }

  // 创建 ListLiteral 对象
  static ListLiteral create(
      const SourceRange& range,
      const List<Expr>& inputs) {
    return ListLiteral(Compound::create(TK_LIST_LITERAL, range, {inputs}));
  }
};

// TupleLiteral 表示元组字面量。
struct TupleLiteral : public Expr {
  explicit TupleLiteral(const TreeRef& tree) : Expr(tree) {
    tree_->match(TK_TUPLE_LITERAL);  // 匹配并初始化 TupleLiteral 对象
  }

  // 返回元组字面量的元素列表
  List<Expr> inputs() const {
    return subtree(0);
  }

  // 创建 TupleLiteral 对象
  static TupleLiteral create(
      const SourceRange& range,
      const List<Expr>& inputs) {
    return TupleLiteral(Compound::create(TK_TUPLE_LITERAL, range, {inputs}));
  }
};

// DictLiteral 表示字典字面量。
struct DictLiteral : public Expr {
  explicit DictLiteral(const TreeRef& tree) : Expr(tree) {
    tree_->match(TK_DICT_LITERAL);  // 匹配并初始化 DictLiteral 对象
  }

  // 返回字典字面量的键列表
  List<Expr> key_inputs() const {
    return subtree(0);
  }

  // 返回字典字面量的值列表
  List<Expr> value_inputs() const {
    return subtree(1);
  }

  // 创建 DictLiteral 对象
  static DictLiteral create(
      const SourceRange& range,
      const List<Expr>& keys,
      const List<Expr>& values) {
    return DictLiteral(
        Compound::create(TK_DICT_LITERAL, range, {keys, values}));
  }
};

// Starred 表示星号表达式。
struct Starred : public Expr {
  explicit Starred(const TreeRef& tree) : Expr(tree) {
    tree_->match(TK_STARRED);  // 匹配并初始化 Starred 对象
  }

  // 返回星号表达式的表达式
  Expr expr() const {
    return Expr(subtree(0));
  }



# 返回一个表达式对象，其子树的根节点为索引为0的子树
return Expr(subtree(0));



  static Starred create(const SourceRange& range, const Expr& expr) {
    return Starred(Compound::create(TK_STARRED, range, {expr}));
  }



# 创建一个 Starred 对象，并使用给定的范围和表达式
static Starred create(const SourceRange& range, const Expr& expr) {
    # 调用 Compound 类的静态方法 create，创建一个复合对象，
    # 类型为 TK_STARRED，范围为 range，包含一个表达式的列表 [expr]
    return Starred(Compound::create(TK_STARRED, range, {expr}));
}
};

// 结构体 Delete，继承自 Stmt
struct Delete : public Stmt {
  // 构造函数，接受一个树节点作为参数，并调用基类的构造函数
  explicit Delete(const TreeRef& tree) : Stmt(tree) {
    // 匹配树节点类型为 TK_DELETE
    tree_->match(TK_DELETE);
  }
  // 返回删除语句的目标列表
  List<Expr> targets() const {
    return subtree(0);
  }
  // 创建 Delete 对象的静态方法，接受源范围和目标列表作为参数
  static Delete create(const SourceRange& range, const List<Expr>& targets) {
    // 调用 Compound 的静态方法创建一个包含 TK_DELETE 类型和目标列表的复合树节点
    return Delete(Compound::create(TK_DELETE, range, {targets}));
  }
};

/*
 * NOTE: 将 PEP 604 union 转换为等效的 union 类型
 *
 * NOTE: Union[int, float] 解析为:
 * <EXPR> expr:(subscript
 *  (variable (ident Union))
 *  (list
 *    (variable (ident int))
 *    (variable (ident float))))
 * <KIND> subscript
 *
 * NOTE: (int | float) 解析为:
 * <EXPR> expr:(|
 *  (variable (ident int))
 *  (variable (ident float)))
 * <KIND> |
 */

// 内联函数，将可能嵌套的 PEP 604 union 表达式展平为一维表达式列表
inline void _flatten_pep604_union(
    const torch::jit::Expr& node,
    std::vector<torch::jit::Expr>* result) {
  // 如果节点类型为 '|'，则进行二元操作展开
  if (node.kind() == '|') {
    auto as_binop = torch::jit::BinOp(node);
    // 递归展开左右子节点
    _flatten_pep604_union(as_binop.lhs(), result);
    _flatten_pep604_union(as_binop.rhs(), result);
  } else {
    // 将节点添加到结果向量中
    result->push_back(node);
  }
}

// 返回 PEP 604 union 成员的表达式列表
inline std::vector<Expr> get_pep604_union_members(const Expr& node) {
  std::vector<Expr> result;
  // 调用内部函数，将 PEP 604 union 表达式展平
  _flatten_pep604_union(node, &result);
  return result;
}

// 将 PEP 604 union 转换为传统的 union 类型表达式
// 例如，((x | y) | z) 被转换为 Union[x, y, z]
inline Expr pep604union_to_union(const Expr& expr) {
  // 如果不是 PEP 604 union，直接返回原表达式
  if (expr.kind() != '|')
    return expr;

  // 支持多于两个操作数的 union ((x|y)|z)，需要递归展平 | 表达式树
  auto members = get_pep604_union_members(expr);
  // 创建 Subscript 表达式，表示 Union 类型
  auto synthesised_union = Subscript::create(
      expr.range(),
      Var::create(expr.range(), Ident::create(expr.range(), "Union")),
      List<Expr>::create(expr.range(), members));
  return std::move(synthesised_union);
}

} // namespace jit
} // namespace torch

namespace std {

// 模板特化，定义 torch::jit::ListIterator<T> 的迭代器特性
template <typename T>
struct iterator_traits<torch::jit::ListIterator<T>>
    : std::iterator_traits<torch::jit::TreeList::const_iterator> {};

} // namespace std
```