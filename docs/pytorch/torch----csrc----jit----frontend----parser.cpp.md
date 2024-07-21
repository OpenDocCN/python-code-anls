# `.\pytorch\torch\csrc\jit\frontend\parser.cpp`

```py
// 包含 Torch 的 JIT 前端解析器头文件
#include <torch/csrc/jit/frontend/parser.h>

// 包含 C10 的 Optional 实用工具头文件
#include <c10/util/Optional.h>
// 包含 Torch 的 JIT 前端词法分析器头文件
#include <torch/csrc/jit/frontend/lexer.h>
// 包含 Torch 的 JIT 前端解析字符串字面量头文件
#include <torch/csrc/jit/frontend/parse_string_literal.h>
// 包含 Torch 的 JIT 前端树结构头文件
#include <torch/csrc/jit/frontend/tree.h>
// 包含 Torch 的 JIT 前端树视图头文件
#include <torch/csrc/jit/frontend/tree_views.h>

// Torch 的 JIT 命名空间
namespace torch::jit {

// 合并来自类型注释的类型到声明中
Decl mergeTypesFromTypeComment(
    const Decl& decl,
    const Decl& type_annotation_decl,
    bool is_method) {
  // 计算预期的注释数量
  auto expected_num_annotations = decl.params().size();
  if (is_method) {
    // 对于方法，排除 `self` 参数
    expected_num_annotations -= 1;
  }
  // 检查注释数量是否与预期参数数量匹配
  if (expected_num_annotations != type_annotation_decl.params().size()) {
    throw ErrorReport(decl.range())
        << "Number of type annotations ("
        << type_annotation_decl.params().size()
        << ") did not match the number of "
        << (is_method ? "method" : "function") << " parameters ("
        << expected_num_annotations << ")";
  }
  // 备份原始参数列表
  auto old = decl.params();
  auto _new = type_annotation_decl.params();
  // 合并签名标识和范围与注释类型
  std::vector<Param> new_params;
  size_t i = is_method ? 1 : 0;
  size_t j = 0;
  if (is_method) {
    new_params.push_back(old[0]);
  }
  // 逐个合并参数类型到新参数列表中
  for (; i < decl.params().size(); ++i, ++j) {
    new_params.emplace_back(old[i].withType(_new[j].type()));
  }
  // 创建新的声明对象并返回
  return Decl::create(
      decl.range(),
      List<Param>::create(decl.range(), new_params),
      type_annotation_decl.return_type());
}

// 解析器实现结构体
struct ParserImpl {
  // 解析器实现结构体的构造函数，接受共享指针指向的源对象
  explicit ParserImpl(const std::shared_ptr<Source>& source)
      : L(source), shared(sharedParserData()) {}

  // 解析标识符
  Ident parseIdent() {
    auto t = L.expect(TK_IDENT);
    // 每当解析具有 TreeView 类型的内容时，总是使用其 create 方法，
    // 以便访问器和复合树的构造函数位于相同位置。
    return Ident::create(t.range, t.text());
  }

  // 创建应用表达式
  TreeRef createApply(const Expr& expr) {
    TreeList attributes;
    auto range = L.cur().range;
    TreeList inputs;
    // 解析参数和属性
    parseArguments(inputs, attributes);
    // 创建并返回应用表达式对象
    return Apply::create(
        range,
        expr,
        List<Expr>(makeList(range, std::move(inputs))),
        List<Attribute>(makeList(range, std::move(attributes))));
  }

  // 判断是否跟随元组的表达式
  static bool followsTuple(int kind) {
    switch (kind) {
      // 以下情况表示表达式后可以跟随元组
      case TK_PLUS_EQ:
      case TK_MINUS_EQ:
      case TK_TIMES_EQ:
      case TK_DIV_EQ:
      case TK_MOD_EQ:
      case TK_BIT_OR_EQ:
      case TK_BIT_AND_EQ:
      case TK_BIT_XOR_EQ:
      case TK_LSHIFT_EQ:
      case TK_RSHIFT_EQ:
      case TK_POW_EQ:
      case TK_NEWLINE:
      case '=':
      case ')':
        return true;
      // 其他情况不跟随元组
      default:
        return false;
    }
  }

  // 解析表达式或表达式元组
  Expr parseExpOrExpTuple() {
    auto prefix = parseExp();
    // ...
    // 如果当前 token 是逗号 ','
    if (L.cur().kind == ',') {
      // 创建表达式向量，并加入第一个前缀表达式
      std::vector<Expr> exprs = {prefix};
      // 循环读取逗号后的表达式
      while (L.nextIf(',')) {
        // 如果下一个 token 是元组开始的标记，则退出循环
        if (followsTuple(L.cur().kind))
          break;
        // 解析并加入下一个表达式到向量中
        exprs.push_back(parseExp());
      }
      // 创建表达式列表对象
      auto list = List<Expr>::create(prefix.range(), exprs);
      // 创建元组字面量对象
      prefix = TupleLiteral::create(list.range(), list);
    }
    // 返回处理后的前缀表达式
    return prefix;
  }
  // 处理不是一元或二元表达式且优先级高于它们的表达式，如 a 1.0 或 a(4)
  TreeRef parseBaseExp() {
    TreeRef prefix;
    // 无限循环，直到遇到 break
    while (true) {
      // 如果下一个 token 是点号 '.'
      if (L.nextIf('.')) {
        // 解析标识符并创建选择器对象
        const auto name = parseIdent();
        prefix = Select::create(name.range(), Expr(prefix), Ident(name));
      } else if (L.cur().kind == '(') {
        // 如果当前 token 是左括号 '('，创建应用对象
        prefix = createApply(Expr(prefix));
      } else if (L.cur().kind == '[') {
        // 如果当前 token 是左方括号 '['，解析下标操作
        prefix = parseSubscript(prefix);
      } else {
        // 其他情况退出循环
        break;
      }
    }
    // 返回解析后的前缀表达式
    return prefix;
  }
  // 尝试解析赋值操作符
  std::optional<TreeRef> maybeParseAssignmentOp() {
    auto r = L.cur().range;
    // 根据当前 token 类型进行不同的处理
    switch (L.cur().kind) {
      // 复合赋值运算符，如 +=, -=, *= 等
      case TK_PLUS_EQ:
      case TK_MINUS_EQ:
      case TK_TIMES_EQ:
      case TK_DIV_EQ:
      case TK_BIT_OR_EQ:
      case TK_BIT_AND_EQ:
      case TK_BIT_XOR_EQ:
      case TK_MOD_EQ: {
        // 获取运算符并返回相应的复合赋值表达式
        int modifier = L.next().text()[0];
        return create_compound(modifier, r, {});
      } break;
      // 左移赋值运算符
      case TK_LSHIFT_EQ: {
        L.next();
        return create_compound(TK_LSHIFT, r, {});
      } break;
      // 右移赋值运算符
      case TK_RSHIFT_EQ: {
        L.next();
        return create_compound(TK_RSHIFT, r, {});
      } break;
      // 指数赋值运算符
      case TK_POW_EQ: {
        L.next();
        return create_compound(TK_POW, r, {});
      } break;
      // 普通赋值操作符 '='
      case '=': {
        L.next();
        return create_compound('=', r, {}); // no reduction
      } break;
      // 默认情况返回空值
      default:
        return c10::nullopt;
    }
  }
  // 解析三元表达式
  TreeRef parseTrinary(
      TreeRef true_branch,
      const SourceRange& range,
      int binary_prec) {
    // 解析条件表达式
    auto cond = parseExp();
    // 确保存在 TK_ELSE，解析假分支表达式
    L.expect(TK_ELSE);
    auto false_branch = parseExp(binary_prec);
    // 创建三元表达式对象并返回
    return create_compound(
        TK_IF_EXPR, range, {cond, std::move(true_branch), false_branch});
  }
  // 解析具有比给定优先级更高的二元运算符的最长表达式
  // 当 precedence == 0 时，解析所有表达式
  // 这是自顶向下优先级解析的核心循环
  Expr parseExp() {
    return parseExp(0);
  }
  // 解析具有比给定优先级更高的二元运算符的表达式
  Expr parseExp(int precedence) {
    TreeRef prefix;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int unary_prec;
    // 检查当前标记是否为一元操作符，并获取其优先级
    if (shared.isUnary(L.cur().kind, &unary_prec)) {
      // 获取当前操作符的类型和范围
      auto kind = L.cur().kind;
      auto pos = L.cur().range;
      // 移动到下一个标记
      L.next();
      // 确定一元操作符的具体类型
      auto unary_kind = kind == '*' ? TK_STARRED
          : kind == '-'             ? TK_UNARY_MINUS
                                    : kind;
      // 解析一元操作符后的表达式
      auto subexp = parseExp(unary_prec);
      // 如果是负号并且子表达式是常量，则将负号与常量合并
      if (unary_kind == TK_UNARY_MINUS && subexp.kind() == TK_CONST) {
        prefix = Const::create(subexp.range(), "-" + Const(subexp).text());
      } else {
        // 否则创建复合表达式
        prefix = create_compound(unary_kind, pos, {subexp});
      }
    } else {
      // 如果不是一元操作符，则解析基础表达式
      prefix = parseBaseExp();
    }
    // 忽略下一行的lint警告，避免初始化变量
    // 初始化二元操作符的优先级
    int binary_prec;
    // 当当前标记为二元操作符且优先级大于给定的precedence时循环
    while (shared.isBinary(L.cur().kind, &binary_prec)) {
      if (binary_prec <= precedence) // 不允许解析优先级小于或等于 'precedence' 的表达式
        break;

      // 获取当前操作符的类型和范围
      int kind = L.cur().kind;
      auto pos = L.cur().range;
      // 移动到下一个标记
      L.next();
      // 如果是右结合的操作符，降低下一个表达式的解析优先级
      if (shared.isRightAssociative(kind))
        binary_prec--;

      // 如果是 'not in' 操作符
      if (kind == TK_NOTIN) {
        // 将 'not in' 转换为 'not (in)' 的结构，不引入新的树视图结构，而是在当前树视图中进行嵌套调用
        prefix = create_compound(TK_IN, pos, {prefix, parseExp(binary_prec)});
        prefix = create_compound(TK_NOT, pos, {prefix});
        continue;
      }

      // 对于三元操作符的特殊情况
      if (kind == TK_IF) {
        // 解析三元操作符
        prefix = parseTrinary(prefix, pos, binary_prec);
        continue;
      }

      // 对于 'for' 操作符的特殊情况
      if (kind == TK_FOR) {
        // 'for' 目标应只解析优先级大于4的表达式，这些表达式按照Python语法应位于左侧
        auto target = parseLHSExp();
        L.expect(TK_IN);
        auto iter = parseExp();
        prefix = ListComp::create(pos, Expr(prefix), target, iter);
        continue;
      }

      // 创建二元操作符的复合表达式
      prefix = create_compound(kind, pos, {prefix, parseExp(binary_prec)});
    }
    // 返回处理后的表达式
    return Expr(prefix);
  }

  // 解析序列表达式
  void parseSequence(
      int begin,
      int sep,
      int end,
      const std::function<void()>& parse) {
    // 如果起始标记不是 TK_NOTHING，则期望起始标记
    if (begin != TK_NOTHING) {
      L.expect(begin);
    }
    // 循环解析直到结束标记
    while (end != L.cur().kind) {
      // 解析单个表达式
      parse();
      // 如果未能找到分隔符，则检查是否需要期望结束标记
      if (!L.nextIf(sep)) {
        if (end != TK_NOTHING) {
          L.expect(end);
        }
        return;
      }
    }
    // 最后期望结束标记
    L.expect(end);
  }

  // 解析列表
  template <typename T>
  List<T> parseList(int begin, int sep, int end, T (ParserImpl::*parse)()) {
    // 获取当前位置范围
    auto r = L.cur().range;
    // 创建元素向量
    std::vector<T> elements;
    // 解析序列表达式
    parseSequence(
        begin, sep, end, [&] { elements.emplace_back((this->*parse)()); });
    // 返回列表对象
    return List<T>::create(r, elements);
  }

  // 解析常量
  Const parseConst() {
    // 获取当前标记的范围
    auto range = L.cur().range;
    // 期望下一个标记为数字常量
    auto t = L.expect(TK_NUMBER);
  return Const::create(t.range, t.text());
}

// 解析连接的字符串字面量
StringLiteral parseConcatenatedStringLiterals() {
  auto range = L.cur().range;
  std::string ss;
  while (L.cur().kind == TK_STRINGLITERAL) {
    auto literal_range = L.cur().range;
    // 解析单个字符串字面量并追加到结果字符串中
    ss.append(parseStringLiteral(literal_range, L.next().text()));
  }
  // 创建包含连接字符串的字符串字面量对象并返回
  return StringLiteral::create(range, ss);
}

// 解析属性值表达式
Expr parseAttributeValue() {
  return parseExp();
}

// 解析函数调用或索引中的参数列表
void parseArguments(TreeList& inputs, TreeList& attributes) {
  parseSequence('(', ',', ')', [&] {
    if (L.cur().kind == TK_IDENT && L.lookahead().kind == '=') {
      auto ident = parseIdent();
      L.expect('=');
      auto v = parseAttributeValue();
      // 创建带有属性名称和值的属性对象并加入到属性列表中
      attributes.push_back(Attribute::create(ident.range(), Ident(ident), v));
    } else {
      // 解析普通的表达式参数并加入到输入参数列表中
      inputs.push_back(parseExp());
    }
  });
}

// 解析左侧可接受的表达式，其优先级大于4，按照Python语法
Expr parseLHSExp() {
  return parseExp(4);
}

// 解析形式为[a:], [:b], [a:b], [:]和包含"::"的所有变体的切片表达式
Expr parseSubscriptExp() {
  TreeRef first, second, third;
  auto range = L.cur().range;
  if (L.cur().kind != ':') {
    first = parseExp();
  }
  if (L.nextIf(':')) {
    if (L.cur().kind != ',' && L.cur().kind != ']' && L.cur().kind != ':') {
      second = parseExp();
    }
    if (L.nextIf(':')) {
      if (L.cur().kind != ',' && L.cur().kind != ']') {
        third = parseExp();
      }
    }
    // 创建切片表达式对象并返回
    auto maybe_first = first ? Maybe<Expr>::create(range, Expr(first))
                             : Maybe<Expr>::create(range);
    auto maybe_second = second ? Maybe<Expr>::create(range, Expr(second))
                               : Maybe<Expr>::create(range);
    auto maybe_third = third ? Maybe<Expr>::create(range, Expr(third))
                             : Maybe<Expr>::create(range);
    return SliceExpr::create(range, maybe_first, maybe_second, maybe_third);
  } else {
    return Expr(first);
  }
}

// 解析给定值的下标表达式
TreeRef parseSubscript(const TreeRef& value) {
  const auto range = L.cur().range;

  // 解析下标表达式列表并创建下标对象
  auto subscript_exprs =
      parseList('[', ',', ']', &ParserImpl::parseSubscriptExp);

  const auto whole_range =
      SourceRange(range.source(), range.start(), L.cur().range.start());
  return Subscript::create(whole_range, Expr(value), subscript_exprs);
}

// 可能解析类型注释表达式
Maybe<Expr> maybeParseTypeAnnotation() {
  if (L.nextIf(':')) {
    // 注意：不要内联调用parseExp，因为L.cur().range在parseExp()调用时可能会被修改。
    auto expr = parseExp();
    return Maybe<Expr>::create(expr.range(), expr);
  } else {
    return Maybe<Expr>::create(L.cur().range);
  }
}

// 解析形式参数
TreeRef parseFormalParam(bool kwarg_only) {
  auto ident = parseIdent();
  // 解析可能的类型注释
  TreeRef type = maybeParseTypeAnnotation();
  TreeRef def;
    if (L.nextIf('=')) {
      // 如果下一个字符是'='，则执行以下代码块
      // 注意：parseExp不能在此处内联调用，因为当 L.cur().range 与 parseExp() 调用时相比，参数求值顺序会改变。
      auto expr = parseExp(); // 解析表达式并存储在 expr 中
      def = Maybe<Expr>::create(expr.range(), expr); // 创建一个 Maybe 包含表达式的定义
    } else {
      // 如果没有'='，则执行以下代码块
      def = Maybe<Expr>::create(L.cur().range); // 创建一个 Maybe，包含当前位置的范围作为定义
    }
    return Param::create(
        type->range(), // 参数类型的范围
        Ident(ident), // 参数标识符
        Maybe<Expr>(type), // 可能包含参数类型的表达式
        Maybe<Expr>(def), // 可能包含参数默认值的表达式
        kwarg_only); // 是否仅限关键字参数的标志
  }

  Param parseBareTypeAnnotation() {
    auto type = parseExp(); // 解析表达式并存储在 type 中
    return Param::create(
        type.range(), // 参数类型的范围
        Ident::create(type.range(), ""), // 创建一个空标识符
        Maybe<Expr>::create(type.range(), type), // 可能包含类型表达式的 Maybe
        Maybe<Expr>::create(type.range()), // 创建一个空的 Maybe 表达式
        /*kwarg_only=*/false); // 标识不是仅限关键字参数
  }

  Decl parseTypeComment() {
    auto range = L.cur().range; // 当前位置的范围
    L.expect(TK_TYPE_COMMENT); // 确保当前令牌是类型注释
    auto param_types =
        parseList('(', ',', ')', &ParserImpl::parseBareTypeAnnotation); // 解析参数类型列表
    TreeRef return_type;
    if (L.nextIf(TK_ARROW)) {
      auto return_type_range = L.cur().range; // 返回类型的范围
      return_type = Maybe<Expr>::create(return_type_range, parseExp()); // 可能包含返回类型表达式的 Maybe
    } else {
      return_type = Maybe<Expr>::create(L.cur().range); // 创建一个空的 Maybe 表达式
    }
    return Decl::create(range, param_types, Maybe<Expr>(return_type)); // 创建声明对象
  }

  // 'first' has already been parsed since expressions can exist
  // alone on a line:
  // first[,other,lhs] = rhs
  TreeRef parseAssign(const Expr& lhs) {
    auto type = maybeParseTypeAnnotation(); // 可能解析类型注释
    auto maybeOp = maybeParseAssignmentOp(); // 可能解析赋值操作符
    if (maybeOp) {
      // 如果存在赋值操作符，解析右手边表达式并生成赋值语句
      auto rhs = parseExpOrExpTuple(); // 解析表达式或表达式元组
      if (maybeOp.value()->kind() == '=') {
        std::vector<Expr> lhs_list = {lhs}; // 创建左手边表达式列表，初始化为 lhs
        while (L.nextIf('=')) {
          lhs_list.push_back(rhs); // 将右手边表达式添加到左手边表达式列表中
          rhs = parseExpOrExpTuple(); // 继续解析下一个右手边表达式
        }
        if (type.present() && lhs_list.size() > 1) {
          throw ErrorReport(type.range())
              << "Annotated multiple assignment is not supported in python"; // 抛出错误，Python 不支持多重赋值注解
        }
        L.expect(TK_NEWLINE); // 确保下一个令牌是新行
        return Assign::create(
            lhs.range(), // 左手边表达式的范围
            List<Expr>::create(lhs_list[0].range(), lhs_list), // 创建左手边表达式列表
            Maybe<Expr>::create(rhs.range(), rhs), // 可能包含右手边表达式的 Maybe
            type); // 赋值语句的类型注释
      } else {
        L.expect(TK_NEWLINE); // 确保下一个令牌是新行
        // 这是一个增强赋值操作
        if (lhs.kind() == TK_TUPLE_LITERAL) {
          throw ErrorReport(lhs.range())
              << " augmented assignment can only have one LHS expression"; // 抛出错误，增强赋值只能有一个左手边表达式
        }
        return AugAssign::create(
            lhs.range(), // 左手边表达式的范围
            lhs, // 左手边表达式
            AugAssignKind(*maybeOp), // 增强赋值操作的种类
            Expr(rhs)); // 右手边表达式
      }
    } else {
      // 如果没有赋值操作符，则形式为 `lhs : <type>`
      TORCH_INTERNAL_ASSERT(type.present()); // 断言确保类型注释存在
      L.expect(TK_NEWLINE); // 确保下一个令牌是新行
      return Assign::create(
          lhs.range(), // 左手边表达式的范围
          List<Expr>::create(lhs.range(), {lhs}), // 创建左手边表达式列表
          Maybe<Expr>::create(lhs.range()), // 可能包含左手边表达式的 Maybe
          type); // 赋值语句的类型注释
    }
  }

  # 解析语句并返回语法树引用
  TreeRef parseStmt(bool in_class = false) {
    # 根据当前 token 的类型进行不同的语句解析
    switch (L.cur().kind) {
      case TK_IF:
        return parseIf();  # 解析 if 语句并返回语法树引用
      case TK_WHILE:
        return parseWhile();  # 解析 while 语句并返回语法树引用
      case TK_FOR:
        return parseFor();  # 解析 for 语句并返回语法树引用
      case TK_GLOBAL: {
        auto range = L.next().range;  # 获取当前 token 的范围
        auto idents =
            parseList(TK_NOTHING, ',', TK_NOTHING, &ParserImpl::parseIdent);  # 解析逗号分隔的标识符列表
        L.expect(TK_NEWLINE);  # 确保下一个 token 是换行符
        return Global::create(range, idents);  # 创建全局声明语法树节点并返回
      }
      case TK_RETURN: {
        auto range = L.next().range;  # 获取当前 token 的范围
        Expr value = L.cur().kind != TK_NEWLINE
            ? parseExpOrExpTuple()  # 解析表达式或表达式元组
            : Expr(create_compound(TK_NONE, range, {}));  # 创建一个空的复合表达式
        L.expect(TK_NEWLINE);  # 确保下一个 token 是换行符
        return Return::create(range, value);  # 创建返回语句的语法树节点并返回
      }
      case TK_RAISE: {
        auto range = L.next().range;  # 获取当前 token 的范围
        auto expr = parseExp();  # 解析表达式
        L.expect(TK_NEWLINE);  # 确保下一个 token 是换行符
        return Raise::create(range, expr);  # 创建异常抛出语法树节点并返回
      }
      case TK_ASSERT: {
        auto range = L.next().range;  # 获取当前 token 的范围
        auto cond = parseExp();  # 解析条件表达式
        Maybe<Expr> maybe_first = Maybe<Expr>::create(range);  # 创建一个空的表达式 Maybe 对象
        if (L.nextIf(',')) {
          auto msg = parseExp();  # 解析断言消息表达式
          maybe_first = Maybe<Expr>::create(range, Expr(msg));  # 创建带有消息表达式的 Maybe 对象
        }
        L.expect(TK_NEWLINE);  # 确保下一个 token 是换行符
        return Assert::create(range, cond, maybe_first);  # 创建断言语法树节点并返回
      }
      case TK_BREAK: {
        auto range = L.next().range;  # 获取当前 token 的范围
        L.expect(TK_NEWLINE);  # 确保下一个 token 是换行符
        return Break::create(range);  # 创建 break 语句的语法树节点并返回
      }
      case TK_CONTINUE: {
        auto range = L.next().range;  # 获取当前 token 的范围
        L.expect(TK_NEWLINE);  # 确保下一个 token 是换行符
        return Continue::create(range);  # 创建 continue 语句的语法树节点并返回
      }
      case TK_PASS: {
        auto range = L.next().range;  # 获取当前 token 的范围
        L.expect(TK_NEWLINE);  # 确保下一个 token 是换行符
        return Pass::create(range);  # 创建 pass 语句的语法树节点并返回
      }
      case TK_DEF: {
        return parseFunction(/*is_method=*/in_class);  # 解析函数定义语句并返回语法树引用，根据 in_class 参数确定是否是类方法
      }
      case TK_DELETE: {
        auto range = L.next().range;  # 获取当前 token 的范围
        auto targets =
            parseList(TK_NOTHING, ',', TK_NOTHING, &ParserImpl::parseExp);  # 解析逗号分隔的表达式列表
        L.expect(TK_NEWLINE);  # 确保下一个 token 是换行符
        return Delete::create(range, targets);  # 创建删除语句的语法树节点并返回
      }
      case TK_WITH: {
        return parseWith();  # 解析 with 语句并返回语法树引用
      }
      default: {
        auto lhs = parseExpOrExpTuple();  # 解析表达式或表达式元组
        if (L.cur().kind != TK_NEWLINE) {
          return parseAssign(lhs);  # 如果当前 token 不是换行符，则解析赋值语句并返回语法树引用
        } else {
          L.expect(TK_NEWLINE);  # 确保下一个 token 是换行符
          return ExprStmt::create(lhs.range(), lhs);  # 创建表达式语句的语法树节点并返回
        }
      }
    }
  }

  # 解析 with 语句的子项并返回 WithItem 对象
  WithItem parseWithItem() {
    auto target = parseExp();  # 解析 with 语句的目标表达式

    if (L.cur().kind == TK_AS) {
      // 如果当前 token 是 TK_AS，则这个 with 子项的形式为 "expression as target"
      auto token = L.expect(TK_AS);  # 获取 TK_AS 的 token
      Ident ident = parseIdent();  # 解析标识符
      auto var = Var::create(ident.range(), ident);  # 创建变量节点
      return WithItem::create(
          token.range, target, Maybe<Var>::create(ident.range(), var));  # 创建带有变量的 with 子项对象并返回
    } else {
      // 如果不是，则这个 with 子项的形式为 "expression"
      return WithItem::create(
          target.range(), target, Maybe<Var>::create(target.range()));  # 创建没有变量的 with 子项对象并返回
  }
  // 解析一个 if 语句
  TreeRef parseIf(bool expect_if = true) {
    auto r = L.cur().range;
    // 如果期望有 if 关键字，则期望并读取 TK_IF
    if (expect_if)
      L.expect(TK_IF);
    // 解析条件表达式
    auto cond = parseExp();
    // 读取 ':' 分隔符
    L.expect(':');
    // 解析真实分支的语句列表
    auto true_branch = parseStatements(/*expect_indent=*/true);
    // 初始化假分支为空列表
    auto false_branch = makeList(L.cur().range, {});
    // 如果遇到 TK_ELSE 关键字
    if (L.nextIf(TK_ELSE)) {
      // 读取 ':' 分隔符
      L.expect(':');
      // 解析假分支的语句列表
      false_branch = parseStatements(/*expect_indent=*/true);
    } else if (L.nextIf(TK_ELIF)) {
      // 注意：这里需要分为独立语句，因为调用 parseIf 会改变词法分析器状态，
      // 这可能导致编译器在评估参数表达式时发生 heap-use-after-free
      auto range = L.cur().range;
      // 解析 elif 分支，递归调用 parseIf(false) 获取其结果作为假分支
      false_branch = makeList(range, {parseIf(false)});
    }
    // 创建并返回 If 对象
    return If::create(
        r, Expr(cond), List<Stmt>(true_branch), List<Stmt>(false_branch));
  }
  // 解析一个 while 循环语句
  TreeRef parseWhile() {
    auto r = L.cur().range;
    // 读取并期望 TK_WHILE
    L.expect(TK_WHILE);
    // 解析循环条件表达式
    auto cond = parseExp();
    // 读取 ':' 分隔符
    L.expect(':');
    // 解析循环体的语句列表
    auto body = parseStatements(/*expect_indent=*/true);
    // 创建并返回 While 对象
    return While::create(r, Expr(cond), List<Stmt>(body));
  }

  // 解析一个 for 循环语句
  TreeRef parseFor() {
    auto r = L.cur().range;
    // 读取并期望 TK_FOR
    L.expect(TK_FOR);
    // 解析循环目标列表
    auto targets = parseList(TK_NOTHING, ',', TK_IN, &ParserImpl::parseLHSExp);
    // 解析迭代器表达式列表
    auto itrs = parseList(TK_NOTHING, ',', ':', &ParserImpl::parseExp);
    // 解析循环体的语句列表
    auto body = parseStatements(/*expect_indent=*/true);
    // 创建并返回 For 对象
    return For::create(r, targets, itrs, body);
  }

  // 解析一个 with 语句
  TreeRef parseWith() {
    auto r = L.cur().range;
    // 读取并期望 TK_WITH
    L.expect(TK_WITH);
    // 解析 with 表达式及其目标列表
    auto targets = parseList(TK_NOTHING, ',', ':', &ParserImpl::parseWithItem);
    // 解析 with 语句体的语句列表
    auto body = parseStatements(/*expect_indent=*/true);
    // 创建并返回 With 对象
    return With::create(r, targets, body);
  }

  // 解析语句列表
  TreeRef parseStatements(bool expect_indent, bool in_class = false) {
    auto r = L.cur().range;
    // 如果期望有缩进，则读取并期望 TK_INDENT
    if (expect_indent) {
      L.expect(TK_INDENT);
    }
    // 初始化语句列表
    TreeList stmts;
    // 循环解析语句，直到遇到 TK_DEDENT
    do {
      stmts.push_back(parseStmt(in_class));
    } while (!L.nextIf(TK_DEDENT));
    // 创建并返回复合语句对象
    return create_compound(TK_LIST, r, std::move(stmts));
  }

  // 解析可能的返回值类型注解
  Maybe<Expr> parseReturnAnnotation() {
    // 如果遇到 TK_ARROW，则说明有返回值类型注解
    if (L.nextIf(TK_ARROW)) {
      // 读取返回值类型的范围
      auto return_type_range = L.cur().range;
      // 解析返回值类型表达式并返回 Maybe 包装的 Expr 对象
      return Maybe<Expr>::create(return_type_range, parseExp());
    } else {
      // 否则返回一个空的 Maybe 包装
      return Maybe<Expr>::create(L.cur().range);
    }
  }

  // 解析形式参数列表
  List<Param> parseFormalParams() {
    auto r = L.cur().range;
    // 初始化参数列表
    std::vector<Param> params;
    // 是否仅接受关键字参数
    bool kwarg_only = false;
    // 解析参数序列，以 '(' 开始，',' 分隔，以 ')' 结束
    parseSequence('(', ',', ')', [&] {
      // 如果不是关键字参数，遇到 '*' 符号则标记为仅关键字参数
      if (!kwarg_only && L.nextIf('*')) {
        kwarg_only = true;
      } else {
        // 否则解析并添加形式参数到列表中
        params.emplace_back(parseFormalParam(kwarg_only));
      }
    });
    // 创建并返回参数列表对象
    return List<Param>::create(r, params);
  }
  // 解析声明语句
  Decl parseDecl() {
    // 解析返回值类型注解
    List<Param> paramlist = parseFormalParams();
    // 初始化返回值类型对象
    TreeRef return_type;
    // 解析可能的返回值类型注解
    Maybe<Expr> return_annotation = parseReturnAnnotation();
    L.expect(':');
    // 调用 Lexer 对象的 expect 方法，期望当前 token 是冒号
    return Decl::create(
        paramlist.range(), List<Param>(paramlist), return_annotation);
  }

  TreeRef parseClass() {
    L.expect(TK_CLASS_DEF);
    // 期望当前 token 是类定义关键字（TK_CLASS_DEF），并解析类名
    const auto name = parseIdent();
    // 创建一个可能为空的表达式作为父类，默认为类名
    Maybe<Expr> superclass = Maybe<Expr>::create(name.range());
    if (L.nextIf('(')) {
      // 如果下一个 token 是左括号，表示有父类继承
      // 目前仅支持从 NamedTuple 继承
      auto id = parseExp();
      superclass = Maybe<Expr>::create(id.range(), id);
      L.expect(')');
    }
    L.expect(':');
    // 期望当前 token 是冒号，表示类定义的结束
    const auto statements =
        parseStatements(/*expect_indent=*/true, /*in_class=*/true);
    // 创建类定义节点，包括类名、父类、类体语句列表
    return ClassDef::create(
        name.range(), name, superclass, List<Stmt>(statements));
  }

  TreeRef parseFunction(bool is_method) {
    L.expect(TK_DEF);
    // 期望当前 token 是函数定义关键字（TK_DEF），并解析函数名
    auto name = parseIdent();
    // 解析函数声明
    auto decl = parseDecl();

    TreeRef stmts_list;
    if (L.nextIf(TK_INDENT)) {
      // 如果下一个 token 是缩进（TK_INDENT），处理多行函数体
      // 检查是否有类型注解作为函数体的第一行注释
      if (L.cur().kind == TK_TYPE_COMMENT) {
        // 如果当前 token 是类型注释（TK_TYPE_COMMENT），解析并创建声明
        auto type_annotation_decl = Decl(parseTypeComment());
        L.expect(TK_NEWLINE);
        // 合并函数声明和类型注释的信息
        decl = mergeTypesFromTypeComment(decl, type_annotation_decl, is_method);
      }

      stmts_list = parseStatements(false);
    } else {
      // 否则，处理单行函数体的特殊情况
      // Python 语法允许单行函数只有一条语句
      if (L.cur().kind == TK_TYPE_COMMENT) {
        // 如果当前 token 是类型注释（TK_TYPE_COMMENT），解析并创建声明
        auto type_annotation_decl = Decl(parseTypeComment());
        decl = mergeTypesFromTypeComment(decl, type_annotation_decl, is_method);
      }

      TreeList stmts;
      // 解析单条语句作为函数体
      stmts.push_back(parseStmt(is_method));
      stmts_list = create_compound(TK_LIST, L.cur().range, std::move(stmts));
    }

    // 创建函数定义节点，包括函数名、声明、函数体语句列表
    return Def::create(
        name.range(), Ident(name), Decl(decl), List<Stmt>(stmts_list));
  }
  Lexer& lexer() {
    // 返回当前对象的 Lexer 引用
    return L;
  }

 private:
  // short helpers to create nodes
  TreeRef create_compound(
      int kind,
      const SourceRange& range,
      TreeList&& trees) {
    // 创建复合节点的辅助方法，包括节点类型、范围和子树列表
    return Compound::create(kind, range, std::move(trees));
  }
  TreeRef makeList(const SourceRange& range, TreeList&& trees) {
    // 创建列表节点的辅助方法，类型为 TK_LIST，包括范围和子树列表
    return create_compound(TK_LIST, range, std::move(trees));
  }
  Lexer L;  // Lexer 对象用于词法分析
  SharedParserData& shared;  // 共享的解析器数据
};

Parser::Parser(const std::shared_ptr<Source>& src)
    : pImpl(new ParserImpl(src)) {}  // 使用源文件的共享指针创建 ParserImpl 对象的实例

Parser::~Parser() = default;  // 默认析构函数的定义

TreeRef Parser::parseFunction(bool is_method) {
  return pImpl->parseFunction(is_method);  // 调用 ParserImpl 中的 parseFunction 方法解析函数
}

TreeRef Parser::parseClass() {
  return pImpl->parseClass();  // 调用 ParserImpl 中的 parseClass 方法解析类
}

Lexer& Parser::lexer() {
  return pImpl->lexer();  // 返回 ParserImpl 中的 lexer 对象引用
}

Decl Parser::parseTypeComment() {
  return pImpl->parseTypeComment();  // 调用 ParserImpl 中的 parseTypeComment 方法解析类型注释
}

Expr Parser::parseExp() {
  return pImpl->parseExp();  // 调用 ParserImpl 中的 parseExp 方法解析表达式
}

} // namespace torch::jit  // 结束 torch::jit 命名空间
```