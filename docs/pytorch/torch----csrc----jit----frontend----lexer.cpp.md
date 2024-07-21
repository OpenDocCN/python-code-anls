# `.\pytorch\torch\csrc\jit\frontend\lexer.cpp`

```py
namespace torch::jit {

// 静态常量，表示不同二元操作符的优先级映射表
static const std::unordered_map<int, int> binary_prec = {
    {TK_IF, 1},         // 条件语句 IF 的优先级为 1
    {TK_FOR, 1},        // 循环语句 FOR 的优先级为 1
    {TK_AND, 2},        // 逻辑 AND 的优先级为 2
    {TK_OR, 2},         // 逻辑 OR 的优先级为 2
    {TK_IN, 4},         // 成员运算符 IN 的优先级为 4
    {TK_NOTIN, 4},      // 成员运算符 NOT IN 的优先级为 4
    {'<', 4},           // 小于运算符的优先级为 4
    {'>', 4},           // 大于运算符的优先级为 4
    {TK_IS, 4},         // 身份运算符 IS 的优先级为 4
    {TK_ISNOT, 4},      // 身份运算符 IS NOT 的优先级为 4
    {TK_EQ, 4},         // 等于运算符的优先级为 4
    {TK_LE, 4},         // 小于等于运算符的优先级为 4
    {TK_GE, 4},         // 大于等于运算符的优先级为 4
    {TK_NE, 4},         // 不等于运算符的优先级为 4
    {'|', 5},           // 按位或运算符的优先级为 5
    {'^', 6},           // 按位异或运算符的优先级为 6
    {'&', 7},           // 按位与运算符的优先级为 7
    {TK_LSHIFT, 8},     // 左移运算符的优先级为 8
    {TK_RSHIFT, 8},     // 右移运算符的优先级为 8
    {'+', 9},           // 加法运算符的优先级为 9
    {'-', 9},           // 减法运算符的优先级为 9
    {'*', 10},          // 乘法运算符的优先级为 10
    {'/', 10},          // 除法运算符的优先级为 10
    {TK_FLOOR_DIV, 10}, // 地板除法运算符的优先级为 10
    {'%', 10},          // 取模运算符的优先级为 10
    {'@', 10},          // 矩阵乘法运算符的优先级为 10
    {TK_POW, 11},       // 幂运算符的优先级为 11
};

// 静态常量，表示不同一元操作符的优先级映射表
static const std::unordered_map<int, int> unary_prec = {
    {TK_NOT, 3},    // 逻辑 NOT 的优先级为 3
    {'~', 3},       // 按位取反的优先级为 3
    {'-', 10},      // 负号的优先级为 10
    {'*', 10},      // 解引用操作符的优先级为 10
};

// 判断给定操作符是否为一元操作符，并返回其优先级
bool SharedParserData::isUnary(int kind, int* prec) {
  auto it = unary_prec.find(kind);
  if (it != unary_prec.end()) {
    *prec = it->second;
    return true;
  }
  return false;
}

// 判断给定操作符是否为二元操作符，并返回其优先级
bool SharedParserData::isBinary(int kind, int* prec) {
  auto it = binary_prec.find(kind);
  if (it != binary_prec.end()) {
    *prec = it->second;
    return true;
  }
  return false;
}

// 将字符串表示的操作符转换为对应的整数表示
C10_EXPORT int stringToKind(const std::string& str) {
  static std::unordered_map<std::string, int> str_to_kind = []() {
    std::unordered_map<std::string, int> ret_str_to_kind;
    // 遍历所有有效的单字符操作符，构建映射表
    for (char tok : std::string(valid_single_char_tokens))
      // NOLINTNEXTLINE(bugprone-signed-char-misuse)
      ret_str_to_kind[std::string(1, tok)] = tok;
#define DEFINE_CASE(tok, _, str) \
  if (std::string(str) != "")    \
    ret_str_to_kind[str] = tok;  // 将操作符及其对应的整数值存入映射表
    TC_FORALL_TOKEN_KINDS(DEFINE_CASE)  // 宏展开，用于处理所有 Token 类别
#undef DEFINE_CASE
    return ret_str_to_kind;
  }();
  try {
    return str_to_kind.at(str);  // 查找并返回对应操作符的整数值
  } catch (std::out_of_range&) {
    throw std::out_of_range("unknown token in stringToKind");
  }
}

// 将操作符的整数表示转换为对应的字符串表示
C10_EXPORT std::string kindToString(int kind) {
  if (kind < 256)
    return std::string(1, kind);  // 如果操作符是单字符，则直接返回对应字符的字符串形式
  switch (kind) {
#define DEFINE_CASE(tok, str, _) \
  case tok:                      \
    return str;                  // 返回操作符的字符串表示
    TC_FORALL_TOKEN_KINDS(DEFINE_CASE)  // 宏展开，处理所有 Token 类别
#undef DEFINE_CASE
    default:
      throw std::runtime_error("Unknown kind: " + std::to_string(kind));  // 抛出异常，表示未知的操作符
  }
}

// 返回共享的解析器数据对象的引用
C10_EXPORT SharedParserData& sharedParserData() {
  static SharedParserData data;  // 线程安全地初始化共享数据对象
  return data;  // 返回共享的解析器数据对象
}

} // namespace torch::jit
```