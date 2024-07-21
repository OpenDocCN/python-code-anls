# `.\pytorch\torch\csrc\jit\jit_opt_limit.cpp`

```py
// 定义一个静态函数，返回一个引用到字符串到整数的无序映射，用于跟踪每个优化通道的计数器
static std::unordered_map<std::string, int64_t>& passes_to_current_counter() {
  // 静态变量，存储每个优化通道的当前计数器值
  static std::unordered_map<std::string, int64_t> passes_to_current_counter;
  return passes_to_current_counter;
}

// 解析优化限制字符串，返回整数表示的限制值
static int parseOptLimit(const std::string& opt_limit) {
  try {
    // 将字符串转换为整数
    int64_t n = std::stoi(opt_limit);
    return n;
  } catch (...) {
    // 如果转换失败，返回-1
    return -1;
  }
}

// 解析传入的JIT优化限制选项字符串，返回一个映射，将优化通道名称映射到其限制值
static std::unordered_map<std::string, int64_t> parseJITOptLimitOption(
    const char* option) {
  // 用stringstream处理传入的选项字符串
  std::stringstream in_ss;
  if (option) {
    in_ss << option;
  }
  // 存储优化通道名称和其限制值的映射
  std::unordered_map<std::string, int64_t> passes_to_opt_limits;
  std::string line;
  // 按行解析选项字符串，每行以冒号分隔
  while (std::getline(in_ss, line, ':')) {
    if (line.empty()) {
      continue;
    }
    // 查找最后一个'='字符的位置
    auto index_at = line.find_last_of('=');
    // 获取优化通道名称
    auto pass_name = line.substr(0, index_at);
    pass_name = c10::detail::ExcludeFileExtension(pass_name); // 排除文件扩展名
    // 获取优化限制值
    auto opt_limit = parseOptLimit(line.substr(index_at + 1));
    // 插入到映射中
    passes_to_opt_limits.insert({pass_name, opt_limit});
  }

  return passes_to_opt_limits;
}

// 检查给定优化通道是否在限制内
bool opt_limit(const char* pass_name) {
  // 获取环境变量PYTORCH_JIT_OPT_LIMIT的值
  static const char* opt_limit = std::getenv("PYTORCH_JIT_OPT_LIMIT");
  // 如果未提供环境变量，允许所有优化通道
  if (!opt_limit) {
    return true;
  }

  // 解析优化限制选项，并存储在静态映射中
  static const std::unordered_map<std::string, int64_t> passes_to_opt_limits =
      parseJITOptLimitOption(opt_limit);
  
  // 处理传入的优化通道名称，排除其基本文件名和扩展名
  std::string pass{pass_name};
  pass = c10::detail::StripBasename(pass); // 剥离基本文件名
  pass = c10::detail::ExcludeFileExtension(pass); // 排除文件扩展名

  // 查找优化通道在静态映射中的限制值
  auto opt_limit_it = passes_to_opt_limits.find(pass);
  if (opt_limit_it == passes_to_opt_limits.end()) {
    // 如果未找到优化通道，允许执行优化
    return true;
  }

  // 获取当前优化通道的执行计数器
  auto current_count_it = passes_to_current_counter().find(pass);
  if (current_count_it == passes_to_current_counter().end()) {
    // 如果未找到当前计数器，将其初始化为0
    passes_to_current_counter().insert({pass, 0});
  }

  // 再次获取当前计数器
  current_count_it = passes_to_current_counter().find(pass);
  // 如果当前计数器大于或等于优化通道的限制值，则不执行优化
  if (current_count_it->second >= opt_limit_it->second) {
    return false;
  }

  // 增加当前优化通道的执行计数器，并允许执行优化
  current_count_it->second++;
  return true;
}
```