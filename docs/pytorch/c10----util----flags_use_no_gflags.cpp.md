# `.\pytorch\c10\util\flags_use_no_gflags.cpp`

```py
#ifndef C10_USE_GFLAGS
// 如果没有定义 C10_USE_GFLAGS，则进入命名空间 c10
namespace c10 {

// 使用 std::string
using std::string;

// 定义 C10FlagsRegistry 类型的注册表 C10FlagsRegistry
C10_DEFINE_REGISTRY(C10FlagsRegistry, C10FlagParser, const string&);

// 匿名命名空间，定义静态变量 gCommandLineFlagsParsed 并初始化为 false
namespace {
static bool gCommandLineFlagsParsed = false;

// 静态函数 GlobalInitStream 返回静态的 std::stringstream 对象 ss
std::stringstream& GlobalInitStream() {
  static std::stringstream ss;
  return ss;
}

// 静态变量 gUsageMessage 初始化为 "(Usage message not set.)"
static const char* gUsageMessage = "(Usage message not set.)";
} // namespace

// 设置使用消息的函数，静态变量 usage_message_safe_copy 保存传入的字符串副本
C10_EXPORT void SetUsageMessage(const string& str) {
  static string usage_message_safe_copy = str;
  gUsageMessage = usage_message_safe_copy.c_str();
}

// 返回使用消息的函数，返回静态变量 gUsageMessage
C10_EXPORT const char* UsageMessage() {
  return gUsageMessage;
}

// 解析命令行参数的函数
C10_EXPORT bool ParseCommandLineFlags(int* pargc, char*** pargv) {
  // 如果参数个数为 0，直接返回 true
  if (*pargc == 0)
    return true;

  // 获取命令行参数数组的指针和计数器
  char** argv = *pargv;

  // 成功标志
  bool success = true;

  // 输出解析命令行参数的消息
  GlobalInitStream() << "Parsing commandline arguments for c10." << std::endl;

  // 写入未使用参数的位置
  int write_head = 1;

  // 遍历命令行参数数组
  for (int i = 1; i < *pargc; ++i) {
    // 获取参数并转为 string 类型
    string arg(argv[i]);

    // 如果参数中包含 "--help"
    if (arg.find("--help") != string::npos) {
      // 打印使用消息和帮助信息，然后退出程序
      std::cout << UsageMessage() << std::endl;
      std::cout << "Arguments: " << std::endl;
      for (const auto& help_msg : C10FlagsRegistry()->HelpMessage()) {
        std::cout << "    " << help_msg.first << ": " << help_msg.second
                  << std::endl;
      }
      exit(0);
    }

    // 如果参数不是以 "--" 开头，忽略该参数
    if (arg[0] != '-' || arg[1] != '-') {
      GlobalInitStream()
          << "C10 flag: commandline argument does not match --name=var "
             "or --name format: "
          << arg << ". Ignoring this argument." << std::endl;
      argv[write_head++] = argv[i];
      continue;
    }

    // 初始化 key 和 value
    string key;
    string value;

    // 查找参数中的等号位置
    size_t prefix_idx = arg.find('=');

    // 如果没有等号，说明值在下一个参数中
    if (prefix_idx == string::npos) {
      key = arg.substr(2, arg.size() - 2);
      ++i;
      // 如果到达最后一个参数但是缺少值，报错并退出
      if (i == *pargc) {
        GlobalInitStream()
            << "C10 flag: reached the last commandline argument, but "
               "I am expecting a value for "
            << arg;
        success = false;
        break;
      }
      value = string(argv[i]);
    } else {
      // 如果有等号，直接从等号后面取值
      key = arg.substr(2, prefix_idx - 2);
      value = arg.substr(prefix_idx + 1, string::npos);
    }

    // 如果标志未注册，忽略该参数
    // 如果 C10FlagsRegistry 中不存在指定的 key，表示命令行参数不被识别
    if (!C10FlagsRegistry()->Has(key)) {
      // 输出错误信息到全局初始化流，指示未识别的命令行参数
      GlobalInitStream() << "C10 flag: unrecognized commandline argument: "
                         << arg << std::endl;
      // 设置解析过程失败标志为 false
      success = false;
      // 跳出循环，停止进一步解析
      break;
    }
    // 根据 key 和 value 创建一个 C10FlagParser 的唯一指针
    std::unique_ptr<C10FlagParser> parser(
        C10FlagsRegistry()->Create(key, value));
    // 如果解析器标记为失败
    if (!parser->success()) {
      // 输出错误信息到全局初始化流，指示非法的命令行参数
      GlobalInitStream() << "C10 flag: illegal argument: " << arg << std::endl;
      // 设置解析过程失败标志为 false
      success = false;
      // 跳出循环，停止进一步解析
      break;
    }
  }
  // 更新传出参数的值，指向当前解析位置之后的位置
  *pargc = write_head;
  // 标记命令行标志解析完成
  gCommandLineFlagsParsed = true;
  // TODO: 当命令行标志解析失败时，我们是继续，还是直接退出并输出大声通知？
  // 目前我们继续计算，但由于解析失败，后续可能出现问题，因此直接退出可能更合理。
  // 如果解析过程中有失败，输出全局初始化流中的错误信息到标准错误流
  if (!success) {
    std::cerr << GlobalInitStream().str();
  }
  // 清空全局初始化流中的内容
  GlobalInitStream().str(std::string());
  // 返回解析是否成功的标志
  return success;
#ifdef C10_USE_GFLAGS

// 如果定义了宏 C10_USE_GFLAGS，则编译以下代码块


} // namespace c10

// 结束命名空间 c10


#endif // C10_USE_GFLAGS

// 结束条件编译，关闭宏 C10_USE_GFLAGS 的定义检查
```