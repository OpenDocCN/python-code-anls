# `.\pytorch\c10\util\Flags.h`

```
#ifndef C10_UTIL_FLAGS_H_
#define C10_UTIL_FLAGS_H_

/* Commandline flags support for C10.
 *
 * This is a portable commandline flags tool for c10, so we can optionally
 * choose to use gflags or a lightweight custom implementation if gflags is
 * not possible on a certain platform. If you have gflags installed, set the
 * macro C10_USE_GFLAGS will seamlessly route everything to gflags.
 *
 * To define a flag foo of type bool default to true, do the following in the
 * *global* namespace:
 *     C10_DEFINE_bool(foo, true, "An example.");
 *
 * To use it in another .cc file, you can use C10_DECLARE_* as follows:
 *     C10_DECLARE_bool(foo);
 *
 * In both cases, you can then access the flag via FLAGS_foo.
 *
 * It is recommended that you build with gflags. To learn more about the flags
 * usage, refer to the gflags page here:
 *
 * https://gflags.github.io/gflags/
 *
 * Note about Python users / devs: gflags is initiated from a C++ function
 * ParseCommandLineFlags, and is usually done in native binaries in the main
 * function. As Python does not have a modifiable main function, it is usually
 * difficult to change the flags after Python starts. Hence, it is recommended
 * that one sets the default value of the flags to one that's acceptable in
 * general - that will allow Python to run without wrong flags.
 */

#include <c10/macros/Export.h>
#include <string>

#include <c10/util/Registry.h>

namespace c10 {
/**
 * Sets the usage message when a commandline tool is called with "--help".
 */
C10_API void SetUsageMessage(const std::string& str);

/**
 * Returns the usage message for the commandline tool set by SetUsageMessage.
 */
C10_API const char* UsageMessage();

/**
 * Parses the commandline flags.
 *
 * This command parses all the commandline arguments passed in via pargc
 * and argv. Once it is finished, partc and argv will contain the remaining
 * commandline args that c10 does not deal with. Note that following
 * convention, argv[0] contains the binary name and is not parsed.
 */
C10_API bool ParseCommandLineFlags(int* pargc, char*** pargv);

/**
 * Checks if the commandline flags has already been passed.
 */
C10_API bool CommandLineFlagsHasBeenParsed();

} // namespace c10

////////////////////////////////////////////////////////////////////////////////
// Below are gflags and non-gflags specific implementations.
// In general, they define the following macros for one to declare (use
// C10_DECLARE) or define (use C10_DEFINE) flags:
// C10_{DECLARE,DEFINE}_{int,int64,double,bool,string}
////////////////////////////////////////////////////////////////////////////////

#ifdef C10_USE_GFLAGS

////////////////////////////////////////////////////////////////////////////////
// Begin gflags section: most functions are basically rerouted to gflags.
////////////////////////////////////////////////////////////////////////////////
#include <gflags/gflags.h>

// C10 uses hidden visibility by default. However, in gflags, it only uses
// 在 Windows 平台上使用 dllexport 导出，但在 Linux/Mac 上使用默认可见性不导出。因此，
// 为了确保总是导出全局变量，我们将在构建 C10 作为共享库时重新定义 GFLAGS_DLL_DEFINE_FLAG 宏。
// 这必须在包含 gflags 之后完成，因为一些早期版本的 gflags.h（例如 Ubuntu 14.04 上的 2.0 版本）
// 直接定义了这些宏，所以我们需要在 gflags 完成后进行重新定义。
#ifdef GFLAGS_DLL_DEFINE_FLAG
#undef GFLAGS_DLL_DEFINE_FLAG
#endif // GFLAGS_DLL_DEFINE_FLAG
#ifdef GFLAGS_DLL_DECLARE_FLAG
#undef GFLAGS_DLL_DECLARE_FLAG
#endif // GFLAGS_DLL_DECLARE_FLAG
#define GFLAGS_DLL_DEFINE_FLAG C10_EXPORT
#define GFLAGS_DLL_DECLARE_FLAG C10_IMPORT

// gflags 在 2.0 版本之前使用 google 命名空间，2.1 版本及之后使用 gflags 命名空间。
// 使用 GFLAGS_GFLAGS_H_ 宏来捕捉这个变化。
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif // GFLAGS_GFLAGS_H_

// 关于 gflags 封装的动机：
// (1) 我们需要确保 gflags 版本和非 gflags 版本的 C10 暴露相同的标志抽象。应明确使用 FLAGS_flag_name 访问标志。
// (2) 对于标志名称，建议以 c10_ 开头以区分它们与常规 gflags 标志。例如，使用 C10_DEFINE_BOOL(c10_my_flag, true, "An example");
// 可以使用 FLAGS_c10_my_flag 访问它。
// (3) Gflags 存在一个设计问题，即在使用 -fvisibility=hidden 编译库时无法正确暴露全局标志。
// 当前的 gflags（截至 2018 年 8 月）仅处理 Windows 情况，使用 dllexport，而不处理 Linux 等情况。
// 因此，我们显式使用 C10_EXPORT 来导出 C10 中定义的标志。这通过全局引用实现，因此标志本身不会重复 - 在幕后，它是同一个全局 gflags 标志。
#define C10_GFLAGS_DEF_WRAPPER(type, real_type, name, default_value, help_str) \
  DEFINE_##type(name, default_value, help_str);

#define C10_DEFINE_int(name, default_value, help_str) \
  C10_GFLAGS_DEF_WRAPPER(int32, gflags::int32, name, default_value, help_str)
#define C10_DEFINE_int32(name, default_value, help_str) \
  C10_DEFINE_int(name, default_value, help_str)
#define C10_DEFINE_int64(name, default_value, help_str) \
  C10_GFLAGS_DEF_WRAPPER(int64, gflags::int64, name, default_value, help_str)
#define C10_DEFINE_double(name, default_value, help_str) \
  C10_GFLAGS_DEF_WRAPPER(double, double, name, default_value, help_str)
#define C10_DEFINE_bool(name, default_value, help_str) \
  C10_GFLAGS_DEF_WRAPPER(bool, bool, name, default_value, help_str)
#define C10_DEFINE_string(name, default_value, help_str) \
  C10_GFLAGS_DEF_WRAPPER(string, ::fLS::clstring, name, default_value, help_str)

// DECLARE_typed_var 应该在头文件和全局命名空间中使用。
#define C10_GFLAGS_DECLARE_WRAPPER(type, real_type, name) DECLARE_##type(name);
#define C10_DECLARE_int(name) \
  C10_GFLAGS_DECLARE_WRAPPER(int32, gflags::int32, name)
#define C10_DECLARE_int32(name) C10_DECLARE_int(name)
#define C10_DECLARE_int64(name) \
  C10_GFLAGS_DECLARE_WRAPPER(int64, gflags::int64, name)
#define C10_DECLARE_double(name) \
  C10_GFLAGS_DECLARE_WRAPPER(double, double, name)
#define C10_DECLARE_bool(name) C10_GFLAGS_DECLARE_WRAPPER(bool, bool, name)
#define C10_DECLARE_string(name) \
  C10_GFLAGS_DECLARE_WRAPPER(string, ::fLS::clstring, name)



// 定义宏 C10_DECLARE_int，将 name 参数声明为一个 int32 类型的 gflags 变量
#define C10_DECLARE_int(name) \
  C10_GFLAGS_DECLARE_WRAPPER(int32, gflags::int32, name)

// 定义宏 C10_DECLARE_int32，将其映射到 C10_DECLARE_int
#define C10_DECLARE_int32(name) C10_DECLARE_int(name)

// 定义宏 C10_DECLARE_int64，将 name 参数声明为一个 int64 类型的 gflags 变量
#define C10_DECLARE_int64(name) \
  C10_GFLAGS_DECLARE_WRAPPER(int64, gflags::int64, name)

// 定义宏 C10_DECLARE_double，将 name 参数声明为一个 double 类型的变量
#define C10_DECLARE_double(name) \
  C10_GFLAGS_DECLARE_WRAPPER(double, double, name)

// 定义宏 C10_DECLARE_bool，将 name 参数声明为一个 bool 类型的变量
#define C10_DECLARE_bool(name) C10_GFLAGS_DECLARE_WRAPPER(bool, bool, name)

// 定义宏 C10_DECLARE_string，将 name 参数声明为一个 string 类型的 fLS::clstring 变量
#define C10_DECLARE_string(name) \
  C10_GFLAGS_DECLARE_WRAPPER(string, ::fLS::clstring, name)



////////////////////////////////////////////////////////////////////////////////
// End gflags section.
////////////////////////////////////////////////////////////////////////////////



// gflags 部分结束的注释
#endif // C10_USE_GFLAGS



////////////////////////////////////////////////////////////////////////////////
// Begin non-gflags section: providing equivalent functionality.
////////////////////////////////////////////////////////////////////////////////

namespace c10 {

// C10FlagParser 类的定义，提供了解析标志的功能
class C10_API C10FlagParser {
 public:
  // 判断标志是否解析成功的方法
  bool success() {
    return success_;
  }

 protected:
  // 模板方法，用于解析标志的具体实现，子类需要实现此方法
  template <typename T>
  bool Parse(const std::string& content, T* value);
  
  // 标志解析是否成功的标识
  bool success_{false};
};

// C10FlagsRegistry 类的声明，用于注册标志
C10_DECLARE_REGISTRY(C10FlagsRegistry, C10FlagParser, const std::string&);

} // namespace c10

// 宏的定义在 c10 命名空间外部，因此在使用时也应在任何命名空间外部定义



#define C10_DEFINE_typed_var(type, name, default_value, help_str)       \
  C10_EXPORT type FLAGS_##name = default_value;                         \
  namespace c10 {                                                       \
  namespace {                                                           \
  // 定义了特定类型变量的标志解析器类，继承自 C10FlagParser
  class C10FlagParser_##name : public C10FlagParser {                   \
   public:                                                              \
    // 构造函数，根据内容解析标志并设置成功标识
    explicit C10FlagParser_##name(const std::string& content) {         \
      success_ = C10FlagParser::Parse<type>(content, &FLAGS_##name);    \
    }                                                                   \
  };                                                                    \
  }                                                                     \
  // 注册器对象，用于注册标志解析器和其描述信息
  RegistererC10FlagsRegistry g_C10FlagsRegistry_##name(                 \
      #name,                                                            \
      C10FlagsRegistry(),                                               \
      RegistererC10FlagsRegistry::DefaultCreator<C10FlagParser_##name>, \
      "(" #type ", default " #default_value ") " help_str);             \
  }

// 定义宏 C10_DEFINE_int，用于定义一个 int 类型的标志变量
#define C10_DEFINE_int(name, default_value, help_str) \
  C10_DEFINE_typed_var(int, name, default_value, help_str)

// 定义宏 C10_DEFINE_int32，将其映射到 C10_DEFINE_int
#define C10_DEFINE_int32(name, default_value, help_str) \
  C10_DEFINE_int(name, default_value, help_str)

// 定义宏 C10_DEFINE_int64，用于定义一个 int64_t 类型的标志变量
#define C10_DEFINE_int64(name, default_value, help_str) \
  C10_DEFINE_typed_var(int64_t, name, default_value, help_str)



#endif // End non-gflags section



// non-gflags 部分结束的注释
// 定义一个宏，用于定义一个双精度浮点类型的命令行标志
#define C10_DEFINE_double(name, default_value, help_str) \
  C10_DEFINE_typed_var(double, name, default_value, help_str)

// 定义一个宏，用于定义一个布尔类型的命令行标志
#define C10_DEFINE_bool(name, default_value, help_str) \
  C10_DEFINE_typed_var(bool, name, default_value, help_str)

// 定义一个宏，用于定义一个字符串类型的命令行标志
#define C10_DEFINE_string(name, default_value, help_str) \
  C10_DEFINE_typed_var(std::string, name, default_value, help_str)

// 声明一个命令行标志的变量，在头文件和全局命名空间中使用
#define C10_DECLARE_typed_var(type, name) C10_API extern type FLAGS_##name

// 声明一个整数类型的命令行标志
#define C10_DECLARE_int(name) C10_DECLARE_typed_var(int, name)

// 声明一个32位整数类型的命令行标志
#define C10_DECLARE_int32(name) C10_DECLARE_int(name)

// 声明一个64位整数类型的命令行标志
#define C10_DECLARE_int64(name) C10_DECLARE_typed_var(int64_t, name)

// 声明一个双精度浮点类型的命令行标志
#define C10_DECLARE_double(name) C10_DECLARE_typed_var(double, name)

// 声明一个布尔类型的命令行标志
#define C10_DECLARE_bool(name) C10_DECLARE_typed_var(bool, name)

// 声明一个字符串类型的命令行标志
#define C10_DECLARE_string(name) C10_DECLARE_typed_var(std::string, name)

////////////////////////////////////////////////////////////////////////////////
// End non-gflags section.
////////////////////////////////////////////////////////////////////////////////

#endif // C10_USE_GFLAGS

#endif // C10_UTIL_FLAGS_H_
```