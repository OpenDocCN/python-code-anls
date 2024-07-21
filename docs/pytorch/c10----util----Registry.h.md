# `.\pytorch\c10\util\Registry.h`

```py
// 防止头文件重复包含的宏定义
#ifndef C10_UTIL_REGISTRY_H_
#define C10_UTIL_REGISTRY_H_

/**
 * Simple registry implementation that uses static variables to
 * register object creators during program initialization time.
 */
// 简单的注册表实现，使用静态变量在程序初始化时注册对象创建者

// 注意事项：当使用其他命名空间时，此注册表的工作效果较差。
// 所有宏的调用应从 at 命名空间内部进行。

#include <cstdio>           // C标准输入输出库
#include <cstdlib>          // C标准库
#include <functional>       // C++ 函数对象
#include <memory>           // C++ 智能指针
#include <mutex>            // C++ 互斥锁
#include <stdexcept>        // C++ 标准异常类
#include <string>           // C++ 字符串
#include <unordered_map>    // C++ 无序映射容器
#include <vector>           // C++ 向量容器

#include <c10/macros/Export.h>   // C10 导出宏定义
#include <c10/macros/Macros.h>   // C10 宏定义
#include <c10/util/Type.h>       // C10 类型工具

namespace c10 {

template <typename KeyType>
inline std::string KeyStrRepr(const KeyType& /*key*/) {
  return "[key type printing not supported]";
}

template <>
inline std::string KeyStrRepr(const std::string& key) {
  return key;
}

enum RegistryPriority {
  REGISTRY_FALLBACK = 1,    // 注册表优先级：后备
  REGISTRY_DEFAULT = 2,     // 注册表优先级：默认
  REGISTRY_PREFERRED = 3,   // 注册表优先级：首选
};

/**
 * @brief A template class that allows one to register classes by keys.
 *
 * The keys are usually a std::string specifying the name, but can be anything
 * that can be used in a std::map.
 *
 * You should most likely not use the Registry class explicitly, but use the
 * helper macros below to declare specific registries as well as registering
 * objects.
 */
template <class SrcType, class ObjectPtrType, class... Args>
class Registry {
 public:
  typedef std::function<ObjectPtrType(Args...)> Creator;

  // 注册表构造函数，默认警告为开启状态
  Registry(bool warning = true) : registry_(), priority_(), warning_(warning) {}

  // 注册函数，注册指定键和创建函数
  void Register(
      const SrcType& key,
      Creator creator,
      const RegistryPriority priority = REGISTRY_DEFAULT) {
    std::lock_guard<std::mutex> lock(register_mutex_);
    // 下面的 if 语句与以下一行实际上是等价的：
    // TORCH_CHECK_EQ(registry_.count(key), 0) << "Key " << key
    //                                   << " registered twice.";
    // 但是，TORCH_CHECK_EQ 依赖于 Google 日志，由于注册是在静态初始化时进行的，
    // 我们不希望显式依赖于 glog 的初始化函数。

    // 如果已经有相同的键注册，根据优先级决定是否覆盖
    if (registry_.count(key) != 0) {
      auto cur_priority = priority_[key];
      if (priority > cur_priority) {
#ifdef DEBUG
        // 输出警告信息，说明正在覆盖已注册的项
        std::string warn_msg =
            "Overwriting already registered item for key " + KeyStrRepr(key);
        fprintf(stderr, "%s\n", warn_msg.c_str());
#endif
        // 更新注册的对象和优先级
        registry_[key] = creator;
        priority_[key] = priority;
      }
    } else {
      // 如果没有该键的注册，直接注册新的对象和优先级
      registry_[key] = creator;
      priority_[key] = priority;
    }
  }

 private:
  std::unordered_map<SrcType, Creator> registry_;   // 注册表，键为 SrcType，值为 Creator
  std::unordered_map<SrcType, RegistryPriority> priority_;  // 注册表的优先级
  std::mutex register_mutex_;   // 注册表操作的互斥锁
  bool warning_;    // 是否开启警告
};

} // namespace c10

#endif // C10_UTIL_REGISTRY_H_
#endif
      // 如果注册表中不存在这个键，则将其注册进去
      registry_[key] = creator;
      // 设置键的优先级
      priority_[key] = priority;
    } else if (priority == cur_priority) {
      // 如果优先级与当前注册的优先级相同，则报错
      std::string err_msg =
          "Key already registered with the same priority: " + KeyStrRepr(key);
      fprintf(stderr, "%s\n", err_msg.c_str());
      // 如果设置了 terminate_，则退出程序
      if (terminate_) {
        std::exit(1);
      } else {
        // 否则抛出运行时异常
        throw std::runtime_error(err_msg);
      }
    } else if (warning_) {
      // 如果设置了 warning_，则输出警告信息
      std::string warn_msg =
          "Higher priority item already registered, skipping registration of " +
          KeyStrRepr(key);
      fprintf(stderr, "%s\n", warn_msg.c_str());
    }
  }

  // 注册一个键到注册表中，带有帮助信息
  void Register(
      const SrcType& key,
      Creator creator,
      const std::string& help_msg,
      const RegistryPriority priority = REGISTRY_DEFAULT) {
    // 调用 Register 函数进行注册
    Register(key, creator, priority);
    // 将帮助信息关联到键上
    help_message_[key] = help_msg;
  }

  // 判断注册表中是否存在某个键
  inline bool Has(const SrcType& key) {
    return (registry_.count(key) != 0);
  }

  // 根据键创建对象指针
  ObjectPtrType Create(const SrcType& key, Args... args) {
    auto it = registry_.find(key);
    if (it == registry_.end()) {
      // 如果键不存在，则返回 nullptr
      // 如果需要创建的对象未注册，则返回 nullptr
      return nullptr;
    }
    // 调用注册的对象创建函数来创建对象
    return it->second(args...);
  }

  /**
   * 返回当前注册的所有键作为 std::vector。
   */
  std::vector<SrcType> Keys() const {
    // 创建一个空的向量用于存储所有的键
    std::vector<SrcType> keys;
    // 预留足够的空间以容纳所有的键
    keys.reserve(registry_.size());
    // 遍历注册表并将所有的键添加到向量中
    for (const auto& it : registry_) {
      keys.push_back(it.first);
    }
    // 返回存储所有键的向量
    return keys;
  }

  // 返回帮助信息的引用
  inline const std::unordered_map<SrcType, std::string>& HelpMessage() const {
    return help_message_;
  }

  // 返回特定键的帮助信息
  const char* HelpMessage(const SrcType& key) const {
    auto it = help_message_.find(key);
    if (it == help_message_.end()) {
      // 如果键不存在，则返回 nullptr
      return nullptr;
    }
    // 否则返回帮助信息的 C 字符串形式
    return it->second.c_str();
  }

  // 用于测试，如果未设置 terminate_，Registry 将抛出异常而不是调用 std::exit
  void SetTerminate(bool terminate) {
    terminate_ = terminate;
  }

 private:
  // 存储键到创建函数的映射
  std::unordered_map<SrcType, Creator> registry_;
  // 存储键到优先级的映射
  std::unordered_map<SrcType, RegistryPriority> priority_;
  // 确定是否调用 std::exit 的标志
  bool terminate_{true};
  // 确定是否输出警告信息的标志
  const bool warning_;
  // 存储键到帮助信息的映射
  std::unordered_map<SrcType, std::string> help_message_;
  // 用于注册时的互斥锁
  std::mutex register_mutex_;

  // 禁止复制和赋值的宏，防止意外的复制
  C10_DISABLE_COPY_AND_ASSIGN(Registry);
};
    # 使用注册表对象调用Register方法，注册一个对象的创建器及相关信息
    registry->Register(key, creator, help_msg, priority);
  }

  # 模板函数，用于创建默认的对象实例
  template <class DerivedType>
  static ObjectPtrType DefaultCreator(Args... args) {
    # 返回一个指向DerivedType类型对象的智能指针，使用给定的参数args进行构造
    return ObjectPtrType(new DerivedType(args...));
  }
/**
 * C10_DECLARE_TYPED_REGISTRY is a macro that expands to a function
 * declaration, as well as creating a convenient typename for its corresponding
 * registerer.
 */
// 注意：以下C10_IMPORT和C10_EXPORT的说明：
// DECLARE需要显式标记为IMPORT，DEFINE需要标记为EXPORT，因为这些注册宏也会用于下游的共享库中，
// 并且无法使用*_API宏 - API宏将根据每个共享库的基础定义。从语义上讲，声明类型注册表始终是IMPORT，
// 而定义注册表（仅在源文件中进行，且只能进行一次）始终是EXPORT。
//
// 唯一的特殊情况是在同一文件中同时进行DECLARE和DEFINE操作 - 在Windows编译器中，
// 这将生成警告，指出dllimport和dllexport混用，但这个警告是可以接受的，并且链接器将正确导出符号。
// gflags标志声明和定义案例中也是相同情况。
#define C10_DECLARE_TYPED_REGISTRY(                                      \
    RegistryName, SrcType, ObjectType, PtrType, ...)                     \
  C10_API ::c10::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>*  \
  RegistryName();                                                        \
  typedef ::c10::Registerer<SrcType, PtrType<ObjectType>, ##__VA_ARGS__> \
      Registerer##RegistryName

#define TORCH_DECLARE_TYPED_REGISTRY(                                     \
    RegistryName, SrcType, ObjectType, PtrType, ...)                      \
  TORCH_API ::c10::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>* \
  RegistryName();                                                         \
  typedef ::c10::Registerer<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>  \
      Registerer##RegistryName

/**
 * C10_DEFINE_TYPED_REGISTRY is a macro that defines a function for retrieving
 * or creating a registry instance, ensuring it's exported for shared libraries.
 */
#define C10_DEFINE_TYPED_REGISTRY(                                         \
    RegistryName, SrcType, ObjectType, PtrType, ...)                       \
  C10_EXPORT ::c10::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>* \
  RegistryName() {                                                         \
    static ::c10::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>*   \
        registry = new ::c10::                                             \
            Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>();       \
    return registry;                                                       \
  }

/**
 * C10_DEFINE_TYPED_REGISTRY_WITHOUT_WARNING is a variant of C10_DEFINE_TYPED_REGISTRY
 * that starts the definition without suppressing any warnings related to
 * mixed dllimport and dllexport on Windows compilers.
 */
#define C10_DEFINE_TYPED_REGISTRY_WITHOUT_WARNING(                            \
    RegistryName, SrcType, ObjectType, PtrType, ...)                          \
  C10_EXPORT ::c10::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>*    \
  RegistryName() {                                                            \
    static ::c10::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>*      \
        registry =                                                            \
            new ::c10::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>( \
                false);                                                       \
    // 创建静态变量 registry，类型为 ::c10::Registry，用于存储注册表对象
    // 使用 new 关键字在堆上分配内存，初始化一个新的 ::c10::Registry 对象
    // ::c10::Registry 的模板参数为 SrcType、PtrType<ObjectType>，以及可变数量的可选参数 ##__VA_ARGS__
    // 初始化时传入 false，可能表示是否延迟初始化的标志

    return registry;                                                          \
    // 返回静态变量 registry 的指针，该变量用于注册特定类型的对象
  }
// 注册一个具有类型参数化的创建器，允许在模板参数中使用逗号
#define C10_REGISTER_TYPED_CREATOR(RegistryName, key, ...)                  \
  static Registerer##RegistryName C10_ANONYMOUS_VARIABLE(g_##RegistryName)( \
      key, RegistryName(), ##__VA_ARGS__);

// 注册一个带有优先级的类型参数化的创建器，允许在模板参数中使用逗号
#define C10_REGISTER_TYPED_CREATOR_WITH_PRIORITY(                           \
    RegistryName, key, priority, ...)                                       \
  static Registerer##RegistryName C10_ANONYMOUS_VARIABLE(g_##RegistryName)( \
      key, priority, RegistryName(), ##__VA_ARGS__);

// 注册一个带有类型参数化的类，包括默认创建器和类型解析
#define C10_REGISTER_TYPED_CLASS(RegistryName, key, ...)                    \
  static Registerer##RegistryName C10_ANONYMOUS_VARIABLE(g_##RegistryName)( \
      key,                                                                  \
      RegistryName(),                                                       \
      Registerer##RegistryName::DefaultCreator<__VA_ARGS__>,                \
      ::c10::demangle_type<__VA_ARGS__>());

// 注册一个带有优先级的类型参数化的类，包括默认创建器和类型解析
#define C10_REGISTER_TYPED_CLASS_WITH_PRIORITY(                             \
    RegistryName, key, priority, ...)                                       \
  static Registerer##RegistryName C10_ANONYMOUS_VARIABLE(g_##RegistryName)( \
      key,                                                                  \
      priority,                                                             \
      RegistryName(),                                                       \
      Registerer##RegistryName::DefaultCreator<__VA_ARGS__>,                \
      ::c10::demangle_type<__VA_ARGS__>());

// 声明一个类型化的注册表，使用 std::string 作为键类型，默认使用 std::unique_ptr
#define C10_DECLARE_REGISTRY(RegistryName, ObjectType, ...) \
  C10_DECLARE_TYPED_REGISTRY(                               \
      RegistryName, std::string, ObjectType, std::unique_ptr, ##__VA_ARGS__)

// Torch 特定声明，声明一个类型化的注册表，使用 std::string 作为键类型，默认使用 std::unique_ptr
#define TORCH_DECLARE_REGISTRY(RegistryName, ObjectType, ...) \
  TORCH_DECLARE_TYPED_REGISTRY(                               \
      RegistryName, std::string, ObjectType, std::unique_ptr, ##__VA_ARGS__)

// 定义一个类型化的注册表，使用 std::string 作为键类型，默认使用 std::unique_ptr
#define C10_DEFINE_REGISTRY(RegistryName, ObjectType, ...) \
  C10_DEFINE_TYPED_REGISTRY(                               \
      RegistryName, std::string, ObjectType, std::unique_ptr, ##__VA_ARGS__)

// 定义一个不产生警告的类型化的注册表，使用 std::string 作为键类型，默认使用 std::unique_ptr
#define C10_DEFINE_REGISTRY_WITHOUT_WARNING(RegistryName, ObjectType, ...) \
  C10_DEFINE_TYPED_REGISTRY_WITHOUT_WARNING(                               \
      RegistryName, std::string, ObjectType, std::unique_ptr, ##__VA_ARGS__)

// 声明一个使用 std::shared_ptr 作为值类型的类型化的注册表，使用 std::string 作为键类型
#define C10_DECLARE_SHARED_REGISTRY(RegistryName, ObjectType, ...) \
  C10_DECLARE_TYPED_REGISTRY(                                      \
      RegistryName, std::string, ObjectType, std::shared_ptr, ##__VA_ARGS__)


这些宏定义用于在C++代码中注册和声明各种类型化的创建器和注册表。每个宏都有特定的功能，例如注册创建器、注册类、声明注册表等，使用了模板参数化和预处理指令来实现灵活性和通用性。
#define TORCH_DECLARE_SHARED_REGISTRY(RegistryName, ObjectType, ...) \
  TORCH_DECLARE_TYPED_REGISTRY(                                      \
      RegistryName, std::string, ObjectType, std::shared_ptr, ##__VA_ARGS__)

声明宏 `TORCH_DECLARE_SHARED_REGISTRY`，它扩展为调用 `TORCH_DECLARE_TYPED_REGISTRY` 宏，用于声明一个带有类型的注册表。注册表的名称为 `RegistryName`，键类型为 `std::string`，值类型为 `ObjectType` 的 `std::shared_ptr`。可变参数 `...` 是可选的，用于额外的模板参数。


#define C10_DEFINE_SHARED_REGISTRY(RegistryName, ObjectType, ...) \
  C10_DEFINE_TYPED_REGISTRY(                                      \
      RegistryName, std::string, ObjectType, std::shared_ptr, ##__VA_ARGS__)

定义宏 `C10_DEFINE_SHARED_REGISTRY`，它扩展为调用 `C10_DEFINE_TYPED_REGISTRY` 宏，用于定义一个带有类型的注册表。注册表的名称为 `RegistryName`，键类型为 `std::string`，值类型为 `ObjectType` 的 `std::shared_ptr`。可变参数 `...` 是可选的，用于额外的模板参数。


#define C10_DEFINE_SHARED_REGISTRY_WITHOUT_WARNING( \
    RegistryName, ObjectType, ...)                  \
  C10_DEFINE_TYPED_REGISTRY_WITHOUT_WARNING(        \
      RegistryName, std::string, ObjectType, std::shared_ptr, ##__VA_ARGS__)

定义宏 `C10_DEFINE_SHARED_REGISTRY_WITHOUT_WARNING`，它扩展为调用 `C10_DEFINE_TYPED_REGISTRY_WITHOUT_WARNING` 宏，用于定义一个带有类型的注册表，但不生成警告。注册表的名称为 `RegistryName`，键类型为 `std::string`，值类型为 `ObjectType` 的 `std::shared_ptr`。可变参数 `...` 是可选的，用于额外的模板参数。


#define C10_REGISTER_CREATOR(RegistryName, key, ...) \
  C10_REGISTER_TYPED_CREATOR(RegistryName, #key, __VA_ARGS__)

定义宏 `C10_REGISTER_CREATOR`，它扩展为调用 `C10_REGISTER_TYPED_CREATOR` 宏，用于注册创建函数到指定注册表 `RegistryName` 中。注册表的键使用字符串化的 `#key`，用于标识注册的对象类型。可变参数 `...` 是可选的，用于额外的模板参数。


#define C10_REGISTER_CREATOR_WITH_PRIORITY(RegistryName, key, priority, ...) \
  C10_REGISTER_TYPED_CREATOR_WITH_PRIORITY(                                  \
      RegistryName, #key, priority, __VA_ARGS__)

定义宏 `C10_REGISTER_CREATOR_WITH_PRIORITY`，它扩展为调用 `C10_REGISTER_TYPED_CREATOR_WITH_PRIORITY` 宏，用于注册带有优先级的创建函数到指定注册表 `RegistryName` 中。注册表的键使用字符串化的 `#key`，用于标识注册的对象类型。`priority` 参数指定注册的优先级。可变参数 `...` 是可选的，用于额外的模板参数。


#define C10_REGISTER_CLASS(RegistryName, key, ...) \
  C10_REGISTER_TYPED_CLASS(RegistryName, #key, __VA_ARGS__)

定义宏 `C10_REGISTER_CLASS`，它扩展为调用 `C10_REGISTER_TYPED_CLASS` 宏，用于注册类到指定注册表 `RegistryName` 中。注册表的键使用字符串化的 `#key`，用于标识注册的类类型。可变参数 `...` 是可选的，用于额外的模板参数。


#define C10_REGISTER_CLASS_WITH_PRIORITY(RegistryName, key, priority, ...) \
  C10_REGISTER_TYPED_CLASS_WITH_PRIORITY(                                  \
      RegistryName, #key, priority, __VA_ARGS__)

定义宏 `C10_REGISTER_CLASS_WITH_PRIORITY`，它扩展为调用 `C10_REGISTER_TYPED_CLASS_WITH_PRIORITY` 宏，用于注册带有优先级的类到指定注册表 `RegistryName` 中。注册表的键使用字符串化的 `#key`，用于标识注册的类类型。`priority` 参数指定注册的优先级。可变参数 `...` 是可选的，用于额外的模板参数。


} // namespace c10

#endif // C10_UTIL_REGISTRY_H_

结束 `c10` 命名空间，并关闭头文件防止重复包含。
```