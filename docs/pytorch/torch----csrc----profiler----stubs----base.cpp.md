# `.\pytorch\torch\csrc\profiler\stubs\base.cpp`

```py
#include <torch/csrc/profiler/stubs/base.h>

#include <c10/util/Exception.h>

namespace torch {
namespace profiler {
namespace impl {

// 定义 ProfilerStubs 类的析构函数，使用默认实现
ProfilerStubs::~ProfilerStubs() = default;

namespace {
// 默认的 ProfilerStubs 实现类 DefaultStubs，继承自 ProfilerStubs
struct DefaultStubs : public ProfilerStubs {
  // 构造函数，初始化名称
  DefaultStubs(const char* name) : name_{name} {}

  // 下面是一系列虚函数的具体实现，都调用了 fail() 函数来报错
  void record(c10::DeviceIndex*, ProfilerVoidEventStub*, int64_t*)
      const override {
    fail();
  }
  float elapsed(const ProfilerVoidEventStub*, const ProfilerVoidEventStub*)
      const override {
    fail();
    return 0.f;
  }
  void mark(const char*) const override {
    fail();
  }
  void rangePush(const char*) const override {
    fail();
  }
  void rangePop() const override {
    fail();
  }
  bool enabled() const override {
    return false;
  }
  void onEachDevice(std::function<void(int)>) const override {
    fail();
  }
  void synchronize() const override {
    fail();
  }
  // 默认析构函数
  ~DefaultStubs() override = default;

 private:
  // 私有函数，调用 AT_ERROR 抛出异常
  void fail() const {
    AT_ERROR(name_, " used in profiler but not enabled.");
  }

  const char* const name_;  // 存储名称的常量指针
};
} // namespace

// 定义宏 REGISTER_DEFAULT，注册默认的 ProfilerStubs 实现
#define REGISTER_DEFAULT(name, upper_name)                                   \
  namespace {                                                                \
  // 声明一个名为 default_##name##_stubs 的常量 DefaultStubs 实例
  const DefaultStubs default_##name##_stubs{#upper_name};                    \
  // 声明一个指向 default_##name##_stubs 的常量指针
  constexpr const DefaultStubs* default_##name##_stubs_addr =                \
      &default_##name##_stubs;                                               \
                                                                             \
  /* Constant initialization, so it is guaranteed to be initialized before*/ \
  /* static initialization calls which may invoke register<name>Methods*/    \
  // 内联函数，返回一个指向 ProfilerStubs 的常量指针引用
  inline const ProfilerStubs*& name##_stubs() {                              \
    // 静态局部变量，保存 static_cast 后的 default_##name##_stubs_addr
    static const ProfilerStubs* stubs_ =                                     \
        static_cast<const ProfilerStubs*>(default_##name##_stubs_addr);      \
    return stubs_;                                                           \
  }                                                                          \
  } /*namespace*/                                                            \
                                                                             \
  // 返回 name##_stubs() 的结果，将其转换为 const ProfilerStubs*
  const ProfilerStubs* name##Stubs() {                                       \
    return name##_stubs();                                                   \
  }                                                                          \
                                                                             \
  // 注册 name##Methods 的方法，将传入的 ProfilerStubs* 赋值给 name##_stubs()
  void register##upper_name##Methods(ProfilerStubs* stubs) {                 \
    name##_stubs() = stubs;                                                  \
  }

// 使用宏注册各种默认的 ProfilerStubs
REGISTER_DEFAULT(cuda, CUDA)
REGISTER_DEFAULT(itt, ITT)
REGISTER_DEFAULT(privateuse1, PrivateUse1)
#undef REGISTER_DEFAULT

} // namespace impl
} // namespace profiler
} // namespace torch
```