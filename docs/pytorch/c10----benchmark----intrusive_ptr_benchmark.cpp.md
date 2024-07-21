# `.\pytorch\c10\benchmark\intrusive_ptr_benchmark.cpp`

```py
// 包含 C10 库中的相关头文件
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>

// 包含 Google Benchmark 库的头文件
#include <benchmark/benchmark.h>
#include <memory> // 包含标准库中的内存管理相关头文件

// 使用 C10 命名空间
using c10::intrusive_ptr;
using c10::intrusive_ptr_target;
using c10::make_intrusive;

// 匿名命名空间，用于定义局部作用域的类和函数

// 使用 intrusive_ptr 的类 Foo，继承自 intrusive_ptr_target
class Foo : public intrusive_ptr_target {
 public:
  Foo(int param_) : param(param_) {} // Foo 类的构造函数，初始化 param 成员变量
  int param; // Foo 类的成员变量，整数类型的 param
};

// 使用 std::shared_ptr 的类 Bar，使用 enable_shared_from_this 支持从 this 创建 shared_ptr
class Bar : public std::enable_shared_from_this<Bar> {
 public:
  Bar(int param_) : param(param_) {} // Bar 类的构造函数，初始化 param 成员变量
  int param; // Bar 类的成员变量，整数类型的 param
};

// Google Benchmark 测试函数，测试 intrusive_ptr 的构造和析构
static void BM_IntrusivePtrCtorDtor(benchmark::State& state) {
  intrusive_ptr<Foo> var = make_intrusive<Foo>(0); // 使用 intrusive_ptr 构造 Foo 类对象 var
  while (state.KeepRunning()) {
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    volatile intrusive_ptr<Foo> var2 = var; // 使用 volatile 修饰的局部变量 var2
  }
}
BENCHMARK(BM_IntrusivePtrCtorDtor); // 注册 BM_IntrusivePtrCtorDtor 函数到 Google Benchmark

// Google Benchmark 测试函数，测试 std::shared_ptr 的构造和析构
static void BM_SharedPtrCtorDtor(benchmark::State& state) {
  std::shared_ptr<Bar> var = std::make_shared<Bar>(0); // 使用 std::shared_ptr 构造 Bar 类对象 var
  while (state.KeepRunning()) {
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    volatile std::shared_ptr<Bar> var2 = var; // 使用 volatile 修饰的局部变量 var2
  }
}
BENCHMARK(BM_SharedPtrCtorDtor); // 注册 BM_SharedPtrCtorDtor 函数到 Google Benchmark

// Google Benchmark 测试函数，测试 intrusive_ptr 数组的使用
static void BM_IntrusivePtrArray(benchmark::State& state) {
  intrusive_ptr<Foo> var = make_intrusive<Foo>(0); // 使用 intrusive_ptr 构造 Foo 类对象 var
  const size_t kLength = state.range(0); // 获取测试参数范围的长度
  std::vector<intrusive_ptr<Foo>> vararray(kLength); // 使用 intrusive_ptr 的 Foo 类对象数组
  while (state.KeepRunning()) {
    for (const auto i : c10::irange(kLength)) {
      vararray[i] = var; // 将 var 赋值给 vararray 数组的元素
    }
    for (const auto i : c10::irange(kLength)) {
      vararray[i].reset(); // 重置 vararray 数组的元素
    }
  }
}
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-avoid-magic-numbers)
BENCHMARK(BM_IntrusivePtrArray)->RangeMultiplier(2)->Range(16, 4096); // 注册 BM_IntrusivePtrArray 函数到 Google Benchmark，指定参数范围

// Google Benchmark 测试函数，测试 std::shared_ptr 数组的使用
static void BM_SharedPtrArray(benchmark::State& state) {
  std::shared_ptr<Bar> var = std::make_shared<Bar>(0); // 使用 std::shared_ptr 构造 Bar 类对象 var
  const size_t kLength = state.range(0); // 获取测试参数范围的长度
  std::vector<std::shared_ptr<Bar>> vararray(kLength); // 使用 std::shared_ptr 的 Bar 类对象数组
  while (state.KeepRunning()) {
    for (const auto i : c10::irange(kLength)) {
      vararray[i] = var; // 将 var 赋值给 vararray 数组的元素
    }
    for (const auto i : c10::irange(kLength)) {
      vararray[i].reset(); // 重置 vararray 数组的元素
    }
  }
}
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-avoid-magic-numbers)
BENCHMARK(BM_SharedPtrArray)->RangeMultiplier(2)->Range(16, 4096); // 注册 BM_SharedPtrArray 函数到 Google Benchmark，指定参数范围

// Google Benchmark 测试函数，测试 intrusive_ptr 独占所有权的性能
static void BM_IntrusivePtrExclusiveOwnership(benchmark::State& state) {
  while (state.KeepRunning()) {
    volatile auto var = make_intrusive<Foo>(0); // 使用 volatile 修饰的局部变量 var
  }
}
BENCHMARK(BM_IntrusivePtrExclusiveOwnership); // 注册 BM_IntrusivePtrExclusiveOwnership 函数到 Google Benchmark

// Google Benchmark 测试函数，测试 std::shared_ptr 独占所有权的性能
static void BM_SharedPtrExclusiveOwnership(benchmark::State& state) {
  while (state.KeepRunning()) {
    volatile auto var = std::make_shared<Foo>(0); // 使用 volatile 修饰的局部变量 var
  }
}
BENCHMARK(BM_SharedPtrExclusiveOwnership); // 注册 BM_SharedPtrExclusiveOwnership 函数到 Google Benchmark

// 结束匿名命名空间
} // namespace

BENCHMARK_MAIN(); // 定义 Google Benchmark 主函数入口
```