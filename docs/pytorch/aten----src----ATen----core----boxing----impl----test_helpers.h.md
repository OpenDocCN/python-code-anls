# `.\pytorch\aten\src\ATen\core\boxing\impl\test_helpers.h`

```py
#pragma once

#include <gtest/gtest.h>  // 包含 Google Test 的头文件
#include <gmock/gmock.h>  // 包含 Google Mock 的头文件

#include <ATen/core/Tensor.h>  // 包含 PyTorch ATen 库中的 Tensor 头文件
#include <ATen/core/dispatch/Dispatcher.h>  // 包含 PyTorch ATen 库中的 Dispatcher 头文件
#include <ATen/core/ivalue.h>  // 包含 PyTorch ATen 库中的 IValue 头文件
#include <c10/core/CPUAllocator.h>  // 包含 PyTorch c10 库中的 CPUAllocator 头文件
#include <c10/util/irange.h>  // 包含 PyTorch c10 库中的 irange 头文件

template<class... Inputs>
inline std::vector<c10::IValue> makeStack(Inputs&&... inputs) {
  return {std::forward<Inputs>(inputs)...};  // 创建一个 c10::IValue 类型的向量并返回，包含传入的所有参数
}

inline at::Tensor dummyTensor(c10::DispatchKeySet ks, bool requires_grad=false) {
  auto* allocator = c10::GetCPUAllocator();  // 获取 CPUAllocator 实例指针
  int64_t nelements = 1;  // 元素个数设为1
  auto dtype = caffe2::TypeMeta::Make<float>();  // 创建一个 float 类型的 TypeMeta 实例
  int64_t size_bytes = nelements * dtype.itemsize();  // 计算存储空间的字节数
  auto storage_impl = c10::make_intrusive<c10::StorageImpl>(  // 创建 StorageImpl 实例
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),  // 分配指定大小的内存
      allocator,
      /*resizable=*/true);  // 设置为可调整大小的存储
  at::Tensor t = at::detail::make_tensor<c10::TensorImpl>(storage_impl, ks, dtype);  // 创建 Tensor 实例 t
  // TODO: 我们添加这个来模拟理想情况，即仅在 Tensor 需要梯度时才有 Autograd 后端键
  //       但当前 Autograd 键默认在 TensorImpl 构造函数中添加。
  if (!requires_grad) {
    t.unsafeGetTensorImpl()->remove_autograd_key();  // 如果不需要梯度，移除 Autograd 后端键
  }
  return t;  // 返回创建的 Tensor 实例
}

inline at::Tensor dummyTensor(c10::DispatchKey dispatch_key, bool requires_grad=false) {
  return dummyTensor(c10::DispatchKeySet(dispatch_key), requires_grad);  // 调用上面的函数，将 dispatch_key 转换为 DispatchKeySet
}

template<class... Args>
inline std::vector<c10::IValue> callOp(const c10::OperatorHandle& op, Args... args) {
  auto stack = makeStack(std::forward<Args>(args)...);  // 创建输入参数的 IValue 栈
  op.callBoxed(&stack);  // 调用操作符的 callBoxed 方法，传入参数栈的地址
  return stack;  // 返回处理后的参数栈
}

template<class Result, class... Args>
inline Result callOpUnboxed(const c10::OperatorHandle& op, Args... args) {
  return op.typed<Result(Args...)>().call(std::forward<Args>(args)...);  // 调用操作符的 call 方法，并返回结果
}

template<class Result, class... Args>
inline Result callOpUnboxedWithDispatchKey(const c10::OperatorHandle& op, c10::DispatchKey dispatchKey, Args... args) {
  return op.typed<Result(Args...)>().callWithDispatchKey(dispatchKey, std::forward<Args>(args)...);  // 使用指定的 dispatchKey 调用操作符的 callWithDispatchKey 方法
}

template<class Result, class... Args>
inline Result callOpUnboxedWithPrecomputedDispatchKeySet(const c10::OperatorHandle& op, c10::DispatchKeySet ks, Args... args) {
  return op.typed<Result(Args...)>().redispatch(ks, std::forward<Args>(args)...);  // 使用预先计算的 DispatchKeySet 重新调度操作符
}

inline void expectDoesntFindKernel(const char* op_name, c10::DispatchKey dispatch_key) {
  auto op = c10::Dispatcher::singleton().findSchema({op_name, ""});  // 查找指定操作名的操作模式
  EXPECT_ANY_THROW(
    callOp(*op, dummyTensor(dispatch_key), 5);  // 期望在调用 op 时抛出异常，传入一个虚拟的 Tensor 和一个整数作为参数
  );
}

inline void expectDoesntFindOperator(const char* op_name) {
  auto op = c10::Dispatcher::singleton().findSchema({op_name, ""});  // 查找指定操作名的操作模式
  EXPECT_FALSE(op.has_value());  // 期望找不到该操作模式
}

template<class Exception, class Functor>
inline void expectThrows(Functor&& functor, const char* expectMessageContains) {
  try {
    std::forward<Functor>(functor)();  // 执行传入的可调用对象 functor
  } catch (const Exception& e) {
    EXPECT_THAT(e.what(), testing::HasSubstr(expectMessageContains));  // 期望捕获到特定类型的异常，并检查异常消息中是否包含特定字符串
    return;
  }


// 如果前面的条件不满足，直接返回，终止函数执行
    ADD_FAILURE() << "Expected to throw exception containing \""
    << expectMessageContains << "\" but didn't throw";


// 使用 Google Test 的 ADD_FAILURE() 宏来生成一个测试失败的消息，
// 指示预期的异常未被抛出，并输出预期的异常消息片段
}

// 模板函数：比较期望的数组引用和实际的 std::array 的元素是否相等
template<class T, size_t N>
void expectListEquals(c10::ArrayRef<T> expected, std::array<T, N> actual) {
  // 检查期望数组和实际数组的大小是否相等
  EXPECT_EQ(expected.size(), actual.size());
  // 遍历期望数组和实际数组，逐个比较它们的元素
  for (const auto i : c10::irange(expected.size())) {
    EXPECT_EQ(expected[i], actual[i]);
  }
}

// 模板函数：比较两个数组引用的元素是否相等
template<class T>
void expectListEquals(c10::ArrayRef<T> expected, c10::ArrayRef<T> actual) {
  // 检查期望数组和实际数组的大小是否相等
  EXPECT_EQ(expected.size(), actual.size());
  // 遍历期望数组和实际数组，逐个比较它们的元素
  for (const auto i : c10::irange(expected.size())) {
    EXPECT_EQ(expected[i], actual[i]);
  }
}

// 模板函数：比较期望的数组引用和实际的 c10::List 的元素是否相等
template<class T>
void expectListEquals(c10::ArrayRef<T> expected, c10::List<T> actual) {
  // 检查期望数组和实际 List 的大小是否相等
  EXPECT_EQ(expected.size(), actual.size());
  // 遍历期望数组和实际 List，逐个比较它们的元素
  for (const auto i : c10::irange(expected.size())) {
    EXPECT_EQ(expected[i], actual.get(i));
  }
}

// 模板函数：比较期望的数组引用和实际的 std::vector 的元素是否相等
template<class T>
void expectListEquals(c10::ArrayRef<T> expected, std::vector<T> actual) {
  // 检查期望数组和实际 vector 的大小是否相等
  EXPECT_EQ(expected.size(), actual.size());
  // 遍历期望数组和实际 vector，逐个比较它们的元素
  for (const auto i : c10::irange(expected.size())) {
    EXPECT_EQ(expected[i], actual[i]);
  }
}

// 注意：这里的设计并不完全健全，但是所有的类型集合都是单例的，所以这种设计是可接受的
// 静态内联函数：从给定的张量中提取分发键
static inline c10::DispatchKey extractDispatchKey(const at::Tensor& t) {
  return legacyExtractDispatchKey(t.key_set());
}
```