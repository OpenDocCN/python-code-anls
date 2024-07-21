# `.\pytorch\aten\src\ATen\test\StorageUtils_test.cpp`

```
#include <gtest/gtest.h>

#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <ATen/StorageUtils.h>

using namespace ::testing;

// 定义测试用例 StorageUtilsTest.shm_storage_refcount
TEST(StorageUtilsTest, shm_storage_refcount) {
  // 创建一个 5x5 的长整型张量，所有元素初始化为 7，在 CPU 上
  auto t1 = std::make_unique<at::Tensor>(
      at::full({5, 5}, 7, at::dtype(at::kLong).device(at::kCPU)));
  // t1 的切片，仅包含第一个维度的前 3 行
  auto t2 = std::make_unique<at::Tensor>(t1->slice(0, 0, 3));

  // 创建一个 t1 的克隆张量作为验证用
  auto verificationTensor = t1->clone();
  
  // 断言 t1 和 t2 的存储使用计数为 2
  ASSERT_EQ(t1->storage().use_count(), 2);
  ASSERT_EQ(t2->storage().use_count(), 2);
  
  // 验证张量 verificationTensor 的存储使用计数为 1
  ASSERT_EQ(verificationTensor.storage().use_count(), 1);

  // 将 t1 分享到共享内存
  at::share_memory_(*t1);
  
  // 断言 t1 的存储分配器为 nullptr，即原始存储分配器已分离
  ASSERT_EQ(t1->storage().allocator(), nullptr)
      << "Expect original storage allocator to be detached";
  
  // 验证 verificationTensor 的存储分配器不为 nullptr
  ASSERT_NE(verificationTensor.storage().allocator(), nullptr);
  
  // 再次验证 t1 和 t2 的存储使用计数仍为 2
  ASSERT_EQ(t1->storage().use_count(), 2) << "Expect refcount to be the same";
  ASSERT_EQ(t2->storage().use_count(), 2);

  // 断言 t1 和 verificationTensor 张量的内容相等
  ASSERT_TRUE(t1->equal(verificationTensor));
  
  // 获取 t1 的弱引用存储指针
  auto weakStoragePtr = t1->storage().getWeakStorageImpl();
  
  // 弱引用使用计数为 2（如果存在强引用则再加 1，由于 intrusive_ptr 计数机制）
  ASSERT_EQ(weakStoragePtr.weak_use_count(), 2);
  
  // 重置 t1 和 t2，使其析构，预期 weakStoragePtr 弱引用应该过期
  t1.reset();
  t2.reset();
  ASSERT_TRUE(weakStoragePtr.expired());
}
```