# `.\pytorch\c10\test\core\impl\cow_test.cpp`

```py
// 引入所需的头文件：COW.h 和 COWDeleter.h 提供了与 Copy-On-Write 相关的实现。
#include <c10/core/impl/COW.h>
#include <c10/core/impl/COWDeleter.h>

// 引入 CPUAllocator.h 和 StorageImpl.h，提供了内存分配和存储实现的功能。
#include <c10/core/CPUAllocator.h>
#include <c10/core/StorageImpl.h>

// 引入 gmock 和 gtest 提供的测试框架，用于单元测试。
#include <gmock/gmock.h>
#include <gtest/gtest.h>

// 引入标准库头文件
#include <cstddef>   // 提供了标准的大小类型定义
#include <memory>    // 提供了智能指针 std::unique_ptr



// NOLINTBEGIN(clang-analyzer-cplusplus*)
namespace c10::impl {
namespace {

// DeleteTracker 类，用于跟踪对象的删除次数
class DeleteTracker {
 public:
  // 构造函数，初始化时传入一个整数引用，用于记录删除次数
  explicit DeleteTracker(int& delete_count) : delete_count_(delete_count) {}
  
  // 析构函数，在对象销毁时增加删除计数
  ~DeleteTracker() {
    ++delete_count_;
  }

 private:
  int& delete_count_;  // 引用，用于记录删除次数
};

// ContextTest 类，继承自 testing::Test，用于测试上下文相关功能
class ContextTest : public testing::Test {
 protected:
  // 返回当前的删除计数
  auto delete_count() const -> int {
    return delete_count_;
  }
  
  // 创建并返回一个新的 DeleteTracker 对象的智能指针
  auto new_delete_tracker() -> std::unique_ptr<void, DeleterFnPtr> {
    return {new DeleteTracker(delete_count_), +[](void* ptr) {
              delete static_cast<DeleteTracker*>(ptr);
            }};
  }

 private:
  int delete_count_ = 0;  // 初始删除计数为 0
};

// 测试用例 ContextTest.Basic，测试基本上下文操作
TEST_F(ContextTest, Basic) {
  // 创建一个 COWDeleterContext 对象，并使用 new_delete_tracker 返回的 DeleteTracker 进行初始化
  auto& context = *new cow::COWDeleterContext(new_delete_tracker());
  ASSERT_THAT(delete_count(), testing::Eq(0));  // 断言删除计数为 0

  context.increment_refcount();  // 增加引用计数

  {
    // 这是一个子作用域，因为调用 decrement_refcount 预期会给我们一个共享锁
    auto result = context.decrement_refcount();  // 减少引用计数
    ASSERT_THAT(
        std::holds_alternative<cow::COWDeleterContext::NotLastReference>(
            result),
        testing::IsTrue());  // 断言结果是 NotLastReference 类型
    ASSERT_THAT(delete_count(), testing::Eq(0));  // 断言删除计数为 0
  }

  {
    auto result = context.decrement_refcount();  // 再次减少引用计数
    ASSERT_THAT(
        std::holds_alternative<cow::COWDeleterContext::LastReference>(result),
        testing::IsTrue());  // 断言结果是 LastReference 类型
    // 结果持有 DeleteTracker。
    ASSERT_THAT(delete_count(), testing::Eq(0));  // 断言删除计数为 0
  }

  // 当结果被删除时，DeleteTracker 也会被删除。
  ASSERT_THAT(delete_count(), testing::Eq(1));  // 断言删除计数为 1
}

// 测试用例 ContextTest.cow_deleter，测试 cow_deleter 函数
TEST_F(ContextTest, cow_deleter) {
  // 创建一个 COWDeleterContext 对象，并使用 new_delete_tracker 返回的 DeleteTracker 进行初始化
  auto& context = *new cow::COWDeleterContext(new_delete_tracker());
  ASSERT_THAT(delete_count(), testing::Eq(0));  // 断言删除计数为 0

  cow::cow_deleter(&context);  // 调用 cow_deleter 函数
  ASSERT_THAT(delete_count(), testing::Eq(1));  // 断言删除计数为 1
}

// 自定义的 Matcher，用于检查是否为 Copy-On-Write 数据指针
MATCHER(is_copy_on_write, "") {
  const c10::StorageImpl& storage = std::ref(arg);  // 获取 StorageImpl 的引用
  return cow::is_cow_data_ptr(storage.data_ptr());  // 返回是否为 Copy-On-Write 数据指针的检查结果
}
TEST(lazy_clone_storage_test, no_context) {
  // 创建一个原始的 StorageImpl 对象，使用默认的 CPU 分配器和固定大小，不可调整大小
  StorageImpl original_storage(
      {}, /*size_bytes=*/7, GetDefaultCPUAllocator(), /*resizable=*/false);
  // 断言原始的 storage 不是 copy-on-write
  ASSERT_THAT(original_storage, testing::Not(is_copy_on_write()));
  // 断言原始的 storage 有简单的数据指针
  ASSERT_TRUE(cow::has_simple_data_ptr(original_storage));

  // 对原始 storage 进行延迟克隆，返回新的 intrusive_ptr<StorageImpl> 对象
  intrusive_ptr<StorageImpl> new_storage =
      cow::lazy_clone_storage(original_storage);
  // 断言新的 storage 对象不为空
  ASSERT_THAT(new_storage.get(), testing::NotNull());

  // 原始 storage 被原地修改，现在包含一个 copy-on-write 上下文
  ASSERT_THAT(original_storage, is_copy_on_write());

  // 结果是一个不同的 storage 实现
  ASSERT_THAT(&*new_storage, testing::Ne(&original_storage));
  // 但它也是 copy-on-write
  ASSERT_THAT(*new_storage, is_copy_on_write());
  // 它们共享相同的数据！
  ASSERT_THAT(new_storage->data(), testing::Eq(original_storage.data()));
}

struct MyDeleterContext {
  MyDeleterContext(void* bytes) : bytes(bytes) {}

  ~MyDeleterContext() {
    delete[] static_cast<std::byte*>(bytes);
  }

  void* bytes;
};

void my_deleter(void* ctx) {
  delete static_cast<MyDeleterContext*>(ctx);
}

TEST(lazy_clone_storage_test, different_context) {
  // 创建一个新的数据块，并用 MyDeleterContext 包装它作为删除器上下文
  void* bytes = new std::byte[5];
  StorageImpl storage(
      {},
      /*size_bytes=*/5,
      at::DataPtr(
          /*data=*/bytes,
          /*ctx=*/new MyDeleterContext(bytes),
          /*ctx_deleter=*/my_deleter,
          /*device=*/Device(Device::Type::CPU)),
      /*allocator=*/nullptr,
      /*resizable=*/false);

  // 我们无法处理任意的上下文
  ASSERT_THAT(cow::lazy_clone_storage(storage), testing::IsNull());
}

TEST(lazy_clone_storage_test, already_copy_on_write) {
  // 创建一个已经是 copy-on-write 的 storage
  std::unique_ptr<void, DeleterFnPtr> data(
      new std::byte[5],
      +[](void* bytes) { delete[] static_cast<std::byte*>(bytes); });
  void* data_ptr = data.get();
  StorageImpl original_storage(
      {},
      /*size_bytes=*/5,
      at::DataPtr(
          /*data=*/data_ptr,
          /*ctx=*/new cow::COWDeleterContext(std::move(data)),
          cow::cow_deleter,
          Device(Device::Type::CPU)),
      /*allocator=*/nullptr,
      /*resizable=*/false);

  // 断言原始 storage 是 copy-on-write
  ASSERT_THAT(original_storage, is_copy_on_write());

  // 对原始 storage 进行延迟克隆，返回新的 intrusive_ptr<StorageImpl> 对象
  intrusive_ptr<StorageImpl> new_storage =
      cow::lazy_clone_storage(original_storage);
  // 断言新的 storage 对象不为空
  ASSERT_THAT(new_storage.get(), testing::NotNull());

  // 结果是一个不同的 storage
  ASSERT_THAT(&*new_storage, testing::Ne(&original_storage));
  // 但它也是 copy-on-write
  ASSERT_THAT(*new_storage, is_copy_on_write());
  // 它们共享相同的数据！
  ASSERT_THAT(new_storage->data(), testing::Eq(original_storage.data()));
}

TEST(materialize_test, not_copy_on_write_context) {
  // 创建一个新的 storage，使用默认的 CPU 分配器和固定大小，不可调整大小
  StorageImpl storage(
      {}, /*size_bytes=*/6, GetCPUAllocator(), /*resizable=*/false);
  // 断言 storage 不是 copy-on-write
  ASSERT_THAT(storage, testing::Not(is_copy_on_write()));

  // 获取原始数据的指针
  void const* original_data = storage.data();

  // 没有需要实现的内容
  ASSERT_THAT(storage.mutable_data(), testing::Eq(original_data));
}
TEST(materialize_test, copy_on_write_single_reference) {
  // 测试用例：copy_on_write_single_reference
  // 当只有单个引用的情况下，复制写入存储可以在实体化时放弃复制写入上下文。

  // 创建一个唯一指针，使用自定义删除器释放 std::byte 数组内存
  std::unique_ptr<void, DeleterFnPtr> data(
      new std::byte[4],
      +[](void* bytes) { delete[] static_cast<std::byte*>(bytes); });

  // 获取数据指针
  void* data_ptr = data.get();

  // 创建 StorageImpl 对象
  StorageImpl storage(
      {},
      /*size_bytes=*/4,
      at::DataPtr(
          /*data=*/data_ptr,
          /*ctx=*/new cow::COWDeleterContext(std::move(data)),
          cow::cow_deleter,
          Device(Device::Type::CPU)),
      /*allocator=*/nullptr,
      /*resizable=*/false);

  // 断言 storage 是否为复制写入
  ASSERT_THAT(storage, is_copy_on_write());

  // 断言 storage 的数据指针与 data_ptr 相同
  ASSERT_THAT(storage.data(), testing::Eq(data_ptr));

  // 获取原始数据指针
  void const* original_data = storage.data();

  // 实体化 storage。由于只有一个引用，因此不会有新的分配。
  ASSERT_THAT(storage.mutable_data(), testing::Eq(original_data));

  // 但此时 storage 不再是复制写入
  ASSERT_THAT(storage, testing::Not(is_copy_on_write()));
}

bool buffers_are_equal(const void* a, const void* b, size_t nbytes) {
  const char* a_ = static_cast<const char*>(a);
  const char* b_ = static_cast<const char*>(b);

  // 比较两个缓冲区是否相等
  for (size_t idx = 0; idx < nbytes; idx++) {
    if (a_[idx] != b_[idx]) {
      return false;
    }
  }
  return true;
}

TEST(materialize_test, copy_on_write) {
  // 测试用例：copy_on_write

  // 创建原始 StorageImpl 对象
  StorageImpl original_storage(
      {}, /*size_bytes=*/6, GetCPUAllocator(), /*resizable=*/false);

  // 将数据 "abcd" 复制到原始 storage 的可变数据中
  std::memcpy(original_storage.mutable_data(), "abcd", 4);

  // 获取原始数据指针
  void const* original_data = original_storage.data();

  // 使用 lazy_clone_storage 创建新的 storage
  auto new_storage = cow::lazy_clone_storage(original_storage);
  ASSERT_THAT(new_storage, testing::NotNull());

  // 获取新 storage 的上下文
  auto context = new_storage->data_ptr().cast_context<cow::COWDeleterContext>(
      cow::cow_deleter);
  ASSERT_THAT(context, testing::NotNull());

  // 断言实体化后新 storage 有新的数据副本
  ASSERT_THAT(new_storage->mutable_data(), testing::Ne(original_data));

  // 断言原始 storage 仍然有原始数据副本
  ASSERT_THAT(original_storage.data(), testing::Eq(original_data));

  // 断言它们的数据是相同的
  ASSERT_TRUE(new_storage->nbytes() == original_storage.nbytes());
  ASSERT_TRUE(buffers_are_equal(
      new_storage->data(), original_storage.data(), new_storage->nbytes()));
}

} // namespace
} // namespace c10::impl
// NOLINTEND(clang-analyzer-cplusplus*)
```