# `.\pytorch\c10\core\impl\COWDeleter.cpp`

```py
namespace c10::impl {
    // 定义静态函数，用于释放引用计数上下文中的资源
    void cow::cow_deleter(void* ctx) {
        static_cast<cow::COWDeleterContext*>(ctx)->decrement_refcount();
    }

    // COWDeleterContext 构造函数，接收一个具有自定义删除器函数指针的 unique_ptr 作为参数
    cow::COWDeleterContext::COWDeleterContext(
        std::unique_ptr<void, DeleterFnPtr> data)
        : data_(std::move(data)) {
        // 断言：我们从不包装 COWDeleterContext。
        TORCH_INTERNAL_ASSERT(data_.get_deleter() != cow::cow_deleter);
    }

    // 增加引用计数
    auto cow::COWDeleterContext::increment_refcount() -> void {
        auto refcount = ++refcount_;
        // 断言：引用计数应该大于 1
        TORCH_INTERNAL_ASSERT(refcount > 1);
    }

    // 减少引用计数，可能返回不同的引用状态
    auto cow::COWDeleterContext::decrement_refcount()
        -> std::variant<NotLastReference, LastReference> {
        auto refcount = --refcount_;
        // 断言：引用计数应不小于 0，显示当前的引用计数
        TORCH_INTERNAL_ASSERT(refcount >= 0, refcount);
        if (refcount == 0) {
            // 获得独占锁
            std::unique_lock lock(mutex_);
            // 移动数据并释放锁
            auto result = std::move(data_);
            lock.unlock();
            // 删除当前对象
            delete this;
            return {std::move(result)};
        }

        // 返回共享锁
        return std::shared_lock(mutex_);
    }

    // COWDeleterContext 析构函数，断言引用计数为 0
    cow::COWDeleterContext::~COWDeleterContext() {
        TORCH_INTERNAL_ASSERT(refcount_ == 0);
    }

} // namespace c10::impl
```