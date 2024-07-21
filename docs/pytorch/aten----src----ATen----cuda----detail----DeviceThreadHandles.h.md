# `.\pytorch\aten\src\ATen\cuda\detail\DeviceThreadHandles.h`

```
// Some stateful GPU libraries, such as cuDNN, cuBLAS, use handles to store states.
// These handles are tied to device, and these libraries require/recommend not sharing handles across host threads.
// 
// These libraries suggest using one handle per host thread to avoid synchronization issues, even though creating and
// destroying handles is costly. DataParallel, for instance, generates new threads for each forward pass.
// 
// This file introduces a handle pool mechanism. It dynamically provides handles to threads as they request them. If
// all existing handles in the pool are in use, it creates a new one. When threads finish, they return handles to the
// pool for potential reuse. This approach ensures the handle pool never creates more handles than the peak number of
// active threads, optimizing efficiency with DataParallel.

#pragma once

#include <unordered_map>
#include <vector>
#include <utility>
#include <mutex>
#include <memory>

#include <c10/util/Exception.h>

namespace at::cuda { namespace {

template <typename Handle_t, void Create(Handle_t *), void Destroy(Handle_t)>
struct DeviceThreadHandlePool : public std::enable_shared_from_this<DeviceThreadHandlePool<Handle_t, Create, Destroy>> {

    // Represents a handle with associated creation and destruction behaviors.
    struct Handle {
        Handle_t handle; // The actual handle instance.
        Handle(bool create = false) : handle(nullptr)
        {
            if(create) Create(&handle); // Optionally create the handle upon initialization.
        }

        // Deleting copy constructor to avoid accidental copying of handles.
        Handle(const Handle& rhs) = delete;

        // Move constructor to facilitate efficient handle management within containers.
        Handle(Handle&& rhs) : Handle() { std::swap(handle, rhs.handle); }

        // Assignment operator with pass-by-value parameter for efficient handle swapping.
        Handle& operator=(Handle rhs) { std::swap(handle, rhs.handle); return *this; }

        // Destructor that destroys the handle if it exists, ensuring proper cleanup.
        ~Handle() {
            if(handle) Destroy(handle);
        }
    };

    std::mutex mutex; // Mutex to synchronize access to the handle pool.

    // Map to store vectors of handles associated with each device.
    // Handles are dynamically created as threads request them, but remain in the pool until process termination.
    // The maximum number of handles per device equals the peak number of concurrent threads requesting handles.
    // Threads release handles back into the pool upon termination.
    // 否则，每次生成新线程时都会创建新句柄，导致 Python 模块性能下降，特别是那些频繁生成新线程集合的模块（如 DataParallel，每次前向传播都会创建新线程集合）。
    //
    // 为了避免潜在的死锁，我们明确选择不限制每个设备创建的句柄数量。
    // 危险示例：如果我们将最大句柄数限制为 4，并且有 5 个线程共享一个设备，则只有 4 个能同时向前进展。其余 4 个直到退出前都不会释放其句柄，因此第五个线程无法继续进展。除非它们中的所有 5 个线程在中间点尝试某种同步（即在它们任何一个退出之前）。我们无法预测或强制用户线程不会尝试这种中间同步。
    // 确保安全的唯一方法是避免对句柄数量施加限制。
    std::unordered_map<int, std::vector<Handle>> created_handles;
    std::unordered_map<int, std::vector<Handle_t>> available_handles;

    // PoolWindow 惰性地创建和缓存特定线程正在使用的句柄，因此在常见情况下，句柄访问既不会导致句柄创建，也不会导致互斥锁。
    class PoolWindow
    {
    public:
        PoolWindow(std::shared_ptr<DeviceThreadHandlePool> parent): weak_parent(std::move(parent)) {}
        ~PoolWindow(){ release(); }

        Handle_t reserve(int device)
        {
            // 如果该线程已经为此设备有句柄，则返回该句柄
            if(my_handles.find(device) != my_handles.end())
                return my_handles[device];

            // 否则，如果池中有可用的句柄，则从中获取一个，如果没有，则创建一个新的句柄。
            auto parent = weak_parent.lock();
            TORCH_CHECK(parent, "Cannot create handle during program termination");
            std::lock_guard<std::mutex> guard(parent->mutex);

            if(parent->available_handles[device].size() > 0)
            {
                my_handles[device] = parent->available_handles[device].back();
                parent->available_handles[device].pop_back();
            }
            else
            {
                // 在本地测试中，观察到 emplace_back 有时会经过临时变量，导致调用移动构造函数和析构函数。参见上面 Handle 中的注释。
                parent->created_handles[device].emplace_back(true /*create*/);
                my_handles[device] = parent->created_handles[device].back().handle;
            }

            return my_handles[device];
        }

    private:
        // 存储当前线程拥有的每个设备的句柄
        std::unordered_map<int, Handle_t> my_handles;

        std::weak_ptr<DeviceThreadHandlePool> weak_parent;

        // 在析构函数中调用。将该线程的句柄释放回池中。
    void release() {
        // 如果存在已分配的句柄
        if(my_handles.size() > 0) {
            // 尝试获取弱引用指向的父对象
            auto parent = weak_parent.lock();
            // 如果父对象已经不存在
            if (!parent) {
                // 如果该线程在atexit处理程序完成后退出，cuda上下文本身可能无效，因此我们必须泄漏句柄。
                return;
            }

            // 锁定父对象的互斥量，确保线程安全地访问
            std::lock_guard<std::mutex> guard(parent->mutex);
            // 将本对象的句柄放回父对象的可用句柄池中
            for(auto d_h : my_handles)
                parent->available_handles[d_h.first].push_back(d_h.second);
        }
    }
    };

    // 警告:
    // 如果要更改此函数，请注意此函数将由多个线程调用，并且没有互斥锁保护对此函数的调用，
    // 因此请确保您的实现是线程安全的。
    PoolWindow *newPoolWindow() {
        // 返回的指针将由线程局部变量拥有，以便不同线程不共享同一个PoolWindow实例。
        return new PoolWindow(this->shared_from_this());
    }
};

}}  // namespace at::cuda::detail::<anonymous>
```