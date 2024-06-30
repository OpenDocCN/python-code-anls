# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\fast_matrix_market\include\fast_matrix_market\thirdparty\task_thread_pool.hpp`

```
// SPDX-License-Identifier: BSD-2-Clause OR MIT OR BSL-1.0
// 如果 AL_TASK_THREAD_POOL_HPP 宏没有定义，则定义 AL_TASK_THREAD_POOL_HPP 宏
#ifndef AL_TASK_THREAD_POOL_HPP
// 定义 AL_TASK_THREAD_POOL_HPP 版本号的主、次、修订版本号
#define AL_TASK_THREAD_POOL_HPP

// 版本宏定义
// 主版本号为 1
#define TASK_THREAD_POOL_VERSION_MAJOR 1
// 次版本号为 0
#define TASK_THREAD_POOL_VERSION_MINOR 0
// 修订版本号为 7
#define TASK_THREAD_POOL_VERSION_PATCH 7

// 包含标准库头文件
#include <condition_variable> // 条件变量
#include <functional>         // 函数对象和函数包装器
#include <future>             // 异步执行（future-promise机制）
#include <mutex>              // 互斥量
#include <queue>              // 队列容器
#include <thread>             // 线程管理
#include <type_traits>        // 类型特性判断

// MSVC 默认情况下未正确设置 __cplusplus 宏，因此需要从 _MSVC_LANG 中读取
// 参考：https://devblogs.microsoft.com/cppblog/msvc-now-correctly-reports-__cplusplus/
#if __cplusplus >= 201703L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
#define TTP_CXX17 1 // 如果支持 C++17 及以上版本，定义 TTP_CXX17 宏为 1
#else
#define TTP_CXX17 0 // 否则定义为 0
#endif

#if TTP_CXX17
#define NODISCARD [[nodiscard]] // 如果支持 C++17 及以上版本，定义 NODISCARD 为 [[nodiscard]]
#else
#define NODISCARD // 否则不定义 NODISCARD
#endif

namespace task_thread_pool {

#if !TTP_CXX17
    /**
     * A reimplementation of std::decay_t, which is only available since C++14.
     * 重新实现 std::decay_t，因为它只在 C++14 及以上版本中可用。
     */
    template <class T>
    using decay_t = typename std::decay<T>::type; // 定义 decay_t 类型模板别名，用于获取类型 T 的衰减类型
#endif

    /**
     * A fast and lightweight thread pool that uses C++11 threads.
     * 一个快速轻量级的线程池，使用 C++11 的线程。
     */
    class task_thread_pool {
#if TTP_CXX17
            typename R = std::invoke_result_t<std::decay_t<F>, std::decay_t<A>...>
#else
            typename R = typename std::result_of<decay_t<F>(decay_t<A>...)>::type
        #endif
        >
        // 定义 submit 函数模板，返回一个 future 对象，用于异步获取任务结果
        NODISCARD std::future<R> submit(F&& func, A&&... args) {
            // 创建一个 shared_ptr 智能指针，指向一个 packaged_task 对象，封装了 func(args...)
            std::shared_ptr<std::packaged_task<R()>> ptask = std::make_shared<std::packaged_task<R()>>(std::bind(std::forward<F>(func), std::forward<A>(args)...));
            // 提交一个分离的任务，该任务执行 ptask 指向的函数对象
            submit_detach([ptask] { (*ptask)(); });
            // 返回任务的 future 对象，以便异步获取任务结果
            return ptask->get_future();
        }

        /**
         * Submit a zero-argument Callable for the pool to execute.
         *
         * @param func The Callable to execute. Can be a function, a lambda, std::packaged_task, std::function, etc.
         */
        // 提交一个没有参数的 Callable 给线程池执行
        template <typename F>
        void submit_detach(F&& func) {
            // 锁住任务队列的互斥量
            const std::lock_guard<std::mutex> tasks_lock(task_mutex);
            // 将 func 移动或复制到任务队列中
            tasks.emplace(std::forward<F>(func));
            // 通知一个等待中的线程开始执行任务
            task_cv.notify_one();
        }

        /**
         * Submit a Callable with arguments for the pool to execute.
         *
         * @param func The Callable to execute. Can be a function, a lambda, std::packaged_task, std::function, etc.
         */
        // 提交一个带参数的 Callable 给线程池执行
        template <typename F, typename... A>
        void submit_detach(F&& func, A&&... args) {
            // 锁住任务队列的互斥量
            const std::lock_guard<std::mutex> tasks_lock(task_mutex);
            // 将一个绑定了 func 和 args 的任务添加到任务队列中
            tasks.emplace(std::bind(std::forward<F>(func), std::forward<A>(args)...));
            // 通知一个等待中的线程开始执行任务
            task_cv.notify_one();
        }

        /**
         * Block until the task queue is empty. Some tasks may be in-progress when this method returns.
         */
        // 阻塞当前线程，直到任务队列为空。一些任务可能在此方法返回时仍在执行中
        void wait_for_queued_tasks() {
            // 锁住任务队列的互斥量
            std::unique_lock<std::mutex> tasks_lock(task_mutex);
            // 设置一个标志，通知任务完成条件变量继续等待
            notify_task_finish = true;
            // 等待，直到任务队列为空
            task_finished_cv.wait(tasks_lock, [&] { return tasks.empty(); });
            // 重置任务完成通知标志
            notify_task_finish = false;
        }

        /**
         * Block until all tasks have finished.
         */
        // 阻塞当前线程，直到所有任务都完成
        void wait_for_tasks() {
            // 锁住任务队列的互斥量
            std::unique_lock<std::mutex> tasks_lock(task_mutex);
            // 设置一个标志，通知任务完成条件变量继续等待
            notify_task_finish = true;
            // 等待，直到任务队列为空且没有正在执行的任务
            task_finished_cv.wait(tasks_lock, [&] { return tasks.empty() && num_inflight_tasks == 0; });
            // 重置任务完成通知标志
            notify_task_finish = false;
        }

    };
}

// 清理
#undef NODISCARD
#undef TTP_CXX17

#endif
```