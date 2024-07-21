# `.\pytorch\caffe2\utils\threadpool\pthreadpool.h`

```
// pthreadpool header from https://github.com/Maratyszcza/pthreadpool
// for NNPACK

// 防止重复包含该头文件
#ifndef CAFFE2_UTILS_PTHREADPOOL_H_
#define CAFFE2_UTILS_PTHREADPOOL_H_

// 包含通用线程池相关的头文件
#include "ThreadPoolCommon.h"

// 包含标准库头文件，用于 size_t 和 uint32_t 的定义
#include <stddef.h> // for size_t
#include <stdint.h> // for uint32_t

// 如果定义了 USE_PTHREADPOOL 宏，则进入条件编译
#if defined(USE_PTHREADPOOL)

// 这是一个 hack。
// 主要介绍如下：
// 1. NNPACK 可以编译为使用内部旧的线程池实现，因为 C2 的大部分依赖于此。
// 2. 然后，如果我们想在 PyTorch 中使用 NNPACK，PyTorch 使用新的 pthreadpool，
//    那么我们将新的 pthreadpool 指针提供给 NNPACK 是不起作用的，如果 NNPACK 是
//    用内部旧线程池编译的。因此，此保护与 pthreadpool_impl.cc 中的更改允许我们覆盖
//    该行为。它使我们能够使用 PyTorch 中的 NNPACK，使用 `caffe2::pthreadpool_()`
namespace caffe2 {
// 用于将使用新线程池转换的类
class WithCastToNewThreadPool {
  public:
    explicit WithCastToNewThreadPool(bool use_new_threadpool);
    ~WithCastToNewThreadPool();
  private:
    bool use_new_threadpool_;
};
}
#endif

// 定义了一个 legacy_pthreadpool_t 类型，表示指向 pthreadpool 结构的指针
typedef struct pthreadpool* legacy_pthreadpool_t;

// 定义了几种函数指针类型，用于描述线程池中函数的形式
typedef void (*legacy_pthreadpool_function_1d_t)(void*, size_t);
typedef void (*legacy_pthreadpool_function_1d_tiled_t)(void*, size_t, size_t);
typedef void (*legacy_pthreadpool_function_2d_t)(void*, size_t, size_t);
typedef void (*legacy_pthreadpool_function_2d_tiled_t)(void*, size_t, size_t, size_t, size_t);
typedef void (*legacy_pthreadpool_function_3d_tiled_t)(
    void*,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t);
typedef void (*legacy_pthreadpool_function_4d_tiled_t)(
    void*,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t);

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 创建一个具有指定线程数的线程池。
 *
 * @param[in]  threads_count  线程池中的线程数。
 *    当 threads_count 为 0 时有特殊解释：它为系统中可用的每个处理器核心创建一个线程。
 *
 * @returns  指向不透明线程池对象的指针。
 *    如果出错，函数返回 NULL 并相应设置 errno。
 */
legacy_pthreadpool_t legacy_pthreadpool_create(size_t threads_count);

/**
 * 查询线程池中的线程数。
 *
 * @param[in]  threadpool  要查询的线程池。
 *
 * @returns  线程池中的线程数。
 */
size_t legacy_pthreadpool_get_threads_count(legacy_pthreadpool_t threadpool);
/**
 * Processes items in parallel using threads from a thread pool.
 *
 * When the call returns, all items have been processed and the thread pool is
 * ready for a new task.
 *
 * @note If multiple threads call this function with the same thread pool, the
 *    calls are serialized.
 *
 * @param[in]  threadpool  The thread pool to use for parallelisation.
 * @param[in]  function    The function to call for each item.
 * @param[in]  argument    The first argument passed to the @a function.
 * @param[in]  range       The number of items to process. The @a function
 *                         will be called once for each item.
 */
void legacy_pthreadpool_compute_1d(
    legacy_pthreadpool_t threadpool,
    legacy_pthreadpool_function_1d_t function,
    void* argument,
    size_t range);

/**
 * Parallelizes the processing of items in 1D using threads from a thread pool.
 *
 * Allows additional flags to control behavior.
 *
 * @param[in]  threadpool  The thread pool to use for parallelisation.
 * @param[in]  function    The function to call for each item.
 * @param[in]  argument    The first argument passed to the @a function.
 * @param[in]  range       The number of items to process. The @a function
 *                         will be called once for each item.
 * @param[in]  flags       Additional flags to modify behavior (not used here).
 */
void legacy_pthreadpool_parallelize_1d(
    legacy_pthreadpool_t threadpool,
    legacy_pthreadpool_function_1d_t function,
    void* argument,
    size_t range,
    uint32_t flags);

/**
 * Processes items in 1D using tiles and threads from a thread pool.
 *
 * Allows defining a tile size to split the range of items.
 *
 * @param[in]  threadpool  The thread pool to use for parallelisation.
 * @param[in]  function    The function to call for each tile of items.
 * @param[in]  argument    The first argument passed to the @a function.
 * @param[in]  range       The number of items to process in total.
 * @param[in]  tile        The size of each tile to process as a subset of @a range.
 */
void legacy_pthreadpool_compute_1d_tiled(
    legacy_pthreadpool_t threadpool,
    legacy_pthreadpool_function_1d_tiled_t function,
    void* argument,
    size_t range,
    size_t tile);

/**
 * Processes items in 2D using threads from a thread pool.
 *
 * @param[in]  threadpool  The thread pool to use for parallelisation.
 * @param[in]  function    The function to call for each item in 2D grid.
 * @param[in]  argument    The first argument passed to the @a function.
 * @param[in]  range_i     The number of items along the first dimension.
 * @param[in]  range_j     The number of items along the second dimension.
 */
void legacy_pthreadpool_compute_2d(
    legacy_pthreadpool_t threadpool,
    legacy_pthreadpool_function_2d_t function,
    void* argument,
    size_t range_i,
    size_t range_j);

/**
 * Processes items in 2D using tiles and threads from a thread pool.
 *
 * Allows defining tile sizes for both dimensions.
 *
 * @param[in]  threadpool  The thread pool to use for parallelisation.
 * @param[in]  function    The function to call for each tile of items in 2D grid.
 * @param[in]  argument    The first argument passed to the @a function.
 * @param[in]  range_i     The number of items along the first dimension.
 * @param[in]  range_j     The number of items along the second dimension.
 * @param[in]  tile_i      The size of each tile along the first dimension.
 * @param[in]  tile_j      The size of each tile along the second dimension.
 */
void legacy_pthreadpool_compute_2d_tiled(
    legacy_pthreadpool_t threadpool,
    legacy_pthreadpool_function_2d_tiled_t function,
    void* argument,
    size_t range_i,
    size_t range_j,
    size_t tile_i,
    size_t tile_j);

/**
 * Processes items in 3D using tiles and threads from a thread pool.
 *
 * Allows defining tile sizes for all three dimensions.
 *
 * @param[in]  threadpool  The thread pool to use for parallelisation.
 * @param[in]  function    The function to call for each tile of items in 3D grid.
 * @param[in]  argument    The first argument passed to the @a function.
 * @param[in]  range_i     The number of items along the first dimension.
 * @param[in]  range_j     The number of items along the second dimension.
 * @param[in]  range_k     The number of items along the third dimension.
 * @param[in]  tile_i      The size of each tile along the first dimension.
 * @param[in]  tile_j      The size of each tile along the second dimension.
 * @param[in]  tile_k      The size of each tile along the third dimension.
 */
void legacy_pthreadpool_compute_3d_tiled(
    legacy_pthreadpool_t threadpool,
    legacy_pthreadpool_function_3d_tiled_t function,
    void* argument,
    size_t range_i,
    size_t range_j,
    size_t range_k,
    size_t tile_i,
    size_t tile_j,
    size_t tile_k);

/**
 * Processes items in 4D using tiles and threads from a thread pool.
 *
 * Allows defining tile sizes for all four dimensions.
 *
 * @param[in]  threadpool  The thread pool to use for parallelisation.
 * @param[in]  function    The function to call for each tile of items in 4D grid.
 * @param[in]  argument    The first argument passed to the @a function.
 * @param[in]  range_i     The number of items along the first dimension.
 * @param[in]  range_j     The number of items along the second dimension.
 * @param[in]  range_k     The number of items along the third dimension.
 * @param[in]  range_l     The number of items along the fourth dimension.
 * @param[in]  tile_i      The size of each tile along the first dimension.
 * @param[in]  tile_j      The size of each tile along the second dimension.
 * @param[in]  tile_k      The size of each tile along the third dimension.
 * @param[in]  tile_l      The size of each tile along the fourth dimension.
 */
void legacy_pthreadpool_compute_4d_tiled(
    legacy_pthreadpool_t threadpool,
    legacy_pthreadpool_function_4d_tiled_t function,
    void* argument,
    size_t range_i,
    size_t range_j,
    size_t range_k,
    size_t range_l,
    size_t tile_i,
    size_t tile_j,
    size_t tile_k,
    size_t tile_l);

/**
 * Terminates threads in the thread pool and releases associated resources.
 *
 * @warning  Accessing the thread pool after a call to this function constitutes
 *    undefined behaviour and may cause data corruption.
 *
 * @param[in,out]  threadpool  The thread pool to destroy.
 */
void legacy_pthreadpool_destroy(legacy_pthreadpool_t threadpool);

/**
 * Defines macros to alias pthreadpool types and function types for internal use.
 *
 * This set of macros aliases pthreadpool types and function types to legacy names
 * for compatibility with older code or specific implementations.
 */
#define pthreadpool_t legacy_pthreadpool_t
#define pthreadpool_function_1d_t legacy_pthreadpool_function_1d_t
#define pthreadpool_function_1d_tiled_t legacy_pthreadpool_function_1d_tiled_t
#define pthreadpool_function_2d_t legacy_pthreadpool_function_2d_t
#define pthreadpool_function_2d_tiled_t legacy_pthreadpool_function_2d_tiled_t
#define pthreadpool_function_3d_tiled_t legacy_pthreadpool_function_3d_tiled_t
#ifdef USE_INTERNAL_PTHREADPOOL_IMPL
// 如果定义了 USE_INTERNAL_PTHREADPOOL_IMPL 宏，则定义一系列 pthreadpool 相关的宏

#define pthreadpool_function_4d_tiled_t legacy_pthreadpool_function_4d_tiled_t
#define pthreadpool_create legacy_pthreadpool_create
#define pthreadpool_destroy legacy_pthreadpool_destroy
#define pthreadpool_get_threads_count legacy_pthreadpool_get_threads_count
#define pthreadpool_compute_1d legacy_pthreadpool_compute_1d
#define pthreadpool_parallelize_1d legacy_pthreadpool_parallelize_1d
#define pthreadpool_compute_1d_tiled legacy_pthreadpool_compute_1d_tiled
#define pthreadpool_compute_2d legacy_pthreadpool_compute_2d
#define pthreadpool_compute_2d_tiled legacy_pthreadpool_compute_2d_tiled
#define pthreadpool_compute_3d_tiled legacy_pthreadpool_compute_3d_tiled
#define pthreadpool_compute_4d_tiled legacy_pthreadpool_compute_4d_tiled
#endif // USE_INTERNAL_PTHREADPOOL_IMPL

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif // CAFFE2_UTILS_PTHREADPOOL_H_
```