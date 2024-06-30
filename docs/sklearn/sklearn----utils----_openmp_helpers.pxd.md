# `D:\src\scipysrc\scikit-learn\sklearn\utils\_openmp_helpers.pxd`

```
# Helpers to safely access OpenMP routines
#
# no-op implementations are provided for the case where OpenMP is not available.
#
# All calls to OpenMP routines should be cimported from this module.

cdef extern from *:
    """
    #ifdef _OPENMP
        #include <omp.h>
        #define SKLEARN_OPENMP_PARALLELISM_ENABLED 1
    #else
        #define SKLEARN_OPENMP_PARALLELISM_ENABLED 0
        #define omp_lock_t int
        #define omp_init_lock(l) (void)0
        #define omp_destroy_lock(l) (void)0
        #define omp_set_lock(l) (void)0
        #define omp_unset_lock(l) (void)0
        #define omp_get_thread_num() 0
        #define omp_get_max_threads() 1
    #endif
    """
    # 定义一个布尔类型的变量，表示是否启用了 OpenMP 并设置为 1
    bint SKLEARN_OPENMP_PARALLELISM_ENABLED

    # 定义一个结构体类型 omp_lock_t，但不提供具体实现
    ctypedef struct omp_lock_t:
        pass

    # 定义几个 OpenMP 相关函数的原型，包括初始化锁、销毁锁、设置锁、解除锁、获取线程号以及获取最大线程数
    # 这些函数在没有全局锁保护下运行，并且不抛出异常
    void omp_init_lock(omp_lock_t*) noexcept nogil
    void omp_destroy_lock(omp_lock_t*) noexcept nogil
    void omp_set_lock(omp_lock_t*) noexcept nogil
    void omp_unset_lock(omp_lock_t*) noexcept nogil
    int omp_get_thread_num() noexcept nogil
    int omp_get_max_threads() noexcept nogil
```