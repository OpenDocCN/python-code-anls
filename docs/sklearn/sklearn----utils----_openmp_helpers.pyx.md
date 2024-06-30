# `D:\src\scipysrc\scikit-learn\sklearn\utils\_openmp_helpers.pyx`

```
import os
from joblib import cpu_count

# 缓存不同配置下的 CPU 核心数，避免重复计算
_CPU_COUNTS = {}


def _openmp_parallelism_enabled():
    """确定 scikit-learn 是否已使用 OpenMP 构建

    允许在运行时检索在编译时收集的信息。
    """
    # SKLEARN_OPENMP_PARALLELISM_ENABLED 在编译时解析，并在 _openmp_helpers.pxd 中作为布尔值定义。此函数将其暴露给 Python。
    return SKLEARN_OPENMP_PARALLELISM_ENABLED


cpdef _openmp_effective_n_threads(n_threads=None, only_physical_cores=True):
    """确定用于 OpenMP 调用的有效线程数

    - 对于 ``n_threads = None``,
      - 如果设置了 ``OMP_NUM_THREADS`` 环境变量，则返回 ``openmp.omp_get_max_threads()``
      - 否则，返回 ``openmp.omp_get_max_threads()`` 和 CPU 数量的最小值，考虑到 cgroups 的配额。cgroups 的配额通常可以由 Docker 等工具设置。
      ``omp_get_max_threads`` 的结果可以受到环境变量 ``OMP_NUM_THREADS`` 或在运行时通过 ``omp_set_num_threads`` 的影响。

    - 对于 ``n_threads > 0``, 返回此数作为并行 OpenMP 调用的最大线程数。

    - 对于 ``n_threads < 0``, 返回最大线程数减去 ``|n_threads + 1|``。特别地，``n_threads = -1`` 将使用机器上可用的所有核心作为线程数。

    - 对于 ``n_threads = 0``, 抛出 ValueError 异常。

    通过传递 `only_physical_cores=False` 标志，可以使用额外的线程进行 SMT/HyperThreading 逻辑核心。已经经验性地观察到，在某些情况下使用尽可能多的 SMT 核心可以略微提高性能，但在其他情况下可能会严重降低性能。因此，建议仅在已经进行了经验性研究以评估 SMT 对情况影响的情况下（使用各种输入数据形状，特别是小数据形状），才使用 `only_physical_cores=True`。

    如果 scikit-learn 在没有 OpenMP 支持的情况下构建，则始终返回 1。
    """
    if n_threads == 0:
        raise ValueError("n_threads = 0 是无效的")

    if not SKLEARN_OPENMP_PARALLELISM_ENABLED:
        # 在编译时禁用了 OpenMP => 顺序模式
        return 1

    if os.getenv("OMP_NUM_THREADS"):
        # 回退到用户提供的线程数，使其可能超过 CPU 数量。
        max_n_threads = omp_get_max_threads()
    else:
        try:
            n_cpus = _CPU_COUNTS[only_physical_cores]
        except KeyError:
            n_cpus = cpu_count(only_physical_cores=only_physical_cores)
            _CPU_COUNTS[only_physical_cores] = n_cpus
        max_n_threads = min(omp_get_max_threads(), n_cpus)
    # 如果 n_threads 为 None，则返回最大线程数 max_n_threads
    if n_threads is None:
        return max_n_threads
    
    # 如果 n_threads 小于 0，则返回 max(1, max_n_threads + n_threads + 1)，
    # 保证返回值不小于 1，并且保持原有最大线程数的基础上增加 n_threads + 1
    elif n_threads < 0:
        return max(1, max_n_threads + n_threads + 1)
    
    # 如果 n_threads 大于等于 0，则直接返回 n_threads
    return n_threads
```