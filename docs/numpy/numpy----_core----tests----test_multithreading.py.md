# `.\numpy\numpy\_core\tests\test_multithreading.py`

```
import concurrent.futures  # 导入并发执行的 futures 模块
import threading  # 导入线程模块

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 测试框架

from numpy.testing import IS_WASM  # 从 NumPy 测试模块中导入 IS_WASM 标志

if IS_WASM:
    pytest.skip(allow_module_level=True, reason="no threading support in wasm")
    # 如果在 wasm 环境下，跳过测试并给出原因：wasm 不支持多线程

def run_threaded(func, iters, pass_count=False):
    # 使用线程池执行多线程任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as tpe:
        if pass_count:
            # 如果传递计数，提交带参数的任务到线程池
            futures = [tpe.submit(func, i) for i in range(iters)]
        else:
            # 否则提交无参数的任务到线程池
            futures = [tpe.submit(func) for _ in range(iters)]
        for f in futures:
            f.result()  # 等待每个任务执行完成并获取结果

def test_parallel_randomstate_creation():
    # 测试并行创建 RandomState 实例的线程安全性
    def func(seed):
        np.random.RandomState(seed)  # 创建一个 RandomState 实例

    run_threaded(func, 500, pass_count=True)  # 并发执行 500 次随机实例创建任务

def test_parallel_ufunc_execution():
    # 测试并行执行 ufunc 操作的线程安全性
    def func():
        arr = np.random.random((25,))  # 创建一个随机数组
        np.isnan(arr)  # 执行一个 ufunc 操作

    run_threaded(func, 500)  # 并发执行 500 次 ufunc 操作任务

    # 查看 GitHub 问题 gh-26690

    NUM_THREADS = 50  # 定义线程数量

    b = threading.Barrier(NUM_THREADS)  # 创建线程屏障对象

    a = np.ones(1000)  # 创建长度为 1000 的全为 1 的数组

    def f():
        b.wait()  # 等待所有线程都到达屏障
        return a.sum()  # 返回数组所有元素的和

    threads = [threading.Thread(target=f) for _ in range(NUM_THREADS)]  # 创建线程列表

    [t.start() for t in threads]  # 启动所有线程
    [t.join() for t in threads]  # 等待所有线程执行完成

def test_temp_elision_thread_safety():
    # 测试临时缓存操作的线程安全性

    amid = np.ones(50000)  # 创建长度为 50000 的全为 1 的数组 amid
    bmid = np.ones(50000)  # 创建长度为 50000 的全为 1 的数组 bmid
    alarge = np.ones(1000000)  # 创建长度为 1000000 的全为 1 的数组 alarge
    blarge = np.ones(1000000)  # 创建长度为 1000000 的全为 1 的数组 blarge

    def func(count):
        if count % 4 == 0:
            (amid * 2) + bmid  # 对 amid 和 bmid 执行乘法和加法
        elif count % 4 == 1:
            (amid + bmid) - 2  # 对 amid 和 bmid 执行加法和减法
        elif count % 4 == 2:
            (alarge * 2) + blarge  # 对 alarge 和 blarge 执行乘法和加法
        else:
            (alarge + blarge) - 2  # 对 alarge 和 blarge 执行加法和减法

    run_threaded(func, 100, pass_count=True)  # 并发执行 100 次临时缓存操作任务
```