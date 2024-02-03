# `.\PaddleOCR\ppstructure\table\table_metric\parallel.py`

```
# 导入必要的库
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 定义一个并行版本的 map 函数，带有进度条
def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=0):
    """
        A parallel version of the map function with a progress bar.
        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
                keyword arguments to function
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job.
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    # 我们先串行运行前几次迭代以捕获错误
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a)
                 for a in array[:front_num]]
    else:
        front = []
    # 如果将 n_jobs 设置为 1，只需运行一个列表推导式。这对于基准测试和调试很有用。
    if n_jobs == 1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    # 组装工作进程
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        # 将 array 的元素传递给 function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        # 当任务完成时打印出进度
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    # 从 futures 中获取结果
    # 使用 tqdm 函数遍历 futures 列表，并获取索引和元素
    for i, future in tqdm(enumerate(futures)):
        # 尝试获取 future 的结果并将其添加到 out 列表中
        try:
            out.append(future.result())
        # 如果出现异常，将异常信息添加到 out 列表中
        except Exception as e:
            out.append(e)
    # 返回 front 和 out 列表的组合作为结果
    return front + out
```