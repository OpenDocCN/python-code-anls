# `.\pytorch\torch\utils\data\_utils\worker.py`

```py
# mypy: allow-untyped-defs
# 定义了 _BaseDataLoaderIter 工作线程使用的方法。
# 这些方法需要在全局范围内定义，因为 Py2 不支持序列化静态方法。

import os  # 导入操作系统相关功能的模块
import queue  # 导入队列模块，用于多线程间的通信
import random  # 导入随机数生成模块
from dataclasses import dataclass  # 导入用于创建数据类的装饰器
from typing import Optional, TYPE_CHECKING, Union  # 导入类型提示相关模块

import torch  # 导入 PyTorch 模块
from torch._utils import ExceptionWrapper  # 导入异常包装器

from . import HAS_NUMPY, IS_WINDOWS, MP_STATUS_CHECK_INTERVAL, signal_handling  # 导入自定义模块

if TYPE_CHECKING:
    from torch.utils.data import Dataset  # 如果是类型检查模式，导入数据集类型提示

if IS_WINDOWS:
    import ctypes  # 导入 ctypes 模块，用于访问 Windows API
    from ctypes.wintypes import BOOL, DWORD, HANDLE  # 导入 Windows API 所需类型

    # 在 Windows 平台上，当管理进程结束后，工作进程的父进程 ID 保持不变，
    # 唯一的检查方式是让工作进程拥有管理进程的进程句柄，并检查进程状态是否发生变化。
    class ManagerWatchdog:
        def __init__(self):
            self.manager_pid = os.getppid()  # 获取父进程 ID

            # 使用 ctypes 调用 Windows kernel32.dll 中的函数
            self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
            self.kernel32.OpenProcess.argtypes = (DWORD, BOOL, DWORD)
            self.kernel32.OpenProcess.restype = HANDLE
            self.kernel32.WaitForSingleObject.argtypes = (HANDLE, DWORD)
            self.kernel32.WaitForSingleObject.restype = DWORD

            # 从 https://msdn.microsoft.com/en-us/library/ms684880.aspx 获取的值
            SYNCHRONIZE = 0x00100000
            # 打开管理进程的进程句柄
            self.manager_handle = self.kernel32.OpenProcess(
                SYNCHRONIZE, 0, self.manager_pid
            )

            if not self.manager_handle:
                raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]

            self.manager_dead = False  # 初始化管理进程是否已结束的标志

        def is_alive(self):
            if not self.manager_dead:
                # 从 https://msdn.microsoft.com/en-us/library/windows/desktop/ms687032.aspx 获取的值
                # 检查管理进程的状态是否为 signaled 状态
                self.manager_dead = (
                    self.kernel32.WaitForSingleObject(self.manager_handle, 0) == 0
                )
            return not self.manager_dead  # 返回管理进程是否仍然存活的布尔值

else:
    # 在非 Windows 平台上，简单地监视父进程 ID 是否变化来判断管理进程是否存活
    class ManagerWatchdog:
        def __init__(self):
            self.manager_pid = os.getppid()  # 获取父进程 ID
            self.manager_dead = False  # 初始化管理进程是否已结束的标志

        def is_alive(self):
            if not self.manager_dead:
                # 检查父进程 ID 是否发生变化
                self.manager_dead = os.getppid() != self.manager_pid
            return not self.manager_dead  # 返回管理进程是否仍然存活的布尔值


# 定义一个全局变量，用于存储 WorkerInfo 对象，初始值为 None
_worker_info: Optional["WorkerInfo"] = None


@dataclass
class WorkerInfo:
    id: int
    num_workers: int
    seed: int
    dataset: "Dataset"
    __initialized = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__keys = tuple(kwargs.keys())  # 存储传入参数的键名元组
        self.__initialized = True  # 设置初始化完成标志
    # 设置对象的属性。如果对象已经初始化完成，则抛出运行时错误
    def __setattr__(self, key, val):
        # 检查对象是否已经初始化，如果是则抛出运行时错误
        if self.__initialized:
            raise RuntimeError(
                f"Cannot assign attributes to {self.__class__.__name__} objects"
            )
        # 调用父类的 __setattr__ 方法设置属性
        return super().__setattr__(key, val)

    # 返回对象的字符串表示形式，包括所有指定的属性
    def __repr__(self):
        # 创建一个空列表用于存储属性和对应的值
        items = []
        # 遍历对象的属性列表
        for k in self.__keys:
            # 将每个属性及其值添加到列表中
            items.append(f"{k}={getattr(self, k)}")
        # 返回对象的类名和所有属性及其值的字符串表示形式
        return f"{self.__class__.__name__}({', '.join(items)})"
# 返回当前 DataLoader 迭代器的 worker 进程信息，可能为 None
def get_worker_info() -> Optional[WorkerInfo]:
    r"""Returns the information about the current
    :class:`~torch.utils.data.DataLoader` iterator worker process.
    
    When called in a worker, this returns an object guaranteed to have the
    following attributes:
    
    * :attr:`id`: the current worker id.
    * :attr:`num_workers`: the total number of workers.
    * :attr:`seed`: the random seed set for the current worker. This value is
      determined by main process RNG and the worker id. See
      :class:`~torch.utils.data.DataLoader`'s documentation for more details.
    * :attr:`dataset`: the copy of the dataset object in **this** process. Note
      that this will be a different object in a different process than the one
      in the main process.
    
    When called in the main process, this returns ``None``.
    
    .. note::
       When used in a :attr:`worker_init_fn` passed over to
       :class:`~torch.utils.data.DataLoader`, this method can be useful to
       set up each worker process differently, for instance, using ``worker_id``
       to configure the ``dataset`` object to only read a specific fraction of a
       sharded dataset, or use ``seed`` to seed other libraries used in dataset
       code.
    """
    return _worker_info


# Dummy class used to signal the end of an IterableDataset
@dataclass(frozen=True)
class _IterableDatasetStopIteration:
    worker_id: int


# Dummy class used to resume the fetching when worker reuse is enabled
@dataclass(frozen=True)
class _ResumeIteration:
    seed: Optional[int] = None


# The function `_generate_state` is adapted from `numpy.random.SeedSequence`
# from https://github.com/numpy/numpy/blob/main/numpy/random/bit_generator.pyx
# It's MIT licensed, here is the copyright:

# Copyright (c) 2015 Melissa E. O'Neill
# Copyright (c) 2019 NumPy Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# This function generates an array of int32 as the seed for
# 根据 `numpy.random` 的种子和算法，为了避免与 `random` 模块的种子状态冲突，
# 实现类似 `torch.random` 的 `SeedSequence` 对象。

def _generate_state(base_seed, worker_id):
    INIT_A = 0x43B0D7E5
    MULT_A = 0x931E8875
    INIT_B = 0x8B51F9DD
    MULT_B = 0x58F38DED
    MIX_MULT_L = 0xCA01F9DD
    MIX_MULT_R = 0x4973F715
    XSHIFT = 4 * 8 // 2
    MASK32 = 0xFFFFFFFF

    # 初始化熵池和哈希常数
    entropy = [worker_id, base_seed & MASK32, base_seed >> 32, 0]
    pool = [0] * 4
    hash_const_A = INIT_A

    # 定义哈希函数，用于混合状态
    def hash(value):
        nonlocal hash_const_A
        value = (value ^ hash_const_A) & MASK32
        hash_const_A = (hash_const_A * MULT_A) & MASK32
        value = (value * hash_const_A) & MASK32
        value = (value ^ (value >> XSHIFT)) & MASK32
        return value

    # 定义混合函数，用于混合状态池中的值
    def mix(x, y):
        result_x = (MIX_MULT_L * x) & MASK32
        result_y = (MIX_MULT_R * y) & MASK32
        result = (result_x - result_y) & MASK32
        result = (result ^ (result >> XSHIFT)) & MASK32
        return result

    # 将熵值加入到熵池中
    for i in range(len(pool)):
        pool[i] = hash(entropy[i])

    # 混合所有位，以使后面的位影响前面的位
    for i_src in range(len(pool)):
        for i_dst in range(len(pool)):
            if i_src != i_dst:
                pool[i_dst] = mix(pool[i_dst], hash(pool[i_src]))

    # 使用第二个哈希常数生成状态
    hash_const_B = INIT_B
    state = []
    for i_dst in range(4):
        data_val = pool[i_dst]
        data_val = (data_val ^ hash_const_B) & MASK32
        hash_const_B = (hash_const_B * MULT_B) & MASK32
        data_val = (data_val * hash_const_B) & MASK32
        data_val = (data_val ^ (data_val >> XSHIFT)) & MASK32
        state.append(data_val)
    return state


def _worker_loop(
    dataset_kind,
    dataset,
    index_queue,
    data_queue,
    done_event,
    auto_collation,
    collate_fn,
    drop_last,
    base_seed,
    init_fn,
    worker_id,
    num_workers,
    persistent_workers,
    shared_seed,
):
    # 详见 [ Data Loader Multiprocessing Shutdown Logic ] 的注释，了解此函数的逻辑。

    # 处理键盘中断异常，主进程将会主动引发 KeyboardInterrupt 异常。
    except KeyboardInterrupt:
        pass

    # 如果完成事件已设置，取消数据队列的线程加入，关闭数据队列。
    if done_event.is_set():
        data_queue.cancel_join_thread()
        data_queue.close()
```