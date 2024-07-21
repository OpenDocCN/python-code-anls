# `.\pytorch\test\test_multiprocessing.py`

```
# Owner(s): ["module: multiprocessing"]

import contextlib  # 导入上下文管理工具库
import copy  # 导入对象复制函数库
import gc  # 导入垃圾回收模块
import os  # 导入操作系统功能库
import sys  # 导入系统相关功能库
import time  # 导入时间处理库
import unittest  # 导入单元测试框架
from sys import platform  # 导入系统平台信息

import torch  # 导入PyTorch深度学习库
import torch.cuda  # 导入PyTorch CUDA相关功能
import torch.multiprocessing as mp  # 导入PyTorch多进程功能
import torch.utils.hooks  # 导入PyTorch钩子函数
from torch.nn import Parameter  # 导入PyTorch神经网络参数
from torch.testing._internal.common_cuda import IS_JETSON  # 导入CUDA测试工具
from torch.testing._internal.common_utils import (  # 导入通用测试工具
    IS_MACOS,
    IS_WINDOWS,
    load_tests,
    NO_MULTIPROCESSING_SPAWN,
    run_tests,
    slowTest,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    TEST_WITH_TORCHDYNAMO,
    TEST_WITH_TSAN,
    TestCase,
)

# load_tests函数用于在sandcastle上自动过滤测试用例以进行分片，下面的赋值消除flake警告
load_tests = load_tests

TEST_REPEATS = 30  # 测试重复次数设为30
HAS_SHM_FILES = os.path.isdir("/dev/shm")  # 检查是否有共享内存文件夹
MAX_WAITING_TIME_IN_SECONDS = 30  # 最大等待时间设为30秒

TEST_CUDA_IPC = (
    torch.cuda.is_available()  # 检查CUDA是否可用
    and sys.platform != "darwin"  # 并且不在macOS平台上
    and sys.platform != "win32"  # 并且不在Windows平台上
    and not IS_JETSON  # 并且不在Jetson平台上
    and not TEST_WITH_ROCM  # 并且不在ROCm环境下
)  # https://github.com/pytorch/pytorch/issues/90940

TEST_MULTIGPU = TEST_CUDA_IPC and torch.cuda.device_count() > 1  # 检查是否支持多GPU并行

if TEST_CUDA_IPC:
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")  # 设置CUDA内存分配器参数为不可扩展


class SubProcess(mp.Process):
    def __init__(self, tensor):
        super().__init__()  # 调用父类构造函数
        self.tensor = tensor  # 初始化子进程的张量
        self.daemon = True  # 设置进程为守护进程

    def run(self):
        self.tensor.add_(3)  # 子进程执行操作：张量中所有元素加3


def _test_cuda_ipc_deadlock_actor(queue, iterations):
    for i in range(iterations):
        if not queue.empty():  # 如果队列不为空
            queue.get()  # 从队列中获取数据
        time.sleep(0.01)  # 线程休眠0.01秒


def _test_cuda_ipc_deadlock_learner(queue, iterations):
    net = torch.nn.LSTM(1, 1).cuda()  # 创建一个CUDA上的LSTM模型
    for i in range(iterations):
        if not queue.full():  # 如果队列未满
            queue.put(copy.deepcopy(net.state_dict()))  # 深拷贝并放入队列
        time.sleep(0.01)  # 线程休眠0.01秒


def simple_fill(queue, event):
    data = queue.get()  # 从队列中获取数据
    data[0][:] = 4  # 修改数据的第一行为4
    event.set()  # 设置事件状态为已触发


def simple_pool_fill(tensor):
    tensor.fill_(4)  # 用4填充张量
    return tensor.add(1)  # 张量所有元素加1后返回


def send_tensor(queue, event, device, dtype):
    t = torch.ones(5, 5, device=device, dtype=dtype)  # 创建指定设备和数据类型的全1张量
    queue.put(t)  # 将张量放入队列
    queue.put(t)  # 再次将张量放入队列
    event.wait()  # 等待事件触发


def send_and_delete_tensors(queue, event, device, dtype, count, size=5):
    for i in range(count):
        t = torch.full([size], i, device=device, dtype=dtype)  # 创建指定设备和数据类型的全值张量
        queue.put(t)  # 将张量放入队列
        del t  # 删除张量对象
    event.wait()  # 等待事件触发


def receive_and_send_sum(queue, out_queue, event, device, dtype, count, size=5):
    s = torch.full([size], 0, device=device, dtype=dtype)  # 创建指定设备和数据类型的全0张量
    for i in range(count):
        t = queue.get()  # 从队列中获取张量
        s += t  # 将张量加到总和张量上
    out_queue.put(s)  # 将结果张量放入输出队列
    event.wait()  # 等待事件触发


def receive_and_send(queue, out_queue, event, count):
    for i in range(count):
        t = queue.get()  # 从队列中获取张量
        out_queue.put(t.clone())  # 克隆张量并放入输出队列
    event.wait()  # 等待事件触发


def sum_tensors(inq, outq):
    pass  # 空函数，暂未实现功能
    # 使用上下文管理器切换到 CUDA 设备 1，所有后续的 CUDA 操作都将在该设备上执行
    with torch.cuda.device(1):
        # 从输入队列中获取张量数据
        tensors = inq.get()
        # 遍历每一个张量
        for tensor in tensors:
            # 将张量的总和作为标量值，设备索引，元素数量和存储器大小放入输出队列
            outq.put(
                (
                    tensor.sum().item(),        # 获取张量所有元素的和并转换为 Python 数字
                    tensor.get_device(),        # 获取张量所在的 CUDA 设备索引
                    tensor.numel(),             # 获取张量的元素数量
                    tensor.storage().size(),    # 获取张量底层存储器的大小
                )
            )
# 从输入队列中获取异常并放入输出队列
def queue_get_exception(inqueue, outqueue):
    # 关闭标准错误流，隐藏预期的错误消息
    os.close(2)
    try:
        # 在 CUDA 上创建一个 5x5 的零张量，并将其放在异常处理块中
        torch.zeros(5, 5).cuda()
    except Exception as e:
        # 将捕获到的异常放入输出队列
        outqueue.put(e)
    else:
        # 如果没有异常，将字符串 "no exception" 放入输出队列
        outqueue.put("no exception")


# 在一个单独的 CUDA 流中将张量乘以二
def cuda_multiply_two(queue, ready, done):
    # 设置就绪标志
    ready.set()
    with torch.cuda.stream(torch.cuda.Stream()):
        # 从队列中获取 CUDA 事件和张量
        cuda_event, tensor = queue.get()
        # 等待 CUDA 事件完成
        cuda_event.wait()
        # 将张量乘以二
        tensor.mul_(2)
        # 记录 CUDA 事件
        cuda_event.record()
        # 设置完成标志
        done.set()
        # 删除 CUDA 事件对象
        del cuda_event


# 共享变量的梯度需求
def requires_grad_variable_sharing(queue, ready):
    # 从队列中获取变量
    var = queue.get()
    # 设置就绪标志
    ready.set()
    # 将变量是否需要梯度的信息放回队列
    queue.put(var.requires_grad)


# 整数参数的序列化
def integer_parameter_serialization(iparam):
    # 增加整数参数值并返回结果（但没有返回语句）


# 自动求导共享
def autograd_sharing(queue, ready, master_modified, device, is_parameter):
    # 从队列中获取变量
    var = queue.get()
    # 设置就绪标志
    ready.set()
    # 等待主修改标志
    master_modified.wait()

    # 创建一个预期的张量
    expected_var = torch.arange(1.0, 26, device=device).view(5, 5)
    expected_var[0, 0] = 1000

    # 检查当前变量是否等于预期的张量
    is_ok = var.data.equal(expected_var)
    # 将当前变量的数据设为全为一
    var.data[:] = torch.ones(5, 5, device=device)

    # 检查当前变量的梯度是否为 None
    is_ok &= var.grad is None
    # 检查当前变量是否没有反向钩子
    is_ok &= not var._backward_hooks

    # 如果是参数，则检查当前变量的类型是否为 Parameter
    if is_parameter:
        is_ok &= type(var) == Parameter
    else:
        is_ok &= type(var) == torch.Tensor

    # 设置当前变量的梯度为全为一
    var._grad = torch.ones(5, 5, device=device)

    # 将检查结果放入队列
    queue.put(is_ok)


# 生产混合类型的张量
def mixed_type_producer(queue, event):
    for _ in range(10):
        # 创建一个浮点数张量并放入队列
        float_tensor = torch.ones(2, 2).float().cuda()
        queue.put(float_tensor)
        # 创建一个字节类型张量并放入队列
        byte_tensor = torch.zeros(2, 2).byte().cuda()
        queue.put(byte_tensor)
        # 等待事件完成并清除事件标志
        event.wait()
        event.clear()


# 简单的自动求导函数
def simple_autograd_function(a=1):
    # 创建一个需要梯度的随机张量，并计算其均值的梯度
    torch.rand(3).requires_grad_(True).mean().backward()
    return a**2


# 文件系统共享上下文管理器
@contextlib.contextmanager
def fs_sharing():
    # 获取当前的共享策略
    prev_strategy = mp.get_sharing_strategy()
    # 设置共享策略为文件系统
    mp.set_sharing_strategy("file_system")
    try:
        yield
    finally:
        # 恢复之前的共享策略
        mp.set_sharing_strategy(prev_strategy)


# 泄漏检查器类
class leak_checker:
    def __init__(self, test_case):
        # 初始化检查的进程 ID 和测试用例
        self.checked_pids = [os.getpid()]
        self.test_case = test_case

    def __enter__(self):
        # 获取当前进程的下一个文件描述符列表
        self.next_fds = self._get_next_fds(10)
        return self

    def __exit__(self, *args):
        # 如果 CUDA 可用，则收集 CUDA IPC
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
        if args[0] is None:
            # 检查在测试结束时第 10 个可用文件描述符的增量是否超过 5
            # TODO: 禁用此检查因为其不稳定
            # available_fds = self._get_next_fds(10)
            # self.test_case.assertLessEqual(
            #     available_fds[-1] - self.next_fds[-1], 5)
            # 检查是否存在共享内存文件
            self.test_case.assertFalse(self.has_shm_files())
        return False

    def check_pid(self, pid):
        # 添加要检查的进程 ID 到列表中
        self.checked_pids.append(pid)
    # 返回指定数量（默认为1）的未使用文件描述符的副本列表
    def _get_next_fds(self, n=1):
        fds = [os.dup(0) for i in range(n)]  # 使用 os.dup(0) 创建新的文件描述符副本列表
        for fd in fds:
            os.close(fd)  # 关闭每个文件描述符副本
        return fds  # 返回文件描述符副本列表

    # 检查是否存在共享内存文件
    def has_shm_files(self, wait=True):
        if not HAS_SHM_FILES:  # 如果系统不支持共享内存文件，则返回 False
            return False

        result = self._has_shm_files()  # 调用内部方法检查是否存在共享内存文件
        if not result or mp.get_sharing_strategy() != "file_system" or not wait:
            return result  # 如果没有共享内存文件或者共享策略不是文件系统或者不需要等待，则直接返回结果

        total_waiting_time = 0  # 总等待时间
        waiting_time = 0.5  # 每次等待时间

        # 在指定的最大等待时间内轮询检查是否存在共享内存文件
        while total_waiting_time <= MAX_WAITING_TIME_IN_SECONDS and result:
            time.sleep(waiting_time)  # 等待一段时间
            total_waiting_time += waiting_time  # 累加已等待的时间
            result = self._has_shm_files()  # 重新检查是否存在共享内存文件

        return result  # 返回最终的检查结果

    # 内部方法：检查是否存在共享内存文件
    def _has_shm_files(self):
        gc.collect()  # 手动触发垃圾回收，释放不再使用的内存
        names = ["torch_" + str(pid) for pid in self.checked_pids]  # 生成需要匹配的共享内存文件名列表
        for filename in os.listdir("/dev/shm"):  # 遍历共享内存目录下的文件列表
            for name in names:  # 遍历预定义的文件名列表
                if filename.startswith(name):  # 如果文件名以预定义的名称开头
                    return True  # 发现匹配的共享内存文件，返回 True
        return False  # 未找到匹配的共享内存文件，返回 False
@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN is not fork-safe since we're forking in a multi-threaded environment",
)
class TestMultiprocessing(TestCase):
    # 在测试之后执行，确保每个测试互相隔离
    def tearDown(self):
        if torch.cuda.is_available():
            # 收集 CUDA IPC 内存
            torch.cuda.ipc_collect()

    def _test_sharing(self, ctx=mp, device="cpu", dtype=torch.float, repeat=1):
        def test_fill():
            # 创建一个大小为 5x5 的零张量，并将其移到指定的设备和数据类型上
            x = torch.zeros(5, 5).to(device, dtype)
            # 创建一个进程间队列和事件
            q = ctx.Queue()
            e = ctx.Event()

            data = [x, x[:, 1]]
            # 将数据放入队列中
            q.put(data)

            # 创建一个简单填充数据的子进程
            p = ctx.Process(target=simple_fill, args=(q, e))
            p.daemon = True
            # 检查并记录子进程的 PID
            lc.check_pid(p.pid)
            # 启动子进程
            p.start()

            total_waiting_time = 0
            waiting_time = 0.5
            is_set = False
            # 当子进程完成后，它会设置事件来通知父进程
            while total_waiting_time <= MAX_WAITING_TIME_IN_SECONDS and not is_set:
                time.sleep(waiting_time)
                total_waiting_time += waiting_time
                is_set = e.is_set()

            # 断言事件已被设置
            self.assertTrue(is_set)
            # 断言数据第一个张量中的所有元素都为 4
            self.assertTrue(data[0].eq(4).all())
            # 断言数据第二个张量中的所有元素都为 4
            self.assertTrue(data[1].eq(4).all())

            # 等待子进程完成，最多等待 100 秒
            p.join(100)
            # 断言子进程已结束
            self.assertFalse(p.is_alive())

        def test_receive():
            # 创建一个进程间队列和事件
            q = ctx.Queue()
            e = ctx.Event()

            # 创建一个发送张量数据的子进程
            p = ctx.Process(target=send_tensor, args=(q, e, device, dtype))
            p.daemon = True
            # 检查并记录子进程的 PID
            lc.check_pid(p.pid)
            # 启动子进程
            p.start()

            # 从队列中获取张量数据
            t1 = q.get()
            t2 = q.get()
            # 断言第一个张量中所有元素都为 1
            self.assertTrue(t1.eq(1).all())
            # 获取张量的存储对象
            s1 = t1.storage()
            s2 = t2.storage()
            # 断言存储对象的类型相同
            self.assertEqual(type(s1), type(s2))
            # 断言存储对象的数据指针相同
            self.assertEqual(s1.data_ptr(), s1.data_ptr())
            # 断言两个张量的内容相同
            self.assertEqual(s1, s2)

            # 需要删除这些张量，以便让子进程能够正确回收它们
            del t1, t2

            # 标记事件为完成，并等待子进程结束，最多等待 100 秒
            e.set()
            p.join(100)
            # 断言子进程已结束
            self.assertFalse(p.is_alive())

        with leak_checker(self) as lc:
            # 循环执行 test_fill 和 test_receive 函数，重复次数为 repeat
            for _ in range(repeat):
                test_fill()
                test_receive()

    def _test_preserve_sharing(self, ctx=mp, repeat=1):
        def do_test():
            # 创建一个大小为 5x5 的正态分布张量
            x = torch.randn(5, 5)
            data = [x.storage(), x, x[2], x[:, 1]]
            # 创建一个进程间队列，并将数据放入队列中
            q = ctx.Queue()
            q.put(data)
            # 从队列中获取新的数据
            new_data = q.get(timeout=1)
            # 断言新数据与原始数据相等，绝对误差和相对误差为 0
            self.assertEqual(new_data, data, atol=0, rtol=0)
            # 获取原始数据第一个张量的 C 数据指针
            storage_cdata = data[0]._cdata
            # 断言新数据第一个张量的 C 数据指针与原始数据相同
            self.assertEqual(new_data[0]._cdata, storage_cdata)
            # 断言新数据中每个张量的存储对象的 C 数据指针与原始数据相同
            for t in new_data[1:]:
                self.assertEqual(t.storage()._cdata, storage_cdata)

        with leak_checker(self):
            # 循环执行 do_test 函数，重复次数为 repeat
            for _ in range(repeat):
                do_test()
    # 定义一个私有方法 _test_pool，用于执行多进程池的测试
    def _test_pool(self, ctx=mp, repeat=1):
        # 内部函数 do_test 用于执行具体的测试任务
        def do_test():
            # 创建具有两个工作进程的进程池对象 p
            p = ctx.Pool(2)
            # 检查进程池中每个进程的进程ID是否正常
            for proc in p._pool:
                lc.check_pid(proc.pid)

            # 创建一个包含四个零张量的列表 buffers
            buffers = [torch.zeros(2, 2) for i in range(4)]
            # 使用进程池 p 对 simple_pool_fill 函数进行映射计算，将结果保存在 results 中
            results = p.map(simple_pool_fill, buffers, 1)
            # 断言结果列表的长度与 buffers 相同
            self.assertEqual(len(results), len(buffers))
            # 断言每个结果 r 都等于全为 5 的 2x2 张量
            for r in results:
                self.assertEqual(r, torch.ones(2, 2) * 5, atol=0, rtol=0)
            # 断言每个缓冲区 b 都等于全为 4 的 2x2 张量
            for b in buffers:
                self.assertEqual(b, torch.ones(2, 2) * 4, atol=0, rtol=0)

            # 关闭并销毁进程池 p
            p.close()
            p.join()

        # 使用 leak_checker 上下文管理器检查内存泄漏
        with leak_checker(self) as lc:
            # 多次执行 do_test 函数，根据 repeat 参数指定的次数
            for _ in range(repeat):
                do_test()

    # 根据当前平台是否为 macOS，选择性地跳过测试
    @unittest.skipIf(
        platform == "darwin", "file descriptor strategy is not supported on macOS"
    )
    # 测试文件描述符共享功能
    def test_fd_sharing(self):
        self._test_sharing(repeat=TEST_REPEATS)

    # 根据当前平台是否为 macOS，选择性地跳过测试
    @unittest.skipIf(
        platform == "darwin", "file descriptor strategy is not supported on macOS"
    )
    # 测试文件描述符保持共享功能
    def test_fd_preserve_sharing(self):
        self._test_preserve_sharing(repeat=TEST_REPEATS)

    # 根据当前平台是否为 macOS，选择性地跳过测试
    @unittest.skipIf(
        platform == "darwin", "file descriptor strategy is not supported on macOS"
    )
    # 测试文件描述符进程池功能
    def test_fd_pool(self):
        self._test_pool(repeat=TEST_REPEATS)

    # 根据是否使用 ASAN 工具链，选择性地跳过测试
    @unittest.skipIf(
        TEST_WITH_ASAN,
        "seems to hang with ASAN, see https://github.com/pytorch/pytorch/issues/5326",
    )
    # 测试文件系统共享功能
    def test_fs_sharing(self):
        with fs_sharing():
            # 如果在 macOS 上，仅运行一次，否则根据 TEST_REPEATS 的值执行多次
            repeat = 1 if IS_MACOS else TEST_REPEATS
            self._test_sharing(repeat=repeat)

    # 根据是否使用 TorchDynamo，选择性地跳过测试
    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "Fail to clean up temporary /dev/shm/torch_* file, see https://github.com/pytorch/pytorch/issues/91467",
    )
    # 测试文件系统保持共享功能
    def test_fs_preserve_sharing(self):
        with fs_sharing():
            self._test_preserve_sharing(repeat=TEST_REPEATS)

    # 根据是否使用 TorchDynamo，选择性地跳过测试
    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "Fail to clean up temporary /dev/shm/torch_* file, see https://github.com/pytorch/pytorch/issues/91467",
    )
    # 测试文件系统进程池功能
    def test_fs_pool(self):
        with fs_sharing():
            self._test_pool(repeat=TEST_REPEATS)

    # 检查是否存在共享内存文件，并根据情况选择性地跳过测试
    @unittest.skipIf(not HAS_SHM_FILES, "don't not how to check if shm files exist")
    # 根据是否使用 TorchDynamo，选择性地跳过测试
    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "Fail to clean up temporary /dev/shm/torch_* file, see https://github.com/pytorch/pytorch/issues/91467",
    )
    # 定义一个测试方法，用于测试文件系统功能
    def test_fs(self):
        
        # 定义一个内部方法，用于将数据放入队列
        def queue_put():
            # 创建一个双精度存储对象
            x = torch.DoubleStorage(4)
            # 创建一个多进程队列
            q = mp.Queue()
            # 断言当前没有共享内存文件存在
            self.assertFalse(lc.has_shm_files())
            # 将数据放入队列
            q.put(x)
            # 等待一段时间，队列异步序列化数据
            time.sleep(0.05)
            # 断言已经存在共享内存文件，但不等待其完全创建
            self.assertTrue(lc.has_shm_files(wait=False))
            # 从队列中取出数据
            q.get()

        # 运行在文件系统共享环境下的测试代码块，同时进行内存泄漏检查
        with fs_sharing(), leak_checker(self) as lc:
            # 多次重复运行数据放入队列的操作
            for _ in range(TEST_REPEATS):
                queue_put()

    # 定义一个测试方法，用于测试张量的继承
    def test_inherit_tensor(self):
        # 创建一个5x5全零张量
        t = torch.zeros(5, 5)
        # 创建一个子进程，共享张量内存
        p = SubProcess(t.share_memory_())
        # 启动子进程
        p.start()
        # 等待子进程结束，最多等待2秒
        p.join(2)
        # 如果子进程未能及时结束，输出提示信息
        if p.exitcode is None:
            print("test_inherit_tensor: SubProcess too slow")
        else:
            # 断言张量值是否为全3的5x5张量，允许误差为0
            self.assertEqual(t, torch.ones(5, 5) * 3, atol=0, rtol=0)

    # 如果在Windows平台，跳过该测试用例（需要使用fork多进程模式）
    @unittest.skipIf(IS_WINDOWS, "Test needs to use fork multiprocessing")
    def test_autograd_errors(self):
        # 获取fork多进程上下文
        ctx = mp.get_context("fork")
        # 调用简单的自动求导函数
        simple_autograd_function()
        # 当存在GPU或其他特定加速设备时
        if (
            torch.cuda.is_available()
            or torch.backends.mps.is_available()
            or torch.xpu.is_available()
        ):
            # 断言运行时错误包含特定字符串
            with self.assertRaisesRegex(RuntimeError, r"Unable to handle autograd"):
                # 使用进程池映射函数到数据列表
                with ctx.Pool(3) as pool:
                    pool.map(simple_autograd_function, [1, 2, 3])
        else:
            # 在非加速设备情况下，使用进程池映射函数到数据列表
            with ctx.Pool(3) as pool:
                pool.map(simple_autograd_function, [1, 2, 3])

    # 如果不支持使用spawn启动方法，跳过该测试用例
    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Test needs to use spawn multiprocessing"
    )
    def test_autograd_fine_with_spawn(self):
        # 获取spawn多进程上下文
        ctx = mp.get_context("spawn")
        # 调用简单的自动求导函数
        simple_autograd_function()
        # 使用进程池映射函数到数据列表
        with ctx.Pool(3) as pool:
            pool.map(simple_autograd_function, [1, 2, 3])

    # 如果不支持使用spawn启动方法，或CUDA IPC不可用，跳过该测试用例
    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_cuda_simple(self):
        # 初始化一个CUDA浮点数张量（在内存泄漏检查之外）
        torch.cuda.FloatTensor([1])
        # 调用测试共享函数，使用spawn多进程上下文，传入CUDA相关参数
        self._test_sharing(mp.get_context("spawn"), "cuda", torch.float)

    # 如果不支持使用spawn启动方法，或CUDA IPC不可用，跳过该测试用例
    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_cuda_memory_allocation(self):
        # 获取spawn多进程上下文
        ctx = mp.get_context("spawn")
        # 创建一个多进程队列
        q = ctx.Queue()
        # 创建一个多进程事件
        e = ctx.Event()
        # 创建一个子进程，目标函数为发送和删除张量，传入CUDA相关参数
        p = ctx.Process(
            target=send_and_delete_tensors, args=(q, e, "cuda", torch.int, 5)
        )
        # 启动子进程
        p.start()
        # 初始化一个空列表
        t = []
        # 循环5次
        for _ in range(5):
            # 从队列中获取数据，添加到列表中
            t.append(q.get())
        # 断言列表第一个元素为全0的长度为5的整型32位张量
        self.assertEqual(t[0], torch.full([5], 0, dtype=torch.int32))
        # 删除列表
        del t
        # 设置事件状态为已触发
        e.set()
        # 等待子进程结束，最多等待1秒
        p.join(1)
    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    跳过测试，如果不支持多进程的 spawn 启动方法，或者 CUDA IPC 不可用

    def test_cuda_ipc_deadlock(self):
        定义一个名为 test_cuda_ipc_deadlock 的测试方法

        ctx = mp.get_context("spawn")
        使用 multiprocessing 模块获取 spawn 上下文

        queue = ctx.Queue(1)
        创建一个最大容量为 1 的队列

        processes = dict(
            a=ctx.Process(target=_test_cuda_ipc_deadlock_actor, args=(queue, 100)),
            l=ctx.Process(target=_test_cuda_ipc_deadlock_learner, args=(queue, 100)),
        )
        创建两个进程，分别指向 _test_cuda_ipc_deadlock_actor 和 _test_cuda_ipc_deadlock_learner 函数，传入相同的队列和参数 100

        for p in processes.values():
            启动每个进程
            p.start()

        for p in processes.values():
            等待每个进程最多 10 秒钟
            p.join(10)

        for p in processes.values():
            断言每个进程都已经终止
            self.assertFalse(p.is_alive())

    @slowTest
    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    跳过测试，如果不支持多进程的 spawn 启动方法，或者 CUDA IPC 不可用

    def test_cuda_send_many(self, name=None, size=5, count=100000):
        定义一个名为 test_cuda_send_many 的测试方法，接受可选参数 name，默认参数 size 和 count

        ctx = mp.get_context("spawn")
        使用 multiprocessing 模块获取 spawn 上下文

        q1 = ctx.Queue()
        q2 = ctx.Queue()
        q3 = ctx.Queue()
        创建三个空队列 q1, q2, q3

        e1 = ctx.Event()
        e2 = ctx.Event()
        e3 = ctx.Event()
        创建三个事件对象 e1, e2, e3

        p1 = ctx.Process(
            target=send_and_delete_tensors,
            args=(q1, e1, "cuda", torch.long, count, size),
        )
        创建一个进程 p1，指向 send_and_delete_tensors 函数，并传入参数 q1, e1, "cuda", torch.long, count, size

        p2 = ctx.Process(target=receive_and_send, args=(q1, q2, e2, count))
        创建一个进程 p2，指向 receive_and_send 函数，并传入参数 q1, q2, e2, count

        p3 = ctx.Process(
            target=receive_and_send_sum,
            args=(q2, q3, e3, "cuda", torch.long, count, size),
        )
        创建一个进程 p3，指向 receive_and_send_sum 函数，并传入参数 q2, q3, e3, "cuda", torch.long, count, size

        p1.start()
        启动进程 p1

        p2.start()
        启动进程 p2

        p3.start()
        启动进程 p3

        result = q3.get()
        从队列 q3 中获取结果

        self.assertEqual(result[0], int(count * (count - 1) / 2))
        断言结果的第一个元素等于 count * (count - 1) / 2 的整数部分

        del result
        删除结果变量

        e1.set()
        设置事件 e1

        e2.set()
        设置事件 e2

        e3.set()
        设置事件 e3

        p1.join(1)
        等待进程 p1 最多 1 秒钟

        p2.join(1)
        等待进程 p2 最多 1 秒钟

        p3.join(1)
        等待进程 p3 最多 1 秒钟

    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    @unittest.skipIf(not TEST_MULTIGPU, "found only 1 GPU")
    跳过测试，如果不支持多进程的 spawn 启动方法，或者 CUDA IPC 不可用，或者检测到只有一个 GPU
    def test_cuda_small_tensors(self):
        # 检查多个小张量，这些张量可能会使用相同的底层缓存分配
        ctx = mp.get_context("spawn")
        # 创建一个空列表来存放张量
        tensors = []
        # 循环创建五个张量
        for i in range(5):
            # 确定设备索引，交替选择两个设备
            device = i % 2
            # 生成并加入到张量列表中，使用CUDA加速
            tensors += [torch.arange(i * 5.0, (i + 1) * 5).cuda(device)]

        # 创建进程间通信的队列
        inq = ctx.Queue()
        outq = ctx.Queue()
        # 将张量列表放入输入队列
        inq.put(tensors)
        # 创建新的进程，目标函数为sum_tensors，传入输入输出队列作为参数
        p = ctx.Process(target=sum_tensors, args=(inq, outq))
        # 启动进程
        p.start()

        # 创建一个空列表来接收结果
        results = []
        # 从输出队列中获取结果，循环五次
        for _ in range(5):
            results.append(outq.get())
        # 等待进程结束
        p.join()

        # 遍历张量列表，并依次验证结果
        for i, _tensor in enumerate(tensors):
            # 解包结果元组
            v, device, tensor_size, storage_size = results[i]
            # 断言验证张量元素和
            self.assertEqual(v, torch.arange(i * 5.0, (i + 1) * 5).sum())
            # 断言验证设备索引
            self.assertEqual(device, i % 2)
            # 断言验证张量大小
            self.assertEqual(tensor_size, 5)

            # 尽管可能期望如此，实际上并非如此！在CUDA缓存分配器经过IPC后，
            # 存储大小是整个内存块的缓存cudaMalloc的大小，而不仅仅是存储的大小。
            # 更多信息请参阅注释 [CUDA IPC and the caching allocator]

        # 释放最后一个张量的引用
        del _tensor
        # 删除张量列表的引用
        del tensors

        # 收集当前进程（生产者）的文件，确保没有东西保持对发送张量的引用

        # 由于性能原因，我们需要收集CUDA MP实现中的一个共享内存“文件”
        torch.cuda.ipc_collect()

    @unittest.skipIf(IS_WINDOWS, "not applicable to Windows (only fails with fork)")
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_bad_call(self):
        # 初始化CUDA
        t = torch.zeros(5, 5).cuda().cpu()
        # 创建多进程间通信的队列
        inq = mp.Queue()
        outq = mp.Queue()
        # 创建新的进程，目标函数为queue_get_exception，传入输入输出队列作为参数
        p = mp.Process(target=queue_get_exception, args=(inq, outq))
        # 启动进程
        p.start()
        # 将张量放入输入队列
        inq.put(t)
        # 等待进程结束
        p.join()
        # 断言异常类型为RuntimeError
        self.assertIsInstance(outq.get(), RuntimeError)

    @unittest.skipIf(IS_WINDOWS, "not applicable to Windows (only fails with fork)")
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_wrong_cuda_fork(self):
        # 运行带有PyTorch API使用的标准错误流的TestCase
        stderr = TestCase.runWithPytorchAPIUsageStderr(
            """\
# 导入torch模块
import torch
# 从torch.multiprocessing模块中导入Process类
from torch.multiprocessing import Process

# 定义一个函数run，参数为rank
def run(rank):
    # 设置当前CUDA设备为rank指定的设备
    torch.cuda.set_device(rank)

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 定义并设置进程数量为2
    size = 2
    # 创建一个空列表用于存放进程对象
    processes = []
    # 遍历从0到size-1的排名
    for rank in range(size):
        # 创建一个大小为20x2的随机张量，并将其放置在CUDA设备上
        x = torch.rand(20, 2).cuda()
        # 创建一个新的进程对象，目标函数为run，参数为当前的rank
        p = Process(target=run, args=(rank,))
        # 启动进程
        p.start()
        # 将进程对象添加到列表中
        processes.append(p)
    # 等待所有进程结束
    for p in processes:
        p.join()
    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    @unittest.skipIf(not TEST_MULTIGPU, "found only 1 GPU")
    def test_event_handle_multi_gpu(self):
        # 定义两个 CUDA 设备对象
        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")
        with torch.cuda.device(d0):
            # 在第一个 CUDA 设备上创建一个 CUDA 事件对象 e0
            e0 = torch.cuda.Event(enable_timing=False, interprocess=True)

        with torch.cuda.device(d1):
            # 在第二个 CUDA 设备上，使用 e0 的 IPC 句柄创建一个事件句柄
            e0.ipc_handle()

        with torch.cuda.device(d0):
            # 在第一个 CUDA 设备上创建另一个 CUDA 事件对象 e1，并创建一个 CUDA 流对象
            e1 = torch.cuda.Event(enable_timing=False, interprocess=True)
            stream = torch.cuda.Stream()
            torch.cuda._sleep(50000000)  # 自旋等待大约 50 毫秒
            e1.record(stream)

        with torch.cuda.device(d1):
            # 在第二个 CUDA 设备上，使用 e1 的 IPC 句柄创建一个事件句柄
            e1.ipc_handle()

    @staticmethod
    def _test_event_handle_importer_consumer(handle, p2c, c2p):
        # 从 IPC 句柄创建一个 CUDA 事件对象 e1
        e1 = torch.cuda.Event.from_ipc_handle(0, handle)
        c2p.put(0)  # 通知父进程子进程已经准备好
        p2c.get()  # 等待父进程记录完毕
        e1.synchronize()
        c2p.put(1)  # 通知子进程同步完成
        p2c.get()  # 等待父进程在销毁子事件之前完成

    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_event_handle_importer(self):
        # 在当前 CUDA 设备上创建一个 CUDA 事件对象 e0
        e0 = torch.cuda.Event(enable_timing=False, interprocess=True)
        self.assertTrue(e0.query())

        # 使用 spawn 启动方法创建一个 multiprocessing 上下文
        ctx = mp.get_context("spawn")
        p2c = ctx.SimpleQueue()
        c2p = ctx.SimpleQueue()
        p = ctx.Process(
            target=TestMultiprocessing._test_event_handle_importer_consumer,
            args=(e0.ipc_handle(), p2c, c2p),
        )
        p.start()

        c2p.get()  # 等待子进程准备就绪
        torch.cuda._sleep(50000000)  # 自旋等待大约 50 毫秒
        e0.record()
        p2c.put(0)  # 通知子事件已经记录完毕

        self.assertFalse(e0.query())
        c2p.get()  # 等待子进程同步完成
        self.assertTrue(e0.query())
        p2c.put(1)  # 通知子进程父进程已经完成
        p.join()

    @staticmethod
    def _test_event_handle_exporter_consumer(handle, p2c, c2p):
        # 在 CUDA 流中创建一个 CUDA 事件对象 e1，使用 IPC 句柄从父进程当前设备创建 e1
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            e1 = torch.cuda.Event.from_ipc_handle(torch.cuda.current_device(), handle)
            torch.cuda._sleep(50000000)  # 自旋等待大约 50 毫秒
            e1.record()
            c2p.put(0)
            # 等待父进程完成同步后再销毁 e1
            p2c.get()
    # 使用装饰器跳过测试，条件是不支持使用 spawn 启动方法的环境
    # 或者 CUDA IPC 不可用时跳过测试
    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that don't support multiprocessing with spawn start method",
    )
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_event_handle_exporter(self):
        # 创建一个 CUDA 事件对象，禁用计时功能，并启用跨进程通信
        e0 = torch.cuda.Event(enable_timing=False, interprocess=True)

        # 获取 spawn 上下文
        ctx = mp.get_context("spawn")
        # 创建父进程到子进程的简单队列
        p2c = ctx.SimpleQueue()
        # 创建子进程到父进程的简单队列
        c2p = ctx.SimpleQueue()
        # 创建子进程，目标函数是 _test_event_handle_exporter_consumer，
        # 参数包括 CUDA 事件的 IPC 句柄以及两个队列
        p = ctx.Process(
            target=TestMultiprocessing._test_event_handle_exporter_consumer,
            args=(e0.ipc_handle(), p2c, c2p),
        )
        # 启动子进程
        p.start()
        # 等待子进程中的事件被记录
        c2p.get()

        # 检查事件是否未记录
        self.assertFalse(e0.query())
        # 同步事件
        e0.synchronize()
        # 检查事件是否已记录
        self.assertTrue(e0.query())
        # 向父进程到子进程的队列中放入数据
        p2c.put(0)
        # 等待子进程结束
        p.join()

    # 测试空张量的共享
    def _test_empty_tensor_sharing(self, dtype, device):
        # 创建一个空张量，指定数据类型和设备
        empty = torch.tensor([], dtype=dtype, device=device)
        # 将空张量放入队列
        q.put(empty)
        # 从队列中获取数据，设置超时为1秒
        out = q.get(timeout=1)
        # 断言获取的数据与放入的空张量相等
        self.assertEqual(out, empty)

    # 测试空张量的共享，CPU 设备
    def test_empty_tensor_sharing(self):
        self._test_empty_tensor_sharing(torch.float32, torch.device("cpu"))
        self._test_empty_tensor_sharing(torch.int64, torch.device("cpu"))

    # 使用装饰器跳过测试，条件是 CUDA 不可用
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_empty_tensor_sharing_cuda(self):
        # 测试空张量的共享，CUDA 设备
        self._test_empty_tensor_sharing(torch.float32, torch.device("cuda"))
        self._test_empty_tensor_sharing(torch.int64, torch.device("cuda"))

    # 测试自动求导的共享
    def _test_autograd_sharing(self, var, ctx=mp, is_parameter=False):
        # 确定变量所在设备
        device = "cuda" if var.is_cuda else "cpu"

        # 创建两个事件对象
        ready = ctx.Event()
        master_modified = ctx.Event()
        # 创建一个队列
        queue = ctx.Queue()
        # 创建子进程，目标函数是 autograd_sharing，传入参数包括队列、两个事件、设备类型和是否是参数
        p = ctx.Process(
            target=autograd_sharing,
            args=(queue, ready, master_modified, device, is_parameter),
        )
        # 将子进程设置为守护进程
        p.daemon = True
        # 启动子进程
        p.start()

        # 定义一个不可序列化的钩子函数，用于测试
        @torch.utils.hooks.unserializable_hook
        def hook(*unused):
            pass

        # 如果变量需要梯度，则注册钩子函数
        if var.requires_grad:
            var.register_hook(hook)
        # 设置梯度为全零张量
        var._grad = torch.zeros(5, 5, device=device)
        # 将变量放入队列
        queue.put(var)

        # 等待子进程就绪
        ready.wait()
        # 修改变量数据
        var.data[0, 0] = 1000
        # 修改梯度数据
        var.grad.data[:] = torch.ones(5, 5, device=device) * 4
        # 通知子进程主进程已修改
        master_modified.set()

        # 从队列获取子进程返回的结果
        worker_ok = queue.get()
        # 断言子进程成功处理
        self.assertTrue(worker_ok)

        # 断言变量数据与预期一致
        self.assertEqual(var.data, torch.ones(5, 5, device=device))
        # 断言梯度数据与预期一致
        self.assertEqual(var.grad.data, torch.ones(5, 5, device=device) * 4)
        # 等待子进程结束，设置超时时间为100秒
        p.join(100)
        # 检查子进程是否已结束
        self.assertFalse(p.is_alive())

    # 检查使用 cudaMalloc 分配在不同存储类型间的共享
    # (Issue #11422)
    # 测试混合类型 CUDA 共享
    def _test_mixed_types_cuda_sharing(self, ctx=mp):
        # 创建一个全为1的浮点数张量
        all_ones = torch.ones(2, 2).float()
        # 创建一个全为0的字节张量
        all_zeros = torch.zeros(2, 2).byte()
        # 使用给定的上下文创建一个队列
        queue = ctx.Queue()
        # 使用给定的上下文创建一个事件
        event = ctx.Event()

        # 创建一个进程，目标函数为 mixed_type_producer，传入队列和事件作为参数
        p = ctx.Process(target=mixed_type_producer, args=(queue, event))

        # 启动进程
        p.start()

        # 循环10次
        for _ in range(10):
            # 从队列中获取浮点数张量
            float_tensor = queue.get()
            # 从队列中获取字节张量
            byte_tensor = queue.get()
            # 断言获取的浮点数张量等于预设的全为1的张量
            self.assertEqual(float_tensor, all_ones)
            # 断言获取的字节张量等于预设的全为0的张量
            self.assertEqual(byte_tensor, all_zeros)
            # 删除浮点数张量和字节张量的引用
            del float_tensor, byte_tensor
            # 设置事件状态为已触发
            event.set()

        # 等待5秒
        time.sleep(5)
        # 等待进程结束
        p.join()

    # 跳过测试如果正在使用 ASAN，因为它会在 https://github.com/pytorch/pytorch/issues/94024 中非确定性地挂起
    @unittest.skipIf(
        TEST_WITH_ASAN,
        "non-deterministically hangs with ASAN https://github.com/pytorch/pytorch/issues/94024",
    )
    def test_variable_sharing(self):
        # 对于每个需要梯度的状态（True 或 False）
        for requires_grad in [True, False]:
            # 创建一个张量，从1到25，形状为5x5，设置是否需要梯度
            var = torch.arange(1.0, 26).view(5, 5).requires_grad_(requires_grad)
            # 调用 _test_autograd_sharing 方法，传入张量作为参数
            self._test_autograd_sharing(var)

    # 参见 https://github.com/pytorch/pytorch/issues/14997
    @unittest.skipIf(TEST_WITH_ASAN, "non-deterministically hangs with ASAN")
    def test_leaf_variable_sharing(self):
        # 定义设备列表，默认为 CPU
        devices = ["cpu"]
        # 如果 CUDA 可用且不禁用多处理池的 spawn 方法，并且测试 CUDA IPC
        if torch.cuda.is_available() and not NO_MULTIPROCESSING_SPAWN and TEST_CUDA_IPC:
            # 添加 CUDA 到设备列表中
            devices.append("cuda")
        # 遍历设备列表
        for device in devices:
            # 对于每个需要梯度的状态（True 或 False）
            for requires_grad in [True, False]:
                # 创建一个张量，从1到25，在指定设备上创建，并设置是否需要梯度
                var = (
                    torch.arange(1.0, 26, device=device)
                    .view(5, 5)
                    .requires_grad_(requires_grad)
                )
                # 断言张量是叶子节点
                self.assertTrue(var.is_leaf)
                # 获取 multiprocessing 的上下文，如果设备为 CUDA，则使用 spawn 方法
                ctx = mp.get_context("spawn") if device == "cuda" else mp
                # 创建一个事件对象
                ready = ctx.Event()
                # 创建一个队列对象
                queue = ctx.Queue()
                # 创建一个进程，目标函数为 requires_grad_variable_sharing，传入队列和事件作为参数
                p = ctx.Process(
                    target=requires_grad_variable_sharing, args=(queue, ready)
                )
                # 将进程设置为守护进程
                p.daemon = True
                # 启动进程
                p.start()
                # 将张量放入队列
                queue.put(var)
                # 等待事件为真
                ready.wait()
                # 从队列获取 worker_requires_grad
                worker_requires_grad = queue.get()
                # 断言 worker_requires_grad 是否等于 requires_grad
                self.assertTrue(worker_requires_grad == requires_grad)

    # 测试非叶子节点的变量共享
    def test_non_leaf_variable_sharing(self):
        # 定义设备列表，如果 CUDA 可用则包括 CUDA
        devices = ["cpu"] if not torch.cuda.is_available() else ["cpu", "cuda"]
        # 遍历设备列表
        for device in devices:
            # 创建一个张量 var0，从1到25，在指定设备上创建，设置需要梯度
            var0 = torch.arange(1.0, 26, device=device).view(5, 5).requires_grad_(True)
            # 创建张量 var，是 var0 的两倍
            var = var0 * 2
            # 创建一个 multiprocessing 的简单队列
            queue = mp.SimpleQueue()
            # 断言在放入队列时会引发 RuntimeError 异常，内容包含 "requires_grad"
            self.assertRaisesRegex(
                RuntimeError, r"requires_grad", lambda: queue.put(var)
            )

    # 如果不支持 multiprocessing 的 spawn 启动方法，则禁用该测试
    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    # 如果 CUDA IPC 不可用，则跳过该测试
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    # 测试 CUDA 变量共享的功能
    def test_cuda_variable_sharing(self):
        # 针对是否需要梯度进行迭代，创建一个在 CUDA 设备上的张量
        for requires_grad in [True, False]:
            var = (
                torch.arange(1.0, 26, device="cuda")  # 创建从 1 到 25 的张量，放在 CUDA 设备上
                .view(5, 5)  # 将张量重新形状为 5x5 的矩阵
                .requires_grad_(requires_grad)  # 设置是否需要梯度
            )
            # 调用 _test_autograd_sharing 方法，测试自动求导共享功能
            self._test_autograd_sharing(var, mp.get_context("spawn"))

    # 在不支持 spawn 启动方法的环境下禁用测试
    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    # 如果 CUDA IPC 不可用，则跳过测试
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    # 测试混合类型的 CUDA 共享功能
    def test_mixed_types_cuda_sharing(self):
        # 调用 _test_mixed_types_cuda_sharing 方法，使用 spawn 上下文进行测试
        self._test_mixed_types_cuda_sharing(mp.get_context("spawn"))

    # 测试参数共享的功能
    def test_parameter_sharing(self):
        # 创建一个参数张量，需要梯度
        param = Parameter(torch.arange(1.0, 26).view(5, 5))
        # 调用 _test_autograd_sharing 方法，测试自动求导共享功能
        self._test_autograd_sharing(param, is_parameter=True)

    # 在不支持 spawn 启动方法的环境下禁用测试
    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    # 如果 CUDA IPC 不可用，则跳过测试
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    # 测试 CUDA 参数共享的功能
    def test_cuda_parameter_sharing(self):
        # 创建一个在 CUDA 设备上的参数张量
        param = Parameter(torch.arange(1.0, 26, device="cuda").view(5, 5))
        # 调用 _test_autograd_sharing 方法，使用 spawn 上下文进行测试，参数共享
        self._test_autograd_sharing(param, mp.get_context("spawn"), is_parameter=True)

    # 在不支持 spawn 启动方法的环境下禁用测试
    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    # 测试整数参数在 CPU 上的序列化
    def test_integer_parameter_serialization_cpu(self):
        # 调用 _test_integer_parameter_serialization 方法，在 CPU 上进行测试
        self._test_integer_parameter_serialization(device="cpu")

    # 在不支持 spawn 启动方法的环境下禁用测试
    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    # 如果 CUDA IPC 不可用，则跳过测试
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    # 测试整数参数在 CUDA 上的序列化
    def test_integer_parameter_serialization_cuda(self):
        # 调用 _test_integer_parameter_serialization 方法，在 CUDA 上进行测试
        self._test_integer_parameter_serialization(device="cuda")

    # 测试整数参数的序列化功能
    def _test_integer_parameter_serialization(self, device):
        # 创建一个整数张量参数，设备由参数指定
        param = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.int64, device=device), requires_grad=False
        )

        # 获取 spawn 上下文
        ctx = mp.get_context("spawn")
        # 创建一个进程，调用 integer_parameter_serialization 函数进行序列化
        p = ctx.Process(target=integer_parameter_serialization, args=(param,))
        p.start()
        p.join()

        # 断言进程退出码为 0，表明序列化成功
        self.assertEqual(
            0,
            p.exitcode,
            msg=f'Failed to serialize successfully for "{device}" device!',
        )

    # 测试空的共享内存张量
    def test_empty_shared(self):
        # 创建一个空的张量，并共享其内存
        t = torch.tensor([])
        t.share_memory_()

    # 测试张量是否共享内存
    def _test_is_shared(self):
        # 创建一个随机数填充的张量，并断言其不共享内存
        t = torch.randn(5, 5)
        self.assertFalse(t.is_shared())
        # 共享张量内存
        t.share_memory_()
        # 断言张量共享内存
        self.assertTrue(t.is_shared())

    # 如果运行平台是 macOS，则跳过测试，因为文件描述符策略不受支持
    @unittest.skipIf(
        platform == "darwin", "file descriptor strategy is not supported on macOS"
    )
    # 测试张量是否共享内存
    def test_is_shared(self):
        # 调用 _test_is_shared 方法进行测试
        self._test_is_shared()

    # 测试文件系统是否共享内存
    def test_fs_is_shared(self):
        # 使用 fs_sharing 上下文，测试张量是否共享内存
        with fs_sharing():
            self._test_is_shared()
    # 如果 CUDA 不可用，则跳过测试。用于检查是否在 CUDA 上共享张量。
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_is_shared_cuda(self):
        # 创建一个5x5的随机张量，并将其放置在 CUDA 设备上
        t = torch.randn(5, 5).cuda()
        # 断言张量是否在多个进程间共享
        self.assertTrue(t.is_shared())
    
    # 如果操作系统不是 Linux，则跳过测试。需要 prctl(2) 库支持。
    @unittest.skipIf(
        sys.platform != "linux",
        "Only runs on Linux; requires prctl(2)",
    )
    def test_set_thread_name(self):
        # 设置线程名称为 "test name"
        name = "test name"
        mp._set_thread_name(name)
        # 断言获取的线程名称与设置的名称相同
        self.assertEqual(mp._get_thread_name(), name)
# 如果当前脚本被作为主程序运行，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```