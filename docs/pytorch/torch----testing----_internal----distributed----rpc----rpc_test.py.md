# `.\pytorch\torch\testing\_internal\distributed\rpc\rpc_test.py`

```
# 忽略 MyPy 的错误提示

import concurrent.futures  # 引入并发执行任务的库
import contextlib  # 提供上下文管理工具的模块
import json  # JSON 编码和解码模块
import os  # 提供了与操作系统交互的功能
import sys  # 提供了访问 Python 解释器的变量和函数
import threading  # 提供了线程相关的操作和同步原语
import time  # 提供了时间操作相关的函数

from collections import namedtuple  # 命名元组，创建带字段名称的数据结构
from functools import partial  # 创建偏函数的工具
from threading import Event  # 线程事件对象，用于线程间通信
from threading import Lock  # 线程锁，用于同步线程之间的访问

from unittest import mock  # 提供了用于单元测试的模拟对象

import torch  # PyTorch 深度学习库
import torch.nn as nn  # PyTorch 神经网络模块
import torch.distributed as dist  # PyTorch 分布式通信模块
import torch.distributed.rpc as rpc  # PyTorch 分布式 RPC 模块
import torch.distributed.autograd as dist_autograd  # PyTorch 分布式自动求导模块
from torch.distributed.rpc import RRef, _get_debug_info, _rref_context_get_debug_info, WorkerInfo  # RPC 相关的对象和函数
from torch.distributed.rpc.api import _use_rpc_pickler, _thread_local_var, _wait_all  # RPC 相关的 API 函数
from torch.distributed.rpc.internal import (
    PythonUDF,  # RPC 内部使用的用户定义函数对象
    RPCExecMode,  # RPC 执行模式
    _internal_rpc_pickler,  # 内部 RPC 序列化器
    _build_rpc_profiling_key,  # 构建 RPC 性能分析的键
)
from torch.futures import Future  # 异步操作的 Future 对象
from torch.testing._internal.common_distributed import (
    skip_if_lt_x_gpu,  # 如果 GPU 数量少于指定值，则跳过测试
    captured_output,  # 捕获标准输出和错误的上下文管理器
    tp_transports,  # 单元测试传输协议的模块
)
from torch.testing._internal.common_utils import (
    IS_MACOS,  # 当前操作系统是否为 macOS
    load_tests,  # 加载测试用例的函数
    skip_but_pass_in_sandcastle_if,  # 如果在沙堡环境中，跳过测试但是标记为通过
    get_cycles_per_ms,  # 获取每毫秒的 CPU 循环数
)

from torch.testing._internal.dist_utils import (
    dist_init,  # 分布式初始化函数
    get_function_event,  # 获取函数调用事件
    initialize_pg,  # 初始化进程组
    wait_until_node_failure,  # 等待直到节点失败
    wait_until_pending_futures_and_users_flushed,  # 等待未决的 Futures 和用户完成
    wait_until_owners_and_forks_on_rank,  # 等待特定排名上的所有所有者和派生对象
    worker_name,  # 获取当前工作进程的名称
)
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,  # RPC 代理测试夹具类
)
from torch.testing._internal.common_utils import TemporaryFileName  # 临时文件名生成器

from torch.autograd.profiler_legacy import profile as _profile  # 旧版性能分析器

import operator  # 操作符模块，提供了 Python 中常见的操作符的函数实现


def foo_add():
    # 执行 torch.add 操作，将两个张量各元素相加
    return torch.add(torch.ones(1), torch.ones(1))

def udf_with_torch_ops(device=-1, use_record_function=False):
    # 设备上下文管理器，根据传入参数选择在 CPU 还是 GPU 上运行
    device_ctx = contextlib.nullcontext() if device == -1 else torch.cuda.device(device)
    # 记录函数上下文管理器，根据传入参数选择是否记录函数调用
    record_function_ctx = (
        torch.autograd.profiler.record_function("##forward##")
        if use_record_function
        else contextlib.nullcontext()
    )
    # 使用设备和记录函数的上下文管理器
    with device_ctx, record_function_ctx:
        t1, t2 = torch.ones(1), torch.ones(1)  # 创建两个张量 t1 和 t2，元素值为 1
        t = torch.add(t1, t2)  # 执行张量相加操作
        t = torch.mul(t, t)  # 执行张量元素对应相乘操作
        t = t.relu()  # 执行张量的 ReLU 激活函数操作
        t = t.sigmoid()  # 执行张量的 sigmoid 激活函数操作

# 预期作为上述函数的一部分执行的远程操作事件列表
EXPECTED_REMOTE_EVENTS = [
    "aten::ones",  # 创建元素值为 1 的张量
    "aten::ones",  # 创建元素值为 1 的张量
    "aten::add",  # 张量相加操作
    "aten::mul",  # 张量元素对应相乘操作
    "aten::relu",  # 张量的 ReLU 激活函数操作
    "aten::clamp_min",  # 张量的最小值截断操作
    "aten::sigmoid",  # 张量的 sigmoid 激活函数操作
]

# 用于 RPC 性能分析的远程操作前缀字符串
REMOTE_OP_STR = "#remote_op: "

VALUE_FUTURE = concurrent.futures.Future()  # 创建一个并发 Future 对象
DONE_FUTURE = concurrent.futures.Future()  # 创建另一个并发 Future 对象

FIFTY_MIL_CYCLES = 50000000  # 定义 5000 万个 CPU 循环数的常量

_rpc_barrier_count = 0  # RPC 障碍计数器的全局变量

def _increment_count():
    # 增加 RPC 障碍计数器的函数
    global _rpc_barrier_count
    _rpc_barrier_count += 1

def _reset_count():
    # 重置 RPC 障碍计数器的函数
    global _rpc_barrier_count
    _rpc_barrier_count = 0

class StubRpcAgent:
    def __init__(self, world_size):
        self.world_size = world_size
        # RPC 代理的初始化函数，设置世界大小
    # 获取工作进程信息的方法
    def get_worker_infos(self):
        # 返回一个集合，集合中包含多个 WorkerInfo 对象，每个对象有工作进程的名称和ID
        return {
            WorkerInfo(name=worker_name(rank), id=rank)
            for rank in range(self.world_size)
        }
# 创建一个模拟对象并返回，用作 RpcBackendOptions 的替代品
def _stub_construct_rpc_backend_options_handler(**kwargs):
    return mock.Mock()  # RpcBackendOptions.


# 初始化 StubRpcAgent 实例，用于模拟远程过程调用后端
def _stub_init_rpc_backend_handler(store, name, rank, world_size, rpc_backend_options):
    return StubRpcAgent(world_size=world_size)


# 设置一个 Future 对象的结果为指定的值
def set_value(value):
    VALUE_FUTURE.set_result(value)


# 等待 Future 对象的结果并返回
def wait_for_value_future():
    return VALUE_FUTURE.result()


# 设置一个 Future 对象的结果为指定的值，并等待另一个 Future 对象的结果并返回
def set_and_check_done(value):
    VALUE_FUTURE.set_result(value)
    return DONE_FUTURE.result()


# 用于测试通过远程过程调用在 Python 用户定义函数上执行的功能
# 用于测试通过远程过程调用在 Python 用户定义类和方法上执行的功能
TensorClass = namedtuple("TensorClass", ["tensors"])

# 实现 pickle 时用到的类，用于序列化和反序列化 Python 用户定义函数
class MyPickleClass:
    def __init__(self):
        self.t = None

    def __getstate__(self):
        # 序列化 PythonUDF 对象和张量数据
        (pickled_python_udf, tensors) = _internal_rpc_pickler.serialize(
            PythonUDF(my_tensor_function, (torch.ones(2, 2), torch.ones(2, 2)), None)
        )
        return (pickled_python_udf, tensors)

    def __setstate__(self, obj):
        # 反序列化对象并执行相应的 Python 用户定义函数
        python_udf = _internal_rpc_pickler.deserialize(obj[0], obj[1])
        result = python_udf.func(python_udf.args[0], python_udf.args[1])
        self.t = result

    # 设置属性值的方法
    def set(self, val):
        self.t = val


# 用于测试 pickle 时的类，模拟序列化和反序列化的延迟过程
class SlowPickleClass:
    def __init__(self, t):
        self.t = t

    def __getstate__(self):
        # 模拟需要一定时间的序列化过程
        time.sleep(self.t)
        return (self.t, )

    def __setstate__(self, obj):
        # 模拟需要一定时间的反序列化过程
        self.t = obj[0]
        time.sleep(self.t)


# 示例类，包含不同类型的方法用于远程过程调用测试
class MyClass:
    def __init__(self, a, delay=False):
        self.a = a
        # 如果指定延迟，模拟延迟初始化过程
        if delay:
            time.sleep(2)

    # 实例方法
    def my_instance_method(self, b):
        return self.a + b

    # 类方法
    @classmethod
    def my_class_method(cls, d, e):
        return d + e

    # 静态方法
    @staticmethod
    def my_static_method(f):
        return f > 10

    # 实例方法，增加属性值
    def increment_value(self, increment):
        self.a += increment

    # 实例方法，返回属性值
    def get_value(self):
        return self.a

    # 模拟耗时方法，用于测试
    def my_slow_method(self, my_tensor_arg):
        time.sleep(5)
        return torch.add(self.a, my_tensor_arg)


# 在远程引用对象上调用方法的函数
def _call_method_on_rref(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


# 创建包含 MyClass 实例的远程引用对象列表
def get_rref_list(values):
    return [RRef(MyClass(a)) for a in values]


# 将远程引用对象返回本地并与给定值相加
def add_rref_to_value(rref, value):
    return rref.to_here() + value


# 运行嵌套 pickle 测试，返回 pickle 类实例的属性与张量的和
def run_nested_pickle(pickle_cls_instance, tensor):
    return pickle_cls_instance.t + tensor


# 构建稀疏张量对象
def build_sparse_tensor(coalesce=False):
    i = [[0, 1, 1], [2, 0, 2]]
    v = [3, 4, 5]
    tensor = torch.sparse_coo_tensor(i, v, (2, 3))
    if coalesce:
        tensor = tensor.coalesce()
    return tensor


# 构建包含复杂张量的列表和字典对象
def build_complex_tensors():
    a = torch.ones(3, 3)
    b = [a, a]
    c = [b, b]
    d = [a, b]
    e = {a: d}
    return [a, b, c, d, e]


# 测试张量的连续性
def non_cont_test(t_view, t_cont):
    if t_view.is_contiguous():
        raise Exception('t_view is contiguous!')  # noqa: TRY002
    # 检查张量 t_cont 是否是连续的
    if not t_cont.is_contiguous():
        # 如果不连续，抛出异常，提示 t_cont 不是连续的
        raise Exception('t_cont is not contiguous!')  # noqa: TRY002
    
    # 检查张量 t_view 是否与 t_cont 相等
    if not torch.equal(t_view, t_cont):
        # 如果不相等，抛出异常，提示 t_view 不等于 t_cont
        raise Exception('t_view is not equal to t_cont!')  # noqa: TRY002
    
    # 如果以上检查通过，则返回 t_view 张量
    return t_view
# 定义一个简单的函数，将三个参数相加并返回结果
def my_function(a, b, c):
    return a + b + c

# 定义一个函数，对两个张量进行加法操作并返回结果
def my_tensor_function(a, b):
    return a + b

# 定义一个函数，对传入的列表中的张量进行累加操作，并返回累加结果
def my_container_sum(a):
    # 初始结果为列表中第一个张量
    result = a[0]
    # 遍历列表中除第一个元素外的所有张量，将它们累加到结果中
    for tensor in a[1:]:
        result += tensor
    return result

# 定义一个函数，休眠指定秒数后返回张量乘法的结果
def my_sleep_func(seconds=1):
    # 休眠指定秒数
    time.sleep(seconds)
    # 返回张量乘法的结果
    return torch.mul(torch.tensor(1), torch.tensor(1))

# 定义一个复杂的张量处理函数，对列表、张量类和字典中的值进行累加并返回
def my_complex_tensor_function(list_input, tensor_class_input, dict_input):
    # 初始结果为列表中第一个元素
    res = list_input[0]
    # 遍历列表中的所有元素，将它们累加到结果中
    for t in list_input:
        res += t
    # 遍历字典中的所有值，将它们累加到结果中
    for v in dict_input.values():
        res += v
    # 获取张量类中的复杂张量并返回结果
    complex_tensors = tensor_class_input.tensors
    return (res, complex_tensors[0], complex_tensors[1], complex_tensors[2])

# 定义一个函数，对两个远程引用对象执行加法操作并返回结果
def my_rref_function(rref_a, rref_b):
    return rref_a.to_here() + rref_b.to_here()

# 定义一个函数，延迟指定时间后对两个参数执行加法并返回结果
def delayed_add(a, b, seconds=0.05):
    # 延迟指定秒数
    time.sleep(seconds)
    # 返回加法结果
    return a + b

# 定义一个简单的函数，直接返回输入的参数
def identity(a):
    return a

# 定义一个函数，打印一条消息并不返回任何值
def no_result():
    print("do nothing")

# 定义一个函数，根据张量的元素数量判断是否抛出异常或返回增加后的结果
def raise_or_inc(value):
    if value.numel() == 2:
        raise ValueError("Expected error")
    return value + 1

# 定义一个函数，执行嵌套的远程过程调用并返回结果
def nested_rpc(dst):
    return rpc.rpc_sync(dst, torch.add, args=(torch.ones(2, 2), 1))

# 定义一个函数，执行嵌套的稀疏张量的远程过程调用并返回结果
def nested_rpc_sparse(dst):
    return rpc.rpc_sync(
        dst,
        torch.add,
        args=(build_sparse_tensor(), build_sparse_tensor())
    )

# 定义一个函数，执行多层嵌套的异步远程过程调用，递归生成额外的请求
def multi_layer_nested_async_rpc(dst, world_size, ttl):
    # 如果 TTL 大于 0，则继续递归调用下一层异步远程过程
    if ttl > 0:
        current_dst = worker_name(dst)
        next_dst = (dst + 1) % world_size
        rpc.rpc_async(
            current_dst,
            multi_layer_nested_async_rpc,
            args=(next_dst, world_size, ttl - 1),
        )
        return 0

# 定义一个函数，执行嵌套的远程引用对象创建并返回结果
def nested_rref(dst):
    return (
        rpc.remote(dst, torch.add, args=(torch.ones(2, 2), 1)),
        rpc.remote(dst, torch.add, args=(torch.ones(2, 2), 2)),
    )

# 定义一个函数，执行嵌套的稀疏张量的远程引用对象创建并返回结果
def nested_rref_sparse(dst):
    return (
        rpc.remote(
            dst,
            torch.add,
            args=(build_sparse_tensor(), build_sparse_tensor())
        ),
        rpc.remote(
            dst,
            torch.add,
            args=(build_sparse_tensor(), build_sparse_tensor())
        ),
    )

# 定义一个函数，执行嵌套的远程引用对象创建并返回结果
def nested_remote(dst):
    rref = rpc.remote(dst, torch.add, args=(torch.ones(2, 2), 3))
    return rref.to_here()

# 定义一个函数，执行嵌套的稀疏张量的远程引用对象创建并返回结果
def nested_remote_sparse(dst):
    rref = rpc.remote(dst, torch.add, args=(build_sparse_tensor(), build_sparse_tensor()))
    return rref.to_here()

# 定义一个函数，执行远程引用对象的链式调用并返回结果
def rref_forward_chain(dst, world_size, rref, ttl):
    # 如果 TTL 大于 0，则递归调用下一层远程引用对象
    if ttl > 0:
        current_dst = worker_name(dst)
        next_dst = (dst + 1) % world_size
        ret_rref = rpc.remote(
            current_dst, rref_forward_chain, args=(next_dst, world_size, rref, ttl - 1)
        )
        return [ret_rref]
    else:
        return rref.to_here()

# 定义一个函数，执行远程过程调用并返回远程引用对象
def rpc_return_rref(dst):
    return rpc.remote(dst, torch.add, args=(torch.ones(2, 2), 1))

# 定义一个函数，执行轻量级的远程过程调用并返回结果
def light_rpc():
    return 0

# 定义一个函数，执行重量级的远程过程调用并返回结果
def heavy_rpc(tensor):
    # 循环计算从1到99的乘除操作，最终结果对应的tensor值
    for i in range(1, 100):
        # 乘以当前循环变量i
        tensor *= i
        # 除以当前循环变量i加1
        tensor /= i + 1
    # 循环结束后返回整数0
    return 0
# 定义一个函数，执行大量计算操作，但返回值始终为 0
def heavy_rpc_sparse(tensor):
    # 循环执行 1 到 99 的计算操作
    for i in range(1, 100):
        # 将张量乘以当前循环索引
        tensor *= i
        # 将张量除以当前循环索引加 1
        tensor = tensor / (i + 1)
    # 返回固定值 0
    return 0

# 使用 TorchScript 标注修饰的函数，执行类似的计算操作
@torch.jit.script
def heavy_rpc_torchscript(tensor):
    # 循环执行 1 到 99 的计算操作
    for i in range(1, 100):
        # 将张量乘以当前循环索引
        tensor *= i
        # 将张量除以当前循环索引加 1
        tensor /= i + 1
    # 返回固定值 0
    return 0

# 使用 TorchScript 标注修饰的函数，对输入张量执行加法操作，并返回结果
@torch.jit.script
def my_script_func(tensor):
    return torch.add(tensor, tensor)

# 定义一个预期会出错的错误信息字符串
expected_err = "Expected error"

# 定义一个自定义异常类，继承自 Exception，包含布尔值和消息
# 参考 rpc/internal.py 中的注释
class CustomException(Exception):
    def __init__(self, bool, msg):
        self.bool = bool
        super().__init__(msg)

# 抛出 ValueError 异常，包含预期的错误信息
def raise_func():
    raise ValueError(expected_err)

# 抛出自定义异常 CustomException 异常，包含布尔值和消息
def custom_raise_func():
    raise CustomException(True, "foo")

# 使用 TorchScript 标注修饰的函数，抛出 ValueError 异常
@torch.jit.script
def raise_func_script(expected_err: str) -> torch.Tensor:
    raise ValueError(expected_err)

# 定义一个包含换行的错误信息字符串
expected_err_escape = "\nFirst line of error \n next line of error \n last line of error"
# 抛出 ValueError 异常，包含预期的多行错误信息
def raise_func_escape():
    raise ValueError(expected_err_escape)

# 定义一个全局变量，用于存储远程引用对象
global_rref = None

# 设置全局变量 global_rref 的值为指定的远程引用对象
def set_global_rref(rref):
    global global_rref
    global_rref = rref

# 清空全局变量 global_rref 的值
def clear_global_rref():
    global global_rref
    global_rref = None

# 检查给定的远程引用对象 rref 是否已由所有者确认
def check_rref_confirmed(rref):
    return rref.confirmed_by_owner()

# 获取远程引用的调试信息
def get_rref_debug_info():
    return _rref_context_get_debug_info()

# 创建一个并发 Future 对象，调用 RPC 异步方法后，使用回调函数设置其结果
def add_use_future_cb(to, x, y, z):
    out = concurrent.futures.Future()

    def callback(fut):
        out.set_result(fut.wait() + z)

    fut = rpc.rpc_async(to, torch.add, args=(x, y))
    fut.then(callback)
    return out.result()

# 从性能分析远程引用中获取事件处理函数
def get_events_from_profile(profile_rref):
    return profile_rref.local_value().process_global_function_events

# 创建一个 Torch Future 对象，调用 RPC 异步方法后，使用 Lambda 表达式设置其结果
def add_use_future_set_result(to, x, y, z):
    out = torch.futures.Future()
    fut = rpc.rpc_async(to, torch.add, args=(x, y))
    fut.then(lambda fut : out.set_result(fut.wait() + z))
    return out.wait()

# 创建一个 Torch Future 对象，调用嵌套的 RPC 异步方法后，设置其结果
def add_use_future_nested_cb(to, x, y, z):
    out = torch.futures.Future()

    def callback(fut1):
        fut2 = rpc.rpc_async(to, torch.add, args=(fut1.wait(), z))
        fut2.then(lambda fut2 : out.set_result(fut2.wait()))

    fut1 = rpc.rpc_async(to, torch.add, args=(x, y))
    fut1.then(callback)
    return out.wait()

# 不执行任何操作的函数，接收一个 Future 对象作为参数
def fail_on_fut(fut):
    pass

# 使用异步执行修饰的函数，抛出 RuntimeError 异常
@rpc.functions.async_execution
def async_raise_func():
    raise RuntimeError("Expected error")

# 使用异步执行修饰的函数，返回一个张量，而不是抛出异常
@rpc.functions.async_execution
def async_wrong_type():
    return torch.zeros(2, 2)

# 使用异步执行修饰的函数，调用 RPC 异步方法执行加法操作
@rpc.functions.async_execution
def async_add(to, x, y):
    return rpc.rpc_async(to, torch.add, args=(x, y))

# 延迟执行的加法函数，休眠 1 秒后将两个张量相加并返回结果
def slow_add(x, y, device="cpu"):
    time.sleep(1)
    x = x.to(device)
    y = y.to(device)
    return torch.add(x, y).cpu()

# 使用异步执行修饰的函数，调用延迟执行的加法函数
@rpc.functions.async_execution
def slow_async_add(to, x, y, device="cpu"):
    return rpc.rpc_async(to, slow_add, args=(x, y, device))

# 使用异步执行修饰的函数，创建一个 Torch Future 对象，用于异步 RPC 调用
@rpc.functions.async_execution
def async_add_with_future_ctor(to, x, y, z):
    fut = torch.futures.Future()
    # 异步调用 RPC 方法 rpc_async，向目标地址发送 torch.add 方法，参数为 x 和 y
    rpc.rpc_async(to, torch.add, args=(x, y)).then(
        # 使用 Lambda 表达式定义回调函数，当 fut1 完成时设置 fut 的结果为 fut1 的结果加上 z
        lambda fut1: fut.set_result(fut1.wait() + z)
    )
    # 返回 fut 对象
    return fut
# 使用装饰器将函数标记为异步执行
@rpc.functions.async_execution
def async_add_chained(to, x, y, z):
    # 发起异步 RPC 调用，调用 torch.add 函数，并在完成后执行 lambda 表达式，将结果与 z 相加
    return rpc.rpc_async(to, torch.add, args=(x, y)).then(
        lambda fut: fut.wait() + z
    )


# 使用装饰器将函数标记为异步执行
@rpc.functions.async_execution
def async_add_chained_multi(to, x, num, step):
    # 发起异步 RPC 调用，调用 torch.add 函数，通过循环创建链式调用，将结果逐步累加
    fut = rpc.rpc_async(to, torch.add, args=(x, 0))
    for _ in range(num):
        fut = fut.then(lambda fut: fut.wait() + step)
    return fut


# 使用装饰器将函数标记为异步执行
@rpc.functions.async_execution
def async_add_nested(to, x, y, z):
    # 发起异步 RPC 调用，调用 async_add 函数，并在完成后执行 lambda 表达式，将结果与 z 相加
    return rpc.rpc_async(to, async_add, args=(to, x, y)).then(
        lambda fut: fut.wait() + z
    )


# 使用装饰器将函数标记为异步执行
@rpc.functions.async_execution
def async_add_multi_fanout(to, x, num, step):
    # 创建多个异步 RPC 调用并存储在 futs 列表中，使用 torch.futures.collect_all 收集所有结果
    futs = []
    for i in range(num):
        if i == 0:
            futs.append(rpc.rpc_async(to, torch.add, args=(x, step)))
        else:
            futs.append(rpc.rpc_async(to, torch.add, args=(0, step)))

    # 设置状态和结果的 Future 对象
    lock = Lock()
    state = {"cnt": 0, "ret": torch.zeros_like(x)}
    ret_future = torch.futures.Future()

    def inc_and_set(fut):
        # 在锁的保护下递增计数器并累加结果
        with lock:
            state["cnt"] += 1
            state["ret"] += fut.wait()
            # 如果所有异步调用完成，则设置结果并完成 Future 对象
            if state["cnt"] >= len(futs):
                ret_future.set_result(state["ret"])

    # 对每个 Future 对象注册回调函数
    for fut in futs:
        fut.then(inc_and_set)

    # 返回结果的 Future 对象
    return ret_future


# 使用装饰器将函数标记为异步执行
@rpc.functions.async_execution
def async_cuda_sleep_and_set_to_one(t):
    # 在 CUDA 线程上进行异步操作，等待一段时间后将张量填充为 1，并设置结果的 Future 对象返回
    device = t.device
    original_stream = torch.cuda.current_stream(device)
    new_stream = torch.cuda.Stream(device)
    new_stream.wait_stream(original_stream)
    with torch.cuda.stream(new_stream):
        torch.cuda._sleep(int(1000 * get_cycles_per_ms()))
        t.fill_(1)
        fut = Future(devices=[device])
        fut.set_result(t)
        return fut


# 使用装饰器将函数标记为异步执行
@rpc.functions.async_execution
def async_cuda_nested_add(to, x, y, z):
    # 发起异步 RPC 调用，调用 torch.add 函数，并在完成后执行回调函数 cb，处理结果并返回
    def cb(fut):
        torch.cuda._sleep(int(1000 * get_cycles_per_ms()))
        return fut.value() + z

    return rpc.rpc_async(to, torch.add, args=(x, y)).then(cb)


# 自定义 Python 类，包含张量及相关状态，用于测试 Python pickler 提取张量的能力
class TensorWrapper:
    __slots__ = ("tensor", "lock", "event", "thread")

    def __init__(self, t):
        # 初始化对象，包括张量及其它状态
        self.tensor = t
        self.lock = Lock()
        self.event = torch.cuda.Event(enable_timing=True)
        self.thread = threading.Thread()
        self.thread.start()

    def increase(self, v):
        # 增加张量的值，并在锁的保护下执行
        with self.lock:
            self.tensor += v

    def sum(self):
        # 计算张量的总和，并记录 CUDA 事件，返回总和结果
        with self.lock:
            self.event.record()
            return self.tensor.sum()


class AsyncExecutionClass:

    @staticmethod
    @rpc.functions.async_execution
    def static_async_add(to, x, y, z):
        # 发起异步 RPC 调用，调用 torch.add 函数，并在完成后执行 lambda 表达式，将结果与 z 相加
        return rpc.rpc_async(to, torch.add, args=(x, y)).then(
            lambda fut: fut.wait() + z
        )

    @classmethod
    @rpc.functions.async_execution
    def class_async_method(cls, to, x, y, z):
        # 使用类方法进行异步 RPC 调用，调用 torch.add 函数，并在完成后执行 lambda 表达式，将结果与 z 相加
        return rpc.rpc_async(to, torch.add, args=(x, y)).then(
            lambda fut: fut.wait() + z
        )
    # 定义一个类方法，异步执行加法操作，并返回一个 Torch Future 对象
    def class_async_add(cls, to, x, y, z):
        # 创建一个 Torch Future 对象用于存储返回结果
        ret_fut = torch.futures.Future()
        # 发起 RPC 异步调用，调用 torch.add 方法，传入参数 x 和 y
        rpc.rpc_async(to, torch.add, args=(x, y)).then(
            # 设置回调函数，将异步调用返回的 Future 对象的结果与 z 相加，并设置为 ret_fut 的结果
            lambda fut: ret_fut.set_result(fut.wait() + z)
        )
        # 返回 ret_fut 对象，以便调用者可以等待操作完成并获取结果
        return ret_fut

    # 使用装饰器定义一个异步执行函数，执行加法操作
    @rpc.functions.async_execution
    def bound_async_add(self, to, x, y, z):
        # 发起 RPC 异步调用，调用 torch.add 方法，传入参数 x 和 y
        return rpc.rpc_async(to, torch.add, args=(x, y)).then(
            # 设置回调函数，等待异步调用的结果并与 z 相加，作为函数的返回值
            lambda fut: fut.wait() + z
        )
# 返回一个 Torch Future 对象，用于异步操作的结果获取
def return_future():
    return torch.futures.Future()


# FooBackendOptions 类继承自 rpc.RpcBackendOptions 类
# 初始化方法接收一个参数 init_method，需直接调用父类的 __init__ 方法，因为 pybind 的限制
class FooBackendOptions(rpc.RpcBackendOptions):
    def __init__(self, init_method):
        rpc.RpcBackendOptions.__init__(self)
        self.init_method = init_method


# 从 common_utils 中导入 load_tests 函数，用于在 sandcastle 上自动筛选测试以进行分片
# 该行代码用于抑制 flake 警告
load_tests = load_tests


# MyEmbeddingBagModel 类继承自 torch.nn.Module 类
# 初始化方法接收一个参数 sparse，创建一个 EmbeddingBag 层
class MyEmbeddingBagModel(torch.nn.Module):
    def __init__(self, sparse):
        super().__init__()
        self.eb = torch.nn.EmbeddingBag(
            10,
            10,
            sparse=sparse
        )

    # 前向传播方法，接收输入 x，并将其传递给 EmbeddingBag 层
    def forward(self, x):
        return self.eb(x)


# MyParameterServer 类实现一个简单的参数服务器
# 初始化方法接收一个参数 trainers，初始化锁、训练器数量、迭代次数、更新次数、Future 对象列表、总和及梯度
class MyParameterServer:
    def __init__(self, trainers):
        self.lock = Lock()
        self.trainers = trainers
        self.iteration = 0
        self.updates = 0
        self.futures = []
        self.total = None
        self.gradient = None

    # 静态方法，从远程引用 rref 中获取梯度信息
    @staticmethod
    def get_gradient(rref):
        return rref.local_value().gradient

    # 异步方法，计算并返回平均梯度
    @staticmethod
    @rpc.functions.async_execution
    def average(rref, riteration, tensor):
        self = rref.local_value()
        fut = torch.futures.Future()
        with self.lock:
            if riteration > self.iteration:
                self.iteration = riteration
                self.updates = 0
                self.futures.clear()
            self.futures.append(fut)
            if self.total is None:
                self.total = tensor
            else:
                self.total += tensor
            self.updates += 1
            if self.trainers == self.updates:
                self.gradient = self.total / float(self.trainers)
                for fut in self.futures:
                    result = self.total / float(self.trainers)
                    fut.set_result(result)
        return fut


# MyConvNetForMNIST 类继承自 nn.Module 类，实现一个简单的卷积神经网络模型
# 初始化方法接收一个参数 device，创建卷积层、激活函数、池化层、全连接层等
class MyConvNetForMNIST(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(1),
            nn.Linear(4608, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        ).to(device)
        self.device = device

    # 前向传播方法，接收输入 x 和一个可选参数 is_rref，根据 is_rref 决定是否转移到本地内存
    def forward(self, x, is_rref=False):
        x = x.to_here() if is_rref else x
        with torch.cuda.stream(torch.cuda.current_stream(self.device)):
            # 故意在当前 CUDA 流中添加延迟
            torch.cuda._sleep(10 * FIFTY_MIL_CYCLES)
            return self.net(x)

    # 序列化模型状态时调用的方法，返回空字典以避免在所有者上检查模型内容
    def __getstate__(self):
        return {}


# RpcTestCommon 类的声明暂时省略注释，因为它并未完全展示在提供的代码片段中
    # 根据指定的执行模式调用远程过程调用（RPC）函数，并返回结果
    def _run_func_in_mode(self, to, fn, mode, args=None, kwargs=None):
        if mode == RPCExecMode.SYNC:
            return rpc.rpc_sync(to, fn, args=args, kwargs=kwargs)
        elif mode == RPCExecMode.ASYNC:
            # 异步模式下调用RPC函数并等待其完成
            return rpc.rpc_async(to, fn, args=args, kwargs=kwargs).wait()
        elif mode == RPCExecMode.REMOTE:
            # 在远程执行模式下调用RPC函数，并将结果取回本地
            return rpc.remote(to, fn, args=args, kwargs=kwargs).to_here()

    # 在本地运行测试的远程用户定义函数（UDF）
    def _self_py_udf_remote(self, worker_info, x, y, z):
        # 在指定的工作节点上创建远程引用（RRef），调用自定义函数，并验证结果
        rref = rpc.remote(worker_info, my_function, args=(x, y, z))
        self.assertEqual(rref.to_here(), x + y + z)

    # 将本地创建的RRef作为RPC函数的参数远程传输到目标节点，并验证返回结果
    def _self_remote_rref_as_rpc_arg(self, dst, x, y, z):
        self_worker_info = rpc.get_worker_info()
        rref = rpc.remote(self_worker_info, my_function, args=(x, y, z))
        fut = rpc.rpc_async(dst, add_rref_to_value, args=(rref, x))
        ret = rpc.rpc_sync(dst, add_rref_to_value, args=(rref, x + y))
        self.assertEqual(ret, x + y + z + x + y)
        self.assertEqual(fut.wait(), x + y + z + x)

    # 将本地创建的RRef作为远程函数的参数远程传输到目标节点，并验证返回结果
    def _self_remote_rref_as_remote_arg(self, dst, x, y, z):
        self_worker_info = rpc.get_worker_info()
        rref = rpc.remote(self_worker_info, my_function, args=(x, y, z))
        ret_rref = rpc.remote(dst, add_rref_to_value, args=(rref, x))
        self.assertEqual(
            ret_rref.to_here(), x + y + z + x
        )

    # 在单节点环境下，初始化RPC并运行同步、异步及远程RPC函数的测试
    def _world_size_one(self, a, b):
        if self.rank == 0:
            rpc.init_rpc(
                name="me",
                backend=self.rpc_backend,
                rank=0,
                world_size=1,
                rpc_backend_options=self.rpc_backend_options,
            )

            # 定义同步RPC函数，调用自定义张量函数并验证结果
            def _rpc_sync(x, y):
                expect = x * 2
                result = rpc.rpc_sync(
                    "me",
                    my_tensor_function,
                    args=(x, y)
                )
                self.assertEqual(expect, result)

            # 定义异步RPC函数，调用自定义张量函数并验证结果
            def _rpc_async(x, y):
                expect = x * 2
                result = rpc.rpc_async(
                    "me",
                    my_tensor_function,
                    args=(x, y)
                ).wait()
                self.assertEqual(expect, result)

            # 定义远程RPC函数，调用自定义张量函数并验证结果
            def _remote(x, y):
                expect = x * 2
                result = rpc.remote(
                    "me",
                    my_tensor_function,
                    args=(x, y)
                ).to_here()
                self.assertEqual(expect, result)

            # 分别运行同步、异步及远程RPC函数的测试
            _rpc_sync(a, b)
            _rpc_async(a, b)
            _remote(a, b)

            # 关闭RPC连接
            rpc.shutdown()
    def _multi_rpc(self, sparse):
        # 计算目标排名，使其为当前排名加一对世界大小取模
        dst_rank = (self.rank + 1) % self.world_size
        # 循环20次
        for i in range(20):
            # 计算当前循环迭代的数值
            n = i + self.rank + 1
            if sparse:
                # 如果稀疏为真，构建稀疏张量并乘以n
                x = build_sparse_tensor() * n
                y = build_sparse_tensor() * n
            else:
                # 否则创建一个全为1的2x2张量
                x = torch.ones(2, 2)
                y = torch.ones(2, 2)
            # 使用RPC同步调用，将x和y相加，并发送给目标工作节点
            ret = rpc.rpc_sync(
                worker_name(dst_rank),
                torch.add,
                args=(x, y),
            )
            # 断言返回值等于x乘以2
            self.assertEqual(ret, x * 2)

    def _run_uneven_workload(self, f, x, num_repeat=30):
        # 如果当前进程的排名为0
        if self.rank == 0:
            # 断言世界大小至少为3
            self.assertTrue(self.world_size >= 3)

            # 第一阶段：只有worker1有工作负载
            dst = "worker1"
            futs = []
            # 多次重复执行RPC异步调用
            for _ in range(num_repeat):
                fut = rpc.rpc_async(dst, f, args=(x,))
                futs.append(fut)

            # 等待所有异步调用的完成
            for fut in torch.futures.collect_all(futs).wait():
                # 断言每个异步调用的结果为0
                self.assertEqual(fut.wait(), 0)

            # 第二阶段：只有worker2有工作负载
            # 如果join没有正确实现，worker2现在应该已经关闭
            dst = "worker2"
            futs = []
            # 再次重复执行RPC异步调用
            for _ in range(num_repeat):
                fut = rpc.rpc_async(dst, f, args=(x,))
                futs.append(fut)

            # 等待所有异步调用的完成
            for val in torch.futures.wait_all(futs):
                # 断言每个异步调用的结果为0
                self.assertEqual(val, 0)

    def _wait_all_workers(self, f, x):
        # 初始化进程组并设定初始方法、排名和世界大小
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        # 初始化RPC，设置当前进程的名称、后端、排名、世界大小和选项
        rpc.init_rpc(
            name="worker%d" % self.rank,
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        # 运行不均匀的工作负载函数
        self._run_uneven_workload(f, x)

        # worker0在等待RPC响应后调用此函数
        # worker1/2立即调用此函数并在其后执行一些工作
        # worker3立即调用此函数并无后续工作
        rpc.api._wait_all_workers()

        # 在继续关闭之前等待，以确保worker0的RPC传递到其他worker
        dist.barrier()
        rpc.shutdown(graceful=False)
    # 初始化分布式训练环境，设置每个进程的初始方法、排名和总进程数
    initialize_pg(self.file_init_method, self.rank, self.world_size)
    # 初始化当前进程的 RPC，指定名称、后端、排名、总进程数和后端选项
    rpc.init_rpc(
        name="worker%d" % self.rank,
        backend=self.rpc_backend,
        rank=self.rank,
        world_size=self.world_size,
        rpc_backend_options=self.rpc_backend_options,
    )

    # 运行不均匀的工作负载函数
    self._run_uneven_workload(f, x)

    # worker0 在等待 RPC 响应后调用此方法
    # worker1/2 在调用后立即执行后续工作
    # worker3 在调用后立即执行且无后续工作
    rpc.api._wait_all_workers()
    rpc.api._wait_all_workers()

    # 在继续关闭之前等待，确保 worker0 的 RPC 调用传递到其他 worker
    dist.barrier()
    # 关闭 RPC，强制关闭，不等待其他进程
    rpc.shutdown(graceful=False)
    # 使用给定的参数创建一个远程过程调用（RPC）远程引用对象，并进行加法操作
    def _py_rref_args(self, a, b, x, y, expected):
        # 计算当前进程的排名
        n = self.rank + 1
        # 确定目标进程的排名
        dst_rank = n % self.world_size
        # 创建远程引用对象rref_a，调用torch.add函数对a和b进行加法操作
        rref_a = rpc.remote(
            worker_name(dst_rank), torch.add, args=(a, b)
        )
        # 创建远程引用对象rref_b，调用torch.add函数对x和y进行加法操作
        rref_b = rpc.remote(
            worker_name(dst_rank), torch.add, args=(x, y)
        )
        # 创建远程引用对象rref_c，调用自定义函数my_rref_function处理rref_a和rref_b
        rref_c = rpc.remote(
            worker_name(dst_rank), my_rref_function, args=(rref_a, rref_b)
        )
        # 断言rref_c对象的本地结果等于预期结果
        self.assertEqual(rref_c.to_here(), expected)

    # 使用给定的参数创建一个用户共享的远程过程调用（RPC）远程引用对象，并进行处理
    def _py_rref_args_user_share(self, a, b, c, x, y, z, expected):
        # 计算当前进程的排名
        n = self.rank + 1
        # 确定数据所有者的进程排名
        owner_rank = n % self.world_size
        # 确定用户所在进程的排名
        user_rank = (n + 1) % self.world_size
        # 创建远程引用对象rref_a，调用my_function处理a, b, c
        rref_a = rpc.remote(
            worker_name(owner_rank), my_function, args=(a, b, c)
        )
        # 创建远程引用对象rref_b，调用my_function处理x, y, z
        rref_b = rpc.remote(
            worker_name(owner_rank), my_function, args=(x, y, z)
        )
        # 创建远程引用对象rref_c，调用my_rref_function处理rref_a和rref_b
        rref_c = rpc.remote(
            worker_name(user_rank), my_rref_function, args=(rref_a, rref_b)
        )
        # 断言rref_c对象的本地结果等于预期结果
        self.assertEqual(rref_c.to_here(), expected)

    # 使用给定的参数创建一个远程过程调用（RPC）远程引用对象，并进行处理
    def _py_rpc_rref_args(self, a, b, c, x, y, z, expected):
        # 计算当前进程的排名
        n = self.rank + 1
        # 确定目标进程的排名
        dst_rank = n % self.world_size
        # 创建远程引用对象rref_a，调用my_function处理a, b, c
        rref_a = rpc.remote(
            worker_name(dst_rank), my_function, args=(a, b, c)
        )
        # 创建远程引用对象rref_b，调用my_function处理x, y, z
        rref_b = rpc.remote(
            worker_name(dst_rank), my_function, args=(x, y, z)
        )
        # 同步调用远程过程，将rref_a和rref_b作为参数传递给my_rref_function处理
        c = rpc.rpc_sync(
            worker_name(dst_rank), my_rref_function, args=(rref_a, rref_b)
        )
        # 断言c的值等于预期结果
        self.assertEqual(c, expected)

    # 使用给定的函数和参数创建一个嵌套的远程过程调用（RPC）远程引用对象，并进行处理
    def _nested_remote(self, f, expected):
        # 计算当前进程的排名
        n = self.rank + 1
        # 确定目标进程的排名
        dst_rank1 = n % self.world_size
        # 确定第二个目标进程的排名
        dst_rank2 = (n + 1) % self.world_size
        # 创建远程引用对象rref，调用函数f并将worker_name(dst_rank2)作为参数传递
        rref = rpc.remote(
            worker_name(dst_rank1),
            f,
            args=(worker_name(dst_rank2),),
        )
        # 断言rref对象的本地结果等于预期结果
        self.assertEqual(rref.to_here(), expected)

    # 使用给定的函数和参数创建一个嵌套的远程过程调用（RPC）远程引用对象，并进行处理
    def _nested_rref(self, f, expected1, expected2):
        # 计算当前进程的排名
        n = self.rank + 1
        # 确定第一个目标进程的排名
        dst_rank1 = n % self.world_size
        # 确定第二个目标进程的排名
        dst_rank2 = (n + 1) % self.world_size
        # 创建远程引用对象rref_of_rrefs，调用函数f并将worker_name(dst_rank2)作为参数传递
        rref_of_rrefs = rpc.remote(
            worker_name(dst_rank1),
            f,
            args=(worker_name(dst_rank2),),
        )

        # 获取rref_of_rrefs对象的本地结果
        rrefs = rref_of_rrefs.to_here()

        # 断言rrefs列表的长度为2
        self.assertEqual(len(rrefs), 2)
        # 断言rrefs列表中第一个元素的本地结果等于expected1
        self.assertEqual(rrefs[0].to_here(), expected1)
        # 断言rrefs列表中第二个元素的本地结果等于expected2
        self.assertEqual(rrefs[1].to_here(), expected2)
    # 定义一个方法，用于测试嵌套远程引用的压力情况
    def _nested_rref_stress(self, f, expected1, expected2):
        # 计算当前进程数加一的余数，作为第一个目标进程的索引
        n = self.rank + 1
        dst_rank1 = n % self.world_size
        # 计算当前进程数加二的余数，作为第二个目标进程的索引
        dst_rank2 = (n + 1) % self.world_size
        # 初始化一个空列表，用于存储所有的远程引用对象
        all_rrefs = []
        # 执行20次迭代
        for _ in range(20):
            # 向第一个目标进程异步发送远程过程调用请求
            all_rrefs.append(
                rpc.remote(
                    worker_name(dst_rank1),
                    f,
                    args=(worker_name(dst_rank2),),
                )
            )

        # 遍历所有的远程引用对象
        for i in range(20):
            # 获取第i个远程引用对象
            rref_of_rrefs = all_rrefs[i]
            # 将远程引用对象拉取到本地，获取其包含的两个远程引用对象
            rrefs = rref_of_rrefs.to_here()
            # 断言获取到的远程引用对象数目为2
            self.assertEqual(len(rrefs), 2)
            # 断言第一个远程引用对象的值与期望值expected1相等
            self.assertEqual(rrefs[0].to_here(), expected1)
            # 断言第二个远程引用对象的值与期望值expected2相等
            self.assertEqual(rrefs[1].to_here(), expected2)

    # 定义一个方法，用于训练模型的远程函数
    def _trainer_func(self, rref, sparse):
        # 初始化一个稀疏或非稀疏的嵌入袋模型
        m = MyEmbeddingBagModel(sparse=sparse)
        # 定义均方误差损失函数
        loss_fn = nn.MSELoss()
        # 执行10次迭代
        for i in range(10):
            # 随机生成大小为(10, 10)的长整型张量，输入到模型中得到输出
            outputs = m(torch.rand(10, 10).long())
            # 计算模型输出与随机(10, 10)张量之间的均方误差，反向传播
            loss_fn(outputs, torch.rand(10, 10)).backward()
            # 获取第一个参数的梯度
            gradient = next(iter(m.parameters())).grad
            # 发送异步RPC调用请求，计算并返回梯度的平均值
            fut = rref.rpc_async().average(rref, i, gradient)
            # 等待异步操作完成，获取梯度
            gradient = fut.wait()
            # 如果梯度是稀疏的，则转换成密集型的双精度张量
            if gradient.is_sparse:
                gradient = gradient.to_dense().double()
            # 使用RPC同步调用请求，获取参数服务器上的梯度
            ps_gradient = rref.rpc_sync().get_gradient(rref)
            # 如果参数服务器上的梯度是稀疏的，则转换成密集型的双精度张量
            if ps_gradient.is_sparse:
                ps_gradient = ps_gradient.to_dense().double()
            # 断言本地计算得到的梯度与参数服务器上的梯度相等
            self.assertTrue(torch.equal(gradient, ps_gradient))

    # 定义一个方法，用于模拟我的参数服务器行为
    def _my_parameter_server(self, sparse):
        # 创建一个参数服务器的远程引用
        ps_rref = RRef(MyParameterServer(self.world_size - 1))
        # 初始化一个空列表，用于存储所有的异步RPC调用请求
        futures = []
        # 遍历所有的目标进程索引，发送训练函数的RPC异步调用请求
        for index in range(1, self.world_size):
            futures.append(
                rpc.rpc_async(
                    worker_name((self.rank + index) % self.world_size),
                    self._trainer_func,
                    args=(
                        ps_rref,
                        sparse
                    ),
                )
            )
        # 等待所有异步操作请求完成
        torch.futures.wait_all(futures)
    # 定义一个测试方法，用于测试 CUDA 下的异步数据处理与同步操作
    def _test_cuda_future_extraction(self, wrapper, unwrapper, sparse_tensor):
        # 我们通过在一个 CUDA 流中添加数据以获取预期值，并在另一个流中读取数据，来检查正确的 CUDA 流同步。
        # 创建一个 Future 对象，指定设备为 "cuda:0"
        future = Future(devices=["cuda:0"])
        
        # 在 CUDA 设备 "cuda:0" 上创建一个流
        with torch.cuda.device("cuda:0"):
            stream = torch.cuda.Stream()
            another_stream = torch.cuda.Stream()
            
            # 在当前流（stream）中执行以下操作
            with torch.cuda.stream(stream):
                if sparse_tensor:
                    # 如果是稀疏张量，则构建一个稀疏张量，并将其移动到 "cuda:0" 设备
                    tensor = build_sparse_tensor().to("cuda:0")
                    add_tensor = build_sparse_tensor().to("cuda:0")
                    # 计算预期的张量，并进行稀疏化处理
                    expected_tensor = (tensor + add_tensor).coalesce()
                else:
                    # 如果不是稀疏张量，则创建一个在 "cuda:0" 设备上全零的张量
                    tensor = torch.zeros((100,), device="cuda:0")
                    add_tensor = torch.ones((100,), device="cuda:0")
                    # 计算预期的张量
                    expected_tensor = tensor + add_tensor
                
                # 在当前 CUDA 流中执行一个休眠操作，时间为当前设备时钟周期数的 1000 倍
                torch.cuda._sleep(int(1000 * get_cycles_per_ms()))
                
                # 将 add_tensor 加到 tensor 上
                tensor += add_tensor
                
                # 如果是稀疏张量，则将 tensor 稀疏化
                if sparse_tensor:
                    tensor = tensor.coalesce()
                
                # 将处理后的 tensor 作为结果设置到 Future 对象中
                future.set_result(wrapper(tensor))
            
            # 在另一个 CUDA 流中执行以下操作
            with torch.cuda.stream(another_stream):
                # 从 Future 对象中获取结果，并进行解包操作
                tensor = unwrapper(future.wait())
                
                # 如果是稀疏张量，则进行以下断言
                if sparse_tensor:
                    # 断言 tensor 的 indices 和 values 与预期的 indices 和 values 相等
                    self.assertTrue(torch.eq(tensor.indices(), expected_tensor.indices()).all().item())
                    self.assertTrue(torch.eq(tensor.values(), expected_tensor.values()).all().item())
                    # 断言 tensor 的尺寸与预期的尺寸相等
                    self.assertEqual(tensor.size(), expected_tensor.size())
                else:
                    # 如果不是稀疏张量，则断言 tensor 与预期的张量相等
                    self.assertTrue(torch.eq(tensor, expected_tensor).all().item())
# 定义一个名为 RpcTest 的测试类，继承自 RpcAgentTestFixture 和 RpcTestCommon
class RpcTest(RpcAgentTestFixture, RpcTestCommon):

    # 使用 dist_init 装饰器初始化测试方法 test_worker_id
    @dist_init
    def test_worker_id(self):
        # 计算当前进程的等级 n
        n = self.rank + 1
        # 计算对等进程的等级
        peer_rank = n % self.world_size
        # 获取当前进程的 worker 信息
        self_worker_info = rpc.get_worker_info()
        # 获取对等进程的 worker 信息
        peer_worker_info = rpc.get_worker_info(worker_name(peer_rank))

        # 断言当前进程的名称与其等级对应的名称相等
        self.assertEqual(self_worker_info.name, worker_name(self.rank))
        # 断言对等进程的名称与对等进程等级对应的名称相等
        self.assertEqual(peer_worker_info.name, worker_name(peer_rank))

        # 使用断言检查运行时错误，并包含特定的错误信息
        with self.assertRaisesRegex(RuntimeError, "could not find destination"):
            # 尝试获取一个未知 worker 的 worker 信息
            unknown_worker_id = rpc.get_worker_info("WorkerUnknown")

    # 使用 dist_init 装饰器初始化测试方法 test_get_worker_infos
    @dist_init
    def test_get_worker_infos(self):
        # 获取当前 RPC 代理的所有 worker 信息
        worker_infos = rpc.api._get_current_rpc_agent().get_worker_infos()

        # 从 worker_infos 中提取所有 worker 的名称到集合 worker_names
        worker_names = {worker_info.name for worker_info in worker_infos}
        # 创建期望的 worker 名称集合，包含所有等级的 worker 名称
        expected_worker_names = {
            worker_name(rank) for rank in range(self.world_size)
        }
        # 断言实际的 worker 名称集合与期望的 worker 名称集合相等
        self.assertEqual(worker_names, expected_worker_names)

        # 从 worker_infos 中提取所有 worker 的 ID 到集合 worker_ids
        worker_ids = {worker_info.id for worker_info in worker_infos}
        # 创建期望的 worker ID 集合，包含所有等级的 worker ID
        expected_worker_ids = set(range(self.world_size))
        # 断言实际的 worker ID 集合与期望的 worker ID 集合相等
        self.assertEqual(worker_ids, expected_worker_ids)

    # 使用 dist_init 装饰器初始化测试方法 test_self_add
    @dist_init
    def test_self_add(self):
        # 获取当前进程的 worker 信息
        self_worker_info = rpc.get_worker_info()
        # 获取当前进程的名称
        self_worker_name = worker_name(self.rank)
        # 使用异步 RPC 发送操作，计算一个矩阵的加法
        fut = rpc.rpc_async(self_worker_info, torch.add, args=(torch.ones(2, 2), 1))
        # 使用同步 RPC 发送操作，计算一个矩阵的加法
        ret = rpc.rpc_sync(self_worker_info, torch.add, args=(torch.ones(2, 2), 1))
        # 使用断言检查异步操作的结果是否符合预期
        self.assertEqual(fut.wait(), torch.ones(2, 2) + 1)
        # 使用断言检查同步操作的结果是否符合预期
        self.assertEqual(ret, torch.ones(2, 2) + 1)

    # 使用 dist_init 装饰器初始化测试方法 test_send_to_rank
    @dist_init
    def test_send_to_rank(self):
        # 计算目标等级，确保环形发送到下一个进程
        dst_rank = (self.rank + 1) % self.world_size

        # 测试稠密张量的发送
        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            # 在不同的执行模式下，调用 _run_func_in_mode 方法，测试张量的加法操作
            ret = self._run_func_in_mode(dst_rank, torch.add, exec_mode, args=(torch.ones(2, 2), 1))
            # 使用断言检查返回值是否符合预期
            self.assertEqual(ret, torch.ones(2, 2) + 1)

        # 测试无效的等级
        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            # 使用断言检查是否抛出 RuntimeError 异常
            with self.assertRaises(RuntimeError):
                # 尝试在超出 world_size 的等级上执行 _run_func_in_mode 方法
                self._run_func_in_mode(self.world_size + 1, torch.add, exec_mode, args=(torch.ones(2, 2), 1))

        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            # 使用断言检查是否抛出 RuntimeError 异常
            with self.assertRaises(RuntimeError):
                # 尝试在负数等级上执行 _run_func_in_mode 方法
                self._run_func_in_mode(-1, torch.add, exec_mode, args=(torch.ones(2, 2), 1))

        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            # 使用断言检查是否抛出 ValueError 异常
            with self.assertRaises(ValueError):
                # 尝试在非整数等级上执行 _run_func_in_mode 方法
                self._run_func_in_mode(dst_rank + 0.5, torch.add, exec_mode, args=(torch.ones(2, 2), 1))

        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            # 使用断言检查是否抛出 ValueError 异常
            with self.assertRaises(ValueError):
                # 尝试在非整数等级上执行 _run_func_in_mode 方法
                self._run_func_in_mode(dst_rank - 0.5, torch.add, exec_mode, args=(torch.ones(2, 2), 1))
    def test_self_py_udf_remote(self):
        # 调用 _self_py_udf_remote 方法，传入当前 worker 的信息、一个2x2的全1张量，以及参数1和3
        self._self_py_udf_remote(
            rpc.get_worker_info(),
            torch.ones(2, 2),
            1,
            3
        )

    @dist_init
    def test_self_remote_rref_as_rpc_arg(self):
        # 获取下一个 worker 的名称，然后调用 _self_remote_rref_as_rpc_arg 方法，
        # 传入目标 worker 的名称、一个2x2的全1张量，以及参数1和3
        dst = worker_name((self.rank + 1) % self.world_size)
        self._self_remote_rref_as_rpc_arg(
            dst,
            torch.ones(2, 2),
            1,
            3
        )

    @dist_init
    def test_self_remote_rref_as_self_rpc_arg(self):
        # 调用 _self_remote_rref_as_rpc_arg 方法，传入当前 worker 的信息、
        # 一个2x2的全1张量，以及参数1和3
        self._self_remote_rref_as_rpc_arg(
            rpc.get_worker_info(),
            torch.ones(2, 2),
            1,
            3
        )

    @dist_init
    def test_self_remote_rref_as_remote_arg(self):
        # 获取下一个 worker 的名称，然后调用 _self_remote_rref_as_remote_arg 方法，
        # 传入目标 worker 的名称、一个2x2的全1张量，以及参数1和3
        dst = worker_name((self.rank + 1) % self.world_size)
        self._self_remote_rref_as_remote_arg(
            dst,
            torch.ones(2, 2),
            1,
            3
        )

    @dist_init
    def test_self_remote_rref_as_self_remote_arg(self):
        # 调用 _self_remote_rref_as_remote_arg 方法，传入当前 worker 的信息、
        # 一个2x2的全1张量，以及参数1和3
        self._self_remote_rref_as_remote_arg(
            rpc.get_worker_info(),
            torch.ones(2, 2),
            1,
            3
        )

    @dist_init
    def test_rref_proxy_non_exist(self):
        # 获取下一个 worker 的名称，然后远程调用 my_function 方法，传入一个2x2的全1张量，
        # 以及参数1和3，返回一个远程引用对象 rref
        dst = worker_name((self.rank + 1) % self.world_size)
        rref = rpc.remote(dst, my_function, args=(torch.ones(2, 2), 1, 3))
        
        # 定义错误消息字符串
        msg = "has no attribute 'non_exist'"
        
        # 测试 rref.rpc_sync() 调用不存在的方法 non_exist 是否抛出 AttributeError 异常
        with self.assertRaisesRegex(AttributeError, msg):
            rref.rpc_sync().non_exist()

        # 测试 rref.rpc_async().non_exist().wait() 调用不存在的方法 non_exist 是否抛出 AttributeError 异常
        with self.assertRaisesRegex(AttributeError, msg):
            rref.rpc_async().non_exist().wait()

        # 测试 rref.remote().non_exist() 调用不存在的方法 non_exist 是否抛出 AttributeError 异常
        with self.assertRaisesRegex(AttributeError, msg):
            rref.remote().non_exist()

    def _test_rref_proxy_tensor(self, dst):
        # 远程调用 my_function 方法，传入一个2x2的全1张量，以及参数1和3，返回一个远程引用对象 rref
        rref = rpc.remote(dst, my_function, args=(torch.ones(2, 2), 1, 3))

        # 期望的张量结果是全1的2x2张量加1加3
        expected = torch.ones(2, 2) + 1 + 3
        
        # 使用 rpc_sync() 方法同步调用远程引用 rref 上的方法 size()，并断言结果与 expected 的 size() 相等
        self.assertEqual(expected.size(), rref.rpc_sync().size())
        
        # 使用 rpc_async() 方法异步调用远程引用 rref 上的方法 add(1)，并等待返回结果，断言结果与 expected + 1 相等
        self.assertEqual(expected + 1, rref.rpc_async().add(1).wait())
        
        # 使用 remote() 方法调用远程引用 rref 上的方法 view(1, 4)，并将结果转移到本地进行断言
        self.assertEqual(expected.view(1, 4), rref.remote().view(1, 4).to_here())

    @dist_init
    def test_rref_proxy_tensor(self):
        # 调用 _test_rref_proxy_tensor 方法，传入下一个 worker 的名称作为目标
        self._test_rref_proxy_tensor(worker_name((self.rank + 1) % self.world_size))

    @dist_init
    def test_rref_proxy_tensor_self(self):
        # 调用 _test_rref_proxy_tensor 方法，传入当前 worker 的信息作为目标
        self._test_rref_proxy_tensor(rpc.get_worker_info())

    @dist_init
    # 定义测试函数 test_rref_proxy_reuse，用于测试远程引用代理的重复使用情况
    def test_rref_proxy_reuse(self):
        # 创建远程引用对象 rref，指向下一个工作节点（循环环境中的下一个）
        rref = rpc.remote(
            worker_name((self.rank + 1) % self.world_size),
            my_function,
            args=(torch.ones(2, 2), 1, 3)
        )
        # 生成预期的张量，为全1的2x2张量加上1和3
        expected = torch.ones(2, 2) + 1 + 3

        # 创建同步代理 RPC 对象，用于同步调用远程方法
        proxy_rpc_sync = rref.rpc_sync()
        # 创建异步代理 RPC 对象，用于异步调用远程方法
        proxy_rpc_async = rref.rpc_async()
        # 创建远程代理对象，用于将远程方法调用发送到远程工作节点
        proxy_remote = rref.remote()

        # 断言：验证同步代理 RPC 对象的尺寸与预期张量尺寸相同
        self.assertEqual(expected.size(), proxy_rpc_sync.size())
        # 断言：验证同步代理 RPC 对象调用 add(1) 后的结果与预期张量加1相同
        self.assertEqual(expected + 1, proxy_rpc_sync.add(1))
        # 断言：验证同步代理 RPC 对象调用 view(1, 4) 后的结果与预期张量视图相同
        self.assertEqual(expected.view(1, 4), proxy_rpc_sync.view(1, 4))

        # 断言：验证异步代理 RPC 对象的尺寸与预期张量尺寸相同
        self.assertEqual(expected.size(), proxy_rpc_async.size().wait())
        # 断言：验证异步代理 RPC 对象调用 add(3).wait() 后的结果与预期张量加3相同
        self.assertEqual(expected + 3, proxy_rpc_async.add(3).wait())
        # 断言：验证异步代理 RPC 对象调用 view(4, 1).wait() 后的结果与预期张量视图相同
        self.assertEqual(expected.view(4, 1), proxy_rpc_async.view(4, 1).wait())

        # 断言：验证远程代理对象的尺寸与预期张量尺寸相同
        self.assertEqual(expected.size(), proxy_remote.size().to_here())
        # 断言：验证远程代理对象调用 add(5).to_here() 后的结果与预期张量加5相同
        self.assertEqual(expected + 5, proxy_remote.add(5).to_here())
        # 断言：验证远程代理对象调用 view(-1).to_here() 后的结果与预期张量视图相同
        self.assertEqual(expected.view(-1), proxy_remote.view(-1).to_here())
    # 定义一个测试方法，用于验证远程 RPC 代理类的行为
    def _test_rref_proxy_class(self, dst):
        # 在目标节点上创建一个远程对象引用 (RRef)，表示 MyClass 类的实例，参数为 7
        rref = rpc.remote(dst, MyClass, args=(7,))
        # 创建预期的 MyClass 实例，值为 7
        expected = MyClass(7)
        
        # 使用 RPC 同步调用方式，验证远程对象的 get_value 方法返回值与预期值相同
        self.assertEqual(expected.get_value(), rref.rpc_sync().get_value())
        # 使用 RPC 异步调用方式，验证远程对象的 get_value 方法返回值与预期值相同
        self.assertEqual(expected.get_value(), rref.rpc_async().get_value().wait())
        # 使用远程调用方式，验证远程对象的 get_value 方法返回值与预期值相同
        self.assertEqual(expected.get_value(), rref.remote().get_value().to_here())

        # 预期对象的值增加 3
        expected.increment_value(3)
        # 使用 RPC 同步调用方式，调用远程对象的 increment_value 方法增加值为 1
        self.assertEqual(None, rref.rpc_sync().increment_value(1))
        # 使用 RPC 异步调用方式，调用远程对象的 increment_value 方法增加值为 1
        self.assertEqual(None, rref.rpc_async().increment_value(1).wait())
        # 使用远程调用方式，调用远程对象的 increment_value 方法增加值为 1
        self.assertEqual(None, rref.remote().increment_value(1).to_here())

        # 再次验证远程对象的 get_value 方法返回值与预期值相同
        self.assertEqual(expected.get_value(), rref.rpc_sync().get_value())
        self.assertEqual(expected.get_value(), rref.rpc_async().get_value().wait())
        self.assertEqual(expected.get_value(), rref.remote().get_value().to_here())

        # 使用 RPC 同步调用方式，验证远程对象的 my_instance_method 方法返回值与预期值相同
        self.assertEqual(
            expected.my_instance_method(2),
            rref.rpc_sync().my_instance_method(2)
        )
        # 使用 RPC 异步调用方式，验证远程对象的 my_instance_method 方法返回值与预期值相同
        self.assertEqual(
            expected.my_instance_method(3),
            rref.rpc_async().my_instance_method(3).wait()
        )
        # 使用远程调用方式，验证远程对象的 my_instance_method 方法返回值与预期值相同
        self.assertEqual(
            expected.my_instance_method(4),
            rref.remote().my_instance_method(4).to_here()
        )

        # 使用 RPC 同步调用方式，验证远程对象的 my_static_method 方法返回值与预期值相同
        self.assertEqual(
            expected.my_static_method(9),
            rref.rpc_sync().my_static_method(9)
        )
        # 使用 RPC 异步调用方式，验证远程对象的 my_static_method 方法返回值与预期值相同
        self.assertEqual(
            expected.my_static_method(10),
            rref.rpc_async().my_static_method(10).wait()
        )
        # 使用远程调用方式，验证远程对象的 my_static_method 方法返回值与预期值相同
        self.assertEqual(
            expected.my_static_method(11),
            rref.remote().my_static_method(11).to_here()
        )

        # 使用 RPC 同步调用方式，验证远程对象的 my_class_method 方法返回值与预期值相同
        self.assertEqual(
            expected.my_class_method(2, torch.zeros(2, 2)),
            rref.rpc_sync().my_class_method(2, torch.zeros(2, 2))
        )
        # 使用 RPC 异步调用方式，验证远程对象的 my_class_method 方法返回值与预期值相同
        self.assertEqual(
            expected.my_class_method(2, torch.ones(3, 3)),
            rref.rpc_async().my_class_method(2, torch.ones(3, 3)).wait()
        )
        # 使用远程调用方式，验证远程对象的 my_class_method 方法返回值与预期值相同
        self.assertEqual(
            expected.my_class_method(2, torch.ones(4, 4)),
            rref.remote().my_class_method(2, torch.ones(4, 4)).to_here()
        )

    # 使用分布式初始化装饰器，测试远程 RPC 代理类的行为
    @dist_init
    def test_rref_proxy_class(self):
        # 调用 _test_rref_proxy_class 方法，传入目标工作节点名称，测试 RPC 代理类行为
        self._test_rref_proxy_class(worker_name((self.rank + 1) % self.world_size))

    # 使用分布式初始化装饰器，测试在当前节点上调用远程 RPC 代理类的行为
    @dist_init
    def test_rref_proxy_class_self(self):
        # 调用 _test_rref_proxy_class 方法，传入当前节点的 RPC 代理信息，测试 RPC 代理类行为
        self._test_rref_proxy_class(rpc.get_worker_info())

    # 使用 mock.patch 装饰器，模拟分布式自动求导模块的初始化过程
    # 使用 mock.patch 装饰器，模拟设置和启动 RPC 代理的过程
    # 使用分布式初始化装饰器，测试注册 RPC 后端并设置和启动 RPC 后端的过程
    def test_register_rpc_backend_and_set_and_start_rpc_backend(
        self, mock_rpc_agent, mock_dist_autograd_init
    ):
        # 设置后端名称为 "stub_backend"
        backend_name = "stub_backend"

        # 注册一个名为 "stub_backend" 的 RPC 后端
        backend = rpc.backend_registry.register_backend(
            backend_name,
            _stub_construct_rpc_backend_options_handler,
            _stub_init_rpc_backend_handler,
        )

        # 使用断言检查是否会抛出 RuntimeError 异常，异常信息开头为 "RPC backend .+: already registered"
        with self.assertRaisesRegex(
            RuntimeError, "^RPC backend .+: already registered$"
        ):
            # 再次尝试注册名为 "stub_backend" 的 RPC 后端，预期会抛出异常
            backend = rpc.backend_registry.register_backend(
                backend_name,
                _stub_construct_rpc_backend_options_handler,
                _stub_init_rpc_backend_handler,
            )

        # 初始化 RPC，使用指定的名称 "worker1"、后端、排名、世界大小和 RPC 后端选项
        rpc.init_rpc(
            name="worker1",
            backend=backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

    @dist_init(setup_rpc=False)
    def test_duplicate_name(self):
        # 使用断言检查是否会抛出 RuntimeError 异常，异常信息包含 "is not unique"
        with self.assertRaisesRegex(RuntimeError, "is not unique"):
            # 获取分布式 rendezvous 的存储信息，并初始化 RPC 后端，名称为 "duplicate_name"
            store, _, _ = next(
                torch.distributed.rendezvous(
                    self.init_method, rank=self.rank, world_size=self.world_size
                )
            )
            rpc._init_rpc_backend(
                backend=self.rpc_backend,
                store=store,
                name="duplicate_name",
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )

    @dist_init(setup_rpc=False)
    def test_duplicate_name_2(self):
        # 使用断言检查是否会抛出 RuntimeError 异常，异常信息包含 "is not unique"
        with self.assertRaisesRegex(RuntimeError, "is not unique"):
            # 初始化 RPC，使用计算出的工作节点名称、后端、排名、世界大小和 RPC 后端选项
            rpc.init_rpc(
                name=worker_name(self.rank % (self.world_size - 1)),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )

    @dist_init(setup_rpc=False)
    def test_reinit(self):
        # 初始化 RPC
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        # 使用指定方法初始化进程组
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        # 等待所有初始化完成
        dist.barrier()

        # 根据环境变量和进程等级设置预期的重新初始化错误信息
        if os.environ.get("RPC_INIT_WITH_TCP", None) == "1" and self.rank == 0:
            expected_reinit_err = "Address already in use"
        else:
            expected_reinit_err = "is already initialized"

        # 断言在重新初始化 RPC 时抛出预期的错误
        with self.assertRaisesRegex(RuntimeError, expected_reinit_err):
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )
        # 关闭 RPC
        rpc.shutdown()

    @dist_init(setup_rpc=False)
    def test_pg_init_no_rpc_init(self):
        # 初始化进程组，但不设置 RPC
        dist.init_process_group(
            backend='gloo',
            init_method=self.file_init_method,
            rank=self.rank,
            world_size=self.world_size)

        # 定义一个简单的神经网络模型
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(3, 4)

            def forward(self, x):
                return self.lin(x)

        # 创建模型实例，并使用分布式数据并行处理器包装
        model = MyModel()
        model.train()
        model = torch.nn.parallel.DistributedDataParallel(model)

        # 断言当前 RPC 代理未设置的错误
        with self.assertRaisesRegex(RuntimeError, 'Current RPC agent is not set! Did you initialize the RPC framework'):
            params = []
            # 将模型参数转换为远程引用
            for param in model.parameters():
                params.append(RRef(param))

    def test_world_size_one(self):
        # 测试当世界大小为1时的函数 _world_size_one
        self._world_size_one(
            torch.ones(2, 2),
            torch.ones(2, 2)
        )

    @dist_init(setup_rpc=False)
    def test_invalid_names(self):
        # 测试无效的工作器名

        worker_id = 0
        # 断言工作器名必须匹配特定格式的错误
        with self.assertRaisesRegex(RuntimeError, "Worker name must match"):
            info = WorkerInfo("abc*", worker_id)

        with self.assertRaisesRegex(RuntimeError, "Worker name must match"):
            info = WorkerInfo(" ", worker_id)

        with self.assertRaisesRegex(RuntimeError, "must be non-empty"):
            info = WorkerInfo("", worker_id)

        # 断言 RPC WorkerInfo 中最大名字长度小于128的错误
        # 如果消息中的数字不匹配，可能是 RPC WorkerInfo 中的 MAX_NAME_LEN 值已更改。
        with self.assertRaisesRegex(RuntimeError, "shorter than 128"):
            info = WorkerInfo("".join(["a" for i in range(500)]), worker_id)

    # 测试 WorkerInfo 能否在 RPC 调用中被序列化和发送
    @dist_init
    # 定义一个测试方法，用于测试通过 RPC 获取下一个节点的信息
    def test_worker_info_pickle(self):
        # 计算目标节点的排名，确保在世界大小范围内循环
        dst_rank = (self.rank + 1) % self.world_size
        # 获取当前节点的工作信息
        worker_info = rpc.api.get_worker_info()
        # 通过 RPC 调用同步函数，将工作信息发送给目标节点，然后返回结果
        ret = rpc.rpc_sync(worker_name(dst_rank), identity, args=(worker_info,))
        # 断言返回的结果与工作信息相等
        self.assertEqual(ret, worker_info)

    # 使用 dist_init 装饰器定义的测试方法，用于测试向 Tensor 添加相同值的操作
    @dist_init
    def test_add(self):
        # 计算当前节点的排名加一，确保在世界大小范围内循环
        n = self.rank + 1
        dst_rank = n % self.world_size
        # 通过 RPC 调用同步函数，将 torch.add 函数应用到两个相同大小的矩阵上
        ret = rpc.rpc_sync(
            worker_name(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        # 断言返回的结果与期望的两倍矩阵相等
        self.assertEqual(ret, torch.ones(n, n) * 2)

    # 静态方法，用于返回调用者的 ID
    @staticmethod
    def return_callee_id():
        return rpc.get_worker_info().id

    # 使用 dist_init 装饰器定义的测试方法，用于测试向具有 ID 的节点发送整数调用的操作
    @dist_init
    def test_int_callee(self):
        # 计算目标节点的排名，确保在世界大小范围内循环
        dst_rank = (self.rank + 1) % self.world_size
        # 通过 RPC 调用同步函数，将返回调用者 ID 的方法发送给目标节点
        ret = rpc.rpc_sync(dst_rank, RpcTest.return_callee_id)
        # 断言返回的结果与目标节点的排名相等
        self.assertEqual(ret, dst_rank)

    # 使用 dist_init 装饰器定义的测试方法，用于测试向带有 ID 的节点添加操作的 RPC 调用
    @dist_init
    def test_add_with_id(self):
        # 计算当前节点的排名加一，确保在世界大小范围内循环
        n = self.rank + 1
        dst_rank = n % self.world_size
        # 获取目标节点的工作信息
        workder_info = rpc.get_worker_info(worker_name(dst_rank))
        # 通过 RPC 调用同步函数，将 torch.add 函数应用到两个相同大小的矩阵上
        ret = rpc.rpc_sync(
            workder_info, torch.add, args=(torch.ones(n, n), torch.ones(n, n))
        )
        # 断言返回的结果与期望的两倍矩阵相等
        self.assertEqual(ret, torch.ones(n, n) * 2)

    # 使用 dist_init 装饰器定义的测试方法，用于测试向节点添加标量值的操作
    @dist_init
    def test_scalar_add(self):
        # 计算当前节点的排名加一，确保在世界大小范围内循环
        n = self.rank + 1
        dst_rank = n % self.world_size
        # 通过 RPC 调用同步函数，将 torch.add 函数应用到矩阵和标量上
        ret = rpc.rpc_sync(
            worker_name(dst_rank), torch.add, args=(torch.ones(n, n), n)
        )
        # 断言返回的结果与期望的矩阵加上标量值相等
        self.assertEqual(ret, (torch.ones(n, n) + n))

    # 使用 dist_init 装饰器定义的测试方法，用于测试异步添加操作的 RPC 调用
    @dist_init
    def test_async_add(self):
        # 计算当前节点的排名加一，确保在世界大小范围内循环
        n = self.rank + 1
        dst_rank = n % self.world_size
        # 通过 RPC 调用异步函数，将 torch.add 函数应用到两个相同大小的矩阵上
        fut = rpc.rpc_async(
            worker_name(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        # 等待异步操作完成，并断言返回的结果与期望的两倍矩阵相等
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

    # 使用 dist_init 装饰器定义的测试方法，用于测试向节点发送非零元素的 RPC 调用
    @dist_init
    def test_nonzero(self):
        # 计算当前节点的排名加一，确保在世界大小范围内循环
        n = self.rank + 1
        dst_rank = n % self.world_size
        # 创建一个全一矩阵，并将当前节点位置的元素设为零
        x = torch.ones(self.world_size, self.world_size)
        x[self.rank][self.rank] = 0
        # 通过 RPC 调用同步函数，将 torch.nonzero 函数应用到矩阵上
        ret = rpc.rpc_sync(worker_name(dst_rank), torch.nonzero, args=(x,))
        # 断言返回的结果与矩阵中非零元素的位置相等
        self.assertEqual(ret, x.nonzero())

    # 使用 dist_init 装饰器定义的测试方法，用于测试多重 RPC 调用
    @dist_init
    def test_multi_rpc(self):
        # 调用内部方法 _multi_rpc 来测试多重 RPC 调用的情况
        self._multi_rpc(False)

    # 使用 dist_init 装饰器定义的测试方法，用于测试等待两次异步操作的情况
    @dist_init
    def test_future_wait_twice(self):
        # 计算目标节点的名称
        dst = worker_name((self.rank + 1) % self.world_size)
        futs = []
        # 向目标节点发送多次异步函数调用
        for i in range(20):
            futs.append(rpc.rpc_async(dst, raise_func))

        # 断言在等待所有异步操作时抛出预期的 ValueError 异常
        with self.assertRaisesRegex(ValueError, "Expected error"):
            torch.futures.wait_all(futs)

        # 对每个异步操作的 Future 对象，断言在等待时抛出预期的 ValueError 异常
        for fut in futs:
            with self.assertRaisesRegex(ValueError, "Expected error"):
                fut.wait()

    # 使用 dist_init 装饰器定义的测试方法，用于初始化 RPC，但不设置其它内容
    @dist_init(setup_rpc=False)
    def test_wait_all_workers_timeout(self):
        # 初始化分布式环境
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        # 初始化 RPC
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        # 保存原始的 _wait_all_workers 函数
        og_func = rpc.api._wait_all_workers

        # 定义一个新的 _wait_all_workers 函数，用于等待所有工作节点休眠
        def wait_all_workers_sleep(timeout):
            rpc.api._all_gather(SlowPickleClass(0.5), timeout=timeout)

        # 替换 _wait_all_workers 函数为 wait_all_workers_sleep
        rpc.api._wait_all_workers = wait_all_workers_sleep

        try:
            # 测试是否在超时时间内触发 RuntimeError 异常
            with self.assertRaisesRegex(RuntimeError, ''):
                rpc.shutdown(graceful=True, timeout=0.01)
        finally:
            # 恢复原始的 _wait_all_workers 函数
            rpc.api._wait_all_workers = og_func

        # 同步所有工作节点的操作
        dist.barrier()

    def test_wait_all_workers_dense(self):
        # 测试等待所有工作节点完成密集型 RPC 操作
        self._wait_all_workers(heavy_rpc, torch.ones(100, 100))

    def test_wait_all_workers_twice_dense(self):
        # 测试两次等待所有工作节点完成密集型 RPC 操作
        self._wait_all_workers_twice(heavy_rpc, torch.ones(100, 100))

    @dist_init
    def test_all_gather(self):
        # 测试全局聚合操作 _all_gather
        info = rpc.get_worker_info()
        results = rpc.api._all_gather(info.id)
        expected = {}
        for info in rpc._get_current_rpc_agent().get_worker_infos():
            expected[info.name] = info.id

        # 断言聚合结果与预期结果一致
        self.assertEqual(expected, results)

    @dist_init
    def test_all_gather_timeout(self):
        # 设置 RPC 超时时间为 0.1 秒
        rpc._set_rpc_timeout(0.1)

        if self.rank == 0:
            # 如果当前节点为 rank 为 0，则测试超时异常
            with self.assertRaisesRegex(
                RuntimeError,
                "timed out in _all_gather after 0\\.10 seconds"
            ):
                rpc.api._all_gather(SlowPickleClass(0.5))
        else:
            # 否则，预期超时异常信息
            expected_error = self.get_timeout_error_regex()
            with self.assertRaisesRegex(RuntimeError, expected_error):
                rpc.api._all_gather(SlowPickleClass(0.5))

    def _test_barrier_helper(self, info, names, multi_threaded=False):
        # 对帮助函数 _test_barrier_helper 进行测试
        names = sorted(names)
        leader = names[0]
        # 同步调用，重置计数器
        rpc.rpc_sync(leader, _reset_count)
        if not multi_threaded and info.name == leader:
            # 如果非多线程且当前节点为 leader，则验证计数器为 0
            self.assertEqual(_rpc_barrier_count, 0)
        # 执行分布式屏障操作
        rpc.api._barrier(names)
        # 同步调用，增加计数器
        rpc.rpc_sync(leader, _increment_count)
        # 再次执行分布式屏障操作
        rpc.api._barrier(names)
        if not multi_threaded and info.name == leader:
            # 如果非多线程且当前节点为 leader，则验证计数器与节点数一致
            self.assertEqual(_rpc_barrier_count, len(names))

    @dist_init
    def test_rpc_barrier_all(self):
        # 测试全局 RPC 屏障操作，传入所有工作节点的信息
        info = rpc.get_worker_info()
        all_worker_info = rpc._get_current_rpc_agent().get_worker_infos()
        names = [worker.name for worker in all_worker_info]
        self._test_barrier_helper(info, names)
    def test_rpc_barrier_subset(self):
        # 测试 RPC 屏障，当进程使用完整列表的不同子集进行调用时
        # 获取当前进程的信息
        info = rpc.get_worker_info()
        # 获取所有工作进程的信息
        all_worker_info = rpc._get_current_rpc_agent().get_worker_infos()
        # 根据进程 ID 是否为奇数，选择相应的工作进程名列表
        if info.id % 2:
            names = [worker.name for worker in all_worker_info if worker.id % 2]
        else:
            names = [worker.name for worker in all_worker_info if not worker.id % 2]
        # 调用辅助函数，测试屏障
        self._test_barrier_helper(info, names)

    @dist_init
    def test_rpc_barrier_partial_subset(self):
        # 测试 RPC 屏障，当某些进程不参与屏障时
        # 获取当前进程的信息
        info = rpc.get_worker_info()
        # 获取所有工作进程的信息
        all_worker_info = rpc._get_current_rpc_agent().get_worker_infos()
        # 根据进程 ID 是否为奇数，选择相应的工作进程名列表或当前进程名
        if info.id % 2:
            names = [worker.name for worker in all_worker_info if worker.id % 2]
        else:
            names = [f"worker{info.id}"]
        # 调用辅助函数，测试屏障
        self._test_barrier_helper(info, names)

    @dist_init
    def test_rpc_barrier_multithreaded(self):
        # 测试多线程调用屏障的实现
        # 获取当前进程的信息
        info = rpc.get_worker_info()
        # 获取所有工作进程的信息
        all_worker_info = rpc._get_current_rpc_agent().get_worker_infos()
        # 获取所有工作进程的名称列表
        names = [worker.name for worker in all_worker_info]
        # 创建多个线程，并调用辅助函数，测试屏障，确保不会 hang
        threads = []
        for _ in range(3):
            th = threading.Thread(target=self._test_barrier_helper, args=(info, names, True))
            threads.append(th)
            th.start()
        for th in threads:
            th.join()

    @dist_init
    def test_graceful_shutdown_with_uneven_workload(self):
        """测试优雅关闭"""
        # 运行不均衡工作负载的函数
        self._run_uneven_workload(heavy_rpc, torch.ones(100, 100))

    @dist_init(setup_rpc=False)
    def test_shutdown_followed_by_rpc(self):
        # 初始化 RPC
        rpc.init_rpc(
            name="worker%d" % self.rank,
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )
        # 设置一个变量 n
        n = self.rank + 1
        # 计算目标进程的排名
        dst_rank = n % self.world_size
        # 远程同步调用 RPC，执行加法操作
        ret = rpc.rpc_sync(
            worker_name(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        # 断言返回结果与预期值相等
        self.assertEqual(ret, torch.ones(n, n) * 2)
        # 关闭 RPC
        rpc.shutdown()

        # 使用断言检查是否抛出了预期的 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "^RPC has not been initialized"):
            rpc.rpc_sync(
                worker_name(dst_rank),
                torch.add,
                args=(torch.ones(n, n), torch.ones(n, n)),
            )

    @dist_init
    def test_expected_src(self):
        # 计算目标进程的排名和期望的源进程排名
        dst_rank = (self.rank + 1) % self.world_size
        expected_src_rank = (self.rank - 1) % self.world_size
        # 远程同步调用 RPC，设置一个值，并获取返回值
        ret = rpc.rpc_sync(worker_name(dst_rank), set_value, args=(self.rank,))
        value = VALUE_FUTURE.result()
        # 断言返回值与期望的源进程排名相等
        self.assertEqual(value, expected_src_rank)
    @dist_init
    def test_py_built_in(self):
        # 定义一个分布式测试函数，使用内置函数进行 RPC 调用和断言检查
        n = self.rank + 1
        dst_rank = n % self.world_size
        # 同步 RPC 调用，调用 min 函数，传入参数 n, n + 1, n + 2
        ret = rpc.rpc_sync(worker_name(dst_rank), min, args=(n, n + 1, n + 2))
        # 断言 RPC 返回值与 n, n + 1, n + 2 的最小值相等
        self.assertEqual(ret, min(n, n + 1, n + 2))

    @dist_init
    def test_py_user_defined(self):
        # 定义一个分布式测试函数，使用用户自定义函数进行 RPC 调用和断言检查
        n = self.rank + 1
        dst_rank = n % self.world_size
        # 同步 RPC 调用，调用 my_function 函数，传入关键字参数 {"a": n, "b": n + 1, "c": n + 2}
        ret = rpc.rpc_sync(
            worker_name(dst_rank),
            my_function,
            kwargs={"a": n, "b": n + 1, "c": n + 2},
        )
        # 断言 RPC 返回值与 my_function(n, n + 1, n + 2) 的返回值相等
        self.assertEqual(ret, my_function(n, n + 1, n + 2))

    def test_build_rpc_profiling_key(self):
        # 测试构建 RPC 分析键的函数，验证生成的键包含必要的信息
        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            # 调用 _build_rpc_profiling_key 函数，生成 RPC 分析键
            rpc_profiling_key = _build_rpc_profiling_key(
                exec_mode, "foo", "worker0", "worker1"
            )
            # 断言 exec_mode.value, "foo", "worker0", "worker1" 在生成的键中
            self.assertIn(exec_mode.value, rpc_profiling_key)
            self.assertIn("foo", rpc_profiling_key)
            self.assertIn("worker0", rpc_profiling_key)
            self.assertIn("worker1", rpc_profiling_key)

    def check_profiling_info(self, self_worker_name, dst_worker_name, func, rpc_event, rpc_exec_mode):
        # 检查分析信息函数，验证事件名称中包含必要的信息
        self.assertTrue(self_worker_name in rpc_event.name)
        self.assertTrue(dst_worker_name in rpc_event.name)
        if isinstance(func, torch.jit.ScriptFunction):
            self.assertTrue(torch._jit_internal._qualified_name(func) in rpc_event.name)
        else:
            self.assertTrue(func.__name__ in rpc_event.name)
        self.assertTrue(rpc_exec_mode.value in rpc_event.name)
        self.assertEqual(rpc_event.count, 1)

    @dist_init
    def test_profiler_rpc_record_shapes(self):
        # 分布式测试函数，验证远程 RPC 记录形状时的行为
        if self.rank != 1:
            return
        dst = (self.rank + 1) % self.world_size
        dst_worker = worker_name(dst)
        t1, t2 = torch.ones(100), torch.ones(100)
        # 使用 _profile 上下文管理器记录形状，并进行 RPC 调用
        with _profile(record_shapes=True) as prof:
            rpc.rpc_sync(dst_worker, torch.add, args=(t1, t2))

        function_events = prof.function_events
        # 获取远程事件列表
        remote_events = [event for event in function_events if event.is_remote]
        # 获取包含 "aten::add" 的远程事件
        remote_add_event = next(
            event for event in remote_events if "aten::add" in event.name
        )
        # 获取远程 aten::add 事件的输入形状
        remote_add_input_shapes = remote_add_event.input_shapes
        # 在等效的本地操作上运行分析器，验证形状是否相同
        with _profile(record_shapes=True) as prof:
            torch.add(t1, t2)

        local_function_events = prof.function_events
        # 获取本地 aten::add 事件
        local_add_event = next(
            event for event in local_function_events if "aten::add" in event.name
        )
        # 获取本地 aten::add 事件的输入形状
        local_add_input_shapes = local_add_event.input_shapes
        # 断言远程 aten::add 事件的输入形状与本地相同
        self.assertEqual(remote_add_input_shapes, local_add_input_shapes)
    # 如果不是排名为1的进程，直接返回，不进行测试
    def test_profiler_rpc_memory(self):
        if self.rank != 1:
            return
        # 计算下一个目标进程的排名，并生成其工作名称
        dst = (self.rank + 1) % self.world_size
        dst_worker = worker_name(dst)
        # 使用 _profile 对象，开启内存分析功能，记录分析结果到 p
        with _profile(profile_memory=True) as p:
            # 异步 RPC 调用，调用 udf_with_torch_ops 函数
            fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
            # 等待 RPC 调用完成
            res = fut.wait()

        # 获取记录的函数事件
        function_events = p.function_events
        # 从函数事件中提取 CPU 内存使用量，生成集合
        event_cpu_mem_usages = {event.cpu_memory_usage for event in function_events}
        # 如果事件中的 CPU 内存使用量集合只包含 0，则表示未进行内存分析
        # 因为未传播 cpu_memory_usage 到网络，集合中只包含 0 表示没有进行内存分析
        self.assertNotEqual({0}, event_cpu_mem_usages)

        # 重新测试，关闭内存分析
        # 使用 _profile 对象，关闭内存分析功能，记录分析结果到 p
        with _profile(profile_memory=False) as p:
            # 再次异步 RPC 调用，调用 udf_with_torch_ops 函数
            fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
            # 等待 RPC 调用完成
            res = fut.wait()

        # 获取记录的函数事件
        function_events = p.function_events
        # 从函数事件中提取 CPU 内存使用量，生成集合
        event_cpu_mem_usages = {event.cpu_memory_usage for event in function_events}
        # 如果事件中的 CPU 内存使用量集合为 {0}，则表示内存分析已关闭
        self.assertEqual({0}, event_cpu_mem_usages)

    # 分布式初始化装饰器下的测试函数，仅在排名为1的进程上执行
    @dist_init
    def test_profiler_export_trace(self):
        if self.rank != 1:
            return
        # 计算下一个目标进程的排名，并生成其工作名称
        dst = (self.rank + 1) % self.world_size
        dst_worker = worker_name(dst)
        # 使用 _profile 对象，记录分析结果到 p
        with _profile() as p:
            # 异步 RPC 调用，调用 udf_with_torch_ops 函数
            fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
            # 等待 RPC 调用完成
            res = fut.wait()

        # 获取记录的函数事件
        events = p.function_events
        # 使用临时文件名，导出 Chrome 跟踪文件
        with TemporaryFileName() as fname:
            path = fname
            p.export_chrome_trace(path)
            # 打开导出的 Chrome 跟踪文件
            with open(path) as f:
                trace = json.load(f)
                # 获取事件名称列表
                event_names = [event['name'] for event in trace]
                # 检查预期的远程事件及 RPCExecMode.ASYNC.value 是否存在于事件名称中
                for expected_event_name in EXPECTED_REMOTE_EVENTS + [RPCExecMode.ASYNC.value]:
                    event_exists = any(expected_event_name in event_name for event_name in event_names)
                    self.assertTrue(event_exists)
    def _run_test_profiler_remote_events_profiled(self):
        # 测试在远程节点上成功调用分析器，并将远程事件收集到本地分析器中。
        # 如果当前节点的排名不为1，则直接返回，不进行测试。
        if self.rank != 1:
            return

        # 获取除当前节点外的所有目标节点的排名列表
        dst_ranks = [rank for rank in range(0, self.world_size) if rank != self.rank]
        for dst in dst_ranks:
            # 获取目标节点的节点名
            dst_worker = worker_name(dst)
            # 使用 _profile 上下文管理器开始性能分析
            with _profile() as prof:
                # 向目标节点发送 RPC 异步调用请求，并获取返回的 Future 对象
                fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
                # 等待 RPC 调用完成并获取返回值
                ret = fut.wait()

            # 获取性能分析中的函数事件列表
            events = prof.function_events

            # 从函数事件中获取特定类型的 RPC 事件
            rpc_event = get_function_event(events, RPCExecMode.ASYNC.value)
            # 检查性能分析信息
            self.check_profiling_info(
                worker_name(self.rank),
                dst_worker,
                udf_with_torch_ops,
                rpc_event,
                RPCExecMode.ASYNC,
            )

            # 获取所有远程事件的字典，键为事件名，值为事件对象
            remote_events = {event.name: event for event in events if event.is_remote}
            # 构建 RPC 性能分析键
            rpc_profiling_key = _build_rpc_profiling_key(
                RPCExecMode.ASYNC,
                udf_with_torch_ops.__qualname__,
                worker_name(self.rank),
                worker_name(dst),
            )

            # 遍历预期的所有远程事件名
            for expected_remote_event_name in EXPECTED_REMOTE_EVENTS:
                # 构建预期的远程事件键
                expected_key = rpc_profiling_key + REMOTE_OP_STR + expected_remote_event_name
                # 断言预期的远程事件键存在于远程事件字典中
                self.assertTrue(expected_key in remote_events)
                # 获取对应的远程事件对象
                remote_event = remote_events[expected_key]
                # 断言远程事件的节点 ID 与目标节点的排名相符
                self.assertEqual(remote_event.node_id, dst)

            # 验证远程事件在性能分析输出中的顺序
            def convert_remote_to_local(event_name):
                remote_op_key = rpc_profiling_key + REMOTE_OP_STR
                return event_name[
                    event_name.find(remote_op_key)
                    + len(remote_op_key) :
                ]

            # 提取所有符合预期远程事件的本地事件名列表
            remote_events_list = [
                convert_remote_to_local(event.name)
                for event in events
                if convert_remote_to_local(event.name) in EXPECTED_REMOTE_EVENTS
            ]
            # 断言性能分析中的远程事件列表与预期事件集合相匹配
            self.assertEqual(
                set(remote_events_list),
                set(EXPECTED_REMOTE_EVENTS),
                f"Mismatch between profiled events: {set(remote_events_list)} and expected events: {set(EXPECTED_REMOTE_EVENTS)}",
            )

    @dist_init
    def test_profiler_remote_events_profiled(self):
        self._run_test_profiler_remote_events_profiled()

    @dist_init
    def test_profiler_remote_events_profiled_single_threaded(self):
        self._run_test_profiler_remote_events_profiled()
    @dist_init
    # 使用分布式初始化装饰器标记测试函数
    def test_rpc_profiling_async_function(self):
        # 使用给定的初始化方法和进程信息初始化进程组
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        # 执行异步 RPC profiling 测试函数
        self._run_rpc_profiling_async_function()
        # 如果CUDA可用，等待所有进程完成
        if torch.cuda.is_available():
            dist.barrier()
            # 使用CUDA设备运行异步 RPC profiling 测试函数
            self._run_rpc_profiling_async_function(device="cuda:0")

    @dist_init
    # 使用分布式初始化装饰器标记测试函数
    def test_rpc_profiling_async_function_single_threaded(self):
        # 使用给定的初始化方法和进程信息初始化进程组
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        # 执行异步 RPC profiling 测试函数
        self._run_rpc_profiling_async_function()
        # 如果CUDA可用，等待所有进程完成
        if torch.cuda.is_available():
            dist.barrier()
            # 使用CUDA设备运行异步 RPC profiling 测试函数
            self._run_rpc_profiling_async_function(device="cuda:0")

    def _run_rpc_profiling_async_function(self, device="cpu"):
        # 如果当前进程不是排名为1的进程，则直接返回
        if self.rank != 1:
            return

        # 确定目标节点dst1和dst2
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)
        # 创建两个大小为2的torch张量x和y
        x = torch.ones(2)
        y = torch.ones(2)
        # 使用_profiling上下文管理器开始性能分析
        with _profile() as prof:
            # 异步RPC调用，调用slow_async_add函数
            ret = rpc.rpc_async(
                dst1, slow_async_add, args=(dst2, x, y, device), timeout=20
            )
            # 等待RPC调用完成并获取返回结果
            out = ret.wait()

        # 获取函数事件列表
        function_events = prof.function_events
        # 构建RPC profiling键的前缀
        key_prefix = _build_rpc_profiling_key(
            RPCExecMode.ASYNC, slow_async_add.__qualname__, worker_name(self.rank), dst1
        )
        # 构建嵌套RPC profiling键的前缀
        nested_rpc_key_prefix = _build_rpc_profiling_key(
            RPCExecMode.ASYNC, slow_add.__qualname__, dst1, dst2
        )
        # 期望的键应包含以下两个前缀
        expected_key = key_prefix + REMOTE_OP_STR + nested_rpc_key_prefix
        # 从函数事件中选择远程事件
        remote_events = [event for event in function_events if event.is_remote]
        # 选择与期望键匹配的RPC远程事件
        rpc_remote_event = [
            event for event in remote_events if event.name == expected_key
        ]
        # 断言只有一个RPC远程事件与期望键匹配
        self.assertEqual(1, len(rpc_remote_event))
        # 获取匹配的RPC远程事件
        rpc_remote_event = rpc_remote_event[0]
        # 断言RPC远程事件的node_id与dst1相匹配
        self.assertEqual(rpc_remote_event.node_id, (self.rank + 1) % self.world_size)
        # 构建远程加法操作键
        remote_add_key = (
            expected_key + REMOTE_OP_STR + torch.jit._builtins._find_builtin(torch.add)
        )
        # 选择与远程加法操作键匹配的远程加法事件
        remote_add_event = [
            event for event in remote_events if event.name == remote_add_key
        ]
        # 断言只有一个远程加法事件与远程加法操作键匹配
        self.assertEqual(1, len(remote_add_event))
        # 获取匹配的远程加法事件
        remote_add_event = remote_add_event[0]
        # 断言远程加法事件的node_id与dst2相匹配
        self.assertEqual(remote_add_event.node_id, (self.rank + 2) % self.world_size)
    def test_rpc_profiling_remote_record_function(self):
        # 测试远程 RPC 运行的函数，使用 record_function 显示预期的性能分析块。
        if self.rank != 1:
            return
        # 获取除了当前进程外的所有目标进程的 ranks
        dst_ranks = [i for i in range(self.world_size) if i != self.rank]
        for dst_rank in dst_ranks:
            # 获取目标进程的名称
            dst_worker = worker_name(dst_rank)
            # 使用 _profile() 上下文管理器来进行性能分析
            with _profile() as prof:
                # 发送 RPC 异步调用到目标进程 dst_worker，执行 udf_with_torch_ops 函数
                fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=(-1, True))
                fut.wait()

            # 获取函数事件
            function_events = prof.function_events
            # 从函数事件中筛选出包含 "##forward##" 字符串的远程记录函数事件
            record_function_remote_event = [
                evt for evt in function_events if "##forward##" in evt.name
            ]
            # 断言找到的远程记录函数事件只有一个
            self.assertEqual(1, len(record_function_remote_event))
            record_function_remote_event = record_function_remote_event[0]
            # 断言远程记录函数事件的节点 ID 与目标进程的 rank 相符
            self.assertEqual(record_function_remote_event.node_id, dst_rank)
            # cpu_children 只返回直接子节点，因此需要递归获取所有子节点。

            def get_cpu_children(event):
                if not event.cpu_children:
                    return []
                cpu_children = event.cpu_children
                for e in event.cpu_children:
                    cpu_children.extend(get_cpu_children(e))
                return cpu_children

            # 获取远程记录函数事件的所有 CPU 子节点
            remote_children = get_cpu_children(record_function_remote_event)
            # 获取本地函数事件并找到本地的记录函数事件
            with _profile() as prof:
                udf_with_torch_ops(-1, True)

            # 获取本地函数事件
            local_function_events = prof.function_events
            # 从本地函数事件中找到包含 "##forward##" 字符串的记录函数事件
            local_record_function_event = next(
                evt for evt in local_function_events if "##forward##" in evt.name
            )
            # 获取本地记录函数事件的所有 CPU 子节点
            local_children = get_cpu_children(local_record_function_event)
            # 获取本地子节点的名称列表
            local_children_names = [
                evt.name for evt in local_children
            ]

            # 远程操作字符串前缀
            REMOTE_OP_STR = "#remote_op: "

            # 将远程事件名转换为本地事件名
            def convert_remote_to_local(event_name):
                remote_op_key = REMOTE_OP_STR
                return event_name[
                    event_name.find(remote_op_key) + len(remote_op_key) :
                ]

            # 验证远程子节点是否存在于本地子节点中
            for evt in remote_children:
                local_name = convert_remote_to_local(evt.name)
                self.assertTrue(local_name in local_children_names)
    # 在验证性能分析工作负载的函数中，验证远程事件是否包含特定操作（例如 "aten::mul"）
    def validate_profiling_workload(self, dst, prof):

        # 将远程操作事件名转换为本地操作名称
        def convert_remote_to_local(event_name):
            return event_name[event_name.find(REMOTE_OP_STR) + len(REMOTE_OP_STR):]

        # 获取性能分析对象中的所有事件
        events = prof.function_events
        # 从所有事件中筛选出远程操作的事件，并以操作名称（去掉前缀后的部分）为键，事件对象为值构成字典
        remote_events = {
            convert_remote_to_local(event.name): event
            for event in events
            if event.is_remote
        }
        # 断言 "aten::mul" 在远程事件中存在
        self.assertTrue("aten::mul" in remote_events)
        # 获取特定远程操作事件 "aten::mul"
        remote_mul_event = remote_events["aten::mul"]
        # 断言特定远程操作事件的节点 ID 符合预期的目标节点 ID（dst）
        self.assertEqual(remote_mul_event.node_id, dst)
        # 检查性能分析信息的完整性和准确性
        self.check_profiling_info(
            worker_name(self.rank),
            worker_name(dst),
            torch.mul,
            remote_mul_event,
            RPCExecMode.ASYNC,
        )

    # 在自动微分上下文中运行性能分析测试
    def _run_test_profiler_with_autograd_context(self):
        # 计算目标节点 ID
        dst = (self.rank + 1) % self.world_size
        if self.rank == 1:
            # 在分布式自动微分上下文中运行性能分析工作负载
            with dist_autograd.context() as context_id:
                with _profile() as prof:
                    self.run_profiling_workload(dst)

            # 验证性能分析工作负载的结果
            self.validate_profiling_workload(dst, prof)

            # 确保上下文管理器顺序颠倒时事件记录如预期一样
            with _profile() as prof:
                with dist_autograd.context() as context_id:
                    self.run_profiling_workload(dst)

            # 再次验证性能分析工作负载的结果
            self.validate_profiling_workload(dst, prof)

    # 单线程模式下运行具有自动微分上下文的性能分析测试
    @dist_init
    def test_profiler_with_autograd_context_single_threaded(self):
        self._run_test_profiler_with_autograd_context()

    # 运行具有自动微分上下文的性能分析测试
    @dist_init
    def test_profiler_with_autograd_context(self):
        self._run_test_profiler_with_autograd_context()

    # 在同步 RPC 用户定义函数中运行性能分析测试
    def _run_test_profiler_with_sync_rpc_udf(self):
        self._profiler_test_with_rpc(RPCExecMode.SYNC, my_sleep_func, args=(1,))
        self._profiler_test_with_rpc(RPCExecMode.SYNC, my_sleep_func, args=(1,),
                                     use_record_function=True)

    # 单线程模式下运行同步 RPC 用户定义函数的性能分析测试
    @dist_init
    def test_profiler_with_sync_rpc_udf(self):
        self._run_test_profiler_with_sync_rpc_udf()

    # 运行同步 RPC 用户定义函数的性能分析测试
    @dist_init
    def test_profiler_with_sync_rpc_udf_single_threaded(self):
        self._run_test_profiler_with_sync_rpc_udf()

    # 在同步 RPC 内置函数中运行性能分析测试
    def _run_test_profiler_with_sync_rpc_builtin(self):
        self._profiler_test_with_rpc(
            RPCExecMode.SYNC, torch.mul, args=(torch.ones(1), torch.ones(1))
        )
        self._profiler_test_with_rpc(
            RPCExecMode.SYNC, torch.mul, args=(torch.ones(1), torch.ones(1)),
            use_record_function=True
        )

    # 运行同步 RPC 内置函数的性能分析测试
    @dist_init
    def test_profiler_with_sync_rpc_builtin(self):
        self._run_test_profiler_with_sync_rpc_builtin()

    # 在同步 RPC 用户定义函数和内置函数中运行性能分析测试
    @dist_init
    def test_profiler_with_sync_rpc_builtin_single_threaded(self):
        # 调用内部方法 _run_test_profiler_with_sync_rpc_builtin，测试同步远程过程调用（RPC）
        self._run_test_profiler_with_sync_rpc_builtin()

    def _run_test_profiler_with_async_rpc_udf(self):
        # 使用异步模式进行 RPC 测试，执行自定义的睡眠函数 my_sleep_func
        self._profiler_test_with_rpc(RPCExecMode.ASYNC, my_sleep_func, args=(1,))
        # 使用异步模式进行 RPC 测试，执行自定义的睡眠函数 my_sleep_func，并启用记录函数
        self._profiler_test_with_rpc(RPCExecMode.ASYNC, my_sleep_func, args=(1,),
                                     use_record_function=True)
        # 测试 kineto 分析器在 RPC 中的使用，不应该启用 RPC 的分析，并且不会引起问题
        self._profiler_test_with_rpc(
            RPCExecMode.ASYNC, my_sleep_func, args=(1,), kineto_profile=True
        )

    @dist_init
    def test_profiler_with_async_rpc_udf(self):
        # 调用内部方法 _run_test_profiler_with_async_rpc_udf，测试异步远程过程调用（RPC）
        self._run_test_profiler_with_async_rpc_udf()

    @dist_init
    def test_profiler_with_async_rpc_udf_single_threaded(self):
        # 调用内部方法 _run_test_profiler_with_async_rpc_udf，测试单线程环境下的异步远程过程调用（RPC）
        self._run_test_profiler_with_async_rpc_udf()

    def _run_test_profiler_with_async_rpc_builtin(self):
        # 使用异步模式进行 RPC 测试，执行 torch.mul 内置函数
        self._profiler_test_with_rpc(
            RPCExecMode.ASYNC, torch.mul, args=(torch.ones(1), torch.ones(1))
        )
        # 使用异步模式进行 RPC 测试，执行 torch.mul 内置函数，并启用记录函数
        self._profiler_test_with_rpc(
            RPCExecMode.ASYNC, torch.mul, args=(torch.ones(1), torch.ones(1)),
            use_record_function=True
        )

    @dist_init
    def test_profiler_with_async_rpc_builtin(self):
        # 调用内部方法 _run_test_profiler_with_async_rpc_builtin，测试异步远程过程调用（RPC）
        self._run_test_profiler_with_async_rpc_builtin()

    @dist_init
    def test_profiler_with_async_rpc_builtin_single_threaded(self):
        # 调用内部方法 _run_test_profiler_with_async_rpc_builtin，测试单线程环境下的异步远程过程调用（RPC）
        self._run_test_profiler_with_async_rpc_builtin()

    def _run_test_profiler_with_remote_udf(self):
        # 使用远程模式进行 RPC 测试，执行自定义的睡眠函数 my_sleep_func
        self._profiler_test_with_rpc(RPCExecMode.REMOTE, my_sleep_func, args=(1,))
        # 使用远程模式进行 RPC 测试，执行自定义的睡眠函数 my_sleep_func，并启用记录函数
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE, my_sleep_func, args=(1,), use_record_function=True
        )
        # 测试远程调用到自身的情况
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE, my_sleep_func, args=(1,), dst=self.rank
        )

    @dist_init
    def test_profiler_with_remote_udf(self):
        # 调用内部方法 _run_test_profiler_with_remote_udf，测试远程过程调用（RPC）使用自定义的睡眠函数
        self._run_test_profiler_with_remote_udf()

    @dist_init
    def test_profiler_with_remote_udf_single_threaded(self):
        # 调用内部方法 _run_test_profiler_with_remote_udf，测试单线程环境下的远程过程调用（RPC）使用自定义的睡眠函数
        self._run_test_profiler_with_remote_udf()

    def _run_test_profiler_with_remote_builtin(self):
        # 使用远程模式进行 RPC 测试，执行 torch.mul 内置函数
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE, torch.mul, args=(torch.ones(1), torch.ones(1))
        )
        # 使用远程模式进行 RPC 测试，执行 torch.mul 内置函数，并启用记录函数
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE, torch.mul, args=(torch.ones(1), torch.ones(1)),
            use_record_function=True
        )
        # 测试远程调用到自身的情况
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE,
            torch.mul,
            args=(torch.ones(1), torch.ones(1)),
            dst=self.rank,
        )

    @dist_init
    def test_profiler_with_remote_builtin(self):
        # 调用内部方法 _run_test_profiler_with_remote_builtin，测试远程过程调用（RPC）使用内置函数
        self._run_test_profiler_with_remote_builtin()

    @dist_init
    # 定义一个测试方法，用于测试使用内置函数的性能分析器，单线程执行
    def test_profiler_with_remote_builtin_single_threaded(self):
        # 调用内部方法来执行远程内置函数的性能分析
        self._run_test_profiler_with_remote_builtin()

    # 运行测试性能分析器与脚本异步远程过程调用（RPC）
    def _run_test_profiler_with_script_async_rpc(self):
        # 使用异步模式执行远程过程调用，调用名为 my_script_func 的函数，并传入参数 torch.tensor(1)
        self._profiler_test_with_rpc(
            RPCExecMode.ASYNC, my_script_func, args=(torch.tensor(1),)
        )
        # 使用异步模式执行远程过程调用，同时使用记录函数，调用 my_script_func 函数，并传入参数 torch.tensor(1)
        self._profiler_test_with_rpc(
            RPCExecMode.ASYNC,
            my_script_func,
            args=(torch.tensor(1),),
            use_record_function=True,
        )

    # 使用分布式初始化装饰器，测试脚本异步RPC性能分析
    @dist_init
    def test_profiler_with_script_async_rpc(self):
        # 调用方法以运行测试脚本异步RPC
        self._run_test_profiler_with_script_async_rpc()

    # 使用分布式初始化装饰器，测试脚本异步RPC单线程性能分析
    @dist_init
    def test_profiler_with_script_async_rpc_single_threaded(self):
        # 调用方法以运行测试脚本异步RPC单线程性能分析
        self._run_test_profiler_with_script_async_rpc()

    # 运行测试性能分析器与脚本同步远程过程调用（RPC）
    def _run_test_profiler_with_script_sync_rpc(self):
        # 使用同步模式执行远程过程调用，调用名为 my_script_func 的函数，并传入参数 torch.tensor(1)
        self._profiler_test_with_rpc(
            RPCExecMode.SYNC, my_script_func, args=(torch.tensor(1),)
        )
        # 使用同步模式执行远程过程调用，同时使用记录函数，调用 my_script_func 函数，并传入参数 torch.tensor(1)
        self._profiler_test_with_rpc(
            RPCExecMode.SYNC,
            my_script_func,
            args=(torch.tensor(1),),
            use_record_function=True,
        )

    # 使用分布式初始化装饰器，测试脚本同步RPC性能分析
    @dist_init
    def test_profiler_with_script_sync_rpc(self):
        # 调用方法以运行测试脚本同步RPC
        self._run_test_profiler_with_script_sync_rpc()

    # 使用分布式初始化装饰器，测试脚本同步RPC单线程性能分析
    @dist_init
    def test_profiler_with_script_sync_rpc_single_threaded(self):
        # 调用方法以运行测试脚本同步RPC单线程性能分析
        self._run_test_profiler_with_script_sync_rpc()

    # 运行测试性能分析器与脚本远程过程调用（RPC）
    def _run_test_profiler_with_script_remote_rpc(self):
        # 使用远程模式执行远程过程调用，调用名为 my_script_func 的函数，并传入参数 torch.tensor(1)
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE, my_script_func, args=(torch.tensor(1),)
        )
        # 使用远程模式执行远程过程调用，同时使用记录函数，调用 my_script_func 函数，并传入参数 torch.tensor(1)
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE,
            my_script_func,
            args=(torch.tensor(1),),
            use_record_function=True,
        )
        # 测试远程到自身的远程过程调用
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE, my_script_func, args=(torch.tensor(1),), dst=self.rank
        )

    # 使用分布式初始化装饰器，测试脚本远程RPC性能分析
    @dist_init
    def test_profiler_with_script_remote_rpc(self):
        # 调用方法以运行测试脚本远程RPC
        self._run_test_profiler_with_script_remote_rpc()

    # 使用分布式初始化装饰器，测试脚本远程RPC单线程性能分析
    @dist_init
    def test_profiler_with_script_remote_rpc_single_threaded(self):
        # 调用方法以运行测试脚本远程RPC单线程性能分析
        self._run_test_profiler_with_script_remote_rpc()
    # 定义一个测试函数，用于验证全局事件处理中的顶层事件是否符合预期
    def _assert_top_level_events(self, process_global_events, expected_top_level_event_names):
        # 初始化空列表，用于存储所有线程的顶层事件名
        top_level_event_names = []
        # 遍历每个线程上的事件列表
        for thread_local_events in process_global_events:
            # 初始化上一个事件结束时间为0
            last_end_time = 0
            # 遍历每个事件对象
            for event in thread_local_events:
                # 获取事件的名称和时间范围
                event_name = event.name
                time_range = event.time_range
                # 如果当前事件的起始时间晚于上一个事件的结束时间，则将其视为顶层事件
                if time_range.start > last_end_time:
                    top_level_event_names.append(event_name)
                    last_end_time = time_range.end
        # 对顶层事件名列表进行排序
        top_level_event_names = sorted(top_level_event_names)
        # 对预期的顶层事件名列表进行排序
        expected_top_level_event_names = sorted(expected_top_level_event_names)
        # 使用断言验证实际得到的顶层事件名列表是否与预期的一致
        self.assertEqual(
            top_level_event_names,
            expected_top_level_event_names,
            f"Expected events {expected_top_level_event_names}, but got {top_level_event_names}",
        )

    # 标记一个函数作为分布式初始化的入口
    @dist_init
    def test_server_process_global_profiler(self):
        # 如果当前进程的排名不为0，则返回，不进行后续操作
        if self.rank != 0:
            return

        # 计算目标排名，环形取模操作，获取下一个工作进程的排名
        dst_rank = (self.rank + 1) % self.world_size
        # 根据目标排名获取对应的工作进程名称
        dst_worker_name = worker_name(dst_rank)

        # 创建两个张量，分别为1和2
        x = torch.tensor(1)
        y = torch.tensor(2)

        # 在目标工作进程上远程调用函数，启动全局分析器
        outer_profile_rref = rpc.remote(dst_worker_name, rpc._server_process_global_profile)
        # 进入远程上下文管理器
        outer_profile_rref.rpc_sync().__enter__()
        # 在目标工作进程上同步远程调用函数，执行 torch.add 操作
        rpc.rpc_sync(dst_worker_name, torch.add, (x, y))
        # 再次远程调用函数，启动内部全局分析器
        inner_profile_rref = rpc.remote(dst_worker_name, rpc._server_process_global_profile)
        # 进入远程内部上下文管理器
        inner_profile_rref.rpc_sync().__enter__()
        # 在目标工作进程上同步远程调用函数，执行 torch.sub 操作
        rpc.rpc_sync(dst_worker_name, torch.sub, (x, y))
        # 退出内部远程上下文管理器
        inner_profile_rref.rpc_sync().__exit__(None, None, None)
        # 退出外部远程上下文管理器
        outer_profile_rref.rpc_sync().__exit__(None, None, None)

        # 从内部全局分析器获取事件列表
        inner_events = rpc.rpc_sync(dst_worker_name, get_events_from_profile, (inner_profile_rref,))
        # 预期的内部事件列表为 ['aten::sub']
        expected_inner_events = ['aten::sub']
        # 预期的外部事件列表为 ['aten::add', 'aten::sub']
        expected_outer_events = expected_inner_events + ['aten::add']

        # 使用自定义的顶层事件验证函数，验证内部事件
        self._assert_top_level_events(inner_events, expected_inner_events)
        # 再次使用自定义的顶层事件验证函数，验证外部事件
        outer_events = rpc.rpc_sync(dst_worker_name, get_events_from_profile, (outer_profile_rref,))
        self._assert_top_level_events(outer_events, expected_outer_events)

        # 同步内部全局分析器对象的键平均值统计
        inner_profile_rref.rpc_sync().key_averages()
        # 同步外部全局分析器对象的键平均值统计
        outer_profile_rref.rpc_sync().key_averages()

    # 标记一个函数作为分布式初始化的入口
    @dist_init
    @dist_init
    def test_async_record_function_double_end_callbacks(self):
        # 确定每个进程的睡眠秒数
        num_sleep_seconds = 1
        # 如果进程的排名是1
        if self.rank == 1:
            # 验证调用函数两次是否会导致错误
            with _profile() as pf:
                # 创建一个记录函数 "foo" 的上下文
                with torch.autograd.profiler.record_function("foo") as rf:
                    # 调用异步 RPC 并传递睡眠函数作为参数
                    fut = rpc.rpc_async(
                        worker_name(0), my_sleep_func, args=(num_sleep_seconds,)
                    )
                    # 在 Future 上调用记录函数的结束回调
                    rf._call_end_callbacks_on_future(fut)
                    # 使用断言验证是否引发了 RuntimeError，错误信息为 "can only be called once."
                    with self.assertRaisesRegex(
                        RuntimeError, "can only be called once."
                    ):
                        # 再次在 Future 上调用记录函数的结束回调
                        rf._call_end_callbacks_on_future(fut)
                # 等待 Future 的完成
                fut.wait()

    @dist_init
    def test_async_record_function_legacy(self):
        # 测试旧版 _record_function 操作是否正常工作
        # 注意：这些操作是为了与 TorchScript 的向后兼容性而存在
        num_sleep_seconds = 1
        # 如果进程的排名是1
        if self.rank == 1:
            with _profile() as pf:
                try:
                    # 调用底层 _record_function_enter 进入记录函数 "foo"
                    handle = torch.ops.profiler._record_function_enter("foo", None)
                    # 调用异步 RPC 并传递睡眠函数作为参数
                    fut = rpc.rpc_async(
                        worker_name(0), my_sleep_func, args=(num_sleep_seconds,)
                    )
                    # 在 Future 上调用记录函数的结束回调
                    torch.ops.profiler._call_end_callbacks_on_jit_fut(handle, fut)
                finally:
                    # 调用底层 _record_function_exit 退出记录函数
                    torch.ops.profiler._record_function_exit(handle)

                # 等待 Future 的完成
                fut.wait()

    @dist_init
    def test_async_record_function_cbs_jit_call(self):
        # 如果进程的排名是1
        if self.rank == 1:
            with _profile() as pf:
                # 构建 RPC 的性能分析键值
                key = _build_rpc_profiling_key(
                    RPCExecMode.ASYNC,
                    torch._jit_internal._qualified_name(my_script_func),
                    "worker1",
                    "worker0",
                )
                # 创建一个记录函数的上下文，使用性能分析键值作为函数名
                with torch.autograd.profiler.record_function(key) as rf:
                    # 调用异步 RPC 并传递 my_script_func 和参数 torch.tensor(1)
                    fut = rpc.rpc_async(
                        worker_name(0), my_script_func, args=(torch.tensor(1),)
                    )
                    # 故意调用 record_function 的内部接口
                    fut = torch.ops.profiler._call_end_callbacks_on_jit_fut(rf.record, fut)
                # 等待 Future 的完成，并获取其结果
                result = fut.wait()
                # 验证性能分析 Future 返回的结果与 RPC Future 返回的结果是否相同
                expected = torch.add(torch.tensor(1), torch.tensor(1))
                self.assertEqual(result, expected)
            # 获取性能分析的事件列表
            events = pf.function_events
            # 获取特定函数事件，使用 my_script_func 的限定名
            rpc_event = get_function_event(
                events, torch._jit_internal._qualified_name(my_script_func)
            )
            # 断言特定函数在事件中的名称是否存在
            self.assertTrue(torch._jit_internal._qualified_name(my_script_func) in rpc_event.name)

    @dist_init
    def test_py_class_constructor(self):
        # 根据进程的排名计算目标进程的排名
        n = self.rank + 1
        dst_rank = n % self.world_size
        # 使用同步 RPC 调用 MyClass 构造函数，并传递参数 n
        ret = rpc.rpc_sync(worker_name(dst_rank), MyClass, args=(n,))
        # 验证返回结果中的属性 a 是否与 n 相等
        self.assertEqual(ret.a, n)
    # 使用 @dist_init 装饰器初始化分布式设置
    @dist_init
    # 定义测试类的实例方法，用于测试分布式环境下的 RPC 调用
    def test_py_class_instance_method(self):
        # 计算目标排名
        n = self.rank + 1
        dst_rank = n % self.world_size
        # 同步 RPC 调用，调用 MyClass 的实例方法 my_instance_method
        ret = rpc.rpc_sync(worker_name(dst_rank), MyClass(2).my_instance_method, args=(n,))
        # 断言返回值与预期调用结果相等
        self.assertEqual(ret, MyClass(2).my_instance_method(n))
    
    # 使用 @dist_init 装饰器初始化分布式设置
    @dist_init
    # 定义测试类的类方法，用于测试分布式环境下的 RPC 调用
    def test_py_class_method(self):
        # 计算目标排名
        n = self.rank + 1
        dst_rank = n % self.world_size
        # 同步 RPC 调用，调用 MyClass 的类方法 my_class_method
        ret = rpc.rpc_sync(worker_name(dst_rank), MyClass.my_class_method, args=(n, n + 1))
        # 断言返回值与预期调用结果相等
        self.assertEqual(ret, MyClass.my_class_method(n, n + 1))
    
    # 使用 @dist_init 装饰器初始化分布式设置
    @dist_init
    # 定义测试类的静态方法，用于测试分布式环境下的 RPC 调用
    def test_py_class_static_method(self):
        # 计算目标排名
        n = self.rank + 1
        dst_rank = n % self.world_size
        # 同步 RPC 调用，调用 MyClass 的静态方法 my_static_method
        ret = rpc.rpc_sync(worker_name(dst_rank), MyClass.my_static_method, args=(n + 10,))
        # 断言返回值与预期调用结果相等
        self.assertEqual(ret, MyClass.my_static_method(n + 10))
    
    # 使用 @dist_init 装饰器初始化分布式设置
    @dist_init
    # 定义测试类的多个异步调用方法，用于测试分布式环境下的 RPC 异步调用
    def test_py_multi_async_call(self):
        # 计算目标排名
        n = self.rank + 1
        dst_rank = n % self.world_size
        # 获取目标 worker 的信息
        dst_worker_info = rpc.get_worker_info(worker_name(dst_rank))
        # 发起第一个异步 RPC 调用，调用 MyClass 的静态方法 my_static_method
        fut1 = rpc.rpc_async(dst_worker_info, MyClass.my_static_method, args=(n + 10,))
        # 发起第二个异步 RPC 调用，调用 min 函数
        fut2 = rpc.rpc_async(dst_worker_info, min, args=(n, n + 1, n + 2))
        # 等待第一个异步调用完成，并断言返回值与预期调用结果相等
        self.assertEqual(fut1.wait(), MyClass.my_static_method(n + 10))
        # 等待第二个异步调用完成，并断言返回值与预期调用结果相等
        self.assertEqual(fut2.wait(), min(n, n + 1, n + 2))
    
    # 使用 @dist_init 装饰器初始化分布式设置
    @dist_init
    # 定义测试类的无返回结果方法，用于测试分布式环境下的 RPC 调用
    def test_py_no_return_result(self):
        # 计算目标排名
        n = self.rank + 1
        dst_rank = n % self.world_size
        # 同步 RPC 调用，调用 no_result 函数
        ret = rpc.rpc_sync(worker_name(dst_rank), no_result)
        # 断言返回值与预期调用结果相等
        self.assertEqual(ret, no_result())
    
    # 使用 @dist_init 装饰器初始化分布式设置
    @dist_init
    # 定义测试类的张量方法，用于测试分布式环境下的 RPC 调用
    def test_py_tensors(self):
        # 计算目标排名
        n = self.rank + 1
        dst_rank = n % self.world_size
        # 同步 RPC 调用，调用 my_tensor_function 函数
        ret = rpc.rpc_sync(
            worker_name(dst_rank),
            my_tensor_function,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        # 断言返回值与预期调用结果相等
        self.assertEqual(ret, my_tensor_function(torch.ones(n, n), torch.ones(n, n)))
    
    # 使用 @dist_init 装饰器初始化分布式设置
    @dist_init
    # 定义测试类的多个张量异步调用方法，用于测试分布式环境下的 RPC 异步调用
    def test_py_tensors_multi_async_call(self):
        # 初始化空的 future 列表
        futs = []
        n = self.rank + 1
        dst_rank = n % self.world_size
        # 循环发起 100 个异步 RPC 调用
        for i in range(100):
            fut = rpc.rpc_async(
                worker_name(dst_rank),
                my_tensor_function,
                args=(torch.ones(i, i), torch.ones(i, i)),
            )
            futs.append(fut)
    
        j = 0
        # 等待所有异步调用完成，并逐一断言返回值与预期调用结果相等
        for val in torch.futures.wait_all(futs):
            self.assertEqual(
                val, my_tensor_function(torch.ones(j, j), torch.ones(j, j))
            )
            j += 1
    # 定义一个测试方法，用于测试包含 PyTorch 张量的容器
    def test_py_tensors_in_container(self):
        # 计算当前进程的等级加一，用于确定目标进程的等级
        n = self.rank + 1
        dst_rank = n % self.world_size
        # 创建包含两个相同大小的全1张量的列表
        a = [torch.ones(n, n), torch.ones(n, n)]
        # 使用构建复杂张量的函数创建 TensorClass 的实例
        b = TensorClass(build_complex_tensors())
        # 创建包含两个相同大小的全1张量的字典
        c = {"foo": torch.ones(n, n), "bar": torch.ones(n, n)}
        # 使用 RPC 同步调用远程函数，传递参数 a, b, c 给目标进程
        ret = rpc.rpc_sync(
            worker_name(dst_rank), my_complex_tensor_function, args=(a, b, c)
        )
        # 断言 RPC 返回值与本地函数调用的返回值相等
        self.assertEqual(ret, my_complex_tensor_function(a, b, c))

    # 带有分布式初始化装饰器的测试方法，用于测试嵌套的 Pickle 操作
    @dist_init
    def test_py_nested_pickle(self):
        # 计算当前进程的等级加一，用于确定目标进程的等级
        n = self.rank + 1
        dst_rank = n % self.world_size
        # 使用 RPC 同步调用远程函数，传递参数 MyPickleClass 实例和一个全1张量给目标进程
        ret = rpc.rpc_sync(
            worker_name(dst_rank),
            run_nested_pickle,
            args=(MyPickleClass(), torch.ones(2, 2)),
        )
        # 创建 MyPickleClass 的实例
        m = MyPickleClass()
        # 调用其方法设置其属性为一个张量函数的结果
        m.set(my_tensor_function(torch.ones(2, 2), torch.ones(2, 2)))
        # 断言 RPC 返回值与本地函数调用的返回值相等
        self.assertEqual(ret, run_nested_pickle(m, torch.ones(2, 2)))

    # 带有分布式初始化装饰器的测试方法，用于测试用户函数抛出异常的情况
    @dist_init
    def test_py_function_exception(self):
        # 计算当前进程的等级加一，用于确定目标进程的等级
        n = self.rank + 1
        dst_rank = n % self.world_size
        # 使用 RPC 同步调用远程函数，传递参数 10 给目标进程，预期抛出 TypeError 异常
        with self.assertRaises(TypeError):
            ret = rpc.rpc_sync(worker_name(dst_rank), no_result, args=(10,))

    # 带有分布式初始化装饰器的测试方法，用于测试用户函数中引发异常的情况
    @dist_init
    def test_py_raise_in_user_func(self):
        # 使用 captured_output 上下文管理器捕获输出
        with captured_output() as (_, err):
            # 这个 barrier 防止主线程尚未进入上下文管理器时，远程函数已经运行
            initialize_pg(self.file_init_method, self.rank, self.world_size)
            dist.barrier()
            # 计算当前进程的等级加一，用于确定目标进程的等级
            n = self.rank + 1
            dst_rank = n % self.world_size
            # 使用 RPC 异步调用远程函数 raise_func
            fut = rpc.rpc_async(worker_name(dst_rank), raise_func)
            # 断言远程函数在等待期间引发 ValueError 异常，异常信息符合预期
            with self.assertRaisesRegex(ValueError, expected_err):
                fut.wait()
            # 这个 barrier 防止主线程退出上下文管理器时，远程函数尚未运行
            dist.barrier()

        # 验证训练器在运行函数时是否记录了错误
        stderr_lines = err.getvalue()
        self.assertTrue(expected_err in stderr_lines)

    # 带有分布式初始化装饰器的测试方法，用于测试用户函数中引发异常并且异常消息含有转义字符的情况
    @dist_init
    def test_py_raise_in_user_func_escaped_str(self):
        # 计算当前进程的等级加一，用于确定目标进程的等级
        n = self.rank + 1
        dst_rank = n % self.world_size
        # 使用 RPC 异步调用远程函数 raise_func_escape
        fut = rpc.rpc_async(worker_name(dst_rank), raise_func_escape)
        try:
            fut.wait()
        except ValueError as e:
            msg = str(e)
            # 确保消息中的换行符被取消转义，提供更好的错误表示
            self.assertEqual(msg, msg.encode("utf-8").decode("unicode_escape"))
        else:
            # 如果没有引发 ValueError 异常，则测试失败
            self.assertTrue(False, "expected raise_func_escape to raise ValueError.")

    # 带有分布式初始化装饰器的测试方法，用于测试嵌套 RPC 调用
    @dist_init
    def test_nested_rpc(self):
        # 调用私有方法 _nested_rpc，传递 nested_rpc 函数和一个全1张量加1作为参数
        self._nested_rpc(nested_rpc, torch.ones(2, 2) + 1)

    # 带有分布式初始化装饰器的测试方法，用于测试轻负载 RPC 测试
    @dist_init
    def test_stress_light_rpc(self):
        # 调用私有方法 _stress_test_rpc，传递 light_rpc 函数作为参数
        self._stress_test_rpc(light_rpc)

    # 带有分布式初始化装饰器的测试方法，用于测试重负载 RPC 测试
    @dist_init
    def test_stress_heavy_rpc(self):
        # 调用私有方法 _stress_test_rpc，传递 heavy_rpc 函数、重复次数为 20 和一个大小为 100x100 的张量作为参数
        self._stress_test_rpc(heavy_rpc, repeat=20, args=(torch.ones(100, 100),))
    def test_stress_heavy_rpc_torchscript(self):
        # 调用 _stress_test_rpc 方法，测试重负载的 TorchScript 远程过程调用
        self._stress_test_rpc(heavy_rpc_torchscript, repeat=20, args=(torch.ones(100, 100),))

    @dist_init
    def test_builtin_remote_ret(self):
        # 使用 _builtin_remote_ret 方法，测试内置远程调用返回值的情况
        self._builtin_remote_ret(
            torch.ones(2, 2),
            torch.ones(2, 2),
            torch.ones(2, 2) * 2
        )

    @dist_init
    def test_builtin_remote_self(self):
        # 使用 _builtin_remote_self 方法，测试内置远程调用自身的情况
        self._builtin_remote_self(
            torch.ones(2, 2),
            torch.ones(2, 2),
            torch.ones(2, 2) * 2
        )

    @staticmethod
    def _multi_args_fn(n, sparse=False):
        # 根据参数生成多个函数参数的元组
        if sparse:
            return (build_sparse_tensor(), build_sparse_tensor())
        else:
            return (torch.ones(n, n), torch.ones(n, n))

    @dist_init
    def test_multi_builtin_remote_ret(self):
        # 测试多个内置远程调用返回值的情况
        self._test_multi_remote_call(
            torch.add, False,
            args_fn=RpcTest._multi_args_fn
        )

    @dist_init
    def test_py_udf_remote(self):
        # 使用用户定义函数远程调用
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref = rpc.remote(
            worker_name(dst_rank),
            my_function,
            kwargs={"a": n, "b": n + 1, "c": n + 2},
        )
        # 断言远程调用的结果与预期的 my_function 结果相等
        self.assertEqual(rref.to_here(), my_function(n, n + 1, n + 2))

    @staticmethod
    def _multi_kwargs_fn(n, sparse=False):
        # 根据参数生成多个函数参数的字典
        if sparse:
            return {
                "a": build_sparse_tensor(),
                "b": build_sparse_tensor(),
                "c": build_sparse_tensor()
            }
        else:
            return {"a": torch.ones(n, n), "b": torch.ones(n, n), "c": torch.ones(n, n)}

    @dist_init
    def test_multi_py_udf_remote(self):
        # 测试多个用户定义函数远程调用
        self._test_multi_remote_call(
            my_function,
            False,
            kwargs_fn=RpcTest._multi_kwargs_fn
        )

    @dist_init
    def test_py_rref_args(self):
        # 测试使用参数的远程引用
        self._py_rref_args(
            torch.ones(2, 2),
            1,
            torch.ones(2, 2),
            2,
            torch.ones(2, 2) * 2 + 3)

    @dist_init
    def test_py_rref_args_user_share(self):
        # 测试使用用户共享参数的远程引用
        self._py_rref_args_user_share(
            torch.ones(2, 2),
            1,
            2,
            torch.ones(2, 2),
            3,
            4,
            torch.ones(2, 2) * 2 + 10
        )

    @dist_init
    def test_py_rpc_rref_args(self):
        # 测试使用参数的远程 RPC 引用
        self._py_rpc_rref_args(
            torch.ones(2, 2),
            1,
            2,
            torch.ones(2, 2),
            3,
            4,
            torch.ones(2, 2) * 2 + 10
        )

    @dist_init
    def test_nested_remote(self):
        # 测试嵌套远程调用
        self._nested_remote(
            nested_remote,
            torch.ones(2, 2) + 3
        )

    @dist_init
    def test_nested_rref(self):
        # 测试嵌套远程引用
        self._nested_rref(
            nested_rref,
            torch.ones(2, 2) + 1,
            torch.ones(2, 2) + 2
        )
    def test_nested_rref_stress(self):
        # 调用 _nested_rref_stress 方法，传入 nested_rref 函数和两个创建的 2x2 全 1 和全 2 的张量作为参数
        self._nested_rref_stress(
            nested_rref,
            torch.ones(2, 2) + 1,
            torch.ones(2, 2) + 2
        )

    @dist_init
    def test_multi_layer_nested_async_rpc(self):
        # 这个测试会立即退出，但会有一系列的异步 RPC 调用链。终止算法应该能正确检测到这些消息。
        # 否则，某些对等节点可能会过早退出，导致其他节点出现超时错误或连接关闭错误。
        ttl = 20
        n = self.rank + 1
        dst_rank = n % self.world_size

        # 调用 multi_layer_nested_async_rpc 方法，传入目标排名、全局大小和超时时间作为参数
        multi_layer_nested_async_rpc(dst_rank, self.world_size, ttl)

    @dist_init
    def test_remote_with_exception(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        # 检查对其他工作节点的引用
        rref = rpc.remote(worker_name(dst_rank), raise_func)
        # 断言会抛出 ValueError 异常
        with self.assertRaises(ValueError):
            rref.to_here()
        # 检查对自身的引用
        rref = rpc.remote(worker_name(self.rank), no_result, args=(10,))
        # 断言会抛出 TypeError 异常
        with self.assertRaises(TypeError):
            rref.to_here()

    @dist_init
    def test_rpc_return_rref(self):
        n = self.rank + 1
        dst_rank1 = n % self.world_size
        dst_rank2 = (n + 1) % self.world_size
        # 同步 RPC 调用，将 rpc_return_rref 函数作为远程调用目标，并传入目标排名作为参数
        rref = rpc.rpc_sync(
            worker_name(dst_rank1),
            rpc_return_rref,
            args=(worker_name(dst_rank2),),
        )
        # 断言返回的远程引用的值应为全 1 的 2x2 张量加 1
        self.assertEqual(rref.to_here(), torch.ones(2, 2) + 1)

    @dist_init
    def test_rref_forward_chain(self):
        ttl = 8
        n = self.rank + 1
        dst_rank = n % self.world_size

        # 创建远程引用，调用 torch.add 函数，并传入全 1 的 n x n 张量和标量 1 作为参数
        rref = rpc.remote(
            worker_name(dst_rank), torch.add, args=(torch.ones(n, n), 1)
        )

        # 调用 rref_forward_chain 方法，传入目标排名、全局大小、远程引用和超时时间作为参数
        ret_rref = rref_forward_chain(dst_rank, self.world_size, rref, ttl)

        # 循环检查返回的远程引用链，应当为长度为 1 的列表，直到达到超时时间
        for i in range(ttl):
            self.assertEqual(len(ret_rref), 1)
            ret_rref = ret_rref[0].to_here()

        # 检查最终返回的结果应为全 1 的 n x n 张量加 1
        ret = ret_rref
        self.assertEqual(ret, torch.add(torch.ones(n, n), 1))

    @dist_init
    def test_local_rref_no_fork(self):
        # 创建本地远程引用，其值为 35
        local_rref = RRef(35)
        # 断言本地远程引用的本地值为 35
        self.assertEqual(local_rref.local_value(), 35)
    def test_local_value_not_on_owner(self):
        # 确保在非拥有节点上调用 local_value() 时会抛出错误消息。
        # 计算下一个节点的排名，确保循环在集群大小内
        next_rank = (self.rank + 1) % self.world_size
        # 在远程节点上调用 torch.add 函数，创建远程引用对象 rref
        rref = rpc.remote(
            worker_name(next_rank), torch.add, args=(torch.ones(1), torch.ones(1))
        )
        # 使用 assertRaisesRegex 确保抛出特定异常类型和消息内容
        with self.assertRaisesRegex(
            RuntimeError, (
                fr"For UserRRef\(rref_id=GloballyUniqueId\(created_on={self.rank}, local_id=0\), "
                fr"fork_id=GloballyUniqueId\(created_on={self.rank}, local_id=1\)\), "
                r"can't call localValue\(\) on user "
                fr"WorkerInfo\(id={self.rank}, name={worker_name(self.rank)}\). "
                fr"Call it on owner WorkerInfo\(id={next_rank}, name={worker_name(next_rank)}\)"
            )
        ):
            # 在 rref 对象上调用 local_value() 方法
            rref.local_value()

    @dist_init
    def test_return_local_rrefs(self):
        # 根据当前进程的排名计算目标节点的排名
        n = self.rank + 1
        dst_rank = n % self.world_size

        # 通过 RPC 调用 worker_name(dst_rank) 节点上的 get_rref_list 函数
        rref_list = rpc.rpc_sync(
            worker_name(dst_rank), get_rref_list, args=([1, 2, 3],)
        )

        # 遍历 rref_list 中的每个远程引用对象 rref
        for rref in rref_list:
            # 同步调用 rref.owner() 所指定的节点上的 _call_method_on_rref 方法
            rpc.rpc_sync(
                rref.owner(),
                _call_method_on_rref,
                args=(MyClass.increment_value, rref, 10),
            )

        # 构造一个列表 rets，包含所有 rref_list 中每个 rref 的远程方法调用结果
        rets = [
            rpc.rpc_sync(
                rref.owner(), _call_method_on_rref, args=(MyClass.get_value, rref)
            )
            for rref in rref_list
        ]

        # 使用 self.assertEqual 检查 rets 是否等于预期的值 [11, 12, 13]
        self.assertEqual(rets, [11, 12, 13])
    # 定义测试函数，用于验证远程引用对象的类型获取功能，接受一个布尔值参数用于指定是否阻塞模式

    def _test_rref_type(self, blocking):

        # 内部函数：判断事件列表中是否有符合预期名称的 RPC 事件
        def launched_rpc(events):
            expected_name = f"rpc_{RPCExecMode.ASYNC.value}#_rref_typeof_on_owner"
            return any(e.name.startswith(expected_name) for e in events)

        # 计算目标 worker 名称
        dst = worker_name((self.rank + 1) % self.world_size)
        
        # 在目标 worker 上异步执行 torch.add 操作，并获取远程引用对象
        rref = rpc.remote(dst, torch.add, args=(torch.ones(2), 1))

        # 使用 _profile 上下文进行性能分析
        with _profile() as p:
            # 获取远程引用对象的类型，根据 blocking 参数决定是否阻塞等待结果
            t = rref._get_type(blocking=blocking)
            if not blocking:
                t = t.wait()

        # 断言是否有符合预期的 RPC 被调用
        self.assertTrue(launched_rpc(p.function_events))

        # 预期的类型应为 torch.Tensor 类型
        expected_type = type(torch.ones(2))
        self.assertEqual(t, expected_type)

        # 初始化一个空列表用于存储 future 对象
        futs = []

        # 内部函数：验证 future 对象是否符合预期类型
        def verify(fut):
            self.assertEqual(fut.value(), expected_type)

        # 使用 _profile 上下文进行性能分析
        with _profile() as p:
            for _ in range(10):
                # 获取远程引用对象的类型，根据 blocking 参数决定是否阻塞等待结果
                t = rref._get_type(blocking=blocking)
                if not blocking:
                    # 对于非阻塞模式，添加 future 到列表，并为 future 添加完成后的回调函数
                    futs.append(t)
                    t.add_done_callback(verify)
                    t = t.wait()
                # 断言获取的类型与预期类型一致
                self.assertEqual(t, expected_type)

        # 如果非阻塞模式下
        if not blocking:
            # 注意：所有使用 blocking=False 的缓存调用都返回相同的原始 future 对象
            first_fut = futs[0]
            for f in futs[1:]:
                self.assertTrue(f is first_fut)

        # 断言不会再次触发新的 RPC 调用，除非是第一次调用
        self.assertFalse(launched_rpc(p.function_events))

        # 最终再次确认获取的类型与预期类型一致
        self.assertEqual(t, type(torch.ones(2)))

        # 在目标 worker 上异步执行 MyClass 类初始化，并获取远程引用对象的类型
        rref = rpc.remote(dst, MyClass, args=(0,))
        rref_type = rref._get_type(blocking=blocking)

        # 根据 blocking 参数决定是否阻塞等待结果
        if not blocking:
            rref_type = rref_type.wait()

        # 断言获取的远程引用对象类型与预期类型 MyClass 一致
        self.assertEqual(rref_type, MyClass)


    # 测试阻塞模式下的远程引用对象类型获取功能
    def test_rref_type_blocking(self):
        self._test_rref_type(blocking=True)

    # 测试非阻塞模式下的远程引用对象类型获取功能
    def test_rref_type_non_blocking(self):
        self._test_rref_type(blocking=False)


    # 使用 @dist_init 装饰器初始化分布式环境
    @dist_init

    # 定义带有错误处理的测试函数，用于验证远程引用对象类型获取功能，接受一个布尔值参数用于指定是否阻塞模式
    def _test_rref_type_with_error(self, blocking):

        # 计算目标 worker 名称
        dst = worker_name((self.rank + 1) % self.world_size)
        
        # 在目标 worker 上异步执行 raise_func 函数，设置 10 毫秒超时
        rref = rpc.remote(dst, raise_func)
        
        # 如果是阻塞模式：内联触发错误
        if blocking:
            with self.assertRaisesRegex(ValueError, "Expected error"):
                rref._get_type(blocking=blocking)
        else:
            # 如果是非阻塞模式：立即返回 future 对象，并在等待时阻塞
            fut = rref._get_type(blocking=blocking)
            with self.assertRaisesRegex(ValueError, "Expected error"):
                fut.wait()


    # 测试带有错误处理的阻塞模式下的远程引用对象类型获取功能
    def test_rref_type_with_error_blocking(self):
        self._test_rref_type_with_error(blocking=True)

    # 测试带有错误处理的非阻塞模式下的远程引用对象类型获取功能
    def test_rref_type_with_error_non_blocking(self):
        self._test_rref_type_with_error(blocking=False)
    def _test_rref_type_owner(self, blocking):
        # 创建一个包含值为2的张量的远程引用对象
        rref = RRef(torch.ones(2) + 1)
        # 获取远程引用对象的类型
        rref_type = rref._get_type(blocking=blocking)
        # 如果非阻塞模式，则等待类型获取完成
        if not blocking:
            rref_type = rref_type.wait()
        # 断言远程引用对象的类型与torch.ones(2)的类型相同
        self.assertEqual(rref_type, type(torch.ones(2)))

        # 创建一个包含自定义类实例的远程引用对象
        rref = RRef(MyClass(0))
        # 获取远程引用对象的类型
        rref_type = rref._get_type(blocking=blocking)
        # 如果非阻塞模式，则等待类型获取完成
        if not blocking:
            rref_type = rref_type.wait()
        # 断言远程引用对象的类型为MyClass
        self.assertEqual(rref_type, MyClass)

    def test_rref_type_owner_blocking(self):
        # 测试阻塞模式下的远程引用对象类型获取
        self._test_rref_type_owner(blocking=True)

    def test_rref_type_owner_non_blocking(self):
        # 测试非阻塞模式下的远程引用对象类型获取
        self._test_rref_type_owner(blocking=False)

    @staticmethod
    def _slow_add(x, y):
        # 模拟一个耗时1秒的函数
        time.sleep(1)
        return x + y

    @dist_init
    def test_rref_type_slow_init(self):
        # 获取下一个工作节点的名称
        dst = worker_name((self.rank + 1) % self.world_size)
        # 在远程工作节点上调用_slow_add函数，返回远程引用对象
        rref = rpc.remote(dst, RpcTest._slow_add, args=(torch.ones(2), 1))
        # 断言远程引用对象的类型为torch.ones(2)的类型
        self.assertEqual(rref._get_type(), type(torch.ones(2)))

    @dist_init
    def test_owner_equality(self):
        # 创建两个包含整数40和50的远程引用对象
        a = RRef(40)
        b = RRef(50)

        # 计算另一个工作节点的排名
        other_rank = (self.rank + 1) % self.world_size
        # 创建一个远程对象，代表另一个工作节点上的torch.add函数
        other_a = rpc.remote(
            worker_name(other_rank), torch.add, args=(torch.ones(1), 1)
        )
        # 创建另一个远程对象，代表另一个工作节点上的torch.add函数
        other_b = rpc.remote(
            worker_name(other_rank), torch.add, args=(torch.ones(1), 1)
        )
        # 等待远程对象传输到本地完成
        other_a.to_here()
        other_b.to_here()

        # 断言a的所有者不等于23
        self.assertNotEqual(a.owner(), 23)
        # 断言other_a和other_b的所有者相同
        self.assertEqual(other_a.owner(), other_b.owner())
        # 断言a和other_a的所有者不相同
        self.assertNotEqual(a.owner(), other_a.owner())
        # 断言other_a的所有者与自身相同
        self.assertEqual(other_a.owner(), other_a.owner())
        # 断言other_a和other_b的所有者相同
        self.assertEqual(other_a.owner(), other_b.owner())
        # 断言a的所有者与自身相同
        self.assertEqual(a.owner(), a.owner())
        # 断言a和b的所有者相同
        self.assertEqual(a.owner(), b.owner())
        # 断言a的所有者与rpc.get_worker_info()返回的信息相同
        self.assertEqual(a.owner(), rpc.get_worker_info())
        # 创建一个空字典x，并将a和other_a作为键和值添加到字典中
        x = {}
        x[a.owner()] = a
        x[other_a.owner()] = other_a
        # 断言x中a的所有者对应的值为a
        self.assertEqual(x[a.owner()], a)
        # 断言x中b的所有者对应的值为a
        self.assertEqual(x[b.owner()], a)
        # 断言x中other_a的所有者对应的值为other_a
        self.assertEqual(x[other_a.owner()], other_a)
        # 断言x中other_b的所有者对应的值为other_a
        self.assertEqual(x[other_b.owner()], other_a)
        # 断言字典x的长度为2
        self.assertEqual(len(x), 2)

    @dist_init
    def test_pass_local_rrefs(self):
        # 计算本地进程的排名加1
        n = self.rank + 1
        # 计算目标排名
        dst_rank = n % self.world_size
        # 获取目标工作节点的名称
        dst_worker = worker_name(dst_rank)

        # 创建一个包含整数40的远程引用对象
        rref = RRef(40)
        # 使用RPC同步调用将rref添加到值50上，并断言返回结果为90
        self.assertEqual(
            rpc.rpc_sync(dst_worker, add_rref_to_value, args=(rref, 50)), 90
        )
        # 使用RPC异步调用将rref添加到值50上，并等待结果返回，并断言返回结果为90
        self.assertEqual(
            rpc.rpc_async(dst_worker, add_rref_to_value, args=(rref, 50)).wait(), 90
        )
        # 在远程工作节点上调用add_rref_to_value函数，并等待结果返回到本地，并断言返回结果为90
        self.assertEqual(
            rpc.remote(dst_worker, add_rref_to_value, args=(rref, 50)).to_here(), 90
        )
    def test_remote_same_worker(self):
        # 计算目标 worker 的 rank
        n = self.rank + 1
        dst_rank = n % self.world_size
        # 创建远程引用对象 rref_a，调用 torch.add 在目标 worker 上执行操作
        rref_a = rpc.remote(
            worker_name(dst_rank), torch.add, args=(torch.ones(n, n), 2)
        )
        # 创建远程引用对象 rref_b，调用 torch.add 在目标 worker 上执行操作
        rref_b = rpc.remote(
            worker_name(dst_rank), torch.add, args=(torch.ones(n, n), 1)
        )
        # 创建远程引用对象 rref_c，调用自定义函数 my_rref_function 在目标 worker 上执行操作
        rref_c = rpc.remote(
            worker_name(dst_rank), my_rref_function, args=(rref_a, rref_b)
        )
        # 断言远程引用对象 rref_c 的值为 torch.ones(n, n) + 4
        self.assertEqual(rref_c.to_here(), torch.ones(n, n) + 4)

    @dist_init(setup_rpc=True)
    def test_call_method_on_rref(self):
        """
        Tests that it is possible to call an instance method on a remote object
        by using rref.owner() as destination of the call.
        """
        vals = [10, 2, 5, 7]
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = worker_name(dst_rank)

        # 创建一个远程对象 rref，使用 MyClass 的构造函数和参数 vals[0]
        rref = rpc.remote(dst_worker, MyClass, args=(vals[0],))

        # 同步调用远程对象 rref 上的 MyClass.increment_value 方法，并传入 vals[1] 作为参数
        rpc.rpc_sync(
            rref.owner(),
            _call_method_on_rref,
            args=(MyClass.increment_value, rref, vals[1]),
        )

        # 异步调用远程对象 rref 上的 MyClass.increment_value 方法，并传入 vals[2] 作为参数
        rpc.rpc_async(
            rref.owner(),
            _call_method_on_rref,
            args=(MyClass.increment_value, rref, vals[2]),
        ).wait()

        # 调用远程对象 rref 上的 MyClass.increment_value 方法，并传入 vals[3] 作为参数，等待执行完成
        rpc.remote(
            rref.owner(),
            _call_method_on_rref,
            args=(MyClass.increment_value, rref, vals[3]),
        ).to_here()

        # 同步调用获取远程对象 rref 上的 MyClass.get_value 方法的结果
        result = rpc.rpc_sync(
            dst_worker, _call_method_on_rref, args=(MyClass.get_value, rref)
        )

        # 断言获取的结果与 vals 的总和相等
        self.assertEqual(result, sum(vals))

    # 注意 `rpc.api.shutdown()` 访问 `_delete_all_user_and_unforked_owner_rrefs`
    # 通过 `torch.distributed.rpc.api`，因此对 `torch.distributed.rpc._delete_all_user_and_unforked_owner_rrefs` 的
    # 修补不会起作用。
    @mock.patch.object(torch.distributed.rpc.api, "_delete_all_user_and_unforked_owner_rrefs")
    def _test_rref_leak(self, _mock_delete_all_user_and_unforked_owner_rrefs, ignore_leak):
        # 初始化 RPC，设置当前 worker 的名称、后端、rank、world_size 和 RPC 后端选项
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        # 初始化进程组，使用指定的初始化方法、当前 rank 和 world_size
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        # 等待所有初始化完成
        dist.barrier()

        # 创建一个远程对象 rref，在下一个 rank 的 worker 上执行 torch.add 操作
        rref = rpc.remote(
            worker_name((self.rank + 1) % self.world_size),
            torch.add,
            args=(torch.ones(2, 2), 1),
        )

        import torch.distributed.rpc.api as api

        # 如果忽略 RRef 泄漏，则设置 `_ignore_rref_leak` 为 True 并关闭 RPC
        if ignore_leak:
            api._ignore_rref_leak = True
            rpc.shutdown(graceful=True)
        else:
            # 否则设置 `_ignore_rref_leak` 为 False，并使用断言捕获 RuntimeError 异常，提示 RRef 泄漏
            api._ignore_rref_leak = False
            with self.assertRaisesRegex(RuntimeError, "Leaking RRef"):
                rpc.shutdown(graceful=True)
    @dist_init(setup_rpc=False)
    # 使用 @dist_init 装饰器初始化分布式环境，但不设置 RPC
    def test_rref_leak(self):
        # 调用 _test_rref_leak 方法，测试 RRef 的内存泄漏情况，忽略泄漏设置为 False
        self._test_rref_leak(ignore_leak=False)

    @dist_init(setup_rpc=False)
    # 使用 @dist_init 装饰器初始化分布式环境，但不设置 RPC
    def test_ignore_rref_leak(self):
        # 调用 _test_rref_leak 方法，测试 RRef 的内存泄漏情况，忽略泄漏设置为 True
        self._test_rref_leak(ignore_leak=True)

    @dist_init
    # 使用 @dist_init 装饰器初始化分布式环境，设置 RPC
    def test_rref_str(self):
        # 创建 RRef 对象 rref1，用于测试其 __str__ 方法的输出
        rref1 = RRef(self.rank)
        id_class = "GloballyUniqueId"
        # 断言 rref1.__str__() 方法的输出结果
        self.assertEqual(
            f"OwnerRRef({id_class}(created_on={self.rank}, local_id=0))", rref1.__str__()
        )

        # 将任务发送到下一个节点 dst_rank，并创建 RRef 对象 rref2
        rref2 = rpc.remote(
            worker_name(dst_rank), torch.add, args=(torch.ones(2, 2), 1)
        )
        # 断言 rref2.__str__() 方法的输出结果
        self.assertEqual(
            rref2.__str__(),
            f"UserRRef(RRefId = {id_class}(created_on={self.rank}, local_id=1), "
            f"ForkId = {id_class}(created_on={self.rank}, local_id=2))",
        )

    @dist_init
    # 使用 @dist_init 装饰器初始化分布式环境，设置 RPC
    def test_rref_get_future(self):
        # 测试能否获取远端 RRef 创建对应的 Future
        if self.rank == 0:
            # 测试内置函数作为远程调用任务
            rref = rpc.remote(worker_name(1), torch.add, args=(1, 1))
            rref.to_here()
            fut = rref._get_future()
            # 断言 fut 的类型为 torch._C.Future
            self.assertIsInstance(fut, torch._C.Future)

            # 测试自定义函数 foo_add 作为远程调用任务
            rref = rpc.remote(worker_name(1), foo_add, args=())
            rref.to_here()
            fut = rref._get_future()
            # 断言 fut 的类型为 torch._C.Future
            self.assertIsInstance(fut, torch._C.Future)

            # 测试脚本函数 my_script_func 作为远程调用任务
            rref = rpc.remote(worker_name(1), my_script_func, args=(torch.tensor(1), ))
            rref.to_here()
            fut = rref._get_future()
            # 断言 fut 的类型为 torch._C.Future
            self.assertIsInstance(fut, torch._C.Future)
    def test_rref_context_debug_info(self):
        # This test checks local states that are modified by remote workers.
        # This means that we would need barrier before and after every check.
        # The barrier before the check makes sure that all previous states are
        # cleared globally, the barrier after ensures that no following states
        # change gets into the current check.
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        # Check 1: local RRef does not update owners_ map or add a pending user.
        #################################################

        rref1 = RRef(self.rank)

        # don't need a barrier here as local RRef is handled by this thread
        info = _rref_context_get_debug_info()
        self.assertIn("num_owner_rrefs", info)
        self.assertIn("num_pending_users", info)
        # RRef on local value is not added to context until shared across RPC
        self.assertEqual(0, int(info["num_owner_rrefs"]))
        self.assertEqual(0, int(info["num_pending_users"]))
        # barrier after the check 1
        dist.barrier()

        # Check 2: Sharing RRef as an arg should update owners_ map
        ###########################################################

        dst_rank = (self.rank + 1) % self.world_size
        rpc.rpc_sync(worker_name(dst_rank), set_global_rref, args=(rref1,))

        # barrier before check 2
        wait_until_pending_futures_and_users_flushed()
        dist.barrier()

        info = _rref_context_get_debug_info()
        self.assertIn("num_owner_rrefs", info)
        self.assertEqual(1, int(info["num_owner_rrefs"]))
        # no pending users since the fork is finished
        self.assertEqual(0, int(info["num_pending_users"]))
        # barrier after check 2
        dist.barrier()

        # clear states for check 2
        rpc.rpc_sync(worker_name(dst_rank), clear_global_rref)

        # Wait for owner rref to be cleared.
        while int(info["num_owner_rrefs"]) != 0:
            info = _rref_context_get_debug_info()
            time.sleep(0.1)
        dist.barrier()

        # Check 3: rpc.remote call should update owners_ map
        ####################################################
        rref2 = rpc.remote(
            worker_name(dst_rank), torch.add, args=(torch.ones(2, 2), 1)
        )
        rref3 = rpc.remote(
            worker_name(dst_rank), torch.add, args=(torch.ones(2, 2), 1)
        )
        rref2.to_here()
        rref3.to_here()

        # barrier before check 3
        wait_until_pending_futures_and_users_flushed()
        dist.barrier()

        info = _rref_context_get_debug_info()
        self.assertIn("num_owner_rrefs", info)
        self.assertEqual(2, int(info["num_owner_rrefs"]))
        # no pending users since the fork is finished
        self.assertEqual(0, int(info["num_pending_users"]))

        # barrier after check 3
        dist.barrier()
    def test_disable_gil_profiling(self):
        # 测试 rpc.enable_gil_profiling(false) 是否会导致 GIL 等待时间不被记录。

        # 默认情况下应该禁用 GIL 分析。
        # 计算目标 rank，确保环绕世界的范围内
        dst_rank = (self.rank + 1) % self.world_size
        # 使用 RPC 同步调用，将两个张量相加
        rpc.rpc_sync(
            worker_name(dst_rank), torch.add, args=(torch.ones(1), torch.ones(1))
        )
        # 获取当前 RPC agent 的调试信息
        info = rpc.api._get_current_rpc_agent().get_debug_info()
        # 断言确保键 "agent.gil_average_wait_time_us" 不存在
        self.assertRaises(KeyError, lambda: info["agent.gil_average_wait_time_us"])
        # 启用 GIL 分析
        rpc.enable_gil_profiling(True)
        # 再次进行 RPC 同步调用，将两个张量相加
        rpc.rpc_sync(
            worker_name(dst_rank), torch.add, args=(torch.ones(1), torch.ones(1))
        )
        # 获取更新后的调试信息
        info = rpc.api._get_current_rpc_agent().get_debug_info()
        # 断言确保键 "agent.gil_average_wait_time_us" 存在
        self.assertIn("agent.gil_average_wait_time_us", info)

    @dist_init(setup_rpc=False)
    def test_local_shutdown(self):
        # 测试我们能否启动 RPC 并立即本地关闭，而不发送任何消息。

        # 初始化 RPC，使用当前 rank 和 RPC 后端选项
        rpc.init_rpc(
            name="worker%d" % self.rank,
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )
        # 传入 graceful=False 确保我们不等待其他 worker
        # 执行 RPC 关闭
        rpc.shutdown(graceful=False)

    @dist_init
    def test_debug_info(self):
        # 在此测试中只检测键，值应该在各个模块的调试信息测试中涵盖。

        # 导入分布式自动微分模块
        import torch.distributed.autograd as dist_autograd

        # 获取调试信息
        info = _get_debug_info()
        # 获取远程引用调试信息
        rref_info = _rref_context_get_debug_info()
        # 获取当前 RPC agent 的调试信息
        agent_info = rpc.api._get_current_rpc_agent().get_debug_info()
        # 获取自动微分调试信息
        autograd_info = dist_autograd._get_debug_info()
        # 找到所有三者共有的键
        common_keys = rref_info.keys() & agent_info.keys() & autograd_info.keys()
        # 断言共有的键的数量为零
        self.assertEqual(0, len(common_keys))
        # 创建预期字典，整合所有调试信息
        expected = {}
        expected.update(rref_info)
        expected.update(agent_info)
        expected.update(autograd_info)
        # 注意：在 Python 3.6+ 中才会保留键的顺序，因此这里手动检查键是否相等。
        # 检查预期的每个键是否在 info 的键中
        for key in expected.keys():
            self.assertIn(key, info.keys())

        # 检查 info 的每个键是否在预期中的键中
        for key in info.keys():
            self.assertIn(key, expected.keys())

    @dist_init(setup_rpc=False)
    @skip_but_pass_in_sandcastle_if(
        IS_MACOS,
        "Test is flaky on MacOS since libuv error handling is not as robust as TCP",
    )
    def test_handle_send_exceptions(self):
        # 测试如果被调用节点已经关闭，我们应该抛出适当的异常，而不是简单崩溃。
        rpc.init_rpc(
            name="worker%d" % self.rank,  # 初始化 RPC，设置节点名称
            backend=self.rpc_backend,     # 指定 RPC 后端
            rank=self.rank,               # 设置当前节点的排名
            world_size=self.world_size,   # 设置整个集群的节点数
            rpc_backend_options=self.rpc_backend_options,  # RPC 后端选项
        )
        rpc._set_rpc_timeout(10)  # 设置 RPC 超时时间为 10 秒
        # 需要此屏障来确保某些工作节点在其他节点启动之前不会退出。
        initialize_pg(self.file_init_method, self.rank, self.world_size)  # 初始化进程组
        dist.barrier()  # 同步所有节点，等待所有节点完成初始化
        if self.rank == 1:
            dst_rank = (self.rank + 1) % self.world_size  # 计算目标节点的排名
            dst_worker = worker_name(dst_rank)  # 获取目标节点的名称
            # 允许目标工作节点在不加入的情况下退出
            error_str = self.get_shutdown_error_regex()  # 获取用于检测关闭错误的正则表达式
            wait_until_node_failure(dst_rank, error_str)  # 等待直到目标节点发生故障
            fut = rpc.rpc_async(dst_worker, torch.add, args=(torch.ones(1), 3))
            # 由于关闭顺序未明确定义，因此我们可以看到任何在 get_shutdown_error_regex 中定义的错误消息。
            with self.assertRaisesRegex(RuntimeError, error_str):
                fut.wait()
        # 非优雅地关闭所有工作节点。
        rpc.shutdown(graceful=False)

    @dist_init
    def test_deadlock(self):
        # 这个测试是从 https://github.com/pytorch/pytorch/issues/45089 复制过来的
        if self.rank == 1:
            dst1 = worker_name((self.rank + 1) % self.world_size)  # 计算目标节点的名称
            x = torch.ones(2)
            y = torch.ones(2)
            rpc.rpc_async(dst1, RpcTest._slow_add, args=(x, y), timeout=15).wait()  # 异步 RPC 调用

        dist_initialized = dist.is_initialized()  # 检查分布式是否已初始化
        if not dist_initialized:
            dist.init_process_group(
                backend="gloo",  # 指定后端为 gloo
                init_method=self.file_init_method,  # 初始化方法
                rank=self.rank,  # 设置当前节点的排名
                world_size=self.world_size,  # 设置整个集群的节点数
            )

    @dist_init(setup_rpc=False)
    def test_local_shutdown_with_rpc(self):
        # 测试我们能否启动 RPC，发送 RPC，并执行本地关闭操作。

        # 初始化 RPC，使用给定的名称、后端、排名、总大小和选项
        rpc.init_rpc(
            name="worker%d" % self.rank,
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        # 计算目标排名以发送 RPC
        n = self.rank + 1
        dst_rank = n % self.world_size

        # 同步 RPC 调用到目标 worker_name(dst_rank)，使用 torch.add 函数
        rpc.rpc_sync(
            worker_name(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )

        # 需要一个 barrier 来确保所有 RPC 都被处理。
        # 否则，一些 RPC 可能会超时，因为接收端已终止。
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        dist.barrier()

        # 使用 graceful=False 参数来确保我们不等待其他 worker。
        rpc.shutdown(graceful=False)

    @dist_init(setup_rpc=False)
    def test_set_and_get_default_rpc_timeout(self):
        timeout = 0.5

        # 当访问 self.rpc_backend_options 时，构造一个新的 `RpcBackendOptions`。

        # 设置 RPC 超时时间为 timeout
        rpc_backend_options = self.rpc_backend_options
        rpc_backend_options.rpc_timeout = timeout

        # 初始化 RPC，使用当前 worker 的名称、后端、排名、总大小和新的选项
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=rpc_backend_options,
        )

        # 获取当前 RPC 的超时时间
        set_timeout = rpc.get_rpc_timeout()

        # 断言设置的超时时间与获取的超时时间相等
        self.assertEqual(timeout, set_timeout)

        # 关闭 RPC
        rpc.shutdown()

    @dist_init
    def test_default_timeout_used(self):
        """
        Tests that if no timeout is passed into rpc_async and rpc_sync, then the
        default timeout is used.
        """
        # 计算目标排名，确保循环的封闭性
        dst_rank = (self.rank + 1) % self.world_size
        # 设置RPC超时为1毫秒
        rpc._set_rpc_timeout(0.001)  # 1 ms
        # 期望超时的未来操作，应标记异常以指示超时
        futs = [
            rpc.rpc_async(worker_name(dst_rank), my_sleep_func, args=())
            for _ in range(10)
        ]
        # 准备捕获预期的超时错误正则表达式
        expected_error = self.get_timeout_error_regex()
        # 对每个未来操作进行检查，确保它们抛出预期的超时异常
        for fut in futs:
            with self.assertRaisesRegex(RuntimeError, expected_error):
                fut.wait()

        # 确保如果设置新的超时时间，旧的未来操作不会超时，但新的未来操作会超时
        rpc._set_rpc_timeout(200)  # 200 seconds
        # 创建一个长时间运行的RPC操作
        fut1 = rpc.rpc_async(worker_name(dst_rank), my_sleep_func, args=(1,))
        # 现在，设置一个短超时时间
        rpc._set_rpc_timeout(0.001)
        # fut2 应该超时，fut1 不应该超时
        fut2 = rpc.rpc_async(worker_name(dst_rank), my_sleep_func, args=(1,))
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut2.wait()
        fut1.wait()

        # 零超时意味着无限超时，因此未来操作应该运行到完成
        rpc._set_rpc_timeout(0)
        rpc.rpc_async(worker_name(dst_rank), my_sleep_func, args=()).wait()

        # 重置为默认超时时间，以便关闭消息可以干净地处理
        rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)
    def test_rpc_timeouts(self):
        # TODO: enable timeouts for rpc.remote/RRef (https://github.com/pytorch/pytorch/issues/33803)
        
        # 计算目标 rank，使其为当前 rank 加一，循环到 world_size 以内
        dst_rank = (self.rank + 1) % self.world_size
        # 获取目标 worker 的名称
        dst_worker = worker_name(dst_rank)
        # 设置超时时间为 100 毫秒
        timeout = 0.1  # 100 ms
        # 获取超时时的预期错误信息的正则表达式
        expected_error = self.get_timeout_error_regex()
        
        # 测试异步 UDF
        fut = rpc.rpc_async(dst_worker, my_sleep_func, args=(1,), timeout=timeout)
        # 确保异步操作抛出预期的 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()

        # 确保在没有超时且使用默认的 RPC 超时时，能正常运行完成
        rpc.rpc_async(dst_worker, my_sleep_func, args=(1,)).wait()

        # 测试同步 UDF
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rpc.rpc_sync(dst_worker, my_sleep_func, args=(1,), timeout=timeout)

        # 确保在没有超时且使用默认的 RPC 超时时，能正常运行完成
        rpc.rpc_sync(dst_worker, my_sleep_func, args=(1,))

        # 如果为 RPC 设置了默认超时时间，应该遵守该超时时间，
        # 虽然可以通过 API 传入不同的超时时间进行覆盖
        rpc._set_rpc_timeout(0.001)
        fut = rpc.rpc_async(dst_worker, my_sleep_func, args=(1,))
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rpc.rpc_sync(dst_worker, my_sleep_func, args=(1,))

        # 由于我们覆盖了超时时间，RPC 应该能够正常完成
        rpc.rpc_async(dst_worker, my_sleep_func, args=(1,), timeout=5).wait()
        rpc.rpc_sync(dst_worker, my_sleep_func, args=(1,), timeout=5)
        
        # 传入零超时时间应确保 RPC 不会超时
        rpc.rpc_async(dst_worker, my_sleep_func, args=(1,), timeout=0).wait()
        rpc.rpc_sync(dst_worker, my_sleep_func, args=(1,), timeout=0)
        
        # 重置以进行清理关闭
        rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)

    def test_dist_init_decorator(self):
        @dist_init(setup_rpc=False)
        def test_func(self):
            return "expected result"

        # 确保装饰器正常工作且返回预期结果
        self.assertEqual(test_func(self), "expected result")

        @dist_init
        def test_func(self):
            return "expected result"

        # 确保装饰器正常工作且返回预期结果
        self.assertEqual(test_func(self), "expected result")

    def test_use_rpc_pickler(self):
        class TestPickler:
            pass

        test_pickler = TestPickler()
        with _use_rpc_pickler(test_pickler):
            # 确保在使用 _use_rpc_pickler 上下文时默认 pickler 是 test_pickler
            self.assertTrue(torch.distributed.rpc.api._default_pickler is test_pickler)
        # 确保在退出 _use_rpc_pickler 上下文后，默认 pickler 是 _internal_rpc_pickler
        self.assertTrue(
            torch.distributed.rpc.api._default_pickler is _internal_rpc_pickler
        )

    @dist_init
    def test_wait_all(self):
        # 进入 _wait_all 上下文环境
        with _wait_all():
            # 断言 future_list 为空列表
            self.assertTrue(_thread_local_var.future_list == [])
            # 计算目标 worker 的名称
            dst = worker_name((self.rank + 1) % self.world_size)
            # 异步 RPC 调用，计算 torch.ones(2, 2) + 1
            fut = rpc.rpc_async(dst, torch.add, (torch.ones(2, 2), 1))
            # 断言 future_list 中有一个元素
            self.assertTrue(len(_thread_local_var.future_list) == 1)
            # 断言 future_list 的第一个元素是 torch._C.Future 类型
            self.assertTrue(isinstance(_thread_local_var.future_list[0], torch._C.Future))
        # 断言 fut 已经完成
        self.assertTrue(fut.done())
        # 断言 fut 的结果是 torch.ones(2, 2) + 1
        self.assertEqual(fut.wait(), torch.ones(2, 2) + 1)
        # 断言 _thread_local_var 没有 future_list 属性
        self.assertFalse(hasattr(_thread_local_var, "future_list"))

    @dist_init
    def test_wait_all_multiple_call(self):
        # 进入 _wait_all 上下文环境
        with _wait_all():
            # 断言 future_list 为空列表
            self.assertTrue(_thread_local_var.future_list == [])
            # 计算目标 worker 的名称
            dst = worker_name((self.rank + 1) % self.world_size)
            # 循环多次进行 RPC 调用和同步等待
            for i in range(20):
                # 异步 RPC 调用，计算 torch.ones(i, i) + 1
                fut = rpc.rpc_async(dst, torch.add, (torch.ones(i, i), 1))
                # 同步 RPC 调用，计算 torch.ones(i, i) + 1
                res = rpc.rpc_sync(dst, torch.add, (torch.ones(i, i), 1))
                # 断言 res 的值等于 torch.ones(i, i) + 1
                self.assertEqual(res, torch.ones(i, i) + 1)
                # 断言 fut 的结果是 torch.ones(i, i) + 1
                self.assertEqual(fut.wait(), torch.ones(i, i) + 1)
            # 断言 future_list 中有 20 个元素
            self.assertTrue(len(_thread_local_var.future_list) == 20)
        # 断言 _thread_local_var 没有 future_list 属性
        self.assertFalse(hasattr(_thread_local_var, "future_list"))

    @dist_init
    def test_wait_all_timeout(self):
        # 获取超时错误的预期正则表达式
        expected_error = self.get_timeout_error_regex()
        # 断言在运行时期间捕获 RuntimeError 异常，并且错误信息匹配预期错误
        with self.assertRaisesRegex(RuntimeError, expected_error):
            # 进入 _wait_all 上下文环境
            with _wait_all():
                # 断言 future_list 为空列表
                self.assertTrue(_thread_local_var.future_list == [])
                # 计算目标 worker 的名称
                dst = worker_name((self.rank + 1) % self.world_size)
                # 设置超时时间为 0.1 秒
                timeout = 0.1  # 100 ms
                # 异步 RPC 调用，调用带有超时参数的函数 my_sleep_func(1)
                fut = rpc.rpc_async(dst, my_sleep_func, args=(1,), timeout=timeout)
        # 断言 _thread_local_var 没有 future_list 属性
        self.assertFalse(hasattr(_thread_local_var, "future_list"))

    @dist_init
    def test_wait_all_raise_in_user_func(self):
        # 断言在运行时期间捕获 ValueError 异常
        with self.assertRaises(ValueError):
            # 进入 _wait_all 上下文环境
            with _wait_all():
                # 断言 future_list 为空列表
                self.assertTrue(_thread_local_var.future_list == [])
                # 计算目标 worker 的名称
                dst = worker_name((self.rank + 1) % self.world_size)
                # 异步 RPC 调用，调用会引发 ValueError 的函数 raise_func()
                fut = rpc.rpc_async(dst, raise_func)
        # 断言 _thread_local_var 没有 future_list 属性
        self.assertFalse(hasattr(_thread_local_var, "future_list"))

    @dist_init
    def test_wait_all_raise_in_body(self):
        # 断言在运行时期间捕获 ValueError 异常
        with self.assertRaises(ValueError):
            # 进入 _wait_all 上下文环境
            with _wait_all():
                # 在上下文环境内部抛出 ValueError 异常
                raise_func()
        # 断言 _thread_local_var 没有 future_list 属性
        self.assertFalse(hasattr(_thread_local_var, "future_list"))
    def test_custom_exception_throw_during_reconstruction(self):
        """
        Test that we still throw info about the remote side exception even when
        we cannot recreate it on client side.
        """
        # 初始化分布式进程组
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        
        # 如果当前进程不是主进程（rank != 0）
        if self.rank != 0:
            # 初始化异常捕获标志
            exc_caught = False
            # 目标 worker 的名称
            dst = worker_name(0)
            try:
                # 在目标 worker 上同步调用自定义异常函数
                rpc.rpc_sync(dst, custom_raise_func, args=())
            except RuntimeError as e:
                # 捕获 RuntimeError 异常
                exc_caught = True
                # 提取异常信息字符串
                msg = str(e)
                # 打印异常信息
                print(f"Got msg {msg}")
                # 断言异常信息包含特定字符串
                self.assertTrue("Original exception on remote side was" in msg)
                self.assertTrue("CustomException" in msg)
            except BaseException as e:
                # 捕获其他异常，重新抛出 RuntimeError
                raise RuntimeError(
                    f"Failure - expected RuntimeError, got {e}"
                ) from e
            finally:
                # 断言已捕获到异常
                self.assertTrue(exc_caught)

        # 等待所有进程达到同步点
        dist.barrier()

    timed_out_rpc_event = None

    @staticmethod
    def timed_out_rpc():
        # 静态方法，等待超时 RPC 事件
        RpcTest.timed_out_rpc_event.wait()

    @dist_init
    def test_wait_all_exit_early_python(self):
        # 在子进程中初始化事件
        RpcTest.timed_out_rpc_event = Event()

        # 等待所有进程初始化事件
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        dist.barrier()

        # 目标 worker 的名称
        dst = worker_name((self.rank + 1) % self.world_size)
        # 异步 RPC 调用三个函数
        fut1 = rpc.rpc_async(dst, RpcTest.timed_out_rpc)
        fut2 = rpc.rpc_async(dst, raise_func)
        fut3 = rpc.rpc_async(dst, raise_func)

        # 断言应该从 fut2 接收到错误
        with self.assertRaisesRegex(ValueError, expected_err):
            torch.futures.wait_all([fut1, fut2, fut3])

        # 解锁 fut1 的 RPC 线程
        RpcTest.timed_out_rpc_event.set()

    @dist_init
    def test_wait_all_exit_early_builtin(self):
        # 在子进程中初始化事件
        RpcTest.timed_out_rpc_event = Event()

        # 等待所有进程初始化事件
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        dist.barrier()

        # 目标 worker 的名称
        dst = worker_name((self.rank + 1) % self.world_size)
        # 异步 RPC 调用三个 torch.add 函数
        fut1 = rpc.rpc_async(dst, RpcTest.timed_out_rpc)
        fut2 = rpc.rpc_async(dst, torch.add, args=(torch.rand(10), torch.rand(5)))
        fut3 = rpc.rpc_async(dst, torch.add, args=(torch.rand(10), torch.rand(5)))

        # 断言应该从 fut2 接收到错误
        with self.assertRaisesRegex(RuntimeError, "size of tensor"):
            torch.futures.wait_all([fut1, fut2, fut3])

        # 解锁 fut1 的 RPC 线程
        RpcTest.timed_out_rpc_event.set()

    @dist_init
    # 定义一个测试函数，测试在早期退出脚本功能时的等待行为
    def test_wait_all_exit_early_script_function(self):
        # 在子进程中初始化事件
        RpcTest.timed_out_rpc_event = Event()

        # 等待所有进程初始化事件
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        dist.barrier()

        # 计算目标工作节点名称
        dst = worker_name((self.rank + 1) % self.world_size)
        
        # 异步调用远程过程：timed_out_rpc
        fut1 = rpc.rpc_async(dst, RpcTest.timed_out_rpc)
        # 异步调用远程过程：raise_func_script，并传递参数 expected_err
        fut2 = rpc.rpc_async(dst, raise_func_script, args=(expected_err,))
        fut3 = rpc.rpc_async(dst, raise_func_script, args=(expected_err,))

        # 我们应该从 fut2 收到 RuntimeError 错误
        with self.assertRaisesRegex(RuntimeError, expected_err):
            torch.futures.wait_all([fut1, fut2, fut3])

        # 解除对 fut1 的 RPC 线程阻塞
        RpcTest.timed_out_rpc_event.set()


    # 使用 dist_init 装饰器定义一个测试函数
    @dist_init
    def test_function_not_on_callee(self):
        # 测试：如果调用方不存在某个函数，我们不会崩溃，而是得到 AttributeError 表明函数不存在
        this_module = sys.modules[__name__]
        caller_worker = "worker0"
        callee_worker = "worker1"

        if self.rank == 1:
            # 使用 delattr 在当前节点上移除函数绑定
            delattr(this_module, "foo_add")
            # 通知远程节点我们已经移除了它
            rpc.rpc_sync(caller_worker, set_value, args=(self.rank,))

        if self.rank == 0:
            # 调用方存在该函数，但被调用方不存在
            # 等待远程端移除 foo_add 函数绑定
            wait_for_value_future()
            # 确保在这个模块上有这个属性，否则由于调用方的序列化错误，测试可能会失败
            self.assertTrue(hasattr(this_module, "foo_add"))
            with self.assertRaisesRegex(
                RuntimeError, "RPC pickler does not serialize"
            ):
                rpc.rpc_sync(callee_worker, foo_add, args=())


    # 使用 dist_init 装饰器定义一个测试函数
    @dist_init
    def test_non_garbage_collected_user_rref_due_to_local_circular_dependency(self):
        # 计算目标工作节点名称
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        # 创建 MyClass 类的实例 a 和 b
        a = MyClass(1)
        b = MyClass(2)

        # 防止 Python 对 a 和 b 进行垃圾回收
        a.other = b
        b.other = a

        n = self.rank
        # 使用 RPC 在远程节点上调用 torch.add，并传递参数
        a.rref = rpc.remote(
            dst_worker_name,
            torch.add,
            args=(torch.ones(n, n), 2)
        )
    def test_use_rref_after_shutdown(self):
        # 初始化 RPC，连接到指定的 worker
        rpc.init_rpc(
            name="worker%d" % self.rank,
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )
        # 计算目标 worker 的排名
        n = self.rank + 1
        dst_rank = n % self.world_size
        # 在远程 worker 上创建远程引用 (RRef)，执行 torch.add 操作
        rref = rpc.remote(
            worker_name(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        # 关闭 RPC 连接，释放资源，设置 graceful=True 确保本地 UserRRefs 被删除
        rpc.shutdown(graceful=True)

        # 断言捕获异常，检查是否能在删除后调用 rref.to_here()
        with self.assertRaisesRegex(
            RuntimeError, "Cannot call to_here\\(\\) on it after deletion."
        ):
            rref.to_here()

        # 断言捕获异常，检查是否能在删除后调用 fork UserRRef
        with self.assertRaisesRegex(
            RuntimeError, "Cannot call fork an UserRRef after deletion."
        ):
            import torch.distributed.rpc.internal as internal
            internal.serialize(rref)

    @staticmethod
    def _return_gpu_tensor():
        # 返回一个在 GPU 上生成的随机张量
        return torch.rand(3, 3).cuda(0)

    @staticmethod
    def _return_gpu_tensor_list():
        # 返回一个包含两个在不同 GPU 上生成的随机张量的列表
        return [torch.rand(3, 3).cuda(0), torch.rand(3, 3).cuda(1)]

    @staticmethod
    def _gpu_tensor_list_arg(tensor_list):
        # 返回一个在 CPU 上生成的随机张量
        return torch.rand(3, 3)

    def _create_rref(self):
        # 计算拥有者的 worker 排名
        owner_rank = (self.rank + 2) % self.world_size
        # 在远程 worker 上创建远程引用 (RRef)，执行 torch.add 操作
        return rpc.remote(
            worker_name(owner_rank),
            torch.add,
            args=(torch.zeros(2, 2), 1)
        )

    @dist_init
    def test_user_rrefs_confirmed(self):
        # 计算目标 worker 的排名
        dst_rank = (self.rank + 1) % self.world_size
        # 创建远程引用 (RRef)
        rref = self._create_rref()
        # 在目标 worker 上同步调用函数 check_rref_confirmed，检查 RRef 是否确认
        ret = rpc.rpc_sync(
            worker_name(dst_rank),
            check_rref_confirmed,
            args=(rref,)
        )
        # 断言确认结果为 True
        self.assertEqual(ret, True)

    @dist_init
    def test_user_rrefs_confirmed_remote(self):
        # 计算目标 worker 的排名
        dst_rank = (self.rank + 1) % self.world_size
        # 创建远程引用 (RRef)
        rref = self._create_rref()
        # 在目标 worker 上创建远程引用 (RRef)，执行函数 check_rref_confirmed，然后同步到本地
        ret_rref = rpc.remote(
            worker_name(dst_rank),
            check_rref_confirmed,
            args=(rref,)
        )
        # 断言远程引用的结果为 True
        self.assertEqual(ret_rref.to_here(), True)

    @dist_init
    def test_rref_py_pickle_not_supported(self):
        # 创建一个本地 RRef，其中包含一个整数
        local_rref = RRef(35)
        with TemporaryFileName() as fname:
            # 断言捕获异常，尝试使用 Python pickler 序列化本地 RRef
            with self.assertRaisesRegex(RuntimeError, "Can not pickle rref in python pickler"):
                torch.save(local_rref, fname)

    @dist_init
    def test_remote_throw(self):
        # 在远程 worker 上创建远程引用 (RRef)，执行函数 raise_or_inc，传递 torch.ones(2) 作为参数
        rref = rpc.remote(worker_name((self.rank + 1) % self.world_size),
                          raise_or_inc,
                          args=(torch.ones(2),))
        # 断言捕获异常，检查是否抛出预期的异常消息
        with self.assertRaisesRegex(Exception, ".*Expected error.*"):
            rref.to_here()

    @dist_init
    @dist_init
    # 使用 dist_init 装饰器标记测试函数，表示它需要在分布式环境中初始化
    def test_non_cont_tensors(self):
        if self.rank == 0:
            # 如果当前进程的排名为 0

            # 创建一个非连续的张量
            t = torch.rand(5, 5)
            # 从 t 张量中选取一个视图，从第1维（列）的第2位置开始，长度为2
            t_view = t.narrow(1, 2, 2)
            # 断言 t_view 张量是否为非连续的
            self.assertFalse(t_view.is_contiguous())
            # 将 t_view 张量转换为连续的张量
            t_cont = t_view.contiguous()
            # 断言 t_cont 张量是否为连续的
            self.assertTrue(t_cont.is_contiguous())
            # 断言 t_view 和 t_cont 张量的内容是否相等
            self.assertEqual(t_view, t_cont)

            # 将非连续张量 t_view 通过 RPC 发送到另一进程
            next_rank = (self.rank + 1) % self.world_size
            t_ret = rpc.rpc_sync(worker_name(next_rank), non_cont_test, args=(t_view, t_cont))

            # 验证返回的张量
            self.assertEqual(t_view, t_ret)
            # 断言返回的张量 t_ret 是否为非连续的
            self.assertFalse(t_ret.is_contiguous())

    @dist_init
    # 使用 dist_init 装饰器标记测试函数，表示它需要在分布式环境中初始化
    def test_callback_simple(self):
        set_by_cb = concurrent.futures.Future()
        n = self.rank + 1

        def callback(fut):
            # 回调函数，等待未来对象 fut 完成
            ret = fut.wait()
            # 断言 fut 的返回值 ret 是否为 n x n 大小的全1张量乘以2
            self.assertEqual(ret, torch.ones(n, n) * 2)
            # 将结果设置到 set_by_cb 中，结果为 ret 的克隆加1
            set_by_cb.set_result(ret.clone() + 1)

        # 发起异步 RPC 调用，调用 torch.add 函数
        fut = rpc.rpc_async(
            worker_name(n % self.world_size),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n))
        )

        # 注册回调函数 callback
        fut.then(callback)

        # 等待 fut 完成，并断言其返回值为 n x n 大小的全1张量乘以2
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)
        # 断言 set_by_cb 中保存的结果为 n x n 大小的全1张量乘以2 加1
        self.assertEqual(set_by_cb.result(), torch.ones(n, n) * 2 + 1)
        # 再次等待 fut 完成，并断言其返回值为 n x n 大小的全1张量乘以2
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

    @dist_init
    # 使用 dist_init 装饰器标记测试函数，表示它需要在分布式环境中初始化
    def test_callback_wrong_arg_num(self):
        set_by_cb = concurrent.futures.Future()
        n = self.rank + 1

        # 发起异步 RPC 调用，调用 torch.add 函数
        fut = rpc.rpc_async(
            worker_name(n % self.world_size),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n))
        )

        # 使用错误数量的参数注册回调函数 my_function
        cb_fut = fut.then(my_function)

        # 等待 fut 完成，并断言其返回值为 n x n 大小的全1张量乘以2
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

        # 使用断言检查异常是否被引发，检查错误消息中是否包含特定字符串
        with self.assertRaisesRegex(
            RuntimeError,
            "my\\_function\\(\\) missing 2 required positional arguments"
        ):
            cb_fut.wait()

    @dist_init
    # 使用 dist_init 装饰器标记测试函数，表示它需要在分布式环境中初始化
    def test_callback_wrong_arg_type(self):
        dst = worker_name((self.rank + 1) % self.world_size)

        # 发起异步 RPC 调用，调用 torch.add 函数，但第二个参数类型不匹配
        fut0 = rpc.rpc_async(dst, torch.add, args=(torch.ones(2, 2), 1))
        # 注册回调函数，对返回值进行加法操作
        fut1 = fut0.then(lambda x: x + 1)

        # 使用断言检查异常是否被引发，检查错误消息中是否包含特定字符串
        with self.assertRaisesRegex(
            RuntimeError,
            "unsupported operand type\\(s\\) for \\+"
        ):
            fut1.wait()
    @dist_init
    def test_callback_multi(self):
        # 定义回调函数的数量
        num_cbs = 10
        # 根据当前进程的排名获取一个数字
        n = self.rank + 1

        # 定义回调函数，接受索引和Future对象作为参数
        def callback(idx, fut):
            # 等待Future对象完成并返回结果
            ret = fut.wait()
            # 断言Future对象的返回结果为全1矩阵乘以2
            self.assertEqual(ret, torch.ones(n, n) * 2)
            # 返回结果加上索引
            return ret + idx

        # 异步RPC调用，调用torch.add函数，并传入两个全1矩阵作为参数
        fut = rpc.rpc_async(
            worker_name(n % self.world_size),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n))
        )

        # 初始化回调Future对象列表
        cb_futs = []
        # 对于每一个索引，添加一个回调Future对象到列表中
        for idx in range(num_cbs):
            cb_futs.append(fut.then(partial(callback, idx)))

        # 等待RPC异步调用的结果，并断言结果为全1矩阵乘以2
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

        # 对于每一个回调Future对象，等待其结果并进行断言
        for idx in range(num_cbs):
            self.assertEqual(
                cb_futs[idx].wait(),
                torch.ones(n, n) * 2 + idx
            )

        # 再次等待RPC异步调用的结果，并断言结果为全1矩阵乘以2
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

    @dist_init
    def test_callback_chain(self):
        # 根据当前进程的排名获取一个数字
        n = self.rank + 1
        # 获取目标worker的名称
        dst = worker_name(n % self.world_size)

        # 定义回调函数，接受一个Future对象作为参数
        def callback(fut):
            # 等待Future对象完成并返回结果加1
            return fut.wait() + 1

        # 异步RPC调用，调用torch.add函数，并传入全1矩阵和1作为参数
        fut = rpc.rpc_async(
            worker_name(n % self.world_size),
            torch.add,
            args=(torch.ones(n, n), 1)
        )

        # 定义回调链的长度
        num_cbs = 20
        # 构建回调链
        for _ in range(num_cbs):
            fut = fut.then(callback)

        # 等待RPC异步调用的结果，并断言结果为全1矩阵加1加上回调链的长度
        self.assertEqual(fut.wait(), torch.ones(n, n) + 1 + num_cbs)

    @dist_init
    def test_callback_in_rpc(self):
        # 获取目标worker的名称
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)

        # 同步RPC调用，调用add_use_future_cb函数，并传入参数
        ret = rpc.rpc_sync(
            dst1,
            add_use_future_cb,
            args=(dst2, torch.ones(2, 2), 1, 2)
        )
        # 断言RPC调用的结果为全1矩阵加1加2
        self.assertEqual(ret, torch.ones(2, 2) + 1 + 2)

    @dist_init
    def test_callback_with_ret(self):
        # 获取目标worker的名称
        dst = worker_name((self.rank + 1) % self.world_size)

        # 定义回调函数，接受一个Future对象作为参数
        def callback(fut0):
            # 异步RPC调用，调用torch.add函数，并传入Future对象的等待结果和1作为参数
            fut2 = rpc.rpc_async(
                dst,
                torch.add,
                args=(fut0.wait(), 1)
            ).then(lambda fut1: fut1.wait() + 1)

            # 等待RPC异步调用的结果并返回
            return fut2.wait()

        # 异步RPC调用，调用torch.add函数，并传入全1矩阵和1作为参数，并建立回调链
        fut3 = rpc.rpc_async(
            dst,
            torch.add,
            args=(torch.ones(2, 2), 1)
        ).then(callback)

        # 等待RPC异步调用的结果，并断言结果为全1矩阵加3
        self.assertEqual(fut3.wait(), torch.ones(2, 2) + 3)

    @dist_init
    def test_callback_with_error(self):
        # 获取目标worker的名称
        dst = worker_name((self.rank + 1) % self.world_size)

        # 定义回调函数，接受一个Future对象作为参数
        def callback(fut0):
            # 断言Future对象会引发预期的值错误
            with self.assertRaisesRegex(ValueError, "Expected error"):
                fut0.wait()
            # 抛出另一个预期的运行时错误
            raise RuntimeError("Another expected error")

        # 异步RPC调用，调用raise_func函数，并建立回调链
        fut1 = rpc.rpc_async(dst, raise_func).then(callback)

        # 断言等待RPC异步调用时会引发预期的运行时错误
        with self.assertRaisesRegex(RuntimeError, "Another expected error"):
            fut1.wait()

    @dist_init
    def test_callback_none(self):
        # 获取目标worker的名称
        dst = worker_name((self.rank + 1) % self.world_size)
        # 断言调用RPC异步调用时传入空值会引发类型错误
        with self.assertRaisesRegex(
            TypeError,
            "incompatible function arguments."
        ):
            rpc.rpc_async(dst, raise_func).then(None)
    def test_add_done_callback(self):
        # 初始化一个标志，用于检测回调函数是否被设置
        set_by_cb = False
        # 计算工作节点编号
        n = self.rank + 1

        # 定义一个回调函数，用于设置回调函数触发标志
        def callback(fut):
            nonlocal set_by_cb
            # 等待 future 对象完成
            fut.wait()
            # 设置回调标志为 True
            set_by_cb = True

        # 异步调用远程过程调用，返回一个 future 对象
        fut = rpc.rpc_async(
            worker_name(n % self.world_size),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n))
        )

        # 将回调函数添加到 future 对象的回调函数列表中
        fut.add_done_callback(callback)
        
        # 添加一个 'then' 回调函数，以确保在第一个回调函数执行完毕后再执行
        fut_then = fut.then(lambda _: True)

        # 等待 future 对象完成，并验证其返回值
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

        # 提示：我们无法保证 add_done_callback 函数会在测试完成前执行。
        # 因此，添加一个 'then' 回调函数来确保我们等待第一个回调函数的执行
        fut_then.wait()

        # 验证回调函数已经被设置
        self.assertTrue(set_by_cb)

        # 再次等待 future 对象完成，并验证其返回值
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

    @dist_init
    def test_mark_future_twice(self):
        # 异步调用远程过程调用，返回一个 future 对象
        fut = rpc.rpc_async(
            worker_name((self.rank + 1) % self.world_size),
            torch.add,
            args=(torch.zeros(2, 2), 1)
        )

        # 等待 future 对象完成，并验证其返回值
        self.assertEqual(fut.wait(), torch.zeros(2, 2) + 1)

        # 使用 assertRaisesRegex 来检查是否抛出指定异常
        with self.assertRaisesRegex(
            RuntimeError,
            "Future can only be marked completed once"
        ):
            # 尝试第二次标记 future 对象为完成状态
            fut.set_result(1)

    @dist_init
    def test_pickle_future(self):
        # 创建一个新的 torch future 对象
        fut = torch.futures.Future()
        errMsg = "Can not pickle torch.futures.Future"

        # 获取目标 worker 的名称
        dst = worker_name((self.rank + 1) % self.world_size)

        # 使用临时文件名进行远程同步调用，预期会抛出运行时异常
        with TemporaryFileName() as fname:
            with self.assertRaisesRegex(RuntimeError, errMsg):
                rpc.rpc_sync(dst, fail_on_fut, args=(fut,))

        # 使用临时文件名进行远程异步调用，预期会抛出运行时异常
        with TemporaryFileName() as fname:
            with self.assertRaisesRegex(RuntimeError, errMsg):
                rpc.rpc_async(dst, fail_on_fut, args=(fut,))

        # 使用临时文件名进行远程调用，预期会抛出运行时异常
        with TemporaryFileName() as fname:
            with self.assertRaisesRegex(RuntimeError, errMsg):
                rpc.remote(dst, fail_on_fut, args=(fut,))

    @dist_init
    def test_future_done(self):
        # 获取目标 worker 的名称
        dst = worker_name((self.rank + 1) % self.world_size)

        # 异步调用远程过程调用，返回一个 future 对象
        fut = rpc.rpc_async(dst, torch.add, args=(torch.zeros(2), 1))

        # 等待 future 对象完成
        fut.wait()

        # 验证 future 对象是否已完成
        self.assertTrue(fut.done())

    @dist_init
    def test_future_done_exception(self):
        # 获取目标 worker 的名称
        dst = worker_name((self.rank + 1) % self.world_size)

        # 异步调用远程过程调用，返回一个 future 对象，预期会引发异常
        fut = rpc.rpc_async(dst, raise_func)

        # 使用 assertRaisesRegex 来检查是否抛出指定异常
        with self.assertRaisesRegex(ValueError, "Expected error"):
            # 等待 future 对象完成
            fut.wait()

        # 验证 future 对象是否已完成
        self.assertTrue(fut.done())

    def _test_future_cb(self, func):
        # 获取两个目标 worker 的名称
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)

        # 在目标 worker 上同步调用指定函数，返回执行结果
        ret = rpc.rpc_sync(
            dst1,
            func,
            args=(dst2, torch.ones(2, 2), 1, 2)
        )

        # 验证返回结果是否符合预期
        self.assertEqual(ret, torch.ones(2, 2) + 1 + 2)

    @dist_init
    def test_future_in_rpc(self):
        # 在测试方法 test_future_cb 中调用 _test_future_cb 函数
        self._test_future_cb(add_use_future_set_result)

    @dist_init
    def test_future_nested_callback(self):
        # 调用 _test_future_cb 方法测试使用嵌套回调的未来函数
        self._test_future_cb(add_use_future_nested_cb)

    def _test_async_function_raise(self, mode):
        # 使用 assertRaisesRegex 检查运行时错误，确保抛出 "Expected error" 异常
        with self.assertRaisesRegex(RuntimeError, "Expected error"):
            # 在指定模式下运行 async_raise_func 函数
            self._run_func_in_mode(
                worker_name((self.rank + 1) % self.world_size),
                async_raise_func,
                mode
            )

    @dist_init
    def test_async_function_raise(self):
        # 在分布式环境中测试 async_raise_func 函数的同步执行模式
        self._test_async_function_raise(RPCExecMode.SYNC)

    @dist_init
    def test_async_function_raise_async(self):
        # 在分布式环境中测试 async_raise_func 函数的异步执行模式
        self._test_async_function_raise(RPCExecMode.ASYNC)

    @dist_init
    def test_async_function_raise_remote(self):
        # 在分布式环境中测试 async_raise_func 函数的远程执行模式
        self._test_async_function_raise(RPCExecMode.REMOTE)

    def _test_async_function_wrong_return_type(self, mode):
        # 设置错误信息，检查异步函数返回类型错误
        errMsg = (
            "Functions decorated with @rpc\\.async_function must return a "
            "torch\\.futures\\.Future object,"
        )
        with self.assertRaisesRegex(RuntimeError, errMsg):
            # 在指定模式下运行 async_wrong_type 函数
            self._run_func_in_mode(
                worker_name((self.rank + 1) % self.world_size),
                async_wrong_type,
                mode
            )

    @dist_init
    def test_async_function_wrong_return_type(self):
        # 在分布式环境中测试 async_wrong_type 函数的同步执行模式
        self._test_async_function_wrong_return_type(RPCExecMode.SYNC)

    @dist_init
    def test_async_function_wrong_return_type_async(self):
        # 在分布式环境中测试 async_wrong_type 函数的异步执行模式
        self._test_async_function_wrong_return_type(RPCExecMode.ASYNC)

    @dist_init
    def test_async_function_wrong_return_type_remote(self):
        # 在分布式环境中测试 async_wrong_type 函数的远程执行模式
        self._test_async_function_wrong_return_type(RPCExecMode.REMOTE)

    @dist_init
    def test_async_function_simple(self):
        # 定义两个目标 worker
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)

        # 使用 rpc_sync 远程调用 async_add 函数，并检查返回值是否正确
        ret = rpc.rpc_sync(dst1, async_add, args=(dst2, torch.ones(2, 2), 1))
        self.assertEqual(ret, torch.ones(2, 2) + 1)

    def _test_async_function(self, fn, mode=RPCExecMode.SYNC):
        # 定义两个目标 worker
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)

        # 准备参数
        args = (dst2, torch.ones(2, 2), 1, 2)
        
        # 在指定模式下运行 fn 函数，并检查返回值是否正确
        ret = self._run_func_in_mode(dst1, fn, mode, args=args)
        self.assertEqual(ret, torch.ones(2, 2) + 3)

    @dist_init
    def test_async_function_with_future_ctor(self):
        # 在分布式环境中测试 async_add_with_future_ctor 函数
        self._test_async_function(async_add_with_future_ctor)

    @dist_init
    def test_async_function_with_future_ctor_remote(self):
        # 在分布式环境中测试 async_add_with_future_ctor 函数的远程执行模式
        self._test_async_function(
            async_add_with_future_ctor,
            RPCExecMode.REMOTE
        )

    @dist_init
    def test_async_function_chained(self):
        # 在分布式环境中测试 async_add_chained 函数
        self._test_async_function(async_add_chained)

    @dist_init
    def test_async_function_chained_remote(self):
        # 在分布式环境中测试 async_add_chained 函数的远程执行模式
        self._test_async_function(async_add_chained, RPCExecMode.REMOTE)

    @dist_init
    def test_async_function_nested(self):
        # 在分布式环境中测试 async_add_nested 函数
        self._test_async_function(async_add_nested)

    @dist_init
    # 测试异步函数在远程执行时的嵌套调用情况
    def test_async_function_nested_remote(self):
        self._test_async_function(async_add_nested, RPCExecMode.REMOTE)

    # 使用分布式初始化装饰器，测试异步静态方法的执行
    @dist_init
    def test_async_static_method(self):
        self._test_async_function(AsyncExecutionClass.static_async_add)

    # 使用分布式初始化装饰器，测试异步静态方法在远程执行的情况
    @dist_init
    def test_async_static_method_remote(self):
        self._test_async_function(
            AsyncExecutionClass.static_async_add,
            RPCExecMode.REMOTE
        )

    # 使用分布式初始化装饰器，测试异步类方法的执行
    @dist_init
    def test_async_class_method(self):
        self._test_async_function(AsyncExecutionClass.class_async_add)

    # 使用分布式初始化装饰器，测试异步类方法在远程执行的情况
    @dist_init
    def test_async_class_method_remote(self):
        self._test_async_function(
            AsyncExecutionClass.class_async_add,
            RPCExecMode.REMOTE
        )

    # 测试异步类的远程引用代理功能
    def _test_test_async_class_rref_proxy(self, mode=RPCExecMode.SYNC):
        # 计算目标工作节点名称
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)
        # 在目标节点上创建远程引用对象
        rref = rpc.remote(dst1, AsyncExecutionClass)

        x = torch.ones(2, 2)
        y = torch.ones(2, 2) + 1
        # 根据执行模式选择同步执行、异步执行或远程执行的方式
        if mode == RPCExecMode.SYNC:
            ret = rref.rpc_sync().static_async_add(dst2, x, x, y)
            ret += rref.rpc_sync().class_async_add(dst2, x, x, y)
            ret += rref.rpc_sync().bound_async_add(dst2, x, x, y)
        elif mode == RPCExecMode.ASYNC:
            ret = rref.rpc_async().static_async_add(dst2, x, x, y).wait()
            ret += rref.rpc_async().class_async_add(dst2, x, x, y).wait()
            ret += rref.rpc_async().bound_async_add(dst2, x, x, y).wait()
        elif mode == RPCExecMode.REMOTE:
            ret = rref.remote().static_async_add(dst2, x, x, y).to_here()
            ret += rref.remote().class_async_add(dst2, x, x, y).to_here()
            ret += rref.remote().bound_async_add(dst2, x, x, y).to_here()

        # 断言返回结果与预期值的相等性
        self.assertEqual(ret, 3 * 4 * x)

    # 使用分布式初始化装饰器，测试异步类的远程引用代理功能
    @dist_init
    def test_async_class_rref_proxy(self):
        self._test_test_async_class_rref_proxy()

    # 使用分布式初始化装饰器，测试异步类的远程引用代理功能（异步执行）
    @dist_init
    def test_async_class_rref_proxy_async(self):
        self._test_test_async_class_rref_proxy(mode=RPCExecMode.ASYNC)

    # 使用分布式初始化装饰器，测试异步类的远程引用代理功能（远程执行）
    @dist_init
    def test_async_class_rref_proxy_remote(self):
        self._test_test_async_class_rref_proxy(mode=RPCExecMode.REMOTE)

    # 测试多步骤的异步函数执行
    def _test_async_function_multi(self, fn, mode=RPCExecMode.SYNC):
        # 计算目标工作节点名称
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)

        num = 20
        step = 3
        args = (dst2, torch.ones(2, 2), num, step)
        # 根据执行模式运行函数，并获取结果
        ret = self._run_func_in_mode(dst1, fn, mode, args=args)
        # 断言返回结果与预期值的相等性
        self.assertEqual(ret, torch.ones(2, 2) + num * step)

    # 使用分布式初始化装饰器，测试多步骤的异步函数执行（链式调用）
    @dist_init
    def test_async_function_multi_chained(self):
        self._test_async_function_multi(async_add_chained_multi)

    # 使用分布式初始化装饰器，测试多步骤的异步函数执行（链式调用，异步执行）
    @dist_init
    def test_async_function_multi_chained_async(self):
        self._test_async_function_multi(
            async_add_chained_multi,
            RPCExecMode.ASYNC
        )
    # 测试异步函数多链式远程调用的方法
    def test_async_function_multi_chained_remote(self):
        self._test_async_function_multi(
            async_add_chained_multi,   # 异步函数：多链式远程调用
            RPCExecMode.REMOTE         # RPC执行模式：远程
        )

    @dist_init
    # 测试异步函数多扇出的方法
    def test_async_function_multi_fanout(self):
        self._test_async_function_multi(async_add_multi_fanout)

    @dist_init
    # 测试异步函数多扇出的异步方法
    def test_async_function_multi_fanout_async(self):
        self._test_async_function_multi(
            async_add_multi_fanout,    # 异步函数：多扇出
            RPCExecMode.ASYNC           # RPC执行模式：异步
        )

    @dist_init
    # 测试异步函数多扇出的远程调用方法
    def test_async_function_multi_fanout_remote(self):
        self._test_async_function_multi(
            async_add_multi_fanout,    # 异步函数：多扇出
            RPCExecMode.REMOTE          # RPC执行模式：远程
        )

    # 测试返回Future的方法
    def _test_return_future(self, mode):
        # 使用断言检查是否抛出特定异常
        with self.assertRaisesRegex(
            RuntimeError,                # 异常类型：运行时错误
            "Can not pickle torch.futures.Future"  # 异常信息
        ):
            self._run_func_in_mode(
                worker_name((self.rank + 1) % self.world_size),  # 计算下一个工作进程的名称
                return_future,             # 调用返回Future的函数
                mode                       # 指定的RPC执行模式
            )

    @dist_init
    # 测试同步返回Future的方法
    def test_return_future(self):
        self._test_return_future(RPCExecMode.SYNC)

    @dist_init
    # 测试异步返回Future的方法
    def test_return_future_async(self):
        self._test_return_future(RPCExecMode.ASYNC)

    @dist_init
    # 测试远程返回Future的方法
    def test_return_future_remote(self):
        self._test_return_future(RPCExecMode.REMOTE)

    @dist_init
    # 测试RRef超时的方法
    def test_rref_timeout(self):
        # 这个测试类似于FaultyProcessGroupTest中的测试，但意在与除ProcessGroup外的其他后端一起运行。
        if self.rank != 0:   # 如果当前进程不是rank为0的进程，则直接返回
            return

        dst_rank = (self.rank + 1) % self.world_size   # 计算目标rank
        dst_worker = f"worker{dst_rank}"               # 目标worker的名称
        # 设置10毫秒超时
        rref = rpc.remote(dst_worker, my_sleep_func, args=(2, ), timeout=0.01)
        # 应该超时的远程创建对应的Future
        expected_error = self.get_timeout_error_regex()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rref._get_future().wait()
        # 调用以确保运行待处理的回调
        wait_until_pending_futures_and_users_flushed()
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rref.to_here()

        wait_until_owners_and_forks_on_rank(1, 1, rank=1)

    @dist_init(setup_rpc=False)
    @skip_but_pass_in_sandcastle_if(
        os.environ.get("RPC_INIT_WITH_TCP", None) == "1",
        "init_pg_then_rpc does not work with TCP init, see https://github.com/pytorch/pytorch/issues/41614."
    )
    @dist_init(setup_rpc=False)
    @skip_but_pass_in_sandcastle_if(
        os.environ.get("RPC_INIT_WITH_TCP", None) == "1",
        "Test does not work with TCP init, see https://github.com/pytorch/pytorch/issues/46491",
    )
    def test_init_pg_then_rpc(self):
        # 初始化进程组，使用"gloo"后端，设置初始方法、进程编号、总进程数
        dist.init_process_group(
            backend="gloo",
            init_method=self.init_method,
            rank=self.rank,
            world_size=self.world_size,
        )

        # 初始化 RPC，指定名称、后端、进程编号、总进程数以及 RPC 后端选项
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        # 测试 RPC。
        next_rank = (self.rank + 1) % self.world_size
        ret = rpc.rpc_sync(worker_name(next_rank), torch.add, args=(torch.ones(2, 2), 1))
        # 断言 RPC 返回值是否符合预期
        self.assertEqual(ret, torch.ones(2, 2) + 1)

        # 测试进程组
        dist.barrier()

        # 关闭 RPC
        rpc.shutdown()

    @dist_init
    def test_init_rpc_then_pg(self):
        # 初始化 RPC，指定名称、后端、进程编号、总进程数以及 RPC 后端选项
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        # 初始化进程组，使用"gloo"后端，设置初始方法、进程编号、总进程数
        dist.init_process_group(
            backend="gloo",
            init_method=self.init_method,
            rank=self.rank,
            world_size=self.world_size,
        )

        # 测试 RPC。
        next_rank = (self.rank + 1) % self.world_size
        ret = rpc.rpc_sync(worker_name(next_rank), torch.add, args=(torch.ones(2, 2), 1))
        # 断言 RPC 返回值是否符合预期
        self.assertEqual(ret, torch.ones(2, 2) + 1)

        # 测试进程组
        dist.barrier()

        # 关闭 RPC
        rpc.shutdown()

    @dist_init
    def test_wait_all_with_exception(self):
        # 创建异步 RPC 调用任务列表
        futs = []
        dst = worker_name((self.rank + 1) % self.world_size)
        for _ in range(10):
            futs.append(rpc.rpc_async(dst, raise_func))

        # 断言捕获到预期的异常
        with self.assertRaisesRegex(ValueError, "Expected error"):
            ret = torch.futures.wait_all(futs)

    @dist_init
    def test_wait_all_with_partial_exception(self):
        # 创建异步 RPC 调用任务列表，其中包含一个会抛出异常的任务
        futs = []
        dst = worker_name((self.rank + 1) % self.world_size)
        for _ in range(10):
            futs.append(rpc.rpc_async(dst, torch.add, args=(torch.ones(2), 1)))

        futs.append(rpc.rpc_async(dst, raise_func))

        # 断言捕获到预期的异常
        with self.assertRaisesRegex(ValueError, "Expected error"):
            ret = torch.futures.wait_all(futs)
    # 测试初始化 RPC 两次的情况
    def test_init_rpc_twice(self):
        # 使用指定的初始化方法、排名和世界大小进行 PG 初始化
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        # 第一次初始化 RPC
        rpc.init_rpc(
            name=worker_name(self.rank),  # 设置当前 worker 的名称
            backend=self.rpc_backend,     # 指定 RPC 后端
            rank=self.rank,               # 当前 worker 的排名
            world_size=self.world_size,   # 整个群集的大小
            rpc_backend_options=self.rpc_backend_options,  # RPC 后端选项
        )
        rpc.shutdown()  # 关闭 RPC

        # 等待所有初始化完成
        dist.barrier()

        # 为下一次初始化准备一个不同的文件名
        new_backend_options = self.rpc_backend_options
        new_backend_options.init_method += "init_2"

        # 确保 RPC 初始化可以再次正常工作
        rpc.init_rpc(
            name=worker_name(self.rank),  # 设置当前 worker 的名称
            backend=self.rpc_backend,     # 指定 RPC 后端
            rank=self.rank,               # 当前 worker 的排名
            world_size=self.world_size,   # 整个群集的大小
            rpc_backend_options=new_backend_options,  # 更新后的 RPC 后端选项
        )

        # 验证重新初始化后 RPC 是否可以正常工作
        dst = worker_name((self.rank + 1) % self.world_size)
        rpc.rpc_sync(dst, torch.add, args=(torch.ones(2, 2), 1))  # 使用 RPC 发送同步调用
        rpc.rpc_sync(dst, foo_add, args=())  # 使用 RPC 发送同步调用

        rpc.shutdown()  # 关闭 RPC

    # 测试不正确的参数类型的情况
    def test_wrong_types(self):
        # 断言应该抛出 TypeError 异常，其中 backend 参数必须是 BackendType 的成员
        with self.assertRaisesRegex(
            TypeError,
            "Argument backend must be a member of BackendType",
        ):
            rpc.init_rpc(
                name=worker_name(self.rank),  # 设置当前 worker 的名称
                rank=self.rank,               # 当前 worker 的排名
                world_size=self.world_size,   # 整个群集的大小
                backend="TENSORPIPE",         # 错误的 RPC 后端类型
            )

        # 断言应该抛出 TypeError 异常，其中 rpc_backend_options 参数必须是 RpcBackendOptions 的实例
        with self.assertRaisesRegex(
            TypeError,
            "Argument rpc_backend_options must be an instance of RpcBackendOptions",
        ):
            rpc.init_rpc(
                name=worker_name(self.rank),              # 设置当前 worker 的名称
                rank=self.rank,                          # 当前 worker 的排名
                world_size=self.world_size,              # 整个群集的大小
                backend=self.rpc_backend,                # 正确的 RPC 后端类型
                rpc_backend_options={"init_method": self.init_method}  # 错误的 RPC 后端选项类型
            )

    # 测试无法从选项推断出后端类型的情况
    def test_cannot_infer_backend_from_options(self):
        # 如果未指定后端但提供了非任何已知代理选项类的选项，则应该抛出异常
        rpc_backend_options = FooBackendOptions(self.init_method)

        with self.assertRaisesRegex(TypeError, "Could not infer backend for options"):
            rpc.init_rpc(
                name=worker_name(self.rank),    # 设置当前 worker 的名称
                rank=self.rank,                # 当前 worker 的排名
                world_size=self.world_size,    # 整个群集的大小
                rpc_backend_options=rpc_backend_options,  # 提供错误选项类型
            )
    # 定义一个测试方法，用于测试 RRef 对象的反向传播功能
    def test_owner_rref_backward(self):
        # 计算目标 worker 的名称
        dst = worker_name((self.rank + 1) % self.world_size)
        # 创建一个 10x10 的随机张量，并设置 requires_grad=True
        t1 = torch.rand(10, 10, requires_grad=True)
        # 创建一个 RRef 对象，其值为 t1.sum() + t1.sum()
        rref = rpc.RRef(t1.sum() + t1.sum())
        # 对 RRef 对象进行反向传播
        rref.backward()
        # 预期的梯度应为全1的张量
        expected_grad = torch.ones_like(t1) * 2
        # 断言 t1 的梯度与预期的梯度相等
        self.assertEqual(expected_grad, t1.grad)

        # 使用 dist_autograd.context() 创建一个上下文，用于异步执行分布式自动求导
        with dist_autograd.context() as context_id:
            # 在目标 worker 上调用 rpc_sync，计算 t1 + t1 的结果
            t2 = rpc.rpc_sync(dst, torch.add, args=(t1, t1))
            # 创建一个 RRef 对象，其值为 t2.sum()
            rref = rpc.RRef(t2.sum())
            # 在指定的上下文中进行反向传播
            rref.backward(context_id)
            # 断言 t1 的梯度与预期的梯度相等
            self.assertEqual(expected_grad, dist_autograd.get_gradients(context_id)[t1])

        # 进行双重反向传播测试
        with dist_autograd.context() as context_id:
            # 再次在目标 worker 上调用 rpc_sync，计算 t1 + t1 的结果
            t2 = rpc.rpc_sync(dst, torch.add, args=(t1, t1))
            # 创建一个 RRef 对象，其值为 t2.sum()
            rref = rpc.RRef(t2.sum())
            # 在指定的上下文中进行反向传播，并保留计算图以备后续使用
            rref.backward(context_id, retain_graph=True)
            # 再次对同一上下文进行反向传播
            rref.backward(context_id)
            # 断言 t1 的梯度为预期梯度的两倍
            self.assertEqual(expected_grad * 2, dist_autograd.get_gradients(context_id)[t1])

        # 测试错误情况
        with self.assertRaisesRegex(RuntimeError, "tensors does not require grad and does not have a grad_fn"):
            # 尝试对不需要梯度计算的 tensor 创建 RRef 并进行反向传播
            rpc.RRef(torch.rand(10)).backward()

        with self.assertRaisesRegex(RuntimeError, "grad can be implicitly created only for scalar outputs"):
            # 尝试对非标量输出创建 RRef 并进行反向传播
            rpc.RRef(torch.rand(10, requires_grad=True)).backward()

        with self.assertRaisesRegex(RuntimeError, "Could not find autograd context with id: 100"):
            # 尝试在不存在的上下文中进行反向传播
            rpc.RRef(torch.rand(10, requires_grad=True).sum()).backward(100)

        with self.assertRaisesRegex(RuntimeError, "RRef should contain a tensor for .backward()"):
            # 尝试对不包含 tensor 的 RRef 进行反向传播
            rpc.RRef("foo").backward()

    @staticmethod
    def _sum(x):
        return x.sum()

    @staticmethod
    def _identity(x):
        return x

    @dist_init
    # 定义一个测试方法，用于测试用户自定义函数的 RRef 对象的反向传播功能
    def test_user_rref_backward(self):
        # 计算目标 worker 的名称
        dst = worker_name((self.rank + 1) % self.world_size)
        # 创建一个 10 维随机张量，并设置 requires_grad=True
        t = torch.rand(10, requires_grad=True)
        # 使用 dist_autograd.context() 创建一个上下文，用于异步执行分布式自动求导
        with dist_autograd.context() as context_id:
            # 在目标 worker 上调用 rpc.remote，远程执行 _sum 函数，并传入参数 t
            rref = rpc.remote(dst, RpcTest._sum, args=(t,))
            # 在指定的上下文中进行反向传播，并保留计算图以备后续使用
            rref.backward(context_id, retain_graph=True)
            # 再次对同一上下文进行反向传播
            rref.backward(context_id)
            # 断言 t 的梯度为全1张量乘以2
            self.assertEqual(torch.ones_like(t) * 2, dist_autograd.get_gradients(context_id)[t])

        with dist_autograd.context() as context_id:
            # 在目标 worker 上调用 rpc.remote，远程执行 _identity 函数，并传入参数 "foo"
            rref = rpc.remote(dst, RpcTest._identity, args=("foo",))
            # 尝试对不包含 tensor 的 RRef 进行反向传播应该抛出异常
            with self.assertRaisesRegex(RuntimeError, "RRef should contain a tensor for .backward()"):
                rref.backward(context_id)

            # 尝试在未指定上下文的情况下进行用户自定义 RRef 的反向传播应该抛出异常
            with self.assertRaisesRegex(RuntimeError, "User RRefs require 'dist_autograd_ctx_id' to be specified"):
                rref.backward()

    @dist_init(setup_rpc=False)
    # 定义测试方法，用于测试 RPC 关闭时的错误处理情况
    def test_shutdown_errors(self):
        # 初始化参数服务器
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        # 初始化 RPC
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        # 如果当前进程不是排名为0的主节点
        if self.rank != 0:
            # 备份原始的 _broadcast_to_followers 和 _delete_all_user_and_unforked_owner_rrefs 方法
            og_func = rpc.api._broadcast_to_followers
            og_rref_func = rpc.api._delete_all_user_and_unforked_owner_rrefs

            # 通过 Monkey-patch 修改 _broadcast_to_followers 方法，使其抛出异常，用于模拟错误
            def raise_error(sequence_id, objects_map):
                og_func(sequence_id, objects_map)
                raise RuntimeError('simulation')

            # 通过 Monkey-patch 修改 _delete_all_user_and_unforked_owner_rrefs 方法，使其抛出异常，用于模拟错误
            def rref_error():
                raise RuntimeError('simulation rref')

            try:
                # 应用修改后的方法
                rpc.api._broadcast_to_followers = raise_error
                rpc.api._delete_all_user_and_unforked_owner_rrefs = rref_error
                # 使用断言捕获 RuntimeError 异常，并验证异常消息是否为 'simulation rref'
                with self.assertRaisesRegex(RuntimeError, 'simulation rref'):
                    rpc.shutdown()
            finally:
                # 恢复原始的 _broadcast_to_followers 和 _delete_all_user_and_unforked_owner_rrefs 方法
                rpc.api._broadcast_to_followers = og_func
                rpc.api._delete_all_user_and_unforked_owner_rrefs = og_rref_func
        else:
            # 对于排名为0的主节点，使用断言捕获 RuntimeError 异常，并验证异常消息是否为 'timed out in _all_gather'
            with self.assertRaisesRegex(RuntimeError, 'timed out in _all_gather'):
                rpc.shutdown()

        # 执行分布式进程间的同步
        dist.barrier()

    # 使用 dist_init 装饰器定义分布式初始化方法，用于测试参数服务器
    @dist_init
    def test_my_parameter_server(self):
        # 调用参数服务器的测试方法，传入参数 False
        self._my_parameter_server(False)
class CudaRpcTest(RpcAgentTestFixture):
    
    @skip_if_lt_x_gpu(2)  # 如果 GPU 数量小于 2，则跳过测试
    @dist_init  # 分布式初始化装饰器
    def test_profiler_remote_cuda(self):
        if self.rank != 1:  # 如果当前进程的 rank 不是 1，则直接返回，不执行下面的代码
            return

        dst_cuda_0 = (self.rank + 1) % self.world_size  # 计算第一个目标 CUDA 设备的 rank
        dst_cuda_1 = (self.rank + 2) % self.world_size  # 计算第二个目标 CUDA 设备的 rank
        dst_worker_cuda_0 = worker_name(dst_cuda_0)  # 获取第一个目标 CUDA 设备的 worker 名称
        dst_worker_cuda_1 = worker_name(dst_cuda_1)  # 获取第二个目标 CUDA 设备的 worker 名称

        with _profile(use_cuda=True) as p:  # 使用 _profile 上下文管理器，并启用 CUDA 用于性能分析
            fut1 = rpc.rpc_async(dst_worker_cuda_0, udf_with_torch_ops, args=(0, ))  # 异步 RPC 调用到第一个目标 CUDA 设备上执行指定的 Torch 操作函数
            fut2 = rpc.rpc_async(dst_worker_cuda_1, udf_with_torch_ops, args=(1, ))  # 异步 RPC 调用到第二个目标 CUDA 设备上执行指定的 Torch 操作函数
            fut1.wait()  # 等待第一个异步 RPC 调用完成
            fut2.wait()  # 等待第二个异步 RPC 调用完成

        def get_name(event):
            return event.name[event.name.find(REMOTE_OP_STR) + len(REMOTE_OP_STR):]  # 从事件名称中提取远程操作的名称

        function_events = p.function_events  # 获取性能分析器中记录的所有函数事件

        for event in function_events:
            if event.is_async:  # 如果事件是异步操作
                self.assertEqual(0, event.device_time_total)  # 断言异步操作的总设备时间为 0
                self.assertEqual([], event.kernels)  # 断言异步操作没有内核执行
                self.assertEqual(0, event.device_time)  # 断言异步操作的设备时间为 0
            else:
                if event.node_id == 1:  # 如果事件的节点 ID 为 1，则跳过该事件
                    continue
                self.assertTrue(event.node_id in [dst_cuda_0, dst_cuda_1])  # 断言事件的节点 ID 在目标 CUDA 设备列表中
                if get_name(event) in EXPECTED_REMOTE_EVENTS:  # 如果事件名称在预期的远程事件列表中
                    self.assertGreater(event.device_time_total, 0)  # 断言事件的总设备时间大于 0
                    self.assertEqual(1, len(event.kernels))  # 断言事件的内核数量为 1
                    kernel = event.kernels[0]  # 获取事件的第一个内核
                    if event.node_id == dst_cuda_0:
                        self.assertEqual(kernel.device, 0)  # 断言第一个目标 CUDA 设备的内核位于设备 0
                    if event.node_id == dst_cuda_1:
                        self.assertEqual(kernel.device, 1)  # 断言第二个目标 CUDA 设备的内核位于设备 1
                    self.assertGreater(event.device_time, 0)  # 断言事件的设备时间大于 0

        # 验证 EXPECTED_REMOTE_EVENTS 是否是远程性能分析事件的子集
        remote_events = [event for event in function_events if event.is_remote]  # 获取所有远程事件
        remote_event_names = [get_name(event) for event in remote_events if get_name(event) in EXPECTED_REMOTE_EVENTS]  # 获取预期远程事件名称列表
        self.assertEqual(set(remote_event_names), set(EXPECTED_REMOTE_EVENTS))  # 断言远程事件名称集合与预期的远程事件名称集合相等


class TensorPipeAgentRpcTest(RpcAgentTestFixture, RpcTestCommon):

    def test_mismatched_type_for_options(self):
        # 如果选项不是 TensorPipeRpcBackendOptions 的实例，则应该抛出异常
        rpc_backend_options = FooBackendOptions(self.init_method)

        with self.assertRaisesRegex(
            TypeError, "`rpc_backend_options` must be a `TensorPipeRpcBackendOptions`"
        ):
            rpc.init_rpc(
                name=worker_name(self.rank),
                rank=self.rank,
                world_size=self.world_size,
                backend=rpc.BackendType.TENSORPIPE,
                rpc_backend_options=rpc_backend_options,
            )
    # 测试从选项中推断使用的后端类型

    # 创建 TensorPipeRpcBackendOptions 对象，指定初始化方法和传输方式
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
        init_method=self.init_method,
        _transports=tp_transports()
    )

    # 初始化 RPC，设置节点名称、节点排名、全局节点数量，传入后端选项
    rpc.init_rpc(
        name=worker_name(self.rank),
        rank=self.rank,
        world_size=self.world_size,
        # 不要传递后端类型
        rpc_backend_options=rpc_backend_options,
    )

    # 断言当前 RPC 代理是 TensorPipeAgent 类型
    self.assertIsInstance(rpc.api._get_current_rpc_agent(), rpc.TensorPipeAgent)

# FIXME 将此测试与 RpcTest 中对应的测试合并。
@dist_init(setup_rpc=False)
def test_set_and_get_num_worker_threads(self):
    # 设置工作线程数为 27
    NUM_THREADS = 27
    # 创建 TensorPipeRpcBackendOptions 对象，指定初始化方法、工作线程数和传输方式
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
        init_method=self.rpc_backend_options.init_method,
        num_worker_threads=NUM_THREADS,
        _transports=tp_transports(),
    )
    # 初始化 RPC，设置节点名称、后端类型、节点排名、全局节点数量，传入后端选项
    rpc.init_rpc(
        name=worker_name(self.rank),
        backend=self.rpc_backend,
        rank=self.rank,
        world_size=self.world_size,
        rpc_backend_options=rpc_backend_options,
    )

    # 获取当前 RPC 代理的调试信息，断言线程池大小与设定的工作线程数相等
    info = rpc.api._get_current_rpc_agent().get_debug_info()
    self.assertEqual(int(info["agent.thread_pool_size"]), NUM_THREADS)
    rpc.shutdown()

# FIXME 将此测试与 RpcTest 中对应的测试合并。
@dist_init(setup_rpc=False)
def test_tensorpipe_set_default_timeout(self):
    # 设置较长的超时时间，确保测试不会由于机器性能差导致超时
    timeout = 100
    # 创建 TensorPipeRpcBackendOptions 对象，指定初始化方法、工作线程数、RPC 超时和传输方式
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
        init_method=self.rpc_backend_options.init_method,
        num_worker_threads=self.rpc_backend_options.num_worker_threads,
        rpc_timeout=timeout,
        _transports=tp_transports(),
    )
    # 初始化 RPC，设置节点名称、后端类型、节点排名、全局节点数量，传入后端选项
    rpc.init_rpc(
        name=worker_name(self.rank),
        backend=self.rpc_backend,
        rank=self.rank,
        world_size=self.world_size,
        rpc_backend_options=rpc_backend_options,
    )

    # 获取当前 RPC 的默认超时时间，断言与设置的超时时间相等
    default_timeout = rpc.get_rpc_timeout()
    self.assertEqual(default_timeout, timeout)
    rpc.shutdown()

# FIXME 将此测试与 RpcTest 中对应的测试合并。
@dist_init(setup_rpc=False)
def test_tensorpipe_options_throw_on_timedelta_timeout(self):
    from datetime import timedelta

    timeout = timedelta()
    # 确保使用 timedelta 构造 TensorPipeRpcBackendOptions 会失败，抛出 TypeError 异常
    with self.assertRaisesRegex(TypeError, "incompatible constructor arguments"):
        rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
            init_method=self.rpc_backend_options.init_method,
            num_worker_threads=self.rpc_backend_options.num_worker_threads,
            rpc_timeout=timeout,
        )
    # 测试在获取 RRef 的类型时超时的情况，根据是否阻塞选择不同的测试方式
    def _test_rref_get_type_timeout(self, blocking):
        # 确定目标排名，以便找到相应的 worker
        dst_rank = (self.rank + 1) % self.world_size
        dst = worker_name(dst_rank)
        # 创建一个慢速的 RRef 对象，远程调用 MyClass 的构造函数，并传入参数
        slow_rref = rpc.remote(dst, MyClass, args=(torch.ones(2, 2), True))
        # 设置超时时间
        timeout = 0.5
        # 获取预期的超时错误正则表达式
        expected_err = self.get_timeout_error_regex()
        
        # 如果是阻塞模式
        if blocking:
            # 使用断言检查是否抛出了预期的 RuntimeError 异常
            with self.assertRaisesRegex(RuntimeError, expected_err):
                slow_rref._get_type(timeout=timeout, blocking=blocking)
        # 如果是非阻塞模式
        else:
            # 发起异步调用获取类型，返回一个 Future 对象
            fut = slow_rref._get_type(timeout=timeout, blocking=blocking)
            # 使用断言检查是否抛出了预期的 RuntimeError 异常
            with self.assertRaisesRegex(RuntimeError, expected_err):
                fut.wait()
    
        # FIXME：我们等待远程完成创建 OwnerRRef，因为在关闭 RPC 之前存在竞争条件。
        slow_rref.to_here()
    
    # 测试阻塞模式下的 RRef 获取类型超时情况
    def test_rref_get_type_timeout_blocking(self):
        self._test_rref_get_type_timeout(blocking=True)
    
    # 测试非阻塞模式下的 RRef 获取类型超时情况
    def test_rref_get_type_timeout_non_blocking(self):
        self._test_rref_get_type_timeout(blocking=False)
    
    # 使用分布式初始化装饰器初始化测试环境，并测试使用无效参数调用操作符时是否抛出预期的异常
    @dist_init
    def test_op_with_invalid_args(self):
        # 确定目标 worker
        dst = worker_name((self.rank + 1) % self.world_size)
        # 使用断言检查是否抛出了预期的 RuntimeError 异常，说明 Python 中调用的 torch 操作符没有匹配到任何模式
        with self.assertRaisesRegex(
            RuntimeError, "Overloaded torch operator invoked from Python failed to match any schema"
        ):
            rpc.rpc_sync(dst, torch.add, args=())
    # 在测试中使用 RRef 代理超时功能
    def _test_rref_proxy_timeout(self, rref_proxy_api):
        # 计算目标节点的排名
        dst_rank = (self.rank + 1) % self.world_size
        # 获取目标节点的名称
        dst = worker_name(dst_rank)
        # 在目标节点上创建一个远程的 RRef 对象
        rref = rpc.remote(dst, MyClass, args=(torch.ones(2, 2), ))
        
        # 确保 RRef 在远程节点上已经创建
        rref.to_here()
        
        # 获取 RRef 的特定代理 API
        rref_api = getattr(rref, rref_proxy_api)
        # 断言是否成功获取了 RRef 代理 API
        self.assertTrue(rref_api is not None, f"Failed to get RRef proxy api: {rref_proxy_api}")
        
        # 获取超时异常的正则表达式
        expected_error = self.get_timeout_error_regex()
        timeout = 2
        
        # 断言在指定的超时时间内发生 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, expected_error):
            result = rref_api(timeout=timeout).my_slow_method(torch.ones(2, 2))
            # 如果使用的是 rpc_async，则等待结果返回
            if rref_api == rref.rpc_async:
                result.wait()
            # 如果使用的是 remote，则等待远程调用完成
            elif rref_api == rref.remote:
                result._get_future().wait()
        
        # 处理 rpc.remote() 被卡住且超过了超时时间的情况
        slow_rref = rpc.remote(dst, MyClass, args=(torch.ones(2, 2), True))
        timeout = 0.01
        rref_api = getattr(slow_rref, rref_proxy_api)
        
        # 注意，即使在此情况下调用 rref.rpc_async()，我们也会超时于未来的创建，而不是等待未来。
        # 这是因为 rref 代理函数在返回未来之前调用 rref._get_type，这会阻塞在所有者节点上创建 RRef，
        # 直到指定的超时时间。
        with self.assertRaisesRegex(RuntimeError, expected_error):
            result = rref_api(timeout=timeout).my_instance_method(torch.ones(2, 2))
            # 如果 rref_api 是 slow_rref.rpc_async，则立即返回并通过 wait() 抛出超时异常
            if rref_api == slow_rref.rpc_async:
                result.wait()
        
        # FIXME 我们等待远程节点完成创建 OwnerRRef
        # 因为目前如果在此之前关闭 RPC，会存在竞争条件。
        slow_rref.to_here()

    @dist_init
    def test_rref_proxy_timeout(self):
        # 使用不同的 RPC API 测试 RRef 代理超时功能
        for rpc_api in ["rpc_sync", "rpc_async", "remote"]:
            self._test_rref_proxy_timeout(rpc_api)

    @dist_init
    def test_send_to_rank_sparse(self):
        dst_rank = (self.rank + 1) % self.world_size

        # 测试稀疏张量的发送
        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            x = build_sparse_tensor()
            y = build_sparse_tensor()
            expected_tensor = (x + y)
            # 在指定执行模式下运行函数，并断言结果与预期相等
            ret = self._run_func_in_mode(dst_rank, torch.add, exec_mode, args=(x, y))
            self.assertEqual(expected_tensor, ret)

        # 使用 coalesce=True 构建稀疏张量的发送
        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            x = build_sparse_tensor(coalesce=True)
            y = build_sparse_tensor(coalesce=True)
            expected_tensor = (x + y)
            # 在指定执行模式下运行函数，并断言结果与预期相等
            ret = self._run_func_in_mode(dst_rank, torch.add, exec_mode, args=(x, y))
            self.assertEqual(expected_tensor, ret)
    def test_self_py_udf_remote_sparse(self):
        # 调用 _self_py_udf_remote 方法，传入当前节点的信息、三个稀疏张量作为参数
        self._self_py_udf_remote(
            rpc.get_worker_info(),
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor()
        )

    @dist_init
    def test_self_remote_rref_as_rpc_arg_sparse(self):
        # 使用 dist_init 装饰器初始化分布式环境
        # 计算目标节点的名称
        dst = worker_name((self.rank + 1) % self.world_size)
        # 调用 _self_remote_rref_as_rpc_arg 方法，传入目标节点的名称、三个稀疏张量作为参数
        self._self_remote_rref_as_rpc_arg(
            dst,
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor()
        )

    @dist_init
    def test_self_remote_rref_as_self_rpc_arg_sparse(self):
        # 使用 dist_init 装饰器初始化分布式环境
        # 调用 _self_remote_rref_as_rpc_arg 方法，传入当前节点的信息、三个稀疏张量作为参数
        self._self_remote_rref_as_rpc_arg(
            rpc.get_worker_info(),
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor()
        )

    @dist_init
    def test_self_remote_rref_as_remote_arg_sparse(self):
        # 使用 dist_init 装饰器初始化分布式环境
        # 计算目标节点的名称
        dst = worker_name((self.rank + 1) % self.world_size)
        # 调用 _self_remote_rref_as_remote_arg 方法，传入目标节点的名称、三个稀疏张量作为参数
        self._self_remote_rref_as_remote_arg(
            dst,
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor()
        )

    @dist_init
    def test_self_remote_rref_as_self_remote_arg_sparse(self):
        # 使用 dist_init 装饰器初始化分布式环境
        # 调用 _self_remote_rref_as_remote_arg 方法，传入当前节点的信息、三个稀疏张量作为参数
        self._self_remote_rref_as_remote_arg(
            rpc.get_worker_info(),
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor()
        )

    def test_world_size_one_sparse(self):
        # 调用 _world_size_one 方法，传入两个稀疏张量作为参数
        self._world_size_one(
            build_sparse_tensor(),
            build_sparse_tensor()
        )

    @dist_init
    def test_multi_rpc_sparse(self):
        # 使用 dist_init 装饰器初始化分布式环境
        # 调用 _multi_rpc 方法，传入 True 作为参数
        self._multi_rpc(True)

    def test_wait_all_workers_sparse(self):
        # 调用 _wait_all_workers 方法，传入 heavy_rpc_sparse 和一个稀疏张量作为参数
        self._wait_all_workers(heavy_rpc_sparse, build_sparse_tensor())

    def test_wait_all_workers_twice_sparse(self):
        # 调用 _wait_all_workers_twice 方法，传入 heavy_rpc_sparse 和一个稀疏张量作为参数
        self._wait_all_workers_twice(heavy_rpc_sparse, build_sparse_tensor())

    @dist_init
    def test_py_sparse_tensors_in_container(self):
        # 使用 dist_init 装饰器初始化分布式环境
        # 计算当前节点的排名并计算目标节点的排名
        n = self.rank + 1
        dst_rank = n % self.world_size
        # 构建包含两个稀疏张量的列表
        a = [build_sparse_tensor(), build_sparse_tensor()]
        # 使用 rpc.rpc_sync 方法调用目标节点的 my_container_sum 函数，传入列表 a 作为参数
        ret = rpc.rpc_sync(
            worker_name(dst_rank), my_container_sum, args=(a,)
        )
        # 断言返回值与 my_container_sum(a) 相等
        self.assertEqual(ret, my_container_sum(a))

    @dist_init
    def test_nested_rpc_sparse(self):
        # 使用 dist_init 装饰器初始化分布式环境
        # 调用 _nested_rpc 方法，传入 nested_rpc_sparse 和两倍的稀疏张量作为参数
        self._nested_rpc(nested_rpc_sparse, build_sparse_tensor() * 2)

    @dist_init
    def test_stress_heavy_rpc_sparse(self):
        # 使用 dist_init 装饰器初始化分布式环境
        # 调用 _stress_test_rpc 方法，传入 heavy_rpc_sparse 、重复次数为 20 和稀疏张量作为参数
        self._stress_test_rpc(heavy_rpc_sparse, repeat=20, args=(build_sparse_tensor(),))

    @dist_init
    def test_builtin_remote_ret_sparse(self):
        # 使用 dist_init 装饰器初始化分布式环境
        # 调用 _builtin_remote_ret 方法，传入三个稀疏张量作为参数
        self._builtin_remote_ret(
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor() * 2
        )

    @dist_init
    def test_builtin_remote_self_sparse(self):
        # 使用 dist_init 装饰器初始化分布式环境
        # 调用 _builtin_remote_self 方法，传入三个稀疏张量作为参数
        self._builtin_remote_self(
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor() * 2
        )
    def test_multi_builtin_remote_ret_sparse(self):
        # 调用 _test_multi_remote_call 方法，使用 torch.add 函数作为远程调用的目标，返回稀疏结果
        self._test_multi_remote_call(
            torch.add, True,
            args_fn=RpcTest._multi_args_fn
        )

    @dist_init
    def test_multi_py_udf_remote_sparse(self):
        # 调用 _test_multi_remote_call 方法，使用自定义函数 my_function 作为远程调用的目标，返回稀疏结果
        self._test_multi_remote_call(
            my_function,
            True,
            kwargs_fn=RpcTest._multi_kwargs_fn
        )

    @dist_init
    def test_py_rref_args_sparse(self):
        # 调用 _py_rref_args 方法，传递多个稀疏张量作为参数
        self._py_rref_args(
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor() * 4
        )

    @dist_init
    def test_py_rref_args_user_share_sparse(self):
        # 调用 _py_rref_args_user_share 方法，传递多个稀疏张量作为参数
        self._py_rref_args_user_share(
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor() * 6
        )

    @dist_init
    def test_py_rpc_rref_args_sparse(self):
        # 调用 _py_rpc_rref_args 方法，传递多个稀疏张量作为参数
        self._py_rpc_rref_args(
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor() * 6
        )

    @dist_init
    def test_nested_remote_sparse(self):
        # 调用 _nested_remote 方法，传递 nested_remote_sparse 函数和两个稀疏张量的加法结果作为参数
        self._nested_remote(
            nested_remote_sparse,
            build_sparse_tensor() + build_sparse_tensor()
        )

    @dist_init
    def test_nested_rref_sparse(self):
        # 调用 _nested_rref 方法，传递 nested_rref_sparse 函数和两个稀疏张量的乘法结果作为参数
        self._nested_rref(
            nested_rref_sparse,
            build_sparse_tensor() * 2,
            build_sparse_tensor() * 2
        )

    @dist_init
    def test_nested_rref_stress_sparse(self):
        # 调用 _nested_rref_stress 方法，传递 nested_rref_sparse 函数和两个稀疏张量的乘法结果作为参数
        self._nested_rref_stress(
            nested_rref_sparse,
            build_sparse_tensor() * 2,
            build_sparse_tensor() * 2
        )

    @dist_init
    def test_my_parameter_server_sparse(self):
        # 调用 _my_parameter_server 方法，传递 True 作为参数
        self._my_parameter_server(True)

    # Test init_rpc without world_size argument
    @dist_init(setup_rpc=False)
    def test_dynamic_rpc_init_rpc(self):
        # 使用 rpc.init_rpc 初始化 RPC，使用 worker_name(self.rank) 生成 worker 的名称，并配置相关选项
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            rpc_backend_options=self.rpc_backend_options,
        )
        # 关闭 RPC
        rpc.shutdown()

    # Dynamic RPC new ranks communicate with existing ranks
    @dist_init(setup_rpc=False)
    # 定义一个测试方法，用于测试动态 RPC 中新加入的进程能否与现有的进程通信
    def test_dynamic_rpc_new_rank_can_communicated_with_existing_rank(self):
        # 初始化进程组
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        # 如果当前进程的 rank 是 0
        if self.rank == 0:
            # 初始化 RPC，为 rank 0 创建 RPC 环境
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                rpc_backend_options=self.rpc_backend_options,
            )

        # 等待所有进程初始化完成
        dist.barrier()

        # 对于非 rank 0 的进程
        if self.rank != 0:
            # 初始化 RPC，新加入的进程可以与 rank 0 通信
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                rpc_backend_options=self.rpc_backend_options,
            )
            # 使用 RPC 调用 rank 0 进程上的 torch.add 函数
            result = rpc.rpc_sync(worker_name(0), torch.add, args=(torch.tensor(1), torch.tensor(1)))
            # 断言结果与预期相等
            self.assertEqual(torch.add(torch.tensor(1), torch.tensor(1)), result)

        # 等待所有 rpc_sync 调用完成
        dist.barrier()
        # 关闭 RPC
        rpc.shutdown()

    # Dynamic RPC 中现有进程可以与新加入的进程通信
    @dist_init(setup_rpc=False)
    def test_dynamic_rpc_existing_rank_can_communicate_with_new_rank(self):
        # 初始化进程组
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        # 如果当前进程的 rank 是 0
        if self.rank == 0:
            # 初始化 RPC，为 rank 0 创建 RPC 环境
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                rpc_backend_options=self.rpc_backend_options,
            )

        # 等待所有进程初始化完成
        dist.barrier()

        # 对于非 rank 0 的进程
        if self.rank != 0:
            # 初始化 RPC，新加入的进程可以与 rank 0 通信
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                rpc_backend_options=self.rpc_backend_options,
            )

        # 等待所有 rpc_sync 调用完成
        dist.barrier()

        # 如果当前进程的 rank 是 0
        if self.rank == 0:
            # 对所有其他 rank 发起 RPC 调用，调用 torch.add 函数
            for i in range(1, self.world_size):
                result = rpc.rpc_sync(worker_name(i), torch.add, args=(torch.tensor(1), torch.tensor(1)))
                # 断言结果与预期相等
                self.assertEqual(torch.add(torch.tensor(1), torch.tensor(1)), result)

        # 等待所有 rpc_sync 调用完成
        dist.barrier()
        # 关闭 RPC
        rpc.shutdown()

    # Dynamic RPC 中现有进程可以与新加入的进程使用 CUDA RPC 进行通信
    @skip_if_lt_x_gpu(2)
    @dist_init(setup_rpc=False)
    # 定义测试函数，用于测试动态 RPC 中现有的 rank 是否能够与新加入的 rank 通过 CUDA 进行通信
    def test_dynamic_rpc_existing_rank_can_communicate_with_new_rank_cuda(self):
        # 使用指定的初始化方法和参数初始化进程组
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        # 如果当前 rank 是 0
        if self.rank == 0:
            # 获取 RPC 后端的选项
            options = self.rpc_backend_options
            # 遍历除了自己以外的所有 rank
            for i in range(1, self.world_size):
                # 获取第 i 个 worker 的名称
                dst = worker_name(i)
                # 设置从当前设备(1)到目标设备(0)的映射
                options.set_device_map(dst, {1: 0})
                # 设置从当前设备(0)到目标设备(1)的映射
                options.set_device_map(dst, {0: 1})
            # 初始化当前 rank 的 RPC
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                rpc_backend_options=options,
            )

        # 等待所有进程到达同步点
        dist.barrier()

        # 如果当前 rank 不是 0
        if self.rank != 0:
            # 初始化当前 rank 的 RPC，使用预设的 RPC 后端选项
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                rpc_backend_options=self.rpc_backend_options,
            )

        # 屏蔽下面未完成的 CUDA RPC 测试代码段
        # TODO: Cuda RPC is failing due to:
        # terminate called after throwing an instance of 'c10::Error'
        # what():  0 <= device && static_cast<size_t>(device) < device_allocator.size()
        # INTERNAL ASSERT FAILED at "../c10/cuda/CUDACachingAllocator.cpp":1937,
        # please report a bug to PyTorch. Allocator not initialized for device 1: did you call init?
        # dist.barrier()
        # if self.rank == 0:
        #     for i in range(1, self.world_size):
        #         x = torch.ones(2)
        #         result_on_device_0 = rpc.rpc_sync(worker_name(i), torch.add, args=(x.to(0), 1))
        #         result_on_device_1 = rpc.rpc_sync(worker_name(i), torch.add, args=(x.to(1), 1))
        #         self.assertEqual(torch.add(torch.ones(2), 1), result_on_device_0)
        #         self.assertEqual(torch.device('cuda:0'), result_on_device_0.device)
        #         self.assertEqual(torch.add(torch.ones(2), 1), result_on_device_1)
        #         self.assertEqual(torch.device('cuda:1'), result_on_device_1.device)

        # 等待所有进程到达同步点，确保所有 rpc_sync 调用已完成
        dist.barrier()
        # 关闭 RPC
        rpc.shutdown()

    @dist_init(setup_rpc=False)
    def test_dynamic_rpc_init_rpc_without_rank(self):
        # 没有指定 rank 参数时，使用文件初始化会抛出 ValueError 异常
        with self.assertRaisesRegex(ValueError, "rank parameter missing"):
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rpc_backend_options=self.rpc_backend_options,
            )

        # 使用环境变量初始化时，如果缺少 RANK 变量会抛出 ValueError 异常
        with self.assertRaisesRegex(ValueError, "environment variable RANK expected"):
            rpc_backend_options = rpc.TensorPipeRpcBackendOptions(init_method="env://")
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rpc_backend_options=rpc_backend_options,
            )

        # 使用 TCP 初始化时，如果缺少 rank 参数会抛出 ValueError 异常
        with self.assertRaisesRegex(ValueError, "rank parameter missing"):
            rpc_backend_options = rpc.TensorPipeRpcBackendOptions(init_method="tcp://127.0.0.1:23456")
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rpc_backend_options=rpc_backend_options,
            )

    @dist_init(setup_rpc=False)
    def test_dynamic_and_static_init_rpc_together(self):
        # 使用 gloo 后端初始化静态 RPC 组，rank 和 world_size 由参数指定
        dist.init_process_group(
            backend='gloo',
            init_method=self.file_init_method,
            rank=self.rank,
            world_size=self.world_size)

        world_size_minus_one = self.world_size - 1
        if self.rank < world_size_minus_one:
            # 初始化动态 RPC 组成员，rank 和 world_size 由参数指定
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=world_size_minus_one,
                rpc_backend_options=self.rpc_backend_options,
            )

        dist.barrier()

        # 尝试添加额外的动态组成员时，如果静态和动态初始化的成员混合，会抛出 RuntimeError 异常
        if self.rank == world_size_minus_one:
            with self.assertRaisesRegex(RuntimeError, "RPC group mixes statically and dynamically initialized members which is not supported."):
                rpc.init_rpc(
                    name=worker_name(self.rank),
                    backend=self.rpc_backend,
                    rank=self.rank,
                    rpc_backend_options=self.rpc_backend_options,
                )
# 定义一个测试类，继承自 RpcAgentTestFixture 和 RpcTestCommon
class TensorPipeAgentCudaRpcTest(RpcAgentTestFixture, RpcTestCommon):

    # 测试设备映射函数
    def _test_device_maps(self, options, errMsg):
        # 使用断言检查是否抛出预期的 ValueError 异常，并且异常信息要匹配给定的 errMsg
        with self.assertRaisesRegex(ValueError, errMsg):
            # 初始化 RPC，设置节点名、后端、排名、总节点数和 RPC 后端选项
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=options,
            )

        # 断言当前 RPC 代理是否未设置
        self.assertFalse(rpc.api._is_current_rpc_agent_set())

    # 跳过 GPU 少于 2 个的测试
    @skip_if_lt_x_gpu(2)
    def test_device_maps_wrong_worker_name(self):
        # 复制 RPC 后端选项
        options = self.rpc_backend_options
        # 设置设备映射为一个不存在的节点名和映射关系 {0: 1}
        options.set_device_map("none_exist", {0: 1})

        # 调用 _test_device_maps 函数进行测试，预期出错信息为指定的错误消息
        self._test_device_maps(
            options,
            errMsg="Node worker0 has invalid target node names in its device maps"
        )

    # 跳过 GPU 少于 1 个的测试
    @skip_if_lt_x_gpu(1)
    def test_device_maps_invalid_max_local_device(self):
        # 复制 RPC 后端选项
        options = self.rpc_backend_options
        # 确定目标节点为当前节点排名加一取余总节点数后的节点名
        dst = worker_name((self.rank + 1) % self.world_size)
        # 设置设备映射为目标节点和映射关系 {torch.cuda.device_count(): 0}
        options.set_device_map(dst, {torch.cuda.device_count(): 0})

        # 调用 _test_device_maps 函数进行测试，预期出错信息为指定的错误消息
        self._test_device_maps(
            options,
            errMsg="Node worker0 has source devices with invalid indices in its device map for worker1"
        )

    # 跳过 GPU 少于 1 个的测试
    @skip_if_lt_x_gpu(1)
    def test_device_maps_invalid_max_remote_device(self):
        # 复制 RPC 后端选项
        options = self.rpc_backend_options
        # 确定目标节点为当前节点排名加一取余总节点数后的节点名
        dst = worker_name((self.rank + 1) % self.world_size)
        # 设置设备映射为目标节点和映射关系 {0: torch.cuda.device_count()}
        options.set_device_map(dst, {0: torch.cuda.device_count()})

        # 调用 _test_device_maps 函数进行测试，预期出错信息为指定的错误消息
        self._test_device_maps(
            options,
            errMsg="Node worker0 has target devices with invalid indices in its device map for worker1"
        )

    # 跳过 GPU 少于 2 个的测试
    @skip_if_lt_x_gpu(2)
    def test_device_maps_many_to_one(self):
        # 复制 RPC 后端选项
        options = self.rpc_backend_options
        # 确定目标节点为当前节点排名加一取余总节点数后的节点名
        dst = worker_name((self.rank + 1) % self.world_size)
        # 设置设备映射为目标节点和两个映射关系 {1: 0} 和 {0: 0}（覆盖上一个映射）
        options.set_device_map(dst, {1: 0})
        options.set_device_map(dst, {0: 0})

        # 调用 _test_device_maps 函数进行测试，预期出错信息为指定的错误消息
        self._test_device_maps(
            options,
            errMsg="Node worker0 has duplicated target devices in its device map for worker1"
        )

    # 跳过 GPU 少于 2 个的测试
    @skip_if_lt_x_gpu(2)
    def test_device_maps_one_to_many(self):
        # 如果当前节点排名为 0
        if self.rank == 0:
            # 复制 RPC 后端选项
            options = self.rpc_backend_options
            # 确定目标节点为当前节点排名加一取余总节点数后的节点名
            dst = worker_name((self.rank + 1) % self.world_size)
            # 设置设备映射为目标节点和映射关系 {0: 1}
            options.set_device_map(dst, {0: 1})
            # 使用断言检查是否抛出预期的 ValueError 异常，并且异常信息要匹配给定的错误消息
            with self.assertRaisesRegex(
                ValueError, "`set_device_map` only supports 1-to-1 mapping"
            ):
                # 尝试设置一个非法的一对多映射关系
                options.set_device_map(dst, {0: 0})
    # 测试设备映射中的无效最小设备索引情况
    def test_device_maps_invalid_min_device(self):
        # 获取RPC后端选项
        options = self.rpc_backend_options
        # 计算目标工作节点名称
        dst = worker_name((self.rank + 1) % self.world_size)
        # 断言捕获运行时错误，并检查错误消息是否包含“Device index must not be negative”
        with self.assertRaisesRegex(
            RuntimeError, "Device index must not be negative"
        ):
            # 设置设备映射，其中包含无效的设备索引(-1: 0)
            options.set_device_map(dst, {-1: 0})

        with self.assertRaisesRegex(
            RuntimeError, "Device index must not be negative"
        ):
            # 设置设备映射，其中包含无效的设备索引(0: -1)
            options.set_device_map(dst, {0: -1})

    # 静态方法：在GPU上执行加法操作
    @staticmethod
    def _gpu_add(x, y):
        # 检查x和y是否在CUDA上，并且它们的设备索引都为1
        if all([x.is_cuda, x.device.index == 1, y.is_cuda, y.device.index == 1]):
            # 返回x和y的加法结果，并将结果转移到设备索引为0的位置
            return (x + y).to(0)
        else:
            # 抛出数值错误，指示设备亲和性错误
            raise ValueError("Wrong device affinity")

    # 如果GPU数目小于2，则跳过测试
    @skip_if_lt_x_gpu(2)
    def test_device_maps_gpu(self):
        # 获取RPC后端选项
        options = self.rpc_backend_options
        # 计算目标工作节点名称
        dst = worker_name((self.rank + 1) % self.world_size)
        # 设置设备映射，将设备0映射到设备1，将设备1映射到设备0
        options.set_device_map(dst, {0: 1, 1: 0})

        # 初始化RPC
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        # 使用RPC同步调用执行GPU加法测试
        ret = rpc.rpc_sync(
            dst,
            TensorPipeAgentCudaRpcTest._gpu_add,
            args=(torch.zeros(2).to(0), torch.ones(2).to(0))
        )
        # 断言返回值的设备为torch.device(1)
        self.assertEqual(ret.device, torch.device(1))
        # 断言返回值等于(torch.zeros(2) + torch.ones(2)).to(1)
        self.assertEqual(ret, (torch.zeros(2) + torch.ones(2)).to(1))
        # 关闭RPC
        rpc.shutdown()

    # 静态方法：根据给定设备执行GPU加法操作
    @staticmethod
    def _gpu_add_given_devices(x, y, x_to, y_to, z_to):
        # 获取x和y的设备
        x_device = "cpu" if x.device.type == "cpu" else x.device.index
        y_device = "cpu" if y.device.type == "cpu" else y.device.index
        # 检查x和y的设备是否与目标设备匹配
        if x_device == x_to and y_device == y_to:
            # 返回x和y的加法结果，并将结果转移到z_to指定的设备
            return x.to(z_to) + y.to(z_to)
        else:
            # 抛出数值错误，指示设备亲和性错误
            raise ValueError("Wrong device affinity")

    # 测试使用设备映射在GPU上执行操作
    def _test_device_maps_gpu(self, x_from, y_from, z_to, device_map, dst=None, fn=None):
        # 如果fn未指定，则使用_gpu_add_given_devices作为默认函数
        fn = TensorPipeAgentCudaRpcTest._gpu_add_given_devices if fn is None else fn
        # 根据设备映射获取x和y的目标设备索引
        x_to = device_map[x_from]
        y_to = device_map[y_from]

        # 获取RPC后端选项
        options = self.rpc_backend_options
        # 如果目标工作节点名称未指定，则计算其值
        dst = worker_name((self.rank + 1) % self.world_size) if dst is None else dst
        # 设置设备映射
        options.set_device_map(dst, device_map)

        # 初始化RPC
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        # 创建在x_from设备上的torch张量x和y
        x = torch.zeros(2).to(x_from)
        y = torch.ones(2).to(y_from)

        # 使用RPC同步调用执行fn函数，传递x、y和目标设备索引
        ret = rpc.rpc_sync(dst, fn, args=(x, y, x_to, y_to, z_to))

        # 反向设备映射，用于确定返回值的设备
        reverse_device_map = {device_map[k] : k for k in device_map}
        z_from = reverse_device_map[z_to]

        # 获取返回值的设备索引
        ret_device = "cpu" if ret.device.type == "cpu" else ret.device.index
        # 断言返回值的设备与z_from匹配
        self.assertEqual(ret_device, z_from)
        # 断言返回值等于torch.ones(2).to(z_from)
        self.assertEqual(ret, torch.ones(2).to(z_from))

        # 关闭RPC
        rpc.shutdown()
    @skip_if_lt_x_gpu(2)
    # 使用装饰器，如果 GPU 数量小于 2，则跳过测试
    def test_device_map_cpu(self):
        # 调用 _test_device_maps_gpu 方法测试 CPU 设备映射
        self._test_device_maps_gpu(
            x_from="cpu",       # 源张量从 CPU 发送
            y_from="cpu",       # 源张量从 CPU 发送
            z_to="cpu",         # 目标设备为 CPU
            device_map={"cpu" : "cpu"},  # 设备映射，CPU 到 CPU
            fn=TensorPipeAgentCudaRpcTest._gpu_add_given_devices,  # 调用的函数
        )

    @skip_if_lt_x_gpu(1)
    # 使用装饰器，如果 GPU 数量小于 1，则跳过测试
    def test_device_map_cpu_to_gpu_default(self):
        # 调用 _test_device_maps_gpu 方法测试 CPU 到 GPU 的默认映射
        self._test_device_maps_gpu(
            x_from="cpu",       # 源张量从 CPU 发送
            y_from="cpu",       # 源张量从 CPU 发送
            z_to=0,             # 目标设备为 GPU 0
            device_map={"cpu" : 0},     # 设备映射，CPU 到 GPU 0
            fn=TensorPipeAgentCudaRpcTest._gpu_add_given_devices,  # 调用的函数
        )

    @skip_if_lt_x_gpu(2)
    # 使用装饰器，如果 GPU 数量小于 2，则跳过测试
    def test_device_map_cpu_to_gpu_non_default(self):
        # 调用 _test_device_maps_gpu 方法测试 CPU 到 GPU 的非默认映射
        self._test_device_maps_gpu(
            x_from="cpu",       # 源张量从 CPU 发送
            y_from="cpu",       # 源张量从 CPU 发送
            z_to=1,             # 目标设备为 GPU 1
            device_map={"cpu" : 1},     # 设备映射，CPU 到 GPU 1
            fn=TensorPipeAgentCudaRpcTest._gpu_add_given_devices,  # 调用的函数
        )

    @skip_if_lt_x_gpu(1)
    # 使用装饰器，如果 GPU 数量小于 1，则跳过测试
    def test_device_map_gpu_to_cpu_default(self):
        # 调用 _test_device_maps_gpu 方法测试 GPU 到 CPU 的默认映射
        self._test_device_maps_gpu(
            x_from=0,           # 源张量从 GPU 0 发送
            y_from=0,           # 源张量从 GPU 0 发送
            z_to="cpu",         # 目标设备为 CPU
            device_map={0 : "cpu"},     # 设备映射，GPU 0 到 CPU
            fn=TensorPipeAgentCudaRpcTest._gpu_add_given_devices,  # 调用的函数
        )

    @skip_if_lt_x_gpu(2)
    # 使用装饰器，如果 GPU 数量小于 2，则跳过测试
    def test_device_map_gpu_to_cpu_non_default(self):
        # 调用 _test_device_maps_gpu 方法测试 GPU 到 CPU 的非默认映射
        self._test_device_maps_gpu(
            x_from=1,           # 源张量从 GPU 1 发送
            y_from=1,           # 源张量从 GPU 1 发送
            z_to="cpu",         # 目标设备为 CPU
            device_map={1 : "cpu"},     # 设备映射，GPU 1 到 CPU
            fn=TensorPipeAgentCudaRpcTest._gpu_add_given_devices,  # 调用的函数
        )

    @skip_if_lt_x_gpu(2)
    # 使用装饰器，如果 GPU 数量小于 2，则跳过测试
    def test_device_map_gpu_default(self):
        # 调用 _test_device_maps_gpu 方法测试 GPU 到 GPU 的默认映射
        self._test_device_maps_gpu(
            x_from=0,           # 源张量从 GPU 0 发送
            y_from=0,           # 源张量从 GPU 0 发送
            z_to=0,             # 目标设备为 GPU 0
            device_map={0 : 0}  # 设备映射，GPU 0 到 GPU 0
        )

    @skip_if_lt_x_gpu(2)
    # 使用装饰器，如果 GPU 数量小于 2，则跳过测试
    def test_device_map_gpu_non_default(self):
        # 调用 _test_device_maps_gpu 方法测试 GPU 到 GPU 的非默认映射
        self._test_device_maps_gpu(
            x_from=1,           # 源张量从 GPU 1 发送
            y_from=1,           # 源张量从 GPU 1 发送
            z_to=1,             # 目标设备为 GPU 1
            device_map={1 : 1}  # 设备映射，GPU 1 到 GPU 1
        )

    @skip_if_lt_x_gpu(2)
    # 使用装饰器，如果 GPU 数量小于 2，则跳过测试
    def test_device_map_gpu_default_to_non_default(self):
        # 调用 _test_device_maps_gpu 方法测试 GPU 的默认到非默认映射
        self._test_device_maps_gpu(
            x_from=0,           # 源张量从 GPU 0 发送
            y_from=0,           # 源张量从 GPU 0 发送
            z_to=1,             # 目标设备为 GPU 1
            device_map={0 : 1}  # 设备映射，GPU 0 到 GPU 1
        )

    @skip_if_lt_x_gpu(2)
    # 使用装饰器，如果 GPU 数量小于 2，则跳过测试
    def test_device_map_gpu_non_default_to_default(self):
        # 调用 _test_device_maps_gpu 方法测试 GPU 的非默认到默认映射
        self._test_device_maps_gpu(
            x_from=1,           # 源张量从 GPU 1 发送
            y_from=1,           # 源张量从 GPU 1 发送
            z_to=0,             # 目标设备为 GPU 0
            device_map={1 : 0}  # 设备映射，GPU 1 到 GPU 0
        )

    @skip_if_lt_x_gpu(2)
    # 使用装饰器，如果 GPU 数量小于 2，则跳过测试
    def test_device_map_gpu_mixed_1(self):
        # 调用 _test_device_maps_gpu 方法测试混合 GPU 设备映射情况 1
        self._test_device_maps_gpu(
            x_from=0,           # 源张量从 GPU 0 发送
            y_from=1,           # 源张量从 GPU 1 发送
            z_to=0,             # 目标设备为 GPU 0
            device_map={0 : 0, 1 : 1}  # 设备映射，GPU 0 到 GPU 0，GPU 1 到 GPU 1
        )

    @skip_if_lt_x_gpu(2)
    # 使用装饰器，如果 GPU 数量小于 2，则跳过测试
    def test_device_map_gpu_mixed_2(self):
        # 调用 _test_device_maps_gpu 方法测试混合 GPU 设备映射情况 2
        self._test_device_maps_gpu(
            x_from=0,           # 源张量从 GPU 0 发送
            y_from=1,           # 源张量从 GPU 1 发送
            z_to=1,             # 目标设备为 GPU 1
            device_map={0 : 0, 1 : 1}  # 设备映射，GPU 0 到 GPU 0，GPU 1 到 GPU 1
        )

    @skip_if_lt_x_gpu(2)
    # 使用装饰器，如果 GPU 数量小于 2，则跳过测试
    def test_device_map_gpu_mixed_3(self):
        # 调用 _test_device_maps_gpu 方法测试混合 GPU 设备映射情况 3
        self._test_device_maps_gpu(
            x_from=1,           # 源张量从 GPU 1 发送
            y_from=0,           # 源张
    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_4(self):
        self._test_device_maps_gpu(
            x_from=1,  # 设置 x_from 参数为 1
            y_from=0,  # 设置 y_from 参数为 0
            z_to=1,    # 设置 z_to 参数为 1
            device_map={0 : 0, 1 : 1}  # 设置 device_map 参数为 {0: 0, 1: 1}
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_5(self):
        self._test_device_maps_gpu(
            x_from=0,  # 设置 x_from 参数为 0
            y_from=1,  # 设置 y_from 参数为 1
            z_to=0,    # 设置 z_to 参数为 0
            device_map={0 : 1, 1 : 0}  # 设置 device_map 参数为 {0: 1, 1: 0}
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_6(self):
        self._test_device_maps_gpu(
            x_from=0,  # 设置 x_from 参数为 0
            y_from=1,  # 设置 y_from 参数为 1
            z_to=1,    # 设置 z_to 参数为 1
            device_map={0 : 1, 1 : 0}  # 设置 device_map 参数为 {0: 1, 1: 0}
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_7(self):
        self._test_device_maps_gpu(
            x_from=1,  # 设置 x_from 参数为 1
            y_from=0,  # 设置 y_from 参数为 0
            z_to=0,    # 设置 z_to 参数为 0
            device_map={0 : 1, 1 : 0}  # 设置 device_map 参数为 {0: 1, 1: 0}
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_8(self):
        self._test_device_maps_gpu(
            x_from=1,  # 设置 x_from 参数为 1
            y_from=0,  # 设置 y_from 参数为 0
            z_to=1,    # 设置 z_to 参数为 1
            device_map={0 : 1, 1 : 0}  # 设置 device_map 参数为 {0: 1, 1: 0}
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_1(self):
        self._test_device_maps_gpu(
            x_from=0,                       # 设置 x_from 参数为 0
            y_from=1,                       # 设置 y_from 参数为 1
            z_to=0,                         # 设置 z_to 参数为 0
            device_map={0 : 0, 1 : 1},       # 设置 device_map 参数为 {0: 0, 1: 1}
            dst=worker_name(self.rank)      # 设置 dst 参数为 worker_name(self.rank) 的返回值
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_2(self):
        self._test_device_maps_gpu(
            x_from=0,                       # 设置 x_from 参数为 0
            y_from=1,                       # 设置 y_from 参数为 1
            z_to=1,                         # 设置 z_to 参数为 1
            device_map={0 : 0, 1 : 1},       # 设置 device_map 参数为 {0: 0, 1: 1}
            dst=worker_name(self.rank)      # 设置 dst 参数为 worker_name(self.rank) 的返回值
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_3(self):
        self._test_device_maps_gpu(
            x_from=1,                       # 设置 x_from 参数为 1
            y_from=0,                       # 设置 y_from 参数为 0
            z_to=0,                         # 设置 z_to 参数为 0
            device_map={0 : 0, 1 : 1},       # 设置 device_map 参数为 {0: 0, 1: 1}
            dst=worker_name(self.rank)      # 设置 dst 参数为 worker_name(self.rank) 的返回值
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_4(self):
        self._test_device_maps_gpu(
            x_from=1,                       # 设置 x_from 参数为 1
            y_from=0,                       # 设置 y_from 参数为 0
            z_to=1,                         # 设置 z_to 参数为 1
            device_map={0 : 0, 1 : 1},       # 设置 device_map 参数为 {0: 0, 1: 1}
            dst=worker_name(self.rank)      # 设置 dst 参数为 worker_name(self.rank) 的返回值
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_5(self):
        self._test_device_maps_gpu(
            x_from=0,                       # 设置 x_from 参数为 0
            y_from=1,                       # 设置 y_from 参数为 1
            z_to=0,                         # 设置 z_to 参数为 0
            device_map={0 : 1, 1 : 0},       # 设置 device_map 参数为 {0: 1, 1: 0}
            dst=worker_name(self.rank)      # 设置 dst 参数为 worker_name(self.rank) 的返回值
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_6(self):
        self._test_device_maps_gpu(
            x_from=0,                       # 设置 x_from 参数为 0
            y_from=1,                       # 设置 y_from 参数为 1
            z_to=1,                         # 设置 z_to 参数为 1
            device_map={0 : 1, 1 : 0},       # 设置 device_map 参数为 {0: 1, 1: 0}
            dst=worker_name(self.rank)      # 设置 dst 参数为 worker_name(self.rank) 的返回值
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_7(self):
        self._test_device_maps_gpu(
            x_from=1,                       # 设置 x_from 参数为 1
            y_from=0,                       # 设置 y_from 参数为 0
            z_to=0,                         # 设置 z_to 参数为 0
            device_map={0 : 1, 1 : 0},       # 设置 device_map 参数为 {0: 1, 1: 0}
            dst=worker_name(self.rank)      # 设置 dst 参数为 worker_name(self.rank) 的返回值
        )
    def test_device_map_gpu_mixed_self_8(self):
        # 调用 _test_device_maps_gpu 方法，测试混合 GPU 设备映射
        self._test_device_maps_gpu(
            x_from=1,  # 源设备为 GPU 1
            y_from=0,  # 源设备为 GPU 0
            z_to=1,    # 目标设备为 GPU 1
            device_map={0: 1, 1: 0},  # 设备映射字典，从 GPU 0 映射到 GPU 1，从 GPU 1 映射到 GPU 0
            dst=worker_name(self.rank)  # 目标 worker 名称为当前 rank 的 worker 名称
        )

    @staticmethod
    def _gpu_add_multi_gpu(x, y):
        # 检查 x 和 y 是否在正确的 GPU 设备上，并进行相应的加法和减法操作
        if all([x.is_cuda, x.device.index == 1, y.is_cuda, y.device.index == 0]):
            return x.to(0) + y, x - y.to(1)
        else:
            raise ValueError("Wrong device affinity")  # 抛出设备亲和性错误

    def _test_device_maps_multi_gpu(self, dst):
        # 获取 RPC 后端选项
        options = self.rpc_backend_options
        # 设置设备映射，将 GPU 0 映射到 GPU 1
        options.set_device_map(dst, {0: 1})
        # 设置设备映射，将 GPU 1 映射到 GPU 0
        options.set_device_map(dst, {1: 0})

        # 初始化 RPC
        rpc.init_rpc(
            name=worker_name(self.rank),  # 当前 worker 的名称
            backend=self.rpc_backend,     # RPC 后端
            rank=self.rank,               # 当前 worker 的排名
            world_size=self.world_size,   # RPC 群集的大小
            rpc_backend_options=options,  # RPC 后端选项
        )

        # 在 GPU 0 上创建全零张量 x
        x = torch.zeros(2).to(0)
        # 在 GPU 1 上创建全一张量 y
        y = torch.ones(2).to(1)
        # 使用 RPC 同步调用 _gpu_add_multi_gpu 方法
        rets = rpc.rpc_sync(
            dst,  # 目标 worker 名称
            TensorPipeAgentCudaRpcTest._gpu_add_multi_gpu,  # 调用的静态方法
            args=(x, y)  # 方法参数
        )

        # 断言返回的张量在正确的设备上
        self.assertEqual(rets[0].device, torch.device(1))
        self.assertEqual(rets[1].device, torch.device(0))
        # 断言返回的张量与预期相符
        self.assertEqual(rets[0], (torch.zeros(2) + torch.ones(2)).to(1))
        self.assertEqual(rets[1], (torch.zeros(2) - torch.ones(2)).to(0))
        # 关闭 RPC
        rpc.shutdown()

    @skip_if_lt_x_gpu(2)
    def test_device_maps_multi_gpu(self):
        # 获取目标 worker 的名称
        dst = worker_name((self.rank + 1) % self.world_size)
        # 调用 _test_device_maps_multi_gpu 方法测试多 GPU 设备映射
        self._test_device_maps_multi_gpu(dst)

    @skip_if_lt_x_gpu(2)
    def test_device_maps_multi_gpu_self(self):
        # 获取目标 worker 的名称为当前 worker 的名称
        dst = worker_name(self.rank)
        # 调用 _test_device_maps_multi_gpu 方法测试多 GPU 设备映射
        self._test_device_maps_multi_gpu(dst)

    @staticmethod
    def _gpu_add_return_to_gpu(x, y):
        # 检查 x 和 y 是否在 CPU 上，并执行加、减、乘、除操作，分别放置在不同的 GPU 上
        if x.device.type == 'cpu' and y.device.type == 'cpu':
            return (x + y).to(0), (x - y).to(1), (x * y).to(2), (x / y).to(3)
        else:
            raise ValueError("Wrong device affinity")  # 抛出设备亲和性错误

    @skip_if_lt_x_gpu(2)
    # 测试在选项中设置设备映射
    def test_device_maps_in_options(self):
        # 计算目标 worker 的名称
        dst = worker_name((self.rank + 1) % self.world_size)
        # 获取 RPC 后端选项
        options = self.rpc_backend_options

        # 初始化 RPC
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                init_method=options.init_method,
                num_worker_threads=options.num_worker_threads,
                device_maps={dst: {0: 1, 1: 0}},
                _transports=tp_transports()
            )
        )

        # 进行同步 RPC 调用
        rets = rpc.rpc_sync(
            dst,
            TensorPipeAgentCudaRpcTest._gpu_add_multi_gpu,
            args=(torch.zeros(2).to(0), torch.ones(2).to(1))
        )
        # 断言返回值的设备
        self.assertEqual(rets[0].device, torch.device(1))
        self.assertEqual(rets[1].device, torch.device(0))
        self.assertEqual(rets[0], (torch.zeros(2) + torch.ones(2)).to(1))
        self.assertEqual(rets[1], (torch.zeros(2) - torch.ones(2)).to(0))
        # 关闭 RPC
        rpc.shutdown()

    # 测试设备映射返回到 GPU
    def _test_device_maps_return_to_gpu(self, dst):
        # 获取 RPC 后端选项
        options = self.rpc_backend_options

        # 设置设备映射
        options.set_device_map(dst, {0: 1})
        options.set_device_map(dst, {1: 2})
        options.set_device_map(dst, {2: 3})
        options.set_device_map(dst, {3: 0})

        # 初始化 RPC
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        # 进行同步 RPC 调用
        rets = rpc.rpc_sync(
            dst,
            TensorPipeAgentCudaRpcTest._gpu_add_return_to_gpu,
            args=(torch.zeros(2), torch.ones(2))
        )
        # 断言返回值的设备
        for i in range(len(rets)):
            self.assertEqual(rets[i].device, torch.device((3 + i) % 4))
        self.assertEqual(rets[0], (torch.zeros(2) + torch.ones(2)).to(3))
        self.assertEqual(rets[1], (torch.zeros(2) - torch.ones(2)).to(0))
        self.assertEqual(rets[2], (torch.zeros(2) * torch.ones(2)).to(1))
        self.assertEqual(rets[3], (torch.zeros(2) / torch.ones(2)).to(2))
        # 关闭 RPC
        rpc.shutdown()

    # 测试设备映射返回到 GPU
    @skip_if_lt_x_gpu(4)
    def test_device_maps_return_to_gpu(self):
        # 计算目标 worker 的名称
        dst = worker_name((self.rank + 1) % self.world_size)
        # 调用内部方法进行测试
        self._test_device_maps_return_to_gpu(dst)

    # 测试设备映射返回到 GPU（自身）
    @skip_if_lt_x_gpu(4)
    def test_device_maps_return_to_gpu_self(self):
        # 计算目标 worker 的名称
        dst = worker_name(self.rank)
        # 调用内部方法进行测试
        self._test_device_maps_return_to_gpu(dst)

    # 静态方法：将数据添加到 GPU
    @staticmethod
    def _add_to_gpu(x, y):
        return (x + y).to(0)
    # 测试在缺少配置时是否正确触发异常的情况
    def _test_device_maps_missing_config(self, mode):
        # 计算目标工作进程名称
        dst = worker_name((self.rank + 1) % self.world_size)
        # 定义错误信息字符串模板，用于异常断言
        errMsg = (
            "TensorPipe RPC backend only supports CPU tensors by default.*"
            "`set_device_map` on `TensorPipeRpcBackendOptions`"
        )

        # 使用断言检查是否抛出期望的 RuntimeError 异常，并匹配错误信息字符串模板
        with self.assertRaisesRegex(RuntimeError, errMsg):
            # 根据不同的执行模式进行远程过程调用（RPC）操作
            if mode == RPCExecMode.SYNC:
                rpc.rpc_sync(dst, torch.add, args=(torch.zeros(2).to(0), 1))
            elif mode == RPCExecMode.REMOTE:
                rpc.remote(dst, torch.add, args=(torch.zeros(2).to(0), 1)).to_here()
            else:
                raise ValueError(f"unexpected mode {mode}")

        # 确保 RPC 仍然正常工作
        ret = rpc.rpc_sync(dst, torch.add, args=(torch.ones(2), 1))
        # 断言返回值与预期结果相等
        self.assertEqual(ret, torch.ones(2) + 1)

    # 测试在缺少配置响应时是否正确触发异常的情况
    def _test_device_maps_missing_config_response(self, mode):
        # 计算目标工作进程名称
        dst = worker_name((self.rank + 1) % self.world_size)
        # 定义错误信息字符串，用于异常断言
        errMsg = "Response device mapping is not available"

        # 使用断言检查是否抛出期望的 RuntimeError 异常，并匹配错误信息字符串
        with self.assertRaisesRegex(RuntimeError, errMsg):
            # 根据不同的执行模式进行远程过程调用（RPC）操作
            if mode == RPCExecMode.SYNC:
                rpc.rpc_sync(
                    dst,
                    TensorPipeAgentCudaRpcTest._add_to_gpu,
                    args=(torch.zeros(2), 1)
                )
            elif mode == RPCExecMode.REMOTE:
                rpc.remote(
                    dst,
                    TensorPipeAgentCudaRpcTest._add_to_gpu,
                    args=(torch.zeros(2), 1)
                ).to_here()
            else:
                raise ValueError(f"unexpected mode {mode}")

        # 确保 RPC 仍然正常工作
        ret = rpc.rpc_sync(dst, torch.add, args=(torch.ones(2), 1))
        # 断言返回值与预期结果相等
        self.assertEqual(ret, torch.ones(2) + 1)

    # 在至少有一个 GPU 的条件下，初始化分布式测试环境，并执行设备映射缺失配置的同步测试
    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_device_maps_missing_config(self):
        self._test_device_maps_missing_config(RPCExecMode.SYNC)

    # 在至少有一个 GPU 的条件下，测试设备映射缺失配置且不超时的情况
    @skip_if_lt_x_gpu(1)
    def test_device_maps_missing_config_not_timeout(self):
        # 计算目标工作进程名称
        dst = worker_name((self.rank + 1) % self.world_size)
        # 获取 RPC 后端选项
        options = self.rpc_backend_options

        # 初始化 RPC
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options
        )

        # 获取 RPC 超时时间
        timeout = rpc.get_rpc_timeout()

        # 计时开始
        tik = time.time()
        # 执行设备映射缺失配置的同步测试
        self._test_device_maps_missing_config(RPCExecMode.SYNC)
        # 关闭 RPC
        rpc.shutdown()
        # 计时结束
        tok = time.time()

        # 断言 RPC 执行时间未超过超时时间
        self.assertTrue(tok - tik < timeout)

    # 在至少有一个 GPU 的条件下，初始化分布式测试环境，并执行设备映射缺失配置的循环测试
    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_device_maps_missing_config_loop(self):
        # 循环执行设备映射缺失配置的同步测试，次数为 num_worker_threads + 5
        for _ in range(self.rpc_backend_options.num_worker_threads + 5):
            self._test_device_maps_missing_config(RPCExecMode.SYNC)

    # 在至少有一个 GPU 的条件下，初始化分布式测试环境，并执行设备映射缺失配置响应的同步测试
    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_device_maps_missing_config_response(self):
        self._test_device_maps_missing_config_response(RPCExecMode.SYNC)
    # 如果当前 GPU 小于 1，则跳过测试
    @skip_if_lt_x_gpu(1)
    # 初始化分布式环境
    @dist_init
    # 测试设备映射在缺少配置时的响应循环
    def test_device_maps_missing_config_response_loop(self):
        # 根据当前 RPC 后端选项的工作线程数加上5，执行设备映射缺少配置时的响应测试
        for _ in range(self.rpc_backend_options.num_worker_threads + 5):
            self._test_device_maps_missing_config_response(RPCExecMode.SYNC)

    # 如果当前 GPU 小于 1，则跳过测试
    @skip_if_lt_x_gpu(1)
    # 初始化分布式环境
    @dist_init
    # 测试设备映射在缺少配置时的远程调用
    def test_device_maps_missing_config_remote(self):
        self._test_device_maps_missing_config(RPCExecMode.REMOTE)

    # 如果当前 GPU 小于 1，则跳过测试
    @skip_if_lt_x_gpu(1)
    # 初始化分布式环境
    @dist_init
    # 测试设备映射在缺少配置时的远程调用响应
    def test_device_maps_missing_config_remote_response(self):
        self._test_device_maps_missing_config_response(RPCExecMode.REMOTE)

    # 如果当前 GPU 小于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    # 测试远程设备映射
    def test_device_maps_remote(self):
        # 获取当前 RPC 后端选项
        options = self.rpc_backend_options
        # 计算下一个工作节点的名称
        dst = worker_name((self.rank + 1) % self.world_size)
        # 设置设备映射
        options.set_device_map(dst, {1: 0})

        # 初始化 RPC
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        # 在远程节点上执行操作
        rref = rpc.remote(
            dst,
            TensorPipeAgentCudaRpcTest._add_to_gpu,
            args=(torch.zeros(2), 1)
        )

        # 断言远程结果的设备索引为1
        self.assertEqual(rref.to_here().device.index, 1)
        # 断言远程结果与预期一致
        self.assertEqual(rref.to_here(), torch.ones(2).to(1))

        # 关闭 RPC
        rpc.shutdown()

    # 静态方法：在用户流上执行慢速加法
    @staticmethod
    def _slow_add_on_user_stream(x, y):
        # 获取当前 CUDA 流
        s0 = torch.cuda.current_stream(x.device)
        # 创建新的 CUDA 流
        s1 = torch.cuda.Stream(device=x.device)
        # 等待 s0 流完成
        s1.wait_stream(s0)
        # 记录 x 和 y 到 s1 流
        x.record_stream(s1)
        y.record_stream(s1)
        # 在 s1 流中执行慢速加法
        with torch.cuda.stream(s1):
            torch.cuda._sleep(10 * FIFTY_MIL_CYCLES)
            z = x + y
        # 等待 s1 流完成
        s0.wait_stream(s1)
        # 记录结果 z 到 s0 流
        z.record_stream(s0)
        # 返回计算结果 z
        return z

    # 测试自定义流
    def _test_custom_stream(self, fn, device_map):
        # 获取当前 RPC 后端选项
        options = self.rpc_backend_options
        # 计算下一个工作节点的名称
        dst = worker_name((self.rank + 1) % self.world_size)
        # 设置设备映射
        options.set_device_map(dst, device_map)

        # 初始化 RPC
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        # 执行传入的函数 fn
        fn(dst)

        # 关闭 RPC
        rpc.shutdown()

    # 测试同步流
    def _test_stream_sync(self, dst):
        # 创建一个大小为 2x2 的张量并放置在 GPU 0 上
        x = torch.ones(2, 2).to(0)
        # 使用 RPC 同步执行慢速加法函数
        ret = rpc.rpc_sync(
            dst,
            TensorPipeAgentCudaRpcTest._slow_add_on_user_stream,
            args=(x, x)
        )
        # 断言返回值等于 2 * x
        self.assertEqual(ret, 2 * x)

    # 如果当前 GPU 小于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    # 测试自定义流
    def test_custom_stream(self):
        # 调用 _test_custom_stream 方法，传入 _test_stream_sync 函数和设备映射字典
        self._test_custom_stream(self._test_stream_sync, {"cuda:0": "cuda:1"})
    def _test_stream_multi_async(self, dst):
        # 创建一个空列表，用于存储所有的 future 对象
        futs = []
        # 循环 20 次，创建张量并进行异步 RPC 调用
        for i in range(20):
            x = torch.ones(2, 2).to(0) * i  # 创建大小为 2x2 的张量 x
            # 将 RPC 异步调用的 future 对象添加到 futs 列表中
            futs.append(
                rpc.rpc_async(
                    dst,
                    TensorPipeAgentCudaRpcTest._slow_add_on_user_stream,
                    args=(x, x)
                )
            )

        # 循环 20 次，等待每个 future 对象完成，并进行断言
        for i in range(20):
            self.assertEqual(futs[i].wait(), 2 * torch.ones(2, 2).to(0) * i)

    @skip_if_lt_x_gpu(2)
    def test_custom_stream_multi(self):
        # 调用 _test_custom_stream 方法，传入 _test_stream_multi_async 和目标映射字典
        self._test_custom_stream(
            self._test_stream_multi_async,
            {"cuda:0": "cuda:1"}
        )

    @staticmethod
    def _nested_slow_add_on_user_stream(dst, x, y, z):
        # 在目标 dst 上同步调用 _slow_add_on_user_stream 方法，返回结果作为新的参数再次调用
        ret = rpc.rpc_sync(
            dst,
            TensorPipeAgentCudaRpcTest._slow_add_on_user_stream,
            args=(x, y)
        )

        return TensorPipeAgentCudaRpcTest._slow_add_on_user_stream(ret, z)

    def _test_stream_nested_sync(self, dst):
        # 创建大小为 2x2 的三个张量 x, y, z
        x = torch.ones(2, 2).to(0)
        y = torch.ones(2, 2).to(0) * 2
        z = torch.ones(2, 2).to(0) * 3
        # 计算并返回在目标 dst 上同步调用 _nested_slow_add_on_user_stream 的结果
        nested_dst = worker_name((self.rank + 2) % self.world_size)
        ret = rpc.rpc_sync(
            dst,
            TensorPipeAgentCudaRpcTest._nested_slow_add_on_user_stream,
            args=(nested_dst, x, y, z)
        )
        # 断言返回的结果是否与 6*x 相等
        self.assertEqual(ret, 6 * x)

    @skip_if_lt_x_gpu(2)
    def test_custom_stream_nested(self):
        # 调用 _test_custom_stream 方法，传入 _test_stream_nested_sync 和目标映射字典
        self._test_custom_stream(
            self._test_stream_nested_sync,
            {"cuda:0": "cuda:1", "cuda:1": "cuda:0"}
        )

    def _test_stream_nested_multi_async(self, dst):
        # 如果当前进程的排名为 0，则执行以下代码块
        if self.rank == 0:
            futs = []
            n = 5
            xs, ys, zs = [], [], []
            # 循环 n 次，创建大小为 2x2 的张量 x, y, z，并添加到对应的列表中
            for i in range(n):
                x = torch.ones(2, 2).to(0) * (i - 1)
                y = torch.ones(2, 2).to(0) * i
                z = torch.ones(2, 2).to(0) * (i + 1)
                xs.append(x)
                ys.append(y)
                zs.append(z)
                # 计算嵌套目标并将异步 RPC 调用的 future 对象添加到 futs 列表中
                nested_dst = worker_name((self.rank + 2) % self.world_size)
                futs.append(
                    rpc.rpc_async(
                        dst,
                        TensorPipeAgentCudaRpcTest._nested_slow_add_on_user_stream,
                        args=(nested_dst, x, y, z)
                    )
                )

            # 循环 n 次，等待每个 future 对象完成，并进行断言
            for i in range(n):
                self.assertEqual(futs[i].wait(), xs[i] + ys[i] + zs[i])

    @skip_if_lt_x_gpu(2)
    def test_custom_stream_nested_multi(self):
        # 调用 _test_custom_stream 方法，传入 _test_stream_nested_multi_async 和目标映射字典
        self._test_custom_stream(
            self._test_stream_nested_multi_async,
            {"cuda:0": "cuda:1", "cuda:1": "cuda:0"}
        )

    @staticmethod
    def _gpu_add_wrong_gpus(x, y):
        # 如果 x 和 y 都在 CUDA 设备上，则将 x 转移到 CPU，y 转移到 CUDA 设备上进行加法运算
        if x.is_cuda and y.is_cuda:
            return x.cpu() + y.cuda()
        else:
            # 否则抛出设备不匹配的异常
            raise ValueError("Wrong device affinity")

    @skip_if_lt_x_gpu(1)
    # 定义一个测试方法，用于检查设备不匹配的情况
    def test_device_mismatch(self):
        # 计算目标工作进程的名称
        dst = worker_name((self.rank + 1) % self.world_size)
        # 获取 RPC 后端选项
        options = self.rpc_backend_options
        # 设置设备映射，将本地设备 0 映射到目标设备的设备 0
        options.set_device_map(dst, {0: 0})

        # 初始化 RPC
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        # 创建一个张量 x 在设备 0 上
        x = torch.zeros(2).to(0)
        # 创建一个张量 y 在设备 0 上
        y = torch.ones(2).to(0)

        # 使用断言检查是否抛出 RuntimeError，且错误信息指出找到至少两个不同设备的张量
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected all tensors to be on the same device, but found at least two devices"
        ):
            # 调用 RPC 同步方法，尝试在目标进程上执行一个需要同一设备上的操作
            rets = rpc.rpc_sync(
                dst,
                TensorPipeAgentCudaRpcTest._gpu_add_wrong_gpus,
                args=(x, y)
            )

        # 关闭 RPC
        rpc.shutdown()

    # 定义一个私有方法，用于测试远程引用同步
    def _test_rref_synchronization(self, local_device, remote_device):
        # 计算目标工作进程的名称
        dst = worker_name((self.rank + 1) % self.world_size)
        # 获取 RPC 后端选项
        options = self.rpc_backend_options
        # 设置设备映射，将本地设备映射到远程设备
        options.set_device_map(dst, {local_device : remote_device})

        # 初始化 RPC
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        if self.rank == 1:
            # 在此测试中，比较 rref.rpc_sync().forward(x) 和 rref.remote().forward(x).to_here()
            # 如果 to_here() 与 forward(x) 同步，结果必须相同
            # 此测试需要多次迭代和大批量数据，以模拟类似于 MNIST 数据的 CNN 训练
            # 参见 https://github.com/pytorch/pytorch/issues/54771
            # 创建一个本地设备上的张量 x
            rref = rpc.remote(dst, MyConvNetForMNIST, args=(remote_device,))
            for _ in range(10):
                x = torch.randn(200, 1, 28, 28).to(local_device)
                # 获取远程结果并同步到本地
                actual = rref.remote().forward(x).to_here()
                # 使用 RPC 同步调用获取期望结果
                expected = rref.rpc_sync().forward(x)
                # 使用断言检查实际结果与期望结果是否相等
                self.assertEqual(actual, expected)

        # 关闭 RPC
        rpc.shutdown()

    # 标记为跳过条件，至少需要 1 个 GPU
    @skip_if_lt_x_gpu(1)
    def test_rref_to_here_synchronization1(self):
        # 执行 _test_rref_synchronization 方法，测试 cuda:0 到 cuda:0 的同步
        self._test_rref_synchronization("cuda:0", "cuda:0")

    # 标记为跳过条件，至少需要 2 个 GPU
    @skip_if_lt_x_gpu(2)
    def test_rref_to_here_synchronization2(self):
        # 执行 _test_rref_synchronization 方法，测试 cuda:1 到 cuda:0 的同步
        self._test_rref_synchronization("cuda:1", "cuda:0")

    # 标记为跳过条件，至少需要 2 个 GPU
    @skip_if_lt_x_gpu(2)
    def test_rref_to_here_synchronization3(self):
        # 执行 _test_rref_synchronization 方法，测试 cuda:1 到 cuda:1 的同步
        self._test_rref_synchronization("cuda:1", "cuda:1")

    # 标记为跳过条件，至少需要 2 个 GPU
    @skip_if_lt_x_gpu(2)
    def test_rref_to_here_synchronization4(self):
        # 执行 _test_rref_synchronization 方法，测试 cuda:0 到 cuda:1 的同步
        self._test_rref_synchronization("cuda:0", "cuda:1")

    # 定义一个私有方法，用于测试作为参数的远程引用同步
    def _test_rref_as_arg_synchronization(
        self,
        local_device,
        remote_device,
        devicesOptions=None
    ):
    ):
        # 计算下一个工作进程的目标名称
        dst = worker_name((self.rank + 1) % self.world_size)
        # 获取 RPC 后端选项
        options = self.rpc_backend_options
        # 设置设备映射，将本地设备映射到远程设备
        options.set_device_map(dst, {local_device: remote_device})

        # 计算上一个工作进程的源名称
        input_src = worker_name((self.rank - 1 + self.world_size) % self.world_size)
        # 设置设备映射，将远程设备映射到本地设备
        options.set_device_map(input_src, {remote_device: local_device})

        # 如果提供了设备选项，则设置设备
        if devicesOptions is not None:
            options.set_devices(devicesOptions[self.rank])

        # 初始化 RPC
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        # 如果进程排名为1，执行以下测试
        if self.rank == 1:
            # 这个测试比较 rref.rpc_sync().forward(x) 和 rref.remote().forward(x).to_here()
            # 如果 to_here() 与 forward(x) 同步正常，则结果必须相同
            # 这个测试需要多次迭代和大批量大小来模拟类似于 MNIST 的 CNN 真实训练。
            # 参见 https://github.com/pytorch/pytorch/issues/54771
            rref = rpc.remote(dst, MyConvNetForMNIST, args=(remote_device,))
            for _ in range(10):
                rref_x = RRef(torch.randn(200, 1, 28, 28).to(local_device))
                actual = rref.remote().forward(rref_x, True).to_here()
                expected = rref.rpc_sync().forward(rref_x, True)
                self.assertEqual(actual, expected)

        # 关闭 RPC
        rpc.shutdown()

    @skip_if_lt_x_gpu(1)
    def test_rref_as_arg_synchronization1(self):
        # 测试函数，测试 RRef 作为参数的同步性
        self._test_rref_as_arg_synchronization("cuda:0", "cuda:0")

    @skip_if_lt_x_gpu(2)
    def test_rref_as_arg_synchronization2(self):
        # 测试函数，测试 RRef 作为参数的同步性
        self._test_rref_as_arg_synchronization("cuda:1", "cuda:0")

    @skip_if_lt_x_gpu(2)
    def test_rref_as_arg_synchronization3(self):
        # 测试函数，测试 RRef 作为参数的同步性
        self._test_rref_as_arg_synchronization("cuda:1", "cuda:1")

    @skip_if_lt_x_gpu(2)
    def test_rref_as_arg_synchronization4(self):
        # 测试函数，测试 RRef 作为参数的同步性
        self._test_rref_as_arg_synchronization("cuda:0", "cuda:1")

    @skip_if_lt_x_gpu(1)
    def test_rref_as_arg_synchronization5(self):
        # 测试函数，测试 RRef 作为参数的同步性
        self._test_rref_as_arg_synchronization(
            "cuda:0",
            "cuda:0",
            [["cuda:0"] for _ in range(4)],  # devicesOptions
        )

    @staticmethod
    def _rref_relay(rref):
        # 静态方法，返回 RRef 的结果
        return rref.to_here()
    # 定义一个私有方法，用于测试 RRef 同步前向传播
    def _test_rref_forward_synchronization(self, local_device, remote_device):
        # 获取 RPC 后端选项
        options = self.rpc_backend_options

        # 设置输入源的 worker 名称
        input_src = worker_name(0)
        # 设置模型目标的 worker 名称
        model_dst = worker_name(1)
        # 设置输出中继的 worker 名称
        out_relay = worker_name(2)

        # 根据当前进程的 rank 执行不同的配置
        if self.rank == 0:
            # 对于 rank 为 0 的进程：
            # 1) 配置模型构建和前向执行的设备映射
            options.set_device_map(model_dst, {local_device: remote_device})

            # 2) 前向输出将首先复制到中继节点，然后返回给 worker。
            # 这是故意为之，用于测试 RRef 前向 CUDA 流的同步。
            options.set_device_map(out_relay, {local_device: local_device})
        elif self.rank == 1:
            # 对于 rank 为 1 的进程：
            # worker1 托管模型并运行前向传播。前向函数调用 RRef.to_here()，因此需要配置设备映射。
            options.set_device_map(input_src, {remote_device: local_device})
        elif self.rank == 2:
            # 对于 rank 为 2 的进程：
            # worker2 将获得输出的 RRef 并调用 to_here()，因此需要配置设备映射。
            options.set_device_map(model_dst, {local_device: remote_device})

        # 初始化 RPC
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        if self.rank == 0:
            # 如果当前进程的 rank 是 0：
            # 此测试比较 rref.rpc_sync().forward(x) 和 rref.remote().forward(x).to_here()。
            # 如果 to_here() 与 forward(x) 同步正常，结果必须完全相同。
            # 这个测试需要多次迭代和显著的批量大小，以模拟类似于 MNIST 数据的 CNN 训练。
            # 参见 https://github.com/pytorch/pytorch/issues/54771
            # 创建远程 RRef 对象，表示远程的模型和处理 CUDA 设备的参数
            rref = rpc.remote(model_dst, MyConvNetForMNIST, args=(remote_device,))
            for _ in range(10):
                # 创建 RRef 对象，表示本地设备上的随机数据输入
                rref_input = RRef(torch.randn(200, 1, 28, 28).to(local_device))
                # 调用远程模型的前向传播，并将结果发送到 out_relay 中继节点
                rref_out = rref.remote().forward(rref_input, True)
                # 将结果从中继节点传回到本地，并确保与预期结果一致
                out = rpc.remote(
                    out_relay,
                    TensorPipeAgentCudaRpcTest._rref_relay,
                    args=(rref_out,)
                ).to_here()
                # 获取预期的前向传播结果，并使用同步的方式进行比较
                expected = rref.rpc_sync().forward(rref_input, True)
                # 使用断言来检查输出结果与预期结果是否相等
                self.assertEqual(out, expected)

        # 关闭 RPC
        rpc.shutdown()

    # 如果 GPU 数量小于 1，则跳过此测试方法
    @skip_if_lt_x_gpu(1)
    def test_rref_forward_synchronization1(self):
        # 调用 _test_rref_forward_synchronization 方法，测试 CUDA 设备 0 到 CUDA 设备 0 的 RRef 同步
        self._test_rref_forward_synchronization("cuda:0", "cuda:0")

    # 如果 GPU 数量小于 2，则跳过此测试方法
    @skip_if_lt_x_gpu(2)
    def test_rref_forward_synchronization2(self):
        # 调用 _test_rref_forward_synchronization 方法，测试 CUDA 设备 0 到 CUDA 设备 1 的 RRef 同步
        self._test_rref_forward_synchronization("cuda:0", "cuda:1")

    # 如果 GPU 数量小于 2，则跳过此测试方法
    @skip_if_lt_x_gpu(2)
    def test_rref_forward_synchronization3(self):
        # 调用 _test_rref_forward_synchronization 方法，测试 CUDA 设备 1 到 CUDA 设备 0 的 RRef 同步
        self._test_rref_forward_synchronization("cuda:1", "cuda:0")

    # 如果 GPU 数量小于 2，则跳过此测试方法
    @skip_if_lt_x_gpu(2)
    def test_rref_forward_synchronization4(self):
        # 调用 _test_rref_forward_synchronization 方法，测试 CUDA 设备 1 到 CUDA 设备 1 的 RRef 同步
        self._test_rref_forward_synchronization("cuda:1", "cuda:1")
    # 测试函数，用于验证所有权RRef的前向同步功能，参数包括本地设备和远程设备
    def _test_owner_rref_forward_synchronization(self, local_device, remote_device):
        # 如果当前进程的rank为0
        if self.rank == 0:
            # 获取RPC后端选项
            options = self.rpc_backend_options
            # 设置设备映射关系，将本地设备映射到远程设备
            options.set_device_map("w0", {local_device: remote_device})
            # 初始化RPC
            rpc.init_rpc(
                "w0",
                rank=0,
                world_size=1,
                rpc_backend_options=options
            )

            # 在远程节点上创建一个线性模型
            model = rpc.remote(
                "w0", torch.nn.Linear, (2048, 20000)
            ).remote().to(remote_device)
            
            # 执行30次迭代
            for _ in range(30):
                # 生成随机数据，并将其移动到本地设备
                data = torch.rand(2048, 2048).to(local_device)
                # 在远程模型上执行同步RPC调用，并传递数据
                output = model.rpc_sync().forward(data)
                
                # 创建远程引用(RRef)对象，并计算其所有元素的和，然后将结果转移到当前节点
                v0 = rpc.RRef(output).remote().sum().to_here().item()
                # 计算本地输出张量的元素和
                v1 = output.sum().item()
                # 断言两个和相等
                self.assertEqual(v0, v1)

            # 关闭RPC
            rpc.shutdown()

    # 使用至少一个GPU时执行测试函数
    @skip_if_lt_x_gpu(1)
    def test_owner_rref_forward_synchronization1(self):
        # 调用测试函数，设定本地设备和远程设备均为cuda:0
        self._test_owner_rref_forward_synchronization("cuda:0", "cuda:0")

    # 使用至少两个GPU时执行测试函数
    @skip_if_lt_x_gpu(2)
    def test_owner_rref_forward_synchronization2(self):
        # 调用测试函数，本地设备为cuda:0，远程设备为cuda:1
        self._test_owner_rref_forward_synchronization("cuda:0", "cuda:1")

    # 使用至少两个GPU时执行测试函数
    @skip_if_lt_x_gpu(2)
    def test_owner_rref_forward_synchronization3(self):
        # 调用测试函数，本地设备为cuda:1，远程设备为cuda:0
        self._test_owner_rref_forward_synchronization("cuda:1", "cuda:0")

    # 使用至少两个GPU时执行测试函数
    @skip_if_lt_x_gpu(2)
    def test_owner_rref_forward_synchronization4(self):
        # 调用测试函数，本地设备和远程设备均为cuda:1
        self._test_owner_rref_forward_synchronization("cuda:1", "cuda:1")

    # 静态方法，返回一个Tensor视图
    @staticmethod
    def _return_tensor_view(i):
        # 在cuda:0上创建一个形状为1000x200的张量，并乘以i
        x = torch.ones(1000, 200).cuda(0) * i
        # GPU休眠，模拟计算延迟
        torch.cuda._sleep(10 * FIFTY_MIL_CYCLES)
        # 将张量按行分割成两个子张量，并返回第一个子张量
        # 返回值的序列化将创建一个新的张量视图，这在用户函数外部完成。
        return x.split(100)[0]

    # 使用至少一个GPU时执行测试函数
    @skip_if_lt_x_gpu(1)
    def test_tensor_view_as_return_value(self):
        # 获取目标工作节点的名称
        dst = worker_name((self.rank + 1) % self.world_size)
        # 获取RPC后端选项
        options = self.rpc_backend_options
        # 设置设备映射关系，将本地GPU 0映射到目标节点的GPU 0
        options.set_device_map(dst, {0 : 0})

        # 初始化RPC
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        # 初始化异步RPC调用列表
        futs = []
        for i in range(5):
            # 向目标节点异步发送RPC调用，调用_return_tensor_view方法，并传递参数i
            futs.append(rpc.rpc_async(
                dst,
                TensorPipeAgentCudaRpcTest._return_tensor_view,
                args=(i,)
            ))

        # 等待所有异步调用完成，并使用断言验证返回的张量是否与预期一致
        for i in range(5):
            self.assertEqual(torch.ones(100, 200) * i, futs[i].wait())

        # 关闭RPC
        rpc.shutdown()
    # 定义测试方法，用于验证设备选项不匹配情况
    def test_devices_option_mismatch(self):
        # 使用上下文管理器验证是否抛出预期异常类型和消息
        with self.assertRaisesRegex(
            ValueError,
            "Node worker0 has unexpected source devices in its device map for worker1"
        ):
            # 计算目标 worker 的名称
            dst = worker_name((self.rank + 1) % self.world_size)
            # 获取当前 RPC 后端选项
            options = self.rpc_backend_options
            # 设置目标 worker 的设备映射
            options.set_device_map(dst, {0 : 0})
            # 设置当前 worker 的设备列表
            options.set_devices([1])

            # 初始化 RPC
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=options,
            )

            # 关闭 RPC
            rpc.shutdown()

    # 根据条件跳过 GPU 数量小于 2 的测试
    @skip_if_lt_x_gpu(2)
    def test_devices_option_mismatch_reverse(self):
        # 使用上下文管理器验证是否抛出预期异常类型和消息
        with self.assertRaisesRegex(
            ValueError,
            "Node worker0 has unexpected target devices in its device map for worker1"
        ):
            # 计算目标 worker 的名称
            dst = worker_name((self.rank + 1) % self.world_size)

            # 设置 RPC 后端选项
            options = rpc.TensorPipeRpcBackendOptions(
                init_method=self.rpc_backend_options.init_method,
                num_worker_threads=self.rpc_backend_options.num_worker_threads,
                device_maps={dst: {0 : 1}},
                devices=[0]
            )

            # 初始化 RPC
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=options,
            )

            # 关闭 RPC
            rpc.shutdown()

    # 根据条件跳过 GPU 数量小于 1 的测试
    @skip_if_lt_x_gpu(1)
    def test_cuda_future_device_as_int(self):
        # 创建 Future 对象，指定设备为 GPU 0
        fut = Future(devices=[0])

    # 根据条件跳过 GPU 数量小于 1 的测试
    @skip_if_lt_x_gpu(1)
    def test_cuda_future_device_as_str(self):
        # 创建 Future 对象，指定设备为 "cuda:0"
        fut = Future(devices=["cuda:0"])

    # 根据条件跳过 GPU 数量小于 1 的测试
    @skip_if_lt_x_gpu(1)
    def test_cuda_future_device_as_device(self):
        # 创建 Future 对象，指定设备为 Torch 的 CUDA 设备 0
        fut = Future(devices=[torch.device("cuda", 0)])

    # 根据条件跳过 GPU 数量小于 1 的测试
    @skip_if_lt_x_gpu(1)
    def test_cuda_future_device_not_cuda(self):
        # 使用上下文管理器验证是否抛出预期异常类型和消息
        with self.assertRaisesRegex(
            ValueError, "Expected devices to have indices, got cpu"
        ):
            # 创建 Future 对象，指定设备为 "cpu"，预期抛出异常
            fut = Future(devices=["cpu"])

    # 根据条件跳过 GPU 数量小于 1 的测试
    @skip_if_lt_x_gpu(1)
    def test_cuda_future_can_extract_cuda_tensor(self):
        # 调用内部方法测试是否可以提取 CUDA 张量
        self._test_cuda_future_extraction(
            wrapper=lambda t: t, unwrapper=lambda v: v, sparse_tensor=False
        )

    # 根据条件跳过 GPU 数量小于 1 的测试
    @skip_if_lt_x_gpu(1)
    def test_cuda_future_can_extract_list_with_cuda_tensor(self):
        # 调用内部方法测试是否可以提取包含 CUDA 张量的列表
        self._test_cuda_future_extraction(
            wrapper=lambda t: [t], unwrapper=operator.itemgetter(0), sparse_tensor=False
        )

    # 根据条件跳过 GPU 数量小于 1 的测试
    @skip_if_lt_x_gpu(1)
    def test_cuda_future_can_extract_custom_class_with_cuda_tensor(self):
        # 调用内部方法测试是否可以提取自定义类中的 CUDA 张量
        self._test_cuda_future_extraction(
            wrapper=TensorWrapper, unwrapper=lambda v: v.tensor, sparse_tensor=False
        )

    # 根据条件跳过 GPU 数量小于 2 的测试
    @skip_if_lt_x_gpu(2)
    # 定义一个测试函数，用于验证 CUDA 流的正确同步性
    def test_cuda_future_callback_changes_devices(self):
        # 在一个 CUDA 设备上创建一个全零张量
        tensor0 = torch.zeros((100,), device="cuda:0")
        # 在另一个 CUDA 设备上创建一个全零张量
        tensor1 = torch.zeros((100,), device="cuda:1")
        # 创建一个 Future 对象，指定了它的设备为 cuda:0 和 cuda:1
        parent_future = Future(devices=["cuda:0", "cuda:1"])

        # 定义一个回调函数，用于处理 Future 的结果
        def cb(fut):
            # 从 Future 中获取值
            t0 = fut.value()
            # 将 t0 的值复制到 tensor1 中，非阻塞模式
            tensor1.copy_(t0, non_blocking=True)
            return tensor1

        # 创建一个子 Future，通过 parent_future 注册回调函数 cb
        child_future = parent_future.then(cb)

        # 在 cuda:0 设备上执行下面的代码块
        with torch.cuda.device("cuda:0"):
            # 创建一个 CUDA 流
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                # 睡眠以模拟一些计算时间
                torch.cuda._sleep(int(1000 * get_cycles_per_ms()))
                # 填充 tensor0 的所有元素为 1
                tensor0.fill_(1)
                # 设置 parent_future 的结果为 tensor0
                parent_future.set_result(tensor0)

        # 在 cuda:1 设备上执行下面的代码块
        with torch.cuda.device("cuda:1"):
            # 创建另一个 CUDA 流
            another_stream = torch.cuda.Stream()
            with torch.cuda.stream(another_stream):
                # 确保 child_future 的结果中所有元素都等于 1
                self.assertTrue(torch.eq(child_future.wait(), 1).all().item())

    # 如果 GPU 小于两个，则跳过测试
    @skip_if_lt_x_gpu(2)
    def test_cuda_future_value_on_bad_device(self):
        # 在 cuda:0 设备上创建一个全零张量
        tensor0 = torch.zeros((100,), device="cuda:0")
        # 在 cuda:1 设备上创建一个全零张量
        tensor1 = torch.zeros((100,), device="cuda:1")
        # 创建一个 Future 对象，指定了它的设备为 cuda:1
        parent_future = Future(devices=["cuda:1"])

        # 定义一个回调函数，用于处理 Future 的结果
        def cb(fut):
            # 在 cuda:1 设备上执行以下代码块
            with torch.cuda.device("cuda:1"):
                # 睡眠以模拟一些计算时间
                torch.cuda._sleep(int(1000 * get_cycles_per_ms()))
                # 填充 tensor1 的所有元素为 1
                tensor1.fill_(1)
                return tensor1

        # 创建一个子 Future，通过 parent_future 注册回调函数 cb
        child_future = parent_future.then(cb)

        # 在 cuda:0 设备上执行下面的代码块
        with torch.cuda.device("cuda:0"):
            # 创建一个 CUDA 流
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                # 睡眠以模拟一些计算时间
                torch.cuda._sleep(int(1000 * get_cycles_per_ms()))
                # 填充 tensor0 的所有元素为 1
                tensor0.fill_(1)
                # 设置 parent_future 的结果为 tensor0
                parent_future.set_result(tensor0)

        # 验证在设备 cuda:1 上等待 parent_future 时抛出 ValueError 异常
        with self.assertRaisesRegex(
            ValueError,
            r"The result contained tensors residing on device\(s\) cuda:0 "
            r"which are not among the expected device\(s\) cuda:1",
        ):
            parent_future.wait()

        # 在 cuda:1 设备上执行下面的代码块
        with torch.cuda.device("cuda:1"):
            # 创建另一个 CUDA 流
            another_stream = torch.cuda.Stream()
            with torch.cuda.stream(another_stream):
                # 确保 child_future 的结果中所有元素都等于 1
                self.assertTrue(torch.eq(child_future.wait(), 1).all().item())
    @skip_if_lt_x_gpu(1)
    # 装饰器：如果当前环境的 GPU 少于 1 个，则跳过该测试
    def test_async_execution_with_cuda_future(self):
        # 计算目标 worker 的名称，环绕世界大小的下一个 rank
        dst = worker_name((self.rank + 1) % self.world_size)
        # 获取 RPC 后端选项
        options = self.rpc_backend_options
        # 设置设备映射，将 "cuda:0" 映射到目标 worker 的 "cuda:0"
        options.set_device_map(dst, {"cuda:0": "cuda:0"})

        # 初始化 RPC
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        # 创建一个包含 100 个零的 CUDA 张量
        t = torch.zeros((100,), device="cuda:0")
        # 在指定的 worker 上异步调用 RPC 函数，并传递参数 t
        fut = rpc.rpc_async(dst, async_cuda_sleep_and_set_to_one, args=(t,))
        # 创建另一个 CUDA 流
        another_stream = torch.cuda.Stream("cuda:0")
        # 使用另一个 CUDA 流上下文管理器
        with torch.cuda.stream(another_stream):
            # 断言异步 RPC 调用返回的 Future 等待结束后的结果是否全为 1
            self.assertTrue(torch.eq(fut.wait(), 1).all().item())

        # 关闭 RPC
        rpc.shutdown()

    @skip_if_lt_x_gpu(1)
    # 装饰器：如果当前环境的 GPU 少于 1 个，则跳过该测试
    def test_async_execution_nested_with_cuda_future(self):
        # 计算目标 worker 的名称，环绕世界大小的下一个 rank
        dst = worker_name((self.rank + 1) % self.world_size)
        # 计算嵌套目标 worker 的名称，环绕世界大小的下两个 rank
        nested_dst = worker_name((self.rank + 2) % self.world_size)
        # 获取 RPC 后端选项
        options = self.rpc_backend_options
        # 设置设备映射，将 "cuda:0" 映射到目标 worker 的 "cuda:0"
        options.set_device_map(dst, {"cuda:0": "cuda:0"})

        # 初始化 RPC
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        # 创建三个包含 100 个元素全为 1 的 CUDA 张量
        a = torch.ones((100,), device="cuda:0")
        b = torch.ones((100,), device="cuda:0")
        c = torch.ones((100,), device="cuda:0")
        # 在指定的 worker 上异步调用 RPC 函数，传递参数 nested_dst, a, b, c
        fut = rpc.rpc_async(dst, async_cuda_nested_add, args=(nested_dst, a, b, c))
        # 创建另一个 CUDA 流
        another_stream = torch.cuda.Stream("cuda:0")
        # 使用另一个 CUDA 流上下文管理器
        with torch.cuda.stream(another_stream):
            # 断言异步 RPC 调用返回的 Future 等待结束后的结果是否全为 3
            self.assertTrue(torch.eq(fut.wait(), 3).all().item())

        # 关闭 RPC
        rpc.shutdown()

    @skip_if_lt_x_gpu(1)
    # 装饰器：如果当前环境的 GPU 少于 1 个，则跳过该测试
    def test_cuda_future_modify_tensor_inplace(self):
        # 创建一个包含 100 个零的 CUDA 张量
        tensor = torch.zeros((100,), device="cuda:0")
        # 创建一个 Future 对象，并设置其结果为 tensor
        future = Future(devices=["cuda:0"])
        future.set_result(tensor)
        # 修改 tensor 的值，这种在 Future 完成后修改的行为在技术上是可能的，
        # 但目前被认为是未定义行为（实际上，Future 会忽略修改并与原始值同步）。
        # 将来我们可能会添加逻辑来检测并警告或在这种情况下抛出异常，但目前我们
        # 只是检查这不会导致崩溃。
        tensor.fill_(1)
        # 等待 Future 完成
        future.wait()
    def test_cuda_future_replace_tensor(self):
        tensor_list = [torch.zeros((100,), device="cuda:0")]
        future = Future(devices=["cuda:0"])
        future.set_result(tensor_list)
        # 当将 Future 完成后的结果替换为另一个值时，这是一种奇怪的操作。
        # 技术上是可能的，但目前被视为未定义行为（实际上 Future 会忽略修改并与原始值同步）。
        # 未来我们可以添加逻辑来检测并在这种情况下发出警告或抛出异常，但目前只是检查这不会导致崩溃。
        # 我们设置这样的环境，使得一旦我们用另一个值替换原列表中的原始张量，原始张量将被删除。
        # 这将使 Future 持有的任何缓存信息失效。
        tensor_list[0] = torch.ones((100,), device="cuda:0")
        future.wait()

    @skip_if_lt_x_gpu(1)
    def test_rref_with_unpickleable_attributes(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        options = self.rpc_backend_options
        options.set_device_map(dst, {"cuda:0": "cuda:0"})

        # 初始化 RPC
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        # 在远程 worker 上创建一个远程引用对象
        rref = rpc.remote(dst, TensorWrapper, args=(torch.zeros(42, device="cuda:0"),))
        rref.rpc_sync().increase(1)
        # 远程调用函数增加后返回结果
        ret = rref.rpc_sync().sum()
        self.assertEqual(ret, 42)

        # 关闭 RPC
        rpc.shutdown()

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_can_extract_cuda_sparse_tensor(self):
        self._test_cuda_future_extraction(
            wrapper=lambda t: t, unwrapper=lambda v: v, sparse_tensor=True
        )

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_can_extract_list_with_cuda_sparse_tensor(self):
        self._test_cuda_future_extraction(
            wrapper=lambda t: [t], unwrapper=operator.itemgetter(0), sparse_tensor=True
        )

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_can_extract_custom_class_with_cuda_sparse_tensor(self):
        self._test_cuda_future_extraction(
            wrapper=TensorWrapper, unwrapper=lambda v: v.tensor, sparse_tensor=True
        )
```