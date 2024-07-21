# `.\pytorch\torch\testing\_internal\distributed\rpc\jit\rpc_test.py`

```
# 忽略 mypy 的类型检查错误
# 导入必要的时间、IO、类型相关的库
import time
import io
from typing import Dict, List, Tuple, Any

# 导入 PyTorch 相关模块
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.autograd.profiler import record_function
from torch.distributed.rpc import RRef
from torch.distributed.rpc.internal import RPCExecMode, _build_rpc_profiling_key
from torch.futures import Future
from torch.testing._internal.common_utils import TemporaryFileName
from torch.testing._internal.dist_utils import (
    dist_init,
    get_function_event,
    initialize_pg,
    worker_name,
)
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)

# 导入旧版的性能分析模块
from torch.autograd.profiler_legacy import profile as _profile

# 检查给定的 rref 是否为指定类型的实例
def rref_isinstance(rref, cls_to_check):
    return isinstance(rref.local_value(), cls_to_check)

# 使当前线程休眠指定的时间（秒）
def sleep(t):
    time.sleep(t)

# 在指定的远程 worker 上调用 torch.add，返回一个远程引用（RRef）
def rpc_return_rref(dst):
    return rpc.remote(dst, torch.add, args=(torch.ones(2, 2), 1))

# JIT 脚本函数：获取 RRef 中的本地值（Tensor）
@torch.jit.script
def rref_local_value(rref: RRef[Tensor]) -> Tensor:
    return rref.local_value()

# JIT 脚本函数：创建并返回一个整数列表
@torch.jit.script
def list_create() -> List[int]:
    global_list = [1, 2, 3]
    return global_list

# JIT 脚本函数：修改 RRef 中的整数列表
@torch.jit.script
def rref_list_mutate(rref: RRef[List[int]]) -> None:
    rref.local_value().append(4)  # 向本地列表添加元素
    rref.to_here().append(5)  # 在本地同步添加元素
    rref.to_here(5.0).append(6)  # 在本地同步添加元素

# 返回给定的整数值
def return_value(value: int) -> int:
    return value

# RRefAPITest 类
class RRefAPITest:
    # 分布式初始化装饰器
    @dist_init
    # 测试函数：检查 RRef 是否为所有者
    def test_rref_is_owner(self):
        # 获取目标 worker 的名称
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        # 在目标 worker 上调用 rpc_return_rref 获取远程引用
        rref_var = rpc_return_rref(dst_worker_name)

        # JIT 脚本函数：检查 Tensor 类型的 RRef 是否为所有者
        @torch.jit.script
        def rref_tensor_is_owner(rref_var: RRef[Tensor]) -> bool:
            return rref_var.is_owner()

        # 调用 rref_tensor_is_owner 函数检查远程引用是否为所有者
        res = rref_tensor_is_owner(rref_var)
        self.assertEqual(res, False)  # 断言结果为 False

    # 分布式初始化装饰器
    @dist_init
    # 测试函数：获取 RRef 中的本地值
    def test_rref_local_value(self):
        # 如果当前进程的 rank 不为 0，则返回
        if self.rank != 0:
            return

        # 获取目标 worker 的名称
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        # 在目标 worker 上调用 rpc_return_rref 获取远程引用
        rref = rpc_return_rref(dst_worker_name)

        # 使用断言捕获 RuntimeError，测试非所有者的 RRef 是否能调用 local_value
        with self.assertRaisesRegex(
            RuntimeError, r"Can't call RRef.local_value\(\) on a non-owner RRef"
        ):
            rref_local_value(rref)

        # 同步调用远程的 rref_local_value 函数并获取返回结果
        ret = rpc.rpc_sync(dst_worker_name, rref_local_value, (rref,))
        self.assertEqual(ret, torch.add(torch.ones(2, 2), 1))  # 断言返回结果正确

    # 分布式初始化装饰器
    @dist_init
    # 测试函数：测试本地 RRef 的本地值
    def test_local_rref_local_value(self):
        # 如果当前进程的 rank 不为 0，则返回
        if self.rank != 0:
            return

        # 获取当前 worker 的名称
        dst_worker_name = worker_name(self.rank)
        # 在当前 worker 上调用 return_value 函数，返回一个远程引用（RRef）
        rref = rpc.remote(dst_worker_name, return_value, (5,), {})

        # 调用 rref_local_value 获取 RRef 中的本地值，并断言结果正确
        ret = rref_local_value(rref)
        self.assertEqual(ret, 5)

    # 创建并返回一个远程引用（RRef）
    def _create_rref(self):
        # 计算所有者的 rank
        owner_rank = (self.rank + 2) % self.world_size
        # 在目标 worker 上调用 torch.add 函数，返回一个远程引用（RRef）
        return rpc.remote(
            worker_name(owner_rank), torch.add, args=(torch.zeros(2, 2), 1)
        )

    # 分布式初始化装饰器
    @dist_init
    # 测试函数，用于确认用户远程引用是否已确认
    def test_user_rrefs_confirmed(self):
        # 计算目标排名，确保在环中循环
        dst_rank = (self.rank + 1) % self.world_size
        # 创建远程引用
        rref = self._create_rref()
        # 使用 RPC 同步调用远程函数，检查远程引用是否已确认
        ret = rpc.rpc_sync(
            worker_name(dst_rank), script_check_rref_confirmed, args=(rref,)
        )
        # 断言返回结果为 True
        self.assertEqual(ret, True)

    # 分布式初始化装饰器后的测试函数，用于确认用户远程引用是否已确认（远程方式）
    @dist_init
    def test_user_rrefs_confirmed_remote(self):
        # 计算目标排名，确保在环中循环
        dst_rank = (self.rank + 1) % self.world_size
        # 创建远程引用
        rref = self._create_rref()
        # 使用 RPC 远程调用远程函数，检查远程引用是否已确认
        ret_rref = rpc.remote(
            worker_name(dst_rank), script_check_rref_confirmed, args=(rref,)
        )
        # 断言远程调用返回结果已到本地为 True
        self.assertEqual(ret_rref.to_here(), True)

    # 分布式初始化装饰器后的测试函数，用于测试远程引用列表的变化
    @dist_init
    def test_rref_list_mutate(self):
        # 计算目标工作节点名称，确保在环中循环
        dst = worker_name((self.rank + 1) % self.world_size)
        # 远程创建列表引用
        list_rref = rpc.remote(dst, list_create)

        # 使用 RPC 同步调用远程函数，测试列表引用的变化
        rpc.rpc_sync(dst, rref_list_mutate, args=(list_rref,))
        # 断言列表引用已在远程节点上更新为 [1, 2, 3, 4, 5, 6]
        self.assertEqual(list_rref.to_here(), [1, 2, 3, 4, 5, 6])
@torch.jit.script
def no_arg():
    # 无参数的 Torch 脚本函数，返回整数 0
    return 0


@torch.jit.script
def one_arg(value):
    # 接受一个参数 value，并返回 value + 1 的结果
    return value + 1


@torch.jit.script
def script_add_ones(x):
    # 接受一个参数 x，将 x 和一个全为 1 的 Tensor 相加并返回结果
    return torch.add(x, torch.ones(1))


@torch.jit.script
def script_add_ones_with_record_function(x, block: str):
    # 接受一个参数 x 和一个字符串 block，使用 record_function 包装下面的操作
    with record_function(block):
        return torch.add(x, torch.ones(1))


@torch.jit.script
def record_function_on_caller_rpc_async(dst_worker_name: str, block: str) -> Tensor:
    # 使用 record_function 包装的异步 RPC 调用函数
    t: Tensor = torch.ones(1)
    with record_function(block) as rf:
        fut1 = rpc.rpc_async(dst_worker_name, script_add_ones, (t, ))
        # 额外的操作调用，避免下一个异步调用的去重
        zero = torch.zeros_like(t)
        fut2 = rpc.rpc_async(dst_worker_name, script_add_ones, (t, ))
        res = fut1.wait() + fut2.wait() + zero
    return res


@torch.jit.script
def script_fork_wait_udf(tensor):
    # 使用 fork 和 wait 操作对输入的 tensor 进行异步处理
    fut = torch.jit._fork(script_add_ones, tensor)
    x = torch.jit._wait(fut)
    return x


@torch.jit.script
def rref_to_here(rref_var: RRef[Tensor]) -> Tensor:
    # 将 RRef 类型的变量转换为本地 Tensor
    return rref_var.to_here()


@torch.jit.script
def return_rref(rref_var: RRef[Tensor]) -> RRef[Tensor]:
    # 直接返回传入的 RRef 变量
    return rref_var


@torch.jit.script
def script_raise_func(value):
    # 接受一个参数 value，如果其元素个数为 2，则抛出 ValueError 异常，否则返回 value + 1
    if value.numel() == 2:
        raise ValueError("Expected error")
    return value + 1


@torch.jit.script
def script_fork_wait_throw(invalue):
    # 使用 fork 和 wait 操作对输入的 invalue 进行异步处理，可能会抛出异常
    fut = torch.jit._fork(script_raise_func, invalue)
    value = torch.jit._wait(fut)
    return value


@torch.jit.script
def call_rpc_with_profiling(record: torch.classes.profiler._RecordFunction, dst_worker_name: str) -> Tensor:
    # 在 Torch 脚本中调用 RPC，并确保能够附加性能分析回调函数
    fut = rpc.rpc_async(dst_worker_name, one_arg, (torch.tensor(1),))
    torch.ops.profiler._call_end_callbacks_on_jit_fut(record, fut)
    ret = fut.wait()
    return ret


@torch.jit.script
def call_rpc_torchscript_with_record_function(dst_worker_name: str, block: str) -> Tensor:
    # 在 Torch 脚本中调用带有 record_function 的 RPC
    fut = rpc.rpc_async(dst_worker_name, script_add_ones_with_record_function, (torch.tensor(1), block))
    return fut.wait()


@torch.jit.script
def call_fork_with_profiling(record: torch.classes.profiler._RecordFunction) -> Tensor:
    # 在 Torch 脚本中调用 fork，并确保能够附加性能分析回调函数
    fut = torch.jit._fork(one_arg, torch.tensor(1))
    torch.ops.profiler._call_end_callbacks_on_jit_fut(record, fut)
    ret = fut.wait()
    return ret


class MyScriptModuleWithRRefs(torch.jit.ScriptModule):
    def __init__(self, dst_worker):
        super().__init__()
        self.rrefs = []
        for _ in range(4):
            self.rrefs.append(rpc_return_rref(dst_worker))

    @torch.jit.script_method
    # 带有 RRef 的 Torch 脚本模块，初始化时创建多个 RRef，并存储在 self.rrefs 中
    def forward(self):
        pass
    # 定义一个方法 `forward`，返回类型为 Tensor
    def forward(self) -> Tensor:
        # 创建一个 2x2 的张量，所有元素初始化为 1
        res_tensor = torch.ones(2, 2)
        # 遍历 self.rrefs 中的每一个远程引用 rref
        for rref in self.rrefs:
            # 将远程引用 rref 的数据从远程节点获取，并加到 res_tensor 上
            res_tensor += rref.to_here()

        # 返回最终计算得到的张量 res_tensor
        return res_tensor
# 使用装饰器 @torch.jit.ignore 来标记函数，使得其在 Torch 脚本编译时被忽略
@torch.jit.ignore
def rref_python_annotation(rref_var: RRef[Tensor]) -> RRef[Tensor]:
    return rref_var

# 使用装饰器 @torch.jit.script 来将函数编译为 Torch 脚本
@torch.jit.script
def rref_script_annotation(rref_var: RRef[Tensor]) -> Tensor:
    # 调用被 @torch.jit.ignore 标记的函数，并将其结果转移到本地
    return rref_python_annotation(rref_var).to_here()

# 类 RRefTypingTest 包含测试分布式环境中远程引用的方法
class RRefTypingTest:
    @dist_init
    def test_rref_as_arg_and_return(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        local_ret = one_arg(torch.ones(2, 2))

        # 在当前进程上创建远程引用 rref
        rref = rpc.remote(worker_name(self.rank), one_arg, args=(torch.ones(2, 2),))

        # 将 rref 传递给另一个用户的 RPC 调用
        ret = rpc.rpc_sync(worker_name(dst_rank), rref_to_here, args=(rref,))
        self.assertEqual(ret, local_ret)

        # 在 RPC 调用中返回 rref
        rref1 = rpc.rpc_sync(worker_name(dst_rank), return_rref, args=(rref,))
        self.assertEqual(rref1.to_here(), local_ret)

        # 将 rref 传递给另一个用户的远程调用
        rref2 = rpc.remote(worker_name(dst_rank), rref_to_here, args=(rref,))
        self.assertEqual(rref2.to_here(), local_ret)

        # 在远程调用中返回 rref
        rref3 = rpc.remote(worker_name(dst_rank), return_rref, args=(rref,))
        self.assertEqual(rref3.to_here().to_here(), local_ret)

    @dist_init
    def test_my_script_module_with_rrefs(self):
        n = self.rank + 1
        dst_rank = n % self.world_size

        # 使用包含远程引用的自定义 Torch 脚本模块
        module_with_rrefs = MyScriptModuleWithRRefs(worker_name(dst_rank))
        res = module_with_rrefs()
        self.assertEqual(res, torch.ones(2, 2) * 9)

    @dist_init
    def test_rref_python_annotation(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        # 使用 RPC 返回的远程引用
        rref_var = rpc_return_rref(worker_name(dst_rank))

        # 调用已经编译为 Torch 脚本的函数 rref_script_annotation
        res = rref_script_annotation(rref_var)
        self.assertEqual(res, torch.ones(2, 2) + 1)


class FutureTypingTest:
    @dist_init
    def test_future_passed_between_python_and_jit(self):
        dst_rank = (self.rank + 1) % self.world_size
        inputs = (torch.tensor([1, 1]), torch.tensor([2, 2]))

        # 发送异步 RPC 调用并获取返回的 Future 对象
        ret_fut = rpc.rpc_async(worker_name(dst_rank), two_args_two_kwargs, args=inputs)
        expected_res = torch.tensor([10, 10])

        # 在 Torch 脚本中等待 Future 对象的完成并获取结果
        @torch.jit.script
        def future_wait_in_script(fut: Future[Tensor]) -> Tensor:
            return fut.wait()

        self.assertEqual(future_wait_in_script(ret_fut), expected_res)

        # 在 Torch 脚本中返回异步 RPC 调用的 Future 对象
        @torch.jit.script
        def future_return_to_python(
            dst_rank: int, inputs: Tuple[Tensor, Tensor]
        ) -> Future[Tensor]:
            return rpc.rpc_async(
                f"worker{dst_rank}", two_args_two_kwargs, inputs
            )

        fut_res = future_return_to_python(dst_rank, inputs)
        self.assertEqual(fut_res.wait(), expected_res)
    # 定义一个测试函数，用于测试未来的Python注解
    def test_future_python_annotation(self):
        # 如果当前进程不是rank为0的进程，则直接返回，不进行后续操作
        if self.rank != 0:
            return

        # 计算下一个worker的名称，使用模运算确保在world_size内循环
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        
        # 创建一个2x2的全1张量作为输入数据
        input_0 = torch.ones(2, 2)
        
        # 设置一个整数作为第二个输入
        input_1 = 1
        
        # 期望的结果是将input_0和input_1相加得到的张量
        expected_res = torch.add(input_0, input_1)

        # 定义一个被torch.jit.ignore修饰的Python函数，返回一个Future[Tensor]类型的对象
        @torch.jit.ignore
        def python_return_future() -> Future[Tensor]:
            # 异步调用rpc，发送torch.add函数和对应的参数，得到一个Future对象
            fut = rpc.rpc_async(dst_worker_name, torch.add, (input_0, input_1), {})
            return fut

        # 定义一个被torch.jit.script修饰的脚本函数，返回一个Tensor类型的对象
        @torch.jit.script
        def script_use_future() -> Tensor:
            # 调用python_return_future函数，得到一个Future对象
            fut = python_return_future()
            # 等待Future对象的完成，并返回其结果
            return fut.wait()

        # 执行script_use_future函数，获取其返回结果
        res = script_use_future()
        
        # 断言返回的结果与期望的结果相等
        self.assertEqual(res, expected_res)
@torch.jit.script
class MyScriptClass:
    def __init__(self, a: int):
        self.a = a
    # 初始化方法，设置实例变量a为传入的整数参数

    def get_value(self) -> int:
        return self.a
    # 返回实例变量a的值


@torch.jit.interface
class MyModuleInterface(torch.nn.Module):
    def forward(self) -> Tensor:
        # pyre-ignore[7]: Pyre and torch.jit.interface don't mix well
        pass
    # 定义接口方法forward，返回一个Tensor，忽略类型检查


class MyScriptModule(torch.jit.ScriptModule):
    def __init__(self, rank):
        super().__init__()
        self.a = torch.ones(rank)
    # 初始化方法，创建一个包含rank个1的Tensor

    @torch.jit.script_method
    def forward(self) -> Tensor:
        return self.a
    # 覆盖父类的forward方法，返回实例变量a

    @torch.jit.script_method
    def custom_func(self) -> Tensor:
        return self.a
    # 返回实例变量a的自定义方法


def owner_create_rref_my_script_class(a):
    return rpc.RRef(MyScriptClass(a))
# 使用RPC创建一个远程引用，引用MyScriptClass的实例


def owner_create_rref_my_script_module(a):
    return rpc.RRef(MyScriptModule(a), type_hint=MyModuleInterface)
# 使用RPC创建一个远程引用，引用MyScriptModule的实例，并提供类型提示为MyModuleInterface


@torch.jit.script
def script_rref_get_value_my_script_class(rref: RRef[MyScriptClass]) -> int:
    return rref.to_here().get_value()
# 脚本化函数，接收一个RRef参数，返回远程引用中MyScriptClass实例的get_value()方法结果


@torch.jit.script
def script_rref_run_forward_my_script_module(rref: RRef[MyModuleInterface]) -> Tensor:
    return rref.to_here().forward()
# 脚本化函数，接收一个RRef参数，返回远程引用中MyModuleInterface实例的forward()方法结果


class LocalRRefTest:
    @dist_init
    def test_create_local_script_class_rref_in_py(self):
        if self.rank != 0:
            return

        # Create a local RRef<MyScriptClass>.
        rref_script_class = rpc.RRef(MyScriptClass(self.rank))
        ret = rref_script_class.to_here().get_value()
        self.assertEqual(ret, self.rank)
    # 在Python中创建一个本地RRef<MyScriptClass>，并验证get_value()方法返回值与rank相等

    @dist_init
    def test_create_local_script_module_rref_in_py(self):
        if self.rank != 0:
            return

        # Create a local RRef<MyModuleInterface>.
        rref_script_module = rpc.RRef(MyScriptModule(self.rank), MyModuleInterface)
        ret = rref_script_module.to_here().forward()
        self.assertEqual(ret, torch.ones(self.rank))
        # 在Python中创建一个本地RRef<MyModuleInterface>，并验证forward()方法返回值为rank个1的Tensor

        # Create a local RRef<MyModuleInterface> without type hint.
        with self.assertRaisesRegex(
            RuntimeError,
            (
                "The RRef being created contains a ScriptModule, "
                "must provide its ModuleInterface type hint."
            ),
        ):
            rref_script_module = rpc.RRef(MyScriptModule(self.rank))
    # 如果没有提供类型提示，则在Python中创建一个本地RRef<MyModuleInterface>，应该引发错误
    def test_return_local_script_class_rref_in_py_and_use_in_script(self):
        if self.rank != 0:
            return
        
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        # 在 Python 中远程创建一个本地的 RRef<MyScriptClass>。
        rref = rpc.rpc_sync(
            dst_worker_name, owner_create_rref_my_script_class, args=(self.rank,)
        )

        def use_rref_on_owner(rref: RRef[MyScriptClass]) -> int:
            args = (rref,)
            kwargs: Dict[str, Any] = {}
            fut = rpc.rpc_async(
                rref.owner(), script_rref_get_value_my_script_class, args, kwargs
            )
            # 等待异步 RPC 完成并获取结果
            ret = fut.wait()
            return ret

        # 在本地 Python RPC 和远程脚本运行中使用 RRef<MyScriptClass>。
        ret = use_rref_on_owner(rref)
        self.assertEqual(ret, self.rank)

        # 在本地脚本 RPC 和远程脚本运行中使用 RRef<MyScriptClass>。
        use_rref_on_owner_script = torch.jit.script(use_rref_on_owner)
        ret = use_rref_on_owner_script(rref)
        self.assertEqual(ret, self.rank)

    @dist_init
    def test_return_local_script_module_rref_in_py_and_use_in_script(self):
        if self.rank != 0:
            return
        
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        # 在 Python 中远程创建一个本地的 RRef<MyModuleInterface>。
        rref = rpc.rpc_sync(
            dst_worker_name, owner_create_rref_my_script_module, args=(self.rank,)
        )

        def use_rref_on_owner(rref: RRef[MyModuleInterface]) -> Tensor:
            args = (rref,)
            kwargs: Dict[str, Any] = {}
            fut = rpc.rpc_async(
                rref.owner_name(),
                script_rref_run_forward_my_script_module,
                args,
                kwargs,
            )
            # 等待异步 RPC 完成并获取结果
            ret = fut.wait()
            return ret

        # 在本地 Python RPC 和远程脚本运行中使用 RRef<MyScriptClass>。
        ret = use_rref_on_owner(rref)
        self.assertEqual(ret, torch.ones(self.rank))

        # 在本地脚本 RPC 和远程脚本运行中使用 RRef<MyScriptClass>。
        use_rref_on_owner_script = torch.jit.script(use_rref_on_owner)
        ret = use_rref_on_owner_script(rref)
        self.assertEqual(ret, torch.ones(self.rank))
# 定义一个简单的 Python 函数，返回固定值 0
def python_function():
    return 0


# 使用 Torch 的 JIT 装饰器将函数编译成脚本
@torch.jit.script
# 定义一个带有两个位置参数和两个关键字参数的函数
def two_args_two_kwargs(
    first_arg,
    second_arg,
    first_kwarg=torch.tensor([3, 3]),  # 第一个关键字参数，默认为 [3, 3]
    second_kwarg=torch.tensor([4, 4]),  # 第二个关键字参数，默认为 [4, 4]
):
    return first_arg + second_arg + first_kwarg + second_kwarg


# 使用 Torch 的 JIT 装饰器将函数编译成脚本
@torch.jit.script
# 定义一个带有各种类型参数和关键字参数的函数
def assorted_types_args_kwargs(
    tensor_arg: Tensor,  # 强制类型注解为 Tensor 类型
    str_arg: str,  # 字符串类型参数
    int_arg: int,  # 整数类型参数
    tensor_kwarg: Tensor = torch.tensor([2, 2]),  # Tensor 类型的默认关键字参数，默认为 [2, 2]
    str_kwarg: str = "str_kwarg",  # 字符串类型的默认关键字参数，默认为 "str_kwarg"
    int_kwarg: int = 2,  # 整数类型的默认关键字参数，默认为 2
):
    return tensor_arg + tensor_kwarg, str_arg + str_kwarg, int_arg + int_kwarg


# 使用 Torch 的 JIT 装饰器将函数编译成脚本
@torch.jit.script
# 定义一个抛出异常的函数
def raise_script():
    raise RuntimeError("Expected error")


# 使用 Torch 的 JIT 装饰器将函数编译成脚本
@torch.jit.script
# 定义一个远程调用 RPC 的异步函数
def script_rpc_async_call(
    dst_worker_name: str,  # 目标工作节点名称，字符串类型
    args: Tuple[Tensor, Tensor],  # 位置参数为包含两个 Tensor 的元组
    kwargs: Dict[str, Tensor]  # 关键字参数为包含 Tensor 的字典
):
    fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs)  # 异步 RPC 调用
    ret = fut.wait()  # 等待异步调用完成并获取返回值
    return ret


# 使用 Torch 的 JIT 装饰器将函数编译成脚本
@torch.jit.script
# 定义一个远程调用 RPC 的同步函数
def script_rpc_sync_call(
    dst_worker_name: str,  # 目标工作节点名称，字符串类型
    args: Tuple[Tensor, Tensor],  # 位置参数为包含两个 Tensor 的元组
    kwargs: Dict[str, Tensor]  # 关键字参数为包含 Tensor 的字典
):
    res = rpc.rpc_sync(dst_worker_name, two_args_two_kwargs, args, kwargs)  # 同步 RPC 调用
    return res


# 使用 Torch 的 JIT 装饰器将函数编译成脚本
@torch.jit.script
# 定义一个远程调用 RPC 的远程函数
def script_rpc_remote_call(
    dst_worker_name: str,  # 目标工作节点名称，字符串类型
    args: Tuple[Tensor, Tensor],  # 位置参数为包含两个 Tensor 的元组
    kwargs: Dict[str, Tensor]  # 关键字参数为包含 Tensor 的字典
):
    rref_res = rpc.remote(dst_worker_name, two_args_two_kwargs, args, kwargs)  # 远程 RPC 调用
    return rref_res.to_here()  # 获取远程调用结果


# 定义一个类 JitRpcOpTest
class JitRpcOpTest:
    # 使用分布式初始化装饰器
    @dist_init
    # 测试所有关键字参数是否都被默认填充的函数
    def test_all_kwargs_are_populated_by_defaults(self):
        if self.rank != 0:  # 如果当前进程不是排名为 0 的进程，则直接返回
            return

        dst_worker_name = worker_name((self.rank + 1) % self.world_size)  # 计算目标工作节点名称

        args = (torch.tensor([1, 1]), torch.tensor([2, 2]))  # 设置位置参数
        kwargs = {}  # 设置空的关键字参数字典

        for script_op in [script_rpc_async_call, script_rpc_sync_call, script_rpc_remote_call]:
            ret = script_op(
                dst_worker_name, args, kwargs  # 调用各种 RPC 脚本函数
            )
            self.assertEqual(ret, torch.tensor([10, 10]))  # 断言返回值与预期的 Tensor 相等

    # 使用分布式初始化装饰器
    @dist_init
    # 测试部分关键字参数是否都被默认填充的函数
    def test_some_kwargs_are_populated_by_defaults(self):
        if self.rank != 0:  # 如果当前进程不是排名为 0 的进程，则直接返回
            return

        dst_worker_name = worker_name((self.rank + 1) % self.world_size)  # 计算目标工作节点名称

        args = (torch.tensor([1, 1]), torch.tensor([2, 2]))  # 设置位置参数
        kwargs = {"first_kwarg": torch.tensor([2, 2])}  # 设置部分关键字参数

        for script_op in [script_rpc_async_call, script_rpc_sync_call, script_rpc_remote_call]:
            ret = script_op(
                dst_worker_name, args, kwargs  # 调用各种 RPC 脚本函数
            )
            self.assertEqual(ret, torch.tensor([9, 9]))  # 断言返回值与预期的 Tensor 相等
    # 定义测试函数，验证当排名不为零时退出函数
    def test_no_kwargs_are_populated_by_defaults(self):
        if self.rank != 0:
            return

        # 获取目标工作节点名称
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        # 设置函数调用的参数 args 和 kwargs
        args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
        kwargs = {
            "first_kwarg": torch.tensor([2, 2]),
            "second_kwarg": torch.tensor([3, 3]),
        }
        # 遍历不同的远程调用函数列表
        for script_op in [script_rpc_async_call, script_rpc_sync_call, script_rpc_remote_call]:
            # 执行远程调用函数，并接收返回值 ret
            ret = script_op(
                dst_worker_name, args, kwargs
            )
            # 断言返回值为 torch.tensor([8, 8])
            self.assertEqual(ret, torch.tensor([8, 8]))

    # 使用 dist_init 装饰器定义的测试函数，验证不同类型的参数在 RPC 调用中能正常处理
    @dist_init
    def test_args_and_kwargs_contain_different_types(self):
        if self.rank != 0:
            return

        # 获取目标工作节点名称
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        # 使用 Torch JIT 定义异步 RPC 调用的函数 script_rpc_async_call_with_assorted_types
        @torch.jit.script
        def script_rpc_async_call_with_assorted_types(
            dst_worker_name: str,
        ):
            # 设置包含不同类型数据的参数 args 和 kwargs
            args = (torch.tensor([1, 1]), "str_arg", 1)
            # 在定义字典 kwargs 时，必须注明值的类型为 `Any`，因为 JIT 类型推断不支持多种类型
            kwargs: Dict[str, Any] = {
                "tensor_kwarg": torch.tensor([3, 3]),
                "str_kwarg": "_str_kwarg",
                "int_kwarg": 3,
            }
            # 发起异步 RPC 调用，等待返回值 fut
            fut = rpc.rpc_async(
                dst_worker_name, assorted_types_args_kwargs, args, kwargs
            )
            # 等待异步调用完成，获取返回值 ret
            ret = fut.wait()
            return ret

        # 执行异步 RPC 调用函数 script_rpc_async_call_with_assorted_types，并接收返回值 ret
        ret = script_rpc_async_call_with_assorted_types(
            dst_worker_name
        )
        # 断言返回值为 (torch.tensor([4, 4]), "str_arg_str_kwarg", 4)
        self.assertEqual(ret, (torch.tensor([4, 4]), "str_arg_str_kwarg", 4))

    # 使用 dist_init 装饰器定义的测试函数，验证当不传递 kwargs 时的 RPC 调用行为
    @dist_init
    def test_kwargs_not_passed(self):
        if self.rank != 0:
            return

        # 获取目标工作节点名称
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        # 使用 Torch JIT 定义异步 RPC 调用的函数 script_rpc_async_call_without_kwargs_passed
        @torch.jit.script
        def script_rpc_async_call_without_kwargs_passed(
            dst_worker_name: str,
        ):
            # 设置不包含 kwargs 的参数 args
            args = ()
            # 发起异步 RPC 调用，等待返回值 fut
            fut = rpc.rpc_async(dst_worker_name, no_arg, args)
            # 等待异步调用完成，获取返回值 ret
            ret = fut.wait()
            return ret

        # 执行异步 RPC 调用函数 script_rpc_async_call_without_kwargs_passed，并接收返回值 ret
        ret = script_rpc_async_call_without_kwargs_passed(
            dst_worker_name
        )
        # 断言返回值为 0
        self.assertEqual(ret, 0)

    # 使用 dist_init 装饰器定义的测试函数，验证当不传递 args 和 kwargs 时的 RPC 调用行为
    @dist_init
    def test_args_kwargs_are_neither_passed(self):
        if self.rank != 0:
            return

        # 获取目标工作节点名称
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        # 使用 Torch JIT 定义异步 RPC 调用的函数 script_rpc_async_call_without_args_kwargs_passed
        @torch.jit.script
        def script_rpc_async_call_without_args_kwargs_passed(
            dst_worker_name: str,
        ):
            # 发起异步 RPC 调用，等待返回值 fut
            fut = rpc.rpc_async(dst_worker_name, no_arg)
            # 等待异步调用完成，获取返回值 ret
            ret = fut.wait()
            return ret

        # 执行异步 RPC 调用函数 script_rpc_async_call_without_args_kwargs_passed，并接收返回值 ret
        ret = script_rpc_async_call_without_args_kwargs_passed(
            dst_worker_name
        )
        # 断言返回值为 0
        self.assertEqual(ret, 0)
    def test_less_than_needed_args_are_specified(self):
        # 如果不是主进程，则直接返回，不执行测试
        if self.rank != 0:
            return

        # 计算下一个工作节点的名称
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        # 在 Torch 脚本中，参数匹配发生在脚本化期间
        # 断言捕获运行时错误，并验证错误信息是否包含特定内容
        with self.assertRaisesRegex(RuntimeError, "Argument second_arg not provided"):

            @torch.jit.script
            def script_rpc_async_call_with_less_args(
                dst_worker_name: str,  # noqa: E999
            ):
                # 定义参数和关键字参数
                args = (torch.tensor([1, 1]),)
                kwargs = {}
                # 异步 RPC 调用
                fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs)
                # 等待 RPC 调用完成，并返回结果
                ret = fut.wait()
                return ret

    @dist_init
    def test_more_than_needed_args_are_specified(self):
        # 如果不是主进程，则直接返回，不执行测试
        if self.rank != 0:
            return

        # 计算下一个工作节点的名称
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        # 在 Torch 脚本中，参数匹配发生在脚本化期间
        # 断言捕获运行时错误，并验证错误信息是否包含特定内容
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected at most 4 arguments but found 5 positional arguments",
        ):

            @torch.jit.script
            def script_rpc_async_call_with_more_args(
                dst_worker_name: str,
            ):
                # 定义多余的参数和关键字参数
                args = (
                    torch.tensor([1, 1]),
                    torch.tensor([2, 2]),
                    torch.tensor([3, 3]),
                    torch.tensor([4, 4]),
                    torch.tensor([5, 5]),
                )
                kwargs = {}
                # 异步 RPC 调用
                fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs)
                # 等待 RPC 调用完成，并返回结果
                ret = fut.wait()
                return ret

    @dist_init
    def test_unexepected_kwarg_is_specified(self):
        # 如果不是主进程，则直接返回，不执行测试
        if self.rank != 0:
            return

        # 计算下一个工作节点的名称
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        # 在 Torch 脚本中，关键字参数匹配发生在执行期间
        @torch.jit.script
        def script_rpc_async_call_with_unexpected_kwarg(
            dst_worker_name: str,  # noqa: E999
        ):
            # 定义参数和包含未预期关键字的关键字参数
            args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
            kwargs = {"third_kwarg": torch.tensor([1, 1])}
            # 异步 RPC 调用
            fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs)
            # 等待 RPC 调用完成，并返回结果
            ret = fut.wait()
            return ret

        # 断言捕获运行时错误，并验证错误信息是否包含特定内容
        with self.assertRaisesRegex(
            RuntimeError, "Unknown keyword argument 'third_kwarg'"
        ):
            # 调用定义的 Torch 脚本函数，并验证返回结果
            ret = script_rpc_async_call_with_unexpected_kwarg(dst_worker_name)
            self.assertEqual(ret, 0)
    def test_call_python_function_remotely_from_script_not_supported(self):
        # 如果当前进程的排名不是0，则直接返回，不执行后续代码
        if self.rank != 0:
            return

        # 计算目标工作节点的名称
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        # 使用TorchScript装饰器定义一个异步调用远程Python函数的函数
        @torch.jit.script
        def rpc_async_call_remote_py_function_in_torchscript(dst_worker_name: str):
            # 初始化参数和关键字参数
            args = ()
            kwargs = {}
            # 异步调用RPC远程函数
            fut = rpc.rpc_async(dst_worker_name, python_function, args, kwargs)
            # 等待异步调用的完成，并获取返回值
            ret = fut.wait()
            return ret

        # 使用断言检测是否捕获到预期的运行时错误
        with self.assertRaisesRegex(
            RuntimeError, "attempted to get undefined function"
        ):
            # 调用定义的TorchScript函数，预期抛出异常
            ret = rpc_async_call_remote_py_function_in_torchscript(dst_worker_name)
            # 断言返回值为0
            self.assertEqual(ret, 0)

    @dist_init
    def test_call_script_function_that_raises_remotely_from_script(self):
        # 如果当前进程的排名不是0，则直接返回，不执行后续代码
        if self.rank != 0:
            return

        # 计算目标工作节点的名称
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        # 使用TorchScript装饰器定义一个在TorchScript中抛出异常的异步调用函数
        @torch.jit.script
        def rpc_async_call_remote_raising_torchscript_in_torchscript(
            dst_worker_name: str,
        ):
            # 初始化参数和关键字参数
            args = ()
            kwargs = {}
            # 异步调用RPC远程函数，该函数在TorchScript中抛出异常
            fut = rpc.rpc_async(dst_worker_name, raise_script, args, kwargs)
            # 等待异步调用的完成，并获取返回值
            ret = fut.wait()
            return ret

        # 使用断言检测是否捕获到预期的运行时错误
        with self.assertRaisesRegex(RuntimeError, "Expected error"):
            # 调用定义的TorchScript函数，预期抛出异常
            ret = rpc_async_call_remote_raising_torchscript_in_torchscript(
                dst_worker_name
            )
            # 断言返回值为0
            self.assertEqual(ret, 0)

    @dist_init
    def test_call_script_function_that_not_exists_remotely_from_script(self):
        # 如果当前进程的排名不是0，则直接返回，不执行后续代码
        if self.rank != 0:
            return

        # 计算目标工作节点的名称
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        # 使用TorchScript装饰器定义一个不存在的TorchScript函数
        @torch.jit.script
        def nonexisting_script():
            return 0

        # 使用TorchScript装饰器定义一个异步调用不存在的TorchScript函数的函数
        @torch.jit.script
        def rpc_async_call_remote_nonexisting_torchscript_in_torchscript(
            dst_worker_name: str,
        ):
            # 初始化参数和关键字参数
            args = ()
            kwargs = {}
            # 异步调用RPC远程函数，该函数引用了一个不存在的TorchScript函数
            fut = rpc.rpc_async(dst_worker_name, nonexisting_script, args, kwargs)
            # 等待异步调用的完成，并获取返回值
            ret = fut.wait()
            return ret

        # 使用断言检测是否捕获到预期的运行时错误
        with self.assertRaisesRegex(
            RuntimeError, "attempted to get undefined function nonexisting_script"
        ):
            # 调用定义的TorchScript函数，预期抛出异常
            ret = rpc_async_call_remote_nonexisting_torchscript_in_torchscript(
                dst_worker_name
            )
            # 断言返回值为0
            self.assertEqual(ret, 0)
# 定义一个忽略 Torch JIT 编译的函数装饰器
@torch.jit.ignore
# 初始化一个自定义的 Torch 脚本模块接口，根据给定的 rank 返回一个 MyScriptModule 对象
def my_script_module_init(rank: int) -> MyModuleInterface:
    return MyScriptModule(rank)


# 使用 Torch 脚本装饰器定义一个构造 MyScriptModule 的函数，返回一个 MyModuleInterface 对象
@torch.jit.script
def construct_my_script_module(rank: int) -> MyModuleInterface:
    return my_script_module_init(rank)


# 使用 Torch 脚本装饰器定义一个运行引用脚本模块的函数
@torch.jit.script
def run_ref_script_module(
    ref_script_module: RRef[MyModuleInterface], t: Tensor
) -> Tensor:
    # 将远程引用的脚本模块转移至当前位置
    module = ref_script_module.to_here()
    # 调用模块的 forward 方法并与给定的张量 t 相加，返回结果张量
    return module.forward() + t


# 使用 Torch 脚本装饰器定义一个检查 RRef 是否被所有者确认的函数
@torch.jit.script
def script_check_rref_confirmed(rref: RRef[Tensor]) -> bool:
    return rref.confirmed_by_owner()


# 使用 Torch 脚本装饰器定义一个保存 RRef 的函数
@torch.jit.script
def save_rref(rref_var: RRef[Tensor], fname: str) -> None:
    # 将 RRef 对象保存到指定的文件中
    torch.save(rref_var, fname)


# 使用 Torch 脚本装饰器定义一个张量相加的函数
@torch.jit.script
def script_add(x: Tensor, y: Tensor) -> Tensor:
    return x + y


# 使用 Torch 脚本和 RPC 函数装饰器定义一个异步执行的张量相加函数
@rpc.functions.async_execution
@torch.jit.script
def async_add(to: str, x: Tensor, y: Tensor) -> Future[Tensor]:
    # 异步调用指定 worker 上的 script_add 函数，返回一个 Future 对象
    return rpc.rpc_async(to, script_add, (x, y))


# 使用 Torch 脚本和 RPC 函数装饰器定义一个错误类型的异步函数
@rpc.functions.async_execution
@torch.jit.script
def async_wrong_type() -> Tensor:
    # 返回一个创建错误类型的张量的 Torch 脚本对象
    return torch.zeros(2)


# 定义一个函数，从序列化的 Torch 脚本模块加载带有 Pickled RRef 的内容
def load_script_module_with_pickled_rref(pickled_script_module):
    # 使用字节流初始化一个内存中的文件对象
    f = io.BytesIO(pickled_script_module)
    # 加载 Torch 脚本模块并返回实例化后的对象
    m = torch.jit.load(f)
    return m()


# 定义一个包含多个测试的类，继承自多个测试和 RpcAgentTestFixture
class JitRpcTest(
    RRefAPITest,
    RRefTypingTest,
    LocalRRefTest,
    JitRpcOpTest,
    FutureTypingTest,
    RpcAgentTestFixture,
):
    # 使用分布式初始化装饰器定义一个 Torch 脚本函数测试
    @dist_init
    def test_torchscript_function(self):
        # 计算目标 worker 的名称
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        # 调用本地的 one_arg 函数，并将结果存储在 local_ret 中
        local_ret = one_arg(torch.ones(2, 2))
        # 在目标 worker 上同步调用 one_arg 函数，并将结果存储在 ret 中
        ret = rpc.rpc_sync(dst_worker_name, one_arg, args=(torch.ones(2, 2),))
        # 断言本地调用和远程调用的结果相等
        self.assertEqual(ret, local_ret)
        # 在目标 worker 上异步调用 one_arg 函数，返回一个 RRef 对象
        rref = rpc.remote(dst_worker_name, one_arg, args=(torch.ones(2, 2),))
        # 断言远程调用返回的结果与本地调用结果相等
        self.assertEqual(rref.to_here(), local_ret)
        # 创建一个指向自身的 RRef 对象
        local_rref = rpc.remote(
            worker_name(self.rank), one_arg, args=(torch.ones(2, 2),)
        )
        # 断言本地 RRef 对象返回的结果与本地调用结果相等
        self.assertEqual(local_rref.to_here(), local_ret)

    # 使用分布式初始化装饰器定义一个 Torch 脚本函数异常测试
    @dist_init
    def test_torchscript_function_exception(self):
        # 计算目标 worker 的名称
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        # 使用断言捕获预期的 RuntimeError 异常，验证同步调用中的参数类型错误
        with self.assertRaisesRegex(RuntimeError, r"one_arg\(\) expected at most"):
            ret = rpc.rpc_sync(dst_worker_name, one_arg, args=(10, 20))

        # 使用断言捕获预期的 RuntimeError 异常，验证远程调用中的参数类型错误
        with self.assertRaisesRegex(RuntimeError, r"one_arg\(\) expected at most"):
            rref = rpc.remote(dst_worker_name, one_arg, args=(10, 20))
    def test_torchscript_functions_not_supported(self):
        # 计算目标工作节点的名称
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        # 在本地实例化 MyScriptModule 不是线程安全的操作，
        # 等待本地实例化完成，以避免与后续服务器线程的并行实例化冲突
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        # 等待所有进程完成初始化
        dist.barrier()

        # rpc_sync 仍然接受脚本类，并在相同的代码路径中运行它
        ret = rpc.rpc_sync(dst_worker_name, MyScriptClass, args=(self.rank,))

        # rpc_sync 不接受脚本模块方法
        # Python 3.5 和 Python 3.6 抛出不同的错误消息，唯一共同的词是 "pickle"
        with self.assertRaisesRegex(TypeError, "pickle"):
            ret = rpc.rpc_async(
                dst_worker_name, my_local_script_module.forward, args=()
            )

    @dist_init
    def test_remote_script_module(self):
        # TODO, 需要进一步调查
        # 关闭时存在 RRef 泄漏，怀疑是因为将引用作为参数传递到 pybind 边界，
        # 而在调用 shutdown() 时 Python 未能回收该引用
        import torch.distributed.rpc.api as api

        # 忽略 RRef 泄漏警告
        api._ignore_rref_leak = True

        # 创建本地计算结果
        local_ret = torch.ones(self.rank) + torch.ones(self.rank)

        n = self.rank + 1
        dst_rank = n % self.world_size
        # 在远程工作节点上创建一个远程对象引用
        remote_ref = rpc.remote(
            worker_name(dst_rank), construct_my_script_module, args=(self.rank,)
        )

        # 将 RRef 参数传递给所有者
        ret = rpc.rpc_sync(
            worker_name(dst_rank),
            run_ref_script_module,
            args=(remote_ref, torch.ones(self.rank)),
        )
        # 断言远程调用返回的结果与本地计算结果相等
        self.assertEqual(ret, local_ret)

        # 将 RRef 参数传递给自身或用户时会引发异常
        with self.assertRaisesRegex(
            RuntimeError,
            "is an RRef to a ScriptModule. It can't be sent through RPC from owner,",
        ):
            ret = rpc.rpc_sync(
                worker_name(self.rank),
                run_ref_script_module,
                args=(remote_ref, torch.ones(self.rank)),
            )

    @dist_init
    # 定义测试方法，用于在远程创建脚本模块
    def test_create_script_module_on_remote(self):
        # 根据当前进程排名和总进程数计算远程目标名称
        dst_name = worker_name((self.rank + 1) % self.world_size)
        
        # 使用rpc_sync在远程端构建脚本模块
        created_script_module = rpc.rpc_sync(
            dst_name, MyScriptModule, args=(self.rank,)
        )
        
        # 断言创建的模块是torch.jit.ScriptModule的实例
        self.assertTrue(isinstance(created_script_module, torch.jit.ScriptModule))
        
        # 调用模块的forward方法，预期输出一个由self.rank确定的全1张量
        rank_ones_tensor = created_script_module()
        self.assertEqual(torch.ones(self.rank), rank_ones_tensor)

        # 使用rpc.remote在远程端构建ScriptModule
        remote_script_module = rpc.remote(dst_name, MyScriptModule, args=(self.rank,))
        
        # 使用rpc_sync验证远程端的模块是ScriptModule的实例
        remote_end_is_script = rpc.rpc_sync(
            remote_script_module.owner(),
            rref_isinstance,
            args=(remote_script_module, torch.jit.ScriptModule),
        )
        self.assertTrue(remote_end_is_script)
        
        # 在远程端运行forward方法
        remote_forward_output = remote_script_module.rpc_sync().forward()
        self.assertEqual(remote_forward_output, torch.ones(self.rank))
        
        # 在远程端运行定义在ScriptModule中的自定义函数custom_func
        remote_func_output = remote_script_module.rpc_sync().custom_func()
        self.assertEqual(remote_func_output, torch.ones(self.rank))
        
        # 确保能将ScriptModule的RRef传输到本地，并在本地运行forward方法
        local_script_module = remote_script_module.to_here()
        self.assertTrue(isinstance(local_script_module, torch.jit.ScriptModule))
        rank_ones_tensor = local_script_module()
        self.assertEqual(rank_ones_tensor, torch.ones(self.rank))
        
        # 在本地运行ScriptModule中定义的自定义函数custom_func
        local_script_func_output = local_script_module.custom_func()
        self.assertEqual(local_script_func_output, torch.ones(self.rank))

    # 使用dist_init装饰器初始化分布式环境，加载带有pickled RRef的脚本模块
    @dist_init
    def test_load_script_module_with_pickled_rref(self):
        # 根据当前进程排名和总进程数计算远程目标名称
        dst_name = worker_name((self.rank + 1) % self.world_size)
        
        # 创建带有RRefs的MyScriptModuleWithRRefs实例
        m1 = MyScriptModuleWithRRefs(dst_name)
        m2 = MyScriptModuleWithRRefs(dst_name)

        # 创建一个字节流对象
        f = io.BytesIO()

        # 启用JIT RRef Pickle功能
        rpc._enable_jit_rref_pickle()
        
        # 将m1保存到字节流中
        torch.jit.save(m1, f)
        
        # 禁用JIT RRef Pickle功能
        rpc._disable_jit_rref_pickle()

        # 使用rpc_sync加载pickled RRef并返回输出
        out1 = rpc.rpc_sync(
            dst_name,
            load_script_module_with_pickled_rref,
            args=(f.getvalue(),)
        )
        
        # 调用m2，预期输出与out1相等
        out2 = m2()
        self.assertEqual(out1, out2)

    # 使用dist_init装饰器初始化分布式环境，测试不支持的RRef JIT Pickle功能
    @dist_init
    def test_rref_jit_pickle_not_supported(self):
        # 根据当前进程排名计算目标进程的排名
        n = self.rank + 1
        dst_rank = n % self.world_size
        
        # 调用rpc_return_rref获取远程引用变量
        rref_var = rpc_return_rref(worker_name(dst_rank))
        
        # 使用临时文件名进行保存RRef，并验证是否引发了期望的异常
        with TemporaryFileName() as fname:
            with self.assertRaisesRegex(
                RuntimeError, "RRef jit pickling is only allowed inside RPC calls"
            ):
                save_rref(rref_var, fname)
    @dist_init
    def test_remote_script_throw(self):
        # 创建一个远程 RPC 对象，调用位于下一个工作节点的函数 script_raise_func，
        # 并传入参数 torch.ones(2)
        rref = rpc.remote(
            worker_name((self.rank + 1) % self.world_size),
            script_raise_func,
            args=(torch.ones(2),),
        )
        # 断言抛出的异常信息包含 "Expected error"
        with self.assertRaisesRegex(Exception, ".*Expected error.*"):
            # 等待远程调用完成
            rref.to_here()

    @dist_init
    def test_remote_script_udf(self):
        # 创建一个远程 RPC 对象，调用位于下一个工作节点的函数 script_fork_wait_udf，
        # 并传入参数 torch.ones(2)
        rref = rpc.remote(
            worker_name((self.rank + 1) % self.world_size),
            script_fork_wait_udf,
            args=(torch.ones(2),),
        )
        # 断言远程调用返回的结果与 torch.ones(2) * 2 相等
        self.assertEqual(rref.to_here(), torch.ones(2) * 2)

    @dist_init
    def test_async_script_udf(self):
        # 创建一个异步 RPC 对象，调用位于下一个工作节点的函数 script_fork_wait_udf，
        # 并传入参数 torch.ones(2)
        future = rpc.rpc_async(
            worker_name((self.rank + 1) % self.world_size),
            script_fork_wait_udf,
            args=(torch.ones(2),),
        )
        # 等待异步调用完成，并断言返回的结果与 torch.ones(2) * 2 相等
        self.assertEqual(future.wait(), torch.ones(2) * 2)

    @dist_init
    def test_callback_simple(self):
        # 定义一个回调函数，将异步 RPC 调用的结果加上 1
        def callback(fut):
            return fut.wait() + 1

        # 创建一个异步 RPC 对象，调用位于下一个工作节点的函数 script_fork_wait_udf，
        # 并传入参数 torch.ones(2)，然后添加回调函数 callback
        future = rpc.rpc_async(
            worker_name((self.rank + 1) % self.world_size),
            script_fork_wait_udf,
            args=(torch.ones(2),),
        ).then(callback)
        # 等待异步调用完成，并断言返回的结果与 torch.ones(2) * 2 + 1 相等
        self.assertEqual(future.wait(), torch.ones(2) * 2 + 1)

    @dist_init
    def test_callback_chain(self):
        # 定义一个回调函数，将异步 RPC 调用的结果加上 1
        def callback(fut):
            return fut.wait() + 1

        n = self.rank + 1
        # 获取目标工作节点的名称
        dst = worker_name(n % self.world_size)

        # 创建一个异步 RPC 对象，调用位于目标工作节点的函数 one_arg，
        # 并传入参数 torch.ones(n, n)
        fut = rpc.rpc_async(
            worker_name(n % self.world_size), one_arg, args=(torch.ones(n, n),)
        )

        num_cbs = 20
        # 循环添加多个回调函数 callback 到异步 RPC 对象 fut 上
        for _ in range(num_cbs):
            fut = fut.then(callback)

        # 等待异步调用完成，并断言返回的结果与 torch.ones(n, n) + 1 + num_cbs 相等
        self.assertEqual(fut.wait(), torch.ones(n, n) + 1 + num_cbs)

    @dist_init
    def test_add_done_callback(self):
        callback_called = None

        # 定义一个回调函数，将异步 RPC 调用的结果乘以 2，并将结果存储到 callback_called 中
        def callback(fut):
            nonlocal callback_called
            callback_called = fut.wait() * 2

        # 创建一个异步 RPC 对象，调用位于下一个工作节点的函数 script_fork_wait_udf，
        # 并传入参数 torch.ones(2)
        future = rpc.rpc_async(
            worker_name((self.rank + 1) % self.world_size),
            script_fork_wait_udf,
            args=(torch.ones(2),),
        )

        # 将回调函数 callback 添加到异步 RPC 对象 future 上
        future.add_done_callback(callback)

        # 创建一个 'then' 回调，以确保在测试结束前等待第一个回调函数执行
        future_then = future.then(lambda _: True)

        # 等待异步调用完成，并断言返回的结果与 torch.ones(2) * 2 相等
        self.assertEqual(future.wait(), torch.ones(2) * 2)

        # 由于无法保证 add_done_callback 函数会在测试结束前执行，
        # 添加一个 'then' 回调来确保等待第一个回调函数执行完毕
        future_then.wait()

        # 断言 callback_called 的值与 torch.ones(2) * 4 相等
        self.assertEqual(callback_called, torch.ones(2) * 4)

    @dist_init
    def test_async_script_throw(self):
        # 创建一个异步 RPC 对象，调用位于下一个工作节点的函数 script_fork_wait_throw，
        # 并传入参数 torch.ones(2)
        future = rpc.rpc_async(
            worker_name((self.rank + 1) % self.world_size),
            script_fork_wait_throw,
            args=(torch.ones(2),),
        )
        # 断言异步调用在等待过程中抛出异常，并且异常信息包含 "Expected error"
        with self.assertRaisesRegex(Exception, ".*Expected error.*"):
            future.wait()

    @dist_init
    def test_callback_with_exception(self):
        # 定义一个回调函数，用于处理异步操作的结果
        def callback(fut):
            # 断言捕获特定异常类型，并验证异常消息包含"Expected error"
            with self.assertRaisesRegex(Exception, ".*Expected error.*"):
                fut.wait()
            # 抛出一个预期的运行时异常
            raise RuntimeError("Another expected error")

        # 发起一个远程过程调用的异步操作，并在其完成后调用回调函数
        future = rpc.rpc_async(
            worker_name((self.rank + 1) % self.world_size),
            script_fork_wait_throw,
            args=(torch.ones(2),),
        ).then(callback)

        # 断言捕获运行时异常，并验证异常消息包含"Another expected error"
        with self.assertRaisesRegex(RuntimeError, "Another expected error"):
            future.wait()

    @dist_init
    def test_call_rpc_with_profiling(self):
        # 确保能够在调用rpc_async的脚本函数中，对一个JIT future调用torch.ops.profiler._call_end_callbacks_on_jit_fut
        # 如果当前进程的rank为0
        if self.rank == 0:
            with _profile() as prof:
                # 构建RPC调用的性能分析键
                prof_key = _build_rpc_profiling_key(
                    RPCExecMode.ASYNC,
                    torch._jit_internal._qualified_name(one_arg),
                    "worker0",
                    "worker1",
                )
                # 使用torch.autograd.profiler.record_function记录性能事件
                with torch.autograd.profiler.record_function(prof_key) as rf:
                    # 调用带有性能分析的RPC函数
                    ret = call_rpc_with_profiling(rf.record, "worker1")
            # TODO: 由于难以估计远程端非UDF的执行时间，无法获取可靠的此性能事件的时间
            # 这可以通过https://github.com/pytorch/pytorch/issues/36272解决
            # 之后，此测试应修改以验证函数执行时间
            events = prof.function_events
            # 获取特定性能事件
            function_event = get_function_event(events, prof_key)
            # 断言验证函数名在获取的函数事件名称中
            self.assertTrue(torch._jit_internal._qualified_name(one_arg) in function_event.name)
    # 定义一个测试函数，用于测试在 TorchScript 函数内部调用 rpc_async 是否进行了性能分析。
    def test_rpc_async_jit_profiled(self):
        # 如果当前进程的 rank 为 0，则执行以下测试代码
        if self.rank == 0:
            # 计算目标进程的 rank，这里是环形结构
            dst_rank = (self.rank + 1) % self.world_size
            # 获取目标进程的 worker 名称
            dst_worker_name = worker_name(dst_rank)
            # 准备传递给远程调用的参数
            args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
            kwargs = {}
            # 使用 _profile 上下文管理器进行性能分析
            with _profile() as prof:
                # 调用 TorchScript 函数中的 rpc_async 调用
                script_rpc_async_call(
                    dst_worker_name, args, kwargs
                )

            # 确保 rpc_async 调用被正确地进行了性能分析
            function_events = prof.function_events
            # 获取 TorchScript 函数的完整名称
            qual_name = torch._jit_internal._qualified_name(two_args_two_kwargs)
            # 查找与当前 rank 相关且符合 TorchScript 函数名的 rpc_async 事件
            rpc_async_jit_event = [
                event
                for event in function_events
                if qual_name in event.name and event.node_id == self.rank
            ]
            # 确保只有一个符合条件的 rpc_async 事件
            self.assertEqual(len(rpc_async_jit_event), 1)
            rpc_async_jit_event = rpc_async_jit_event[0]
            # 构建 rpc_async 调用的性能分析键值名称
            profiled_name = _build_rpc_profiling_key(
                RPCExecMode.ASYNC_JIT,
                qual_name,
                worker_name(self.rank),
                dst_worker_name,
            )
            # 确保性能分析键值名称与 rpc_async 事件名称相匹配
            self.assertEqual(profiled_name, rpc_async_jit_event.name)
            # 获取所有远程事件
            remote_events = [event for event in function_events if event.is_remote]
            # 确保所有远程事件发生在目标进程的 rank 上
            remote_event_node_ids = {
                remote_event.node_id for remote_event in remote_events
            }
            self.assertEqual(remote_event_node_ids, {dst_rank})
            # script_rpc_async_call 调用了 add 运算符，因此我们应该在远程事件中看到它
            remote_add = next(
                remote_event
                for remote_event in remote_events
                if "aten::add" in remote_event.name
            )
            # 构建远程 add 运算符的性能分析名称
            remote_add_profiled_name = f"{profiled_name}#remote_op: aten::add"
            # 确保远程 add 事件名称与构建的性能分析名称匹配
            self.assertEqual(remote_add.name, remote_add_profiled_name)
    # 定义测试方法，用于异步 RPC 调用中的记录函数功能测试
    def test_record_function_on_caller_rpc_async(self):
        # 如果当前进程在全局的第一个进程上
        if self.rank == 0:
            # 计算目标进程的排名，循环世界大小后的下一个进程排名
            dst_rank = (self.rank + 1) % self.world_size
            # 获取目标进程的工作名称
            dst_worker_name = worker_name(dst_rank)
            # 定义块的作用域名称为 "foo"
            block_scope = "foo"
            # 使用 _profile() 上下文管理器开始性能分析
            with _profile() as prof:
                # 在 JIT 下的记录函数中运行 2 次 rpc_async 调用
                record_function_on_caller_rpc_async(dst_worker_name, block_scope)

            # 确保记录函数事件被正确地进行了性能分析
            function_events = prof.function_events
            # 查找与块作用域名称相符的记录函数事件
            record_function_scope_event = [
                event for event in function_events if event.name == block_scope
            ]
            # 断言只有一次块作用域的记录函数事件
            self.assertEqual(1, len(record_function_scope_event))
            # 获取记录函数作用域事件的具体信息
            record_function_scope_event = record_function_scope_event[0]
            # 确保 RPC 未来事件被正确地进行了性能分析
            expected_key = _build_rpc_profiling_key(
                RPCExecMode.ASYNC_JIT,
                torch._jit_internal._qualified_name(script_add_ones),
                worker_name(self.rank),
                dst_worker_name,
            )
            # 查找与预期键值相符的 JIT RPC 事件
            jit_rpc_events = [
                event for event in function_events if event.name == expected_key
            ]
            # 断言有两次 JIT RPC 事件
            self.assertEqual(2, len(jit_rpc_events))
            # 验证记录函数作用域时间大于每个单独的 RPC 异步调用时间
            # 这里并不要求它大于总和，因为两者可能并行执行
            for jit_rpc_event in jit_rpc_events:
                self.assertTrue(
                    record_function_scope_event.cpu_time_total
                    > jit_rpc_event.cpu_time_total
                )

    @dist_init
    # 测试通过使用 record_function(...) 在 RPC 中对 torchscript 函数进行性能分析。
    def test_rpc_torchscript_record_function(self):
        REMOTE_OP_STR = "#remote_op: "
        # 如果当前进程的排名是 0
        if self.rank == 0:
            # 计算目标进程的排名
            dst_rank = (self.rank + 1) % self.world_size
            # 获取目标进程的工作名称
            dst_worker_name = worker_name(dst_rank)
            # 定义代码块的范围为 "foo"
            block_scope = "foo"
            # 使用 _profile() 进行性能分析
            with _profile() as prof:
                # 调用包含 record_function 的 RPC torchscript 函数
                call_rpc_torchscript_with_record_function(dst_worker_name, block_scope)

            # 需要调用以下函数以填充 CPU 子进程。
            prof.key_averages()
            # 获取函数事件列表
            function_events = prof.function_events
            # 构建预期的远程操作的 profiling key
            expected_key = (
                _build_rpc_profiling_key(
                    RPCExecMode.ASYNC_JIT,
                    torch._jit_internal._qualified_name(
                        script_add_ones_with_record_function
                    ),
                    worker_name(self.rank),
                    dst_worker_name,
                )
                + REMOTE_OP_STR
                + block_scope
            )
            # 在函数事件中找到符合预期名称的远程记录函数事件
            remote_record_function_event = next(
                evt for evt in function_events if evt.name == expected_key
            )
            # 断言代码块范围在远程记录函数事件名称中
            self.assertTrue(block_scope in remote_record_function_event.name)
            # 获取远程记录函数事件的 CPU 子进程
            remote_children = remote_record_function_event.cpu_children
            # 断言子进程中包含 "aten::add"
            self.assertTrue("aten::add" in child.name for child in remote_children)

    # 测试在 jit 结束回调函数中使用 torch.jit.fork 时的记录功能
    def test_record_function_jit_end_callbacks_with_fork(self):
        # 确保我们可以在 Python eager 模式中使用 torch.jit.fork 对 jit future 进行结束回调
        sleep_interval = 1
        with _profile() as prof:
            with torch.autograd.profiler.record_function("foo") as rf:
                # 使用 torch.jit.fork 调用 sleep 函数
                fut = torch.jit._fork(sleep, sleep_interval)
                # 在记录函数 rf 上调用结束回调函数 _call_end_callbacks_on_future
                rf._call_end_callbacks_on_future(fut)
            # 等待 future 完成
            fut.wait()

        # 获取函数事件列表
        function_events = prof.function_events
        # 获取名为 "foo" 的函数事件
        sleep_event = get_function_event(function_events, "foo")
        # 断言事件名称为 "foo"
        self.assertEqual(sleep_event.name, "foo")
        # 通过检查 profiling 事件的 CPU 时间来验证回调是否在正确的时间触发
        self.assertGreaterAlmostEqual(sleep_event.cpu_time * 1e-6, sleep_interval)

    # 测试在 jit 中使用性能分析调用 torch.ops.profiler._call_end_callbacks_on_jit_fut
    def test_call_fork_in_jit_with_profiling(self):
        # 确保我们可以在脚本函数中使用 torch.jit.fork 调用 torch.ops.profiler._call_end_callbacks_on_jit_fut
        with _profile() as prof:
            with torch.autograd.profiler.record_function("foo") as rf:
                # 调用使用性能分析的 call_fork_with_profiling 函数
                ret = call_fork_with_profiling(rf.record)

        # 获取函数事件列表
        events = prof.function_events
        # 获取名为 "foo" 的函数事件
        function_event = get_function_event(events, "foo")
        # 断言事件名称为 "foo"
        self.assertEqual(function_event.name, "foo")
    @dist_init
    def test_async_function_simple(self):
        # 计算目标 worker 的名称
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)

        # 同步远程过程调用（RPC），调用 async_add 函数并传递参数
        ret = rpc.rpc_sync(
            dst1, async_add, args=(dst2, torch.ones(2, 2), torch.ones(2, 2))
        )
        # 断言返回值为 torch.ones(2, 2) + 1
        self.assertEqual(ret, torch.ones(2, 2) + 1)

    @dist_init
    def test_async_function_wrong_return_type(self):
        # 断言捕获 RuntimeError 异常，异常信息指出返回类型错误
        with self.assertRaisesRegex(
            RuntimeError,
            "Async functions must return an IValue of Future type, but got Tensor",
        ):
            # 同步远程过程调用（RPC），调用 async_wrong_type 函数
            rpc.rpc_sync(
                worker_name((self.rank + 1) % self.world_size), async_wrong_type
            )

    @dist_init
    def test_async_function_wrong_decorator_order(self):
        # 捕获 RuntimeError 异常，此处不关心具体的异常信息，避免测试代码依赖于 RPC 错误处理代码
        with self.assertRaises(RuntimeError):
            # 使用 torch.jit.script 装饰器，但由于 rpc 未定义，会引发错误
            @torch.jit.script
            @rpc.functions.async_execution
            def async_wrong_decorator_order(
                to: str, x: Tensor, y: Tensor
            ) -> Future[Tensor]:
                # 使用 rpc.rpc_async 发起异步远程过程调用（RPC）
                return rpc.rpc_async(to, script_add, (x, y))

    @dist_init
    def test_async_function_remote(self):
        # 计算目标 worker 的名称
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)

        # 远程调用（RPC），返回一个远程引用（RRef）
        rref = rpc.remote(
            dst1, async_add, args=(dst2, torch.ones(2, 2), torch.ones(2, 2))
        )
        # 断言远程引用的结果为 torch.ones(2, 2) + 1
        self.assertEqual(rref.to_here(), torch.ones(2, 2) + 1)

    @dist_init
    def test_async_function_remote_multi(self):
        # 计算目标 worker 的名称
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)

        num = 20
        rrefs = []
        # 循环创建多个远程引用（RRef）
        for i in range(num):
            rrefs.append(
                rpc.remote(
                    dst1, async_add, args=(dst2, torch.ones(2, 2), torch.ones(2, 2) * i)
                )
            )

        # 遍历远程引用，断言每个远程引用的结果为 torch.ones(2, 2) + i
        for i in range(num):
            self.assertEqual(rrefs[i].to_here(), torch.ones(2, 2) + i)

    @dist_init
    def test_async_function_wrong_return_type_remote(self):
        # 发起远程调用（RPC），返回一个远程引用（RRef）
        rref = rpc.remote(
            worker_name((self.rank + 1) % self.world_size), async_wrong_type
        )

        # 断言捕获 RuntimeError 异常，异常信息指出返回类型错误
        with self.assertRaisesRegex(
            RuntimeError,
            "Async functions must return an IValue of Future type, but got Tensor",
        ):
            # 等待远程引用结果
            rref.to_here()
```