# `.\pytorch\torch\testing\_internal\distributed\nn\api\remote_module_test.py`

```py
# mypy: ignore-errors

# 导入必要的库和模块
import enum  # 导入枚举类型模块
from typing import Tuple  # 导入类型提示模块

import torch  # 导入PyTorch库
import torch.distributed.rpc as rpc  # 导入分布式RPC模块
import torch.testing._internal.dist_utils as dist_utils  # 导入内部测试工具模块
from torch import Tensor, nn  # 导入张量和神经网络模块
from torch._jit_internal import Future  # 导入内部jit模块
from torch.distributed.nn import RemoteModule  # 导入远程模块
from torch.distributed.nn.api.remote_module import _REMOTE_MODULE_PICKLED_ATTRIBUTES  # 导入远程模块的属性
from torch.distributed.nn.api.remote_module import _RemoteModule  # 导入远程模块
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入分布式测试工具
from torch.testing._internal.common_utils import TemporaryFileName  # 导入临时文件名工具
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (  # 导入RPC代理测试工具
    RpcAgentTestFixture,
)

_PARAM_VAL = torch.nn.Parameter(torch.ones(1))  # 创建一个参数张量


# RPC处理器，用于查询目标工作进程的设备
def remote_device(module_rref):
    for param in module_rref.local_value().parameters():
        return param.device


# RPC处理器，用于查询目标工作进程的__dict__
def remote_module_attributes(remote_module):
    return remote_module.__dict__


# RPC处理器，用于在目标工作进程上运行前向传播
def remote_forward(remote_module, args):
    return remote_module.forward(*args)

# RPC处理器，用于在目标工作进程上异步运行前向传播
def remote_forward_async(remote_module, args):
    # 由于无法将future序列化并通过RPC层发送，
    # 必须等待并表现得像``forward_sync``一样。
    return remote_module.forward_async(*args).wait()

# RPC处理器，用于获取目标工作进程上的训练模式
def get_remote_training_arg(module_rref):
    return module_rref.local_value().training


# 枚举类型，定义模块创建模式
class ModuleCreationMode(enum.Enum):
    MODULE_CTOR_WITH_INTERFACE = "module_ctor_with_interface"
    MODULE_CTOR = "module_ctor"


# 定义一个接口，继承自torch.jit.interface，用于自定义模块接口
@torch.jit.interface
class MyModuleInterface:
    def forward(
        self, tensor: Tensor, number: int, word: str = "default"
    ) -> Tuple[str, int, Tensor]:
        # pyre-ignore[7]: Pyre and torch.jit.interface don't mix well
        pass


# 定义另一个接口，继承自torch.jit.interface，用于远程自定义模块接口
@torch.jit.interface
class RemoteMyModuleInterface:
    def forward(
        self, tensor: Tensor, number: int, word: str = "default"
    ) -> Tuple[str, int, Tensor]:
        # pyre-ignore[7]: Pyre and torch.jit.interface don't mix well
        pass

    def forward_async(
        self, tensor: Tensor, number: int, word: str = "default"
    ) -> Future[Tuple[str, int, Tensor]]:
        pass


# 定义一个普通的PyTorch模块，实现了MyModuleInterface接口
class MyModule(nn.Module):
    def __init__(self, first_arg, first_kwarg=-1):
        super().__init__()
        self.param1 = _PARAM_VAL

    def forward(
        self, tensor: Tensor, number: int, word: str = "default"
    ) -> Tuple[str, int, Tensor]:
        return word, number, tensor


# 定义一个不合规范的模块类，未继承自nn.Module
class BadModule:
    def __init__(self, first_arg, first_kwarg=-1):
        pass


# 创建一个脚本化的PyTorch模块
def create_scripted_module(first_arg, first_kwarg=-1):
    module = MyModule(first_arg, first_kwarg=first_kwarg)
    scripted_module = torch.jit.script(module)
    return scripted_module
# Common utils for both CPU and CUDA test suites
class CommonRemoteModuleTest(RpcAgentTestFixture):
    @property
    def world_size(self):  # Override setting in RpcAgentTestFixture
        # 返回测试用例的分布式环境中的进程数量为 2
        return 2

    @staticmethod
    def _create_remote_module_iter(remote_device, modes=None):
        # 如果 modes 未指定，则使用所有 ModuleCreationMode 的值
        if modes is None:
            modes = ModuleCreationMode.__members__.values()

        args = (1,)
        kwargs = dict(first_kwarg=2)

        if ModuleCreationMode.MODULE_CTOR in modes:
            # 创建一个远程模块对象，使用给定的远程设备和参数
            remote_module = RemoteModule(remote_device, MyModule, args, kwargs)
            yield remote_module

        if ModuleCreationMode.MODULE_CTOR_WITH_INTERFACE in modes:
            # 创建一个脚本化远程模块对象，使用给定的远程设备和参数
            remote_module = _RemoteModule(
                remote_device,
                create_scripted_module,
                args,
                kwargs,
                _module_interface_cls=MyModuleInterface,
            )
            scripted_remote_module = torch.jit.script(remote_module)
            yield scripted_remote_module


class RemoteModuleTest(CommonRemoteModuleTest):
    @dist_utils.dist_init
    def test_bad_module(self):
        if self.rank != 0:
            return
        # 计算目标工作节点名称，使用分布式工具获取下一个工作节点的名称
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        remote_device = f"{dst_worker_name}/cpu"
        args = (1,)
        kwargs = dict(first_kwarg=2)

        with self.assertRaisesRegex(
            ValueError,
            r"Expect `module_cls\(\*args, \*\*kwargs\)` returns an instance of <class nn.Module>,",
        ):
            # 断言创建的远程模块为正确的模块类型，期望引发 ValueError 异常
            RemoteModule(remote_device, BadModule, args, kwargs).forward()

        with self.assertRaisesRegex(
            ValueError,
            r"Expect `module_cls\(\*args, \*\*kwargs\)` returns an instance of <class nn.Module>,",
        ):
            # 断言创建的远程模块为正确的模块类型，期望引发 ValueError 异常
            RemoteModule(remote_device, BadModule, args, kwargs).forward()


    @dist_utils.dist_init
    def test_forward_async(self):
        if self.rank != 0:
            return
        # 计算目标工作节点名称，使用分布式工具获取下一个工作节点的名称
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        args = (torch.ones(1), 2, "3")
        # 遍历远程模块迭代器，创建远程模块并进行异步前向传播测试
        for remote_module in self._create_remote_module_iter(dst_worker_name):
            ret_fut = remote_module.forward_async(*args)
            # 等待异步操作完成，并获取返回结果
            ret = ret_fut.wait()
            # 断言异步前向传播的返回结果与预期相符
            self.assertEqual(ret, tuple(reversed(args)))

    @dist_utils.dist_init
    @dist_utils.dist_init
    # 使用 dist_init 装饰器来标记此方法作为分布式初始化方法
    def test_forward_async_script(self):
        # 如果当前进程的 rank 不是 0，则直接返回，不执行后续代码
        if self.rank != 0:
            return
        # 计算下一个目标工作节点的名称
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        # 创建远程模块的迭代器，并获取下一个远程模块（支持接口构造模式）
        scripted_remote_module = next(
            self._create_remote_module_iter(
                dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR_WITH_INTERFACE]
            )
        )

        # 定义一个 Torch Script 函数 run_forward_async，接受一个 RemoteMyModuleInterface 类型的参数
        @torch.jit.script
        def run_forward_async(scripted_remote_module: RemoteMyModuleInterface):
            # 调用远程模块的异步 forward 方法，传入参数 torch.ones(1), 2, "3"
            ret_fut = scripted_remote_module.forward_async(torch.ones(1), 2, "3")
            # 等待异步操作完成并获取返回值
            ret = ret_fut.wait()
            return ret

        # 调用 run_forward_async 函数，并传入 scripted_remote_module 作为参数
        ret = run_forward_async(scripted_remote_module)

        # 断言返回值 ret 是否等于 ("3", 2, torch.ones(1))
        self.assertEqual(ret, ("3", 2, torch.ones(1)))
    @dist_utils.dist_init
    # 使用分布式初始化装饰器，确保该测试在分布式设置中运行
    def test_remote_parameters(self):
        # 如果当前进程的排名不是0，则退出，只在排名为0的进程上运行该测试
        if self.rank != 0:
            return
        # 计算目标工作节点的名称，使用排名加1取模世界大小来确定目标工作节点
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        # 只测试Python的nn.Module，因为脚本模块方法不支持remote_parameters。
        # 通过迭代创建远程模块，指定模块创建模式为MODULE_CTOR
        for remote_module in self._create_remote_module_iter(
            dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            # 获取远程模块的参数远程引用
            param_rrefs = remote_module.remote_parameters()
            # 断言远程参数引用列表的长度为1
            self.assertEqual(len(param_rrefs), 1)
            # 断言远程参数引用的值是否与_PARAM_VAL相等
            self.assertTrue(torch.equal(param_rrefs[0].to_here(), _PARAM_VAL))

    @dist_utils.dist_init
    # 使用分布式初始化装饰器，确保该测试在分布式设置中运行
    def test_get_module_rref(self):
        # 如果当前进程的排名不是0，则退出，只在排名为0的进程上运行该测试
        if self.rank != 0:
            return
        # 计算目标工作节点的名称，使用排名加1取模世界大小来确定目标工作节点
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        # 只测试Python的nn.Module，因为脚本模块方法不支持get_module_rref。
        # 通过迭代创建远程模块，指定模块创建模式为MODULE_CTOR
        for remote_module in self._create_remote_module_iter(
            dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            # 获取远程模块的模块远程引用
            rref = remote_module.get_module_rref()
            # 断言获取的模块远程引用与remote_module.module_rref相等
            self.assertEqual(rref, remote_module.module_rref)
            # 遍历远程引用，断言其中每个参数是否与_PARAM_VAL相等
            for param in rref.to_here().parameters():
                self.assertTrue(torch.equal(param, _PARAM_VAL))

    @dist_utils.dist_init
    # 使用分布式初始化装饰器，确保该测试在分布式设置中运行
    def test_train_eval(self):
        # 如果当前进程的排名不是0，则退出，只在排名为0的进程上运行该测试
        if self.rank != 0:
            return
        # 计算目标工作节点的名称，使用排名加1取模世界大小来确定目标工作节点
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        # 通过迭代创建远程模块，指定模块创建模式为MODULE_CTOR
        for remote_module in self._create_remote_module_iter(
            dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            # 设置远程模块为训练模式
            remote_module.train()
            # 使用RPC同步调用获取远程训练参数，断言返回值为True
            ret1 = rpc.rpc_sync(dst_worker_name, get_remote_training_arg, args=(remote_module.get_module_rref(),))
            self.assertEqual(ret1, True)

            # 设置远程模块为评估模式
            remote_module.eval()
            # 使用RPC同步调用获取远程训练参数，断言返回值为False
            ret2 = rpc.rpc_sync(dst_worker_name, get_remote_training_arg, args=(remote_module.get_module_rref(),))
            self.assertEqual(ret2, False)
    # 如果当前进程的 rank 不为 0，则直接返回，不执行后续代码
    def test_send_remote_module_with_a_new_attribute_not_pickled_over_the_wire(self):
        if self.rank != 0:
            return
        # 计算目标 worker 的名称，使用环形方式确定下一个 worker
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        # 对于每一个通过 _create_remote_module_iter 创建的 remote_module
        # 通过 RPC 将新添加的属性发送到远程模块时，
        # 新添加的字段不会被 pickle，因为它未在 _REMOTE_MODULE_PICKLED_ATTRIBUTES 中指定。
        # 需要注意的是，除构造函数外添加新属性的情况应该很少发生。
        # 如果在 RemoteModule 构造函数中添加新属性，
        # 则会进行健全性检查，强制开发人员将此属性添加到 _REMOTE_MODULE_PICKLED_ATTRIBUTES 或 _REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING 中。
        for remote_module in self._create_remote_module_iter(
            dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            # 定义新属性的名称
            new_attr_name = "new_attr"
            # 设置 remote_module 的新属性
            setattr(remote_module, new_attr_name, 1)

            # 使用 RPC 同步调用远程模块的属性获取函数，获取属性列表 attrs
            attrs = rpc.rpc_sync(
                dst_worker_name, remote_module_attributes, (remote_module,)
            )
            # 断言新添加的属性名不在 attrs 列表中
            self.assertNotIn(new_attr_name, attrs)

    # 使用分布式初始化装饰器 dist_init 运行的测试函数
    @dist_utils.dist_init
    def test_remote_module_py_pickle_not_supported(self):
        # 如果当前进程的 rank 不为 0，则直接返回，不执行后续代码
        if self.rank != 0:
            return
        # 计算目标 worker 的名称，使用环形方式确定下一个 worker
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        # 对于每一个通过 _create_remote_module_iter 创建的 remote_module
        # 使用临时文件名 fname 进行测试，断言无法使用 Python Pickler 对 RemoteModule 进行序列化
        # 抛出 RuntimeError 异常，提示 RemoteModule 只能在使用 RPC 时进行序列化
        for remote_module in self._create_remote_module_iter(
            dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            with TemporaryFileName() as fname:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Cannot pickle RemoteModule in python pickler. RemoteModule can only be pickled when using RPC",
                ):
                    torch.save(remote_module, fname)

    # 使用分布式初始化装饰器 dist_init 运行的测试函数
    def test_remote_module_py_pickle_not_supported_script(self):
        # 如果当前进程的 rank 不为 0，则直接返回，不执行后续代码
        if self.rank != 0:
            return
        # 计算目标 worker 的名称，使用环形方式确定下一个 worker
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        # 对于每一个通过 _create_remote_module_iter 创建的 remote_module
        # 使用临时文件名 fname 进行测试，断言无法使用 Torch Script 对 RemoteModule 进行序列化
        # 抛出 torch.jit.Error 异常，提示 RemoteModule 只能在使用 RPC 时进行序列化
        for remote_module in self._create_remote_module_iter(
            dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR_WITH_INTERFACE]
        ):
            with TemporaryFileName() as fname:
                with self.assertRaisesRegex(torch.jit.Error, "can only be pickled when using RPC"):
                    torch.save(remote_module, fname)
class ThreeWorkersRemoteModuleTest(CommonRemoteModuleTest):
    @property
    def world_size(self):  # Override setting in CommonRemoteModuleTest
        return 3

    @dist_utils.dist_init
    def test_send_remote_module_over_the_wire(self):
        if self.rank != 0:
            return

        # Determine the names of destination workers based on current rank
        dst_worker1_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        dst_worker2_name = dist_utils.worker_name((self.rank + 2) % self.world_size)

        # Unpickled attributes include inherent RemoteModule attributes and additional methods
        expected_unpickled_attrs = list(_REMOTE_MODULE_PICKLED_ATTRIBUTES)
        expected_unpickled_attrs.append("forward_async")
        expected_unpickled_attrs.append("forward")

        # Iterate over remote modules created on worker1
        for remote_module in self._create_remote_module_iter(
            dst_worker1_name, modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            # Query simple attributes from worker2 and assert their equality
            attrs = rpc.rpc_sync(dst_worker2_name, remote_module_attributes, (remote_module,))
            self.assertListEqual(list(attrs.keys()), expected_unpickled_attrs)
            self.assertEqual(attrs["on"], "worker1")
            self.assertEqual(attrs["device"], "cpu")
            self.assertFalse(attrs["is_device_map_set"])
            self.assertFalse(attrs["is_scriptable"])

            # Test invoking installed methods on worker1 from worker2 via RPC
            # Note: Normally, a remote module should execute methods on the worker where it resides.
            args = (torch.ones(1), 2, "3")
            ret1 = rpc.rpc_sync(dst_worker2_name, remote_forward, (remote_module, args))
            self.assertEqual(ret1, tuple(reversed(args)))
            ret2 = rpc.rpc_sync(dst_worker2_name, remote_forward_async, (remote_module, args))
            self.assertEqual(ret2, tuple(reversed(args)))

    @dist_utils.dist_init
    def test_send_remote_module_over_the_wire_script_not_supported(self):
        # 如果当前进程的rank不是0，则直接返回，不执行后续代码
        if self.rank != 0:
            return
        
        # 计算目标worker的名称，使用循环顺序找到下一个和再下一个worker的名称
        dst_worker1_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        dst_worker2_name = dist_utils.worker_name((self.rank + 2) % self.world_size)

        # 期望被反序列化的远程模块的属性列表，包括_RemoteModule_PICKLED_ATTRIBUTES中的属性和两个额外的安装方法
        expected_unpickled_attrs = list(_REMOTE_MODULE_PICKLED_ATTRIBUTES)
        expected_unpickled_attrs.append("forward_async")
        expected_unpickled_attrs.append("forward")

        # 使用assertRaisesRegex检查运行时错误，确保不支持通过RPC传输脚本类型的RemoteModule
        with self.assertRaisesRegex(
            RuntimeError, "Passing a script RemoteModule over RPC is not supported."
        ):
            # 在worker1上创建一个远程模块，然后通过RPC层将其传递给worker2
            for remote_module in self._create_remote_module_iter(
                dst_worker1_name, modes=[ModuleCreationMode.MODULE_CTOR_WITH_INTERFACE]
            ):
                # 测试从worker2查询一些简单的属性
                attrs = rpc.rpc_sync(
                    dst_worker2_name, remote_module_attributes, (remote_module,)
                )

    @dist_utils.dist_init
    def test_create_remote_module_from_module_rref(self):
        # 如果当前进程的rank不是0，则直接返回，不执行后续代码
        if self.rank != 0:
            return
        
        # 计算目标worker的名称，使用循环顺序找到下一个和再下一个worker的名称
        dst_worker1_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        dst_worker2_name = dist_utils.worker_name((self.rank + 2) % self.world_size)

        # 在worker1上创建一个远程模块，并通过RPC层将其`module_rref`传递给worker2
        for remote_module in self._create_remote_module_iter(
            dst_worker1_name, modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            remote_module2 = rpc.rpc_sync(
                dst_worker2_name,
                RemoteModule.init_from_module_rref,
                (dst_worker2_name, remote_module.get_module_rref()),
            )

            # 设置参数args，调用远程函数remote_forward并通过RPC同步执行，分别在worker1和worker2上
            args = (torch.ones(1), 2, "3")
            ret1 = rpc.rpc_sync(
                dst_worker1_name, remote_forward, (remote_module, args)
            )
            ret2 = rpc.rpc_sync(
                dst_worker2_name, remote_forward, (remote_module2, args)
            )
            # 断言确保在worker1和worker2上得到的结果ret1和ret2相等
            self.assertEqual(ret1, ret2)
class CudaRemoteModuleTest(CommonRemoteModuleTest):
    # CudaRemoteModuleTest 类，继承自 CommonRemoteModuleTest

    @skip_if_lt_x_gpu(1)
    @dist_utils.dist_init
    # 使用装饰器 skip_if_lt_x_gpu(1) 和 dist_utils.dist_init 对 test_valid_device 方法进行装饰，用于跳过没有足够 GPU 的情况和初始化分布式环境

    def test_valid_device(self):
        # 定义测试方法 test_valid_device，用于验证设备合法性
        if self.rank != 0:
            # 如果当前进程不是 rank 为 0，则直接返回
            return
        
        dst_rank = (self.rank + 1) % self.world_size
        # 计算目标 rank，这里采用 (当前 rank + 1) % 总进程数 的方式确定目标 rank
        dst_worker_name = dist_utils.worker_name(dst_rank)
        # 获取目标 worker 的名称，通过 dist_utils.worker_name 方法获取

        for remote_module in self._create_remote_module_iter(
            f"{dst_worker_name}/cuda:0", modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            # 遍历远程模块迭代器，使用给定的目标 worker 名称和设备名 "cuda:0" 创建远程模块
            device = rpc.rpc_sync(
                dst_worker_name, remote_device, (remote_module.module_rref,)
            )
            # 使用 RPC 进行同步调用，获取远程设备信息
            self.assertEqual(device.type, "cuda")
            # 断言设备类型为 "cuda"
            self.assertEqual(device.index, 0)
            # 断言设备索引为 0

        # Test rank works as well.
        # 同样测试 rank 的功能

        for remote_module in self._create_remote_module_iter(
            f"rank:{dst_rank}/cuda:0", modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            # 再次遍历远程模块迭代器，使用 "rank:<dst_rank>/cuda:0" 创建远程模块
            device = rpc.rpc_sync(
                dst_worker_name, remote_device, (remote_module.module_rref,)
            )
            # 使用 RPC 进行同步调用，获取远程设备信息
            self.assertEqual(device.type, "cuda")
            # 断言设备类型为 "cuda"
            self.assertEqual(device.index, 0)
            # 断言设备索引为 0
    # 测试无效设备情况的方法
    def test_invalid_devices(self):
        # 如果当前进程的 rank 不是 0，则直接返回，不执行后续代码
        if self.rank != 0:
            return
        
        # 计算目标 worker 的名称，使用分布式工具函数来确定下一个 worker 的名称
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        # 测试运行时错误，验证是否会捕获指定的异常信息
        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected one of .+ device type at start of device string",
        ):
            # 使用列表推导式迭代创建远程模块，捕获期望的运行时错误
            [
                m.forward()
                for m in self._create_remote_module_iter(
                    f"{dst_worker_name}/foo",
                    modes=[ModuleCreationMode.MODULE_CTOR],
                )
            ]

        # 测试运行时错误，验证是否会捕获 CUDA 错误的异常信息
        with self.assertRaisesRegex(
            RuntimeError, r"CUDA error: invalid device ordinal"
        ):
            # 使用列表推导式迭代创建远程模块，捕获 CUDA 错误的异常信息
            [
                m.forward()
                for m in self._create_remote_module_iter(
                    f"{dst_worker_name}/cuda:100",
                    modes=[ModuleCreationMode.MODULE_CTOR],
                )
            ]

        # 测试运行时错误，验证是否会捕获指定的异常信息
        with self.assertRaisesRegex(RuntimeError, r"Invalid device string: 'cpu2'"):
            # 使用列表推导式迭代创建远程模块，捕获期望的运行时错误
            [
                m.forward()
                for m in self._create_remote_module_iter(
                    f"{dst_worker_name}/cpu2",
                    modes=[ModuleCreationMode.MODULE_CTOR],
                )
            ]

        # 测试运行时错误，验证是否会捕获指定的异常信息
        with self.assertRaisesRegex(RuntimeError, r"Device string must not be empty"):
            # 使用列表推导式迭代创建远程模块，捕获期望的运行时错误
            [
                m.forward()
                for m in self._create_remote_module_iter(
                    f"{dst_worker_name}/",
                    modes=[ModuleCreationMode.MODULE_CTOR],
                )
            ]

        # 测试数值错误，验证是否会捕获指定的异常信息
        with self.assertRaisesRegex(
            ValueError,
            r"Could not parse remote_device: worker1/cuda:0/cuda:1. The valid format is '<workername>/<device>'",
        ):
            # 使用列表推导式迭代创建远程模块，捕获期望的数值错误
            [
                m.forward()
                for m in self._create_remote_module_iter(
                    f"{dst_worker_name}/cuda:0/cuda:1",
                    modes=[ModuleCreationMode.MODULE_CTOR],
                )
            ]

        # 测试数值错误，验证是否会捕获指定的异常信息
        with self.assertRaisesRegex(
            ValueError,
            r"Could not parse remote_device: /. The valid format is '<workername>/<device>'",
        ):
            # 使用列表推导式迭代创建远程模块，捕获期望的数值错误
            [
                m.forward()
                for m in self._create_remote_module_iter(
                    "/",
                    modes=[ModuleCreationMode.MODULE_CTOR],
                )
            ]

        # 测试数值错误，验证是否会捕获指定的异常信息
        with self.assertRaisesRegex(
            ValueError,
            r"Could not parse remote_device: /cuda:0. The valid format is '<workername>/<device>'",
        ):
            # 使用列表推导式迭代创建远程模块，捕获期望的数值错误
            [
                m.forward()
                for m in self._create_remote_module_iter(
                    "/cuda:0",
                    modes=[ModuleCreationMode.MODULE_CTOR],
                )
            ]

    @skip_if_lt_x_gpu(1)
    @dist_utils.dist_init
    # 定义测试方法，验证输入是否正确移动到 CUDA 设备上
    def test_input_moved_to_cuda_device(self):
        # 如果当前进程不是主进程 (rank != 0)，则直接返回
        if self.rank != 0:
            return
        # 计算目标工作节点的名称，根据当前进程号和总进程数确定
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        # 创建一个值为 1 的 CPU Tensor
        t1 = torch.ones(1)
        # 将 t1 和整数 2 组成参数元组 args
        args = (t1, 2)
        # 计算 t1 * 2 得到另一个 CPU Tensor t2
        t2 = t1 * 2
        # 将 t2 作为字典中的一个值，关联键名 'word'
        kwargs = dict(word=t2)

        # 只测试 Python 的 nn.Module，因为脚本模块方法不支持接收 kwargs
        # 遍历创建远程模块迭代器，远程模块位于指定 CUDA 设备上
        for remote_module in self._create_remote_module_iter(
            f"{dst_worker_name}/cuda:0", modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            # 异步调用远程模块的 forward 方法，传入 args 和 kwargs
            ret_fut = remote_module.forward_async(*args, **kwargs)
            # 等待异步结果
            ret = ret_fut.wait()
            # 断言返回结果与预期结果相等，反转 args 并添加 t2
            self.assertEqual(ret, tuple(reversed(args + (t2,))))
            # 检查返回结果的第一个和第三个元素的设备类型应为 CPU
            self.assertEqual(ret[0].device.type, "cpu")
            self.assertEqual(ret[2].device.type, "cpu")

            # 调用远程模块的 forward 方法，传入 args 和 kwargs
            ret = remote_module.forward(*args, **kwargs)
            # 断言返回结果与预期结果相等，反转 args 并添加 t2
            self.assertEqual(ret, tuple(reversed(args + (t2,))))
            # 检查返回结果的第一个和第三个元素的设备类型应为 CPU
            self.assertEqual(ret[0].device.type, "cpu")
            self.assertEqual(ret[2].device.type, "cpu")

    # 装饰器，如果系统中 GPU 数量少于 1，则跳过此测试
    @skip_if_lt_x_gpu(1)
    # 分布式初始化装饰器
    @dist_utils.dist_init
    # 定义测试方法，验证脚本模块的输入是否正确移动到 CUDA 设备上
    def test_input_moved_to_cuda_device_script(self):
        # 如果当前进程不是主进程 (rank != 0)，则直接返回
        if self.rank != 0:
            return
        # 计算目标工作节点的名称，根据当前进程号和总进程数确定
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        # 通过迭代器创建一个带有接口的脚本模块远程对象
        scripted_remote_module = next(
            self._create_remote_module_iter(
                f"{dst_worker_name}/cuda:0",
                modes=[ModuleCreationMode.MODULE_CTOR_WITH_INTERFACE],
            )
        )

        # 定义一个 Torch Script 函数，用于运行远程脚本模块的 forward 方法
        @torch.jit.script
        def run_forward(scripted_remote_module: MyModuleInterface):
            # 调用远程脚本模块的 forward 方法，传入 Tensor、整数和字符串
            ret = scripted_remote_module.forward(torch.ones(1), 2, "3")
            return ret

        # 执行 Torch Script 函数，传入远程脚本模块对象，并获取返回值
        ret = run_forward(scripted_remote_module)

        # 断言返回结果与预期结果相等
        self.assertEqual(ret, ("3", 2, torch.ones(1)))
        # 检查返回结果的第三个元素的设备类型应为 CPU
        self.assertEqual(ret[2].device.type, "cpu")
```