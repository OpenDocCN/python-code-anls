# `.\pytorch\torch\distributed\rpc\internal.py`

```py
# mypy: allow-untyped-defs
# 导入必要的库和模块
import collections  # 导入collections模块，用于处理集合数据类型
import copyreg  # 导入copyreg模块，用于注册自定义的对象序列化和反序列化函数
import io  # 导入io模块，提供了用于读取和写入流数据的核心工具
import pickle  # 导入pickle模块，用于序列化和反序列化Python对象
import sys  # 导入sys模块，提供了对Python运行时环境的访问
import threading  # 导入threading模块，提供了线程相关的操作功能
import traceback  # 导入traceback模块，用于提取和格式化异常堆栈信息
from enum import Enum  # 从enum模块导入Enum类，用于定义枚举类型

import torch  # 导入torch模块，PyTorch深度学习框架的核心库
import torch.distributed as dist  # 导入torch.distributed模块，提供分布式计算支持
from torch._C._distributed_rpc import _get_current_rpc_agent  # 导入_get_current_rpc_agent函数，用于获取当前的RPC代理对象

__all__ = ["RPCExecMode", "serialize", "deserialize", "PythonUDF", "RemoteException"]

# 线程本地的张量表，用于在序列化torch.Tensor对象时存储张量
_thread_local_tensor_tables = threading.local()
# 定义pickle.Pickler类的别名_pickler，用于序列化对象
_pickler = pickle.Pickler
# 定义pickle.Unpickler类的别名_unpickler，用于反序列化对象
_unpickler = pickle.Unpickler


class RPCExecMode(Enum):
    # 定义RPC执行模式枚举类，包括同步、异步、异步JIT编译和远程四种模式
    SYNC = "sync"
    ASYNC = "async"
    ASYNC_JIT = "async_jit"
    REMOTE = "remote"


class _InternalRPCPickler:
    r"""
    This class provides serialize() and deserialize() interfaces to serialize
    data to be "binary string + tensor table" format
    So for RPC python UDF function and args, non tensor data will be serialized
    into regular binary string, tensor data will be put into thread local tensor
    tables, this serialization format is consistent with builtin operator and args
    using JIT pickler. This format will make tensor handling in C++ much easier,
    e.g. attach tensor to distributed autograd graph in C++
    """

    def __init__(self):
        # 忽略类型错误，因为dispatch_table在第三方包中定义
        # 初始化_dispatch_table，复制copyreg模块的dispatch_table
        self._dispatch_table = copyreg.dispatch_table.copy()  # type: ignore[attr-defined]
        # 将torch.Tensor类注册到_dispatch_table中，使用自定义的张量序列化函数_tensor_reducer
        self._dispatch_table[torch.Tensor] = self._tensor_reducer
        # 用于注册自定义的pickler
        self._class_reducer_dict = {}

    def _register_reducer(self, obj_class, reducer):
        # 对于同一个类，只注册一次reducer函数
        if obj_class not in self._class_reducer_dict:
            self._class_reducer_dict[obj_class] = reducer

    @classmethod
    def _tensor_receiver(cls, tensor_index):
        # 全局_thread_local_tensor_tables中获取接收表recv_tables的张量数据
        global _thread_local_tensor_tables
        return _thread_local_tensor_tables.recv_tables[tensor_index]

    def _tensor_reducer(self, tensor):
        # 全局_thread_local_tensor_tables中将发送表send_tables添加张量数据tensor
        global _thread_local_tensor_tables
        _thread_local_tensor_tables.send_tables.append(tensor)
        tensor_index = len(_thread_local_tensor_tables.send_tables) - 1
        # 返回张量接收函数和索引，用于反序列化
        return (_InternalRPCPickler._tensor_receiver, (tensor_index,))

    @classmethod
    def _py_rref_receiver(cls, rref_fork_data):
        # 返回dist.rpc.PyRRef._deserialize方法的结果，反序列化远程引用数据
        return dist.rpc.PyRRef._deserialize(rref_fork_data)

    def _py_rref_reducer(self, py_rref):
        # 序列化PyRRef对象为其序列化数据
        rref_fork_data = py_rref._serialize()
        return (_InternalRPCPickler._py_rref_receiver, (rref_fork_data,))

    def _rref_reducer(self, rref):
        # 序列化远程引用对象为其序列化数据
        return self._py_rref_reducer(rref)

    @classmethod
    def _script_module_receiver(cls, script_module_serialized):
        """
        Given a serialized representation of a ScriptModule created with torch.jit.save,
        loads and returns the ScriptModule.
        """
        # 从序列化字节流中加载torch.jit.save创建的ScriptModule对象
        f = io.BytesIO(script_module_serialized)
        m = torch.jit.load(f)
        return m
    # 定义一个方法，用于序列化 ScriptModule。
    def _script_module_reducer(self, script_module):
        """
        Serializes a ScriptModule.
        """
        # 创建一个字节流对象
        f = io.BytesIO()
        # 使用 torch.jit.save 将 script_module 序列化到字节流中
        torch.jit.save(script_module, f)
        # 返回一个元组，包含一个方法和一个元组，用于在 RPC 过程中传输数据
        return (_InternalRPCPickler._script_module_receiver, (f.getvalue(),))

    # 定义一个方法，将对象序列化为二进制字符串，张量数据序列化为张量表
    def serialize(self, obj):
        r"""
        Serialize non tensor data into binary string, tensor data into
        tensor table
        """
        # 创建一个字节流对象
        f = io.BytesIO()
        # 创建一个 _pickler 对象，使用自定义的 dispatch_table
        p = _pickler(f)
        p.dispatch_table = self._dispatch_table

        # 设置 dispatch_table 以支持用户定义的 pickler 对象序列化 RRef
        p.dispatch_table[dist.rpc.PyRRef] = self._py_rref_reducer  # type: ignore[index]
        p.dispatch_table[dist.rpc.RRef] = self._rref_reducer  # type: ignore[index]

        # 如果 obj 是 torch.jit.ScriptModule 的实例，添加相应的 reducer 到 dispatch_table
        if isinstance(obj, torch.jit.ScriptModule):
            p.dispatch_table[obj.__class__] = self._script_module_reducer  # type: ignore[index]

        # 安装自定义的 picklers
        for class_name in self._class_reducer_dict.keys():
            p.dispatch_table[class_name] = self._class_reducer_dict[class_name]  # type: ignore[index]

        # 如果 _thread_local_tensor_tables 中有 send_tables 属性，保存它的值
        global _thread_local_tensor_tables
        if hasattr(_thread_local_tensor_tables, "send_tables"):
            old_send_tables = _thread_local_tensor_tables.send_tables
        else:
            old_send_tables = None
        _thread_local_tensor_tables.send_tables = []

        # 使用 _pickler 对象将 obj 序列化到字节流中
        p.dump(obj)

        # 恢复 _thread_local_tensor_tables 的 send_tables 属性
        tensors = _thread_local_tensor_tables.send_tables
        if old_send_tables is not None:
            _thread_local_tensor_tables.send_tables = old_send_tables
        else:
            del _thread_local_tensor_tables.send_tables

        # 返回包含序列化后的数据和张量表的元组
        return (f.getvalue(), tensors)
    def deserialize(self, binary_data, tensor_table):
        r"""
        Deserialize binary string + tensor table to original obj
        """
        # 保存 _thread_local_tensor_tables.recv_tables，如果在嵌套调用中存在的话
        global _thread_local_tensor_tables
        if hasattr(_thread_local_tensor_tables, "recv_tables"):
            # 存储旧的接收表以便稍后恢复
            old_recv_tables = _thread_local_tensor_tables.recv_tables
        else:
            old_recv_tables = None
        # 设置当前线程的接收表为传入的 tensor_table
        _thread_local_tensor_tables.recv_tables = tensor_table

        try:
            # 创建一个 unpickler 对象，从二进制数据中加载对象
            unpickler = _unpickler(io.BytesIO(binary_data))
            ret = unpickler.load()
        except AttributeError as e:
            # 在反序列化时发生 AttributeError，通常是因为模块或类中找不到函数定义
            # 添加详细的异常信息和建议，以确保用户明白如何解决问题
            except_str = (
                str(e)
                + """ Default RPC pickler does not serialize
            function code. Ensure that UDFs are defined on both caller and
            callee modules."""
            )
            ret = AttributeError(except_str)
            # 确保保留原始异常的堆栈信息
            ret.__cause__ = e

        # 如果存在旧的接收表，则恢复为之前保存的值
        # 如果没有嵌套调用，则清理掉接收表
        if old_recv_tables is not None:
            _thread_local_tensor_tables.recv_tables = old_recv_tables
        else:
            del _thread_local_tensor_tables.recv_tables

        # 返回反序列化得到的对象或者异常对象
        return ret
# 创建 _internal_rpc_pickler 对象，用于序列化和反序列化 RPC 数据
_internal_rpc_pickler = _InternalRPCPickler()


def serialize(obj):
    # 使用 _internal_rpc_pickler 对象序列化给定的对象
    return _internal_rpc_pickler.serialize(obj)


def deserialize(binary_data, tensor_table):
    # 使用 _internal_rpc_pickler 对象反序列化二进制数据，同时传入张量表
    return _internal_rpc_pickler.deserialize(binary_data, tensor_table)


def _run_function(python_udf):
    r"""
    This function is exclusively called from C++.
    See ``torch/csrc/distributed/rpc/python_rpc_handler.cpp``.

    Runs a Python UDF and returns its return value.
    Wraps any exception in ``RemoteException`` if the function raises.
    """
    try:
        # 如果 python_udf 是 AttributeError 类型，则抛出该异常
        if isinstance(python_udf, AttributeError):
            raise python_udf
        # 否则调用 python_udf.func 方法执行用户定义的函数，传入参数和关键字参数
        result = python_udf.func(*python_udf.args, **python_udf.kwargs)
    except Exception as e:
        # 如果出现异常，则捕获异常信息和堆栈轨迹
        except_str = (
            f"On {_get_current_rpc_agent().get_worker_info()}:\n"
            f"{repr(e)}\n{traceback.format_exc()}"
        )
        # 打印异常信息到标准错误输出
        print(except_str, file=sys.stderr)
        # 将异常信息封装成 RemoteException 对象，用于返回
        result = RemoteException(except_str, type(e))
    return result


def _handle_exception(result):
    if isinstance(result, RemoteException):
        # 解码异常消息为 Unicode 编码
        exception_msg = result.msg.encode("utf-8").decode("unicode_escape")
        # 尝试重新创建异常对象，以避免某些异常类无法直接从字符串构造的问题
        exc = None
        try:
            exc = result.exception_type(exception_msg)
        except BaseException as e:
            # 如果无法创建异常类型，则抛出运行时异常
            raise RuntimeError(
                f"Failed to create original exception type. Error msg was {str(e)}"
                f" Original exception on remote side was {exception_msg}"
            ) from e

        # 如果成功创建异常对象，则抛出该异常
        if exc is not None:
            raise exc


def _build_rpc_profiling_key(
    exec_type, func_name, current_worker_name, dst_worker_name
):
    """
    Builds the key that RPC calls are profiled with using the autograd profiler.
    This will be the name of the corresponding Event recorded in the profiler.

    Args:
        exec_type (RPCExecMode): Type of RPC/RRef call
        func_name (str): Name of function being profiled.
        current_worker_name (str): Name of current worker.
        dst_worker_name (str): Name of the destination worker.

    Returns:
        String representing profiling key
    """
    # 构建用于 RPC 调用的性能分析键，包括执行类型、函数名、当前和目标 worker 名称
    profile_key = (
        f"rpc_{exec_type.value}#{func_name}({current_worker_name} -> {dst_worker_name})"
    )
    return profile_key


def _start_record_function(exec_type, func_name, current_worker_name, dest_worker_name):
    """
    This function should be called from RPC/RRef functions to create a
    RecordFunction object for profiling. This function also runs the before
    callbacks that start the profiling, though the user is responsible for
    running the appropriate callbacks when the function to be profiled finishes.
    """
    # 此函数应从 RPC/RRef 函数中调用，用于创建一个 RecordFunction 对象以进行性能分析
    # 同时运行开始性能分析的回调，但用户需负责在函数执行完毕时运行相应的回调
    Args:
        exec_type (RPCExecMode): RPC/RRef调用的类型
        func_name (str): 正在被分析的函数的名称。
        current_worker_name (str): 当前worker的名称。
        dest_worker_name (str): 目标worker的名称。

    Returns:
        `torch.autograd._RecordFunction`的一个实例。
    """
    # 断言，确保自动求导分析器已启用
    assert torch.autograd._profiler_enabled(), "Autograd profiler should be enabled."
    # 根据给定的参数构建用于分析的唯一标识符
    profile_key = f"rpc_{exec_type.value}#{str(func_name)}({current_worker_name} -> {dest_worker_name})"
    # 创建一个记录函数对象
    rf = torch.autograd._RecordFunction()  # type: ignore[attr-defined]
    # 运行记录函数对象的前置回调
    torch.autograd._run_before_callbacks(rf, profile_key)  # type: ignore[attr-defined]
    # 返回记录函数对象
    return rf
# 创建名为 PythonUDF 的命名元组，包含字段 func、args、kwargs，用于表示 Python 中的用户定义函数信息
PythonUDF = collections.namedtuple("PythonUDF", ["func", "args", "kwargs"])

# 创建名为 RemoteException 的命名元组，包含字段 msg 和 exception_type，用于表示远程异常的信息
RemoteException = collections.namedtuple("RemoteException", ["msg", "exception_type"])
```