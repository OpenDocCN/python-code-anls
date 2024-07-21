# `.\pytorch\torch\distributed\rpc\api.py`

```
# mypy: allow-untyped-defs

# 导入必要的模块和库
import collections
import contextlib
import functools
import inspect
import logging
import threading
from typing import Any, Dict, Generic, Set, TYPE_CHECKING, TypeVar

# 导入PyTorch相关模块和函数
import torch
from torch._C._distributed_rpc import (
    _cleanup_python_rpc_handler,
    _delete_all_user_and_unforked_owner_rrefs,
    _destroy_rref_context,
    _get_current_rpc_agent,
    _invoke_remote_builtin,
    _invoke_remote_python_udf,
    _invoke_remote_torchscript,
    _invoke_rpc_builtin,
    _invoke_rpc_python_udf,
    _invoke_rpc_torchscript,
    _is_current_rpc_agent_set,
    _reset_current_rpc_agent,
    _set_and_start_rpc_agent,
    get_rpc_timeout,
    PyRRef,
    RemoteProfilerManager,
    TensorPipeAgent,
    WorkerInfo,
)
from torch.futures import Future

# 导入局部模块和函数
from ._utils import _group_membership_management, _update_group_membership
from .constants import DEFAULT_SHUTDOWN_TIMEOUT, UNSET_RPC_TIMEOUT
from .internal import (
    _build_rpc_profiling_key,
    _internal_rpc_pickler,
    PythonUDF,
    RPCExecMode,
)

# 暴露给外部的接口名称列表
__all__ = [
    "shutdown",
    "get_worker_info",
    "remote",
    "rpc_sync",
    "rpc_async",
    "RRef",
    "AllGatherStates",
    "method_factory",
    "new_method",
]

# 设置日志记录器
logger = logging.getLogger(__name__)

# 忽略在关闭时可能发生的RRef泄漏问题
# 默认情况下，设置为True，表示忽略RRef泄漏
_ignore_rref_leak = True

# 默认使用的序列化器
_default_pickler = _internal_rpc_pickler


@contextlib.contextmanager
def _use_rpc_pickler(rpc_pickler):
    r"""
    rpc_pickler: (.internal._InternalRPCPickler) Overrides the default RPC pickler
    """
    global _default_pickler
    _default_pickler = rpc_pickler
    try:
        yield
    finally:
        _default_pickler = _internal_rpc_pickler


def _require_initialized(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 检查当前的RPC代理是否已设置，若未设置则抛出运行时错误
        if not _is_current_rpc_agent_set():
            raise RuntimeError(
                "RPC has not been initialized. Call "
                "torch.distributed.rpc.init_rpc first."
            )
        return func(*args, **kwargs)

    return wrapper


class AllGatherStates:
    # 这里是类的定义，包含各种属性和方法，具体实现在后续代码中进行
    def __init__(self):
        """
        构造函数初始化对象

        每个 `gathered_objects` 在开始时是一个空字典。
        领导者(worker)被选为按照工作者名称排序后的第一个工作者(worker)。
        每当有一个工作者(worker)进入 `_all_gather()`，它会在领导者上运行 `_gather_to_leader()`，
        将自己的名称和数据对象添加到这个字典中。领导者还在调用 `_all_gather()` 时将自己的名称添加到字典中。
        
        一旦 `set(gathered_objects.keys()) == _ALL_WORKER_NAMES`，领导者将把收集到的字典广播给所有的跟随者(workers)，
        并设置它们的 `gathered_objects` 字段和 `proceed_signal` 字段。
        """
        # `gathered_objects` 字段，用于存储收集到的对象信息的字典
        self.gathered_objects = {}

        # `proceed_signal` 字段，所有工作者(worker)在此信号上等待，直到接收到所有收集到的对象
        self.proceed_signal = threading.Event()
# 在 `def _all_gather()` 中使用的状态变量。
# `_ALL_WORKER_NAMES` 在初始化 RPC 层时被初始化为一个空集合。
# `_all_gather_dict_lock` 是一个线程锁，用于保护多线程环境下的字典操作。
# `_all_gather_sequence_id` 是一个字典，用于存储序列 ID 到整数的映射。
# `_all_gather_sequence_id_to_states` 是一个 defaultdict，默认值为 `AllGatherStates` 类的实例，用于存储序列 ID 到状态对象的映射。

def _init_rpc_states(agent):
    # 获取代理对象的所有工作节点信息
    worker_infos = agent.get_worker_infos()
    # 将所有工作节点的名称存入全局变量 `_ALL_WORKER_NAMES`
    global _ALL_WORKER_NAMES
    _ALL_WORKER_NAMES = {worker_info.name for worker_info in worker_infos}

    # 如果当前 RPC 代理尚未设置，则设置并启动 RPC 代理
    if not _is_current_rpc_agent_set():
        _set_and_start_rpc_agent(agent)


def _gather_to_leader(sequence_id, worker_name, obj, worker_names=None):
    # 使用 `_all_gather_dict_lock` 进行加锁，确保线程安全
    with _all_gather_dict_lock:
        # 如果未指定工作节点名称，则使用 `_ALL_WORKER_NAMES`
        if not worker_names:
            worker_names = _ALL_WORKER_NAMES
            # 断言当前工作节点名称在预期的工作节点名称集合中
            assert (
                worker_name in worker_names
            ), f"{worker_name} is not expected by leader."
        # 获取当前序列 ID 对应的状态对象
        states = _all_gather_sequence_id_to_states[sequence_id]
        # 断言当前工作节点名称在已收集对象的状态中不存在
        assert (
            worker_name not in states.gathered_objects
        ), f"{worker_name} reported intent sequence id {sequence_id} twice. "
        # 将当前工作节点的数据对象存入已收集对象的状态中
        states.gathered_objects[worker_name] = obj
        # 如果已收集完所有预期的工作节点数据，则设置继续信号
        if worker_names == set(states.gathered_objects.keys()):
            states.proceed_signal.set()


def _broadcast_to_followers(sequence_id, objects_map):
    # 使用 `_all_gather_dict_lock` 进行加锁，确保线程安全
    with _all_gather_dict_lock:
        # 获取当前序列 ID 对应的状态对象
        states = _all_gather_sequence_id_to_states[sequence_id]

    # 断言当前未设置终止信号
    assert (
        not states.proceed_signal.is_set()
    ), f"Termination signal sequence id {sequence_id} got set twice."
    # 设置已收集对象的状态为给定的对象映射
    states.gathered_objects = objects_map
    # 设置继续信号
    states.proceed_signal.set()


# 线程本地变量
_thread_local_var = threading.local()


@contextlib.contextmanager
def _wait_all():
    r"""
    一个上下文管理器，用于收集所有 `rpc_async` 返回的 futures，并在退出上下文管理器时等待它们的完成，避免用户需要显式调用 wait。

    示例::
        >>> # xdoctest: +SKIP("distributed")
        >>> # 在工作节点 0 上:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> with rpc._wait_all():
        >>>    fut_1 = rpc.rpc_async(dst, torch.add, (torch.ones(2, 2), 1))
        >>>    fut_2 = rpc.rpc_async(dst, torch.add, (torch.ones(2, 2), 1))
        >>> # fut_1 和 fut_2 将会被等待
    """
    _thread_local_var.future_list = []
    try:
        yield
    finally:
        try:
            # 等待所有线程本地变量中的 futures 完成
            torch.futures.wait_all(_thread_local_var.future_list)
        finally:
            # 清除线程本地变量中的 futures 列表
            del _thread_local_var.future_list


@_require_initialized
def _all_gather(obj, worker_names=None, timeout: float = UNSET_RPC_TIMEOUT):
    r"""
    类似于 `torch.distributed.all_gather()`，但使用 RPC 实现。
    选择名称最小的工作节点作为领导者，然后所有跟随者将它们的数据 `obj` 发送给领导者。
    领导者在收到所有数据后广播给所有跟随者。

    注意：此函数需要在 RPC 初始化后调用。

    参数：
    - `obj`：要收集的数据对象
    - `worker_names`：要收集数据的工作节点名称集合，默认为 `None`
    - `timeout`：RPC 超时时间，默认为未设置的 RPC 超时时间
    """
    has received all, it will broadcast the results back to all followers. This
    function blocks until all workers have received the gathered results.
    """
    如果没有指定 worker_names，则确保 _ALL_WORKER_NAMES 不为空，
    否则会抛出异常，表明 `_ALL_WORKER_NAMES` 未在 `def _all_gather` 中初始化。
    """
    # 如果未提供 worker_names 参数，则使用全局变量 _ALL_WORKER_NAMES
    if not worker_names:
        assert (
            _ALL_WORKER_NAMES is not None
        ), "`_ALL_WORKER_NAMES` is not initialized for `def _all_gather`."
        worker_names = _ALL_WORKER_NAMES

    # 从 worker_names 中选择字典序最小的名称作为 leader_name
    leader_name = min(worker_names)

    # 获取当前节点的名称
    self_name = _get_current_rpc_agent().get_worker_info().name

    # 使用线程安全的方式处理 _all_gather_dict_lock
    with _all_gather_dict_lock:
        # 将 worker_names 按照字典序连接成一个字符串
        concat_names = "".join(sorted(worker_names))
        # 获取当前连接字符串对应的序列号，若不存在则初始化为 0
        sequence_num = _all_gather_sequence_id.get(concat_names, 0)
        # 更新序列号并生成完整的 sequence_id
        _all_gather_sequence_id[concat_names] = sequence_num + 1
        sequence_id = concat_names + str(sequence_num)

    # 判断当前节点是否为 leader
    is_leader = leader_name == self_name

    # 根据 timeout 的不同取值设置 rpc_timeout 和 signal_timeout
    if timeout == UNSET_RPC_TIMEOUT:
        # 使用默认的 RPC 超时时间
        rpc_timeout = get_rpc_timeout()
        # 信号超时设为 None
        signal_timeout = None
    elif timeout == DEFAULT_SHUTDOWN_TIMEOUT:
        # RPC 超时设为默认的关闭超时时间
        rpc_timeout = timeout
        # 信号超时设为 None
        signal_timeout = None
    else:
        # RPC 超时和信号超时均设为 timeout 指定的时间
        signal_timeout = rpc_timeout = timeout

    # Phase 1: Followers send it's object to the leader
    # 第一阶段：Followers 将自己的对象发送给 leader
    if is_leader:
        # 如果当前节点是 leader，则调用 _gather_to_leader 函数进行数据收集
        _gather_to_leader(sequence_id, self_name, obj, worker_names)
    else:
        # 如果当前节点不是 leader，则通过 rpc_sync 向 leader 发送数据收集请求
        rpc_sync(
            leader_name,
            _gather_to_leader,
            args=(sequence_id, self_name, obj, worker_names),
            timeout=rpc_timeout,
        )

    # 使用线程安全的方式处理 _all_gather_dict_lock
    with _all_gather_dict_lock:
        # 获取与当前 sequence_id 对应的状态对象 states
        states = _all_gather_sequence_id_to_states[sequence_id]

    # 设置信号超时时间，用于等待状态的改变
    states.proceed_signal.wait(timeout=signal_timeout)

    # Phase 2: Leader broadcast gathered results to all followers
    # 第二阶段：Leader 将收集的结果广播给所有的 followers
    # Leader 的信号首先解除阻塞，等待接收所有 followers 的数据对象后才解除阻塞。
    if is_leader:
        # 如果当前节点是 leader，则向所有 followers 发送广播请求
        worker_name_to_response_future_dict = {}
        for follower_name in worker_names - {leader_name}:
            fut = rpc_async(
                follower_name,
                _broadcast_to_followers,
                args=(sequence_id, states.gathered_objects),
                timeout=rpc_timeout,
            )
            worker_name_to_response_future_dict[follower_name] = fut

        # 处理可能出现的超时异常
        errors = []
        for follower_name, fut in worker_name_to_response_future_dict.items():
            try:
                fut.wait()
            except RuntimeError as ex:
                errors.append((follower_name, ex))

        # 如果有超时异常发生，则抛出 RuntimeError
        if errors:
            raise RuntimeError(
                f"Followers {[e[0] for e in errors]} timed out in _all_gather "
                f"after {rpc_timeout:.2f} seconds. The first exception is {errors[0][1]}"
            )

    # 使用 sequence_id 清理状态信息
    # Clean up for the states using the sequence_id
    # 使用全局锁 `_all_gather_dict_lock` 进行上下文管理，确保操作的原子性和线程安全
    with _all_gather_dict_lock:
        # 从字典 `_all_gather_sequence_id_to_states` 中弹出指定 `sequence_id` 对应的值，并赋给 `states`
        states = _all_gather_sequence_id_to_states.pop(sequence_id)
    # 返回从 `states` 中提取的 `gathered_objects` 属性
    return states.gathered_objects
# 要求函数调用前必须进行初始化，否则会引发异常
@_require_initialized
def _barrier(worker_names):
    r"""
    同步本地和远程的 RPC 进程。

    这会阻塞，直到所有指定的本地和远程 RPC 进程在 worker_names 中达到此方法以等待所有未完成的工作完成为止。

    Args:
        worker_names (List[str]): 要同步的 worker 名称集合。

    """
    try:
        # 调用 _all_gather 函数，同步所有指定 worker 的信息
        _all_gather(None, set(worker_names))
    except RuntimeError as ex:
        # 如果出现运行时异常，记录错误日志并抛出异常
        logger.error("Failed to complete barrier, got error %s", ex)


# 要求函数调用前必须进行初始化
@_require_initialized
def _wait_all_workers(timeout=DEFAULT_SHUTDOWN_TIMEOUT):
    r"""
    阻塞，直到所有本地和远程 RPC 进程达到此方法并等待所有未完成的工作完成。
    每个 RPC 进程在退出之前必须调用此方法以执行优雅的关闭。
    该方法用于终止 RPC 框架，方法返回后不能保证 RPC 框架继续工作。
    """
    try:
        # 调用 _all_gather 函数，同步所有 RPC 进程的信息，设置超时时间
        _all_gather(None, timeout=timeout)
    except RuntimeError as ex:
        # 如果出现运行时异常，记录错误日志并抛出异常
        logger.error(
            "Failed to respond to 'Shutdown Proceed' in time, got error %s", ex
        )
        raise ex


# 要求函数调用前必须进行初始化
@_require_initialized
def shutdown(graceful=True, timeout=DEFAULT_SHUTDOWN_TIMEOUT):
    r"""
    执行 RPC 代理的关闭操作，然后销毁 RPC 代理。
    停止本地代理接受未完成请求，并通过终止所有 RPC 线程来关闭 RPC 框架。
    如果 graceful=True，则会阻塞，直到所有本地和远程 RPC 进程达到此方法并等待所有未完成的工作完成。
    否则，如果 graceful=False，则只是本地关闭，不会等待其他 RPC 进程达到此方法。

    .. warning::
        对于由 :meth:`~torch.distributed.rpc.rpc_async` 返回的 :class:`~torch.futures.Future` 对象，
        在调用 ``shutdown()`` 后不应再调用 ``future.wait()``。

    Args:
        graceful (bool): 是否进行优雅关闭。如果为 True，
                         这将 1) 等待 ``UserRRefs`` 的所有待处理系统消息并删除它们；2) 阻塞，
                         直到所有本地和远程 RPC 进程达到此方法并等待所有未完成的工作完成。

        timeout (int): 等待超时时间，单位为秒，默认为 DEFAULT_SHUTDOWN_TIMEOUT。

    """
    if graceful:
        try:
            # 获取当前的 RPC 代理
            agent = _get_current_rpc_agent()
            # 检查代理是否为 TensorPipeAgent 类型并且是否为静态组
            if not isinstance(agent, TensorPipeAgent) or agent.is_static_group:
                # 等待所有工作进程完成，超时时间为 timeout
                _wait_all_workers(timeout)
                # 删除所有用户和非分叉所有者的 RRef
                _delete_all_user_and_unforked_owner_rrefs()
                # 关闭代理并加入，等待操作完成
                agent.join(shutdown=True, timeout=timeout)
            else:
                # 这是一个动态组，需要获取操作的令牌
                my_worker_info = agent.get_worker_info()
                my_name = my_worker_info.name
                # 执行组成员管理操作
                with _group_membership_management(agent.store, my_name, False):
                    # 获取所有工作进程的信息
                    all_worker_infos = agent.get_worker_infos()
                    # 遍历所有工作进程
                    for worker in all_worker_infos:
                        if worker.name != my_name:
                            # 同步发送更新组成员信息的 RPC 请求
                            rpc_sync(
                                worker.name,
                                _update_group_membership,
                                args=(my_worker_info, [], {}, False),
                            )
                    # 关闭代理并加入，等待操作完成
                    agent.join(shutdown=True, timeout=timeout)
        finally:
            # 处理错误情况下的本地关闭
            _finalize_shutdown()
    else:
        # 执行最终的本地关闭操作
        _finalize_shutdown()
def _finalize_shutdown():
    try:
        # 检查是否存在 RRef 泄漏，可能会引发 `TORCH_CHECK()` 异常。
        _destroy_rref_context(_ignore_rref_leak)
    finally:
        # 关闭当前 RPC Agent 实例。
        _get_current_rpc_agent().shutdown()
        # 在 shutdown() 中清理 Python RPC 处理器，参见 PythonRpcHandler::cleanup()，
        # 在 Python API 中调用它是因为 cleanup() 函数有 Python 依赖，假设 Python 解释器已存在。
        # 无论是否引发 RRef 泄漏异常，这段清理代码都必须运行，以避免在 Python 3.5 中出现破坏性段错误。
        #
        # 在 shutdown() 后不应调用 future.wait()。
        # pythonRpcHandler 在 shutdown() 中被清理，在 shutdown() 后，无法解析从 rpc Python 调用返回的 Python 对象。
        _cleanup_python_rpc_handler()
        # 重置当前 RPC Agent 实例。
        _reset_current_rpc_agent()


@_require_initialized
def get_worker_info(worker_name=None):
    r"""
    获取给定 worker 名称的 :class:`~torch.distributed.rpc.WorkerInfo`。
    使用这个 :class:`~torch.distributed.rpc.WorkerInfo` 来避免在每次调用时传递昂贵的字符串。

    Args:
        worker_name (str): worker 的字符串名称。如果为 ``None``，返回当前 worker 的 id。 (默认为 ``None``)

    Returns:
        :class:`~torch.distributed.rpc.WorkerInfo` 实例，对于给定的 ``worker_name``，或者如果 ``worker_name`` 是 ``None``，则返回当前 worker 的 :class:`~torch.distributed.rpc.WorkerInfo`。
    """
    if worker_name is not None:
        return _get_current_rpc_agent().get_worker_info(worker_name)
    else:
        return _get_current_rpc_agent().get_worker_info()


def _to_worker_info(to):
    if isinstance(to, WorkerInfo):
        return to
    elif isinstance(to, (str, int)):
        return get_worker_info(to)
    else:
        raise ValueError(f"Cannot get WorkerInfo from name {to}")


def _rref_typeof_on_owner(rref, blocking: bool = True):
    rref_type = type(rref.local_value())
    if blocking:
        return rref_type
    else:
        # 将结果封装到已完成的 Future 中。这样，无论是在用户端还是所有者端，如果指定了 blocking=`False`，我们都返回一个 Future。
        future = Future[type]()
        future.set_result(rref_type)
        return future


def _rref_typeof_on_user(
    rref, timeout: float = UNSET_RPC_TIMEOUT, blocking: bool = True
):
    # 异步调用，获取所有者端的 rref 类型。
    fut = rpc_async(rref.owner(), _rref_typeof_on_owner, args=(rref,), timeout=timeout)
    if blocking:
        return fut.wait()
    else:
        return fut


T = TypeVar("T")
GenericWithOneTypeVar = Generic[T]


if TYPE_CHECKING:

    class RRef(PyRRef[T], Generic[T]):
        pass

else:
    try:
        # 合并实现类和类型类。
        class RRef(PyRRef, Generic[T]):
            pass
    except TypeError:
        # 当捕获到 TypeError 异常时执行以下代码块
        # TypeError: metaclass conflict: the metaclass of a derived class
        # must be a (non-strict) subclass of the metaclasses of all its bases
        # Mypy 不理解 __class__ (mypy bug #4177)

        # 定义一个新的元类 RRefMeta，继承自 PyRRef.__class__ 和 GenericWithOneTypeVar.__class__
        class RRefMeta(PyRRef.__class__, GenericWithOneTypeVar.__class__):  # type: ignore[name-defined, misc, valid-type]
            pass

        # 结合实现类和类型类。
        # 为期望特定泛型参数的类提供类型（mypy bug #7791）
        # 定义一个新的类 RRef，它继承自 PyRRef 和 GenericWithOneTypeVar，并使用 RRefMeta 作为元类
        class RRef(PyRRef, GenericWithOneTypeVar, metaclass=RRefMeta):  # type: ignore[misc, no-redef, valid-type]
            pass
# 从 `PyRRef` 到 `RRef` 安装文档字符串。
#
# 这是因为 pybind11 生成的参数 `self` 类型为 `rpc.PyRRef`，
# 因此在 `.. autoclass:: RRef` 下的 `:inherited-members:` 不起作用。
# 我们必须执行以下过程将 `rpc.PyRRef` 替换为 `rpc.RRef`。

def method_factory(method_name, docstring):
    # 创建一个内部方法，调用 `super(RRef, self)` 的 `method_name` 方法，并返回结果
    def method(self, *args, **kwargs):
        return getattr(super(RRef, self), method_name)(*args, **kwargs)

    # 如果方法已经有文档字符串，则替换为传入的 `docstring`
    if method.__doc__:
        method.__doc__ = docstring
    return method


# 遍历 `PyRRef` 的所有方法和对应的名称
for method_name, method in inspect.getmembers(PyRRef):
    # 忽略魔术方法，除非是 "__str__"
    if method_name.startswith("_") and method_name != "__str__":
        continue

    # 获取由 pybind11 生成的文档字符串
    docstring = getattr(method, "__doc__", None)
    assert docstring is not None, "RRef user-facing methods should all have docstrings."

    # 在 pybind11 生成的文档字符串中执行替换
    docstring = docstring.replace(
        "torch.distributed.rpc.PyRRef", "torch.distributed.rpc.RRef"
    )

    # 使用修改后的文档字符串附加用户可见的 RRef 方法
    new_method = method_factory(method_name, docstring)
    setattr(RRef, method_name, new_method)


# `_require_initialized` 装饰器修饰的函数，要求在调用之前需要初始化
@_require_initialized
def remote(to, func, args=None, kwargs=None, timeout=UNSET_RPC_TIMEOUT):
    r"""
    在远程工作器 `to` 上调用 `func` 函数，并立即返回结果值的 :class:`~torch.distributed.rpc.RRef`。
    工作器 `to` 将成为返回的 :class:`~torch.distributed.rpc.RRef` 的所有者，
    调用 `remote` 的工作器是用户。所有者管理其 :class:`~torch.distributed.rpc.RRef` 的全局引用计数，
    并且只有当全局引用计数为零时，才会销毁所有者 :class:`~torch.distributed.rpc.RRef`。
    Args:
        to (str or WorkerInfo or int): 目标 worker 的名称/等级/WorkerInfo 对象。
        func (Callable): 可调用函数，例如 Python 可调用对象、内置运算符（如 :meth:`~torch.add`）和注释的 TorchScript 函数。
        args (tuple): 传递给 func 的参数元组。
        kwargs (dict): 传递给 func 的关键字参数字典。

        timeout (float, optional): 远程调用的超时时间，单位为秒。如果在当前 worker 上在超时时间内未能成功处理目标 worker 上的 :class:`~torch.distributed.rpc.RRef` 创建，那么下次尝试使用 RRef（如 ``to_here()``）时将引发超时错误。0 表示无限超时，即永不引发超时错误。如果未提供，则使用在初始化期间或使用 ``_set_rpc_timeout`` 设置的默认值。

    Returns:
        返回一个用户定义的 :class:`~torch.distributed.rpc.RRef` 实例，指向结果值。使用阻塞 API :meth:`torch.distributed.rpc.RRef.to_here` 在本地检索结果值。

    .. warning ::
        ``remote`` API 在发送参数张量之前不会复制存储，这可能由不同线程（取决于 RPC 后端类型）执行。调用者应确保这些张量的内容保持完整，直到所有者确认返回的 RRef，可以使用 :meth:`torch.distributed.rpc.RRef.confirmed_by_owner` API 进行检查。

    .. warning ::
        对于 ``remote`` API 的错误（例如超时），我们采取尽力而为的处理方式。这意味着当由 ``remote`` 启动的远程调用失败时（例如超时错误），我们会以尽力而为的方式处理错误。这意味着错误会异步处理并设置到结果的 RRef 上。如果 RRef 在此处理前尚未被应用程序使用（如 ``to_here`` 或 fork 调用），则未来对 ``RRef`` 的使用将适当地引发错误。然而，用户应用程序可能会在错误处理之前使用 ``RRef``，此时可能不会引发错误，因为错误尚未被处理。
    """
    # 记录一次使用分布式 RPC 的 API 调用
    torch._C._log_api_usage_once("torch.distributed.rpc_remote")
    # 获取 TorchScript 函数的限定名称
    qualified_name = torch.jit._builtins._find_builtin(func)
    # 将远程目标信息转换为适当的格式
    dst_worker_info = _to_worker_info(to)
    # 获取是否应该进行性能分析的标志
    should_profile = _get_should_profile()

    # 启用 RPC 分析器的上下文管理器
    ctx_manager = _enable_rpc_profiler(
        should_profile, qualified_name, func, RPCExecMode.REMOTE, dst_worker_info
    )
    """
    # 使用上下文管理器 `ctx_manager` 执行一系列远程函数调用，并返回远程引用 `rref`
    with ctx_manager as rf:
        # 如果 `args` 为假值，则设为空元组
        args = args if args else ()
        # 如果 `kwargs` 为假值，则设为空字典
        kwargs = kwargs if kwargs else {}
    
        # 检查函数 `func` 是否具有 `_wrapped_async_rpc_function` 属性，用于判断是否异步执行
        is_async_exec = hasattr(func, "_wrapped_async_rpc_function")
    
        # 如果是异步执行，则将 `func` 替换为 `_wrapped_async_rpc_function` 属性指向的函数
        if is_async_exec:
            wrapped = func._wrapped_async_rpc_function
            # 如果替换后的函数是 `torch.jit.ScriptFunction` 类型，则将 `func` 更新为 `wrapped`
            if isinstance(wrapped, torch.jit.ScriptFunction):
                func = wrapped
    
        # 如果提供了 `qualified_name` 参数，则调用内置函数 `_invoke_remote_builtin` 执行远程调用
        if qualified_name is not None:
            rref = _invoke_remote_builtin(
                dst_worker_info, qualified_name, timeout, *args, **kwargs
            )
        # 如果 `func` 是 `torch.jit.ScriptFunction` 类型，则调用 `_invoke_remote_torchscript` 执行远程 TorchScript 脚本调用
        elif isinstance(func, torch.jit.ScriptFunction):
            rref = _invoke_remote_torchscript(
                dst_worker_info.name,
                torch._jit_internal._qualified_name(func),
                timeout,
                is_async_exec,
                *args,
                **kwargs,
            )
        else:
            # 否则，使用 `_default_pickler.serialize` 序列化 PythonUDF 对象和相关张量
            (pickled_python_udf, tensors) = _default_pickler.serialize(
                PythonUDF(func, args, kwargs)
            )
            # 调用 `_invoke_remote_python_udf` 执行远程 Python UDF 调用
            rref = _invoke_remote_python_udf(
                dst_worker_info, pickled_python_udf, tensors, timeout, is_async_exec
            )
        
        # 如果 `should_profile` 为真，则附加性能分析信息
        if should_profile:
            # 断言当前是否启用了 PyTorch 自动求导的性能分析
            assert torch.autograd._profiler_enabled()
            # 断言 `rf` 不为空
            assert rf is not None
            # 获取远程引用 `rref` 的未来对象，并在其上调用结束回调函数
            fut = rf._call_end_callbacks_on_future(rref._get_future())
            # 将性能分析的未来对象设置给远程引用 `rref`
            rref._set_profiling_future(fut)
    
    # 返回远程引用 `rref`
    return rref
# 定义一个函数用于执行 RPC 调用，支持同步和异步执行
def _invoke_rpc(
    to, func, rpc_type, args=None, kwargs=None, rpc_timeout: float = UNSET_RPC_TIMEOUT
):
    # 检查 func 是否可调用，否则抛出类型错误异常
    if not callable(func):
        raise TypeError("function should be callable.")

    # 根据 func 查找其在 Torch 中的内置函数的限定名
    qualified_name = torch.jit._builtins._find_builtin(func)
    
    # 将目标地址 to 转换为 worker 的信息
    dst_worker_info = _to_worker_info(to)

    # 获取是否应该进行性能分析的标志
    should_profile = _get_should_profile()

    # 启用 RPC 调用性能分析的上下文管理器
    ctx_manager = _enable_rpc_profiler(
        should_profile, qualified_name, func, rpc_type, dst_worker_info
    )

    # 进入性能分析的上下文管理器
    with ctx_manager as rf:
        # 如果 args 为 None，则设置为空元组
        args = args if args else ()
        # 如果 kwargs 为 None，则设置为空字典
        kwargs = kwargs if kwargs else {}

        # 检查 func 是否具有 _wrapped_async_rpc_function 属性，表示异步执行
        is_async_exec = hasattr(func, "_wrapped_async_rpc_function")

        # 如果是异步执行，则将 func 替换为其 _wrapped_async_rpc_function
        if is_async_exec:
            wrapped = func._wrapped_async_rpc_function
            if isinstance(wrapped, torch.jit.ScriptFunction):
                func = wrapped

        # 如果找到了限定名，则调用内置函数进行 RPC 调用
        if qualified_name is not None:
            fut = _invoke_rpc_builtin(
                dst_worker_info, qualified_name, rpc_timeout, *args, **kwargs
            )
        # 如果 func 是 TorchScript 函数，则调用 TorchScript 版本的 RPC 调用
        elif isinstance(func, torch.jit.ScriptFunction):
            fut = _invoke_rpc_torchscript(
                dst_worker_info.name,
                torch._jit_internal._qualified_name(func),
                args,
                kwargs,
                rpc_timeout,
                is_async_exec,
            )
        # 否则，使用默认的 Python 序列化器序列化 PythonUDF，并进行 RPC 调用
        else:
            (pickled_python_udf, tensors) = _default_pickler.serialize(
                PythonUDF(func, args, kwargs)
            )
            fut = _invoke_rpc_python_udf(
                dst_worker_info, pickled_python_udf, tensors, rpc_timeout, is_async_exec
            )
        
        # 如果应该进行性能分析，则确保自动微分的性能分析已启用
        if should_profile:
            assert torch.autograd._profiler_enabled()
            assert rf is not None
            # 安排性能分析回调在未来完成时运行
            # 返回的未来会在原始未来完成且性能分析回调完成后也完成，
            # 以确保 fut.wait() 完成性能分析，新未来将包含与原始未来相同的值
            fut = rf._call_end_callbacks_on_future(fut)

    # 返回 RPC 调用的未来对象
    return fut


# 装饰器函数，确保 RPC 初始化后才能调用的同步 RPC 函数
@_require_initialized
def rpc_sync(to, func, args=None, kwargs=None, timeout: float = UNSET_RPC_TIMEOUT):
    r"""
    Make a blocking RPC call to run function ``func`` on worker ``to``. RPC
    messages are sent and received in parallel to execution of Python code. This
    method is thread-safe.
    """
    # 记录 API 使用情况到 Torch C++ 库的日志中，此处是在使用 `torch.distributed.rpc_sync` API
    torch._C._log_api_usage_once("torch.distributed.rpc_sync")
    # 调用 _invoke_rpc 函数执行 RPC 同步调用，并获取 Future 对象
    fut = _invoke_rpc(to, func, RPCExecMode.SYNC, args, kwargs, timeout)
    # 等待 RPC 的 Future 对象完成并返回结果
    return fut.wait()
# 标记函数为要求已初始化状态的装饰器，确保在调用 RPC 异步方法前系统已经初始化
@_require_initialized
# 定义一个异步 RPC 调用函数，用于在目标节点上运行指定函数
def rpc_async(to, func, args=None, kwargs=None, timeout=UNSET_RPC_TIMEOUT):
    r"""
    Make a non-blocking RPC call to run function ``func`` on worker ``to``. RPC
    messages are sent and received in parallel to execution of Python code. This
    method is thread-safe. This method will immediately return a
    :class:`~torch.futures.Future` that can be awaited on.

    Args:
        to (str or WorkerInfo or int): name/rank/``WorkerInfo`` of the destination worker.
        func (Callable): a callable function, such as Python callables, builtin
                         operators (e.g. :meth:`~torch.add`) and annotated
                         TorchScript functions.
        args (tuple): the argument tuple for the ``func`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``func``
                       invocation.
        timeout (float, optional): timeout in seconds to use for this RPC. If
                                   the RPC does not complete in this amount of
                                   time, an exception indicating it has
                                   timed out will be raised. A value of 0
                                   indicates an infinite timeout, i.e. a timeout
                                   error will never be raised. If not provided,
                                   the default value set during initialization
                                   or with ``_set_rpc_timeout`` is used.


    Returns:
        Returns a :class:`~torch.futures.Future` object that can be waited
        on. When completed, the return value of ``func`` on ``args`` and
        ``kwargs`` can be retrieved from the :class:`~torch.futures.Future`
        object.

    .. warning ::
        Using GPU tensors as arguments or return values of ``func`` is not
        supported since we don't support sending GPU tensors over the wire. You
        need to explicitly copy GPU tensors to CPU before using them as
        arguments or return values of ``func``.

    .. warning ::
        The ``rpc_async`` API does not copy storages of argument tensors until
        sending them over the wire, which could be done by a different thread
        depending on the RPC backend type. The caller should make sure that the
        contents of those tensors stay intact until the returned
        :class:`~torch.futures.Future` completes.
    # 调用 Torch C++ 库中的 API 记录使用情况，标记为一次使用
    torch._C._log_api_usage_once("torch.distributed.rpc_async")
    # 调用 _invoke_rpc 函数发起远程过程调用（RPC），采用异步执行模式
    fut = _invoke_rpc(to, func, RPCExecMode.ASYNC, args, kwargs, timeout)
    # 如果 _thread_local_var 对象有 future_list 属性，将当前的 fut 追加到 future_list 中
    if hasattr(_thread_local_var, "future_list"):
        _thread_local_var.future_list.append(fut)
    # 返回异步调用的 Future 对象
    return fut
# 获取是否应该启用性能分析器的标志位
def _get_should_profile():
    # 获取当前活动的性能分析器类型
    ActiveProfilerType = torch._C._profiler.ActiveProfilerType
    return (
        torch.autograd._profiler_enabled()  # 检查自动求导性能分析是否启用
        and torch._C._autograd._profiler_type()  # 获取当前自动求导性能分析器类型
        == ActiveProfilerType.LEGACY  # 检查当前自动求导性能分析器是否为传统类型
    )


# 启用 RPC 性能分析器
def _enable_rpc_profiler(
    should_profile, qualified_name, func, rpc_type, dst_worker_info
):
    ctx_manager = contextlib.nullcontext()  # 创建一个空的上下文管理器

    if should_profile:
        # 根据函数的类型（内置函数、脚本函数、Python函数）创建相应的函数名表示
        if qualified_name is None:
            func_name = (
                torch._jit_internal._qualified_name(func)  # 获取脚本函数的限定名
                if isinstance(func, torch.jit.ScriptFunction)
                else func.__qualname__  # 获取普通 Python 函数的限定名
            )
        else:
            func_name = qualified_name  # 使用提供的限定名作为函数名

        # 构建 RPC 分析的关键字
        rpc_profiling_key = _build_rpc_profiling_key(
            rpc_type,  # RPC 类型
            func_name,  # 函数名
            get_worker_info().name,  # 获取当前工作节点的名称
            dst_worker_info.name,  # 获取目标工作节点的名称
        )
        RemoteProfilerManager.set_current_profiling_key(rpc_profiling_key)  # 设置当前远程分析器的分析键
        # 在上下文管理器中记录 RPC 性能分析
        ctx_manager = torch.autograd.profiler.record_function(rpc_profiling_key)  # type: ignore[assignment]

    return ctx_manager  # 返回上下文管理器
```